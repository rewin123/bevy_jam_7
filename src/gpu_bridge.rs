//! GPU Bridge — shares Bevy's wgpu device with Burn for zero-copy style transfer.
//!
//! Pipeline: render_texture → buffer → RGBA8→f32 compute →
//!           Burn inference → f32→RGBA8 compute → buffer → display_texture
//!
//! Single GPU context. No CPU copies. No threads. No channels.

#![allow(dead_code)]
#[cfg(feature = "burn-backend")]
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
#[cfg(feature = "burn-backend")]
use bevy::asset::AssetId;
#[cfg(feature = "burn-backend")]
use bevy::render::{
    render_asset::RenderAssets,
    renderer::{RenderAdapter, RenderAdapterInfo, RenderDevice, RenderInstance, RenderQueue},
    texture::GpuImage,
    Extract, Render, RenderApp, RenderSystems,
};

#[cfg(feature = "burn-backend")]
use burn::backend::wgpu::{init_device, RuntimeOptions, WgpuDevice, WgpuSetup};

use crate::inference_common::{RENDER_HEIGHT, RENDER_WIDTH};

// ── Resources ──────────────────────────────────────────────────────

/// Shared Burn device that reuses Bevy's wgpu context.
#[cfg(feature = "burn-backend")]
#[derive(Resource, Clone)]
pub struct SharedBurnDevice(pub WgpuDevice);

/// Raw wgpu handles from Bevy's renderer for direct GPU operations.
#[derive(Resource, Clone)]
pub struct BevyGpuHandles {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// GPU textures shared between render world and main world.
/// The render world populates these once the GpuImages are ready.
#[derive(Resource, Clone)]
pub struct SharedGpuTextures {
    pub inner: Arc<Mutex<GpuTextureHandles>>,
}

pub struct GpuTextureHandles {
    pub render_texture: Option<wgpu::Texture>,
    pub display_texture: Option<wgpu::Texture>,
}

/// Compute pipelines and reusable buffers for GPU format conversion.
#[derive(Resource)]
pub struct FormatConversionPipelines {
    /// RGBA8 (u32 packed) → f32 CHW [1,3,H,W]
    pub rgba8_to_f32_pipeline: wgpu::ComputePipeline,
    pub rgba8_to_f32_bind_group_layout: wgpu::BindGroupLayout,

    /// f32 CHW [1,3,H,W] → RGBA8 (u32 packed)
    pub f32_to_rgba8_pipeline: wgpu::ComputePipeline,
    pub f32_to_rgba8_bind_group_layout: wgpu::BindGroupLayout,

    // Reusable GPU buffers (sized for RENDER_WIDTH × RENDER_HEIGHT)
    /// Staging buffer for texture ↔ buffer copies (RGBA8, W*H*4 bytes)
    pub staging_buffer: wgpu::Buffer,
    /// Input tensor buffer (f32, 3*W*H*4 bytes) — fed to Burn
    pub input_tensor_buffer: wgpu::Buffer,
    /// Output RGBA8 buffer (u32 packed, W*H*4 bytes) — result after f32→RGBA8
    pub output_rgba_buffer: wgpu::Buffer,
    /// Uniform params buffer (width, height as u32)
    pub params_buffer: wgpu::Buffer,
    /// Pre-built bind groups (rebuilt if sizes change, but we use fixed resolution)
    pub rgba8_to_f32_bind_group: wgpu::BindGroup,
    pub f32_to_rgba8_bind_group: wgpu::BindGroup,
}

// ── Plugin ─────────────────────────────────────────────────────────

pub struct GpuBridgePlugin;

impl Plugin for GpuBridgePlugin {
    fn build(&self, _app: &mut App) {}

    #[cfg(feature = "burn-backend")]
    fn finish(&self, app: &mut App) {
        // ── Extract raw wgpu handles from Bevy's render sub-app ──
        let render_app = app.sub_app(RenderApp);

        let device: wgpu::Device = render_app
            .world()
            .resource::<RenderDevice>()
            .wgpu_device()
            .clone();

        let queue: wgpu::Queue = {
            let rq = render_app.world().resource::<RenderQueue>();
            // RenderQueue → Arc<WgpuWrapper<Queue>> → WgpuWrapper<Queue> → Queue
            let q_ref: &wgpu::Queue = rq;
            q_ref.clone()
        };

        let adapter: wgpu::Adapter = {
            let ra = render_app.world().resource::<RenderAdapter>();
            let a_ref: &wgpu::Adapter = ra;
            a_ref.clone()
        };

        let instance: wgpu::Instance = {
            let ri = render_app.world().resource::<RenderInstance>();
            let i_ref: &wgpu::Instance = ri;
            i_ref.clone()
        };

        let backend: wgpu::Backend = {
            let info = render_app.world().resource::<RenderAdapterInfo>();
            let info_ref: &wgpu::AdapterInfo = info;
            info_ref.backend
        };

        // ── Register Bevy's device with cubecl/Burn ──
        let setup = WgpuSetup {
            instance,
            adapter,
            device: device.clone(),
            queue: queue.clone(),
            backend,
        };

        let burn_device = init_device(setup, RuntimeOptions::default());
        info!(
            "GPU bridge: Burn using Bevy's wgpu device as {:?}",
            burn_device
        );

        app.insert_resource(SharedBurnDevice(burn_device));
        app.insert_resource(BevyGpuHandles {
            device: device.clone(),
            queue: queue.clone(),
        });

        // ── Shared GPU texture handles (populated by render world) ──
        let shared_textures = SharedGpuTextures {
            inner: Arc::new(Mutex::new(GpuTextureHandles {
                render_texture: None,
                display_texture: None,
            })),
        };

        // ── Create format conversion compute pipelines ──
        let pipelines = create_format_conversion_pipelines(&device, &queue);
        app.insert_resource(pipelines);

        // ── Register render world systems for texture extraction ──
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            // Insert shared textures into render world too (same Arc)
            render_app.insert_resource(shared_textures.clone());
            render_app.init_resource::<StyleImageIds>();

            render_app.add_systems(
                bevy::render::ExtractSchedule,
                extract_style_image_ids,
            );
            render_app.add_systems(
                Render,
                populate_gpu_textures.in_set(RenderSystems::PrepareAssets),
            );
        }

        // Insert into main app (after cloning for render app)
        app.insert_resource(shared_textures);
    }

    #[cfg(not(feature = "burn-backend"))]
    fn finish(&self, _app: &mut App) {}
}

// ── Compute pipeline creation ──────────────────────────────────────

fn create_format_conversion_pipelines(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> FormatConversionPipelines {
    let w = RENDER_WIDTH;
    let h = RENDER_HEIGHT;
    let num_pixels = (w * h) as u64;

    // ── Buffers ──
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("style_staging"),
        size: num_pixels * 4, // RGBA8: 4 bytes per pixel
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let input_tensor_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("style_input_tensor"),
        size: 3 * num_pixels * 4, // f32 CHW: 3 channels × 4 bytes
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let output_rgba_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("style_output_rgba"),
        size: num_pixels * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("style_params"),
        size: 32, // 8 × u32 (padded to 16-byte alignment)
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Write initial params (strides will be updated per-frame for f32→RGBA8)
    // [width, height, stride_c, stride_h, stride_w, 0, 0, 0]
    let total = w * h;
    queue.write_buffer(
        &params_buffer,
        0,
        bytemuck::bytes_of(&[w, h, total, w, 1u32, 0u32, 0u32, 0u32]),
    );

    // ── RGBA8 → f32 CHW pipeline ──
    let rgba8_to_f32_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rgba8_to_f32_chw"),
        source: wgpu::ShaderSource::Wgsl(RGBA8_TO_F32_WGSL.into()),
    });

    let rgba8_to_f32_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rgba8_to_f32_bgl"),
            entries: &[
                // binding 0: input RGBA8 (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: output f32 (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let rgba8_to_f32_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rgba8_to_f32_pl"),
            bind_group_layouts: &[&rgba8_to_f32_bind_group_layout],
            push_constant_ranges: &[],
        });

    let rgba8_to_f32_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rgba8_to_f32"),
            layout: Some(&rgba8_to_f32_pipeline_layout),
            module: &rgba8_to_f32_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    // ── f32 CHW → RGBA8 pipeline ──
    let f32_to_rgba8_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("f32_chw_to_rgba8"),
        source: wgpu::ShaderSource::Wgsl(F32_TO_RGBA8_WGSL.into()),
    });

    let f32_to_rgba8_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("f32_to_rgba8_bgl"),
            entries: &[
                // binding 0: input f32 CHW (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: output RGBA8 (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let f32_to_rgba8_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("f32_to_rgba8_pl"),
            bind_group_layouts: &[&f32_to_rgba8_bind_group_layout],
            push_constant_ranges: &[],
        });

    let f32_to_rgba8_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("f32_to_rgba8"),
            layout: Some(&f32_to_rgba8_pipeline_layout),
            module: &f32_to_rgba8_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    // ── Bind groups ──
    let rgba8_to_f32_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rgba8_to_f32_bg"),
        layout: &rgba8_to_f32_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: staging_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_tensor_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Note: f32_to_rgba8_bind_group will bind the BURN OUTPUT tensor buffer
    // at binding 0. We create a placeholder here; the real bind group is created
    // per-frame when we have the output tensor's buffer.
    // For now, create with input_tensor_buffer as placeholder.
    let f32_to_rgba8_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f32_to_rgba8_bg"),
        layout: &f32_to_rgba8_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_tensor_buffer.as_entire_binding(), // placeholder
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_rgba_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    FormatConversionPipelines {
        rgba8_to_f32_pipeline,
        rgba8_to_f32_bind_group_layout,
        f32_to_rgba8_pipeline,
        f32_to_rgba8_bind_group_layout,
        staging_buffer,
        input_tensor_buffer,
        output_rgba_buffer,
        params_buffer,
        rgba8_to_f32_bind_group,
        f32_to_rgba8_bind_group,
    }
}

// ── GPU copy helpers ───────────────────────────────────────────────

/// Copy a wgpu texture to the staging buffer (GPU→GPU).
/// The texture must have COPY_SRC usage and be RGBA8 format.
pub fn copy_texture_to_staging(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    staging: &wgpu::Buffer,
    width: u32,
    height: u32,
) {
    let bytes_per_row = width * 4;
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

/// Copy the output RGBA buffer back to a texture (GPU→GPU).
pub fn copy_staging_to_texture(
    encoder: &mut wgpu::CommandEncoder,
    staging: &wgpu::Buffer,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) {
    let bytes_per_row = width * 4;
    encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

/// Dispatch the RGBA8→f32 CHW compute shader.
pub fn dispatch_rgba8_to_f32(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &FormatConversionPipelines,
) {
    let num_pixels = (RENDER_WIDTH * RENDER_HEIGHT) as u32;
    let workgroup_count = (num_pixels + 255) / 256;

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("rgba8_to_f32"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline.rgba8_to_f32_pipeline);
    pass.set_bind_group(0, &pipeline.rgba8_to_f32_bind_group, &[]);
    pass.dispatch_workgroups(workgroup_count, 1, 1);
}

/// Dispatch the f32 CHW→RGBA8 compute shader with a custom bind group
/// (to bind the burn output tensor's buffer).
pub fn dispatch_f32_to_rgba8(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
) {
    let num_pixels = (RENDER_WIDTH * RENDER_HEIGHT) as u32;
    let workgroup_count = (num_pixels + 255) / 256;

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("f32_to_rgba8"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(workgroup_count, 1, 1);
}

/// Update the stride parameters in the params buffer for the f32→RGBA8 shader.
/// Strides come from the burn output tensor's CubeTensor.
pub fn update_output_strides(
    queue: &wgpu::Queue,
    pipelines: &FormatConversionPipelines,
    stride_c: u32,
    stride_h: u32,
    stride_w: u32,
) {
    // Write strides at offset 8 (after width and height)
    queue.write_buffer(
        &pipelines.params_buffer,
        8,
        bytemuck::bytes_of(&[stride_c, stride_h, stride_w]),
    );
}

// ── WGSL compute shaders (inlined) ────────────────────────────────

const RGBA8_TO_F32_WGSL: &str = r#"
struct Params {
    width: u32,
    height: u32,
    stride_c: u32,
    stride_h: u32,
    stride_w: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel = id.x;
    let total = params.width * params.height;
    if (pixel >= total) {
        return;
    }

    // unpack4x8unorm: u32 → vec4<f32> in [0,1]
    // byte order (little-endian): byte0=R, byte1=G, byte2=B, byte3=A
    let rgba = unpack4x8unorm(input[pixel]);

    // Write contiguous CHW layout: [1, 3, H, W]
    // Channel 0 (R) at offset 0
    // Channel 1 (G) at offset total
    // Channel 2 (B) at offset 2*total
    output[pixel] = rgba.r;
    output[total + pixel] = rgba.g;
    output[2u * total + pixel] = rgba.b;
}
"#;

const F32_TO_RGBA8_WGSL: &str = r#"
struct Params {
    width: u32,
    height: u32,
    stride_c: u32,
    stride_h: u32,
    stride_w: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel = id.x;
    let total = params.width * params.height;
    if (pixel >= total) {
        return;
    }

    // Compute 2D position from flat pixel index
    let y = pixel / params.width;
    let x = pixel % params.width;

    // Read using tensor strides (handles both NCHW and NHWC layouts)
    // Element [0, c, y, x] = offset c*stride_c + y*stride_h + x*stride_w
    let base = y * params.stride_h + x * params.stride_w;
    let r = clamp(input[base], 0.0, 1.0);
    let g = clamp(input[base + params.stride_c], 0.0, 1.0);
    let b = clamp(input[base + 2u * params.stride_c], 0.0, 1.0);

    // pack4x8unorm: vec4<f32> [0,1] → u32 packed RGBA8
    output[pixel] = pack4x8unorm(vec4(r, g, b, 1.0));
}
"#;

// ── Render world texture extraction ───────────────────────────────

/// Holds asset IDs for style images in the render world.
#[cfg(feature = "burn-backend")]
#[derive(Resource, Default)]
struct StyleImageIds {
    render_id: Option<AssetId<Image>>,
    display_id: Option<AssetId<Image>>,
}

/// Extract system: copies Handle<Image> IDs from main world's StyleTarget
/// into the render world's StyleImageIds resource (runs in ExtractSchedule).
#[cfg(feature = "burn-backend")]
fn extract_style_image_ids(
    main_target: Extract<Option<Res<crate::post_process::StyleTarget>>>,
    mut ids: ResMut<StyleImageIds>,
) {
    if let Some(target) = main_target.as_ref() {
        ids.render_id = Some(target.render_image.id());
        ids.display_id = Some(target.display_image.id());
    }
}

/// Render world system: looks up GpuImage from RenderAssets<GpuImage> and
/// stores the raw wgpu::Texture handles in SharedGpuTextures (shared with main world via Arc).
#[cfg(feature = "burn-backend")]
fn populate_gpu_textures(
    ids: Res<StyleImageIds>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    shared: Res<SharedGpuTextures>,
) {
    let mut textures = shared.inner.lock().unwrap();

    if let Some(render_id) = ids.render_id {
        if let Some(gpu_img) = gpu_images.get(render_id) {
            // GpuImage.texture is Bevy's Texture which Derefs to wgpu::Texture
            let wgpu_tex: &wgpu::Texture = &gpu_img.texture;
            textures.render_texture = Some(wgpu_tex.clone());
        }
    }

    if let Some(display_id) = ids.display_id {
        if let Some(gpu_img) = gpu_images.get(display_id) {
            let wgpu_tex: &wgpu::Texture = &gpu_img.texture;
            textures.display_texture = Some(wgpu_tex.clone());
        }
    }
}
