//! Burn-based inference backend for style transfer.
//!
//! On native (with GpuBridgePlugin): models loaded on Bevy's shared GPU device,
//! inference runs in a Bevy system with zero-copy GPU pipeline.
//!
//! On WASM: async device init via `spawn_local`, sync forward pass,
//!          async GPU readback via `into_data_async`.

#![cfg(feature = "burn-backend")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bevy::prelude::*;
use burn::backend::Wgpu;
use burn::prelude::*;

use crate::inference_common::*;

type BurnBackend = Wgpu;

// Include all generated burn models.
// Each .onnx file in assets/models/styles/ generates a module during build.
// Models are discovered at build time by build.rs.
include!(concat!(env!("OUT_DIR"), "/burn_models.rs"));

// ===== Native (system-based, shared GPU device) path =====

#[cfg(not(target_arch = "wasm32"))]
use crate::gpu_bridge::{
    BevyGpuHandles, FormatConversionPipelines, SharedBurnDevice, SharedGpuTextures,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::post_process::{StyleBypass, StyleTarget};
#[cfg(not(target_arch = "wasm32"))]
use crate::style_transfer::scan_model_directory;

/// Burn models loaded on the shared GPU device. Main-world resource.
#[cfg(not(target_arch = "wasm32"))]
pub struct BurnModels {
    models: Vec<BurnModelWrapper>,
    device: burn::backend::wgpu::WgpuDevice,
}

// SAFETY: BurnModels is only accessed from the main inference system (single-threaded).
// The internal OnceCell<Tensor> in Burn's Param<T> is fully initialized after model loading.
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for BurnModels {}
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Sync for BurnModels {}
#[cfg(not(target_arch = "wasm32"))]
impl Resource for BurnModels {}

/// Startup system: load burn models on the shared GPU device.
#[cfg(not(target_arch = "wasm32"))]
pub fn setup_burn_models(
    mut commands: Commands,
    shared_device: Option<Res<SharedBurnDevice>>,
    test_mode: Option<Res<TestInferenceMode>>,
) {
    let is_test = test_mode.is_some();
    if is_test {
        let test_done = Arc::new(AtomicBool::new(false));
        commands.insert_resource(TestInferenceDone(test_done.clone()));
        // Test mode still needs early exit if no models
        let models = scan_model_directory();
        if models.is_empty() {
            error!("No style models found — cannot run test");
            test_done.store(true, Ordering::Release);
        }
    }

    let device = if let Some(shared) = shared_device {
        info!("Burn: using shared GPU device {:?}", shared.0);
        shared.0.clone()
    } else {
        warn!("Burn: SharedBurnDevice not available, falling back to default device");
        burn::backend::wgpu::WgpuDevice::default()
    };

    let models = load_burn_models(&device);
    let names: Vec<String> = burn_model_names();

    info!(
        "Burn models loaded on shared device: {} models ({:?})",
        models.len(),
        names
    );

    commands.insert_resource(BurnModels {
        models,
        device: device.clone(),
    });
    commands.insert_resource(CurrentStyle {
        index: 0,
        names,
    });
}

/// Throttle inference to avoid overwhelming the GPU
#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource)]
pub struct InferenceTimer(pub Timer);

/// Track whether an inference is in-flight (to skip frames)
#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource)]
pub struct InferenceInFlight(pub bool);

/// When set to true, the next inference frame will dump intermediate data for debugging.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource)]
pub struct DebugDumpFrame(pub bool);

/// System to trigger debug dump with 'P' key
#[cfg(not(target_arch = "wasm32"))]
pub fn debug_dump_trigger(keys: Res<ButtonInput<KeyCode>>, mut dump: ResMut<DebugDumpFrame>) {
    if keys.just_pressed(KeyCode::KeyP) {
        dump.0 = true;
        info!("Debug dump: will capture next inference frame");
    }
}

/// GPU style transfer system — runs in Update schedule.
///
/// Pipeline:
/// 1. copy render texture → staging buffer (GPU→GPU)
/// 2. RGBA8 → f32 CHW compute shader (GPU)
/// 3. Copy our buffer → burn input tensor buffer (GPU→GPU)
/// 4. Burn model.forward() (GPU compute, same device!)
/// 5. f32 CHW → RGBA8 compute shader on burn output (GPU)
/// 6. Copy output buffer → display texture (GPU→GPU)
///
/// All GPU operations, no CPU copies.
#[cfg(not(target_arch = "wasm32"))]
pub fn gpu_style_transfer_system(
    time: Res<Time>,
    mut timer: ResMut<InferenceTimer>,
    mut in_flight: ResMut<InferenceInFlight>,
    gpu: Res<BevyGpuHandles>,
    pipelines: Res<FormatConversionPipelines>,
    shared_textures: Res<SharedGpuTextures>,
    burn_models: Res<BurnModels>,
    style: Res<CurrentStyle>,
    bypass: Res<StyleBypass>,
    mut dump: ResMut<DebugDumpFrame>,
) {
    // Throttle
    timer.0.tick(time.delta());
    if !timer.0.just_finished() {
        return;
    }

    if bypass.0 || burn_models.models.is_empty() {
        return;
    }

    if in_flight.0 {
        return;
    }

    // Get GPU textures (populated by render world)
    let textures = shared_textures.inner.lock().unwrap();
    let Some(render_tex) = textures.render_texture.as_ref() else {
        return;
    };
    let Some(display_tex) = textures.display_texture.as_ref() else {
        return;
    };
    // Clone the Arc-based wgpu::Texture handles so we can drop the lock
    let render_tex = render_tex.clone();
    let display_tex = display_tex.clone();
    drop(textures);

    in_flight.0 = true;

    let w = RENDER_WIDTH;
    let h = RENDER_HEIGHT;

    // ── Step 1: Copy render texture → staging buffer ──
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("style_transfer"),
        });

    crate::gpu_bridge::copy_texture_to_staging(
        &mut encoder,
        &render_tex,
        &pipelines.staging_buffer,
        w,
        h,
    );

    // ── Step 2: RGBA8 → f32 CHW compute shader ──
    crate::gpu_bridge::dispatch_rgba8_to_f32(&mut encoder, &pipelines);

    // Submit steps 1+2
    gpu.queue.submit(std::iter::once(encoder.finish()));

    // ── Step 3: Copy our f32 buffer → burn input tensor ──
    // Create an empty burn tensor of the right shape, then copy our data into its buffer.
    let input_tensor: Tensor<BurnBackend, 4> = Tensor::empty(
        [1, 3, h as usize, w as usize],
        &burn_models.device,
    );

    // Get the raw wgpu buffer backing the burn tensor
    let input_primitive = input_tensor.clone().into_primitive().tensor();
    // Flush any pending cubecl commands (e.g. from tensor allocation) before
    // we write directly into burn's buffer via wgpu commands.
    input_primitive.client.flush();

    let input_binding = input_primitive.handle.clone().binding();
    let input_resource = input_primitive.client.get_resource(input_binding);
    let input_wgpu = input_resource.resource();

    // Copy our format-converted f32 buffer into burn's tensor buffer
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("style_copy_to_burn"),
        });
    encoder.copy_buffer_to_buffer(
        &pipelines.input_tensor_buffer,
        0,
        &input_wgpu.buffer,
        input_wgpu.offset,
        input_wgpu.size.min(3 * (w as u64) * (h as u64) * 4),
    );
    gpu.queue.submit(std::iter::once(encoder.finish()));

    // ── Step 4: Burn inference ──
    let model_idx = style.index % burn_models.models.len();
    let output_tensor = run_burn_inference(&burn_models.models[model_idx], input_tensor.clone());

    // ── Debug dump (if requested via 'P' key) ──
    if dump.0 {
        dump.0 = false;
        info!("Debug dump: reading back tensors from GPU...");
        // Clone tensors and do sync GPU readback (slow, only for debugging)
        let input_data = input_tensor.into_data();
        let output_data = output_tensor.clone().into_data();
        save_debug_tensors(&input_data, &output_data, w as usize, h as usize);
        info!("Debug dump: saved to /tmp/debug_input.png and /tmp/debug_output.png");
    }

    // ── Step 5: Get output tensor's raw buffer, run f32→RGBA8 compute ──
    let output_primitive = output_tensor.into_primitive().tensor();

    // Log tensor layout once for diagnostics
    {
        use std::sync::atomic::{AtomicBool, Ordering as AO};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if !LOGGED.swap(true, AO::Relaxed) {
            info!(
                "Output tensor layout: shape={:?}, strides={:?}, is_contiguous={}, dtype={:?}",
                output_primitive.shape.dims,
                output_primitive.strides,
                output_primitive.is_contiguous(),
                output_primitive.dtype,
            );
        }
    }

    // CRITICAL: Flush cubecl's batched GPU commands (inference dispatches) so they
    // are submitted to the wgpu queue BEFORE our output processing commands.
    output_primitive.client.flush();

    // Update stride params so the f32→RGBA8 shader can handle any tensor layout
    // (e.g., NHWC strides from burn's conv2d output)
    let stride_c = output_primitive.strides[1] as u32;
    let stride_h = output_primitive.strides[2] as u32;
    let stride_w = output_primitive.strides[3] as u32;
    crate::gpu_bridge::update_output_strides(&gpu.queue, &pipelines, stride_c, stride_h, stride_w);

    let output_binding = output_primitive.handle.clone().binding();
    let output_resource = output_primitive.client.get_resource(output_binding);
    let output_wgpu = output_resource.resource();

    // Create a bind group that binds the burn output buffer as input
    let f32_to_rgba8_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f32_to_rgba8_output_bg"),
        layout: &pipelines.f32_to_rgba8_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &output_wgpu.buffer,
                    offset: output_wgpu.offset,
                    size: std::num::NonZeroU64::new(output_wgpu.size),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pipelines.output_rgba_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pipelines.params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("style_output"),
        });

    crate::gpu_bridge::dispatch_f32_to_rgba8(
        &mut encoder,
        &pipelines.f32_to_rgba8_pipeline,
        &f32_to_rgba8_bind_group,
    );

    // ── Step 6: Copy output RGBA buffer → display texture ──
    crate::gpu_bridge::copy_staging_to_texture(
        &mut encoder,
        &pipelines.output_rgba_buffer,
        &display_tex,
        w,
        h,
    );

    gpu.queue.submit(std::iter::once(encoder.finish()));

    in_flight.0 = false;
}

/// System to handle bypass mode — copy render texture directly to display texture
#[cfg(not(target_arch = "wasm32"))]
pub fn gpu_bypass_copy_system(
    gpu: Res<BevyGpuHandles>,
    shared_textures: Res<SharedGpuTextures>,
    bypass: Res<StyleBypass>,
    time: Res<Time>,
    mut timer: ResMut<InferenceTimer>,
) {
    if !bypass.0 {
        return;
    }

    timer.0.tick(time.delta());
    if !timer.0.just_finished() {
        return;
    }

    let textures = shared_textures.inner.lock().unwrap();
    let Some(render_tex) = textures.render_texture.as_ref() else {
        return;
    };
    let Some(display_tex) = textures.display_texture.as_ref() else {
        return;
    };
    let render_tex = render_tex.clone();
    let display_tex = display_tex.clone();
    drop(textures);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bypass_copy"),
        });

    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &render_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: &display_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: RENDER_WIDTH,
            height: RENDER_HEIGHT,
            depth_or_array_layers: 1,
        },
    );

    gpu.queue.submit(std::iter::once(encoder.finish()));
}

/// Debug helper: save input and output tensors as PNG images.
/// Tensors are in CHW [1,3,H,W] format with f32 values in [0,1].
#[cfg(not(target_arch = "wasm32"))]
fn save_debug_tensors(
    input_data: &burn::tensor::TensorData,
    output_data: &burn::tensor::TensorData,
    w: usize,
    h: usize,
) {
    fn chw_to_rgba_png(data: &burn::tensor::TensorData, w: usize, h: usize, path: &str) {
        let values: Vec<f32> = data.to_vec().unwrap();
        let expected = 3 * h * w;
        if values.len() < expected {
            warn!(
                "Debug dump: tensor has {} values, expected {} (3*{}*{})",
                values.len(),
                expected,
                h,
                w
            );
            return;
        }
        let mut rgba = vec![255u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                rgba[idx] = (values[y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
                rgba[idx + 1] = (values[h * w + y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
                rgba[idx + 2] = (values[2 * h * w + y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
        if let Some(img) = image::RgbaImage::from_raw(w as u32, h as u32, rgba) {
            match img.save(path) {
                Ok(_) => info!("Debug dump: saved {}", path),
                Err(e) => warn!("Debug dump: failed to save {}: {}", path, e),
            }
        }
    }

    chw_to_rgba_png(input_data, w, h, "/tmp/debug_input.png");
    chw_to_rgba_png(output_data, w, h, "/tmp/debug_output.png");

    // Also save raw f32 data for Python analysis
    let input_values: Vec<f32> = input_data.to_vec().unwrap();
    let output_values: Vec<f32> = output_data.to_vec().unwrap();
    std::fs::write(
        "/tmp/debug_input.bin",
        bytemuck::cast_slice::<f32, u8>(&input_values),
    )
    .ok();
    std::fs::write(
        "/tmp/debug_output.bin",
        bytemuck::cast_slice::<f32, u8>(&output_values),
    )
    .ok();
    info!(
        "Debug dump: raw data saved ({} input floats, {} output floats)",
        input_values.len(),
        output_values.len()
    );
}

// ===== WASM (async wgpu) path =====
// Unchanged — WASM still uses channels + async readback since it can't share device
// with Bevy (different async init requirements).

#[cfg(target_arch = "wasm32")]
use crossbeam_channel::{bounded, Receiver, Sender};
#[cfg(target_arch = "wasm32")]
use std::sync::Mutex;

/// Convert RGBA pixels to a burn tensor [1, 3, H, W] float32 [0, 1]
#[cfg(target_arch = "wasm32")]
fn rgba_to_burn_tensor(
    pixels: &[u8],
    w: usize,
    h: usize,
    device: &<BurnBackend as Backend>::Device,
) -> Tensor<BurnBackend, 4> {
    let mut data = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            data[y * w + x] = pixels[idx] as f32 / 255.0;
            data[h * w + y * w + x] = pixels[idx + 1] as f32 / 255.0;
            data[2 * h * w + y * w + x] = pixels[idx + 2] as f32 / 255.0;
        }
    }
    Tensor::from_floats(
        burn::tensor::TensorData::new(data, [1, 3, h, w]),
        device,
    )
}

/// Convert CHW float32 data [3, H, W] in [0,1] to RGBA pixel buffer
#[cfg(target_arch = "wasm32")]
fn chw_f32_to_rgba(data: &[f32], w: usize, h: usize) -> Vec<u8> {
    let mut rgba = vec![255u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            rgba[idx] = (data[y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
            rgba[idx + 1] = (data[h * w + y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
            rgba[idx + 2] = (data[2 * h * w + y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    rgba
}

/// Initialization phase for the WASM burn backend
#[cfg(target_arch = "wasm32")]
enum WasmInitPhase {
    /// Async wgpu device init + model loading in progress
    Initializing,
    /// Models loaded, ready for inference
    Ready(Vec<BurnModelWrapper>),
}

/// Holds burn models and async state for WASM inference.
#[cfg(target_arch = "wasm32")]
pub struct BurnWasmState {
    phase: Arc<Mutex<WasmInitPhase>>,
    device: Arc<Mutex<Option<burn::backend::wgpu::WgpuDevice>>>,
    current: usize,
    recv_frame: Receiver<FrameData>,
    send_styled: Sender<StyledFrame>,
    recv_switch: Receiver<StyleSwitch>,
    inference_running: Arc<AtomicBool>,
}

// SAFETY: wasm32 is single-threaded, so Send+Sync are trivially satisfied.
#[cfg(target_arch = "wasm32")]
unsafe impl Send for BurnWasmState {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for BurnWasmState {}

#[cfg(target_arch = "wasm32")]
impl Resource for BurnWasmState {}

/// Setup burn inference for WASM — uses shared GPU device if available (WebGPU),
/// otherwise falls back to async device init.
#[cfg(target_arch = "wasm32")]
pub fn setup_burn_inference_wasm(
    mut commands: Commands,
    shared_device: Option<Res<crate::gpu_bridge::SharedBurnDevice>>,
    test_mode: Option<Res<TestInferenceMode>>,
) {
    let (send_frame, recv_frame) = bounded::<FrameData>(1);
    let (send_styled, recv_styled) = bounded::<StyledFrame>(1);
    let (send_switch, recv_switch) = bounded::<StyleSwitch>(4);

    if test_mode.is_some() {
        let test_done = Arc::new(AtomicBool::new(false));
        commands.insert_resource(TestInferenceDone(test_done));
    }

    let names = burn_model_names();
    info!("Style models (burn backend, wasm/wgpu): {:?}", names);

    let phase = Arc::new(Mutex::new(WasmInitPhase::Initializing));
    let phase_clone = phase.clone();
    let device_holder = Arc::new(Mutex::new(None::<burn::backend::wgpu::WgpuDevice>));
    let device_holder_clone = device_holder.clone();

    if let Some(shared) = shared_device {
        // WebGPU: use shared device from GpuBridgePlugin (same GPU context as Bevy)
        let device = shared.0.clone();
        info!("WASM: using shared GPU device {:?}", device);
        let models = load_burn_models(&device);
        info!("WASM: {} burn models loaded on shared device", models.len());
        *device_holder.lock().unwrap() = Some(device);
        *phase_clone.lock().unwrap() = WasmInitPhase::Ready(models);
    } else {
        // WebGL fallback: async device init (separate GPU context)
        wasm_bindgen_futures::spawn_local(async move {
            use burn::backend::wgpu::graphics::AutoGraphicsApi;
            use burn::backend::wgpu::{init_setup_async, RuntimeOptions};

            let device = burn::backend::wgpu::WgpuDevice::default();
            info!("WASM: initializing wgpu device async...");

            let _setup =
                init_setup_async::<AutoGraphicsApi>(&device, RuntimeOptions::default()).await;
            info!("WASM: wgpu device initialized");

            let models = load_burn_models(&device);
            info!("WASM: {} burn models loaded", models.len());

            *device_holder_clone.lock().unwrap() = Some(device);
            *phase_clone.lock().unwrap() = WasmInitPhase::Ready(models);
        });
    }

    commands.insert_resource(BurnWasmState {
        phase,
        device: device_holder,
        current: 0,
        recv_frame,
        send_styled,
        recv_switch,
        inference_running: Arc::new(AtomicBool::new(false)),
    });

    commands.insert_resource(StyleChannels {
        send_frame,
        recv_styled,
        send_switch,
    });
    commands.insert_resource(CurrentStyle {
        index: 0,
        names,
    });
}

/// WASM inference system: sync forward pass + async GPU readback
#[cfg(target_arch = "wasm32")]
pub fn burn_wasm_inference_system(mut state: ResMut<BurnWasmState>) {
    while let Ok(switch) = state.recv_switch.try_recv() {
        let model_count = state
            .phase
            .lock()
            .ok()
            .and_then(|p| match &*p {
                WasmInitPhase::Ready(models) => Some(models.len()),
                _ => None,
            })
            .unwrap_or(0);
        if model_count > 0 {
            state.current = switch.index % model_count;
        }
    }

    if state.inference_running.load(Ordering::Relaxed) {
        return;
    }

    let Ok(frame) = state.recv_frame.try_recv() else {
        return;
    };

    let output_tensor = {
        let phase = state.phase.lock().unwrap();
        let models = match &*phase {
            WasmInitPhase::Ready(models) if !models.is_empty() => models,
            _ => return,
        };

        let device_guard = state.device.lock().unwrap();
        let Some(ref device) = *device_guard else {
            return;
        };

        let w = RENDER_WIDTH as usize;
        let h = RENDER_HEIGHT as usize;
        let resized = resize_rgba(
            &frame.pixels,
            frame.width,
            frame.height,
            RENDER_WIDTH,
            RENDER_HEIGHT,
        );

        let input = rgba_to_burn_tensor(&resized, w, h, device);
        let model_idx = state.current % models.len();
        run_burn_inference(&models[model_idx], input)
    };

    let send = state.send_styled.clone();
    let running = state.inference_running.clone();
    running.store(true, Ordering::Relaxed);

    wasm_bindgen_futures::spawn_local(async move {
        match output_tensor.into_data_async().await {
            Ok(data) => {
                let values: Vec<f32> = data.to_vec().unwrap();
                let rgba = chw_f32_to_rgba(
                    &values,
                    RENDER_WIDTH as usize,
                    RENDER_HEIGHT as usize,
                );
                let _ = send.try_send(StyledFrame {
                    pixels: rgba,
                    width: RENDER_WIDTH,
                    height: RENDER_HEIGHT,
                });
            }
            Err(e) => {
                warn!("WASM burn readback error: {e}");
            }
        }
        running.store(false, Ordering::Relaxed);
    });
}
