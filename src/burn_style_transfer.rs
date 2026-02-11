//! Burn-based inference backend for style transfer.
//!
//! Uses burn-import generated model code from ONNX files.
//! Uses wgpu (GPU) backend for accelerated inference.
//! Burn creates its own wgpu device (separate from Bevy's renderer)
//! because Bevy 0.18 uses wgpu 27 while Burn 0.20.1 uses wgpu 26.
//!
//! On native: runs inference in a separate thread (same as ort backend).
//! On WASM: runs inference synchronously in a Bevy system (no threads).

#![cfg(feature = "burn-backend")]

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::Ordering;

use bevy::prelude::*;
use burn::backend::Wgpu;
use burn::prelude::*;
use crossbeam_channel::{bounded, Receiver, Sender};

use crate::inference_common::*;
#[cfg(not(target_arch = "wasm32"))]
use crate::style_transfer::scan_model_directory;

type BurnBackend = Wgpu;

// Include all generated burn models.
// Each .onnx file in assets/models/styles/ generates a module during build.
// Models are discovered at build time by build.rs.
include!(concat!(env!("OUT_DIR"), "/burn_models.rs"));

/// Convert RGBA pixels to a burn tensor [1, 3, H, W] float32 [0, 1]
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

/// Convert a burn tensor [1, 3, H, W] float32 [0, 1] to RGBA pixels
fn burn_tensor_to_rgba(tensor: Tensor<BurnBackend, 4>) -> Vec<u8> {
    let shape = tensor.shape();
    let h = shape.dims[2];
    let w = shape.dims[3];

    let data: Vec<f32> = tensor.into_data().to_vec().unwrap();

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

// ===== Native (threaded) path =====

/// Setup the burn inference thread (Startup system) — native builds
#[cfg(not(target_arch = "wasm32"))]
pub fn setup_burn_inference_thread(
    mut commands: Commands,
    test_mode: Option<Res<TestInferenceMode>>,
) {
    let (send_frame, recv_frame) = bounded::<FrameData>(1);
    let (send_styled, recv_styled) = bounded::<StyledFrame>(1);
    let (send_switch, recv_switch) = bounded::<StyleSwitch>(4);

    let is_test = test_mode.is_some();
    let test_done = Arc::new(AtomicBool::new(false));
    if is_test {
        commands.insert_resource(TestInferenceDone(test_done.clone()));
    }

    let models = scan_model_directory();
    let names: Vec<String> = models.iter().map(|(name, _)| name.clone()).collect();

    if models.is_empty() {
        if is_test {
            error!("No style models found — cannot run test");
            test_done.store(true, Ordering::Release);
        } else {
            warn!("No style models found in assets/models/styles/");
        }
    }

    info!("Style models (burn backend, native): {:?}", names);

    let model_names = names.clone();
    std::thread::spawn(move || {
        burn_inference_thread_main(
            recv_frame,
            send_styled,
            recv_switch,
            &model_names,
            is_test,
            test_done,
        );
    });

    commands.insert_resource(StyleChannels {
        send_frame,
        recv_styled,
        send_switch,
    });
    commands.insert_resource(CurrentStyle { index: 0, names });
}

#[cfg(not(target_arch = "wasm32"))]
fn burn_inference_thread_main(
    recv: Receiver<FrameData>,
    send: Sender<StyledFrame>,
    recv_switch: Receiver<StyleSwitch>,
    model_names: &[String],
    test_mode: bool,
    test_done: Arc<AtomicBool>,
) {
    let device = burn::backend::wgpu::WgpuDevice::default();
    let num_models = model_names.len();
    let mut current_model = 0usize;
    let mut test_saved = false;
    let start_time = std::time::Instant::now();

    let models = load_burn_models(&device);

    info!(
        "Inference thread (burn): {} models loaded",
        models.len()
    );

    while let Ok(frame) = recv.recv() {
        while let Ok(switch) = recv_switch.try_recv() {
            if num_models > 0 {
                current_model = switch.index % num_models;
                info!("Inference thread (burn): switched to model {}", current_model);
            }
        }

        if models.is_empty() {
            let _ = send.try_send(StyledFrame {
                pixels: frame.pixels,
                width: frame.width,
                height: frame.height,
            });
            continue;
        }

        let w = RENDER_WIDTH as usize;
        let h = RENDER_HEIGHT as usize;
        let resized = resize_rgba(
            &frame.pixels,
            frame.width,
            frame.height,
            RENDER_WIDTH,
            RENDER_HEIGHT,
        );

        let test_ready = test_mode
            && !test_saved
            && start_time.elapsed() >= std::time::Duration::from_secs(1);
        if test_ready {
            if let Some(img) =
                image::RgbaImage::from_raw(RENDER_WIDTH, RENDER_HEIGHT, resized.clone())
            {
                img.save("test_input.png")
                    .unwrap_or_else(|e| error!("Failed to save test_input.png: {}", e));
                info!("Test: saved test_input.png");
            }
        }

        let input_tensor = rgba_to_burn_tensor(&resized, w, h, &device);
        let model_idx = current_model % models.len();
        let output_tensor = run_burn_inference(&models[model_idx], input_tensor);
        let output_rgba = burn_tensor_to_rgba(output_tensor);

        let final_pixels = resize_rgba(
            &output_rgba,
            RENDER_WIDTH,
            RENDER_HEIGHT,
            RENDER_WIDTH,
            RENDER_HEIGHT,
        );

        if test_ready {
            if let Some(img) = image::RgbaImage::from_raw(
                RENDER_WIDTH,
                RENDER_HEIGHT,
                final_pixels.clone(),
            ) {
                img.save("test_bevy_output.png")
                    .unwrap_or_else(|e| error!("Failed to save test_bevy_output.png: {}", e));
                info!("Test: saved test_bevy_output.png");
            }
            test_saved = true;
            test_done.store(true, Ordering::Release);
        }

        let _ = send.try_send(StyledFrame {
            pixels: final_pixels,
            width: RENDER_WIDTH,
            height: RENDER_HEIGHT,
        });
    }

    info!("Inference thread (burn): shutting down");
}

// ===== WASM (synchronous) path =====

/// Holds burn models and inference-side channel ends for WASM sync inference.
///
/// On WASM, there are no threads — this system directly receives frames,
/// runs inference, and sends results back through channels.
#[cfg(target_arch = "wasm32")]
pub struct BurnWasmState {
    models: Vec<BurnModelWrapper>,
    current: usize,
    recv_frame: Receiver<FrameData>,
    send_styled: Sender<StyledFrame>,
    recv_switch: Receiver<StyleSwitch>,
}

// SAFETY: wasm32 is single-threaded, so Send+Sync are trivially satisfied.
// These impls allow BurnWasmState to be used as a Bevy Resource.
#[cfg(target_arch = "wasm32")]
unsafe impl Send for BurnWasmState {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for BurnWasmState {}

#[cfg(target_arch = "wasm32")]
impl Resource for BurnWasmState {}

/// Setup burn inference for WASM — loads models synchronously, no threads
#[cfg(target_arch = "wasm32")]
pub fn setup_burn_inference_thread(
    mut commands: Commands,
    test_mode: Option<Res<TestInferenceMode>>,
) {
    let (send_frame, recv_frame) = bounded::<FrameData>(1);
    let (send_styled, recv_styled) = bounded::<StyledFrame>(1);
    let (send_switch, recv_switch) = bounded::<StyleSwitch>(4);

    if test_mode.is_some() {
        let test_done = Arc::new(AtomicBool::new(false));
        commands.insert_resource(TestInferenceDone(test_done));
    }

    // On WASM, filesystem is unavailable — use build-time generated model names
    let names = burn_model_names();
    info!("Style models (burn backend, wasm): {:?}", names);

    let device = burn::backend::wgpu::WgpuDevice::default();
    let burn_models = load_burn_models(&device);

    // Store the inference-side channel ends + models in BurnWasmState
    commands.insert_resource(BurnWasmState {
        models: burn_models,
        current: 0,
        recv_frame,
        send_styled,
        recv_switch,
    });

    // Store the main-thread-side channel ends (used by post_process, fever, etc.)
    commands.insert_resource(StyleChannels {
        send_frame,
        recv_styled,
        send_switch,
    });
    commands.insert_resource(CurrentStyle { index: 0, names });
}

/// WASM: synchronous inference system — processes one frame per Bevy Update tick
#[cfg(target_arch = "wasm32")]
pub fn burn_wasm_inference_system(mut state: ResMut<BurnWasmState>) {
    // Process style switch commands
    while let Ok(switch) = state.recv_switch.try_recv() {
        if !state.models.is_empty() {
            state.current = switch.index % state.models.len();
        }
    }

    // Try to receive a frame from the main thread
    let Ok(frame) = state.recv_frame.try_recv() else {
        return;
    };

    if state.models.is_empty() {
        return;
    }

    let device = burn::backend::wgpu::WgpuDevice::default();
    let w = RENDER_WIDTH as usize;
    let h = RENDER_HEIGHT as usize;
    let resized = resize_rgba(
        &frame.pixels,
        frame.width,
        frame.height,
        RENDER_WIDTH,
        RENDER_HEIGHT,
    );

    let input_tensor = rgba_to_burn_tensor(&resized, w, h, &device);
    let model_idx = state.current % state.models.len();
    let output_tensor = run_burn_inference(&state.models[model_idx], input_tensor);
    let output_rgba = burn_tensor_to_rgba(output_tensor);

    let _ = state.send_styled.try_send(StyledFrame {
        pixels: output_rgba,
        width: RENDER_WIDTH,
        height: RENDER_HEIGHT,
    });
}
