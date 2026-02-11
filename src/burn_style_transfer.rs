//! Burn-based inference backend for style transfer.
//!
//! Uses burn-import generated model code from ONNX files.
//! Uses wgpu (GPU) backend for accelerated inference.
//! Burn creates its own wgpu device (separate from Bevy's renderer).
//!
//! On native: runs inference in a separate thread with sync GPU readback.
//! On WASM: async device init via `spawn_local`, sync forward pass,
//!          async GPU readback via `into_data_async`.

#![cfg(feature = "burn-backend")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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

/// Convert CHW float32 data [3, H, W] in [0,1] to RGBA pixel buffer
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

/// Convert a burn tensor [1, 3, H, W] float32 [0, 1] to RGBA pixels (sync readback)
#[cfg(not(target_arch = "wasm32"))]
fn burn_tensor_to_rgba(tensor: Tensor<BurnBackend, 4>) -> Vec<u8> {
    let shape = tensor.shape();
    let h = shape.dims[2];
    let w = shape.dims[3];
    let data: Vec<f32> = tensor.into_data().to_vec().unwrap();
    chw_f32_to_rgba(&data, w, h)
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

// ===== WASM (async wgpu) path =====
//
// On WASM, wgpu device creation and GPU buffer readback are inherently async
// (JavaScript Promises). cubecl's `read_sync` uses `poll_once` which panics
// if the future isn't immediately ready.
//
// Solution:
// 1. Async device init via `spawn_local` + `init_setup_async`
// 2. Model loading after device init (tensor writes to GPU are sync)
// 3. Forward pass is sync (all GPU compute, no CPU readback)
// 4. Output readback via `into_data_async` in a `spawn_local` task

#[cfg(target_arch = "wasm32")]
use std::sync::Mutex;

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
    current: usize,
    recv_frame: Receiver<FrameData>,
    send_styled: Sender<StyledFrame>,
    recv_switch: Receiver<StyleSwitch>,
    /// Prevents overlapping async readback tasks
    inference_running: Arc<AtomicBool>,
}

// SAFETY: wasm32 is single-threaded, so Send+Sync are trivially satisfied.
#[cfg(target_arch = "wasm32")]
unsafe impl Send for BurnWasmState {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for BurnWasmState {}

#[cfg(target_arch = "wasm32")]
impl Resource for BurnWasmState {}

/// Setup burn inference for WASM — async device init + model loading
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

    let names = burn_model_names();
    info!("Style models (burn backend, wasm/wgpu): {:?}", names);

    let phase = Arc::new(Mutex::new(WasmInitPhase::Initializing));
    let phase_clone = phase.clone();

    // Async initialization: wgpu device creation is a JS Promise,
    // so we must use spawn_local to await it.
    wasm_bindgen_futures::spawn_local(async move {
        use burn::backend::wgpu::{RuntimeOptions, init_setup_async};
        use burn::backend::wgpu::graphics::AutoGraphicsApi;

        let device = burn::backend::wgpu::WgpuDevice::default();
        info!("WASM: initializing wgpu device async...");

        // Pre-register the cubecl compute client so subsequent
        // ComputeClient::load() calls find it without calling DeviceState::init
        let _setup = init_setup_async::<AutoGraphicsApi>(
            &device,
            RuntimeOptions::default(),
        )
        .await;
        info!("WASM: wgpu device initialized");

        // Model loading: from_data writes to GPU (sync after device init)
        let models = load_burn_models(&device);
        info!("WASM: {} burn models loaded", models.len());

        *phase_clone.lock().unwrap() = WasmInitPhase::Ready(models);
    });

    commands.insert_resource(BurnWasmState {
        phase,
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
    commands.insert_resource(CurrentStyle { index: 0, names });
}

/// WASM inference system: sync forward pass + async GPU readback
#[cfg(target_arch = "wasm32")]
pub fn burn_wasm_inference_system(mut state: ResMut<BurnWasmState>) {
    // Process style switch commands
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

    // Don't start new inference if async readback is still running
    if state.inference_running.load(Ordering::Relaxed) {
        return;
    }

    // Try to receive a frame
    let Ok(frame) = state.recv_frame.try_recv() else {
        return;
    };

    // Forward pass (sync): lock models briefly, run GPU compute, release lock
    let output_tensor = {
        let phase = state.phase.lock().unwrap();
        let models = match &*phase {
            WasmInitPhase::Ready(models) if !models.is_empty() => models,
            _ => return, // Still initializing or no models
        };

        let w = RENDER_WIDTH as usize;
        let h = RENDER_HEIGHT as usize;
        let device = burn::backend::wgpu::WgpuDevice::default();
        let resized = resize_rgba(
            &frame.pixels,
            frame.width,
            frame.height,
            RENDER_WIDTH,
            RENDER_HEIGHT,
        );

        let input = rgba_to_burn_tensor(&resized, w, h, &device);
        let model_idx = state.current % models.len();
        run_burn_inference(&models[model_idx], input)
        // phase guard dropped here
    };

    // Async readback: GPU→CPU buffer mapping is a JS Promise on WASM
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
