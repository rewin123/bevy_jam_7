use bevy::prelude::*;

pub use crate::inference_common::*;

pub struct StyleTransferPlugin;

impl Plugin for StyleTransferPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(feature = "burn-backend")]
        {
            // Native: system-based inference on shared GPU device
            #[cfg(not(target_arch = "wasm32"))]
            {
                app.add_systems(
                    Startup,
                    crate::burn_style_transfer::setup_burn_models,
                );
                app.insert_resource(
                    crate::burn_style_transfer::InferenceTimer(Timer::from_seconds(
                        0.05,
                        TimerMode::Repeating,
                    )),
                );
                app.insert_resource(crate::burn_style_transfer::InferenceInFlight(false));
                app.insert_resource(crate::burn_style_transfer::DebugDumpFrame(false));
                app.add_systems(
                    Update,
                    (
                        crate::burn_style_transfer::debug_dump_trigger,
                        crate::burn_style_transfer::gpu_style_transfer_system,
                        crate::burn_style_transfer::gpu_bypass_copy_system,
                    ),
                );
            }

            // WASM: async device init + channel-based inference
            #[cfg(target_arch = "wasm32")]
            {
                app.add_systems(
                    Startup,
                    crate::burn_style_transfer::setup_burn_inference_wasm,
                );
                app.add_systems(
                    Update,
                    crate::burn_style_transfer::burn_wasm_inference_system,
                );
            }
        }

        #[cfg(feature = "ort-backend")]
        app.add_systems(Startup, setup_ort_inference_thread);
    }
}

// ===== ort backend (unchanged) =====

#[cfg(feature = "ort-backend")]
use crossbeam_channel::{bounded, Receiver, Sender};
#[cfg(feature = "ort-backend")]
use ndarray::Array4;
#[cfg(feature = "ort-backend")]
use ort::session::Session;
#[cfg(feature = "ort-backend")]
use ort::value::Tensor;
#[cfg(feature = "ort-backend")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "ort-backend")]
use std::sync::Arc;

/// Scan `assets/models/styles/` for .onnx files, return sorted (name, path) pairs.
pub fn scan_model_directory() -> Vec<(String, String)> {
    let model_dir = "assets/models/styles";
    let mut models = Vec::new();

    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("onnx")) {
                let name = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                models.push((name, path.to_string_lossy().to_string()));
            }
        }
    }

    models.sort_by(|a, b| a.0.cmp(&b.0));
    models
}

#[cfg(feature = "ort-backend")]
fn rgba_to_tensor(pixels: &[u8], w: usize, h: usize) -> Array4<f32> {
    let mut input = Array4::<f32>::zeros((1, 3, h, w));
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            input[[0, 0, y, x]] = pixels[idx] as f32 / 255.0;
            input[[0, 1, y, x]] = pixels[idx + 1] as f32 / 255.0;
            input[[0, 2, y, x]] = pixels[idx + 2] as f32 / 255.0;
        }
    }
    input
}

#[cfg(feature = "ort-backend")]
fn tensor_to_rgba(tensor: &ndarray::ArrayViewD<'_, f32>, w: usize, h: usize) -> Vec<u8> {
    let mut rgba = vec![255u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            rgba[idx] = (tensor[[0, 0, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
            rgba[idx + 1] = (tensor[[0, 1, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
            rgba[idx + 2] = (tensor[[0, 2, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    rgba
}

#[cfg(feature = "ort-backend")]
fn build_session(path: &str) -> Session {
    Session::builder()
        .unwrap()
        .with_execution_providers([
            ort::execution_providers::CPUExecutionProvider::default().build(),
        ])
        .unwrap()
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .unwrap()
        .commit_from_file(path)
        .unwrap_or_else(|e| panic!("Failed to load model {}: {}", path, e))
}

#[cfg(feature = "ort-backend")]
fn setup_ort_inference_thread(mut commands: Commands, test_mode: Option<Res<TestInferenceMode>>) {
    let (send_frame, recv_frame) = bounded::<FrameData>(1);
    let (send_styled, recv_styled) = bounded::<StyledFrame>(1);
    let (send_switch, recv_switch) = bounded::<StyleSwitch>(4);

    let is_test = test_mode.is_some();
    let test_done = Arc::new(AtomicBool::new(false));
    if is_test {
        commands.insert_resource(TestInferenceDone(test_done.clone()));
    }

    let models = scan_model_directory();
    if models.is_empty() {
        if is_test {
            error!("No .onnx style models found — cannot run test");
            test_done.store(true, Ordering::Release);
        } else {
            warn!("No .onnx style models found in assets/models/styles/");
        }
    }

    let names: Vec<String> = models.iter().map(|(name, _)| name.clone()).collect();
    let paths: Vec<String> = models.iter().map(|(_, path)| path.clone()).collect();

    info!("Style models (ort backend): {:?}", names);

    std::thread::spawn(move || {
        ort_inference_thread_main(
            recv_frame,
            send_styled,
            recv_switch,
            &paths,
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

#[cfg(feature = "ort-backend")]
fn ort_inference_thread_main(
    recv: Receiver<FrameData>,
    send: Sender<StyledFrame>,
    recv_switch: Receiver<StyleSwitch>,
    model_paths: &[String],
    test_mode: bool,
    test_done: Arc<AtomicBool>,
) {
    let num_models = model_paths.len();
    let mut sessions: Vec<Option<Session>> = (0..num_models).map(|_| None).collect();
    let mut current_model = 0usize;
    let mut test_saved = false;
    let start_time = std::time::Instant::now();

    info!(
        "Inference thread (ort): {} models available, loading lazily",
        num_models
    );

    while let Ok(frame) = recv.recv() {
        while let Ok(switch) = recv_switch.try_recv() {
            if num_models > 0 {
                current_model = switch.index % num_models;
                info!("Inference thread: switched to model {}", current_model);
            }
        }

        if num_models == 0 {
            let _ = send.try_send(StyledFrame {
                pixels: frame.pixels,
                width: frame.width,
                height: frame.height,
            });
            continue;
        }

        if sessions[current_model].is_none() {
            info!("Loading model '{}'...", model_paths[current_model]);
            sessions[current_model] = Some(build_session(&model_paths[current_model]));
            info!("Model loaded");
        }
        let session = sessions[current_model].as_mut().unwrap();

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
                info!("Test: saved test_input.png ({}x{})", RENDER_WIDTH, RENDER_HEIGHT);
            }
        }

        let input_nd = rgba_to_tensor(&resized, w, h);
        let input_tensor = match Tensor::from_array(input_nd) {
            Ok(t) => t,
            Err(e) => {
                error!("Failed to create input tensor: {}", e);
                continue;
            }
        };

        let outputs = match session.run(ort::inputs!["input" => input_tensor]) {
            Ok(o) => o,
            Err(e) => {
                error!("Inference failed: {}", e);
                continue;
            }
        };

        let output = match outputs["output"].try_extract_array::<f32>() {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to extract output: {}", e);
                continue;
            }
        };

        let out_h = output.shape()[2];
        let out_w = output.shape()[3];
        let output_rgba = tensor_to_rgba(&output, out_w, out_h);

        let final_pixels = resize_rgba(
            &output_rgba,
            out_w as u32,
            out_h as u32,
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

    info!("Inference thread (ort): shutting down");
}
