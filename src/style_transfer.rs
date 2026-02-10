use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bevy::prelude::*;
use crossbeam_channel::{bounded, Receiver, Sender};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;

// ===== Inference resolution =====
// Model5Seq has dynamic axes; we use 512 (training resolution) as height.
pub const INFERENCE_SIZE: u32 = 512;

/// Render target dimensions — 16:9 aspect ratio at INFERENCE_SIZE height
pub const RENDER_WIDTH: u32 = INFERENCE_SIZE * 16 / 9;
pub const RENDER_HEIGHT: u32 = INFERENCE_SIZE;

// ===== Common types =====

pub struct FrameData {
    pub pixels: Vec<u8>, // RGBA, row-major
    pub width: u32,
    pub height: u32,
}

pub struct StyledFrame {
    pub pixels: Vec<u8>, // RGBA
    pub width: u32,
    pub height: u32,
}

pub struct StyleSwitch {
    pub index: usize,
}

#[derive(Resource)]
pub struct StyleChannels {
    pub send_frame: Sender<FrameData>,
    pub recv_styled: Receiver<StyledFrame>,
    pub send_switch: Sender<StyleSwitch>,
}

#[derive(Resource)]
pub struct CurrentStyle {
    pub index: usize,
    pub names: Vec<String>,
}

/// Insert this resource to enable test-inference mode:
/// saves raw input + styled output from the first inference pass, then signals done.
#[derive(Resource)]
pub struct TestInferenceMode;

/// Inserted when test mode is active. Becomes true when test frames are saved.
#[derive(Resource)]
pub struct TestInferenceDone(pub Arc<AtomicBool>);

pub struct StyleTransferPlugin;

impl Plugin for StyleTransferPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_inference_thread);
    }
}

// ===== Helpers =====

/// Resize RGBA pixel buffer using Lanczos3 filter
fn resize_rgba(pixels: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    if src_w == dst_w && src_h == dst_h {
        return pixels.to_vec();
    }
    let img = image::RgbaImage::from_raw(src_w, src_h, pixels.to_vec())
        .expect("resize_rgba: invalid buffer size");
    let resized =
        image::imageops::resize(&img, dst_w, dst_h, image::imageops::FilterType::Lanczos3);
    resized.into_raw()
}

/// RGBA [u8] -> [1, 3, H, W] float32 [0, 1]
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

/// [1, 3, H, W] float32 [0, 1] -> RGBA [u8]
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

/// Scan `assets/models/styles/` for .onnx files, return sorted (name, path) pairs.
fn scan_model_directory() -> Vec<(String, String)> {
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

// ===== Setup =====

fn setup_inference_thread(mut commands: Commands, test_mode: Option<Res<TestInferenceMode>>) {
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
            error!("No .onnx style models found in assets/models/styles/ — cannot run test");
            test_done.store(true, Ordering::Release);
        } else {
            warn!("No .onnx style models found in assets/models/styles/");
        }
    }

    let names: Vec<String> = models.iter().map(|(name, _)| name.clone()).collect();
    let paths: Vec<String> = models.iter().map(|(_, path)| path.clone()).collect();

    info!("Style models: {:?}", names);

    std::thread::spawn(move || {
        inference_thread_main(recv_frame, send_styled, recv_switch, &paths, is_test, test_done);
    });

    commands.insert_resource(StyleChannels {
        send_frame,
        recv_styled,
        send_switch,
    });
    commands.insert_resource(CurrentStyle { index: 0, names });
}

// ===== Inference thread =====

fn inference_thread_main(
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
        "Inference thread: {} models available, loading lazily",
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

        // Lazy-load on first use
        if sessions[current_model].is_none() {
            info!("Loading model '{}'...", model_paths[current_model]);
            sessions[current_model] = Some(build_session(&model_paths[current_model]));
            info!("Model loaded");
        }
        let session = sessions[current_model].as_mut().unwrap();

        // Resize to render dimensions (16:9 at INFERENCE_SIZE height)
        let w = RENDER_WIDTH as usize;
        let h = RENDER_HEIGHT as usize;
        let resized = resize_rgba(
            &frame.pixels,
            frame.width,
            frame.height,
            RENDER_WIDTH,
            RENDER_HEIGHT,
        );

        // Test mode: wait 1s for the scene to stabilize, then save frames
        let test_ready = test_mode && !test_saved
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

        // RGBA -> [1, 3, H, W] float32 [0, 1]
        let input_nd = rgba_to_tensor(&resized, w, h);
        let input_tensor = match Tensor::from_array(input_nd) {
            Ok(t) => t,
            Err(e) => {
                error!("Failed to create input tensor: {}", e);
                continue;
            }
        };

        // Run inference
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

        // Resize to display dimensions if model output differs
        let final_pixels = resize_rgba(
            &output_rgba,
            out_w as u32,
            out_h as u32,
            RENDER_WIDTH,
            RENDER_HEIGHT,
        );

        // Test mode: save the output frame
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

    info!("Inference thread: shutting down");
}
