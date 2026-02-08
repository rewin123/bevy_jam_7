use bevy::prelude::*;
use crossbeam_channel::{bounded, Receiver, Sender};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;

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

/// Message to tell the inference thread which model to use
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

pub struct StyleTransferPlugin;

impl Plugin for StyleTransferPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_inference_thread);
    }
}

fn setup_inference_thread(mut commands: Commands) {
    let (send_frame, recv_frame) = bounded::<FrameData>(1);
    let (send_styled, recv_styled) = bounded::<StyledFrame>(1);
    let (send_switch, recv_switch) = bounded::<StyleSwitch>(4);

    let model_names = vec![
        "candy-9".to_string(),
        "mosaic-9".to_string(),
        "rain-princess-9".to_string(),
        "udnie-9".to_string(),
        "pointilism-9".to_string(),
    ];

    let names_clone = model_names.clone();

    std::thread::spawn(move || {
        inference_thread_main(recv_frame, send_styled, recv_switch, &names_clone);
    });

    commands.insert_resource(StyleChannels {
        send_frame,
        recv_styled,
        send_switch,
    });
    commands.insert_resource(CurrentStyle {
        index: 0,
        names: model_names,
    });
}

fn inference_thread_main(
    recv: Receiver<FrameData>,
    send: Sender<StyledFrame>,
    recv_switch: Receiver<StyleSwitch>,
    model_names: &[String],
) {
    info!("Inference thread: loading ONNX models...");

    let mut sessions: Vec<Session> = model_names
        .iter()
        .map(|name| {
            let path = format!("assets/models/{}.onnx", name);
            Session::builder()
                .unwrap()
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
                .unwrap()
                .commit_from_file(&path)
                .unwrap_or_else(|e| panic!("Failed to load model {}: {}", path, e))
        })
        .collect();

    info!("Inference thread: all {} models loaded", sessions.len());

    let mut current_model = 0usize;

    while let Ok(frame) = recv.recv() {
        // Check for style switch messages
        while let Ok(switch) = recv_switch.try_recv() {
            current_model = switch.index % sessions.len();
            info!("Inference thread: switched to model {}", current_model);
        }

        let w = frame.width as usize;
        let h = frame.height as usize;

        // RGBA → [1, 3, H, W] float32 (0-255 range as model expects)
        let mut input = Array4::<f32>::zeros((1, 3, h, w));
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                input[[0, 0, y, x]] = frame.pixels[idx] as f32;
                input[[0, 1, y, x]] = frame.pixels[idx + 1] as f32;
                input[[0, 2, y, x]] = frame.pixels[idx + 2] as f32;
            }
        }

        // Create ort Tensor from ndarray
        let tensor = match Tensor::from_array(input) {
            Ok(t) => t,
            Err(e) => {
                error!("Failed to create tensor: {}", e);
                continue;
            }
        };

        // Run inference
        let outputs = match sessions[current_model].run(ort::inputs!["input1" => tensor]) {
            Ok(o) => o,
            Err(e) => {
                error!("Inference failed: {}", e);
                continue;
            }
        };

        let out_view = match outputs["output1"].try_extract_array::<f32>() {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to extract output: {}", e);
                continue;
            }
        };

        // [1, 3, H, W] → RGBA Vec<u8>
        let mut rgba = vec![255u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                rgba[idx] = out_view[[0, 0, y, x]].clamp(0.0, 255.0) as u8;
                rgba[idx + 1] = out_view[[0, 1, y, x]].clamp(0.0, 255.0) as u8;
                rgba[idx + 2] = out_view[[0, 2, y, x]].clamp(0.0, 255.0) as u8;
            }
        }

        let _ = send.try_send(StyledFrame {
            pixels: rgba,
            width: frame.width,
            height: frame.height,
        });
    }

    info!("Inference thread: shutting down");
}
