use bevy::prelude::*;
use crossbeam_channel::{bounded, Receiver, Sender};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;

// ===== Compile-time mutual exclusivity =====
#[cfg(all(feature = "style-johnson", feature = "style-adain"))]
compile_error!("Features `style-johnson` and `style-adain` are mutually exclusive");
#[cfg(all(feature = "style-johnson", feature = "style-microast"))]
compile_error!("Features `style-johnson` and `style-microast` are mutually exclusive");
#[cfg(all(feature = "style-adain", feature = "style-microast"))]
compile_error!("Features `style-adain` and `style-microast` are mutually exclusive");
#[cfg(not(any(feature = "style-johnson", feature = "style-adain", feature = "style-microast")))]
compile_error!("Exactly one style backend must be enabled: style-johnson, style-adain, or style-microast");

// ===== Common types (all backends) =====

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

pub struct StyleTransferPlugin;

impl Plugin for StyleTransferPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_inference_thread);
    }
}

// ===== Shared helpers for AdaIN and MicroAST =====

#[cfg(any(feature = "style-adain", feature = "style-microast"))]
fn scan_style_directory() -> Vec<(String, String)> {
    let style_dir = "assets/styles";
    let mut styles = Vec::new();

    if let Ok(entries) = std::fs::read_dir(style_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext_lower = ext.to_string_lossy().to_lowercase();
                if ext_lower == "jpg" || ext_lower == "jpeg" || ext_lower == "png" {
                    let name = path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    styles.push((name, path.to_string_lossy().to_string()));
                }
            }
        }
    }

    styles.sort_by(|a, b| a.0.cmp(&b.0));
    styles
}

#[cfg(any(feature = "style-adain", feature = "style-microast"))]
fn load_style_image(path: &str, target_w: u32, target_h: u32) -> Array4<f32> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to load style image {}: {}", path, e))
        .resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3)
        .into_rgb8();

    let mut tensor = Array4::<f32>::zeros((1, 3, target_h as usize, target_w as usize));
    for y in 0..target_h as usize {
        for x in 0..target_w as usize {
            let pixel = img.get_pixel(x as u32, y as u32);
            tensor[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
            tensor[[0, 1, y, x]] = pixel[1] as f32 / 255.0;
            tensor[[0, 2, y, x]] = pixel[2] as f32 / 255.0;
        }
    }
    tensor
}

#[cfg(any(feature = "style-adain", feature = "style-microast"))]
fn rgba_to_tensor_01(pixels: &[u8], w: usize, h: usize) -> Array4<f32> {
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

#[cfg(any(feature = "style-adain", feature = "style-microast"))]
fn tensor_01_to_rgba(tensor: &ndarray::ArrayViewD<'_, f32>, w: usize, h: usize) -> Vec<u8> {
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

#[cfg(any(feature = "style-adain", feature = "style-microast"))]
fn build_session(path: &str) -> Session {
    Session::builder()
        .unwrap()
        .with_execution_providers([
            ort::execution_providers::CPUExecutionProvider::default().build(),
        ])
        .unwrap()
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Disable)
        .unwrap()
        .commit_from_file(path)
        .unwrap_or_else(|e| panic!("Failed to load model {}: {}", path, e))
}

// =============================================================================
// JOHNSON BACKEND
// =============================================================================

#[cfg(feature = "style-johnson")]
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

#[cfg(feature = "style-johnson")]
fn inference_thread_main(
    recv: Receiver<FrameData>,
    send: Sender<StyledFrame>,
    recv_switch: Receiver<StyleSwitch>,
    model_names: &[String],
) {
    info!("Inference thread [Johnson]: loading ONNX models...");

    let mut sessions: Vec<Session> = model_names
        .iter()
        .map(|name| {
            let path = format!("assets/models/{}.onnx", name);
            Session::builder()
                .unwrap()
                .with_execution_providers([
                    ort::execution_providers::CPUExecutionProvider::default().build(),
                ])
                .unwrap()
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
                .unwrap()
                .commit_from_file(&path)
                .unwrap_or_else(|e| panic!("Failed to load model {}: {}", path, e))
        })
        .collect();

    info!(
        "Inference thread [Johnson]: all {} models loaded",
        sessions.len()
    );

    let mut current_model = 0usize;

    while let Ok(frame) = recv.recv() {
        while let Ok(switch) = recv_switch.try_recv() {
            current_model = switch.index % sessions.len();
            info!(
                "Inference thread [Johnson]: switched to model {}",
                current_model
            );
        }

        let w = frame.width as usize;
        let h = frame.height as usize;

        // RGBA -> [1, 3, H, W] float32 (0-255 range)
        let mut input = Array4::<f32>::zeros((1, 3, h, w));
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                input[[0, 0, y, x]] = frame.pixels[idx] as f32;
                input[[0, 1, y, x]] = frame.pixels[idx + 1] as f32;
                input[[0, 2, y, x]] = frame.pixels[idx + 2] as f32;
            }
        }

        let tensor = match Tensor::from_array(input) {
            Ok(t) => t,
            Err(e) => {
                error!("Failed to create tensor: {}", e);
                continue;
            }
        };

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

        // [1, 3, H, W] -> RGBA Vec<u8>
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

    info!("Inference thread [Johnson]: shutting down");
}

// =============================================================================
// ADAIN BACKEND
// =============================================================================

#[cfg(feature = "style-adain")]
fn adain(content: &Array4<f32>, style: &Array4<f32>) -> Array4<f32> {
    use ndarray::s;

    let (_, c, _, _) = content.dim();
    let mut result = Array4::<f32>::zeros(content.dim());

    for ch in 0..c {
        let content_slice = content.slice(s![0, ch, .., ..]);
        let style_slice = style.slice(s![0, ch, .., ..]);

        let c_mean = content_slice.mean().unwrap_or(0.0);
        let s_mean = style_slice.mean().unwrap_or(0.0);

        let c_var = content_slice.mapv(|x| (x - c_mean).powi(2)).mean().unwrap_or(0.0);
        let s_var = style_slice.mapv(|x| (x - s_mean).powi(2)).mean().unwrap_or(0.0);

        let c_std = (c_var + 1e-5_f32).sqrt();
        let s_std = (s_var + 1e-5_f32).sqrt();

        result
            .slice_mut(s![0, ch, .., ..])
            .assign(&content_slice.mapv(|x| (x - c_mean) / c_std * s_std + s_mean));
    }
    result
}

#[cfg(feature = "style-adain")]
fn setup_inference_thread(mut commands: Commands) {
    let (send_frame, recv_frame) = bounded::<FrameData>(1);
    let (send_styled, recv_styled) = bounded::<StyledFrame>(1);
    let (send_switch, recv_switch) = bounded::<StyleSwitch>(4);

    let styles = scan_style_directory();
    if styles.is_empty() {
        error!("AdaIN: no style images found in assets/styles/! Add .jpg or .png files.");
    }

    let names: Vec<String> = styles.iter().map(|(name, _)| name.clone()).collect();
    let paths: Vec<String> = styles.iter().map(|(_, path)| path.clone()).collect();

    std::thread::spawn(move || {
        inference_thread_main_adain(recv_frame, send_styled, recv_switch, &paths);
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

#[cfg(feature = "style-adain")]
fn inference_thread_main_adain(
    recv: Receiver<FrameData>,
    send: Sender<StyledFrame>,
    recv_switch: Receiver<StyleSwitch>,
    style_paths: &[String],
) {
    use crate::post_process::INFERENCE_SIZE;

    info!("Inference thread [AdaIN]: loading encoder and decoder...");
    let mut encoder = build_session("assets/models/adain-vgg.onnx");
    let mut decoder = build_session("assets/models/adain-decoder.onnx");

    // Lazy cache: encode style images through VGG on demand
    let inf_size = INFERENCE_SIZE;
    let num_styles = style_paths.len();
    let mut style_features: Vec<Option<Array4<f32>>> = vec![None; num_styles];

    info!("AdaIN: {} style images found, encoding on demand at {}x{}", num_styles, inf_size, inf_size);

    let mut current_style = 0usize;

    while let Ok(frame) = recv.recv() {
        while let Ok(switch) = recv_switch.try_recv() {
            if num_styles > 0 {
                current_style = switch.index % num_styles;
                info!("Inference thread [AdaIN]: switched to style {}", current_style);
            }
        }

        if num_styles == 0 {
            let _ = send.try_send(StyledFrame {
                pixels: frame.pixels,
                width: frame.width,
                height: frame.height,
            });
            continue;
        }

        // Lazy-encode current style if not cached
        if style_features[current_style].is_none() {
            info!("AdaIN: encoding style {} '{}'...", current_style, style_paths[current_style]);
            let style_tensor = load_style_image(&style_paths[current_style], inf_size, inf_size);
            let tensor = Tensor::from_array(style_tensor).unwrap();
            let outputs = encoder.run(ort::inputs!["input" => tensor]).unwrap();
            let feat = outputs["output"].try_extract_array::<f32>().unwrap();
            style_features[current_style] = Some(
                feat.to_owned().into_dimensionality::<ndarray::Ix4>().unwrap()
            );
            info!("AdaIN: style {} encoded", current_style);
        }

        let w = frame.width as usize;
        let h = frame.height as usize;

        let content_nd = rgba_to_tensor_01(&frame.pixels, w, h);
        let content_tensor = match Tensor::from_array(content_nd) {
            Ok(t) => t,
            Err(e) => {
                error!("AdaIN: failed to create content tensor: {}", e);
                continue;
            }
        };

        // Encode content
        let enc_out = match encoder.run(ort::inputs!["input" => content_tensor]) {
            Ok(o) => o,
            Err(e) => {
                error!("AdaIN encoder failed: {}", e);
                continue;
            }
        };
        let content_features = match enc_out["output"].try_extract_array::<f32>() {
            Ok(v) => v.to_owned().into_dimensionality::<ndarray::Ix4>().unwrap(),
            Err(e) => {
                error!("AdaIN: failed to extract encoder output: {}", e);
                continue;
            }
        };

        // AdaIN: transfer statistics
        let transferred = adain(&content_features, style_features[current_style].as_ref().unwrap());

        // Decode
        let dec_tensor = match Tensor::from_array(transferred) {
            Ok(t) => t,
            Err(e) => {
                error!("AdaIN: failed to create decoder tensor: {}", e);
                continue;
            }
        };
        let dec_out = match decoder.run(ort::inputs!["input" => dec_tensor]) {
            Ok(o) => o,
            Err(e) => {
                error!("AdaIN decoder failed: {}", e);
                continue;
            }
        };
        let output = match dec_out["output"].try_extract_array::<f32>() {
            Ok(v) => v,
            Err(e) => {
                error!("AdaIN: failed to extract decoder output: {}", e);
                continue;
            }
        };

        // The decoder may output a different spatial size due to VGG pooling/upsampling
        // Get actual output dimensions from tensor shape
        let out_shape = output.shape();
        let out_h = out_shape[2];
        let out_w = out_shape[3];
        let rgba = tensor_01_to_rgba(&output, out_w, out_h);

        let _ = send.try_send(StyledFrame {
            pixels: rgba,
            width: out_w as u32,
            height: out_h as u32,
        });
    }

    info!("Inference thread [AdaIN]: shutting down");
}

// =============================================================================
// MICROAST BACKEND
// =============================================================================

#[cfg(feature = "style-microast")]
fn setup_inference_thread(mut commands: Commands) {
    let (send_frame, recv_frame) = bounded::<FrameData>(1);
    let (send_styled, recv_styled) = bounded::<StyledFrame>(1);
    let (send_switch, recv_switch) = bounded::<StyleSwitch>(4);

    let styles = scan_style_directory();
    if styles.is_empty() {
        error!("MicroAST: no style images found in assets/styles/! Add .jpg or .png files.");
    }

    let names: Vec<String> = styles.iter().map(|(name, _)| name.clone()).collect();
    let paths: Vec<String> = styles.iter().map(|(_, path)| path.clone()).collect();

    std::thread::spawn(move || {
        inference_thread_main_microast(recv_frame, send_styled, recv_switch, &paths);
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

#[cfg(feature = "style-microast")]
fn inference_thread_main_microast(
    recv: Receiver<FrameData>,
    send: Sender<StyledFrame>,
    recv_switch: Receiver<StyleSwitch>,
    style_paths: &[String],
) {
    use crate::post_process::INFERENCE_SIZE;

    info!("Inference thread [MicroAST]: loading model...");
    let mut session = build_session("assets/models/microast.onnx");

    // Lazy-load style images on demand
    let num_styles = style_paths.len();
    let mut style_tensors: Vec<Option<Array4<f32>>> = vec![None; num_styles];
    info!("MicroAST: {} style images found, loading on demand", num_styles);

    let mut current_style = 0usize;

    while let Ok(frame) = recv.recv() {
        while let Ok(switch) = recv_switch.try_recv() {
            if num_styles > 0 {
                current_style = switch.index % num_styles;
                info!(
                    "Inference thread [MicroAST]: switched to style {}",
                    current_style
                );
            }
        }

        if num_styles == 0 {
            let _ = send.try_send(StyledFrame {
                pixels: frame.pixels,
                width: frame.width,
                height: frame.height,
            });
            continue;
        }

        // Lazy-load current style image if not cached
        if style_tensors[current_style].is_none() {
            info!("MicroAST: loading style {} '{}'...", current_style, style_paths[current_style]);
            style_tensors[current_style] = Some(load_style_image(&style_paths[current_style], INFERENCE_SIZE, INFERENCE_SIZE));
            info!("MicroAST: style {} loaded", current_style);
        }

        let w = frame.width as usize;
        let h = frame.height as usize;

        let content = rgba_to_tensor_01(&frame.pixels, w, h);
        let content_tensor = match Tensor::from_array(content) {
            Ok(t) => t,
            Err(e) => {
                error!("MicroAST: failed to create content tensor: {}", e);
                continue;
            }
        };
        let style_tensor = match Tensor::from_array(style_tensors[current_style].as_ref().unwrap().clone()) {
            Ok(t) => t,
            Err(e) => {
                error!("MicroAST: failed to create style tensor: {}", e);
                continue;
            }
        };

        let outputs = match session
            .run(ort::inputs!["content" => content_tensor, "style" => style_tensor])
        {
            Ok(o) => o,
            Err(e) => {
                error!("MicroAST inference failed: {}", e);
                continue;
            }
        };

        let output = match outputs["output"].try_extract_array::<f32>() {
            Ok(v) => v,
            Err(e) => {
                error!("MicroAST: failed to extract output: {}", e);
                continue;
            }
        };

        let rgba = tensor_01_to_rgba(&output, w, h);

        let _ = send.try_send(StyledFrame {
            pixels: rgba,
            width: frame.width,
            height: frame.height,
        });
    }

    info!("Inference thread [MicroAST]: shutting down");
}
