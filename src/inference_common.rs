//! Common types shared between ort and burn inference backends.

use bevy::prelude::*;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Inference resolution: 512x288 — 16:9, matches training resolution
pub const RENDER_WIDTH: u32 = 512;
pub const RENDER_HEIGHT: u32 = 288;

/// Currently active style
#[derive(Resource)]
pub struct CurrentStyle {
    pub index: usize,
    pub names: Vec<String>,
}

/// Insert this resource to enable test-inference mode
#[derive(Resource)]
pub struct TestInferenceMode;

/// Becomes true when test frames are saved
#[derive(Resource)]
pub struct TestInferenceDone(pub Arc<AtomicBool>);

// ===== Channel-based types (used by WASM burn path and ort backend) =====

/// Raw frame data from the renderer (RGBA, row-major)
pub struct FrameData {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Styled frame data from inference (RGBA)
pub struct StyledFrame {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Command to switch to a different style model
pub struct StyleSwitch {
    pub index: usize,
}

/// Channels for communication between main thread and inference
#[derive(Resource)]
pub struct StyleChannels {
    pub send_frame: crossbeam_channel::Sender<FrameData>,
    pub recv_styled: crossbeam_channel::Receiver<StyledFrame>,
    pub send_switch: crossbeam_channel::Sender<StyleSwitch>,
}

/// Resize RGBA pixel buffer using Lanczos3 filter
pub fn resize_rgba(pixels: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    if src_w == dst_w && src_h == dst_h {
        return pixels.to_vec();
    }
    let img = image::RgbaImage::from_raw(src_w, src_h, pixels.to_vec())
        .expect("resize_rgba: invalid buffer size");
    let resized =
        image::imageops::resize(&img, dst_w, dst_h, image::imageops::FilterType::Lanczos3);
    resized.into_raw()
}
