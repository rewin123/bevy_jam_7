use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy_camera::RenderTarget;

use crate::inference_common::{CurrentStyle, RENDER_HEIGHT, RENDER_WIDTH};
use crate::player::PlayerCamera;

#[derive(Resource)]
pub struct StyleTarget {
    pub render_image: Handle<Image>,
    pub display_image: Handle<Image>,
}

#[derive(Component)]
pub struct StyledDisplay;

#[derive(Resource)]
struct RenderTargetAssigned(bool);

/// When true, style transfer is bypassed and the raw scene is displayed
#[derive(Resource)]
pub struct StyleBypass(pub bool);

pub struct PostProcessPlugin;

impl Plugin for PostProcessPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_render_targets)
            .add_systems(
                Update,
                (keyboard_style_switch, assign_render_target).chain(),
            );

        // WASM still uses Screenshot-based capture + channels
        #[cfg(all(target_arch = "wasm32", feature = "burn-backend"))]
        {
            app.insert_resource(CaptureTimer(Timer::from_seconds(0.1, TimerMode::Repeating)));
            app.insert_resource(CaptureInFlight(false));
            app.add_systems(
                Update,
                (wasm_periodic_capture, wasm_receive_styled_frame).chain(),
            );
        }

        // ort backend still uses Screenshot-based capture + channels
        #[cfg(feature = "ort-backend")]
        {
            app.insert_resource(CaptureTimer(Timer::from_seconds(0.1, TimerMode::Repeating)));
            app.insert_resource(CaptureInFlight(false));
            app.add_systems(
                Update,
                (ort_periodic_capture, ort_receive_styled_frame).chain(),
            );
        }

        // WebGL lacks TEXTURE_FORMAT_16BIT_NORM
        #[cfg(target_arch = "wasm32")]
        app.add_systems(Update, downgrade_16bit_images);
    }
}

/// Downgrade Rgba16Unorm images to Rgba8UnormSrgb for WebGL compatibility.
#[cfg(target_arch = "wasm32")]
fn downgrade_16bit_images(mut images: ResMut<Assets<Image>>) {
    let ids: Vec<_> = images
        .iter()
        .filter(|(_, img)| img.texture_descriptor.format == TextureFormat::Rgba16Unorm)
        .map(|(id, _)| id)
        .collect();

    for id in ids {
        let Some(image) = images.get_mut(id) else {
            continue;
        };
        if let Some(ref data) = image.data {
            let new_data: Vec<u8> = data.chunks_exact(2).map(|c| c[1]).collect();
            image.data = Some(new_data);
            image.texture_descriptor.format = TextureFormat::Rgba8UnormSrgb;
            warn!("Downgraded Rgba16Unorm image to Rgba8UnormSrgb for WebGL");
        }
    }
}

fn setup_render_targets(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let size = Extent3d {
        width: RENDER_WIDTH,
        height: RENDER_HEIGHT,
        depth_or_array_layers: 1,
    };

    // Render target: scene camera renders here
    let mut render_image = Image::default();
    render_image.texture_descriptor.size = size;
    render_image.texture_descriptor.format = TextureFormat::Rgba8UnormSrgb;
    render_image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_SRC
        | TextureUsages::RENDER_ATTACHMENT;
    render_image.texture_descriptor.dimension = TextureDimension::D2;
    render_image.resize(size);
    let render_handle = images.add(render_image);

    // Display image: styled result shown to user
    let mut display_image = Image::default();
    display_image.texture_descriptor.size = size;
    display_image.texture_descriptor.format = TextureFormat::Rgba8UnormSrgb;
    display_image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC; // COPY_SRC for bypass direct copy
    display_image.texture_descriptor.dimension = TextureDimension::D2;
    display_image.resize(size);
    let display_handle = images.add(display_image);

    commands.insert_resource(StyleTarget {
        render_image: render_handle,
        display_image: display_handle.clone(),
    });

    commands.insert_resource(RenderTargetAssigned(false));
    commands.insert_resource(StyleBypass(false));

    // Camera to render UI to the window
    commands.spawn((
        Camera2d,
        Camera {
            order: 10,
            ..default()
        },
    ));

    // Fullscreen UI to display the styled image
    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            ..default()
        })
        .with_child((
            StyledDisplay,
            ImageNode::new(display_handle),
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                ..default()
            },
        ));
}

/// Assign the render target to the player camera
fn assign_render_target(
    mut commands: Commands,
    mut assigned: ResMut<RenderTargetAssigned>,
    style_target: Res<StyleTarget>,
    camera_q: Query<Entity, With<PlayerCamera>>,
) {
    if assigned.0 {
        return;
    }

    for entity in &camera_q {
        commands
            .entity(entity)
            .insert(RenderTarget::from(style_target.render_image.clone()));
        assigned.0 = true;
        info!("Render target assigned to player camera");
    }
}

/// Switch styles with keyboard: 0 = no style, 1-N = specific style
fn keyboard_style_switch(
    keys: Res<ButtonInput<KeyCode>>,
    mut bypass: ResMut<StyleBypass>,
    mut style: ResMut<CurrentStyle>,
    // For WASM / ort: send switch command via channel
    #[cfg(any(target_arch = "wasm32", feature = "ort-backend"))]
    channels: Option<Res<crate::inference_common::StyleChannels>>,
) {
    let digit_keys = [
        (KeyCode::Digit0, 0u32),
        (KeyCode::Digit1, 1),
        (KeyCode::Digit2, 2),
        (KeyCode::Digit3, 3),
        (KeyCode::Digit4, 4),
        (KeyCode::Digit5, 5),
        (KeyCode::Digit6, 6),
        (KeyCode::Digit7, 7),
        (KeyCode::Digit8, 8),
        (KeyCode::Digit9, 9),
    ];

    for (key, digit) in digit_keys {
        if !keys.just_pressed(key) {
            continue;
        }

        if digit == 0 {
            bypass.0 = true;
            info!("Style: OFF (raw scene)");
            return;
        }

        let style_idx = (digit - 1) as usize;
        if style_idx >= style.names.len() {
            continue;
        }

        bypass.0 = false;
        style.index = style_idx;
        info!("Style: {} ({})", digit, style.names[style_idx]);

        #[cfg(any(target_arch = "wasm32", feature = "ort-backend"))]
        if let Some(ref channels) = channels {
            let _ = channels
                .send_switch
                .try_send(crate::inference_common::StyleSwitch { index: style_idx });
        }
        return;
    }
}

// ===== Screenshot-based capture for WASM and ort backends =====

/// Throttle screenshot capture
#[cfg(any(all(target_arch = "wasm32", feature = "burn-backend"), feature = "ort-backend"))]
#[derive(Resource)]
struct CaptureTimer(Timer);

/// Tracks whether a screenshot capture is in-flight
#[cfg(any(all(target_arch = "wasm32", feature = "burn-backend"), feature = "ort-backend"))]
#[derive(Resource)]
struct CaptureInFlight(bool);

#[cfg(all(target_arch = "wasm32", feature = "burn-backend"))]
fn wasm_periodic_capture(
    mut commands: Commands,
    time: Res<Time>,
    mut timer: ResMut<CaptureTimer>,
    mut in_flight: ResMut<CaptureInFlight>,
    style_target: Res<StyleTarget>,
    channels: Option<Res<crate::inference_common::StyleChannels>>,
) {
    use bevy::render::view::window::screenshot::{Screenshot, ScreenshotCaptured};
    use crate::inference_common::FrameData;

    let Some(channels) = channels else { return };
    timer.0.tick(time.delta());
    if !timer.0.just_finished() || in_flight.0 || channels.send_frame.is_full() {
        return;
    }

    in_flight.0 = true;
    let send_frame = channels.send_frame.clone();

    commands
        .spawn(Screenshot::image(style_target.render_image.clone()))
        .observe(move |trigger: On<ScreenshotCaptured>,
                       mut commands: Commands,
                       mut in_flight: ResMut<CaptureInFlight>,
                       bypass: Res<StyleBypass>,
                       style_target: Res<StyleTarget>,
                       mut images: ResMut<Assets<Image>>| {
            in_flight.0 = false;
            commands.entity(trigger.event_target()).despawn();
            let image = &trigger.image;
            let Some(ref data) = image.data else { return };
            if data.is_empty() { return; }

            if bypass.0 {
                if let Some(display) = images.get_mut(&style_target.display_image) {
                    display.data = Some(data.clone());
                }
            } else {
                let frame = FrameData { pixels: data.clone(), width: image.width(), height: image.height() };
                let _ = send_frame.try_send(frame);
            }
        });
}

#[cfg(all(target_arch = "wasm32", feature = "burn-backend"))]
fn wasm_receive_styled_frame(
    channels: Option<Res<crate::inference_common::StyleChannels>>,
    style_target: Res<StyleTarget>,
    mut images: ResMut<Assets<Image>>,
    bypass: Res<StyleBypass>,
) {
    if bypass.0 {
        if let Some(channels) = channels {
            while channels.recv_styled.try_recv().is_ok() {}
        }
        return;
    }
    let Some(channels) = channels else { return };
    if let Ok(styled) = channels.recv_styled.try_recv() {
        if let Some(image) = images.get_mut(&style_target.display_image) {
            image.data = Some(styled.pixels);
        }
    }
}

#[cfg(feature = "ort-backend")]
fn ort_periodic_capture(
    mut commands: Commands,
    time: Res<Time>,
    mut timer: ResMut<CaptureTimer>,
    mut in_flight: ResMut<CaptureInFlight>,
    style_target: Res<StyleTarget>,
    channels: Option<Res<crate::inference_common::StyleChannels>>,
) {
    use bevy::render::view::window::screenshot::{Screenshot, ScreenshotCaptured};
    use crate::inference_common::FrameData;

    let Some(channels) = channels else { return };
    timer.0.tick(time.delta());
    if !timer.0.just_finished() || in_flight.0 || channels.send_frame.is_full() {
        return;
    }

    in_flight.0 = true;
    let send_frame = channels.send_frame.clone();

    commands
        .spawn(Screenshot::image(style_target.render_image.clone()))
        .observe(move |trigger: On<ScreenshotCaptured>,
                       mut commands: Commands,
                       mut in_flight: ResMut<CaptureInFlight>,
                       bypass: Res<StyleBypass>,
                       style_target: Res<StyleTarget>,
                       mut images: ResMut<Assets<Image>>| {
            in_flight.0 = false;
            commands.entity(trigger.event_target()).despawn();
            let image = &trigger.image;
            let Some(ref data) = image.data else { return };
            if data.is_empty() { return; }

            if bypass.0 {
                if let Some(display) = images.get_mut(&style_target.display_image) {
                    display.data = Some(data.clone());
                }
            } else {
                let frame = FrameData { pixels: data.clone(), width: image.width(), height: image.height() };
                let _ = send_frame.try_send(frame);
            }
        });
}

#[cfg(feature = "ort-backend")]
fn ort_receive_styled_frame(
    channels: Option<Res<crate::inference_common::StyleChannels>>,
    style_target: Res<StyleTarget>,
    mut images: ResMut<Assets<Image>>,
    bypass: Res<StyleBypass>,
) {
    if bypass.0 {
        if let Some(channels) = channels {
            while channels.recv_styled.try_recv().is_ok() {}
        }
        return;
    }
    let Some(channels) = channels else { return };
    if let Ok(styled) = channels.recv_styled.try_recv() {
        if let Some(image) = images.get_mut(&style_target.display_image) {
            image.data = Some(styled.pixels);
        }
    }
}
