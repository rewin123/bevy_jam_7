use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::view::window::screenshot::{Screenshot, ScreenshotCaptured};
use bevy_camera::RenderTarget;

use crate::player::PlayerCamera;
use crate::style_transfer::{FrameData, StyleChannels};

/// Resolution for inference — model expects exactly 224x224
pub const INFERENCE_SIZE: u32 = 224;

#[derive(Resource)]
pub struct StyleTarget {
    pub render_image: Handle<Image>,
    pub display_image: Handle<Image>,
}

#[derive(Component)]
pub struct StyledDisplay;

#[derive(Resource)]
struct RenderTargetAssigned(bool);

/// Throttle screenshot capture to avoid overwhelming the inference thread
#[derive(Resource)]
struct CaptureTimer(Timer);

/// Tracks whether a screenshot capture is currently in-flight
#[derive(Resource)]
struct CaptureInFlight(bool);

pub struct PostProcessPlugin;

impl Plugin for PostProcessPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_render_targets)
            .add_systems(
                Update,
                (
                    assign_render_target,
                    periodic_capture,
                    receive_styled_frame,
                )
                    .chain(),
            );
    }
}

fn setup_render_targets(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let size = Extent3d {
        width: INFERENCE_SIZE,
        height: INFERENCE_SIZE,
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
    display_image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    display_image.texture_descriptor.dimension = TextureDimension::D2;
    display_image.resize(size);
    let display_handle = images.add(display_image);

    commands.insert_resource(StyleTarget {
        render_image: render_handle,
        display_image: display_handle.clone(),
    });

    commands.insert_resource(RenderTargetAssigned(false));
    commands.insert_resource(CaptureTimer(Timer::from_seconds(
        0.1,
        TimerMode::Repeating,
    )));
    commands.insert_resource(CaptureInFlight(false));

    // Camera to render UI to the window (scene camera goes to offscreen target)
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

/// Periodically capture the render target via Screenshot API and send to inference
fn periodic_capture(
    mut commands: Commands,
    time: Res<Time>,
    mut timer: ResMut<CaptureTimer>,
    mut in_flight: ResMut<CaptureInFlight>,
    style_target: Res<StyleTarget>,
    channels: Option<Res<StyleChannels>>,
) {
    let Some(channels) = channels else { return };

    timer.0.tick(time.delta());
    if !timer.0.just_finished() {
        return;
    }

    // Don't capture if a previous screenshot is still in-flight
    if in_flight.0 {
        return;
    }

    // Don't capture if inference is still processing
    if channels.send_frame.is_full() {
        return;
    }

    in_flight.0 = true;
    let send_frame = channels.send_frame.clone();

    commands
        .spawn(Screenshot::image(style_target.render_image.clone()))
        .observe(move |trigger: On<ScreenshotCaptured>, mut commands: Commands, mut in_flight: ResMut<CaptureInFlight>| {
            in_flight.0 = false;

            // Despawn the screenshot entity to avoid accumulation
            commands.entity(trigger.event_target()).despawn();

            let image = &trigger.image;
            let Some(ref data) = image.data else { return };
            if data.is_empty() {
                return;
            }

            let w = image.width();
            let h = image.height();

            let frame = FrameData {
                pixels: data.clone(),
                width: w,
                height: h,
            };
            let _ = send_frame.try_send(frame);
        });
}

/// Receive styled frames from the inference thread and update the display image
fn receive_styled_frame(
    channels: Option<Res<StyleChannels>>,
    style_target: Res<StyleTarget>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(channels) = channels else { return };

    if let Ok(styled) = channels.recv_styled.try_recv() {
        if let Some(image) = images.get_mut(&style_target.display_image) {
            image.data = Some(styled.pixels);
        }
    }
}
