use std::sync::atomic::Ordering;

use avian3d::prelude::*;
use bevy::log::LogPlugin;
use bevy::prelude::*;
use bevy_skein::SkeinPlugin;

use fever_dream::*;

fn main() {
    let mut app = App::new();

    let is_test_inference = std::env::args().any(|a| a == "--test-inference");

    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Fever Dream".into(),
                    resolution: (1280u32, 720u32).into(),
                    ..default()
                }),
                ..default()
            })
            .set(LogPlugin {
                filter: "ort=error".into(),
                ..default()
            }),
    )
    .add_plugins((
        PhysicsPlugins::default(),
        SkeinPlugin::default(),
        player::PlayerPlugin,
        level::LevelPlugin,
        style_transfer::StyleTransferPlugin,
        post_process::PostProcessPlugin,
        fever::FeverPlugin,
        triggers::TriggersPlugin,
        world_layer::WorldLayerPlugin,
        skybox::SkyboxPlugin,
    ));

    if is_test_inference {
        app.insert_resource(inference_common::TestInferenceMode);
        app.add_systems(Update, check_test_inference_done);
    }

    if std::env::args().any(|a| a == "--screenshot-and-exit") {
        app.add_systems(Update, auto_screenshot);
    }

    app.run();
}

fn check_test_inference_done(
    test_done: Option<Res<inference_common::TestInferenceDone>>,
    mut exit: MessageWriter<AppExit>,
) {
    if let Some(done) = test_done {
        if done.0.load(Ordering::Acquire) {
            info!("Test inference frames saved, exiting");
            exit.write(AppExit::Success);
        }
    }
}



fn auto_screenshot(
    mut commands: Commands,
    mut frame_count: Local<u32>,
    mut exit: MessageWriter<AppExit>,
) {
    use bevy::render::view::window::screenshot::{save_to_disk, Screenshot};

    *frame_count += 1;
    // Wait longer to allow style transfer pipeline to produce output
    if *frame_count == 600 {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk("screenshot-0.png"));
    }
    if *frame_count == 610 {
        exit.write(AppExit::Success);
    }
}
