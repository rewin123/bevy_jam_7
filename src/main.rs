use bevy::prelude::*;

mod fever;
mod level;
mod player;
mod post_process;
mod style_transfer;

fn main() {
    let mut app = App::new();

    app.add_plugins(
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Fever Dream".into(),
                resolution: (1280u32, 720u32).into(),
                ..default()
            }),
            ..default()
        }),
    )
    .add_plugins((
        player::PlayerPlugin,
        level::LevelPlugin,
        style_transfer::StyleTransferPlugin,
        post_process::PostProcessPlugin,
        fever::FeverPlugin,
    ));

    if std::env::args().any(|a| a == "--screenshot-and-exit") {
        app.add_systems(Update, auto_screenshot);
    }

    app.run();
}

fn auto_screenshot(
    mut commands: Commands,
    mut frame_count: Local<u32>,
    mut exit: MessageWriter<AppExit>,
) {
    use bevy::render::view::window::screenshot::{save_to_disk, Screenshot};

    *frame_count += 1;
    // Wait longer to allow style transfer pipeline to produce output
    if *frame_count == 180 {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk("screenshot-0.png"));
    }
    if *frame_count == 190 {
        exit.write(AppExit::Success);
    }
}
