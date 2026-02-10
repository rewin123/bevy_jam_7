use bevy::prelude::*;
use bevy::render::render_resource::{TextureViewDescriptor, TextureViewDimension};

pub struct SkyboxPlugin;

impl Plugin for SkyboxPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_skybox);
        app.add_systems(Update, reinterpret_cubemap_once);
    }
}

/// Tracks the skybox image handle so we can reinterpret it once loaded.
#[derive(Resource)]
struct SkyboxImageHandle(Handle<Image>);

fn setup_skybox(mut commands: Commands, asset_server: Res<AssetServer>) {
    let handle: Handle<Image> = asset_server.load("environment_maps/skybox.png");
    commands.insert_resource(SkyboxImageHandle(handle));
}

/// After the PNG loads, reinterpret the vertical strip (W × 6W) as a cubemap.
fn reinterpret_cubemap_once(
    mut commands: Commands,
    handle: Option<Res<SkyboxImageHandle>>,
    mut images: ResMut<Assets<Image>>,
    camera_q: Query<Entity, With<Camera3d>>,
) {
    let Some(handle) = handle else { return };

    let Some(image) = images.get_mut(&handle.0) else {
        return;
    };

    // Only reinterpret once: check if already a cube
    if image.texture_descriptor.array_layer_count() > 1 {
        return;
    }

    let face_count = image.height() / image.width();
    if face_count != 6 {
        warn!(
            "Skybox image has unexpected aspect ratio: {}x{} ({} faces expected 6)",
            image.width(),
            image.height(),
            face_count
        );
        return;
    }

    info!("Reinterpreting skybox {}x{} as cubemap", image.width(), image.height());

    image
        .reinterpret_stacked_2d_as_array(face_count)
        .expect("Failed to reinterpret skybox as array texture");

    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });

    // Add Skybox component to all Camera3d entities
    let skybox = bevy::core_pipeline::Skybox {
        image: handle.0.clone(),
        brightness: 1500.0,
        ..default()
    };

    for entity in &camera_q {
        commands.entity(entity).insert(skybox.clone());
    }

    // Remove the resource so this system stops running
    commands.remove_resource::<SkyboxImageHandle>();
}
