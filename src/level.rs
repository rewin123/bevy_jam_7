use avian3d::prelude::*;
use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::scene::SceneInstanceReady;

use crate::world_layer::{ChangeWorldLayer, MAX_WORLDS};

/// Marker: place on an empty in Blender to set the player spawn point.
#[derive(Component, Reflect, Default, Debug)]
#[reflect(Component)]
pub struct PlayerStart;

/// Marker: place on mesh objects in Blender to auto-generate trimesh colliders.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
pub struct AutoMeshCollider;

pub struct LevelPlugin;

impl Plugin for LevelPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_systems(Startup, (spawn_blender_level, spawn_lighting));

        app.add_systems(Update, with_auto_mesh);
    }
}

fn spawn_blender_level(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands
        .spawn(SceneRoot(
            asset_server.load(GltfAssetLabel::Scene(0).from_asset("levels/Untitled.glb")),
        )).observe(on_scene_ready);
}


fn with_auto_mesh(
    mut commands: Commands,
    auto_colliders: Query<(Entity, &AutoMeshCollider, Option<&Children>)>,
    q_meshes: Query<&Mesh3d>,
    meshes: Res<Assets<Mesh>>,
) {
    for (entity, _, children) in auto_colliders.iter() {
        info!("New AutoMeshCollider");
        let Some(children) = children else {
            warn!("AutoMeshCollider without childrens");
            continue;
        };

        for &e in children {
            if let Ok(mesh_handle) = q_meshes.get(e) {
                if let Some(mesh) = meshes.get(&mesh_handle.0) {
                    if let Some(collider) = Collider::trimesh_from_mesh(mesh) {
                        commands.entity(entity).insert((RigidBody::Static, collider));
                        commands.entity(entity).remove::<AutoMeshCollider>();
                        info!("AutoMeshCollider: added trimesh collider to {entity}");
                    } else {
                        warn!("AutoMeshCollider: failed to generate trimesh for {entity}");
                    }
                } else {
                    warn!("AutoMeshCollider: mesh not loaded for {entity}");
                }
            }
        }
        
    }
}

fn on_scene_ready(
    _trigger: On<SceneInstanceReady>,
    mut commands: Commands,
) {
    commands.write_message(ChangeWorldLayer(0));
}

fn spawn_lighting(mut commands: Commands) {
    // Light must be on ALL render layers (0 = shared, 1..=MAX_WORLDS = world layers)
    let all_layers = RenderLayers::from_layers(
        &(0..=MAX_WORLDS).collect::<Vec<_>>(),
    );

    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.5, 0.0)),
        all_layers,
    ));

    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.6, 0.5, 0.7),
        brightness: 800.0,
        affects_lightmapped_meshes: true,
    });
}

/// Procedural fallback level (unused when loading from Blender).
#[allow(dead_code)]
fn spawn_level(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    use rand::Rng;
    let mut rng = rand::rng();

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(80.0, 80.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.15, 0.25, 0.15),
            perceptual_roughness: 0.9,
            ..default()
        })),
        RigidBody::Static,
        Collider::cuboid(80.0, 0.01, 80.0),
    ));

    let fever_colors = [
        Color::srgb(0.9, 0.1, 0.3),
        Color::srgb(0.1, 0.2, 0.9),
        Color::srgb(0.9, 0.8, 0.1),
        Color::srgb(0.6, 0.1, 0.9),
        Color::srgb(0.1, 0.9, 0.5),
        Color::srgb(0.9, 0.4, 0.1),
        Color::srgb(0.2, 0.8, 0.9),
        Color::srgb(0.9, 0.2, 0.7),
    ];

    for _ in 0..20 {
        let color = fever_colors[rng.random_range(0..fever_colors.len())];
        let w = rng.random_range(1.0..4.0_f32);
        let h = rng.random_range(1.5..8.0_f32);
        let d = rng.random_range(1.0..4.0_f32);
        let x = rng.random_range(-30.0..30.0_f32);
        let z = rng.random_range(-30.0..30.0_f32);
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(w, h, d))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                perceptual_roughness: rng.random_range(0.3..0.9),
                metallic: rng.random_range(0.0..0.5),
                ..default()
            })),
            Transform::from_xyz(x, h / 2.0, z)
                .with_rotation(Quat::from_rotation_y(rng.random_range(0.0..std::f32::consts::TAU))),
            RigidBody::Static,
            Collider::cuboid(w, h, d),
        ));
    }
}
