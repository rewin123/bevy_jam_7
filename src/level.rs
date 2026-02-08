use bevy::prelude::*;
use rand::Rng;

pub struct LevelPlugin;

impl Plugin for LevelPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (spawn_level, spawn_lighting));
    }
}

fn spawn_lighting(mut commands: Commands) {
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.5, 0.0)),
    ));

    // Ambient light
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.6, 0.5, 0.7),
        brightness: 200.0,
        affects_lightmapped_meshes: true,
    });
}

fn spawn_level(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut rng = rand::rng();

    // Floor
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(80.0, 80.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.15, 0.25, 0.15),
            perceptual_roughness: 0.9,
            ..default()
        })),
    ));

    // Fever dream color palette
    let fever_colors = [
        Color::srgb(0.9, 0.1, 0.3),  // hot pink
        Color::srgb(0.1, 0.2, 0.9),  // deep blue
        Color::srgb(0.9, 0.8, 0.1),  // yellow
        Color::srgb(0.6, 0.1, 0.9),  // purple
        Color::srgb(0.1, 0.9, 0.5),  // green
        Color::srgb(0.9, 0.4, 0.1),  // orange
        Color::srgb(0.2, 0.8, 0.9),  // cyan
        Color::srgb(0.9, 0.2, 0.7),  // magenta
    ];

    // Walls around the arena
    let wall_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.35, 0.5),
        perceptual_roughness: 0.7,
        ..default()
    });
    let wall_mesh = meshes.add(Cuboid::new(80.0, 6.0, 0.5));

    for (pos, rot) in [
        (Vec3::new(0.0, 3.0, -40.0), 0.0),
        (Vec3::new(0.0, 3.0, 40.0), 0.0),
        (Vec3::new(-40.0, 3.0, 0.0), std::f32::consts::FRAC_PI_2),
        (Vec3::new(40.0, 3.0, 0.0), std::f32::consts::FRAC_PI_2),
    ] {
        commands.spawn((
            Mesh3d(wall_mesh.clone()),
            MeshMaterial3d(wall_material.clone()),
            Transform::from_translation(pos).with_rotation(Quat::from_rotation_y(rot)),
        ));
    }

    // Scattered cubes of random sizes and colors
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
        ));
    }

    // Tall pillars
    for _ in 0..8 {
        let color = fever_colors[rng.random_range(0..fever_colors.len())];
        let radius = rng.random_range(0.3..1.0_f32);
        let height = rng.random_range(5.0..15.0_f32);
        let x = rng.random_range(-25.0..25.0_f32);
        let z = rng.random_range(-25.0..25.0_f32);

        commands.spawn((
            Mesh3d(meshes.add(Cylinder::new(radius, height))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                metallic: 0.8,
                perceptual_roughness: 0.2,
                ..default()
            })),
            Transform::from_xyz(x, height / 2.0, z),
        ));
    }

    // Floating spheres (fever dream!)
    for _ in 0..15 {
        let color = fever_colors[rng.random_range(0..fever_colors.len())];
        let radius = rng.random_range(0.5..2.5_f32);
        let x = rng.random_range(-30.0..30.0_f32);
        let y = rng.random_range(3.0..12.0_f32);
        let z = rng.random_range(-30.0..30.0_f32);

        commands.spawn((
            Mesh3d(meshes.add(Sphere::new(radius))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                emissive: color.into(),
                ..default()
            })),
            Transform::from_xyz(x, y, z),
        ));
    }
}
