use std::sync::atomic::Ordering;

use avian3d::prelude::*;
use bevy::log::LogPlugin;
use bevy::prelude::*;
use bevy_skein::SkeinPlugin;

use fever_dream::*;

fn main() {
    let mut app = App::new();

    let is_test_inference = std::env::args().any(|a| a == "--test-inference");
    let is_test_puzzle = std::env::args().any(|a| a == "--test-puzzle");

    let default_plugins = DefaultPlugins
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
        });

    // On WASM: force WebGPU backend (not WebGL) so compute shaders & storage buffers work.
    // This enables the GPU bridge for zero-copy style transfer on the web.
    #[cfg(target_arch = "wasm32")]
    let default_plugins = default_plugins.set(bevy::render::RenderPlugin {
        render_creation: bevy::render::settings::WgpuSettings {
            backends: Some(wgpu::Backends::BROWSER_WEBGPU),
            ..default()
        }
        .into(),
        ..default()
    });

    app.add_plugins(default_plugins)
    .add_plugins((
        PhysicsPlugins::default(),
        SkeinPlugin::default(),
        player::PlayerPlugin,
        level::LevelPlugin,
        gpu_bridge::GpuBridgePlugin, // Must be before StyleTransferPlugin (provides SharedBurnDevice)
        style_transfer::StyleTransferPlugin,
        post_process::PostProcessPlugin,
        fever::FeverPlugin,
        triggers::TriggersPlugin,
        world_layer::WorldLayerPlugin,
        skybox::SkyboxPlugin,
        tooltip::TooltipPlugin,
        puzzle_objects::PuzzlePlugin,
        movable_platform::MovablePlatformPlugin,
    ));

    if is_test_puzzle {
        app.insert_resource(level::TestPuzzleMode);
        app.add_systems(Startup, spawn_test_puzzle_level);
    }

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

/// Процедурный тестовый уровень для проверки puzzle системы
/// Запуск: cargo run --release -- --test-puzzle
fn spawn_test_puzzle_level(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    use fever_dream::interaction::*;
    use fever_dream::puzzle_objects::*;
    use fever_dream::world_layer::WorldLayer;

    info!("Spawning test puzzle level");

    // 1. Floor
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(40.0, 40.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.25, 0.2),
            perceptual_roughness: 0.9,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
        RigidBody::Static,
        Collider::cuboid(40.0, 0.01, 40.0),
        WorldLayer {
            world_0: true,
            world_1: true,
            ..default()
        },
    ));

    // 2. Grabbable Cube #1 (красный) - В WORLD_1
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.1, 0.1),
            perceptual_roughness: 0.5,
            metallic: 0.1,
            ..default()
        })),
        Transform::from_xyz(-3.0, 1.5, -8.0),
        RigidBody::Dynamic,
        Collider::cuboid(1.0, 1.0, 1.0),
        WeightedCube { mass: 10.0 },
        WorldLayer {
            world_0: false,
            world_1: true,
            ..default()
        },
    ));

    // 3. Grabbable Cube #2 (синий) - В WORLD_1
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.1, 0.9),
            perceptual_roughness: 0.5,
            metallic: 0.1,
            ..default()
        })),
        Transform::from_xyz(3.0, 1.5, -8.0),
        RigidBody::Dynamic,
        Collider::cuboid(1.0, 1.0, 1.0),
        WeightedCube { mass: 10.0 },
        WorldLayer {
            world_0: false,
            world_1: true,
            ..default()
        },
    ));

    // 4. Pressure Plate #1 (left, жёлтая) - В WORLD_0
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(1.0, 0.2))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.9, 0.1),
            perceptual_roughness: 0.7,
            ..default()
        })),
        Transform::from_xyz(-4.0, 0.1, 0.0),
        PressurePlate {
            state_name: "button1".to_string(),
            trigger_radius: 1.2,
            require_cube: true,
            stay_pressed: false,
        },
        WorldLayer {
            world_0: true,
            world_1: false,
            ..default()
        },
    ));

    // 5. Pressure Plate #2 (right, оранжевая) - В WORLD_0
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(1.0, 0.2))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.5, 0.1),
            perceptual_roughness: 0.7,
            ..default()
        })),
        Transform::from_xyz(4.0, 0.1, 0.0),
        PressurePlate {
            state_name: "button2".to_string(),
            trigger_radius: 1.2,
            require_cube: true,
            stay_pressed: false,
        },
        WorldLayer {
            world_0: true,
            world_1: false,
            ..default()
        },
    ));

    // 6. AND Gate (invisible logic entity)
    commands.spawn((
        Transform::from_xyz(0.0, 0.0, 0.0),
        Visibility::default(),
        AndNamedState {
            inps: vec!["button1".to_string(), "button2".to_string()],
            out_state: "door_trigger".to_string(),
        },
    ));

    // 7. Door (зелёная, двигается вверх)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(4.0, 4.0, 0.5))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.9, 0.1),
            perceptual_roughness: 0.6,
            metallic: 0.3,
            ..default()
        })),
        Transform::from_xyz(0.0, 2.0, 5.0),
        RigidBody::Kinematic,
        Collider::cuboid(4.0, 4.0, 0.5),
        Door {
            state_name: "door_trigger".to_string(),
            delta_position: Vec3::new(0.0, 5.0, 0.0),
            speed: 2.0,
            start_open: false,
        },
        WorldLayer {
            world_0: true,
            world_1: true,
            ..default()
        },
    ));

    // 8. Walls для визуального контекста
    let wall_positions = [
        (Vec3::new(-10.0, 2.0, 0.0), Vec3::new(0.5, 4.0, 20.0)),
        (Vec3::new(10.0, 2.0, 0.0), Vec3::new(0.5, 4.0, 20.0)),
        (Vec3::new(0.0, 2.0, -10.0), Vec3::new(20.0, 4.0, 0.5)),
    ];

    for (pos, size) in wall_positions {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::from_size(size))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.3, 0.3, 0.35),
                perceptual_roughness: 0.8,
                ..default()
            })),
            Transform::from_translation(pos),
            RigidBody::Static,
            Collider::cuboid(size.x, size.y, size.z),
            WorldLayer {
                world_0: true,
                world_1: true,
                ..default()
            },
        ));
    }

    info!("Test puzzle level spawned:");
    info!("  - 2 cubes (red/blue) in WORLD_1");
    info!("  - 2 pressure plates (yellow/orange) in WORLD_0");
    info!("  - 1 AND gate + 1 door (both worlds)");
    info!("");
    info!("Multi-world puzzle instructions:");
    info!("  1. Press E to switch to world_1 (see cubes)");
    info!("  2. Grab red cube with F");
    info!("  3. Press E to switch to world_0 - cube should follow!");
    info!("  4. Drop cube on yellow plate");
    info!("  5. Repeat for blue cube on orange plate");
    info!("  6. Door opens when both plates pressed!");
}
