use avian3d::prelude::*;
use bevy::camera::visibility::RenderLayers;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions};

use crate::level::PlayerStart;
use crate::world_layer::NextWorld;

#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct FpsController {
    pub speed: f32,
    pub sprint_multiplier: f32,
    pub jump_impulse: f32,
    pub sensitivity: f32,
    pub damping: f32,
    pub pitch: f32,
    pub yaw: f32,
}

impl Default for FpsController {
    fn default() -> Self {
        Self {
            speed: 30.0,
            sprint_multiplier: 1.8,
            jump_impulse: 7.0 / 1.5,
            sensitivity: 0.003,
            damping: 0.9,
            pitch: 0.0,
            yaw: 0.0,
        }
    }
}

#[derive(Component)]
pub struct PlayerCamera;

/// Marker added/removed each frame based on ground detection.
#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct Grounded;

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_player).add_systems(
            FixedUpdate,
            (grab_cursor, update_grounded, player_look, player_movement).chain(),
        );

        app.add_systems(
            Update,
            on_player_start
        );
    }
}

fn spawn_player(mut commands: Commands) {
    // Player: membership = bit 0 (player/default), filter = bit 0 + bit 1 (world_0)
    // The filter gets updated by ChangeWorldLayer at runtime.
    commands
        .spawn((
            Player,
            FpsController::default(),
            Transform::from_xyz(0.0, 2.0, 0.0),
            Visibility::default(),
            // Physics
            RigidBody::Kinematic, // Freeze player and wait to level loading, then change to dynamic
            Collider::capsule(0.4, 1.0),
            CollisionLayers::from_bits(1, 1 | (1 << 1)), // member=default, filter=default+world_0
            LockedAxes::ROTATION_LOCKED,
            Friction::new(2.0),
            Restitution::ZERO.with_combine_rule(CoefficientCombine::Min),
            GravityScale(1.0),
            // Ground detection via shape cast
            ShapeCaster::new(
                Collider::capsule(0.4 * 0.99, 0.99),
                Vec3::ZERO,
                Quat::default(),
                Dir3::NEG_Y,
            )
            .with_max_distance(0.4),
        ))
        .with_children(|parent| {
            // Camera: layer 0 (shared/UI) + layer 1 (world_0)
            // Updated by ChangeWorldLayer at runtime.
            parent.spawn((
                PlayerCamera,
                Camera3d::default(),
                Camera::default(),
                Transform::from_xyz(0.0, 0.8, 0.0),
                RenderLayers::layer(0).with(1),
            ));
        });
}

fn grab_cursor(
    mouse: Res<ButtonInput<MouseButton>>,
    key: Res<ButtonInput<KeyCode>>,
    mut cursor_q: Query<&mut CursorOptions>,
) {
    let Ok(mut cursor) = cursor_q.single_mut() else {
        return;
    };
    if mouse.just_pressed(MouseButton::Left) {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    }
    if key.just_pressed(KeyCode::Escape) {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    }
}

fn update_grounded(
    mut commands: Commands,
    query: Query<(Entity, &ShapeHits, &Rotation), With<Player>>,
) {
    let max_slope_angle = Some((30.0 as avian3d::math::Scalar).to_radians());
    for (entity, hits, rotation) in &query {
        // let is_grounded = hits.iter().any(|hit| {
        //     hit.normal2.angle_between(Vec3::Y).abs() <= 0.8 // ~45 degrees
        // });

        let is_grounded = hits.iter().any(|hit| {
            if let Some(angle) = max_slope_angle {
                (rotation * -hit.normal2).angle_between(Vec3::Y).abs() <= angle
            } else {
                true
            }
        });

        if is_grounded {
            commands.entity(entity).insert(Grounded);
        } else {
            commands.entity(entity).remove::<Grounded>();
        }
    }
}

fn player_movement(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<(&FpsController, &mut LinearVelocity, &ShapeHits, Option<&Grounded>), With<Player>>,
    mut commands: Commands,
) {
    if keys.just_pressed(KeyCode::KeyE) {
        commands.write_message(NextWorld {
            max_world: 2
        });
    }


    for (ctrl, mut lin_vel, hits, grounded) in &mut query {
        let is_grounded = grounded.is_some();

        let yaw_rot = Quat::from_rotation_y(ctrl.yaw);

        let mut input_dir = Vec3::ZERO;
        if keys.pressed(KeyCode::KeyW) {
            input_dir.z -= 1.0;
        }
        if keys.pressed(KeyCode::KeyS) {
            input_dir.z += 1.0;
        }
        if keys.pressed(KeyCode::KeyA) {
            input_dir.x -= 1.0;
        }
        if keys.pressed(KeyCode::KeyD) {
            input_dir.x += 1.0;
        }
        let input_dir = input_dir.normalize_or_zero();
        let world_dir = yaw_rot * input_dir;

        let speed = if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
            ctrl.speed * ctrl.sprint_multiplier
        } else {
            ctrl.speed
        };

        lin_vel.x += world_dir.x * speed * time.delta_secs();
        lin_vel.z += world_dir.z * speed * time.delta_secs();

        // Jump
        if keys.just_pressed(KeyCode::Space) && is_grounded {
            lin_vel.y = ctrl.jump_impulse;
        }

        // XZ damping
        if !is_grounded {
            lin_vel.x *= ctrl.damping;
            lin_vel.z *= ctrl.damping;
        }
    }
}

fn player_look(
    accumulated_mouse_motion: Res<AccumulatedMouseMotion>,
    mut player_q: Query<(&mut Transform, &mut FpsController), With<Player>>,
    mut camera_q: Query<&mut Transform, (With<PlayerCamera>, Without<Player>)>,
) {
    let delta = accumulated_mouse_motion.delta;
    if delta == Vec2::ZERO {
        return;
    }

    let Ok((mut player_transform, mut ctrl)) = player_q.single_mut() else {
        return;
    };

    ctrl.yaw -= delta.x * ctrl.sensitivity;
    ctrl.pitch = (ctrl.pitch - delta.y * ctrl.sensitivity)
        .clamp(-std::f32::consts::FRAC_PI_2 + 0.01, std::f32::consts::FRAC_PI_2 - 0.01);

    // Yaw rotates the player body (physics won't fight this since rotation is locked)
    player_transform.rotation = Quat::from_rotation_y(ctrl.yaw);

    let pitch = ctrl.pitch;
    for mut cam_transform in &mut camera_q {
        cam_transform.rotation = Quat::from_rotation_x(pitch);
    }
}


fn on_player_start(
    mut q_players: Query<&mut Transform, With<Player>>,
    q_starts: Query<&Transform, (Added<PlayerStart>, Without<Player>)>
) {

    if q_starts.count() == 0 {
        return;
    }

    let Ok(st) = q_starts.single() else {
        error!("Multiple player starts");
        return;
    };

    for mut pt in &mut q_players {
      pt.translation = st.translation;  
    }
}