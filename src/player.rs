use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions};

#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct FpsController {
    pub speed: f32,
    pub sensitivity: f32,
    pub pitch: f32,
    pub yaw: f32,
}

impl Default for FpsController {
    fn default() -> Self {
        Self {
            speed: 10.0,
            sensitivity: 0.003,
            pitch: 0.0,
            yaw: 0.0,
        }
    }
}

#[derive(Component)]
pub struct PlayerCamera;

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_player)
            .add_systems(Update, (grab_cursor, player_movement, player_look).chain());
    }
}

fn spawn_player(mut commands: Commands) {
    // Player entity (invisible, just a transform)
    commands
        .spawn((
            Player,
            FpsController::default(),
            Transform::from_xyz(0.0, 2.0, 8.0),
            Visibility::default(),
        ))
        .with_children(|parent| {
            // Camera as child
            parent.spawn((
                PlayerCamera,
                Camera3d::default(),
                Camera::default(),
                Transform::from_xyz(0.0, 0.8, 0.0),
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

fn player_movement(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<(&mut Transform, &FpsController), With<Player>>,
) {
    for (mut transform, ctrl) in &mut query {
        let forward = transform.forward().as_vec3();
        let right = transform.right().as_vec3();

        let forward_flat = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
        let right_flat = Vec3::new(right.x, 0.0, right.z).normalize_or_zero();

        let mut direction = Vec3::ZERO;
        if keys.pressed(KeyCode::KeyW) {
            direction += forward_flat;
        }
        if keys.pressed(KeyCode::KeyS) {
            direction -= forward_flat;
        }
        if keys.pressed(KeyCode::KeyD) {
            direction += right_flat;
        }
        if keys.pressed(KeyCode::KeyA) {
            direction -= right_flat;
        }
        if keys.pressed(KeyCode::Space) {
            direction += Vec3::Y;
        }
        if keys.pressed(KeyCode::ShiftLeft) {
            direction -= Vec3::Y;
        }

        transform.translation += direction.normalize_or_zero() * ctrl.speed * time.delta_secs();
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

    player_transform.rotation = Quat::from_rotation_y(ctrl.yaw);

    let pitch = ctrl.pitch;
    for mut cam_transform in &mut camera_q {
        cam_transform.rotation = Quat::from_rotation_x(pitch);
    }
}
