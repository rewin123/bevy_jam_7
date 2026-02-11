use avian3d::prelude::*;
use bevy::app::Propagate;
use bevy::prelude::*;

use crate::player::{Player, PlayerCamera};
use crate::world_layer::{ActiveWorld, WorldLayer};

pub struct GrabPlugin;

impl Plugin for GrabPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                try_grab_or_release,
                update_grabbed_position,
                sync_grabbed_world_layer,
            )
                .chain(),
        );
    }
}

/// Marker on an entity currently grabbed by the player.
#[derive(Component)]
pub struct Grabbed {
    /// The player entity holding this object.
    pub holder: Entity,
    /// Distance in front of the camera to hold the object.
    pub hold_distance: f32,
}

const GRAB_RANGE: f32 = 5.0;
const HOLD_DISTANCE: f32 = 3.0;
const HOLD_LERP_SPEED: f32 = 15.0;

fn try_grab_or_release(
    keys: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    spatial_query: SpatialQuery,
    player_q: Query<Entity, With<Player>>,
    camera_q: Query<&GlobalTransform, With<PlayerCamera>>,
    grabbed_q: Query<(Entity, &Grabbed)>,
    rigid_body_q: Query<&RigidBody, Without<Grabbed>>,
    active_world: Res<ActiveWorld>,
) {
    if !keys.just_pressed(KeyCode::KeyF) {
        return;
    }

    let Ok(player_entity) = player_q.single() else {
        return;
    };
    let Ok(cam_gt) = camera_q.single() else {
        return;
    };

    // If already holding something, release it
    if let Some((held_entity, _)) = grabbed_q.iter().find(|(_, g)| g.holder == player_entity) {
        commands.entity(held_entity).remove::<Grabbed>();
        commands.entity(held_entity).insert(RigidBody::Dynamic);
        info!("Released entity {held_entity}");
        return;
    }

    // Try to grab: raycast from camera
    let ray_origin = cam_gt.translation();
    let ray_dir = cam_gt.forward();

    if let Some(hit) = spatial_query.cast_ray(
        ray_origin,
        ray_dir,
        GRAB_RANGE,
        true,
        &SpatialQueryFilter::default(),
    ) {
        let target = hit.entity;

        // Only grab dynamic bodies
        if !rigid_body_q
            .get(target)
            .is_ok_and(|rb| *rb == RigidBody::Dynamic)
        {
            return;
        }

        info!("Grabbed entity {target}");

        // Switch to kinematic so we control position directly
        commands.entity(target).insert((
            Grabbed {
                holder: player_entity,
                hold_distance: HOLD_DISTANCE,
            },
            RigidBody::Kinematic,
        ));

        // Sync WorldLayer to the player's active world
        let new_wl = WorldLayer::from_world_index(active_world.0);
        let render_layers = new_wl.to_render_layers();
        let collision_layers = new_wl.to_collision_layers();
        commands
            .entity(target)
            .insert((new_wl, Propagate(render_layers), collision_layers));
    }
}

/// Smoothly move grabbed objects to the hold position in front of the camera.
fn update_grabbed_position(
    time: Res<Time>,
    camera_q: Query<&GlobalTransform, With<PlayerCamera>>,
    mut grabbed_q: Query<(&Grabbed, &mut Transform)>,
) {
    let Ok(cam_gt) = camera_q.single() else {
        return;
    };

    let cam_pos = cam_gt.translation();
    let cam_forward = cam_gt.forward();

    for (grabbed, mut transform) in &mut grabbed_q {
        let target_pos = cam_pos + *cam_forward * grabbed.hold_distance;
        let t = (HOLD_LERP_SPEED * time.delta_secs()).min(1.0);
        transform.translation = transform.translation.lerp(target_pos, t);
    }
}

/// When the active world changes, update all grabbed objects to match.
///
/// This is the key architectural fix: instead of modifying `handle_world_switch`
/// (which would create a circular dependency grab↔world_layer), we react to
/// `ActiveWorld` resource changes here. This keeps the O(1) world-switch design
/// clean and avoids coupling.
fn sync_grabbed_world_layer(
    active_world: Res<ActiveWorld>,
    mut commands: Commands,
    grabbed_q: Query<Entity, With<Grabbed>>,
) {
    if !active_world.is_changed() {
        return;
    }

    let new_wl = WorldLayer::from_world_index(active_world.0);
    let render_layers = new_wl.to_render_layers();
    let collision_layers = new_wl.to_collision_layers();

    for entity in &grabbed_q {
        commands
            .entity(entity)
            .insert((new_wl.clone(), Propagate(render_layers.clone()), collision_layers));
        info!(
            "Synced grabbed entity {entity} to world {}",
            active_world.0
        );
    }
}
