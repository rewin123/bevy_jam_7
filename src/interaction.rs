use avian3d::prelude::*;
use bevy::prelude::*;

use crate::player::{Player, PlayerCamera};
use crate::world_layer::{ActiveWorld, WorldLayer};

pub struct InteractionPlugin;

impl Plugin for InteractionPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Grabbable>()
            .add_systems(
                Update,
                (
                    setup_grab_raycast,
                    update_grab_raycast,
                    highlight_grabbable.after(update_grab_raycast),
                    handle_grab_input.after(highlight_grabbable),
                    update_held_object_position.after(handle_grab_input),
                    sync_grabbed_object_world_layer,
                ),
            )
            .add_systems(FixedUpdate, apply_grab_spring_force);
    }
}

/// Component: объект можно схватить
#[derive(Component, Reflect, Debug)]
#[reflect(Component, Default)]
pub struct Grabbable {
    pub grab_distance: f32,   // На каком расстоянии от камеры держать
    pub spring_strength: f32, // Жёсткость пружины
    pub damping: f32,         // Затухание колебаний
    pub max_force: f32,       // Макс сила притяжения
}

impl Default for Grabbable {
    fn default() -> Self {
        Self {
            grab_distance: 2.5,
            spring_strength: 50.0,
            damping: 5.0,
            max_force: 500.0,
        }
    }
}

/// Component: объект сейчас захвачен
#[derive(Component)]
pub struct GrabbedObject {
    pub holder: Entity,        // Player entity
    pub target_position: Vec3, // Куда притягиваем
    pub original_gravity: f32, // Восстановить при release
    pub original_damping: f32, // Восстановить при release
}

/// Component: player держит объект
#[derive(Component)]
pub struct HoldingObject {
    pub held_entity: Entity,
}

/// Component: raycast для определения grabbable объектов
#[derive(Component)]
pub struct GrabRaycast {
    pub max_distance: f32,              // Макс дистанция grab
    pub current_target: Option<Entity>, // На что сейчас смотрим
}

impl Default for GrabRaycast {
    fn default() -> Self {
        Self {
            max_distance: 3.5,
            current_target: None,
        }
    }
}

/// Component: визуальный highlight (emission bump)
#[derive(Component)]
pub struct Highlighted {
    pub original_emissive: Color, // Восстановить при unhighlight
}

fn setup_grab_raycast(
    mut commands: Commands,
    player_q: Query<Entity, Added<Player>>,
) {
    for player in &player_q {
        commands.entity(player).insert(GrabRaycast::default());
        info!("Added GrabRaycast to player {player}");
    }
}

fn update_grab_raycast(
    mut raycast_q: Query<(Entity, &mut GrabRaycast), With<Player>>,
    camera_q: Query<&GlobalTransform, With<PlayerCamera>>,
    grabbable_q: Query<Entity, With<Grabbable>>,
    spatial_query: SpatialQuery,
) {
    let Ok((player_entity, mut raycast)) = raycast_q.single_mut() else {
        return;
    };
    let Ok(cam_transform) = camera_q.single() else {
        return;
    };

    let origin = cam_transform.translation();
    let direction = cam_transform.forward();

    // Exclude player from raycast
    let filter = SpatialQueryFilter::default().with_excluded_entities([player_entity]);

    if let Some(hit) = spatial_query.cast_ray(
        origin,
        direction.into(),
        raycast.max_distance,
        true,
        &filter,
    ) {
        if grabbable_q.contains(hit.entity) {
            raycast.current_target = Some(hit.entity);
            return;
        }
    }

    raycast.current_target = None;
}

fn highlight_grabbable(
    raycast_q: Query<&GrabRaycast, With<Player>>,
    mut commands: Commands,
    highlighted_q: Query<(Entity, &Highlighted)>,
    grabbable_q: Query<(Option<&Children>, Option<&MeshMaterial3d<StandardMaterial>>), With<Grabbable>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    material_q: Query<&MeshMaterial3d<StandardMaterial>>,
) {
    let Ok(raycast) = raycast_q.single() else {
        return;
    };

    // Unhighlight previous target
    for (entity, highlighted) in &highlighted_q {
        if let Ok((children_opt, mat_opt)) = grabbable_q.get(entity) {
            // Try direct material first
            if let Some(mat_handle) = mat_opt {
                if let Some(mat) = materials.get_mut(&mat_handle.0) {
                    mat.emissive = highlighted.original_emissive.into();
                }
            }
            // Then try children
            if let Some(children) = children_opt {
                for child in children.iter() {
                    if let Ok(mat_handle) = material_q.get(child) {
                        if let Some(mat) = materials.get_mut(&mat_handle.0) {
                            mat.emissive = highlighted.original_emissive.into();
                        }
                    }
                }
            }
        }
        commands.entity(entity).remove::<Highlighted>();
    }

    // Highlight new target
    if let Some(target) = raycast.current_target {
        if let Ok((children_opt, mat_opt)) = grabbable_q.get(target) {
            let mut original_emissive = Color::BLACK;
            let mut found_material = false;

            // Try direct material first
            if let Some(mat_handle) = mat_opt {
                if let Some(mat) = materials.get_mut(&mat_handle.0) {
                    original_emissive = mat.emissive.into();
                    mat.emissive = LinearRgba::from(Color::srgb(0.3, 0.3, 0.3));
                    found_material = true;
                }
            }

            // Then try children
            if !found_material {
                if let Some(children) = children_opt {
                    for child in children.iter() {
                        if let Ok(mat_handle) = material_q.get(child) {
                            if let Some(mat) = materials.get_mut(&mat_handle.0) {
                                original_emissive = mat.emissive.into();
                                mat.emissive = LinearRgba::from(Color::srgb(0.3, 0.3, 0.3));
                                found_material = true;
                                break;
                            }
                        }
                    }
                }
            }

            if found_material {
                commands.entity(target).insert(Highlighted {
                    original_emissive,
                });
            }
        }
    }
}

fn handle_grab_input(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    player_q: Query<(Entity, &GrabRaycast, Option<&HoldingObject>), With<Player>>,
    grabbable_q: Query<Option<&GravityScale>, With<Grabbable>>,
    grabbed_q: Query<&GrabbedObject>,
    camera_q: Query<&GlobalTransform, With<PlayerCamera>>,
) {
    if !keys.just_pressed(KeyCode::KeyF) {
        return;
    }

    let Ok((player_entity, raycast, holding)) = player_q.single() else {
        return;
    };
    let Ok(cam_transform) = camera_q.single() else {
        return;
    };

    // Release if holding
    if let Some(holding) = holding {
        let held = holding.held_entity;

        if let Ok(grabbed) = grabbed_q.get(held) {
            commands
                .entity(held)
                .remove::<GrabbedObject>()
                .insert(GravityScale(grabbed.original_gravity))
                .insert(LinearDamping(grabbed.original_damping));
        }

        commands.entity(player_entity).remove::<HoldingObject>();
        info!("Released object {held}");
        return;
    }

    // Grab if looking at grabbable
    if let Some(target) = raycast.current_target {
        if let Ok(gravity_opt) = grabbable_q.get(target) {
            // Use existing GravityScale or default to 1.0
            let original_gravity = gravity_opt.map(|g| g.0).unwrap_or(1.0);
            let target_pos = cam_transform.translation() + cam_transform.forward() * 2.5;

            commands.entity(target).insert(GrabbedObject {
                holder: player_entity,
                target_position: target_pos,
                original_gravity,
                original_damping: 0.1,
            });

            commands
                .entity(target)
                .insert(GravityScale(0.0))
                .insert(LinearDamping(2.0));

            commands.entity(player_entity).insert(HoldingObject {
                held_entity: target,
            });

            info!("Grabbed object {target}");
        }
    }
}

fn update_held_object_position(
    camera_q: Query<&GlobalTransform, With<PlayerCamera>>,
    mut grabbed_q: Query<&mut GrabbedObject>,
    holding_q: Query<&HoldingObject, With<Player>>,
) {
    let Ok(holding) = holding_q.single() else {
        return;
    };
    let Ok(cam_transform) = camera_q.single() else {
        return;
    };
    let Ok(mut grabbed) = grabbed_q.get_mut(holding.held_entity) else {
        return;
    };

    grabbed.target_position = cam_transform.translation() + cam_transform.forward() * 2.5;
}

fn apply_grab_spring_force(
    time: Res<Time>,
    grabbable_q: Query<&Grabbable>,
    mut grabbed_q: Query<(
        Entity,
        &GrabbedObject,
        &Transform,
        &mut LinearVelocity,
        Option<&Mass>,
    )>,
) {
    for (entity, grabbed, transform, mut velocity, mass_opt) in &mut grabbed_q {
        let Ok(grabbable) = grabbable_q.get(entity) else {
            continue;
        };

        let mass = mass_opt.map(|m| m.0).unwrap_or(1.0);

        // Spring force: F = -k * (x - x0) - c * v
        let displacement = transform.translation - grabbed.target_position;
        let spring_force = -grabbable.spring_strength * displacement;
        let damping_force = -grabbable.damping * velocity.0;
        let total_force = spring_force + damping_force;

        // Clamp force
        let force_magnitude = total_force.length();
        let clamped_force = if force_magnitude > grabbable.max_force {
            total_force.normalize() * grabbable.max_force
        } else {
            total_force
        };

        // Применить acceleration (F = ma, a = F/m)
        let acceleration = clamped_force / mass;
        velocity.0 += acceleration * time.delta_secs();
    }
}

fn sync_grabbed_object_world_layer(
    active_world: Res<ActiveWorld>,
    grabbed_q: Query<(Entity, &GrabbedObject)>,
    mut commands: Commands,
) {
    // Only run when ActiveWorld changes
    if !active_world.is_changed() {
        return;
    }

    let grabbed_count = grabbed_q.iter().count();
    if grabbed_count == 0 {
        return;
    }

    // Create WorldLayer based on active world (0 or 1)
    let new_world_layer = match active_world.0 {
        0 => WorldLayer {
            world_0: true,
            world_1: false,
            ..default()
        },
        1 => WorldLayer {
            world_0: false,
            world_1: true,
            ..default()
        },
        _ => WorldLayer {
            world_0: true,
            world_1: false,
            ..default()
        },
    };

    for (entity, _grabbed) in &grabbed_q {
        commands.entity(entity).insert(new_world_layer.clone());
        info!("Synced grabbed object {entity} to world {} (world_0={}, world_1={})",
            active_world.0, new_world_layer.world_0, new_world_layer.world_1);
    }
}
