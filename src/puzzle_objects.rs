use std::collections::HashSet;

use avian3d::prelude::*;
use bevy::prelude::*;

use crate::interaction::Grabbable;
use crate::player::Player;
use crate::puzzle_state::{PuzzleStateRegistry, StateChanged, StateListener};

pub struct PuzzleObjectsPlugin;

impl Plugin for PuzzleObjectsPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<WeightedCube>()
            .register_type::<PressurePlate>()
            .register_type::<Door>()
            .register_type::<AndNamedState>()
            .register_type::<OrNamedState>()
            .add_systems(
                PreUpdate,
                (
                    setup_weighted_cube,
                    setup_pressure_plate,
                    setup_door,
                    setup_and_gate,
                    setup_or_gate,
                ),
            )
            .add_systems(Update, (pressure_plate_update_state, animate_doors))
            .add_observer(evaluate_and_gates)
            .add_observer(evaluate_or_gates)
            .add_observer(door_react_to_state);
    }
}

/// Component: кубик с весом (для pressure plates)
#[derive(Component, Reflect, Debug)]
#[reflect(Component, Default)]
pub struct WeightedCube {
    pub mass: f32,
}

impl Default for WeightedCube {
    fn default() -> Self {
        Self { mass: 10.0 }
    }
}

fn setup_weighted_cube(mut commands: Commands, cubes: Query<Entity, Added<WeightedCube>>) {
    for entity in &cubes {
        commands.entity(entity).insert(Grabbable::default());
        commands.entity(entity).insert(RigidBody::Dynamic);
        info!("Setup WeightedCube {entity}");
    }
}

/// Component: pressure plate / button
#[derive(Component, Reflect, Debug)]
#[reflect(Component, Default)]
pub struct PressurePlate {
    pub state_name: String,
    pub trigger_radius: f32,
    pub require_cube: bool,
    pub stay_pressed: bool,
}

impl Default for PressurePlate {
    fn default() -> Self {
        Self {
            state_name: String::new(),
            trigger_radius: 1.0,
            require_cube: true,
            stay_pressed: false,
        }
    }
}

/// Runtime state
#[derive(Component)]
pub struct PressurePlateState {
    pub is_pressed: bool,
    pub objects_on_plate: HashSet<Entity>,
}

impl Default for PressurePlateState {
    fn default() -> Self {
        Self {
            is_pressed: false,
            objects_on_plate: HashSet::new(),
        }
    }
}

fn setup_pressure_plate(
    mut commands: Commands,
    plates: Query<(Entity, &PressurePlate), Without<Collider>>,
) {
    for (entity, plate) in &plates {
        commands
            .entity(entity)
            .insert(Collider::cylinder(plate.trigger_radius, 0.2))
            .insert(Sensor)
            .insert(CollisionEventsEnabled)
            .insert(CollidingEntities::default())
            .insert(PressurePlateState::default())
            .observe(on_plate_collision_start)
            .observe(on_plate_collision_end);

        info!("Setup PressurePlate '{}' on {entity}", plate.state_name);
    }
}

fn on_plate_collision_start(
    trigger: On<CollisionStart>,
    mut plate_state_q: Query<(&PressurePlate, &mut PressurePlateState)>,
    player_q: Query<(), With<Player>>,
    cube_q: Query<(), With<WeightedCube>>,
) {
    let Some(plate_entity) = trigger.body1 else {
        return;
    };
    let Some(other_entity) = trigger.body2 else {
        return;
    };

    let Ok((plate, mut state)) = plate_state_q.get_mut(plate_entity) else {
        return;
    };

    let is_valid = if plate.require_cube {
        cube_q.contains(other_entity)
    } else {
        cube_q.contains(other_entity) || player_q.contains(other_entity)
    };

    if is_valid {
        state.objects_on_plate.insert(other_entity);
        info!("Object {other_entity} entered plate {plate_entity}");
    }
}

fn on_plate_collision_end(
    trigger: On<CollisionEnd>,
    mut plate_state_q: Query<(&PressurePlate, &mut PressurePlateState)>,
) {
    let Some(plate_entity) = trigger.body1 else {
        return;
    };
    let Some(other_entity) = trigger.body2 else {
        return;
    };

    let Ok((plate, mut state)) = plate_state_q.get_mut(plate_entity) else {
        return;
    };

    if !plate.stay_pressed {
        state.objects_on_plate.remove(&other_entity);
        info!("Object {other_entity} exited plate {plate_entity}");
    }
}

fn pressure_plate_update_state(
    mut registry: ResMut<PuzzleStateRegistry>,
    mut commands: Commands,
    mut plates: Query<(&PressurePlate, &mut PressurePlateState), Changed<PressurePlateState>>,
) {
    for (plate, mut state) in &mut plates {
        let should_be_pressed = !state.objects_on_plate.is_empty();

        if state.is_pressed != should_be_pressed {
            state.is_pressed = should_be_pressed;

            let old_value = registry.get_state(&plate.state_name);
            if registry.set_state(&plate.state_name, should_be_pressed) {
                commands.trigger(StateChanged {
                    name: plate.state_name.clone(),
                    old_value,
                    new_value: should_be_pressed,
                });
                info!(
                    "PressurePlate '{}' -> {}",
                    plate.state_name, should_be_pressed
                );
            }
        }
    }
}

/// Component: door
#[derive(Component, Reflect, Debug)]
#[reflect(Component, Default)]
pub struct Door {
    pub state_name: String,
    pub delta_position: Vec3,
    pub speed: f32,
    pub start_open: bool,
}

impl Default for Door {
    fn default() -> Self {
        Self {
            state_name: String::new(),
            delta_position: Vec3::ZERO,
            speed: 2.0,
            start_open: false,
        }
    }
}

/// Runtime state
#[derive(Component)]
pub struct DoorState {
    pub is_open: bool,
    pub closed_position: Vec3,
    pub animation_progress: f32,
}

fn setup_door(
    mut commands: Commands,
    doors: Query<(Entity, &Door, &Transform), Added<Door>>,
) {
    for (entity, door, transform) in &doors {
        let closed_pos = transform.translation;
        let initial_progress = if door.start_open { 1.0 } else { 0.0 };

        commands
            .entity(entity)
            .insert(DoorState {
                is_open: door.start_open,
                closed_position: closed_pos,
                animation_progress: initial_progress,
            })
            .insert(StateListener {
                state_name: door.state_name.clone(),
                invert: false,
            })
            .insert(RigidBody::Kinematic);

        info!("Setup Door listening to '{}' on {entity}", door.state_name);
    }
}

fn door_react_to_state(
    trigger: On<StateChanged>,
    mut doors: Query<(&Door, &StateListener, &mut DoorState)>,
) {
    let event = trigger.event();

    for (_door, listener, mut state) in &mut doors {
        if listener.state_name == event.name {
            let should_open = if listener.invert {
                !event.new_value
            } else {
                event.new_value
            };

            if state.is_open != should_open {
                state.is_open = should_open;
                info!(
                    "Door {} -> {}",
                    if should_open { "opening" } else { "closing" },
                    event.name
                );
            }
        }
    }
}

fn animate_doors(time: Res<Time>, mut doors: Query<(&Door, &mut DoorState, &mut Transform)>) {
    for (door, mut state, mut transform) in &mut doors {
        let target_progress = if state.is_open { 1.0 } else { 0.0 };

        if (state.animation_progress - target_progress).abs() < 0.001 {
            state.animation_progress = target_progress;
            continue;
        }

        let delta = if state.is_open {
            time.delta_secs() * door.speed / door.delta_position.length().max(0.001)
        } else {
            -time.delta_secs() * door.speed / door.delta_position.length().max(0.001)
        };

        state.animation_progress = (state.animation_progress + delta).clamp(0.0, 1.0);

        let t = ease_in_out_cubic(state.animation_progress);
        transform.translation = state.closed_position + door.delta_position * t;
    }
}

fn ease_in_out_cubic(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    }
}

/// Component: AND gate для состояний
#[derive(Component, Reflect, Debug)]
#[reflect(Component, Default)]
pub struct AndNamedState {
    pub inps: Vec<String>,
    pub out_state: String,
}

impl Default for AndNamedState {
    fn default() -> Self {
        Self {
            inps: Vec::new(),
            out_state: String::new(),
        }
    }
}

fn setup_and_gate(
    _commands: Commands,
    gates: Query<(Entity, &AndNamedState), Added<AndNamedState>>,
) {
    for (entity, gate) in &gates {
        for inp in &gate.inps {
            info!("AndGate {entity}: listening to '{}'", inp);
        }
        info!(
            "Setup AndNamedState on {entity}: {:?} -> {}",
            gate.inps, gate.out_state
        );
    }
}

fn evaluate_and_gates(
    trigger: On<StateChanged>,
    mut registry: ResMut<PuzzleStateRegistry>,
    mut commands: Commands,
    gates: Query<&AndNamedState>,
) {
    let event = trigger.event();

    for gate in &gates {
        if gate.inps.contains(&event.name) {
            let all_true = gate.inps.iter().all(|inp| registry.get_state(inp));

            let old_value = registry.get_state(&gate.out_state);

            if registry.set_state(&gate.out_state, all_true) {
                commands.trigger(StateChanged {
                    name: gate.out_state.clone(),
                    old_value,
                    new_value: all_true,
                });
                info!(
                    "AndGate: {} -> {} ({})",
                    gate.inps.join(" & "),
                    gate.out_state,
                    all_true
                );
            }
        }
    }
}

/// Component: OR gate для состояний
#[derive(Component, Reflect, Debug)]
#[reflect(Component, Default)]
pub struct OrNamedState {
    pub inps: Vec<String>,
    pub out_state: String,
}

impl Default for OrNamedState {
    fn default() -> Self {
        Self {
            inps: Vec::new(),
            out_state: String::new(),
        }
    }
}

fn setup_or_gate(
    _commands: Commands,
    gates: Query<(Entity, &OrNamedState), Added<OrNamedState>>,
) {
    for (entity, gate) in &gates {
        for inp in &gate.inps {
            info!("OrGate {entity}: listening to '{}'", inp);
        }
        info!(
            "Setup OrNamedState on {entity}: {:?} -> {}",
            gate.inps, gate.out_state
        );
    }
}

fn evaluate_or_gates(
    trigger: On<StateChanged>,
    mut registry: ResMut<PuzzleStateRegistry>,
    mut commands: Commands,
    gates: Query<&OrNamedState>,
) {
    let event = trigger.event();

    for gate in &gates {
        if gate.inps.contains(&event.name) {
            let any_true = gate.inps.iter().any(|inp| registry.get_state(inp));

            let old_value = registry.get_state(&gate.out_state);

            if registry.set_state(&gate.out_state, any_true) {
                commands.trigger(StateChanged {
                    name: gate.out_state.clone(),
                    old_value,
                    new_value: any_true,
                });
                info!(
                    "OrGate: {} -> {} ({})",
                    gate.inps.join(" | "),
                    gate.out_state,
                    any_true
                );
            }
        }
    }
}

// Re-export главного плагина
pub struct PuzzlePlugin;

impl Plugin for PuzzlePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            crate::puzzle_state::PuzzleStatePlugin,
            InteractionPlugin,
            PuzzleObjectsPlugin,
        ));
    }
}

use crate::interaction::InteractionPlugin;
