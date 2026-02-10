use avian3d::prelude::*;
use bevy::prelude::*;

use crate::{player::Player, world_layer::ChangeWorldLayer};

pub struct TriggersPlugin;


impl Plugin for TriggersPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreUpdate, attach_cube_trigger);
    }
}


/// Trigger zone: place on objects in Blender, configurable via Skein.
/// When the player enters, fires a NamedEvent and optionally switches world.
#[derive(Component, Reflect, Debug)]
#[reflect(Component, Default)]
pub struct CubeTrigger {
    pub size: f32,
    pub trigger_name: String,
    pub destroy_on_trigger: bool,
    /// World index to switch to (0..7). Set to -1 to not switch.
    pub switch_to_world: i32,
}

impl Default for CubeTrigger {
    fn default() -> Self {
        Self {
            size: 2.0,
            trigger_name: String::new(),
            destroy_on_trigger: false,
            switch_to_world: -1,
        }
    }
}


#[derive(Event)]
pub struct NamedEvent {
    pub name : String
}


fn attach_cube_trigger(
    mut commands: Commands,
    q_triggers: Query<(Entity, &CubeTrigger), Without<Collider>>
) {
    for (e, t) in &q_triggers {
        commands.entity(e).insert(
            Collider::cuboid(
                t.size * 2.0,
                t.size * 2.0,
                t.size * 2.0
            )
        ).insert(Sensor)
        .insert(CollisionEventsEnabled)
        .insert(CollidingEntities::default())
        .observe(trigger_collision);

        info!("New cube trigger");
    }
}


fn trigger_collision(
    trigger: On<CollisionStart>,
    mut commands: Commands,
    q_players: Query<&Player>,
    q_triggers: Query<&CubeTrigger>
) {
    info!("Muhahaha {:?} {:?}", trigger.body1, trigger.body2);

    let Some(trigger_body) = trigger.body1 else {
        return;
    };

    let Ok(cube_trigger) = q_triggers.get(trigger_body) else {
        return;
    };

    if let Some(body) = trigger.body2 {
        if q_players.contains(body) {
            commands.trigger(NamedEvent {
                name: cube_trigger.trigger_name.clone(),
            });
            info!("Trigger: {}", cube_trigger.trigger_name);

            if cube_trigger.switch_to_world >= 0 {
                commands.write_message(ChangeWorldLayer(cube_trigger.switch_to_world as u32));
            }
        }
    }
}