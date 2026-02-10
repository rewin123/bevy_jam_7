use bevy::{ecs::entity_disabling::Disabled, prelude::*};




pub struct WorldLayerPlugin;

impl Plugin for WorldLayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<ChangeWorldLayer>();
        
        app.add_systems(Last, reactivate_entities);


    }
}


#[derive(Component, Reflect, Debug, Default)]
#[reflect(Component, Default)]
pub struct WorldLayer {
    pub layer: i32
}


#[derive(Message)]
pub struct ChangeWorldLayer(pub i32);


fn reactivate_entities(
    mut trigger: MessageReader<ChangeWorldLayer>,
    mut commands: Commands,
    q_disabled: Query<(Entity, &WorldLayer), With<Disabled>>,
    q_enabled: Query<(Entity, &WorldLayer), Without<Disabled>>
) {
    let trigger = trigger.read().last();
    let Some(trigger) =  trigger else {
        return;
    };
    info!("Change world layer to {}", trigger.0);
    for (e, l) in q_disabled.iter().chain(q_enabled.iter()) {
        if l.layer == trigger.0 {
            commands.entity(e).remove_recursive::<Children, Disabled>();
            info!("Enable {}", e);
        } else {
            commands.entity(e).insert_recursive::<Children>(Disabled);
            info!("Disable {}", e);
        }
    }
}