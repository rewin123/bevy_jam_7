use avian3d::prelude::*;
use bevy::app::HierarchyPropagatePlugin;
use bevy::app::Propagate;
use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;

use crate::player::{Player, PlayerCamera};
use crate::inference_common::{CurrentStyle, StyleChannels, StyleSwitch};

/// Total number of parallel worlds supported.
pub const MAX_WORLDS: usize = 8;

pub struct WorldLayerPlugin;

impl Plugin for WorldLayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(HierarchyPropagatePlugin::<RenderLayers>::new(PostUpdate));

        app.add_message::<ChangeWorldLayer>();
        app.add_message::<NextWorld>();

        app.register_type::<WorldLayer>();
        app.insert_resource(ActiveWorld(0));

        app.add_systems(Update, on_world_layer_added);
        app.add_systems(Update, on_next_world);
        app.add_systems(Update, handle_world_switch);
        app.add_systems(Last, propagate_world_layer);
    }
}

/// Blender-friendly component: 8 bool toggles for world membership.
///
/// In Blender/Skein panel you simply check the boxes for which worlds
/// this object should appear in.
#[derive(Component, Reflect, Debug, Clone)]
#[reflect(Component, Default)]
pub struct WorldLayer {
    pub world_0: bool,
    pub world_1: bool,
    pub world_2: bool,
    pub world_3: bool,
    pub world_4: bool,
    pub world_5: bool,
    pub world_6: bool,
    pub world_7: bool,
}

impl Default for WorldLayer {
    fn default() -> Self {
        Self {
            world_0: true,
            world_1: false,
            world_2: false,
            world_3: false,
            world_4: false,
            world_5: false,
            world_6: false,
            world_7: false,
        }
    }
}

impl WorldLayer {
    /// Convert bools to a bitmask (bit 0 = world_0, bit 1 = world_1, ...).
    pub fn to_mask(&self) -> u32 {
        let bools = [
            self.world_0,
            self.world_1,
            self.world_2,
            self.world_3,
            self.world_4,
            self.world_5,
            self.world_6,
            self.world_7,
        ];
        let mut mask = 0u32;
        for (i, &b) in bools.iter().enumerate() {
            if b {
                mask |= 1 << i;
            }
        }
        mask
    }

    /// Convert bools to `RenderLayers`.
    ///
    /// Each world index `i` maps to render layer `i + 1`
    /// (layer 0 is reserved for shared/UI).
    pub fn to_render_layers(&self) -> RenderLayers {
        let mut layers = RenderLayers::none();
        let bools = [
            self.world_0,
            self.world_1,
            self.world_2,
            self.world_3,
            self.world_4,
            self.world_5,
            self.world_6,
            self.world_7,
        ];
        for (i, &b) in bools.iter().enumerate() {
            if b {
                layers = layers.with(i + 1); // +1: layer 0 = shared
            }
        }
        layers
    }

    /// Convert bools to physics `CollisionLayers`.
    ///
    /// Membership bits are shifted by 1 (bit 0 = default layer in avian).
    /// Filters allow interaction with the Player layer (bit 0) + own world layers.
    pub fn to_collision_layers(&self) -> CollisionLayers {
        let mask = self.to_mask();
        // Shift by 1 because avian bit 0 is the default layer
        let membership_bits = mask << 1;
        // Filter: collide with player (bit 0) + own world layers
        let filter_bits = membership_bits | 1; // bit 0 = player/default
        CollisionLayers::from_bits(membership_bits, filter_bits)
    }
}

/// Message to switch the active world. Payload is the world index (0..7).
#[derive(Message)]
pub struct ChangeWorldLayer(pub u32);

/// Tracks which world is currently active.
#[derive(Resource)]
pub struct ActiveWorld(pub u32);

#[derive(Message)]
pub struct NextWorld {
    pub max_world: u32
}

fn on_next_world(
    mut msgs: MessageReader<NextWorld>,
    mut commands: Commands,
    cur_world: Res<ActiveWorld>
) {
    let Some(msg) = msgs.read().last() else {
        return;
    };

    let next = (cur_world.0 + 1) % msg.max_world;
    commands.write_message(ChangeWorldLayer(next));
}

/// When a `WorldLayer` is added (e.g. from glTF/Skein), generate
/// `RenderLayers` + `Propagate` + `CollisionLayers`.
fn on_world_layer_added(
    // trigger: On<Change, WorldLayer>,
    q: Query<(Entity, &WorldLayer), Changed<WorldLayer>>,
    mut commands: Commands,
) {
    // let entity = trigger.event_target();
    // let Ok(wl) = q.get(entity) else { return };

    for (entity, wl) in &q {
        let render_layers = wl.to_render_layers();
        let collision_layers = wl.to_collision_layers();

        info!(
            "WorldLayer on {entity}: mask=0b{:08b}, render={:?}, collision={:?}",
            wl.to_mask(),
            render_layers,
            collision_layers
        );

        commands.entity(entity).insert((
            // Propagate(render_layers),
            collision_layers,
        ));

        commands.entity(entity).insert_recursive::<Children>(render_layers);
    }
}

/// Propagate `WorldLayer` down the hierarchy: if a child lacks its own
/// `WorldLayer`, it inherits the parent's.  Children that already carry
/// a `WorldLayer` keep their own value (propagation stops there).
fn propagate_world_layer(
    roots: Query<(&WorldLayer, &Children), Changed<WorldLayer>>,
    children_q: Query<Option<&Children>>,
    has_wl: Query<(), With<WorldLayer>>,
    mut commands: Commands,
) {
    for (wl, children) in &roots {
        propagate_wl_recursive(wl, children, &children_q, &has_wl, &mut commands);
    }
}

fn propagate_wl_recursive(
    wl: &WorldLayer,
    children: &Children,
    children_q: &Query<Option<&Children>>,
    has_wl: &Query<(), With<WorldLayer>>,
    commands: &mut Commands,
) {
    for child in children.iter() {
        if has_wl.get(child).is_ok() {
            continue;
        }
        commands.entity(child).insert(wl.clone());
        if let Ok(Some(grandchildren)) = children_q.get(child) {
            propagate_wl_recursive(wl, grandchildren, children_q, has_wl, commands);
        }
    }
}

/// O(1) world switch: only mutate the camera RenderLayers + player CollisionLayers.
fn handle_world_switch(
    mut messages: MessageReader<ChangeWorldLayer>,
    mut active: ResMut<ActiveWorld>,
    mut camera_q: Query<&mut RenderLayers, With<PlayerCamera>>,
    mut player_q: Query<&mut CollisionLayers, With<Player>>,
    mut style: ResMut<CurrentStyle>,
    channels: Option<Res<StyleChannels>>,
) {
    let Some(msg) = messages.read().last() else {
        return;
    };

    let world_idx = msg.0.min((MAX_WORLDS - 1) as u32);
    active.0 = world_idx;

    info!("Switching to world {world_idx}");

    // 1. Camera sees layer 0 (shared) + target world layer
    let target_render = RenderLayers::layer(0).with((world_idx + 1) as usize);
    for mut layers in &mut camera_q {
        *layers = target_render.clone();
    }

    // 2. Player collides with default (bit 0) + target world (bit world_idx+1)
    let physics_bit = 1u32 << (world_idx + 1);
    for mut col in &mut player_q {
        *col = CollisionLayers::from_bits(1, 1 | physics_bit);
    }

    // 3. Switch style transfer to match the world
    let style_idx = world_idx as usize % style.names.len().max(1);
    if !style.names.is_empty() && style_idx != style.index {
        style.index = style_idx;
        info!("Style -> {} ({})", style_idx, style.names[style_idx]);
        if let Some(ref ch) = channels {
            let _ = ch.send_switch.try_send(StyleSwitch { index: style_idx });
        }
    }
}
