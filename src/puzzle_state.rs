use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

pub struct PuzzleStatePlugin;

impl Plugin for PuzzleStatePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<StateListener>()
            .insert_resource(PuzzleStateRegistry::default())
            .add_observer(on_state_listener_added);
    }
}

/// Resource: централизованное хранилище boolean состояний
#[derive(Resource, Default)]
pub struct PuzzleStateRegistry {
    states: HashMap<String, bool>,
    watchers: HashMap<String, HashSet<Entity>>,
}

impl PuzzleStateRegistry {
    /// Установить состояние, вернуть true если значение изменилось
    pub fn set_state(&mut self, name: &str, value: bool) -> bool {
        let old_value = self.states.get(name).copied().unwrap_or(false);
        if old_value != value {
            self.states.insert(name.to_string(), value);
            true
        } else {
            false
        }
    }

    /// Получить текущее значение состояния
    pub fn get_state(&self, name: &str) -> bool {
        self.states.get(name).copied().unwrap_or(false)
    }

    /// Зарегистрировать watcher (для оптимизации)
    pub fn register_watcher(&mut self, state_name: &str, entity: Entity) {
        self.watchers
            .entry(state_name.to_string())
            .or_default()
            .insert(entity);
    }

    /// Получить всех watchers для состояния
    pub fn get_watchers(&self, state_name: &str) -> Option<&HashSet<Entity>> {
        self.watchers.get(state_name)
    }
}

/// Event: состояние изменилось
#[derive(Event)]
pub struct StateChanged {
    pub name: String,
    pub old_value: bool,
    pub new_value: bool,
}

/// Component: слушает состояние (на Door, например)
#[derive(Component, Reflect, Debug)]
#[reflect(Component, Default)]
pub struct StateListener {
    pub state_name: String,
    pub invert: bool, // Если true: активироваться когда state=false
}

impl Default for StateListener {
    fn default() -> Self {
        Self {
            state_name: String::new(),
            invert: false,
        }
    }
}

/// Observer: автоматически регистрировать StateListener как watcher при добавлении
fn on_state_listener_added(
    trigger: On<Add, StateListener>,
    q: Query<&StateListener>,
    mut registry: ResMut<PuzzleStateRegistry>,
) {
    let entity = trigger.event_target();
    let Ok(listener) = q.get(entity) else {
        return;
    };
    registry.register_watcher(&listener.state_name, entity);
    info!(
        "Registered StateListener on {entity} for state '{}'",
        listener.state_name
    );
}
