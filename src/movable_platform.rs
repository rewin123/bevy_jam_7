use avian3d::prelude::*;
use bevy::{platform::collections::HashMap, prelude::*};

use crate::puzzle_state::{PuzzleStateRegistry, StateListener};

pub const PLATFORM_SPEED: f32 = 3.0;
pub const PLATFORM_FRW_SMOOTH: f32 = 0.5;
const ACCEPT_RADIUS: f32 = 0.1;

pub struct MovablePlatformPlugin;

impl Plugin for MovablePlatformPlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(on_add_config);
        app.add_systems(Last, convert_config);
        app.add_systems(Update, platform_move);
    }
}


#[derive(Component, Reflect, Debug, Default)]
#[reflect(Component, Default)]
pub struct MovablePlatformConfig {
    pub route_points: String,

    pub station_time: f32,
    pub look_forward: bool,
    pub enable_signal: String, // If empty, work always
}

#[derive(Component, Default)]
pub struct MovablePlatform {
    pub route_idx: usize,
    pub route_points: Vec<Entity>,
    pub stop_timer: Option<Timer>,
}

impl MovablePlatform {
    fn get_target(&self) -> Entity {
        self.route_points[self.route_idx]
    }

    fn next_target(&mut self) {
        self.route_idx = (self.route_idx + 1) % self.route_points.len(); 
    }
}


fn on_add_config(
    trigger: On<Add, MovablePlatformConfig>,
    mut commands: Commands,
) {
    commands.entity(trigger.event_target()).insert(RigidBody::Kinematic);
}




fn convert_config(
    mut commands: Commands,
    q_only_configs: Query<(Entity, &MovablePlatformConfig), Without<MovablePlatform>>,
    q_names: Query<(Entity, &Name)>
) {
    if q_only_configs.count() == 0 {
        return;
    }

    let mut name_cache = HashMap::new();
    for (e, n) in &q_names {
        name_cache.insert(n.to_string(), e);
    }

    for (e, config) in &q_only_configs {
        let mut p = MovablePlatform::default();
        // let route_names = config.route_points.split(";");
        for target_name in config.route_points.split(";") {
            p.route_points.push(name_cache[target_name]);
        }

        if config.enable_signal != "" {
            commands.entity(e).insert(StateListener {
                state_name: config.enable_signal.clone(),
                ..Default::default()
            });
        }

        info!("Inited platform with routes: {}/{:?}", &config.route_points, &p.route_points);

        commands.entity(e).insert(p);
    }
}


fn platform_move(
    mut q_platforms: Query<(
        Entity,
        &Transform,
        &mut LinearVelocity,
        &mut MovablePlatform,
        Option<&StateListener>,
        &MovablePlatformConfig,
    )>,
    time: Res<Time>,
    q_targets: Query<&Transform, Without<MovablePlatform>>,
    puzzle: Res<PuzzleStateRegistry>,
) {
    for (e, transform, mut lin_vel, mut platform, state, config) in &mut q_platforms {
        if let Some(state) = state {
            if !puzzle.get_state(&state.state_name) {
                lin_vel.0 = Vec3::ZERO;
                continue;
            }
        }

        let mut kill_timer = false;
        if let Some(timer) = platform.stop_timer.as_mut() {
            timer.tick(time.delta());
            if timer.is_finished() {
                kill_timer = true;
            } else {
                lin_vel.0 = Vec3::ZERO;
                continue; // Wait
            }
        }

        if kill_timer {
            platform.stop_timer = None;
        }

        let Ok(target_transform) = q_targets.get(platform.get_target()) else {
            error!("Can not find platform target");
            lin_vel.0 = Vec3::ZERO;
            continue;
        };

        let diff = target_transform.translation - transform.translation;
        if diff.length() < ACCEPT_RADIUS {
            lin_vel.0 = Vec3::ZERO;
            platform.next_target();
            platform.stop_timer = Some(Timer::from_seconds(config.station_time, TimerMode::Once));
        } else {
            lin_vel.0 = diff.normalize() * PLATFORM_SPEED;
        }
    }
}