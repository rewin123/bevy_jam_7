use bevy::prelude::*;
use rand::Rng;

use crate::style_transfer::{CurrentStyle, StyleChannels, StyleSwitch};

#[derive(Resource)]
pub struct FeverTimer {
    pub timer: Timer,
}

pub struct FeverPlugin;

impl Plugin for FeverPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(FeverTimer {
            timer: Timer::from_seconds(15.0, TimerMode::Repeating),
        })
        .add_systems(Update, fever_tick);
    }
}

fn fever_tick(
    time: Res<Time>,
    mut fever: ResMut<FeverTimer>,
    mut style: ResMut<CurrentStyle>,
    channels: Option<Res<StyleChannels>>,
) {
    fever.timer.tick(time.delta());

    if fever.timer.just_finished() {
        // let mut rng = rand::rng();
        // let mut new_idx = rng.random_range(0..style.names.len());
        // while new_idx == style.index && style.names.len() > 1 {
        //     new_idx = rng.random_range(0..style.names.len());
        // }

        // info!(
        //     "FEVER: Style switching {} -> {}",
        //     style.names[style.index], style.names[new_idx]
        // );

        // style.index = new_idx;

        // if let Some(channels) = channels {
        //     let _ = channels.send_switch.try_send(StyleSwitch { index: new_idx });
        // }
    }
}
