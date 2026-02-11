use std::collections::HashSet;

use avian3d::prelude::*;
use bevy::prelude::*;

use crate::player::Player;

pub struct TooltipPlugin;

impl Plugin for TooltipPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<TooltipTrigger>()
            .insert_resource(TooltipConfig::default())
            .insert_resource(TriggeredTooltips::default())
            .add_systems(PreUpdate, setup_tooltip_trigger)
            .add_systems(Update, (animate_tooltip, cleanup_finished_tooltips).chain())
            .add_observer(spawn_tooltip_ui);
    }
}

/// Tooltip trigger component: place on objects in Blender, configurable via Skein.
/// When the player enters the zone, displays a tooltip UI overlay.
#[derive(Component, Reflect, Debug)]
#[reflect(Component, Default)]
pub struct TooltipTrigger {
    pub text: String,
    pub size: f32,
    pub duration: f32,
    pub trigger_once: bool,
}

impl Default for TooltipTrigger {
    fn default() -> Self {
        Self {
            text: String::new(),
            size: 2.0,
            duration: 0.0,
            trigger_once: true,
        }
    }
}

/// Event triggered when player enters a tooltip zone
#[derive(Event)]
pub struct TooltipTriggered {
    trigger_entity: Entity,
    text: String,
    duration: f32,
}

/// Configuration for tooltip UI appearance and animation
#[derive(Resource)]
pub struct TooltipConfig {
    pub fade_in_duration: f32,
    pub fade_out_duration: f32,
    pub font_size: f32,
    pub padding: UiRect,
    pub max_width: f32,
    pub background_color: Color,
    pub text_color: Color,
}

impl Default for TooltipConfig {
    fn default() -> Self {
        Self {
            fade_in_duration: 0.3,
            fade_out_duration: 0.3,
            font_size: 32.0,
            padding: UiRect::all(Val::Px(24.0)),
            max_width: 600.0,
            background_color: Color::srgba(0.0, 0.0, 0.0, 0.9),
            text_color: Color::srgba(1.0, 1.0, 1.0, 1.0),
        }
    }
}

/// Tracks which triggers have already fired (for trigger_once)
#[derive(Resource, Default)]
pub struct TriggeredTooltips {
    entities: HashSet<Entity>,
}

/// Active tooltip UI component with animation state
#[derive(Component)]
struct ActiveTooltip {
    timer: Timer,
    fade_state: FadeState,
    trigger_entity: Entity,
    trigger_once: bool,
    duration: f32,
}

#[derive(Debug)]
enum FadeState {
    FadeIn(f32),
    Hold,
    FadeOut(f32),
}

/// Marker component for tooltip text entity
#[derive(Component)]
struct TooltipText;

/// Marker component for tooltip background entity
#[derive(Component)]
struct TooltipBackground;

/// Setup system: attaches physics components to tooltip triggers
fn setup_tooltip_trigger(
    mut commands: Commands,
    q_triggers: Query<(Entity, &TooltipTrigger), Without<Collider>>,
) {
    for (e, t) in &q_triggers {
        commands
            .entity(e)
            .insert(Collider::sphere(t.size))
            .insert(Sensor)
            .insert(CollisionEventsEnabled)
            .insert(CollidingEntities::default())
            .observe(on_tooltip_collision)
            .observe(on_tooltip_exit);

        info!("New tooltip trigger: \"{}\"", t.text);
    }
}

/// Observer: fires when player collides with tooltip trigger
fn on_tooltip_collision(
    trigger: On<CollisionStart>,
    mut commands: Commands,
    q_players: Query<&Player>,
    q_triggers: Query<&TooltipTrigger>,
    mut triggered: ResMut<TriggeredTooltips>,
) {
    let Some(trigger_body) = trigger.body1 else {
        return;
    };

    let Ok(tooltip_trigger) = q_triggers.get(trigger_body) else {
        return;
    };

    if let Some(body) = trigger.body2 {
        if q_players.contains(body) {
            // Check if already triggered (for trigger_once)
            if tooltip_trigger.trigger_once && triggered.entities.contains(&trigger_body) {
                return;
            }

            // Fire tooltip event
            commands.trigger(TooltipTriggered {
                trigger_entity: trigger_body,
                text: tooltip_trigger.text.clone(),
                duration: tooltip_trigger.duration,
            });

            info!("Tooltip triggered: {}", tooltip_trigger.text);

            // Mark as triggered
            if tooltip_trigger.trigger_once {
                triggered.entities.insert(trigger_body);
            }
        }
    }
}

/// Observer: fires when player exits tooltip trigger zone
fn on_tooltip_exit(
    trigger: On<CollisionEnd>,
    q_players: Query<&Player>,
    mut q_active: Query<&mut ActiveTooltip>,
) {
    let Some(trigger_body) = trigger.body1 else {
        return;
    };

    if let Some(body) = trigger.body2 {
        if q_players.contains(body) {
            // Find active tooltip for this trigger
            for mut active in &mut q_active {
                if active.trigger_entity == trigger_body && active.duration == 0.0 {
                    // Start fade out for infinite tooltips when player exits
                    if matches!(active.fade_state, FadeState::Hold) {
                        active.fade_state = FadeState::FadeOut(1.0);
                        info!("Player exited infinite tooltip zone, starting fade out");
                    }
                }
            }
        }
    }
}

/// System: spawns tooltip UI when TooltipTriggered event fires
fn spawn_tooltip_ui(
    trigger: On<TooltipTriggered>,
    mut commands: Commands,
    config: Res<TooltipConfig>,
    q_active: Query<&ActiveTooltip>,
    q_triggers: Query<&TooltipTrigger>,
) {
    // Only one active tooltip at a time
    if !q_active.is_empty() {
        return;
    }

    let event = trigger.event();
    let trigger_entity = event.trigger_entity;

    // Get trigger_once from the trigger component
    let trigger_once = q_triggers
        .get(trigger_entity)
        .map(|t| t.trigger_once)
        .unwrap_or(false);

    // Root container - fullscreen overlay with content in bottom third
    let root = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::FlexEnd,
                padding: UiRect::bottom(Val::Px(80.0)),
                ..default()
            },
            GlobalZIndex(11),
        ))
        .id();

    // Background panel
    let panel = commands
        .spawn((
            TooltipBackground,
            Node {
                max_width: Val::Px(config.max_width),
                padding: config.padding,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(config.background_color),
        ))
        .id();

    // Text
    let text_entity = commands
        .spawn((
            TooltipText,
            Text::new(&event.text),
            TextFont {
                font_size: config.font_size,
                ..default()
            },
            TextColor(config.text_color),
            TextLayout {
                justify: Justify::Center,
                ..default()
            },
        ))
        .id();

    // Build hierarchy
    commands.entity(panel).add_child(text_entity);
    commands.entity(root).add_child(panel);

    // Add active tooltip component
    commands.entity(root).insert(ActiveTooltip {
        timer: Timer::from_seconds(event.duration, TimerMode::Once),
        fade_state: FadeState::FadeIn(0.0),
        trigger_entity,
        trigger_once,
        duration: event.duration,
    });

    info!("Spawned tooltip UI: {}", event.text);
}

/// System: animates tooltip fade in/out
fn animate_tooltip(
    time: Res<Time>,
    config: Res<TooltipConfig>,
    mut tooltips: Query<&mut ActiveTooltip>,
    mut backgrounds: Query<&mut BackgroundColor, With<TooltipBackground>>,
    mut texts: Query<&mut TextColor, With<TooltipText>>,
) {
    for mut tooltip in &mut tooltips {
        tooltip.timer.tick(time.delta());

        let alpha = match &mut tooltip.fade_state {
            FadeState::FadeIn(progress) => {
                *progress += time.delta_secs() / config.fade_in_duration;
                if *progress >= 1.0 {
                    tooltip.fade_state = FadeState::Hold;
                    1.0
                } else {
                    *progress
                }
            }
            FadeState::Hold => {
                if tooltip.timer.just_finished() {
                    tooltip.fade_state = FadeState::FadeOut(1.0);
                }
                1.0
            }
            FadeState::FadeOut(progress) => {
                *progress -= time.delta_secs() / config.fade_out_duration;
                progress.max(0.0)
            }
        };

        // Apply alpha to background and text
        // Simply iterate through all backgrounds and texts in the scene
        // (there's only one active tooltip at a time anyway)
        for mut bg in &mut backgrounds {
            let mut color = config.background_color;
            color.set_alpha(color.alpha() * alpha);
            *bg = BackgroundColor(color);
        }

        for mut text_color in &mut texts {
            let mut color = config.text_color;
            color.set_alpha(color.alpha() * alpha);
            *text_color = TextColor(color);
        }
    }
}

/// System: cleans up finished tooltips
fn cleanup_finished_tooltips(
    mut commands: Commands,
    tooltips: Query<(Entity, &ActiveTooltip)>,
) {
    for (entity, tooltip) in &tooltips {
        // Check if fade out is complete
        if let FadeState::FadeOut(progress) = tooltip.fade_state {
            if progress <= 0.0 {
                // Despawn tooltip UI (children will be despawned automatically)
                commands.entity(entity).despawn();
                info!("Cleaned up tooltip UI");

                // Despawn trigger if trigger_once
                if tooltip.trigger_once {
                    commands.entity(tooltip.trigger_entity).despawn();
                    info!("Despawned trigger (trigger_once)");
                }
            }
        }
    }
}
