# Fever Dream — Bevy Jam #7

> **Тема:** "Extremely Incohesive Fever Dream"
> **Дедлайн:** 2026-02-16 07:59:59 UTC
> **Движок:** Bevy 0.18

Для python используй uv

## Что это

FPS-игра с **real-time neural style transfer** как постпроцессинг, создавая эффект лихорадочного сна.

## Архитектура

Всегда поддерживай чистую и легкорасширяемую архитектуру кода.

```
Bevy рендерит 3D → Screenshot API (100ms) → CPU → ort inference thread → styled RGBA → fullscreen UI quad
```

**Модули:**
- `main.rs` — точка входа, plugins, `--screenshot-and-exit`
- `player.rs` — FPS контроллер (WASD + mouselook, avian3d physics, jump)
- `level.rs` — загрузка Blender glTF через Skein + fallback процедурка
- `style_transfer.rs` — ort инференс в отдельном потоке (3 backend'а через features)
- `post_process.rs` — render target, Screenshot capture, fullscreen display, keyboard switch (0-9)
- `fever.rs` — авто-смена стилей каждые 15 сек

**Features (взаимоисключающие):**
- `style-johnson` (default) — 5 фиксированных стилей, 224x224, 0-255
- `style-adain` — произвольные стили из `assets/styles/`, 256x256, 0-1
- `style-microast` — лёгкая модель, произвольные стили, 512x512, 0-1

## Сборка и запуск

```bash
cargo run --release                          # default (Johnson)
cargo run --release --features style-adain   # AdaIN backend
cargo run --release --features style-microast # MicroAST backend
```

**Управление:** WASD — движение, мышь — обзор, ЛКМ — захват курсора, Escape — отпустить, 0 — raw сцена, 1-5 — конкретный стиль.

## Blender + Skein workflow

Аддон: `bevy_skein-0.1.14.zip` в корне проекта.

1. Blender: Install from Disk → `bevy_skein-0.1.14.zip`
2. `cargo run` — Skein поднимает BRP-сервер
3. Blender: Edit → Fetch a Remote Type Registry
4. Моделировать, добавлять компоненты через Skein Panel
5. Export → glTF 2.0 → `assets/levels/Untitled.glb`

Загрузка уровня: `level.rs` → `spawn_blender_level()` → `GltfAssetLabel::Scene(0)`.
Для avian3d коллайдеров: использовать marker component в Blender + observer в Bevy.

## Структура assets

```
assets/
├── levels/Untitled.glb          # Blender уровень (активный)
├── models/                      # ONNX модели
│   ├── candy-9.onnx             # Johnson (5 шт по ~6.5MB)
│   ├── mosaic-9.onnx
│   ├── rain-princess-9.onnx
│   ├── udnie-9.onnx
│   ├── pointilism-9.onnx
│   ├── adain-vgg.onnx           # AdaIN encoder (14MB)
│   ├── adain-decoder.onnx       # AdaIN decoder (14MB)
│   └── microast.onnx (+.data)   # MicroAST (281KB + 1.8MB)
└── styles/                      # Стили для AdaIN/MicroAST
```

## Верификация

```bash
# Screenshot test
xvfb-run -a -s "-screen 0 1280x720x24" timeout 15 cargo run --release -- --screenshot-and-exit
# → screenshot-0.png — открыть и посмотреть визуально
```

На скриншоте должна быть стилизованная 3D-сцена (объекты узнаваемы, текстуры художественные).

## API-справка (Bevy 0.18 + avian3d 0.5 + bevy_skein 0.5)

### Bevy 0.18 — изменения от 0.15-0.17

- **Observers:** `Trigger` переименован в `On`. Первый параметр observer-системы — `On<Event>` или `On<Event, Component>`.
- **Lifecycle events:** `OnAdd` → `Add`, `OnInsert` → `Insert`, `OnRemove` → `Remove`, `OnReplace` → `Replace`.
- **`On::target()`** → **`On::event_target()`** — получить Entity, на котором сработал observer.
- **`MessageWriter`** вместо `EventWriter` для `AppExit`: `exit: MessageWriter<AppExit>`, `exit.write(AppExit::Success)`.
- **`CursorOptions`** — query `Query<&mut CursorOptions>` для управления курсором (grab_mode, visible).
- **`AccumulatedMouseMotion`** — `Res<AccumulatedMouseMotion>`, поле `.delta: Vec2`.
- **`SceneInstanceReady`** — событие для `.observe()` на spawned `SceneRoot`.
- **`GltfAssetLabel::Scene(0).from_asset("path.glb")`** — загрузка glTF-сцен.
- **Component auto-registration:** в Bevy 0.18 компоненты с `Reflect` авто-регистрируются, но лучше явно `app.register_type::<T>()`.

### avian3d 0.5

**Зависимость:** `avian3d = "0.5"` (Bevy 0.18).

**Плагин:** `PhysicsPlugins::default()` — добавить в app.

**RigidBody:**
- `RigidBody::Dynamic` — физический объект
- `RigidBody::Static` — неподвижный (полы, стены)
- `RigidBody::Kinematic` — управляемый кодом

**Collider (конструкторы):**
- `Collider::capsule(radius, length)` — length без полусфер
- `Collider::cuboid(x, y, z)`
- `Collider::sphere(radius)`
- `Collider::cylinder(radius, height)`
- `Collider::trimesh_from_mesh(&Mesh) -> Option<Collider>`
- `Collider::convex_hull_from_mesh(&Mesh)`

**glTF-сцены с коллайдерами:**
```rust
commands.spawn((
    SceneRoot(asset_server.load(...)),
    ColliderConstructorHierarchy::new(ColliderConstructor::TrimeshFromMesh),
    RigidBody::Static,
));
```

**Velocity:** `LinearVelocity(Vec3)`, `AngularVelocity(Vec3)` — прямо задаются.

**Constraints:** `LockedAxes::ROTATION_LOCKED` — блокировать вращение (FPS-персонаж).

**Friction/Restitution:**
```rust
Friction::ZERO.with_combine_rule(CoefficientCombine::Min)
Restitution::ZERO.with_combine_rule(CoefficientCombine::Min)
```

**Гравитация:** `GravityScale(2.0)` — множитель.

**Ground detection (ShapeCaster):**
```rust
ShapeCaster::new(
    Collider::capsule(0.4 * 0.99, 1.0 * 0.99), // чуть меньше основного
    Vec3::ZERO,
    Quat::default(),
    Dir3::NEG_Y,
).with_max_distance(0.2)
```
Результат: `ShapeHits` — итерировать `.iter()`, каждый hit имеет `.normal2`.

**Типы:** avian reexportит `Vec3`/`Quat` из bevy, **не нужно** импортировать `avian3d::math::Vector`/`Quaternion` — использовать обычные `Vec3`, `Quat`.

### bevy_skein 0.5

**Плагин:** `SkeinPlugin::default()` — автоматически поднимает BRP-сервер в debug builds.

**Skein-компоненты (видимые в Blender):**
```rust
#[derive(Component, Reflect, Default, Debug)]
#[reflect(Component, Default)]  // ОБЯЗАТЕЛЬНО #[reflect(Component)]!
pub struct MyMarker;
```
Без `#[reflect(Component)]` Skein **panic** при десериализации.

**Регистрация:** `app.register_type::<MyMarker>()` в Plugin::build.

**glTF extras формат Skein:** `{"skein": [{"fever_dream::level::MyMarker": {}}]}`

**Observer для компонентов из glTF:**
```rust
app.add_observer(|trigger: On<Add, MyMarker>, ...| { ... });
```

**Workflow:** cargo run → Blender: Fetch Remote Type Registry → добавить компоненты → Export glTF.

### Наши маркер-компоненты

- `fever_dream::level::PlayerStart` — пустой объект в Blender, задаёт точку спавна игрока
- `fever_dream::level::AutoMeshCollider` — на мешах, автоматически генерирует trimesh-коллайдер + RigidBody::Static

## TODO

- [ ] Создать полноценный уровень в Blender
- [x] Физика/коллайдеры (avian3d) — базовый контроллер + маркер AutoMeshCollider
- [ ] Glitch-эффекты при смене стилей
- [ ] README.md (требование джема)
- [ ] wonnx GPU pipeline (Фаза 2, после джема)

## Ссылки

| Ресурс | URL |
|--------|-----|
| Bevy Jam #7 | https://itch.io/jam/bevy-jam-7 |
| Bevy 0.18 | https://bevy.org/news/bevy-0-18/ |
| bevy_skein | https://bevyskein.dev/ |
| ort | https://docs.rs/ort |
| ONNX Style Models | https://github.com/onnx/models/tree/main/validated/vision/style_transfer/fast_neural_style |
| avian3d | https://github.com/avianphysics/avian |
