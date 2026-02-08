# Bevy Jam #7 — "Extremely Incohesive Fever Dream"
## Двухфазный план для Claude Code: FPS-игра с нейронным стайл-трансфером

> **Джем:** https://itch.io/jam/bevy-jam-7
> **Дедлайн:** 2026-02-16 07:59:59 UTC
> **Тема:** "Extremely Incohesive Fever Dream"
> **Bevy версия:** 0.18 (released Jan 13, 2026)

---

Для python используй uv

## КОНТЕКСТ ПРОЕКТА

Это FPS-игра (от первого лица) для Bevy Jam #7. Ключевая фича — **real-time neural style transfer** как постпроцессинг: каждые 10–30 секунд стиль изображения полностью меняется (candy, mosaic, rain-princess, udnie, pointilism), создавая эффект "лихорадочного сна". Между переключениями — glitch-переходы.

Архитектура: Bevy 0.18 рендерит 3D-сцену → readback с GPU → ONNX-инференс в отдельном потоке → результат отображается как fullscreen quad.

---

## ФАЗА 1: Рабочая игра с ort (ONNX Runtime)

### Шаг 1.1 — Scaffold проекта

```bash
cargo init fever_dream --name fever_dream
cd fever_dream
```

**Cargo.toml:**
```toml
[package]
name = "fever_dream"
version = "0.1.0"
edition = "2024"

[dependencies]
bevy = { version = "0.18", features = ["3d"] }
avian3d = { git = "https://github.com/avianphysics/avian", tag = "v0.6.0-rc.1" }  # физика + коллизии (RC, Bevy 0.18)
bevy_skein = "0.4"                   # Blender → glTF pipeline (опционально, если делаем уровень в Blender)
ort = { version = "2", features = ["download-binaries"] }
ndarray = "0.16"
crossbeam-channel = "0.5"
image = "0.25"
rand = "0.9"

[profile.dev]
opt-level = 1

[profile.dev.package."*"]
opt-level = 3
```

**Ключевые зависимости и ссылки:**
- `bevy 0.18` — https://bevy.org/news/bevy-0-18/ — встроенный `FreeCamera` / `FreeCameraPlugin`
- `avian3d 0.6.0-rc.1` — https://github.com/avianphysics/avian — ECS-driven 3D физика для Bevy 0.18
  - **Релиз:** 7 февраля 2026, pre-release (RC). Если нужна стабильность — откатиться на `avian3d = "0.5"` (crates.io)
  - **Новое в 0.6 vs 0.5:** Collision hooks (фильтрация/модификация контактов — one-way platforms, конвейеры), physics diagnostics (debug UI), per-manifold material properties, reworked contact pairs (меньше аллокаций), faster collisions/spatial queries
  - Документация: https://docs.rs/avian3d (пока на 0.5, API 0.6 совместим с мелкими изменениями)
  - Примеры: https://github.com/avianphysics/avian/tree/v0.6.0-rc.1/crates/avian3d/examples
  - **Нет встроенного character controller** — нужно писать свой (kinematic capsule + raycast для ground check)
  - **⚠️ Если сторонние crates (bevy_ahoy, bevy-tnua-avian3d) зависят от avian3d 0.5**, нужен `[patch.crates-io]`:
    ```toml
    [patch.crates-io]
    avian3d = { git = "https://github.com/avianphysics/avian", tag = "v0.6.0-rc.1" }
    ```

**Создание уровней — два варианта:**

**Вариант A: Процедурная сцена (рекомендуется для быстрого старта)**
Генерируем уровень из кода — кубы, сферы, цилиндры с avian3d коллайдерами. Занимает 0 минут на setup, сразу работает. Для fever dream эстетики — рандомные цвета, размеры, позиции объектов.

**Вариант B: Blender + bevy_skein (для красивого уровня)**
Моделируем уровень в Blender, добавляем Bevy компоненты (RigidBody, Collider) через Skein addon, экспортируем как .glb.

Skein workflow:
1. `cargo add bevy_skein` + добавить `SkeinPlugin::default()`
2. Запустить Bevy app → Skein поднимает HTTP-сервер (BRP)
3. В Blender: установить Skein addon (drag-and-drop .zip с https://bevyskein.dev/)
4. В Blender: Fetch Registry → появятся все Bevy компоненты (включая avian3d Collider, RigidBody)
5. Моделировать уровень, добавлять компоненты на объекты
6. Export → .glb → загрузить в Bevy через `SceneRoot(asset_server.load("level.glb#Scene0"))`

**Рекомендация для джема:** Начать с Варианта A (процедурка) чтобы сразу видеть стайл-трансфер. Если останется время — сделать нормальный уровень в Blender через Skein.
- `bevy_skein 0.4` — https://bevyskein.dev/ — Blender ↔ Bevy интеграция через glTF extensions
- `ort 2.x` — https://ort.pyke.io/ — ONNX Runtime Rust bindings, автоскачивание бинарников
- ONNX модели style transfer: https://github.com/onnx/models/tree/main/validated/vision/style_transfer/fast_neural_style

### Шаг 1.2 — Скачать ONNX модели

```bash
mkdir -p assets/models
# Прямые ссылки на ONNX Model Zoo (opset 9, float32, вход/выход [1,3,H,W])
curl -L -o assets/models/candy-9.onnx \
  "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/candy-9.onnx"
curl -L -o assets/models/mosaic-9.onnx \
  "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx"
curl -L -o assets/models/rain-princess-9.onnx \
  "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/rain-princess-9.onnx"
curl -L -o assets/models/udnie-9.onnx \
  "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/udnie-9.onnx"
curl -L -o assets/models/pointilism-9.onnx \
  "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/pointilism-9.onnx"
```

**Формат моделей:**
- Вход: `input1` — `[1, 3, H, W]` float32, значения 0–255
- Выход: `output1` — `[1, 3, H, W]` float32, значения 0–255
- Операторы: Conv, InstanceNormalization, Relu, ConvTranspose, Pad, Add
- Все операторы полностью поддержаны в ort

### Шаг 1.3 — Структура файлов

```
fever_dream/
├── Cargo.toml
├── assets/
│   ├── models/          # .onnx файлы (5 стилей)
│   └── levels/           # Blender .glb уровни (если используем Skein)
│       └── fever.glb     # основной уровень
├── src/
│   ├── main.rs          # точка входа, plugin setup
│   ├── player.rs        # FPS контроллер (Bevy 0.18 FreeCamera или свой)
│   ├── level.rs         # процедурная генерация уровня / загрузка .glb через Skein
│   ├── style_transfer.rs # ort инференс pipeline
│   ├── post_process.rs  # readback с GPU + fullscreen quad отображение
│   └── fever.rs         # логика смены стилей, glitch-эффекты, таймеры
└── README.md
```

### Шаг 1.4 — main.rs: Plugin setup

```rust
use bevy::prelude::*;
use avian3d::prelude::*;

mod player;
mod level;
mod style_transfer;
mod post_process;
mod fever;

fn main() // DefaultPlugins + WindowPlugin(1280x720) + PhysicsPlugins + все наши плагины
```

### Шаг 1.5 — player.rs: FPS-контроллер на avian3d

Avian3d **не имеет встроенного character controller** (см. https://docs.rs/avian3d — "you will need to implement it yourself"). Есть два варианта: написать свой на основе avian3d примеров, или использовать стороннюю библиотеку.

**Вариант A: Свой kinematic FPS-контроллер (рекомендуется для джема)**

Основан на avian3d `kinematic_character_3d` example (https://github.com/avianphysics/avian/blob/main/crates/avian3d/examples/kinematic_character_3d/main.rs):

```rust
use avian3d::prelude::*;
use bevy::prelude::*;
use bevy::input::mouse::AccumulatedMouseMotion;

#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct FpsController {
    pub speed: f32,
    pub jump_impulse: f32,
    pub sensitivity: f32,
    pub pitch: f32,
    pub grounded: bool,
}

pub struct PlayerPlugin; // Startup: spawn_player; Update: ground_check → player_movement → player_look

fn spawn_player(mut commands: Commands)
// Player body: (Player, FpsController, RigidBody::Kinematic, Collider::capsule(0.4, 1.0), Transform(0,2,5))
// Camera child: (Camera3d, Transform(0, 0.8, 0))

fn ground_check(mut query: Query<(&Transform, &mut FpsController), With<Player>>, spatial_query: SpatialQuery)
// Raycast вниз 0.55 от центра капсулы → ctrl.grounded

fn player_movement(keys: Res<ButtonInput<KeyCode>>, time: Res<Time>, mut query: Query<(&Transform, &mut LinearVelocity, &FpsController), With<Player>>)
// WASD → forward_flat/right_flat * speed → velocity.x/z; гравитация + Space=jump

fn player_look(accumulated_mouse_motion: Res<AccumulatedMouseMotion>, mut player_q: Query<(&mut Transform, &mut FpsController), With<Player>>, mut camera_q: Query<&mut Transform, (With<Camera3d>, Without<Player>)>)
// Yaw: rotate_y тела игрока; Pitch: clamp камеры ±π/2
```

**⚠️ ВАЖНО:** Код выше — скелет. При реализации:
- `AccumulatedMouseMotion` — проверить точный путь импорта в Bevy 0.18 (может быть `bevy::input::mouse::AccumulatedMouseMotion`)
- `LinearVelocity` на `RigidBody::Kinematic` — проверить работает ли в avian3d 0.6. Если нет — использовать `RigidBody::Dynamic` + `GravityScale(0.0)` + ручная гравитация через `ExternalForce`
- Ground check — `SpatialQuery` API может отличаться, см. https://docs.rs/avian3d/latest/avian3d/spatial_query/

**Вариант B: bevy_ahoy (готовый FPS controller для avian3d)**

https://github.com/janhohenheim/bevy_ahoy — kinematic character controller для Avian 3D с кучей фич (прыжок, приседание, surf, coyote time). Использует `bevy_enhanced_input`. **Но:** требует `[patch.crates-io]` для unreleased avian — может сломать совместимость с другими зависимостями.

```toml
# Если решили использовать ahoy:
[dependencies]
bevy_ahoy = { git = "https://github.com/janhohenheim/bevy_ahoy" }
bevy_enhanced_input = "0.12"  # проверить версию

[patch.crates-io]
avian3d = { git = "https://github.com/avianphysics/avian" }
```

**Вариант C: FreeCamera для быстрого прототипа (fallback, без физики)**

Если avian3d контроллер отнимает много времени — начать с встроенного FreeCamera (нет коллизий, нет гравитации, просто летающая камера):

```rust
use bevy::camera_controllers::{FreeCamera, FreeCameraPlugin};

fn spawn_player(mut commands: Commands)
// (Camera3d, Transform(0,2,5), FreeCamera::default())
```

**Решение:** Начать с Варианта C (FreeCamera) чтобы быстро увидеть сцену и стайл-трансфер. Затем заменить на Вариант A (свой avian3d контроллер) когда pipeline работает. Вариант B (ahoy) — если нужен polish.

### Шаг 1.6 — level.rs: Генерация уровня

**Основной подход — процедурная сцена с avian3d коллайдерами:**

```rust
use avian3d::prelude::*;
use bevy::prelude::*;

pub struct LevelPlugin; // Startup: spawn_level + spawn_lighting

fn spawn_lighting(mut commands: Commands)
// DirectionalLight(illuminance=10000, shadows=true), Transform с euler rotation

fn spawn_test_scene(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>)
// Пол: RigidBody::Static + Collider::half_space(Y) + Plane3d 50x50
// 4 стены: RigidBody::Static + Collider::cuboid(2,3,2) + цветные кубы
// 5 динамических сфер: RigidBody::Dynamic + Collider::sphere(0.5) + рандомные цвета

fn spawn_blender_level(mut commands: Commands, asset_server: Res<AssetServer>)
// SceneRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("levels/fever.glb")))
```

### Шаг 1.7 — style_transfer.rs: ort инференс pipeline

**Ключевая архитектура:** Отдельный std::thread для инференса, связанный с Bevy через crossbeam channels.

```rust
use bevy::prelude::*;
use crossbeam_channel::{Sender, Receiver, bounded};
use ort::{Session, SessionBuilder, GraphOptimizationLevel};
use ndarray::{Array4, ArrayView4};

#[derive(Resource)]
pub struct StyleChannels {
    pub send_frame: Sender<FrameData>,      // main → inference thread
    pub recv_styled: Receiver<StyledFrame>,   // inference thread → main
}

pub struct FrameData { pub pixels: Vec<u8>, pub width: u32, pub height: u32 }  // RGBA, row-major
pub struct StyledFrame { pub pixels: Vec<u8>, pub width: u32, pub height: u32 } // RGBA

#[derive(Resource)]
pub struct CurrentStyle { pub index: usize, pub names: Vec<String> }

pub struct StyleTransferPlugin; // Startup: setup_inference_thread

fn setup_inference_thread(mut commands: Commands)
// bounded channels (capacity 2), загрузка 5 моделей, std::thread::spawn → inference_thread_main
// Ресурсы: StyleChannels, CurrentStyle

fn inference_thread_main(recv: Receiver<FrameData>, send: Sender<StyledFrame>, model_names: &[String])
// Загрузить все Session из assets/models/{name}.onnx с GraphOptimizationLevel::Level3
// Loop: recv frame → RGBA→[1,3,H,W] float32 (0-255) → session.run(inputs!["input1"]) → outputs["output1"] → [1,3,H,W]→RGBA → send
```

**ВАЖНЫЕ ПРИМЕЧАНИЯ по ort API:**
- `ort::inputs!` макрос — создаёт HashMap для Session::run
- `Session::builder()` → `SessionBuilder` → `.commit_from_file(path)` → `Session`
- Выход: `outputs["output_name"].try_extract_tensor::<f32>()`
- Документация API: https://docs.rs/ort

### Шаг 1.8 — post_process.rs: GPU readback + fullscreen отображение

Это самая сложная часть. Нужно:
1. Рендерить сцену в offscreen текстуру (render target)
2. Читать пиксели с GPU → CPU (readback)
3. Отправлять в inference thread
4. Получать стилизованный кадр → обновлять текстуру для отображения

**Подход:** Использовать две камеры:
- Камера 1: рендерит сцену в `Image` (render target)
- Камера 2: отображает fullscreen quad со стилизованной текстурой

```rust
use bevy::prelude::*;
use bevy::render::render_resource::{TextureUsages, TextureFormat, Extent3d, TextureDimension};

pub const INFERENCE_WIDTH: u32 = 256;
pub const INFERENCE_HEIGHT: u32 = 256;

#[derive(Component)]
pub struct SceneCamera;

#[derive(Component)]
pub struct DisplayCamera;

#[derive(Resource)]
pub struct StyleTarget {
    pub render_image: Handle<Image>,   // куда рендерим сцену
    pub display_image: Handle<Image>,  // что показываем (стилизованное)
}

pub struct PostProcessPlugin; // Startup: setup_render_targets; Update: readback_and_send → receive_styled_frame

fn setup_render_targets(mut commands: Commands, mut images: ResMut<Assets<Image>>)
// render_image: Rgba8UnormSrgb, TEXTURE_BINDING | COPY_SRC | RENDER_ATTACHMENT
// display_image: Rgba8UnormSrgb, TEXTURE_BINDING | COPY_DST
// Player camera target = render_image; UI camera + fullscreen quad = display_image

fn readback_and_send(style_target: Res<StyleTarget>, images: Res<Assets<Image>>, channels: Option<Res<StyleChannels>>)
// Читаем Image::data из render_image → send_frame.try_send(FrameData)
// ВНИМАНИЕ: Image::data для render targets НЕ автоматически обновляется из GPU
// Использовать Screenshot API или headless renderer pattern

fn receive_styled_frame(channels: Option<Res<StyleChannels>>, style_target: Res<StyleTarget>, mut images: ResMut<Assets<Image>>)
// recv_styled.try_recv() → image.data = styled.pixels
```

**КРИТИЧЕСКИЕ ЗАМЕТКИ по GPU readback:**
- В Bevy нет простого способа читать render target обратно на CPU каждый кадр
- `Image::data` для render targets НЕ автоматически обновляется из GPU
- Для джема есть два рабочих подхода:
  1. **Screenshot API** (Bevy 0.18): `Screenshot::primary_window()` + observer — проще, но захватывает уже после постпроцессинга
  2. **Headless renderer pattern**: см. `bevy/examples/app/headless_renderer.rs` — использует `RenderGraph` ноду для копирования GPU→Buffer→Channel
- Ссылка на пример: https://github.com/bevyengine/bevy/blob/main/examples/app/headless_renderer.rs
- **Рекомендация для джема:** Начать с Screenshot API. Если не работает для offscreen камеры — взять код из headless_renderer.rs

### Шаг 1.9 — fever.rs: Логика смены стилей

```rust
use bevy::prelude::*;
use rand::Rng;

#[derive(Resource)]
pub struct FeverTimer {
    pub timer: Timer,          // 15 сек repeating
    pub glitch_timer: Timer,   // 0.5 сек once
    pub is_glitching: bool,
}

pub struct FeverPlugin; // insert_resource(FeverTimer) + Update: fever_tick

fn fever_tick(time: Res<Time>, mut fever: ResMut<FeverTimer>, mut style: ResMut<CurrentStyle>)
// timer.tick → random ДРУГОЙ стиль → style.index; glitch_timer для переходного эффекта
```

### Шаг 1.10 — Компиляция и запуск

```bash
cargo build 2>&1 | head -50
cargo run --release
```

---

## ФАЗА 2: Форк wonnx — допиливание недостающих операторов

### Контекст

wonnx (https://github.com/webonnx/wonnx) — ONNX runtime на 100% Rust через wgpu. Заархивирован 2025-05-07. Для fast_neural_style моделей не хватает двух операторов:
- **InstanceNormalization** — нормализация по (N,C) с runtime mean/variance
- **ConvTranspose** — транспонированная свёртка (upsampling)

Преимущество wonnx: использует тот же wgpu что и Bevy → потенциально zero-copy GPU pipeline (без readback на CPU).

### Шаг 2.1 — Форк и структура

```bash
git clone https://github.com/webonnx/wonnx.git
cd wonnx
```

**Структура проекта:**
```
wonnx/
├── wonnx/
│   ├── src/
│   │   ├── compiler.rs      # ГЛАВНЫЙ ФАЙЛ: маршрутизация ONNX op → WGSL шейдер
│   │   ├── sequencer.rs     # Оптимизации и scheduling
│   │   ├── gpu.rs           # wgpu pipeline, buffer management
│   │   ├── lib.rs           # Session API
│   │   └── utils.rs
│   └── templates/           # WGSL шейдеры (compute shaders)
│       ├── endomorphism/
│       │   ├── map.wgsl                    # Abs, Sin, Relu и т.д.
│       │   ├── arithmetic.wgsl             # Add, Mul, Div
│       │   ├── batchnormalization.wgsl     # BatchNorm — ОСНОВА для InstanceNorm
│       │   └── copy.wgsl                   # Reshape, Flatten
│       ├── pool/
│       │   ├── conv.wgsl                   # Conv 2D — ОСНОВА для ConvTranspose
│       │   ├── conv_kernel_1.wgsl
│       │   ├── conv_kernel_3.wgsl
│       │   └── aggregate.wgsl             # MaxPool, AveragePool
│       ├── matrix/                         # MatMul, Gemm
│       └── snippets/                       # Общие WGSL утилиты
├── wonnx-preprocessing/                    # Shape inference, graph optimization
├── wonnx-cli/                              # CLI тулза
└── Cargo.toml
```

### Шаг 2.2 — Реализация InstanceNormalization

**Спецификация ONNX:** https://onnx.ai/onnx/operators/onnx__InstanceNormalization.html

**Математика:**
```
y[n,c,h,w] = scale[c] * (x[n,c,h,w] - mean[n,c]) / sqrt(var[n,c] + epsilon) + bias[c]
где mean[n,c] = avg(x[n,c,:,:]),  var[n,c] = avg((x[n,c,:,:] - mean[n,c])^2)
```

**Входы:**
- input: `[N, C, H, W]` float32
- scale: `[C]` float32
- B (bias): `[C]` float32

**Атрибут:** `epsilon` (default 1e-5)

**Стратегия реализации:**
1. Скопировать `batchnormalization.wgsl` → `instancenormalization.wgsl`
2. BatchNorm использует предвычисленные mean/var — InstanceNorm вычисляет их в рантайме
3. Нужен **двухпроходный** шейдер:
   - **Pass 1 (reduction):** Для каждого (n,c) вычислить sum и sum_sq по всем H×W → workgroup reduction → промежуточный буфер stats[N*C*2]
   - **Pass 2 (normalize):** stats → mean/var → `scale[c] * (x - mean) * inv_std + bias[c]`
4. В `compiler.rs` добавить обработку `"InstanceNormalization"` — два ComputePass с промежуточным буфером
5. В `wonnx-preprocessing/src/shape_inference.rs` — output shape = input shape

### Шаг 2.3 — Реализация ConvTranspose

**Спецификация ONNX:** https://onnx.ai/onnx/operators/onnx__ConvTranspose.html

**Математика (упрощённо для stride=2, fast_neural_style):**
ConvTranspose — обратная операция к Conv. Для stride=s это эквивалентно:
1. Вставить (s-1) нулей между каждым элементом входа
2. Применить обычную свёртку с перевёрнутым ядром

**Для fast_neural_style параметры фиксированные:**
- kernel_shape: [3, 3] или [4, 4]
- strides: [2, 2]
- pads: [1, 1, 1, 1]
- output_padding: [0, 0] или [1, 1]

**Стратегия реализации:**
1. Скопировать `conv.wgsl` → `conv_transpose.wgsl`
2. Для каждого выходного пикселя: `ih = (oh + pad - kh) / stride`, проверить делимость на stride, bounds check
3. Веса: `[c_in, c_out, kH, kW]` (транспонированный layout относительно Conv)
4. В `compiler.rs` — парсить атрибуты: kernel_shape, strides, pads, output_padding, group
5. Output shape: `out_h = stride * (in_h - 1) + kernel_h - 2*pad + output_padding`
6. В shape inference — добавить формулу output shape

### Шаг 2.4 — Тестирование wonnx операторов

Тесты в `wonnx/tests/`:
- `test_instance_normalization` — модельный граф с одним InstanceNorm, сравнить с numpy/PyTorch
- `test_conv_transpose` — stride=2, kernel=3x3, input 2x2 → output 4x4
- `test_candy_style_transfer` — candy-9.onnx, проверить output shape и диапазон значений

### Шаг 2.5 — Интеграция wonnx в Bevy (zero-copy pipeline)

```toml
[dependencies]
wonnx = { path = "../wonnx/wonnx" }
```

**Преимущество:** wonnx использует wgpu → можно шарить `wgpu::Device` с Bevy → нет GPU↔CPU readback. Весь pipeline остаётся на GPU:
1. Bevy рендерит в wgpu текстуру
2. wonnx читает ту же текстуру как input
3. wonnx пишет output в другую текстуру
4. Bevy отображает output текстуру

Это требует доступа к `wgpu::Device` из Bevy render world — доступен через `bevy::render::renderer::RenderDevice`.

---

## СЕКЦИЯ ВЕРИФИКАЦИИ: Как Claude Code проверяет работоспособность

### ВАЖНО: Визуальная верификация через чтение скриншотов

Claude Code умеет читать изображения. На каждом ключевом этапе **делай скриншот и открывай его** чтобы визуально убедиться что всё правильно. Не полагайся только на exit codes и логи — смотри глазами.

Паттерн визуальной проверки:
```bash
# 1. Запустить игру с автоскриншотом
xvfb-run -a -s "-screen 0 1280x720x24" timeout 15 cargo run --release -- --screenshot-and-exit 2>&1

# 2. Открыть скриншот и ПОСМОТРЕТЬ на него
# (Claude Code: используй tool для чтения изображения screenshot-0.png)

# 3. Оценить визуально:
#    - Видна ли 3D сцена? (не чёрный экран, не белый)
#    - Есть ли объекты/геометрия?
#    - Применён ли стайл-трансфер? (стилизация видна — текстуры "масляные"/пиксельные/мозаичные)
#    - Нет ли артефактов рендеринга? (розовые текстуры = missing assets, Z-fighting и т.д.)
```

**Когда читать скриншоты:**
- После шага 1.6 (level.rs): убедиться что сцена рендерится, видны объекты и освещение
- После шага 1.7 (style_transfer.rs): убедиться что ort инференс не крашится
- После шага 1.8 (post_process.rs): убедиться что стилизованный кадр отображается на экране — это самая критичная проверка, здесь чаще всего будут баги
- После шага 1.9 (fever.rs): сделать 2-3 скриншота с интервалом и убедиться что стили разные
- После Фазы 2: сравнить визуально output wonnx vs ort — должны быть похожи

**Паттерн для отладки чёрного экрана:**
Если скриншот чёрный или пустой:
1. Сделать скриншот БЕЗ стайл-трансфера (отключить постпроцессинг) → если сцена видна, проблема в pipeline
2. Сохранить raw output из ort как PNG → если output нормальный, проблема в отображении
3. Проверить что render target camera действительно рендерит (добавить debug UI поверх)

### Проверки

1. **Компиляция:** `cargo check`, `cargo clippy`
2. **ONNX модели:** `ls -la assets/models/*.onnx` — 5 файлов, каждый > 1MB
3. **ort инференс:** unit test — Session::builder → commit_from_file("candy-9.onnx") → run([1,3,256,256]) → assert output shape и диапазон
4. **Bevy smoke test:** MinimalPlugins + ScheduleRunnerPlugin → 2 кадра → AppExit::Success
5. **Screenshot test:** `xvfb-run ... cargo run --release -- --screenshot-and-exit` → проверить PNG, визуально оценить
6. **wonnx тесты (Фаза 2):** `cargo test` в wonnx/, тесты новых операторов, визуальное сравнение wonnx vs ort
7. **E2E style transfer:** градиентный input → candy-9 → assert avg_diff > 10.0, сохранить PNG для визуальной проверки

**Screenshot test** требует CLI аргумент `--screenshot-and-exit` в main.rs:
```rust
fn auto_screenshot(mut commands: Commands, mut frame_count: Local<u32>, mut exit: EventWriter<AppExit>)
// frame 60: Screenshot::primary_window() + observe(save_to_disk("screenshot-0.png"))
// frame 65: AppExit::Success
```

**Что должно быть видно на скриншоте (по этапам):**
| Этап | Ожидаемая картинка |
|------|--------------------|
| После level.rs | 3D сцена: цветные кубы на зелёном полу, направленный свет, тени |
| После post_process.rs (без стиля) | Та же сцена, но возможно с меньшим разрешением (256×256 upscaled) |
| После style_transfer + post_process | Сцена в стиле "масляной живописи" / мозаики / пуантилизма — объекты узнаваемы, но текстуры художественные |
| Финальная версия | Стилизованная сцена + возможно UI элементы (название стиля, debug info) |

---

## ЧЕКЛИСТ

### Фаза 1 (для джема):
- [x] Проект инициализирован, компилируется
- [ ] avian3d PhysicsPlugins подключён, физика работает (объекты падают) — отложено, используется FreeCamera-подобный контроллер
- [x] ONNX модели скачаны (5 стилей)
- [x] ort unit test проходит (candy-9 инференс) — проверено через запуск приложения
- [x] FPS контроллер работает (FreeCamera-стиль с WASD + mouselook)
- [ ] Коллайдеры на сцене — отложено (нет avian3d, используется noclip)
- [x] 3D сцена загружается (процедурная генерация)
- [x] GPU readback pipeline работает (Screenshot API → CPU → inference thread)
- [x] Style transfer thread запускается и производит output
- [x] Fullscreen quad отображает стилизованный кадр
- [x] Смена стилей по таймеру работает (15 сек)
- [x] Screenshot тест проходит (--screenshot-and-exit)
- [ ] README.md написан (правила джема требуют)

### Фаза 2 (после джема):
- [ ] wonnx форк, компиляция
- [ ] InstanceNormalization WGSL шейдер написан
- [ ] InstanceNormalization Rust glue в compiler.rs
- [ ] ConvTranspose WGSL шейдер написан
- [ ] ConvTranspose Rust glue в compiler.rs
- [ ] Shape inference для обоих операторов
- [ ] Unit тесты для новых операторов
- [ ] candy-9.onnx проходит через wonnx
- [ ] Все 5 моделей проходят через wonnx
- [ ] Интеграция wonnx в Bevy (замена ort)
- [ ] Zero-copy GPU pipeline (без readback)

---

## ССЫЛКИ

| Ресурс | URL |
|--------|-----|
| Bevy Jam #7 | https://itch.io/jam/bevy-jam-7 |
| Bevy 0.18 Release Notes | https://bevy.org/news/bevy-0-18/ |
| Bevy 0.18 FreeCamera | https://bevy.org/news/bevy-0-18/#first-party-camera-controllers |
| bevy_skein | https://bevyskein.dev/ |
| bevy_flycam | https://github.com/sburris0/bevy_flycam |
| ort crate | https://ort.pyke.io/ |
| ort API docs | https://docs.rs/ort |
| ONNX Style Transfer Models | https://github.com/onnx/models/tree/main/validated/vision/style_transfer/fast_neural_style |
| candy-9.onnx (direct) | https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/candy-9.onnx |
| mosaic-9.onnx (direct) | https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx |
| rain-princess-9.onnx (direct) | https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/rain-princess-9.onnx |
| udnie-9.onnx (direct) | https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/udnie-9.onnx |
| pointilism-9.onnx (direct) | https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/pointilism-9.onnx |
| wonnx (archived) | https://github.com/webonnx/wonnx |
| wonnx compiler.rs | https://github.com/webonnx/wonnx/blob/master/wonnx/src/compiler.rs |
| wonnx WGSL templates | https://github.com/webonnx/wonnx/tree/master/wonnx/templates |
| BatchNorm WGSL (reference) | https://github.com/webonnx/wonnx/blob/master/wonnx/templates/endomorphism/batchnormalization.wgsl |
| Conv WGSL (reference) | https://github.com/webonnx/wonnx/blob/master/wonnx/templates/pool/conv.wgsl |
| ONNX InstanceNormalization spec | https://onnx.ai/onnx/operators/onnx__InstanceNormalization.html |
| ONNX ConvTranspose spec | https://onnx.ai/onnx/operators/onnx__ConvTranspose.html |
| Bevy Screenshot API | https://bevy.org/examples/window/screenshot/ |
| Bevy Headless Renderer example | https://github.com/bevyengine/bevy/blob/main/examples/app/headless_renderer.rs |
| Bevy Post-Processing example | https://bevy.org/examples/shaders/custom-post-processing/ |
| Bevy EasyScreenshotPlugin (0.18) | https://bevy.org/news/bevy-0-18/#easy-screenshots--screen-recording |
| bevy_github_ci_template | https://github.com/bevyengine/bevy_github_ci_template |
