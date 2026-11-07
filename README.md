# ornament_gen

Генератор гильош-орнаментов документного качества (банкноты, паспорта, сертификаты).
Многослойная композиция, z-buffer плетение, симметрия, деформация по reference,
переменная толщина линий, воспроизводимость через JSON-spec, режим для векторизации.

---

## Что умеет

- **9 сеток:** `warped`, `lens`, `polar`, `spiral`, `spirograph`, `lissajous`, `rose`, `star`, `flow`
- **5 узоров:** `dots`, `crosses`, `contours`, `voronoi_fill`, `rosette`
- **5 эффектов:** `glow`, `inner_glow`, `emboss`, `shadow`, `micro_echo`
- **13 blend-режимов** по W3C Compositing and Blending
- **Плетение G3** (z-buffer) с тенями на пересечениях
- **Симметрия:** осевая, центральная, радиальная
- **Ref-warp:** деформация сетки по яркости reference-изображения
- **Переменная толщина** линий (опционально, модуляция вдоль параметра)
- **10 готовых рецептов** + случайная генерация
- **Глобальные оверрайды** параметров слоёв из CLI
- **Воспроизводимость** через JSON-spec (бит-в-бит)
- **Режимы вывода:** RGB-документ, RGBA-встраивание, векторизационный flat-color
- **Размер:** 1110×756 … A4@300dpi+ (ISO 216, px, DPI, ориентация)
- **RAM-guard** с auto-downgrade quality
- **LRU-кэш** curves для повторных вызовов

---

## Требования

- Python ≥ 3.10
- Зависимости: `numpy>=1.24`, `Pillow>=10.1.0`, `scipy>=1.10`

---

## Установка

```bash
pip install -e .
```

Даёт команду `ornament_gen` и `python -m ornament_gen`.

---

## Быстрый старт

```bash
# A4 альбом, 300 dpi, классическое плетение
ornament_gen --n 3 --size a4 --dpi 300 --quality normal \
    --recipe woven_classic --out ./out

# Прозрачный PNG для встраивания в документ
ornament_gen --n 1 --size a4 --dpi 600 --quality high \
    --recipe document_classic --background none --out ./out

# Векторизационный режим (монохром, без шума, без сжатия)
ornament_gen --n 1 --size a4 --dpi 300 --quality high \
    --recipe woven_classic --flat-color --background none --no-noise \
    --compress-level 0 --out ./out

# Симметрия + деформация по reference
ornament_gen --n 2 --size a4 --dpi 300 --quality normal \
    --recipe rosette_banknote --symmetry radial4 \
    --ref my_gradient.png --ref-fixed --ref-warp 0.7 \
    --out ./out

# Переменная толщина линий
ornament_gen --n 1 --size a4 --dpi 300 --quality normal \
    --recipe woven_classic --variable-thickness \
    --thickness-amplitude 0.4 --thickness-period 5.0 \
    --out ./out

# Глобальная перенастройка слоёв
ornament_gen --n 1 --size a4 --dpi 300 --quality normal \
    --recipe woven_three --layer-opacity 0.75 --layer-blend multiply \
    --weave-z-period 6.0 --weave-z-phase 1.57 \
    --out ./out
```

Результат: `00000.png`, `00001.png`, …, `metadata.jsonl`, опционально `preview.png`.

---

## CLI — все флаги

### Батч и вывод

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--n N` | `1` | Количество изображений (1..50) |
| `--out DIR` | `ornament_out` | Директория результата |
| `--seed INT` | `0` | Базовый seed (влияет на все слои) |
| `--preview N` | `0` | Превью-сетка из первых N (0=выкл) |
| `--compress-level 0..9` | `0` | PNG сжатие (0 = без сжатия) |

### Размер

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--size` | — | `a4`, `a5`, `a6`, `letter`, `passport`, `id_card` |
| `--width` / `--height` | — | Размер в пикселях |
| `--dpi 150..600` | `300` | Разрешение (используется с `--size`) |
| `--orientation` | `landscape` | `landscape` \| `portrait` |

Размер: `round(mm / 25.4 × dpi)`, ориентация применяется после. Минимум `1110×756` (в любой ориентации). `--size` приоритетнее `--width/--height`.

### Качество

| Флаг | По умолчанию | ss | Точек/кривая |
|---|---|---|---|
| `--quality draft` | | 2 | 1500 |
| `--quality normal` | | 3 | 3000 |
| `--quality high` | ✓ | 4 | 4000 |
| `--quality max` | | 4 | 6000 |

`--noise FLOAT` (по умолчанию `0.0`, для документа рекомендуется оставить 0), `--no-noise` — синоним.

### Стиль

| Флаг | Описание |
|---|---|
| `--recipe NAME` | Один из 10 пресетов (см. ниже) |
| `--random-layers N` | Случайная композиция из N слоёв (несовместим с `--recipe`) |
| `--layers-spec FILE` | Воспроизвести из JSON-spec (генерирует 1 изображение) |
| `--dump-spec FILE` | Сохранить spec первого изображения |

### Плетение

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--weave` | из рецепта | `none` \| `zbuffer` |
| `--weave-shadow` | из рецепта | Сила тени на пересечениях `[0,1]` |
| `--weave-z-preset` | — | Переопределить z-пресет слоёв без своего z_preset: `sine` \| `checkerboard` \| `saw` \| `constant` |
| `--weave-z-period FLOAT` | из рецепта | Глобально переопределить `z_period` всех слоёв |
| `--weave-z-phase FLOAT` | из рецепта | Глобально переопределить `z_phase` всех слоёв |

### Оверрайды слоёв

| Флаг | Описание |
|---|---|
| `--layer-opacity FLOAT` | Глобально переопределить `opacity` всех слоёв `[0,1]` |
| `--layer-blend MODE` | Глобально переопределить `blend` всех слоёв (любой из 13) |

Применяются ко всем слоям после загрузки рецепта, до генерации. В `metadata.jsonl` попадают уже переопределённые значения.

### Переменная толщина

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--variable-thickness` | off | Включить модуляцию толщины |
| `--thickness-base FLOAT` | из рецепта | Множитель базовой толщины |
| `--thickness-amplitude FLOAT` | из рецепта | Амплитуда модуляции (0 = без модуляции) |
| `--thickness-period FLOAT` | из рецепта | Период модуляции вдоль параметра |
| `--thickness-phase FLOAT` | из рецепта | Фаза модуляции |

`--thickness-*` требуют `--variable-thickness`, иначе `ap.error`.

**Формула:** `w(t) = base * width * (1 + amplitude * sin(2π * period * t + phase))`, где `t ∈ [0,1]` вдоль кривой. При `--variable-thickness` все слои принудительно получают `thickness.kind = "modulated"`, и `compose._expand_with_thickness` режет кривую на 16 сегментов.

### Цвет и фон

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--ref PATH` | — | Изображение или папка-референс для цвета |
| `--ref-fixed` | off | Без случайного кропа/отражений reference |
| `--ref-blur FLOAT` | `0.004` | Сглаживание цветового поля (доля от размера) |
| `--ref-warp FLOAT` | `0.0` | Деформация сетки по яркости reference (`0.3–1.2` разумно) |
| `--ref-smooth FLOAT` | `0.03` | σ сглаживания для ref-warp |
| `--background` | `paper` | `paper` \| `white` \| `black` \| `none` (RGBA) |
| `--flat-color` | off | Монохромный чёрный (для векторизации) |

### Симметрия

| Флаг | Описание |
|---|---|
| `--symmetry` | `none` \| `vertical` \| `horizontal` \| `central` \| `radialN` (N ≥ 2) |

Применяется после блендинга слоёв, до смешивания с фоном.

### RAM и кэш

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--max-ram FLOAT` | `4096` | Лимит пиковой RAM (MB); при превышении quality понижается (`<=0` — выкл) |
| `--no-cache` | off | Отключить LRU-кэш curves |

---

## Рецепты

| Рецепт | Характер |
|---|---|
| `woven_classic` | 2 слоя warped + sine-плетение, тени |
| `woven_three` | 3 слоя spirograph, плетение |
| `rosette_banknote` | Розетка + Лиссажу + полярная паутина |
| `multiscale` | Warped + spirograph + точки, разные масштабы |
| `spiral_moire` | Две спирали (multiply/screen), без плетения |
| `flow_organic` | Два flow-слоя, органические линии |
| `lissajous_web` | Два Лиссажу с checkerboard-плетением |
| `star_geometric` | Звезда `{8/3}` + полярная паутина |
| `embossed` | Warped с emboss + shadow |
| `document_classic` | Polar + розетка + contours — «банкнотный» |

---

## Размеры и качество — практические заметки

- **A4 @ 300 dpi** = 3508×2480 (ландшафт) / 2480×3508 (портрет)
- При `--quality high` (ss=4) A4 превышает жёсткий лимит SS-пикселей (`120M`) → `ValueError`. Для A4 и выше используйте `--quality normal` (ss=3) или `--max-ram` для auto-downgrade.
- Маленькие форматы (`1110×756`) работают на любом quality, включая `max`.

---

## Воспроизводимость через JSON-spec

Каждый прогон пишет `metadata.jsonl` (по одной строке на изображение) с полным spec.
Для воспроизведения одного изображения:

```bash
# Сохранить spec
ornament_gen --n 1 --size a4 --dpi 300 --quality normal \
    --recipe woven_three --seed 42 --ref ref.png --ref-fixed --ref-warp 0.6 \
    --dump-spec spec.json --out ./a

# Воспроизвести (бит-в-бит)
ornament_gen --layers-spec spec.json --out ./b

sha256sum a/00000.png b/00000.png   # должны совпасть
```

Spec включает: размер, seed, quality, симметрию, ref-warp, цвет, weave, все слои с эффектами, толщинами, z-параметрами.

**Ограничения:**
- `--layers-spec` генерирует всегда **1** изображение
- `--symmetry`, `--recipe`, `--quality` при `--layers-spec` игнорируются (spec авторитет)
- `--seed != 0` переопределяет seed из spec
- Reference подгружается из `spec.color.path` (файл должен существовать)
- `--layer-opacity`, `--layer-blend`, `--weave-z-period`, `--weave-z-phase`, `--thickness-*` при `--layers-spec` **игнорируются** с warning в stderr

---

## Симметрия

Применяется к финальному canvas после блендинга и weave, до смешивания с paper. `np.maximum` — линии из оригинала и отражения не теряются.

- `vertical` — зеркало по центру по X
- `horizontal` — по Y
- `central` — поворот 180°
- `radialN` — N поворотов с шагом 360°/N (для прямоугольников pad до квадрата)

Для банкнот типично `radial4`, `radial6`, `radial8`.

---

## Ref-warp

Деформация сетки по яркости reference (`--ref-warp != 0` требует `--ref` или spec с reference):

- `>0` — сгущение линий в ярких областях
- `<0` — в тёмных
- `0.3–1.2` — разумный диапазон
- `>1.4` — линии могут складываться
- `0` — выкл

`--ref-smooth` задаёт крупность деталей, которые сетка «замечает» (больше = мягче).

**Не работает с `--flat-color`** — flat-color отключает reference-цвет, а с ним и `ref_lum`.

---

## Переменная толщина

Классический банкнотный признак — толщина линии меняется вдоль её параметра. Включается `--variable-thickness`, тонкая настройка через `--thickness-*`:

```bash
ornament_gen --recipe woven_classic --size a4 --dpi 300 --quality normal \
    --variable-thickness --thickness-amplitude 0.5 --thickness-period 6.0 \
    --thickness-phase 0.0 --thickness-base 1.0 --out ./out
```

**Замечание:** `--variable-thickness` применяется ко всем слоям. Для разной толщины на разных слоях используйте `--layers-spec` (в spec каждый слой имеет свой `thickness.kind`).

---

## Глобальные оверрайды слоёв

`--layer-opacity`, `--layer-blend`, `--weave-z-period`, `--weave-z-phase` переопределяют **все** слои после загрузки рецепта:

```bash
# Один рецепт — разные стили без правки recipes.py
ornament_gen --recipe woven_classic --layer-blend multiply   --out ./a
ornament_gen --recipe woven_classic --layer-blend screen     --out ./b
ornament_gen --recipe woven_classic --layer-opacity 0.6      --out ./c
```

**Порядок применения:** рецепт → `--weave` / `--weave-shadow` → `--weave-z-preset` (только для слоёв без своего preset) → `_apply_layer_overrides` (`z-period`, `z-phase`, `opacity`, `blend`, `thickness`).

---

## Фоновые режимы

| `--background` | PIL mode | Смысл |
|---|---|---|
| `paper` | RGB | Бумага (случайный пастеризованный оттенок) под линиями |
| `white` | RGB | Белая бумага |
| `black` | RGB | Чёрная бумага |
| `none` | RGBA | Только линии на прозрачном фоне |

Для документа/печати — `paper`. Для встраивания — `none`.

---

## Векторизационный режим

Комбинация для дальнейшей трассировки (potrace, VTracer):

```bash
ornament_gen --recipe woven_classic --size a4 --dpi 300 --quality high \
    --flat-color --background none --no-noise --compress-level 0 --out ./vec
```

- `--flat-color` — однотонные чёрные линии без градиента
- `--no-noise` — без шума
- `--background none` — RGBA только с линиями
- `--compress-level 0` — без потерь сжатия

Векторизатор (вне scope проекта) даст чистые замкнутые контуры.

---

## Архитектура (кратко)

```
cli.py           # argparse, батч, dump/load spec, оверрайды слоёв
  ↓
compose.py       # пайплайн: curves → mask_ss → weave → effects → blend → symmetry → paper
  ├── nets/      # 9 генераторов (warped, lens, polar, spiral, spirograph, …)
  ├── patterns/  # 5 узоров (dots, crosses, contours, voronoi_fill, rosette)
  ├── effects.py # glow, inner_glow, emboss, shadow, micro_echo
  ├── weave.py   # z-buffer resolve (G3)
  ├── symmetry.py# none/vertical/horizontal/central/radialN
  ├── warp.py    # ref-warp: поле смещений из яркости
  ├── blend.py   # 13 режимов
  ├── color.py   # градиенты + reference
  ├── cache.py   # LRU curves
  └── ram.py     # оценка RAM + auto-downgrade quality
```

Пайплайн `make()`:
1. Цвет/яркость из reference или встроенного градиента (один раз)
2. Для каждого слоя: `gen() → warp → thickness → rasterize(+z)`
3. Weave resolve (если `zbuffer`)
4. Блендинг на чёрном (C-base-1): `below` → `main` → `above`
5. Симметрия
6. Смешивание с paper (RGB) или упаковка alpha (RGBA)

---

## Расширение

### Добавить сетку

1. Создать `ornament_gen/nets/my_net.py`:
   ```python
   def generate(ctx, rng, params) -> list[Curve]: ...
   ```
2. Зарегистрировать в `ornament_gen/nets/__init__.py` в `NETS`.

### Добавить узор

Аналогично, в `ornament_gen/patterns/my_pattern.py` + регистрация в `PATTERNS`.

### Добавить эффект

В `ornament_gen/effects.py` — новая функция + ветка в `build_effect_layers`.
Формат: вернуть `EffectLayer` (для `above`/`below`) или модифицированную `main_mask`.

### Добавить рецепт

В `ornament_gen/recipes.py` — функция без аргументов, возвращающая `{layers: [...], weave: {...}}`. Зарегистрировать в `RECIPES`.

---

## Ограничения (известные)

### ⚠ Критическое

**C-base-1 в `compose.py`:** `canvas = np.zeros(...)` + `blend("multiply")` = 0 вне линий → `paper * 0 = 0`. **Все рецепты с `blend="multiply"` дают чёрное изображение** (без эффектов). Тесты проходят (проверяют только shape/dtype), но реальный рендер не соответствует документному качеству.

**Смягчение:** используйте `--background none` (RGBA) — там `rgb = canvas`, `alpha = mask`, прозрачные пиксели сохраняются.

**Фикс (не реализован):** `canvas = np.ones(...)` вместо `zeros` в начале композита. Изменит все результаты bit-for-bit, но восстановит работоспособность.

### Прочие

- **`MAX_SS_PIXELS = 120M`** в `compose.py` — A4@300 quality=high (139M) → `ValueError`. Для A4+ используйте quality ≤ normal или `--max-ram`.
- **`--n > 50`** — запрещено (жёсткий лимит).
- **`--random-layers`** может дать «мусор» — пресеты надёжнее. Может включить `contours` (медленно, ~30 сек на слой).
- **Кэш curves** помогает только при повторных вызовах в одной сессии; в батче seed меняется каждый раз — кэш не срабатывает (by design).
- **`--flat-color` несовместим с `--ref-warp`** (нет reference).
- **`--symmetry` игнорируется при `--layers-spec`** (spec авторитет).
- **`--layer-opacity`, `--layer-blend`, `--weave-z-*`, `--thickness-*` игнорируются при `--layers-spec`** (spec авторитет, warning в stderr — неполный).
- **`--variable-thickness` применяется ко всем слоям одновременно** — для разной толщины используйте `--layers-spec`.
- **RadialN на больших прямоугольниках** — pad до квадрата, поворот, crop; на N=12 и A4 может пиково занимать ~1.2 GB RAM.
- **A4 SS=3 + weave** — пиково ~5 GB RAM (учтено в `--max-ram`).
- **`spiral_moire`** — обе спирали идентичны (не настоящий moiré), только разные blend.
- **`FileNotFoundError`** при `--layers-spec`/`--dump-spec` в несуществующий путь — traceback вместо `ap.error`.
- **`--random-layers -5`** → пустой `layers` → чёрное изображение (без ошибки).
- **`_as_color_array` fallback на белый** для final-цвета — `glow`/`micro_echo` используют белый вместо base color.
- **`--seed 0` при `--layers-spec`** не override (нельзя явно задать 0).

---

## Тесты

```bash
pip install -e ".[test]"
python -m pytest -q
```

**~280 тестов**, покрывают:
- `test_foundation.py` (23) — `sizes`, `core`, `blend`, `metadata`
- `test_nets.py` (28) — 9 сеток: smoke, детерминизм, покрытие
- `test_weave.py` (27) — z-пресеты, EDT, resolve, интеграция
- `test_patterns.py` (22) — 5 узоров + edge cases
- `test_effects.py` (26) — 5 эффектов + `build_effect_layers`
- `test_compose.py` (25) — пайплайн, weave, effects, symmetry, ref-warp, spec
- `test_spec.py` (15) — JSON roundtrip, валидация
- `test_cli.py` (31) — все CLI-флаги, batch, spec, overrides
- `test_symmetry.py` (22) — валидатор, все режимы, 3D
- `test_warp.py` (13) — `warp_fields`, `apply_warp_to_curves`, bilinear
- `test_cache.py` (11) — LRU, snapshot/restore, интеграция
- `test_ram.py` (8) — оценка, auto-downgrade

**Замечание:** тесты не проверяют **визуальное содержимое** (`arr.std()`, средняя яркость, цвет) — C-base-1 баг не ловится. Рекомендуется добавить тест на `paper background not black`.

---