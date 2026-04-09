# E-commerce Product Segmentation

Решение для задачи сегментации главного объекта на фотографиях товаров e-commerce. Задача -- по входному RGB-изображению 1024x1024 предсказать alpha-маску (grayscale 1024x1024), отделяющую товар от фона.

**Результат: MSE 185, 3-е место** (Kaggle, 3 сабмита на финальном лидерборде).

## Примеры работы

| Оригинал | Coarse-маска | После калибровки |
|----------|-------------|-----------------|
| ![](examples/example_basket.png) | ![](examples/example_basket.png) | ![](examples/example_basket.png) |
| ![](examples/example_pan.png) | ![](examples/example_pan.png) | ![](examples/example_pan.png) |
| ![](examples/example_doll.png) | ![](examples/example_doll.png) | ![](examples/example_doll.png) |

> Три колонки на каждом скриншоте: оригинальное изображение, raw-предсказание модели (Coarse), результат после gamma/threshold калибровки (Refined).

## Подход

Пайплайн состоит из двух моделей, TTA и постобработки:

```
Входное изображение (1024x1024)
          |
    +-----+------+
    |              |
 BiRefNet       RMBG-1.4
 (дообученный)   (pretrained)
    |              |
   TTA            TTA
  (flips +       (flips +
  multiscale)    multiscale)
    |              |
    +-- Ensemble --+
     (0.6 / 0.4)
          |
   Постобработка
  (gamma + threshold)
          |
  Alpha-маска (1024x1024)
```

### Модели

- **BiRefNet** ([ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)) -- Bilateral Reference Network, SOTA 2024 для Dichotomous Image Segmentation. Дообучен на публичных matting-датасетах (DIS5K, P3M-10k, AM-2K, HIM2K) с замороженным энкодером.
- **RMBG-1.4** ([briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)) -- IS-Net от BRIA AI, специализирован на удалении фона для e-commerce. Используется as-is без дообучения.

### Энсамбль

Взвешенное усреднение предсказаний двух моделей: BiRefNet (вес 0.6) + RMBG (вес 0.4). BiRefNet точнее на границах, RMBG стабильнее на простых объектах.

### Test-Time Augmentation (TTA)

Для каждой модели применяется:
- **Flip TTA**: original + horizontal flip + vertical flip + H+V flip (4 прохода)
- **Multi-scale TTA**: инференс на масштабах 0.875x, 1.0x, 1.125x с resize обратно к 1024x1024

Итого 4 flips x 3 scales x 2 модели = **24 forward pass** на изображение.

### Постобработка

Два легковесных шага поверх raw-предсказания:

1. **Gamma-коррекция** (gamma=1.4): `alpha' = alpha^1.4` -- подавляет слабые тени и полупрозрачные артефакты, не трогая уверенные области
2. **Soft threshold** (t_low=0.04, t_high=0.97): линейный ремаппинг `[t_low, t_high] -> [0, 1]` с клампом -- убирает остаточный шум около 0 и 255

Параметры подобраны визуально на валидации.

## Дообучение BiRefNet (`train.py`)

Скрипт для fine-tuning BiRefNet на произвольных matting/segmentation датасетах.

### Стратегия обучения

- **Замороженный энкодер** (Swin Transformer backbone) -- обучается только декодер и refinement-модули. При batch_size=2 разморозка энкодера дестабилизирует BatchNorm.
- **Frozen BatchNorm** -- все BN-слои зафиксированы в eval-режиме
- **Mixed precision** (FP16) через `torch.cuda.amp`
- **Gradient accumulation** (2 шага) -- эффективный batch_size = 4
- **Cosine schedule** с 500-шаговым warmup, lr=1e-5
- **Early stopping** по val_mse с patience=6

### Аугментации

Модель обучается на изображениях, скомпозированных на случайные фоны:
- **Белый фон** (25%) -- основной кейс для e-commerce
- **Серый фон с шумом** (25%) -- чтобы модель не запоминала "белое = фон"
- **Цветной фон** (25%) -- случайные пастельные оттенки
- **Градиентный фон** (25%) -- плавные переходы сверху вниз

Дополнительно:
- **Синтетические тени** (p=0.10) -- размытая/смещенная alpha накладывается как затемнение фона
- **White-on-white** (p=0.10) -- осветление переднего плана для имитации низкоконтрастных случаев
- **Стандартные**: horizontal flip, slight shift/scale/rotate, color jitter

### Лосс

`L = MSE(pred, target) + 0.1 * L1(pred, target)`

Комбинация MSE (штрафует большие ошибки) и L1 (стабилизирует мелкие). Edge-weighted вариант тестировался, но приводил к переобучению на текстуры.

### Данные

Поддерживает два формата:
- **PNG**: директории `train/images/` + `train/alpha/` (одинаковые имена файлов)
- **LMDB**: для быстрого I/O на больших датасетах (pickled dict с ключами `image`, `alpha`)

## Инференс (`solution_final.ipynb`)

Jupyter-ноутбук с полным пайплайном:

1. Загрузка BiRefNet (с опциональными fine-tuned весами) и RMBG-1.4
2. Preprocessing для каждой модели (ImageNet norm для BiRefNet, custom для RMBG)
3. TTA (flip + multiscale)
4. Weighted ensemble
5. Постобработка (gamma, threshold, опциональный guided filter и shadow cleanup)
6. Визуализация и batch-инференс
7. Формирование submission (CSV с base64-encoded PNG масками)

## Структура проекта

```
.
├── train.py                 # Скрипт дообучения BiRefNet
│                            #   - MattingDataset / LMDBMattingDataset
│                            #   - MattingLoss (MSE + L1)
│                            #   - Background augmentation pipeline
│                            #   - Training loop с checkpointing и early stopping
│
├── solution_final.ipynb     # Ноутбук инференса
│                            #   - Загрузка BiRefNet + RMBG-1.4
│                            #   - TTA (flip + multiscale)
│                            #   - Ensemble + постобработка
│                            #   - Визуализация и генерация submission
│
├── examples/                # Примеры результатов
│   ├── example_basket.png
│   ├── example_pan.png
│   └── example_doll.png
│
├── requirements.txt         # Зависимости
├── .gitignore
└── README.md
```

## Запуск

### Установка

```bash
pip install -r requirements.txt
```

### Инференс

Открыть `solution_final.ipynb`, указать:
- `TEST_DIR` -- путь к папке с тестовыми изображениями
- `FINETUNED_CHECKPOINT_PATH` -- путь к чекпоинту (или отключить `USE_FINETUNED_BIREFNET = False`)

Запустить все ячейки. Результат -- CSV с alpha-масками в base64.

### Дообучение

```bash
python train.py
```

Перед запуском настроить пути к данным и параметры в секции `CONFIGURATION` в начале файла.

### Требования

- **GPU**: NVIDIA с 12+ GB VRAM (тестировалось на RTX 3060 12GB)
- **VRAM инференса**: ~6-7 GB для двух моделей с TTA
- **Время инференса**: ~14 минут на 100 изображений (24 forward pass на картинку)

## Ссылки

- [BiRefNet: Bilateral Reference for High-Resolution Dichotomous Image Segmentation](https://arxiv.org/abs/2401.03407) (Zheng et al., 2024)
- [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) -- BRIA AI
- [DIS5K](https://xuebinqin.github.io/dis/index.html), [P3M-10k](https://github.com/JizhiziLi/P3M), [AM-2K / HIM2K](https://github.com/JizhiziLi/GFM)
