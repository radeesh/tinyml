# TinyML Image Recognition for ESP32-S3

A complete Python pipeline for training, quantizing, and deploying tiny image classification models on **ESP32-S3** using **MicroPython + TFLite Micro**.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Desktop (Python + uv)                       │
│                                                                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐   │
│  │ Prepare  │──▶│  Train   │──▶│ Convert  │──▶│  Export    │   │
│  │ Dataset  │   │  CNN     │   │ to TFLite│   │  for ESP32 │   │
│  └──────────┘   └──────────┘   └──────────┘   └─────┬──────┘   │
│                                                     │          │
└─────────────────────────────────────────────────────┼──────────┘
                                                      │
                          mpremote cp esp32/* :       │
                                                      ▼
┌────────────────────────────────────────────────────────────────┐
│                     ESP32-S3 (MicroPython)                     │
│                                                                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐   │
│  │ Camera   │──▶│Preprocess│──▶│ TFLite   │──▶│ Output     │   │
│  │ Capture  │   │ Image    │   │ Inference│   │ Prediction │   │
│  └──────────┘   └──────────┘   └──────────┘   └────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Prepare Your Dataset

Organize your images in folders — one folder per class:

```
data/
├── cats/
│   ├── cat001.jpg
│   ├── cat002.jpg
│   └── ...
├── dogs/
│   ├── dog001.jpg
│   └── ...
└── birds/
    ├── bird001.jpg
    └── ...
```

Then split into train/val:

```bash
uv run tinyml-prepare
```

### 3. Train the Model

```bash
uv run tinyml-train
```

Optional arguments:
```bash
uv run tinyml-train --epochs 100 --input-size 48 --batch-size 16
uv run tinyml-train --num-filters 8 16 --grayscale  # even smaller model
```

### 4. Convert to TFLite (INT8 Quantized)

```bash
uv run tinyml-convert
```

### 5. Evaluate (Optional)

```bash
uv run tinyml-evaluate           # Keras model
uv run tinyml-evaluate --tflite  # TFLite model (verify quantization)
```

### 6. Export for ESP32-S3

```bash
uv run tinyml-export
```

This generates all deployment files in `esp32/`:
```
esp32/
├── model.tflite    # Quantized model
├── labels.txt      # Class names
├── main.py         # MicroPython inference script
├── boot.py         # Boot configuration
└── config.json     # Model metadata
```

### 7. Deploy to ESP32-S3

```bash
# Install mpremote
pip install mpremote

# Copy files to ESP32
mpremote cp esp32/boot.py :boot.py
mpremote cp esp32/main.py :main.py
mpremote cp esp32/model.tflite :model.tflite
mpremote cp esp32/labels.txt :labels.txt
mpremote cp esp32/config.json :config.json

# Reset the board
mpremote reset
```

## ESP32-S3 Firmware Requirements

You need MicroPython firmware with **microlite** (TFLite Micro) support:

1. **Option A: Pre-built firmware**
   - Download from [micropython-microlite releases](https://github.com/mocleiri/tensorflow-micropython-examples)
   - Flash: `esptool.py --chip esp32s3 write_flash 0x0 firmware.bin`

2. **Option B: Build from source**
   ```bash
   git clone https://github.com/mocleiri/tensorflow-micropython-examples
   cd tensorflow-micropython-examples
   # Follow build instructions for ESP32-S3
   ```

## Model Architecture

The CNN uses **depthwise separable convolutions** to minimize parameters:

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Input | 96×96×3 | 0 |
| Conv2D 16 + BN | 48×48×16 | ~448 |
| SepConv2D 32 + BN | 24×24×32 | ~720 |
| SepConv2D 32 + BN | 12×12×32 | ~1,184 |
| GlobalAvgPool | 32 | 0 |
| Dense 64 | 64 | 2,112 |
| Dense N (softmax) | N | 65×N |

**Total: ~4,600 params (3 classes) → ~5 KB INT8 model**

## Configuration

All settings are in [`src/tinyml/config.py`](src/tinyml/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_size` | 96 | Image width/height (pixels) |
| `input_channels` | 3 | 3=RGB, 1=grayscale |
| `num_filters` | [16,32,32] | Conv filters per block |
| `dense_units` | 64 | Dense layer size |
| `dropout_rate` | 0.3 | Dropout for regularization |
| `epochs` | 50 | Max training epochs |
| `batch_size` | 32 | Training batch size |
| `quantize_int8` | True | Full INT8 quantization |

### Tuning for Your Use Case

**Smaller model** (fewer classes, tight memory):
```bash
uv run tinyml-train --input-size 48 --num-filters 8 16 --grayscale
```

**Larger model** (more classes, ESP32-S3 with PSRAM):
```bash
uv run tinyml-train --input-size 96 --num-filters 16 32 64
```

## Project Structure

```
tinyml/
├── pyproject.toml              # uv project configuration
├── README.md
├── src/tinyml/
│   ├── __init__.py
│   ├── config.py               # Model & training configuration
│   ├── model.py                # CNN architecture
│   ├── prepare_data.py         # Dataset preparation
│   ├── train.py                # Training pipeline
│   ├── convert.py              # TFLite conversion + quantization
│   ├── evaluate.py             # Model evaluation
│   └── export.py               # ESP32 deployment export
├── data/                       # Your image dataset
│   ├── train/                  # (generated) training split
│   └── val/                    # (generated) validation split
├── models/                     # (generated) saved models
│   ├── model.keras             # Trained Keras model
│   ├── model_int8.tflite       # Quantized TFLite model
│   └── labels.txt              # Class labels
└── esp32/                      # (generated) ESP32 deployment files
    ├── model.tflite
    ├── labels.txt
    ├── main.py
    ├── boot.py
    └── config.json
```

## Tips

- **Minimum images**: Aim for 50+ images per class for reasonable accuracy. Data augmentation helps but can't replace real data.
- **Image quality**: Use consistent lighting and framing. The model sees 96×96 pixels — make sure your subject is clearly visible at that resolution.
- **Memory**: If you get memory errors on ESP32, reduce `input_size` to 48 and `num_filters` to `[8, 16]`.
- **Speed**: INT8 inference on ESP32-S3 at 240 MHz typically runs in 50-200ms per image depending on model size.
