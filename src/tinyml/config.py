"""
ESP32-S3 model constraints and training configuration.

All model architecture decisions are driven by ESP32-S3 hardware limits:
  - ~512 KB SRAM (usable ~320 KB after MicroPython overhead)
  - ~8 MB PSRAM (if available, slower access)
  - No FPU for double precision — INT8 quantization is essential
  - Input images should be small (48x48 or 96x96 max)
"""

from dataclasses import dataclass, field
from pathlib import Path


# ── Project paths ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ESP32_DIR = PROJECT_ROOT / "esp32"

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, ESP32_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for the TinyML CNN model.

    These defaults are tuned for ESP32-S3 with PSRAM. If targeting a
    bare ESP32 without PSRAM, reduce input_size to 48 and num_filters
    to [8, 16].
    """

    # ── Input ───────────────────────────────────────────────────────
    input_size: int = 96          # Input image width & height (square)
    input_channels: int = 3       # 3 for RGB, 1 for grayscale
    num_classes: int = 3          # Override based on your dataset

    # ── Architecture ────────────────────────────────────────────────
    num_filters: list[int] = field(default_factory=lambda: [16, 32, 32])
    kernel_size: int = 3
    dense_units: int = 64
    dropout_rate: float = 0.3

    # ── Training ────────────────────────────────────────────────────
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10

    # ── Quantization ────────────────────────────────────────────────
    quantize_int8: bool = True    # Full INT8 quantization for ESP32

    # ── Paths ───────────────────────────────────────────────────────
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    esp32_dir: Path = ESP32_DIR

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.input_size, self.input_size, self.input_channels)

    @property
    def keras_model_path(self) -> Path:
        return self.models_dir / "model.keras"

    @property
    def tflite_model_path(self) -> Path:
        suffix = "_int8" if self.quantize_int8 else "_float32"
        return self.models_dir / f"model{suffix}.tflite"

    @property
    def labels_path(self) -> Path:
        return self.models_dir / "labels.txt"
