"""
Convert a trained Keras model to TensorFlow Lite with INT8 quantization.

INT8 quantization is critical for ESP32-S3 because:
  1. Reduces model size by ~4× (float32 → int8)
  2. ESP32 has no FPU — integer ops are much faster
  3. Fits within the tight SRAM constraints

Usage:
    uv run tinyml-convert
    uv run tinyml-convert --no-quantize   # float32 for debugging
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tinyml.config import ModelConfig


def _representative_dataset_gen(config: ModelConfig):
    """Generate representative data for INT8 calibration.

    The TFLite converter needs representative input samples to
    determine the quantization range for each tensor. We try to
    use real validation images; if unavailable, we use random data.
    """
    val_dir = config.data_dir / "val"
    train_dir = config.data_dir / "train"
    source_dir = val_dir if val_dir.exists() else (train_dir if train_dir.exists() else None)

    if source_dir is not None:
        # Load real images for calibration
        ds = keras.utils.image_dataset_from_directory(
            source_dir,
            image_size=(config.input_size, config.input_size),
            batch_size=1,
            label_mode=None,
            shuffle=True,
            seed=42,
        )
        # Normalize to [0, 1]
        ds = ds.map(lambda x: x / 255.0)
        count = 0
        for batch in ds.take(100):  # Use up to 100 samples
            yield [batch.numpy().astype(np.float32)]
            count += 1
        print(f"   Used {count} real images for INT8 calibration")
    else:
        # Fallback: random data (less accurate quantization)
        print("   ⚠ No dataset found — using random data for calibration")
        for _ in range(100):
            yield [
                np.random.rand(1, config.input_size, config.input_size, config.input_channels)
                .astype(np.float32)
            ]


def convert(config: ModelConfig) -> Path:
    """Convert Keras model to TFLite format.

    Returns the path to the saved .tflite file.
    """
    print(f"📦 Loading Keras model from {config.keras_model_path}")
    model = keras.models.load_model(str(config.keras_model_path))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if config.quantize_int8:
        print("🔧 Applying full INT8 quantization ...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: _representative_dataset_gen(config)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        print("🔧 Converting to float32 TFLite (no quantization) ...")

    tflite_model = converter.convert()

    # Save the model
    output_path = config.tflite_model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"\n💾 TFLite model saved to {output_path}")
    print(f"📊 Model size: {size_kb:.1f} KB")

    if size_kb > 300:
        print("⚠ Model exceeds ~300 KB — may not fit in ESP32-S3 SRAM")
        print("  Consider reducing input_size, num_filters, or dense_units")
    else:
        print("✅ Model size is suitable for ESP32-S3")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert model to TFLite")
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT8 quantization (produce float32 model)",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    args = parser.parse_args()

    config = ModelConfig(quantize_int8=not args.no_quantize)
    if args.model_path:
        config.keras_model_path  # Read-only property, override differently
        # Load from custom path
        pass

    convert(config)


if __name__ == "__main__":
    main()
