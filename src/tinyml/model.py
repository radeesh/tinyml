"""
Build and return the TinyML CNN model.

Architecture overview (for 96×96 RGB input, 3 classes):
    ┌─────────────────────────────────┐
    │  Input: 96 × 96 × 3            │
    ├─────────────────────────────────┤
    │  Conv2D 16 (3×3) + BN + ReLU   │
    │  MaxPool 2×2                    │   → 48×48×16
    ├─────────────────────────────────┤
    │  DepthwiseSep Conv2D 32 + BN    │
    │  MaxPool 2×2                    │   → 24×24×32
    ├─────────────────────────────────┤
    │  DepthwiseSep Conv2D 32 + BN    │
    │  MaxPool 2×2                    │   → 12×12×32
    ├─────────────────────────────────┤
    │  GlobalAveragePooling2D         │   → 32
    ├─────────────────────────────────┤
    │  Dense 64 + ReLU + Dropout      │
    │  Dense num_classes + Softmax    │
    └─────────────────────────────────┘

Uses depthwise separable convolutions (after the first layer) to
dramatically reduce parameter count while maintaining accuracy —
critical for fitting in ESP32-S3 SRAM.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from keras import layers

from tinyml.config import ModelConfig


def build_model(config: ModelConfig) -> keras.Model:
    """Build a tiny CNN optimized for ESP32-S3 deployment.

    The model uses:
    - Depthwise separable convolutions to minimize parameters
    - Batch normalization for training stability
    - Global average pooling instead of flatten (fewer params)
    - Dropout for regularization
    """
    inp = keras.Input(shape=config.input_shape, name="image_input")
    x = inp

    # ── First conv block: standard Conv2D ───────────────────────────
    x = layers.Conv2D(
        config.num_filters[0],
        config.kernel_size,
        padding="same",
        use_bias=False,
        name="conv_0",
    )(x)
    x = layers.BatchNormalization(name="bn_0")(x)
    x = layers.ReLU(name="relu_0")(x)
    x = layers.MaxPooling2D(pool_size=2, name="pool_0")(x)

    # ── Subsequent blocks: depthwise separable convolutions ─────────
    for i, filters in enumerate(config.num_filters[1:], start=1):
        x = layers.SeparableConv2D(
            filters,
            config.kernel_size,
            padding="same",
            use_bias=False,
            name=f"sep_conv_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.ReLU(name=f"relu_{i}")(x)
        x = layers.MaxPooling2D(pool_size=2, name=f"pool_{i}")(x)

    # ── Head ────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(config.dense_units, name="dense_0")(x)
    x = layers.ReLU(name="relu_dense")(x)
    x = layers.Dropout(config.dropout_rate, name="dropout")(x)
    out = layers.Dense(config.num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inp, outputs=out, name="tinyml_classifier")
    return model


def compile_model(model: keras.Model, config: ModelConfig) -> keras.Model:
    """Compile the model with Adam optimizer and categorical crossentropy."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def print_model_info(model: keras.Model) -> None:
    """Print model summary and estimated size for ESP32."""
    model.summary()
    total_params = model.count_params()
    # INT8 quantization: ~1 byte per parameter + overhead
    estimated_size_kb = total_params / 1024
    print(f"\n📊 Total parameters:     {total_params:,}")
    print(f"📊 Est. INT8 model size: ~{estimated_size_kb:.1f} KB")
    print(f"   ESP32-S3 SRAM limit:  ~320 KB (after MicroPython)")
    if estimated_size_kb > 300:
        print("   ⚠ Model may be too large — consider reducing filters or input_size")
    else:
        print("   ✅ Model should fit in ESP32-S3 SRAM")
