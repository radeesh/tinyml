"""
Train the TinyML model on a custom image dataset.

Usage:
    uv run tinyml-train
    uv run tinyml-train --epochs 100 --input-size 48
    uv run tinyml-train --data-dir /path/to/my/images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import image_dataset_from_directory

from tinyml.config import ModelConfig
from tinyml.model import build_model, compile_model, print_model_info


def load_dataset(
    config: ModelConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Load train and validation datasets from directory structure.

    Expects:
        data/train/<class_name>/images...
        data/val/<class_name>/images...

    Falls back to a single data/ dir with auto-split if train/val
    subdirectories don't exist.
    """
    train_dir = config.data_dir / "train"
    val_dir = config.data_dir / "val"

    common_kwargs = dict(
        image_size=(config.input_size, config.input_size),
        batch_size=config.batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )

    if train_dir.exists() and val_dir.exists():
        print(f"📂 Loading training data from {train_dir}")
        train_ds = image_dataset_from_directory(train_dir, **common_kwargs)
        print(f"📂 Loading validation data from {val_dir}")
        val_ds = image_dataset_from_directory(val_dir, **common_kwargs)
        class_names = train_ds.class_names
    else:
        # Auto-split from a flat directory
        print(f"📂 Loading data from {config.data_dir} with auto-split")
        train_ds = image_dataset_from_directory(
            config.data_dir,
            validation_split=config.validation_split,
            subset="training",
            **common_kwargs,
        )
        val_ds = image_dataset_from_directory(
            config.data_dir,
            validation_split=config.validation_split,
            subset="validation",
            **common_kwargs,
        )
        class_names = train_ds.class_names

    return train_ds, val_ds, class_names


def normalize_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Normalize pixel values from [0, 255] to [0, 1]."""
    normalization = keras.layers.Rescaling(1.0 / 255)
    return ds.map(lambda x, y: (normalization(x), y))


def augment_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Apply data augmentation to training dataset.

    Uses simple augmentations that don't add inference overhead —
    they're only applied during training.
    """
    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )
    return ds.map(lambda x, y: (augmentation(x, training=True), y))


def train(config: ModelConfig) -> keras.Model:
    """Full training pipeline: load data → build model → train → save."""

    # ── Load data ───────────────────────────────────────────────────
    train_ds, val_ds, class_names = load_dataset(config)
    config.num_classes = len(class_names)

    print(f"\n🏷  Classes ({config.num_classes}): {class_names}")

    # Save labels
    config.labels_path.parent.mkdir(parents=True, exist_ok=True)
    config.labels_path.write_text("\n".join(class_names) + "\n")
    print(f"💾 Labels saved to {config.labels_path}")

    # ── Preprocess ──────────────────────────────────────────────────
    train_ds = normalize_dataset(train_ds)
    train_ds = augment_dataset(train_ds)
    val_ds = normalize_dataset(val_ds)

    # Optimize pipeline performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # ── Build model ─────────────────────────────────────────────────
    model = build_model(config)
    model = compile_model(model, config)
    print_model_info(model)

    # ── Callbacks ───────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(config.keras_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Train ───────────────────────────────────────────────────────
    print(f"\n🚀 Training for up to {config.epochs} epochs ...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save training history ───────────────────────────────────────
    history_path = config.models_dir / "training_history.json"
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_path.write_text(json.dumps(history_dict, indent=2))
    print(f"\n💾 Model saved to {config.keras_model_path}")
    print(f"📈 Training history saved to {history_path}")

    # ── Final metrics ───────────────────────────────────────────────
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n🎯 Final validation accuracy: {val_acc:.4f}")
    print(f"🎯 Final validation loss:     {val_loss:.4f}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyML image classifier")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-filters", type=int, nargs="+", default=[16, 32, 32])
    parser.add_argument("--grayscale", action="store_true", help="Use grayscale input")
    args = parser.parse_args()

    config = ModelConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        input_size=args.input_size,
        learning_rate=args.learning_rate,
        num_filters=args.num_filters,
        input_channels=1 if args.grayscale else 3,
    )
    if args.data_dir:
        config.data_dir = args.data_dir

    train(config)


if __name__ == "__main__":
    main()
