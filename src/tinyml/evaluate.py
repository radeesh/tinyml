"""
Evaluate a trained model and generate performance reports.

Usage:
    uv run tinyml-evaluate
    uv run tinyml-evaluate --tflite   # evaluate the TFLite model instead
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


def evaluate_keras(config: ModelConfig) -> None:
    """Evaluate the Keras model on the validation set."""
    print(f"📦 Loading Keras model from {config.keras_model_path}")
    model = keras.models.load_model(str(config.keras_model_path))

    val_dir = config.data_dir / "val"
    if not val_dir.exists():
        print("❌ No validation directory found. Run tinyml-prepare first.")
        return

    val_ds = image_dataset_from_directory(
        val_dir,
        image_size=(config.input_size, config.input_size),
        batch_size=config.batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

    loss, accuracy = model.evaluate(val_ds, verbose=1)
    print(f"\n🎯 Validation loss:     {loss:.4f}")
    print(f"🎯 Validation accuracy: {accuracy:.4f}")

    # Per-class accuracy
    class_names = val_ds.class_names
    _per_class_report(model, val_ds, class_names)


def evaluate_tflite(config: ModelConfig) -> None:
    """Evaluate the TFLite model to verify quantization accuracy."""
    tflite_path = config.tflite_model_path
    print(f"📦 Loading TFLite model from {tflite_path}")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"   Input:  {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
    print(f"   Output: {output_details[0]['shape']} dtype={output_details[0]['dtype']}")

    val_dir = config.data_dir / "val"
    if not val_dir.exists():
        print("❌ No validation directory found.")
        return

    val_ds = image_dataset_from_directory(
        val_dir,
        image_size=(config.input_size, config.input_size),
        batch_size=1,
        label_mode="int",
        shuffle=False,
    )

    class_names = val_ds.class_names
    correct = 0
    total = 0

    input_scale, input_zero_point = input_details[0].get("quantization", (1.0, 0))

    for images, labels in val_ds:
        image = images[0].numpy()

        # Quantize input if INT8 model
        if input_details[0]["dtype"] == np.int8:
            image = (image / 255.0) / input_scale + input_zero_point
            image = np.clip(image, -128, 127).astype(np.int8)
        else:
            image = (image / 255.0).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], np.expand_dims(image, 0))
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])[0]
        predicted = np.argmax(output)
        actual = labels[0].numpy()

        if predicted == actual:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\n🎯 TFLite validation accuracy: {accuracy:.4f} ({correct}/{total})")

    # Compare with Keras model accuracy
    history_path = config.models_dir / "training_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        keras_acc = history.get("val_accuracy", [0])[-1]
        diff = accuracy - keras_acc
        print(f"   Keras val accuracy:  {keras_acc:.4f}")
        print(f"   Accuracy difference: {diff:+.4f}")
        if abs(diff) < 0.02:
            print("   ✅ Quantization accuracy loss is minimal")
        else:
            print("   ⚠ Significant accuracy loss from quantization")


def _per_class_report(
    model: keras.Model,
    dataset: tf.data.Dataset,
    class_names: list[str],
) -> None:
    """Print per-class accuracy breakdown."""
    from sklearn.metrics import classification_report

    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    print("\n📊 Per-class classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TinyML model")
    parser.add_argument(
        "--tflite",
        action="store_true",
        help="Evaluate the TFLite model instead of Keras",
    )
    args = parser.parse_args()

    config = ModelConfig()

    if args.tflite:
        evaluate_tflite(config)
    else:
        evaluate_keras(config)


if __name__ == "__main__":
    main()
