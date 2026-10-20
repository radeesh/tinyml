"""
Prepare a custom image dataset for training.

Expects images organized as:
    data/
    ├── class_a/
    │   ├── img001.jpg
    │   ├── img002.png
    │   └── ...
    ├── class_b/
    │   └── ...
    └── class_c/
        └── ...

Produces:
    data/train/   (80% of images, organized by class)
    data/val/     (20% of images, organized by class)
    models/labels.txt
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from tinyml.config import ModelConfig


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def discover_classes(data_dir: Path) -> list[str]:
    """Return sorted list of class names (subdirectory names)."""
    classes = sorted(
        d.name
        for d in data_dir.iterdir()
        if d.is_dir() and d.name not in {"train", "val", "test", "__pycache__"}
    )
    if not classes:
        raise FileNotFoundError(
            f"No class subdirectories found in {data_dir}.\n"
            "Expected structure:\n"
            "  data/\n"
            "    ├── class_a/\n"
            "    │   ├── img001.jpg\n"
            "    │   └── ...\n"
            "    └── class_b/\n"
            "        └── ..."
        )
    return classes


def collect_images(class_dir: Path) -> list[Path]:
    """Collect all image files from a class directory."""
    images = [
        f for f in class_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(images)


def split_dataset(
    data_dir: Path,
    val_split: float = 0.2,
    seed: int = 42,
) -> dict[str, dict[str, list[Path]]]:
    """Split images into train/val sets, stratified by class.

    Returns:
        {"train": {"class_a": [paths], ...}, "val": {"class_a": [paths], ...}}
    """
    rng = random.Random(seed)
    classes = discover_classes(data_dir)
    splits: dict[str, dict[str, list[Path]]] = {"train": {}, "val": {}}

    for cls in classes:
        images = collect_images(data_dir / cls)
        if not images:
            print(f"  ⚠ Warning: No images found in {data_dir / cls}")
            continue

        rng.shuffle(images)
        n_val = max(1, int(len(images) * val_split))
        splits["val"][cls] = images[:n_val]
        splits["train"][cls] = images[n_val:]

    return splits


def prepare(config: ModelConfig, seed: int = 42) -> None:
    """Organize raw images into train/val splits and write labels.txt."""
    data_dir = config.data_dir
    print(f"📂 Scanning dataset in {data_dir} ...")

    classes = discover_classes(data_dir)
    print(f"   Found {len(classes)} classes: {classes}")

    splits = split_dataset(data_dir, val_split=config.validation_split, seed=seed)

    # Copy images into train/ and val/ subdirectories
    for split_name, class_map in splits.items():
        for cls, images in class_map.items():
            dest = data_dir / split_name / cls
            dest.mkdir(parents=True, exist_ok=True)
            for img_path in images:
                shutil.copy2(img_path, dest / img_path.name)
            print(f"   {split_name}/{cls}: {len(images)} images")

    # Write labels file
    config.labels_path.parent.mkdir(parents=True, exist_ok=True)
    config.labels_path.write_text("\n".join(classes) + "\n")
    print(f"✅ Labels written to {config.labels_path}")

    # Update config with discovered class count
    config.num_classes = len(classes)
    print(f"✅ Dataset prepared: {config.num_classes} classes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare custom image dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to raw image directory (default: data/)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of images to use for validation (default: 0.2)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = ModelConfig()
    if args.data_dir:
        config.data_dir = args.data_dir
    config.validation_split = args.val_split

    prepare(config, seed=args.seed)


if __name__ == "__main__":
    main()
