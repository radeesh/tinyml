"""
Generate a synthetic demo dataset for testing the pipeline.

Creates 3 classes of simple geometric shapes:
  - circles
  - squares
  - triangles

Each image is 96x96 RGB with random colors and positions.
This lets you verify the full pipeline without needing real images.

Usage:
    uv run python scripts/generate_demo_data.py
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NUM_IMAGES_PER_CLASS = 100
IMAGE_SIZE = 96
CLASSES = ["circle", "square", "triangle"]


def random_color() -> tuple[int, int, int]:
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def random_bg_color() -> tuple[int, int, int]:
    return (random.randint(0, 80), random.randint(0, 80), random.randint(0, 80))


def draw_circle(draw: ImageDraw.Draw, size: int) -> None:
    margin = size // 6
    cx = random.randint(margin + 10, size - margin - 10)
    cy = random.randint(margin + 10, size - margin - 10)
    r = random.randint(10, min(cx, cy, size - cx, size - cy) - 2)
    color = random_color()
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=color)


def draw_square(draw: ImageDraw.Draw, size: int) -> None:
    margin = size // 6
    cx = random.randint(margin + 10, size - margin - 10)
    cy = random.randint(margin + 10, size - margin - 10)
    half = random.randint(8, min(cx, cy, size - cx, size - cy) - 2)
    color = random_color()
    draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=color)


def draw_triangle(draw: ImageDraw.Draw, size: int) -> None:
    margin = size // 6
    cx = random.randint(margin + 15, size - margin - 15)
    cy = random.randint(margin + 15, size - margin - 15)
    r = random.randint(10, min(cx, cy, size - cx, size - cy) - 5)
    color = random_color()
    angle_offset = random.uniform(0, 2 * math.pi)
    points = []
    for i in range(3):
        angle = angle_offset + (2 * math.pi * i / 3)
        px = cx + r * math.cos(angle)
        py = cy + r * math.sin(angle)
        points.append((px, py))
    draw.polygon(points, fill=color)


SHAPE_DRAWERS = {
    "circle": draw_circle,
    "square": draw_square,
    "triangle": draw_triangle,
}


def generate_dataset() -> None:
    print(f"🎨 Generating demo dataset in {DATA_DIR}")
    print(f"   {NUM_IMAGES_PER_CLASS} images × {len(CLASSES)} classes")

    for cls in CLASSES:
        cls_dir = DATA_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        drawer = SHAPE_DRAWERS[cls]

        for i in range(NUM_IMAGES_PER_CLASS):
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), random_bg_color())
            draw = ImageDraw.Draw(img)

            # Draw 1-3 shapes of the target class
            num_shapes = random.randint(1, 3)
            for _ in range(num_shapes):
                drawer(draw, IMAGE_SIZE)

            # Add some noise dots
            for _ in range(random.randint(5, 20)):
                x = random.randint(0, IMAGE_SIZE - 1)
                y = random.randint(0, IMAGE_SIZE - 1)
                draw.point((x, y), fill=random_color())

            img.save(cls_dir / f"{cls}_{i:04d}.png")

        print(f"   ✅ {cls}: {NUM_IMAGES_PER_CLASS} images")

    print(f"\n✅ Demo dataset ready at {DATA_DIR}")
    print("   Next: run 'uv run tinyml-prepare' to split into train/val")


if __name__ == "__main__":
    generate_dataset()
