"""Подготовка датасета патологий для U-Net из DENTEX diagnosis.

DENTEX diagnosis содержит боксы 4 классов:
  0 - Impacted
  1 - Caries
  2 - Periapical Lesion
  3 - Deep Caries

Боксы превращаем в приближённые маски (заполненные прямоугольники).
Это не идеальные контуры, но позволяет обучить U-Net без ручной разметки.

Результат:
    data/pathology/
        train/images + train/masks
        valid/images + valid/masks
        test/images  + test/masks
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "pathology"

CLASS_NAMES = {
    0: "impacted",
    1: "caries",
    2: "periapical",
    3: "deep_caries",
}

SPLIT_DIRS = {
    "train": "train",
    "validation": "valid",
    "test": "test",
}


def _clear_split(folder: str) -> None:
    for name in ("images", "masks"):
        path = DATASET_DIR / folder / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def _box_to_mask(mask: np.ndarray, bbox: list, class_id: int) -> None:
    x, y, w, h = [int(round(v)) for v in bbox]
    height, width = mask.shape

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = class_id + 1  # 0 = background


def _get_category_id(cat) -> int | None:
    if isinstance(cat, int):
        return cat if cat in CLASS_NAMES else None

    name = str(cat).lower().strip()
    mapping = {
        "impacted": 0,
        "caries": 1,
        "periapical lesion": 2,
        "periapical": 2,
        "deep caries": 3,
        "deep_caries": 3,
    }
    return mapping.get(name)


def _write_sample(image_dir: Path, mask_dir: Path, row: dict, image_id: str) -> int:
    image = row["image"]
    width, height = image.size
    mask = np.zeros((height, width), dtype=np.uint8)

    objects = row.get("objects", {})
    bboxes = objects.get("bbox", [])
    categories = objects.get("category", objects.get("category_id", objects.get("diagnosis", [])))

    count = 0
    for bbox, cat in zip(bboxes, categories):
        class_id = _get_category_id(cat)
        if class_id is None:
            continue
        _box_to_mask(mask, bbox, class_id)
        count += 1

    image.save(image_dir / f"{image_id}.jpg", format="JPEG", quality=95)
    Image.fromarray(mask).save(mask_dir / f"{image_id}.png")
    return count


def _write_split(split_name: str, rows) -> int:
    folder = SPLIT_DIRS[split_name]
    _clear_split(folder)

    image_dir = DATASET_DIR / folder / "images"
    mask_dir = DATASET_DIR / folder / "masks"

    total_boxes = 0
    seen = set()

    for index, row in enumerate(rows, start=1):
        image_id = str(row.get("image_id", index))
        if image_id in seen:
            image_id = f"{image_id}_{index}"
        seen.add(image_id)

        total_boxes += _write_sample(image_dir, mask_dir, row, image_id)

        if index % 50 == 0:
            print(f"  {folder}: записано {index} снимков")

    print(f"  {folder}: снимков {len(seen)}, патологий {total_boxes}")
    return total_boxes


def prepare(force: bool = False) -> None:
    if (DATASET_DIR / "train" / "images").exists() and not force:
        print(f"Датасет уже есть в {DATASET_DIR}. Для пересборки: --force")
        return

    print("Качаю DENTEX diagnosis...")
    dataset = load_dataset("tirandazdylan/dentex-panoramic", "diagnosis")

    print("Сплиты:", list(dataset.keys()))

    total = 0
    for split_name in ("train", "validation", "test"):
        if split_name in dataset:
            total += _write_split(split_name, dataset[split_name])

    classes_path = DATASET_DIR / "classes.txt"
    classes_path.write_text(
        "\n".join(f"{i} {name}" for i, name in CLASS_NAMES.items()) + "\n",
        encoding="utf-8",
    )

    print(f"\nГотово. Всего патологий: {total}")
    print(f"Классы: {CLASS_NAMES}")
    print(f"Данные: {DATASET_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(force=args.force)


if __name__ == "__main__":
    main()