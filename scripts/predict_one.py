"""Прогнать итоговую модель по одному снимку и показать, что она нашла.

Сеть при этом не учится: читаются готовые веса из best_model/.

Запуск из корня проекта:

    .venv\\Scripts\\python.exe scripts\\predict_one.py путь\\к\\снимку.jpg

Если путь не передан, берётся первый снимок из data/pathology/test/images.

На экране — доля пикселей каждого класса. Рядом сохраняется картинка:
оригинал с полупрозрачной маской поверх. Её и нужно смотреть глазами.
Маска должна попадать на область патологии, а не на фон.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathology.segmenter import Segmenter

WEIGHTS = ROOT / "best_model" / "pathology_unet_resnet34.pth"
TEST_IMAGES = ROOT / "data" / "pathology" / "test" / "images"
PREVIEW = ROOT / "runs" / "predict" / "one.png"

CLASS_NAMES = {0: "background", 1: "impacted", 2: "caries", 3: "periapical", 4: "deep_caries"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Найти патологии на одном снимке")
    parser.add_argument("image", nargs="?", type=Path, help="путь к jpg или png")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.5, help="порог вероятности: ниже — фон")
    parser.add_argument("--imgsz", type=int, default=512)
    return parser.parse_args()


def _default_image() -> Path:
    images = sorted(TEST_IMAGES.glob("*.jpg"))
    if not images:
        raise SystemExit("Нет снимка. Передайте путь: scripts\\predict_one.py снимок.jpg")
    return images[0]


def _colorize(mask: np.ndarray) -> np.ndarray:
    palette = np.array([
        [0, 0, 0],
        [220, 60, 60],
        [60, 200, 60],
        [60, 60, 220],
        [220, 200, 60],
    ], dtype=np.uint8)
    return palette[mask]


def main() -> None:
    args = _parse_args()
    image_path = args.image or _default_image()
    if not args.weights.is_file():
        raise SystemExit(f"Нет весов {args.weights}")
    if not image_path.is_file():
        raise SystemExit(f"Нет снимка {image_path}")

    segmenter = Segmenter(args.weights, imgsz=args.imgsz)
    mask = segmenter.predict(image_path, conf=args.conf)

    original = np.array(Image.open(image_path).convert("RGB"))
    if mask.shape != original.shape[:2]:
        mask_img = Image.fromarray(mask).resize((original.shape[1], original.shape[0]), Image.NEAREST)
        mask = np.array(mask_img)

    print(f"Снимок: {image_path}")
    print(f"Веса:   {args.weights}")
    total = mask.size
    for class_id, name in CLASS_NAMES.items():
        if class_id == 0:
            continue
        pixels = int((mask == class_id).sum())
        if pixels:
            print(f"  {name:<14} {pixels:>8} px  ({100 * pixels / total:.2f}%)")

    color = _colorize(mask)
    overlay = (0.6 * original + 0.4 * color).astype(np.uint8)
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(PREVIEW)
    print()
    print(f"Картинка с маской: {PREVIEW}")


if __name__ == "__main__":
    main()