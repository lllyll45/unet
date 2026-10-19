"""Проверить обученные веса U-Net на test. Сеть при этом не меняется.

Числа считаются на test — снимках, которые не использовались ни для обучения,
ни для выбора лучшей эпохи. Это честная оценка «как модель ведёт себя на новых
плёнках».

Важно: маски в data/pathology получены из bbox DENTEX, то есть это заполненные
прямоугольники, а не точные контуры. Dice и IoU ограничены сверху этой грубой
разметкой: даже идеальная модель не покажет 1.0 на таких масках.

Глазами смотрите картинки в runs/predict/pathology_preview: предсказанная маска
должна попадать на область патологии, а не на фон.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "pathology"
DEFAULT_WEIGHTS = ROOT / "best_model" / "pathology_unet_resnet34.pth"
PREVIEW_DIR = ROOT / "runs" / "predict" / "pathology_preview"

NUM_CLASSES = 5
CLASS_NAMES = {0: "background", 1: "impacted", 2: "caries", 3: "periapical", 4: "deep_caries"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestDataset(Dataset):
    def __init__(self, imgsz: int):
        self.image_dir = DATA_DIR / "test" / "images"
        self.mask_dir = DATA_DIR / "test" / "masks"
        self.files = sorted(self.image_dir.glob("*.jpg"))
        self.transform = A.Compose([
            A.Resize(imgsz, imgsz),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        self.original = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        self.original.append((img_path, image))
        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"].long()


def _dice_iou(pred: np.ndarray, target: np.ndarray, num_classes: int):
    """Dice и IoU по каждому классу 1..num_classes-1. Фон исключён."""
    dice = np.full(num_classes, np.nan)
    iou = np.full(num_classes, np.nan)
    for c in range(1, num_classes):
        pred_c = pred == c
        target_c = target == c
        inter = np.logical_and(pred_c, target_c).sum()
        pred_sum = pred_c.sum()
        target_sum = target_c.sum()
        union = pred_sum + target_sum - inter
        if pred_sum + target_sum > 0:
            dice[c] = 2 * inter / (pred_sum + target_sum)
        if union > 0:
            iou[c] = inter / union
    return dice, iou


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Посчитать Dice/IoU и сохранить превью масок")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--preview", type=int, default=8, help="сколько тестовых снимков нарисовать")
    return parser.parse_args()


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
    if not args.weights.is_file():
        raise SystemExit(f"Нет весов {args.weights}. Сначала запустите scripts/train.py")
    if not (DATA_DIR / "test" / "images").is_dir():
        raise SystemExit("Нет data/pathology/test/images. Сначала запустите scripts/prepare_dataset.py")

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(args.device)
    model.load_state_dict(torch.load(args.weights, map_location=args.device))
    model.eval()

    dataset = TestDataset(imgsz=args.imgsz)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    dice_sum = np.zeros(NUM_CLASSES)
    iou_sum = np.zeros(NUM_CLASSES)
    count = 0

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    previews_left = args.preview

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="test"):
            images = images.to(args.device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            targets = masks.numpy()
            for pred, target in zip(preds, targets):
                dice, iou = _dice_iou(pred, target, NUM_CLASSES)
                dice_sum += np.nan_to_num(dice)
                iou_sum += np.nan_to_num(iou)
                count += 1

    dice_mean = dice_sum / max(count, 1)
    iou_mean = iou_sum / max(count, 1)

    print()
    print("По классам (Dice, IoU):")
    for c in range(1, NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:<14} Dice={dice_mean[c]:.4f}  IoU={iou_mean[c]:.4f}")
    print()
    print(f"macro Dice (без фона): {np.nanmean(dice_mean[1:]):.4f}")
    print(f"macro IoU  (без фона): {np.nanmean(iou_mean[1:]):.4f}")
    print()

    # Превью: оригинал + предсказанная маска (цветная)
    with torch.no_grad():
        for idx, (img_path, original) in enumerate(dataset.original[: args.preview]):
            tensor = dataset.transform(image=original)["image"].unsqueeze(0).to(args.device)
            pred = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            color = _colorize(pred)
            out = (0.6 * original + 0.4 * color).astype(np.uint8)
            Image.fromarray(out).save(PREVIEW_DIR / f"{img_path.stem}_overlay.png")
            previews_left -= 1
            if previews_left <= 0:
                break

    print(f"Превью масок: {PREVIEW_DIR}")


if __name__ == "__main__":
    main()