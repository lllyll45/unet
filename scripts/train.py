"""Один раз обучить U-Net на патологиях DENTEX и сохранить веса.

Что происходит, если говорить языком сервисов:

1. Берётся U-Net с энкодером resnet34, предобученным на ImageNet. Это
   стартовая точка, а не пустая сеть.
2. Скрипт показывает сети пары «снимок + маска патологии» из
   data/pathology/train. Маски получены из bbox DENTEX: это заполненные
   прямоугольники, а не точные контуры. Верхний предел качества ограничен
   этой грубой разметкой.
3. После каждой эпохи модель проверяется на data/pathology/valid, которые
   в обучении не участвовали. Эпоха — один полный проход по train.
4. Лучшая по val_loss эпоха сохраняется в runs/segment/<name>/weights/best.pth.
   Финальная копия для приложения кладётся в best_model/ только в main().

Запуск из корня проекта:

    .venv\\Scripts\\python.exe scripts\\train.py

На CPU (без NVIDIA GPU) обучение идёт очень долго. Для realistic-результата
нужна видеокарта или облако. Если видеопамяти мало, уменьшите --batch.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "pathology"
WEIGHTS_DIR = ROOT / "weights"
BEST_MODEL = ROOT / "best_model" / "pathology_unet_resnet34.pth"
RUNS_DIR = ROOT / "runs" / "segment"

NUM_CLASSES = 5  # 0 = background + 4 патологии
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = {0: "background", 1: "impacted", 2: "caries", 3: "periapical", 4: "deep_caries"}


class PathologyDataset(Dataset):
    def __init__(self, split: str, transform=None):
        self.image_dir = DATA_DIR / split / "images"
        self.mask_dir = DATA_DIR / split / "masks"
        self.files = sorted(self.image_dir.glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()

        return image, mask


def get_transforms(train: bool = True, imgsz: int = 512):
    if train:
        return A.Compose([
            A.Resize(imgsz, imgsz),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(imgsz, imgsz),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def train_once(
    *,
    encoder: str,
    imgsz: int,
    batch: int,
    lr: float,
    epochs: int,
    patience: int,
    device: str,
    workers: int,
    name: str,
) -> Path:
    """Обучить одну модель. Вернуть путь к best.pth этого прогона.

    Веса пишутся только в runs/segment/<name>/. Копирование в best_model/
    делает вызывающий код (main или sweep), чтобы промежуточный прогон
    sweep не затёр финальные веса.
    """
    train_ds = PathologyDataset("train", transform=get_transforms(train=True, imgsz=imgsz))
    val_ds = PathologyDataset("valid", transform=get_transforms(train=False, imgsz=imgsz))

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers)

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "weights" / "best.pth" if (run_dir / "weights").exists()
                       else (run_dir.mkdir(parents=True, exist_ok=True) or (run_dir / "weights").mkdir(exist_ok=True) or run_dir / "weights" / "best.pth"))
            print(f"  → сохранены веса эпохи {epoch} (val_loss={val_loss:.4f})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  → ранняя остановка: val_loss не улучшался {patience} эпох")
                break

    (run_dir / "metrics.txt").write_text(
        f"encoder={encoder}\nimgsz={imgsz}\nbatch={batch}\nlr={lr}\n"
        f"best_epoch={best_epoch}\nbest_val_loss={best_val_loss:.6f}\n",
        encoding="utf-8",
    )
    return run_dir / "weights" / "best.pth"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучить U-Net на патологиях DENTEX")
    parser.add_argument("--encoder", default="resnet34", help="энкодер segmentation_models_pytorch")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10, help="ранняя остановка, если val_loss не падает")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="pathology", help="имя папки прогона в runs/segment")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not (DATA_DIR / "train" / "images").is_dir():
        raise SystemExit(
            "Нет data/pathology/train/images. Сначала выполните: "
            ".venv\\Scripts\\python.exe scripts\\prepare_dataset.py"
        )

    print(f"Device: {args.device}")
    best = train_once(
        encoder=args.encoder,
        imgsz=args.imgsz,
        batch=args.batch,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        name=args.name,
    )
    if not best.is_file():
        raise SystemExit(f"Обучение закончилось, но файл весов не найден: {best}")

    BEST_MODEL.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, BEST_MODEL)
    shutil.copy2(best, WEIGHTS_DIR / "pathology_best.pth")

    print()
    print("Обучение закончено.")
    print(f"Веса для приложения: {BEST_MODEL}")
    print(f"Полный отчёт прогона: {best.parent.parent}")
    print("Дальше качество смотрите так: .venv\\Scripts\\python.exe scripts\\evaluate.py")


if __name__ == "__main__":
    main()