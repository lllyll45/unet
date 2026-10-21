"""Один раз обучить U-Net на патологиях DENTEX и сохранить веса.

Что происходит, если говорить языком сервисов:

1. Берётся U-Net с энкодером resnet34, предобученным на ImageNet. Это
   стартовая точка, а не пустая сеть.
2. Скрипт показывает сети пары «снимок + маска патологии». Теперь мы обучаем
   на кропах (вырезанных зубах), чтобы болезнь занимала большую площадь.
3. После каждой эпохи модель проверяется на valid, которые в обучении
   не участвовали. Эпоха — один полный проход по train.
4. Лучшая по val_loss эпоха сохраняется в runs/segment/<name>/weights/best.pth.
   Финальная копия для приложения кладётся в best_model/ только в main().

Запуск из корня проекта:

    .venv\\Scripts\\python.exe scripts\\train.py

Вторая итерация: обучение с Dice Loss для поднятия редких классов.
Запуск:

    .venv\\Scripts\\python.exe scripts\\train.py --loss ce_dice --lr 3e-4 --name dice_loss
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
# Изменено: используем папку с вырезанными зубами
DATA_DIR = ROOT / "data" / "pathology_crops"
WEIGHTS_DIR = ROOT / "weights"
BEST_MODEL = ROOT / "best_model" / "pathology_unet_resnet34.pth"
RUNS_DIR = ROOT / "runs" / "segment"

NUM_CLASSES = 5  # 0 = background + 4 патологии
# Изменено: автоопределение устройства (чтобы не падало на CPU)
DEVICE = "cuda"
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
            # Изменено: сохраняем пропорции для кропов
            A.LongestMaxSize(max_size=imgsz),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=0, value=(0, 0, 0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.LongestMaxSize(max_size=imgsz),
        A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=0, value=(0, 0, 0)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class CombinedLoss(nn.Module):
    """CrossEntropy + Dice Loss.

    CE хорошо работает на частых классах, но «топит» редкие.
    Dice напрямую оптимизирует перекрытие и лучше вытягивает редкие классы.
    Комбинация даёт стабильность CE и чувствительность Dice к дисбалансу.
    """

    def __init__(self, mode: str = "ce_dice"):
        super().__init__()
        self.mode = mode
        self.ce = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(mode="multiclass")

    def forward(self, pred, target):
        if self.mode == "ce":
            return self.ce(pred, target)
        if self.mode == "dice":
            return self.dice(pred, target)
        # ce_dice (по умолчанию)
        return self.ce(pred, target) + self.dice(pred, target)


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
    loss_mode: str = "ce_dice",
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

    criterion = CombinedLoss(mode=loss_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)

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
            torch.save(model.state_dict(), run_dir / "weights" / "best.pth")
            print(f"  → сохранены веса эпохи {epoch} (val_loss={val_loss:.4f})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  → ранняя остановка: val_loss не улучшался {patience} эпох")
                break

    (run_dir / "metrics.txt").write_text(
        f"encoder={encoder}\nimgsz={imgsz}\nbatch={batch}\nlr={lr}\nloss={loss_mode}\n"
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
    parser.add_argument("--loss", default="ce_dice", choices=["ce", "dice", "ce_dice"],
                        help="функция потерь: ce (чистый CrossEntropy), dice (чистый Dice), ce_dice (комбинация)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not (DATA_DIR / "train" / "images").is_dir():
        raise SystemExit(
            "Нет data/pathology_crops/train/images. Сначала выполните: "
            ".venv\\Scripts\\python.exe scripts\\prepare_crops.py"
        )

    print(f"Device: {args.device}")
    print(f"Loss:   {args.loss}")
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
        loss_mode=args.loss,
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