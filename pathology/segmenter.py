"""Загрузка уже обученных весов U-Net и сегментация патологий на снимке.

Приложение вызывает только этот класс. Скрипт обучения к запуску сервиса
не относится: веса читаются с диска так же, как конфиг или файл модели.
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

NUM_CLASSES = 5
CLASS_NAMES = {0: "background", 1: "impacted", 2: "caries", 3: "periapical", 4: "deep_caries"}


class Segmenter:
    """Ищет на панорамном снимке маску патологий.

    Модель обучена на 5 классах: 0 — фон, 1 — impacted, 2 — caries,
    3 — periapical, 4 — deep_caries. Возвращает маску HxW с этими id.
    """

    def __init__(self, weights_path: str | Path, encoder: str = "resnet34", imgsz: int = 512) -> None:
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Файл весов не найден: {path}. "
                "Готовые веса лежат в best_model/pathology_unet_resnet34.pth."
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.imgsz = imgsz
        self.transform = A.Compose([
            A.Resize(imgsz, imgsz),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        ).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def predict_proba(self, image_path: str | Path) -> np.ndarray:
        """Вернуть HxWx5 вероятности softmax в исходном разрешении снимка."""
        image = Path(image_path)
        if not image.is_file():
            raise FileNotFoundError(f"Снимок не найден: {image}")
        original = np.array(Image.open(image).convert("RGB"))
        tensor = self.transform(image=original)["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        probs_img = np.stack([
            np.array(Image.fromarray(probs[c]).resize(
                (original.shape[1], original.shape[0]), Image.BILINEAR
            ))
            for c in range(NUM_CLASSES)
        ], axis=-1)
        return probs_img

    def predict(self, image_path: str | Path, conf: float = 0.5) -> np.ndarray:
        """Вернуть маску HxW uint8: 0=фон, 1..4=классы патологий.

        conf — порог максимальной вероятности. Если сеть не уверена хотя бы
        на conf, пиксель считается фоном. Это убирает размазанные края.
        """
        probs = self.predict_proba(image_path)
        mask = probs.argmax(axis=-1).astype(np.uint8)
        max_prob = probs.max(axis=-1)
        mask[max_prob < conf] = 0
        return mask

    def predict_lesions(
        self,
        image_path: str | Path,
        conf: float = 0.5,
        min_area: int = 50,
    ) -> list[dict]:
        """Вернуть список областей-патологий с координатами и классом.

        Это то, что нужно агрегатору: маска HxW разбивается на связные
        компоненты (blobs), и каждая область описывается словарём:

            class_id    — 1..4
            class_name  — impacted / caries / periapical / deep_caries
            bbox        — [x, y, w, h] в пикселях исходного снимка
            area        — площадь области в пикселях
            centroid    — [cx, cy] центр области
            mean_prob   — средняя вероятность класса по пикселям области
            mask        — bool HxW, точная форма области (для overlap-проверки)

        min_area отбрасывает мелкие шумовые пятна: на 512x512 область
        меньше 50 px — это, как правило, артефакт, а не патология.
        """
        image = Path(image_path)
        probs = self.predict_proba(image)
        mask = probs.argmax(axis=-1).astype(np.uint8)
        max_prob = probs.max(axis=-1)
        mask[max_prob < conf] = 0

        lesions: list[dict] = []
        for class_id in range(1, NUM_CLASSES):
            binary = (mask == class_id).astype(np.uint8)
            if binary.sum() == 0:
                continue
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, n):  # 0 — фон
                x, y, w, h, area = (int(v) for v in stats[i])
                if area < min_area:
                    continue
                component_mask = labels == i
                lesions.append({
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "bbox": [x, y, w, h],
                    "area": area,
                    "centroid": [float(centroids[i][0]), float(centroids[i][1])],
                    "mean_prob": float(probs[..., class_id][component_mask].mean()),
                    "mask": component_mask,
                })
        return lesions