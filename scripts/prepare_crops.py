"""Нарезка панорамных снимков на отдельные зубы с помощью YOLO."""

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "pathology"
TARGET_DIR = ROOT / "data" / "pathology_crops"
YOLO_WEIGHTS = ROOT / "weights" / "yolo_teeth.pt"

PADDING = 0.1  # 10% отступа вокруг зуба

def process_split(split: str, model: YOLO):
    print(f"\nОбработка {split}...")
    img_dir = SOURCE_DIR / split / "images"
    mask_dir = SOURCE_DIR / split / "masks"
    
    if not img_dir.exists():
        print(f"  Папка {img_dir} не найдена, пропускаю.")
        return

    out_img_dir = TARGET_DIR / split / "images"
    out_mask_dir = TARGET_DIR / split / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(img_dir.glob("*.jpg"))
    total_crops = 0

    for img_path in tqdm(files, desc=split):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h_img, w_img = image.shape[:2]

        # Загружаем маску (если есть)
        mask_path = mask_dir / (img_path.stem + ".png")
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((h_img, w_img), dtype=np.uint8)

        # Ищем зубы через YOLO
        results = model.predict(str(img_path), conf=0.25, imgsz=1280, verbose=False)
        
        # Проверяем, нашла ли YOLO что-то
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            continue

        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Добавляем отступ
            w, h = x2 - x1, y2 - y1
            pad_w, pad_h = int(w * PADDING), int(h * PADDING)
            
            x1 = max(0, int(x1 - pad_w))
            y1 = max(0, int(y1 - pad_h))
            x2 = min(w_img, int(x2 + pad_w))
            y2 = min(h_img, int(y2 + pad_h))

            if x2 <= x1 or y2 <= y1:
                continue

            # Вырезаем
            crop_img = image[y1:y2, x1:x2]
            crop_mask = mask[y1:y2, x1:x2]

            # Сохраняем
            crop_name = f"{img_path.stem}_tooth{i}"
            cv2.imwrite(str(out_img_dir / f"{crop_name}.jpg"), crop_img)
            cv2.imwrite(str(out_mask_dir / f"{crop_name}.png"), crop_mask)
            total_crops += 1

    print(f"  {split}: получено {total_crops} кропов")

def main():
    if not YOLO_WEIGHTS.is_file():
        raise SystemExit(f"ОШИБКА: Нет весов YOLO по пути {YOLO_WEIGHTS}")
    
    print(f"Загрузка YOLO из {YOLO_WEIGHTS}...")
    model = YOLO(str(YOLO_WEIGHTS))
    
    for split in ("train", "valid", "test"):
        if (SOURCE_DIR / split / "images").exists():
            process_split(split, model)
        else:
            print(f"Папка {split} не найдена.")
    
    print(f"\nГотово! Вырезанные зубы сохранены в {TARGET_DIR}")

if __name__ == "__main__":
    main()