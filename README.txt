## Установка зависимостей

```bash
pip install -r requirements.txt
```
## Шаг 1: Подготовка датасета + обучение
```bash
python scripts/prepare_dataset.py
python scripts/train.py --epochs 40 --batch 4 --lr 1e-4
```