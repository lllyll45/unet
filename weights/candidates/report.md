# Отчёт обучения U-Net

Обновлён: 2026-10-20 02:09:59

Цикл завершён.

Цифры — val_loss с той эпохи, которую train_once сохранил в best.pth. Test здесь не используется.
Маски получены из bbox DENTEX (прямоугольники), поэтому потолок качества ограничен.

| место | вариант | энкодер | imgsz | batch | lr | эпоха | val_loss | статус |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | r34_512_lr3e4 | resnet34 | 512 | 4 | 0.0003 | 14 | 0.146576 | ok |
| 2 | r34_512_lr1e4 | resnet34 | 512 | 4 | 0.0001 | 9 | 0.156969 | ok |
| 3 | r34_384_lr1e4 | resnet34 | 384 | 4 | 0.0001 | 11 | 0.158280 | ok |
| 4 | mbv2_512_lr1e4 | mobilenet_v2 | 512 | 4 | 0.0001 | 14 | 0.159757 | ok |
| 5 | r34_512_lr1e5 | resnet34 | 512 | 4 | 1e-05 | 40 | 0.309818 | ok |

## Как читать

- val_loss — кросс-энтропия на valid. Меньше — лучше.
- val_loss участвовал в выборе эпохи, поэтому это не честная оценка на новых данных.
- Честная оценка — scripts/evaluate.py на test (Dice/IoU).

Победителя один раз проверьте на test:

```
.venv\Scripts\python.exe scripts\evaluate.py
```

## Файлы каждого прогона

### r34_512_lr3e4

- веса: `E:\мои документы\Питон проекты\Univer\RO\tooth_detector\unet\weights\candidates\r34_512_lr3e4.pth`
- метрики: `E:\мои документы\Питон проекты\Univer\RO\tooth_detector\unet\runs\segment\r34_512_lr3e4\metrics.txt`

### r34_512_lr1e4

- веса: `E:\мои документы\Питон проекты\Univer\RO\tooth_detector\unet\weights\candidates\r34_512_lr1e4.pth`
- метрики: `E:\мои документы\Питон проекты\Univer\RO\tooth_detector\unet\runs\segment\r34_512_lr1e4\metrics.txt`

### r34_384_lr1e4

- веса: `E:\мои документы\Питон проекты\Univer\RO\tooth_detector\unet\weights\candidates\r34_384_lr1e4.pth`
- метрики: `E:\мои документы\Питон проекты\Univer\RO\tooth_detector\unet\runs\segment\r34_384_lr1e4\metrics.txt`

### mbv2_512_lr1e4

- веса: `E:\мои документы\Питон проекты\Univer\RO\tooth_detector\unet\weights\candidates\mbv2_512_lr1e4.pth`
- метрики: `E:\мои документы\Питон проекты\Univer\RO\tooth_detector\unet\runs\segment\mbv2_512_lr1e4\metrics.txt`

### r34_512_lr1e5

- веса: `E:\мои документы\Питон проекты\Univer\RO\tooth_detector\unet\weights\candidates\r34_512_lr1e5.pth`
- метрики: `E:\мои документы\Питон проекты\Univer\RO\tooth_detector\unet\runs\segment\r34_512_lr1e5\metrics.txt`

## Выбранный вариант

По лучшему val_loss выбран `r34_512_lr3e4`. Его веса скопированы в `E:\мои документы\Питон проекты\Univer\RO\tooth_detector\unet\best_model\pathology_unet_resnet34.pth`.
Это ещё не оценка на test.
