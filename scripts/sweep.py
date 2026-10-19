"""Обучить несколько вариантов U-Net подряд и собрать таблицу для выбора лучшего.

ВАЖНО: запускать на GPU. На CPU один прогон 40 эпох идёт дни, а перебор
из четырёх вариантов — недели. Если GPU нет, используйте этот скрипт с
--epochs 5 только для грубого сравнения, а финальную модель обучайте
обычным scripts/train.py с лучшими найденными параметрами.

Следующий вариант стартует, когда предыдущий записал best.pth. Уже готовый
вариант пропускается, поэтому оборванный цикл можно запустить ещё раз:
он продолжит с того места.

Запуск полного перебора из корня проекта:

    .venv\\Scripts\\python.exe scripts\\sweep.py

Каждый вариант — не случайные числа, а осознанное отклонение от базового
рецепта под нашу задачу «найти патологию на панораме».
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import BEST_MODEL, DATA_DIR, ROOT, train_once

CANDIDATES = ROOT / "weights" / "candidates"
SUMMARY_PATH = CANDIDATES / "summary.csv"
REPORT_PATH = CANDIDATES / "report.md"

# База: resnet34, 512, batch 4, lr 1e-4.
# Остальные строки меняют по одному рычагу, чтобы по таблице было видно,
# что именно помогло.
EXPERIMENTS = (
    {"name": "r34_512_lr1e4", "encoder": "resnet34", "imgsz": 512, "batch": 4, "lr": 1e-4},
    {"name": "r34_512_lr3e4", "encoder": "resnet34", "imgsz": 512, "batch": 4, "lr": 3e-4},
    {"name": "r34_512_lr1e5", "encoder": "resnet34", "imgsz": 512, "batch": 4, "lr": 1e-5},
    {"name": "r34_384_lr1e4", "encoder": "resnet34", "imgsz": 384, "batch": 4, "lr": 1e-4},
    {"name": "mbv2_512_lr1e4", "encoder": "mobilenet_v2", "imgsz": 512, "batch": 8, "lr": 1e-4},
)

COLUMNS = ("name", "encoder", "imgsz", "batch", "lr", "epoch", "val_loss", "weights", "status")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Перебор нескольких настроек обучения U-Net")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--only", default="", help="имя одного варианта, например r34_512_lr1e4")
    parser.add_argument("--force", action="store_true", help="переобучить даже если веса уже есть")
    return parser.parse_args()


def _load_metrics(run_dir: Path) -> dict[str, str]:
    metrics_path = run_dir / "metrics.txt"
    if not metrics_path.is_file():
        raise RuntimeError(f"Нет metrics.txt в {run_dir}")
    data: dict[str, str] = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def _write_summary(records: list[dict[str, str]], *, sweep_done: bool) -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    REPORT_PATH.write_text(_render_report(records, sweep_done=sweep_done), encoding="utf-8")


def _render_report(records: list[dict[str, str]], *, sweep_done: bool) -> str:
    finished = [row for row in records if row["status"] == "ok"]
    failed = [row for row in records if row["status"] != "ok"]
    finished.sort(key=lambda row: float(row["val_loss"]))
    done_names = {row["name"] for row in records}
    pending = [item["name"] for item in EXPERIMENTS if item["name"] not in done_names]

    lines = [
        "# Отчёт обучения U-Net",
        "",
        f"Обновлён: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Цикл завершён." if sweep_done else "Цикл ещё идёт. В таблице только уже законченные варианты.",
        "",
        "Цифры — val_loss с той эпохи, которую train_once сохранил в best.pth. Test здесь не используется.",
        "Маски получены из bbox DENTEX (прямоугольники), поэтому потолок качества ограничен.",
        "",
        "| место | вариант | энкодер | imgsz | batch | lr | эпоха | val_loss | статус |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for place, row in enumerate(finished, start=1):
        lines.append(
            f"| {place} | {row['name']} | {row['encoder']} | {row['imgsz']} | {row['batch']} | "
            f"{row['lr']} | {row['epoch']} | {row['val_loss']} | ok |"
        )
    for row in failed:
        lines.append(
            f"| — | {row['name']} | {row['encoder']} | {row['imgsz']} | {row['batch']} | "
            f"{row['lr']} | — | — | {row['status']} |"
        )
    if pending:
        lines.append("")
        lines.append("Ещё не запускались: " + ", ".join(pending) + ".")

    lines.extend(["", "## Как читать", ""])
    lines.append("- val_loss — кросс-энтропия на valid. Меньше — лучше.")
    lines.append("- val_loss участвовал в выборе эпохи, поэтому это не честная оценка на новых данных.")
    lines.append("- Честная оценка — scripts/evaluate.py на test (Dice/IoU).")
    lines.append("")
    lines.append("Победителя один раз проверьте на test:")
    lines.append("")
    lines.append("```")
    lines.append(r".venv\Scripts\python.exe scripts\evaluate.py")
    lines.append("```")
    lines.append("")
    lines.append("## Файлы каждого прогона")
    lines.append("")
    for row in finished + failed:
        run_dir = ROOT / "runs" / "segment" / row["name"]
        lines.append(f"### {row['name']}")
        lines.append("")
        if row["status"] == "ok":
            lines.append(f"- веса: `{row['weights']}`")
        lines.append(f"- метрики: `{run_dir / 'metrics.txt'}`")
        lines.append("")

    if sweep_done and finished:
        winner = finished[0]
        lines.append("## Выбранный вариант")
        lines.append("")
        lines.append(
            f"По лучшему val_loss выбран `{winner['name']}`. "
            f"Его веса скопированы в `{BEST_MODEL}`."
        )
        lines.append("Это ещё не оценка на test.")
    elif finished:
        lines.append("## Промежуточный лидер")
        lines.append("")
        lines.append(
            f"Сейчас ниже всех val_loss у `{finished[0]['name']}`. "
            f"Файл `{BEST_MODEL}` будет записан, когда цикл дойдёт до конца."
        )
    lines.append("")
    return "\n".join(lines)


def _load_summary() -> list[dict[str, str]]:
    if not SUMMARY_PATH.is_file():
        return []
    with SUMMARY_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _print_table(records: list[dict[str, str]]) -> None:
    finished = [row for row in records if row["status"] == "ok"]
    failed = [row for row in records if row["status"] != "ok"]
    finished.sort(key=lambda row: float(row["val_loss"]))
    print()
    print(f"{'name':<18} {'val_loss':>10}  status")
    for row in finished + failed:
        if row["status"] != "ok":
            print(f"{row['name']:<18} {'':>10}  {row['status']}")
            continue
        print(f"{row['name']:<18} {float(row['val_loss']):10.4f}  ok")
    if not finished:
        return
    winner = finished[0]
    BEST_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winner["weights"], BEST_MODEL)
    print()
    print(f"Лучший по val_loss: {winner['name']}")
    print(f"Его веса скопированы в {BEST_MODEL}")
    print("Это ещё не финальная оценка. Один раз проверьте победителя на test:")
    print(".venv\\Scripts\\python.exe scripts\\evaluate.py")


def main() -> None:
    args = _parse_args()
    if not (DATA_DIR / "train" / "images").is_dir():
        raise SystemExit(
            "Нет data/pathology/train/images. Сначала запустите scripts/prepare_dataset.py"
        )
    if args.device == "cpu":
        print("ВНИМАНИЕ: sweep на CPU идёт очень долго. Используйте GPU или --epochs 5.")

    selected = [item for item in EXPERIMENTS if not args.only or item["name"] == args.only]
    if not selected:
        names = ", ".join(item["name"] for item in EXPERIMENTS)
        raise SystemExit(f"Нет варианта {args.only}. Доступны: {names}")

    records = [row for row in _load_summary() if row["name"] not in {item["name"] for item in selected}]
    for experiment in selected:
        name = experiment["name"]
        destination = CANDIDATES / f"{name}.pth"
        if destination.is_file() and not args.force:
            print(f"Пропускаю {name}: веса уже есть в {destination}")
            previous = next((row for row in _load_summary() if row["name"] == name), None)
            if previous:
                records.append(previous)
            continue

        print(f"Старт {name}: encoder={experiment['encoder']} imgsz={experiment['imgsz']} "
              f"batch={experiment['batch']} lr={experiment['lr']}")
        try:
            best = train_once(
                encoder=experiment["encoder"],
                imgsz=experiment["imgsz"],
                batch=experiment["batch"],
                lr=experiment["lr"],
                epochs=args.epochs,
                patience=args.patience,
                device=args.device,
                workers=args.workers,
                name=name,
            )
            metrics = _load_metrics(best.parent.parent)
            CANDIDATES.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, destination)
            records.append({
                "name": name,
                "encoder": experiment["encoder"],
                "imgsz": str(experiment["imgsz"]),
                "batch": str(experiment["batch"]),
                "lr": str(experiment["lr"]),
                "epoch": metrics.get("best_epoch", ""),
                "val_loss": metrics.get("best_val_loss", ""),
                "weights": str(destination),
                "status": "ok",
            })
        except Exception as error:
            records.append({
                "name": name,
                "encoder": experiment["encoder"],
                "imgsz": str(experiment["imgsz"]),
                "batch": str(experiment["batch"]),
                "lr": str(experiment["lr"]),
                "epoch": "",
                "val_loss": "",
                "weights": "",
                "status": f"error: {error}",
            })
            print(f"{name} остановился с ошибкой: {error}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _write_summary(records, sweep_done=False)

    _print_table(records)
    _write_summary(records, sweep_done=True)
    print(f"Таблица: {SUMMARY_PATH}")
    print(f"Отчёт:   {REPORT_PATH}")


if __name__ == "__main__":
    main()