"""Дообучение детектора YOLO11n-pose (4 угла знака, 2 класса компоновки).

Данные: выход generator.generate_dataset --format yolo (синтетика) и, при
наличии, реальная часть датасета (см. --real-dataset: конвертируется в тот же
формат). Исходные веса yolo11n-pose.pt (COCO-pose, AGPL-3.0) скачиваются
скриптом training/download_assets.sh.

python training/train_detector.py --data work/det/data.yaml --epochs 30 --device 0
"""
import argparse
import csv
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.generate_dataset import yolo_line  # noqa: E402


def add_real(dataset_dir, det_dir, val_every=5):
    """Добавляет реальные изображения датасета в выборку детектора (компоновка по типу/пропорциям)."""
    rows = {}
    with open(os.path.join(dataset_dir, "meta.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if r["is_synthetic"] != "1":
                rows.setdefault(r["image"], []).append(r)
    for k, (rel, rs) in enumerate(sorted(rows.items())):
        img = cv2.imread(os.path.join(dataset_dir, rel))
        if img is None:
            continue
        H, W = img.shape[:2]
        split = "val" if k % val_every == 0 else "train"
        name = "real_" + os.path.splitext(os.path.basename(rel))[0]
        shutil.copy(os.path.join(dataset_dir, rel), os.path.join(det_dir, "images", split, name + os.path.splitext(rel)[1]))
        with open(os.path.join(det_dir, "labels", split, name + ".txt"), "w") as f:
            for r in rs:
                q = np.array(r["quad"].split(","), np.float32).reshape(4, 2)
                wq = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
                hq = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
                two = r["plate_type"] == "type1a" or (r["plate_type"] == "other" and wq / max(hq, 1) < 2.6)
                f.write(yolo_line(int(two), q, W, H, visibility=True) + "\n")
    print("real images added:", len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml от generate_dataset --format yolo")
    ap.add_argument("--weights", default="work/dl/yolo11n-pose.pt")
    ap.add_argument("--real-dataset", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--project", default="work/det_runs")
    ap.add_argument("--name", default="yolo11n_pose_grz")
    args = ap.parse_args()

    if args.real_dataset:
        add_real(args.real_dataset, os.path.dirname(os.path.abspath(args.data)))

    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.train(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device,
        workers=args.workers, fraction=args.fraction, project=os.path.abspath(args.project), name=args.name,
        exist_ok=True, seed=0, deterministic=False, plots=False, patience=50,
        # знак нельзя отражать (текст), остальные аугментации – умеренные
        fliplr=0.0, flipud=0.0, mosaic=1.0, close_mosaic=3, degrees=3.0, translate=0.1, scale=0.5,
        perspective=0.0002, hsv_h=0.01, hsv_s=0.5, hsv_v=0.4, mixup=0.0,
        cos_lr=True, lr0=0.005, warmup_epochs=1, pose=12.0, kobj=1.0,
    )


if __name__ == "__main__":
    main()
