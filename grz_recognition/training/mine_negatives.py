"""Жёсткие негативы для класса «не знак» распознавателя.

Детектор прогоняется по реальным фото с полной разметкой номеров
(AUTO.RIA Numberplate Dataset, CC BY 4.0, см. training/import_via.py).
Срабатывания, не пересекающиеся ни с одним размеченным номером, – эмблемы,
надписи, фонари, решётки – выпрямляются так же, как на инференсе, и
сохраняются с классом 4 («не знак») и пустым текстом.

python training/mine_negatives.py --images work/det_real/images --weights weights --out work/ocr_neg
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grz.detector import PlateDetector  # noqa: E402
from grz.rectify import OCR_H, OCR_W, order_quad, rectify  # noqa: E402


def gt_boxes(label_path, W, H):
    boxes = []
    if os.path.exists(label_path):
        for line in open(label_path):
            v = [float(x) for x in line.split()[1:5]]
            cx, cy, w, h = v[0] * W, v[1] * H, v[2] * W, v[3] * H
            boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def overlap(b, boxes):
    x0, y0, x1, y1 = b
    for a0, b0, a1, b1 in boxes:
        iw = max(0, min(x1, a1) - max(x0, a0))
        ih = max(0, min(y1, b1) - max(y0, b0))
        if iw * ih > 0:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="каталог images/ выборки детектора (train/val)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--max", type=int, default=8000)
    args = ap.parse_args()
    det = PlateDetector(os.path.join(args.weights, "detector.onnx"), ["CPUExecutionProvider"], conf=args.conf)
    files = sorted(glob.glob(os.path.join(args.images, "*", "*.jpg")))
    crops = []
    for k, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            continue
        H, W = img.shape[:2]
        lp = f.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)[:-4] + ".txt"
        boxes = gt_boxes(lp, W, H)
        for d in det(img):
            x, y, w, h = d.box
            if overlap((x, y, x + w, y + h), boxes):
                continue
            crops.append(rectify(img, order_quad(d.quad), d.layout == 1))
        if (k + 1) % 200 == 0:
            print(k + 1, len(crops), flush=True)
        if len(crops) >= args.max:
            break
    os.makedirs(args.out, exist_ok=True)
    X = np.lib.format.open_memmap(os.path.join(args.out, "X.npy"), mode="w+", dtype=np.uint8,
                                  shape=(len(crops), OCR_H, OCR_W, 3))
    for i, c in enumerate(crops):
        X[i] = c
    X.flush()
    with open(os.path.join(args.out, "y.tsv"), "w") as f:
        for _ in crops:
            f.write("\t4\t0\n")
    print("negatives:", len(crops))


if __name__ == "__main__":
    main()
