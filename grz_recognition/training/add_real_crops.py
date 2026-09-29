"""Реальные кропы номеров для распознавателя (только обучение, в датасет не входят).

Источник: AUTO.RIA Numberplate OCR RU (Nomeroff Net, https://nomeroff.net.ua/datasets/) –
выровненные кропы однострочных знаков РФ (тип 1) с проверенным текстом.

Из них строятся:
  * type1  – кроп с небольшим случайным полем/поворотом -> лента 224x48;
  * «псевдо-1А» – левая половина ленты «A123», правая – «BC» + регион, как у
    выпрямленного двухстрочного знака (grz/rectify.py). Внешне такая лента мало
    отличается от однострочной, поэтому метка типа для неё не задаётся (-1):
    пример учит только чтение символов (CTC) на реальных текстурах.

python training/add_real_crops.py --src work/nomeroff/ocr_ru --out work/ocr_real --n1 30000 --n1a 10000
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grz.plate_format import PLATE_RE  # noqa: E402
from grz.rectify import OCR_H, OCR_W  # noqa: E402

# Доля ширины однострочного знака 520 мм, где заканчивается «A123».
X_BODY = 0.47


def jitter(img, rng, pad=0.06):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    d = np.float32([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [w * pad, h * pad * 1.5]
    dst = src + d * rng.uniform(0, 1, (4, 1)) + rng.normal(0, 0.012, (4, 2)) * [w, h]
    M = cv2.getPerspectiveTransform(dst.astype(np.float32), src)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def to_strip(img):
    return cv2.resize(img, (OCR_W, OCR_H), interpolation=cv2.INTER_AREA)


def pseudo_1a(img, rng):
    h, w = img.shape[:2]
    a = int(w * (X_BODY + rng.uniform(-0.02, 0.02)))
    y0, y1 = int(h * rng.uniform(0.0, 0.1)), int(h * rng.uniform(0.9, 1.0))
    top = img[y0:y1, :a]
    bottom = img[y0:y1, a:]
    half = OCR_W // 2
    return np.hstack([cv2.resize(top, (half, OCR_H), interpolation=cv2.INTER_AREA),
                      cv2.resize(bottom, (half, OCR_H), interpolation=cv2.INTER_AREA)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n1", type=int, default=30000)
    ap.add_argument("--n1a", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    anns = sorted(glob.glob(os.path.join(args.src, "**", "train", "ann", "*.json"), recursive=True))
    items = []
    for a in anns:
        t = json.load(open(a))["description"]
        if PLATE_RE["type1"].match(t):
            img = a.replace(os.sep + "ann" + os.sep, os.sep + "img" + os.sep)[:-5] + ".png"
            items.append((img, t))
    rng.shuffle(items)
    N = args.n1 + args.n1a
    os.makedirs(args.out, exist_ok=True)
    X = np.lib.format.open_memmap(os.path.join(args.out, "X.npy"), mode="w+", dtype=np.uint8, shape=(N, OCR_H, OCR_W, 3))
    labels = []
    k = 0
    for img_path, text in items:
        if k >= N:
            break
        img = cv2.imread(img_path)
        if img is None or img.shape[1] < 60:
            continue
        img = jitter(img, rng)
        if k < args.n1:
            X[k] = to_strip(img)
            labels.append((text, 0))
        else:
            X[k] = pseudo_1a(img, rng)
            labels.append((text, -1))
        k += 1
    X.flush()
    with open(os.path.join(args.out, "y.tsv"), "w") as f:
        for text, t in labels[:k]:
            f.write(f"{text}\t{t}\t0\n")
    print("crops:", k, "of", len(items))


if __name__ == "__main__":
    main()
