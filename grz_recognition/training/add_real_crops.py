import argparse
import glob
import json
import os
import re
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


# region_id Options-датасета (nomeroff_net CLASS_REGION_ALL)
RU, RU_MILITARY, GARBAGE = "6", "16", "0"
ALNUM = re.compile(r"^[0-9A-Z]{3,12}$")


def whole_quad(img):
    h, w = img.shape[:2]
    return np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])


def options_crops(root, rng, rep_1a=6, max_other=6000, max_garbage=5000, max_ru1=3000):
    """Кропы из Options-датасета -> список (лента 224x48, текст, класс)."""
    from grz.rectify import rectify
    anns = sorted(glob.glob(os.path.join(root, "**", "ann", "*.json"), recursive=True))
    rng.shuffle(anns)
    out, n_other, n_garbage, n_ru1 = [], 0, 0, 0
    for a in anns:
        d = json.load(open(a))
        reg, lines = str(d.get("region_id")), str(d.get("count_lines"))
        text = str(d.get("description", "")).upper()
        img_path = a.replace(os.sep + "ann" + os.sep, os.sep + "img" + os.sep)[:-5] + ".png"
        if reg == GARBAGE or lines == "0":
            if n_garbage >= max_garbage:
                continue
            kind, reps = ("", 4), 1
            n_garbage += 1
        elif lines not in ("1", "2") or not ALNUM.match(text):
            continue
        elif reg == RU and lines == "2" and PLATE_RE["type1a"].match(text):
            kind, reps = (text, 1), rep_1a
        elif reg == RU and lines == "1" and PLATE_RE["type1"].match(text):
            if n_ru1 >= max_ru1:
                continue
            kind, reps = (text, 0), 1
            n_ru1 += 1
        else:
            if n_other >= max_other and not (reg == RU and lines == "2"):
                continue
            kind, reps = (text, 3), 2 if (reg == RU and lines == "2") else 1
            n_other += 1
        img = cv2.imread(img_path)
        if img is None or min(img.shape[:2]) < 12:
            continue
        for _ in range(reps):
            q = whole_quad(img)
            q = q + rng.normal(0, 0.02, q.shape).astype(np.float32) * [img.shape[1], img.shape[0]]
            out.append((rectify(img, q, lines == "2", pad=rng.uniform(-0.02, 0.05)), kind[0], kind[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--options", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n1", type=int, default=30000)
    ap.add_argument("--n1a", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    if args.options:
        crops = options_crops(args.options, rng)
        os.makedirs(args.out, exist_ok=True)
        X = np.lib.format.open_memmap(os.path.join(args.out, "X.npy"), mode="w+", dtype=np.uint8,
                                      shape=(len(crops), OCR_H, OCR_W, 3))
        with open(os.path.join(args.out, "y.tsv"), "w") as f:
            for i, (c, text, t) in enumerate(crops):
                X[i] = c
                f.write(f"{text}\t{t}\t0\n")
        X.flush()
        from collections import Counter
        print("options crops:", len(crops), Counter(t for _, _, t in crops))
        return
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
