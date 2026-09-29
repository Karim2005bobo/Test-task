"""Подготовка выборки для распознавателя: выпрямленные кропы 224x48.

Синтетика генерируется тем же кодом, что и сцены (generator.scenes), с шумом
углов, имитирующим детектор. Реальные знаки (dataset/meta.csv, is_synthetic=0)
вырезаются по размеченным углам и повторяются --real-repeat раз с разным шумом.

Результат: <out>/X.npy (uint8, N x 48 x 224 x 3) и <out>/y.tsv (текст, класс).
Класс: 0..3 – PLATE_TYPES, 4 – «не знак».
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import scenes  # noqa: E402
from grz.plate_format import TYPE2IDX  # noqa: E402
from grz.rectify import OCR_H, OCR_W, rectify  # noqa: E402

TYPE_PROBS = {"type1": 0.28, "type1a": 0.30, "type1b": 0.27, "other": 0.15}
NONE_FRAC = 0.06
_POOL, _SEED = None, 0


def _init(carparts, coco, seed):
    global _POOL, _SEED
    cv2.setNumThreads(1)
    _POOL, _SEED = scenes.BackgroundPool(carparts, coco), seed


def _one(i):
    rng = np.random.default_rng([_SEED, i])
    if rng.random() < NONE_FRAC:
        return scenes.make_none_sample(rng, _POOL), "", 4
    crop, plate = scenes.make_ocr_sample(rng, _POOL, type_probs=TYPE_PROBS)
    return crop, plate.text, TYPE2IDX[plate.plate_type]


def is_holdout(image, mod):
    """Детерминированный отложенный тест: каждое mod-е изображение по хешу имени."""
    import hashlib
    return mod > 0 and int(hashlib.md5(image.encode()).hexdigest(), 16) % mod == 0


def real_crops(dataset_dir, repeat, seed, holdout_mod=0):
    """Кропы реальных знаков по разметке meta.csv (с шумом углов)."""
    rng = np.random.default_rng(seed)
    meta = os.path.join(dataset_dir, "meta.csv")
    out = []
    with open(meta, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if r["is_synthetic"] == "1" or "#" in r["plate_num"] or is_holdout(r["image"], holdout_mod):
                continue
            img = cv2.imread(os.path.join(dataset_dir, r["image"]))
            if img is None:
                continue
            q = np.array(r["quad"].split(","), np.float32).reshape(4, 2)
            wq = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
            hq = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
            two = r["plate_type"] == "type1a" or (r["plate_type"] == "other" and wq / max(hq, 1) < 2.6)
            diag = np.linalg.norm(q[2] - q[0])
            for k in range(repeat):
                nq = q + (rng.normal(0, 0.012 * diag, q.shape) if k else 0)
                out.append((rectify(img, nq, two, pad=rng.uniform(0, 0.05)), r["plate_num"], TYPE2IDX[r["plate_type"]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--carparts", default=None)
    ap.add_argument("--coco", default=None)
    ap.add_argument("--real-dataset", default=None, help="каталог датасета с реальными знаками")
    ap.add_argument("--real-repeat", type=int, default=20)
    ap.add_argument("--holdout-mod", type=int, default=5, help="каждое N-е реальное изображение – отложенный тест")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    real = real_crops(args.real_dataset, args.real_repeat, args.seed, args.holdout_mod) if args.real_dataset else []
    N = args.n + len(real)
    X = np.lib.format.open_memmap(os.path.join(args.out, "X.npy"), mode="w+", dtype=np.uint8,
                                  shape=(N, OCR_H, OCR_W, 3))
    labels = []
    with Pool(args.workers, initializer=_init, initargs=(args.carparts, args.coco, args.seed)) as p:
        for i, (crop, text, t) in enumerate(p.imap(_one, range(args.n), chunksize=64)):
            X[i] = crop
            labels.append((text, t, 1))
            if (i + 1) % 10000 == 0:
                print(f"{i + 1}/{args.n}", flush=True)
    for j, (crop, text, t) in enumerate(real):
        X[args.n + j] = crop
        labels.append((text, t, 0))
    X.flush()
    with open(os.path.join(args.out, "y.tsv"), "w") as f:
        for text, t, syn in labels:
            f.write(f"{text}\t{t}\t{syn}\n")
    print("real crops:", len(real), "total:", N)


if __name__ == "__main__":
    main()
