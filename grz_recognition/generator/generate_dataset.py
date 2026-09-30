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

META_FIELDS = ["image", "plate_num", "plate_type", "bbox", "quad", "is_vehicle",
               "is_synthetic", "source", "license", "conditions"]

_POOL = None
_ARGS = None


def _init(args):
    global _POOL, _ARGS
    _ARGS = args
    cv2.setNumThreads(1)
    _POOL = scenes.BackgroundPool(args.carparts, args.coco)


def bbox_from_quad(q):
    qi = np.round(np.asarray(q, np.float64)).astype(int)
    x0, y0 = qi.min(0)
    x1, y1 = qi.max(0)
    return f"{x0},{y0},{x1 - x0},{y1 - y0}"


def yolo_line(cls, quad, W, H, visibility=False):
  lip(np.asarray(quad, np.float64), 0, [W - 1, H - 1])
    x0, y0 = q.min(0)
    x1, y1 = q.max(0)
    vals = [f"{(x0 + x1) / 2 / W:.6f}", f"{(y0 + y1) / 2 / H:.6f}", f"{(x1 - x0) / W:.6f}", f"{(y1 - y0) / H:.6f}"]
    for x, y in q:
        vals += [f"{x / W:.6f}", f"{y / H:.6f}"] + (["2"] if visibility else [])
    return f"{cls} " + " ".join(vals)


def _one(i):
    args = _ARGS
    rng = np.random.default_rng([args.seed, i])
    for _ in range(20):
        img, anns, cond = scenes.make_scene(rng, _POOL, allow_empty=(args.format == "yolo"))
        if anns or args.format == "yolo":
            break
    H, W = img.shape[:2]
    if args.format == "yolo":
        split = "val" if i % 20 == 0 else "train"
        name = f"s{args.seed}_{i:06d}"
        cv2.imwrite(os.path.join(args.out, "images", split, name + ".jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 95))])
        with open(os.path.join(args.out, "labels", split, name + ".txt"), "w") as f:
            for a in anns:
                f.write(yolo_line(int(a["two_line"]), a["quad"], W, H, visibility=True) + "\n")
        return []
    name = f"syn_{i:06d}"
    rel = f"images/synthetic/{name}.jpg"
    cv2.imwrite(os.path.join(args.out, rel), img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 95))])
    rows = []
    with open(os.path.join(args.out, "labels", name + ".txt"), "w") as f:
        for a in anns:
            q = a["quad"]
            f.write(yolo_line(TYPE2IDX[a["plate_type"]], q, W, H) + "\n")
            x0, y0 = q.min(0)
            x1, y1 = q.max(0)
            rows.append({
                "image": rel,
                "plate_num": a["text"],
                "plate_type": a["plate_type"],
                "bbox": bbox_from_quad(q),
                "quad": ",".join(str(int(round(v))) for v in q.reshape(-1)),
                "is_vehicle": a["is_vehicle"],
                "is_synthetic": 1,
                "source": f"generator.generate_dataset seed={args.seed} idx={i}",
                "license": "CC BY 4.0",
                "conditions": ",".join(cond),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--format", choices=["dataset", "yolo"], default="dataset")
    ap.add_argument("--carparts", default=None, help="каталог Carparts-Seg (только для обучения)")
    ap.add_argument("--coco", default=None, help="каталог COCO128 (только для обучения)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    if args.format == "yolo":
        for s in ("train", "val"):
            os.makedirs(os.path.join(args.out, "images", s), exist_ok=True)
            os.makedirs(os.path.join(args.out, "labels", s), exist_ok=True)
    else:
        if args.carparts or args.coco:
            print("ВНИМАНИЕ: сторонние фоны в сдаваемом датасете нарушают лицензию CC BY 4.0", file=sys.stderr)
        os.makedirs(os.path.join(args.out, "images", "synthetic"), exist_ok=True)
        os.makedirs(os.path.join(args.out, "images", "real"), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)

    idx = range(args.start, args.start + args.n)
    with Pool(args.workers, initializer=_init, initargs=(args,)) as p:
        results = []
        for k, r in enumerate(p.imap(_one, idx, chunksize=8)):
            results.append(r)
            if (k + 1) % 500 == 0:
                print(f"{k + 1}/{args.n}", flush=True)

    if args.format == "yolo":
        with open(os.path.join(args.out, "data.yaml"), "w") as f:
            f.write(f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\n"
                    "kpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\nnames:\n  0: plate_1line\n  1: plate_2line\n")
        return

    meta = os.path.join(args.out, "meta.csv")
    old = []
    if os.path.exists(meta):
        with open(meta, encoding="utf-8") as f:
            old = [r for r in csv.DictReader(f, delimiter=";") if r["is_synthetic"] != "1"]
    with open(meta, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS, delimiter=";")
        w.writeheader()
        w.writerows(old)
        for rows in results:
            w.writerows(rows)


if __name__ == "__main__":
    main()
