import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import scenes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--carparts", default=None)
    ap.add_argument("--coco", default=None)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    pool = scenes.BackgroundPool(args.carparts, args.coco)
    with open(os.path.join(args.out, "labels.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["image", "plate_num", "plate_type", "is_vehicle"])
        for i in range(args.n):
            rng = np.random.default_rng([args.seed, i])
            img, anns, _ = scenes.make_scene(rng, pool)
            name = f"test_{i:04d}.jpg"
            cv2.imwrite(os.path.join(args.out, "images", name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            for a in anns:
                w.writerow([name, a["text"], a["plate_type"], a["is_vehicle"]])


if __name__ == "__main__":
    main()
