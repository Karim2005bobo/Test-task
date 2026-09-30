import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.generate_dataset import yolo_line  # noqa: E402
from grz.rectify import order_quad  # noqa: E402


def quad_aspect(q):
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return w / max(h, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n_img = n_plate = 0
    for split in ("train", "val"):
        meta = json.load(open(os.path.join(args.src, split, "via_region_data.json")))["_via_img_metadata"]
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)
        for v in meta.values():
            path = os.path.join(args.src, split, v["filename"])
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            lines = []
            for r in v["regions"]:
                sa = r["shape_attributes"]
                if sa.get("name") != "polygon" or len(sa["all_points_x"]) != 4:
                    continue
                q = order_quad(np.stack([sa["all_points_x"], sa["all_points_y"]], 1).astype(np.float32))
                lines.append(yolo_line(int(quad_aspect(q) < 2.6), q, W, H, visibility=True))
            if not lines:
                continue
            stem = "ria_" + os.path.splitext(v["filename"])[0]
            shutil.copy(path, os.path.join(args.out, "images", split, stem + ".jpg"))
            with open(os.path.join(args.out, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            n_img += 1
            n_plate += len(lines)
    print("images", n_img, "plates", n_plate)


if __name__ == "__main__":
    main()
