import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.generate_dataset import META_FIELDS, yolo_line  # noqa: E402
from grz.plate_format import PLATE_RE, TYPE2IDX  # noqa: E402

RU, RU_MILITARY = "6", "16"
ALNUM = re.compile(r"^[0-9A-Z]{3,12}$")
OPT_SRC = "AUTO.RIA Numberplate Options Dataset 2021-09-03 (Nomeroff Net), https://nomeroff.net.ua/datasets/"
EU_SRC = "EU License Plates Images, https://zenodo.org/records/3967850"


def day_night(img):
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2].mean()
    return "night" if v < 60 else "day"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--options", required=True)
    ap.add_argument("--eu", default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n1a", type=int, default=300)
    ap.add_argument("--n1", type=int, default=60)
    ap.add_argument("--n-other-ru2", type=int, default=80)
    ap.add_argument("--n-other-foreign", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    real_dir = os.path.join(args.dataset, "images", "real")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dataset, "labels"), exist_ok=True)

    anns = sorted(glob.glob(os.path.join(args.options, "**", "ann", "*.json"), recursive=True))
    rng.shuffle(anns)
    buckets = {"type1a": [], "type1": [], "other_ru2": [], "other_foreign": []}
    for a in anns:
        d = json.load(open(a))
        reg, lines, text = str(d.get("region_id")), str(d.get("count_lines")), str(d.get("description", "")).upper()
        if lines not in ("1", "2") or not ALNUM.match(text):
            continue
        img = a.replace(os.sep + "ann" + os.sep, os.sep + "img" + os.sep)[:-5] + ".png"
        if reg == RU and lines == "2" and PLATE_RE["type1a"].match(text):
            buckets["type1a"].append((img, text, "type1a"))
        elif reg == RU and lines == "1" and PLATE_RE["type1"].match(text):
            buckets["type1"].append((img, text, "type1"))
        elif reg == RU and lines == "2":
            buckets["other_ru2"].append((img, text, "other"))
        elif reg not in (RU, "0"):
            buckets["other_foreign"].append((img, text, "other"))
    limits = {"type1a": args.n1a, "type1": args.n1, "other_ru2": args.n_other_ru2,
              "other_foreign": args.n_other_foreign}

    meta = os.path.join(args.dataset, "meta.csv")
    rows_old = []
    if os.path.exists(meta):
        with open(meta, encoding="utf-8") as f:
            rows_old = list(csv.DictReader(f, delimiter=";"))
    seen_hash, seen_plate = set(), set()
    new_rows = []

    def add(src_path, text, ptype, source):
        img = cv2.imread(src_path)
        if img is None or min(img.shape[:2]) < 20:
            return False
        h = hashlib.md5(open(src_path, "rb").read()).hexdigest()
        if h in seen_hash or (ptype != "other" and text in seen_plate):
            return False
        seen_hash.add(h)
        seen_plate.add(text)
        name = f"real_{len(new_rows):05d}.jpg"
        cv2.imwrite(os.path.join(real_dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        H, W = img.shape[:2]
        q = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
        with open(os.path.join(args.dataset, "labels", name[:-4] + ".txt"), "w") as f:
            f.write(yolo_line(TYPE2IDX[ptype], q, W, H) + "\n")
        new_rows.append({"image": f"images/real/{name}", "plate_num": text, "plate_type": ptype,
                         "bbox": f"0,0,{W - 1},{H - 1}", "quad": f"0,0,{W - 1},0,{W - 1},{H - 1},0,{H - 1}",
                         "is_vehicle": 1, "is_synthetic": 0,
                         "source": source + f" | {os.path.basename(src_path)} | conditions: оценка по яркости",
                         "license": "CC BY 4.0", "conditions": day_night(img)})
        return True

    stats = {}
    for key, items in buckets.items():
        n = 0
        for img, text, ptype in items:
            if n >= limits[key]:
                break
            n += add(img, text, ptype, OPT_SRC)
        stats[key] = n
    if args.eu:
        n = 0
        for p in sorted(glob.glob(os.path.join(args.eu, "**", "*.jpg"), recursive=True)):
            text = re.sub(r"[^0-9A-Z]", "", os.path.splitext(os.path.basename(p))[0].upper())
            n += add(p, text if ALNUM.match(text) else "", "other", EU_SRC)
        stats["eu"] = n

    with open(meta, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS, delimiter=";")
        w.writeheader()
        w.writerows([r for r in rows_old if not r["image"].startswith("images/real/real_")])
        w.writerows(new_rows)
    print(stats, "всего:", len(new_rows))


if __name__ == "__main__":
    main()
