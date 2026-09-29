"""Перенос вручную проверенных находок (tools/prelabel.py) в реальную часть датасета.

review.json – список проверенных знаков:
  [{"id": 418, "text": "AE79277", "type": "type1b"}, ...]
id – номер находки в prelabel.jsonl (углы берутся оттуда), text/type –
значения после ручной проверки (нечитаемые позиции – '#').

Для каждого изображения: лица размываются (YuNet, tools/faces.py),
изображение копируется в images/real/, пишутся labels/*.txt и строки meta.csv
с источником (страница, автор) и лицензией из candidates.jsonl.

python tools/apply_review.py --review review.json --prelabel work/prelabel/prelabel.jsonl \
    --candidates work/commons/candidates.jsonl --images work/commons/img --dataset dataset
"""
import argparse
import csv
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.generate_dataset import META_FIELDS, yolo_line  # noqa: E402
from grz.plate_format import TYPE2IDX  # noqa: E402
from grz.rectify import order_quad  # noqa: E402
from tools.faces import blur_faces  # noqa: E402

SPDX = {"CC0": "CC0-1.0", "Public domain": "Public domain"}


def spdx(lic):
    lic = lic.strip()
    if lic in SPDX:
        return SPDX[lic]
    m = re.match(r"CC BY (\d\.\d)", lic)
    return f"CC-BY-{m.group(1)}" if m else lic


def file_name(c):
    if c.get("id"):
        return "ov_" + c["id"] + ".jpg"
    return "wc_" + re.sub(r"[^\w.-]+", "_", c["title"][5:])[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--prelabel", required=True)
    ap.add_argument("--candidates", required=True, nargs="+")
    ap.add_argument("--images", required=True, nargs="+")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--prefix", default="real_web_")
    args = ap.parse_args()
    pre = {r["id"]: r for r in map(json.loads, open(args.prelabel, encoding="utf-8"))}
    cands = {}
    for cf in args.candidates:
        for line in open(cf, encoding="utf-8"):
            c = json.loads(line)
            cands[file_name(c)] = c
    review = json.load(open(args.review, encoding="utf-8"))
    by_img = {}
    for rv in review:
        p = pre[rv["id"]]
        by_img.setdefault(p["image"], []).append((p, rv))

    meta = os.path.join(args.dataset, "meta.csv")
    with open(meta, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter=";") if not r["image"].startswith(f"images/real/{args.prefix}")]
    new = []
    faces = 0
    for k, (src, items) in enumerate(sorted(by_img.items())):
        path = next(os.path.join(d, src) for d in args.images if os.path.exists(os.path.join(d, src)))
        img = cv2.imread(path)
        faces += blur_faces(img)
        H, W = img.shape[:2]
        name = f"{args.prefix}{k:04d}.jpg"
        cv2.imwrite(os.path.join(args.dataset, "images", "real", name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        c = cands.get(src, {})
        v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2].mean()
        lines = []
        for p, rv in items:
            q = order_quad(np.array(p["quad"], np.float32))
            q = np.clip(q, 0, [W - 1, H - 1])
            lines.append(yolo_line(TYPE2IDX[rv["type"]], q, W, H))
            x0, y0 = q.min(0)
            x1, y1 = q.max(0)
            new.append({
                "image": f"images/real/{name}", "plate_num": rv["text"], "plate_type": rv["type"],
                "bbox": f"{int(round(x0))},{int(round(y0))},{int(round(x1 - x0))},{int(round(y1 - y0))}",
                "quad": ",".join(str(int(round(t))) for t in q.reshape(-1)),
                "is_vehicle": rv.get("is_vehicle", 1), "is_synthetic": 0,
                "source": f"{c.get('page', src)} (автор: {c.get('artist', '?')[:80]}) | разметка: предразметка "
                          f"собственной моделью, текст/тип/углы проверены и исправлены вручную",
                "license": spdx(c.get("license", "")),
                "conditions": ",".join(["night" if v < 60 else "day"] + rv.get("conditions", [])),
            })
        with open(os.path.join(args.dataset, "labels", name[:-4] + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    with open(meta, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS, delimiter=";")
        w.writeheader()
        w.writerows(rows)
        w.writerows(new)
    print("изображений:", len(by_img), "знаков:", len(new), "размыто лиц:", faces)


if __name__ == "__main__":
    main()
