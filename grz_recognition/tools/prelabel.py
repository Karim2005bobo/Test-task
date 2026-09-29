"""Предразметка реальных фото для ручной проверки.

Пайплайн находит знаки (4 угла + текст + тип). Для каждой находки
сохраняется запись в JSONL и кроп с контекстом, собранный в листы-сетки с
номерами. Затем человек просматривает листы, исправляет текст/тип или
отбраковывает находку (tools/apply_review.py). В meta.csv такие строки
помечаются в поле source как предразмеченные собственной моделью и
проверенные вручную – это не «автоматическая разметка, выданная за ручную».

python tools/prelabel.py --images work/commons/img --weights weights --out work/prelabel
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grz.pipeline import Pipeline, imread, yellow_fraction  # noqa: E402
from grz.rectify import rectify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--det-size", type=int, default=960)
    ap.add_argument("--min-width", type=int, default=40, help="минимальная ширина знака, px")
    ap.add_argument("--per-sheet", type=int, default=30)
    ap.add_argument("--skip", nargs="*", default=[], help="prelabel.jsonl уже обработанных партий")
    ap.add_argument("--id-offset", type=int, default=0, help="начальный номер находок (уникальность между партиями)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pipe = Pipeline(args.weights, "cpu", args.det_size, det_conf=0.25, min_conf=0.0, target_conf=0.0)
    recs, tiles = [], []
    done = set()
    for sp in args.skip:
        done |= {json.loads(line)["image"] for line in open(sp, encoding="utf-8")}
        done |= {line.strip() for line in open(sp[:-len(".jsonl")] + ".files", encoding="utf-8")} \
            if os.path.exists(sp[:-len(".jsonl")] + ".files") else set()
    files = sorted(f for f in os.listdir(args.images)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")) and f not in done)
    with open(os.path.join(args.out, "prelabel.files"), "w", encoding="utf-8") as f:
        f.write("\n".join(files) + "\n")
    for k, name in enumerate(files):
        img = imread(os.path.join(args.images, name))
        if img is None:
            continue
        for r in pipe(img):
            q = r.quad
            w = np.linalg.norm(q[1] - q[0])
            if w < args.min_width:
                continue
            two = (np.linalg.norm(q[1] - q[0]) / max(np.linalg.norm(q[3] - q[0]), 1)) < 2.6
            strip = rectify(img, q, two)
            rid = args.id_offset + len(recs)
            recs.append({"id": rid, "image": name, "quad": q.round(1).tolist(), "text": r.plate_num,
                         "type": r.plate_type, "conf": round(r.confidence, 3),
                         "yellow": round(yellow_fraction(strip), 3), "width": round(float(w), 1),
                         "H": img.shape[0], "W": img.shape[1]})
            x0, y0 = q.min(0).astype(int)
            x1, y1 = q.max(0).astype(int)
            pw, ph = int((x1 - x0) * 0.5), int((y1 - y0) * 1.0)
            ctx = img[max(0, y0 - ph):y1 + ph, max(0, x0 - pw):x1 + pw]
            ctx = cv2.resize(ctx, (224, 112))
            tile = np.vstack([ctx, strip, np.full((22, 224, 3), 255, np.uint8)])
            cv2.putText(tile, f"#{rid} {r.plate_num} {r.plate_type[-2:]}", (2, 178), 0, 0.45, (0, 0, 200), 1)
            tiles.append(tile)
        if (k + 1) % 100 == 0:
            print(k + 1, len(recs), flush=True)
    with open(os.path.join(args.out, "prelabel.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s in range(0, len(tiles), args.per_sheet):
        chunk = tiles[s:s + args.per_sheet]
        while len(chunk) % 5:
            chunk.append(np.full_like(tiles[0], 255))
        sheet = np.vstack([np.hstack(chunk[i:i + 5]) for i in range(0, len(chunk), 5)])
        cv2.imwrite(os.path.join(args.out, f"sheet_{s // args.per_sheet:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("найдено знаков:", len(recs))


if __name__ == "__main__":
    main()
