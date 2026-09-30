import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.generate_dataset import META_FIELDS, yolo_line  # noqa: E402
from grz.plate_format import TYPE2IDX  # noqa: E402
from grz.rectify import order_quad  # noqa: E402
from tools.faces import blur_faces  # noqa: E402

TYPES = {"1": "type1", "1a": "type1a", "1b": "type1b", "o": "other"}


def blur_rect(img, x, y, w, h):
    roi = img[y:y + h, x:x + w]
    if roi.size:
        k = max(15, (max(w, h) // 3) | 1)
        img[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--source", default="own_photo")
    ap.add_argument("--license", default="CC BY 4.0")
    args = ap.parse_args()
    out_img = os.path.join(args.dataset, "images", "real")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(os.path.join(args.dataset, "labels"), exist_ok=True)
    meta = os.path.join(args.dataset, "meta.csv")
    new = not os.path.exists(meta)
    done = set()
    if not new:
        with open(meta, encoding="utf-8") as f:
            done = {os.path.basename(r["image"]) for r in csv.DictReader(f, delimiter=";")}

    files = sorted(f for f in os.listdir(args.src) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    for fn in files:
        name = "real_" + os.path.splitext(fn)[0] + ".jpg"
        if name in done:
            continue
        img = cv2.imread(os.path.join(args.src, fn))
        if img is None:
            continue
        print(f"{fn}: автоматически размыто лиц: {blur_faces(img)}")
        scale = min(1.0, 1400 / max(img.shape[:2]))
        clicks, plates_, blur_mode = [], [], []
        state = {"blur": False}

        def on_mouse(ev, x, y, *_):
            if ev == cv2.EVENT_LBUTTONDOWN:
                (blur_mode if state["blur"] else clicks).append((x / scale, y / scale))

        cv2.namedWindow("annotate")
        cv2.setMouseCallback("annotate", on_mouse)
        while True:
            if state["blur"] and len(blur_mode) == 2:
                (x0, y0), (x1, y1) = blur_mode
                blur_rect(img, int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0)))
                blur_mode.clear()
                state["blur"] = False
            vis = cv2.resize(img, None, fx=scale, fy=scale)
            for p in plates_:
                cv2.polylines(vis, [(p["quad"] * scale).astype(np.int32)], True, (0, 255, 0), 2)
            for c in clicks:
                cv2.circle(vis, (int(c[0] * scale), int(c[1] * scale)), 4, (0, 0, 255), -1)
            cv2.imshow("annotate", vis)
            key = cv2.waitKey(30) & 0xFF
            if len(clicks) == 4:
                q = order_quad(np.array(clicks, np.float32))
                clicks.clear()
                num = input("номер: ").strip().upper()
                t = TYPES.get(input("тип [1/1a/1b/o]: ").strip().lower(), "type1")
                veh = input("на ТС? [1/0] (Enter=1): ").strip() or "1"
                cond = input("условия (day,night,rain,snow,dirt,glare,motion_blur,angle): ").strip()
                plates_.append(dict(quad=q, num=num, type=t, veh=veh, cond=cond))
            elif key == ord("u") and plates_:
                plates_.pop()
            elif key == ord("b"):
                state["blur"] = True
            elif key in (ord("n"), ord("q")):
                break
        if plates_:
            cv2.imwrite(os.path.join(out_img, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            H, W = img.shape[:2]
            with open(os.path.join(args.dataset, "labels", os.path.splitext(name)[0] + ".txt"), "w") as f:
                for p in plates_:
                    f.write(yolo_line(TYPE2IDX[p["type"]], p["quad"], W, H) + "\n")
            with open(meta, "a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=META_FIELDS, delimiter=";")
                if new:
                    w.writeheader()
                    new = False
                for p in plates_:
                    q = p["quad"]
                    x0, y0 = q.min(0)
                    x1, y1 = q.max(0)
                    w.writerow({"image": f"images/real/{name}", "plate_num": p["num"], "plate_type": p["type"],
                                "bbox": f"{int(x0)},{int(y0)},{int(x1 - x0)},{int(y1 - y0)}",
                                "quad": ",".join(str(int(round(v))) for v in q.reshape(-1)),
                                "is_vehicle": p["veh"], "is_synthetic": 0, "source": args.source,
                                "license": args.license, "conditions": p["cond"]})
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
