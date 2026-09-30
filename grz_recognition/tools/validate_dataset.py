import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

FIELDS = ["image", "plate_num", "plate_type", "bbox", "quad", "is_vehicle", "is_synthetic", "source", "license", "conditions"]
TYPES = {"type1", "type1a", "type1b", "other"}
CONDITIONS = {"day", "night", "rain", "snow", "dirt", "glare", "motion_blur", "angle"}
_L, _D = "[ABEKMHOPCTYX#]", "[0-9#]"
TYPE_RE = {
    "type1": re.compile(f"^{_L}{_D}{{3}}{_L}{{2}}{_D}{{2,3}}$"),
    "type1a": re.compile(f"^{_L}{_D}{{3}}{_L}{{2}}{_D}{{2,3}}$"),
    "type1b": re.compile(f"^{_L}{{2}}{_D}{{3}}{_D}{{2,3}}$"),
}
ANY_RE = re.compile(r"^[0-9A-Z#]{2,12}$")
RECOMMENDED = {"type1a": (150, 50), "type1b": (300, 100), "other": (50, None)}


def clockwise_from_tl(q):
    # площадь со знаком > 0 в системе координат с осью y вниз => по часовой
    x, y = q[:, 0], q[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return area > 0 and int(np.argmin(q.sum(1))) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--report", default=None)
    ap.add_argument("--check-images", action="store_true", help="открывать каждое изображение (медленно)")
    args = ap.parse_args()
    root = args.root
    errors, warnings = [], []
    lines = []

    for d in ("images/real", "images/synthetic", "labels", "generator"):
        if not os.path.isdir(os.path.join(root, d)):
            (errors if d != "generator" else warnings).append(f"нет каталога {d}/")
    for f in ("meta.csv", "README.md", "LICENSE"):
        if not os.path.isfile(os.path.join(root, f)):
            errors.append(f"нет файла {f}")
    if errors and not os.path.isfile(os.path.join(root, "meta.csv")):
        print("\n".join(errors))
        sys.exit(1)

    with open(os.path.join(root, "meta.csv"), encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        if reader.fieldnames != FIELDS:
            errors.append(f"заголовок meta.csv {reader.fieldnames} != {FIELDS}")
        rows = list(reader)

    per_image = defaultdict(list)
    sizes = {}
    for k, r in enumerate(rows, 2):
        e = lambda m: errors.append(f"meta.csv:{k}: {m}")  # noqa: E731
        path = os.path.join(root, r["image"])
        if not os.path.isfile(path):
            e(f"нет файла {r['image']}")
            continue
        if r["image"] not in sizes:
            if args.check_images or r["is_synthetic"] == "0":
                im = cv2.imread(path)
                sizes[r["image"]] = im.shape[:2] if im is not None else None
            else:
                im = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_8)
                sizes[r["image"]] = (im.shape[0] * 8, im.shape[1] * 8) if im is not None else None
        if r["plate_type"] not in TYPES:
            e(f"недопустимый plate_type {r['plate_type']}")
        num = r["plate_num"]
        if r["plate_type"] in TYPE_RE and not TYPE_RE[r["plate_type"]].match(num):
            e(f"номер {num} не соответствует маске")
        if r["plate_type"] == "other" and not ANY_RE.match(num):
            e(f"номер other {num}: допустимы 0-9, A-Z, #")
        try:
            bx = [float(v) for v in r["bbox"].split(",")]
            q = np.array([float(v) for v in r["quad"].split(",")]).reshape(4, 2)
            assert len(bx) == 4 and bx[2] > 0 and bx[3] > 0
        except Exception:
            e("некорректный bbox/quad")
            continue
        if not clockwise_from_tl(q):
            e("quad не по часовой стрелке от левого верхнего угла")
        qx0, qy0 = q.min(0)
        qx1, qy1 = q.max(0)
        if max(abs(qx0 - bx[0]), abs(qy0 - bx[1]), abs(qx1 - bx[0] - bx[2]), abs(qy1 - bx[1] - bx[3])) > 2:
            warnings.append(f"meta.csv:{k}: bbox не совпадает с описывающим прямоугольником quad")
        hw = sizes.get(r["image"])
        if hw and (qx0 < -1 or qy0 < -1 or qx1 > hw[1] + 8 or qy1 > hw[0] + 8):
            e("quad выходит за пределы изображения")
        for fld in ("is_vehicle", "is_synthetic"):
            if r[fld] not in ("0", "1"):
                e(f"{fld} должен быть 0/1")
        if not r["source"] or not r["license"]:
            e("пустые source/license")
        bad = set(c for c in r["conditions"].split(",") if c) - CONDITIONS
        if bad:
            e(f"неизвестные условия съёмки {bad}")
        per_image[r["image"]].append(r)

    # labels/*.txt
    for img, rs in per_image.items():
        stem = os.path.splitext(os.path.basename(img))[0]
        lp = os.path.join(root, "labels", stem + ".txt")
        if not os.path.isfile(lp):
            errors.append(f"нет разметки labels/{stem}.txt")
            continue
        lines_ = [line.split() for line in open(lp) if line.strip()]
        n = len(lines_)
        if any(len(t) not in (5, 13) for t in lines_):
            errors.append(f"labels/{stem}.txt: ожидается 5 или 13 чисел в строке")
        if n != len(rs):
            errors.append(f"labels/{stem}.txt: {n} строк, в meta.csv {len(rs)}")
    listed = set(per_image)
    for sub in ("real", "synthetic"):
        d = os.path.join(root, "images", sub)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if f"images/{sub}/{fn}" not in listed:
                    warnings.append(f"images/{sub}/{fn} отсутствует в meta.csv")

    # статистика
    lines.append("# Отчёт самопроверки датасета\n")
    for syn, title in (("0", "Реальные"), ("1", "Синтетические")):
        sub = [r for r in rows if r["is_synthetic"] == syn]
        imgs = {r["image"] for r in sub}
        lines.append(f"## {title}: {len(imgs)} изображений, {len(sub)} знаков\n")
        lines.append("| тип | изображений | знаков | уникальных ГРЗ | рекомендуемо (изобр./уник.) |")
        lines.append("|---|---|---|---|---|")
        for t in ("type1", "type1a", "type1b", "other"):
            ts = [r for r in sub if r["plate_type"] == t]
            rec = RECOMMENDED.get(t) if syn == "0" else None
            rec_s = f"{rec[0]} / {rec[1] or '–'}" if rec else "–"
            lines.append(f"| {t} | {len({r['image'] for r in ts})} | {len(ts)} | {len({r['plate_num'] for r in ts})} | {rec_s} |")
        cond = Counter(c for r in sub for c in r["conditions"].split(",") if c)
        lines.append("\nУсловия съёмки: " + ", ".join(f"{k}={v}" for k, v in sorted(cond.items())))
        lines.append(f"is_vehicle=0: {sum(r['is_vehicle'] == '0' for r in sub)}\n")
    syn_imgs = len({r['image'] for r in rows if r['is_synthetic'] == '1'})
    if syn_imgs < 5000:
        warnings.append(f"синтетических изображений {syn_imgs} < 5000 (рекомендация)")
    for t, (ni, nu) in RECOMMENDED.items():
        ts = [r for r in rows if r["is_synthetic"] == "0" and r["plate_type"] == t]
        if len({r["image"] for r in ts}) < ni:
            warnings.append(f"реальных изображений {t}: {len({r['image'] for r in ts})} < {ni} (рекомендация)")
    lines.append(f"## Ошибки: {len(errors)}\n")
    lines += [f"- {m}" for m in errors[:200]]
    lines.append(f"\n## Предупреждения: {len(warnings)}\n")
    lines += [f"- {m}" for m in warnings[:200]]
    text = "\n".join(lines)
    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
