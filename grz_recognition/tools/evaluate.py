import argparse
import csv
from collections import defaultdict

import numpy as np


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return d[-1]


def match(cost):
    pairs, used_r, used_c = [], set(), set()
    for k in np.argsort(cost, axis=None):
        i, j = divmod(int(k), cost.shape[1])
        if i not in used_r and j not in used_c:
            pairs.append((i, j))
            used_r.add(i)
            used_c.add(j)
    return [i for i, _ in pairs], [j for _, j in pairs]


def read(path):
    rows = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            rows[r["image"].strip()].append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()
    gt, pr = read(args.gt), read(args.pred)
    stats = defaultdict(lambda: {"n": 0, "exact": 0, "char": 0.0, "type_ok": 0, "found": 0})
    tp = fp = 0
    other_as_target = 0
    for img in sorted(set(gt) | set(pr)):
        g, p = gt.get(img, []), pr.get(img, [])
        if g and p:
            cost = np.array([[lev(a["plate_num"], b["plate_num"]) / max(len(a["plate_num"]), 1) for b in p] for a in g])
            ri, ci = match(cost)
        else:
            ri, ci, cost = [], [], None
        matched_p = set()
        for i, j in zip(ri, ci):
            if cost[i, j] > 0.5:
                continue
            a, b = g[i], p[j]
            s = stats[a["plate_type"]]
            s["found"] += 1
            s["exact"] += int(a["plate_num"] == b["plate_num"])
            s["char"] += 1 - cost[i, j]
            s["type_ok"] += int(a["plate_type"] == b["plate_type"])
            if a["plate_type"] == "other" and b["plate_type"] != "other":
                other_as_target += 1
            matched_p.add(j)
            tp += 1
        for a in g:
            stats[a["plate_type"]]["n"] += 1
        fp += len(p) - len(matched_p)
    n_gt = sum(s["n"] for s in stats.values())
    print(f"{'type':8} {'n':>5} {'recall':>7} {'exact':>7} {'char':>7} {'type_acc':>8}")
    for t in ("type1", "type1a", "type1b", "other"):
        s = stats.get(t)
        if not s or not s["n"]:
            continue
        f = max(s["found"], 1)
        print(f"{t:8} {s['n']:5d} {s['found'] / s['n']:7.3f} {s['exact'] / s['n']:7.3f} "
              f"{s['char'] / s['n']:7.3f} {s['type_ok'] / f:8.3f}")
    print(f"detection precision {tp / max(tp + fp, 1):.3f}  recall {tp / max(n_gt, 1):.3f}  FP {fp}")
    print(f"'other' выданы как целевой тип: {other_as_target}")


if __name__ == "__main__":
    main()
