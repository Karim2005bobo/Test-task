import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grz.pipeline import Pipeline, imread  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--det-size", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    pipe = Pipeline(device=args.device, det_size=args.det_size)
    files = sorted(os.path.join(args.input, f) for f in os.listdir(args.input)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    pipe(imread(files[0]))  # прогрев
    t_read, t_det, t_all = [], [], []
    for _ in range(args.repeat):
        for f in files:
            t0 = time.perf_counter()
            img = imread(f)
            t1 = time.perf_counter()
            dets = pipe.det(img)
            t2 = time.perf_counter()
            pipe.process(img, dets)
            t3 = time.perf_counter()
            t_read.append(t1 - t0)
            t_det.append(t2 - t1)
            t_all.append(t3 - t2)
    ms = lambda v: 1000 * float(np.mean(v))  # noqa: E731
    print(f"провайдер: {pipe.det.sess.get_providers()[0]}, вход детектора {pipe.det.imgsz}")
    print(f"чтение JPEG: {ms(t_read):.1f} мс; детектор: {ms(t_det):.1f} мс; "
          f"распознавание и постобработка: {ms(t_all):.1f} мс; "
          f"итого на кадр: {ms(t_read) + ms(t_det) + ms(t_all):.1f} мс "
          f"(p95 {1000 * np.percentile(np.add(np.add(t_read, t_det), t_all), 95):.1f})")


if __name__ == "__main__":
    main()
