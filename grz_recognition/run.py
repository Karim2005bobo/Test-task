"""Распознавание ГРЗ в каталоге изображений -> CSV.

Использование:
  python run.py --input /path/to/images --output result.csv
Путь можно передать и через переменные окружения GRZ_INPUT / GRZ_OUTPUT
или конфигурационный файл (--config config.yaml, ключи совпадают с аргументами).
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grz.pipeline import Pipeline, WEIGHTS, imread  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_config(path):
    cfg = {}
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if ":" in line:
            k, v = line.split(":", 1)
            cfg[k.strip().replace("-", "_")] = v.strip().strip("'\"")
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Распознавание нестандартных ГРЗ (типы 1, 1А, 1Б)")
    ap.add_argument("--input", default=os.environ.get("GRZ_INPUT"), help="каталог с изображениями")
    ap.add_argument("--output", default=os.environ.get("GRZ_OUTPUT", "result.csv"), help="CSV с результатами")
    ap.add_argument("--config", default=None, help="YAML-подобный файл key: value")
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--device", default=os.environ.get("GRZ_DEVICE", "auto"), choices=["auto", "cpu", "cuda"])
    ap.add_argument("--det-size", type=int, default=640, help="размер входа детектора")
    ap.add_argument("--det-conf", type=float, default=0.3)
    ap.add_argument("--char-thr", type=float, default=0.0,
                    help="порог уверенности символа, ниже – '#' (0 – всегда лучшая догадка: '#' в ответе "
                         "засчитывается как ошибка)")
    ap.add_argument("--min-conf", type=float, default=0.25, help="минимальная уверенность для вывода знака")
    ap.add_argument("--target-conf", type=float, default=0.55,
                    help="знаки целевых типов с уверенностью ниже порога выводятся как other")
    ap.add_argument("--no-other", action="store_true", help="не выводить знаки типа other")
    ap.add_argument("--vehicle-filter", action="store_true",
                    help="понижать уверенность знаков вне ТС (доп. детектор ТС, +~15 мс)")
    args = ap.parse_args()
    if args.config:
        for k, v in load_config(args.config).items():
            if hasattr(args, k):
                cur = getattr(args, k)
                setattr(args, k, type(cur)(v) if cur is not None and not isinstance(cur, bool)
                        else (v.lower() in ("1", "true", "yes") if isinstance(cur, bool) else v))
    if not args.input or not os.path.isdir(args.input):
        ap.error("укажите существующий каталог --input (или GRZ_INPUT)")

    pipe = Pipeline(args.weights, args.device, args.det_size, args.det_conf, args.char_thr,
                    emit_other=not args.no_other, vehicle_filter=args.vehicle_filter, min_conf=args.min_conf,
                    target_conf=args.target_conf)
    files = sorted(f for f in os.listdir(args.input) if os.path.splitext(f)[1].lower() in EXTS)
    # прогрев (инициализация CUDA/графа не входит в замер)
    if files:
        img = imread(os.path.join(args.input, files[0]))
        if img is not None:
            pipe(img)

    t_total, n = 0.0, 0
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["image", "plate_num", "plate_type", "confidence"])
        for name in files:
            t0 = time.perf_counter()
            img = imread(os.path.join(args.input, name))
            if img is None:
                print(f"не удалось прочитать {name}", file=sys.stderr)
                continue
            res = pipe(img)
            t_total += time.perf_counter() - t0
            n += 1
            for r in sorted(res, key=lambda r: r.quad[:, 0].min()):
                w.writerow([name, r.plate_num, r.plate_type, f"{r.confidence:.2f}"])
    if n:
        print(f"Обработано {n} изображений, среднее время {1000 * t_total / n:.1f} мс/изобр. -> {args.output}")


if __name__ == "__main__":
    main()
