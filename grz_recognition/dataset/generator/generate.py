"""Воспроизведение синтетической части датасета (5000 изображений, seed=2025).

Запуск из любого каталога:
    python dataset/generator/generate.py [--n 5000] [--seed 2025]

Код генератора находится в grz_recognition/generator (рендер знаков по ГОСТ,
штриховой шрифт, сцены, эффекты) и grz_recognition/grz (маски номеров,
выпрямление). Скрипт подключает их и пишет результат в этот датасет:
images/synthetic/, labels/, meta.csv (строки реальной части сохраняются).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from generator import generate_dataset  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--out" not in args:
        args += ["--out", DATASET]
    if "--n" not in args:
        args += ["--n", "5000"]
    if "--seed" not in args:
        args += ["--seed", "2025"]
    sys.argv = [sys.argv[0]] + args + ["--format", "dataset"]
    generate_dataset.main()
