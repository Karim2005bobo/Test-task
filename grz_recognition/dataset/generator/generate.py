import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from generator import generate_dataset 

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
