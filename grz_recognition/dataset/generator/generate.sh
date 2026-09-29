#!/usr/bin/env bash
# Воспроизведение синтетической части датасета (5000 изображений, seed=2025).
# Код генератора: grz_recognition/generator (рендер знаков, сцены, аугментации)
# и grz_recognition/grz (формат номера, выпрямление) – запускается из корня проекта.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python -m generator.generate_dataset --out dataset --n "${N:-5000}" --seed "${SEED:-2025}" --format dataset
