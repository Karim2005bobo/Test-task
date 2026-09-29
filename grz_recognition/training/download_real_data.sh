#!/usr/bin/env bash
# Реальные данные для ОБУЧЕНИЯ (все – CC BY 4.0, ARS Online OU / Nomeroff Net):
#   autoriaNumberplateDataset-2018-11-20  – 1,5 тыс. сцен с полигонами номеров (детектор, негативы);
#   autoriaNumberplateOcrRu-2021-09-01    – 57 тыс. кропов знаков РФ с текстом (распознаватель);
#   autoriaNumberplateOptionsDataset-2021-09-03 – кропы с атрибутами страна/строки (1А, «двойники», «мусор»).
set -euo pipefail
DIR=${1:-work/nomeroff}
mkdir -p "$DIR"
cd "$DIR"
BASE=https://nomeroff.net.ua/datasets
for f in autoriaNumberplateDataset-2018-11-20 autoriaNumberplateOcrRu-2021-09-01 autoriaNumberplateOptionsDataset-2021-09-03; do
  [ -d "$f" ] || { curl -fL -o "$f.zip" "$BASE/$f.zip" && unzip -q "$f.zip" && rm "$f.zip"; }
done
echo "OK: $(pwd)"
