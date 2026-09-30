#!/usr/bin/env bash
# Загрузка сторонних материалов для ОБУЧЕНИЯ (в сдаваемый датасет не входят):
#   yolo11n-pose.pt, yolo11n.pt – предобученные веса Ultralytics (AGPL-3.0);
#   carparts-seg.zip – Ultralytics Carparts-Seg (фото автомобилей с разметкой бамперов);
#   coco128.zip      – первые 128 изображений COCO train2017 (CC BY 4.0, изображения Flickr).
set -euo pipefail
DL=${1:-work/dl}
mkdir -p "$DL"
cd "$DL"
BASE=https://github.com/ultralytics/assets/releases/download
for f in v8.3.0/yolo11n-pose.pt v8.3.0/yolo11n.pt v0.0.0/carparts-seg.zip v0.0.0/coco128.zip; do
  [ -f "$(basename $f)" ] || curl -fL -O "$BASE/$f"
done
[ -d images ] || unzip -q carparts-seg.zip
[ -d coco128 ] || unzip -q coco128.zip
echo "OK: $(pwd)"
