# Распознавание нестандартных ГРЗ (типы 1, 1А, 1Б) — полуфинал

Пайплайн находит на изображении государственные регистрационные знаки РФ,
определяет тип (`type1` / `type1a` / `type1b` / `other`) и читает номер.
Инференс работает офлайн на ONNX Runtime (CPU или GPU), PyTorch для запуска
не нужен. Все веса лежат в `weights/`.

```
изображение ─► детектор YOLO11n-pose ─► 4 угла знака + компоновка (1 или 2 строки)
            ─► гомография, выпрямление: 1 строка → 224×48; 2 строки → разрезка и склейка в ленту 224×48
            ─► распознаватель CNN+BiLSTM: CTC-символы + тип (type1/1a/1b/other/не знак)
            ─► декодирование по маске РФ (Витерби по CTC), '#' для неуверенных позиций
            ─► CSV
```

## Быстрый запуск

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py --input /path/to/images --output result.csv
```

Точная команда для жюри (из каталога `grz_recognition/`):

```bash
python run.py --input <КАТАЛОГ_С_ИЗОБРАЖЕНИЯМИ> --output result.csv
```

Путь можно передать иначе:
* переменные окружения: `GRZ_INPUT=/data/images GRZ_OUTPUT=out.csv python run.py`;
* конфигурационный файл: `python run.py --config config.yaml` (см. `config.example.yaml`).

Формат результата: `image;plate_num;plate_type;confidence`, разделитель `;`,
UTF-8, первая строка — заголовок. После обработки печатается среднее время на
изображение.

### GPU (GTX 1050 Ti)

```bash
pip uninstall -y onnxruntime && pip install onnxruntime-gpu   # нужны CUDA 12 и cuDNN 9
python run.py --input ... --output ... --device cuda
```
При `--device auto` (по умолчанию) GPU используется, если доступен
`CUDAExecutionProvider`, иначе решение работает на CPU.

### Параметры

| параметр | по умолчанию | назначение |
|---|---|---|
| `--det-size` | 640 | размер входа детектора (больше — лучше для мелких знаков, медленнее) |
| `--det-conf` | 0.3 | порог детектора |
| `--char-thr` | 0.35 | символы с уверенностью ниже порога заменяются на `#` |
| `--min-conf` | 0.25 | знаки с итоговой уверенностью ниже порога не выводятся |
| `--no-other` | выкл. | не выводить знаки типа `other` (по ТЗ допустимы оба варианта) |
| `--vehicle-filter` | выкл. | дополнительный детектор ТС (YOLO11n COCO): понижает уверенность знаков вне ТС (щиты, витрины) |

## Структура

```
grz/                 инференс: формат номера, детектор, выпрямление, распознаватель, пайплайн
run.py               точка входа (каталог → CSV)
weights/             detector.onnx, recognizer.onnx, vehicle.onnx
generator/           генератор синтетики: штриховой шрифт ГОСТ, рендер знаков, сцены, эффекты
training/            подготовка данных, обучение, экспорт в ONNX
tools/               evaluate.py (метрики по эталону), validate_dataset.py, annotate.py (разметка фото)
dataset/             датасет по разделу 6 (см. dataset/README.md)
docs/                пояснительная записка
```

## Воспроизведение обучения

```bash
pip install -r requirements-train.txt
bash training/download_assets.sh work/dl          # предобученные веса и фоновые фото (только для обучения)

# 1. детектор: синтетические сцены + реальная часть датасета
python -m generator.generate_dataset --format yolo --out work/det --n 12000 --seed 7 \
       --carparts work/dl --coco work/dl/coco128
python training/train_detector.py --data work/det/data.yaml --weights work/dl/yolo11n-pose.pt \
       --real-dataset dataset --epochs 30 --device 0

# 2. распознаватель
python training/make_ocr_data.py --out work/ocr --n 100000 --carparts work/dl --coco work/dl/coco128 \
       --real-dataset dataset --real-repeat 20
python training/train_recognizer.py --data work/ocr --out work/rec --epochs 12

# 3. экспорт
python training/export_onnx.py --rec work/rec/best.pt \
       --det work/det_runs/yolo11n_pose_grz/weights/best.pt --vehicle work/dl/yolo11n.pt
```

## Датасет

```bash
bash dataset/generator/generate.sh                 # 5000 синтетических изображений, seed=2025
python tools/validate_dataset.py dataset --report dataset/validation_report.md
python tools/annotate.py --src my_photos --dataset dataset   # разметка собственных фото
```

## Оценка на отладочном наборе

```bash
python run.py --input debug/images --output result.csv
python tools/evaluate.py --pred result.csv --gt debug/labels.csv
```
