# Распознавание нестандартных ГРЗ (типы 1, 1А, 1Б) — полуфинал «Волга-IT 2026»

Решение находит на изображениях государственные регистрационные знаки РФ,
определяет их тип (`type1` / `type1a` / `type1b` / `other`) и читает номер.

- Работает офлайн на ONNX Runtime, на CPU или GPU. PyTorch для запуска не нужен.
- Все веса лежат в `weights/`.
- Пояснительная записка: [`docs/explanatory_note.md`](docs/explanatory_note.md).

```
изображение ─► детектор YOLO11n-pose ─► 4 угла знака + компоновка (1 или 2 строки)
            ─► гомография: 1 строка → лента 224×48; 2 строки → строки рядом в ленте 224×48
            ─► распознаватель CNN+BiLSTM: CTC-символы + «не знак»
            ─► тип: компоновка × жёлтый фон × соответствие маске (1/1А: A123BC77, 1Б: AB12377)
            ─► Витерби по CTC с маской типа ─► CSV
```

## Запуск

Требования: Python 3.9–3.12.

```bash
cd grz_recognition
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Команда запуска на каталоге изображений:**

```bash
python run.py --input <КАТАЛОГ_С_ИЗОБРАЖЕНИЯМИ> --output result.csv
```

Путь можно передать и иначе:
- переменные окружения: `GRZ_INPUT=/data/images GRZ_OUTPUT=result.csv python run.py`;
- конфигурационный файл: `python run.py --config config.yaml` (образец — `config.example.yaml`).

Результат: `image;plate_num;plate_type;confidence`, разделитель `;`, UTF-8,
первая строка — заголовок. Если на изображении несколько знаков, для него
будет несколько строк. После обработки печатается среднее время на кадр.

### GPU (референсная GTX 1050 Ti)

```bash
pip uninstall -y onnxruntime && pip install onnxruntime-gpu   # CUDA 12 + cuDNN 9
python run.py --input <КАТАЛОГ> --output result.csv            # GPU подхватится сам
```

При `--device auto` (по умолчанию) решение использует GPU, если доступен
`CUDAExecutionProvider`, иначе CPU. Вход детектора тоже выбирается
автоматически: 960 на GPU (лучше находит мелкие знаки) и 640 на CPU
(укладывается в 100 мс/кадр).

### Параметры

| параметр | по умолчанию | назначение |
|---|---|---|
| `--device` | `auto` | `auto` / `cpu` / `cuda` |
| `--det-size` | авто | 640 или 960; авто: 960 на GPU, 640 на CPU |
| `--det-conf` | 0.3 | порог детектора |
| `--target-conf` | 0.55 | знак целевого типа с уверенностью ниже порога выводится как `other` |
| `--min-conf` | 0.25 | находки с уверенностью ниже порога не выводятся |
| `--char-thr` | 0 | символы с уверенностью ниже порога заменяются на `#` (0 — всегда лучшая догадка: `#` засчитывается как ошибка) |
| `--no-other` | выкл. | не выводить знаки `other` (по ТЗ допустимы оба варианта) |
| `--vehicle-filter` | выкл. | дополнительный детектор ТС (YOLO11n COCO) понижает уверенность знаков вне ТС (щиты, витрины) |

## Качество и скорость

**Отладочный набор организаторов**, официальный `evaluate.py`:

| вход | засчитано | A_type1 | A_type1a | precision |
|---|---|---|---|---|
| 960 (GPU) | 93 / 115 | 0,60 | 0,84 | 0,93 |
| 640 (CPU) | 92 / 115 | 0,33 | 0,87 | 0,94 |

**Скорость на CPU** (4 ядра Xeon 2,1 ГГц, с чтением JPEG): 87 мс/кадр при
входе 640 и 152 мс/кадр при входе 960. Подробности — в записке.

```bash
python tools/benchmark.py --input <КАТАЛОГ> --device cpu --det-size 640
python run.py --input debug_set --output result.csv
python evaluate.py --gt debug_labels.csv --pred result.csv --images debug_set   # скрипт организаторов
```

## Структура

```
grz/            инференс: маски номеров, детектор, выпрямление, распознаватель, пайплайн
run.py          точка входа (каталог → CSV)
weights/        detector_640.onnx, detector_960.onnx, recognizer.onnx, vehicle.onnx
generator/      генератор синтетики: штриховой шрифт ГОСТ, рендер знаков, сцены, эффекты
training/       подготовка данных, обучение, экспорт в ONNX
tools/          сбор и разметка реальных данных, размытие лиц, самопроверка датасета, бенчмарк
dataset/        датасет по разделу 6: 578+ реальных и 5000 синтетических изображений (см. dataset/README.md)
docs/           пояснительная записка
```

## Датасет

```bash
python dataset/generator/generate.py      # пересоздать синтетику (5000 изображений, seed = 2025)
python validate_dataset.py dataset        # валидатор организаторов; отчёт — dataset/validation_report.txt
python tools/annotate.py --src my_photos --dataset dataset   # разметка собственных фото (лица размываются)
```

## Воспроизведение обучения

```bash
pip install -r requirements-train.txt
bash training/download_assets.sh work/dl            # YOLO11n(-pose), фоновые фото (только обучение)
bash training/download_real_data.sh work/nomeroff   # AUTO.RIA Numberplate (CC BY 4.0)

# 1. Детектор: синтетические сцены, затем дообучение на смеси с реальными сценами
python -m generator.generate_dataset --format yolo --out work/det --n 12000 --seed 7 \
       --carparts work/dl --coco work/dl/coco128
python training/train_detector.py --data work/det/data.yaml --weights work/dl/yolo11n-pose.pt --epochs 1
python training/import_via.py --src work/nomeroff/autoriaNumberplateDataset-2018-11-20 --out work/det_real
#    work/det_mix/data.yaml: train = 2000 синтетических + реальные сцены work/det_real, val – то же
python training/train_detector.py --data work/det_mix/data.yaml \
       --weights work/det_runs/yolo11n_pose_grz/weights/last.pt --epochs 5

# 2. Распознаватель
python training/make_ocr_data.py --out work/ocr --n 100000 --carparts work/dl --coco work/dl/coco128
python training/add_real_crops.py --src work/nomeroff/autoriaNumberplateOcrRu-2021-09-01 --out work/ocr_real
python training/add_real_crops.py --options work/nomeroff/autoriaNumberplateOptionsDataset-2021-09-03 --out work/ocr_opts
python training/mine_negatives.py --images work/det_real/images --weights weights --out work/ocr_neg
python training/train_recognizer.py --data work/ocr work/ocr_real work/ocr_opts work/ocr_neg --out work/rec --epochs 3

# 3. Экспорт в weights/
python training/export_onnx.py --rec work/rec/best.pt --det work/det_runs/det_mix/weights/best.pt \
       --vehicle work/dl/yolo11n.pt
```
