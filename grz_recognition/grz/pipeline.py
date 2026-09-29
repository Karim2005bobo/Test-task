"""Полный пайплайн: детекция -> выпрямление -> распознавание -> постобработка."""
import os
from dataclasses import dataclass

import cv2
import numpy as np

from grz.detector import PlateDetector, VehicleDetector
from grz.plate_format import MASKS, TO_DIGIT, TO_LETTER, allowed_chars
from grz.recognizer import Recognizer, decode_ru, greedy_with_conf
from grz.rectify import order_quad, rectify, to_tensor

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")

@dataclass
class PlateResult:
    plate_num: str
    plate_type: str
    confidence: float
    quad: np.ndarray


def providers_for(device):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if device in ("auto", "cuda") and "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda":
        print("CUDAExecutionProvider недоступен, используется CPU")
    return ["CPUExecutionProvider"]


def _fix_by_mask(s, plate_type):
    """Позиционная замена похожих символов; None, если строку нельзя привести к маске."""
    for mask in MASKS[plate_type]:
        if len(s) != len(mask):
            continue
        out = []
        for ch, m in zip(s, mask):
            allowed = allowed_chars(m)
            if ch not in allowed:
                ch = (TO_LETTER if m == "L" else TO_DIGIT).get(ch, ch)
            if ch not in allowed:
                break
            out.append(ch)
        else:
            return "".join(out)
    return None


class Pipeline:
    def __init__(self, weights_dir=WEIGHTS, device="auto", det_size=0, det_conf=0.3,
                 char_thr=0.0, emit_other=True, vehicle_filter=False, min_conf=0.25,
                 max_gap=6.0, min_char_conf=0.4, yellow_thr=0.25, target_conf=0.55):
        prov = providers_for(device)
        # Детектор экспортирован с фиксированным входом: detector_<size>.onnx.
        # По умолчанию: 960 на GPU (мелкие знаки), 640 на CPU (укладывается в 100 мс).
        if not det_size:
            det_size = 960 if prov[0] == "CUDAExecutionProvider" else 640
        det_path = os.path.join(weights_dir, f"detector_{det_size}.onnx")
        if not os.path.exists(det_path):
            det_path = os.path.join(weights_dir, "detector.onnx")
        self.det = PlateDetector(det_path, prov, det_size, det_conf)
        self.rec = Recognizer(os.path.join(weights_dir, "recognizer.onnx"), prov)
        self.veh = VehicleDetector(os.path.join(weights_dir, "vehicle.onnx"), prov) if vehicle_filter else None
        self.char_thr, self.emit_other, self.min_conf = char_thr, emit_other, min_conf
        self.max_gap, self.min_char_conf, self.yellow_thr = max_gap, min_char_conf, yellow_thr
        self.target_conf = target_conf

    def __call__(self, img):
        return self.process(img, self.det(img))

    def process(self, img, dets):
        """Выпрямление, распознавание и постобработка найденных знаков."""
        if not dets:
            return []
        crops = []
        for d in dets:
            q = order_quad(d.quad)
            # вырожденные точки -> берём прямоугольник детекции
            area = cv2.contourArea(q)
            bx, by, bw, bh = d.box
            if d.kpt_conf < 0.3 or area < 0.3 * bw * bh:
                q = np.array([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]], np.float32)
            d.quad = q
            crops.append(rectify(img, q, d.layout == 1))
        ctc, cls = self.rec(to_tensor(crops))
        vboxes = self.veh(img) if self.veh is not None else None
        results = []
        for d, p, tp, crop in zip(dets, ctc, cls, crops):
            r = self.decide(d, p, tp, crop)
            if r is None:
                continue
            text, ptype, conf = r
            if vboxes is not None and not _on_vehicle(d.quad, vboxes):
                conf *= 0.3
            if ptype != "other" and conf < self.target_conf:
                # неуверенный знак выдаём как other: такие строки не штрафуются,
                # а ложное «прочтение» эмблемы/надписи как type1 – штрафуется
                ptype = "other"
            if conf < self.min_conf:
                continue
            if ptype == "other" and not self.emit_other:
                continue
            results.append(PlateResult(text, ptype, conf, d.quad))
        return results

    def decide(self, d, p, tp, crop):
        """Выбор типа и текста знака.

        Тип определяется совместно: компоновка от детектора (1 или 2 строки),
        цвет фона (жёлтый – только 1Б), соответствие прочитанного текста маске
        типа и уверенность распознавателя. Мерой соответствия маске служит
        «цена» маски: насколько лучший путь CTC, удовлетворяющий маске, хуже
        лучшего пути без ограничений (в натах). Знак, текст которого не
        укладывается ни в одну маску целевых типов, получает тип other – такие
        строки не штрафуются при проверке и не выдаются за целевые.
        """
        if tp[4] > 0.9 and tp[4] > 3 * max(tp[:4]):
            return None                                   # уверенно «не знак»
        g_text, g_confs, g_score = greedy_with_conf(p)
        yellow = yellow_fraction(crop)
        # штраф (в натах) за несоответствие цвета фона типу: жёлтый – только 1Б.
        # Цвет – мягкий признак: белый знак на жёлтом автобусе даёт много «жёлтых»
        # пикселей в кропе, а выбор 1 или 1Б надёжнее делает маска (A123BC vs AB123).
        if d.layout == 1:
            cands = {"type1a": 0.0}
        else:
            cands = {"type1": 0.0 if yellow < 0.6 else 2.0,
                     "type1b": 0.0 if yellow >= self.yellow_thr else 3.0}
        best = None
        for ptype, pen in cands.items():
            fixed = _fix_by_mask(g_text, ptype)
            if fixed is not None:
                text, confs, gap = fixed, g_confs, float(sum(fx != gx for fx, gx in zip(fixed, g_text)))
            else:
                r = decode_ru(p, ptype)
                if r is None:
                    continue
                text, confs, s = r
                gap = max(0.0, g_score - s)
            gap += pen
            if best is None or gap < best[3]:
                best = (text, confs, ptype, gap)
        char_conf = float(np.mean(g_confs)) if g_confs else 0.0
        if best is not None and best[3] <= self.max_gap:
            text, confs, ptype, gap = best
            char_conf = float(np.mean(confs))
            if char_conf >= self.min_char_conf:
                text = "".join(c if cf >= self.char_thr else "#" for c, cf in zip(text, confs))
                conf = float(np.clip(d.score ** 0.3 * char_conf ** 0.7 * np.exp(-0.1 * gap), 0, 1))
                return text, ptype, conf
        if not g_text:
            return None
        return g_text, "other", float(np.clip(d.score ** 0.3 * char_conf ** 0.7, 0, 1))


def yellow_fraction(crop):
    """Доля «жёлтых» пикселей фона знака (тип 1Б)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (h >= 12) & (h <= 38) & (s >= 70) & (v >= 60)
    return float(m.mean())


def _on_vehicle(quad, vboxes):
    cx, cy = quad.mean(0)
    for x0, y0, x1, y1 in vboxes:
        mx, my = (x1 - x0) * 0.1, (y1 - y0) * 0.1
        if x0 - mx <= cx <= x1 + mx and y0 - my <= cy <= y1 + my:
            return True
    return False


def imread(path):
    """Чтение с поддержкой не-ASCII путей (Windows)."""
    try:
        data = np.fromfile(path, np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None
