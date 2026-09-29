"""Полный пайплайн: детекция -> выпрямление -> распознавание -> постобработка."""
import os
from dataclasses import dataclass

import cv2
import numpy as np

from grz.detector import PlateDetector, VehicleDetector
from grz.plate_format import MASK_8, MASK_9, PLATE_TYPES, TO_DIGIT, TO_LETTER, allowed_chars
from grz.recognizer import Recognizer, decode_ru, greedy_with_conf
from grz.rectify import order_quad, rectify, to_tensor

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")

# Совместимость «компоновка детектора» -> тип знака (type1, type1a, type1b, other, none)
LAYOUT_PRIOR = {
    0: np.array([1.0, 0.05, 1.0, 1.0, 1.0]),   # однострочный
    1: np.array([0.05, 1.0, 0.05, 1.0, 1.0]),  # двухстрочный
}


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


def _fix_by_mask(s):
    """Позиционная замена похожих символов; None, если строку нельзя привести к маске."""
    for mask in (MASK_8, MASK_9):
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
    def __init__(self, weights_dir=WEIGHTS, device="auto", det_size=640, det_conf=0.3,
                 char_thr=0.35, emit_other=True, vehicle_filter=False, min_conf=0.25):
        prov = providers_for(device)
        self.det = PlateDetector(os.path.join(weights_dir, "detector.onnx"), prov, det_size, det_conf)
        self.rec = Recognizer(os.path.join(weights_dir, "recognizer.onnx"), prov)
        self.veh = VehicleDetector(os.path.join(weights_dir, "vehicle.onnx"), prov) if vehicle_filter else None
        self.char_thr, self.emit_other, self.min_conf = char_thr, emit_other, min_conf

    def __call__(self, img):
        dets = self.det(img)
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
        for d, p, tp in zip(dets, ctc, cls):
            tp = tp * LAYOUT_PRIOR[d.layout]
            tp = tp / tp.sum()
            t = int(tp.argmax())
            if t == 4:           # «не знак» – ложная детекция
                continue
            ptype = PLATE_TYPES[t]
            if ptype == "other":
                if not self.emit_other:
                    continue
                text, confs, _ = greedy_with_conf(p)
            else:
                text, confs, _ = greedy_with_conf(p)
                fixed = _fix_by_mask(text)
                if fixed is None:
                    text, confs, _ = decode_ru(p)
                else:
                    text = fixed
            if not text:
                continue
            text = "".join(c if cf >= self.char_thr else "#" for c, cf in zip(text, confs))
            char_conf = float(np.mean(confs)) if confs else 0.0
            conf = float(np.clip((d.score * tp[t] * char_conf) ** (1 / 3), 0, 1))
            if vboxes is not None and not _on_vehicle(d.quad, vboxes):
                conf *= 0.3
            if conf < self.min_conf:
                continue
            results.append(PlateResult(text, ptype, conf, d.quad))
        return results


def _on_vehicle(quad, vboxes):
    cx, cy = quad.mean(0)
    for x0, y0, x1, y1 in vboxes:
        mx, my = (x1 - x0) * 0.1, (y1 - y0) * 0.1
        if x0 - mx <= cx <= x1 + mx and y0 - my <= cy <= y1 + my:
            return True
    return False


def imread(path):
    """Чтение с поддержкой не-ASCII путей (Windows)."""
    data = np.fromfile(path, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img
