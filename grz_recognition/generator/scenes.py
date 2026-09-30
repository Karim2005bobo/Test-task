import glob
import math
import os

import cv2
import numpy as np

from generator import effects, plates
from grz.rectify import order_quad, rectify

VEHICLE_COCO = {2, 3, 5, 7}
BUMPER_CLASSES = {0, 8}  # back_bumper, front_bumper в Carparts-Seg

class BackgroundPool:
    def __init__(self, carparts_dir=None, coco_dir=None, weights=(0.6, 0.1, 0.3)):
        self.carparts, self.coco = [], []
        if carparts_dir and os.path.isdir(carparts_dir):
            for lbl in sorted(glob.glob(os.path.join(carparts_dir, "labels", "*", "*.txt"))):
                boxes = []
                for line in open(lbl):
                    v = line.split()
                    if int(v[0]) in BUMPER_CLASSES and len(v) >= 7:
                        xy = np.array(v[1:], np.float32).reshape(-1, 2)
                        x0, y0 = xy.min(0)
                        x1, y1 = xy.max(0)
                        if x1 - x0 > 0.15:
                            boxes.append((x0, y0, x1 - x0, y1 - y0))
                img = lbl.replace(os.sep + "labels" + os.sep, os.sep + "images" + os.sep)[:-4] + ".jpg"
                if boxes and os.path.exists(img):
                    self.carparts.append((img, boxes))
        if coco_dir and os.path.isdir(coco_dir):
            for lbl in sorted(glob.glob(os.path.join(coco_dir, "labels", "*", "*.txt"))):
                boxes = []
                for line in open(lbl):
                    v = line.split()
                    if int(v[0]) in VEHICLE_COCO:
                        cx, cy, w, h = map(float, v[1:5])
                        if w > 0.08:
                            # «бампер» – нижняя треть бокса ТС
                            boxes.append((cx - w / 2, cy + h / 6, w, h / 3))
                img = lbl.replace(os.sep + "labels" + os.sep, os.sep + "images" + os.sep)[:-4] + ".jpg"
                if boxes and os.path.exists(img):
                    self.coco.append((img, boxes))
        w = np.array(weights, np.float64) * np.array([bool(self.carparts), bool(self.coco), True])
        self.weights = w / w.sum()

    def random_patch(self, rng, w, h):
        src = rng.choice(3, p=self.weights)
        if src == 2:
            return procedural_patch(rng, w, h)
        lst = self.carparts if src == 0 else self.coco
        img = cv2.imread(lst[int(rng.integers(0, len(lst)))][0])
        H, W = img.shape[:2]
        s = rng.uniform(0.3, 1.0)
        pw, ph = max(int(W * s), 8), max(int(W * s * h / w), 8)
        pw, ph = min(pw, W), min(ph, H)
        x, y = int(rng.integers(0, W - pw + 1)), int(rng.integers(0, H - ph + 1))
        return cv2.resize(img[y:y + ph, x:x + pw], (w, h), interpolation=cv2.INTER_LINEAR)

    def sample(self, rng):
        """-> (BGR-изображение, список якорей (x, y, w, h, is_vehicle) в пикселях)."""
        src = rng.choice(3, p=self.weights)
        if src == 2:
            return procedural_scene(rng)
        path, boxes = (self.carparts if src == 0 else self.coco)[int(rng.integers(0, len(self.carparts if src == 0 else self.coco)))]
        img = cv2.imread(path)
        H, W = img.shape[:2]
        # случайное масштабирование кадра (разрешение камер разное)
        s = rng.uniform(0.8, 2.2)
        img = cv2.resize(img, (int(W * s), int(H * s * rng.uniform(0.85, 1.15))), interpolation=cv2.INTER_LINEAR)
        H2, W2 = img.shape[:2]
        anchors = [(x * W2, y * H2, w * W2, h * H2, 1) for x, y, w, h in boxes]
        return img, anchors


def _rand_color(rng, lo=0, hi=255):
    return tuple(int(v) for v in rng.integers(lo, hi, 3))


def _draw_vehicle(img, rng, x, y, w):
    h = int(w * rng.uniform(0.55, 0.9))
    body = _rand_color(rng, 20, 235)
    x, y = int(x), int(y)
    cv2.rectangle(img, (x, y + h // 3), (x + w, y + h), body, -1)
    # крыша/стекло
    gl = tuple(int(v * 0.35) for v in body)
    cv2.rectangle(img, (x + w // 8, y), (x + w - w // 8, y + h // 3), body, -1)
    cv2.rectangle(img, (x + w // 6, y + h // 20), (x + w - w // 6, y + h // 3), gl, -1)
    # фары
    lc = (40, 40, 200) if rng.random() < 0.5 else (230, 230, 230)
    for lx in (x + w // 20, x + w - w // 20 - w // 6):
        cv2.rectangle(img, (lx, y + h // 2 - h // 12), (lx + w // 6, y + h // 2 + h // 12), lc, -1)
    # бампер
    by = y + int(h * 0.62)
    bc = tuple(int(np.clip(v * rng.uniform(0.6, 1.1), 0, 255)) for v in body)
    cv2.rectangle(img, (x, by), (x + w, y + h), bc, -1)
    # колёса
    for wx in (x + w // 10, x + w - w // 10 - w // 7):
        cv2.rectangle(img, (wx, y + h), (wx + w // 7, y + h + h // 10), (20, 20, 20), -1)
    return (x + w * 0.05, by, w * 0.9, (y + h) - by, 1)


def procedural_patch(rng, w, h):
    img = np.empty((h, w, 3), np.uint8)
    t = np.linspace(0, 1, h)[:, None, None]
    img[:] = (np.array(_rand_color(rng)) * (1 - t) + np.array(_rand_color(rng)) * t).astype(np.uint8)
    for _ in range(int(rng.integers(0, 8))):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.rectangle(img, (x0, y0), (x0 + int(rng.integers(2, w)), y0 + int(rng.integers(2, h))),
                      _rand_color(rng), -1 if rng.random() < 0.6 else int(rng.integers(1, 4)))
    return effects.noise(img, rng)


def procedural_scene(rng):
    W = int(rng.choice([640, 800, 960, 1280, 1600]))
    H = int(W * rng.uniform(0.56, 0.8))
    img = np.zeros((H, W, 3), np.uint8)
    # небо/стена + дорога, градиенты
    horizon = int(H * rng.uniform(0.2, 0.6))
    top, bottom = _rand_color(rng, 60, 230), _rand_color(rng, 30, 140)
    for i, (a, b, y0, y1) in enumerate([(top, _rand_color(rng, 60, 230), 0, horizon),
                                         (bottom, _rand_color(rng, 30, 140), horizon, H)]):
        t = np.linspace(0, 1, max(y1 - y0, 1))[:, None, None]
        img[y0:y1] = (np.array(a) * (1 - t) + np.array(b) * t).astype(np.uint8)
    # «здания», столбы, разметка – структурированный шум
    for _ in range(int(rng.integers(5, 30))):
        x0, y0 = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.rectangle(img, (x0, y0), (x0 + int(rng.integers(5, W // 3)), y0 + int(rng.integers(5, H // 3))),
                      _rand_color(rng), -1 if rng.random() < 0.7 else int(rng.integers(1, 5)))
    for _ in range(int(rng.integers(0, 10))):
        cv2.line(img, (int(rng.integers(0, W)), int(rng.integers(0, H))),
                 (int(rng.integers(0, W)), int(rng.integers(0, H))), _rand_color(rng), int(rng.integers(1, 8)))
    # «текстовые» отвлекающие объекты: вывески с буквами и цифрами (не ГРЗ)
    for _ in range(int(rng.integers(0, 3))):
        x0, y0 = int(rng.integers(0, W - 60)), int(rng.integers(0, H - 30))
        s = "".join(rng.choice(list("ABCDEFGHKMOPTX0123456789 "), int(rng.integers(3, 10))))
        cv2.putText(img, s, (x0, y0 + 20), int(rng.integers(0, 7)), rng.uniform(0.5, 2.0), _rand_color(rng),
                    int(rng.integers(1, 4)))
    anchors = []
    for _ in range(int(rng.integers(1, 4))):
        w = int(W * rng.uniform(0.12, 0.6))
        x = rng.uniform(0, max(W - w, 1))
        y = rng.uniform(horizon * 0.5, max(H - w * 0.9, horizon * 0.5 + 1))
        anchors.append(_draw_vehicle(img, rng, x, y, w))
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 2.0))
    img = effects.noise(img, rng)
    # рекламный щит/экран со знаком (is_vehicle = 0)
    if rng.random() < 0.15:
        bw = int(W * rng.uniform(0.2, 0.4))
        bh = int(bw * rng.uniform(0.5, 0.8))
        bx, by = int(rng.uniform(0, W - bw)), int(rng.uniform(0, max(H * 0.5 - bh, 1)))
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), _rand_color(rng), -1)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (30, 30, 30), max(2, bw // 60))
        cv2.putText(img, "SALE" if rng.random() < 0.5 else "AUTO", (bx + bw // 10, by + bh // 4),
                    0, bw / 250, _rand_color(rng), max(1, bw // 150))
        anchors.append((bx + bw * 0.1, by + bh * 0.45, bw * 0.8, bh * 0.4, 0))
    return img, anchors

def project_quad(rng, size_mm, center, width_px, max_yaw=50, max_pitch=25, max_roll=12):
    Wm, Hm = size_mm
    yaw = math.radians(rng.uniform(-max_yaw, max_yaw) * (rng.random() < 0.6))
    pitch = math.radians(rng.uniform(-max_pitch, max_pitch) * (rng.random() < 0.5))
    roll = math.radians(np.clip(rng.normal(0, max_roll / 2.5), -max_roll, max_roll))
    P = np.array([[-Wm / 2, -Hm / 2, 0], [Wm / 2, -Hm / 2, 0], [Wm / 2, Hm / 2, 0], [-Wm / 2, Hm / 2, 0]])
    Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
    Q = P @ (Rz @ Rx @ Ry).T
    Z = Wm * rng.uniform(1.5, 6.0)
    q = Q[:, :2] / (Q[:, 2:3] + Z)
    q -= q.mean(axis=0)
    span = q[:, 0].max() - q[:, 0].min()
    q = q * (width_px / span) + np.asarray(center)
    ang = abs(math.degrees(yaw)) > 25 or abs(math.degrees(pitch)) > 15 or abs(math.degrees(roll)) > 8
    return q.astype(np.float32), ang


def paste_plate(img, plate_img, plate_mask, quad, rng, holder=True):
    ph, pw = plate_img.shape[:2]
    qw = np.linalg.norm(quad[1] - quad[0])
    scale = min(1.0, 2.0 * qw / pw)
    if scale < 1.0:
        plate_img = cv2.resize(plate_img, (max(int(pw * scale), 4), max(int(ph * scale), 4)), interpolation=cv2.INTER_AREA)
        plate_mask = cv2.resize(plate_mask, (plate_img.shape[1], plate_img.shape[0]), interpolation=cv2.INTER_AREA)
        ph, pw = plate_img.shape[:2]
    src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], np.float32)
    H, W = img.shape[:2]
    if holder and rng.random() < 0.45:
        m = int(max(2, 0.04 * ph))
        col = _rand_color(rng, 0, 60) if rng.random() < 0.8 else _rand_color(rng, 150, 230)
        big = cv2.copyMakeBorder(plate_img, m, m, m, m, cv2.BORDER_CONSTANT, value=col)
        bmask = cv2.copyMakeBorder(np.full_like(plate_mask, 255), m, m, m, m, cv2.BORDER_CONSTANT, value=255)
        big[m:-m, m:-m][plate_mask == 0] = col
        src_b = src + m
        M = cv2.getPerspectiveTransform(src_b, quad.astype(np.float32))
        _composite(img, big, bmask, M, W, H)
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    _composite(img, plate_img, plate_mask, M, W, H)


def _composite(img, src_img, src_mask, M, W, H):
    warped = cv2.warpPerspective(src_img, M, (W, H), flags=cv2.INTER_LINEAR)
    a = cv2.warpPerspective(src_mask, M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
    ys, xs = np.nonzero(a > 0)
    if len(xs) == 0:
        return
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    aa = a[y0:y1, x0:x1, None]
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * (1 - aa) + warped[y0:y1, x0:x1] * aa).astype(np.uint8)


def _match_light(plate_img, bg, quad, rng):
    x0, y0 = np.maximum(quad.min(0).astype(int), 0)
    x1, y1 = quad.max(0).astype(int)
    patch = bg[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
    m = patch.mean() if patch.size else 128
    g = np.clip((m / 140) ** 0.4, 0.55, 1.1) * rng.uniform(0.8, 1.1)
    return np.clip(plate_img.astype(np.float32) * g, 0, 255).astype(np.uint8)

def _plate_width_for_anchor(rng, plate, anchor_w):
    return anchor_w * rng.uniform(0.2, 0.4) * plate.size_mm[0] / 520


def make_scene(rng, pool, type_probs=None, allow_empty=True):
    img, anchors = pool.sample(rng)
    H, W = img.shape[:2]
    rng.shuffle(anchors)
    anns, cond = [], set()
    n = 0 if (allow_empty and rng.random() < 0.03) else min(len(anchors) + int(rng.random() < 0.25), int(rng.choice([1, 1, 1, 2, 2, 3])))
    for i in range(n):
        plate = plates.render_random(rng, type_probs=type_probs)
        pimg, pc = effects.degrade_plate(plate.image, rng)
        cond.update(pc)
        if i < len(anchors):
            ax, ay, aw, ah, veh = anchors[i]
            pw = _plate_width_for_anchor(rng, plate, aw)
            c = (ax + aw * rng.uniform(0.4, 0.6), ay + ah * rng.uniform(0.3, 0.7))
        else:
            veh = 1
            pw = W * rng.uniform(0.03, 0.25) * plate.size_mm[0] / 520
            c = (rng.uniform(0.1, 0.9) * W, rng.uniform(0.2, 0.95) * H)
        pw = max(pw, 18)
        quad, ang = project_quad(rng, plate.size_mm, c, pw)
        if quad[:, 0].min() < 0 or quad[:, 1].min() < 0 or quad[:, 0].max() >= W or quad[:, 1].max() >= H:
            continue
        if any(_iou(quad, a["quad"]) > 0.05 for a in anns):
            continue
        if ang:
            cond.add("angle")
        pimg = _match_light(pimg, img, quad, rng)
        paste_plate(img, pimg, plate.mask, quad, rng)
        anns.append(dict(quad=order_quad(quad), text=plate.text, plate_type=plate.plate_type,
                         kind=plate.kind, two_line=plate.two_line, is_vehicle=veh))
    img, sc = effects.degrade_scene(img, rng)
    cond.update(sc)
    return img, anns, sorted(cond)


def _iou(q1, q2):
    a0, a1 = q1.min(0), q1.max(0)
    b0, b1 = q2.min(0), q2.max(0)
    iw = max(0, min(a1[0], b1[0]) - max(a0[0], b0[0]))
    ih = max(0, min(a1[1], b1[1]) - max(a0[1], b0[1]))
    inter = iw * ih
    u = (a1 - a0).prod() + (b1 - b0).prod() - inter
    return inter / max(u, 1e-6)


def make_ocr_sample(rng, pool, type_probs=None, text=None):
    plate = plates.render_random(rng, type_probs=type_probs, text=text)
    pimg, _ = effects.degrade_plate(plate.image, rng)
    # целевая ширина знака в пикселях: от «очень далеко» до крупного плана
    pw = float(np.exp(rng.uniform(np.log(40), np.log(320))))
    if plate.two_line:
        pw *= rng.uniform(0.6, 0.9)
    ph = pw * plate.size_mm[1] / plate.size_mm[0]
    cw, ch = int(pw * 1.8) + 8, int(ph * 2.2) + 8
    canvas = pool.random_patch(rng, cw, ch)
    quad, _ = project_quad(rng, plate.size_mm, (cw / 2, ch / 2), pw, max_yaw=45, max_pitch=20, max_roll=10)
    pimg = _match_light(pimg, canvas, quad, rng)
    paste_plate(canvas, pimg, plate.mask, quad, rng)
    canvas, _ = effects.degrade_scene(canvas, rng)
    canvas = effects.jpeg(canvas, rng, 25, 95)
    # шум положения углов ~ ошибка детектора ключевых точек
    diag = np.linalg.norm(quad[2] - quad[0])
    nq = quad + rng.normal(0, 0.015 * diag, quad.shape).astype(np.float32)
    crop = rectify(canvas, nq, plate.two_line, pad=rng.uniform(0.0, 0.06))
    return crop, plate


def make_none_sample(rng, pool):
    w = int(rng.integers(40, 300))
    h = int(w / rng.uniform(1.5, 5.0))
    crop = pool.random_patch(rng, w, h)
    if rng.random() < 0.4:
        s = "".join(rng.choice(list("ABCDEFGHKMOPTX0123456789"), int(rng.integers(2, 8))))
        cv2.putText(crop, s, (int(rng.integers(0, w // 3)), int(h * rng.uniform(0.5, 0.9))),
                    int(rng.integers(0, 7)), h / 40, _rand_color(rng), max(1, h // 15))
    crop, _ = effects.degrade_scene(crop, rng)
    quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    return rectify(crop, quad, bool(rng.random() < 0.3), pad=0)
