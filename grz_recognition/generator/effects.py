"""Аугментации, имитирующие реальные условия съёмки: грязь, блики, тени,
ночь, дождь, снег, смаз, сжатие JPEG. Каждая функция детерминирована
относительно переданного генератора случайных чисел rng.
"""
import cv2
import numpy as np


# ------------------------------------------------------------- на знаке

def plate_dirt(img, rng, strength=None):
    """Пятна грязи/пыли: полупрозрачные коричнево-серые кляксы."""
    h, w = img.shape[:2]
    s = strength if strength is not None else rng.uniform(0.2, 0.7)
    layer = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(3, 25))):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        ax = (int(rng.uniform(0.02, 0.25) * w), int(rng.uniform(0.05, 0.5) * h))
        cv2.ellipse(layer, c, ax, float(rng.uniform(0, 180)), 0, 360, float(rng.uniform(0.3, 1)), -1)
    # нижняя кромка знака обычно грязнее
    grad = np.linspace(0, 1, h, dtype=np.float32)[:, None] ** 2
    layer = np.clip(layer + grad * rng.uniform(0, 0.8), 0, 1)
    layer = cv2.GaussianBlur(layer, (0, 0), max(1.0, w / 80))
    noise = cv2.resize(rng.random((max(h // 6, 2), max(w // 6, 2))).astype(np.float32), (w, h))
    alpha = (layer * noise * s)[..., None]
    color = np.array([rng.uniform(40, 90), rng.uniform(60, 110), rng.uniform(80, 130)], np.float32)
    return (img.astype(np.float32) * (1 - alpha) + color * alpha).astype(np.uint8)


def plate_glare(img, rng):
    """Блик: яркое размытое пятно, частично «засвечивающее» символы."""
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.float32)
    c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
    ax = (int(rng.uniform(0.05, 0.4) * w), int(rng.uniform(0.2, 1.0) * h))
    cv2.ellipse(layer, c, ax, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), max(1.0, w / 30)) * rng.uniform(0.4, 0.95)
    return (img.astype(np.float32) * (1 - layer[..., None]) + 255 * layer[..., None]).astype(np.uint8)


def plate_shadow(img, rng):
    """Резкая или мягкая тень (от рамки, бампера, веток)."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    pts = rng.uniform([-0.2 * w, -0.2 * h], [1.2 * w, 1.2 * h], (int(rng.integers(3, 6)), 2)).astype(np.int32)
    cv2.fillPoly(mask, [cv2.convexHull(pts)], 1.0)
    k = rng.uniform(0.5, w / 25)
    mask = cv2.GaussianBlur(mask, (0, 0), k)
    f = 1 - mask[..., None] * rng.uniform(0.3, 0.7)
    return (img.astype(np.float32) * f).astype(np.uint8)


def plate_fade(img, rng):
    """Выцветание/износ краски: снижение контраста."""
    k = rng.uniform(0.55, 0.9)
    m = img.mean(axis=(0, 1), keepdims=True)
    return np.clip(m + (img.astype(np.float32) - m) * k, 0, 255).astype(np.uint8)


def degrade_plate(img, rng):
    """Возвращает (изображение знака, список условий: dirt, glare)."""
    cond = []
    if rng.random() < 0.35:
        img = plate_dirt(img, rng)
        cond.append("dirt")
    if rng.random() < 0.2:
        img = plate_fade(img, rng)
    if rng.random() < 0.2:
        img = plate_shadow(img, rng)
    if rng.random() < 0.15:
        img = plate_glare(img, rng)
        cond.append("glare")
    return img, cond


# ------------------------------------------------------------- на кадре

def night(img, rng):
    f = rng.uniform(0.15, 0.45)
    out = img.astype(np.float32) * f
    out[..., 0] *= rng.uniform(0.9, 1.3)   # холодный/жёлтый оттенок освещения
    out[..., 2] *= rng.uniform(0.9, 1.3)
    h, w = img.shape[:2]
    # пятна освещения (фонари, фары, подсветка номера)
    for _ in range(int(rng.integers(1, 4))):
        layer = np.zeros((h, w), np.float32)
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(layer, c, int(rng.uniform(0.05, 0.4) * max(h, w)), 1.0, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), max(h, w) / 12)
        out += img.astype(np.float32) * layer[..., None] * rng.uniform(0.3, 1.0)
    out += rng.normal(0, rng.uniform(3, 10), img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def rain(img, rng):
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.float32)
    n = int(rng.uniform(200, 1500) * h * w / 640 / 640)
    ang = rng.uniform(-0.3, 0.3)
    ln = rng.uniform(8, 25) * max(h, w) / 640
    xs, ys = rng.uniform(0, w, n), rng.uniform(0, h, n)
    for x, y in zip(xs, ys):
        cv2.line(layer, (int(x), int(y)), (int(x + ln * ang), int(y + ln)), float(rng.uniform(0.3, 0.8)), 1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0) * 0.6
    out = img.astype(np.float32) * (1 - layer[..., None]) + 210 * layer[..., None]
    out = out * rng.uniform(0.7, 0.95)  # пасмурно
    return np.clip(cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.0)), 0, 255).astype(np.uint8)


def snow(img, rng):
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.float32)
    n = int(rng.uniform(300, 2000) * h * w / 640 / 640)
    for x, y in zip(rng.uniform(0, w, n), rng.uniform(0, h, n)):
        cv2.circle(layer, (int(x), int(y)), int(rng.uniform(1, 3.5) * max(h, w) / 640), float(rng.uniform(0.5, 1)), -1)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
    out = img.astype(np.float32) * (1 - layer[..., None]) + 245 * layer[..., None]
    out = out * rng.uniform(0.85, 1.0) + rng.uniform(10, 40)  # дымка
    return np.clip(out, 0, 255).astype(np.uint8)


def motion_blur(img, rng, k=None):
    k = k or int(rng.integers(3, 12))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(-30, 30)), 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def color_jitter(img, rng):
    out = img.astype(np.float32)
    out = out * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30)
    out = out * rng.uniform(0.9, 1.1, 3)
    gray = out.mean(axis=2, keepdims=True)
    out = gray + (out - gray) * rng.uniform(0.6, 1.3)   # насыщенность
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg(img, rng, lo=30, hi=95):
    q = int(rng.integers(lo, hi))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def noise(img, rng):
    return np.clip(img + rng.normal(0, rng.uniform(2, 12), img.shape), 0, 255).astype(np.uint8)


def degrade_scene(img, rng):
    """Возвращает (изображение, список условий съёмки для meta.csv)."""
    cond = []
    r = rng.random()
    if r < 0.2:
        img = night(img, rng)
        cond.append("night")
    else:
        cond.append("day")
    r = rng.random()
    if r < 0.12:
        img = rain(img, rng)
        cond.append("rain")
    elif r < 0.22:
        img = snow(img, rng)
        cond.append("snow")
    if rng.random() < 0.15:
        img = motion_blur(img, rng)
        cond.append("motion_blur")
    elif rng.random() < 0.25:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.4))
    img = color_jitter(img, rng)
    if rng.random() < 0.5:
        img = noise(img, rng)
    return img, cond
