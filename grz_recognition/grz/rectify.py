"""Выпрямление знака по четырём углам и приведение к единому входу OCR.

Однострочный знак (520x112) выпрямляется целиком в 224x48.
Двухстрочный (тип 1А 290x170, мотоциклетный и т.п.) выпрямляется в 232x136,
делится на верхнюю и нижнюю строку, строки ставятся рядом в одну «ленту»
224x48. Благодаря этому один и тот же CTC-распознаватель читает обе
компоновки, а добавление новой компоновки сводится к новому правилу раскладки.
"""
import cv2
import numpy as np

OCR_W, OCR_H = 224, 48
TWO_W, TWO_H = 232, 136
# Границы строк двухстрочного знака (доля высоты), с небольшим перекрытием.
TOP_ROW = (0.02, 0.54)
BOTTOM_ROW = (0.46, 0.98)


def order_quad(pts):
    """Упорядочивает 4 точки по часовой стрелке от левой верхней."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # по часовой (ось y вниз), начиная с угла ~ -135°
    s = pts.sum(axis=1)
    k = int(np.argmin(s))
    return np.roll(pts, -k, axis=0)


def _warp(img, quad, w, h, pad=0.0):
    quad = order_quad(quad)
    if pad:
        c = quad.mean(axis=0)
        quad = c + (quad - c) * (1 + pad)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def rectify(img, quad, two_line, pad=0.03):
    """Возвращает BGR-изображение 224x48 для распознавателя."""
    if not two_line:
        return _warp(img, quad, OCR_W, OCR_H, pad)
    sq = _warp(img, quad, TWO_W, TWO_H, pad)
    half_w = OCR_W // 2
    rows = []
    for a, b in (TOP_ROW, BOTTOM_ROW):
        r = sq[int(a * TWO_H):int(b * TWO_H)]
        rows.append(cv2.resize(r, (half_w, OCR_H), interpolation=cv2.INTER_AREA))
    return np.hstack(rows)


def to_tensor(crops):
    """Список BGR 224x48 -> float32 NCHW, нормировка в [-1, 1]."""
    x = np.stack(crops).astype(np.float32) / 127.5 - 1.0
    return x.transpose(0, 3, 1, 2).copy()
