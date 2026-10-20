"""Metrics. `score` is always oriented so that larger is better."""
from __future__ import annotations

import numpy as np

METRICS = ("rmse", "mae", "r2", "macro_f1", "weighted_f1", "accuracy", "logloss")


def _f1_per_class(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros(k), where=den > 0), cm.sum(1)


def raw_metric(name: str, y, pred) -> float:
    y = np.asarray(y)
    if name == "rmse":
        return float(np.sqrt(np.mean((y - pred) ** 2)))
    if name == "mae":
        return float(np.mean(np.abs(y - pred)))
    if name == "r2":
        return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    if name == "logloss":
        p = np.clip(pred, 1e-15, 1)
        return float(-np.mean(np.log(p[np.arange(len(y)), y.astype(int)])))
    labels = pred.argmax(1) if pred.ndim == 2 else pred
    y = y.astype(int)
    labels = labels.astype(int)
    k = int(max(y.max(), labels.max())) + 1
    if name == "accuracy":
        return float(np.mean(y == labels))
    f1, support = _f1_per_class(y, labels, k)
    if name == "macro_f1":
        return float(f1[support > 0].mean()) if (support > 0).any() else 0.0
    if name == "weighted_f1":
        return float((f1 * support).sum() / support.sum())
    raise ValueError(name)


LOWER_IS_BETTER = {"rmse", "mae", "logloss"}


def score(name: str, y, pred) -> float:
    v = raw_metric(name, y, pred)
    return -v if name in LOWER_IS_BETTER else v


def tune_class_weights(proba: np.ndarray, y, metric: str = "macro_f1", rounds: int = 3):
    """Coordinate ascent on per-class multiplicative weights (helps macro-F1 on rare classes)."""
    k = proba.shape[1]
    w = np.ones(k)
    best = score(metric, y, proba * w)
    grid = np.array([0.4, 0.6, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0, 3.0])
    for _ in range(rounds):
        for c in range(k):
            for g in grid:
                w2 = w.copy()
                w2[c] *= g
                s = score(metric, y, proba * w2)
                if s > best + 1e-9:
                    best, w = s, w2
    return w, best
