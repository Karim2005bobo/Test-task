"""Task understanding: read the statement and files, infer the ML task, pick a validation scheme."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("agent.data")

METRIC_PATTERNS = [
    ("macro_f1", r"macro[\s_-]*f1"), ("weighted_f1", r"weighted[\s_-]*f1"),
    ("rmse", r"\brmse\b"), ("mae", r"\bmae\b"), ("logloss", r"log[\s_-]?loss"),
    ("accuracy", r"\baccuracy\b"), ("r2", r"\br2\b|r\^2"),
]


@dataclass
class TaskSpec:
    id_col: str
    target: str
    task_type: str                 # regression | classification
    metric: str
    cat_cols: list[str]
    num_cols: list[str]
    classes: list | None = None
    description: str = ""
    dictionary: dict = field(default_factory=dict)   # feature -> "unit; description"
    split_kind: str = "random"     # random | forward
    split_col: str | None = None

    @property
    def n_classes(self) -> int:
        return len(self.classes) if self.classes else 0


def read_statement(data_dir: Path) -> str:
    chunks = []
    for p in sorted(data_dir.glob("*.md")) + sorted(data_dir.glob("*.txt")):
        if p.name.lower().startswith("requirements"):
            continue
        try:
            chunks.append(p.read_text(encoding="utf-8")[:20000])
        except OSError:
            pass
    return "\n\n".join(chunks)


def detect_metric(text: str, task_type: str) -> str:
    low = text.lower()
    hits = [(low.find(m.group(0)), name) for name, pat in METRIC_PATTERNS
            if (m := re.search(pat, low))]
    for _, name in sorted(hits):
        if (name in ("rmse", "mae", "r2")) == (task_type == "regression"):
            return name
    return "rmse" if task_type == "regression" else "macro_f1"


def load_task(data_dir: Path, statement: str = ""):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sub = pd.read_csv(data_dir / "sample_submission.csv", nrows=5)
    dd_path = data_dir / "data_dictionary.csv"
    dd = pd.read_csv(dd_path) if dd_path.exists() else None

    id_col, target = sub.columns[0], sub.columns[1]
    if target not in train.columns:
        extra = [c for c in train.columns if c not in test.columns]
        target = extra[0]
    y = train[target]

    dictionary, dd_cat = {}, set()
    if dd is not None and "feature" in dd.columns:
        for _, r in dd.iterrows():
            dictionary[r["feature"]] = f"{r.get('unit', '')}; {r.get('description_ru', '')}"
            if str(r.get("dtype", "")).lower() in ("category", "categorical", "object", "string"):
                dd_cat.add(r["feature"])

    feats = [c for c in test.columns if c != id_col and c in train.columns]
    cat_cols = [c for c in feats if c in dd_cat or not pd.api.types.is_numeric_dtype(train[c])]
    num_cols = [c for c in feats if c not in cat_cols]

    n_unique = y.nunique()
    is_float = pd.api.types.is_float_dtype(y) and not np.all(np.mod(y.dropna(), 1) == 0)
    task_type = "regression" if (pd.api.types.is_numeric_dtype(y) and (is_float or n_unique > 50)) \
        else "classification"
    description = statement or read_statement(data_dir)
    metric = detect_metric(description, task_type)

    classes = None
    if task_type == "classification":
        classes = sorted(y.dropna().unique().tolist())
    spec = TaskSpec(id_col, target, task_type, metric, cat_cols, num_cols, classes,
                    description, dictionary)
    spec.split_kind, spec.split_col = detect_split(train, test, feats, spec)
    log.info("task=%s metric=%s target=%s id=%s split=%s(%s) cats=%d nums=%d",
             task_type, metric, target, id_col, spec.split_kind, spec.split_col,
             len(cat_cols), len(num_cols))
    return train, test, spec


def detect_split(train, test, feats, spec):
    """If test lies strictly after train along a period-like column, validate forward in time."""
    best = None
    for c in feats:
        if not pd.api.types.is_numeric_dtype(train[c]):
            continue
        tr, te = train[c].dropna(), test[c].dropna()
        if tr.empty or te.empty or tr.nunique() > 40 or tr.nunique() < 2:
            continue
        if not np.all(np.mod(tr, 1) == 0):
            continue
        if te.min() > tr.max():
            name = c.lower()
            pref = 0 if ("year" in name or "period" in name or "time" in name or "date" in name) else 1
            if best is None or pref < best[0]:
                best = (pref, c)
    return ("forward", best[1]) if best else ("random", None)


def make_splits(train: pd.DataFrame, spec: TaskSpec, seed: int, n_val: int = 2):
    """List of (train_idx, val_idx). Fixed for the whole run so experiments are comparable."""
    n = len(train)
    if spec.split_kind == "forward":
        vals = np.sort(train[spec.split_col].dropna().unique())
        splits = []
        for v in vals[::-1][:n_val][::-1]:
            va = np.where(train[spec.split_col].values == v)[0]
            tr = np.where(train[spec.split_col].values < v)[0]
            if len(tr) >= 0.1 * n and len(va) > 0:
                splits.append((tr, va))
        if splits:
            return splits
    if spec.task_type == "classification":
        # stratified: two disjoint validation folds out of five (rare classes need enough support)
        from sklearn.model_selection import StratifiedKFold
        y = train[spec.target].values
        counts = pd.Series(y).value_counts()
        k = int(min(5, max(2, counts.min())))
        skf = StratifiedKFold(k, shuffle=True, random_state=seed)
        return [(np.sort(tr), np.sort(va)) for _, (tr, va) in zip(range(2), skf.split(np.zeros(n), y))]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    cut = int(0.8 * n)
    tr, va = perm[:cut], perm[cut:]
    return [(np.sort(tr), np.sort(va))]


def describe_profile(train: pd.DataFrame, spec: TaskSpec, top: int = 12) -> str:
    y = train[spec.target]
    lines = [f"rows={len(train)} features={len(spec.cat_cols) + len(spec.num_cols)} "
             f"cat={len(spec.cat_cols)}"]
    if spec.task_type == "regression":
        lines.append("target: " + ", ".join(f"{k}={v:.3g}" for k, v in y.describe().items())
                     + f", skew={y.skew():.2f}")
    else:
        lines.append("target class shares: " + str((y.value_counts(normalize=True).round(3)).to_dict()))
    miss = train[spec.cat_cols + spec.num_cols].isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(top)
    lines.append("missing share: " + str(miss.round(3).to_dict()))
    for c in spec.cat_cols[:6]:
        lines.append(f"cat {c}: {train[c].nunique()} levels")
    return "\n".join(lines)
