"""Experiment executor. Runs in a child process so crashes, hangs and OOM never kill the agent.

usage: python -m agent.worker job.json
job: {"mode": "eval"|"final", "cfg": {...}, "cache": path, "out": path, "n_iter": int|null}
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import traceback

import numpy as np
import pandas as pd

from . import features, metrics, models


def _fwd(tt, y):
    return {"none": y, "log1p": np.log1p(y), "sqrt": np.sqrt(y)}[tt]


def _inv(tt, p):
    return {"none": p, "log1p": np.expm1(p), "sqrt": np.square(np.clip(p, 0, None))}[tt]


def prepare(cfg, cache):
    ctx = {**cache["ctx"], "top_features": cfg.get("top_features")}
    X = features.build(cache["X_all"], cfg, ctx)
    n = cache["n_train"]
    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True)


def run(job):
    cache = pickle.load(open(job["cache"], "rb"))
    spec = cache["ctx"]
    cfg = job["cfg"]
    task, metric = spec["task_type"], spec["metric"]
    tt = cfg.get("target_transform", "none") if task == "regression" else "none"
    Xtr_all, Xte = prepare(cfg, cache)
    y = cache["y"]
    model, params, seed = cfg.get("model", "lgbm"), cfg.get("params", {}), job.get("seed", 42)
    t0 = time.time()
    nc = spec["n_classes"]

    if job["mode"] == "eval":
        scores, iters, vals, yvals = [], [], [], []
        for tr, va in cache["splits"]:
            (pva,), used = models.fit_predict(
                model, params, task, Xtr_all.iloc[tr], _fwd(tt, y[tr]), Xtr_all.iloc[va],
                _fwd(tt, y[va]), [Xtr_all.iloc[va]], nc, seed)
            pva = _inv(tt, pva)
            if not np.all(np.isfinite(pva)):
                raise ValueError("non-finite predictions")
            scores.append(metrics.score(metric, y[va], pva))
            iters.append(used)
            vals.append(pva)
            yvals.append(y[va])
        np.save(job["out"] + ".val.npy", np.concatenate(vals))
        return {"score": float(np.mean(scores)), "per_split": [float(s) for s in scores],
                "n_iter": int(np.mean(iters)),
                "importance": dict(sorted(models.LAST_IMPORTANCE.items(), key=lambda kv: -kv[1])[:20]), "n_features": Xtr_all.shape[1],
                "seconds": time.time() - t0}

    # final: refit on the whole training set, predict test
    n_iter = max(50, int(job["n_iter"] * 1.15))
    (pte,), _ = models.fit_predict(model, params, task, Xtr_all, _fwd(tt, y), None, None, [Xte],
                                   nc, seed, n_iter=n_iter)
    pte = _inv(tt, pte)
    if not np.all(np.isfinite(pte)):
        raise ValueError("non-finite predictions")
    np.save(job["out"] + ".test.npy", pte)
    return {"seconds": time.time() - t0}


def main():
    job = json.load(open(sys.argv[1]))
    try:
        res = {"ok": True, **run(job)}
    except Exception as e:  # reported back to the agent (and to the LLM as feedback)
        tb = traceback.format_exc()
        res = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": tb[-700:]}
    json.dump(res, open(job["out"] + ".json", "w"))


if __name__ == "__main__":
    main()
