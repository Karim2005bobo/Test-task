"""End-to-end orchestration: understand -> safe baseline -> AutoResearch -> blend -> submission."""
from __future__ import annotations

import json
import logging
import os
import pickle
import time

import numpy as np
import pandas as pd

from . import metrics
from .config import Config
from .data import describe_profile, load_task, make_splits
from .llm import LLM
from .research import Researcher, Runner, Experiment

log = logging.getLogger("agent.pipeline")


def write_submission(path, spec, test, pred, classes):
    """Atomic write so a valid file exists even if the process is killed mid-way."""
    if spec.task_type == "classification":
        labels = np.asarray(classes)[np.asarray(pred).argmax(1)]
    else:
        labels = np.asarray(pred, dtype=float)
    sub = pd.DataFrame({spec.id_col: test[spec.id_col].values, spec.target: labels})
    assert sub[spec.id_col].is_unique and np.all(pd.notna(sub[spec.target]))
    tmp = str(path) + ".tmp"
    sub.to_csv(tmp, index=False)
    os.replace(tmp, path)


def greedy_blend(val_preds: list[np.ndarray], y, metric, task, steps=25):
    """Caruana hill-climbing with replacement. Returns weights summing to 1."""
    loss_metric = "logloss" if task == "classification" else metric
    counts = np.zeros(len(val_preds))
    cur, best_s = None, -np.inf
    for _ in range(steps):
        cand = []
        for i, p in enumerate(val_preds):
            mix = p if cur is None else (cur * counts.sum() + p) / (counts.sum() + 1)
            cand.append((metrics.score(loss_metric, y, mix), i, mix))
        s, i, mix = max(cand, key=lambda t: t[0])
        if s <= best_s + 1e-12 and cur is not None:
            break
        counts[i] += 1
        best_s, cur = s, mix
    return counts / counts.sum()


def run(config: Config):
    log.info("config: data=%s out=%s budget=%.0fs", config.data_dir, config.output_path, config.time_budget)
    train, test, spec = load_task(config.data_dir, config.statement)
    feats = spec.cat_cols + spec.num_cols
    y_raw = train[spec.target]
    classes = spec.classes
    if spec.task_type == "classification":
        y = pd.Categorical(y_raw, categories=classes).codes.astype(int)
    else:
        y = y_raw.astype(float).values

    X_all = pd.concat([train[feats], test[feats]], ignore_index=True)
    splits = make_splits(train, spec, config.seed)
    ctx = dict(task_type=spec.task_type, metric=spec.metric, target=spec.target, cat_cols=spec.cat_cols,
               num_cols=spec.num_cols, n_classes=spec.n_classes, split_kind=spec.split_kind,
               split_col=spec.split_col, description=spec.description, dictionary=spec.dictionary)
    cache_path = config.workdir / "cache.pkl"
    pickle.dump(dict(X_all=X_all, n_train=len(train), y=y, splits=splits, ctx=ctx), open(cache_path, "wb"))
    log.info("validation: %s", [(len(a), len(b)) for a, b in splits])

    llm = LLM(config.llm_base_url, config.llm_api_key, config.llm_model, config.llm_token_budget,
              config.llm_max_calls, config.llm_timeout)
    runner = Runner(config, cache_path)
    res = Researcher(config, llm, runner, ctx, feats)

    # 1) task analysis by the LLM (optional) -------------------------------------------------
    res.analyse_task(describe_profile(train, spec))

    # 2) fast reliable baseline + immediate submission ---------------------------------------
    base_cfg = dict(model="lgbm", params={}, recipes=[], feature_codes=[], drop_columns=[],
                    target_transform="none")
    r, out = runner.run("eval", base_cfg, timeout=config.remaining() * 0.3, tag="base")
    if not r.get("ok"):
        raise RuntimeError(f"baseline failed: {r.get('error')}\n{r.get('trace')}")
    base = Experiment(0, "baseline LightGBM on raw features", base_cfg, "improved", r["score"], 0.0,
                      r["seconds"], r["n_iter"], out)
    res.best = base
    res.history.append(base)
    res.top_features = list(r.get("importance", {}))
    base.cfg = {**base_cfg, "top_features": res.top_features}
    log.info("baseline %s=%.5f (%d iters, %.0fs)", spec.metric, abs(r["score"]), r["n_iter"], r["seconds"])
    fr, fout = runner.run("final", base.cfg, timeout=config.remaining() * 0.4, n_iter=base.n_iter, tag="finalbase")
    if fr.get("ok"):
        write_submission(config.output_path, spec, test, np.load(fout + ".test.npy"), classes)
        log.info("baseline submission written")
        final_cache = {0: fout + ".test.npy"}
    else:
        final_cache = {}
        log.error("baseline final fit failed: %s", fr.get("error"))

    # 3) AutoResearch loop -------------------------------------------------------------------
    res.run_loop(config.research_frac)

    # 4) final: refit the best distinct pipelines on all data, blend on validation -----------
    ok = sorted([e for e in res.history if e.status in ("improved", "rejected") and e.out],
                key=lambda e: -e.score)
    pool, seen = [], set()
    for e in ok:   # diversity: at most 2 per model family
        k = e.cfg["model"]
        if sum(1 for p in pool if p.cfg["model"] == k) < 2:
            pool.append(e)
        if len(pool) >= 4:
            break
    yv = np.concatenate([y[va] for _, va in splits])
    members, vals = [], []
    for e in pool:
        if e.id in final_cache:
            tp = final_cache[e.id]
        else:
            est = max(e.seconds, 30) * 0.8
            if config.remaining() < est * 1.5:
                log.info("skip refit of exp#%d: no time", e.id)
                continue
            fr, fout = runner.run("final", e.cfg, timeout=config.remaining() * 0.8, n_iter=e.n_iter,
                                  tag=f"final{e.id}")
            if not fr.get("ok"):
                log.warning("final fit exp#%d failed: %s", e.id, fr.get("error"))
                continue
            tp = fout + ".test.npy"
            final_cache[e.id] = tp
        members.append((e, tp))
        vals.append(np.load(e.out + ".val.npy"))

    if members:
        w = greedy_blend(vals, yv, spec.metric, spec.task_type)
        test_pred = sum(wi * np.load(tp) for wi, (_, tp) in zip(w, members) if wi > 0)
        val_pred = sum(wi * v for wi, v in zip(w, vals))
        if spec.task_type == "classification":
            cw, s_t = metrics.tune_class_weights(val_pred, yv, spec.metric)
            s_0 = metrics.score(spec.metric, yv, val_pred)
            if s_t > s_0 + 0.003:     # require a real gain: weights are fitted on the same validation
                test_pred = test_pred * cw
                log.info("class weights applied: %.4f -> %.4f", s_0, s_t)
        log.info("blend weights %s | blend val score %.5f (best single %.5f)",
                 {e.id: round(float(x), 2) for (e, _), x in zip(members, w)},
                 metrics.score(spec.metric, yv, val_pred), ok[0].score)
        write_submission(config.output_path, spec, test, test_pred, classes)

    report = [dict(id=e.id, hypothesis=e.hypothesis, source=e.source, status=e.status, score=e.score,
                   delta=e.delta, seconds=round(e.seconds, 1), error=e.error) for e in res.history]
    (config.workdir / "experiments.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    log.info("done in %.0fs; llm calls=%d tokens=%d; submission=%s", time.time() - config.start_time,
             llm.calls, llm.tokens_used, config.output_path)
