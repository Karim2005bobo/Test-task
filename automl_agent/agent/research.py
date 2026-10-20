"""AutoResearch loop: hypothesis -> experiment -> evaluation on a fixed validation -> keep/discard."""
from __future__ import annotations

import copy
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import features
from .llm import LLM

log = logging.getLogger("agent.research")
PKG_ROOT = Path(__file__).resolve().parents[1]

SYSTEM = """You are the research brain of an autonomous ML agent working on a tabular task.
Each turn you propose ONE hypothesis as a JSON patch to the current best pipeline. The agent
runs it on a fixed validation (the hidden test is never visible) and tells you the result.
Rules:
- Reply with a single JSON object, nothing else.
- Hypotheses must be specific and justified by the domain, the data profile or the previous results.
- Do not repeat experiments from the history; if something helped, build on it; if several attempts
  failed in one direction, change direction.
- Feature code defines `def make_features(df): ...` returning df with NEW columns added. df holds
  train+test feature rows only (no target, no id), so the target cannot be used. Allowed imports:
  numpy, pandas, math. Guard divisions by zero; keep it vectorised and fast.
- Only improve generalisation to the hidden test; never try to game the metric."""

SCHEMA = """JSON schema:
{"reflection": "what the last results tell us (1-2 sentences)",
 "hypothesis": "one sentence",
 "change": {
   "add_feature_code": "optional python source with make_features(df)",
   "add_recipes": ["optional subset of: %s"],
   "drop_columns": ["optional column names to remove"],
   "model": "optional: lgbm | catboost | xgb",
   "params": {"optional": "model hyper-parameters to override"},
   "target_transform": "optional (regression only): none | log1p | sqrt",
   "reset_features": false}}"""


@dataclass
class Experiment:
    id: int
    hypothesis: str
    cfg: dict
    status: str = "pending"      # improved | rejected | error | duplicate
    score: float | None = None
    delta: float | None = None
    seconds: float = 0.0
    n_iter: int | None = None
    out: str | None = None
    error: str | None = None
    source: str = "builtin"


class Runner:
    def __init__(self, config, cache_path: Path):
        self.config, self.cache_path = config, cache_path
        self.count = 0

    def run(self, mode: str, cfg: dict, timeout: float, n_iter=None, tag="exp", seed=None) -> tuple[dict, str]:
        self.count += 1
        out = str(self.config.workdir / f"{tag}_{self.count}")
        job = {"mode": mode, "cfg": cfg, "cache": str(self.cache_path), "out": out, "n_iter": n_iter,
               "seed": seed if seed is not None else self.config.seed}
        jp = out + ".job.json"
        Path(jp).write_text(json.dumps(job))
        try:
            subprocess.run([sys.executable, "-m", "agent.worker", jp], cwd=PKG_ROOT, timeout=max(30, timeout),
                           capture_output=True, check=False)
            return json.loads(Path(out + ".json").read_text()), out
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {timeout:.0f}s"}, out
        except Exception as e:
            return {"ok": False, "error": f"worker crashed: {e}"}, out


def apply_change(best_cfg: dict, change: dict) -> dict:
    cfg = copy.deepcopy(best_cfg)
    if change.get("reset_features"):
        cfg["recipes"], cfg["feature_codes"], cfg["drop_columns"] = [], [], []
    for r in change.get("add_recipes") or []:
        if r in features.RECIPES and r not in cfg["recipes"]:
            cfg["recipes"].append(r)
    code = change.get("add_feature_code")
    if code:
        features.validate_code(code)
        cfg["feature_codes"].append(code)
    for c in change.get("drop_columns") or []:
        if c not in cfg["drop_columns"]:
            cfg["drop_columns"].append(c)
    if change.get("model"):
        if change["model"] not in ("lgbm", "catboost", "xgb"):
            raise ValueError(f"bad model {change['model']}")
        if change["model"] != cfg["model"]:
            cfg["params"] = {}
        cfg["model"] = change["model"]
    if isinstance(change.get("params"), dict):
        cfg["params"] = {**cfg["params"], **change["params"]}
    if change.get("target_transform") in ("none", "log1p", "sqrt"):
        cfg["target_transform"] = change["target_transform"]
    return cfg


def builtin_queue(spec) -> list[tuple[str, dict]]:
    q = [
        ("Missing-value flags and category frequencies carry signal", {"add_recipes": ["missing_flags", "freq_encode"]}),
        ("Group statistics of key numerics inside categories expose deviations from peers", {"add_recipes": ["group_stats"]}),
        ("CatBoost's ordered target statistics handle high-cardinality categories better", {"model": "catboost"}),
        ("Ratios/differences of the most important numerics add interactions", {"add_recipes": ["num_pairs"]}),
        ("A smoother LightGBM (smaller leaves, lower lr, more regularisation) generalises better",
         {"model": "lgbm", "params": {"num_leaves": 31, "learning_rate": 0.03, "min_child_samples": 60, "reg_lambda": 5.0}}),
        ("XGBoost gives a diverse and competitive alternative", {"model": "xgb"}),
        ("Log-transforming skewed numerics stabilises splits", {"add_recipes": ["log_skewed"]}),
        ("Row-wise aggregates summarise overall feature profile", {"add_recipes": ["row_stats"]}),
    ]
    if spec["task_type"] == "regression":
        q.insert(2, ("Training on log1p(target) reduces the influence of extreme targets",
                     {"target_transform": "log1p"}))
        q.insert(5, ("sqrt(target) is a milder variance-stabilising transform", {"target_transform": "sqrt"}))
    if spec["split_kind"] == "forward":
        q.insert(1, (f"The period column '{spec['split_col']}' cannot extrapolate to unseen periods; "
                     "dropping it may generalise better", {"drop_columns": [spec["split_col"]]}))
    q.append(("A deeper LightGBM captures higher-order interactions",
              {"model": "lgbm", "params": {"num_leaves": 127, "min_child_samples": 10, "colsample_bytree": 0.5}}))
    return [(h, {"change": c}) for h, c in q]


class Researcher:
    def __init__(self, config, llm: LLM, runner: Runner, spec: dict, base_cols: list[str]):
        self.config, self.llm, self.runner, self.spec = config, llm, runner, spec
        self.base_cols = base_cols
        self.history: list[Experiment] = []
        self.best: Experiment | None = None
        self.notes = ""
        self.queue = builtin_queue(spec)
        self.top_features: list[str] = []
        self.llm_bad_streak = 0

    # ---- helpers
    def analyse_task(self, profile: str):
        """One LLM call: task understanding + domain hypotheses (kept as compact notes)."""
        dic = "\n".join(f"- {k}: {v}" for k, v in list(self.spec["dictionary"].items())[:80])
        user = (f"Task statement:\n{self.spec['description'][:6000]}\n\nColumns:\n{dic}\n\nData profile:\n{profile}\n\n"
                f"Detected: type={self.spec['task_type']} metric={self.spec['metric']} target={self.spec['target']}"
                f" validation={self.spec['split_kind']}({self.spec['split_col']}).\n"
                "Return JSON {\"task_type\":..., \"metric\":..., \"notes\": \"<=150 words: domain insights, pitfalls, "
                "and the 5 most promising feature/model ideas for this data\"}")
        obj = self.llm.ask_json("You are a senior ML engineer analysing a new task.", user, max_tokens=900)
        if obj and isinstance(obj.get("notes"), str):
            self.notes = obj["notes"][:1500]
            log.info("LLM notes: %s", self.notes)

    def _history_text(self) -> str:
        rows = []
        for e in self.history[-14:]:
            sc = f"{e.score:.5g}" if e.score is not None else "-"
            rows.append(f"#{e.id} [{e.status}] score={sc} :: {e.hypothesis[:110]}"
                        + (f" ERR={e.error[:160]}" if e.error else ""))
        return "\n".join(rows) or "(none yet)"

    def propose(self):
        """Next (hypothesis, change, source). LLM first, built-in queue as fallback."""
        if self.llm.available and self.best and self.llm_bad_streak < 2:
            b = self.best
            user = (f"{SCHEMA % ', '.join(features.RECIPES)}\n\nTask: {self.spec['task_type']}, metric={self.spec['metric']} "
                    f"(score shown is larger-is-better; for rmse it is -rmse), validation={self.spec['split_kind']}.\n"
                    f"Notes: {self.notes}\nBase columns: {', '.join(self.base_cols)}\n"
                    f"Current best #{b.id}: score={b.score:.5g}, model={b.cfg['model']}, params={b.cfg['params']}, "
                    f"transform={b.cfg.get('target_transform', 'none')}, recipes={b.cfg['recipes']}, "
                    f"n_feature_codes={len(b.cfg['feature_codes'])}, dropped={b.cfg['drop_columns']}\n"
                    f"Top importances: {self.top_features[:15]}\nHistory:\n{self._history_text()}\n"
                    f"Time left: {self.config.remaining() / 60:.0f} min. Propose the next experiment.")
            obj = self.llm.ask_json(SYSTEM, user, max_tokens=2200)
            if obj and isinstance(obj.get("change"), dict):
                return str(obj.get("hypothesis", "llm hypothesis"))[:300], obj["change"], "llm"
        self.llm_bad_streak = 0   # give the LLM another chance after one built-in hypothesis
        while self.queue:
            h, c = self.queue.pop(0)
            return h, c["change"], "builtin"
        return None

    # ---- main loop
    def evaluate(self, hypothesis, change, source, timeout):
        base = self.best.cfg
        exp = Experiment(len(self.history), hypothesis, base, source=source)
        self.history.append(exp)
        try:
            return self._evaluate(exp, base, change, timeout)
        finally:
            if source == "llm":
                self.llm_bad_streak = self.llm_bad_streak + 1 if exp.status in ("error", "duplicate") else 0

    def _evaluate(self, exp, base, change, timeout):
        try:
            exp.cfg = apply_change(base, change)
        except Exception as e:
            exp.status, exp.error = "error", f"invalid change: {e}"
            return exp
        if exp.cfg == base:
            exp.status, exp.error = "duplicate", "change does not alter the pipeline"
            return exp
        res, out = self.runner.run("eval", exp.cfg, timeout, tag=f"exp{exp.id}")
        if not res.get("ok"):
            exp.status, exp.error = "error", res.get("error", "unknown")
            return exp
        exp.score, exp.seconds, exp.n_iter, exp.out = res["score"], res["seconds"], res["n_iter"], out
        exp.delta = exp.score - self.best.score
        thr = max(1e-4 * abs(self.best.score), 1e-9)
        if exp.delta > thr:
            exp.status = "improved"
            self.best = exp
            self.top_features = [f for f in res.get("importance", {})]
            exp.cfg = {**exp.cfg, "top_features": self.top_features}
        else:
            exp.status = "rejected"
        return exp

    def run_loop(self, until_frac: float):
        fails = 0
        while len(self.history) < self.config.max_experiments:
            avg = np.mean([e.seconds for e in self.history if e.seconds] or [60])
            if self.config.elapsed_frac() >= until_frac or self.config.remaining() < 4 * avg:
                log.info("research stopped: time budget")
                break
            nxt = self.propose()
            if nxt is None:
                log.info("research stopped: no more hypotheses")
                break
            h, change, src = nxt
            t_left = self.config.remaining()
            exp = self.evaluate(h, change, src, timeout=min(max(4 * avg, 600), t_left * 0.25))
            log.info("exp#%d [%s/%s] %s -> %s score=%s delta=%s %s", exp.id, src, exp.status, h[:80],
                     exp.status, exp.score, exp.delta, exp.error or "")
            fails = fails + 1 if exp.status in ("error", "duplicate") else 0
            if fails >= 6:
                log.info("research stopped: repeated failures")
                break
