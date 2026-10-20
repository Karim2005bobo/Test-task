"""Runtime configuration.

Everything that depends on the platform contract (where the data lives, where the
submission goes, how the LLM is reached) is collected here. The platform interface
is fixed in the organisers' baseline.py; if it differs from the guesses below, this
is the only module that has to be adapted.
"""
from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR_ENV = ("DATA_DIR", "INPUT_DIR", "DATASET_DIR", "TASK_DIR", "DATA_PATH")
OUTPUT_ENV = ("OUTPUT_PATH", "SUBMISSION_PATH", "SUBMISSION_FILE")
OUTPUT_DIR_ENV = ("OUTPUT_DIR", "RESULT_DIR")


def _first_env(names, default=None):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def find_data_dir(start: str | None = None) -> Path:
    """Locate the directory that holds train.csv (explicit env first, then a shallow search)."""
    explicit = _first_env(DATA_DIR_ENV, start)
    roots = [Path(explicit)] if explicit else []
    roots += [Path.cwd(), Path("/data"), Path("/input"), Path("/workspace"), Path("/app"),
              Path(__file__).resolve().parents[1]]
    for root in roots:
        if not root.exists():
            continue
        for depth_glob in ("train.csv", "*/train.csv", "*/*/train.csv", "*/*/*/train.csv"):
            hit = sorted(root.glob(depth_glob))
            if hit:
                return hit[0].parent
    raise FileNotFoundError("train.csv not found; set DATA_DIR")


def locate_task(arg: str | None):
    """Platform contract (see baseline.py): `solution.py task.md`; paths inside task.md are relative to
    the directory of task.md, which is also the cwd and the place for submission.csv.
    Returns (md_dir, statement_text, data_dir)."""
    md = Path(arg) if arg else Path("task.md")
    if not md.is_absolute():
        md = Path.cwd() / md
    if md.is_dir():
        found = sorted(md.glob("*.md"))
        md = found[0] if found else md / "task.md"
    md_dir = md.parent
    text = md.read_text(encoding="utf-8") if md.is_file() else ""
    data_dir = None
    for m in re.finditer(r"([\w./-]*train\.csv)", text):
        cand = (md_dir / m.group(1)).parent
        if (cand / "train.csv").exists():
            data_dir = cand
            break
    if data_dir is None:
        for g in ("train.csv", "*/train.csv", "*/*/train.csv"):
            hit = sorted(md_dir.glob(g))
            if hit:
                data_dir = hit[0].parent
                break
    if data_dir is None:
        data_dir = find_data_dir()
    return md_dir, text, data_dir


@dataclass
class Config:
    data_dir: Path
    output_path: Path
    workdir: Path
    time_budget: float = 3 * 3600.0      # platform limit per task
    safety_frac: float = 0.88            # never plan past this share of the limit
    research_frac: float = 0.55          # share of budget for the AutoResearch loop
    max_experiments: int = 40
    seed: int = 42
    n_jobs: int = max(1, os.cpu_count() or 1)
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "Qwen3.8-27B"
    llm_token_budget: int = 150_000
    llm_max_calls: int = 30
    llm_timeout: float = 120.0
    statement: str = ""
    start_time: float = field(default_factory=time.time)

    @property
    def deadline(self) -> float:
        return self.start_time + self.time_budget * self.safety_frac

    def remaining(self) -> float:
        return self.deadline - time.time()

    def elapsed_frac(self) -> float:
        return (time.time() - self.start_time) / self.time_budget

    @classmethod
    def from_env(cls, task_arg: str | None = None) -> "Config":
        md_dir, statement, data_dir = locate_task(task_arg)
        out = _first_env(OUTPUT_ENV)
        if out:
            output = Path(out)
        else:
            out_dir = _first_env(OUTPUT_DIR_ENV)
            output = (Path(out_dir) if out_dir else md_dir) / "submission.csv"
        workdir = Path(os.environ.get("AGENT_WORKDIR", "/tmp/automl_agent_work"))
        shutil.rmtree(workdir, ignore_errors=True)   # never reuse artefacts of a previous task/run
        workdir.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        return cls(
            data_dir=data_dir,
            statement=statement,
            output_path=output,
            workdir=workdir,
            time_budget=float(os.environ.get("TIME_BUDGET_SEC", 3 * 3600)),
            max_experiments=int(os.environ.get("MAX_EXPERIMENTS", 40)),
            seed=int(os.environ.get("SEED", 42)),
            llm_base_url=_first_env(("LLM_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE")),
            llm_api_key=_first_env(("LLM_API_KEY", "OPENAI_API_KEY", "API_KEY")),
            llm_model=_first_env(("LLM_MODEL", "OPENAI_MODEL", "MODEL_NAME"), "Qwen3.8-27B"),
            llm_token_budget=int(os.environ.get("LLM_TOKEN_BUDGET", 150_000)),
            llm_max_calls=int(os.environ.get("LLM_MAX_CALLS", 30)),
        )
