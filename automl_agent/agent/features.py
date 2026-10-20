"""Feature engineering: built-in recipes + sandboxed free-form code proposed by the LLM.

Features are built on train+test concatenated WITHOUT the target, so code (human or LLM
written) cannot leak the label. Only unsupervised/row-wise/group statistics are possible.
"""
from __future__ import annotations

import ast
import builtins

import numpy as np
import pandas as pd

ALLOWED_IMPORTS = {"numpy", "pandas", "math", "re", "itertools", "functools", "collections"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "input",
                   "os", "sys", "subprocess", "socket", "shutil", "pathlib", "builtins", "getattr",
                   "setattr", "delattr"}


class UnsafeCode(Exception):
    pass


def validate_code(code: str):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise UnsafeCode(f"import {a.name} not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise UnsafeCode(f"import from {node.module} not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCode(f"name {node.id} not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCode("dunder attributes not allowed")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "make_features" for n in tree.body):
        raise UnsafeCode("code must define make_features(df)")


def run_code(code: str, df: pd.DataFrame) -> pd.DataFrame:
    validate_code(code)
    safe_builtins = {k: v for k, v in vars(builtins).items() if k not in FORBIDDEN_NAMES}
    ns = {"np": np, "pd": pd, "__builtins__": safe_builtins}
    exec(compile(code, "<llm_features>", "exec"), ns)  # noqa: S102 - AST-validated, runs in worker subprocess
    out = ns["make_features"](df.copy())
    if not isinstance(out, pd.DataFrame) or len(out) != len(df):
        raise ValueError("make_features must return a DataFrame with the same number of rows")
    out.index = df.index
    return out


# ---------------------------------------------------------------- built-in recipes
def _top_numeric(df, num_cols, k):
    cols = [c for c in num_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            and not isinstance(df[c].dtype, pd.CategoricalDtype) and df[c].nunique() > 10]
    return cols[:k]


def recipe_missing_flags(df, ctx):
    out = {}
    for c in ctx["num_cols"] + ctx["cat_cols"]:
        if c in df.columns and df[c].isna().any():
            out[f"{c}__isna"] = df[c].isna().astype(np.int8)
    out["__n_missing"] = df[ctx["num_cols"] + ctx["cat_cols"]].isna().sum(1)
    return out


def recipe_freq_encode(df, ctx):
    return {f"{c}__freq": df[c].map(df[c].value_counts(normalize=True))
            for c in ctx["cat_cols"] if c in df.columns}


def recipe_group_stats(df, ctx):
    """Mean/std of the most important numeric features inside each low-cardinality category."""
    out = {}
    nums = _top_numeric(df, ctx.get("top_features") or ctx["num_cols"], 6)
    for c in ctx["cat_cols"]:
        if c not in df.columns or df[c].nunique() > 300:
            continue
        g = df.groupby(c, observed=True)
        for n in nums:
            m = g[n].transform("mean")
            out[f"{n}__gm_{c}"] = m
            out[f"{n}__gd_{c}"] = df[n] - m
    return out


def recipe_num_pairs(df, ctx):
    nums = _top_numeric(df, ctx.get("top_features") or ctx["num_cols"], 6)
    out = {}
    for i, a in enumerate(nums):
        for b in nums[i + 1:]:
            out[f"{a}__div__{b}"] = df[a] / df[b].replace(0, np.nan)
            out[f"{a}__sub__{b}"] = df[a] - df[b]
    return out


def recipe_log_skewed(df, ctx):
    out = {}
    for c in ctx["num_cols"]:
        s = df[c]
        if c in df.columns and s.notna().any() and s.min() >= 0 and abs(s.skew()) > 2:
            out[f"{c}__log1p"] = np.log1p(s)
    return out


def recipe_row_stats(df, ctx):
    nums = df[ctx["num_cols"]]
    return {"__row_mean": nums.mean(1), "__row_std": nums.std(1)}


RECIPES = {
    "missing_flags": recipe_missing_flags, "freq_encode": recipe_freq_encode,
    "group_stats": recipe_group_stats, "num_pairs": recipe_num_pairs,
    "log_skewed": recipe_log_skewed, "row_stats": recipe_row_stats,
}


def build(df_all: pd.DataFrame, cfg: dict, ctx: dict) -> pd.DataFrame:
    """Apply recipes, LLM code and column drops to the concatenated feature frame."""
    df = df_all.copy()
    for name in cfg.get("recipes", []):
        fn = RECIPES.get(name)
        if fn:
            new = fn(df, ctx)
            df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    for code in cfg.get("feature_codes", []):
        df = run_code(code, df)
    drop = [c for c in cfg.get("drop_columns", []) if c in df.columns]
    df = df.drop(columns=drop)
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:  # anything non-numeric becomes a category
        if not pd.api.types.is_numeric_dtype(df[c]) and not isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype("category")
        elif df[c].dtype == bool:
            df[c] = df[c].astype(np.int8)
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32) if df[c].abs().max(skipna=True) < 1e30 else df[c]
    for c in ctx["cat_cols"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df
