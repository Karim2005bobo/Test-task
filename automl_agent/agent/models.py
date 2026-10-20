"""Model zoo: thin uniform wrapper over LightGBM / CatBoost / XGBoost (+ sklearn fallback)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

NCPU = max(1, os.cpu_count() or 1)
LAST_IMPORTANCE: dict = {}
MODELS = ("lgbm", "catboost", "xgb", "hgb")


def default_params(model: str) -> dict:
    return {
        "lgbm": dict(learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8,
                     subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0),
        "catboost": dict(learning_rate=0.08, depth=8, l2_leaf_reg=3.0),
        "xgb": dict(learning_rate=0.05, max_depth=8, min_child_weight=3, subsample=0.8,
                    colsample_bytree=0.7, reg_lambda=1.0),
        "hgb": dict(learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0),
    }[model]


def _cat_names(X: pd.DataFrame):
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def fit_predict(model, params, task, Xtr, ytr, Xva, yva, Xouts, n_classes, seed, n_iter=None,
                max_iter=4000, patience=100):
    """Fit on (Xtr, ytr). If n_iter is None use early stopping on (Xva, yva).

    Returns (list of predictions for each X in Xouts, number of iterations used).
    Regression -> 1-D array, classification -> (n, n_classes) probabilities.
    """
    if model == "catboost":
        max_iter = min(max_iter, 1500)   # keep multiclass CatBoost inside the experiment time limit
    p = {**default_params(model), **(params or {})}
    cls = task == "classification"
    cats = _cat_names(Xtr)
    es = n_iter is None and Xva is not None
    n_est = n_iter or max_iter

    if model == "lgbm":
        import lightgbm as lgb
        cls_ = lgb.LGBMClassifier if cls else lgb.LGBMRegressor
        m = cls_(n_estimators=n_est, n_jobs=NCPU, verbose=-1, random_state=seed, **p)
        kw = {}
        if es:
            kw = dict(eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(patience, verbose=False)])
        m.fit(Xtr, ytr, **kw)
        used = (m.best_iteration_ or n_est) if es else n_est
    elif model == "catboost":
        import catboost as cb
        conv = lambda X: X.assign(**{c: X[c].astype(str) for c in cats})
        Xtr_, Xva_ = conv(Xtr), (conv(Xva) if Xva is not None else None)
        Xouts = [conv(X) for X in Xouts]
        base = dict(iterations=n_est, thread_count=NCPU, verbose=0, random_seed=seed, allow_writing_files=False,
                    cat_features=cats, **p)
        if es:
            base.update(early_stopping_rounds=patience)
        m = cb.CatBoostClassifier(loss_function="MultiClass", **base) if cls \
            else cb.CatBoostRegressor(loss_function="RMSE", **base)
        if es:
            m.fit(Xtr_, ytr, eval_set=(Xva_, yva), use_best_model=True)
            used = m.get_best_iteration() + 1
        else:
            m.fit(Xtr_, ytr)
            used = n_est
    elif model == "xgb":
        import xgboost as xgb
        kw = dict(n_estimators=n_est, tree_method="hist", enable_categorical=True, n_jobs=NCPU,
                  random_state=seed, **p)
        if es:
            kw["early_stopping_rounds"] = patience
        m = xgb.XGBClassifier(**kw) if cls else xgb.XGBRegressor(**kw)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)] if es else None, verbose=False)
        used = (m.best_iteration + 1) if es else n_est
    elif model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        cat_mask = [isinstance(Xtr[c].dtype, pd.CategoricalDtype) for c in Xtr.columns]
        cls_ = HistGradientBoostingClassifier if cls else HistGradientBoostingRegressor
        m = cls_(max_iter=min(n_est, 1500), categorical_features=cat_mask if any(cat_mask) else None,
                 early_stopping=False, random_state=seed, **p)
        m.fit(Xtr, ytr)
        used = m.n_iter_
    else:
        raise ValueError(f"unknown model {model}")

    try:
        imp = np.asarray(m.feature_importances_, dtype=float)
        LAST_IMPORTANCE.clear()
        LAST_IMPORTANCE.update(zip(Xtr.columns, (imp / (imp.sum() or 1)).tolist()))
    except Exception:
        pass
    outs = [m.predict_proba(X) if cls else m.predict(X) for X in Xouts]
    return outs, int(used)
