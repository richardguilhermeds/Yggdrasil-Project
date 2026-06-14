"""Importância de features — univariada (model-free) e multivariada (surrogate).

Cobre o item obrigatório 7. Univariada: IV, KS, Gini/AUC, mutual information.
Multivariada: surrogate sklearn (RandomForest) + permutation importance, com SHAP
opcional. Consolida tudo num ranking e sinaliza suspeitas de leakage.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..config import ColumnConfig
from ..metrics import ks_statistic
from .binning import binning_table
from .config import EDAConfig
from .dtypes import as_numeric, classify_features, has_target, infer_feature_kind


def _surrogate(problem_type: str):
    if problem_type == "classification":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)


def _build_matrix(df: pd.DataFrame, cfg: ColumnConfig, kinds: Dict[str, str]) -> pd.DataFrame:
    """Matriz numérica (categóricas ordinal-encoded, missing imputado) p/ MI/RF."""
    X = pd.DataFrame(index=df.index)
    for c in cfg.feature_columns(df):
        if kinds[c] == "categorical":
            X[c] = df[c].astype("category").cat.codes.replace(-1, np.nan)
        else:
            X[c] = as_numeric(df[c])
    medians = X.median(numeric_only=True)
    return X.fillna(medians).fillna(0.0)


def iv_score(df, col, cfg, eda_cfg=None, problem_type=None) -> float:
    """Information Value (só classificação binária)."""
    if not has_target(df, cfg) or problem_type != "classification":
        return float("nan")
    tab = binning_table(df, col, cfg, eda_cfg, "classification")
    return float(tab.attrs.get("iv", float("nan")))


def univariate_ks_gini(df, col, cfg, problem_type=None) -> Dict[str, float]:
    """KS, AUC e Gini univariados (feature como score) — classificação numérica."""
    nan = {"ks_univ": np.nan, "auc_univ": np.nan, "gini_univ": np.nan}
    if (not has_target(df, cfg) or problem_type != "classification"
            or infer_feature_kind(df[col]) == "categorical"):
        return nan
    x = as_numeric(df[col])
    y = df[cfg.target_col]
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if x.nunique() < 2 or y.nunique() < 2:
        return nan
    auc = roc_auc_score(y, x)
    auc = max(auc, 1 - auc)  # direção-agnóstico
    return {"ks_univ": round(ks_statistic(y, x), 4),
            "auc_univ": round(float(auc), 4), "gini_univ": round(2 * auc - 1, 4)}


def mutual_information(df, cfg, eda_cfg=None, problem_type=None) -> pd.Series:
    """Mutual information de cada feature com o alvo (em lote)."""
    eda_cfg = eda_cfg or EDAConfig()
    if not has_target(df, cfg):
        return pd.Series(dtype=float)
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    kinds = classify_features(df, cfg, eda_cfg)
    dev = df[df[cfg.sample_col] == cfg.dev_sample]
    if len(dev) == 0:
        dev = df
    dev = dev[dev[cfg.target_col].notna()]
    if len(dev) > eda_cfg.sample_size:
        dev = dev.sample(eda_cfg.sample_size, random_state=42)
    X = _build_matrix(dev, cfg, kinds)
    y = dev[cfg.target_col]
    disc = [kinds[c] in ("categorical", "binary") for c in X.columns]
    fn = mutual_info_classif if problem_type == "classification" else mutual_info_regression
    mi = fn(X, y, discrete_features=disc, random_state=42)
    return pd.Series(np.round(mi, 6), index=list(X.columns), name="mutual_info")


def model_importance(df, cfg, eda_cfg=None, problem_type=None) -> pd.DataFrame:
    """Importância multivariada: surrogate RF (feature_importances_) + permutation."""
    eda_cfg = eda_cfg or EDAConfig()
    if not has_target(df, cfg):
        return pd.DataFrame(columns=["feature", "rf_importance", "permutation"])
    from sklearn.inspection import permutation_importance
    kinds = classify_features(df, cfg, eda_cfg)
    dev = df[df[cfg.sample_col] == cfg.dev_sample]
    if len(dev) == 0:
        dev = df
    dev = dev[dev[cfg.target_col].notna()]
    if len(dev) > eda_cfg.sample_size:
        dev = dev.sample(eda_cfg.sample_size, random_state=42)
    X = _build_matrix(dev, cfg, kinds)
    y = dev[cfg.target_col]
    model = _surrogate(problem_type).fit(X, y)
    # n_jobs=1: evita o problema de limpeza de memmap do joblib no Windows.
    perm = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=1)
    return pd.DataFrame({
        "feature": list(X.columns),
        "rf_importance": np.round(model.feature_importances_, 6),
        "permutation": np.round(perm.importances_mean, 6),
    })


def importance_ranking(df, cfg, eda_cfg=None, problem_type=None) -> pd.DataFrame:
    """Consolida métricas de importância num ranking (com flag de leakage)."""
    eda_cfg = eda_cfg or EDAConfig()
    feats = cfg.feature_columns(df)
    uni: List[dict] = []
    for c in feats:
        row = {"feature": c, "iv": iv_score(df, c, cfg, eda_cfg, problem_type)}
        row.update(univariate_ks_gini(df, c, cfg, problem_type))
        uni.append(row)
    out = pd.DataFrame(uni)

    if has_target(df, cfg):
        mi = mutual_information(df, cfg, eda_cfg, problem_type)
        out = out.merge(mi.rename("mutual_info").reset_index().rename(columns={"index": "feature"}),
                        on="feature", how="left")
        out = out.merge(model_importance(df, cfg, eda_cfg, problem_type), on="feature", how="left")

    # score composto = média dos ranks (maior = mais importante) das métricas disponíveis
    metric_cols = [c for c in ["iv", "ks_univ", "gini_univ", "mutual_info", "rf_importance", "permutation"]
                   if c in out.columns and out[c].notna().any()]
    if metric_cols:
        ranks = out[metric_cols].rank(ascending=True, na_option="keep")
        out["score"] = ranks.mean(axis=1)
        out = out.sort_values("score", ascending=False).reset_index(drop=True)

    out["leakage_flag"] = (
        (out.get("gini_univ", pd.Series(np.nan, index=out.index)).fillna(0).abs() > (2 * eda_cfg.leakage_auc - 1))
        | (out.get("iv", pd.Series(np.nan, index=out.index)).fillna(0) > eda_cfg.iv_leakage)
    )
    return out


def leakage_suspects(ranking: pd.DataFrame) -> List[str]:
    """Lista de features marcadas como suspeitas de vazamento do alvo."""
    if "leakage_flag" not in ranking.columns:
        return []
    return ranking.loc[ranking["leakage_flag"], "feature"].tolist()


def shap_surrogate_importance(df, cfg, eda_cfg=None, problem_type=None) -> pd.DataFrame:
    """(Opcional) Importância SHAP sobre o surrogate. Best-effort."""
    eda_cfg = eda_cfg or EDAConfig()
    if not has_target(df, cfg):
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    from ..interpretability import compute_shap, shap_feature_importance
    kinds = classify_features(df, cfg, eda_cfg)
    dev = df[df[cfg.sample_col] == cfg.dev_sample]
    dev = (dev if len(dev) else df)
    dev = dev[dev[cfg.target_col].notna()]
    X = _build_matrix(dev, cfg, kinds)
    y = dev[cfg.target_col]
    try:
        model = _surrogate(problem_type).fit(X, y)
        sv, Xs = compute_shap(model, X, problem_type, sample_size=eda_cfg.sample_size)
        return shap_feature_importance(sv, Xs.columns)
    except Exception:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])


__all__ = [
    "iv_score", "univariate_ks_gini", "mutual_information", "model_importance",
    "importance_ranking", "leakage_suspects", "shap_surrogate_importance",
]
