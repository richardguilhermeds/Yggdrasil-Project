"""Correlação e redundância entre features (multicolinearidade).

Spearman/Pearson (numéricas), Cramér's V (categóricas), VIF (implementado à mão,
sem statsmodels) e clusters de redundância com sugestão de representante.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from ..config import ColumnConfig
from .config import EDAConfig
from .dtypes import as_numeric, categorical_features, classify_features, numeric_features


def correlation_matrix(
    df: pd.DataFrame, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None, method: str = "spearman",
) -> pd.DataFrame:
    """Matriz de correlação entre features numéricas."""
    eda_cfg = eda_cfg or EDAConfig()
    num = [c for c in numeric_features(classify_features(df, cfg, eda_cfg))
           if df[c].nunique(dropna=True) >= 2]  # exclui constantes (corr indefinida)
    if len(num) < 2:
        return pd.DataFrame()
    X = df[num].apply(as_numeric)
    return X.corr(method=method).round(4)


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    tab = pd.crosstab(x, y)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = tab.values.sum()
    denom = min(tab.shape[0] - 1, tab.shape[1] - 1)
    return float(np.sqrt((chi2 / n) / denom)) if denom > 0 and n > 0 else np.nan


def cramers_v_matrix(df: pd.DataFrame, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None) -> pd.DataFrame:
    """Matriz de associação de Cramér's V entre features categóricas."""
    eda_cfg = eda_cfg or EDAConfig()
    cats = categorical_features(classify_features(df, cfg, eda_cfg))
    if len(cats) < 2:
        return pd.DataFrame()
    M = pd.DataFrame(index=cats, columns=cats, dtype=float)
    for i, a in enumerate(cats):
        for b in cats[i:]:
            v = _cramers_v(df[a], df[b])
            M.loc[a, b] = v
            M.loc[b, a] = v
    return M.round(4)


def vif_table(df: pd.DataFrame, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None) -> pd.DataFrame:
    """VIF por feature numérica (1/(1-R²) regredindo cada uma nas demais)."""
    eda_cfg = eda_cfg or EDAConfig()
    from sklearn.linear_model import LinearRegression
    num = [c for c in numeric_features(classify_features(df, cfg, eda_cfg))
           if df[c].nunique(dropna=True) >= 2]  # VIF não faz sentido p/ constantes
    if len(num) < 2:
        return pd.DataFrame(columns=["feature", "vif"])
    X = df[num].apply(as_numeric)
    X = X.fillna(X.median()).fillna(0.0)
    rows = []
    for c in num:
        others = [o for o in num if o != c]
        r2 = LinearRegression().fit(X[others], X[c]).score(X[others], X[c])
        vif = 1.0 / (1.0 - r2) if r2 < 1 - 1e-12 else np.inf
        rows.append({"feature": c, "vif": round(float(vif), 4) if np.isfinite(vif) else np.inf,
                     "flag": "alto" if vif > eda_cfg.vif_high else "ok"})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def redundancy_clusters(
    df: pd.DataFrame, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None, threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Agrupa features numéricas redundantes (|corr| > threshold) em clusters."""
    eda_cfg = eda_cfg or EDAConfig()
    threshold = threshold if threshold is not None else eda_cfg.corr_high
    corr = correlation_matrix(df, cfg, eda_cfg, "spearman")
    if corr.empty or len(corr) < 2:
        return pd.DataFrame(columns=["feature", "cluster"])
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    d = 1 - corr.abs().values
    d = np.nan_to_num(d, nan=1.0)  # correlação indefinida => máxima distância
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2
    Z = linkage(squareform(d, checks=False), method="average")
    labels = fcluster(Z, t=1 - threshold, criterion="distance")
    return pd.DataFrame({"feature": list(corr.columns), "cluster": labels}).sort_values("cluster").reset_index(drop=True)


__all__ = ["correlation_matrix", "cramers_v_matrix", "vif_table", "redundancy_clusters"]
