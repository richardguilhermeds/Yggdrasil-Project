"""Estabilidade das features no tempo (PSI/CSI) — reusa ``monitoring.psi``.

Cobre o item obrigatório 8: PSI de cada feature DES→OOT e por safra, mais o CSI
(contribuição de cada faixa ao PSI) para localizar qual faixa migrou.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from ..monitoring.psi import classify_psi, psi_categorical
from .binning import MISSING, FeatureBinner
from .config import EDAConfig
from .dtypes import as_numeric, has_target, infer_feature_kind
from .profile import _period_series

_EPS = 1e-6


def _cat_with_missing(series: pd.Series) -> pd.Series:
    return series.astype(object).where(series.notna(), MISSING).astype(str)


def _numeric_psi_with_missing(base, oot, bins: int, eps: float = _EPS) -> float:
    """PSI numérico que conta os **faltantes como um bin próprio**.

    Cortes por quantil fixados na base não-faltante; os percentuais de cada bin são
    medidos sobre a população TOTAL (incluindo NaN), e os ausentes entram como um
    bin extra. Assim uma feature numérica que passa a faltar ao longo do tempo é
    detectada — coerente com o ramo categórico (``_cat_with_missing``) e com o
    ``csi_by_bin`` (FeatureBinner cria o bin MISSING).

    Para features SEM faltantes nas duas amostras o resultado é idêntico ao de
    ``monitoring.psi`` (o bin de faltantes zera e não contribui)."""
    e = as_numeric(base).to_numpy(dtype=float)
    a = as_numeric(oot).to_numpy(dtype=float)
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    e_obs, a_obs = e[~np.isnan(e)], a[~np.isnan(a)]
    edges = (np.unique(np.quantile(e_obs, np.linspace(0, 1, bins + 1)))
             if len(e_obs) else np.array([]))
    if len(edges) < 2:                                  # base sem cortes válidos
        exp = np.array([len(e_obs) / len(e)])
        act = np.array([len(a_obs) / len(a)])
    else:
        edges = edges.astype(float)
        edges[0], edges[-1] = -np.inf, np.inf
        exp = np.histogram(e_obs, bins=edges)[0] / len(e)
        act = np.histogram(a_obs, bins=edges)[0] / len(a)
    # bin extra de faltantes (fração sobre o total de cada amostra)
    exp = np.append(exp, (len(e) - len(e_obs)) / len(e))
    act = np.append(act, (len(a) - len(a_obs)) / len(a))
    exp = np.where(exp <= 0, eps, exp)
    act = np.where(act <= 0, eps, act)
    return float(np.sum((act - exp) * np.log(act / exp)))


def feature_psi(
    df: pd.DataFrame, col: str, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None,
    baseline: Optional[str] = None,
) -> float:
    """PSI escalar da feature: baseline (DES) vs OOT."""
    eda_cfg = eda_cfg or EDAConfig()
    baseline = baseline or cfg.dev_sample
    base = df[df[cfg.sample_col] == baseline][col]
    oot = df[df[cfg.sample_col] == cfg.oot_sample][col]
    if len(base) == 0 or len(oot) == 0:
        return float("nan")
    if infer_feature_kind(df[col]) == "categorical":
        cats = sorted(set(_cat_with_missing(df[col]).unique()))
        return psi_categorical(_cat_with_missing(base), _cat_with_missing(oot), categories=cats)
    return _numeric_psi_with_missing(base, oot, bins=eda_cfg.n_bins)


def feature_psi_over_time(
    df: pd.DataFrame, col: str, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None,
    baseline: Optional[str] = None,
) -> pd.DataFrame:
    """PSI da feature por safra contra a baseline DES — colunas [periodo, psi, n, flag]."""
    eda_cfg = eda_cfg or EDAConfig()
    baseline = baseline or cfg.dev_sample
    per = _period_series(df, cfg, eda_cfg.time_freq)
    categorica = infer_feature_kind(df[col]) == "categorical"
    base = df[df[cfg.sample_col] == baseline][col]
    rows: List[dict] = []
    if categorica:
        cats = sorted(set(_cat_with_missing(df[col]).unique()))
        b = _cat_with_missing(base)
        for p, idx in df.groupby(per).groups.items():
            a = _cat_with_missing(df.loc[idx, col])
            val = psi_categorical(b, a, categories=cats)
            rows.append({"periodo": p, "psi": round(val, 6), "n": len(idx), "flag": classify_psi(val)})
    else:
        for p, idx in df.groupby(per).groups.items():
            val = _numeric_psi_with_missing(base, df.loc[idx, col], bins=eda_cfg.n_bins)
            rows.append({"periodo": p, "psi": round(val, 6), "n": len(idx), "flag": classify_psi(val)})
    return pd.DataFrame(rows).sort_values("periodo").reset_index(drop=True)


def csi_by_bin(
    df: pd.DataFrame, col: str, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None,
    baseline: Optional[str] = None,
) -> pd.DataFrame:
    """Decomposição do PSI por bin (CSI) com cortes fixados no DES."""
    eda_cfg = eda_cfg or EDAConfig()
    baseline = baseline or cfg.dev_sample
    binner = FeatureBinner("quantile", eda_cfg.n_bins, eda_cfg.rare_level_pct)
    binner.fit(df, col, cfg, None)
    base_bins = binner.transform(df[df[cfg.sample_col] == baseline][col])
    oot_bins = binner.transform(df[df[cfg.sample_col] == cfg.oot_sample][col])
    cats = list(base_bins.cat.categories)
    e = base_bins.value_counts(normalize=True).reindex(cats).fillna(0.0)
    a = oot_bins.value_counts(normalize=True).reindex(cats).fillna(0.0)
    e = e.replace(0.0, _EPS)
    a = a.replace(0.0, _EPS)
    csi = (a - e) * np.log(a / e)
    out = pd.DataFrame({"bin": cats, "exp_pct": e.values.round(4),
                        "act_pct": a.values.round(4), "csi": csi.values.round(6)})
    out.attrs["psi"] = round(float(csi.sum()), 6)
    return out


def stability_summary(df: pd.DataFrame, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None) -> pd.DataFrame:
    """Resumo de estabilidade por feature: PSI DES→OOT e PSI máximo entre safras."""
    eda_cfg = eda_cfg or EDAConfig()
    rows: List[dict] = []
    for c in cfg.feature_columns(df):
        p_oot = feature_psi(df, c, cfg, eda_cfg)
        ts = feature_psi_over_time(df, c, cfg, eda_cfg)
        p_max = float(ts["psi"].max()) if len(ts) else np.nan
        rows.append({
            "feature": c,
            "psi_oot": round(p_oot, 6) if np.isfinite(p_oot) else np.nan,
            "flag": classify_psi(p_oot),
            "psi_max_safra": round(p_max, 6) if np.isfinite(p_max) else np.nan,
        })
    return pd.DataFrame(rows)


__all__ = ["feature_psi", "feature_psi_over_time", "csi_by_bin", "stability_summary"]
