"""Estabilidade das features no tempo (PSI/CSI) — reusa ``monitoring.psi``.

Cobre o item obrigatório 8: PSI de cada feature DES→OOT e por safra, mais o CSI
(contribuição de cada faixa ao PSI) para localizar qual faixa migrou.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from ..monitoring.psi import classify_psi, psi, psi_categorical
from .binning import MISSING, FeatureBinner
from .config import EDAConfig
from .dtypes import as_numeric, has_target, infer_feature_kind
from .profile import _period_series

_EPS = 1e-6


def _cat_with_missing(series: pd.Series) -> pd.Series:
    return series.astype(object).where(series.notna(), MISSING).astype(str)


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
    return psi(as_numeric(base).dropna(), as_numeric(oot).dropna(), bins=eda_cfg.n_bins)


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
        bnum = as_numeric(base).dropna()
        for p, idx in df.groupby(per).groups.items():
            a = as_numeric(df.loc[idx, col]).dropna()
            val = psi(bnum, a, bins=eda_cfg.n_bins)
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
