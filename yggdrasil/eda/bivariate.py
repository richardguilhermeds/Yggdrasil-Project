"""Análise bivariada feature × alvo.

Cobre o item obrigatório 5 (relação com a variável resposta) e adiciona WoE/IV,
diagnóstico de monotonicidade e drift do poder preditivo ao longo das safras.
Degrada graciosamente quando não há target.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from .binning import MISSING, FeatureBinner, binning_table
from .config import EDAConfig
from .dtypes import as_numeric, has_target, infer_feature_kind
from .profile import _period_series

_EPS = 1e-6

# Faixas de poder preditivo do IV (prática de mercado).
def iv_power(iv: float) -> str:
    if not np.isfinite(iv):
        return "n/a"
    if iv < 0.02:
        return "inutil"
    if iv < 0.1:
        return "fraco"
    if iv < 0.3:
        return "medio"
    if iv < 0.5:
        return "forte"
    return "suspeito_leakage"


def _wilson_ci(p: float, n: int, z: float = 1.96):
    if n == 0 or not np.isfinite(p):
        return (np.nan, np.nan)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _iv_from_binned(bins, y) -> float:
    t = pd.DataFrame({"bin": np.asarray(bins), "y": np.asarray(y, dtype=float)})
    # Contagens vetorizadas (o apply por bin era uma passada Python por grupo).
    maus = (t["y"] == 1).groupby(t["bin"], observed=True).sum().astype(float)
    bons = (t["y"] == 0).groupby(t["bin"], observed=True).sum().astype(float)
    dm = maus / max(maus.sum(), _EPS)
    db = bons / max(bons.sum(), _EPS)
    woe = np.log((db + _EPS) / (dm + _EPS))
    return float(((db - dm) * woe).sum())


def event_rate_by_bin(
    df: pd.DataFrame, col: str, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None,
    problem_type: Optional[str] = None,
) -> pd.DataFrame:
    """Event rate (classificação) ou target médio (regressão) por bin + IC."""
    eda_cfg = eda_cfg or EDAConfig()
    binner = FeatureBinner(eda_cfg.binning_method, eda_cfg.n_bins, eda_cfg.rare_level_pct)
    binner.fit(df, col, cfg, problem_type)
    tab = binning_table(df, col, cfg, eda_cfg, problem_type, binner=binner)
    if not has_target(df, cfg):
        return tab

    if problem_type == "classification" and "event_rate" in tab.columns:
        cis = [_wilson_ci(p, n) for p, n in zip(tab["event_rate"], tab["n"])]
        tab["lower_ci"] = [round(c[0], 6) if np.isfinite(c[0]) else np.nan for c in cis]
        tab["upper_ci"] = [round(c[1], 6) if np.isfinite(c[1]) else np.nan for c in cis]
    elif "target_medio" in tab.columns:
        bins = binner.transform(df[col])
        sem = (pd.DataFrame({"bin": bins.values, "t": df[cfg.target_col].values})
               .groupby("bin", observed=True)["t"].sem())
        tab["lower_ci"] = (tab["target_medio"] - 1.96 * tab["bin"].map(sem)).round(6)
        tab["upper_ci"] = (tab["target_medio"] + 1.96 * tab["bin"].map(sem)).round(6)
    return tab


def feature_target_corr(
    df: pd.DataFrame, col: str, cfg: ColumnConfig, problem_type: Optional[str] = None,
) -> Dict[str, float]:
    """Correlação feature↔alvo apropriada ao tipo (numérica)."""
    if not has_target(df, cfg) or infer_feature_kind(df[col]) == "categorical":
        return {}
    from scipy.stats import pearsonr, pointbiserialr, spearmanr

    x = as_numeric(df[col])
    y = df[cfg.target_col]
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 3 or x.nunique() < 2:
        return {}
    if problem_type == "classification":
        try:
            r, p = pointbiserialr(y.astype(float), x.astype(float))
            return {"point_biserial": round(float(r), 4), "p_value": round(float(p), 4)}
        except Exception:
            return {}
    return {"spearman": round(float(spearmanr(x, y).correlation), 4),
            "pearson": round(float(pearsonr(x, y)[0]), 4)}


def woe_iv_table(
    df: pd.DataFrame, col: str, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None,
) -> pd.DataFrame:
    """Tabela de WoE/IV (só classificação binária). IV total em ``.attrs['iv']``."""
    tab = binning_table(df, col, cfg, eda_cfg, "classification")
    tab.attrs["iv_power"] = iv_power(tab.attrs.get("iv", float("nan")))
    return tab


def monotonicity_diagnostic(table: pd.DataFrame, value_col: Optional[str] = None) -> Dict:
    """Trend (crescente/decrescente/não-monotônica), inversões e Spearman."""
    if value_col is None:
        value_col = "event_rate" if "event_rate" in table.columns else "target_medio"
    if value_col not in table.columns:
        return {"trend": "n/a", "n_inversoes": 0, "spearman": np.nan}
    sub = table[table["bin"] != MISSING]
    vals = sub[value_col].dropna().values
    if len(vals) < 2:
        return {"trend": "n/a", "n_inversoes": 0, "spearman": np.nan}
    diffs = np.diff(vals)
    trend = ("crescente" if (diffs >= 0).all()
             else "decrescente" if (diffs <= 0).all() else "nao_monotonica")
    n_inv = int((np.sign(diffs[:-1]) != np.sign(diffs[1:])).sum()) if len(diffs) > 1 else 0
    from scipy.stats import spearmanr
    rho = spearmanr(range(len(vals)), vals).correlation
    return {"trend": trend, "n_inversoes": n_inv,
            "spearman": round(float(rho), 4) if np.isfinite(rho) else np.nan}


def bivariate_over_time(
    df: pd.DataFrame, col: str, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None,
    problem_type: Optional[str] = None,
) -> pd.DataFrame:
    """Drift do poder preditivo por safra (bins fixos do DES)."""
    eda_cfg = eda_cfg or EDAConfig()
    if not has_target(df, cfg):
        return pd.DataFrame()
    binner = FeatureBinner(eda_cfg.binning_method, eda_cfg.n_bins, eda_cfg.rare_level_pct)
    binner.fit(df, col, cfg, problem_type)
    tmp = pd.DataFrame({
        "periodo": _period_series(df, cfg, eda_cfg.time_freq).values,
        "bin": binner.transform(df[col]).values,
        "t": df[cfg.target_col].values,
    })
    rows = []
    for p, d in tmp.groupby("periodo"):
        d = d.dropna(subset=["t"])
        if problem_type == "classification":
            if d["t"].nunique() >= 2:
                rows.append({"periodo": p, "iv": round(_iv_from_binned(d["bin"], d["t"]), 6),
                             "event_rate": round(float(d["t"].mean()), 6), "n": len(d)})
        else:
            rows.append({"periodo": p, "target_medio": round(float(d["t"].mean()), 6), "n": len(d)})
    return pd.DataFrame(rows).sort_values("periodo").reset_index(drop=True) if rows else pd.DataFrame()


__all__ = [
    "event_rate_by_bin", "feature_target_corr", "woe_iv_table",
    "monotonicity_diagnostic", "bivariate_over_time", "iv_power",
]
