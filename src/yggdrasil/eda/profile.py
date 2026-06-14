"""Perfil univariado das features: missing, percentis, outliers, cardinalidade.

Cobre os itens obrigatórios 1 (missing global), 2 (missing por safra) e 3
(variação de percentis, global e no tempo), além de qualidade básica e overview.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from ..data import analysis_samples_present, infer_problem_type
from .config import EDAConfig
from .dtypes import (
    as_numeric,
    categorical_features,
    classify_features,
    has_target,
    numeric_features,
)


def _period_series(df: pd.DataFrame, cfg: ColumnConfig, freq: str = "M") -> pd.Series:
    """Coluna de data agregada por período (mês por padrão)."""
    return pd.to_datetime(df[cfg.date_col]).dt.to_period(freq).dt.to_timestamp()


# ── Item 1 — missing global ──────────────────────────────────────────────
def missing_summary(df: pd.DataFrame, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None) -> pd.DataFrame:
    """% e contagem de missing por feature (global e por amostra DES/OOT)."""
    eda_cfg = eda_cfg or EDAConfig()
    kinds = classify_features(df, cfg, eda_cfg)
    n = len(df)
    dev = df[df[cfg.sample_col] == cfg.dev_sample]
    oot = df[df[cfg.sample_col] == cfg.oot_sample]
    rows: List[dict] = []
    for c in cfg.feature_columns(df):
        s = df[c]
        nm = int(s.isna().sum())
        pct = nm / n if n else np.nan
        if kinds[c] in ("numeric", "binary"):
            pz = float((as_numeric(s) == 0).sum()) / n if n else np.nan
        else:
            pz = np.nan
        flag = ("descartar" if pct >= eda_cfg.missing_drop
                else "atencao" if pct >= eda_cfg.missing_warn else "ok")
        rows.append({
            "feature": c, "tipo": kinds[c], "n": n, "n_missing": nm,
            "pct_missing": round(pct, 4),
            "pct_zero": round(pz, 4) if np.isfinite(pz) else np.nan,
            "pct_missing_dev": round(float(dev[c].isna().mean()), 4) if len(dev) else np.nan,
            "pct_missing_oot": round(float(oot[c].isna().mean()), 4) if len(oot) else np.nan,
            "flag": flag,
        })
    return pd.DataFrame(rows)


# ── Item 2 — missing por safra ───────────────────────────────────────────
def missing_over_time(
    df: pd.DataFrame, col: str, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None,
    break_delta: float = 0.20,
) -> pd.DataFrame:
    """% de missing de uma feature por safra, com detecção de quebra/tendência."""
    eda_cfg = eda_cfg or EDAConfig()
    tmp = pd.DataFrame({"periodo": _period_series(df, cfg, eda_cfg.time_freq),
                        "miss": df[col].isna().values})
    tab = (tmp.groupby("periodo")["miss"]
              .agg(pct_missing="mean", n="size").reset_index()
              .sort_values("periodo").reset_index(drop=True))
    delta = float(tab["pct_missing"].diff().abs().max()) if len(tab) > 1 else 0.0
    tab.attrs["delta_max"] = delta
    tab.attrs["quebra"] = delta > break_delta
    if len(tab) > 2:
        slope = float(np.polyfit(range(len(tab)), tab["pct_missing"].values, 1)[0])
    else:
        slope = 0.0
    tab.attrs["tendencia"] = slope
    return tab


# ── Item 3 — percentis (global e no tempo) ───────────────────────────────
def _stats(x: pd.Series, qs) -> pd.Series:
    x = x.dropna()
    d: Dict[str, float] = {f"p{int(q * 100):02d}": x.quantile(q) for q in qs}
    d.update({
        "mean": x.mean(), "std": x.std(), "min": x.min(), "max": x.max(),
        "skew": x.skew(), "kurtosis": x.kurtosis(),
        "iqr": x.quantile(0.75) - x.quantile(0.25),
    })
    return pd.Series(d)


def percentile_table(
    df: pd.DataFrame, col: str, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None,
    by_sample: bool = True,
) -> pd.DataFrame:
    """Tabela de percentis + momentos da feature (GERAL e por amostra)."""
    eda_cfg = eda_cfg or EDAConfig()
    qs = list(eda_cfg.percentiles)
    s = as_numeric(df[col])
    linhas = {"GERAL": _stats(s, qs)}
    if by_sample:
        for samp in analysis_samples_present(df, cfg):
            linhas[samp] = _stats(as_numeric(df[df[cfg.sample_col] == samp][col]), qs)
    return pd.DataFrame(linhas).T.round(6)


def percentiles_over_time(
    df: pd.DataFrame, col: str, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None,
) -> pd.DataFrame:
    """Grade de percentis da feature por safra (para o fan chart e drift)."""
    eda_cfg = eda_cfg or EDAConfig()
    qs = list(eda_cfg.percentiles)
    tmp = pd.DataFrame({"periodo": _period_series(df, cfg, eda_cfg.time_freq),
                        "v": as_numeric(df[col]).values})
    g = tmp.groupby("periodo")["v"]
    out = g.quantile(qs).unstack()
    out.columns = [f"p{int(q * 100):02d}" for q in out.columns]
    out["mean"] = g.mean()
    out["n"] = g.size()
    return out.reset_index().sort_values("periodo").reset_index(drop=True)


# ── qualidade básica ─────────────────────────────────────────────────────
def outlier_summary(series: pd.Series, k: float = 1.5) -> Dict[str, float]:
    """Outliers por IQR (Tukey) e por z-score robusto (MAD) + cauda."""
    x = as_numeric(series).dropna()
    if len(x) < 5:
        return {"pct_outlier_iqr": np.nan, "pct_outlier_mad": np.nan,
                "kurtosis": np.nan, "min": np.nan, "max": np.nan}
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    pct_iqr = float(((x < lo) | (x > hi)).mean())
    med = x.median()
    mad = float((x - med).abs().median())
    pct_mad = float(((x - med).abs() / (1.4826 * mad) > 3.5).mean()) if mad > 0 else 0.0
    return {"pct_outlier_iqr": round(pct_iqr, 4), "pct_outlier_mad": round(pct_mad, 4),
            "kurtosis": round(float(x.kurtosis()), 4), "min": float(x.min()), "max": float(x.max())}


def cardinality_summary(df: pd.DataFrame, cols, eda_cfg: Optional[EDAConfig] = None) -> pd.DataFrame:
    """Cardinalidade, dominância do valor modal e flags de (quase-)constante."""
    eda_cfg = eda_cfg or EDAConfig()
    rows: List[dict] = []
    for c in cols:
        s = df[c]
        nu = int(s.nunique(dropna=False))
        vc = s.value_counts(normalize=True, dropna=False)
        top1 = float(vc.iloc[0]) if len(vc) else np.nan
        rows.append({"feature": c, "nunique": nu, "top1_share": round(top1, 4) if np.isfinite(top1) else np.nan,
                     "constante": nu <= 1,
                     "quase_constante": bool(np.isfinite(top1) and top1 >= eda_cfg.near_constant)})
    return pd.DataFrame(rows)


def dataset_overview(df: pd.DataFrame, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None) -> Dict:
    """Panorama do dataset: tamanho, tipos, período, amostras, prevalência."""
    eda_cfg = eda_cfg or EDAConfig()
    kinds = classify_features(df, cfg, eda_cfg)
    datas = pd.to_datetime(df[cfg.date_col])
    ov: Dict = {
        "n_linhas": len(df),
        "n_features": len(cfg.feature_columns(df)),
        "n_numericas": len(numeric_features(kinds)),
        "n_categoricas": len(categorical_features(kinds)),
        "periodo_min": str(datas.min().date()),
        "periodo_max": str(datas.max().date()),
        "amostras": df[cfg.sample_col].value_counts().to_dict(),
        "tem_target": has_target(df, cfg),
        "n_duplicadas": int(df.duplicated().sum()),
    }
    if has_target(df, cfg):
        pt = infer_problem_type(df, cfg)
        ov["problem_type"] = pt
        chave = "event_rate" if pt == "classification" else "target_medio"
        ov[chave] = round(float(df[cfg.target_col].mean()), 4)
    return ov


def coverage_over_time(df: pd.DataFrame, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None) -> pd.DataFrame:
    """Volumetria por safra (e event rate, se houver target)."""
    eda_cfg = eda_cfg or EDAConfig()
    tmp = df.assign(_p=_period_series(df, cfg, eda_cfg.time_freq))
    out = tmp.groupby("_p").size().rename("n").reset_index().rename(columns={"_p": "periodo"})
    if has_target(df, cfg):
        out["target_medio"] = tmp.groupby("_p")[cfg.target_col].mean().values
    return out.sort_values("periodo").reset_index(drop=True)


__all__ = [
    "missing_summary", "missing_over_time", "percentile_table", "percentiles_over_time",
    "outlier_summary", "cardinality_summary", "dataset_overview", "coverage_over_time",
]
