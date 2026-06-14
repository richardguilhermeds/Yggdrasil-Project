"""Population Stability Index (PSI) — numérico, categórico e ao longo do tempo.

O PSI mede o deslocamento de uma distribuição em relação a uma referência
(baseline). Aqui é usado para:

* estabilidade do **score** (numérico) entre DES e cada safra;
* estabilidade da distribuição de cada **rating** (categórico) ao longo do tempo;
* PSI agregado DES -> OOT, logado como métrica no MLflow.

Faixas de interpretação (CMN 4.966 / prática de mercado):
``< 0.10`` estável, ``0.10–0.25`` atenção, ``> 0.25`` instável.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import ColumnConfig

PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25


def classify_psi(value: float) -> str:
    """Classifica um valor de PSI em estável / atenção / instável."""
    if not np.isfinite(value):
        return "n/a"
    if value < PSI_STABLE:
        return "estavel"
    if value < PSI_SIGNIFICANT:
        return "atencao"
    return "instavel"


def psi(expected, actual, bins: int = 10, eps: float = 1e-6) -> float:
    """PSI numérico com cortes por quantil da distribuição de referência."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf

    exp_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    act_pct = np.histogram(actual, bins=edges)[0] / len(actual)
    exp_pct = np.where(exp_pct == 0, eps, exp_pct)
    act_pct = np.where(act_pct == 0, eps, act_pct)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_categorical(
    expected_labels,
    actual_labels,
    categories: Optional[Sequence] = None,
    eps: float = 1e-6,
) -> float:
    """PSI sobre a distribuição de categorias (ex.: grupos de rating)."""
    e = pd.Series(list(expected_labels)).value_counts(normalize=True)
    a = pd.Series(list(actual_labels)).value_counts(normalize=True)
    if categories is None:
        categories = sorted(set(e.index) | set(a.index))
    e = e.reindex(categories).fillna(0.0).to_numpy()
    a = a.reindex(categories).fillna(0.0).to_numpy()
    if e.sum() == 0 or a.sum() == 0:
        return float("nan")
    e = np.where(e == 0, eps, e)
    a = np.where(a == 0, eps, a)
    return float(np.sum((a - e) * np.log(a / e)))


def _month_series(df: pd.DataFrame, cfg: ColumnConfig) -> pd.Series:
    """Converte a coluna de data em início do mês (timestamp) para agregação."""
    return pd.to_datetime(df[cfg.date_col]).dt.to_period("M").dt.to_timestamp()


def psi_rating_over_time(
    df: pd.DataFrame,
    rating_col: str,
    cfg: ColumnConfig,
    baseline: Optional[str] = None,
) -> pd.DataFrame:
    """PSI da distribuição de um rating, por mês, contra a baseline (DES).

    Retorna um DataFrame com colunas ``[mes, psi, n, flag]`` ordenado no tempo.
    """
    baseline = baseline or cfg.dev_sample
    base_labels = df.loc[df[cfg.sample_col] == baseline, rating_col].dropna()
    categorias = sorted(df[rating_col].dropna().unique())

    meses = _month_series(df, cfg)
    linhas: List[dict] = []
    for mes, idx in df.groupby(meses).groups.items():
        sub = df.loc[idx, rating_col].dropna()
        valor = psi_categorical(base_labels, sub, categories=categorias)
        linhas.append({"mes": mes, "psi": valor, "n": len(sub), "flag": classify_psi(valor)})
    out = pd.DataFrame(linhas).sort_values("mes").reset_index(drop=True)
    return out


def psi_score_over_time(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    baseline: Optional[str] = None,
    bins: int = 10,
) -> pd.DataFrame:
    """PSI numérico do score, por mês, contra a baseline (DES)."""
    baseline = baseline or cfg.dev_sample
    base_scores = df.loc[df[cfg.sample_col] == baseline, cfg.score_col].dropna()

    meses = _month_series(df, cfg)
    linhas: List[dict] = []
    for mes, idx in df.groupby(meses).groups.items():
        sub = df.loc[idx, cfg.score_col].dropna()
        valor = psi(base_scores, sub, bins=bins)
        linhas.append({"mes": mes, "psi": valor, "n": len(sub), "flag": classify_psi(valor)})
    out = pd.DataFrame(linhas).sort_values("mes").reset_index(drop=True)
    return out


def psi_summary(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    rating_cols: Sequence[str],
    bins: int = 10,
) -> Dict[str, float]:
    """PSI agregado DES -> OOT do score e de cada rating (para o MLflow).

    Chaves: ``psi_score_oot`` e ``psi_rating_<metodo>_oot``.
    """
    dev = df[df[cfg.sample_col] == cfg.dev_sample]
    oot = df[df[cfg.sample_col] == cfg.oot_sample]
    resumo: Dict[str, float] = {}
    if len(dev) and len(oot):
        resumo["psi_score_oot"] = round(psi(dev[cfg.score_col], oot[cfg.score_col], bins=bins), 6)
        for col in rating_cols:
            metodo = col.replace("rating_", "")
            cats = sorted(df[col].dropna().unique())
            resumo[f"psi_rating_{metodo}_oot"] = round(
                psi_categorical(dev[col], oot[col], categories=cats), 6
            )
    return resumo
