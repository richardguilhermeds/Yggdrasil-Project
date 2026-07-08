"""Métricas por amostra e cálculo de *shifts* entre DES e OOT.

O *shift* mede a degradação (ou ganho) de uma métrica entre a amostra de
desenvolvimento e a *out-of-time*, atendendo ao requisito de acompanhar
deslocamentos de KS, AUC, RMSE, MAE etc. no experimento.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from ..data import analysis_samples_present
from .classification import classification_metrics, ks_optimal_cutoff
from .regression import regression_metrics


def compute_metrics(
    y_true,
    y_score,
    problem_type: str,
    cutoff: Optional[float] = None,
) -> Dict[str, float]:
    """Despacha para o pacote de métricas conforme o tipo de problema."""
    if problem_type == "classification":
        return classification_metrics(y_true, y_score, cutoff=cutoff)
    if problem_type == "regression":
        return regression_metrics(y_true, y_score)
    raise ValueError(f"problem_type inválido: {problem_type!r}")


def metric_by_sample(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    problem_type: str,
) -> Dict[str, Dict[str, float]]:
    """Calcula métricas para cada amostra de análise presente.

    Para classificação, o limiar KS-ótimo é estimado **na amostra de
    desenvolvimento** e reaplicado às demais, garantindo comparabilidade
    de Acurácia/F1 entre DES e OOT.
    """
    resultado: Dict[str, Dict[str, float]] = {}
    cutoff = None
    if problem_type == "classification":
        dev = df[df[cfg.sample_col] == cfg.dev_sample]
        if len(dev):
            cutoff = ks_optimal_cutoff(dev[cfg.target_col], dev[cfg.score_col])

    for amostra in analysis_samples_present(df, cfg):
        sub = df[df[cfg.sample_col] == amostra]
        if len(sub) == 0:
            continue
        resultado[amostra] = compute_metrics(
            sub[cfg.target_col], sub[cfg.score_col], problem_type, cutoff=cutoff
        )
    return resultado


def metric_shifts(
    metrics_ref: Dict[str, float],
    metrics_cmp: Dict[str, float],
) -> Dict[str, float]:
    """Shifts absoluto e relativo de ``ref`` (DES) para ``cmp`` (OOT).

    ``{m}_shift_abs = cmp - ref`` e ``{m}_shift_rel = (cmp - ref) / |ref|``.
    """
    shifts: Dict[str, float] = {}
    for m, ref in metrics_ref.items():
        if m == "ks_cutoff":  # corte não é métrica de performance
            continue
        cmp = metrics_cmp.get(m, np.nan)
        if not (np.isfinite(ref) and np.isfinite(cmp)):
            continue
        shifts[f"{m}_shift_abs"] = round(float(cmp - ref), 6)
        shifts[f"{m}_shift_rel"] = (
            round(float((cmp - ref) / abs(ref)), 6) if ref != 0 else float("nan")
        )
    return shifts


def sample_shifts(
    metrics_by_sample: Dict[str, Dict[str, float]],
    cfg: ColumnConfig,
) -> Dict[str, float]:
    """Atalho: shifts entre as amostras dev e OOT de um dict por amostra."""
    ref = metrics_by_sample.get(cfg.dev_sample)
    cmp = metrics_by_sample.get(cfg.oot_sample)
    if ref is None or cmp is None:
        return {}
    return metric_shifts(ref, cmp)
