"""Métricas por amostra e cálculo de *shifts* entre DES e OOT.

O *shift* mede a degradação (ou ganho) de uma métrica entre a amostra de
desenvolvimento e a *out-of-time*, atendendo ao requisito de acompanhar
deslocamentos de KS, AUC, RMSE, MAE etc. no experimento. Cada shift ganha
também uma *flag* de degradação (``{m}_shift_flag``) que interpreta a variação
na direção ruim da métrica via :data:`HIGHER_IS_BETTER`.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from ..data import analysis_samples_present
from .classification import HIGHER_IS_BETTER as _HIB_CLASSIFICATION
from .classification import classification_metrics, ks_optimal_cutoff
from .regression import HIGHER_IS_BETTER as _HIB_REGRESSION
from .regression import regression_metrics

# Direção "boa" de cada métrica (classificação + regressão): ``True`` = maior é
# melhor (AUC, KS, R²...), ``False`` = menor é melhor (RMSE, Brier...) e
# ``None`` = viés, avaliado pela magnitude ``|valor|`` (ideal perto de zero).
HIGHER_IS_BETTER: Dict[str, Optional[bool]] = {
    **_HIB_CLASSIFICATION,
    **_HIB_REGRESSION,
}

# Limiares default das flags: degradação relativa na direção ruim da métrica.
FLAG_ATENCAO = 0.10    # > 10% de queda relativa => 'atencao'
FLAG_DEGRADADO = 0.20  # > 20% de queda relativa => 'degradado'


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


def shift_flag(
    metric: str,
    ref: float,
    cmp: float,
    atencao: float = FLAG_ATENCAO,
    degradado: float = FLAG_DEGRADADO,
) -> Optional[str]:
    """Classifica o shift de uma métrica em ``'ok'``/``'atencao'``/``'degradado'``.

    A degradação é a variação relativa na direção *ruim* da métrica, segundo
    :data:`HIGHER_IS_BETTER`:

    * maior-é-melhor (AUC, KS, R²...): queda ``(ref - cmp) / |ref|``;
    * menor-é-melhor (RMSE, Brier...): aumento ``(cmp - ref) / |ref|``;
    * viés (``mean_bias``): crescimento da magnitude ``(|cmp| - |ref|) / |ref|``.

    Devolve ``None`` quando a direção da métrica é desconhecida ou a degradação
    relativa não é computável (referência zero ou valores não finitos).
    """
    if metric not in HIGHER_IS_BETTER:
        return None
    ref, cmp = float(ref), float(cmp)
    if not (np.isfinite(ref) and np.isfinite(cmp)) or ref == 0:
        return None
    sentido = HIGHER_IS_BETTER[metric]
    if sentido is True:
        degradacao = (ref - cmp) / abs(ref)
    elif sentido is False:
        degradacao = (cmp - ref) / abs(ref)
    else:  # viés: importa o afastamento de zero, não o sinal
        degradacao = (abs(cmp) - abs(ref)) / abs(ref)
    if degradacao > degradado:
        return "degradado"
    if degradacao > atencao:
        return "atencao"
    return "ok"


def metric_shifts(
    metrics_ref: Dict[str, float],
    metrics_cmp: Dict[str, float],
    *,
    flag_atencao: float = FLAG_ATENCAO,
    flag_degradado: float = FLAG_DEGRADADO,
) -> Dict[str, Union[float, str]]:
    """Shifts absoluto/relativo e flag de degradação de ``ref`` (DES) para ``cmp`` (OOT).

    ``{m}_shift_abs = cmp - ref`` e ``{m}_shift_rel = (cmp - ref) / |ref|``.
    ``{m}_shift_flag`` classifica a degradação (:func:`shift_flag`) conforme os
    limiares ``flag_atencao``/``flag_degradado``.
    """
    shifts: Dict[str, Union[float, str]] = {}
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
        flag = shift_flag(m, ref, cmp, atencao=flag_atencao, degradado=flag_degradado)
        if flag is not None:
            shifts[f"{m}_shift_flag"] = flag
    return shifts


def sample_shifts(
    metrics_by_sample: Dict[str, Dict[str, float]],
    cfg: ColumnConfig,
    *,
    flag_atencao: float = FLAG_ATENCAO,
    flag_degradado: float = FLAG_DEGRADADO,
) -> Dict[str, Union[float, str]]:
    """Atalho: shifts entre as amostras dev e OOT de um dict por amostra."""
    ref = metrics_by_sample.get(cfg.dev_sample)
    cmp = metrics_by_sample.get(cfg.oot_sample)
    if ref is None or cmp is None:
        return {}
    return metric_shifts(
        ref, cmp, flag_atencao=flag_atencao, flag_degradado=flag_degradado
    )
