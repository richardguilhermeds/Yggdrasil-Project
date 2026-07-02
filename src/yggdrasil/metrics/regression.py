"""Métricas de erro/calibração para modelos de regressão.

Inclui RMSE, MAE, MAPE (robusto), sMAPE, R², viés médio e MedAE.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

# NOTA DE DESEMPENHO: sklearn.metrics é importado **lazy** (dentro da função),
# não no topo — mesmo padrão de metrics/classification.py. Este módulo é puxado
# por `import yggdrasil` (via pipeline); o import no topo anulava o esforço de
# baratear a 1ª célula do notebook.

# Métricas em que "maior é melhor".
HIGHER_IS_BETTER = {
    "rmse": False,
    "mae": False,
    "mape": False,
    "smape": False,
    "medae": False,
    "r2": True,
    "mean_bias": None,  # viés: ideal perto de zero, sinal indica direção
}


def _as_arrays(y_true, y_pred):
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def robust_mape(y_true, y_pred, eps: float = 1e-2) -> float:
    """MAPE ignorando alvos próximos de zero (que inflariam a métrica).

    Segue a abordagem do protótipo: descarta ``|y| < eps`` antes de dividir.
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    """sMAPE simétrico (%), robusto a zeros no denominador."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if mask.sum() == 0:
        return float("nan")
    return float(
        np.mean(np.abs(y_pred[mask] - y_true[mask]) / (denom[mask] / 2)) * 100
    )


def regression_metrics(y_true, y_pred, digits: int = 6) -> Dict[str, float]:
    """Calcula o pacote de métricas de regressão."""
    from sklearn.metrics import (mean_absolute_error, median_absolute_error,
                                 mean_squared_error, r2_score)

    y_true, y_pred = _as_arrays(y_true, y_pred)
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": robust_mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan"),
        "mean_bias": float(np.mean(y_pred - y_true)),  # >0 => previsão acima do real
    }
    return {k: round(float(v), digits) if np.isfinite(v) else float("nan")
            for k, v in metrics.items()}
