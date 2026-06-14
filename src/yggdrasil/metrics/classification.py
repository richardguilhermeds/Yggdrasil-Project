"""Métricas de discriminação/calibração para modelos de classificação.

Inclui KS, AUC, Gini, Acurácia, F1, precisão, recall, Brier e log loss.
KS e Gini seguem o padrão usado em risco de crédito (CMN 4.966, Art. 18).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Métricas em que "maior é melhor" (usado para interpretar shifts e flags).
HIGHER_IS_BETTER = {
    "auc": True,
    "gini": True,
    "ks": True,
    "accuracy": True,
    "f1": True,
    "precision": True,
    "recall": True,
    "brier": False,
    "logloss": False,
}


def _as_arrays(y_true, y_score):
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    return y_true, y_score


def ks_statistic(y_true, y_score) -> float:
    """KS = máxima distância entre as CDFs dos scores de bons e maus."""
    y_true, y_score = _as_arrays(y_true, y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(ks_2samp(pos, neg).statistic)


def ks_optimal_cutoff(y_true, y_score) -> float:
    """Limiar que maximiza TPR − FPR (ponto de KS na curva ROC)."""
    y_true, y_score = _as_arrays(y_true, y_score)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = np.argmax(tpr - fpr)
    corte = thresholds[j]
    # roc_curve usa +inf no primeiro threshold; protege contra isso.
    if not np.isfinite(corte):
        corte = 1.0
    return float(corte)


def classification_metrics(
    y_true,
    y_score,
    cutoff: Optional[float] = None,
    digits: int = 6,
) -> Dict[str, float]:
    """Calcula o pacote de métricas de classificação.

    ``y_score`` é a probabilidade prevista da classe positiva. Quando ``cutoff``
    é ``None``, usa-se o limiar KS-ótimo para derivar a classe prevista (e o
    próprio corte é devolvido em ``ks_cutoff``).
    """
    y_true, y_score = _as_arrays(y_true, y_score)
    tem_duas_classes = len(np.unique(y_true)) >= 2

    auc = roc_auc_score(y_true, y_score) if tem_duas_classes else float("nan")
    gini = 2 * auc - 1 if tem_duas_classes else float("nan")
    ks = ks_statistic(y_true, y_score)

    corte = ks_optimal_cutoff(y_true, y_score) if cutoff is None else float(cutoff)
    y_pred = (y_score >= corte).astype(int)

    try:
        brier = brier_score_loss(y_true, y_score)
    except ValueError:
        brier = float("nan")
    try:
        ll = log_loss(y_true, np.clip(y_score, 1e-15, 1 - 1e-15), labels=[0, 1])
    except ValueError:
        ll = float("nan")

    metrics = {
        "auc": auc,
        "gini": gini,
        "ks": ks,
        "ks_cutoff": corte,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier": brier,
        "logloss": ll,
    }
    return {k: round(float(v), digits) if np.isfinite(v) else float("nan")
            for k, v in metrics.items()}
