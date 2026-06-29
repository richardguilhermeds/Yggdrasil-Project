"""Métricas de discriminação/calibração para modelos de classificação.

Inclui KS, AUC, Gini, Acurácia, F1, precisão, recall, Brier e log loss.
KS e Gini seguem o padrão usado em risco de crédito (CMN 4.966, Art. 18).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# NOTA DE DESEMPENHO: scipy.stats e sklearn.metrics são importados **lazy** (dentro
# das funções), não no topo. Esses imports são caros e este módulo é puxado por
# `model/segmenter.py` (e, via pacote, por `import yggdrasil`); mantê-los aqui no
# topo fazia a 1ª célula do notebook Databricks pagar o stack inteiro de métricas
# antes mesmo de existir um modelo. O lookup em sys.modules após o 1º uso é barato.

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
    from scipy.stats import ks_2samp
    y_true, y_score = _as_arrays(y_true, y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(ks_2samp(pos, neg).statistic)


def ks_optimal_cutoff(y_true, y_score) -> float:
    """Limiar que maximiza TPR − FPR (ponto de KS na curva ROC)."""
    from sklearn.metrics import roc_curve
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


def _roc_pack(y_true, y_score):
    """Calcula ``roc_curve`` UMA vez e deriva (auc, gini, ks, corte_ks) do mesmo
    resultado — em vez de ordenar o score 3× (roc_auc_score + ks_2samp + roc_curve).

    Como este pacote roda **por amostra a cada fit/refresh** das UIs, eliminar
    duas ordenações O(n log n) por chamada corta latência diretamente. A
    equivalência KS = max(TPR−FPR) ↔ estatística de Kolmogorov-Smirnov entre as
    CDFs de bons/maus é exata para a CDF empírica (padrão em risco de crédito).
    Devolve ``None`` quando há < 2 classes (métricas ficam NaN)."""
    from sklearn.metrics import auc as _auc, roc_curve
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, thr = roc_curve(y_true, y_score)
    auc = float(_auc(fpr, tpr))
    j = int(np.argmax(tpr - fpr))
    ks = float(tpr[j] - fpr[j])
    corte = float(thr[j]) if np.isfinite(thr[j]) else 1.0
    return auc, 2 * auc - 1, ks, corte


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
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
    )
    y_true, y_score = _as_arrays(y_true, y_score)

    # roc_curve uma única vez → AUC, Gini, KS e corte KS-ótimo do mesmo cálculo.
    pack = _roc_pack(y_true, y_score)
    if pack is not None:
        auc, gini, ks, corte_ks = pack
    else:
        auc = gini = ks = float("nan")
        corte_ks = 0.5

    corte = corte_ks if cutoff is None else float(cutoff)
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
