"""Lift, curva CAP e Accuracy Ratio para ordenação de risco (classificação).

Ferramentas de *ranking* sobre eventos binários: tabela de lift por faixa de
score (default: decis, do pior para o melhor score), curva CAP (Cumulative
Accuracy Profile / Lorenz) e Accuracy Ratio (AR) — matematicamente equivalente
ao Gini (``2·AUC − 1``). As convenções de ordenação e fatiamento seguem os
``plot_cap``/``plot_lift`` dos segmentadores (faixa 1 = piores scores).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .classification import _as_arrays

# NOTA DE DESEMPENHO: mesmo padrão de `classification.py` — nada de sklearn no
# topo; tudo aqui usa apenas numpy/pandas (uma única ordenação O(n log n)).

LIFT_TABLE_COLS = ["faixa", "n", "n_eventos", "taxa_evento", "lift",
                   "pop_acum", "captura_acum"]


def _sorted_events(y_true, y_score) -> np.ndarray:
    """Alvo ordenado do PIOR para o MELHOR score (maior prob. de evento
    primeiro); ``mergesort`` estável p/ resultado determinístico em empates."""
    y_true, y_score = _as_arrays(y_true, y_score)
    order = np.argsort(-y_score, kind="mergesort")
    return y_true[order]


def lift_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Tabela de lift por faixa de score (default: decis).

    A carteira é ordenada do **pior** para o **melhor** score e fatiada em
    ``n_bins`` grupos de tamanho ~igual (robusto a empates de score — mesma
    convenção do ``plot_lift`` dos segmentadores). Colunas:

    * ``faixa`` — 1 = piores scores (maior probabilidade de evento);
    * ``n`` / ``n_eventos`` — volume e nº de eventos da faixa;
    * ``taxa_evento`` — taxa de evento da faixa;
    * ``lift`` — taxa da faixa / taxa geral (1 = modelo aleatório);
    * ``pop_acum`` — fração acumulada da carteira até a faixa (0–1);
    * ``captura_acum`` — fração acumulada de eventos capturados até a faixa (0–1).
    """
    y_ord = _sorted_events(y_true, y_score)
    n = y_ord.size
    if n == 0:
        return pd.DataFrame(columns=LIFT_TABLE_COLS)
    n_bins = max(1, min(int(n_bins), n))
    # fatia em n_bins grupos de tamanho ~igual, na ordem pior → melhor
    idx = np.minimum((np.arange(n) * n_bins) // n, n_bins - 1)
    n_faixa = np.bincount(idx, minlength=n_bins).astype(float)
    ev_faixa = np.bincount(idx, weights=y_ord, minlength=n_bins)
    taxa = np.divide(ev_faixa, n_faixa,
                     out=np.full(n_bins, np.nan), where=n_faixa > 0)
    taxa_geral = float(y_ord.mean())
    lift = taxa / taxa_geral if taxa_geral > 0 else np.full(n_bins, np.nan)
    tot_ev = float(ev_faixa.sum())
    captura = (np.cumsum(ev_faixa) / tot_ev if tot_ev > 0
               else np.full(n_bins, np.nan))
    return pd.DataFrame({
        "faixa": np.arange(1, n_bins + 1),
        "n": n_faixa.astype(int),
        "n_eventos": ev_faixa.astype(int),
        "taxa_evento": taxa,
        "lift": lift,
        "pop_acum": np.cumsum(n_faixa) / n,
        "captura_acum": captura,
    })


def cap_curve(y_true, y_score) -> Tuple[np.ndarray, np.ndarray]:
    """Curva CAP (Cumulative Accuracy Profile / Lorenz).

    Devolve ``(x, y)``: fração acumulada da carteira ordenada do pior para o
    melhor score (``x``) e fração acumulada de eventos capturados (``y``),
    ambas iniciando em 0 — pronto para ``ax.plot(x, y)``.
    """
    y_ord = _sorted_events(y_true, y_score)
    n = y_ord.size
    if n == 0:
        return np.array([0.0]), np.array([0.0])
    cum_port = np.arange(1, n + 1) / n
    cum_ev = np.cumsum(y_ord) / max(float(y_ord.sum()), 1.0)
    return (np.concatenate(([0.0], cum_port)),
            np.concatenate(([0.0], cum_ev)))


def accuracy_ratio(y_true, y_score) -> float:
    """Accuracy Ratio (AR) da curva CAP — equivalente ao Gini (``2·AUC − 1``).

    ``AR = (área do modelo − 0,5) / (área do modelo perfeito − 0,5)``, com a
    área sob a CAP pela regra do trapézio (manual: independe da versão do
    numpy — ``trapz`` foi deprecado na 2.0 e ``trapezoid`` não existe <2.0).
    Sem empates de score a igualdade com o Gini da ROC é exata; com empates a
    diferença é da ordem da fração empatada. NaN quando falta uma das classes.
    """
    y_true, y_score = _as_arrays(y_true, y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    x, y = cap_curve(y_true, y_score)
    area_mod = float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0))
    tx_ev = float(y_true.mean())
    area_perf = 1.0 - tx_ev / 2.0                       # modelo perfeito
    if area_perf <= 0.5:                                # tudo evento: AR indefinido
        return float("nan")
    return (area_mod - 0.5) / (area_perf - 0.5)


__all__ = ["lift_table", "cap_curve", "accuracy_ratio", "LIFT_TABLE_COLS"]
