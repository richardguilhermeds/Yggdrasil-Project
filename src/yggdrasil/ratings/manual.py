"""Ratings MANUAIS: o usuário define a segmentação do score.

* :class:`ManualScoreRating` — cortes de **score** informados à mão (lista de
  limiares); cada faixa (corte_i, corte_{i+1}] vira um rating.
* :class:`PercentileRating` — lista de **percentis** (ex.: ``[20, 40, 60, 80]``);
  os cortes são os quantis do score na amostra de desenvolvimento naqueles
  percentis. Útil para testar uma régua por percentis escolhidos.

Ambas sem fusão monotônica por padrão (a segmentação é a que o usuário pediu).
"""

from __future__ import annotations

import numpy as np

from .base import RatingStrategy, quantile_edges


class ManualScoreRating(RatingStrategy):
    """Rating por **cortes de score** definidos à mão.

    ``cuts``: lista de limiares de score (ex.: ``[0.2, 0.5, 0.8]`` → 4 ratings).
    Faixas no estilo ``R01..Rn`` (ordem crescente de score)."""

    name = "manual_score"

    def __init__(self, cuts=None):
        super().__init__(monotonic_fusion=False, label_style="rank")
        self.cuts = sorted(float(c) for c in (cuts or []))
        self.edges_: np.ndarray = np.array([])

    def _fit_binner(self, scores_dev: np.ndarray, target_dev: np.ndarray) -> None:
        self.edges_ = (np.array([-np.inf, *self.cuts, np.inf]) if self.cuts
                       else np.array([-np.inf, np.inf]))

    def _raw_groups(self, scores: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.edges_, scores, side="right") - 1
        return np.clip(idx, 0, len(self.edges_) - 2)


class PercentileRating(RatingStrategy):
    """Rating por **percentis** do score na amostra de desenvolvimento.

    ``percentiles``: lista em 0–100 (ex.: ``[20, 40, 60, 80]``); os cortes são os
    quantis do score nesses percentis. Faixas ``R01..Rn`` crescentes no score."""

    name = "manual_percentil"

    def __init__(self, percentiles=None):
        super().__init__(monotonic_fusion=False, label_style="rank")
        ps = [float(p) for p in (percentiles if percentiles is not None else [20, 40, 60, 80])]
        self.percentiles = sorted(p for p in ps if 0.0 < p < 100.0)
        self.edges_: np.ndarray = np.array([])

    def _fit_binner(self, scores_dev: np.ndarray, target_dev: np.ndarray) -> None:
        q = np.array([0.0, *[p / 100.0 for p in self.percentiles], 1.0])
        self.edges_ = quantile_edges(scores_dev, np.unique(q))

    def _raw_groups(self, scores: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.edges_, scores, side="right") - 1
        return np.clip(idx, 0, len(self.edges_) - 2)
