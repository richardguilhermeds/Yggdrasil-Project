"""Rating por decis (10 grupos de igual frequência no score)."""

from __future__ import annotations

import numpy as np

from .base import RatingStrategy, quantile_edges


class DecileRating(RatingStrategy):
    """Decis puros: ``n`` faixas por quantil do score na amostra DES.

    Não aplica fusão monotônica — é a metodologia de referência (decis) exigida
    no requisito. Rótulos no estilo ``R01..R10`` para deixar a ordem explícita.
    """

    name = "decis"

    def __init__(self, n: int = 10):
        super().__init__(monotonic_fusion=False, label_style="rank")
        self.n = n
        self.edges_: np.ndarray = np.array([])

    def _fit_binner(self, scores_dev: np.ndarray, target_dev: np.ndarray) -> None:
        q = np.linspace(0.0, 1.0, self.n + 1)
        self.edges_ = quantile_edges(scores_dev, q)

    def _raw_groups(self, scores: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.edges_, scores, side="right") - 1
        return np.clip(idx, 0, len(self.edges_) - 2)
