"""Rating por quantis finos + fusão monotônica por inversão (do protótipo)."""

from __future__ import annotations

import numpy as np

from .base import RatingStrategy, quantile_edges


class QuantileMonotonicRating(RatingStrategy):
    """Quantis de passo ``step`` (ex.: 0.05) fundidos por inversão no OOT.

    Replica o RATING 1 do protótipo: corta o score em ~``1/step`` faixas e funde
    grupos adjacentes cuja inversão no OOT não é estatisticamente significativa.
    """

    name = "quantil"

    def __init__(self, step: float = 0.05, alpha: float = 0.05):
        super().__init__(monotonic_fusion=True, alpha=alpha, label_style="letter")
        self.step = step
        self.edges_: np.ndarray = np.array([])

    def _fit_binner(self, scores_dev: np.ndarray, target_dev: np.ndarray) -> None:
        q = np.arange(0.0, 1.0 + 1e-9, self.step)
        self.edges_ = quantile_edges(scores_dev, q)

    def _raw_groups(self, scores: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.edges_, scores, side="right") - 1
        return np.clip(idx, 0, len(self.edges_) - 2)
