"""Rating por binning ótimo (OptBinning) — faixas ótimas e monotônicas.

Usa ``OptimalBinning`` (classificação) ou ``ContinuousOptimalBinning``
(regressão) para encontrar cortes ótimos do score que maximizam a separação
com restrição de monotonicidade. Como o próprio OptBinning já impõe tendência
monotônica, não aplicamos a fusão por inversão. Import preguiçoso.
"""

from __future__ import annotations

import numpy as np

from .base import RatingStrategy


class OptBinningRating(RatingStrategy):
    name = "optbin"

    def __init__(
        self,
        max_n_bins: int = 10,
        min_prebin_size: float = 0.02,
        monotonic_trend: str = "auto_asc_desc",
    ):
        super().__init__(monotonic_fusion=False, label_style="letter")
        self.max_n_bins = max_n_bins
        self.min_prebin_size = min_prebin_size
        self.monotonic_trend = monotonic_trend
        self.splits_: np.ndarray = np.array([])

    def _fit_binner(self, scores_dev: np.ndarray, target_dev: np.ndarray) -> None:
        if self._problem_type == "classification":
            from optbinning import OptimalBinning

            ob = OptimalBinning(
                name="score",
                dtype="numerical",
                max_n_bins=self.max_n_bins,
                min_prebin_size=self.min_prebin_size,
                monotonic_trend=self.monotonic_trend,
            )
            ob.fit(scores_dev, target_dev.astype(int))
        else:
            from optbinning import ContinuousOptimalBinning

            ob = ContinuousOptimalBinning(
                name="score",
                dtype="numerical",
                max_n_bins=self.max_n_bins,
                min_prebin_size=self.min_prebin_size,
                monotonic_trend=self.monotonic_trend,
            )
            ob.fit(scores_dev, target_dev)

        self.ob_ = ob
        self.splits_ = np.asarray(ob.splits, dtype=float)

    def _raw_groups(self, scores: np.ndarray) -> np.ndarray:
        if self.splits_.size == 0:
            return np.zeros(len(scores), dtype=int)
        return np.searchsorted(self.splits_, np.asarray(scores, dtype=float), side="right")
