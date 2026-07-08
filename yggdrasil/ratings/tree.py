"""Rating por árvore de decisão (score -> target) + fusão monotônica.

Replica o RATING 2 do protótipo: uma ``DecisionTreeRegressor`` particiona o
score em folhas (mesmo para classificação, regredir o alvo 0/1 equivale a
estimar a taxa de evento por folha), as folhas são ordenadas pela média do
target no DES e a fusão por inversão garante a monotonicidade.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base import RatingStrategy

# NOTA DE DESEMPENHO: sklearn.tree é importado **lazy** (dentro de _fit_binner),
# não no topo — este módulo é puxado por `import yggdrasil` (via ratings) e o
# import no topo anulava o padrão lazy documentado em metrics/classification.py.
# (A anotação `DecisionTreeRegressor | None` não é avaliada em runtime graças ao
# `from __future__ import annotations`.)


class TreeRating(RatingStrategy):
    name = "arvore"

    def __init__(
        self,
        max_leaf_nodes: int = 10,
        min_samples_leaf_frac: float = 0.05,
        min_samples_leaf_abs: int = 50,
        alpha: float = 0.05,
        random_state: int = 42,
    ):
        super().__init__(monotonic_fusion=True, alpha=alpha, label_style="letter")
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf_frac = min_samples_leaf_frac
        self.min_samples_leaf_abs = min_samples_leaf_abs
        self.random_state = random_state
        self.tree_: DecisionTreeRegressor | None = None
        self.leaf_to_rank_: Dict[int, int] = {}

    def _fit_binner(self, scores_dev: np.ndarray, target_dev: np.ndarray) -> None:
        from sklearn.tree import DecisionTreeRegressor

        min_leaf = max(
            int(len(scores_dev) * self.min_samples_leaf_frac),
            self.min_samples_leaf_abs,
        )
        min_leaf = max(1, min(min_leaf, len(scores_dev)))
        tree = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=min_leaf,
            random_state=self.random_state,
        )
        X = scores_dev.reshape(-1, 1)
        tree.fit(X, target_dev)
        self.tree_ = tree

        # Ordena folhas pela média do target no DES (crescente) -> rank 0,1,2,...
        leaves_dev = tree.apply(X)
        folha_media = (
            pd.DataFrame({"folha": leaves_dev, "t": target_dev})
            .groupby("folha")["t"]
            .mean()
            .sort_values()
        )
        self.leaf_to_rank_ = {int(folha): rank for rank, folha in enumerate(folha_media.index)}

    def _raw_groups(self, scores: np.ndarray) -> np.ndarray:
        leaves = self.tree_.apply(np.asarray(scores, dtype=float).reshape(-1, 1))
        return np.array([self.leaf_to_rank_.get(int(l), 0) for l in leaves])
