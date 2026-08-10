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
        # Particionamento 1-D equivalente à árvore: limiares ordenados e o rank
        # de cada intervalo entre eles (base do transform e da serialização).
        self.thresholds_: np.ndarray = np.array([])
        self.interval_rank_: np.ndarray = np.array([0], dtype=int)

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

        # Converte a árvore 1-D em intervalos: percurso em-ordem da estrutura —
        # cada nó interno (limiar t; x <= t vai à esquerda) separa faixas
        # (t_k, t_{k+1}], então os limiares em-ordem saem crescentes e as folhas
        # em-ordem são os intervalos entre eles. Além de vetorizar _raw_groups
        # (searchsorted em vez de tree.apply + loop), permite serializar a
        # estratégia sem persistir o objeto sklearn.
        tt = tree.tree_
        limiares: list = []
        ranks: list = []

        def _walk(no: int) -> None:
            if tt.children_left[no] < 0:               # folha
                ranks.append(self.leaf_to_rank_.get(int(no), 0))
                return
            _walk(int(tt.children_left[no]))
            limiares.append(float(tt.threshold[no]))
            _walk(int(tt.children_right[no]))

        _walk(0)
        self.thresholds_ = np.asarray(limiares, dtype=float)
        self.interval_rank_ = np.asarray(ranks, dtype=int)

    def _raw_groups(self, scores: np.ndarray) -> np.ndarray:
        # Reproduz tree.apply: o sklearn compara em float32 (cast interno do X),
        # então aplicamos o mesmo cast antes do searchsorted. Convenção da
        # árvore (x <= limiar vai à esquerda) => side="left": empate no limiar
        # cai no intervalo anterior.
        s32 = np.asarray(scores, dtype=np.float32)
        idx = np.searchsorted(self.thresholds_, s32, side="left")
        return self.interval_rank_[idx]

    def _params_dict(self) -> dict:
        return {
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf_frac": self.min_samples_leaf_frac,
            "min_samples_leaf_abs": self.min_samples_leaf_abs,
            "alpha": self.alpha,
            "random_state": self.random_state,
        }

    def _state_dict(self) -> dict:
        return {
            "thresholds": [float(t) for t in self.thresholds_],
            "interval_rank": [int(r) for r in self.interval_rank_],
        }

    def _load_state(self, state: dict) -> None:
        # Restaura só o particionamento (limiares + rank por intervalo); o
        # objeto sklearn (tree_) não é necessário para o transform.
        self.thresholds_ = np.asarray(state.get("thresholds", []), dtype=float)
        self.interval_rank_ = np.asarray(state.get("interval_rank", [0]), dtype=int)
