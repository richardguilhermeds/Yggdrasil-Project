"""Contrato base das estratégias de rating (grupos homogêneos).

Uma estratégia aprende, na amostra de desenvolvimento, como mapear o score
previsto em grupos homogêneos ordenados (rating). O fluxo padrão (template
method) é:

1. ``_fit_binner`` — aprende o particionamento do score (cortes, árvore, ...);
2. ``_raw_groups`` — atribui a cada linha um grupo bruto inteiro, **crescente
   no score** (0 = menor score);
3. fusão monotônica opcional (via OOT) ou rotulação direta;
4. ``transform`` — reaplica binner + mapeamento em qualquer amostra.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from ..utils import idx_para_letra
from .monotonic import fundir_por_inversao


def _num_to_json(v: float):
    """Float JSON-estrito: ±inf/NaN viram strings (``"inf"``, ``"-inf"``, ``"nan"``).

    ``json.dumps`` padrão emite ``Infinity`` (JSON inválido para parsers
    estritos); codificar as bordas abertas dos ``edges_`` como string mantém o
    ``ratings.json`` legível por qualquer parser."""
    v = float(v)
    if np.isfinite(v):
        return v
    if np.isnan(v):
        return "nan"
    return "inf" if v > 0 else "-inf"


def _num_from_json(v) -> float:
    """Inverso de :func:`_num_to_json` (``float`` aceita ``"inf"``/``"-inf"``/``"nan"``)."""
    return float(v)


def quantile_edges(scores_dev: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Cortes de quantil do score com bordas abertas ``[-inf, ..., inf]``.

    Helper compartilhado por :class:`DecileRating` e :class:`QuantileMonotonicRating`.
    Trata o caso degenerado de **score (quase) constante** no DES: aí ``np.unique``
    dos quantis devolve um único elemento e sobrescrever ``edges[0]``/``edges[-1]``
    no mesmo elemento produziria ``[inf]`` — em ``_raw_groups`` isso vira
    ``np.clip(idx, 0, -1)`` (limite mínimo > máximo), atribuindo o grupo inválido
    ``-1`` a todas as linhas. Nesse caso devolvemos ``[-inf, inf]``, gerando um único
    grupo válido ``0`` (rating único, coerente com score sem poder discriminante).
    """
    edges = np.unique(np.quantile(scores_dev, q)).astype(float)
    if edges.size < 2:
        return np.array([-np.inf, np.inf])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


class RatingStrategy(ABC):
    """Classe base para metodologias de grupos homogêneos."""

    #: identificador curto usado no nome da coluna (ex.: ``rating_decis``)
    name: str = "rating"

    def __init__(
        self,
        monotonic_fusion: bool = True,
        alpha: float = 0.05,
        label_style: str = "letter",
    ) -> None:
        self.monotonic_fusion = monotonic_fusion
        self.alpha = alpha
        self.label_style = label_style  # "letter" (A,B,...) ou "rank" (R01,R02,...)
        self.raw_to_label_: Dict[int, str] = {}
        self.labels_: list = []
        self._problem_type: str = "regression"
        self._fitted = False

    # -- a ser implementado por cada estratégia -----------------------------
    @abstractmethod
    def _fit_binner(self, scores_dev: np.ndarray, target_dev: np.ndarray) -> None:
        """Aprende o particionamento do score na amostra de desenvolvimento."""

    @abstractmethod
    def _raw_groups(self, scores: np.ndarray) -> np.ndarray:
        """Atribui grupos brutos inteiros (crescentes no score) a cada linha."""

    # -- template method ----------------------------------------------------
    @property
    def column(self) -> str:
        return f"rating_{self.name}"

    def _label(self, ordinal: int) -> str:
        if self.label_style == "rank":
            return f"R{ordinal + 1:02d}"
        return idx_para_letra(ordinal)

    def fit(self, df: pd.DataFrame, cfg: ColumnConfig, problem_type: str = "regression"):
        self._problem_type = problem_type
        dev = df[df[cfg.sample_col] == cfg.dev_sample]
        scores_dev = np.asarray(dev[cfg.score_col], dtype=float)
        target_dev = np.asarray(dev[cfg.target_col], dtype=float)
        self._fit_binner(scores_dev, target_dev)

        raw_all = self._raw_groups(np.asarray(df[cfg.score_col], dtype=float))

        if self.monotonic_fusion:
            self.raw_to_label_ = fundir_por_inversao(
                raw_all,
                df[cfg.target_col].values,
                df[cfg.sample_col].values,
                oot_sample=cfg.oot_sample,
                alpha=self.alpha,
                problem_type=problem_type,
            )
        else:
            grupos = sorted(int(g) for g in np.unique(raw_all))
            self.raw_to_label_ = {g: self._label(i) for i, g in enumerate(grupos)}

        self.labels_ = sorted(set(self.raw_to_label_.values()))
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, cfg: ColumnConfig) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Estratégia de rating não foi ajustada (chame fit).")
        raw = self._raw_groups(np.asarray(df[cfg.score_col], dtype=float))
        labels = [self.raw_to_label_.get(int(r)) for r in raw]
        return pd.Series(labels, index=df.index, name=self.column, dtype="object")

    def fit_transform(
        self, df: pd.DataFrame, cfg: ColumnConfig, problem_type: str = "regression"
    ) -> pd.Series:
        return self.fit(df, cfg, problem_type).transform(df, cfg)

    # -- serialização (to_dict / from_dict) ---------------------------------
    def _params_dict(self) -> Dict:
        """Parâmetros de construção da estratégia (kwargs de ``__init__``)."""
        return {}

    def _state_dict(self) -> Dict:
        """Estado ajustado específico da estratégia (JSON-serializável).

        Por padrão serializa ``edges_`` (cortes com bordas ``±inf``), o que
        cobre as estratégias baseadas em ``searchsorted`` sobre cortes.
        Estratégias com estado próprio (árvore, optbinning) sobrescrevem este
        método e :meth:`_load_state`.
        """
        edges = getattr(self, "edges_", None)
        if edges is None:
            return {}
        return {"edges": [_num_to_json(v) for v in np.asarray(edges, dtype=float)]}

    def _load_state(self, state: Dict) -> None:
        """Restaura o estado gravado por :meth:`_state_dict`."""
        if "edges" in state:
            self.edges_ = np.array(
                [_num_from_json(v) for v in state["edges"]], dtype=float
            )

    def to_dict(self) -> Dict:
        """Serializa a estratégia **ajustada** em um dicionário JSON-estrito.

        Contém o nome da classe (para *dispatch* em
        :func:`yggdrasil.ratings.rating_from_dict`), os parâmetros de
        construção, o estado do binner (cortes/limiares), o mapeamento
        ``raw_to_label_`` e os rótulos finais — tudo o que é necessário para
        reaplicar os grupos **sem refit** via :meth:`transform`.
        """
        if not self._fitted:
            raise RuntimeError("Estratégia de rating não foi ajustada (chame fit).")
        return {
            "classe": type(self).__name__,
            "name": self.name,
            "params": self._params_dict(),
            "estado": self._state_dict(),
            "raw_to_label": {str(int(k)): v for k, v in self.raw_to_label_.items()},
            "labels": list(self.labels_),
            "problem_type": self._problem_type,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RatingStrategy":
        """Reconstrói uma estratégia já ajustada a partir de :meth:`to_dict`.

        Não refaz o ``fit``: restaura cortes e mapeamentos e marca a instância
        como ajustada, pronta para :meth:`transform`.
        """
        obj = cls(**(data.get("params") or {}))
        obj._load_state(data.get("estado") or {})
        obj.raw_to_label_ = {
            int(k): str(v) for k, v in (data.get("raw_to_label") or {}).items()
        }
        labels = data.get("labels")
        obj.labels_ = list(labels) if labels else sorted(set(obj.raw_to_label_.values()))
        obj._problem_type = data.get("problem_type", "regression")
        obj._fitted = True
        return obj
