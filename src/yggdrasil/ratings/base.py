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
