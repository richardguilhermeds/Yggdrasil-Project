"""Adaptador opcional de treino via PyCaret.

Mantém a paridade com o protótipo (setup / compare_models / finalize_model),
mas é totalmente opcional: o import do PyCaret é preguiçoso e só ocorre quando
o treinador é usado. Instale com ``pip install -e ".[pycaret]"``.
"""

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd

from ..config import ColumnConfig


class PyCaretTrainer:
    """Treina e seleciona o melhor modelo via PyCaret na amostra de desenvolvimento."""

    def __init__(
        self,
        problem_type: str = "regression",
        include: Optional[List[str]] = None,
        sort: Optional[str] = None,
        session_id: int = 42,
        setup_kwargs: Optional[dict] = None,
    ):
        self.problem_type = problem_type
        self.include = include
        self.sort = sort or ("AUC" if problem_type == "classification" else "RMSE")
        self.session_id = session_id
        self.setup_kwargs = setup_kwargs or {}
        self.leaderboard_: Optional[pd.DataFrame] = None

    def train(self, df_dev: pd.DataFrame, cfg: ColumnConfig) -> Any:
        if self.problem_type == "classification":
            from pycaret.classification import (
                compare_models, finalize_model, pull, setup,
            )
        else:
            from pycaret.regression import (
                compare_models, finalize_model, pull, setup,
            )

        cols = cfg.feature_columns(df_dev) + [cfg.target_col]
        data = df_dev[cols].copy()

        setup(
            data=data,
            target=cfg.target_col,
            session_id=self.session_id,
            n_jobs=-1,
            **self.setup_kwargs,
        )
        best = compare_models(include=self.include, sort=self.sort)
        self.leaderboard_ = pull()
        model = finalize_model(best)
        return model


__all__ = ["PyCaretTrainer"]
