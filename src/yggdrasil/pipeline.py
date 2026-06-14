"""Orquestrador end-to-end da esteira de ML (``MLPipeline``).

Fluxo: valida a entrada → (treina, se um ``trainer`` for dado) → score em
**todas** as amostras → atribui grupos homogêneos (ratings) em todas as amostras
→ calcula métricas/shifts/PSI/relatórios **apenas nas amostras de análise**
(DES/OOT) → loga tudo no MLflow. Amostras *scoring-only* (ex.: SIMUL, BACKTEST)
saem no DataFrame final já scoradas e com rating, sem entrar em nenhuma análise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .config import ColumnConfig
from .data import validate_input
from .metrics import metric_by_sample, sample_shifts
from .monitoring import psi_summary
from .ratings import RatingStrategy, build_ratings
from .reporting import group_reports_all
from .tracking import DEFAULT_EXPERIMENT, log_pipeline_run

RatingSpec = Union[str, RatingStrategy]


@dataclass
class PipelineResult:
    """Resultado da execução da esteira."""

    df_scored: pd.DataFrame
    rating_cols: List[str]
    metrics_by_sample: Dict[str, Dict[str, float]]
    shifts: Dict[str, float]
    psi_metrics: Dict[str, float]
    reports: Dict[str, pd.DataFrame]
    model: Any = None
    run_id: Optional[str] = None
    strategies: List[RatingStrategy] = field(default_factory=list)


class MLPipeline:
    """Esteira de ML governada (métricas, ratings, PSI, SHAP, relatórios, MLflow)."""

    def __init__(
        self,
        cfg: Optional[ColumnConfig] = None,
        problem_type: str = "regression",
        ratings: Optional[Sequence[RatingSpec]] = None,
    ):
        self.cfg = cfg or ColumnConfig()
        if problem_type not in ("classification", "regression"):
            raise ValueError(f"problem_type inválido: {problem_type!r}")
        self.problem_type = problem_type
        self.strategies = self._resolve_ratings(ratings)

    @staticmethod
    def _resolve_ratings(ratings: Optional[Sequence[RatingSpec]]) -> List[RatingStrategy]:
        if ratings is None:
            return build_ratings(None)
        nomes, instancias = [], []
        for r in ratings:
            (instancias if isinstance(r, RatingStrategy) else nomes).append(r)
        if nomes and instancias:
            # mistura de nomes e instâncias: resolve nomes e mantém a ordem original
            resolvidos = []
            for r in ratings:
                resolvidos.append(r if isinstance(r, RatingStrategy) else build_ratings([r])[0])
            return resolvidos
        return instancias if instancias else build_ratings(nomes)

    # ------------------------------------------------------------------
    def _predict(self, model, X: pd.DataFrame) -> np.ndarray:
        if self.problem_type == "classification" and hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return np.asarray(model.predict(X), dtype=float)

    def run(
        self,
        df: pd.DataFrame,
        model: Any = None,
        trainer: Any = None,
        *,
        experiment: str = DEFAULT_EXPERIMENT,
        run_name: Optional[str] = None,
        tags: Optional[dict] = None,
        params: Optional[dict] = None,
        log_mlflow: bool = True,
        log_shap: bool = True,
        registered_model_name: Optional[str] = None,
    ) -> PipelineResult:
        cfg = self.cfg
        validate_input(df, cfg)
        df = df.copy()
        feature_cols = cfg.feature_columns(df)

        # ── treino opcional ─────────────────────────────────────────────
        if trainer is not None:
            dev = df[df[cfg.sample_col] == cfg.dev_sample]
            model = trainer.train(dev, cfg)
        if model is None:
            raise ValueError("Forneça um modelo treinado (model=) ou um trainer=.")

        # ── score em todas as amostras ──────────────────────────────────
        df[cfg.score_col] = self._predict(model, df[feature_cols])

        # ── ratings em todas as amostras ────────────────────────────────
        rating_cols: List[str] = []
        for strat in self.strategies:
            strat.fit(df, cfg, self.problem_type)
            df[strat.column] = strat.transform(df, cfg)
            rating_cols.append(strat.column)

        # ── análise apenas nas amostras de análise (DES/OOT) ────────────
        df_analise = df[df[cfg.sample_col].isin(cfg.analysis_samples)].copy()
        metrics_by_sample = metric_by_sample(df_analise, cfg, self.problem_type)
        shifts = sample_shifts(metrics_by_sample, cfg)
        psi_metrics = psi_summary(df_analise, cfg, rating_cols)
        reports = group_reports_all(df_analise, rating_cols, cfg, self.problem_type)

        # ── logging no MLflow ───────────────────────────────────────────
        run_id = None
        if log_mlflow:
            X_dev = df[df[cfg.sample_col] == cfg.dev_sample][feature_cols]
            run_id = log_pipeline_run(
                model=model,
                df_scored=df_analise,
                cfg=cfg,
                problem_type=self.problem_type,
                rating_cols=rating_cols,
                metrics_by_sample=metrics_by_sample,
                shifts=shifts,
                psi_metrics=psi_metrics,
                reports=reports,
                params=params,
                tags=tags,
                experiment=experiment,
                run_name=run_name,
                X_train=X_dev,
                registered_model_name=registered_model_name,
                log_shap=log_shap,
            )

        return PipelineResult(
            df_scored=df,
            rating_cols=rating_cols,
            metrics_by_sample=metrics_by_sample,
            shifts=shifts,
            psi_metrics=psi_metrics,
            reports=reports,
            model=model,
            run_id=run_id,
            strategies=list(self.strategies),
        )


__all__ = ["MLPipeline", "PipelineResult"]
