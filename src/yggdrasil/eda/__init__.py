"""Esteira de análise exploratória (EDA) de features do ``yggdrasil``.

Subpacote **isolado** — não interfere na esteira de avaliação de modelos. Entrega
perfil de missing, percentis, histograma, relação com o alvo, binning (WoE/IV),
importância (univariada + multivariada), estabilidade (PSI) e um relatório
consolidado com veredito por feature.

Uso rápido
----------
>>> from yggdrasil import ColumnConfig
>>> from yggdrasil.eda import run_feature_eda, EDAConfig
>>> report = run_feature_eda(df, ColumnConfig(), EDAConfig(), problem_type="classification")
>>> report.feature_profile      # tabela mestra com veredito por feature
"""

from __future__ import annotations

from .config import DEFAULT_MISSING_CODES, EDAConfig
from .report import FeatureEDAReport, build_feature_profile, run_feature_eda, verdict

__all__ = [
    "run_feature_eda",
    "FeatureEDAReport",
    "build_feature_profile",
    "verdict",
    "EDAConfig",
    "DEFAULT_MISSING_CODES",
]
