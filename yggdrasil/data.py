"""Validação e preparação da tabela de entrada da esteira.

Funções puramente em pandas que checam o contrato de dados descrito em
:class:`yggdrasil.config.ColumnConfig` e separam features/target/amostras.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ColumnConfig


def validate_input(df: pd.DataFrame, cfg: ColumnConfig) -> None:
    """Valida o contrato mínimo da tabela de entrada.

    Levanta ``ValueError`` se faltar alguma coluna obrigatória, se não houver
    features, ou se as amostras de desenvolvimento/OOT estiverem ausentes.
    """
    obrigatorias = [cfg.date_col, cfg.sample_col, cfg.target_col]
    faltando = [c for c in obrigatorias if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {faltando}")

    # Garante ao menos uma feature.
    cfg.feature_columns(df)

    amostras_presentes = set(df[cfg.sample_col].dropna().unique())
    if cfg.dev_sample not in amostras_presentes:
        raise ValueError(
            f"Amostra de desenvolvimento '{cfg.dev_sample}' não encontrada na "
            f"coluna '{cfg.sample_col}'. Amostras presentes: {sorted(amostras_presentes)}"
        )
    if cfg.oot_sample not in amostras_presentes:
        raise ValueError(
            f"Amostra OOT '{cfg.oot_sample}' não encontrada na coluna "
            f"'{cfg.sample_col}'. Amostras presentes: {sorted(amostras_presentes)}"
        )


def sample_mask(df: pd.DataFrame, cfg: ColumnConfig, sample: str) -> pd.Series:
    """Máscara booleana das linhas de uma amostra."""
    return df[cfg.sample_col] == sample


def split_samples(df: pd.DataFrame, cfg: ColumnConfig) -> Dict[str, pd.DataFrame]:
    """Quebra o DataFrame em um dicionário ``{amostra: sub-DataFrame}``."""
    return {
        amostra: sub
        for amostra, sub in df.groupby(cfg.sample_col, observed=True)
    }


def analysis_samples_present(df: pd.DataFrame, cfg: ColumnConfig) -> List[str]:
    """Amostras de análise efetivamente presentes na tabela (dev primeiro)."""
    presentes = set(df[cfg.sample_col].dropna().unique())
    ordem: List[str] = []
    for s in (cfg.dev_sample, cfg.oot_sample):
        if s in presentes:
            ordem.append(s)
    for s in cfg.analysis_samples:
        if s in presentes and s not in ordem:
            ordem.append(s)
    return ordem


def scoring_only_samples_present(df: pd.DataFrame, cfg: ColumnConfig) -> List[str]:
    """Amostras *scoring-only* presentes (ex.: SIMUL, BACKTEST)."""
    presentes = set(df[cfg.sample_col].dropna().unique())
    return sorted(s for s in presentes if not cfg.is_analysis_sample(s))


def get_X_y(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    sample: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Retorna ``(X, y)`` para a tabela (ou só para uma amostra, se informada)."""
    sub = df if sample is None else df[sample_mask(df, cfg, sample)]
    X = sub[cfg.feature_columns(df)].copy()
    y = sub[cfg.target_col].copy()
    return X, y


def infer_problem_type(df: pd.DataFrame, cfg: ColumnConfig) -> str:
    """Heurística simples: target binário => classification, senão regression."""
    valores = pd.unique(df[cfg.target_col].dropna())
    if len(valores) <= 2 and set(np.asarray(valores).astype(float)) <= {0.0, 1.0}:
        return "classification"
    return "regression"
