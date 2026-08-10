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


def validate_input_report(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    problem_type: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """Relatório **não bloqueante** de qualidade do contrato de entrada.

    Complementa :func:`validate_input` (que levanta erros duros): devolve uma
    lista de findings ``(severidade, coluna, mensagem)``, com severidade
    ``'erro'`` (quase certamente quebra ou distorce a esteira) ou ``'atencao'``
    (vale conferir antes de seguir). Nunca levanta exceção — coluna ausente
    apenas pula as checagens correspondentes.

    Checagens: data não conversível, target com NaN/não numérico nas amostras
    de análise, feature não numérica, coluna de score pré-existente, amostra
    NaN, índice duplicado e divergência entre ``problem_type`` informado e o
    inferido pelo target.
    """
    findings: List[Tuple[str, str, str]] = []

    # ── coluna de data conversível a datetime ──────────────────────────
    if cfg.date_col in df.columns:
        col = df[cfg.date_col]
        if not pd.api.types.is_datetime64_any_dtype(col):
            convertido = pd.to_datetime(col, errors="coerce")
            invalido = convertido.isna() & col.notna()
            if invalido.any():
                exemplo = col[invalido].iloc[0]
                findings.append((
                    "erro", cfg.date_col,
                    f"{int(invalido.sum())} valor(es) não conversível(is) a data "
                    f"(ex.: {exemplo!r}) — análises temporais (PSI por safra) quebram",
                ))

    # ── target nas amostras de análise ─────────────────────────────────
    if cfg.target_col in df.columns and cfg.sample_col in df.columns:
        alvo = df.loc[df[cfg.sample_col].isin(cfg.analysis_samples), cfg.target_col]
        n_nan = int(alvo.isna().sum())
        if n_nan:
            findings.append((
                "erro", cfg.target_col,
                f"{n_nan} NaN no target dentro das amostras de análise "
                f"({', '.join(cfg.analysis_samples)}) — métricas ficam inválidas",
            ))
        if not (pd.api.types.is_numeric_dtype(alvo) or pd.api.types.is_bool_dtype(alvo)):
            findings.append((
                "erro", cfg.target_col,
                f"target não numérico (dtype {alvo.dtype}) nas amostras de análise",
            ))

    # ── features não numéricas (dtype object/str/categoria) ────────────
    try:
        feats = cfg.feature_columns(df)
    except ValueError:
        feats = []
    for c in feats:
        if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])):
            findings.append((
                "atencao", c,
                f"feature não numérica (dtype {df[c].dtype}) — modelos sklearn "
                "esperam numérico; converta ou aplique encoding antes do fit",
            ))

    # ── coluna de score pré-existente ──────────────────────────────────
    if cfg.score_col in df.columns:
        findings.append((
            "atencao", cfg.score_col,
            f"coluna de score '{cfg.score_col}' já existe e será sobrescrita "
            "pela predição do modelo",
        ))

    # ── amostra NaN ────────────────────────────────────────────────────
    if cfg.sample_col in df.columns:
        n_nan = int(df[cfg.sample_col].isna().sum())
        if n_nan:
            findings.append((
                "atencao", cfg.sample_col,
                f"{n_nan} linha(s) com amostra NaN — recebem score, mas ficam "
                "fora das análises e dos relatórios",
            ))

    # ── índice duplicado ───────────────────────────────────────────────
    n_dup = int(df.index.duplicated().sum())
    if n_dup:
        findings.append((
            "atencao", "(índice)",
            f"{n_dup} rótulo(s) de índice duplicado(s) — pode distorcer "
            "alinhamentos/joins durante a esteira",
        ))

    # ── problem_type informado × inferido pelo target ──────────────────
    if problem_type is not None and cfg.target_col in df.columns:
        try:
            inferido = infer_problem_type(df, cfg)
        except (TypeError, ValueError):
            inferido = None  # target não numérico: já reportado acima
        if inferido is not None and inferido != problem_type:
            findings.append((
                "atencao", cfg.target_col,
                f"problem_type informado ('{problem_type}') diverge do inferido "
                f"pelo target ('{inferido}')",
            ))

    return findings


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
