"""Validação e tipagem semântica das features para a EDA.

Diferente de `yggdrasil.data.validate_input` (que EXIGE target e DES/OOT), aqui a
validação é **tolerante a target ausente** — EDA pura precisa rodar só com as
features, a data e a amostra.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from ..config import ColumnConfig
from .config import EDAConfig

FEATURE_KINDS = ("numeric", "binary", "categorical", "datetime")


def has_target(df: pd.DataFrame, cfg: ColumnConfig) -> bool:
    """True se a coluna alvo existe e tem ao menos um valor observado."""
    return cfg.target_col in df.columns and bool(df[cfg.target_col].notna().any())


def validate_input_eda(df: pd.DataFrame, cfg: ColumnConfig) -> None:
    """Valida o contrato mínimo para EDA (target é opcional)."""
    faltando = [c for c in (cfg.date_col, cfg.sample_col) if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes para EDA: {faltando}")
    cfg.feature_columns(df)  # levanta ValueError se não houver colunas feat_


def _is_binary(values: np.ndarray) -> bool:
    try:
        vals = set(np.asarray(values, dtype=float))
    except (TypeError, ValueError):
        return False
    return len(vals) <= 2 and vals <= {0.0, 1.0}


def infer_feature_kind(series: pd.Series, eda_cfg: EDAConfig | None = None) -> str:
    """Inferência semântica: numeric | binary | categorical | datetime."""
    s = series
    if is_datetime64_any_dtype(s):
        return "datetime"
    if is_bool_dtype(s):
        return "binary"
    u = s.dropna().unique()
    if is_numeric_dtype(s):
        if _is_binary(u):
            return "binary"
        return "numeric"
    # object/category: tenta numérica mascarada de string
    coerced = pd.to_numeric(s, errors="coerce")
    nao_nulos = s.notna().sum()
    if nao_nulos > 0 and coerced.notna().sum() >= 0.95 * nao_nulos:
        return "binary" if _is_binary(coerced.dropna().unique()) else "numeric"
    return "categorical"


def classify_features(
    df: pd.DataFrame, cfg: ColumnConfig, eda_cfg: EDAConfig | None = None
) -> Dict[str, str]:
    """Mapeia cada feature ``feat_`` ao seu tipo semântico."""
    eda_cfg = eda_cfg or EDAConfig()
    return {c: infer_feature_kind(df[c], eda_cfg) for c in cfg.feature_columns(df)}


def numeric_features(kinds: Dict[str, str]) -> List[str]:
    """Features tratadas como numéricas (numeric + binary)."""
    return [f for f, k in kinds.items() if k in ("numeric", "binary")]


def categorical_features(kinds: Dict[str, str]) -> List[str]:
    return [f for f, k in kinds.items() if k == "categorical"]


def as_numeric(series: pd.Series) -> pd.Series:
    """Coage uma feature numérica (inclusive numérica-mascarada-de-string) para float."""
    if is_numeric_dtype(series) or is_bool_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")


def apply_missing_codes(df: pd.DataFrame, cols, codes) -> pd.DataFrame:
    """Converte sentinelas (``codes``) em NaN nas colunas indicadas (opt-in)."""
    if not codes:
        return df
    out = df.copy()
    codes = list(codes)
    for c in cols:
        if c in out.columns:
            out[c] = out[c].where(~out[c].isin(codes), other=np.nan)
    return out


__all__ = [
    "FEATURE_KINDS",
    "has_target",
    "validate_input_eda",
    "infer_feature_kind",
    "classify_features",
    "numeric_features",
    "categorical_features",
    "as_numeric",
    "apply_missing_codes",
]
