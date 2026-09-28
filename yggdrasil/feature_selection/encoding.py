"""Codificação de features não numéricas para a seleção (pandas ou PySpark).

As etapas de importância, redundância e Boruta só enxergam colunas numéricas; texto,
``category`` e booleanas ficariam de fora sem aviso. Com
``FeatureSelectionConfig(encode_categoricals=True)`` elas viram números antes da
seleção:

* **booleana** → 0/1 (nulo segue nulo);
* **texto / category** → *target encoding*: média do alvo por categoria (taxa de
  maus em classificação), aprendida na **mesma base da seleção** (a amostra de
  desenvolvimento, quando existe). Nulo vira a categoria própria ``(vazio)`` — em
  crédito o missing costuma carregar risco. Categorias com share abaixo de
  ``encoding_min_share`` são agrupadas em ``OUTROS``, o que segura o otimismo do
  encoding em alta cardinalidade; colunas com mais de ``encoding_max_categories``
  categorias distintas não são codificadas (típico de ID/CPF/CEP).

O dicionário devolvido por :func:`fit_encodings` é serializável (JSON) e reaplicável
em qualquer base com :func:`apply_encodings` — categoria nunca vista cai em ``OUTROS``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils import get_logger
from .backend import is_pandas
from .config import FeatureSelectionConfig
from .spark_stats import _require_functions

_logger = get_logger("yggdrasil.feature_selection")

MISSING = "(vazio)"
TIPO_BOOL = "bool_0_1"
TIPO_TE = "target_encoding"


# ── tipos ───────────────────────────────────────────────────────────────
def classify_columns(df, features: List[str]) -> Tuple[List[str], List[str]]:
    """Separa as features em (booleanas, categóricas) — o resto é numérico ou ignorado.

    Categóricas = texto (``object``/``string``) e ``category`` no pandas; ``string``
    no Spark. Datas e outros tipos não entram em nenhuma das duas listas.
    """
    if is_pandas(df):
        from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype
        bools, cats = [], []
        for c in features:
            s = df[c]
            if is_bool_dtype(s):
                bools.append(c)
            elif isinstance(s.dtype, pd.CategoricalDtype) or is_object_dtype(s) or is_string_dtype(s):
                cats.append(c)
        return bools, cats
    tipos = {k: str(v).lower() for k, v in df.dtypes}
    bools = [c for c in features if tipos.get(c) == "boolean"]
    cats = [c for c in features if tipos.get(c) == "string"]
    return bools, cats


# ── fit ─────────────────────────────────────────────────────────────────
def _category_stats(df, col: str, target: str, limit: int) -> pd.DataFrame:
    """Contagem e média do alvo por categoria (nulo = ``MISSING``); até ``limit`` linhas."""
    if is_pandas(df):
        y = pd.to_numeric(df[target], errors="coerce")
        ok = y.notna().to_numpy()
        x = df[col].astype("object")
        x = x.where(x.notna(), MISSING).astype(str)
        g = (pd.DataFrame({"x": x.to_numpy()[ok], "y": y.to_numpy(dtype=float)[ok]})
             .groupby("x")["y"].agg(n="count", media="mean"))
        return g.head(limit)
    F = _require_functions()
    x = F.coalesce(F.col(col).cast("string"), F.lit(MISSING)).alias("x")
    rows = (df.where(F.col(target).isNotNull())
              .select(x, F.col(target).cast("double").alias("y"))
              .groupBy("x").agg(F.count(F.lit(1)).alias("n"), F.avg("y").alias("media"))
              .limit(limit).collect())
    return pd.DataFrame([r.asDict() for r in rows], columns=["x", "n", "media"]).set_index("x")


def _fit_target_encoding(stats: pd.DataFrame, min_share: float) -> dict:
    total = float(stats["n"].sum())
    prior = float((stats["n"] * stats["media"]).sum() / total) if total else np.nan
    share = stats["n"] / total if total else stats["n"] * 0.0
    raras = stats[share < min_share]
    fortes = stats[share >= min_share]
    if len(raras):
        outros = float((raras["n"] * raras["media"]).sum() / raras["n"].sum())
    else:
        outros = prior
    return {
        "tipo": TIPO_TE,
        "mapa": {str(k): round(float(v), 6) for k, v in fortes["media"].items()},
        "outros": round(outros, 6),
        "prior": round(prior, 6),
        "n_categorias": int(len(stats)),
        "n_raras": int(len(raras)),
    }


def fit_encodings(df, features: List[str], target: str,
                  cfg: FeatureSelectionConfig) -> Dict[str, dict]:
    """Aprende a codificação das features booleanas e categóricas de ``features``.

    Retorna ``{feature: especificação}``. Colunas com cardinalidade acima de
    ``cfg.encoding_max_categories`` ficam de fora (com aviso no log).
    """
    bools, cats = classify_columns(df, features)
    enc: Dict[str, dict] = {c: {"tipo": TIPO_BOOL} for c in bools}
    for c in cats:
        stats = _category_stats(df, c, target, cfg.encoding_max_categories + 1)
        if len(stats) > cfg.encoding_max_categories:
            _logger.warning(
                "Feature '%s' não codificada: mais de %d categorias distintas "
                "(encoding_max_categories). Agrupe antes ou remova (ID/CPF/CEP?).",
                c, cfg.encoding_max_categories,
            )
            continue
        if stats.empty:
            continue
        enc[c] = _fit_target_encoding(stats, cfg.encoding_min_share)
    if enc:
        _logger.debug("Features não numéricas codificadas: %s", sorted(enc))
    return enc


# ── apply ───────────────────────────────────────────────────────────────
def apply_encodings(df, encodings: Dict[str, dict]):
    """Aplica ``encodings`` (de :func:`fit_encodings`) mantendo os nomes das colunas.

    Colunas ausentes em ``df`` são ignoradas. Categoria não vista no fit → ``outros``.
    Devolve um novo DataFrame (pandas: cópia; Spark: novo plano).
    """
    encodings = {c: e for c, e in encodings.items() if c in df.columns}
    if not encodings:
        return df
    if is_pandas(df):
        out = df.copy()
        for c, e in encodings.items():
            if e["tipo"] == TIPO_BOOL:
                out[c] = out[c].astype("float")
            else:
                x = out[c].astype("object")
                x = x.where(x.notna(), MISSING).astype(str)
                out[c] = x.map(e["mapa"]).fillna(e["outros"]).astype(float)
        return out
    F = _require_functions()
    from itertools import chain
    out = df
    for c, e in encodings.items():
        if e["tipo"] == TIPO_BOOL:
            out = out.withColumn(c, F.col(c).cast("double"))
            continue
        if e["mapa"]:
            mapa = F.create_map(*chain.from_iterable(
                (F.lit(k), F.lit(float(v))) for k, v in e["mapa"].items()))
            valor = mapa[F.coalesce(F.col(c).cast("string"), F.lit(MISSING))]
        else:
            valor = F.lit(None).cast("double")
        out = out.withColumn(c, F.coalesce(valor, F.lit(float(e["outros"]))))
    return out


__all__ = ["fit_encodings", "apply_encodings", "classify_columns", "MISSING"]
