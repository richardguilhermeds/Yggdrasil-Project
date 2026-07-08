"""Primitivas estatísticas distribuídas (PySpark) para a seleção de features.

Todas as funções recebem um Spark DataFrame e devolvem resultados pequenos já
coletados no driver (``pandas.Series`` / ``pandas.DataFrame``), prontos para a
lógica de consenso. O import do ``pyspark`` é *gated* (mesmo padrão de
:meth:`yggdrasil.credit_risk.tree.segmenter.TreeSegmenter.apply_spark`):
o módulo importa sem pyspark instalado e só falha — com mensagem clara — ao
executar uma função distribuída.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import FeatureSelectionConfig

# Tipos numéricos do Spark sobre os quais approxQuantile/Correlation operam.
_NUMERIC_PREFIXES = ("int", "bigint", "smallint", "tinyint", "double", "float", "decimal", "long")


def _require_functions():
    """Importa ``pyspark.sql.functions`` ou levanta erro instrutivo."""
    try:
        from pyspark.sql import functions as F  # noqa: WPS433
        return F
    except ImportError as e:  # pragma: no cover - depende do ambiente
        raise ImportError(
            "A seleção de features requer pyspark — instale com: pip install 'yggdrasil[spark]'"
        ) from e


def numeric_columns(sdf, features: List[str]) -> List[str]:
    """Subconjunto de ``features`` cujo tipo Spark é numérico."""
    tipos = dict(sdf.dtypes)
    return [c for c in features if str(tipos.get(c, "")).lower().startswith(_NUMERIC_PREFIXES)]


# ── missing ─────────────────────────────────────────────────────────────
def missing_rate(sdf, features: List[str]) -> pd.Series:
    """Fração de valores ausentes (NULL e, p/ floats, NaN) por feature — uma passada."""
    F = _require_functions()
    n = sdf.count()
    if n == 0:
        return pd.Series({c: np.nan for c in features}, name="pct_missing")
    tipos = dict(sdf.dtypes)

    def _miss(c: str):
        cond = F.col(c).isNull()
        if str(tipos.get(c, "")).lower() in ("double", "float"):
            cond = cond | F.isnan(F.col(c))
        return F.sum(cond.cast("long")).alias(c)

    row = sdf.agg(*[_miss(c) for c in features]).collect()[0].asDict()
    return pd.Series({c: (row[c] or 0) / n for c in features}, name="pct_missing")


# ── variância / cardinalidade ───────────────────────────────────────────
def variance_flags(sdf, features: List[str], cfg: Optional[FeatureSelectionConfig] = None) -> pd.DataFrame:
    """Teste de variância por percentis (P1 vs P99 por padrão) + quase-constância.

    Colunas: ``feature, p_low, p_high, nunique_approx, top1_share, sem_variancia,
    near_constante``. Uma feature é "sem variância" se ``p_high - p_low <= var_tol``
    (numéricas) ou se tiver no máximo 1 valor distinto; é "quase-constante" se o
    valor modal cobre ``>= near_constant`` das linhas não-nulas.
    """
    F = _require_functions()
    cfg = cfg or FeatureSelectionConfig()
    num = set(numeric_columns(sdf, features))

    # Cardinalidade aproximada (uma passada para todas as features).
    nuniq_row = sdf.agg(
        *[F.approx_count_distinct(F.col(c)).alias(c) for c in features]
    ).collect()[0].asDict()

    # Percentis das numéricas (uma chamada multi-coluna).
    quantis: Dict[str, list] = {}
    num_list = [c for c in features if c in num]
    if num_list:
        q = sdf.approxQuantile(num_list, [cfg.var_p_low, cfg.var_p_high], cfg.approx_rel_error)
        quantis = {c: q[i] for i, c in enumerate(num_list)}

    rows = []
    for c in features:
        nun = int(nuniq_row.get(c) or 0)
        ql = quantis.get(c, [])
        p_low = float(ql[0]) if len(ql) >= 1 else np.nan
        p_high = float(ql[1]) if len(ql) >= 2 else np.nan
        if c in num and np.isfinite(p_low) and np.isfinite(p_high):
            sem_var = (p_high - p_low) <= cfg.var_tol
        else:
            sem_var = nun <= 1
        rows.append({
            "feature": c, "p_low": p_low, "p_high": p_high,
            "nunique_approx": nun, "sem_variancia": bool(sem_var),
        })
    out = pd.DataFrame(rows)

    # Share do valor modal (quase-constância) — só p/ quem ainda tem variância.
    top1 = _top1_share(sdf, [r["feature"] for r in rows if not r["sem_variancia"]])
    out["top1_share"] = out["feature"].map(top1).astype(float)
    out["near_constante"] = out["top1_share"] >= cfg.near_constant
    return out


def _top1_share(sdf, features: List[str]) -> Dict[str, float]:
    """Share do valor modal por feature (ignora nulos). Um job por feature."""
    if not features:
        return {}
    F = _require_functions()
    share: Dict[str, float] = {}
    for c in features:
        sub = sdf.select(c).where(F.col(c).isNotNull())
        total = sub.count()
        if total == 0:
            share[c] = np.nan
            continue
        modal = sub.groupBy(c).count().agg(F.max("count").alias("m")).collect()[0]["m"]
        share[c] = float(modal) / total
    return share


# ── correlação ──────────────────────────────────────────────────────────
def _impute_assemble(sdf, cols: List[str], cfg: FeatureSelectionConfig):
    """Imputa mediana nos nulos e monta o vetor de features (sem nulos)."""
    _require_functions()  # erro instrutivo se pyspark ausente
    from pyspark.ml.feature import VectorAssembler

    medians = sdf.approxQuantile(cols, [0.5], cfg.approx_rel_error)
    fill = {c: (medians[i][0] if medians[i] else 0.0) for i, c in enumerate(cols)}
    sdf2 = sdf.select(*cols).na.fill(fill).na.fill(0.0)
    asm = VectorAssembler(inputCols=cols, outputCol="__vec__", handleInvalid="skip")
    return asm.transform(sdf2).select("__vec__")


def _corr_matrix(sdf, cols: List[str], cfg: FeatureSelectionConfig, method: str) -> pd.DataFrame:
    from pyspark.ml.stat import Correlation
    vec = _impute_assemble(sdf, cols, cfg)
    m = Correlation.corr(vec, "__vec__", method).head()[0].toArray()
    return pd.DataFrame(np.round(m, 4), index=cols, columns=cols)


def correlation_matrices(
    sdf, features: List[str], cfg: Optional[FeatureSelectionConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """Matrizes de correlação **Pearson e Spearman** entre as features numéricas."""
    cfg = cfg or FeatureSelectionConfig()
    num = [c for c in numeric_columns(sdf, features)]
    if len(num) < 2:
        vazio = pd.DataFrame()
        return {"pearson": vazio, "spearman": vazio}
    return {
        "pearson": _corr_matrix(sdf, num, cfg, "pearson"),
        "spearman": _corr_matrix(sdf, num, cfg, "spearman"),
    }


def corr_with_target(
    sdf, features: List[str], target: str, cfg: Optional[FeatureSelectionConfig] = None,
    method: str = "spearman",
) -> pd.Series:
    """Correlação (sinalizada) de cada feature numérica com o alvo."""
    cfg = cfg or FeatureSelectionConfig()
    num = numeric_columns(sdf, features)
    if not num:
        return pd.Series(dtype=float, name="corr_target")
    cols = num + [target]
    full = _corr_matrix(sdf, cols, cfg, method)
    serie = full.loc[num, target] if target in full.columns else pd.Series(np.nan, index=num)
    return serie.round(4).rename("corr_target")


# ── redundância ─────────────────────────────────────────────────────────
def redundancy_clusters(
    corr: pd.DataFrame, threshold: float, importance: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Agrupa features com ``|corr| > threshold`` e elege um representante por cluster.

    Recebe uma matriz de correlação já calculada (mesma lógica de
    :func:`yggdrasil.eda.correlation.redundancy_clusters`, mas sobre matriz pronta).
    O representante é a feature de maior ``importance`` (se fornecida); as demais do
    cluster ficam marcadas como redundantes. Colunas: ``feature, cluster,
    representante, redundante_com``.
    """
    if corr is None or corr.empty or len(corr) < 2:
        feats = list(corr.columns) if corr is not None and len(corr.columns) else []
        return pd.DataFrame({
            "feature": feats, "cluster": list(range(1, len(feats) + 1)),
            "representante": [True] * len(feats), "redundante_com": [None] * len(feats),
        })
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    d = 1.0 - corr.abs().values
    d = np.nan_to_num(d, nan=1.0)
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2.0
    Z = linkage(squareform(d, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    feats = list(corr.columns)
    imp = importance if importance is not None else pd.Series(0.0, index=feats)
    rows = []
    for cl in np.unique(labels):
        membros = [feats[i] for i in range(len(feats)) if labels[i] == cl]
        rep = max(membros, key=lambda f: (imp.get(f, 0.0) if np.isfinite(imp.get(f, np.nan)) else -np.inf))
        for f in membros:
            rows.append({
                "feature": f, "cluster": int(cl),
                "representante": f == rep,
                "redundante_com": None if f == rep else rep,
            })
    return pd.DataFrame(rows)


__all__ = [
    "numeric_columns", "missing_rate", "variance_flags",
    "correlation_matrices", "corr_with_target", "redundancy_clusters",
]
