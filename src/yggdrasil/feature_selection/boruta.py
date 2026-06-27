"""Boruta — seleção *all-relevant* por comparação com *shadow features*.

Implementação própria (sem o pacote ``boruta``), em duas variantes:

* **Spark-native** (``backend="spark"``): a cada iteração cria um bloco de shadows
  (cópias das features com a ordem das linhas permutada, o que destrói a relação
  com o alvo), treina um ``RandomForest`` do ``pyspark.ml`` em ``[reais + shadows]``
  e conta um *hit* para cada feature real cuja importância supera o limiar das
  shadows.
* **driver** (``backend="driver"`` — *fallback*): amostra para o driver e roda o
  mesmo laço com ``sklearn`` (permutação independente por coluna).

A decisão final (confirmada / tentativa / rejeitada) vem de um teste binomial
bicaudal com correção de Bonferroni sobre a contagem de hits — núcleo puro
(:func:`boruta_decision`) reutilizado pelas duas variantes.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .config import FeatureSelectionConfig
from .importance import _impute_features, maybe_sample
from .spark_stats import _require_functions, numeric_columns


# ── decisão (núcleo puro, testável sem Spark) ───────────────────────────
def boruta_decision(
    hits: pd.Series, n_iter: int, alpha: float = 0.05,
) -> pd.DataFrame:
    """Classifica cada feature em confirmada/tentativa/rejeitada via teste binomial.

    Sob a hipótese nula (feature tão relevante quanto ruído), ``hits`` ~
    Binomial(``n_iter``, 0.5). Aplica Bonferroni pelo nº de features testadas.
    """
    from scipy.stats import binom

    n_features = max(len(hits), 1)
    alpha_c = alpha / n_features
    rows = []
    for feat, h in hits.items():
        h = int(h)
        p_accept = float(binom.sf(h - 1, n_iter, 0.5))  # P(X >= h)
        p_reject = float(binom.cdf(h, n_iter, 0.5))      # P(X <= h)
        if p_accept <= alpha_c and p_accept <= p_reject:
            dec = "confirmada"
        elif p_reject <= alpha_c:
            dec = "rejeitada"
        else:
            dec = "tentativa"
        rows.append({"feature": feat, "hits": h, "n_iter": n_iter,
                     "hit_rate": round(h / n_iter, 4) if n_iter else np.nan, "decisao": dec})
    return pd.DataFrame(rows)


# ── variante driver (sklearn) ───────────────────────────────────────────
def _sk_model(problem_type: str, cfg: FeatureSelectionConfig, seed: int):
    if problem_type == "classification":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=cfg.rf_n_estimators, max_depth=cfg.rf_max_depth,
                                      random_state=seed, n_jobs=-1)
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=cfg.rf_n_estimators, max_depth=cfg.rf_max_depth,
                                 random_state=seed, n_jobs=-1)


def boruta_driver(
    X: pd.DataFrame, y: pd.Series, problem_type: str, cfg: FeatureSelectionConfig,
) -> pd.DataFrame:
    """Boruta no driver (pandas/sklearn) — permutação independente por coluna."""
    feats = list(X.columns)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).fillna(0.0)
    y = pd.to_numeric(y, errors="coerce")
    mask = y.notna()
    X, y = X[mask], y[mask]
    if len(X) == 0 or len(feats) == 0:
        return boruta_decision(pd.Series(0, index=feats, dtype=int), cfg.boruta_max_iter, cfg.boruta_alpha)

    rng = np.random.default_rng(cfg.rf_seed)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    hits = pd.Series(0, index=feats, dtype=int)
    for t in range(cfg.boruta_max_iter):
        shadow = pd.DataFrame({f"shadow_{c}": rng.permutation(X[c].to_numpy()) for c in feats})
        full = pd.concat([X, shadow], axis=1)
        model = _sk_model(problem_type, cfg, seed=cfg.rf_seed + t).fit(full, y)
        imp = np.asarray(model.feature_importances_, dtype=float)
        real_imp, shadow_imp = imp[:len(feats)], imp[len(feats):]
        thr = np.percentile(shadow_imp, cfg.boruta_perc) if len(shadow_imp) else 0.0
        hits += (real_imp > thr).astype(int)
    return boruta_decision(hits, cfg.boruta_max_iter, cfg.boruta_alpha)


# ── variante Spark-native ───────────────────────────────────────────────
def boruta_spark(
    sdf, features: List[str], target: str, problem_type: str, cfg: FeatureSelectionConfig,
) -> pd.DataFrame:
    """Boruta distribuído (pyspark.ml) — permutação em bloco das shadows por iteração."""
    F = _require_functions()
    from pyspark.sql import Window
    from pyspark.ml.feature import VectorAssembler
    if problem_type == "classification":
        from pyspark.ml.classification import RandomForestClassifier as RF
    else:
        from pyspark.ml.regression import RandomForestRegressor as RF

    real = numeric_columns(sdf, features)
    if not real:
        return boruta_decision(pd.Series(dtype=int), cfg.boruta_max_iter, cfg.boruta_alpha)

    base = sdf.where(F.col(target).isNotNull())
    base = _impute_features(base, real, cfg).withColumn("__label__", F.col(target).cast("double"))
    base = base.select("__label__", *real).withColumn(
        "__rid__", F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))
    ).cache()
    base.count()  # materializa o cache

    allcols = real + [f"shadow_{c}" for c in real]
    hits = pd.Series(0, index=real, dtype=int)
    try:
        for t in range(cfg.boruta_max_iter):
            shadows = (base.select("__rid__", *real)
                       .withColumn("__pid__", F.row_number().over(Window.orderBy(F.rand(cfg.rf_seed + t))))
                       .select(F.col("__pid__").alias("__rid__"),
                               *[F.col(c).alias(f"shadow_{c}") for c in real]))
            joined = base.join(shadows, on="__rid__", how="inner")
            vec = VectorAssembler(inputCols=allcols, outputCol="__vec__", handleInvalid="skip").transform(joined)
            model = RF(featuresCol="__vec__", labelCol="__label__", numTrees=cfg.rf_n_estimators,
                       maxDepth=cfg.rf_max_depth, subsamplingRate=cfg.rf_subsampling,
                       seed=cfg.rf_seed + t).fit(vec)
            imp = model.featureImportances.toArray()
            real_imp, shadow_imp = imp[:len(real)], imp[len(real):]
            thr = np.percentile(shadow_imp, cfg.boruta_perc) if len(shadow_imp) else 0.0
            hits += pd.Series((real_imp > thr).astype(int), index=real)
    finally:
        base.unpersist()
    return boruta_decision(hits, cfg.boruta_max_iter, cfg.boruta_alpha)


def boruta_select(
    sdf, features: List[str], target: str, problem_type: str,
    cfg: Optional[FeatureSelectionConfig] = None,
) -> pd.DataFrame:
    """Roda o Boruta no backend configurado e devolve as decisões por feature."""
    cfg = cfg or FeatureSelectionConfig()
    num = numeric_columns(sdf, features)
    if cfg.backend == "driver":
        amostra = maybe_sample(sdf, cfg).select(*num, target).toPandas()
        return boruta_driver(amostra[num], amostra[target], problem_type, cfg)
    return boruta_spark(maybe_sample(sdf, cfg), features, target, problem_type, cfg)


__all__ = ["boruta_decision", "boruta_driver", "boruta_spark", "boruta_select"]
