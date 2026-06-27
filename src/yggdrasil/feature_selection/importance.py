"""Indicadores de importância de features (Spark-native).

Consolida, num único ranking por feature:

* **multivariada**: importância por impureza de um ``RandomForest`` do
  ``pyspark.ml`` (escala em todos os dados);
* **univariada (classificação)**: IV (WoE, escala Siddiqi), KS, AUC e Gini, via
  binning por quantis calculado no Spark;
* **relação com o alvo**: correlação (Spearman) com o target.

O ``score`` final é a média dos *ranks* das métricas disponíveis (mesmo método de
:func:`yggdrasil.eda.importance.importance_ranking`), e há uma flag de suspeita de
*leakage*. Tudo é devolvido como ``pandas.DataFrame`` (pequeno) já no driver.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import FeatureSelectionConfig
from .spark_stats import _require_functions, corr_with_target, numeric_columns


def maybe_sample(sdf, cfg: FeatureSelectionConfig):
    """Amostra ``sample_size`` linhas para as etapas de modelo (se configurado)."""
    if cfg.sample_size and cfg.sample_size > 0:
        n = sdf.count()
        if n > cfg.sample_size:
            return sdf.sample(False, cfg.sample_size / n, seed=cfg.rf_seed)
    return sdf


def _impute_features(sdf, cols: List[str], cfg: FeatureSelectionConfig):
    """Seleciona/imputa (mediana) as colunas numéricas para a modelagem."""
    medians = sdf.approxQuantile(cols, [0.5], cfg.approx_rel_error)
    fill = {c: (medians[i][0] if medians[i] else 0.0) for i, c in enumerate(cols)}
    return sdf.na.fill(fill).na.fill(0.0)


def rf_importances(
    sdf, features: List[str], target: str, problem_type: str, cfg: FeatureSelectionConfig,
) -> pd.Series:
    """Importância por impureza de um RandomForest (pyspark.ml). Só features numéricas."""
    F = _require_functions()
    from pyspark.ml.feature import VectorAssembler

    num = numeric_columns(sdf, features)
    if not num:
        return pd.Series(dtype=float, name="rf_importance")

    base = sdf.where(F.col(target).isNotNull())
    base = _impute_features(base, num, cfg).withColumn("__label__", F.col(target).cast("double"))
    vec = VectorAssembler(inputCols=num, outputCol="__vec__", handleInvalid="skip").transform(base)

    if problem_type == "classification":
        from pyspark.ml.classification import RandomForestClassifier as RF
    else:
        from pyspark.ml.regression import RandomForestRegressor as RF
    model = RF(
        featuresCol="__vec__", labelCol="__label__",
        numTrees=cfg.rf_n_estimators, maxDepth=cfg.rf_max_depth,
        subsamplingRate=cfg.rf_subsampling, seed=cfg.rf_seed,
    ).fit(vec)

    imp = model.featureImportances.toArray()
    return pd.Series(np.round(imp, 6), index=num, name="rf_importance")


# ── univariada (classificação): IV / KS / AUC / Gini ────────────────────
def _binned_metrics(good: np.ndarray, bad: np.ndarray) -> Dict[str, float]:
    """IV/KS/AUC/Gini a partir de bins ordenados por valor da feature."""
    G, B = good.sum(), bad.sum()
    if G <= 0 or B <= 0 or len(good) < 2:
        return {"iv": np.nan, "ks": np.nan, "auc": np.nan, "gini": np.nan}
    dg, db = good / G, bad / B
    k = len(good)
    dg_s = (good + 0.5) / (G + 0.5 * k)
    db_s = (bad + 0.5) / (B + 0.5 * k)
    woe = np.log(dg_s / db_s)
    iv = float(np.sum((dg_s - db_s) * woe))
    ks = float(np.max(np.abs(np.cumsum(dg) - np.cumsum(db))))
    bad_below = np.cumsum(bad) - bad  # bad estritamente abaixo de cada bin
    auc = float(np.sum(good * (bad_below + bad / 2.0)) / (G * B))
    auc = max(auc, 1.0 - auc)  # direção-agnóstico
    return {"iv": round(iv, 4), "ks": round(ks, 4),
            "auc": round(auc, 4), "gini": round(2 * auc - 1, 4)}


def univariate_metrics(
    sdf, features: List[str], target: str, cfg: FeatureSelectionConfig,
) -> pd.DataFrame:
    """IV/KS/AUC/Gini univariados por feature numérica (alvo binário)."""
    F = _require_functions()
    from pyspark.ml.feature import Bucketizer

    num = numeric_columns(sdf, features)
    probs = [i / cfg.n_bins for i in range(1, cfg.n_bins)]
    rows = []
    for c in num:
        sub = sdf.select(c, target).where(F.col(c).isNotNull() & F.col(target).isNotNull())
        cuts = sorted(set(sub.approxQuantile(c, probs, cfg.approx_rel_error)))
        if len(cuts) < 1:
            rows.append({"feature": c, "iv": np.nan, "ks": np.nan, "auc": np.nan, "gini": np.nan})
            continue
        splits = [-float("inf")] + cuts + [float("inf")]
        bkt = Bucketizer(splits=splits, inputCol=c, outputCol="__b__", handleInvalid="skip")
        grp = (bkt.transform(sub)
               .groupBy("__b__")
               .agg(F.count(F.lit(1)).alias("n"), F.sum(F.col(target).cast("double")).alias("bad"))
               .orderBy("__b__")
               .collect())
        n = np.array([r["n"] for r in grp], dtype=float)
        bad = np.array([r["bad"] or 0.0 for r in grp], dtype=float)
        good = n - bad
        m = _binned_metrics(good, bad)
        m["feature"] = c
        rows.append(m)
    cols = ["feature", "iv", "ks", "auc", "gini"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# ── consolidação ────────────────────────────────────────────────────────
def importance_indicators(
    sdf, features: List[str], target: str, problem_type: str,
    cfg: Optional[FeatureSelectionConfig] = None,
) -> pd.DataFrame:
    """Consolida todos os indicadores de importância num ranking por feature."""
    cfg = cfg or FeatureSelectionConfig()
    model_sdf = maybe_sample(sdf, cfg)

    out = pd.DataFrame({"feature": list(features)})
    rf = rf_importances(model_sdf, features, target, problem_type, cfg)
    out = out.merge(rf.reset_index().rename(columns={"index": "feature"}), on="feature", how="left")

    ct = corr_with_target(sdf, features, target, cfg)
    out = out.merge(ct.reset_index().rename(columns={"index": "feature"}), on="feature", how="left")

    if problem_type == "classification":
        uni = univariate_metrics(sdf, features, target, cfg)
        out = out.merge(uni, on="feature", how="left")

    # score composto = média dos ranks (maior = mais importante).
    rankaveis = []
    for c in ("rf_importance", "iv", "ks", "gini", "corr_target"):
        if c in out.columns and out[c].notna().any():
            base = out[c].abs() if c == "corr_target" else out[c]
            out[f"__r_{c}"] = base.rank(ascending=True, na_option="keep")
            rankaveis.append(f"__r_{c}")
    if rankaveis:
        out["score"] = out[rankaveis].mean(axis=1)
        out = out.drop(columns=rankaveis).sort_values("score", ascending=False)
    else:
        out["score"] = np.nan
    out = out.reset_index(drop=True)

    gini = out.get("gini", pd.Series(np.nan, index=out.index)).fillna(0).abs()
    iv = out.get("iv", pd.Series(np.nan, index=out.index)).fillna(0)
    out["leakage_flag"] = (gini > (2 * cfg.leakage_auc - 1)) | (iv > cfg.iv_leakage)
    return out


__all__ = [
    "maybe_sample", "rf_importances", "univariate_metrics", "importance_indicators",
]
