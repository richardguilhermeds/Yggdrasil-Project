"""Primitivas estatísticas locais (pandas) — espelho de :mod:`spark_stats`.

Cada função aqui tem a mesma assinatura e o mesmo formato de retorno da sua irmã
distribuída, para que o orquestrador (:mod:`selector`) não precise saber qual
backend está rodando. O despacho por tipo de DataFrame acontece nas funções
públicas de :mod:`spark_stats`, :mod:`importance` e :mod:`boruta`.

Duas diferenças **esperadas** em relação ao Spark, ambas a favor do backend local:

* percentis e cardinalidade são **exatos** (``quantile`` / ``nunique``), enquanto o
  Spark usa ``approxQuantile`` / ``approx_count_distinct`` com ``approx_rel_error``.
  Perto de um limiar, uma feature pode cair de um lado no Spark e do outro aqui;
* a importância multivariada usa o ``RandomForest`` do ``sklearn`` em vez do
  ``pyspark.ml``. O ``subsamplingRate`` não tem equivalente direto e é ignorado.

A lógica de decisão (consenso, redundância, Boruta, painéis) é a mesma nos dois
backends — só a camada de estatística muda.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from .config import FeatureSelectionConfig


def numeric_columns(pdf: pd.DataFrame, features: List[str]) -> List[str]:
    """Subconjunto de ``features`` com dtype numérico.

    Booleanos ficam de fora para casar com o Spark, onde ``boolean`` não está entre
    os tipos sobre os quais ``approxQuantile``/``Correlation`` operam.
    """
    from pandas.api.types import is_bool_dtype, is_numeric_dtype

    return [
        c for c in features
        if c in pdf.columns and is_numeric_dtype(pdf[c]) and not is_bool_dtype(pdf[c])
    ]


# ── missing ─────────────────────────────────────────────────────────────
def missing_rate(pdf: pd.DataFrame, features: List[str]) -> pd.Series:
    """Fração de valores ausentes por feature (``isna`` cobre None e NaN)."""
    n = len(pdf)
    if n == 0:
        return pd.Series({c: np.nan for c in features}, name="pct_missing")
    return pd.Series(
        {c: float(pdf[c].isna().sum()) / n for c in features}, name="pct_missing",
    )


# ── variância / cardinalidade ───────────────────────────────────────────
def variance_flags(
    pdf: pd.DataFrame, features: List[str], cfg: Optional[FeatureSelectionConfig] = None,
) -> pd.DataFrame:
    """Teste de variância por percentis + quase-constância (ver :func:`spark_stats.variance_flags`)."""
    cfg = cfg or FeatureSelectionConfig()
    num = set(numeric_columns(pdf, features))

    rows = []
    for c in features:
        col = pdf[c]
        nun = int(col.nunique(dropna=True))
        p_low = p_high = np.nan
        if c in num:
            serie = pd.to_numeric(col, errors="coerce").dropna()
            if len(serie):
                p_low = float(serie.quantile(cfg.var_p_low))
                p_high = float(serie.quantile(cfg.var_p_high))
        if c in num and np.isfinite(p_low) and np.isfinite(p_high):
            sem_var = (p_high - p_low) <= cfg.var_tol
        else:
            sem_var = nun <= 1
        rows.append({
            "feature": c, "p_low": p_low, "p_high": p_high,
            "nunique_approx": nun, "sem_variancia": bool(sem_var),
        })
    out = pd.DataFrame(rows)

    top1 = _top1_share(pdf, [r["feature"] for r in rows if not r["sem_variancia"]])
    out["top1_share"] = out["feature"].map(top1).astype(float)
    out["near_constante"] = out["top1_share"] >= cfg.near_constant
    return out


def _top1_share(pdf: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    """Share do valor modal por feature (ignora nulos)."""
    share: Dict[str, float] = {}
    for c in features:
        sub = pdf[c].dropna()
        if len(sub) == 0:
            share[c] = np.nan
            continue
        share[c] = float(sub.value_counts().iloc[0]) / len(sub)
    return share


# ── correlação ──────────────────────────────────────────────────────────
def _impute_frame(pdf: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Recorta ``cols``, coage para numérico e imputa a mediana nos nulos (cópia)."""
    out = pdf.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce")
    return out.fillna(out.median()).fillna(0.0)


def _corr_matrix(pdf: pd.DataFrame, cols: List[str], method: str) -> pd.DataFrame:
    return _impute_frame(pdf, cols).corr(method=method).round(4)


def correlation_matrices(
    pdf: pd.DataFrame, features: List[str], cfg: Optional[FeatureSelectionConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """Matrizes de correlação **Pearson e Spearman** entre as features numéricas."""
    num = numeric_columns(pdf, features)
    if len(num) < 2:
        vazio = pd.DataFrame()
        return {"pearson": vazio, "spearman": vazio}
    return {
        "pearson": _corr_matrix(pdf, num, "pearson"),
        "spearman": _corr_matrix(pdf, num, "spearman"),
    }


def corr_with_target(
    pdf: pd.DataFrame, features: List[str], target: str,
    cfg: Optional[FeatureSelectionConfig] = None, method: str = "spearman",
) -> pd.Series:
    """Correlação (sinalizada) de cada feature numérica com o alvo."""
    num = numeric_columns(pdf, features)
    if not num:
        return pd.Series(dtype=float, name="corr_target")
    full = _corr_matrix(pdf, num + [target], method)
    serie = full.loc[num, target] if target in full.columns else pd.Series(np.nan, index=num)
    return serie.round(4).rename("corr_target")


# ── amostragem / importância ────────────────────────────────────────────
def maybe_sample(pdf: pd.DataFrame, cfg: FeatureSelectionConfig) -> pd.DataFrame:
    """Amostra ``sample_size`` linhas para as etapas de modelo (se configurado)."""
    if cfg.sample_size and cfg.sample_size > 0 and len(pdf) > cfg.sample_size:
        return pdf.sample(n=cfg.sample_size, random_state=cfg.rf_seed)
    return pdf


def rf_importances(
    pdf: pd.DataFrame, features: List[str], target: str, problem_type: str,
    cfg: FeatureSelectionConfig,
) -> pd.Series:
    """Importância por impureza de um ``RandomForest`` do sklearn. Só features numéricas."""
    from .boruta import _sk_model  # import local: boruta importa importance no topo

    num = numeric_columns(pdf, features)
    if not num:
        return pd.Series(dtype=float, name="rf_importance")

    y = pd.to_numeric(pdf[target], errors="coerce")
    mask = y.notna()
    if not mask.any():
        return pd.Series(np.nan, index=num, name="rf_importance")

    X = _impute_frame(pdf.loc[mask], num)
    model = _sk_model(problem_type, cfg, seed=cfg.rf_seed).fit(X, y[mask])
    imp = np.asarray(model.feature_importances_, dtype=float)
    return pd.Series(np.round(imp, 6), index=num, name="rf_importance")


def univariate_metrics(
    pdf: pd.DataFrame, features: List[str], target: str, cfg: FeatureSelectionConfig,
) -> pd.DataFrame:
    """IV/KS/AUC/Gini univariados por feature numérica (alvo binário)."""
    from .importance import _binned_metrics  # núcleo numpy puro, compartilhado

    num = numeric_columns(pdf, features)
    probs = [i / cfg.n_bins for i in range(1, cfg.n_bins)]
    cols = ["feature", "iv", "ks", "auc", "gini"]
    vazio = {"iv": np.nan, "ks": np.nan, "auc": np.nan, "gini": np.nan}

    rows = []
    for c in features:
        if c not in num:
            continue
        sub = pdf.loc[:, [c, target]]
        x = pd.to_numeric(sub[c], errors="coerce")
        y = pd.to_numeric(sub[target], errors="coerce")
        ok = (x.notna() & y.notna()).to_numpy()
        x = x.to_numpy(dtype=float)[ok]
        y = y.to_numpy(dtype=float)[ok]
        cuts = sorted(set(np.quantile(x, probs))) if (len(x) and probs) else []
        if not cuts:
            rows.append({"feature": c, **vazio})
            continue

        # Mesma convenção do Bucketizer: valor em [split_i, split_{i+1}) => bin i.
        b = np.searchsorted(np.asarray(cuts, dtype=float), x, side="right")
        n_bins = len(cuts) + 1
        n = np.bincount(b, minlength=n_bins).astype(float)
        bad = np.bincount(b, weights=y, minlength=n_bins).astype(float)
        keep = n > 0                      # o groupBy do Spark só devolve bins com linhas
        n, bad = n[keep], bad[keep]

        m = _binned_metrics(n - bad, bad)
        m["feature"] = c
        rows.append(m)
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# ── suporte ao orquestrador ─────────────────────────────────────────────
def infer_problem_type(pdf: pd.DataFrame, cfg: ColumnConfig) -> str:
    """Heurística: alvo binário {0,1} => classification, senão regression."""
    distintos = list(pd.unique(pdf[cfg.target_col].dropna()))[:3]
    try:
        vals = {float(v) for v in distintos}
    except (TypeError, ValueError):
        return "classification"
    return "classification" if len(distintos) <= 2 and vals <= {0.0, 1.0} else "regression"


__all__ = [
    "numeric_columns", "missing_rate", "variance_flags", "correlation_matrices",
    "corr_with_target", "maybe_sample", "rf_importances", "univariate_metrics",
    "infer_problem_type",
]
