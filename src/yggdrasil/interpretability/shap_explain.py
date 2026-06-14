"""Interpretabilidade via SHAP.

Calcula valores SHAP de forma robusta (TreeExplainer / Linear / fallback Kernel),
gera a importância global por feature (média de |SHAP|) e salva os gráficos
(beeswarm e barras) para serem logados como artefatos no MLflow.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _sample_X(X: pd.DataFrame, sample_size: Optional[int], random_state: int = 42) -> pd.DataFrame:
    if sample_size and len(X) > sample_size:
        return X.sample(sample_size, random_state=random_state)
    return X


def compute_shap(
    model,
    X: pd.DataFrame,
    problem_type: str = "classification",
    sample_size: Optional[int] = 2000,
    random_state: int = 42,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Calcula a matriz de valores SHAP ``(n_amostras, n_features)``.

    Retorna ``(shap_values, X_amostrado)``. Para classificação multi-saída,
    seleciona a contribuição da classe positiva.
    """
    import shap

    Xs = _sample_X(X, sample_size, random_state)

    # 1) API unificada (cobre árvore e modelos lineares automaticamente).
    try:
        explainer = shap.Explainer(model, Xs)
        exp = explainer(Xs)
        vals = np.asarray(exp.values)
        if vals.ndim == 3:  # (n, features, classes)
            vals = vals[:, :, -1]
        return vals, Xs
    except Exception:
        pass

    # 2) TreeExplainer direto.
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xs)
        if isinstance(sv, list):
            sv = sv[-1]
        return np.asarray(sv), Xs
    except Exception:
        pass

    # 3) Fallback Kernel (lento) com background reduzido.
    if problem_type == "classification" and hasattr(model, "predict_proba"):
        f = lambda d: model.predict_proba(d)[:, 1]
    else:
        f = model.predict
    bg = shap.sample(Xs, min(100, len(Xs)), random_state=random_state)
    explainer = shap.KernelExplainer(f, bg)
    sv = explainer.shap_values(Xs, nsamples=100)
    if isinstance(sv, list):
        sv = sv[-1]
    return np.asarray(sv), Xs


def shap_feature_importance(shap_values: np.ndarray, feature_names) -> pd.DataFrame:
    """Importância global = média do valor absoluto de SHAP por feature."""
    imp = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": list(feature_names), "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def save_shap_plots(shap_values: np.ndarray, Xs: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    """Salva beeswarm e gráfico de barras de SHAP; devolve os caminhos."""
    import matplotlib.pyplot as plt
    import shap

    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    plt.figure()
    shap.summary_plot(shap_values, Xs, show=False)
    beeswarm = os.path.join(out_dir, "shap_beeswarm.png")
    plt.savefig(beeswarm, dpi=120, bbox_inches="tight")
    plt.close()
    paths["beeswarm"] = beeswarm

    plt.figure()
    shap.summary_plot(shap_values, Xs, plot_type="bar", show=False)
    bar = os.path.join(out_dir, "shap_importance_bar.png")
    plt.savefig(bar, dpi=120, bbox_inches="tight")
    plt.close()
    paths["bar"] = bar

    return paths


def shap_report(
    model,
    X: pd.DataFrame,
    feature_names,
    problem_type: str,
    out_dir: str,
    sample_size: Optional[int] = 2000,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Pipeline completo de SHAP: importância + gráficos salvos em ``out_dir``.

    Robusto a falhas: se o SHAP não suportar o modelo, devolve um DataFrame
    vazio e nenhum gráfico em vez de quebrar a esteira.
    """
    try:
        shap_values, Xs = compute_shap(model, X, problem_type, sample_size)
    except Exception as exc:  # noqa: BLE001 - SHAP é best-effort
        return pd.DataFrame(columns=["feature", "mean_abs_shap"]), {"error": str(exc)}

    importance = shap_feature_importance(shap_values, feature_names)
    os.makedirs(out_dir, exist_ok=True)
    importance.to_csv(os.path.join(out_dir, "shap_importance.csv"), index=False)
    try:
        paths = save_shap_plots(shap_values, Xs, out_dir)
    except Exception as exc:  # noqa: BLE001
        paths = {"error": str(exc)}
    paths["importance_csv"] = os.path.join(out_dir, "shap_importance.csv")
    return importance, paths
