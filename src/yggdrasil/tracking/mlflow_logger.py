"""Registro da esteira no MLflow.

Centraliza o logging de um run: parâmetros, métricas por amostra (DES/OOT),
shifts, PSI (agregado e séries temporais por rating), relatórios por grupo,
dashboard, SHAP e o próprio modelo. Estende o padrão ``log_credit_model`` da
referência de MLflow do projeto.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from ..interpretability import shap_report
from ..monitoring import psi_rating_over_time, psi_score_over_time
from ..reporting import group_reports_to_html, save_dashboard

DEFAULT_EXPERIMENT = "/Shared/Yggdrasil/esteira_ml"


def _log_metric_dict(mlflow, metrics: Dict[str, float], suffix: str = "") -> None:
    for nome, valor in metrics.items():
        if valor is None or not np.isfinite(valor):
            continue
        chave = f"{nome}{suffix}"
        mlflow.log_metric(chave, float(valor))


def _plot_psi_series(series: pd.DataFrame, titulo: str, path: str) -> None:
    import matplotlib.pyplot as plt
    from ..monitoring.psi import PSI_SIGNIFICANT, PSI_STABLE
    from ..reporting.style import COR_NEUTRA, COR_PRIMARIA, COR_SECUNDARIA

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series["mes"], series["psi"], marker="o", color=COR_PRIMARIA, linewidth=2)
    ax.axhline(PSI_STABLE, color=COR_NEUTRA, ls="--", lw=1, label=f"estável ({PSI_STABLE})")
    ax.axhline(PSI_SIGNIFICANT, color=COR_SECUNDARIA, ls="--", lw=1, label=f"instável ({PSI_SIGNIFICANT})")
    ax.set_title(titulo, fontweight="bold")
    ax.set_ylabel("PSI")
    ax.set_xlabel("Mês de referência")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _model_predict_for_signature(model, X, problem_type):
    if problem_type == "classification" and hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def log_pipeline_run(
    *,
    model,
    df_scored: pd.DataFrame,
    cfg: ColumnConfig,
    problem_type: str,
    rating_cols: Sequence[str],
    metrics_by_sample: Dict[str, Dict[str, float]],
    shifts: Dict[str, float],
    psi_metrics: Dict[str, float],
    reports: Dict[str, pd.DataFrame],
    params: Optional[dict] = None,
    tags: Optional[dict] = None,
    experiment: str = DEFAULT_EXPERIMENT,
    run_name: Optional[str] = None,
    X_train: Optional[pd.DataFrame] = None,
    registered_model_name: Optional[str] = None,
    log_shap: bool = True,
    artifacts_dir: Optional[str] = None,
) -> str:
    """Loga um run completo da esteira e retorna o ``run_id``."""
    import mlflow
    from mlflow.models.signature import infer_signature

    rating_cols = list(rating_cols)
    params = dict(params or {})
    tmp = artifacts_dir or tempfile.mkdtemp(prefix="yggdrasil_run_")
    os.makedirs(tmp, exist_ok=True)

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        # ── parâmetros ──────────────────────────────────────────────────
        params.setdefault("problem_type", problem_type)
        params.setdefault("ratings", ",".join(c.replace("rating_", "") for c in rating_cols))
        params.setdefault("model_class", type(model).__name__)
        mlflow.log_params(params)

        # ── métricas por amostra (sufixo _des / _oot / ...) ─────────────
        for amostra, met in metrics_by_sample.items():
            _log_metric_dict(mlflow, met, suffix=f"_{amostra.lower()}")

        # ── shifts e PSI agregado ───────────────────────────────────────
        _log_metric_dict(mlflow, shifts)
        _log_metric_dict(mlflow, psi_metrics)

        # ── tags ────────────────────────────────────────────────────────
        base_tags = {
            "framework": "yggdrasil-ml",
            "problem_type": problem_type,
            "trained_by": "richard-guilherme",
        }
        base_tags.update(tags or {})
        mlflow.set_tags(base_tags)

        # ── artefatos: relatórios por grupo ─────────────────────────────
        for col, rep in reports.items():
            rep.to_csv(os.path.join(tmp, f"group_report_{col}.csv"), index=False)
        html = group_reports_to_html(reports)
        html_path = os.path.join(tmp, "group_reports.html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        mlflow.log_artifacts(tmp, artifact_path="reports")

        # ── artefatos: PSI ao longo do tempo (por rating + score) ───────
        psi_dir = os.path.join(tmp, "psi")
        os.makedirs(psi_dir, exist_ok=True)
        score_ts = psi_score_over_time(df_scored, cfg)
        score_ts.to_csv(os.path.join(psi_dir, "psi_score_over_time.csv"), index=False)
        _plot_psi_series(score_ts, "PSI do score ao longo do tempo",
                         os.path.join(psi_dir, "psi_score_over_time.png"))
        for col in rating_cols:
            ts = psi_rating_over_time(df_scored, col, cfg)
            metodo = col.replace("rating_", "")
            ts.to_csv(os.path.join(psi_dir, f"psi_{metodo}_over_time.csv"), index=False)
            _plot_psi_series(ts, f"PSI do rating '{metodo}' ao longo do tempo",
                             os.path.join(psi_dir, f"psi_{metodo}_over_time.png"))
        mlflow.log_artifacts(psi_dir, artifact_path="psi")

        # ── artefato: dashboard ─────────────────────────────────────────
        dash_path = os.path.join(tmp, "dashboard.png")
        oot_metrics = metrics_by_sample.get(cfg.oot_sample, {})
        try:
            save_dashboard(df_scored, rating_cols, cfg, problem_type, dash_path,
                           metrics=oot_metrics, model=model, X_shap=X_train)
            mlflow.log_artifact(dash_path, artifact_path="dashboard")
        except Exception as exc:  # noqa: BLE001 - dashboard é best-effort
            mlflow.set_tag("dashboard_error", str(exc)[:250])

        # ── artefatos: SHAP ─────────────────────────────────────────────
        if log_shap and X_train is not None:
            shap_dir = os.path.join(tmp, "shap")
            importance, _ = shap_report(
                model, X_train, list(X_train.columns), problem_type, shap_dir
            )
            if os.path.isdir(shap_dir):
                mlflow.log_artifacts(shap_dir, artifact_path="shap")

        # ── importância de features do modelo (se houver) ───────────────
        if hasattr(model, "feature_importances_") and X_train is not None:
            fi = dict(zip(X_train.columns, np.asarray(model.feature_importances_, dtype=float)))
            mlflow.log_dict(fi, "feature_importance.json")

        # ── modelo com assinatura ───────────────────────────────────────
        try:
            if X_train is not None:
                preds = _model_predict_for_signature(model, X_train, problem_type)
                signature = infer_signature(X_train, preds)
            else:
                signature = None
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                registered_model_name=registered_model_name,
            )
        except Exception as exc:  # noqa: BLE001 - não derruba o run por causa do modelo
            mlflow.set_tag("model_log_error", str(exc)[:250])

        return run.info.run_id


__all__ = ["log_pipeline_run", "DEFAULT_EXPERIMENT"]
