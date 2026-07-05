"""
Registro de modelos satélite no MLflow (Guia §4.4, §6.2 ``tracking``)
=====================================================================
Versiona um *run* de modelo satélite — especificação, parâmetros, métricas,
diagnóstico e artefatos — no MLflow, atendendo à governança do ciclo de modelos
(§4.4): inventário e versionamento de modelos e parâmetros, trilha de auditoria e
documentação por modelo. Em Databricks, o registro natural é o MLflow para
experimentos e o Unity Catalog para as tabelas de parâmetros e projeções.

Segue o mesmo padrão de :mod:`yggdrasil.credit_risk.capital.tracking`:

* ``mlflow`` é importado **tardiamente**, dentro da função — o pacote continua
  importável sem MLflow (no cálculo puro, sem *tracking*);
* artefatos (tabelas e figuras) são *best-effort*: qualquer falha vira uma *tag*
  de erro em vez de derrubar o *run* (a especificação e as métricas, já logadas,
  são o que importa).
"""
from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .base import FitResult, Projection
    from .selection import SearchResult

DEFAULT_EXPERIMENT = "/Shared/Yggdrasil/modelos_econometricos"


def _log_metric_safe(mlflow, nome: str, valor) -> None:
    if valor is None:
        return
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return
    if np.isfinite(v):
        mlflow.log_metric(nome, v)


def log_satellite_run(
    fit: "FitResult",
    *,
    series=None,
    projection: "Projection" = None,
    search: "SearchResult" = None,
    params: Optional[dict] = None,
    tags: Optional[dict] = None,
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
) -> str:
    """Registra um *run* de modelo satélite no MLflow e devolve o ``run_id``.

    Loga a especificação e o *link* como parâmetros; AIC/BIC/logL/R²/σ e o nº de
    testes de diagnóstico aprovados como métricas; e — *best-effort* — a tabela de
    coeficientes, a bateria de diagnóstico, a projeção e as figuras (ajuste,
    resíduos, leque) como artefatos.

    Parameters
    ----------
    fit:
        O :class:`~yggdrasil.credit_risk.econometric.base.FitResult` do modelo.
    series:
        A :class:`RiskSeries` (para as figuras de ajuste/diagnóstico).
    projection:
        Uma :class:`Projection` (para o gráfico em leque e a tabela de projeção).
    search:
        Um :class:`SearchResult` (loga o RMSE fora da amostra do melhor modelo e
        salva o ranking champion-challenger).
    params, tags:
        Extras do usuário, mesclados aos base.
    experiment, run_name, artifacts_dir:
        Como em :func:`yggdrasil.credit_risk.capital.tracking.log_capital_run`.
    """
    import mlflow  # import tardio: pacote importável sem MLflow

    params = dict(params or {})
    tmp = artifacts_dir or tempfile.mkdtemp(prefix="yggdrasil_satelite_")
    os.makedirs(tmp, exist_ok=True)

    if experiment:
        mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as run:
        # ── parâmetros ──────────────────────────────────────────────
        params.setdefault("model", fit.model_name)
        params.setdefault("kind", fit.kind)
        params.setdefault("link", fit.link)
        if fit.spec is not None:
            params.setdefault("spec", fit.spec.describe())
            params.setdefault("ar", fit.spec.ar)
            params.setdefault("variables", ",".join(fit.spec.variables()) or "(univariado)")
        params.setdefault("nobs", fit.nobs)
        mlflow.log_params(params)

        # ── métricas in-sample ──────────────────────────────────────
        for nome, val in [("AIC", fit.aic), ("BIC", fit.bic), ("logL", fit.llf),
                          ("R2", fit.rsquared), ("sigma", fit.sigma)]:
            _log_metric_safe(mlflow, nome, val)

        # ── diagnóstico: nº de testes aprovados ─────────────────────
        try:
            diag = fit.diagnostics()
            _log_metric_safe(mlflow, "diag_testes_ok", int(diag["ok"].fillna(False).sum()))
            _log_metric_safe(mlflow, "diag_testes_total", int(diag["ok"].notna().sum()))
        except Exception as exc:  # noqa: BLE001
            mlflow.set_tag("diag_error", str(exc)[:200])
            diag = None

        # ── métricas fora da amostra (se houver busca) ──────────────
        if search is not None and getattr(search, "best_spec", None) is not None:
            qual = search.ranking[search.ranking["status"] == "qualificado"]
            if not qual.empty:
                _log_metric_safe(mlflow, "oos_rmse_melhor", float(qual["oos_rmse"].iloc[0]))
                if "vs_arima" in qual.columns:
                    _log_metric_safe(mlflow, "vs_arima_melhor", float(qual["vs_arima"].iloc[0]))

        # ── tags ────────────────────────────────────────────────────
        base_tags = {"framework": "yggdrasil-ml", "model_type": "satellite_econometric",
                     "trained_by": "richard-guilherme"}
        base_tags.update(tags or {})
        mlflow.set_tags(base_tags)

        # ── artefatos: tabelas (best-effort) ────────────────────────
        try:
            fit.coef_frame().to_csv(os.path.join(tmp, "coeficientes.csv"))
            if diag is not None:
                diag.to_csv(os.path.join(tmp, "diagnostico.csv"), index=False)
            if projection is not None:
                projection.to_frame().to_csv(os.path.join(tmp, "projecao.csv"), index=False)
            if search is not None:
                search.ranking.to_csv(os.path.join(tmp, "ranking.csv"), index=False)
            mlflow.log_artifacts(tmp, artifact_path="tables")
        except Exception as exc:  # noqa: BLE001
            mlflow.set_tag("tables_error", str(exc)[:200])

        # ── artefatos: figuras (best-effort, dependem de matplotlib) ─
        try:
            _log_figures(mlflow, fit, series, projection, tmp)
        except Exception as exc:  # noqa: BLE001
            mlflow.set_tag("figures_error", str(exc)[:200])

        return run.info.run_id


def _log_figures(mlflow, fit, series, projection, tmp: str) -> None:
    from . import report
    import matplotlib.pyplot as plt

    def _save(fig, path):
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)

    if series is not None:
        p = os.path.join(tmp, "ajuste.png")
        _save(report.plot_fit(fit, series), p)
        mlflow.log_artifact(p, artifact_path="figures")
        p = os.path.join(tmp, "residuos.png")
        _save(report.plot_residual_diagnostics(fit), p)
        mlflow.log_artifact(p, artifact_path="figures")
    if projection is not None:
        p = os.path.join(tmp, "projecao.png")
        _save(report.plot_projection(projection, series), p)
        mlflow.log_artifact(p, artifact_path="figures")


__all__ = ["log_satellite_run", "DEFAULT_EXPERIMENT"]
