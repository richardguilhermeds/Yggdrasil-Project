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


def _coverage_table(backtest):
    """Normaliza o insumo de cobertura de intervalos numa tabela por passo.

    Aceita a própria tabela de
    :func:`~yggdrasil.credit_risk.econometric.selection.interval_coverage`
    (DataFrame), o dict de
    :func:`~yggdrasil.credit_risk.econometric.selection.backtest_projection`
    (chave ``coverage``) ou o dict de um ``walk_forward`` com bandas (chave
    ``bands``, cobertura calculada aqui). Devolve ``None`` se não há o que medir.
    """
    import pandas as pd

    if backtest is None:
        return None
    if isinstance(backtest, pd.DataFrame):
        return backtest
    if isinstance(backtest, dict):
        if backtest.get("coverage") is not None:
            return backtest["coverage"]
        bands = backtest.get("bands")
        if bands is not None and len(bands):
            from .selection import interval_coverage

            return interval_coverage(bands, backtest.get("alpha"))
    return None


#: colunas da aba Resumo do relatório (comparação dos runs do experimento)
_REPORT_METRICS_SPEC = [
    ("params.model", "modelo"),
    ("metrics.AIC", "AIC"),
    ("metrics.oos_rmse_melhor", "RMSE (OOS)"),
    ("metrics.diag_testes_ok", "diag. ok"),
    ("metrics.diag_testes_total", "diag. total"),
]


def _log_satellite_report(mlflow, run, fit, projection, search, diag, tmp: str) -> None:
    """Monta e loga o relatório HTML em abas (``report.html``) do *run*.

    Import tardio de :mod:`.._mlflow_report` (padrão do pacote). Abas: Resumo
    (AIC · RMSE fora da amostra · diagnóstico, comparando os runs do
    experimento) · Coeficientes + diagnóstico · Projeção por cenário (esta só
    quando há projeção no run).
    """
    from .._mlflow_report import build_tabbed_report_html, runs_comparison_df

    resumo = runs_comparison_df(mlflow, run.info.experiment_id, run.info.run_id,
                                metrics_spec=_REPORT_METRICS_SPEC)
    blocos_resumo = [("Comparação dos runs do experimento — AIC · RMSE (OOS) · diagnóstico",
                      resumo)]
    ranking = getattr(search, "ranking", None)
    if ranking is not None and len(ranking):
        blocos_resumo.append(("Champion-challenger — ranking da busca (top 10)",
                              ranking.head(10)))
    tabs = [("① Resumo · runs", blocos_resumo)]

    coefs = fit.coef_frame().reset_index().rename(columns={"index": "termo"})
    blocos_coef = [("Coeficientes", coefs)]
    if diag is not None and len(diag):
        blocos_coef.append(("Bateria de diagnóstico de resíduos", diag))
    tabs.append(("② Coeficientes · diagnóstico", blocos_coef))

    if projection is not None:
        tabs.append(("③ Projeção por cenário",
                     [("Projeção condicional aos cenários — média e intervalo",
                       projection.to_frame())]))

    titulo = f"Modelo satélite — {fit.model_name} ({fit.kind})"
    sub = f"link {fit.link} · nobs={fit.nobs} · run {run.info.run_id[:8]}"
    html_doc = build_tabbed_report_html(titulo, sub, tabs, highlight=("", "➤"))
    path = os.path.join(tmp, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    mlflow.log_artifact(path)


def log_satellite_run(
    fit: "FitResult",
    *,
    series=None,
    projection: "Projection" = None,
    search: "SearchResult" = None,
    backtest=None,
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
    resíduos, leque) como artefatos, além de um relatório HTML autocontido em
    abas (``report.html``): Resumo (AIC · RMSE fora da amostra · diagnóstico,
    comparando os runs do experimento) · Coeficientes + diagnóstico · Projeção
    por cenário.

    Parameters
    ----------
    fit:
        O :class:`~yggdrasil.credit_risk.econometric.base.FitResult` do modelo.
    series:
        A :class:`RiskSeries` (para as figuras de ajuste/diagnóstico).
    projection:
        Uma :class:`Projection` (para o gráfico em leque e a tabela de projeção).
    search:
        Um :class:`SearchResult` (loga o RMSE fora da amostra — e a cobertura de
        intervalos, quando presente no ranking — do melhor modelo e salva o
        ranking champion-challenger).
    backtest:
        O backtest de cobertura de intervalos: o dict de
        :func:`~yggdrasil.credit_risk.econometric.selection.backtest_projection`
        (ou de um ``walk_forward`` com bandas), ou a própria tabela de
        :func:`~yggdrasil.credit_risk.econometric.selection.interval_coverage`.
        Loga cobertura empírica × nominal e os p-valores de Kupiec e
        Christoffersen; salva a tabela por passo como artefato.
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
                # cobertura de intervalos do melhor (colunas aditivas do ranking)
                if "cobertura" in qual.columns:
                    _log_metric_safe(mlflow, "cobertura_melhor", qual["cobertura"].iloc[0])
                if "kupiec_pvalue" in qual.columns:
                    _log_metric_safe(mlflow, "kupiec_pvalue_melhor", qual["kupiec_pvalue"].iloc[0])

        # ── backtest de projeções: cobertura de intervalos ──────────
        cov_table = None
        if backtest is not None:
            try:
                cov_table = _coverage_table(backtest)
            except Exception as exc:  # noqa: BLE001
                mlflow.set_tag("backtest_error", str(exc)[:200])
            if cov_table is not None and len(cov_table):
                tot = cov_table[cov_table["passo"] == "todos"]
                tot = tot.iloc[0] if len(tot) else cov_table.iloc[-1]
                _log_metric_safe(mlflow, "cobertura_intervalo", tot.get("cobertura"))
                _log_metric_safe(mlflow, "cobertura_nominal", tot.get("nominal"))
                _log_metric_safe(mlflow, "kupiec_stat", tot.get("kupiec_stat"))
                _log_metric_safe(mlflow, "kupiec_pvalue", tot.get("kupiec_pvalue"))
                h1 = cov_table[cov_table["passo"] == 1]
                if len(h1):  # independência das violações no 1º passo (cadeia limpa)
                    _log_metric_safe(mlflow, "christoffersen_pvalue_h1",
                                     h1.iloc[0].get("christoffersen_pvalue"))

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
            if cov_table is not None and len(cov_table):
                cov_table.to_csv(os.path.join(tmp, "backtest_cobertura.csv"), index=False)
            mlflow.log_artifacts(tmp, artifact_path="tables")
        except Exception as exc:  # noqa: BLE001
            mlflow.set_tag("tables_error", str(exc)[:200])

        # ── artefatos: figuras (best-effort, dependem de matplotlib) ─
        try:
            _log_figures(mlflow, fit, series, projection, tmp)
        except Exception as exc:  # noqa: BLE001
            mlflow.set_tag("figures_error", str(exc)[:200])

        # ── artefato: relatório HTML em abas (best-effort) ──────────
        try:
            _log_satellite_report(mlflow, run, fit, projection, search, diag, tmp)
        except Exception as exc:  # noqa: BLE001
            mlflow.set_tag("report_error", str(exc)[:200])

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
