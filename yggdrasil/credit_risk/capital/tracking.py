"""Registro de capital econômico no MLflow.

Versiona um *run* de capital econômico — parâmetros, métricas e artefatos — no
MLflow, atendendo às Seções 7/8 do guia (versionamento de parâmetros e
resultados em MLflow/Unity Catalog, manual do modelo, inventário de modelos e
monitoramento). A rastreabilidade do capital econômico é exigência do processo
de **ICAAP** (Resolução CMN 4.557/2017): a autoridade e a auditoria interna
precisam reproduzir o número de capital a partir dos insumos versionados.

O módulo segue o padrão de :mod:`yggdrasil.tracking.mlflow_logger`:

* ``mlflow`` é importado **tardiamente**, dentro da função, de modo que o
  pacote ``capital`` continua importável em ambientes sem MLflow (por exemplo,
  no cálculo puro de capital, sem *tracking*).
* parâmetros via ``log_params``, métricas via ``log_metric``, tags via
  ``set_tags`` e artefatos escritos num diretório temporário e enviados com
  ``log_artifacts``.
* artefatos são *best-effort*: qualquer falha vira uma ``tag`` de erro em vez de
  derrubar o *run* (o número de capital, já logado, é o que importa).

O elo com os motores está em :mod:`.monte_carlo` (``SimulationResult``),
:mod:`.asrf` (``AsrfResult``), :mod:`.allocation` e :mod:`.measures`
(``LossDistribution``).
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # apenas para type hints; sem custo em runtime
    from .asrf import AsrfResult
    from .measures import LossDistribution
    from .monte_carlo import SimulationResult
    from .portfolio import Portfolio

DEFAULT_EXPERIMENT = "/Shared/Yggdrasil/capital_economico"


# ======================================================================
# Auxiliares de logging
# ======================================================================
def _log_metric_safe(mlflow, nome: str, valor) -> None:
    """Loga uma métrica só se for um número finito (evita NaN/inf no MLflow)."""
    if valor is None:
        return
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return
    if not np.isfinite(v):
        return
    mlflow.log_metric(nome, v)


def _log_capital_figures(mlflow, result, allocation, tmp: str) -> None:
    """Gera as figuras de report (*best-effort*) e as anexa ao *run*.

    Importa :mod:`.report` tardiamente: o módulo de relatório é opcional e
    depende de ``matplotlib``. Qualquer falha (módulo ausente, função ausente,
    erro de plotagem) é registrada como *tag* e não interrompe o *run*.
    """
    from . import report  # import tardio: matplotlib só entra aqui
    import matplotlib.pyplot as plt

    # As funções de report RETORNAM uma Figure (não gravam em disco): geramos a
    # figura, salvamos com savefig e fechamos.
    def _save(fig, path: str) -> None:
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)

    # Distribuição de perdas (Figura 1 do guia).
    plot_dist = getattr(report, "plot_loss_distribution", None)
    if callable(plot_dist):
        fig_dist = os.path.join(tmp, "loss_distribution.png")
        _save(plot_dist(result, q=result.q), fig_dist)
        if os.path.exists(fig_dist):
            mlflow.log_artifact(fig_dist, artifact_path="figures")

    # Alocação de capital por segmento (só se houver alocação disponível).
    if allocation is not None:
        plot_alloc = getattr(report, "plot_allocation", None)
        if callable(plot_alloc):
            fig_alloc = os.path.join(tmp, "allocation.png")
            _save(plot_alloc(allocation), fig_alloc)
            if os.path.exists(fig_alloc):
                mlflow.log_artifact(fig_alloc, artifact_path="figures")


#: colunas da aba Resumo do relatório (comparação dos runs do experimento)
_REPORT_METRICS_SPEC = [
    ("params.method", "método"),
    ("metrics.EL", "EL"),
    ("metrics.VaR", "VaR"),
    ("metrics.ES", "ES"),
    ("metrics.CE_var", "CE (VaR)"),
    ("metrics.CE_es", "CE (ES)"),
]


def _log_capital_report(mlflow, run, portfolio, result, allocation, asrf,
                        benchmark, div, tmp: str) -> None:
    """Monta e loga o relatório HTML em abas (``report.html``) do *run*.

    Import tardio de :mod:`.._mlflow_report` (padrão do pacote). A aba Resumo
    compara EL/VaR/ES/CE entre os runs do experimento; as abas de Alocação de
    Euler e de Validação/benchmark só entram quando os dados existem no run.
    """
    from .._mlflow_report import build_tabbed_report_html, runs_comparison_df

    resumo = runs_comparison_df(mlflow, run.info.experiment_id, run.info.run_id,
                                metrics_spec=_REPORT_METRICS_SPEC)
    blocos_resumo = [
        ("Comparação dos runs do experimento — EL · VaR · ES · CE", resumo),
        ("Distribuição de perdas do run atual", result.summary()),
        ("Carteira por segmento", portfolio.summary()),
    ]
    tabs = [("Resumo · runs", blocos_resumo)]

    if allocation is not None and len(allocation):
        tabs.append(("Alocação de Euler",
                     [("Capital alocado por segmento (contribuição à cauda)",
                       allocation)]))

    blocos_val = []
    if asrf is not None or benchmark is not None:
        linhas = [{"medida": "Monte Carlo (run atual)",
                   "CE": result.economic_capital(metric="var"),
                   "VaR": result.var()}]
        if asrf is not None:
            linhas.append({"medida": "ASRF (benchmark analítico)",
                           "CE": asrf.economic_capital,
                           "VaR": asrf.value_at_risk})
        if benchmark is not None:
            linhas.append({"medida": "distribuição de referência",
                           "CE": benchmark.economic_capital(result.q, metric=result.metric),
                           "VaR": benchmark.var(result.q)})
        blocos_val.append(("Benchmark de capital — modelo × referência",
                           pd.DataFrame(linhas)))
    if div is not None:
        blocos_val.append(("Benefício de diversificação", pd.DataFrame([div])))
    if blocos_val:
        tabs.append(("Validação · benchmark", blocos_val))

    nums = "①②③④⑤"                                   # numeração conforme as abas presentes
    tabs = [(f"{nums[i]} {nome}", blocos) for i, (nome, blocos) in enumerate(tabs)]

    titulo = f"Capital econômico — {portfolio.name}"
    sub = (f"q={result.q} · {result.n_scenarios:,} cenários · seed={result.seed}"
           f" · run {run.info.run_id[:8]}").replace(",", ".")
    html_doc = build_tabbed_report_html(titulo, sub, tabs, highlight=("", "➤"))
    path = os.path.join(tmp, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    mlflow.log_artifact(path)


# ======================================================================
# API pública
# ======================================================================
def log_capital_run(
    portfolio: "Portfolio",
    result: "SimulationResult",
    *,
    allocation: Optional[pd.DataFrame] = None,
    asrf: Optional["AsrfResult"] = None,
    benchmark: Optional["LossDistribution"] = None,
    params: Optional[dict] = None,
    tags: Optional[dict] = None,
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
) -> str:
    """Registra um *run* de capital econômico no MLflow e retorna o ``run_id``.

    Loga os insumos e resultados de uma simulação de Monte Carlo de capital
    econômico, opcionalmente enriquecidos com a alocação de Euler, o *benchmark*
    ASRF e uma distribuição de referência. Além das tabelas e figuras, loga um
    relatório HTML autocontido em abas (``report.html``): Resumo (EL/VaR/ES/CE
    comparando os runs do experimento) · Alocação de Euler · Validação/benchmark
    — as duas últimas apenas quando os dados existem no run.

    Parameters
    ----------
    portfolio:
        A carteira simulada (:class:`.portfolio.Portfolio`). Seu ``summary()`` é
        salvo como artefato e ``n_segments``/``n_factors`` viram parâmetros.
    result:
        A :class:`.monte_carlo.SimulationResult` do *run* principal. Fonte de
        EL, VaR, ES, CE e (quando ``segment_losses`` foi armazenado) do
        benefício de diversificação.
    allocation:
        Tabela de alocação de Euler (saída de
        :func:`.allocation.euler_allocation` ou ``result.allocate()``). Se dada,
        é salva como ``allocation.csv`` e vira a figura ``allocation.png``.
    asrf:
        :class:`.asrf.AsrfResult` do *benchmark* analítico. Se dado, loga
        ``CE_asrf`` e ``VaR_asrf`` para comparação com o modelo multifatorial.
    benchmark:
        :class:`.measures.LossDistribution` de referência (por exemplo, uma
        distribuição analítica de sanidade). Se dada, loga ``CE_benchmark``.
    params:
        Parâmetros extras do usuário (ex.: ``rho_default``, ``stochastic_lgd``),
        mesclados aos parâmetros-base.
    tags:
        Tags extras do usuário, mescladas às tags-base do framework.
    experiment:
        Nome do experimento. Se *truthy*, chama ``set_experiment``; senão usa o
        experimento ativo da sessão (mesmo comportamento do ``mlflow_logger``).
        Passe :data:`DEFAULT_EXPERIMENT` explicitamente para forçar o padrão.
    run_name:
        Nome do *run* (opcional).
    artifacts_dir:
        Diretório para os artefatos. Se ``None``, cria um diretório temporário.

    Returns
    -------
    str
        O ``run_id`` do *run* registrado.
    """
    import mlflow  # import tardio: pacote importável sem MLflow instalado

    params = dict(params or {})
    tmp = artifacts_dir or tempfile.mkdtemp(prefix="yggdrasil_capital_")
    os.makedirs(tmp, exist_ok=True)

    if experiment:                                   # explícito vence; senão, sessão
        mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as run:
        # ── parâmetros ──────────────────────────────────────────────────
        params.setdefault("n_segments", portfolio.n_segments)
        params.setdefault("n_factors", portfolio.n_factors)
        params.setdefault("q", result.q)
        params.setdefault("n_scenarios", result.n_scenarios)
        params.setdefault("seed", result.seed)
        params.setdefault("method", "monte_carlo")
        mlflow.log_params(params)

        # ── métricas do run principal ───────────────────────────────────
        dist = result.distribution()
        _log_metric_safe(mlflow, "EL", dist.el)
        _log_metric_safe(mlflow, "VaR", result.var())
        _log_metric_safe(mlflow, "ES", result.es())
        _log_metric_safe(mlflow, "CE_var", result.economic_capital(metric="var"))
        _log_metric_safe(mlflow, "CE_es", result.economic_capital(metric="es"))

        # ── incerteza amostral (ruído de Monte Carlo) do VaR e do ES ────
        # Erro-padrão do quantil (estatísticas de ordem) e da média da cauda
        # (bootstrap) — insumo do diagnóstico de convergência do run. Em
        # distribuições ponderadas (analíticas) os métodos retornam NaN e o
        # _log_metric_safe simplesmente não loga.
        _log_metric_safe(mlflow, "VaR_se", dist.var_se(result.q))
        _log_metric_safe(mlflow, "ES_se", dist.es_se(result.q, seed=result.seed))

        # ── benefício de diversificação (se segment_losses disponível) ──
        try:
            div = result.diversification_benefit()
        except (ValueError, AttributeError):
            div = None
        if div is not None:
            _log_metric_safe(mlflow, "capital_isolado", div.get("capital_isolado"))
            _log_metric_safe(mlflow, "capital_integrado", div.get("capital_integrado"))
            _log_metric_safe(mlflow, "beneficio_diversificacao",
                             div.get("beneficio_diversificacao"))

        # ── benchmark ASRF (comparação analítica) ───────────────────────
        if asrf is not None:
            _log_metric_safe(mlflow, "CE_asrf", asrf.economic_capital)
            _log_metric_safe(mlflow, "VaR_asrf", asrf.value_at_risk)

        # ── benchmark: distribuição de referência ───────────────────────
        if benchmark is not None:
            _log_metric_safe(mlflow, "CE_benchmark",
                             benchmark.economic_capital(result.q, metric=result.metric))

        # ── tags ────────────────────────────────────────────────────────
        base_tags = {
            "framework": "yggdrasil-ml",
            "model_type": "economic_capital",
            "trained_by": "richard-guilherme",
        }
        base_tags.update(tags or {})
        mlflow.set_tags(base_tags)

        # ── artefatos: tabelas (best-effort) ────────────────────────────
        try:
            result.summary().to_csv(
                os.path.join(tmp, "distribution_summary.csv"), index=False)
            if allocation is not None:
                allocation.to_csv(os.path.join(tmp, "allocation.csv"), index=False)
            portfolio.summary().to_csv(
                os.path.join(tmp, "portfolio_summary.csv"), index=False)
            mlflow.log_artifacts(tmp, artifact_path="tables")
        except Exception as exc:  # noqa: BLE001 - artefatos são best-effort
            mlflow.set_tag("tables_error", str(exc)[:250])

        # ── artefatos: figuras (best-effort, dependem de matplotlib) ────
        try:
            _log_capital_figures(mlflow, result, allocation, tmp)
        except Exception as exc:  # noqa: BLE001 - figuras são best-effort
            mlflow.set_tag("figures_error", str(exc)[:250])

        # ── artefato: relatório HTML em abas (best-effort) ──────────────
        try:
            _log_capital_report(mlflow, run, portfolio, result, allocation,
                                asrf, benchmark, div, tmp)
        except Exception as exc:  # noqa: BLE001 - relatório é best-effort
            mlflow.set_tag("report_error", str(exc)[:250])

        return run.info.run_id


__all__ = ["log_capital_run", "DEFAULT_EXPERIMENT"]
