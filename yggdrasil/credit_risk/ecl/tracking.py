"""
Registro do ECL no MLflow (carregado sob demanda)
=================================================
Versiona no MLflow o que a governança precisa reproduzir: a **curva de PD
lifetime**, a **tabela ELBE**, a **base de CCF** e o **ECL** da carteira —
parâmetros, métricas e artefatos.

Segue o mesmo padrão de :mod:`yggdrasil.credit_risk.capital.tracking` e de
:mod:`yggdrasil.tracking.mlflow_logger`:

* ``mlflow`` é importado **dentro** da função, de modo que o subpacote continua
  importável em ambientes sem MLflow;
* parâmetros por ``log_params``, métricas por ``log_metric`` (só valores
  finitos), tags por ``set_tags``, artefatos escritos em diretório temporário e
  enviados com ``log_artifact``;
* artefatos são *best-effort*: qualquer falha vira **tag de erro** em vez de
  derrubar o *run* — o número, já logado, é o que importa.

A rastreabilidade aqui não é conforto: a provisão sob a Resolução CMN 4.966/2021
e o IFRS 9 é auditada, e a auditoria precisa reconstruir o número a partir dos
insumos versionados — qual curva, qual calibração, qual cenário, qual data.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # apenas type hints; sem custo em runtime
    from .ccf import CCFDataset
    from .ecl import ECLResult
    from .elbe import ELBETable
    from .lifetime_pd import LifetimePD

DEFAULT_EXPERIMENT = "/Shared/Yggdrasil/ecl"

BASE_TAGS = {"framework": "yggdrasil-ml", "model_type": "ecl"}


# ======================================================================
# Auxiliares
# ======================================================================
def _log_metric_safe(mlflow, nome: str, valor) -> None:
    """Loga a métrica só se for número finito (evita NaN/inf no MLflow)."""
    if valor is None:
        return
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return
    if np.isfinite(v):
        mlflow.log_metric(nome, v)


def _tmp(prefixo: str, artifacts_dir: Optional[str]) -> str:
    caminho = artifacts_dir or tempfile.mkdtemp(prefix=prefixo)
    os.makedirs(caminho, exist_ok=True)
    return caminho


def _log_csv(mlflow, frame: pd.DataFrame, tmp: str, nome: str,
             artifact_path: Optional[str] = None, index: bool = False) -> None:
    """Grava um CSV e anexa ao *run* (best-effort — falha vira tag)."""
    try:
        caminho = os.path.join(tmp, nome)
        frame.to_csv(caminho, index=index, encoding="utf-8")
        mlflow.log_artifact(caminho, artifact_path=artifact_path)
    except Exception as e:  # pragma: no cover - depende do ambiente
        mlflow.set_tag(f"erro_artefato_{nome}", f"{type(e).__name__}: {e}")


def _log_figure(mlflow, fabrica, tmp: str, nome: str) -> None:
    """Gera uma figura pela ``fabrica`` e anexa como PNG (best-effort)."""
    try:
        import matplotlib.pyplot as plt

        fig = fabrica()
        caminho = os.path.join(tmp, nome)
        fig.savefig(caminho, dpi=110, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(caminho, artifact_path="figures")
    except Exception as e:  # pragma: no cover - matplotlib é opcional aqui
        mlflow.set_tag(f"erro_figura_{nome}", f"{type(e).__name__}: {e}")


def _log_json(mlflow, obj, tmp: str, nome: str) -> None:
    try:
        caminho = os.path.join(tmp, nome)
        with open(caminho, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
        mlflow.log_artifact(caminho)
    except Exception as e:  # pragma: no cover
        mlflow.set_tag(f"erro_artefato_{nome}", f"{type(e).__name__}: {e}")


def _abrir_run(mlflow, experiment: Optional[str], run_name: Optional[str]):
    if experiment:                      # explícito vence; senão, experimento da sessão
        mlflow.set_experiment(experiment)
    return mlflow.start_run(run_name=run_name)


# ======================================================================
# PD lifetime
# ======================================================================
def log_lifetime_pd(
    model: "LifetimePD",
    *,
    backtest: Optional[pd.DataFrame] = None,
    params: Optional[dict] = None,
    tags: Optional[dict] = None,
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
) -> str:
    """Registra a curva de PD *lifetime* e devolve o ``run_id``.

    Loga o método e o horizonte como parâmetros; PD de 12 meses e *lifetime* de
    cada grupo como métricas; e como artefatos a tabela das curvas
    (``curvas.csv``), a serialização completa do modelo (``lifetime_pd.json``,
    que reconstrói o objeto via
    :meth:`~yggdrasil.credit_risk.ecl.lifetime_pd.LifetimePD.from_json`), o
    *backtest* e as figuras."""
    import mlflow  # import tardio

    tmp = _tmp("yggdrasil_ecl_pd_", artifacts_dir)
    with _abrir_run(mlflow, experiment, run_name) as run:
        p = dict(params or {})
        p.setdefault("metodo", model.method_)
        p.setdefault("horizonte", model.horizon)
        p.setdefault("freq", model.freq)
        p.setdefault("by", model.by)
        p.setdefault("n_curvas", len(model.curves_))
        p.setdefault("features", ",".join(model.features) or "—")
        p.setdefault("ajustes", ",".join(a["tipo"] for a in model.adjustments_) or "—")
        mlflow.log_params(p)

        resumo = model.summary()
        for _, linha in resumo.iterrows():
            sufixo = "" if linha["grupo"] == model.GLOBAL else f"_{linha['grupo']}"
            _log_metric_safe(mlflow, f"pd_12m{sufixo}", linha["pd_12m"])
            _log_metric_safe(mlflow, f"pd_lifetime{sufixo}", linha["pd_lifetime"])

        if backtest is not None and len(backtest):
            _log_metric_safe(mlflow, "backtest_erro_absoluto_medio",
                             backtest["erro_absoluto"].abs().mean())
            _log_metric_safe(mlflow, "backtest_pct_dentro_do_ic",
                             backtest["dentro_do_ic"].mean())
            _log_csv(mlflow, backtest, tmp, "backtest.csv")
            _log_figure(mlflow, lambda: __import__(
                "yggdrasil.credit_risk.ecl.report", fromlist=["x"]
            ).plot_backtest(backtest), tmp, "backtest.png")

        mlflow.set_tags({**BASE_TAGS, "risk_parameter": "pd_lifetime", **(tags or {})})

        _log_csv(mlflow, resumo, tmp, "resumo_curvas.csv")
        _log_csv(mlflow, model.frame("cumulative"), tmp, "curvas_acumuladas.csv", index=True)
        _log_csv(mlflow, model.frame("hazard"), tmp, "curvas_hazard.csv", index=True)
        _log_json(mlflow, model.to_dict(), tmp, "lifetime_pd.json")
        _log_figure(mlflow, lambda: model.plot("cumulative"), tmp, "pd_acumulada.png")
        _log_figure(mlflow, lambda: model.plot("hazard"), tmp, "pd_condicional.png")
        return run.info.run_id


# ======================================================================
# ELBE
# ======================================================================
def log_elbe(
    table: "ELBETable",
    *,
    params: Optional[dict] = None,
    tags: Optional[dict] = None,
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
) -> str:
    """Registra a tabela ELBE e devolve o ``run_id``."""
    import mlflow

    tmp = _tmp("yggdrasil_ecl_elbe_", artifacts_dir)
    with _abrir_run(mlflow, experiment, run_name) as run:
        p = dict(params or {})
        for chave in ("kind", "unit", "ultimate", "addon", "floor", "discount_rate",
                      "monotonic", "cohort", "tol"):
            p.setdefault(chave, table.meta.get(chave))
        p.setdefault("n_contratos", table.meta.get("n_contratos"))
        mlflow.log_params(p)

        _log_metric_safe(mlflow, "lgd_ciclo_completo", table.lgd)
        _log_metric_safe(mlflow, "recuperacao_no_workout", table.ultimate_recovery)
        _log_metric_safe(mlflow, "horizonte_workout", table.workout)
        _log_metric_safe(mlflow, "elbe_mes_0", float(table.frame["elbe"].iloc[0]))
        _log_metric_safe(mlflow, "elbe_no_workout", table.elbe_at(table.workout))
        _log_metric_safe(mlflow, "exposicao_inicial",
                         float(table.frame["exposicao_inicial"].iloc[0]))

        mlflow.set_tags({**BASE_TAGS, "risk_parameter": "elbe", **(tags or {})})

        _log_csv(mlflow, table.frame, tmp, "elbe.csv", index=True)
        _log_csv(mlflow, table.summary(), tmp, "elbe_resumo.csv")
        _log_json(mlflow, {k: v for k, v in table.to_dict().items() if k != "frame"},
                  tmp, "elbe_meta.json")
        _log_figure(mlflow, table.plot, tmp, "elbe.png")
        return run.info.run_id


# ======================================================================
# CCF
# ======================================================================
def log_ccf(
    data: "CCFDataset",
    *,
    pooled: Optional[pd.DataFrame] = None,
    backtest: Optional[pd.DataFrame] = None,
    params: Optional[dict] = None,
    tags: Optional[dict] = None,
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
) -> str:
    """Registra a base de CCF, o agrupamento e o *backtest* de EAD."""
    import mlflow

    tmp = _tmp("yggdrasil_ecl_ccf_", artifacts_dir)
    with _abrir_run(mlflow, experiment, run_name) as run:
        p = dict(params or {})
        p.setdefault("method", data.method)
        p.setdefault("horizon", data.horizon)
        p.setdefault("measure", data.measure)
        for chave, valor in data.meta.items():
            p.setdefault(chave, valor)
        mlflow.log_params(p)

        resumo = data.summary().iloc[0]
        for coluna in ("n_observacoes", "n_contratos", "media", "mediana", "desvio",
                       "p10", "p90", "massa_em_0", "massa_em_1", "n_excluidos"):
            _log_metric_safe(mlflow, f"ccf_{coluna}", resumo[coluna])
        _log_metric_safe(mlflow, "ccf_downturn_p90", data.downturn(0.9))
        if backtest is not None and len(backtest):
            global_ = backtest[backtest["grupo"] == "__global__"]
            if len(global_):
                _log_metric_safe(mlflow, "ead_vies_relativo",
                                 float(global_["vies_relativo"].iloc[0]))
                _log_metric_safe(mlflow, "ead_erro_absoluto_relativo",
                                 float(global_["erro_absoluto_relativo"].iloc[0]))
            _log_csv(mlflow, backtest, tmp, "backtest_ead.csv")

        mlflow.set_tags({**BASE_TAGS, "risk_parameter": "ccf", **(tags or {})})

        _log_csv(mlflow, data.summary(), tmp, "ccf_resumo.csv")
        _log_csv(mlflow, data.excluded_frame(), tmp, "ccf_exclusoes.csv")
        _log_csv(mlflow, data.distribution(), tmp, "ccf_distribuicao.csv")
        if pooled is not None:
            _log_csv(mlflow, pooled, tmp, "ccf_agrupado.csv")
        _log_figure(mlflow, data.plot, tmp, "ccf_distribuicao.png")
        return run.info.run_id


# ======================================================================
# ECL da carteira
# ======================================================================
def log_ecl_run(
    result: "ECLResult",
    *,
    model: Optional["LifetimePD"] = None,
    scenarios: Optional[dict] = None,
    by: Optional[list] = None,
    params: Optional[dict] = None,
    tags: Optional[dict] = None,
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
) -> str:
    """Registra o **ECL da carteira** e devolve o ``run_id``.

    Loga o total, a taxa de provisão e a abertura por estágio; e, quando dados,
    o modelo de PD usado, os cenários ponderados e as aberturas pedidas em
    ``by`` (ex.: ``['produto']``, ``['safra']``). A carteira **inteira não é
    logada** — só as agregações; a granularidade de contrato é dado sensível e
    pesado, e o que a auditoria reconstrói é a conta, não a base."""
    import mlflow

    tmp = _tmp("yggdrasil_ecl_", artifacts_dir)
    with _abrir_run(mlflow, experiment, run_name) as run:
        p = dict(params or {})
        p.setdefault("horizonte", result.horizon)
        p.setdefault("n_contratos", len(result.frame))
        p.setdefault("stage_rules", json.dumps({str(k): v for k, v in result.stage_rules.items()}))
        for chave in ("periods_per_year", "discount_rate", "lgd", "ead"):
            p.setdefault(chave, result.meta.get(chave))
        if model is not None:
            p.setdefault("pd_metodo", model.method_)
            p.setdefault("pd_horizonte", model.horizon)
        mlflow.log_params(p)

        _log_metric_safe(mlflow, "ecl_total", result.total)
        _log_metric_safe(mlflow, "exposicao_total", float(result.frame["exposicao"].sum()))
        _log_metric_safe(mlflow, "taxa_provisao", result.coverage)
        _log_metric_safe(mlflow, "ecl_12m_total", float(result.frame["ecl_12m"].sum()))
        _log_metric_safe(mlflow, "ecl_lifetime_total", float(result.frame["ecl_lifetime"].sum()))
        _log_metric_safe(mlflow, "pd_12m_media", float(result.frame["pd_12m"].mean()))
        for _, linha in result.summary().iterrows():
            _log_metric_safe(mlflow, f"ecl_estagio_{linha['estagio']}", linha["ecl"])
            _log_metric_safe(mlflow, f"cobertura_estagio_{linha['estagio']}",
                             linha["taxa_provisao"])

        if scenarios is not None:
            _log_metric_safe(mlflow, "ecl_ponderado_cenarios", scenarios.get("ponderado"))
            por_cenario = scenarios.get("por_cenario")
            if por_cenario is not None:
                for _, linha in por_cenario.iterrows():
                    _log_metric_safe(mlflow, f"ecl_cenario_{linha['cenario']}", linha["ecl"])
                _log_csv(mlflow, por_cenario, tmp, "ecl_cenarios.csv")
                _log_figure(mlflow, lambda: __import__(
                    "yggdrasil.credit_risk.ecl.report", fromlist=["x"]
                ).plot_ecl_scenarios(scenarios), tmp, "ecl_cenarios.png")

        mlflow.set_tags({**BASE_TAGS, "risk_parameter": "ecl", **(tags or {})})

        _log_csv(mlflow, result.summary(), tmp, "ecl_por_estagio.csv")
        _log_json(mlflow, result.to_dict(), tmp, "ecl_resumo.json")
        for coluna in (by or []):
            if coluna in result.frame.columns:
                _log_csv(mlflow, result.by(coluna), tmp, f"ecl_por_{coluna}.csv")
        if model is not None:
            _log_json(mlflow, model.to_dict(), tmp, "lifetime_pd.json")
        return run.info.run_id


__all__ = ["log_lifetime_pd", "log_elbe", "log_ccf", "log_ecl_run", "DEFAULT_EXPERIMENT"]
