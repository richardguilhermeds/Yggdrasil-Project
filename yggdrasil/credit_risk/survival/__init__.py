"""
yggdrasil.credit_risk.survival
==============================
**Análise de sobrevivência para a PD lifetime**, com interface interativa.

O subpacote :mod:`~yggdrasil.credit_risk.ecl` já traz os motores da estrutura a
termo (curva de safra, Kaplan-Meier, *hazard* em tempo discreto, Markov) e a
fachada :class:`~yggdrasil.credit_risk.ecl.LifetimePD`. Este subpacote é a
**bancada de trabalho** em cima deles: as ferramentas que a análise de
sobrevivência usa para construir, escolher e defender uma curva, e a interface
(:class:`SurvivalUI`) que conduz o estudo aba a aba, no mesmo desenho das demais
UIs do ``credit_risk`` (abas, console, tema claro/escuro, configuração em JSON,
MLflow).

Os blocos
---------

**Tabela de vida** (:mod:`.lifetable`)
    :func:`life_table` (base em risco, censura, Kaplan-Meier com Greenwood,
    Nelson-Aalen), :func:`logrank_test`/:func:`pairwise_logrank` (as curvas dos
    segmentos diferem?), :func:`median_survival` e
    :func:`restricted_mean_survival` (a curva em unidades de tempo).

**Paramétrico e cauda** (:mod:`.parametric`)
    :class:`ParametricSurvival` (exponencial, Weibull, log-normal,
    log-logística, Gompertz por máxima verossimilhança em tempo discreto),
    :func:`fit_parametric` (ranking por AIC) e :func:`splice_curves` (a curva
    empírica até onde a base sustenta, a cauda paramétrica dali em diante).

**Validação** (:mod:`.validation`)
    :func:`backtest_curve` (prevista × KM observada com ``z`` por horizonte),
    :func:`discrimination_by_horizon` (AUC/Gini/KS do *default* em ``h``),
    :func:`concordance_index` (C-index de Harrell), :func:`calibration_by_decile`
    (Hosmer-Lemeshow) e :func:`ph_test` (riscos proporcionais por razão de
    verossimilhança).

**O estudo** (:mod:`.study`)
    :class:`SurvivalConfig` (a configuração serializável) e
    :func:`run_survival_study` (partição → curva → cauda → calibração/ciclo →
    validação numa passada), devolvendo :class:`SurvivalResult` com o
    :class:`LifetimePD` final.

**Painel de referência** (:mod:`.synthetic`)
    :func:`make_reference_panel`: carteira sintética com maturação, covariáveis
    e censura de processo gerador conhecido, para aprender a ferramenta e para
    os testes de recuperação de parâmetros.

**A interface** (:mod:`.ui`, carregada sob demanda)
    :class:`SurvivalUI`, sete abas: Painel · Kaplan-Meier · Hazard ·
    Paramétrico · Calibração & Ciclo · Validação · Exportar.

Uso típico::

    from yggdrasil.credit_risk.survival import SurvivalUI
    ui = SurvivalUI(df, origin_col="safra_origem", segment_col="produto",
                    features=["feat_score", "feat_ltv"])
    ui                                     # dentro do Jupyter/Databricks

    # sem interface: o estudo declarativo
    from yggdrasil.credit_risk.survival import SurvivalConfig, run_survival_study
    cfg = SurvivalConfig(origin_col="safra_origem", segment_col="produto", by="produto",
                         method="km", tail="parametric", distribution="weibull",
                         split="origin", split_value="2021-01-01")
    res = run_survival_study(df, cfg)
    res.model.apply(carteira, age_col="idade", term_col="prazo")

O núcleo roda em ``numpy``/``pandas``/``scipy``/``scikit-learn``; ``ipywidgets``
e ``matplotlib`` só entram quando a interface é construída.
"""
from __future__ import annotations

from .lifetable import (
    GLOBAL,
    LOGRANK_WEIGHTS,
    life_table,
    logrank_test,
    median_survival,
    pairwise_logrank,
    restricted_mean_survival,
    smooth_hazard,
)
from .parametric import (
    DISTRIBUTION_LABELS,
    DISTRIBUTIONS,
    ParametricSurvival,
    fit_parametric,
    junction_age,
    splice_curves,
)
from .study import (
    SPLIT_MODES,
    STUDY_METHODS,
    TAILS,
    SurvivalConfig,
    SurvivalResult,
    run_survival_study,
    split_panel,
)
from .synthetic import FEATURES, ReferencePanel, make_reference_panel
from .validation import (
    backtest_curve,
    calibration_by_decile,
    concordance_index,
    contract_outcomes,
    discrimination_by_horizon,
    model_concordance,
    ph_test,
)

__all__ = [
    # tabela de vida
    "life_table", "logrank_test", "pairwise_logrank", "median_survival",
    "restricted_mean_survival", "smooth_hazard", "LOGRANK_WEIGHTS", "GLOBAL",
    # paramétrico
    "ParametricSurvival", "fit_parametric", "splice_curves", "junction_age",
    "DISTRIBUTIONS", "DISTRIBUTION_LABELS",
    # validação
    "backtest_curve", "discrimination_by_horizon", "concordance_index", "model_concordance",
    "calibration_by_decile", "ph_test", "contract_outcomes",
    # estudo
    "SurvivalConfig", "SurvivalResult", "run_survival_study", "split_panel",
    "STUDY_METHODS", "SPLIT_MODES", "TAILS",
    # referência
    "make_reference_panel", "ReferencePanel", "FEATURES",
    # sob demanda
    "ui", "SurvivalUI",
]


def __getattr__(name):
    # A interface (ipywidgets/matplotlib) só entra quando pedida.
    import importlib

    if name == "ui":
        return importlib.import_module(f"{__name__}.ui")
    if name == "SurvivalUI":
        return importlib.import_module(f"{__name__}.ui").SurvivalUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
