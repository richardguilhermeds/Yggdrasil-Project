"""
yggdrasil.credit_risk.econometric
=================================
**Modelos econométricos (satélite / macro) de PD, LGD e CCF** — ligam as séries
temporais agregadas dos parâmetros de risco às variáveis macroeconômicas
(desemprego, renda, juros, câmbio, inadimplência) e respondem à pergunta central:
*dado um cenário macro, quanto valem a PD, a LGD e o CCF deste segmento nos
próximos trimestres?* Essa projeção condicional alimenta os quatro usos do guia:
o **forward-looking do ECL** (IFRS 9 / Resolução CMN 4.966), os **testes de
estresse**, o **capital econômico** (a ligação fator-macro de
:mod:`yggdrasil.credit_risk.capital`) e o **planejamento** de provisão e apetite.

É o eixo **temporal** — complementar ao eixo transversal da escoragem
(:mod:`yggdrasil.credit_risk.model`/:mod:`~yggdrasil.credit_risk.tree`): o modelo
transversal ordena o risco; o satélite **desloca o nível** da curva conforme o
ciclo.

Arquitetura (Tabela 2 do guia)
------------------------------
Contrato e dados sintéticos
    :class:`RiskSeries`, :func:`simulate_pd_series`/:func:`simulate_lgd_series`/
    :func:`simulate_ccf_series`, :func:`make_reference_study`.
Transformações (:mod:`.transforms`)
    :func:`logit`/:func:`probit`, :func:`vasicek_z` (fator ``Z``), defasagens e
    dummies sazonais/evento/quebra.
Diagnóstico (:mod:`.diagnostics`)
    ADF/KPSS/PP, Ljung-Box, Breusch-Godfrey/Pagan/White, Jarque-Bera, ARCH-LM,
    VIF, Chow/Quandt-Andrews, CUSUM — saída tabular padronizada.
Modelos com interface comum (``fit``/``predict``/``project``/``diagnostics``)
    :class:`ARDL` (principal), :class:`ARIMA` (benchmark), :class:`VasicekZ`
    (ponte com capital), :class:`BetaRegression`/:class:`FractionalLogit`
    (LGD/CCF), :class:`RandomWalk`/:class:`HistoricalMean`/:class:`SeasonalNaive`
    (ingênuos), :class:`VARModel`/:class:`VECMModel` (equilíbrio de longo prazo),
    :class:`PanelSatellite` (muitos segmentos).
Seleção (:mod:`.selection`)
    :func:`make_grid`, :func:`search`, :func:`walk_forward`,
    :func:`diebold_mariano`, :func:`compare` — filtros de sinal/VIF e ranking
    champion-challenger.
Cenários (:mod:`.scenarios`)
    :class:`Scenario`/:class:`ScenarioSet`, :func:`project`,
    :func:`ecl_projection`, :func:`standard_scenarios`.
Pipeline declarativo (:mod:`.config`)
    :class:`StudyConfig`, :func:`run_study` — as "cinco chamadas" do guia.

As visualizações (:mod:`.report`, matplotlib) e o registro no MLflow
(:mod:`.tracking`) são carregados **sob demanda**. Este subpacote requer
``statsmodels`` e ``arch`` (extra ``econometric``); o restante de
``yggdrasil.credit_risk`` não os exige.

Uso típico (o estudo em poucas chamadas)::

    from yggdrasil.credit_risk.econometric import (
        make_reference_study, StudyConfig, run_study)

    est = make_reference_study()
    cfg = StudyConfig(kind="pd", candidates=["desemprego", "renda", "juros"],
                      expected_signs={"desemprego": 1, "renda": -1, "juros": 1})
    r = run_study(cfg, est.pd.series, est.macro)
    r.summary(); r.projection.mean_frame()      # ranking → projeção em uma passada
"""
from __future__ import annotations

# --- contrato e dados sintéticos ---------------------------------------
from .series import (
    RISK_KINDS,
    ReferenceStudy,
    RiskSeries,
    SyntheticSeries,
    make_reference_study,
    simulate_ccf_series,
    simulate_lgd_series,
    simulate_macro,
    simulate_pd_series,
)

# --- transformações ----------------------------------------------------
from . import transforms
from .transforms import (
    default_rate_from_z,
    event_dummies,
    inv_logit,
    inv_probit,
    logit,
    probit,
    seasonal_dummies,
    step_dummy,
    vasicek_z,
)

# --- diagnóstico -------------------------------------------------------
from . import diagnostics
from .diagnostics import residual_report, stationarity_report, vif

# --- interface e resultados --------------------------------------------
from .base import FitResult, Projection, SatelliteModel, Specification

# --- modelos -----------------------------------------------------------
from .ardl import ARDL
from .arima import ARIMA
from .vasicek import VasicekZ
from .fractional import BetaRegression, FractionalLogit, decompose_lgd
from .benchmarks import HistoricalMean, RandomWalk, SeasonalNaive
from .var_vecm import VARModel, VECMModel, engle_granger, johansen_test
from .panel import PanelSatellite

# --- seleção -----------------------------------------------------------
from . import selection
from .selection import SearchResult, compare, diebold_mariano, make_grid, search, walk_forward

# --- cenários ----------------------------------------------------------
from . import scenarios
from .scenarios import (
    Scenario,
    ScenarioSet,
    ecl_projection,
    project,
    shock_scenarios,
    standard_scenarios,
)

# --- pipeline declarativo ----------------------------------------------
from .config import MODEL_REGISTRY, StudyConfig, StudyResult, run_study

__all__ = [
    # contrato e dados sintéticos
    "RiskSeries", "SyntheticSeries", "ReferenceStudy", "RISK_KINDS",
    "simulate_pd_series", "simulate_lgd_series", "simulate_ccf_series",
    "simulate_macro", "make_reference_study",
    # transformações
    "transforms", "logit", "inv_logit", "probit", "inv_probit",
    "vasicek_z", "default_rate_from_z", "seasonal_dummies", "event_dummies", "step_dummy",
    # diagnóstico
    "diagnostics", "stationarity_report", "residual_report", "vif",
    # interface e resultados
    "SatelliteModel", "Specification", "FitResult", "Projection",
    # modelos
    "ARDL", "ARIMA", "VasicekZ", "BetaRegression", "FractionalLogit", "decompose_lgd",
    "RandomWalk", "HistoricalMean", "SeasonalNaive",
    "VARModel", "VECMModel", "johansen_test", "engle_granger", "PanelSatellite",
    # seleção
    "selection", "make_grid", "search", "walk_forward", "diebold_mariano", "compare",
    "SearchResult",
    # cenários
    "scenarios", "Scenario", "ScenarioSet", "project", "ecl_projection",
    "shock_scenarios", "standard_scenarios",
    # pipeline
    "StudyConfig", "StudyResult", "run_study", "MODEL_REGISTRY",
    # carregados sob demanda
    "report", "tracking", "log_satellite_run",
]


def __getattr__(name):
    # Visualizações (matplotlib) e MLflow carregados só quando pedidos.
    import importlib

    if name in ("report", "tracking"):
        return importlib.import_module(f"{__name__}.{name}")
    if name == "log_satellite_run":
        return importlib.import_module(f"{__name__}.tracking").log_satellite_run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
