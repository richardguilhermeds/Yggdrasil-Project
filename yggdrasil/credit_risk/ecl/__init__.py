"""
yggdrasil.credit_risk.ecl
=========================
**Os parâmetros de risco da perda esperada** — PD *lifetime*, ELBE e CCF/EAD — e
a conta que os junta na provisão (ECL) de IFRS 9 / Resolução CMN 4.966/2021.

É o eixo que faltava no pacote. Os outros três já existiam:

* :mod:`~yggdrasil.credit_risk.tree` e :mod:`~yggdrasil.credit_risk.model`
  **ordenam** o risco entre clientes (o eixo transversal, de 12 meses);
* :mod:`~yggdrasil.credit_risk.econometric` **desloca o nível** conforme o ciclo
  (o eixo temporal/macro);
* :mod:`~yggdrasil.credit_risk.capital` mede a perda **inesperada**.

Este subpacote responde à outra metade da pergunta: *quando* a perda esperada
acontece ao longo da vida do contrato, *quanto* sobra a perder no que já quebrou,
e *sobre qual exposição* ela incide.

Os três blocos
--------------

**PD lifetime** (:mod:`.lifetime_pd`) — a estrutura a termo da PD.
    :class:`LifetimePD` é a fachada: uma classe, cinco motores escolhidos por
    ``method=`` (``'constant'``, ``'vintage'``, ``'km'``, ``'hazard'``,
    ``'markov'``), todos devolvendo a mesma :class:`PDCurve`. A curva aceita e
    entrega as **quatro** representações (condicional, marginal, acumulada,
    sobrevivência) por identidades exatas — a fonte nº 1 de erro de um período em
    projeto de ECL. :meth:`LifetimePD.calibrate_to` cola o nível da curva na PD
    de 12 meses do modelo transversal; :meth:`LifetimePD.condition` desloca ao
    ciclo por Vasicek.

**ELBE** (:mod:`.elbe`) — a perda dos contratos **já em *default***.
    :func:`elbe_table` monta a tabela a partir de duas coisas: a **exposição
    inicial** e as **colunas de LGD por mês em *default***. Entrega a curva de
    recuperação (encadeada pela recuperação marginal, para a coorte variável não
    distorcer a comparação entre horizontes), o horizonte de *workout*, a ELBE
    ``(1 − r̄(T*)) / (1 − r̄(t))`` e a LGD *in default*.

**CCF / EAD** (:mod:`.ccf`) — a exposição no *default* dos rotativos.
    :func:`reference_dataset` monta a base pelos três desenhos da literatura
    (coorte, horizonte fixo, horizonte variável), calcula as quatro medidas
    ex-post (CCF/LEQ, EADF, AUF, EAD direto), limpa o dado e expõe a
    bimodalidade; :func:`pooled_ccf` agrupa e :func:`backtest_ead` testa o
    parâmetro contra a exposição realizada.

**A montagem** (:mod:`.ecl`) — ``ECL = Σ PD_marginal · LGD · EAD · desconto``,
com o corte de horizonte vindo do estágio (1 → 12 meses, 2 → *lifetime*, 3 →
ELBE) e o *forward-looking* por :func:`ecl_scenarios`. A regra de transferência
entre estágios (SICR) **não** está aqui: é política da instituição, e o módulo
recebe a coluna pronta.

Núcleo em ``numpy``/``pandas``/``scipy``/``scikit-learn`` — nada aqui exige o
extra ``[econometric]``. As visualizações (:mod:`.report`, matplotlib) e o
registro no MLflow (:mod:`.tracking`) são carregados **sob demanda**.

Uso típico::

    from yggdrasil.credit_risk.ecl import ContractPanel, LifetimePD, elbe_table, ecl_table

    painel = ContractPanel(df, origin_col="safra_origem", segment_col="produto")
    pd_lt = LifetimePD(method="vintage", horizon=60).fit(painel, by="produto")
    pd_lt = pd_lt.calibrate_to({"cartao": 0.078, "consignado": 0.021})   # nível do scorecard

    elbe = elbe_table(defaults, exposure_col="exposicao_inicial", lgd_prefix="lgd_m")
    carteira = apply_elbe(carteira, elbe, months_col="meses_em_default")

    res = ecl_table(carteira, model=pd_lt, lgd="lgd", ead="saldo",
                    stage_col="estagio", elbe="elbe", discount_rate="taxa_efetiva",
                    age_col="idade", term_col="prazo")
    res.total, res.summary()
"""
from __future__ import annotations

# --- contrato de dados -------------------------------------------------
from .panel import PERIODS_PER_YEAR, ContractPanel, periods_per_year

# --- a curva e seus construtores ---------------------------------------
from .curves import CURVE_COLUMNS, PDCurve, constant_hazard, curve_frame, vintage_curve

# --- motores de curva ---------------------------------------------------
from .survival import BASELINES, HAZARD_LINKS, DiscreteHazard, kaplan_meier
from .markov import MarkovPD, pd_curve_from_matrix

# --- fachada de PD lifetime ---------------------------------------------
from .lifetime_pd import AGE_INDEXED, METHODS, LifetimePD, pit_from_ttc

# --- ELBE ----------------------------------------------------------------
from .elbe import (
    LGD_KINDS,
    LGD_UNITS,
    ELBETable,
    apply_elbe,
    detect_month_columns,
    elbe_frame,
    elbe_table,
    recovery_curve,
    workout_horizon,
)

# --- CCF / EAD ------------------------------------------------------------
from .ccf import (
    MEASURE_WEIGHT,
    MEASURES,
    REFERENCE_METHODS,
    CCFDataset,
    backtest_ead,
    ccf_downturn,
    ccf_psi,
    compare_measures,
    ead_from_ccf,
    ead_from_measure,
    pooled_ccf,
    reference_dataset,
)

# --- montagem do ECL -------------------------------------------------------
from .ecl import DEFAULT_STAGE_RULES, EAD_METHODS, ECLResult, ead_schedule, ecl_scenarios, ecl_table

__all__ = [
    # contrato
    "ContractPanel", "PERIODS_PER_YEAR", "periods_per_year",
    # curva
    "PDCurve", "constant_hazard", "vintage_curve", "curve_frame", "CURVE_COLUMNS",
    # motores
    "kaplan_meier", "DiscreteHazard", "HAZARD_LINKS", "BASELINES",
    "MarkovPD", "pd_curve_from_matrix",
    # fachada
    "LifetimePD", "pit_from_ttc", "METHODS", "AGE_INDEXED",
    # ELBE
    "ELBETable", "elbe_table", "elbe_frame", "apply_elbe", "recovery_curve",
    "workout_horizon", "detect_month_columns", "LGD_KINDS", "LGD_UNITS",
    # CCF
    "CCFDataset", "reference_dataset", "pooled_ccf", "ead_from_ccf", "ead_from_measure",
    "backtest_ead", "compare_measures", "ccf_psi", "ccf_downturn",
    "REFERENCE_METHODS", "MEASURES", "MEASURE_WEIGHT",
    # ECL
    "ecl_table", "ECLResult", "ecl_scenarios", "ead_schedule",
    "DEFAULT_STAGE_RULES", "EAD_METHODS",
    # carregados sob demanda
    "report", "tracking", "log_lifetime_pd", "log_elbe", "log_ccf", "log_ecl_run",
]


def __getattr__(name):
    # Visualizações (matplotlib) e MLflow só entram quando pedidos. Usa
    # importlib.import_module (e não ``from . import x``) para evitar a recursão
    # de ``_handle_fromlist`` → ``__getattr__`` neste próprio módulo.
    import importlib

    if name in ("report", "tracking"):
        return importlib.import_module(f"{__name__}.{name}")
    if name in ("log_lifetime_pd", "log_elbe", "log_ccf", "log_ecl_run"):
        return getattr(importlib.import_module(f"{__name__}.tracking"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
