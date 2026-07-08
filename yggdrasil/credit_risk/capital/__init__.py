"""
yggdrasil.credit_risk.capital
=============================
**Modelos de capital econômico para risco de crédito** — a estimativa do
capital necessário para absorver as **perdas inesperadas** da carteira em um
horizonte de 1 ano e num nível de confiança compatível com o apetite de risco
(ex.: 99,9%). Complementa a provisão (ECL, que cobre a perda **esperada** sob
IFRS 9 / Resolução CMN 4.966) e o capital regulatório de Pilar 1, trazendo a
visão **interna** da instituição: parâmetros próprios, correlações próprias e a
estrutura de dependência real entre produtos (cartão, consignado, veículos),
com os efeitos de concentração e diversificação que o Pilar 1 ignora.

O pacote implementa os "projetos" do guia de construção, das definições ao uso
gerencial:

Contrato de dados
    :class:`Segment`, :class:`Portfolio` — segmento homogêneo (PD TTC, LGD/CCF
    downturn, ρ, fator sistêmico) e carteira (matriz de correlação entre fatores).

Distribuição de perdas e medidas
    :class:`LossDistribution`, :func:`value_at_risk`, :func:`expected_shortfall`,
    :func:`economic_capital` (``CE = VaR_q − EL``).

Motores de cálculo
    :func:`asrf_capital` (v1, ASRF/Vasicek analítico) · :func:`simulate`
    (v2, Monte Carlo multifatorial) · :func:`creditrisk_plus` (benchmark
    atuarial) · :class:`MigrationModel` (CreditMetrics / migração de estágio).

Insumos e parâmetros
    :func:`pit_to_ttc`, :func:`lgd_downturn_from_series`, :func:`ccf_downturn`,
    :func:`basel_correlation`, :func:`basel_irb_capital` (Pilar 1) ·
    correlações: :func:`asset_correlation_moments`, :func:`asset_correlation_mle`,
    :func:`factor_correlation_matrix`, :func:`nearest_correlation`.

Alocação e uso gerencial
    :func:`euler_allocation` (contribuição à cauda), :func:`raroc`,
    :func:`raroc_table`.

Validação
    :func:`sensitivity`, :func:`correlation_stress`, :func:`benchmark`,
    :func:`pillar1_comparison`, :func:`backtest_expected_loss`,
    :func:`convergence`.

Produtos
    :func:`preset`, :data:`PRESETS` — particularidades de cartão, consignado,
    veículos e afins (Tabela 3 do guia).

As visualizações (:mod:`.report`, matplotlib) e o registro no MLflow
(:mod:`.tracking`) são carregados **sob demanda**.

Uso típico::

    from yggdrasil.credit_risk.capital import Portfolio, Segment

    carteira = Portfolio([
        Segment("cartao_revolver", pd=0.06, lgd=0.75, ead=8e6, rho=0.10,
                n_obligors=40_000, product="cartao", factor="cartao"),
        Segment("consig_inss", pd=0.01, lgd=0.30, ead=12e6, rho=0.04,
                n_obligors=60_000, product="consignado", factor="consignado"),
    ], factor_corr=[[1.0, 0.25], [0.25, 1.0]], factor_names=["cartao", "consignado"])

    carteira.asrf_capital(q=0.999).summary()          # v1 analítico
    sim = carteira.simulate(n_scenarios=200_000, q=0.999, seed=42)   # v2
    sim.economic_capital(), sim.allocate(metric="es")  # capital + alocação
"""
from __future__ import annotations

# --- contrato de dados -------------------------------------------------
from .portfolio import Portfolio, Segment

# --- distribuição de perdas e medidas ----------------------------------
from .measures import (
    DEFAULT_CONFIDENCE,
    LossDistribution,
    economic_capital,
    expected_loss,
    expected_shortfall,
    loss_volatility,
    unexpected_loss,
    value_at_risk,
)

# --- motores de cálculo ------------------------------------------------
from .asrf import AsrfResult, asrf_capital, capital_ratio, conditional_pd
from .monte_carlo import SimulationResult, simulate
from .creditrisk_plus import creditrisk_plus, panjer_recursion
from .migration import (
    MigrationModel,
    migration_thresholds,
    simulate_creditmetrics,
    two_state_matrix,
)

# --- alocação e uso gerencial ------------------------------------------
from .allocation import euler_allocation, raroc, raroc_table

# --- insumos: correlações e parâmetros ---------------------------------
from .correlation import (
    asset_correlation_mle,
    asset_correlation_moments,
    asset_params_mle,
    factor_correlation_matrix,
    is_positive_definite,
    macro_factor_correlation,
    nearest_correlation,
)
from .parameters import (
    basel_capital_portfolio,
    basel_correlation,
    basel_irb_capital,
    basel_rwa,
    ccf_downturn,
    lgd_downturn_addon,
    lgd_downturn_from_series,
    pit_to_ttc,
)

# --- produtos ----------------------------------------------------------
from .products import PRESETS, ProductPreset, apply_preset, list_presets, preset, presets_frame

# --- validação ---------------------------------------------------------
from .validation import (
    backtest_expected_loss,
    benchmark,
    convergence,
    correlation_stress,
    pillar1_comparison,
    sensitivity,
)

__all__ = [
    # contrato
    "Segment", "Portfolio",
    # medidas
    "LossDistribution", "DEFAULT_CONFIDENCE", "expected_loss", "value_at_risk",
    "expected_shortfall", "unexpected_loss", "economic_capital", "loss_volatility",
    # motores
    "asrf_capital", "AsrfResult", "conditional_pd", "capital_ratio",
    "simulate", "SimulationResult",
    "creditrisk_plus", "panjer_recursion",
    "MigrationModel", "simulate_creditmetrics", "migration_thresholds", "two_state_matrix",
    # alocação
    "euler_allocation", "raroc", "raroc_table",
    # correlações
    "asset_correlation_moments", "asset_correlation_mle", "asset_params_mle",
    "factor_correlation_matrix", "nearest_correlation", "is_positive_definite",
    "macro_factor_correlation",
    # parâmetros
    "pit_to_ttc", "lgd_downturn_addon", "lgd_downturn_from_series", "ccf_downturn",
    "basel_correlation", "basel_irb_capital", "basel_rwa", "basel_capital_portfolio",
    # produtos
    "ProductPreset", "PRESETS", "preset", "list_presets", "presets_frame", "apply_preset",
    # validação
    "sensitivity", "correlation_stress", "benchmark", "pillar1_comparison",
    "backtest_expected_loss", "convergence",
    # carregados sob demanda
    "report", "tracking", "log_capital_run",
]


def __getattr__(name):
    # Visualizações (matplotlib) e MLflow carregados só quando pedidos.
    # Usa importlib.import_module (e não ``from . import x``) para evitar a
    # recursão de ``_handle_fromlist`` → ``__getattr__`` neste próprio módulo.
    import importlib

    if name in ("report", "tracking"):
        return importlib.import_module(f"{__name__}.{name}")
    if name == "log_capital_run":
        return importlib.import_module(f"{__name__}.tracking").log_capital_run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
