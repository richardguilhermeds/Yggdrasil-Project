"""
Testes do pacote de capital econômico ``yggdrasil.credit_risk.capital``.

Cobre a espinha (Segment/Portfolio, medidas de risco), os motores (ASRF
analítico e Monte Carlo multifatorial), a alocação de Euler e a validação. A
propriedade-chave de sanidade do guia é testada explicitamente: **um único
fator + carteira granular ⇒ a simulação reproduz o ASRF**.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.capital import (
    LossDistribution,
    Portfolio,
    Segment,
)
from yggdrasil.credit_risk.capital.asrf import asrf_capital, conditional_pd, capital_ratio
from yggdrasil.credit_risk.capital.measures import (
    economic_capital,
    expected_loss,
    expected_shortfall,
    value_at_risk,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
def toy_single_factor():
    """Carteira granular de fator único (para reproduzir o ASRF)."""
    segs = [
        Segment(f"s{i}", pd=p, lgd=0.45, ead=1_000_000, rho=0.15,
                n_obligors=100_000, factor="F")
        for i, p in enumerate([0.01, 0.02, 0.03, 0.05])
    ]
    return Portfolio(segs, name="toy_single")


def toy_multifactor():
    segs = [
        Segment("cartao", pd=0.05, lgd=0.75, ead=3_000_000, rho=0.10,
                n_obligors=50_000, factor="cartao", product="cartao"),
        Segment("consig", pd=0.01, lgd=0.30, ead=3_000_000, rho=0.04,
                n_obligors=50_000, factor="consig", product="consignado"),
    ]
    corr = np.array([[1.0, 0.2], [0.2, 1.0]])
    return Portfolio(segs, factor_corr=corr, factor_names=["cartao", "consig"])


# ----------------------------------------------------------------------
# Medidas de risco
# ----------------------------------------------------------------------
def test_measures_relations():
    rng = np.random.default_rng(0)
    losses = rng.gamma(2.0, 50_000, size=100_000)
    el = expected_loss(losses)
    var = value_at_risk(losses, 0.99)
    es = expected_shortfall(losses, 0.99)
    assert es >= var >= el > 0                         # ES >= VaR >= EL
    assert economic_capital(losses, 0.99, "var") == pytest.approx(var - el)
    assert economic_capital(losses, 0.99, "es") == pytest.approx(es - el)


def test_lossdistribution_weighted():
    vals = np.array([0.0, 100.0, 200.0, 300.0])
    probs = np.array([0.7, 0.2, 0.07, 0.03])
    ld = LossDistribution(vals, weights=probs, expected=float((vals * probs).sum()))
    assert ld.el == pytest.approx(43.0)
    assert ld.var(0.95) == 200.0                       # quantil ponderado (inversa da CDF)
    assert ld.es(0.95) >= ld.var(0.95)


def test_measures_reject_bad_input():
    with pytest.raises(ValueError):
        value_at_risk([1, 2, 3], q=1.5)
    with pytest.raises(ValueError):
        expected_loss([])


# ----------------------------------------------------------------------
# Segment / Portfolio (contrato)
# ----------------------------------------------------------------------
def test_segment_validation():
    with pytest.raises(ValueError):
        Segment("x", pd=1.2, lgd=0.5, ead=1.0)
    with pytest.raises(ValueError):
        Segment("x", pd=0.1, lgd=0.5, ead=-1.0)
    with pytest.raises(ValueError):
        Segment("x", pd=0.1, lgd=0.5, ead=1.0, rho=1.0)
    s = Segment("x", pd=0.1, lgd=0.5, ead=1_000, product="cartao")
    assert s.factor == "cartao"                         # fator herda do produto
    assert s.expected_loss == pytest.approx(50.0)


def test_portfolio_basics_and_el():
    port = toy_multifactor()
    # EL = 0.05*0.75*3e6 + 0.01*0.30*3e6 = 112500 + 9000
    assert port.expected_loss() == pytest.approx(121_500.0)
    assert port.total_ead() == pytest.approx(6_000_000.0)
    assert port.n_factors == 2
    assert list(port.factor_of()) == [0, 1]


def test_portfolio_rejects_bad_corr():
    segs = [Segment("a", 0.02, 0.5, 1e6, factor="F1"),
            Segment("b", 0.02, 0.5, 1e6, factor="F2")]
    with pytest.raises(ValueError):                     # matriz de tamanho errado
        Portfolio(segs, factor_corr=np.eye(3), factor_names=["F1", "F2"])


def test_portfolio_from_frame():
    df = pd.DataFrame({
        "segmento": ["a", "b"], "pd": [0.02, 0.03], "lgd": [0.4, 0.5],
        "ead": [1e6, 2e6], "rho": [0.1, 0.12], "fator": ["F", "F"],
    })
    port = Portfolio.from_frame(df)
    assert port.n_segments == 2
    assert port.expected_loss() == pytest.approx(0.02 * 0.4 * 1e6 + 0.03 * 0.5 * 2e6)


# ----------------------------------------------------------------------
# ASRF analítico
# ----------------------------------------------------------------------
def test_conditional_pd_monotone_and_bounds():
    # rho=0 -> não há amplificação sistêmica: PD condicional = PD em qualquer q.
    assert conditional_pd(0.02, 0.0, 0.999) == pytest.approx(0.02, abs=1e-6)
    assert conditional_pd(0.02, 0.0, 0.5) == pytest.approx(0.02, abs=1e-6)
    # Monótona crescente em q; no quantil adverso 99,9% a PD condicional dispara.
    assert conditional_pd(0.02, 0.15, 0.999) > conditional_pd(0.02, 0.15, 0.9) > 0.02
    # No fator mediano (q=0.5) a PD condicional fica ABAIXO da PD incondicional
    # (a média é recuperada integrando sobre Y, não no ponto mediano).
    assert conditional_pd(0.02, 0.15, 0.5) < 0.02


def test_asrf_capital_additive():
    port = toy_single_factor()
    res = asrf_capital(port, q=0.999)
    # capital agregado = soma dos capitais por segmento (aditividade)
    assert res.economic_capital == pytest.approx(res.per_segment["capital"].sum())
    assert res.value_at_risk == pytest.approx(res.expected_loss + res.economic_capital)
    assert (res.per_segment["capital"] > 0).all()


def test_capital_ratio_matches_formula():
    k = capital_ratio(0.02, 0.45, 0.15, 0.999)
    expected = 0.45 * (conditional_pd(0.02, 0.15, 0.999) - 0.02)
    assert k == pytest.approx(expected)


# ----------------------------------------------------------------------
# Monte Carlo: reproduz o ASRF (sanidade do guia) + diversificação
# ----------------------------------------------------------------------
def test_mc_reproduces_asrf_single_factor_granular():
    port = toy_single_factor()
    a = asrf_capital(port, q=0.999)
    sim = port.simulate(n_scenarios=300_000, q=0.999, seed=7, granular=True)
    # Tolerância ampla: o quantil 99,9% tem ruído de Monte Carlo mesmo com 300k.
    assert sim.var() == pytest.approx(a.value_at_risk, rel=0.05)
    assert sim.distribution().el == pytest.approx(a.expected_loss, rel=1e-9)


def test_mc_diversification_benefit_positive():
    port = toy_multifactor()
    sim = port.simulate(n_scenarios=150_000, q=0.999, seed=3, granular=True)
    div = sim.diversification_benefit()
    # fatores correlacionados < 1 => capital integrado < soma dos isolados
    assert div["beneficio_diversificacao"] > 0
    assert div["capital_integrado"] < div["capital_isolado"]


def test_mc_stochastic_lgd_increases_capital():
    # LGD estocástica com correlação adversa PD-LGD aumenta o capital.
    segs = [Segment("veic", pd=0.03, lgd=0.40, ead=5_000_000, rho=0.08,
                    n_obligors=40_000, factor="veic", lgd_vol=0.25)]
    port = Portfolio(segs)
    base = port.simulate(120_000, q=0.999, seed=5, stochastic_lgd=False)
    stoch = port.simulate(120_000, q=0.999, seed=5, stochastic_lgd=True, pd_lgd_corr=0.5)
    assert stoch.economic_capital(metric="var") > base.economic_capital(metric="var")


def test_mc_concentration_raises_tail():
    # Poucos devedores (não-granular) => risco idiossincrático engorda a cauda.
    seg_gran = [Segment("g", pd=0.03, lgd=0.5, ead=1e6, rho=0.10, n_obligors=100_000, factor="F")]
    seg_conc = [Segment("c", pd=0.03, lgd=0.5, ead=1e6, rho=0.10, n_obligors=50, factor="F")]
    var_gran = Portfolio(seg_gran).simulate(120_000, q=0.999, seed=1, granular=False).var()
    var_conc = Portfolio(seg_conc).simulate(120_000, q=0.999, seed=1, granular=False).var()
    assert var_conc > var_gran


# ----------------------------------------------------------------------
# Alocação de Euler
# ----------------------------------------------------------------------
def test_euler_allocation_additive():
    port = toy_multifactor()
    sim = port.simulate(150_000, q=0.999, seed=11)
    alloc = sim.allocate(metric="es")
    # a soma do capital alocado = ES - EL empírica (aditividade de Euler)
    total = sim.es() - sim.segment_losses.mean(axis=0).sum()
    assert alloc["capital_alocado"].sum() == pytest.approx(total, rel=1e-6)
    assert set(["segmento", "capital_alocado", "capital_isolado",
                "beneficio_diversificacao", "share_capital"]).issubset(alloc.columns)


def test_raroc():
    from yggdrasil.credit_risk.capital.allocation import raroc
    assert raroc(receita=100, custo=20, perda_esperada=30, capital=200) == pytest.approx(0.25)
    assert np.isnan(raroc(1, 1, 1, 0))


# ----------------------------------------------------------------------
# Validação
# ----------------------------------------------------------------------
def test_sensitivity_asrf():
    from yggdrasil.credit_risk.capital.validation import sensitivity
    port = toy_single_factor()
    df = sensitivity(port, q=0.999, shocks=(-0.1, 0.1), params=("pd", "lgd", "rho"))
    assert "base" in df["parametro"].values
    # aumentar LGD aumenta o capital
    up = df[(df.parametro == "lgd") & (df.choque == 0.1)]["CE"].iloc[0]
    base = df[df.parametro == "base"]["CE"].iloc[0]
    assert up > base


def test_convergence_shape():
    from yggdrasil.credit_risk.capital.validation import convergence
    port = toy_single_factor()
    df = convergence(port, n_grid=(2_000, 10_000), q=0.99, seed=0)
    assert list(df["n_cenarios"]) == [2_000, 10_000]
    assert (df["VaR"] > df["EL"]).all()
