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
    apply_scenario,
    scenario_capital,
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
# IC de VaR/ES (estatísticas de ordem + bootstrap da cauda)
# ----------------------------------------------------------------------
def test_var_ci_coverage_normal():
    """O IC por estatísticas de ordem cobre o quantil teórico ~95% das vezes."""
    from scipy.stats import norm

    rng = np.random.default_rng(123)
    q, alpha, n, n_rep = 0.95, 0.05, 1_000, 200
    teorico = float(norm.ppf(q))                       # quantil verdadeiro da N(0,1)
    cobre = 0
    for _ in range(n_rep):
        ld = LossDistribution(rng.standard_normal(n))
        lo, hi = ld.var_ci(q, alpha)
        assert lo <= hi
        cobre += int(lo <= teorico <= hi)
    cobertura = cobre / n_rep
    # cobertura garantida >= 1-alpha (IC conservador); tolerância p/ ruído de simulação
    assert 0.92 <= cobertura <= 1.0


def test_es_ci_and_se_empirical():
    rng = np.random.default_rng(0)
    ld = LossDistribution(rng.gamma(2.0, 50_000, size=50_000))
    lo, hi = ld.es_ci(0.99, n_boot=300, seed=42)
    assert lo <= ld.es(0.99) <= hi                     # banda envolve o ES pontual
    assert ld.es_ci(0.99, n_boot=300, seed=42) == (lo, hi)   # reprodutível c/ semente
    assert ld.var_se(0.99) > 0
    assert ld.es_se(0.99, n_boot=300, seed=42) > 0
    var_lo, var_hi = ld.var_ci(0.99)
    assert var_lo <= ld.var(0.99) <= var_hi


def test_ci_weighted_nan_and_summary_columns():
    # Distribuição ponderada (analítica): banda não se aplica -> NaN, sem quebrar.
    vals = np.array([0.0, 100.0, 200.0, 300.0])
    probs = np.array([0.7, 0.2, 0.07, 0.03])
    ld = LossDistribution(vals, weights=probs)
    assert all(np.isnan(x) for x in ld.var_ci(0.95))
    assert all(np.isnan(x) for x in ld.es_ci(0.95, seed=1))
    assert np.isnan(ld.var_se(0.95)) and np.isnan(ld.es_se(0.95))
    df = ld.summary()
    for col in ("VaR_lo", "VaR_hi", "ES_lo", "ES_hi"):
        assert col in df.columns
        assert df[col].isna().all()
    # Amostral: bandas finitas envolvendo a estimativa pontual.
    emp = LossDistribution(np.random.default_rng(1).gamma(2.0, 1.0, size=20_000))
    dfe = emp.summary(confidence_levels=(0.99,), n_boot=100, seed=7)
    assert dfe.loc[0, "VaR_lo"] <= dfe.loc[0, "VaR"] <= dfe.loc[0, "VaR_hi"]
    assert dfe.loc[0, "ES_lo"] <= dfe.loc[0, "ES"] <= dfe.loc[0, "ES_hi"]


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


# ----------------------------------------------------------------------
# Estresse: cenário macro → carteira estressada → capital (stress.py)
# ----------------------------------------------------------------------
def test_scenario_capital_adverse_increases_el_and_var():
    port = toy_multifactor()
    df = scenario_capital(port, {"adverso": {"pd_mult": 2.0, "lgd_mult": 1.1}}, engine="asrf")
    assert list(df["cenario"]) == ["base", "adverso"]      # base entra como referência
    base = df[df.cenario == "base"].iloc[0]
    adv = df[df.cenario == "adverso"].iloc[0]
    assert adv["EL"] > base["EL"] and adv["VaR"] > base["VaR"]
    assert adv["delta_EL"] > 0 and adv["delta_CE"] > 0
    assert base["delta_EL"] == pytest.approx(0.0) and base["delta_CE"] == pytest.approx(0.0)


def test_scenario_capital_identity_zero_delta():
    port = toy_multifactor()
    df = scenario_capital(port, {"base": {"pd_mult": 1.0, "lgd_mult": 1.0}}, engine="asrf")
    assert list(df["cenario"]) == ["base"]                 # sem linha duplicada de base
    row = df.iloc[0]
    assert row["delta_EL"] == pytest.approx(0.0, abs=1e-9)
    assert row["delta_VaR"] == pytest.approx(0.0, abs=1e-9)
    assert row["delta_CE"] == pytest.approx(0.0, abs=1e-9)


def test_scenario_capital_monte_carlo_engine():
    port = toy_multifactor()
    df = scenario_capital(port, {"adverso": {"pd_mult": 1.5}},
                          engine="monte_carlo", n_scenarios=20_000, seed=0)
    base = df[df.cenario == "base"].iloc[0]
    adv = df[df.cenario == "adverso"].iloc[0]
    assert np.isfinite(adv["ES"]) and adv["ES"] >= adv["VaR"]
    assert adv["EL"] > base["EL"] and adv["VaR"] > base["VaR"]


def test_apply_scenario_abs_and_validation():
    port = toy_multifactor()
    p2 = apply_scenario(port, {"pd_abs": 0.10, "lgd_abs": 0.5})
    assert all(s.pd == pytest.approx(0.10) for s in p2.segments)
    assert all(s.lgd == pytest.approx(0.5) for s in p2.segments)
    assert port.segments[0].pd == pytest.approx(0.05)      # original intocada
    with pytest.raises(ValueError):
        apply_scenario(port, {"pd_mult": 1.2, "pd_abs": 0.1})   # mult e abs juntos
    with pytest.raises(ValueError):
        apply_scenario(port, {"chave_errada": 1.0})
    with pytest.raises(ValueError):
        scenario_capital(port, {"x": {"pd_mult": -1.0}})
    with pytest.raises(ValueError):
        scenario_capital(port, {"x": {}}, engine="motor_inexistente")


def test_scenario_capital_conditional_z():
    # Carteira homogênea: o Z implicado reproduz exatamente o estresse direto.
    port = Portfolio([Segment("s", pd=0.03, lgd=0.5, ead=1e6, rho=0.12, n_obligors=10_000)])
    df = scenario_capital(port, {"adverso": {"pd_mult": 2.0}}, engine="asrf", conditional_z=True)
    adv = df[df.cenario == "adverso"].iloc[0]
    assert adv["z_implicito"] < 0                          # cenário ruim → Z na cauda inferior
    assert adv["EL"] == pytest.approx(0.06 * 0.5 * 1e6)    # PD condicionada == PD × mult
    # Heterogênea: o MESMO Z estressa mais o segmento de rho maior.
    port2 = Portfolio([Segment("baixo", pd=0.03, lgd=0.5, ead=1e6, rho=0.04),
                       Segment("alto", pd=0.03, lgd=0.5, ead=1e6, rho=0.20)])
    stressed = apply_scenario(port2, {"pd_mult": 2.0}, conditional_z=True)
    p_baixo, p_alto = stressed.segments[0].pd, stressed.segments[1].pd
    assert p_alto > p_baixo > 0.03


def test_scenario_capital_from_projection_ducktype():
    port = toy_multifactor()

    class _Proj:
        kind = "pd"

        def __init__(self, paths):
            self.paths = paths

    paths = {"base": pd.DataFrame({"mean": np.full(6, 0.02)}),
             "adverso": pd.DataFrame({"mean": np.full(6, 0.05)})}
    df = scenario_capital(port, _Proj(paths), engine="asrf")
    assert set(df["cenario"]) == {"base", "adverso"}
    base = df[df.cenario == "base"].iloc[0]
    adv = df[df.cenario == "adverso"].iloc[0]
    assert base["delta_EL"] == pytest.approx(0.0)          # mult 1.0 no cenário base
    assert adv["EL"] == pytest.approx(2.5 * base["EL"])    # mult 0.05/0.02 nas PDs → EL 2.5×

    # ScenarioSet-like sem model → erro amigável; com model fake → projeta e segue.
    class _FakeSet:
        scenarios = []

    with pytest.raises(ValueError):
        scenario_capital(port, _FakeSet())

    class _FakeModel:
        def project(self, scenarios, horizon=None, **kw):
            return _Proj(paths)

    df2 = scenario_capital(port, _FakeSet(), model=_FakeModel(), engine="asrf")
    assert set(df2["cenario"]) == {"base", "adverso"}


# ----------------------------------------------------------------------
# Migração: estimação empírica (coorte/duração) + z-shift do ciclo
# ----------------------------------------------------------------------
TM_3E = np.array([
    [0.90, 0.08, 0.02],
    [0.10, 0.82, 0.08],
    [0.00, 0.00, 1.00],       # default absorvente
])
RATINGS_3E = ["A", "B", "D"]


def _painel_migracao(tm, ratings, n_ids=3_000, n_periodos=8, seed=0):
    """Painel longo simulado de uma cadeia de Markov com matriz conhecida."""
    rng = np.random.default_rng(seed)
    n = len(ratings)
    estados = np.empty((n_ids, n_periodos), dtype=int)
    estados[:, 0] = rng.integers(0, n - 1, size=n_ids)      # começa fora do default
    cum = np.cumsum(tm, axis=1)
    for t in range(1, n_periodos):
        u = rng.random(n_ids)
        estados[:, t] = (u[:, None] > cum[estados[:, t - 1], :]).sum(axis=1)
    return pd.DataFrame({
        "id": np.repeat(np.arange(n_ids), n_periodos),
        "safra": np.tile(np.arange(n_periodos), n_ids),
        "rating": np.asarray(ratings, dtype=object)[estados.ravel()],
    })


def test_estimate_cohort_recovers_known_matrix():
    from yggdrasil.credit_risk.capital.migration import estimate_transition_matrix
    df = _painel_migracao(TM_3E, RATINGS_3E, seed=42)
    tm_hat, labels = estimate_transition_matrix(
        df, "id", "rating", "safra", method="cohort", ratings=RATINGS_3E)
    assert labels == RATINGS_3E
    assert np.allclose(tm_hat.sum(axis=1), 1.0)
    assert np.allclose(tm_hat, TM_3E, atol=0.02)            # recupera a matriz geradora
    assert np.allclose(tm_hat[-1], [0.0, 0.0, 1.0])         # default absorvente exato


def test_estimate_duration_valid_and_close():
    from yggdrasil.credit_risk.capital.migration import estimate_transition_matrix
    df = _painel_migracao(TM_3E, RATINGS_3E, seed=42)
    tm_hat, _ = estimate_transition_matrix(
        df, "id", "rating", "safra", method="duration", ratings=RATINGS_3E)
    assert np.allclose(tm_hat.sum(axis=1), 1.0)
    assert (tm_hat >= 0).all() and (tm_hat <= 1).all()
    assert np.allclose(tm_hat[-1], [0.0, 0.0, 1.0])         # gerador preserva o absorvente
    # expm do gerador ~ matriz de coorte (mesma ordem de grandeza por célula)
    assert np.allclose(tm_hat, TM_3E, atol=0.05)


def test_estimate_smoothing_fills_empty_cells_keeps_absorbing():
    from yggdrasil.credit_risk.capital.migration import estimate_transition_matrix
    # Painel mínimo: A→D nunca observado; D observado apenas como absorvente.
    df = pd.DataFrame({
        "id":     [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        "safra":  [0, 1, 2, 0, 1, 0, 1, 2, 0, 1],
        "rating": ["A", "A", "A", "A", "B", "B", "B", "D", "D", "D"],
    })
    tm_sem, _ = estimate_transition_matrix(df, "id", "rating", "safra",
                                           ratings=["A", "B", "D"])
    assert tm_sem[0, 2] == 0.0                              # célula vazia fica zero
    tm_com, _ = estimate_transition_matrix(df, "id", "rating", "safra",
                                           ratings=["A", "B", "D"], smoothing=0.5)
    assert tm_com[0, 2] > 0.0                               # suavização preenche
    assert np.allclose(tm_com.sum(axis=1), 1.0)
    assert np.allclose(tm_com[-1], [0.0, 0.0, 1.0])         # absorvente intacto


def test_estimate_rejects_bad_input():
    from yggdrasil.credit_risk.capital.migration import estimate_transition_matrix
    df = pd.DataFrame({"id": [1, 1], "safra": [0, 1], "rating": ["A", "B"]})
    with pytest.raises(ValueError):
        estimate_transition_matrix(df, "id", "rating", "safra", method="xxx")
    with pytest.raises(ValueError):
        estimate_transition_matrix(df, "id", "rating", "safra", smoothing=-1.0)
    dup = pd.DataFrame({"id": [1, 1], "safra": [0, 0], "rating": ["A", "B"]})
    with pytest.raises(ValueError):                          # (id, período) duplicado
        estimate_transition_matrix(dup, "id", "rating", "safra")


def test_zshift_identity_at_zero():
    from yggdrasil.credit_risk.capital.migration import zshift_transition_matrix
    same = zshift_transition_matrix(TM_3E, z=0.0, rho=0.2)
    assert np.allclose(same, TM_3E, atol=1e-9)              # idempotência em z=0
    # rho=0: o ciclo não carrega — matriz inalterada para qualquer z.
    assert np.allclose(zshift_transition_matrix(TM_3E, z=-3.0, rho=0.0), TM_3E, atol=1e-9)


def test_zshift_adverse_increases_downgrade():
    from yggdrasil.credit_risk.capital.migration import zshift_transition_matrix
    piora = zshift_transition_matrix(TM_3E, z=-2.0, rho=0.2)    # z<0 = ciclo adverso
    melhora = zshift_transition_matrix(TM_3E, z=+2.0, rho=0.2)  # z>0 = ciclo benigno
    assert np.allclose(piora.sum(axis=1), 1.0)
    assert np.allclose(melhora.sum(axis=1), 1.0)
    for i in range(2):                                       # linhas vivas (A e B)
        # adverso: mais massa nos estados piores que o atual e mais default
        assert piora[i, -1] > TM_3E[i, -1]
        assert piora[i, i + 1:].sum() > TM_3E[i, i + 1:].sum()
        # benigno: menos default
        assert melhora[i, -1] < TM_3E[i, -1]
    # o estado absorvente não se move em nenhum cenário
    assert np.allclose(piora[-1], [0.0, 0.0, 1.0])
    assert np.allclose(melhora[-1], [0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        zshift_transition_matrix(TM_3E, z=1.0, rho=1.5)


def test_estimate_feeds_migration_model():
    from yggdrasil.credit_risk.capital.migration import (
        MigrationModel, estimate_transition_matrix, zshift_transition_matrix)
    df = _painel_migracao(TM_3E, RATINGS_3E, n_ids=500, n_periodos=6, seed=1)
    tm_hat, labels = estimate_transition_matrix(
        df, "id", "rating", "safra", ratings=RATINGS_3E, smoothing=0.5)
    tm_pit = zshift_transition_matrix(tm_hat, z=-1.0, rho=0.15)
    model = MigrationModel(tm_pit, ratings=labels, values=np.array([1.0, 0.95, 0.4]))
    dist = model.simulate(exposures=[1e6, 5e5], ratings_idx=[0, 1],
                          n_scenarios=5_000, seed=3)
    assert np.isfinite(dist.el)


# ----------------------------------------------------------------------
# Incerteza de rho: bootstrap em blocos, multi-start na MLE e sanidade
# ----------------------------------------------------------------------
def _serie_vasicek(T, rho, pd_incond, seed):
    """Taxas de default simuladas do modelo de Vasicek de fator único."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(T)
    return norm.cdf((norm.ppf(pd_incond) - np.sqrt(rho) * y) / np.sqrt(1.0 - rho))


def test_asset_correlation_ci_contains_true_rho():
    from yggdrasil.credit_risk.capital.correlation import asset_correlation_ci
    dr = _serie_vasicek(60, rho=0.15, pd_incond=0.02, seed=2)
    res = asset_correlation_ci(dr, n_boot=300, block=4, alpha=0.05, seed=7)
    assert res["ic_inferior"] <= 0.15 <= res["ic_superior"]   # cobre o rho verdadeiro
    assert res["ic_inferior"] <= res["rho"] <= res["ic_superior"]
    assert res["rhos_boot"].shape == (300,)
    # reprodutível com a mesma semente
    res2 = asset_correlation_ci(dr, n_boot=300, block=4, alpha=0.05, seed=7)
    assert res2["ic_inferior"] == res["ic_inferior"]
    assert res2["ic_superior"] == res["ic_superior"]
    # MLE exige exposições para reconstruir as contagens
    with pytest.raises(ValueError):
        asset_correlation_ci(dr, n_boot=10, method="mle")
    with pytest.raises(ValueError):
        asset_correlation_ci(dr, alpha=1.5)


def test_asset_params_mle_multistart_not_worse():
    from scipy.stats import norm
    from yggdrasil.credit_risk.capital.correlation import asset_params_mle
    rng = np.random.default_rng(5)
    T, n_oblig, rho_true, pd_true = 20, 500, 0.12, 0.03
    y = rng.standard_normal(T)
    p = norm.cdf((norm.ppf(pd_true) - np.sqrt(rho_true) * y) / np.sqrt(1 - rho_true))
    k = rng.binomial(n_oblig, p)
    n = np.full(T, n_oblig)
    multi = asset_params_mle(k, n, return_details=True)
    single = asset_params_mle(k, n, rho_starts=(0.10,), return_details=True)
    # multi-start nunca piora a log-verossimilhança vs chute único
    assert multi["loglik"] >= single["loglik"] - 1e-6
    assert not multi["fallback_momentos"]
    assert multi["rho"] == pytest.approx(rho_true, abs=0.08)
    # retorno-tupla (compatibilidade) bate com o detalhado
    pd_hat, rho_hat = asset_params_mle(k, n)
    assert pd_hat == pytest.approx(multi["pd"])
    assert rho_hat == pytest.approx(multi["rho"])
    with pytest.raises(ValueError):                          # grade inválida
        asset_params_mle(k, n, rho_starts=())


def test_rho_sanity_report_flags_out_of_range():
    from yggdrasil.credit_risk.capital.correlation import (
        IRB_CORPORATE_RHO, IRB_RETAIL_RHO, rho_sanity_report)
    series = {
        "alto": _serie_vasicek(40, rho=0.50, pd_incond=0.05, seed=3),
        "ok": _serie_vasicek(40, rho=0.08, pd_incond=0.03, seed=4),
    }
    rep = rho_sanity_report(series, segment_type="retail",
                            n_boot=100, block=4, seed=1).set_index("segmento")
    assert bool(rep.loc["alto", "fora_da_faixa"])            # rho~0.5 >> teto retail
    assert not bool(rep.loc["ok", "fora_da_faixa"])
    assert rep.loc["ok", "faixa_irb_min"] == IRB_RETAIL_RHO[0]
    assert rep.loc["ok", "faixa_irb_max"] == IRB_RETAIL_RHO[1]
    assert (rep.loc["alto", "ic_inferior"]
            <= rep.loc["alto", "rho_momentos"]
            <= rep.loc["alto", "ic_superior"])
    assert rep["rho_mle"].isna().all()                       # sem exposições -> NaN
    # tipo por segmento via dict: 'alto' avaliado na faixa corporativa
    rep2 = rho_sanity_report({"alto": series["alto"]},
                             segment_type={"alto": "corporate"},
                             n_boot=50, seed=1).iloc[0]
    assert rep2["tipo"] == "corporate"
    assert rep2["faixa_irb_max"] == IRB_CORPORATE_RHO[1]
    assert bool(rep2["fora_da_faixa"])                       # 0.5 > 0.24


# ----------------------------------------------------------------------
# Tracking MLflow: relatório em abas + comparação de runs generalizada
# ----------------------------------------------------------------------
def test_runs_comparison_df_default_and_custom_spec():
    """Regressão: sem metrics_spec o resumo mantém as colunas dos segmentadores;
    com metrics_spec, as colunas são as pedidas (métricas numéricas, params texto)."""
    from yggdrasil.credit_risk import _mlflow_report as R

    runs = pd.DataFrame({
        "run_id": ["abc12345", "def67890"],
        "start_time": pd.to_datetime(["2026-01-02 10:00", "2026-01-01 09:00"], utc=True),
        "tags.mlflow.runName": ["run_a", None],
        "params.algorithm": [None, "cart"],
        "params.task_type": ["clf", None],
        "metrics.val_ks": [0.51234, 0.4],
        "metrics.val_psi": [0.02, 0.03],
        "metrics.EL": [10.0, 20.0],
    })

    class _FakeMlflow:
        def search_runs(self, experiment_ids, order_by):
            return runs

    fake = _FakeMlflow()
    out = R.runs_comparison_df(fake, "0", "abc12345")
    assert list(out.columns) == ["", "run", "quando", "algoritmo",
                                 "KS (OOT)", "RMSE (OOT)", "PSI (OOT)"]
    assert out.loc[0, ""] == "➤"                          # linha do run atual
    assert out.loc[0, "algoritmo"] == "clf"               # fallback em task_type
    assert out.loc[0, "KS (OOT)"] == pytest.approx(0.5123)
    assert np.isnan(out.loc[0, "RMSE (OOT)"])             # métrica ausente -> NaN
    assert out.loc[1, "run"] == "def67890"                # runName ausente -> run_id[:8]

    custom = R.runs_comparison_df(fake, "0", "def67890",
                                  metrics_spec=[("metrics.EL", "EL"),
                                                ("params.task_type", "tarefa")])
    assert list(custom.columns) == ["", "run", "quando", "EL", "tarefa"]
    assert custom.loc[1, ""] == "➤"
    assert custom.loc[1, "EL"] == pytest.approx(20.0)
    assert custom.loc[0, "tarefa"] == "clf"


def test_log_capital_run_generates_tabbed_report(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.capital.asrf import asrf_capital
    from yggdrasil.credit_risk.capital.tracking import log_capital_run

    port = toy_multifactor()
    sim = port.simulate(20_000, q=0.99, seed=2)
    alloc = sim.allocate(metric="es")
    a = asrf_capital(port, q=0.99)
    prev_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
        run_id = log_capital_run(port, sim, allocation=alloc, asrf=a,
                                 run_name="teste_report",
                                 artifacts_dir=str(tmp_path / "arts"))
        run = mlflow.get_run(run_id)
        arts = [f.path for f in mlflow.MlflowClient().list_artifacts(run_id)]
        html = open(mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="report.html"), encoding="utf-8").read()
    finally:
        mlflow.set_tracking_uri(prev_uri)
    assert "report.html" in arts                          # artefato na raiz do run
    assert "report_error" not in run.data.tags            # relatório sem falha
    assert {"EL", "VaR", "ES", "CE_var", "CE_es", "CE_asrf"} <= set(run.data.metrics)
    # abas: Resumo sempre; Euler e Validação porque os dados existem no run
    for aba in ("Resumo · runs", "Alocação de Euler", "Validação · benchmark"):
        assert aba in html
    assert "➤" in html                                    # run atual destacado
