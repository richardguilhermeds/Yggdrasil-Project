"""
Testes dos motores e insumos de suporte do pacote de capital econômico:
CreditRisk+ (benchmark atuarial), estimação de correlações, calibração de
parâmetros e fórmulas regulatórias IRB, presets de produto, migração
(CreditMetrics) e visualizações.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from yggdrasil.credit_risk.capital import (
    MigrationModel,
    Portfolio,
    Segment,
    asset_correlation_mle,
    asset_correlation_moments,
    basel_capital_portfolio,
    basel_correlation,
    basel_irb_capital,
    ccf_downturn,
    creditrisk_plus,
    factor_correlation_matrix,
    is_positive_definite,
    lgd_downturn_from_series,
    list_presets,
    nearest_correlation,
    pit_to_ttc,
    preset,
    presets_frame,
    two_state_matrix,
)


def toy_portfolio():
    return Portfolio([
        Segment("cartao", pd=0.06, lgd=0.75, ead=8e6, rho=0.10, n_obligors=40_000,
                product="cartao", factor="cartao"),
        Segment("consig", pd=0.01, lgd=0.30, ead=12e6, rho=0.04, n_obligors=60_000,
                product="consignado", factor="consignado"),
    ], factor_corr=[[1.0, 0.25], [0.25, 1.0]], factor_names=["cartao", "consignado"])


# ----------------------------------------------------------------------
# CreditRisk+
# ----------------------------------------------------------------------
def test_creditrisk_plus_el_matches_analytic():
    port = toy_portfolio()
    dist = creditrisk_plus(port, sigma=0.5)
    # A EL da distribuição bate com Σ PD·LGD·EAD (erro de discretização pequeno).
    assert dist.el == pytest.approx(port.expected_loss(), rel=0.03)
    assert dist.var(0.999) > dist.el                       # há cauda
    # weights de uma distribuição discreta somam ~1
    assert dist.weights is not None
    assert dist.weights.sum() == pytest.approx(1.0, abs=1e-3)


def test_creditrisk_plus_sigma_fattens_tail():
    port = toy_portfolio()
    low = creditrisk_plus(port, sigma=0.1).var(0.999)
    high = creditrisk_plus(port, sigma=1.0).var(0.999)
    assert high > low                                      # mais vol sistêmica ⇒ cauda maior


# ----------------------------------------------------------------------
# Correlação
# ----------------------------------------------------------------------
def _vasicek_series(T, n, rho, pd, seed):
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal(T)
    p = norm.cdf((norm.ppf(pd) - np.sqrt(rho) * Y) / np.sqrt(1 - rho))
    return rng.binomial(n, p), np.full(T, n)


def test_asset_correlation_recovers_rho():
    k, n = _vasicek_series(300, 600, rho=0.10, pd=0.03, seed=1)
    mom = asset_correlation_moments(k / n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mle = asset_correlation_mle(k, n)
    assert mom == pytest.approx(0.10, abs=0.05)
    assert mle == pytest.approx(0.10, abs=0.05)


def test_nearest_correlation_and_pd():
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert not is_positive_definite(bad)
    fixed = nearest_correlation(bad)
    assert is_positive_definite(fixed)
    assert np.allclose(np.diag(fixed), 1.0, atol=1e-6)
    assert is_positive_definite(np.eye(3))


def test_factor_correlation_matrix():
    k1, n1 = _vasicek_series(200, 500, 0.10, 0.03, seed=2)
    k2, n2 = _vasicek_series(200, 500, 0.06, 0.02, seed=3)
    df = pd.DataFrame({"cartao": k1 / n1, "consig": k2 / n2})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        M = factor_correlation_matrix(df)
    assert M.shape == (2, 2)
    assert np.allclose(np.diag(M), 1.0, atol=1e-6)
    assert is_positive_definite(M)


# ----------------------------------------------------------------------
# Parâmetros e fórmulas regulatórias
# ----------------------------------------------------------------------
def test_pit_to_ttc_mean():
    assert pit_to_ttc([0.02, 0.05, 0.03]) == pytest.approx(0.0333, abs=1e-3)


def test_lgd_downturn_and_ccf():
    # LGD nos piores períodos de default é maior que a média.
    dr = np.array([0.01, 0.02, 0.08, 0.03, 0.09, 0.015])
    lgd = np.array([0.30, 0.32, 0.55, 0.34, 0.60, 0.31])
    dt = lgd_downturn_from_series(lgd, dr, worst_frac=0.34)
    assert dt > lgd.mean()
    assert 0.0 <= ccf_downturn(np.array([0.3, 0.5, 0.7, 0.9, 0.95])) <= 1.0


def test_basel_correlation_classes():
    assert basel_correlation(0.5, "revolving") == pytest.approx(0.04)
    assert basel_correlation(0.99, "mortgage") == pytest.approx(0.15)
    r = basel_correlation(0.02, "other_retail")
    assert 0.03 <= r <= 0.16
    # decrescente na PD (varejo)
    assert basel_correlation(0.001, "other_retail") > basel_correlation(0.20, "other_retail")
    with pytest.raises((ValueError, KeyError)):
        basel_correlation(0.02, "inexistente")


def test_basel_irb_capital_positive_and_portfolio():
    k = basel_irb_capital(0.02, 0.45, "other_retail")
    assert k > 0
    port = toy_portfolio()
    df = basel_capital_portfolio(port, {"cartao": "revolving", "consig": "other_retail"})
    assert "capital" in df.columns
    assert (df["capital"] >= 0).all()


# ----------------------------------------------------------------------
# Produtos
# ----------------------------------------------------------------------
def test_products_presets():
    nomes = list_presets()
    for p in ("cartao", "consignado", "veiculos"):
        assert p in nomes
    assert preset("cartao").basel_class == "revolving"
    assert preset("imobiliario").horizonte_1ano_ok is False
    fr = presets_frame()
    assert len(fr) == len(nomes) >= 6


def test_apply_preset_builds_segment():
    # apply_preset devolve só kwargs aceitos por Segment (sem chaves 'recomenda_*').
    from yggdrasil.credit_risk.capital import apply_preset
    seg = Segment(name="cartao_x", pd=0.12, lgd=0.75, ead=1e6, **apply_preset("cartao"))
    assert seg.product == "cartao" and seg.rho is not None


def test_moments_estimator_deterministic():
    # A CDF bivariada agora é determinística (integral 1-D, sem quasi-MC).
    rng = np.random.default_rng(0)
    dr = rng.uniform(0.01, 0.05, 60)
    assert asset_correlation_moments(dr) == asset_correlation_moments(dr)


# ----------------------------------------------------------------------
# Migração / CreditMetrics
# ----------------------------------------------------------------------
def test_migration_default_only_reproduces_bernoulli():
    tm, ratings, _ = two_state_matrix(0.03)
    mm = MigrationModel(tm, ratings, np.array([1.0, 1.0 - 0.45]), rho=0.08)
    exp = np.array([1e6] * 800)
    dist = mm.simulate(exp, np.zeros(len(exp), dtype=int), n_scenarios=40_000, q=0.999, seed=2)
    # EL ≈ PD·LGD·EAD_total
    assert dist.el == pytest.approx(0.03 * 0.45 * exp.sum(), rel=0.1)


def test_migration_rho_fattens_tail():
    tm, ratings, _ = two_state_matrix(0.03)
    vals = np.array([1.0, 1.0 - 0.45])
    exp = np.array([1e6] * 800)
    idx = np.zeros(len(exp), dtype=int)
    lo = MigrationModel(tm, ratings, vals, rho=0.02).simulate(exp, idx, n_scenarios=40_000, seed=1).var(0.999)
    hi = MigrationModel(tm, ratings, vals, rho=0.25).simulate(exp, idx, n_scenarios=40_000, seed=1).var(0.999)
    assert hi > lo


def test_migration_reference_is_origin_not_best():
    # Exposições começam num rating INTERMEDIÁRIO (B): upgrades geram ganho
    # (perda negativa) e a EL bate com a fórmula analítica relativa à origem —
    # prova de que a referência mark-to-market é o valor de ORIGEM, não o melhor.
    tm = np.array([[0.90, 0.08, 0.02], [0.10, 0.80, 0.10], [0.0, 0.0, 1.0]])
    vals = np.array([1.0, 0.9, 0.55])                  # A, B, D
    mm = MigrationModel(tm, ["A", "B", "D"], vals, rho=0.10)
    exp = np.array([1e6] * 500)
    dist = mm.simulate(exp, np.ones(len(exp), dtype=int), n_scenarios=40_000, q=0.999, seed=1)
    # EL/unidade = 0.10*(0.9-1.0) + 0.80*0 + 0.10*(0.9-0.55) = 0.025
    assert dist.el == pytest.approx(0.025 * exp.sum(), rel=1e-6)
    assert dist.losses.min() < 0                       # há cenários de ganho (upgrade p/ A)


# ----------------------------------------------------------------------
# Monte Carlo: cópula t e LGD beta
# ----------------------------------------------------------------------
def test_mc_defaults_reproduce_legacy_path():
    """Regressão: defaults (gaussiana + normal-clipada) reproduzem o algoritmo
    histórico bit a bit — mesma ordem de sorteios, mesmas contas."""
    port = Portfolio([
        Segment("a", pd=0.05, lgd=0.60, ead=2e6, rho=0.12, n_obligors=10_000,
                factor="F1", lgd_vol=0.20),
        Segment("b", pd=0.02, lgd=0.40, ead=3e6, rho=0.06, n_obligors=10_000,
                factor="F2", lgd_vol=0.0),
    ], factor_corr=[[1.0, 0.3], [0.3, 1.0]], factor_names=["F1", "F2"])
    n, seed, c = 5_000, 42, 0.4
    sim = port.simulate(n, q=0.999, seed=seed, stochastic_lgd=True, pd_lgd_corr=c)

    # réplica manual do algoritmo pré-cópula (um bloco, granular)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(np.asarray(port.factor_corr, dtype=float))
    Y = (rng.standard_normal((n, 2)) @ L.T)[:, port.factor_of()]
    inv_pd = norm.ppf(np.clip(port.pds(), 1e-12, 1 - 1e-12))
    rhos = port.rhos()
    cond = norm.cdf((inv_pd[None, :] - np.sqrt(rhos)[None, :] * Y)
                    / np.sqrt(1.0 - rhos)[None, :])
    zeta = rng.standard_normal((n, 2))
    lat = -c * Y + np.sqrt(1.0 - c ** 2) * zeta
    lgd_eff = np.clip(port.lgds()[None, :] + port.lgd_vols()[None, :] * lat, 0.0, 1.0)
    esperado = (cond * lgd_eff * port.eads()[None, :]).sum(axis=1)
    np.testing.assert_array_equal(sim.losses, esperado)

    # explicitar os defaults novos não muda nada (mesmo fluxo do RNG)
    sim2 = port.simulate(n, q=0.999, seed=seed, stochastic_lgd=True, pd_lgd_corr=c,
                         copula="gaussian", lgd_dist="normal")
    np.testing.assert_array_equal(sim.losses, sim2.losses)


def test_mc_t_copula_fattens_tail_and_preserves_el():
    port = toy_portfolio()
    n, seed = 100_000, 7
    g = port.simulate(n, q=0.999, seed=seed)
    t4 = port.simulate(n, q=0.999, seed=seed, copula="t", t_dof=4.0)
    # ν baixo ⇒ dependência de cauda ⇒ VaR 99,9% maior que o gaussiano
    assert t4.var() > g.var()
    # limiar t_ν⁻¹(PD) preserva a PD incondicional ⇒ EL empírica ≈ analítica
    assert t4.losses.mean() == pytest.approx(port.expected_loss(), rel=0.05)


def test_mc_beta_lgd_bounds_and_no_atoms():
    """LGD beta respeita [0, 1] com média/desvio casados e sem a massa
    artificial em 0/1 que o clipping da normal cria."""
    # pd≈1 e rho=0 ⇒ fração de default ~1: a perda por cenário revela a LGD.
    def _lgds(dist):
        p = Portfolio([Segment("s", pd=1.0, lgd=0.5, ead=1.0, rho=0.0,
                               n_obligors=10_000, factor="F", lgd_vol=0.3)])
        return p.simulate(40_000, q=0.999, seed=3, stochastic_lgd=True,
                          pd_lgd_corr=0.4, lgd_dist=dist).losses

    lgd_beta, lgd_norm = _lgds("beta"), _lgds("normal")
    atoms = lambda x: float(np.mean((x <= 1e-9) | (x >= 1.0 - 1e-9)))
    assert lgd_beta.min() >= 0.0 and lgd_beta.max() <= 1.0
    # moment matching: média e desvio reproduzidos
    assert lgd_beta.mean() == pytest.approx(0.5, abs=0.01)
    assert lgd_beta.std() == pytest.approx(0.3, abs=0.01)
    # normal-clipada concentra massa nas bordas; a beta não
    assert atoms(lgd_norm) > 0.02
    assert atoms(lgd_beta) < 1e-3


def test_mc_t_copula_with_beta_lgd_smoke():
    # combinação t + beta: perdas finitas, não-negativas e limitadas por LGD·EAD
    port = Portfolio([Segment("veic", pd=0.03, lgd=0.40, ead=5e6, rho=0.08,
                              n_obligors=40_000, factor="veic", lgd_vol=0.25)])
    sim = port.simulate(30_000, q=0.999, seed=1, copula="t", t_dof=6.0,
                        stochastic_lgd=True, pd_lgd_corr=0.3, lgd_dist="beta")
    assert np.isfinite(sim.losses).all()
    assert sim.losses.min() >= 0.0
    assert sim.losses.max() <= 5e6  # perda ≤ EAD (LGD ∈ [0, 1], fração ≤ 1)


def test_mc_copula_params_validated():
    port = toy_portfolio()
    with pytest.raises(ValueError):
        port.simulate(1_000, seed=0, copula="clayton")
    with pytest.raises(ValueError):
        port.simulate(1_000, seed=0, copula="t", t_dof=0.0)
    with pytest.raises(ValueError):
        port.simulate(1_000, seed=0, lgd_dist="lognormal")


# ----------------------------------------------------------------------
# Visualizações (matplotlib Agg)
# ----------------------------------------------------------------------
def test_report_plots_return_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from yggdrasil.credit_risk.capital import report

    port = toy_portfolio()
    sim = port.simulate(30_000, q=0.999, seed=1)
    for fig in (report.plot_loss_distribution(sim, q=0.999),
                report.plot_allocation(sim.allocate())):
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ----------------------------------------------------------------------
# Registro no MLflow
# ----------------------------------------------------------------------
def test_log_capital_run(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None, raising=False)
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())

    from yggdrasil.credit_risk.capital import log_capital_run

    port = toy_portfolio()
    sim = port.simulate(20_000, q=0.999, seed=1)
    run_id = log_capital_run(port, sim, allocation=sim.allocate(),
                             asrf=port.asrf_capital(q=0.999),
                             experiment="/tmp/test_capital")
    run = mlflow.get_run(run_id)
    # métricas essenciais logadas
    for k in ("EL", "VaR", "ES", "CE_var"):
        assert k in run.data.metrics
    # as duas figuras foram de fato geradas e anexadas (regressão do bug de path)
    figs = {a.path for a in mlflow.artifacts.list_artifacts(run_id=run_id, artifact_path="figures")}
    assert "figures/loss_distribution.png" in figs
    assert "figures/allocation.png" in figs
