"""Testes das métricas de calibração (IC binomial, CITL, slope/intercept)."""

import numpy as np
import pytest

from yggdrasil.metrics import (
    binomial_ci,
    calibration_in_the_large,
    calibration_slope_intercept,
    reliability_table,
)


def test_binomial_ci_jeffreys_contem_taxa():
    k, n = 30, 100
    inf, sup = binomial_ci(k, n, alpha=0.05, method="jeffreys")
    assert 0.0 <= inf < k / n < sup <= 1.0


def test_binomial_ci_clopper_pearson_contem_taxa():
    k, n = 30, 100
    inf, sup = binomial_ci(k, n, alpha=0.05, method="clopper-pearson")
    assert 0.0 <= inf < k / n < sup <= 1.0
    # Clopper-Pearson (exato) é mais conservador que Jeffreys
    j_inf, j_sup = binomial_ci(k, n, method="jeffreys")
    assert inf <= j_inf and sup >= j_sup


def test_binomial_ci_bordas_e_invalido():
    inf0, _ = binomial_ci(0, 50)
    _, sup1 = binomial_ci(50, 50)
    assert inf0 == 0.0
    assert sup1 == 1.0
    inf_nan, sup_nan = binomial_ci(0, 0)          # n inválido → NaN
    assert np.isnan(inf_nan) and np.isnan(sup_nan)


def test_binomial_ci_vetorial():
    inf, sup = binomial_ci([2, 30, 98], 100)
    assert isinstance(inf, np.ndarray) and len(inf) == 3
    assert (inf < sup).all()
    assert (np.diff(inf) > 0).all()               # taxas maiores → limites maiores


def test_binomial_ci_metodo_desconhecido():
    with pytest.raises(ValueError):
        binomial_ci(5, 10, method="wald")


def test_calibration_in_the_large():
    y = [0, 1, 0, 1]
    p = [0.25, 0.25, 0.25, 0.25]
    assert np.isclose(calibration_in_the_large(y, p), 2.0)  # observado 0.5 / previsto 0.25
    assert np.isnan(calibration_in_the_large([], []))


def test_calibration_slope_intercept_bem_calibrado():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, size=20000)
    y = rng.binomial(1, p)                         # score = probabilidade verdadeira
    slope, intercept = calibration_slope_intercept(y, p)
    assert np.isclose(slope, 1.0, atol=0.1)
    assert np.isclose(intercept, 0.0, atol=0.1)


def test_calibration_slope_intercept_alvo_nao_binario():
    slope, intercept = calibration_slope_intercept([0.2, 0.5, 0.7], [0.3, 0.4, 0.6])
    assert np.isnan(slope) and np.isnan(intercept)


def test_reliability_table_estrutura():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, size=5000)
    y = rng.binomial(1, p)
    tab = reliability_table(y, p, n_bins=10)
    for c in ["faixa", "n", "p_medio", "taxa_observada", "ic_inf", "ic_sup", "calibrado"]:
        assert c in tab.columns
    assert tab["n"].sum() == len(p)
    # IC contém a taxa observada da própria faixa (k/n razoável)
    ok = tab["n"] > 0
    assert (tab.loc[ok, "ic_inf"] <= tab.loc[ok, "taxa_observada"]).all()
    assert (tab.loc[ok, "taxa_observada"] <= tab.loc[ok, "ic_sup"]).all()
    # score bem calibrado por construção → maioria das faixas dentro do IC
    assert tab["calibrado"].mean() > 0.5
