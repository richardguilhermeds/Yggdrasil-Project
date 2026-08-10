"""Testes de lift_table, cap_curve e accuracy_ratio."""

import numpy as np
import pytest

from yggdrasil.metrics import (accuracy_ratio, cap_curve, classification_metrics,
                               lift_table)


def _score_perfeito(taxa=0.1, n=100, seed=0):
    """Alvo com ``taxa`` de eventos e score contínuo que os ordena perfeitamente."""
    rng = np.random.default_rng(seed)
    n_ev = int(n * taxa)
    y = np.array([1] * n_ev + [0] * (n - n_ev))
    score = np.concatenate([rng.uniform(0.8, 1.0, n_ev),
                            rng.uniform(0.0, 0.5, n - n_ev)])
    return y, score


def test_score_perfeito_ar_1_e_captura_total_no_primeiro_decil():
    # 10% de eventos → com score perfeito o 1º decil captura 100% dos eventos
    y, score = _score_perfeito(taxa=0.1, n=100)
    assert accuracy_ratio(y, score) == pytest.approx(1.0)
    tab = lift_table(y, score, n_bins=10)
    assert tab["captura_acum"].iloc[0] == pytest.approx(1.0)
    assert tab["lift"].iloc[0] == pytest.approx(10.0)   # taxa 1,0 / taxa geral 0,1


def test_score_aleatorio_ar_zero_e_lift_um():
    rng = np.random.default_rng(7)
    n = 20000
    y = (rng.random(n) < 0.3).astype(int)
    score = rng.random(n)
    assert abs(accuracy_ratio(y, score)) < 0.05
    tab = lift_table(y, score, n_bins=10)
    assert np.all(np.abs(tab["lift"].to_numpy() - 1.0) < 0.15)


def test_accuracy_ratio_consistente_com_gini():
    rng = np.random.default_rng(3)
    n = 5000
    score = rng.random(n)                               # contínuo: sem empates
    y = (rng.random(n) < score).astype(int)             # sinal real no score
    m = classification_metrics(y, score)
    assert accuracy_ratio(y, score) == pytest.approx(m["gini"], abs=1e-6)
    assert m["accuracy_ratio"] == m["gini"]             # aditivo no dict


def test_lift_table_estrutura_e_acumulados():
    y, score = _score_perfeito(taxa=0.2, n=250, seed=5)
    tab = lift_table(y, score, n_bins=10)
    assert list(tab.columns) == ["faixa", "n", "n_eventos", "taxa_evento",
                                 "lift", "pop_acum", "captura_acum"]
    assert len(tab) == 10
    assert int(tab["n"].sum()) == 250
    assert int(tab["n_eventos"].sum()) == int(y.sum())
    assert tab["pop_acum"].iloc[-1] == pytest.approx(1.0)
    assert tab["captura_acum"].iloc[-1] == pytest.approx(1.0)
    assert tab["captura_acum"].is_monotonic_increasing


def test_cap_curve_extremos_e_monotonia():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, size=400)
    score = rng.random(400)
    x, yc = cap_curve(y, score)
    assert x[0] == 0.0 and yc[0] == 0.0
    assert x[-1] == pytest.approx(1.0) and yc[-1] == pytest.approx(1.0)
    assert np.all(np.diff(yc) >= 0)


def test_degenerados():
    # uma classe só → AR NaN; lift_table ainda sai (lift NaN se taxa geral = 0)
    y = np.zeros(30)
    score = np.linspace(0, 1, 30)
    assert np.isnan(accuracy_ratio(y, score))
    tab = lift_table(y, score, n_bins=5)
    assert len(tab) == 5 and tab["lift"].isna().all()
    vazia = lift_table([], [])
    assert vazia.empty
