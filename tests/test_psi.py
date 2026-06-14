"""Testes de PSI (numérico, categórico e ao longo do tempo)."""

import numpy as np

from yggdrasil.monitoring import (
    classify_psi,
    psi,
    psi_categorical,
    psi_rating_over_time,
)


def test_psi_zero_para_mesma_distribuicao():
    rng = np.random.default_rng(0)
    base = rng.normal(size=5000)
    igual = rng.normal(size=5000)
    assert psi(base, igual) < 0.05


def test_psi_alto_para_shift_grande():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, size=5000)
    deslocado = rng.normal(3, 1, size=5000)  # média muito diferente
    assert psi(base, deslocado) > 0.25


def test_psi_categorical_identico_e_zero():
    labels = ["A", "B", "C"] * 100
    assert psi_categorical(labels, labels) < 1e-6


def test_classify_psi_faixas():
    assert classify_psi(0.05) == "estavel"
    assert classify_psi(0.15) == "atencao"
    assert classify_psi(0.40) == "instavel"


def test_psi_rating_over_time_estrutura(scored_clf, cfg):
    df = scored_clf.df_scored
    col = scored_clf.rating_cols[0]
    ts = psi_rating_over_time(df, col, cfg)
    assert {"mes", "psi", "n", "flag"}.issubset(ts.columns)
    assert len(ts) >= 1
