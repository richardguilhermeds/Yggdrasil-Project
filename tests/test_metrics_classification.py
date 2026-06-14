"""Testes das métricas de classificação."""

import numpy as np

from yggdrasil.metrics import classification_metrics, ks_optimal_cutoff, ks_statistic


def test_separacao_perfeita():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = classification_metrics(y, score)
    assert m["auc"] == 1.0
    assert m["ks"] == 1.0
    assert m["gini"] == 1.0
    assert m["accuracy"] == 1.0


def test_score_aleatorio_auc_em_torno_de_meio():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=4000)
    score = rng.random(4000)
    m = classification_metrics(y, score)
    assert 0.4 < m["auc"] < 0.6


def test_chaves_presentes():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=500)
    score = rng.random(500)
    m = classification_metrics(y, score)
    for chave in ["auc", "gini", "ks", "accuracy", "f1", "precision", "recall",
                  "brier", "logloss", "ks_cutoff"]:
        assert chave in m


def test_ks_statistic_uma_classe_e_nan():
    y = np.zeros(10)
    score = np.linspace(0, 1, 10)
    assert np.isnan(ks_statistic(y, score))


def test_cutoff_otimo_entre_0_e_1():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.2, 0.4, 0.6, 0.8])
    corte = ks_optimal_cutoff(y, score)
    assert 0.0 <= corte <= 1.0
