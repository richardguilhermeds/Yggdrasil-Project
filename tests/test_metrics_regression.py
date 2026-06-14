"""Testes das métricas de regressão."""

import numpy as np

from yggdrasil.metrics import regression_metrics
from yggdrasil.metrics.regression import robust_mape


def test_previsao_perfeita():
    y = np.array([0.1, 0.5, 0.9, 1.2])
    m = regression_metrics(y, y)
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0
    assert m["mean_bias"] == 0.0


def test_vies_positivo_quando_preve_acima():
    y = np.array([1.0, 2.0, 3.0])
    yhat = y + 0.5
    m = regression_metrics(y, yhat)
    assert m["mean_bias"] > 0
    assert np.isclose(m["mae"], 0.5)


def test_mape_robusto_ignora_zeros():
    y = np.array([0.0, 0.0, 1.0, 2.0])
    yhat = np.array([5.0, 9.0, 1.0, 2.0])  # erro só nos zeros (descartados)
    assert robust_mape(y, yhat) == 0.0


def test_chaves_presentes():
    rng = np.random.default_rng(0)
    y = rng.random(200)
    yhat = y + rng.normal(0, 0.1, 200)
    m = regression_metrics(y, yhat)
    for chave in ["rmse", "mae", "mape", "smape", "medae", "r2", "mean_bias"]:
        assert chave in m
