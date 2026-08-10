"""Testes do IC bootstrap de métricas e da significância dos shifts."""

import numpy as np
import pandas as pd
import pytest

from yggdrasil.config import ColumnConfig
from yggdrasil.metrics import (bootstrap_metric_ci, metric_by_sample,
                               metric_shifts, shift_significance)


def _base_clf(n, seed=0, sinal=2.0):
    """Alvo binário com score correlacionado (sinal > 0 => AUC > 0.5)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    score = 1 / (1 + np.exp(-(sinal * (y - 0.5) + rng.normal(0, 1, n))))
    return y.astype(float), score


def test_ic_contem_o_ponto():
    y, score = _base_clf(800)
    for metric in ("auc", "gini", "ks"):
        r = bootstrap_metric_ci(y, score, metric=metric, n_boot=200, seed=1)
        assert r["ic_low"] <= r["valor"] <= r["ic_high"]
        assert r["se"] > 0


def test_ic_r2_regressao():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, 500)
    pred = y + rng.normal(0, 0.5, 500)
    r = bootstrap_metric_ci(y, pred, metric="r2", n_boot=200, seed=2)
    assert r["ic_low"] <= r["valor"] <= r["ic_high"]
    assert 0.0 < r["valor"] < 1.0


def test_seed_reprodutivel():
    y, score = _base_clf(400)
    r1 = bootstrap_metric_ci(y, score, metric="auc", n_boot=100, seed=42)
    r2 = bootstrap_metric_ci(y, score, metric="auc", n_boot=100, seed=42)
    assert r1 == r2


def test_amostra_maior_ic_mais_estreito():
    y_p, s_p = _base_clf(200, seed=5)
    y_g, s_g = _base_clf(5000, seed=5)
    r_p = bootstrap_metric_ci(y_p, s_p, metric="auc", n_boot=200, seed=7)
    r_g = bootstrap_metric_ci(y_g, s_g, metric="auc", n_boot=200, seed=7)
    assert (r_g["ic_high"] - r_g["ic_low"]) < (r_p["ic_high"] - r_p["ic_low"])


def test_metric_callable():
    y, score = _base_clf(300)
    r = bootstrap_metric_ci(
        y, score, metric=lambda yt, ys: float(np.mean(ys)), n_boot=100, seed=0
    )
    assert r["ic_low"] <= r["valor"] <= r["ic_high"]


def test_metric_invalida_levanta_erro():
    y, score = _base_clf(100)
    with pytest.raises(ValueError, match="metric"):
        bootstrap_metric_ci(y, score, metric="metrica_inventada")


def test_entrada_vazia_devolve_nan():
    r = bootstrap_metric_ci([], [], metric="auc")
    assert all(np.isnan(v) for v in r.values())


def _df_amostras(n_des=600, n_oot=400, sinal_oot=2.0):
    y_des, s_des = _base_clf(n_des, seed=10)
    y_oot, s_oot = _base_clf(n_oot, seed=11, sinal=sinal_oot)
    return pd.DataFrame({
        "amostra": ["DES"] * n_des + ["OOT"] * n_oot,
        "target": np.concatenate([y_des, y_oot]),
        "prediction": np.concatenate([s_des, s_oot]),
        "dt_ref": "2024-01",
        "feat_x": 0.0,
    })


def test_metric_by_sample_with_ci_anexa_colunas():
    df = _df_amostras()
    cfg = ColumnConfig()
    sem_ci = metric_by_sample(df, cfg, "classification")
    com_ci = metric_by_sample(
        df, cfg, "classification", with_ci=True, n_boot=50, seed=0
    )
    for amostra in ("DES", "OOT"):
        assert "auc_ic_low" not in sem_ci[amostra]  # default inalterado
        for m in ("auc", "gini", "ks"):
            r = com_ci[amostra]
            assert r[f"{m}_ic_low"] <= r[m] <= r[f"{m}_ic_high"]
            assert f"{m}_se" in r


def test_metric_shifts_ignora_colunas_de_ic():
    df = _df_amostras()
    com_ci = metric_by_sample(
        df, ColumnConfig(), "classification", with_ci=True, n_boot=50, seed=0
    )
    s = metric_shifts(com_ci["DES"], com_ci["OOT"])
    # sem skip, apareceriam chaves como 'auc_ic_low_shift_abs'/'auc_se_shift_abs'
    assert not any("_ic_" in k or "_se_shift" in k for k in s)
    assert "auc_shift_abs" in s


def test_shift_significance_degradacao_real():
    y_des, s_des = _base_clf(1500, seed=20, sinal=2.5)   # AUC alta
    rng = np.random.default_rng(21)
    y_oot = rng.integers(0, 2, size=1500).astype(float)
    s_oot = rng.random(1500)                             # score aleatório
    r = shift_significance(y_des, s_des, y_oot, s_oot, metric="auc",
                           n_boot=200, seed=0)
    assert r["significancia"] == "degradacao_real"
    assert r["flag"] == "degradado"
    assert r["ic_high_cmp"] < r["ic_low_ref"]


def test_shift_significance_dentro_do_ruido():
    y, score = _base_clf(2000, seed=30)
    meio = 1000  # duas metades da mesma distribuição: shift é só ruído
    r = shift_significance(y[:meio], score[:meio], y[meio:], score[meio:],
                           metric="auc", n_boot=200, seed=0)
    assert r["significancia"] == "dentro_do_ruido"


def test_shift_significance_callable_exige_direcao():
    y, score = _base_clf(200)
    metrica = lambda yt, ys: float(np.mean(ys))  # noqa: E731
    with pytest.raises(ValueError, match="higher_is_better"):
        shift_significance(y, score, y, score, metric=metrica)
    r = shift_significance(y, score, y, score, metric=metrica,
                           n_boot=50, seed=0, higher_is_better=True)
    assert r["significancia"] == "dentro_do_ruido"
