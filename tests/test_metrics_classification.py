"""Testes das métricas de classificação."""

import numpy as np

from yggdrasil.metrics import (classification_metrics, ks_optimal_cutoff,
                               ks_statistic, metric_shifts)
from yggdrasil.metrics.shift import shift_flag


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


def test_shift_flags_maior_melhor():
    ref = {"auc": 0.80, "ks": 0.50, "gini": 0.60}
    cmp = {"auc": 0.60, "ks": 0.44, "gini": 0.58}
    s = metric_shifts(ref, cmp)
    assert s["auc_shift_flag"] == "degradado"   # queda relativa de 25%
    assert s["ks_shift_flag"] == "atencao"      # queda relativa de 12%
    assert s["gini_shift_flag"] == "ok"         # queda relativa de ~3%


def test_shift_flags_metricas_de_erro_por_aumento():
    ref = {"brier": 0.10, "logloss": 0.30}
    cmp = {"brier": 0.13, "logloss": 0.27}
    s = metric_shifts(ref, cmp)
    assert s["brier_shift_flag"] == "degradado"  # erro subiu 30%
    assert s["logloss_shift_flag"] == "ok"       # erro caiu (melhora)


def test_shift_flags_limiares_configuraveis():
    ref, cmp = {"auc": 0.80}, {"auc": 0.60}
    s = metric_shifts(ref, cmp, flag_atencao=0.30, flag_degradado=0.50)
    assert s["auc_shift_flag"] == "ok"


def test_ks_cutoff_sem_shift_nem_flag():
    s = metric_shifts({"ks_cutoff": 0.5, "ks": 0.5}, {"ks_cutoff": 0.7, "ks": 0.5})
    assert not any(k.startswith("ks_cutoff") for k in s)
    assert s["ks_shift_flag"] == "ok"


def test_shift_flag_metrica_desconhecida_ou_ref_zero():
    assert shift_flag("metrica_inventada", 1.0, 0.5) is None
    assert shift_flag("auc", 0.0, 0.5) is None  # referência zero: não computável
