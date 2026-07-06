"""Testes das features de seleção de variáveis e reprodutibilidade do ModelSegmenter:

- ``random_state``: default 42, ``None``→42, propagação reprodutível ao modelo,
  persistência em ``to_dict``/``from_dict`` (legado → 42);
- ``backward_elimination(features=...)`` + âncora ``attrs['feats0']``;
- ``apply_backward_selection``: critérios parcimônia/best, direção pela MÉTRICA
  (não pelo task_type), reconstrução ancorada por identidade (idempotente),
  guard de resultado legado divergente, e reprojeção de régua manual após refit.

Cobrem os defeitos apontados na revisão adversarial (identidade vs contagem;
higher_better por métrica; rating_ defasado após refit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.model.segmenter import ModelSegmenter


def _clf_df(n=900, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.normal(size=n),
        "x3": rng.normal(size=n), "x4": rng.normal(size=n),
        "amostra": np.where(np.arange(n) < int(0.72 * n), "DES", "OOT"),
    })
    p = 1 / (1 + np.exp(-(1.6 * df.x1 - 1.1 * df.x2 + 0.02 * df.x3)))
    df["target"] = (rng.uniform(size=n) < p).astype(int)
    return df


def _seg(**kw):
    return ModelSegmenter(_clf_df(), target="target", sample_col="amostra",
                          verbose=False, **kw)


# ------------------------------------------------------------------ random_state
def test_random_state_default_and_none():
    assert _seg().random_state == 42
    assert _seg(random_state=None).random_state == 42
    assert _seg(random_state=7).random_state == 7


def test_random_state_reproducivel_no_modelo():
    a, b, c = _seg(random_state=123), _seg(random_state=123), _seg(random_state=999)
    a.fit("random_forest"); b.fit("random_forest"); c.fit("random_forest")
    assert np.allclose(a.score_.to_numpy(), b.score_.to_numpy())      # mesma seed → igual
    assert not np.allclose(a.score_.to_numpy(), c.score_.to_numpy())  # seed diferente → difere


def test_random_state_nao_sobrescreve_hyperparam_do_usuario():
    # setdefault preserva a precedência usuário > segmenter
    seg = _seg(random_state=1)
    seg.fit("random_forest", hyperparams={"random_state": 5, "n_estimators": 10})
    assert seg.model.named_steps["est"].get_params()["random_state"] == 5


def test_random_state_persistido_e_legado():
    d = _seg(random_state=7).to_dict()
    assert d["meta"]["random_state"] == 7
    df = _clf_df()
    assert ModelSegmenter.from_dict(d, df).random_state == 7
    d_legacy = {**d, "meta": {k: v for k, v in d["meta"].items() if k != "random_state"}}
    assert ModelSegmenter.from_dict(d_legacy, df).random_state == 42


# ------------------------------------------------------- backward + apply (core)
def test_backward_ancora_feats0_e_features_param():
    seg = _seg(); seg.fit("random_forest")
    res = seg.backward_elimination(min_features=1)
    assert set(res.attrs["feats0"]) == {"x1", "x2", "x3", "x4"}
    res2 = seg.backward_elimination(min_features=1, features=["x1", "x2"])
    assert set(res2.attrs["feats0"]) == {"x1", "x2"}
    assert int(res2["n_variaveis"].max()) == 2


def test_apply_parsimonia_escolhe_subconjunto_informativo():
    seg = _seg(); seg.fit("random_forest")
    res = seg.backward_elimination(min_features=1)
    info = seg.apply_backward_selection(res, criterion="parsimony", tol=0.01,
                                        refit=True, rebuild_ratings=False)
    assert info["target_n"] == len(info["features"])
    assert seg.included == set(info["features"])
    assert set(seg.model_features) == set(info["features"])
    # as variáveis informativas (x1, x2) devem sobreviver à poda
    assert {"x1", "x2"}.issubset(set(info["features"]))


def test_apply_ancorado_e_idempotente():
    seg = _seg(); seg.fit("random_forest")
    res = seg.backward_elimination(min_features=1)
    info = seg.apply_backward_selection(res, criterion="parsimony", refit=True)
    # após o refit model_features encolheu; reaplicar o MESMO res reconstrói do
    # feats0 ANCORADO (não do estado atual) → mesmo subconjunto, sem erro
    again = seg.apply_backward_selection(res, criterion="parsimony", refit=False)
    assert set(again["features"]) == set(info["features"])


def test_apply_guard_legado_divergente():
    seg = _seg(); seg.fit("random_forest")
    res_legacy = pd.DataFrame({"n_variaveis": [5, 4, 3, 2, 1],
                               "removida": ["z", "y", "x", "w", "—"],
                               "ks": [0.50, 0.52, 0.53, 0.51, 0.40]})  # sem attrs['feats0']
    with pytest.raises(RuntimeError):
        seg.apply_backward_selection(res_legacy)   # feats0 cai p/ estado atual (4) != 5


def test_apply_direcao_pela_metrica_regressao():
    # regressão: r2 é higher-is-better; rmse é lower-is-better — a direção deve vir
    # da MÉTRICA, não do task_type (senão r2 escolheria o pior passo)
    rng = np.random.RandomState(1)
    dfR = pd.DataFrame({c: rng.normal(size=300) for c in "abcd"})
    dfR["amostra"] = "DES"; dfR["y"] = 2 * dfR.a - dfR.b + rng.normal(size=300) * 0.1
    segR = ModelSegmenter(dfR, target="y", task_type="regression",
                          sample_col="amostra", verbose=False)
    res = pd.DataFrame({"n_variaveis": [4, 3, 2, 1], "removida": ["c", "d", "b", "—"],
                        "r2": [0.90, 0.92, 0.88, 0.50], "rmse": [0.30, 0.28, 0.33, 0.60]})
    res.attrs["feats0"] = ["a", "b", "c", "d"]
    assert segR.apply_backward_selection(res, metric="r2", criterion="best",
                                         refit=False)["target_n"] == 3   # maior r2
    assert segR.apply_backward_selection(res, metric="rmse", criterion="best",
                                         refit=False)["target_n"] == 3   # menor rmse


def test_apply_reprojeta_regua_manual_apos_refit():
    seg = _seg(); seg.fit("random_forest")
    seg.build_ratings(method="manual_score", cuts=[0.3, 0.5, 0.7])   # escala crua 0–1
    res = seg.backward_elimination(min_features=1)
    info = seg.apply_backward_selection(res, criterion="parsimony", refit=True,
                                        rebuild_ratings=True)
    assert info["ratings_rebuilt"] is False       # régua manual NÃO é regenerada
    assert info["ratings_reprojected"] is True    # mas É reprojetada no novo score
    assert seg.rating_ is not None and len(seg.rating_) == len(seg.df)
    assert seg.rating_config.get("method") == "manual_score"
