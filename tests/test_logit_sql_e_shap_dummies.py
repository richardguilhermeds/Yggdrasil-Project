"""Fórmula da logística em SQL (score 0–1000 + probabilidade), SHAP com as
dummies de scorecard como UMA variável e rótulos distintos por versão."""
from __future__ import annotations

import contextlib
import io
import math
import sqlite3
import warnings

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.model import ModelSegmenter


def _df(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.beta(2.5, 3, n) * 1.4 + 0.3
    score[rng.random(n) < .05] = np.nan
    gar = rng.choice(["A", "B", "C", "D", "O'Neil"], n).astype(object)
    gar[rng.random(n) < .03] = None
    uf = rng.choice(["SP", "RJ", "MG"], n).astype(object)
    flag = rng.random(n) < .3
    renda = rng.gamma(2, 1500, n)
    renda[rng.random(n) < .04] = np.nan
    risco = {"A": -.8, "B": -.2, "C": .3, "D": .9, "O'Neil": .5}
    logit = (-2 + 2.2 * (np.nan_to_num(score, nan=1.2) - 0.9)
             + np.array([risco.get(g, .4) for g in gar]) + .5 * flag)
    return pd.DataFrame({"score_b": score, "garantia": gar, "uf": uf, "flag": flag,
                         "renda": renda,
                         "target": rng.binomial(1, 1 / (1 + np.exp(-logit))).astype(float)})


def _seg(df):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = ModelSegmenter(df, target="target")
    seg.set_manual_bins("score_b", "0.7, 1.0, 1.3", missing="pior")
    seg.set_scorecard_dummies("score_b")
    seg.set_manual_bins("garantia", [["A"], ["B"], ["C", "D", "O'Neil"]])
    return seg


def _fit(seg, **kw):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.fit(**kw)


def _roda_sql(sql, df):
    con = sqlite3.connect(":memory:")
    con.create_function("EXP", 1, math.exp)
    df.drop(columns="target").to_sql("minha_tabela", con, index=False)
    return pd.read_sql(sql.rstrip(";"), con)


# ───────────────────────── fórmula da logística ─────────────────────────
@pytest.mark.parametrize("transform", ["raw", "woe"])
def test_logit_sql_reproduz_probabilidade_e_score(transform):
    df = _df()
    seg = _seg(df)
    cat = seg.create_categorical("garantia", dummies=True)
    feats = ["score_b", "garantia", "uf", "flag", "renda", cat]
    _fit(seg, features=feats, transform=transform)
    r = _roda_sql(seg.logit_sql(), df)
    assert np.abs(r["probabilidade"].to_numpy() - seg.score_.to_numpy()).max() < 1e-9
    assert np.abs(r["score"].to_numpy() - seg.score_points_.to_numpy()).max() < 1e-6
    assert r["score"].between(0, 1000).all()


@pytest.mark.parametrize("metodo", ["intercept", "platt", "isotonic"])
def test_logit_sql_com_calibracao(metodo):
    df = _df()
    seg = _seg(df)
    _fit(seg, features=["score_b", "garantia", "renda"], transform="woe")
    with contextlib.redirect_stdout(io.StringIO()):
        seg.calibrate(method=metodo)
    r = _roda_sql(seg.logit_sql(), df)
    assert np.abs(r["probabilidade"].to_numpy() - seg.score_.to_numpy()).max() < 1e-9


def test_logit_sql_so_para_logistica():
    df = _df(n=1500)
    seg = _seg(df)
    _fit(seg, algorithm="random_forest", features=["score_b", "renda"])
    with pytest.raises(ValueError, match="logística"):
        seg.logit_sql()


# ───────────────────────── SHAP: dummies como uma variável ─────────────────────────
def test_beeswarm_agrupa_dummies_de_scorecard():
    df = _df(n=2000)
    seg = _seg(df)
    _fit(seg, features=["score_b", "renda", "uf"], transform="raw")
    sv, Xs = seg.shap_values(sample_size=300)
    sv2, Xs2 = seg._shap_agrupa_dummies(sv, Xs)
    dums = [c for c in Xs.columns if str(c).startswith("dum__score_b=")]
    assert len(dums) == 3
    assert "dum__score_b" in Xs2.columns and not any("=" in str(c) for c in Xs2.columns)
    j = list(Xs2.columns).index("dum__score_b")
    idx = [list(Xs.columns).index(c) for c in dums]
    assert np.allclose(sv2[:, j], np.asarray(sv)[:, idx].sum(axis=1))
    assert set(np.unique(Xs2["dum__score_b"])) <= {0.0, 1.0, 2.0, 3.0}
    assert seg._display_feature_name("dum__score_b") == "score_b (faixas)"
    import matplotlib
    matplotlib.use("Agg")
    fig = seg.plot_shap_beeswarm(sample_size=300)
    rotulos = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert "score_b (faixas)" in rotulos and not any(" = (" in r for r in rotulos)
    assert "score_b" in list(seg.shap_importance_grouped(sample_size=300)["variavel"])


# ───────────────────────── rótulos por versão ─────────────────────────
def test_versoes_da_mesma_variavel_tem_rotulos_distintos():
    seg = _seg(_df(n=1500))
    a = seg.create_categorical("garantia", dummies=True)
    b = seg.create_categorical("garantia", dummies=False)
    c = seg.create_categorical("garantia", dummies=True)
    d = seg.create_categorical("garantia", new_name="garantia_sc_v4", dummies=True)
    assert [a, b, c, d] == ["garantia_cat", "garantia_cat_2", "garantia_cat_3", "garantia_sc_v4"]
    rotulos = [seg.label(x) for x in (a, b, c, d)]
    assert rotulos == ["garantia (dummies)", "garantia (cat. v2)", "garantia (dummies v3)",
                       "garantia_sc_v4"]
    assert len(set(rotulos)) == 4
    assert seg._nome_livre("garantia_cat") == "garantia_cat_4"
    # faixa de derivada sem chaves duplas
    assert "{{" not in " ".join(seg.scorecard_table(a)["faixa"])


# ───────────────────────── UI ─────────────────────────
@pytest.fixture
def ui():
    pytest.importorskip("ipywidgets")
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    df = _df(n=2500)
    df["dt_ref"] = pd.Timestamp("2024-01-01")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ModelSegmenterUI(df, target="target", task_type="classification",
                                date_col="dt_ref")


def test_ui_card_da_formula_so_com_logistica(ui):
    assert ui.card_logit_sql.layout.display == "none"
    ui.dd_algo.value = "logistica"
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui._on_fit(None)
    assert ui.card_logit_sql.layout.display == ""
    ui._on_logit_sql(None)
    assert "AS probabilidade" in ui.out_logit_sql.value and "AS score" in ui.out_logit_sql.value


def test_ui_placeholder_sugere_proxima_versao(ui):
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_var2.value = "garantia"
    ui.seg.set_manual_bins("garantia", "A; B; C, D")
    ui._render_bin_hint("garantia")
    assert ui.tx_new_cat.placeholder == "garantia_cat"
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_create_cat(None)
    assert ui.tx_new_cat.placeholder == "garantia_cat_2"
