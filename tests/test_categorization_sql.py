"""SQL da categorização (``categorization_sql``): rodado num SQLite, tem de bater
linha a linha com o Python — faixas, faltantes alocados, WoE, dummies de
scorecard e variáveis derivadas."""
from __future__ import annotations

import contextlib
import io
import sqlite3
import warnings

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.model import ModelSegmenter


def _df(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.beta(2.5, 3, n) * 1.4 + 0.3
    score[rng.random(n) < .05] = np.nan
    gar = rng.choice(["A", "B", "C", "D", "O'Neil"], n).astype(object)
    gar[rng.random(n) < .03] = None
    flag = rng.random(n) < .3
    risco = {"A": -.8, "B": -.2, "C": .3, "D": .9, "O'Neil": .5}
    logit = (-2 + 2.2 * (np.nan_to_num(score, nan=1.2) - 0.9)
             + np.array([risco.get(g, .4) for g in gar]) + .5 * flag)
    return pd.DataFrame({"score": score, "garantia": gar, "flag": flag,
                         "renda": rng.gamma(2, 1500, n),
                         "target": rng.binomial(1, 1 / (1 + np.exp(-logit))).astype(float)})


@pytest.fixture(scope="module")
def cenario():
    df = _df()
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = ModelSegmenter(df, target="target")
        seg.set_manual_bins("score", "0.7, 1.0, 1.3", missing="pior")
        seg.set_scorecard_dummies("score")
        seg.set_manual_bins("garantia", [["A"], ["B"], ["C", "D", "O'Neil"]])
        novas = seg.create_scorecard_dummies("score")
        cat = seg.create_categorical("garantia")
        feats = ["score", "garantia", "flag", "renda", cat] + novas
        seg.fit(features=feats, transform="woe")
        sql = seg.categorization_sql(table="base")
    con = sqlite3.connect(":memory:")
    df.drop(columns="target").to_sql("base", con, index=False)
    res = pd.read_sql(sql.rstrip(";"), con)
    X = seg._apply_derived(df.drop(columns="target"))
    return seg, sql, res, X, feats, novas, cat


def test_sql_e_um_select_valido_com_cabecalho(cenario):
    _seg, sql, res, *_ = cenario
    assert sql.startswith("-- Categorização das variáveis")
    assert "FROM base;" in sql and len(res) == 5000


def test_faixas_batem_com_o_python(cenario):
    seg, _sql, res, X, *_ = cenario
    py = seg.recreate_categories(X, features=["score", "garantia", "flag", "renda"])
    for c in py.columns:
        a = res[c].where(res[c].notna(), None).tolist()
        b = py[c].astype(object).where(py[c].notna(), None).tolist()
        assert a == b, c


def test_woe_bate_com_o_modelo(cenario):
    seg, _sql, res, X, feats, novas, cat = cenario
    pre = seg.model.named_steps["pre"]
    enc = pd.DataFrame(np.asarray(pre.transform(X[feats]), dtype=float),
                       columns=list(pre.get_feature_names_out()))
    for f in ["garantia", "flag", "renda", cat] + novas:
        col = next(c for c in enc.columns if c.endswith(f"WoE({f})"))
        assert np.abs(res[f"{f}_woe"].to_numpy(float) - enc[col].to_numpy()).max() < 1e-12


def test_dummies_do_modelo_e_derivadas_batem(cenario):
    seg, _sql, res, X, feats, novas, cat = cenario
    pre = seg.model.named_steps["pre"]
    enc = pd.DataFrame(np.asarray(pre.transform(X[feats]), dtype=float),
                       columns=list(pre.get_feature_names_out()))
    for c in [c for c in enc.columns if c.startswith("dum__")]:
        f, faixa = c[len("dum__"):].split("=", 1)
        k = seg._dummy_spec(f)["labels"].index(faixa) + 1
        assert (res[f"{f}__d{k}"].to_numpy() == enc[c].to_numpy()).all()
    for c in novas:
        assert (res[c].to_numpy() == seg.df[c].to_numpy()).all()
    a = res[cat].where(res[cat].notna(), None).tolist()
    b = seg.df[cat].astype(object).where(seg.df[cat].notna(), None).tolist()
    assert a == b


def test_apostrofo_escapado_e_bool_literal(cenario):
    _seg, sql, *_ = cenario
    assert "'O''Neil'" in sql
    assert "flag = TRUE" in sql or "flag = FALSE" in sql


def test_sem_woe_e_escopo_de_variaveis():
    df = _df(n=1500)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = ModelSegmenter(df, target="target")
        seg.set_manual_bins("score", "0.7, 1.0")
        sql = seg.categorization_sql(features=["score"], woe=False)
    assert "score_faixa" in sql and "_woe" not in sql and "garantia" not in sql
