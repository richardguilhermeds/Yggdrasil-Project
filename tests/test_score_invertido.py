"""Score de negócio invertido (padrão na classificação): 1000 = melhor cliente,
0 = pior. ``score_invertido=False`` volta ao ``p × 1000``; métricas, calibração
e ratings seguem na escala crua."""
import sqlite3
import warnings

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.model import ModelSegmenter


def _df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-1.2 + 1.1 * x1 - 0.6 * x2)))
    return pd.DataFrame({"x1": x1, "x2": x2,
                         "target": (rng.random(n) < p).astype(int),
                         "amostra": np.where(rng.random(n) < 0.7, "DES", "OOT")})


def _seg(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = ModelSegmenter(df, target="target", sample_col="amostra",
                             verbose=False, **kw)
        seg.fit("logistica")
    return seg


@pytest.fixture(scope="module")
def df():
    return _df()


def test_padrao_invertido_1000_melhor(df):
    seg = _seg(df)
    assert seg.score_invertido and seg._score_inv
    pts = seg.score_points_
    assert np.allclose(pts, 1000 * (1 - seg.score_))
    # quem tem MAIS risco (prob. maior) tem score MENOR
    assert np.corrcoef(pts, seg.score_)[0, 1] < -0.999
    # evento concentrado nos scores baixos
    y = df["target"].to_numpy()
    assert pts[y == 1].mean() < pts[y == 0].mean()
    pred = seg.predict(df.head(200))
    assert np.allclose(pred["score"], 1000 * (1 - seg.score_.head(200)))
    assert np.allclose(seg.assign()["score"], pts)


def test_desligar_volta_a_probabilidade(df):
    seg = _seg(df, score_invertido=False)
    assert not seg._score_inv
    assert np.allclose(seg.score_points_, 1000 * seg.score_)


def test_metricas_e_ratings_nao_mudam(df):
    a, b = _seg(df), _seg(df, score_invertido=False)
    ma, mb = a.metrics(), b.metrics()
    for c in ("auc", "gini", "ks"):
        if c in ma.columns:
            assert np.allclose(ma[c], mb[c])
    # ks_cutoff é um limiar: acompanha a escala de negócio
    assert np.allclose(ma["ks_cutoff"], 1000 - mb["ks_cutoff"])
    a.build_ratings(method="quantil", n_ratings=5)
    b.build_ratings(method="quantil", n_ratings=5)
    assert list(a.rating_) == list(b.rating_)


def test_to_sql_invertido_reproduz_ratings(df):
    seg = _seg(df)
    seg.build_ratings(method="quantil", n_ratings=5)
    sql = seg.to_sql(table="base")
    assert "= melhor" in sql
    pred = seg.predict(df)
    con = sqlite3.connect(":memory:")
    try:
        pred[["score"]].to_sql("base", con, index=False)
        got = pd.read_sql_query(sql, con)
    finally:
        con.close()
    assert got["rating"].tolist() == pred["rating"].tolist()


def test_logit_sql_score_invertido(df):
    seg = _seg(df)
    sql = seg.logit_sql(table="base")
    assert "(1.0 - probabilidade)" in sql
    con = sqlite3.connect(":memory:")
    con.create_function("EXP", 1, np.exp)
    try:
        df[["x1", "x2"]].to_sql("base", con, index=False)
        got = pd.read_sql_query(sql, con)
    finally:
        con.close()
    assert np.abs(got["score"].to_numpy() - seg.score_points_.to_numpy()).max() < 1e-6


def test_regressao_nao_inverte():
    rng = np.random.default_rng(1)
    n = 800
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "target": np.clip(0.4 + 0.1 * x + rng.normal(0, .05, n), 0, 1),
                       "amostra": "DES"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = ModelSegmenter(df, target="target", task_type="regression",
                             sample_col="amostra", verbose=False)
        seg.fit("linear")
    assert not seg._score_inv
    assert np.allclose(seg.score_points_, 1000 * seg.score_)


def test_persistencia(df):
    seg = _seg(df, score_invertido=False)
    d = seg.to_dict()
    assert d["meta"]["score_invertido"] is False
    novo = ModelSegmenter.from_dict(d, df, verbose=False)
    assert novo.score_invertido is False
    # modelo salvo antes da opção (sem a chave) ⇒ mantém o score antigo (p × 1000)
    d = _seg(df).to_dict()
    assert d["meta"]["score_invertido"] is True
    d["meta"].pop("score_invertido")
    assert ModelSegmenter.from_dict(d, df, verbose=False).score_invertido is False


def test_ui_cortes_manuais_na_escala_invertida(df):
    pytest.importorskip("ipywidgets")
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui = ModelSegmenterUI(df, target="target", sample_col="amostra")
        ui.seg.fit("logistica")
        ui.dd_method.value = "manual_score"
        ui.tx_manual.value = "300, 600"
        ui._on_build_ratings(None)
    pts = ui.seg.score_points_.to_numpy()
    rat = np.asarray(ui.seg.rating_)
    assert len(ui.seg.rating_labels_) == 3
    # cada rating ocupa uma faixa contígua de pontos cortada em 300/600
    grupos = [pts[rat == lab] for lab in ui.seg.rating_labels_]
    faixas = sorted((g.min(), g.max()) for g in grupos if g.size)
    assert faixas[0][1] <= 300 + 1e-6 and faixas[1][0] >= 300 - 1e-6
    assert faixas[1][1] <= 600 + 1e-6 and faixas[2][0] >= 600 - 1e-6
