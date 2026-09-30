"""Rodada 2 de desempenho: cada caminho rápido contra o cálculo direto antigo
(ratings, métricas por safra, Spearman, calibração, bootstrap, amostra dos
gráficos da aba Análise)."""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.model import ModelSegmenter
from yggdrasil.credit_risk.model.segmenter import _spearman_pairwise


def _df(n=8000, seed=3):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 0.9 * x1 + 0.4 * rng.normal(size=n)
    x3 = rng.integers(0, 5, n).astype(float)             # empates
    x1[rng.random(n) < .05] = np.nan
    c = rng.choice(list("ABCDE"), n).astype(object)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + np.nan_to_num(x1))))).astype(float)
    meses = pd.date_range("2023-01-01", periods=10, freq="MS")
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "c": c, "target": y,
                       "dt_ref": rng.choice(meses, n)})
    df.loc[:4, "dt_ref"] = pd.NaT
    df["amostra"] = np.where(df["dt_ref"] >= meses[7], "OOT", "DES")
    return df


@pytest.fixture(scope="module")
def seg():
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ModelSegmenter(_df(), target="target", sample_col="amostra", date_col="dt_ref")
        s.fit(features=["x1", "x2", "c"], transform="woe")
        s.build_ratings(n_ratings=6) if "n_ratings" in s.build_ratings.__code__.co_varnames \
            else s.build_ratings()
    return s


def _safra_str(df):
    return pd.to_datetime(df["dt_ref"], errors="coerce").dt.to_period("M").astype(str)


# ───────────────────────── Spearman ─────────────────────────
def test_spearman_igual_ao_pandas():
    df = _df()[["x1", "x2", "x3"]]
    df["const"] = 1.0
    ref = df.corr(method="spearman")
    got = _spearman_pairwise(df)
    assert np.allclose(got.to_numpy(), ref.to_numpy(), atol=1e-10, equal_nan=True)


# ───────────────────────── ratings ─────────────────────────
def test_psi_de_ratings_igual_ao_direto(seg):
    r = seg.rating_
    ref_m = seg.df["amostra"] == "DES"
    labs = seg.rating_labels_

    def dist(m):
        n = max(int((r.notna() & m).sum()), 1)
        return [max(int(((r == l) & m).sum()) / n, 1e-6) for l in labs]
    ref = dist(ref_m)
    oot = [int(((r == l) & (seg.df["amostra"] == "OOT")).sum()) for l in labs]
    n_o = sum(oot)
    esperado = sum((o / n_o - e) * np.log(max(o / n_o, 1e-6) / e) for o, e in zip(oot, ref))
    got = float(seg.psi().set_index("amostra").loc["OOT", "psi"])
    assert got == pytest.approx(round(esperado, 4), abs=1e-4)


def test_psi_de_ratings_por_safra_contagens(seg):
    t = seg.rating_psi_by_safra()
    saf = _safra_str(seg.df)
    for _, row in t.iterrows():
        n_direto = int((seg.rating_.notna() & (saf == row["safra"])).sum())
        assert row["n"] == n_direto


def test_inversao_de_ratings_por_safra(seg):
    inv = seg.rating_inversion()
    saf = _safra_str(seg.df)
    for per, vals in inv["safra_series"].items():
        for lab, v in vals.items():
            y = seg.df.loc[(saf == per) & (seg.rating_ == lab), "target"].dropna()
            esperado = y.mean() if len(y) else np.nan
            assert (np.isnan(v) and np.isnan(esperado)) or v == pytest.approx(esperado)


def test_rating_transform_igual_ao_dict_por_linha(seg):
    strat = seg.rating_strategy
    frame = seg._rating_frame()
    cfg = seg._make_cfg("_amostra")
    raw = strat._raw_groups(np.asarray(frame[cfg.score_col], dtype=float))
    esperado = [strat.raw_to_label_.get(int(g)) for g in raw]
    assert list(strat.transform(frame, cfg)) == esperado


# ───────────────────────── métricas por safra ─────────────────────────
def test_metrics_by_safra_igual_ao_groupby(seg):
    from yggdrasil.metrics import classification_metrics
    got = seg.metrics_by_safra().set_index("safra")
    saf = _safra_str(seg.df)
    for per in got.index:
        m = (saf == per).to_numpy()
        y = seg.df.loc[m, "target"].to_numpy(float)
        sc = seg.score_.to_numpy(float)[m]
        assert got.loc[per, "n"] == len(y)
        esperado = classification_metrics(y, sc)["auc"]
        assert got.loc[per, "auc"] == pytest.approx(esperado)


def test_amostra_dominante_por_safra(seg):
    got = seg._amostra_dominante_por_safra("dt_ref")
    saf = _safra_str(seg.df)
    for per, am in got.items():
        assert am == seg.df.loc[saf == per, "amostra"].mode().iat[0]


# ───────────────────────── calibração com score cru em cache ─────────────────────────
@pytest.mark.parametrize("metodo", ["intercept", "platt", "isotonic"])
def test_calibracao_usa_score_cru_do_modelo(metodo):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ModelSegmenter(_df(seed=5), target="target", sample_col="amostra")
        s.fit(features=["x1", "x2"], transform="raw")
        s.calibrate(method=metodo)
    direto = s._compute_score(s.df)                     # re-escora com a camada
    assert np.allclose(s.score_.to_numpy(), direto.to_numpy(), atol=1e-12)
    with contextlib.redirect_stdout(io.StringIO()):
        s.decalibrate()
    assert np.allclose(s.score_.to_numpy(), s._compute_score(s.df).to_numpy(), atol=1e-12)


# ───────────────────────── bootstrap ─────────────────────────
def test_bootstrap_multinomial_tem_a_largura_certa(seg):
    ci = seg.bootstrap_ci(n_boot=2000).dropna(subset=["ic_low"])
    for _, r in ci.iterrows():
        p = r[[c for c in ci.columns if c.startswith("valor_")][0]]
        n = r["n"]
        se = np.sqrt(p * (1 - p) / n)
        assert r["ic_low"] <= p <= r["ic_high"]
        # IC 95% ≈ ±1,96·EP (binomial) — largura dentro de 25%
        assert r["amplitude"] == pytest.approx(2 * 1.96 * se, rel=0.25, abs=2e-3)


# ───────────────────────── amostra dos gráficos ─────────────────────────
def test_amostra_graficos_so_dentro_do_contexto_e_ranking_intacto():
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ModelSegmenter(_df(n=6000), target="target", sample_col="amostra",
                           date_col="dt_ref")
    s.max_linhas_graficos = 1000
    rk_antes = s.variable_iv(["x1", "c"], with_psi=True)
    assert s._rows_mask("DES").sum() == (s.df["amostra"] == "DES").sum()
    with s._amostra_graficos() as ativa:
        assert ativa
        assert s._rows_mask(all_rows=True).sum() == 1000
        assert s._rows_mask("DES", amostrar=False).sum() == (s.df["amostra"] == "DES").sum()
        t = s.variable_by_safra("x1", all_samples=True)
        assert t["n"].sum() <= 1000
        s._iv_row_cache.clear()
        rk_dentro = s.variable_iv(["x1", "c"], with_psi=True)
    pd.testing.assert_frame_equal(rk_antes.drop(columns=["incluida", "categoria"], errors="ignore"),
                                  rk_dentro.drop(columns=["incluida", "categoria"], errors="ignore"))
    assert s._rows_mask(all_rows=True).sum() == 6000
    s.max_linhas_graficos = None
    with s._amostra_graficos() as ativa:
        assert not ativa
