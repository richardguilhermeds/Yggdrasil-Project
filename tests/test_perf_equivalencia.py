"""As otimizações de desempenho não podem mudar resultado: caminhos rápidos
(factorize, searchsorted, bincount, caches) contra a regra de referência."""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk import _common
from yggdrasil.credit_risk.model import ModelSegmenter
from yggdrasil.credit_risk.model.segmenter import (
    _bin_codes, _bin_codes_numericos, _bin_masks, _cat_group_masks)


def _ref_masks(series, bins):
    """Regra de referência (a implementação antiga, por faixa)."""
    out = []
    for b in bins:
        if b["kind"] == "na":
            m = series.isna()
        elif b["kind"] == "num":
            m = series.between(b["lo"], b["hi"], inclusive="right")
        else:
            m = series.astype(str).isin(b["cats"])
        if b.get("include_na"):
            m = m | series.isna()
        out.append(m.to_numpy(dtype=bool, na_value=False))
    return out


def _ref_codes(series, bins):
    codes = np.full(len(series), -1, dtype=np.int32)
    for i, m in enumerate(_ref_masks(series, bins)):
        codes[m & (codes < 0)] = i
    return codes


@pytest.fixture
def dados():
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.normal(size=n)
    x[rng.random(n) < .1] = np.nan
    c = rng.choice(["A", "B", "C", "O'Neil", "7"], n).astype(object)
    c[rng.random(n) < .05] = None
    c[:50] = 7                                     # número em coluna de texto
    return pd.Series(x), pd.Series(c)


NUM = [{"kind": "num", "lo": -np.inf, "hi": -0.5}, {"kind": "num", "lo": -0.5, "hi": 0.3},
       {"kind": "num", "lo": 0.3, "hi": np.inf}]


@pytest.mark.parametrize("bins", [
    NUM + [{"kind": "na"}],
    [dict(NUM[0], include_na=True), NUM[1], NUM[2]],
    NUM,
    [NUM[0], {"kind": "num", "lo": 0.0, "hi": np.inf}, {"kind": "na"}],   # com buraco
])
def test_bin_codes_numericos_igual_a_regra(dados, bins):
    x, _ = dados
    assert (_bin_codes(x, bins) == _ref_codes(x, bins)).all()
    for a, b in zip(_bin_masks(x, bins), _ref_masks(x, bins)):
        assert (a == b).all()


def test_caminho_rapido_so_para_faixas_contiguas(dados):
    x, _ = dados
    assert _bin_codes_numericos(x, NUM) is not None
    assert _bin_codes_numericos(x, [NUM[0], {"kind": "num", "lo": 0.0, "hi": np.inf}]) is None


def test_mascaras_categoricas_igual_a_regra(dados):
    _, c = dados
    bins = [{"kind": "cat", "cats": ["A", "7"]}, {"kind": "cat", "cats": ["O'Neil"]},
            {"kind": "cat", "cats": ["B", "C"], "include_na": True}]
    for a, b in zip(_bin_masks(c, bins), _ref_masks(c, bins)):
        assert (a == b).all()
    assert (_bin_codes(c, bins) == _ref_codes(c, bins)).all()
    assert (_cat_group_masks(c, [["A"]])[0] == c.astype(str).isin(["A"]).to_numpy()).all()


def test_optbinning_amostra_so_acima_do_teto():
    class Falso:
        def fit(self, x, y):
            self.n = len(x)
            self.splits = [0.0]
    b = Falso()
    _common.fit_optbinning_splits(b, np.zeros(1000), np.zeros(1000))
    assert b.n == 1000
    _common.fit_optbinning_splits(b, np.zeros(10), np.zeros(10), max_rows=5)
    assert b.n == 5


# ───────────── análises por safra/amostra: vetorizado × referência ─────────────
@pytest.fixture(scope="module")
def seg():
    rng = np.random.default_rng(1)
    n = 6000
    x = rng.normal(size=n)
    x[rng.random(n) < .05] = np.nan
    c = rng.choice(list("ABCDEFGHIJ"), n).astype(object)
    c[rng.random(n) < .03] = None
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + np.nan_to_num(x))))).astype(float)
    meses = pd.date_range("2023-01-01", periods=10, freq="MS")
    df = pd.DataFrame({"x": x, "c": c, "target": y, "dt_ref": rng.choice(meses, n)})
    df.loc[:9, "dt_ref"] = pd.NaT
    df["amostra"] = np.where(df["dt_ref"] >= meses[7], "OOT", "DES")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ModelSegmenter(df, target="target", sample_col="amostra", date_col="dt_ref")


def test_variable_by_safra_igual_ao_groupby(seg):
    got = seg.variable_by_safra("x", all_samples=True)
    df = seg.df
    safra = pd.to_datetime(df["dt_ref"]).dt.to_period("M")
    for _, r in got.iterrows():
        g = df.loc[safra.astype(str) == r["safra"], "x"]
        v = g.dropna().to_numpy()
        assert r["n"] == len(g)
        assert r["p5"] == pytest.approx(round(float(np.percentile(v, 5)), 3))
        assert r["media"] == pytest.approx(round(float(v.mean()), 3))


def test_inversao_por_safra_igual_a_referencia(seg):
    bins, _ = seg._resolve_bins("x", 6, 0.05)
    s = seg._variable_bin_series("x", bins)
    df = seg.df
    safra = pd.to_datetime(df["dt_ref"]).dt.to_period("M").astype(str)
    masks = _ref_masks(df["x"], bins)
    for j, rot in enumerate(s["xs_safra"]):
        sel = (safra == rot).to_numpy()
        for i, m in enumerate(masks):
            y = df.loc[sel & m, "target"].to_numpy()
            esperado = y.mean() if y.size else np.nan
            got = s["ser_safra"][i][j]
            assert (np.isnan(esperado) and np.isnan(got)) or got == pytest.approx(esperado)


def test_share_categorias_por_safra(seg):
    got = seg.variable_share_by_safra("c", all_samples=True, top=4).set_index("safra")
    assert "outras" in got.columns and "(faltante)" in got.columns
    assert np.allclose(got.sum(axis=1), 100, atol=0.6)


def test_cache_do_ranking_so_recalcula_a_variavel_alterada(seg):
    seg.variable_iv(["x", "c"], with_psi=False)
    antes = dict(seg._iv_row_cache)
    seg.set_manual_bins("x", "-1, 0, 1")
    rk = seg.variable_iv(["x", "c"], with_psi=False).set_index("variavel")
    assert rk.loc["x", "n_bins"] >= 4
    novas = set(seg._iv_row_cache) - set(antes)
    assert len(novas) == 1 and next(iter(novas))[0] == "x"
    seg.clear_manual_bins("x")


# ───────────────────────── excluir variável criada ─────────────────────────
def test_remove_derived(seg):
    seg.set_manual_bins("c", "A, B; C, D, E; F, G, H, I, J")
    nova = seg.create_categorical("c")
    filha = seg.create_categorical(nova)
    with pytest.raises(ValueError, match="origem"):
        seg.remove_derived(nova)
    seg.remove_derived(filha)
    seg.remove_derived(nova)
    assert nova not in seg.df.columns and nova not in seg.candidates
    assert nova not in seg.var_meta
    with pytest.raises(ValueError, match="criada"):
        seg.remove_derived("x")


def test_ui_botao_excluir_da_base():
    pytest.importorskip("ipywidgets")
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    rng = np.random.default_rng(2)
    n = 2000
    df = pd.DataFrame({"v1": rng.normal(size=n), "c": rng.choice(list("ABCD"), n).astype(object),
                       "target": (rng.random(n) < .2).astype(float),
                       "dt_ref": pd.Timestamp("2024-01-01")})
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui = ModelSegmenterUI(df, target="target", date_col="dt_ref")
        ui.seg.set_manual_bins("c", "A, B; C, D")
        nova = ui.seg.create_categorical("c")
        ui._refresh_candidates()
    ui.dd_var.value = "v1"
    assert ui.btn_drop_derived.disabled                      # original: não apaga
    ui.dd_var.value = nova
    assert not ui.btn_drop_derived.disabled
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_drop_derived(None)
    assert nova not in ui.seg.df.columns and nova not in ui.seg.candidates
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_undo(None)                                    # desfazer recria a coluna
    assert nova in ui.seg.df.columns and nova in ui.seg.candidates
