"""Variável CRIADA a partir das faixas de outra (create_categorical, com ou sem
dummies): a aba Análise tem de ver as MESMAS faixas da origem — mesma ordem, mesmo
IV/monotonicidade, "(faltante)" como faixa de faltantes — em vez de re-binar a
categórica (que reordenava por risco e contava o faltante como faixa comum)."""
import contextlib
import io
import sqlite3
import warnings

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.model import ModelSegmenter

FAIXAS = ["(-inf, 1500]", "(1500, 3000]", "(3000, 5000]", "(5000, inf]", "(faltante)"]


def _df(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    renda = rng.gamma(2, 1500, n)
    y = rng.binomial(1, 1 / (1 + np.exp(1.5 + 0.0004 * (renda - 3000)))).astype(float)
    renda[rng.random(n) < 0.12] = np.nan
    meses = pd.date_range("2023-01-01", periods=6, freq="MS")
    df = pd.DataFrame({"renda": renda, "idade": rng.normal(40, 10, n), "target": y})
    df["dt_ref"] = rng.choice(meses, size=n)
    df["amostra"] = np.where(df["dt_ref"] >= meses[4], "OOT", "DES")
    return df


@pytest.fixture
def cenario():
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = ModelSegmenter(_df(), target="target", sample_col="amostra",
                             date_col="dt_ref", verbose=False)
        seg.set_manual_bins("renda", [1500, 3000, 5000])
        cat = seg.create_categorical("renda", new_name="renda_cat")
        seg.set_scorecard_dummies("renda")
        dum = seg.create_categorical("renda", new_name="renda_dum", dummies=True)
    return seg, cat, dum


def test_mesmas_faixas_ordem_iv_e_monotonicidade(cenario):
    seg, cat, dum = cenario
    base = seg.variable_table("renda")
    assert list(base["faixa"]) == FAIXAS
    for f in (cat, dum):
        t = seg.variable_table(f)
        assert list(t["faixa"]) == FAIXAS
        assert t["n"].tolist() == base["n"].tolist()
        assert t.attrs["iv"] == base.attrs["iv"]
        assert t.attrs["mono_ok"] == base.attrs["mono_ok"]
        s, s0 = seg.variable_summary(f), seg.variable_summary("renda")
        assert (s["tendencia"], s["n_inversoes"]) == (s0["tendencia"], s0["n_inversoes"])


def test_faltante_da_origem_e_faltante(cenario):
    seg, cat, dum = cenario
    n0 = seg.missing_info("renda")["n"]
    assert n0 > 0
    for f in (cat, dum):
        assert seg.missing_info(f)["n"] == n0
        assert seg.missing_info(f)["faixa"] == "(faltante)"
        assert seg.variable_summary(f)["pct_missing"] == seg.variable_summary("renda")["pct_missing"]
    bins, _ = seg._resolve_bins(cat)
    assert bins[-1] == {"kind": "na", "cats": ["(faltante)"]}


def test_destino_dos_faltantes_na_derivada(cenario):
    seg, cat, _dum = cenario
    seg.set_manual_bins(cat, [[FAIXAS[0], FAIXAS[1]], [FAIXAS[2]], [FAIXAS[3]], [FAIXAS[4]]])
    seg.set_missing_bin(cat, "pior")
    t = seg.variable_table(cat)
    assert "(faltante)" not in list(t["faixa"])                 # sem faixa própria
    assert t["faixa"].iloc[0].endswith("+ faltante")            # pior = 1ª faixa
    ref = seg._frame(seg.ref_sample)
    assert int(t["n"].sum()) == len(ref)                        # ninguém sem faixa


def test_sql_e_woe_da_derivada_batem_com_o_modelo(cenario):
    seg, cat, _dum = cenario
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.fit("logistica", features=[cat, "idade"], transform="woe")
        sql = seg.categorization_sql(table="base")
    X = seg.df[["renda", "idade"]]
    con = sqlite3.connect(":memory:")
    try:
        X.to_sql("base", con, index=False)
        res = pd.read_sql(sql.rstrip(";"), con)
    finally:
        con.close()
    pre = seg.model.named_steps["pre"]
    Xd = seg._apply_derived(X)
    enc = pd.DataFrame(np.asarray(pre.transform(Xd[[cat, "idade"]]), dtype=float),
                       columns=list(pre.get_feature_names_out()))
    col = next(c for c in enc.columns if c.endswith(f"WoE({cat})"))
    assert np.abs(res[f"{cat}_woe"].to_numpy(float) - enc[col].to_numpy()).max() < 1e-12
    assert res[cat].tolist() == seg.df[cat].tolist()        # derivada sai com o próprio nome
    assert "IS NULL OR" in sql                              # faixa de faltantes: NULL ou '(faltante)'


def test_ui_caixas_na_ordem_da_origem(cenario):
    pytest.importorskip("ipywidgets")
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui = ModelSegmenterUI(_df(), target="target", sample_col="amostra", date_col="dt_ref")
        ui.dd_var2.value = "renda"
        ui.tg_binmode.value = "Manual"; ui.tx_cuts.value = "1500, 3000, 5000"
        ui._on_apply_bins(None)
        ui.tx_new_cat.value = "renda_cat"; ui._on_create_cat(None)
        ui.dd_var2.value = "renda_cat"
        ui.tg_binmode.value = "Manual"
        ui._rebuild_an_cat_box(force=True)
    assert list(ui._an_cat_widgets) == FAIXAS
    assert "renda" in ui.out_bin_hint.value or "faixas da origem" in ui.out_bin_status.value
