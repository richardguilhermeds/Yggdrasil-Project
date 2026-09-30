"""Codificação ordinal de scorecard no ModelSegmenter (só categorização manual).

Pior faixa = 0 ... melhor = maior código ⇒ com alvo 1 = mau a logística ganha
coeficiente negativo por variável; o fit avisa quando o multivariado inverte.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.model import ModelSegmenter


def _df(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.beta(2.5, 3, n) * 1.4 + 0.3
    gar = rng.choice(list("ABCD"), n, p=[.5, .22, .18, .1]).astype(object)
    lg = {"A": -0.8, "B": -0.2, "C": 0.3, "D": 0.9}
    logit = -2 + 2.2 * (score - 0.9) + np.array([lg[g] for g in gar])
    df = pd.DataFrame({"score": score, "garantia": gar, "renda": rng.gamma(2, 1500, n),
                       "uf": rng.choice(["SP", "RJ", "MG"], n).astype(object),
                       "target": rng.binomial(1, 1 / (1 + np.exp(-logit))).astype(float)})
    df["amostra"] = np.where(rng.random(n) < .7, "DES", "OOT")
    return df


def _seg(df=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return ModelSegmenter(_df() if df is None else df, target="target",
                              task_type="classification", sample_col="amostra",
                              ref_sample="DES")


def _seg_ordinal():
    seg = _seg()
    seg.set_manual_bins("score", "0.7, 1.0, 1.3")
    seg.set_manual_bins("garantia", "A; B; C, D")
    seg.set_scorecard_ordinal("score")
    seg.set_scorecard_ordinal("garantia")
    return seg


def _fit(seg, **kw):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        seg.fit(**kw)
    return [str(x.message) for x in w if "Scorecard" in str(x.message)]


# ───────────────────────── configuração ─────────────────────────
def test_exige_categorizacao_manual():
    seg = _seg()
    with pytest.raises(ValueError, match="categorização manual"):
        seg.set_scorecard_ordinal("score")
    assert not seg.scorecard_ordinal("score")


def test_limpar_bins_desliga_ordinal():
    seg = _seg_ordinal()
    seg.clear_manual_bins("score")
    assert not seg.scorecard_ordinal("score")
    assert "ordinal_scorecard" not in seg.var_meta["score"]
    seg.set_manual_bins("garantia", "")          # spec vazia também limpa
    assert not seg.scorecard_ordinal("garantia")


def test_tabela_codigo_zero_e_a_pior_faixa():
    seg = _seg_ordinal()
    for f in ("score", "garantia"):
        tab = seg.scorecard_ordinal_table(f)
        assert list(tab["codigo"]) == list(range(len(tab)))
        taxa = tab["taxa_maus"].to_numpy()
        assert np.all(np.diff(taxa) <= 0)         # código sobe ⇒ risco cai
    tab = seg.scorecard_ordinal_table("garantia")
    assert tab.loc[0, "faixa"] == "{C, D}" and tab.iloc[-1]["faixa"] == "{A}"


# ───────────────────────── treino ─────────────────────────
@pytest.mark.parametrize("transform", ["raw", "woe"])
def test_coeficientes_negativos_nos_dois_transforms(transform):
    seg = _seg_ordinal()
    avisos = _fit(seg, features=["score", "garantia"], transform=transform)
    coef = seg.model_coefficients()
    assert set(coef["termo"]) == {"ord(score)", "ord(garantia)"}
    assert (coef["coef"] < 0).all()
    chk = seg.scorecard_sign_check()
    assert chk["sinal_ok"].all() and set(chk["variavel"]) == {"score", "garantia"}
    assert avisos == []
    assert np.isfinite(seg.score_).all()


def test_mistura_com_variaveis_nao_ordinais_no_raw():
    seg = _seg_ordinal()
    _fit(seg, features=["score", "garantia", "renda", "uf"], transform="raw")
    termos = list(seg.model_coefficients(use_labels=False)["termo"])
    assert "ord(score)" in termos and "ord(garantia)" in termos
    assert "renda" in termos                       # numérica crua segue igual
    assert any(t.startswith("uf_") for t in termos)  # one-hot segue igual
    assert len(seg.scorecard_sign_check()) == 2


def test_escoragem_de_base_nova_usa_os_mesmos_codigos():
    seg = _seg_ordinal()
    _fit(seg, features=["score", "garantia"], transform="raw")
    novo = _df(n=300, seed=7)
    novo.loc[:5, "garantia"] = "Z"                 # categoria nunca vista → código 0
    p = seg.model.predict_proba(novo[["score", "garantia"]])[:, 1]
    assert np.isfinite(p).all()
    pre = seg.model.named_steps["pre"]
    X = pre.transform(novo[["score", "garantia"]])
    col = list(pre.get_feature_names_out()).index("ord__ord(garantia)")
    assert (X[:6, col] == 0).all()


def test_aviso_quando_multivariado_inverte_o_sinal():
    """x2 é arriscada sozinha (correlação com x1), mas protetora no multivariado:
    a ordinal de x2 (ordenada pelo risco univariado) fica com coeficiente > 0."""
    rng = np.random.default_rng(3)
    n = 8000
    x1 = rng.normal(size=n)
    x2 = 0.9 * x1 + 0.44 * rng.normal(size=n)
    logit = -1.5 + 2.0 * x1 - 1.0 * x2
    df = pd.DataFrame({"x1": x1, "x2": x2,
                       "target": rng.binomial(1, 1 / (1 + np.exp(-logit))).astype(float)})
    df["amostra"] = "DES"
    seg = _seg(df)
    for f in ("x1", "x2"):
        seg.set_manual_bins(f, "-1, 0, 1")
        seg.set_scorecard_ordinal(f)
    avisos = _fit(seg, features=["x1", "x2"], transform="raw")
    chk = seg.scorecard_sign_check().set_index("variavel")
    assert not chk.loc["x2", "sinal_ok"] and chk.loc["x1", "sinal_ok"]
    assert len(avisos) == 1 and "x2" in avisos[0]


def test_sem_ordinais_nao_ha_checagem():
    seg = _seg()
    _fit(seg, features=["score", "renda"], transform="raw")
    assert seg.scorecard_sign_check().empty


def test_persistencia_do_marcador():
    seg = _seg_ordinal()
    _fit(seg, features=["score", "garantia"], transform="raw")
    d = seg.to_dict()
    with contextlib.redirect_stdout(io.StringIO()):
        seg2 = ModelSegmenter.from_dict(d, _df())
    assert seg2.scorecard_ordinal("score") and seg2.scorecard_ordinal("garantia")


# ───────────────────────── UI ─────────────────────────
@pytest.fixture
def ui():
    pytest.importorskip("ipywidgets")
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    df = _df()
    df["dt_ref"] = pd.Timestamp("2024-01-01")
    with contextlib.redirect_stdout(io.StringIO()):
        return ModelSegmenterUI(df, target="target", task_type="classification",
                                sample_col="amostra", ref_sample="DES", date_col="dt_ref")


def test_ui_checkbox_so_aparece_com_bins_manuais(ui):
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_var2.value = "score"
    assert ui.cb_ordinal.layout.display == "none"
    ui.tg_binmode.value = "Manual"
    ui.tx_cuts.value = "0.7, 1.0, 1.3"
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_apply_bins(None)
    assert ui.cb_ordinal.layout.display == "" and not ui.cb_ordinal.value

    ui.cb_ordinal.value = True
    assert ui.seg.scorecard_ordinal("score")
    assert "0 = pior faixa" in ui.out_ord_table.value

    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_undo(None) if hasattr(ui, "_on_undo") else ui.btn_undo.click()
    assert not ui.seg.scorecard_ordinal("score") and not ui.cb_ordinal.value

    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_clear_bins(None)
    assert ui.cb_ordinal.layout.display == "none"


def test_ui_treino_mostra_conferencia_de_sinal(ui):
    seg = ui.seg
    seg.set_manual_bins("score", "0.7, 1.0, 1.3")
    seg.set_scorecard_ordinal("score")
    for f in list(seg.included):
        seg.exclude(f)
    seg.include("score")
    ui.dd_algo.value = "logistica"
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_fit(None)
    assert "Scorecard" in ui.out_fit_status.value
    assert "coeficiente negativo" in ui.out_fit_status.value
