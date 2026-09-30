"""Dummies de scorecard no ModelSegmenter (só categorização manual).

Cada faixa vira 0/1 e a pior faixa (maior risco na DES) é a referência omitida ⇒
com alvo 1 = mau todos os coeficientes saem negativos; o fit avisa quando o
multivariado inverte alguma dummy.
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


def _seg_dummies():
    seg = _seg()
    seg.set_manual_bins("score", "0.7, 1.0, 1.3")
    seg.set_manual_bins("garantia", "A; B; C, D")
    seg.set_scorecard_dummies("score")
    seg.set_scorecard_dummies("garantia")
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
        seg.set_scorecard_dummies("score")
    assert not seg.scorecard_dummies("score")


def test_limpar_bins_desliga():
    seg = _seg_dummies()
    seg.clear_manual_bins("score")
    assert not seg.scorecard_dummies("score")
    assert "scorecard_dummies" not in seg.var_meta["score"]
    seg.set_manual_bins("garantia", "")
    assert not seg.scorecard_dummies("garantia")


def test_tabela_referencia_e_a_pior_faixa():
    seg = _seg_dummies()
    for f in ("score", "garantia"):
        tab = seg.scorecard_table(f)
        assert list(tab["papel"]) == ["referência"] + ["dummy"] * (len(tab) - 1)
        assert np.all(np.diff(tab["taxa_maus"].to_numpy()) <= 0)   # pior → melhor
    assert seg.scorecard_table("garantia").loc[0, "faixa"] == "{C, D}"


def test_alias_ordinal_da_0014_vira_dummies():
    seg = _seg()
    seg.set_manual_bins("score", "0.7, 1.0, 1.3")
    with pytest.warns(DeprecationWarning):
        seg.set_scorecard_ordinal("score")
    assert seg.scorecard_dummies("score") and seg.scorecard_ordinal("score")
    seg.var_meta["garantia"]["splits"] = [["A"], ["B"], ["C", "D"]]
    seg.var_meta["garantia"]["ordinal_scorecard"] = True           # JSON salvo na 0.0.14
    assert seg.scorecard_dummies("garantia")


# ───────────────────────── treino ─────────────────────────
@pytest.mark.parametrize("transform", ["raw", "woe"])
def test_uma_dummy_por_faixa_menos_a_pior_todas_negativas(transform):
    seg = _seg_dummies()
    avisos = _fit(seg, features=["score", "garantia"], transform=transform)
    coef = seg.model_coefficients(use_labels=False)
    esperado = {"score = (-inf, 0.7]", "score = (0.7, 1]", "score = (1, 1.3]",
                "garantia = {A}", "garantia = {B}"}
    assert set(coef["termo"]) == esperado                 # (1.3, inf] e {C, D} = referência
    assert (coef["coef"] < 0).all()
    chk = seg.scorecard_sign_check()
    assert chk["sinal_ok"].all() and len(chk) == 5
    assert avisos == []
    assert np.isfinite(seg.score_).all()


def test_mistura_com_nao_dummies_e_shap_agrupado():
    seg = _seg_dummies()
    _fit(seg, features=["score", "garantia", "renda", "uf"], transform="raw")
    termos = list(seg.model_coefficients(use_labels=False)["termo"])
    assert "renda" in termos and any(t.startswith("uf_") for t in termos)
    assert seg._original_feature_of("dum__score=(0.7, 1]") == "score"


def test_categoria_nova_cai_na_referencia():
    seg = _seg_dummies()
    _fit(seg, features=["score", "garantia"], transform="raw")
    novo = _df(n=50, seed=7)
    novo.loc[:4, "garantia"] = "Z"
    pre = seg.model.named_steps["pre"]
    X = pd.DataFrame(pre.transform(novo[["score", "garantia"]]),
                     columns=pre.get_feature_names_out())
    cols = [c for c in X.columns if "garantia=" in c]
    assert (X.loc[:4, cols] == 0).all().all()


def test_aviso_quando_multivariado_inverte():
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
        seg.set_scorecard_dummies(f)
    avisos = _fit(seg, features=["x1", "x2"], transform="raw")
    chk = seg.scorecard_sign_check()
    assert not chk[chk["variavel"] == "x2"]["sinal_ok"].all()
    assert len(avisos) == 1 and "x2 =" in avisos[0]


def test_persistencia():
    seg = _seg_dummies()
    _fit(seg, features=["score", "garantia"], transform="raw")
    with contextlib.redirect_stdout(io.StringIO()):
        seg2 = ModelSegmenter.from_dict(seg.to_dict(), _df())
    assert seg2.scorecard_dummies("score") and seg2.scorecard_dummies("garantia")


# ───────────────────────── variáveis dummy criadas ─────────────────────────
def test_criar_variaveis_dummy():
    seg = _seg_dummies()
    novas = seg.create_scorecard_dummies("score")
    assert novas == ["score_d1", "score_d2", "score_d3"]      # (1.3, inf] = referência
    assert seg.label("score_d2") == "score = (0.7, 1]"
    assert set(np.unique(seg.df[novas].to_numpy())) <= {0, 1}
    assert (seg.df[novas].sum(axis=1) <= 1).all()
    _fit(seg, features=novas, transform="raw")
    X = _df(n=20, seed=5).drop(columns="target")                # só a coluna de origem
    assert np.isfinite(seg.predict(X)["score"]).all()


def test_criar_dummies_exige_manual():
    seg = _seg()
    with pytest.raises(ValueError, match="categorização manual"):
        seg.create_scorecard_dummies("score")


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


def test_ui_checkbox_tabela_e_botao(ui):
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_var2.value = "score"
    assert ui.cb_scorecard.layout.display == "none"
    ui.tg_binmode.value = "Manual"
    ui.tx_cuts.value = "0.7, 1.0, 1.3"
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_apply_bins(None)
    assert ui.cb_scorecard.layout.display == ""
    assert ui.btn_create_cat.description == "Criar variável categórica"

    ui.cb_scorecard.value = True
    assert ui.seg.scorecard_dummies("score")
    assert "referência" in ui.out_faixas_table.value
    assert ui.btn_create_cat.description == "Criar variável (dummies)"
    assert "dummies (3 + ref.)" in ui.out_vars.value          # coluna 'codificacao'

    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_create_cat(None)
    assert "score_cat" in ui.seg.candidates                     # UMA variável
    assert not any(c.startswith("score_d") for c in ui.seg.candidates)
    assert ui.seg.scorecard_dummies("score_cat")

    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_undo(None)
    assert "score_cat" not in ui.seg.candidates
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_undo(None)
    assert not ui.seg.scorecard_dummies("score") and not ui.cb_scorecard.value


def test_ui_treino_mostra_conferencia_de_sinal(ui):
    seg = ui.seg
    seg.set_manual_bins("score", "0.7, 1.0, 1.3")
    seg.set_scorecard_dummies("score")
    for f in list(seg.included):
        seg.exclude(f)
    seg.include("score")
    ui.dd_algo.value = "logistica"
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_fit(None)
    assert "Scorecard" in ui.out_fit_status.value
    assert "coeficiente negativo" in ui.out_fit_status.value


# ───────────────────────── caminho 1: uma variável, dummies no modelo ─────────────────────────
def test_variavel_original_fica_uma_linha_no_ranking():
    seg = _seg_dummies()
    rk = seg.variable_iv(["score", "garantia"], with_psi=False)
    assert list(rk["variavel"]) == ["score", "garantia"] or set(rk["variavel"]) == {"score", "garantia"}
    assert len(rk) == 2
    assert int(rk.set_index("variavel").loc["score", "n_bins"]) == 4


def test_create_categorical_com_dummies_gera_uma_variavel_ja_em_dummies():
    seg = _seg_dummies()
    nova = seg.create_categorical("score")                     # segue a opção da origem
    assert nova == "score_cat" and seg.label(nova) == "score (dummies)"
    assert seg.scorecard_dummies(nova)
    assert len(seg.scorecard_table(nova)) == 4
    _fit(seg, features=[nova], transform="raw")
    coef = seg.model_coefficients(use_labels=False)
    assert len(coef) == 3 and (coef["coef"] < 0).all()
    assert seg.shap_importance_grouped().iloc[:, 0].tolist() == [nova]
    X = _df(n=30, seed=4).drop(columns="target")                # só a coluna de origem
    assert np.isfinite(seg.predict(X)["score"]).all()


def test_create_categorical_sem_dummies_segue_categorica():
    seg = _seg()
    seg.set_manual_bins("score", "0.7, 1.0")
    nova = seg.create_categorical("score")
    assert seg.label(nova) == "score (cat.)" and not seg.scorecard_dummies(nova)
