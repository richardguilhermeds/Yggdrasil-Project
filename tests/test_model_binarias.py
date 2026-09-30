"""Variáveis binárias no ModelSegmenter: um nível por faixa, sem optbinning.

O binning ótimo exige 5% da base por faixa; numa flag rara ele não separava os
dois níveis e o IV saía NaN/0 — a variável sumia do ranking sendo forte.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.model import ModelSegmenter


def _df(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    f40 = (rng.random(n) < .40).astype(int)
    f3 = (rng.random(n) < .03).astype(int)
    f1 = (rng.random(n) < .01).astype(int)
    logit = -2.2 + 0.8 * f40 + 1.2 * f3 + 1.5 * f1
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    df = pd.DataFrame({
        "f40": f40, "f3": f3, "f1": f1, "f3_bool": f3.astype(bool),
        "f3_sn": np.where(f3 == 1, "S", "N"), "f3_12": f3 + 1,
        "f40_na": pd.Series(f40, dtype="float").mask(rng.random(n) < .1),
        "cont": rng.normal(size=n), "target": y})
    df["amostra"] = np.where(rng.random(n) < .7, "DES", "OOT")
    return df


def _seg(df=None):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ModelSegmenter(_df() if df is None else df, target="target",
                              sample_col="amostra")


def _iv(seg, feats):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return seg.variable_iv(feats, with_psi=False).set_index("variavel")


def _labels(seg, f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bins, _ = seg._resolve_bins(f)
    return [seg._bin_label(f, b) for b in bins]


@pytest.mark.parametrize("f", ["f3", "f1", "f3_bool", "f3_sn", "f3_12"])
def test_flag_rara_tem_duas_faixas_e_iv_real(f):
    seg = _seg()
    rk = _iv(seg, [f])
    assert rk.loc[f, "n_bins"] == 2
    assert rk.loc[f, "iv"] > 0.02                    # antes: NaN/0


def test_mesmo_iv_que_a_categorizacao_manual_nivel_a_nivel():
    seg = _seg()
    iv_auto = _iv(seg, ["f3", "f3_bool"])["iv"]
    seg.set_manual_bins("f3", "0.5")
    seg.set_manual_bins("f3_bool", "False; True")
    iv_man = _iv(seg, ["f3", "f3_bool"])["iv"]
    assert iv_auto.round(6).tolist() == iv_man.round(6).tolist()


def test_rotulos_e_faltantes():
    seg = _seg()
    assert _labels(seg, "f3") == ["(-inf, 0.5]", "(0.5, inf]"]
    assert _labels(seg, "f3_12") == ["(-inf, 1.5]", "(1.5, inf]"]
    assert _labels(seg, "f3_bool") == ["{False}", "{True}"]
    assert _labels(seg, "f3_sn") == ["{N}", "{S}"]
    assert _labels(seg, "f40_na") == ["(-inf, 0.5]", "(0.5, inf]", "(faltante)"]


def test_flag_frequente_nao_muda():
    seg = _seg()
    assert _iv(seg, ["f40"]).loc["f40", "n_bins"] == 2


def test_continua_segue_no_binning_otimo():
    seg = _seg()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bins, _ = seg._resolve_bins("cont")
    assert all(b["kind"] in ("num", "na") for b in bins)
    assert seg._bins_binaria("cont", seg._frame(None, ["cont", "target"]), "num") is None


def test_aviso_so_para_nivel_raro():
    seg = _seg()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        seg.variable_iv(["f40", "f3", "f1"], with_psi=False)
    msgs = [str(x.message) for x in w if "nível raro" in str(x.message)]
    assert len(msgs) == 1 and "'f1'" in msgs[0]


def test_manual_tem_precedencia():
    seg = _seg()
    seg.set_manual_bins("f3_sn", "N, S")             # usuário junta os dois níveis
    assert _labels(seg, "f3_sn") == ["{N, S}"]


def test_modelo_woe_e_ordinal_com_binaria():
    seg = _seg()
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.fit(features=["f3", "f40", "cont"], transform="woe")
    termos = set(seg.model_coefficients(use_labels=False)["termo"])
    assert {"WoE(f3)", "WoE(f40)"} <= termos
    assert np.isfinite(seg.score_).all()
