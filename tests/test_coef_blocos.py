"""Coeficientes da logística em BLOCOS por variável (e por magnitude)."""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.model import ModelSegmenter


def _df(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.beta(2.5, 3, n) * 1.4 + 0.3
    gar = rng.choice(list("ABCD"), n).astype(object)
    uf = rng.choice(["SP", "RJ", "MG"], n).astype(object)
    risco = {"A": -.8, "B": -.2, "C": .3, "D": .9}
    logit = -2 + 2.2 * (score - 0.9) + np.array([risco[g] for g in gar])
    return pd.DataFrame({"score_b": score, "garantia": gar, "uf": uf,
                         "renda": rng.gamma(2, 1500, n),
                         "target": rng.binomial(1, 1 / (1 + np.exp(-logit))).astype(float)})


@pytest.fixture
def seg():
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ModelSegmenter(_df(), target="target", feature_labels={"score_b": "Score Bureau"})
        s.set_manual_bins("score_b", "0.7, 1.0, 1.3")
        s.set_scorecard_dummies("score_b")
        s.fit(features=["score_b", "garantia", "uf", "renda"], transform="raw")
    return s


def test_termos_contiguos_por_variavel_e_blocos_por_magnitude(seg):
    co = seg.model_coefficients()
    # cada variável aparece num bloco contíguo
    vistos, anterior = [], None
    for v in co["variavel"]:
        if v != anterior:
            assert v not in vistos
            vistos.append(v)
            anterior = v
    assert set(vistos) == {"score_b", "garantia", "uf", "renda"}
    # blocos na ordem da maior |coef| do bloco
    maximos = [co[co["variavel"] == v]["coef"].abs().max() for v in vistos]
    assert maximos == sorted(maximos, reverse=True)
    # dentro do bloco, por |coef|
    for v in vistos:
        a = co[co["variavel"] == v]["coef"].abs().tolist()
        assert a == sorted(a, reverse=True)


def test_termo_curto_e_rotulos(seg):
    co = seg.model_coefficients().set_index("termo")
    assert co.loc["Score Bureau = (0.7, 1]", "termo_curto"] == "(0.7, 1]"
    assert co.loc["Score Bureau = (0.7, 1]", "variavel_label"] == "Score Bureau"
    assert co.loc["uf_SP", "termo_curto"] == "SP"
    assert co.loc["renda", "termo_curto"] == "renda"


def test_ordem_magnitude_mantem_o_comportamento_antigo(seg):
    co = seg.model_coefficients(ordem="magnitude")
    a = co["coef"].abs().tolist()
    assert a == sorted(a, reverse=True)


def test_ui_renderiza_blocos():
    pytest.importorskip("ipywidgets")
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    df = _df(n=2500)
    df["dt_ref"] = pd.Timestamp("2024-01-01")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui = ModelSegmenterUI(df, target="target", date_col="dt_ref")
        ui.seg.set_manual_bins("score_b", "0.7, 1.0, 1.3")
        ui.seg.set_scorecard_dummies("score_b")
        ui.dd_algo.value = "logistica"
        ui._on_fit(None)
    html = ui.out_formula.value
    assert "<tr class='blk'>" in html and "class='in-blk'" in html
    assert "termos · |coef| máx" in html
    assert "mseg-eq-blk" in html
