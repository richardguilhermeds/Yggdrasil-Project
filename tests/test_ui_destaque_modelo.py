"""Destaque, na aba "Análise de variáveis", das variáveis que estão no modelo.

* ModelSegmenterUI: ✓ = no modelo treinado (verde) · ● = selecionada, ainda
  fora do modelo (azul) · sem marca = fora da seleção.
* TreeSegmenterUI: ✓ = usada em alguma quebra da árvore (verde); o destaque
  segue a variável ANALISADA (a que está em tela).
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pytest


def _df(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.beta(2.5, 3, n) * 1.4 + 0.3
    gar = rng.choice(list("ABCD"), n, p=[0.5, 0.22, 0.18, 0.1]).astype(object)
    lg = {"A": 0.0, "B": 0.10, "C": 0.16, "D": 0.30}
    risco = 0.1 + 0.4 * (score - 0.5) + np.array([lg[g] for g in gar])
    meses = pd.date_range("2023-01-01", periods=8, freq="MS")
    df = pd.DataFrame({"score": score, "garantia": gar, "ruido": rng.normal(size=n),
                       "target": (rng.uniform(0, 1, n) < np.clip(risco, .02, .95)).astype(float)})
    df["dt_ref"] = rng.choice(meses, size=n)
    df["amostra"] = np.where(df["dt_ref"] >= meses[6], "OOT", "DES")
    return df


def _classes(w):
    return set(w._dom_classes)


# ───────────────────────── ModelSegmenterUI ─────────────────────────
@pytest.fixture
def mui():
    pytest.importorskip("ipywidgets")
    pytest.importorskip("optbinning")
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    with contextlib.redirect_stdout(io.StringIO()):
        return ModelSegmenterUI(_df(), target="target", task_type="classification",
                                sample_col="amostra", ref_sample="DES", date_col="dt_ref")


def _rotulo(ui, feat):
    return next(lbl for lbl, v in ui.dd_var2.options if v == feat)


def _ver(ui, feat):
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_var2.value = feat          # o observer re-renderiza e sincroniza o destaque


def test_model_modelo_treinado_fica_verde_e_marcado(mui):
    ui = mui
    for f in list(ui.seg.included):
        ui.seg.exclude(f)
    ui.seg.include("score")
    ui.seg.include("garantia")
    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.fit(features=["score"])
    ui._refresh_bar()

    assert _rotulo(ui, "score").startswith("✓ ")
    assert _rotulo(ui, "garantia").startswith("● ")          # selecionada, fora do modelo
    assert not _rotulo(ui, "ruido")[:2] in ("✓ ", "● ")

    _ver(ui, "score")
    assert "mseg-var-modelo" in _classes(ui._tab_an)
    assert "st-modelo" in ui.out_an_status.value and "modelo treinado" in ui.out_an_status.value

    _ver(ui, "garantia")
    assert "mseg-var-sel" in _classes(ui._tab_an)
    assert "mseg-var-modelo" not in _classes(ui._tab_an)
    assert "próximo re-treino" in ui.out_an_status.value

    _ver(ui, "ruido")
    assert not ({"mseg-var-modelo", "mseg-var-sel"} & _classes(ui._tab_an))
    assert "fora do modelo" in ui.out_an_status.value


def test_model_sem_treino_selecionada_fica_azul(mui):
    ui = mui
    ui.seg.include("score")
    ui._refresh_bar()
    _ver(ui, "score")
    assert "mseg-var-sel" in _classes(ui._tab_an)
    assert "ainda não foi treinado" in ui.out_an_status.value


def test_model_marcas_acompanham_inclusao_e_preservam_selecao(mui):
    ui = mui
    for f in list(ui.seg.included):
        ui.seg.exclude(f)
    ui._refresh_bar()
    _ver(ui, "ruido")
    assert not _rotulo(ui, "ruido").startswith("● ")
    ui.dd_var.value = "ruido"
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_include_var(None)
    assert _rotulo(ui, "ruido").startswith("● ")
    assert ui.dd_var2.value == "ruido"                         # mesma variável em tela
    assert "mseg-var-sel" in _classes(ui._tab_an)


def test_model_marca_convive_com_ordenacao_por_iv(mui):
    ui = mui
    ui.seg.include("score")
    ui.tg_an_iv.value = True
    rot = _rotulo(ui, "score")
    assert rot.startswith("● ") and "(IV " in rot


# ───────────────────────── TreeSegmenterUI ─────────────────────────
@pytest.fixture
def tui():
    pytest.importorskip("ipywidgets")
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.tree import TreeSegmenterUI
    with contextlib.redirect_stdout(io.StringIO()):
        return TreeSegmenterUI(_df(), target="target", task_type="classification",
                               sample_col="amostra", ref_sample="DES", date_col="dt_ref")


def _analisa(ui, feat):
    lbl = next(l for l, f in ui._var_by_label.items() if f == feat)
    ui.dd_var.value = lbl
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_var_analyze(None)


def test_tree_quebra_marca_e_destaca_variavel_usada(tui):
    ui = tui
    assert ui._vars_na_arvore() == {}
    assert not any(l.startswith("✓ ") for l in ui.dd_var.options)

    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
    usadas = ui._vars_na_arvore()
    assert usadas and "ruido" not in usadas
    for lbl, f in ui._var_by_label.items():
        assert lbl.startswith("✓ ") == (f in usadas)

    feat = next(iter(usadas))
    _analisa(ui, feat)
    assert "treeui-var-modelo" in _classes(ui._tab_var)
    assert "st-modelo" in ui.out_var_status.value and "quebra" in ui.out_var_status.value

    _analisa(ui, "ruido")
    assert "treeui-var-modelo" not in _classes(ui._tab_var)
    assert "não é usada" in ui.out_var_status.value


def test_tree_recolher_arvore_tira_destaque_da_variavel_em_tela(tui):
    ui = tui
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
    feat = next(iter(ui._vars_na_arvore()))
    _analisa(ui, feat)
    assert "treeui-var-modelo" in _classes(ui._tab_var)

    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.collapse("root", verbose=False)
        ui._refresh()
    assert ui._vars_na_arvore() == {}
    assert "treeui-var-modelo" not in _classes(ui._tab_var)
    assert not any(l.startswith("✓ ") for l in ui.dd_var.options)
    assert ui._sel_var(warn=False) == feat                    # seleção preservada
