"""Onda 3C — cache das figuras da aba Análise (e da prévia da aba Variáveis)
chaveado pela ASSINATURA da própria variável, no lugar do ``_rank_version``
global: editar outra variável não invalida; editar a própria, recriar a
derivada com o mesmo nome ou mudar a amostra dos gráficos invalida; trocar o
tema reaproveita. LRU com limite de quantidade e de tamanho do HTML."""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import re
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ipywidgets")
pytest.importorskip("optbinning")


def _df(n=2500, seed=4, regressao=False):
    rng = np.random.default_rng(seed)
    xc = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(xc - .3)))
    y = (np.clip(p + rng.normal(0, .1, n), 0, 1) if regressao
         else (rng.random(n) < p).astype(float))
    meses = [dt.date(2023 + m // 12, m % 12 + 1, 1) for m in range(10)]
    df = pd.DataFrame({
        "amostra": np.where(rng.random(n) < .7, "DES", "OOT"),
        "target": y,
        "a_num": np.where(rng.random(n) < .1, np.nan, xc + rng.normal(0, .5, n)),
        "b_num": rng.gamma(2.0, 3.0, n) + (xc > 0) * 2,
        "c_cat": rng.choice(list("ABCD"), n).astype(object),
    })
    df["dt_ref"] = np.array(meses, dtype=object)[rng.integers(0, 10, n)]
    return df


def _ui(regressao=False):
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui = ModelSegmenterUI(_df(regressao=regressao), target="target",
                              task_type="regression" if regressao else "classification",
                              sample_col="amostra", ref_sample="DES", date_col="dt_ref")
    ui.tx_time2.value = "dt_ref"
    return ui


class _Espiao:
    """Conta as chamadas de plot_variable_* / variable_table / variable_summary
    por variável (1º argumento) — sem as da prévia da aba Variáveis (risco por
    safra / logodds), que tem cache próprio."""

    def __init__(self, ui, monkeypatch):
        self.n = {}
        seg = ui.seg
        nomes = [m for m in dir(type(seg)) if m.startswith("plot_variable_")
                 and m not in ("plot_variable_risk_by_safra", "plot_variable_logodds")]
        for m in nomes + ["variable_table", "variable_summary"]:
            orig = getattr(seg, m)

            def w(feature, *a, _o=orig, **k):
                self.n[feature] = self.n.get(feature, 0) + 1
                return _o(feature, *a, **k)
            monkeypatch.setattr(seg, m, w)

    def de(self, f):
        return self.n.get(f, 0)


def _analisa(ui, feat):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if ui.dd_var2.value != feat:
            ui._refresh_an_var_options()
            ui.dd_var2.value = feat                     # o observer já analisa
        else:
            ui._on_analyze(None)
    return _valores(ui)


def _valores(ui):
    return tuple(w.value for w in (ui.out_an_distbad, ui.out_an_table, ui.out_an_inv_sample,
                                   ui.out_an_cards, ui.out_an_time, ui.out_an_inv_safra,
                                   ui.out_an_psi, ui.out_an_optbin_share))


def _norm(vals):
    """O Styler sorteia o id da tabela (``#T_xxxxx``) a cada render."""
    return tuple(re.sub(r"T_[0-9a-f]{5}", "T_id", v) for v in vals)


def _forcado(ui, feat):
    """Render do zero (sem o cache de figuras) — a referência dos 8 values
    (normalizados por :func:`_norm`)."""
    salvo = ui._an_fig_cache
    ui._an_fig_cache = type(salvo)(salvo.max_itens, salvo.max_bytes)
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ui._on_analyze(None)
        return _norm(_valores(ui))
    finally:
        ui._an_fig_cache = salvo


def _quieto(fn):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


@pytest.mark.parametrize("regressao", [False, True])
def test_editar_outra_variavel_nao_invalida(regressao, monkeypatch):
    ui = _ui(regressao)
    seg = ui.seg
    B, A = "b_num", "a_num"
    vb = _analisa(ui, B)
    assert all(v.strip() for v in vb)
    esp = _Espiao(ui, monkeypatch)

    acoes = [
        lambda: seg.set_manual_bins(A, [-0.5, 0.5]),
        lambda: seg.set_missing_bin(A, "pior"),
        lambda: seg.create_categorical(A),
        lambda: seg.create_scorecard_dummies(A),
        lambda: seg.remove_derived(seg.create_categorical(A, new_name="a_tmp")),
        lambda: seg.set_manual_bins("c_cat", [["A", "B"], ["C", "D"]]),
        lambda: seg.clear_manual_bins(A),
    ]
    for acao in acoes:
        with ui._undoable("teste"):
            _quieto(acao)
        assert _analisa(ui, B) == vb
        assert esp.de(B) == 0
    # desfazer/refazer (o _restore subia o _rank_version global)
    for _ in range(3):
        _quieto(lambda: ui._on_undo(None))
        assert _analisa(ui, B) == vb
    for _ in range(2):
        _quieto(lambda: ui._on_redo(None))
        assert _analisa(ui, B) == vb
    assert esp.de(B) == 0
    # o hit é igual a um render forçado do zero (mesmo HTML/PNG)
    assert _forcado(ui, B) == _norm(vb)
    assert esp.de(B) > 0


def test_editar_a_propria_variavel_invalida(monkeypatch):
    ui = _ui()
    seg = ui.seg
    B = "b_num"
    _analisa(ui, B)
    esp = _Espiao(ui, monkeypatch)
    passos = [
        lambda: seg.set_manual_bins(B, [3.0, 6.0]),
        lambda: seg.set_missing_bin(B, "melhor"),
        lambda: seg.feature_labels.__setitem__(B, "Renda B"),
        lambda: seg.clear_manual_bins(B),
    ]
    for passo in passos:
        _quieto(passo)
        antes = esp.de(B)
        v = _analisa(ui, B)
        assert esp.de(B) > antes                       # miss: re-renderiza
        assert _norm(v) == _forcado(ui, B)


def test_derivada_recriada_com_mesmo_nome_invalida(monkeypatch):
    ui = _ui()
    seg = ui.seg
    _quieto(lambda: seg.set_manual_bins("a_num", [-0.5, 0.5]))
    nome = _quieto(lambda: seg.create_categorical("a_num", new_name="a_der"))
    v1 = _analisa(ui, nome)
    k1 = ui._assinatura_variavel(nome)
    _quieto(lambda: seg.remove_derived(nome))
    _quieto(lambda: seg.set_manual_bins("a_num", [0.0, 1.0]))
    assert _quieto(lambda: seg.create_categorical("a_num", new_name="a_der")) == nome
    assert ui._assinatura_variavel(nome) != k1
    esp = _Espiao(ui, monkeypatch)
    v2 = _analisa(ui, nome)
    assert esp.de(nome) > 0 and v2 != v1
    assert _norm(v2) == _forcado(ui, nome)


def test_tema_reaproveita_e_amostra_dos_graficos_invalida(monkeypatch):
    ui = _ui()
    B = "b_num"
    _analisa(ui, B)
    esp = _Espiao(ui, monkeypatch)
    ui.cb_dark.value = True                            # troca de tema: hit, HTML remapeado
    v_escuro = _analisa(ui, B)
    assert esp.de(B) == 0
    assert ui._an_fig_cache.busca(next(k for k in ui._an_fig_cache if k[0] == B)) == v_escuro
    ui.cb_dark.value = False
    # teto da amostra dos gráficos abaixo da base: outra amostra ⇒ miss
    monkeypatch.setattr(ui.seg, "max_linhas_graficos", 1000)
    _analisa(ui, B)
    assert esp.de(B) > 0


def test_previa_da_aba_variaveis_por_variavel(monkeypatch):
    ui = _ui()
    seg = ui.seg
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_var.value = "b_num"
        ui._refresh_var_preview()
    html = ui.out_var_preview.value
    chamadas = []
    orig = seg.plot_variable_risk_by_safra
    monkeypatch.setattr(seg, "plot_variable_risk_by_safra",
                        lambda f, *a, **k: chamadas.append(f) or orig(f, *a, **k))
    _quieto(lambda: seg.set_manual_bins("a_num", [-0.5, 0.5]))   # outra variável
    _quieto(ui._refresh_var_preview)
    assert chamadas == [] and ui.out_var_preview.value == html
    _quieto(lambda: seg.set_manual_bins("b_num", [3.0, 6.0]))    # a própria
    _quieto(ui._refresh_var_preview)
    assert chamadas == ["b_num"]


def test_lru_por_quantidade_e_tamanho():
    from yggdrasil.credit_risk.model.ui import _CacheHtmlLRU
    c = _CacheHtmlLRU(max_itens=3, max_bytes=100)
    for i in range(4):
        c.guarda(i, ("x" * 10, "y" * 5))
    assert list(c) == [1, 2, 3] and c.n_bytes == 45
    assert c.busca(1) is not None and list(c) == [2, 3, 1]      # uso move p/ o fim
    c.guarda(4, ("z" * 60,))                                     # estoura bytes
    assert list(c) == [3, 1, 4] and c.n_bytes == 90
    c.guarda(5, "w" * 500)                                       # sozinha > limite: fica
    assert list(c) == [5] and c.n_bytes == 500
    d = c.remapeia(lambda v: v.upper())
    assert isinstance(d, _CacheHtmlLRU) and d[5] == "W" * 500 and d.n_bytes == 500
    d.clear()
    assert d.n_bytes == 0 and not d
