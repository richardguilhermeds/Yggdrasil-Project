"""
Testes de regressão da UI ``ModelSegmenterUI`` (ipywidgets).

Cobrem o bug em que a aba **Análise de variáveis** não atualizava ao trocar a
variável no seletor (`dd_var2`): o observer só sincronizava os controles de bin,
sem re-renderizar tabela/gráficos/cards, então os painéis ficavam presos na
variável anterior até um clique em "Analisar".
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pytest


def make_df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.beta(2.5, 3, n) * 1.4 + 0.3
    renda = rng.gamma(2.0, 1500, n)
    gar = rng.choice(list("ABCD"), n, p=[0.5, 0.22, 0.18, 0.1]).astype(object)
    lg = {"A": 0.0, "B": 0.10, "C": 0.16, "D": 0.30}
    risco = (0.1 + 0.35 * (score - 0.5) + 2e-5 * renda
             + np.array([lg[g] for g in gar]))
    target = (rng.uniform(0, 1, n) < np.clip(risco, 0.02, 0.95)).astype(float)
    meses = pd.date_range("2023-01-01", periods=8, freq="MS")
    df = pd.DataFrame({"score": score, "renda": renda, "garantia": gar,
                       "target": target})
    df["dt_ref"] = rng.choice(meses, size=n)
    df["amostra"] = np.where(df["dt_ref"] >= meses[6], "OOT", "DES")
    return df


def _build(**kw):
    pytest.importorskip("ipywidgets")
    pytest.importorskip("optbinning")
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    df = kw.pop("df", None)
    if df is None:
        df = make_df()
    with contextlib.redirect_stdout(io.StringIO()):
        return ModelSegmenterUI(df, target="target", task_type="classification",
                                sample_col="amostra", ref_sample="DES",
                                date_col="dt_ref")


def _feats(ui):
    return [v for _, v in ui.dd_var2.options]


def test_trocar_variavel_atualiza_analise():
    """Mudar `dd_var2` deve re-renderizar a análise (não ficar na variável antiga)."""
    ui = _build()
    ui.tx_time2.value = ""              # sem safra: pula as séries temporais (rápido)
    feats = _feats(ui)
    assert len(feats) >= 2

    a, b = feats[0], feats[-1]
    ui.dd_var2.value = a
    ui._on_analyze(None)               # primeiro render garantido
    tbl_a = ui.out_an_table.value
    cards_a = ui.out_an_cards.value
    assert tbl_a.strip(), "análise inicial não renderizou a tabela"

    ui.dd_var2.value = b               # troca de variável dispara o observer -> _on_analyze
    tbl_b = ui.out_an_table.value
    cards_b = ui.out_an_cards.value

    assert tbl_b.strip(), "trocar a variável deixou a tabela vazia"
    assert tbl_b != tbl_a, "a tabela não atualizou ao trocar a variável"
    assert cards_b != cards_a, "os cards não atualizaram ao trocar a variável"


def test_on_analyze_noop_sem_paineis():
    """Guarda: `_on_analyze` não pode estourar se chamado antes dos painéis existirem."""
    pytest.importorskip("ipywidgets")
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    obj = ModelSegmenterUI.__new__(ModelSegmenterUI)   # sem __init__/_build
    obj._on_analyze(None)              # não deve levantar
