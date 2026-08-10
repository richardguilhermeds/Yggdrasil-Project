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


def _build_reg(df=None):
    pytest.importorskip("ipywidgets")
    pytest.importorskip("optbinning")
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    if df is None:
        rng = np.random.default_rng(1)
        n = 2500
        X = {f"x{k}": rng.normal(size=n) for k in range(4)}
        lin = sum((k + 1) * 0.1 * X[f"x{k}"] for k in range(4))
        df = pd.DataFrame(X)
        df["target"] = np.clip(0.5 + 0.1 * lin + rng.normal(0, 0.1, n), 0, 1)
        meses = pd.date_range("2023-01-01", periods=8, freq="MS")
        df["dt_ref"] = rng.choice(meses, size=n)
        df["amostra"] = np.where(df["dt_ref"] >= meses[6], "OOT", "DES")
    with contextlib.redirect_stdout(io.StringIO()):
        return ModelSegmenterUI(df, target="target", task_type="regression",
                                sample_col="amostra", ref_sample="DES", date_col="dt_ref")


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_backward_cache_invalidado_no_retreino(task):
    """Retreinar o modelo deve DESCARTAR o resultado cacheado do backward elimination.

    Bug: `_backelim_result`/`_backelim_optimal` não eram invalidados no retreino, então
    a 'escolha ótima'/'aplicar Nº' reaplicava a seleção calculada sobre o modelo ANTERIOR
    (a guarda de identidade só olhava features+amostra, ignorando o algoritmo). Ocorria
    tanto em classificação quanto em regressão.
    """
    ui = _build() if task == "classification" else _build_reg()
    feats0 = list(ui.seg.selected_features() or ui.seg.model_features or ui.seg.candidates)
    assert len(feats0) >= 2
    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.fit(features=feats0)
        ui._backelim_result = ui.seg.backward_elimination(
            sample=ui.dd_backelim_sample.value, min_features=1, features=feats0)
        assert ui._backelim_result is not None
        ui._on_fit(None)                       # retreino via UI dispara a invalidação
    assert ui._backelim_result is None, "o cache do backward deveria ser invalidado no retreino"
    assert getattr(ui, "_backelim_optimal", None) is None


def test_backward_guarda_por_algoritmo():
    """A guarda de reuso da 'escolha ótima' deve rejeitar um resultado de OUTRO algoritmo
    (mesmas features e amostra) — senão reusaria a ordem/métricas do modelo antigo."""
    ui = _build()
    feats0 = list(ui.seg.selected_features() or ui.seg.model_features or ui.seg.candidates)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.fit(algorithm="logistica", features=feats0)
        res = ui.seg.backward_elimination(sample=ui.dd_backelim_sample.value,
                                          min_features=1, features=feats0)
        ui.seg.fit(algorithm="random_forest", features=feats0)   # troca só o algoritmo
    feats_now = list(ui.seg.selected_features() or ui.seg.model_features or ui.seg.candidates)
    same = (res is not None and len(res)
            and set(res.attrs.get("feats0", [])) == set(feats_now)
            and res.attrs.get("eval_sample") == ui.dd_backelim_sample.value
            and res.attrs.get("algorithm") == ui.seg.algorithm
            and res.attrs.get("transform") == ui.seg.feature_transform
            and res.attrs.get("hyperparams") == dict(ui.seg.hyperparams or {}))
    assert same is False, "não deveria reusar um backward de 'logistica' num modelo 'random_forest'"


def test_backward_bloqueado_em_two_stage():
    """Rodar o backward num modelo Two-Stage (hurdle) não deve crashar — bloqueia com aviso
    (antes: ValueError 'Algoritmo desconhecido: two_stage:...' dentro do worker)."""
    ui = _build_reg()
    feats0 = list(ui.seg.selected_features() or ui.seg.model_features or ui.seg.candidates)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.fit_two_stage(threshold=0.5, features=feats0)
        ui._on_backelim(None)              # não deve iniciar a execução nem levantar
    assert "Two-Stage" in ui.out_backelim_status.value
    assert ui._backelim_thread is None


def test_toggle_balancear_classes_clf():
    """Classificação: o toggle aparece com a taxa de evento da referência ao lado
    e o fit via UI repassa class_balance ao segmenter (class_weight='balanced')."""
    ui = _build()
    assert ui.row_balance.layout.display != "none"
    assert "taxa de evento" in ui.lb_balance.value
    ui.cb_balance.value = True
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_fit(None)
    assert ui.seg.class_balance is True
    assert getattr(ui.seg.model.named_steps["est"], "class_weight", None) == "balanced"
    assert "classes balanceadas" in ui.out_fit_status.value


def test_toggle_balancear_classes_oculto_na_regressao():
    """Regressão: o toggle fica oculto e o fit segue sem balanceamento."""
    ui = _build_reg()
    assert ui.row_balance.layout.display == "none"
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_fit(None)
    assert getattr(ui.seg, "class_balance", False) is False


def test_grupo_avancado_metricas_e_ratings():
    """Aba Avançado: dropdown de colunas de contexto (não-features) + botões de
    métricas e ratings por grupo renderizam tabelas; ratings exigem a régua."""
    ui = _build()
    # 'amostra' (sample_col) é coluna categórica de contexto; features ficam fora
    assert "amostra" in list(ui.dd_group_col.options)
    assert "score" not in list(ui.dd_group_col.options)
    ui.dd_group_col.value = "amostra"
    # sem modelo treinado → aviso amigável
    ui._on_adv_group_metrics(None)
    assert "Treine o modelo" in ui.out_adv_group_metrics.value
    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.fit()
        ui._on_adv_group_metrics(None)
    assert "<table" in ui.out_adv_group_metrics.value
    assert "DES" in ui.out_adv_group_metrics.value
    # ratings por grupo exigem a régua construída
    ui._on_adv_group_rating(None)
    assert "Gere os ratings" in ui.out_adv_group_rating.value
    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.build_ratings(method="quantil", n_ratings=4)
        ui._on_adv_group_rating(None)
    assert "<table" in ui.out_adv_group_rating.value
    # re-treinar limpa as saídas da aba Avançado (modelo antigo fora da tela)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_fit(None)
    assert ui.out_adv_group_metrics.value == ""
    assert ui.out_adv_group_rating.value == ""

def test_toggle_monotonicidade():
    """A linha 'restrições de monotonicidade' só aparece nos boostings com
    suporte nativo, e o fit via UI aplica monotonic_cst (dict por nome) com as
    direções da tendência univariada."""
    ui = _build()
    from yggdrasil.credit_risk.model.segmenter import MONOTONE_ALGORITHMS
    ui.dd_algo.value = "logistica"                    # sem suporte → linha oculta
    assert ui.row_monotone.layout.display == "none"
    ui.dd_algo.value = "hist_gradient_boosting"
    assert "hist_gradient_boosting" in MONOTONE_ALGORITHMS
    assert ui.row_monotone.layout.display != "none"
    ui.cb_monotone.value = True
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_fit(None)
    cst = ui.seg.model.named_steps["est"].monotonic_cst
    assert cst and all(v in (-1, 1) for v in cst.values())
    # 'score' tem relação positiva clara com o alvo no make_df
    assert cst.get("num__score") == 1
    assert ui.seg.monotone == "auto"
    assert "monotonicidade" in ui.out_fit_status.value


def test_toggle_monotonicidade_ignorado_sem_suporte():
    """Toggle marcado + algoritmo sem suporte: o fit via UI NÃO repassa a opção
    (nem aviso, nem restrição) — a linha fica oculta, mas o valor persiste."""
    ui = _build()
    ui.dd_algo.value = "hist_gradient_boosting"
    ui.cb_monotone.value = True
    ui.dd_algo.value = "logistica"                    # troca esconde a linha
    assert ui.row_monotone.layout.display == "none"
    import warnings as _w
    with contextlib.redirect_stdout(io.StringIO()), \
            _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        ui._on_fit(None)
    assert not [w for w in rec if "monotonicidade" in str(w.message)]
    assert "✓" in ui.out_fit_status.value             # fit concluiu normalmente
    assert ui.seg.algorithm == "logistica"
    assert ui.seg.monotone is None and ui.seg.monotone_dirs_ == {}
