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

def test_rating_table_status_teste_colorido():
    """A tabela de ratings da UI mostra o teste de calibração (ic_low/ic_high/
    status_teste) com o semáforo pintado pelos tokens semânticos de tema."""
    ui = _build()
    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.fit()
        ui.seg.build_ratings(method="quantil", n_ratings=5)
        ui._render_ratings()
    html = ui.out_rating_table.value
    assert "status_teste" in html and "ic_low" in html
    # célula do semáforo pintada com token de tema (nunca hex fixo)
    assert ("var(--ok-bg)" in html or "var(--warn-bg)" in html
            or "var(--bad-bg)" in html)
    # o CSS do semáforo cobre os três estados com os tokens corretos
    assert "var(--ok-bg)" in ui._semaforo_css("ok")
    assert "var(--warn-bg)" in ui._semaforo_css("atencao")
    assert "var(--bad-bg)" in ui._semaforo_css("alerta")


def test_indicador_separacao_ratings():
    """Aba Ratings & Score: indicador compacto ✅/⚠ da separação estatística
    entre ratings vizinhos ao lado da tabela — ⚠ com sugestão de fusão quando o
    par não separa; HTML apenas com tokens de tema (sem hex fixo)."""
    ui = _build()
    # integração: _render_ratings preenche o widget ao lado da tabela
    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.fit()
        ui.seg.build_ratings(method="quantil", n_ratings=5)
        ui._render_ratings()
    assert ui.out_rating_septest.value.strip()
    assert ("✅" in ui.out_rating_septest.value
            or "⚠" in ui.out_rating_septest.value)
    # casos determinísticos com régua de 2 faixas montada à mão
    seg = ui.seg
    n = len(seg.df)
    seg.rating_labels_ = ["R1", "R2"]
    seg.rating_ = pd.Series(np.where(np.arange(n) < n // 2, "R1", "R2"),
                            index=seg.df.index)
    # 1. alvo com distribuição idêntica nas duas faixas → ⚠ não separa + sugestão
    seg.df["target"] = (np.arange(n) % 2).astype(float)
    html = ui._rating_septest_html()
    assert "⚠" in html and "Sugestão" in html and "R1 × R2" in html
    assert "var(--" in html and "#" not in html          # só tokens de tema
    # 2. faixas bem separadas → ✅ (sem chips nem sugestão)
    seg.df["target"] = (np.arange(n) >= n // 2).astype(float)
    html2 = ui._rating_septest_html()
    assert "✅" in html2 and "⚠" not in html2 and "Sugestão" not in html2


def test_bootstrap_forest_card():
    """Aba Ratings & Score: card do IC bootstrap — aviso amigável sem régua; com a
    régua, o clique renderiza o forest plot (matplotlib), o resumo n_ok/n_tot e a
    tabela com o status pintado por tokens de tema; regenerar a régua invalida a
    saída (pede recálculo)."""
    import re
    ui = _build()
    ui._on_boot(None)                        # sem ratings → aviso, sem exceção
    assert "Gere os ratings" in ui.out_boot.value
    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.fit()
        ui.seg.build_ratings(method="quantil", n_ratings=4)
        ui.sl_boot.value = 200
        ui._on_boot(None)
    html = ui.out_boot.value
    assert "<img" in html                    # forest plot (PNG base64 inline)
    assert "ratings dentro do IC" in html    # resumo de aderência
    assert re.search(r"<b>\d+/\d+</b>", html)
    assert "status (OOT)" in html and "média (DES)" in html
    # status pintado só com tokens semânticos de tema (nunca hex fixo no HTML)
    assert "var(--ok-bg)" in html or "var(--bad-bg)" in html
    # regenerar a régua deixa o IC obsoleto → nota pedindo recálculo
    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.build_ratings(method="quantil", n_ratings=3)
        ui._render_ratings()
    assert "Calcular IC bootstrap" in ui.out_boot.value
    assert "<img" not in ui.out_boot.value


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


def test_metrics_ci_na_tabela_por_checkbox():
    """Checkbox 'IC bootstrap' anexa o IC discreto ao lado do valor na tabela de
    métricas e a linha de qualificação da queda DES→OOT (dentro do ruído ×
    degradação real); desmarcar volta à tabela simples. Antes do treino o toggle
    é inofensivo (nada a renderizar)."""
    ui = _build()
    ui.cb_metrics_ci.value = True          # antes do modelo: não pode estourar
    assert ui.out_metrics.value == ""
    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.fit()
        ui._render_metrics()               # com o checkbox ligado → tabela com IC
    html = ui.out_metrics.value
    assert "var(--sub-ink)'>[" in html                     # IC discreto ao lado do valor
    assert "dentro do ruído" in html or "degradação real" in html
    with contextlib.redirect_stdout(io.StringIO()):
        ui.cb_metrics_ci.value = False     # observer re-renderiza sem IC
    assert "var(--sub-ink)'>[" not in ui.out_metrics.value


def test_shap_local_card():
    """Card SHAP local: 'Calcular SHAP' popula o seletor do dependence; o botão
    renderiza o gráfico; o campo de índice valida a linha (aviso amigável para
    índice inexistente) e o waterfall da observação é renderizado."""
    pytest.importorskip("shap")
    ui = _build()
    # sem modelo treinado: guardas amigáveis, sem exceção nem render
    ui._on_shap_dep(None)
    ui._on_shap_row(None)
    assert ui.out_shap_dep.value == "" and ui.out_shap_row.value == ""
    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.fit()
        ui._on_shap(None)                   # calcula SHAP e popula o seletor
    assert len(ui.dd_shap_dep.options) >= 1
    ui.dd_shap_dep.value = ui.dd_shap_dep.options[0][1]
    ui._on_shap_dep(None)
    assert "<img" in ui.out_shap_dep.value  # dependence renderizado (PNG inline)
    # índice inexistente → aviso amigável, sem gráfico
    ui.tx_shap_row.value = "999999"
    ui._on_shap_row(None)
    assert "<img" not in ui.out_shap_row.value
    assert "existente" in ui.out_shap_row.value
    # índice válido → waterfall renderizado
    ui.tx_shap_row.value = str(ui.seg.df.index[0])
    ui._on_shap_row(None)
    assert "<img" in ui.out_shap_row.value


def test_tuning_cv_e_lambda_na_ui():
    """Gaveta do tuning: controles 'CV (k)' e 'penalizar instabilidade (λ)' com
    defaults desligados; `_on_tune` traduz 0 ⇒ None e repassa cv/stability_penalty
    ao tune_optuna; o resumo final exibe as escolhas."""
    ui = _build()
    assert ui.dd_tune_cv.value == 0 and ui.fl_tune_lambda.value == 0.0
    captured = {}

    def fake_tune(**kw):                   # evita um tuning real (lento) no teste
        captured.update(kw)
        return {"algorithm": kw.get("algorithm"), "metric": "auc", "n_trials": 1,
                "n_failed": 0, "n_pruned": 0, "best_value": 0.75,
                "best_params": {"n_estimators": 100}, "degenerate": False,
                "cancelled": False, "cv": kw.get("cv"), "time_aware": False,
                "stability_penalty": kw.get("stability_penalty"), "pruner": None}

    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.fit()                       # _finish_tune re-renderiza as métricas
        ui.dd_tune_cv.value = 3
        ui.fl_tune_lambda.value = 0.5
        ui.seg.tune_optuna = fake_tune
        ui._on_tune(None)
        ui._tune_thread.join(timeout=60)
    assert captured["cv"] == 3
    assert captured["stability_penalty"] == 0.5
    assert "CV k=3" in ui.out_tune.value and "λ=0.5" in ui.out_tune.value
    assert "média dos folds" in ui.out_tune.value


def test_champion_challenger_card():
    """Aba Modelo: 'Congelar como baseline' fotografa o modelo em memória; o
    snapshot SOBREVIVE ao re-treino e 'Comparar com o baseline' renderiza os
    deltas com semáforo (tokens de tema), PSI dos scores e diff de config; um
    novo re-treino marca a comparação como obsoleta sem perder o baseline."""
    ui = _build()
    # sem modelo: congelar avisa e não cria snapshot
    ui._on_freeze(None)
    assert ui.seg.snapshots_ == {}
    assert "Treine" in ui.out_baseline.value
    assert ui.btn_compare_base.disabled is True
    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.fit()
        ui._on_freeze(None)
    assert "baseline" in ui.seg.snapshots_
    assert "❄" in ui.out_baseline.value
    assert ui.btn_compare_base.disabled is False
    # re-treina com OUTRA config via UI — o baseline sobrevive
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_algo.value = "random_forest"
        ui.sl_n_est.value = 60
        ui._on_fit(None)
    assert "baseline" in ui.seg.snapshots_
    ui._on_compare_baseline(None)
    html = ui.out_compare_base.value
    assert "<table" in html and "veredicto" in html
    assert ("var(--ok-bg)" in html or "var(--bad-bg)" in html
            or "var(--neutral-bg)" in html)          # semáforo por tokens de tema
    assert "PSI" in html
    assert "<img" in html                            # ROC sobreposta (classificação)
    # novo re-treino: comparação vira nota de obsolescência; baseline permanece
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_fit(None)
    assert "Comparar com o baseline" in ui.out_compare_base.value
    assert "<table" not in ui.out_compare_base.value
    assert "baseline" in ui.seg.snapshots_ and "❄" in ui.out_baseline.value


def test_card_comparar_modelo_salvo(tmp_path):
    """Aba Validar & Exportar: card 'Comparar com modelo salvo' — avisos
    amigáveis sem caminho/modelo e, com um .json salvo, renderiza concordância,
    matriz de migração de ratings e as tabelas de diff."""
    ui = _build()
    ui._on_diff(None)                                # sem caminho → aviso
    assert "Informe o caminho" in ui.out_diff.value
    ui.tx_diff_path.value = "qualquer.json"
    ui._on_diff(None)                                # sem modelo vigente → aviso
    assert "Treine" in ui.out_diff.value
    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.fit()
        ui.seg.build_ratings(method="quantil", n_ratings=4)
        p = tmp_path / "modelo_b.json"
        ui.seg.save(str(p))
        ui.tx_diff_path.value = str(p)
        ui._on_diff(None)
    html = ui.out_diff.value
    assert "Concordância" in html
    assert "Matriz de migração" in html
    assert "<table" in html
    # caminho inexistente → erro amigável, sem exceção
    ui.tx_diff_path.value = str(tmp_path / "nao_existe.json")
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_diff(None)
    assert "Erro ao comparar" in ui.out_diff.value


def test_card_calibracao_aplica_remove_e_retreino_limpa():
    """Aba Modelo · card 'Calibração do score': aplicar ajusta a camada no seg
    (média casa a taxa da amostra), renderiza status + antes×depois; 'Remover'
    volta ao score cru e limpa as saídas; um re-treino descarta a camada e o
    card acompanha."""
    ui = _build()
    # sem modelo: aplicar avisa e não cria camada
    ui._on_calibrate(None)
    assert getattr(ui.seg, "calibration_", None) is None
    assert "Treine" in ui.out_calib_status.value
    with contextlib.redirect_stdout(io.StringIO()):
        # classes balanceadas descalibram a média de propósito
        ui.seg.fit("logistica", class_balance=True)
    # tendência-alvo inválida → erro amigável, sem camada
    ui.tx_calib_target.value = "abc"
    ui._on_calibrate(None)
    assert "inválida" in ui.out_calib_status.value
    assert ui.seg.calibration_ is None
    # aplica o intercepto sem alvo ⇒ casa a taxa observada na amostra do ajuste
    ui.tx_calib_target.value = ""
    ui.dd_calib.value = "intercept"
    ui._on_calibrate(None)
    assert ui.seg.calibration_ is not None
    assert "Calibração vigente" in ui.out_calib_status.value
    assert "<table" in ui.out_calib_table.value      # brier/logloss antes×depois
    assert "<img" in ui.out_calib_plot.value         # plot de calibração
    mask = ui.seg._frame_mask("DES")
    taxa = float(ui.seg.df.loc[mask, "target"].mean())
    assert abs(float(ui.seg.score_[mask].mean()) - taxa) < 1e-6
    # remover volta ao cru e limpa as saídas do card
    ui._on_decalibrate(None)
    assert ui.seg.calibration_ is None
    assert "removida" in ui.out_calib_status.value
    assert ui.out_calib_table.value == "" and ui.out_calib_plot.value == ""
    # re-aplica e re-treina pela UI: a camada morre e o card é limpo
    ui._on_calibrate(None)
    assert ui.seg.calibration_ is not None
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_fit(None)
    assert ui.seg.calibration_ is None
    assert ui.out_calib_status.value == ""
    assert ui.out_calib_table.value == "" and ui.out_calib_plot.value == ""


def test_card_redundancia():
    """Aba Variáveis · card de multicolinearidade: 'Analisar redundância' lista o
    par quase-duplicado (tabela + heatmap) e habilita a poda; 'Excluir
    redundantes' exclui do modelo a variável de menor IV do par pelo fluxo
    normal (lista sincronizada, categoria 'descartar').

    Correlação e VIF ficam LADO A LADO. Sem modelo treinado o painel de VIF não
    some: ele explica que o VIF é lido na matriz de desenho e por isso exige um
    ajuste feito — some-lo deixava o usuário sem saber por que não aparecia."""
    df = make_df()
    rng = np.random.default_rng(7)
    df["score_dup"] = df["score"] + rng.normal(0, 0.005, len(df))   # ρ ≈ 1
    ui = _build(df=df)
    assert ui.btn_redund_drop.disabled is True          # sem análise ainda
    ui.fl_redund_thr.value = 0.95
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_redund(None)
    html = ui.out_redund.value
    assert "<table" in html and "<img" in html          # pares + heatmap
    assert "display:flex" in html                       # correlação | VIF lado a lado
    assert "VIF" in html and "Treine um modelo" in html  # sem modelo: orienta, não some
    assert ui.btn_redund_drop.disabled is False
    poda = list(ui._redund_report.attrs["poda_sugerida"])
    assert len(poda) == 1 and poda[0] in ("score", "score_dup")
    incl_antes = set(ui.seg.included)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_redund_drop(None)
    assert set(ui.seg.included) == incl_antes - set(poda)
    assert ui.seg.var_meta[poda[0]]["categoria"] == "descartar"
    assert "redundante" in ui.seg.var_meta[poda[0]]["motivo"]
    assert ui.btn_redund_drop.disabled is True          # poda já aplicada
    assert "Poda aplicada" in ui.out_redund.value
    # com modelo logístico vigente, a análise passa a incluir o VIF
    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.fit("logistica")
        ui._on_redund(None)
    assert "VIF" in ui.out_redund.value


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


def test_undo_redo_da_configuracao():
    """Barra ↶/↷: auto-selecionar é desfazível (o desfazer devolve o conjunto
    anterior de variáveis), o refazer reaplica, e uma exceção DENTRO da ação faz
    rollback automático da configuração sem sujar o histórico."""
    ui = _build()
    assert ui.btn_undo.disabled is True and ui.btn_redo.disabled is True
    antes = set(ui.seg.included)
    ui.sl_min_iv.value = 0.9                    # critério impossível → esvazia
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_auto_select(None)
    depois = set(ui.seg.included)
    assert depois != antes and ui.btn_undo.disabled is False
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_undo(None)
    assert set(ui.seg.included) == antes
    assert ui.btn_redo.disabled is False
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_redo(None)
    assert set(ui.seg.included) == depois
    # ação que levanta: estado E pilhas voltam ao que eram (rollback automático)
    n_undo, n_redo = len(ui._undo), len(ui._redo)
    estado = set(ui.seg.included)

    def _boom(*a, **kw):
        ui.seg.included = set()                 # mutação parcial antes do erro
        raise RuntimeError("falha simulada")

    ui.seg.auto_select = _boom
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_auto_select(None)
    assert set(ui.seg.included) == estado
    assert (len(ui._undo), len(ui._redo)) == (n_undo, n_redo)


def test_undo_cobre_derivada_e_transformacao():
    """O snapshot é de CONFIGURAÇÃO: desfazer remove a variável derivada do
    DataFrame/candidatas (e o refazer a recria) e devolve a transformação
    (valores crus ↔ WoE/bins) do treino."""
    ui = _build()
    cands0 = list(ui.seg.candidates)
    ui.dd_var2.value = "score"
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_create_cat(None)
    nova = [c for c in ui.seg.candidates if c not in cands0]
    assert len(nova) == 1 and nova[0] in ui.seg.df.columns
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_undo(None)
    assert list(ui.seg.candidates) == cands0 and nova[0] not in ui.seg.df.columns
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_redo(None)
    assert nova[0] in ui.seg.candidates and nova[0] in ui.seg.df.columns
    # trocar a transformação também é desfazível (o toggle volta ao valor antigo)
    ui.cb_woe.value = True
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_undo(None)
    assert ui.cb_woe.value is False
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_redo(None)
    assert ui.cb_woe.value is True


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_placar_saude_quatro_vereditos(task):
    """Placar de saúde na aba Modelo: 4 vereditos com a evidência numérica,
    renderizando em classificação E em regressão (paridade clf×reg), pintado só
    com tokens semânticos de tema e invalidado quando o modelo muda."""
    ui = _build() if task == "classification" else _build_reg()
    ui._on_diag(None)                      # sem modelo treinado → aviso amigável
    assert "Treine" in ui.out_diag.value

    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.fit()
        ui.seg.build_ratings(method="quantil", n_ratings=4)
        ui._on_diag(None)
    html = ui.out_diag.value
    for dim in ("Discriminação", "Estabilidade", "Calibração", "Estrutura"):
        assert dim in html, f"veredito ausente: {dim}"
    # evidência numérica de cada dimensão (métrica da tarefa, PSI e estrutura)
    assert ("AUC" in html) if task == "classification" else ("R²" in html)
    assert "PSI da régua" in html and "inversão(ões)" in html
    assert "pares sem separação" in html
    # vereditos coloridos só com tokens de tema (nunca hex fixo)
    assert "border-left:4px solid var(--" in html
    assert any(t in html for t in ("var(--ok-bg)", "var(--warn-bg)", "var(--bad-bg)"))

    # mudança de configuração ⇒ placar renderizado vira "desatualizado"
    ui._mark_dirty()
    assert "desatualizado" in ui.out_diag.value
    # re-treino limpa a saída (placar do modelo antigo sai da tela)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_fit(None)
    assert ui.out_diag.value == ""
    # botão "Ocultar" limpa o placar já renderizado
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_diag(None)
    assert ui.out_diag.value.strip()
    ui._on_diag_hide(None)
    assert ui.out_diag.value == ""


def test_card_sql_da_regua_de_ratings():
    """Aba Validar & Exportar: card do CASE WHEN da régua de ratings — guarda sem
    régua, nomes de tabela/colunas do formulário e invalidação da saída."""
    ui = _build()
    ui._on_sql(None)                       # sem régua de ratings → aviso na caixa
    assert "Gere os ratings" in ui.out_sql.value
    with contextlib.redirect_stdout(io.StringIO()):
        ui.seg.fit()
        ui.seg.build_ratings(method="quantil", n_ratings=4)
    ui.tx_sql_table.value = "cat.esq.carteira"
    ui.tx_sql_score.value = "pontuacao"
    ui.tx_sql_rating.value = "faixa_risco"
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_sql(None)
    sql = ui.out_sql.value
    assert "FROM cat.esq.carteira;" in sql and "END AS faixa_risco" in sql
    assert "pontuacao" in sql and "AS valor_previsto" in sql
    for lab in ui.seg.rating_labels_:
        assert f"THEN '{lab}'" in sql
    ui._mark_dirty()                       # modelo alterado ⇒ SQL desatualizado
    assert "desatualizado" in ui.out_sql.value


# ----------------------------------------------------------------------
# Aba Variáveis · card "Esteira de seleção"
# ----------------------------------------------------------------------
def test_card_esteira_selecao_roda_e_desfaz():
    """O card monta com as etapas pré-marcadas conforme STEPS_DEFAULT (rótulo em
    pt-BR + descrição no tooltip); 'Rodar seleção' executa a esteira, aplica a
    decisão no modelo, preenche progresso/funil/decisões (com as cores semânticas
    de tema) e é DESFAZÍVEL pelo ↶."""
    from yggdrasil.credit_risk.model.selection import SELECTION_STEPS, STEPS_DEFAULT

    ui = _build()
    marcadas = [n for n, cb in ui._sel_step_cbs.items() if cb.value]
    assert marcadas == list(STEPS_DEFAULT)
    assert set(ui._sel_step_cbs) == set(SELECTION_STEPS)
    for nome, cb in ui._sel_step_cbs.items():
        assert cb.description == SELECTION_STEPS[nome].rotulo
        assert cb.tooltip == SELECTION_STEPS[nome].descricao

    antes = set(ui.seg.included)
    n_undo = len(ui._undo)
    ui.fl_sel_min_iv.value = 0.90            # régua impossível → exclui tudo
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_selection_run(None)

    assert ui.seg.selection_ is not None
    assert set(ui.seg.included) != antes
    assert len(ui._undo) == n_undo + 1                     # passou pelo _checkpoint
    assert "Seleção aplicada" in ui.out_sel_status.value
    assert "Progresso da esteira" in ui.out_sel_progress.value
    assert any(r["status"] == "ok" for r in ui._sel_progress)
    corpo = ui.out_sel_result.value
    assert "Funil por etapa" in corpo and "Decisão por variável" in corpo
    assert "<img" in corpo                                 # gráficos do relatório
    assert "var(--bad-bg)" in corpo                        # decisão pintada por token
    # nenhuma cor hex fixa no HTML do card (só tokens semânticos de tema) — os
    # '#T_xxxx' do Styler são seletores de id, não cores
    import re
    assert re.search(r"#[0-9a-fA-F]{3,8}\b", corpo.split("base64,")[0]) is None

    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_undo(None)
    assert set(ui.seg.included) == antes


def test_esteira_selecao_modo_simular_nao_muda_nada():
    """'apenas simular' roda a esteira e mostra o resultado sem tocar no
    segmentador (seleção intacta e nada empilhado no desfazer)."""
    ui = _build()
    antes = set(ui.seg.included)
    cats = {f: (ui.seg.var_meta[f] or {}).get("categoria") for f in ui.seg.candidates}
    n_undo = len(ui._undo)
    ui.cb_sel_simular.value = True
    ui.fl_sel_min_iv.value = 0.90
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_selection_run(None)

    assert set(ui.seg.included) == antes
    assert {f: (ui.seg.var_meta[f] or {}).get("categoria")
            for f in ui.seg.candidates} == cats
    assert len(ui._undo) == n_undo                          # simulação não é ponto de retorno
    assert "Simulação" in ui.out_sel_status.value
    assert ui.seg.selection_.politica["aplicado"] is False
    assert "Decisão por variável" in ui.out_sel_result.value


def test_esteira_selecao_sem_etapa_marcada():
    """Nenhuma etapa marcada: aviso no card e nada é executado."""
    ui = _build()
    for cb in ui._sel_step_cbs.values():
        cb.value = False
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_selection_run(None)
    assert "ao menos" in ui.out_sel_status.value
    assert ui.seg.selection_ is None


def _ui_com_selecao(steps=("missing", "iv")):
    """UI com uma esteira curta já rodada em modo simulação (teste rápido)."""
    ui = _build()
    ui.cb_sel_simular.value = True
    for nome, cb in ui._sel_step_cbs.items():
        cb.value = nome in steps
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_selection_run(None)
    return ui


def test_esteira_selecao_exporta_relatorio_html(tmp_path):
    """Botão 'Relatório (.html)': grava a página autocontida (completando o
    sufixo do caminho) e confirma no card."""
    ui = _ui_com_selecao()
    destino = tmp_path / "selecao"                     # sem sufixo: a UI completa
    ui.tx_sel_html.value = str(destino)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_selection_report(None)
    html = destino.with_suffix(".html")
    assert html.exists() and "Relatório salvo" in ui.out_sel_export.value
    assert html.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_esteira_selecao_exporta_excel(tmp_path):
    """Botão 'Exportar Excel': grava o .xlsx multi-abas (openpyxl é OPCIONAL —
    sem ele o teste é pulado e a UI mostra o ImportError amigável)."""
    pytest.importorskip("openpyxl")
    ui = _ui_com_selecao()
    xlsx = tmp_path / "selecao.xlsx"
    ui.tx_sel_xlsx.value = str(xlsx)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_selection_xlsx(None)
    assert xlsx.exists() and "Excel salvo" in ui.out_sel_export.value
    abas = pd.read_excel(xlsx, sheet_name=None)
    assert set(abas) == {"Decisoes", "Funil", "Politica"}
