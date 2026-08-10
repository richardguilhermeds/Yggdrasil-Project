"""
Testes da interface de modelos satélite (:class:`SatelliteUI`).

Cobrem o esqueleto e as abas já preenchidas: construção a partir de
``RiskSeries`` / ``pandas.Series`` + macro / sem dados, validação do alinhamento
temporal, nomes definitivos das abas, relatório de estacionariedade, ajuste único
com tabela de coeficientes, o ida-e-volta com :class:`StudyConfig`, a **busca
champion-challenger** (dimensionamento da grade, ranking, motivo do descarte,
escolha manual e Diebold-Mariano), a **bateria de diagnóstico** (placar por
família, tabela completa, leitura em texto e invalidação), os **cenários e a
projeção** (padrão, choque com persistência, colagem da trajetória, leque,
projeção ponderada e exportação) e o **backtest de cobertura** (erro por
horizonte, Kupiec/Christoffersen, veredito e gráfico das violações).

Tudo roda em séries sintéticas **curtas** e grades minúsculas (2 candidatas, 2
defasagens, 1 variável por especificação) — a busca completa é do usuário, não
da suíte.
"""
from __future__ import annotations

import contextlib
import io
import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ipywidgets")
pytest.importorskip("statsmodels")


def _ui(**kwargs):
    """Constrói a interface com matplotlib em modo Agg (sem janela)."""
    import matplotlib

    matplotlib.use("Agg")
    from yggdrasil.credit_risk.econometric.ui import SatelliteUI

    with contextlib.redirect_stdout(io.StringIO()):
        return SatelliteUI(**kwargs)


def _sintetico(n=60, seed=3):
    """Série de PD sintética curta + a macro que a dirigiu."""
    from yggdrasil.credit_risk.econometric import simulate_pd_series

    return simulate_pd_series(seed=seed, n_periods=n)


# ======================================================================
# Construção
# ======================================================================
def test_constroi_com_risk_series():
    syn = _sintetico()
    ui = _ui(series=syn.series, macro=syn.macro)
    assert ui.series is syn.series
    assert ui.kind == "pd"
    assert ui.macro is syn.macro
    # estado esperado pelos demais blocos da interface
    for attr in ("fit_", "model_", "search_", "projection_", "scenarios_", "backtest_"):
        assert getattr(ui, attr) is None
    # cabeçalho e gráficos da aba Série renderizaram
    assert "observações" in ui.out_serie_head.value
    assert "<img" in ui.out_serie_nivel.value and "<img" in ui.out_serie_link.value


def test_constroi_com_series_e_macro_crus():
    syn = _sintetico()
    serie = syn.series.values          # pandas.Series pura, sem RiskSeries
    ui = _ui(series=serie, macro=syn.macro, kind="lgd", segment="varejo")
    assert ui.kind == "lgd"
    assert ui.series.segment == "varejo"
    assert len(ui.series) == len(serie)
    assert set(ui.candidates()) == set(syn.macro.columns)   # todas candidatas por padrão


def test_constroi_a_partir_de_synthetic_series():
    """``SyntheticSeries`` traz série e macro juntas — a interface aceita direto."""
    syn = _sintetico()
    ui = _ui(series=syn)
    assert ui.series is syn.series
    assert ui.macro is syn.macro


def test_indices_desalinhados_erro_claro():
    syn = _sintetico(n=48)
    macro_deslocada = syn.macro.copy()
    macro_deslocada.index = macro_deslocada.index + pd.DateOffset(years=5)
    with pytest.raises(ValueError, match="desalinhad"):
        _ui(series=syn.series, macro=macro_deslocada)


def test_indices_de_tipos_diferentes_erro_claro():
    syn = _sintetico(n=36)
    macro_period = syn.macro.copy()
    macro_period.index = macro_period.index.to_period("M")
    with pytest.raises(ValueError, match="tipos diferentes"):
        _ui(series=syn.series, macro=macro_period)


def test_sem_dados_e_botao_de_estudo_de_referencia():
    ui = _ui()
    assert ui.series is None and ui.macro is None
    assert ui.btn_ref_study is not None
    ui.sl_ref_n.value = 48                     # estudo curto: teste rápido
    ui.btn_ref_study.click()
    assert ui.series is not None, "o estudo de referência não foi carregado"
    assert ui.macro is not None and len(ui.macro.columns) >= 3
    assert len(ui.series) == 48
    assert ui.candidates(), "as macros do estudo deveriam virar candidatas"
    assert "<img" in ui.out_serie_nivel.value


def test_estudo_de_referencia_por_parametro():
    """O parâmetro do estudo de referência é escolhido na tela (PD/LGD/CCF)."""
    ui = _ui()
    ui.dd_ref_kind.value = "lgd"
    ui.sl_ref_n.value = 48
    ui.btn_ref_study.click()
    assert ui.kind == "lgd"
    assert ui.series.kind == "lgd"


# ======================================================================
# Esqueleto
# ======================================================================
def test_abas_com_nomes_definitivos():
    ui = _ui()
    esperadas = ["Série", "Especificação", "Seleção", "Diagnóstico",
                 "Cenários & Projeção", "Backtest", "Exportar"]
    titulos = [ui.tabs.get_title(i) for i in range(len(ui.tabs.children))]
    assert titulos == esperadas
    # as abas ainda não preenchidas existem como VBox (os próximos blocos só
    # trocam ``children``)
    for box in (ui.box_selecao, ui.box_diagnostico, ui.box_cenarios,
                ui.box_backtest, ui.box_exportar):
        assert box is not None and len(box.children) >= 1


def test_import_do_modulo_nao_exige_ipywidgets():
    """O import do subpacote/módulo não pode puxar a dependência opcional."""
    import importlib

    mod = importlib.import_module("yggdrasil.credit_risk.econometric.ui")
    assert hasattr(mod, "SatelliteUI")
    eco = importlib.import_module("yggdrasil.credit_risk.econometric")
    assert eco.SatelliteUI is mod.SatelliteUI     # export lazy no __init__


# ======================================================================
# Aba Série — estacionariedade
# ======================================================================
def test_relatorio_estacionariedade_renderiza():
    syn = _sintetico(n=72)
    ui = _ui(series=syn.series, macro=syn.macro)
    ui.btn_estac.click()
    tab = ui.stationarity_
    assert isinstance(tab, pd.DataFrame) and not tab.empty
    for col in ("série", "ADF", "KPSS", "Phillips-Perron", "I(d)"):
        assert col in tab.columns
    # alvo (link e nível) + todas as macros
    assert len(tab) == 2 + len(syn.macro.columns)
    html = ui.out_estac.value
    assert "ADF" in html and "I(" in html
    assert ui.out_estac_resumo.value.strip(), "faltou a leitura em texto dos testes"
    assert "I(" in ui.out_estac_resumo.value


def test_estacionariedade_sem_dados_nao_estoura():
    ui = _ui()
    ui.btn_estac.click()
    assert ui.stationarity_ is None
    assert "Carregue" in ui.out_estac_status.value


# ======================================================================
# Aba Especificação — ajuste único
# ======================================================================
def _ui_pronta_para_ajuste(n=72):
    syn = _sintetico(n=n)
    ui = _ui(series=syn.series, macro=syn.macro)
    for var, cb in ui._sign_cbs.items():
        cb.value = var in ("desemprego", "renda")
    ui._sign_tgs["desemprego"].value = 1
    ui._sign_tgs["renda"].value = -1
    ui._sign_lags["desemprego"].value = 1
    return ui


def test_ajustar_agora_produz_fit_e_tabela_de_coeficientes():
    ui = _ui_pronta_para_ajuste()
    ui.btn_fit_now.click()
    assert ui.fit_ is not None, ui.out_fit_status.value
    assert ui.model_ is not None
    assert ui.fit_.model_name == "ARDL"
    coef = ui.out_fit_coef.value
    assert "termo" in coef and "p-valor" in coef
    assert "desemprego_l1" in coef        # a defasagem escolhida na matriz
    assert "coerência" in coef            # sinal estimado × sinal esperado
    assert "AIC" in ui.out_fit_metrics.value and "BIC" in ui.out_fit_metrics.value
    assert "<img" in ui.out_fit_plot.value


def test_ajuste_fica_desatualizado_ao_mexer_na_especificacao():
    ui = _ui_pronta_para_ajuste()
    ui.btn_fit_now.click()
    assert ui._dirty_since_fit is False
    ui.tx_ar_orders.value = "2"
    assert ui._dirty_since_fit is True
    assert "desatualizado" in ui.out_fit_warn.value


def test_ajustar_agora_erro_claro_sem_dados():
    ui = _ui()
    ui.btn_fit_now.click()
    assert ui.fit_ is None
    assert "sem série carregada" in ui.out_fit_status.value


def test_ajuste_com_benchmark_ingenuo():
    """Modelo sem coeficientes não pode quebrar a renderização."""
    ui = _ui_pronta_para_ajuste(n=60)
    ui.dd_model.value = "media_historica"
    ui.btn_fit_now.click()
    assert ui.fit_ is not None, ui.out_fit_status.value
    assert "sem coeficientes" in ui.out_fit_coef.value


def test_campos_especificos_aparecem_por_modelo():
    ui = _ui_pronta_para_ajuste(n=48)
    ui.dd_model.value = "vasicek"
    assert ui.fl_rho.layout.display is None          # visível
    assert ui.tx_arima_order.layout.display == "none"
    ui.dd_model.value = "arima"
    assert ui.fl_rho.layout.display == "none"
    assert ui.tx_arima_order.layout.display is None


def test_vasicek_ajusta_e_usa_rho():
    ui = _ui_pronta_para_ajuste(n=72)
    ui.dd_model.value = "vasicek"
    ui.fl_rho.value = 0.12
    ui.btn_fit_now.click()
    assert ui.fit_ is not None, ui.out_fit_status.value
    assert ui.fit_.link == "vasicek"
    assert ui.model_.rho == pytest.approx(0.12)


# ======================================================================
# Configuração declarativa
# ======================================================================
def test_to_config_reflete_a_tela():
    ui = _ui_pronta_para_ajuste(n=48)
    ui.tx_lag_set.value = "0,2"
    ui.sl_max_vars.value = 2
    ui.dd_criterion.value = "bic"
    cfg = ui.to_config()
    assert cfg.kind == "pd" and cfg.model == "ardl"
    assert set(cfg.candidates) == {"desemprego", "renda"}
    assert cfg.expected_signs == {"desemprego": 1, "renda": -1}
    assert tuple(cfg.lag_set) == (0, 2)
    assert cfg.max_vars == 2 and cfg.criterion == "bic"
    assert cfg.min_train is None                      # 0 na tela = automático


def test_config_roundtrip():
    ui = _ui_pronta_para_ajuste(n=48)
    ui.tx_lag_set.value = "0,1,4"
    ui.sl_horizon.value = 9
    ui.sl_min_train.value = 30
    ui.cb_seasonal.value = True
    ui.tx_nome.value = "estudo_teste"
    cfg = ui.to_config()

    # mexe em tudo e restaura
    ui.tx_lag_set.value = "0"
    ui.sl_horizon.value = 3
    ui.cb_seasonal.value = False
    ui.tx_nome.value = "outro"
    ui._set_all_signs(0)
    ui._set_all_candidates(False)
    ui.from_config(cfg)

    assert ui.to_config().to_dict() == cfg.to_dict()
    assert ui.tx_nome.value == "estudo_teste"
    assert set(ui.candidates()) == {"desemprego", "renda"}
    assert ui.expected_signs() == {"desemprego": 1, "renda": -1}


def test_config_roundtrip_via_dict_e_vasicek():
    ui = _ui_pronta_para_ajuste(n=48)
    ui.dd_model.value = "vasicek"
    ui.cb_ttc_auto.value = False
    ui.fl_pd_ttc.value = 0.05
    ui.fl_rho.value = 0.18
    cfg = ui.to_config()
    assert cfg.model == "vasicek" and cfg.rho == pytest.approx(0.18)
    assert cfg.pd_ttc == pytest.approx(0.05)

    ui.dd_model.value = "ardl"
    ui.cb_ttc_auto.value = True
    ui.from_config(cfg.to_dict())                 # aceita dict, não só StudyConfig
    assert ui.dd_model.value == "vasicek"
    assert ui.fl_pd_ttc.value == pytest.approx(0.05)
    assert ui.to_config().to_dict() == cfg.to_dict()


def test_config_de_benchmark_exporta_modelo_candidato():
    """ARIMA/ingênuos entram como benchmark: a config sai com um modelo válido."""
    from yggdrasil.credit_risk.econometric.config import MODEL_REGISTRY

    ui = _ui_pronta_para_ajuste(n=48)
    ui.dd_model.value = "arima"
    cfg = ui.to_config()
    assert cfg.model in MODEL_REGISTRY


# ======================================================================
# Aba Seleção — busca champion-challenger
# ======================================================================
def _ui_para_busca(n=54, sinal_desemprego=1):
    """Interface com uma grade **minúscula**: 2 candidatas, 2 defasagens, 1 variável."""
    ui = _ui_pronta_para_ajuste(n=n)
    ui._sign_tgs["desemprego"].value = sinal_desemprego
    ui.tx_lag_set.value = "0,1"
    ui.sl_max_vars.value = 1                 # 2 candidatas × 2 defasagens = 4 specs
    ui.sl_horizon.value = 2
    ui.sl_min_train.value = 40
    return ui


def test_tamanho_da_grade_avisa_antes_de_rodar():
    ui = _ui_para_busca()
    info = ui._render_grid_info()
    assert info == {"candidatas": 2, "total": 4, "efetivo": 4, "janelas": 13,
                    "min_train": 40, "ajustes": 4 * 14}
    html = ui.out_grid_info.value
    assert "a avaliar" in html and "janelas walk-forward" in html

    # grade maior que o teto de especificações: o corte precisa ser explícito
    ui.tx_lag_set.value = "0,1,3,6"
    ui.sl_max_vars.value = 2
    ui.sl_max_specs.value = 5
    info2 = ui._grid_size()
    assert info2["total"] == 2 * 4 + 1 * 16      # C(2,1)·4 + C(2,2)·4²
    assert info2["efetivo"] == 5
    ui._render_grid_info()
    assert "não serão avaliadas" in ui.out_grid_info.value


def test_grade_sem_janela_de_validacao_nao_roda():
    ui = _ui_para_busca(n=54)
    ui.sl_min_train.value = 60                   # maior que a própria série
    ui.btn_search.click()
    assert ui.search_ is None
    assert "janela de validação" in ui.out_search_status.value


def test_busca_produz_ranking_e_adota_a_campea():
    ui = _ui_para_busca()
    ui.btn_search.click()
    res = ui.search_
    assert res is not None, ui.out_search_status.value
    assert len(res.ranking) == 4 + len(res.benchmarks)      # grade + benchmarks
    # a campeã entra como modelo vigente (o seletor permite discordar)
    assert res.best_spec is not None
    assert ui.selected_spec_ is res.best_spec
    assert ui.fit_ is not None and ui.model_ is res.best
    assert ui._dirty_since_fit is False

    rank = ui.out_search_rank.value
    for col in ("especificação", "variáveis", "defasagens", "RMSE fora", "VIF máx.",
                "sinais", "situação", "AIC"):
        assert col in rank, col
    assert "★" in rank                                       # a campeã marcada
    assert "random_walk" in rank                             # benchmarks na mesma régua
    assert "qualificadas" in ui.out_search_resumo.value
    # progresso por etapa com o tempo decorrido (a busca não dá sinal de vida sozinha)
    prog = ui.out_search_progress.value
    assert "Progresso da busca" in prog and "concluída" in prog
    assert ui._search_secs is not None and ui._search_secs > 0
    # o seletor traz a campeã pré-escolhida
    assert ui.dd_pick_spec.value == res.best_spec.describe()
    assert any("campeã" in rot for rot, _ in ui.dd_pick_spec.options)


def test_busca_explicita_o_motivo_do_descarte():
    """Sinal declarado ao contrário: as especificações caem no filtro duro e a
    interface diz exatamente por quê."""
    ui = _ui_para_busca(sinal_desemprego=-1)
    ui.btn_search.click()
    assert ui.search_ is not None, ui.out_search_status.value
    desq = ui.out_search_desq.value
    assert "sinal econômico invertido em desemprego" in desq
    assert "motivo" in desq
    assert "sinal invertido" in ui.out_search_resumo.value
    # as descartadas continuam disponíveis para escolha manual, marcadas como tal
    assert any(rot.startswith("⚠ descartada") for rot, _ in ui.dd_pick_spec.options)


def test_escolha_manual_sobrepoe_a_campea():
    """Champion-challenger de verdade: dá para adotar até uma descartada."""
    ui = _ui_para_busca(sinal_desemprego=-1)
    ui.btn_search.click()
    campea = ui.selected_spec_
    alvo = next(v for rot, v in ui.dd_pick_spec.options if rot.startswith("⚠ descartada"))
    ui.dd_pick_spec.value = alvo
    ui.btn_pick_fit.click()
    assert ui.selected_spec_ is not campea
    assert ui.selected_spec_.describe() == alvo
    assert ui.fit_ is not None and ui.fit_.spec.describe() == alvo
    assert "adotada" in ui.out_pick_status.value
    assert "p-valor" in ui.out_pick_info.value       # tabela de coeficientes da adotada
    assert ui._dirty_since_fit is False


def test_comparacao_com_benchmarks_e_diebold_mariano():
    ui = _ui_para_busca()
    ui.btn_search.click()
    ui.btn_dm.click()
    tab = ui.compare_
    assert isinstance(tab, pd.DataFrame) and not tab.empty
    for col in ("referência", "RMSE da referência", "RMSE da adotada", "DM (estat.)",
                "p-valor", "veredito"):
        assert col in tab.columns, col
    assert {"random_walk", "media_historica"} <= set(tab["referência"])
    assert tab["DM (estat.)"].notna().any(), "o teste deveria produzir estatística"
    assert "veredito" in ui.out_dm.value
    assert ui.out_dm_status.value.strip()


def test_busca_sem_dados_e_sem_candidatas_avisa():
    ui = _ui()
    ui.btn_search.click()
    assert ui.search_ is None
    assert "Sem série carregada" in ui.out_search_status.value

    ui2 = _ui_para_busca()
    ui2._set_all_candidates(False)
    ui2.btn_search.click()
    assert ui2.search_ is None
    assert "candidata" in ui2.out_search_status.value


# ======================================================================
# Aba Diagnóstico
# ======================================================================
def test_diagnostico_placar_tabela_e_graficos():
    ui = _ui_pronta_para_ajuste(n=72)
    ui.btn_fit_now.click()
    assert ui.dd_chow_break.options, "as datas candidatas do Chow saem do design do ajuste"
    ui.btn_diag.click()
    tab = ui.diagnostics_
    assert isinstance(tab, pd.DataFrame) and not tab.empty
    testes = set(tab["teste"])
    # a bateria completa: resíduo, heterocedasticidade, normalidade, estabilidade, VIF
    for t in ("Ljung-Box", "Breusch-Godfrey", "Durbin-Watson", "Breusch-Pagan", "White",
              "ARCH-LM", "Jarque-Bera", "CUSUM", "Chow", "Quandt-Andrews sup-F", "VIF"):
        assert t in testes, t
    blocos = ui.diag_blocks_
    assert [b["bloco"] for b in blocos] == ["Resíduo", "Heterocedasticidade", "Normalidade",
                                            "Estabilidade", "Colinearidade"]
    assert all(b["nivel"] in ("ok", "warn", "bad", "na") for b in blocos)
    placar = ui.out_diag_placar.value
    assert "Resíduo" in placar and "p = " in placar        # veredito + evidência
    assert "Ljung-Box" in ui.out_diag_tabela.value
    assert "H0" in ui.out_diag_tabela.value                # a nula de cada teste
    assert "VIF" in ui.out_diag_vif.value
    assert "<img" in ui.out_diag_plot_fit.value
    assert "<img" in ui.out_diag_plot_resid.value
    assert ui.vif_ is not None and len(ui.vif_) == 2


def test_diagnostico_reprovado_traz_o_que_fazer():
    """Média histórica deixa toda a dinâmica no resíduo: os blocos reprovam e a
    leitura em texto diz o que fazer."""
    ui = _ui_pronta_para_ajuste(n=72)
    ui.dd_model.value = "media_historica"
    ui.btn_fit_now.click()
    ui.btn_diag.click()
    ruins = [b for b in ui.diag_blocks_ if b["nivel"] == "bad"]
    assert ruins, "o placar deveria reprovar ao menos um bloco"
    leitura = ui.out_diag_leitura.value
    assert "Autocorrelação residual" in leitura
    assert "ordem AR" in leitura                    # conselho acionável, não tratado
    assert "reprovado" in ui.out_diag_status.value.lower()
    # modelo sem regressores: nada de VIF/Chow, e isso é dito em vez de omitido
    assert "não há colinearidade a medir" in ui.out_diag_vif.value
    assert any(b["bloco"] == "Colinearidade" and b["nivel"] == "na"
               for b in ui.diag_blocks_)


def test_diagnostico_invalidado_quando_o_modelo_muda():
    ui = _ui_pronta_para_ajuste(n=72)
    ui.btn_fit_now.click()
    ui.btn_diag.click()
    assert ui.diagnostics_ is not None
    ui.tx_ar_orders.value = "2"                     # mexe na especificação
    assert ui.diagnostics_ is None and ui.diag_blocks_ is None
    assert "desatualizado" in ui.out_diag_notice.value
    assert ui.out_diag_placar.value == ""
    assert ui.out_diag_plot_fit.value == ""


def test_diagnostico_sem_ajuste_avisa():
    ui = _ui()
    ui.btn_diag.click()
    assert ui.diagnostics_ is None
    assert "Nenhum modelo ajustado" in ui.out_diag_status.value


def test_troca_de_dados_zera_selecao_e_diagnostico():
    ui = _ui_para_busca()
    ui.btn_search.click()
    ui.btn_diag.click()
    outra = _sintetico(n=48, seed=11)
    ui.set_data(outra.series, outra.macro)
    assert ui.search_ is None and ui.compare_ is None and ui.selected_spec_ is None
    assert ui.diagnostics_ is None and ui.diag_blocks_ is None
    assert ui.out_search_rank.value == "" and ui.out_diag_placar.value == ""
    assert not ui.dd_pick_spec.options


# ======================================================================
# Aba Cenários & Projeção
# ======================================================================
def _ui_com_ajuste(n=72, horizonte=6):
    """Interface já ajustada e com o horizonte de projeção curto (teste rápido)."""
    ui = _ui_pronta_para_ajuste(n=n)
    ui.btn_fit_now.click()
    assert ui.fit_ is not None, ui.out_fit_status.value
    ui.sl_scen_horizon.value = horizonte
    ui.sl_scen_sims.value = 120                # banda simulada, mas barata
    ui.dd_stress_var.value = "desemprego"
    return ui


def test_cenarios_padrao_um_clique():
    ui = _ui_com_ajuste()
    ui.btn_scen_padrao.click()
    ss = ui.scenarios_
    assert ss is not None, ui.out_scen_padrao.value
    assert set(ss.names()) == {"base", "adverso", "otimista"}
    assert all(s.horizon == 6 for s in ss.scenarios)
    pesos = ss.probabilities()                 # levanta se não somarem 1
    assert pesos["base"] == pytest.approx(0.5)
    # o adverso precisa mesmo estressar a variável escolhida
    adv = ss.get("adverso").macro["desemprego"].to_numpy()
    base = ss.get("base").macro["desemprego"].to_numpy()
    assert (adv > base).all()
    assert "cenário" in ui.out_scen_tabela.value and "adverso" in ui.out_scen_tabela.value
    assert "<img" in ui.out_scen_plot.value    # trajetórias com o futuro sombreado


def test_choque_com_persistencia_decai():
    """Persistência < 1 torna o choque temporário (a variável volta à base)."""
    ui = _ui_com_ajuste()
    ui.fl_shock_mag.value = 2.0
    ui.fl_shock_persist.value = 0.5
    ui.btn_scen_choque.click()
    ss = ui.scenarios_
    assert ss is not None, ui.out_scen_choque.value
    desvio = (ss.get("adverso").macro["desemprego"].to_numpy()
              - ss.get("base").macro["desemprego"].to_numpy())
    assert desvio[0] > 0
    assert desvio[1] == pytest.approx(desvio[0] * 0.5, rel=1e-6)
    assert desvio[-1] < desvio[0] / 10

    ui.fl_shock_persist.value = 1.0            # permanente: choque constante
    ui.btn_scen_choque.click()
    desvio2 = (ui.scenarios_.get("adverso").macro["desemprego"].to_numpy()
               - ui.scenarios_.get("base").macro["desemprego"].to_numpy())
    assert desvio2[-1] == pytest.approx(desvio2[0])


def test_cenario_colado_completa_as_variaveis_que_faltam():
    """O caminho que mais falta: colar a trajetória que veio da área econômica."""
    ui = _ui_com_ajuste()
    ui.btn_scen_padrao.click()
    ui.ta_scen_paste.value = "desemprego\n" + "\n".join(f"{9 + 0.5 * i:.2f}" for i in range(6))
    ui.tx_scen_nome.value = "economia"
    ui.fl_scen_peso.value = 0.25
    ui.btn_scen_add.click()
    ss = ui.scenarios_
    assert "economia" in ss.names(), ui.out_scen_manual.value
    cen = ss.get("economia")
    assert len(cen.macro) == 6
    assert cen.macro["desemprego"].iloc[0] == pytest.approx(9.0)
    # as variáveis ausentes na colagem vêm da trajetória base, e a tela diz quais
    assert "completadas pela base" in ui.out_scen_manual.value
    assert "renda" in ui.out_scen_manual.value
    assert set(cen.macro.columns) == set(ui._macro_cols())
    assert sum(s.probability for s in ss.scenarios) == pytest.approx(1.0)


def test_cenario_colado_com_numero_de_linhas_errado_recusa():
    ui = _ui_com_ajuste()
    ui.ta_scen_paste.value = "desemprego\n9.0\n9.1"       # 2 linhas, horizonte 6
    ui.btn_scen_add.click()
    assert ui.scenarios_ is None
    assert "horizonte da tela é 6" in ui.out_scen_manual.value


def test_gabarito_de_colagem_traz_a_trajetoria_base():
    ui = _ui_com_ajuste()
    ui.btn_scen_modelo.click()
    linhas = ui.ta_scen_paste.value.strip().splitlines()
    assert len(linhas) == 7                              # cabeçalho + horizonte
    assert "desemprego" in linhas[0]
    # o gabarito colado de volta reproduz o cenário base
    ui.tx_scen_nome.value = "base_editada"
    ui.btn_scen_add.click()
    assert "base_editada" in ui.scenarios_.names(), ui.out_scen_manual.value


def test_projecao_em_leque_e_ponderada():
    ui = _ui_com_ajuste()
    ui.btn_scen_padrao.click()
    ui.btn_project.click()
    proj = ui.projection_
    assert proj is not None, ui.out_proj_status.value
    assert set(proj.paths) == {"base", "adverso", "otimista"}
    assert proj.horizon == 6
    # bandas de verdade (simulação ligada)
    assert proj.paths["base"]["lower"].notna().all()
    # o cenário adverso precisa doer: a partir do 2º período (a macro entra com
    # defasagem 1, então o 1º passo ainda usa a macro observada)
    assert (proj.paths["adverso"]["mean"].iloc[1:]
            > proj.paths["otimista"]["mean"].iloc[1:]).all()
    assert "<img" in ui.out_proj_plot.value
    for nome in ("base", "adverso", "otimista", "ponderada"):
        assert nome in ui.out_proj_tabela.value, nome
    # a curva única: entre o otimista e o adverso, período a período
    w = ui.weighted_
    assert w is not None and len(w) == 6
    assert (w.to_numpy() <= proj.paths["adverso"]["mean"].to_numpy() + 1e-12).all()
    assert (w.to_numpy() >= proj.paths["otimista"]["mean"].to_numpy() - 1e-12).all()
    assert "ponderada" in ui.out_pond_tabela.value
    assert "último observado" in ui.out_pond_tiles.value


def test_exportacao_da_projecao():
    ui = _ui_com_ajuste()
    ui.btn_scen_padrao.click()
    ui.btn_project.click()
    df = ui.projection_frame()
    for col in ("parametro", "segmento", "cenario", "periodo", "mean", "lower", "upper"):
        assert col in df.columns, col
    assert "ponderado" in set(df["cenario"])            # a curva única vai junto
    assert len(df) == 6 * 4                             # 3 cenários + ponderado
    ui.dd_export_fmt.value = "csv_br"
    ui.btn_scen_export.click()
    texto = ui.ta_scen_csv.value
    assert texto.splitlines()[0].count(";") >= 6        # ponto e vírgula p/ Excel pt-BR
    assert "ponderado" in texto
    assert ui.projection_table_ is not None


def test_projecao_exige_modelo_e_cenario():
    ui = _ui()
    ui.btn_project.click()
    assert ui.projection_ is None
    assert "modelo vigente" in ui.out_proj_status.value

    ui2 = _ui_com_ajuste(n=60)
    ui2.btn_project.click()                             # sem cenário montado
    assert ui2.projection_ is None
    assert "cenário" in ui2.out_proj_status.value


def test_projecao_invalidada_mas_cenarios_preservados():
    """Mexer no modelo derruba a projeção — a trajetória montada é do usuário."""
    ui = _ui_com_ajuste()
    ui.btn_scen_padrao.click()
    ui.btn_project.click()
    assert ui.projection_ is not None
    ui.tx_ar_orders.value = "2"                         # especificação mudou
    assert ui.projection_ is None and ui.weighted_ is None
    assert ui.out_proj_tabela.value == ""
    assert "desatualizada" in ui.out_proj_status.value
    assert ui.scenarios_ is not None and len(ui.scenarios_) == 3


# ======================================================================
# Aba Backtest
# ======================================================================
def _ui_para_backtest(n=60):
    ui = _ui_pronta_para_ajuste(n=n)
    ui.btn_fit_now.click()
    ui.sl_bt_min_train.value = 44
    ui.sl_bt_horizon.value = 2
    ui.sl_bt_sims.value = 60                            # bandas baratas
    return ui


def test_dimensionamento_das_janelas_do_backtest():
    ui = _ui_para_backtest()
    p = ui._render_bt_info()
    assert p["janelas"] == 60 - 2 + 1 - 44
    assert p["pontos"] == p["janelas"] * 2
    assert "violações esperadas" in ui.out_bt_info.value
    ui.sl_bt_min_train.value = 70                       # maior que a série
    assert ui._bt_params()["janelas"] == 0
    ui.btn_backtest.click()
    assert ui.backtest_ is None
    assert "Sem janela de validação" in ui.out_bt_status.value


def test_backtest_cobertura_kupiec_e_christoffersen():
    ui = _ui_para_backtest()
    ui.btn_backtest.click()
    wf = ui.backtest_
    assert wf is not None, ui.out_bt_status.value
    assert wf["n_windows"] > 0
    bandas = wf["bands"]
    assert set(("passo", "previsto", "lower", "upper", "real", "violacao")) <= set(bandas.columns)

    cov = ui.coverage_
    assert cov is not None and "todos" in [str(p) for p in cov["passo"]]
    for col in ("cobertura", "kupiec_pvalue", "christoffersen_pvalue"):
        assert col in cov.columns, col

    # veredito visível: placar + leitura em texto com os números
    placar = ui.out_bt_placar.value
    assert "Cobertura das bandas" in placar and "Kupiec" in placar
    assert "Christoffersen" in placar
    leitura = ui.out_bt_leitura.value
    assert "cobriram" in leitura and "Kupiec" in leitura
    assert "RMSE" in leitura
    # erro por horizonte e tabela de cobertura
    erros = ui.out_bt_erros.value
    assert "RMSE" in erros and "MAPE" in erros and "viés" in erros
    tabela = ui.out_bt_cobertura.value
    assert "cobertura empírica" in tabela and "p Kupiec" in tabela
    # progresso por etapa e gráfico das violações
    assert "Progresso do backtest" in ui.out_bt_progress.value
    assert "concluída" in ui.out_bt_progress.value
    assert ui.dd_bt_passo.options and "<img" in ui.out_bt_plot.value
    ui.dd_bt_passo.value = 2
    assert "<img" in ui.out_bt_plot.value


def test_backtest_erros_por_passo_crescem_com_o_horizonte():
    ui = _ui_para_backtest()
    ui.btn_backtest.click()
    tab = ui._bt_erros_por_passo(ui.backtest_["bands"])
    assert list(tab["passo"]) == ["1", "2", "todos"]
    assert (tab["n"] > 0).all()
    assert tab.loc[tab["passo"] == "2", "RMSE"].iloc[0] > 0


def test_backtest_sem_modelo_e_invalidacao():
    ui = _ui()
    ui.btn_backtest.click()
    assert ui.backtest_ is None
    assert "modelo vigente" in ui.out_bt_status.value

    ui2 = _ui_para_backtest()
    ui2.btn_backtest.click()
    assert ui2.backtest_ is not None
    ui2.tx_ar_orders.value = "2"
    assert ui2.backtest_ is None and ui2.coverage_ is None
    assert "desatualizado" in ui2.out_bt_notice.value
    assert ui2.out_bt_cobertura.value == "" and ui2.out_bt_plot.value == ""


def test_troca_de_dados_zera_cenarios_e_backtest():
    ui = _ui_para_backtest()
    ui.sl_scen_horizon.value = 4
    ui.btn_scen_padrao.click()
    ui.btn_project.click()
    ui.btn_backtest.click()
    outra = _sintetico(n=48, seed=11)
    ui.set_data(outra.series, outra.macro)
    assert ui.scenarios_ is None and ui.projection_ is None and ui.weighted_ is None
    assert ui.backtest_ is None and ui.coverage_ is None
    assert ui.out_scen_tabela.value == "" and ui.out_bt_cobertura.value == ""
    assert ui.ta_scen_csv.value == ""


# ======================================================================
# Tema
# ======================================================================
def _htmls(widget, acc=None):
    import ipywidgets as W

    acc = [] if acc is None else acc
    if isinstance(widget, W.HTML):
        acc.append(widget.value)
    for filho in getattr(widget, "children", ()) or ():
        _htmls(filho, acc)
    return acc


_HEX = re.compile(r"#[0-9a-fA-F]{3}\b|#[0-9a-fA-F]{6}\b")


def test_html_gerado_usa_tokens_de_tema_e_nao_hex():
    """O HTML montado no Python só pode usar var(--...): o tema escuro redefine os
    tokens, e um hex fixo ficaria ilegível no escuro."""
    ui = _ui_pronta_para_ajuste(n=60)
    ui.btn_estac.click()
    ui.btn_fit_now.click()
    ui.btn_macro_plot.click()
    ui.cb_dark.value = True                     # alterna o tema (só troca a classe)
    assert "dark" in ui.panel._dom_classes

    for html in _htmls(ui.panel):
        if "<style>" in html:                   # a folha de estilo define os tokens
            continue
        achados = _HEX.findall(html)
        assert not achados, f"hex fixo no HTML gerado: {achados[:3]} em {html[:120]!r}"


def test_html_da_selecao_e_do_diagnostico_usa_tokens_de_tema():
    """Mesma regra nas abas de seleção e diagnóstico (placar, progresso, ranking)."""
    ui = _ui_para_busca()
    ui.btn_search.click()
    ui.btn_dm.click()
    ui.btn_diag.click()
    ui.cb_dark.value = True
    for html in _htmls(ui.panel):
        if "<style>" in html:
            continue
        achados = _HEX.findall(html)
        assert not achados, f"hex fixo no HTML gerado: {achados[:3]} em {html[:120]!r}"


def test_html_de_cenarios_e_backtest_usa_tokens_de_tema():
    """Idem nas abas de projeção e backtest (leque, placar de cobertura, tabelas)."""
    ui = _ui_para_backtest()
    ui.sl_scen_horizon.value = 4
    ui.sl_scen_sims.value = 60
    ui.btn_scen_padrao.click()
    ui.btn_project.click()
    ui.btn_scen_export.click()
    ui.btn_backtest.click()
    ui.cb_dark.value = True
    for html in _htmls(ui.panel):
        if "<style>" in html:
            continue
        achados = _HEX.findall(html)
        assert not achados, f"hex fixo no HTML gerado: {achados[:3]} em {html[:120]!r}"
