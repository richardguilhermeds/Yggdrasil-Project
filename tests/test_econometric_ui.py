"""
Testes da interface de modelos satélite (:class:`SatelliteUI`).

Cobrem o esqueleto e as duas primeiras abas: construção a partir de
``RiskSeries`` / ``pandas.Series`` + macro / sem dados, validação do alinhamento
temporal, nomes definitivos das abas, relatório de estacionariedade, ajuste único
com tabela de coeficientes e o ida-e-volta com :class:`StudyConfig`.

Tudo roda em séries sintéticas **curtas** e grades pequenas — nenhuma busca.
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
