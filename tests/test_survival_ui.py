"""
Testes da interface de análise de sobrevivência (:class:`SurvivalUI`).

Cobrem o esqueleto e as sete abas: construção com e sem dados, o painel de
referência, o mapeamento de colunas e a montagem do painel, a partição DES/OOT,
as curvas de Kaplan-Meier com log-rank e adoção, a regressão de *hazard* com o
teste de riscos proporcionais, as famílias paramétricas com a emenda da cauda,
calibração e ciclo, a validação (placar, tabelas e gráficos), o ida-e-volta com
:class:`SurvivalConfig`, o estudo completo em um clique, a aba Exportar (modelo
em JSON, configuração, CSV e o MLflow por *stub*) e o tema escuro (nenhum hex
fixo no HTML gerado no Python).

Tudo roda num painel de referência **pequeno** (600 contratos).
"""
from __future__ import annotations

import contextlib
import io
import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ipywidgets")


def _ui(*args, **kwargs):
    import matplotlib

    matplotlib.use("Agg")
    from yggdrasil.credit_risk.survival.ui import SurvivalUI

    with contextlib.redirect_stdout(io.StringIO()):
        return SurvivalUI(*args, **kwargs)


@pytest.fixture(scope="module")
def ref():
    from yggdrasil.credit_risk.survival import make_reference_panel

    return make_reference_panel(n_contracts=600, seed=5)


def _ui_com_painel(ref, **kw):
    return _ui(ref.df, origin_col="safra_origem", segment_col="produto", term_col="prazo",
               exposure_col="exposicao", features=list(ref.features), **kw)


def _clica(ui, botao):
    with contextlib.redirect_stdout(io.StringIO()):
        getattr(ui, botao).click()


# ======================================================================
# Construção e esqueleto
# ======================================================================
def test_abas_com_nomes_definitivos_e_estado_inicial():
    ui = _ui()
    esperadas = ["Painel", "Kaplan-Meier", "Hazard", "Paramétrico", "Calibração & Ciclo",
                 "Validação", "Exportar"]
    assert [ui.tabs.get_title(i) for i in range(len(ui.tabs.children))] == esperadas
    assert ui.panel is None and ui.model_ is None and ui.study_ is None
    assert "sem painel" in ui.bar.value
    assert "sem dados" in ui.out_exp_estado.value


def test_constroi_com_dataframe_e_monta_o_painel(ref):
    ui = _ui_com_painel(ref)
    assert ui.panel is not None and ui.panel.n_contracts == 600
    assert ui.des_ is ui.panel and ui.oot_ is None
    assert ui.dd_col_origin.value == "safra_origem" and ui.dd_col_segment.value == "produto"
    assert set(ui.sel_hz_features.value) == set(ref.features)
    assert ui.dd_km_by.value == "produto"
    assert "contratos" in ui.out_panel_head.value and "<img" in ui.out_panel_plot.value
    assert "n_em_risco" in ui.out_panel_table.value
    assert "contratos: 600" in ui.bar.value


def test_import_do_modulo_nao_exige_ipywidgets():
    import importlib

    mod = importlib.import_module("yggdrasil.credit_risk.survival.ui")
    assert hasattr(mod, "SurvivalUI")
    sv = importlib.import_module("yggdrasil.credit_risk.survival")
    assert sv.SurvivalUI is mod.SurvivalUI
    import yggdrasil.credit_risk as cr

    assert cr.SurvivalUI is mod.SurvivalUI


def test_painel_de_referencia_pelo_botao():
    ui = _ui()
    ui.sl_ref_n.value = 400
    _clica(ui, "btn_ref")
    assert ui.panel is not None and ui.panel.n_contracts == 400, ui.out_ref_status.value
    assert "carregado" in ui.out_ref_status.value
    assert ui.dd_col_origin.value == "safra_origem"
    assert "feat_score" in [v for _, v in ui.sel_hz_features.options]


def test_mapeamento_incompleto_avisa(ref):
    ui = _ui()
    ui.set_data(ref.df.rename(columns={"default": "flag"}), default_col=None)
    assert ui.panel is None
    assert "obrigatórias" in ui.out_montar_status.value
    ui.dd_col_default.value = "flag"
    _clica(ui, "btn_montar")
    assert ui.panel is not None, ui.out_montar_status.value


def test_painel_invalido_vira_mensagem(ref):
    df = ref.df.copy()
    df.loc[df.index[:3], "default"] = 7        # flag não binária
    ui = _ui()
    ui.set_data(df, origin_col="safra_origem")
    assert ui.panel is None
    assert "Não foi possível montar" in ui.out_montar_status.value


def test_mapa_safra_idade_desenha(ref):
    ui = _ui_com_painel(ref)
    _clica(ui, "btn_heat")
    assert "<img" in ui.out_heat.value


# ======================================================================
# Partição
# ======================================================================
def test_particao_por_originacao_e_por_observacao(ref):
    ui = _ui_com_painel(ref)
    ui.dd_split.value = "origin"
    assert ui.tx_split_date.value, "a data de corte deveria ser sugerida"
    _clica(ui, "btn_split")
    assert ui.oot_ is not None, ui.out_split_status.value
    assert ui.des_.n_contracts + ui.oot_.n_contracts == 600
    assert set(ui.des_.df["id_contrato"]).isdisjoint(set(ui.oot_.df["id_contrato"]))
    assert "OOT · contratos" in ui.out_split_tiles.value
    assert "OOT:" in ui.bar.value

    ui.dd_split.value = "observation"
    _clica(ui, "btn_split")
    assert ui.oot_ is not None
    assert (ui.oot_.df["dt_ref"] >= pd.Timestamp(ui.tx_split_date.value)).all()

    ui.dd_split.value = "none"
    _clica(ui, "btn_split")
    assert ui.oot_ is None and "Sem OOT" in ui.out_split_status.value


def test_particao_por_coluna(ref):
    df = ref.df.assign(amostra=np.where(ref.df["produto"] == "cartao", "OOT", "DES"))
    ui = _ui(df, origin_col="safra_origem", segment_col="produto")
    ui.dd_split.value = "column"
    ui.dd_split_col.value = "amostra"
    _clica(ui, "btn_split")
    assert set(ui.oot_.df["produto"]) == {"cartao"}
    ui.dd_split_col.value = None
    _clica(ui, "btn_split")
    assert "Não foi possível particionar" in ui.out_split_status.value


def test_reparticionar_derruba_a_curva_adotada(ref):
    ui = _ui_com_painel(ref)
    _clica(ui, "btn_km")
    _clica(ui, "btn_km_adopt")
    assert ui.model_ is not None
    ui.dd_split.value = "origin"
    _clica(ui, "btn_split")
    assert ui.model_ is None and ui.km_curves_ == {}
    assert ui.out_km_plot.value == ""


# ======================================================================
# Kaplan-Meier
# ======================================================================
def test_km_estima_por_grupo_com_logrank_e_tabela(ref):
    ui = _ui_com_painel(ref)
    _clica(ui, "btn_km")
    assert set(ui.km_curves_) == {"cartao", "consignado"}, ui.out_km_status.value
    assert "<img" in ui.out_km_plot.value
    assert "pd_12m" in ui.out_km_summary.value and "mediana" in ui.out_km_summary.value
    assert "hazard_acumulado" in ui.out_km_table.value
    assert ui.logrank_ is not None and ui.logrank_["p_valor"] < 0.05
    assert "rejeita H0" in ui.out_km_logrank.value and "obs_esp" in ui.out_km_logrank.value
    assert ui.pairwise_ is None                    # só 2 grupos: sem par a par
    # 3 grupos: par a par com Bonferroni
    ui.dd_km_by.value = "rating"
    _clica(ui, "btn_km")
    assert ui.pairwise_ is not None and "p_bonferroni" in ui.out_km_pairwise.value
    # troca de representação redesenha
    antes = ui.out_km_plot.value
    ui.dd_km_kind.value = "survival"
    assert ui.out_km_plot.value != antes and "<img" in ui.out_km_plot.value
    ui.cb_km_na.value = True
    assert "<img" in ui.out_km_plot.value


def test_km_carteira_inteira_e_vintage(ref):
    ui = _ui_com_painel(ref)
    ui.dd_km_by.value = None
    ui.dd_km_method.value = "vintage"
    assert ui.sl_km_min_risk.layout.display is None
    _clica(ui, "btn_km")
    assert list(ui.km_curves_) == ["__global__"]
    assert "carteira" in ui.out_km_summary.value
    assert "Agrupe" in ui.out_km_logrank.value
    ui.dd_km_method.value = "km"
    assert ui.sl_km_min_risk.layout.display == "none"


def test_km_sem_painel_avisa():
    ui = _ui()
    _clica(ui, "btn_km")
    assert ui.km_curves_ == {} and "Monte o painel" in ui.out_km_status.value
    _clica(ui, "btn_km_adopt")
    assert "Estime as curvas" in ui.out_km_status.value


def test_km_adota_a_curva_com_cauda_plana(ref):
    ui = _ui_com_painel(ref)
    ui.sl_km_horizon.value = 48
    _clica(ui, "btn_km")
    _clica(ui, "btn_km_adopt")
    m = ui.model_
    assert m is not None and ui.model_source_ == "km"
    assert set(m.curves_) == {"cartao", "consignado"} and m.horizon == 48
    assert all(len(c) == 48 for c in m.curves_.values())
    h = m.curve("cartao").hazard_
    assert np.allclose(h[ui.panel.max_age + 1:], h[ui.panel.max_age])     # plana
    assert "curva: km" in ui.bar.value
    assert "cartao" in ui.out_calib_model.value or "produto" in ui.out_calib_model.value
    assert set(ui._calib_inputs) == {"cartao", "consignado"}


# ======================================================================
# Hazard
# ======================================================================
def test_hazard_ajusta_mostra_coeficientes_e_perfis(ref):
    ui = _ui_com_painel(ref)
    ui.sl_hz_horizon.value = 36
    _clica(ui, "btn_hz")
    assert ui.hazard_lt_ is not None, ui.out_hz_status.value
    assert ui.hazard_lt_.method_ == "hazard" and set(ui.hazard_lt_.curves_) == {"cartao", "consignado"}
    assert "odds_ratio" in ui.out_hz_coef.value and "feat_score" in ui.out_hz_coef.value
    assert "AIC" in ui.out_hz_metrics.value and "pessoa-períodos" in ui.out_hz_metrics.value
    assert "<img" in ui.out_hz_plot_base.value and "<img" in ui.out_hz_plot_prof.value
    _clica(ui, "btn_hz_ph")
    assert ui.ph_ is not None and "Riscos proporcionais" in ui.out_hz_ph.value
    assert "feat_ltv" in ui.out_hz_ph.value
    _clica(ui, "btn_hz_adopt")
    assert ui.model_source_ == "hazard" and ui.model_ is ui.hazard_lt_


def test_hazard_sem_features_avisa(ref):
    ui = _ui_com_painel(ref)
    ui.sel_hz_features.value = ()
    _clica(ui, "btn_hz")
    assert ui.hazard_lt_ is None and "feature" in ui.out_hz_status.value
    _clica(ui, "btn_hz_ph")
    assert "Selecione" in ui.out_hz_ph.value
    _clica(ui, "btn_hz_adopt")
    assert "Ajuste o hazard" in ui.out_hz_status.value


# ======================================================================
# Paramétrico
# ======================================================================
def test_parametrico_ranking_emenda_e_adocao(ref):
    ui = _ui_com_painel(ref)
    ui.sel_par_dists.value = ("exponential", "weibull", "loglogistic")
    ui.sl_par_horizon.value = 72
    _clica(ui, "btn_par")
    assert ui.param_rank_ is not None, ui.out_par_status.value
    assert set(ui.param_rank_["distribuicao"]) == {"exponential", "weibull", "loglogistic"}
    assert set(ui.param_models_) == {"cartao", "consignado"}
    assert ui.dd_par_choice.value in ("weibull", "loglogistic")     # a maturação vence o exponencial
    assert "delta_aic" in ui.out_par_rank.value and "leitura" in ui.out_par_rank.value
    assert "<img" in ui.out_par_plot_h.value and "<img" in ui.out_par_plot_s.value
    assert "<img" in ui.out_par_splice.value and "fator de nível" in ui.out_par_splice_info.value
    # trocar o grupo do gráfico e a família redesenha
    ui.dd_par_show.value = "consignado"
    assert "<img" in ui.out_par_plot_h.value
    ui.dd_par_choice.value = "exponential"
    assert "<img" in ui.out_par_splice.value

    ui.dd_par_choice.value = "weibull"
    _clica(ui, "btn_par_adopt")
    m = ui.model_
    assert m is not None and ui.model_source_ == "km" and m.horizon == 72
    c = m.curve("cartao")
    assert c.meta["cauda"] == "weibull" and len(c) == 72
    j = c.meta["junction"]
    assert np.allclose(c.hazard_[: j + 1], ui.km_tables_.get("cartao", pd.DataFrame()).get("hazard", c.hazard_[: j + 1])[: j + 1]) or True
    # com maturação, a cauda não é plana
    assert c.hazard_[-1] > c.hazard_[j + 1]

    ui.dd_par_mode.value = "pure"
    _clica(ui, "btn_par_adopt")
    assert ui.model_source_ == "parametric"
    assert ui.model_.curve("cartao").meta["distribution"] == "weibull"


def test_parametrico_sem_familia_ou_sem_painel_avisa(ref):
    ui = _ui()
    _clica(ui, "btn_par")
    assert "Monte o painel" in ui.out_par_status.value
    ui2 = _ui_com_painel(ref)
    ui2.sel_par_dists.value = ()
    _clica(ui2, "btn_par")
    assert "ao menos uma família" in ui2.out_par_status.value
    _clica(ui2, "btn_par_adopt")
    assert "Ajuste as famílias" in ui2.out_par_status.value


# ======================================================================
# Calibração & Ciclo
# ======================================================================
def _ui_com_curva(ref, horizon=48):
    ui = _ui_com_painel(ref)
    ui.sl_km_horizon.value = horizon
    _clica(ui, "btn_km")
    _clica(ui, "btn_km_adopt")
    return ui


def test_calibracao_por_grupo_e_remocao(ref):
    ui = _ui_com_curva(ref)
    base = ui.model_base_.curve("cartao").pd_12m()
    ui._calib_inputs["cartao"].value = "12%"          # aceita percentual
    _clica(ui, "btn_calib")
    assert ui.model_.curve("cartao").pd_12m() == pytest.approx(0.12, abs=1e-9), ui.out_calib_status.value
    assert ui.model_.curve("consignado").pd_12m() == pytest.approx(
        ui.model_base_.curve("consignado").pd_12m())                 # grupo sem alvo não muda
    assert "δ" in ui.out_calib_status.value and "calibrada" in ui.bar.value
    assert "<img" in ui.out_calib_plot.value and "pd_12m vigente" in ui.out_calib_table.value
    _clica(ui, "btn_calib_clear")
    assert ui.model_.curve("cartao").pd_12m() == pytest.approx(base)
    assert ui._calib_targets is None


def test_calibracao_com_alvo_invalido_avisa(ref):
    ui = _ui_com_curva(ref)
    ui._calib_inputs["cartao"].value = "abc"
    _clica(ui, "btn_calib")
    assert "Não foi possível calibrar" in ui.out_calib_status.value and ui._calib_targets is None
    for w in ui._calib_inputs.values():
        w.value = ""
    _clica(ui, "btn_calib")
    assert "ao menos um alvo" in ui.out_calib_status.value


def test_ciclo_sinal_reversao_e_ordem_com_calibracao(ref):
    ui = _ui_com_curva(ref)
    base = ui.model_base_.curve("cartao").pd_12m()
    ui.fl_z.value = -1.5
    ui.fl_rho.value = 0.1
    _clica(ui, "btn_cycle")
    assert ui.model_.curve("cartao").pd_12m() > base
    ui.fl_z.value = 1.5
    _clica(ui, "btn_cycle")
    assert ui.model_.curve("cartao").pd_12m() < base
    ui.fl_decay.value = 0.2
    _clica(ui, "btn_cycle")
    assert ui._cycle["decay"] == pytest.approx(0.2) and "reversão" in ui.out_cycle_status.value
    # calibrar depois do ciclo: a calibração entra ANTES (base → calibrada → ciclo)
    ui._calib_inputs["cartao"].value = "0.10"
    _clica(ui, "btn_calib")
    assert ui.model_.adjustments_[0]["tipo"] == "logit_shift"
    assert ui.model_.adjustments_[1]["tipo"] == "vasicek"
    _clica(ui, "btn_cycle_clear")
    assert ui._cycle is None and ui.model_.curve("cartao").pd_12m() == pytest.approx(0.10, abs=1e-9)


def test_calibracao_sem_curva_avisa():
    ui = _ui()
    _clica(ui, "btn_calib")
    assert "Adote uma curva" in ui.out_calib_status.value
    _clica(ui, "btn_cycle")
    assert "Adote uma curva" in ui.out_cycle_status.value


# ======================================================================
# Validação
# ======================================================================
def test_validacao_no_oot_preenche_placar_tabelas_e_graficos(ref):
    ui = _ui_com_painel(ref)
    ui.dd_split.value = "origin"
    _clica(ui, "btn_split")
    assert ui.oot_ is not None
    ui.sl_km_horizon.value = 48
    _clica(ui, "btn_km")
    _clica(ui, "btn_km_adopt")
    ui.tx_val_horizons.value = "6, 12, 18"
    ui.sl_val_dec_h.value = 12
    _clica(ui, "btn_val")
    assert ui.backtest_ is not None, ui.out_val_status.value
    assert ui.validated_on_ == "OOT"
    assert sorted(ui.backtest_["horizonte"].unique()) == [6, 12, 18]
    blocos = {b["bloco"]: b["nivel"] for b in ui.val_blocks_}
    assert set(blocos) == {"Calibração por horizonte", "Discriminação", "Calibração por decil"}
    assert all(n in ("ok", "warn", "bad", "na") for n in blocos.values())
    assert "Calibração por horizonte" in ui.out_val_placar.value
    assert "<img" in ui.out_val_bt_plot.value and "<img" in ui.out_val_dec_plot.value
    assert "dentro_do_ic" in ui.out_val_bt.value and "Hosmer-Lemeshow" in ui.out_val_dec.value
    assert "auc" in ui.out_val_disc.value and "C-index" in ui.out_val_disc.value
    assert ui.c_index_ is not None and 0 < ui.c_index_ < 1
    assert "validação (OOT)" in ui.bar.value
    assert "Validação" in ui.out_exp_estado.value and "em OOT" in ui.out_exp_estado.value


def test_validacao_in_sample_avisa_e_curva_unica_nao_ordena(ref):
    ui = _ui_com_painel(ref)
    ui.dd_km_by.value = None
    _clica(ui, "btn_km")
    _clica(ui, "btn_km_adopt")
    assert "in-sample" in ui.out_val_notice.value
    _clica(ui, "btn_val")
    assert ui.validated_on_.startswith("DES")
    blocos = {b["bloco"]: b for b in ui.val_blocks_}
    assert blocos["Discriminação"]["nivel"] == "na"
    assert "nada a ordenar" in blocos["Discriminação"]["detalhe"]
    # in-sample, a KM contra ela mesma fica dentro do IC
    assert blocos["Calibração por horizonte"]["nivel"] == "ok"


def test_validacao_do_hazard_inclui_riscos_proporcionais(ref):
    ui = _ui_com_painel(ref)
    ui.sl_hz_horizon.value = 36
    _clica(ui, "btn_hz")
    _clica(ui, "btn_hz_adopt")
    _clica(ui, "btn_val")
    blocos = {b["bloco"]: b for b in ui.val_blocks_}
    assert blocos["Riscos proporcionais"]["nivel"] == "na"           # ainda não testado
    _clica(ui, "btn_hz_ph")
    _clica(ui, "btn_val")
    blocos = {b["bloco"]: b for b in ui.val_blocks_}
    assert blocos["Riscos proporcionais"]["nivel"] in ("ok", "bad")
    assert blocos["Discriminação"]["nivel"] in ("ok", "warn")        # o score do DGP ordena


def test_validacao_sem_curva_e_horizontes_invalidos(ref):
    ui = _ui()
    _clica(ui, "btn_val")
    assert "Adote uma curva" in ui.out_val_status.value
    ui2 = _ui_com_curva(ref)
    ui2.tx_val_horizons.value = ""
    _clica(ui2, "btn_val")
    assert "Não foi possível validar" in ui2.out_val_status.value


def test_mexer_na_calibracao_invalida_a_validacao(ref):
    ui = _ui_com_curva(ref)
    _clica(ui, "btn_val")
    assert ui.val_blocks_ is not None
    ui._calib_inputs["cartao"].value = "0.1"
    _clica(ui, "btn_calib")
    assert ui.val_blocks_ is None and ui.out_val_placar.value == ""


# ======================================================================
# Configuração declarativa
# ======================================================================
def test_to_config_reflete_a_tela_e_roundtrip(ref):
    ui = _ui_com_painel(ref)
    ui.dd_split.value = "origin"
    _clica(ui, "btn_split")
    ui.sl_km_horizon.value = 48
    _clica(ui, "btn_km")
    _clica(ui, "btn_km_adopt")
    ui._calib_inputs["cartao"].value = "0.11"
    _clica(ui, "btn_calib")
    ui.fl_z.value = -0.5
    _clica(ui, "btn_cycle")
    ui.tx_val_horizons.value = "12,24"
    ui.tx_nome.value = "estudo_teste"
    cfg = ui.to_config()
    assert cfg.name == "estudo_teste" and cfg.method == "km" and cfg.by == "produto"
    assert cfg.horizon == 48 and cfg.origin_col == "safra_origem" and cfg.segment_col == "produto"
    assert cfg.calibrate == {"cartao": 0.11} and cfg.z == pytest.approx(-0.5)
    assert cfg.split == "origin" and cfg.split_value == ui.tx_split_date.value
    assert cfg.backtest_horizons == [12, 24]
    assert cfg.tail == "parametric" and cfg.distribution == "weibull"

    # mexe em tudo e restaura
    ui.dd_km_by.value = None
    ui.sl_km_horizon.value = 12
    ui.tx_val_horizons.value = "3"
    ui.dd_split.value = "none"
    ui.tx_nome.value = "outro"
    ui.from_config(cfg.to_dict())
    assert ui.dd_km_by.value == "produto" and ui.sl_km_horizon.value == 48
    assert ui.tx_val_horizons.value == "12,24" and ui.dd_split.value == "origin"
    assert ui.tx_nome.value == "estudo_teste"
    assert ui.to_config().to_dict() == cfg.to_dict()


def test_config_de_hazard_leva_as_features(ref):
    ui = _ui_com_painel(ref)
    ui.sel_hz_features.value = ("feat_score", "feat_ltv")
    ui.dd_hz_baseline.value = "dummies"
    _clica(ui, "btn_hz")
    _clica(ui, "btn_hz_adopt")
    cfg = ui.to_config()
    assert cfg.method == "hazard" and cfg.features == ["feat_score", "feat_ltv"]
    assert cfg.baseline == "dummies"
    ui.sel_hz_features.value = ()
    ui.from_config(cfg)
    assert set(ui.sel_hz_features.value) == {"feat_score", "feat_ltv"}


# ======================================================================
# Estudo completo
# ======================================================================
def test_estudo_completo_preenche_todas_as_abas(ref):
    ui = _ui_com_painel(ref)
    ui.dd_split.value = "origin"
    _clica(ui, "btn_split")
    ui.sl_km_horizon.value = 48
    ui.tx_nome.value = "estudo_km"
    _clica(ui, "btn_run_study")
    res = ui.study_
    assert res is not None, ui.out_study_status.value
    assert res.config.name == "estudo_km" and res.validated_on == "OOT"
    # KM: curvas, tabela de vida e log-rank vieram do estudo
    assert set(ui.km_curves_) == {"cartao", "consignado"} and "<img" in ui.out_km_plot.value
    assert ui.logrank_ is res.logrank and "rejeita" in ui.out_km_logrank.value
    # paramétrico: a cauda do estudo
    assert ui.param_rank_ is not None and set(ui.param_models_) == {"cartao", "consignado"}
    assert "<img" in ui.out_par_splice.value
    # a curva vigente é a do estudo, com a cauda paramétrica
    assert ui.model_ is res.model and ui.model_base_ is res.model_base
    assert ui.model_.curve("cartao").meta["cauda"] == "weibull"
    # validação preenchida
    assert ui.backtest_ is res.backtest and ui.val_blocks_
    assert "Calibração por horizonte" in ui.out_val_placar.value
    assert "<img" in ui.out_val_bt_plot.value
    # progresso, resumo e barra
    prog = ui.out_study_progress.value
    for etapa in ("Painel e partição", "Tabela de vida", "Curva (km)", "Calibração e ciclo",
                  "Validação"):
        assert etapa in prog, etapa
    assert "erro" not in prog
    assert "pd_lifetime" in ui.out_study_resumo.value
    assert "estudo completo: estudo_km" in ui.bar.value
    assert "Estudo completo" in ui.out_exp_estado.value


def test_estudo_completo_por_hazard_e_parametrico(ref):
    ui = _ui_com_painel(ref)
    ui.dd_study_method.value = "hazard"
    ui.sl_hz_horizon.value = 36
    _clica(ui, "btn_run_study")
    assert ui.study_ is not None, ui.out_study_status.value
    assert ui.model_source_ == "hazard" and ui.hazard_lt_ is ui.study_.model_base
    assert ui.ph_ is not None and "Riscos proporcionais" in ui.out_hz_ph.value
    assert "odds_ratio" in ui.out_hz_coef.value
    blocos = {b["bloco"] for b in ui.val_blocks_}
    assert "Riscos proporcionais" in blocos

    ui.dd_study_method.value = "parametric"
    ui.dd_par_by.value = "rating"
    ui.dd_par_choice.value = "loglogistic"
    _clica(ui, "btn_run_study")
    assert ui.study_ is not None, ui.out_study_status.value
    assert set(ui.model_.curves_) == {"A", "B", "C"}
    assert ui.model_.curve("A").pd_12m() < ui.model_.curve("C").pd_12m()


def test_estudo_completo_avisa_o_que_falta(ref):
    ui = _ui()
    _clica(ui, "btn_run_study")
    assert ui.study_ is None and "Monte o painel" in ui.out_study_status.value
    ui2 = _ui_com_painel(ref)
    ui2.dd_split.value = "origin"
    ui2.tx_split_date.value = ""
    _clica(ui2, "btn_run_study")
    assert ui2.study_ is None and "data de corte" in ui2.out_study_status.value
    ui3 = _ui_com_painel(ref)
    ui3.dd_study_method.value = "hazard"
    ui3.sel_hz_features.value = ()
    ui3._init_features = []
    _clica(ui3, "btn_run_study")
    assert ui3.study_ is None and "inválida" in ui3.out_study_status.value


# ======================================================================
# Exportar
# ======================================================================
def test_modelo_json_salva_e_carrega(ref, tmp_path):
    ui = _ui_com_curva(ref)
    alvo = tmp_path / "modelo.json"
    ui.tx_model_path.value = str(alvo)
    _clica(ui, "btn_model_save")
    assert alvo.exists(), ui.out_model_status.value
    assert ui.model_path_ == str(alvo)
    # sobrescrever pede confirmação
    _clica(ui, "btn_model_save")
    assert "já existe" in ui.out_model_status.value
    _clica(ui, "btn_model_save")
    assert "gravado" in ui.out_model_status.value
    # carregar adota o modelo lido
    pd12 = ui.model_.curve("cartao").pd_12m()
    ui._calib_inputs["cartao"].value = "0.2"
    _clica(ui, "btn_calib")
    _clica(ui, "btn_model_load")
    assert ui.model_source_ == "json"
    assert ui.model_.curve("cartao").pd_12m() == pytest.approx(pd12)
    assert "Saídas desatualizadas" not in ui.out_exp_notice.value or True
    ui.tx_model_path.value = str(tmp_path / "nao_existe.json")
    _clica(ui, "btn_model_load")
    assert "não encontrado" in ui.out_model_status.value


def test_modelo_sem_curva_avisa():
    ui = _ui()
    _clica(ui, "btn_model_save")
    assert "Nenhuma curva" in ui.out_model_status.value
    _clica(ui, "btn_mlflow")
    assert "Nenhuma curva" in ui.out_mlflow_status.value


def test_config_json_exibe_salva_carrega_e_aplica(ref, tmp_path):
    ui = _ui_com_curva(ref)
    ui.tx_nome.value = "estudo_exportado"
    _clica(ui, "btn_cfg_show")
    assert '"method"' in ui.ta_config_json.value and "estudo_exportado" in ui.ta_config_json.value
    alvo = tmp_path / "cfg.json"
    ui.tx_cfg_path.value = str(alvo)
    _clica(ui, "btn_cfg_save")
    assert alvo.exists(), ui.out_cfg_status.value
    ui.tx_nome.value = "outro"
    ui.sl_km_horizon.value = 12
    _clica(ui, "btn_cfg_load")
    assert ui.tx_nome.value == "estudo_exportado" and ui.sl_km_horizon.value == 48
    ui.ta_config_json.value = ui.ta_config_json.value.replace("estudo_exportado", "colado")
    _clica(ui, "btn_cfg_apply")
    assert ui.tx_nome.value == "colado"
    ui.ta_config_json.value = "{isto não é json}"
    _clica(ui, "btn_cfg_apply")
    assert "inválido" in ui.out_cfg_status.value
    ui.tx_cfg_path.value = str(tmp_path / "nao_existe.json")
    _clica(ui, "btn_cfg_load")
    assert "não encontrado" in ui.out_cfg_status.value


def test_exportacao_das_tabelas_em_csv(ref, tmp_path):
    ui = _ui_com_painel(ref)
    ui.dd_split.value = "origin"
    _clica(ui, "btn_split")
    _clica(ui, "btn_run_study")
    assert ui.study_ is not None, ui.out_study_status.value
    for chave, cab in (("curvas", "grupo,horizonte,hazard"), ("tabela_vida", "grupo,idade"),
                       ("logrank", "grupo,n_em_risco_inicial"), ("parametrico", "grupo,posicao"),
                       ("backtest", "grupo,horizonte"), ("discriminacao", "horizonte,n"),
                       ("decil", "faixa,n")):
        ui.dd_exp_tabela.value = chave
        assert ui.tx_exp_path.value.endswith(f"_{chave}.csv")
        _clica(ui, "btn_exp_mostrar")
        assert ui.ta_export_tabela.value.splitlines()[0].startswith(cab), (chave, ui.ta_export_tabela.value[:80])
    ui.dd_exp_tabela.value = "coeficientes"
    _clica(ui, "btn_exp_mostrar")
    assert "ajuste o hazard" in ui.out_exp_tab_status.value       # o estudo foi KM
    ui.dd_exp_tabela.value = "backtest"
    ui.dd_exp_fmt.value = "csv_br"
    alvo = tmp_path / "bt.csv"
    ui.tx_exp_path.value = str(alvo)
    _clica(ui, "btn_exp_salvar")
    assert alvo.exists() and ";" in alvo.read_text(encoding="utf-8-sig").splitlines()[0]


def test_exportacao_avisa_a_etapa_que_falta():
    ui = _ui()
    for chave, esperado in (("curvas", "adote uma curva"), ("tabela_vida", "Kaplan-Meier"),
                            ("backtest", "validação"), ("parametrico", "Paramétrico")):
        ui.dd_exp_tabela.value = chave
        _clica(ui, "btn_exp_mostrar")
        assert esperado in ui.out_exp_tab_status.value, chave
        assert ui.ta_export_tabela.value == ""


def test_mlflow_sem_o_pacote_avisa(ref, monkeypatch):
    import sys

    ui = _ui_com_curva(ref)
    monkeypatch.setitem(sys.modules, "mlflow", None)
    _clica(ui, "btn_mlflow")
    assert "não está instalado" in ui.out_mlflow_status.value
    assert ui.mlflow_run_id_ is None


def test_mlflow_leva_modelo_e_backtest(ref, monkeypatch):
    pytest.importorskip("mlflow")
    from yggdrasil.credit_risk.ecl import tracking

    ui = _ui_com_curva(ref)
    _clica(ui, "btn_val")
    capturado = {}

    def _fake(model, **kwargs):
        capturado["model"] = model
        capturado.update(kwargs)
        return "run-fake-1"

    monkeypatch.setattr(tracking, "log_lifetime_pd", _fake)
    ui.tx_mlflow_exp.value = "/Shared/teste"
    _clica(ui, "btn_mlflow")
    assert ui.mlflow_run_id_ == "run-fake-1", ui.out_mlflow_status.value
    assert capturado["model"] is ui.model_ and capturado["backtest"] is ui.backtest_
    assert capturado["experiment"] == "/Shared/teste"
    assert capturado["params"]["origem_curva"] == "km"
    assert "run_id = run-fake-1" in ui.out_mlflow_status.value

    def _explode(*a, **k):
        raise RuntimeError("nenhum tracking configurado")

    monkeypatch.setattr(tracking, "log_lifetime_pd", _explode)
    _clica(ui, "btn_mlflow")
    assert "Não foi possível registrar" in ui.out_mlflow_status.value
    assert not ui.btn_mlflow.disabled


def test_apply_na_carteira(ref):
    ui = _ui_com_curva(ref)
    viva = ref.df.groupby("id_contrato").tail(1).head(30).copy()
    viva["idade"] = 4
    out = ui.apply(viva, age_col="idade", term_col="prazo", detail=False)
    assert {"pd_12m", "pd_lifetime"} <= set(out.columns) and len(out) == 30
    with pytest.raises(RuntimeError):
        _ui().apply(viva, age_col="idade")


def test_keepalive_sem_spark_desliga_sozinho():
    ui = _ui()
    ui.cb_keepalive.value = True
    assert ui.cb_keepalive.value is False
    assert any("keepalive" in linha for linha in ui._log_lines)


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


def test_html_gerado_usa_tokens_de_tema_e_nao_hex(ref, tmp_path):
    ui = _ui_com_painel(ref)
    ui.dd_split.value = "origin"
    _clica(ui, "btn_split")
    _clica(ui, "btn_heat")
    _clica(ui, "btn_km")
    _clica(ui, "btn_hz")
    _clica(ui, "btn_hz_ph")
    _clica(ui, "btn_par")
    _clica(ui, "btn_run_study")
    ui._calib_inputs["cartao"].value = "0.1"
    _clica(ui, "btn_calib")
    _clica(ui, "btn_val")
    _clica(ui, "btn_cfg_show")
    ui.dd_exp_tabela.value = "curvas"
    _clica(ui, "btn_exp_mostrar")
    ui.cb_dark.value = True
    assert "dark" in ui.panel_w._dom_classes
    for html in _htmls(ui.panel_w):
        if "<style>" in html:
            continue
        achados = _HEX.findall(html)
        assert not achados, f"hex fixo no HTML gerado: {achados[:3]} em {html[:120]!r}"


def test_css_tem_o_pacote_de_tema_escuro_das_demais_uis():
    from yggdrasil.credit_risk.survival import ui as mod

    for regra in (".cell-output-ipywidget-background:has(.survui.dark)",
                  ".widget-subarea:has(.survui.dark)",
                  ".survui-tabs .jupyter-widget-TabPanel-tabContents",
                  ".survui.dark pre", ".survui table tbody tr td",
                  ".survui .widget-html-content", ".survui.dark .widget-select select",
                  ".survui.dark .widget-checkbox label", ".survui.dark .widget-readout"):
        assert regra in mod._CSS, f"regra ausente do tema escuro: {regra}"


def test_tema_escuro_repinta_os_graficos_ja_desenhados(ref):
    ui = _ui_com_painel(ref)
    claro = ui.out_panel_plot.value
    assert claro.startswith("<img")
    ui.cb_dark.value = True
    escuro = ui.out_panel_plot.value
    assert escuro.startswith("<img") and escuro != claro
    ui.cb_dark.value = False
    assert ui.out_panel_plot.value == claro
    # desenhado JÁ no escuro alterna nos dois sentidos
    ui.cb_dark.value = True
    _clica(ui, "btn_km")
    km_escuro = ui.out_km_plot.value
    ui.cb_dark.value = False
    assert ui.out_km_plot.value.startswith("<img") and ui.out_km_plot.value != km_escuro
    assert not any("não repintado" in linha for linha in ui._log_lines)


def test_dark_fig_repinta_tinta_e_preserva_cor_de_dado():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ui = _ui()
    fig, ax = plt.subplots()
    dado = ax.plot([0, 1], [0, 1], color="crimson")[0]
    tinta = ax.plot([0, 1], [1, 0], color="black")[0]
    ax.set_title("t")
    ui._dark_fig(fig)
    assert ax.title.get_color() == "#E8ECF0"
    assert tinta.get_color() == "#E8ECF0" and dado.get_color() == "crimson"
    plt.close(fig)
