"""
SurvivalUI
==========
Camada interativa (``ipywidgets``) sobre a **análise de sobrevivência para a PD
lifetime**: a bancada que constrói, escolhe e defende a estrutura a termo da PD,
no mesmo desenho das demais UIs do ``credit_risk`` (abas, console, tema
claro/escuro, configuração em JSON, MLflow), para o fluxo **painel de contratos →
curva → cauda → calibração/ciclo → validação**:

* **Painel**: o mapeamento das colunas do painel longo (ou o painel de
  referência sintético), o retrato do painel (contratos, quebras, censura por
  idade, o mapa safra × idade), a **partição DES/OOT** e o botão **Rodar estudo
  completo**;
* **Kaplan-Meier**: as curvas não paramétricas por grupo nas quatro
  representações, com a banda de Greenwood, o Nelson-Aalen, a tabela de vida, a
  mediana e o tempo médio de sobrevivência, e o **log-rank** que diz se os grupos
  diferem (global e par a par);
* **Hazard**: a regressão de *hazard* em tempo discreto com covariáveis: o
  *baseline* em idade, os coeficientes com razão de chances, as curvas por perfil
  de risco (P10/P50/P90 do preditor linear) e o teste de **riscos
  proporcionais**;
* **Paramétrico**: as cinco famílias ajustadas por máxima verossimilhança com o
  ranking por AIC, sobrepostas ao KM, e a **emenda** da curva empírica com a cauda
  paramétrica além da idade em que a base deixa de sustentar a taxa;
* **Calibração & Ciclo**: o nível colado na PD de 12 meses do modelo transversal
  (por grupo) e o deslocamento ao ciclo por Vasicek, com o antes × depois;
* **Validação**: no OOT da partição (ou no próprio DES, sinalizado): backtest por
  horizonte com ``z`` de Greenwood, discriminação (AUC/Gini/KS e C-index),
  calibração por decil com Hosmer-Lemeshow e riscos proporcionais, num placar por
  família com a leitura em texto;
* **Exportar**: o modelo em JSON (``LifetimePD``, que o ECL consome), a
  configuração do estudo em JSON (exibir, salvar, carregar, aplicar), as tabelas em
  CSV e o registro do *run* no MLflow.

Qualquer aba **adota** a curva que produziu como a curva do estudo; a partir daí
Calibração, Validação e Exportar operam sobre ela. O botão **Rodar estudo
completo** encadeia tudo a partir da configuração da tela.

    from yggdrasil.credit_risk.survival import SurvivalUI
    ui = SurvivalUI()                                            # painel de referência
    ui = SurvivalUI(df, origin_col="safra_origem", segment_col="produto",
                    features=["feat_score", "feat_ltv"])         # o seu painel longo
    ui

``ipywidgets``, ``IPython`` e ``matplotlib`` são opcionais: o import deste módulo
não os exige (são carregados na construção, com mensagem clara quando faltarem).
"""
from __future__ import annotations

import html as _html
import re
import unicodedata
from contextlib import contextmanager, suppress
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from ..ecl.curves import PDCurve, curve_frame, vintage_curve
from ..ecl.lifetime_pd import LifetimePD
from ..ecl.panel import ContractPanel
from ..ecl.survival import BASELINES, HAZARD_LINKS, kaplan_meier
from .lifetable import (
    GLOBAL,
    life_table,
    logrank_test,
    median_survival,
    pairwise_logrank,
    restricted_mean_survival,
)
from .parametric import (
    DISTRIBUTION_LABELS,
    DISTRIBUTIONS,
    ParametricSurvival,
    fit_parametric,
    junction_age,
    splice_curves,
)
from .study import STUDY_METHODS, SurvivalConfig, run_survival_study, split_panel
from .synthetic import FEATURES, make_reference_panel
from .validation import (
    backtest_curve,
    calibration_by_decile,
    discrimination_by_horizon,
    model_concordance,
    ph_test,
)

# ----------------------------------------------------------------------
# Dependências opcionais (import tardio)
# ----------------------------------------------------------------------
W = None            # ipywidgets
_clear_output = None
_display = None


def _require_widgets() -> None:
    """Importa ``ipywidgets``/``IPython`` na **primeira** construção da interface."""
    global W, _clear_output, _display
    if W is not None:
        return
    try:
        import ipywidgets as _W
        from IPython.display import clear_output as _co
        from IPython.display import display as _di
    except Exception as exc:  # pragma: no cover - depende do ambiente
        raise ImportError(
            "SurvivalUI requer ipywidgets e IPython (Jupyter). "
            "Instale com: pip install ipywidgets  (ou o extra 'ui' do yggdrasil)."
        ) from exc
    W, _clear_output, _display = _W, _co, _di


def _sem_acento(txt: str) -> str:
    n = unicodedata.normalize("NFKD", str(txt))
    return "".join(c for c in n if not unicodedata.combining(c)).lower()


def _esc(txt) -> str:
    """Escapa texto vindo de dados antes de interpolar em HTML."""
    return _html.escape(str(txt), quote=True)


# ======================================================================
# Estilo (mesmos tokens semânticos das demais UIs do credit_risk)
# ======================================================================
_CSS = """
<style>
/* SEM @import externo de fontes: no Databricks o cluster costuma não ter egress. */
.survui { --ac:#3b4a63; --ac-deep:#27324a; --ac-soft:#eef1f5; --ac-border:#cdd5e0;
  --ink:#1f2733; --muted:#6b7480; --line:#e7e9ee;
  --ok-ink:#157a52; --ok-bg:#e7f5ee; --ok-tx:#137a3e;
  --warn-ink:#9a6f12; --warn-bg:#fbf3e0; --warn-tx:#9a6b00;
  --bad-ink:#b23a2a; --bad-bg:#fbe7e4; --bad-tx:#b3261e;
  --info-ink:#1f5fa8; --info-bg:#e7eef8; --sus-ink:#6b46c1; --sus-bg:#efe9fb;
  --code-ink:#0b63ce; --code-bg:#e7eef8;
  --strong-ink:#15324a; --body-ink:#3a4658; --sub-ink:#8a93a3;
  --faint-ink:#9aa3ad; --hair:#eef0f3; --tile-bg:#f7f8fa; --rule-bg:#fff;
  --help-bg:#f5f8fc; --help-line:#dbe5f1; --neutral-bg:#f1f3f5;
  --tbl-line:#e1e5ec; --tbl-line-strong:#cdd5e0; --tbl-head-bg:#eef1f5;
  --tbl-head-ink:#27324a; --tbl-head-line:#b9c2d0; --tbl-zebra:#fafbfc;
  --tbl-hover:#eef3f8;
  --notice-bg:#fff8e6; --notice-border:#f0c36d; --notice-ink:#664d03;
  font-family:'IBM Plex Sans', -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  color:var(--ink); }
.survui .mono { font-family:'IBM Plex Mono', ui-monospace, Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums; }
.survui-banner { display:flex; align-items:center; gap:11px; background:#fff;
  border:1px solid var(--line); border-radius:13px; padding:11px 16px; margin-bottom:10px;
  box-shadow:0 1px 3px rgba(16,24,40,.08); }
.survui-banner .logo { width:30px; height:30px; border-radius:9px; background:var(--ac);
  color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700;
  font-size:12px; flex:none; }
.survui-banner .t { font-size:15px; font-weight:600; color:var(--ink); line-height:1.2; }
.survui-banner .s { font-size:11.5px; color:var(--muted); margin-top:1px; }
.survui-card { background:#fff; border:1px solid var(--line); border-radius:12px;
  padding:13px 15px; box-shadow:0 1px 3px rgba(16,24,40,.06); margin-bottom:11px;
  overflow-x:clip; }
.survui-h { font-weight:600; font-size:11px; color:var(--muted); text-transform:uppercase;
  letter-spacing:.07em; margin-bottom:9px; }
.survui-bar { background:#fff; border:1px solid var(--line); border-radius:11px;
  box-shadow:0 1px 3px rgba(16,24,40,.05); padding:8px 12px; overflow-x:auto; }
.pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:11.5px;
  font-weight:600; margin:2px 4px 2px 0; }
.pill-muted  { background:var(--ac-soft); color:var(--ac-deep); }
.pill-green  { background:var(--ok-bg); color:var(--ok-ink); }
.pill-yellow { background:var(--warn-bg); color:var(--warn-ink); }
.pill-red    { background:var(--bad-bg); color:var(--bad-ink); }
.survui-legend { font-size:11px; color:var(--muted); margin:6px 0 2px; line-height:1.55; }
.survui-help { background:var(--help-bg); border:1px solid var(--help-line);
  border-left:3px solid var(--ac); border-radius:9px; padding:11px 14px; font-size:11.5px;
  color:var(--body-ink); line-height:1.62; margin-top:8px; }
.survui-help .ttl { font-size:12px; font-weight:600; color:var(--ink); margin-bottom:5px; }
.survui-help ul { margin:5px 0 0; padding-left:6px; list-style:none; }
.survui-help li { margin:4px 0; }
.survui-help code, .survui-help .pname { font-family:'IBM Plex Mono', ui-monospace, monospace;
  font-weight:600; color:var(--code-ink); background:var(--code-bg); padding:1px 6px;
  border-radius:5px; }
.survui-notice { border:1px solid var(--notice-border); background:var(--notice-bg);
  border-radius:10px; padding:9px 12px; font-size:12px; color:var(--notice-ink);
  margin-bottom:8px; }
.survui-metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(104px,1fr)); gap:6px; }
.survui-metric { background:var(--tile-bg); border:1px solid var(--hair); border-radius:9px;
  padding:7px 10px; }
.survui-metric .k { font-size:10px; text-transform:uppercase; letter-spacing:.04em;
  color:var(--sub-ink); }
.survui-metric .v { font-size:16px; font-weight:600; color:var(--ink); margin-top:2px;
  font-variant-numeric: tabular-nums; }
.survui-placar { display:grid; grid-template-columns:repeat(auto-fit,minmax(196px,1fr)); gap:8px; }
.survui-bloco { background:var(--tile-bg); border:1px solid var(--hair); border-radius:10px;
  padding:8px 11px; border-left-width:3px; border-left-style:solid;
  border-left-color:var(--faint-ink); }
.survui-bloco.ok   { border-left-color:var(--ok-ink); }
.survui-bloco.warn { border-left-color:var(--warn-ink); }
.survui-bloco.bad  { border-left-color:var(--bad-ink); }
.survui-bloco .k { font-size:10px; text-transform:uppercase; letter-spacing:.04em;
  color:var(--sub-ink); }
.survui-bloco .v { font-size:13px; font-weight:600; margin:3px 0 2px; }
.survui-bloco .d { font-size:11px; color:var(--muted); line-height:1.5;
  font-variant-numeric: tabular-nums; }
.survui-prog { border-collapse:collapse; font-size:12px; width:100%; margin:2px 0 8px; }
.survui-prog th { padding:4px 10px; text-align:left; background:var(--tbl-head-bg);
  color:var(--tbl-head-ink); font-weight:600; }
.survui-prog td { padding:4px 10px; border-top:1px solid var(--tbl-line); }
/* abas — "segmented control" (pílulas), igual às demais UIs do credit_risk */
.survui-tabs { margin-top:10px; border:none !important; box-shadow:none !important; }
.survui-tabs > .widget-tab-contents { padding:30px 2px 2px !important; background:transparent;
  border:none !important; box-shadow:none !important; }
.survui-tabs .lm-TabBar.jupyter-widget-tab-nav,
.survui-tabs .p-TabBar.jupyter-widget-tab-nav { border-bottom:1px solid var(--line) !important;
  padding-bottom:14px !important; margin-bottom:0 !important; box-shadow:none !important; }
.survui-tabs .lm-TabBar-content, .survui-tabs .p-TabBar-content { gap:7px;
  align-items:stretch; border:none; }
.survui-tabs .lm-TabBar-tab, .survui-tabs .p-TabBar-tab { font-size:13px;
  min-width:max-content !important; max-width:none !important; flex:0 0 auto !important;
  margin:0 !important; padding:8px 16px !important;
  border:1px solid var(--line) !important; border-radius:9px !important;
  background:#fff !important; color:var(--muted) !important; font-weight:500;
  line-height:1.15; outline:none !important; box-shadow:none !important;
  transition:background .15s, color .15s, border-color .15s; }
.survui-tabs .lm-TabBar-tab::before, .survui-tabs .lm-TabBar-tab::after,
.survui-tabs .p-TabBar-tab::before, .survui-tabs .p-TabBar-tab::after {
  display:none !important; content:none !important; background:none !important; }
.survui-tabs .lm-TabBar-tab:hover, .survui-tabs .p-TabBar-tab:hover {
  background:var(--ac-soft) !important; color:var(--ac-deep) !important;
  border-color:var(--ac-border) !important; }
.survui-tabs .lm-TabBar-tabLabel, .survui-tabs .p-TabBar-tabLabel {
  white-space:nowrap !important; overflow:visible !important;
  text-overflow:clip !important; max-width:none !important; }
.survui-tabs .lm-TabBar-tab.lm-mod-current,
.survui-tabs .p-TabBar-tab.p-mod-current { color:#fff !important; font-weight:600;
  background:var(--ac) !important; border:1px solid var(--ac) !important;
  outline:none !important; box-shadow:none !important; }
.survui-tabs .lm-TabBar-tab.lm-mod-current:hover,
.survui-tabs .p-TabBar-tab.p-mod-current:hover {
  background:var(--ac-deep) !important; color:#fff !important;
  border-color:var(--ac-deep) !important; }
.survui .jupyter-button { border-radius:8px; font-family:inherit; }
.survui .jupyter-widgets { min-width:0 !important; }
/* ===== TEMA ESCURO (classe .dark no painel raiz) =====
   Paleta alinhada ao dark mode do Databricks (design system DuBois). */
.survui.dark { --ink:#E8ECF0; --muted:#92A4B3; --line:#37444F; --ac-soft:#37444F;
  --ac-border:#5F7281; --ac-deep:#E8ECF0; --ac:#4299E0;
  --ok-ink:#3BA65E; --ok-bg:rgba(39,124,67,.16); --ok-tx:#3BA65E;
  --warn-ink:#DE7921; --warn-bg:rgba(190,80,30,.16); --warn-tx:#DE7921;
  --bad-ink:#E65B77; --bad-bg:rgba(200,45,76,.16); --bad-tx:#E65B77;
  --info-ink:#8ACAFF; --info-bg:rgba(138,202,255,.16);
  --sus-ink:#B592E5; --sus-bg:rgba(138,99,191,.24);
  --code-ink:#8ACAFF; --code-bg:rgba(138,202,255,.16);
  --strong-ink:#E8ECF0; --body-ink:#C0CDD8; --sub-ink:#8396A5;
  --faint-ink:#5F7281; --hair:#37444F; --tile-bg:#11171C; --rule-bg:#11171C;
  --help-bg:#11171C; --help-line:#37444F; --neutral-bg:rgba(144,164,181,.16);
  --tbl-line:#37444F; --tbl-line-strong:#445461; --tbl-head-bg:#11171C;
  --tbl-head-ink:#E8ECF0; --tbl-head-line:#445461; --tbl-zebra:rgba(189,205,219,.04);
  --tbl-hover:rgba(189,205,219,.08);
  --notice-bg:rgba(190,80,30,.16); --notice-border:#DE7921; --notice-ink:#E8ECF0;
  background:#11171C; padding:8px; border-radius:12px; }
.survui.dark .survui-banner, .survui.dark .survui-card, .survui.dark .survui-bar {
  background:#1F272D !important; border-color:#37444F !important; box-shadow:none !important; }
.survui.dark .survui-banner .t { color:#E8ECF0; }
.survui.dark .survui-banner .logo { color:#11171C; }
.survui.dark .survui-tabs .p-TabBar-tab, .survui.dark .survui-tabs .lm-TabBar-tab {
  background:#1F272D !important; color:#92A4B3 !important; border-color:#37444F !important; }
.survui.dark .survui-tabs .p-TabBar-tab:hover,
.survui.dark .survui-tabs .lm-TabBar-tab:hover { background:rgba(138,202,255,.08) !important;
  color:#8ACAFF !important; border-color:#8ACAFF !important; }
.survui.dark .survui-tabs .p-TabBar-tab.p-mod-current,
.survui.dark .survui-tabs .lm-TabBar-tab.lm-mod-current { background:#4299E0 !important;
  color:#11171C !important; border-color:#4299E0 !important; }
.survui.dark .survui-tabs .p-TabBar-tab.p-mod-current:hover,
.survui.dark .survui-tabs .lm-TabBar-tab.lm-mod-current:hover {
  background:#8ACAFF !important; color:#11171C !important; border-color:#8ACAFF !important; }
.survui.dark .widget-text input, .survui.dark .widget-dropdown select, .survui.dark textarea,
.survui.dark .widget-select select {
  background:#11171C !important; color:#E8ECF0 !important; border-color:#37444F !important; }
.survui.dark .widget-select select option { background:#11171C; color:#E8ECF0; }
.survui.dark .widget-label, .survui.dark .jupyter-widgets label { color:#D1D9E1 !important; }
.survui.dark .widget-checkbox label, .survui.dark .widget-checkbox label span,
.survui.dark .widget-label-basic, .survui.dark .widget-label-basic span {
  color:#D1D9E1 !important; }
.survui.dark .widget-readout { color:#E8ECF0 !important; }
.survui.dark .jupyter-button:not(.mod-primary):not(.mod-success):not(.mod-info):not(.mod-warning):not(.mod-danger) { background:#37444F !important; color:#E8ECF0 !important; }
.survui.dark .jupyter-button.mod-active { background:#4299E0 !important; color:#11171C !important; }
/* ===== pacote de tema escuro (portado das demais UIs) ===== */
.survui table tbody tr td, .survui table tbody tr th { color:var(--ink); }
.survui .widget-html-content { color:var(--ink); }
.survui-tabs, .survui-tabs .lm-TabBar, .survui-tabs .p-TabBar,
.survui-tabs .widget-tab-contents,
.survui-tabs .jupyter-widget-TabPanel-tabContents {
  background:transparent !important; }
.cell-output-ipywidget-background:has(.survui.dark),
.jp-OutputArea-output:has(.survui.dark),
.jp-Cell-outputArea:has(.survui.dark),
.widget-subarea:has(.survui.dark) {
  background:transparent !important; }
.survui.dark pre { color:var(--ink) !important; background:transparent !important; }
.survui.dark img { border-radius:6px; }
</style>
"""

#: Representações da curva oferecidas nos gráficos.
KINDS = (("PD acumulada", "cumulative"), ("Sobrevivência", "survival"),
         ("PD condicional (hazard)", "hazard"), ("PD marginal", "marginal"))

#: Rótulo do eixo Y por representação.
_ROTULO = {"cumulative": "PD acumulada", "survival": "S(t)", "hazard": "hazard h(t)",
           "marginal": "PD marginal"}

#: Motores oferecidos ao botão "Rodar estudo completo" e ao seletor da aba KM.
METODOS_ESTUDO = (("Kaplan-Meier (+ cauda paramétrica)", "km"),
                  ("Safra / vintage (+ cauda paramétrica)", "vintage"),
                  ("Hazard em tempo discreto (covariáveis)", "hazard"),
                  ("Paramétrico puro", "parametric"))

#: Como ler o veredito de cada bloco da validação quando ele reprova.
_CONSELHO_VAL = {
    "Calibração por horizonte":
        "a curva erra o <b>nível</b> em algum horizonte: se o erro tem o mesmo sinal em "
        "todos, recalibre à PD de 12 meses do OOT (aba <i>Calibração</i>); se muda de "
        "sinal, o <b>formato</b> da maturação está errado — revise a cauda paramétrica ou o "
        "baseline do hazard.",
    "Discriminação":
        "a curva não ordena o risco no horizonte: com curva única isso é esperado (o "
        "ordenamento vem do scorecard); com covariáveis, revise as features ou o "
        "<i>baseline</i>.",
    "Calibração por decil":
        "há faixas de PD prevista cuja taxa observada fica fora do IC: o modelo separa "
        "bem no agregado mas erra nas pontas — típico de covariável com efeito "
        "não linear ou de cauda mal emendada.",
    "Riscos proporcionais":
        "o efeito das covariáveis <b>muda com a idade</b>: a curva do contrato jovem e a "
        "do maduro têm inclinações distintas. Considere interações com a idade ou um "
        "modelo por faixa de idade.",
}


class SurvivalUI:
    """Interface de análise de sobrevivência para a PD *lifetime*.

    Parameters
    ----------
    df:
        O painel **longo** de contratos (uma linha por contrato × safra de
        observação), ou ``None`` para começar vazio e carregar o painel de
        referência na aba **Painel**.
    id_col, date_col, default_col, age_col, origin_col, term_col, segment_col,
    exposure_col, freq:
        O mapeamento de colunas do
        :class:`~yggdrasil.credit_risk.ecl.panel.ContractPanel`. Todos podem ser
        trocados na aba **Painel** antes de montar o painel.
    features:
        Covariáveis pré-marcadas para a regressão de *hazard*.
    by:
        Coluna de agrupamento inicial das curvas (padrão: ``segment_col``).
    horizon:
        Horizonte da curva em períodos (padrão ``60``).
    name:
        Nome do estudo (vai para a :class:`SurvivalConfig`, os arquivos e o MLflow).

    Attributes
    ----------
    panel:
        O :class:`ContractPanel` montado (``None`` até haver dados).
    des_, oot_:
        A partição corrente (``oot_`` é ``None`` sem partição).
    life_table_, logrank_, pairwise_:
        Saídas da aba Kaplan-Meier.
    hazard_lt_, ph_:
        O :class:`LifetimePD` de *hazard* ajustado e o teste de riscos proporcionais.
    param_rank_, param_models_:
        Ranking por AIC (todas as famílias, por grupo) e ``{grupo: {família: modelo}}``.
    model_base_, model_, model_source_:
        A curva **adotada** (antes dos ajustes), a vigente (com calibração e ciclo)
        e de onde ela veio (``'km'``, ``'vintage'``, ``'hazard'``, ``'parametric'``,
        ``'json'``, ``'estudo'``).
    backtest_, discrimination_, decile_, c_index_, val_blocks_:
        Saídas da aba Validação.
    study_:
        O :class:`SurvivalResult` do botão **Rodar estudo completo**.
    """

    _TABLE_STYLES = [
        {"selector": "", "props": [("border-collapse", "collapse"),
                                   ("border", "1px solid var(--tbl-line-strong)"),
                                   ("width", "100%")]},
        {"selector": "th, td", "props": [("border", "1px solid var(--tbl-line)"),
                                         ("padding", "4px 9px"), ("text-align", "right"),
                                         ("white-space", "nowrap")]},
        {"selector": "thead th", "props": [("background-color", "var(--tbl-head-bg)"),
                                           ("color", "var(--tbl-head-ink)"),
                                           ("font-weight", "600"),
                                           ("border-bottom", "2px solid var(--tbl-head-line)"),
                                           ("position", "sticky"), ("top", "0"), ("z-index", "1")]},
        {"selector": "tbody tr:nth-child(even) td",
         "props": [("background-color", "var(--tbl-zebra)")]},
        {"selector": "tbody tr:hover td", "props": [("background-color", "var(--tbl-hover)")]},
    ]

    ABAS = ("Painel", "Kaplan-Meier", "Hazard", "Paramétrico", "Calibração & Ciclo",
            "Validação", "Exportar")

    #: colunas exportáveis em CSV: (chave, rótulo)
    _TABELAS_EXPORT = (
        ("curvas", "curvas do modelo vigente (formato longo)"),
        ("tabela_vida", "tabela de vida (Kaplan-Meier + Nelson-Aalen)"),
        ("logrank", "log-rank por grupo"),
        ("coeficientes", "coeficientes da regressão de hazard"),
        ("parametrico", "ranking das famílias paramétricas"),
        ("backtest", "backtest por horizonte"),
        ("discriminacao", "discriminação por horizonte"),
        ("decil", "calibração por decil"),
    )

    # ==================================================================
    # Construção
    # ==================================================================
    def __init__(self, df: Optional[pd.DataFrame] = None, *, id_col="id_contrato",
                 date_col="dt_ref", default_col="default", age_col=None, origin_col=None,
                 term_col=None, segment_col=None, exposure_col=None, freq="M",
                 features=None, by=None, horizon=60, name="estudo_sobrevivencia"):
        _require_widgets()
        self.study_name = str(name or "estudo_sobrevivencia")
        self.df: Optional[pd.DataFrame] = None
        self.panel: Optional[ContractPanel] = None
        self.cols: Dict[str, Optional[str]] = dict(
            id_col=id_col, date_col=date_col, default_col=default_col, age_col=age_col,
            origin_col=origin_col, term_col=term_col, segment_col=segment_col,
            exposure_col=exposure_col, freq=freq)
        self._init_features = list(features or [])
        self._init_by = by
        self._init_horizon = int(horizon)

        # --- estado ---------------------------------------------------------
        self.des_: Optional[ContractPanel] = None
        self.oot_: Optional[ContractPanel] = None
        self.life_table_: Optional[pd.DataFrame] = None
        self.logrank_: Optional[dict] = None
        self.pairwise_: Optional[pd.DataFrame] = None
        self.km_curves_: Dict[object, PDCurve] = {}
        self.km_tables_: Dict[object, pd.DataFrame] = {}
        self.hazard_lt_: Optional[LifetimePD] = None
        self.ph_: Optional[dict] = None
        self.param_rank_: Optional[pd.DataFrame] = None
        self.param_models_: Dict[object, Dict[str, ParametricSurvival]] = {}
        self.param_curves_: Dict[object, PDCurve] = {}
        self.model_base_: Optional[LifetimePD] = None
        self.model_: Optional[LifetimePD] = None
        self.model_source_: Optional[str] = None
        self._calib_targets: Optional[Dict[object, float]] = None
        self._cycle: Optional[dict] = None
        self.backtest_: Optional[pd.DataFrame] = None
        self.discrimination_: Optional[pd.DataFrame] = None
        self.decile_: Optional[pd.DataFrame] = None
        self.c_index_: Optional[float] = None
        self.val_blocks_: Optional[list] = None
        self.validated_on_: Optional[str] = None
        self.study_ = None
        self.mlflow_run_id_ = None
        self.model_path_ = None

        self._log_lines: list = []
        self._study_steps: list = []
        self._study_secs = None
        self._keepalive = None
        self._suspend_ka = False
        self._fig_slots: dict = {}
        self._calib_inputs: Dict[object, object] = {}

        self._build()
        if df is not None:
            self.set_data(df, **self.cols, features=self._init_features, by=by)
        else:
            self._sync_column_widgets()
            self._refresh_all()

    # ------------------------------------------------------------------ dados
    def set_data(self, df: pd.DataFrame, *, features=None, by=None, **cols):
        """Troca o DataFrame (e, opcionalmente, o mapeamento) e remonta o painel.

        Descarta tudo o que foi calculado sobre os dados anteriores."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df deve ser um pandas.DataFrame no formato longo.")
        self.df = df
        for k, v in cols.items():
            if k in self.cols:
                self.cols[k] = v
        if features is not None:
            self._init_features = list(features)
        if by is not None:
            self._init_by = by
        self._sync_column_widgets()
        self._montar_painel(silencioso=True)
        return self

    def _reset_state(self):
        """Zera tudo o que depende do painel."""
        self.des_ = self.oot_ = None
        self.life_table_ = self.logrank_ = self.pairwise_ = None
        self.km_curves_, self.km_tables_ = {}, {}
        self.hazard_lt_ = self.ph_ = None
        self.param_rank_ = None
        self.param_models_, self.param_curves_ = {}, {}
        self._drop_model()
        self.study_ = None
        self._study_steps, self._study_secs = [], None

    def _drop_model(self):
        """Zera a curva adotada e tudo o que depende dela."""
        self.model_base_ = self.model_ = self.model_source_ = None
        self._calib_targets = self._cycle = None
        self._invalidate_validation()
        self.mlflow_run_id_ = self.model_path_ = None

    def _invalidate_validation(self):
        self.backtest_ = self.discrimination_ = self.decile_ = None
        self.c_index_ = self.val_blocks_ = self.validated_on_ = None

    # -- colunas candidatas --------------------------------------------------
    def _reserved(self) -> set:
        c = self.cols
        r = {c["id_col"], c["date_col"], c["default_col"], c["age_col"], c["origin_col"],
             c["term_col"], c["exposure_col"], ContractPanel.AGE_DERIVED}
        return {x for x in r if x}

    def _numeric_cols(self) -> List[str]:
        if self.df is None:
            return []
        res = self._reserved()
        return [c for c in self.df.columns
                if c not in res and pd.api.types.is_numeric_dtype(self.df[c])
                and not pd.api.types.is_bool_dtype(self.df[c])]

    def _group_cols(self, max_levels: int = 12) -> List[str]:
        if self.df is None:
            return []
        res = {self.cols["id_col"], self.cols["date_col"], self.cols["default_col"],
               self.cols["origin_col"]}
        out = []
        for c in self.df.columns:
            if c in res or c is None:
                continue
            s = self.df[c]
            if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_float_dtype(s):
                continue
            with suppress(Exception):
                if 2 <= s.nunique(dropna=True) <= max_levels:
                    out.append(c)
        return out

    def _all_cols(self) -> List[str]:
        return [] if self.df is None else [str(c) for c in self.df.columns]

    # ==================================================================
    # Utilitários de renderização (o padrão das UIs do credit_risk)
    # ==================================================================
    _DARK_FIG = {"bg": "#1F272D", "ink": "#E8ECF0", "line": "#37444F"}

    @staticmethod
    def _tinta_escura(cor) -> bool:
        import matplotlib.colors as mcolors

        try:
            r, g, b = mcolors.to_rgb(cor)
        except (ValueError, TypeError):
            return False
        sat = max(r, g, b) - min(r, g, b)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        return lum < 0.35 and sat < 0.25

    @staticmethod
    def _tinta_clara(cor) -> bool:
        import matplotlib.colors as mcolors

        try:
            r, g, b, a = mcolors.to_rgba(cor)
        except (ValueError, TypeError):
            return False
        return a > 0 and min(r, g, b) > 0.9 and (max(r, g, b) - min(r, g, b)) < 0.1

    def _dark_fig(self, fig):
        """Repinta a figura para o tema escuro: fundo, tinta, eixos, grades e
        legenda; cores de DADO ficam intactas."""
        bg, ink, line = (self._DARK_FIG[k] for k in ("bg", "ink", "line"))
        fig.patch.set_facecolor(bg)
        for ax in fig.get_axes():
            ax.set_facecolor(bg)
            for sp in ax.spines.values():
                sp.set_color(line)
            ax.tick_params(colors=ink, which="both")
            ax.xaxis.label.set_color(ink)
            ax.yaxis.label.set_color(ink)
            ax.title.set_color(ink)
            for gl in ax.get_xgridlines() + ax.get_ygridlines():
                gl.set_color(line)
            for ln in ax.get_lines():
                if self._tinta_escura(ln.get_color()):
                    ln.set_color(ink)
            for pt in ax.patches:
                if self._tinta_clara(pt.get_edgecolor()):
                    pt.set_edgecolor(bg)
            leg = ax.get_legend()
            if leg is not None:
                leg.get_frame().set_facecolor(bg)
                leg.get_frame().set_edgecolor(line)
                for t in leg.get_texts():
                    if self._tinta_escura(t.get_color()):
                        t.set_color(ink)
            for t in ax.texts:
                if self._tinta_escura(t.get_color()):
                    t.set_color(ink)
        for t in fig.texts:
            if self._tinta_escura(t.get_color()):
                t.set_color(ink)
        return fig

    def _fig_png(self, fig, border=False, tight=True, stretch=False) -> str:
        import base64
        import io as _io

        buf = _io.BytesIO()
        save_kw = {"format": "png", "dpi": min(int(fig.get_dpi()), 110),
                   "facecolor": fig.get_facecolor()}
        if tight:
            save_kw["bbox_inches"] = "tight"
        fig.savefig(buf, **save_kw)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        style = "width:100%;height:auto" if stretch else "max-width:100%;height:auto"
        if border:
            style += ";border:1px solid var(--line);border-radius:6px"
        return f"<img src='data:image/png;base64,{b64}' style='{style}'/>"

    def _fig_html(self, fig, border=False, tight=True, stretch=False, slot=None):
        """Figura → ``<img>`` base64 (fecha a figura). Com ``slot`` (o widget de
        saída), guarda os PNGs claro/escuro para a troca de tema."""
        import matplotlib.pyplot as plt

        kw = {"border": border, "tight": tight, "stretch": stretch}
        claro = self._fig_png(fig, **kw)
        escuro = self._fig_png(self._dark_fig(fig), **kw) if self._dark_on() else None
        plt.close(fig)
        if slot is None:
            return escuro or claro
        self._fig_slots[id(slot)] = {"w": slot, "fig": fig, "kw": kw,
                                     "claro": claro, "escuro": escuro}
        return escuro or claro

    def _dark_on(self) -> bool:
        cb = getattr(self, "cb_dark", None)
        return bool(cb is not None and cb.value)

    def _repinta_figuras(self, dark: bool) -> None:
        for chave, slot in list(self._fig_slots.items()):
            if slot["w"].value not in (slot["claro"], slot["escuro"]):
                del self._fig_slots[chave]
                continue
            try:
                if dark and slot["escuro"] is None:
                    slot["escuro"] = self._fig_png(self._dark_fig(slot["fig"]), **slot["kw"])
                slot["w"].value = slot["escuro"] if dark else slot["claro"]
            except Exception as exc:  # noqa: BLE001 - cosmético, nunca fatal
                self._log(f"[tema] gráfico não repintado ({type(exc).__name__}): {exc}")

    # -- colorações semânticas ------------------------------------------------
    @staticmethod
    def _css_ok(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "color:var(--muted);background-color:var(--neutral-bg)"
        s = str(v).strip().lower()
        if s in ("true", "sim", "ok", "✓"):
            return "color:var(--ok-tx);background-color:var(--ok-bg);font-weight:600"
        if s in ("false", "não", "nao", "✕", "✗"):
            return "color:var(--bad-tx);background-color:var(--bad-bg);font-weight:600"
        return "color:var(--muted);background-color:var(--neutral-bg)"

    @staticmethod
    def _css_pvalor(v):
        try:
            p = float(v)
        except (TypeError, ValueError):
            return ""
        if p != p:
            return ""
        if p <= 0.05:
            return "color:var(--ok-tx);font-weight:600"
        if p <= 0.10:
            return "color:var(--warn-tx);font-weight:600"
        return "color:var(--muted)"

    @staticmethod
    def _css_pvalor_h0(v):
        """p-valor de um teste em que **rejeitar é ruim** (calibração, PH):
        vermelho a 5%, âmbar a 10%, verde acima."""
        try:
            p = float(v)
        except (TypeError, ValueError):
            return ""
        if p != p:
            return ""
        if p <= 0.05:
            return "color:var(--bad-tx);font-weight:600"
        if p <= 0.10:
            return "color:var(--warn-tx);font-weight:600"
        return "color:var(--ok-tx);font-weight:600"

    @staticmethod
    def _css_z(v):
        try:
            z = abs(float(v))
        except (TypeError, ValueError):
            return ""
        if z != z:
            return ""
        if z > 1.96:
            return "color:var(--bad-tx);font-weight:600"
        if z > 1.64:
            return "color:var(--warn-tx);font-weight:600"
        return "color:var(--ok-tx)"

    @staticmethod
    def _css_posicao(v):
        try:
            return ("color:var(--ok-tx);font-weight:600" if int(v) == 1
                    else "color:var(--muted)")
        except (TypeError, ValueError):
            return ""

    def _df_html(self, df, max_height=None, center=False, color_map=None,
                 pct_cols=None, precision=4, fmt_cols=None):
        """Tabela HTML no estilo da casa (tokens de tema, nunca hex)."""
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_bool_dtype(df[c]):
                df[c] = df[c].map({True: "✓", False: "✗"})
        sty = (df.style.hide(axis="index").set_table_styles(self._TABLE_STYLES)
               .set_properties(**{"font-size": "12px"}))
        if center:
            sty = sty.set_properties(**{"text-align": "center"})
            sty = sty.set_table_styles([{"selector": "th, td",
                                         "props": [("text-align", "center")]}],
                                       overwrite=False)
        else:
            txt = [c for c in df.columns if df[c].dtype == object]
            if txt:
                sty = sty.set_properties(subset=txt, **{"text-align": "left"})
        sty = sty.format(na_rep="—", precision=precision)
        if pct_cols:
            present = [c for c in pct_cols if c in df.columns]
            if present:
                sty = sty.format(lambda v: "—" if pd.isna(v) else f"{v * 100:.2f}%",
                                 subset=present)
        for col, fmt in (fmt_cols or {}).items():
            if col in df.columns:
                sty = sty.format(
                    lambda v, _f=fmt: "—" if pd.isna(v) else _f.format(v), subset=[col])
        for col, fn in (color_map or {}).items():
            if col in df.columns:
                sty = sty.map(fn, subset=[col])
        html = sty.to_html()
        if max_height:
            html = f"<div style='max-height:{max_height};overflow:auto'>{html}</div>"
        return html

    @staticmethod
    def _pill(text, cls="muted"):
        return f"<span class='pill pill-{cls}'>{text}</span>"

    def _metric_tiles(self, itens: Mapping) -> str:
        blocos = []
        for k, v in itens.items():
            if isinstance(v, (bool, np.bool_)):
                txt = "sim" if v else "não"
            elif isinstance(v, (int, np.integer)):
                txt = f"{int(v):,}".replace(",", ".")
            elif isinstance(v, (float, np.floating)):
                txt = "—" if not np.isfinite(v) else (f"{v:.4f}" if abs(v) < 1e4 else f"{v:,.1f}")
            else:
                txt = "—" if v is None else _esc(v)
            blocos.append(f"<div class='survui-metric'><div class='k'>{k}</div>"
                          f"<div class='v'>{txt}</div></div>")
        return "<div class='survui-metrics'>" + "".join(blocos) + "</div>"

    def _placar_html(self, blocos) -> str:
        cor = {"ok": "var(--ok-tx)", "warn": "var(--warn-tx)", "bad": "var(--bad-tx)",
               "na": "var(--muted)"}
        icone = {"ok": "✅", "warn": "⚠️", "bad": "❌", "na": "—"}
        cards = []
        for b in blocos:
            n = b["nivel"]
            cls = n if n in ("ok", "warn", "bad") else ""
            cards.append(
                f"<div class='survui-bloco {cls}'><div class='k'>{b['bloco']}</div>"
                f"<div class='v' style='color:{cor[n]}'>{icone[n]} {b['veredito']}</div>"
                f"<div class='d'>{b['detalhe']}</div></div>")
        return "<div class='survui-placar'>" + "".join(cards) + "</div>"

    @staticmethod
    def _notice(msg: str) -> str:
        return f"<div class='survui-notice'>{msg}</div>"

    @staticmethod
    def _ok_msg(msg: str) -> str:
        return f"<div class='survui-legend' style='color:var(--ok-ink)'>✓ {msg}</div>"

    @staticmethod
    def _fmt_pct(v) -> str:
        try:
            return "—" if v is None or not np.isfinite(float(v)) else f"{float(v):.2%}"
        except (TypeError, ValueError):
            return "—"

    @staticmethod
    def _rot(grupo) -> str:
        return "carteira" if grupo == GLOBAL else str(grupo)

    def _log(self, msg):
        self._log_lines.append(str(msg))
        if len(self._log_lines) > 40:
            self._log_lines = self._log_lines[-40:]
        with self.out_log:
            _clear_output(wait=True)
            print("\n".join(self._log_lines))

    def _on_clear_log(self, _):
        self._log_lines = []
        self.out_log.clear_output()

    @contextmanager
    def _busy(self, *botoes, status=None, msg="processando…"):
        busy_html = f"<div class='survui-legend'><i>⏳ {msg}</i></div>"
        for b in botoes:
            b.disabled = True
        if status is not None:
            status.value = busy_html
        try:
            yield
        finally:
            for b in botoes:
                b.disabled = False
            if status is not None and status.value == busy_html:
                status.value = ""

    def _confirm_twice(self, btn, action, timeout=5.0):
        import threading
        import time

        if not hasattr(btn, "_cc_desc"):
            btn._cc_desc = btn.description
            btn._cc_style = btn.button_style
        now = time.monotonic()
        armado = getattr(btn, "_cc_armed", 0.0)
        if armado and now - armado <= timeout:
            btn._cc_armed = 0.0
            btn.description = btn._cc_desc
            btn.button_style = btn._cc_style
            action()
            return
        btn._cc_armed = now
        btn.description = "Confirmar?"
        btn.button_style = "danger"

        def _revert():
            if getattr(btn, "_cc_armed", 0.0) == now:
                btn._cc_armed = 0.0
                btn.description = btn._cc_desc
                btn.button_style = btn._cc_style

        temporizador = threading.Timer(timeout, _revert)
        temporizador.daemon = True
        temporizador.start()

    @staticmethod
    def _fmt_dur(segundos) -> str:
        s = float(segundos)
        if s < 60:
            return f"{s:.1f}s"
        m, r = divmod(int(round(s)), 60)
        return f"{m}min {r:02d}s"

    def _render_progress(self, steps, widget, titulo="Progresso"):
        if not steps:
            widget.value = ""
            return
        icone = {"run": "⏳", "ok": "✅", "err": "❌", "skip": "➖"}
        cor = {"run": "var(--warn-ink)", "ok": "var(--ok-ink)", "err": "var(--bad-ink)",
               "skip": "var(--muted)"}
        rot = {"run": "em andamento…", "ok": "concluída", "err": "erro", "skip": "pulada"}
        trs = ""
        for r in steps:
            st = r.get("status", "run")
            trs += (f"<tr><td>{icone.get(st, '')}</td><td>{r['label']}</td>"
                    f"<td style='color:{cor.get(st, 'var(--ink)')};font-weight:600'>"
                    f"{rot.get(st, st)}</td>"
                    f"<td style='color:var(--muted)'>{r.get('detail', '')}</td></tr>")
        widget.value = (
            f"<div class='survui-legend' style='margin-top:6px'>{titulo}</div>"
            "<table class='survui-prog'><thead><tr><th></th><th>Etapa</th><th>Status</th>"
            f"<th>Detalhe</th></tr></thead><tbody>{trs}</tbody></table>")

    def _prog(self, steps, widget, titulo, key, label, status, detail=""):
        for row in steps:
            if row["key"] == key:
                row["status"] = status
                if detail:
                    row["detail"] = detail
                break
        else:
            steps.append({"key": key, "label": label, "status": status, "detail": detail})
        self._render_progress(steps, widget, titulo)

    def _prog_erro(self, steps, widget, titulo, exc):
        for row in reversed(steps):
            if row.get("status") == "run":
                row["status"] = "err"
                row["detail"] = type(exc).__name__
                break
        self._render_progress(steps, widget, titulo)

    # -- gráficos -----------------------------------------------------------------
    @staticmethod
    def _cores(n: int) -> list:
        from ...reporting.style import COR_PRIMARIA, gradient

        if n <= 1:
            return [COR_PRIMARIA]
        return gradient(n)[::-1]

    def _fig_curves(self, curvas: Mapping, kind: str, tables: Optional[Mapping] = None,
                    titulo: str = "", extras: Optional[Mapping] = None, na: bool = False,
                    junction: Optional[int] = None):
        """Curvas por grupo na representação ``kind``; com ``tables`` (KM),
        desenha a banda de Greenwood; ``extras`` são curvas tracejadas
        (Nelson-Aalen, paramétricas); ``junction`` sombreia a extrapolação."""
        import matplotlib.pyplot as plt

        from ...reporting.style import COR_NEUTRA

        fig, ax = plt.subplots(figsize=(9, 4.6))
        itens = list(curvas.items())
        cores = self._cores(len(itens))
        for (rot, c), cor in zip(itens, cores):
            s = getattr(c, kind)()
            ax.plot(s.index, s.to_numpy(), lw=2, color=cor, label=self._rot(rot))
            tab = (tables or {}).get(rot)
            if tab is not None and kind in ("cumulative", "survival"):
                col = "pd_acumulada" if kind == "cumulative" else "sobrevivencia"
                n = min(len(tab), len(s))
                ax.fill_between(s.index[:n], tab[f"{col}_ic_inf"].to_numpy()[:n],
                                tab[f"{col}_ic_sup"].to_numpy()[:n], color=cor, alpha=0.15)
            if na and tab is not None and kind == "survival":
                n = min(len(tab), len(s))
                if "sobrevivencia_na" in tab.columns:
                    s_na = tab["sobrevivencia_na"].to_numpy()[:n]
                else:
                    n_r = tab["n_em_risco"].to_numpy(dtype=float)[:n]
                    d_r = tab["n_default"].to_numpy(dtype=float)[:n]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        s_na = np.exp(-np.cumsum(np.where(n_r > 0, d_r / n_r, 0.0)))
                ax.plot(s.index[:n], s_na, ls=":", lw=1.4, color=cor,
                        label=f"{self._rot(rot)} (Nelson-Aalen)")
        for rot, c in (extras or {}).items():
            s = getattr(c, kind)()
            ax.plot(s.index, s.to_numpy(), lw=1.5, ls="--", alpha=0.9, label=str(rot))
        if junction is not None:
            ax.axvspan(junction + 1.5, ax.get_xlim()[1], color=COR_NEUTRA, alpha=0.08)
            ax.axvline(junction + 1.5, color=COR_NEUTRA, ls=":", lw=1.2)
            ax.annotate("extrapolação", xy=(junction + 2, ax.get_ylim()[1] * 0.97),
                        fontsize=8, color=COR_NEUTRA, va="top")
        ax.set_xlabel("Horizonte (períodos)")
        ax.set_ylabel(_ROTULO.get(kind, kind))
        ax.set_title(titulo or _ROTULO.get(kind, kind))
        ax.grid(alpha=0.25)
        if kind != "survival":
            ax.set_ylim(bottom=0)
        if len(itens) + len(extras or {}) <= 14:
            ax.legend(fontsize=8, ncol=2)
        return fig

    def _fig_at_risk(self, tab: pd.DataFrame):
        """Base em risco, censura e quebras por idade (as três contagens que
        sustentam a curva) num quadro só."""
        import matplotlib.pyplot as plt

        from ...reporting.style import COR_NEUTRA, COR_PRIMARIA, COR_SECUNDARIA

        fig, ax = plt.subplots(figsize=(9, 4.2))
        x = tab["idade"].to_numpy()
        ax.bar(x, tab["n_em_risco"], color=COR_PRIMARIA, alpha=0.75, label="em risco")
        ax.bar(x, tab["n_censurado"], color=COR_NEUTRA, alpha=0.7, label="censurados na idade")
        ax.set_xlabel("Idade (períodos desde a originação)")
        ax.set_ylabel("Contratos")
        ax.grid(alpha=0.25, axis="y")
        ax2 = ax.twinx()
        ax2.plot(x, tab["hazard"], "o-", color=COR_SECUNDARIA, lw=1.8, ms=3.5,
                 label="hazard observado")
        ax2.set_ylabel("hazard h(t)", color=COR_SECUNDARIA)
        ax2.tick_params(axis="y", labelcolor=COR_SECUNDARIA)
        ax2.set_ylim(bottom=0)
        linhas = ax.containers[:2]
        h1 = [c for c in linhas]
        l1 = [c.get_label() for c in linhas]
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
        ax.set_title("Base em risco, censura e hazard por idade")
        return fig

    # ==================================================================
    # Construção da interface
    # ==================================================================
    def _build(self):
        self.banner = W.HTML()
        self.bar = W.HTML()
        self.out_log = W.Output(layout=W.Layout(max_height="160px", overflow="auto"))
        self.btn_clear_log = W.Button(description="Limpar log", icon="eraser",
                                      layout=W.Layout(width="140px"))
        self.btn_clear_log.on_click(self._on_clear_log)

        self.box_painel = W.VBox(layout=W.Layout(padding="2px"))
        self.box_km = W.VBox(layout=W.Layout(padding="2px"))
        self.box_hazard = W.VBox(layout=W.Layout(padding="2px"))
        self.box_param = W.VBox(layout=W.Layout(padding="2px"))
        self.box_calib = W.VBox(layout=W.Layout(padding="2px"))
        self.box_val = W.VBox(layout=W.Layout(padding="2px"))
        self.box_exportar = W.VBox(layout=W.Layout(padding="2px"))
        self.box_painel.children = self._build_tab_painel()
        self.box_km.children = self._build_tab_km()
        self.box_hazard.children = self._build_tab_hazard()
        self.box_param.children = self._build_tab_param()
        self.box_calib.children = self._build_tab_calib()
        self.box_val.children = self._build_tab_val()
        self.box_exportar.children = self._build_tab_exportar()

        self.tabs = W.Tab(children=[self.box_painel, self.box_km, self.box_hazard,
                                    self.box_param, self.box_calib, self.box_val,
                                    self.box_exportar])
        for i, t in enumerate(self.ABAS):
            self.tabs.set_title(i, t)
        self.tabs.add_class("survui-tabs")
        self.tabs.observe(self._on_tab_change, names="selected_index")

        console = W.VBox([
            W.HBox([W.HTML("<div class='survui-h'>Console</div>"), self.btn_clear_log],
                   layout=W.Layout(justify_content="space-between", align_items="center")),
            self.out_log])
        console.add_class("survui-card")

        self.cb_dark = W.ToggleButton(value=False, description="🌙 Tema escuro",
                                      layout=W.Layout(width="150px"))
        self.cb_dark.observe(self._on_dark, names="value")
        self.cb_keepalive = W.ToggleButton(
            value=False, description="☕ Manter cluster ativo",
            tooltip="Databricks: job Spark mínimo a cada 2 min para o cluster não desligar",
            layout=W.Layout(width="190px"))
        self.cb_keepalive.observe(self._on_keepalive, names="value")
        topbar = W.HBox([W.HTML(""), W.HBox([self.cb_keepalive, self.cb_dark])],
                        layout=W.Layout(justify_content="space-between"))
        self.panel_w = W.VBox([W.HTML(_CSS), topbar, self.banner, self.bar, self.tabs, console])
        self.panel_w.add_class("survui")

    def _on_tab_change(self, change):
        with suppress(Exception):
            idx = change.get("new")
            if idx == 4:
                self._render_calib()
            elif idx == 5:
                self._render_val_notice()
            elif idx == 6:
                self._render_export_estado()

    def _refresh_all(self):
        self._refresh_bar()
        self._render_export_estado()
        self._render_val_notice()

    # ==================================================================
    # Aba Painel
    # ==================================================================
    def _build_tab_painel(self):
        # --- card: dados / painel de referência ------------------------------
        self.sl_ref_n = W.BoundedIntText(value=1500, min=200, max=20000, step=100,
                                         description="contratos:",
                                         style={"description_width": "initial"},
                                         layout=W.Layout(width="170px"))
        self.sl_ref_seed = W.BoundedIntText(value=7, min=0, max=9999, description="semente:",
                                            style={"description_width": "initial"},
                                            layout=W.Layout(width="140px"))
        self.btn_ref = W.Button(description="Carregar painel de referência", icon="flask",
                                button_style="info", layout=W.Layout(width="auto", min_width="250px"),
                                tooltip="Carteira sintética com maturação, covariáveis e censura "
                                        "de processo gerador conhecido.")
        self.btn_ref.on_click(self._on_ref)
        self.out_ref_status = W.HTML()
        card_dados = W.VBox([
            W.HTML("<div class='survui-h'>Dados do estudo</div>"),
            W.HTML("<div class='survui-legend'>A interface trabalha com o <b>painel longo</b> "
                   "de contratos: uma linha por contrato × safra de observação, com a flag de "
                   "<b>entrada em default</b> no período. A idade (<i>months on book</i>) pode "
                   "vir pronta, ser derivada da safra de originação ou da posição na trajetória. "
                   "Sem dados em mãos, carregue o painel de referência: ele é sintético, com "
                   "maturação e censura conhecidas.</div>"),
            W.HBox([self.sl_ref_n, self.sl_ref_seed, self.btn_ref],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_ref_status,
        ])
        card_dados.add_class("survui-card")

        # --- card: mapeamento de colunas -------------------------------------
        def _dd(desc, width="230px"):
            return W.Dropdown(options=[("(nenhuma)", None)], value=None, description=desc,
                              style={"description_width": "initial"},
                              layout=W.Layout(width=width))
        self.dd_col_id = _dd("id do contrato:")
        self.dd_col_date = _dd("safra de observação:")
        self.dd_col_default = _dd("flag de default:")
        self.dd_col_age = _dd("idade (opcional):")
        self.dd_col_origin = _dd("originação (opcional):")
        self.dd_col_term = _dd("prazo remanescente:")
        self.dd_col_segment = _dd("segmento:")
        self.dd_col_exposure = _dd("exposição:")
        self.dd_freq = W.Dropdown(options=[("mensal (M)", "M"), ("trimestral (Q)", "Q"),
                                           ("semestral (S)", "S"), ("anual (A)", "A")],
                                  value="M", description="frequência:",
                                  style={"description_width": "initial"},
                                  layout=W.Layout(width="190px"))
        self.btn_montar = W.Button(description="Montar painel", icon="table",
                                   button_style="primary",
                                   layout=W.Layout(width="auto", min_width="160px"),
                                   tooltip="Valida as colunas, deriva a idade, aplica o default "
                                           "absorvente e reconta a base em risco.")
        self.btn_montar.on_click(lambda b: self._montar_painel())
        self.out_montar_status = W.HTML()
        card_map = W.VBox([
            W.HTML("<div class='survui-h'>Mapeamento das colunas do painel</div>"),
            W.HBox([self.dd_col_id, self.dd_col_date, self.dd_col_default],
                   layout=W.Layout(flex_flow="row wrap")),
            W.HBox([self.dd_col_age, self.dd_col_origin, self.dd_col_term],
                   layout=W.Layout(flex_flow="row wrap")),
            W.HBox([self.dd_col_segment, self.dd_col_exposure, self.dd_freq, self.btn_montar],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HTML("<div class='survui-legend'>A flag de default é o <b>evento</b> (1 no "
                   "período em que o contrato entra em default). Se a base traz o "
                   "<i>estado</i>, o painel reduz ao evento descartando as observações "
                   "posteriores ao primeiro default (<i>default absorvente</i>).</div>"),
            self.out_montar_status,
        ])
        card_map.add_class("survui-card")

        # --- card: o painel ------------------------------------------------------
        self.out_panel_head = W.HTML()
        self.out_panel_plot = W.HTML()
        self.out_panel_table = W.HTML()
        card_head = W.VBox([W.HTML("<div class='survui-h'>O painel</div>"),
                            self.out_panel_head, self.out_panel_plot,
                            W.HTML("<div class='survui-legend'>Tabela de vida da carteira "
                                   "(primeiras idades): a base em risco é <b>recontada idade a "
                                   "idade</b>, e é isso que trata a censura à direita: um "
                                   "contrato jovem entra nas idades baixas e simplesmente não "
                                   "aparece nas altas.</div>"),
                            self.out_panel_table])
        card_head.add_class("survui-card")

        # --- card: safra × idade ---------------------------------------------------
        self.dd_cohort = W.Dropdown(options=[("trimestre", "Q"), ("mês", "M"), ("ano", "A")],
                                    value="Q", description="coorte por:",
                                    style={"description_width": "initial"},
                                    layout=W.Layout(width="190px"))
        self.btn_heat = W.Button(description="Desenhar mapa safra × idade", icon="th",
                                 layout=W.Layout(width="auto", min_width="230px"))
        self.btn_heat.on_click(self._on_heat)
        self.out_heat = W.HTML()
        card_heat = W.VBox([
            W.HTML("<div class='survui-h'>Safra × idade</div>"),
            W.HTML("<div class='survui-legend'>PD acumulada por safra de originação (linhas) e "
                   "idade (colunas): a <b>maturação</b> é andar para a direita; a <b>qualidade "
                   "da safra</b> é mudar de linha. Um bloco de safras mais escuro na mesma "
                   "idade é deterioração de originação, não de ciclo.</div>"),
            W.HBox([self.dd_cohort, self.btn_heat], layout=W.Layout(align_items="center")),
            self.out_heat,
        ])
        card_heat.add_class("survui-card")

        # --- card: partição --------------------------------------------------------
        self.dd_split = W.Dropdown(
            options=[("nenhuma (tudo é DES)", "none"),
                     ("por safra de originação (contratos novos = OOT)", "origin"),
                     ("por data de observação (janela posterior = OOT)", "observation"),
                     ("por coluna de amostra", "column")],
            value="none", description="partição:", style={"description_width": "initial"},
            layout=W.Layout(width="420px"))
        self.dd_split.observe(lambda c: self._sync_split_fields(), names="value")
        self.tx_split_date = W.Text(value="", placeholder="AAAA-MM-DD", description="corte:",
                                    style={"description_width": "initial"},
                                    layout=W.Layout(width="200px"))
        self.dd_split_col = _dd("coluna:", "200px")
        self.tx_split_oot = W.Text(value="OOT", description="valor OOT:",
                                   style={"description_width": "initial"},
                                   layout=W.Layout(width="160px"))
        self.btn_split = W.Button(description="Particionar", icon="scissors",
                                  button_style="primary",
                                  layout=W.Layout(width="auto", min_width="140px"))
        self.btn_split.on_click(self._on_split)
        self.out_split_status = W.HTML()
        self.out_split_tiles = W.HTML()
        card_split = W.VBox([
            W.HTML("<div class='survui-h'>Partição DES / OOT</div>"),
            W.HTML("<div class='survui-help'><div class='ttl'>Por que particionar</div>"
                   "A curva é ajustada no <b>DES</b> e validada no <b>OOT</b>. Por <b>safra de "
                   "originação</b> é a validação mais exigente (contratos que o ajuste nunca "
                   "viu, com toda a sua maturação); por <b>data de observação</b> testa a "
                   "curva a partir das idades em que a carteira viva está. Sem partição a "
                   "validação roda no próprio DES e a aba <i>Validação</i> avisa que é "
                   "<i>in-sample</i>.</div>"),
            W.HBox([self.dd_split, self.tx_split_date, self.dd_split_col, self.tx_split_oot,
                    self.btn_split],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_split_status, self.out_split_tiles,
        ])
        card_split.add_class("survui-card")

        # --- card: estudo completo -------------------------------------------------
        self.dd_study_method = W.Dropdown(options=list(METODOS_ESTUDO), value="km",
                                          description="motor:",
                                          style={"description_width": "initial"},
                                          layout=W.Layout(width="360px"))
        self.tx_nome = W.Text(value=self.study_name, description="nome do estudo:",
                              style={"description_width": "initial"},
                              layout=W.Layout(width="330px"))
        self.btn_run_study = W.Button(description="Rodar estudo completo", icon="rocket",
                                      button_style="success",
                                      layout=W.Layout(width="auto", min_width="210px"),
                                      tooltip="Partição → curva → cauda → calibração/ciclo → "
                                              "validação, com a configuração da tela.")
        self.btn_run_study.on_click(self._on_run_study)
        self.out_study_status = W.HTML()
        self.out_study_progress = W.HTML()
        self.out_study_resumo = W.HTML()
        card_study = W.VBox([
            W.HTML("<div class='survui-h'>Estudo completo em um clique</div>"),
            W.HTML("<div class='survui-legend'>Monta a <code>SurvivalConfig</code> a partir de "
                   "todos os controles das abas (grupo, horizonte, features, família da cauda, "
                   "calibração, ciclo, horizontes de validação) e encadeia as etapas, "
                   "preenchendo cada aba com o resultado. As abas continuam servindo para "
                   "conduzir o estudo passo a passo, e para discordar de qualquer etapa."
                   "</div>"),
            W.HBox([self.dd_study_method, self.tx_nome, self.btn_run_study],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_study_status, self.out_study_progress, self.out_study_resumo,
        ])
        card_study.add_class("survui-card")

        self._sync_split_fields()
        return (card_dados, card_map, card_head, card_heat, card_split, card_study)

    # ------------------------------------------------------------------ colunas
    def _sync_column_widgets(self):
        """Repõe as opções dos dropdowns de coluna com as colunas do DataFrame."""
        cols = self._all_cols()
        opcoes = [("(nenhuma)", None)] + [(c, c) for c in cols]
        pares = (("dd_col_id", "id_col"), ("dd_col_date", "date_col"),
                 ("dd_col_default", "default_col"), ("dd_col_age", "age_col"),
                 ("dd_col_origin", "origin_col"), ("dd_col_term", "term_col"),
                 ("dd_col_segment", "segment_col"), ("dd_col_exposure", "exposure_col"))
        for w, k in pares:
            dd = getattr(self, w)
            dd.options = opcoes
            v = self.cols.get(k)
            dd.value = v if v in cols else None
        self.dd_split_col.options = opcoes
        with suppress(Exception):
            self.dd_freq.value = str(self.cols.get("freq") or "M").upper()[:1]

    def _cols_from_widgets(self) -> dict:
        return dict(id_col=self.dd_col_id.value, date_col=self.dd_col_date.value,
                    default_col=self.dd_col_default.value, age_col=self.dd_col_age.value,
                    origin_col=self.dd_col_origin.value, term_col=self.dd_col_term.value,
                    segment_col=self.dd_col_segment.value,
                    exposure_col=self.dd_col_exposure.value, freq=self.dd_freq.value)

    def _montar_painel(self, silencioso: bool = False):
        """Monta o :class:`ContractPanel` com o mapeamento da tela."""
        if self.df is None:
            self.out_montar_status.value = self._notice(
                "Sem dados: passe o DataFrame no construtor, use <code>ui.set_data(df)</code> "
                "ou carregue o painel de referência.")
            return
        cols = self._cols_from_widgets()
        faltando = [k for k in ("id_col", "date_col", "default_col") if not cols[k]]
        if faltando:
            self.out_montar_status.value = self._notice(
                "Informe as colunas obrigatórias: <b>id do contrato</b>, <b>safra de "
                "observação</b> e <b>flag de default</b>.")
            return
        with self._busy(self.btn_montar, status=self.out_montar_status, msg="montando o painel…"):
            try:
                painel = ContractPanel(self.df, **cols)
            except Exception as exc:  # noqa: BLE001
                self.out_montar_status.value = self._notice(
                    f"Não foi possível montar o painel: {_esc(exc)}")
                self._log(f"[painel] ERRO: {type(exc).__name__}: {exc}")
                return
        self.cols.update(cols)
        self.panel = painel
        self._reset_state()
        self.des_ = painel
        self._sync_data_widgets()
        self._render_panel()
        self._clear_km_outputs()
        self._clear_hazard_outputs()
        self._clear_param_outputs()
        self._clear_val_outputs()
        self._clear_exportar_outputs()
        self.out_heat.value = ""
        self.out_split_tiles.value = ""
        self.out_split_status.value = ""
        self._render_calib()
        self._refresh_all()
        s = painel.summary().iloc[0]
        self.out_montar_status.value = self._ok_msg(
            f"Painel montado: {painel.n_contracts:,} contratos, {len(painel):,} observações, "
            f"{int(s['n_defaults']):,} defaults, idade máxima {painel.max_age}"
            + (f", {int(s['obs_pos_default_descartadas']):,} observação(ões) pós-default "
               "descartada(s)" if s["obs_pos_default_descartadas"] else "") + "."
        ).replace(",", ".")
        if not silencioso:
            self._log(f"[painel] montado: {painel!r}")

    def _sync_data_widgets(self):
        """Repõe as opções que dependem do painel (grupos, features, colunas)."""
        grupos = [("(carteira inteira)", None)] + [(c, c) for c in self._group_cols()]
        by0 = self._init_by if self._init_by in self._group_cols() else (
            self.cols["segment_col"] if self.cols["segment_col"] in self._group_cols() else None)
        for dd in (self.dd_km_by, self.dd_hz_by, self.dd_par_by):
            dd.options = grupos
            dd.value = by0
        nums = self._numeric_cols()
        self.sel_hz_features.options = [(c, c) for c in nums]
        self.sel_hz_features.value = tuple(c for c in self._init_features if c in nums)
        topo = max(int(self.panel.max_age) + 1, 1) if self.panel is not None else 60
        for w in (self.sl_km_horizon, self.sl_hz_horizon, self.sl_par_horizon):
            w.value = int(self._init_horizon)
        self.sl_par_junction.max = max(topo, 1)
        self.cb_km_weighted.disabled = self.cols.get("exposure_col") is None
        if self.cb_km_weighted.disabled:
            self.cb_km_weighted.value = False
        if self.panel is not None and not self.tx_split_date.value:
            self._suggest_split_date()

    def _suggest_split_date(self):
        """Sugere a data de corte: o último terço das **originações** (modo por
        safra) ou das **observações** (modo por janela) vira OOT."""
        if self.panel is None:
            return
        d = self.panel.df
        if self.dd_split.value == "origin":
            if self.panel.origin_col and self.panel.origin_col in d.columns:
                datas = pd.to_datetime(d[self.panel.origin_col])
            else:
                datas = pd.to_datetime(d.groupby(self.panel.id_col, sort=False)[self.panel.date_col]
                                       .transform("min"))
        else:
            datas = pd.to_datetime(d[self.panel.date_col])
        corte = pd.Timestamp(datas.quantile(0.67)).normalize()
        self.tx_split_date.value = str(corte.date())

    # ------------------------------------------------------------------ referência
    def _on_ref(self, b):
        if self.panel is not None:
            self._confirm_twice(self.btn_ref, self._load_reference)
            return
        self._load_reference()

    def _load_reference(self):
        with self._busy(self.btn_ref, status=self.out_ref_status, msg="gerando o painel…"):
            try:
                ref = make_reference_panel(n_contracts=int(self.sl_ref_n.value),
                                           seed=int(self.sl_ref_seed.value))
                self.tx_split_date.value = ""
                self.set_data(ref.df, id_col="id_contrato", date_col="dt_ref",
                              default_col="default", age_col=None, origin_col="safra_origem",
                              term_col="prazo", segment_col="produto", exposure_col="exposicao",
                              freq="M", features=list(FEATURES), by="produto")
            except Exception as exc:  # noqa: BLE001
                self.out_ref_status.value = self._notice(
                    f"Não foi possível gerar o painel: {_esc(exc)}")
                self._log(f"[dados] erro ao gerar o painel de referência: {exc}")
                return
        self.out_ref_status.value = self._ok_msg(
            f"Painel de referência carregado: {self.panel.n_contracts} contratos, "
            f"{len(self.panel)} observações, maturação Weibull (k = 1,35), 3 covariáveis, "
            "2 produtos e 3 ratings.")
        self._log(f"[dados] painel de referência carregado ({self.panel!r}).")

    # ------------------------------------------------------------------ o painel
    def _render_panel(self):
        if self.panel is None:
            self.out_panel_head.value = self._notice("Nenhum painel montado.")
            self.out_panel_plot.value = ""
            self.out_panel_table.value = ""
            return
        p = self.panel
        s = p.summary().iloc[0]
        tiles = {
            "contratos": int(p.n_contracts), "observações": int(len(p)),
            "defaults": int(s["n_defaults"]),
            "taxa por contrato": self._fmt_pct(s["taxa_default_contratos"]),
            "idade máxima": int(p.max_age),
            "safras": f"{str(s['safra_min'])[:7]} → {str(s['safra_max'])[:7]}",
            "segmentos": int(s["n_segmentos"]),
            "pós-default descartadas": int(s["obs_pos_default_descartadas"]),
            "frequência": str(p.freq),
        }
        self.out_panel_head.value = self._metric_tiles(tiles)
        try:
            tab = life_table(p)
            fig = self._fig_at_risk(tab)
            self.out_panel_plot.value = self._fig_html(fig, slot=self.out_panel_plot)
            cols = ["idade", "n_em_risco", "n_default", "n_censurado", "hazard",
                    "sobrevivencia", "pd_acumulada", "hazard_acumulado"]
            self.out_panel_table.value = self._df_html(
                tab[cols], max_height="260px", pct_cols=["hazard", "pd_acumulada"],
                fmt_cols={"sobrevivencia": "{:.4f}", "hazard_acumulado": "{:.4f}"})
        except Exception as exc:  # noqa: BLE001
            self.out_panel_plot.value = self._notice(f"Não foi possível desenhar: {_esc(exc)}")
            self.out_panel_table.value = ""

    def _on_heat(self, b):
        if self.panel is None:
            self.out_heat.value = self._notice("Monte o painel antes.")
            return
        with self._busy(self.btn_heat, status=self.out_heat, msg="desenhando…"):
            try:
                from ..ecl.report import plot_vintage_heatmap

                fig = plot_vintage_heatmap(self.panel, cohort_freq=self.dd_cohort.value)
                self.out_heat.value = self._fig_html(fig, slot=self.out_heat)
            except Exception as exc:  # noqa: BLE001
                self.out_heat.value = self._notice(f"Não foi possível desenhar: {_esc(exc)}")

    # ------------------------------------------------------------------ partição
    def _sync_split_fields(self):
        modo = self.dd_split.value
        self.tx_split_date.layout.display = None if modo in ("origin", "observation") else "none"
        if modo in ("origin", "observation") and getattr(self, "panel", None) is not None:
            self._suggest_split_date()
        self.dd_split_col.layout.display = None if modo == "column" else "none"
        self.tx_split_oot.layout.display = None if modo == "column" else "none"

    def _on_split(self, b):
        if self.panel is None:
            self.out_split_status.value = self._notice("Monte o painel antes.")
            return
        modo = self.dd_split.value
        with self._busy(self.btn_split, status=self.out_split_status, msg="particionando…"):
            try:
                des, oot = split_panel(self.panel, mode=modo,
                                       value=(self.tx_split_date.value or None),
                                       column=self.dd_split_col.value,
                                       oot_value=self.tx_split_oot.value)
            except Exception as exc:  # noqa: BLE001
                self.out_split_status.value = self._notice(f"Não foi possível particionar: {_esc(exc)}")
                return
        mudou = (self.des_ is not des)
        self.des_, self.oot_ = des, oot
        if mudou:
            # a curva foi ajustada em outro DES: tudo o que veio dela cai
            self.life_table_ = self.logrank_ = self.pairwise_ = None
            self.km_curves_, self.km_tables_ = {}, {}
            self.hazard_lt_ = self.ph_ = None
            self.param_rank_, self.param_models_, self.param_curves_ = None, {}, {}
            self._drop_model()
            self.study_ = None
            self._clear_km_outputs()
            self._clear_hazard_outputs()
            self._clear_param_outputs()
            self._clear_val_outputs()
            self._clear_exportar_outputs()
            self._render_calib()
        self._render_split()
        self._refresh_all()
        self._log(f"[partição] modo={modo}: DES {des.n_contracts} contratos"
                  + (f", OOT {oot.n_contracts} contratos" if oot is not None else ", sem OOT"))

    def _render_split(self):
        if self.des_ is None:
            self.out_split_tiles.value = ""
            return
        if self.oot_ is None:
            self.out_split_status.value = self._notice(
                "Sem OOT (nenhuma observação caiu do lado OOT com este corte, ou a partição é "
                "'nenhuma'): a validação rodará no próprio DES (<i>in-sample</i>).")
            self.out_split_tiles.value = self._metric_tiles({
                "DES · contratos": self.des_.n_contracts, "DES · observações": len(self.des_),
                "DES · defaults": int(self.des_.df[self.des_.default_col].sum())})
            return
        self.out_split_status.value = self._ok_msg("Partição aplicada. A validação usará o OOT.")
        self.out_split_tiles.value = self._metric_tiles({
            "DES · contratos": self.des_.n_contracts, "DES · observações": len(self.des_),
            "DES · defaults": int(self.des_.df[self.des_.default_col].sum()),
            "OOT · contratos": self.oot_.n_contracts, "OOT · observações": len(self.oot_),
            "OOT · defaults": int(self.oot_.df[self.oot_.default_col].sum()),
            "OOT · idade mín.": int(self.oot_.df[self.oot_.age_col].min()),
        })

    # ==================================================================
    # Aba Kaplan-Meier
    # ==================================================================
    def _build_tab_km(self):
        self.dd_km_by = W.Dropdown(options=[("(carteira inteira)", None)], value=None,
                                   description="agrupar por:",
                                   style={"description_width": "initial"},
                                   layout=W.Layout(width="260px"))
        self.dd_km_method = W.Dropdown(options=[("Kaplan-Meier", "km"), ("Safra / vintage", "vintage")],
                                       value="km", description="motor:",
                                       style={"description_width": "initial"},
                                       layout=W.Layout(width="230px"))
        self.dd_km_method.observe(lambda c: self._sync_km_fields(), names="value")
        self.dd_km_kind = W.Dropdown(options=list(KINDS), value="cumulative",
                                     description="representação:",
                                     style={"description_width": "initial"},
                                     layout=W.Layout(width="270px"))
        self.dd_km_kind.observe(lambda c: self._render_km_plot(), names="value")
        self.sl_km_horizon = W.BoundedIntText(value=60, min=1, max=600, description="horizonte:",
                                              style={"description_width": "initial"},
                                              layout=W.Layout(width="150px"))
        self.sl_km_from = W.BoundedIntText(value=0, min=0, max=600, description="idade inicial:",
                                           style={"description_width": "initial"},
                                           layout=W.Layout(width="160px"))
        self.fl_km_alpha = W.BoundedFloatText(value=0.05, min=0.001, max=0.5, step=0.01,
                                              description="α do IC:",
                                              style={"description_width": "initial"},
                                              layout=W.Layout(width="140px"))
        self.sl_km_min_risk = W.BoundedIntText(value=30, min=1, max=100000,
                                               description="base mínima (vintage):",
                                               style={"description_width": "initial"},
                                               layout=W.Layout(width="220px"))
        self.cb_km_weighted = W.Checkbox(value=False, indent=False,
                                         description="ponderar por exposição")
        self.cb_km_ci = W.Checkbox(value=True, indent=False, description="banda de Greenwood")
        self.cb_km_ci.observe(lambda c: self._render_km_plot(), names="value")
        self.cb_km_na = W.Checkbox(value=False, indent=False, description="Nelson-Aalen")
        self.cb_km_na.observe(lambda c: self._render_km_plot(), names="value")
        self.btn_km = W.Button(description="Estimar curvas", icon="line-chart",
                               button_style="primary",
                               layout=W.Layout(width="auto", min_width="170px"))
        self.btn_km.on_click(self._on_km)
        self.btn_km_adopt = W.Button(description="Adotar como curva do estudo", icon="check",
                                     button_style="success",
                                     layout=W.Layout(width="auto", min_width="240px"),
                                     tooltip="A(s) curva(s) desta aba passam a ser a curva do "
                                             "estudo (cauda plana até o horizonte; a aba "
                                             "Paramétrico troca a cauda).")
        self.btn_km_adopt.on_click(self._on_km_adopt)
        self.out_km_status = W.HTML()
        self.out_km_plot = W.HTML()
        self.out_km_summary = W.HTML()
        self.out_km_table = W.HTML()
        self.out_km_logrank = W.HTML()
        self.out_km_pairwise = W.HTML()

        card_ctrl = W.VBox([
            W.HTML("<div class='survui-h'>Curvas não paramétricas</div>"),
            W.HTML("<div class='survui-help'><div class='ttl'>Como ler</div>"
                   "<b>Kaplan-Meier</b> e a <b>curva de safra</b> coincidem no ponto estimado "
                   "(em tempo discreto ambos são o produto-limite <code>Π(1 − d_t/n_t)</code>); o "
                   "KM acrescenta o erro padrão de <b>Greenwood</b> e o IC log-log, a safra "
                   "acrescenta a <b>base mínima</b> por idade (idades com pouca base repetem o "
                   "último hazard válido). <b>Nelson-Aalen</b> estima o hazard acumulado "
                   "<code>Σ d_t/n_t</code>; a sobrevivência <code>exp(−H)</code> fica sempre um "
                   "pouco acima do KM e as duas se afastam onde a base rareia.</div>"),
            W.HBox([self.dd_km_method, self.dd_km_by, self.dd_km_kind],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.sl_km_horizon, self.sl_km_from, self.fl_km_alpha, self.sl_km_min_risk],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.cb_km_weighted, self.cb_km_ci, self.cb_km_na, self.btn_km,
                    self.btn_km_adopt],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_km_status,
        ])
        card_ctrl.add_class("survui-card")
        card_plot = W.VBox([W.HTML("<div class='survui-h'>As curvas</div>"), self.out_km_plot,
                            self.out_km_summary])
        card_plot.add_class("survui-card")
        card_lr = W.VBox([
            W.HTML("<div class='survui-h'>Os grupos diferem? (log-rank)</div>"),
            W.HTML("<div class='survui-legend'>H0: as curvas de sobrevivência dos grupos são "
                   "iguais. Rejeitar (p pequeno) é o argumento para curvas <b>por segmento</b>; "
                   "não rejeitar sugere uma curva única (ou fundir os pares que não se "
                   "distinguem). A razão <b>obs/esp</b> é a leitura de risco relativo de cada "
                   "grupo.</div>"),
            self.out_km_logrank, self.out_km_pairwise,
        ])
        card_lr.add_class("survui-card")
        card_tab = W.VBox([W.HTML("<div class='survui-h'>Tabela de vida</div>"), self.out_km_table])
        card_tab.add_class("survui-card")
        self._sync_km_fields()
        return (card_ctrl, card_plot, card_lr, card_tab)

    def _sync_km_fields(self):
        self.sl_km_min_risk.layout.display = None if self.dd_km_method.value == "vintage" else "none"

    def _clear_km_outputs(self):
        for w in ("out_km_status", "out_km_plot", "out_km_summary", "out_km_table",
                  "out_km_logrank", "out_km_pairwise"):
            getattr(self, w).value = ""

    def _on_km(self, b):
        if self.des_ is None:
            self.out_km_status.value = self._notice("Monte o painel na aba <b>Painel</b> antes.")
            return
        by = self.dd_km_by.value
        with self._busy(self.btn_km, self.btn_km_adopt, status=self.out_km_status,
                        msg="estimando…"):
            try:
                kw = dict(from_age=int(self.sl_km_from.value), alpha=float(self.fl_km_alpha.value),
                          weighted=bool(self.cb_km_weighted.value))
                self.life_table_ = life_table(self.des_, by=by, **kw)
                partes = {GLOBAL: self.des_} if by is None else self.des_.by(by)
                curvas, tabelas = {}, {}
                for rot, parte in partes.items():
                    rotulo = "" if rot == GLOBAL else str(rot)
                    if self.dd_km_method.value == "vintage":
                        c, t = vintage_curve(parte, label=rotulo, return_table=True,
                                             min_at_risk=int(self.sl_km_min_risk.value),
                                             fill="ffill", **kw)
                        _, tkm = kaplan_meier(parte, label=rotulo, return_table=True, **kw)
                        tabelas[rot] = tkm
                    else:
                        c, t = kaplan_meier(parte, label=rotulo, return_table=True, **kw)
                        tabelas[rot] = t
                    curvas[rot] = c
                self.km_curves_, self.km_tables_ = curvas, tabelas
                self.logrank_ = self.pairwise_ = None
                if by is not None and len(partes) >= 2 and not self.cb_km_weighted.value:
                    self.logrank_ = logrank_test(self.des_, by, from_age=int(self.sl_km_from.value))
                    if len(partes) > 2:
                        self.pairwise_ = pairwise_logrank(self.des_, by,
                                                          from_age=int(self.sl_km_from.value))
            except Exception as exc:  # noqa: BLE001
                self.out_km_status.value = self._notice(f"Não foi possível estimar: {_esc(exc)}")
                self._log(f"[km] ERRO: {type(exc).__name__}: {exc}")
                return
        self._render_km()
        n = len(self.km_curves_)
        self.out_km_status.value = self._ok_msg(
            f"{n} curva(s) estimada(s) com {self.dd_km_method.value}"
            + (f", agrupadas por <b>{_esc(by)}</b>" if by else "")
            + f". Idade máxima observada: {self.des_.max_age}.")
        self._log(f"[km] {n} curva(s) ({self.dd_km_method.value}, by={by}).")

    def _km_summary_table(self) -> pd.DataFrame:
        n_ano = self.des_.periods_per_year
        linhas = []
        for rot, c in self.km_curves_.items():
            t = self.km_tables_.get(rot)
            linhas.append({
                "grupo": self._rot(rot),
                "n_contratos": int(t["n_em_risco"].iloc[0]) if t is not None else np.nan,
                "n_defaults": int(t["n_default"].sum()) if t is not None else np.nan,
                "n_censurados": int(t["n_censurado"].sum()) if t is not None else np.nan,
                "idade_max": int(len(c)),
                "pd_12m": c.pd_12m(),
                f"pd_{min(2 * n_ano, len(c))}p": c.pd_lifetime(min(2 * n_ano, len(c))),
                "pd_obs_max": c.pd_lifetime(),
                "mediana": median_survival(c),
                "rmst": restricted_mean_survival(c),
            })
        return pd.DataFrame(linhas)

    def _render_km(self):
        if not self.km_curves_:
            return
        self._render_km_plot()
        tab = self._km_summary_table()
        pct = [c for c in tab.columns if c.startswith("pd_")]
        self.out_km_summary.value = (
            "<div class='survui-legend'><b>pd_obs_max</b> é a PD acumulada até a última idade "
            "observada (não é a PD lifetime do contrato: a cauda entra na aba <i>Paramétrico</i>); "
            "<b>mediana</b> = período em que S(t) cruza 50% (vazio se não cruza); <b>rmst</b> = "
            "tempo médio de sobrevivência restrito ao horizonte observado, em períodos.</div>"
            + self._df_html(tab, pct_cols=pct, fmt_cols={"mediana": "{:.0f}", "rmst": "{:.1f}"}))
        lt = self.life_table_
        if lt is not None:
            cols = [c for c in ("grupo", "idade", "n_em_risco", "n_default", "n_censurado", "hazard",
                                "sobrevivencia", "se_greenwood", "sobrevivencia_ic_inf",
                                "sobrevivencia_ic_sup", "pd_acumulada", "hazard_acumulado",
                                "sobrevivencia_na") if c in lt.columns]
            self.out_km_table.value = self._df_html(
                lt[cols], max_height="320px", pct_cols=["hazard", "pd_acumulada"], precision=4)
        if self.logrank_ is None:
            self.out_km_logrank.value = (
                "<div class='survui-legend'>Agrupe por uma coluna com dois ou mais níveis (sem "
                "ponderação por exposição) para rodar o log-rank.</div>")
            self.out_km_pairwise.value = ""
            return
        r = self.logrank_
        rejeita = r["p_valor"] <= float(self.fl_km_alpha.value)
        nivel = "ok" if rejeita else "warn"
        leitura = ("as curvas <b>diferem</b>: vale manter curvas por grupo."
                   if rejeita else "as curvas <b>não se distinguem</b> a este nível: uma curva "
                   "única (ou a fusão de grupos) é defensável.")
        blocos = [{"bloco": "Log-rank (Mantel-Cox)", "nivel": nivel,
                   "veredito": "rejeita H0" if rejeita else "não rejeita H0",
                   "detalhe": f"χ² = {r['estatistica']:.2f} · gl = {r['gl']} · "
                              f"p = {r['p_valor']:.4f}"}]
        g = r["grupos"].copy()
        g["grupo"] = g["grupo"].map(self._rot)
        self.out_km_logrank.value = (
            self._placar_html(blocos)
            + f"<div class='survui-legend'>Leitura: {leitura}</div>"
            + self._df_html(g, fmt_cols={"esperados": "{:.1f}", "obs_esp": "{:.2f}"}))
        if self.pairwise_ is not None:
            self.out_km_pairwise.value = (
                "<div class='survui-legend' style='margin-top:8px'>Par a par (Bonferroni): "
                "pares com <b>p</b> alto são candidatos à fusão.</div>"
                + self._df_html(self.pairwise_, color_map={"p_valor": self._css_pvalor,
                                                           "p_bonferroni": self._css_pvalor}))
        else:
            self.out_km_pairwise.value = ""

    def _render_km_plot(self):
        if not self.km_curves_:
            return
        kind = self.dd_km_kind.value
        try:
            fig = self._fig_curves(self.km_curves_, kind,
                                   tables=self.km_tables_ if self.cb_km_ci.value else None,
                                   na=bool(self.cb_km_na.value),
                                   titulo=f"{_ROTULO[kind]} · {self.dd_km_method.value}"
                                          + (f" por {self.dd_km_by.value}" if self.dd_km_by.value else ""))
            self.out_km_plot.value = self._fig_html(fig, slot=self.out_km_plot)
        except Exception as exc:  # noqa: BLE001
            self.out_km_plot.value = self._notice(f"Não foi possível desenhar: {_esc(exc)}")

    def _on_km_adopt(self, b):
        if not self.km_curves_:
            self.out_km_status.value = self._notice("Estime as curvas antes de adotar.")
            return
        H = int(self.sl_km_horizon.value)
        curvas = {rot: c.extend(H) for rot, c in self.km_curves_.items()}
        lt = LifetimePD.from_curves(curvas, method=self.dd_km_method.value)
        lt.by = self.dd_km_by.value
        lt.freq = self.des_.freq
        lt.tables_ = dict(self.km_tables_)
        lt.meta.update({"metodo": self.dd_km_method.value, "by": lt.by, "horizon": H,
                        "tail": "flat", "n_contratos": self.des_.n_contracts})
        self._adopt(lt, self.dd_km_method.value)
        self.out_km_status.value = self._ok_msg(
            f"Curva do estudo adotada ({self.dd_km_method.value}, {len(curvas)} curva(s), "
            f"horizonte {H}, cauda plana além da idade {self.des_.max_age}). Troque a cauda na "
            "aba <b>Paramétrico</b>; calibre e valide nas abas seguintes.")

    # ==================================================================
    # Aba Hazard
    # ==================================================================
    def _build_tab_hazard(self):
        self.sel_hz_features = W.SelectMultiple(options=[], rows=7, description="features:",
                                                style={"description_width": "initial"},
                                                layout=W.Layout(width="330px"))
        self.dd_hz_by = W.Dropdown(options=[("(carteira inteira)", None)], value=None,
                                   description="modelo por grupo:",
                                   style={"description_width": "initial"},
                                   layout=W.Layout(width="270px"))
        self.dd_hz_baseline = W.Dropdown(
            options=[("spline (B-spline em idade)", "spline"), ("dummies (uma por idade)", "dummies"),
                     ("linear", "linear"), ("log(1 + idade)", "log")],
            value="spline", description="baseline:", style={"description_width": "initial"},
            layout=W.Layout(width="290px"))
        self.dd_hz_link = W.Dropdown(options=[("logit", "logit"), ("cloglog (Cox discreto; statsmodels)", "cloglog")],
                                     value="logit", description="link:",
                                     style={"description_width": "initial"},
                                     layout=W.Layout(width="290px"))
        self.fl_hz_C = W.BoundedFloatText(value=1e6, min=1e-4, max=1e12, description="C (1/L2):",
                                          style={"description_width": "initial"},
                                          layout=W.Layout(width="170px"))
        self.sl_hz_knots = W.BoundedIntText(value=6, min=3, max=30, description="nós do spline:",
                                            style={"description_width": "initial"},
                                            layout=W.Layout(width="170px"))
        self.sl_hz_max_age = W.BoundedIntText(value=0, min=0, max=600,
                                              description="idade máx. (0 = todas):",
                                              style={"description_width": "initial"},
                                              layout=W.Layout(width="210px"))
        self.sl_hz_horizon = W.BoundedIntText(value=60, min=1, max=600, description="horizonte:",
                                              style={"description_width": "initial"},
                                              layout=W.Layout(width="150px"))
        self.btn_hz = W.Button(description="Ajustar hazard", icon="cogs", button_style="primary",
                               layout=W.Layout(width="auto", min_width="160px"))
        self.btn_hz.on_click(self._on_hz)
        self.btn_hz_ph = W.Button(description="Testar riscos proporcionais", icon="balance-scale",
                                  layout=W.Layout(width="auto", min_width="230px"))
        self.btn_hz_ph.on_click(self._on_hz_ph)
        self.btn_hz_adopt = W.Button(description="Adotar como curva do estudo", icon="check",
                                     button_style="success",
                                     layout=W.Layout(width="auto", min_width="240px"))
        self.btn_hz_adopt.on_click(self._on_hz_adopt)
        self.out_hz_status = W.HTML()
        self.out_hz_metrics = W.HTML()
        self.out_hz_coef = W.HTML()
        self.out_hz_plot_base = W.HTML(layout=W.Layout(width="49%"))
        self.out_hz_plot_prof = W.HTML(layout=W.Layout(width="49%"))
        self.out_hz_ph = W.HTML()

        card_ctrl = W.VBox([
            W.HTML("<div class='survui-h'>Regressão de hazard em tempo discreto</div>"),
            W.HTML("<div class='survui-help'><div class='ttl'>O que este modelo faz</div>"
                   "Expande o painel em <b>pessoa-período</b> e ajusta "
                   "<code>P(quebra em t | vivo em t) = g⁻¹(f(idade) + xβ)</code>: a idade entra "
                   "como <b>baseline</b> (a maturação), as covariáveis deslocam a linha de base. "
                   "A curva passa a ser <b>por contrato</b>, que é o que o ECL por contrato pede. "
                   "<code>spline</code> suaviza a cauda onde a base rareia; <code>dummies</code> é "
                   "totalmente flexível (uma indicadora por idade); <code>logit</code> roda no "
                   "núcleo, <code>cloglog</code> é o análogo discreto de Cox e exige "
                   "<code>statsmodels</code>. Categóricas devem vir codificadas: a política de "
                   "categorização é a do <code>ModelSegmenter</code>.</div>"),
            W.HBox([self.sel_hz_features,
                    W.VBox([W.HBox([self.dd_hz_by, self.dd_hz_baseline],
                                   layout=W.Layout(flex_flow="row wrap")),
                            W.HBox([self.dd_hz_link, self.fl_hz_C],
                                   layout=W.Layout(flex_flow="row wrap")),
                            W.HBox([self.sl_hz_knots, self.sl_hz_max_age, self.sl_hz_horizon],
                                   layout=W.Layout(flex_flow="row wrap"))])],
                   layout=W.Layout(align_items="flex-start")),
            W.HBox([self.btn_hz, self.btn_hz_ph, self.btn_hz_adopt],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_hz_status,
        ])
        card_ctrl.add_class("survui-card")
        card_fit = W.VBox([W.HTML("<div class='survui-h'>Ajuste</div>"), self.out_hz_metrics,
                           W.HBox([self.out_hz_plot_base, self.out_hz_plot_prof],
                                  layout=W.Layout(justify_content="space-between")),
                           W.HTML("<div class='survui-legend'>À esquerda, a curva do <b>contrato "
                                  "médio</b> (covariáveis na média) contra o Kaplan-Meier do mesmo "
                                  "grupo: se a paramétrica se afasta do KM em idades com base "
                                  "grande, o problema é do baseline. À direita, as curvas dos "
                                  "perfis P10 / P50 / P90 do preditor linear: a <b>abertura</b> "
                                  "entre elas é o quanto as covariáveis discriminam.</div>"),
                           self.out_hz_coef])
        card_fit.add_class("survui-card")
        card_ph = W.VBox([
            W.HTML("<div class='survui-h'>Riscos proporcionais</div>"),
            W.HTML("<div class='survui-legend'>H0: o efeito de cada covariável é o <b>mesmo em "
                   "todas as idades</b>. O teste compara, por razão de verossimilhança, o modelo "
                   "ajustado contra o mesmo modelo com interações <code>feature × ln(1 + idade)</code>. "
                   "Rejeitar significa que a curva do contrato jovem e a do maduro têm inclinações "
                   "distintas: a interação deveria entrar no modelo.</div>"),
            self.out_hz_ph,
        ])
        card_ph.add_class("survui-card")
        return (card_ctrl, card_fit, card_ph)

    def _clear_hazard_outputs(self):
        for w in ("out_hz_status", "out_hz_metrics", "out_hz_coef", "out_hz_plot_base",
                  "out_hz_plot_prof", "out_hz_ph"):
            getattr(self, w).value = ""

    def _hz_kwargs(self) -> dict:
        return dict(baseline=self.dd_hz_baseline.value, link=self.dd_hz_link.value,
                    C=float(self.fl_hz_C.value), n_knots=int(self.sl_hz_knots.value),
                    max_age=(int(self.sl_hz_max_age.value) or None))

    def _on_hz(self, b):
        if self.des_ is None:
            self.out_hz_status.value = self._notice("Monte o painel na aba <b>Painel</b> antes.")
            return
        feats = list(self.sel_hz_features.value)
        if not feats:
            self.out_hz_status.value = self._notice(
                "Selecione ao menos uma <b>feature</b> (numérica, sem NaN). Sem covariáveis, "
                "a curva não paramétrica da aba <i>Kaplan-Meier</i> é o modelo.")
            return
        by = self.dd_hz_by.value
        with self._busy(self.btn_hz, self.btn_hz_ph, self.btn_hz_adopt, status=self.out_hz_status,
                        msg="ajustando…"):
            try:
                lt = LifetimePD(method="hazard", horizon=int(self.sl_hz_horizon.value),
                                **self._hz_kwargs()).fit(self.des_, by=by, features=feats)
            except Exception as exc:  # noqa: BLE001
                self.out_hz_status.value = self._notice(f"Não foi possível ajustar: {_esc(exc)}")
                self._log(f"[hazard] ERRO: {type(exc).__name__}: {exc}")
                return
        self.hazard_lt_ = lt
        self.ph_ = None
        self.out_hz_ph.value = ""
        self._render_hazard()
        self.out_hz_status.value = self._ok_msg(
            f"Hazard ajustado: {len(feats)} feature(s), baseline <b>{lt.engine_kwargs['baseline']}</b>, "
            f"{len(lt.curves_)} curva(s) de referência" + (f" por <b>{_esc(by)}</b>" if by else "") + ".")
        self._log(f"[hazard] ajustado com {feats} (by={by}).")

    def _hz_loglik(self, modelo, parte: ContractPanel) -> tuple:
        d = parte.spells(features=modelo.features, max_age=modelo.max_age)
        X = d[modelo.features].to_numpy(dtype=float) if modelo.features else None
        h = np.clip(modelo.predict_hazard(d[parte.age_col].to_numpy(dtype=int), X), 1e-12, 1 - 1e-12)
        y = d[parte.default_col].to_numpy(dtype=float)
        ll = float(np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)))
        return ll, len(d), int(y.sum())

    def _render_hazard(self):
        lt = self.hazard_lt_
        if lt is None:
            return
        by = lt.by
        partes = {GLOBAL: self.des_} if by is None else self.des_.by(by)
        # métricas: verossimilhança, AIC, pessoa-períodos
        ll_tot = n_tot = ev_tot = 0.0
        k_tot = 0
        for rot, modelo in lt.hazard_models_.items():
            ll, n, ev = self._hz_loglik(modelo, partes[rot])
            ll_tot += ll
            n_tot += n
            ev_tot += ev
            k_tot += int(modelo._coef.size) + 1
        aic = 2 * k_tot - 2 * ll_tot
        self.out_hz_metrics.value = self._metric_tiles({
            "pessoa-períodos": int(n_tot), "defaults": int(ev_tot), "features": len(lt.features),
            "parâmetros": int(k_tot), "log-verossimilhança": ll_tot, "AIC": aic,
            "modelos (grupos)": len(lt.hazard_models_)})
        # coeficientes (só intercepto + features; o baseline fica na tabela expandível)
        frames = []
        for rot, modelo in lt.hazard_models_.items():
            cf = modelo.coef_frame()
            cf.insert(0, "grupo", self._rot(rot))
            frames.append(cf)
        coef = pd.concat(frames, ignore_index=True)
        feat_mask = coef["termo"].isin(["(intercepto)"] + list(lt.features))
        base = coef[~feat_mask]
        html = self._df_html(coef[feat_mask], fmt_cols={"coeficiente": "{:.4f}", "odds_ratio": "{:.3f}"})
        if len(base):
            html += ("<details class='survui-guide' style='margin-top:8px'><summary>Termos do "
                     f"baseline em idade ({len(base)})</summary>"
                     + self._df_html(base, max_height="240px",
                                     fmt_cols={"coeficiente": "{:.4f}", "odds_ratio": "{:.3f}"})
                     + "</details>")
        self.out_hz_coef.value = (
            "<div class='survui-legend'><b>odds_ratio</b> = exp(β): quanto a chance condicional "
            "de quebrar em cada período multiplica por unidade da feature (só no link logit).</div>"
            + html)
        # gráfico 1: baseline vs KM do grupo
        try:
            kms = {}
            for rot, parte in partes.items():
                kms[f"KM · {self._rot(rot)}"] = kaplan_meier(parte, horizon=lt.horizon)
            fig = self._fig_curves(lt.curves_, "cumulative", extras=kms,
                                   titulo="Contrato médio (hazard) × Kaplan-Meier")
            self.out_hz_plot_base.value = self._fig_html(fig, slot=self.out_hz_plot_base)
        except Exception as exc:  # noqa: BLE001
            self.out_hz_plot_base.value = self._notice(f"Não foi possível desenhar: {_esc(exc)}")
        # gráfico 2: perfis P10/P50/P90 do preditor linear (primeiro grupo)
        try:
            rot0 = next(iter(lt.hazard_models_))
            modelo = lt.hazard_models_[rot0]
            parte = partes[rot0]
            primeiro = parte.df.groupby(parte.id_col, sort=False).head(1)
            X = primeiro[lt.features].to_numpy(dtype=float)
            beta = modelo._coef[-len(lt.features):]
            eta = X @ beta
            perfis = {}
            for q, nome in ((0.10, "P10 (menor risco)"), (0.50, "P50"), (0.90, "P90 (maior risco)")):
                i = int(np.argmin(np.abs(eta - np.quantile(eta, q))))
                perfis[nome] = modelo.predict_curve(X[i], horizon=lt.horizon, label=nome)
            fig2 = self._fig_curves(perfis, "cumulative",
                                    titulo=f"Perfis de risco · {self._rot(rot0)}")
            self.out_hz_plot_prof.value = self._fig_html(fig2, slot=self.out_hz_plot_prof)
        except Exception as exc:  # noqa: BLE001
            self.out_hz_plot_prof.value = self._notice(f"Não foi possível desenhar: {_esc(exc)}")

    def _on_hz_ph(self, b):
        if self.des_ is None:
            self.out_hz_ph.value = self._notice("Monte o painel antes.")
            return
        feats = list(self.sel_hz_features.value)
        if not feats:
            self.out_hz_ph.value = self._notice("Selecione as features a testar.")
            return
        with self._busy(self.btn_hz, self.btn_hz_ph, status=self.out_hz_ph, msg="testando…"):
            try:
                self.ph_ = ph_test(self.des_, feats, alpha=float(self.fl_km_alpha.value),
                                   **self._hz_kwargs())
            except Exception as exc:  # noqa: BLE001
                self.out_hz_ph.value = self._notice(f"Não foi possível testar: {_esc(exc)}")
                self._log(f"[ph] ERRO: {type(exc).__name__}: {exc}")
                return
        self._render_ph()

    def _render_ph(self):
        r = self.ph_
        if r is None:
            return
        ok = r["proporcional"]
        blocos = [{"bloco": "Riscos proporcionais (LR)", "nivel": "ok" if ok else "bad",
                   "veredito": "não rejeita H0" if ok else "rejeita H0",
                   "detalhe": f"χ² = {r['estatistica']:.2f} · gl = {r['gl']} · p = {r['p_valor']:.4f}"}]
        leitura = ("os efeitos são estáveis na idade: o modelo sem interações basta."
                   if ok else _CONSELHO_VAL["Riscos proporcionais"])
        self.out_hz_ph.value = (
            self._placar_html(blocos) + f"<div class='survui-legend'>Leitura: {leitura}</div>"
            + self._df_html(r["interacoes"][["feature", "coeficiente", "leitura"]],
                            fmt_cols={"coeficiente": "{:.4f}"}))

    def _on_hz_adopt(self, b):
        if self.hazard_lt_ is None:
            self.out_hz_status.value = self._notice("Ajuste o hazard antes de adotar.")
            return
        self._adopt(self.hazard_lt_, "hazard")
        self.out_hz_status.value = self._ok_msg(
            "Curva do estudo adotada (hazard com covariáveis). O <code>apply</code> na carteira "
            "usa as features de cada contrato; a validação mede a discriminação.")

    # ==================================================================
    # Aba Paramétrico
    # ==================================================================
    def _build_tab_param(self):
        self.sel_par_dists = W.SelectMultiple(
            options=[(DISTRIBUTION_LABELS[d], d) for d in DISTRIBUTIONS], value=tuple(DISTRIBUTIONS),
            rows=5, description="famílias:", style={"description_width": "initial"},
            layout=W.Layout(width="330px"))
        self.dd_par_by = W.Dropdown(options=[("(carteira inteira)", None)], value=None,
                                    description="por grupo:", style={"description_width": "initial"},
                                    layout=W.Layout(width="260px"))
        self.sl_par_min_risk = W.BoundedIntText(value=30, min=1, max=100000,
                                                description="base mínima da junção:",
                                                style={"description_width": "initial"},
                                                layout=W.Layout(width="230px"))
        self.sl_par_junction = W.BoundedIntText(value=0, min=0, max=600,
                                                description="junção manual (0 = auto):",
                                                style={"description_width": "initial"},
                                                layout=W.Layout(width="230px"))
        self.sl_par_horizon = W.BoundedIntText(value=60, min=1, max=600, description="horizonte:",
                                               style={"description_width": "initial"},
                                               layout=W.Layout(width="150px"))
        self.cb_par_match = W.Checkbox(value=True, indent=False,
                                       description="casar o nível na junção")
        self.dd_par_mode = W.Dropdown(options=[("emenda: empírica + cauda paramétrica", "splice"),
                                               ("paramétrica pura (toda a curva)", "pure")],
                                      value="splice", description="curva adotada:",
                                      style={"description_width": "initial"},
                                      layout=W.Layout(width="360px"))
        self.dd_par_choice = W.Dropdown(options=[(DISTRIBUTION_LABELS[d], d) for d in DISTRIBUTIONS],
                                        value="weibull", description="família adotada:",
                                        style={"description_width": "initial"},
                                        layout=W.Layout(width="330px"))
        self.dd_par_choice.observe(lambda c: self._render_param_splice(), names="value")
        self.dd_par_show = W.Dropdown(options=[("(carteira)", GLOBAL)], value=GLOBAL,
                                      description="grupo no gráfico:",
                                      style={"description_width": "initial"},
                                      layout=W.Layout(width="260px"))
        self.dd_par_show.observe(lambda c: (self._render_param_plots(), self._render_param_splice()),
                                 names="value")
        self.btn_par = W.Button(description="Ajustar famílias", icon="area-chart",
                                button_style="primary", layout=W.Layout(width="auto", min_width="170px"))
        self.btn_par.on_click(self._on_par)
        self.btn_par_adopt = W.Button(description="Adotar como curva do estudo", icon="check",
                                      button_style="success",
                                      layout=W.Layout(width="auto", min_width="240px"))
        self.btn_par_adopt.on_click(self._on_par_adopt)
        self.out_par_status = W.HTML()
        self.out_par_rank = W.HTML()
        self.out_par_plot_h = W.HTML(layout=W.Layout(width="49%"))
        self.out_par_plot_s = W.HTML(layout=W.Layout(width="49%"))
        self.out_par_splice = W.HTML()
        self.out_par_splice_info = W.HTML()

        card_ctrl = W.VBox([
            W.HTML("<div class='survui-h'>Famílias paramétricas e a cauda</div>"),
            W.HTML("<div class='survui-help'><div class='ttl'>Por que uma cauda paramétrica</div>"
                   "A curva empírica só vai até onde a carteira foi observada; um contrato de 60 "
                   "meses numa carteira com 30 de histórico precisa de 30 meses que os dados não "
                   "mostram. A extensão plana (repetir o último hazard) é a hipótese mínima; a "
                   "<b>alternativa defensável</b> é ajustar uma família ao trecho observado e "
                   "deixá-la dizer o que a maturação faz depois. <b>Weibull</b> com "
                   "<code>k &gt; 1</code> é hazard crescente (maturação), <code>k &lt; 1</code> "
                   "decrescente (seleção); <b>log-normal</b> e <b>log-logística</b> fazem "
                   "corcova (sobe e depois cai, o padrão de crédito ao consumidor); "
                   "<b>Gompertz</b> cresce exponencialmente. O <b>AIC</b> desempata; o gráfico "
                   "contra o KM decide.</div>"),
            W.HBox([self.sel_par_dists,
                    W.VBox([W.HBox([self.dd_par_by, self.sl_par_horizon],
                                   layout=W.Layout(flex_flow="row wrap")),
                            W.HBox([self.sl_par_min_risk, self.sl_par_junction],
                                   layout=W.Layout(flex_flow="row wrap")),
                            W.HBox([self.cb_par_match], layout=W.Layout(flex_flow="row wrap"))])],
                   layout=W.Layout(align_items="flex-start")),
            W.HBox([self.btn_par], layout=W.Layout(align_items="center")),
            self.out_par_status,
        ])
        card_ctrl.add_class("survui-card")
        card_rank = W.VBox([W.HTML("<div class='survui-h'>Ranking por AIC</div>"),
                            W.HTML("<div class='survui-legend'><b>delta_aic</b> abaixo de 2 = "
                                   "famílias indistinguíveis pelos dados; escolha a forma mais "
                                   "plausível para o produto.</div>"),
                            self.out_par_rank,
                            W.HBox([self.dd_par_show], layout=W.Layout(align_items="center")),
                            W.HBox([self.out_par_plot_h, self.out_par_plot_s],
                                   layout=W.Layout(justify_content="space-between"))])
        card_rank.add_class("survui-card")
        card_splice = W.VBox([
            W.HTML("<div class='survui-h'>A curva emendada</div>"),
            W.HTML("<div class='survui-legend'>Até a <b>junção</b> (última idade cuja base em "
                   "risco ainda é ≥ base mínima) manda a curva empírica (KM do grupo); dali em "
                   "diante, a família adotada. <i>Casar o nível</i> multiplica a cauda pela razão "
                   "entre a média empírica e a paramétrica nas últimas idades antes da junção: "
                   "sem degrau, e o <b>formato</b> continua o da família.</div>"),
            W.HBox([self.dd_par_mode, self.dd_par_choice, self.btn_par_adopt],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_par_splice_info, self.out_par_splice,
        ])
        card_splice.add_class("survui-card")
        return (card_ctrl, card_rank, card_splice)

    def _clear_param_outputs(self):
        for w in ("out_par_status", "out_par_rank", "out_par_plot_h", "out_par_plot_s",
                  "out_par_splice", "out_par_splice_info"):
            getattr(self, w).value = ""

    def _par_partes(self) -> Dict[object, ContractPanel]:
        by = self.dd_par_by.value
        return {GLOBAL: self.des_} if by is None else self.des_.by(by)

    def _on_par(self, b):
        if self.des_ is None:
            self.out_par_status.value = self._notice("Monte o painel na aba <b>Painel</b> antes.")
            return
        dists = list(self.sel_par_dists.value)
        if not dists:
            self.out_par_status.value = self._notice("Selecione ao menos uma família.")
            return
        with self._busy(self.btn_par, self.btn_par_adopt, status=self.out_par_status,
                        msg="ajustando as famílias…"):
            try:
                partes = self._par_partes()
                ranks, modelos = [], {}
                for rot, parte in partes.items():
                    rk, ms = fit_parametric(parte, distributions=dists)
                    rk.insert(0, "grupo", self._rot(rot))
                    ranks.append(rk)
                    modelos[rot] = ms
                self.param_rank_ = pd.concat(ranks, ignore_index=True)
                self.param_models_ = modelos
            except Exception as exc:  # noqa: BLE001
                self.out_par_status.value = self._notice(f"Não foi possível ajustar: {_esc(exc)}")
                self._log(f"[paramétrico] ERRO: {type(exc).__name__}: {exc}")
                return
        # a família adotada por padrão: menor AIC somado entre os grupos
        soma = self.param_rank_.groupby("distribuicao")["aic"].sum()
        melhor = str(soma.idxmin())
        self.dd_par_choice.options = [(DISTRIBUTION_LABELS[d], d) for d in dists]
        self.dd_par_choice.value = melhor if melhor in dists else dists[0]
        self.dd_par_show.options = [(self._rot(r), r) for r in self.param_models_]
        self.dd_par_show.value = next(iter(self.param_models_))
        self._render_param()
        self.out_par_status.value = self._ok_msg(
            f"{len(dists)} família(s) × {len(self.param_models_)} grupo(s) ajustadas. Melhor AIC "
            f"agregado: <b>{DISTRIBUTION_LABELS[melhor]}</b>.")
        self._log(f"[paramétrico] famílias {dists}; melhor agregado: {melhor}.")

    def _render_param(self):
        if self.param_rank_ is None:
            return
        rk = self.param_rank_[["grupo", "posicao", "distribuicao", "parametros", "loglik", "aic",
                               "bic", "delta_aic", "convergiu", "leitura"]]
        self.out_par_rank.value = self._df_html(
            rk, color_map={"posicao": self._css_posicao},
            fmt_cols={"loglik": "{:.1f}", "aic": "{:.1f}", "bic": "{:.1f}", "delta_aic": "{:.1f}"})
        self._render_param_plots()
        self._render_param_splice()

    def _render_param_plots(self):
        if not self.param_models_:
            return
        rot = self.dd_par_show.value
        if rot not in self.param_models_:
            return
        parte = self._par_partes().get(rot)
        if parte is None:
            return
        try:
            _, tkm = kaplan_meier(parte, return_table=True)
            km = kaplan_meier(parte)
            H = int(self.sl_par_horizon.value)
            extras = {DISTRIBUTION_LABELS[d].split(" (")[0]: m.curve(horizon=H)
                      for d, m in self.param_models_[rot].items()}
            fig_h = self._fig_curves({rot: km}, "hazard", extras=extras,
                                     titulo=f"hazard observado × famílias · {self._rot(rot)}")
            self.out_par_plot_h.value = self._fig_html(fig_h, slot=self.out_par_plot_h)
            fig_s = self._fig_curves({rot: km}, "cumulative", tables={rot: tkm}, extras=extras,
                                     titulo=f"PD acumulada: KM (IC) × famílias · {self._rot(rot)}")
            self.out_par_plot_s.value = self._fig_html(fig_s, slot=self.out_par_plot_s)
        except Exception as exc:  # noqa: BLE001
            self.out_par_plot_h.value = self._notice(f"Não foi possível desenhar: {_esc(exc)}")
            self.out_par_plot_s.value = ""

    def _par_build_curves(self) -> Dict[object, PDCurve]:
        """As curvas (emendadas ou puras) da família escolhida, por grupo."""
        dist = self.dd_par_choice.value
        H = int(self.sl_par_horizon.value)
        partes = self._par_partes()
        curvas = {}
        for rot, ms in self.param_models_.items():
            if dist not in ms:
                continue
            rotulo = "" if rot == GLOBAL else str(rot)
            m = ms[dist]
            if self.dd_par_mode.value == "pure":
                curvas[rot] = m.curve(horizon=H, label=rotulo)
                continue
            parte = partes[rot]
            km = kaplan_meier(parte, label=rotulo)
            j = int(self.sl_par_junction.value) or junction_age(parte, int(self.sl_par_min_risk.value))
            if j < 0:
                j = 0
            curvas[rot] = splice_curves(km, m, junction=j, horizon=H,
                                        match_level=bool(self.cb_par_match.value))
        return curvas

    def _render_param_splice(self):
        if not self.param_models_:
            return
        try:
            curvas = self._par_build_curves()
            self.param_curves_ = curvas
            rot = self.dd_par_show.value if self.dd_par_show.value in curvas else next(iter(curvas))
            c = curvas[rot]
            j = c.meta.get("junction")
            fig = self._fig_curves({rot: c}, "hazard", junction=j if self.dd_par_mode.value == "splice" else None,
                                   titulo=f"Curva adotável · {self._rot(rot)} · "
                                          f"{DISTRIBUTION_LABELS[self.dd_par_choice.value]}")
            self.out_par_splice.value = self._fig_html(fig, slot=self.out_par_splice)
            linhas = []
            for r, cc in curvas.items():
                linhas.append({"grupo": self._rot(r), "junção": cc.meta.get("junction", np.nan),
                               "fator de nível": cc.meta.get("fator_nivel", np.nan),
                               "pd_12m": cc.pd_12m(), "pd_lifetime": cc.pd_lifetime(),
                               "mediana": median_survival(cc)})
            self.out_par_splice_info.value = self._df_html(
                pd.DataFrame(linhas), pct_cols=["pd_12m", "pd_lifetime"],
                fmt_cols={"fator de nível": "{:.3f}", "mediana": "{:.0f}", "junção": "{:.0f}"})
        except Exception as exc:  # noqa: BLE001
            self.out_par_splice.value = self._notice(f"Não foi possível montar a emenda: {_esc(exc)}")

    def _on_par_adopt(self, b):
        if not self.param_models_:
            self.out_par_status.value = self._notice("Ajuste as famílias antes de adotar.")
            return
        try:
            curvas = self._par_build_curves()
        except Exception as exc:  # noqa: BLE001
            self.out_par_status.value = self._notice(f"Não foi possível montar as curvas: {_esc(exc)}")
            return
        lt = LifetimePD.from_curves(curvas, method="km")
        lt.by = self.dd_par_by.value
        lt.freq = self.des_.freq
        modo = self.dd_par_mode.value
        lt.meta.update({"metodo": "parametric" if modo == "pure" else "km",
                        "tail": "parametric" if modo == "splice" else None,
                        "distribution": self.dd_par_choice.value, "by": lt.by,
                        "horizon": int(self.sl_par_horizon.value), "n_contratos": self.des_.n_contracts})
        self._adopt(lt, "parametric" if modo == "pure" else "km")
        self.out_par_status.value = self._ok_msg(
            f"Curva do estudo adotada: {DISTRIBUTION_LABELS[self.dd_par_choice.value]}"
            + (" como cauda da curva empírica" if modo == "splice" else " em toda a curva")
            + f", {len(curvas)} curva(s), horizonte {int(self.sl_par_horizon.value)}.")

    # ==================================================================
    # A curva adotada
    # ==================================================================
    def _adopt(self, lt: LifetimePD, source: str):
        """Torna ``lt`` a curva do estudo: zera ajustes e validação anteriores."""
        self.model_base_ = lt
        self.model_source_ = source
        self._calib_targets = self._cycle = None
        self.model_ = lt
        self._invalidate_validation()
        self.mlflow_run_id_ = self.model_path_ = None
        self.study_ = None
        self._clear_val_outputs()
        self._invalidate_exportar("a curva do estudo mudou")
        self._render_calib()
        self._refresh_all()
        self._log(f"[curva] adotada: {source} ({len(lt.curves_)} curva(s), H={lt.horizon}).")

    def _rebuild_model(self):
        """Reaplica calibração e ciclo sobre a curva base, na ordem."""
        if self.model_base_ is None:
            self.model_ = None
            return
        m = self.model_base_
        if self._calib_targets:
            m = m.calibrate_to(self._calib_targets)
        if self._cycle:
            c = self._cycle
            m = m.condition(c["z"], rho=c["rho"], decay=c.get("decay"), mode=c.get("mode", "shift"))
        self.model_ = m
        self._invalidate_validation()
        self._clear_val_outputs()
        self._invalidate_exportar("a curva vigente mudou (calibração/ciclo)")
        self._render_calib()
        self._refresh_all()

    # ==================================================================
    # Aba Calibração & Ciclo
    # ==================================================================
    def _build_tab_calib(self):
        self.out_calib_model = W.HTML()
        self.box_calib_inputs = W.VBox([])
        self.btn_calib = W.Button(description="Calibrar nível", icon="crosshairs",
                                  button_style="primary", layout=W.Layout(width="auto", min_width="150px"))
        self.btn_calib.on_click(self._on_calib)
        self.btn_calib_clear = W.Button(description="Remover calibração", icon="undo",
                                        layout=W.Layout(width="auto", min_width="170px"))
        self.btn_calib_clear.on_click(self._on_calib_clear)
        self.out_calib_status = W.HTML()

        self.fl_z = W.FloatText(value=-1.0, description="z (fator sistêmico):",
                                style={"description_width": "initial"}, layout=W.Layout(width="220px"))
        self.fl_rho = W.BoundedFloatText(value=0.10, min=0.0, max=0.99, step=0.01, description="ρ:",
                                         style={"description_width": "initial"},
                                         layout=W.Layout(width="130px"))
        self.fl_decay = W.BoundedFloatText(value=0.0, min=0.0, max=1.0, step=0.05,
                                           description="reversão (decay, 0 = nenhuma):",
                                           style={"description_width": "initial"},
                                           layout=W.Layout(width="280px"))
        self.dd_mode = W.Dropdown(options=[("shift (idempotente em z = 0)", "shift"),
                                           ("conditional (lei do fator único)", "conditional")],
                                  value="shift", description="modo:",
                                  style={"description_width": "initial"},
                                  layout=W.Layout(width="300px"))
        self.btn_cycle = W.Button(description="Condicionar ao ciclo", icon="cloud",
                                  button_style="primary", layout=W.Layout(width="auto", min_width="190px"))
        self.btn_cycle.on_click(self._on_cycle)
        self.btn_cycle_clear = W.Button(description="Remover ciclo", icon="undo",
                                        layout=W.Layout(width="auto", min_width="150px"))
        self.btn_cycle_clear.on_click(self._on_cycle_clear)
        self.out_cycle_status = W.HTML()
        self.dd_calib_kind = W.Dropdown(options=list(KINDS), value="cumulative",
                                        description="representação:",
                                        style={"description_width": "initial"},
                                        layout=W.Layout(width="270px"))
        self.dd_calib_kind.observe(lambda c: self._render_calib_plot(), names="value")
        self.out_calib_plot = W.HTML()
        self.out_calib_table = W.HTML()

        card_model = W.VBox([W.HTML("<div class='survui-h'>Curva vigente</div>"), self.out_calib_model])
        card_model.add_class("survui-card")
        card_calib = W.VBox([
            W.HTML("<div class='survui-h'>Calibração de nível (a ponte com o scorecard)</div>"),
            W.HTML("<div class='survui-help'><div class='ttl'>O desenho usual</div>"
                   "O modelo transversal (scorecard, <code>ModelSegmenter</code>) dá o <b>nível</b> "
                   "da PD de 12 meses por grupo; a curva de sobrevivência dá o <b>formato</b> da "
                   "maturação. <code>calibrate_to</code> desloca o hazard no logit até a PD "
                   "acumulada de 12 meses bater com o alvo, preservando a forma. Informe o alvo por "
                   "grupo (vazio = não calibra o grupo).</div>"),
            self.box_calib_inputs,
            W.HBox([self.btn_calib, self.btn_calib_clear],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_calib_status,
        ])
        card_calib.add_class("survui-card")
        card_cycle = W.VBox([
            W.HTML("<div class='survui-h'>Ciclo econômico (Vasicek, TTC → PIT)</div>"),
            W.HTML("<div class='survui-help'><div class='ttl'>Convenção de sinal</div>"
                   "<code>z &gt; 0</code> = ciclo <b>benigno</b> (PD cai); <code>z &lt; 0</code> = "
                   "adverso (PD sobe). <code>shift</code> aplica "
                   "<code>Φ(Φ⁻¹(PD) − √ρ·z)</code> a cada hazard e não muda nada em z = 0; "
                   "<code>conditional</code> reescala por <code>1/√(1−ρ)</code> (a lei exata do fator "
                   "único, a mesma do motor de capital) e não é idempotente em zero. A "
                   "<b>reversão</b> dissipa o choque ao longo do horizonte "
                   "(<code>z_t = z·(1−decay)^t</code>): forte nos primeiros períodos, sumindo depois, "
                   "que é como o ciclo costuma entrar num horizonte lifetime. O <code>z</code> por "
                   "horizonte pode vir da projeção de um modelo satélite (<code>econometric</code>) "
                   "via <code>ui.model_base_.condition(z_vetor, rho)</code>.</div>"),
            W.HBox([self.fl_z, self.fl_rho, self.fl_decay, self.dd_mode],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.btn_cycle, self.btn_cycle_clear],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_cycle_status,
        ])
        card_cycle.add_class("survui-card")
        card_plot = W.VBox([W.HTML("<div class='survui-h'>Antes × depois</div>"),
                            W.HBox([self.dd_calib_kind], layout=W.Layout(align_items="center")),
                            self.out_calib_plot, self.out_calib_table])
        card_plot.add_class("survui-card")
        return (card_model, card_calib, card_cycle, card_plot)

    def _rebuild_calib_inputs(self):
        """Um campo de alvo por curva do modelo base (pré-preenchido com a PD atual)."""
        self._calib_inputs = {}
        if self.model_base_ is None:
            self.box_calib_inputs.children = (W.HTML(self._notice(
                "Adote uma curva (abas <b>Kaplan-Meier</b>, <b>Hazard</b> ou <b>Paramétrico</b>) "
                "para calibrar.")),)
            return
        filhos = []
        alvos = self._calib_targets or {}
        for rot, c in self.model_base_.curves_.items():
            w = W.Text(value=(f"{alvos[rot]:.4f}" if rot in alvos else ""),
                       placeholder=f"atual {c.pd_12m():.4f}",
                       description=f"{self._rot(rot)}:", style={"description_width": "initial"},
                       layout=W.Layout(width="260px"))
            self._calib_inputs[rot] = w
            filhos.append(w)
        linhas = [W.HBox(filhos[i:i + 3], layout=W.Layout(flex_flow="row wrap"))
                  for i in range(0, len(filhos), 3)]
        self.box_calib_inputs.children = tuple(linhas)

    def _parse_targets(self) -> Dict[object, float]:
        alvos = {}
        for rot, w in self._calib_inputs.items():
            txt = str(w.value or "").strip().replace(",", ".").replace("%", "")
            if not txt:
                continue
            v = float(txt)
            if v > 1.0:
                v = v / 100.0
            if not (0.0 < v < 1.0):
                raise ValueError(f"alvo de {self._rot(rot)} deve estar em (0, 1); recebido {txt}.")
            alvos[rot] = v
        return alvos

    def _on_calib(self, b):
        if self.model_base_ is None:
            self.out_calib_status.value = self._notice("Adote uma curva antes.")
            return
        try:
            alvos = self._parse_targets()
            if not alvos:
                raise ValueError("informe ao menos um alvo de PD de 12 meses.")
            self._calib_targets = alvos
            self._rebuild_model()
        except Exception as exc:  # noqa: BLE001
            self._calib_targets = None
            self.out_calib_status.value = self._notice(f"Não foi possível calibrar: {_esc(exc)}")
            self._log(f"[calibração] ERRO: {type(exc).__name__}: {exc}")
            return
        deltas = [f"{self._rot(r)}: δ = {self.model_.curves_[r].meta.get('calibracao', {}).get('delta', float('nan')):+.3f}"
                  for r in alvos]
        self.out_calib_status.value = self._ok_msg(
            "Nível calibrado (deslocamento no logit do hazard): " + " · ".join(deltas) + ".")
        self._log(f"[calibração] alvos={ {self._rot(k): v for k, v in alvos.items()} }")

    def _on_calib_clear(self, b):
        self._calib_targets = None
        for w in self._calib_inputs.values():
            w.value = ""
        self._rebuild_model()
        self.out_calib_status.value = self._ok_msg("Calibração removida.")

    def _on_cycle(self, b):
        if self.model_base_ is None:
            self.out_cycle_status.value = self._notice("Adote uma curva antes.")
            return
        decay = float(self.fl_decay.value) or None
        self._cycle = {"z": float(self.fl_z.value), "rho": float(self.fl_rho.value),
                       "decay": decay, "mode": self.dd_mode.value}
        try:
            self._rebuild_model()
        except Exception as exc:  # noqa: BLE001
            self._cycle = None
            self.out_cycle_status.value = self._notice(f"Não foi possível condicionar: {_esc(exc)}")
            return
        c = self._cycle
        self.out_cycle_status.value = self._ok_msg(
            f"Ciclo aplicado: z = {c['z']:+.2f}, ρ = {c['rho']:.2f}, "
            f"{'reversão ' + format(c['decay'], '.2f') + ' por período' if c['decay'] else 'sem reversão'}, "
            f"modo {c['mode']}.")
        self._log(f"[ciclo] {c}")

    def _on_cycle_clear(self, b):
        self._cycle = None
        self._rebuild_model()
        self.out_cycle_status.value = self._ok_msg("Ciclo removido.")

    def _render_calib(self):
        if getattr(self, "out_calib_model", None) is None:
            return
        self._rebuild_calib_inputs()
        if self.model_ is None:
            self.out_calib_model.value = self._notice("Nenhuma curva adotada ainda.")
            self.out_calib_plot.value = ""
            self.out_calib_table.value = ""
            return
        m = self.model_
        pills = [self._pill(f"origem: {self.model_source_}", "muted"),
                 self._pill(f"{len(m.curves_)} curva(s)", "muted"),
                 self._pill(f"horizonte {m.horizon} · freq {m.freq}", "muted")]
        if m.by:
            pills.append(self._pill(f"por {_esc(m.by)}", "muted"))
        pills.append(self._pill("calibrada" if self._calib_targets else "sem calibração",
                                "green" if self._calib_targets else "yellow"))
        pills.append(self._pill("condicionada ao ciclo" if self._cycle else "TTC (sem ciclo)",
                                "green" if self._cycle else "muted"))
        self.out_calib_model.value = "<div class='survui-bar'>" + "".join(pills) + "</div>"
        self._render_calib_plot()
        linhas = []
        for rot, c in m.curves_.items():
            b = self.model_base_.curves_[rot]
            linhas.append({"grupo": self._rot(rot), "pd_12m base": b.pd_12m(), "pd_12m vigente": c.pd_12m(),
                           "pd_lifetime base": b.pd_lifetime(), "pd_lifetime vigente": c.pd_lifetime(),
                           "mediana base": median_survival(b), "mediana vigente": median_survival(c)})
        self.out_calib_table.value = self._df_html(
            pd.DataFrame(linhas), pct_cols=["pd_12m base", "pd_12m vigente", "pd_lifetime base",
                                            "pd_lifetime vigente"],
            fmt_cols={"mediana base": "{:.0f}", "mediana vigente": "{:.0f}"})

    def _render_calib_plot(self):
        if self.model_ is None:
            return
        kind = self.dd_calib_kind.value
        try:
            base = {f"{self._rot(r)} · base": c for r, c in self.model_base_.curves_.items()}
            fig = self._fig_curves(self.model_.curves_, kind, extras=base,
                                   titulo=f"{_ROTULO[kind]}: vigente (cheia) × base (tracejada)")
            self.out_calib_plot.value = self._fig_html(fig, slot=self.out_calib_plot)
        except Exception as exc:  # noqa: BLE001
            self.out_calib_plot.value = self._notice(f"Não foi possível desenhar: {_esc(exc)}")

    # ==================================================================
    # Aba Validação
    # ==================================================================
    def _build_tab_val(self):
        self.out_val_notice = W.HTML()
        self.tx_val_horizons = W.Text(value="12,24,36", description="horizontes:",
                                      style={"description_width": "initial"},
                                      layout=W.Layout(width="230px"))
        self.sl_val_dec_h = W.BoundedIntText(value=12, min=1, max=600, description="horizonte do decil:",
                                             style={"description_width": "initial"},
                                             layout=W.Layout(width="200px"))
        self.sl_val_bins = W.BoundedIntText(value=10, min=2, max=50, description="faixas:",
                                            style={"description_width": "initial"},
                                            layout=W.Layout(width="130px"))
        self.fl_val_alpha = W.BoundedFloatText(value=0.05, min=0.001, max=0.5, step=0.01,
                                               description="α:", style={"description_width": "initial"},
                                               layout=W.Layout(width="120px"))
        self.btn_val = W.Button(description="Validar", icon="check-square-o", button_style="primary",
                                layout=W.Layout(width="auto", min_width="140px"))
        self.btn_val.on_click(self._on_val)
        self.out_val_status = W.HTML()
        self.out_val_placar = W.HTML()
        self.out_val_leitura = W.HTML()
        self.out_val_bt_plot = W.HTML(layout=W.Layout(width="49%"))
        self.out_val_dec_plot = W.HTML(layout=W.Layout(width="49%"))
        self.out_val_bt = W.HTML()
        self.out_val_disc = W.HTML()
        self.out_val_dec = W.HTML()

        card_ctrl = W.VBox([
            W.HTML("<div class='survui-h'>Validação da curva vigente</div>"),
            W.HTML("<div class='survui-help'><div class='ttl'>Os quatro blocos</div>"
                   "<b>Calibração por horizonte</b>: PD acumulada prevista × observada por "
                   "Kaplan-Meier, com o <code>z</code> de Greenwood. <b>Discriminação</b>: AUC/Gini/KS "
                   "do default em <code>h</code> períodos contra a PD prevista, e o <b>C-index</b> de "
                   "Harrell (ordenação com censura). <b>Calibração por decil</b>: prevista × observada "
                   "por faixa de PD prevista com IC binomial e Hosmer-Lemeshow. <b>Riscos "
                   "proporcionais</b>: o teste da aba Hazard, quando a curva é de hazard. Contratos "
                   "são avaliados a partir da <b>primeira observação</b> no painel de validação; "
                   "censurados antes do horizonte saem do denominador da discriminação e do decil."
                   "</div>"),
            self.out_val_notice,
            W.HBox([self.tx_val_horizons, self.sl_val_dec_h, self.sl_val_bins, self.fl_val_alpha,
                    self.btn_val],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_val_status,
        ])
        card_ctrl.add_class("survui-card")
        card_placar = W.VBox([W.HTML("<div class='survui-h'>Placar</div>"), self.out_val_placar,
                              self.out_val_leitura,
                              W.HBox([self.out_val_bt_plot, self.out_val_dec_plot],
                                     layout=W.Layout(justify_content="space-between"))])
        card_placar.add_class("survui-card")
        card_tabs = W.VBox([W.HTML("<div class='survui-h'>Tabelas</div>"),
                            W.HTML("<div class='survui-legend'>Backtest por horizonte</div>"), self.out_val_bt,
                            W.HTML("<div class='survui-legend'>Discriminação por horizonte</div>"), self.out_val_disc,
                            W.HTML("<div class='survui-legend'>Calibração por faixa de PD prevista</div>"),
                            self.out_val_dec])
        card_tabs.add_class("survui-card")
        return (card_ctrl, card_placar, card_tabs)

    def _clear_val_outputs(self):
        for w in ("out_val_status", "out_val_placar", "out_val_leitura", "out_val_bt_plot",
                  "out_val_dec_plot", "out_val_bt", "out_val_disc", "out_val_dec"):
            widget = getattr(self, w, None)
            if widget is not None:
                widget.value = ""

    def _render_val_notice(self):
        if getattr(self, "out_val_notice", None) is None:
            return
        if self.model_ is None:
            self.out_val_notice.value = self._notice("Nenhuma curva adotada: adote uma nas abas "
                                                     "anteriores antes de validar.")
        elif self.oot_ is None:
            self.out_val_notice.value = self._notice(
                "⚠️ Sem partição: a validação rodará <b>no próprio DES</b> (<i>in-sample</i>). O "
                "backtest de uma curva empírica contra os mesmos dados é trivialmente perfeito; "
                "particione na aba <b>Painel</b> para uma validação de verdade.")
        else:
            self.out_val_notice.value = self._ok_msg(
                f"Validação no OOT: {self.oot_.n_contracts} contratos, "
                f"{int(self.oot_.df[self.oot_.default_col].sum())} defaults.")

    @staticmethod
    def _parse_ints(txt, campo="horizontes") -> list:
        vals = []
        for tok in re.split(r"[,\s;]+", str(txt or "").strip()):
            if tok:
                vals.append(int(tok))
        if not vals:
            raise ValueError(f"informe ao menos um valor em {campo}.")
        return sorted(set(vals))

    def _on_val(self, b):
        if self.model_ is None:
            self.out_val_status.value = self._notice("Adote uma curva antes de validar.")
            return
        alvo = self.oot_ if self.oot_ is not None else self.des_
        if alvo is None:
            self.out_val_status.value = self._notice("Monte o painel antes.")
            return
        with self._busy(self.btn_val, status=self.out_val_status, msg="validando…"):
            try:
                hs = self._parse_ints(self.tx_val_horizons.value)
                alpha = float(self.fl_val_alpha.value)
                self._run_validation(alvo, hs, int(self.sl_val_dec_h.value),
                                     int(self.sl_val_bins.value), alpha)
            except Exception as exc:  # noqa: BLE001
                self.out_val_status.value = self._notice(f"Não foi possível validar: {_esc(exc)}")
                self._log(f"[validação] ERRO: {type(exc).__name__}: {exc}")
                return
        self._render_val()
        self._refresh_all()
        self.out_val_status.value = self._ok_msg(
            f"Validação concluída em <b>{self.validated_on_}</b>.")
        self._log(f"[validação] {self.validated_on_}: blocos="
                  f"{[(x['bloco'], x['nivel']) for x in (self.val_blocks_ or [])]}")

    def _run_validation(self, alvo: ContractPanel, hs, dec_h, n_bins, alpha):
        m = self.model_
        self.validated_on_ = "OOT" if self.oot_ is not None else "DES (in-sample)"
        try:
            self.backtest_ = backtest_curve(m, alvo, horizons=hs, alpha=alpha)
        except ValueError as exc:
            self.backtest_ = None
            self._log(f"[validação] backtest indisponível: {exc}")
        try:
            self.discrimination_ = discrimination_by_horizon(m, alvo, horizons=hs)
        except Exception as exc:  # noqa: BLE001
            self.discrimination_ = None
            self._log(f"[validação] discriminação indisponível: {exc}")
        try:
            self.c_index_ = model_concordance(m, alvo, horizon=max(hs))
        except Exception as exc:  # noqa: BLE001
            self.c_index_ = None
            self._log(f"[validação] C-index indisponível: {exc}")
        try:
            self.decile_ = calibration_by_decile(m, alvo, horizon=dec_h, n_bins=n_bins, alpha=alpha)
        except Exception as exc:  # noqa: BLE001
            self.decile_ = None
            self._log(f"[validação] decil indisponível: {exc}")
        self.val_blocks_ = self._val_blocks(alpha)

    def _val_blocks(self, alpha: float) -> list:
        blocos = []
        bt = self.backtest_
        if bt is None or not len(bt):
            blocos.append({"bloco": "Calibração por horizonte", "nivel": "na", "veredito": "indisponível",
                           "detalhe": "horizontes além da janela observada"})
        else:
            dentro = int(bt["dentro_do_ic"].sum())
            n = len(bt)
            pior = bt.loc[bt["z"].abs().idxmax()] if bt["z"].notna().any() else None
            nivel = "ok" if dentro == n else ("warn" if dentro >= n / 2 else "bad")
            det = f"{dentro} de {n} horizonte(s) dentro do IC"
            if pior is not None:
                det += f" · pior z = {pior['z']:+.2f} ({self._rot(pior['grupo'])}, h = {int(pior['horizonte'])})"
            blocos.append({"bloco": "Calibração por horizonte", "nivel": nivel,
                           "veredito": "dentro do IC" if nivel == "ok" else "fora do IC", "detalhe": det})
        d = self.discrimination_
        if d is None or d["auc"].isna().all():
            obs = (d["observacao"].iloc[0] if d is not None and len(d) else "indisponível")
            blocos.append({"bloco": "Discriminação", "nivel": "na", "veredito": "não se aplica",
                           "detalhe": _esc(obs)})
        else:
            auc = float(d["auc"].mean())
            nivel = "ok" if auc >= 0.65 else ("warn" if auc >= 0.55 else "bad")
            det = f"AUC médio {auc:.3f}"
            if self.c_index_ is not None and np.isfinite(self.c_index_):
                det += f" · C-index {self.c_index_:.3f}"
            blocos.append({"bloco": "Discriminação", "nivel": nivel,
                           "veredito": "ordena" if nivel != "bad" else "não ordena", "detalhe": det})
        dec = self.decile_
        if dec is None or not len(dec):
            blocos.append({"bloco": "Calibração por decil", "nivel": "na", "veredito": "indisponível",
                           "detalhe": "sem contratos observados até o horizonte"})
        else:
            p = dec.attrs.get("hl_p_valor", np.nan)
            fora = int((~dec["dentro_do_ic"]).sum())
            nivel = "ok" if (p > alpha and fora == 0) else ("warn" if p > alpha / 5 else "bad")
            blocos.append({"bloco": "Calibração por decil", "nivel": nivel,
                           "veredito": "aderente" if nivel == "ok" else "desvios",
                           "detalhe": f"HL p = {p:.4f} · {fora} de {len(dec)} faixa(s) fora do IC"})
        if self.model_source_ == "hazard" and self.ph_ is not None:
            r = self.ph_
            blocos.append({"bloco": "Riscos proporcionais", "nivel": "ok" if r["proporcional"] else "bad",
                           "veredito": "não rejeita H0" if r["proporcional"] else "rejeita H0",
                           "detalhe": f"χ² = {r['estatistica']:.2f} · gl = {r['gl']} · p = {r['p_valor']:.4f}"})
        elif self.model_source_ == "hazard":
            blocos.append({"bloco": "Riscos proporcionais", "nivel": "na", "veredito": "não testado",
                           "detalhe": "rode o teste na aba Hazard"})
        return blocos

    def _render_val(self):
        if self.val_blocks_ is None:
            return
        self.out_val_placar.value = self._placar_html(self.val_blocks_)
        ruins = [b for b in self.val_blocks_ if b["nivel"] in ("bad", "warn")]
        if ruins:
            itens = "".join(f"<li><b>{b['bloco']}</b>: {_CONSELHO_VAL.get(b['bloco'], '')}</li>"
                            for b in ruins)
            self.out_val_leitura.value = (f"<div class='survui-help'><div class='ttl'>O que fazer</div>"
                                          f"<ul>{itens}</ul></div>")
        else:
            self.out_val_leitura.value = ("<div class='survui-legend'>Todos os blocos aprovados em "
                                          f"<b>{self.validated_on_}</b>.</div>")
        if self.backtest_ is not None:
            bt = self.backtest_.copy()
            bt["grupo"] = bt["grupo"].map(self._rot)
            self.out_val_bt.value = self._df_html(
                bt[["grupo", "horizonte", "n_em_risco_inicial", "n_em_risco_h", "pd_prevista",
                    "pd_observada", "ic_inf", "ic_sup", "erro_absoluto", "z", "p_valor", "dentro_do_ic"]],
                pct_cols=["pd_prevista", "pd_observada", "ic_inf", "ic_sup", "erro_absoluto"],
                color_map={"z": self._css_z, "p_valor": self._css_pvalor_h0, "dentro_do_ic": self._css_ok},
                fmt_cols={"z": "{:+.2f}", "p_valor": "{:.4f}"})
            try:
                from ..ecl.report import plot_backtest

                fig = plot_backtest(self.backtest_)
                self.out_val_bt_plot.value = self._fig_html(fig, slot=self.out_val_bt_plot)
            except Exception as exc:  # noqa: BLE001
                self.out_val_bt_plot.value = self._notice(f"Não foi possível desenhar: {_esc(exc)}")
        else:
            self.out_val_bt.value = self._notice("Backtest indisponível para estes horizontes.")
            self.out_val_bt_plot.value = ""
        if self.discrimination_ is not None:
            d = self.discrimination_
            self.out_val_disc.value = self._df_html(
                d, pct_cols=["taxa_observada", "pd_prevista_media"],
                fmt_cols={"auc": "{:.4f}", "gini": "{:.4f}", "ks": "{:.4f}"})
            if self.c_index_ is not None and np.isfinite(self.c_index_):
                self.out_val_disc.value += (f"<div class='survui-legend'>C-index de Harrell (score = PD "
                                            f"acumulada no maior horizonte): <b>{self.c_index_:.4f}</b>."
                                            "</div>")
        else:
            self.out_val_disc.value = ""
        if self.decile_ is not None:
            dec = self.decile_
            self.out_val_dec.value = (
                f"<div class='survui-legend'>Horizonte {dec.attrs.get('horizonte')} · Hosmer-Lemeshow "
                f"χ² = {dec.attrs.get('hl_estatistica', float('nan')):.2f}, gl = {dec.attrs.get('hl_gl')}, "
                f"p = {dec.attrs.get('hl_p_valor', float('nan')):.4f}</div>"
                + self._df_html(dec, pct_cols=["pd_prevista", "pd_observada", "ic_inf", "ic_sup",
                                               "pd_min", "pd_max"],
                                color_map={"dentro_do_ic": self._css_ok}))
            try:
                self.out_val_dec_plot.value = self._fig_html(self._fig_decile(dec), slot=self.out_val_dec_plot)
            except Exception as exc:  # noqa: BLE001
                self.out_val_dec_plot.value = self._notice(f"Não foi possível desenhar: {_esc(exc)}")
        else:
            self.out_val_dec.value = ""
            self.out_val_dec_plot.value = ""

    def _fig_decile(self, dec: pd.DataFrame):
        import matplotlib.pyplot as plt

        from ...reporting.style import COR_PRIMARIA, COR_SECUNDARIA

        fig, ax = plt.subplots(figsize=(9, 4.6))
        x = np.arange(len(dec))
        ax.bar(x - 0.2, dec["pd_prevista"], width=0.4, color=COR_PRIMARIA, alpha=0.85, label="prevista")
        ax.bar(x + 0.2, dec["pd_observada"], width=0.4, color=COR_SECUNDARIA, alpha=0.85, label="observada")
        ax.errorbar(x + 0.2, dec["pd_observada"],
                    yerr=[dec["pd_observada"] - dec["ic_inf"], dec["ic_sup"] - dec["pd_observada"]],
                    fmt="none", ecolor="black", capsize=3, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{int(f)}\nn={int(n)}" for f, n in zip(dec["faixa"], dec["n"])], fontsize=8)
        ax.set_ylabel(f"PD acumulada em {dec.attrs.get('horizonte')} períodos")
        ax.set_title("Calibração por faixa de PD prevista")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8)
        return fig

    # ==================================================================
    # Aba Exportar
    # ==================================================================
    def _build_tab_exportar(self):
        base = self._nome_arquivo()
        self.out_exp_notice = W.HTML()
        self.out_exp_estado = W.HTML()
        self.btn_exp_estado = W.Button(description="Atualizar", icon="refresh",
                                       layout=W.Layout(width="auto", min_width="130px"))
        self.btn_exp_estado.on_click(lambda b: self._render_export_estado())
        card_estado = W.VBox([
            W.HTML("<div class='survui-h'>O que já está pronto nesta sessão</div>"),
            self.out_exp_notice, self.out_exp_estado,
            W.HBox([self.btn_exp_estado], layout=W.Layout(align_items="center")),
        ])
        card_estado.add_class("survui-card")

        # --- modelo em JSON ------------------------------------------------------
        self.tx_model_path = W.Text(value=f"{base}_lifetime_pd.json", description="arquivo:",
                                    style={"description_width": "initial"}, layout=W.Layout(width="420px"))
        self.btn_model_save = W.Button(description="Salvar modelo", icon="save", button_style="success",
                                       layout=W.Layout(width="auto", min_width="160px"))
        self.btn_model_save.on_click(self._on_model_save)
        self.btn_model_load = W.Button(description="Carregar modelo", icon="upload",
                                       layout=W.Layout(width="auto", min_width="170px"),
                                       tooltip="Lê um LifetimePD em JSON e o adota como curva do estudo.")
        self.btn_model_load.on_click(self._on_model_load)
        self.out_model_status = W.HTML()
        card_model = W.VBox([
            W.HTML("<div class='survui-h'>O modelo (LifetimePD) em JSON</div>"),
            W.HTML("<div class='survui-legend'>A curva vigente serializada com a linhagem (método, "
                   "cauda, calibração, ciclo). É o objeto que <code>ecl_table</code> consome: "
                   "<code>LifetimePD.from_json(caminho)</code> reconstrói as curvas e o "
                   "<code>apply</code>. No motor de hazard, os coeficientes por contrato ficam em "
                   "<code>ui.model_.hazard_models_</code> (persista com <code>joblib</code>); o JSON "
                   "leva as curvas de referência.</div>"),
            W.HBox([self.tx_model_path, self.btn_model_save, self.btn_model_load],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_model_status,
        ])
        card_model.add_class("survui-card")

        # --- MLflow ---------------------------------------------------------------------
        self.tx_mlflow_exp = W.Text(value="", placeholder="vazio = experimento ativo da sessão",
                                    description="experimento:", style={"description_width": "initial"},
                                    layout=W.Layout(width="460px"))
        self.tx_mlflow_run = W.Text(value=base, description="nome do run:",
                                    style={"description_width": "initial"}, layout=W.Layout(width="330px"))
        self.btn_mlflow = W.Button(description="Registrar no MLflow", icon="database", button_style="primary",
                                   layout=W.Layout(width="auto", min_width="200px"))
        self.btn_mlflow.on_click(self._on_mlflow)
        self.out_mlflow_status = W.HTML()
        card_mlflow = W.VBox([
            W.HTML("<div class='survui-h'>Registrar o run no MLflow</div>"),
            W.HTML("<div class='survui-legend'>Parâmetros (método, horizonte, grupo, features, "
                   "ajustes), métricas (PD 12m e lifetime por grupo, erro do backtest), as curvas em "
                   "CSV, o JSON do modelo e as figuras, via <code>log_lifetime_pd</code>. Exige "
                   "<code>mlflow</code> instalado e o <i>tracking</i> configurado.</div>"),
            W.HBox([self.tx_mlflow_exp, self.tx_mlflow_run],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.btn_mlflow], layout=W.Layout(align_items="center")),
            self.out_mlflow_status,
        ])
        card_mlflow.add_class("survui-card")

        # --- config JSON ------------------------------------------------------------------
        self.btn_cfg_show = W.Button(description="Ver JSON da sessão", icon="code",
                                     layout=W.Layout(width="auto", min_width="190px"))
        self.btn_cfg_show.on_click(self._on_cfg_show)
        self.tx_cfg_path = W.Text(value=f"{base}.json", description="arquivo:",
                                  style={"description_width": "initial"}, layout=W.Layout(width="380px"))
        self.btn_cfg_save = W.Button(description="Salvar", icon="save", button_style="success",
                                     layout=W.Layout(width="auto", min_width="120px"))
        self.btn_cfg_save.on_click(self._on_cfg_save)
        self.btn_cfg_load = W.Button(description="Carregar", icon="upload",
                                     layout=W.Layout(width="auto", min_width="130px"))
        self.btn_cfg_load.on_click(self._on_cfg_load)
        self.btn_cfg_apply = W.Button(description="Aplicar o JSON abaixo", icon="check",
                                      layout=W.Layout(width="auto", min_width="200px"))
        self.btn_cfg_apply.on_click(self._on_cfg_apply)
        self.out_cfg_status = W.HTML()
        self.ta_config_json = W.Textarea(placeholder="o JSON da SurvivalConfig aparece aqui",
                                         layout=W.Layout(width="99%", height="220px"))
        card_config = W.VBox([
            W.HTML("<div class='survui-h'>Configuração do estudo (JSON): reprodutibilidade</div>"),
            W.HTML("<div class='survui-help'><div class='ttl'>Por que salvar isto</div>"
                   "A curva só é reproduzível se a <b>configuração que a gerou</b> viajar junto: "
                   "mapeamento de colunas, grupo, motor, features, família da cauda e junção, "
                   "alvos de calibração, ciclo, partição e horizontes de validação. Este JSON é a "
                   "<code>SurvivalConfig</code>, o mesmo objeto que <code>run_survival_study</code> "
                   "consome fora do notebook. <b>Carregar</b> repõe os controles; rode o estudo "
                   "completo para reproduzir.</div>"),
            W.HBox([self.btn_cfg_show, self.btn_cfg_apply],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.tx_cfg_path, self.btn_cfg_save, self.btn_cfg_load],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_cfg_status, self.ta_config_json,
        ])
        card_config.add_class("survui-card")

        # --- tabelas CSV ---------------------------------------------------------------------
        self.dd_exp_tabela = W.Dropdown(options=[(rot, ch) for ch, rot in self._TABELAS_EXPORT],
                                        value="curvas", description="tabela:",
                                        style={"description_width": "initial"},
                                        layout=W.Layout(width="420px"))
        self.dd_exp_fmt = W.Dropdown(options=[("CSV: ponto decimal, vírgula", "csv"),
                                              ("CSV: vírgula decimal, ponto e vírgula", "csv_br"),
                                              ("TSV: colar no Excel", "tsv")],
                                     value="csv", description="formato:",
                                     style={"description_width": "initial"}, layout=W.Layout(width="320px"))
        self.btn_exp_mostrar = W.Button(description="Mostrar para copiar", icon="clipboard",
                                        layout=W.Layout(width="auto", min_width="190px"))
        self.btn_exp_mostrar.on_click(self._on_exp_mostrar)
        self.tx_exp_path = W.Text(value=f"{base}_curvas.csv", description="arquivo:",
                                  style={"description_width": "initial"}, layout=W.Layout(width="380px"))
        self.btn_exp_salvar = W.Button(description="Salvar CSV", icon="download", button_style="primary",
                                       layout=W.Layout(width="auto", min_width="150px"))
        self.btn_exp_salvar.on_click(self._on_exp_salvar)
        self.out_exp_tab_status = W.HTML()
        self.ta_export_tabela = W.Textarea(placeholder="a tabela escolhida aparece aqui (Ctrl+A, Ctrl+C)",
                                           layout=W.Layout(width="99%", height="200px"))
        self.dd_exp_tabela.observe(self._on_exp_tabela_change, names="value")
        card_tabelas = W.VBox([
            W.HTML("<div class='survui-h'>Tabelas em CSV</div>"),
            W.HBox([self.dd_exp_tabela, self.dd_exp_fmt, self.btn_exp_mostrar],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.tx_exp_path, self.btn_exp_salvar],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_exp_tab_status, self.ta_export_tabela,
        ])
        card_tabelas.add_class("survui-card")

        card_apply = W.VBox([
            W.HTML("<div class='survui-h'>Aplicar na carteira</div>"),
            W.HTML("<div class='survui-help'><div class='ttl'>Fora da interface</div>"
                   "<code>ui.apply(carteira, age_col='idade', term_col='prazo')</code> devolve a "
                   "carteira com <code>pd_12m</code>, <code>pd_lifetime</code> e as marginais por "
                   "horizonte (a linha ``i``, coluna ``t`` é a probabilidade de quebrar exatamente em "
                   "``t`` vista de hoje). A idade desloca o início da curva; o prazo remanescente "
                   "zera as marginais além dele. É o insumo direto de "
                   "<code>ecl_table(carteira, model=ui.model_, ...)</code>.</div>"),
        ])
        card_apply.add_class("survui-card")
        self._render_export_estado()
        return (card_estado, card_model, card_mlflow, card_config, card_tabelas, card_apply)

    def _nome_arquivo(self) -> str:
        tx = getattr(self, "tx_nome", None)
        bruto = _sem_acento(tx.value if tx is not None else self.study_name)
        return re.sub(r"[^a-z0-9_.-]+", "_", bruto).strip("_") or "estudo"

    def _estado_blocos(self) -> list:
        def _b(bloco, nivel, veredito, detalhe):
            return {"bloco": bloco, "nivel": nivel, "veredito": veredito, "detalhe": detalhe}

        blocos = []
        if self.panel is None:
            blocos.append(_b("Painel", "bad", "sem dados", "monte o painel na aba Painel"))
        else:
            blocos.append(_b("Painel", "ok", f"{self.panel.n_contracts} contratos",
                             f"{len(self.panel)} observações · idade máx. {self.panel.max_age}"))
        if self.oot_ is None:
            blocos.append(_b("Partição", "warn", "sem OOT", "a validação será in-sample"))
        else:
            blocos.append(_b("Partição", "ok", f"OOT {self.oot_.n_contracts} contratos",
                             f"DES {self.des_.n_contracts} contratos"))
        if self.model_ is None:
            blocos.append(_b("Curva do estudo", "warn", "nenhuma adotada",
                             "adote nas abas Kaplan-Meier, Hazard ou Paramétrico"))
        else:
            ajustes = []
            if self._calib_targets:
                ajustes.append("calibrada")
            if self._cycle:
                ajustes.append("ciclo")
            blocos.append(_b("Curva do estudo", "ok", f"{self.model_source_} · {len(self.model_.curves_)} curva(s)",
                             f"horizonte {self.model_.horizon}" + (" · " + ", ".join(ajustes) if ajustes else "")))
        if self.val_blocks_ is None:
            blocos.append(_b("Validação", "na", "não rodada", "aba Validação"))
        else:
            ruins = [x["bloco"] for x in self.val_blocks_ if x["nivel"] == "bad"]
            blocos.append(_b("Validação", "bad" if ruins else "ok",
                             f"{len(ruins)} reprovado(s)" if ruins else "sem reprovações",
                             f"em {self.validated_on_}" + (": " + ", ".join(ruins) if ruins else "")))
        if self.study_ is not None:
            blocos.append(_b("Estudo completo", "ok", self.study_.config.name,
                             f"{len(self.study_.steps)} etapas · {self._fmt_dur(self._study_secs or 0)}"))
        return blocos

    def _render_export_estado(self):
        if getattr(self, "out_exp_estado", None) is None:
            return
        self.out_exp_estado.value = self._placar_html(self._estado_blocos())

    @staticmethod
    def _existe(caminho: str) -> bool:
        import os

        return bool(caminho) and os.path.exists(caminho)

    def _grava_com_confirmacao(self, btn, status, caminho, acao):
        if not self._existe(caminho):
            acao()
            return
        if not getattr(btn, "_cc_armed", 0.0):
            status.value = self._notice(
                f"O arquivo <code>{_esc(caminho)}</code> <b>já existe</b>. Clique de novo "
                "(<i>Confirmar?</i>) para sobrescrever.")
        self._confirm_twice(btn, acao)

    # ------------------------------------------------------------------ modelo JSON
    def _on_model_save(self, b):
        if self.model_ is None:
            self.out_model_status.value = self._notice("Nenhuma curva adotada para salvar.")
            return
        caminho = str(self.tx_model_path.value or "").strip()
        if not caminho:
            self.out_model_status.value = self._notice("Informe o <b>arquivo</b> de destino.")
            return
        self._grava_com_confirmacao(self.btn_model_save, self.out_model_status, caminho,
                                    lambda: self._salva_modelo(caminho))

    def _salva_modelo(self, caminho):
        with self._busy(self.btn_model_save, status=self.out_model_status, msg="gravando…"):
            try:
                self.model_.to_json(caminho)
            except Exception as exc:  # noqa: BLE001
                self.out_model_status.value = self._notice(f"Não foi possível gravar: {_esc(exc)}")
                return
        self.model_path_ = caminho
        self.out_model_status.value = self._ok_msg(
            f"Modelo gravado em <code>{_esc(caminho)}</code>. Reconstrua com "
            "<code>LifetimePD.from_json(caminho)</code>.")
        self._log(f"[modelo] gravado em {caminho}.")
        self._render_export_estado()

    def _on_model_load(self, b):
        caminho = str(self.tx_model_path.value or "").strip()
        if not caminho:
            self.out_model_status.value = self._notice("Informe o <b>arquivo</b> a carregar.")
            return
        with self._busy(self.btn_model_load, status=self.out_model_status, msg="lendo…"):
            try:
                lt = LifetimePD.from_json(caminho)
            except FileNotFoundError:
                self.out_model_status.value = self._notice(f"Arquivo não encontrado: <code>{_esc(caminho)}</code>.")
                return
            except Exception as exc:  # noqa: BLE001
                self.out_model_status.value = self._notice(f"Não foi possível carregar: {_esc(exc)}")
                return
        self._adopt(lt, "json")
        self.out_model_status.value = self._ok_msg(
            f"Modelo de <code>{_esc(caminho)}</code> adotado como curva do estudo "
            f"({len(lt.curves_)} curva(s), horizonte {lt.horizon}).")

    # ------------------------------------------------------------------ MLflow
    def _on_mlflow(self, b):
        if self.model_ is None:
            self.out_mlflow_status.value = self._notice("Nenhuma curva adotada para registrar.")
            return
        with self._busy(self.btn_mlflow, status=self.out_mlflow_status, msg="registrando no MLflow…"):
            try:
                import mlflow  # noqa: F401
            except ImportError:
                self.out_mlflow_status.value = self._notice(
                    "<b>MLflow não está instalado</b> neste ambiente (<code>pip install mlflow</code>). "
                    "O JSON do modelo e da configuração não dependem dele.")
                self._log("[mlflow] pacote ausente; registro não realizado.")
                return
            cfg = None
            with suppress(Exception):
                cfg = self.to_config()
            params = {"origem_curva": self.model_source_ or "—",
                      "calibrada": bool(self._calib_targets), "ciclo": bool(self._cycle),
                      "validado_em": self.validated_on_ or "—"}
            if cfg is not None:
                params.update({"tail": cfg.tail, "distribution": cfg.distribution, "split": cfg.split})
            tags = {"estudo": str(self.tx_nome.value or self.study_name), "interface": "SurvivalUI"}
            try:
                from ..ecl import tracking

                rid = tracking.log_lifetime_pd(
                    self.model_, backtest=self.backtest_, params=params, tags=tags,
                    experiment=(str(self.tx_mlflow_exp.value).strip() or None),
                    run_name=(str(self.tx_mlflow_run.value).strip() or self._nome_arquivo()))
            except Exception as exc:  # noqa: BLE001
                self.out_mlflow_status.value = self._notice(
                    f"Não foi possível registrar ({type(exc).__name__}): {_esc(str(exc)[:300])}<br>"
                    "Confira o <b>tracking</b> (<code>MLFLOW_TRACKING_URI</code>, ou rode no "
                    "Databricks) e a permissão de escrita no experimento.")
                self._log(f"[mlflow] ERRO: {type(exc).__name__}: {exc}")
                return
        self.mlflow_run_id_ = rid
        self.out_mlflow_status.value = self._ok_msg(
            f"Run registrado: <code>run_id = {_esc(rid)}</code> (também em <code>ui.mlflow_run_id_</code>)"
            + ("; o backtest foi junto." if self.backtest_ is not None else "; sem backtest no estado."))
        self._log(f"[mlflow] run_id = {rid}")
        self._render_export_estado()

    # ------------------------------------------------------------------ config JSON
    def _config_json(self) -> str:
        import json

        return json.dumps(self.to_config().to_dict(), indent=2, ensure_ascii=False, default=str)

    def _on_cfg_show(self, b):
        try:
            self.ta_config_json.value = self._config_json()
        except Exception as exc:  # noqa: BLE001
            self.out_cfg_status.value = self._notice(f"{_esc(exc)}")
            return
        self.out_cfg_status.value = self._ok_msg(
            "Configuração corrente serializada; o mesmo objeto está em <code>ui.to_config()</code>.")

    def _on_cfg_save(self, b):
        caminho = str(self.tx_cfg_path.value or "").strip()
        if not caminho:
            self.out_cfg_status.value = self._notice("Informe o <b>arquivo</b> de destino.")
            return
        self._grava_com_confirmacao(self.btn_cfg_save, self.out_cfg_status, caminho,
                                    lambda: self._salva_config(caminho))

    def _salva_config(self, caminho):
        with self._busy(self.btn_cfg_save, status=self.out_cfg_status, msg="gravando…"):
            try:
                texto = self._config_json()
                with open(caminho, "w", encoding="utf-8") as fh:
                    fh.write(texto)
            except Exception as exc:  # noqa: BLE001
                self.out_cfg_status.value = self._notice(f"Não foi possível gravar: {_esc(exc)}")
                return
        self.ta_config_json.value = texto
        self.out_cfg_status.value = self._ok_msg(f"Configuração gravada em <code>{_esc(caminho)}</code>.")
        self._log(f"[config] gravada em {caminho}.")

    def _on_cfg_load(self, b):
        import json

        caminho = str(self.tx_cfg_path.value or "").strip()
        if not caminho:
            self.out_cfg_status.value = self._notice("Informe o <b>arquivo</b> a carregar.")
            return
        with self._busy(self.btn_cfg_load, status=self.out_cfg_status, msg="lendo…"):
            try:
                with open(caminho, "r", encoding="utf-8") as fh:
                    dados = json.load(fh)
                self.ta_config_json.value = json.dumps(dados, indent=2, ensure_ascii=False)
                self.from_config(dados)
            except FileNotFoundError:
                self.out_cfg_status.value = self._notice(f"Arquivo não encontrado: <code>{_esc(caminho)}</code>.")
                return
            except Exception as exc:  # noqa: BLE001
                self.out_cfg_status.value = self._notice(f"Não foi possível carregar: {_esc(exc)}")
                return
        self.out_cfg_status.value = self._ok_msg(
            f"Configuração de <code>{_esc(caminho)}</code> aplicada; rode o <b>estudo completo</b> "
            "para reproduzir o resultado.")

    def _on_cfg_apply(self, b):
        import json

        texto = str(self.ta_config_json.value or "").strip()
        if not texto:
            self.out_cfg_status.value = self._notice("A caixa está vazia.")
            return
        try:
            dados = json.loads(texto)
            if not isinstance(dados, dict):
                raise TypeError("o JSON precisa ser um objeto com os campos da SurvivalConfig.")
            self.from_config(dados)
        except Exception as exc:  # noqa: BLE001
            self.out_cfg_status.value = self._notice(f"JSON inválido: {_esc(exc)}")
            return
        self.out_cfg_status.value = self._ok_msg("JSON aplicado à interface.")

    # ------------------------------------------------------------------ tabelas
    def _curves_long(self) -> pd.DataFrame:
        m = self.model_
        frames = []
        for kind in ("hazard", "marginal", "cumulative", "survival"):
            f = curve_frame(m.curves_, kind=kind)
            f.columns = [self._rot(c) if c == GLOBAL else str(c) for c in m.curves_]
            longo = f.reset_index().melt(id_vars="horizonte", var_name="grupo", value_name=kind)
            frames.append(longo.set_index(["grupo", "horizonte"]))
        return pd.concat(frames, axis=1).reset_index()

    def _tabela_export(self, chave):
        if chave == "curvas":
            if self.model_ is None:
                raise RuntimeError("nada a exportar: adote uma curva antes.")
            return self._curves_long(), "curvas"
        if chave == "tabela_vida":
            if self.life_table_ is None:
                raise RuntimeError("nada a exportar: estime as curvas na aba Kaplan-Meier antes.")
            return self.life_table_.copy(), "tabela_vida"
        if chave == "logrank":
            if self.logrank_ is None:
                raise RuntimeError("nada a exportar: rode o Kaplan-Meier por grupo antes.")
            return self.logrank_["grupos"].copy(), "logrank"
        if chave == "coeficientes":
            if self.hazard_lt_ is None:
                raise RuntimeError("nada a exportar: ajuste o hazard na aba Hazard antes.")
            frames = []
            for rot, mh in self.hazard_lt_.hazard_models_.items():
                cf = mh.coef_frame()
                cf.insert(0, "grupo", self._rot(rot))
                frames.append(cf)
            return pd.concat(frames, ignore_index=True), "coeficientes"
        if chave == "parametrico":
            if self.param_rank_ is None:
                raise RuntimeError("nada a exportar: ajuste as famílias na aba Paramétrico antes.")
            return self.param_rank_.copy(), "parametrico"
        if chave == "backtest":
            if self.backtest_ is None:
                raise RuntimeError("nada a exportar: rode a validação antes.")
            return self.backtest_.copy(), "backtest"
        if chave == "discriminacao":
            if self.discrimination_ is None:
                raise RuntimeError("nada a exportar: rode a validação antes.")
            return self.discrimination_.copy(), "discriminacao"
        if chave == "decil":
            if self.decile_ is None:
                raise RuntimeError("nada a exportar: rode a validação antes.")
            return self.decile_.copy(), "decil"
        raise RuntimeError(f"tabela desconhecida: {chave!r}.")

    def _on_exp_tabela_change(self, change):
        with suppress(Exception):
            self.tx_exp_path.value = f"{self._nome_arquivo()}_{change['new']}.csv"

    def _exp_sep(self):
        return {"tsv": ("\t", "."), "csv": (",", "."), "csv_br": (";", ",")}[self.dd_exp_fmt.value]

    def _on_exp_mostrar(self, b):
        try:
            df, _ = self._tabela_export(self.dd_exp_tabela.value)
        except RuntimeError as exc:
            self.out_exp_tab_status.value = self._notice(str(exc))
            self.ta_export_tabela.value = ""
            return
        sep, dec = self._exp_sep()
        self.ta_export_tabela.value = df.to_csv(sep=sep, index=False, decimal=dec)
        self.out_exp_tab_status.value = self._ok_msg(
            f"{len(df)} linha(s) × {df.shape[1]} coluna(s) prontas (Ctrl+A, Ctrl+C).")

    def _on_exp_salvar(self, b):
        caminho = str(self.tx_exp_path.value or "").strip()
        if not caminho:
            self.out_exp_tab_status.value = self._notice("Informe o <b>arquivo</b> de destino.")
            return
        try:
            df, _ = self._tabela_export(self.dd_exp_tabela.value)
        except RuntimeError as exc:
            self.out_exp_tab_status.value = self._notice(str(exc))
            return
        self._grava_com_confirmacao(self.btn_exp_salvar, self.out_exp_tab_status, caminho,
                                    lambda: self._salva_tabela(df, caminho))

    def _salva_tabela(self, df, caminho):
        sep, dec = self._exp_sep()
        with self._busy(self.btn_exp_salvar, status=self.out_exp_tab_status, msg="gravando…"):
            try:
                df.to_csv(caminho, sep=sep, index=False, decimal=dec, encoding="utf-8-sig")
            except Exception as exc:  # noqa: BLE001
                self.out_exp_tab_status.value = self._notice(f"Não foi possível gravar: {_esc(exc)}")
                return
        self.out_exp_tab_status.value = self._ok_msg(
            f"{len(df)} linha(s) gravadas em <code>{_esc(caminho)}</code>.")
        self._log(f"[exportar] {len(df)} linha(s) gravadas em {caminho}.")

    def _clear_exportar_outputs(self):
        for w in ("out_model_status", "out_mlflow_status", "out_cfg_status", "out_exp_tab_status",
                  "ta_config_json", "ta_export_tabela", "out_study_status", "out_study_progress",
                  "out_study_resumo"):
            widget = getattr(self, w, None)
            if widget is not None:
                widget.value = ""
        self._study_steps, self._study_secs = [], None
        self._invalidate_exportar("os dados mudaram")

    def _invalidate_exportar(self, motivo="a curva mudou"):
        tinha = self.mlflow_run_id_ is not None or self.model_path_ is not None
        self.mlflow_run_id_ = self.model_path_ = None
        if getattr(self, "out_exp_notice", None) is None:
            return
        for w in ("out_model_status", "out_mlflow_status", "out_exp_tab_status"):
            getattr(self, w).value = ""
        self.ta_export_tabela.value = ""
        self.out_exp_notice.value = self._notice(
            f"⚠️ <b>Saídas desatualizadas</b>: {motivo}. O modelo e o run gravados antes descrevem "
            "a curva <b>ANTERIOR</b>; grave-os de novo.") if tinha else ""
        self._render_export_estado()

    # ==================================================================
    # Configuração declarativa
    # ==================================================================
    def to_config(self, method: Optional[str] = None) -> SurvivalConfig:
        """A :class:`SurvivalConfig` que descreve a tela.

        ``method`` força o motor; por padrão usa a origem da curva adotada (ou o
        seletor do estudo completo, quando nenhuma foi adotada)."""
        cols = self._cols_from_widgets() if self.df is not None else dict(self.cols)
        metodo = method or (self.model_source_ if self.model_source_ in STUDY_METHODS
                            else self.dd_study_method.value)
        feats = list(self.sel_hz_features.value)
        if metodo == "hazard" and not feats:
            feats = list(self._init_features)
        if metodo == "hazard":
            by = self.dd_hz_by.value
            horizon = int(self.sl_hz_horizon.value)
        elif metodo == "parametric":
            by = self.dd_par_by.value
            horizon = int(self.sl_par_horizon.value)
        else:
            by = self.dd_km_by.value
            horizon = int(self.sl_km_horizon.value)
        alvos = {}
        if self._calib_targets:
            alvos = {(GLOBAL if k == GLOBAL else str(k)): float(v) for k, v in self._calib_targets.items()}
        cyc = self._cycle or {}
        split = self.dd_split.value
        return SurvivalConfig(
            name=str(self.tx_nome.value or self.study_name),
            id_col=cols["id_col"] or "id_contrato", date_col=cols["date_col"] or "dt_ref",
            default_col=cols["default_col"] or "default", age_col=cols["age_col"],
            origin_col=cols["origin_col"], term_col=cols["term_col"],
            segment_col=cols["segment_col"], exposure_col=cols["exposure_col"],
            freq=str(cols.get("freq") or "M"),
            method=metodo, by=by, horizon=horizon, from_age=int(self.sl_km_from.value),
            min_at_risk=int(self.sl_par_min_risk.value), alpha=float(self.fl_km_alpha.value),
            weighted=bool(self.cb_km_weighted.value),
            features=feats if metodo == "hazard" else list(feats),
            baseline=self.dd_hz_baseline.value, link=self.dd_hz_link.value,
            C=float(self.fl_hz_C.value), n_knots=int(self.sl_hz_knots.value),
            max_age=(int(self.sl_hz_max_age.value) or None),
            tail=("parametric" if self.dd_par_mode.value == "splice" else "flat")
            if metodo != "parametric" else "parametric",
            distribution=self.dd_par_choice.value,
            junction=(int(self.sl_par_junction.value) or None),
            match_level=bool(self.cb_par_match.value),
            calibrate=alvos, z=cyc.get("z"), rho=float(cyc.get("rho", self.fl_rho.value)),
            decay=cyc.get("decay"), mode=cyc.get("mode", self.dd_mode.value),
            split=split, split_value=(self.tx_split_date.value or None) if split in ("origin", "observation") else None,
            split_col=self.dd_split_col.value if split == "column" else None,
            oot_value=str(self.tx_split_oot.value or "OOT"),
            backtest_horizons=self._parse_ints(self.tx_val_horizons.value),
            decile_horizon=int(self.sl_val_dec_h.value), n_bins=int(self.sl_val_bins.value),
        )

    def from_config(self, config) -> "SurvivalUI":
        """Repõe os controles a partir de uma :class:`SurvivalConfig` (ou dict)."""
        cfg = config if isinstance(config, SurvivalConfig) else SurvivalConfig.from_dict(config)
        self.tx_nome.value = cfg.name
        if self.df is not None:
            for k in ("id_col", "date_col", "default_col", "age_col", "origin_col", "term_col",
                      "segment_col", "exposure_col"):
                v = getattr(cfg, k)
                self.cols[k] = v if (v in self.df.columns) else None
            self.cols["freq"] = cfg.freq
            self._sync_column_widgets()
        self.dd_study_method.value = cfg.method
        grupos = [v for _, v in self.dd_km_by.options]
        by = cfg.by if cfg.by in grupos else None
        for dd in (self.dd_km_by, self.dd_hz_by, self.dd_par_by):
            dd.value = by
        for w in (self.sl_km_horizon, self.sl_hz_horizon, self.sl_par_horizon):
            w.value = int(cfg.horizon)
        self.sl_km_from.value = int(cfg.from_age)
        self.fl_km_alpha.value = float(cfg.alpha)
        self.fl_val_alpha.value = float(cfg.alpha)
        if not self.cb_km_weighted.disabled:
            self.cb_km_weighted.value = bool(cfg.weighted)
        self.sl_par_min_risk.value = int(cfg.min_at_risk)
        self.sl_km_min_risk.value = int(cfg.min_at_risk)
        if cfg.method in ("km", "vintage"):
            self.dd_km_method.value = cfg.method
        disponiveis = [v for _, v in self.sel_hz_features.options]
        self.sel_hz_features.value = tuple(f for f in cfg.features if f in disponiveis)
        self.dd_hz_baseline.value = cfg.baseline if cfg.baseline in BASELINES else "spline"
        self.dd_hz_link.value = cfg.link if cfg.link in HAZARD_LINKS else "logit"
        self.fl_hz_C.value = float(cfg.C)
        self.sl_hz_knots.value = int(cfg.n_knots)
        self.sl_hz_max_age.value = int(cfg.max_age or 0)
        self.dd_par_mode.value = "pure" if cfg.method == "parametric" else (
            "splice" if cfg.tail == "parametric" else "splice")
        with suppress(Exception):
            self.dd_par_choice.value = cfg.distribution
        self.sl_par_junction.value = int(cfg.junction or 0)
        self.cb_par_match.value = bool(cfg.match_level)
        if cfg.z is not None:
            self.fl_z.value = float(cfg.z)
        self.fl_rho.value = float(cfg.rho)
        self.fl_decay.value = float(cfg.decay or 0.0)
        self.dd_mode.value = cfg.mode
        self.dd_split.value = cfg.split
        if cfg.split in ("origin", "observation") and cfg.split_value:
            self.tx_split_date.value = str(cfg.split_value)
        if cfg.split == "column":
            with suppress(Exception):
                self.dd_split_col.value = cfg.split_col
        self.tx_split_oot.value = str(cfg.oot_value)
        self.tx_val_horizons.value = ",".join(str(h) for h in cfg.backtest_horizons)
        self.sl_val_dec_h.value = int(cfg.decile_horizon)
        self.sl_val_bins.value = int(cfg.n_bins)
        # alvos de calibração: só preenchem os campos se já houver curva adotada
        if cfg.calibrate and self._calib_inputs:
            for rot, w in self._calib_inputs.items():
                chave = GLOBAL if rot == GLOBAL else str(rot)
                if chave in cfg.calibrate:
                    w.value = f"{cfg.calibrate[chave]:.4f}"
        self._sync_split_fields()
        self._sync_km_fields()
        self._refresh_bar()
        return self

    # ==================================================================
    # Estudo completo
    # ==================================================================
    def _study_prog(self, key, label, status, detail=""):
        self._prog(self._study_steps, self.out_study_progress, "Progresso do estudo",
                   key, label, status, detail)

    def _on_run_study(self, b):
        if self.panel is None:
            self.out_study_status.value = self._notice("Monte o painel antes de rodar o estudo.")
            return
        metodo = self.dd_study_method.value
        try:
            cfg = self.to_config(method=metodo)
        except Exception as exc:  # noqa: BLE001
            self.out_study_status.value = self._notice(f"Configuração inválida: {_esc(exc)}")
            return
        if cfg.split in ("origin", "observation") and not cfg.split_value:
            self.out_study_status.value = self._notice(
                "A partição escolhida exige a <b>data de corte</b> na aba Painel.")
            return
        import time

        self._study_steps = []
        self.out_study_progress.value = ""
        self.out_study_resumo.value = ""
        ini = time.monotonic()
        with self._busy(self.btn_run_study, status=self.out_study_status, msg="rodando o estudo…"):
            try:
                res = run_survival_study(self.panel, cfg, progress=self._study_prog)
            except Exception as exc:  # noqa: BLE001
                self._prog_erro(self._study_steps, self.out_study_progress, "Progresso do estudo", exc)
                self.out_study_status.value = self._notice(f"O estudo falhou: {_esc(exc)}")
                self._log(f"[estudo] ERRO: {type(exc).__name__}: {exc}")
                return
        self._study_secs = time.monotonic() - ini
        self._adota_estudo(res, cfg)
        self.out_study_status.value = self._ok_msg(
            f"Estudo <b>{_esc(cfg.name)}</b> concluído em {self._fmt_dur(self._study_secs)}: "
            f"curva {cfg.method} ({len(res.model.curves_)} curva(s)), validação em <b>{res.validated_on}</b>.")
        self._log(f"[estudo] concluído ({cfg.method}) em {self._fmt_dur(self._study_secs)}.")

    def _adota_estudo(self, res, cfg: SurvivalConfig):
        """Espalha o resultado do estudo pelas abas."""
        self.study_ = res
        self.des_, self.oot_ = res.panel_des, res.panel_oot
        self._render_split()
        # KM
        self.life_table_ = res.life_table
        self.logrank_ = res.logrank
        self.pairwise_ = None
        partes = {GLOBAL: res.panel_des} if cfg.by is None else res.panel_des.by(cfg.by)
        curvas, tabelas = {}, {}
        for rot, parte in partes.items():
            with suppress(ValueError):
                c, t = kaplan_meier(parte, from_age=cfg.from_age, alpha=cfg.alpha,
                                    weighted=cfg.weighted, return_table=True,
                                    label="" if rot == GLOBAL else str(rot))
                curvas[rot], tabelas[rot] = c, t
        self.km_curves_, self.km_tables_ = curvas, tabelas
        self._render_km()
        # Hazard
        if cfg.method == "hazard":
            self.hazard_lt_ = res.model_base
            self.ph_ = res.ph
            self._render_hazard()
            self._render_ph()
        # Paramétrico
        if res.parametric:
            linhas = []
            for rot, m in res.parametric.items():
                s = m.summary()
                s.insert(0, "grupo", self._rot(rot))
                s.insert(1, "posicao", 1)
                s["delta_aic"] = 0.0
                linhas.append(s)
            self.param_rank_ = pd.concat(linhas, ignore_index=True)
            self.param_models_ = {rot: {m.distribution: m} for rot, m in res.parametric.items()}
            self.dd_par_choice.options = [(DISTRIBUTION_LABELS[cfg.distribution], cfg.distribution)]
            self.dd_par_choice.value = cfg.distribution
            self.dd_par_show.options = [(self._rot(r), r) for r in self.param_models_]
            self.dd_par_show.value = next(iter(self.param_models_))
            self._render_param()
        # a curva e os ajustes
        self.model_base_ = res.model_base
        self.model_source_ = "estudo" if cfg.method not in STUDY_METHODS else cfg.method
        self._calib_targets = None
        if cfg.calibrate:
            alvos = {}
            for k, v in cfg.calibrate.items():
                for c in res.model_base.curves_:
                    if c == k or str(c) == str(k) or (k in ("", "global") and c == GLOBAL):
                        alvos[c] = float(v)
            self._calib_targets = alvos or None
        self._cycle = ({"z": float(cfg.z), "rho": float(cfg.rho), "decay": cfg.decay, "mode": cfg.mode}
                       if cfg.z is not None else None)
        self.model_ = res.model
        # validação
        self.backtest_, self.discrimination_ = res.backtest, res.discrimination
        self.decile_, self.c_index_ = res.decile, res.c_index
        self.validated_on_ = res.validated_on
        self.val_blocks_ = self._val_blocks(cfg.alpha)
        self._render_calib()
        self._render_val()
        self.mlflow_run_id_ = self.model_path_ = None
        self.out_exp_notice.value = ""
        self._refresh_all()
        # resumo
        s = res.summary().copy()
        s["grupo"] = s["grupo"].map(self._rot)
        pct = [c for c in s.columns if c.startswith("pd_") or c == "pct_dentro_do_ic"]
        etapas = " · ".join(f"{e} {self._fmt_dur(t)}" for e, t in res.steps)
        self.out_study_resumo.value = (
            f"<div class='survui-legend'>Etapas: {etapas}</div>" + self._df_html(s, pct_cols=pct))

    # ==================================================================
    # Tema, keepalive, barra, display
    # ==================================================================
    def _on_dark(self, change):
        dark = bool(change["new"])
        if dark:
            self.panel_w.add_class("dark")
            self.cb_dark.description = "☀ Tema claro"
        else:
            self.panel_w.remove_class("dark")
            self.cb_dark.description = "🌙 Tema escuro"
        self._repinta_figuras(dark)

    def _desliga_keepalive(self, msg):
        self._suspend_ka = True
        self.cb_keepalive.value = False
        self._suspend_ka = False
        self.cb_keepalive.description = "☕ Manter cluster ativo"
        self._log(f"[keepalive] {msg}")

    def _on_keepalive(self, change):
        if change["new"]:
            try:
                from ...utils.keepalive import ClusterKeepAlive

                if self._keepalive is None:
                    self._keepalive = ClusterKeepAlive(interval_seconds=120)
                if not self._keepalive.has_spark():
                    self._desliga_keepalive("nenhuma SparkSession ativa; recurso só funciona no "
                                            "Databricks (ou com Spark local).")
                    return
                self._keepalive.start()
            except Exception as exc:  # noqa: BLE001
                self._desliga_keepalive(f"não foi possível ligar ({type(exc).__name__}): {exc}")
                return
            self.cb_keepalive.description = "☕ Cluster ativo ✓"
            self._log("[keepalive] ligado.")
        else:
            if self._suspend_ka:
                return
            if self._keepalive is not None:
                with suppress(Exception):
                    self._keepalive.stop()
            self.cb_keepalive.description = "☕ Manter cluster ativo"
            self._log("[keepalive] desligado.")

    def _refresh_bar(self):
        self.banner.value = (
            "<div class='survui-banner'><div class='logo'>SV</div><div>"
            "<div class='t'>Análise de sobrevivência: PD lifetime</div>"
            "<div class='s'>Painel de contratos → curva (KM · hazard · paramétrica) → cauda → "
            "calibração e ciclo → validação → ECL</div></div></div>")
        pills = []
        if self.panel is None:
            pills.append(self._pill("sem painel", "yellow"))
        else:
            pills.append(self._pill(f"contratos: {self.panel.n_contracts}", "muted"))
            pills.append(self._pill(f"defaults: {int(self.panel.df[self.panel.default_col].sum())}", "muted"))
            pills.append(self._pill(f"idade máx.: {self.panel.max_age}", "muted"))
            pills.append(self._pill(f"OOT: {self.oot_.n_contracts} contratos" if self.oot_ is not None
                                    else "sem OOT", "green" if self.oot_ is not None else "yellow"))
        if self.model_ is None:
            pills.append(self._pill("curva: nenhuma adotada", "yellow"))
        else:
            pills.append(self._pill(f"curva: {self.model_source_} · {len(self.model_.curves_)} · H={self.model_.horizon}",
                                    "green"))
            if self._calib_targets:
                pills.append(self._pill("calibrada", "green"))
            if self._cycle:
                pills.append(self._pill(f"ciclo z={self._cycle['z']:+.1f}", "muted"))
        if self.val_blocks_:
            ruins = [b for b in self.val_blocks_ if b["nivel"] == "bad"]
            pills.append(self._pill(f"validação ({self.validated_on_}): "
                                    + (f"{len(ruins)} bloco(s) reprovado(s)" if ruins else "sem reprovações"),
                                    "red" if ruins else "green"))
        if self.study_ is not None:
            pills.append(self._pill(f"estudo completo: {_esc(self.study_.config.name)}", "green"))
        self.bar.value = "<div class='survui-bar'>" + "".join(pills) + "</div>"

    # ------------------------------------------------------------------ uso programático
    def apply(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Cola a curva vigente na carteira (atalho para ``ui.model_.apply``)."""
        if self.model_ is None:
            raise RuntimeError("nenhuma curva adotada; adote uma nas abas da interface.")
        return self.model_.apply(df, **kwargs)

    @property
    def panel_widget(self):
        """O widget raiz (para compor com outros painéis)."""
        return self.panel_w

    def _ipython_display_(self):
        _display(self.panel_w)

    def display(self):
        _display(self.panel_w)

    def __repr__(self) -> str:  # pragma: no cover
        n = self.panel.n_contracts if self.panel is not None else 0
        return (f"SurvivalUI(contratos={n}, curva={self.model_source_ or 'nenhuma'}, "
                f"validada={'sim' if self.val_blocks_ else 'não'})")


__all__ = ["SurvivalUI", "KINDS", "METODOS_ESTUDO"]
