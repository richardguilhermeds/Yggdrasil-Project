"""
ModelSegmenterUI
================
Camada interativa (ipywidgets) sobre o :class:`ModelSegmenter` — unifica a UI de
**classificação** e **regressão** via ``task_type``. Replica o estilo e os estudos
do ``LGDSegmenterUI`` (abas, IV, inversão, placar), porém para um fluxo orientado
a modelo:

* **Variáveis** — analisa cada variável (logodds/WoE, IV, inversão) e decide
  o que entra no modelo (incluir/categorizar; auto-seleção por IV/PSI/monotonia);
* **Análise de variáveis** — mergulho por variável: logodds por faixa,
  distribuição, inversão entre amostras/safras, série temporal e PSI por safra;
* **Modelo** — escolhe o algoritmo (Logística/Linear, RandomForest, ExtraTrees,
  GradientBoosting, HistGradientBoosting e — via extras — LightGBM/XGBoost/CatBoost)
  e treina (ou usa um modelo pré-ajustado); métricas por amostra, gráficos do
  modelo, fórmula (modelos lineares) e **SHAP**;
* **Ratings & Score** — segmenta o score em ratings (decis/quantil/árvore/optbin),
  com o número escolhido pelo usuário; tabela, badrate, distribuição e inversão
  entre ratings;
* **Validar & Exportar** — backtest por safra, PSI, **escorar uma base** (rating +
  valor previsto do alvo daquele rating, via :meth:`ModelSegmenter.rating_ruler`),
  exportar DataFrame rotulado, salvar/carregar e registrar no MLflow.

    from yggdrasil.credit_risk.model import ModelSegmenterUI
    ui = ModelSegmenterUI(df, target="target", task_type="classification",
                          sample_col="amostra", ref_sample="DES", date_col="dt_ref")
    ui
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd

try:
    import ipywidgets as W
    from IPython.display import clear_output, display
except Exception as e:  # pragma: no cover
    raise ImportError("Este módulo requer ipywidgets e IPython (Jupyter).") from e

from .segmenter import (ADVANCED_HYPERPARAMS, ALGORITHMS, BOOSTING_ALGORITHMS,
                        OPTUNA_SEARCH_SPACE, ModelSegmenter)

#: Ordem e rótulo de exibição dos hiperparâmetros na gaveta de "Ajuste do tuning
#: (Optuna)" — a união dos parâmetros de :data:`OPTUNA_SEARCH_SPACE`.
_SPACE_ORDER = ["C", "n_estimators", "max_iter", "iterations", "num_leaves",
                "max_depth", "depth", "min_samples_leaf", "learning_rate",
                "subsample", "colsample_bytree", "l2_regularization", "max_features"]
_SPACE_LABEL = {
    "C": "C (regularização)", "n_estimators": "n_estimators", "max_iter": "max_iter",
    "iterations": "iterations", "num_leaves": "num_leaves", "max_depth": "max_depth",
    "depth": "depth", "min_samples_leaf": "min_samples_leaf",
    "learning_rate": "learning_rate", "subsample": "subsample",
    "colsample_bytree": "colsample_bytree", "l2_regularization": "L2 (l2_regularization)",
    "max_features": "max_features",
}

#: Nomes do parâmetro de regularização L2 nos diferentes motores — todos são
#: expostos pela MESMA linha "L2" na UI (a gaveta de avançados).
_L2_PARAM_NAMES = ("l2_regularization", "reg_lambda", "l2_leaf_reg")


def _l2_param_for(algo: str):
    """Nome do parâmetro L2 do algoritmo (ou ``None`` se ele não tem um)."""
    for name in _L2_PARAM_NAMES:
        if name in ADVANCED_HYPERPARAMS.get(algo, ()):  # 1 por algoritmo
            return name
    return None

_CSS = """
<style>
/* SEM @import externo de fonts.googleapis.com: no Databricks o cluster costuma não
   ter egress, e um @import render-blocking no topo do <style> deixa a requisição
   pendente até o timeout de rede, adiando o 1º paint da UI (parece "não carrega").
   Usa-se a font-stack do sistema declarada em font-family abaixo (IBM Plex se
   instalada localmente; senão -apple-system/Segoe UI/Roboto...). */
.mseg { --ac:#3b4a63; --ac-deep:#27324a; --ac-soft:#eef1f5; --ac-border:#cdd5e0;
  --ink:#1f2733; --muted:#6b7480; --line:#e7e9ee;
  /* tokens semânticos (status, tabelas, realces): o HTML gerado no Python usa
     var(--...) em vez de hex — o tema escuro só redefine os tokens aqui */
  --ok-ink:#157a52; --ok-bg:#e7f5ee; --ok-tx:#137a3e;
  --warn-ink:#9a6f12; --warn-bg:#fbf3e0; --warn-tx:#9a6b00;
  --bad-ink:#b23a2a; --bad-bg:#fbe7e4; --bad-tx:#b3261e;
  --info-ink:#1f5fa8; --info-bg:#e7eef8; --sus-ink:#6b46c1; --sus-bg:#efe9fb;
  --code-ink:#0b63ce; --code-bg:#e7eef8;
  --gauge-ok:#2ea06b; --gauge-bad:#d64545; --gauge-track:#eef0f3;
  --strong-ink:#15324a; --body-ink:#3a4658; --sub-ink:#8a93a3;
  --faint-ink:#9aa3ad; --hair:#eef0f3; --tile-bg:#f7f8fa; --rule-bg:#fff;
  --help-bg:#f5f8fc; --help-line:#dbe5f1; --neutral-bg:#f1f3f5;
  --formula-bg:#0f1620;
  --tbl-line:#e1e5ec; --tbl-line-strong:#cdd5e0; --tbl-head-bg:#eef1f5;
  --tbl-head-ink:#27324a; --tbl-head-line:#b9c2d0; --tbl-zebra:#fafbfc;
  --tbl-hover:#eef3f8;
  --notice-bg:#fff8e6; --notice-border:#f0c36d; --notice-ink:#664d03;
  font-family:'IBM Plex Sans', -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  color:var(--ink); }
.mseg .mono { font-family:'IBM Plex Mono', ui-monospace, Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums; }
.mseg-banner { display:flex; align-items:center; gap:11px; background:#fff;
  border:1px solid var(--line); border-radius:13px; padding:11px 16px; margin-bottom:10px;
  box-shadow:0 1px 3px rgba(16,24,40,.08); }
.mseg-banner .logo { width:30px; height:30px; border-radius:9px; background:var(--ac);
  color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700;
  font-size:12px; flex:none; }
.mseg-banner .t { font-size:15px; font-weight:600; color:var(--ink); line-height:1.2; }
.mseg-banner .s { font-size:11.5px; color:var(--muted); margin-top:1px; }
.mseg-card { background:#fff; border:1px solid var(--line); border-radius:12px;
  padding:13px 15px; box-shadow:0 1px 3px rgba(16,24,40,.06); margin-bottom:11px;
  overflow-x:clip; }
.mseg-h { font-weight:600; font-size:11px; color:var(--muted); text-transform:uppercase;
  letter-spacing:.07em; margin-bottom:9px; }
.mseg-bar { background:#fff; border:1px solid var(--line); border-radius:11px;
  box-shadow:0 1px 3px rgba(16,24,40,.05); padding:8px 12px; overflow-x:auto; }
.pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:11.5px;
  font-weight:600; margin:2px 4px 2px 0; }
.pill-muted  { background:var(--ac-soft); color:var(--ac-deep); }
.pill-green  { background:var(--ok-bg); color:var(--ok-ink); }
.pill-yellow { background:var(--warn-bg); color:var(--warn-ink); }
.pill-red    { background:var(--bad-bg); color:var(--bad-ink); }
.mseg-legend { font-size:11px; color:var(--muted); margin:6px 0 2px; line-height:1.55; }
/* caixa de tutorial (algoritmo/parâmetros) e legenda de categorias */
.mseg-help { background:var(--help-bg); border:1px solid var(--help-line);
  border-left:3px solid var(--ac);
  border-radius:9px; padding:11px 14px; font-size:11.5px; color:var(--body-ink);
  line-height:1.62; margin-top:8px; }
.mseg-help .ttl { font-size:12px; font-weight:600; color:var(--ink); margin-bottom:5px; }
.mseg-help ul { margin:5px 0 0; padding-left:6px; list-style:none; }
.mseg-help li { margin:4px 0; }
.mseg-help .pname { font-family:'IBM Plex Mono', ui-monospace, monospace; font-weight:600;
  color:var(--code-ink); background:var(--code-bg); padding:1px 6px; border-radius:5px; }
.mseg-help .none { color:var(--muted); font-style:italic; }
.mseg-cat { display:inline-block; padding:2px 9px; border-radius:999px; font-size:11px;
  font-weight:600; }
.mseg-cat-keep { background:var(--ok-bg); color:var(--ok-ink); }
.mseg-cat-rev  { background:var(--warn-bg); color:var(--warn-ink); }
.mseg-cat-drop { background:var(--bad-bg); color:var(--bad-ink); }
.mseg-rule { margin-top:8px; padding:11px 13px; background:var(--rule-bg);
  border:1px solid var(--line); border-radius:8px; font-size:11px; }
.mseg-rule .rt { font-weight:600; color:var(--ink); margin-bottom:8px;
  font-family:'IBM Plex Mono', ui-monospace, monospace; }
.mseg-rule .rr { display:flex; align-items:baseline; gap:9px; margin:7px 0; line-height:1.65; }
.mseg-rule .rr .mseg-cat { flex:0 0 72px; text-align:center; }
.mseg-rule .rx { flex:1; color:var(--body-ink); }
.mseg-rule i { color:var(--muted); }
/* dicionário/guia de bolso das métricas (bloco recolhível) */
.mseg-guide { margin-top:10px; font-size:11.5px; }
.mseg-guide > summary { cursor:pointer; font-weight:600; color:var(--ac-deep);
  padding:8px 12px; background:var(--ac-soft); border:1px solid var(--ac-border);
  border-radius:8px; list-style:none; user-select:none; }
.mseg-guide > summary::-webkit-details-marker { display:none; }
.mseg-guide > summary::before { content:'▸ '; }
.mseg-guide[open] > summary::before { content:'▾ '; }
.mseg-guide > summary:hover { background:#e3e9f1; }
.mseg-guide .lg { margin:9px 2px 7px; color:var(--muted); line-height:1.75; }
.mseg-guide table td { text-align:left !important; vertical-align:top; }
.mseg-guide .mseg-cat { font-size:10px; }
.mseg-metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(96px,1fr)); gap:6px; }
.mseg-metric { background:var(--tile-bg); border:1px solid var(--hair); border-radius:9px;
  padding:7px 10px; }
.mseg-metric .k { font-size:10px; text-transform:uppercase; letter-spacing:.04em;
  color:var(--sub-ink); }
.mseg-metric .v { font-size:16px; font-weight:600; color:var(--ink); margin-top:2px; }
/* abas — estilo "segmented control" (pílulas), alinhado às UIs lgd/pd */
.mseg-tabs { margin-top:10px; border:none !important; box-shadow:none !important; }
/* respiro entre a barra de abas e os cards do conteúdo abaixo
   (!important vence a regra própria do ipywidgets p/ .widget-tab-contents);
   border/box-shadow:none remove a "caixa" padrão do Tab ao redor de tudo */
.mseg-tabs > .widget-tab-contents { padding:30px 2px 2px !important; background:transparent;
  border:none !important; box-shadow:none !important; }
.mseg-tabs .lm-TabBar.jupyter-widget-tab-nav,
.mseg-tabs .p-TabBar.jupyter-widget-tab-nav { border-bottom:1px solid var(--line) !important;
  padding-bottom:14px !important; margin-bottom:0 !important; box-shadow:none !important; }
.mseg-tabs .lm-TabBar-content, .mseg-tabs .p-TabBar-content { gap:7px;
  align-items:stretch; border:none; }
.mseg-tabs .lm-TabBar-tab, .mseg-tabs .p-TabBar-tab { font-size:13px;
  /* !important vence a regra de mesma especificidade do ipywidgets
     (flex/max-width: var(--jp-widgets-horizontal-tab-width)) que cortava o título) */
  min-width:max-content !important; max-width:none !important; flex:0 0 auto !important;
  margin:0 !important; padding:8px 16px !important;
  border:1px solid var(--line) !important; border-radius:9px !important;
  background:#fff !important; color:var(--muted) !important; font-weight:500;
  line-height:1.15; outline:none !important; box-shadow:none !important;
  transition:background .15s, color .15s, border-color .15s; }
/* o tema do Jupyter desenha a "barrinha azul" da aba ativa como um pseudo-
   elemento ::before (background var(--jp-brand-color1)); aqui ele some de vez */
.mseg-tabs .lm-TabBar-tab::before, .mseg-tabs .lm-TabBar-tab::after,
.mseg-tabs .p-TabBar-tab::before, .mseg-tabs .p-TabBar-tab::after {
  display:none !important; content:none !important; background:none !important; }
.mseg-tabs .lm-TabBar-tab:hover, .mseg-tabs .p-TabBar-tab:hover {
  background:var(--ac-soft) !important; color:var(--ac-deep) !important;
  border-color:var(--ac-border) !important; }
.mseg-tabs .lm-TabBar-tabLabel, .mseg-tabs .p-TabBar-tabLabel {
  white-space:nowrap !important; overflow:visible !important;
  text-overflow:clip !important; max-width:none !important; }
.mseg-tabs .lm-TabBar-tab.lm-mod-current,
.mseg-tabs .p-TabBar-tab.p-mod-current { color:#fff !important; font-weight:600;
  background:var(--ac) !important; border:1px solid var(--ac) !important;
  outline:none !important; box-shadow:none !important; }
.mseg-tabs .lm-TabBar-tab.lm-mod-current:hover,
.mseg-tabs .p-TabBar-tab.p-mod-current:hover {
  background:var(--ac-deep) !important; color:#fff !important;
  border-color:var(--ac-deep) !important; }
/* fórmula do modelo — preditor linear z como termos que quebram linha */
.mseg-formula { background:var(--formula-bg); color:#e8edf4; border-radius:10px; padding:13px 15px;
  font-family:'IBM Plex Mono', ui-monospace, Menlo, Consolas, monospace; font-size:12.5px;
  line-height:1.9; }
.mseg-zlead { color:#8aa0b6; font-weight:600; margin-right:4px; }
.mseg-term { display:inline-block; border-radius:7px; padding:2px 9px; margin:3px 5px 3px 0;
  white-space:nowrap; }
.mseg-b0  { background:#23303d; color:#cfdae6; }
.mseg-pos { background:rgba(46,160,107,.18); color:#86e3b4; }
.mseg-neg { background:rgba(214,69,69,.18); color:#f2a3a3; }
.mseg-op  { opacity:.65; margin-right:3px; } .mseg-mul { opacity:.5; margin:0 4px; }
.mseg-cf  { font-weight:600; } .mseg-vn { font-weight:500; }
.mseg-eq { font-size:13px; color:var(--ink); background:var(--ac-soft);
  border:1px solid var(--ac-border); border-radius:9px; padding:8px 12px; margin-bottom:8px;
  font-family:'IBM Plex Mono', ui-monospace, monospace; }
/* tabela de coeficientes — sinal colorido + barra de magnitude + leitura */
.mseg-coef { border-collapse:collapse; width:100%; font-size:12px; margin-top:4px; }
.mseg-coef th { background:var(--tbl-head-bg); color:var(--tbl-head-ink); font-weight:600;
  text-align:right;
  padding:6px 10px; border-bottom:2px solid var(--tbl-head-line); white-space:nowrap; }
.mseg-coef td { padding:5px 10px; border-bottom:1px solid var(--hair); text-align:right;
  white-space:nowrap; }
.mseg-coef th.term, .mseg-coef td.term { text-align:left; font-weight:500; }
.mseg-coef td.read, .mseg-coef th.read { text-align:left; }
.mseg-coef td.num { font-variant-numeric:tabular-nums; font-weight:600; }
.mseg-coef tbody tr:hover td { background:var(--tbl-hover); }
.mseg-coef tr.base td { background:var(--tbl-zebra); color:var(--muted); font-style:italic; }
.mseg-barcell { width:150px; }
.mseg-barwrap { background:var(--gauge-track); border-radius:5px; height:9px; width:100%;
  overflow:hidden; }
.mseg-bar-pos { background:var(--gauge-ok); height:100%; }
.mseg-bar-neg { background:var(--gauge-bad); height:100%; }
.mseg-pos-tx { color:var(--ok-ink); } .mseg-neg-tx { color:var(--bad-ink); }
.mseg .jupyter-button { border-radius:8px; font-family:inherit; }
.mseg .jupyter-widgets { min-width:0 !important; }
/* ===== TEMA ESCURO (classe .dark no painel raiz) =====
   Paleta alinhada ao dark mode do Databricks (design system DuBois):
   fundo grey800 #11171C, superfícies grey700 #1F272D, bordas grey650 #37444F,
   texto #E8ECF0/#92A4B3, ação primária blue500 #4299E0 com texto escuro. */
.mseg.dark { --ink:#E8ECF0; --muted:#92A4B3; --line:#37444F; --ac-soft:#37444F;
  --ac-border:#5F7281; --ac-deep:#E8ECF0; --ac:#4299E0;
  --ok-ink:#3BA65E; --ok-bg:rgba(39,124,67,.16); --ok-tx:#3BA65E;
  --warn-ink:#DE7921; --warn-bg:rgba(190,80,30,.16); --warn-tx:#DE7921;
  --bad-ink:#E65B77; --bad-bg:rgba(200,45,76,.16); --bad-tx:#E65B77;
  --info-ink:#8ACAFF; --info-bg:rgba(138,202,255,.16);
  --sus-ink:#B592E5; --sus-bg:rgba(138,99,191,.24);
  --code-ink:#8ACAFF; --code-bg:rgba(138,202,255,.16);
  --gauge-ok:#3BA65E; --gauge-bad:#E65B77; --gauge-track:#37444F;
  --strong-ink:#E8ECF0; --body-ink:#C0CDD8; --sub-ink:#8396A5;
  --faint-ink:#5F7281; --hair:#37444F; --tile-bg:#11171C; --rule-bg:#11171C;
  --help-bg:#11171C; --help-line:#37444F; --neutral-bg:rgba(144,164,181,.16);
  --formula-bg:#11171C;
  --tbl-line:#37444F; --tbl-line-strong:#445461; --tbl-head-bg:#11171C;
  --tbl-head-ink:#E8ECF0; --tbl-head-line:#445461; --tbl-zebra:rgba(189,205,219,.04);
  --tbl-hover:rgba(189,205,219,.08);
  --notice-bg:rgba(190,80,30,.16); --notice-border:#DE7921; --notice-ink:#E8ECF0;
  background:#11171C; padding:8px; border-radius:12px; }
.mseg.dark .mseg-banner, .mseg.dark .mseg-card, .mseg.dark .mseg-bar {
  background:#1F272D !important; border-color:#37444F !important;
  box-shadow:none !important; }
.mseg.dark .mseg-banner .t { color:#E8ECF0; }
/* ação primária DuBois: azul com texto ESCURO (não branco) */
.mseg.dark .mseg-banner .logo { color:#11171C; }
.mseg.dark .mseg-guide > summary:hover { background:#445461; }
.mseg.dark .mseg-tabs .p-TabBar-tab, .mseg.dark .mseg-tabs .lm-TabBar-tab {
  background:#1F272D !important; color:#92A4B3 !important; border-color:#37444F !important; }
.mseg.dark .mseg-tabs .p-TabBar-tab:hover,
.mseg.dark .mseg-tabs .lm-TabBar-tab:hover { background:rgba(138,202,255,.08) !important;
  color:#8ACAFF !important; border-color:#8ACAFF !important; }
.mseg.dark .mseg-tabs .p-TabBar-tab.p-mod-current,
.mseg.dark .mseg-tabs .lm-TabBar-tab.lm-mod-current { background:#4299E0 !important;
  color:#11171C !important; border-color:#4299E0 !important; }
.mseg.dark .mseg-tabs .p-TabBar-tab.p-mod-current:hover,
.mseg.dark .mseg-tabs .lm-TabBar-tab.lm-mod-current:hover {
  background:#8ACAFF !important; color:#11171C !important; border-color:#8ACAFF !important; }
.mseg.dark .widget-text input, .mseg.dark .widget-dropdown select, .mseg.dark textarea {
  background:#11171C !important; color:#E8ECF0 !important; border-color:#37444F !important; }
.mseg.dark .widget-label, .mseg.dark .jupyter-widgets label { color:#D1D9E1 !important; }
/* botões ipywidgets sem button_style: seguem a superfície DuBois */
.mseg.dark .jupyter-button:not(.mod-primary):not(.mod-success):not(.mod-info):not(.mod-warning):not(.mod-danger) { background:#37444F !important; color:#E8ECF0 !important; }
.mseg.dark .jupyter-button.mod-active { background:#4299E0 !important; color:#11171C !important; }
</style>
"""


class ModelSegmenterUI:
    # cores via var(--tbl-*): o CSS do Styler é injetado dentro do painel
    # .mseg, então os tokens resolvem no tema ativo (claro ou escuro) sem
    # precisar re-renderizar a tabela quando o usuário alterna o tema.
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

    def __init__(self, df, target="target", task_type="classification", sample_col=None,
                 ref_sample="DES", feature_labels=None, features=None, date_col=None,
                 random_state=42):
        self.seg = ModelSegmenter(df, target=target, task_type=task_type,
                                  sample_col=sample_col, ref_sample=ref_sample,
                                  feature_labels=feature_labels, features=features,
                                  date_col=date_col, random_state=random_state, verbose=False)
        self.df = df
        self.task_type = task_type
        self.date_col = date_col
        self.result = None
        self.score_df = None   # base externa opcional p/ escorar (ui.score_df = df_novo)
        # --- estado de desempenho da UI ---------------------------------------
        # buffer limitado do console: W.Output acumula TODOS os prints (o estado
        # serializado cresce sem limite e trafega pelo comm a cada ação). Mantemos
        # só as últimas N linhas reescrevendo a área com clear_output.
        self._log_lines: list = []
        # cache do <img> base64 da prévia da variável por (feature, versão de bins):
        # trocar/clicar na lista de variáveis regenerava a figura a cada vez.
        self._preview_cache: dict = {}
        # cache dos HTMLs das figuras da aba "Análise de variáveis" (aba 2) por
        # (feature, amostra, coluna-de-safra, versão de bins): revisitar uma
        # variável já analisada não re-renderiza nem re-encoda os 4–8 PNGs.
        self._an_fig_cache: dict = {}
        # ranking LAZY: variable_iv() de todas as candidatas é caro demais para
        # pagar antes do primeiro paint — só computa sob demanda (⟳ Recalcular
        # ou o primeiro fluxo que force, ex.: auto-selecionar / pós-load).
        self._vars_ready: bool = False
        # modelo "desatualizado": alterações de variáveis/bins/WoE feitas DEPOIS
        # de um treino deixam o modelo defasado até o próximo fit.
        self._dirty_since_fit: bool = False
        self._build()
        self._refresh_bar()
        self._refresh_vars()
        self._sync_bin_controls()

    # ------------------------------------------------------------------ render utils
    def _fig_html(self, fig, border=False, tight=True, stretch=False):
        import base64
        import io as _io
        import matplotlib.pyplot as plt
        buf = _io.BytesIO()
        # tight=False → o PNG sai exatamente em figsize×dpi (mesma figsize ⇒ mesmo
        # tamanho de imagem), garantindo gráficos lado a lado com a MESMA altura.
        # dpi limitado a 110 nas PRÉVIAS inline (este método só gera <img> para
        # exibição; export usa save_path nos plot_*): corta o PNG/base64 ~40% sem
        # perda visual perceptível, aliviando o tráfego kernel↔browser.
        save_kw = {"format": "png", "dpi": min(int(fig.get_dpi()), 110)}
        if tight:
            save_kw["bbox_inches"] = "tight"
        fig.savefig(buf, **save_kw)
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        # stretch=True ⇒ a imagem preenche a largura da coluna (dashboards lado a
        # lado); senão fica no tamanho natural, limitada a 100% do contêiner.
        style = "width:100%;height:auto" if stretch else "max-width:100%;height:auto"
        if border:
            style += ";border:1px solid var(--line);border-radius:6px"
        return f"<img src='data:image/png;base64,{b64}' style='{style}'/>"

    _CAT_COLORS = {"manter": ("var(--ok-ink)", "var(--ok-bg)"),
                   "revisar": ("var(--warn-ink)", "var(--warn-bg)"),
                   "descartar": ("var(--bad-ink)", "var(--bad-bg)")}

    # cores da força do IV (forte→verde, médio→azul, fraco→amarelo, inútil→vermelho,
    # suspeito→roxo = IV alto demais, possível vazamento)
    _FORCA_COLORS = {"forte": ("var(--ok-ink)", "var(--ok-bg)"),
                     "médio": ("var(--info-ink)", "var(--info-bg)"),
                     "fraco": ("var(--warn-ink)", "var(--warn-bg)"),
                     "inútil": ("var(--bad-ink)", "var(--bad-bg)"),
                     "suspeito": ("var(--sus-ink)", "var(--sus-bg)")}

    # cores da estabilidade (PSI): estável→verde · atenção→amarelo · instável→vermelho
    _ESTABILIDADE_COLORS = {"estável": ("var(--ok-ink)", "var(--ok-bg)"),
                            "atenção": ("var(--warn-ink)", "var(--warn-bg)"),
                            "instável": ("var(--bad-ink)", "var(--bad-bg)")}

    # CSS por nível de qualidade de uma métrica (identificador visual da tabela)
    _LEVEL_CSS = {
        "bom":     "background-color:var(--ok-bg);color:var(--ok-ink);font-weight:600",
        "atencao": "background-color:var(--warn-bg);color:var(--warn-ink);font-weight:600",
        "ruim":    "background-color:var(--bad-bg);color:var(--bad-ink);font-weight:600"}

    # Dicionário + guia de bolso das métricas. ``dir``: 'up' = maior melhor,
    # 'down' = menor melhor, 'zero' = perto de 0 melhor, 'info' = sem julgamento.
    # ``bands`` = (limiar_bom, limiar_atencao); abaixo/acima ⇒ 'ruim'.
    _METRIC_GUIDE = {
        "auc": dict(nome="AUC", dir="up", bands=(0.75, 0.65),
                    desc="Discriminação: prob. de ordenar um evento acima de um não-evento.",
                    rec="≥ 0,70 (bom) · ≥ 0,80 ótimo", baixa="< 0,65 fraco · 0,5 = acaso"),
        "gini": dict(nome="Gini", dir="up", bands=(0.50, 0.30),
                     desc="= 2·AUC − 1. Mesma ideia da AUC, escala 0–1.",
                     rec="≥ 0,40", baixa="< 0,30"),
        "ks": dict(nome="KS", dir="up", bands=(0.40, 0.30),
                   desc="Máxima separação entre as CDFs de evento e não-evento.",
                   rec="≥ 0,30 (bom) · ≥ 0,40 forte", baixa="< 0,20"),
        "ks_cutoff": dict(nome="KS cutoff", dir="info",
                          desc="Score onde o KS é máximo (ponto de corte, na escala do "
                               "score 0–1000). Informativo.",
                          rec="—", baixa="—"),
        "accuracy": dict(nome="Acurácia", dir="up", bands=(0.80, 0.70),
                         desc="% de acertos no corte 0,5. ⚠ enganosa em base desbalanceada.",
                         rec="comparar à taxa da classe maioria", baixa="≤ taxa da maioria"),
        "f1": dict(nome="F1", dir="up", bands=(0.60, 0.45),
                   desc="Média harmônica de precisão e recall (equilíbrio dos dois).",
                   rec="≥ 0,50 (depende do corte)", baixa="< 0,45"),
        "precision": dict(nome="Precisão", dir="up", bands=(0.60, 0.45),
                          desc="Dos previstos evento, quantos eram evento (controla falso-positivo).",
                          rec="≥ 0,60 (contextual)", baixa="< 0,45"),
        "recall": dict(nome="Recall", dir="up", bands=(0.60, 0.45),
                       desc="Dos eventos reais, quantos foram pegos (controla falso-negativo).",
                       rec="≥ 0,60 (contextual)", baixa="< 0,45"),
        "brier": dict(nome="Brier", dir="down", bands=(0.12, 0.20),
                      desc="Erro quadrático médio da probabilidade (calibração). Menor melhor.",
                      rec="≤ 0,12", baixa="> 0,20"),
        "logloss": dict(nome="LogLoss", dir="down", bands=(0.40, 0.55),
                        desc="Penaliza prob. confiantes e erradas. Menor melhor.",
                        rec="≤ 0,40", baixa="> 0,55"),
        # regressão
        "rmse": dict(nome="RMSE", dir="info",
                     desc="Raiz do erro quadrático médio (unidade do alvo). Depende da escala.",
                     rec="comparar ao desvio-padrão do alvo", baixa="≥ desvio do alvo"),
        "mae": dict(nome="MAE", dir="info",
                    desc="Erro absoluto médio (unidade do alvo). Depende da escala.",
                    rec="comparar ao alvo médio", baixa="—"),
        "mape": dict(nome="MAPE", dir="down", bands=(0.10, 0.25),
                     desc="Erro percentual absoluto médio. ⚠ explode com alvo perto de 0.",
                     rec="≤ 0,10 (10%)", baixa="> 0,25"),
        "smape": dict(nome="sMAPE", dir="down", bands=(0.10, 0.25),
                      desc="MAPE simétrico (0–1). Menor melhor.",
                      rec="≤ 0,10", baixa="> 0,25"),
        "medae": dict(nome="MedAE", dir="info",
                      desc="Erro absoluto mediano (robusto a outliers). Menor melhor.",
                      rec="—", baixa="—"),
        "r2": dict(nome="R²", dir="up", bands=(0.70, 0.40),
                   desc="Fração da variância do alvo explicada (1 perfeito, 0 = média).",
                   rec="≥ 0,40", baixa="< 0,40 · < 0 pior que a média"),
        "mean_bias": dict(nome="Viés médio", dir="zero", bands=(0.02, 0.05),
                          desc="Média (previsto − real). ~0 = sem viés sistemático.",
                          rec="≈ 0 (|viés| ≤ 0,02)", baixa="|viés| > 0,05"),
    }

    def _metric_level(self, metric, value):
        """Classifica o valor de uma métrica em 'bom'/'atencao'/'ruim' pela regra de
        bolso (ou None quando informativa/desconhecida/inválida)."""
        g = self._METRIC_GUIDE.get(metric)
        if not g or g.get("dir") in (None, "info") or not g.get("bands"):
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if v != v:                                   # NaN
            return None
        d, (a, b) = g["dir"], g["bands"]
        if d == "up":
            return "bom" if v >= a else "atencao" if v >= b else "ruim"
        if d == "down":
            return "bom" if v <= a else "atencao" if v <= b else "ruim"
        if d == "zero":
            m = abs(v)
            return "bom" if m <= a else "atencao" if m <= b else "ruim"
        return None

    def _df_html(self, df, max_height=None, color_categoria=False, center=False,
                 color_forca=False, color_tendencia=False, color_estabilidade=False,
                 color_validation=False, pct_cols=None):
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
        if color_categoria and "categoria" in df.columns:
            def _cat_css(v):
                fg, bg = self._CAT_COLORS.get(v, ("", ""))
                return (f"color:{fg};background-color:{bg};font-weight:600" if fg else "")
            sty = sty.map(_cat_css, subset=["categoria"])
        if color_forca and "forca" in df.columns:
            def _forca_css(v):
                fg, bg = self._FORCA_COLORS.get(v, ("", ""))
                return (f"color:{fg};background-color:{bg};font-weight:600" if fg else "")
            sty = sty.map(_forca_css, subset=["forca"])
        if color_tendencia and "tendencia" in df.columns:
            def _tend_css(v):                       # crescente=verde · decrescente=vermelho · senão cinza
                s = str(v)
                if "decrescente" in s:              # antes de 'crescente' (substring!)
                    return "color:var(--bad-tx);background-color:var(--bad-bg);font-weight:600"
                if "crescente" in s:
                    return "color:var(--ok-tx);background-color:var(--ok-bg);font-weight:600"
                return "color:var(--muted);background-color:var(--neutral-bg);font-weight:600"
            sty = sty.map(_tend_css, subset=["tendencia"])
        if color_estabilidade and "estabilidade" in df.columns:
            def _estab_css(v):
                fg, bg = self._ESTABILIDADE_COLORS.get(v, ("", ""))
                return (f"color:{fg};background-color:{bg};font-weight:600" if fg else "")
            sty = sty.map(_estab_css, subset=["estabilidade"])
        if color_validation:
            # backtest/PSI dos ratings: status (ok/alerta), gap, psi e classificacao
            if "status" in df.columns:
                def _status_css(v):
                    return ("color:var(--ok-tx);background-color:var(--ok-bg);font-weight:600"
                            if str(v).strip().lower() == "ok"
                            else "color:var(--bad-tx);background-color:var(--bad-bg);font-weight:600")
                sty = sty.map(_status_css, subset=["status"])
            if "classificacao" in df.columns:
                def _clf_css(v):
                    fg, bg = self._ESTABILIDADE_COLORS.get(str(v).strip(), ("", ""))
                    return (f"color:{fg};background-color:{bg};font-weight:600" if fg else "")
                sty = sty.map(_clf_css, subset=["classificacao"])
            if "psi" in df.columns and pd.api.types.is_numeric_dtype(df["psi"]):
                def _psi_css(v):                       # verde <0.10 · âmbar <0.25 · vermelho
                    if pd.isna(v):
                        return ""
                    a = abs(float(v))
                    c = ("var(--ok-tx)" if a < 0.10
                         else "var(--warn-tx)" if a < 0.25 else "var(--bad-tx)")
                    return f"color:{c};font-weight:600"
                sty = sty.map(_psi_css, subset=["psi"])
            if "gap" in df.columns and pd.api.types.is_numeric_dtype(df["gap"]):
                def _gap_css(v):                       # |gap| <=0.05 verde · <=0.10 âmbar · vermelho
                    if pd.isna(v):
                        return ""
                    a = abs(float(v))
                    c = ("var(--ok-tx)" if a <= 0.05
                         else "var(--warn-tx)" if a <= 0.10 else "var(--bad-tx)")
                    return f"color:{c};font-weight:600"
                sty = sty.map(_gap_css, subset=["gap"])
        if pct_cols:
            # exibe taxas em % (camada de exibição — o dado segue numérico, então
            # a coloração por valor, ex.: gap, continua funcionando)
            present = [c for c in pct_cols if c in df.columns]
            if present:
                sty = sty.format(lambda v: "" if pd.isna(v) else f"{v * 100:.1f}%",
                                 subset=present)
        html = sty.to_html()
        if max_height:
            html = f"<div style='max-height:{max_height};overflow:auto'>{html}</div>"
        return html

    def _log(self, msg):
        # mantém só as últimas 40 linhas: reescreve a área (clear_output) em vez de
        # acumular indefinidamente o estado do W.Output (que trafega pelo comm).
        self._log_lines.append(str(msg))
        if len(self._log_lines) > 40:
            self._log_lines = self._log_lines[-40:]
        with self.out_log:
            clear_output(wait=True)
            print("\n".join(self._log_lines))

    def _on_clear_log(self, _):
        """Limpa o console (histórico de mensagens) — botão no cabeçalho."""
        self._log_lines = []
        self.out_log.clear_output()

    @staticmethod
    def _pill(text, cls="muted"):
        return f"<span class='pill pill-{cls}'>{text}</span>"

    def _opts(self, names):
        """(alias, nome_cru) para dropdowns/listas — exibe o feature_label e mantém
        o valor cru (o .value continua sendo o nome real da coluna, então toda a
        lógica de incluir/excluir/analisar segue inalterada)."""
        return [(self.seg.label(n), n) for n in names]

    # ------------------------------------------------------------------ helpers de UX
    @contextmanager
    def _busy(self, *botoes, status=None, msg="processando…"):
        """Desabilita ``botoes`` enquanto uma ação síncrona roda e mostra um
        aviso "ocupado" em ``status`` (widget HTML), na mesma mecânica do
        :meth:`_on_fit` (disable + status + ``finally``). Ao sair, re-habilita
        os botões SEMPRE e limpa o status apenas se o handler não o substituiu
        por um resultado/erro próprio."""
        busy_html = f"<div class='mseg-legend'><i>⏳ {msg}</i></div>"
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
        """Confirmação em DOIS cliques para ações destrutivas: o 1º clique arma o
        botão (vira "Confirmar?" em vermelho por ``timeout`` segundos), o 2º
        clique executa ``action``. Sem o 2º clique, o botão desarma sozinho."""
        import threading
        import time
        if not hasattr(btn, "_cc_desc"):            # guarda o rótulo/estilo originais
            btn._cc_desc = btn.description
            btn._cc_style = btn.button_style
        now = time.monotonic()
        armado = getattr(btn, "_cc_armed", 0.0)
        if armado and now - armado <= timeout:      # 2º clique dentro da janela
            btn._cc_armed = 0.0
            btn.description = btn._cc_desc
            btn.button_style = btn._cc_style
            action()
            return
        btn._cc_armed = now                         # 1º clique: arma
        btn.description = "Confirmar?"
        btn.button_style = "danger"

        def _revert():
            # só desarma se ainda for ESTA armada (não houve 2º clique/rearme)
            if getattr(btn, "_cc_armed", 0.0) == now:
                btn._cc_armed = 0.0
                btn.description = btn._cc_desc
                btn.button_style = btn._cc_style
        threading.Timer(timeout, _revert).start()

    def _mark_dirty(self):
        """Marca o modelo como DESATUALIZADO (variáveis/derivadas/bins/WoE OU
        algoritmo/hiperparâmetros mudaram depois do treino): pill âmbar na barra +
        tarja sobre o card de métricas. No-op sem modelo treinado; a flag limpa no
        próximo fit bem-sucedido."""
        if self.seg.score_ is None or self._dirty_since_fit:
            return
        self._dirty_since_fit = True
        self.out_dirty_warn.value = (
            "<div style='border:1px solid var(--notice-border);background:var(--notice-bg);"
            "border-radius:10px;"
            "padding:9px 12px;font-size:12px;color:var(--notice-ink);margin-bottom:8px'>"
            "⚠️ <b>Modelo desatualizado</b> — variáveis/bins/WoE ou "
            "algoritmo/hiperparâmetros mudaram depois do treino; as métricas e gráficos "
            "abaixo refletem o modelo ANTIGO. Re-treine na aba Modelo.</div>")
        self._refresh_bar()

    def _clear_dirty(self):
        """Limpa a flag de modelo desatualizado (fit/tune bem-sucedido ou load)."""
        self._dirty_since_fit = False
        self.out_dirty_warn.value = ""

    # ------------------------------------------------------------------ build
    def _build(self):
        cands = self.seg.candidates
        algos = [(ALGORITHMS[a]["label"], a) for a in ALGORITHMS
                 if self.task_type in ALGORITHMS[a]["tasks"]]
        samples = self.seg._samples()

        # banner + bar + console
        self.banner = W.HTML()
        self.bar = W.HTML()
        self.out_log = W.Output(layout=W.Layout(max_height="160px", overflow="auto"))
        self.btn_clear_log = W.Button(description="Limpar log", icon="eraser",
                                      tooltip="Limpa o histórico de mensagens do console",
                                      layout=W.Layout(width="140px"))
        self.btn_clear_log.on_click(self._on_clear_log)

        # ---------- Aba 1: Variáveis ----------
        self.dd_var = W.Dropdown(options=self._opts(cands), description="Variável:",
                                 style={"description_width": "initial"})
        self.sel_included = W.SelectMultiple(options=self._opts(cands), value=tuple(self.seg.included),
                                             # rolagem a partir de ~14 itens: com muitas
                                             # candidatas, 1 <option> por linha estica o DOM
                                             # e encarece cada re-render do <select>.
                                             rows=min(14, max(4, len(cands))),
                                             description="No modelo:",
                                             style={"description_width": "initial"},
                                             layout=W.Layout(width="99%"))
        self.dd_categoria = W.Dropdown(
            options=[("— sem categoria", "—"),
                     ("✓ manter — entra no modelo", "manter"),
                     ("● revisar — decisão pendente", "revisar"),
                     ("✕ descartar — fora do modelo", "descartar")],
            value="—", description="Categoria:",
            style={"description_width": "initial"})
        self.dd_categoria.tooltip = ("Rótulo de triagem da variável (só anotação — não treina "
                                     "nem remove sozinho; quem entra no modelo é a lista 'No modelo')")
        self.out_cat_hint = W.HTML(
            "<details class='mseg-guide'>"
            "<summary>O que é “categoria”?</summary>"
            "<div class='mseg-help'>"
            "É um <b>rótulo de triagem</b> que você dá a cada variável para registrar a sua "
            "decisão durante a análise. Ele aparece na coluna <code>categoria</code> do ranking e "
            "é só documentação: <b>não treina nem remove a variável sozinho</b> — quem define o que "
            "entra no modelo é a lista <b>“No modelo”</b>, controlada pelos botões "
            "<i>Incluir variável no modelo</i> / <i>Excluir do modelo</i> (uma por vez)."
            "<ul>"
            "<li><span class='mseg-cat mseg-cat-keep'>✓ manter</span> &nbsp;variável boa, "
            "deve entrar no modelo.</li>"
            "<li><span class='mseg-cat mseg-cat-rev'>● revisar</span> &nbsp;promissora, mas "
            "precisa de uma segunda olhada (IV baixo, inversão, PSI etc.) antes de decidir.</li>"
            "<li><span class='mseg-cat mseg-cat-drop'>✕ descartar</span> &nbsp;não deve ser "
            "usada (instável, sem poder, redundante…).</li>"
            "<li><span class='mseg-cat' style='background:var(--ac-soft);color:var(--muted)'>— sem categoria</span> "
            "&nbsp;ainda não avaliada (limpa o rótulo).</li>"
            "</ul>"
            "<b>Manual:</b> escolha a variável, selecione a categoria e clique em "
            "<i>Categorizar</i> (uma a uma).<br>"
            "<b>Automático:</b> clique em <i>Auto-categorizar</i> para rotular "
            "<b>todas</b> de uma vez pela regra abaixo (usa os controles "
            "<i>IV mín.</i>, <i>PSI máx.</i> e <i>exigir monotonia</i>). Ela só rotula; a "
            "justificativa aparece na coluna <code>motivo</code> do ranking."
            "<div class='mseg-rule'>"
            "<div class='rt'>Regra (Regressão Logística)</div>"
            "<div class='rr'><span class='mseg-cat mseg-cat-drop'>descartar</span>"
            "<span class='rx'>IV &lt; <i>IV mín.</i> &nbsp;<b>ou</b>&nbsp; "
            "PSI &gt; <i>PSI máx.</i></span></div>"
            "<div class='rr'><span class='mseg-cat mseg-cat-rev'>revisar</span>"
            "<span class='rx'>IV alto demais (força “suspeito” → vazamento) &nbsp;<b>ou</b>&nbsp; "
            "IV fraco (IV mín. ≤ IV &lt; 0,10) &nbsp;<b>ou</b>&nbsp; PSI em atenção "
            "(0,10–PSI máx.) &nbsp;<b>ou</b>&nbsp; não-monotônica/inversões</span></div>"
            "<div class='rr'><span class='mseg-cat mseg-cat-keep'>manter</span>"
            "<span class='rx'>o restante (IV médio/forte, estável e monotônica)</span></div>"
            "</div></div></details>")
        self.sl_min_iv = W.BoundedFloatText(value=0.02, min=0.0, max=1.0, step=0.01,
                                            description="IV mín.:",
                                            style={"description_width": "initial"},
                                            layout=W.Layout(width="150px"))
        self.sl_max_psi = W.BoundedFloatText(value=0.25, min=0.0, max=1.0, step=0.05,
                                             description="PSI máx.:",
                                             style={"description_width": "initial"},
                                             layout=W.Layout(width="155px"))
        self.cb_require_mono = W.Checkbox(value=False, description="exigir monotonia")
        self.out_mono_hint = W.HTML(
            "<div style='font-size:11.5px;color:var(--muted);margin:2px 0 0 2px;line-height:1.5'>"
            "<b>exigir monotonia</b>: no <i>Auto-selecionar</i> / <i>Auto-categorizar</i>, "
            "só mantém a variável quando o risco (PD na classificação · alvo médio na "
            "regressão) é <b>monotônico</b> entre as faixas do binning — sobe ou desce de "
            "forma consistente, sem inversões. Quem inverte vai para <i>revisar</i> / "
            "<i>descartar</i>. Desmarcado, a monotonia não entra no critério.</div>")
        self.btn_auto = W.Button(description="Auto-selecionar", button_style="primary",
                                 icon="magic",
                                 tooltip="Inclui/exclui variáveis no modelo pelos critérios "
                                         "(IV mín., PSI máx., monotonia) e marca manter/descartar.")
        self.btn_auto_cat = W.Button(description="Auto-categorizar", icon="tags",
                                     tooltip="Classifica TODAS as variáveis em manter / revisar / "
                                             "descartar pela regra (IV · PSI · monotonia). "
                                             "Só rotula — não altera a seleção do modelo.")
        self.btn_include = W.Button(description="Incluir variável no modelo",
                                    button_style="success", icon="plus",
                                    layout=W.Layout(width="auto", min_width="234px"),
                                    tooltip="Inclui no modelo a variável escolhida em 'Variável:' "
                                            "— uma por vez, aditivo (não substitui as já incluídas).")
        self.btn_exclude = W.Button(description="Excluir do modelo", icon="minus",
                                    layout=W.Layout(width="auto", min_width="166px"),
                                    tooltip="Remove do modelo a variável escolhida em 'Variável:'.")
        self.btn_set_cat = W.Button(description="Categorizar", icon="tag")
        self.btn_incl_all = W.Button(description="Incluir todas", icon="plus")
        self.btn_clear = W.Button(description="Limpar", icon="trash")
        self.btn_refresh_vars = W.Button(description="Recalcular", icon="refresh")
        self.btn_clear_derived = W.Button(description="Resetar variáveis criadas", icon="eraser",
                                          button_style="warning",
                                          tooltip="Remove todas as variáveis categóricas criadas "
                                                  "na aba 'Análise de variáveis' (create_categorical), "
                                                  "voltando ao conjunto original.")
        self.out_vars = W.HTML()
        self.out_var_preview_h = W.HTML("<div class='mseg-h'>Estabilidade da variável no tempo</div>")
        self.out_var_preview = W.HTML()
        # --- Feature 1: definir a seleção de variáveis (escolha ótima × manual) ---
        self.btn_feat_optimal = W.Button(
            description="Escolha ótima (backward)", button_style="primary", icon="star",
            layout=W.Layout(width="auto", min_width="224px"),
            tooltip="Roda o backward elimination e aplica o subconjunto mais PARCIMONIOSO "
                    "(menor nº de variáveis dentro de 1% da melhor métrica), retreina o "
                    "modelo e regenera os ratings. Substitui a seleção vigente (confirma em "
                    "2 cliques).")
        self.btn_feat_manual = W.Button(
            description="Aplicar escolha manual", icon="hand-pointer-o",
            layout=W.Layout(width="auto", min_width="202px"),
            tooltip="Aplica como seleção do modelo exatamente as variáveis MARCADAS na lista "
                    "'No modelo' acima (limpa e inclui as marcadas).")
        self.out_star_ratings = W.HTML(
            "<div class='mseg-legend' style='margin-top:2px'>"
            "<b style='color:var(--sus-ink)'>★</b> <b>Como os ratings são elaborados</b>: ao "
            "aplicar as variáveis, o modelo é retreinado e seu score é cortado em faixas "
            "ordenadas do menor ao maior risco (decis, quantis, árvore ou binning ótimo, "
            "conforme o método na aba <i>Ratings &amp; Score</i>). Faixas vizinhas cuja "
            "inversão de risco não é significativa no OOT são <b>fundidas</b> para garantir "
            "monotonicidade; cada rating recebe como valor previsto o <b>risco médio</b> da "
            "faixa na DES (event rate na classificação, média na regressão). A escolha ótima "
            "do nº de faixas usa a régua mais parcimoniosa que mantém a ordem de risco estável "
            "entre amostras e retém quase toda a discriminação (Gini) do modelo.</div>")
        self.out_feat_sel = W.HTML()

        self.btn_auto.on_click(self._on_auto_select)
        self.btn_auto_cat.on_click(self._on_auto_categorize)
        self.btn_include.on_click(self._on_include_var)
        self.btn_exclude.on_click(self._on_exclude_var)
        self.btn_set_cat.on_click(self._on_set_cat)
        self.btn_incl_all.on_click(lambda b: (self.seg.include_all(), self._sync_sel(),
                                              self._refresh_vars(), self._mark_dirty(),
                                              self._refresh_bar()))
        # ações DESTRUTIVAS (esvaziar a seleção / remover derivadas): confirmação
        # em dois cliques — o 1º arma o botão ("Confirmar?"), o 2º executa.
        self.btn_clear.on_click(lambda b: self._confirm_twice(
            self.btn_clear,
            lambda: (self.seg.clear_features(), self._sync_sel(), self._refresh_vars(),
                     self._mark_dirty(), self._refresh_bar(),
                     self._log("[limpar] lista 'No modelo' esvaziada."))))
        self.btn_refresh_vars.on_click(lambda b: self._refresh_vars(force=True))
        self.btn_clear_derived.on_click(
            lambda b: self._confirm_twice(self.btn_clear_derived,
                                          lambda: self._on_clear_derived(b)))
        # botões largos (~19%) → só uma pequena folga entre eles na linha
        for _b in (self.btn_incl_all, self.btn_clear,
                   self.btn_refresh_vars, self.btn_clear_derived):
            _b.layout = W.Layout(width="24%")
        self.dd_var.observe(lambda c: self._refresh_var_preview(), names="value")
        # clicar numa variável na lista "No modelo" também atualiza a prévia/análise
        self.sel_included.observe(self._on_sel_click, names="value")
        self.btn_feat_manual.on_click(self._on_feat_manual)
        # escolha ótima é destrutiva (roda backward, troca a seleção, retreina e
        # regenera ratings): confirmação em dois cliques.
        self.btn_feat_optimal.on_click(lambda b: self._confirm_twice(
            self.btn_feat_optimal, lambda: self._on_feat_optimal(None)))

        tab_vars = W.VBox([
            W.HTML("<div class='mseg-h'>Seleção & categorização de variáveis</div>"),
            W.HBox([self.sl_min_iv, self.sl_max_psi, self.cb_require_mono,
                    self.btn_auto, self.btn_auto_cat]),
            self.out_mono_hint,
            # incluir/excluir UMA variável por vez — a escolhida em 'Variável:'
            W.HBox([self.dd_var, self.btn_include, self.btn_exclude]),
            W.VBox([self.sel_included,
                    W.HBox([self.btn_incl_all, self.btn_clear,
                            self.btn_refresh_vars, self.btn_clear_derived],
                           layout=W.Layout(justify_content="space-between", width="99%"))]),
            W.HTML("<div class='mseg-h' style='margin-top:4px'>Definir a seleção do modelo</div>"),
            W.HBox([self.btn_feat_manual]),
            self.out_star_ratings,
            self.out_feat_sel,
            W.HBox([self.dd_categoria, self.btn_set_cat]),
            self.out_cat_hint,
            W.HBox([W.VBox([W.HTML("<div class='mseg-h'>Ranking (IV / força / inversão / PSI)</div>"),
                            self.out_vars], layout=W.Layout(width="58%")),
                    W.VBox([self.out_var_preview_h,
                            self.out_var_preview], layout=W.Layout(width="42%"))]),
        ], layout=W.Layout(padding="2px"))

        # ---------- Aba 2: Análise de variáveis ----------
        self.dd_var2 = W.Dropdown(options=self._opts(cands), description="Variável:",
                                  style={"description_width": "initial"})
        # "(referência)" já É a ref_sample — lista só as demais para não duplicar
        self.dd_sample2 = W.Dropdown(options=["(referência)"]
                                     + [s for s in samples if s != self.seg.ref_sample],
                                     value="(referência)", description="Amostra:",
                                     style={"description_width": "initial"})
        self.tx_time2 = W.Text(value=self.date_col or "", description="Coluna safra:",
                               style={"description_width": "initial"})
        self.btn_analyze = W.Button(description="Analisar variável", button_style="primary",
                                    icon="search")
        # categorização "na mão" dos bins da variável (como nos projetos de árvore)
        self.tg_binmode = W.ToggleButtons(options=["Ótimo", "Manual"], value="Ótimo",
                                          style={"button_width": "auto"},
                                          tooltips=["Binning ótimo (optbinning)",
                                                    "Bins definidos na mão"])
        self.tx_cuts = W.Text(value="", description="Bins:",
                              placeholder="num: 0.7, 0.9   |   cat: A,B; C,D",
                              style={"description_width": "initial"},
                              layout=W.Layout(width="46%"))
        self.btn_apply_bins = W.Button(description="Aplicar bins", icon="check",
                                       button_style="success",
                                       tooltip="Aplica os bins manuais a TODA a análise univariada "
                                               "(tabela, IV, logodds, PSI e inversão)")
        self.btn_clear_bins = W.Button(description="Bins ótimos", icon="magic",
                                       tooltip="Remove os bins manuais e volta ao binning ótimo")
        self.tx_new_cat = W.Text(value="", placeholder="nome (opcional)",
                                 description="Nova variável:",
                                 style={"description_width": "initial"},
                                 layout=W.Layout(width="320px"))
        self.btn_create_cat = W.Button(description="Criar variável categórica",
                                       icon="plus-square", button_style="info",
                                       layout=W.Layout(width="auto", min_width="232px"),
                                       tooltip="Materializa a binagem/agrupamento atual (faixas "
                                               "numéricas ou grupos de categorias) como uma NOVA "
                                               "variável categórica, candidata ao modelo e recriada "
                                               "ao escorar. Junte categorias na mão como na árvore.")
        self.out_bin_hint = W.HTML()
        # categóricas no modo Manual: uma "caixa" (Dropdown de grupo) por categoria,
        # para alocar cada categoria a um grupo — como no TreeSegmenter. As numéricas
        # seguem no campo de cortes (tx_cuts). Preenchida por _rebuild_an_cat_box.
        self.an_cat_box = W.VBox(layout=W.Layout(display="none", margin="4px 0 0 0",
                                                 max_height="320px", overflow="auto"))
        self._an_cat_widgets: dict = {}     # categoria -> Dropdown de grupo
        self._an_cat_ctx = None             # variável para a qual a caixa foi montada
        self.btn_apply_bins.on_click(self._on_apply_bins)
        self.btn_clear_bins.on_click(self._on_clear_bins)
        self.btn_create_cat.on_click(self._on_create_cat)
        self.tg_binmode.observe(lambda c: self._sync_binmode(), names="value")
        # trocar a variável re-sincroniza os controles de bin E re-renderiza a
        # análise (tabela/gráficos/cards). Antes só sincronizava os bins, então os
        # painéis ficavam presos na variável anterior até clicar em "Analisar".
        self.dd_var2.observe(
            lambda c: (self._sync_bin_controls(), self._on_analyze(None)), names="value")
        self.out_an_cards = W.HTML()
        self.out_an_distbad = W.HTML()    # distribuição + % de maus (gráfico único)
        self.out_an_table = W.HTML()
        self.out_an_inv_sample = W.HTML()
        self.out_an_inv_safra = W.HTML()
        self.out_an_time = W.HTML()
        self.out_an_psi = W.HTML()
        self.out_an_optbin_share = W.HTML()   # distribuição ACUMULADA das faixas optbin no tempo
        self.btn_analyze.on_click(self._on_analyze)

        bin_card = W.VBox([
            W.HTML("<div class='mseg-h'>Categorizar a variável na mão (bins manuais)</div>"),
            W.HBox([self.tg_binmode, self.tx_cuts, self.btn_apply_bins, self.btn_clear_bins]),
            self.an_cat_box,          # caixas de grupo por categoria (categóricas, modo Manual)
            self.out_bin_hint,
            W.HBox([self.tx_new_cat, self.btn_create_cat]),
        ])
        bin_card.add_class("mseg-card")
        bin_card.layout = W.Layout(margin="26px 0 0 0")   # respiro até a linha da variável

        # rótulo do eixo de risco por tipo de problema (clf = % de maus · reg = alvo médio)
        _dist_h = "% de maus" if self.task_type == "classification" else "alvo médio"
        # espaço entre "categorizar na mão" e a faixa de métricas
        self.out_an_cards.layout = W.Layout(margin="20px 0 6px 0")
        # distribuição AO LADO da tabela por faixa + inversão entre amostras (chart menor)
        # grade 3x2 da "Análise de variáveis"
        _col = lambda titulo, out: W.VBox(
            [W.HTML(f"<div class='mseg-h'>{titulo}</div>"), out],
            layout=W.Layout(width="49%"))
        _row = lambda a, b: W.HBox(
            [a, b], layout=W.Layout(justify_content="space-between", align_items="flex-start"))
        # linha 1: tabela por faixa · distribuição vs risco médio (PD/LGD)
        row1 = _row(_col("Tabela por faixa", self.out_an_table),
                    _col(f"Distribuição &amp; {_dist_h} por faixa", self.out_an_distbad))
        # linha 2: risco das faixas por amostra · por safra
        row2 = _row(_col("Risco das faixas por amostra", self.out_an_inv_sample),
                    _col("Risco das faixas por safra", self.out_an_inv_safra))
        # linha 3: variável ao longo do tempo (percentis) · PSI por safra
        row3 = _row(_col("Variável ao longo do tempo · percentis por safra", self.out_an_time),
                    _col("PSI da variável por safra vs DES", self.out_an_psi))
        # linha 4 (largura TOTAL, 2 colunas): distribuição ACUMULADA (área empilhada)
        # das faixas do OPTIMAL BINNING ao longo do tempo — só numéricas, da primeira
        # faixa (base) até a última (topo). Substitui os antigos gráficos de share.
        row_optbin = W.VBox(
            [W.HTML("<div class='mseg-h'>Distribuição acumulada das faixas do optimal binning "
                    "ao longo do tempo (numéricas)</div>"),
             self.out_an_optbin_share], layout=W.Layout(width="100%"))
        tab_an = W.VBox([
            W.HBox([self.dd_var2, self.dd_sample2, self.tx_time2, self.btn_analyze]),
            bin_card,
            self.out_an_cards,
            row1, row2, row3, row_optbin,
        ], layout=W.Layout(padding="2px"))

        # ---------- Aba 3: Modelo ----------
        self.dd_algo = W.Dropdown(options=algos, value=algos[0][1], description="Algoritmo:",
                                  style={"description_width": "initial"})
        self.sl_n_est = W.IntSlider(value=200, min=50, max=600, step=50,
                                    description="n_estimators",
                                    style={"description_width": "initial"})
        self.cb_max_depth = W.Checkbox(value=False, description="limitar max_depth",
                                       indent=False)
        self.sl_max_depth = W.IntSlider(value=6, min=1, max=20, description="max_depth",
                                        style={"description_width": "initial"})
        self.tx_C = W.FloatText(value=1.0, description="C (regul.)",
                                style={"description_width": "initial"})
        self.tx_C.tooltip = "Inverso da regularização da Regressão Logística (menor C = mais regularização)"
        self.tx_lr = W.FloatText(value=0.05, step=0.01, description="learning_rate",
                                 style={"description_width": "initial"})
        self.tx_lr.tooltip = "Taxa de aprendizado dos modelos de boosting (menor = mais árvores/regularização)"
        # --- hiperparâmetros AVANÇADOS (opcionais, por algoritmo) -----------
        # Só entram no fit quando habilitados; ficam numa gaveta colapsada.
        adv_style = {"description_width": "initial"}
        self.cb_min_leaf = W.Checkbox(value=False, indent=False,
                                      description="definir min_samples_leaf")
        self.sl_min_leaf = W.IntSlider(value=20, min=1, max=200, description="min_samples_leaf",
                                       style=adv_style)
        self.sl_min_leaf.tooltip = "Mínimo de amostras por folha da árvore (maior = mais regularização)"
        self.dd_max_feat = W.Dropdown(
            options=[("padrão do modelo", "__default__"), ("sqrt", "sqrt"),
                     ("log2", "log2"), ("todas as variáveis", "__none__")],
            value="__default__", description="max_features", style=adv_style)
        self.dd_max_feat.tooltip = "Nº de variáveis sorteadas por split (sqrt/log2 = mais aleatoriedade/robustez)"
        self.cb_subsample = W.Checkbox(value=False, indent=False, description="definir subsample")
        self.fl_subsample = W.FloatSlider(value=1.0, min=0.5, max=1.0, step=0.05,
                                          description="subsample", style=adv_style)
        self.fl_subsample.tooltip = "Fração de linhas amostradas por árvore (< 1 = boosting estocástico/regulariza)"
        self.cb_colsample = W.Checkbox(value=False, indent=False, description="definir colsample_bytree")
        self.fl_colsample = W.FloatSlider(value=1.0, min=0.5, max=1.0, step=0.05,
                                          description="colsample_bytree", style=adv_style)
        self.fl_colsample.tooltip = "Fração de colunas amostradas por árvore (< 1 = mais regularização)"
        self.cb_num_leaves = W.Checkbox(value=False, indent=False, description="definir num_leaves")
        self.sl_num_leaves = W.IntSlider(value=31, min=8, max=255, description="num_leaves",
                                         style=adv_style)
        self.sl_num_leaves.tooltip = "Nº máx. de folhas por árvore (LightGBM, crescimento leaf-wise); maior = mais complexo"
        self.cb_l2 = W.Checkbox(value=False, indent=False, description="definir regularização L2")
        self.fl_l2 = W.FloatSlider(value=1.0, min=0.0, max=20.0, step=0.5,
                                   description="L2", style=adv_style)
        self.fl_l2.tooltip = "Regularização L2 (HistGB: l2_regularization · LightGBM/XGBoost: reg_lambda · CatBoost: l2_leaf_reg)"
        # linhas (rótulo → HBox), reveladas conforme o algoritmo e o checkbox
        self._adv_rows = {
            "min_samples_leaf": W.HBox([self.cb_min_leaf, self.sl_min_leaf]),
            "max_features": W.HBox([self.dd_max_feat]),
            "subsample": W.HBox([self.cb_subsample, self.fl_subsample]),
            "colsample_bytree": W.HBox([self.cb_colsample, self.fl_colsample]),
            "num_leaves": W.HBox([self.cb_num_leaves, self.sl_num_leaves]),
            "l2": W.HBox([self.cb_l2, self.fl_l2]),
        }
        box_adv_inner = W.VBox(list(self._adv_rows.values()))
        self.box_adv = W.Accordion(children=[box_adv_inner])
        self.box_adv.set_title(0, "Hiperparâmetros avançados (opcional)")
        self.box_adv.selected_index = None            # colapsado por padrão
        for _cb in (self.cb_min_leaf, self.cb_subsample, self.cb_colsample,
                    self.cb_num_leaves, self.cb_l2):
            _cb.observe(lambda c: self._sync_algo_visibility(), names="value")
        # --- gaveta "Ajuste do tuning (Optuna)": quais HP tunar + intervalos ---
        self._build_tuning_space()
        # caixas que aparecem/somem conforme o algoritmo escolhido
        self.box_logit = W.HBox([self.tx_C])
        self.box_ensemble = W.VBox([self.sl_n_est,
                                    W.HBox([self.cb_max_depth, self.sl_max_depth])])
        self.box_lr = W.HBox([self.tx_lr])
        self.btn_fit = W.Button(description="Treinar modelo", button_style="primary",
                                icon="cogs")
        self.btn_shap = W.Button(description="Calcular SHAP", icon="bar-chart")
        # --- tuning bayesiano (Optuna) ---
        self.sl_trials = W.IntSlider(description="trials", min=5, max=500, value=30,
                                     layout=W.Layout(width="60%"))
        self.cb_tune_mlflow = W.Checkbox(value=False, indent=False,
                                         description="registrar trials + modelo no MLflow")
        self.cb_tune_mlflow.tooltip = ("Cada trial vira um run aninhado no MLflow, com "
                                       "parâmetros e métricas agrupadas (modelagem/ e "
                                       "monitoramento/), sob um run-pai com o resumo do estudo. "
                                       "O modelo re-treinado com os melhores hiperparâmetros "
                                       "também é logado no run-pai (e registrado no Model "
                                       "Registry se você preencher 'Modelo (UC)' na aba Validar "
                                       "& Exportar; nomes duplicados viram nome_v2, nome_v3…). "
                                       "Usa o campo 'Experimento' da aba Validar & Exportar; "
                                       "se vazio, registra no experimento ativo da sessão "
                                       "(no Databricks, o do próprio notebook).")
        self.btn_tune = W.Button(description="Tunar com Optuna", button_style="warning",
                                 icon="magic")
        self.btn_tune.tooltip = ("Otimização bayesiana (Optuna): busca hiperparâmetros "
                                 "do algoritmo selecionado maximizando AUC (classificação) / "
                                 "R² (regressão) no OOT, e treina com os melhores. Os trials "
                                 "podem ser registrados no MLflow.")
        # botão para CANCELAR o tuning em andamento (o tuning roda numa thread de
        # fundo para a UI seguir responsiva; ver _on_tune/_on_cancel_tune).
        self.btn_cancel_tune = W.Button(description="Cancelar", button_style="danger",
                                        icon="times", disabled=True,
                                        layout=W.Layout(width="auto"))
        self.btn_cancel_tune.tooltip = ("Interrompe o tuning bayesiano em andamento "
                                        "(para após o trial atual). O modelo vigente é "
                                        "preservado — não é reajustado.")
        self._tune_thread = None            # thread do tuning em andamento (ou None)
        self.out_tune = W.HTML()
        # barra de progresso do tuning (escondida até começar)
        self.pb_tune = W.IntProgress(value=0, min=0, max=100, description="0/0",
                                     bar_style="info", orientation="horizontal",
                                     layout=W.Layout(width="70%", visibility="hidden"),
                                     style={"description_width": "initial"})
        # barra de status do treino (escondida até treinar)
        self.pb_fit = W.IntProgress(value=0, min=0, max=1, description="treinar:",
                                    bar_style="info", orientation="horizontal",
                                    layout=W.Layout(width="60%", visibility="hidden"),
                                    style={"description_width": "initial"})
        self.out_fit_status = W.HTML()     # status do fit (treinando / concluído / erro)
        self.btn_tune.on_click(self._on_tune)
        self.btn_cancel_tune.on_click(self._on_cancel_tune)
        self.cb_woe = W.Checkbox(value=False, indent=False,
                                 description="Transformar variáveis (WoE por bins)")
        self.cb_woe.tooltip = (
            "Em vez dos valores crus, alimenta o modelo com a transformação por bins "
            "(estilo scorecard): contínuas viram faixas e categóricas viram grupos "
            "— exatamente os bins definidos na aba 'Análise de variáveis' — e cada bin "
            "é codificado pelo seu WoE (classificação) ou risco médio (regressão).")
        self.out_woe_help = W.HTML()
        # alternar o WoE com modelo já treinado deixa o modelo desatualizado
        self.cb_woe.observe(lambda c: (self._sync_woe_hint(), self._mark_dirty()),
                            names="value")
        self.out_algo_help = W.HTML()   # tutorial do algoritmo/parâmetros selecionado
        self.out_metrics = W.HTML()
        self.out_metric_shift = W.HTML()   # gráfico do shift DES→OOT das principais métricas
        self.out_formula = W.HTML()
        self.out_model_a = W.HTML()
        self.out_model_b = W.HTML()
        self.out_model_c = W.HTML()
        self.out_shap = W.HTML()
        self.out_shap_bar = W.HTML()
        self.out_shap_rel = W.HTML()       # importância relativa por variável (categóricas agregadas)
        self.btn_fit.on_click(self._on_fit)
        self.btn_shap.on_click(self._on_shap)
        self.dd_algo.observe(lambda c: (self._sync_algo_visibility(), self._mark_dirty()),
                             names="value")
        self.cb_max_depth.observe(lambda c: (self._sync_algo_visibility(), self._mark_dirty()),
                                  names="value")
        # trocar algoritmo/hiperparâmetros depois do treino também desatualiza o modelo
        for _hp in (self.sl_n_est, self.sl_max_depth, self.tx_C, self.tx_lr):
            _hp.observe(lambda c: self._mark_dirty(), names="value")

        # --- Two-Stage (hurdle de LGD): opção só para regressão -------------
        _tgt = pd.to_numeric(self.df[self.seg.target], errors="coerce")
        if _tgt.notna().any():
            _tmin, _tmax = float(np.nanmin(_tgt.to_numpy())), float(np.nanmax(_tgt.to_numpy()))
            _tmed = float(np.nanmedian(_tgt.to_numpy()))
        else:
            _tmin, _tmax, _tmed = 0.0, 1.0, 0.5
        if not (_tmax > _tmin):                       # alvo constante → evita slider inválido
            _tmax = _tmin + 1.0
        _tmed = min(max(_tmed, _tmin), _tmax)
        _tstep = round((_tmax - _tmin) / 100.0, 6) or 0.01
        self.cb_twostage = W.Checkbox(value=False, indent=False,
                                      description="Two-Stage (hurdle de LGD)")
        self.cb_twostage.tooltip = ("Modela a regressão em DUAS etapas: um classificador para "
                                    "P(y ≥ threshold) e uma regressão no grupo acima; a resposta "
                                    "final é E[y] combinando os dois. O rating usa essa resposta.")
        self.sl_ts_threshold = W.FloatSlider(
            value=round(_tmed, 6), min=round(_tmin, 6), max=round(_tmax, 6), step=_tstep,
            description="threshold da resposta:", readout_format=".3f",
            style={"description_width": "initial"}, layout=W.Layout(width="420px"))
        self.sl_ts_threshold.tooltip = ("Acima do threshold → 1 (classe de perda, modelada pela "
                                        "regressão); abaixo → 0 (âncora).")
        _clf_algos = [(ALGORITHMS[a]["label"], a) for a in ALGORITHMS
                      if "classification" in ALGORITHMS[a]["tasks"]]
        _reg_algos = [(ALGORITHMS[a]["label"], a) for a in ALGORITHMS
                      if "regression" in ALGORITHMS[a]["tasks"]]
        self.dd_ts_clf = W.Dropdown(options=_clf_algos, value="logistica",
                                    description="Classificador (etapa 1):",
                                    style={"description_width": "initial"})
        self.dd_ts_reg = W.Dropdown(options=_reg_algos, value="linear",
                                    description="Regressão (etapa 2):",
                                    style={"description_width": "initial"})
        self.box_twostage = W.VBox([
            W.HTML("<div class='mseg-legend'>Etapa 1 classifica <b>P(y ≥ threshold)</b>; etapa 2 "
                   "regride <b>y</b> no grupo acima; a resposta final é <b>E[y] = P(≥t)·reg(x) + "
                   "(1−P)·âncora</b> (âncora = média abaixo do threshold). O rating é construído "
                   "sobre essa resposta combinada. Usa os valores crus (sem WoE).</div>"),
            self.sl_ts_threshold,
            W.HBox([self.dd_ts_clf, self.dd_ts_reg]),
        ])
        self.cb_twostage.observe(lambda c: (self._sync_algo_visibility(), self._mark_dirty()),
                                 names="value")
        self.row_twostage = W.HBox([self.cb_twostage])
        self.row_algo = W.HBox([self.dd_algo, self.cb_woe])
        self.box_tune = W.VBox([
            W.HTML("<div class='mseg-legend'>Tuning bayesiano (Optuna): busca os melhores "
                   "hiperparâmetros do algoritmo selecionado e treina com eles.</div>"),
            W.HBox([self.sl_trials, self.btn_tune, self.btn_cancel_tune]),
            W.HBox([self.cb_tune_mlflow]),
            self.pb_tune,
            self.out_tune,
        ])
        if self.task_type != "regression":           # Two-Stage só existe em regressão
            self.row_twostage.layout.display = "none"
            self.box_twostage.layout.display = "none"

        train_card = W.VBox([
            W.HTML("<div class='mseg-h'>Treinar (ou usar modelo pré-ajustado via set_model)</div>"),
            self.row_algo,
            self.row_twostage, self.box_twostage,
            self.box_logit, self.box_ensemble, self.box_lr, self.box_adv,
            self.box_tuning_space,
            self.out_algo_help,
            self.out_woe_help,
            W.HBox([self.btn_fit, self.btn_shap]),
            self.pb_fit,
            self.out_fit_status,
            self.box_tune,
        ])
        train_card.add_class("mseg-card")
        self.formula_card = W.VBox([
            W.HTML("<div class='mseg-h'>Fórmula do modelo (logística/linear)</div>"),
            self.out_formula])
        self.formula_card.add_class("mseg-card")

        # rótulos da 2ª linha de gráficos conforme a tarefa (reg: calibração +
        # resíduos · clf: ROC + KS)
        if self.task_type == "classification":
            _head_a, _head_b = "Curva ROC", "Curva KS"
        else:
            _head_a, _head_b = "Calibração · DES", "Resíduos · DES"
        # linha 1: shift das métricas (ampliado) + distribuição do score, lado a lado
        row_shift_dist = W.HBox([
            W.VBox([W.HTML("<div class='mseg-h'>Shift das principais métricas · DES → OOT</div>"),
                    self.out_metric_shift], layout=W.Layout(width="58%")),
            W.VBox([W.HTML("<div class='mseg-h'>Distribuição do score</div>"),
                    self.out_model_c], layout=W.Layout(width="41%")),
        ], layout=W.Layout(justify_content="space-between"))
        # linha 2: calibração (DES) + resíduos, lado a lado
        row_calib_resid = W.HBox([
            W.VBox([W.HTML(f"<div class='mseg-h'>{_head_a}</div>"), self.out_model_a],
                   layout=W.Layout(width="49.5%")),
            W.VBox([W.HTML(f"<div class='mseg-h'>{_head_b}</div>"), self.out_model_b],
                   layout=W.Layout(width="49.5%")),
        ], layout=W.Layout(justify_content="space-between"))
        # tarja de aviso "modelo desatualizado" (aparece sobre o card de métricas
        # quando variáveis/bins/WoE mudam depois do treino — ver _mark_dirty)
        self.out_dirty_warn = W.HTML()
        metrics_card = W.VBox([
            self.out_dirty_warn,
            W.HTML("<div class='mseg-h'>Métricas por amostra</div>"), self.out_metrics,
            row_shift_dist, row_calib_resid,
        ]); metrics_card.add_class("mseg-card")

        tab_model = W.VBox([
            train_card,
            metrics_card,
            self.formula_card,
            W.HBox([W.VBox([W.HTML("<div class='mseg-h'>SHAP — beeswarm</div>"), self.out_shap],
                           layout=W.Layout(width="55%")),
                    W.VBox([W.HTML("<div class='mseg-h'>SHAP — importância</div>"), self.out_shap_bar],
                           layout=W.Layout(width="45%"))]),
            W.VBox([W.HTML("<div class='mseg-h'>SHAP — importância relativa por variável</div>"),
                    W.HTML("<div class='mseg-legend'>Importância relativa (%) de <b>todas</b> as "
                           "variáveis que entraram no modelo, com as <i>dummies</i> de cada "
                           "variável categórica somadas numa única barra (uma linha por variável "
                           "original). Complementa os gráficos acima, que mostram cada coluna "
                           "codificada em separado.</div>"),
                    self.out_shap_rel]),
        ], layout=W.Layout(padding="2px"))
        self._sync_algo_visibility()

        # ---------- Aba 4: Ratings & Score ----------
        self.dd_method = W.Dropdown(options=[("Decis", "decis"), ("Quantil + fusão", "quantil"),
                                             ("Árvore + fusão", "arvore"), ("OptBinning", "optbin"),
                                             ("Manual · cortes de score", "manual_score"),
                                             ("Manual · percentis", "manual_percentil")],
                                    value="quantil", description="Metodologia:",
                                    style={"description_width": "initial"})
        # entrada dos métodos manuais (cortes de score OU percentis), só quando aplicável.
        # Os cortes de score são digitados na escala de negócio (0–1000).
        self.tx_manual = W.Text(value="", placeholder="cortes de score 0–1000 (ex.: 200, 500, 800)",
                                description="cortes/percentis:",
                                style={"description_width": "initial"},
                                layout=W.Layout(width="42%", display="none"))
        self.dd_method.observe(lambda c: self._sync_rating_method(), names="value")
        self.sl_nratings = W.IntSlider(value=10, min=2, max=20, description="nº ratings")
        self.cb_fusion = W.Checkbox(value=True, description="fusão monotônica (inversão)")
        self.btn_suggest_n = W.Button(description="Sugerir nº", icon="magic",
                                      tooltip="O algoritmo escolhe o nº de ratings: a régua mais "
                                              "enxuta que mantém monotonia entre amostras, volume "
                                              "mínimo por faixa e quase todo o poder de "
                                              "discriminação. Preenche o slider; depois clique em "
                                              "Gerar ratings.")
        self.btn_build_ratings = W.Button(description="Gerar ratings", button_style="primary",
                                          icon="sitemap")
        self.btn_suggest_n.on_click(self._on_suggest_n)
        self.out_rating_auto = W.HTML()
        self.out_rating_table = W.HTML()
        self.out_rating_badrate = W.HTML()
        self.out_rating_dist = W.HTML()
        self.out_rating_inv_s = W.HTML()
        self.out_rating_inv_t = W.HTML()
        # Zoom dos gráficos de inversão: aproxima o eixo Y para revelar cruzamentos
        # (inversões) que somem quando o eixo começa em 0 — em especial inversões
        # ENTRE SAFRAS. Campos mín/máx em % são opcionais (vazios ⇒ zoom automático
        # justo aos dados); o botão ↺ volta ao eixo cheio.
        self._inv_s_zoom = False
        self._inv_t_zoom = False
        _zoom_lay = W.Layout(width="82px")
        self.btn_zoom_inv_s = W.Button(description="🔍 Zoom", layout=W.Layout(width="98px"),
                                       tooltip="Aproxima o eixo Y (zoom) para revelar "
                                               "cruzamentos/inversões entre amostras. Deixe os "
                                               "campos vazios para zoom automático, ou informe "
                                               "mín/máx em %.")
        self.btn_reset_inv_s = W.Button(description="↺", layout=W.Layout(width="42px"),
                                        tooltip="Volta ao eixo cheio (começando em 0).")
        self.tx_inv_s_min = W.Text(value="", placeholder="mín %", layout=_zoom_lay,
                                   continuous_update=False)
        self.tx_inv_s_max = W.Text(value="", placeholder="máx %", layout=_zoom_lay,
                                   continuous_update=False)
        self.btn_zoom_inv_t = W.Button(description="🔍 Zoom", layout=W.Layout(width="98px"),
                                       tooltip="Aproxima o eixo Y (zoom) para revelar inversões "
                                               "ENTRE SAFRAS. Deixe os campos vazios para zoom "
                                               "automático, ou informe mín/máx em %.")
        self.btn_reset_inv_t = W.Button(description="↺", layout=W.Layout(width="42px"),
                                        tooltip="Volta ao eixo cheio (começando em 0).")
        self.tx_inv_t_min = W.Text(value="", placeholder="mín %", layout=_zoom_lay,
                                   continuous_update=False)
        self.tx_inv_t_max = W.Text(value="", placeholder="máx %", layout=_zoom_lay,
                                   continuous_update=False)
        self.btn_zoom_inv_s.on_click(self._on_zoom_inv_s)
        self.btn_reset_inv_s.on_click(self._on_reset_inv_s)
        self.btn_zoom_inv_t.on_click(self._on_zoom_inv_t)
        self.btn_reset_inv_t.on_click(self._on_reset_inv_t)
        self.tx_inv_s_min.observe(lambda c: self._render_inv_sample(), names="value")
        self.tx_inv_s_max.observe(lambda c: self._render_inv_sample(), names="value")
        self.tx_inv_t_min.observe(lambda c: self._render_inv_safra(), names="value")
        self.tx_inv_t_max.observe(lambda c: self._render_inv_safra(), names="value")
        self.box_zoom_inv_s = W.HBox([self.btn_zoom_inv_s, self.btn_reset_inv_s,
                                      self.tx_inv_s_min, self.tx_inv_s_max])
        self.box_zoom_inv_t = W.HBox([self.btn_zoom_inv_t, self.btn_reset_inv_t,
                                      self.tx_inv_t_min, self.tx_inv_t_max])
        self.out_rating_psi_safra = W.HTML()   # PSI dos ratings ao longo do tempo (por safra)
        self.out_rating_psi_sample = W.HTML()  # PSI dos ratings por amostra (DES × OOT/ESTABILIDADE)
        self.out_rating_mono = W.HTML()
        self.btn_build_ratings.on_click(self._on_build_ratings)

        tab_rating = W.VBox([
            W.HBox([self.dd_method, self.sl_nratings, self.cb_fusion,
                    self.btn_suggest_n, self.btn_build_ratings]),
            W.HTML("<div class='mseg-legend'><b>Fusão monotônica (inversão):</b> quando marcada, "
                   "funde faixas de rating vizinhas cuja ordem de risco se inverte (rating de score "
                   "maior com risco observado menor que o do rating anterior). A fusão só acontece "
                   "se a inversão <b>não</b> for estatisticamente significativa (Mann-Whitney na "
                   "regressão, qui-quadrado na classificação; α=0,05). Assim a régua fica "
                   "monotônica sem descartar separações de risco que são reais.</div>"),
            self.tx_manual,
            self.out_rating_auto,
            W.VBox([W.HTML("<div class='mseg-h'>Régua de ratings (risco por amostra)</div>"),
                    self.out_rating_table]),
            W.HBox([W.VBox([W.HTML("<div class='mseg-h'>Risco por rating</div>"),
                            self.out_rating_badrate], layout=W.Layout(width="50%")),
                    W.VBox([W.HTML("<div class='mseg-h'>Distribuição dos ratings</div>"),
                            self.out_rating_dist], layout=W.Layout(width="50%"))]),
            W.HBox([W.VBox([W.HTML("<div class='mseg-h'>Inversão entre ratings · amostras</div>"),
                            self.out_rating_inv_s, self.box_zoom_inv_s],
                           layout=W.Layout(width="50%")),
                    W.VBox([W.HTML("<div class='mseg-h'>Inversão entre ratings · safras</div>"),
                            self.out_rating_inv_t, self.box_zoom_inv_t],
                           layout=W.Layout(width="50%"))]),
            W.VBox([W.HTML("<div class='mseg-h'>PSI dos ratings por amostra · DES × OOT e "
                           "ESTABILIDADE</div>"),
                    W.HTML("<div class='mseg-legend'>Estabilidade da régua entre amostras: PSI da "
                           "distribuição dos ratings de cada amostra vs a referência (DES). "
                           "Verde &lt; 0,10 (estável) · amarelo &lt; 0,25 (atenção) · vermelho "
                           "≥ 0,25 (instável).</div>"),
                    self.out_rating_psi_sample]),
            W.VBox([W.HTML("<div class='mseg-h'>PSI dos ratings ao longo do tempo · por safra "
                           "vs DES</div>"),
                    W.HTML("<div class='mseg-legend'>Estabilidade da régua no tempo: PSI da "
                           "distribuição dos ratings de cada safra vs a referência (DES). "
                           "Verde &lt; 0,10 (estável) · amarelo &lt; 0,25 (atenção) · vermelho "
                           "≥ 0,25 (instável). Requer coluna de data (<code>date_col</code>).</div>"),
                    self.out_rating_psi_safra]),
            W.VBox([W.HTML("<div class='mseg-h'>Monotonicidade por amostra</div>"),
                    self.out_rating_mono]),
        ], layout=W.Layout(padding="2px"))

        # ---------- Aba 5: Validar & Exportar ----------
        self.btn_export = W.Button(description="Exportar DataFrame", icon="download")
        self.out_export = W.HTML()
        # rótulo e largura idênticos nos três campos ⇒ as caixas (e os botões que
        # vêm depois) ficam alinhadas verticalmente no card de Persistência.
        _persist_sty = {"description_width": "84px"}
        _persist_lay = W.Layout(width="280px")
        self.tx_save = W.Text(value="modelo_segmenter.json", description="Arquivo:",
                              style=_persist_sty, layout=_persist_lay)
        self.btn_save = W.Button(description="Salvar", button_style="success", icon="save")
        self.btn_load = W.Button(description="Carregar", icon="upload")
        # confirmação de sobrescrita INLINE (área dedicada no card, abaixo do
        # campo de caminho): o console do rodapé fica fora da viewport e um
        # clear_output posterior apagava o diálogo pendente (o save se perdia).
        # Botões criados UMA única vez e reutilizados (sem leak de handlers).
        self._ow_pending = None                # do_save aguardando confirmação
        self.out_overwrite_msg = W.HTML()
        self.btn_ow_yes = W.Button(description="Sobrescrever", button_style="danger",
                                   icon="exclamation-triangle")
        self.btn_ow_no = W.Button(description="Cancelar", icon="times")
        self.btn_ow_yes.on_click(self._on_overwrite_yes)
        self.btn_ow_no.on_click(self._on_overwrite_no)
        self.box_overwrite = W.VBox([self.out_overwrite_msg,
                                     W.HBox([self.btn_ow_yes, self.btn_ow_no])])
        self.box_overwrite.layout.display = "none"
        self.tx_pdf = W.Text(value="relatorio_modelo.pdf", description="PDF:",
                             style=_persist_sty, layout=_persist_lay)
        self.btn_pdf = W.Button(description="Gerar relatório PDF", button_style="primary",
                                icon="file-pdf-o")
        self.out_pdf = W.HTML()
        self.btn_pdf.on_click(self._on_pdf)
        self.tx_md = W.Text(value="relatorio_modelo.md", description="Markdown:",
                            style=_persist_sty, layout=_persist_lay)
        self.btn_md = W.Button(description="Gerar relatório Markdown", button_style="info",
                               icon="file-text-o")
        self.out_md = W.HTML()
        self.btn_md.on_click(self._on_md)
        self.tx_experiment = W.Text(value="", description="Experimento:",
                                    style={"description_width": "initial"})
        self.tx_model = W.Text(value="", description="Modelo (UC):",
                               style={"description_width": "initial"})
        self.btn_mlflow = W.Button(description="Registrar no MLflow", icon="database")
        # escorar base: rating + valor previsto do alvo por rating (a "régua")
        self.tx_value_col = W.Text(value="valor_previsto", description="Coluna do valor:",
                                   style={"description_width": "150px"})
        self.tx_in_table = W.Text(value="", description="Tabela (Databricks):",
                                  placeholder="catalog.schema.tabela",
                                  style={"description_width": "150px"},
                                  layout=W.Layout(width="46%"))
        self.tx_out_table = W.Text(value="", description="Gravar em (opcional):",
                                   placeholder="catalog.schema.saida",
                                   style={"description_width": "150px"},
                                   layout=W.Layout(width="46%"))
        self.cb_recreate = W.Checkbox(value=False, indent=False,
                                      description="recriar categorias/faixas das variáveis")
        self.btn_ruler = W.Button(description="Ver régua (rating → valor)", icon="list-ol")
        self.btn_score = W.Button(description="Escorar base", button_style="primary",
                                  icon="bolt")
        self.out_ruler = W.HTML()
        self.out_score = W.HTML()
        # tabela de progresso da escoragem (carregando/escorando/salvando)
        self.out_score_progress = W.HTML()
        self._score_steps: list = []      # [{key,label,status,detail}] em ordem
        self.btn_export.on_click(self._on_export)
        self.btn_ruler.on_click(self._on_ruler)
        self.btn_score.on_click(self._on_score)
        self.btn_save.on_click(self._on_save)
        self.btn_load.on_click(self._on_load)
        self.btn_mlflow.on_click(self._on_mlflow)

        self.tx_in_table.layout = W.Layout(width="48%")
        self.tx_out_table.layout = W.Layout(width="48%")
        card_score = W.VBox([
            W.HTML("<div class='mseg-h'>Escoragem da base · score + rating + valor previsto "
                   "(régua)</div>"),
            W.HTML("<div class='mseg-legend'>Devolve <code>score</code> (escala de negócio "
                   "<b>0–1000</b>), <code>rating</code> e o "
                   "valor previsto do alvo daquele rating (ex.: LGD/PD previsto, na unidade do "
                   "alvo). A base só precisa "
                   "das <b>variáveis originais do modelo</b> — binagem/WoE e faixas são recriadas ao "
                   "escorar. <b>Tabela Databricks</b>: <code>catalog.schema.tabela</code> (via Spark; "
                   "saída opcional) → <code>ui.result</code>. <b>Em memória</b>: deixe em branco (ou "
                   "<code>ui.score_df = df_novo</code>) → <code>ui.result</code>.</div>"),
            W.HBox([self.tx_in_table, self.tx_out_table],
                   layout=W.Layout(justify_content="space-between")),
            W.HBox([self.tx_value_col, self.cb_recreate]),
            W.HBox([self.btn_ruler, self.btn_score]),
            self.out_score_progress,
            self.out_ruler, self.out_score,
            W.HTML("<div class='mseg-h' style='margin-top:8px'>Exportar DataFrame rotulado</div>"),
            W.HBox([self.btn_export]), self.out_export,
        ]); card_score.add_class("mseg-card")
        card_persist = W.VBox([
            W.HTML("<div class='mseg-h'>Persistência · JSON + modelo joblib · relatório "
                   "PDF/Markdown</div>"),
            W.HBox([self.tx_save, self.btn_save, self.btn_load]),
            self.box_overwrite,
            W.HBox([self.tx_pdf, self.btn_pdf]),
            self.out_pdf,
            W.HBox([self.tx_md, self.btn_md]),
            self.out_md,
        ], layout=W.Layout(width="49%")); card_persist.add_class("mseg-card")
        card_mlflow = W.VBox([
            W.HTML("<div class='mseg-h'>Registrar no MLflow / Unity Catalog</div>"),
            W.HBox([self.tx_experiment, self.tx_model, self.btn_mlflow]),
        ], layout=W.Layout(width="49%")); card_mlflow.add_class("mseg-card")
        tab_export = W.VBox([
            card_score,
            W.HBox([card_persist, card_mlflow],
                   layout=W.Layout(justify_content="space-between", align_items="stretch")),
        ], layout=W.Layout(padding="2px"))

        # ---------- Aba 6: Avançado ----------
        # análises complementares de discriminação (CAP/Lift), métricas por safra
        # e backtest gráfico. Saídas invalidadas quando o modelo é re-treinado.
        self.dd_sample_adv = W.Dropdown(options=["(referência)"]
                                        + [s for s in samples if s != self.seg.ref_sample],
                                        value="(referência)", description="Amostra:",
                                        style={"description_width": "initial"})
        self.btn_cap = W.Button(description="Curva CAP", button_style="primary",
                                icon="area-chart",
                                tooltip="Curva CAP/Lorenz: % acumulado de eventos × % da "
                                        "carteira (pior → melhor score), com o AR por amostra. "
                                        "Somente classificação.")
        self.btn_lift = W.Button(description="Lift/Gains", icon="bar-chart",
                                 tooltip="Lift por decil de score (decil 1 = piores scores) + "
                                         "gains acumulado. Somente classificação.")
        self.out_adv_cap = W.HTML()
        self.out_adv_lift = W.HTML()
        self.btn_cap.on_click(self._on_adv_cap)
        self.btn_lift.on_click(self._on_adv_lift)
        card_adv_disc = W.VBox([
            W.HTML("<div class='mseg-h'>Poder discriminante · CAP e Lift/Gains</div>"),
            W.HTML(
                "<div class='mseg-help'><div class='ttl'>📈 Curva CAP "
                "(<i>Cumulative Accuracy Profile</i> / Lorenz)</div>"
                "Ordena toda a carteira do <b>pior score ao melhor</b> (do maior risco "
                "previsto ao menor) e traça, no eixo X, o <b>% acumulado da carteira</b> e, no "
                "eixo Y, o <b>% acumulado dos eventos</b> (ex.: <i>defaults</i>) já capturados. "
                "Responde: <i>“ao pegar os X% de pior score, que fração de todos os maus eu já "
                "peguei?”</i>."
                "<br>• A <b>diagonal</b> pontilhada é o <b>modelo aleatório</b> — X% da carteira "
                "⇒ X% dos maus."
                "<br>• A curva do <b>modelo perfeito</b> sobe quase na vertical: concentra 100% "
                "dos maus já nos piores scores."
                "<br>• Quanto mais a curva <b>abaúla para cima</b> da diagonal, maior o poder de "
                "discriminação."
                "<br>O <b>AR</b> (<i>Accuracy Ratio</i>) na legenda resume isso: é a área entre a "
                "curva e a diagonal, normalizada pela do modelo perfeito — <b>1 = perfeito, "
                "0 = aleatório</b>. Ele é igual ao <b>Gini</b> "
                "(<code>AR = Gini = 2·AUC − 1</code>).</div>"
                "<div class='mseg-help'><div class='ttl'>📊 Lift &amp; Gains por decil de "
                "score</div>"
                "A carteira é dividida em <b>10 decis por score</b>, do <b>decil 1 = piores "
                "scores</b> (maior risco) ao decil 10 (melhores)."
                "<br>• <b>Lift do decil</b> (barras) = <code>taxa de evento no decil ÷ taxa média "
                "da carteira</code>. Lift <b>&gt; 1</b> num decil significa concentração de "
                "eventos <b>acima da média</b>; a linha tracejada em <b>lift = 1</b> é o modelo "
                "aleatório. Ex.: lift 3,0 no decil 1 ⇒ o pior decil tem <b>3×</b> a taxa média de "
                "maus (o modelo separa bem)."
                "<br>• <b>Gains acumulado</b> (linha, eixo direito) = <b>% de todos os eventos já "
                "capturados</b> ao varrer do pior decil para o melhor. Ex.: <i>“os 20% de pior "
                "score concentram 55% dos maus”</i>."
                "<br>É a leitura <b>operacional</b> do poder de separação — apoia políticas de "
                "corte por faixa de score (aprovar/negar, priorizar cobrança). "
                "<b>CAP e Lift/Gains exigem alvo binário (só classificação).</b></div>"),
            W.HBox([self.dd_sample_adv, self.btn_cap, self.btn_lift]),
            W.HBox([W.VBox([self.out_adv_cap], layout=W.Layout(width="49.5%")),
                    W.VBox([self.out_adv_lift], layout=W.Layout(width="49.5%"))],
                   layout=W.Layout(justify_content="space-between")),
        ]); card_adv_disc.add_class("mseg-card")

        # seletor de quais métricas plotar ao longo das safras (além de DES: a amostra
        # é escolhida em dd_sample_adv; "(referência)" cobre toda a base)
        if self.task_type == "classification":
            _msafra_opts, _msafra_def = ["ks", "auc", "gini", "taxa_evento"], ("ks", "auc")
        else:
            _msafra_opts = ["rmse", "mae", "r2", "previsto_medio", "realizado_medio"]
            _msafra_def = ("rmse", "mae")
        self.sm_msafra_metrics = W.SelectMultiple(
            options=_msafra_opts, value=_msafra_def, rows=4, description="Métricas:",
            style={"description_width": "initial"}, layout=W.Layout(width="auto", min_width="150px"))
        self.btn_msafra = W.Button(description="Calcular", button_style="primary",
                                   icon="calendar",
                                   tooltip="Métricas do modelo por safra (escolha quais no "
                                           "seletor ao lado). Requer coluna de data (date_col).")
        self.out_adv_msafra_tab = W.HTML()
        self.out_adv_msafra_fig = W.HTML()
        self.btn_msafra.on_click(self._on_adv_msafra)
        card_adv_safra = W.VBox([
            W.HTML("<div class='mseg-h'>Discriminação por safra</div>"),
            W.HTML("<div class='mseg-legend'>Evolução das métricas do modelo mês a mês — "
                   "queda persistente indica degradação do poder discriminante. Escolha as "
                   "métricas no seletor e a amostra em <b>Amostra:</b> (\"(referência)\" = toda "
                   "a base). Requer coluna de data (<code>date_col</code>).</div>"),
            W.HBox([self.sm_msafra_metrics, self.btn_msafra]),
            self.out_adv_msafra_tab,
            self.out_adv_msafra_fig,
        ]); card_adv_safra.add_class("mseg-card")

        self.tx_tol_adv = W.BoundedFloatText(value=20.0, min=0.0, max=100.0, step=5.0,
                                             description="Tolerância (%):",
                                             style={"description_width": "initial"},
                                             layout=W.Layout(width="180px"))
        self.btn_backtest_plot = W.Button(description="Backtest gráfico",
                                          button_style="primary", icon="line-chart",
                                          tooltip="Previsto × realizado por safra com banda de "
                                                  "tolerância em torno do previsto; meses fora "
                                                  "da banda ficam destacados.")
        self.out_adv_backtest = W.HTML()
        self.btn_backtest_plot.on_click(self._on_adv_backtest)
        card_adv_bt = W.VBox([
            W.HTML("<div class='mseg-h'>Backtest gráfico · previsto × realizado por safra</div>"),
            W.HBox([self.tx_tol_adv, self.btn_backtest_plot]),
            self.out_adv_backtest,
        ]); card_adv_bt.add_class("mseg-card")

        # perfil (estabilidade) de TODAS as variáveis do modelo por safra (dbase)
        self.btn_varprofile = W.Button(description="Perfil das variáveis por safra",
                                       button_style="primary", icon="table",
                                       layout=W.Layout(width="auto", min_width="240px"),
                                       tooltip="Tabela por safra (dbase) com % de missings, "
                                               "média (numéricas) e moda de CADA variável do "
                                               "modelo — estabilidade das variáveis no tempo. "
                                               "Requer coluna de data.")
        self.out_adv_varprofile = W.HTML()
        self.btn_varprofile.on_click(self._on_adv_varprofile)
        card_adv_varprofile = W.VBox([
            W.HTML("<div class='mseg-h'>Perfil das variáveis do modelo por safra</div>"),
            W.HTML("<div class='mseg-legend'>Para <b>todas as variáveis do modelo</b>, por safra "
                   "(<i>dbase</i>): <b>% de missings</b>, <b>média</b> (numéricas) e <b>moda</b> "
                   "(valor mais frequente). Uma única tabela, agrupada por safra — acompanha a "
                   "estabilidade das variáveis ao longo do tempo. Requer coluna de data.</div>"),
            W.HBox([self.btn_varprofile]),
            self.out_adv_varprofile,
        ]); card_adv_varprofile.add_class("mseg-card")

        tab_adv = W.VBox([card_adv_disc, card_adv_safra, card_adv_varprofile, card_adv_bt],
                         layout=W.Layout(padding="2px"))

        # ---------- Aba: Importância (Backward Elimination) ----------
        _samples = list(dict.fromkeys(self.seg._samples()))
        self.dd_backelim_sample = W.Dropdown(
            options=_samples, value=self.seg._oot_sample(),
            description="Amostra de avaliação:", style={"description_width": "initial"})
        _n_feats0 = len(self.seg.model_features or self.seg.selected_features()
                        or self.seg.candidates)
        self.sl_backelim_min = W.IntSlider(
            value=1, min=1, max=max(2, _n_feats0 - 1), description="mín. variáveis",
            style={"description_width": "initial"})
        if self.task_type == "classification":
            _mopts = ["ks", "auc", "gini", "accuracy", "f1", "precision", "recall",
                      "brier", "logloss"]
            _mdef = ("ks", "auc")
        else:
            _mopts = ["rmse", "mae", "mape", "smape", "medae", "r2", "mean_bias"]
            _mdef = ("rmse", "mae")
        self.sm_backelim_metrics = W.SelectMultiple(
            options=_mopts, value=_mdef, rows=5, description="métricas (gráfico):",
            style={"description_width": "initial"})
        self.btn_backelim = W.Button(description="Rodar backward elimination",
                                     button_style="primary", icon="filter")
        self.btn_backelim.tooltip = (
            "Reajusta o modelo removendo a cada passo a variável menos importante "
            "(permutation importance) e registra as métricas do conjunto restante. "
            "Não altera o modelo vigente.")
        self.pb_backelim = W.IntProgress(
            value=0, min=0, max=1, description="0/0", bar_style="info",
            layout=W.Layout(width="60%", visibility="hidden"),
            style={"description_width": "initial"})
        self.out_backelim_status = W.HTML()
        self.out_backelim_table = W.HTML()
        self.out_backelim_plot = W.HTML()
        self.out_backelim_apply = W.HTML()
        # escolha MANUAL do nº de variáveis (qualquer ponto da curva) — além do ótimo (★)
        self.dd_backelim_n = W.Dropdown(
            description="Nº de variáveis:", options=[], disabled=True,
            style={"description_width": "initial"},
            layout=W.Layout(width="auto", min_width="250px"),
            tooltip="Escolhe um ponto QUALQUER da curva do backward (nº de variáveis) — não "
                    "só o ótimo (★). O subconjunto são as variáveis que sobrevivem àquele "
                    "passo (as menos importantes já foram removidas).")
        self.btn_backelim_apply_n = W.Button(
            description="Retreinar com o Nº escolhido", icon="check", disabled=True,
            layout=W.Layout(width="auto", min_width="230px"),
            tooltip="Aplica como seleção do modelo o subconjunto do nº de variáveis escolhido "
                    "ao lado, retreina e regenera os ratings. Substitui a seleção vigente "
                    "(confirma em 2 cliques).")
        self._backelim_optimal = None      # dict do passo ótimo (backward_optimal_step)
        self._backelim_thread = None
        self._backelim_result = None
        self.btn_backelim.on_click(self._on_backelim)
        self.sm_backelim_metrics.observe(lambda c: self._render_backelim_plot(), names="value")
        # retreino com o nº escolhido é destrutivo (troca seleção, retreina, ratings): 2 cliques
        self.btn_backelim_apply_n.on_click(lambda b: self._confirm_twice(
            self.btn_backelim_apply_n, lambda: self._on_backelim_apply_n()))
        # caixa p/ FIXAR o range do eixo y do gráfico. Habilitada por padrão; só entra no
        # layout em REGRESSÃO (na classificação o eixo já é fixado em 0–1 automaticamente).
        self.cb_backelim_yfix = W.Checkbox(
            value=True, description="Fixar eixo y", indent=False,
            layout=W.Layout(width="auto"),
            tooltip="Fixa o intervalo do eixo y do gráfico. Recomendado em regressão "
                    "(RMSE/MAE não estão em 0–1). Desmarque para autoescala.")
        self.ft_backelim_ymin = W.FloatText(value=0.0, description="mín:",
            style={"description_width": "initial"}, layout=W.Layout(width="130px"))
        self.ft_backelim_ymax = W.FloatText(value=1.0, description="máx:",
            style={"description_width": "initial"}, layout=W.Layout(width="130px"))
        for _w in (self.cb_backelim_yfix, self.ft_backelim_ymin, self.ft_backelim_ymax):
            _w.observe(lambda c: self._render_backelim_plot(), names="value")
        _yfix_row = ([W.HBox([self.cb_backelim_yfix, self.ft_backelim_ymin,
                              self.ft_backelim_ymax])]
                     if self.task_type == "regression" else [])

        card_backelim = W.VBox([
            W.HTML("<div class='mseg-h'>Backward elimination — impacto de remover variáveis</div>"),
            W.HTML("<div class='mseg-legend'>Treina o modelo removendo, a cada passo, a "
                   "variável <b>menos importante</b> (permutation importance) e mede as "
                   "métricas do conjunto restante, na amostra de avaliação escolhida. O modelo "
                   "vigente <b>não</b> é alterado (treina modelos temporários). Métricas: "
                   "classificação (KS, ROC-AUC, Gini, …) · regressão (RMSE, MAE, MAPE, …).</div>"),
            W.HBox([self.dd_backelim_sample, self.sl_backelim_min]),
            W.HBox([self.sm_backelim_metrics, self.btn_backelim]),
            *_yfix_row,
            self.pb_backelim,
            self.out_backelim_status,
        ]); card_backelim.add_class("mseg-card")
        tab_backelim = W.VBox([
            card_backelim,
            W.VBox([W.HTML("<div class='mseg-h'>Métricas × nº de variáveis</div>"),
                    self.out_backelim_plot]),
            W.VBox([W.HTML("<div class='mseg-h'>Tabela — variável removida a cada passo e "
                           "métricas <span style='font-weight:400;color:var(--sub-ink)'>"
                           "(★ = nº de variáveis ótimo · parcimônia)</span></div>"),
                    self.out_backelim_table,
                    W.HBox([self.btn_feat_optimal,
                            W.HTML("<span style='padding:0 8px;color:var(--sub-ink)'>ou</span>"),
                            self.dd_backelim_n, self.btn_backelim_apply_n],
                           layout=W.Layout(margin="8px 0 0 0", flex_flow="row wrap",
                                           align_items="center")),
                    self.out_backelim_apply]),
        ], layout=W.Layout(padding="2px"))

        self.tabs = W.Tab(children=[tab_vars, tab_an, tab_model, tab_backelim, tab_rating,
                                    tab_export, tab_adv])
        for i, t in enumerate(["Variáveis", "Análise de variáveis", "Modelo",
                               "Backward Elim.", "Ratings & Score", "Validar & Exportar",
                               "Avançado"]):
            self.tabs.set_title(i, t)
        self.tabs.add_class("mseg-tabs")

        console = W.VBox([
            W.HBox([W.HTML("<div class='mseg-h'>Console</div>"), self.btn_clear_log],
                   layout=W.Layout(justify_content="space-between", align_items="center")),
            self.out_log])
        console.add_class("mseg-card")
        self.cb_dark = W.ToggleButton(value=False, description="🌙 Tema escuro",
                                      tooltip="Alterna o tema claro/escuro da interface",
                                      layout=W.Layout(width="150px"))
        self.cb_dark.observe(self._on_dark, names="value")
        # mantém o cluster Databricks ativo enquanto a interface está aberta — no-op
        # fora do Databricks/Spark (ver yggdrasil.utils.keepalive).
        self._keepalive = None
        self.cb_keepalive = W.ToggleButton(
            value=False, description="☕ Manter cluster ativo",
            tooltip="Databricks: dispara um job Spark mínimo a cada 2 min para o cluster "
                    "não desligar por inatividade enquanto a interface está aberta",
            layout=W.Layout(width="190px"))
        self.cb_keepalive.observe(self._on_keepalive, names="value")
        topbar = W.HBox([self.cb_keepalive, self.cb_dark],
                        layout=W.Layout(justify_content="flex-end"))
        self.panel = W.VBox([W.HTML(_CSS), topbar, self.banner, self.bar, self.tabs, console])
        self.panel.add_class("mseg")

    def _on_dark(self, change):
        if change["new"]:
            self.panel.add_class("dark")
            self.cb_dark.description = "☀ Tema claro"
        else:
            self.panel.remove_class("dark")
            self.cb_dark.description = "🌙 Tema escuro"

    def _on_keepalive(self, change):
        from ...utils.keepalive import ClusterKeepAlive
        if change["new"]:
            if self._keepalive is None:
                self._keepalive = ClusterKeepAlive(interval_seconds=120)
            if not self._keepalive.has_spark():
                self._suspend_ka = True
                self.cb_keepalive.value = False              # reverte o toggle
                self._suspend_ka = False
                self.cb_keepalive.description = "☕ Manter cluster ativo"
                self._log("[keepalive] nenhuma SparkSession ativa — recurso só funciona "
                          "no Databricks (ou com Spark local).")
                return
            self._keepalive.start()
            self.cb_keepalive.description = "☕ Cluster ativo ✓"
            self._log("[keepalive] ligado — job Spark mínimo a cada 2 min mantém o cluster "
                      "ativo enquanto a interface estiver aberta. Desligue ao terminar.")
        else:
            if getattr(self, "_suspend_ka", False):
                return
            if self._keepalive is not None:
                self._keepalive.stop()
            self.cb_keepalive.description = "☕ Manter cluster ativo"
            self._log("[keepalive] desligado.")

    # ------------------------------------------------------------------ refresh
    def _refresh_bar(self):
        s = self.seg
        treinado = "sim" if s.score_ is not None else "não"
        nrat = len(s.rating_labels_) if s.rating_ is not None else 0
        self.banner.value = (
            "<div class='mseg-banner'><div class='logo'>MS</div><div>"
            f"<div class='t'>ModelSegmenter — {self.task_type}</div>"
            f"<div class='s'>alvo '{s.target}' · referência {s.ref_sample}</div></div></div>")
        # pill do modelo: verde (treinado) · amarela (não treinado) · âmbar com
        # aviso quando variáveis/bins/WoE mudaram DEPOIS do treino (desatualizado)
        if s.score_ is not None and getattr(self, "_dirty_since_fit", False):
            pill_modelo = self._pill("modelo desatualizado — re-treinar", "yellow")
        else:
            pill_modelo = self._pill(f"modelo treinado: {treinado}",
                                     "green" if s.score_ is not None else "yellow")
        self.bar.value = (
            "<div class='mseg-bar'>"
            + self._pill(f"task: {self.task_type}", "muted")
            + self._pill(f"candidatas: {len(s.candidates)}", "muted")
            + self._pill(f"incluídas: {len(s.included)}", "green")
            + pill_modelo
            + self._pill(f"ratings: {nrat}", "muted")
            + "</div>")

    def _sync_sel(self):
        self.sel_included.value = tuple(f for f in self.seg.candidates if f in self.seg.included)

    def _refresh_vars(self, force=False):
        """Ranking IV/PSI das candidatas. LAZY na construção: ``variable_iv()`` de
        TODAS as candidatas é caro (optbinning por variável) e bloqueava o primeiro
        paint — até o usuário pedir (⟳ Recalcular, ``force=True``) ou um fluxo
        forçar (auto-selecionar/pós-load), mostra só um placeholder. Depois do
        primeiro cálculo, as chamadas seguintes atualizam normalmente."""
        if not force and not self._vars_ready:
            self.out_vars.value = (
                "<div style='font-size:12px;color:var(--sub-ink);padding:10px 6px;"
                "line-height:1.6'>Ranking pendente — clique em <b>⟳ Recalcular</b> "
                "para ranquear as variáveis (IV/força/inversão/PSI de todas as "
                "candidatas; pode levar alguns segundos).</div>")
            self._refresh_var_preview()
            return
        self._vars_ready = True
        try:
            rk = self.seg.variable_iv().drop(columns="n_inversoes", errors="ignore")
            if "variavel" in rk.columns:                  # exibe o alias (feature_labels)
                rk["variavel"] = rk["variavel"].map(self.seg.label)
            for c in rk.columns:
                if c.startswith("psi_") or c in ("iv", "pior_psi"):
                    rk[c] = rk[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
            rk["incluida"] = rk["incluida"].map(lambda b: "✓" if b else "")
            rk["categoria"] = rk["categoria"].fillna("—")
            if "tendencia" in rk.columns:
                setas = {"crescente": "↑ crescente", "decrescente": "↓ decrescente",
                         "não-monotônica": "⇅ não-monotônica"}
                rk["tendencia"] = rk["tendencia"].map(lambda t: setas.get(t, t))
            if "bins_manuais" in rk.columns:
                rk["bins_manuais"] = rk["bins_manuais"].map(lambda b: "✎" if b else "")
                rk = rk.rename(columns={"bins_manuais": "manual"})
            self.out_vars.value = self._df_html(rk, max_height="320px",
                                                color_categoria=True, color_forca=True,
                                                color_tendencia=True, color_estabilidade=True)
        except Exception as e:
            self.out_vars.value = f"<i>falha ao calcular IV: {e}</i>"
        self._refresh_var_preview()

    def _refresh_var_preview(self):
        """Prévia ao lado do ranking: a **estabilidade no tempo** da variável — risco
        de cada faixa por safra (cruzamentos/safras sombreadas = inversão ao longo do
        tempo), complementando as colunas de PSI/estabilidade/inversão do ranking.
        Sem coluna de safra, cai para o logodds por faixa."""
        feat = self.dd_var.value
        try:
            cat = self.seg._detect_kind(feat) == "cat"
            if self.seg.date_col:
                titulo = ("PD por categoria ao longo do tempo" if cat
                          else "risco dos bins (n_bins) ao longo do tempo")
                hdr = f"<div class='mseg-h'>Estabilidade no tempo · {titulo}</div>"
            else:
                hdr = ("<div class='mseg-h'>Logodds da variável por faixa "
                       "<span style='font-weight:400;text-transform:none'>(defina uma coluna de "
                       "safra para ver a estabilidade no tempo)</span></div>")
            self.out_var_preview_h.value = hdr
            # cache do <img> por (feature, versão de bins): revisitar a mesma
            # variável (clicar na lista / trocar dropdown) reusa o PNG em vez de
            # re-renderizar a figura e reencodá-la a cada interação.
            ck = (feat, bool(self.seg.date_col), self.seg._rank_version)
            html = self._preview_cache.get(ck)
            if html is None:
                fig = (self.seg.plot_variable_risk_by_safra(feat) if self.seg.date_col
                       else self.seg.plot_variable_logodds(feat))
                html = self._fig_html(fig, tight=False)
                if len(self._preview_cache) > 128:      # backstop de memória
                    self._preview_cache.clear()
                self._preview_cache[ck] = html
            self.out_var_preview.value = html
        except Exception as e:
            self.out_var_preview.value = f"<i>{e}</i>"

    # ------------------------------------------------------------------ Aba 1 handlers
    def _on_auto_select(self, b):
        try:
            self.seg.auto_select(min_iv=self.sl_min_iv.value, max_psi=self.sl_max_psi.value,
                                 require_monotonic=self.cb_require_mono.value)
            # o ranking já foi computado pelo auto_select → exibe (force)
            self._sync_sel(); self._refresh_vars(force=True)
            self._mark_dirty(); self._refresh_bar()
            self._log(f"[auto] incluídas {len(self.seg.included)} variáveis.")
        except Exception as e:
            self._log(f"[auto] erro: {e}")

    def _on_feat_manual(self, b):
        """Escolha manual: aplica como seleção do modelo exatamente as variáveis
        MARCADAS na lista 'No modelo' (SelectMultiple)."""
        try:
            chosen = [f for f in tuple(self.sel_included.value) if f in self.seg.candidates]
            if not chosen:
                self.out_feat_sel.value = (
                    "<div class='mseg-legend'>Marque ao menos uma variável na lista "
                    "<b>No modelo</b> acima e clique novamente.</div>")
                self._log("[escolha manual] nenhuma variável marcada.")
                return
            self.seg.clear_features()
            for f in chosen:
                self.seg.include(f)
            self._sync_sel(); self._refresh_vars(); self._mark_dirty(); self._refresh_bar()
            self.out_feat_sel.value = (
                f"<div class='mseg-legend'>Escolha manual aplicada: <b>{len(chosen)}</b> "
                f"variável(is) no modelo.</div>")
            self._log(f"[escolha manual] {len(chosen)} variável(is) aplicada(s).")
        except Exception as e:
            self.out_feat_sel.value = f"<div style='color:var(--bad-tx)'>Erro: {e}</div>"
            self._log(f"[escolha manual] erro: {e}")

    def _on_feat_optimal(self, b):
        """Escolha ótima: aplica o subconjunto PARCIMONIOSO do backward elimination
        (reusa o resultado da aba 'Backward Elim.' quando compatível; senão roda numa
        thread), retreina o modelo e regenera os ratings."""
        import threading
        if getattr(self, "_backelim_thread", None) is not None and self._backelim_thread.is_alive():
            self._log("[escolha ótima] backward elimination em andamento — aguarde.")
            return
        # otimiza a SELEÇÃO VIGENTE (included) — respeita variáveis marcadas depois
        # do treino; cai para model_features/candidates só se não houver seleção.
        feats0 = list(self.seg.selected_features() or self.seg.model_features
                      or self.seg.candidates)
        if len(feats0) < 2:
            self.out_backelim_apply.value = (
                "<div class='mseg-legend'>Selecione ao menos 2 variáveis antes da "
                "escolha ótima.</div>")
            self._log("[escolha ótima] menos de 2 variáveis.")
            return

        def _apply(res):
            self._apply_backward_optimal(res, self.out_backelim_apply)

        # reusa o resultado do backward só se for da MESMA seleção (IDENTIDADE, não só
        # contagem) E da MESMA amostra de avaliação — senão o passo ótimo viria de
        # métricas de outra amostra/outro conjunto de variáveis.
        res0 = getattr(self, "_backelim_result", None)
        same = (res0 is not None and len(res0)
                and set(res0.attrs.get("feats0", [])) == set(feats0)
                and res0.attrs.get("eval_sample") == self.dd_backelim_sample.value)
        if same:
            try:
                _apply(res0)
            except Exception as e:
                self.out_backelim_apply.value = f"<div style='color:var(--bad-tx)'>Erro: {e}</div>"
                self._log(f"[escolha ótima] erro ao aplicar: {e}")
            return

        # senão roda o backward (na seleção vigente) numa thread e aplica ao final
        sample = self.dd_backelim_sample.value
        min_features = max(1, min(int(self.sl_backelim_min.value), len(feats0) - 1))
        total = max(1, len(feats0) - min_features + 1)
        self.pb_backelim.max = total; self.pb_backelim.value = 0
        self.pb_backelim.description = f"0/{total}"; self.pb_backelim.bar_style = "info"
        self.pb_backelim.layout.visibility = "visible"
        self.btn_feat_optimal.disabled = True
        self.out_backelim_apply.value = "<i>Rodando backward elimination para a escolha ótima…</i>"

        def _progress(done, total_, nvar):
            self.pb_backelim.value = done
            self.pb_backelim.description = f"{done}/{total_}"
            self.out_backelim_apply.value = (
                f"<div class='mseg-legend'>Backward: passo {done}/{total_} · "
                f"{nvar} variáveis…</div>")

        def _worker():
            try:
                res = self.seg.backward_elimination(sample=sample, min_features=min_features,
                                                    features=feats0, progress_callback=_progress)
                self._backelim_result = res
                self.pb_backelim.value = self.pb_backelim.max
                self.pb_backelim.bar_style = "success"; self.pb_backelim.description = "concluído ✓"
                try:                              # atualiza também a aba Backward Elim.
                    self._render_backelim_plot()
                    self._render_backelim_table(res)
                except Exception:
                    pass
                _apply(res)
            except Exception as e:
                self.pb_backelim.bar_style = "danger"
                self.out_backelim_apply.value = (
                    f"<div style='color:var(--bad-tx)'>Erro: {type(e).__name__}: {e}</div>")
                self._log(f"[escolha ótima] erro: {e}")
            finally:
                self.btn_feat_optimal.disabled = False

        self._backelim_thread = threading.Thread(target=_worker, daemon=True)
        self._backelim_thread.start()

    def _on_auto_categorize(self, b):
        try:
            rk = self.seg.auto_categorize(min_iv=self.sl_min_iv.value,
                                          max_psi=self.sl_max_psi.value,
                                          require_monotonic=self.cb_require_mono.value)
            # o ranking já foi computado pelo auto_categorize → exibe (force)
            self._refresh_vars(force=True); self._refresh_bar()
            vc = rk["categoria"].value_counts().to_dict()
            resumo = " · ".join(f"{k}: {vc.get(k, 0)}"
                                for k in ("manter", "revisar", "descartar"))
            self._log(f"[auto-categoria] {resumo} (veja a coluna 'motivo' no ranking)")
        except Exception as e:
            self._log(f"[auto-categoria] erro: {e}")

    def _on_include_var(self, b):
        feat = self.dd_var.value
        if not feat:
            return
        self.seg.include(feat)
        self._sync_sel(); self._refresh_vars(); self._mark_dirty(); self._refresh_bar()
        self._log(f"[incluir] '{self.seg.label(feat)}' no modelo · {len(self.seg.included)} no total.")

    def _on_exclude_var(self, b):
        feat = self.dd_var.value
        if not feat:
            return
        self.seg.exclude(feat)
        self._sync_sel(); self._refresh_vars(); self._mark_dirty(); self._refresh_bar()
        self._log(f"[excluir] '{self.seg.label(feat)}' fora do modelo · {len(self.seg.included)} no total.")

    def _on_sel_click(self, change):
        """Ao clicar numa variável na lista 'No modelo', aponta a prévia (gráfico de
        estabilidade no tempo) para ela — selecionando-a no dropdown 'Variável'."""
        new = set(change.get("new") or ()); old = set(change.get("old") or ())
        clicada = list(new - old) or list(old - new)
        if clicada and clicada[-1] in self.seg.candidates:   # valores crus (não as tuplas de options)
            self.dd_var.value = clicada[-1]      # dispara _refresh_var_preview

    def _on_clear_derived(self, b):
        removidas = self.seg.clear_derived()
        self._refresh_candidates(); self._refresh_vars()
        if removidas:
            self._mark_dirty()
        self._refresh_bar()
        if removidas:
            self._log(f"[reset] {len(removidas)} variável(is) criada(s) removida(s): "
                      f"{', '.join(removidas)}.")
        else:
            self._log("[reset] nenhuma variável criada para remover.")

    def _on_set_cat(self, b):
        cat = None if self.dd_categoria.value == "—" else self.dd_categoria.value
        self.seg.set_category(self.dd_var.value, cat)
        self._refresh_vars()
        self._log(f"[categoria] {self.seg.label(self.dd_var.value)} = {cat}")

    # ------------------------------------------------------------------ Aba 2 handlers
    def _on_analyze(self, b):
        # o observer de dd_var2 pode chamar isto durante a construção/refresh —
        # antes dos painéis existirem ou sem variável selecionada: no-op seguro.
        if (getattr(self, "dd_var2", None) is None or self.dd_var2.value is None
                or not hasattr(self, "out_an_table")):
            return
        feat = self.dd_var2.value
        sample = None if self.dd_sample2.value == "(referência)" else self.dd_sample2.value
        tcol = self.tx_time2.value.strip() or None
        _an_ws = (self.out_an_distbad, self.out_an_table, self.out_an_inv_sample,
                  self.out_an_cards, self.out_an_time, self.out_an_inv_safra,
                  self.out_an_psi, self.out_an_optbin_share)
        # cache das figuras por (feature, amostra, safra, versão de bins): revisitar
        # uma variável já analisada restaura os HTMLs sem re-renderizar/re-encodar os
        # PNGs. _rank_version invalida quando bins/derivadas/amostra mudam.
        _ck = (feat, sample, tcol, self.seg._rank_version)
        _cached = self._an_fig_cache.get(_ck)
        if _cached is not None:
            for _w, _v in zip(_an_ws, _cached):
                _w.value = _v
            return
        # limpa os painéis da variável anterior antes de recompor — evita a tela
        # exibir, por alguns segundos, gráficos da variável antiga junto dos novos.
        for _w in _an_ws:
            _w.value = ""
        _an_ok = True
        self.btn_analyze.disabled = True          # evita duplo clique durante o render
        try:
            self.out_an_distbad.value = self._fig_html(
                self.seg.plot_variable_distribution_badrate(feat, sample=sample,
                                                            figsize=(6.4, 3.4)), tight=False)
            vt = self.seg.variable_table(feat, sample=sample)
            self.out_an_table.value = self._df_html(vt, max_height="240px", center=True)
            # "risco das faixas por amostra" esticado p/ preencher a coluna (menos
            # espaço em branco até o gráfico da direita)
            self.out_an_inv_sample.value = self._fig_html(
                self.seg.plot_variable_inversion_by_sample(feat, figsize=(7.2, 3.4)),
                stretch=True)
            self.out_an_cards.value = self._var_cards(self.seg.variable_summary(feat, sample))
        except Exception as e:
            self.out_an_distbad.value = f"<i>{e}</i>"
            _an_ok = False
        if tcol:
            # (out, fn, stretch): tudo estica p/ preencher a coluna. O par
            # "percentis · PSI" usa a MESMA figsize (mesmo tamanho) e TODAS as
            # safras da base (all_samples) — o PSI já considera todas.
            _ts = (8.8, 3.8)
            specs = ((self.out_an_time,
                      lambda: self.seg.plot_variable_timeseries(feat, tcol, sample,
                                                                figsize=_ts, all_samples=True), True),
                     (self.out_an_inv_safra,
                      lambda: self.seg.plot_variable_inversion_by_safra(feat, tcol, sample), True),
                     (self.out_an_psi,
                      lambda: self.seg.plot_variable_psi_by_safra(feat, tcol, figsize=_ts), True),
                     (self.out_an_optbin_share,
                      lambda: self.seg.plot_variable_optbin_cumshare_timeseries(
                          feat, tcol, sample, all_samples=True), True))
            for out, fn, stretch in specs:
                try:
                    out.value = self._fig_html(fn(), stretch=stretch)
                except Exception as e:
                    out.value = f"<i>{e}</i>"
                    _an_ok = False            # não memoiza figura de safra com erro
        else:
            self.out_an_optbin_share.value = (
                "<div class='mseg-legend'>Informe a coluna de safra para ver a "
                "distribuição ao longo do tempo.</div>")
        self.btn_analyze.disabled = False
        # memoiza os HTMLs renderizados (só em caso de sucesso) para revisita
        # instantânea desta variável — evita re-render/re-encode dos PNGs
        if _an_ok:
            if len(self._an_fig_cache) > 64:      # backstop de memória
                self._an_fig_cache.clear()
            self._an_fig_cache[_ck] = tuple(_w.value for _w in _an_ws)

    def _var_cards(self, s):
        def card(k, v):
            return f"<div class='mseg-metric'><div class='k'>{k}</div><div class='v'>{v}</div></div>"
        cells = [card("tipo", s["tipo"]), card("%falt.", f"{s['pct_missing']:.1f}%"),
                 card("IV", "—" if s["iv"] is None else f"{s['iv']:.4f}"),
                 card("força", s["forca"]), card("tendência", s["tendencia"]),
                 card("inversões", s["n_inversoes"]),
                 card("PSI pior", "—" if s["pior_psi"] is None else f"{s['pior_psi']:.4f}")]
        if s["tipo"] == "num" and "media" in s:
            cells += [card("média", s["media"]), card("mediana", s["mediana"])]
        return f"<div class='mseg-metrics'>{''.join(cells)}</div>"

    # ---- bins manuais (categorizar "na mão") ----
    def _sync_binmode(self):
        """Mostra os controles de bin só no modo Manual: campo de cortes para
        NUMÉRICAS, caixas de grupo por categoria para CATEGÓRICAS."""
        manual = self.tg_binmode.value == "Manual"
        feat = self.dd_var2.value
        is_cat = (feat is not None and self.seg._detect_kind(feat) == "cat")
        # numérica → campo de cortes; categórica → caixas de grupo (uma por categoria)
        self.tx_cuts.layout.display = "" if (manual and not is_cat) else "none"
        self.an_cat_box.layout.display = "" if (manual and is_cat) else "none"
        self.btn_apply_bins.layout.display = "" if manual else "none"
        self.tx_cuts.placeholder = "cortes, ex.: 0.7, 0.9"
        if manual and is_cat:
            self._rebuild_an_cat_box()

    def _rebuild_an_cat_box(self, force=False):
        """Monta uma "caixa" (Dropdown de grupo) por categoria da variável — como no
        TreeSegmenter. As categorias vêm da amostra de referência, ordenadas por
        risco; o grupo inicial de cada uma reflete os bins manuais atuais (senão cada
        categoria começa no seu próprio grupo). Só reconstrói quando a variável muda
        (preserva as escolhas do usuário entre sincronizações)."""
        feat = self.dd_var2.value
        if feat is None or self.seg._detect_kind(feat) != "cat":
            self.an_cat_box.children = (); self._an_cat_widgets = {}; self._an_cat_ctx = None
            return
        if not force and self._an_cat_ctx == feat and self._an_cat_widgets:
            return
        ref = self.seg._frame(self.seg.ref_sample)
        col = ref[feat]
        valid = ref[col.notna()]
        self._an_cat_widgets = {}
        self._an_cat_ctx = feat
        if len(valid) == 0:
            self.an_cat_box.children = (W.HTML(
                "<div class='mseg-legend'>Sem categorias na amostra de referência.</div>"),)
            return
        risco = (valid.assign(_c=valid[feat].astype(str))
                 .groupby("_c")[self.seg.target].mean().sort_values())
        order = list(risco.index)
        n = len(order)
        # grupo inicial: a partir dos grupos manuais atuais (se houver), senão 1..n
        cat_group = {}
        splits = self.seg.manual_bins(feat)
        if splits and isinstance(splits[0], (list, tuple)):
            for gi, grp in enumerate(splits, 1):
                for c in grp:
                    cat_group[str(c)] = gi
        risco_h = "% de maus" if self.task_type == "classification" else "alvo médio"
        rows = [W.HTML("<div class='mseg-legend'>Aloque cada categoria a um <b>grupo</b> — "
                       "categorias no <b>mesmo grupo</b> viram uma única faixa. Ordenadas por "
                       f"{risco_h} (referência). Faltantes (NaN) viram uma faixa própria."
                       "</div>")]
        for k, c in enumerate(order, 1):
            val = min(max(int(cat_group.get(c, k)), 1), n)
            dd = W.Dropdown(options=[(f"grupo {g}", g) for g in range(1, n + 1)], value=val,
                            layout=W.Layout(width="120px"),
                            style={"description_width": "initial"})
            self._an_cat_widgets[c] = dd
            lab = W.HTML(f"<span style='font-size:12px'><b>{c}</b>"
                         f"<span style='color:var(--sub-ink)'> · {risco_h} {risco[c]:.3f}</span></span>")
            rows.append(W.HBox([dd, lab], layout=W.Layout(align_items="center", gap="8px")))
        na_n = int(col.isna().sum())
        if na_n:
            rows.append(W.HTML(f"<div class='mseg-legend' style='color:var(--warn-tx)'>+ "
                               f"<b>(faltante)</b>: {na_n} linhas → faixa própria automática</div>"))
        self.an_cat_box.children = tuple(rows)

    def _an_cat_groups(self):
        """Grupos de categorias a partir das caixas: ``[[A,B],[C,D]]`` (ou ``None``)."""
        if not self._an_cat_widgets:
            return None
        grupos: dict = {}
        for c, dd in self._an_cat_widgets.items():
            grupos.setdefault(dd.value, []).append(c)
        return [grupos[g] for g in sorted(grupos)] or None

    def _sync_bin_controls(self):
        """Sincroniza o modo/campos com os bins manuais da variável selecionada."""
        feat = self.dd_var2.value
        is_cat = (feat is not None and self.seg._detect_kind(feat) == "cat")
        # numéricas mostram o texto dos cortes; categóricas usam as caixas de grupo
        self.tx_cuts.value = "" if is_cat else self.seg.manual_bins_spec(feat)
        self.tg_binmode.value = "Manual" if self.seg.manual_bins(feat) else "Ótimo"
        self._an_cat_ctx = None                 # força remontar as caixas p/ a nova variável
        self._sync_binmode()
        self._render_bin_hint(feat)

    def _render_bin_hint(self, feat):
        is_cat = (feat is not None and self.seg._detect_kind(feat) == "cat")
        if self.seg.manual_bins(feat):
            self.out_bin_hint.value = (
                "<div class='mseg-legend'>✎ Bins <b>manuais</b> ativos nesta variável — "
                "aplicados à tabela, IV, logodds/WoE, PSI e inversão.</div>")
        elif is_cat:
            self.out_bin_hint.value = (
                "<div class='mseg-legend'>Binning <b>ótimo</b> (optbinning). No modo "
                "<b>Manual</b>, aloque cada categoria a um <b>grupo</b> nas caixas acima e "
                "clique em <i>Aplicar bins</i>.</div>")
        else:
            self.out_bin_hint.value = (
                "<div class='mseg-legend'>Binning <b>ótimo</b> (optbinning). No modo "
                "<b>Manual</b>, informe os cortes separados por vírgula (ex.: "
                "<code>0.7, 0.9</code>).</div>")

    def _on_apply_bins(self, b):
        feat = self.dd_var2.value
        is_cat = (feat is not None and self.seg._detect_kind(feat) == "cat")
        try:
            # categórica → grupos das caixas (lista de listas, robusto a vírgulas no
            # nome); numérica → cortes do campo de texto.
            spec = self._an_cat_groups() if is_cat else self.tx_cuts.value
            self.seg.set_manual_bins(feat, spec)
            if not self.seg.manual_bins(feat):
                self._log(f"[bins] '{self.seg.label(feat)}': nada para aplicar.")
            else:
                self._log(f"[bins] '{self.seg.label(feat)}': bins manuais aplicados.")
            self._render_bin_hint(feat)
            self._mark_dirty()
            self._on_analyze(None)
            self._refresh_vars()
        except Exception as e:
            self._log(f"[bins] erro: {e}")

    def _on_clear_bins(self, b):
        feat = self.dd_var2.value
        self.seg.clear_manual_bins(feat)
        self.tx_cuts.value = ""
        self.tg_binmode.value = "Ótimo"
        self._rebuild_an_cat_box(force=True)     # reseta as caixas (cada categoria no seu grupo)
        self._render_bin_hint(feat)
        self._mark_dirty()
        self._log(f"[bins] '{self.seg.label(feat)}': voltou ao binning ótimo.")
        self._on_analyze(None)
        self._refresh_vars()

    def _refresh_candidates(self):
        """Atualiza as listas de variáveis (após criar uma variável derivada)."""
        cands = list(self.seg.candidates)
        opts = self._opts(cands)
        for dd in (self.dd_var, self.dd_var2):
            cur = dd.value
            dd.options = opts
            if cur in cands:
                dd.value = cur
        self.sel_included.options = opts
        self.sel_included.value = tuple(f for f in cands if f in self.seg.included)

    def _on_create_cat(self, b):
        feat = self.dd_var2.value
        name = self.tx_new_cat.value.strip() or None
        try:
            new = self.seg.create_categorical(feat, new_name=name)
            ncat = int(self.seg.df[new].nunique(dropna=True))
            self.tx_new_cat.value = ""
            self._refresh_candidates()
            self._refresh_vars()
            self._mark_dirty()
            self._refresh_bar()
            self._log(f"[nova variável] '{new}' criada de '{self.seg.label(feat)}' ({ncat} categorias) — "
                      f"já disponível na seleção e no modelo.")
        except Exception as e:
            self._log(f"[nova variável] erro: {e}")

    # ------------------------------------------------------------------ Aba 3 handlers
    def _build_tuning_space(self):
        """Monta a gaveta 'Ajuste do tuning (Optuna)': por parâmetro, um checkbox
        (tunar ou não) e o intervalo [mín, máx] — ou, para categóricos, os valores
        candidatos. As linhas visíveis e os limites default seguem o algoritmo
        selecionado (:data:`OPTUNA_SEARCH_SPACE`); só os habilitados entram na
        busca (os demais ficam no default do estimador)."""
        # tipo de cada parâmetro (o mesmo nome tem o mesmo tipo em todo algoritmo)
        types = {}
        for params in OPTUNA_SEARCH_SPACE.values():
            for name, spec in params.items():
                types.setdefault(name, spec.get("type"))
        self._sp_enable, self._sp_low, self._sp_high = {}, {}, {}
        self._sp_choices, self._sp_rows = {}, {}
        self._sp_last_algo = None
        num_lay = W.Layout(width="118px")
        num_sty = {"description_width": "30px"}
        rows = []
        for name in _SPACE_ORDER:
            t = types.get(name)
            if t is None:
                continue
            cb = W.Checkbox(value=True, indent=False, description=_SPACE_LABEL.get(name, name),
                            layout=W.Layout(width="240px"))
            self._sp_enable[name] = cb
            if t == "categorical":
                sel = W.SelectMultiple(
                    options=[("sqrt", "sqrt"), ("log2", "log2"), ("todas", "(todas)")],
                    value=("sqrt", "log2", "(todas)"), rows=3,
                    layout=W.Layout(width="150px"))
                self._sp_choices[name] = sel
                row = W.HBox([cb, sel])
            else:
                Box = W.IntText if t == "int" else W.FloatText
                lo = Box(value=0, description="mín", layout=num_lay, style=num_sty)
                hi = Box(value=1, description="máx", layout=num_lay, style=num_sty)
                self._sp_low[name] = lo
                self._sp_high[name] = hi
                row = W.HBox([cb, lo, hi])
            self._sp_rows[name] = row
            rows.append(row)
        inner = W.VBox([
            W.HTML("<div class='mseg-legend'>Marque quais hiperparâmetros o Optuna deve "
                   "buscar e ajuste o intervalo <b>[mín, máx]</b>. Desmarcados ficam no "
                   "valor padrão do algoritmo. Os limites se ajustam ao algoritmo "
                   "escolhido — edite à vontade.</div>"),
        ] + rows)
        self.box_tuning_space = W.Accordion(children=[inner])
        self.box_tuning_space.set_title(
            0, "Ajuste do tuning (Optuna) · hiperparâmetros e intervalos")
        self.box_tuning_space.selected_index = None      # colapsado por padrão

    def _apply_space_defaults(self, algo):
        """Reseta os limites (mín/máx) dos parâmetros numéricos para os defaults
        do algoritmo (:data:`OPTUNA_SEARCH_SPACE`). Categóricos preservam a
        seleção do usuário."""
        for name, spec in OPTUNA_SEARCH_SPACE.get(algo, {}).items():
            lo, hi = self._sp_low.get(name), self._sp_high.get(name)
            if lo is None or hi is None:
                continue
            if spec.get("type") == "int":
                lo.value, hi.value = int(spec["low"]), int(spec["high"])
            else:
                lo.value, hi.value = float(spec["low"]), float(spec["high"])

    def _collect_search_space(self, algo):
        """Monta o ``search_space`` para :meth:`ModelSegmenter.tune_optuna` a
        partir dos controles habilitados do algoritmo. Retorna ``None`` quando
        nada foi habilitado (⇒ usa o catálogo padrão)."""
        space = {}
        for name, spec in OPTUNA_SEARCH_SPACE.get(algo, {}).items():
            cb = self._sp_enable.get(name)
            if cb is None or not cb.value:
                continue                             # desabilitado → não tuna
            if spec.get("type") == "categorical":
                sel = self._sp_choices[name].value
                choices = [None if c == "(todas)" else c for c in sel]
                if choices:
                    space[name] = {"type": "categorical", "choices": choices}
            else:
                lo = self._sp_low[name].value
                hi = self._sp_high[name].value
                if hi < lo:                          # tolera inversão do usuário
                    lo, hi = hi, lo
                if spec.get("log"):                  # escala log exige limites > 0
                    floor = spec.get("low") or 1e-12
                    floor = floor if floor > 0 else 1e-12
                    if lo < floor:                   # 0/negativo faria o Optuna rejeitar
                        self._log(f"[tune] '{name}' é log-scale: mín {lo:g} ajustado "
                                  f"para {floor:g} (deve ser > 0).")
                        lo = floor
                    hi = max(hi, lo)
                new = dict(spec)                     # preserva type/step/log
                new["low"], new["high"] = lo, hi
                space[name] = new
        return space or None

    def _sync_algo_visibility(self):
        """Mostra só os hiperparâmetros do algoritmo escolhido: C (logística),
        n_estimators/max_depth (random forest · gradient boosting), nada (linear).

        No modo **Two-Stage** (regressão), esconde os controles de modelo único e
        o tuning e revela a caixa de duas etapas (threshold + dois algoritmos)."""
        two = (self.task_type == "regression"
               and getattr(self, "cb_twostage", None) is not None
               and self.cb_twostage.value)
        self.box_twostage.layout.display = "" if two else "none"
        if two:
            self.row_algo.layout.display = "none"
            for _bx in (self.box_logit, self.box_ensemble, self.box_lr, self.box_adv,
                        self.box_tuning_space, self.box_tune, self.formula_card):
                _bx.layout.display = "none"
            self.out_algo_help.value = ""
            self.out_woe_help.value = ""
            return
        self.row_algo.layout.display = ""
        self.box_tune.layout.display = ""
        algo = self.dd_algo.value
        ensemble = algo not in ("logistica", "linear")
        self.box_logit.layout.display = "" if algo == "logistica" else "none"
        self.box_ensemble.layout.display = "" if ensemble else "none"
        self.box_lr.layout.display = "" if algo in BOOSTING_ALGORITHMS else "none"
        self.sl_max_depth.layout.display = "" if self.cb_max_depth.value else "none"
        # --- hiperparâmetros avançados: revela só as linhas do algoritmo ----
        adv = set(ADVANCED_HYPERPARAMS.get(algo, ()))
        l2_name = _l2_param_for(algo)
        show_row = {
            "min_samples_leaf": "min_samples_leaf" in adv,
            "max_features": "max_features" in adv,
            "subsample": "subsample" in adv,
            "colsample_bytree": "colsample_bytree" in adv,
            "num_leaves": "num_leaves" in adv,
            "l2": l2_name is not None,
        }
        for key, row in self._adv_rows.items():
            row.layout.display = "" if show_row[key] else "none"
        # o sub-controle das linhas com checkbox só aparece quando marcado
        self.sl_min_leaf.layout.display = "" if self.cb_min_leaf.value else "none"
        self.fl_subsample.layout.display = "" if self.cb_subsample.value else "none"
        self.fl_colsample.layout.display = "" if self.cb_colsample.value else "none"
        self.sl_num_leaves.layout.display = "" if self.cb_num_leaves.value else "none"
        self.fl_l2.layout.display = "" if self.cb_l2.value else "none"
        self.box_adv.layout.display = "" if adv else "none"
        # --- espaço de busca do tuning: revela só os HP do algoritmo e, quando o
        # algoritmo MUDA, reseta os limites para os defaults dele (preservando
        # edições enquanto o algoritmo é o mesmo).
        sp = OPTUNA_SEARCH_SPACE.get(algo, {})
        for name, row in self._sp_rows.items():
            row.layout.display = "" if name in sp else "none"
        if self._sp_last_algo != algo:
            self._apply_space_defaults(algo)
            self._sp_last_algo = algo
        self.box_tuning_space.layout.display = "" if sp else "none"
        # a fórmula só faz sentido para modelos lineares/logísticos
        linear = algo in ("logistica", "linear")
        self.formula_card.layout.display = "" if linear else "none"
        self.out_algo_help.value = self._algo_help_html(algo)

    @staticmethod
    def _algo_help_html(algo):
        """Tutorial curto do algoritmo escolhido e do que cada hiperparâmetro faz."""
        P = {  # nome do parâmetro -> explicação (em HTML)
            "C": "<span class='pname'>C (regul.)</span> — inverso da força de regularização "
                 "(L2). <b>C alto</b> → pouca regularização: ajusta mais aos dados, coeficientes "
                 "maiores, risco de <i>overfit</i>. <b>C baixo</b> → mais regularização: encolhe "
                 "os coeficientes, modelo mais estável/generalizável. Faixa típica: 0,1 a 10.",
            "n_estimators": "<span class='pname'>n_estimators</span> — número de árvores do "
                 "ensemble. Mais árvores = previsão mais estável, porém mais lento; o ganho "
                 "satura a partir de certo ponto.",
            "max_depth": "<span class='pname'>max_depth</span> — profundidade máxima de cada "
                 "árvore. Controla a complexidade: valores altos (ou sem limite) capturam "
                 "interações finas mas tendem a <i>overfit</i>; valores baixos regularizam.",
            "learning_rate": "<span class='pname'>learning_rate</span> — peso de cada árvore na "
                 "soma do boosting. <b>Menor</b> = aprende devagar, precisa de mais árvores "
                 "(<code>n_estimators</code>), mas costuma generalizar melhor. Troca-se "
                 "<code>learning_rate</code> por <code>n_estimators</code>.",
            "min_samples_leaf": "<span class='pname'>min_samples_leaf</span> — mínimo de amostras "
                 "por folha. <b>Maior</b> = folhas mais populosas, modelo mais suave e "
                 "regularizado (menos <i>overfit</i>); <b>menor</b> = capta padrões finos. "
                 "<i>(avançado, opcional)</i>",
            "max_features": "<span class='pname'>max_features</span> — nº de variáveis sorteadas "
                 "em cada divisão. <code>sqrt</code>/<code>log2</code> descorrelacionam as "
                 "árvores (mais robustez); <i>todas</i> = cada split vê o conjunto inteiro. "
                 "<i>(avançado, opcional)</i>",
            "subsample": "<span class='pname'>subsample</span> — fração de <b>linhas</b> "
                 "amostradas por árvore. Abaixo de 1 vira boosting <i>estocástico</i>: "
                 "adiciona aleatoriedade e regulariza. <i>(avançado, opcional)</i>",
            "colsample_bytree": "<span class='pname'>colsample_bytree</span> — fração de "
                 "<b>colunas</b> amostradas por árvore. Abaixo de 1 descorrelaciona as árvores "
                 "e reduz <i>overfit</i>. <i>(avançado, opcional)</i>",
            "num_leaves": "<span class='pname'>num_leaves</span> — nº máximo de folhas por árvore "
                 "no LightGBM (crescimento <i>leaf-wise</i>). <b>Maior</b> = mais capacidade e "
                 "risco de <i>overfit</i>; costuma ser o principal controle de complexidade. "
                 "<i>(avançado, opcional)</i>",
            "l2": "<span class='pname'>regularização L2</span> — penaliza pesos/folhas grandes "
                 "(<code>l2_regularization</code> no HistGB, <code>reg_lambda</code> no "
                 "LightGBM/XGBoost, <code>l2_leaf_reg</code> no CatBoost). <b>Maior</b> = mais "
                 "regularização. <i>(avançado, opcional)</i>",
        }
        meta = {
            "logistica": ("Regressão Logística",
                "Modelo linear e <b>interpretável</b>: estima a probabilidade via "
                "<code>logit(p) = β₀ + Σ βᵢ·xᵢ</code>. Cada coeficiente vira um "
                "<code>odds_ratio</code> (veja a aba/fórmula). Ótimo ponto de partida e padrão "
                "em risco de crédito (scorecards).", ["C"]),
            "linear": ("Regressão Linear (OLS)",
                "Ajuste por mínimos quadrados para alvo contínuo: "
                "<code>ŷ = β₀ + Σ βᵢ·xᵢ</code>. Interpretável e <b>sem hiperparâmetros</b> "
                "a calibrar.", []),
            "random_forest": ("Random Forest",
                "Conjunto de árvores em <i>bagging</i> (cada uma vê uma amostra/colunas "
                "diferentes); a média reduz variância. Robusto e pouco sensível a ajuste fino.",
                ["n_estimators", "max_depth"]),
            "extra_trees": ("Extra Trees",
                "Como a Random Forest, mas com cortes <b>aleatórios</b> nas divisões — mais "
                "rápido e com variância ainda menor (viés um pouco maior).",
                ["n_estimators", "max_depth"]),
            "gradient_boosting": ("Gradient Boosting",
                "Árvores treinadas em <b>sequência</b>, cada uma corrigindo o erro da anterior. "
                "Costuma ter ótima performance, mas exige cuidado com <i>overfit</i> "
                "(balancear <code>learning_rate</code> × <code>n_estimators</code>).",
                ["n_estimators", "max_depth", "learning_rate"]),
            "hist_gradient_boosting": ("Hist Gradient Boosting",
                "Gradient boosting com <i>binning</i> por histogramas — bem mais rápido em bases "
                "grandes, mesma lógica de parâmetros.",
                ["n_estimators", "max_depth", "learning_rate"]),
            "lightgbm": ("LightGBM",
                "Boosting de alta performance (crescimento <i>leaf-wise</i>), muito rápido em "
                "bases grandes. Calibre principalmente "
                "<code>learning_rate</code> × <code>n_estimators</code>.",
                ["n_estimators", "max_depth", "learning_rate"]),
            "xgboost": ("XGBoost",
                "Boosting robusto e amplamente usado, com regularização embutida. Mesma troca "
                "<code>learning_rate</code> × <code>n_estimators</code>; <code>max_depth</code> "
                "controla a complexidade.",
                ["n_estimators", "max_depth", "learning_rate"]),
            "catboost": ("CatBoost",
                "Boosting que lida bem com variáveis categóricas e reduz <i>overfit</i> com "
                "<i>ordered boosting</i>. Mesmos parâmetros principais de calibração.",
                ["n_estimators", "max_depth", "learning_rate"]),
        }
        title, desc, params = meta.get(algo, (algo, "", []))
        params = list(params)
        # anexa os avançados (opcionais) expostos para este algoritmo
        for name in ADVANCED_HYPERPARAMS.get(algo, ()):
            key = "l2" if name in _L2_PARAM_NAMES else name
            if key in P and key not in params:
                params.append(key)
        items = "".join(f"<li>{P[p]}</li>" for p in params if p in P)
        body = (f"<ul>{items}</ul>" if items
                else "<div class='none'>Sem hiperparâmetros a ajustar.</div>")
        return (f"<div class='mseg-help'><div class='ttl'>{title}</div>{desc}{body}</div>")

    def _render_formula(self):
        if self.seg.algorithm not in ("logistica", "linear"):
            self.out_formula.value = (
                "<div class='mseg-legend'>Fórmula fechada indisponível para modelos "
                "não-lineares — use os gráficos SHAP acima.</div>")
            return
        try:
            fm = self.seg.model_formula()
        except Exception as e:
            self.out_formula.value = f"<i>{e}</i>"
            return
        is_clf = self.task_type == "classification"
        coef = fm["coef"]
        b0 = float(fm["intercept"])
        if is_clf:
            eq = ("p = 1 / (1 + e<sup>−z</sup>)&nbsp;&nbsp;onde&nbsp;&nbsp;"
                  "z = logit(p) = ln[ p / (1 − p) ]")
        else:
            eq = "ŷ = z"

        # z = intercepto + Σ coefᵢ·termoᵢ numa LINHA de equação limpa e discreta
        # (coeficiente colorido pelo sinal); o detalhe rigoroso fica na TABELA abaixo.
        parts = [f"<span class='mseg-{'pos' if b0 >= 0 else 'neg'}-tx'>{b0:+.4f}</span>"]
        for _, r in coef.iterrows():
            c = float(r["coef"]); cls = "pos" if c >= 0 else "neg"
            parts.append(
                f" <span style='color:var(--faint-ink)'>{'+' if c >= 0 else '−'}</span> "
                f"<span class='mseg-{cls}-tx'>{abs(c):.4f}</span>"
                f"<span style='color:var(--faint-ink)'>·</span>{r['termo']}")
        z_html = "<span style='color:var(--muted);font-weight:600'>z =</span> " + "".join(parts)

        # tabela de coeficientes com barra de magnitude (|coef|) e leitura do efeito
        cmax = float(coef["coef"].abs().max()) or 1.0
        has_p = "p_valor" in coef.columns          # logística → teste de hipótese (Wald)
        rows = []
        for _, r in coef.iterrows():
            c = float(r["coef"]); cls = "pos" if c >= 0 else "neg"
            bar = (f"<div class='mseg-barwrap'><div class='mseg-bar-{cls}' "
                   f"style='width:{100 * abs(c) / cmax:.1f}%'></div></div>")
            if is_clf:
                orr = float(r["odds_ratio"])
                pct = abs(orr - 1.0) * 100
                read = (f"{'↑' if c >= 0 else '↓'} chance: odds ×{orr:.2f} "
                        f"({'+' if c >= 0 else '−'}{pct:.0f}%)")
                extra = f"<td class='num'>{orr:.4f}</td><td class='read mseg-{cls}-tx'>{read}</td>"
            else:
                read = f"{'↑' if c >= 0 else '↓'} ŷ em {abs(c):.4f} por +1 un."
                extra = f"<td class='read mseg-{cls}-tx'>{read}</td>"
            if has_p:                              # p-valor + estrelas (verde = significativo)
                pv = r["p_valor"]; sg = r.get("signif", "")
                sig_ok = sg not in ("", "n.s.", ".")
                cor = "var(--ok-tx)" if sig_ok else "var(--faint-ink)"
                pv_txt = "—" if (pv is None or (isinstance(pv, float) and np.isnan(pv))) else f"{pv:.4f}"
                extra += f"<td class='num' style='color:{cor}'>{pv_txt} {sg}</td>"
            rows.append(
                f"<tr><td class='term'>{r['termo']}</td>"
                f"<td class='num mseg-{cls}-tx'>{c:+.4f}</td>"
                f"<td class='barcell'>{bar}</td>{extra}</tr>")
        # linha do intercepto (baseline) — sem barra/leitura por unidade
        base_extra = (f"<td class='num'>{np.exp(b0):.4f}</td>"
                      "<td class='read'>odds base (todos os termos = 0)</td>"
                      if is_clf else "<td class='read'>valor base de ŷ</td>")
        if has_p:
            base_extra += "<td></td>"
        base_row = (f"<tr class='base'><td class='term'>intercepto (β₀)</td>"
                    f"<td class='num'>{b0:+.4f}</td><td></td>{base_extra}</tr>")
        head_extra = "<th>odds_ratio</th><th class='read'>leitura</th>" if is_clf \
            else "<th class='read'>leitura</th>"
        if has_p:
            head_extra += "<th>p-valor (Wald)</th>"
        table = (
            "<table class='mseg-coef'><thead><tr>"
            "<th class='term'>termo</th><th>coef</th><th>magnitude |coef|</th>"
            f"{head_extra}</tr></thead><tbody>"
            f"{base_row}{''.join(rows)}</tbody></table>")

        woe_note = ("Termos <code>WoE(var)</code> = a variável transformada no WoE do seu "
                    "bin (binagem da aba Análise). " if self.seg.feature_transform == "woe"
                    else "")
        legend = (
            "<div class='mseg-legend'>" + woe_note + "Termos ordenados por |coef|. A barra compara "
            "<b>|coef|</b> (relevante quando os termos estão na mesma escala, p.ex. WoE). "
            + ("<b>odds_ratio</b> = e<sup>coef</sup>: a cada +1 no termo, a razão de "
               "chances é multiplicada por esse fator." if is_clf
               else "<b>coef</b>: variação de ŷ a cada +1 no termo.")
            + ("<br><b>p-valor (Wald)</b>: significância do coeficiente (H₀: coef = 0) — "
               "<code>***</code> p&lt;0,001 · <code>**</code> p&lt;0,01 · <code>*</code> "
               "p&lt;0,05 · <code>.</code> p&lt;0,10 · <code>n.s.</code> não significativo. "
               "Aproximação (a logística do sklearn é regularizada)." if has_p else "")
            + "</div>")
        # a TABELA é o elemento principal; as equações (link + preditor) ficam
        # discretas acima e a explicação vem por último.
        self.out_formula.value = (
            f"<div class='mseg-eq'>{eq}</div>"
            f"<div class='mseg-eq' style='line-height:1.95'>{z_html}</div>"
            f"<div style='max-height:460px;overflow:auto;margin-top:6px'>{table}</div>"
            f"{legend}")

    def _collect_hyperparams(self, algo):
        if algo == "logistica":
            return {"C": float(self.tx_C.value)}
        if algo == "linear":
            return {}
        hp = {"n_estimators": int(self.sl_n_est.value)}
        if self.cb_max_depth.value:
            hp["max_depth"] = int(self.sl_max_depth.value)
        elif algo == "gradient_boosting":
            hp["max_depth"] = 3
        if algo in BOOSTING_ALGORITHMS:
            hp["learning_rate"] = float(self.tx_lr.value)
        # --- hiperparâmetros avançados (só os habilitados p/ este algoritmo) --
        adv = set(ADVANCED_HYPERPARAMS.get(algo, ()))
        if "min_samples_leaf" in adv and self.cb_min_leaf.value:
            hp["min_samples_leaf"] = int(self.sl_min_leaf.value)
        if "max_features" in adv and self.dd_max_feat.value != "__default__":
            hp["max_features"] = (None if self.dd_max_feat.value == "__none__"
                                  else self.dd_max_feat.value)
        if "subsample" in adv and self.cb_subsample.value:
            hp["subsample"] = float(self.fl_subsample.value)
        if "colsample_bytree" in adv and self.cb_colsample.value:
            hp["colsample_bytree"] = float(self.fl_colsample.value)
        if "num_leaves" in adv and self.cb_num_leaves.value:
            hp["num_leaves"] = int(self.sl_num_leaves.value)
        l2_name = _l2_param_for(algo)
        if l2_name and self.cb_l2.value:
            hp[l2_name] = float(self.fl_l2.value)
        return hp

    def _sync_woe_hint(self):
        if self.cb_woe.value:
            alvo = ("WoE" if self.task_type == "classification" else "risco médio do bin")
            self.out_woe_help.value = (
                "<div class='mseg-help'><div class='ttl'>Variáveis transformadas (WoE por bins)</div>"
                "O modelo será treinado com a <b>transformação por bins</b> (estilo scorecard): "
                "contínuas viram faixas e categóricas viram grupos — os mesmos bins da aba "
                "<i>Análise de variáveis</i> (manuais quando existirem, senão o binning ótimo) — "
                f"e cada bin é codificado pelo seu <b>{alvo}</b>. Os termos da fórmula aparecem "
                "como <code>WoE(variável)</code>. Defina/ajuste os bins na aba Análise antes de treinar.</div>")
        else:
            self.out_woe_help.value = ""

    def _on_fit(self, b):
        two = self.task_type == "regression" and self.cb_twostage.value
        algo = self.dd_algo.value
        transform = "woe" if self.cb_woe.value else "raw"
        # barra "ocupada" enquanto o fit (síncrono) roda
        self.btn_fit.disabled = True
        self.pb_fit.bar_style = "info"
        self.pb_fit.value = self.pb_fit.max
        self.pb_fit.description = "treinando (2 etapas)…" if two else "treinando…"
        self.pb_fit.layout.visibility = "visible"
        if two:
            self.out_fit_status.value = ("<div class='mseg-legend'><i>ajustando o Two-Stage "
                                         "(classificação + regressão)… pode levar alguns "
                                         "segundos.</i></div>")
        else:
            self.out_fit_status.value = (f"<div class='mseg-legend'><i>ajustando o modelo "
                                         f"({algo})… pode levar alguns segundos.</i></div>")
        try:
            if two:
                thr = float(self.sl_ts_threshold.value)
                self.seg.fit_two_stage(threshold=thr, clf_algorithm=self.dd_ts_clf.value,
                                       reg_algorithm=self.dd_ts_reg.value)
                self._log(f"[fit] Two-Stage treinado (threshold={thr:.4f}; "
                          f"clf={self.dd_ts_clf.value}, reg={self.dd_ts_reg.value}).")
                self.out_fit_status.value = (
                    f"<div class='mseg-legend'><span style='color:var(--ok-ink);font-weight:600'>"
                    f"✓ Two-Stage treinado (threshold={thr:.4f}) · "
                    f"{len(self.seg.model_features)} variáveis.</span></div>")
            else:
                self.seg.fit(algo, hyperparams=self._collect_hyperparams(algo), transform=transform)
                modo = "WoE/bins" if transform == "woe" else "valores crus"
                self._log(f"[fit] {algo} treinado com {len(self.seg.model_features)} "
                          f"variáveis ({modo}).")
                self.out_fit_status.value = (
                    f"<div class='mseg-legend'><span style='color:var(--ok-ink);font-weight:600'>"
                    f"✓ {algo} treinado com {len(self.seg.model_features)} variáveis "
                    f"({modo}).</span></div>")
            self.pb_fit.bar_style = "success"
            self.pb_fit.description = "concluído ✓"
            self._render_metrics()
            self._render_model_plots()
            self._render_formula()
            self._clear_dirty()               # modelo recém-treinado ⇒ em dia
            self._clear_adv_outputs()         # descarta gráficos do modelo antigo
            self._refresh_bar()
            if two:                           # já traz o rating sobre a resposta combinada
                self._auto_build_rating()
            else:                             # mantém a régua coerente com o NOVO score
                self._reproject_ratings_after_refit()
        except Exception as e:
            self.pb_fit.bar_style = "danger"
            self.pb_fit.description = "erro"
            self.out_fit_status.value = (f"<div style='color:var(--bad-ink);font-size:12px'>"
                                         f"<b>Erro no fit:</b> {type(e).__name__}: {e}</div>")
            self._log(f"[fit] erro: {e}")
        finally:
            self.btn_fit.disabled = False

    def _on_tune(self, b):
        import threading
        if self._tune_thread is not None and self._tune_thread.is_alive():
            self._log("[tune] já há um tuning em andamento.")
            return
        algo = self.dd_algo.value
        transform = "woe" if self.cb_woe.value else "raw"
        n_trials = int(self.sl_trials.value)
        # prepara a barra de progresso (atualizada a cada trial via callback)
        self.pb_tune.max = n_trials
        self.pb_tune.value = 0
        self.pb_tune.bar_style = "info"
        self.pb_tune.description = f"0/{n_trials}"
        self.pb_tune.layout.visibility = "visible"
        self.btn_tune.disabled = True
        self.btn_fit.disabled = True                    # evita treino concorrente no mesmo seg
        # limpa a flag de cancelamento AQUI (main thread), ANTES de habilitar o botão
        # "Cancelar" e de iniciar a thread — assim um clique em Cancelar nunca é
        # apagado por um clear tardio na thread de fundo (o tune_optuna não limpa no
        # início justamente por isso).
        self.seg._tuning_cancel.clear()
        self.btn_cancel_tune.disabled = False           # habilita o "Cancelar"
        self.out_tune.value = "<i>Rodando otimização bayesiana (Optuna)…</i>"

        def _progress(done, total, best):
            self.pb_tune.value = done
            self.pb_tune.description = f"{done}/{total}"
            best_txt = f"{best:.4f}" if best == best else "—"   # NaN-safe (best!=best se NaN)
            self.out_tune.value = (f"<div class='mseg-legend'>Optuna · trial {done}/{total} · "
                                   f"melhor até agora = <b>{best_txt}</b></div>")

        log_mlflow = self.cb_tune_mlflow.value
        search_space = self._collect_search_space(algo)     # limites escolhidos na gaveta
        experiment = self.tx_experiment.value or None
        model_name = self.tx_model.value or None

        # O tuning roda numa thread de FUNDO para a UI seguir responsiva — assim o
        # botão "Cancelar" (que chama seg.cancel_tuning()) é processado enquanto o
        # estudo roda. As atualizações de widget a partir da thread são propagadas
        # pelo canal de comm do ipywidgets.
        def _worker():
            try:
                res = self.seg.tune_optuna(algorithm=algo, n_trials=n_trials,
                                           transform=transform, fit_best=True,
                                           progress_callback=_progress,
                                           log_mlflow=log_mlflow,
                                           mlflow_experiment=experiment,
                                           search_space=search_space,
                                           register_model=log_mlflow,
                                           mlflow_model_name=model_name)
            except Exception as e:
                self.pb_tune.bar_style = "danger"
                self.out_tune.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro no "
                                       f"tuning: {type(e).__name__}: {e}</div>")
                self._log(f"[tune] erro: {e}")
                return
            finally:
                self.btn_tune.disabled = False
                self.btn_fit.disabled = False
                self.btn_cancel_tune.disabled = True
            self._finish_tune(res, algo, log_mlflow)

        self._tune_thread = threading.Thread(target=_worker, daemon=True)
        self._tune_thread.start()

    def _finish_tune(self, res, algo, log_mlflow):
        """Atualiza a UI ao término do tuning (chamado pela thread de fundo). Trata o
        caso de **cancelamento** (modelo preservado, sem re-render)."""
        if res.get("cancelled"):
            self.pb_tune.bar_style = "warning"
            self.pb_tune.description = f"cancelado ({res['n_trials']} trials)"
            self.out_tune.value = (
                "<div style='color:var(--warn-ink);font-size:12px'><b>Tuning cancelado.</b> "
                f"{res['n_trials']} trials concluídos. O modelo vigente foi preservado "
                "(não reajustado).</div>")
            self._log(f"[tune] cancelado pelo usuário após {res['n_trials']} trials; "
                      "modelo preservado.")
            return
        if not res.get("n_trials"):
            _fail = res.get("n_failed") or 0
            self.pb_tune.bar_style = "danger"
            _extra = (f" Todos os {_fail} trials <b>falharam</b> — verifique o algoritmo/espaço "
                      "de busca (ex.: combinação de hiperparâmetros inválida)." if _fail else "")
            self.out_tune.value = ("<div style='color:var(--bad-tx);font-size:12px'>Nenhum trial "
                                   f"concluído — nada a aplicar.{_extra}</div>")
            self._log(f"[tune] nenhum trial concluído (falhas: {_fail}).")
            return
        if res.get("degenerate"):            # trials "concluíram" mas com métrica inválida
            self.pb_tune.bar_style = "danger"
            self.pb_tune.description = "degenerado"
            self.out_tune.value = ("<div style='color:var(--bad-tx);font-size:12px'>Todos os "
                                   "trials produziram <b>métrica inválida</b> (AUC/R² não-finito "
                                   "— ex.: validação de uma só classe). Modelo <b>não</b> "
                                   "reajustado; revise variáveis/amostra/espaço de busca.</div>")
            self._log("[tune] estudo degenerado (métrica inválida); modelo preservado.")
            return
        self.pb_tune.value = self.pb_tune.max
        self.pb_tune.bar_style = "success"
        self.pb_tune.description = f"{res['n_trials']}/{res['n_trials']} ✓"
        bp = "<br>".join(f"<code>{k}</code> = {v}" for k, v in res["best_params"].items())
        _fail = res.get("n_failed") or 0
        _fail_txt = (f" · <span style='color:var(--warn-ink)'>{_fail} trial(s) falharam "
                     "(ignorados)</span>" if _fail else "")
        self.out_tune.value = (
            f"<div class='mseg-legend'><b>Optuna</b> · {res['n_trials']} trials · melhor "
            f"<b>{res['metric'].upper()} = {res['best_value']:.4f}</b> (no OOT/validação)"
            f"{_fail_txt}<br>{bp}</div>")
        if log_mlflow:
            _mod = self.tx_model.value or "(artefato, sem registry)"
            _mlflow_msg = f" Trials + modelo registrados no MLflow (modelo: {_mod})."
        else:
            _mlflow_msg = ""
        self._log(f"[tune] {algo}: melhor {res['metric']}={res['best_value']:.4f} "
                  f"em {res['n_trials']} trials; modelo re-treinado com os melhores."
                  + _mlflow_msg)
        self._render_metrics()
        self._render_model_plots()
        self._render_formula()
        self._clear_dirty()                   # modelo re-treinado no tuning ⇒ em dia
        self._clear_adv_outputs()
        self._refresh_bar()
        self._reproject_ratings_after_refit()  # ratings coerentes com o score re-treinado

    def _on_cancel_tune(self, b):
        """Sinaliza o cancelamento do tuning em andamento (a thread de fundo termina
        após o trial atual). No-op se não houver tuning rodando."""
        if self._tune_thread is None or not self._tune_thread.is_alive():
            self._log("[tune] não há tuning em andamento para cancelar.")
            return
        self.btn_cancel_tune.disabled = True
        self.pb_tune.bar_style = "warning"
        self.out_tune.value = ("<div style='color:var(--warn-ink);font-size:12px'><i>Cancelando… "
                               "aguardando o trial atual terminar.</i></div>")
        self.seg.cancel_tuning()
        self._log("[tune] cancelamento solicitado.")

    def _render_metrics(self):
        if getattr(self.seg, "two_stage", False):
            self._render_metrics_twostage()
        else:
            m = self.seg.metrics().round(4)
            self.out_metrics.value = (self._metrics_table_html(m)
                                      + self._metrics_guide_html(list(m.columns)))
        # shift DES→OOT das principais métricas (só quando há OOT para comparar).
        # O título vem do layout (row_shift_dist); aqui vai só a figura, ampliada
        # e esticada p/ preencher a coluna ao lado da distribuição do score.
        if self.seg.metric_shifts():
            try:
                self.out_metric_shift.value = self._fig_html(
                    self.seg.plot_metric_shift(figsize=(8.4, 4.7)), stretch=True)
            except Exception as e:
                self.out_metric_shift.value = f"<i>{e}</i>"
        else:
            self.out_metric_shift.value = ("<div class='mseg-legend'>Sem amostra OOT "
                                           "para comparar.</div>")

    def _render_metrics_twostage(self):
        """Três visões do Two-Stage: classificador (etapa 1), regressão (etapa 2,
        no grupo y≥t) e resposta combinada E[y] (base do rating)."""
        t = self.seg.two_stage_threshold or 0.0
        anchor = getattr(self.seg.model, "anchor0", 0.0)
        clf = self.seg.metrics_classifier().round(4)
        reg = self.seg.metrics_regressor().round(4)
        comb = self.seg.metrics().round(4)
        self.out_metrics.value = (
            f"<div class='mseg-legend'><b>Two-Stage (hurdle)</b> · threshold da resposta = "
            f"<b>{t:.4f}</b>. Etapa 1 classifica P(y ≥ t); etapa 2 regride y no grupo ≥ t; a "
            f"resposta final é <b>E[y] = P(≥t)·reg(x) + (1−P)·âncora</b> "
            f"(âncora = {anchor:.4f}).</div>"
            "<div class='mseg-h'>① Classificador · P(y ≥ threshold)</div>"
            + self._df_html(clf, center=True)
            + "<div class='mseg-h'>② Regressão · restrita ao grupo y ≥ threshold</div>"
            + self._df_html(reg, center=True)
            + "<div class='mseg-h'>③ Resposta combinada E[y] — base do rating</div>"
            + self._df_html(comb, center=True))

    def _auto_build_rating(self):
        """Best-effort: constrói e renderiza o rating sobre a resposta atual do
        modelo — chamado logo após o fit Two-Stage para já trazer o rating da
        resposta combinada. Métodos manuais (que exigem cortes) são deixados para
        a aba Ratings & Score."""
        method = self.dd_method.value
        if method in ("manual_score", "manual_percentil"):
            self._log("[ratings] método manual selecionado — gere o rating na aba "
                      "Ratings & Score informando os cortes.")
            return
        try:
            self.seg.build_ratings(method=method, n_ratings=int(self.sl_nratings.value),
                                   monotonic_fusion=self.cb_fusion.value)
            self._render_ratings()
            self._log(f"[ratings] gerado sobre a resposta combinada "
                      f"({len(self.seg.rating_labels_)} faixas) — ver a aba Ratings & Score.")
        except Exception as e:
            self._log(f"[ratings] não foi possível gerar automaticamente: {e}")

    def _reproject_ratings_after_refit(self):
        """Após um retreino que troca ``seg.score_`` (fit simples / tuning), mantém os
        ratings coerentes com o NOVO score: reprojeta a régua vigente sobre o novo
        score (mesmos cortes/fusão, via ``rating_strategy.transform`` — mesmo padrão de
        :meth:`apply_backward_selection`) e re-renderiza a aba. Sem ratings prévios é
        no-op; se não der para reprojetar, limpa a aba para não exibir/exportar ratings
        do modelo anterior."""
        seg = self.seg
        if getattr(seg, "rating_", None) is None:
            return                                        # nunca houve rating — nada a fazer
        if seg.rating_strategy is None:
            self._clear_ratings_outputs()
            self._log("[ratings] limpos após o retreino (sem estratégia para reprojetar).")
            return
        try:
            seg.rating_ = seg.rating_strategy.transform(
                seg._rating_frame(), seg._make_cfg("_amostra"))
            self._render_ratings()
            self._log(f"[ratings] reprojetados sobre o novo score "
                      f"({len(seg.rating_labels_)} faixas).")
        except Exception as e:
            self._clear_ratings_outputs()
            self._log(f"[ratings] limpos após o retreino (não foi possível "
                      f"reprojetar: {e}).")

    def _clear_ratings_outputs(self):
        """Invalida a régua de rating no ``seg`` e zera a aba Ratings & Score, para que
        nada (tabela, gráficos, ``assign``/export) use ratings de um modelo anterior."""
        self.seg.rating_ = None
        self.seg.rating_strategy = None
        self.seg.rating_labels_ = []
        self.seg.rating_config = {}
        for w in (self.out_rating_auto, self.out_rating_table, self.out_rating_badrate,
                  self.out_rating_dist, self.out_rating_inv_s, self.out_rating_inv_t,
                  self.out_rating_psi_sample, self.out_rating_psi_safra,
                  self.out_rating_mono):
            w.value = ""
        self._refresh_bar()

    def _metrics_table_html(self, m):
        """Tabela de métricas centralizada e com identificador visual: cada célula
        ganha cor (verde/amarelo/vermelho) conforme o guia de bolso da métrica."""
        metric_cols = [c for c in m.columns if c in self._METRIC_GUIDE]
        fmt = {c: "{:.4f}" for c in m.columns if c not in ("amostra", "n")}
        # centraliza cabeçalho E células: as regras `th`/`td` precisam vir DEPOIS do
        # `th, td {text-align:right}` de _TABLE_STYLES e com a mesma especificidade,
        # senão o `#T td` (id+elemento) vence o estilo por célula e volta p/ direita.
        sty = (m.style.hide(axis="index").set_table_styles(self._TABLE_STYLES)
               .set_table_styles([{"selector": "th", "props": [("text-align", "center")]},
                                  {"selector": "td", "props": [("text-align", "center"),
                                                               ("font-size", "12px")]}],
                                 overwrite=False)
               .format(fmt))

        def _color(col):
            return [self._LEVEL_CSS.get(self._metric_level(col.name, v), "") for v in col]

        if metric_cols:
            sty = sty.apply(_color, axis=0, subset=metric_cols)
        return sty.to_html()

    def _metrics_guide_html(self, cols):
        """Dicionário das métricas + guia de bolso (o que mede, recomendado, quando
        é baixa), num bloco recolhível, com a legenda de cores."""
        rows = []
        for c in cols:
            g = self._METRIC_GUIDE.get(c)
            if not g:
                continue
            rows.append(f"<tr><td class='term'><b>{g['nome']}</b> "
                        f"<span class='mono' style='color:var(--sub-ink)'>{c}</span></td>"
                        f"<td class='read'>{g['desc']}</td>"
                        f"<td class='read'>{g['rec']}</td>"
                        f"<td class='read'>{g['baixa']}</td></tr>")
        if not rows:
            return ""
        return (
            "<details class='mseg-guide'>"
            "<summary>📖 Dicionário de métricas & guia de bolso</summary>"
            "<div class='lg'>Identificador visual: "
            "<span class='mseg-cat mseg-cat-keep'>bom</span> "
            "<span class='mseg-cat mseg-cat-rev'>atenção</span> "
            "<span class='mseg-cat mseg-cat-drop'>ruim</span> — as faixas são regras "
            "de bolso; em crédito, compare também DES × OOT e contra um baseline.</div>"
            "<table class='mseg-coef'><thead><tr>"
            "<th class='term'>métrica</th><th class='read'>o que mede</th>"
            "<th class='read'>recomendado</th><th class='read'>considerada baixa</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></details>")

    def _render_model_plots(self):
        # linha 2 (calibração + resíduos · reg / ROC + KS · clf) e a distribuição
        # do score, que fica na linha 1 ao lado do shift. stretch=True ⇒ cada
        # gráfico preenche a largura da sua coluna.
        if self.task_type == "classification":
            fn_a, fn_b = self.seg.plot_roc, self.seg.plot_ks
        else:
            fn_a, fn_b = self.seg.plot_calibration, self.seg.plot_residuals
        specs = [(self.out_model_a, fn_a, (6.0, 5.0)),
                 (self.out_model_b, fn_b, (6.4, 5.0)),
                 (self.out_model_c, self.seg.plot_score_distribution, (6.4, 4.7))]
        for out, fn, fs in specs:
            try:
                out.value = self._fig_html(fn(figsize=fs), tight=False, stretch=True)
            except Exception as e:
                out.value = f"<i>{e}</i>"

    def _on_shap(self, b):
        if self.seg.score_ is None:
            self._log("[shap] treine o modelo primeiro.")
            return
        try:
            self.out_shap.value = self._fig_html(self.seg.plot_shap_beeswarm(sample_size=800))
            self.out_shap_bar.value = self._fig_html(self.seg.plot_shap_bar(sample_size=800))
            # tamanho natural (menor), não esticado à largura do painel como antes
            self.out_shap_rel.value = self._fig_html(
                self.seg.plot_shap_importance_relative(sample_size=800, figsize=(6.0, 3.4)))
            self._log("[shap] gráficos gerados.")
        except Exception as e:
            self.out_shap.value = f"<i>SHAP indisponível: {e}</i>"
            self._log(f"[shap] erro: {e}")

    # ------------------------------------------------------ Aba Backward Elimination
    def _on_backelim(self, b):
        import threading
        if self._backelim_thread is not None and self._backelim_thread.is_alive():
            self._log("[backward] já há uma execução em andamento.")
            return
        feats0 = (self.seg.model_features or self.seg.selected_features()
                  or self.seg.candidates)
        if len(feats0) < 2:
            self.out_backelim_status.value = ("<i>Selecione ao menos 2 variáveis (aba "
                                              "Variáveis) ou treine o modelo primeiro.</i>")
            self._log("[backward] menos de 2 variáveis disponíveis.")
            return
        sample = self.dd_backelim_sample.value
        # clamp ao conjunto vigente (como _on_feat_optimal): o max do slider é fixado
        # na build e pode ficar >= nº atual de variáveis se a seleção encolheu.
        min_features = max(1, min(int(self.sl_backelim_min.value), len(feats0) - 1))
        total = max(1, len(feats0) - min_features + 1)
        self.pb_backelim.max = total
        self.pb_backelim.value = 0
        self.pb_backelim.description = f"0/{total}"
        self.pb_backelim.bar_style = "info"
        self.pb_backelim.layout.visibility = "visible"
        self.btn_backelim.disabled = True
        self.out_backelim_status.value = "<i>Rodando backward elimination…</i>"

        def _progress(done, total_, nvar):
            self.pb_backelim.value = done
            self.pb_backelim.description = f"{done}/{total_}"
            self.out_backelim_status.value = (
                f"<div class='mseg-legend'>Passo {done}/{total_} · avaliando com "
                f"{nvar} variáveis…</div>")

        def _worker():
            try:
                res = self.seg.backward_elimination(sample=sample, min_features=min_features,
                                                    progress_callback=_progress)
            except Exception as e:
                self.pb_backelim.bar_style = "danger"
                self.out_backelim_status.value = (
                    f"<div style='color:var(--bad-tx);font-size:12px'>Erro: "
                    f"{type(e).__name__}: {e}</div>")
                self._log(f"[backward] erro: {e}")
                return
            finally:
                self.btn_backelim.disabled = False
            self._backelim_result = res
            self.pb_backelim.value = self.pb_backelim.max
            self.pb_backelim.bar_style = "success"
            self.pb_backelim.description = "concluído ✓"
            self.out_backelim_status.value = (
                f"<div class='mseg-legend'>Concluído · {len(res)} passos · avaliação em "
                f"<b>{res.attrs.get('eval_sample')}</b> (algoritmo "
                f"<code>{res.attrs.get('algorithm')}</code>).</div>")
            self._render_backelim_plot()
            self._render_backelim_table(res)      # ★ na linha do nº ótimo + habilita retreino
            self._log(f"[backward] concluído: {len(res)} passos (avaliação em "
                      f"{res.attrs.get('eval_sample')}).")

        self._backelim_thread = threading.Thread(target=_worker, daemon=True)
        self._backelim_thread.start()

    def _render_backelim_plot(self):
        """(Re)desenha o gráfico de métricas × nº de variáveis com as métricas
        escolhidas em ``sm_backelim_metrics`` (observer). No-op sem resultado."""
        res = self._backelim_result
        if res is None or len(res) == 0:
            return
        metrics = list(self.sm_backelim_metrics.value) or None
        y_range = None
        if self.task_type == "regression" and self.cb_backelim_yfix.value:
            y_range = (self.ft_backelim_ymin.value, self.ft_backelim_ymax.value)
        try:
            self.out_backelim_plot.value = self._fig_html(
                self.seg.plot_backward_elimination(res, metrics=metrics, y_range=y_range,
                                                   figsize=(9.6, 4.6)),
                stretch=True)
        except Exception as e:
            self.out_backelim_plot.value = f"<i>{e}</i>"

    def _render_backelim_table(self, res):
        """Renderiza a tabela do backward destacando (★ + negrito) a linha do nº de
        variáveis ÓTIMO (parcimônia · dentro de 1% da melhor métrica) e habilita o
        botão de retreino com esse subconjunto. No-op silencioso sem resultado."""
        if res is None or len(res) == 0:
            self.out_backelim_table.value = ""
            self._backelim_optimal = None
            self.dd_backelim_n.options = []
            self.dd_backelim_n.disabled = True
            self.btn_backelim_apply_n.disabled = True
            return
        target_n = None
        try:
            pick = self.seg.backward_optimal_step(res, criterion="parsimony", tol=0.01)
            self._backelim_optimal = pick
            target_n = int(pick["target_n"])
        except Exception:
            self._backelim_optimal = None
        disp = res.round(4).copy()
        ns = disp["n_variaveis"].astype(int)
        disp.insert(0, "★", ns.map(lambda n: "★" if n == target_n else ""))
        sty = (disp.style.hide(axis="index").set_table_styles(self._TABLE_STYLES)
               .set_properties(**{"font-size": "12px", "text-align": "center"})
               .set_table_styles([{"selector": "th, td",
                                   "props": [("text-align", "center")]}], overwrite=False))
        if target_n is not None:
            def _bold_optimal(row):                # linha ótima: negrito + fundo destacado
                hit = int(row["n_variaveis"]) == target_n
                return [("font-weight:700;background-color:var(--sus-bg)" if hit else "")] * len(row)
            sty = sty.apply(_bold_optimal, axis=1)
            sty = sty.map(lambda v: ("color:var(--sus-ink);font-weight:700" if v == "★" else ""),
                          subset=["★"])
        self.out_backelim_table.value = (
            f"<div style='max-height:380px;overflow:auto'>{sty.to_html()}</div>")
        self._sync_backelim_n_options(res, target_n)

    def _sync_backelim_n_options(self, res, target_n):
        """Popula o seletor manual de nº de variáveis com os passos do backward
        (rótulo com a métrica de cada passo; ★ marca o ótimo) e habilita o retreino.
        O default do seletor é o próprio nº ótimo."""
        try:
            _m = (self._backelim_optimal or {}).get("metric")
            if not _m:
                _prefs = (["ks", "auc", "gini"] if self.task_type == "classification"
                          else ["rmse", "mae", "smape"])
                _m = next((c for c in _prefs if c in res.columns), None)
            opts = []
            for _, r in res.iterrows():
                k = int(r["n_variaveis"])
                star = " ★" if k == target_n else ""
                if _m and _m in res.columns and pd.notna(r.get(_m)):
                    opts.append((f"{k} variáveis · {_m.upper()}={float(r[_m]):.4f}{star}", k))
                else:
                    opts.append((f"{k} variáveis{star}", k))
            self.dd_backelim_n.options = opts
            valid = [k for _, k in opts]
            self.dd_backelim_n.value = (target_n if target_n in valid
                                        else (valid[0] if valid else None))
            self.dd_backelim_n.disabled = not opts
            self.btn_backelim_apply_n.disabled = not opts
        except Exception:
            self.dd_backelim_n.options = []
            self.dd_backelim_n.disabled = True
            self.btn_backelim_apply_n.disabled = True

    def _apply_backward_optimal(self, res, out_widget=None, n_variaveis=None):
        """Aplica um subconjunto do backward e atualiza toda a UI: o passo ÓTIMO
        (parcimônia · 1%) ou, se ``n_variaveis`` for dado, o passo com esse nº de
        variáveis (escolha MANUAL). Troca a seleção, retreina, regenera/reprojeta os
        ratings, escreve o resumo em ``out_widget`` (se dado) e loga. Devolve o dict-info."""
        info = self.seg.apply_backward_selection(
            res, criterion="parsimony", tol=0.01, refit=True, rebuild_ratings=True,
            n_variaveis=n_variaveis)
        self._sync_sel(); self._refresh_vars(force=True)
        # modelo retreinado ⇒ atualiza métricas/gráficos/fórmula e limpa "dirty"
        self._render_metrics(); self._render_model_plots(); self._render_formula()
        self._clear_dirty(); self._clear_adv_outputs()
        if info.get("ratings_rebuilt") or info.get("ratings_reprojected"):
            self._render_ratings()            # ratings regenerados OU reprojetados no novo score
        self._refresh_bar()
        _rt = (" e ratings regenerados" if info.get("ratings_rebuilt")
               else " e ratings reprojetados" if info.get("ratings_reprojected") else "")
        feats = ", ".join(self.seg.label(f) for f in info["features"])
        _manual = info.get("criterion") == "manual"
        if out_widget is not None:
            if _manual:
                out_widget.value = (
                    f"<div class='mseg-legend'><b>Nº escolhido aplicado</b>: "
                    f"<b>{info['target_n']}</b> variável(is) — retreino no ponto da curva do "
                    f"backward selecionado. Modelo retreinado{_rt}.<br>"
                    f"<span style='color:var(--sub-ink)'>{feats}</span></div>")
            else:
                out_widget.value = (
                    f"<div class='mseg-legend'><b style='color:var(--sus-ink)'>★</b> "
                    f"<b>Escolha ótima aplicada</b>: <b>{info['target_n']}</b> variável(is) — "
                    f"critério parcimônia (métrica {str(info['metric']).upper()}). Modelo "
                    f"retreinado{_rt}.<br><span style='color:var(--sub-ink)'>{feats}</span></div>")
        self._log(f"[{'nº escolhido' if _manual else 'escolha ótima'}] "
                  f"{info['target_n']} variáveis + retreino{_rt}.")
        return info

    def _on_backelim_apply_n(self):
        """Retreina o modelo com o Nº de variáveis ESCOLHIDO na curva do backward
        (qualquer ponto, não só o ótimo). Usa o último resultado do backward."""
        res = getattr(self, "_backelim_result", None)
        if res is None or len(res) == 0:
            self.out_backelim_apply.value = (
                "<i>Rode o backward elimination primeiro (botão acima).</i>")
            return
        n = self.dd_backelim_n.value
        if n is None:
            self.out_backelim_apply.value = "<i>Escolha um nº de variáveis ao lado.</i>"
            return
        self.btn_backelim_apply_n.disabled = True
        try:
            self._apply_backward_optimal(res, self.out_backelim_apply, n_variaveis=int(n))
        except Exception as e:
            self.out_backelim_apply.value = f"<div style='color:var(--bad-tx)'>Erro: {e}</div>"
            self._log(f"[nº escolhido] erro: {e}")
        finally:
            self.btn_backelim_apply_n.disabled = False

    # ------------------------------------------------------------------ Aba 4 handlers
    def _on_suggest_n(self, b):
        if self.seg.score_ is None:
            self._log("[ratings] treine o modelo primeiro.")
            return
        try:
            sug = self.seg.suggest_n_ratings(method=self.dd_method.value,
                                             monotonic_fusion=self.cb_fusion.value)
            best = int(sug["best"])
            self.sl_nratings.value = max(self.sl_nratings.min,
                                         min(best, self.sl_nratings.max))
            self.out_rating_auto.value = self._render_suggestion(sug)
            self._log(f"[ratings] sugestão: {best} ratings (slider ajustado; "
                      f"clique em Gerar ratings).")
        except Exception as e:
            self.out_rating_auto.value = f"<i>{e}</i>"
            self._log(f"[ratings] erro na sugestão: {e}")

    def _render_suggestion(self, sug):
        t = sug["table"].copy()
        rec = int(sug["best"])
        t.insert(0, "rec", t["n_alvo"].map(lambda n: "★" if int(n) == rec else ""))
        t["ok"] = t["ok"].map(lambda v: "✓" if v else "—")
        if "gini" in t.columns:
            t["gini"] = t["gini"].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        return (
            "<div class='mseg-help'><div class='ttl'>Nº de ratings sugerido: "
            f"{rec}</div>{sug['reason']}"
            "<div class='mseg-legend'>Critério: monotonia de risco entre amostras "
            "(0 inversões), volume mínimo por faixa e ganho de discriminação (Gini). "
            "★ = recomendado · <code>ok</code> = passa nos critérios.</div></div>"
            + self._df_html(t, center=True, max_height="280px"))

    def _sync_rating_method(self):
        """Mostra o campo de cortes/percentis só nos métodos manuais e ajusta o
        placeholder; nº ratings e fusão não se aplicam aos manuais."""
        m = self.dd_method.value
        manual = m in ("manual_score", "manual_percentil")
        self.tx_manual.layout.display = "" if manual else "none"
        if m == "manual_score":
            self.tx_manual.placeholder = "cortes de score 0–1000 (ex.: 200, 500, 800)"
        elif m == "manual_percentil":
            self.tx_manual.placeholder = "percentis 0–100 (ex.: 20, 40, 60, 80)"
        self.sl_nratings.disabled = manual
        self.cb_fusion.disabled = manual

    def _on_build_ratings(self, b):
        if self.seg.score_ is None:
            self._log("[ratings] treine o modelo primeiro.")
            return
        method = self.dd_method.value
        kw = {}
        if method in ("manual_score", "manual_percentil"):
            try:
                nums = [float(x) for x in self.tx_manual.value.replace(";", ",").split(",")
                        if x.strip() != ""]
            except ValueError:
                self._log("[ratings] valores inválidos em cortes/percentis."); return
            if not nums:
                self._log("[ratings] informe os cortes (manual_score) ou percentis "
                          "(manual_percentil)."); return
            if method == "manual_score":
                # o usuário digita na escala de negócio (0–1000); build_ratings opera
                # no score CRU (0–1) → converte dividindo por score_scale.
                scale = getattr(self.seg, "score_scale", 1.0) or 1.0
                kw = {"cuts": [c / scale for c in nums]}
            else:
                kw = {"percentiles": nums}
        try:
            self.seg.build_ratings(method=method,
                                   n_ratings=int(self.sl_nratings.value),
                                   monotonic_fusion=self.cb_fusion.value, **kw)
            self._log(f"[ratings] {len(self.seg.rating_labels_)} faixas ({method}).")
            self._render_ratings()
        except Exception as e:
            self._log(f"[ratings] erro: {e}")

    # ---------------------------------------------------- zoom dos gráficos de inversão
    def _read_inv_ylim(self, tx_min, tx_max):
        """(baixo, alto) em fração a partir de dois campos de % (texto); lado
        vazio/inválido vira ``None``. Retorna ``None`` se ambos vazios."""
        def _p(t):
            s = (t.value or "").strip().replace("%", "").replace(",", ".")
            if not s:
                return None
            try:
                return float(s) / 100.0
            except ValueError:
                return None
        lo, hi = _p(tx_min), _p(tx_max)
        return None if (lo is None and hi is None) else (lo, hi)

    def _render_inv_sample(self):
        """Renderiza o gráfico de inversão por amostra respeitando o estado de zoom."""
        if getattr(self.seg, "rating_", None) is None:
            return
        try:
            ylim = self._read_inv_ylim(self.tx_inv_s_min, self.tx_inv_s_max)
            self.out_rating_inv_s.value = self._fig_html(
                self.seg.plot_rating_inversion_by_sample(
                    figsize=(8.4, 4.0), ylim=ylim, auto_zoom=self._inv_s_zoom),
                tight=False)
        except Exception as e:
            self.out_rating_inv_s.value = f"<i>{e}</i>"

    def _render_inv_safra(self):
        """Renderiza o gráfico de inversão por safra respeitando o estado de zoom."""
        if getattr(self.seg, "rating_", None) is None:
            return
        try:
            ylim = self._read_inv_ylim(self.tx_inv_t_min, self.tx_inv_t_max)
            self.out_rating_inv_t.value = self._fig_html(
                self.seg.plot_rating_inversion_by_safra(
                    figsize=(8.4, 4.0), ylim=ylim, auto_zoom=self._inv_t_zoom),
                tight=False)
        except Exception as e:
            self.out_rating_inv_t.value = f"<i>{e}</i>"

    def _on_zoom_inv_s(self, b):
        self._inv_s_zoom = True
        self.btn_zoom_inv_s.button_style = "info"
        self._render_inv_sample()

    def _on_reset_inv_s(self, b):
        self._inv_s_zoom = False
        self.btn_zoom_inv_s.button_style = ""
        self.tx_inv_s_min.value = ""
        self.tx_inv_s_max.value = ""
        self._render_inv_sample()

    def _on_zoom_inv_t(self, b):
        self._inv_t_zoom = True
        self.btn_zoom_inv_t.button_style = "info"
        self._render_inv_safra()

    def _on_reset_inv_t(self, b):
        self._inv_t_zoom = False
        self.btn_zoom_inv_t.button_style = ""
        self.tx_inv_t_min.value = ""
        self.tx_inv_t_max.value = ""
        self._render_inv_safra()

    def _render_ratings(self):
        """Renderiza tabela e gráficos dos ratings a partir do estado atual do
        ``seg`` (usado ao gerar ratings e ao carregar um modelo já ratingado)."""
        rt = self.seg.rating_table().round(4)
        rate_cols = [c for c in rt.columns if c.startswith(("event_rate", "alvo"))]
        self.out_rating_table.value = self._df_html(rt, center=True, pct_cols=rate_cols)
        self.out_rating_badrate.value = self._fig_html(self.seg.plot_rating_badrate())
        self.out_rating_dist.value = self._fig_html(self.seg.plot_rating_distribution())
        # os dois gráficos de inversão passam pelos helpers de zoom (mesma figsize +
        # tight=False ⇒ MESMA altura nas colunas 50/50), respeitando o estado de zoom.
        self._render_inv_sample()
        self.out_rating_mono.value = self._df_html(
            self.seg.monotonicity_report(), center=True)
        self._render_inv_safra()
        try:
            self.out_rating_psi_sample.value = self._fig_html(
                self.seg.plot_rating_psi_by_sample(figsize=(9.6, 4.2)), stretch=True)
        except Exception as e:
            self.out_rating_psi_sample.value = f"<i>{e}</i>"
        try:
            self.out_rating_psi_safra.value = self._fig_html(
                self.seg.plot_rating_psi_by_safra(figsize=(9.6, 4.2)), stretch=True)
        except Exception as e:
            self.out_rating_psi_safra.value = f"<i>{e}</i>"
        self._refresh_bar()

    # ------------------------------------------------------------------ Aba 5 handlers
    def _on_export(self, b):
        try:
            self.result = self.seg.assign()
            self.out_export.value = (f"<div class='mseg-legend'>DataFrame exportado: "
                                     f"{self.result.shape[0]} linhas × {self.result.shape[1]} colunas "
                                     f"(em <code>ui.result</code>).</div>")
            self._log("[export] DataFrame rotulado em ui.result.")
        except Exception as e:
            self._log(f"[export] erro: {e}")

    def _on_ruler(self, b):
        try:
            col_value = self.tx_value_col.value.strip() or "valor_previsto"
            ruler = self.seg.rating_ruler(col_value=col_value)
            self.out_ruler.value = self._df_html(ruler.round(6), max_height="280px")
            self._log(f"[régua] {len(ruler)} ratings (valor na amostra "
                      f"{self.seg.ref_sample}).")
        except Exception as e:
            self.out_ruler.value = f"<i>{e}</i>"
            self._log(f"[régua] erro: {e}")

    def _on_score(self, b):
        col_value = self.tx_value_col.value.strip() or "valor_previsto"
        recreate = self.cb_recreate.value
        in_tbl = self.tx_in_table.value.strip()
        out_tbl = self.tx_out_table.value.strip() or None
        self._score_steps = []                            # zera a tabela de progresso
        self._render_score_progress()
        self.out_score.value = ""                         # limpa a saída anterior
        self.btn_score.disabled = True
        cb = self._score_progress_cb
        try:
            if in_tbl:                                   # tabela do Databricks (Spark)
                sout = self.seg.score_table(in_tbl, col_value=col_value,
                                            recreate_categories=recreate,
                                            output_table=out_tbl, progress_callback=cb)
                self.result = sout                       # Spark DataFrame (lazy)
                ncols = len(sout.columns)                # só o schema — não dispara compute
                # sem PRÉVIA da base na interface (não damos display do escorado): evita
                # também o toPandas() extra que reexecutava a escoragem no driver.
                gravou = (f" Gravado em <code>{out_tbl}</code>." if out_tbl
                          else " Spark DataFrame (lazy) em <code>ui.result</code>.")
                self.out_score.value = (
                    f"<div class='mseg-legend'>✓ Tabela Databricks <code>{in_tbl}</code> "
                    f"escorada ({ncols} colunas).{gravou}</div>")
                self._log(f"[escorar] tabela '{in_tbl}' escorada ({ncols} colunas)"
                          + (f" e gravada em '{out_tbl}'." if out_tbl else " (ui.result)."))
                return
            # em memória (pandas): base carregada ou ui.score_df
            base = self.score_df if self.score_df is not None else self.seg.df
            origem = "ui.score_df" if self.score_df is not None else "base carregada"
            out = self.seg.score_table(base, col_value=col_value,
                                       recreate_categories=recreate, progress_callback=cb)
            self.result = out
            novas = [c for c in out.columns if c not in base.columns]
            cols = ", ".join(f"<code>{c}</code>" for c in novas)
            # sem prévia da base na interface — só a confirmação; os dados ficam em ui.result
            self.out_score.value = (
                f"<div class='mseg-legend'>✓ Base escorada ({origem}): "
                f"{out.shape[0]} linhas × {out.shape[1]} colunas, em <code>ui.result</code>. "
                f"Colunas adicionadas: {cols}.</div>")
            self._log(f"[escorar] {origem} escorada em ui.result ({out.shape[0]} linhas).")
        except Exception as e:
            # o erro vai para o CONSOLE (log), não fica na área da base escorada: a
            # interface só mostra um aviso curto apontando o Console.
            for row in reversed(self._score_steps):
                if row["status"] == "run":
                    row["status"] = "err"
                    row["detail"] = type(e).__name__       # detalhe completo só no Console
                    break
            self._render_score_progress()
            self.out_score.value = ("<div class='mseg-legend' style='color:var(--bad-ink)'>✗ Falha na "
                                    "escoragem — veja o <b>Console</b> (rodapé) para o detalhe."
                                    "</div>")
            self._log(f"[escorar] ERRO: {type(e).__name__}: {e}")
        finally:
            self.btn_score.disabled = False

    def _score_progress_cb(self, key, label, status, detail=""):
        """Callback de progresso da escoragem (passado a ``score_table``): cria ou
        atualiza a linha da etapa ``key`` e re-renderiza a tabela de progresso."""
        for row in self._score_steps:
            if row["key"] == key:
                row["status"] = status
                if detail:
                    row["detail"] = detail
                break
        else:
            self._score_steps.append({"key": key, "label": label,
                                      "status": status, "detail": detail})
        self._render_score_progress()

    def _render_score_progress(self):
        """Renderiza a tabela de progresso da escoragem (carregando/escorando/salvando)."""
        if not self._score_steps:
            self.out_score_progress.value = ""
            return
        icon = {"run": "⏳", "ok": "✅", "err": "❌"}
        cor = {"run": "var(--warn-ink)", "ok": "var(--ok-ink)", "err": "var(--bad-ink)"}
        rot = {"run": "processando…", "ok": "concluído", "err": "erro"}
        trs = ""
        for r in self._score_steps:
            st = r["status"]
            trs += (f"<tr><td style='padding:4px 10px'>{icon.get(st, '')}</td>"
                    f"<td style='padding:4px 10px'>{r['label']}</td>"
                    f"<td style='padding:4px 10px;color:{cor.get(st, 'var(--ink)')};font-weight:600'>"
                    f"{rot.get(st, st)}</td>"
                    f"<td style='padding:4px 10px;color:var(--muted)'>{r.get('detail', '')}</td></tr>")
        self.out_score_progress.value = (
            "<div class='mseg-legend' style='margin-top:6px'>Progresso da escoragem</div>"
            "<table style='border-collapse:collapse;font-size:12px;width:100%;margin:2px 0 8px'>"
            "<thead><tr style='background:var(--tbl-head-bg)'>"
            "<th style='padding:4px 10px'></th>"
            "<th style='padding:4px 10px;text-align:left'>Etapa</th>"
            "<th style='padding:4px 10px;text-align:left'>Status</th>"
            "<th style='padding:4px 10px;text-align:left'>Detalhe</th>"
            f"</tr></thead><tbody>{trs}</tbody></table>")

    def _confirm_overwrite(self, path, do_save):
        """Se ``path`` já existir, mostra a confirmação INLINE (área dedicada no
        card de persistência, sob o campo de caminho) e só executa ``do_save()``
        quando o usuário clica em 'Sobrescrever'. Sem conflito (ou ``path`` vazio),
        salva direto. Diferente do diálogo antigo (desenhado no console do rodapé,
        que se perdia a cada ``clear_output``), o inline fica visível e persistente."""
        import html as _html
        import os
        path = (path or "").strip()
        if not path or not os.path.exists(path):
            do_save(); return
        self._ow_pending = do_save
        self.out_overwrite_msg.value = (
            "<div style='border:1px solid var(--notice-border);background:var(--notice-bg);"
            "border-radius:10px;"
            "padding:10px 12px;font-size:12.5px;color:var(--notice-ink);line-height:1.5'>"
            "<b>⚠️ O arquivo já existe</b><br>"
            f"<code>{_html.escape(path)}</code><br>Deseja sobrescrever?</div>")
        self.box_overwrite.layout.display = ""      # revela o diálogo inline
        self.btn_save.disabled = True               # trava Salvar enquanto pendente

    def _on_overwrite_yes(self, b):
        do_save = self._ow_pending
        self._ow_pending = None
        self.box_overwrite.layout.display = "none"
        self.out_overwrite_msg.value = ""
        self.btn_save.disabled = False
        if do_save is not None:
            do_save()

    def _on_overwrite_no(self, b):
        self._ow_pending = None
        self.box_overwrite.layout.display = "none"
        self.out_overwrite_msg.value = ""
        self.btn_save.disabled = False
        self._log("[save] cancelado — o arquivo não foi sobrescrito.")

    def _on_save(self, b):
        path = (self.tx_save.value or "").strip()
        self._confirm_overwrite(path, lambda: self._do_save(path))

    def _do_save(self, path):
        try:
            self.seg.save(path)
            self._log(f"[save] salvo em {path} (+ .model.joblib).")
        except Exception as e:
            self._log(f"[save] erro: {e}")

    def _on_pdf(self, b):
        if self.seg.score_ is None:
            self.out_pdf.value = "<i>Treine o modelo antes de gerar o relatório.</i>"; return
        path = (self.tx_pdf.value or "").strip()
        if not path:
            self.out_pdf.value = "<i>Informe o caminho do .pdf.</i>"; return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            self.seg.report_pdf(path)
        except Exception as e:
            self.out_pdf.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro ao gerar PDF: "
                                  f"{type(e).__name__}: {e}</div>")
            self._log(f"[pdf] erro: {e}"); return
        self.out_pdf.value = f"<div class='mseg-legend'>✅ Relatório salvo em <code>{path}</code>.</div>"
        self._log(f"[pdf] relatório salvo em {path}")

    def _on_md(self, b):
        if self.seg.score_ is None:
            self.out_md.value = "<i>Treine o modelo antes de gerar o relatório.</i>"; return
        path = (self.tx_md.value or "").strip()
        if not path:
            self.out_md.value = "<i>Informe o caminho do .md.</i>"; return
        if not path.lower().endswith(".md"):
            path += ".md"
        try:
            self.seg.report_markdown(path)
        except Exception as e:
            self.out_md.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro ao gerar "
                                 f"Markdown: {type(e).__name__}: {e}</div>")
            self._log(f"[md] erro: {e}"); return
        self.out_md.value = (f"<div class='mseg-legend'>✅ Relatório salvo em <code>{path}</code> "
                             "(imagens salvas ao lado).</div>")
        self._log(f"[md] relatório salvo em {path}")

    def _on_load(self, b):
        try:
            self.seg = self.seg.load(self.tx_save.value, self.df)
            self.refresh_model()
            self._log(f"[load] carregado de {self.tx_save.value}.")
        except Exception as e:
            self._log(f"[load] erro: {e}")

    def refresh_model(self):
        """Sincroniza a UI com o estado atual de ``self.seg`` — útil depois de
        carregar (botão 'Carregar') **ou** de injetar um modelo por fora
        (``ui.seg.set_model(...)`` / ``ui.seg = ...``). Ajusta o algoritmo
        selecionado, atualiza a barra/variáveis e re-renderiza métricas, gráficos,
        fórmula e — se já houver — os ratings, sem precisar re-treinar."""
        s = self.seg
        # o load pode ter trazido outro conjunto de candidatas / variáveis
        # derivadas — reconstrói as listas ANTES de sincronizar a seleção, senão
        # _sync_sel atribui valores fora das options (TraitError, aborta o refresh).
        self._refresh_candidates()
        # espelha o algoritmo do modelo no dropdown, quando for uma opção válida
        algo = getattr(s, "algorithm", None)
        valid = {v for _, v in self.dd_algo.options}
        if algo in valid and self.dd_algo.value != algo:
            self.dd_algo.value = algo          # dispara _sync_algo_visibility
        self._sync_hyperparam_widgets()        # controles refletem o modelo carregado
        self._sync_twostage_widgets()          # reflete o modo Two-Stage carregado
        self._clear_dirty()                    # modelo carregado está "em dia"
        self._clear_adv_outputs()
        self._refresh_bar(); self._refresh_vars(force=True); self._sync_sel()
        if s.score_ is not None:
            self.out_fit_status.value = ("<div style='color:var(--ok-ink);font-size:12px'>"
                                         "<b>Modelo carregado</b> — pronto para métricas, "
                                         "ratings e escoragem.</div>")
            try:
                self._render_metrics(); self._render_model_plots(); self._render_formula()
            except Exception as e:
                self._log(f"[load] falha ao renderizar métricas/gráficos: {e}")
        if getattr(s, "rating_", None) is not None:
            try:
                self._render_ratings()
            except Exception as e:
                self._log(f"[load] falha ao renderizar ratings: {e}")

    def _sync_hyperparam_widgets(self):
        """Espelha ``seg.hyperparams`` (modelo carregado/treinado) nos widgets de
        hiperparâmetro — sem isso, após 'Carregar' os controles mostram os defaults
        e um novo 'Treinar' usaria outros valores que não os do modelo em memória.
        Valores fora do range de um widget são ignorados (try/except)."""
        hp = getattr(self.seg, "hyperparams", None) or {}

        def _set(w, key, cast, enable=None):
            if key in hp and w is not None:
                try:
                    w.value = cast(hp[key])
                    if enable is not None:      # habilita o checkbox do parâmetro avançado
                        enable.value = True
                except Exception:               # noqa: BLE001 - fora de range ⇒ ignora
                    pass

        _set(self.tx_C, "C", float)
        _set(self.sl_n_est, "n_estimators", int)
        _set(self.sl_max_depth, "max_depth", int, self.cb_max_depth)
        _set(self.tx_lr, "learning_rate", float)
        _set(self.sl_min_leaf, "min_samples_leaf", int, self.cb_min_leaf)
        _set(self.fl_subsample, "subsample", float, self.cb_subsample)
        _set(self.fl_colsample, "colsample_bytree", float, self.cb_colsample)
        _set(self.sl_num_leaves, "num_leaves", int, self.cb_num_leaves)
        l2 = _l2_param_for(self.dd_algo.value)   # nome do L2 varia por algoritmo
        if l2:
            _set(self.fl_l2, l2, float, self.cb_l2)

    def _sync_twostage_widgets(self):
        """Reflete o estado Two-Stage do modelo (após load/set_model) nos controles:
        checkbox, threshold e algoritmos das duas etapas."""
        if self.task_type != "regression" or not hasattr(self, "cb_twostage"):
            return
        two = bool(getattr(self.seg, "two_stage", False))
        if self.cb_twostage.value != two:
            self.cb_twostage.value = two       # observer sincroniza a visibilidade
        if two:
            t = self.seg.two_stage_threshold
            if t is not None:
                try:
                    self.sl_ts_threshold.value = min(max(float(t), self.sl_ts_threshold.min),
                                                     self.sl_ts_threshold.max)
                except Exception:              # noqa: BLE001 - fora de range ⇒ ignora
                    pass
            hp = getattr(self.seg, "hyperparams", {}) or {}
            ca, ra = hp.get("clf_algorithm"), hp.get("reg_algorithm")
            if ca in [v for _, v in self.dd_ts_clf.options]:
                self.dd_ts_clf.value = ca
            if ra in [v for _, v in self.dd_ts_reg.options]:
                self.dd_ts_reg.value = ra
        self._sync_algo_visibility()

    def _on_mlflow(self, b):
        # validações antes de rodar (mesmo espírito da TreeSegmenterUI): exige
        # modelo treinado e, se o nome parecer Unity Catalog, o formato de 3 níveis.
        if self.seg.score_ is None:
            self._log("[mlflow] treine o modelo antes de registrar.")
            return
        model_name = (self.tx_model.value or "").strip()
        if model_name and "." in model_name and model_name.count(".") != 2:
            self._log(f"[mlflow] nome inválido: '{model_name}'. Para o Unity Catalog use "
                      "3 níveis (catalogo.schema.modelo).")
            return
        with self._busy(self.btn_mlflow, msg="registrando no MLflow…"):
            try:
                rid = self.seg.log_to_mlflow(experiment=self.tx_experiment.value or None,
                                             registered_model_name=model_name or None,
                                             verbose=False)
                self._log(f"[mlflow] run_id = {rid}")
            except Exception as e:
                self._log(f"[mlflow] erro: {e}")

    # ------------------------------------------------------------------ Aba 6 (Avançado) handlers
    def _adv_sample(self):
        """Amostra escolhida na aba Avançado ("(referência)" → None = ref_sample)."""
        v = self.dd_sample_adv.value
        return None if v == "(referência)" else v

    def _clear_adv_outputs(self):
        """Zera as saídas da aba Avançado — chamado ao re-treinar para não deixar
        gráficos/tabelas do modelo ANTIGO na tela."""
        for w in (self.out_adv_cap, self.out_adv_lift, self.out_adv_msafra_tab,
                  self.out_adv_msafra_fig, self.out_adv_backtest, self.out_adv_varprofile):
            w.value = ""

    def _on_adv_cap(self, b):
        if self.seg.score_ is None:
            self.out_adv_cap.value = "<i>Treine o modelo primeiro.</i>"; return
        if self.task_type != "classification":
            self.out_adv_cap.value = ("<div class='mseg-legend'>A curva CAP é exclusiva de "
                                      "classificação (eventos binários). Em regressão, use "
                                      "calibração/resíduos na aba Modelo.</div>"); return
        with self._busy(self.btn_cap, self.btn_lift, msg="gerando a curva CAP…"):
            try:
                self.out_adv_cap.value = self._fig_html(self.seg.plot_cap(), stretch=True)
                self._log("[avançado] curva CAP gerada.")
            except Exception as e:
                self.out_adv_cap.value = f"<i>{e}</i>"
                self._log(f"[avançado] erro na CAP: {e}")

    def _on_adv_lift(self, b):
        if self.seg.score_ is None:
            self.out_adv_lift.value = "<i>Treine o modelo primeiro.</i>"; return
        if self.task_type != "classification":
            self.out_adv_lift.value = ("<div class='mseg-legend'>Lift/Gains é exclusivo de "
                                       "classificação (eventos binários).</div>"); return
        with self._busy(self.btn_cap, self.btn_lift, msg="gerando lift/gains…"):
            try:
                self.out_adv_lift.value = self._fig_html(
                    self.seg.plot_lift(sample=self._adv_sample()), stretch=True)
                self._log("[avançado] lift/gains gerado.")
            except Exception as e:
                self.out_adv_lift.value = f"<i>{e}</i>"
                self._log(f"[avançado] erro no lift: {e}")

    def _on_adv_msafra(self, b):
        if self.seg.score_ is None:
            self.out_adv_msafra_tab.value = "<i>Treine o modelo primeiro.</i>"; return
        if not (self.date_col or "").strip():
            self.out_adv_msafra_tab.value = ("<div class='mseg-legend'>Informe a coluna de "
                                             "data (<code>date_col</code>) para métricas por "
                                             "safra.</div>")
            self.out_adv_msafra_fig.value = ""; return
        with self._busy(self.btn_msafra, msg="calculando métricas por safra…"):
            try:
                ms = self.seg.metrics_by_safra(sample=self._adv_sample())
                self.out_adv_msafra_tab.value = self._df_html(ms.round(4),
                                                              max_height="300px", center=True)
                mets = tuple(self.sm_msafra_metrics.value) or (
                    ("ks", "auc") if self.task_type == "classification" else ("rmse", "mae"))
                self.out_adv_msafra_fig.value = self._fig_html(
                    self.seg.plot_metrics_by_safra(sample=self._adv_sample(), metrics=mets),
                    stretch=True)
                self._log("[avançado] métricas por safra calculadas.")
            except Exception as e:
                self.out_adv_msafra_tab.value = f"<i>{e}</i>"
                self.out_adv_msafra_fig.value = ""
                self._log(f"[avançado] erro nas métricas por safra: {e}")

    def _on_adv_backtest(self, b):
        if self.seg.score_ is None:
            self.out_adv_backtest.value = "<i>Treine o modelo primeiro.</i>"; return
        if not (self.date_col or "").strip():
            self.out_adv_backtest.value = ("<div class='mseg-legend'>Informe a coluna de "
                                           "data (<code>date_col</code>) para o backtest por "
                                           "safra.</div>"); return
        with self._busy(self.btn_backtest_plot, msg="gerando o backtest gráfico…"):
            try:
                tol = float(self.tx_tol_adv.value) / 100.0
                self.out_adv_backtest.value = self._fig_html(
                    self.seg.plot_backtest(sample=self._adv_sample(), tolerancia=tol),
                    stretch=True)
                self._log("[avançado] backtest gráfico gerado.")
            except Exception as e:
                self.out_adv_backtest.value = f"<i>{e}</i>"
                self._log(f"[avançado] erro no backtest gráfico: {e}")

    def _on_adv_varprofile(self, b):
        if not (self.date_col or "").strip():
            self.out_adv_varprofile.value = ("<div class='mseg-legend'>Informe a coluna de "
                                             "data (<code>date_col</code>) para o perfil por "
                                             "safra.</div>"); return
        feats = (self.seg.model_features or self.seg.selected_features()
                 or self.seg.candidates)
        if not feats:
            self.out_adv_varprofile.value = ("<i>Nenhuma variável no modelo — treine ou "
                                             "selecione variáveis primeiro.</i>"); return
        with self._busy(self.btn_varprofile,
                        msg="calculando o perfil das variáveis por safra…"):
            try:
                tab = self.seg.variables_profile_by_safra()
                self.out_adv_varprofile.value = self._df_html(tab, max_height="420px",
                                                              center=True)
                self._log(f"[avançado] perfil das variáveis por safra ({len(feats)} variáveis).")
            except Exception as e:
                self.out_adv_varprofile.value = f"<i>{e}</i>"
                self._log(f"[avançado] erro no perfil das variáveis: {e}")

    # ------------------------------------------------------------------ display
    def _ipython_display_(self):
        display(self.panel)

    def display(self):
        display(self.panel)
