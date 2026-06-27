"""
ModelSegmenterUI
================
Camada interativa (ipywidgets) sobre o :class:`ModelSegmenter` — unifica a UI de
**classificação** e **regressão** via ``task_type``. Replica o estilo e os estudos
do ``LGDSegmenterUI`` (abas, IV, inversão, placar), porém para um fluxo orientado
a modelo:

* **① Variáveis** — analisa cada variável (logodds/WoE, IV, inversão) e decide
  o que entra no modelo (incluir/categorizar; auto-seleção por IV/PSI/monotonia);
* **② Análise de variáveis** — mergulho por variável: logodds por faixa,
  distribuição, inversão entre amostras/safras, série temporal e PSI por safra;
* **③ Modelo** — escolhe o algoritmo (Logística/Linear, RandomForest,
  GradientBoosting) e treina (ou usa um modelo pré-ajustado); métricas por amostra,
  gráficos do modelo e **SHAP**;
* **④ Ratings & Score** — segmenta o score em ratings (decis/quantil/árvore/optbin),
  com o número escolhido pelo usuário; tabela, badrate, distribuição e inversão
  entre ratings;
* **⑤ Validar & Exportar** — backtest por safra, PSI, exportar DataFrame rotulado,
  salvar/carregar e registrar no MLflow.

    from yggdrasil.credit_risk.model import ModelSegmenterUI
    ui = ModelSegmenterUI(df, target="target", task_type="classification",
                          sample_col="amostra", ref_sample="DES", date_col="dt_ref")
    ui
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import ipywidgets as W
    from IPython.display import display
except Exception as e:  # pragma: no cover
    raise ImportError("Este módulo requer ipywidgets e IPython (Jupyter).") from e

from .segmenter import ALGORITHMS, ModelSegmenter

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
.mseg { --ac:#3b4a63; --ac-deep:#27324a; --ac-soft:#eef1f5; --ac-border:#cdd5e0;
  --ink:#1f2733; --muted:#6b7480; --line:#e7e9ee;
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
.pill-green  { background:#e7f5ee; color:#157a52; }
.pill-yellow { background:#fbf3e0; color:#9a6f12; }
.pill-red    { background:#fbe7e4; color:#b23a2a; }
.mseg-legend { font-size:11px; color:var(--muted); margin:6px 0 2px; line-height:1.55; }
.mseg-metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(96px,1fr)); gap:6px; }
.mseg-metric { background:#f7f8fa; border:1px solid #eef0f3; border-radius:9px; padding:7px 10px; }
.mseg-metric .k { font-size:10px; text-transform:uppercase; letter-spacing:.04em; color:#8a93a3; }
.mseg-metric .v { font-size:16px; font-weight:600; color:var(--ink); margin-top:2px; }
.mseg-tabs { margin-top:10px; }
.mseg-tabs > .widget-tab-contents { padding:14px 2px 2px; background:transparent; }
.mseg-tabs .lm-TabBar-tab, .mseg-tabs .p-TabBar-tab { font-size:13px;
  min-width:max-content !important; max-width:none !important; flex:0 0 auto !important;
  margin:0 7px 0 0 !important; padding:7px 15px !important;
  border:1px solid var(--ac-border) !important; border-radius:10px !important;
  background:#fff !important; color:var(--muted) !important; font-weight:500; }
.mseg-tabs .lm-TabBar-tab.lm-mod-current, .mseg-tabs .p-TabBar-tab.p-mod-current {
  color:#fff !important; font-weight:600; background:var(--ac) !important;
  border-color:var(--ac) !important; }
.mseg .jupyter-button { border-radius:8px; font-family:inherit; }
.mseg .jupyter-widgets { min-width:0 !important; }
</style>
"""


class ModelSegmenterUI:
    _TABLE_STYLES = [
        {"selector": "", "props": [("border-collapse", "collapse"),
                                   ("border", "1px solid #cdd5e0"), ("width", "100%")]},
        {"selector": "th, td", "props": [("border", "1px solid #e1e5ec"),
                                         ("padding", "4px 9px"), ("text-align", "right"),
                                         ("white-space", "nowrap")]},
        {"selector": "thead th", "props": [("background-color", "#eef1f5"),
                                           ("color", "#27324a"), ("font-weight", "600"),
                                           ("border-bottom", "2px solid #b9c2d0"),
                                           ("position", "sticky"), ("top", "0"), ("z-index", "1")]},
        {"selector": "tbody tr:nth-child(even) td", "props": [("background-color", "#fafbfc")]},
        {"selector": "tbody tr:hover td", "props": [("background-color", "#eef3f8")]},
    ]

    def __init__(self, df, target="target", task_type="classification", sample_col=None,
                 ref_sample="DES", feature_labels=None, features=None, date_col=None):
        self.seg = ModelSegmenter(df, target=target, task_type=task_type,
                                  sample_col=sample_col, ref_sample=ref_sample,
                                  feature_labels=feature_labels, features=features,
                                  date_col=date_col, verbose=False)
        self.df = df
        self.task_type = task_type
        self.date_col = date_col
        self.result = None
        self._build()
        self._refresh_bar()
        self._refresh_vars()

    # ------------------------------------------------------------------ render utils
    def _fig_html(self, fig, border=False):
        import base64
        import io as _io
        import matplotlib.pyplot as plt
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=fig.get_dpi(), bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        style = "max-width:100%;height:auto"
        if border:
            style += ";border:1px solid #e6e8eb;border-radius:6px"
        return f"<img src='data:image/png;base64,{b64}' style='{style}'/>"

    def _df_html(self, df, max_height=None):
        sty = (df.style.hide(axis="index").set_table_styles(self._TABLE_STYLES)
               .set_properties(**{"font-size": "12px"}))
        txt = [c for c in df.columns if df[c].dtype == object]
        if txt:
            sty = sty.set_properties(subset=txt, **{"text-align": "left"})
        html = sty.to_html()
        if max_height:
            html = f"<div style='max-height:{max_height};overflow:auto'>{html}</div>"
        return html

    def _log(self, msg):
        with self.out_log:
            print(msg)

    @staticmethod
    def _pill(text, cls="muted"):
        return f"<span class='pill pill-{cls}'>{text}</span>"

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

        # ---------- Aba 1: Variáveis ----------
        self.dd_var = W.Dropdown(options=cands, description="Variável:",
                                 style={"description_width": "initial"})
        self.sel_included = W.SelectMultiple(options=cands, value=tuple(self.seg.included),
                                             rows=min(10, max(4, len(cands))),
                                             description="No modelo:",
                                             style={"description_width": "initial"},
                                             layout=W.Layout(width="98%"))
        self.dd_categoria = W.Dropdown(options=["—", "manter", "revisar", "descartar"],
                                       value="—", description="Categoria:",
                                       style={"description_width": "initial"})
        self.sl_min_iv = W.FloatSlider(value=0.02, min=0.0, max=0.5, step=0.01,
                                       description="IV mín.", readout_format=".2f")
        self.sl_max_psi = W.FloatSlider(value=0.25, min=0.0, max=1.0, step=0.05,
                                        description="PSI máx.", readout_format=".2f")
        self.cb_require_mono = W.Checkbox(value=False, description="exigir monotonia")
        self.btn_auto = W.Button(description="Auto-selecionar", button_style="primary",
                                 icon="magic")
        self.btn_apply_sel = W.Button(description="Aplicar seleção", button_style="success",
                                      icon="check")
        self.btn_set_cat = W.Button(description="Categorizar", icon="tag")
        self.btn_incl_all = W.Button(description="Incluir todas", icon="plus")
        self.btn_clear = W.Button(description="Limpar", icon="trash")
        self.btn_refresh_vars = W.Button(description="Recalcular", icon="refresh")
        self.out_vars = W.HTML()
        self.out_var_preview = W.HTML()

        self.btn_auto.on_click(self._on_auto_select)
        self.btn_apply_sel.on_click(self._on_apply_sel)
        self.btn_set_cat.on_click(self._on_set_cat)
        self.btn_incl_all.on_click(lambda b: (self.seg.include_all(), self._sync_sel(),
                                              self._refresh_vars(), self._refresh_bar()))
        self.btn_clear.on_click(lambda b: (self.seg.clear_features(), self._sync_sel(),
                                           self._refresh_vars(), self._refresh_bar()))
        self.btn_refresh_vars.on_click(lambda b: self._refresh_vars())
        self.dd_var.observe(lambda c: self._refresh_var_preview(), names="value")

        tab_vars = W.VBox([
            W.HTML("<div class='mseg-h'>Seleção & categorização de variáveis</div>"),
            W.HBox([self.sl_min_iv, self.sl_max_psi, self.cb_require_mono, self.btn_auto]),
            W.HBox([self.sel_included, W.VBox([self.btn_apply_sel, self.btn_incl_all,
                                               self.btn_clear, self.btn_refresh_vars])]),
            W.HBox([self.dd_var, self.dd_categoria, self.btn_set_cat]),
            W.HBox([W.VBox([W.HTML("<div class='mseg-h'>Ranking (IV / força / inversão / PSI)</div>"),
                            self.out_vars], layout=W.Layout(width="58%")),
                    W.VBox([W.HTML("<div class='mseg-h'>Logodds da variável</div>"),
                            self.out_var_preview], layout=W.Layout(width="42%"))]),
        ], layout=W.Layout(padding="2px"))

        # ---------- Aba 2: Análise de variáveis ----------
        self.dd_var2 = W.Dropdown(options=cands, description="Variável:",
                                  style={"description_width": "initial"})
        self.dd_sample2 = W.Dropdown(options=["(referência)"] + samples, value="(referência)",
                                     description="Amostra:",
                                     style={"description_width": "initial"})
        self.tx_time2 = W.Text(value=self.date_col or "", description="Coluna safra:",
                               style={"description_width": "initial"})
        self.btn_analyze = W.Button(description="Analisar variável", button_style="primary",
                                    icon="search")
        self.out_an_cards = W.HTML()
        self.out_an_logodds = W.HTML()
        self.out_an_dist = W.HTML()
        self.out_an_table = W.HTML()
        self.out_an_inv_sample = W.HTML()
        self.out_an_inv_safra = W.HTML()
        self.out_an_time = W.HTML()
        self.out_an_psi = W.HTML()
        self.btn_analyze.on_click(self._on_analyze)

        tab_an = W.VBox([
            W.HBox([self.dd_var2, self.dd_sample2, self.tx_time2, self.btn_analyze]),
            self.out_an_cards,
            W.HBox([W.VBox([W.HTML("<div class='mseg-h'>Logodds por faixa</div>"),
                            self.out_an_logodds], layout=W.Layout(width="50%")),
                    W.VBox([W.HTML("<div class='mseg-h'>Distribuição</div>"),
                            self.out_an_dist], layout=W.Layout(width="50%"))]),
            W.HBox([W.VBox([W.HTML("<div class='mseg-h'>Tabela por faixa</div>"),
                            self.out_an_table], layout=W.Layout(width="50%")),
                    W.VBox([W.HTML("<div class='mseg-h'>Inversão entre amostras</div>"),
                            self.out_an_inv_sample], layout=W.Layout(width="50%"))]),
            W.HBox([W.VBox([W.HTML("<div class='mseg-h'>Comportamento no tempo</div>"),
                            self.out_an_time], layout=W.Layout(width="50%")),
                    W.VBox([W.HTML("<div class='mseg-h'>Inversão por safra</div>"),
                            self.out_an_inv_safra], layout=W.Layout(width="50%"))]),
            W.VBox([W.HTML("<div class='mseg-h'>PSI por safra vs DES</div>"), self.out_an_psi]),
        ], layout=W.Layout(padding="2px"))

        # ---------- Aba 3: Modelo ----------
        self.dd_algo = W.Dropdown(options=algos, value=algos[0][1], description="Algoritmo:",
                                  style={"description_width": "initial"})
        self.sl_n_est = W.IntSlider(value=200, min=50, max=600, step=50,
                                    description="n_estimators")
        self.cb_max_depth = W.Checkbox(value=False, description="limitar max_depth")
        self.sl_max_depth = W.IntSlider(value=6, min=1, max=20, description="max_depth")
        self.tx_C = W.FloatText(value=1.0, description="C (regul.)",
                                style={"description_width": "initial"})
        self.btn_fit = W.Button(description="Treinar modelo", button_style="primary",
                                icon="cogs")
        self.btn_shap = W.Button(description="Calcular SHAP", icon="bar-chart")
        self.out_metrics = W.HTML()
        self.out_model_a = W.HTML()
        self.out_model_b = W.HTML()
        self.out_model_c = W.HTML()
        self.out_shap = W.HTML()
        self.out_shap_bar = W.HTML()
        self.btn_fit.on_click(self._on_fit)
        self.btn_shap.on_click(self._on_shap)

        tab_model = W.VBox([
            W.HTML("<div class='mseg-h'>Treinar (ou usar modelo pré-ajustado via set_model)</div>"),
            W.HBox([self.dd_algo, self.tx_C]),
            W.HBox([self.sl_n_est, self.cb_max_depth, self.sl_max_depth]),
            W.HBox([self.btn_fit, self.btn_shap]),
            W.VBox([W.HTML("<div class='mseg-h'>Métricas por amostra</div>"), self.out_metrics]),
            W.HBox([W.VBox([self.out_model_a], layout=W.Layout(width="34%")),
                    W.VBox([self.out_model_b], layout=W.Layout(width="33%")),
                    W.VBox([self.out_model_c], layout=W.Layout(width="33%"))]),
            W.HBox([W.VBox([W.HTML("<div class='mseg-h'>SHAP — beeswarm</div>"), self.out_shap],
                           layout=W.Layout(width="55%")),
                    W.VBox([W.HTML("<div class='mseg-h'>SHAP — importância</div>"), self.out_shap_bar],
                           layout=W.Layout(width="45%"))]),
        ], layout=W.Layout(padding="2px"))

        # ---------- Aba 4: Ratings & Score ----------
        self.dd_method = W.Dropdown(options=[("Decis", "decis"), ("Quantil + fusão", "quantil"),
                                             ("Árvore + fusão", "arvore"), ("OptBinning", "optbin")],
                                    value="quantil", description="Metodologia:",
                                    style={"description_width": "initial"})
        self.sl_nratings = W.IntSlider(value=10, min=2, max=20, description="nº ratings")
        self.cb_fusion = W.Checkbox(value=True, description="fusão monotônica (inversão)")
        self.btn_build_ratings = W.Button(description="Gerar ratings", button_style="primary",
                                          icon="sitemap")
        self.out_rating_table = W.HTML()
        self.out_rating_badrate = W.HTML()
        self.out_rating_dist = W.HTML()
        self.out_rating_inv_s = W.HTML()
        self.out_rating_inv_t = W.HTML()
        self.out_rating_mono = W.HTML()
        self.btn_build_ratings.on_click(self._on_build_ratings)

        tab_rating = W.VBox([
            W.HBox([self.dd_method, self.sl_nratings, self.cb_fusion, self.btn_build_ratings]),
            W.VBox([W.HTML("<div class='mseg-h'>Régua de ratings (risco por amostra)</div>"),
                    self.out_rating_table]),
            W.HBox([W.VBox([W.HTML("<div class='mseg-h'>Risco por rating</div>"),
                            self.out_rating_badrate], layout=W.Layout(width="50%")),
                    W.VBox([W.HTML("<div class='mseg-h'>Distribuição dos ratings</div>"),
                            self.out_rating_dist], layout=W.Layout(width="50%"))]),
            W.HBox([W.VBox([W.HTML("<div class='mseg-h'>Inversão entre ratings · amostras</div>"),
                            self.out_rating_inv_s], layout=W.Layout(width="50%")),
                    W.VBox([W.HTML("<div class='mseg-h'>Inversão entre ratings · safras</div>"),
                            self.out_rating_inv_t], layout=W.Layout(width="50%"))]),
            W.VBox([W.HTML("<div class='mseg-h'>Monotonicidade por amostra</div>"),
                    self.out_rating_mono]),
        ], layout=W.Layout(padding="2px"))

        # ---------- Aba 5: Validar & Exportar ----------
        self.tx_time3 = W.Text(value=self.date_col or "", description="Coluna safra:",
                               style={"description_width": "initial"})
        self.btn_backtest = W.Button(description="Backtest", button_style="primary",
                                     icon="calendar-check-o")
        self.out_backtest = W.HTML()
        self.out_psi = W.HTML()
        self.btn_export = W.Button(description="Exportar DataFrame", icon="download")
        self.out_export = W.HTML()
        self.tx_save = W.Text(value="modelo_segmenter.json", description="Arquivo:",
                              style={"description_width": "initial"})
        self.btn_save = W.Button(description="Salvar", button_style="success", icon="save")
        self.btn_load = W.Button(description="Carregar", icon="upload")
        self.tx_experiment = W.Text(value="", description="Experimento:",
                                    style={"description_width": "initial"})
        self.tx_model = W.Text(value="", description="Modelo (UC):",
                               style={"description_width": "initial"})
        self.btn_mlflow = W.Button(description="Registrar no MLflow", icon="database")
        self.btn_backtest.on_click(self._on_backtest)
        self.btn_export.on_click(self._on_export)
        self.btn_save.on_click(self._on_save)
        self.btn_load.on_click(self._on_load)
        self.btn_mlflow.on_click(self._on_mlflow)

        tab_export = W.VBox([
            W.HBox([self.tx_time3, self.btn_backtest]),
            W.HBox([W.VBox([W.HTML("<div class='mseg-h'>Backtest por safra</div>"),
                            self.out_backtest], layout=W.Layout(width="60%")),
                    W.VBox([W.HTML("<div class='mseg-h'>PSI dos ratings</div>"),
                            self.out_psi], layout=W.Layout(width="40%"))]),
            W.HBox([self.btn_export]), self.out_export,
            W.HTML("<div class='mseg-h'>Salvar / carregar (JSON + modelo joblib)</div>"),
            W.HBox([self.tx_save, self.btn_save, self.btn_load]),
            W.HTML("<div class='mseg-h'>Registrar no MLflow</div>"),
            W.HBox([self.tx_experiment, self.tx_model, self.btn_mlflow]),
        ], layout=W.Layout(padding="2px"))

        self.tabs = W.Tab(children=[tab_vars, tab_an, tab_model, tab_rating, tab_export])
        for i, t in enumerate(["① Variáveis", "② Análise de variáveis", "③ Modelo",
                               "④ Ratings & Score", "⑤ Validar & Exportar"]):
            self.tabs.set_title(i, t)
        self.tabs.add_class("mseg-tabs")

        console = W.VBox([W.HTML("<div class='mseg-h'>Console</div>"), self.out_log])
        console.add_class("mseg-card")
        self.panel = W.VBox([W.HTML(_CSS), self.banner, self.bar, self.tabs, console])
        self.panel.add_class("mseg")

    # ------------------------------------------------------------------ refresh
    def _refresh_bar(self):
        s = self.seg
        treinado = "sim" if s.score_ is not None else "não"
        nrat = len(s.rating_labels_) if s.rating_ is not None else 0
        self.banner.value = (
            "<div class='mseg-banner'><div class='logo'>MS</div><div>"
            f"<div class='t'>ModelSegmenter — {self.task_type}</div>"
            f"<div class='s'>alvo '{s.target}' · referência {s.ref_sample}</div></div></div>")
        self.bar.value = (
            "<div class='mseg-bar'>"
            + self._pill(f"task: {self.task_type}", "muted")
            + self._pill(f"candidatas: {len(s.candidates)}", "muted")
            + self._pill(f"incluídas: {len(s.included)}", "green")
            + self._pill(f"modelo treinado: {treinado}", "green" if s.score_ is not None else "yellow")
            + self._pill(f"ratings: {nrat}", "muted")
            + "</div>")

    def _sync_sel(self):
        self.sel_included.value = tuple(f for f in self.seg.candidates if f in self.seg.included)

    def _refresh_vars(self):
        try:
            rk = self.seg.variable_iv()
            for c in rk.columns:
                if c.startswith("psi_") or c in ("iv", "pior_psi"):
                    rk[c] = rk[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
            rk["incluida"] = rk["incluida"].map(lambda b: "✓" if b else "")
            rk["categoria"] = rk["categoria"].fillna("—")
            self.out_vars.value = self._df_html(rk, max_height="320px")
        except Exception as e:
            self.out_vars.value = f"<i>falha ao calcular IV: {e}</i>"
        self._refresh_var_preview()

    def _refresh_var_preview(self):
        try:
            fig = self.seg.plot_variable_logodds(self.dd_var.value)
            self.out_var_preview.value = self._fig_html(fig)
        except Exception as e:
            self.out_var_preview.value = f"<i>{e}</i>"

    # ------------------------------------------------------------------ Aba 1 handlers
    def _on_auto_select(self, b):
        try:
            self.seg.auto_select(min_iv=self.sl_min_iv.value, max_psi=self.sl_max_psi.value,
                                 require_monotonic=self.cb_require_mono.value)
            self._sync_sel(); self._refresh_vars(); self._refresh_bar()
            self._log(f"[auto] incluídas {len(self.seg.included)} variáveis.")
        except Exception as e:
            self._log(f"[auto] erro: {e}")

    def _on_apply_sel(self, b):
        self.seg.included = set(self.sel_included.value)
        self._refresh_vars(); self._refresh_bar()
        self._log(f"[seleção] {len(self.seg.included)} variáveis no modelo.")

    def _on_set_cat(self, b):
        cat = None if self.dd_categoria.value == "—" else self.dd_categoria.value
        self.seg.set_category(self.dd_var.value, cat)
        self._refresh_vars()
        self._log(f"[categoria] {self.dd_var.value} = {cat}")

    # ------------------------------------------------------------------ Aba 2 handlers
    def _on_analyze(self, b):
        feat = self.dd_var2.value
        sample = None if self.dd_sample2.value == "(referência)" else self.dd_sample2.value
        tcol = self.tx_time2.value.strip() or None
        try:
            self.out_an_logodds.value = self._fig_html(
                self.seg.plot_variable_logodds(feat, sample=sample))
            self.out_an_dist.value = self._fig_html(
                self.seg.plot_variable_distribution(feat, sample=sample))
            vt = self.seg.variable_table(feat, sample=sample)
            self.out_an_table.value = self._df_html(vt, max_height="280px")
            self.out_an_inv_sample.value = self._fig_html(
                self.seg.plot_variable_inversion_by_sample(feat))
            self.out_an_cards.value = self._var_cards(self.seg.variable_summary(feat, sample))
        except Exception as e:
            self.out_an_logodds.value = f"<i>{e}</i>"
        if tcol:
            for out, fn in ((self.out_an_time,
                             lambda: self.seg.plot_variable_timeseries(feat, tcol, sample)),
                            (self.out_an_inv_safra,
                             lambda: self.seg.plot_variable_inversion_by_safra(feat, tcol, sample)),
                            (self.out_an_psi,
                             lambda: self.seg.plot_variable_psi_by_safra(feat, tcol))):
                try:
                    out.value = self._fig_html(fn())
                except Exception as e:
                    out.value = f"<i>{e}</i>"

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

    # ------------------------------------------------------------------ Aba 3 handlers
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
        return hp

    def _on_fit(self, b):
        algo = self.dd_algo.value
        try:
            self.seg.fit(algo, hyperparams=self._collect_hyperparams(algo))
            self._log(f"[fit] {algo} treinado com {len(self.seg.model_features)} variáveis.")
            self._render_metrics()
            self._render_model_plots()
            self._refresh_bar()
        except Exception as e:
            self._log(f"[fit] erro: {e}")

    def _render_metrics(self):
        m = self.seg.metrics().round(4)
        self.out_metrics.value = self._df_html(m)

    def _render_model_plots(self):
        if self.task_type == "classification":
            specs = [(self.out_model_a, self.seg.plot_roc),
                     (self.out_model_b, self.seg.plot_ks),
                     (self.out_model_c, self.seg.plot_score_distribution)]
        else:
            specs = [(self.out_model_a, self.seg.plot_calibration),
                     (self.out_model_b, self.seg.plot_residuals),
                     (self.out_model_c, self.seg.plot_score_distribution)]
        for out, fn in specs:
            try:
                out.value = self._fig_html(fn())
            except Exception as e:
                out.value = f"<i>{e}</i>"

    def _on_shap(self, b):
        if self.seg.score_ is None:
            self._log("[shap] treine o modelo primeiro.")
            return
        try:
            self.out_shap.value = self._fig_html(self.seg.plot_shap_beeswarm(sample_size=800))
            self.out_shap_bar.value = self._fig_html(self.seg.plot_shap_bar(sample_size=800))
            self._log("[shap] gráficos gerados.")
        except Exception as e:
            self.out_shap.value = f"<i>SHAP indisponível: {e}</i>"
            self._log(f"[shap] erro: {e}")

    # ------------------------------------------------------------------ Aba 4 handlers
    def _on_build_ratings(self, b):
        if self.seg.score_ is None:
            self._log("[ratings] treine o modelo primeiro.")
            return
        try:
            self.seg.build_ratings(method=self.dd_method.value,
                                   n_ratings=int(self.sl_nratings.value),
                                   monotonic_fusion=self.cb_fusion.value)
            self._log(f"[ratings] {len(self.seg.rating_labels_)} faixas "
                      f"({self.dd_method.value}).")
            self.out_rating_table.value = self._df_html(self.seg.rating_table().round(4))
            self.out_rating_badrate.value = self._fig_html(self.seg.plot_rating_badrate())
            self.out_rating_dist.value = self._fig_html(self.seg.plot_rating_distribution())
            self.out_rating_inv_s.value = self._fig_html(self.seg.plot_rating_inversion_by_sample())
            self.out_rating_mono.value = self._df_html(self.seg.monotonicity_report())
            try:
                self.out_rating_inv_t.value = self._fig_html(
                    self.seg.plot_rating_inversion_by_safra())
            except Exception as e:
                self.out_rating_inv_t.value = f"<i>{e}</i>"
            self._refresh_bar()
        except Exception as e:
            self._log(f"[ratings] erro: {e}")

    # ------------------------------------------------------------------ Aba 5 handlers
    def _on_backtest(self, b):
        tcol = self.tx_time3.value.strip() or None
        try:
            self.out_backtest.value = self._df_html(self.seg.backtest(tcol).round(4),
                                                    max_height="320px")
        except Exception as e:
            self.out_backtest.value = f"<i>{e}</i>"
        try:
            self.out_psi.value = self._df_html(self.seg.psi())
        except Exception as e:
            self.out_psi.value = f"<i>{e}</i>"

    def _on_export(self, b):
        try:
            self.result = self.seg.assign()
            self.out_export.value = (f"<div class='mseg-legend'>DataFrame exportado: "
                                     f"{self.result.shape[0]} linhas × {self.result.shape[1]} colunas "
                                     f"(em <code>ui.result</code>).</div>")
            self._log("[export] DataFrame rotulado em ui.result.")
        except Exception as e:
            self._log(f"[export] erro: {e}")

    def _on_save(self, b):
        try:
            self.seg.save(self.tx_save.value)
            self._log(f"[save] salvo em {self.tx_save.value} (+ .model.joblib).")
        except Exception as e:
            self._log(f"[save] erro: {e}")

    def _on_load(self, b):
        try:
            self.seg = self.seg.load(self.tx_save.value, self.df)
            self._refresh_bar(); self._refresh_vars()
            self._log(f"[load] carregado de {self.tx_save.value}.")
        except Exception as e:
            self._log(f"[load] erro: {e}")

    def _on_mlflow(self, b):
        try:
            rid = self.seg.log_to_mlflow(experiment=self.tx_experiment.value or None,
                                         registered_model_name=self.tx_model.value or None,
                                         verbose=False)
            self._log(f"[mlflow] run_id = {rid}")
        except Exception as e:
            self._log(f"[mlflow] erro: {e}")

    # ------------------------------------------------------------------ display
    def _ipython_display_(self):
        display(self.panel)

    def display(self):
        display(self.panel)
