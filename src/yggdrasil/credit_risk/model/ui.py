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

import numpy as np
import pandas as pd

try:
    import ipywidgets as W
    from IPython.display import display
except Exception as e:  # pragma: no cover
    raise ImportError("Este módulo requer ipywidgets e IPython (Jupyter).") from e

from .segmenter import ALGORITHMS, BOOSTING_ALGORITHMS, ModelSegmenter

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
/* caixa de tutorial (algoritmo/parâmetros) e legenda de categorias */
.mseg-help { background:#f5f8fc; border:1px solid #dbe5f1; border-left:3px solid var(--ac);
  border-radius:9px; padding:11px 14px; font-size:11.5px; color:#3a4658; line-height:1.62;
  margin-top:8px; }
.mseg-help .ttl { font-size:12px; font-weight:600; color:var(--ink); margin-bottom:5px; }
.mseg-help ul { margin:5px 0 0; padding-left:6px; list-style:none; }
.mseg-help li { margin:4px 0; }
.mseg-help .pname { font-family:'IBM Plex Mono', ui-monospace, monospace; font-weight:600;
  color:#0b63ce; background:#e7eef8; padding:1px 6px; border-radius:5px; }
.mseg-help .none { color:var(--muted); font-style:italic; }
.mseg-cat { display:inline-block; padding:2px 9px; border-radius:999px; font-size:11px;
  font-weight:600; }
.mseg-cat-keep { background:#e7f5ee; color:#157a52; }
.mseg-cat-rev  { background:#fbf3e0; color:#9a6f12; }
.mseg-cat-drop { background:#fbe7e4; color:#b23a2a; }
.mseg-rule { margin-top:8px; padding:11px 13px; background:#fff; border:1px solid #e3e9f1;
  border-radius:8px; font-size:11px; }
.mseg-rule .rt { font-weight:600; color:var(--ink); margin-bottom:8px;
  font-family:'IBM Plex Mono', ui-monospace, monospace; }
.mseg-rule .rr { display:flex; align-items:baseline; gap:9px; margin:7px 0; line-height:1.65; }
.mseg-rule .rr .mseg-cat { flex:0 0 72px; text-align:center; }
.mseg-rule .rx { flex:1; color:#3a4658; }
.mseg-rule i { color:#6b7480; }
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
.mseg-metric { background:#f7f8fa; border:1px solid #eef0f3; border-radius:9px; padding:7px 10px; }
.mseg-metric .k { font-size:10px; text-transform:uppercase; letter-spacing:.04em; color:#8a93a3; }
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
.mseg-formula { background:#0f1620; color:#e8edf4; border-radius:10px; padding:13px 15px;
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
.mseg-coef th { background:#eef1f5; color:#27324a; font-weight:600; text-align:right;
  padding:6px 10px; border-bottom:2px solid #b9c2d0; white-space:nowrap; }
.mseg-coef td { padding:5px 10px; border-bottom:1px solid #eef0f3; text-align:right;
  white-space:nowrap; }
.mseg-coef th.term, .mseg-coef td.term { text-align:left; font-weight:500; }
.mseg-coef td.read, .mseg-coef th.read { text-align:left; }
.mseg-coef td.num { font-variant-numeric:tabular-nums; font-weight:600; }
.mseg-coef tbody tr:hover td { background:#f5f8fb; }
.mseg-coef tr.base td { background:#fbfcfe; color:var(--muted); font-style:italic; }
.mseg-barcell { width:150px; }
.mseg-barwrap { background:#eef0f3; border-radius:5px; height:9px; width:100%; overflow:hidden; }
.mseg-bar-pos { background:#2ea06b; height:100%; } .mseg-bar-neg { background:#d64545; height:100%; }
.mseg-pos-tx { color:#157a52; } .mseg-neg-tx { color:#b23a2a; }
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
        self.score_df = None   # base externa opcional p/ escorar (ui.score_df = df_novo)
        self._build()
        self._refresh_bar()
        self._refresh_vars()
        self._sync_bin_controls()

    # ------------------------------------------------------------------ render utils
    def _fig_html(self, fig, border=False, tight=True):
        import base64
        import io as _io
        import matplotlib.pyplot as plt
        buf = _io.BytesIO()
        # tight=False → o PNG sai exatamente em figsize×dpi (mesma figsize ⇒ mesmo
        # tamanho de imagem), garantindo gráficos lado a lado com a MESMA altura.
        save_kw = {"format": "png", "dpi": fig.get_dpi()}
        if tight:
            save_kw["bbox_inches"] = "tight"
        fig.savefig(buf, **save_kw)
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        style = "max-width:100%;height:auto"
        if border:
            style += ";border:1px solid #e6e8eb;border-radius:6px"
        return f"<img src='data:image/png;base64,{b64}' style='{style}'/>"

    _CAT_COLORS = {"manter": ("#157a52", "#e7f5ee"), "revisar": ("#9a6f12", "#fbf3e0"),
                   "descartar": ("#b23a2a", "#fbe7e4")}

    # CSS por nível de qualidade de uma métrica (identificador visual da tabela)
    _LEVEL_CSS = {
        "bom":     "background-color:#e7f5ee;color:#157a52;font-weight:600",
        "atencao": "background-color:#fbf3e0;color:#9a6f12;font-weight:600",
        "ruim":    "background-color:#fbe7e4;color:#b23a2a;font-weight:600"}

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
                          desc="Score onde o KS é máximo (ponto de corte). Informativo.",
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

    def _df_html(self, df, max_height=None, color_categoria=False, center=False):
        sty = (df.style.hide(axis="index").set_table_styles(self._TABLE_STYLES)
               .set_properties(**{"font-size": "12px"}))
        if center:
            sty = sty.set_properties(**{"text-align": "center"})
            sty = sty.set_table_styles([{"selector": "th", "props": [("text-align", "center")]}],
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
                                             rows=max(4, len(cands)),  # mostra todas, sem rolagem
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
            "<div class='mseg-help'>"
            "<div class='ttl'>O que é “categoria”?</div>"
            "É um <b>rótulo de triagem</b> que você dá a cada variável para registrar a sua "
            "decisão durante a análise. Ele aparece na coluna <code>categoria</code> do ranking e "
            "é só documentação: <b>não treina nem remove a variável sozinho</b> — quem define o que "
            "entra no modelo é a lista <b>“No modelo”</b> (botão <i>Aplicar seleção</i>)."
            "<ul>"
            "<li><span class='mseg-cat mseg-cat-keep'>✓ manter</span> &nbsp;variável boa, "
            "deve entrar no modelo.</li>"
            "<li><span class='mseg-cat mseg-cat-rev'>● revisar</span> &nbsp;promissora, mas "
            "precisa de uma segunda olhada (IV baixo, inversão, PSI etc.) antes de decidir.</li>"
            "<li><span class='mseg-cat mseg-cat-drop'>✕ descartar</span> &nbsp;não deve ser "
            "usada (instável, sem poder, redundante…).</li>"
            "<li><span class='mseg-cat' style='background:#eef1f5;color:#6b7480'>— sem categoria</span> "
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
            "</div></div>")
        self.sl_min_iv = W.BoundedFloatText(value=0.02, min=0.0, max=1.0, step=0.01,
                                            description="IV mín.:",
                                            style={"description_width": "initial"},
                                            layout=W.Layout(width="150px"))
        self.sl_max_psi = W.BoundedFloatText(value=0.25, min=0.0, max=1.0, step=0.05,
                                             description="PSI máx.:",
                                             style={"description_width": "initial"},
                                             layout=W.Layout(width="155px"))
        self.cb_require_mono = W.Checkbox(value=False, description="exigir monotonia")
        self.btn_auto = W.Button(description="Auto-selecionar", button_style="primary",
                                 icon="magic",
                                 tooltip="Inclui/exclui variáveis no modelo pelos critérios "
                                         "(IV mín., PSI máx., monotonia) e marca manter/descartar.")
        self.btn_auto_cat = W.Button(description="Auto-categorizar", icon="tags",
                                     tooltip="Classifica TODAS as variáveis em manter / revisar / "
                                             "descartar pela regra (IV · PSI · monotonia). "
                                             "Só rotula — não altera a seleção do modelo.")
        self.btn_apply_sel = W.Button(description="Aplicar seleção", button_style="success",
                                      icon="check")
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

        self.btn_auto.on_click(self._on_auto_select)
        self.btn_auto_cat.on_click(self._on_auto_categorize)
        self.btn_apply_sel.on_click(self._on_apply_sel)
        self.btn_set_cat.on_click(self._on_set_cat)
        self.btn_incl_all.on_click(lambda b: (self.seg.include_all(), self._sync_sel(),
                                              self._refresh_vars(), self._refresh_bar()))
        self.btn_clear.on_click(lambda b: (self.seg.clear_features(), self._sync_sel(),
                                           self._refresh_vars(), self._refresh_bar()))
        self.btn_refresh_vars.on_click(lambda b: self._refresh_vars())
        self.btn_clear_derived.on_click(self._on_clear_derived)
        self.dd_var.observe(lambda c: self._refresh_var_preview(), names="value")
        # clicar numa variável na lista "No modelo" também atualiza a prévia/análise
        self.sel_included.observe(self._on_sel_click, names="value")

        tab_vars = W.VBox([
            W.HTML("<div class='mseg-h'>Seleção & categorização de variáveis</div>"),
            W.HBox([self.sl_min_iv, self.sl_max_psi, self.cb_require_mono,
                    self.btn_auto, self.btn_auto_cat]),
            W.VBox([self.sel_included,
                    W.HBox([self.btn_apply_sel, self.btn_incl_all, self.btn_clear,
                            self.btn_refresh_vars, self.btn_clear_derived],
                           layout=W.Layout(justify_content="space-between", width="99%"))]),
            W.HBox([self.dd_var, self.dd_categoria, self.btn_set_cat]),
            self.out_cat_hint,
            W.HBox([W.VBox([W.HTML("<div class='mseg-h'>Ranking (IV / força / inversão / PSI)</div>"),
                            self.out_vars], layout=W.Layout(width="58%")),
                    W.VBox([self.out_var_preview_h,
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
                                       tooltip="Materializa a binagem/agrupamento atual (faixas "
                                               "numéricas ou grupos de categorias) como uma NOVA "
                                               "variável categórica, candidata ao modelo e recriada "
                                               "ao escorar. Junte categorias na mão como na árvore.")
        self.out_bin_hint = W.HTML()
        self.btn_apply_bins.on_click(self._on_apply_bins)
        self.btn_clear_bins.on_click(self._on_clear_bins)
        self.btn_create_cat.on_click(self._on_create_cat)
        self.tg_binmode.observe(lambda c: self._sync_binmode(), names="value")
        self.dd_var2.observe(lambda c: self._sync_bin_controls(), names="value")
        self.out_an_cards = W.HTML()
        self.out_an_distbad = W.HTML()    # distribuição + % de maus (gráfico único)
        self.out_an_table = W.HTML()
        self.out_an_inv_sample = W.HTML()
        self.out_an_inv_safra = W.HTML()
        self.out_an_time = W.HTML()
        self.out_an_psi = W.HTML()
        self.btn_analyze.on_click(self._on_analyze)

        bin_card = W.VBox([
            W.HTML("<div class='mseg-h'>Categorizar a variável na mão (bins manuais)</div>"),
            W.HBox([self.tg_binmode, self.tx_cuts, self.btn_apply_bins, self.btn_clear_bins]),
            self.out_bin_hint,
            W.HBox([self.tx_new_cat, self.btn_create_cat]),
        ])
        bin_card.add_class("mseg-card")
        bin_card.layout = W.Layout(margin="26px 0 0 0")   # respiro até a linha da variável

        tab_an = W.VBox([
            W.HBox([self.dd_var2, self.dd_sample2, self.tx_time2, self.btn_analyze]),
            bin_card,
            self.out_an_cards,
            W.VBox([W.HTML("<div class='mseg-h'>Distribuição & % de maus por faixa</div>"),
                    self.out_an_distbad]),
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
        # caixas que aparecem/somem conforme o algoritmo escolhido
        self.box_logit = W.HBox([self.tx_C])
        self.box_ensemble = W.VBox([self.sl_n_est,
                                    W.HBox([self.cb_max_depth, self.sl_max_depth])])
        self.box_lr = W.HBox([self.tx_lr])
        self.btn_fit = W.Button(description="Treinar modelo", button_style="primary",
                                icon="cogs")
        self.btn_formula = W.Button(description="Ver fórmula", icon="superscript",
                                    tooltip="Mostra a equação ajustada (intercepto + coeficientes) "
                                            "da Regressão Logística/Linear")
        self.btn_shap = W.Button(description="Calcular SHAP", icon="bar-chart")
        self.cb_woe = W.Checkbox(value=False, indent=False,
                                 description="Transformar variáveis (WoE por bins)")
        self.cb_woe.tooltip = (
            "Em vez dos valores crus, alimenta o modelo com a transformação por bins "
            "(estilo scorecard): contínuas viram faixas e categóricas viram grupos "
            "— exatamente os bins definidos na aba 'Análise de variáveis' — e cada bin "
            "é codificado pelo seu WoE (classificação) ou risco médio (regressão).")
        self.out_woe_help = W.HTML()
        self.cb_woe.observe(lambda c: self._sync_woe_hint(), names="value")
        self.out_algo_help = W.HTML()   # tutorial do algoritmo/parâmetros selecionado
        self.out_metrics = W.HTML()
        self.out_formula = W.HTML()
        self.out_model_a = W.HTML()
        self.out_model_b = W.HTML()
        self.out_model_c = W.HTML()
        self.out_shap = W.HTML()
        self.out_shap_bar = W.HTML()
        self.btn_fit.on_click(self._on_fit)
        self.btn_formula.on_click(self._on_formula)
        self.btn_shap.on_click(self._on_shap)
        self.dd_algo.observe(lambda c: self._sync_algo_visibility(), names="value")
        self.cb_max_depth.observe(lambda c: self._sync_algo_visibility(), names="value")

        train_card = W.VBox([
            W.HTML("<div class='mseg-h'>Treinar (ou usar modelo pré-ajustado via set_model)</div>"),
            W.HBox([self.dd_algo, self.cb_woe]),
            self.box_logit, self.box_ensemble, self.box_lr,
            self.out_algo_help,
            self.out_woe_help,
            W.HBox([self.btn_fit, self.btn_formula, self.btn_shap]),
        ])
        train_card.add_class("mseg-card")
        self.formula_card = W.VBox([
            W.HTML("<div class='mseg-h'>Fórmula do modelo (logística/linear)</div>"),
            self.out_formula])
        self.formula_card.add_class("mseg-card")

        tab_model = W.VBox([
            train_card,
            W.VBox([W.HTML("<div class='mseg-h'>Métricas por amostra</div>"), self.out_metrics]),
            self.formula_card,
            W.HBox([W.VBox([self.out_model_a], layout=W.Layout(width="33.33%")),
                    W.VBox([self.out_model_b], layout=W.Layout(width="33.33%")),
                    W.VBox([self.out_model_c], layout=W.Layout(width="33.33%"))]),
            W.HBox([W.VBox([W.HTML("<div class='mseg-h'>SHAP — beeswarm</div>"), self.out_shap],
                           layout=W.Layout(width="55%")),
                    W.VBox([W.HTML("<div class='mseg-h'>SHAP — importância</div>"), self.out_shap_bar],
                           layout=W.Layout(width="45%"))]),
        ], layout=W.Layout(padding="2px"))
        self._sync_algo_visibility()

        # ---------- Aba 4: Ratings & Score ----------
        self.dd_method = W.Dropdown(options=[("Decis", "decis"), ("Quantil + fusão", "quantil"),
                                             ("Árvore + fusão", "arvore"), ("OptBinning", "optbin")],
                                    value="quantil", description="Metodologia:",
                                    style={"description_width": "initial"})
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
        self.out_rating_mono = W.HTML()
        self.btn_build_ratings.on_click(self._on_build_ratings)

        tab_rating = W.VBox([
            W.HBox([self.dd_method, self.sl_nratings, self.cb_fusion,
                    self.btn_suggest_n, self.btn_build_ratings]),
            self.out_rating_auto,
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
        # escorar base: rating + valor previsto do alvo por rating (a "régua")
        self.tx_value_col = W.Text(value="valor_previsto", description="Coluna do valor:",
                                   style={"description_width": "initial"})
        self.tx_in_table = W.Text(value="", description="Tabela (Databricks):",
                                  placeholder="catalog.schema.tabela",
                                  style={"description_width": "initial"},
                                  layout=W.Layout(width="46%"))
        self.tx_out_table = W.Text(value="", description="Gravar em (opcional):",
                                   placeholder="catalog.schema.saida",
                                   style={"description_width": "initial"},
                                   layout=W.Layout(width="46%"))
        self.cb_recreate = W.Checkbox(value=True, indent=False,
                                      description="recriar categorias/faixas das variáveis")
        self.btn_ruler = W.Button(description="Ver régua (rating → valor)", icon="list-ol")
        self.btn_score = W.Button(description="Escorar base", button_style="primary",
                                  icon="bolt")
        self.out_ruler = W.HTML()
        self.out_score = W.HTML()
        self.btn_backtest.on_click(self._on_backtest)
        self.btn_export.on_click(self._on_export)
        self.btn_ruler.on_click(self._on_ruler)
        self.btn_score.on_click(self._on_score)
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
            W.HTML("<div class='mseg-h'>Escorar base — rating + valor previsto do alvo "
                   "por rating (régua)</div>"),
            W.HTML("<div class='mseg-legend'>Devolve <code>score</code>, <code>rating</code> e o "
                   "valor previsto do alvo daquele rating (ex.: LGD previsto). A base só precisa "
                   "ter as <b>variáveis originais do modelo</b> — a binagem/WoE é refeita ao "
                   "escorar e, quando uma variável foi categorizada, as <b>faixas/grupos são "
                   "recriados</b> na saída.<br>• <b>Tabela Databricks</b>: informe "
                   "<code>catalog.schema.tabela</code> (lê via Spark) e, opcionalmente, uma tabela "
                   "de saída para gravar; o Spark DataFrame fica em <code>ui.result</code>.<br>"
                   "• <b>Em memória</b>: deixe a tabela em branco para escorar a base carregada "
                   "(ou <code>ui.score_df = df_novo</code>). Resultado (pandas) em "
                   "<code>ui.result</code>.</div>"),
            W.HBox([self.tx_in_table, self.tx_out_table]),
            W.HBox([self.tx_value_col, self.cb_recreate]),
            W.HBox([self.btn_ruler, self.btn_score]),
            self.out_ruler,
            self.out_score,
            W.HTML("<div class='mseg-h'>Salvar / carregar (JSON + modelo joblib)</div>"),
            W.HBox([self.tx_save, self.btn_save, self.btn_load]),
            W.HTML("<div class='mseg-h'>Registrar no MLflow</div>"),
            W.HBox([self.tx_experiment, self.tx_model, self.btn_mlflow]),
        ], layout=W.Layout(padding="2px"))

        self.tabs = W.Tab(children=[tab_vars, tab_an, tab_model, tab_rating, tab_export])
        for i, t in enumerate(["Variáveis", "Análise de variáveis", "Modelo",
                               "Ratings & Score", "Validar & Exportar"]):
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
            rk = self.seg.variable_iv().drop(columns="n_inversoes", errors="ignore")
            for c in rk.columns:
                if c.startswith("psi_") or c in ("iv", "pior_psi"):
                    rk[c] = rk[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
            rk["incluida"] = rk["incluida"].map(lambda b: "✓" if b else "")
            rk["categoria"] = rk["categoria"].fillna("—")
            if "bins_manuais" in rk.columns:
                rk["bins_manuais"] = rk["bins_manuais"].map(lambda b: "✎" if b else "")
                rk = rk.rename(columns={"bins_manuais": "manual"})
            self.out_vars.value = self._df_html(rk, max_height="320px", color_categoria=True)
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
                self.out_var_preview_h.value = f"<div class='mseg-h'>Estabilidade no tempo · {titulo}</div>"
                fig = self.seg.plot_variable_risk_by_safra(feat)
            else:
                self.out_var_preview_h.value = (
                    "<div class='mseg-h'>Logodds da variável por faixa "
                    "<span style='font-weight:400;text-transform:none'>(defina uma coluna de "
                    "safra para ver a estabilidade no tempo)</span></div>")
                fig = self.seg.plot_variable_logodds(feat)
            self.out_var_preview.value = self._fig_html(fig, tight=False)
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

    def _on_auto_categorize(self, b):
        try:
            rk = self.seg.auto_categorize(min_iv=self.sl_min_iv.value,
                                          max_psi=self.sl_max_psi.value,
                                          require_monotonic=self.cb_require_mono.value)
            self._refresh_vars(); self._refresh_bar()
            vc = rk["categoria"].value_counts().to_dict()
            resumo = " · ".join(f"{k}: {vc.get(k, 0)}"
                                for k in ("manter", "revisar", "descartar"))
            self._log(f"[auto-categoria] {resumo} (veja a coluna 'motivo' no ranking)")
        except Exception as e:
            self._log(f"[auto-categoria] erro: {e}")

    def _on_apply_sel(self, b):
        self.seg.included = set(self.sel_included.value)
        self._refresh_vars(); self._refresh_bar()
        self._log(f"[seleção] {len(self.seg.included)} variáveis no modelo.")

    def _on_sel_click(self, change):
        """Ao clicar numa variável na lista 'No modelo', aponta a prévia (gráfico de
        estabilidade no tempo) para ela — selecionando-a no dropdown 'Variável'."""
        new = set(change.get("new") or ()); old = set(change.get("old") or ())
        clicada = list(new - old) or list(old - new)
        if clicada and clicada[-1] in self.dd_var.options:
            self.dd_var.value = clicada[-1]      # dispara _refresh_var_preview

    def _on_clear_derived(self, b):
        removidas = self.seg.clear_derived()
        self._refresh_candidates(); self._refresh_vars(); self._refresh_bar()
        if removidas:
            self._log(f"[reset] {len(removidas)} variável(is) criada(s) removida(s): "
                      f"{', '.join(removidas)}.")
        else:
            self._log("[reset] nenhuma variável criada para remover.")

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
            self.out_an_distbad.value = self._fig_html(
                self.seg.plot_variable_distribution_badrate(feat, sample=sample), tight=False)
            vt = self.seg.variable_table(feat, sample=sample)
            self.out_an_table.value = self._df_html(vt, max_height="280px", center=True)
            self.out_an_inv_sample.value = self._fig_html(
                self.seg.plot_variable_inversion_by_sample(feat))
            self.out_an_cards.value = self._var_cards(self.seg.variable_summary(feat, sample))
        except Exception as e:
            self.out_an_distbad.value = f"<i>{e}</i>"
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

    # ---- bins manuais (categorizar "na mão") ----
    def _sync_binmode(self):
        """Mostra o campo de cortes/grupos só no modo Manual."""
        manual = self.tg_binmode.value == "Manual"
        self.tx_cuts.layout.display = "" if manual else "none"
        self.btn_apply_bins.layout.display = "" if manual else "none"
        kind = self.seg._detect_kind(self.dd_var2.value)
        self.tx_cuts.placeholder = ("cortes, ex.: 0.7, 0.9" if kind == "num"
                                    else "grupos, ex.: A,B; C,D")

    def _sync_bin_controls(self):
        """Sincroniza o modo/campo com os bins manuais da variável selecionada."""
        feat = self.dd_var2.value
        spec = self.seg.manual_bins_spec(feat)
        self.tx_cuts.value = spec
        self.tg_binmode.value = "Manual" if spec else "Ótimo"
        self._sync_binmode()
        self._render_bin_hint(feat)

    def _render_bin_hint(self, feat):
        if self.seg.manual_bins(feat):
            self.out_bin_hint.value = (
                "<div class='mseg-legend'>✎ Bins <b>manuais</b> ativos nesta variável — "
                "aplicados à tabela, IV, logodds/WoE, PSI e inversão.</div>")
        else:
            self.out_bin_hint.value = (
                "<div class='mseg-legend'>Binning <b>ótimo</b> (optbinning). "
                "Numérica: cortes separados por vírgula. Categórica: grupos por "
                "<code>;</code> e categorias por <code>,</code>.</div>")

    def _on_apply_bins(self, b):
        feat = self.dd_var2.value
        try:
            self.seg.set_manual_bins(feat, self.tx_cuts.value)
            if not self.seg.manual_bins(feat):
                self._log(f"[bins] '{feat}': nada para aplicar (campo vazio).")
            else:
                self._log(f"[bins] '{feat}': bins manuais aplicados.")
            self._render_bin_hint(feat)
            self._on_analyze(None)
            self._refresh_vars()
        except Exception as e:
            self._log(f"[bins] erro: {e}")

    def _on_clear_bins(self, b):
        feat = self.dd_var2.value
        self.seg.clear_manual_bins(feat)
        self.tx_cuts.value = ""
        self.tg_binmode.value = "Ótimo"
        self._render_bin_hint(feat)
        self._log(f"[bins] '{feat}': voltou ao binning ótimo.")
        self._on_analyze(None)
        self._refresh_vars()

    def _refresh_candidates(self):
        """Atualiza as listas de variáveis (após criar uma variável derivada)."""
        cands = list(self.seg.candidates)
        for dd in (self.dd_var, self.dd_var2):
            cur = dd.value
            dd.options = cands
            if cur in cands:
                dd.value = cur
        self.sel_included.options = cands
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
            self._refresh_bar()
            self._log(f"[nova variável] '{new}' criada de '{feat}' ({ncat} categorias) — "
                      f"já disponível na seleção e no modelo.")
        except Exception as e:
            self._log(f"[nova variável] erro: {e}")

    # ------------------------------------------------------------------ Aba 3 handlers
    def _sync_algo_visibility(self):
        """Mostra só os hiperparâmetros do algoritmo escolhido: C (logística),
        n_estimators/max_depth (random forest · gradient boosting), nada (linear)."""
        algo = self.dd_algo.value
        ensemble = algo not in ("logistica", "linear")
        self.box_logit.layout.display = "" if algo == "logistica" else "none"
        self.box_ensemble.layout.display = "" if ensemble else "none"
        self.box_lr.layout.display = "" if algo in BOOSTING_ALGORITHMS else "none"
        self.sl_max_depth.layout.display = "" if self.cb_max_depth.value else "none"
        # a fórmula só faz sentido para modelos lineares/logísticos
        linear = algo in ("logistica", "linear")
        self.btn_formula.layout.display = "" if linear else "none"
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
        items = "".join(f"<li>{P[p]}</li>" for p in params)
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

        # z = intercepto + Σ coefᵢ·termoᵢ — cada termo é uma "pílula" que quebra linha,
        # com o sinal do coeficiente colorido (verde sobe z, vermelho desce z)
        chips = [f"<span class='mseg-term mseg-b0'>{b0:+.4f}</span>"]
        for _, r in coef.iterrows():
            c = float(r["coef"]); cls = "pos" if c >= 0 else "neg"
            chips.append(
                f"<span class='mseg-term mseg-{cls}'>"
                f"<span class='mseg-op'>{'+' if c >= 0 else '−'}</span>"
                f"<span class='mseg-cf'>{abs(c):.4f}</span>"
                f"<span class='mseg-mul'>×</span>"
                f"<span class='mseg-vn'>{r['termo']}</span></span>")
        z_html = "<span class='mseg-zlead'>z =</span>" + "".join(chips)

        # tabela de coeficientes com barra de magnitude (|coef|) e leitura do efeito
        cmax = float(coef["coef"].abs().max()) or 1.0
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
            rows.append(
                f"<tr><td class='term'>{r['termo']}</td>"
                f"<td class='num mseg-{cls}-tx'>{c:+.4f}</td>"
                f"<td class='barcell'>{bar}</td>{extra}</tr>")
        # linha do intercepto (baseline) — sem barra/leitura por unidade
        base_extra = (f"<td class='num'>{np.exp(b0):.4f}</td>"
                      "<td class='read'>odds base (todos os termos = 0)</td>"
                      if is_clf else "<td class='read'>valor base de ŷ</td>")
        base_row = (f"<tr class='base'><td class='term'>intercepto (β₀)</td>"
                    f"<td class='num'>{b0:+.4f}</td><td></td>{base_extra}</tr>")
        head_extra = "<th>odds_ratio</th><th class='read'>leitura</th>" if is_clf \
            else "<th class='read'>leitura</th>"
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
            + "</div>")
        self.out_formula.value = (
            f"<div class='mseg-eq'>{eq}</div>"
            f"<div class='mseg-formula'>{z_html}</div>"
            f"{legend}"
            f"<div style='max-height:320px;overflow:auto'>{table}</div>")

    def _on_formula(self, b):
        if self.seg.model is None:
            self._log("[fórmula] treine o modelo primeiro.")
            return
        self._render_formula()

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
        algo = self.dd_algo.value
        transform = "woe" if self.cb_woe.value else "raw"
        try:
            self.seg.fit(algo, hyperparams=self._collect_hyperparams(algo), transform=transform)
            modo = "WoE/bins" if transform == "woe" else "valores crus"
            self._log(f"[fit] {algo} treinado com {len(self.seg.model_features)} "
                      f"variáveis ({modo}).")
            self._render_metrics()
            self._render_model_plots()
            self._render_formula()
            self._refresh_bar()
        except Exception as e:
            self._log(f"[fit] erro: {e}")

    def _render_metrics(self):
        m = self.seg.metrics().round(4)
        self.out_metrics.value = (self._metrics_table_html(m)
                                  + self._metrics_guide_html(list(m.columns)))

    def _metrics_table_html(self, m):
        """Tabela de métricas centralizada e com identificador visual: cada célula
        ganha cor (verde/amarelo/vermelho) conforme o guia de bolso da métrica."""
        metric_cols = [c for c in m.columns if c in self._METRIC_GUIDE]
        fmt = {c: "{:.4f}" for c in m.columns if c not in ("amostra", "n")}
        sty = (m.style.hide(axis="index").set_table_styles(self._TABLE_STYLES)
               .set_properties(**{"font-size": "12px", "text-align": "center"})
               .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}],
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
                        f"<span class='mono' style='color:#8a93a3'>{c}</span></td>"
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
        if self.task_type == "classification":
            specs = [(self.out_model_a, self.seg.plot_roc),
                     (self.out_model_b, self.seg.plot_ks),
                     (self.out_model_c, self.seg.plot_score_distribution)]
        else:
            specs = [(self.out_model_a, self.seg.plot_calibration),
                     (self.out_model_b, self.seg.plot_residuals),
                     (self.out_model_c, self.seg.plot_score_distribution)]
        # figsize comum + tight=False ⇒ os 3 gráficos saem do mesmo tamanho
        common = (5.2, 4.3)
        for out, fn in specs:
            try:
                out.value = self._fig_html(fn(figsize=common), tight=False)
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
            self.out_rating_table.value = self._df_html(
                self.seg.rating_table().round(4), center=True)
            self.out_rating_badrate.value = self._fig_html(self.seg.plot_rating_badrate())
            self.out_rating_dist.value = self._fig_html(self.seg.plot_rating_distribution())
            # mesma figsize + tight=False ⇒ os dois gráficos de inversão (amostras ×
            # safras) saem com a MESMA altura nas colunas 50/50
            inv_size = (8.4, 4.0)
            self.out_rating_inv_s.value = self._fig_html(
                self.seg.plot_rating_inversion_by_sample(figsize=inv_size), tight=False)
            self.out_rating_mono.value = self._df_html(
                self.seg.monotonicity_report(), center=True)
            try:
                self.out_rating_inv_t.value = self._fig_html(
                    self.seg.plot_rating_inversion_by_safra(figsize=inv_size), tight=False)
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
        try:
            if in_tbl:                                   # tabela do Databricks (Spark)
                sout = self.seg.score_table(in_tbl, col_value=col_value,
                                            recreate_categories=recreate,
                                            output_table=out_tbl)
                self.result = sout                       # Spark DataFrame
                prev = sout.limit(10).toPandas()
                ncols = len(sout.columns)
                gravou = (f" Gravado em <code>{out_tbl}</code>." if out_tbl
                          else " Spark DataFrame em <code>ui.result</code>.")
                self.out_score.value = (
                    f"<div class='mseg-legend'>Tabela Databricks <code>{in_tbl}</code> "
                    f"escorada ({ncols} colunas).{gravou} Prévia (10 linhas):</div>"
                    + self._df_html(prev.round(6), max_height="320px"))
                self._log(f"[escorar] tabela '{in_tbl}' escorada"
                          + (f" e gravada em '{out_tbl}'." if out_tbl else " (ui.result)."))
                return
            # em memória (pandas): base carregada ou ui.score_df
            base = self.score_df if self.score_df is not None else self.seg.df
            origem = "ui.score_df" if self.score_df is not None else "base carregada"
            out = self.seg.score_table(base, col_value=col_value, recreate_categories=recreate)
            self.result = out
            novas = [c for c in out.columns if c not in base.columns]
            cols = ", ".join(f"<code>{c}</code>" for c in novas)
            self.out_score.value = (
                f"<div class='mseg-legend'>Base escorada ({origem}): "
                f"{out.shape[0]} linhas × {out.shape[1]} colunas, em <code>ui.result</code>. "
                f"Colunas adicionadas: {cols}.</div>"
                + self._df_html(out.head(10).round(6), max_height="320px"))
            self._log(f"[escorar] {origem} escorada em ui.result ({out.shape[0]} linhas).")
        except Exception as e:
            self.out_score.value = f"<i>{e}</i>"
            self._log(f"[escorar] erro: {e}")

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
