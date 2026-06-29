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
    from IPython.display import clear_output, display
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
/* ===== TEMA ESCURO (classe .dark no painel raiz) ===== */
.mseg.dark { --ink:#e7ebf2; --muted:#9aa6ba; --line:#2c3a55; --ac-soft:#243049;
  --ac-border:#3a4a6a; --ac-deep:#d6deec; --ac:#6076a0; background:#0e1521;
  padding:8px; border-radius:12px; }
.mseg.dark .mseg-banner, .mseg.dark .mseg-card, .mseg.dark .mseg-bar { background:#16202f !important;
  border-color:#27344c !important; }
.mseg.dark .mseg-banner .t { color:#e7ebf2; }
.mseg.dark .mseg-help { background:#1a2740 !important; border-color:#2c3a55 !important; }
.mseg.dark .mseg-tabs .p-TabBar-tab, .mseg.dark .mseg-tabs .lm-TabBar-tab {
  background:#16202f !important; color:#9aa6ba !important; border-color:#27344c !important; }
.mseg.dark .mseg-tabs .p-TabBar-tab.p-mod-current,
.mseg.dark .mseg-tabs .lm-TabBar-tab.lm-mod-current { background:#243049 !important; color:#e7ebf2 !important; }
.mseg.dark .widget-text input, .mseg.dark .widget-dropdown select, .mseg.dark textarea {
  background:#0e1521 !important; color:#e7ebf2 !important; border-color:#2c3a55 !important; }
.mseg.dark .widget-label, .mseg.dark .jupyter-widgets label { color:#c4cdde !important; }
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
        # --- estado de desempenho da UI ---------------------------------------
        # buffer limitado do console: W.Output acumula TODOS os prints (o estado
        # serializado cresce sem limite e trafega pelo comm a cada ação). Mantemos
        # só as últimas N linhas reescrevendo a área com clear_output.
        self._log_lines: list = []
        # cache do <img> base64 da prévia da variável por (feature, versão de bins):
        # trocar/clicar na lista de variáveis regenerava a figura a cada vez.
        self._preview_cache: dict = {}
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
        # dpi limitado a 110 nas PRÉVIAS inline (este método só gera <img> para
        # exibição; export usa save_path nos plot_*): corta o PNG/base64 ~40% sem
        # perda visual perceptível, aliviando o tráfego kernel↔browser.
        save_kw = {"format": "png", "dpi": min(int(fig.get_dpi()), 110)}
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

    # cores da força do IV (forte→verde, médio→azul, fraco→amarelo, inútil→vermelho,
    # suspeito→roxo = IV alto demais, possível vazamento)
    _FORCA_COLORS = {"forte": ("#157a52", "#e7f5ee"), "médio": ("#1f5fa8", "#e7eef8"),
                     "fraco": ("#9a6f12", "#fbf3e0"), "inútil": ("#b23a2a", "#fbe7e4"),
                     "suspeito": ("#6b46c1", "#efe9fb")}

    # cores da estabilidade (PSI): estável→verde · atenção→amarelo · instável→vermelho
    _ESTABILIDADE_COLORS = {"estável": ("#157a52", "#e7f5ee"),
                            "atenção": ("#9a6f12", "#fbf3e0"),
                            "instável": ("#b23a2a", "#fbe7e4")}

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
                    return "color:#b3261e;background-color:#fdecea;font-weight:600"
                if "crescente" in s:
                    return "color:#137a3e;background-color:#e9f6ee;font-weight:600"
                return "color:#6b7480;background-color:#f1f3f5;font-weight:600"
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
                    return ("color:#137a3e;background-color:#e9f6ee;font-weight:600"
                            if str(v).strip().lower() == "ok"
                            else "color:#b3261e;background-color:#fdecea;font-weight:600")
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
                    c = "#137a3e" if a < 0.10 else "#9a6b00" if a < 0.25 else "#b3261e"
                    return f"color:{c};font-weight:600"
                sty = sty.map(_psi_css, subset=["psi"])
            if "gap" in df.columns and pd.api.types.is_numeric_dtype(df["gap"]):
                def _gap_css(v):                       # |gap| <=0.05 verde · <=0.10 âmbar · vermelho
                    if pd.isna(v):
                        return ""
                    a = abs(float(v))
                    c = "#137a3e" if a <= 0.05 else "#9a6b00" if a <= 0.10 else "#b3261e"
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

    @staticmethod
    def _pill(text, cls="muted"):
        return f"<span class='pill pill-{cls}'>{text}</span>"

    def _opts(self, names):
        """(alias, nome_cru) para dropdowns/listas — exibe o feature_label e mantém
        o valor cru (o .value continua sendo o nome real da coluna, então toda a
        lógica de incluir/excluir/analisar segue inalterada)."""
        return [(self.seg.label(n), n) for n in names]

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
            "<div style='font-size:11.5px;color:#6b7480;margin:2px 0 0 2px;line-height:1.5'>"
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

        self.btn_auto.on_click(self._on_auto_select)
        self.btn_auto_cat.on_click(self._on_auto_categorize)
        self.btn_include.on_click(self._on_include_var)
        self.btn_exclude.on_click(self._on_exclude_var)
        self.btn_set_cat.on_click(self._on_set_cat)
        self.btn_incl_all.on_click(lambda b: (self.seg.include_all(), self._sync_sel(),
                                              self._refresh_vars(), self._refresh_bar()))
        self.btn_clear.on_click(lambda b: (self.seg.clear_features(), self._sync_sel(),
                                           self._refresh_vars(), self._refresh_bar()))
        self.btn_refresh_vars.on_click(lambda b: self._refresh_vars())
        self.btn_clear_derived.on_click(self._on_clear_derived)
        # botões largos (~19%) → só uma pequena folga entre eles na linha
        for _b in (self.btn_incl_all, self.btn_clear,
                   self.btn_refresh_vars, self.btn_clear_derived):
            _b.layout = W.Layout(width="24%")
        self.dd_var.observe(lambda c: self._refresh_var_preview(), names="value")
        # clicar numa variável na lista "No modelo" também atualiza a prévia/análise
        self.sel_included.observe(self._on_sel_click, names="value")

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
                                       layout=W.Layout(width="auto", min_width="232px"),
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
        tab_an = W.VBox([
            W.HBox([self.dd_var2, self.dd_sample2, self.tx_time2, self.btn_analyze]),
            bin_card,
            self.out_an_cards,
            row1, row2, row3,
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
        self.btn_shap = W.Button(description="Calcular SHAP", icon="bar-chart")
        # --- tuning bayesiano (Optuna) ---
        self.sl_trials = W.IntSlider(description="trials", min=5, max=100, value=30,
                                     layout=W.Layout(width="60%"))
        self.btn_tune = W.Button(description="Tunar com Optuna", button_style="warning",
                                 icon="magic")
        self.btn_tune.tooltip = ("Otimização bayesiana (Optuna): busca hiperparâmetros "
                                 "do algoritmo selecionado maximizando AUC (classificação) / "
                                 "R² (regressão) no OOT, e treina com os melhores. Requer "
                                 "pip install yggdrasil[optuna].")
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
        self.btn_shap.on_click(self._on_shap)
        self.dd_algo.observe(lambda c: self._sync_algo_visibility(), names="value")
        self.cb_max_depth.observe(lambda c: self._sync_algo_visibility(), names="value")

        train_card = W.VBox([
            W.HTML("<div class='mseg-h'>Treinar (ou usar modelo pré-ajustado via set_model)</div>"),
            W.HBox([self.dd_algo, self.cb_woe]),
            self.box_logit, self.box_ensemble, self.box_lr,
            self.out_algo_help,
            self.out_woe_help,
            W.HBox([self.btn_fit, self.btn_shap]),
            self.pb_fit,
            self.out_fit_status,
            W.HTML("<div class='mseg-legend'>Tuning bayesiano (Optuna): busca os melhores "
                   "hiperparâmetros do algoritmo selecionado e treina com eles.</div>"),
            W.HBox([self.sl_trials, self.btn_tune]),
            self.pb_tune,
            self.out_tune,
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
                                             ("Árvore + fusão", "arvore"), ("OptBinning", "optbin"),
                                             ("Manual · cortes de score", "manual_score"),
                                             ("Manual · percentis", "manual_percentil")],
                                    value="quantil", description="Metodologia:",
                                    style={"description_width": "initial"})
        # entrada dos métodos manuais (cortes de score OU percentis), só quando aplicável
        self.tx_manual = W.Text(value="", placeholder="cortes de score (ex.: 0.2, 0.5, 0.8)",
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
        self.tx_pdf = W.Text(value="relatorio_modelo.pdf", description="PDF:",
                             style={"description_width": "initial"})
        self.btn_pdf = W.Button(description="Gerar relatório PDF", button_style="primary",
                                icon="file-pdf-o")
        self.out_pdf = W.HTML()
        self.btn_pdf.on_click(self._on_pdf)
        self.tx_md = W.Text(value="relatorio_modelo.md", description="Markdown:",
                            style={"description_width": "initial"})
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

        self.tx_in_table.layout = W.Layout(width="48%")
        self.tx_out_table.layout = W.Layout(width="48%")
        card_valid = W.VBox([
            W.HTML("<div class='mseg-h'>Validação · backtest e estabilidade dos ratings</div>"),
            W.HBox([self.tx_time3, self.btn_backtest]),
            W.HBox([W.VBox([W.HTML("<div class='mseg-h'>Backtest por safra</div>"),
                            self.out_backtest], layout=W.Layout(width="60%")),
                    W.VBox([W.HTML("<div class='mseg-h'>PSI dos ratings</div>"),
                            self.out_psi], layout=W.Layout(width="38%"))],
                   layout=W.Layout(justify_content="space-between")),
        ]); card_valid.add_class("mseg-card")
        card_score = W.VBox([
            W.HTML("<div class='mseg-h'>Escoragem da base · score + rating + valor previsto "
                   "(régua)</div>"),
            W.HTML("<div class='mseg-legend'>Devolve <code>score</code>, <code>rating</code> e o "
                   "valor previsto do alvo daquele rating (ex.: LGD/PD previsto). A base só precisa "
                   "das <b>variáveis originais do modelo</b> — binagem/WoE e faixas são recriadas ao "
                   "escorar. <b>Tabela Databricks</b>: <code>catalog.schema.tabela</code> (via Spark; "
                   "saída opcional) → <code>ui.result</code>. <b>Em memória</b>: deixe em branco (ou "
                   "<code>ui.score_df = df_novo</code>) → <code>ui.result</code>.</div>"),
            W.HBox([self.tx_in_table, self.tx_out_table],
                   layout=W.Layout(justify_content="space-between")),
            W.HBox([self.tx_value_col, self.cb_recreate]),
            W.HBox([self.btn_ruler, self.btn_score]),
            self.out_ruler, self.out_score,
            W.HTML("<div class='mseg-h' style='margin-top:8px'>Exportar DataFrame rotulado</div>"),
            W.HBox([self.btn_export]), self.out_export,
        ]); card_score.add_class("mseg-card")
        card_persist = W.VBox([
            W.HTML("<div class='mseg-h'>Persistência · JSON + modelo joblib · relatório "
                   "PDF/Markdown</div>"),
            W.HBox([self.tx_save, self.btn_save, self.btn_load]),
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
            card_valid, card_score,
            W.HBox([card_persist, card_mlflow],
                   layout=W.Layout(justify_content="space-between", align_items="stretch")),
        ], layout=W.Layout(padding="2px"))

        self.tabs = W.Tab(children=[tab_vars, tab_an, tab_model, tab_rating, tab_export])
        for i, t in enumerate(["Variáveis", "Análise de variáveis", "Modelo",
                               "Ratings & Score", "Validar & Exportar"]):
            self.tabs.set_title(i, t)
        self.tabs.add_class("mseg-tabs")

        console = W.VBox([W.HTML("<div class='mseg-h'>Console</div>"), self.out_log])
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

    def _on_include_var(self, b):
        feat = self.dd_var.value
        if not feat:
            return
        self.seg.include(feat)
        self._sync_sel(); self._refresh_vars(); self._refresh_bar()
        self._log(f"[incluir] '{self.seg.label(feat)}' no modelo · {len(self.seg.included)} no total.")

    def _on_exclude_var(self, b):
        feat = self.dd_var.value
        if not feat:
            return
        self.seg.exclude(feat)
        self._sync_sel(); self._refresh_vars(); self._refresh_bar()
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
        self._log(f"[categoria] {self.seg.label(self.dd_var.value)} = {cat}")

    # ------------------------------------------------------------------ Aba 2 handlers
    def _on_analyze(self, b):
        feat = self.dd_var2.value
        sample = None if self.dd_sample2.value == "(referência)" else self.dd_sample2.value
        tcol = self.tx_time2.value.strip() or None
        try:
            self.out_an_distbad.value = self._fig_html(
                self.seg.plot_variable_distribution_badrate(feat, sample=sample,
                                                            figsize=(6.4, 3.4)), tight=False)
            vt = self.seg.variable_table(feat, sample=sample)
            self.out_an_table.value = self._df_html(vt, max_height="240px", center=True)
            self.out_an_inv_sample.value = self._fig_html(
                self.seg.plot_variable_inversion_by_sample(feat, figsize=(6.0, 3.1)))
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
                self._log(f"[bins] '{self.seg.label(feat)}': nada para aplicar (campo vazio).")
            else:
                self._log(f"[bins] '{self.seg.label(feat)}': bins manuais aplicados.")
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
            self._refresh_bar()
            self._log(f"[nova variável] '{new}' criada de '{self.seg.label(feat)}' ({ncat} categorias) — "
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

        # z = intercepto + Σ coefᵢ·termoᵢ numa LINHA de equação limpa e discreta
        # (coeficiente colorido pelo sinal); o detalhe rigoroso fica na TABELA abaixo.
        parts = [f"<span class='mseg-{'pos' if b0 >= 0 else 'neg'}-tx'>{b0:+.4f}</span>"]
        for _, r in coef.iterrows():
            c = float(r["coef"]); cls = "pos" if c >= 0 else "neg"
            parts.append(
                f" <span style='color:#9aa3ad'>{'+' if c >= 0 else '−'}</span> "
                f"<span class='mseg-{cls}-tx'>{abs(c):.4f}</span>"
                f"<span style='color:#9aa3ad'>·</span>{r['termo']}")
        z_html = "<span style='color:#6b7480;font-weight:600'>z =</span> " + "".join(parts)

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
                cor = "#137a3e" if sig_ok else "#9aa3ad"
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
        # barra "ocupada" enquanto o fit (síncrono) roda
        self.btn_fit.disabled = True
        self.pb_fit.bar_style = "info"
        self.pb_fit.value = self.pb_fit.max
        self.pb_fit.description = "treinando…"
        self.pb_fit.layout.visibility = "visible"
        self.out_fit_status.value = (f"<div class='mseg-legend'><i>ajustando o modelo "
                                     f"({algo})… pode levar alguns segundos.</i></div>")
        try:
            self.seg.fit(algo, hyperparams=self._collect_hyperparams(algo), transform=transform)
            modo = "WoE/bins" if transform == "woe" else "valores crus"
            self._log(f"[fit] {algo} treinado com {len(self.seg.model_features)} "
                      f"variáveis ({modo}).")
            self.pb_fit.bar_style = "success"
            self.pb_fit.description = "concluído ✓"
            self.out_fit_status.value = (
                f"<div class='mseg-legend'><span style='color:#157a52;font-weight:600'>"
                f"✓ {algo} treinado com {len(self.seg.model_features)} variáveis "
                f"({modo}).</span></div>")
            self._render_metrics()
            self._render_model_plots()
            self._render_formula()
            self._refresh_bar()
        except Exception as e:
            self.pb_fit.bar_style = "danger"
            self.pb_fit.description = "erro"
            self.out_fit_status.value = (f"<div style='color:#b23a2a;font-size:12px'>"
                                         f"<b>Erro no fit:</b> {type(e).__name__}: {e}</div>")
            self._log(f"[fit] erro: {e}")
        finally:
            self.btn_fit.disabled = False

    def _on_tune(self, b):
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
        self.out_tune.value = "<i>Rodando otimização bayesiana (Optuna)…</i>"

        def _progress(done, total, best):
            self.pb_tune.value = done
            self.pb_tune.description = f"{done}/{total}"
            self.out_tune.value = (f"<div class='mseg-legend'>Optuna · trial {done}/{total} · "
                                   f"melhor até agora = <b>{best:.4f}</b></div>")

        try:
            res = self.seg.tune_optuna(algorithm=algo, n_trials=n_trials,
                                       transform=transform, fit_best=True,
                                       progress_callback=_progress)
        except Exception as e:
            self.pb_tune.bar_style = "danger"
            self.btn_tune.disabled = False
            self.out_tune.value = (f"<div style='color:#b3261e;font-size:12px'>Erro no tuning: "
                                   f"{type(e).__name__}: {e}</div>")
            self._log(f"[tune] erro: {e}")
            return
        self.pb_tune.value = self.pb_tune.max
        self.pb_tune.bar_style = "success"
        self.pb_tune.description = f"{res['n_trials']}/{res['n_trials']} ✓"
        self.btn_tune.disabled = False
        bp = "<br>".join(f"<code>{k}</code> = {v}" for k, v in res["best_params"].items())
        self.out_tune.value = (
            f"<div class='mseg-legend'><b>Optuna</b> · {res['n_trials']} trials · melhor "
            f"<b>{res['metric'].upper()} = {res['best_value']:.4f}</b> (no OOT/validação)"
            f"<br>{bp}</div>")
        self._log(f"[tune] {algo}: melhor {res['metric']}={res['best_value']:.4f} "
                  f"em {res['n_trials']} trials; modelo re-treinado com os melhores.")
        self._render_metrics()
        self._render_model_plots()
        self._render_formula()
        self._refresh_bar()

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

    def _sync_rating_method(self):
        """Mostra o campo de cortes/percentis só nos métodos manuais e ajusta o
        placeholder; nº ratings e fusão não se aplicam aos manuais."""
        m = self.dd_method.value
        manual = m in ("manual_score", "manual_percentil")
        self.tx_manual.layout.display = "" if manual else "none"
        if m == "manual_score":
            self.tx_manual.placeholder = "cortes de score (ex.: 0.2, 0.5, 0.8)"
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
            kw = {"cuts": nums} if method == "manual_score" else {"percentiles": nums}
        try:
            self.seg.build_ratings(method=method,
                                   n_ratings=int(self.sl_nratings.value),
                                   monotonic_fusion=self.cb_fusion.value, **kw)
            self._log(f"[ratings] {len(self.seg.rating_labels_)} faixas ({method}).")
            rt = self.seg.rating_table().round(4)
            rate_cols = [c for c in rt.columns if c.startswith(("event_rate", "alvo"))]
            self.out_rating_table.value = self._df_html(rt, center=True, pct_cols=rate_cols)
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
            self.out_backtest.value = self._df_html(
                self.seg.backtest(tcol).round(4), max_height="320px", center=True,
                color_validation=True,
                pct_cols=["previsto_medio", "realizado_medio", "gap"])
        except Exception as e:
            self.out_backtest.value = f"<i>{e}</i>"
        try:
            self.out_psi.value = self._df_html(self.seg.psi(), center=True,
                                               color_validation=True)
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

    def _confirm_overwrite(self, path, do_save):
        """Se ``path`` já existir, abre uma janela de confirmação (no console) e
        só executa ``do_save()`` quando o usuário clica em 'Sobrescrever'. Se o
        arquivo não existir (ou ``path`` for vazio), salva direto."""
        import html as _html
        import os
        path = (path or "").strip()
        if not path or not os.path.exists(path):
            do_save(); return
        aviso = W.HTML(
            "<div style='border:1px solid #f0c36d;background:#fff8e6;border-radius:10px;"
            "padding:10px 12px;font-size:12.5px;color:#664d03;line-height:1.5'>"
            "<b>⚠️ O arquivo já existe</b><br>"
            f"<code>{_html.escape(path)}</code><br>Deseja sobrescrever?</div>")
        btn_yes = W.Button(description="Sobrescrever", button_style="danger",
                           icon="exclamation-triangle")
        btn_no = W.Button(description="Cancelar", icon="times")
        def _yes(_):
            self.out_log.clear_output()
            do_save()
        def _no(_):
            self._log(f"[save] cancelado — '{path}' não foi sobrescrito.")
        btn_yes.on_click(_yes); btn_no.on_click(_no)
        with self.out_log:
            clear_output(wait=True)
            display(W.VBox([aviso, W.HBox([btn_yes, btn_no])]))

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
            self.out_pdf.value = (f"<div style='color:#b3261e;font-size:12px'>Erro ao gerar PDF: "
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
            self.out_md.value = (f"<div style='color:#b3261e;font-size:12px'>Erro ao gerar "
                                 f"Markdown: {type(e).__name__}: {e}</div>")
            self._log(f"[md] erro: {e}"); return
        self.out_md.value = (f"<div class='mseg-legend'>✅ Relatório salvo em <code>{path}</code> "
                             "(imagens salvas ao lado).</div>")
        self._log(f"[md] relatório salvo em {path}")

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
