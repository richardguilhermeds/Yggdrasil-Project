"""
TreeSegmenterUI
===============
Camada interativa (ipywidgets) sobre o `TreeSegmenter`, **unificada por**
``task_type`` ("classification" = PD/alvo binário · "regression" = LGD/alvo
contínuo) — a mesma UI atende os dois, mudando só o parâmetro.

Construa a árvore de segmentação clicando em botões, dentro do Jupyter, operando
sobre o DataFrame e o alvo reais. Recursos:
- árvore colorida pelo alvo médio que se atualiza a cada ação;
- **PSI ao vivo** por amostra (OOT, ESTABILIDADE, ...) no topo do painel;
- **discriminação ao vivo** por amostra: KS/AUC (classificação) ou R² (regressão);
- tabela de folhas com **PSI por amostra** e **p-valor** do teste entre folhas adjacentes;
- gráficos por tarefa: ROC/KS/taxa-default/distribuição (clf) ou boxplot/histograma (reg);
- travar folhas como finais (cadeado), podar, resetar e exportar o DataFrame rotulado.

    from yggdrasil.credit_risk.tree import TreeSegmenterUI
    ui = TreeSegmenterUI(df, target="target", task_type="classification",
                         sample_col="amostra", ref_sample="DES", feature_labels=labels)
    ui
"""
from __future__ import annotations

import pandas as pd

try:
    import ipywidgets as W
    from IPython.display import display
except Exception as e:  # pragma: no cover
    raise ImportError("Este módulo requer ipywidgets e IPython (Jupyter).") from e

from .segmenter import TreeSegmenter


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
.treeui { --ac:#3b4a63; --ac-deep:#27324a; --ac-soft:#eef1f5; --ac-border:#cdd5e0;
  --ink:#1f2733; --muted:#6b7480; --line:#e7e9ee;
  font-family:'IBM Plex Sans', -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  color:var(--ink); }
.treeui .mono { font-family:'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas,
  monospace; font-variant-numeric: tabular-nums; }
/* top bar (estilo mockup): branca, com chip PD grafite */
.treeui-banner { display:flex; align-items:center; gap:11px; background:#fff;
  border:1px solid var(--line); border-radius:13px; padding:11px 16px; margin-bottom:10px;
  box-shadow:0 1px 3px rgba(16,24,40,.08); }
.treeui-banner .logo { width:30px; height:30px; border-radius:9px; background:var(--ac);
  color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700;
  font-size:12px; flex:none; }
.treeui-banner .t { font-size:15px; font-weight:600; color:var(--ink); line-height:1.2; }
.treeui-banner .s { font-size:11.5px; color:var(--muted); margin-top:1px; }
/* cards */
.treeui-card { background:#fff; border:1px solid var(--line); border-radius:12px;
  padding:13px 15px; box-shadow:0 1px 3px rgba(16,24,40,.06); margin-bottom:11px; }
.treeui-h { font-weight:600; font-size:11px; color:var(--muted); text-transform:uppercase;
  letter-spacing:.07em; margin-bottom:9px; }
/* rótulos das faixas (cockpit/diagnóstico) e chips da folha ativa */
.treeui-band { font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.08em;
  color:var(--ac); margin:6px 2px 4px; }
.treeui-band-muted { color:#9aa2b1; margin-top:14px; }
.treeui-chips { display:flex; align-items:center; gap:6px; flex-wrap:wrap; padding:0 2px 4px; }
.treeui-chips .lab { font-size:11px; color:var(--muted); margin-right:2px; }
.treeui-chips .chip { font-size:11px; font-family:'IBM Plex Mono', ui-monospace, monospace;
  padding:2px 9px; border-radius:999px; border:1px solid var(--line); background:#fff; }
/* faixa de KPIs (health strip) sempre visível acima das abas */
.treeui-bar { background:#fff; border:1px solid var(--line); border-radius:11px;
  box-shadow:0 1px 3px rgba(16,24,40,.05); padding:0; overflow-x:auto; }
.pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:11.5px;
  font-weight:600; margin:2px 4px 2px 0; }
.pill-muted  { background:var(--ac-soft); color:var(--ac-deep); }
.pill-green  { background:#e7f5ee; color:#157a52; }
.pill-yellow { background:#fbf3e0; color:#9a6f12; }
.pill-red    { background:#fbe7e4; color:#b23a2a; }
.treeui-legend { font-size:11px; color:var(--muted); margin:6px 0 2px; line-height:1.55; }
.treeui-tree { line-height:1.55; }
/* abas do workbench — estilo "segmented control" (pílulas) */
.treeui-tabs { margin-top:10px; border:none !important; box-shadow:none !important; }
/* respiro entre a barra de abas e os cards do conteúdo abaixo
   (!important vence a regra própria do ipywidgets p/ .widget-tab-contents);
   border/box-shadow:none remove a "caixa" padrão do Tab ao redor de tudo */
.treeui-tabs > .widget-tab-contents { padding:30px 2px 2px !important; background:transparent;
  border:none !important; box-shadow:none !important; }
.treeui-tabs .lm-TabBar.jupyter-widget-tab-nav,
.treeui-tabs .p-TabBar.jupyter-widget-tab-nav { border-bottom:1px solid var(--line) !important;
  padding-bottom:14px !important; margin-bottom:0 !important; box-shadow:none !important; }
.treeui-tabs .lm-TabBar-content, .treeui-tabs .p-TabBar-content { gap:7px;
  align-items:stretch; border:none; }
.treeui-tabs .lm-TabBar-tab, .treeui-tabs .p-TabBar-tab { font-size:13px;
  /* !important vence a regra de mesma especificidade do ipywidgets
     (flex/max-width: var(--jp-widgets-horizontal-tab-width)) que cortava o título */
  min-width:max-content !important; max-width:none !important; flex:0 0 auto !important;
  margin:0 !important; padding:8px 16px !important;
  border:1px solid var(--line) !important; border-radius:9px !important;
  background:#fff !important; color:var(--muted) !important; font-weight:500;
  line-height:1.15; outline:none !important; box-shadow:none !important;
  transition:background .15s, color .15s, border-color .15s; }
/* o tema do Jupyter desenha a "barrinha azul" da aba ativa como um pseudo-
   elemento ::before (background var(--jp-brand-color1)); aqui ele some de vez */
.treeui-tabs .lm-TabBar-tab::before, .treeui-tabs .lm-TabBar-tab::after,
.treeui-tabs .p-TabBar-tab::before, .treeui-tabs .p-TabBar-tab::after {
  display:none !important; content:none !important; background:none !important; }
.treeui-tabs .lm-TabBar-tab:hover, .treeui-tabs .p-TabBar-tab:hover {
  background:var(--ac-soft) !important; color:var(--ac-deep) !important;
  border-color:var(--ac-border) !important; }
.treeui-tabs .lm-TabBar-tabLabel, .treeui-tabs .p-TabBar-tabLabel {
  white-space:nowrap !important; overflow:visible !important;
  text-overflow:clip !important; max-width:none !important; }
.treeui-tabs .lm-TabBar-tab.lm-mod-current,
.treeui-tabs .p-TabBar-tab.p-mod-current { color:#fff !important; font-weight:600;
  background:var(--ac) !important; border:1px solid var(--ac) !important;
  outline:none !important; box-shadow:none !important; }
.treeui-tabs .lm-TabBar-tab.lm-mod-current:hover,
.treeui-tabs .p-TabBar-tab.p-mod-current:hover {
  background:var(--ac-deep) !important; color:#fff !important;
  border-color:var(--ac-deep) !important; }
/* cabeçalho da folha selecionada (métricas em chips) — auto-fit estica os chips
   para preencher toda a largura (linhas com menos chips ficam mais largas) */
.treeui-metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(92px,1fr));
  gap:6px; }
.treeui-metric { background:#f7f8fa; border:1px solid #eef0f3; border-radius:9px;
  padding:7px 10px; overflow:hidden; }
.treeui-metric .k { font-size:10px; text-transform:uppercase; letter-spacing:.04em;
  color:#8a93a3; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.treeui-metric .v { font-size:16px; font-weight:600; color:var(--ink); margin-top:2px;
  white-space:nowrap; }
/* botões: cantos mais suaves, alinhados ao mockup */
.treeui .jupyter-button { border-radius:8px; font-family:inherit; }
/* sliders/controles encolhem para caber na coluna (min-width:0 libera o flex)
   e os cards clipam qualquer sobra horizontal — elimina a barra de rolagem
   horizontal que aparecia embaixo dos cards na aba Construir */
.treeui .jupyter-widgets { min-width:0 !important; }
.treeui-card { overflow-x:clip; }
/* ===== TEMA ESCURO (classe .dark no painel raiz) ===== */
.treeui.dark { --ink:#e7ebf2; --muted:#9aa6ba; --line:#2c3a55; --ac-soft:#243049;
  --ac-border:#3a4a6a; --ac-deep:#d6deec; --ac:#6076a0; background:#0e1521;
  padding:8px; border-radius:12px; }
.treeui.dark .treeui-banner, .treeui.dark .treeui-card, .treeui.dark .treeui-bar,
.treeui.dark .treeui-chips .chip { background:#16202f !important; border-color:#27344c !important; }
.treeui.dark .treeui-banner .t, .treeui.dark .treeui-band { color:#e7ebf2; }
.treeui.dark .treeui-band { background:#1c283a; }
.treeui.dark .treeui-tabs .p-TabBar-tab,
.treeui.dark .treeui-tabs .lm-TabBar-tab { background:#16202f !important; color:#9aa6ba !important;
  border-color:#27344c !important; }
.treeui.dark .treeui-tabs .p-TabBar-tab.p-mod-current,
.treeui.dark .treeui-tabs .lm-TabBar-tab.lm-mod-current { background:#243049 !important;
  color:#e7ebf2 !important; }
.treeui.dark .widget-text input, .treeui.dark .widget-dropdown select,
.treeui.dark textarea { background:#0e1521 !important; color:#e7ebf2 !important;
  border-color:#2c3a55 !important; }
.treeui.dark .widget-label, .treeui.dark .jupyter-widgets label { color:#c4cdde !important; }
</style>
"""


class TreeSegmenterUI:
    # mesma figsize p/ os dois gráficos lado a lado da faixa de detalhe
    # (distribuição da variável + cortes  e  histograma da PD/target da folha)
    _PREVIEW_FIGSIZE = (6.0, 3.6)

    # estilo das tabelas (Styler): bordas em cada célula p/ a divisão de colunas
    # ficar nítida, cabeçalho grafite fixo no topo e linhas com zebra leve.
    _TABLE_STYLES = [
        {"selector": "", "props": [("border-collapse", "collapse"),
                                   ("border", "1px solid #cdd5e0"),
                                   ("width", "100%")]},
        {"selector": "th, td", "props": [("border", "1px solid #e1e5ec"),
                                         ("padding", "4px 9px"),
                                         ("text-align", "right"),
                                         ("white-space", "nowrap")]},
        {"selector": "thead th", "props": [("background-color", "#eef1f5"),
                                           ("color", "#27324a"),
                                           ("font-weight", "600"),
                                           ("border-bottom", "2px solid #b9c2d0"),
                                           ("position", "sticky"),
                                           ("top", "0"), ("z-index", "1")]},
        {"selector": "tbody tr:nth-child(even) td", "props": [("background-color", "#fafbfc")]},
        {"selector": "tbody tr:hover td", "props": [("background-color", "#eef3f8")]},
    ]

    # Camada de estilo das tabelas "Detalhe por safra": cabeçalho claro (via
    # _TABLE_STYLES) + coluna 'safra' ancorada à esquerda (sticky horizontal).
    _SAFRA_HEADER_STYLES = [
        # cabeçalho idêntico ao das demais tabelas (claro, via _TABLE_STYLES);
        # aqui só ancoramos a coluna 'safra' à esquerda (sticky horizontal).
        {"selector": "tbody td:first-child", "props": [
            ("text-align", "left"),
            ("font-family", "'IBM Plex Sans',sans-serif"),
            ("font-weight", "600"), ("color", "#27324a"),
            ("position", "sticky"), ("left", "0"), ("z-index", "2"),
            ("background-color", "#f4f6f9"),
            ("border-right", "1px solid #cdd5e0")]},
        {"selector": "thead th:first-child", "props": [
            ("text-align", "left"), ("position", "sticky"),
            ("left", "0"), ("z-index", "3")]},
        {"selector": "tbody tr:nth-child(odd) td:first-child", "props": [
            ("background-color", "#f4f6f9")]},
        {"selector": "tbody tr:nth-child(even) td:first-child", "props": [
            ("background-color", "#eef1f5")]},
        {"selector": "tbody tr:hover td:first-child", "props": [
            ("background-color", "#eef3f8")]},
    ]

    @staticmethod
    def _blues_set_bad():
        """Cópia do cmap 'Blues' com 'bad'/'under' brancos, p/ que uma coluna
        categórica toda-NaN não vire barra preta sob background_gradient."""
        import matplotlib as mpl
        try:                                   # matplotlib >= 3.6
            cmap = mpl.colormaps["Blues"].copy()
        except Exception:                      # matplotlib < 3.6
            import matplotlib.cm as cm
            cmap = cm.get_cmap("Blues").copy()
        cmap.set_bad("#ffffff")
        cmap.set_under("#ffffff")
        return cmap

    @staticmethod
    def _accent_ramp_css(v, vmin, vmax, *, ceiling=0.55, na="#ffffff"):
        """Rampa branco → accent #3b4a63 interpolada à mão (tons pálidos).
        Fallback do heatmap categórico quando background_gradient falha."""
        if v is None or pd.isna(v):
            return "background-color:%s;color:#6b7480" % na
        span = (vmax - vmin)
        t = 0.0 if span <= 0 else (float(v) - vmin) / span
        t = min(max(t, 0.0), 1.0) * ceiling
        r = int(round(255 + (59 - 255) * t))
        g = int(round(255 + (74 - 255) * t))
        b = int(round(255 + (99 - 255) * t))
        fg = "#ffffff" if t > 0.40 else "#1f2733"
        return "background-color:rgb(%d,%d,%d);color:%s" % (r, g, b, fg)

    def __init__(self, df, target="target", task_type="classification",
                 sample_col=None, ref_sample="DES",
                 feature_labels=None, features=None, tree_samples=None, date_col=None):
        # task_type: "classification" (PD, alvo binário) ou "regression" (LGD,
        # alvo contínuo) — define métricas, IV, cor e os gráficos exibidos.
        # tree_samples: amostras cujo alvo médio aparece nas folhas da árvore.
        # None = todas; ex.: tree_samples=["DES","OOT"] mostra só DES e OOT.
        # date_col: coluna de data/safra — FORA da modelagem, só p/ gráficos no tempo.
        self.task_type = task_type
        self._is_clf = task_type == "classification"
        self._risk_label = "PD" if self._is_clf else "LGD"   # rótulo do alvo na UI
        self._risk_mean = "PD média" if self._is_clf else "LGD médio"   # frase "X média/médio"
        self._tree_samples_cfg = tree_samples
        self.date_col = date_col
        self._kwargs = dict(target=target, task_type=task_type, sample_col=sample_col,
                            ref_sample=ref_sample, feature_labels=feature_labels,
                            date_col=date_col, verbose=False)
        self.df = df
        self.target = target
        self.sample_col = sample_col
        self.ref_sample = ref_sample
        if features is None:
            skip = {target, sample_col, date_col}
            skip.discard(None)
            # datas/datetime nunca são variáveis do modelo (mesmo sem date_col)
            skip |= {c for c in df.columns
                     if pd.api.types.is_datetime64_any_dtype(df[c])}
            features = [c for c in df.columns if c not in skip]
        self.features = features

        self.seg = TreeSegmenter(df, **self._kwargs)
        self.locked: set = set()
        self._pending = None
        self.result = None
        self.spark_result = None      # último Spark DataFrame com a régua aplicada
        self._undo: list = []        # pilha de estados p/ desfazer splits/fusões
        self._redo: list = []        # pilha de estados p/ refazer

        # máscaras de amostra (fixas) e amostras ≠ referência (ex.: OOT)
        if sample_col is not None:
            self._samples = list(df[sample_col].dropna().unique())
            self._nonref = [a for a in self._samples if a != ref_sample]
            self._sample_masks = {a: (df[sample_col] == a) for a in self._samples}
            # amostras SEM variável resposta (ex.: ESTABILIDADE = público recente
            # só para validação): entram no PSI, mas não têm PD para exibir.
            self._psi_only = [a for a in self._nonref
                              if df.loc[self._sample_masks[a], target].notna().sum() == 0]
            # não-referência COM alvo (entra nas células/colunas de PD)
            self._pd_nonref = [a for a in self._nonref if a not in self._psi_only]
            # não-referência a EXIBIR na árvore (default: todas com PD)
            if tree_samples is not None:
                self._tree_nonref = [a for a in tree_samples
                                     if a in self._pd_nonref]
            else:
                self._tree_nonref = list(self._pd_nonref)
        else:
            self._samples, self._nonref, self._sample_masks = [], [], {}
            self._tree_nonref, self._psi_only, self._pd_nonref = [], [], []

        # --- estado de desempenho da UI ---------------------------------------
        # flag p/ suspender o observer de seleção de folha enquanto o _refresh
        # reatribui dd_leaf.value (senão _on_leaf_change re-dispara DENTRO do
        # _refresh → árvore/IV/histograma renderizados 2× por mutação).
        self._suspend_leaf_obs = False
        # cache de HTML por widget (hash-and-skip): só reescreve .value quando o
        # conteúdo muda — evita reenviar blobs idênticos pelo comm kernel↔browser.
        self._last_html: dict = {}
        # cache do PNG (base64) do histograma da folha, por (sid, versão da árvore)
        self._leaf_hist_cache: dict = {}

        self._build()
        self._on_mode_change(None)   # estado inicial de visibilidade dos controles
        self._sync_autoconc_visibility()   # sliders de concentração do auto-fit
        self._refresh()              # _refresh_iv já mescla o PSI/CSI por variável

    # ==================================================================
    # Construção dos widgets
    # ==================================================================
    def _build(self):
        full = W.Layout(width="98%")
        dstyle = {"description_width": "82px"}

        # dropdowns de folha/variável: Layout próprio (largo) e rótulo curto,
        # para mostrar o máximo possível do texto da opção selecionada
        self.dd_leaf = W.Dropdown(description="Folha", layout=W.Layout(width="100%"),
                                  style={"description_width": "52px"})
        # opções com o NOME DE EXIBIÇÃO (feature_labels) — valor = nome da coluna
        feat_opts = [(self.seg.feature_labels.get(f, f), f) for f in self.features]
        self.dd_feature = W.Dropdown(description="Variável", options=feat_opts,
                                     layout=W.Layout(width="100%"),
                                     style={"description_width": "62px"})
        self.tg_mode = W.ToggleButtons(options=["Ótimo", "Manual"], value="Ótimo",
                                       style={"button_width": "auto"},
                                       layout=W.Layout(width="100%"))
        self.sl_bins = W.IntSlider(description="máx. bins", min=2, max=15, value=4,
                                   layout=W.Layout(width="98%"), style=dstyle)
        # critério do split desta folha (modo Ótimo): optbin (multi-bin) ou CART/CHAID
        if self._is_clf:
            _scrit = [("Binning ótimo (IV)", "optbin"), ("Gini", "gini"),
                      ("Entropy / IG", "entropy"), ("KS", "ks"), ("IV gain", "iv"),
                      ("Qui-quadrado (CHAID)", "chi2")]
        else:
            _scrit = [("Binning ótimo (IV)", "optbin"), ("Redução de variância", "variance"),
                      ("Redução de MAE", "mae"), ("F-test / ANOVA", "ftest")]
        self.dd_split_criterion = W.Dropdown(description="critério", options=_scrit,
                                             value="optbin", layout=W.Layout(width="98%"),
                                             style=dstyle)
        self.dd_split_criterion.tooltip = ("Critério para escolher os cortes ao dividir ESTA "
                                           "folha no modo Ótimo (optbin = multi-bin por IV; "
                                           "demais = split binário CART/CHAID).")
        # Ótimo: limites OPCIONAIS de tamanho de bin (fração da folha) — optbinning
        self.cb_minbin = W.Checkbox(value=False, indent=False,
                                    description="limitar tamanho mínimo da bin",
                                    layout=W.Layout(width="98%"))
        self.sl_minbin = W.FloatSlider(description="mín. bin", min=0.01, max=0.30, step=0.01,
                                       value=0.05, readout_format=".0%",
                                       layout=W.Layout(width="98%"), style=dstyle)
        self.cb_maxbin = W.Checkbox(value=False, indent=False,
                                    description="limitar tamanho máximo da bin",
                                    layout=W.Layout(width="98%"))
        self.sl_maxbin = W.FloatSlider(description="máx. bin", min=0.10, max=1.0, step=0.05,
                                       value=0.50, readout_format=".0%",
                                       layout=W.Layout(width="98%"), style=dstyle)
        # Ótimo: diferença mínima de taxa de default exigida entre bins consecutivas
        self.cb_mindiff = W.Checkbox(value=False, indent=False,
                                     description="exigir diferença mínima entre bins",
                                     layout=W.Layout(width="98%"))
        self.sl_mindiff = W.FloatSlider(description=f"Δ{self._risk_label} bins", min=0.0,
                                        max=0.20, step=0.005,
                                        value=0.02, readout_format=".3f",
                                        layout=W.Layout(width="98%"), style=dstyle)
        _diff_tip = "taxa de default" if self._is_clf else "alvo médio (LGD)"
        self.sl_mindiff.tooltip = (f"Diferença mínima de {_diff_tip} entre duas bins "
                                   "consecutivas no binning ótimo (min_mean_diff)")
        self.tx_cuts = W.Text(description="Cortes", layout=W.Layout(width="98%"), style=dstyle,
                              placeholder="num: 0.7,0.9  |  cat: a,b; c")

        self.sl_repr = W.FloatSlider(description="min repr%", min=0, max=10, step=0.5,
                                     value=3.0, layout=full, style=dstyle)
        self.sl_repr.tooltip = "Representatividade mínima por folha (%); abaixo disso, funde com a irmã"
        self.sl_gap = W.FloatSlider(description=f"Δ{self._risk_label} mínimo", min=0, max=0.10, step=0.002,
                                    value=0.02, readout_format=".3f", layout=full, style=dstyle)
        self.sl_gap.tooltip = f"Diferença mínima de {self._risk_label} entre irmãs; abaixo disso, as duas são unidas (0.02 = 2 p.p.)"
        self.dd_test = W.Dropdown(description="Teste",
                                  options=[("Mann-Whitney", "mannwhitney"), ("Welch t", "welch")],
                                  value="mannwhitney", layout=W.Layout(width="100%"),
                                  style={"description_width": "44px"})

        def mk(desc, style, tip, icon):
            return W.Button(description=desc, button_style=style, tooltip=tip, icon=icon,
                            layout=W.Layout(width="98%", margin="2px 0"))
        self.btn_preview = mk("Preview", "info", f"Mostra {self._risk_label} e representatividade (não altera)", "eye")
        self.btn_split = mk("Criar segmento", "success", "Efetiva o split na folha", "scissors")
        self.btn_sugcuts = mk("Sugerir cortes & máx. bins", "warning",
                              "Roda o binning ótimo da variável NESTA folha e preenche o 'máx. "
                              "bins' e os 'Cortes' com a sugestão", "magic")
        self.btn_lock = mk("Fechar folha", "warning", "Trava a folha (não será dividida)", "lock")
        self.btn_unlock = mk("Reabrir folha", "", "Destrava a folha", "unlock")
        self.btn_prune = mk("Podar", "danger",
                            "Funde folhas-irmãs com representatividade < min repr% ou diferença "
                            f"de {self._risk_label} < Δ{self._risk_label} mínimo", "cut")
        self.btn_reset = mk("Reset", "", "Recomeça do zero", "refresh")
        self.btn_export = mk("Exportar", "primary", "Gera ui.result com o rótulo", "download")
        # copiar a tabela de folhas p/ o Excel (TSV pronto p/ colar)
        self.btn_copy_table = W.Button(
            description="Copiar p/ Excel (TSV)", icon="table",
            tooltip="Gera a tabela em TSV — selecione tudo (Ctrl+A), copie (Ctrl+C) "
                    "e cole no Excel: colunas certas e números em pt-BR",
            layout=W.Layout(width="auto", margin="6px 0 2px"))
        self.out_table_tsv = W.Textarea(
            value="", layout=W.Layout(width="99%", height="150px", display="none"),
            placeholder="TSV das folhas — selecione tudo (Ctrl+A) e copie (Ctrl+C)")
        self.btn_collapse = mk("Recolher p/ o pai", "danger",
                               "Desfaz o split: recolhe a folha de volta ao segmento pai", "compress")
        self.btn_merge_l = mk("Fundir ◀", "warning",
                              f"Funde a folha com a vizinha de menor corte (num) / menor {self._risk_label} (cat)", "arrow-left")
        self.btn_merge_r = mk("Fundir ▶", "warning",
                              f"Funde a folha com a vizinha de maior corte (num) / maior {self._risk_label} (cat)", "arrow-right")
        self.btn_merge_na = mk("Juntar missings", "warning",
                               "Junta o nó de faltantes/missings (NaN) deste split dentro da folha "
                               "populada selecionada — a regra vira 'bin OU missing'", "link")
        self.btn_suggest = mk("Sugerir split", "info",
                              "Recomenda a variável de maior IV para a folha selecionada", "lightbulb-o")
        self.btn_autofit = mk("Auto-fit (árvore)", "info",
                              "Constrói uma árvore gulosa por IV até a profundidade escolhida", "magic")
        self.sl_depth = W.IntSlider(description="profundidade", min=1, max=5, value=3,
                                    layout=W.Layout(width="98%"), style=dstyle)
        # critério do split automático: "optbin" (binning ótimo multi-bin) ou um
        # critério CART/CHAID (split binário). Opções conforme o tipo de alvo.
        if self._is_clf:
            crit_opts = [("Binning ótimo (IV multi-bin)", "optbin"), ("Gini (CART)", "gini"),
                         ("Entropy / Information Gain", "entropy"), ("KS (separação good/bad)", "ks"),
                         ("IV gain", "iv"), ("Qui-quadrado (CHAID)", "chi2")]
        else:
            crit_opts = [("Binning ótimo (IV multi-bin)", "optbin"),
                         ("Redução de variância", "variance"),
                         ("Redução de MAE (robusto)", "mae"), ("F-test / ANOVA", "ftest")]
        self.dd_criterion = W.Dropdown(description="critério", options=crit_opts,
                                       value="optbin", layout=W.Layout(width="98%"), style=dstyle)
        self.dd_criterion.tooltip = ("Como escolher os cortes no Auto-fit: binning ótimo "
                                     "(multi-bin por IV) ou um critério de split binário (CART/CHAID).")
        # ---- widgets da aba "Avançado" (sugerir splits, importância, SQL, diff) ----
        self.btn_suggest3 = mk("Sugerir TOP 3 splits", "info",
                               "Lista as 3 melhores variáveis p/ dividir a folha selecionada", "lightbulb-o")
        self.out_suggest = W.HTML()
        self.btn_importance = mk("Calcular importância", "info",
                                 "Importância das variáveis que entraram na árvore", "bar-chart")
        self.out_importance = W.HTML()
        self.out_importance_chart = W.HTML()   # gráfico de importância relativa (ao lado da tabela)
        self.out_importance_legend = W.HTML()  # legenda explicativa (abaixo de tabela + gráfico)
        self.btn_sql = mk("Gerar SQL (CASE WHEN)", "primary",
                          "Gera a régua como SQL copiável", "database")
        self.tx_sql_table = W.Text(description="tabela", value="minha_tabela",
                                   layout=W.Layout(width="60%"), style=dstyle)
        self.out_sql = W.Textarea(layout=W.Layout(width="99%", height="240px"))
        self.tx_diff_path = W.Text(description="árvore B (JSON)", placeholder="caminho do .json salvo",
                                   layout=W.Layout(width="95%"), style=dstyle)
        self.btn_diff = mk("Comparar com árvore B", "warning",
                           "Carrega outra árvore (JSON) e compara com a atual", "exchange")
        self.out_diff = W.HTML()
        # concentração das folhas no auto-fit — REPRESENTATIVIDADE GLOBAL (% da
        # carteira inteira). Cada uma só atua se o respectivo checkbox estiver marcado.
        self.cb_autoconc_min = W.Checkbox(value=True, indent=False,
                                          description="concentração mínima da folha (% carteira)",
                                          layout=W.Layout(width="98%"))
        self.sl_autoconc_min = W.FloatSlider(description="conc. mín.", min=0.005, max=0.25,
                                             step=0.005, value=0.03, readout_format=".1%",
                                             layout=W.Layout(width="98%"), style=dstyle)
        self.sl_autoconc_min.tooltip = ("Cada folha terminal reterá ao menos esta fração da "
                                        "CARTEIRA inteira (não da folha-mãe)")
        self.cb_autoconc_max = W.Checkbox(value=False, indent=False,
                                          description="concentração máxima por quebra (% carteira)",
                                          layout=W.Layout(width="98%"))
        self.sl_autoconc_max = W.FloatSlider(description="conc. máx.", min=0.20, max=0.90,
                                             step=0.05, value=0.50, readout_format=".0%",
                                             layout=W.Layout(width="98%"), style=dstyle)
        self.sl_autoconc_max.tooltip = ("Nenhuma quebra concentrará mais que esta fração da "
                                        "carteira (força granularidade em segmentos dominantes; "
                                        "amplia o nº de bins automaticamente)")
        self.tx_experiment = W.Text(description="experimento", placeholder="opcional (usa o do notebook)",
                                    layout=full, style=dstyle)
        self.tx_runname = W.Text(description="run", placeholder="opcional",
                                 layout=full, style=dstyle)
        self.tx_model = W.Text(description="modelo", placeholder="catalogo.schema.modelo",
                               layout=full, style=dstyle)
        self.cb_uc = W.Checkbox(value=True, description="Registrar no Unity Catalog",
                                indent=False, layout=W.Layout(width="98%"))
        self.btn_mlflow = mk("Salvar no MLflow", "primary",
                             "Loga régua, métricas e o modelo pyfunc, e registra a versão no Model Registry", "save")
        self.btn_clear_log = mk("Limpar log", "", "Limpa a área de preview/log", "eraser")
        self.sl_boot = W.IntSlider(description="reamostras", min=200, max=5000, step=100,
                                   value=1000, layout=full, style=dstyle)
        self.btn_boot = mk("Calcular IC bootstrap", "primary",
                           f"Calcula o IC {'da PD' if self._is_clf else 'do LGD'} por folha e a aderência em OOT", "random")
        # --- placar de saúde do modelo (aba Diagnóstico) ---
        self.btn_diag = mk("Avaliar modelo (placar)", "primary",
                           "Calcula o placar de saúde: discriminação (KS/AUC/Gini), estabilidade "
                           "(PSI/CSI), calibração (previsto×observado) e estrutura "
                           "(monotonicidade · distinção entre folhas-irmãs)", "stethoscope")
        self.btn_diag_hide = mk("Ocultar", "", "Limpa a avaliação já renderizada", "eye-slash")
        # --- comparação de folhas-irmãs (inversão entre amostras/safras) ---
        sib_style = {"description_width": "118px"}
        self.dd_sib_group = W.Dropdown(description="Grupo de irmãs", layout=full,
                                       style=sib_style)
        self.dd_sib_sample = W.Dropdown(description="amostra (safra)", layout=full,
                                        style=sib_style)
        self.tx_sib_time = W.Text(description="coluna safra", value=(self.date_col or "dt_ref"),
                                  layout=full, style=sib_style,
                                  placeholder="coluna de safra (ex.: dt_ref)")
        self.btn_sib = mk("Analisar folhas-irmãs (inversão)", "primary",
                          f"Compara {'a PD média' if self._is_clf else 'o LGD médio'} das folhas de mesmo pai por amostra e por "
                          "safra e sinaliza inversões da ordem de risco", "exchange")
        # --- validação regulatória (monotonicidade, calibração, backtest) e relatório ---
        self.tx_time_col = W.Text(description="coluna tempo", value="dt_ref",
                                  layout=full, style=dstyle,
                                  placeholder="coluna de safra p/ o backtest (ex.: dt_ref)")
        self.btn_validate = mk("Validar (monoton. · calibração · backtest)", "info",
                               "Mostra monotonicidade das notas, calibração prevista×realizada e "
                               "backtest por safra", "check-square-o")
        self.tx_report_path = W.Text(description="relatório", value="relatorio_validacao.md",
                                     layout=full, style=dstyle, placeholder="caminho .md")
        self.btn_report = mk("Gerar relatório de validação (MD)", "success",
                             "Gera um documento Markdown com árvore, folhas, PSI, CSI, discriminação, "
                             "calibração e backtest (+ imagens)", "file-text-o")
        # --- discriminação (KS · ROC) / dispersão do alvo na regressão ---
        self.btn_roc = mk("Curva ROC (AUC/Gini)", "info",
                          "Curva ROC da régua por amostra, com a AUC e o Gini", "line-chart")
        self.btn_ks = mk("Curva KS", "info",
                         "Curva KS — distribuições acumuladas de bons e maus pelo score", "area-chart")

        # --- undo/redo, auto-merge e persistência da árvore (JSON) ---
        self.btn_undo = mk("◀ Desfazer", "", "Desfaz a última alteração na árvore", "undo")
        self.btn_redo = mk("Refazer ▶", "", "Refaz a alteração desfeita", "repeat")
        self.btn_undo.disabled = True
        self.btn_redo.disabled = True
        self.btn_automerge = mk("Auto-fundir folhas", "warning",
                                "Funde automaticamente folhas-irmãs indistinguíveis (p > alpha)", "compress")
        self.sl_alpha = W.FloatSlider(description="alpha", min=0.01, max=0.50, step=0.01,
                                      value=0.05, readout_format=".2f", layout=full, style=dstyle)
        self.cb_automerge_na = W.Checkbox(value=False, indent=False,
                                          description="também juntar faltantes ao bin mais próximo",
                                          layout=W.Layout(width="98%"))
        self.tx_json_path = W.Text(description="arquivo", value="arvore_pd.json",
                                   layout=full, style=dstyle, placeholder="caminho .json")
        self.btn_save_json = mk("Salvar árvore (JSON)", "success",
                                "Salva a estrutura da árvore num arquivo JSON", "save")
        self.btn_load_json = mk("Carregar árvore (JSON)", "info",
                                "Carrega uma árvore salva e reaplica ao DataFrame atual", "upload")
        # --- imagem da árvore (matplotlib) ---
        self.tx_img_path = W.Text(description="imagem", value="arvore_pd.png",
                                  layout=full, style=dstyle,
                                  placeholder="caminho .png/.svg (opcional)")
        self.btn_plot = mk("Ver / salvar árvore (imagem)", "info",
                           f"Renderiza a árvore como imagem ({self._risk_mean} e % por folha) e salva "
                           "se um caminho for informado", "picture-o")
        self.btn_plot_hide = mk("Recolher imagem", "", "Oculta a imagem da árvore", "eye-slash")
        # --- relatório PDF do modelo (capa + métricas + árvore + folhas + calibração) ---
        self.tx_pdf_path = W.Text(description="arquivo", value="relatorio_arvore.pdf",
                                  layout=full, style=dstyle,
                                  placeholder="caminho .pdf onde salvar o relatório")
        self.btn_pdf = mk("Gerar relatório PDF", "primary",
                          "Salva um relatório PDF do modelo no caminho informado", "file-pdf-o")
        self.out_pdf = W.HTML()
        # --- aplicar a régua numa tabela Spark ("reconstruir as folhas") ---
        # inputs com mais respiro vertical (tabela/saída mais espaçadas)
        spark_lay = W.Layout(width="98%", margin="9px 0")
        self.tx_spark_in = W.Text(description="tabela", layout=spark_lay, style=dstyle,
                                  placeholder="tabela Spark de entrada (catalogo.schema.tabela)")
        self.tx_spark_out = W.Text(description="saída", layout=spark_lay, style=dstyle,
                                   placeholder="opcional: grava o resultado nesta tabela")
        self.btn_spark_apply = mk("Reconstruir folhas (Spark)", "primary",
                                  f"Aplica a régua à tabela Spark (segmento, nota e {self._risk_label} por linha), "
                                  "desde que as colunas tenham o mesmo nome", "table")
        # --- controles da aba "Análise de variáveis" ---
        # opções com o NOME DE EXIBIÇÃO (feature_labels) — valor = nome da coluna
        var_opts = [(self.seg.feature_labels.get(f, f), f) for f in self.features]
        self.dd_var = W.Dropdown(description="Variável", options=var_opts,
                                 layout=full, style=dstyle)
        self.dd_var_leaf = W.Dropdown(description="Folha", layout=full, style=dstyle)
        self.tx_var_time = W.Text(description="coluna safra", value=(self.date_col or "dt_ref"),
                                  layout=full, style=dstyle,
                                  placeholder="coluna de safra (ex.: dt_ref) — opcional")
        self.btn_var_analyze = mk("Analisar variável", "primary",
                                  "Calcula distribuição, estatísticas, PSI atual e o "
                                  "comportamento por safra da variável na folha escolhida",
                                  "search")
        # --- preview da árvore como imagem, no fim do Construir (sem exportar) ---
        self.btn_tree_preview = mk("Ver árvore (imagem)", "info",
                                   "Renderiza a árvore como imagem aqui mesmo — sem exportar/salvar",
                                   "sitemap")
        self.btn_tree_preview_hide = mk("Ocultar", "", "Oculta a imagem da árvore", "eye-slash")

        self.btn_preview.on_click(self._on_preview)
        self.btn_sugcuts.on_click(self._on_suggest_cuts)
        self.btn_split.on_click(self._on_split)
        self.btn_lock.on_click(self._on_lock)
        self.btn_unlock.on_click(self._on_unlock)
        self.btn_prune.on_click(self._on_prune)
        self.btn_reset.on_click(self._on_reset)
        self.btn_export.on_click(self._on_export)
        self.dd_leaf.observe(self._on_leaf_change, names="value")
        self.dd_test.observe(lambda _: self._refresh_table(), names="value")
        self.btn_copy_table.on_click(self._on_copy_table)
        self.btn_collapse.on_click(self._on_collapse)
        self.btn_merge_l.on_click(lambda _: self._on_merge("left"))
        self.btn_merge_r.on_click(lambda _: self._on_merge("right"))
        self.btn_merge_na.on_click(self._on_merge_missing)
        self.btn_suggest.on_click(self._on_suggest)
        self.btn_suggest3.on_click(self._on_suggest3)
        self.btn_importance.on_click(self._on_importance)
        self.btn_sql.on_click(self._on_sql)
        self.btn_diff.on_click(self._on_diff)
        self.btn_autofit.on_click(self._on_autofit)
        self.btn_mlflow.on_click(self._on_mlflow)
        self.btn_clear_log.on_click(self._on_clear_log)
        self.btn_boot.on_click(self._on_boot)
        self.btn_diag.on_click(self._on_diag)
        self.btn_diag_hide.on_click(self._on_diag_hide)
        self.btn_sib.on_click(self._on_sib_analyze)
        # amostras p/ a análise por safra das folhas-irmãs (fixas — não mudam
        # com a árvore): "todas" + a referência (DES) + as demais com PD.
        sib_samples = [("todas as amostras", "__all__")]
        if self.sample_col is not None:
            sib_samples += [(self.ref_sample, self.ref_sample)]
            sib_samples += [(a, a) for a in self._pd_nonref]
        self.dd_sib_sample.options = sib_samples
        self.btn_validate.on_click(self._on_validate)
        self.btn_report.on_click(self._on_report)
        if self._is_clf:
            self.btn_roc.on_click(self._on_roc)
            self.btn_ks.on_click(self._on_ks)
        else:
            # regressão: reusa os 2 botões da discriminação p/ boxplot e
            # histograma do alvo (ROC/KS não se aplica a alvo contínuo)
            self.btn_roc.description = "📦 Boxplot por folha"
            self.btn_ks.description = "📊 Histograma do alvo"
            self.btn_roc.tooltip = "Boxplot do alvo (LGD) por folha — dispersão dentro de cada folha"
            self.btn_ks.tooltip = "Histograma do alvo (LGD) na carteira"
            self.btn_roc.on_click(self._on_box)
            self.btn_ks.on_click(self._on_hist)
        self.btn_undo.on_click(self._on_undo)
        self.btn_redo.on_click(self._on_redo)
        self.btn_automerge.on_click(self._on_automerge)
        self.btn_save_json.on_click(self._on_save_json)
        self.btn_load_json.on_click(self._on_load_json)
        self.btn_pdf.on_click(self._on_pdf)
        self.btn_plot.on_click(self._on_plot)
        self.btn_plot_hide.on_click(self._on_plot_hide)
        self.btn_spark_apply.on_click(self._on_spark_apply)
        self.btn_var_analyze.on_click(self._on_var_analyze)
        self.btn_tree_preview.on_click(self._on_tree_preview)
        self.btn_tree_preview_hide.on_click(lambda _: setattr(self.out_tree_img, "value", ""))
        self.tg_mode.observe(self._on_mode_change, names="value")
        self.dd_feature.observe(self._on_feature_change, names="value")
        self.cb_minbin.observe(lambda _: self._sync_optbin_visibility(), names="value")
        self.cb_maxbin.observe(lambda _: self._sync_optbin_visibility(), names="value")
        self.cb_mindiff.observe(lambda _: self._sync_optbin_visibility(), names="value")
        self.cb_autoconc_min.observe(lambda _: self._sync_autoconc_visibility(), names="value")
        self.cb_autoconc_max.observe(lambda _: self._sync_autoconc_visibility(), names="value")

        # HTML widgets (.value substitui o conteúdo de forma confiável em qualquer
        # frontend — Jupyter e Databricks — evitando a duplicação que o
        # Output+display+clear_output causa quando o clear não limpa).
        self.bar = W.HTML()
        self.out_tree = W.HTML()
        # bloco <style> que aplica o REALCE da folha selecionada via CSS (data-leaf).
        # Trocar a folha atualiza só este blob minúsculo, sem remontar/reenviar a
        # árvore inteira pelo comm — o _tree_html passa a ser independente da seleção.
        self.tree_sel_style = W.HTML()
        self.out_metrics = W.HTML()
        self.out_iv = W.HTML()
        self.out_leaf_hist = W.HTML()                     # PD da folha
        self.out_plot = W.HTML()
        self.out_boot = W.HTML()
        self.out_validate = W.HTML()
        self.out_discrim = W.HTML()    # ROC/KS (clf) · boxplot/histograma do alvo (reg)
        self.out_sib = W.HTML()     # comparação de folhas-irmãs (inversão)
        self.out_diag = W.HTML()    # placar de saúde do modelo (Diagnóstico)
        self.out_log = W.Output(layout=W.Layout(max_height="320px", overflow="auto"))
        self.out_preview_chart = W.HTML()   # distribuição da variável + cortes (ao lado do histograma)
        self.out_preview_seg = W.HTML()     # segmentação proposta (dentro de "Dividir a folha")
        self.out_table = W.HTML()
        # aba "Análise de variável"
        self.out_var_dist = W.HTML()
        self.out_var_time = W.HTML()
        self.out_var_psi = W.HTML()
        self.out_var_table = W.HTML()
        self.out_var_cards = W.HTML()
        self.out_tree_img = W.HTML()                      # preview da árvore
        self.cat_box = W.VBox([], layout=W.Layout(width="98%", display="none",
                                                  border="1px solid #eef1f4",
                                                  padding="6px 8px", margin="2px 0"))
        self.leaf_header = W.HTML()   # resumo da folha selecionada (faixa de detalhe)
        self.leaf_chips = W.HTML()    # resumo curto da folha ativa (régua do topo)

        # ================================================================
        # WORKBENCH EM ABAS
        # Sempre visíveis no topo: banner + faixa de KPIs (saúde da árvore).
        # As ações ficam organizadas em 5 abas; o LOG vai para um console
        # persistente abaixo das abas, para que mensagens de qualquer aba
        # apareçam (um widget só pode estar em um lugar da árvore de widgets).
        # ================================================================
        if self._is_clf:
            _bg_logo, _bg_titulo = "PD", "Segmentação de PD"
            _bg_sub = "optimal binning binário · KS/AUC ao vivo"
        else:
            _bg_logo, _bg_titulo = "LGD", "Segmentação de LGD"
            _bg_sub = "optimal binning contínuo · MAE/RMSE/R² ao vivo"
        banner = W.HTML(_CSS +
            f"<div class='treeui-banner'><div class='logo'>{_bg_logo}</div>"
            f"<div><div class='t'>{_bg_titulo}</div>"
            f"<div class='s'>Construtor de árvore · {_bg_sub} · "
            "PSI ao vivo (DES) · teste de hipótese entre folhas adjacentes</div></div></div>")
        bar_box = W.VBox([self.bar]); bar_box.add_class("treeui-bar")

        # ---- legendas reutilizadas (task-aware) -------------------------
        _rl = "PD" if self._is_clf else "LGD"
        tree_legend = W.HTML(
            f"<div class='treeui-legend'>cor do quadrado = {_rl} "
            "(<span style='color:#1aa64b'>baixo</span> &rarr; "
            "<span style='color:#caa000'>médio</span> &rarr; "
            "<span style='color:#d6453e'>alto</span>) · 🔒 folha fechada</div>")
        if self._is_clf:
            _iv_intro = "<b>IV</b> (optbinning · WoE binário) = poder de"
            _iv_faixas = ("Faixas (Siddiqi): <span style='color:#137a3e'>forte (0,3–0,5)</span> · "
                          "<span style='color:#9a6b00'>médio (0,1–0,3)</span> · fraco/inútil (&lt;0,1) · "
                          "<span style='color:#6b3fa0'>suspeito (&ge;0,5)</span> (alto demais, verifique vazamento).")
        else:
            _iv_intro = "<b>IV</b> (optbinning · contínuo) = poder de"
            _iv_faixas = ("Faixas: <span style='color:#137a3e'>forte (0,1–0,35)</span> · "
                          "<span style='color:#9a6b00'>médio (0,03–0,1)</span> · fraco/inútil (&lt;0,03) · "
                          "<span style='color:#6b3fa0'>suspeito (&ge;0,35)</span>.")
        iv_legend = W.HTML(
            f"<div class='treeui-legend'>{_iv_intro} "
            f"separação da variável na <b>folha selecionada</b> (★ = maior). {_iv_faixas} "
            "<b>bins</b> = nº de faixas ideais do binning ótimo na folha.</div>"
            "<div class='treeui-legend' style='margin-top:6px;padding-top:6px;"
            "border-top:1px solid #eef1f4'><b>PSI</b> = estabilidade da variável (DES × demais "
            "amostras), calculado <b>nos mesmos bins do IV</b>, pior caso: "
            "<span style='color:#137a3e'>&lt;0.10 estável</span> · "
            "<span style='color:#9a6b00'>0.10–0.25 atenção</span> · "
            "<span style='color:#b3261e'>&ge;0.25 instável</span>.</div>")

        # ================================================================
        # ABA ① CONSTRUIR — "Cockpit em T"
        #   TOPO: Árvore & quebras  ·AO LADO·  Information Value  + régua da folha
        #   DETALHE: folha (detalhe) | dividir | ações + auto-fit  e, abaixo,
        #     distribuição da variável+cortes | histograma da PD da folha
        #   RODAPÉ: Preview da árvore (imagem) em largura total  (Assistente OFF)
        # ================================================================
        sep_top = W.HTML("<div class='treeui-band'>① Topo · loop árvore → folha → IV → agir</div>")

        tree_scroll = W.Box([self.out_tree],
                            layout=W.Layout(overflow="auto", width="100%",
                                            max_height="420px"))
        card_tree = W.VBox([
            W.HTML("<div class='treeui-h'>Árvore &amp; quebras</div>"),
            tree_legend, tree_scroll,
        ], layout=W.Layout(width="54%"))
        card_tree.add_class("treeui-card")
        card_iv = W.VBox([
            W.HTML("<div class='treeui-h'>Information Value · qual variável segmentar</div>"),
            iv_legend, self.out_iv,
        ], layout=W.Layout(width="44%"))
        card_iv.add_class("treeui-card")
        top_cols = W.HBox([card_tree, card_iv],
                          layout=W.Layout(width="100%", align_items="flex-start",
                                          justify_content="space-between"))

        # ---- DETALHE · linha 1: folha (detalhe) | dividir | ações + auto-fit
        sep_det = W.HTML("<div class='treeui-band treeui-band-muted'>② Detalhe / inspeção — "
                         "role quando precisar</div>")
        card_leaf = W.VBox([self.leaf_header]); card_leaf.add_class("treeui-card")
        det_c1 = W.VBox([card_leaf], layout=W.Layout(width="30%"))

        self.btn_sugcuts.layout.width = "99%"
        card_split = W.VBox([
            W.HTML("<div class='treeui-h'>Dividir a folha selecionada</div>"),
            self.dd_leaf, self.dd_feature, self.btn_sugcuts, self.tg_mode,
            self.sl_bins, self.dd_split_criterion,
            self.cb_minbin, self.sl_minbin, self.cb_maxbin, self.sl_maxbin,
            self.cb_mindiff, self.sl_mindiff,
            self.tx_cuts, self.cat_box,
            W.HBox([self.btn_preview, self.btn_split]),
            self.out_preview_seg,
        ]); card_split.add_class("treeui-card")
        det_c2 = W.VBox([card_split], layout=W.Layout(width="44%"))

        for _b in (self.btn_lock, self.btn_unlock, self.btn_collapse,
                   self.btn_merge_l, self.btn_merge_r, self.btn_merge_na):
            _b.layout.width = "100%"
            _b.layout.margin = "2px 0"
        card_actions = W.VBox([
            W.HTML("<div class='treeui-h'>Ações da folha</div>"),
            W.HBox([self.btn_undo, self.btn_redo]),
            self.btn_lock, self.btn_unlock, self.btn_collapse,
            W.HBox([self.btn_merge_l, self.btn_merge_r]),   # fundir ◀ / ▶ lado a lado
            self.btn_merge_na,
        ], layout=W.Layout(width="100%"))
        card_actions.add_class("treeui-card")
        card_autofit = W.VBox([
            W.HTML("<div class='treeui-h'>Auto-fit</div>"),
            W.HTML("<div class='treeui-legend'>Constrói a árvore gulosa por IV até a "
                   "profundidade escolhida. As concentrações são <b>% da carteira inteira</b>: "
                   "<b>mín.</b> evita folhas terminais pequenas; <b>máx.</b> impede que uma "
                   "quebra concentre demais. Com uma <b>folha selecionada</b> (≠ raiz), cresce "
                   "<b>apenas aquela folha</b>; na raiz, reconstrói tudo.</div>"),
            self.sl_depth,
            self.dd_criterion,
            self.cb_autoconc_min, self.sl_autoconc_min,
            self.cb_autoconc_max, self.sl_autoconc_max,
            W.HBox([self.btn_autofit, self.btn_reset]),
        ]); card_autofit.add_class("treeui-card")
        det_c3 = W.VBox([card_actions, card_autofit], layout=W.Layout(width="24%"))

        det_row = W.HBox([det_c1, det_c2, det_c3],
                         layout=W.Layout(width="100%", align_items="flex-start",
                                         justify_content="space-between"))

        # ---- DETALHE · linha 2: distribuição+cortes (preview) | histograma da PD
        card_preview = W.VBox([
            W.HTML("<div class='treeui-h'>Distribuição da variável · cortes sugeridos</div>"),
            W.HTML("<div class='treeui-legend'>Distribuição da variável na folha selecionada "
                   "(DES), com os cortes propostos marcados.</div>"),
            self.out_preview_chart,
        ], layout=W.Layout(width="49%")); card_preview.add_class("treeui-card")
        if self._is_clf:
            _hist_h = "PD da folha (taxa de default · DES)"
            _hist_leg = ("Taxa de default da folha selecionada (DES), com IC de Wilson e a "
                         "PD da carteira como referência.")
        else:
            _hist_h = "LGD da folha (alvo médio · DES)"
            _hist_leg = ("Distribuição do alvo (LGD) na folha selecionada (DES), com a média "
                         "da folha e a da carteira como referência.")
        card_hist = W.VBox([
            W.HTML(f"<div class='treeui-h'>{_hist_h}</div>"),
            W.HTML(f"<div class='treeui-legend'>{_hist_leg}</div>"),
            self.out_leaf_hist,
        ], layout=W.Layout(width="49%")); card_hist.add_class("treeui-card")
        det_bottom = W.HBox([card_preview, card_hist],
                            layout=W.Layout(width="100%", align_items="stretch",
                                            justify_content="space-between"))

        # ---- Assistente: DESATIVADO por enquanto -------------------------
        # (sugerir · auto-fundir · podar) — widgets seguem criados; p/ reativar,
        # monte o card e inclua-o numa coluna do detalhe.
        #   self.btn_suggest · self.sl_alpha, self.cb_automerge_na, self.btn_automerge
        #   self.sl_repr, self.sl_gap, self.dd_test, self.btn_prune
        # (o seletor "Teste" mora aqui; com o Assistente off, usa o padrão Mann-Whitney)

        # ---- RODAPÉ: Preview da árvore (imagem), largura total -----------
        sep_img = W.HTML("<div class='treeui-band treeui-band-muted'>③ Preview da árvore — "
                         "imagem em largura total</div>")
        self.btn_tree_preview.layout.width = "auto"
        self.btn_tree_preview_hide.layout.width = "auto"
        card_tree_img = W.VBox([
            W.HBox([W.HTML("<div class='treeui-h' style='margin:0;flex:1'>Preview da árvore "
                           "(imagem)</div>"),
                    self.btn_tree_preview, self.btn_tree_preview_hide],
                   layout=W.Layout(align_items="center", width="100%")),
            self.out_tree_img,
        ], layout=W.Layout(width="100%")); card_tree_img.add_class("treeui-card")

        tab_build = W.VBox([sep_top, self.leaf_chips, top_cols,
                            sep_det, det_row, det_bottom, sep_img, card_tree_img])

        # ================================================================
        # ABA ③ DIAGNÓSTICO — folhas · discriminação · métricas · bootstrap · qualidade
        # ================================================================
        tbl_legend = W.HTML(
            "<div class='treeui-legend'>"
            "<b>PSI por amostra</b> (estabilidade da folha entre DES e a amostra): "
            "<span style='background:#e6f6ec;padding:1px 5px;border-radius:3px'>&lt;0.10 estável</span> "
            "<span style='background:#fdf3da;padding:1px 5px;border-radius:3px'>0.10–0.25 atenção</span> "
            "<span style='background:#fde7e7;padding:1px 5px;border-radius:3px'>&ge;0.25 instável</span>"
            "<br><b>p (irmãs)</b> = p-valor de um <b>teste de hipótese</b> que compara a "
            f"<b>distribuição do alvo ({'default' if self._is_clf else 'LGD'})</b> da folha com a da <b>irmã adjacente</b> (mesmo "
            f"pai, na amostra de referência DES). H₀: as duas irmãs têm {'a mesma PD' if self._is_clf else 'o mesmo LGD'}. "
            "O teste é o <b>Mann-Whitney U</b> (não-paramétrico, padrão) ou o <b>t de Welch</b> "
            "(médias, variâncias desiguais) — escolha no seletor <b>Teste</b>. "
            "<span style='background:#fde7e7;padding:1px 5px;border-radius:3px'>p alto (&gt;0,05, em vermelho)</span> "
            "⇒ <b>não</b> dá para distinguir as irmãs ⇒ candidatas a fusão; "
            "<span style='color:#137a3e'>p baixo</span> ⇒ folhas bem separadas. "
            "Só <b>irmãs</b> são comparadas (a última de cada grupo e o nó de faltantes ficam em branco).</div>")
        card_table = W.VBox([W.HTML("<div class='treeui-h'>Folhas criadas · PSI &amp; teste de hipótese (irmãs)</div>"),
                             tbl_legend, self.out_table,
                             W.HBox([self.btn_copy_table]), self.out_table_tsv])
        card_table.add_class("treeui-card")

        sib_legend = W.HTML(
            f"<div class='treeui-legend'>Compara o <b>{_rl} médio</b> das folhas de um mesmo "
            "pai (<b>folhas-irmãs</b>) e checa se a <b>ordem de risco</b> se mantém. "
            f"A ordem de <b>referência</b> é o {_rl} na <b>DES</b>; uma <b>inversão</b> ocorre "
            f"quando, numa amostra ou safra, uma folha de menor risco passa a ter {_rl} "
            "<i>maior</i> que uma irmã de maior risco (as linhas se cruzam). "
            f"O gráfico da esquerda mostra o {_rl} por <b>amostra</b> (DES, OOT, …) e o da "
            "direita por <b>safra</b> ao longo do tempo (faixas vermelhas = safras com "
            "inversão). O <b>indicador</b> resume: "
            "<span style='background:#e7f5ee;padding:1px 5px;border-radius:3px'>verde sem inversão</span> "
            "<span style='background:#fbf3e0;padding:1px 5px;border-radius:3px'>amarelo inverte em algumas safras</span> "
            "<span style='background:#fbe7e4;padding:1px 5px;border-radius:3px'>vermelho inverte entre amostras ou em muitas safras</span>.</div>")
        card_sib = W.VBox([
            W.HTML("<div class='treeui-h'>Folhas-irmãs · inversão entre amostras &amp; safras</div>"),
            sib_legend,
            W.HBox([self.dd_sib_group], layout=W.Layout(width="100%")),
            W.HBox([self.tx_sib_time, self.dd_sib_sample],
                   layout=W.Layout(width="100%")),
            W.HBox([self.btn_sib]),
            self.out_sib,
        ], layout=W.Layout(width="100%"))
        card_sib.add_class("treeui-card")
        self._card_sib = card_sib

        if self._is_clf:
            _dh = "Discriminação · curva ROC &amp; curva KS"
            discrim_legend = W.HTML(
                "<div class='treeui-legend'>Poder de <b>ordenação de risco</b> da régua (score = PD "
                "prevista por folha). <b>KS</b> = máxima separação entre as acumuladas de bons e "
                "maus; <b>AUC</b>/<b>Gini</b> = área sob a ROC. Avalie quando a árvore estiver "
                "fechada.</div>")
        else:
            _dh = "Dispersão do alvo por folha · boxplot &amp; histograma"
            discrim_legend = W.HTML(
                "<div class='treeui-legend'>Dispersão do <b>alvo (LGD)</b> por folha — curva ROC/KS "
                "não se aplica a alvo contínuo. <b>Boxplot por folha</b> mostra mediana, quartis e "
                "outliers; <b>histograma do alvo</b> mostra a frequência dos valores. Avalie quando "
                "a árvore estiver fechada.</div>")
        card_discrim = W.VBox([
            W.HTML(f"<div class='treeui-h'>{_dh}</div>"),
            discrim_legend,
            W.HBox([self.btn_roc, self.btn_ks]),
            self.out_discrim,
        ])
        card_discrim.add_class("treeui-card")

        if self._is_clf:
            _ml = ("a régua prediz a PD pela taxa de default do segmento na referência (DES); "
                   "avaliada como modelo em cada amostra · <b>KS</b>/<b>AUC</b>/<b>Gini</b> "
                   "altos = a segmentação ordena bem o risco · <b>Acurácia</b>/<b>F1</b> no "
                   "corte KS-ótimo")
            _mh = "Discriminação (régua como modelo de PD)"
        else:
            _ml = ("a régua prediz o LGD pela média do alvo no segmento (referência DES); "
                   "avaliada como modelo em cada amostra · <b>MAE</b>/<b>RMSE</b> menores e "
                   "<b>R²</b> maior = a régua reproduz melhor o alvo")
            _mh = "Desempenho (régua como modelo de LGD)"
        metrics_legend = W.HTML(f"<div class='treeui-legend'>{_ml}</div>")
        card_metrics = W.VBox([
            W.HTML(f"<div class='treeui-h'>{_mh}</div>"),
            metrics_legend, self.out_metrics])
        card_metrics.add_class("treeui-card")

        if self._is_clf:
            boot_legend = W.HTML(
                "<div class='treeui-legend'>IC da PD (taxa de default) por folha via bootstrap na "
                "referência (DES). Se houver OOT, mostra a PD de OOT e verifica a "
                "<b>aderência</b>: <span style='color:#137a3e'>dentro</span> do IC = estável; "
                "<span style='color:#b3261e'>acima/abaixo</span> = PD deslocou além da incerteza "
                "amostral. Calcule quando a árvore estiver fechada.</div>")
        else:
            boot_legend = W.HTML(
                "<div class='treeui-legend'>IC do alvo (LGD) por folha via bootstrap na "
                "referência (DES). Se houver OOT, mostra o LGD de OOT e verifica a "
                "<b>aderência</b>: <span style='color:#137a3e'>dentro</span> do IC = estável; "
                "<span style='color:#b3261e'>acima/abaixo</span> = LGD deslocou além da incerteza "
                "amostral. Calcule quando a árvore estiver fechada.</div>")
        card_boot = W.VBox([
            W.HTML("<div class='treeui-h'>Intervalos de confiança (bootstrap) &amp; aderência OOT</div>"),
            boot_legend,
            W.HBox([self.sl_boot, self.btn_boot],
                   layout=W.Layout(align_items="center")),
            self.out_boot])
        card_boot.add_class("treeui-card")

        # ---- PLACAR DE SAÚDE DO MODELO (visão estatística de relance) -------
        sep_diag = W.HTML("<div class='treeui-band'>Placar de saúde do modelo · "
                          "discriminação · estabilidade · calibração · estrutura</div>")
        self.btn_diag.layout.width = "auto"
        self.btn_diag_hide.layout.width = "auto"
        _diag_metrics = "AUC/Gini/KS" if self._is_clf else "MAE/RMSE/R²"
        card_score = W.VBox([
            W.HTML("<div class='treeui-legend'>Veredito de relance em 4 dimensões "
                   "(verde/amarelo/vermelho) reunindo os testes das outras abas — " + _diag_metrics + ", "
                   "PSI/CSI, calibração prevista×observada e monotonicidade · distinção entre "
                   "folhas-irmãs — com a evidência logo abaixo. Clique para (re)calcular.</div>"),
            W.HBox([self.btn_diag, self.btn_diag_hide], layout=W.Layout(gap="6px")),
            self.out_diag,
        ], layout=W.Layout(width="100%"))
        card_score.add_class("treeui-card")
        _diag_detail = ("discriminação (ROC/KS)" if self._is_clf
                        else "dispersão do alvo (boxplot/histograma)")
        sep_diag2 = W.HTML("<div class='treeui-band treeui-band-muted'>Evidência detalhada · "
                           f"folhas · {_diag_detail} · métricas · IC bootstrap</div>")
        tab_diag = W.VBox([sep_diag, card_score, sep_diag2,
                           card_metrics, card_table, card_sib, card_discrim,
                           card_boot])

        # ================================================================
        # ABA ④ VALIDAR & EXPORTAR — duas faixas: validação · exportar/registrar
        # ================================================================
        sep_val = W.HTML("<div class='treeui-band'>① Validação regulatória · "
                         "monotonicidade · calibração · backtest</div>")
        valid_legend = W.HTML(
            f"<div class='treeui-legend'>Roda as três checagens: <b>monotonicidade</b> do {_rl} nas "
            "notas (DES e demais amostras), <b>calibração</b> prevista (DES) × realizada (OOT) por "
            f"folha, e <b>backtest</b> do {_rl} previsto × realizado por safra (informe a coluna de "
            "tempo). O <b>relatório</b> reúne tudo num Markdown com as imagens.</div>")
        card_validacao = W.VBox([
            W.HTML("<div class='treeui-h'>Rodar validação</div>"),
            valid_legend,
            W.HBox([self.tx_time_col, self.btn_validate],
                   layout=W.Layout(align_items="center")),
            self.out_validate,
            W.HTML("<div class='treeui-h' style='margin-top:10px'>Relatório de validação (Markdown)</div>"),
            W.HBox([self.tx_report_path, self.btn_report],
                   layout=W.Layout(align_items="center")),
        ], layout=W.Layout(width="100%"))
        card_validacao.add_class("treeui-card")
        self._card_validacao = card_validacao

        sep_exp = W.HTML("<div class='treeui-band treeui-band-muted'>② Exportar &amp; registrar</div>")
        card_export_df = W.VBox([
            W.HTML("<div class='treeui-h'>Exportar DataFrame rotulado</div>"),
            W.HTML("<div class='treeui-legend'>Gera <b>ui.result</b> (pandas) com a coluna de "
                   "segmento e a nota (folha) por linha.</div>"),
            W.HBox([self.btn_export]),
        ], layout=W.Layout(width="100%"))
        card_export_df.add_class("treeui-card")
        card_mlflow = W.VBox([
            W.HTML("<div class='treeui-h'>Registrar no MLflow / Unity Catalog</div>"),
            W.HTML("<div class='treeui-legend'>Loga régua, métricas e o modelo pyfunc e registra a "
                   "versão no Model Registry.</div>"),
            self.tx_model, self.cb_uc, self.tx_experiment, self.tx_runname,
            W.Box([], layout=W.Layout(flex="1 1 auto")),   # espaçador: empurra o botão p/ a base
            W.HBox([self.btn_mlflow]),
        ], layout=W.Layout(width="49%"))
        card_mlflow.add_class("treeui-card")
        card_spark = W.VBox([
            W.HTML("<div class='treeui-h'>Reconstruir folhas em tabela Spark</div>"),
            W.HTML("<div class='treeui-legend'>Aplica a régua a uma tabela Spark (segmento, nota e "
                   "valor por linha), gravando opcionalmente o resultado.</div>"),
            self.tx_spark_in, self.tx_spark_out,
            W.Box([], layout=W.Layout(flex="1 1 auto")),   # alinha "Reconstruir folhas" com "Salvar no MLflow"
            W.HBox([self.btn_spark_apply]),
        ], layout=W.Layout(width="49%"))
        card_spark.add_class("treeui-card")
        export_row = W.HBox([card_mlflow, card_spark],
                            layout=W.Layout(width="100%", align_items="stretch",
                                            justify_content="space-between"))
        tab_valid = W.VBox([sep_val, card_validacao, sep_exp, card_export_df, export_row])

        # ================================================================
        # ABA ⑤ HISTÓRICO — persistência (JSON) · imagem da árvore (lado a lado)
        # ================================================================
        sep_hist = W.HTML("<div class='treeui-band'>Histórico &amp; persistência</div>")
        card_json = W.VBox([
            W.HTML("<div class='treeui-h'>Salvar / carregar árvore (JSON)</div>"),
            W.HTML("<div class='treeui-legend'>Salva a estrutura completa (regras e folhas "
                   "fechadas) num .json e recarrega depois. Para o passo a passo, use "
                   "◀ Desfazer / Refazer ▶ na aba <b>Construir</b>.</div>"),
            self.tx_json_path,
            W.HBox([self.btn_save_json, self.btn_load_json]),
        ], layout=W.Layout(width="49%"))
        card_json.add_class("treeui-card")
        card_img = W.VBox([
            W.HTML(f"<div class='treeui-h'>Imagem da árvore ({_rl} médio &amp; % por folha)</div>"),
            self.tx_img_path,
            W.HBox([self.btn_plot, self.btn_plot_hide]),
            self.out_plot,
        ], layout=W.Layout(width="49%"))
        card_img.add_class("treeui-card")
        hist_row = W.HBox([card_json, card_img],
                          layout=W.Layout(width="100%", align_items="stretch",
                                          justify_content="space-between"))
        card_pdf = W.VBox([
            W.HTML("<div class='treeui-h'>Relatório do modelo (PDF)</div>"),
            W.HTML("<div class='treeui-legend'>Gera um PDF com capa (parâmetros), métricas por "
                   "amostra, imagem da árvore, folhas e calibração — salvo no caminho informado.</div>"),
            self.tx_pdf_path,
            W.HBox([self.btn_pdf]),
            self.out_pdf,
        ])
        card_pdf.add_class("treeui-card")
        tab_hist = W.VBox([sep_hist, hist_row, card_pdf])

        # ================================================================
        # ABA ② ANÁLISE DE VARIÁVEL — perfil, distribuição e estabilidade
        # ================================================================
        self.dd_var.layout = W.Layout(width="30%")
        self.dd_var.style.description_width = "62px"
        self.dd_var_leaf.layout = W.Layout(width="42%")
        self.dd_var_leaf.style.description_width = "46px"
        self.tx_var_time.layout = W.Layout(width="22%")
        self.btn_var_analyze.layout = W.Layout(width="auto")
        var_controls = W.VBox([
            W.HTML("<div class='treeui-h'>Análise de variáveis</div>"),
            W.HTML("<div class='treeui-legend'>Perfil de uma variável de entrada numa folha: "
                   "distribuição, %missing, média/mediana/desvio, faixa de percentis, PSI atual "
                   "e o comportamento por safra (percentis e PSI). Informe a <b>coluna de "
                   "safra</b> (ex.: dt_ref) para as análises temporais.</div>"),
            W.HBox([self.dd_var, self.dd_var_leaf, self.tx_var_time, self.btn_var_analyze],
                   layout=W.Layout(align_items="flex-end", justify_content="space-between",
                                   width="100%")),
        ])
        var_controls.add_class("treeui-card")
        card_var_dist = W.VBox([
            W.HTML("<div class='treeui-h'>Comportamento da variável · distribuição</div>"),
            self.out_var_dist], layout=W.Layout(width="52%"))
        card_var_dist.add_class("treeui-card")
        card_var_cards = W.VBox([
            W.HTML("<div class='treeui-h'>Resumo &amp; estabilidade</div>"),
            self.out_var_cards], layout=W.Layout(width="46%"))
        card_var_cards.add_class("treeui-card")
        var_row_a = W.HBox([card_var_dist, card_var_cards],
                           layout=W.Layout(justify_content="space-between",
                                           align_items="stretch", width="100%"))
        card_var_time = W.VBox([
            W.HTML("<div class='treeui-h'>Comportamento ao longo do tempo · por safra</div>"),
            W.HTML("<div class='treeui-legend'>Numérica: percentis (min–max, p5–p95, média) por "
                   "safra. Categórica: representatividade (%) de cada categoria por safra.</div>"),
            self.out_var_time])
        card_var_time.add_class("treeui-card")
        card_var_table = W.VBox([
            W.HTML("<div class='treeui-h'>Detalhe por safra</div>"),
            self.out_var_table], layout=W.Layout(width="49%"))
        card_var_table.add_class("treeui-card")
        card_var_psi = W.VBox([
            W.HTML("<div class='treeui-h'>PSI por safra · vs. data de referência (DES)</div>"),
            self.out_var_psi], layout=W.Layout(width="49%"))
        card_var_psi.add_class("treeui-card")
        var_row_b = W.HBox([card_var_table, card_var_psi],
                           layout=W.Layout(justify_content="space-between",
                                           align_items="stretch", width="100%"))
        tab_var = W.VBox([var_controls, var_row_a, card_var_time, var_row_b])

        # ---- ABA AVANÇADO: sugerir splits · auto-merge · importância · SQL · diff ----
        card_sug = W.VBox([
            W.HTML("<div class='treeui-h'>Sugerir splits (TOP 3)</div>"),
            W.HTML("<div class='treeui-legend'>As 3 variáveis de maior IV para a <b>folha "
                   "selecionada</b> (aba Construir), com nº de bins, PSI por amostra (OOT/"
                   "ESTABILIDADE), se a separação de risco passa no teste de hipótese e o IV.</div>"),
            self.btn_suggest3, self.out_suggest]); card_sug.add_class("treeui-card")
        card_merge = W.VBox([
            W.HTML("<div class='treeui-h'>Auto-merge de folhas semelhantes</div>"),
            W.HTML("<div class='treeui-legend'>Funde folhas-irmãs com risco estatisticamente "
                   "<b>indistinguível</b> (p &gt; α no teste entre adjacentes).</div>"),
            self.sl_alpha, self.cb_automerge_na, self.btn_automerge]); card_merge.add_class("treeui-card")
        imp_row = W.HBox(
            [W.VBox([self.out_importance], layout=W.Layout(width="49%")),
             W.VBox([self.out_importance_chart], layout=W.Layout(width="49%"))],
            layout=W.Layout(width="100%", justify_content="space-between",
                            align_items="flex-start"))
        card_imp = W.VBox([
            W.HTML("<div class='treeui-h'>Importância das variáveis (na árvore)</div>"),
            W.HTML("<div class='treeui-legend'>Ganho de IV ponderado pela representatividade do nó, "
                   "somado por variável que <b>entrou</b> na árvore.</div>"),
            self.btn_importance, imp_row, self.out_importance_legend])
        card_imp.add_class("treeui-card")
        card_sql = W.VBox([
            W.HTML("<div class='treeui-h'>Exportar como SQL (CASE WHEN)</div>"),
            W.HTML("<div class='treeui-legend'>Régua pronta para copiar e colar. Ajuste o nome da "
                   "tabela de origem.</div>"),
            W.HBox([self.tx_sql_table, self.btn_sql]), self.out_sql]); card_sql.add_class("treeui-card")
        card_diff = W.VBox([
            W.HTML("<div class='treeui-h'>Comparar duas árvores (versões)</div>"),
            W.HTML("<div class='treeui-legend'>Carrega outra árvore salva em JSON e compara com a "
                   "atual: migração de notas, concordância e métricas lado a lado.</div>"),
            W.HBox([self.tx_diff_path, self.btn_diff]), self.out_diff]); card_diff.add_class("treeui-card")
        tab_avancado = W.VBox([
            W.HBox([card_sug, card_merge],
                   layout=W.Layout(justify_content="space-between", width="100%")),
            card_imp, card_sql, card_diff])

        # ---- montagem das abas (Análise de variável vem em 2º) ----------
        tabs = W.Tab(children=[tab_build, tab_var, tab_diag, tab_valid, tab_avancado, tab_hist])
        for i, titulo in enumerate(["Construir", "Análise de variáveis", "Diagnóstico",
                                    "Validar & Exportar", "Avançado", "Histórico"]):
            tabs.set_title(i, titulo)
        tabs.add_class("treeui-tabs")
        # a tabela de IV (optbinning de TODAS as variáveis na folha) é o item mais
        # caro do open/refresh e fica na aba 1 (não-visível por padrão). Adiamos seu
        # cálculo até a aba ser realmente aberta (render preguiçoso) — ver _refresh_iv.
        self.tabs = tabs
        self._iv_tab_index = 1
        tabs.observe(self._on_tab_change, names="selected_index")

        # ---- console persistente (log de todas as abas) -----------------
        self.btn_clear_log.layout.width = "150px"
        console = W.VBox([
            W.HBox([W.HTML("<div class='treeui-h' style='margin-bottom:0'>"
                           "Console · mensagens das ações</div>"),
                    self.btn_clear_log],
                   layout=W.Layout(justify_content="space-between", align_items="center")),
            self.out_log,
        ])
        console.add_class("treeui-card")

        self.cb_dark = W.ToggleButton(value=False, description="🌙 Tema escuro",
                                      tooltip="Alterna o tema claro/escuro da interface",
                                      layout=W.Layout(width="150px"))
        self.cb_dark.observe(self._on_dark, names="value")
        topbar = W.HBox([self.cb_dark], layout=W.Layout(justify_content="flex-end"))
        self.panel = W.VBox([topbar, banner, bar_box, tabs, console, self.tree_sel_style])
        self.panel.add_class("treeui")

    def _on_dark(self, change):
        if change["new"]:
            self.panel.add_class("dark")
            self.cb_dark.description = "☀ Tema claro"
        else:
            self.panel.remove_class("dark")
            self.cb_dark.description = "🌙 Tema escuro"

    # ==================================================================
    # Render
    # ==================================================================
    @staticmethod
    def _color(pdv, lo, hi):
        if hi <= lo or pd.isna(pdv):
            t = 0.5
        else:
            t = max(0.0, min(1.0, (pdv - lo) / (hi - lo)))
        r = int(40 + (214 - 40) * min(1, 2 * t))
        g = int(166 - (166 - 69) * max(0, 2 * t - 1)) if t > 0.5 else 166
        return f"rgb({r},{g},69)"

    def _node_value(self, sid, sample=None):
        # lê só a coluna-alvo (não materializa o subframe inteiro) — chamado por nó
        # × amostra em _tree_html, que percorre a árvore inteira a cada render.
        m = self.seg.segments[sid]["mask"]
        if sample is not None and sample in self._sample_masks:
            m = m & self._sample_masks[sample]
        sr = self.df[self.target][m]
        return sr.mean() if len(sr) else float("nan")

    def _leaf_values(self):
        ref = self.ref_sample if self.sample_col is not None else None
        vals = [self._node_value(sid, ref)
                for sid, s in self.seg.segments.items() if s["is_leaf"]]
        vals = [v for v in vals if not pd.isna(v)]
        return (min(vals), max(vals)) if vals else (0.0, 1.0)

    @staticmethod
    def _psi_class(p):
        return "green" if p < 0.10 else "yellow" if p < 0.25 else "red"

    def _sample_value_test(self, sid, a, b, min_n=8):
        """Teste de hipótese comparando a PD (taxa de default) da MESMA folha entre
        as amostras `a` (ex.: DES) e `b` (ex.: OOT) — aderência da PD entre amostras.
        Usa o teste do seletor (Mann-Whitney ou Welch t).
        Retorna (nome_exibido, p_valor, n_a, n_b)."""
        name = "Welch t" if self.dd_test.value == "welch" else "Mann-Whitney"
        sm = self._sample_masks
        if a not in sm or b not in sm:
            return name, float("nan"), 0, 0
        leaf = self.seg.segments[sid]["mask"]
        va = self.df.loc[leaf & sm[a], self.target].dropna().to_numpy()
        vb = self.df.loc[leaf & sm[b], self.target].dropna().to_numpy()
        if len(va) < min_n or len(vb) < min_n:
            return name, float("nan"), len(va), len(vb)
        try:
            from scipy.stats import mannwhitneyu, ttest_ind
            if self.dd_test.value == "welch":
                p = float(ttest_ind(va, vb, equal_var=False).pvalue)
            else:
                p = float(mannwhitneyu(va, vb, alternative="two-sided").pvalue)
        except Exception:
            p = float("nan")
        return name, p, len(va), len(vb)

    def _sibling_adjacent_tests(self, sid):
        """Teste de PD (na amostra DES) entre a folha e cada IRMÃ ADJACENTE de
        MESMO PAI — indica se a folha é estatisticamente distinta da vizinha.
        Irmãs ordenadas por corte (num) ou por PD média (cat); o nó de faltantes
        (na) não entra. Retorna (nome_do_teste, [(lado, descrição_irmã, p_valor)])."""
        seg = self.seg
        name = "Welch t" if self.dd_test.value == "welch" else "Mann-Whitney"
        s = seg.segments.get(sid)
        if s is None or s["parent"] is None or not s["is_leaf"]:
            return name, []
        irmaos = [i for i, v in seg.segments.items()
                  if v["parent"] == s["parent"] and v["is_leaf"]
                  and v["conditions"] and v["conditions"][-1]["kind"] != "na"]
        if sid not in irmaos or len(irmaos) < 2:
            return name, []
        if all(seg.segments[c]["conditions"][-1]["kind"] == "num" for c in irmaos):
            irmaos.sort(key=lambda c: seg.segments[c]["conditions"][-1]["lo"])
        else:
            irmaos.sort(key=lambda c: (seg._leaf_target(c).mean()
                                       if len(seg._leaf_target(c)) else float("inf")))
        i = irmaos.index(sid)
        out = []
        for lado, j in (("◀", i - 1), ("▶", i + 1)):
            if 0 <= j < len(irmaos):
                nb = irmaos[j]
                p = seg._pair_pvalue(sid, nb, test=self.dd_test.value)
                desc = seg._descrever([seg.segments[nb]["conditions"][-1]])
                out.append((lado, desc, p))
        return name, out

    def _leaf_chips_html(self):
        """Resumo curto da folha ativa para a régua do topo (nº da folha, rótulo,
        PD DES, volumetria e repr.) — o detalhe completo fica na faixa de baixo."""
        sid = self.dd_leaf.value
        if sid is None or sid not in self.seg.segments:
            return ("<div class='treeui-chips'><span class='lab'>Nenhuma folha "
                    "selecionada</span></div>")
        s = self.seg.segments[sid]
        nota_map, _ = self.seg._grade_map()
        nota = nota_map.get(sid, "?")
        n = int(s["mask"].sum())
        rep = 100 * n / len(self.df) if len(self.df) else 0.0
        ref = self.ref_sample if self.sample_col is not None else None
        pdv = self._node_value(sid, ref)
        pd_txt = "—" if pd.isna(pdv) else f"{pdv * 100:.2f}%"
        label = ("TODA A CARTEIRA" if s["parent"] is None
                 else self.seg._descrever(s["conditions"]))
        if len(label) > 46:
            label = label[:43] + "…"
        vol = f"{n:,}".replace(",", ".")
        lock = " 🔒" if sid in self.locked else ""
        return (
            "<div class='treeui-chips'><span class='lab'>folha ativa</span>"
            f"<span class='chip'><b>#{nota}</b> · {label}{lock}</span>"
            f"<span class='chip'>{self._risk_label} {pd_txt}</span>"
            f"<span class='chip'>vol {vol}</span>"
            f"<span class='chip'>repr. {rep:.1f}%</span></div>")

    def _status_html(self):
        """Health strip (estilo mockup): células com rótulo maiúsculo, número
        grande (mono) e, quando aplicável, um badge de status."""
        seg = self.seg
        n_folhas = sum(s["is_leaf"] for s in seg.segments.values())
        prof = max(s["depth"] for s in seg.segments.values())
        n_lock = len(self.locked & {sid for sid, s in seg.segments.items() if s["is_leaf"]})
        hexc = {"green": "#157a52", "yellow": "#9a6f12", "red": "#b23a2a"}
        bgc = {"green": "#e7f5ee", "yellow": "#fbf3e0", "red": "#fbe7e4"}

        def cell(label, value, color="#1f2733", badge=None, cls=None):
            bh = ""
            if badge and cls:
                bh = (f"<span style='font-size:10px;font-weight:600;color:{hexc[cls]};"
                      f"background:{bgc[cls]};border-radius:20px;padding:2px 8px;"
                      f"margin-left:7px'>{badge}</span>")
            return (f"<div style='flex:1;min-width:86px;padding:8px 14px;"
                    f"border-right:1px solid #eef0f3'>"
                    f"<div style='font-size:10px;font-weight:600;letter-spacing:.07em;"
                    f"text-transform:uppercase;color:#8a93a3;white-space:nowrap'>{label}</div>"
                    f"<div style='display:flex;align-items:center;margin-top:2px'>"
                    f"<span class='mono' style='font-size:19px;font-weight:600;color:{color}'>"
                    f"{value}</span>{bh}</div></div>")
        cells = [cell("Folhas", n_folhas), cell("Profundidade", prof),
                 cell("Fechadas", n_lock)]
        if self.sample_col is not None and n_folhas >= 1:
            try:
                for _, r in seg.psi().iterrows():
                    c = self._psi_class(r["psi"])
                    cells.append(cell(f"PSI {r['amostra']}", f"{r['psi']:.3f}",
                                      badge=r["classificacao"], cls=c))
            except Exception:
                pass
        # discriminação ao vivo: KS/AUC (classificação) ou R² (regressão)
        try:
            for _, r in seg.metrics().iterrows():
                if self._is_clf:
                    ks, auc = r["KS"], r["AUC"]
                    if pd.isna(ks):
                        cells.append(cell(f"KS {r['amostra']}", "—", color="#8a93a3"))
                    else:
                        c = "green" if ks >= 0.30 else "yellow" if ks >= 0.20 else "red"
                        badge = "bom" if c == "green" else "atenção" if c == "yellow" else "fraco"
                        cells.append(cell(f"KS {r['amostra']}", f"{ks:.3f}", color=hexc[c],
                                          badge=badge, cls=c))
                    if pd.isna(auc):
                        cells.append(cell(f"AUC {r['amostra']}", "—", color="#8a93a3"))
                    else:
                        c = "green" if auc >= 0.70 else "yellow" if auc >= 0.60 else "red"
                        cells.append(cell(f"AUC {r['amostra']}", f"{auc:.3f}", color=hexc[c]))
                else:
                    r2 = r["R2"]
                    if pd.isna(r2):
                        cells.append(cell(f"R² {r['amostra']}", "—", color="#8a93a3"))
                    else:
                        c = "green" if r2 >= 0.5 else "yellow" if r2 >= 0.2 else "red"
                        cells.append(cell(f"R² {r['amostra']}", f"{r2:.3f}", color=hexc[c]))
        except Exception:
            pass
        return f"<div style='display:flex;align-items:stretch'>{''.join(cells)}</div>"

    def _min_nota_fn(self, filhos, nota_map):
        """min_nota(sid) = menor nota do ramo — ordena os filhos esquerda→direita
        de forma consistente com a numeração (nota = posição na árvore)."""
        cache: dict = {}

        def min_nota(sid):
            if sid not in cache:
                if self.seg.segments[sid]["is_leaf"]:
                    cache[sid] = nota_map.get(sid, 10 ** 9)
                else:
                    cache[sid] = min((min_nota(c) for c in filhos.get(sid, [])),
                                     default=10 ** 9)
            return cache[sid]

        return min_nota

    def _tree_html(self):
        seg = self.seg
        filhos: dict = {}
        for sid, s in seg.segments.items():
            filhos.setdefault(s["parent"], []).append(sid)
        nota_map, _ = seg._grade_map()
        min_nota = self._min_nota_fn(filhos, nota_map)
        lo, hi = self._leaf_values()
        n_total = len(self.df)
        rows = []

        def stat(sid):
            # mask.sum() em vez de materializar self.df[mask] só para contar linhas
            n = int(seg.segments[sid]["mask"].sum())
            return n, 100 * n / n_total

        def value_str(sid):
            if self.sample_col is not None:
                parts = [f"{self.ref_sample} {self._node_value(sid, self.ref_sample) * 100:.2f}%"]
                for a in self._tree_nonref:          # só amostras COM alvo (sem ESTABILIDADE)
                    parts.append(f"{a} {self._node_value(sid, a) * 100:.2f}%")
                return self._risk_label + " " + " ".join(parts)
            return f"{self._risk_label} {self._node_value(sid) * 100:.2f}%"

        psi_hex = {"green": "#1aa64b", "yellow": "#caa000", "red": "#d6453e"}
        # "barrinha" vertical que separa o bloco PD do bloco PSI na linha da folha
        sep_bar = ("<span style='display:inline-block;width:0;border-left:1px solid "
                   "#aab4be;height:11px;margin:0 8px;vertical-align:middle'></span>")

        def psi_str(sid):
            if self.sample_col is None or not self._nonref:
                return ""
            parts = []
            for a in self._nonref:
                p = self._leaf_psi(sid, a)
                if pd.isna(p):
                    continue
                ab = "ESTAB" if a == "ESTABILIDADE" else a
                parts.append(f"<span style='color:{psi_hex[self._psi_class(p)]}'>"
                             f"{ab} {p:.2f}</span>")
            return (sep_bar + "PSI " + " ".join(parts)) if parts else ""

        def rotulo(sid):
            s = seg.segments[sid]
            return "TODA A CARTEIRA" if s["parent"] is None else seg._descrever([s["conditions"][-1]])

        mono = "white-space:pre;font-family:ui-monospace,Menlo,monospace"

        def rec(sid, prefix, is_last, is_root):
            n, rep = stat(sid)
            s = seg.segments[sid]
            conn = "" if is_root else ("└─ " if is_last else "├─ ")
            ref = self.ref_sample if self.sample_col is not None else None
            color = self._color(self._node_value(sid, ref), lo, hi)
            sw = (f"<span style='display:inline-block;width:11px;height:11px;background:{color};"
                  f"border-radius:2px;vertical-align:middle;margin:0 5px'></span>")
            # HTML INDEPENDENTE DA SELEÇÃO: o realce da folha ativa é aplicado por
            # CSS (ver _leaf_highlight_style), via o atributo data-leaf=<nota>. Assim
            # trocar de folha não remonta nem reenvia a árvore inteira.
            tags = ""
            sel_marker = ""
            if s["is_leaf"]:
                tags += f" · <b>folha {nota_map.get(sid, '?')}</b>"
                if sid in self.locked:
                    tags += " 🔒"
                sel_marker = "<i class='tsel'></i>"   # ::after injeta '◀ selecionada'
            # continuação do prefixo (mantém os traços verticais alinhados na 2ª linha)
            cont = "" if is_root else prefix + ("   " if is_last else "│  ")
            psi_html = psi_str(sid) if s["is_leaf"] else ""
            # linha 1 — rótulo (condição do nó) + nº da folha
            linha1 = (f"<div style='{mono};font-size:12px;padding:1px 2px 0'>"
                      f"{prefix}{conn}{sw}<b class='tlname' style='color:#15324a'>"
                      f"{rotulo(sid)}</b>{tags}{sel_marker}</div>")
            # linha 2 — métricas EMBAIXO: volumetria, representatividade, PD e PSI
            vol = f"{n:,}".replace(",", ".")        # separador de milhar pt-BR
            linha2 = (f"<div style='{mono};font-size:11px;color:#7c8893;padding:0 2px 3px'>"
                      f"{cont}    vol {vol} · repr. {rep:.1f}%{sep_bar}{value_str(sid)}{psi_html}</div>")
            data_attr = f' data-leaf="{nota_map.get(sid)}"' if s["is_leaf"] else ""
            rows.append(f"<div class='tnode'{data_attr}>{linha1}{linha2}</div>")
            ch = sorted(filhos.get(sid, []), key=min_nota)
            for i, c in enumerate(ch):
                child_prefix = "" if is_root else prefix + ("   " if is_last else "│  ")
                rec(c, child_prefix, i == len(ch) - 1, False)

        rec("root", "", True, True)
        return "<div class='treeui-tree'>" + "".join(rows) + "</div>"

    def _leaf_highlight_style(self):
        """Bloco <style> que realça a folha selecionada (por data-leaf=<nota>):
        fundo âmbar, nome laranja e o marcador '◀ selecionada' via ::after. É o
        único blob que muda ao trocar de folha (vs. remontar a árvore inteira)."""
        sid = self.dd_leaf.value
        if sid is None or sid not in self.seg.segments:
            return ""
        nota_map, _ = self.seg._grade_map()
        n = nota_map.get(sid)
        if n is None:
            return ""
        return (
            "<style>"
            f'.tnode[data-leaf="{n}"]{{background:#fff5e6;border-radius:5px;'
            "box-shadow:inset 3px 0 0 #e8870b;}"
            f'.tnode[data-leaf="{n}"] .tlname{{color:#e8870b !important;}}'
            f'.tnode[data-leaf="{n}"] .tsel::after{{content:" ◀ selecionada";'
            "color:#e8870b;font-weight:700;}"
            "</style>")

    def _style_leaves(self, lv):
        psi_cols = [c for c in lv.columns if c.startswith("psi_")]

        def psi_bg(v):
            if pd.isna(v):
                return ""
            a = abs(v)
            c = "#e6f6ec" if a < 0.10 else "#fdf3da" if a < 0.25 else "#fde7e7"
            return f"background-color:{c}"

        def p_bg(v):
            if pd.isna(v):
                return "color:#aab"
            return "background-color:#fde7e7;font-weight:600" if v > 0.05 else "color:#137a3e"

        sty = lv.style
        for c in psi_cols:
            sty = sty.map(psi_bg, subset=[c])
        if "p_vs_prox" in lv.columns:
            sty = sty.map(p_bg, subset=["p_vs_prox"])
        fmt = {"repr_%": "{:.1f}"}
        for c in lv.columns:
            if c.startswith("repr_") and c.endswith("_%"):   # % por amostra
                fmt[c] = "{:.1f}"
            elif c.startswith("pd_"):       # PD em % (coerente com a árvore)
                fmt[c] = "{:.2%}"
            elif c.startswith("psi_"):      # PSI é adimensional → decimal
                fmt[c] = "{:.4f}"
        if "p_vs_prox" in lv.columns:
            fmt["p_vs_prox"] = "{:.3f}"
        sty = (sty.format(fmt, na_rep="—")
                  .hide(axis="index")
                  .set_table_styles(self._TABLE_STYLES)
                  .set_properties(**{"font-size": "12px"})
                  # texto centralizado em toda a tabela (cabeçalho e células)
                  .set_table_styles([{"selector": "th, td",
                                      "props": [("text-align", "center")]}],
                                    overwrite=False))
        return sty

    def _leaf_label(self, sid):
        s = self.seg.segments[sid]
        txt = "TODA A CARTEIRA" if s["parent"] is None else self.seg._descrever(s["conditions"])
        if len(txt) > 72:
            txt = txt[:69] + "…"
        return ("🔒 " if sid in self.locked else "") + txt

    def _leaf_header_html(self):
        """Cartão-resumo da folha selecionada: volumetria e representatividade por
        amostra (DES, OOT, ESTAB…); PD média de DES e das demais amostras com o
        incremento de cada uma vs DES; teste de aderência DES→amostra (nome +
        p-valor); distinção vs folha-irmã; e estabilidade (PSI por amostra) com
        barrinha verde/amarelo/vermelho."""
        sid = self.dd_leaf.value
        if sid is None or sid not in self.seg.segments:
            return ("<div style='font-size:12px;color:#889'>Nenhuma folha selecionada — "
                    "crie um split ou rode o Auto-fit na coluna do centro.</div>")
        s = self.seg.segments[sid]
        leaf = s["mask"]
        n = int(leaf.sum())
        lo, hi = self._leaf_values()
        ref = self.ref_sample if self.sample_col is not None else None
        color = self._color(self._node_value(sid, ref), lo, hi)
        nota_map, _ = self.seg._grade_map()
        nota = nota_map.get(sid, "?")
        label = ("TODA A CARTEIRA" if s["parent"] is None
                 else self.seg._descrever(s["conditions"]))
        if len(label) > 80:
            label = label[:77] + "…"
        badge = ("<span class='pill pill-yellow'>folha fechada 🔒</span>"
                 if sid in self.locked
                 else "<span class='pill pill-green'>folha aberta</span>")
        # selo: esta folha recebe os faltantes (NaN) no scoring (include_na) —
        # atribuição conservadora à folha-irmã de pior risco quando o split não
        # gerou nó de faltantes próprio
        na_badge = ("<span class='pill' style='background:#fbe7e4;color:#b23a2a' "
                    "title='Recebe os faltantes (NaN) no scoring — atribuição "
                    "conservadora à folha-irmã de pior risco'>+ faltantes</span>"
                    if any(c.get("include_na") for c in s["conditions"]) else "")

        def ab(a):
            return "ESTAB" if a == "ESTABILIDADE" else a

        def chip(k, v, c=None, sub=None):
            sty = f" style='color:{c}'" if c else ""
            sub_html = (f"<div style='font-size:10.5px;margin-top:1px;white-space:nowrap;"
                        f"color:#8a93a3'>{sub}</div>") if sub else ""
            return (f"<div class='treeui-metric'><div class='k'>{k}</div>"
                    f"<div class='v mono'{sty}>{v}</div>{sub_html}</div>")

        head = (
            "<div style='display:flex;align-items:center;gap:9px;margin-bottom:4px;"
            "flex-wrap:wrap'>"
            f"<span style='width:13px;height:13px;border-radius:4px;background:{color};"
            "flex:none'></span>"
            f"<span style='font-size:15px;font-weight:600;color:#15324a'>{label}</span>"
            f"{badge}{na_badge}<span class='pill pill-muted'>folha {nota}</span></div>")

        sec_h = ("<div class='treeui-h' style='margin-top:11px'>{}</div>").format

        if self.sample_col is None:
            rep = 100 * n / len(self.df) if len(self.df) else 0.0
            cells = (chip("Volumetria", f"{n:,}".replace(",", "."))
                     + chip("Repr.", f"{rep:.1f}%")
                     + chip(self._risk_label, f"{self._node_value(sid) * 100:.2f}%"))
            return head + f"<div class='treeui-metrics'>{cells}</div>"

        sm = self._sample_masks
        ordered_nonref = list(self._pd_nonref) + list(self._psi_only)
        samples_all = [self.ref_sample] + ordered_nonref

        # 1) Volumetria & representatividade da folha por amostra
        sec1 = chip("Volumetria", f"{n:,}".replace(",", "."))
        for a in samples_all:
            m = sm.get(a)
            tot = int(m.sum()) if m is not None else 0
            rp = (100 * int((leaf & m).sum()) / tot) if tot else float("nan")
            sec1 += chip(f"Repr. {ab(a)}", "—" if pd.isna(rp) else f"{rp:.1f}%")

        # 2) PD média (DES e demais) + incremento de cada amostra vs DES
        pd_ref = self._node_value(sid, self.ref_sample)
        sec2 = chip(f"{self._risk_label} {self.ref_sample}",
                    "—" if pd.isna(pd_ref) else f"{pd_ref * 100:.2f}%", sub="referência")
        for a in self._pd_nonref:
            v = self._node_value(sid, a)
            if pd.isna(v) or pd.isna(pd_ref):
                sec2 += chip(f"{self._risk_label} {ab(a)}", "—" if pd.isna(v) else f"{v * 100:.2f}%")
                continue
            d = (v - pd_ref) * 100      # incremento em pontos percentuais
            sig = "+" if d >= 0 else "−"
            dcol = "#b3261e" if d > 0 else "#137a3e"   # PD subindo = pior (vermelho)
            sub = f"<span style='color:{dcol}'>Δ vs DES {sig}{abs(d):.2f} p.p.</span>"
            sec2 += chip(f"{self._risk_label} {ab(a)}", f"{v * 100:.2f}%", sub=sub)

        # 3) Aderência DES → amostra (teste de hipótese: nome + p-valor)
        test_rows = ""
        for a in self._pd_nonref:
            name, p, na, nb = self._sample_value_test(sid, self.ref_sample, a)
            if pd.isna(p):
                pv, verdict = "—", "<span class='pill pill-muted'>amostra insuficiente</span>"
            else:
                pv = f"{p:.4f}"
                verdict = ("<span class='pill pill-green'>aderente · não rejeita H₀ "
                           "(p&gt;0,05)</span>" if p > 0.05 else
                           "<span class='pill pill-red'>não aderente · rejeita H₀ "
                           "(p≤0,05)</span>")
            test_rows += (
                "<div style='display:flex;align-items:center;gap:7px;flex-wrap:wrap;"
                "font-size:12px;color:#3a4250;margin:3px 0'>"
                f"<b>DES → {ab(a)}</b>"
                f"<span style='color:#6b7480'>teste:</span><b>{name}</b>"
                f"<span style='color:#6b7480'>p-valor:</span>"
                f"<b class='mono'>{pv}</b>{verdict}</div>")

        # 4) Distinção vs folha-irmã adjacente (mesmo pai)
        sib_name, sib_tests = self._sibling_adjacent_tests(sid)
        sib_rows = ""
        for lado, desc, p in sib_tests:
            if pd.isna(p):
                pv, verdict = "—", "<span class='pill pill-muted'>amostra insuficiente</span>"
            else:
                pv = f"{p:.4f}"
                verdict = ("<span class='pill pill-green'>distinta · diferença "
                           "significativa (p≤0,05)</span>" if p <= 0.05 else
                           "<span class='pill pill-yellow'>indistinguível · candidata "
                           "a fusão (p&gt;0,05)</span>")
            d = desc if len(desc) <= 40 else desc[:37] + "…"
            sib_rows += (
                "<div style='display:flex;align-items:center;gap:7px;flex-wrap:wrap;"
                "font-size:12px;color:#3a4250;margin:3px 0'>"
                f"<b>{lado} {d}</b>"
                f"<span style='color:#6b7480'>teste:</span><b>{sib_name}</b>"
                f"<span style='color:#6b7480'>p-valor:</span>"
                f"<b class='mono'>{pv}</b>{verdict}</div>")

        # 5) Estabilidade · PSI por amostra com barrinha verde/amarelo/vermelho
        psi_hex = {"green": "#137a3e", "yellow": "#9a6b00", "red": "#b3261e"}

        def gauge(p):
            if pd.isna(p):
                return ("<div style='flex:1;height:9px;border-radius:5px;"
                        "background:#eceff3'></div>")
            pos = min(max(p, 0.0) / 0.50, 1.0) * 100
            return (
                "<div style='position:relative;flex:1;height:9px;border-radius:5px;"
                "background:linear-gradient(to right,#2bb673 0%,#2bb673 20%,"
                "#e6b800 20%,#e6b800 50%,#e0584f 50%,#e0584f 100%)'>"
                f"<div style='position:absolute;left:calc({pos:.1f}% - 1px);top:-2px;"
                "width:2px;height:13px;background:#15324a;border-radius:1px'></div></div>")

        psi_rows = ""
        for a in ordered_nonref:      # DES é a referência (PSI ≡ 0), por isso fica de fora
            p = self._leaf_psi(sid, a)
            if pd.isna(p):
                pv, pcol = "—", "#8a93a3"
            else:
                pv, pcol = f"{p:.3f}", psi_hex[self._psi_class(p)]
            psi_rows += (
                "<div style='display:flex;align-items:center;gap:9px;margin:5px 0'>"
                f"<div style='width:78px;font-size:11px;color:#6b7480;white-space:nowrap'>"
                f"PSI {ab(a)}</div>"
                f"<div class='mono' style='width:48px;font-size:12.5px;font-weight:600;"
                f"color:{pcol}'>{pv}</div>{gauge(p)}</div>")
        psi_legend = (
            "<div style='font-size:10px;color:#8a93a3;margin-top:5px'>"
            "<span style='color:#2bb673'>■</span> &lt;0,10 estável &nbsp; "
            "<span style='color:#e6b800'>■</span> 0,10–0,25 atenção &nbsp; "
            "<span style='color:#e0584f'>■</span> &gt;0,25 crítico</div>")

        out = (head
               + sec_h("Volumetria &amp; representatividade")
               + f"<div class='treeui-metrics'>{sec1}</div>"
               + sec_h(f"{self._risk_mean} &amp; incremento vs DES")
               + f"<div class='treeui-metrics'>{sec2}</div>")
        h0_css = "font-size:10.5px;color:#8a93a3;margin:1px 0 6px;line-height:1.5"
        if test_rows:
            out += (sec_h("Aderência DES → amostra (teste de hipótese)")
                    + f"<div style='{h0_css}'><b>H₀:</b> a folha tem a <b>mesma "
                      f"distribuição de {self._risk_label}</b> na DES e na amostra. "
                      "<i>p&gt;0,05</i> ⇒ não rejeita H₀ (aderente); "
                      "<i>p≤0,05</i> ⇒ rejeita H₀ (não aderente).</div>"
                    + test_rows)
        if sib_rows:
            out += (sec_h("Distinção vs folha-irmã adjacente (mesmo pai)")
                    + f"<div style='{h0_css}'><b>H₀:</b> as <b>duas folhas-irmãs têm "
                      f"{'a mesma PD' if self._is_clf else 'o mesmo LGD'}</b>. "
                      "<i>p≤0,05</i> ⇒ rejeita H₀ (folhas distintas); "
                      "<i>p&gt;0,05</i> ⇒ não rejeita H₀ (indistinguíveis · candidatas "
                      "a fusão).</div>"
                    + sib_rows)
        if psi_rows:
            out += sec_h("Estabilidade · PSI") + psi_rows + psi_legend
        return out

    def _leaf_psi(self, sid, sample, eps=1e-6):
        """PSI de uma folha entre a referência (DES) e `sample` — mesma fórmula
        da tabela de folhas (_append_psi_cols), restrita a esta folha."""
        import math
        if self.sample_col is None or sample not in self._sample_masks:
            return float("nan")
        ref_mask = self._sample_masks.get(self.ref_sample)
        if ref_mask is None:
            return float("nan")
        leaf = self.seg.segments[sid]["mask"]
        s_mask = self._sample_masks[sample]
        n_ref, n_s = int(ref_mask.sum()), int(s_mask.sum())
        if n_ref == 0 or n_s == 0:
            return float("nan")
        p_ref = max((leaf & ref_mask).sum() / n_ref, eps)
        p_cur = max((leaf & s_mask).sum() / n_s, eps)
        return float((p_cur - p_ref) * math.log(p_cur / p_ref))

    def _leaf_table_spec(self):
        """(lv, cols, headers) da tabela de folhas — fonte única para a versão
        renderizada (HTML) e para a versão copiável (TSV p/ Excel)."""
        lv = self.seg.leaves(with_psi=True, with_test=True, test=self.dd_test.value)
        lv = lv.rename(columns={"nota": "folha"})   # chamamos de folha, não nota

        def ab(a):
            return "ESTAB" if a == "ESTABILIDADE" else a

        # Colunas em blocos legíveis: identificação · % por amostra · PD média
        # por amostra (só as que têm alvo) · PSI por amostra · teste de hipótese.
        # `headers` renomeia só a EXIBIÇÃO (a formatação segue pelos nomes reais).
        cols = ["folha", "descricao"]
        headers = {"folha": "folha", "descricao": "descrição"}
        if self.sample_col is None:
            for c, h in (("repr_%", "repr. %"), ("valor_medio", self._risk_mean)):
                if c in lv.columns:
                    cols.append(c); headers[c] = h
        else:
            for a in [self.ref_sample] + self._nonref:       # % DES · % OOT · % ESTAB
                c = f"repr_{a}_%"
                if c in lv.columns:
                    cols.append(c); headers[c] = f"% {ab(a)}"
            for a in [self.ref_sample] + self._pd_nonref:    # PD DES · PD OOT
                c = f"valor_{a}"
                if c in lv.columns:
                    cols.append(c); headers[c] = f"{self._risk_label} {ab(a)}"
            for a in self._nonref:                           # PSI OOT · PSI ESTAB
                c = f"psi_{a}"
                if c in lv.columns:
                    cols.append(c); headers[c] = f"PSI {ab(a)}"
        if "p_vs_prox" in lv.columns:
            cols.append("p_vs_prox"); headers["p_vs_prox"] = "p (irmãs)"
        return lv, cols, headers

    def _refresh_table(self):
        lv, cols, headers = self._leaf_table_spec()
        sty = self._style_leaves(lv[cols]).relabel_index(
            [headers[c] for c in cols], axis="columns")
        # tabela larga (muitas amostras): garante a largura natural para NÃO cortar
        # a última coluna (ex.: PSI ESTAB) — rola na horizontal dentro do container.
        sty = sty.set_table_styles(
            [{"selector": "", "props": [("min-width", "max-content")]}], overwrite=False)
        self.out_table.value = self._styler_html(sty, max_height="320px")

    def _leaves_tsv(self):
        """Tabela de folhas em TSV (tab = coluna, números em pt-BR) — cola direto
        no Excel: colunas separadas certas e células numéricas de verdade."""
        lv, cols, headers = self._leaf_table_spec()

        def br(x):   # pt-BR: vírgula decimal (Excel reconhece como número)
            return str(x).replace(".", ",")

        def fmt(col, v):
            if pd.isna(v):
                return ""                       # vazio (não "—") p/ a célula ficar limpa
            if col.startswith("pd_"):           # PD em % (igual à tela)
                return br(f"{v:.2%}")
            if col.startswith("psi_"):
                return br(f"{v:.4f}")
            if col == "p_vs_prox":
                return br(f"{v:.3f}")
            if col == "repr_%" or (col.startswith("repr_") and col.endswith("_%")):
                return br(f"{v:.1f}")
            if isinstance(v, float):
                return br(f"{v:g}")
            return str(v)

        linhas = ["\t".join(headers[c] for c in cols)]
        for _, row in lv[cols].iterrows():
            linhas.append("\t".join(fmt(c, row[c]) for c in cols))
        return "\n".join(linhas)

    def _on_copy_table(self, _):
        self.out_table_tsv.value = self._leaves_tsv()
        self.out_table_tsv.layout.display = ""      # revela a caixa p/ copiar

    def _refresh_metrics(self):
        m = self.seg.metrics()

        def ks_bg(v):
            if pd.isna(v):
                return "color:#aab"
            c = "#e6f6ec" if v >= 0.30 else "#fdf3da" if v >= 0.20 else "#fde7e7"
            return f"background-color:{c};font-weight:600"

        def auc_bg(v):
            if pd.isna(v):
                return "color:#aab"
            c = "#e6f6ec" if v >= 0.70 else "#fdf3da" if v >= 0.60 else "#fde7e7"
            return f"background-color:{c};font-weight:600"

        def r2_bg(v):
            if pd.isna(v):
                return "color:#aab"
            c = "#e6f6ec" if v >= 0.5 else "#fdf3da" if v >= 0.2 else "#fde7e7"
            return f"background-color:{c};font-weight:600"

        if self._is_clf:
            fmt = {c: "{:.4f}" for c in ("taxa_default", "KS", "AUC", "Gini", "Acuracia", "F1")}
            sty = (m.style.map(ks_bg, subset=["KS"]).map(auc_bg, subset=["AUC"]))
        else:
            fmt = {c: "{:.4f}" for c in ("MAE", "RMSE", "R2")}
            sty = m.style.map(r2_bg, subset=["R2"])
        sty = (sty
               .format(fmt, na_rep="—")
               .hide(axis="index")
               .set_table_styles(self._TABLE_STYLES)
               .set_properties(**{"font-size": "12px"})
               # mesmo visual da tabela de folhas: bordas + cabeçalho grafite +
               # zebra + texto centralizado (cabeçalho e células)
               .set_table_styles([{"selector": "th, td",
                                   "props": [("text-align", "center")]}],
                                 overwrite=False))
        self.out_metrics.value = self._styler_html(sty)

    def _ordered_leaf_options(self):
        """Opções do dropdown na MESMA ordem da árvore (esquerda→direita por nota),
        com a DESCRIÇÃO COMPLETA da folha (todas as condições) — sem truncar, para
        identificar a folha por inteiro."""
        seg = self.seg
        filhos: dict = {}
        for sid, s in seg.segments.items():
            filhos.setdefault(s["parent"], []).append(sid)
        nota_map, _ = seg._grade_map()
        n_total = len(self.df)
        min_nota = self._min_nota_fn(filhos, nota_map)

        def value_of(sid):
            sub = self.df[seg.segments[sid]["mask"]]
            return sub[self.target].mean() if len(sub) else float("inf")

        opts = []

        def rec(sid):
            s = seg.segments[sid]
            if s["is_leaf"]:
                own = ("TODA A CARTEIRA" if s["parent"] is None
                       else seg._descrever(s["conditions"]))   # caminho COMPLETO, sem cortar
                rep = 100 * s["mask"].sum() / n_total
                lock = "🔒 " if sid in self.locked else ""
                nota = nota_map.get(sid, "?")
                label = f"[{nota:>2}] {lock}{own}  ({self._risk_label} {value_of(sid) * 100:.2f}% · {rep:.0f}%)"
                opts.append((label, sid))
            for c in sorted(filhos.get(sid, []), key=min_nota):   # esquerda→direita
                rec(c)

        rec("root")
        return opts

    def _set_html(self, widget, key, html):
        """Escreve ``widget.value`` SÓ quando o conteúdo muda (hash-and-skip).
        Reatribuir .value sempre dispara um update completo pelo comm kernel↔
        browser (reparse do HTML/CSS inline do Styler ou do <img> base64); pular
        os updates idênticos corta a maior parte do tráfego redundante por ação."""
        if self._last_html.get(key) != html:
            widget.value = html
            self._last_html[key] = html

    def _refresh_lock_labels(self):
        """Atualiza SÓ o que depende de self.locked (rótulo 🔒): a árvore e os
        rótulos dos dropdowns de folha. Usado por lock/unlock para não pagar o
        _refresh completo (IV/PSI/metrics/tabela/PNG) só para alternar um cadeado."""
        self._suspend_leaf_obs = True
        try:
            cur = self.dd_leaf.value
            opts = self._ordered_leaf_options()
            self.dd_leaf.options = opts
            if cur in [s for _, s in opts]:
                self.dd_leaf.value = cur
            var_opts = [("TODA A CARTEIRA (raiz)", "root")] + opts
            cur_v = self.dd_var_leaf.value
            self.dd_var_leaf.options = var_opts
            self.dd_var_leaf.value = cur_v if cur_v in [s for _, s in var_opts] else "root"
        finally:
            self._suspend_leaf_obs = False
        self._set_html(self.out_tree, "tree", self._tree_html())
        self.tree_sel_style.value = self._leaf_highlight_style()
        self._set_html(self.leaf_chips, "chips", self._leaf_chips_html())
        self._set_html(self.leaf_header, "header", self._leaf_header_html())
        self._set_html(self.bar, "bar", self._status_html())

    def _refresh(self):
        # suspende o observer de folha enquanto reatribuímos os dropdowns: senão a
        # troca de dd_leaf.value re-dispara _on_leaf_change DENTRO do _refresh,
        # renderizando árvore/IV/histograma 2× por mutação.
        self._suspend_leaf_obs = True
        try:
            opts = self._ordered_leaf_options()
            leaves = [sid for _, sid in opts]
            cur = self.dd_leaf.value
            self.dd_leaf.options = opts
            if cur in leaves:
                self.dd_leaf.value = cur
            elif opts:
                self.dd_leaf.value = opts[0][1]
            # dropdown de folha da aba "Análise de variáveis": raiz + folhas
            cur_v = self.dd_var_leaf.value
            var_opts = [("TODA A CARTEIRA (raiz)", "root")] + opts
            self.dd_var_leaf.options = var_opts
            self.dd_var_leaf.value = cur_v if cur_v in [s for _, s in var_opts] else "root"
            # seletor de grupos de folhas-irmãs (aba Diagnóstico)
            sib_opts = [(g["label"], g["parent"]) for g in self.seg.sibling_leaf_groups()]
            cur_sib = self.dd_sib_group.value
            self.dd_sib_group.options = sib_opts
            if cur_sib in [p for _, p in sib_opts]:
                self.dd_sib_group.value = cur_sib
            elif sib_opts:
                self.dd_sib_group.value = sib_opts[0][1]
        finally:
            self._suspend_leaf_obs = False

        self._set_html(self.bar, "bar", self._status_html())
        self._set_html(self.out_tree, "tree", self._tree_html())
        self.tree_sel_style.value = self._leaf_highlight_style()
        self._set_html(self.leaf_header, "header", self._leaf_header_html())
        self._set_html(self.leaf_chips, "chips", self._leaf_chips_html())
        self._refresh_iv()
        self._refresh_leaf_hist()
        self._refresh_metrics()
        self._refresh_table()
        # o IC bootstrap, a discriminação e a imagem ficam obsoletos após mudanças
        self.out_boot.value = ("<div style='font-size:12px;color:#889'>Árvore alterada — "
                               "clique em <b>Calcular IC bootstrap</b> para (re)calcular.</div>")
        self.out_discrim.value = ("<div style='font-size:12px;color:#889'>Árvore alterada — "
                                  "clique em <b>Curva ROC</b> ou <b>Curva KS</b> para renderizar.</div>")
        self.out_plot.value = ("<div style='font-size:12px;color:#889'>Árvore alterada — "
                               "clique em <b>Ver / salvar árvore (imagem)</b> para renderizar.</div>")

    # ==================================================================
    # Entrada / handlers
    # ==================================================================
    def _selected_leaf(self):
        return self.dd_leaf.value

    def _feature_kind(self):
        sid = self.dd_leaf.value
        sub = (self.df if sid is None or sid not in self.seg.segments
               else self.df[self.seg.segments[sid]["mask"]])
        return self.seg._detect_kind(sub, self.dd_feature.value, None)

    def _on_mode_change(self, _):
        """Mostra o controle certo conforme modo e tipo da variável.

        Trocar o MODO (Ótimo↔Manual) invalida o preview pendente, mas MANTÉM os
        gráficos como referência — ex.: ao passar do Ótimo para o Manual para
        digitar os cortes, o gráfico do ótimo permanece à vista. Quem limpa os
        gráficos é a troca de VARIÁVEL/FOLHA (:meth:`_on_feature_change`)."""
        self._pending = None
        manual = self.tg_mode.value == "Manual"
        cat = self._feature_kind() == "cat"
        self.sl_bins.layout.display = "none" if manual else ""           # máx. bins: só Ótimo
        self.dd_split_criterion.layout.display = "none" if manual else ""  # critério: só Ótimo
        self.tx_cuts.layout.display = "" if (manual and not cat) else "none"   # cortes: Manual numérico
        self.cat_box.layout.display = "" if (manual and cat) else "none"      # grupos: Manual categórico
        self._sync_optbin_visibility()                                   # limites de bin: só Ótimo
        if manual and cat:
            self._rebuild_cat_box()

    def _on_feature_change(self, _):
        """Trocar a VARIÁVEL/FOLHA limpa o preview (o gráfico era de outra
        seleção) e reconfigura os controles do modo atual."""
        if hasattr(self, "out_preview_seg"):      # widgets podem não existir na 1ª chamada
            self.out_preview_seg.value = ""
            self.out_preview_chart.value = ""
        self._on_mode_change(_)

    def _sync_optbin_visibility(self):
        """Limites de tamanho de bin (min/max) aparecem só no modo Ótimo; cada
        slider só quando o respectivo checkbox está marcado."""
        otimo = self.tg_mode.value == "Ótimo"
        self.cb_minbin.layout.display = "" if otimo else "none"
        self.cb_maxbin.layout.display = "" if otimo else "none"
        self.cb_mindiff.layout.display = "" if otimo else "none"
        self.sl_minbin.layout.display = "" if (otimo and self.cb_minbin.value) else "none"
        self.sl_maxbin.layout.display = "" if (otimo and self.cb_maxbin.value) else "none"
        self.sl_mindiff.layout.display = "" if (otimo and self.cb_mindiff.value) else "none"

    def _sync_autoconc_visibility(self):
        """Cada slider de concentração do auto-fit só aparece com o checkbox marcado."""
        self.sl_autoconc_min.layout.display = "" if self.cb_autoconc_min.value else "none"
        self.sl_autoconc_max.layout.display = "" if self.cb_autoconc_max.value else "none"

    def _optbin_extra(self):
        """kwargs de tamanho de bin (fração da folha) p/ o binning ótimo, conforme
        os checkboxes marcados. Vazio = usa os defaults do optbinning."""
        extra = {}
        if self.cb_minbin.value:
            extra["min_bin_size"] = float(self.sl_minbin.value)
        if self.cb_maxbin.value:
            extra["max_bin_size"] = float(self.sl_maxbin.value)
        if self.cb_mindiff.value:
            extra["min_mean_diff"] = float(self.sl_mindiff.value)
        return extra

    def _rebuild_cat_box(self):
        """Monta um seletor de grupo por categoria presente na folha (ordenadas por PD)."""
        sid = self.dd_leaf.value
        feat = self.dd_feature.value
        # guarda: recriar N Dropdowns (novos modelos no comm + nós no DOM) é caro;
        # se o contexto (variável, folha) não mudou e os widgets já existem, mantém.
        # _on_feature_change/_on_mode_change disparam a cada troca de folha, mas
        # navegar entre folhas no modo Manual+cat não precisa reinstanciar tudo.
        if (getattr(self, "_cat_ctx", None) == (feat, sid)
                and getattr(self, "_cat_widgets", None)):
            return
        self._cat_widgets = {}
        self._cat_ctx = (feat, sid)
        if sid is None or sid not in self.seg.segments:
            self.cat_box.children = (); return
        sub = self.df[self.seg.segments[sid]["mask"]]
        s = sub[feat]
        valid = sub[s.notna()]
        if len(valid) == 0:
            self.cat_box.children = (W.HTML(
                "<div style='font-size:11px;color:#889'>Sem categorias nesta folha.</div>"),)
            return
        means = (valid.assign(_c=valid[feat].astype(str))
                 .groupby("_c")[self.target].mean().sort_values())
        order = means.index.tolist()
        n = len(order)
        rows = [W.HTML("<div style='font-size:11px;color:#667;margin-bottom:4px'>"
                       f"Categorias no <b>mesmo grupo</b> viram um nó. Ordenadas por {self._risk_label}. "
                       "Faltantes (NaN) já viram um nó próprio.</div>")]
        for k, c in enumerate(order, 1):
            dd = W.Dropdown(options=[(f"grupo {g}", g) for g in range(1, n + 1)], value=k,
                            layout=W.Layout(width="110px"))
            self._cat_widgets[c] = dd
            lab = W.HTML(f"<span style='font-size:12px'><b>{c}</b>"
                         f"<span style='color:#889'> · {self._risk_label} {means[c]:.3f}</span></span>")
            rows.append(W.HBox([dd, lab], layout=W.Layout(align_items="center")))
        na_n = int(s.isna().sum())
        if na_n:
            rows.append(W.HTML(f"<div style='font-size:11px;color:#9a6b00;margin-top:3px'>"
                               f"+ <b>(faltante)</b>: {na_n} linhas → nó próprio automático</div>"))
        self.cat_box.children = tuple(rows)

    def _cat_groups(self):
        if (getattr(self, "_cat_ctx", None) != (self.dd_feature.value, self.dd_leaf.value)
                or not getattr(self, "_cat_widgets", None)):
            self._rebuild_cat_box()
        grupos = {}
        for c, dd in self._cat_widgets.items():
            grupos.setdefault(dd.value, []).append(c)
        return [grupos[g] for g in sorted(grupos)]

    def _on_collapse(self, _):
        sid = self._selected_leaf()
        with self.out_log:
            self.out_log.clear_output(wait=True)
            if sid is None or sid not in self.seg.segments:
                print("Nenhuma folha selecionada."); return
            parent = self.seg.segments[sid]["parent"]
            if parent is None:
                print("Esta folha é a raiz — não há pai para recolher."); return
            self._checkpoint()
            self.seg.collapse(parent)
        self.locked &= set(self.seg.segments)
        self._pending = None
        self._refresh()
        if parent in [s for s, seg in self.seg.segments.items() if seg["is_leaf"]]:
            self.dd_leaf.value = parent

    def _on_merge(self, side):
        sid = self._selected_leaf()
        with self.out_log:
            self.out_log.clear_output(wait=True)
            if sid is None or sid not in self.seg.segments:
                print("Selecione uma folha."); return
            parent = self.seg.segments[sid]["parent"]
            before = set(self.seg.segments)
            self._checkpoint()
            self.seg.merge_leaf(sid, side=side)
        self.locked &= set(self.seg.segments)
        self._pending = None
        novos = [i for i in self.seg.segments
                 if i not in before and self.seg.segments[i]["is_leaf"]]
        self._refresh()
        folhas = [s for s, seg in self.seg.segments.items() if seg["is_leaf"]]
        alvo = (novos[0] if novos else (parent if parent in folhas else None))
        if alvo in folhas:
            self.dd_leaf.value = alvo

    def _on_merge_missing(self, _):
        sid = self._selected_leaf()
        with self.out_log:
            self.out_log.clear_output(wait=True)
            if sid is None or sid not in self.seg.segments:
                print("Selecione a folha POPULADA de destino."); return
            before = set(self.seg.segments)
            redo_bak = list(self._redo)
            self._checkpoint()
            self.seg.merge_missing(sid)
            if set(self.seg.segments) == before:
                self._undo.pop()                 # nada mudou — não polui o histórico
                self._redo[:] = redo_bak         # ...nem destrói a pilha de refazer
                self._sync_undo_buttons()
                return
        self.locked &= set(self.seg.segments)
        self._pending = None
        novos = [i for i in self.seg.segments
                 if i not in before and self.seg.segments[i]["is_leaf"]]
        self._refresh()
        folhas = [s for s, seg in self.seg.segments.items() if seg["is_leaf"]]
        if novos and novos[0] in folhas:
            self.dd_leaf.value = novos[0]

    def _on_suggest(self, _):
        sid = self._selected_leaf()
        with self.out_log:
            self.out_log.clear_output(wait=True)
            if sid is None:
                print("Selecione uma folha."); return
            sug = self.seg.suggest_split(sid)
            if sug["feature"] is None:
                print("Nenhuma variável informativa para esta folha — IV muito baixo.")
                return
            if sug["feature"] in list(self.dd_feature.options):
                self.dd_feature.value = sug["feature"]
            self.tg_mode.value = "Ótimo"
            lbl = self.seg.feature_labels.get(sug["feature"], sug["feature"])
            print(f"Sugestão para esta folha: dividir por '{lbl}' "
                  f"(IV={sug['iv']:.4f}, {sug['forca']}).")
            print("Já deixei a variável selecionada no modo Ótimo — "
                  "rode o 👁 Preview e depois Criar segmento.")

    def _on_suggest3(self, _):
        sid = self._selected_leaf()
        with self.out_log:
            self.out_log.clear_output(wait=True)
            if sid is None:
                self.out_suggest.value = "<i>Selecione uma folha na aba Construir.</i>"
                return
            try:
                sug = self.seg.suggest_splits(sid, top=3)
            except Exception as e:
                self.out_suggest.value = (f"<div style='color:#b3261e;font-size:12px'>Erro: "
                                          f"{type(e).__name__}: {e}</div>")
                return
            if sug.empty:
                self.out_suggest.value = "<i>Nenhuma variável informativa para esta folha.</i>"
                print("Sugestão: nenhuma variável com IV suficiente nesta folha.")
                return
            disp = sug.copy()
            disp["passa_teste"] = disp["passa_teste"].map({True: "✅", False: "—"})
            disp = disp.rename(columns={"n_bins": "nº bins", "passa_teste": "passa teste"})
            self.out_suggest.value = self._df_html(disp, center=True, color=True)
            print(f"TOP {len(sug)} splits sugeridos para a folha selecionada.")

    def _on_importance(self, _):
        with self.out_log:
            self.out_log.clear_output(wait=True)
            try:
                fi = self.seg.feature_importance()
            except Exception as e:
                self.out_importance.value = (f"<div style='color:#b3261e;font-size:12px'>Erro: "
                                             f"{type(e).__name__}: {e}</div>")
                self.out_importance_chart.value = ""; self.out_importance_legend.value = ""
                return
            if fi.empty:
                self.out_importance.value = ("<i>A árvore ainda não tem splits — construa a "
                                             "segmentação primeiro.</i>")
                self.out_importance_chart.value = ""; self.out_importance_legend.value = ""
                return
            col = "importancia_%" if "importancia_%" in fi.columns else "importancia"
            vmax = float(fi[col].max()) or 1.0

            def _imp_bg(v):                          # cor proporcional à importância
                frac = 0.0 if vmax <= 0 else max(0.0, float(v)) / vmax
                r = int(232 - 150 * frac); g = int(245 - 35 * frac); b = int(233 - 165 * frac)
                peso = "700" if frac >= 0.66 else "600" if frac >= 0.33 else "400"
                return f"background-color:rgb({r},{g},{b});font-weight:{peso}"

            fmt = {"importancia": "{:.4f}"}
            if "importancia_%" in fi.columns:
                fmt["importancia_%"] = "{:.1f}%"
            sty = (fi.style.hide(axis="index")
                   .set_table_styles(self._TABLE_STYLES)
                   .set_properties(**{"font-size": "12px"})
                   .set_table_styles([{"selector": "th, td",
                                       "props": [("text-align", "center")]}], overwrite=False)
                   .format(fmt)
                   .map(_imp_bg, subset=[col]))
            # gráfico de importância relativa (barras horizontais) — ao lado da tabela
            try:
                chart = self._fig_html(self.seg.plot_importance_bar())
            except Exception as e:
                chart = (f"<div style='color:#b3261e;font-size:12px'>Erro no gráfico: "
                         f"{type(e).__name__}: {e}</div>")
            dic = (
                "<div class='treeui-legend' style='margin-top:8px'>"
                "<b>O que é a importância?</b> Em cada nó interno, a variável do split contribui "
                "com <b>(IV da variável no nó) × (representatividade do nó)</b> — ganho de "
                "separação ponderado pela população afetada. A importância de uma variável é a "
                "<b>soma</b> dessas contribuições nos nós em que ela dividiu; "
                "<b>importancia_%</b> normaliza para 100% (quanto cada variável pesa na árvore). "
                "<b>n_splits</b> = em quantos nós ela foi usada. "
                "<span style='background:rgb(232,245,233);padding:0 5px'>cor clara = baixa</span> "
                "&rarr; <span style='background:rgb(82,210,68);padding:0 5px;font-weight:700'>"
                "cor forte = alta</span>.</div>")
            self.out_importance.value = self._styler_html(sty)
            self.out_importance_chart.value = chart
            self.out_importance_legend.value = dic
            print("Importância das variáveis na árvore calculada.")

    def _on_sql(self, _):
        with self.out_log:
            self.out_log.clear_output(wait=True)
            tbl = (self.tx_sql_table.value or "minha_tabela").strip()
            try:
                self.out_sql.value = self.seg.to_sql(table=tbl)
                print("SQL gerado — selecione tudo na caixa e copie (Ctrl+C).")
            except Exception as e:
                self.out_sql.value = f"-- Erro ao gerar SQL: {type(e).__name__}: {e}"

    def _on_diff(self, _):
        from .segmenter import TreeSegmenter
        with self.out_log:
            self.out_log.clear_output(wait=True)
            path = (self.tx_diff_path.value or "").strip()
            if not path:
                self.out_diff.value = "<i>Informe o caminho do JSON da árvore B.</i>"
                return
            try:
                other = TreeSegmenter.load(path, self.df)
                d = self.seg.diff_trees(other)
            except Exception as e:
                self.out_diff.value = (f"<div style='color:#b3261e;font-size:12px'>Erro ao "
                                       f"comparar: {type(e).__name__}: {e}</div>")
                return
            mig = d["migracao"].copy()
            mig.index = [f"A·{i}" for i in mig.index]
            mig.columns = [f"B·{c}" for c in mig.columns]
            html = (f"<div class='treeui-legend'>Concordância de notas (A=B): "
                    f"<b>{d['concordancia']:.1%}</b></div>"
                    + self._df_html(d["resumo"], center=True)
                    + "<div class='treeui-h' style='margin-top:8px'>Migração de notas "
                      "(linhas = árvore A · colunas = árvore B)</div>"
                    + mig.to_html(border=0))
            self.out_diff.value = html
            print(f"Comparação concluída — concordância {d['concordancia']:.1%}.")

    def _on_autofit(self, _):
        sid = self._selected_leaf()
        so_folha = sid is not None and sid != "root" and sid in self.seg.segments
        depth = int(self.sl_depth.value)
        criterion = self.dd_criterion.value
        cmin = float(self.sl_autoconc_min.value) if self.cb_autoconc_min.value else None
        cmax = float(self.sl_autoconc_max.value) if self.cb_autoconc_max.value else None
        with self.out_log:
            self.out_log.clear_output(wait=True)
            alvo = self._leaf_label(sid) if so_folha else "TODA A CARTEIRA"
            lim = []
            if cmin is not None:
                lim.append(f"folha ≥ {cmin:.1%}")
            if cmax is not None:
                lim.append(f"quebra ≤ {cmax:.0%}")
            slim = (", " + " · ".join(lim)) if lim else ""
            scrit = "" if criterion == "optbin" else f", critério={criterion}"
            print(f"Auto-fit em '{alvo}' (profundidade ≤ {depth}{slim}{scrit})…")
            self._checkpoint()
            self.seg.fit_auto(max_depth=depth, min_leaf_repr=cmin, max_bin_repr=cmax,
                              criterion=criterion, subtree=sid if so_folha else None,
                              from_scratch=not so_folha)
        if so_folha:
            self.locked &= set(self.seg.segments)   # só folhas removidas saem
        else:
            self.locked.clear()
        self._pending = None
        self._refresh()
        if so_folha and sid in self.seg.segments and not self.seg.segments[sid]["is_leaf"]:
            novas = [s for s, v in self.seg.segments.items()
                     if v["is_leaf"] and self.seg._is_descendant_or_self(s, sid)]
            if novas:
                self.dd_leaf.value = novas[0]
        with self.out_log:
            n = sum(s["is_leaf"] for s in self.seg.segments.values())
            escopo = "nesta folha" if so_folha else "na árvore"
            print(f"Auto-fit concluído {escopo}: {n} folhas no total. "
                  "Refine à mão: funda, recolha ou divida onde quiser.")

    def _on_mlflow(self, _):
        with self.out_log:
            self.out_log.clear_output(wait=True)
            exp = self.tx_experiment.value.strip() or None
            run = self.tx_runname.value.strip() or None
            model_name = self.tx_model.value.strip() or None
            uc = self.cb_uc.value
            if uc and not model_name:
                print("Para registrar no Unity Catalog, informe o nome no formato "
                      "catalogo.schema.modelo.")
                return
            if uc and model_name.count(".") != 2:
                print(f"Nome UC inválido: '{model_name}'. Use 3 níveis: catalogo.schema.modelo.")
                return
            print("Salvando no MLflow…")
            try:
                rid = self.seg.log_to_mlflow(
                    experiment=exp, run_name=run,
                    registered_model_name=model_name,
                    registry_uri="databricks-uc" if uc else None,
                    verbose=False)
                msg = f"✓ Run {rid[:8]}… salvo (régua, métricas e modelo pyfunc)."
                if model_name:
                    msg += f"\nModelo registrado em '{model_name}' — nova versão no Model Registry."
                    print(msg)
                    print(f"Para scoring: mlflow.pyfunc.load_model('models:/{model_name}/<versão>')"
                          " e use .predict.")
                else:
                    print(msg)
            except ImportError:
                print("MLflow não está instalado neste ambiente. Instale com: %pip install mlflow")
            except Exception as e:
                print(f"Erro ao salvar no MLflow: {type(e).__name__}: {e}")

    def _on_clear_log(self, _):
        self.out_log.clear_output()       # limpa a área de preview/log

    def _on_spark_apply(self, _):
        with self.out_log:
            self.out_log.clear_output(wait=True)
            name = self.tx_spark_in.value.strip()
            if not name:
                print("Informe o nome da tabela Spark de entrada."); return
            try:
                from pyspark.sql import SparkSession
            except ImportError:
                print("PySpark não está disponível neste ambiente. No Databricks já "
                      "vem no cluster; fora dele: %pip install pyspark."); return
            spark = SparkSession.getActiveSession()
            if spark is None:
                try:
                    spark = SparkSession.builder.getOrCreate()
                except Exception as e:
                    print("Nenhuma SparkSession ativa:", type(e).__name__, e); return
            try:
                sdf = spark.table(name)
            except Exception as e:
                print(f"Não foi possível ler a tabela '{name}':", type(e).__name__, e); return
            try:
                out = self.seg.apply_spark(sdf)
            except ValueError as e:                 # colunas faltando / árvore vazia
                print("⚠", e); return
            except Exception as e:
                print("Erro ao aplicar a régua:", type(e).__name__, e); return

            self.spark_result = out
            out_name = self.tx_spark_out.value.strip()
            if out_name:
                try:
                    out.write.mode("overwrite").saveAsTable(out_name)
                    print(f"✓ tabela '{out_name}' gravada (segmento, nota, valor_regua).")
                except Exception as e:
                    print(f"Régua aplicada, mas falhou ao gravar '{out_name}':",
                          type(e).__name__, e)
            print(f"✓ régua aplicada em '{name}'. Spark DataFrame em  ui.spark_result.")
            try:
                dist = out.groupBy("nota").count().orderBy("nota").toPandas()
                display(dist)
            except Exception as e:
                print("(não consegui resumir a distribuição:", type(e).__name__, e, ")")

    def _parse_cuts(self, feature, sid):
        sub = self.df[self.seg.segments[sid]["mask"]]
        kind = self.seg._detect_kind(sub, feature, None)
        if kind == "num":
            raw = self.tx_cuts.value.strip()
            return [float(x) for x in raw.replace(";", ",").split(",")
                    if x.strip()] if raw else None
        grupos = self._cat_groups()
        return grupos if grupos else None

    def _on_leaf_change(self, _):
        # ignora o disparo programático durante o _refresh (a árvore/IV/histograma
        # já são renderizados lá) — evita renderização dupla por mutação.
        if self._suspend_leaf_obs:
            return
        # trocar a folha NÃO altera a estrutura: a árvore HTML é a mesma. O realce
        # é aplicado por CSS (data-leaf) → só atualizamos o <style> minúsculo, sem
        # remontar nem reenviar a árvore inteira pelo comm.
        self.tree_sel_style.value = self._leaf_highlight_style()
        self._set_html(self.leaf_header, "header", self._leaf_header_html())
        self._set_html(self.leaf_chips, "chips", self._leaf_chips_html())
        self._refresh_iv()
        self._refresh_leaf_hist()
        self._on_feature_change(None)   # nova folha: limpa o preview e recompõe os grupos

    def _on_tab_change(self, change):
        """Ao abrir a aba 'Análise de variáveis', calcula a tabela de IV se estiver
        pendente (render preguiçoso — não pagamos o optbinning de todas as variáveis
        na abertura nem em cada mutação enquanto a aba não está à vista)."""
        if change.get("new") == self._iv_tab_index and getattr(self, "_iv_dirty", False):
            self._compute_iv()

    def _refresh_iv(self):
        # só calcula se a aba de variáveis estiver à vista; senão marca pendente e
        # mostra um placeholder (o cálculo roda quando a aba for aberta).
        if getattr(self, "tabs", None) is not None and \
                self.tabs.selected_index != self._iv_tab_index:
            self._iv_dirty = True
            self._set_html(self.out_iv, "iv",
                           "<div style='font-size:12px;color:#889'>Abra esta aba para "
                           "calcular o IV/PSI por variável da folha selecionada.</div>")
            return
        self._compute_iv()

    def _compute_iv(self):
        self._iv_dirty = False
        sid = self.dd_leaf.value
        iv = self.seg.variable_iv(sid)
        pd_med = iv.attrs.get("valor_medio")
        has_psi = "pior_psi" in iv.columns
        disp = (iv[["variavel", "n_bins", "iv", "forca"]].copy()
                .rename(columns={"n_bins": "bins"}))
        if has_psi:
            disp["psi"] = iv["pior_psi"].values
            disp["psi_status"] = iv["psi_classificacao"].values
        disp["variavel"] = disp["variavel"].map(
            lambda v: self.seg.feature_labels.get(v, v))
        if len(disp):
            disp.loc[0, "variavel"] = "★ " + str(disp.loc[0, "variavel"])
        disp = disp.rename(columns={"variavel": "variável", "forca": "força",
                                    "psi_status": "estab."})

        # estilo editorial: sem grade vertical, só régua de cabeçalho + filetes
        # horizontais; força/PSI como TEXTO colorido (sem preenchimentos).
        iv_styles = [
            {"selector": "", "props": [("border-collapse", "collapse"),
                                       ("width", "100%")]},
            {"selector": "th, td", "props": [("padding", "7px 12px"),
                                             ("border", "none"),
                                             ("border-bottom", "1px solid #eef1f4"),
                                             ("white-space", "nowrap"),
                                             ("text-align", "right")]},
            {"selector": "thead th", "props": [("text-transform", "uppercase"),
                                               ("font-size", "10px"),
                                               ("letter-spacing", ".06em"),
                                               ("color", "#8a93a3"),
                                               ("font-weight", "600"),
                                               ("padding-bottom", "6px"),
                                               ("border-bottom", "1.5px solid #d7dde6")]},
            {"selector": "thead th:first-child", "props": [("text-align", "left")]},
            {"selector": "tbody td:first-child", "props": [("text-align", "left")]},
            {"selector": "tbody tr:hover td", "props": [("background-color", "#f7f9fc")]},
            {"selector": "tbody tr:last-child td", "props": [("border-bottom", "none")]},
        ]

        def forca_txt(v):
            return {
                "forte": "color:#137a3e;font-weight:600",
                "médio": "color:#9a6b00;font-weight:600",
                "suspeito": "color:#6b3fa0;font-weight:600",
            }.get(v, "color:#9aa2b1")

        def psi_txt(v):
            if pd.isna(v):
                return "color:#9aa2b1"
            a = abs(v)
            c = "#137a3e" if a < 0.10 else "#9a6b00" if a < 0.25 else "#b3261e"
            return f"color:{c};font-weight:600"

        def estab_txt(v):
            return {
                "estável": "color:#137a3e",
                "atenção": "color:#9a6b00;font-weight:600",
                "instável": "color:#b3261e;font-weight:600",
            }.get(v, "color:#9aa2b1")

        def reco_row(r):
            # variável recomendada (★, maior IV): filete de acento + negrito,
            # tint quase imperceptível — destaque discreto, sem realce pesado.
            if r.name != 0:
                return [""] * len(r)
            css = ["background-color:#fafbfd"] * len(r)
            css[0] = ("background-color:#fafbfd;border-left:3px solid #3b4a63;"
                      "font-weight:600;color:#27324a")
            return css

        fmt = {"iv": "{:.4f}",
               "bins": lambda v: "—" if (pd.isna(v) or v == 0) else f"{int(v)}"}
        if has_psi:
            fmt["psi"] = "{:.4f}"
        num_cols = [c for c in ["bins", "iv", "psi"] if c in disp.columns]
        sty = (disp.style.format(fmt, na_rep="—")
               .hide(axis="index")
               .set_table_styles(iv_styles)
               .set_properties(**{"font-size": "12px", "color": "#3a4250"}))
        if num_cols:
            sty = sty.set_properties(subset=num_cols, **{
                "font-family": "'IBM Plex Mono', ui-monospace, monospace",
                "font-variant-numeric": "tabular-nums"})
        if len(disp):
            sty = sty.apply(reco_row, axis=1)
        sty = sty.map(forca_txt, subset=["força"])
        if has_psi:
            sty = (sty.map(psi_txt, subset=["psi"])
                      .map(estab_txt, subset=["estab."]))
        qual = "TODA A CARTEIRA" if (sid in (None, "root")) else self._leaf_label(sid)
        _iv_kind = "binário" if self._is_clf else "contínuo"
        hint = (f"<div style='font-size:11px;color:#667;margin-bottom:4px'>folha: "
                f"<b>{qual}</b> · {self._risk_mean} (DES) = {pd_med} · IV {_iv_kind} (optbinning)"
                + (" · PSI nos mesmos bins do IV (DES × amostra)" if has_psi else "")
                + "</div>")
        self._set_html(self.out_iv, "iv", hint + self._styler_html(sty))

    def _refresh_leaf_hist(self):
        """Alvo da folha selecionada (DES): taxa de default + IC de Wilson
        (classificação) ou histograma do alvo (regressão)."""
        sid = self.dd_leaf.value
        if sid is None or sid not in self.seg.segments:
            self._set_html(self.out_leaf_hist, "leaf_hist",
                           "<div style='font-size:11px;color:#889'>—</div>")
            return
        # cache do PNG por (sid, versão da árvore): revisitar a mesma folha (ou um
        # _refresh após lock/seleção que não mudou a massa da folha) reusa o blob
        # base64 em vez de re-renderizar a figura e reencodá-la a cada ação.
        ck = (sid, self.seg._tree_version)
        html = self._leaf_hist_cache.get(ck)
        if html is None:
            try:
                plot = (self.seg.plot_leaf_target_hist if self._is_clf
                        else self.seg.plot_leaf_value_hist)
                html = self._fig_html(plot(sid, figsize=self._PREVIEW_FIGSIZE))
            except Exception as e:
                html = (f"<div style='font-size:11px;color:#b3261e'>"
                        f"(gráfico não gerado: {type(e).__name__})</div>")
            if len(self._leaf_hist_cache) > 256:      # backstop de memória
                self._leaf_hist_cache.clear()
            self._leaf_hist_cache[ck] = html
        self._set_html(self.out_leaf_hist, "leaf_hist", html)

    # ==================================================================
    # Aba "Análise de variáveis"
    # ==================================================================
    def _var_cards_html(self, s, trend):
        psi_hex = {"green": "#137a3e", "yellow": "#9a6b00", "red": "#b3261e"}
        tipo = s.get("tipo")

        def chip(k, v, sub="", vcolor=None):
            sty = f" style='color:{vcolor}'" if vcolor else ""
            subh = (f"<div style='font-size:10px;color:#8a93a3;margin-top:2px;"
                    f"line-height:1.35'>{sub}</div>" if sub else "")
            return (f"<div class='treeui-metric' style='padding:9px 11px'>"
                    f"<div class='k'>{k}</div><div class='v mono'{sty}>{v}</div>{subh}</div>")

        def fnum(x, nd=2):
            return f"{x:.{nd}f}" if isinstance(x, (int, float)) and x == x else "—"

        def grid(cards, ncol, top=False):
            mt = "margin-top:6px;" if top else ""
            return (f"<div class='treeui-metrics' style='{mt}grid-template-columns:"
                    f"repeat({ncol},minmax(0,1fr))'>" + "".join(cards) + "</div>")

        miss = s.get("pct_missing")
        qual = [chip("% missing",
                     f"{miss:.1f}%" if (miss is not None and miss == miss) else "—",
                     f"{s.get('n_missing', 0)} de {s.get('n', 0)}")]
        iv = s.get("iv")
        if iv is not None:
            qual.append(chip("IV (binário)", f"{iv:.4f}", s.get("forca", "—")))
        if tipo == "num" and s.get("p5") is not None:
            qual.append(chip("P5–P95", f"{fnum(s.get('p5'))} – {fnum(s.get('p95'))}",
                             f"min {fnum(s.get('min'))} · max {fnum(s.get('max'))}"))
        html = grid(qual, len(qual))

        if tipo == "num" and s.get("media") is not None:
            html += grid([chip("Média", fnum(s.get("media"), 3)),
                          chip("Mediana", fnum(s.get("mediana"), 3)),
                          chip("Desvio", fnum(s.get("desvio"), 3)),
                          chip("N", f"{s.get('n', 0):,}".replace(",", "."))], 4, top=True)
        elif tipo == "cat" and s.get("top_categorias"):
            linhas = "".join(
                f"<div style='display:flex;justify-content:space-between;font-size:12px;"
                f"padding:3px 0;border-top:1px solid #f1f3f6'><span>{c}</span>"
                f"<span class='mono'>{p:.1f}%</span></div>"
                for c, p in s["top_categorias"][:8])
            html += ("<div class='treeui-metric' style='margin-top:6px;padding:8px 11px'>"
                     "<div class='k'>Categorias (share)</div>" + linhas + "</div>")

        psi = {a: v for a, v in (s.get("psi") or {}).items() if v is not None}
        if psi:
            def gauge(p):
                pos = min(max(p, 0.0) / 0.50, 1.0) * 100
                return ("<div style='position:relative;flex:1;height:8px;border-radius:5px;"
                        "background:linear-gradient(to right,#2bb673 0%,#2bb673 20%,"
                        "#e6b800 20%,#e6b800 50%,#e0584f 50%,#e0584f 100%)'>"
                        f"<div style='position:absolute;left:calc({pos:.1f}% - 1px);top:-2px;"
                        "width:2px;height:12px;background:#15324a;border-radius:1px'></div></div>")
            rows = ""
            for a, v in psi.items():
                ab = "ESTAB" if a == "ESTABILIDADE" else a
                cls = self._psi_class(v)
                txt = {"green": "estável", "yellow": "atenção", "red": "instável"}[cls]
                rows += ("<div style='display:flex;align-items:center;gap:9px;margin:6px 0'>"
                         f"<div style='width:74px;font-size:11.5px;color:#6b7480;"
                         f"white-space:nowrap'>PSI {ab}</div>"
                         f"<div class='mono' style='width:50px;font-size:13px;font-weight:600;"
                         f"color:{psi_hex[cls]}'>{v:.3f}</div>{gauge(v)}"
                         f"<div style='width:54px;text-align:right;font-size:10.5px;"
                         f"color:{psi_hex[cls]}'>{txt}</div></div>")
            legend = ("<div style='font-size:10px;color:#8a93a3;margin-top:4px'>"
                      "<span style='color:#2bb673'>■</span> &lt;0,10 estável &nbsp;"
                      "<span style='color:#e6b800'>■</span> 0,10–0,25 atenção &nbsp;"
                      "<span style='color:#e0584f'>■</span> &gt;0,25 instável</div>")
            html += ("<div class='treeui-h' style='margin-top:13px'>Estabilidade · PSI por "
                     "amostra (vs. DES)</div>" + rows + legend)

        if trend:
            arrow = "↑" if trend["pct"] >= 0 else "↓"
            tc = ("#b3261e" if abs(trend["pct"]) >= 10
                  else "#9a6b00" if abs(trend["pct"]) >= 3 else "#137a3e")
            html += ("<div style='display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;"
                     "font-size:11.5px;margin-top:12px;padding-top:8px;"
                     "border-top:1px solid #eef1f4'>"
                     "<span style='color:#6b7480'>Tendência da média</span>"
                     f"<b style='color:{tc};font-size:13px'>{arrow} {trend['pct']:+.0f}%</b>"
                     f"<span style='color:#8a93a3'>{trend['de']:.2f} → {trend['para']:.2f} · "
                     f"{trend['n_safras']} safras ({trend['ini']} → {trend['fim']})</span></div>")
        return html

    def _style_var_safra(self, bs):
        """Detalhe por safra (numérica) — visual editorial: cabeçalho claro (como
        as demais tabelas), coluna 'safra' ancorada, números mono, 'média' como
        coluna-foco e %missing por severidade (só cor de texto)."""
        order = ["safra", "min", "p5", "media", "p95", "max", "pct_missing"]
        cols = [c for c in order if c in bs.columns]
        bs = bs[cols].copy()

        num_cols = [c for c in ("min", "p5", "media", "p95", "max") if c in cols]
        fmt = {c: "{:.3f}" for c in num_cols}
        if "pct_missing" in cols:
            fmt["pct_missing"] = "{:.1f}%"
        labels = {"safra": "safra", "min": "mín", "p5": "p5", "media": "média",
                  "p95": "p95", "max": "máx", "pct_missing": "% falt."}

        sty = (bs.style.format(fmt, na_rep="—")
                       .hide(axis="index")
                       .set_properties(**{"font-size": "12px"}))

        val_cols = [c for c in cols if c != "safra"]
        if val_cols:
            sty = sty.set_properties(
                subset=val_cols,
                **{"font-family": "'IBM Plex Mono',ui-monospace,monospace",
                   "font-variant-numeric": "tabular-nums"})

        if "media" in cols:
            sty = sty.set_properties(
                subset=["media"],
                **{"font-weight": "700", "color": "#27324a",
                   "background-color": "#eef1f5",
                   "border-left": "1px solid #cdd5e0",
                   "border-right": "1px solid #cdd5e0"})

        if "pct_missing" in cols:
            def _sev(s):
                out = []
                for v in s:
                    if pd.isna(v):
                        out.append("color:#6b7480")
                    elif v >= 20:
                        out.append("color:#b3261e;font-weight:600")
                    elif v > 0:
                        out.append("color:#9a6b00;font-weight:600")
                    else:
                        out.append("color:#137a3e")
                return out
            sty = sty.apply(_sev, axis=0, subset=["pct_missing"])

        extra = list(self._TABLE_STYLES) + list(self._SAFRA_HEADER_STYLES)
        extra.append({"selector": "th, td", "props": [("padding", "5px 11px")]})
        sty = sty.set_table_styles(extra)

        sty = sty.relabel_index([labels.get(c, c) for c in cols], axis=1)
        return sty

    def _style_var_share(self, sh):
        """Detalhe por safra (categórica) — representatividade (%) por categoria
        com heatmap monocromático grafite-azul (escala global 0..100, tons
        pálidos), coluna 'safra' ancorada e baldes residuais em cinza."""
        cols = list(sh.columns)
        cat_cols = [c for c in cols if c != "safra"]
        fmt = {c: "{:.1f}%" for c in cat_cols}

        sty = (sh.style.format(fmt, na_rep="—")
                       .hide(axis="index")
                       .set_properties(**{"font-size": "12px"}))

        if cat_cols:
            sty = sty.set_properties(
                subset=cat_cols,
                **{"font-family": "'IBM Plex Mono',ui-monospace,monospace",
                   "font-variant-numeric": "tabular-nums",
                   "min-width": "56px"})

        if cat_cols:
            applied = False
            try:
                cmap = self._blues_set_bad()
                sty = sty.background_gradient(
                    cmap=cmap, subset=cat_cols, axis=None,
                    vmin=0.0, vmax=100.0, low=0.0, high=0.55)
                applied = True
            except Exception:
                applied = False

            if applied:
                def _ink(s):
                    out = []
                    for v in s:
                        if pd.isna(v):
                            out.append("color:#6b7480")
                        elif v >= 70:
                            out.append("color:#ffffff")
                        else:
                            out.append("color:#1f2733")
                    return out
                sty = sty.apply(_ink, axis=0, subset=cat_cols)
            else:
                def _heat(s):
                    return [self._accent_ramp_css(v, 0.0, 100.0, ceiling=0.55)
                            for v in s]
                sty = sty.apply(_heat, axis=0, subset=cat_cols)

        for special in ("outras", "(faltante)"):
            if special in cat_cols:
                sty = sty.set_properties(subset=[special], **{"color": "#6b7480"})

        extra = list(self._TABLE_STYLES) + list(self._SAFRA_HEADER_STYLES)
        extra.append({"selector": "th, td", "props": [("padding", "5px 10px")]})
        sty = sty.set_table_styles(extra)

        sty = sty.relabel_index(list(cols), axis=1)
        return sty

    def _on_var_analyze(self, _):
        feat = self.dd_var.value
        sid = self.dd_var_leaf.value
        tcol = self.tx_var_time.value.strip()
        for o in (self.out_var_dist, self.out_var_time, self.out_var_psi, self.out_var_table):
            o.value = ""                       # HTML widgets: limpa via .value
        self.out_var_cards.value = ""

        def err(what, e):
            return (f"<div style='font-size:11px;color:#b3261e'>({what} não gerada: "
                    f"{type(e).__name__})</div>")
        bs, trend = None, None
        with self.out_log:
            self.out_log.clear_output(wait=True)
            if feat is None:
                print("Selecione uma variável para analisar."); return
            try:
                summ = self.seg.variable_summary(feat, sid=sid)
            except Exception as e:
                print("Erro no resumo da variável:", type(e).__name__, e); return
            kind = summ.get("tipo")
            if kind == "num" and tcol and tcol in self.df.columns:
                try:
                    bs = self.seg.variable_by_safra(feat, tcol, sid=sid)
                    med = bs["media"].dropna()
                    if len(med) >= 2 and med.iloc[0] != 0:
                        trend = {"pct": 100 * (med.iloc[-1] - med.iloc[0]) / abs(med.iloc[0]),
                                 "n_safras": len(bs), "de": float(med.iloc[0]),
                                 "para": float(med.iloc[-1]),
                                 "ini": str(bs["safra"].iloc[0]),
                                 "fim": str(bs["safra"].iloc[-1])}
                except Exception as e:
                    print("(percentis por safra:", type(e).__name__, e, ")")
            lbl = self.seg.feature_labels.get(feat, feat)
            print(f"Análise de '{lbl}' concluída"
                  + (f" · folha {self._leaf_label(sid)}" if sid not in (None, 'root') else "")
                  + ".")
        self.out_var_cards.value = self._var_cards_html(summ, trend)
        try:
            self.out_var_dist.value = self._fig_html(
                self.seg.plot_variable_distribution(feat, sid=sid))
        except Exception as e:
            self.out_var_dist.value = err("distribuição", e)
        if tcol and tcol in self.df.columns:
            try:
                self.out_var_time.value = self._fig_html(
                    self.seg.plot_variable_timeseries(feat, tcol, sid=sid), full_width=True)
            except Exception as e:
                self.out_var_time.value = err("série temporal", e)
            try:
                if kind == "cat":
                    self.out_var_table.value = self._styler_html(self._style_var_share(
                        self.seg.variable_share_by_safra(feat, tcol, sid=sid)), max_height="360px")
                else:
                    bs2 = bs if bs is not None else self.seg.variable_by_safra(feat, tcol, sid=sid)
                    self.out_var_table.value = self._styler_html(
                        self._style_var_safra(bs2), max_height="360px")
            except Exception as e:
                self.out_var_table.value = err("tabela por safra", e)
            try:
                if self.sample_col is not None:
                    self.out_var_psi.value = self._fig_html(
                        self.seg.plot_variable_psi_by_safra(feat, tcol, sid=sid))
                else:
                    self.out_var_psi.value = ("<div style='font-size:12px;color:#889'>PSI por "
                                              "safra requer amostras (DES/OOT).</div>")
            except Exception as e:
                self.out_var_psi.value = err("PSI por safra", e)
        else:
            self.out_var_time.value = ("<div style='font-size:12px;color:#889'>Informe a "
                                       "<b>coluna de safra</b> (ex.: dt_ref) acima para ver o "
                                       "comportamento ao longo do tempo, os percentis por safra "
                                       "e o PSI por safra.</div>")

    def _prepare_split(self):
        """Monta self._pending a partir dos controles atuais. Valida via show_grow."""
        import contextlib
        import io
        sid = self._selected_leaf()
        if sid is None:
            return False, "Nenhuma folha selecionada."
        if sid in self.locked:
            return False, "⚠ Folha fechada — reabra (🔓) para dividir."
        feature = self.dd_feature.value
        try:
            if self.tg_mode.value == "Ótimo":
                splits = None
                extra = dict(max_n_bins=self.sl_bins.value,
                             criterion=self.dd_split_criterion.value, **self._optbin_extra())
            else:
                splits, extra = self._parse_cuts(feature, sid), {}
                if not splits:
                    return False, "⚠ Preencha 'Cortes' para o modo Manual."
            with contextlib.redirect_stdout(io.StringIO()):
                self.seg.show_grow(feature, splits=splits, only_segments=[sid], **extra)
            self._pending = dict(feature=feature, splits=splits, only_segments=[sid], **extra)
            return True, None
        except Exception as e:
            self._pending = None
            return False, f"Erro ao preparar a divisão: {type(e).__name__}: {e}"

    def _on_suggest_cuts(self, _):
        """Sugere o binning ótimo da variável selecionada NESTA folha: ajusta o
        'máx. bins' e preenche os 'Cortes' (Manual) com a sugestão."""
        sid = self._selected_leaf()
        feat = self.dd_feature.value
        with self.out_log:
            self.out_log.clear_output(wait=True)
            if sid is None or sid not in self.seg.segments:
                print("Selecione uma folha."); return
            try:
                r = self.seg.best_binning(sid, feat, max_n_bins=int(self.sl_bins.max))
            except Exception as e:
                print(f"Não consegui sugerir cortes: {type(e).__name__}: {e}"); return
            lbl = self.seg.feature_labels.get(feat, feat)
            if r["n_bins"] < 2:
                print(f"Sem corte ótimo para '{lbl}' nesta folha "
                      "(variável pouco informativa aqui)."); return
            self.sl_bins.value = max(self.sl_bins.min, min(self.sl_bins.max, r["n_bins"]))
            if r["kind"] == "num":
                cuts = ", ".join(f"{c:.4g}" for c in r["cuts"])
                self.tx_cuts.value = cuts
                print(f"Sugestão p/ '{lbl}': {r['n_bins']} bins · cortes: {cuts}. "
                      "Em 'Ótimo' o máx. bins já foi ajustado; em 'Manual' os cortes foram "
                      "preenchidos. Clique em 👁 Preview.")
            else:
                grupos = " | ".join("{" + ", ".join(g) + "}" for g in r["groups"])
                print(f"Sugestão p/ '{lbl}' (categórica): {r['n_bins']} grupos: {grupos}. "
                      "No modo Ótimo o máx. bins já foi ajustado; clique em 👁 Preview.")

    def _on_preview(self, _):
        self.out_preview_seg.value = ""
        self.out_preview_chart.value = ""
        with self.out_log:
            self.out_log.clear_output(wait=True)
            ok, msg = self._prepare_split()
            if not ok:
                print(msg); return
            feature = self._pending["feature"]
            kind = self._feature_kind()
            graf = ("segmentação (em Dividir) + distribuição/cortes (ao lado do histograma)"
                    if kind == "num" else "segmentação (em Dividir)")
            print(f"Preview de '{self.seg.feature_labels.get(feature, feature)}' "
                  f"({graf}) — revise os gráficos e clique em ✂ Criar segmento.")
        p = self._pending
        sid = p["only_segments"][0]
        splits = p.get("splits")
        mnb, mbs, xbs = p.get("max_n_bins", 4), p.get("min_bin_size", 0.05), p.get("max_bin_size")
        mmd = p.get("min_mean_diff", 0.0)
        # SEGMENTAÇÃO PROPOSTA (barras repr. × PD por faixa) — dentro de "Dividir".
        try:
            self.out_preview_seg.value = self._fig_html(self.seg.plot_feature_value(
                p["feature"], sid=sid, splits=splits, max_n_bins=mnb,
                min_bin_size=mbs, max_bin_size=xbs, min_mean_diff=mmd))
        except Exception as e:
            self.out_preview_seg.value = (f"<div style='color:#b3261e;font-size:11px'>"
                                          f"(segmentação não gerada: {type(e).__name__})</div>")
        # DISTRIBUIÇÃO DA VARIÁVEL + cortes sugeridos — ao lado do histograma.
        if self._feature_kind() == "num":
            try:
                self.out_preview_chart.value = self._fig_html(self.seg.plot_feature_hist(
                    p["feature"], sid=sid, splits=splits, max_n_bins=max(mnb, 6),
                    min_bin_size=mbs, max_bin_size=xbs, min_mean_diff=mmd,
                    figsize=self._PREVIEW_FIGSIZE))
            except Exception as e:
                self.out_preview_chart.value = (f"<div style='color:#b3261e;font-size:11px'>"
                                                f"(distribuição não gerada: {type(e).__name__})</div>")
        else:
            self.out_preview_chart.value = (
                "<div style='font-size:11px;color:#889'>variável categórica — sem histograma "
                "de distribuição; veja a segmentação no card <b>Dividir a folha</b>.</div>")

    def _on_split(self, _):
        with self.out_log:
            if self._pending is None:          # sem Preview: prepara a partir dos controles
                ok, msg = self._prepare_split()
                if not ok:
                    self.out_log.clear_output(wait=True)
                    print(msg); return
            try:
                self._checkpoint()
                self.seg.grow(**self._pending)
                self._pending = None
            except Exception as e:
                print("Erro ao criar segmento:", type(e).__name__, e); return
        self._refresh()

    def _on_lock(self, _):
        sid = self._selected_leaf()
        if sid is not None:
            self.locked.add(sid)
            with self.out_log:
                print("🔒 fechada:", self._leaf_label(sid))
            # lock só muda o rótulo 🔒: atualiza árvore/dropdowns, NÃO o _refresh
            # completo (IV/PSI/metrics/tabela/PNG são idênticos após travar).
            self._refresh_lock_labels()

    def _on_unlock(self, _):
        sid = self._selected_leaf()
        if sid in self.locked:
            self.locked.discard(sid)
            with self.out_log:
                print("🔓 reaberta:", self._leaf_label(sid))
            self._refresh_lock_labels()

    def _on_prune(self, _):
        with self.out_log:
            self.out_log.clear_output(wait=True)
            try:
                self._checkpoint()
                self.seg.prune(min_repr=self.sl_repr.value, min_valor_gap=self.sl_gap.value,
                               protect=set(self.locked))
            except Exception as e:
                print("Erro na poda:", type(e).__name__, e); return
        self.locked &= set(self.seg.segments)
        self._refresh()

    def _on_reset(self, _):
        self._checkpoint()
        self.seg = TreeSegmenter(self.df, **self._kwargs)
        self.locked.clear()
        self._pending = None
        with self.out_log:
            self.out_log.clear_output(wait=True)
            print("Árvore reiniciada.")
        self._refresh()

    def _on_export(self, _):
        self.result = self.seg.assign("segmento")
        with self.out_log:
            self.out_log.clear_output(wait=True)
            print("DataFrame rotulado em  ui.result  · shape", self.result.shape)
            display(self.result["segmento_pd_nota"].value_counts().sort_index())

    def _boot_forest_html(self, bc):
        """Forest plot: barra de IC por folha + marcador do ponto (DES) e da PD OOT."""
        ref = bc.attrs.get("sample") or "todos"
        chk = bc.attrs.get("check_sample")
        lo_col, hi_col = "ic_low", "ic_high"
        ref_col = f"valor_{ref}"
        vals = []
        for _, r in bc.iterrows():
            for c in [lo_col, hi_col, ref_col] + ([f"valor_{chk}"] if chk else []):
                if c in bc and not pd.isna(r[c]):
                    vals.append(r[c])
        if not vals:
            return "<div style='color:#889'>sem dados para o gráfico</div>"
        xmin, xmax = min(vals), max(vals)
        pad = (xmax - xmin) * 0.08 or 0.02
        xmin, xmax = max(0, xmin - pad), min(1, xmax + pad)
        span = (xmax - xmin) or 1.0

        def pos(v):
            return 100 * (v - xmin) / span

        rows = ["<div style='font-family:ui-monospace,Menlo,monospace;font-size:11px'>"]
        for _, r in bc.iterrows():
            if pd.isna(r[lo_col]):
                continue
            x0, x1, xp = pos(r[lo_col]), pos(r[hi_col]), pos(r[ref_col])
            bar = (f"<div style='position:absolute;left:{x0:.1f}%;width:{max(0.5,x1-x0):.1f}%;"
                   f"top:8px;height:4px;background:#9bb7c9;border-radius:2px'></div>"
                   f"<div style='position:absolute;left:{xp:.1f}%;top:4px;width:2px;height:12px;"
                   f"background:#0f3d57' title='DES'></div>")
            ootmark = ""
            if chk and not pd.isna(r.get(f"valor_{chk}", float("nan"))):
                xo = pos(r[f"valor_{chk}"])
                inside = r.get("aderente")
                col = "#1aa64b" if inside else "#d6453e"
                ootmark = (f"<div style='position:absolute;left:{xo:.1f}%;top:3px;width:10px;"
                           f"height:10px;background:{col};border:1.5px solid #fff;border-radius:50%;"
                           f"transform:translateX(-4px)' title='{chk}'></div>")
            label = (r["descricao"][:40] + "…") if len(r["descricao"]) > 40 else r["descricao"]
            rows.append(
                f"<div style='display:flex;align-items:center;margin:3px 0'>"
                f"<div style='width:34px;color:#555'>[{r['nota']}]</div>"
                f"<div style='width:300px;color:#333;white-space:nowrap;overflow:hidden;"
                f"text-overflow:ellipsis'>{label}</div>"
                f"<div style='position:relative;flex:1;height:20px;background:#f3f6f9;"
                f"border-radius:3px'>{bar}{ootmark}</div></div>")
        leg = (f"<div style='font-size:10.5px;color:#778;margin-top:5px'>"
               f"barra cinza = IC {int(bc.attrs.get('ci',0.95)*100)}% (DES) · "
               f"traço azul = {self._risk_label} {ref} · ")
        if chk:
            leg += (f"círculo = {self._risk_label} {chk} (<span style='color:#1aa64b'>verde dentro</span> / "
                    f"<span style='color:#d6453e'>vermelho fora</span>)")
        leg += "</div>"
        rows.append(leg + "</div>")
        return "".join(rows)

    def _on_boot(self, _):
        try:
            bc = self.seg.bootstrap_ci(n_boot=int(self.sl_boot.value))
        except Exception as e:
            self.out_boot.value = (f"<div style='color:#b3261e;font-size:12px'>Erro no "
                                   f"bootstrap: {type(e).__name__}: {e}</div>")
            return

        def status_bg(v):
            if v == "dentro":
                return "background-color:#e6f6ec;color:#137a3e;font-weight:600"
            if v in ("acima", "abaixo"):
                return "background-color:#fde7e7;color:#b3261e;font-weight:600"
            return "color:#aab"
        fmt = {c: "{:.4f}" for c in bc.columns if c.startswith("pd_")}
        fmt.update({"ic_low": "{:.4f}", "ic_high": "{:.4f}", "amplitude": "{:.4f}"})
        sty = bc.style.format(fmt, na_rep="—").hide(axis="index").set_properties(
            **{"font-size": "12px"})
        if "status_oot" in bc.columns:
            sty = sty.map(status_bg, subset=["status_oot"])
        resumo = ""
        if "aderente" in bc.columns:
            n_ok = int((bc["aderente"] == True).sum())
            n_tot = int(bc["aderente"].notna().sum())
            chk = bc.attrs.get("check_sample")
            resumo = (f"<div style='font-size:12px;color:#15324a;margin:6px 0'>Aderência "
                      f"<b>{chk}</b>: {n_ok}/{n_tot} folhas com {self._risk_label} dentro do IC bootstrap "
                      f"(n_boot={bc.attrs.get('n_boot')}).</div>")
        self.out_boot.value = self._boot_forest_html(bc) + resumo + self._styler_html(sty)

    # ==================================================================
    # Diagnóstico — placar de saúde do modelo (4 vereditos)
    # ==================================================================
    def _on_diag(self, _):
        with self.out_log:
            self.out_log.clear_output(wait=True)
            try:
                html = self._diag_scorecard_html()
            except Exception as e:
                self.out_diag.value = (f"<div style='color:#b3261e;font-size:12px'>Erro ao "
                                       f"avaliar o modelo: {type(e).__name__}: {e}</div>")
                print("Erro no placar:", type(e).__name__, e); return
            print("Placar de saúde do modelo calculado.")
        self.out_diag.value = html

    def _on_diag_hide(self, _):
        self.out_diag.value = ""    # oculta/limpa a avaliação já renderizada

    def _diag_scorecard_html(self):
        """Placar de 4 vereditos (Discriminação · Estabilidade · Calibração ·
        Estrutura) + evidência estatística — reúne os testes das outras abas.
        No PD a discriminação usa AUC/Gini/KS (alvo binário)."""
        psi_hex = {"green": "#137a3e", "yellow": "#9a6b00", "red": "#b3261e"}
        bgc = {"green": "#e7f5ee", "yellow": "#fbf3e0", "red": "#fbe7e4"}
        words = {"green": "OK", "yellow": "ATENÇÃO", "red": "CRÍTICO"}

        # --- discriminação em DES: AUC/Gini (clf) ou R² (reg) ---
        met = self.seg.metrics()
        row_des = met[met["amostra"] == self.ref_sample]
        if not len(row_des):
            row_des = met[met["amostra"] == "todos"]
        if self._is_clf:
            auc = float(row_des["AUC"].iloc[0]) if len(row_des) else None
            gini = float(row_des["Gini"].iloc[0]) if (len(row_des) and "Gini" in met.columns) else None
            r2 = None
        else:
            auc = gini = None
            r2 = float(row_des["R2"].iloc[0]) if len(row_des) else None

        # --- estabilidade: pior PSI da segmentação (DES × amostras) ---
        psi_df = self.seg.psi() if self.sample_col is not None else None
        pior_psi = (float(psi_df["psi"].max())
                    if (psi_df is not None and len(psi_df)) else None)

        # --- calibração: maior |gap| previsto(DES) × observado(OOT) ---
        calib, max_gap = None, None
        if self.sample_col is not None:
            try:
                calib = self.seg.calibration_table().rename(columns={"nota": "folha"})
                if "gap" in calib.columns and calib["gap"].notna().any():
                    max_gap = float(calib["gap"].abs().max())
            except Exception:
                calib = None

        # --- estrutura: monotonicidade + distinção entre folhas-irmãs ---
        mono = self.seg.monotonicity_report()
        mono_ok = bool(mono["monotonico"].all())
        n_inv = int(mono["n_inversoes"].sum())
        try:
            lv = self.seg.leaves(with_psi=False, with_test=True, test=self.dd_test.value)
            pares = lv["p_vs_prox"].dropna() if "p_vs_prox" in lv.columns else []
            n_pares, n_indist = len(pares), int((pares > 0.05).sum()) if len(pares) else 0
        except Exception:
            n_pares = n_indist = 0

        def v_disc():
            if self._is_clf:
                if auc is None or auc != auc:
                    return "yellow", "—"
                c = "green" if auc >= 0.70 else "yellow" if auc >= 0.60 else "red"
                g = f" · Gini {gini:.3f}" if (gini is not None and gini == gini) else ""
                return c, f"AUC DES {auc:.3f}{g}"
            if r2 is None or r2 != r2:
                return "yellow", "—"
            c = "green" if r2 >= 0.5 else "yellow" if r2 >= 0.2 else "red"
            return c, f"R² DES {r2:.3f}"

        def v_estab():
            if pior_psi is None:
                return "yellow", "sem amostras"
            return self._psi_class(pior_psi), f"pior PSI {pior_psi:.3f}"

        def v_calib():
            if max_gap is None:
                return "yellow", "—"
            c = "green" if max_gap <= 0.02 else "yellow" if max_gap <= 0.05 else "red"
            return c, f"máx |gap| {max_gap:.3f}"

        def v_estrut():
            if not mono_ok:
                return "red", f"{n_inv} inversão(ões)"
            if n_pares and n_indist > 0:
                return "yellow", f"{n_indist}/{n_pares} irmãs indistintas"
            return "green", "monotônico · distintas"

        _rl = self._risk_label                       # "PD" (clf) ou "LGD" (reg)
        _obs = "taxa de default observada" if self._is_clf else f"{_rl} observado"
        _disc_q = ("o modelo separa bom × mau?" if self._is_clf
                   else "o modelo explica a variação do alvo?")
        dims = [("Discriminação", _disc_q, *v_disc()),
                ("Estabilidade", "população estável (DES→amostras)?", *v_estab()),
                ("Calibração", f"o {_rl} previsto por folha bate com o realizado?", *v_calib()),
                ("Estrutura", "folhas monotônicas e distintas?", *v_estrut())]

        def light(dim, q, c, val):
            return (f"<div class='treeui-metric' style='padding:11px 13px;border-left:4px solid "
                    f"{psi_hex[c]};background:{bgc[c]}'>"
                    f"<div class='k' style='color:{psi_hex[c]}'>{dim} · {words[c]}</div>"
                    f"<div class='v mono' style='color:{psi_hex[c]};font-size:15px'>{val}</div>"
                    f"<div style='font-size:10px;color:#6b7480;margin-top:3px'>{q}</div></div>")
        scorecard = ("<div class='treeui-metrics' style='grid-template-columns:"
                     "repeat(4,minmax(0,1fr))'>"
                     + "".join(light(*d) for d in dims) + "</div>")
        # explicação da CALIBRAÇÃO (o que o diagnóstico mede)
        _ref = self.ref_sample if self.sample_col is not None else "todos"
        _chk = (calib.attrs.get("check_sample") if calib is not None
                and hasattr(calib, "attrs") else None) or "OOT"
        calib_ajuda = (
            "<div class='treeui-legend' style='margin-top:8px'>"
            "<b>O que é calibração aqui?</b> Cada folha vira um segmento com um "
            f"<b>{_rl} previsto</b> = média do alvo na <b>{_ref}</b> (a régua). A calibração "
            f"checa se esse valor <b>se confirma fora da amostra</b>: para cada folha, compara "
            f"o {_rl} previsto (na {_ref}) com o <b>{_obs}</b> na amostra de aferição "
            f"(<b>{_chk}</b>). O <b>gap</b> = previsto − realizado por folha; o placar usa o "
            "<b>máx |gap|</b> entre as folhas: "
            "<span style='color:#137a3e'>&le;0,02 OK</span> · "
            "<span style='color:#9a6b00'>0,02–0,05 atenção</span> · "
            "<span style='color:#b3261e'>&gt;0,05 crítico</span>. "
            "Gap alto = a folha promete um risco que não se realiza (régua "
            "des-calibrada) — veja o gráfico de calibração e o backtest na aba "
            "<b>Validar &amp; Exportar</b>.</div>")

        ev = ""
        if psi_df is not None and len(psi_df):
            def bar(p):
                pos = min(max(p, 0.0) / 0.50, 1.0) * 100
                return ("<div style='position:relative;flex:1;height:8px;border-radius:5px;"
                        "background:linear-gradient(to right,#2bb673 0%,#2bb673 20%,#e6b800 20%,"
                        "#e6b800 50%,#e0584f 50%,#e0584f 100%)'>"
                        f"<div style='position:absolute;left:calc({pos:.1f}% - 1px);top:-2px;"
                        "width:2px;height:12px;background:#15324a;border-radius:1px'></div></div>")
            rows = ""
            for _, r in psi_df.iterrows():
                a = r["amostra"]; ab = "ESTAB" if a == "ESTABILIDADE" else a
                p = float(r["psi"]); cls = self._psi_class(p)
                rows += ("<div style='display:flex;align-items:center;gap:9px;margin:5px 0'>"
                         f"<div style='width:80px;font-size:11px;color:#6b7480'>PSI {ab}</div>"
                         f"<div class='mono' style='width:52px;font-size:12.5px;font-weight:600;"
                         f"color:{psi_hex[cls]}'>{p:.3f}</div>{bar(p)}"
                         f"<div style='width:62px;text-align:right;font-size:10.5px;"
                         f"color:{psi_hex[cls]}'>{r['classificacao']}</div></div>")
            ev += ("<div class='treeui-h' style='margin-top:14px'>Estabilidade · PSI da "
                   "segmentação (DES × amostras)</div>" + rows)
        if calib is not None and len(calib):
            cols = [c for c in ["folha", "n", "valor_previsto", "valor_realizado", "gap"]
                    if c in calib.columns]
            ev += (f"<div class='treeui-h' style='margin-top:14px'>Calibração · {_rl} previsto (DES) × "
                   "realizado por folha</div>" + self._df_html(calib[cols], max_height="240px",
                                                               center=True))
        # estrutura: monotonicidade + nº de inversões e QUAIS folhas invertem
        def _inv_str(inv):
            if not inv:
                return "—"
            return " · ".join(f"folha {a} ▸ folha {b}" for a, b in inv)
        mono_disp = mono[["amostra", "monotonico", "n_inversoes"]].copy()
        mono_disp["folhas que invertem"] = mono["inversoes"].apply(_inv_str)
        mono_disp = mono_disp.rename(columns={"monotonico": "monotônico",
                                              "n_inversoes": "nº inversões"})
        ev += (f"<div class='treeui-h' style='margin-top:14px'>Estrutura · monotonicidade do "
               f"{_rl} por amostra</div>"
               f"<div class='treeui-legend'>Cada inversão é um par de folhas adjacentes (pela "
               f"ordem da régua) cujo {_rl} está fora de ordem — <b>folha a ▸ folha b</b> indica "
               f"que a folha <b>a</b> tem {_rl} maior que a <b>b</b>, que deveria ser ≥.</div>"
               + self._df_html(mono_disp, center=True))
        return scorecard + calib_ajuda + ev

    # ==================================================================
    # Validação (monotonicidade · calibração · backtest) e relatório
    # ==================================================================
    def _on_validate(self, _):
        parts = []
        try:
            mr = self.seg.monotonicity_report()
            ok = bool(mr["monotonico"].all())
            parts.append("<div style='font-size:12px;margin:2px 0 6px'>"
                         + (("✅ PD monotônica crescente em todas as amostras." if self._is_clf
                             else "✅ LGD monotônico crescente em todas as amostras.")
                            if ok else "⚠️ Há inversões de monotonicidade (ver tabela).")
                         + "</div>")
            parts.append(self._df_html(mr[["amostra", "monotonico", "n_inversoes"]]))
        except Exception as e:
            parts.append(f"<div style='color:#b3261e;font-size:12px'>Erro na monotonicidade: "
                         f"{type(e).__name__}</div>")
        if self.sample_col is not None:
            try:
                parts.append("<div class='treeui-h' style='margin-top:10px'>Calibração "
                             "(prevista DES × realizada)</div>")
                parts.append(self._fig_html(self.seg.plot_calibration()))
                ct = self.seg.calibration_table().rename(columns={"nota": "folha"})
                parts.append(self._df_html(ct[["folha", "n", "valor_previsto",
                                               "valor_realizado", "gap"]]))
            except Exception as e:
                parts.append(f"<div style='color:#b3261e;font-size:12px'>Erro na calibração: "
                             f"{type(e).__name__}</div>")
        tcol = self.tx_time_col.value.strip()
        if not tcol:
            parts.append("<div style='font-size:12px;color:#889'>(informe a coluna de tempo "
                         "para o backtest)</div>")
        elif tcol not in self.df.columns:
            parts.append(f"<div style='font-size:12px;color:#889'>(coluna de tempo '{tcol}' "
                         f"não existe no DataFrame — backtest pulado)</div>")
        else:
            try:
                parts.append(f"<div class='treeui-h' style='margin-top:10px'>Backtest por "
                             f"'{tcol}'</div>")
                parts.append(self._df_html(self.seg.backtest(tcol), max_height="300px"))
            except Exception as e:
                parts.append(f"<div style='color:#b3261e;font-size:12px'>Erro no backtest: "
                             f"{type(e).__name__}</div>")
        self.out_validate.value = "".join(parts)

    def _on_report(self, _):
        with self.out_log:
            self.out_log.clear_output(wait=True)
            path = self.tx_report_path.value.strip()
            if not path:
                print("Informe o caminho do relatório (.md)."); return
            tcol = self.tx_time_col.value.strip() or None
            if tcol and tcol not in self.df.columns:
                print(f"(coluna de tempo '{tcol}' inexistente — relatório sem backtest)")
                tcol = None
            try:
                out = self.seg.validation_report(path, time_col=tcol)
                print(f"📄 relatório de validação gerado em '{out}' (imagens salvas ao lado).")
            except Exception as e:
                print("Erro ao gerar relatório:", type(e).__name__, e)

    # ==================================================================
    # Discriminação (ROC · KS) e qualidade dos segmentos
    # ==================================================================
    def _fig_html(self, fig, border=False, full_width=False):
        """Converte uma figura matplotlib em <img> base64 (string HTML).

        ``full_width=True`` faz a imagem ESTICAR até a largura do container
        (``width:100%``) em vez de só limitar (``max-width:100%``) — elimina o
        espaço em branco à direita em cartões largos."""
        import base64
        import io as _io
        buf = _io.BytesIO()
        # dpi limitado a 110 nas prévias inline (export usa save_path nos plot_*):
        # corta o PNG/base64 ~40% sem perda visual perceptível, aliviando o comm.
        buf_dpi = min(int(fig.get_dpi()), 110)
        fig.savefig(buf, format="png", dpi=buf_dpi, bbox_inches="tight")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        style = ("width:100%;height:auto" if full_width else "max-width:100%;height:auto")
        if border:
            style += ";border:1px solid #e6e8eb;border-radius:6px"
        return f"<img src='data:image/png;base64,{b64}' style='{style}'/>"

    @staticmethod
    def _styler_html(styler, max_height=None):
        """HTML de um pandas Styler, opcionalmente num container rolável."""
        html = styler.to_html()
        if max_height:
            return f"<div style='max-height:{max_height};overflow:auto'>{html}</div>"
        return html

    @staticmethod
    def _css_forca(v):                       # força do IV (forte/médio/suspeito)
        return {"forte": "color:#137a3e;font-weight:600",
                "médio": "color:#9a6b00;font-weight:600",
                "suspeito": "color:#6b3fa0;font-weight:600"}.get(v, "color:#9aa2b1")

    @staticmethod
    def _css_estab(v):                       # estabilidade (estável/atenção/instável)
        return {"estável": "color:#137a3e",
                "atenção": "color:#9a6b00;font-weight:600",
                "instável": "color:#b3261e;font-weight:600"}.get(v, "color:#9aa2b1")

    @staticmethod
    def _css_psi(v):                         # PSI numérico (verde<0.10<amarelo<0.25<vermelho)
        if pd.isna(v):
            return "color:#9aa2b1"
        a = abs(v)
        c = "#137a3e" if a < 0.10 else "#9a6b00" if a < 0.25 else "#b3261e"
        return f"color:{c};font-weight:600"

    @staticmethod
    def _css_passa(v):                       # passa no teste de hipótese (✅)
        return "color:#137a3e;font-weight:600" if str(v).strip() == "✅" else "color:#9aa2b1"

    def _df_html(self, df, max_height=None, center=False, color=False):
        """HTML de um DataFrame cru (sem índice), p/ atribuir a um widget HTML.
        Aplica bordas por célula (divisão de colunas nítida). Por padrão alinha à
        esquerda as colunas de texto; com ``center=True`` centraliza tudo
        (cabeçalho e células). Com ``color=True`` colore por NOME de coluna: força
        (IV), psi_* numéricas, psi_classificacao (estabilidade) e passa teste."""
        sty = (df.style.hide(axis="index")
                       .set_table_styles(self._TABLE_STYLES)
                       .set_properties(**{"font-size": "12px"}))
        if center:
            sty = sty.set_table_styles([{"selector": "th, td",
                                         "props": [("text-align", "center")]}],
                                       overwrite=False)
        else:
            txt_cols = [c for c in df.columns if df[c].dtype == object]
            if txt_cols:
                sty = sty.set_properties(subset=txt_cols, **{"text-align": "left"})
        if color:
            for c in df.columns:
                lc = str(c).lower()
                if lc in ("forca", "força"):
                    sty = sty.map(self._css_forca, subset=[c])
                elif lc in ("psi_classificacao", "estabilidade", "estab."):
                    sty = sty.map(self._css_estab, subset=[c])
                elif lc.startswith("psi") and pd.api.types.is_numeric_dtype(df[c]):
                    sty = sty.map(self._css_psi, subset=[c])
                elif lc in ("passa teste", "passa_teste"):
                    sty = sty.map(self._css_passa, subset=[c])
        return self._styler_html(sty, max_height)

    def _display_fig(self, fig, border=True):
        display(W.HTML(self._fig_html(fig, border=border)))

    def _on_roc(self, _):
        try:
            self.out_discrim.value = self._fig_html(self.seg.plot_roc())
        except Exception as e:
            self.out_discrim.value = (f"<div style='color:#b3261e;font-size:12px'>Erro na "
                                      f"curva ROC: {type(e).__name__}: {e}</div>")

    def _on_ks(self, _):
        try:
            self.out_discrim.value = self._fig_html(self.seg.plot_ks())
        except Exception as e:
            self.out_discrim.value = (f"<div style='color:#b3261e;font-size:12px'>Erro na "
                                      f"curva KS: {type(e).__name__}: {e}</div>")

    # plots de REGRESSÃO (alvo contínuo): dispersão e distribuição do alvo —
    # ambos renderizam em out_discrim (toggle no card de discriminação)
    def _on_box(self, _):
        try:
            self.out_discrim.value = self._fig_html(self.seg.plot_leaf_boxplots(), full_width=True)
        except Exception as e:
            self.out_discrim.value = (f"<div style='color:#b3261e;font-size:12px'>Erro no "
                                      f"boxplot: {type(e).__name__}: {e}</div>")

    def _on_hist(self, _):
        try:
            self.out_discrim.value = self._fig_html(self.seg.plot_target_hist(color="steelblue"), full_width=True)
        except Exception as e:
            self.out_discrim.value = (f"<div style='color:#b3261e;font-size:12px'>Erro no "
                                      f"histograma: {type(e).__name__}: {e}</div>")

    # ==================================================================
    # Folhas-irmãs: inversão da PD entre amostras e safras
    # ==================================================================
    def _sib_indicator_html(self, s):
        """Indicador de inversão (pílula de status + contagens + safras)."""
        pill = {"green": "pill-green", "yellow": "pill-yellow", "red": "pill-red"}[s["status"]]
        rotulo = {"green": "Sem inversão", "yellow": "Inversão em algumas safras",
                  "red": "Inversão relevante"}[s["status"]]
        nota, pdr = s["nota"], s["pd_ref"]
        ordem = " &lt; ".join(
            f"folha {nota.get(sid)} ({pdr[sid]:.1%})" if not pd.isna(pdr[sid])
            else f"folha {nota.get(sid)}" for sid in s["ordered"])
        ams_inv = [r for r in s["samples"]
                   if r["amostra"] != s["ref_sample"] and r["n_inv"] > 0]
        if ams_inv:
            am_txt = "; ".join(f"{r['amostra']}: {r['n_inv']}/{r['n_pares']} pares" for r in ams_inv)
            am_line = (f"<b>Entre amostras:</b> "
                       f"<span style='color:#b3261e'>{am_txt}</span>")
        else:
            am_line = "<b>Entre amostras:</b> <span style='color:#137a3e'>nenhuma inversão</span>"
        if s.get("safra_err"):
            sf_line = (f"<b>Entre safras:</b> <span style='color:#889'>não avaliado "
                       f"({s['safra_err']})</span>")
        elif s["n_safras"]:
            pct = 100 * s["safra_rate"]
            cor = "#b3261e" if s["safras_inv"] else "#137a3e"
            sf_line = (f"<b>Entre safras:</b> <span style='color:{cor}'>"
                       f"{s['safras_inv']}/{s['n_safras']} safras com inversão "
                       f"({pct:.0f}%)</span>")
            piores = [r for r in s["safras"] if r["n_inv"] > 0][:8]
            if piores:
                chips = " ".join(
                    f"<span style='background:#fbe7e4;color:#b23a2a;border-radius:3px;"
                    f"padding:1px 5px;font-size:10.5px' class='mono'>{r['safra']} "
                    f"({r['n_inv']})</span>" for r in piores)
                sf_line += f"<div style='margin-top:4px'>{chips}</div>"
        else:
            sf_line = "<b>Entre safras:</b> <span style='color:#889'>sem safras avaliáveis</span>"
        return (
            "<div class='treeui-card' style='margin:6px 0'>"
            f"<div style='margin-bottom:6px'><span class='pill {pill}'>● {rotulo}</span>"
            f"<span style='color:#6b7480;font-size:11.5px;margin-left:8px'>"
            f"{s['n_pairs']} par(es) de irmãs comparados</span></div>"
            f"<div style='font-size:12px;line-height:1.7'>{am_line}<br>{sf_line}</div>"
            f"<div style='font-size:11px;color:#6b7480;margin-top:6px'>"
            f"Ordem de referência ({self._risk_label} na {s['ref_sample']}): {ordem}</div>"
            "</div>")

    def _on_sib_analyze(self, _):
        pid = self.dd_sib_group.value
        if not pid:
            self.out_sib.value = ("<div style='font-size:12px;color:#889'>Nenhum grupo de "
                                  "folhas-irmãs — faça ao menos um split para criar folhas de "
                                  "mesmo pai.</div>")
            return
        tcol = (self.tx_sib_time.value or "").strip() or None
        samp = self.dd_sib_sample.value
        samp = None if samp in (None, "__all__") else samp

        def err(what, e):
            return (f"<div style='font-size:11px;color:#b3261e'>({what} não gerado: "
                    f"{type(e).__name__}: {e})</div>")

        try:
            summ = self.seg.sibling_inversion_summary(pid, time_col=tcol, sample=samp)
            ind = self._sib_indicator_html(summ)
        except Exception as e:
            ind = err("indicador de inversão", e)
        try:
            h1 = self._fig_html(self.seg.plot_sibling_value_by_sample(pid))
        except Exception as e:
            h1 = err("gráfico por amostra", e)
        try:
            h2 = self._fig_html(self.seg.plot_sibling_value_by_safra(
                pid, time_col=tcol, sample=samp))
        except Exception as e:
            h2 = err("gráfico por safra", e)
        charts = (f"<div style='display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start'>"
                  f"<div style='flex:1 1 320px;min-width:300px'>{h1}</div>"
                  f"<div style='flex:1 1 420px;min-width:340px'>{h2}</div></div>")
        self.out_sib.value = ind + charts

    # ==================================================================
    # Undo / redo de splits (e demais alterações estruturais da árvore)
    # ==================================================================
    def _snapshot(self):
        """Estado restaurável: estrutura da árvore + folhas travadas."""
        return {"segments": self.seg.to_dict()["segments"], "locked": set(self.locked)}

    def _checkpoint(self):
        """Empilha o estado atual para permitir desfazer; zera a pilha de refazer."""
        self._undo.append(self._snapshot())
        if len(self._undo) > 50:
            self._undo.pop(0)
        self._redo.clear()
        self._sync_undo_buttons()

    def _restore(self, snap):
        # registra as máscaras atuais (por condições) antes de carregar: os
        # segmentos que o undo/redo NÃO altera viram cache-hit e não são
        # recalculados via _match_conditions_pandas (o freeze do desfazer/refazer).
        self.seg._prime_mask_cache()
        self.seg._load_segments(snap["segments"])
        self.locked = set(snap["locked"]) & set(self.seg.segments)

    def _sync_undo_buttons(self):
        self.btn_undo.disabled = not self._undo
        self.btn_redo.disabled = not self._redo

    def _on_undo(self, _):
        if not self._undo:
            return
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        self._pending = None
        self._sync_undo_buttons()
        with self.out_log:
            self.out_log.clear_output(wait=True)
            print("↶ desfeito.")
        self._refresh()

    def _on_redo(self, _):
        if not self._redo:
            return
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        self._pending = None
        self._sync_undo_buttons()
        with self.out_log:
            self.out_log.clear_output(wait=True)
            print("↷ refeito.")
        self._refresh()

    # ==================================================================
    # Auto-merge: funde folhas-irmãs indistinguíveis automaticamente
    # ==================================================================
    def _on_automerge(self, _):
        import contextlib
        import io
        with self.out_log:
            self.out_log.clear_output(wait=True)
            n0 = sum(s["is_leaf"] for s in self.seg.segments.values())
            self._checkpoint()
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    self.seg.auto_merge(alpha=self.sl_alpha.value,
                                        min_valor_gap=self.sl_gap.value,
                                        test=self.dd_test.value,
                                        protect=set(self.locked),
                                        include_missing=self.cb_automerge_na.value)
            except Exception as e:
                print("Erro no auto-merge:", type(e).__name__, e)
                return
            n1 = sum(s["is_leaf"] for s in self.seg.segments.values())
            print(buf.getvalue().strip() or "Auto-merge concluído.")
            if n1 == n0:
                print("Nenhuma folha-irmã indistinguível (p > alpha) — nada a fundir. "
                      f"Aumente o alpha ou o 'Δ{self._risk_label} mínimo' para fundir mais.")
        self.locked &= set(self.seg.segments)
        self._pending = None
        self._refresh()

    def _on_pdf(self, _):
        with self.out_log:
            self.out_log.clear_output(wait=True)
            path = (self.tx_pdf_path.value or "").strip()
            if not path:
                self.out_pdf.value = "<i>Informe o caminho do .pdf.</i>"; return
            if not path.lower().endswith(".pdf"):
                path += ".pdf"
            try:
                tcol = self.date_col if (self.date_col and self.date_col in self.df.columns) else None
                self.seg.report_pdf(path, time_col=tcol)
            except Exception as e:
                self.out_pdf.value = (f"<div style='color:#b3261e;font-size:12px'>Erro ao gerar "
                                      f"PDF: {type(e).__name__}: {e}</div>")
                print(f"[pdf] erro: {e}"); return
            self.out_pdf.value = (f"<div class='treeui-legend'>✅ Relatório salvo em "
                                  f"<code>{path}</code>.</div>")
            print(f"[pdf] relatório salvo em {path}")

    # ==================================================================
    # Persistência: salvar / carregar a árvore em JSON
    # ==================================================================
    def _on_save_json(self, _):
        import json
        with self.out_log:
            self.out_log.clear_output(wait=True)
            path = self.tx_json_path.value.strip()
            if not path:
                print("Informe o caminho do arquivo .json."); return
            try:
                data = self.seg.to_dict()
                data["_ui"] = {"locked": sorted(self.locked)}
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                n = sum(s["is_leaf"] for s in self.seg.segments.values())
                print(f"💾 árvore salva em '{path}' ({n} folhas).")
            except Exception as e:
                print("Erro ao salvar:", type(e).__name__, e)

    def _on_load_json(self, _):
        import json
        import os
        with self.out_log:
            self.out_log.clear_output(wait=True)
            path = self.tx_json_path.value.strip()
            if not path:
                print("Informe o caminho do arquivo .json."); return
            if not os.path.exists(path):
                print(f"Arquivo não encontrado: '{path}'."); return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.get("meta", {})
                if meta.get("target") and meta.get("target") != self.target:
                    print(f"⚠ aviso: árvore salva com target='{meta.get('target')}', "
                          f"mas esta UI usa '{self.target}'. Carregando mesmo assim.")
                self._checkpoint()
                self.seg._load_segments(data["segments"])
                self.locked = set(data.get("_ui", {}).get("locked", [])) & set(self.seg.segments)
                self._pending = None
                n = sum(s["is_leaf"] for s in self.seg.segments.values())
                print(f"📂 árvore carregada de '{path}' ({n} folhas).")
            except Exception as e:
                print("Erro ao carregar:", type(e).__name__, e); return
        self._refresh()

    # ==================================================================
    # Imagem da árvore (matplotlib)
    # ==================================================================
    def _on_plot(self, _):
        path = self.tx_img_path.value.strip() or None
        try:
            # sem destaque da folha selecionada: todas as folhas com o mesmo estilo
            fig = self.seg.plot_tree(save_path=path)    # repr. % + alvo (DES)
            self.out_plot.value = self._fig_html(fig, border=True)
        except Exception as e:
            self.out_plot.value = (f"<div style='color:#b3261e;font-size:12px'>Erro ao "
                                   f"desenhar a árvore: {type(e).__name__}: {e}</div>")
            return
        if path:
            with self.out_log:
                self.out_log.clear_output(wait=True)
                print(f"🖼️ imagem da árvore salva em '{path}' (tamanho real).")

    def _on_plot_hide(self, _):
        self.out_plot.value = ""          # recolhe (esvazia) a imagem

    def _on_tree_preview(self, _):
        """Preview da árvore como imagem, na própria aba Construir (sem exportar).
        Sem realce da folha selecionada — a imagem mostra a árvore "neutra"."""
        try:
            self.out_tree_img.value = self._fig_html(
                self.seg.plot_tree(), border=True)
        except Exception as e:
            self.out_tree_img.value = (f"<div style='color:#b3261e;font-size:12px'>Erro ao "
                                       f"desenhar a árvore: {type(e).__name__}: {e}</div>")

    def _ipython_display_(self):
        display(self.panel)

    def display(self):
        display(self.panel)
