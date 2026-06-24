"""
LGDSegmenterUI
==============
Camada interativa (ipywidgets) sobre o `SequentialLGDSegmenter`.

Construa a árvore de segmentação de LGD clicando em botões, dentro do Jupyter,
operando sobre o DataFrame e o LGD reais. Recursos:
- árvore colorida por LGD que se atualiza a cada ação;
- **PSI ao vivo** por amostra (OOT, ESTABILIDADE, ...) no topo do painel;
- tabela de folhas com **PSI por amostra** e **p-valor** do teste entre folhas adjacentes;
- travar folhas como finais (cadeado), podar, resetar e exportar o DataFrame rotulado.

    from yggdrasil.credit_risk.lgd import LGDSegmenterUI
    ui = LGDSegmenterUI(df, target="lgd", sample_col="amostra",
                        ref_sample="DES", feature_labels=labels)
    ui
"""
from __future__ import annotations

import pandas as pd

try:
    import ipywidgets as W
    from IPython.display import display
except Exception as e:  # pragma: no cover
    raise ImportError("Este módulo requer ipywidgets e IPython (Jupyter).") from e

from .segmenter import SequentialLGDSegmenter


_CSS = """
<style>
.lgdui { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
.lgdui-banner {
  background: linear-gradient(95deg, #0f3d57 0%, #1f6f8b 60%, #2a9d8f 100%);
  color: #fff; padding: 13px 18px; border-radius: 12px; margin-bottom: 10px;
  box-shadow: 0 2px 8px rgba(15,61,87,.25);
}
.lgdui-banner .t { font-size: 17px; font-weight: 700; letter-spacing:.2px; }
.lgdui-banner .s { font-size: 12px; opacity: .82; margin-top: 2px; }
.lgdui-card {
  background: #ffffff; border: 1px solid #e6e8eb; border-radius: 11px;
  padding: 12px 14px; box-shadow: 0 1px 3px rgba(16,24,40,.05);
}
.lgdui-h { font-weight: 600; font-size: 12.5px; color: #15324a;
  text-transform: uppercase; letter-spacing: .5px; margin-bottom: 8px; }
.lgdui-bar { padding: 8px 10px; background:#f7f9fb; border:1px solid #e6e8eb;
  border-radius: 10px; }
.pill { display:inline-block; padding:4px 11px; border-radius:999px; font-size:13px;
  font-weight:600; margin:2px 5px 2px 0; }
.pill-muted  { background:#eef1f4; color:#3a4a5a; }
.pill-green  { background:#e6f6ec; color:#137a3e; }
.pill-yellow { background:#fdf3da; color:#9a6b00; }
.pill-red    { background:#fde7e7; color:#b3261e; }
.lgdui-legend { font-size:11px; color:#667; margin:6px 0 2px; }
.lgdui-tree { line-height:1.5; }
</style>
"""


class LGDSegmenterUI:
    def __init__(self, df, target="lgd", sample_col=None, ref_sample="DES",
                 feature_labels=None, features=None, tree_samples=None):
        # tree_samples: amostras cujo LGD médio aparece nas folhas da árvore.
        # None = todas; ex.: tree_samples=["DES","OOT"] mostra só DES e OOT.
        self._tree_samples_cfg = tree_samples
        self._kwargs = dict(target=target, sample_col=sample_col,
                            ref_sample=ref_sample, feature_labels=feature_labels,
                            verbose=False)
        self.df = df
        self.target = target
        self.sample_col = sample_col
        self.ref_sample = ref_sample
        if features is None:
            features = [c for c in df.columns if c not in (target, sample_col)]
        self.features = features

        self.seg = SequentialLGDSegmenter(df, **self._kwargs)
        self.locked: set = set()
        self._pending = None
        self.result = None
        self.spark_result = None      # último Spark DataFrame com a régua aplicada
        self._undo: list = []        # pilha de estados p/ desfazer splits/fusões
        self._redo: list = []        # pilha de estados p/ refazer
        self._csi_cache = None       # CSI por variável (independe da árvore)

        # máscaras de amostra (fixas) e amostras ≠ referência (ex.: OOT)
        if sample_col is not None:
            self._samples = list(df[sample_col].dropna().unique())
            self._nonref = [a for a in self._samples if a != ref_sample]
            self._sample_masks = {a: (df[sample_col] == a) for a in self._samples}
            # não-referência a EXIBIR na árvore (default: todas)
            if tree_samples is not None:
                self._tree_nonref = [a for a in tree_samples
                                     if a in self._samples and a != ref_sample]
            else:
                self._tree_nonref = list(self._nonref)
        else:
            self._samples, self._nonref, self._sample_masks = [], [], {}
            self._tree_nonref = []

        self._build()
        self._on_mode_change(None)   # estado inicial de visibilidade dos controles
        self._refresh()
        self._refresh_csi()          # CSI renderizado uma vez (não muda com a árvore)

    # ==================================================================
    # Construção dos widgets
    # ==================================================================
    def _build(self):
        full = W.Layout(width="98%")
        dstyle = {"description_width": "82px"}

        self.dd_leaf = W.Dropdown(description="Folha", layout=full, style=dstyle)
        self.dd_feature = W.Dropdown(description="Variável", options=self.features,
                                     layout=full, style=dstyle)
        self.tg_mode = W.ToggleButtons(options=["Ótimo", "Manual"], value="Ótimo",
                                       style={"button_width": "92px"})
        self.sl_bins = W.IntSlider(description="máx. bins", min=2, max=8, value=4,
                                   layout=W.Layout(width="98%"), style=dstyle)
        self.tx_cuts = W.Text(description="Cortes", layout=W.Layout(width="98%"), style=dstyle,
                              placeholder="num: 0.7,0.9  |  cat: a,b; c")

        self.sl_repr = W.FloatSlider(description="min repr%", min=0, max=10, step=0.5,
                                     value=3.0, layout=full, style=dstyle)
        self.sl_repr.tooltip = "Representatividade mínima por folha (%); abaixo disso, funde com a irmã"
        self.sl_gap = W.FloatSlider(description="ΔLGD mínimo", min=0, max=0.15, step=0.005,
                                    value=0.03, readout_format=".3f", layout=full, style=dstyle)
        self.sl_gap.tooltip = "Diferença mínima de LGD entre irmãs; abaixo disso, as duas são unidas (0.03 = 3%)"
        self.dd_test = W.Dropdown(description="Teste",
                                  options=[("Mann-Whitney", "mannwhitney"), ("Welch t", "welch")],
                                  value="mannwhitney", layout=full, style=dstyle)

        def mk(desc, style, tip, icon):
            return W.Button(description=desc, button_style=style, tooltip=tip, icon=icon,
                            layout=W.Layout(width="98%", margin="2px 0"))
        self.btn_preview = mk("Preview", "info", "Mostra LGD e representatividade (não altera)", "eye")
        self.btn_split = mk("Criar segmento", "success", "Efetiva o split na folha", "scissors")
        self.btn_lock = mk("Fechar folha", "warning", "Trava a folha (não será dividida)", "lock")
        self.btn_unlock = mk("Reabrir folha", "", "Destrava a folha", "unlock")
        self.btn_prune = mk("Podar", "danger",
                            "Funde folhas-irmãs com representatividade < min repr% ou diferença "
                            "de LGD < ΔLGD mínimo", "cut")
        self.btn_reset = mk("Reset", "", "Recomeça do zero", "refresh")
        self.btn_export = mk("Exportar", "primary", "Gera ui.result com o rótulo", "download")
        self.btn_collapse = mk("Recolher p/ o pai", "danger",
                               "Desfaz o split: recolhe a folha de volta ao segmento pai", "compress")
        self.btn_merge_l = mk("Fundir ◀ esquerda", "warning",
                              "Funde a folha com a vizinha de menor corte (num) / menor LGD (cat)", "arrow-left")
        self.btn_merge_r = mk("Fundir direita ▶", "warning",
                              "Funde a folha com a vizinha de maior corte (num) / maior LGD (cat)", "arrow-right")
        self.btn_merge_na = mk("Juntar faltante nesta folha", "warning",
                               "Junta o nó de faltantes (NaN) deste split dentro da folha populada "
                               "selecionada — a regra vira 'bin OU faltante'", "link")
        self.btn_suggest = mk("Sugerir split", "info",
                              "Recomenda a variável de maior IV para a folha selecionada", "lightbulb-o")
        self.btn_autofit = mk("Auto-fit (árvore)", "info",
                              "Constrói uma árvore gulosa por IV até a profundidade escolhida", "magic")
        self.sl_depth = W.IntSlider(description="profundidade", min=1, max=5, value=3,
                                    layout=W.Layout(width="98%"), style=dstyle)
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
                           "Calcula o IC da média de LGD por folha e a aderência em OOT", "random")
        # --- validação regulatória (monotonicidade, calibração, backtest) e relatório ---
        self.tx_time_col = W.Text(description="coluna tempo", value="dt_ref",
                                  layout=full, style=dstyle,
                                  placeholder="coluna de safra p/ o backtest (ex.: dt_ref)")
        self.btn_validate = mk("Validar (monoton. · calibração · backtest)", "info",
                               "Mostra monotonicidade das notas, calibração previsto×realizado e "
                               "backtest por safra", "check-square-o")
        self.tx_report_path = W.Text(description="relatório", value="relatorio_validacao_lgd.md",
                                     layout=full, style=dstyle, placeholder="caminho .md")
        self.btn_report = mk("Gerar relatório de validação (MD)", "success",
                             "Gera um documento Markdown com árvore, folhas, PSI, CSI, métricas, "
                             "calibração e backtest (+ imagens)", "file-text-o")
        # --- qualidade dos segmentos (dispersão, distribuição, preview da variável) ---
        self.btn_box = mk("Dispersão do LGD por folha (boxplot)", "info",
                          "Boxplot do LGD dentro de cada folha — mostra a heterogeneidade intra-folha",
                          "bar-chart")
        self.btn_hist = mk("Distribuição do LGD (carteira)", "info",
                           "Histograma do LGD — revela bimodalidade/concentração", "area-chart")

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
                                          description="auto-fundir também junta faltantes ao bin mais próximo",
                                          layout=W.Layout(width="98%"))
        self.tx_json_path = W.Text(description="arquivo", value="arvore_lgd.json",
                                   layout=full, style=dstyle, placeholder="caminho .json")
        self.btn_save_json = mk("Salvar árvore (JSON)", "success",
                                "Salva a estrutura da árvore num arquivo JSON", "save")
        self.btn_load_json = mk("Carregar árvore (JSON)", "info",
                                "Carrega uma árvore salva e reaplica ao DataFrame atual", "upload")
        # --- imagem da árvore (matplotlib) ---
        self.tx_img_path = W.Text(description="imagem", value="arvore_lgd.png",
                                  layout=full, style=dstyle,
                                  placeholder="caminho .png/.svg (opcional)")
        self.btn_plot = mk("Ver / salvar árvore (imagem)", "info",
                           "Renderiza a árvore como imagem (LGD médio e % por folha) e salva "
                           "se um caminho for informado", "picture-o")
        self.btn_plot_hide = mk("Recolher imagem", "", "Oculta a imagem da árvore", "eye-slash")
        # --- aplicar a régua numa tabela Spark ("reconstruir as folhas") ---
        self.tx_spark_in = W.Text(description="tabela", layout=full, style=dstyle,
                                  placeholder="tabela Spark de entrada (catalogo.schema.tabela)")
        self.tx_spark_out = W.Text(description="saída", layout=full, style=dstyle,
                                   placeholder="opcional: grava o resultado nesta tabela")
        self.btn_spark_apply = mk("Reconstruir folhas (Spark)", "primary",
                                  "Aplica a régua à tabela Spark (segmento, nota e LGD por linha), "
                                  "desde que as colunas tenham o mesmo nome", "table")

        self.btn_preview.on_click(self._on_preview)
        self.btn_split.on_click(self._on_split)
        self.btn_lock.on_click(self._on_lock)
        self.btn_unlock.on_click(self._on_unlock)
        self.btn_prune.on_click(self._on_prune)
        self.btn_reset.on_click(self._on_reset)
        self.btn_export.on_click(self._on_export)
        self.dd_leaf.observe(self._on_leaf_change, names="value")
        self.dd_test.observe(lambda _: self._refresh_table(), names="value")
        self.btn_collapse.on_click(self._on_collapse)
        self.btn_merge_l.on_click(lambda _: self._on_merge("left"))
        self.btn_merge_r.on_click(lambda _: self._on_merge("right"))
        self.btn_merge_na.on_click(self._on_merge_missing)
        self.btn_suggest.on_click(self._on_suggest)
        self.btn_autofit.on_click(self._on_autofit)
        self.btn_mlflow.on_click(self._on_mlflow)
        self.btn_clear_log.on_click(self._on_clear_log)
        self.btn_boot.on_click(self._on_boot)
        self.btn_validate.on_click(self._on_validate)
        self.btn_report.on_click(self._on_report)
        self.btn_box.on_click(self._on_box)
        self.btn_hist.on_click(self._on_hist)
        self.btn_undo.on_click(self._on_undo)
        self.btn_redo.on_click(self._on_redo)
        self.btn_automerge.on_click(self._on_automerge)
        self.btn_save_json.on_click(self._on_save_json)
        self.btn_load_json.on_click(self._on_load_json)
        self.btn_plot.on_click(self._on_plot)
        self.btn_plot_hide.on_click(self._on_plot_hide)
        self.btn_spark_apply.on_click(self._on_spark_apply)
        self.tg_mode.observe(self._on_mode_change, names="value")
        self.dd_feature.observe(self._on_mode_change, names="value")

        self.bar = W.HTML()
        self.out_tree = W.HTML()
        self.out_metrics = W.Output(layout=W.Layout(overflow="auto"))
        self.out_iv = W.Output(layout=W.Layout(overflow="auto"))
        self.out_csi = W.Output(layout=W.Layout(overflow="auto"))
        self.out_plot = W.Output(layout=W.Layout(overflow="auto"))
        self.out_boot = W.Output(layout=W.Layout(overflow="auto"))
        self.out_validate = W.Output(layout=W.Layout(overflow="auto"))
        self.out_quality = W.Output(layout=W.Layout(overflow="auto"))
        self.out_log = W.Output(layout=W.Layout(max_height="320px", overflow="auto"))
        self.out_preview_table = W.Output(layout=W.Layout(max_height="320px", overflow="auto"))
        self.out_preview_chart = W.Output(layout=W.Layout(max_height="360px", overflow="auto"))
        self.out_table = W.Output(layout=W.Layout(max_height="300px", overflow="auto"))
        self.cat_box = W.VBox([], layout=W.Layout(width="98%", display="none",
                                                  border="1px solid #eef1f4",
                                                  padding="6px 8px", margin="2px 0"))

        card_build = W.VBox([
            W.HTML("<div class='lgdui-h'>1 · Criar segmento</div>"),
            W.HBox([self.btn_undo, self.btn_redo]),
            self.dd_leaf, self.dd_feature, self.tg_mode, self.sl_bins, self.tx_cuts, self.cat_box,
            W.HBox([self.btn_preview, self.btn_split]),
            W.HBox([self.btn_lock, self.btn_unlock]),
            W.HBox([self.btn_collapse]),
            W.HBox([self.btn_merge_l, self.btn_merge_r]),
            W.HBox([self.btn_merge_na]),
            W.HTML("<div class='lgdui-h' style='margin-top:8px'>Assistente</div>"),
            W.HBox([self.btn_suggest, self.btn_autofit]),
            self.sl_depth,
            W.HBox([self.btn_automerge]),
            self.sl_alpha,
            self.cb_automerge_na,
        ], layout=W.Layout(width="49%"))
        card_build.add_class("lgdui-card")

        card_refine = W.VBox([
            W.HTML("<div class='lgdui-h'>2 · Podar &amp; exportar</div>"),
            W.HTML("<div class='lgdui-legend'>Poda = funde <b>folhas-irmãs</b>: une as que têm "
                   "<b>repr. &lt; min repr%</b> (imateriais) ou <b>diferença de LGD &lt; ΔLGD "
                   "mínimo</b> (ex.: 0,03 = 3%). O seletor <b>Teste</b> define o teste de hipótese "
                   "do <b>p_vs_prox</b> (tabela de folhas) e do auto-fundir.</div>"),
            self.sl_repr, self.sl_gap, self.dd_test,
            W.HBox([self.btn_prune, self.btn_reset]),
            self.btn_export,
            W.HTML("<div class='lgdui-h' style='margin-top:8px'>Salvar / carregar árvore (JSON)</div>"),
            self.tx_json_path,
            W.HBox([self.btn_save_json, self.btn_load_json]),
            W.HTML("<div class='lgdui-h' style='margin-top:8px'>Reconstruir folhas em tabela Spark</div>"),
            self.tx_spark_in, self.tx_spark_out,
            W.HBox([self.btn_spark_apply]),
            W.HTML("<div class='lgdui-h' style='margin-top:8px'>Salvar no MLflow</div>"),
            self.tx_model, self.cb_uc, self.tx_experiment, self.tx_runname,
            W.HBox([self.btn_mlflow, self.btn_clear_log]),
        ], layout=W.Layout(width="49%"))
        card_refine.add_class("lgdui-card")

        controls = W.HBox([card_build, card_refine],
                          layout=W.Layout(justify_content="space-between"))

        # Seção abaixo dos controles: log + tabela do preview + gráfico, lado a lado
        col_log = W.VBox([W.HTML("<div class='lgdui-h'>Log</div>"), self.out_log],
                         layout=W.Layout(width="27%"))
        col_ptable = W.VBox([W.HTML("<div class='lgdui-h'>Preview · tabela</div>"),
                             self.out_preview_table], layout=W.Layout(width="31%"))
        col_pchart = W.VBox([W.HTML("<div class='lgdui-h'>Preview · representatividade × LGD</div>"),
                             self.out_preview_chart], layout=W.Layout(width="40%"))
        card_preview = W.VBox([
            W.HTML("<div class='lgdui-h'>Preview / log</div>"),
            W.HTML("<div class='lgdui-legend'>Mensagens das ações (log) e — ao clicar em "
                   "👁 <b>Preview</b> — a <b>tabela</b> e o <b>gráfico</b> (barras = "
                   "representatividade, linha = LGD médio) da divisão proposta.</div>"),
            W.HBox([col_log, col_ptable, col_pchart],
                   layout=W.Layout(justify_content="space-between", align_items="flex-start")),
        ])
        card_preview.add_class("lgdui-card")

        iv_legend = W.HTML(
            "<div class='lgdui-legend'>Information Value de cada variável na <b>folha "
            "selecionada</b> (LGD binarizado pela mediana da folha) — indica qual variável "
            "melhor separa o LGD ali. ★ = maior IV. Faixas: "
            "<span style='color:#137a3e'>forte</span> · "
            "<span style='color:#9a6b00'>médio</span> · fraco/inútil · "
            "<span style='color:#6b3fa0'>suspeito</span> (alto demais, verifique vazamento)</div>")
        card_iv = W.VBox([
            W.HTML("<div class='lgdui-h'>Qual variável segmentar? (Information Value)</div>"),
            iv_legend, self.out_iv])
        card_iv.add_class("lgdui-card")

        csi_legend = W.HTML(
            "<div class='lgdui-legend'>CSI por <b>variável de entrada</b>: estabilidade da "
            "distribuição de cada característica entre a referência (DES) e as demais amostras "
            "(bins fixados no DES). Aponta <b>qual variável</b> está migrando, mesmo antes de "
            "entrar na árvore. Faixas: "
            "<span style='background:#e6f6ec;padding:1px 5px;border-radius:3px'>&lt;0.10 estável</span> "
            "<span style='background:#fdf3da;padding:1px 5px;border-radius:3px'>0.10–0.25 atenção</span> "
            "<span style='background:#fde7e7;padding:1px 5px;border-radius:3px'>&ge;0.25 instável</span></div>")
        card_csi = W.VBox([
            W.HTML("<div class='lgdui-h'>PSI por variável (CSI · DES → demais amostras)</div>"),
            csi_legend, self.out_csi])
        card_csi.add_class("lgdui-card")
        self._card_csi = card_csi

        bar_box = W.VBox([self.bar]); bar_box.add_class("lgdui-bar")

        metrics_legend = W.HTML(
            "<div class='lgdui-legend'>a régua prediz o LGD pela média do segmento na "
            "referência (DES); avaliada como modelo em cada amostra · "
            "R² alto = a segmentação explica bem a variação do LGD</div>")
        card_metrics = W.VBox([
            W.HTML("<div class='lgdui-h'>Métricas (régua como modelo de LGD)</div>"),
            metrics_legend, self.out_metrics])
        card_metrics.add_class("lgdui-card")

        boot_legend = W.HTML(
            "<div class='lgdui-legend'>IC da média de LGD por folha via bootstrap na "
            "referência (DES). Se houver OOT, mostra o LGD de OOT e verifica a "
            "<b>aderência</b>: <span style='color:#137a3e'>dentro</span> do IC = estável; "
            "<span style='color:#b3261e'>acima/abaixo</span> = LGD deslocou além da incerteza "
            "amostral. Calcule quando a árvore estiver fechada.</div>")
        card_boot = W.VBox([
            W.HTML("<div class='lgdui-h'>Intervalos de confiança (bootstrap) &amp; aderência OOT</div>"),
            boot_legend,
            W.HBox([self.sl_boot, self.btn_boot],
                   layout=W.Layout(align_items="center")),
            self.out_boot])
        card_boot.add_class("lgdui-card")

        quality_legend = W.HTML(
            "<div class='lgdui-legend'>Qualidade dos segmentos: <b>boxplot</b> da dispersão do LGD "
            "dentro de cada folha (caixa estreita = folha homogênea; larga = separa pouco) e "
            "<b>histograma</b> do LGD da carteira (bimodalidade). "
            "<i>O LGD por faixa da variável aparece embaixo da tabela ao clicar em 👁 Preview.</i></div>")
        card_quality = W.VBox([
            W.HTML("<div class='lgdui-h'>Qualidade dos segmentos</div>"),
            quality_legend,
            W.HBox([self.btn_box, self.btn_hist]),
            self.out_quality,
        ])
        card_quality.add_class("lgdui-card")
        self._card_quality = card_quality

        valid_legend = W.HTML(
            "<div class='lgdui-legend'>Validação regulatória: <b>monotonicidade</b> do LGD nas "
            "notas (DES e demais amostras), <b>calibração</b> previsto (DES) × realizado (OOT) por "
            "folha, e <b>backtest</b> do LGD previsto × realizado por safra (informe a coluna de "
            "tempo). O <b>relatório</b> reúne tudo num Markdown com as imagens.</div>")
        card_validacao = W.VBox([
            W.HTML("<div class='lgdui-h'>Validação &amp; relatório (monotonicidade · calibração · backtest)</div>"),
            valid_legend,
            W.HBox([self.tx_time_col, self.btn_validate],
                   layout=W.Layout(align_items="center")),
            self.out_validate,
            W.HTML("<div class='lgdui-h' style='margin-top:8px'>Relatório de validação (Markdown)</div>"),
            W.HBox([self.tx_report_path, self.btn_report],
                   layout=W.Layout(align_items="center")),
        ])
        card_validacao.add_class("lgdui-card")
        self._card_validacao = card_validacao

        tree_legend = W.HTML(
            "<div class='lgdui-legend'>cor do quadrado = LGD "
            "(<span style='color:#1aa64b'>baixo</span> &rarr; "
            "<span style='color:#caa000'>médio</span> &rarr; "
            "<span style='color:#d6453e'>alto</span>) · 🔒 folha fechada</div>")
        card_tree = W.VBox([W.HTML("<div class='lgdui-h'>Árvore atual</div>"),
                            tree_legend, self.out_tree,
                            W.HTML("<div class='lgdui-h' style='margin-top:8px'>Imagem da árvore "
                                   "(LGD médio &amp; % por folha)</div>"),
                            self.tx_img_path,
                            W.HBox([self.btn_plot, self.btn_plot_hide]), self.out_plot])
        card_tree.add_class("lgdui-card")

        tbl_legend = W.HTML(
            "<div class='lgdui-legend'>"
            "<b>PSI por amostra</b> (estabilidade da folha entre DES e a amostra): "
            "<span style='background:#e6f6ec;padding:1px 5px;border-radius:3px'>&lt;0.10 estável</span> "
            "<span style='background:#fdf3da;padding:1px 5px;border-radius:3px'>0.10–0.25 atenção</span> "
            "<span style='background:#fde7e7;padding:1px 5px;border-radius:3px'>&ge;0.25 instável</span>"
            "<br><b>p_vs_prox</b> = p-valor de um <b>teste de hipótese</b> que compara a "
            "<b>distribuição de LGD</b> da folha com a da <b>irmã adjacente</b> (mesmo pai, "
            "na amostra de referência DES). H₀: as duas irmãs têm o mesmo LGD. "
            "O teste é o <b>Mann-Whitney U</b> (não-paramétrico, padrão) ou o <b>t de Welch</b> "
            "(médias, variâncias desiguais) — escolha no seletor <b>Teste</b>. "
            "<span style='background:#fde7e7;padding:1px 5px;border-radius:3px'>p alto (&gt;0,05, em vermelho)</span> "
            "⇒ <b>não</b> dá para distinguir as irmãs ⇒ candidatas a fusão; "
            "<span style='color:#137a3e'>p baixo</span> ⇒ folhas bem separadas. "
            "Só <b>irmãs</b> são comparadas (a última de cada grupo e o nó de faltantes ficam em branco).</div>")
        card_table = W.VBox([W.HTML("<div class='lgdui-h'>Folhas criadas · PSI &amp; teste de hipótese (irmãs)</div>"),
                             tbl_legend, self.out_table])
        card_table.add_class("lgdui-card")

        banner = W.HTML(_CSS +
            "<div class='lgdui-banner'><div class='t'>🌳 Construtor de Segmentação de LGD</div>"
            "<div class='s'>Optimal binning híbrido · PSI ao vivo (DES como referência) · "
            "teste de hipótese entre folhas adjacentes</div></div>")

        # Information Value (importância) e PSI por variável (CSI) lado a lado,
        # separados por uma barra vertical intermediária.
        if self.sample_col is not None:        # CSI/PSI exige amostras (DES vs OOT/…)
            card_iv.layout.width = "48.5%"
            card_csi.layout.width = "48.5%"
            barra = W.HTML("<div style='width:2px;background:#d6dbe0;border-radius:1px;"
                           "align-self:stretch;min-height:160px;margin:0 4px'></div>")
            card_imp_psi = W.HBox([card_iv, barra, card_csi],
                                  layout=W.Layout(width="100%", align_items="stretch",
                                                  justify_content="space-between"))
        else:
            card_imp_psi = card_iv             # sem amostras não há PSI por variável

        # Ordem: segmentação → folhas → IV | PSI → métricas → bootstrap →
        #        qualidade dos segmentos → validação
        cards = [banner, controls, card_preview, bar_box, card_tree, card_table,
                 card_imp_psi, card_metrics, card_boot, card_quality, card_validacao]
        self.panel = W.VBox(cards)
        self.panel.add_class("lgdui")

    # ==================================================================
    # Render
    # ==================================================================
    @staticmethod
    def _color(lgd, lo, hi):
        if hi <= lo or pd.isna(lgd):
            t = 0.5
        else:
            t = max(0.0, min(1.0, (lgd - lo) / (hi - lo)))
        r = int(40 + (214 - 40) * min(1, 2 * t))
        g = int(166 - (166 - 69) * max(0, 2 * t - 1)) if t > 0.5 else 166
        return f"rgb({r},{g},69)"

    def _node_lgd(self, sid, sample=None):
        m = self.seg.segments[sid]["mask"]
        if sample is not None and sample in self._sample_masks:
            m = m & self._sample_masks[sample]
        sub = self.df[m]
        return sub[self.target].mean() if len(sub) else float("nan")

    def _leaf_lgds(self):
        ref = self.ref_sample if self.sample_col is not None else None
        vals = [self._node_lgd(sid, ref)
                for sid, s in self.seg.segments.items() if s["is_leaf"]]
        vals = [v for v in vals if not pd.isna(v)]
        return (min(vals), max(vals)) if vals else (0.0, 1.0)

    @staticmethod
    def _psi_class(p):
        return "green" if p < 0.10 else "yellow" if p < 0.25 else "red"

    def _status_html(self):
        seg = self.seg
        n_folhas = sum(s["is_leaf"] for s in seg.segments.values())
        prof = max(s["depth"] for s in seg.segments.values())
        n_lock = len(self.locked & {sid for sid, s in seg.segments.items() if s["is_leaf"]})
        pills = [f"<span class='pill pill-muted'>Folhas {n_folhas}</span>",
                 f"<span class='pill pill-muted'>Profundidade {prof}</span>",
                 f"<span class='pill pill-muted'>Fechadas {n_lock}</span>"]
        if self.sample_col is not None and n_folhas >= 1:
            try:
                for _, r in seg.psi().iterrows():
                    cls = self._psi_class(r["psi"])
                    pills.append(f"<span class='pill pill-{cls}'>PSI {r['amostra']} "
                                 f"{r['psi']:.3f} · {r['classificacao']}</span>")
            except Exception:
                pass
        # badges de R² (régua como modelo)
        try:
            for _, r in seg.metrics().iterrows():
                r2 = r["R2"]
                cls = "muted" if pd.isna(r2) else ("green" if r2 >= 0.5 else "yellow" if r2 >= 0.2 else "red")
                txt = "—" if pd.isna(r2) else f"{r2:.3f}"
                pills.append(f"<span class='pill pill-{cls}'>R² {r['amostra']} {txt}</span>")
        except Exception:
            pass
        return "<div>" + "".join(pills) + "</div>"

    def _tree_html(self):
        seg = self.seg
        filhos: dict = {}
        for sid, s in seg.segments.items():
            filhos.setdefault(s["parent"], []).append(sid)
        nota_map, _ = seg._grade_map()
        lo, hi = self._leaf_lgds()
        n_total = len(self.df)
        rows = []

        def stat(sid):
            sub = self.df[seg.segments[sid]["mask"]]
            return len(sub), 100 * len(sub) / n_total

        def lgd_str(sid):
            if self.sample_col is not None:
                parts = [f"{self.ref_sample} {self._node_lgd(sid, self.ref_sample):.3f}"]
                for a in self._tree_nonref:          # só as amostras configuradas
                    parts.append(f"{a} {self._node_lgd(sid, a):.3f}")
                return " · ".join(parts)
            return f"LGD {self._node_lgd(sid):.3f}"

        def lgd_of(sid):
            sub = self.df[seg.segments[sid]["mask"]]
            return sub[self.target].mean() if len(sub) else float("inf")

        def rotulo(sid):
            s = seg.segments[sid]
            return "TODA A CARTEIRA" if s["parent"] is None else seg._descrever([s["conditions"][-1]])

        def rec(sid, prefix, is_last, is_root):
            n, rep = stat(sid)
            s = seg.segments[sid]
            conn = "" if is_root else ("└─ " if is_last else "├─ ")
            ref = self.ref_sample if self.sample_col is not None else None
            color = self._color(self._node_lgd(sid, ref), lo, hi)
            sw = (f"<span style='display:inline-block;width:11px;height:11px;background:{color};"
                  f"border-radius:2px;vertical-align:middle;margin:0 5px'></span>")
            tags = ""
            if s["is_leaf"]:
                tags += f" · <b>nota {nota_map.get(sid, '?')}</b>"
                if sid in self.locked:
                    tags += " 🔒"
            sel = "background:#fff3cd;border-radius:3px;" if (
                s["is_leaf"] and sid == self.dd_leaf.value) else ""
            rows.append(
                f"<div style='{sel}white-space:pre;font-family:ui-monospace,Menlo,monospace;"
                f"font-size:12px;padding:1px 2px'>{prefix}{conn}{sw}{rotulo(sid)}"
                f"<span style='color:#8a97a3'>  (n={n}, {rep:.1f}% · {lgd_str(sid)})</span>{tags}</div>")
            ch = sorted(filhos.get(sid, []), key=lgd_of)
            for i, c in enumerate(ch):
                child_prefix = "" if is_root else prefix + ("   " if is_last else "│  ")
                rec(c, child_prefix, i == len(ch) - 1, False)

        rec("root", "", True, True)
        return "<div class='lgdui-tree'>" + "".join(rows) + "</div>"

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
        fmt = {"repr_%": "{:.1f}", "lgd_medio": "{:.4f}"}
        for c in lv.columns:
            if c.startswith("psi_") or c.startswith("lgd_"):
                fmt[c] = "{:.4f}"
        if "p_vs_prox" in lv.columns:
            fmt["p_vs_prox"] = "{:.3f}"
        return (sty.format(fmt, na_rep="—")
                   .hide(axis="index")
                   .set_properties(**{"font-size": "12px"}))

    def _leaf_label(self, sid):
        s = self.seg.segments[sid]
        txt = "TODA A CARTEIRA" if s["parent"] is None else self.seg._descrever(s["conditions"])
        if len(txt) > 72:
            txt = txt[:69] + "…"
        return ("🔒 " if sid in self.locked else "") + txt

    def _refresh_table(self):
        with self.out_table:
            self.out_table.clear_output(wait=True)
            lv = self.seg.leaves(with_psi=True, with_test=True, test=self.dd_test.value)
            cols = ["nota_lgd", "descricao", "repr_%", "lgd_medio"]
            cols += [c for c in lv.columns if c.startswith("psi_")]
            if "p_vs_prox" in lv.columns:
                cols.append("p_vs_prox")
            display(self._style_leaves(lv[cols]))

    def _refresh_metrics(self):
        with self.out_metrics:
            self.out_metrics.clear_output(wait=True)
            m = self.seg.metrics()

            def r2_bg(v):
                if pd.isna(v):
                    return "color:#aab"
                c = "#e6f6ec" if v >= 0.5 else "#fdf3da" if v >= 0.2 else "#fde7e7"
                return f"background-color:{c};font-weight:600"
            sty = (m.style
                   .map(r2_bg, subset=["R2"])
                   .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}"}, na_rep="—")
                   .hide(axis="index")
                   .set_properties(**{"font-size": "13.5px"}))
            display(sty)

    def _ordered_leaf_options(self):
        """Opções do dropdown na MESMA ordem da árvore (DFS, filhos por LGD),
        com indentação por profundidade e a nota — fácil de localizar."""
        seg = self.seg
        filhos: dict = {}
        for sid, s in seg.segments.items():
            filhos.setdefault(s["parent"], []).append(sid)
        nota_map, _ = seg._grade_map()
        n_total = len(self.df)

        def lgd_of(sid):
            sub = self.df[seg.segments[sid]["mask"]]
            return sub[self.target].mean() if len(sub) else float("inf")

        opts = []

        def rec(sid, depth):
            s = seg.segments[sid]
            if s["is_leaf"]:
                own = ("TODA A CARTEIRA" if s["parent"] is None
                       else seg._descrever([s["conditions"][-1]]))
                if len(own) > 46:
                    own = own[:43] + "…"
                rep = 100 * s["mask"].sum() / n_total
                indent = "· " * depth
                lock = "🔒 " if sid in self.locked else ""
                nota = nota_map.get(sid, "?")
                label = f"[{nota:>2}] {indent}{lock}{own}  (LGD {lgd_of(sid):.3f} · {rep:.0f}%)"
                opts.append((label, sid))
            for c in sorted(filhos.get(sid, []), key=lgd_of):
                rec(c, depth + 1)

        rec("root", 0)
        return opts

    def _refresh(self):
        opts = self._ordered_leaf_options()
        leaves = [sid for _, sid in opts]
        cur = self.dd_leaf.value
        self.dd_leaf.options = opts
        if cur in leaves:
            self.dd_leaf.value = cur
        elif opts:
            self.dd_leaf.value = opts[0][1]

        self.bar.value = self._status_html()
        self.out_tree.value = self._tree_html()
        self._refresh_iv()
        self._refresh_metrics()
        self._refresh_table()
        # o IC bootstrap e a imagem ficam obsoletos após mudanças na árvore
        with self.out_boot:
            self.out_boot.clear_output(wait=True)
            display(W.HTML("<div style='font-size:12px;color:#889'>Árvore alterada — "
                           "clique em <b>Calcular IC bootstrap</b> para (re)calcular.</div>"))
        with self.out_plot:
            self.out_plot.clear_output(wait=True)
            display(W.HTML("<div style='font-size:12px;color:#889'>Árvore alterada — "
                           "clique em <b>Ver / salvar árvore (imagem)</b> para renderizar.</div>"))

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
        """Mostra o controle certo conforme modo e tipo da variável."""
        manual = self.tg_mode.value == "Manual"
        cat = self._feature_kind() == "cat"
        self.sl_bins.layout.display = "none" if manual else ""           # máx. bins: só Ótimo
        self.tx_cuts.layout.display = "" if (manual and not cat) else "none"   # cortes: Manual numérico
        self.cat_box.layout.display = "" if (manual and cat) else "none"      # grupos: Manual categórico
        if manual and cat:
            self._rebuild_cat_box()

    def _rebuild_cat_box(self):
        """Monta um seletor de grupo por categoria presente na folha (ordenadas por LGD)."""
        sid = self.dd_leaf.value
        feat = self.dd_feature.value
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
                       "Categorias no <b>mesmo grupo</b> viram um nó. Ordenadas por LGD. "
                       "Faltantes (NaN) já viram um nó próprio.</div>")]
        for k, c in enumerate(order, 1):
            dd = W.Dropdown(options=[(f"grupo {g}", g) for g in range(1, n + 1)], value=k,
                            layout=W.Layout(width="110px"))
            self._cat_widgets[c] = dd
            lab = W.HTML(f"<span style='font-size:12px'><b>{c}</b>"
                         f"<span style='color:#889'> · LGD {means[c]:.3f}</span></span>")
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
            if sid is None:
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

    def _on_autofit(self, _):
        with self.out_log:
            self.out_log.clear_output(wait=True)
            print("Construindo árvore automática…")
            self._checkpoint()
            self.seg.fit_auto(max_depth=int(self.sl_depth.value))
        self.locked.clear()
        self._pending = None
        self._refresh()
        with self.out_log:
            n = sum(s["is_leaf"] for s in self.seg.segments.values())
            print(f"Auto-fit concluído: {n} folhas (profundidade ≤ {int(self.sl_depth.value)}). "
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
                    print(f"✓ tabela '{out_name}' gravada (segmento_lgd, nota_lgd, lgd_regua).")
                except Exception as e:
                    print(f"Régua aplicada, mas falhou ao gravar '{out_name}':",
                          type(e).__name__, e)
            print(f"✓ régua aplicada em '{name}'. Spark DataFrame em  ui.spark_result.")
            try:
                dist = out.groupBy("nota_lgd").count().orderBy("nota_lgd").toPandas()
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
        # categórico: grupos vêm dos seletores por categoria
        grupos = self._cat_groups()
        return grupos if grupos else None

    def _on_leaf_change(self, _):
        self.out_tree.value = self._tree_html()
        self._refresh_iv()
        self._on_mode_change(None)   # recompõe os grupos categóricos para a nova folha

    def _refresh_iv(self):
        with self.out_iv:
            self.out_iv.clear_output(wait=True)
            sid = self.dd_leaf.value
            iv = self.seg.variable_iv(sid)
            cut = iv.attrs.get("cutoff")
            disp = iv.copy()
            disp["variavel"] = disp["variavel"].map(
                lambda v: self.seg.feature_labels.get(v, v))
            if len(disp):
                disp.loc[0, "variavel"] = "★ " + str(disp.loc[0, "variavel"])

            def forca_bg(v):
                return {
                    "forte": "background-color:#e6f6ec;color:#137a3e;font-weight:600",
                    "médio": "background-color:#fdf3da;color:#9a6b00;font-weight:600",
                    "suspeito": "background-color:#efe7fb;color:#6b3fa0;font-weight:600",
                }.get(v, "color:#8a97a3")

            sty = (disp.style.format({"iv": "{:.4f}"}, na_rep="—")
                   .hide(axis="index")
                   .map(forca_bg, subset=["forca"])
                   .set_properties(**{"font-size": "12px"}))
            qual = "TODA A CARTEIRA" if (sid in (None, "root")) else self._leaf_label(sid)
            hint = f"<div style='font-size:11px;color:#667;margin-bottom:4px'>folha: " \
                   f"<b>{qual}</b> · corte do LGD (mediana) = {cut} · evento = LGD ≥ corte</div>"
            display(W.HTML(hint))
            display(sty)

    def _on_preview(self, _):
        sid = self._selected_leaf()
        self.out_preview_table.clear_output()
        self.out_preview_chart.clear_output()
        previews = None
        with self.out_log:
            self.out_log.clear_output(wait=True)
            if sid is None:
                print("Nenhuma folha selecionada."); return
            if sid in self.locked:
                print("⚠ Folha fechada — reabra (🔓) para dividir."); return
            feature = self.dd_feature.value
            try:
                if self.tg_mode.value == "Ótimo":
                    splits, extra = None, dict(max_n_bins=self.sl_bins.value)
                else:
                    splits, extra = self._parse_cuts(feature, sid), {}
                    if not splits:
                        print("⚠ Preencha 'Cortes' para o modo Manual."); return
                previews = self.seg.show_grow(feature, splits=splits, only_segments=[sid], **extra)
                self._pending = dict(feature=feature, splits=splits, only_segments=[sid], **extra)
            except Exception as e:
                self._pending = None
                print("Erro no preview:", type(e).__name__, e); return
        if previews and sid in previews:
            drop = [c for c in ["lgd_std"] if c in previews[sid]]
            with self.out_preview_table:                 # tabela na 2ª coluna
                display(previews[sid].drop(columns=drop))
            with self.out_preview_chart:                 # gráfico na 3ª coluna
                try:
                    fig = self.seg.plot_feature_lgd(
                        feature, sid=sid, splits=splits,
                        max_n_bins=extra.get("max_n_bins", 4))
                    self._display_fig(fig, border=False)
                except Exception as e:
                    print("(gráfico não gerado:", type(e).__name__, e, ")")

    def _on_split(self, _):
        with self.out_log:
            if self._pending is None:
                print("Rode o 👁 Preview antes de criar o segmento."); return
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
            self._refresh()

    def _on_unlock(self, _):
        sid = self._selected_leaf()
        if sid in self.locked:
            self.locked.discard(sid)
            with self.out_log:
                print("🔓 reaberta:", self._leaf_label(sid))
            self._refresh()

    def _on_prune(self, _):
        with self.out_log:
            self.out_log.clear_output(wait=True)
            try:
                self._checkpoint()
                self.seg.prune(min_repr=self.sl_repr.value, min_lgd_gap=self.sl_gap.value,
                               protect=set(self.locked))
            except Exception as e:
                print("Erro na poda:", type(e).__name__, e); return
        self.locked &= set(self.seg.segments)
        self._refresh()

    def _on_reset(self, _):
        self._checkpoint()
        self.seg = SequentialLGDSegmenter(self.df, **self._kwargs)
        self.locked.clear()
        self._pending = None
        with self.out_log:
            self.out_log.clear_output(wait=True)
            print("Árvore reiniciada.")
        self._refresh()

    def _on_export(self, _):
        self.result = self.seg.assign("segmento_lgd")
        with self.out_log:
            self.out_log.clear_output(wait=True)
            print("DataFrame rotulado em  ui.result  · shape", self.result.shape)
            display(self.result["segmento_lgd_nota"].value_counts().sort_index())

    def _boot_forest_html(self, bc):
        """Forest plot: barra de IC por folha + marcador do ponto (DES) e do LGD OOT."""
        ref = bc.attrs.get("sample") or "todos"
        chk = bc.attrs.get("check_sample")
        lo_col, hi_col = "ic_low", "ic_high"
        ref_col = f"lgd_{ref}"
        # escala comum
        vals = []
        for _, r in bc.iterrows():
            for c in [lo_col, hi_col, ref_col] + ([f"lgd_{chk}"] if chk else []):
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
            if chk and not pd.isna(r.get(f"lgd_{chk}", float("nan"))):
                xo = pos(r[f"lgd_{chk}"])
                inside = r.get("aderente")
                col = "#1aa64b" if inside else "#d6453e"
                ootmark = (f"<div style='position:absolute;left:{xo:.1f}%;top:3px;width:10px;"
                           f"height:10px;background:{col};border:1.5px solid #fff;border-radius:50%;"
                           f"transform:translateX(-4px)' title='{chk}'></div>")
            label = (r["descricao"][:40] + "…") if len(r["descricao"]) > 40 else r["descricao"]
            rows.append(
                f"<div style='display:flex;align-items:center;margin:3px 0'>"
                f"<div style='width:34px;color:#555'>[{r['nota_lgd']}]</div>"
                f"<div style='width:300px;color:#333;white-space:nowrap;overflow:hidden;"
                f"text-overflow:ellipsis'>{label}</div>"
                f"<div style='position:relative;flex:1;height:20px;background:#f3f6f9;"
                f"border-radius:3px'>{bar}{ootmark}</div></div>")
        leg = (f"<div style='font-size:10.5px;color:#778;margin-top:5px'>"
               f"barra cinza = IC {int(bc.attrs.get('ci',0.95)*100)}% (DES) · "
               f"traço azul = LGD {ref} · ")
        if chk:
            leg += (f"círculo = LGD {chk} (<span style='color:#1aa64b'>verde dentro</span> / "
                    f"<span style='color:#d6453e'>vermelho fora</span>)")
        leg += "</div>"
        rows.append(leg + "</div>")
        return "".join(rows)

    def _on_boot(self, _):
        with self.out_boot:
            self.out_boot.clear_output(wait=True)
            try:
                bc = self.seg.bootstrap_ci(n_boot=int(self.sl_boot.value))
            except Exception as e:
                print("Erro no bootstrap:", type(e).__name__, e); return

            # forest plot
            display(W.HTML(self._boot_forest_html(bc)))

            # tabela estilizada com aderência
            def status_bg(v):
                if v == "dentro":
                    return "background-color:#e6f6ec;color:#137a3e;font-weight:600"
                if v in ("acima", "abaixo"):
                    return "background-color:#fde7e7;color:#b3261e;font-weight:600"
                return "color:#aab"
            fmt = {c: "{:.4f}" for c in bc.columns if c.startswith("lgd_")}
            fmt.update({"ic_low": "{:.4f}", "ic_high": "{:.4f}", "amplitude": "{:.4f}"})
            sty = bc.style.format(fmt, na_rep="—").hide(axis="index").set_properties(
                **{"font-size": "12px"})
            if "status_oot" in bc.columns:
                sty = sty.map(status_bg, subset=["status_oot"])
            if "aderente" in bc.columns:
                n_ok = int((bc["aderente"] == True).sum())
                n_tot = int(bc["aderente"].notna().sum())
                chk = bc.attrs.get("check_sample")
                print(f"Aderência {chk}: {n_ok}/{n_tot} folhas com LGD dentro do IC "
                      f"bootstrap (n_boot={bc.attrs.get('n_boot')}).")
            display(sty)

    # ==================================================================
    # Validação (monotonicidade · calibração · backtest) e relatório
    # ==================================================================
    def _on_validate(self, _):
        with self.out_validate:
            self.out_validate.clear_output(wait=True)
            # monotonicidade
            try:
                mr = self.seg.monotonicity_report()
                ok = bool(mr["monotonico"].all())
                print("✅ LGD monotônico crescente em todas as amostras."
                      if ok else "⚠️ Há inversões de monotonicidade (ver tabela).")
                display(mr[["amostra", "monotonico", "n_inversoes"]])
            except Exception as e:
                print("Erro na monotonicidade:", type(e).__name__, e)
            # calibração (previsto DES × realizado OOT)
            if self.sample_col is not None:
                try:
                    self._display_fig(self.seg.plot_calibration(), border=False)
                    ct = self.seg.calibration_table()
                    display(ct[["nota_lgd", "n", "lgd_previsto", "lgd_realizado", "gap"]])
                except Exception as e:
                    print("Erro na calibração:", type(e).__name__, e)
            # backtest por safra
            tcol = self.tx_time_col.value.strip()
            if not tcol:
                print("(informe a coluna de tempo para o backtest)")
            elif tcol not in self.df.columns:
                print(f"(coluna de tempo '{tcol}' não existe no DataFrame — backtest pulado)")
            else:
                try:
                    print(f"\nBacktest por '{tcol}':")
                    display(self.seg.backtest(tcol))
                except Exception as e:
                    print("Erro no backtest:", type(e).__name__, e)

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
    # Qualidade dos segmentos (dispersão · distribuição · preview da variável)
    # ==================================================================
    def _display_fig(self, fig, border=True):
        """Exibe uma figura matplotlib escalada para caber no painel (a versão
        salva em arquivo continua em tamanho real). Evita a imagem ser cortada
        quando a árvore tem muitas folhas."""
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
        display(W.HTML(f"<img src='data:image/png;base64,{b64}' style='{style}'/>"))

    def _on_box(self, _):
        with self.out_quality:
            self.out_quality.clear_output(wait=True)
            try:
                self._display_fig(self.seg.plot_leaf_boxplots(), border=False)
            except Exception as e:
                print("Erro no boxplot:", type(e).__name__, e)

    def _on_hist(self, _):
        with self.out_quality:
            self.out_quality.clear_output(wait=True)
            try:
                # só a amostra de referência (DES), preenchido
                self._display_fig(self.seg.plot_target_hist(), border=False)
            except Exception as e:
                print("Erro no histograma:", type(e).__name__, e)

    # ==================================================================
    # CSI por variável (estabilidade das entradas) — independe da árvore,
    # por isso é calculado uma vez e cacheado.
    # ==================================================================
    def _refresh_csi(self):
        if self.sample_col is None:
            return
        with self.out_csi:
            self.out_csi.clear_output(wait=True)
            if self._csi_cache is None:
                try:
                    self._csi_cache = self.seg.csi(features=self.features)
                except Exception as e:
                    display(W.HTML("<div style='font-size:12px;color:#b3261e'>"
                                   f"Erro ao calcular CSI: {type(e).__name__}: {e}</div>"))
                    return
            disp = self._csi_cache.copy()
            disp["variavel"] = disp["variavel"].map(
                lambda v: self.seg.feature_labels.get(v, v))
            csi_cols = [c for c in disp.columns
                        if c.startswith("csi_") or c == "pior_csi"]

            def csi_bg(v):
                if pd.isna(v):
                    return "color:#aab"
                a = abs(v)
                c = "#e6f6ec" if a < 0.10 else "#fdf3da" if a < 0.25 else "#fde7e7"
                return f"background-color:{c};font-weight:600"

            def cls_bg(v):
                return {
                    "estável": "color:#137a3e", "atenção": "color:#9a6b00;font-weight:600",
                    "instável": "background-color:#fde7e7;color:#b3261e;font-weight:600",
                }.get(v, "color:#8a97a3")

            sty = disp.style
            for c in csi_cols:
                sty = sty.map(csi_bg, subset=[c])
            if "classificacao" in disp.columns:
                sty = sty.map(cls_bg, subset=["classificacao"])
            sty = (sty.format({c: "{:.4f}" for c in csi_cols}, na_rep="—")
                      .hide(axis="index").set_properties(**{"font-size": "12px"}))
            display(sty)

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
                                        min_lgd_gap=self.sl_gap.value,
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
                      "Aumente o alpha ou o 'min gap LGD' para fundir mais.")
        self.locked &= set(self.seg.segments)
        self._pending = None
        self._refresh()

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
        with self.out_plot:
            self.out_plot.clear_output(wait=True)
            path = self.tx_img_path.value.strip() or None
            try:
                # arquivo salvo em tamanho real; exibição escalada para caber
                fig = self.seg.plot_tree(save_path=path)   # repr. % + LGD (DES)
                self._display_fig(fig)
            except Exception as e:
                print("Erro ao desenhar a árvore:", type(e).__name__, e)
                return
        if path:
            with self.out_log:
                self.out_log.clear_output(wait=True)
                print(f"🖼️ imagem da árvore salva em '{path}' (tamanho real).")

    def _on_plot_hide(self, _):
        self.out_plot.clear_output()      # recolhe (esvazia) a imagem

    def _ipython_display_(self):
        display(self.panel)

    def display(self):
        display(self.panel)
