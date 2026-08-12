"""
TreeSegmenterUI
===============
Camada interativa (ipywidgets) sobre o `TreeSegmenter`, **unificada por**
``task_type`` ("classification" = alvo binário · "regression" = alvo
contínuo) — a mesma UI atende os dois, mudando só o parâmetro.

Construa a árvore de segmentação clicando em botões, dentro do Jupyter, operando
sobre o DataFrame e o alvo reais. Recursos:
- árvore colorida pelo alvo médio que se atualiza a cada ação;
- **PSI ao vivo** por amostra (OOT, ESTABILIDADE, ...) no topo do painel;
- **discriminação ao vivo** por amostra: KS/AUC (classificação) ou R² (regressão);
- tabela de folhas com **PSI por amostra** e **p-valor** do teste entre folhas adjacentes;
- gráficos por tarefa: ROC/KS/taxa-default/distribuição (clf) ou boxplot/histograma (reg);
- travar folhas como finais (cadeado), podar, resetar e exportar o DataFrame rotulado;
- **cenários nomeados em memória** (aba Avançado): salve versões da árvore na
  sessão, restaure depois (desfazível) e compare-as lado a lado com a atual —
  os cenários NÃO são gravados em disco (para persistir use Salvar/JSON ou MLflow).

    from yggdrasil.credit_risk.tree import TreeSegmenterUI
    ui = TreeSegmenterUI(df, target="target", task_type="classification",
                         sample_col="amostra", ref_sample="DES", feature_labels=labels)
    ui
"""
from __future__ import annotations

import html as _html
from contextlib import contextmanager

import pandas as pd

try:
    import ipywidgets as W
    from IPython.display import display
except Exception as e:  # pragma: no cover
    raise ImportError("Este módulo requer ipywidgets e IPython (Jupyter).") from e

from .segmenter import TreeSegmenter
from .._common import fmt as _fmt      # formatação de limites (corte vigente na UI)


def _esc(txt) -> str:
    """Escapa texto LIVRE do usuário (ex.: apelidos de folha) p/ HTML/atributos."""
    return _html.escape(str(txt), quote=True)


def _running_in_databricks() -> bool:
    """True quando o código roda dentro de um cluster/notebook **Databricks**.

    Detecta pelo env var que o Databricks Runtime sempre injeta
    (``DATABRICKS_RUNTIME_VERSION``); como reforço, reconhece o ``dbutils`` que o
    Databricks expõe no namespace do notebook. Serve para o preview da árvore
    escolher, por padrão, o caminho **autocontido** (PNG data-URL, widget core do
    ipywidgets) em vez do interativo (anywidget) — cujo frontend o Databricks
    busca de um CDN, o que trava num cluster sem egress. Ver ``__init__``."""
    import os
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return True
    try:                                   # dbutils é injetado no notebook Databricks
        import builtins
        if hasattr(builtins, "dbutils") or "dbutils" in getattr(builtins, "__dict__", {}):
            return True
    except Exception:
        pass
    return False


_CSS = """
<style>
/* SEM @import externo de fonts.googleapis.com: no Databricks o cluster costuma não
   ter egress, e um @import render-blocking no topo do <style> deixa a requisição
   pendente até o timeout de rede, adiando o 1º paint da UI (parece "não carrega").
   Usa-se a font-stack do sistema declarada em font-family abaixo (IBM Plex se
   instalada localmente; senão -apple-system/Segoe UI/Roboto...). */
.treeui { --ac:#3b4a63; --ac-deep:#27324a; --ac-soft:#eef1f5; --ac-border:#cdd5e0;
  --ink:#1f2733; --muted:#6b7480; --line:#e7e9ee;
  /* tokens semânticos (status, tabelas, realces): o HTML gerado no Python usa
     var(--...) em vez de hex — o tema escuro só redefine os tokens aqui */
  --ok-ink:#157a52; --ok-bg:#e7f5ee; --ok-tx:#137a3e;
  --warn-ink:#9a6f12; --warn-bg:#fbf3e0; --warn-tx:#9a6b00;
  --bad-ink:#b23a2a; --bad-bg:#fbe7e4; --bad-tx:#b3261e;
  --sus-tx:#6b3fa0;
  --risk-lo:#1aa64b; --risk-mid:#caa000; --risk-hi:#d6453e;
  --gauge-ok:#2bb673; --gauge-warn:#e6b800; --gauge-bad:#e0584f; --gauge-track:#eceff3;
  --strong-ink:#15324a; --body-ink:#3a4250; --sub-ink:#8a93a3; --tree-meta:#7c8893;
  --faint-ink:#aab4be; --hair:#eef0f3; --tile-bg:#f7f8fa;
  --sel-bg:#fff5e6; --sel-ac:#e8870b;
  --tbl-line:#e1e5ec; --tbl-line-strong:#cdd5e0; --tbl-head-bg:#eef1f5;
  --tbl-head-ink:#27324a; --tbl-head-line:#b9c2d0; --tbl-zebra:#fafbfc;
  --tbl-hover:#eef3f8; --tbl-sticky:#f4f6f9;
  --ci-bar:#9bb7c9; --ci-ref:#0f3d57;
  --notice-bg:#fff8e6; --notice-border:#f0c36d; --notice-ink:#664d03;
  /* canvas da aba "Árvore interativa": pontinhos do fundo e face dos cartões */
  --cv-dot:rgba(31,39,51,.10); --cv-node-bg:#fff;
  font-family:'IBM Plex Sans', -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  color:var(--ink); }
.treeui .mono { font-family:'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas,
  monospace; font-variant-numeric: tabular-nums; }
/* top bar (estilo mockup): branca, com chip alvo grafite */
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
.treeui-band-muted { color:var(--sub-ink); margin-top:14px; }
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
.pill-green  { background:var(--ok-bg); color:var(--ok-ink); }
.pill-yellow { background:var(--warn-bg); color:var(--warn-ink); }
.pill-red    { background:var(--bad-bg); color:var(--bad-ink); }
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
.treeui-metric { background:var(--tile-bg); border:1px solid var(--hair); border-radius:9px;
  padding:7px 10px; overflow:hidden; }
.treeui-metric .k { font-size:10px; text-transform:uppercase; letter-spacing:.04em;
  color:var(--sub-ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.treeui-metric .v { font-size:16px; font-weight:600; color:var(--ink); margin-top:2px;
  white-space:nowrap; }
/* botões: cantos mais suaves, alinhados ao mockup */
.treeui .jupyter-button { border-radius:8px; font-family:inherit; }
/* atalhos de sugestão de variável (painel da aba "Árvore interativa"): botões
   largos, texto à esquerda — leem como uma lista, não como uma barra de ações.
   O !important na altura é necessário: a folha do próprio ipywidgets é carregada
   DEPOIS desta e, com a mesma especificidade, zeraria a altura do botão. */
.treeui button.treeui-sug {
  justify-content:flex-start !important; text-align:left; font-size:11.5px;
  height:30px !important; line-height:1.2 !important; padding:0 11px !important;
  margin-bottom:4px; background:var(--tile-bg) !important; color:var(--strong-ink);
  border:1px solid var(--line) !important; }
.treeui button.treeui-sug:hover {
  background:var(--ac-soft) !important; border-color:var(--ac-border) !important; }
/* painel lateral da aba "Árvore interativa": ele tem altura máxima e rola, e as
   caixas do ipywidgets são todas `flex:0 1 auto` — sem isto o conteúdo que
   passa da altura é ESMAGADO (as caixas internas chegam a 0px) em vez de rolar */
.treeui-cvpanel .widget-box, .treeui-cvpanel .jupyter-button,
.treeui-cvpanel .widget-html, .treeui-cvpanel .widget-inline-hbox {
  flex-shrink:0 !important; }
/* o card do mapa é o único que ganha a largura toda: encosta nas bordas do
   painel (margem negativa cobre o padding de .treeui + o do conteúdo das abas)
   e quase zera o próprio padding lateral. Sobra ~70px a mais de plano, e o
   canvas começa mais à esquerda que os cards das outras abas. */
.treeui-card-mapa { padding-left:5px !important; padding-right:5px !important;
  margin-left:-9px; margin-right:-9px; }
/* palco do canvas: âncora do posicionamento absoluto da janelinha de
   confirmação (o Layout do ipywidgets não expõe `position`, então as duas
   classes fazem o par relative/absolute) */
.treeui-cv-stage { position:relative; }
.treeui-cv-modal { position:absolute; left:50%; top:44px; transform:translateX(-50%);
  width:360px; max-width:92%; z-index:30; box-sizing:border-box;
  background-color:var(--cv-node-bg,#fff);
  border-width:1px; border-style:solid; border-color:var(--line,#e7e9ee);
  border-radius:13px; box-shadow:0 16px 44px rgba(16,24,40,.28);
  padding:13px 16px 14px; }
.treeui-cv-modal .widget-box, .treeui-cv-modal .jupyter-button,
.treeui-cv-modal .widget-html { flex-shrink:0 !important; }
/* sliders/controles encolhem para caber na coluna (min-width:0 libera o flex)
   e os cards clipam qualquer sobra horizontal — elimina a barra de rolagem
   horizontal que aparecia embaixo dos cards na aba Construir */
.treeui .jupyter-widgets { min-width:0 !important; }
.treeui-card { overflow-x:clip; }
/* ===== TEMA ESCURO (classe .dark no painel raiz) =====
   Paleta alinhada ao dark mode do Databricks (design system DuBois):
   fundo grey800 #11171C, superfícies grey700 #1F272D, bordas grey650 #37444F,
   texto #E8ECF0/#92A4B3, ação primária blue500 #4299E0 com texto escuro. */
.treeui.dark { --ink:#E8ECF0; --muted:#92A4B3; --line:#37444F; --ac-soft:#37444F;
  --ac-border:#5F7281; --ac-deep:#E8ECF0; --ac:#4299E0;
  --ok-ink:#3BA65E; --ok-bg:rgba(39,124,67,.16); --ok-tx:#3BA65E;
  --warn-ink:#DE7921; --warn-bg:rgba(190,80,30,.16); --warn-tx:#DE7921;
  --bad-ink:#E65B77; --bad-bg:rgba(200,45,76,.16); --bad-tx:#E65B77;
  --sus-tx:#B592E5;
  --risk-lo:#3BA65E; --risk-mid:#DE7921; --risk-hi:#E65B77;
  --gauge-ok:#3BA65E; --gauge-warn:#DE7921; --gauge-bad:#E65B77; --gauge-track:#37444F;
  --strong-ink:#E8ECF0; --body-ink:#C0CDD8; --sub-ink:#8396A5; --tree-meta:#92A4B3;
  --faint-ink:#5F7281; --hair:#37444F; --tile-bg:#11171C;
  --sel-bg:rgba(232,135,11,.15); --sel-ac:#F0A24A;
  --tbl-line:#37444F; --tbl-line-strong:#445461; --tbl-head-bg:#11171C;
  --tbl-head-ink:#E8ECF0; --tbl-head-line:#445461; --tbl-zebra:rgba(189,205,219,.04);
  --tbl-hover:rgba(189,205,219,.08); --tbl-sticky:#11171C;
  --ci-bar:#5F7281; --ci-ref:#8ACAFF;
  --notice-bg:rgba(190,80,30,.16); --notice-border:#DE7921; --notice-ink:#E8ECF0;
  --cv-dot:rgba(232,236,240,.11); --cv-node-bg:#1F272D;
  background:#11171C; padding:8px; border-radius:12px; }
.treeui.dark .treeui-banner, .treeui.dark .treeui-card, .treeui.dark .treeui-bar,
.treeui.dark .treeui-chips .chip { background:#1F272D !important;
  border-color:#37444F !important; box-shadow:none !important; }
.treeui.dark .treeui-banner .t { color:#E8ECF0; }
/* ação primária DuBois: azul com texto ESCURO (não branco) */
.treeui.dark .treeui-banner .logo { color:#11171C; }
.treeui.dark .treeui-band { color:#8ACAFF; }
.treeui.dark .treeui-tabs .p-TabBar-tab,
.treeui.dark .treeui-tabs .lm-TabBar-tab { background:#1F272D !important;
  color:#92A4B3 !important; border-color:#37444F !important; }
.treeui.dark .treeui-tabs .p-TabBar-tab:hover,
.treeui.dark .treeui-tabs .lm-TabBar-tab:hover { background:rgba(138,202,255,.08) !important;
  color:#8ACAFF !important; border-color:#8ACAFF !important; }
.treeui.dark .treeui-tabs .p-TabBar-tab.p-mod-current,
.treeui.dark .treeui-tabs .lm-TabBar-tab.lm-mod-current { background:#4299E0 !important;
  color:#11171C !important; border-color:#4299E0 !important; }
.treeui.dark .treeui-tabs .p-TabBar-tab.p-mod-current:hover,
.treeui.dark .treeui-tabs .lm-TabBar-tab.lm-mod-current:hover {
  background:#8ACAFF !important; color:#11171C !important; border-color:#8ACAFF !important; }
.treeui.dark .widget-text input, .treeui.dark .widget-dropdown select,
.treeui.dark textarea { background:#11171C !important; color:#E8ECF0 !important;
  border-color:#37444F !important; }
.treeui.dark .widget-label, .treeui.dark .jupyter-widgets label { color:#D1D9E1 !important; }
/* botões ipywidgets sem button_style: seguem a superfície DuBois */
.treeui.dark .jupyter-button:not(.mod-primary):not(.mod-success):not(.mod-info):not(.mod-warning):not(.mod-danger) { background:#37444F !important; color:#E8ECF0 !important; }
.treeui.dark .jupyter-button.mod-active { background:#4299E0 !important; color:#11171C !important; }
/* chip do nó selecionado na barra de ações do preview interativo da árvore */
.treeui-imgchip { display:inline-block; font-size:11px; font-weight:600; color:var(--ac-deep);
  background:var(--ac-soft); border:1px solid var(--ac-border); border-radius:999px; padding:3px 11px;
  margin-right:6px; max-width:420px; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; vertical-align:middle; }
.treeui.dark .treeui-imgchip { background:#37444F; border-color:#5F7281; color:#E8ECF0; }
/* barras de ações (preview clicável e mapa da árvore interativa): separador
   vertical entre grupos de botões. A regra vale em qualquer barra dentro da UI —
   estava presa a .treeui-imgbar e o separador nascia 0×0 fora dela. */
.treeui .treeui-vsep { display:inline-block; width:1px; height:24px; flex:none;
  background:var(--line); margin:4px 12px 4px 4px; }
</style>
"""


# ======================================================================
# Preview INTERATIVO da árvore (anywidget): o PNG do plot_tree ganha uma
# camada de "hotspots" (um por nó, posicionados em % — seguem o redimensio-
# namento responsivo da imagem). Clicar num nó sincroniza o trait `selected`
# com o kernel; hover mostra um tooltip com as métricas do nó (montado no
# Python — nenhum cálculo no front). anywidget entrega o JS via o próprio
# comm do widget (traits `_esm`/`_css`), sem labextension pré-instalada —
# por isso funciona em Jupyter, VS Code, Colab e Databricks (DBR recente),
# ao contrário de pontes como ipyevents/ipympl.
# ======================================================================
_TREE_IMG_ESM = """
function render({ model, el }) {
  const scroller = document.createElement("div"); // rolagem horizontal p/ árvores largas
  scroller.className = "ygg-treeimg-scroll";
  const wrap = document.createElement("div");
  wrap.className = "ygg-treeimg-wrap";
  const img = document.createElement("img");
  img.className = "ygg-treeimg-img";
  img.draggable = false;
  const layer = document.createElement("div");     // hotspots (um por nó)
  layer.className = "ygg-treeimg-layer";
  const hov = document.createElement("div");       // contorno do nó sob o mouse
  hov.className = "ygg-treeimg-hov";
  const sel = document.createElement("div");       // contorno do nó selecionado
  sel.className = "ygg-treeimg-sel";
  const tip = document.createElement("div");       // tooltip com métricas do nó
  tip.className = "ygg-treeimg-tip";
  wrap.append(img, layer, hov, sel, tip);
  scroller.appendChild(wrap);
  el.replaceChildren(scroller);

  // clique fora de qualquer nó: limpa a seleção (o Python esconde a barra)
  wrap.addEventListener("click", () => { model.set("selected", ""); model.save_changes(); });

  const place = (box, n, W, H) => {                // posiciona em % da imagem
    box.style.left = (100 * n.x0 / W) + "%";
    box.style.top = (100 * n.y0 / H) + "%";
    box.style.width = (100 * (n.x1 - n.x0) / W) + "%";
    box.style.height = (100 * (n.y1 - n.y0) / H) + "%";
  };

  function syncSel() {
    const W = model.get("width"), H = model.get("height");
    const n = (model.get("nodes") || []).find(n => n.sid === model.get("selected"));
    if (n) { place(sel, n, W, H); sel.style.display = "block"; }
    else sel.style.display = "none";
  }

  // tamanho de exibição: preenche a largura do card; se a árvore for LARGA a
  // ponto de ficar baixa demais, garante min_height (a largura excede e o
  // scroller rola). Nunca amplia além do tamanho natural do PNG (borraria) —
  // salvo quando o usuário pede zoom explícito.
  //
  // Zoom: multiplica a escala ajustada. Como os hotspots, o contorno e o
  // tooltip são posicionados em % do `wrap`, todos acompanham a imagem sem
  // recalcular nada. Com zoom > 1 o scroller ganha altura máxima e rolagem
  // vertical, para a árvore ampliada não empurrar o resto da página; em zoom 1
  // o comportamento é exatamente o anterior.
  function fit() {
    const W = model.get("width"), H = model.get("height");
    const minH = model.get("min_height") || 560;
    const z = model.get("zoom") || 1;
    const avail = scroller.clientWidth || W;
    const s = Math.min(1, Math.max(avail / W, minH / H));
    img.style.width = Math.round(W * s * z) + "px";
    if (z > 1) {
      scroller.style.maxHeight = (model.get("max_height") || 720) + "px";
      scroller.style.overflowY = "auto";
    } else {
      scroller.style.maxHeight = "";
      scroller.style.overflowY = "hidden";
    }
  }

  function rebuild() {
    const W = model.get("width"), H = model.get("height");
    img.src = model.get("src");
    hov.style.display = "none"; tip.style.display = "none";
    layer.replaceChildren();
    for (const n of (model.get("nodes") || [])) {
      const hs = document.createElement("div");
      hs.className = "ygg-treeimg-hot";
      place(hs, n, W, H);
      hs.addEventListener("click", (ev) => {
        ev.stopPropagation();                      // não deixa o wrap deselecionar
        model.set("selected", n.sid); model.save_changes();
      });
      hs.addEventListener("mouseenter", () => {
        place(hov, n, W, H); hov.style.display = "block";
        if (!n.tooltip) return;
        tip.innerHTML = n.tooltip;                 // montado (e escapado) no Python
        // acima do nó; perto do topo da imagem, abre para baixo
        if (100 * n.y0 / H < 22) { tip.style.top = (100 * n.y1 / H) + "%"; tip.style.bottom = "auto"; }
        else { tip.style.bottom = (100 * (H - n.y0) / H) + "%"; tip.style.top = "auto"; }
        tip.style.left = Math.max(4, Math.min(96, 100 * (n.x0 + n.x1) / 2 / W)) + "%";
        tip.style.display = "block";
      });
      hs.addEventListener("mouseleave", () => { hov.style.display = "none"; tip.style.display = "none"; });
      layer.appendChild(hs);
    }
    fit();
    syncSel();
  }

  // um refresh do Python altera src+nodes+width+height de uma vez — o rAF
  // agrupa os 4 eventos de change num único rebuild
  let raf = 0;
  const schedule = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(rebuild); };
  const evs = ["change:src", "change:width", "change:height", "change:nodes"];
  evs.forEach(ev => model.on(ev, schedule));
  model.on("change:selected", syncSel);
  model.on("change:min_height", fit);
  model.on("change:zoom", fit);
  const ro = new ResizeObserver(() => fit());      // card redimensionado → reajusta
  ro.observe(scroller);
  rebuild();
  return () => {
    ro.disconnect();
    evs.forEach(ev => model.off(ev, schedule));
    model.off("change:selected", syncSel);
    model.off("change:min_height", fit);
    model.off("change:zoom", fit);
  };
}
export default { render };
"""

_TREE_IMG_CSS = """
.ygg-treeimg-scroll { width:100%; overflow-x:auto; overflow-y:hidden; }
.ygg-treeimg-wrap { position:relative; display:inline-block; }
.ygg-treeimg-img { display:block; height:auto;
  border:1px solid var(--line, #e6e8eb); border-radius:6px; box-sizing:border-box; }
.ygg-treeimg-hot { position:absolute; cursor:pointer; z-index:3; }
.ygg-treeimg-hov, .ygg-treeimg-sel { position:absolute; display:none;
  border-radius:7px; pointer-events:none; box-sizing:border-box; z-index:2; }
.ygg-treeimg-hov { border:2px solid #2f6fb2; }
.ygg-treeimg-sel { border:3px solid #e8870b; box-shadow:0 0 0 3px rgba(245,166,35,.28); }
.ygg-treeimg-tip { position:absolute; display:none; transform:translateX(-50%);
  background:#1d2733; color:#fff; font-size:11px; line-height:1.5; text-align:left;
  padding:7px 10px; border-radius:7px; max-width:300px; width:max-content;
  white-space:normal; pointer-events:none; z-index:4; margin:4px 0;
  box-shadow:0 4px 14px rgba(0,0,0,.28); }
"""

_TREE_IMG_WIDGET_CLS = None


def _tree_image_widget_cls():
    """Classe do widget de árvore clicável — criada 1× e cacheada.

    Devolve ``None`` quando o anywidget não está instalado (ou falhou ao
    importar): a UI então cai no PNG estático, comportamento anterior."""
    global _TREE_IMG_WIDGET_CLS
    if _TREE_IMG_WIDGET_CLS is not None:
        return _TREE_IMG_WIDGET_CLS
    try:
        import anywidget
        import traitlets
    except Exception:                     # sem anywidget → fallback estático
        return None

    class _TreeImageWidget(anywidget.AnyWidget):
        _esm = _TREE_IMG_ESM
        _css = _TREE_IMG_CSS
        src = traitlets.Unicode("").tag(sync=True)        # PNG em data-URL
        width = traitlets.Int(1).tag(sync=True)           # px naturais do PNG
        height = traitlets.Int(1).tag(sync=True)
        nodes = traitlets.List(traitlets.Dict()).tag(sync=True)   # hitboxes + tooltips
        selected = traitlets.Unicode("").tag(sync=True)   # sid clicado ("" = nenhum)
        # altura MÍNIMA de exibição (px): árvores largas não encolhem além
        # disso — a largura excede o card e rola na horizontal
        min_height = traitlets.Int(560).tag(sync=True)
        # zoom do preview (1 = ajustado ao card). >1 amplia e liga a rolagem
        # vertical dentro de max_height, p/ ler nós de árvores grandes
        zoom = traitlets.Float(1.0).tag(sync=True)
        max_height = traitlets.Int(720).tag(sync=True)

    _TREE_IMG_WIDGET_CLS = _TreeImageWidget
    return _TREE_IMG_WIDGET_CLS


# ======================================================================
# Árvore interativa em CANVAS (aba "Árvore interativa"): cada nó vira um
# cartão num plano navegável (arrastar p/ mover, rolar p/ ampliar) e clicar
# num nó abre, ao lado, o painel onde o corte e as regras de negócio da
# folha são definidos.
#
# Mesma divisão de trabalho do preview clicável acima: TODO o cálculo mora
# no Python — posição de cada cartão, cores, textos e o HTML de dentro do
# nó chegam prontos pelos traits. O front só arrasta, amplia e devolve o
# `selected`. Dois contadores (`fit_token`/`center_token`) deixam o Python
# pedir "enquadre a árvore" e "centralize no nó selecionado" sem inventar
# um protocolo de mensagens: basta incrementá-los.
# ======================================================================
_TREE_CANVAS_ESM = """
function render({ model, el }) {
  const NS = "http://www.w3.org/2000/svg";
  const canvas = document.createElement("div");
  canvas.className = "ygg-cv-canvas";
  const world = document.createElement("div");      // plano que sofre o pan/zoom
  world.className = "ygg-cv-world";
  const svg = document.createElementNS(NS, "svg");  // arestas (curvas de Bézier)
  svg.setAttribute("class", "ygg-cv-edges");
  const layer = document.createElement("div");      // cartões dos nós
  layer.className = "ygg-cv-nodes";
  world.append(svg, layer);
  canvas.appendChild(world);
  const tools = document.createElement("div");      // +/−/enquadrar
  tools.className = "ygg-cv-tools";
  canvas.appendChild(tools);
  el.replaceChildren(canvas);

  let z = 1, px = 0, py = 0, anim = true;
  let drag = null, moved = false;

  const rect = () => {
    const r = canvas.getBoundingClientRect();
    return { w: r.width || 900, h: r.height || 560, left: r.left, top: r.top };
  };
  const apply = () => {
    world.style.transform = "translate(" + px + "px," + py + "px) scale(" + z + ")";
    world.style.transition = anim ? "transform .45s cubic-bezier(.22,1,.36,1)" : "none";
  };
  // zoom ancorado no ponto (cx, cy) do canvas: o que está sob o cursor não sai
  // do lugar — é o gesto esperado de qualquer mapa
  const setZoom = (nz, cx, cy) => {
    nz = Math.min(2.2, Math.max(0.12, nz));
    const k = nz / z;
    px = cx - k * (cx - px); py = cy - k * (cy - py); z = nz;
    apply();
  };
  function fit() {
    const W = model.get("content_w") || 1, H = model.get("content_h") || 1;
    const r = rect();
    let nz = Math.min((r.w - 56) / W, (r.h - 56) / H, 1.1);
    if (!isFinite(nz) || nz <= 0) nz = 1;
    z = nz; px = (r.w - z * W) / 2; py = (r.h - z * H) / 2;
    anim = true; apply();
  }
  function center() {
    const sid = model.get("selected");
    const n = (model.get("nodes") || []).find(n => n.sid === sid);
    if (!n) return;
    const r = rect();
    z = Math.max(z, 0.5);
    px = r.w / 2 - z * (n.x + n.w / 2);
    py = r.h * 0.36 - z * (n.y + n.h / 2);
    anim = true; apply();
  }

  const ic = (d) => '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
    'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">' + d + '</svg>';
  const mkBtn = (inner, title, fn) => {
    const b = document.createElement("button");
    b.className = "ygg-cv-btn"; b.title = title; b.innerHTML = inner;
    b.addEventListener("mousedown", e => e.stopPropagation());
    b.addEventListener("click", e => { e.stopPropagation(); fn(); });
    tools.appendChild(b);
  };
  mkBtn(ic('<path d="M12 5v14"/><path d="M5 12h14"/>'), "Aproximar",
        () => { const r = rect(); anim = true; setZoom(z * 1.25, r.w / 2, r.h / 2); });
  mkBtn(ic('<path d="M5 12h14"/>'), "Afastar",
        () => { const r = rect(); anim = true; setZoom(z / 1.25, r.w / 2, r.h / 2); });
  mkBtn(ic('<path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/>' +
           '<path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/>'),
        "Enquadrar a árvore inteira", () => fit());

  function syncSel() {
    const sel = model.get("selected");
    layer.querySelectorAll(".ygg-cv-node").forEach(
      n => n.classList.toggle("sel", n.dataset.sid === sel));
    svg.querySelectorAll("path").forEach(
      p => p.classList.toggle("hi", p.dataset.child === sel));
  }

  function rebuild() {
    const W = model.get("content_w") || 1, H = model.get("content_h") || 1;
    svg.setAttribute("width", W); svg.setAttribute("height", H);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    const ef = document.createDocumentFragment();
    for (const e of (model.get("edges") || [])) {
      const p = document.createElementNS(NS, "path");
      p.setAttribute("d", e.d); p.dataset.child = e.child;
      ef.appendChild(p);
    }
    svg.replaceChildren(ef);
    const nf = document.createDocumentFragment();
    for (const n of (model.get("nodes") || [])) {
      const d = document.createElement("div");
      d.className = "ygg-cv-node" + (n.leaf ? " leaf" : "");
      d.dataset.sid = n.sid;
      d.style.left = n.x + "px"; d.style.top = n.y + "px";
      d.style.width = n.w + "px"; d.style.height = n.h + "px";
      d.innerHTML = n.html;                        // montado (e escapado) no Python
      d.addEventListener("mousedown", e => e.stopPropagation());
      d.addEventListener("click", e => {
        e.stopPropagation();
        model.set("selected", n.sid); model.save_changes();
      });
      nf.appendChild(d);
    }
    layer.replaceChildren(nf);
    syncSel();
  }

  canvas.addEventListener("mousedown", e => {
    if (e.button !== 0) return;
    moved = false; drag = { mx: e.clientX, my: e.clientY, px, py };
    canvas.classList.add("grabbing");
  });
  const onMove = (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.mx, dy = e.clientY - drag.my;
    if (Math.abs(dx) + Math.abs(dy) > 4) moved = true;
    px = drag.px + dx; py = drag.py + dy; anim = false; apply();
  };
  const onUp = () => { drag = null; canvas.classList.remove("grabbing"); };
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  // clique no vazio limpa a seleção; arrastar não conta como clique
  canvas.addEventListener("click", () => {
    if (moved) { moved = false; return; }
    model.set("selected", ""); model.save_changes();
  });
  const onWheel = (e) => {
    e.preventDefault();
    const r = rect();
    anim = false;
    setZoom(z * Math.exp(-e.deltaY * 0.0012), e.clientX - r.left, e.clientY - r.top);
  };
  canvas.addEventListener("wheel", onWheel, { passive: false });

  // um refresh do Python troca nodes+edges+tamanho de uma vez: o rAF agrupa os
  // eventos num único rebuild
  let raf = 0;
  const schedule = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(rebuild); };
  const onFit = () => fit(), onCenter = () => center();
  const evs = ["change:nodes", "change:edges", "change:content_w", "change:content_h"];
  evs.forEach(ev => model.on(ev, schedule));
  model.on("change:selected", syncSel);
  model.on("change:fit_token", onFit);
  model.on("change:center_token", onCenter);
  rebuild();
  // 1ª medida do canvas só vale depois do layout. Com um nó já em foco a tela
  // abre CENTRALIZADA nele (cartões legíveis); sem foco, enquadra a árvore.
  setTimeout(() => { if (model.get("selected")) center(); else fit(); }, 60);
  return () => {
    cancelAnimationFrame(raf);
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    canvas.removeEventListener("wheel", onWheel);
    evs.forEach(ev => model.off(ev, schedule));
    model.off("change:selected", syncSel);
    model.off("change:fit_token", onFit);
    model.off("change:center_token", onCenter);
  };
}
export default { render };
"""

# As cores/medidas saem dos MESMOS tokens semânticos do resto da UI (ver _CSS):
# o widget é injetado dentro do painel .treeui, então herda inclusive o tema
# escuro. Os fallbacks depois da vírgula cobrem o caso de o CSS da UI não ter
# sido injetado (widget renderizado solto).
#
# NUNCA usar var(--...) dentro do ATALHO `border`/`background` aqui: o Chromium
# não re-resolve a variável dentro do atalho quando o tema muda em tempo de
# execução (a classe .dark entra depois do 1º cálculo de estilo) e a borda fica
# presa na cor clara. Com as longhands (border-color, background-color) o
# recálculo acontece — por isso elas aparecem separadas abaixo.
_TREE_CANVAS_CSS = """
.ygg-cv-canvas { position:relative; width:100%; height:100%; overflow:hidden; cursor:grab;
  border-width:1px; border-style:solid; border-color:var(--line,#e7e9ee);
  border-radius:12px; background-color:var(--tile-bg,#f7f8fa);
  background-image:radial-gradient(var(--cv-dot,rgba(31,39,51,.10)) 1.1px, transparent 1.1px);
  background-size:24px 24px; }
.ygg-cv-canvas.grabbing { cursor:grabbing; }
.ygg-cv-world { position:absolute; left:0; top:0; transform-origin:0 0; }
.ygg-cv-edges { position:absolute; left:0; top:0; overflow:visible; pointer-events:none; }
.ygg-cv-edges path { fill:none; stroke:var(--ac-border,#cdd5e0); stroke-width:2; }
.ygg-cv-edges path.hi { stroke:var(--sel-ac,#e8870b); stroke-width:2.6; }
.ygg-cv-nodes { position:absolute; left:0; top:0; }
.ygg-cv-node { position:absolute; box-sizing:border-box; cursor:pointer;
  background-color:var(--cv-node-bg,#fff);
  border-width:1.5px; border-style:solid; border-color:var(--line,#e7e9ee);
  border-radius:12px; padding:9px 11px 8px; display:flex; flex-direction:column;
  box-shadow:0 1px 3px rgba(16,24,40,.10);
  font-family:'IBM Plex Sans',-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  transition:border-color .15s, box-shadow .15s; }
.ygg-cv-node:hover { border-color:var(--ac-border,#cdd5e0); box-shadow:0 4px 12px rgba(16,24,40,.16); }
.ygg-cv-node.sel { border-color:var(--sel-ac,#e8870b); border-width:2px;
  box-shadow:0 0 0 3px rgba(232,135,11,.22), 0 4px 14px rgba(16,24,40,.18); }
.ygg-cv-node .t { display:flex; align-items:center; gap:5px; }
.ygg-cv-node .pdot { width:9px; height:9px; border-radius:50%; flex:none; }
.ygg-cv-node .lb { font-size:11.5px; font-weight:600; line-height:1.25; flex:1;
  color:var(--strong-ink,#15324a); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.ygg-cv-node .m { display:flex; align-items:baseline; gap:6px; margin-top:2px; }
.ygg-cv-node .v { font-size:17px; font-weight:600; font-variant-numeric:tabular-nums; }
.ygg-cv-node .g { font-size:9.5px; padding:1px 7px; border-radius:999px;
  background-color:var(--ac-soft,#eef1f5); color:var(--ac-deep,#27324a); white-space:nowrap; }
.ygg-cv-node .s { font-size:10px; color:var(--tree-meta,#7c8893); white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }
.ygg-cv-node .bar { margin-top:auto; height:6px; border-radius:999px;
  background-color:var(--gauge-track,#eceff3); overflow:hidden; }
.ygg-cv-node .bar i { display:block; height:100%; border-radius:999px; }
.ygg-cv-tools { position:absolute; left:12px; top:12px; display:flex; flex-direction:column;
  gap:5px; z-index:5; }
.ygg-cv-btn { width:30px; height:30px; display:flex; align-items:center; justify-content:center;
  padding:0; cursor:pointer; border-width:1px; border-style:solid;
  border-color:var(--line,#e7e9ee); border-radius:8px;
  background-color:var(--cv-node-bg,#fff); color:var(--muted,#6b7480);
  box-shadow:0 1px 3px rgba(16,24,40,.10); }
.ygg-cv-btn:hover { background-color:var(--ac-soft,#eef1f5); color:var(--ac-deep,#27324a); }
.ygg-cv-btn svg { width:15px; height:15px; display:block; }
"""

_TREE_CANVAS_WIDGET_CLS = None


def _tree_canvas_widget_cls():
    """Classe do canvas navegável da árvore — criada 1× e cacheada.

    ``None`` quando o anywidget não está instalado: a aba então mostra um
    aviso explicando como habilitá-la (não há como desenhar um plano com
    pan/zoom só com os widgets core do ipywidgets)."""
    global _TREE_CANVAS_WIDGET_CLS
    if _TREE_CANVAS_WIDGET_CLS is not None:
        return _TREE_CANVAS_WIDGET_CLS
    try:
        import anywidget
        import traitlets
    except Exception:
        return None

    class _TreeCanvasWidget(anywidget.AnyWidget):
        _esm = _TREE_CANVAS_ESM
        _css = _TREE_CANVAS_CSS
        nodes = traitlets.List(traitlets.Dict()).tag(sync=True)   # cartões (x/y/html)
        edges = traitlets.List(traitlets.Dict()).tag(sync=True)   # curvas pai→filho
        content_w = traitlets.Int(1).tag(sync=True)               # tamanho do plano
        content_h = traitlets.Int(1).tag(sync=True)
        selected = traitlets.Unicode("").tag(sync=True)           # sid clicado ("" = nenhum)
        fit_token = traitlets.Int(0).tag(sync=True)               # ++ → enquadra tudo
        center_token = traitlets.Int(0).tag(sync=True)            # ++ → centraliza no sel.

    _TREE_CANVAS_WIDGET_CLS = _TreeCanvasWidget
    return _TREE_CANVAS_WIDGET_CLS


class TreeSegmenterUI:
    # mesma figsize p/ os dois gráficos lado a lado da faixa de detalhe
    # (distribuição da variável + cortes  e  histograma da alvo/target da folha)
    _PREVIEW_FIGSIZE = (6.0, 3.6)

    # estilo das tabelas (Styler): bordas em cada célula p/ a divisão de colunas
    # ficar nítida, cabeçalho grafite fixo no topo e linhas com zebra leve.
    # cores via var(--tbl-*): o CSS do Styler é injetado dentro do painel
    # .treeui, então os tokens resolvem no tema ativo (claro ou escuro) sem
    # precisar re-renderizar a tabela quando o usuário alterna o tema.
    _TABLE_STYLES = [
        {"selector": "", "props": [("border-collapse", "collapse"),
                                   ("border", "1px solid var(--tbl-line-strong)"),
                                   ("width", "100%")]},
        {"selector": "th, td", "props": [("border", "1px solid var(--tbl-line)"),
                                         ("padding", "4px 9px"),
                                         ("text-align", "right"),
                                         ("white-space", "nowrap")]},
        {"selector": "thead th", "props": [("background-color", "var(--tbl-head-bg)"),
                                           ("color", "var(--tbl-head-ink)"),
                                           ("font-weight", "600"),
                                           ("border-bottom", "2px solid var(--tbl-head-line)"),
                                           ("position", "sticky"),
                                           ("top", "0"), ("z-index", "1")]},
        {"selector": "tbody tr:nth-child(even) td",
         "props": [("background-color", "var(--tbl-zebra)")]},
        {"selector": "tbody tr:hover td", "props": [("background-color", "var(--tbl-hover)")]},
    ]

    # Camada de estilo das tabelas "Detalhe por safra": cabeçalho claro (via
    # _TABLE_STYLES) + coluna 'safra' ancorada à esquerda (sticky horizontal).
    _SAFRA_HEADER_STYLES = [
        # cabeçalho idêntico ao das demais tabelas (claro, via _TABLE_STYLES);
        # aqui só ancoramos a coluna 'safra' à esquerda (sticky horizontal).
        {"selector": "tbody td:first-child", "props": [
            ("text-align", "left"),
            ("font-family", "'IBM Plex Sans',sans-serif"),
            ("font-weight", "600"), ("color", "var(--tbl-head-ink)"),
            ("position", "sticky"), ("left", "0"), ("z-index", "2"),
            ("background-color", "var(--tbl-sticky)"),
            ("border-right", "1px solid var(--tbl-line-strong)")]},
        {"selector": "thead th:first-child", "props": [
            ("text-align", "left"), ("position", "sticky"),
            ("left", "0"), ("z-index", "3")]},
        {"selector": "tbody tr:nth-child(odd) td:first-child", "props": [
            ("background-color", "var(--tbl-sticky)")]},
        {"selector": "tbody tr:nth-child(even) td:first-child", "props": [
            ("background-color", "var(--tbl-head-bg)")]},
        {"selector": "tbody tr:hover td:first-child", "props": [
            ("background-color", "var(--tbl-hover)")]},
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
            return "background-color:%s;color:var(--muted)" % na
        span = (vmax - vmin)
        t = 0.0 if span <= 0 else (float(v) - vmin) / span
        t = min(max(t, 0.0), 1.0) * ceiling
        r = int(round(255 + (59 - 255) * t))
        g = int(round(255 + (74 - 255) * t))
        b = int(round(255 + (99 - 255) * t))
        # escolhe preto/branco pela LUMINÂNCIA real do fundo (WCAG) — o limiar fixo
        # t>0.40 punha branco sobre fundos claros demais (contraste ~1.6–2.3:1, abaixo
        # de AA), já que t é limitado por ceiling=0.55 (fundo nunca fica escuro).
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        fg = "#ffffff" if lum < 140 else "#1f2733"
        return "background-color:rgb(%d,%d,%d);color:%s" % (r, g, b, fg)

    def __init__(self, df, target="target", task_type="classification",
                 sample_col=None, ref_sample="DES",
                 feature_labels=None, problem_label=None, features=None, tree_samples=None, date_col=None,
                 allow_interactive_tree=None, weight_col=None):
        # task_type: "classification" (alvo binário) ou "regression" (alvo
        # contínuo) — define métricas, IV, cor e os gráficos exibidos.
        # tree_samples: amostras cujo alvo médio aparece nas folhas da árvore.
        # None = todas; ex.: tree_samples=["DES","OOT"] mostra só DES e OOT.
        # date_col: coluna de data/safra — FORA da modelagem, só p/ gráficos no tempo.
        # weight_col: coluna de PESO/EXPOSIÇÃO (ex.: saldo) — FORA da modelagem;
        # liga a visão dupla contratos × saldo no cartão da folha, na tabela de
        # folhas, no TSV e no Excel. Sem ela, a tela não muda em nada. FASE 1:
        # exibição apenas — split e binning ótimo seguem NÃO ponderados.
        self.task_type = task_type
        self._is_clf = task_type == "classification"
        self._risk_label = problem_label or target   # rótulo do alvo na UI
        self._risk_mean = f"média de {self._risk_label}"   # frase "média de X"
        self._tree_samples_cfg = tree_samples
        self.date_col = date_col
        # "Ver árvore" (aba Construir): o preview interativo usa anywidget, cujo
        # FRONTEND o gerenciador de widgets do Databricks busca de um CDN
        # (jsDelivr/unpkg) — num cluster SEM egress a requisição trava e o preview
        # "não carrega". Por isso, em Databricks, o preview cai por padrão no PNG
        # ESTÁTICO AUTOCONTIDO (data-URL via W.HTML, widget CORE do ipywidgets —
        # nunca busca rede), garantindo ZERO download externo. allow_interactive_tree:
        #   None  → automático: interativo fora do Databricks, estático no Databricks;
        #   True  → sempre tenta o interativo (só se o cluster tiver anywidget como
        #           lib instalada, ou tiver egress liberado);
        #   False → sempre estático (garante zero rede em qualquer ambiente).
        if allow_interactive_tree is None:
            allow_interactive_tree = not _running_in_databricks()
        self.allow_interactive_tree = bool(allow_interactive_tree)
        self._kwargs = dict(target=target, task_type=task_type, sample_col=sample_col,
                            ref_sample=ref_sample, feature_labels=feature_labels,
                            problem_label=problem_label,
                            date_col=date_col, weight_col=weight_col, verbose=False)
        self.df = df
        self.target = target
        self.sample_col = sample_col
        self.ref_sample = ref_sample
        self.weight_col = weight_col
        if features is None:
            skip = {target, sample_col, date_col, weight_col}
            skip.discard(None)
            # datas/datetime nunca são variáveis do modelo (mesmo sem date_col)
            skip |= {c for c in df.columns
                     if pd.api.types.is_datetime64_any_dtype(df[c])}
            features = [c for c in df.columns if c not in skip]
        self.features = features

        self.seg = TreeSegmenter(df, **self._kwargs)
        self.locked: set = set()
        # guarda do seletor de lado do "mover corte": trocar as options dispara o
        # observer, que não deve re-sincronizar enquanto a sincronia está em curso
        self._syncing_side = False
        # idem p/ o seletor de variável (ver _refresh_feature_options)
        self._syncing_feat_opts = False
        self._pending = None
        self.result = None
        self.spark_result = None      # último Spark DataFrame com a régua aplicada
        self.score_df = None          # base externa opcional p/ aplicar a régua em
                                      # memória (ui.score_df = df_novo), como no model
        self._spark_steps: list = []  # linhas da tabela de progresso do card Spark
        self._undo: list = []        # pilha de estados p/ desfazer splits/fusões
        self._redo: list = []        # pilha de estados p/ refazer
        self._log_lines: list = []   # buffer do console (últimas 40 linhas) — ver _log
        # cenários nomeados EM MEMÓRIA (nome → to_dict() + locked + linha-resumo):
        # vivem SÓ nesta sessão — fechar o notebook os descarta. Ver card "Cenários".
        self._scenarios: dict = {}

        # máscaras de amostra (fixas) e amostras ≠ referência (ex.: OOT)
        if sample_col is not None:
            self._samples = list(df[sample_col].dropna().unique())
            self._nonref = [a for a in self._samples if a != ref_sample]
            self._sample_masks = {a: (df[sample_col] == a) for a in self._samples}
            # amostras SEM variável resposta (ex.: ESTABILIDADE = público recente
            # só para validação): entram no PSI, mas não têm alvo para exibir.
            self._psi_only = [a for a in self._nonref
                              if df.loc[self._sample_masks[a], target].notna().sum() == 0]
            # não-referência COM alvo (entra nas células/colunas de alvo)
            self._pd_nonref = [a for a in self._nonref if a not in self._psi_only]
            # não-referência a EXIBIR na árvore (default: todas com alvo)
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
        # flag análoga p/ o campo de APELIDO da folha: o _sync_leaf_name_field
        # reatribui tx_leaf_name.value programaticamente (troca de folha, undo) e
        # o observer não pode tratar isso como digitação do usuário.
        self._suspend_name_obs = False
        # cache de HTML por widget (hash-and-skip): só reescreve .value quando o
        # conteúdo muda — evita reenviar blobs idênticos pelo comm kernel↔browser.
        self._last_html: dict = {}
        # cache do PNG (base64) do histograma da folha, por (sid, versão da árvore)
        self._leaf_hist_cache: dict = {}
        # cache do IV por variável usado pelo "ordenar por IV" dos seletores,
        # por (sid, versão da árvore) — lazy: só calcula com o toggle ligado
        self._iv_sort_cache: dict = {}

        self._build()
        self._on_mode_change(None)   # estado inicial de visibilidade dos controles
        self._sync_autoconc_visibility()   # sliders de concentração do auto-fit
        self._refresh()              # _refresh_iv já mescla o PSI/CSI por variável
        # contexto inicial (variável, folha) do guard do _on_feature_change: sem
        # ele, a 1ª re-rotulagem (ordenar por IV) invalidaria o preview à toa
        self._feat_ctx = (self._sel_feature(warn=False), self.dd_leaf.value)

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
        # APELIDO de negócio da folha selecionada (leaf_names do segmentador):
        # aplicado ao confirmar (Enter / perder o foco); texto vazio remove. O
        # apelido é atrelado ao sid da folha — fundir/dividir muda o sid e o
        # apelido é descartado (collapse que restaura o sid antigo o revive).
        self.tx_leaf_name = W.Text(
            description="Apelido", placeholder="apelido de negócio (opcional)",
            continuous_update=False,
            tooltip="Apelido de negócio da folha selecionada — aparece na árvore, "
                    "na tabela de folhas, no assign() e como comentário no SQL. "
                    "Vazio remove. Fundir/dividir a folha descarta o apelido "
                    "(o identificador da folha muda).",
            layout=W.Layout(width="98%"), style={"description_width": "52px"})
        # Seletor de variável: Dropdown, igual ao da folha — a lista inteira abre
        # com um clique em qualquer ponto do campo. Um Combobox foi tentado aqui
        # (busca por digitação), mas o <datalist> do navegador filtra as opções
        # pelo texto do campo: com uma variável já escolhida, o rótulo inteiro fica
        # no input e a lista exibia UMA opção só — impossível navegar. A ordenação
        # por IV (toggle ao lado) cobre a necessidade de achar a variável certa.
        # As opções são os NOMES DE EXIBIÇÃO (feature_labels); o mapa rótulo→coluna
        # fica em _feat_by_label e a resolução em _sel_feature.
        labels, self._feat_by_label = self._feature_option_labels(by_iv=False)
        self.dd_feature = W.Dropdown(description="Variável", options=labels,
                                     value=(labels[0] if labels else None),
                                     layout=W.Layout(width="100%", flex="1 1 auto"),
                                     style={"description_width": "62px"})
        # toggle "ordenar por IV": reordena as opções pelo IV da variável na folha
        # selecionada (memoizado por versão da árvore; cálculo lazy, só ao ligar)
        # e anexa o IV entre parênteses ao rótulo.
        self.tg_feat_iv = W.ToggleButton(
            value=False, description="ordenar por IV", icon="sort-amount-desc",
            tooltip="Reordena as opções pelo IV da variável na FOLHA selecionada "
                    "(calculado ao ligar e ao trocar de folha; memoizado por versão "
                    "da árvore) e mostra o IV entre parênteses no rótulo",
            layout=W.Layout(width="auto", margin="0 0 0 6px"))
        # mesma instância nas 2 views (card da aba + painel do preview da árvore)
        self.box_feature = W.HBox([self.dd_feature, self.tg_feat_iv],
                                  layout=W.Layout(width="100%", align_items="center"))
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
        _diff_tip = "taxa de default" if self._is_clf else "alvo médio"
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
                    "e cole no Excel: colunas certas, ponto decimal e % como fração",
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
        # ---- mover corte numérico (hi da folha ↔ lo da irmã à direita) ----
        # A folha tem DOIS cortes: o da esquerda (que ela divide com a irmã anterior)
        # e o da direita (com a próxima). O segmentador move sempre o `hi` de uma
        # folha, então mover o corte à ESQUERDA de X é mover o corte à DIREITA da
        # irmã anterior — mesma operação, outro dono. O seletor abaixo escolhe o
        # lado e a UI resolve quem é o dono; assim qualquer folha ajusta os dois
        # lados, inclusive a primeira (só direita) e a última (só esquerda).
        self.dd_move_side = W.Dropdown(description="corte", options=[("à direita ▶", "dir")],
                                       value="dir", layout=full, style=dstyle)
        self.dd_move_side.tooltip = ("Qual dos dois cortes da folha mover: o da esquerda "
                                     "(divisa com a irmã anterior) ou o da direita "
                                     "(divisa com a próxima).")
        self.lbl_move_cut = W.HTML()           # corte vigente + intervalo válido
        self.tx_move_cut = W.FloatText(description="novo corte", layout=full,
                                       style=dstyle)
        self.tx_move_cut.tooltip = ("Novo valor do corte — estritamente dentro do intervalo "
                                    "válido mostrado acima")
        self.btn_move_prev = mk("Preview corte", "info",
                                f"Mostra n e {self._risk_label} por amostra dos dois lados "
                                "do novo corte (não altera a árvore)", "eye")
        self.btn_move_cut = mk("Mover corte", "warning",
                               "Move o corte numérico p/ o valor digitado: ajusta o fim desta "
                               "folha e o início da irmã à direita (propagado aos sub-splits "
                               "da irmã), sem recolher o pai", "arrows-h")
        self.out_move_cut = W.HTML()           # preview/erros do mover corte
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
        self.btn_suggest3 = mk("Sugerir TOP 5 splits", "info",
                               "Lista as 5 melhores variáveis p/ dividir a folha selecionada", "lightbulb-o")
        self.out_suggest = W.HTML()
        self.btn_importance = mk("Calcular importância", "info",
                                 "Importância das variáveis que entraram na árvore", "bar-chart")
        self.out_importance = W.HTML()
        self.out_importance_chart = W.HTML()   # gráfico de importância relativa (ao lado da tabela)
        self.out_importance_legend = W.HTML()  # legenda explicativa (abaixo de tabela + gráfico)
        self.btn_sql = mk("Gerar SQL (CASE WHEN)", "primary",
                          "Gera a régua como SQL copiável", "database")
        self.tx_sql_table = W.Text(description="tabela", value="minha_tabela",
                                   layout=W.Layout(width="34%"), style=dstyle)
        # fallback p/ linhas não classificadas no scoring (categoria não vista /
        # faltante sem rota): NULL (padrão) ou a folha de pior risco. A escolha é
        # persistida no segmentador (self.seg.fallback → to_dict/save).
        _fb0 = getattr(self.seg, "fallback", None)
        self.dd_fallback = W.Dropdown(
            description="fallback p/ não classificados",
            options=[("NULL (padrão)", None), ("pior nota (maior risco)", "pior_nota")],
            value=("pior_nota" if _fb0 == "pior_nota" else None),
            tooltip="Destino das linhas que não caem em nenhuma folha ao aplicar "
                    "a régua (comum em OOT/produção)",
            layout=W.Layout(width="40%"), style={"description_width": "initial"})
        self.out_sql = W.Textarea(layout=W.Layout(width="99%", height="240px"))
        self.tx_diff_path = W.Text(description="árvore B (JSON)", placeholder="caminho do .json salvo",
                                   layout=W.Layout(width="95%"), style=dstyle)
        self.btn_diff = mk("Comparar com árvore B", "warning",
                           "Carrega outra árvore (JSON) e compara com a atual", "exchange")
        self.out_diff = W.HTML()
        # ---- cenários nomeados EM MEMÓRIA (card "Cenários", aba Avançado) ----
        self.tx_scn_name = W.Text(description="nome", placeholder="ex.: v1 · 8 folhas",
                                  layout=W.Layout(width="40%"), style=dstyle)
        self.btn_scn_save = mk("Salvar cenário", "success",
                               "Guarda a árvore ATUAL como cenário nomeado em memória "
                               "(vale só nesta sessão; não grava em disco)", "bookmark")
        self.out_scn_summary = W.HTML()   # mini-tabela resumo (atual + cenários)
        self.btn_scn_clear = mk("Limpar todos", "danger",
                                "Remove TODOS os cenários desta sessão (confirma em 2 "
                                "cliques). Cada linha da lista tem um ✕ para remover só "
                                "aquele cenário.", "trash")
        self.box_scn_list = W.VBox([])    # linhas: nome + Restaurar/Comparar/remover
        self.out_scn_diff = W.HTML()      # resultado da comparação cenário × atual
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
        self.cb_savebase = W.Checkbox(value=False, description="Salvar base DES + OOT",
                                      indent=False, layout=W.Layout(width="98%"))
        self.cb_savebase.tooltip = ("Loga as amostras de treino (DES) e validação (OOT) como "
                                    "artefatos parquet no run (útil p/ auditoria/reprodução).")
        self.btn_mlflow = mk("Salvar no MLflow", "primary",
                             "Loga régua, métricas e o modelo pyfunc, e registra a versão no Model Registry", "save")
        self.btn_clear_log = mk("Limpar log", "", "Limpa a área de preview/log", "eraser")
        self.sl_boot = W.IntSlider(description="reamostras", min=200, max=5000, step=100,
                                   value=1000, layout=full, style=dstyle)
        self.btn_boot = mk("Calcular IC bootstrap", "primary",
                           f"Calcula o IC de {self._risk_label} por folha e a aderência em OOT", "random")
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
                          f"Compara a {self._risk_mean} das folhas de mesmo pai por amostra e por "
                          "safra e sinaliza inversões da ordem de risco", "exchange")
        # zoom do eixo Y + eixo em % nos gráficos de ESTABILIDADE (folhas-irmãs):
        # o eixo cheio comprime as linhas quando a diferença entre folhas é pequena.
        self._sib_zoom = False
        _sib_zlay = W.Layout(width="80px")
        self.btn_sib_zoom = W.Button(description="🔍 Zoom", layout=W.Layout(width="92px"),
                                     tooltip="Aperta o eixo Y aos dados (revela cruzamentos "
                                             "comprimidos no eixo cheio)")
        self.btn_sib_reset = W.Button(description="Reset", layout=W.Layout(width="72px"),
                                      tooltip="Volta ao eixo cheio e limpa mín/máx")
        self.tx_sib_ymin = W.Text(value="", placeholder="mín", layout=_sib_zlay,
                                  tooltip="Limite inferior do eixo Y (na unidade exibida: % se "
                                          "'eixo em %' estiver ligado, senão a escala do alvo)")
        self.tx_sib_ymax = W.Text(value="", placeholder="máx", layout=_sib_zlay,
                                  tooltip="Limite superior do eixo Y")
        self.ck_sib_pct = W.Checkbox(value=False, indent=False, description="eixo em %",
                                     tooltip="Mostra o eixo Y em % — útil quando o alvo está em "
                                             "[0,1] (ex.: LGD)")
        # --- validação regulatória (monotonicidade, calibração) e relatório ---
        self.tx_time_col = W.Text(description="coluna tempo", value="dt_ref",
                                  layout=full, style=dstyle,
                                  placeholder="coluna de safra p/ o backtest do relatório (ex.: dt_ref)")
        self.btn_validate = mk("Validar (monotonicidade · calibração)", "info",
                               "Mostra monotonicidade das folhas e calibração prevista×realizada. "
                               "O backtest por safra sai no relatório (a figura fica ilegível "
                               "na tela quando há muitas safras).", "check-square-o")
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
        # CAP/Lift e métricas por safra valem nos DOIS task_type: usam a ordenação
        # do alvo previsto da folha contra o alvo (evento binário ou massa contínua)
        self.btn_cap = mk("Curva CAP (AR)", "info",
                          "Curva CAP/Lorenz da régua por amostra, com o AR (accuracy ratio) "
                          "na legenda", "signal")
        self.btn_lift = mk("Lift por decil", "info",
                           "Lift por decil de score (decil 1 = piores scores) + gains "
                           "acumulado no eixo secundário", "bar-chart")
        self.btn_msafra = mk("Métricas por safra", "info",
                             "Evolução das métricas da régua por safra (requer coluna de "
                             "data)", "calendar")
        # --- estabilidade: métricas por amostra · PSI da segmentação no tempo · concentração ---
        self.btn_estab = mk("Estabilidade & concentração", "info",
                            "Principais métricas por amostra, PSI da segmentação ao longo do "
                            "tempo e concentração das folhas entre amostras", "bar-chart")
        # seletor de métricas + eixo em % do gráfico "principais métricas por amostra"
        _mc_opts = ["KS", "AUC", "Gini"] if self._is_clf else ["RMSE", "MAE", "R2"]
        self.sm_estab_metrics = W.SelectMultiple(
            options=_mc_opts, value=tuple(_mc_opts), rows=3, description="Métricas:",
            style={"description_width": "initial"},
            layout=W.Layout(width="auto", min_width="130px"))
        self.sm_estab_metrics.tooltip = ("Quais métricas mostrar no gráfico 'principais "
                                         "métricas por amostra' (nenhuma marcada = todas).")
        self.ck_estab_pct = W.Checkbox(value=False, indent=False, description="métricas em %")
        self.ck_estab_pct.tooltip = ("Mostra TODAS as métricas do gráfico em % (útil p/ alvo "
                                     "em [0,1], ex.: LGD). Desmarcado = padrão por métrica.")
        # zoom do eixo Y dos gráficos de PSI (por safra e por amostra) — o eixo cheio
        # (0–100%) comprime as barras quando o PSI é pequeno
        self._psi_zoom = False
        _psi_zlay = W.Layout(width="80px")
        self.btn_psi_zoom = W.Button(description="🔍 Zoom PSI", layout=W.Layout(width="104px"),
                                     tooltip="Aperta o eixo Y dos gráficos de PSI (por safra e "
                                             "por amostra) aos dados")
        self.btn_psi_reset = W.Button(description="Reset", layout=W.Layout(width="72px"),
                                      tooltip="Volta o eixo dos PSI ao cheio (0–100%) e limpa mín/máx")
        self.tx_psi_ymin = W.Text(value="", placeholder="mín %", layout=_psi_zlay,
                                  tooltip="Limite inferior do eixo Y dos PSI, em % (ex.: 0)")
        self.tx_psi_ymax = W.Text(value="", placeholder="máx %", layout=_psi_zlay,
                                  tooltip="Limite superior do eixo Y dos PSI, em % (ex.: 5)")
        self.btn_varprofile = mk("Perfil das variáveis por safra", "info",
                                 "Grade por variável da árvore: % de missing (0–100%) e dispersão "
                                 "p5·média·p95 por safra (categóricas: área empilhada com legenda)",
                                 "chart-area")

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
        _suf = self._risk_label.lower()          # "pd" (clf) ou "lgd" (reg) no nome default
        self.tx_json_path = W.Text(description="arquivo", value=f"arvore_{_suf}.json",
                                   layout=full, style=dstyle, placeholder="caminho .json")
        self.btn_save_json = mk("Salvar árvore (JSON)", "success",
                                "Salva a estrutura da árvore num arquivo JSON", "save")
        self.btn_load_json = mk("Carregar árvore (JSON)", "info",
                                "Carrega uma árvore salva e reaplica ao DataFrame atual", "upload")
        # --- confirmação de sobrescrita INLINE (aba Exportar, sob os campos de
        # caminho): aviso + botões criados UMA vez e reutilizados a cada chamada
        # (antes o diálogo era desenhado no console e 2 botões novos vazavam por
        # chamada) — ver _confirm_overwrite.
        self.html_confirm = W.HTML()
        self.btn_confirm_yes = W.Button(description="Sobrescrever", button_style="danger",
                                        icon="exclamation-triangle")
        self.btn_confirm_no = W.Button(description="Cancelar", icon="times")
        self.box_confirm = W.VBox(
            [self.html_confirm, W.HBox([self.btn_confirm_yes, self.btn_confirm_no])],
            layout=W.Layout(display="none", width="100%"))
        self._confirm_pending = None      # {"path", "do_save", "owner"} enquanto aguarda
        # --- imagem da árvore (matplotlib) ---
        self.tx_img_path = W.Text(description="imagem", value=f"arvore_{_suf}.png",
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
        # --- exportação Excel multi-abas (.xlsx) — requer openpyxl (opcional) ---
        self.tx_xlsx_path = W.Text(description="arquivo", value=f"arvore_{_suf}.xlsx",
                                   layout=full, style=dstyle,
                                   placeholder="caminho .xlsx onde salvar o Excel")
        self.btn_xlsx = mk("Exportar Excel (.xlsx)", "primary",
                           "Gera um .xlsx multi-abas: Folhas, Métricas por amostra, PSI, "
                           "IV por variável, Calibração e Régua SQL — requer o pacote "
                           "opcional openpyxl", "file-excel-o")
        # --- aplicar a régua numa tabela Spark ("reconstruir as folhas") ---
        # inputs com mais respiro vertical (tabela/saída mais espaçadas)
        spark_lay = W.Layout(width="98%", margin="9px 0")
        self.tx_spark_in = W.Text(description="tabela", layout=spark_lay, style=dstyle,
                                  placeholder="tabela Spark de entrada (catalogo.schema.tabela)")
        self.tx_spark_out = W.Text(description="saída", layout=spark_lay, style=dstyle,
                                   placeholder="opcional: grava o resultado nesta tabela")
        self.btn_spark_apply = mk("Reconstruir folhas", "primary",
                                  f"Aplica a régua à tabela Spark (segmento, folha e {self._risk_label} por linha), "
                                  "desde que as colunas tenham o mesmo nome; sem o nome da tabela, "
                                  "aplica na base em memória (ui.score_df ou a carregada)", "table")
        self.out_spark_progress = W.HTML()   # tabela de progresso ⏳/✅/❌ por etapa
        self.out_spark = W.HTML()            # resultado/erro resumido + distribuição por folha
        # --- controles da aba "Análise de variáveis" ---
        # Dropdown (não Combobox) pelo mesmo motivo do seletor da aba Construir:
        # a lista toda tem de abrir num clique — rótulos de exibição; mapa em
        # _var_by_label.
        var_labels, self._var_by_label = self._feature_option_labels(by_iv=False)
        self.dd_var = W.Dropdown(description="Variável", options=var_labels,
                                 value=(var_labels[0] if var_labels else None),
                                 layout=full, style=dstyle)
        self.tg_var_iv = W.ToggleButton(
            value=False, description="ordenar por IV", icon="sort-amount-desc",
            tooltip="Reordena as opções pelo IV da variável na folha escolhida ao "
                    "lado (raiz = carteira inteira; memoizado por versão da árvore) "
                    "e mostra o IV entre parênteses no rótulo",
            layout=W.Layout(width="auto", margin="0 0 0 6px"))
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
                                   "Renderiza a árvore como imagem aqui mesmo — sem exportar/salvar. "
                                   "Com o anywidget instalado a imagem é CLICÁVEL: selecionar folha, "
                                   "fundir irmãs e recolher ramos direto nos nós",
                                   "sitemap")
        self.btn_tree_preview_hide = mk("Ocultar", "", "Oculta a imagem da árvore", "eye-slash")
        # --- barra de ações do preview INTERATIVO (aparece ao clicar num nó da imagem) ---
        self.btn_img_merge_l = mk("Fundir ← irmã", "warning",
                                  "Funde a folha clicada com a irmã adjacente à ESQUERDA", "compress")
        self.btn_img_merge_r = mk("Fundir irmã →", "warning",
                                  "Funde a folha clicada com a irmã adjacente à DIREITA", "compress")
        self.btn_img_merge_na = mk("Fundir missing", "",
                                   "Funde a folha-irmã de FALTANTES na folha clicada", "compress")
        self.btn_img_collapse = mk("Recolher", "danger",
                                   "Desfaz a quebra: numa FOLHA, recolhe o pai (a quebra que a criou); "
                                   "num RAMO, recolhe o ramo inteiro numa folha só", "level-up")
        self.btn_img_lock = mk("Travar", "",
                               "Trava/destrava a folha clicada como final (🔒 fica protegida da "
                               "poda e do auto-merge)", "lock")
        # clones de desfazer/refazer/auto-fit/reset para a barra: as instâncias
        # dos cards têm width 98% inline (uma por linha aqui); os clones ficam
        # compactos, lado a lado. Habilitação do desfazer/refazer espelhada por
        # dlink; auto-fit/reset usam os mesmos handlers (sem estado próprio).
        self.btn_img_undo = mk("◀ Desfazer", "", "Desfaz a última alteração na árvore", "undo")
        self.btn_img_redo = mk("Refazer ▶", "", "Refaz a alteração desfeita", "repeat")
        self.btn_img_autofit = mk("Auto-fit", "info",
                                  "Constrói a árvore gulosa por IV até a profundidade escolhida "
                                  "(folha selecionada ≠ raiz: cresce só aquela folha; raiz: "
                                  "reconstrói tudo)", "magic")
        self.btn_img_reset = mk("Resetar", "", "Recomeça a árvore do zero", "refresh")
        # fecha o painel de divisão (fica no cabeçalho do próprio painel)
        W.dlink((self.btn_undo, "disabled"), (self.btn_img_undo, "disabled"))
        W.dlink((self.btn_redo, "disabled"), (self.btn_img_redo, "disabled"))
        for _b in (self.btn_img_merge_l,
                   self.btn_img_merge_r, self.btn_img_merge_na, self.btn_img_collapse,
                   self.btn_img_lock, self.btn_img_undo, self.btn_img_redo,
                   self.btn_img_autofit, self.btn_img_reset):
            _b.layout.width = "auto"
            _b.layout.margin = "2px 8px 2px 0"     # respiro horizontal entre botões
        # pequena espaçada entre os GRUPOS: fundir-irmãs · fundir-missing · recolher
        self.btn_img_merge_na.layout.margin = "2px 8px 2px 18px"
        self.btn_img_collapse.layout.margin = "2px 8px 2px 18px"

        self.btn_preview.on_click(self._on_preview)
        self.btn_sugcuts.on_click(self._on_suggest_cuts)
        self.btn_split.on_click(self._on_split)
        self.btn_lock.on_click(self._on_lock)
        self.btn_unlock.on_click(self._on_unlock)
        # poda e reset são destrutivos: exigem confirmação em 2 cliques (o 1º
        # arma o botão como "Confirmar?"; sem o 2º em 5 s ele desarma sozinho)
        self.btn_prune.on_click(
            lambda b: self._confirm_twice(b, lambda: self._on_prune(None)))
        self.btn_reset.on_click(
            lambda b: self._confirm_twice(b, lambda: self._on_reset(None)))
        self.btn_export.on_click(self._on_export)
        self.dd_leaf.observe(self._on_leaf_change, names="value")
        self.tx_leaf_name.observe(self._on_leaf_name, names="value")
        self.dd_test.observe(lambda _: self._refresh_table(), names="value")
        self.btn_copy_table.on_click(self._on_copy_table)
        self.btn_collapse.on_click(self._on_collapse)
        self.btn_merge_l.on_click(lambda _: self._on_merge("left"))
        self.btn_merge_r.on_click(lambda _: self._on_merge("right"))
        self.btn_merge_na.on_click(self._on_merge_missing)
        self.btn_move_prev.on_click(self._on_move_cut_preview)
        self.btn_move_cut.on_click(self._on_move_cut)
        self.dd_move_side.observe(self._on_move_side, names="value")
        self.btn_suggest.on_click(self._on_suggest)
        self.btn_suggest3.on_click(self._on_suggest3)
        self.btn_importance.on_click(self._on_importance)
        self.btn_sql.on_click(self._on_sql)
        self.dd_fallback.observe(self._on_fallback, names="value")
        self.btn_diff.on_click(self._on_diff)
        self.btn_scn_save.on_click(self._on_scn_save)
        self.btn_scn_clear.on_click(self._on_scn_clear)
        self.btn_autofit.on_click(self._on_autofit)
        self.btn_mlflow.on_click(self._on_mlflow)
        self.btn_clear_log.on_click(self._on_clear_log)
        self.btn_boot.on_click(self._on_boot)
        self.btn_diag.on_click(self._on_diag)
        self.btn_diag_hide.on_click(self._on_diag_hide)
        self.btn_sib.on_click(self._on_sib_analyze)
        self.btn_sib_zoom.on_click(self._on_sib_zoom)
        self.btn_sib_reset.on_click(self._on_sib_reset)
        self.tx_sib_ymin.observe(lambda c: self._render_sib_charts(), names="value")
        self.tx_sib_ymax.observe(lambda c: self._render_sib_charts(), names="value")
        self.ck_sib_pct.observe(lambda c: self._render_sib_charts(), names="value")
        self.btn_estab.on_click(self._on_estab)
        # re-renderiza métricas + PSI ao mudar seletor/% ou zoom (só se já foi gerado)
        self.sm_estab_metrics.observe(lambda c: self._render_estab_charts(), names="value")
        self.ck_estab_pct.observe(lambda c: self._render_estab_charts(), names="value")
        self.btn_psi_zoom.on_click(self._on_psi_zoom)
        self.btn_psi_reset.on_click(self._on_psi_reset)
        self.tx_psi_ymin.observe(lambda c: self._render_estab_charts(), names="value")
        self.tx_psi_ymax.observe(lambda c: self._render_estab_charts(), names="value")
        self.btn_varprofile.on_click(self._on_varprofile)
        # amostras p/ a análise por safra das folhas-irmãs (fixas — não mudam
        # com a árvore): "todas" + a referência (DES) + as demais com alvo.
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
            # regressão: reusa o 1º botão da discriminação p/ o boxplot por folha
            # (ROC/KS não se aplica a alvo contínuo; o histograma do alvo foi
            # removido da UI — segue disponível via seg.plot_target_hist()).
            self.btn_roc.description = "📦 Boxplot por folha"
            self.btn_roc.tooltip = "Boxplot do alvo por folha — dispersão dentro de cada folha"
            self.btn_roc.on_click(self._on_box)
        # CAP · Lift · métricas por safra: comuns aos dois task_type
        self.btn_cap.on_click(self._on_cap)
        self.btn_lift.on_click(self._on_lift)
        self.btn_msafra.on_click(self._on_msafra)
        self.btn_undo.on_click(self._on_undo)
        self.btn_redo.on_click(self._on_redo)
        self.btn_automerge.on_click(self._on_automerge)
        self.btn_save_json.on_click(self._on_save_json)
        self.btn_load_json.on_click(self._on_load_json)
        self.btn_confirm_yes.on_click(self._on_confirm_yes)
        self.btn_confirm_no.on_click(self._on_confirm_no)
        self.btn_pdf.on_click(self._on_pdf)
        self.btn_xlsx.on_click(self._on_xlsx)
        self.btn_plot.on_click(self._on_plot)
        self.btn_plot_hide.on_click(self._on_plot_hide)
        self.btn_spark_apply.on_click(self._on_spark_apply)
        self.btn_var_analyze.on_click(self._on_var_analyze)
        self.btn_tree_preview.on_click(self._on_tree_preview)
        self.btn_tree_preview_hide.on_click(self._on_tree_preview_hide)
        self.btn_img_merge_l.on_click(lambda _: self._on_merge("left"))
        self.btn_img_merge_r.on_click(lambda _: self._on_merge("right"))
        self.btn_img_merge_na.on_click(self._on_merge_missing)
        self.btn_img_collapse.on_click(self._on_img_collapse)
        self.btn_img_lock.on_click(self._on_img_lock)
        self.btn_img_undo.on_click(self._on_undo)
        self.btn_img_redo.on_click(self._on_redo)
        self.btn_img_autofit.on_click(self._on_autofit)
        self.btn_img_reset.on_click(
            lambda b: self._confirm_twice(b, lambda: self._on_reset(None)))
        self.tg_mode.observe(self._on_mode_change, names="value")
        self.dd_feature.observe(self._on_feature_change, names="value")
        # ordenar por IV: recalcula as opções ao ligar/desligar; com o toggle
        # ligado, trocar a folha de análise também reordena (lazy + memoizado)
        self.tg_feat_iv.observe(lambda _: self._refresh_feature_options(), names="value")
        self.tg_var_iv.observe(lambda _: self._refresh_var_options(), names="value")
        self.dd_var_leaf.observe(self._on_var_leaf_iv, names="value")
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
        # botão para calcular o IV/PSI por variável da folha SEM precisar abrir a aba
        # "Análise de variáveis" (o cálculo é caro, por isso fica sob demanda).
        self.btn_iv_refresh = W.Button(
            description="Atualizar", button_style="info", icon="refresh",
            tooltip="Calcula o IV e o PSI (OOT/estabilidade) de todas as variáveis na "
                    "folha selecionada — sem precisar abrir a aba Análise de variáveis",
            layout=W.Layout(width="auto"))
        self.btn_iv_refresh.on_click(lambda _: self._compute_iv())
        self.out_leaf_hist = W.HTML()                     # alvo da folha
        self.out_plot = W.HTML()
        self.out_boot = W.HTML()
        self.out_validate = W.HTML()
        self.out_discrim = W.HTML()    # ROC/KS (clf) · boxplot/histograma do alvo (reg)
        self.out_sib = W.HTML()     # comparação de folhas-irmãs (inversão)
        self.out_estab = W.HTML()   # métricas por amostra | PSI da segmentação no tempo
        self.out_conc = W.HTML()    # concentração das folhas entre amostras
        self.out_varprof_missing = W.HTML()   # % missing por safra (grade de variáveis da árvore)
        self.out_varprof_stats = W.HTML()     # dispersão p5·média·p95 · proporção cat. por safra
        self.out_diag = W.HTML()    # placar de saúde do modelo (Diagnóstico)
        self.out_log = W.Output(layout=W.Layout(max_height="320px", overflow="auto"))
        self.out_preview_chart = W.HTML()   # distribuição da variável + cortes (ao lado do histograma)
        self.out_preview_seg = W.HTML()     # segmentação proposta (dentro de "Dividir a folha")
        self.out_table = W.HTML()
        # aba "Análise de variável"
        self.out_var_dist = W.HTML()          # distribuição & badrate (comportamento)
        self.out_var_time = W.HTML()          # percentis por safra
        self.out_var_psi = W.HTML()           # PSI por safra
        self.out_var_cards = W.HTML()         # resumo & estabilidade
        self.out_var_inv_s = W.HTML()         # inversão por amostra
        self.out_var_inv_t = W.HTML()         # inversão por safra
        self.out_var_optbin = W.HTML()        # distribuição acumulada das faixas (optbin)
        self.out_tree_img = W.HTML()                      # preview da árvore (estático/erros)
        # preview INTERATIVO da árvore: widget clicável (anywidget, lazy) + barra
        # de ações contextual. Sem anywidget, out_tree_img segue com o PNG estático.
        self._tree_img_widget = None       # instância do widget (criada no 1º preview)
        self._img_selected = None          # sid do nó clicado na imagem (folha OU ramo)
        # aba "Árvore interativa": canvas navegável (anywidget, lazy — só é
        # montado quando a aba é aberta) + painel de criação do nó clicado
        self._cv_widget = None
        self._cv_sel = None                # sid do nó em foco no painel
        self._cv_syncing = False           # guarda dos observers durante a sincronia
        self._cv_syncing_side = False      # idem p/ o seletor de lado do mover corte
        self._cv_prev_tbl = None           # tabela das faixas do último preview
        self._cv_cat_ctx = None            # (variável, folha) do agrupador de categorias
        self._cv_cat_widgets: dict = {}    # categoria → Dropdown de grupo
        # self.allow_interactive_tree foi resolvido no __init__ (Databricks → PNG
        # estático autocontido, sem CDN do anywidget) — ver comentário lá.
        # barra de ações em 2 linhas: 1. chip do nó + ações principais da folha;
        # 2. estrutura (fusões/recolher) · histórico (desfazer/refazer) · árvore
        # inteira (auto-fit/resetar — as MESMAS instâncias dos cards, em 2ª view).
        self.tree_img_info = W.HTML(layout=W.Layout(flex="1 1 auto", min_width="0",
                                                    overflow="hidden"))
        _vsep = lambda: W.HTML("<div class='treeui-vsep'></div>")  # noqa: E731
        _row_lay = W.Layout(flex_flow="row wrap", align_items="center", width="100%")
        # zoom do preview: árvore grande vira nó ilegível no ajuste-ao-card. O
        # slider amplia a MESMA imagem (hotspots acompanham, pois são em %) e o
        # scroller passa a rolar nos dois eixos.
        self.sl_tree_zoom = W.FloatSlider(value=1.0, min=1.0, max=4.0, step=0.25,
                                          description="zoom", readout_format=".0%",
                                          continuous_update=False,
                                          layout=W.Layout(width="230px"),
                                          style={"description_width": "38px"})
        self.sl_tree_zoom.tooltip = ("Amplia o preview da árvore. Acima de 100% a imagem "
                                     "rola dentro do card — útil para ler os nós de "
                                     "árvores com muitas folhas.")
        self.btn_tree_zoom_reset = mk("Ajustar", "", "Volta o zoom a 100% (ajustado ao card)",
                                      "compress")
        self.btn_tree_zoom_reset.layout.width = "auto"     # senão ocupa a linha inteira
        self.btn_tree_zoom_reset.layout.margin = "2px 8px 2px 0"
        self.sl_tree_zoom.observe(self._on_tree_zoom, names="value")
        self.btn_tree_zoom_reset.on_click(lambda _: setattr(self.sl_tree_zoom, "value", 1.0))
        self.tree_img_bar = W.VBox([
            W.HBox([self.tree_img_info, self.btn_img_lock], layout=_row_lay),
            W.HBox([self.btn_img_merge_l, self.btn_img_merge_r, self.btn_img_merge_na,
                    self.btn_img_collapse, _vsep(), self.btn_img_undo,
                    self.btn_img_redo, self.btn_img_autofit, self.btn_img_reset,
                    _vsep(), self.sl_tree_zoom, self.btn_tree_zoom_reset],
                   layout=_row_lay),
        ], layout=W.Layout(display="none", margin="0 0 6px 0"))
        self.tree_img_bar.add_class("treeui-imgbar")
        self.box_tree_img = W.VBox([self.tree_img_bar, self.out_tree_img])
        self.cat_box = W.VBox([], layout=W.Layout(width="98%", display="none",
                                                  border="1px solid var(--hair)",
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
        _bg_logo, _bg_titulo = self._risk_label, f"Segmentação de {self._risk_label}"
        if self._is_clf:
            _bg_sub = "optimal binning binário · KS/AUC ao vivo"
        else:
            _bg_sub = "optimal binning contínuo · MAE/RMSE/R² ao vivo"
        banner = W.HTML(_CSS +
            f"<div class='treeui-banner'><div class='logo'>{_bg_logo}</div>"
            f"<div><div class='t'>{_bg_titulo}</div>"
            f"<div class='s'>Interface de construção de modelos de árvore — cultive a sua "
            f"Yggdrasil, ramo a ramo, para os seus modelos de crédito · {_bg_sub} · "
            f"PSI ao vivo ({self.ref_sample}) · teste de hipótese entre folhas adjacentes"
            "</div></div></div>")
        bar_box = W.VBox([self.bar]); bar_box.add_class("treeui-bar")

        # ---- legendas reutilizadas (task-aware) -------------------------
        _rl = self._risk_label
        if self._is_clf:
            _iv_intro = "<b>IV</b> (optbinning · WoE binário) = poder de"
            _iv_faixas = ("Faixas (Siddiqi): <span style='color:var(--ok-tx)'>forte (0,3–0,5)</span> · "
                          "<span style='color:var(--warn-tx)'>médio (0,1–0,3)</span> · fraco/inútil (&lt;0,1) · "
                          "<span style='color:var(--sus-tx)'>suspeito (&ge;0,5)</span> (alto demais, verifique vazamento).")
        else:
            _iv_intro = "<b>IV</b> (optbinning · contínuo) = poder de"
            _iv_faixas = ("Faixas: <span style='color:var(--ok-tx)'>forte (0,1–0,35)</span> · "
                          "<span style='color:var(--warn-tx)'>médio (0,03–0,1)</span> · fraco/inútil (&lt;0,03) · "
                          "<span style='color:var(--sus-tx)'>suspeito (&ge;0,35)</span>.")
        # o "o que é" vem antes das faixas: quem abre a aba pela primeira vez
        # precisa da definição, não do corte.
        _iv_oque = ("<i>Information Value</i> mede o quanto uma variável "
                    "<b>separa bom de mau</b>: divide-se a variável em faixas e compara-se, "
                    "faixa a faixa, a concentração de cada grupo. Se as faixas concentram "
                    "grupos diferentes, a variável carrega informação e o IV sobe; se todas "
                    "se parecem, o IV vai a zero. É o critério para escolher <b>qual variável "
                    "usar na próxima quebra</b>.")
        iv_legend = W.HTML(
            f"<div class='treeui-legend'>{_iv_oque}</div>"
            "<div class='treeui-legend' style='margin-top:6px'>"
            f"{_iv_intro} "
            f"separação da variável na <b>folha selecionada</b> (★ = maior). {_iv_faixas} "
            "<b>bins</b> = nº de faixas ideais do binning ótimo na folha.</div>"
            "<div class='treeui-legend' style='margin-top:6px;padding-top:6px;"
            f"border-top:1px solid var(--hair)'><b>PSI</b> = estabilidade da variável "
            f"({self.ref_sample} × demais amostras), calculado <b>nos bins fixados no "
            f"desenvolvimento</b> ({self.ref_sample}) — os mesmos do IV, para que a comparação "
            "meça deslocamento da população e não mudança de régua. Pior caso: "
            "<span style='color:var(--ok-tx)'>&lt;0.10 estável</span> · "
            "<span style='color:var(--warn-tx)'>0.10–0.25 atenção</span> · "
            "<span style='color:var(--bad-tx)'>&ge;0.25 instável</span>.</div>")

        # ================================================================
        # ABA 1. CONSTRUIR — "Cockpit em T"
        #   TOPO: Árvore & quebras  ·AO LADO·  Information Value  + régua da folha
        #   DETALHE: folha (detalhe) | dividir | ações + auto-fit  e, abaixo,
        #     distribuição da variável+cortes | histograma do alvo da folha
        #   RODAPÉ: Preview da árvore (imagem) em largura total
        # ================================================================
        sep_top = W.HTML("<div class='treeui-band'>1. Topo · loop árvore → folha → IV → agir</div>")

        tree_scroll = W.Box([self.out_tree],
                            layout=W.Layout(overflow="auto", width="100%",
                                            max_height="420px"))
        card_tree = W.VBox([
            W.HTML("<div class='treeui-h'>Árvore &amp; quebras</div>"),
            tree_scroll,
        ], layout=W.Layout(width="54%"))
        card_tree.add_class("treeui-card")
        iv_scroll = W.Box([self.out_iv],
                          layout=W.Layout(overflow="auto", width="100%",
                                          max_height="360px"))
        card_iv = W.VBox([
            W.HBox([W.HTML("<div class='treeui-h' style='margin:0;flex:1'>Information Value "
                           "· qual variável segmentar</div>"),
                    self.btn_iv_refresh],
                   layout=W.Layout(align_items="center", width="100%")),
            iv_legend, iv_scroll,
        ], layout=W.Layout(width="44%"))
        card_iv.add_class("treeui-card")
        top_cols = W.HBox([card_tree, card_iv],
                          layout=W.Layout(width="100%", align_items="flex-start",
                                          justify_content="space-between"))

        # ---- DETALHE · linha 1: folha (detalhe) | dividir | ações + auto-fit
        sep_det = W.HTML("<div class='treeui-band treeui-band-muted'>2. Detalhe / inspeção — "
                         "role quando precisar</div>")
        card_leaf = W.VBox([self.leaf_header, self.tx_leaf_name])
        card_leaf.add_class("treeui-card")
        det_c1 = W.VBox([card_leaf], layout=W.Layout(width="30%"))

        self.btn_sugcuts.layout.width = "99%"
        self.btn_suggest.layout.width = "99%"
        card_split = W.VBox([
            W.HTML("<div class='treeui-h'>Dividir a folha selecionada</div>"),
            self.dd_leaf, self.box_feature, self.btn_suggest, self.btn_sugcuts, self.tg_mode,
            self.sl_bins, self.dd_split_criterion,
            self.cb_minbin, self.sl_minbin, self.cb_maxbin, self.sl_maxbin,
            self.cb_mindiff, self.sl_mindiff,
            self.tx_cuts, self.cat_box,
            W.HBox([self.btn_preview, self.btn_split]),
            self.out_preview_seg,
        ]); card_split.add_class("treeui-card")
        det_c2 = W.VBox([card_split], layout=W.Layout(width="44%"))

        for _b in (self.btn_lock, self.btn_unlock, self.btn_collapse,
                   self.btn_merge_l, self.btn_merge_r, self.btn_merge_na,
                   self.btn_move_prev, self.btn_move_cut):
            _b.layout.width = "100%"
            _b.layout.margin = "2px 0"
        card_actions = W.VBox([
            W.HTML("<div class='treeui-h'>Ações da folha</div>"),
            W.HBox([self.btn_undo, self.btn_redo]),
            self.btn_lock, self.btn_unlock, self.btn_collapse,
            W.HBox([self.btn_merge_l, self.btn_merge_r]),   # fundir ◀ / ▶ lado a lado
            self.btn_merge_na,
            self.lbl_move_cut, self.dd_move_side, self.tx_move_cut,
            W.HBox([self.btn_move_prev, self.btn_move_cut]),  # preview / mover lado a lado
            self.out_move_cut,
        ], layout=W.Layout(width="100%"))
        card_actions.add_class("treeui-card")
        # Ações automáticas sobre a árvore inteira, num card só: crescer (auto-fit),
        # fundir irmãs indistinguíveis (auto-merge) e podar folhas pequenas. As três
        # eram cards separados — as duas últimas moravam no Avançado, longe do
        # ponto onde a árvore é construída. O seletor troca a configuração exibida;
        # cada modo mantém seus próprios controles e seu botão de ação.
        self.tg_build_mode = W.ToggleButtons(
            options=[("Auto-fit", "fit"), ("Auto-merge", "merge"), ("Podar", "prune")],
            value="fit", style={"button_width": "auto"},
            layout=W.Layout(width="100%"))
        self.tg_build_mode.tooltip = ("Auto-fit cresce a árvore · Auto-merge funde irmãs "
                                      "indistinguíveis · Podar remove folhas pouco "
                                      "representativas")
        self.box_build_cfg = W.VBox([])
        self.tg_build_mode.observe(lambda _: self._sync_build_mode(), names="value")
        self._build_panels = {
            "fit": (
                W.HTML("<div class='treeui-legend'>Constrói a árvore gulosa por IV até a "
                       "profundidade escolhida. As concentrações são <b>% da carteira inteira</b>: "
                       "<b>mín.</b> evita folhas terminais pequenas; <b>máx.</b> impede que uma "
                       "quebra concentre demais. Com uma <b>folha selecionada</b> (≠ raiz), cresce "
                       "<b>apenas aquela folha</b>; na raiz, reconstrói tudo.</div>"),
                self.sl_depth, self.dd_criterion,
                self.cb_autoconc_min, self.sl_autoconc_min,
                self.cb_autoconc_max, self.sl_autoconc_max,
                W.HBox([self.btn_autofit, self.btn_reset]),
            ),
            "merge": (
                W.HTML("<div class='treeui-legend'>Funde folhas-irmãs com risco estatisticamente "
                       "<b>indistinguível</b> (p &gt; α no teste entre adjacentes) ou com diferença "
                       f"de {_rl} abaixo do <b>Δ{_rl} mínimo</b>.</div>"),
                self.sl_alpha, self.sl_gap, self.cb_automerge_na, self.btn_automerge,
            ),
            "prune": (
                W.HTML("<div class='treeui-legend'>Funde com a irmã as folhas com "
                       "representatividade abaixo do <b>min repr%</b> ou com diferença de "
                       f"{_rl} menor que o <b>Δ{_rl} mínimo</b> — o mesmo limiar do "
                       "Auto-merge. Folhas fechadas (🔒) são preservadas.</div>"),
                self.sl_repr, self.sl_gap, self.btn_prune,
            ),
        }
        card_autofit = W.VBox([
            W.HTML("<div class='treeui-h'>Construir a árvore automaticamente</div>"),
            self.tg_build_mode, self.box_build_cfg,
        ]); card_autofit.add_class("treeui-card")
        self._sync_build_mode()
        det_c3 = W.VBox([card_actions, card_autofit], layout=W.Layout(width="24%"))

        det_row = W.HBox([det_c1, det_c2, det_c3],
                         layout=W.Layout(width="100%", align_items="flex-start",
                                         justify_content="space-between"))

        # ---- DETALHE · linha 2: distribuição+cortes (preview) | histograma do alvo
        card_preview = W.VBox([
            W.HTML("<div class='treeui-h'>Distribuição da variável · cortes sugeridos</div>"),
            W.HTML("<div class='treeui-legend'>Distribuição da variável na folha selecionada "
                   f"({self.ref_sample}), com os cortes propostos marcados.</div>"),
            self.out_preview_chart,
        ], layout=W.Layout(width="49%")); card_preview.add_class("treeui-card")
        if self._is_clf:
            _hist_h = f"{self._risk_label} da folha (taxa de default · {self.ref_sample})"
            _hist_leg = (f"Taxa de default da folha selecionada ({self.ref_sample}), com IC "
                         f"de Wilson e a {self._risk_label} da carteira como referência.")
        else:
            _hist_h = f"{self._risk_label} da folha (alvo médio · {self.ref_sample})"
            _hist_leg = (f"Distribuição do alvo na folha selecionada ({self.ref_sample}), "
                         "com a média da folha e a da carteira como referência.")
        card_hist = W.VBox([
            W.HTML(f"<div class='treeui-h'>{_hist_h}</div>"),
            W.HTML(f"<div class='treeui-legend'>{_hist_leg}</div>"),
            self.out_leaf_hist,
        ], layout=W.Layout(width="49%")); card_hist.add_class("treeui-card")
        det_bottom = W.HBox([card_preview, card_hist],
                            layout=W.Layout(width="100%", align_items="stretch",
                                            justify_content="space-between"))

        # ---- Assistente (sugerir · auto-fundir · podar): controles DISTRIBUÍDOS --
        # btn_suggest → card "Dividir a folha" (acima) · sl_alpha/sl_gap/
        # cb_automerge_na/btn_automerge → card "Auto-merge" (aba Avançado) ·
        # sl_repr/btn_prune → card "Poda por representatividade" (aba Avançado) ·
        # dd_test → card da tabela de folhas (aba Diagnóstico)

        # ---- RODAPÉ: Preview da árvore (imagem), largura total -----------
        sep_img = W.HTML("<div class='treeui-band treeui-band-muted'>3. Preview da árvore — "
                         "imagem em largura total</div>")
        self.btn_tree_preview.layout.width = "auto"
        self.btn_tree_preview_hide.layout.width = "auto"
        # painel compacto "Dividir a folha" do preview: as MESMAS instâncias do
        card_tree_img = W.VBox([
            W.HBox([W.HTML("<div class='treeui-h' style='margin:0;flex:1'>Preview da árvore "
                           "(imagem)</div>"),
                    self.btn_tree_preview, self.btn_tree_preview_hide],
                   layout=W.Layout(align_items="center", width="100%")),
            self.box_tree_img,
        ], layout=W.Layout(width="100%")); card_tree_img.add_class("treeui-card")

        tab_build = W.VBox([sep_top, self.leaf_chips, top_cols,
                            sep_det, det_row, det_bottom, sep_img, card_tree_img])

        # ================================================================
        # ABA 3. DIAGNÓSTICO — folhas · discriminação · métricas · bootstrap · qualidade
        # ================================================================
        _ref = self.ref_sample
        # visão dupla contratos × saldo: só entra na legenda com weight_col definida
        _wleg = ""
        if self.seg.weight_col is not None:
            _wc = _esc(str(self.seg.weight_col))
            _wleg = (f"<br><b>% saldo</b> e <b>… pond.</b> = visão dupla <b>contratos "
                     f"&times; saldo</b>: o % do saldo da folha (soma de <b>{_wc}</b> "
                     f"sobre o total) e o {self._risk_label} médio <b>ponderado</b> por "
                     f"<b>{_wc}</b>. Compare com o % de contratos: divergência grande "
                     "= a folha concentra exposição fora de proporção ao nº de "
                     "contratos. A <b>árvore em si continua não ponderada</b> — "
                     "critério de split e binning ótimo olham contrato a contrato.")
        tbl_legend = W.HTML(
            "<div class='treeui-legend'>"
            f"<b>PSI por amostra</b> (estabilidade da folha entre {_ref} e a amostra): "
            "<span style='background:var(--ok-bg);padding:1px 5px;border-radius:3px'>&lt;0.10 estável</span> "
            "<span style='background:var(--warn-bg);padding:1px 5px;border-radius:3px'>0.10–0.25 atenção</span> "
            "<span style='background:var(--bad-bg);padding:1px 5px;border-radius:3px'>&ge;0.25 instável</span>"
            # cada teste em seu bloco, com H₀ e a leitura do p em linhas próprias:
            # num parágrafo corrido a hipótese nula se perdia no meio do texto
            "<div style='margin-top:7px'><b>p (irmãs)</b> = p-valor de um "
            "<b>teste de hipótese</b> que compara a <b>distribuição do alvo</b> da folha "
            f"com a da <b>irmã adjacente</b> (mesmo pai, na amostra de referência {_ref}). "
            "O teste é o <b>Mann-Whitney U</b> (não-paramétrico, padrão) ou o <b>t de Welch</b> "
            "(médias, variâncias desiguais) — escolha no seletor <b>Teste</b>."
            f"<div style='margin-top:3px'>H₀: as duas irmãs têm {self._risk_label} igual.</div>"
            "<div style='margin-top:3px'>"
            "<span style='background:var(--bad-bg);padding:1px 5px;border-radius:3px'>p alto (&gt;0,05, em vermelho)</span> "
            "⇒ <b>não</b> dá para distinguir as irmãs ⇒ candidatas a fusão &nbsp;·&nbsp; "
            "<span style='color:var(--ok-tx)'>p baixo</span> ⇒ folhas bem separadas.</div>"
            "<div style='margin-top:3px'>Só <b>irmãs</b> são comparadas (a última de cada "
            "grupo e o nó de faltantes ficam em branco).</div></div>"
            f"<div style='margin-top:7px'><b>p ({_ref}×OOT)</b> = p-valor de um teste de "
            "hipótese da <b>aderência da estimativa</b>: compara a <b>distribuição do alvo</b> "
            f"da MESMA folha entre <b>{_ref}</b> e <b>OOT</b> (mesmo teste do seletor)."
            f"<div style='margin-top:3px'>H₀: a folha tem {self._risk_label} igual em "
            f"{_ref} e OOT.</div>"
            "<div style='margin-top:3px'>Semântica <b>inversa</b> à do p (irmãs): "
            "<span style='color:var(--ok-tx)'>p alto (&gt;0,05)</span> ⇒ estimativa "
            "<b>estável</b> entre as amostras &nbsp;·&nbsp; "
            "<span style='background:var(--bad-bg);padding:1px 5px;border-radius:3px'>p baixo (em vermelho)</span> "
            f"⇒ a estimativa <b>deslocou</b> de {_ref} para OOT (folha pouco aderente).</div></div>"
            + _wleg + "</div>")
        table_scroll = W.Box([self.out_table],
                             layout=W.Layout(overflow="auto", width="100%",
                                             max_height="420px"))
        # o seletor "Teste" (citado na legenda acima) mora AQUI: escolhe o teste
        # usado nas colunas p (irmãs) e p (DES×OOT) — trocar recalcula a tabela.
        self.dd_test.layout = W.Layout(width="300px")
        card_table = W.VBox([W.HTML("<div class='treeui-h'>Folhas criadas · PSI &amp; teste de hipótese (irmãs)</div>"),
                             tbl_legend,
                             W.HBox([self.dd_test]),
                             table_scroll,
                             W.HBox([self.btn_copy_table]), self.out_table_tsv])
        card_table.add_class("treeui-card")

        sib_legend = W.HTML(
            f"<div class='treeui-legend'>Compara o <b>{_rl} médio</b> das folhas de um mesmo "
            "pai (<b>folhas-irmãs</b>) e checa se a <b>ordem de risco</b> se mantém. "
            f"A ordem de <b>referência</b> é o {_rl} na <b>{_ref}</b>; uma <b>inversão</b> ocorre "
            f"quando, numa amostra ou safra, uma folha de menor risco passa a ter {_rl} "
            "<i>maior</i> que uma irmã de maior risco (as linhas se cruzam). "
            f"O gráfico da esquerda mostra o {_rl} por <b>amostra</b> ({_ref}, OOT, …) e o da "
            "direita por <b>safra</b> ao longo do tempo (faixas vermelhas = safras com "
            "inversão). O <b>indicador</b> resume: "
            "<span style='background:var(--ok-bg);padding:1px 5px;border-radius:3px'>verde sem inversão</span> "
            "<span style='background:var(--warn-bg);padding:1px 5px;border-radius:3px'>amarelo inverte em algumas safras</span> "
            "<span style='background:var(--bad-bg);padding:1px 5px;border-radius:3px'>vermelho inverte entre amostras ou em muitas safras</span>.</div>")
        card_sib = W.VBox([
            W.HTML("<div class='treeui-h'>Folhas-irmãs · inversão entre amostras &amp; safras</div>"),
            sib_legend,
            W.HBox([self.dd_sib_group], layout=W.Layout(width="100%")),
            W.HBox([self.tx_sib_time, self.dd_sib_sample],
                   layout=W.Layout(width="100%")),
            W.HBox([self.btn_sib]),
            # zoom do eixo Y (auto/mín-máx) + eixo em % dos gráficos de estabilidade
            W.HBox([self.btn_sib_zoom, self.btn_sib_reset, self.tx_sib_ymin,
                    self.tx_sib_ymax, self.ck_sib_pct],
                   layout=W.Layout(align_items="center", flex_flow="row wrap")),
            self.out_sib,
        ], layout=W.Layout(width="100%"))
        card_sib.add_class("treeui-card")
        self._card_sib = card_sib

        if self._is_clf:
            _dh = "Discriminação · ROC · KS · CAP · Lift · métricas por safra"
            discrim_legend = W.HTML(
                f"<div class='treeui-legend'>Poder de <b>ordenação de risco</b> da régua (score = {self._risk_label} "
                "previsto por folha). <b>KS</b> = máxima separação entre as acumuladas de bons e "
                "maus; <b>AUC</b>/<b>Gini</b> = área sob a ROC; <b>CAP</b>/AR e <b>lift</b> = "
                "concentração de eventos nos piores scores; <b>métricas por safra</b> acompanha "
                "KS/AUC no tempo. Avalie quando a árvore estiver fechada.</div>")
        else:
            _dh = "Dispersão e ordenação do alvo · boxplot · CAP · Lift · métricas por safra"
            discrim_legend = W.HTML(
                "<div class='treeui-legend'>Dispersão do <b>alvo</b> por folha — curva ROC/KS "
                "não se aplica a alvo contínuo. <b>Boxplot por folha</b> mostra mediana, quartis e "
                "outliers de cada folha; a <b>curva CAP</b> e o <b>lift</b> usam a ordenação do "
                "previsto contra a massa do alvo contínuo; <b>métricas por safra</b> acompanha "
                "RMSE/R² no tempo. Avalie quando a árvore estiver fechada.</div>")
        # clf: ROC + KS · reg: boxplot por folha — CAP/Lift/métricas por safra nos dois
        _discrim_btns = ([self.btn_roc, self.btn_ks] if self._is_clf else [self.btn_roc])
        _discrim_btns += [self.btn_cap, self.btn_lift, self.btn_msafra]
        for _b in _discrim_btns:        # o width 98% do mk quebraria o wrap (1/linha)
            _b.layout.width = "auto"
        card_discrim = W.VBox([
            W.HTML(f"<div class='treeui-h'>{_dh}</div>"),
            discrim_legend,
            W.HBox(_discrim_btns, layout=W.Layout(flex_flow="row wrap", gap="4px")),
            self.out_discrim,
        ])
        card_discrim.add_class("treeui-card")

        if self._is_clf:
            _ml = (f"a régua prediz {self._risk_label} pela taxa de default do segmento na referência ({_ref}); "
                   "avaliada como modelo em cada amostra · <b>KS</b>/<b>AUC</b>/<b>Gini</b> "
                   "altos = a segmentação ordena bem o risco · <b>Acurácia</b>/<b>F1</b> no "
                   "corte KS-ótimo")
            _mh = f"Discriminação (régua como modelo de {self._risk_label})"
        else:
            _ml = (f"a régua prediz {self._risk_label} pela média do alvo no segmento (referência {_ref}); "
                   "avaliada como modelo em cada amostra · <b>MAE</b>/<b>RMSE</b> menores e "
                   "<b>R²</b> maior = a régua reproduz melhor o alvo")
            _mh = f"Desempenho (régua como modelo de {self._risk_label})"
        metrics_legend = W.HTML(f"<div class='treeui-legend'>{_ml}</div>")
        card_metrics = W.VBox([
            W.HTML(f"<div class='treeui-h'>{_mh}</div>"),
            metrics_legend, self.out_metrics])
        card_metrics.add_class("treeui-card")

        if self._is_clf:
            boot_legend = W.HTML(
                f"<div class='treeui-legend'>IC de {self._risk_label} (taxa de default) por folha via bootstrap na "
                f"referência ({_ref}). Se houver OOT, mostra {self._risk_label} de OOT e verifica a "
                "<b>aderência</b>: <span style='color:var(--ok-tx)'>dentro</span> do IC = estável; "
                f"<span style='color:var(--bad-tx)'>acima/abaixo</span> = {self._risk_label} deslocou além da incerteza "
                "amostral. Calcule quando a árvore estiver fechada.</div>")
        else:
            boot_legend = W.HTML(
                f"<div class='treeui-legend'>IC de {self._risk_label} (alvo) por folha via bootstrap na "
                f"referência ({_ref}). Se houver OOT, mostra {self._risk_label} de OOT e verifica a "
                "<b>aderência</b>: <span style='color:var(--ok-tx)'>dentro</span> do IC = estável; "
                f"<span style='color:var(--bad-tx)'>acima/abaixo</span> = {self._risk_label} deslocou além da incerteza "
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
                        else "dispersão do alvo (boxplot)")
        sep_diag2 = W.HTML("<div class='treeui-band treeui-band-muted'>Evidência detalhada · "
                           f"folhas · {_diag_detail} · métricas · IC bootstrap</div>")
        # ---- estabilidade: métricas por amostra | PSI da segmentação no tempo · concentração ----
        card_estab = W.VBox([
            W.HTML("<div class='treeui-h'>Estabilidade · métricas por amostra · PSI no tempo · "
                   "concentração das folhas</div>"),
            W.HTML("<div class='treeui-legend'>À esquerda as <b>principais métricas por amostra</b> "
                   f"({_diag_metrics}); à direita o <b>PSI da segmentação ao longo do tempo</b> "
                   "(folhas como bins, vs DES · requer coluna de data); abaixo a <b>concentração "
                   "das folhas entre amostras</b> (representatividade de cada folha em cada "
                   "amostra).</div>"),
            W.HBox([self.btn_estab, self.sm_estab_metrics, self.ck_estab_pct],
                   layout=W.Layout(align_items="center", flex_flow="row wrap")),
            # zoom do eixo Y dos gráficos de PSI (por safra e por amostra)
            W.HBox([self.btn_psi_zoom, self.btn_psi_reset, self.tx_psi_ymin, self.tx_psi_ymax],
                   layout=W.Layout(align_items="center", flex_flow="row wrap")),
            self.out_estab,
            self.out_conc,
        ], layout=W.Layout(width="100%"))
        card_estab.add_class("treeui-card")
        # ---- perfil das variáveis (que entraram na árvore) por safra ----
        card_varprof = W.VBox([
            W.HTML("<div class='treeui-h'>Perfil das variáveis por safra</div>"),
            W.HTML("<div class='treeui-legend'>Para cada variável que ENTROU na árvore: "
                   "<b>% de missing por safra</b> (eixo 0–100%) e a <b>dispersão p5·média·p95</b> "
                   "(categóricas como área empilhada, com legenda por gráfico). Faixas verticais "
                   "pontilhadas marcam a troca de amostra (DES→OOT). Requer coluna de data.</div>"),
            W.HBox([self.btn_varprofile]),
            self.out_varprof_missing,
            self.out_varprof_stats,
        ], layout=W.Layout(width="100%"))
        card_varprof.add_class("treeui-card")
        # Importância (vai p/ Diagnóstico) e export SQL (vai p/ Exportar):
        # definidos aqui porque as abas abaixo já os consomem.
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
                   "tabela de origem. O <b>fallback p/ não classificados</b> define o destino das "
                   "linhas que não caem em nenhuma folha ao aplicar a régua (categoria não vista, "
                   "faltante sem rota) — mesmo com 0 na base atual, isso acontece em OOT/produção; "
                   "com fallback, o <code>ELSE</code> vira a folha escolhida em vez de NULL.</div>"),
            W.HBox([self.tx_sql_table, self.dd_fallback, self.btn_sql]),
            self.out_sql]); card_sql.add_class("treeui-card")
        # importância abre a aba: é a leitura mais direta de "o que a árvore usou"
        # (card_sib saiu daqui para a aba "Árvore interativa" — ver lá o porquê)
        tab_diag = W.VBox([card_imp, sep_diag, card_score, sep_diag2,
                           card_metrics, card_table,
                           card_estab, card_varprof,
                           card_discrim, card_boot])

        # ================================================================
        # ABA 4. VALIDAR & EXPORTAR — duas faixas: validação · exportar/registrar
        # ================================================================
        sep_val = W.HTML("<div class='treeui-band'>Validação regulatória · "
                         "monotonicidade · calibração</div>")
        valid_legend = W.HTML(
            f"<div class='treeui-legend'>Mostra na tela duas checagens: <b>monotonicidade</b> do "
            f"{_rl} nas folhas ({_ref} e demais amostras) e <b>calibração</b> prevista ({_ref}) × "
            f"realizada (OOT) por folha. O <b>backtest</b> por safra ({_rl} previsto × realizado) "
            "entra no <b>relatório</b>, junto do resto — informe a coluna de tempo para incluí-lo. "
            "Na tela ele ficava ilegível com muitas safras.</div>")
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

        sep_exp = W.HTML("<div class='treeui-band'>Exportar &amp; registrar</div>")
        card_export_df = W.VBox([
            W.HTML("<div class='treeui-h'>Exportar DataFrame rotulado</div>"),
            W.HTML("<div class='treeui-legend'>Gera <b>ui.result</b> (pandas) com a coluna de "
                   "segmento e a folha por linha.</div>"),
            W.Box([], layout=W.Layout(flex="1 1 auto")),   # alinha o botão à base
            W.HBox([self.btn_export]),
        ], layout=W.Layout(width="49%"))
        card_export_df.add_class("treeui-card")
        # card novo (Excel multi-abas) empilhado ABAIXO do export original,
        # em largura cheia — o card antigo mantém o formato que tinha.
        card_xlsx = W.VBox([
            W.HTML("<div class='treeui-h'>Exportar Excel (.xlsx)</div>"),
            W.HTML("<div class='treeui-legend'>Arquivo multi-abas: <b>Folhas</b> (tabela "
                   "completa numérica), <b>Métricas por amostra</b>, <b>PSI</b> (resumo + "
                   "detalhe), <b>IV por variável</b>, <b>Calibração</b> e <b>Régua SQL</b>. "
                   "Cabeçalhos congelados e percentuais formatados. Requer o pacote opcional "
                   "<code>openpyxl</code>.</div>"),
            self.tx_xlsx_path,
            W.Box([], layout=W.Layout(flex="1 1 auto")),
            W.HBox([self.btn_xlsx]),
        ], layout=W.Layout(width="49%"))
        card_xlsx.add_class("treeui-card")
        # os dois exports lado a lado (mesma altura): são a mesma ação em formatos
        # diferentes, e empilhados ocupavam a tela toda antes do registro/MLflow.
        export_top = W.HBox([card_export_df, card_xlsx],
                            layout=W.Layout(width="100%", align_items="stretch",
                                            justify_content="space-between"))
        card_mlflow = W.VBox([
            W.HTML("<div class='treeui-h'>Registrar no MLflow / Unity Catalog</div>"),
            W.HTML("<div class='treeui-legend'>Loga régua, métricas e o modelo pyfunc e registra a "
                   "versão no Model Registry.</div>"),
            self.tx_model, self.cb_uc, self.cb_savebase, self.tx_experiment, self.tx_runname,
            W.Box([], layout=W.Layout(flex="1 1 auto")),   # espaçador: empurra o botão p/ a base
            W.HBox([self.btn_mlflow]),
        ], layout=W.Layout(width="49%"))
        card_mlflow.add_class("treeui-card")
        card_spark = W.VBox([
            W.HTML("<div class='treeui-h'>Reconstruir folhas em tabela Spark</div>"),
            W.HTML("<div class='treeui-legend'>Aplica a régua a uma tabela Spark (segmento, folha e "
                   "valor por linha), gravando opcionalmente o resultado. Sem o nome da "
                   "tabela, aplica na base em memória — <code>ui.score_df = df_novo</code> "
                   "ou a base carregada — e o resultado vai para <code>ui.result</code>.</div>"),
            self.tx_spark_in, self.tx_spark_out,
            W.Box([], layout=W.Layout(flex="1 1 auto")),   # alinha "Reconstruir folhas" com "Salvar no MLflow"
            W.HBox([self.btn_spark_apply]),
            self.out_spark_progress, self.out_spark,
        ], layout=W.Layout(width="49%"))
        card_spark.add_class("treeui-card")
        export_row = W.HBox([card_mlflow, card_spark],
                            layout=W.Layout(width="100%", align_items="stretch",
                                            justify_content="space-between"))
        # validação regulatória vai para a aba "Avançado" (abaixo) — a aba principal
        # de exportação não sobrecarrega a decisão do analista com as checagens.
        # A aba Exportar (tab_valid) é montada mais abaixo, já incorporando a seção
        # de Histórico & persistência (que deixou de ser uma aba própria).

        # ================================================================
        # SEÇÃO HISTÓRICO & PERSISTÊNCIA (agora no FIM da aba Exportar):
        #   JSON · imagem da árvore · confirmação de sobrescrita · relatório PDF
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
        # a antiga aba "Histórico" virou uma SEÇÃO no fim da aba Exportar
        tab_valid = W.VBox([sep_exp, export_top, card_sql, export_row,
                            sep_hist, hist_row, self.box_confirm, card_pdf])

        # ================================================================
        # ABA 2. ANÁLISE DE VARIÁVEL — perfil, distribuição e estabilidade
        # ================================================================
        self.dd_var.layout = W.Layout(width="100%", flex="1 1 auto")
        self.dd_var.style.description_width = "62px"
        box_var = W.HBox([self.dd_var, self.tg_var_iv],
                         layout=W.Layout(width="34%", align_items="center"))
        self.dd_var_leaf.layout = W.Layout(width="40%")
        self.dd_var_leaf.style.description_width = "46px"
        self.tx_var_time.layout = W.Layout(width="20%")
        self.btn_var_analyze.layout = W.Layout(width="auto")
        var_controls = W.VBox([
            W.HTML("<div class='treeui-h'>Análise de variáveis</div>"),
            W.HTML("<div class='treeui-legend'>Perfil de uma variável de entrada numa folha: "
                   "distribuição, %missing, média/mediana/desvio, faixa de percentis, PSI atual "
                   "e o comportamento por safra (percentis e PSI). Informe a <b>coluna de "
                   "safra</b> (ex.: dt_ref) para as análises temporais.</div>"),
            W.HBox([box_var, self.dd_var_leaf, self.tx_var_time, self.btn_var_analyze],
                   layout=W.Layout(align_items="flex-end", justify_content="space-between",
                                   width="100%")),
        ])
        var_controls.add_class("treeui-card")
        # ---- topo: comportamento (distribuição & risco) AO LADO do resumo &
        # estabilidade. Logodds/WoE e a tabela por faixa foram removidos a pedido
        # (aba mais enxuta — e menos plots por análise) ----
        card_var_dist = W.VBox([
            W.HTML("<div class='treeui-h'>Comportamento da variável · distribuição &amp; risco</div>"),
            self.out_var_dist], layout=W.Layout(width="49%"))   # 49/49 p/ alinhar com as demais linhas
        card_var_dist.add_class("treeui-card")
        card_var_cards = W.VBox([
            W.HTML("<div class='treeui-h'>Resumo &amp; estabilidade</div>"),
            self.out_var_cards], layout=W.Layout(width="49%"))
        card_var_cards.add_class("treeui-card")
        var_row_a = W.HBox([card_var_dist, card_var_cards],
                           layout=W.Layout(justify_content="space-between",
                                           align_items="stretch", width="100%"))
        card_inv_s = W.VBox([
            W.HTML("<div class='treeui-h'>Inversão da ordem de risco · por amostra</div>"),
            self.out_var_inv_s], layout=W.Layout(width="49%"))
        card_inv_s.add_class("treeui-card")
        card_inv_t = W.VBox([
            W.HTML("<div class='treeui-h'>Inversão da ordem de risco · por safra</div>"),
            self.out_var_inv_t], layout=W.Layout(width="49%"))
        card_inv_t.add_class("treeui-card")
        var_row_inv = W.HBox([card_inv_s, card_inv_t],
                             layout=W.Layout(justify_content="space-between",
                                             align_items="stretch", width="100%"))
        # colunas IGUAIS (49%/49%): com o mesmo figsize (handler), percentis e PSI
        # saem exatamente do mesmo tamanho lado a lado (par renderizado com tight=False)
        card_var_time = W.VBox([
            W.HTML("<div class='treeui-h'>Ao longo do tempo · percentis/share por safra</div>"),
            self.out_var_time], layout=W.Layout(width="49%"))
        card_var_time.add_class("treeui-card")
        card_var_psi = W.VBox([
            W.HTML(f"<div class='treeui-h'>PSI por safra · vs. referência ({self.ref_sample})</div>"),
            self.out_var_psi], layout=W.Layout(width="49%"))
        card_var_psi.add_class("treeui-card")
        var_row_time = W.HBox([card_var_time, card_var_psi],
                              layout=W.Layout(justify_content="space-between",
                                              align_items="stretch", width="100%"))
        card_var_optbin = W.VBox([
            W.HTML("<div class='treeui-h'>Distribuição acumulada das faixas do optimal "
                   "binning · por safra (numéricas)</div>"),
            self.out_var_optbin])
        card_var_optbin.add_class("treeui-card")
        tab_var = W.VBox([var_controls, var_row_a, var_row_inv,
                          var_row_time, card_var_optbin])

        # ---- ABA AVANÇADO: auto-merge · poda · diff de versões · cenários ----
        card_diff = W.VBox([
            W.HTML("<div class='treeui-h'>Comparar duas árvores (versões)</div>"),
            W.HTML("<div class='treeui-legend'>Carrega outra árvore salva em JSON e compara com a "
                   "atual: migração de folhas, concordância e métricas lado a lado.</div>"),
            W.HBox([self.tx_diff_path, self.btn_diff]), self.out_diff]); card_diff.add_class("treeui-card")
        self.btn_scn_save.layout.width = "auto"
        self.btn_scn_clear.layout.width = "auto"
        self.btn_scn_clear.disabled = True          # habilita quando houver cenário
        card_scn = W.VBox([
            W.HTML("<div class='treeui-h'>Cenários (em memória · só nesta sessão)</div>"),
            W.HTML("<div class='treeui-legend'>Guarde versões nomeadas da árvore e compare-as "
                   "sem sair da interface: <b>Salvar cenário</b> fotografa a estrutura atual "
                   "(condições, apelidos, fallback e folhas travadas); <b>Restaurar</b> volta a "
                   "árvore para a foto (desfazível com ↶ Desfazer); <b>Comparar com o atual</b> "
                   "mostra concordância, migração de folhas e Δ de métricas. Os cenários vivem "
                   "<b>apenas nesta sessão</b> (memória do kernel) — para persistir em disco use "
                   "Salvar (JSON) ou o MLflow, na aba Exportar.</div>"),
            W.HBox([self.tx_scn_name, self.btn_scn_save, self.btn_scn_clear],
                   layout=W.Layout(align_items="center")),
            self.out_scn_summary, self.box_scn_list, self.out_scn_diff])
        card_scn.add_class("treeui-card")
        tab_avancado = W.VBox([
            # O que sobrou aqui é o que NÃO é construção da árvore. Auto-merge e
            # poda foram para Construir, junto do auto-fit (as três ações
            # automáticas num card só); "Sugerir splits" saiu por duplicar a
            # sugestão da aba Construir; a importância foi para Diagnóstico e o
            # export SQL para Exportar — cada um perto do que serve.
            card_diff, card_scn,
            # validação regulatória (monotonicidade/calibração/backtest + relatório)
            # movida para cá: é uma etapa de fechamento, não da decisão de segmentação.
            sep_val, card_validacao])

        # ==============================================================
        # Aba "Árvore interativa": o canvas à esquerda e, à direita, o
        # painel que abre no nó clicado. Ao contrário da aba Construir
        # (formulário → folha), aqui a árvore é o ponto de partida.
        # ==============================================================
        self.box_cv_canvas = W.Box(layout=W.Layout(width="100%", height="640px",
                                                   min_width="0"))
        self.out_cv_msg = W.HTML()            # aviso quando o canvas não é possível
        self.btn_cv_fit = W.Button(description="Enquadrar", icon="compress",
                                   tooltip="Voltar a ver a árvore inteira",
                                   layout=W.Layout(width="auto"))
        self.btn_cv_fit.on_click(self._on_cv_fit)

        # -- painel: cabeçalho e métricas do nó
        self.out_cv_head = W.HTML()
        self.out_cv_stats = W.HTML()
        self.out_cv_note = W.HTML()           # ramo já dividido / folha fechada
        self.out_cv_empty = W.HTML(
            "<div class='treeui-legend' style='padding:16px 4px'>Clique num nó do canvas "
            "para abrir o painel. Numa <b>folha</b> você define o corte e as regras de "
            "negócio dela; num <b>ramo já dividido</b>, só as ações de estrutura.</div>")

        # -- painel: sugestões de variável (maior IV nesta folha)
        self.out_cv_sug_h = W.HTML("<div class='treeui-h' style='margin:8px 0 5px'>"
                                   "Sugestões · maior IV nesta folha</div>")
        self.btns_cv_sug = [W.Button(layout=W.Layout(width="100%")) for _ in range(3)]
        for i, b in enumerate(self.btns_cv_sug):
            b.add_class("treeui-sug")
            b.on_click(self._on_cv_sug(i))
        self._cv_sug: list = []

        # -- painel: configuração do corte
        self.dd_cv_feature = W.Dropdown(description="Variável", options=[],
                                        layout=W.Layout(width="100%"),
                                        style={"description_width": "62px"})
        self.dd_cv_feature.observe(self._on_cv_feature, names="value")
        self._cv_feat_by_label: dict = {}
        self.tg_cv_mode = W.ToggleButtons(
            options=["Ótimo", "Manual"], value="Ótimo",
            tooltips=["Deixa o binning ótimo achar os cortes",
                      "Você digita os cortes"],
            layout=W.Layout(width="auto"))
        self.tg_cv_mode.observe(self._on_cv_mode, names="value")
        self.sl_cv_bins = W.IntSlider(description="máx. faixas", value=4, min=2, max=6,
                                      continuous_update=False,
                                      layout=W.Layout(width="100%"),
                                      style={"description_width": "72px"})
        self.dd_cv_crit = W.Dropdown(
            description="critério", value="optbin",
            options=[("Binning ótimo (IV)", "optbin"), ("Gini (CART)", "gini"),
                     ("Entropia / ganho de informação", "entropy"), ("KS", "ks"),
                     ("Qui-quadrado (CHAID)", "chi2")],
            layout=W.Layout(width="100%"), style={"description_width": "72px"})
        self.tx_cv_cuts = W.Text(description="Cortes", placeholder="ex.: 420, 580, 720",
                                 layout=W.Layout(width="100%"),
                                 style={"description_width": "62px"})
        self.btn_cv_sugcuts = W.Button(description="Sugerir", icon="magic",
                                       tooltip="Preencher com o binning ótimo desta folha",
                                       layout=W.Layout(width="auto"))
        self.btn_cv_sugcuts.on_click(self._on_cv_sugcuts)
        self.out_cv_cuts_hint = W.HTML()
        self.out_cv_optbin_hint = W.HTML()    # limites de bin herdados de Construir
        self.box_cv_cuts = W.VBox([W.HBox([self.tx_cv_cuts, self.btn_cv_sugcuts],
                                          layout=W.Layout(align_items="center", width="100%")),
                                   self.out_cv_cuts_hint])
        # variável CATEGÓRICA no modo Manual: um seletor de grupo por categoria,
        # exatamente como na aba Construir — digitar grupos separados por ';' era
        # um segundo jeito de dizer a mesma coisa, e as duas telas divergiam
        self.cv_cat_box = W.VBox(layout=W.Layout(width="100%"))
        self.out_cv_preview = W.HTML()
        self.btn_cv_preview = W.Button(description="Prever divisão", icon="eye",
                                       layout=W.Layout(width="auto"))
        self.btn_cv_preview.on_click(self._on_cv_preview)
        self.btn_cv_apply = W.Button(description="Criar segmentos", icon="scissors",
                                     button_style="primary",
                                     layout=W.Layout(flex="1 1 auto"))
        self.btn_cv_apply.on_click(self._on_cv_apply)
        self.box_cv_split = W.VBox([
            self.out_cv_sug_h, *self.btns_cv_sug,
            W.HTML("<div class='treeui-h' style='margin:10px 0 5px'>Corte desta folha</div>"),
            self.dd_cv_feature,
            # o seletor de modo e os botões de ação ficam CENTRADOS na coluna
            W.HBox([self.tg_cv_mode],
                   layout=W.Layout(width="100%", justify_content="center")),
            self.sl_cv_bins, self.dd_cv_crit,
            self.out_cv_optbin_hint, self.box_cv_cuts, self.cv_cat_box,
            W.HBox([self.btn_cv_preview, self.btn_cv_apply],
                   layout=W.Layout(align_items="center", width="100%", flex_flow="row wrap",
                                   justify_content="center")),
            self.out_cv_preview,
        ], layout=W.Layout(width="100%"))

        # -- painel: regras de negócio (apelido + estrutura)
        self.tx_cv_name = W.Text(description="Apelido", placeholder="nome de negócio do segmento",
                                 continuous_update=False, layout=W.Layout(width="100%"),
                                 style={"description_width": "62px"})
        self.tx_cv_name.observe(self._on_cv_name, names="value")
        self.btn_cv_lock = W.Button(description="🔒 Fechar folha", layout=W.Layout(width="auto"))
        self.btn_cv_lock.on_click(self._on_cv_lock)
        self.btn_cv_merge_l = W.Button(description="◀ Fundir", layout=W.Layout(width="auto"),
                                       tooltip="Fundir com a folha vizinha da esquerda")
        self.btn_cv_merge_l.on_click(self._on_cv_merge("left"))
        self.btn_cv_merge_r = W.Button(description="Fundir ▶", layout=W.Layout(width="auto"),
                                       tooltip="Fundir com a folha vizinha da direita")
        self.btn_cv_merge_r.on_click(self._on_cv_merge("right"))
        self.btn_cv_collapse = W.Button(description="Recolher para o pai",
                                        layout=W.Layout(width="auto"),
                                        tooltip="Desfaz o corte deste ramo: os filhos somem "
                                                "e ele volta a ser folha")
        self.btn_cv_collapse.on_click(self._on_cv_collapse)
        self.btn_cv_missing = W.Button(description="Alocar faltantes aqui",
                                       layout=W.Layout(width="auto"),
                                       tooltip="Junta o nó de faltantes (NaN) deste split DENTRO "
                                               "desta folha — a regra vira 'faixa OU faltante'")
        self.btn_cv_missing.on_click(self._on_cv_missing)
        self.out_cv_merge_p = W.HTML()        # p-valor vs vizinhas (decisão de fusão)
        # -- mover corte: mesmo mecanismo da aba Construir (o segmentador move
        # sempre o `hi` de uma folha, então mover o corte à ESQUERDA desta é mover
        # o da DIREITA da irmã anterior — o seletor escolhe o lado e a UI acha o dono)
        self.dd_cv_move_side = W.Dropdown(description="lado", options=[("à direita ▶", "dir")],
                                          value="dir", layout=W.Layout(width="100%"),
                                          style={"description_width": "62px"})
        self.dd_cv_move_side.observe(self._on_cv_move_side, names="value")
        self.lbl_cv_move = W.HTML()
        self.tx_cv_move = W.FloatText(description="novo corte", layout=W.Layout(width="100%"),
                                      style={"description_width": "62px"})
        self.btn_cv_move_prev = W.Button(description="Prever corte", icon="eye",
                                         layout=W.Layout(width="auto"))
        self.btn_cv_move_prev.on_click(self._on_cv_move_preview)
        self.btn_cv_move = W.Button(description="Mover corte", icon="arrows-h",
                                    layout=W.Layout(width="auto"))
        self.btn_cv_move.on_click(self._on_cv_move)
        self.out_cv_move = W.HTML()
        self.box_cv_move = W.VBox([
            W.HTML("<div class='treeui-h' style='margin:10px 0 4px'>Mover o corte da divisa"
                   "</div>"),
            self.lbl_cv_move, self.dd_cv_move_side, self.tx_cv_move,
            W.HBox([self.btn_cv_move_prev, self.btn_cv_move],
                   layout=W.Layout(flex_flow="row wrap", align_items="center", width="100%")),
            self.out_cv_move,
        ], layout=W.Layout(width="100%"))
        box_cv_regras = W.VBox([
            W.HTML("<div class='treeui-h' style='margin:12px 0 5px'>Regras de negócio</div>"),
            W.HTML("<div class='treeui-legend' style='margin:0 0 6px'>O <b>apelido</b> é o nome "
                   "que este segmento leva para a régua, o Excel e o SQL. <b>Fechar</b> protege "
                   "a folha de novos cortes e da poda automática. <b>Alocar faltantes</b> traz "
                   "o nó de NaN do split para dentro desta folha.</div>"),
            self.tx_cv_name,
            W.HBox([self.btn_cv_lock, self.btn_cv_merge_l, self.btn_cv_merge_r,
                    self.btn_cv_collapse, self.btn_cv_missing],
                   layout=W.Layout(flex_flow="row wrap", align_items="center", width="100%")),
            self.out_cv_merge_p,
            self.box_cv_move,
        ], layout=W.Layout(width="100%"))

        self.box_cv_panel = W.VBox([self.out_cv_head, self.out_cv_stats, self.out_cv_note,
                                    self.box_cv_split, box_cv_regras],
                                   layout=W.Layout(width="100%"))
        painel_cv = W.VBox([self.out_cv_empty, self.box_cv_panel],
                           layout=W.Layout(width="410px", flex="0 0 410px", min_width="0",
                                           max_height="640px", overflow="hidden auto",
                                           padding="0 2px 0 10px"))
        painel_cv.add_class("treeui-cvpanel")
        self.btn_cv_reset = W.Button(description="Resetar árvore", icon="refresh",
                                     button_style="danger", layout=W.Layout(width="auto"),
                                     tooltip="Volta à árvore vazia (só a raiz). Pede confirmação "
                                             "e é desfazível com ↶ Desfazer")
        self.btn_cv_reset.on_click(self._on_cv_reset)
        # barra de ações da árvore INTEIRA. Desfazer/refazer moram aqui porque
        # manipulação direta sem desfazer à mão é desconfortável.
        self.btn_cv_undo = W.Button(description="↶ Desfazer", layout=W.Layout(width="auto"),
                                    tooltip="Desfaz a última alteração na árvore")
        self.btn_cv_undo.on_click(self._on_undo)
        self.btn_cv_redo = W.Button(description="Refazer ↷", layout=W.Layout(width="auto"),
                                    tooltip="Refaz a alteração desfeita")
        self.btn_cv_redo.on_click(self._on_redo)
        # As três ações automáticas NÃO executam direto: cada botão abre uma
        # janelinha no meio do canvas com a configuração daquela ação, e nada
        # muda até o Aplicar. Os controles dentro dela são as MESMAS instâncias
        # da aba Construir (segunda view do mesmo modelo) — mexer num lado
        # mexe no outro, um único lugar de verdade.
        self.btn_cv_autofit = W.Button(description="Auto-fit", icon="magic",
                                       layout=W.Layout(width="auto"),
                                       tooltip="Cresce a árvore gulosa por IV. Com uma FOLHA "
                                               "selecionada no mapa, cresce só aquele ramo; "
                                               "na raiz, reconstrói tudo")
        self.btn_cv_autofit.on_click(self._on_cv_modal_open("fit"))
        self.btn_cv_automerge = W.Button(description="Auto-fundir", icon="compress",
                                         layout=W.Layout(width="auto"),
                                         tooltip="Funde folhas-irmãs com risco estatisticamente "
                                                 "indistinguível (p > α) ou com Δ abaixo do mínimo")
        self.btn_cv_automerge.on_click(self._on_cv_modal_open("merge"))
        self.btn_cv_prune = W.Button(description="Podar", icon="scissors",
                                     layout=W.Layout(width="auto"),
                                     tooltip="Funde as folhas pouco representativas ou com Δ "
                                             "pequeno em relação à irmã")
        self.btn_cv_prune.on_click(self._on_cv_modal_open("prune"))
        # -- a janelinha em si: título + nota de alvo + corpo (controles) + ações
        self.out_cv_modal_head = W.HTML()
        self.box_cv_modal_body = W.VBox(layout=W.Layout(width="100%"))
        self.btn_cv_modal_ok = W.Button(description="Aplicar", icon="check",
                                        button_style="primary",
                                        layout=W.Layout(width="auto"))
        self.btn_cv_modal_ok.on_click(self._on_cv_modal_apply)
        self.btn_cv_modal_cancel = W.Button(description="Cancelar",
                                            layout=W.Layout(width="auto"))
        self.btn_cv_modal_cancel.on_click(self._on_cv_modal_cancel)
        self.box_cv_modal = W.VBox([
            self.out_cv_modal_head, self.box_cv_modal_body,
            W.HBox([self.btn_cv_modal_ok, self.btn_cv_modal_cancel],
                   layout=W.Layout(justify_content="center", width="100%",
                                   align_items="center")),
        ], layout=W.Layout(width="360px", display="none"))
        self.box_cv_modal.add_class("treeui-cv-modal")
        self._cv_modal_kind = None
        # "ir para folha": numa árvore larga, achar uma folha arrastando é chato.
        # O dropdown é uma AÇÃO (voa até a folha e volta ao rótulo), não um estado.
        # o rótulo de ação usa "" como valor (NUNCA None: para o Selection do
        # ipywidgets, value=None significa "sem seleção" e zera o index — o
        # dropdown renderiza em branco). O observer fica no INDEX, o trait
        # primitivo que o frontend envia.
        self.dd_cv_goto = W.Dropdown(options=[("ir para folha…", "")], index=0,
                                     layout=W.Layout(width="200px"))
        self.dd_cv_goto.observe(self._on_cv_goto, names="index")
        self._cv_goto_syncing = False
        # "salvar cenário": o fluxo do mapa é experimental por natureza, e
        # experimento pede foto antes. Reusa a lista de cenários de Avançado
        # (restaurar/comparar continuam lá).
        self.tx_cv_scn = W.Text(placeholder="nome do cenário",
                                layout=W.Layout(width="150px"))
        self.btn_cv_scn = W.Button(description="Salvar cenário", icon="camera",
                                   tooltip="Fotografa a árvore atual como cenário em memória — "
                                           "restaurar/comparar ficam na aba Avançado",
                                   layout=W.Layout(width="auto"))
        self.btn_cv_scn.on_click(self._on_cv_scn_save)
        # palco: canvas + a janelinha de confirmação por cima (par relative/
        # absolute via as classes treeui-cv-stage/-modal — o Layout do
        # ipywidgets não expõe `position`)
        _palco_cv = W.VBox([self.box_cv_canvas, self.out_cv_msg, self.box_cv_modal],
                           layout=W.Layout(flex="1 1 auto", min_width="0"))
        _palco_cv.add_class("treeui-cv-stage")
        card_cv = W.VBox([
            W.HTML("<div class='treeui-h'>Árvore interativa · construa os cortes no mapa</div>"),
            W.HTML("<div class='treeui-legend'>Arraste para navegar, role para ampliar e "
                   "<b>clique num nó</b> para abrir o painel ao lado. Cada cartão traz a "
                   "condição do nó, o " + _esc(self._risk_mean) + ", a volumetria e uma barra "
                   "com a representatividade. A folha clicada aqui vira a folha ativa das "
                   "outras abas.<br/>A tela abre centralizada na folha em foco; "
                   "<b>Enquadrar</b> mostra a árvore inteira — em árvores largas os cartões "
                   "ficam pequenos, então aproxime de volta para ler.</div>"),
            W.HBox([self.btn_cv_undo, self.btn_cv_redo,
                    W.HTML("<div class='treeui-vsep'></div>"),
                    self.btn_cv_autofit, self.btn_cv_automerge, self.btn_cv_prune,
                    W.HTML("<div class='treeui-vsep'></div>"),
                    self.dd_cv_goto, self.btn_cv_fit,
                    W.HTML("<div class='treeui-vsep'></div>"),
                    self.tx_cv_scn, self.btn_cv_scn,
                    W.HTML("<div class='treeui-vsep'></div>"),
                    self.btn_cv_reset],
                   layout=W.Layout(width="100%", align_items="center",
                                   flex_flow="row wrap")),
            W.HBox([_palco_cv, painel_cv],
                   layout=W.Layout(width="100%", align_items="stretch")),
        ])
        card_cv.add_class("treeui-card")
        card_cv.add_class("treeui-card-mapa")
        # o teste entre folhas comparáveis vem junto: ele compara a folha com as
        # IRMÃS ADJACENTES, então pertence ao lugar onde se olha a vizinhança da
        # folha — o mapa — e não a uma aba de diagnóstico separada.
        tab_canvas = W.VBox([card_cv, card_sib])

        # ---- montagem das abas (o canvas entra logo depois de Construir) ----
        tabs = W.Tab(children=[tab_build, tab_canvas, tab_var, tab_diag, tab_valid,
                               tab_avancado])
        for i, titulo in enumerate(["Construir", "Árvore interativa", "Análise de variáveis",
                                    "Diagnóstico", "Exportar", "Avançado"]):
            tabs.set_title(i, titulo)
        tabs.add_class("treeui-tabs")
        # a tabela de IV (optbinning de TODAS as variáveis na folha) é o item mais
        # caro do open/refresh e fica numa aba não-visível por padrão. Adiamos seu
        # cálculo até a aba ser realmente aberta (render preguiçoso) — ver _refresh_iv.
        self.tabs = tabs
        self._build_tab_index = 0
        self._canvas_tab_index = 1
        self._iv_tab_index = 2
        self._diag_tab_index = 3
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
        # mantém o cluster Databricks ativo enquanto a interface está aberta (senão ele
        # desliga por inatividade); no-op fora do Databricks/Spark — ver utils.keepalive.
        self._keepalive = None
        self.cb_keepalive = W.ToggleButton(
            value=False, description="☕ Manter cluster ativo",
            tooltip="Databricks: dispara um job Spark mínimo a cada 2 min para o cluster "
                    "não desligar por inatividade enquanto a interface está aberta",
            layout=W.Layout(width="190px"))
        self.cb_keepalive.observe(self._on_keepalive, names="value")
        topbar = W.HBox([self.cb_keepalive, self.cb_dark],
                        layout=W.Layout(justify_content="flex-end"))
        self.panel = W.VBox([topbar, banner, bar_box, tabs, console, self.tree_sel_style])
        self.panel.add_class("treeui")

        # A interface ABRE na Árvore interativa, já com a importância (IV) de
        # cada variável calculada: trocar o selected_index dispara o
        # _on_tab_change, que desenha o canvas e o painel da raiz — sugestões
        # por IV e o seletor de variável ordenado, sem nenhum clique. É custo
        # pago 1× na abertura (o IV da raiz é memoizado para o resto da sessão).
        tabs.selected_index = self._canvas_tab_index

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
                self.cb_keepalive.value = False          # reverte o toggle
                self._suspend_ka = False
                self.cb_keepalive.description = "☕ Manter cluster ativo"
                self._log("[keepalive] nenhuma SparkSession ativa — este recurso só tem "
                          "efeito no Databricks (ou com Spark local).")
                return
            self._keepalive.start()
            self.cb_keepalive.description = "☕ Cluster ativo ✓"
            self._log("[keepalive] ligado — um job Spark mínimo a cada 2 min mantém o "
                      "cluster ativo enquanto a interface estiver aberta. Desligue ao "
                      "terminar para o cluster poder hibernar normalmente.")
        else:
            if getattr(self, "_suspend_ka", False):
                return
            if self._keepalive is not None:
                self._keepalive.stop()
            self.cb_keepalive.description = "☕ Manter cluster ativo"
            self._log("[keepalive] desligado.")

    # ==================================================================
    # Helpers de UX — console com histórico, estado "ocupado" e confirmação
    # em dois cliques (mesma mecânica do ModelSegmenterUI)
    # ==================================================================
    def _log(self, msg):
        # mantém só as últimas 40 linhas: reescreve a área (clear_output) em vez
        # de apagar o histórico a cada ação — as mensagens das ações anteriores
        # continuam visíveis no console, sem acumular indefinidamente o estado
        # do W.Output (que trafega pelo comm).
        self._log_lines.append(str(msg))
        if len(self._log_lines) > 40:
            self._log_lines = self._log_lines[-40:]
        with self.out_log:
            self.out_log.clear_output(wait=True)
            print("\n".join(self._log_lines))

    @contextmanager
    def _busy(self, *botoes, status=None, msg="processando…"):
        """Desabilita ``botoes`` enquanto uma ação síncrona roda e mostra um
        aviso "ocupado" em ``status`` (widget HTML). Ao sair, re-habilita os
        botões SEMPRE (mesmo com exceção) e limpa o status apenas se o handler
        não o substituiu por um resultado/erro próprio."""
        busy_html = f"<div class='treeui-legend'><i>⏳ {msg}</i></div>"
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

    def _delta_snapshot(self):
        """Métricas compactas do estado ATUAL para a linha de Δ do console:
        nº de folhas, métrica principal (KS na classificação · R² na regressão)
        na amostra de comparação (1ª não-referência, ex.: OOT) e PSI máximo.

        Deve ser capturada ANTES da mutação, no mesmo ponto do
        :meth:`_checkpoint` — como ``metrics()``/``psi()`` são memoizados por
        versão da árvore (``_agg_memo``), ler o "antes" aqui tem custo ~zero
        (cache-hit da última renderização)."""
        snap = {"folhas": sum(s["is_leaf"] for s in self.seg.segments.values()),
                "metrica": None, "psi_max": None}
        col = "KS" if self._is_clf else "R2"
        rotulo = "KS" if self._is_clf else "R²"
        try:
            m = self.seg.metrics()
            # amostra de comparação: 1ª não-referência (ex.: OOT); sem amostras → única
            row = m.iloc[1] if len(m) > 1 else m.iloc[0]
            if pd.notna(row.get(col)):
                snap["metrica"] = (f"{rotulo} {row['amostra']}", float(row[col]))
        except Exception:
            pass                            # sem métrica → a linha sai só com folhas
        try:
            if self.sample_col is not None:
                psi = self.seg.psi()["psi"]
                if len(psi):
                    snap["psi_max"] = float(psi.max())
        except Exception:
            pass
        return snap

    def _log_delta(self, acao, antes):
        """Linha compacta no console com o Δ da ação estrutural vs o estado
        anterior — ex.: ``dividir: folhas 8→10 · KS OOT 0.412→0.428 (+0.016) ·
        PSI máx 0.06→0.07``. ``antes`` vem de :meth:`_delta_snapshot`."""
        if not antes:
            return
        depois = self._delta_snapshot()
        partes = [f"folhas {antes['folhas']}→{depois['folhas']}"]
        m0, m1 = antes.get("metrica"), depois.get("metrica")
        if m0 and m1 and m0[0] == m1[0]:    # mesma métrica/amostra nos dois estados
            partes.append(f"{m1[0]} {m0[1]:.3f}→{m1[1]:.3f} ({m1[1] - m0[1]:+.3f})")
        p0, p1 = antes.get("psi_max"), depois.get("psi_max")
        if p0 is not None and p1 is not None:
            partes.append(f"PSI máx {p0:.2f}→{p1:.2f}")
        self._log(f"{acao}: " + " · ".join(partes))

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
        t = threading.Timer(timeout, _revert)
        t.daemon = True         # não segura o encerramento do processo/kernel
        t.start()

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
        """Teste de hipótese comparando o alvo (taxa de default) da MESMA folha entre
        as amostras `a` (ex.: DES) e `b` (ex.: OOT) — aderência do alvo entre amostras.
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
        """Teste de alvo (na amostra DES) entre a folha e cada IRMÃ TERMINAL
        ADJACENTE de MESMO PAI — indica se a folha é estatisticamente distinta da
        vizinha. A adjacência respeita as *runs* de folhas terminais: uma irmã
        que se expandiu (nó intermediário) QUEBRA a adjacência, então folhas de
        lados opostos dela não são comparadas. O nó de faltantes (na) não entra.
        Retorna (nome_do_teste, [(lado, descrição_irmã, p_valor)])."""
        seg = self.seg
        name = "Welch t" if self.dd_test.value == "welch" else "Mann-Whitney"
        s = seg.segments.get(sid)
        if s is None or s["parent"] is None or not s["is_leaf"]:
            return name, []
        left, right = seg._adjacent_sibling_neighbors(sid)
        out = []
        for lado, nb in (("◀", left), ("▶", right)):
            if nb is not None:
                p = seg._pair_pvalue(sid, nb, test=self.dd_test.value)
                desc = seg._descrever([seg.segments[nb]["conditions"][-1]])
                out.append((lado, desc, p))
        return name, out

    def _leaf_chips_html(self):
        """Resumo curto da folha ativa para a régua do topo (nº da folha, rótulo,
        alvo DES, volumetria e repr.) — o detalhe completo fica na faixa de baixo."""
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
        # apelido de negócio na frente (a descrição segue como complemento)
        nome = self.seg.leaf_name(sid)
        if nome:
            label = f"{_esc(nome)} · {label}"
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
        hexc = {"green": "var(--ok-ink)", "yellow": "var(--warn-ink)", "red": "var(--bad-ink)"}
        bgc = {"green": "var(--ok-bg)", "yellow": "var(--warn-bg)", "red": "var(--bad-bg)"}

        def cell(label, value, color="var(--ink)", badge=None, cls=None, tip=""):
            bh = ""
            if badge and cls:
                bh = (f"<span style='font-size:10px;font-weight:600;color:{hexc[cls]};"
                      f"background:{bgc[cls]};border-radius:20px;padding:2px 8px;"
                      f"margin-left:7px'>{badge}</span>")
            t = f" title='{tip}'" if tip else ""
            return (f"<div{t} style='flex:1;min-width:86px;padding:8px 14px;"
                    f"border-right:1px solid var(--hair)'>"
                    f"<div style='font-size:10px;font-weight:600;letter-spacing:.07em;"
                    f"text-transform:uppercase;color:var(--sub-ink);white-space:nowrap'>{label}</div>"
                    f"<div style='display:flex;align-items:center;margin-top:2px'>"
                    f"<span class='mono' style='font-size:19px;font-weight:600;color:{color}'>"
                    f"{value}</span>{bh}</div></div>")
        # A barra guarda o tamanho da árvore, a estabilidade e a discriminação.
        # Linhas sem rota (órfãs) saíram daqui: são 0 no desenvolvimento e vivem no
        # card Exportar como SQL, junto do fallback que decide o destino delas em
        # OOT/produção.
        _tip_lock = ("Folhas travadas como finais (🔒): não são divididas e ficam "
                     "protegidas da poda e do auto-merge. Trave em Ações da folha.")
        cells = [cell("Folhas", n_folhas), cell("Profundidade", prof),
                 cell("Fechadas", n_lock, tip=_tip_lock)]
        if self.sample_col is not None and n_folhas >= 1:
            try:
                for _, r in seg.psi().iterrows():
                    c = self._psi_class(r["psi"])
                    cells.append(cell(f"PSI {r['amostra']}", f"{r['psi']:.1%}",
                                      badge=r["classificacao"], cls=c))
            except Exception:
                pass
        # discriminação ao vivo: KS/AUC (classificação) ou R² (regressão)
        try:
            for _, r in seg.metrics().iterrows():
                if self._is_clf:
                    ks, auc = r["KS"], r["AUC"]
                    if pd.isna(ks):
                        cells.append(cell(f"KS {r['amostra']}", "—", color="var(--sub-ink)"))
                    else:
                        c = "green" if ks >= 0.30 else "yellow" if ks >= 0.20 else "red"
                        badge = "bom" if c == "green" else "atenção" if c == "yellow" else "fraco"
                        cells.append(cell(f"KS {r['amostra']}", f"{ks:.1%}", color=hexc[c],
                                          badge=badge, cls=c))
                    if pd.isna(auc):
                        cells.append(cell(f"AUC {r['amostra']}", "—", color="var(--sub-ink)"))
                    else:
                        c = "green" if auc >= 0.70 else "yellow" if auc >= 0.60 else "red"
                        cells.append(cell(f"AUC {r['amostra']}", f"{auc:.1%}", color=hexc[c]))
                else:
                    r2 = r["R2"]
                    if pd.isna(r2):
                        cells.append(cell(f"R² {r['amostra']}", "—", color="var(--sub-ink)"))
                    else:
                        c = "green" if r2 >= 0.5 else "yellow" if r2 >= 0.2 else "red"
                        cells.append(cell(f"R² {r['amostra']}", f"{r2:.1%}", color=hexc[c]))
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

        # tokens SEMÂNTICOS de PSI (semáforo), iguais aos de _leaf_header_html/
        # _var_cards_html/_diag_scorecard_html — antes usava a paleta de gradiente de
        # risco (--risk-*), fazendo o verde/amarelo/vermelho do PSI na árvore não bater
        # com o dos cartões.
        psi_hex = {"green": "var(--ok-tx)", "yellow": "var(--warn-tx)", "red": "var(--bad-tx)"}
        # "barrinha" vertical que separa o bloco alvo do bloco PSI na linha da folha
        sep_bar = ("<span style='display:inline-block;width:0;border-left:1px solid "
                   "var(--faint-ink);height:11px;margin:0 8px;vertical-align:middle'></span>")

        def psi_str(sid):
            if self.sample_col is None or not self._nonref:
                return ""
            parts = []
            for a in self._nonref:
                p = self._leaf_psi(sid, a)
                if pd.isna(p):
                    continue
                ab = "ESTAB" if a == "ESTABILIDADE" else a
                # em % como na barra de métricas e na tabela de folhas — o PSI
                # aparecia aqui em decimal, único ponto da tela fora do padrão.
                parts.append(f"<span style='color:{psi_hex[self._psi_class(p)]}'>"
                             f"{ab} {p:.1%}</span>")
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
                # apelido de negócio da folha (quando definido) ao lado do número
                nome = seg.leaf_name(sid)
                if nome:
                    tags += (" · <span style='color:var(--strong-ink);"
                             "font-style:italic' title='apelido do segmento'>"
                             + _esc(nome) + "</span>")
                if sid in self.locked:
                    tags += " 🔒"
                sel_marker = "<i class='tsel'></i>"   # ::after injeta '◀ selecionada'
            # continuação do prefixo (mantém os traços verticais alinhados na 2ª linha)
            cont = "" if is_root else prefix + ("   " if is_last else "│  ")
            psi_html = psi_str(sid) if s["is_leaf"] else ""
            # linha 1 — rótulo (condição do nó) + nº da folha
            linha1 = (f"<div style='{mono};font-size:12px;padding:1px 2px 0'>"
                      f"{prefix}{conn}{sw}<b class='tlname' style='color:var(--strong-ink)'>"
                      f"{rotulo(sid)}</b>{tags}{sel_marker}</div>")
            # linha 2 — métricas EMBAIXO: volumetria, representatividade, alvo e PSI
            vol = f"{n:,}".replace(",", ".")        # separador de milhar pt-BR
            linha2 = (f"<div style='{mono};font-size:11px;color:var(--tree-meta);padding:0 2px 3px'>"
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
            f'.tnode[data-leaf="{n}"]{{background:var(--sel-bg);border-radius:5px;'
            "box-shadow:inset 3px 0 0 var(--sel-ac);}"
            f'.tnode[data-leaf="{n}"] .tlname{{color:var(--sel-ac) !important;}}'
            f'.tnode[data-leaf="{n}"] .tsel::after{{content:" ◀ selecionada";'
            "color:var(--sel-ac);font-weight:700;}"
            "</style>")

    def _style_leaves(self, lv):
        psi_cols = [c for c in lv.columns if c.startswith("psi_")]

        def psi_bg(v):
            if pd.isna(v):
                return ""
            a = abs(v)
            c = ("var(--ok-bg)" if a < 0.10
                 else "var(--warn-bg)" if a < 0.25 else "var(--bad-bg)")
            return f"background-color:{c}"

        def p_bg(v):
            if pd.isna(v):
                return "color:var(--faint-ink)"
            return "background-color:var(--bad-bg);font-weight:600" if v > 0.05 else "color:var(--ok-tx)"

        def p_stab_bg(v):
            # aderência DES×OOT (H₀: mesma estimativa): semântica INVERSA à das irmãs —
            # p alto = estável (verde); p baixo = a estimativa deslocou (alerta).
            if pd.isna(v):
                return "color:var(--faint-ink)"
            return "color:var(--ok-tx)" if v > 0.05 else "background-color:var(--bad-bg);font-weight:600"

        sty = lv.style
        for c in psi_cols:
            sty = sty.map(psi_bg, subset=[c])
        if "p_vs_prox" in lv.columns:
            sty = sty.map(p_bg, subset=["p_vs_prox"])
        if "p_des_oot" in lv.columns:
            sty = sty.map(p_stab_bg, subset=["p_des_oot"])
        # "saldo_%" (visão dupla contratos × saldo) usa a mesma casa decimal do
        # repr_% — a chave é ignorada quando a coluna não está na tabela
        fmt = {"repr_%": "{:.1f}", "saldo_%": "{:.1f}"}
        if "apelido" in lv.columns:          # texto LIVRE do usuário → escapa p/ HTML
            fmt["apelido"] = _esc
        for c in lv.columns:
            if c.startswith("repr_") and c.endswith("_%"):   # % por amostra
                fmt[c] = "{:.1f}"
            elif c.startswith("pd_"):       # alvo em % (coerente com a árvore)
                fmt[c] = "{:.2%}"
            elif c.startswith("psi_"):      # em % como na barra, na árvore e no painel
                fmt[c] = "{:.1%}"
        if "p_vs_prox" in lv.columns:
            fmt["p_vs_prox"] = "{:.3f}"
        if "p_des_oot" in lv.columns:
            fmt["p_des_oot"] = "{:.3f}"
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
        # apelido de negócio na frente; a descrição mecânica vira complemento
        nome = self.seg.leaf_name(sid)
        if nome:
            txt = f"{nome} · {txt}"
        if len(txt) > 72:
            txt = txt[:69] + "…"
        return ("🔒 " if sid in self.locked else "") + txt

    def _leaf_header_html(self):
        """Cartão-resumo da folha selecionada: volumetria e representatividade por
        amostra (DES, OOT, ESTAB…); alvo médio de DES e das demais amostras com o
        incremento de cada uma vs DES; teste de aderência DES→amostra (nome +
        p-valor); distinção vs folha-irmã; e estabilidade (PSI por amostra) com
        barrinha verde/amarelo/vermelho."""
        sid = self.dd_leaf.value
        if sid is None or sid not in self.seg.segments:
            return ("<div style='font-size:12px;color:var(--sub-ink)'>Nenhuma folha selecionada — "
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
        na_badge = ("<span class='pill' style='background:var(--bad-bg);color:var(--bad-ink)' "
                    "title='Recebe os faltantes (NaN) no scoring — atribuição "
                    "conservadora à folha-irmã de pior risco'>+ faltantes</span>"
                    if any(c.get("include_na") for c in s["conditions"]) else "")

        def ab(a):
            return "ESTAB" if a == "ESTABILIDADE" else a

        def chip(k, v, c=None, sub=None):
            sty = f" style='color:{c}'" if c else ""
            sub_html = (f"<div style='font-size:10.5px;margin-top:1px;white-space:nowrap;"
                        f"color:var(--sub-ink)'>{sub}</div>") if sub else ""
            return (f"<div class='treeui-metric'><div class='k'>{k}</div>"
                    f"<div class='v mono'{sty}>{v}</div>{sub_html}</div>")

        # com APELIDO de negócio, ele assume o título e a descrição mecânica vira
        # tooltip + linha-complemento menor (sem apelido, nada muda)
        nome = self.seg.leaf_name(sid)
        titulo = _esc(nome) if nome else label
        tip = f" title='{_esc(label)}'" if nome else ""
        sub_desc = (f"<div style='font-size:11.5px;color:var(--sub-ink);"
                    f"margin:-2px 0 5px 22px'>{label}</div>" if nome else "")
        head = (
            # linha 1: o apelido (quando definido) ou o CORTE (condição da folha)
            "<div style='display:flex;align-items:center;gap:9px;margin-bottom:5px'>"
            f"<span style='width:13px;height:13px;border-radius:4px;background:{color};"
            "flex:none'></span>"
            f"<span style='font-size:15px;font-weight:600;color:var(--strong-ink)'{tip}>"
            f"{titulo}</span>"
            "</div>" + sub_desc +
            # linha 2: SEMPRE abaixo do corte — status (aberta/fechada · faltantes) e
            # qual folha está sendo editada ("folha N")
            "<div style='display:flex;align-items:center;gap:9px;margin-bottom:4px;"
            "flex-wrap:wrap'>"
            f"{badge}{na_badge}<span class='pill pill-muted'>folha {nota}</span></div>")

        sec_h = ("<div class='treeui-h' style='margin-top:11px'>{}</div>").format

        # VISÃO DUPLA contratos × saldo: com weight_col configurada, o cartão ganha
        # o % do SALDO ao lado do % de contratos e o alvo médio PONDERADO (como
        # sublinha dos chips de alvo). Sem weight_col, nada abaixo é acionado.
        wcol = self.seg.weight_col

        def pond_txt(sample=None):
            """'pond. X%' com o alvo ponderado pelo saldo ('' sem weight_col)."""
            if wcol is None:
                return ""
            v = self.seg.weighted_value(sid, sample)
            return "pond. " + ("—" if pd.isna(v) else f"{v * 100:.2f}%")

        if self.sample_col is None:
            rep = 100 * n / len(self.df) if len(self.df) else 0.0
            cells = (chip("Volumetria", f"{n:,}".replace(",", "."))
                     + chip("Repr.", f"{rep:.1f}%"))
            if wcol is not None:
                ws = self.seg.weight_share(sid)
                cells += chip("Repr. saldo", "—" if pd.isna(ws) else f"{ws:.1f}%",
                              sub=f"% de {_esc(str(wcol))}")
            cells += chip(self._risk_label, f"{self._node_value(sid) * 100:.2f}%",
                          sub=pond_txt() or None)
            return head + f"<div class='treeui-metrics'>{cells}</div>"

        sm = self._sample_masks
        ordered_nonref = list(self._pd_nonref) + list(self._psi_only)
        samples_all = [self.ref_sample] + ordered_nonref

        # 1) Volumetria & representatividade da folha por amostra
        sec1 = chip("Volumetria", f"{n:,}".replace(",", "."))
        if wcol is not None:      # % do SALDO da carteira (não por amostra)
            ws = self.seg.weight_share(sid)
            sec1 += chip("Repr. saldo", "—" if pd.isna(ws) else f"{ws:.1f}%",
                         sub=f"% de {_esc(str(wcol))} na carteira")
        for a in samples_all:
            m = sm.get(a)
            tot = int(m.sum()) if m is not None else 0
            rp = (100 * int((leaf & m).sum()) / tot) if tot else float("nan")
            sec1 += chip(f"Repr. {ab(a)}", "—" if pd.isna(rp) else f"{rp:.1f}%")

        # 2) alvo médio (DES e demais) + incremento de cada amostra vs DES
        #    (com weight_col, a sublinha traz também o alvo PONDERADO pelo saldo)
        pd_ref = self._node_value(sid, self.ref_sample)
        sub_ref = "referência"
        if wcol is not None:
            sub_ref += " · " + pond_txt(self.ref_sample)
        sec2 = chip(f"{self._risk_label} {self.ref_sample}",
                    "—" if pd.isna(pd_ref) else f"{pd_ref * 100:.2f}%", sub=sub_ref)
        for a in self._pd_nonref:
            v = self._node_value(sid, a)
            pond = pond_txt(a)
            if pd.isna(v) or pd.isna(pd_ref):
                sec2 += chip(f"{self._risk_label} {ab(a)}",
                             "—" if pd.isna(v) else f"{v * 100:.2f}%",
                             sub=pond or None)
                continue
            d = (v - pd_ref) * 100      # incremento em pontos percentuais
            sig = "+" if d >= 0 else "−"
            dcol = "var(--bad-tx)" if d > 0 else "var(--ok-tx)"   # alvo subindo = pior (vermelho)
            sub = f"<span style='color:{dcol}'>Δ vs DES {sig}{abs(d):.2f} p.p.</span>"
            if pond:
                sub += f" · {pond}"
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
            # duas linhas: identificação + teste em cima, p-valor + veredito embaixo.
            # Numa linha só, o p-valor ficava espremido entre o nome do teste e a
            # pílula, e a quebra caía em lugar diferente conforme o tamanho do texto.
            test_rows += (
                "<div style='font-size:12px;color:var(--body-ink);margin:7px 0'>"
                "<div style='display:flex;align-items:center;gap:7px;flex-wrap:wrap'>"
                f"<b>DES → {ab(a)}</b>"
                f"<span style='color:var(--muted)'>teste:</span><b>{name}</b></div>"
                "<div style='display:flex;align-items:center;gap:7px;flex-wrap:wrap;"
                "margin-top:4px'>"
                f"<span style='color:var(--muted)'>p-valor:</span>"
                f"<b class='mono'>{pv}</b>{verdict}</div></div>")

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
                "<div style='font-size:12px;color:var(--body-ink);margin:7px 0'>"
                "<div style='display:flex;align-items:center;gap:7px;flex-wrap:wrap'>"
                f"<b>{lado} {d}</b>"
                f"<span style='color:var(--muted)'>teste:</span><b>{sib_name}</b></div>"
                "<div style='display:flex;align-items:center;gap:7px;flex-wrap:wrap;"
                "margin-top:4px'>"
                f"<span style='color:var(--muted)'>p-valor:</span>"
                f"<b class='mono'>{pv}</b>{verdict}</div></div>")

        # 5) Estabilidade · PSI por amostra com barrinha verde/amarelo/vermelho
        psi_hex = {"green": "var(--ok-tx)", "yellow": "var(--warn-tx)", "red": "var(--bad-tx)"}

        def gauge(p):
            if pd.isna(p):
                return ("<div style='flex:1;height:9px;border-radius:5px;"
                        "background:var(--gauge-track)'></div>")
            pos = min(max(p, 0.0) / 0.50, 1.0) * 100
            return (
                "<div style='position:relative;flex:1;height:9px;border-radius:5px;"
                "background:linear-gradient(to right,var(--gauge-ok) 0%,var(--gauge-ok) 20%,"
                "var(--gauge-warn) 20%,var(--gauge-warn) 50%,"
                "var(--gauge-bad) 50%,var(--gauge-bad) 100%)'>"
                f"<div style='position:absolute;left:calc({pos:.1f}% - 1px);top:-2px;"
                "width:2px;height:13px;background:var(--strong-ink);border-radius:1px'></div></div>")

        psi_rows = ""
        for a in ordered_nonref:      # DES é a referência (PSI ≡ 0), por isso fica de fora
            p = self._leaf_psi(sid, a)
            if pd.isna(p):
                pv, pcol = "—", "var(--sub-ink)"
            else:
                pv, pcol = f"{p:.1%}", psi_hex[self._psi_class(p)]
            psi_rows += (
                "<div style='display:flex;align-items:center;gap:9px;margin:5px 0'>"
                f"<div style='width:78px;font-size:11px;color:var(--muted);white-space:nowrap'>"
                f"PSI {ab(a)}</div>"
                f"<div class='mono' style='width:48px;font-size:12.5px;font-weight:600;"
                f"color:{pcol}'>{pv}</div>{gauge(p)}</div>")
        psi_legend = (
            "<div style='font-size:10px;color:var(--sub-ink);margin-top:5px'>"
            "<span style='color:var(--gauge-ok)'>■</span> &lt;0,10 estável &nbsp; "
            "<span style='color:var(--gauge-warn)'>■</span> 0,10–0,25 atenção &nbsp; "
            "<span style='color:var(--gauge-bad)'>■</span> &gt;0,25 crítico</div>")

        # com weight_col, os títulos avisam que a leitura é dupla (contratos × saldo)
        _h1 = "Volumetria &amp; representatividade"
        _h2 = f"{self._risk_mean} &amp; incremento vs DES"
        if wcol is not None:
            _h1 += " (contratos &times; saldo)"
            _h2 += f" &middot; pond. por {_esc(str(wcol))}"
        out = (head
               + sec_h(_h1)
               + f"<div class='treeui-metrics'>{sec1}</div>"
               + sec_h(_h2)
               + f"<div class='treeui-metrics'>{sec2}</div>")
        h0_css = "font-size:10.5px;color:var(--sub-ink);margin:1px 0 6px;line-height:1.5"
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
                      f"{self._risk_label} igual</b>. "
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

        # teste de ADERÊNCIA da estimativa (alvo) entre DES e OOT, por folha — vai
        # ao lado do p (irmãs). H₀: a folha tem a mesma estimativa em DES e OOT;
        # p alto = estável, p baixo = a estimativa deslocou. OOT = amostra ≠ DES COM
        # alvo cujo nome remete a OOT (senão a 1ª com alvo).
        oot = next((a for a in self._pd_nonref if str(a).upper() == "OOT"), None) or \
            next((a for a in self._pd_nonref if "OOT" in str(a).upper()), None) or \
            (self._pd_nonref[0] if self._pd_nonref else None)
        if self.sample_col is not None and oot is not None and "segmento" in lv.columns:
            lv["p_des_oot"] = [self._sample_value_test(sid, self.ref_sample, oot)[1]
                               for sid in lv["segmento"]]

        # Colunas em blocos legíveis: identificação · % por amostra · alvo médio
        # por amostra (só as que têm alvo) · PSI por amostra · teste de hipótese.
        # `headers` renomeia só a EXIBIÇÃO (a formatação segue pelos nomes reais).
        # Com weight_col configurada, entram as colunas da VISÃO DUPLA contratos ×
        # saldo (saldo_% e as ponderadas) — elas só existem em leaves() nesse caso,
        # então o `in lv.columns` já garante que nada muda sem pesos.
        cols = ["folha"]
        headers = {"folha": "folha", "apelido": "apelido", "descricao": "descrição"}
        if "apelido" in lv.columns:          # só existe quando há apelido definido
            cols.append("apelido")
        cols.append("descricao")
        if self.sample_col is None:
            for c, h in (("repr_%", "repr. %"), ("saldo_%", "% saldo"),
                         ("valor_medio", self._risk_mean),
                         ("valor_medio_pond", f"{self._risk_mean} pond.")):
                if c in lv.columns:
                    cols.append(c); headers[c] = h
        else:
            for a in [self.ref_sample] + self._nonref:       # % DES · % OOT · % ESTAB
                c = f"repr_{a}_%"
                if c in lv.columns:
                    cols.append(c); headers[c] = f"% {ab(a)}"
            if "saldo_%" in lv.columns:      # % do SALDO da carteira (todas as amostras)
                cols.append("saldo_%"); headers["saldo_%"] = "% saldo"
            for a in [self.ref_sample] + self._pd_nonref:    # alvo DES · alvo OOT
                c = f"valor_{a}"
                if c in lv.columns:
                    cols.append(c); headers[c] = f"{self._risk_label} {ab(a)}"
                cp = f"valor_pond_{a}"       # alvo PONDERADO pelo saldo, na amostra
                if cp in lv.columns:
                    cols.append(cp); headers[cp] = f"{self._risk_label} {ab(a)} pond."
            for a in self._nonref:                           # PSI OOT · PSI ESTAB
                c = f"psi_{a}"
                if c in lv.columns:
                    cols.append(c); headers[c] = f"PSI {ab(a)}"
        if "p_vs_prox" in lv.columns:
            cols.append("p_vs_prox"); headers["p_vs_prox"] = "p (irmãs)"
        if "p_des_oot" in lv.columns:
            cols.append("p_des_oot")
            headers["p_des_oot"] = f"p ({ab(self.ref_sample)}×{ab(oot)})"
        return lv, cols, headers

    def _refresh_table(self):
        # a tabela de folhas mora em Diagnóstico e custa ~0.5s (Styler grande):
        # com a aba fora de vista, só marca pendente — o _on_tab_change renderiza
        # ao abrir. Mesmo padrão preguiçoso do IV.
        if getattr(self, "tabs", None) is not None and \
                self.tabs.selected_index != self._diag_tab_index:
            self._table_dirty = True
            return
        lv, cols, headers = self._leaf_table_spec()
        sty = self._style_leaves(lv[cols]).relabel_index(
            [headers[c] for c in cols], axis="columns")
        # tabela larga (muitas amostras): garante a largura natural para NÃO cortar
        # a última coluna (ex.: PSI ESTAB) — rola na horizontal dentro do container.
        sty = sty.set_table_styles(
            [{"selector": "", "props": [("min-width", "max-content")]}], overwrite=False)
        # o scroller (vertical + horizontal) é o W.Box `table_scroll` que envolve este
        # widget na aba Diagnóstico — não embutimos outro <div> rolável aqui (evita
        # barras aninhadas e dá mais altura para ver a tabela inteira).
        self.out_table.value = self._styler_html(sty)

    def _leaves_tsv(self):
        """Tabela de folhas em TSV (tab = coluna) — cola direto no Excel como células
        numéricas de verdade. Números com PONTO decimal; as colunas de % (repr. por
        amostra e % do saldo, quando há ``weight_col``) saem como FRAÇÃO (0–1),
        prontas p/ formatar como % no Excel."""
        lv, cols, headers = self._leaf_table_spec()

        def fmt(col, v):
            if pd.isna(v):
                return ""                       # vazio (não "—") p/ a célula ficar limpa
            if col in ("repr_%", "saldo_%") or (col.startswith("repr_")
                                                and col.endswith("_%")):
                return f"{v / 100:.4f}"         # FRAÇÃO (não multiplicado) — Excel formata como %
            if col.startswith("pd_"):           # alvo (LGD/PD) — fração 0–1, ponto decimal
                return f"{v:.4f}"
            if col.startswith("psi_"):
                return f"{v:.4f}"
            if col in ("p_vs_prox", "p_des_oot"):
                return f"{v:.3f}"
            if isinstance(v, float):
                return f"{v:g}"
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
                return "color:var(--faint-ink)"
            c = ("var(--ok-bg)" if v >= 0.30
                 else "var(--warn-bg)" if v >= 0.20 else "var(--bad-bg)")
            return f"background-color:{c};font-weight:600"

        def auc_bg(v):
            if pd.isna(v):
                return "color:var(--faint-ink)"
            c = ("var(--ok-bg)" if v >= 0.70
                 else "var(--warn-bg)" if v >= 0.60 else "var(--bad-bg)")
            return f"background-color:{c};font-weight:600"

        def r2_bg(v):
            if pd.isna(v):
                return "color:var(--faint-ink)"
            c = ("var(--ok-bg)" if v >= 0.5
                 else "var(--warn-bg)" if v >= 0.2 else "var(--bad-bg)")
            return f"background-color:{c};font-weight:600"

        if self._is_clf:
            # métricas adimensionais/[0,1] em % (KS, AUC, Gini, taxa, acurácia, F1)
            fmt = {c: "{:.1%}" for c in ("taxa_default", "KS", "AUC", "Gini", "Acuracia", "F1")}
            sty = (m.style.map(ks_bg, subset=["KS"]).map(auc_bg, subset=["AUC"]))
        else:
            # MAE/RMSE seguem na unidade do alvo (decimal); só o R² vira %
            fmt = {c: "{:.4f}" for c in ("MAE", "RMSE")}
            fmt["R2"] = "{:.1%}"
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
        identificar a folha por inteiro.

        Memoizado: são ~0.4s por chamada (grade_map + _descrever + alvo por
        folha) e ela roda 2–3× por mutação (dd_leaf, dd_var_leaf, 'ir para
        folha'). A chave cobre tudo de que os rótulos dependem: estrutura
        (versão da árvore), cadeados e apelidos."""
        key = (self.seg._tree_version, frozenset(self.locked),
               tuple(sorted(self.seg._leaf_names_validos().items())))
        cache = getattr(self, "_leaf_opts_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        seg = self.seg
        filhos: dict = {}
        for sid, s in seg.segments.items():
            filhos.setdefault(s["parent"], []).append(sid)
        nota_map, _ = seg._grade_map()
        n_total = len(self.df)
        min_nota = self._min_nota_fn(filhos, nota_map)
        ref = self.ref_sample if self.sample_col is not None else None

        def value_of(sid):
            # MESMO alvo (DES) e leitura só da coluna-alvo dos chips/header/árvore — antes
            # usava a amostra CHEIA e materializava o subframe inteiro por folha a cada
            # refresh, divergindo do número exibido nos painéis.
            return self._node_value(sid, ref)

        opts = []

        def rec(sid):
            s = seg.segments[sid]
            if s["is_leaf"]:
                own = ("TODA A CARTEIRA" if s["parent"] is None
                       else seg._descrever(s["conditions"]))   # caminho COMPLETO, sem cortar
                nome = seg.leaf_name(sid)
                if nome:                     # apelido na frente, descrição completa atrás
                    own = f"{nome} — {own}"
                rep = 100 * s["mask"].sum() / n_total
                lock = "🔒 " if sid in self.locked else ""
                nota = nota_map.get(sid, "?")
                label = f"[{nota:>2}] {lock}{own}  ({self._risk_label} {value_of(sid) * 100:.2f}% · {rep:.0f}%)"
                opts.append((label, sid))
            for c in sorted(filhos.get(sid, []), key=min_nota):   # esquerda→direita
                rec(c)

        rec("root")
        self._leaf_opts_cache = (key, opts)
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

    def _refresh(self, select=None):
        # suspende o observer de folha enquanto reatribuímos os dropdowns: senão a
        # troca de dd_leaf.value re-dispara _on_leaf_change DENTRO do _refresh,
        # renderizando árvore/IV/histograma 2× por mutação.
        # `select` (opcional) força a folha a ficar em foco — usado pelo desfazer/
        # refazer para voltar à folha que estava selecionada naquele estado.
        self._suspend_leaf_obs = True
        try:
            opts = self._ordered_leaf_options()
            leaves = [sid for _, sid in opts]
            cur = select if select is not None else self.dd_leaf.value
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
            # seletor de grupos de folhas-irmãs (aba Diagnóstico) — cada opção é
            # uma run de folhas terminais adjacentes (chaveada pelos sids)
            sib_groups = self.seg.sibling_leaf_groups()
            self._sib_group_map = {g["key"]: g for g in sib_groups}
            sib_opts = [(g["label"], g["key"]) for g in sib_groups]
            cur_sib = self.dd_sib_group.value
            self.dd_sib_group.options = sib_opts
            if cur_sib in self._sib_group_map:
                self.dd_sib_group.value = cur_sib
            elif sib_opts:
                self.dd_sib_group.value = sib_opts[0][1]
        finally:
            self._suspend_leaf_obs = False
        self._sync_leaf_name_field()     # campo de apelido acompanha a folha em foco
        self._sync_move_cut_field()      # corte vigente acompanha a folha em foco

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
        self.out_boot.value = ("<div style='font-size:12px;color:var(--sub-ink)'>Árvore alterada — "
                               "clique em <b>Calcular IC bootstrap</b> para (re)calcular.</div>")
        self.out_discrim.value = ("<div style='font-size:12px;color:var(--sub-ink)'>Árvore alterada — "
                                  "clique num dos botões do card (curvas · lift · métricas por "
                                  "safra) para renderizar.</div>")
        self.out_plot.value = ("<div style='font-size:12px;color:var(--sub-ink)'>Árvore alterada — "
                               "clique em <b>Ver / salvar árvore (imagem)</b> para renderizar.</div>")
        # o placar de saúde, o SQL gerado e a validação também ficam obsoletos —
        # mas essas saídas nascem vazias/ocultas: só recebem a tarja âmbar de
        # desatualizado quando já havia resultado renderizado na tela.
        stale = ("<div style='font-size:12px;color:var(--warn-tx);background:var(--warn-bg);"
                 "border-radius:6px;padding:4px 8px;display:inline-block'>⚠️ Árvore alterada — "
                 "resultado desatualizado. Clique em <b>{botao}</b> para recalcular.</div>")
        if self.out_diag.value:
            self.out_diag.value = stale.format(botao="Avaliar modelo (placar)")
        if self.out_validate.value:
            self.out_validate.value = stale.format(botao="Validar")
        if self.out_sql.value.strip():
            self.out_sql.value = ("-- Árvore alterada — SQL desatualizado. Clique em "
                                  "'Gerar SQL (CASE WHEN)' para regenerar.")
        # a mini-tabela de cenários compara com o estado ATUAL — acompanha as
        # mutações (custo ~zero: metrics/psi memoizados por versão da árvore) e
        # marca uma comparação já renderizada como desatualizada
        self._refresh_scn_panel(stale_diff=True)
        # preview interativo aberto: re-renderiza imagem + hit-map (a árvore mudou);
        # fechado, nada a fazer — o próximo "Ver árvore" já desenha o estado novo
        if self._tree_img_visible():
            self._refresh_tree_widget()
        # canvas da aba "Árvore interativa": redesenha só se estiver à vista; senão
        # marca pendente e o _on_tab_change desenha ao abrir a aba
        if (getattr(self, "tabs", None) is not None
                and self.tabs.selected_index == self._canvas_tab_index):
            self._refresh_canvas()
        else:
            self._cv_dirty = True

    # ==================================================================
    # Entrada / handlers
    # ==================================================================
    def _selected_leaf(self):
        return self.dd_leaf.value

    # ------------------------------------------------------------------
    # Seletores de variável (Dropdown + ordenação por IV)
    # ------------------------------------------------------------------
    def _iv_map(self, sid):
        """IV por variável p/ ordenar os seletores — memoizado por (folha,
        versão da árvore). O cálculo (variable_iv sem PSI) só roda sob demanda
        (toggle ligado); mutações na árvore invalidam via _tree_version."""
        sid = sid if sid in self.seg.segments else "root"
        key = (sid, self.seg._tree_version)
        if key not in self._iv_sort_cache:
            try:
                iv = self.seg.variable_iv(sid, features=list(self.features),
                                          with_psi=False)
                self._iv_sort_cache[key] = dict(zip(iv["variavel"], iv["iv"]))
            except Exception as e:
                self._log(f"(IV p/ ordenação indisponível: {type(e).__name__}: {e})")
                self._iv_sort_cache[key] = {}
        return self._iv_sort_cache[key]

    def _feature_option_labels(self, by_iv=False, sid=None):
        """(rótulos, mapa rótulo→coluna) das opções dos seletores de variável.
        ``by_iv=True`` reordena por IV (desc) na folha ``sid`` e anexa o IV
        entre parênteses ao rótulo; IV indisponível (NaN) vai para o fim."""
        feats = list(self.features)
        ivm = self._iv_map(sid) if by_iv else {}
        if by_iv:
            def chave(f):
                v = ivm.get(f)
                sem_iv = v is None or pd.isna(v)
                return (sem_iv, -(0.0 if sem_iv else float(v)))
            feats = sorted(feats, key=chave)
        labels, mapa = [], {}
        for f in feats:
            lbl = str(self.seg.feature_labels.get(f, f))
            v = ivm.get(f)
            if by_iv and v is not None and not pd.isna(v):
                lbl = f"{lbl} (IV {v:.4f})"
            if lbl in mapa:                # rótulo repetido → desambigua com a coluna
                lbl = f"{lbl} [{f}]"
            mapa[lbl] = f
            labels.append(lbl)
        return labels, mapa

    def _combo_feature(self, combo, mapa, warn=True):
        """Resolve o rótulo selecionado no seletor para o nome da coluna.

        Com o Dropdown o valor é sempre um rótulo da lista, então o 1º caso
        resolve. Os demais são tolerância a estado antigo (rótulo com o sufixo
        "(IV …)" de uma ordenação anterior, ou o nome cru da coluna vindo de um
        ``to_dict``/cenário salvo)."""
        txt = (combo.value or "").strip()
        if not txt:
            return None
        if txt in mapa:
            return mapa[txt]
        if txt in self.features:           # nome da coluna direto
            return txt
        for lbl, f in mapa.items():        # rótulo sem o sufixo de IV
            if lbl.split(" (IV ")[0] == txt:
                return f
        if warn:
            self._log(f"⚠ Variável '{txt}' não reconhecida — escolha uma opção da lista.")
        return None

    def _sel_feature(self, warn=True):
        """Coluna selecionada no seletor da aba Construir (None se inválida)."""
        return self._combo_feature(self.dd_feature, self._feat_by_label, warn=warn)

    def _sel_var(self, warn=True):
        """Coluna selecionada no seletor da aba Análise (None se inválida)."""
        return self._combo_feature(self.dd_var, self._var_by_label, warn=warn)

    def _refresh_feature_options(self):
        """Reconstrói as opções do seletor da aba Construir (toggle de IV ou
        troca de folha com o toggle ligado), preservando a seleção atual."""
        cur = self._sel_feature(warn=False)
        by_iv = self.tg_feat_iv.value
        labels, mapa = self._feature_option_labels(
            by_iv=by_iv, sid=self.dd_leaf.value if by_iv else None)
        self._feat_by_label = mapa
        # Trocar `options` num Dropdown reseta o `value` por um instante — o que
        # dispara _on_feature_change com uma coluna diferente e apaga o preview
        # pendente. Silencia o observer enquanto a lista é reconstruída: ao final
        # a MESMA coluna volta selecionada, só com outro rótulo.
        self._syncing_feat_opts = True
        try:
            self.dd_feature.options = tuple(labels)
            if cur is not None:
                inv = {f: l for l, f in mapa.items()}
                if cur in inv and inv[cur] != self.dd_feature.value:
                    self.dd_feature.value = inv[cur]
        finally:
            self._syncing_feat_opts = False

    def _refresh_var_options(self):
        """Idem p/ o seletor da aba Análise (folha de referência = dd_var_leaf)."""
        cur = self._sel_var(warn=False)
        by_iv = self.tg_var_iv.value
        labels, mapa = self._feature_option_labels(
            by_iv=by_iv, sid=self.dd_var_leaf.value if by_iv else None)
        self._var_by_label = mapa
        self.dd_var.options = tuple(labels)
        if cur is not None:                        # preserva a coluna, novo rótulo
            inv = {f: l for l, f in mapa.items()}
            if cur in inv and inv[cur] != self.dd_var.value:
                self.dd_var.value = inv[cur]

    def _on_var_leaf_iv(self, _):
        """Trocar a folha da Análise reordena o seletor se o toggle estiver
        ligado (ignora as reatribuições programáticas do _refresh)."""
        if self._suspend_leaf_obs or not getattr(self, "tg_var_iv", None):
            return
        if self.tg_var_iv.value:
            self._refresh_var_options()

    def _set_feature_selection(self, feat):
        """Seleciona ``feat`` no seletor da aba Construir (via rótulo atual)."""
        inv = {f: l for l, f in self._feat_by_label.items()}
        if feat in inv:
            self.dd_feature.value = inv[feat]

    def _feature_kind(self):
        sid = self.dd_leaf.value
        feat = self._sel_feature(warn=False)
        if feat is None:
            return "num"                   # entrada inválida: controles neutros
        sub = (self.df if sid is None or sid not in self.seg.segments
               else self.df[self.seg.segments[sid]["mask"]])
        return self.seg._detect_kind(sub, feat, None)

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
        seleção) e reconfigura os controles do modo atual. Reordenar as opções
        (toggle de IV) muda só o RÓTULO da mesma coluna — nesse caso não há o
        que invalidar (o contexto variável+folha é o mesmo)."""
        if getattr(self, "_syncing_feat_opts", False):
            return                                 # só reconstrução da lista
        ctx = (self._sel_feature(warn=False), self.dd_leaf.value)
        if ctx[0] is not None and ctx == getattr(self, "_feat_ctx", None):
            return
        self._feat_ctx = ctx
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

    def _sync_build_mode(self):
        """Mostra a configuração do modo escolhido (auto-fit · auto-merge · podar).

        O ``Δ<alvo> mínimo`` (``sl_gap``) é COMPARTILHADO pelos modos merge e
        prune — os dois handlers leem o mesmo slider. Como só um painel fica
        montado por vez, o widget simplesmente migra de container na troca."""
        self.box_build_cfg.children = self._build_panels[self.tg_build_mode.value]

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
        """Monta um seletor de grupo por categoria presente na folha (ordenadas por alvo)."""
        sid = self.dd_leaf.value
        feat = self._sel_feature(warn=False)
        if feat is None:
            self.cat_box.children = (); return
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
                "<div style='font-size:11px;color:var(--sub-ink)'>Sem categorias nesta folha.</div>"),)
            return
        means = (valid.assign(_c=valid[feat].astype(str))
                 .groupby("_c")[self.target].mean().sort_values())
        order = means.index.tolist()
        n = len(order)
        rows = [W.HTML("<div style='font-size:11px;color:var(--muted);margin-bottom:4px'>"
                       f"Categorias no <b>mesmo grupo</b> viram um nó. Ordenadas por {self._risk_label}. "
                       "Faltantes (NaN) já viram um nó próprio.</div>")]
        for k, c in enumerate(order, 1):
            dd = W.Dropdown(options=[(f"grupo {g}", g) for g in range(1, n + 1)], value=k,
                            layout=W.Layout(width="110px"))
            self._cat_widgets[c] = dd
            import html as _html             # escapa o nome da categoria (dados)
            lab = W.HTML(f"<span style='font-size:12px'><b>{_html.escape(str(c))}</b>"
                         f"<span style='color:var(--sub-ink)'> · {self._risk_label} {means[c]:.3f}</span></span>")
            rows.append(W.HBox([dd, lab], layout=W.Layout(align_items="center")))
        na_n = int(s.isna().sum())
        if na_n:
            rows.append(W.HTML(f"<div style='font-size:11px;color:var(--warn-tx);margin-top:3px'>"
                               f"+ <b>(faltante)</b>: {na_n} linhas → nó próprio automático</div>"))
        self.cat_box.children = tuple(rows)

    def _cat_groups(self):
        if (getattr(self, "_cat_ctx", None) != (self._sel_feature(warn=False),
                                                self.dd_leaf.value)
                or not getattr(self, "_cat_widgets", None)):
            self._rebuild_cat_box()
        grupos = {}
        for c, dd in self._cat_widgets.items():
            grupos.setdefault(dd.value, []).append(c)
        return [grupos[g] for g in sorted(grupos)]

    def _on_collapse(self, _):
        sid = self._selected_leaf()
        if sid is None or sid not in self.seg.segments:
            self._log("Nenhuma folha selecionada."); return
        parent = self.seg.segments[sid]["parent"]
        if parent is None:
            self._log("Esta folha é a raiz — não há pai para recolher."); return
        redo_bak = list(self._redo)
        antes = self._delta_snapshot()
        self._checkpoint()
        try:
            self.seg.collapse(parent)
        except Exception as e:
            self._revert_checkpoint(redo_bak)
            self._log(f"Erro ao recolher: {type(e).__name__}: {e}"); return
        self.locked &= set(self.seg.segments)
        self._pending = None
        self._refresh()
        if parent in [s for s, seg in self.seg.segments.items() if seg["is_leaf"]]:
            self.dd_leaf.value = parent
        self._log_delta("recolher", antes)

    def _on_merge(self, side):
        sid = self._selected_leaf()
        if sid is None or sid not in self.seg.segments:
            self._log("Selecione uma folha."); return
        parent = self.seg.segments[sid]["parent"]
        before = set(self.seg.segments)
        redo_bak = list(self._redo)
        antes = self._delta_snapshot()
        self._checkpoint()
        try:
            self.seg.merge_leaf(sid, side=side)
        except Exception as e:
            self._revert_checkpoint(redo_bak)
            self._log(f"Erro ao unir folhas: {type(e).__name__}: {e}"); return
        if set(self.seg.segments) == before:      # no-op (ex.: vizinha não é folha)
            self._revert_checkpoint(redo_bak)
            self._log("Nada a unir deste lado (a vizinha não é uma folha terminal).")
            return
        self.locked &= set(self.seg.segments)
        self._pending = None
        novos = [i for i in self.seg.segments
                 if i not in before and self.seg.segments[i]["is_leaf"]]
        self._refresh()
        folhas = [s for s, seg in self.seg.segments.items() if seg["is_leaf"]]
        alvo = (novos[0] if novos else (parent if parent in folhas else None))
        if alvo in folhas:
            self.dd_leaf.value = alvo
        self._log_delta("unir folhas", antes)

    def _on_merge_missing(self, _):
        sid = self._selected_leaf()
        if sid is None or sid not in self.seg.segments:
            self._log("Selecione a folha POPULADA de destino."); return
        before = set(self.seg.segments)
        redo_bak = list(self._redo)
        antes = self._delta_snapshot()
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
        self._log_delta("juntar missings", antes)

    # ------------------------------------------------------------------
    # Mover corte numérico: hi da folha ↔ lo da irmã à direita (sem recolher
    # o pai; a irmã pode ter sub-splits — o segmentador propaga ao subtree)
    # ------------------------------------------------------------------
    def _left_cut_owner(self, sid):
        """Irmã que possui o corte à ESQUERDA de ``sid`` (o ``hi`` dela é o ``lo``
        desta folha), ou ``None``. Reusa o ``movable_cut`` do segmentador em vez de
        reimplementar a adjacência: a dona é a irmã cujo vizinho à direita é ``sid``."""
        s = self.seg.segments.get(sid)
        if s is None or not s.get("conditions"):
            return None
        pai = s.get("parent")
        for cid, outro in self.seg.segments.items():
            if cid == sid or outro.get("parent") != pai:
                continue
            info = self.seg.movable_cut(cid)
            if info and info["sibling"] == sid:
                return cid
        return None

    def _move_cut_owner(self):
        """Folha cujo ``hi`` será movido, conforme o lado escolhido no seletor."""
        sid = self._selected_leaf()
        if sid is None:
            return None
        return sid if self.dd_move_side.value == "dir" else self._left_cut_owner(sid)

    def _sync_move_cut_field(self):
        """Monta o seletor de lado com os cortes que a folha selecionada realmente
        tem e espelha o corte vigente do lado escolhido no campo 'novo corte'.
        Sem corte móvel de nenhum lado, desabilita e explica o porquê."""
        sid = self.dd_leaf.value
        valido = sid is not None and sid in self.seg.segments
        dono_esq = self._left_cut_owner(sid) if valido else None
        tem_dir = bool(self.seg.movable_cut(sid)) if valido else False
        opcoes = ([("◀ à esquerda", "esq")] if dono_esq else []) + \
                 ([("à direita ▶", "dir")] if tem_dir else [])
        self.out_move_cut.value = ""          # preview antigo não vale p/ outra folha
        if not opcoes:
            self.dd_move_side.disabled = True
            self.lbl_move_cut.value = (
                "<div class='treeui-legend'>Mover corte: só para folha vizinha de um "
                "corte numérico — esta não tem corte móvel de nenhum lado.</div>")
            for w in (self.tx_move_cut, self.btn_move_prev, self.btn_move_cut):
                w.disabled = True
            return
        # trocar as options dispara o observer: silencia p/ não re-sincronizar no meio
        self._syncing_side = True
        try:
            manter = self.dd_move_side.value
            self.dd_move_side.options = opcoes
            valores = [v for _, v in opcoes]
            self.dd_move_side.value = manter if manter in valores else valores[-1]
        finally:
            self._syncing_side = False
        self.dd_move_side.disabled = len(opcoes) == 1
        self._render_move_cut_side()

    def _render_move_cut_side(self):
        """Rótulo + valor do campo para o lado atualmente escolhido."""
        dono = self._move_cut_owner()
        info = self.seg.movable_cut(dono) if dono else None
        if info is None:
            self.lbl_move_cut.value = (
                "<div class='treeui-legend'>Corte indisponível deste lado.</div>")
            for w in (self.tx_move_cut, self.btn_move_prev, self.btn_move_cut):
                w.disabled = True
            return
        rot = self.seg.feature_labels.get(info["feature"], info["feature"])
        lado = "à esquerda" if self.dd_move_side.value == "esq" else "à direita"
        self.lbl_move_cut.value = (
            f"<div class='treeui-legend'>Corte {lado} em <b>{_esc(rot)}</b>: "
            f"<b>{_fmt(info['cut'])}</b> · válido entre {_fmt(info['lo'])} e "
            f"{_fmt(info['hi_sib'])} (exclusivo)</div>")
        self.tx_move_cut.value = info["cut"]
        for w in (self.tx_move_cut, self.btn_move_prev, self.btn_move_cut):
            w.disabled = False

    def _on_move_side(self, _):
        if not getattr(self, "_syncing_side", False):
            self.out_move_cut.value = ""      # preview do outro lado não vale
            self._render_move_cut_side()

    def _on_move_cut_preview(self, _):
        sid = self._move_cut_owner()
        if sid is None or sid not in self.seg.segments:
            self._log("Selecione uma folha com corte móvel."); return
        try:
            tbl = self.seg.preview_move_cut(sid, self.tx_move_cut.value)
        except Exception as e:
            self.out_move_cut.value = ("<div style='font-size:12px;color:var(--bad-tx)'>"
                                       f"{_esc(e)}</div>")
            return
        self.out_move_cut.value = self._df_html(tbl, center=True)
        self._log(f"👁 preview do corte em {_fmt(float(self.tx_move_cut.value))} — "
                  "confira os dois lados e clique em Mover corte para efetivar.")

    def _on_move_cut(self, _):
        sid = self._move_cut_owner()
        if sid is None or sid not in self.seg.segments:
            self._log("Selecione uma folha com corte móvel."); return
        before = set(self.seg.segments)
        redo_bak = list(self._redo)
        antes = self._delta_snapshot()
        self._checkpoint()
        try:
            self.seg.move_cut(sid, self.tx_move_cut.value, verbose=False)
        except Exception as e:
            self._revert_checkpoint(redo_bak)
            self._log(f"Erro ao mover o corte: {type(e).__name__}: {e}")
            self.out_move_cut.value = ("<div style='font-size:12px;color:var(--bad-tx)'>"
                                       f"{_esc(e)}</div>")
            return
        if set(self.seg.segments) == before:     # no-op: corte igual ao vigente
            self._revert_checkpoint(redo_bak)
            self._log("O novo corte é igual ao vigente — nada a mudar.")
            return
        self.locked &= set(self.seg.segments)
        self._pending = None
        novos = [i for i in self.seg.segments
                 if i not in before and self.seg.segments[i]["is_leaf"]]
        self._refresh()
        folhas = [s for s, sg in self.seg.segments.items() if sg["is_leaf"]]
        if novos and novos[0] in folhas:         # 1º novo = a folha à esq. do corte
            self.dd_leaf.value = novos[0]
        self._log_delta("mover corte", antes)

    def _on_suggest(self, _):
        sid = self._selected_leaf()
        if sid is None:
            self._log("Selecione uma folha."); return
        with self._busy(self.btn_suggest, msg="procurando a melhor variável…"):
            sug = self.seg.suggest_split(sid)
            if sug["feature"] is None:
                self._log("Nenhuma variável informativa para esta folha — IV muito baixo.")
                return
            self._set_feature_selection(sug["feature"])
            self.tg_mode.value = "Ótimo"
            lbl = self.seg.feature_labels.get(sug["feature"], sug["feature"])
            self._log(f"Sugestão para esta folha: dividir por '{lbl}' "
                      f"(IV={sug['iv']:.4f}, {sug['forca']}).")
            self._log("Já deixei a variável selecionada no modo Ótimo — "
                      "rode o 👁 Preview e depois Criar segmento.")

    def _on_suggest3(self, _):
        sid = self._selected_leaf()
        if sid is None:
            self.out_suggest.value = "<i>Selecione uma folha na aba Construir.</i>"
            return
        with self._busy(self.btn_suggest3, status=self.out_suggest,
                        msg="rankeando os melhores splits…"):
            try:
                sug = self.seg.suggest_splits(sid, top=5)
            except Exception as e:
                self.out_suggest.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro: "
                                          f"{type(e).__name__}: {e}</div>")
                return
            if sug.empty:
                self.out_suggest.value = "<i>Nenhuma variável informativa para esta folha.</i>"
                self._log("Sugestão: nenhuma variável com IV suficiente nesta folha.")
                return
            disp = sug.copy()
            disp["passa_teste"] = disp["passa_teste"].map({True: "✅", False: "—"})
            disp = disp.rename(columns={"n_bins": "nº bins", "passa_teste": "passa teste"})
            self.out_suggest.value = self._df_html(disp, center=True, color=True)
            self._log(f"TOP {len(sug)} splits sugeridos para a folha selecionada.")

    def _on_importance(self, _):
        with self._busy(self.btn_importance, msg="calculando a importância…"):
            try:
                fi = self.seg.feature_importance()
            except Exception as e:
                self.out_importance.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro: "
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
                chart = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro no gráfico: "
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
            self._log("Importância das variáveis na árvore calculada.")

    def _on_fallback(self, ch):
        """Persiste a escolha de fallback no segmentador (viaja no to_dict/save)
        e atualiza o chip de linhas sem rota na barra de status."""
        self.seg.fallback = ch["new"]
        self._set_html(self.bar, "bar", self._status_html())
        self._log("Fallback p/ não classificados: "
                  + ("pior nota (maior risco)." if ch["new"] == "pior_nota"
                     else "NULL (sem fallback)."))

    def _on_sql(self, _):
        tbl = (self.tx_sql_table.value or "minha_tabela").strip()
        try:
            self.out_sql.value = self.seg.to_sql(table=tbl,
                                                 fallback=self.dd_fallback.value)
            self._log("SQL gerado — selecione tudo na caixa e copie (Ctrl+C).")
        except Exception as e:
            self.out_sql.value = f"-- Erro ao gerar SQL: {type(e).__name__}: {e}"

    def _diff_html(self, d, label_a=None, label_b=None) -> str:
        """Renderização COMPARTILHADA da comparação de duas árvores (saída de
        ``diff_trees``): concordância, tabela-resumo (folhas + Δ de métricas) e
        crosstab de migração de notas. Usada pelo card "Comparar duas árvores"
        (JSON em disco) e pelo card "Cenários" (versões em memória). ``label_a``/
        ``label_b`` (já escapados p/ HTML) identificam A e B quando informados."""
        mig = d["migracao"].copy()
        mig.index = [f"A·{i}" for i in mig.index]
        mig.columns = [f"B·{c}" for c in mig.columns]
        rotulo = (f" · A = {label_a} · B = {label_b}" if (label_a or label_b) else "")
        return (f"<div class='treeui-legend'>Concordância de folhas (A=B): "
                f"<b>{d['concordancia']:.1%}</b>{rotulo}</div>"
                + self._df_html(d["resumo"], center=True)
                + "<div class='treeui-h' style='margin-top:8px'>Migração de folhas "
                  "(linhas = árvore A · colunas = árvore B)</div>"
                + mig.to_html(border=0))

    def _on_diff(self, _):
        from .segmenter import TreeSegmenter
        path = (self.tx_diff_path.value or "").strip()
        if not path:
            self.out_diff.value = "<i>Informe o caminho do JSON da árvore B.</i>"
            return
        with self._busy(self.btn_diff, status=self.out_diff,
                        msg="comparando as duas árvores…"):
            try:
                other = TreeSegmenter.load(path, self.df)
                d = self.seg.diff_trees(other)
            except Exception as e:
                self.out_diff.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro ao "
                                       f"comparar: {type(e).__name__}: {e}</div>")
                return
            self.out_diff.value = self._diff_html(d)
            self._log(f"Comparação concluída — concordância {d['concordancia']:.1%}.")

    # ==================================================================
    # Cenários nomeados EM MEMÓRIA (card "Cenários", aba Avançado)
    #   nome → {"data": to_dict(), "locked": folhas travadas, "resumo": linha da
    #   mini-tabela}. Vivem SÓ nesta sessão (memória do kernel): fechar/reiniciar
    #   o notebook os descarta — persistência de verdade é Salvar (JSON)/MLflow.
    #   Restaurar passa pelo _checkpoint (desfazível com ↶); Comparar reconstrói
    #   o cenário em memória via from_dict sobre o MESMO df, compartilhando o
    #   cache de máscaras (_prime_mask_cache) p/ não recomputar condições iguais.
    # ==================================================================
    def _scn_resumo_row(self, seg) -> dict:
        """Linha da mini-tabela de cenários: nº de folhas, discriminação por
        amostra (KS/AUC na classificação · R² na regressão) e PSI máximo."""
        row = {"folhas": int(sum(s["is_leaf"] for s in seg.segments.values()))}
        try:
            m = seg.metrics()                     # memoizado por versão da árvore
            cols = [("KS", "KS"), ("AUC", "AUC")] if self._is_clf else [("R2", "R²")]
            for _, r in m.iterrows():
                for col, rot in cols:
                    v = r.get(col)
                    if pd.notna(v):
                        row[f"{rot} · {r['amostra']}"] = round(float(v), 4)
        except Exception:
            pass                                  # sem métricas → linha só com folhas
        if self.sample_col is not None:
            try:
                p = seg.psi()
                if len(p):
                    row["PSI máx"] = round(float(p["psi"].max()), 4)
            except Exception:
                pass
        return row

    def _scn_row(self, nome):
        """Linha da lista de cenários: nome + botões Restaurar / Comparar / remover."""
        res = self._scenarios[nome]["resumo"]
        lab = W.HTML(f"<div style='font-size:12.5px'>🔖 <b>{_esc(nome)}</b> "
                     f"<span style='color:var(--muted)'>· {res.get('folhas', '?')} "
                     "folhas</span></div>",
                     layout=W.Layout(flex="1 1 auto", min_width="120px"))
        bt_r = W.Button(description="Restaurar", icon="history",
                        tooltip="Volta a árvore para este cenário "
                                "(desfazível com ↶ Desfazer)",
                        layout=W.Layout(width="120px"))
        bt_c = W.Button(description="Comparar com o atual", icon="exchange",
                        button_style="warning",
                        tooltip="Reconstrói o cenário em memória e o compara com a "
                                "árvore atual (concordância, migração e Δ de métricas)",
                        layout=W.Layout(width="190px"))
        bt_x = W.Button(description="✕", tooltip="Remove o cenário desta sessão",
                        layout=W.Layout(width="34px"))
        bt_r.on_click(lambda b, n=nome: self._on_scn_restore(n, b))
        bt_c.on_click(lambda b, n=nome: self._on_scn_compare(n, b))
        bt_x.on_click(lambda b, n=nome: self._on_scn_remove(n))
        return W.HBox([lab, bt_r, bt_c, bt_x],
                      layout=W.Layout(width="100%", align_items="center",
                                      margin="1px 0"))

    def _refresh_scn_panel(self, stale_diff=False, rebuild_rows=False):
        """Atualiza o card de cenários: a mini-tabela resumo (estado ATUAL na 1ª
        linha + um cenário por linha) acompanha toda mutação; as LINHAS de botões
        só são reconstruídas com ``rebuild_rows=True`` (salvar/remover — recriar
        widgets a cada _refresh acumularia instâncias no registry do ipywidgets).
        ``stale_diff=True`` marca uma comparação já renderizada como desatualizada
        (a árvore atual ou a lista de cenários mudou)."""
        self.btn_scn_clear.disabled = not self._scenarios
        if rebuild_rows:
            for row in self.box_scn_list.children:   # libera os widgets antigos
                for w in row.children:
                    w.close()
                row.close()
            self.box_scn_list.children = tuple(self._scn_row(n)
                                               for n in self._scenarios)
        if not self._scenarios:
            self._set_html(self.out_scn_summary, "scn_summary",
                           "<div class='treeui-legend'><i>Nenhum cenário salvo ainda — dê um "
                           "nome (opcional) e clique em <b>Salvar cenário</b>.</i></div>")
            if stale_diff:
                self.out_scn_diff.value = ""
            return
        rows = [{"cenário": "— atual —", **self._scn_resumo_row(self.seg)}]
        rows += [{"cenário": nome, **scn["resumo"]}
                 for nome, scn in self._scenarios.items()]
        self._set_html(self.out_scn_summary, "scn_summary",
                       self._df_html(pd.DataFrame(rows), center=True))
        if stale_diff and self.out_scn_diff.value:
            self.out_scn_diff.value = (
                "<div style='font-size:12px;color:var(--warn-tx);background:var(--warn-bg);"
                "border-radius:6px;padding:4px 8px;display:inline-block'>⚠️ Árvore ou "
                "cenários alterados — comparação desatualizada. Clique em <b>Comparar com "
                "o atual</b> para recalcular.</div>")

    def _on_scn_save(self, _):
        nome = (self.tx_scn_name.value or "").strip()
        if not nome:                              # sem nome → nome sequencial
            nome = f"cenário {len(self._scenarios) + 1}"
        sobrescreve = nome in self._scenarios
        self._scenarios[nome] = {
            "data": self.seg.to_dict(),           # estrutura + apelidos + fallback
            "locked": set(self.locked),           # folhas travadas acompanham a foto
            # resumo calculado UMA vez no save (a foto não muda; metrics/psi do
            # estado atual são memoizados → custo ~zero)
            "resumo": self._scn_resumo_row(self.seg),
        }
        self.tx_scn_name.value = ""
        self._refresh_scn_panel(rebuild_rows=True)
        self._log(f"Cenário '{nome}' " + ("sobrescrito" if sobrescreve else "salvo")
                  + " em memória — vale só nesta sessão (p/ disco: Salvar JSON/MLflow).")

    def _on_scn_restore(self, nome, botao=None):
        """Volta a árvore para o cenário ``nome`` — passa pelo :meth:`_checkpoint`,
        logo é desfazível com ↶ Desfazer (a estrutura; o fallback segue a foto)."""
        scn = self._scenarios.get(nome)
        if scn is None:
            return
        antes = self._delta_snapshot()
        self._checkpoint()                        # restaurar é desfazível
        data = scn["data"]
        meta = data.get("meta", {})
        # máscaras vivas → cache por condições: segmentos iguais entre o estado
        # atual e o cenário não são recalculados (mesma amortização do undo/redo)
        self.seg._prime_mask_cache()
        self.seg._load_segments(data["segments"])
        nomes = meta.get("leaf_names") or {}
        self.seg.leaf_names = {sid: str(n) for sid, n in nomes.items()
                               if n and sid in self.seg.segments
                               and self.seg.segments[sid]["is_leaf"]}
        self.seg.fallback = meta.get("fallback")
        # espelha o fallback restaurado no dropdown (observer re-loga se mudar)
        self.dd_fallback.value = ("pior_nota" if self.seg.fallback == "pior_nota"
                                  else None)
        self.locked = set(scn.get("locked") or set()) & set(self.seg.segments)
        self._pending = None
        self._refresh()
        self._log_delta(f"cenário '{nome}' restaurado", antes)

    def _on_scn_compare(self, nome, botao=None):
        """Compara a árvore ATUAL (A) com o cenário ``nome`` (B) reconstruído em
        memória sobre o MESMO DataFrame, e renderiza no card (helper comum)."""
        from .segmenter import TreeSegmenter
        scn = self._scenarios.get(nome)
        if scn is None:
            return
        botoes = (botao,) if botao is not None else ()
        with self._busy(*botoes, status=self.out_scn_diff,
                        msg="comparando o cenário com a árvore atual…"):
            try:
                # prime → cache compartilhado: o from_dict reusa as máscaras dos
                # segmentos que o cenário tem em comum com a árvore atual
                cache = self.seg._prime_mask_cache()
                other = TreeSegmenter.from_dict(scn["data"], self.df,
                                                mask_cache=cache)
                d = self.seg.diff_trees(other)
            except Exception as e:
                self.out_scn_diff.value = (
                    f"<div style='color:var(--bad-tx);font-size:12px'>Erro ao comparar: "
                    f"{type(e).__name__}: {e}</div>")
                return
            self.out_scn_diff.value = self._diff_html(
                d, label_a="árvore atual", label_b=f"cenário '{_esc(nome)}'")
            self._log(f"Comparação com o cenário '{nome}' — concordância "
                      f"{d['concordancia']:.1%}.")

    def _on_scn_remove(self, nome):
        if self._scenarios.pop(nome, None) is not None:
            self._refresh_scn_panel(stale_diff=True, rebuild_rows=True)
            self._log(f"Cenário '{nome}' removido da sessão.")

    def _on_scn_clear(self, b):
        """Apaga TODOS os cenários da sessão (destrutivo → confirma em 2 cliques)."""
        def _limpar():
            n = len(self._scenarios)
            if not n:
                self._log("Nenhum cenário salvo para limpar.")
                return
            self._scenarios.clear()
            self.out_scn_diff.value = ""
            self._refresh_scn_panel(stale_diff=True, rebuild_rows=True)
            self._log(f"{n} cenário(s) removido(s) da sessão.")
        self._confirm_twice(b, _limpar)

    def _on_autofit(self, _):
        sid = self._selected_leaf()
        so_folha = sid is not None and sid != "root" and sid in self.seg.segments
        depth = int(self.sl_depth.value)
        criterion = self.dd_criterion.value
        cmin = float(self.sl_autoconc_min.value) if self.cb_autoconc_min.value else None
        cmax = float(self.sl_autoconc_max.value) if self.cb_autoconc_max.value else None
        alvo = self._leaf_label(sid) if so_folha else "TODA A CARTEIRA"
        lim = []
        if cmin is not None:
            lim.append(f"folha ≥ {cmin:.1%}")
        if cmax is not None:
            lim.append(f"quebra ≤ {cmax:.0%}")
        slim = (", " + " · ".join(lim)) if lim else ""
        scrit = "" if criterion == "optbin" else f", critério={criterion}"
        self._log(f"Auto-fit em '{alvo}' (profundidade ≤ {depth}{slim}{scrit})…")
        with self._busy(self.btn_autofit, self.btn_img_autofit, self.btn_cv_autofit,
                        msg="rodando o auto-fit…"):
            redo_bak = list(self._redo)
            antes = self._delta_snapshot()
            self._checkpoint()
            try:
                self.seg.fit_auto(max_depth=depth, min_leaf_repr=cmin, max_bin_repr=cmax,
                                  criterion=criterion, subtree=sid if so_folha else None,
                                  from_scratch=not so_folha)
            except Exception as e:
                self._revert_checkpoint(redo_bak)
                self._log(f"Erro no auto-fit: {type(e).__name__}: {e}"); return
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
            n = sum(s["is_leaf"] for s in self.seg.segments.values())
            escopo = "nesta folha" if so_folha else "na árvore"
            self._log(f"Auto-fit concluído {escopo}: {n} folhas no total. "
                      "Refine à mão: funda, recolha ou divida onde quiser.")
            self._log_delta("auto-fit", antes)

    def _on_mlflow(self, _):
        exp = self.tx_experiment.value.strip() or None
        run = self.tx_runname.value.strip() or None
        model_name = self.tx_model.value.strip() or None
        uc = self.cb_uc.value
        if uc and not model_name:
            self._log("Para registrar no Unity Catalog, informe o nome no formato "
                      "catalogo.schema.modelo.")
            return
        if uc and model_name.count(".") != 2:
            self._log(f"Nome UC inválido: '{model_name}'. Use 3 níveis: catalogo.schema.modelo.")
            return
        self._log("Salvando no MLflow…")
        with self._busy(self.btn_mlflow, msg="registrando no MLflow…"):
            try:
                rid = self.seg.log_to_mlflow(
                    experiment=exp, run_name=run,
                    registered_model_name=model_name,
                    registry_uri="databricks-uc" if uc else None,
                    save_base=self.cb_savebase.value,
                    verbose=False)
                msg = f"✓ Run {rid[:8]}… salvo (régua, métricas e modelo pyfunc)."
                if model_name:
                    msg += f"\nModelo registrado em '{model_name}' — nova versão no Model Registry."
                    self._log(msg)
                    self._log(f"Para scoring: mlflow.pyfunc.load_model('models:/{model_name}/<versão>')"
                              " e use .predict.")
                else:
                    self._log(msg)
            except ImportError:
                self._log("MLflow não está instalado neste ambiente. Instale com: %pip install mlflow")
            except Exception as e:
                self._log(f"Erro ao salvar no MLflow: {type(e).__name__}: {e}")

    def _on_clear_log(self, _):
        self._log_lines = []              # zera o histórico do console
        self.out_log.clear_output()       # limpa a área de preview/log

    def _on_spark_apply(self, _):
        """Card 'Reconstruir folhas': com o nome da tabela, aplica via Spark
        (gravando opcionalmente); sem, aplica na base em memória (``ui.score_df``
        ou a carregada). O progresso por etapa sai em ``out_spark_progress`` e o
        erro fica RESUMIDO no card — o detalhe completo vai para o Console."""
        name = self.tx_spark_in.value.strip()
        out_name = self.tx_spark_out.value.strip() or None
        self._spark_steps = []                     # zera a tabela de progresso
        self._render_spark_progress()
        self.out_spark.value = ""                  # limpa o resultado anterior
        cb = self._spark_progress_cb
        with self._busy(self.btn_spark_apply, msg="aplicando a régua…"):
            try:
                if name:                           # tabela Databricks (Spark)
                    out, resumo = self.seg.apply_table(
                        name, col_nota="folha", output_table=out_name,
                        progress_callback=cb)
                    self.spark_result = out        # Spark DataFrame (lazy)
                    gravou = (f" Gravado em <code>{_esc(out_name)}</code>." if out_name
                              else " Spark DataFrame em <code>ui.spark_result</code>.")
                    self.out_spark.value = (
                        f"<div class='treeui-legend'>✓ Régua aplicada em "
                        f"<code>{_esc(name)}</code>.{gravou}</div>"
                        + self._spark_resumo_html(resumo))
                    self._log(f"✓ régua aplicada em '{name}'"
                              + (f" e gravada em '{out_name}'." if out_name
                                 else ". Spark DataFrame em ui.spark_result."))
                else:                              # em memória (pandas), sem Spark
                    base = self.score_df if self.score_df is not None else self.df
                    origem = ("ui.score_df" if self.score_df is not None
                              else "base carregada")
                    out, resumo = self.seg.apply_table(base, col_nota="folha",
                                                       progress_callback=cb)
                    self.result = out
                    self.out_spark.value = (
                        f"<div class='treeui-legend'>✓ Régua aplicada em memória "
                        f"({origem}): {out.shape[0]} linhas × {out.shape[1]} colunas, "
                        f"em <code>ui.result</code>.</div>"
                        + self._spark_resumo_html(resumo))
                    self._log(f"✓ régua aplicada em memória ({origem}) → ui.result "
                              f"({out.shape[0]} linhas).")
            except Exception as e:
                # o erro vai para o CONSOLE; no card fica só o aviso curto e a
                # etapa que falhou marcada com ❌ na tabela de progresso
                for row in reversed(self._spark_steps):
                    if row["status"] == "run":
                        row["status"] = "err"
                        row["detail"] = type(e).__name__   # detalhe completo no Console
                        break
                self._render_spark_progress()
                self.out_spark.value = (
                    "<div class='treeui-legend' style='color:var(--bad-ink)'>✗ Falha ao "
                    "aplicar a régua — veja o <b>Console</b> (rodapé) para o detalhe."
                    "</div>")
                self._log(f"[reconstruir folhas] ERRO: {type(e).__name__}: {e}")

    def _spark_progress_cb(self, key, label, status, detail=""):
        """Callback de progresso da aplicação da régua (passado a ``apply_table``):
        cria ou atualiza a linha da etapa ``key`` e re-renderiza a tabela."""
        for row in self._spark_steps:
            if row["key"] == key:
                row["status"] = status
                if detail:
                    row["detail"] = detail
                break
        else:
            self._spark_steps.append({"key": key, "label": label,
                                      "status": status, "detail": detail})
        self._render_spark_progress()

    def _render_spark_progress(self):
        """Renderiza a tabela de progresso da aplicação da régua (ler tabela →
        aplicar → gravar → resumo), com uma linha ⏳/✅/❌ por etapa."""
        if not self._spark_steps:
            self.out_spark_progress.value = ""
            return
        icon = {"run": "⏳", "ok": "✅", "err": "❌"}
        cor = {"run": "var(--warn-ink)", "ok": "var(--ok-ink)", "err": "var(--bad-ink)"}
        rot = {"run": "processando…", "ok": "concluído", "err": "erro"}
        trs = ""
        for r in self._spark_steps:
            st = r["status"]
            trs += (f"<tr><td style='padding:4px 10px'>{icon.get(st, '')}</td>"
                    f"<td style='padding:4px 10px'>{r['label']}</td>"
                    f"<td style='padding:4px 10px;color:{cor.get(st, 'var(--ink)')};font-weight:600'>"
                    f"{rot.get(st, st)}</td>"
                    f"<td style='padding:4px 10px;color:var(--muted)'>{r.get('detail', '')}</td></tr>")
        self.out_spark_progress.value = (
            "<div class='treeui-legend' style='margin-top:6px'>Progresso da aplicação</div>"
            "<table style='border-collapse:collapse;font-size:12px;width:100%;margin:2px 0 8px'>"
            "<thead><tr style='background:var(--tbl-head-bg)'>"
            "<th style='padding:4px 10px'></th>"
            "<th style='padding:4px 10px;text-align:left'>Etapa</th>"
            "<th style='padding:4px 10px;text-align:left'>Status</th>"
            "<th style='padding:4px 10px;text-align:left'>Detalhe</th>"
            f"</tr></thead><tbody>{trs}</tbody></table>")

    def _spark_resumo_html(self, resumo):
        """Tabela compacta da distribuição por folha (resumo do ``apply_table``,
        já materializado — nada é recomputado aqui): folha · linhas · %."""
        if resumo is None or not len(resumo):
            return ""
        trs = ""
        for _, r in resumo.iterrows():
            folha = "— sem rota" if pd.isna(r.iloc[0]) else r.iloc[0]
            linhas = f"{int(r['linhas']):,}".replace(",", ".")
            pct = f"{float(r['pct']):.1%}"
            trs += (f"<tr><td style='padding:3px 10px'>{folha}</td>"
                    f"<td style='padding:3px 10px;text-align:right'>{linhas}</td>"
                    f"<td style='padding:3px 10px;text-align:right;color:var(--muted)'>"
                    f"{pct}</td></tr>")
        return ("<div class='treeui-legend' style='margin-top:6px'>Distribuição por folha</div>"
                "<table style='border-collapse:collapse;font-size:12px;margin:2px 0 6px'>"
                "<thead><tr style='background:var(--tbl-head-bg)'>"
                f"<th style='padding:3px 10px;text-align:left'>{_esc(resumo.columns[0])}</th>"
                "<th style='padding:3px 10px;text-align:right'>linhas</th>"
                "<th style='padding:3px 10px;text-align:right'>%</th>"
                f"</tr></thead><tbody>{trs}</tbody></table>")

    def _parse_cuts(self, feature, sid):
        sub = self.df[self.seg.segments[sid]["mask"]]
        kind = self.seg._detect_kind(sub, feature, None)
        if kind == "num":
            raw = self.tx_cuts.value.strip()
            return [float(x) for x in raw.replace(";", ",").split(",")
                    if x.strip()] if raw else None
        grupos = self._cat_groups()
        return grupos if grupos else None

    def _sync_leaf_name_field(self):
        """Espelha o apelido da folha selecionada no campo de texto, sem disparar
        o observer (a reatribuição programática não é digitação do usuário)."""
        sid = self.dd_leaf.value
        nome = self.seg.leaf_name(sid) if sid is not None else None
        self._suspend_name_obs = True
        try:
            self.tx_leaf_name.value = nome or ""
            self.tx_leaf_name.disabled = (sid is None or sid not in self.seg.segments)
        finally:
            self._suspend_name_obs = False

    def _on_leaf_name(self, _):
        """Aplica o apelido digitado à folha selecionada — imediato, com undo
        (o :meth:`_checkpoint` guarda o estado anterior dos apelidos)."""
        if self._suspend_name_obs:
            return
        sid = self.dd_leaf.value
        if sid is None or sid not in self.seg.segments or \
                not self.seg.segments[sid]["is_leaf"]:
            return
        novo = " ".join((self.tx_leaf_name.value or "").split())
        atual = self.seg.leaf_name(sid) or ""
        if novo == atual:
            self._sync_leaf_name_field()      # normaliza o texto exibido
            return
        redo_bak = list(self._redo)
        self._checkpoint()                    # desfazer restaura o apelido anterior
        try:
            self.seg.set_leaf_name(sid, novo or None)
        except Exception as e:
            self._revert_checkpoint(redo_bak)
            self._log(f"Erro ao apelidar a folha: {type(e).__name__}: {e}")
            return
        acao = "aplicado" if novo else "removido"
        self._log(f"🏷️ apelido {acao}: {self._leaf_label(sid)}")
        # apelido não muda a estrutura: atualiza árvore/dropdowns/cartões (mesmo
        # caminho leve do cadeado) + a tabela de folhas (coluna 'apelido')
        self._refresh_lock_labels()
        self._refresh_table()
        self._sync_leaf_name_field()

    def _on_leaf_change(self, _):
        # ignora o disparo programático durante o _refresh (a árvore/IV/histograma
        # já são renderizados lá) — evita renderização dupla por mutação.
        if self._suspend_leaf_obs:
            return
        self._sync_leaf_name_field()          # apelido da folha recém-selecionada
        self._sync_move_cut_field()           # corte vigente da folha recém-selecionada
        # trocar a folha NÃO altera a estrutura: a árvore HTML é a mesma. O realce
        # é aplicado por CSS (data-leaf) → só atualizamos o <style> minúsculo, sem
        # remontar nem reenviar a árvore inteira pelo comm.
        self.tree_sel_style.value = self._leaf_highlight_style()
        self._set_html(self.leaf_header, "header", self._leaf_header_html())
        self._set_html(self.leaf_chips, "chips", self._leaf_chips_html())
        self._refresh_iv()
        self._refresh_leaf_hist()
        self._sync_img_selection()      # espelha a folha no preview interativo (contorno+barra)
        self._sync_canvas_selection()   # e no canvas da aba "Árvore interativa"
        if self.tg_feat_iv.value:       # ordenação por IV é POR FOLHA → reordena
            self._refresh_feature_options()
        self._on_feature_change(None)   # nova folha: limpa o preview e recompõe os grupos

    def _on_tab_change(self, change):
        """Render preguiçoso das abas caras: tabela de IV (optbinning de TODAS as
        variáveis), canvas da árvore, tabela de folhas (Diagnóstico) e histograma
        da folha (Construir) só são calculados quando a respectiva aba é aberta —
        não na abertura da UI nem a cada mutação com a aba fora de vista."""
        nova = change.get("new")
        if nova == self._iv_tab_index and getattr(self, "_iv_dirty", False):
            self._compute_iv()
        if nova == self._build_tab_index and getattr(self, "_hist_dirty", False):
            self._hist_dirty = False
            self._refresh_leaf_hist()
        if nova == self._diag_tab_index and getattr(self, "_table_dirty", False):
            self._table_dirty = False
            self._refresh_table()
        if nova == self._canvas_tab_index:
            # 1ª abertura monta o widget e enquadra; nas seguintes só redesenha se
            # a árvore mudou enquanto a aba estava escondida
            primeira = not getattr(self, "_cv_rendered", False)
            if primeira or getattr(self, "_cv_dirty", False):
                self._cv_dirty = False
                self._cv_rendered = True
                if primeira and self._cv_sel is None:
                    self._cv_sel = self.dd_leaf.value or "root"
                # abre CENTRALIZADO no nó em foco, num zoom legível — e não
                # enquadrando tudo: numa árvore larga o enquadramento inicial
                # deixaria os cartões pequenos demais para ler (o botão
                # "Enquadrar" continua ali para a visão geral)
                self._refresh_canvas(center=primeira)
            else:
                # nada a redesenhar, mas mutações feitas noutra aba mudam o que o
                # desfazer/refazer da barra pode fazer
                self._sync_cv_auto_cfg()

    def _refresh_iv(self):
        # só calcula se a aba de variáveis estiver à vista; senão marca pendente e
        # mostra um placeholder (o cálculo roda quando a aba for aberta).
        if getattr(self, "tabs", None) is not None and \
                self.tabs.selected_index != self._iv_tab_index:
            self._iv_dirty = True
            self._set_html(self.out_iv, "iv",
                           "<div style='font-size:12px;color:var(--sub-ink)'>Clique em "
                           "<b>Atualizar</b> (acima) para calcular o IV/PSI por variável "
                           "da folha selecionada — ou abra a aba <b>Análise de variáveis</b>.</div>")
            return
        self._compute_iv()

    def _compute_iv(self):
        self._iv_dirty = False
        sid = self.dd_leaf.value
        iv = self.seg.variable_iv(sid)
        # aproveita o cálculo p/ o "ordenar por IV" dos seletores (mesma folha)
        self._iv_sort_cache[(sid if sid in self.seg.segments else "root",
                             self.seg._tree_version)] = dict(zip(iv["variavel"], iv["iv"]))
        pd_med = iv.attrs.get("valor_medio")
        has_psi = "pior_psi" in iv.columns
        disp = (iv[["variavel", "n_bins", "iv", "forca"]].copy()
                .rename(columns={"n_bins": "bins"}))
        psi_cols = []                       # colunas de PSI exibidas (p/ formato + estilo)
        if has_psi:
            # uma coluna de PSI por amostra de validação (OOT, ESTABILIDADE, …) — todas
            # as não-referência —, mais o pior caso. Ordem: OOT, depois estabilidade,
            # depois o resto, para a leitura priorizar a estabilidade fora do tempo.
            sample_cols = [c for c in iv.columns
                           if c.startswith("psi_") and c != "psi_classificacao"]

            def _rank(c):
                nome = c[4:].upper()
                return (0 if "OOT" in nome else 1 if "ESTAB" in nome else 2, nome)

            for c in sorted(sample_cols, key=_rank):
                amostra = c[4:]
                rotulo = "psi " + ("ESTAB" if amostra == "ESTABILIDADE" else amostra)
                disp[rotulo] = iv[c].values
                psi_cols.append(rotulo)
            disp["psi pior"] = iv["pior_psi"].values
            psi_cols.append("psi pior")
            disp["psi_status"] = iv["psi_classificacao"].values
        # variáveis que ENTRARAM na árvore (selecionadas p/ o modelo) — realçadas
        try:
            _used = set(self.seg.regua_features())
        except Exception:
            _used = set()
        _used_idx = {i for i, v in enumerate(disp["variavel"].tolist()) if v in _used}
        disp["variavel"] = disp["variavel"].map(
            lambda v: self.seg.feature_labels.get(v, v))
        if len(disp):
            disp.loc[0, "variavel"] = "★ " + str(disp.loc[0, "variavel"])
        disp = disp.rename(columns={"variavel": "variável", "forca": "força",
                                    "psi_status": "estab."})

        # estilo editorial: sem grade vertical, só régua de cabeçalho + filetes
        # horizontais; força/PSI como TEXTO colorido (sem preenchimentos).
        iv_styles = [
            # min-width:max-content faz a tabela manter a largura natural das colunas
            # (não comprime): quando há muitas amostras de PSI ela transborda o card e
            # o scroller (iv_scroll) permite deslizar para a direita.
            {"selector": "", "props": [("border-collapse", "collapse"),
                                       ("width", "100%"),
                                       ("min-width", "max-content")]},
            {"selector": "th, td", "props": [("padding", "7px 12px"),
                                             ("border", "none"),
                                             ("border-bottom", "1px solid var(--hair)"),
                                             ("white-space", "nowrap"),
                                             ("text-align", "right")]},
            {"selector": "thead th", "props": [("text-transform", "uppercase"),
                                               ("font-size", "10px"),
                                               ("letter-spacing", ".06em"),
                                               ("color", "var(--sub-ink)"),
                                               ("font-weight", "600"),
                                               ("padding-bottom", "6px"),
                                               ("border-bottom", "1.5px solid var(--tbl-head-line)")]},
            {"selector": "thead th:first-child", "props": [("text-align", "left")]},
            {"selector": "tbody td:first-child", "props": [("text-align", "left")]},
            {"selector": "tbody tr:hover td", "props": [("background-color", "var(--tbl-hover)")]},
            {"selector": "tbody tr:last-child td", "props": [("border-bottom", "none")]},
        ]

        def forca_txt(v):
            return {
                "forte": "color:var(--ok-tx);font-weight:600",
                "médio": "color:var(--warn-tx);font-weight:600",
                "suspeito": "color:var(--sus-tx);font-weight:600",
            }.get(v, "color:var(--sub-ink)")

        def psi_txt(v):
            if pd.isna(v):
                return "color:var(--sub-ink)"
            a = abs(v)
            c = ("var(--ok-tx)" if a < 0.10
                 else "var(--warn-tx)" if a < 0.25 else "var(--bad-tx)")
            return f"color:{c};font-weight:600"

        def estab_txt(v):
            return {
                "estável": "color:var(--ok-tx)",
                "atenção": "color:var(--warn-tx);font-weight:600",
                "instável": "color:var(--bad-tx);font-weight:600",
            }.get(v, "color:var(--sub-ink)")

        def reco_row(r):
            # DESTAQUE: variável que ENTROU na árvore (selecionada) = fundo verde +
            # acento; a variável recomendada (★, maior IV) = acento discreto. A seleção
            # (verde) tem precedência visual sobre a recomendação.
            used = r.name in _used_idx
            top = (r.name == 0)
            if not (used or top):
                return [""] * len(r)
            bg = "var(--ok-bg)" if used else "var(--tbl-zebra)"
            css = [f"background-color:{bg}"] * len(r)
            css[0] = (f"background-color:{bg};border-left:3px solid var(--ok-tx);"
                      "font-weight:600" if used
                      else f"background-color:{bg};border-left:3px solid var(--ac);"
                           "font-weight:600;color:var(--ac-deep)")
            return css

        fmt = {"iv": "{:.4f}",
               "bins": lambda v: "—" if (pd.isna(v) or v == 0) else f"{int(v)}"}
        for c in psi_cols:
            fmt[c] = "{:.4f}"
        num_cols = [c for c in (["bins", "iv"] + psi_cols) if c in disp.columns]
        sty = (disp.style.format(fmt, na_rep="—")
               .hide(axis="index")
               .set_table_styles(iv_styles)
               .set_properties(**{"font-size": "12px", "color": "var(--body-ink)"}))
        if num_cols:
            sty = sty.set_properties(subset=num_cols, **{
                "font-family": "'IBM Plex Mono', ui-monospace, monospace",
                "font-variant-numeric": "tabular-nums"})
        if len(disp):
            sty = sty.apply(reco_row, axis=1)
        sty = sty.map(forca_txt, subset=["força"])
        if has_psi:
            if psi_cols:
                sty = sty.map(psi_txt, subset=psi_cols)
            sty = sty.map(estab_txt, subset=["estab."])
        qual = "TODA A CARTEIRA" if (sid in (None, "root")) else self._leaf_label(sid)
        _iv_kind = "binário" if self._is_clf else "contínuo"
        hint = (f"<div style='font-size:11px;color:var(--muted);margin-bottom:4px'>folha: "
                f"<b>{qual}</b> · {self._risk_mean} (DES) = {pd_med} · IV {_iv_kind} (optbinning)"
                + (" · PSI por amostra de validação (OOT, ESTAB, …) e pior caso, "
                   "nos mesmos bins do IV (DES × amostra)" if has_psi else "")
                + ("  ·  <b style='color:var(--ok-tx)'>verde</b> = variável que entrou "
                   "na árvore" if _used_idx else "")
                + "</div>")
        self._set_html(self.out_iv, "iv", hint + self._styler_html(sty))

    def _refresh_leaf_hist(self):
        """Alvo da folha selecionada (DES): taxa de default + IC de Wilson
        (classificação) ou histograma do alvo (regressão).

        O card mora na aba Construir e desenha uma figura matplotlib (~0.2s);
        com a aba fora de vista (ex.: trabalhando no mapa), só marca pendente."""
        if getattr(self, "tabs", None) is not None and \
                self.tabs.selected_index != self._build_tab_index:
            self._hist_dirty = True
            return
        sid = self.dd_leaf.value
        if sid is None or sid not in self.seg.segments:
            self._set_html(self.out_leaf_hist, "leaf_hist",
                           "<div style='font-size:11px;color:var(--sub-ink)'>—</div>")
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
                html = self._fig_html(plot(sid, figsize=self._PREVIEW_FIGSIZE),
                                      full_width=True)
            except Exception as e:
                html = (f"<div style='font-size:11px;color:var(--bad-tx)'>"
                        f"(gráfico não gerado: {type(e).__name__})</div>")
            if len(self._leaf_hist_cache) > 256:      # backstop de memória
                self._leaf_hist_cache.clear()
            self._leaf_hist_cache[ck] = html
        self._set_html(self.out_leaf_hist, "leaf_hist", html)

    # ==================================================================
    # Aba "Análise de variáveis"
    # ==================================================================
    def _var_cards_html(self, s, trend):
        import html as _html                 # escapa nomes de categoria vindos dos dados
        psi_hex = {"green": "var(--ok-tx)", "yellow": "var(--warn-tx)", "red": "var(--bad-tx)"}
        tipo = s.get("tipo")

        def chip(k, v, sub="", vcolor=None):
            sty = f" style='color:{vcolor}'" if vcolor else ""
            subh = (f"<div style='font-size:10px;color:var(--sub-ink);margin-top:2px;"
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
                f"padding:3px 0;border-top:1px solid var(--hair)'>"
                f"<span>{_html.escape(str(c))}</span>"
                f"<span class='mono'>{p:.1f}%</span></div>"
                for c, p in s["top_categorias"][:8])
            html += ("<div class='treeui-metric' style='margin-top:6px;padding:8px 11px'>"
                     "<div class='k'>Categorias (share)</div>" + linhas + "</div>")

        # exclui None E NaN (v == v é falso p/ NaN): um PSI NaN geraria
        # left:calc(nan%) (marcador some) e _psi_class(nan) cairia em "instável".
        psi = {a: v for a, v in (s.get("psi") or {}).items() if v is not None and v == v}
        if psi:
            def gauge(p):
                pos = min(max(p, 0.0) / 0.50, 1.0) * 100
                return ("<div style='position:relative;flex:1;height:8px;border-radius:5px;"
                        "background:linear-gradient(to right,var(--gauge-ok) 0%,var(--gauge-ok) 20%,"
                        "var(--gauge-warn) 20%,var(--gauge-warn) 50%,"
                        "var(--gauge-bad) 50%,var(--gauge-bad) 100%)'>"
                        f"<div style='position:absolute;left:calc({pos:.1f}% - 1px);top:-2px;"
                        "width:2px;height:12px;background:var(--strong-ink);border-radius:1px'></div></div>")
            rows = ""
            for a, v in psi.items():
                ab = "ESTAB" if a == "ESTABILIDADE" else a
                cls = self._psi_class(v)
                txt = {"green": "estável", "yellow": "atenção", "red": "instável"}[cls]
                rows += ("<div style='display:flex;align-items:center;gap:9px;margin:6px 0'>"
                         f"<div style='width:74px;font-size:11.5px;color:var(--muted);"
                         f"white-space:nowrap'>PSI {ab}</div>"
                         f"<div class='mono' style='width:50px;font-size:13px;font-weight:600;"
                         f"color:{psi_hex[cls]}'>{v:.3f}</div>{gauge(v)}"
                         f"<div style='width:54px;text-align:right;font-size:10.5px;"
                         f"color:{psi_hex[cls]}'>{txt}</div></div>")
            legend = ("<div style='font-size:10px;color:var(--sub-ink);margin-top:4px'>"
                      "<span style='color:var(--gauge-ok)'>■</span> &lt;0,10 estável &nbsp;"
                      "<span style='color:var(--gauge-warn)'>■</span> 0,10–0,25 atenção &nbsp;"
                      "<span style='color:var(--gauge-bad)'>■</span> &gt;0,25 instável</div>")
            html += ("<div class='treeui-h' style='margin-top:13px'>Estabilidade · PSI por "
                     "amostra (vs. DES)</div>" + rows + legend)

        if trend:
            arrow = "↑" if trend["pct"] >= 0 else "↓"
            tc = ("var(--bad-tx)" if abs(trend["pct"]) >= 10
                  else "var(--warn-tx)" if abs(trend["pct"]) >= 3 else "var(--ok-tx)")
            html += ("<div style='display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;"
                     "font-size:11.5px;margin-top:12px;padding-top:8px;"
                     "border-top:1px solid var(--hair)'>"
                     "<span style='color:var(--muted)'>Tendência da média</span>"
                     f"<b style='color:{tc};font-size:13px'>{arrow} {trend['pct']:+.0f}%</b>"
                     f"<span style='color:var(--sub-ink)'>{trend['de']:.2f} → {trend['para']:.2f} · "
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
                **{"font-weight": "700", "color": "var(--tbl-head-ink)",
                   "background-color": "var(--tbl-head-bg)",
                   "border-left": "1px solid var(--tbl-line-strong)",
                   "border-right": "1px solid var(--tbl-line-strong)"})

        if "pct_missing" in cols:
            def _sev(s):
                out = []
                for v in s:
                    if pd.isna(v):
                        out.append("color:var(--muted)")
                    elif v >= 20:
                        out.append("color:var(--bad-tx);font-weight:600")
                    elif v > 0:
                        out.append("color:var(--warn-tx);font-weight:600")
                    else:
                        out.append("color:var(--ok-tx)")
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
                            out.append("color:var(--muted)")
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
                sty = sty.set_properties(subset=[special], **{"color": "var(--muted)"})

        extra = list(self._TABLE_STYLES) + list(self._SAFRA_HEADER_STYLES)
        extra.append({"selector": "th, td", "props": [("padding", "5px 10px")]})
        sty = sty.set_table_styles(extra)

        sty = sty.relabel_index(list(cols), axis=1)
        return sty

    def _on_var_analyze(self, _):
        feat = self._sel_var(warn=False)
        sid = self.dd_var_leaf.value
        tcol = self.tx_var_time.value.strip()
        for o in (self.out_var_dist, self.out_var_time,
                  self.out_var_psi, self.out_var_inv_s,
                  self.out_var_inv_t, self.out_var_optbin):
            o.value = ""                       # HTML widgets: limpa via .value
        self.out_var_cards.value = ""
        bs, trend = None, None
        if feat is None:
            txt = (self.dd_var.value or "").strip()
            self._log(f"⚠ Variável '{txt}' não reconhecida — escolha uma opção da "
                      "lista." if txt
                      else "Selecione uma variável para analisar.")
            return
        with self._busy(self.btn_var_analyze, msg="analisando a variável…"):
            try:
                summ = self.seg.variable_summary(feat, sid=sid)
            except Exception as e:
                self._log(f"Erro no resumo da variável: {type(e).__name__}: {e}"); return
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
                    self._log(f"(percentis por safra: {type(e).__name__}: {e})")
            lbl = self.seg.feature_labels.get(feat, feat)
            self._log(f"Análise de '{lbl}' concluída"
                      + (f" · folha {self._leaf_label(sid)}" if sid not in (None, 'root') else "")
                      + ".")
            self._render_var_analysis(feat, sid, tcol, summ, trend)

    def _render_var_analysis(self, feat, sid, tcol, summ, trend):
        """Renderiza os cards/gráficos da aba 'Análise de variável' a partir do
        resumo já calculado (separado do handler p/ o corpo caber sob o _busy)."""
        kind = summ.get("tipo")

        def err(what, e):
            return (f"<div style='font-size:11px;color:var(--bad-tx)'>({what} não gerada: "
                    f"{type(e).__name__})</div>")
        # Resumo & estabilidade
        self.out_var_cards.value = self._var_cards_html(summ, trend)
        # Comportamento: distribuição & risco (logodds/WoE e tabela por faixa
        # removidos a pedido)
        try:
            # figura mais alta + full_width → o gráfico preenche a altura da box
            # (ao lado do resumo & estabilidade), sem sobrar espaço em branco
            self.out_var_dist.value = self._fig_html(
                self.seg.plot_variable_distribution_badrate(feat, sid=sid, figsize=(8.6, 5.4)),
                full_width=True)
        except Exception as e:
            self.out_var_dist.value = err("distribuição & risco", e)
        # Inversão da ordem de risco · por amostra
        if self.sample_col is not None:
            try:
                self.out_var_inv_s.value = self._fig_html(
                    self.seg.plot_variable_inversion_by_sample(feat, sid=sid),
                    full_width=True)
            except Exception as e:
                self.out_var_inv_s.value = err("inversão por amostra", e)
        else:
            self.out_var_inv_s.value = ("<div style='font-size:12px;color:var(--sub-ink)'>"
                                        "inversão por amostra requer amostras (DES/OOT).</div>")
        # Ao longo do tempo (percentis · PSI · inversão por safra · optbin)
        if tcol and tcol in self.df.columns:
            # percentis/share (esq) e PSI (dir) com o MESMO figsize (8.6×4.2) +
            # tight=False → PNGs de dimensão IDÊNTICA, que em colunas iguais (49%/49%,
            # abaixo) ficam exatamente do mesmo tamanho e alinhados. A legenda do
            # share fica DENTRO do figsize (_legend_below), então tight=False não corta.
            _fs_par = (8.6, 4.2)
            try:
                self.out_var_time.value = self._fig_html(
                    self.seg.plot_variable_timeseries(feat, tcol, sid=sid, figsize=_fs_par),
                    full_width=True, tight=False)
            except Exception as e:
                self.out_var_time.value = err("série temporal", e)
            try:
                if self.sample_col is not None:
                    self.out_var_psi.value = self._fig_html(
                        self.seg.plot_variable_psi_by_safra(feat, tcol, sid=sid, figsize=_fs_par),
                        full_width=True, tight=False)
                else:
                    self.out_var_psi.value = ("<div style='font-size:12px;color:var(--sub-ink)'>"
                                              "PSI por safra requer amostras (DES/OOT).</div>")
            except Exception as e:
                self.out_var_psi.value = err("PSI por safra", e)
            try:
                self.out_var_inv_t.value = self._fig_html(
                    self.seg.plot_variable_inversion_by_safra(feat, sid=sid, time_col=tcol),
                    full_width=True)
            except Exception as e:
                self.out_var_inv_t.value = err("inversão por safra", e)
            if kind == "num":
                try:
                    self.out_var_optbin.value = self._fig_html(
                        self.seg.plot_variable_optbin_cumshare_timeseries(feat, sid=sid, time_col=tcol),
                        full_width=True)
                except Exception as e:
                    self.out_var_optbin.value = err("optbin por safra", e)
            else:
                self.out_var_optbin.value = ("<div style='font-size:12px;color:var(--sub-ink)'>"
                                             "optbin ao longo do tempo é só para variáveis "
                                             "numéricas.</div>")
        else:
            _need = ("<div style='font-size:12px;color:var(--sub-ink)'>Informe a <b>coluna de "
                     "safra</b> (ex.: dt_ref) acima para as análises ao longo do tempo "
                     "(percentis, PSI, inversão por safra e optbin).</div>")
            self.out_var_time.value = _need
            self.out_var_psi.value = _need
            self.out_var_inv_t.value = _need
            self.out_var_optbin.value = _need

    def _prepare_split(self):
        """Monta self._pending a partir dos controles atuais. Valida via show_grow."""
        import contextlib
        import io
        sid = self._selected_leaf()
        if sid is None:
            return False, "Nenhuma folha selecionada."
        if sid in self.locked:
            return False, "⚠ Folha fechada — reabra (🔓) para dividir."
        feature = self._sel_feature(warn=False)
        if feature is None:
            return False, (f"⚠ Variável '{(self.dd_feature.value or '').strip()}' não "
                           "reconhecida — escolha uma opção da lista.")
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
        feat = self._sel_feature()          # inválida → aviso no console
        if feat is None:
            return
        if sid is None or sid not in self.seg.segments:
            self._log("Selecione uma folha."); return
        with self._busy(self.btn_sugcuts, msg="sugerindo os cortes…"):
            try:
                r = self.seg.best_binning(sid, feat, max_n_bins=int(self.sl_bins.max))
            except Exception as e:
                self._log(f"Não consegui sugerir cortes: {type(e).__name__}: {e}"); return
            lbl = self.seg.feature_labels.get(feat, feat)
            if r["n_bins"] < 2:
                self._log(f"Sem corte ótimo para '{lbl}' nesta folha "
                          "(variável pouco informativa aqui)."); return
            self.sl_bins.value = max(self.sl_bins.min, min(self.sl_bins.max, r["n_bins"]))
            if r["kind"] == "num":
                cuts = ", ".join(f"{c:.4g}" for c in r["cuts"])
                self.tx_cuts.value = cuts
                self._log(f"Sugestão p/ '{lbl}': {r['n_bins']} bins · cortes: {cuts}. "
                          "Em 'Ótimo' o máx. bins já foi ajustado; em 'Manual' os cortes foram "
                          "preenchidos. Clique em 👁 Preview.")
            else:
                grupos = " | ".join("{" + ", ".join(g) + "}" for g in r["groups"])
                self._log(f"Sugestão p/ '{lbl}' (categórica): {r['n_bins']} grupos: {grupos}. "
                          "No modo Ótimo o máx. bins já foi ajustado; clique em 👁 Preview.")

    def _on_preview(self, _):
        self.out_preview_seg.value = ""
        self.out_preview_chart.value = ""
        with self._busy(self.btn_preview, msg="gerando o preview…"):
            ok, msg = self._prepare_split()
            if not ok:
                self._log(msg); return
            feature = self._pending["feature"]
            kind = self._feature_kind()
            graf = ("segmentação (em Dividir) + distribuição/cortes (ao lado do histograma)"
                    if kind == "num" else "segmentação (em Dividir)")
            self._log(f"Preview de '{self.seg.feature_labels.get(feature, feature)}' "
                      f"({graf}) — revise os gráficos e clique em ✂ Criar segmento.")
            p = self._pending
            sid = p["only_segments"][0]
            splits = p.get("splits")
            mnb, mbs, xbs = p.get("max_n_bins", 4), p.get("min_bin_size", 0.05), p.get("max_bin_size")
            mmd = p.get("min_mean_diff", 0.0)
            # SEGMENTAÇÃO PROPOSTA (barras repr. × alvo por faixa) — dentro de "Dividir".
            try:
                # full_width: o PNG tem tamanho fixo (figsize×dpi); sem esticar, ele
                # sobra em monitor largo e encolhe em monitor estreito. Com
                # width:100% + height:auto ele acompanha a largura do card.
                self.out_preview_seg.value = self._fig_html(self.seg.plot_feature_value(
                    p["feature"], sid=sid, splits=splits, max_n_bins=mnb,
                    min_bin_size=mbs, max_bin_size=xbs, min_mean_diff=mmd),
                    full_width=True)
            except Exception as e:
                self.out_preview_seg.value = (f"<div style='color:var(--bad-tx);font-size:11px'>"
                                              f"(segmentação não gerada: {type(e).__name__})</div>")
            # DISTRIBUIÇÃO DA VARIÁVEL + cortes sugeridos — ao lado do histograma.
            if self._feature_kind() == "num":
                try:
                    self.out_preview_chart.value = self._fig_html(self.seg.plot_feature_hist(
                        p["feature"], sid=sid, splits=splits, max_n_bins=max(mnb, 6),
                        min_bin_size=mbs, max_bin_size=xbs, min_mean_diff=mmd,
                        figsize=self._PREVIEW_FIGSIZE), full_width=True)
                except Exception as e:
                    self.out_preview_chart.value = (f"<div style='color:var(--bad-tx);font-size:11px'>"
                                                    f"(distribuição não gerada: {type(e).__name__})</div>")
            else:
                self.out_preview_chart.value = (
                    "<div style='font-size:11px;color:var(--sub-ink)'>variável categórica — sem histograma "
                    "de distribuição; veja a segmentação no card <b>Dividir a folha</b>.</div>")

    def _on_split(self, _):
        with self._busy(self.btn_split, msg="criando o segmento…"):
            if self._pending is None:          # sem Preview: prepara a partir dos controles
                ok, msg = self._prepare_split()
                if not ok:
                    self._log(msg); return
            redo_bak = list(self._redo)
            antes = self._delta_snapshot()
            self._checkpoint()
            try:
                self.seg.grow(**self._pending)
                self._pending = None
            except Exception as e:
                self._revert_checkpoint(redo_bak)
                self._log(f"Erro ao criar segmento: {type(e).__name__}: {e}"); return
            self._refresh()
            self._log_delta("dividir", antes)

    def _on_lock(self, _):
        sid = self._selected_leaf()
        if sid is not None:
            self.locked.add(sid)
            self._log(f"🔒 fechada: {self._leaf_label(sid)}")
            # lock só muda o rótulo 🔒: atualiza árvore/dropdowns, NÃO o _refresh
            # completo (IV/PSI/metrics/tabela/PNG são idênticos após travar).
            self._refresh_lock_labels()

    def _on_unlock(self, _):
        sid = self._selected_leaf()
        if sid in self.locked:
            self.locked.discard(sid)
            self._log(f"🔓 reaberta: {self._leaf_label(sid)}")
            self._refresh_lock_labels()

    def _on_prune(self, _):
        with self._busy(self.btn_prune, self.btn_cv_prune, msg="podando a árvore…"):
            redo_bak = list(self._redo)
            antes = self._delta_snapshot()
            self._checkpoint()
            try:
                self.seg.prune(min_repr=self.sl_repr.value, min_valor_gap=self.sl_gap.value,
                               protect=set(self.locked))
            except Exception as e:
                self._revert_checkpoint(redo_bak)
                self._log(f"Erro na poda: {type(e).__name__}: {e}"); return
            self.locked &= set(self.seg.segments)
            self._refresh()
            self._log_delta("poda", antes)

    def _on_reset(self, _):
        with self._busy(self.btn_reset, self.btn_img_reset,
                        msg="reiniciando a árvore…"):
            antes = self._delta_snapshot()
            self._checkpoint()
            self.seg = TreeSegmenter(self.df, **self._kwargs)
            # o segmentador novo nasce sem fallback — reaplica a escolha da UI
            self.seg.fallback = self.dd_fallback.value
            self.locked.clear()
            self._pending = None
            self._log("Árvore reiniciada.")
            self._refresh()
            self._log_delta("resetar", antes)

    def _on_export(self, _):
        # chamamos de "folha" na UI (não "nota"): renomeia as colunas de nota do assign
        self.result = self.seg.assign("segmento").rename(
            columns={"segmento_nota": "folha", "segmento_desc": "folha_desc"})
        self._log(f"DataFrame rotulado em  ui.result  · shape {self.result.shape}")
        try:
            with self.out_log:                  # a tabela vai para o console
                display(self.result["folha"].value_counts().sort_index())
        except Exception as e:
            self._log(f"(distribuição de folhas indisponível: {e})")

    def _boot_forest_html(self, bc):
        """Forest plot: barra de IC por folha + marcador do ponto (DES) e do alvo OOT."""
        import html as _html                 # escapa descrições (categorias) vindas dos dados
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
            return "<div style='color:var(--sub-ink)'>sem dados para o gráfico</div>"
        xmin, xmax = min(vals), max(vals)
        pad = (xmax - xmin) * 0.08 or 0.02
        # clamp em [0,1] só na CLASSIFICAÇÃO (alvo ∈ [0,1]); na regressão (alvo pode ser
        # <0 ou >1) usar os próprios min/max com padding, senão pos(v) estoura as barras.
        if self._is_clf:
            xmin, xmax = max(0, xmin - pad), min(1, xmax + pad)
        else:
            xmin, xmax = xmin - pad, xmax + pad
        span = (xmax - xmin) or 1.0

        def pos(v):
            return 100 * (v - xmin) / span

        rows = ["<div style='font-family:ui-monospace,Menlo,monospace;font-size:11px'>"]
        for _, r in bc.iterrows():
            if pd.isna(r[lo_col]):
                continue
            x0, x1, xp = pos(r[lo_col]), pos(r[hi_col]), pos(r[ref_col])
            bar = (f"<div style='position:absolute;left:{x0:.1f}%;width:{max(0.5,x1-x0):.1f}%;"
                   f"top:8px;height:4px;background:var(--ci-bar);border-radius:2px'></div>"
                   f"<div style='position:absolute;left:{xp:.1f}%;top:4px;width:2px;height:12px;"
                   f"background:var(--ci-ref)' title='DES'></div>")
            ootmark = ""
            if chk and not pd.isna(r.get(f"valor_{chk}", float("nan"))):
                xo = pos(r[f"valor_{chk}"])
                inside = r.get("aderente")
                col = "var(--risk-lo)" if inside else "var(--risk-hi)"
                ootmark = (f"<div style='position:absolute;left:{xo:.1f}%;top:3px;width:10px;"
                           f"height:10px;background:{col};border:1.5px solid var(--tile-bg);border-radius:50%;"
                           f"transform:translateX(-4px)' title='{chk}'></div>")
            _desc = str(r["descricao"])
            label = _html.escape(_desc[:40] + "…" if len(_desc) > 40 else _desc)
            rows.append(
                f"<div style='display:flex;align-items:center;margin:3px 0'>"
                f"<div style='width:34px;color:var(--body-ink)'>[{r['nota']}]</div>"
                f"<div style='width:300px;color:var(--ink);white-space:nowrap;overflow:hidden;"
                f"text-overflow:ellipsis'>{label}</div>"
                f"<div style='position:relative;flex:1;height:20px;background:var(--gauge-track);"
                f"border-radius:3px'>{bar}{ootmark}</div></div>")
        leg = (f"<div style='font-size:10.5px;color:var(--muted);margin-top:5px'>"
               f"barra cinza = IC {int(bc.attrs.get('ci',0.95)*100)}% (DES) · "
               f"traço azul = {self._risk_label} {ref} · ")
        if chk:
            leg += (f"círculo = {self._risk_label} {chk} (<span style='color:var(--risk-lo)'>verde dentro</span> / "
                    f"<span style='color:var(--risk-hi)'>vermelho fora</span>)")
        leg += "</div>"
        rows.append(leg + "</div>")
        return "".join(rows)

    def _on_boot(self, _):
        with self._busy(self.btn_boot, status=self.out_boot,
                        msg="rodando o bootstrap…"):
            self._do_boot()

    def _do_boot(self):
        try:
            bc = self.seg.bootstrap_ci(n_boot=int(self.sl_boot.value))
        except Exception as e:
            self.out_boot.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro no "
                                   f"bootstrap: {type(e).__name__}: {e}</div>")
            return

        def status_bg(v):
            if v == "dentro":
                return "background-color:var(--ok-bg);color:var(--ok-tx);font-weight:600"
            if v in ("acima", "abaixo"):
                return "background-color:var(--bad-bg);color:var(--bad-tx);font-weight:600"
            return "color:var(--faint-ink)"
        # títulos mais claros + "nota" → "folha" + tabela CENTRALIZADA
        smp = bc.attrs.get("sample") or "todos"
        chk = bc.attrs.get("check_sample")
        rename = {"nota": "folha", "descricao": "descrição", "n": "volume",
                  "ic_low": "IC 95% inf.", "ic_high": "IC 95% sup.",
                  "amplitude": "amplitude do IC", "aderente": "aderente?",
                  f"valor_{smp}": f"média ({smp})"}
        if chk:
            rename[f"valor_{chk}"] = f"média ({chk})"
            rename["status_oot"] = f"status ({chk})"
        disp = bc.rename(columns=rename)
        status_col = rename.get("status_oot")
        fmt = {c: "{:.4f}" for c in disp.columns if disp[c].dtype.kind == "f"}
        sty = (disp.style.format(fmt, na_rep="—").hide(axis="index")
               .set_properties(**{"font-size": "12px", "text-align": "center"})
               .set_table_styles([{"selector": "th", "props": [("text-align", "center")]},
                                  {"selector": "", "props": [("margin", "0 auto")]}],
                                 overwrite=False))
        if status_col and status_col in disp.columns:
            sty = sty.map(status_bg, subset=[status_col])
        resumo = ""
        if "aderente" in bc.columns:
            n_ok = int((bc["aderente"] == True).sum())
            n_tot = int(bc["aderente"].notna().sum())
            resumo = (f"<div style='font-size:12px;color:var(--strong-ink);margin:6px 0'>Aderência "
                      f"<b>{chk}</b>: {n_ok}/{n_tot} folhas com {self._risk_label} dentro do IC bootstrap "
                      f"(n_boot={bc.attrs.get('n_boot')}).</div>")
        self.out_boot.value = (self._boot_forest_html(bc) + resumo
                               + f"<div style='text-align:center'>{self._styler_html(sty)}</div>")

    # ==================================================================
    # Diagnóstico — placar de saúde do modelo (4 vereditos)
    # ==================================================================
    def _on_diag(self, _):
        with self._busy(self.btn_diag, msg="avaliando o modelo…"):
            try:
                html = self._diag_scorecard_html()
            except Exception as e:
                self.out_diag.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro ao "
                                       f"avaliar o modelo: {type(e).__name__}: {e}</div>")
                self._log(f"Erro no placar: {type(e).__name__}: {e}"); return
            self._log("Placar de saúde do modelo calculado.")
            self.out_diag.value = html

    def _on_diag_hide(self, _):
        self.out_diag.value = ""    # oculta/limpa a avaliação já renderizada

    def _diag_scorecard_html(self):
        """Placar de 4 vereditos (Discriminação · Estabilidade · Calibração ·
        Estrutura) + evidência estatística — reúne os testes das outras abas.
        No alvo a discriminação usa AUC/Gini/KS (alvo binário)."""
        psi_hex = {"green": "var(--ok-tx)", "yellow": "var(--warn-tx)", "red": "var(--bad-tx)"}
        bgc = {"green": "var(--ok-bg)", "yellow": "var(--warn-bg)", "red": "var(--bad-bg)"}
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
                g = f" · Gini {gini:.1%}" if (gini is not None and gini == gini) else ""
                return c, f"AUC DES {auc:.1%}{g}"
            if r2 is None or r2 != r2:
                return "yellow", "—"
            c = "green" if r2 >= 0.5 else "yellow" if r2 >= 0.2 else "red"
            return c, f"R² DES {r2:.1%}"

        def v_estab():
            if pior_psi is None:
                return "yellow", "sem amostras"
            return self._psi_class(pior_psi), f"pior PSI {pior_psi:.1%}"

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

        _rl = self._risk_label                       # "alvo" (clf) ou "alvo" (reg)
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
                    f"<div style='font-size:10px;color:var(--muted);margin-top:3px'>{q}</div></div>")
        scorecard = ("<div class='treeui-metrics' style='grid-template-columns:"
                     "repeat(4,minmax(0,1fr))'>"
                     + "".join(light(*d) for d in dims) + "</div>")
        ev = ""
        if psi_df is not None and len(psi_df):
            def bar(p):
                pos = min(max(p, 0.0) / 0.50, 1.0) * 100
                return ("<div style='position:relative;flex:1;height:8px;border-radius:5px;"
                        "background:linear-gradient(to right,var(--gauge-ok) 0%,var(--gauge-ok) 20%,"
                        "var(--gauge-warn) 20%,var(--gauge-warn) 50%,"
                        "var(--gauge-bad) 50%,var(--gauge-bad) 100%)'>"
                        f"<div style='position:absolute;left:calc({pos:.1f}% - 1px);top:-2px;"
                        "width:2px;height:12px;background:var(--strong-ink);border-radius:1px'></div></div>")
            rows = ""
            for _, r in psi_df.iterrows():
                a = r["amostra"]; ab = "ESTAB" if a == "ESTABILIDADE" else a
                p = float(r["psi"]); cls = self._psi_class(p)
                rows += ("<div style='display:flex;align-items:center;gap:9px;margin:5px 0'>"
                         f"<div style='width:80px;font-size:11px;color:var(--muted)'>PSI {ab}</div>"
                         f"<div class='mono' style='width:52px;font-size:12.5px;font-weight:600;"
                         f"color:{psi_hex[cls]}'>{p:.1%}</div>{bar(p)}"
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
        # estrutura: monotonicidade — MESMA leitura visual das FOLHAS-IRMÃS,
        # comparando SÓ as folhas que invertem no alvo (não só as sob mesmo pai).
        # Mapeia os pares de NOTAS de mono["inversoes"] para os sids das folhas.
        lv_m = self.seg.leaves()
        nota2sid = dict(zip(lv_m["nota"], lv_m["segmento"]))
        notas_inv = set()
        for inv in mono["inversoes"]:
            for a, b in inv:
                notas_inv.update((int(a), int(b)))
        leaves_inv = [nota2sid[n] for n in sorted(notas_inv) if n in nota2sid]
        ev += (f"<div class='treeui-h' style='margin-top:14px'>Estrutura · monotonicidade do "
               f"{_rl} por amostra</div>")
        if not leaves_inv:
            ev += ("<div class='treeui-legend'>✅ Sem inversões de monotonicidade — o "
                   f"{_rl} cresce com a folha (na ordem de risco) em todas as amostras.</div>")
        else:
            ev += (f"<div class='treeui-legend'><b>{len(leaves_inv)} folha(s)</b> envolvidas em "
                   f"inversão da monotonicidade. Mesma leitura das folhas-irmãs: cada linha é uma "
                   f"folha, ordenada pelo {_rl} na {self.ref_sample}; onde as linhas se "
                   f"<b>cruzam</b> há inversão da ordem de risco. Faixas vermelhas (por safra) = "
                   f"safras com inversão.</div>")
            try:
                tcol = (self.tx_sib_time.value or "").strip() or self.date_col
                _t1 = f"{self.seg._risk_mean} das folhas que invertem · por amostra"
                h1 = self._fig_html(self.seg.plot_sibling_value_by_sample(
                    None, leaves=leaves_inv, title=_t1), full_width=True)
                if tcol and tcol in self.df.columns:
                    _t2 = f"{self.seg._risk_mean} das folhas que invertem · por safra"
                    h2 = self._fig_html(self.seg.plot_sibling_value_by_safra(
                        None, leaves=leaves_inv, time_col=tcol, title=_t2), full_width=True)
                    ev += ("<div style='display:flex;gap:10px;align-items:flex-start'>"
                           f"<div style='flex:76 1 0;min-width:0'>{h1}</div>"
                           f"<div style='flex:96 1 0;min-width:0'>{h2}</div></div>")
                else:
                    ev += h1
            except Exception as e:
                ev += (f"<div style='color:var(--bad-tx);font-size:12px'>Gráficos de inversão "
                       f"não gerados: {type(e).__name__}</div>")
        return scorecard + ev

    # ==================================================================
    # Validação (monotonicidade · calibração · backtest) e relatório
    # ==================================================================
    def _on_validate(self, _):
        with self._busy(self.btn_validate, status=self.out_validate,
                        msg="gerando as análises de validação…"):
            self._do_validate()

    def _do_validate(self):
        # Validação SÓ em análises gráficas (sem tabelas), em grade de 2 colunas.
        cards = []                                   # [(título, html_do_gráfico)]

        def _erro(e):
            return (f"<div style='color:var(--bad-tx);font-size:12px'>Erro: "
                    f"{type(e).__name__}</div>")

        # 1) Ordenação / monotonicidade — o alvo por folha na ordem das notas
        #    (visualiza a monotonicidade). clf: taxa por folha · reg: boxplot por folha.
        status = ""
        try:
            mr = self.seg.monotonicity_report()
            ok = bool(mr["monotonico"].all())
            status = ("✅ Ordenação monotônica em todas as amostras." if ok
                      else "⚠️ Há inversões de monotonicidade — veja o gráfico de ordenação.")
            fig = (self.seg.plot_leaf_badrate(figsize=(6.4, 3.8)) if self._is_clf
                   else self.seg.plot_leaf_boxplots(figsize=(6.4, 3.8)))
            cards.append(("Ordenação / monotonicidade", self._fig_html(fig)))
        except Exception as e:
            cards.append(("Ordenação / monotonicidade", _erro(e)))

        # 2) Calibração (prevista × realizada) — precisa de amostra
        if self.sample_col is not None:
            try:
                cards.append(("Calibração · prevista × realizada",
                              self._fig_html(self.seg.plot_calibration(figsize=(5.4, 5.0)))))
            except Exception as e:
                cards.append(("Calibração · prevista × realizada", _erro(e)))

        # O backtest por safra saiu DA TELA: com muitas safras o eixo x vira um
        # borrão e os marcadores de alerta se sobrepõem — ilegível. Ele continua
        # no RELATÓRIO de validação (Markdown/PDF), onde a figura sai em tamanho
        # real, e na API (`seg.backtest()` / `seg.plot_backtest()`).

        cells = "".join(
            f"<div class='treeui-card' style='margin:0'>"
            f"<div class='treeui-h'>{t}</div>{h}</div>"
            for t, h in cards)
        grid = ("<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;"
                f"align-items:start'>{cells}</div>")
        status_html = (f"<div style='font-size:12px;margin:2px 0 8px'>{status}</div>"
                       if status else "")
        self.out_validate.value = status_html + grid

    def _on_report(self, _):
        path = self.tx_report_path.value.strip()
        if not path:
            self._log("Informe o caminho do relatório (.md)."); return
        tcol = self.tx_time_col.value.strip() or None
        if tcol and tcol not in self.df.columns:
            self._log(f"(coluna de tempo '{tcol}' inexistente — relatório sem backtest)")
            tcol = None
        with self._busy(self.btn_report, msg="gerando o relatório…"):
            try:
                out = self.seg.validation_report(path, time_col=tcol)
                self._log(f"📄 relatório de validação gerado em '{out}' (imagens salvas ao lado).")
            except Exception as e:
                self._log(f"Erro ao gerar relatório: {type(e).__name__}: {e}")

    # ==================================================================
    # Discriminação (ROC · KS) e qualidade dos segmentos
    # ==================================================================
    def _fig_html(self, fig, border=False, full_width=False, tight=True):
        """Converte uma figura matplotlib em <img> base64 (string HTML).

        ``full_width=True`` faz a imagem ESTICAR até a largura do container
        (``width:100%``) em vez de só limitar (``max-width:100%``) — elimina o
        espaço em branco à direita em cartões largos.

        ``tight=False`` salva o PNG EXATAMENTE em ``figsize×dpi`` (sem recorte por
        conteúdo, ``bbox_inches=None``): duas figuras de MESMA altura de figsize saem
        com a MESMA altura na tela em colunas lado a lado (o ``bbox_inches='tight'``
        recorta cada figura ao seu conteúdo e distorce a razão de aspecto)."""
        import base64
        import io as _io
        buf = _io.BytesIO()
        # dpi limitado a 110 nas prévias inline (export usa save_path nos plot_*):
        # corta o PNG/base64 ~40% sem perda visual perceptível, aliviando o comm.
        buf_dpi = min(int(fig.get_dpi()), 110)
        fig.savefig(buf, format="png", dpi=buf_dpi,
                    bbox_inches="tight" if tight else None)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        # fecha a figura: sem isso o pyplot retém TODA figura gerada (Gcf.figs) e a
        # RAM cresce sem limite numa sessão interativa (dezenas de plots por ação).
        import matplotlib.pyplot as _plt
        _plt.close(fig)
        style = ("width:100%;height:auto" if full_width else "max-width:100%;height:auto")
        if border:
            style += ";border:1px solid var(--line);border-radius:6px"
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
        return {"forte": "color:var(--ok-tx);font-weight:600",
                "médio": "color:var(--warn-tx);font-weight:600",
                "suspeito": "color:var(--sus-tx);font-weight:600"}.get(v, "color:var(--sub-ink)")

    @staticmethod
    def _css_estab(v):                       # estabilidade (estável/atenção/instável)
        return {"estável": "color:var(--ok-tx)",
                "atenção": "color:var(--warn-tx);font-weight:600",
                "instável": "color:var(--bad-tx);font-weight:600"}.get(v, "color:var(--sub-ink)")

    @staticmethod
    def _css_psi(v):                         # PSI numérico (verde<0.10<amarelo<0.25<vermelho)
        if pd.isna(v):
            return "color:var(--sub-ink)"
        a = abs(v)
        c = ("var(--ok-tx)" if a < 0.10
             else "var(--warn-tx)" if a < 0.25 else "var(--bad-tx)")
        return f"color:{c};font-weight:600"

    @staticmethod
    def _css_passa(v):                       # passa no teste de hipótese (✅)
        return "color:var(--ok-tx);font-weight:600" if str(v).strip() == "✅" else "color:var(--sub-ink)"

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
            self.out_discrim.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro na "
                                      f"curva ROC: {type(e).__name__}: {e}</div>")

    def _on_ks(self, _):
        try:
            self.out_discrim.value = self._fig_html(self.seg.plot_ks())
        except Exception as e:
            self.out_discrim.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro na "
                                      f"curva KS: {type(e).__name__}: {e}</div>")

    # plots de REGRESSÃO (alvo contínuo): dispersão e distribuição do alvo —
    # ambos renderizam em out_discrim (toggle no card de discriminação)
    def _on_box(self, _):
        try:
            self.out_discrim.value = self._fig_html(self.seg.plot_leaf_boxplots(), full_width=True)
        except Exception as e:
            self.out_discrim.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro no "
                                      f"boxplot: {type(e).__name__}: {e}</div>")

    def _on_hist(self, _):
        try:
            self.out_discrim.value = self._fig_html(self.seg.plot_target_hist(color="steelblue"), full_width=True)
        except Exception as e:
            self.out_discrim.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro no "
                                      f"histograma: {type(e).__name__}: {e}</div>")

    # CAP · Lift · métricas por safra (clf E reg): score = alvo previsto da folha;
    # renderizam em out_discrim (mesmo card), invalidados junto no _refresh
    def _on_cap(self, _):
        try:
            self.out_discrim.value = self._fig_html(self.seg.plot_cap())
        except Exception as e:
            self.out_discrim.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro na "
                                      f"curva CAP: {type(e).__name__}: {e}</div>")

    def _on_lift(self, _):
        try:
            self.out_discrim.value = self._fig_html(self.seg.plot_lift())
        except Exception as e:
            self.out_discrim.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro no "
                                      f"lift: {type(e).__name__}: {e}</div>")

    def _on_msafra(self, _):
        # coluna de tempo: o campo "coluna safra" do card de irmãs (mesma fonte do
        # botão Estabilidade), com fallback no date_col configurado
        tcol = (self.tx_sib_time.value or "").strip() or self.date_col
        if not tcol or tcol not in self.seg.df.columns:
            self.out_discrim.value = ("<div class='treeui-legend'>Métricas por safra requerem "
                                      "uma coluna de data — configure <b>date_col</b> ou preencha "
                                      "a <b>coluna safra</b> no card de folhas-irmãs.</div>")
            return
        try:
            self.out_discrim.value = self._fig_html(
                self.seg.plot_metrics_by_safra(time_col=tcol), full_width=True)
        except Exception as e:
            self.out_discrim.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro nas "
                                      f"métricas por safra: {type(e).__name__}: {e}</div>")

    @staticmethod
    def _estab_err(what, e):
        return (f"<div style='color:var(--bad-tx);font-size:12px'>({what} não gerado: "
                f"{type(e).__name__})</div>")

    def _on_estab(self, _):
        # métricas por amostra (esq) | PSI da segmentação no tempo (dir) · concentração (abaixo)
        if self.sample_col is None:
            self.out_estab.value = ("<div class='treeui-legend'>Requer coluna de amostra "
                                    "(DES/OOT…) para estas análises.</div>")
            self.out_conc.value = ""
            self._estab_ready = False
            return
        with self._busy(self.btn_estab, status=self.out_estab,
                        msg="calculando a estabilidade…"):
            self._estab_tcol = (self.tx_sib_time.value or "").strip() or self.date_col
            # O gráfico de concentração das folhas entre amostras saiu daqui: com
            # muitas folhas as barras ficavam finas demais para ler. O mesmo dado
            # está na TABELA de folhas (colunas % DES / % OOT / % ESTAB), que
            # escala bem e permite ordenar.
            self._estab_ready = True
            self._render_estab_charts()

    def _read_psi_ylim(self):
        """(lo, hi) do zoom manual dos gráficos de PSI, em fração — os campos são
        lidos em % (÷100). ``None`` se ambos vazios."""
        def _p(tx):
            s = (tx.value or "").strip().replace("%", "").replace(",", ".")
            if not s:
                return None
            try:
                return float(s) / 100.0
            except ValueError:
                return None
        lo, hi = _p(self.tx_psi_ymin), _p(self.tx_psi_ymax)
        return None if (lo is None and hi is None) else (lo, hi)

    def _render_estab_charts(self):
        """(Re)renderiza métricas (seletor/%) + PSI da segmentação por safra e por
        amostra (respeitando o zoom do eixo Y), reusando a concentração já gerada.
        Ignora se a aba de estabilidade ainda não foi aberta."""
        if not getattr(self, "_estab_ready", False):
            return
        metrics = list(self.sm_estab_metrics.value) or None    # nada marcado ⇒ todas
        pct = True if self.ck_estab_pct.value else None         # marcado ⇒ tudo em %
        ylim = self._read_psi_ylim(); zoom = self._psi_zoom
        tcol = getattr(self, "_estab_tcol", None)
        try:
            # MESMA altura de figura (4.6) + tight=False → alinha com o PSI nas
            # colunas proporcionais 66:84 (larguras 6.6 : 8.4).
            h_m = self._fig_html(
                self.seg.plot_metrics_comparison(figsize=(6.6, 4.6), metrics=metrics, pct=pct),
                full_width=True, tight=False)
        except Exception as e:
            h_m = self._estab_err("métricas", e)
        if not tcol or tcol not in self.df.columns:
            h_psi_safra = ("<div class='treeui-legend'>Informe a coluna de tempo (no card de "
                           "folhas-irmãs, acima) para o PSI ao longo do tempo.</div>")
        else:
            try:
                h_psi_safra = self._fig_html(
                    self.seg.plot_psi_by_safra(time_col=tcol, figsize=(8.4, 4.6),
                                               ylim=ylim, auto_zoom=zoom),
                    full_width=True, tight=False)
            except Exception as e:
                h_psi_safra = self._estab_err("PSI no tempo", e)
        # PSI da segmentação ENTRE amostras (barras verticais): fica ENTRE as
        # métricas por amostra e o PSI no tempo — a leitura vai do agregado por
        # amostra, para o PSI por amostra, para a evolução no tempo.
        try:
            h_psi_s = self._fig_html(
                self.seg.plot_psi_by_sample(figsize=(6.0, 4.6), ylim=ylim, auto_zoom=zoom),
                full_width=True, tight=False)
        except Exception as e:
            h_psi_s = self._estab_err("PSI entre amostras", e)
        self.out_estab.value = (
            "<div style='display:flex;gap:10px;align-items:flex-start'>"
            f"<div style='flex:60 1 0;min-width:0'>"
            f"<div class='treeui-h'>Principais métricas por amostra</div>{h_m}</div>"
            f"<div style='flex:60 1 0;min-width:0'>"
            "<div class='treeui-h'>PSI da segmentação entre amostras</div>"
            f"{h_psi_s}</div>"
            f"<div style='flex:84 1 0;min-width:0'>"
            "<div class='treeui-h'>PSI da segmentação ao longo do tempo</div>"
            f"{h_psi_safra}</div></div>")
        self.out_conc.value = ""

    def _on_psi_zoom(self, _):
        self._psi_zoom = True
        self.btn_psi_zoom.button_style = "info"
        self._render_estab_charts()

    def _on_psi_reset(self, _):
        self._psi_zoom = False
        self.btn_psi_zoom.button_style = ""
        self.tx_psi_ymin.value = ""
        self.tx_psi_ymax.value = ""
        self._render_estab_charts()

    def _on_varprofile(self, _):
        # grade por variável da árvore: % missing por safra (0–100%) · dispersão p5·média·p95
        # (num.) / proporção das categorias (cat.), com faixas de troca de amostra.
        tcol = (self.tx_sib_time.value or "").strip() or self.date_col
        if not tcol or tcol not in self.df.columns:
            self.out_varprof_missing.value = ("<div class='treeui-legend'>Informe a coluna de "
                "tempo (no card de folhas-irmãs, acima) para o perfil por safra.</div>")
            self.out_varprof_stats.value = ""
            return
        if not self.seg.regua_features():
            self.out_varprof_missing.value = ("<div class='treeui-legend'>Nenhuma variável "
                "entrou na árvore — crie ao menos um split.</div>")
            self.out_varprof_stats.value = ""
            return

        def _err(what, e):
            return (f"<div style='color:var(--bad-tx);font-size:12px'>({what} não gerado: "
                    f"{type(e).__name__})</div>")
        with self._busy(self.btn_varprofile, status=self.out_varprof_missing,
                        msg="gerando o perfil das variáveis…"):
            try:
                self.out_varprof_missing.value = self._fig_html(
                    self.seg.plot_variables_missing_by_safra(time_col=tcol), full_width=True)
            except Exception as e:
                self.out_varprof_missing.value = _err("% missing", e)
            try:
                self.out_varprof_stats.value = self._fig_html(
                    self.seg.plot_variables_stats_by_safra(time_col=tcol), full_width=True)
            except Exception as e:
                self.out_varprof_stats.value = _err("dispersão", e)

    # ==================================================================
    # Folhas-irmãs: inversão do alvo entre amostras e safras
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
                       f"<span style='color:var(--bad-tx)'>{am_txt}</span>")
        else:
            am_line = "<b>Entre amostras:</b> <span style='color:var(--ok-tx)'>nenhuma inversão</span>"
        if s.get("safra_err"):
            sf_line = (f"<b>Entre safras:</b> <span style='color:var(--sub-ink)'>não avaliado "
                       f"({s['safra_err']})</span>")
        elif s["n_safras"]:
            pct = 100 * s["safra_rate"]
            cor = "var(--bad-tx)" if s["safras_inv"] else "var(--ok-tx)"
            sf_line = (f"<b>Entre safras:</b> <span style='color:{cor}'>"
                       f"{s['safras_inv']}/{s['n_safras']} safras com inversão "
                       f"({pct:.0f}%)</span>")
            piores = [r for r in s["safras"] if r["n_inv"] > 0][:8]
            if piores:
                chips = " ".join(
                    f"<span style='background:var(--bad-bg);color:var(--bad-ink);border-radius:3px;"
                    f"padding:1px 5px;font-size:10.5px' class='mono'>{r['safra']} "
                    f"({r['n_inv']})</span>" for r in piores)
                sf_line += f"<div style='margin-top:4px'>{chips}</div>"
        else:
            sf_line = "<b>Entre safras:</b> <span style='color:var(--sub-ink)'>sem safras avaliáveis</span>"
        return (
            "<div class='treeui-card' style='margin:6px 0'>"
            f"<div style='margin-bottom:6px'><span class='pill {pill}'>● {rotulo}</span>"
            f"<span style='color:var(--muted);font-size:11.5px;margin-left:8px'>"
            f"{s['n_pairs']} par(es) de irmãs comparados</span></div>"
            f"<div style='font-size:12px;line-height:1.7'>{am_line}<br>{sf_line}</div>"
            f"<div style='font-size:11px;color:var(--muted);margin-top:6px'>"
            f"Ordem de referência ({self._risk_label} na {s['ref_sample']}): {ordem}</div>"
            "</div>")

    def _on_sib_analyze(self, _):
        key = self.dd_sib_group.value
        g = getattr(self, "_sib_group_map", {}).get(key) if key is not None else None
        if not g:
            self._sib_ctx = None
            self.out_sib.value = ("<div style='font-size:12px;color:var(--sub-ink)'>Nenhum grupo de "
                                  "folhas-irmãs adjacentes — faça ao menos um split que deixe "
                                  "≥2 folhas terminais contíguas sob o mesmo pai.</div>")
            return
        pid, leaves = g["parent"], list(g["leaves"])
        tcol = (self.tx_sib_time.value or "").strip() or None
        samp = self.dd_sib_sample.value
        samp = None if samp in (None, "__all__") else samp

        def err(what, e):
            return (f"<div style='font-size:11px;color:var(--bad-tx)'>({what} não gerado: "
                    f"{type(e).__name__}: {e})</div>")

        with self._busy(self.btn_sib, status=self.out_sib,
                        msg="analisando as folhas-irmãs…"):
            try:
                summ = self.seg.sibling_inversion_summary(pid, time_col=tcol, sample=samp,
                                                          leaves=leaves)
                ind = self._sib_indicator_html(summ)
            except Exception as e:
                ind = err("indicador de inversão", e)
            # guarda o contexto p/ os controles de zoom/% re-renderizarem sem reanalisar
            self._sib_ctx = {"pid": pid, "leaves": leaves, "tcol": tcol, "samp": samp, "ind": ind}
            self._render_sib_charts()

    def _read_sib_ylim(self):
        """(lo, hi) do zoom manual dos gráficos de estabilidade, na escala do alvo.
        Campos vazios viram ``None``; se 'eixo em %' está ligado, o valor digitado é
        lido como % (÷100). ``None`` se ambos vazios."""
        def _p(tx):
            s = (tx.value or "").strip().replace("%", "").replace(",", ".")
            if not s:
                return None
            try:
                v = float(s)
            except ValueError:
                return None
            return v / 100.0 if self.ck_sib_pct.value else v
        lo, hi = _p(self.tx_sib_ymin), _p(self.tx_sib_ymax)
        return None if (lo is None and hi is None) else (lo, hi)

    def _render_sib_charts(self):
        """(Re)renderiza os 2 gráficos de estabilidade das folhas-irmãs respeitando
        o estado dos controles (zoom automático, mín/máx manual, eixo em %)."""
        ctx = getattr(self, "_sib_ctx", None)
        if not ctx:
            return
        pid, leaves, tcol, samp, ind = (ctx["pid"], ctx["leaves"], ctx["tcol"],
                                        ctx["samp"], ctx["ind"])
        ylim = self._read_sib_ylim()
        pct = self.ck_sib_pct.value

        def err(what, e):
            return (f"<div style='font-size:11px;color:var(--bad-tx)'>({what} não gerado: "
                    f"{type(e).__name__}: {e})</div>")

        try:
            # full_width + colunas proporcionais à largura das figuras (7.6 : 9.6),
            # ambas com a MESMA altura de figura (4.0) → os dois gráficos saem com a
            # MESMA altura na tela (a escala largura-coluna/largura-figura é igual).
            h1 = self._fig_html(self.seg.plot_sibling_value_by_sample(
                pid, leaves=leaves, ylim=ylim, auto_zoom=self._sib_zoom, pct=pct),
                full_width=True)
        except Exception as e:
            h1 = err("gráfico por amostra", e)
        try:
            h2 = self._fig_html(self.seg.plot_sibling_value_by_safra(
                pid, time_col=tcol, sample=samp, leaves=leaves,
                ylim=ylim, auto_zoom=self._sib_zoom, pct=pct), full_width=True)
        except Exception as e:
            h2 = err("gráfico por safra", e)
        charts = ("<div style='display:flex;gap:10px;align-items:flex-start'>"
                  f"<div style='flex:76 1 0;min-width:0'>{h1}</div>"
                  f"<div style='flex:96 1 0;min-width:0'>{h2}</div></div>")
        self.out_sib.value = ind + charts

    def _on_sib_zoom(self, _):
        self._sib_zoom = True
        self.btn_sib_zoom.button_style = "info"
        self._render_sib_charts()

    def _on_sib_reset(self, _):
        self._sib_zoom = False
        self.btn_sib_zoom.button_style = ""
        self.tx_sib_ymin.value = ""
        self.tx_sib_ymax.value = ""
        self._render_sib_charts()

    # ==================================================================
    # Undo / redo de splits (e demais alterações estruturais da árvore)
    # ==================================================================
    def _snapshot(self):
        """Estado restaurável: estrutura da árvore + folhas travadas + apelidos de
        negócio + folha selecionada (para o desfazer/refazer voltar à folha em foco)."""
        return {"segments": self.seg.to_dict()["segments"], "locked": set(self.locked),
                "selected": self.dd_leaf.value,
                "leaf_names": dict(self.seg.leaf_names)}

    def _checkpoint(self):
        """Empilha o estado atual para permitir desfazer; zera a pilha de refazer."""
        self._undo.append(self._snapshot())
        if len(self._undo) > 50:
            self._undo.pop(0)
        self._redo.clear()
        self._sync_undo_buttons()

    def _revert_checkpoint(self, redo_bak):
        """Desfaz o último :meth:`_checkpoint` quando a mutação FALHOU ou foi no-op:
        remove o snapshot espúrio do undo e restaura a pilha de redo (que o checkpoint
        havia zerado). Sem isto, um erro na mutação deixaria histórico corrompido."""
        if self._undo:
            self._undo.pop()
        self._redo[:] = redo_bak
        self._sync_undo_buttons()

    def _restore(self, snap):
        # registra as máscaras atuais (por condições) antes de carregar: os
        # segmentos que o undo/redo NÃO altera viram cache-hit e não são
        # recalculados via _match_conditions_pandas (o freeze do desfazer/refazer).
        self.seg._prime_mask_cache()
        self.seg._load_segments(snap["segments"])
        self.locked = set(snap["locked"]) & set(self.seg.segments)
        # apelidos de negócio viajam no snapshot (snapshots antigos → {});
        # sids extintos são filtrados na leitura (leaf_name/_leaf_names_validos)
        self.seg.leaf_names = dict(snap.get("leaf_names", {}))

    def _sync_undo_buttons(self):
        self.btn_undo.disabled = not self._undo
        self.btn_redo.disabled = not self._redo
        # os gêmeos da barra do canvas (podem não existir na 1ª chamada do _build)
        if getattr(self, "btn_cv_undo", None) is not None:
            self.btn_cv_undo.disabled = not self._undo
            self.btn_cv_redo.disabled = not self._redo

    def _on_undo(self, _):
        if not self._undo:
            return
        antes = self._delta_snapshot()      # métricas do estado atual (memoizadas)
        prev = self._undo.pop()
        self._redo.append(self._snapshot())
        self._restore(prev)
        self._pending = None
        self._sync_undo_buttons()
        self._refresh(select=prev.get("selected"))
        self._log_delta("↶ desfeito", antes)

    def _on_redo(self, _):
        if not self._redo:
            return
        antes = self._delta_snapshot()      # métricas do estado atual (memoizadas)
        nxt = self._redo.pop()
        self._undo.append(self._snapshot())
        self._restore(nxt)
        self._pending = None
        self._sync_undo_buttons()
        self._refresh(select=nxt.get("selected"))
        self._log_delta("↷ refeito", antes)

    # ==================================================================
    # Auto-merge: funde folhas-irmãs indistinguíveis automaticamente
    # ==================================================================
    def _on_automerge(self, _):
        import contextlib
        import io
        with self._busy(self.btn_automerge, self.btn_cv_automerge,
                        msg="rodando o auto-merge…"):
            n0 = sum(s["is_leaf"] for s in self.seg.segments.values())
            antes = self._delta_snapshot()
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
                self._log(f"Erro no auto-merge: {type(e).__name__}: {e}")
                return
            n1 = sum(s["is_leaf"] for s in self.seg.segments.values())
            self._log(buf.getvalue().strip() or "Auto-merge concluído.")
            if n1 == n0:
                self._log("Nenhuma folha-irmã indistinguível (p > alpha) — nada a fundir. "
                          f"Aumente o alpha ou o 'Δ{self._risk_label} mínimo' para fundir mais.")
            self.locked &= set(self.seg.segments)
            self._pending = None
            self._refresh()
            self._log_delta("auto-merge", antes)

    def _on_pdf(self, _):
        path = (self.tx_pdf_path.value or "").strip()
        if not path:
            self.out_pdf.value = "<i>Informe o caminho do .pdf.</i>"; return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        with self._busy(self.btn_pdf, status=self.out_pdf, msg="gerando o PDF…"):
            try:
                tcol = self.date_col if (self.date_col and self.date_col in self.df.columns) else None
                self.seg.report_pdf(path, time_col=tcol)
            except Exception as e:
                self.out_pdf.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro ao gerar "
                                      f"PDF: {type(e).__name__}: {e}</div>")
                self._log(f"[pdf] erro: {e}"); return
            self.out_pdf.value = (f"<div class='treeui-legend'>✅ Relatório salvo em "
                                  f"<code>{path}</code>.</div>")
            self._log(f"[pdf] relatório salvo em {path}")

    def _on_xlsx(self, _):
        path = (self.tx_xlsx_path.value or "").strip()
        if not path:
            self._log("Informe o caminho do arquivo .xlsx."); return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        self._confirm_overwrite(path, lambda: self._do_xlsx(path))

    def _do_xlsx(self, path):
        """Gera o Excel multi-abas via :meth:`TreeSegmenter.to_excel` (openpyxl é
        OPCIONAL — sem ele, o ImportError amigável vai para o console)."""
        with self._busy(self.btn_xlsx, msg="gerando o Excel…"):
            tabela = (self.tx_sql_table.value or "").strip() or "minha_tabela"
            try:
                self.seg.to_excel(path, table=tabela)
            except ImportError as e:
                self._log(f"⚠ {e}")
                return
            except Exception as e:
                self._log(f"Erro ao exportar Excel: {type(e).__name__}: {e}")
                return
            self._log(f"📊 Excel salvo em '{path}' (Folhas · Métricas por amostra · "
                      "PSI · IV por variável · Calibração · Régua SQL).")

    # ==================================================================
    # Persistência: salvar / carregar a árvore em JSON
    # ==================================================================
    def _confirm_overwrite(self, path, do_save):
        """Se ``path`` já existir, mostra a confirmação INLINE (aba Exportar, sob
        os campos de caminho) e só executa ``do_save()`` no clique em 'Sobrescrever'.
        Sem conflito (ou ``path`` vazio), salva direto. O diálogo inline não é
        apagado por ``clear_output`` do console (o antigo se perdia no rodapé)."""
        import html as _html
        import os
        if not path or not os.path.exists(path):
            do_save(); return
        self._confirm_pending = {"path": path, "do_save": do_save}
        self.html_confirm.value = (
            "<div style='border:1px solid var(--notice-border);background:var(--notice-bg);"
            "border-radius:10px;"
            "padding:10px 12px;font-size:12.5px;color:var(--notice-ink);line-height:1.5'>"
            "<b>⚠️ O arquivo já existe</b><br>"
            f"<code>{_html.escape(path)}</code><br>Deseja sobrescrever?</div>")
        self.box_confirm.layout.display = ""      # revela o diálogo inline

    def _on_confirm_yes(self, _):
        pend = self._confirm_pending
        self._confirm_pending = None
        self.box_confirm.layout.display = "none"
        self.html_confirm.value = ""
        if pend is not None:
            pend["do_save"]()

    def _on_confirm_no(self, _):
        pend = self._confirm_pending
        self._confirm_pending = None
        self.box_confirm.layout.display = "none"
        self.html_confirm.value = ""
        if pend is not None:
            self._log(f"Operação cancelada — '{pend['path']}' não foi sobrescrito.")

    def _on_save_json(self, _):
        path = self.tx_json_path.value.strip()
        if not path:
            self._log("Informe o caminho do arquivo .json.")
            return
        self._confirm_overwrite(path, lambda: self._do_save_json(path))

    def _do_save_json(self, path):
        import json
        try:
            data = self.seg.to_dict()
            data["_ui"] = {"locked": sorted(self.locked)}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            n = sum(s["is_leaf"] for s in self.seg.segments.values())
            self._log(f"💾 árvore salva em '{path}' ({n} folhas).")
        except Exception as e:
            self._log(f"Erro ao salvar: {type(e).__name__}: {e}")

    def _on_load_json(self, _):
        import json
        import os
        path = self.tx_json_path.value.strip()
        if not path:
            self._log("Informe o caminho do arquivo .json."); return
        if not os.path.exists(path):
            self._log(f"Arquivo não encontrado: '{path}'."); return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("meta", {})
            if meta.get("target") and meta.get("target") != self.target:
                self._log(f"⚠ aviso: árvore salva com target='{meta.get('target')}', "
                          f"mas esta UI usa '{self.target}'. Carregando mesmo assim.")
            antes = self._delta_snapshot()
            self._checkpoint()
            self.seg._load_segments(data["segments"])
            # apelidos de negócio persistidos no JSON (chave ausente em JSONs
            # antigos → {}); sids que não são folhas desta árvore são descartados
            nomes = meta.get("leaf_names") or {}
            self.seg.leaf_names = {
                sid: str(n) for sid, n in nomes.items()
                if n and sid in self.seg.segments
                and self.seg.segments[sid]["is_leaf"]}
            # fallback persistido no JSON: restaura no segmentador e sincroniza o
            # dropdown quando a opção existe na UI (nota int fica só no segmentador)
            if "fallback" in meta:
                self.seg.fallback = meta.get("fallback")
                if self.seg.fallback in (None, "pior_nota"):
                    self.dd_fallback.value = self.seg.fallback
            self.locked = set(data.get("_ui", {}).get("locked", [])) & set(self.seg.segments)
            self._pending = None
            n = sum(s["is_leaf"] for s in self.seg.segments.values())
            self._log(f"📂 árvore carregada de '{path}' ({n} folhas).")
        except Exception as e:
            self._log(f"Erro ao carregar: {type(e).__name__}: {e}"); return
        self._refresh()
        self._log_delta("carregar JSON", antes)

    # ==================================================================
    # Imagem da árvore (matplotlib)
    # ==================================================================
    def _on_plot(self, _):
        import os
        path = self.tx_img_path.value.strip() or None
        if path and os.path.exists(path):
            self._do_plot(None)                       # mostra a árvore sem exportar
            self._confirm_overwrite(path, lambda: self._do_plot(path))
        else:
            self._do_plot(path)

    def _do_plot(self, path):
        try:
            # sem destaque da folha selecionada: todas as folhas com o mesmo estilo
            fig = self.seg.plot_tree(save_path=path)    # repr. % + alvo (DES)
            self.out_plot.value = self._fig_html(fig, border=True)
        except Exception as e:
            self.out_plot.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro ao "
                                   f"desenhar a árvore: {type(e).__name__}: {e}</div>")
            return
        if path:
            self._log(f"🖼️ imagem da árvore salva em '{path}' (tamanho real).")

    def _on_plot_hide(self, _):
        self.out_plot.value = ""          # recolhe (esvazia) a imagem

    # ==================================================================
    # Aba "Árvore interativa" — canvas navegável + painel de criação
    #
    # A aba Construir edita a árvore por formulário (escolher a folha num
    # dropdown, configurar, aplicar). Aqui a ordem se inverte: a árvore é o
    # mapa, clica-se no nó que interessa e o painel da direita abre já no
    # contexto dele — corte e regras de negócio da folha no mesmo lugar.
    # ==================================================================
    _CV_NODE_W = 188          # cartão do nó, em px do PLANO (antes do zoom)
    _CV_NODE_H = 104
    _CV_GAP_X = 26            # respiro entre cartões vizinhos
    _CV_GAP_Y = 68            # respiro entre níveis

    def _canvas_children(self):
        """(filhos de cada nó, mapa de notas). Os filhos saem na MESMA ordem da
        árvore de texto da aba Construir (menor alvo à esquerda) — as duas
        telas mostram a árvore na mesma ordem."""
        filhos: dict = {}
        for sid, s in self.seg.segments.items():
            filhos.setdefault(s["parent"], []).append(sid)
        nota_map, _ = self.seg._grade_map()
        min_nota = self._min_nota_fn(filhos, nota_map)
        return {k: sorted(v, key=min_nota) for k, v in filhos.items()}, nota_map

    def _canvas_layout(self):
        """Posiciona os cartões no plano: cada folha ocupa uma coluna, na ordem
        da árvore, e o pai fica centralizado sobre os filhos. Devolve
        ``(nós, arestas, largura, altura)`` já em px do plano."""
        NW, NH = self._CV_NODE_W, self._CV_NODE_H
        GX, GY = self._CV_GAP_X, self._CV_GAP_Y
        filhos, nota_map = self._canvas_children()
        pos: dict = {}
        col = [0]        # próxima coluna livre (lista = célula mutável no closure)
        prof = [0]

        def walk(sid, d):
            prof[0] = max(prof[0], d)
            ch = filhos.get(sid, [])
            if not ch:
                x = col[0] * (NW + GX)
                col[0] += 1
            else:
                xs = [walk(c, d + 1) for c in ch]
                x = (min(xs) + max(xs)) / 2
            pos[sid] = (x, d * (NH + GY))
            return x

        walk("root", 0)
        lo, hi = self._leaf_values()
        n_total = len(self.df)
        nodes = [dict(sid=sid, x=round(x), y=round(y), w=NW, h=NH,
                      leaf=bool(self.seg.segments[sid]["is_leaf"]),
                      html=self._canvas_node_html(sid, filhos, nota_map, lo, hi, n_total))
                 for sid, (x, y) in pos.items()]
        edges = []
        for sid, ch in filhos.items():
            if sid is None or sid not in pos:
                continue
            x0, y0 = pos[sid][0] + NW / 2, pos[sid][1] + NH
            for c in ch:
                x1, y1 = pos[c][0] + NW / 2, pos[c][1]
                edges.append(dict(child=c, d=(
                    f"M {x0:.1f} {y0:.1f} C {x0:.1f} {y0 + GY * 0.55:.1f}, "
                    f"{x1:.1f} {y1 - GY * 0.55:.1f}, {x1:.1f} {y1:.1f}")))
        largura = max(1, col[0] * (NW + GX) - GX)
        altura = (prof[0] + 1) * (NH + GY) - GY
        return nodes, edges, int(largura), int(altura)

    def _canvas_node_html(self, sid, filhos, nota_map, lo, hi, n_total):
        """Conteúdo do cartão de um nó — montado (e escapado) aqui, como no
        tooltip do preview clicável: o front só injeta a string pronta."""
        s = self.seg.segments[sid]
        n = int(s["mask"].sum())
        rep = 100 * n / n_total if n_total else 0.0
        ref = self.ref_sample if self.sample_col is not None else None
        v = self._node_value(sid, ref)
        cor = self._color(v, lo, hi)
        rot = ("TODA A CARTEIRA" if s["parent"] is None
               else self.seg._descrever([s["conditions"][-1]]))
        v_txt = "—" if pd.isna(v) else f"{v * 100:.2f}%"
        vol = f"{n:,}".replace(",", ".")
        if s["is_leaf"]:
            chip = f"folha {nota_map.get(sid, '?')}"
            nome = self.seg.leaf_name(sid)
            sub = f"{vol} obs · {rep:.1f}% da carteira"
            if nome:
                sub = f"{_esc(nome)} · {sub}"
        else:
            ch = filhos.get(sid, [])
            feat = None
            if ch:
                conds = self.seg.segments[ch[0]]["conditions"]
                feat = conds[-1]["feature"] if conds else None
            lbl = self.seg.feature_labels.get(feat, feat) if feat else None
            chip = f"{len(ch)} ramos"
            sub = (f"dividida por {_esc(str(lbl))}" if lbl
                   else f"{vol} obs · {rep:.1f}% da carteira")
        lock = ("<span title='folha fechada' style='font-size:11px;flex:none'>🔒</span>"
                if sid in self.locked else "")
        # semáforo de PSI da folha (pior amostra não-referência): com ele o mapa
        # vira um heatmap de estabilidade — a cor do ALVO já está no número, a
        # do PSI fica neste ponto. Detalhe por amostra no title (hover).
        psi_dot = ""
        if s["is_leaf"] and self.sample_col is not None and self._nonref:
            partes, pior = [], None
            for a in self._nonref:
                p = self._leaf_psi(sid, a)
                if pd.isna(p):
                    continue
                ab = "ESTAB" if a == "ESTABILIDADE" else a
                partes.append(f"PSI {ab} {p:.1%}")
                pior = p if pior is None else max(pior, p)
            if pior is not None:
                cor_psi = {"green": "var(--ok-tx)", "yellow": "var(--warn-tx)",
                           "red": "var(--bad-tx)"}[self._psi_class(pior)]
                psi_dot = (f"<span class='pdot' style='background:{cor_psi}' "
                           f"title=\"{_esc(' · '.join(partes))}\"></span>")
        return (f"<div class='t'><span class='lb' title=\"{_esc(rot)}\">{_esc(rot)}</span>"
                f"{psi_dot}{lock}</div>"
                f"<div class='m'><span class='v' style='color:{cor}'>{v_txt}</span>"
                f"<span class='g'>{_esc(chip)}</span></div>"
                f"<div class='s'>{sub}</div>"
                f"<div class='bar'><i style='width:{max(3, min(100, rep)):.1f}%;"
                f"background:{cor}'></i></div>")

    def _ensure_canvas_widget(self):
        """Instancia o canvas 1× e o monta no card. ``False`` quando o anywidget
        não está disponível/ligado — a aba então mostra só o aviso."""
        if not self.allow_interactive_tree:
            return False
        if self._cv_widget is None:
            cls = _tree_canvas_widget_cls()
            if cls is None:
                return False
            self._cv_widget = cls(layout=W.Layout(width="100%", height="100%"))
            self._cv_widget.observe(self._on_cv_select, names="selected")
        if self._cv_widget not in self.box_cv_canvas.children:
            self.box_cv_canvas.children = (self._cv_widget,)
        return True

    def _canvas_unavailable_html(self):
        """Aviso quando o canvas não pode ser desenhado — mesma explicação (e
        mesma saída) do preview clicável da aba Construir."""
        if not self.allow_interactive_tree:
            return ("<div class='treeui-legend' style='padding:22px 18px'>🔒 <b>Árvore "
                    "interativa desligada.</b> Ela depende do <code>anywidget</code>, cujo "
                    "frontend o gerenciador de widgets do Databricks busca de um CDN — num "
                    "cluster sem egress a tela ficaria em branco. Por isso o padrão dentro do "
                    "Databricks é desligada.<br/><br/>Para ligá-la num ambiente com egress (ou "
                    "com o anywidget instalado como lib do cluster):<br/>"
                    "<code>TreeSegmenterUI(..., allow_interactive_tree=True)</code>"
                    "<br/><br/>Enquanto isso, a aba <b>Construir</b> faz as mesmas divisões "
                    "pelo formulário, sem depender de rede.</div>")
        return ("<div class='treeui-legend' style='padding:22px 18px'>💡 <b>Instale o "
                "<code>anywidget</code></b> (<code>pip install anywidget</code>) para navegar "
                "a árvore neste plano: arrastar, ampliar e clicar num nó para abrir o painel "
                "de corte.<br/><br/>Sem ele, a aba <b>Construir</b> faz as mesmas divisões "
                "pelo formulário.</div>")

    def _refresh_canvas(self, fit=False, center=False):
        """(Re)desenha o canvas a partir da árvore atual. ``fit`` enquadra tudo,
        ``center`` centraliza no nó selecionado (contadores nos traits)."""
        # o "ir para folha" e o salvar cenário vivem na barra e independem do
        # mapa — as opções acompanham as folhas atuais mesmo no modo offline
        self._sync_cv_goto_options()
        if not self._ensure_canvas_widget():
            # sem o plano navegável, o painel ainda vale: ele é ipywidgets puro
            # (zero rede) e opera sobre a folha ativa da UI — o que se perde é o
            # mapa, não o corte nem as regras de negócio
            self.out_cv_msg.value = self._canvas_unavailable_html()
            self.box_cv_canvas.layout.display = "none"
            self.btn_cv_fit.layout.display = "none"
            self._refresh_cv_panel()
            return
        self.box_cv_canvas.layout.display = ""
        self.btn_cv_fit.layout.display = ""
        self.out_cv_msg.value = ""
        self._sync_cv_auto_cfg()
        w = self._cv_widget
        try:
            nodes, edges, cw, ch = self._canvas_layout()
        except Exception as e:
            self.out_cv_msg.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro ao "
                                     f"desenhar a árvore: {type(e).__name__}: {e}</div>")
            return
        sel = self._cv_sel if self._cv_sel in self.seg.segments else ""
        self._cv_sel = sel or None
        # se a atribuição de `selected` muda o trait, o observer (_on_cv_select)
        # já refaz o painel — refazer aqui de novo pagaria os testes de vizinhas
        # e o mover-corte 2× por abertura
        painel_ja_vai = w.selected != sel
        with w.hold_sync():                    # 1 mensagem só pelo comm
            w.nodes = nodes
            w.edges = edges
            w.content_w, w.content_h = cw, ch
            w.selected = sel
            if fit:
                w.fit_token += 1
            if center and sel:
                w.center_token += 1
        if not painel_ja_vai:
            self._refresh_cv_panel()

    # ---- seleção ----------------------------------------------------
    def _on_cv_select(self, change):
        """Clique num nó do canvas (trait ``selected``). Folha selecionada vira
        também a folha ativa do resto da UI — as abas seguem juntas."""
        sid = change.get("new") or None
        self._cv_sel = sid if sid in self.seg.segments else None
        if (self._cv_sel is not None and self.seg.segments[self._cv_sel]["is_leaf"]
                and self.dd_leaf.value != self._cv_sel
                and self._cv_sel in [s for _, s in self.dd_leaf.options]):
            self.dd_leaf.value = self._cv_sel
        self._refresh_cv_panel()

    def _sync_canvas_selection(self):
        """Espelha na aba do canvas a folha escolhida em OUTRA aba — as telas
        mostram sempre o mesmo nó em foco. Vale mesmo sem o canvas montado: o
        painel sozinho já opera sobre a folha ativa."""
        sid = self.dd_leaf.value
        if sid is None or sid not in self.seg.segments:
            return
        if self._cv_sel != sid:
            self._cv_sel = sid
            # o painel ordena as variáveis por IV NA FOLHA, o que custa um
            # variable_iv. Fora da aba, só anotamos o nó e marcamos pendente: quem
            # trabalha na aba Construir não paga por uma tela que não está vendo.
            if (getattr(self, "tabs", None) is not None
                    and self.tabs.selected_index == self._canvas_tab_index):
                self._refresh_cv_panel()
            else:
                self._cv_dirty = True
        w = self._cv_widget
        if w is not None and w.selected != sid:
            w.selected = sid

    def _cv_node(self):
        """Nó em foco no painel: o clicado no canvas ou, na falta dele, a folha
        ativa da UI (o painel serve mesmo sem o canvas ter sido tocado)."""
        sid = self._cv_sel
        if sid is None or sid not in self.seg.segments:
            sid = self.dd_leaf.value
        return sid if sid in self.seg.segments else None

    # ---- painel de criação ------------------------------------------
    def _refresh_cv_feature_options(self):
        """Opções do seletor de variável do painel, ordenadas por IV NESTA folha
        (o IV é por folha — a ordem muda a cada nó clicado)."""
        sid = self._cv_node()
        labels, mapa = self._feature_option_labels(by_iv=True, sid=sid)
        self._cv_feat_by_label = mapa
        atual = self.dd_cv_feature.value
        col = mapa.get(atual) if atual else None
        self._cv_syncing = True
        try:
            self.dd_cv_feature.options = labels
            alvo = next((l for l, f in mapa.items() if f == col), None)
            self.dd_cv_feature.value = alvo or (labels[0] if labels else None)
        finally:
            self._cv_syncing = False

    def _cv_feature(self, warn=True):
        return self._combo_feature(self.dd_cv_feature, self._cv_feat_by_label, warn=warn)

    def _cv_kind(self, feature=None, sid=None):
        """'num' ou 'cat' para a variável na folha em foco."""
        sid = sid or self._cv_node()
        feature = feature or self._cv_feature(warn=False)
        if sid is None or feature is None:
            return "num"
        sub = self.df[self.seg.segments[sid]["mask"]]
        try:
            return self.seg._detect_kind(sub, feature, None)
        except Exception:
            return "num"

    def _refresh_cv_panel(self):
        """Reconstrói o painel para o nó em foco: cabeçalho, métricas, sugestões
        de corte e quais controles/ações fazem sentido ali."""
        sid = self._cv_node()
        if sid is None:
            self.box_cv_panel.layout.display = "none"
            self.out_cv_empty.layout.display = ""
            return
        self.out_cv_empty.layout.display = "none"
        self.box_cv_panel.layout.display = ""
        s = self.seg.segments[sid]
        is_leaf, is_root = bool(s["is_leaf"]), s["parent"] is None
        travada = sid in self.locked
        nota_map, _ = self.seg._grade_map()
        n = int(s["mask"].sum())
        n_total = len(self.df)
        rep = 100 * n / n_total if n_total else 0.0
        ref = self.ref_sample if self.sample_col is not None else None
        v = self._node_value(sid, ref)
        v_raiz = self._node_value("root", ref)

        # ---- cabeçalho: o que é este nó e onde ele fica na árvore
        if is_root:
            kicker = "Raiz da árvore" if is_leaf else "Raiz · já dividida"
        elif not is_leaf:
            kicker = "Ramo já dividido"
        else:
            kicker = "Folha fechada" if travada else f"Folha {nota_map.get(sid, '?')}"
        rot = ("TODA A CARTEIRA" if is_root
               else self.seg._descrever([s["conditions"][-1]]))
        caminho = ("caminho: " + self.seg._descrever(s["conditions"][:-1])
                   if len(s["conditions"]) > 1 else "direto da raiz")
        self.out_cv_head.value = (
            f"<div style='font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;"
            f"color:var(--ac);font-weight:700'>{_esc(kicker)}</div>"
            f"<div style='font-size:15px;font-weight:600;line-height:1.25;margin-top:2px;"
            f"color:var(--strong-ink)'>{_esc(rot)}</div>"
            f"<div style='font-size:10.5px;color:var(--sub-ink);margin-top:3px'>"
            f"{_esc(caminho if not is_root else 'toda a base')}</div>")

        # ---- métricas do nó
        v_txt = "—" if pd.isna(v) else f"{v * 100:.2f}%"
        if pd.isna(v) or pd.isna(v_raiz) or v_raiz == 0:
            delta = "—"
        else:
            d = (v - v_raiz) * 100
            delta = f"{d:+.2f} p.p."
        tiles = [("população", f"{n:,}".replace(",", ".")),
                 ("% carteira", f"{rep:.1f}%"),
                 (self._risk_label + (f" ({ref})" if ref else ""), v_txt),
                 ("vs. carteira", delta)]
        self.out_cv_stats.value = "<div class='treeui-metrics'>" + "".join(
            f"<div class='treeui-metric'><div class='k'>{_esc(k)}</div>"
            f"<div class='v'>{_esc(val)}</div></div>" for k, val in tiles) + "</div>"

        # ---- o corte só existe para folha aberta
        pode_dividir = is_leaf and not travada
        self.box_cv_split.layout.display = "" if pode_dividir else "none"
        if pode_dividir:
            self._refresh_cv_feature_options()
            self._refresh_cv_suggestions(sid)
            self._sync_cv_mode()
        if not is_leaf:
            self.out_cv_note.value = (
                "<div style='background:var(--ac-soft);color:var(--ac-deep);border-radius:9px;"
                "padding:9px 12px;font-size:11.5px'>Este ramo já foi dividido. Para refazer o "
                "corte, use <b>Recolher para o pai</b> abaixo — os filhos somem e ele volta a "
                "ser folha.</div>")
        elif travada:
            self.out_cv_note.value = (
                "<div style='background:var(--warn-bg);color:var(--warn-ink);border-radius:9px;"
                "padding:9px 12px;font-size:11.5px'>🔒 Folha fechada como final. Reabra abaixo "
                "para poder dividi-la.</div>")
        else:
            self.out_cv_note.value = ""

        # ---- regras de negócio: apelido + ações válidas neste nó
        self._cv_syncing = True
        try:
            self.tx_cv_name.value = (self.seg.leaf_name(sid) or "") if is_leaf else ""
        finally:
            self._cv_syncing = False
        self.tx_cv_name.disabled = not is_leaf
        self._sync_cv_actions()
        self.out_cv_preview.value = ""      # o preview era de outro nó

    def _sync_cv_actions(self):
        """Habilita cada ação do nó em foco conforme o que é válido ali.

        Vive fora do ``_refresh_cv_panel`` de propósito: precisa rodar DEPOIS do
        ``_busy`` dos handlers. O ``_busy`` re-habilita ao sair todos os botões
        que desabilitou, o que apagaria o estado calculado aqui — foi assim que
        "Alocar faltantes" continuava clicável depois de já ter alocado."""
        sid = self._cv_node()
        if sid is None:
            return
        s = self.seg.segments[sid]
        is_leaf, is_root = bool(s["is_leaf"]), s["parent"] is None
        self.btn_cv_lock.description = ("🔓 Reabrir folha" if sid in self.locked
                                        else "🔒 Fechar folha")
        self.btn_cv_lock.disabled = not is_leaf
        # a raiz é folha enquanto a árvore está vazia, mas não tem irmãs
        esq, dir_ = ((None, None) if (not is_leaf or is_root)
                     else self.seg._adjacent_sibling_neighbors(sid))
        self.btn_cv_merge_l.disabled = esq is None
        self.btn_cv_merge_r.disabled = dir_ is None
        self.out_cv_merge_p.value = self._cv_merge_p_html(sid)
        self.btn_cv_collapse.disabled = is_leaf
        self.btn_cv_missing.disabled = self._cv_missing_sibling(sid) is None
        self._sync_cv_move()                # corte da divisa da folha em foco

    def _cv_merge_p_html(self, sid):
        """Teste de hipótese contra cada vizinha ADJACENTE, no ponto onde a
        fusão é decidida: p acima do α (o mesmo do auto-fundir) = alvo
        indistinguível da vizinha = candidata natural a fundir."""
        name, testes = self._sibling_adjacent_tests(sid)
        if not testes:
            return ""
        alpha = float(self.sl_alpha.value)
        partes = []
        for lado, _desc, p in testes:
            if pd.isna(p):
                txt, cor, nota = "—", "var(--sub-ink)", "amostra pequena"
            elif p > alpha:
                txt = f"p={p:.3f}"
                cor, nota = "var(--warn-tx)", "indistinguível — candidata a fusão"
            else:
                txt = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
                cor, nota = "var(--ok-tx)", "distinta"
            partes.append(f"{lado} <b style='color:{cor}'>{txt}</b> ({nota})")
        return (f"<div class='treeui-legend' style='margin:4px 0 0'>{_esc(name)} vs "
                f"vizinhas (α={alpha:g}): " + " · ".join(partes) + "</div>")

    def _refresh_cv_suggestions(self, sid):
        """Três variáveis de maior IV nesta folha, como atalhos: clicar já
        seleciona a variável e mostra o preview do corte ótimo."""
        ivm = self._iv_map(sid)
        top = sorted(((f, iv) for f, iv in ivm.items() if not pd.isna(iv)),
                     key=lambda kv: -kv[1])[:3]
        self._cv_sug = [f for f, _ in top]
        for i, b in enumerate(self.btns_cv_sug):
            if i < len(top):
                f, iv = top[i]
                b.description = f"{self.seg.feature_labels.get(f, f)}  ·  IV {iv:.3f}"
                b.tooltip = f"Usar '{f}' no corte desta folha (IV {iv:.4f})"
                b.layout.display = ""
            else:
                b.layout.display = "none"
        self.out_cv_sug_h.layout.display = "" if top else "none"

    def _sync_cv_mode(self):
        """Ótimo mostra máx. faixas + critério. Manual mostra a caixa de cortes
        (numérica) ou o agrupador de categorias (categórica) — a MESMA regra da
        aba Construir: uma categoria por linha, categorias do mesmo grupo viram
        um nó só."""
        manual = self.tg_cv_mode.value == "Manual"
        cat = self._cv_kind() == "cat"
        self.sl_cv_bins.layout.display = "none" if manual else ""
        self.dd_cv_crit.layout.display = "none" if manual else ""
        self.box_cv_cuts.layout.display = "" if (manual and not cat) else "none"
        self.cv_cat_box.layout.display = "" if (manual and cat) else "none"
        self.out_cv_cuts_hint.value = (
            "<div style='font-size:10.5px;color:var(--sub-ink);margin:2px 0 0 4px'>"
            "numérica — um corte por vírgula; cada corte fecha à direita (≤).</div>")
        # o binning ótimo daqui honra os limites de tamanho de bin marcados em
        # Construir; sem dizer isso, faixas "faltando" pareceriam bug
        limites = self._optbin_extra()
        rot = {"min_bin_size": "mín. por faixa", "max_bin_size": "máx. por faixa",
               "min_mean_diff": "Δ mínimo entre faixas"}
        self.out_cv_optbin_hint.value = (
            "" if (manual or not limites) else
            "<div style='font-size:10.5px;color:var(--sub-ink);margin:2px 0 4px 4px'>"
            "limites herdados da aba Construir: "
            + " · ".join(f"{rot[k]} <b>{v:g}</b>" for k, v in limites.items()) + "</div>")
        if manual and cat:
            self._rebuild_cv_cat_box()

    def _rebuild_cv_cat_box(self):
        """Um seletor de grupo por categoria presente na folha, ordenadas pelo
        alvo — espelha ``_rebuild_cat_box`` da aba Construir, inclusive a guarda
        que evita reinstanciar os Dropdowns quando o contexto não mudou."""
        sid, feat = self._cv_node(), self._cv_feature(warn=False)
        if feat is None or sid is None or sid not in self.seg.segments:
            self.cv_cat_box.children = (); return
        if (getattr(self, "_cv_cat_ctx", None) == (feat, sid)
                and getattr(self, "_cv_cat_widgets", None)):
            return
        self._cv_cat_widgets = {}
        self._cv_cat_ctx = (feat, sid)
        sub = self.df[self.seg.segments[sid]["mask"]]
        s = sub[feat]
        valid = sub[s.notna()]
        if len(valid) == 0:
            self.cv_cat_box.children = (W.HTML(
                "<div style='font-size:11px;color:var(--sub-ink)'>Sem categorias nesta "
                "folha.</div>"),)
            return
        means = (valid.assign(_c=valid[feat].astype(str))
                 .groupby("_c")[self.target].mean().sort_values())
        order = means.index.tolist()
        n = len(order)
        linhas = [W.HTML("<div style='font-size:10.5px;color:var(--muted);margin-bottom:4px'>"
                         f"Categorias no <b>mesmo grupo</b> viram um nó. Ordenadas por "
                         f"{_esc(self._risk_label)}. Faltantes (NaN) já viram um nó "
                         "próprio.</div>")]
        for k, c in enumerate(order, 1):
            dd = W.Dropdown(options=[(f"grupo {g}", g) for g in range(1, n + 1)], value=k,
                            layout=W.Layout(width="96px"))
            self._cv_cat_widgets[c] = dd
            lab = W.HTML(f"<span style='font-size:11.5px'><b>{_esc(c)}</b>"
                         f"<span style='color:var(--sub-ink)'> · {means[c]:.3f}</span></span>")
            linhas.append(W.HBox([dd, lab], layout=W.Layout(align_items="center")))
        na_n = int(s.isna().sum())
        if na_n:
            linhas.append(W.HTML(f"<div style='font-size:10.5px;color:var(--warn-tx);"
                                 f"margin-top:3px'>+ <b>(faltante)</b>: {na_n} linhas → nó "
                                 "próprio automático</div>"))
        self.cv_cat_box.children = tuple(linhas)

    def _cv_cat_groups(self):
        """Grupos montados no agrupador de categorias do painel."""
        if (getattr(self, "_cv_cat_ctx", None) != (self._cv_feature(warn=False),
                                                   self._cv_node())
                or not getattr(self, "_cv_cat_widgets", None)):
            self._rebuild_cv_cat_box()
        grupos: dict = {}
        for c, dd in (getattr(self, "_cv_cat_widgets", None) or {}).items():
            grupos.setdefault(dd.value, []).append(c)
        return [grupos[g] for g in sorted(grupos)]

    def _on_cv_mode(self, _):
        if self._cv_syncing:
            return
        self.out_cv_preview.value = ""
        self._sync_cv_mode()

    def _on_cv_feature(self, _):
        if self._cv_syncing:
            return
        self.out_cv_preview.value = ""
        self._sync_cv_mode()          # o tipo da variável muda o campo de cortes

    def _on_cv_sug(self, i):
        """Atalho de sugestão: seleciona a variável e já mostra o preview."""
        def _handler(_):
            if i >= len(self._cv_sug):
                return
            col = self._cv_sug[i]
            alvo = next((l for l, f in self._cv_feat_by_label.items() if f == col), None)
            if alvo is None:
                return
            self._cv_syncing = True
            try:
                self.dd_cv_feature.value = alvo
            finally:
                self._cv_syncing = False
            self._sync_cv_mode()
            self._on_cv_preview(None)
        return _handler

    def _cv_parse_cuts(self, feature, sid):
        """Cortes do modo Manual: a lista de números da caixa de texto, ou os
        grupos montados no agrupador de categorias (mesma regra da Construir)."""
        if self._cv_kind(feature, sid) != "num":
            return self._cv_cat_groups() or None
        txt = (self.tx_cv_cuts.value or "").strip()
        if not txt:
            return None
        vals = [x.strip() for x in txt.replace(";", ",").split(",") if x.strip()]
        try:
            return [float(x) for x in vals] or None
        except ValueError as e:
            raise ValueError(f"corte não numérico em '{txt}' ({e})") from None

    def _cv_prepare(self):
        """Monta ``self._pending`` a partir dos controles do painel (mesma
        validação por ``show_grow`` do fluxo da aba Construir)."""
        import contextlib
        import io
        sid = self._cv_node()
        if sid is None or not self.seg.segments[sid]["is_leaf"]:
            return False, "Selecione uma FOLHA no canvas para dividir."
        if sid in self.locked:
            return False, "⚠ Folha fechada — reabra para dividir."
        feature = self._cv_feature(warn=False)
        if feature is None:
            return False, "⚠ Escolha uma variável da lista."
        try:
            if self.tg_cv_mode.value == "Ótimo":
                splits = None
                # os limites de tamanho de bin (min/max/Δ mínimo) vêm dos mesmos
                # checkboxes da aba Construir: sem isto o "Ótimo" daqui produziria
                # faixas diferentes das de lá com a UI mostrando a mesma config
                extra = dict(max_n_bins=self.sl_cv_bins.value,
                             criterion=self.dd_cv_crit.value, **self._optbin_extra())
            else:
                splits, extra = self._cv_parse_cuts(feature, sid), {}
                if not splits:
                    return False, "⚠ Preencha os cortes para o modo Manual."
            # show_grow devolve {sid: tabela das faixas} e imprime o preview no
            # stdout — aqui só a tabela interessa (o painel a desenha)
            with contextlib.redirect_stdout(io.StringIO()):
                prev = self.seg.show_grow(feature, splits=splits,
                                          only_segments=[sid], **extra)
            self._cv_prev_tbl = prev.get(sid)
            self._pending = dict(feature=feature, splits=splits,
                                 only_segments=[sid], **extra)
            return True, None
        except Exception as e:
            self._pending = None
            self._cv_prev_tbl = None
            return False, f"Erro ao preparar a divisão: {type(e).__name__}: {e}"

    def _on_cv_sugcuts(self, _):
        """Preenche a caixa de cortes com o binning ótimo desta folha."""
        sid = self._cv_node()
        feat = self._cv_feature()
        if feat is None or sid is None:
            return
        with self._busy(self.btn_cv_sugcuts, msg="sugerindo os cortes…"):
            try:
                r = self.seg.best_binning(sid, feat, max_n_bins=int(self.sl_cv_bins.max))
            except Exception as e:
                self._log(f"Não consegui sugerir cortes: {type(e).__name__}: {e}"); return
            lbl = self.seg.feature_labels.get(feat, feat)
            if r["n_bins"] < 2:
                self._log(f"Sem corte ótimo para '{lbl}' nesta folha."); return
            self.sl_cv_bins.value = max(self.sl_cv_bins.min,
                                        min(self.sl_cv_bins.max, r["n_bins"]))
            if r["kind"] == "num":
                self.tx_cv_cuts.value = ", ".join(f"{c:.4g}" for c in r["cuts"])
            else:
                # categórica: a sugestão vira a ATRIBUIÇÃO de grupo de cada
                # categoria no agrupador, não um texto — é lá que o corte é lido
                self._rebuild_cv_cat_box()
                por_cat = {str(c): g for g, grupo in enumerate(r["groups"], 1) for c in grupo}
                for c, dd in (getattr(self, "_cv_cat_widgets", None) or {}).items():
                    if str(c) in por_cat:
                        dd.value = por_cat[str(c)]
            self._log(f"Sugestão p/ '{lbl}': {r['n_bins']} faixas — cortes preenchidos.")

    def _on_cv_preview(self, _):
        """Mostra as faixas que o corte criaria — volume e alvo de cada uma —
        antes de mexer na árvore."""
        with self._busy(self.btn_cv_preview, msg="prevendo a divisão…"):
            ok, msg = self._cv_prepare()
            if not ok:
                self.out_cv_preview.value = (f"<div style='font-size:11.5px;"
                                             f"color:var(--bad-tx)'>{_esc(msg)}</div>")
                return
            tbl = self._cv_prev_tbl
            if tbl is None or tbl.empty:
                self.out_cv_preview.value = (
                    "<div style='font-size:11.5px;color:var(--warn-tx)'>Sem corte válido para "
                    "esta variável nesta folha — tente outra ou afrouxe o máx. de faixas.</div>")
                return
            self.out_cv_preview.value = self._cv_preview_html(
                tbl, self._pending["only_segments"][0])

    def _cv_preview_html(self, tbl, sid):
        """Linhas do preview — uma por faixa proposta, com barra de volume.
        A % é sobre a FOLHA (não sobre a carteira): o que importa aqui é como o
        corte reparte esta folha."""
        lo, hi = self._leaf_values()
        n_pai = int(self.seg.segments[sid]["mask"].sum())
        linhas = []
        for r in tbl.itertuples(index=False):
            rot = str(getattr(r, "faixa", ""))
            n = int(getattr(r, "n", 0) or 0)
            val = float(getattr(r, "valor_medio", float("nan")))
            share = 100 * n / n_pai if n_pai else 0.0
            cor = self._color(val, lo, hi)
            v_txt = "—" if pd.isna(val) else f"{val * 100:.2f}%"
            linhas.append(
                f"<div style='display:flex;align-items:center;gap:9px;padding:5px 0;"
                f"border-top:1px solid var(--hair)'>"
                f"<div style='flex:1;min-width:0'>"
                f"<div style='font-size:11px;font-weight:600;white-space:nowrap;overflow:hidden;"
                f"text-overflow:ellipsis' title=\"{_esc(rot)}\">{_esc(rot)}</div>"
                f"<div style='height:5px;border-radius:999px;background:var(--gauge-track);"
                f"margin-top:4px;overflow:hidden'><i style='display:block;height:100%;"
                f"width:{max(3, min(100, share)):.1f}%;background:{cor};border-radius:999px'></i>"
                f"</div></div>"
                f"<div style='flex:none;width:46px;text-align:right;font-size:10.5px;"
                f"color:var(--sub-ink)'>{share:.1f}%</div>"
                f"<div style='flex:none;width:56px;text-align:right;font-size:11.5px;"
                f"font-weight:600;color:{cor}'>{v_txt}</div></div>")
        if not linhas:
            return ("<div style='font-size:11.5px;color:var(--warn-tx)'>O corte não separou "
                    "a folha — tente outra variável ou afrouxe o mínimo por faixa.</div>")
        # monotonicidade: faixas com alvo sempre subindo (ou sempre descendo) são o
        # que se espera de uma variável de risco bem comportada; a quebra não impede
        # o corte, mas merece ser vista antes de aplicá-lo
        mono = bool(tbl.attrs.get("mono_ok", True))
        selo = (f"<span style='font-size:10px;font-weight:700;padding:1px 8px;border-radius:999px;"
                f"background:{'var(--ok-bg)' if mono else 'var(--warn-bg)'};"
                f"color:{'var(--ok-ink)' if mono else 'var(--warn-ink)'}'>"
                f"{'monotônica' if mono else 'não monotônica'}</span>")
        return (f"<div style='display:flex;align-items:center;justify-content:space-between;"
                f"margin-bottom:2px'><span style='font-size:9.5px;letter-spacing:.09em;"
                f"text-transform:uppercase;color:var(--sub-ink);font-weight:700'>"
                f"Preview · {len(linhas)} faixas</span>{selo}</div>" + "".join(linhas))

    def _on_cv_apply(self, _):
        """Aplica o corte à folha em foco — mesmo caminho (com desfazer) do
        ✂ Criar segmento da aba Construir."""
        with self._busy(self.btn_cv_apply, msg="criando os segmentos…"):
            ok, msg = self._cv_prepare()
            if not ok:
                self._log(msg); return
            sid = self._pending["only_segments"][0]
            redo_bak = list(self._redo)
            antes = self._delta_snapshot()
            self._checkpoint()
            try:
                self.seg.grow(**self._pending)
                self._pending = None
            except Exception as e:
                self._revert_checkpoint(redo_bak)
                self._log(f"Erro ao criar segmento: {type(e).__name__}: {e}"); return
            self._cv_sel = sid                 # segue no nó recém-dividido
            self._refresh()                    # já redesenha o canvas (aba à vista)
            self._cv_center()
            self._log_delta("dividir (canvas)", antes)

    # ---- regras de negócio do nó ------------------------------------
    def _on_cv_name(self, _):
        """Apelido de negócio da folha — o rótulo que vai para a régua, o Excel
        e o SQL. Imediato e desfazível, como o campo da aba Construir."""
        if self._cv_syncing:
            return
        sid = self._cv_node()
        if sid is None or not self.seg.segments[sid]["is_leaf"]:
            return
        novo = " ".join((self.tx_cv_name.value or "").split())
        if novo == (self.seg.leaf_name(sid) or ""):
            return
        redo_bak = list(self._redo)
        self._checkpoint()
        try:
            self.seg.set_leaf_name(sid, novo or None)
        except Exception as e:
            self._revert_checkpoint(redo_bak)
            self._log(f"Erro ao apelidar a folha: {type(e).__name__}: {e}"); return
        self._log(f"🏷️ apelido {'aplicado' if novo else 'removido'}: {self._leaf_label(sid)}")
        self._refresh_lock_labels()
        self._refresh_table()
        self._sync_leaf_name_field()
        self._refresh_canvas()

    def _on_cv_lock(self, _):
        sid = self._cv_node()
        if sid is None or not self.seg.segments[sid]["is_leaf"]:
            return
        if sid in self.locked:
            self.locked.discard(sid)
            self._log(f"🔓 reaberta: {self._leaf_label(sid)}")
        else:
            self.locked.add(sid)
            self._log(f"🔒 fechada: {self._leaf_label(sid)}")
        self._refresh_lock_labels()
        self._refresh_canvas()

    def _on_cv_merge(self, side):
        """Funde a folha com a irmã adjacente do lado pedido."""
        def _handler(_):
            sid = self._cv_node()
            if sid is None:
                return
            botao = self.btn_cv_merge_l if side == "left" else self.btn_cv_merge_r
            with self._busy(botao, msg="fundindo as folhas…"):
                redo_bak = list(self._redo)
                antes = self._delta_snapshot()
                self._checkpoint()
                try:
                    self.seg.merge_leaf(sid, side=side, verbose=False)
                except Exception as e:
                    self._revert_checkpoint(redo_bak)
                    self._log(f"Erro ao fundir: {type(e).__name__}: {e}"); return
                self._cv_sel = None            # o sid antigo morreu na fusão
                self._refresh()                # já redesenha o canvas (aba à vista)
                self._log_delta("fundir (canvas)", antes)
            self._sync_cv_actions()    # fora do _busy: ver o docstring de lá
        return _handler

    def _on_cv_collapse(self, _):
        """Recolhe o ramo: os filhos somem e o nó volta a ser folha divisível."""
        sid = self._cv_node()
        if sid is None or self.seg.segments[sid]["is_leaf"]:
            return
        with self._busy(self.btn_cv_collapse, msg="recolhendo o ramo…"):
            redo_bak = list(self._redo)
            antes = self._delta_snapshot()
            self._checkpoint()
            try:
                self.seg.collapse(sid, verbose=False)
            except Exception as e:
                self._revert_checkpoint(redo_bak)
                self._log(f"Erro ao recolher: {type(e).__name__}: {e}"); return
            self._cv_sel = sid
            self._refresh()                    # já redesenha o canvas (aba à vista)
            self._cv_center()
            self._log_delta("recolher (canvas)", antes)
        self._sync_cv_actions()        # fora do _busy: ver o docstring de lá

    def _cv_missing_sibling(self, sid):
        """Nó de faltantes (NaN) irmão desta folha, se o split gerou um e ele
        ainda não foi juntado — mesma condição que ``merge_missing`` usa."""
        s = self.seg.segments.get(sid)
        if s is None or not s["is_leaf"] or not s.get("conditions"):
            return None
        if s["conditions"][-1].get("kind") == "na":     # o próprio nó de faltantes
            return None
        return next((c for c, o in self.seg.segments.items()
                     if o.get("parent") == s["parent"] and o["is_leaf"] and o.get("conditions")
                     and o["conditions"][-1].get("kind") == "na"), None)

    def _on_cv_missing(self, _):
        """Traz o nó de faltantes do split para dentro desta folha: a regra da
        folha passa a ser 'faixa OU faltante'."""
        sid = self._cv_node()
        if sid is None or self._cv_missing_sibling(sid) is None:
            self._log("Esta folha não tem nó de faltantes irmão para alocar."); return
        with self._busy(self.btn_cv_missing, msg="alocando os faltantes…"):
            before = set(self.seg.segments)
            redo_bak = list(self._redo)
            antes = self._delta_snapshot()
            self._checkpoint()
            try:
                self.seg.merge_missing(sid, verbose=False)
            except Exception as e:
                self._revert_checkpoint(redo_bak)
                self._log(f"Erro ao alocar os faltantes: {type(e).__name__}: {e}"); return
            if set(self.seg.segments) == before:        # no-op: não polui o histórico
                self._revert_checkpoint(redo_bak)
                self._log("Nada a alocar — este split não tem nó de faltantes."); return
            self.locked &= set(self.seg.segments)
            self._pending = None
            novos = [i for i in self.seg.segments
                     if i not in before and self.seg.segments[i]["is_leaf"]]
            self._cv_sel = novos[0] if novos else None
            self._refresh()
            self._cv_center()
            self._log_delta("alocar faltantes (canvas)", antes)
        self._sync_cv_actions()        # fora do _busy: ver o docstring de lá

    # ---- mover o corte da divisa ------------------------------------
    def _cv_move_owner(self):
        """Folha cujo ``hi`` será movido, conforme o lado escolhido."""
        sid = self._cv_node()
        if sid is None:
            return None
        return sid if self.dd_cv_move_side.value == "dir" else self._left_cut_owner(sid)

    def _sync_cv_move(self):
        """Monta o seletor de lado com os cortes que a folha REALMENTE tem e
        espelha o corte vigente no campo. Sem corte móvel, some da tela: mover
        corte só existe entre folhas vizinhas de um split numérico."""
        sid = self._cv_node()
        valido = (sid is not None and sid in self.seg.segments
                  and self.seg.segments[sid]["is_leaf"])
        dono_esq = self._left_cut_owner(sid) if valido else None
        tem_dir = bool(self.seg.movable_cut(sid)) if valido else False
        opcoes = ([("◀ à esquerda", "esq")] if dono_esq else []) + \
                 ([("à direita ▶", "dir")] if tem_dir else [])
        self.out_cv_move.value = ""            # preview antigo era de outra folha
        self.box_cv_move.layout.display = "" if opcoes else "none"
        if not opcoes:
            return
        self._cv_syncing_side = True
        try:
            manter = self.dd_cv_move_side.value
            self.dd_cv_move_side.options = opcoes
            valores = [v for _, v in opcoes]
            self.dd_cv_move_side.value = manter if manter in valores else valores[-1]
        finally:
            self._cv_syncing_side = False
        self.dd_cv_move_side.disabled = len(opcoes) == 1
        self._render_cv_move_side()

    def _render_cv_move_side(self):
        """Corte vigente e intervalo válido do lado escolhido."""
        dono = self._cv_move_owner()
        info = self.seg.movable_cut(dono) if dono else None
        if info is None:
            self.lbl_cv_move.value = ("<div class='treeui-legend'>Corte indisponível deste "
                                      "lado.</div>")
            for w in (self.tx_cv_move, self.btn_cv_move_prev, self.btn_cv_move):
                w.disabled = True
            return
        rot = self.seg.feature_labels.get(info["feature"], info["feature"])
        lado = "à esquerda" if self.dd_cv_move_side.value == "esq" else "à direita"
        self.lbl_cv_move.value = (
            f"<div class='treeui-legend'>Corte {lado} em <b>{_esc(rot)}</b>: "
            f"<b>{_fmt(info['cut'])}</b> · válido entre {_fmt(info['lo'])} e "
            f"{_fmt(info['hi_sib'])} (exclusivo)</div>")
        self.tx_cv_move.value = info["cut"]
        for w in (self.tx_cv_move, self.btn_cv_move_prev, self.btn_cv_move):
            w.disabled = False

    def _on_cv_move_side(self, _):
        if not getattr(self, "_cv_syncing_side", False):
            self.out_cv_move.value = ""        # o preview era do outro lado
            self._render_cv_move_side()

    def _on_cv_move_preview(self, _):
        dono = self._cv_move_owner()
        if dono is None or dono not in self.seg.segments:
            self._log("Selecione uma folha com corte móvel."); return
        try:
            tbl = self.seg.preview_move_cut(dono, self.tx_cv_move.value)
        except Exception as e:
            self.out_cv_move.value = (f"<div style='font-size:11.5px;color:var(--bad-tx)'>"
                                      f"{_esc(e)}</div>")
            return
        self.out_cv_move.value = self._df_html(tbl, center=True)

    def _on_cv_move(self, _):
        dono = self._cv_move_owner()
        if dono is None or dono not in self.seg.segments:
            self._log("Selecione uma folha com corte móvel."); return
        with self._busy(self.btn_cv_move, msg="movendo o corte…"):
            before = set(self.seg.segments)
            redo_bak = list(self._redo)
            antes = self._delta_snapshot()
            self._checkpoint()
            try:
                self.seg.move_cut(dono, self.tx_cv_move.value, verbose=False)
            except Exception as e:
                self._revert_checkpoint(redo_bak)
                self._log(f"Erro ao mover o corte: {type(e).__name__}: {e}")
                self.out_cv_move.value = (f"<div style='font-size:11.5px;color:var(--bad-tx)'>"
                                          f"{_esc(e)}</div>")
                return
            if set(self.seg.segments) == before:        # corte igual ao vigente
                self._revert_checkpoint(redo_bak)
                self._log("O novo corte é igual ao vigente — nada a mudar."); return
            self.locked &= set(self.seg.segments)
            self._pending = None
            novos = [i for i in self.seg.segments
                     if i not in before and self.seg.segments[i]["is_leaf"]]
            self._cv_sel = novos[0] if novos else None
            self._refresh()
            self._cv_center()
            self._log_delta("mover corte (canvas)", antes)
        self._sync_cv_actions()        # fora do _busy: ver o docstring de lá

    def _on_cv_reset(self, b):
        """Volta à árvore vazia. Pede confirmação em dois cliques (é o botão que
        joga fora o trabalho todo) e continua desfazível pelo ↶ Desfazer."""
        def _resetar():
            with self._busy(self.btn_cv_reset, msg="reiniciando a árvore…"):
                antes = self._delta_snapshot()
                self._checkpoint()
                self.seg = TreeSegmenter(self.df, **self._kwargs)
                self.seg.fallback = self.dd_fallback.value
                self.locked.clear()
                self._pending = None
                self._cv_sel = "root"
                self._log("Árvore reiniciada.")
                self._refresh()            # já redesenha o canvas (aba à vista)
                self._on_cv_fit(None)      # e reenquadra: sobrou só a raiz
                self._log_delta("resetar (canvas)", antes)
        self._confirm_twice(b, _resetar)

    # ---- janelinha de confirmação das ações automáticas ---------------
    def _on_cv_modal_open(self, kind):
        """Abre a janelinha da ação ``kind`` no meio do canvas. Os controles são
        as MESMAS instâncias da aba Construir (segunda view do mesmo modelo do
        ipywidgets) — não há cópia de configuração para divergir. Nada roda até
        o Aplicar; Cancelar fecha sem tocar na árvore."""
        def _abrir(_):
            titulos = {"fit": "Auto-fit — crescer a árvore",
                       "merge": "Auto-fundir — juntar folhas indistinguíveis",
                       "prune": "Podar — remover folhas pouco representativas"}
            if kind == "fit":
                alvo = ("<b>apenas a folha selecionada</b>"
                        if (self._cv_node() or "root") != "root"
                        else "<b>a árvore toda</b> (raiz em foco)")
                nota = (f"Cresce {alvo} de forma gulosa por IV, até a profundidade "
                        "escolhida. As concentrações são % da carteira inteira.")
                corpo = (self.sl_depth, self.dd_criterion,
                         self.cb_autoconc_min, self.sl_autoconc_min,
                         self.cb_autoconc_max, self.sl_autoconc_max)
            elif kind == "merge":
                nota = ("Funde folhas-irmãs com alvo estatisticamente "
                        "indistinguível (p &gt; α no teste entre adjacentes) ou com "
                        "diferença abaixo do Δ mínimo. Folhas fechadas não entram.")
                corpo = (self.sl_alpha, self.sl_gap, self.cb_automerge_na)
            else:
                nota = ("Funde com a irmã as folhas com representatividade abaixo "
                        "do mínimo ou com diferença de alvo menor que o Δ mínimo. "
                        "Folhas fechadas não entram.")
                corpo = (self.sl_repr, self.sl_gap)
            self.out_cv_modal_head.value = (
                f"<div style='font-size:13px;font-weight:600;color:var(--strong-ink);"
                f"margin-bottom:3px'>{titulos[kind]}</div>"
                f"<div class='treeui-legend' style='margin:0 0 8px'>{nota} A ação é "
                f"desfazível com ↶ Desfazer.</div>")
            self.box_cv_modal_body.children = corpo
            self._cv_modal_kind = kind
            self.box_cv_modal.layout.display = ""
            if kind == "fit":                    # sliders condicionais do auto-fit
                self._sync_autoconc_visibility()
        return _abrir

    def _cv_modal_close(self):
        self.box_cv_modal.layout.display = "none"
        self.box_cv_modal_body.children = ()
        self._cv_modal_kind = None

    def _on_cv_modal_cancel(self, _):
        self._cv_modal_close()

    def _on_cv_modal_apply(self, _):
        """Fecha a janelinha e roda o handler REAL da aba Construir — mesmo
        checkpoint, mesmo desfazer, mesma linha de Δ no console."""
        kind = self._cv_modal_kind
        self._cv_modal_close()
        if kind == "fit":
            self._on_autofit(None)
        elif kind == "merge":
            self._on_automerge(None)
        elif kind == "prune":
            self._on_prune(None)

    def _sync_cv_auto_cfg(self):
        """Estado dos botões desfazer/refazer da barra do mapa."""
        self.btn_cv_undo.disabled = not self._undo
        self.btn_cv_redo.disabled = not self._redo

    def _on_cv_goto(self, ch):
        """Atalho da barra: seleciona a folha (mesmo caminho do clique no nó) e
        centraliza o canvas nela. Funciona também sem o mapa (modo offline):
        vira só o foco do painel."""
        if self._cv_goto_syncing:
            return
        idx = ch.get("new")
        if not idx:                            # None ou 0 = o rótulo de ação
            return
        sid = self.dd_cv_goto.options[idx][1]
        if sid is None or sid not in self.seg.segments:
            return
        w = self._cv_widget
        if w is not None and w.selected != sid:
            w.selected = sid                   # dispara _on_cv_select, como o clique
        else:
            self._cv_sel = sid
            if (self.seg.segments[sid]["is_leaf"] and self.dd_leaf.value != sid
                    and sid in [s for _, s in self.dd_leaf.options]):
                self.dd_leaf.value = sid       # sincroniza as outras abas
            else:
                self._refresh_cv_panel()
        self._cv_center()
        self._cv_goto_syncing = True           # o dropdown é ação: volta ao rótulo
        try:
            self.dd_cv_goto.index = 0          # index, não value — ver _sync_cv_goto_options
        finally:
            self._cv_goto_syncing = False

    def _sync_cv_goto_options(self):
        """Opções do 'ir para folha' acompanham as folhas atuais da árvore.

        O reset é pelo ``index`` (não pelo ``value``): trocar as ``options``
        deixa o index em ``None`` mesmo havendo uma opção de valor ``None`` —
        e index ``None`` é um dropdown em branco, sem o rótulo de ação."""
        self._cv_goto_syncing = True
        try:
            self.dd_cv_goto.options = ([("ir para folha…", "")]
                                       + self._ordered_leaf_options())
            self.dd_cv_goto.index = 0
        finally:
            self._cv_goto_syncing = False

    def _on_cv_scn_save(self, _):
        """Fotografa a árvore direto da barra do mapa, reusando o salvar de
        Avançado — mesma lista de cenários, um só formato de foto."""
        self.tx_scn_name.value = (self.tx_cv_scn.value or "").strip()
        self._on_scn_save(None)
        self.tx_cv_scn.value = ""

    def _on_cv_fit(self, _):
        if self._cv_widget is not None:
            self._cv_widget.fit_token += 1

    def _cv_center(self):
        """Traz o nó em foco para o centro do canvas, sem redesenhar a árvore."""
        if self._cv_widget is not None and self._cv_sel:
            self._cv_widget.center_token += 1

    # ==================================================================
    # Preview da árvore — interativo (anywidget) com fallback estático
    # ==================================================================
    def _on_tree_preview(self, _):
        """Preview da árvore na aba Construir. Com anywidget instalado a imagem
        é CLICÁVEL: clicar num nó seleciona a folha (painel Detalhe acompanha),
        hover mostra as métricas e a barra contextual funde/recolhe. Sem
        anywidget, PNG estático (comportamento anterior)."""
        try:
            if self._ensure_tree_widget():
                self.out_tree_img.value = ""
                self._refresh_tree_widget()
            elif not self.allow_interactive_tree:
                # modo offline-safe (padrão em Databricks): PNG autocontido, sem
                # nenhum download externo. A dica explica e ensina a religar o interativo.
                hint = ("<div style='font-size:11px;color:var(--sub-ink);margin-top:4px'>🔒 "
                        "preview <b>offline</b> — imagem autocontida, <b>sem download externo</b>. "
                        "No Databricks a árvore CLICÁVEL (anywidget) fica desligada por padrão "
                        "porque seu frontend seria buscado de um CDN (trava em cluster sem "
                        "egress). Para religá-la num ambiente com egress/anywidget instalado: "
                        "<code>TreeSegmenterUI(..., allow_interactive_tree=True)</code>.</div>")
                self.out_tree_img.value = self._fig_html(self.seg.plot_tree(),
                                                         border=True) + hint
            else:
                hint = ("<div style='font-size:11px;color:var(--sub-ink);margin-top:4px'>💡 instale "
                        "<code>anywidget</code> (<code>pip install anywidget</code>) para uma "
                        "árvore CLICÁVEL: selecionar a folha, fundir irmãs e recolher ramos "
                        "direto na imagem, com métricas no hover.</div>")
                self.out_tree_img.value = self._fig_html(self.seg.plot_tree(),
                                                         border=True) + hint
        except Exception as e:
            self.out_tree_img.value = (f"<div style='color:var(--bad-tx);font-size:12px'>Erro ao "
                                       f"desenhar a árvore: {type(e).__name__}: {e}</div>")

    def _on_tree_preview_hide(self, _):
        """Oculta o preview (estático ou interativo) e a sua barra."""
        self.out_tree_img.value = ""
        self.tree_img_bar.layout.display = "none"
        if self._tree_img_visible():
            self.box_tree_img.children = (self.tree_img_bar, self.out_tree_img)

    def check_tree_preview_offline(self, verbose: bool = True) -> dict:
        """Valida que **"Ver árvore" não baixa nenhuma fonte/arquivo externo**.

        Feito para rodar no **Databricks** e tirar a dúvida na hora: monta o preview
        do jeito que o botão montaria e confere que o HTML só referencia recursos
        EMBUTIDOS (``data:``) — sem ``http(s)``, ``@import`` ou ``url()`` de rede.
        Varre também o JS/CSS do widget interativo. Reporta o modo vigente
        (estático autocontido × interativo), se o ambiente é Databricks e se a
        garantia offline vale.

        Só o modo **estático** é 100% offline; o **interativo** (anywidget) pode ter
        o frontend buscado de um CDN pelo gerenciador de widgets do Databricks — por
        isso ``offline_guaranteed`` só é ``True`` no modo estático. Devolve um dict
        e, com ``verbose``, imprime um resumo amigável.

            ui.check_tree_preview_offline()      # rode numa célula do Databricks
        """
        import re
        in_dbx = _running_in_databricks()
        interactive = bool(self.allow_interactive_tree) and _tree_image_widget_cls() is not None
        err = None
        try:                                  # HTML que o botão injeta (imagem data-URL)
            static_html = self._fig_html(self.seg.plot_tree(), border=True)
        except Exception as e:
            static_html, err = "", f"{type(e).__name__}: {e}"
        # procura QUALQUER referência de rede no HTML estático + no JS/CSS do widget
        # interativo: http(s)://, @import ... e url(...) que não seja data:
        blob = static_html + _TREE_IMG_ESM + _TREE_IMG_CSS
        ext = sorted(
            set(re.findall(r"https?://[^\s\"')]+", blob))
            | {m.strip() for m in re.findall(r"@import[^;{]+", blob)}
            | {u for u in re.findall(r"url\(\s*['\"]?([^)'\"]+)", blob)
               if not u.strip().lower().startswith("data:")}
        )
        report = {
            "in_databricks": in_dbx,
            "mode": "interativo (anywidget)" if interactive else "estático (autocontido)",
            "allow_interactive_tree": bool(self.allow_interactive_tree),
            "external_refs": ext,
            "no_external_refs": not ext,
            "offline_guaranteed": (not ext) and (not interactive),
        }
        if err:
            report["plot_error"] = err
        if verbose:
            ok = "✅" if report["offline_guaranteed"] else ("⚠️" if not ext else "❌")
            print(f"{ok} 'Ver árvore' — verificação de download externo")
            print(f"   ambiente Databricks : {'sim' if in_dbx else 'não'}")
            print(f"   modo do preview     : {report['mode']}")
            print(f"   referências externas: {ext or 'nenhuma (só data:)'}")
            if report["offline_guaranteed"]:
                print("   → OFFLINE garantido: nada é baixado ao ver a árvore.")
            elif interactive:
                print("   → modo interativo: o frontend do anywidget pode vir de um CDN no "
                      "Databricks. Para garantia total, use allow_interactive_tree=False "
                      "(já é o padrão dentro do Databricks).")
            if err:
                print(f"   (obs.: não consegui desenhar a árvore agora — {err})")
        return report

    def _tree_img_visible(self):
        w = self._tree_img_widget
        return w is not None and w in getattr(self.box_tree_img, "children", ())

    def _on_tree_zoom(self, ch):
        """Espelha o slider no trait do widget — o front só reescala a imagem já
        renderizada (nenhum replot no Python)."""
        if self._tree_img_widget is not None:
            self._tree_img_widget.zoom = float(ch["new"])

    def _ensure_tree_widget(self):
        """Garante o widget interativo montado no card (instancia 1× e reusa).
        Devolve False quando o preview interativo está desligado
        (``allow_interactive_tree=False`` — padrão em Databricks, evita o CDN do
        anywidget) ou quando o anywidget não está instalado; nos dois casos a UI
        cai no PNG estático AUTOCONTIDO (data-URL, sem nenhum download externo)."""
        if not self.allow_interactive_tree:      # modo offline-safe (sem rede/CDN)
            return False
        if self._tree_img_widget is None:
            cls = _tree_image_widget_cls()
            if cls is None:
                return False
            self._tree_img_widget = cls()
            self._tree_img_widget.observe(self._on_tree_img_select, names="selected")
            self._tree_img_widget.zoom = float(self.sl_tree_zoom.value)
        if not self._tree_img_visible():
            self.box_tree_img.children = (self.tree_img_bar, self._tree_img_widget,
                                          self.out_tree_img)
        return True

    def _refresh_tree_widget(self):
        """(Re)renderiza o preview interativo: PNG + hit-map + tooltips + seleção."""
        w = self._tree_img_widget
        if w is None:
            return
        import base64
        # dpi 150 (não os 110 das prévias estáticas): o PNG natural maior deixa
        # o preview MAIS ALTO — o front garante min_height e nunca amplia além
        # do natural, então mais pixels = exibição maior e ainda nítida
        data = self.seg.plot_tree_hitmap(dpi=150)
        nodes = [dict(sid=sid, tooltip=self._tree_node_tooltip(sid), **box)
                 for sid, box in data["nodes"].items()]
        sel = self._img_selected if self._img_selected in data["nodes"] else self.dd_leaf.value
        sel = sel if sel in data["nodes"] else ""
        self._img_selected = sel or None
        with w.hold_sync():                    # 1 mensagem só pelo comm
            w.src = "data:image/png;base64," + base64.b64encode(data["png"]).decode("ascii")
            w.width, w.height = data["width"], data["height"]
            w.nodes = nodes
            w.selected = sel
        self._refresh_img_bar()

    def _tree_node_tooltip(self, sid):
        """HTML curto com as métricas do nó (hover do preview interativo)."""
        import html as _html
        s = self.seg.segments[sid]
        nota_map, _ = self.seg._grade_map()
        n = int(s["mask"].sum())
        rep = 100 * n / len(self.df) if len(self.df) else 0.0
        ref = self.ref_sample if self.sample_col is not None else None
        v = self._node_value(sid, ref)
        desc = ("TODA A CARTEIRA" if s["parent"] is None
                else self.seg._descrever(s["conditions"]))     # caminho completo
        head = (f"folha {nota_map.get(sid, '?')}" if s["is_leaf"]
                else ("raiz" if s["parent"] is None else "ramo"))
        v_txt = "—" if pd.isna(v) else f"{v * 100:.2f}%"
        linhas = [f"<b>{_html.escape(desc)}</b>",
                  f"{head} · n {f'{n:,}'.replace(',', '.')} · repr. {rep:.1f}%",
                  f"{self._risk_label}{' (' + str(ref) + ')' if ref else ''}: {v_txt}"]
        if self.sample_col is not None:
            partes = []
            for a in [self.ref_sample] + list(getattr(self, "_pd_nonref", [])):
                va = self._node_value(sid, a)
                if not pd.isna(va):
                    partes.append(f"{_html.escape(str(a))} {va * 100:.2f}%")
            if partes:
                linhas.append(" · ".join(partes))
        linhas.append("<span style='color:#9fb0c6'>clique para selecionar</span>")
        return "<br/>".join(linhas)

    def _on_tree_img_select(self, change):
        """Clique num nó da imagem (via trait ``selected`` do widget)."""
        sid = change.get("new") or None
        if sid is None or sid not in self.seg.segments:
            self._img_selected = None          # clique fora dos nós: limpa a barra
            self._refresh_img_bar()
            return
        self._img_selected = sid
        if self.seg.segments[sid]["is_leaf"] and self.dd_leaf.value != sid:
            if sid in [s for _, s in self.dd_leaf.options]:
                self.dd_leaf.value = sid       # dispara o painel Detalhe (folha ativa)
        self._refresh_img_bar()

    def _sync_img_selection(self):
        """Espelha a folha ativa (dropdown) no preview interativo, se aberto."""
        w = self._tree_img_widget
        sid = self.dd_leaf.value
        if w is None or sid is None or sid not in self.seg.segments:
            return
        self._img_selected = sid
        if w.selected != sid:
            w.selected = sid
        self._refresh_img_bar()

    def _refresh_img_bar(self):
        """Barra contextual do preview: chip do nó clicado + ações válidas."""
        import html as _html
        sid = self._img_selected
        if (self._tree_img_widget is None or sid is None
                or sid not in self.seg.segments or not self._tree_img_visible()):
            self.tree_img_bar.layout.display = "none"
            return
        s = self.seg.segments[sid]
        is_leaf, is_root = bool(s["is_leaf"]), s["parent"] is None
        nota_map, _ = self.seg._grade_map()
        desc = ("TODA A CARTEIRA" if is_root
                else self.seg._descrever([s["conditions"][-1]]))
        v = self._node_value(sid, self.ref_sample if self.sample_col is not None else None)
        v_txt = "—" if pd.isna(v) else f"{v * 100:.2f}%"
        icone = "🍃" if is_leaf else ("🌳" if is_root else "🌿")
        head = (f"folha {nota_map.get(sid, '?')}" if is_leaf
                else ("raiz" if is_root else "ramo"))
        lock_txt = "🔒 " if sid in self.locked else ""
        self.tree_img_info.value = (f"<span class='treeui-imgchip'>{icone} {lock_txt}{head} · "
                                    f"{_html.escape(desc)} · {self._risk_label} {v_txt}</span>")
        for b in (self.btn_img_merge_l, self.btn_img_merge_r,
                  self.btn_img_merge_na, self.btn_img_lock):
            b.disabled = not is_leaf           # fundir/travar só em folha
        self.btn_img_collapse.disabled = is_root
        self.btn_img_collapse.description = ("Recolher quebra (pai)" if is_leaf
                                             else "Recolher ramo")
        if sid in self.locked:
            self.btn_img_lock.description, self.btn_img_lock.icon = "Destravar", "unlock"
        else:
            self.btn_img_lock.description, self.btn_img_lock.icon = "Travar", "lock"
        self.tree_img_bar.layout.display = "flex"

    def _on_img_collapse(self, _):
        """Recolher a partir do nó clicado na imagem: folha → recolhe o PAI
        (fluxo padrão); ramo → recolhe o próprio ramo numa folha só."""
        sid = self._img_selected
        if sid is None or sid not in self.seg.segments:
            return
        if self.seg.segments[sid]["is_leaf"]:
            self._on_collapse(None)            # opera na folha ativa (dd_leaf)
            return
        if self.seg.segments[sid]["parent"] is None:
            self._log("A raiz não pode ser recolhida — use Resetar para zerar a árvore.")
            return
        antes = self._delta_snapshot()
        self._checkpoint()
        self.seg.collapse(sid)
        self.locked &= set(self.seg.segments)
        self._pending = None
        self._refresh()
        folhas = [s for s, seg in self.seg.segments.items() if seg["is_leaf"]]
        if sid in folhas:
            self.dd_leaf.value = sid           # o ramo recolhido virou a folha ativa
        self._log_delta("recolher ramo", antes)

    def _on_img_lock(self, _):
        """Trava/destrava a folha clicada como final (alterna 🔒/🔓)."""
        sid = self._img_selected
        if sid is None or sid not in self.seg.segments \
                or not self.seg.segments[sid]["is_leaf"]:
            return
        if sid in self.locked:
            self._on_unlock(None)              # operam na folha ativa (dd_leaf),
        else:                                  # sincronizada com o clique na imagem
            self._on_lock(None)
        self._refresh_img_bar()                # atualiza rótulo Travar/Destravar + chip

    def _ipython_display_(self):
        display(self.panel)

    def display(self):
        display(self.panel)
