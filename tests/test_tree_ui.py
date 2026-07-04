"""
Testes da UI unificada ``TreeSegmenterUI`` (ipywidgets), parametrizados nos dois
``task_type``. Cobrem: construção, preview→split, desfazer/refazer/auto-merge,
save/load JSON, merge de faltantes, invalidação de preview ao trocar seleção,
plot da árvore, placar de saúde e os botões de plot específicos por tarefa.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

TASKS = ["classification", "regression"]


def make_df(task, n=5000, seed=0, com_na=False):
    rng = np.random.default_rng(seed)
    x = rng.beta(2.5, 3, n) * 1.4 + 0.3
    gar = rng.choice(list("ABCD"), n, p=[0.5, 0.22, 0.18, 0.1]).astype(object)
    if com_na:
        x[rng.random(n) < 0.08] = np.nan
        gar[rng.random(n) < 0.06] = np.nan
    lg = {"A": 0.0, "B": 0.10, "C": 0.16, "D": 0.30}
    risco = 0.1 + 0.4 * np.nan_to_num(x - 0.5, nan=0.35) + np.array([lg.get(g, 0.2) for g in gar])
    meses = pd.date_range("2023-01-01", periods=10, freq="MS")
    if task == "classification":
        target = (rng.uniform(0, 1, n) < np.clip(risco, 0.01, 0.95)).astype(float)
    else:
        target = np.clip(risco + rng.normal(0, 0.07, n), 0, 1)
    df = pd.DataFrame({"score": x, "garantia": gar, "target": target})
    df["dt_ref"] = rng.choice(meses, size=n)
    df["amostra"] = np.where(df["dt_ref"] >= meses[7], "OOT", "DES")
    return df


def _build(task, **kw):
    pytest.importorskip("ipywidgets")
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.tree import TreeSegmenterUI
    df = kw.pop("df", None)
    if df is None:
        df = make_df(task, **{k: kw.pop(k) for k in ("n", "seed", "com_na") if k in kw})
    with contextlib.redirect_stdout(io.StringIO()):
        return TreeSegmenterUI(df, target="target", task_type=task, sample_col="amostra",
                               ref_sample="DES", date_col="dt_ref", **kw)


@pytest.fixture(params=TASKS)
def task(request):
    return request.param


def _nleaf(ui):
    return sum(s["is_leaf"] for s in ui.seg.segments.values())


def test_ui_constroi_e_expoe_task_type(task):
    ui = _build(task)
    assert ui.task_type == task and ui.seg.task_type == task


def test_ui_banner_titulo_por_task(task):
    import re
    ui = _build(task)
    # localiza o banner entre os filhos do painel (após a topbar do tema)
    html = next(c.value for c in ui.panel.children
                if hasattr(c, "value") and "treeui-banner" in (c.value or ""))
    titulo = re.search(r"class='t'>([^<]+)<", html).group(1)
    esperado = "Segmentação de PD" if task == "classification" else "Segmentação de LGD"
    assert titulo == esperado
    assert ui._risk_label == ("PD" if task == "classification" else "LGD")


def test_ui_leaf_hist_por_task(task):
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)                             # dispara _refresh -> _refresh_leaf_hist
    assert "não gerado" not in ui.out_leaf_hist.value    # reg usa plot_leaf_value_hist, clf badrate


def test_ui_preview_split(task):
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_leaf.value = "root"
        ui.dd_feature.value = "score"
        ui.tg_mode.value = "Manual"
        ui.tx_cuts.value = "0.8"
        ui._on_preview(None)
        assert ui._pending is not None
        ui._on_split(None)
    assert _nleaf(ui) >= 2


def test_ui_preview_invalidado_ao_trocar_selecao(task):
    """Regressão (bug _pending): trocar a variável após o Preview invalida o
    split pendente (não cresce na seleção antiga)."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_leaf.value = "root"
        ui.dd_feature.value = "score"
        ui.tg_mode.value = "Manual"
        ui.tx_cuts.value = "0.8"
        ui._on_preview(None)
        assert ui._pending is not None
        ui.dd_feature.value = "garantia"          # troca de variável
    assert ui._pending is None
    assert ui.out_preview_seg.value == ""


def test_ui_undo_redo_automerge_json(task, tmp_path):
    ui = _build(task, n=8000, seed=5)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_leaf.value = "root"
        ui.dd_feature.value = "score"
        ui.tg_mode.value = "Manual"
        ui.tx_cuts.value = "0.6,0.8,1.0,1.2"
        ui._on_preview(None); ui._on_split(None)
        n_split = _nleaf(ui)
        ui._on_undo(None); n_undo = _nleaf(ui)
        ui._on_redo(None); n_redo = _nleaf(ui)
        ui._on_automerge(None)
        p = str(tmp_path / "arvore.json")
        ui.tx_json_path.value = p
        ui._on_save_json(None); n_saved = _nleaf(ui)
        ui._on_reset(None); n_reset = _nleaf(ui)
        ui._on_load_json(None); n_loaded = _nleaf(ui)
    assert n_split >= 2 and n_undo == 1 and n_redo == n_split
    assert n_reset == 1 and n_loaded == n_saved


def test_ui_iv_refresh_e_psi_oot(task):
    """O botão 'Atualizar' do card de IV calcula o IV/PSI por variável da folha
    SEM abrir a aba de variáveis; e a tabela traz o PSI do OOT além do pior caso."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.btn_iv_refresh.click()          # calcula na raiz, sem abrir a aba de variáveis
    html = ui.out_iv.value
    assert "<table" in html.lower() and "iv" in html.lower()
    assert "psi OOT" in html               # coluna do PSI no OOT
    assert "pior caso" in html             # hint menciona OOT + pior caso
    assert "max-content" in html           # tabela transborda → scroller horizontal


def test_ui_iv_psi_estabilidade(task):
    """O ranking de IV traz uma coluna de PSI por amostra de validação, incluindo
    a safra de ESTABILIDADE (além de OOT e do pior caso)."""
    df = make_df(task, n=6000, seed=4)
    idx = df.sample(frac=0.15, random_state=1).index   # ~15% vira safra de estabilidade
    df.loc[idx, "amostra"] = "ESTABILIDADE"
    ui = _build(task, df=df)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.btn_iv_refresh.click()
    html = ui.out_iv.value
    assert "psi OOT" in html and "psi ESTAB" in html


def test_ui_undo_redo_restaura_folha(task):
    """Desfazer/refazer volta à folha que estava selecionada naquele estado."""
    ui = _build(task, n=6000, seed=11)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_leaf.value = "root"
        ui.dd_feature.value = "score"
        ui.tg_mode.value = "Manual"
        ui.tx_cuts.value = "0.8"
        ui._on_preview(None); ui._on_split(None)
        alvo = [s for s, v in ui.seg.segments.items() if v["is_leaf"]][0]
        # seleciona explicitamente `alvo` e o divide (após o split ele deixa de ser folha)
        ui.dd_leaf.value = alvo
        ui.dd_feature.value = "score"
        ui.tg_mode.value = "Manual"
        ui.tx_cuts.value = "0.6"
        ui._on_preview(None); ui._on_split(None)
        sel_pos = ui.dd_leaf.value         # seleção após o 2º split
        ui._on_undo(None); sel_undo = ui.dd_leaf.value
        ui._on_redo(None); sel_redo = ui.dd_leaf.value
    assert sel_undo == alvo                # desfazer volta à folha dividida
    assert sel_redo == sel_pos             # refazer volta à seleção pós-split
    assert alvo != sel_pos                 # garante que o teste é significativo


def test_ui_merge_missing(task):
    ui = _build(task, com_na=True, n=5000, seed=7)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_leaf.value = "root"
        ui.dd_feature.value = "score"
        ui.tg_mode.value = "Manual"
        ui.tx_cuts.value = "1.0"
        ui._on_preview(None); ui._on_split(None)
        # há nó de faltantes
        assert any(v["is_leaf"] and v["conditions"][-1]["kind"] == "na"
                   for v in ui.seg.segments.values())


def _has_anywidget():
    try:
        import anywidget  # noqa: F401
        return True
    except Exception:
        return False


def test_ui_plot_tree(task):
    """Preview da árvore: com anywidget vira o widget CLICÁVEL (hit-map completo);
    sem, cai no PNG estático com a dica de instalação."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui._on_tree_preview(None)
    if _has_anywidget():
        w = ui._tree_img_widget
        assert w is not None and ui._tree_img_visible()
        assert w.src.startswith("data:image/png;base64,")
        assert w.width > 0 and w.height > 0
        assert w.min_height >= 400          # exibição alta (o front garante o mínimo)
        assert {n["sid"] for n in w.nodes} == set(ui.seg.segments)
        assert all(n["tooltip"] for n in w.nodes)
    else:
        assert ui.out_tree_img.value and "img" in ui.out_tree_img.value.lower()
        assert "anywidget" in ui.out_tree_img.value


def test_ui_plot_tree_fallback_sem_anywidget(task, monkeypatch):
    """Sem anywidget o preview mantém o comportamento anterior (PNG estático)."""
    from yggdrasil.credit_risk.tree import ui as ui_mod
    monkeypatch.setattr(ui_mod, "_tree_image_widget_cls", lambda: None)
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui._on_tree_preview(None)
    assert not ui._tree_img_visible()
    assert "img" in ui.out_tree_img.value.lower()
    assert "anywidget" in ui.out_tree_img.value          # dica de instalação


@pytest.mark.skipif(not _has_anywidget(), reason="requer anywidget")
def test_ui_tree_img_clique_seleciona_e_barra(task):
    """Clicar num nó da imagem (trait ``selected``) sincroniza a folha ativa e a
    barra contextual; nó interno desabilita fusões; raiz desabilita recolher."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui._on_tree_preview(None)
    w = ui._tree_img_widget
    folhas = [n["sid"] for n in w.nodes if n["is_leaf"]]
    outra = next(s for s in folhas if s != ui.dd_leaf.value)
    w.selected = outra                       # simula o clique do front
    assert ui.dd_leaf.value == outra         # painel Detalhe segue o clique
    assert ui.tree_img_bar.layout.display == "flex"
    assert not ui.btn_img_merge_l.disabled and not ui.btn_img_collapse.disabled
    assert not ui.btn_img_suggest.disabled and not ui.btn_img_lock.disabled
    w.selected = "root"                      # raiz: só o chip; nada de agir
    assert ui.dd_leaf.value == outra         # nó interno não muda a folha ativa
    assert ui.btn_img_merge_l.disabled and ui.btn_img_collapse.disabled
    assert ui.btn_img_suggest.disabled and ui.btn_img_lock.disabled
    w.selected = ""                          # clique fora dos nós esconde a barra
    assert ui.tree_img_bar.layout.display == "none"


@pytest.mark.skipif(not _has_anywidget(), reason="requer anywidget")
def test_ui_tree_img_acoes_e_refresh(task):
    """As ações da barra mutam a árvore e o preview interativo re-renderiza
    sozinho (hit-map novo); dropdown → imagem também sincroniza a seleção."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui._on_tree_preview(None)
    w = ui._tree_img_widget
    n_antes = _nleaf(ui)
    assert n_antes >= 2
    w.selected = next(n["sid"] for n in w.nodes if n["is_leaf"])
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_img_collapse(None)            # folha → recolhe a quebra do pai
    assert _nleaf(ui) < n_antes
    assert {n["sid"] for n in w.nodes} == set(ui.seg.segments)   # re-render pós-mutação
    # sincronismo inverso: trocar a folha no dropdown move o contorno da imagem
    if _nleaf(ui) >= 2:
        alvo = next(s for _, s in ui.dd_leaf.options if s != ui.dd_leaf.value)
        with contextlib.redirect_stdout(io.StringIO()):
            ui.dd_leaf.value = alvo
        assert w.selected == alvo


def _widgets_de(container):
    """Percorre a árvore de widgets de um container e devolve as instâncias."""
    achados, fila = set(), list(container.children)
    while fila:
        wdg = fila.pop()
        achados.add(wdg)
        fila.extend(getattr(wdg, "children", ()))
    return achados


def test_ui_split_panel_espelha_card_da_aba(task):
    """O painel de divisão do preview contém as MESMAS instâncias do card
    'Dividir a folha' da aba (2ª view sincronizada) e a barra traz os clones de
    desfazer/refazer/auto-fit/resetar. Independe do anywidget."""
    ui = _build(task)
    painel = _widgets_de(ui.tree_img_split)
    for wdg in (ui.dd_leaf, ui.dd_feature, ui.tg_mode, ui.sl_bins,
                ui.dd_split_criterion, ui.tx_cuts, ui.cat_box, ui.btn_sugcuts,
                ui.btn_preview, ui.btn_split, ui.out_preview_seg,
                ui.btn_img_split_close):
        assert wdg in painel, f"widget ausente do painel de divisão: {wdg!r}"
    barra = _widgets_de(ui.tree_img_bar)
    for wdg in (ui.btn_img_suggest, ui.btn_img_lock,
                ui.btn_img_merge_l, ui.btn_img_merge_r, ui.btn_img_merge_na,
                ui.btn_img_collapse, ui.btn_img_undo, ui.btn_img_redo,
                ui.btn_img_autofit, ui.btn_img_reset):
        assert wdg in barra, f"widget ausente da barra do preview: {wdg!r}"
    assert not any(getattr(w, "description", "") == "Dividir…" for w in barra)
    # clones compactos (lado a lado): largura auto, sem o width 98% dos cards
    for wdg in (ui.btn_img_undo, ui.btn_img_redo, ui.btn_img_autofit, ui.btn_img_reset):
        assert wdg.layout.width == "auto"
    # habilitação de desfazer/refazer espelhada dos originais (dlink)
    assert ui.btn_img_undo.disabled and ui.btn_img_redo.disabled   # sem histórico
    # respiro entre os grupos fundir-irmãs · fundir-missing · recolher
    assert ui.btn_img_merge_na.layout.margin.endswith("18px")
    assert ui.btn_img_collapse.layout.margin.endswith("18px")
    # fechado por padrão; 'Sugerir quebra' abre, o 'Fechar' do painel fecha
    assert ui.tree_img_split.layout.display == "none"
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_img_suggest(None)
    assert ui.tree_img_split.layout.display == "flex"
    ui._on_split_panel_close(None)
    assert ui.tree_img_split.layout.display == "none"


def test_ui_grow_pelo_preview(task):
    """Crescer a árvore A PARTIR do preview: 'Sugerir quebra' abre o painel
    compacto e o fluxo variável→cortes→preview→criar segmento funciona idêntico
    ao da aba (mesmos modelos de widget). Funciona com e sem anywidget."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_tree_preview(None)            # abre o preview (interativo ou estático)
        ui._on_img_suggest(None)             # sugere quebra e abre o painel
        assert ui.tree_img_split.layout.display == "flex"
        ui.dd_leaf.value = "root"
        ui.dd_feature.value = "score"
        ui.tg_mode.value = "Manual"
        ui.tx_cuts.value = "0.8"
        ui._on_preview(None); ui._on_split(None)
    assert _nleaf(ui) >= 2
    assert not ui.btn_img_undo.disabled      # split criou histórico → clone habilita
    if _has_anywidget():                     # imagem re-renderizada com a árvore nova
        assert {n["sid"] for n in ui._tree_img_widget.nodes} == set(ui.seg.segments)


@pytest.mark.skipif(not _has_anywidget(), reason="requer anywidget")
def test_ui_tree_img_lock_e_suggest(task):
    """Travar/destravar e sugerir quebra direto da barra do preview."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui._on_tree_preview(None)
    w = ui._tree_img_widget
    folha = next(n["sid"] for n in w.nodes if n["is_leaf"])
    w.selected = folha
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_img_lock(None)                # trava
    assert folha in ui.locked
    assert ui.btn_img_lock.description == "Destravar"
    assert "🔒" in ui.tree_img_info.value    # chip reflete o cadeado
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_img_lock(None)                # destrava
    assert folha not in ui.locked
    assert ui.btn_img_lock.description == "Travar"
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_img_suggest(None)             # sugere variável e abre o painel
    assert ui.tree_img_split.layout.display == "flex"


def test_ui_diag_teste_des_oot(task):
    """A tabela de folhas (Diagnóstico) traz, ao lado de p (irmãs), o teste de
    aderência da estimativa comparando DES × OOT por folha."""
    ui = _build(task, n=8000, seed=3)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)                 # cria folhas -> _refresh_table
    html = ui.out_table.value
    assert "p (irmãs)" in html
    assert "p (DES×OOT)" in html        # nova coluna de aderência DES×OOT
    # a coluna também sai no TSV copiável (Excel)
    assert "p (DES×OOT)" in ui._leaves_tsv().splitlines()[0]


def test_ui_diag_scorecard(task):
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui._on_diag(None)
    d = ui.out_diag.value
    assert d and "Erro" not in d
    assert ("AUC" in d) if task == "classification" else ("R²" in d)


def test_ui_avancado_suggest_importance_sql(task):
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_suggest3(None)                       # TOP3 na raiz
        assert "iv" in ui.out_suggest.value.lower()
        ui._on_autofit(None)
        ui._on_importance(None)
        ui._on_sql(None)
    assert ui.out_importance.value and "Erro" not in ui.out_importance.value
    assert "CASE" in ui.out_sql.value and "WHEN" in ui.out_sql.value


def test_ui_criterio_de_split(task):
    crit = "gini" if task == "classification" else "variance"
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_criterion.value = crit
        ui._on_autofit(None)
    assert ui.seg.task_type == task
    assert sum(s["is_leaf"] for s in ui.seg.segments.values()) >= 2


def test_ui_sugerir_cortes_preenche_controles(task):
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        leaf = [s for s, v in ui.seg.segments.items() if v["is_leaf"]][0]
        ui.dd_leaf.value = leaf
        ui.dd_feature.value = "score"
        ui._on_suggest_cuts(None)
    # numérica: ou preencheu os cortes, ou ajustou o máx. bins p/ a sugestão
    assert ui.tx_cuts.value != "" or ui.sl_bins.value >= 2


def test_ui_importancia_colorida_com_dicionario(task):
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui._on_importance(None)
    assert "rgb(" in ui.out_importance.value                  # cor por importância (gradiente) na tabela
    assert "<img" in ui.out_importance_chart.value            # gráfico de importância relativa ao lado
    assert "O que é a importância" in ui.out_importance_legend.value   # dicionário (abaixo)


def test_ui_diag_explica_calibracao(task):
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui._on_diag(None)
    assert "O que é calibração" in ui.out_diag.value


def test_ui_tema_escuro(task):
    ui = _build(task)
    ui.cb_dark.value = True
    assert "dark" in ui.panel._dom_classes
    ui.cb_dark.value = False
    assert "dark" not in ui.panel._dom_classes


def test_ui_keepalive_toggle(task):
    """O toggle de keepalive existe e, sem Spark ativo, se auto-reverte (no-op)."""
    ui = _build(task)
    assert hasattr(ui, "cb_keepalive")
    with contextlib.redirect_stdout(io.StringIO()):
        ui.cb_keepalive.value = True
    # fora do Databricks/Spark: o toggle volta para False e nada fica rodando
    if not (ui._keepalive and ui._keepalive.has_spark()):
        assert ui.cb_keepalive.value is False
        assert ui._keepalive is None or ui._keepalive.running is False


def test_ui_relatorio_pdf(task, tmp_path):
    ui = _build(task)
    p = str(tmp_path / "rel.pdf")
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui.tx_pdf_path.value = p
        ui._on_pdf(None)
    import os
    assert os.path.exists(p) and os.path.getsize(p) > 1000
    assert "Erro" not in ui.out_pdf.value


def test_ui_diff_de_arvores(task, tmp_path):
    from yggdrasil.credit_risk.tree import TreeSegmenter
    df = make_df(task, n=5000, seed=3)
    b = TreeSegmenter(df, target="target", task_type=task, sample_col="amostra",
                      ref_sample="DES", verbose=False)
    b.fit_auto(max_depth=1, verbose=False)
    p = str(tmp_path / "treeB.json")
    b.save(p)
    ui = _build(task, df=df)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui.tx_diff_path.value = p
        ui._on_diff(None)
    assert "Concord" in ui.out_diff.value and "Erro" not in ui.out_diff.value


def test_ui_plots_especificos_por_task(task):
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        # botão 1: clf = ROC · reg = boxplot por folha — ambos renderizam em
        # out_discrim. O botão 2 (KS) só existe na classificação (o histograma
        # do alvo foi removido da UI de regressão).
        ui.btn_roc.click()
        if ui._is_clf:
            ui.btn_ks.click()
    assert ui.out_discrim.value and "Erro" not in ui.out_discrim.value


def test_ui_overwrite_pede_confirmacao(task, tmp_path):
    """Salvar (JSON) num caminho que já existe NÃO sobrescreve direto: abre a
    janela de confirmação e só grava quando o usuário confirma (do_save)."""
    import json
    import os
    ui = _build(task, n=6000, seed=2)
    p = str(tmp_path / "arvore.json")
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)                       # árvore com várias folhas
        ui.tx_json_path.value = p
        ui._on_save_json(None)                     # não existe -> salva direto
    assert os.path.exists(p)
    with open(p, encoding="utf-8") as f:
        antes = json.load(f)
    # altera a árvore e tenta salvar de novo no MESMO caminho (já existe)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_reset(None)                         # volta para a raiz (1 folha)
        ui._on_save_json(None)                     # existe -> aguarda confirmação
    with open(p, encoding="utf-8") as f:
        depois = json.load(f)
    assert depois == antes                         # não sobrescreveu sem confirmar
    # confirmar (o que o botão 'Sobrescrever' faz) grava de fato
    with contextlib.redirect_stdout(io.StringIO()):
        ui._do_save_json(p)
    with open(p, encoding="utf-8") as f:
        confirmado = json.load(f)
    assert confirmado != antes


def test_ui_confirm_overwrite_gate(task, tmp_path):
    """O gate executa do_save direto quando o arquivo (.json/.png) não existe e o
    adia (aguardando o clique em 'Sobrescrever') quando já existe."""
    ui = _build(task)
    p = str(tmp_path / "img.png")
    chamadas = []
    ui._confirm_overwrite(p, lambda: chamadas.append(1))   # não existe -> executa
    assert chamadas == [1]
    open(p, "w").close()                                   # passa a existir
    chamadas.clear()
    ui._confirm_overwrite(p, lambda: chamadas.append(1))   # existe -> não executa
    assert chamadas == []
