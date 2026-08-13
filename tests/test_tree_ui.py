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
    # sem problem_label, o rótulo dos gráficos é o nome da coluna alvo (nunca "PD"/"LGD")
    assert titulo == "Segmentação de target"
    assert ui._risk_label == "target"


def test_ui_problem_label_sobrescreve_rotulo(task):
    import re
    ui = _build(task, problem_label="Risco")
    html = next(c.value for c in ui.panel.children
                if hasattr(c, "value") and "treeui-banner" in (c.value or ""))
    titulo = re.search(r"class='t'>([^<]+)<", html).group(1)
    assert titulo == "Segmentação de Risco"
    assert ui._risk_label == "Risco"


def test_ui_leaf_hist_por_task(task):
    """Com a Construir OCULTA, o card do histograma da folha nunca renderiza
    (fica pendente, sem custo) — o alvo da folha vive nas Distribuições do
    mapa. O plot em si continua correto por task_type (coberto lá)."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
    assert ui._hist_dirty                                # marcado, nunca desenhado
    assert "não gerado" not in ui.out_leaf_hist.value    # e sem erro renderizado


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


def test_ui_apelido_da_folha_com_undo(task):
    """Campo 'Apelido' do cartão da folha: aplica imediatamente ao segmentador,
    aparece na árvore HTML/dropdown e entra no _checkpoint (undo/redo restauram)."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_leaf.value = "root"
        ui.dd_feature.value = "score"
        ui.tg_mode.value = "Manual"
        ui.tx_cuts.value = "0.8"
        ui._on_preview(None); ui._on_split(None)
        sid = ui.dd_leaf.value                    # folha em foco após o split
        assert ui.seg.segments[sid]["is_leaf"]
        ui.tx_leaf_name.value = "Fatia premium"   # aplicação imediata (observer)
        assert ui.seg.leaf_name(sid) == "Fatia premium"
        assert "Fatia premium" in ui.out_tree.value            # árvore HTML
        assert any("Fatia premium" in lbl for lbl, _ in ui.dd_leaf.options)
        ui._on_undo(None)                         # desfazer remove o apelido
        assert ui.seg.leaf_name(sid) is None
        assert ui.tx_leaf_name.value == ""
        ui._on_redo(None)                         # refazer o traz de volta
        assert ui.seg.leaf_name(sid) == "Fatia premium"
        assert ui.tx_leaf_name.value == "Fatia premium"


def test_ui_move_cut_sync_preview_efetiva_undo(task):
    """Mover corte: o campo espelha o corte vigente da folha (extraído das
    conditions lo/hi), o preview mostra os dois lados sem alterar a árvore, o
    'Mover corte' efetiva com checkpoint e o desfazer restaura o corte anterior."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        # raiz selecionada: sem corte móvel → controles desabilitados
        assert ui.btn_move_cut.disabled and ui.tx_move_cut.disabled
        ui.dd_leaf.value = "root"
        ui.dd_feature.value = "score"
        ui.tg_mode.value = "Manual"
        ui.tx_cuts.value = "0.8"
        ui._on_preview(None); ui._on_split(None)
        # folha à esquerda do corte (hi = 0.8): campo habilita e espelha o corte
        sid = next(s for s, v in ui.seg.segments.items()
                   if v["is_leaf"] and v["conditions"]
                   and v["conditions"][-1].get("hi") == 0.8)
        ui.dd_leaf.value = sid
        assert not ui.btn_move_cut.disabled
        assert ui.tx_move_cut.value == pytest.approx(0.8)
        # preview: renderiza os dois lados e NÃO altera a árvore
        ui.tx_move_cut.value = 0.9
        ui._on_move_cut_preview(None)
        assert "folha (esq.)" in ui.out_move_cut.value
        assert sid in ui.seg.segments
        ui._on_move_cut(None)                       # efetiva (com _checkpoint)
    movida = next(v for v in ui.seg.segments.values()
                  if v["is_leaf"] and v["conditions"]
                  and v["conditions"][-1].get("hi") == 0.9)
    assert int(movida["mask"].sum()) == int((ui.df["score"] <= 0.9).sum())
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_undo(None)                           # desfazer restaura o corte
    assert any(v["is_leaf"] and v["conditions"]
               and v["conditions"][-1].get("hi") == 0.8
               for v in ui.seg.segments.values())


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
    assert not ui.btn_img_lock.disabled
    w.selected = "root"                      # raiz: só o chip; nada de agir
    assert ui.dd_leaf.value == outra         # nó interno não muda a folha ativa
    assert ui.btn_img_merge_l.disabled and ui.btn_img_collapse.disabled
    assert ui.btn_img_lock.disabled
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


def test_ui_barra_do_preview_tem_os_clones(task):
    """A barra do preview traz os clones compactos (desfazer/refazer/auto-fit/
    resetar), o zoom e as ações de folha. O painel compacto "Dividir a folha" foi
    removido junto com o botão 'Sugerir quebra', que era seu único gatilho: clicar
    num nó já seleciona a folha, e a divisão acontece no card da própria aba."""
    ui = _build(task)
    assert not hasattr(ui, "tree_img_split")
    assert not hasattr(ui, "btn_img_suggest")
    barra = _widgets_de(ui.tree_img_bar)
    for wdg in (ui.btn_img_lock,
                ui.btn_img_merge_l, ui.btn_img_merge_r, ui.btn_img_merge_na,
                ui.btn_img_collapse, ui.btn_img_undo, ui.btn_img_redo,
                ui.btn_img_autofit, ui.btn_img_reset,
                ui.sl_tree_zoom, ui.btn_tree_zoom_reset):
        assert wdg in barra, f"widget ausente da barra do preview: {wdg!r}"
    # clones compactos (lado a lado): largura auto, sem o width 98% dos cards
    for wdg in (ui.btn_img_undo, ui.btn_img_redo, ui.btn_img_autofit,
                ui.btn_img_reset, ui.btn_tree_zoom_reset):
        assert wdg.layout.width == "auto"
    # habilitação de desfazer/refazer espelhada dos originais (dlink)
    assert ui.btn_img_undo.disabled and ui.btn_img_redo.disabled   # sem histórico
    # respiro entre os grupos fundir-irmãs · fundir-missing · recolher
    assert ui.btn_img_merge_na.layout.margin.endswith("18px")
    assert ui.btn_img_collapse.layout.margin.endswith("18px")


def test_ui_grow_pelo_preview(task):
    """Crescer a árvore com o preview aberto: o fluxo variável→cortes→preview→
    criar segmento roda no card da aba e a imagem se re-renderiza com a árvore
    nova. Funciona com e sem anywidget."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_tree_preview(None)            # abre o preview (interativo ou estático)
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
def test_ui_tree_img_lock(task):
    """Travar/destravar a folha clicada direto da barra do preview."""
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


def test_ui_diag_teste_des_oot(task):
    """A tabela de folhas (Diagnóstico) traz, ao lado de p (irmãs), o teste de
    aderência da estimativa comparando DES × OOT por folha."""
    ui = _build(task, n=8000, seed=3)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui.tabs.selected_index = ui._diag_tab_index   # a tabela renderiza ao abrir a aba
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


def test_ui_sql_fallback_dropdown_e_chip_sem_rota(task):
    """O card de SQL tem o dropdown de fallback p/ não classificados: NULL por
    padrão; 'pior nota' persiste no segmentador (viaja no to_dict) e troca o
    ELSE do SQL. A cobertura (linhas sem rota) é aferida pelo segmentador — a
    barra de status não a exibe mais."""
    ui = _build(task)
    assert ui.dd_fallback.value is None             # padrão: sem fallback
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui._on_sql(None)
    assert "ELSE NULL" in ui.out_sql.value          # padrão preservado
    assert "Sem rota" not in ui.bar.value           # barra enxuta: sem o chip
    assert "Fechadas" in ui.bar.value               # contador de travadas segue na barra
    assert ui.seg.n_orfas() == 0                    # base atual coberta (DES)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_fallback.value = "pior_nota"          # escolhe o fallback
        ui._on_sql(None)
    assert ui.seg.fallback == "pior_nota"           # persistido no segmentador
    assert ui.seg.to_dict()["meta"]["fallback"] == "pior_nota"
    assert "ELSE NULL" not in ui.out_sql.value      # ELSE vira a folha escolhida
    assert "fallback" in ui.out_sql.value           # comentário no cabeçalho


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


def test_ui_diag_placar_quatro_dimensoes(task):
    """O placar traz as 4 dimensões com veredito, sem o parágrafo explicativo da
    calibração (removido: o card já é denso e a explicação vive na aba de
    validação)."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui._on_diag(None)
    html = ui.out_diag.value
    for dim in ("Discriminação", "Estabilidade", "Calibração", "Estrutura"):
        assert dim in html
    assert "O que é calibração" not in html


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


def test_ui_cenarios_salvar_comparar_restaurar(task):
    """Cenários em memória: salvar → mutar → comparar (diff não-vazio) →
    restaurar volta ao estado salvo (e é desfazível via ↶)."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        # árvore com 1 split → salva como cenário 'v1'
        ui.dd_leaf.value = "root"
        ui.dd_feature.value = "score"
        ui.tg_mode.value = "Manual"
        ui.tx_cuts.value = "0.8"
        ui._on_preview(None); ui._on_split(None)
        n_cen = _nleaf(ui)
        ui.tx_scn_name.value = "v1"
        ui._on_scn_save(None)
        assert "v1" in ui._scenarios
        segs_salvos = ui._scenarios["v1"]["data"]["segments"]
        # mini-tabela resume o estado atual + o cenário; lista ganha os botões
        assert "atual" in ui.out_scn_summary.value and "v1" in ui.out_scn_summary.value
        assert len(ui.box_scn_list.children) == 1
        # muta a árvore (recolhe o split de volta à raiz) → diverge do cenário
        ui._on_collapse(None)
        n_mut = _nleaf(ui)
        assert n_mut != n_cen
        # comparar com o atual: diff não-vazio, renderizado pelo helper comum
        ui._on_scn_compare("v1")
        assert "Concord" in ui.out_scn_diff.value and "Erro" not in ui.out_scn_diff.value
        assert "Migração de folhas" in ui.out_scn_diff.value
        assert "cenário" in ui.out_scn_diff.value        # rótulo A = atual · B = cenário
        # restaurar volta EXATAMENTE ao estado salvo…
        ui._on_scn_restore("v1")
        assert _nleaf(ui) == n_cen
        assert ui.seg.to_dict()["segments"] == segs_salvos
        # …e é desfazível: o ↶ devolve a árvore mutada (pós-auto-fit)
        ui._on_undo(None)
        assert _nleaf(ui) == n_mut


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


def test_ui_cap_lift_metricas_por_safra(task):
    """Os três botões novos do card de discriminação (CAP · Lift · métricas por
    safra) renderizam em out_discrim nos DOIS task_type, e a saída é invalidada
    quando a árvore muda de estrutura (mensagem 'Árvore alterada')."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        for btn in (ui.btn_cap, ui.btn_lift, ui.btn_msafra):
            btn.click()
            assert "<img" in ui.out_discrim.value, btn.description
            assert "Erro" not in ui.out_discrim.value, btn.description
        # mudança estrutural → _refresh marca a saída como obsoleta
        ui._on_autofit(None)
    assert "Árvore alterada" in ui.out_discrim.value


def test_ui_metricas_por_safra_sem_coluna_tempo(task):
    """Sem coluna de tempo válida, o botão de métricas por safra mostra a
    orientação amigável (sem stacktrace) em out_discrim."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui.tx_sib_time.value = "nao_existe"
        ui.btn_msafra.click()
    assert "coluna de data" in ui.out_discrim.value
    assert "Erro" not in ui.out_discrim.value


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


def test_ui_busy_reabilita_botoes_apos_excecao(task):
    """_busy desabilita os botões durante a ação e SEMPRE re-habilita ao sair,
    mesmo quando o handler estoura exceção (finally)."""
    ui = _build(task)
    b1, b2 = ui.btn_autofit, ui.btn_boot
    assert not b1.disabled and not b2.disabled
    with pytest.raises(RuntimeError):
        with ui._busy(b1, b2, msg="testando…"):
            assert b1.disabled and b2.disabled     # ocupado: botões travados
            raise RuntimeError("boom")
    assert not b1.disabled and not b2.disabled     # finally re-habilitou


def test_ui_busy_status_mostra_e_limpa(task):
    """Com ``status``, o _busy mostra o aviso ⏳ durante a ação; ao sair limpa o
    aviso se o handler não escreveu resultado próprio, e preserva o resultado
    quando ele escreveu."""
    ui = _build(task)
    with ui._busy(ui.btn_boot, status=ui.out_boot, msg="rodando…"):
        assert "⏳" in ui.out_boot.value
    assert ui.out_boot.value == ""                 # nada escrito -> limpa o aviso
    with ui._busy(ui.btn_boot, status=ui.out_boot, msg="rodando…"):
        ui.out_boot.value = "<b>resultado</b>"
    assert ui.out_boot.value == "<b>resultado</b>"  # resultado do handler preservado


def test_ui_confirm_twice_nao_dispara_no_primeiro_clique(task):
    """Ações destrutivas: o 1º clique só ARMA o botão ('Confirmar?' em vermelho);
    a ação roda apenas no 2º clique dentro da janela, restaurando rótulo/estilo."""
    ui = _build(task)
    btn = ui.btn_reset
    rotulo, estilo = btn.description, btn.button_style
    chamadas = []
    ui._confirm_twice(btn, lambda: chamadas.append(1), timeout=5.0)   # 1º clique
    assert chamadas == []                          # NÃO disparou
    assert btn.description == "Confirmar?" and btn.button_style == "danger"
    ui._confirm_twice(btn, lambda: chamadas.append(1), timeout=5.0)   # 2º clique
    assert chamadas == [1]                         # agora sim
    assert btn.description == rotulo and btn.button_style == estilo


def test_ui_confirm_twice_desarma_apos_timeout(task):
    """Sem o 2º clique dentro da janela, o clique seguinte apenas REARMA (não
    executa) — o timeout expirado não conta como confirmação."""
    import time
    ui = _build(task)
    btn = ui.btn_prune
    chamadas = []
    ui._confirm_twice(btn, lambda: chamadas.append(1), timeout=0.05)  # arma
    time.sleep(0.3)                                # janela expira (e o Timer desarma)
    ui._confirm_twice(btn, lambda: chamadas.append(1), timeout=0.05)  # rearma, não executa
    assert chamadas == []
    assert btn.description == "Confirmar?"


def test_ui_log_mantem_historico_e_apara_em_40(task):
    """O console usa buffer: mensagens de ações anteriores permanecem (nada de
    clear_output destrutivo) e só as últimas 40 linhas são mantidas; o botão
    'limpar' zera o histórico."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(45):
            ui._log(f"linha {i}")
    assert len(ui._log_lines) == 40
    assert ui._log_lines[0] == "linha 5" and ui._log_lines[-1] == "linha 44"
    ui._on_clear_log(None)
    assert ui._log_lines == []


def test_ui_log_delta_apos_acao_estrutural(task):
    """Cada ação estrutural imprime no console a linha compacta de Δ vs o estado
    anterior: nº de folhas, métrica principal na amostra de comparação (KS na
    classificação · R² na regressão) e PSI máximo; desfazer loga o Δ inverso."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_leaf.value = "root"
        ui.dd_feature.value = "score"
        ui.tg_mode.value = "Manual"
        ui.tx_cuts.value = "0.8"
        ui._on_preview(None); ui._on_split(None)
    linha = next(l for l in reversed(ui._log_lines) if l.startswith("dividir:"))
    assert "folhas 1→2" in linha
    metr = "KS OOT" if task == "classification" else "R² OOT"
    assert metr in linha and "→" in linha and "(" in linha    # a→b (±d)
    assert "PSI máx" in linha
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_undo(None)                       # o desfazer também loga o Δ
    linha = next(l for l in reversed(ui._log_lines) if "desfeito" in l)
    assert "folhas 2→1" in linha


def test_ui_refresh_invalida_diag_sql_validacao(task):
    """Regressão (b): após uma mudança estrutural (ex.: merge), o placar de
    saúde, o SQL gerado e as análises de validação NÃO mantêm o conteúdo antigo
    — são substituídos pela tarja/mensagem de desatualizado."""
    ui = _build(task, n=6000, seed=8)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui._on_diag(None)
        ui._on_sql(None)
    assert "Erro" not in ui.out_diag.value and "CASE" in ui.out_sql.value
    diag0 = ui.out_diag.value
    ui.out_validate.value = "<b>análises antigas</b>"   # simula validação renderizada
    n0 = _nleaf(ui)
    with contextlib.redirect_stdout(io.StringIO()):     # funde o 1º par de folhas-irmãs
        for sid in [s for s, v in ui.seg.segments.items() if v["is_leaf"]]:
            ui.dd_leaf.value = sid
            ui._on_merge("right")
            if _nleaf(ui) < n0:
                break
            ui._on_merge("left")
            if _nleaf(ui) < n0:
                break
    assert _nleaf(ui) < n0                              # o merge de fato aconteceu
    assert "desatualizado" in ui.out_diag.value and ui.out_diag.value != diag0
    assert "var(--warn-bg)" in ui.out_diag.value        # tarja âmbar (token semântico)
    assert "desatualizado" in ui.out_validate.value
    assert ui.out_sql.value.startswith("--") and "desatualizado" in ui.out_sql.value
    assert "THEN" not in ui.out_sql.value               # o corpo do SQL antigo sumiu
    # saídas que nasceram vazias/ocultas seguem vazias (sem tarja espúria)
    ui2 = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui2._on_autofit(None)
    assert ui2.out_diag.value == "" and ui2.out_validate.value == ""
    assert ui2.out_sql.value == ""


def test_ui_seletor_variavel_e_dropdown_com_lista_completa(task):
    """Os seletores de variável são Dropdown — o MESMO widget da folha, que abre
    a lista inteira num clique. Um Combobox já esteve aqui e o <datalist> do
    navegador filtrava as opções pelo texto do campo: com uma variável escolhida,
    só ela aparecia. O Dropdown também torna impossível selecionar valor fora da
    lista (o widget recusa), então não há entrada inválida a validar."""
    import ipywidgets as W
    import traitlets
    ui = _build(task)
    assert isinstance(ui.dd_feature, W.Dropdown) and isinstance(ui.dd_var, W.Dropdown)
    assert isinstance(ui.dd_leaf, W.Dropdown)      # mesma classe do seletor de folha
    # a lista traz TODAS as candidatas, nas duas abas
    assert len(ui.dd_feature.options) == len(ui.features)
    assert len(ui.dd_var.options) == len(ui.features)
    assert ui._sel_feature() == "score"            # sem feature_labels: rótulo = coluna
    with pytest.raises(traitlets.TraitError):      # o widget barra o inválido
        ui.dd_feature.value = "variavel_que_nao_existe"
    assert ui._sel_feature() == "score"            # seleção intacta após a recusa


def test_ui_seletor_ordena_por_iv(task):
    """O toggle 'ordenar por IV' reordena as opções por IV desc e anexa o IV ao
    rótulo; a seleção sobrevive à reordenação e segue resolvendo p/ a coluna."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.tg_feat_iv.value = True
    opts = list(ui.dd_feature.options)
    assert any("(IV " in o for o in opts)
    ivm = ui._iv_map(ui.dd_leaf.value)
    vals = [ivm[ui._feat_by_label[o]] for o in opts
            if not pd.isna(ivm.get(ui._feat_by_label[o], float("nan")))]
    assert vals == sorted(vals, reverse=True)      # ordem decrescente de IV
    assert ui._sel_feature() == "score"            # seleção preservada
    assert "(IV " in ui.dd_feature.value           # rótulo atual ganhou o IV
    with contextlib.redirect_stdout(io.StringIO()):
        ui.tg_feat_iv.value = False                # desligar volta ao rótulo simples
    assert all("(IV " not in o for o in ui.dd_feature.options)
    assert ui._sel_feature() == "score"


def test_ui_reordenar_por_iv_nao_invalida_preview(task):
    """Reordenar por IV só re-rotula a MESMA coluna: o preview pendente não é
    invalidado (contexto variável+folha inalterado)."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_leaf.value = "root"
        ui.dd_feature.value = "score"
        ui.tg_mode.value = "Manual"
        ui.tx_cuts.value = "0.8"
        ui._on_preview(None)
        assert ui._pending is not None
        ui.tg_feat_iv.value = True                 # rótulo vira "score (IV …)"
    assert ui._sel_feature() == "score"
    assert ui._pending is not None                 # preview segue válido
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_split(None)
    assert _nleaf(ui) >= 2                         # split funciona com o rótulo de IV


def test_ui_var_analise_ordena_por_iv_e_mantem_lista(task):
    """Aba Análise: o toggle de IV reordena e re-rotula o seletor sem perder
    nenhuma candidata, e a coluna selecionada sobrevive à reordenação."""
    import traitlets
    ui = _build(task)
    antes = ui._sel_var()
    with contextlib.redirect_stdout(io.StringIO()):
        ui.tg_var_iv.value = True
    assert any("(IV " in o for o in ui.dd_var.options)
    assert len(ui.dd_var.options) == len(ui.features)   # nenhuma sumiu
    assert ui._sel_var() == antes                       # mesma coluna, outro rótulo
    with pytest.raises(traitlets.TraitError):
        ui.dd_var.value = "typo_qualquer"


def test_ui_exportar_excel(task, tmp_path):
    pytest.importorskip("openpyxl")
    import os
    ui = _build(task)
    p = str(tmp_path / "arvore.xlsx")
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui.tx_xlsx_path.value = p
        ui._on_xlsx(None)                          # não existe → salva direto
    assert os.path.exists(p)
    assert any("Excel salvo" in l for l in ui._log_lines)


def test_ui_exportar_excel_sem_openpyxl_avisa(task, tmp_path, monkeypatch):
    """Sem openpyxl, o handler mostra a instrução amigável no console (não
    estoura exceção na UI)."""
    from yggdrasil.credit_risk.tree import TreeSegmenter
    ui = _build(task)

    def boom(self, path, table="minha_tabela"):
        raise ImportError("A exportação para Excel requer o pacote opcional "
                          "'openpyxl' (instale com: pip install openpyxl).")

    monkeypatch.setattr(TreeSegmenter, "to_excel", boom)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.tx_xlsx_path.value = str(tmp_path / "x.xlsx")
        ui._on_xlsx(None)
    assert any("openpyxl" in l for l in ui._log_lines)


def test_ui_reset_e_prune_pedem_confirmacao_no_botao(task):
    """Clicar 1× em Resetar/Podar (inclusive o clone do preview) NÃO muta a
    árvore: o botão só arma o 'Confirmar?'; o 2º clique executa de fato."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
    n0 = _nleaf(ui)
    assert n0 >= 2
    with contextlib.redirect_stdout(io.StringIO()):
        ui.btn_reset.click()                       # 1º clique: só arma
        assert _nleaf(ui) == n0
        assert ui.btn_reset.description == "Confirmar?"
        ui.btn_reset.click()                       # 2º clique: reseta
    assert _nleaf(ui) == 1
    for btn in (ui.btn_prune, ui.btn_img_reset):   # demais destrutivos também armam
        with contextlib.redirect_stdout(io.StringIO()):
            btn.click()
        assert btn.description == "Confirmar?"


def test_ui_spark_card_pandas_e_progresso(task):
    """Card 'Reconstruir folhas': sem o nome da tabela, aplica a régua em
    memória (pandas puro, sem Spark) usando ``ui.score_df`` quando definido —
    com a tabela de progresso ✅ por etapa, a distribuição por folha no card e
    o resultado em ``ui.result``; o _busy re-habilita o botão ao final."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui.score_df = make_df(task, n=300, seed=15)
        ui.tx_spark_in.value = ""                  # sem tabela → caminho pandas
        ui._on_spark_apply(None)
    assert isinstance(ui.result, pd.DataFrame) and len(ui.result) == 300
    assert {"segmento", "folha", "valor_regua"}.issubset(ui.result.columns)
    assert "ui.score_df" in ui.out_spark.value     # origem citada no card
    assert "Distribuição por folha" in ui.out_spark.value
    assert "✅" in ui.out_spark_progress.value     # etapas concluídas
    assert "❌" not in ui.out_spark_progress.value
    assert not ui.btn_spark_apply.disabled         # _busy re-habilitou o botão


def test_ui_spark_card_erro_resumido_no_card(task):
    """Erro na aplicação (base em memória sem as colunas da régua): o card mostra
    só o aviso curto apontando o Console, a etapa que falhou é marcada com ❌ na
    tabela de progresso e o detalhe completo vai para o Console."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui.score_df = pd.DataFrame({"sem_as_colunas": [1.0, 2.0]})
        ui.tx_spark_in.value = ""
        ui._on_spark_apply(None)
    assert "Console" in ui.out_spark.value          # erro resumido no card
    assert "❌" in ui.out_spark_progress.value      # etapa marcada com erro
    assert any("ERRO" in l for l in ui._log_lines)  # detalhe no console
    assert not ui.btn_spark_apply.disabled          # botão re-habilitado


def test_ui_weight_col_visao_dupla_contratos_saldo(task):
    """Com `weight_col`, o cartão da folha ganha o % do SALDO e o alvo ponderado,
    e as colunas novas aparecem na tabela de folhas e no TSV copiável."""
    df = make_df(task, n=4000, seed=5)
    df["saldo"] = np.where(df["score"] > 0.9, 900.0, 100.0)
    ui = _build(task, df=df, weight_col="saldo")
    assert ui.seg.weight_col == "saldo"
    assert "saldo" not in ui.features            # peso não é variável candidata
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
    _lv, cols, headers = ui._leaf_table_spec()
    assert headers.get("saldo_%") == "% saldo"
    assert any(c.startswith("valor_pond_") for c in cols)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.tabs.selected_index = ui._diag_tab_index   # a tabela renderiza ao abrir a aba
    assert "% saldo" in ui.out_table.value        # tabela renderizada
    assert "% saldo" in ui._leaves_tsv().splitlines()[0]
    card = ui._leaf_header_html()
    assert "Repr. saldo" in card and "pond." in card


def test_ui_sem_weight_col_tela_intocada(task):
    """Sem `weight_col` (padrão), nada da visão dupla aparece na tela."""
    ui = _build(task)
    assert ui.seg.weight_col is None
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
    _lv, cols, _headers = ui._leaf_table_spec()
    assert not [c for c in cols if c == "saldo_%" or "pond" in str(c)]
    assert "% saldo" not in ui._leaves_tsv() and "% saldo" not in ui.out_table.value
    assert "Repr. saldo" not in ui._leaf_header_html()


# ======================================================================
# Aba "Árvore interativa" — canvas navegável + painel de criação
# ======================================================================
def _abre_canvas(ui):
    """Abre a aba do canvas como o clique do usuário faria (render preguiçoso)."""
    with contextlib.redirect_stdout(io.StringIO()):
        ui.tabs.selected_index = ui._canvas_tab_index
    return ui._cv_widget


def _foca(ui, w, sid):
    """Põe o nó em foco pelo caminho de verdade: clique no canvas quando ele
    existe, dropdown de folha quando não (modo offline)."""
    with contextlib.redirect_stdout(io.StringIO()):
        if w is not None:
            w.selected = sid
        elif sid in [s for _, s in ui.dd_leaf.options]:
            ui.dd_leaf.value = sid
        else:
            ui._cv_sel = sid
            ui._refresh_cv_panel()


def test_ui_aba_canvas_abre_o_workbench(task):
    """A Árvore interativa é a PRIMEIRA aba — a 'Construir' está OCULTA
    (candidata a exclusão): os widgets dela seguem vivos porque o painel e as
    janelinhas os compartilham, mas ela não aparece na barra de abas. Os
    índices dos renders preguiçosos acompanham."""
    ui = _build(task)
    titulos = [ui.tabs.get_title(i) for i in range(len(ui.tabs.children))]
    assert "Construir" not in titulos
    assert titulos[0] == "Árvore interativa"
    assert ui._canvas_tab_index == 0 and ui._build_tab_index is None
    assert ui._iv_tab_index == titulos.index("Análise de variáveis")
    assert ui._diag_tab_index == titulos.index("Diagnóstico")
    # os widgets da Construir continuam vivos (as janelinhas dependem deles)
    assert ui.sl_depth is not None and ui.dd_feature is not None


def test_ui_canvas_desenha_um_cartao_por_no(task):
    """Cada nó vira um cartão posicionado no plano e cada aresta liga um pai a um
    filho — o layout inteiro é calculado no Python."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
    w = _abre_canvas(ui)
    if w is None:
        pytest.skip("anywidget não instalado")
    assert len(w.nodes) == len(ui.seg.segments)
    assert len(w.edges) == len(ui.seg.segments) - 1        # todo nó menos a raiz
    assert w.content_w > 0 and w.content_h > 0
    folhas = {n["sid"] for n in w.nodes if n["leaf"]}
    assert folhas == {s for s, v in ui.seg.segments.items() if v["is_leaf"]}
    raiz = next(n for n in w.nodes if n["sid"] == "root")
    assert "TODA A CARTEIRA" in raiz["html"] and "class='bar'" in raiz["html"]


def test_ui_canvas_clique_abre_painel_da_folha(task):
    """Clicar num nó abre o painel no contexto dele: cabeçalho, métricas,
    sugestões por IV e os controles de corte."""
    ui = _build(task)
    w = _abre_canvas(ui)
    if w is None:
        pytest.skip("anywidget não instalado")
    _foca(ui, w, "root")
    assert ui._cv_sel == "root"
    assert "Raiz da árvore" in ui.out_cv_head.value
    assert "população" in ui.out_cv_stats.value
    assert ui.box_cv_split.layout.display == ""            # raiz é folha: dá p/ dividir
    assert [b for b in ui.btns_cv_sug if b.layout.display != "none"]


def test_ui_canvas_preview_e_corte(task):
    """Prever mostra as faixas propostas com o selo de monotonicidade; criar
    segmentos divide a folha — desfazível, como na aba Construir."""
    ui = _build(task)
    w = _abre_canvas(ui)
    antes = _nleaf(ui)
    _foca(ui, w, "root")
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_preview(None)
    assert "Preview ·" in ui.out_cv_preview.value
    assert "monotônica" in ui.out_cv_preview.value
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_apply(None)
    assert _nleaf(ui) > antes
    assert ui._undo                                        # desfazível
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_undo(None)
    assert _nleaf(ui) == antes


def _sel_var_cv(ui, col):
    alvo = next(l for l, f in ui._cv_feat_by_label.items() if f == col)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_cv_feature.value = alvo


def test_ui_canvas_corte_manual_numerico(task):
    """Variável numérica no modo Manual: caixa de texto com um corte por vírgula."""
    ui = _build(task)
    _abre_canvas(ui)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.tg_cv_mode.value = "Manual"
    _sel_var_cv(ui, "score")
    assert ui.box_cv_cuts.layout.display == ""
    assert ui.cv_cat_box.layout.display == "none"
    ui.tx_cv_cuts.value = "0.8, 1.1"
    assert ui._cv_parse_cuts("score", ui._cv_node()) == [0.8, 1.1]
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_preview(None)
        ui._on_cv_apply(None)
    assert _nleaf(ui) == 3                                 # 2 cortes → 3 faixas


def test_ui_canvas_corte_manual_categorico_usa_agrupador(task):
    """Variável categórica no modo Manual: o MESMO agrupador da aba Construir —
    um seletor de grupo por categoria, e categorias do mesmo grupo viram um nó."""
    ui = _build(task)
    _abre_canvas(ui)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.tg_cv_mode.value = "Manual"
    _sel_var_cv(ui, "garantia")
    assert ui.cv_cat_box.layout.display == ""              # agrupador no lugar do texto
    assert ui.box_cv_cuts.layout.display == "none"
    assert set(ui._cv_cat_widgets) == set("ABCD")
    assert ui._cv_parse_cuts("garantia", ui._cv_node()) == [["A"], ["B"], ["C"], ["D"]]
    cats = list(ui._cv_cat_widgets)                        # junta as 2 primeiras num grupo
    ui._cv_cat_widgets[cats[1]].value = ui._cv_cat_widgets[cats[0]].value
    grupos = ui._cv_parse_cuts("garantia", ui._cv_node())
    assert sorted(len(g) for g in grupos) == [1, 1, 2]
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_preview(None)
        ui._on_cv_apply(None)
    assert _nleaf(ui) == 3                                 # os 3 grupos montados


def test_ui_canvas_regras_de_negocio_da_folha(task):
    """O painel edita as regras de negócio da folha: apelido, fechar/reabrir e as
    ações de estrutura, cada uma habilitada só onde faz sentido."""
    ui = _build(task)
    w = _abre_canvas(ui)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_apply(None)                              # divide a raiz
    folhas = [s for s, v in ui.seg.segments.items() if v["is_leaf"]]
    _foca(ui, w, folhas[0])
    with contextlib.redirect_stdout(io.StringIO()):
        ui.tx_cv_name.value = "clientes bons"
    assert ui.seg.leaf_name(folhas[0]) == "clientes bons"
    assert ui.btn_cv_merge_l.disabled                      # 1ª folha não tem vizinha à esq.
    assert not ui.btn_cv_merge_r.disabled
    assert ui.btn_cv_collapse.disabled                     # folha não recolhe
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_lock(None)
    assert folhas[0] in ui.locked
    assert "Reabrir" in ui.btn_cv_lock.description
    assert ui.box_cv_split.layout.display == "none"        # fechada não divide
    assert "🔒" in ui.out_cv_note.value


def test_ui_canvas_recolher_e_fundir(task):
    """Recolher devolve um ramo à condição de folha; fundir junta duas vizinhas.
    As duas passam pelo mesmo checkpoint de desfazer das outras abas."""
    ui = _build(task)
    w = _abre_canvas(ui)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_apply(None)
    n_apos_corte = _nleaf(ui)
    folhas = [s for s, v in ui.seg.segments.items() if v["is_leaf"]]
    _foca(ui, w, folhas[0])
    with contextlib.redirect_stdout(io.StringIO()):         # funde as 2 primeiras
        ui._on_cv_merge("right")(None)
    assert _nleaf(ui) == n_apos_corte - 1
    _foca(ui, w, "root")                                    # recolhe a raiz de volta
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_collapse(None)
    assert _nleaf(ui) == 1 and ui.seg.segments["root"]["is_leaf"]


def test_ui_canvas_selecao_espelhada_entre_abas(task):
    """A folha em foco é uma só: escolhida no canvas, ela vira a folha ativa das
    outras abas — e vice-versa."""
    ui = _build(task)
    w = _abre_canvas(ui)
    if w is None:
        pytest.skip("anywidget não instalado")
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_apply(None)
    folhas = [s for s, v in ui.seg.segments.items() if v["is_leaf"]]
    with contextlib.redirect_stdout(io.StringIO()):
        w.selected = folhas[-1]                    # canvas → resto da UI
    assert ui.dd_leaf.value == folhas[-1]
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_leaf.value = folhas[0]               # resto da UI → canvas
    assert w.selected == folhas[0] and ui._cv_node() == folhas[0]


def test_ui_canvas_offline_cai_no_painel_sem_mapa(task):
    """Com `allow_interactive_tree=False` (padrão no Databricks) o mapa some, mas
    o painel continua: ele é ipywidgets puro, sem nenhuma rede."""
    ui = _build(task, allow_interactive_tree=False)
    assert _abre_canvas(ui) is None
    assert ui.box_cv_canvas.layout.display == "none"
    assert "allow_interactive_tree=True" in ui.out_cv_msg.value
    assert ui.box_cv_panel.layout.display == ""
    assert "Raiz da árvore" in ui.out_cv_head.value
    antes = _nleaf(ui)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_preview(None)
        ui._on_cv_apply(None)
    assert _nleaf(ui) > antes                      # o corte funciona sem o mapa


def test_ui_canvas_mover_corte_da_divisa(task):
    """A divisa entre duas folhas vizinhas é editável pelo painel: o bloco só
    aparece quando há corte móvel, e mover aplica exatamente o valor pedido."""
    ui = _build(task)
    w = _abre_canvas(ui)
    _sel_var_cv(ui, "score")                               # corte móvel só existe em num
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_apply(None)
    folhas = [s for s, v in ui.seg.segments.items() if v["is_leaf"]]
    _foca(ui, w, folhas[0])
    assert ui.box_cv_move.layout.display == ""
    vigente = float(ui.tx_cv_move.value)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_move_preview(None)
    assert ui.out_cv_move.value                            # preview dos dois lados
    pedido = round(vigente * 0.97, 6)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.tx_cv_move.value = pedido
        ui._on_cv_move(None)
    assert ui.seg.movable_cut(ui._cv_sel)["cut"] == pytest.approx(pedido)
    assert ui._undo


def test_ui_canvas_aloca_faltantes_na_folha(task):
    """O nó de faltantes (NaN) do split pode ser absorvido por uma folha
    populada — o botão só habilita onde esse nó existe."""
    ui = _build(task, com_na=True)
    w = _abre_canvas(ui)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_apply(None)
    folhas = [s for s, v in ui.seg.segments.items() if v["is_leaf"]]
    na = [s for s in folhas
          if ui.seg.segments[s]["conditions"]
          and ui.seg.segments[s]["conditions"][-1]["kind"] == "na"]
    if not na:
        pytest.skip("o split desta base não gerou nó de faltantes")
    alvo = next(s for s in folhas if ui._cv_missing_sibling(s))
    _foca(ui, w, alvo)
    assert not ui.btn_cv_missing.disabled
    antes = _nleaf(ui)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_missing(None)
    assert _nleaf(ui) == antes - 1                         # o nó de NaN foi absorvido
    _foca(ui, w, ui._cv_sel)
    assert ui.btn_cv_missing.disabled                      # não há mais o que alocar


def test_ui_canvas_reset_pede_confirmacao(task):
    """Resetar joga fora a árvore toda, então pede dois cliques — e continua
    desfazível."""
    ui = _build(task)
    _abre_canvas(ui)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
    antes = _nleaf(ui)
    assert antes > 1
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_reset(ui.btn_cv_reset)                   # 1º clique: só confirma
    assert _nleaf(ui) == antes
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_reset(ui.btn_cv_reset)                   # 2º clique: reseta
    assert _nleaf(ui) == 1
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_undo(None)
    assert _nleaf(ui) == antes


def test_ui_canvas_barra_de_acoes_da_arvore(task):
    """A barra traz desfazer/refazer e as três ações automáticas — que abrem a
    janelinha de confirmação em vez de executar direto (ver teste do modal)."""
    ui = _build(task)
    _abre_canvas(ui)
    assert ui.btn_cv_undo.disabled and ui.btn_cv_redo.disabled
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
    assert _nleaf(ui) > 1 and not ui.btn_cv_undo.disabled
    apos_fit = _nleaf(ui)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_automerge(None)
    assert _nleaf(ui) <= apos_fit
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_undo(None)
    assert _nleaf(ui) == apos_fit and not ui.btn_cv_redo.disabled


def test_ui_canvas_modal_confirma_acoes_automaticas(task):
    """Auto-fit/auto-fundir/podar abrem a janelinha no meio do canvas com os
    controles DA ABA CONSTRUIR (mesmas instâncias — nada para divergir); nada
    roda até Aplicar, e Cancelar fecha sem tocar na árvore."""
    ui = _build(task)
    _abre_canvas(ui)
    assert ui.box_cv_modal.layout.display == "none"        # fechada por padrão
    with contextlib.redirect_stdout(io.StringIO()):
        ui.btn_cv_autofit.click()
    assert ui.box_cv_modal.layout.display == ""
    assert ui._cv_modal_kind == "fit"
    assert ui.sl_depth in ui.box_cv_modal_body.children    # instância da Construir
    with contextlib.redirect_stdout(io.StringIO()):        # cancelar: nada muda
        ui.btn_cv_modal_cancel.click()
    assert ui.box_cv_modal.layout.display == "none" and _nleaf(ui) == 1
    with contextlib.redirect_stdout(io.StringIO()):        # aplicar: roda de verdade
        ui.btn_cv_autofit.click()
        ui.sl_depth.value = 2
        ui.btn_cv_modal_ok.click()
    assert ui.box_cv_modal.layout.display == "none" and _nleaf(ui) > 1
    assert ui._undo                                        # desfazível como sempre
    with contextlib.redirect_stdout(io.StringIO()):        # cada ação leva seu corpo
        ui.btn_cv_automerge.click()
    assert ui.sl_alpha in ui.box_cv_modal_body.children
    assert ui.sl_gap in ui.box_cv_modal_body.children
    with contextlib.redirect_stdout(io.StringIO()):
        ui.btn_cv_modal_cancel.click()
        ui.btn_cv_prune.click()
    assert ui.sl_repr in ui.box_cv_modal_body.children


def test_ui_abre_na_arvore_interativa_com_iv_pronto(task):
    """A UI abre direto na aba do mapa, com o painel da raiz preenchido e a
    importância (IV) de cada variável já calculada — sugestões e seletor
    ordenado sem nenhum clique."""
    ui = _build(task)
    assert ui.tabs.selected_index == ui._canvas_tab_index
    assert "Raiz da árvore" in ui.out_cv_head.value
    sugestoes = [b for b in ui.btns_cv_sug if b.layout.display != "none"]
    assert sugestoes and all("IV" in b.description for b in sugestoes)
    assert ui.dd_cv_feature.options and "(IV " in ui.dd_cv_feature.options[0]


def test_ui_canvas_limites_de_bin_ligados_a_construir(task):
    """Os limites de tamanho de bin do painel são widgets próprios (visibilidade
    segue o modo DO PAINEL) com os VALORES ligados aos da Construir por W.link —
    mexer em qualquer lado atualiza o outro, e o preview honra o limite."""
    ui = _build(task)
    _abre_canvas(ui)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_preview(None)
    livre = ui._cv_prev_tbl.shape[0]
    # bloco visível no modo Ótimo; slider só aparece com o checkbox marcado
    assert ui.box_cv_optbin.layout.display == ""
    assert ui.sl_cv_minbin.layout.display == "none"
    with contextlib.redirect_stdout(io.StringIO()):
        ui.cb_cv_minbin.value = True                    # marca NO PAINEL…
        ui.sl_cv_minbin.value = 0.20
    assert ui.cb_minbin.value and ui.sl_minbin.value == pytest.approx(0.20)  # …reflete lá
    assert ui.sl_cv_minbin.layout.display == ""
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_preview(None)
    assert ui._pending["min_bin_size"] == pytest.approx(0.20)
    assert ui._cv_prev_tbl.shape[0] < livre             # o limite mordeu de fato
    with contextlib.redirect_stdout(io.StringIO()):
        ui.sl_maxbin.value = 0.40                       # e o caminho inverso…
        ui.cb_maxbin.value = True
    assert ui.cb_cv_maxbin.value and ui.sl_cv_maxbin.value == pytest.approx(0.40)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.tg_cv_mode.value = "Manual"                  # Manual esconde o bloco
    assert ui.box_cv_optbin.layout.display == "none"


def test_ui_canvas_recebe_o_teste_entre_folhas_comparaveis(task):
    """O card de folhas-irmãs saiu de Diagnóstico e vive na aba do mapa: ele
    compara a folha com as IRMÃS ADJACENTES, que é o que o mapa mostra."""
    ui = _build(task)
    canvas_tab = ui.tabs.children[ui._canvas_tab_index]
    diag_tab = ui.tabs.children[[ui.tabs.get_title(i)
                                 for i in range(len(ui.tabs.children))].index("Diagnóstico")]
    assert ui._card_sib in canvas_tab.children
    assert ui._card_sib not in diag_tab.children
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        _abre_canvas(ui)
        ui._on_sib_analyze(None)
    assert ui.out_sib.value                                 # roda de dentro da aba nova


def test_ui_canvas_semaforo_de_psi_nos_cartoes(task):
    """Cada FOLHA do mapa ganha um ponto colorido com o pior PSI entre as
    amostras não-referência (detalhe por amostra no hover) — o mapa dobra como
    heatmap de estabilidade. Nós internos não têm PSI de folha."""
    ui = _build(task)
    w = _abre_canvas(ui)
    if w is None:
        pytest.skip("anywidget não instalado")
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_apply(None)
    cards = {n["sid"]: n["html"] for n in w.nodes}
    folhas = [s for s, v in ui.seg.segments.items() if v["is_leaf"]]
    assert all("pdot" in cards[s] for s in folhas)
    assert any("PSI OOT" in cards[s] for s in folhas)      # hover com o detalhe
    assert "pdot" not in cards["root"]                     # raiz virou nó interno


def test_ui_canvas_sem_amostras_nao_ha_semaforo(task):
    """Sem `sample_col` não existe PSI por folha — os cartões saem sem o ponto,
    em vez de um ponto sem significado."""
    pytest.importorskip("ipywidgets")
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.tree import TreeSegmenterUI
    df = make_df(task).drop(columns=["amostra"])
    with contextlib.redirect_stdout(io.StringIO()):
        ui = TreeSegmenterUI(df, target="target", task_type=task, date_col="dt_ref")
    w = _abre_canvas(ui)
    if w is None:
        pytest.skip("anywidget não instalado")
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_apply(None)
    assert all("pdot" not in n["html"] for n in w.nodes)


def test_ui_canvas_p_valor_junto_da_decisao_de_fundir(task):
    """A folha em foco mostra o teste contra cada vizinha adjacente, com o
    veredito na régua do α do auto-fundir: p > α = candidata a fusão."""
    ui = _build(task)
    w = _abre_canvas(ui)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_apply(None)
    folhas = [s for s, v in ui.seg.segments.items() if v["is_leaf"]]
    _foca(ui, w, folhas[1])                                # folha do meio: 2 vizinhas
    txt = ui.out_cv_merge_p.value
    assert txt.count("p=") + txt.count("p<") == 2          # um teste por vizinha
    assert "α=0.05" in txt
    assert "candidata a fusão" in txt or "distinta" in txt
    _foca(ui, w, "root")                                   # nó interno: sem vizinhas
    assert ui.out_cv_merge_p.value == ""


def test_ui_canvas_ir_para_folha(task):
    """O dropdown da barra é uma AÇÃO: escolhe a folha, o canvas voa até ela, a
    seleção sincroniza com as outras abas e ele volta ao rótulo."""
    ui = _build(task)
    w = _abre_canvas(ui)
    if w is None:
        pytest.skip("anywidget não instalado")
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_apply(None)
    folhas = [s for s, v in ui.seg.segments.items() if v["is_leaf"]]
    assert len(ui.dd_cv_goto.options) == len(folhas) + 1   # rótulo + folhas atuais
    centro_antes = w.center_token
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_cv_goto.value = folhas[-1]
    assert ui._cv_sel == folhas[-1]
    assert ui.dd_leaf.value == folhas[-1]
    assert w.center_token > centro_antes                   # o canvas voou até lá
    # voltou ao rótulo de ação — que usa "" como valor, nunca None: p/ o
    # Selection do ipywidgets value=None é "sem seleção" e o dropdown fica branco
    assert ui.dd_cv_goto.index == 0 and ui.dd_cv_goto.value == ""


def test_ui_canvas_salvar_cenario_da_barra(task):
    """O botão da barra fotografa a árvore na MESMA lista de cenários de
    Avançado — um só formato de foto, restaurar/comparar continuam lá."""
    ui = _build(task)
    _abre_canvas(ui)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_apply(None)
        ui.tx_cv_scn.value = "antes do experimento"
        ui._on_cv_scn_save(None)
    assert "antes do experimento" in ui._scenarios
    assert ui.tx_cv_scn.value == ""                        # campo pronto p/ a próxima
    with contextlib.redirect_stdout(io.StringIO()):        # sem nome → nome sequencial
        ui._on_cv_scn_save(None)
    assert len(ui._scenarios) == 2


def test_ui_canvas_criterios_seguem_o_task_type(task):
    """O seletor de critério do painel usa as MESMAS opções (e o mesmo gate por
    task_type) da aba Construir, e TODAS elas preparam um split válido.
    Regressão: a 1ª versão hardcodava a lista de classificação e a regressão
    (LGD) quebrava com "Critério de split desconhecido: 'entropy'"."""
    ui = _build(task)
    _abre_canvas(ui)
    assert list(ui.dd_cv_crit.options) == list(ui.dd_split_criterion.options)
    _sel_var_cv(ui, "score")
    for rotulo, crit in ui.dd_cv_crit.options:
        ui.dd_cv_crit.value = crit
        with contextlib.redirect_stdout(io.StringIO()):
            ok, msg = ui._cv_prepare()
        assert ok, f"critério {crit!r} ({rotulo}) falhou: {msg}"


def test_ui_canvas_toggle_psi_troca_a_linha_dos_cartoes(task):
    """Ligado, o toggle "PSI por folha" troca a volumetria dos cartões pelos
    NÚMEROS de PSI por amostra, coloridos pelo semáforo; desligado, volta."""
    ui = _build(task)
    w = _abre_canvas(ui)
    if w is None:
        pytest.skip("anywidget não instalado")
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_apply(None)
    folhas = [s for s, v in ui.seg.segments.items() if v["is_leaf"]]
    html = {n["sid"]: n["html"] for n in w.nodes}
    assert all("obs ·" in html[s] for s in folhas)          # volumetria por padrão
    with contextlib.redirect_stdout(io.StringIO()):
        ui.tg_cv_psi.value = True
    html = {n["sid"]: n["html"] for n in w.nodes}
    assert all("PSI " in html[s] and "OOT" in html[s] for s in folhas)
    assert all("obs ·" not in html[s] for s in folhas)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.tg_cv_psi.value = False
    html = {n["sid"]: n["html"] for n in w.nodes}
    assert all("obs ·" in html[s] for s in folhas)


def test_ui_canvas_toggle_psi_some_sem_amostras(task):
    """Sem sample_col não há PSI — o toggle nem aparece na barra."""
    pytest.importorskip("ipywidgets")
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.tree import TreeSegmenterUI
    df = make_df(task).drop(columns=["amostra"])
    with contextlib.redirect_stdout(io.StringIO()):
        ui = TreeSegmenterUI(df, target="target", task_type=task, date_col="dt_ref")
    assert ui.tg_cv_psi.layout.display == "none"


def test_ui_canvas_tiles_por_amostra(task):
    """Os tiles do painel trazem o alvo POR AMOSTRA com alvo (DES, OOT) e a
    representatividade DENTRO de cada amostra (DES/OOT/ESTAB) — a inspeção de
    'o alvo segura fora do DES?' e 'a folha mantém o peso?'."""
    df = make_df(task, n=6000, seed=4)
    idx = df.sample(frac=0.15, random_state=1).index
    df.loc[idx, "amostra"] = "ESTABILIDADE"          # público recente…
    df.loc[idx, "target"] = np.nan                   # …ainda SEM alvo (psi-only)
    ui = _build(task, df=df)
    w = _abre_canvas(ui)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_apply(None)
    folhas = [s for s, v in ui.seg.segments.items() if v["is_leaf"]]
    _foca(ui, w, folhas[0])
    html = ui.out_cv_stats.value
    rl = ui._risk_label
    assert f"{rl} DES" in html and f"{rl} OOT" in html
    assert f"{rl} ESTAB" not in html                 # ESTAB não tem alvo → sem tile de alvo
    for r in ("repr. DES", "repr. OOT", "repr. ESTAB"):
        assert r in html


def test_ui_canvas_botao_distribuicoes(task):
    """O botão Distribuições abre a JANELINHA sobre o canvas (como o Auto-fit),
    com a variável + cortes propostos (numérica), as faixas repr. × alvo e o
    alvo da folha; trocar a variável fecha e invalida."""
    ui = _build(task)
    _abre_canvas(ui)
    _sel_var_cv(ui, "score")
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_dist(None)
    assert ui._cv_modal_kind == "dist"               # é a janelinha informativa
    assert ui.box_cv_modal.layout.display == ""
    assert ui.btn_cv_modal_ok.layout.display == "none"
    assert ui.btn_cv_modal_cancel.description == "Fechar"
    assert ui.out_cv_dist in ui.box_cv_modal_body.children
    html = ui.out_cv_dist.value
    assert html.count("<img") >= 3                   # hist + faixas + alvo da folha
    assert "cortes propostos" in html
    assert f"Distribuição de {ui._risk_label} na folha" in html
    _sel_var_cv(ui, "garantia")                      # trocar variável…
    assert ui.out_cv_dist.value == ""                # …invalida
    assert ui.box_cv_modal.layout.display == "none"  # …e fecha a janelinha
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_dist(None)                         # categórica: sem histograma
    html = ui.out_cv_dist.value
    assert html.count("<img") == 2 and "cortes propostos" not in html


def test_ui_canvas_modal_iv_das_variaveis(task):
    """O botão à esquerda do Auto-fit abre a janelinha LARGA com IV, PSI por
    amostra e o veredito de uso por variável; é informativa — sem Aplicar, o
    Cancelar vira Fechar — e reabrir uma ação restaura a moldura padrão."""
    ui = _build(task)
    _abre_canvas(ui)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.btn_cv_iv.click()
    assert ui._cv_modal_kind == "iv"
    assert ui.box_cv_modal.layout.display == "" and ui.box_cv_modal.layout.width == "840px"
    assert ui.btn_cv_modal_ok.layout.display == "none"
    assert ui.btn_cv_modal_cancel.description == "Fechar"
    html = ui.out_cv_iv.value
    assert "PSI OOT" in html and "uso" in html
    assert ("recomendada" in html or "cautela" in html or "evitar" in html)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.btn_cv_modal_cancel.click()
        ui.btn_cv_autofit.click()                    # ação restaura a moldura
    assert ui.box_cv_modal.layout.width == "360px"
    assert ui.btn_cv_modal_ok.layout.display == ""
    assert ui.btn_cv_modal_cancel.description == "Cancelar"


def test_ui_canvas_cores_espelham_a_construir(task):
    """A paleta de ação é a mesma nas duas telas: prever/auto-fit=info,
    criar=success, fundir/podar automático=warning/danger etc."""
    ui = _build(task)
    esperado = {"btn_cv_preview": "info", "btn_cv_dist": "info", "btn_cv_iv": "info",
                "btn_cv_autofit": "info", "btn_cv_move_prev": "info",
                "btn_cv_apply": "success",
                "btn_cv_automerge": "warning", "btn_cv_merge_l": "warning",
                "btn_cv_merge_r": "warning", "btn_cv_missing": "warning",
                "btn_cv_lock": "warning", "btn_cv_move": "warning",
                "btn_cv_sugcuts": "warning",
                "btn_cv_prune": "danger", "btn_cv_collapse": "danger",
                "btn_cv_reset": "danger"}
    for nome, estilo in esperado.items():
        assert getattr(ui, nome).button_style == estilo, nome


def test_ui_diag_avaliacao_unificada(task):
    """Placar de saúde e importância são um BLOCO só no Diagnóstico: o card
    abre a aba, "Avaliar modelo" calcula os dois num clique (gráfico antes da
    tabela) e "Ocultar" limpa tudo."""
    ui = _build(task)
    sec = ui.tabs.children[ui._diag_tab_index].children[0]   # 1ª seção (aberta)
    card = sec.children[1].children[0]                       # corpo → card
    filhos = list(card.children)
    assert ui.out_diag in filhos                       # placar e importância…
    assert ui.out_importance_chart in filhos           # …no MESMO card,
    assert filhos.index(ui.out_diag) < filhos.index(ui.out_importance_chart)
    assert filhos.index(ui.out_importance_chart) < filhos.index(ui.out_importance)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui._on_diag(None)                              # UM clique calcula os dois
    assert ui.out_diag.value
    assert "<img" in ui.out_importance_chart.value
    assert "<table" in ui.out_importance.value.lower()
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_diag_hide(None)                         # Ocultar limpa os dois
    assert not ui.out_diag.value and not ui.out_importance.value
    assert not ui.out_importance_chart.value


def test_ui_varprof_toggle_e_tamanho_natural(task):
    """Perfil por safra: o 2º clique RECOLHE as imagens (toggle), e as figuras
    saem no tamanho natural (cap) — com 1–2 variáveis o full_width virava um
    zoom gigante."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui._on_varprofile(None)                        # 1º clique desenha
    assert "<img" in ui.out_varprof_missing.value
    # tamanho natural: nunca amplia (width:auto), só encolhe se faltar espaço
    assert "max-width:100%;width:auto" in ui.out_varprof_missing.value
    assert "width:100%;height" not in ui.out_varprof_missing.value
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_varprofile(None)                        # 2º clique recolhe
    assert ui.out_varprof_missing.value == "" and ui.out_varprof_stats.value == ""
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_varprofile(None)                        # 3º volta a desenhar
    assert "<img" in ui.out_varprof_missing.value


def test_ui_troca_de_teste_refaz_p_das_vizinhas(task):
    """O seletor de teste vive SÓ no Diagnóstico (a view no card de irmãs foi
    removida a pedido) — e trocá-lo lá refaz na hora as pills de p-valor das
    vizinhas no painel do mapa."""
    ui = _build(task)
    w = _abre_canvas(ui)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_apply(None)
    folhas = [s for s, v in ui.seg.segments.items() if v["is_leaf"]]
    _foca(ui, w, folhas[1])
    assert not hasattr(ui, "dd_sib_test")              # a view extra morreu
    assert "Mann-Whitney" in ui.out_cv_merge_p.value
    assert "pill" in ui.out_cv_merge_p.value           # p-valores como pills
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_test.value = "welch"
    assert "Welch" in ui.out_cv_merge_p.value          # painel refeito na hora
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_test.value = "mannwhitney"
    assert "Mann-Whitney" in ui.out_cv_merge_p.value


def test_ui_canvas_janelinha_alterna_no_segundo_clique(task):
    """Os botões que abrem a janelinha (Auto-fit, IV, Distribuições, …) são
    toggles: o 2º clique no MESMO botão fecha; clicar noutro troca o conteúdo."""
    ui = _build(task)
    _abre_canvas(ui)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.btn_cv_autofit.click()
    assert ui.box_cv_modal.layout.display == "" and ui._cv_modal_kind == "fit"
    with contextlib.redirect_stdout(io.StringIO()):
        ui.btn_cv_autofit.click()                      # 2º clique fecha
    assert ui.box_cv_modal.layout.display == "none" and ui._cv_modal_kind is None
    with contextlib.redirect_stdout(io.StringIO()):
        ui.btn_cv_iv.click()
        ui.btn_cv_automerge.click()                    # outro botão TROCA
    assert ui._cv_modal_kind == "merge" and ui.box_cv_modal.layout.display == ""
    with contextlib.redirect_stdout(io.StringIO()):
        ui.btn_cv_modal_cancel.click()
        _sel_var_cv(ui, "score")
        ui._on_cv_dist(None)
    assert ui._cv_modal_kind == "dist"
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_cv_dist(None)                           # toggle das Distribuições
    assert ui.box_cv_modal.layout.display == "none"


def _secao(ui, key):
    """(botão, resumo, corpo, título) da seção `key`."""
    return ui._secoes[key]


def test_ui_secoes_colapsaveis_nas_tres_abas(task):
    """Diagnóstico/Exportar/Avançado viram índices de seções colapsáveis
    (layout do mockup): TODAS as funções continuam montadas — só mudam de
    lugar — e cada aba abre com UMA seção expandida."""
    ui = _build(task)
    chaves = {"diag_aval", "diag_discrim", "diag_folhas", "diag_estab",
              "diag_varprof", "diag_boot", "exp_arquivos", "exp_producao",
              "exp_persist", "av_scn", "av_diff", "av_val"}
    assert chaves <= set(ui._secoes)
    for aberta in ("diag_aval", "exp_arquivos", "av_scn"):
        assert _secao(ui, aberta)[2].layout.display == ""
    for fechada in ("diag_discrim", "diag_estab", "exp_producao", "av_val"):
        assert _secao(ui, fechada)[2].layout.display == "none"
    # nada se perdeu: cada widget-chave é ALCANÇÁVEL dentro de alguma seção
    def _contem(box, alvo):
        if box is alvo:
            return True
        return any(_contem(c, alvo) for c in getattr(box, "children", ()))
    corpos = [ui._secoes[k][2] for k in chaves]
    for w_ in (ui.out_diag, ui.out_importance, ui.out_table, ui.out_sql,
               ui.btn_mlflow, ui.btn_spark_apply, ui.tx_json_path, ui.btn_pdf,
               ui.tx_scn_name, ui.tx_diff_path, ui.btn_validate):
        assert any(_contem(b, w_) for b in corpos)
    btn, _resumo, corpo, _t = _secao(ui, "diag_estab")
    with contextlib.redirect_stdout(io.StringIO()):
        btn.click()
    assert corpo.layout.display == ""                      # abriu
    assert btn.description.startswith("▾")
    with contextlib.redirect_stdout(io.StringIO()):
        btn.click()
    assert corpo.layout.display == "none"                  # fechou
    assert btn.description.startswith("▸")


def test_ui_secoes_chips_de_resumo(task):
    """Os cabeçalhos carregam o resumo: nº de folhas, métrica na amostra de
    comparação e PSI máx (com semáforo) — atualizados a cada mutação."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
    assert "folhas" in _secao(ui, "diag_folhas")[1].value
    assert "PSI" in _secao(ui, "diag_estab")[1].value
    assert "pill" in _secao(ui, "diag_discrim")[1].value
    # cenários: o chip acompanha a lista
    assert "nenhum cenário" in _secao(ui, "av_scn")[1].value
    with contextlib.redirect_stdout(io.StringIO()):
        ui.tx_scn_name.value = "v1"
        ui._on_scn_save(None)
    assert "1 cenário" in _secao(ui, "av_scn")[1].value


def test_ui_migracao_de_folhas_como_heatmap(task):
    """A matriz de migração da comparação de cenários sai como HEATMAP (fundo
    proporcional à contagem, zeros discretos) — não mais texto pré-formatado."""
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui.tx_scn_name.value = "base"
        ui._on_scn_save(None)
        ui._on_scn_compare("base")
    html = ui.out_scn_diff.value
    assert "mig-heat" in html                              # o heatmap novo
    assert 'class="dataframe"' not in html.split("mig-heat")[1]
    assert "background-color:rgb" in html.split("mig-heat")[1]


def test_ui_tema_escuro_repinta_figuras_e_rampas(task):
    """Com o toggle escuro ligado, o _fig_html repinta a figura no kernel
    (fundo grafite, tinta clara, dados intactos) e a rampa accent troca a base
    branca pela grafite — célula branca no escuro brilhava mais que o dado."""
    ui = _build(task)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    claro = ui._accent_ramp_css(0.5, 0.0, 1.0)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.cb_dark.value = True
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="#d6453e")           # cor de DADO: intacta
    ax.set_title("t")
    html = ui._fig_html(fig)                           # repinta + rasteriza
    assert html.startswith("<img")
    escuro = ui._accent_ramp_css(0.5, 0.0, 1.0)
    assert claro != escuro                             # rampa ciente do tema
    assert "255,255,255" not in escuro                 # sem base branca no escuro
    fig2, ax2 = plt.subplots()
    ax2.set_title("x")
    ui._dark_fig(fig2)
    assert ax2.get_facecolor() != (1.0, 1.0, 1.0, 1.0)  # fundo deixou de ser branco
    assert ax2.title.get_color() == "#E8ECF0"
    plt.close(fig2)
    with contextlib.redirect_stdout(io.StringIO()):
        ui.cb_dark.value = False                       # e o aviso do toggle
    assert any("tema" in l.lower() for l in ui._log_lines)


def test_ui_canvas_nao_carrega_nada_da_rede(task):
    """O JS/CSS do canvas não referencia CDN, fonte externa nem @import — mesma
    garantia offline exigida do preview clicável."""
    import re
    from yggdrasil.credit_risk.tree import ui as mod
    blob = mod._TREE_CANVAS_ESM + mod._TREE_CANVAS_CSS
    # o namespace do SVG (createElementNS) é um IDENTIFICADOR, não uma URL que o
    # navegador busque — é a única ocorrência de http:// tolerada aqui
    urls = [u for u in re.findall(r"https?://[^\s\"')]+", blob)
            if u != "http://www.w3.org/2000/svg"]
    assert not urls
    assert "@import" not in blob
    assert not [u for u in re.findall(r"url\(\s*['\"]?([^)'\"]+)", blob)
                if not u.strip().lower().startswith("data:")]
