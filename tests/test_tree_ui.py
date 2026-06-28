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


def test_ui_plot_tree(task):
    ui = _build(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_autofit(None)
        ui._on_tree_preview(None)
    assert ui.out_tree_img.value and "img" in ui.out_tree_img.value.lower()


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
        # botões 1 e 2: clf = ROC/KS (out_discrim) · reg = boxplot/hist
        ui.btn_roc.click()
        ui.btn_ks.click()
    if task == "classification":
        assert ui.out_discrim.value and "Erro" not in ui.out_discrim.value
    else:
        # boxplot vai p/ out_discrim; histograma p/ out_quality
        assert ui.out_discrim.value and "Erro" not in ui.out_discrim.value
        assert ui.out_quality.value and "Erro" not in ui.out_quality.value
