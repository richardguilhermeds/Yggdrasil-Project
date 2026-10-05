"""Onda 2 (itens 2.1 e 2.2) da árvore: resultados idênticos aos de antes.

- 2.1 ``grow`` só sobe a versão na PRIMEIRA mutação real; na UI, um split que
  não divide nada (e a poda / o auto-merge sem fusões) não refaz a tela nem
  empilha desfazer; o auto-merge reverte o checkpoint quando dá exceção.
- 2.2 ``TreeSegmenter(copy=...)``: ``copy=False`` guarda uma cópia RASA do df
  (compartilha os dados); a UI cria o segmentador assim nos 5 pontos. A cópia
  rasa automática só vale com copy-on-write de fato (``is True`` ou pandas>=3),
  nunca com ``"warn"``.
"""
from __future__ import annotations

import contextlib
import copy as _copy
import io
import json
import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import matplotlib  # noqa: E402
matplotlib.use("Agg")

from yggdrasil.credit_risk.tree import TreeSegmenter  # noqa: E402
from yggdrasil.credit_risk.tree import segmenter as segmod  # noqa: E402

TASKS = ["classification", "regression"]


def _quieto():
    return contextlib.redirect_stdout(io.StringIO())


def _base(task, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.beta(2.5, 3, n) * 1.4 + 0.3
    x[rng.random(n) < 0.06] = np.nan
    gar = rng.choice(list("ABCD"), n, p=[0.5, 0.22, 0.18, 0.1]).astype(object)
    gar[rng.random(n) < 0.05] = None
    lg = {"A": 0.0, "B": 0.10, "C": 0.16, "D": 0.30}
    risco = (0.1 + 0.4 * np.nan_to_num(x - 0.5, nan=0.35)
             + np.array([lg.get(g, 0.2) for g in gar]))
    if task == "classification":
        y = (rng.uniform(0, 1, n) < np.clip(risco, 0.01, 0.95)).astype(float)
    else:
        y = np.clip(risco + rng.normal(0, 0.07, n), 0, 1)     # float64 (to_numpy = view)
    meses = pd.date_range("2023-01-01", periods=10, freq="MS")
    dt = rng.choice(meses, size=n)
    i64 = pd.array(rng.integers(0, 6, n), dtype="Int64")
    i64[rng.random(n) < 0.1] = pd.NA
    bnull = pd.array(rng.random(n) < 0.4, dtype="boolean")
    bnull[rng.random(n) < 0.1] = pd.NA
    dec = rng.choice(np.array([Decimal("1.5"), Decimal("2"), Decimal("3.25"), None],
                              dtype=object), n)
    dt_obj = np.array([pd.Timestamp(d).date() for d in dt], dtype=object)
    df = pd.DataFrame({
        "score": x,
        "garantia": gar,
        "num_00": rng.normal(size=n),
        "const": np.ones(n),                 # variável constante → split no-op
        "flag": (rng.random(n) < 0.2).astype(int),
        "i64": i64,
        "bnull": bnull,
        "dec": dec,
        "dt_obj": dt_obj,
        "saldo": rng.gamma(2.0, 1000.0, n),
        "target": y,
    })
    df["dt_ref"] = dt
    df["amostra"] = np.where(df["dt_ref"] >= meses[7], "OOT", "DES")
    return df


KW = dict(target="target", sample_col="amostra", ref_sample="DES",
          date_col="dt_ref", weight_col="saldo", verbose=False)


def _seg(task, df, **kw):
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TreeSegmenter(df, task_type=task, **KW, **kw)


def _folhas(seg):
    return [s for s, v in seg.segments.items() if v["is_leaf"]]


def _estado(seg):
    """Tudo que a árvore expõe — comparado com igualdade exata."""
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "to_dict": seg.to_dict(),
            "to_sql": seg.to_sql(),
            "regua": json.dumps(seg._regua_dict(), sort_keys=True, default=str),
            "leaves": seg.leaves(),
            "leaves_psi": seg.leaves(with_psi=True),
            "metrics": seg.metrics(),
            "psi": seg.psi(),
            "iv_root": seg.variable_iv("root"),
            "folhas": {s: seg.segments[s]["mask"].to_numpy().copy() for s in _folhas(seg)},
        }


def _igual(a, b):
    assert a.keys() == b.keys()
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, pd.DataFrame):
            pd.testing.assert_frame_equal(va, vb, check_exact=True)
        elif k == "folhas":
            assert va.keys() == vb.keys()
            for s in va:
                np.testing.assert_array_equal(va[s], vb[s])
        else:
            assert va == vb, k


# ======================================================================
# 2.1 — grow só sobe a versão na 1ª mutação real
# ======================================================================
@pytest.mark.parametrize("task", TASKS)
def test_grow_noop_nao_muda_versao_nem_resultado(task, monkeypatch):
    seg = _seg(task, _base(task))
    with _quieto():
        seg.grow("score", splits=[0.8, 1.2])
    e0, v0 = _estado(seg), seg._tree_version
    # o _estado acima já memoizou leaves → um no-op tem de seguir acertando o cache
    chamadas = {"n": 0}
    orig = TreeSegmenter._compute_leaves

    def conta(self, *a, **k):
        chamadas["n"] += 1
        return orig(self, *a, **k)
    monkeypatch.setattr(TreeSegmenter, "_compute_leaves", conta)
    with _quieto():
        seg.grow("const", splits=[0.5])                 # 1 filho não-vazio
        seg.grow("const")                               # ótimo: sem corte válido
        seg.grow("score", splits=[0.8, 1.2], only_segments=["nao_existe"])
    assert seg._tree_version == v0
    _igual(_estado(seg), e0)
    assert chamadas["n"] == 0                           # cache de agregados intacto


@pytest.mark.parametrize("task", TASKS)
def test_grow_valido_sobe_a_versao_uma_vez(task):
    seg = _seg(task, _base(task))
    v0 = seg._tree_version
    with _quieto():
        seg.grow("score", splits=[0.8, 1.2])
    assert seg._tree_version == v0 + 1
    v1 = seg._tree_version
    with _quieto():                                     # várias folhas-alvo: 1 bump só
        seg.grow("garantia")
    assert seg._tree_version == v1 + 1
    assert len(_folhas(seg)) > 3


@pytest.mark.parametrize("task", TASKS)
def test_grow_excecao_na_2a_folha_deixa_cache_coerente(task, monkeypatch):
    seg = _seg(task, _base(task))
    with _quieto():
        seg.grow("score", splits=[0.8, 1.2])
        _ = seg.leaves(), seg._grade_map()              # popula o cache da versão atual
    v0 = seg._tree_version
    orig = TreeSegmenter._resolve_bins_cached
    cont = {"n": 0}

    def explode_na_2a(self, *a, **k):
        cont["n"] += 1
        if cont["n"] == 2:
            raise RuntimeError("falha simulada")
        return orig(self, *a, **k)
    monkeypatch.setattr(TreeSegmenter, "_resolve_bins_cached", explode_na_2a)
    with _quieto(), pytest.raises(RuntimeError):
        seg.grow("garantia")
    # a 1ª folha já foi marcada não-folha → a versão subiu e o cache não é velho
    assert seg._tree_version == v0 + 1
    reais = _folhas(seg)
    with _quieto():
        lv = seg.leaves()
    assert set(lv["segmento"]) == set(reais)
    assert set(seg._grade_map()[0]) == set(reais)


@pytest.mark.parametrize("task", TASKS)
def test_grow_excecao_na_1a_folha_nao_muda_versao(task, monkeypatch):
    seg = _seg(task, _base(task))
    v0 = seg._tree_version

    def explode(self, *a, **k):
        raise RuntimeError("falha simulada")
    monkeypatch.setattr(TreeSegmenter, "_resolve_bins_cached", explode)
    with _quieto(), pytest.raises(RuntimeError):
        seg.grow("score", splits=[0.8, 1.2])
    assert seg._tree_version == v0
    assert _folhas(seg) == ["root"]


# ======================================================================
# 2.1 — UI: no-op não refaz a tela nem empilha desfazer
# ======================================================================
def _ui(task, df=None):
    pytest.importorskip("ipywidgets")
    from yggdrasil.credit_risk.tree import TreeSegmenterUI
    df = _base(task) if df is None else df
    with warnings.catch_warnings(), _quieto():
        warnings.simplefilter("ignore")
        return TreeSegmenterUI(df, target="target", task_type=task, sample_col="amostra",
                               ref_sample="DES", date_col="dt_ref")


def _split(ui, sid, feat, splits):
    ui._pending = dict(feature=feat, splits=splits, only_segments=[sid])
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui._on_split(None)


def _contar_refresh(monkeypatch):
    from yggdrasil.credit_risk.tree import TreeSegmenterUI
    cont = {"n": 0}
    orig = TreeSegmenterUI._refresh

    def wrap(self, *a, **k):
        cont["n"] += 1
        return orig(self, *a, **k)
    monkeypatch.setattr(TreeSegmenterUI, "_refresh", wrap)
    return cont


def _tela(ui):
    return (ui.bar.value, ui.out_sql.value, ui.out_diag.value)


def _ui_com_redo(task):
    """UI com 2 splits feitos e o último desfeito: undo=1, redo=1."""
    ui = _ui(task)
    _split(ui, "root", "score", [0.8, 1.2])
    _split(ui, _folhas(ui.seg)[0], "garantia", None)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui._on_undo(None)
    assert len(ui._undo) == 1 and len(ui._redo) == 1
    return ui


@pytest.mark.parametrize("task", TASKS)
def test_ui_split_noop_preserva_pilhas_e_nao_refaz_tela(task, monkeypatch):
    ui = _ui_com_redo(task)
    undo0, redo0 = _copy.deepcopy(ui._undo), _copy.deepcopy(ui._redo)
    d0, v0, tela0 = ui.seg.to_dict(), ui.seg._tree_version, _tela(ui)
    cont = _contar_refresh(monkeypatch)
    _split(ui, _folhas(ui.seg)[0], "const", [0.5])
    assert cont["n"] == 0
    assert ui._pending is None
    assert ui._undo == undo0 and ui._redo == redo0
    assert ui.seg.to_dict() == d0 and ui.seg._tree_version == v0
    assert _tela(ui) == tela0
    assert not ui.btn_redo.disabled
    # o refazer seguinte refaz a ação REAL anterior (o split de garantia)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui._on_redo(None)
    assert any("garantia" in s for s in ui.seg.segments)


@pytest.mark.parametrize("task", TASKS)
def test_ui_cv_apply_noop(task, monkeypatch):
    ui = _ui_com_redo(task)
    undo0, redo0 = _copy.deepcopy(ui._undo), _copy.deepcopy(ui._redo)
    d0, sel0 = ui.seg.to_dict(), ui._cv_sel
    alvo = _folhas(ui.seg)[0]

    def prep(self=ui):
        ui._pending = dict(feature="const", splits=[0.5], only_segments=[alvo])
        return True, None
    monkeypatch.setattr(ui, "_cv_prepare", prep)
    cont = _contar_refresh(monkeypatch)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui._on_cv_apply(None)
    assert cont["n"] == 0
    assert ui._undo == undo0 and ui._redo == redo0
    assert ui.seg.to_dict() == d0 and ui._cv_sel == sel0


@pytest.mark.parametrize("task", TASKS)
def test_ui_split_valido_sobe_versao_e_undo_restaura(task, monkeypatch):
    ui = _ui(task)
    _split(ui, "root", "score", [0.8, 1.2])
    d0, v0, n_undo = ui.seg.to_dict(), ui.seg._tree_version, len(ui._undo)
    cont = _contar_refresh(monkeypatch)
    _split(ui, _folhas(ui.seg)[0], "garantia", None)
    assert cont["n"] == 1
    assert ui.seg._tree_version > v0
    assert len(ui._undo) == n_undo + 1
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui._on_undo(None)
    assert ui.seg.to_dict() == d0


@pytest.mark.parametrize("task", TASKS)
def test_ui_poda_e_automerge_sem_fusao(task, monkeypatch):
    ui = _ui_com_redo(task)
    ui.locked = set(_folhas(ui.seg))          # tudo travado → nada a fundir
    undo0, redo0 = _copy.deepcopy(ui._undo), _copy.deepcopy(ui._redo)
    d0 = ui.seg.to_dict()
    cont = _contar_refresh(monkeypatch)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui.sl_repr.value = ui.sl_repr.max
        ui._on_prune(None)
        ui._on_automerge(None)
    assert cont["n"] == 0
    assert ui._undo == undo0 and ui._redo == redo0
    assert ui.seg.to_dict() == d0


@pytest.mark.parametrize("task", TASKS)
def test_ui_poda_com_fusao_segue_igual(task, monkeypatch):
    ui = _ui(task)
    _split(ui, "root", "score", [0.35, 1.2])          # (-inf, 0.35] é minúscula
    n_undo = len(ui._undo)
    cont = _contar_refresh(monkeypatch)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui.sl_repr.value = 3.0                        # funde a folha minúscula
        n0 = len(_folhas(ui.seg))
        ui._on_prune(None)
    assert len(_folhas(ui.seg)) < n0
    assert cont["n"] == 1 and len(ui._undo) == n_undo + 1


@pytest.mark.parametrize("task", TASKS)
def test_ui_automerge_excecao_reverte_checkpoint(task, monkeypatch):
    ui = _ui_com_redo(task)
    undo0, redo0 = _copy.deepcopy(ui._undo), _copy.deepcopy(ui._redo)

    def explode(*a, **k):
        raise RuntimeError("falha simulada")
    monkeypatch.setattr(ui.seg, "auto_merge", explode)
    with _quieto():
        ui._on_automerge(None)
    assert ui._undo == undo0 and ui._redo == redo0
    assert not ui.btn_redo.disabled


def _falha_na_2a_fusao(ui, monkeypatch):
    """merge_leaf que funde normalmente na 1ª chamada e levanta na 2ª: simula
    uma exceção no meio de uma ação que funde em rodadas."""
    orig = ui.seg.merge_leaf
    cont = {"n": 0}

    def boom(*a, **k):
        cont["n"] += 1
        if cont["n"] >= 2:
            raise RuntimeError("falha simulada na 2ª fusão")
        return orig(*a, **k)
    monkeypatch.setattr(ui.seg, "merge_leaf", boom)
    return cont


def _acao_com_falha_parcial(ui, monkeypatch, acao):
    """Roda ``acao`` com a 2ª fusão falhando e confere o contrato: a mutação
    parcial fica desfazível (undo cresce 1), a tela é refeita e o ↶ volta
    exatamente ao estado anterior à ação."""
    d0, n0, n_undo = ui.seg.to_dict(), len(_folhas(ui.seg)), len(ui._undo)
    cont_fus = _falha_na_2a_fusao(ui, monkeypatch)
    cont = _contar_refresh(monkeypatch)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acao()
    assert cont_fus["n"] == 2                       # 1 fusão aplicada + a que falhou
    assert len(_folhas(ui.seg)) == n0 - 1           # mutação parcial
    assert len(ui._undo) == n_undo + 1              # snapshot preservado
    assert cont["n"] == 1                           # tela refeita
    assert ui.locked <= set(ui.seg.segments)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui._on_undo(None)
    assert ui.seg.to_dict() == d0
    assert len(_folhas(ui.seg)) == n0


@pytest.mark.parametrize("task", TASKS)
def test_ui_automerge_falha_parcial_fica_desfazivel(task, monkeypatch):
    ui = _ui(task)
    _split(ui, "root", "num_00", [-1.0, -0.5, 0.0, 0.5, 1.0])   # ruído puro
    ui.sl_alpha.value = ui.sl_alpha.max
    ui.sl_gap.value = ui.sl_gap.max
    _acao_com_falha_parcial(ui, monkeypatch, lambda: ui._on_automerge(None))


@pytest.mark.parametrize("task", TASKS)
def test_ui_poda_falha_parcial_fica_desfazivel(task, monkeypatch):
    ui = _ui(task)
    _split(ui, "root", "num_00", [-2.8, -2.4, 2.4, 2.8])       # folhas minúsculas
    ui.sl_repr.value = ui.sl_repr.max
    _acao_com_falha_parcial(ui, monkeypatch, lambda: ui._on_prune(None))


@pytest.mark.parametrize("task", TASKS)
def test_ui_split_falha_parcial_fica_desfazivel(task, monkeypatch):
    """grow com várias folhas-alvo que falha na 2ª: a 1ª divisão fica aplicada
    e desfazível (antes o checkpoint era descartado e o ↶ pulava este estado)."""
    ui = _ui(task)
    _split(ui, "root", "score", [0.8, 1.2])
    alvos = _folhas(ui.seg)[:2]
    d0, n_undo = ui.seg.to_dict(), len(ui._undo)
    orig = TreeSegmenter._resolve_bins_cached
    cont_g = {"n": 0}

    def boom(self, *a, **k):
        cont_g["n"] += 1
        if cont_g["n"] == 2:
            raise RuntimeError("falha simulada na 2ª folha")
        return orig(self, *a, **k)
    monkeypatch.setattr(TreeSegmenter, "_resolve_bins_cached", boom)
    cont = _contar_refresh(monkeypatch)
    ui._pending = dict(feature="garantia", splits=None, only_segments=alvos)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui._on_split(None)
    monkeypatch.setattr(TreeSegmenter, "_resolve_bins_cached", orig)
    assert cont_g["n"] >= 2
    assert ui.seg.to_dict() != d0                   # a 1ª folha foi dividida
    assert len(ui._undo) == n_undo + 1 and cont["n"] == 1
    assert ui._pending is None
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui._on_undo(None)
    assert ui.seg.to_dict() == d0


# ======================================================================
# 2.2 — copy=False / detecção de copy-on-write
# ======================================================================
def _cow_opcao_existe():
    try:
        pd.options.mode.copy_on_write
        return True
    except Exception:
        return False


_PANDAS3 = int(pd.__version__.split(".")[0]) >= 3


def test_deteccao_cow(monkeypatch):
    if not _PANDAS3 and _cow_opcao_existe():
        with pd.option_context("mode.copy_on_write", False):
            assert segmod._copy_on_write_ativo() is False
        with pd.option_context("mode.copy_on_write", True):
            assert segmod._copy_on_write_ativo() is True
        try:
            ctx = pd.option_context("mode.copy_on_write", "warn")
            ctx.__enter__()
        except Exception:
            ctx = None
        if ctx is not None:
            try:
                assert pd.options.mode.copy_on_write == "warn"
                assert segmod._copy_on_write_ativo() is False     # "warn" não é CoW
            finally:
                ctx.__exit__(None, None, None)
    monkeypatch.setattr(segmod.pd, "__version__", "3.0.0")
    assert segmod._copy_on_write_ativo() is True


def _compartilha(seg, df, col="num_00"):
    return np.shares_memory(seg.df[col].to_numpy(), df[col].to_numpy())


@pytest.mark.skipif(_PANDAS3, reason="pandas 3: CoW sempre ligado")
def test_copy_true_sem_cow_e_profunda_e_copy_false_compartilha():
    df = _base("classification", n=500)
    if _cow_opcao_existe():
        with pd.option_context("mode.copy_on_write", False):
            assert not _compartilha(_seg("classification", df), df)
            assert _compartilha(_seg("classification", df, copy=False), df)
        try:
            with pd.option_context("mode.copy_on_write", "warn"):
                assert not _compartilha(_seg("classification", df), df)
        except (ValueError, KeyError, pd.errors.OptionError):
            pass                                   # pandas sem o modo "warn"
    else:
        assert not _compartilha(_seg("classification", df), df)
        assert _compartilha(_seg("classification", df, copy=False), df)


@pytest.mark.skipif(not _PANDAS3 and not _cow_opcao_existe(), reason="sem CoW")
def test_cow_ligado_isola_seg_da_escrita_no_df():
    df = _base("classification", n=500)
    ctx = (contextlib.nullcontext() if _PANDAS3
           else pd.option_context("mode.copy_on_write", True))
    with ctx:
        seg = _seg("classification", df)                  # copy=True → rasa sob CoW
        antes = seg.df["num_00"].to_numpy().copy()
        df.loc[0, "num_00"] = 12345.0
        np.testing.assert_array_equal(seg.df["num_00"].to_numpy(), antes)


def _roteiro(seg, df, tmp_path, copy_flag):
    """Splits num/cat/no-op, fit_auto numa subárvore, poda, auto-merge, collapse,
    from_dict com mask_cache e load de JSON — devolve os estados ao longo dele."""
    estados = []
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.grow("score", splits=[0.8, 1.2])
        estados.append(_estado(seg))
        f = _folhas(seg)
        seg.grow("garantia", only_segments=[f[0]])
        seg.grow("const", splits=[0.5])
        seg.grow("dec", only_segments=[f[-1]])
        estados.append(_estado(seg))
        seg.fit_auto(max_depth=1, subtree=f[1], from_scratch=False, verbose=False)
        estados.append(_estado(seg))
        seg.auto_merge(alpha=0.5, verbose=False)
        seg.prune(min_repr=3.0, verbose=False)
        estados.append(_estado(seg))
        pai = next(s for s, v in seg.segments.items() if not v["is_leaf"] and s != "root")
        seg.collapse(pai, verbose=False)
        estados.append(_estado(seg))
        cache = seg._prime_mask_cache()
        outro = TreeSegmenter.from_dict(seg.to_dict(), df, mask_cache=cache,
                                        copy=copy_flag)
        estados.append(_estado(outro))
        p = tmp_path / f"arvore_{copy_flag}.json"
        seg.save(str(p))
        lido = TreeSegmenter.load(str(p), df, copy=copy_flag)
        estados.append(_estado(lido))
        lido.fit_auto(max_depth=2, verbose=False)
        estados.append(_estado(lido))
    return estados


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("cow", [False, True])
def test_roteiro_copy_true_e_false_iguais(task, cow, tmp_path):
    if cow and not _PANDAS3 and not _cow_opcao_existe():
        pytest.skip("sem CoW")
    df = _base(task)
    snap = df.copy(deep=True)
    ctx = (pd.option_context("mode.copy_on_write", cow)
           if not _PANDAS3 and _cow_opcao_existe() else contextlib.nullcontext())
    with ctx:
        seg_t = _seg(task, df, copy=True)
        seg_f = _seg(task, df, copy=False)
        assert _compartilha(seg_f, df)
        a = _roteiro(seg_t, df, tmp_path, True)
        b = _roteiro(seg_f, df, tmp_path, False)
    assert len(a) == len(b)
    for ea, eb in zip(a, b):
        _igual(ea, eb)
    # nada no segmentador escreve no df do chamador
    pd.testing.assert_frame_equal(df, snap, check_exact=True)


def test_copy_nao_vai_para_o_json():
    df = _base("classification", n=500)
    seg = _seg("classification", df, copy=False)
    d = seg.to_dict()
    assert "copy" not in d["meta"]
    assert d == _seg("classification", df).to_dict()


# ======================================================================
# 2.2 — a UI cria o segmentador com copy=False nos 5 pontos
# ======================================================================
@pytest.mark.parametrize("task", TASKS)
def test_ui_usa_copy_false_em_todas_as_construcoes(task, monkeypatch, tmp_path):
    ui = _ui(task)
    assert _compartilha(ui.seg, ui.df)
    vistos = []
    orig = TreeSegmenter.__init__

    def espiao(self, df, *a, copy=True, **k):
        vistos.append(copy)
        return orig(self, df, *a, copy=copy, **k)
    monkeypatch.setattr(TreeSegmenter, "__init__", espiao)
    _split(ui, "root", "score", [0.8, 1.2])
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # comparação com JSON salvo (load) e com cenário em memória (from_dict)
        p = tmp_path / "b.json"
        ui.seg.save(str(p))
        ui.tx_diff_path.value = str(p)
        ui._on_diff(None)
        ui._on_scn_save(None)
        nome = next(iter(ui._scenarios))
        ui._on_scn_compare(nome)
        ui._on_reset(None)                               # botão Resetar
        assert _compartilha(ui.seg, ui.df)
        ui._on_cv_reset(ui.btn_cv_reset)                 # 2 cliques: arma e confirma
        ui._on_cv_reset(ui.btn_cv_reset)
        assert _compartilha(ui.seg, ui.df)
    assert vistos == [False] * 4, vistos
    # e a UI não escreveu nada no df do usuário
    pd.testing.assert_frame_equal(ui.df, _base(task), check_exact=True)
