"""Onda 1 de desempenho da TreeSegmenterUI: os caminhos novos têm de dar o MESMO
resultado que os antigos (velho × novo, igualdade exata).

- 1.5 tipo da variável lido da base inteira (sem recortar a folha com todas as
  colunas) em ``_feature_kind``/``_cv_kind``/``_parse_cuts``; ``cat_box`` com só
  a variável e o alvo;
- 1.6 uma renderização só do canvas e do painel na construção;
- 1.7 ``variable_inversion`` e ``variable_by_safra`` calculados uma vez por
  análise e entregues aos gráficos (``inv=``/``bs=``); o gráfico por amostra
  sem ``inv`` pula o laço por safra (sentinela ``_SEM_SAFRAS``).

As versões "velhas" dos ``_rebuild_*cat_box`` são reconstruídas do fonte ATUAL
trocando só o recorte pelo antigo: se o código mudar e a troca não casar, o
teste acusa.
"""
from __future__ import annotations

import contextlib
import copy
import inspect
import io
import re
import textwrap
import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ipywidgets")

import matplotlib  # noqa: E402
matplotlib.use("Agg")

from yggdrasil.credit_risk.tree import TreeSegmenter, TreeSegmenterUI  # noqa: E402
from yggdrasil.credit_risk.tree import segmenter as segmod  # noqa: E402
from yggdrasil.credit_risk.tree import ui as uimod  # noqa: E402

TASKS = ["classification", "regression"]


# ----------------------------------------------------------------------
# Fixture: base com os tipos "chatos" (anuláveis, Decimal, category, bool,
# date object, índice duplicado) e uma 2ª coluna de tempo ≠ date_col
# ----------------------------------------------------------------------
def _base(task, n=2500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.beta(2.5, 3, n) * 1.4 + 0.3
    x[rng.random(n) < 0.06] = np.nan
    x[:3] = [np.inf, -np.inf, -0.0]
    gar = rng.choice(list("ABCD"), n, p=[0.5, 0.22, 0.18, 0.1]).astype(object)
    gar[rng.random(n) < 0.05] = None
    lg = {"A": 0.0, "B": 0.10, "C": 0.16, "D": 0.30}
    risco = (0.1 + 0.4 * np.nan_to_num(x - 0.5, nan=0.35, posinf=0.5, neginf=-0.2)
             + np.array([lg.get(g, 0.2) for g in gar]))
    if task == "classification":
        y = (rng.uniform(0, 1, n) < np.clip(risco, 0.01, 0.95)).astype(float)
    else:
        y = np.clip(risco + rng.normal(0, 0.07, n), 0, 1)
    meses = pd.date_range("2023-01-01", periods=10, freq="MS")
    dt = rng.choice(meses, size=n)
    i64 = pd.array(rng.integers(0, 6, n), dtype="Int64")
    i64[rng.random(n) < 0.1] = pd.NA
    f64 = pd.array(rng.normal(size=n), dtype="Float64")
    f64[rng.random(n) < 0.1] = pd.NA
    bnull = pd.array(rng.random(n) < 0.4, dtype="boolean")
    bnull[rng.random(n) < 0.1] = pd.NA
    dec = rng.choice(np.array([Decimal("1.5"), Decimal("1.50"), Decimal("2"),
                               Decimal("NaN"), None], dtype=object), n)
    dt_obj = np.array([pd.Timestamp(d).date() for d in dt], dtype=object)
    dt_obj[rng.random(n) < 0.02] = None
    df = pd.DataFrame({
        "score": x,
        "garantia": gar,
        "x_f32": rng.normal(size=n).astype("float32"),
        "i64": i64,
        "f64": f64,
        "bnull": bnull,
        "b_nat": rng.random(n) < 0.3,
        "flag": (rng.random(n) < 0.2).astype(int),
        "cat_cat": pd.Categorical(rng.choice(["z", "y", "w"], n),
                                  categories=["z", "y", "w"]),
        "cat_str": pd.array(rng.choice(["p", "q", None], n), dtype="string"),
        "dec": dec,
        "nivel_raro": np.where(np.arange(n) < 3, "raro", "comum").astype(object),
        "allna": np.full(n, np.nan),
        "dt_obj": dt_obj,
        "target": y,
    })
    df["dt_ref"] = dt
    df["dt_alt"] = rng.choice(pd.date_range("2022-01-01", periods=6, freq="MS"), size=n)
    df["amostra"] = np.where(df["dt_ref"] >= meses[7], "OOT", "DES")
    df.index = np.r_[np.arange(n // 2), np.arange(n - n // 2)]      # índice duplicado
    return df


FEATS = ["score", "garantia", "x_f32", "i64", "f64", "bnull", "b_nat", "flag",
         "cat_cat", "cat_str", "dec", "nivel_raro", "allna", "dt_obj"]


def _ui(task, df=None, **kw):
    df = _base(task) if df is None else df
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return TreeSegmenterUI(df, target="target", task_type=task, sample_col="amostra",
                               ref_sample="DES", date_col="dt_ref", **kw)


def _quieto():
    return contextlib.redirect_stdout(io.StringIO())


def _split_manual(ui, sid, feat, cortes=None):
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui.dd_leaf.value = sid
        ui.dd_feature.value = next(l for l, c in ui._feat_by_label.items() if c == feat)
        ui.tg_mode.value = "Manual"
        if cortes is not None:
            ui.tx_cuts.value = cortes
        ui._on_preview(None)
        assert ui._pending is not None
        ui._on_split(None)


def _arvore_3_splits(ui):
    """3 quebras: score na raiz, garantia numa folha e nivel_raro noutra (que
    deixa uma folha minúscula); devolve os sids."""
    _split_manual(ui, "root", "score", "0.8, 1.2")
    folhas = [s for s, v in ui.seg.segments.items() if v["is_leaf"]]
    _split_manual(ui, folhas[0], "garantia")
    folhas = [s for s, v in ui.seg.segments.items() if v["is_leaf"]]
    _split_manual(ui, folhas[-1], "nivel_raro")
    return list(ui.seg.segments)


# ======================================================================
# 1.5 — tipo da variável sem recortar a folha
# ======================================================================
@pytest.mark.parametrize("task", TASKS)
def test_detect_kind_recorte_nao_muda_o_tipo(task):
    """Propriedade em que o 1.5 se apoia: o recorte de linhas (raiz, folhas,
    folha vazia, tudo/nada) não muda o dtype, logo nem o tipo."""
    ui = _ui(task)
    sids = _arvore_3_splits(ui)
    df, seg = ui.df, ui.seg
    mascaras = [seg.segments[s]["mask"] for s in sids]
    mascaras += [pd.Series(False, index=df.index), pd.Series(True, index=df.index)]
    for f in FEATS + ["target"]:
        k = seg._detect_kind(df, f, None)
        for m in mascaras:
            assert seg._detect_kind(df[m], f, None) == k, f


@pytest.mark.parametrize("task", TASKS)
def test_feature_kind_cv_kind_parse_cuts_iguais_ao_recorte(task):
    ui = _ui(task)
    sids = _arvore_3_splits(ui)
    for s in sids:
        sub = ui.df[ui.seg.segments[s]["mask"]]          # versão antiga
        for f in FEATS:
            velho = ui.seg._detect_kind(sub, f, None)
            assert ui._cv_kind(f, s) == velho, (s, f)
            ui._sel_feature = (lambda warn=True, _f=f: _f)
            ui._suspend_leaf_obs = True
            try:
                if s in [x for _, x in ui.dd_leaf.options]:
                    ui.dd_leaf.value = s
                    assert ui._feature_kind() == velho, (s, f)
            finally:
                ui._suspend_leaf_obs = False
                del ui._sel_feature
    # _parse_cuts: mesmo tipo → mesmo retorno (num lê a caixa de cortes)
    ui.tx_cuts.value = "0.5; 1,0.9"
    assert ui._parse_cuts("score", sids[-1]) == [0.5, 1.0, 0.9]


@pytest.mark.parametrize("task", TASKS)
def test_caminhos_de_erro_preservados(task):
    ui = _ui(task)
    assert ui._cv_kind("nao_existe", "root") == "num"     # feature ausente → "num"
    with pytest.raises(KeyError):
        ui._cv_kind("score", "sid_inexistente")          # sid inválido → KeyError
    with pytest.raises(KeyError):
        ui._parse_cuts("score", "sid_inexistente")
    assert ui._cv_kind(None, None) in ("num", "cat")     # guarda de None intacta


_NOVO_CAT = re.compile(r'sub = self\.df\.loc\[self\.seg\.segments\[sid\]\["mask"\],\s*'
                       r'list\(dict\.fromkeys\(\[feat, self\.target\]\)\)\]')
_VELHO_CAT = 'sub = self.df[self.seg.segments[sid]["mask"]]'


def _cat_box_velho(nome):
    src = textwrap.dedent(inspect.getsource(getattr(TreeSegmenterUI, nome)))
    src, n = _NOVO_CAT.subn(_VELHO_CAT, src)
    assert n == 1, f"recorte novo não encontrado em {nome}"
    ns: dict = {}
    exec(compile(src, f"<{nome}_velho>", "exec"), dict(vars(uimod)), ns)
    return ns[nome]


def _estado_caixa(box):
    out = []
    for c in box.children:
        if hasattr(c, "children") and len(c.children) == 2:
            dd, lab = c.children
            out.append((tuple(dd.options), dd.value, lab.value))
        else:
            out.append(c.value)
    return out


@pytest.mark.parametrize("task", TASKS)
def test_cat_box_velho_x_novo(task):
    """Os dois agrupadores de categorias (Construir e painel do canvas) saem
    idênticos: ordem, médias formatadas, NaN e os grupos — inclusive com a
    variável igual ao alvo (sem coluna duplicada) e numa folha minúscula."""
    ui = _ui(task)
    sids = _arvore_3_splits(ui)
    velho_b = _cat_box_velho("_rebuild_cat_box")
    velho_cv = _cat_box_velho("_rebuild_cv_cat_box")
    for s in sids:
        for f in ["garantia", "cat_cat", "cat_str", "dec", "bnull", "b_nat", "flag",
                  "nivel_raro", "allna", "dt_obj", "target", "score"]:
            ui._sel_feature = (lambda warn=True, _f=f: _f)
            ui._cv_feature = (lambda warn=True, _f=f: _f)
            ui._suspend_leaf_obs = True
            if s not in [x for _, x in ui.dd_leaf.options]:
                ui.dd_leaf.options = list(ui.dd_leaf.options) + [(s, s)]
            ui.dd_leaf.value = s
            ui._suspend_leaf_obs = False
            ui._cv_sel = s
            res = []
            for fn_b, fn_cv in ((velho_b, velho_cv),
                                (TreeSegmenterUI._rebuild_cat_box,
                                 TreeSegmenterUI._rebuild_cv_cat_box)):
                ui._cat_ctx = ui._cv_cat_ctx = None
                ui._cat_widgets, ui._cv_cat_widgets = {}, {}
                fn_b(ui)
                fn_cv(ui)
                res.append((_estado_caixa(ui.cat_box), _estado_caixa(ui.cv_cat_box),
                            ui._cat_groups(), ui._cv_cat_groups()))
            assert res[0] == res[1], (s, f)
            del ui._sel_feature, ui._cv_feature


@pytest.mark.parametrize("task", TASKS)
def test_medias_do_cat_box_bit_a_bit(task):
    """Médias por categoria do recorte de 2 colunas × recorte da folha inteira
    (assert_series_equal exato, com ordem), na base com índice duplicado."""
    ui = _ui(task)
    sids = _arvore_3_splits(ui)
    for s in sids:
        m = ui.seg.segments[s]["mask"]
        for f in ["garantia", "cat_cat", "dec", "target"]:
            subs = (ui.df[m], ui.df.loc[m, list(dict.fromkeys([f, "target"]))])
            out = []
            for sub in subs:
                valid = sub[sub[f].notna()]
                means = (valid.assign(_c=valid[f].astype(str))
                         .groupby("_c")["target"].mean().sort_values())
                out.append((means, int(sub[f].isna().sum())))
            pd.testing.assert_series_equal(out[0][0], out[1][0], check_exact=True)
            assert out[0][1] == out[1][1]


@pytest.mark.parametrize("task", TASKS)
def test_tipo_nao_recorta_a_base_da_ui(task, monkeypatch):
    """Nenhum recorte booleano de linhas da base da UI com todas as colunas
    durante o preview, a troca de modo e a sincronia do painel."""
    ui = _ui(task)
    _arvore_3_splits(ui)
    folha = [s for s, v in ui.seg.segments.items() if v["is_leaf"]][0]
    orig = pd.DataFrame.__getitem__
    alvo = ui.df

    def espiao(self, key):
        if (self is alvo and isinstance(key, (pd.Series, np.ndarray))
                and len(key) == len(self) and getattr(key, "dtype", None) == bool):
            raise AssertionError("recorte booleano da base da UI")
        return orig(self, key)

    monkeypatch.setattr(pd.DataFrame, "__getitem__", espiao)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in ["score", "garantia"]:
            ui.dd_leaf.value = folha
            ui.dd_feature.value = next(l for l, c in ui._feat_by_label.items() if c == f)
            for modo in ("Manual", "Ótimo"):
                ui.tg_mode.value = modo
                ui._on_mode_change(None)
                if modo == "Manual" and f == "score":
                    ui.tx_cuts.value = "1.0"
                ui._on_preview(None)
            ui._cv_sel = folha
            for modo in ("Manual", "Ótimo"):
                ui.tg_cv_mode.value = modo
                ui._sync_cv_mode()


# ======================================================================
# 1.6 — uma renderização só na construção
# ======================================================================
def _tem_anywidget():
    try:
        import anywidget  # noqa: F401
        return True
    except Exception:
        return False


class _UIVelha(TreeSegmenterUI):
    """Emula a construção antiga: render eager no _build (antes do _refresh) e
    o _refresh redesenhando de novo (o _cv_rendered já é True)."""

    def _build(self):
        super()._build()
        if self.tabs.selected_index == self._canvas_tab_index:
            self._on_tab_change({"new": self._canvas_tab_index})
        else:
            self.tabs.selected_index = self._canvas_tab_index


def _contador(monkeypatch, *nomes):
    cont = {n: 0 for n in nomes}
    for n in nomes:
        orig = getattr(TreeSegmenterUI, n)

        def wrap(self, *a, _o=orig, _n=n, **k):
            cont[_n] += 1
            return _o(self, *a, **k)
        monkeypatch.setattr(TreeSegmenterUI, n, wrap)
    return cont


def _estado_canvas(ui):
    w = ui._cv_widget
    est = {
        "cv_sel": ui._cv_sel, "rendered": ui._cv_rendered, "dirty": ui._cv_dirty,
        "feat_ctx": ui._feat_ctx,
        "cv_feature": (tuple(ui.dd_cv_feature.options), ui.dd_cv_feature.value),
        "cv_goto": (tuple(ui.dd_cv_goto.options), ui.dd_cv_goto.index),
        "sug": [b.description for b in ui.btns_cv_sug],
        "leaf": (tuple(ui.dd_leaf.options), ui.dd_leaf.value),
        "displays": {k: v.layout.display for k, v in vars(ui).items()
                     if k.startswith(("box_cv", "btn_cv", "sl_cv", "dd_cv", "cv_"))
                     and hasattr(v, "layout")},
        "outs": {k: v.value for k, v in vars(ui).items()
                 if k.startswith("out_cv") and hasattr(v, "value")},
    }
    if w is not None:
        est["w"] = (w.nodes, w.edges, w.selected, w.center_token, w.fit_token,
                    w.content_w, w.content_h)
    return est


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("interativo", [True, False])
def test_construcao_renderiza_uma_vez(task, interativo, monkeypatch):
    if interativo and not _tem_anywidget():
        pytest.skip("anywidget não instalado")
    df = _base(task)
    cont = _contador(monkeypatch, "_canvas_layout", "_refresh_cv_panel")
    ui = _ui(task, df=df, allow_interactive_tree=interativo)
    assert cont["_refresh_cv_panel"] == 1
    assert cont["_canvas_layout"] == (1 if interativo else 0)
    assert ui._cv_rendered is True and ui._cv_dirty is False
    assert ui._cv_sel == "root" and ui.dd_leaf.value == "root"
    if interativo:
        assert ui._cv_widget.center_token == 1        # exatamente 1 centralização
        assert ui._cv_widget.selected == "root"
    # mesmo estado final que a construção antiga (2 renderizações)
    for k in cont:
        cont[k] = 0
    with warnings.catch_warnings(), _quieto():
        warnings.simplefilter("ignore")
        velha = _UIVelha(df, target="target", task_type=task, sample_col="amostra",
                         ref_sample="DES", date_col="dt_ref",
                         allow_interactive_tree=interativo)
    assert cont["_refresh_cv_panel"] == 2             # a velha pagava 2×
    assert _estado_canvas(ui) == _estado_canvas(velha)


@pytest.mark.parametrize("task", TASKS)
def test_troca_de_aba_nao_redesenha_e_split_redesenha(task, monkeypatch):
    if not _tem_anywidget():
        pytest.skip("anywidget não instalado")
    ui = _ui(task, allow_interactive_tree=True)
    cont = _contador(monkeypatch, "_canvas_layout", "_refresh_cv_panel")
    ui.tabs.selected_index = ui._iv_tab_index
    ui.tabs.selected_index = ui._canvas_tab_index
    assert cont["_canvas_layout"] == 0 and ui._cv_dirty is False
    # split com o canvas à vista: redesenha na hora
    _split_manual(ui, "root", "score", "1.0")
    assert cont["_canvas_layout"] >= 1
    assert ui._cv_dirty is False
    n_nos = len(ui._cv_widget.nodes)
    assert n_nos == len(ui.seg.segments)
    # mutação com o canvas escondido: fica pendente e desenha ao voltar
    ui.tabs.selected_index = ui._iv_tab_index
    folhas = [s for s, v in ui.seg.segments.items() if v["is_leaf"]]
    _split_manual(ui, folhas[0], "garantia")
    assert ui._cv_dirty is True
    antes = cont["_canvas_layout"]
    ui.tabs.selected_index = ui._canvas_tab_index
    assert cont["_canvas_layout"] == antes + 1 and ui._cv_dirty is False
    assert len(ui._cv_widget.nodes) == len(ui.seg.segments)


# ======================================================================
# 1.7 — inversão e percentis por safra uma vez por análise
# ======================================================================
def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=60)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return buf.getvalue()


def _seg(task):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = TreeSegmenter(_base(task), target="target", task_type=task,
                            sample_col="amostra", date_col="dt_ref", verbose=False)
        seg.grow("score", splits=[0.8, 1.2])
    return seg


def _igual(a, b):
    """Igualdade EXATA que trata NaN == NaN (o ``==`` de dict/lista falha com dois
    ``float('nan')`` distintos — p.ex. uma faixa vazia numa safra —, até entre
    duas chamadas idênticas do código antigo)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(_igual(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(_igual(x, y) for x, y in zip(a, b)))
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return type(a) is type(b) and a == b


_CHAVES_AMOSTRA = ("ordered", "labels", "ref_risco")
_CHAVES_SERIE_AMOSTRA = ("ordered", "labels", "ref_risco", "xs_sample", "ser_sample")


@pytest.mark.parametrize("task", TASKS)
def test_inversion_sem_safras_e_por_tcol_mantem_o_por_amostra(task):
    seg = _seg(task)
    folhas = [s for s, v in seg.segments.items() if v["is_leaf"]]
    for sid in [None, "root", folhas[0], "sid_inexistente"]:
        for f in ["score", "garantia", "i64", "dec"]:
            base = seg.variable_inversion(f, sid=sid)                   # date_col
            alt = seg.variable_inversion(f, sid=sid, time_col="dt_alt")
            sem = seg.variable_inversion(f, sid=sid, time_col=segmod._SEM_SAFRAS)
            if base["series"] is None:
                assert alt["series"] is None and sem["series"] is None
                assert _igual(sem, base)
                continue
            for r in (alt, sem):
                for k in _CHAVES_AMOSTRA:
                    assert _igual(r[k], base[k])
                assert _igual(r["samples"], base["samples"])
                assert _igual(r["sample_inv"], base["sample_inv"])
                for k in _CHAVES_SERIE_AMOSTRA:
                    assert _igual(r["series"][k], base["series"][k])
            assert sem["series"]["xs_safra"] == [] and sem["safras"] == []
            assert sem["n_safras"] == 0
            # o tcol explícito igual ao date_col dá o MESMO dict
            assert _igual(seg.variable_inversion(f, sid=sid, time_col="dt_ref"), base)


@pytest.mark.parametrize("task", TASKS)
def test_plots_com_inv_e_bs_saem_iguais(task):
    """PNG idêntico com e sem a passagem, para tcol == date_col e tcol ≠ date_col;
    e os plots não mutam o que receberam."""
    seg = _seg(task)
    folhas = [s for s, v in seg.segments.items() if v["is_leaf"]]
    for sid in ["root", folhas[-1]]:
        for f in ["score", "garantia", "i64"]:
            for tcol in ["dt_ref", "dt_alt"]:
                inv_velho = seg.variable_inversion(f, sid=sid)       # o que o velho usava
                inv = seg.variable_inversion(f, sid=sid, time_col=tcol)
                guarda = copy.deepcopy(inv)
                ref = _png(seg.plot_variable_inversion_by_sample(f, sid=sid, inv=inv_velho))
                assert _png(seg.plot_variable_inversion_by_sample(f, sid=sid)) == ref
                assert _png(seg.plot_variable_inversion_by_sample(f, sid=sid, inv=inv)) == ref
                ref_t = _png(seg.plot_variable_inversion_by_safra(f, sid=sid, time_col=tcol))
                assert _png(seg.plot_variable_inversion_by_safra(
                    f, sid=sid, time_col=tcol, inv=inv)) == ref_t
                assert inv == guarda
                if seg._detect_kind(seg.df, f, None) == "num":
                    bs = seg.variable_by_safra(f, tcol, sid=sid)
                    bs0 = bs.copy()
                    ref_s = _png(seg.plot_variable_timeseries(f, tcol, sid=sid))
                    assert _png(seg.plot_variable_timeseries(f, tcol, sid=sid, bs=bs)) == ref_s
                    pd.testing.assert_frame_equal(bs, bs0, check_exact=True)
                    pd.testing.assert_frame_equal(
                        bs, seg.variable_by_safra(f, tcol, sid=sid), check_exact=True)


def _analisa(ui, f, sid, tcol):
    ui.dd_var_leaf.value = sid
    ui.dd_var.value = next(l for l, c in ui._var_by_label.items() if c == f)
    ui.tx_var_time.value = tcol
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui._on_var_analyze(None)
    return {o: getattr(ui, o).value for o in
            ("out_var_cards", "out_var_dist", "out_var_time", "out_var_psi",
             "out_var_inv_s", "out_var_inv_t", "out_var_optbin")}


@pytest.mark.parametrize("task", TASKS)
def test_analise_calcula_uma_vez_e_sai_igual(task, monkeypatch):
    ui = _ui(task)
    _split_manual(ui, "root", "score", "1.0")
    folha = [s for s, v in ui.seg.segments.items() if v["is_leaf"]][0]
    seg = ui.seg
    chamadas = {"inv": [], "bs": []}
    inv_orig, bs_orig = TreeSegmenter.variable_inversion, TreeSegmenter.variable_by_safra

    def inv_espiao(self, feature, sid=None, time_col=None, **k):
        chamadas["inv"].append(time_col)
        return inv_orig(self, feature, sid=sid, time_col=time_col, **k)

    def bs_espiao(self, feature, time_col=None, sid=None, sample=None):
        chamadas["bs"].append(time_col)
        return bs_orig(self, feature, time_col, sid=sid, sample=sample)

    monkeypatch.setattr(TreeSegmenter, "variable_inversion", inv_espiao)
    monkeypatch.setattr(TreeSegmenter, "variable_by_safra", bs_espiao)
    for sid in ["root", folha]:
        for f in ["score", "garantia"]:
            for tcol in ["dt_ref", "dt_alt", ""]:
                chamadas["inv"].clear(); chamadas["bs"].clear()
                out = _analisa(ui, f, sid, tcol)
                if tcol:
                    assert chamadas["inv"] == [tcol]          # antes: 2 (date_col + tcol)
                    num = seg._detect_kind(seg.df, f, None) == "num"
                    assert chamadas["bs"] == ([tcol] if num else [])   # antes: 2
                else:
                    assert chamadas["inv"] == [segmod._SEM_SAFRAS]
                    assert chamadas["bs"] == []
                # o que o código antigo desenhava, montado à mão pelo caminho antigo
                fh = lambda fig, **kw: ui._fig_html(fig, full_width=True, **kw)  # noqa: E731
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    esperado_s = fh(seg.plot_variable_inversion_by_sample(
                        f, sid=sid, inv=inv_orig(seg, f, sid=sid)))
                    assert out["out_var_inv_s"] == esperado_s
                    if tcol:
                        assert out["out_var_inv_t"] == fh(seg.plot_variable_inversion_by_safra(
                            f, sid=sid, time_col=tcol,
                            inv=inv_orig(seg, f, sid=sid, time_col=tcol)))
                        bs_v = (bs_orig(seg, f, tcol, sid=sid)
                                if seg._detect_kind(seg.df, f, None) == "num" else None)
                        assert out["out_var_time"] == fh(seg.plot_variable_timeseries(
                            f, tcol, sid=sid, figsize=(8.6, 4.2), bs=bs_v), tight=False)


@pytest.mark.parametrize("task", TASKS)
def test_analise_erro_da_inversao_segue_virando_aviso(task, monkeypatch):
    ui = _ui(task)
    inv_orig = TreeSegmenter.variable_inversion

    def falha_no_tempo(self, feature, sid=None, time_col=None, **k):
        if time_col == "dt_alt":
            raise ZeroDivisionError("teste")
        return inv_orig(self, feature, sid=sid, time_col=time_col, **k)

    monkeypatch.setattr(TreeSegmenter, "variable_inversion", falha_no_tempo)
    out = _analisa(ui, "score", "root", "dt_alt")
    assert "inversão por safra não gerada: ZeroDivisionError" in out["out_var_inv_t"]
    assert out["out_var_inv_s"].startswith("<img")    # o por amostra segue de pé
    assert out["out_var_time"].startswith("<img")
