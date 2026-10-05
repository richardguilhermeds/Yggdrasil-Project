"""Onda 1 de desempenho da TreeSegmenter: os caminhos novos têm de dar o MESMO
resultado que os antigos (velho × novo, igualdade exata).

- 1.1 "degenerado" por min/max e comparação com ``xs[0]`` em vez de ``np.unique``
  (``_resolve_bins`` num/cat e ``_optbin_numeric_bins``);
- 1.2 ``_sample_dominante_por_safra`` com frame de 2 colunas;
- 1.3 ``leaves()`` compondo PSI/teste de adjacência sobre a tabela base memoizada;
- 1.4 ``ref_mask`` morto removido de ``plot_variable_optbin_cumshare_timeseries``.

A versão "velha" de ``_resolve_bins``/``_optbin_numeric_bins`` é reconstruída a
partir do código-fonte ATUAL trocando só as expressões de degenerado pelas
antigas (``np.unique``): se o código mudar e a troca não casar, o teste acusa.
"""
from __future__ import annotations

import inspect
import sys
import textwrap
import tracemalloc
import types
import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.tree import TreeSegmenter
from yggdrasil.credit_risk.tree import segmenter as segmod

TASKS = ["classification", "regression"]


# ----------------------------------------------------------------------
# Fixture: base sintética com os tipos "chatos" do plano de equivalência
# ----------------------------------------------------------------------
def _base(task, n=3000, seed=0, dup_index=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    x[rng.random(n) < 0.08] = np.nan
    x[:5] = [np.inf, -np.inf, -0.0, 0.0, np.inf]
    amostra = rng.choice(["DES", "OOT", "ESTAB"], n, p=[0.6, 0.25, 0.15]).astype(object)
    amostra[rng.random(n) < 0.03] = None
    risco = 0.15 + 0.25 * (np.nan_to_num(x, nan=0.0, posinf=2, neginf=-2) > 0.3)
    if task == "classification":
        y = (rng.random(n) < risco).astype(float)
    else:
        y = np.clip(risco + rng.normal(0, 0.1, n), 0, 1)
    y[(amostra == "OOT") & (rng.random(n) < 0.05)] = np.nan     # alvo NaN só no OOT
    cat = rng.choice(["A", "B", "C", "nan", "None"], n).astype(object)
    cat[rng.random(n) < 0.05] = None
    mix = rng.choice(np.array([1, 1.0, True, "1", "2"], dtype=object), n)
    dec = rng.choice(np.array([Decimal("1.5"), Decimal("1.50"), Decimal("2"),
                               Decimal("NaN"), None], dtype=object), n)
    vazio = rng.choice(["", "a"], n).astype(object)
    i64 = pd.array(rng.integers(0, 5, n), dtype="Int64")
    i64[rng.random(n) < 0.1] = pd.NA
    f64 = pd.array(rng.normal(size=n), dtype="Float64")
    f64[rng.random(n) < 0.1] = pd.NA
    bnull = pd.array(rng.random(n) < 0.4, dtype="boolean")
    bnull[rng.random(n) < 0.1] = pd.NA
    datas = pd.to_datetime("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 365, n), unit="D")
    dt_ref = np.array([d.date() for d in datas], dtype=object)
    dt_ref[rng.random(n) < 0.02] = None
    df = pd.DataFrame({
        "x": x,
        "x_const": np.where(rng.random(n) < 0.1, np.nan, 3.0),
        "x_zero": np.where(rng.random(n) < 0.5, 0.0, -0.0),      # -0.0 == 0.0
        "x_inf": np.where(rng.random(n) < 0.5, np.inf, -np.inf),  # 2 distintos
        "x_infc": np.full(n, np.inf),                              # 1 distinto
        "x_f32": rng.normal(size=n).astype("float32"),
        "x_i64": i64,
        "x_f64": f64,
        "b_null": bnull,
        "b_nat": rng.random(n) < 0.3,
        "flag": (rng.random(n) < 0.2).astype(int),
        "cat": cat,
        "cat_mix": mix,
        "cat_dec": dec,
        "cat_const": np.where(rng.random(n) < 0.1, None, "X").astype(object),
        "cat_vazio": vazio,
        "cat_vazio_c": np.full(n, "", dtype=object),
        "cat_hi": np.array([f"k{i}" for i in rng.integers(0, 600, n)], dtype=object),
        "cat_str": pd.array(rng.choice(["p", "q", None], n), dtype="string"),
        "cat_cat": pd.Categorical(rng.choice(["z", "y", "w"], n),
                                  categories=["z", "y", "w"]),
        "cat_allna": np.full(n, None, dtype=object),
        "amostra": amostra,
        "dt_ref": dt_ref,
        "target": y,
    })
    if dup_index:
        df.index = np.r_[np.arange(n // 2), np.arange(n - n // 2)]
    return df


FEATS = ["x", "x_const", "x_zero", "x_inf", "x_infc", "x_f32", "x_i64", "x_f64",
         "b_null", "b_nat", "flag", "cat", "cat_mix", "cat_dec", "cat_const",
         "cat_vazio", "cat_vazio_c", "cat_hi", "cat_str", "cat_cat", "cat_allna"]


def _seg(task, df=None, **kw):
    df = _base(task) if df is None else df
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TreeSegmenter(df, target="target", task_type=task, sample_col="amostra",
                             date_col="dt_ref", verbose=False, **kw)


# ----------------------------------------------------------------------
# Versões "velhas" reconstruídas do fonte atual (só o degenerado volta ao unique)
# ----------------------------------------------------------------------
_TROCAS_RESOLVE = [
    ("or x_obs.min() == x_obs.max()", "or np.unique(x_obs).size < 2"),
    ("or (self._is_clf and yfit.min() == yfit.max()))",
     "or (self._is_clf and np.unique(yfit).size < 2))"),
    ("degenerado = (len(ys) < 4 or not (xs[1:] != xs[0]).any()",
     "degenerado = (len(ys) < 4 or np.unique(xs).size < 2"),
    ("or (self._is_clf and ys.min() == ys.max()))",
     "or (self._is_clf and np.unique(ys).size < 2))"),
]
_TROCAS_OPTBIN = [
    ("x_obs.size == 0 or x_obs.min() == x_obs.max():",
     "x_obs.size == 0 or np.unique(x_obs).size < 2:"),
]


def _metodo_velho(nome, trocas):
    src = textwrap.dedent(inspect.getsource(getattr(TreeSegmenter, nome)))
    for novo, velho in trocas:
        assert src.count(novo) == 1, f"trecho não encontrado em {nome}: {novo!r}"
        src = src.replace(novo, velho)
    ns: dict = {}
    exec(compile(src, f"<{nome}_velho>", "exec"), dict(vars(segmod)), ns)
    return ns[nome]


_resolve_velho = _metodo_velho("_resolve_bins", _TROCAS_RESOLVE)
_optbin_velho = _metodo_velho("_optbin_numeric_bins", _TROCAS_OPTBIN)


def _eq_bins(a, b):
    """Igualdade exata de (bins, modo, kind) — NaN não aparece nos bins."""
    assert a[1:] == b[1:]
    assert a[0] == b[0]


# ----------------------------------------------------------------------
# 1.1 — propriedade: flag degenerado velho == novo
# ----------------------------------------------------------------------
_ARR_NUM = [
    np.array([1.0]), np.array([2.0, 2.0, 2.0, 2.0]), np.array([0.0, -0.0, 0.0, -0.0]),
    np.array([np.inf, np.inf]), np.array([-np.inf, np.inf]), np.array([np.inf, 1.0]),
    np.array([-np.inf, -np.inf, -np.inf, -np.inf]), np.array([1.0, 1.0, 1.0 + 1e-15, 1.0]),
    np.array([5e-324, 0.0]), np.arange(10, dtype=float),
]
_ARR_STR = [
    np.array(["a"], dtype=object), np.array(["", "", "", ""], dtype=object),
    np.array(["", "a", "", ""], dtype=object), np.array(["nan"] * 4, dtype=object),
    np.array(["nan", "None", "nan", "nan"], dtype=object),
    pd.Series([Decimal("1.0"), Decimal("1.00")] * 2).astype(str).to_numpy(),
    pd.Series([Decimal("1.0")] * 4).astype(str).to_numpy(),
    pd.Series([1, 1.0, True, "1"], dtype=object).astype(str).to_numpy(),
    pd.Series(pd.array(["p", "p", "p", "p"], dtype="string")).astype(str).to_numpy(),
    pd.Series(pd.array(["p", "q", "p", "p"], dtype="string")).astype(str).to_numpy(),
    np.array(["ä", "a", "a", "a"], dtype=object),
]


@pytest.mark.parametrize("a", _ARR_NUM)
def test_degenerado_num_min_max_igual_unique(a):
    assert (np.unique(a).size < 2) == bool(a.min() == a.max())


@pytest.mark.parametrize("xs", _ARR_STR)
def test_degenerado_cat_comparacao_igual_unique(xs):
    assert (np.unique(xs).size < 2) == bool(not (xs[1:] != xs[0]).any())


# ----------------------------------------------------------------------
# 1.1 — _resolve_bins velho × novo em todas as variáveis da fixture
# ----------------------------------------------------------------------
def _recortes(seg):
    """Raiz, um ramo com y constante (só zeros), um ramo minúsculo (<4) e um
    ramo médio — cobrem os ramos do curto-circuito."""
    df = seg.df
    alvo = df["target"]
    return {
        "raiz": pd.Series(True, index=df.index),
        "y_const": (alvo == alvo.min()).fillna(False),
        "tres": pd.Series(np.arange(len(df)) < 3, index=df.index),
        "metade": pd.Series(np.arange(len(df)) % 2 == 0, index=df.index),
    }


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("crit", ["optbin", "outro"])
def test_resolve_bins_velho_novo(task, crit):
    seg = _seg(task)
    criterion = crit if crit == "optbin" else ("gini" if task == "classification"
                                               else "variance")
    velho = types.MethodType(_resolve_velho, seg)
    for nome, m in _recortes(seg).items():
        for feat in FEATS:
            sub = seg._sub(m, feat)
            for msz in (0.05, 0.2):
                kw = dict(splits=None, dtype=None, max_n_bins=4, min_bin_size=msz,
                          criterion=criterion)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    a = velho(sub, feat, **kw)
                    b = seg._resolve_bins(sub, feat, **kw)
                _eq_bins(a, b)


@pytest.mark.parametrize("task", TASKS)
def test_optbin_numeric_bins_velho_novo(task):
    seg = _seg(task)
    velho = types.MethodType(_optbin_velho, seg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.grow("x", max_n_bins=3)
        for sid in [None, *[s for s, v in seg.segments.items() if v["is_leaf"]]]:
            for feat in ["x", "x_const", "x_zero", "x_inf", "x_infc", "x_f32", "x_i64",
                         "x_f64", "flag"]:
                for sample in (None, "OOT"):
                    assert velho(feat, sid=sid, sample=sample) == \
                        seg._optbin_numeric_bins(feat, sid=sid, sample=sample)


@pytest.mark.parametrize("task", TASKS)
def test_resolve_bins_sem_np_unique(task, monkeypatch):
    """Nenhuma chamada DIRETA de np.unique dentro de _resolve_bins (num/cat) nem
    de _optbin_numeric_bins — em object (ordenação de strings Python) falha na hora.
    O np.unique de _best_categorical_split (S:186) segue intocado e é permitido."""
    seg = _seg(task)
    orig = np.unique
    chamadas = []

    def espiao(ar, *a, **k):
        f = sys._getframe(1).f_code.co_name
        if f in ("_resolve_bins", "_optbin_numeric_bins"):
            if np.asarray(ar).dtype == object:
                raise AssertionError(f"np.unique em array object dentro de {f}")
            chamadas.append(f)
        return orig(ar, *a, **k)

    monkeypatch.setattr(np, "unique", espiao)
    raiz = pd.Series(True, index=seg.df.index)
    crit = "gini" if task == "classification" else "variance"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for feat in FEATS:
            seg._resolve_bins(seg._sub(raiz, feat), feat, None, None, 4, 0.05,
                              criterion=crit)
        seg._optbin_numeric_bins("x")
    assert chamadas == []


@pytest.mark.parametrize("task", TASKS)
def test_fit_auto_to_dict_igual_ao_velho(task, monkeypatch):
    """Árvore inteira (fit_auto) com o _resolve_bins velho × novo: to_dict igual."""
    def arvore(resolve):
        seg = _seg(task, _base(task, n=2500, seed=3))
        if resolve is not None:
            monkeypatch.setattr(seg, "_resolve_bins", types.MethodType(resolve, seg))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seg.fit_auto(max_depth=2, features=["x", "x_i64", "cat", "cat_mix",
                                                  "cat_dec", "b_null", "x_zero"],
                         verbose=False)
        return seg.to_dict()

    assert arvore(_resolve_velho) == arvore(None)


# ----------------------------------------------------------------------
# 1.2 — _sample_dominante_por_safra com frame de 2 colunas
# ----------------------------------------------------------------------
def _dominante_velho(seg, time_col=None):
    """Cópia literal da implementação antiga (assign + dropna na base inteira)."""
    time_col = time_col or seg.date_col
    if seg.sample_col is None or time_col is None or time_col not in seg.df.columns:
        return None
    saf = pd.to_datetime(seg.df[time_col], errors="coerce").dt.to_period("M").astype(str)
    return (seg.df.assign(_saf=saf)
            .dropna(subset=[seg.sample_col])
            .groupby("_saf")[seg.sample_col]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None))


def _df_dominante(dtype_amostra="object", dup_index=False, com_saf=False):
    # jan: empate DES/OOT com a "maior" (OOT) aparecendo primeiro; fev: só amostra
    # NaN (ausente); mar: OOT domina; data None → 'NaT'; outra coluna de data.
    linhas = (
        [("2023-01-10", "OOT")] * 3 + [("2023-01-20", "DES")] * 3
        + [("2023-02-05", None)] * 4
        + [("2023-03-01", "OOT")] * 5 + [("2023-03-02", "DES")] * 2
        + [(None, "DES")] * 3 + [(None, "OOT")] * 1
        + [("2023-04-01", "ESTAB")] * 2 + [("2023-04-02", "DES")] * 2
        + [("2023-05-15", "DES")] * 6
    )
    n = len(linhas)
    datas = [None if d is None else pd.Timestamp(d).date() for d, _ in linhas]
    amostras = [a for _, a in linhas]
    if dtype_amostra == "category":
        am = pd.Categorical(amostras, categories=["OOT", "ESTAB", "DES"])
    elif dtype_amostra == "string":
        am = pd.array(amostras, dtype="string")
    elif dtype_amostra == "num":
        mapa = {"DES": 1.0, "OOT": 2.0, "ESTAB": 3.0}
        am = np.array([np.nan if a is None else mapa[a] for a in amostras])
    else:
        am = np.array(amostras, dtype=object)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "dt_ref": np.array(datas, dtype=object),
        "dt_outra": pd.to_datetime("2024-06-30") - pd.to_timedelta(
            rng.integers(0, 120, n), unit="D"),
        "amostra": am,
        "x": rng.normal(size=n),
        "target": (rng.random(n) < 0.3).astype(float),
    })
    if com_saf:
        df["_saf"] = "lixo"
    if dup_index:
        df.index = np.r_[np.arange(n // 2), np.arange(n - n // 2)]
    return df


@pytest.mark.parametrize("dtype_amostra", ["object", "category", "string", "num"])
@pytest.mark.parametrize("dup_index", [False, True])
@pytest.mark.parametrize("com_saf", [False, True])
def test_sample_dominante_velho_novo(dtype_amostra, dup_index, com_saf):
    df = _df_dominante(dtype_amostra, dup_index, com_saf)
    ref = 1.0 if dtype_amostra == "num" else "DES"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = TreeSegmenter(df, target="target", sample_col="amostra", ref_sample=ref,
                            date_col="dt_ref", verbose=False, min_leaf_rows=1)
    for tc in (None, "dt_outra"):
        velho = _dominante_velho(seg, tc)
        novo = seg._sample_dominante_por_safra(tc)
        pd.testing.assert_series_equal(velho, novo, check_names=True, check_exact=True)
        safras = list(velho.index) + ["2099-01", "NaT"]
        seg2 = TreeSegmenter.__new__(TreeSegmenter)
        seg2.__dict__.update(seg.__dict__)
        seg2._samp_by_cache = {(tc or "dt_ref"): velho}
        assert seg._sample_seq(safras, tc) == seg2._sample_seq(safras, tc)
    novo = seg._sample_dominante_por_safra()
    assert "2023-02" not in novo.index                   # safra só com amostra NaN
    assert "NaT" in novo.index                           # data None vira 'NaT'
    if dtype_amostra == "object":
        assert novo["2023-01"] == "DES"                  # empate → menor (mode)


def test_sample_dominante_pico_memoria():
    """O pico da 1ª chamada fica na ordem de poucas colunas, não de 2× a base."""
    rng = np.random.default_rng(0)
    n, k = 40_000, 80
    df = pd.DataFrame(rng.normal(size=(n, k)), columns=[f"v{i}" for i in range(k)])
    df["amostra"] = rng.choice(["DES", "OOT"], n).astype(object)
    df["dt_ref"] = pd.to_datetime("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 700, n), unit="D")
    df["target"] = (rng.random(n) < 0.2).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = TreeSegmenter(df, target="target", sample_col="amostra",
                            date_col="dt_ref", verbose=False)
    tam_base = int(seg.df.memory_usage(deep=False).sum())
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        seg._sample_dominante_por_safra()
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert pico < 0.35 * tam_base, (pico, tam_base)


# ----------------------------------------------------------------------
# 1.3 — leaves() compondo sobre a base memoizada
# ----------------------------------------------------------------------
COMBOS = [(asc, psi, tst, t) for asc in (True, False) for psi in (False, True)
          for tst in (False, True) for t in ("mannwhitney", "welch")]


def _seg_leaves(task, weight, sample, apelidos):
    df = _base(task, n=2500, seed=5)
    df["peso"] = np.random.default_rng(9).uniform(1, 100, len(df))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = TreeSegmenter(df, target="target", task_type=task,
                            sample_col="amostra" if sample else None,
                            weight_col="peso" if weight else None,
                            date_col="dt_ref", verbose=False, min_leaf_rows=5)
        seg.grow("x", max_n_bins=3)
        folhas = [s for s, v in seg.segments.items() if v["is_leaf"]]
        seg.grow("cat", max_n_bins=3, only_segments=folhas[:1])
    if apelidos:
        folhas = [s for s, v in seg.segments.items() if v["is_leaf"]]
        seg.set_leaf_name(folhas[0], "Prime")
        seg.set_leaf_name(folhas[-1], "Pior")
    return seg


def _checa_todas(seg, ordem):
    nomes = seg._leaf_names_validos()
    for asc, psi, tst, t in ordem:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            esperado = seg._compute_leaves(asc, psi, tst, t)
            obtido = seg.leaves(asc, psi, tst, t)
        if nomes:
            assert list(obtido.columns).count("apelido") == 1
            assert obtido.columns[1] == "apelido"
            obtido = obtido.drop(columns="apelido")
        else:
            assert "apelido" not in obtido.columns
        pd.testing.assert_frame_equal(esperado, obtido, check_exact=True,
                                      check_dtype=True, check_names=True)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("weight", [False, True])
@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("apelidos", [False, True])
def test_leaves_composto_igual_compute(task, weight, sample, apelidos):
    seg = _seg_leaves(task, weight, sample, apelidos)
    _checa_todas(seg, COMBOS)                       # base antes (combo 0 é a base)
    seg._agg_cache.clear()
    _checa_todas(seg, list(reversed(COMBOS)))       # composto antes da base
    # mutação (bump de versão): nada da versão anterior pode vazar
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        antes = seg.leaves(True, True, True)
        folhas = [s for s, v in seg.segments.items() if v["is_leaf"]]
        seg.grow("x_f32", max_n_bins=2, only_segments=folhas[-1:])
    _checa_todas(seg, COMBOS[::3] + COMBOS)
    pai = next(s for s, v in seg.segments.items()
               if not v["is_leaf"] and v["parent"] is not None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.collapse(pai, verbose=False)
    _checa_todas(seg, list(reversed(COMBOS)))
    assert len(antes)                               # (só p/ usar a tabela antiga)


def test_leaves_base_em_cache_nao_e_mutada():
    seg = _seg_leaves("classification", True, True, True)
    base = seg.leaves()
    ck = (("leaves", True, False, False, "mannwhitney"), seg._tree_version)
    cacheada = seg._agg_cache[ck].copy()
    seg.leaves(True, True, True, "welch")
    seg.leaves(True, False, True)
    pd.testing.assert_frame_equal(seg._agg_cache[ck], cacheada, check_exact=True)
    pd.testing.assert_frame_equal(seg.leaves(), base, check_exact=True)
    # 'test' sem with_test não cria outra entrada da base
    n = len(seg._agg_cache)
    seg.leaves(True, False, False, "welch")
    assert len(seg._agg_cache) == n


def test_leaves_composto_nao_refaz_compute(monkeypatch):
    """Com a base em cache, as variantes com PSI/teste não refazem _compute_leaves."""
    seg = _seg_leaves("classification", False, True, False)
    seg.leaves()
    chamadas = []
    orig = seg._compute_leaves
    monkeypatch.setattr(seg, "_compute_leaves",
                        lambda *a, **k: chamadas.append(a) or orig(*a, **k))
    for _asc, psi, tst, t in COMBOS:
        seg.leaves(True, psi, tst, t)
    assert chamadas == []


# ----------------------------------------------------------------------
# 1.4 — ref_mask morto: o gráfico segue igual (dados da figura)
# ----------------------------------------------------------------------
def _impressao_fig(fig):
    """Impressão digital dos dados desenhados (textos, polígonos, linhas)."""
    out = []
    for ax in fig.axes:
        out.append(("titulo", ax.get_title(), ax.get_xlabel(), ax.get_ylabel()))
        out.append(("textos", [t.get_text() for t in ax.texts]))
        out.append(("ticks", [t.get_text() for t in ax.get_xticklabels()]))
        for c in ax.collections:
            out.append(("col", [np.asarray(p.vertices).round(12).tolist()
                                for p in c.get_paths()]))
        for ln in ax.lines:
            out.append(("lin", np.asarray(ln.get_xydata(), dtype=float).round(12).tolist()))
        leg = ax.get_legend()
        if leg is not None:
            out.append(("leg", [t.get_text() for t in leg.get_texts()]))
    return out


@pytest.mark.parametrize("feat,sid", [("x", None), ("x_i64", None), ("cat", None),
                                      ("x", "folha")])
def test_cumshare_timeseries_sem_ref_mask(feat, sid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nome = "plot_variable_optbin_cumshare_timeseries"
    assert "ref_mask" not in inspect.getsource(getattr(TreeSegmenter, nome))
    velho = _metodo_velho(nome, [(
        "mask0 = self._leaf_mask(sid)\n",
        "mask0 = self._leaf_mask(sid)\n"
        "    ref_mask = (mask0 & (self.df[self.sample_col] == self.ref_sample)\n"
        "                if self.sample_col is not None else mask0)\n")])
    seg = _seg("classification")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if sid is not None:
            seg.grow("cat_cat", max_n_bins=2)
            sid = [s for s, v in seg.segments.items() if v["is_leaf"]][0]
        f_velho = types.MethodType(velho, seg)(feat, sid=sid)
        f_novo = seg.plot_variable_optbin_cumshare_timeseries(feat, sid=sid)
    try:
        assert _impressao_fig(f_velho) == _impressao_fig(f_novo)
    finally:
        plt.close(f_velho)
        plt.close(f_novo)
