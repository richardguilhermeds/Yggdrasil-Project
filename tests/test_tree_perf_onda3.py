"""Onda 3 de desempenho da TreeSegmenter: os caminhos novos têm de dar o MESMO
resultado que os antigos (velho × novo, igualdade exata).

- 3.1 ``_concentration_bins`` aditivo (autofit com restrição de concentração):
  contagem por bin fino feita uma vez (searchsorted/bincount no numérico com
  cortes monotônicos e sem NaN; value_counts do texto no categórico) e laço de
  fusão sobre contagens inteiras, no lugar de refazer K máscaras por iteração.

A versão "velha" de ``_concentration_bins`` está copiada abaixo, literal. Para
caber no tempo de teste com K grande, o ``_mask_in`` usado por ela pode ser
memoizado por bin (é função pura de frame/feature/bin, então o resultado não
muda). Com K=30000 a cópia literal é inviável (O(K²·n)); ali a referência é o
mesmo laço antigo reescrito sobre um vetor de contagens com ``np.argmin`` e
``np.delete`` (contagens por ``np.unique``, independentes do value_counts).
"""
from __future__ import annotations

import builtins
import json
import time
import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.tree import TreeSegmenter
from yggdrasil.credit_risk.tree import segmenter as segmod


# ----------------------------------------------------------------------
# Versão antiga (cópia literal do código antes da onda 3.1)
# ----------------------------------------------------------------------
def _concentration_bins_velho(self, sub, feature, min_leaf_repr, max_bin_repr, dtype=None):
    N = len(self.df)
    if not N:
        return None, "num"
    kind = self._detect_kind(sub, feature, dtype)
    na_present = bool(sub[feature].isna().any())

    if kind == "num":
        x = sub[feature].dropna().to_numpy(dtype="float64")
        if x.size == 0 or np.unique(x).size < 2:
            return None, kind
        qs = np.quantile(x, np.linspace(0.0, 1.0, 25)[1:-1])   # ~24 faixas finas
        cuts = sorted({round(float(c), 10) for c in qs})
        edges = [-np.inf, *cuts, np.inf]
        core = [{"kind": "num", "lo": edges[i], "hi": edges[i + 1]}
                for i in range(len(edges) - 1)]
    else:
        # uma categoria por grupo, ordenadas por alvo em DES (contiguidade de risco)
        ref = self._fit_frame(sub, 0.0).dropna(subset=[feature, self.target])
        means = (ref.groupby(ref[feature].astype(str))[self.target].mean()
                 .sort_values())
        present = [str(c) for c in pd.unique(sub[feature].dropna().astype(str))]
        ordered = [c for c in means.index if c in present]
        ordered += [c for c in present if c not in ordered]
        if len(ordered) < 2:
            return None, kind
        core = [{"kind": "cat", "cats": [c]} for c in ordered]

    def grepr(b):
        return int(self._mask_in(sub, feature, b).sum()) / N

    na = [{"kind": "na"}] if na_present else []
    if na and min_leaf_repr is not None and grepr(na[0]) < min_leaf_repr - 1e-9:
        return None, kind

    while len(core) > 1 and min_leaf_repr is not None:
        reprs = [grepr(b) for b in core]
        i = int(np.argmin(reprs))
        if reprs[i] >= min_leaf_repr - 1e-9:
            break
        cands = [k for k in (i - 1, i + 1) if 0 <= k < len(core)]
        ok = ([k for k in cands if reprs[i] + reprs[k] <= max_bin_repr + 1e-9]
              if max_bin_repr is not None else cands)
        j = min(ok or cands, key=lambda k: reprs[k])
        a, b2 = sorted((i, j))
        if kind == "num":
            core[a] = {"kind": "num", "lo": core[a]["lo"], "hi": core[b2]["hi"]}
        else:
            core[a] = {"kind": "cat", "cats": core[a]["cats"] + core[b2]["cats"]}
        del core[b2]

    if len(core) < 2:
        return None, kind
    out = core + na
    if min_leaf_repr is not None and any(grepr(b) < min_leaf_repr - 1e-9 for b in out):
        return None, kind
    if max_bin_repr is not None and any(grepr(b) > max_bin_repr + 1e-9 for b in out):
        return None, kind
    return out, kind


class _Contagem:
    """Máscara já somada: o laço antigo só usa `_mask_in(...).sum()`."""
    __slots__ = ("n",)

    def __init__(self, mask):
        self.n = int(mask.sum())

    def sum(self):
        return self.n


def _memo_mask_in(seg):
    """_mask_in memoizado por (frame, feature, bin) — puro, mesmo resultado (o
    laço antigo só consome a soma da máscara, guardada já somada). No bin
    categórico a conversão a texto da coluna também é guardada (a mesma
    expressão `frame[feature].astype(str).isin(cats)` do _mask_in)."""
    orig = TreeSegmenter._mask_in.__get__(seg)
    cache: dict = {}
    textos: dict = {}

    def chave(b):
        if b["kind"] == "na":
            return ("na",)
        if b["kind"] == "num":
            return ("num", b["lo"], b["hi"])
        return ("cat", tuple(b["cats"]))

    def f(frame, feature, b):
        k = (id(frame), feature, chave(b))
        if k not in cache:
            if b["kind"] == "cat":
                kt = (id(frame), feature)
                if kt not in textos:
                    textos[kt] = frame[feature].astype(str)
                cache[k] = _Contagem(textos[kt].isin(b["cats"]))
            else:
                cache[k] = _Contagem(orig(frame, feature, b))
        return cache[k]
    return f


def _velho(seg, sub, feat, mn, mx, dtype=None, memo=None):
    if memo is not None:
        seg._mask_in = memo
    try:
        return _concentration_bins_velho(seg, sub, feat, mn, mx, dtype)
    finally:
        seg.__dict__.pop("_mask_in", None)


def _novo(seg, sub, feat, mn, mx, dtype=None):
    return seg._concentration_bins(sub, feat, mn, mx, dtype)


def _igual(a, b):
    """Igualdade exata de (bins, kind): mesma ordem, mesmas chaves, mesmos
    valores e tipos (float de borda, lista de categorias)."""
    (ba, ka), (bb, kb) = a, b
    assert ka == kb
    if ba is None or bb is None:
        assert ba is None and bb is None, (ba, bb)
        return
    assert len(ba) == len(bb)
    for x, y in zip(ba, bb):
        assert list(x) == list(y)
        for k in x:
            assert type(x[k]) is type(y[k]), (k, x[k], y[k])
            if isinstance(x[k], float) and np.isnan(x[k]):
                assert np.isnan(y[k])
            else:
                assert x[k] == y[k], (k, x[k], y[k])


MINS = [0.005, 0.03]
MAXS = [None, 0.3, 0.5]


# ----------------------------------------------------------------------
# Base com os casos de borda
# ----------------------------------------------------------------------
def _base(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    amostra = rng.choice(["DES", "OOT"], n, p=[0.7, 0.3]).astype(object)
    x = np.round(rng.normal(size=n), 1)                 # empates
    x[rng.random(n) < 0.08] = np.nan
    x[:4] = [np.inf, -np.inf, np.inf, -np.inf]
    # 30×-inf + 70 finitos + 20×+inf (cortes NaN do np.quantile), repetido
    bloco = np.r_[np.full(30, -np.inf), np.arange(70.0), np.full(20, np.inf)]
    x_infrep = np.resize(bloco, n)
    rng.shuffle(x_infrep)
    x_infrep_nan = x_infrep.copy()
    x_infrep_nan[rng.random(n) < 0.05] = np.nan
    cat = rng.choice(["A", "B", "C", "D", "E", "F"], n).astype(object)
    cat_none = cat.copy()
    cat_none[rng.random(n) < 0.04] = None
    lit = rng.choice(["nan", "None", "A", "B", "C"], n).astype(object)
    lit[rng.random(n) < 0.05] = None
    lit[rng.random(n) < 0.05] = np.nan
    dec = rng.choice(np.array([Decimal("1.5"), Decimal("1.50"), Decimal("2"),
                               Decimal("2.0"), Decimal("3.25"), Decimal("-1")],
                              dtype=object), n)
    i64 = pd.array(rng.integers(0, 40, n), dtype="Int64")
    i64[rng.random(n) < 0.05] = pd.NA
    num = pd.array(rng.normal(size=n), dtype="Float64")
    den = pd.array(np.where(rng.random(n) < 0.05, 0.0, 1.0), dtype="Float64")
    num[rng.random(n) < 0.05] = 0.0
    f64nan = num / den                                   # 0/0 → NaN (não NA)
    y = (rng.random(n) < 0.1 + 0.2 * (np.nan_to_num(x, posinf=1, neginf=-1) > 0)
         ).astype(float)
    return pd.DataFrame({
        "x": x,
        "x_infrep": x_infrep,
        "x_infrep_nan": x_infrep_nan,
        "x_int": rng.integers(0, 1000, n),
        "x_big": rng.integers(-2**62, 2**62, n),
        "x_f32": rng.normal(size=n).astype("float32"),
        "x_i64": i64,
        "x_f64nan": f64nan,
        "flag": (rng.random(n) < 0.2).astype(int),
        "cat": cat,
        "cat_none": cat_none,
        "cat_lit": lit,
        "cat_dec": dec,
        "amostra": amostra,
        "target": y,
    })


FEATS = ["x", "x_int", "x_big", "x_f32", "x_i64", "x_f64nan", "flag",
         "cat", "cat_none", "cat_lit", "cat_dec"]
# ±inf repetidos → cortes NaN. Ali o próprio código ANTIGO não é determinístico:
# `{round(float(c), 10) for c in qs}` guarda cada NaN como objeto distinto, com
# hash por id, e a ordem em que o sorted os deixa varia entre chamadas. A
# comparação velho × novo desses casos roda com um `sorted` determinístico
# injetado nos dois (fixture `sorted_det`): mesmos cortes → mesmos bins.
FEATS_INF = ["x_infrep", "x_infrep_nan"]


def _sorted_det(it, *a, **kw):
    lst = list(it)
    tem_nan = any(isinstance(v, float) and v != v for v in lst)
    if a or kw or not tem_nan:
        return builtins.sorted(lst, *a, **kw)
    nans = [v for v in lst if v != v]
    resto = builtins.sorted(v for v in lst if v == v)
    passo = max(1, len(resto) // (len(nans) + 1))
    out = []
    for i, v in enumerate(resto):
        out.append(v)
        if nans and (i + 1) % passo == 0:
            out.append(nans.pop())                   # NaN fora de ordem, posição fixa
    return out + nans


@pytest.fixture
def sorted_det(monkeypatch):
    monkeypatch.setattr(segmod, "sorted", _sorted_det, raising=False)
    monkeypatch.setitem(_concentration_bins_velho.__globals__, "sorted", _sorted_det)


def _seg(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TreeSegmenter(df, target="target", sample_col="amostra", verbose=False)


@pytest.fixture(scope="module")
def seg_base():
    return _seg(_base())


def _subs(seg):
    """Raiz e uma 'folha' (metade das linhas): N é sempre a base inteira."""
    df = seg.df
    folha = df.loc[df["x_int"].to_numpy() < 500]
    return [("raiz", df), ("folha", folha)]


@pytest.mark.parametrize("mx", MAXS)
@pytest.mark.parametrize("mn", MINS)
def test_concentration_bins_velho_novo_bordas(seg_base, mn, mx):
    for nome, sub in _subs(seg_base):
        for feat in FEATS:
            _igual(_velho(seg_base, sub, feat, mn, mx), _novo(seg_base, sub, feat, mn, mx))


@pytest.mark.parametrize("mx", MAXS)
@pytest.mark.parametrize("mn", [None, *MINS])
def test_concentration_bins_velho_novo_inf_repetido(seg_base, sorted_det, mn, mx):
    for _, sub in _subs(seg_base):
        for feat in FEATS_INF:
            _igual(_velho(seg_base, sub, feat, mn, mx), _novo(seg_base, sub, feat, mn, mx))


def test_concentration_bins_sem_minimo_e_dtype_forcado(seg_base):
    for _, sub in _subs(seg_base):
        for feat in FEATS:
            for mx in MAXS:
                _igual(_velho(seg_base, sub, feat, None, mx),
                       _novo(seg_base, sub, feat, None, mx))
        memo = _memo_mask_in(seg_base)
        for feat, dt in [("flag", "cat"), ("x_int", "cat"), ("x_i64", "cat")]:
            _igual(_velho(seg_base, sub, feat, 0.03, 0.5, dt, memo=memo),
                   _novo(seg_base, sub, feat, 0.03, 0.5, dt))


def test_concentration_bins_inf_repetido_cai_no_caminho_antigo(seg_base, monkeypatch):
    """O caso do revisor: ±inf repetidos dão corte NaN — o caminho aditivo não
    pode ser usado (o novo tem de chamar _mask_in, como o antigo)."""
    sub = seg_base.df
    x = sub["x_infrep"].dropna().to_numpy(dtype="float64")
    qs = np.quantile(x, np.linspace(0.0, 1.0, 25)[1:-1])
    assert np.isnan(qs).any()                       # a premissa do caso
    chamadas = []
    orig = TreeSegmenter._mask_in

    def espia(self, frame, feature, b):
        chamadas.append(b["kind"])
        return orig(self, frame, feature, b)
    monkeypatch.setattr(TreeSegmenter, "_mask_in", espia)
    seg_base._concentration_bins(sub, "x_infrep", 0.03, None)
    assert "num" in chamadas


def test_concentration_bins_float32_cai_no_caminho_antigo(seg_base, monkeypatch):
    """float32 compara com o limite em float32 (NEP 50): sem contagem aditiva."""
    chamadas = []
    orig = TreeSegmenter._mask_in

    def espia(self, frame, feature, b):
        chamadas.append(b["kind"])
        return orig(self, frame, feature, b)
    monkeypatch.setattr(TreeSegmenter, "_mask_in", espia)
    seg_base._concentration_bins(seg_base.df, "x_f32", 0.03, None)
    assert "num" in chamadas


@pytest.mark.parametrize("feat", ["x", "x_int", "x_big", "x_i64", "flag",
                                  "cat", "cat_none", "cat_lit", "cat_dec"])
def test_concentration_bins_caminho_aditivo_nao_varre_mascaras(seg_base, monkeypatch, feat):
    """Nos casos 'normais' o novo não chama _mask_in nenhuma vez."""
    def proibido(self, frame, feature, b):
        raise AssertionError("caminho aditivo chamou _mask_in")
    monkeypatch.setattr(TreeSegmenter, "_mask_in", proibido)
    for _, sub in _subs(seg_base):
        seg_base._concentration_bins(sub, feat, 0.03, 0.5)


# ----------------------------------------------------------------------
# Cardinalidade K ∈ {3, 50, 1000}: cópia literal (com _mask_in memoizado)
# ----------------------------------------------------------------------
def _base_k(k, n, seed=1, com_none=False, decimal_=False):
    rng = np.random.default_rng(seed)
    lvl = rng.integers(0, k, n)
    if decimal_:
        niveis = np.array([Decimal(i) / Decimal(100) for i in range(k)], dtype=object)
    else:
        niveis = np.array([f"L{i:05d}" for i in range(k)], dtype=object)
    cat = niveis[lvl]
    if com_none:
        cat = cat.copy()
        cat[rng.random(n) < 0.04] = None
    risco = 0.05 + 0.3 * (np.arange(k) % 7) / 6.0
    y = (rng.random(n) < risco[lvl]).astype(float)
    amostra = rng.choice(["DES", "OOT"], n, p=[0.7, 0.3]).astype(object)
    x = rng.normal(size=n)
    return pd.DataFrame({"c": cat, "x": x, "amostra": amostra, "target": y})


@pytest.mark.parametrize("k,n", [(3, 2000), (50, 3000), (1000, 6000)])
@pytest.mark.parametrize("com_none", [False, True])
def test_concentration_bins_velho_novo_cardinalidade(k, n, com_none):
    seg = _seg(_base_k(k, n, com_none=com_none))
    memo = _memo_mask_in(seg)
    folha = seg.df.loc[seg.df["x"].to_numpy() < 0.3]
    for sub in (seg.df, folha):
        for mn in MINS:
            for mx in MAXS:
                _igual(_velho(seg, sub, "c", mn, mx, memo=memo),
                       _novo(seg, sub, "c", mn, mx))


def test_concentration_bins_velho_novo_decimal_sem_nan():
    seg = _seg(_base_k(400, 5000, decimal_=True))
    memo = _memo_mask_in(seg)
    assert not seg.df["c"].isna().any()
    for mn in MINS:
        for mx in MAXS:
            _igual(_velho(seg, seg.df, "c", mn, mx, memo=memo),
                   _novo(seg, seg.df, "c", mn, mx))


# ----------------------------------------------------------------------
# K = 30000: referência = laço antigo sobre vetor de contagens (np.argmin)
# ----------------------------------------------------------------------
def _ref_array_cat(seg, sub, feat, mn, mx):
    N = len(seg.df)
    ref = seg._fit_frame(sub, 0.0).dropna(subset=[feat, seg.target])
    means = (ref.groupby(ref[feat].astype(str))[seg.target].mean().sort_values())
    present = [str(c) for c in pd.unique(sub[feat].dropna().astype(str))]
    ps = set(present)
    ordered = [c for c in means.index if c in ps]
    vis = set(ordered)
    ordered += [c for c in present if c not in vis]
    txt = sub[feat].astype(str).to_numpy(dtype=object)
    u, c = np.unique(txt, return_counts=True)
    cont = dict(zip(u.tolist(), c.tolist()))
    cnt = np.array([cont.get(o, 0) for o in ordered], dtype=np.int64)
    grupos = [[o] for o in ordered]
    while len(grupos) > 1 and mn is not None:
        reprs = cnt / N
        i = int(np.argmin(reprs))
        if reprs[i] >= mn - 1e-9:
            break
        cands = [k for k in (i - 1, i + 1) if 0 <= k < len(grupos)]
        ok = ([k for k in cands if reprs[i] + reprs[k] <= mx + 1e-9]
              if mx is not None else cands)
        j = min(ok or cands, key=lambda k: reprs[k])
        a, b2 = sorted((i, j))
        grupos[a] = grupos[a] + grupos[b2]
        cnt[a] += cnt[b2]
        del grupos[b2]
        cnt = np.delete(cnt, b2)
    if len(grupos) < 2:
        return None, "cat"
    reps = cnt / N
    if mn is not None and (reps < mn - 1e-9).any():
        return None, "cat"
    if mx is not None and (reps > mx + 1e-9).any():
        return None, "cat"
    return [{"kind": "cat", "cats": g} for g in grupos], "cat"


@pytest.fixture(scope="module")
def seg_30k():
    return _seg(_base_k(30_000, 90_000, seed=3))


@pytest.mark.parametrize("mx", MAXS)
@pytest.mark.parametrize("mn", MINS)
def test_concentration_bins_k30000_contra_referencia(seg_30k, mn, mx):
    sub = seg_30k.df
    assert sub["c"].nunique() > 25_000
    _igual(_ref_array_cat(seg_30k, sub, "c", mn, mx), _novo(seg_30k, sub, "c", mn, mx))


def test_concentration_bins_referencia_bate_com_velho_k1000():
    """A referência de vetor (usada no K=30000) é ela mesma conferida contra a
    cópia literal num K onde a literal ainda roda."""
    seg = _seg(_base_k(1000, 6000, seed=5))
    memo = _memo_mask_in(seg)
    for mn in MINS:
        for mx in MAXS:
            _igual(_velho(seg, seg.df, "c", mn, mx, memo=memo),
                   _ref_array_cat(seg, seg.df, "c", mn, mx))


def test_concentration_bins_k30000_tempo(seg_30k):
    sub = seg_30k.df
    t0 = time.perf_counter()
    bins, kind = seg_30k._concentration_bins(sub, "c", 0.03, None)
    dt = time.perf_counter() - t0
    assert kind == "cat" and bins is not None
    assert dt < 2.0, f"_concentration_bins K=30000 levou {dt:.2f}s"


# ----------------------------------------------------------------------
# fit_auto com restrição: árvore inteira igual (velho × novo)
# ----------------------------------------------------------------------
def _base_fit(n=6000, seed=11):
    rng = np.random.default_rng(seed)
    k = 150
    lvl = rng.integers(0, k, n)
    cat = np.array([f"G{i:03d}" for i in range(k)], dtype=object)[lvl]
    dec = np.array([Decimal(i) / Decimal(10) for i in range(150)],
                   dtype=object)[rng.integers(0, 150, n)]
    x = rng.normal(size=n)
    x[rng.random(n) < 0.02] = np.nan
    x[:6] = [np.inf, -np.inf] * 3
    z = rng.exponential(size=n)
    risco = (0.05 + 0.25 * (np.arange(k) % 5) / 4.0)[lvl] * (1 + 0.5 * (z > 1))
    y = (rng.random(n) < np.clip(risco, 0, 1)).astype(float)
    amostra = rng.choice(["DES", "OOT"], n, p=[0.7, 0.3]).astype(object)
    return pd.DataFrame({"cat": cat, "dec": dec, "x": x, "z": z,
                         "amostra": amostra, "target": y})


def _arvore(seg):
    return json.dumps(seg.to_dict(), default=str, sort_keys=True)


@pytest.mark.parametrize("mn,mx", [(0.03, None), (0.02, 0.5), (0.05, 0.3)])
def test_fit_auto_com_restricao_velho_novo(monkeypatch, mn, mx):
    df = _base_fit()
    novo = _seg(df).fit_auto(max_depth=2, min_leaf_repr=mn, max_bin_repr=mx,
                             verbose=False)
    monkeypatch.setattr(TreeSegmenter, "_concentration_bins", _concentration_bins_velho)
    velho = _seg(df).fit_auto(max_depth=2, min_leaf_repr=mn, max_bin_repr=mx,
                              verbose=False)
    assert _arvore(novo) == _arvore(velho)
    assert sum(s["is_leaf"] for s in novo.segments.values()) > 1
