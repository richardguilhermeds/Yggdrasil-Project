"""Ajustes do optbinning do IV da árvore em threads (``TreeSegmenter(n_jobs=...)``).

``variable_iv`` com ``n_jobs`` > 1 roda os ajustes do optbinning (CART + CP-SAT,
que soltam a GIL) num ``ThreadPoolExecutor`` criado por chamada, em lotes de
``2·n_jobs`` variáveis: preparar (thread principal) → resolver (threads, ``map``
ordenado, dentro de UM ``catch_warnings`` na thread principal) → finalizar
(thread principal, repetindo a lógica do ``_bins_cache``).

Tudo tem de dar o MESMO resultado do ajuste em série (igualdade exata):
- ``_resolve_bins`` (agora preparar → resolver → finalizar) contra a cópia
  LITERAL da versão antiga (abaixo);
- ``variable_iv`` com ``n_jobs=1`` × ``n_jobs=4``: DataFrame (``check_exact``) +
  ``attrs`` e o ``_bins_cache`` inteiro (chaves na mesma ordem, valores e a
  máscara guardada);
- ``warnings.filters`` restaurado; o aviso "optbinning falhou inesperadamente"
  continua saindo (mesma mensagem e origem) com ``n_jobs`` > 1;
- exceção por variável (kind '—', nb 0, IV NaN), backstop do cache no meio do
  lote, variável repetida, amostragem do ajuste e ``cancel_futures`` em erro.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from yggdrasil.credit_risk import _common  # noqa: E402
from yggdrasil.credit_risk.tree import TreeSegmenter  # noqa: E402
from yggdrasil.credit_risk.tree import segmenter as segmod  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from test_tree_perf_onda3_iv import (  # noqa: E402
    FEATS, _base, _folhas, _iguais, _internos, _seg, _variable_iv_velho)

TASKS = ["classification", "regression"]

# Ajustes independentes comparados com igualdade exata: CP-SAT com 1 worker
# (ver ``cpsat_deterministico`` no conftest) para o empate do ótimo não virar
# falha intermitente.
pytestmark = pytest.mark.usefixtures("cpsat_deterministico")

# nomes usados pela cópia literal abaixo (resolvidos no módulo do teste)
_fit_optbinning_splits = segmod._fit_optbinning_splits
_best_numeric_cut = segmod._best_numeric_cut
_best_categorical_split = segmod._best_categorical_split


# ----------------------------------------------------------------------
# Versão antiga (cópia literal de _resolve_bins antes da divisão em etapas)
# ----------------------------------------------------------------------
def _resolve_bins_velho(self, sub, feature, splits, dtype, max_n_bins, min_bin_size,
                  max_bin_size=None, relax_max=False, min_mean_diff=0.0,
                  criterion="optbin"):
    """Resolve os bins de um ramo. Devolve (bins, modo, kind).
    - num: bins = [{'kind':'num','lo','hi'}, ...]
    - cat: bins = [{'kind':'cat','cats':[...]}, ...]
    No modo ótimo (``criterion="optbin"``), o binning é ajustado só na amostra
    de referência (DES) — binário (``OptimalBinning``) p/ classificação,
    contínuo (``ContinuousOptimalBinning``) p/ regressão. Com outro
    ``criterion`` (gini/entropy/ks/iv/chi2 · variance/mae/ftest) faz um split
    BINÁRIO no melhor corte por esse critério. No modo manual: splits numérico
    = lista de cortes; splits categórico = lista de grupos.
    min_mean_diff: diferença mínima do alvo médio (taxa de default em
    classificação) exigida entre bins consecutivas no binning ótimo; 0 = sem
    restrição.
    """
    kind = self._detect_kind(sub, feature, dtype)

    if kind == "num":
        if splits is not None:
            lo, hi = sub[feature].min(), sub[feature].max()
            cortes = [s for s in sorted(splits) if lo < s < hi]
            modo = "manual"
        else:
            fit = self._fit_frame(sub, min_bin_size)
            x = fit[feature].to_numpy(dtype="float64")
            y = fit[self.target].to_numpy(dtype="float64")
            ok = ~np.isnan(y)
            x, y = x[ok], y[ok]                         # alvo NaN não entra no ajuste
            yfit = y.astype(int) if self._is_clf else y    # binário 0/1 só na classif.
            x_obs = x[~np.isnan(x)]
            # classificação exige as 2 classes presentes; regressão não.
            # "Menos de 2 distintos" por min == max (O(n), sem ordenar como o
            # np.unique): sem NaN em x_obs/yfit, equivale a unique().size < 2
            # (inclusive -0.0 == 0.0 e ±inf). O curto-circuito protege o min()
            # de array vazio. Nunca max-min == 0: inf - inf dá NaN.
            degenerado = (len(yfit) < 4 or x_obs.size == 0
                          or x_obs.min() == x_obs.max()
                          or (self._is_clf and yfit.min() == yfit.max()))
            if degenerado:
                cortes = []                            # dados degenerados → sem corte
            elif criterion != "optbin":
                min_n = max(1, int(min_bin_size * len(yfit)))
                t = _best_numeric_cut(x, yfit, criterion, self._is_clf, min_n)
                cortes = [t] if t is not None else []
            else:
                def _opt(mnb, mbs):
                    b = self._OptBin(
                        name=feature, dtype="numerical", max_n_bins=mnb,
                        min_bin_size=min_bin_size, max_bin_size=mbs,
                        monotonic_trend="auto_asc_desc",
                        **{self._diff_kwarg: min_mean_diff})
                    return _fit_optbinning_splits(b, x, yfit)
                cortes = _opt(max_n_bins, max_bin_size)
                if not cortes and max_bin_size is not None and relax_max:
                    # o máximo pode deixar o problema INFEASIBLE p/ este nº de
                    # bins/amostra: tenta mais bins e, por fim, relaxa o máximo
                    for k in range(max_n_bins + 1, 9):
                        cortes = _opt(k, max_bin_size)
                        if cortes:
                            break
                    if not cortes:
                        cortes = _opt(max_n_bins, None)
            modo = "ótimo" if criterion == "optbin" else criterion
        if not cortes:
            return [], modo, kind
        edges = [-np.inf, *cortes, np.inf]
        bins = [{"kind": "num", "lo": edges[i], "hi": edges[i + 1]}
                for i in range(len(edges) - 1)]
        if sub[feature].isna().any():
            bins.append({"kind": "na"})
        return bins, modo, kind

    # ---- categórico ----
    na_present = bool(sub[feature].isna().any())
    if splits is not None:
        grupos = [list(g) for g in splits]   # lista de grupos de categorias
        flat = [str(c) for g in grupos for c in g]
        if len(flat) != len(set(flat)):      # grupos devem ser disjuntos
            dup = sorted({c for c in flat if flat.count(c) > 1})
            raise ValueError(
                f"Grupos categóricos manuais têm categoria(s) repetida(s): {dup}. "
                "Cada categoria deve estar em um único grupo.")
        modo = "manual"
    else:
        fit = self._fit_frame(sub, min_bin_size)
        fit = fit[fit[feature].notna() & fit[self.target].notna()]  # NaN fora do ajuste
        xs = fit[feature].astype(str).to_numpy()
        yf = fit[self.target].to_numpy(dtype="float64")
        ys = yf.astype(int) if self._is_clf else yf       # binário só na classif.
        # ≥2 textos distintos por comparação com o 1º (O(n), sem ordenar
        # strings Python como o np.unique); len(ys) < 4 vem antes e garante
        # xs[0]. Sempre sobre xs JÁ em str: factorize/nunique na coluna bruta
        # fundiria 1/1.0/True e Decimal('1.0')/Decimal('1.00').
        degenerado = (len(ys) < 4 or not (xs[1:] != xs[0]).any()
                      or (self._is_clf and ys.min() == ys.max()))
        if degenerado:
            grupos = []                          # dados degenerados → sem grupos
        elif criterion != "optbin":
            min_n = max(1, int(min_bin_size * len(ys)))
            par = _best_categorical_split(xs, ys, criterion, self._is_clf, min_n)
            grupos = [list(par[0]), list(par[1])] if par is not None else []
        else:
            b = self._OptBin(
                name=feature, dtype="categorical", max_n_bins=max_n_bins,
                min_bin_size=min_bin_size, max_bin_size=max_bin_size,
                monotonic_trend="auto_asc_desc",
                **{self._diff_kwarg: min_mean_diff})
            grupos = [list(arr) for arr in _fit_optbinning_splits(b, xs, ys)]
        modo = "ótimo" if criterion == "optbin" else criterion
    _NA_TOK = {"nan", "NaN", "<NA>", "None"}
    bins = []
    for g in grupos:
        cats = [str(c) for c in g if str(c) not in _NA_TOK]
        if cats:
            bins.append({"kind": "cat", "cats": cats})
    if bins and na_present:
        bins.append({"kind": "na"})
    return bins, modo, kind


# ----------------------------------------------------------------------
# Auxiliares
# ----------------------------------------------------------------------
def _cache(seg):
    """``_bins_cache`` comparável entre instâncias: id(máscara) → sid, na ordem
    de inserção, valores por repr (estrito no tipo, NaN-safe) e a máscara
    guardada conferida com ``is``."""
    por_id = {id(v["mask"]): sid for sid, v in seg.segments.items()}
    out = []
    for k, v in seg._bins_cache.items():
        sid = por_id[k[0]]
        assert v[3] is seg.segments[sid]["mask"]
        out.append(((sid,) + k[1:], repr((v[0], v[1], v[2]))))
    return out


def _par(task, n=3000, **kw):
    a, b = _seg(task, n=n, **kw), _seg(task, n=n, **kw)
    a.n_jobs, b.n_jobs = 1, 4
    return a, b


def _quieto():
    cm = warnings.catch_warnings()
    cm.__enter__()
    warnings.simplefilter("ignore")
    return cm


def _sids(seg):
    return [None, "root", "nao_existe", *_internos(seg), *_folhas(seg)]


# ----------------------------------------------------------------------
# _resolve_bins: preparar → resolver → finalizar == cópia literal
# ----------------------------------------------------------------------
def _casos_resolve(task):
    alt = "gini" if task == "classification" else "variance"
    for crit in ("optbin", alt):
        for msz in (0.05, 0.2):
            yield dict(splits=None, dtype=None, max_n_bins=4, min_bin_size=msz,
                       criterion=crit)
    # máximo por bin (com e sem relaxar), diferença mínima e tipo forçado
    yield dict(splits=None, dtype=None, max_n_bins=3, min_bin_size=0.05,
               max_bin_size=0.3, relax_max=True)
    yield dict(splits=None, dtype=None, max_n_bins=3, min_bin_size=0.05,
               max_bin_size=0.3, relax_max=False)
    yield dict(splits=None, dtype=None, max_n_bins=5, min_bin_size=0.05,
               min_mean_diff=0.02)
    yield dict(splits=None, dtype="cat", max_n_bins=4, min_bin_size=0.05)


@pytest.mark.parametrize("task", TASKS)
def test_resolve_bins_igual_ao_literal(task):
    seg = _seg(task, n=1500)
    cm = _quieto()
    try:
        for sid in ["root", _folhas(seg)[1]]:
            m = seg.segments[sid]["mask"]
            for feat in FEATS:
                sub = seg._sub(m, feat)
                for kw in _casos_resolve(task):
                    try:
                        a = _resolve_bins_velho(seg, sub, feat, **kw)
                    except Exception as e:           # noqa: BLE001
                        with pytest.raises(type(e)):
                            seg._resolve_bins(sub, feat, **kw)
                        continue
                    b = seg._resolve_bins(sub, feat, **kw)
                    assert repr(a) == repr(b), (sid, feat, kw)
        # cortes / grupos manuais (inclusive grupos repetidos → ValueError)
        sub = seg._sub(seg.segments["root"]["mask"], "f64", "cat")
        for sp in ([-0.5, 0.0, 0.7], [100.0], []):
            assert repr(_resolve_bins_velho(seg, sub, "f64", sp, None, 4, 0.05)) == \
                repr(seg._resolve_bins(sub, "f64", sp, None, 4, 0.05))
        for sp in ([["a", "b"], ["c"]], [["a"], ["b", "c", "d"]]):
            assert repr(_resolve_bins_velho(seg, sub, "cat", sp, None, 4, 0.05)) == \
                repr(seg._resolve_bins(sub, "cat", sp, None, 4, 0.05))
        for f in (_resolve_bins_velho, TreeSegmenter._resolve_bins):
            with pytest.raises(ValueError, match="repetida"):
                f(seg, sub, "cat", [["a", "b"], ["b"]], None, 4, 0.05)
    finally:
        cm.__exit__(None, None, None)


@pytest.mark.parametrize("task", TASKS)
def test_preparar_amostrado_da_o_mesmo_ajuste(task, monkeypatch):
    """``amostrar=True`` sorteia no preparo o MESMO índice que o
    fit_optbinning_splits sortearia: resolver/finalizar dão o mesmo resultado."""
    monkeypatch.setattr(_common, "OPTBINNING_MAX_ROWS", 400)
    seg = _seg(task, n=1500)
    sub = seg._sub(seg.segments["root"]["mask"], *FEATS)
    cm = _quieto()
    try:
        n_amostrados = 0
        for feat in FEATS:
            a = seg._resolve_bins(sub, feat, None, None, 4, 0.05)
            p = seg._resolve_bins_preparar(sub, feat, None, None, 4, 0.05, amostrar=True)
            if p["fit"] is not None:
                assert len(p["fit"]["x"]) <= 400
                n_amostrados += len(p["fit"]["x"]) == 400
            b = seg._resolve_bins_finalizar(p, seg._resolve_bins_resolver(p))
            assert repr(a) == repr(b), feat
        assert n_amostrados >= 5            # o caminho amostrado foi exercitado
    finally:
        cm.__exit__(None, None, None)


# ----------------------------------------------------------------------
# variable_iv: n_jobs=1 × n_jobs=4 (DataFrame, attrs e _bins_cache)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("task", TASKS)
def test_variable_iv_njobs_1_x_4_exato(task):
    a, b = _par(task)
    filtros = list(warnings.filters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats_extra = ["f64", "cat", "nao_existe", "target", "f64", "hcard", "cat"]
        for sid in _sids(a):
            for with_psi in (True, False):
                for feats in (None, feats_extra):
                    ra = a.variable_iv(sid, features=feats, with_psi=with_psi)
                    rb = b.variable_iv(sid, features=feats, with_psi=with_psi)
                    _iguais(ra, rb)
                    assert _cache(a) == _cache(b), (sid, with_psi, feats)
        # de novo, já com tudo no cache (hits) — e de um segmentador novo
        for sid in _sids(a):
            _iguais(a.variable_iv(sid), b.variable_iv(sid))
        assert _cache(a) == _cache(b)
        c, d = _par(task)                    # do zero, só com as variáveis padrão
        for sid in _sids(a):
            _iguais(a.variable_iv(sid), d.variable_iv(sid))
            _iguais(c.variable_iv(sid), d.variable_iv(sid))
        assert _cache(c) == _cache(d)
    assert warnings.filters == filtros


@pytest.mark.parametrize("task", TASKS)
def test_variable_iv_njobs_igual_ao_velho(task):
    """n_jobs=4 contra a cópia literal do variable_iv de antes da onda 3.2."""
    seg = _seg(task)
    seg.n_jobs = 4
    velho = _seg(task)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sid in [None, *_folhas(seg)]:
            _iguais(_variable_iv_velho(velho, sid), seg.variable_iv(sid))


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("dup", [False, True])
def test_variable_iv_njobs_indice_e_amostragem(task, dup, monkeypatch):
    """Índice duplicado, sem amostra (sample_col None) e o ajuste AMOSTRADO
    (OPTBINNING_MAX_ROWS pequeno: o lote sorteia no preparo)."""
    monkeypatch.setattr(_common, "OPTBINNING_MAX_ROWS", 700)
    for sample in (True, False):
        a, b = _par(task, dup=dup, sample=sample)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for sid in [None, *_folhas(a)]:
                _iguais(a.variable_iv(sid), b.variable_iv(sid))
        assert _cache(a) == _cache(b)


@pytest.mark.parametrize("task", TASKS)
def test_variable_iv_njobs_usa_threads(task, monkeypatch):
    """Com n_jobs=4 os ajustes rodam fora da thread principal (pelo
    _fit_splits_na_thread); com n_jobs=1, nunca."""
    vistos = []
    orig = segmod._fit_splits_na_thread

    def espiao(b, x, y, falhas):
        vistos.append(threading.current_thread().name)
        return orig(b, x, y, falhas)
    monkeypatch.setattr(segmod, "_fit_splits_na_thread", espiao)
    a, b = _par(task)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a.variable_iv(None)
        assert vistos == []
        b.variable_iv(None)
    assert len(vistos) >= 5
    assert all(n.startswith("ygg-optbin") for n in vistos), set(vistos)
    # o executor é encerrado no fim da chamada: nenhuma thread do lote sobra
    assert not [t for t in threading.enumerate() if t.name.startswith("ygg-optbin")]


def test_variable_iv_features_iteravel_usa_serie():
    a, b = _par("classification")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ra = a.variable_iv(None, features=["f64", "cat", "i64"])
        rb = b.variable_iv(None, features=iter(["f64", "cat", "i64"]))
    _iguais(ra, rb)
    assert _cache(a) == _cache(b)


@pytest.mark.parametrize("task", TASKS)
def test_fit_auto_njobs_igual(task):
    def arvore(nj):
        seg = _seg(task)
        seg.n_jobs = nj
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seg.fit_auto(max_depth=2, features=["f64", "i64", "cat", "mix", "dec",
                                                  "hcard", "flag", "strd"],
                         verbose=False)
        return seg.to_dict()
    assert arvore(1) == arvore(4)


# ----------------------------------------------------------------------
# _bins_cache: backstop no meio do lote e variável repetida
# ----------------------------------------------------------------------
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("cheio", [5990, 5995, 5999, 6000, 6001])
def test_backstop_no_meio_do_lote(task, cheio):
    """O cache já quase no teto: o clear (len > 6000) cai no meio do lote — o
    estado final (chaves, ordem e valores) tem de ser o do ajuste em série,
    inclusive com variáveis repetidas antes e depois do clear."""
    a, b = _par(task)
    feats = ["f64", "cat", "i64", "f64", "hcard", "mix", "cat", "dec", "u8",
             "f64_tie", "flag", "f64", "strd", "catd", "cat_lit", "i64"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # aquece algumas variáveis (hits no preparo que podem sumir no clear)
        for s in (a, b):
            s.variable_iv(None, features=["mix", "dec", "u8"])
            for i in range(cheio - len(s._bins_cache)):
                s._bins_cache[("lixo", i)] = ([], "x", "num", None)
        ra = a.variable_iv(None, features=feats)
        rb = b.variable_iv(None, features=feats)
    _iguais(ra, rb)
    assert _cache_com_lixo(a) == _cache_com_lixo(b)
    # com 5990 cabem as entradas novas (sem clear); nos demais o clear acontece
    assert (len(a._bins_cache) < cheio) == (cheio >= 5995)


def _cache_com_lixo(seg):
    """Como :func:`_cache`, mantendo as entradas de enchimento (na ordem)."""
    por_id = {id(v["mask"]): sid for sid, v in seg.segments.items()}
    out = []
    for k, v in seg._bins_cache.items():
        if k[0] == "lixo":
            out.append(k)
            continue
        assert v[3] is seg.segments[por_id[k[0]]]["mask"]
        out.append(((por_id[k[0]],) + k[1:], repr((v[0], v[1], v[2]))))
    return out


# ----------------------------------------------------------------------
# warnings: filtros restaurados, ruído do ajuste calado, falha avisada
# ----------------------------------------------------------------------
class _OptBinFalha:
    """Binning que falha (TypeError) nas variáveis de ``ruins`` e, nas demais,
    emite um aviso "de ruído" no fit (como o divide-by-zero benigno) e delega."""

    ruins = ("cat", "i64")

    def __init__(self, base, **kw):
        self._b = base(**kw)
        self.name = kw.get("name")

    def fit(self, x, y):
        warnings.warn("ruido do ajuste", RuntimeWarning)
        if self.name in self.ruins:
            raise TypeError(f"quebrou {self.name}")
        return self._b.fit(x, y)

    @property
    def splits(self):
        return self._b.splits


def _com_falha(seg):
    base = seg._OptBin
    seg._OptBin = lambda **kw: _OptBinFalha(base, **kw)
    return seg


@pytest.mark.parametrize("task", TASKS)
def test_avisos_de_falha_iguais_e_filtros_restaurados(task):
    feats = ["f64", "cat", "hcard", "i64", "mix", "cat", "dec", "u8", "f64_tie"]
    regs = {}
    for nj in (1, 4):
        seg = _com_falha(_seg(task))
        seg.n_jobs = nj
        filtros = list(warnings.filters)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            iv = seg.variable_iv(None, features=feats)
            dentro = list(warnings.filters)
        assert warnings.filters == filtros
        assert dentro[0][0] == "always"          # o ignore do lote não vazou
        regs[nj] = ([(str(w.message), w.category, w.filename, w.lineno)
                     for w in rec], iv, _cache(seg))
    assert regs[1][0] == regs[4][0]
    _iguais(regs[1][1], regs[4][1])
    assert regs[1][2] == regs[4][2]
    msgs = [m for m, *_ in regs[4][0]]
    assert any("optbinning falhou inesperadamente em 'cat'" in m for m in msgs)
    assert any("optbinning falhou inesperadamente em 'i64'" in m for m in msgs)
    assert not any("ruido" in m for m in msgs)    # ruído do fit segue calado
    # origem: o próprio fit_optbinning_splits (como no ajuste em série)
    assert {f for _m, _c, f, _l in regs[4][0]} == {_common.__file__}


@pytest.mark.parametrize("task", TASKS)
def test_excecao_por_variavel(task):
    """Exceção no preparo (coluna inexistente) e no ajuste (construtor do
    binning) → tipo '—', 0 bins e IV NaN, igual ao ajuste em série."""
    def seg_com_erro(nj):
        seg = _seg(task)
        seg.n_jobs = nj
        base = seg._OptBin

        def fabrica(**kw):
            if kw.get("name") == "hcard":
                raise RuntimeError("construtor quebrou")
            return base(**kw)
        seg._OptBin = fabrica
        return seg
    a, b = seg_com_erro(1), seg_com_erro(4)
    feats = ["f64", "hcard", "nao_existe", "cat", "hcard", "i64"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ra = a.variable_iv(None, features=feats)
        rb = b.variable_iv(None, features=feats)
    _iguais(ra, rb)
    linha = rb.set_index("variavel").loc[["hcard", "nao_existe"]]
    assert (linha["tipo"] == "—").all() and (linha["n_bins"] == 0).all()
    assert linha["iv"].isna().all()
    assert _cache(a) == _cache(b)


# ----------------------------------------------------------------------
# cancel_futures em erro (KeyboardInterrupt / Cancel do Databricks)
# ----------------------------------------------------------------------
class _ExecutorEspiao(segmod.ThreadPoolExecutor):
    desligamentos: list = []

    def shutdown(self, wait=True, *, cancel_futures=False):
        _ExecutorEspiao.desligamentos.append((wait, cancel_futures))
        return super().shutdown(wait=wait, cancel_futures=cancel_futures)


def test_interrupcao_no_lote_cancela(monkeypatch):
    _ExecutorEspiao.desligamentos = []
    monkeypatch.setattr(segmod, "ThreadPoolExecutor", _ExecutorEspiao)
    seg = _seg("classification")
    seg.n_jobs = 2
    orig = TreeSegmenter._resolve_bins_resolver

    def resolver(self, prep, ajusta=None):
        if prep["fit"]["feature"] == "cat":
            raise KeyboardInterrupt
        return orig(self, prep, ajusta)
    monkeypatch.setattr(TreeSegmenter, "_resolve_bins_resolver", resolver)
    filtros = list(warnings.filters)
    with pytest.raises(KeyboardInterrupt):
        seg.variable_iv(None, features=["f64", "cat", "i64", "hcard", "mix", "dec"])
    assert warnings.filters == filtros
    # interrupção no lote: cancela os pendentes e ESPERA os ajustes em curso
    assert _ExecutorEspiao.desligamentos == [(True, True)]


class _OptBinLento:
    """``cat`` interrompe (KeyboardInterrupt) logo no fit; as demais imitam o
    sklearn/pandas: abrem ``catch_warnings`` + ``simplefilter`` DENTRO do fit (na
    thread do worker), demoram e, já fora dele, emitem um aviso tardio."""

    terminou: list = []

    def __init__(self, base, **kw):
        self._b = base(**kw)
        self.name = kw.get("name")

    def fit(self, x, y):
        if self.name == "cat":
            time.sleep(0.05)                      # o outro já está no catch_warnings
            raise KeyboardInterrupt
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)   # como o check_array
            time.sleep(0.3)
            self._b.fit(x, y)
        warnings.warn(f"aviso tardio de {self.name}", FutureWarning)
        _OptBinLento.terminou.append(self.name)

    @property
    def splits(self):
        return self._b.splits


def test_interrupcao_no_lote_nao_vaza_filtros():
    """Cancel/KeyboardInterrupt com um ajuste em curso que mexe nos filtros
    globais (como o sklearn faz no worker): o lote espera esse ajuste DENTRO do
    seu catch_warnings → nada é exibido depois do cancel e o "ignore" do lote não
    vaza para a sessão (antes: ``warnings.filters[0]`` virava o ignore)."""
    _OptBinLento.terminou = []
    seg = _seg("classification")
    seg.n_jobs = 2
    base = seg._OptBin
    seg._OptBin = lambda **kw: _OptBinLento(base, **kw)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        antes = list(warnings.filters)
        with pytest.raises(KeyboardInterrupt):
            seg.variable_iv(None, features=["cat", "f64"])   # quem interrompe vem 1º
        assert _OptBinLento.terminou == ["f64"]  # o ajuste em curso foi esperado
        assert warnings.filters == antes
        time.sleep(0.5)
        assert warnings.filters == antes         # nada repôs o ignore depois
        assert not any(t.name.startswith("ygg-optbin") for t in threading.enumerate())
        assert [str(w.message) for w in rec] == []   # nada exibido após o cancel
        warnings.warn("aviso da sessão depois do cancel", UserWarning)
        assert [str(w.message) for w in rec] == ["aviso da sessão depois do cancel"]


def test_interrupcao_no_iv_cancela(monkeypatch):
    """BaseException fora do lote (no laço do IV): o executor é encerrado sem
    esperar e com cancel_futures; com sucesso, encerrado esperando."""
    _ExecutorEspiao.desligamentos = []
    monkeypatch.setattr(segmod, "ThreadPoolExecutor", _ExecutorEspiao)
    seg = _seg("classification")
    seg.n_jobs = 2
    feats = ["f64", "cat", "i64", "hcard", "mix", "dec", "u8"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.variable_iv(None, features=feats)
        assert _ExecutorEspiao.desligamentos == [(True, False)]
        _ExecutorEspiao.desligamentos = []
        folha = _folhas(seg)[0]
        orig = TreeSegmenter._bin_ids

        def bin_ids(col, bins):
            if col.name == "hcard":
                raise KeyboardInterrupt
            return orig(col, bins)
        monkeypatch.setattr(TreeSegmenter, "_bin_ids", staticmethod(bin_ids))
        with pytest.raises(KeyboardInterrupt):
            seg.variable_iv(folha, features=feats)
    assert _ExecutorEspiao.desligamentos == [(False, True)]


# ----------------------------------------------------------------------
# n_jobs: normalização, fora do JSON, from_dict/load e a UI
# ----------------------------------------------------------------------
def test_n_jobs_normalizacao_e_nao_persistido(tmp_path):
    df = _base(n=400)
    kw = dict(target="target", sample_col="amostra", verbose=False)
    assert TreeSegmenter(df, **kw).n_jobs == 1
    assert TreeSegmenter(df, n_jobs=None, **kw).n_jobs == 1
    assert TreeSegmenter(df, n_jobs=3, **kw).n_jobs == 3
    assert TreeSegmenter(df, n_jobs=-1, **kw).n_jobs == (os.cpu_count() or 1)
    for ruim in (0, -2):
        with pytest.raises(ValueError, match="n_jobs"):
            TreeSegmenter(df, n_jobs=ruim, **kw)
    s1 = TreeSegmenter(df, n_jobs=1, **kw)
    s4 = TreeSegmenter(df, n_jobs=4, **kw)
    assert s1.to_dict() == s4.to_dict()
    assert "n_jobs" not in str(s4.to_dict())
    d = s4.to_dict()
    assert TreeSegmenter.from_dict(d, df).n_jobs == 1
    assert TreeSegmenter.from_dict(d, df, n_jobs=3).n_jobs == 3
    p = tmp_path / "a.json"
    s4.save(str(p))
    assert TreeSegmenter.load(str(p), df).n_jobs == 1
    assert TreeSegmenter.load(str(p), df, n_jobs=2).n_jobs == 2


def test_ui_usa_n_jobs_ate_8(monkeypatch):
    from yggdrasil.credit_risk.tree import TreeSegmenterUI
    df = _base(n=600)
    vistos = []
    orig = TreeSegmenter.__init__

    def espiao(self, *a, n_jobs=1, **k):
        vistos.append(n_jobs)
        return orig(self, *a, n_jobs=n_jobs, **k)
    monkeypatch.setattr(TreeSegmenter, "__init__", espiao)
    esperado = min(os.cpu_count() or 1, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui = TreeSegmenterUI(df, target="target", sample_col="amostra",
                             features=["f64", "cat", "i64"], date_col="dt")
        assert ui.seg.n_jobs == esperado
        ui._on_reset(None)
        assert ui.seg.n_jobs == esperado
        ui._on_cv_reset(ui.btn_cv_reset)                 # 2 cliques: arma e confirma
        ui._on_cv_reset(ui.btn_cv_reset)
        assert ui.seg.n_jobs == esperado
    assert vistos[:1] == [esperado] and vistos.count(esperado) >= 3
    assert "n_jobs" not in ui._kwargs


def test_cpsat_deterministico_chega_ao_optbinning_nas_threads():
    """A fixture do módulo (``cpsat_deterministico``) vale para o solver que o
    optbinning cria — inclusive dentro das threads ygg-optbin do ``n_jobs``."""
    from optbinning import OptimalBinning
    from ortools.sat.python import cp_model
    workers = []
    cls = cp_model.CpSolver

    class _Espiao(cls):
        def Solve(self, *a, **k):                    # noqa: N802 - API do ortools
            workers.append((threading.current_thread().name,
                            self.parameters.num_workers))
            return super().Solve(*a, **k)
    orig = cp_model.CpSolver
    cp_model.CpSolver = _Espiao
    try:
        rng = np.random.default_rng(0)
        x = rng.choice(list("ABCDEFG"), 3000)
        y = (rng.uniform(size=3000) < np.where(np.isin(x, list("ABC")), 0.3, 0.1)).astype(int)
        OptimalBinning(dtype="categorical", solver="cp").fit(x, y)
        seg = _seg(n=1500)
        seg.n_jobs = 4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seg.variable_iv(features=list(FEATS))
    finally:
        cp_model.CpSolver = orig
    assert workers and all(w == 1 for _, w in workers)
    assert any(n.startswith("ygg-optbin") for n, _ in workers)
