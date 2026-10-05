"""Onda 2A — ``_resolve_bins_uncached`` numérico por gather (sem copiar o recorte)
e rejeição de binária sobre o prefixo (sem ``dropna`` da coluna inteira).

Cada caso compara contra a versão ANTIGA copiada aqui: os (x, y) entregues a
``b.fit`` (``np.array_equal(equal_nan=True)`` e mesmo dtype), os bins e o kind."""
from __future__ import annotations

import contextlib
import decimal
import io
import warnings

import numpy as np
import pandas as pd
import pytest
from optbinning import ContinuousOptimalBinning, OptimalBinning

from yggdrasil.credit_risk import _common
from yggdrasil.credit_risk.model import ModelSegmenter
from yggdrasil.credit_risk.model import segmenter as segmod
from yggdrasil.credit_risk.model.segmenter import _primeiros_observados


# ───────────────────────── versões antigas (referência) ─────────────────────────
def _old_bins_binaria(self, feature, fit, kind):
    col = fit[feature]
    obs = col.dropna()
    if pd.unique(obs.iloc[:10_000]).size > 2:
        return None
    if kind == "num":
        vals = np.sort(pd.unique(obs.to_numpy(dtype="float64")))
        if vals.size != 2:
            return None
        bins = [{"kind": "num", "lo": -np.inf, "hi": float(vals.mean())},
                {"kind": "num", "lo": float(vals.mean()), "hi": np.inf}]
    else:
        if obs.nunique() > 2:
            return None
        niveis = sorted({str(v) for v in pd.unique(obs)} - {"nan", "NaN", "<NA>", "None"})
        if len(niveis) != 2:
            return None
        bins = [{"kind": "cat", "cats": [v]} for v in niveis]
    if col.isna().any():
        bins.append({"kind": "na"})
    self._avisa_binaria_rara(feature, fit, bins)
    return bins


def _old_resolve(self, feature, max_n_bins=5, min_bin_size=0.05, splits=None, sample=None):
    fos = segmod._fit_optbinning_splits
    if splits is None:
        splits = self.var_meta.get(feature, {}).get("splits")
    fit = self._frame(sample, cols=[feature, self.target])
    kind = self._detect_kind(feature, fit)
    derivada = self._faixas_da_origem(feature)
    if derivada is not None:
        return self._bins_derivada(feature, fit, splits, derivada), "cat"
    if splits is None:
        binaria = _old_bins_binaria(self, feature, fit, kind)
        if binaria is not None:
            return binaria, kind
    if kind == "num":
        if splits is not None:
            lo, hi = fit[feature].min(), fit[feature].max()
            cortes = [s for s in sorted(splits) if lo < s < hi]
        else:
            x = fit[feature].to_numpy(dtype="float64")
            y = fit[self.target].to_numpy(dtype="float64")
            ok = ~np.isnan(y)
            x, y = x[ok], y[ok]
            x_obs = x[~np.isnan(x)]
            if len(y) < 4 or x_obs.size == 0 or x_obs.min() == x_obs.max():
                cortes = []
            else:
                if self.task_type == "classification":
                    b = OptimalBinning(name=feature, dtype="numerical",
                                       max_n_bins=max_n_bins, min_bin_size=min_bin_size,
                                       monotonic_trend="auto_asc_desc")
                    cortes = fos(b, x, y.astype(int))
                else:
                    b = ContinuousOptimalBinning(
                        name=feature, dtype="numerical", max_n_bins=max_n_bins,
                        min_bin_size=min_bin_size, monotonic_trend="auto_asc_desc")
                    cortes = fos(b, x, y)
        if not cortes:
            return [], kind
        edges = [-np.inf, *cortes, np.inf]
        bins = [{"kind": "num", "lo": edges[i], "hi": edges[i + 1]}
                for i in range(len(edges) - 1)]
        if fit[feature].isna().any():
            bins.append({"kind": "na"})
        return self._aplica_destino_na(feature, bins, fit, splits is not None), kind
    na_present = bool(fit[feature].isna().any())
    if splits is not None:
        grupos = [list(g) for g in splits]
    else:
        f2 = fit[fit[feature].notna() & fit[self.target].notna()]
        xs = f2[feature].astype(str).to_numpy()
        ys = f2[self.target].to_numpy(dtype="float64")
        if len(ys) < 4 or not (xs != xs[0]).any():
            grupos = []
        else:
            if self.task_type == "classification":
                b = OptimalBinning(name=feature, dtype="categorical",
                                   max_n_bins=max_n_bins, min_bin_size=min_bin_size,
                                   monotonic_trend="auto_asc_desc")
                grupos = [list(a) for a in fos(b, xs, ys.astype(int))]
            else:
                b = ContinuousOptimalBinning(
                    name=feature, dtype="categorical", max_n_bins=max_n_bins,
                    min_bin_size=min_bin_size, monotonic_trend="auto_asc_desc")
                grupos = [list(a) for a in fos(b, xs, ys)]
    _NA_TOK = {"nan", "NaN", "<NA>", "None"}
    bins = []
    for g in grupos:
        cats = [str(c) for c in g if str(c) not in _NA_TOK]
        if cats:
            bins.append({"kind": "cat", "cats": cats})
    if bins and na_present:
        bins.append({"kind": "na"})
    return self._aplica_destino_na(feature, bins, fit, splits is not None), kind


# ───────────────────────── captura do b.fit ─────────────────────────
@pytest.fixture()
def captura(monkeypatch):
    """Lista de (x, y) que chegam a ``b.fit`` (cópias), nos dois binnings."""
    vistos = []
    for cls in (OptimalBinning, ContinuousOptimalBinning):
        orig = cls.fit

        def fit(self, x, y, *a, _orig=orig, **k):
            vistos.append((np.array(x, copy=True), np.array(y, copy=True)))
            return _orig(self, x, y, *a, **k)
        monkeypatch.setattr(cls, "fit", fit)
    return vistos


def _roda(captura, fn):
    captura.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = fn()
    return res, list(captura), [str(x.message) for x in w
                                if issubclass(x.category, UserWarning)
                                and not issubclass(x.category, RuntimeWarning)]


def _mesmos_fits(a, b):
    assert len(a) == len(b)
    for (xa, ya), (xb, yb) in zip(a, b):
        assert xa.dtype == xb.dtype and ya.dtype == yb.dtype
        assert np.array_equal(xa, xb, equal_nan=xa.dtype.kind == "f")
        assert np.array_equal(ya, yb, equal_nan=ya.dtype.kind == "f")


def _iguais(a, b) -> bool:
    """``==`` com NaN igual a NaN (a binária de Float64 com 0/0 já gerava corte NaN)."""
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_iguais(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(map(_iguais, a, b))
    return type(a) is type(b) and a == b


def _compara(seg, captura, feature, sample=None, **kw):
    """Antigo × novo numa variável: bins, kind, (x, y) do fit e avisos."""
    old, fo, wo = _roda(captura, lambda: _old_resolve(seg, feature, sample=sample, **kw))
    new, fn, wn = _roda(captura, lambda: seg._resolve_bins_uncached(feature, sample=sample, **kw))
    assert _iguais(new, old), (feature, sample)
    _mesmos_fits(fo, fn)
    assert wn == wo
    return new, fn


# ───────────────────────── base ─────────────────────────
def _base(n=8000, seed=3, nan_inicio=0):
    rng = np.random.default_rng(seed)
    amostra = np.where(rng.random(n) < .7, "DES", "OOT")
    xc = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(xc - 1)))
    y = (rng.random(n) < p).astype(float)
    y[rng.random(n) < .05] = np.nan                       # alvo com NaN na DES
    df = pd.DataFrame({"amostra": amostra, "target": y})
    df["x_cont"] = np.where(rng.random(n) < .1, np.nan, xc)
    df["x_int"] = rng.integers(0, 40, n)
    xi = pd.array(rng.integers(0, 25, n), dtype="Int64")
    xi[rng.random(n) < .1] = pd.NA
    df["x_Int64"] = xi
    f = pd.Series(rng.integers(0, 6, n).astype(float), dtype="Float64")
    df["x_F64_00_bin"] = f / f                             # 0/0 ⇒ NaN que não é NA
    g = pd.Series(np.where(f.to_numpy(float) == 0, 0.0, 2.0), dtype="Float64")
    df["x_F64_00"] = (f + pd.Series(np.round(xc, 1), dtype="Float64").where(f != 0, 0.0)) / g
    df["x_F64"] = pd.array(np.round(xc, 2), dtype="Float64")
    df["x_toda_nan"] = np.nan
    df["x_const"] = 3.0
    df["x_const_nan"] = np.where(rng.random(n) < .3, np.nan, 7.0)
    df["x_flag"] = (rng.random(n) < .02).astype(int)
    df["x_flag_nan"] = np.where(rng.random(n) < .2, np.nan, (rng.random(n) < .4) * 1.0)
    df["x_inf"] = np.where(rng.random(n) < .05, np.inf, xc)
    df["x_neg0"] = rng.choice([-0.0, 0.0, 1.0, 2.0], n)
    df["x_bigint"] = (2 ** 60 + rng.integers(0, 3, n)).astype(np.int64)
    df["x_3niveis_tarde"] = np.where(np.arange(n) < n // 2, 1.0, rng.choice([1.0, 2.0, 3.0], n))
    df["x_obs_so_alvo_nan"] = np.where(np.isnan(y), xc, np.nan)
    df["c"] = rng.choice(list("abcd"), n)
    df["b"] = rng.random(n) < .3
    if nan_inicio:
        for nome, niveis in (("x_ini2", [0.0, 1.0]), ("x_ini3", [0.0, 1.0, 2.0])):
            v = rng.choice(niveis, n)
            v[:nan_inicio] = np.nan
            df[nome] = v
        v = xc.copy()
        v[:nan_inicio] = np.nan
        df["x_ini_cont"] = v
    return df


def _seg(df, **kw):
    kw.setdefault("sample_col", "amostra")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ModelSegmenter(df, target="target", verbose=False, **kw)


NUMS = ["x_cont", "x_int", "x_Int64", "x_F64_00", "x_F64_00_bin", "x_F64", "x_toda_nan", "x_const",
        "x_const_nan", "x_flag", "x_flag_nan", "x_inf", "x_neg0", "x_bigint",
        "x_3niveis_tarde", "x_obs_so_alvo_nan"]


@pytest.fixture()
def cap_pequeno(monkeypatch):
    monkeypatch.setattr(_common, "OPTBINNING_MAX_ROWS", 2000)   # DES (~5.3k) acima do teto


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_numericas_iguais_ao_antigo(captura, cap_pequeno, task):
    df = _base()
    if task == "regression":
        df["target"] = np.where(df["target"].isna(), np.nan,
                                np.random.default_rng(1).random(len(df)))
    seg = _seg(df, task_type=task)
    algum_fit = 0
    for f in NUMS + ["c", "b"]:
        for sample in (None, "OOT"):
            _, fits = _compara(seg, captura, f, sample=sample)
            algum_fit += len(fits)
    assert algum_fit >= 10                                 # o optbinning rodou de fato
    # o x/y entregue tem o tamanho do teto (sorteio de 2000 da DES)
    _, fits = _compara(seg, captura, "x_cont")
    assert len(fits) == 1 and len(fits[0][0]) == 2000


def test_caminho_rapido_nao_copia_o_recorte(cap_pequeno, monkeypatch):
    seg = _seg(_base())
    chamadas = []
    orig = ModelSegmenter._frame
    monkeypatch.setattr(ModelSegmenter, "_frame",
                        lambda self, *a, **k: chamadas.append(1) or orig(self, *a, **k))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in ["x_cont", "x_int", "x_Int64", "x_F64", "x_inf", "x_obs_so_alvo_nan"]:
            seg._resolve_bins_uncached(f)
        assert chamadas == []
        seg._resolve_bins_uncached("x_flag")              # binária: caminho do recorte
        assert chamadas == [1]
    # posições/sorteio calculados uma vez por (amostra, teto, alvo)
    assert len(seg._ajuste_pos_cache) == 1


def test_sem_sample_col(captura, cap_pequeno):
    seg = _seg(_base().drop(columns="amostra"), sample_col=None)
    for f in NUMS:
        _compara(seg, captura, f)
    assert list(seg._ajuste_pos_cache)[0][0] is None


def test_nan_no_comeco_seguido_de_2_e_3_niveis(captura, cap_pequeno):
    seg = _seg(_base(n=90_000, nan_inicio=60_000))
    for f in ("x_ini2", "x_ini3", "x_ini_cont"):
        for sample in (None, "OOT"):
            _compara(seg, captura, f, sample=sample)
    assert seg._resolve_bins_uncached("x_ini2")[0][0]["hi"] == 0.5     # binária


def test_bins_manuais_com_pior_e_derivada(captura, cap_pequeno):
    seg = _seg(_base())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.set_manual_bins("x_cont", [-0.5, 0.5], missing="pior")
        seg.set_manual_bins("x_Int64", [5, 12], missing=1)
        nova = seg.create_categorical("x_int")
    for f in ("x_cont", "x_Int64", nova, "x_int"):
        for sample in (None, "OOT"):
            _compara(seg, captura, f, sample=sample)
    assert any(b.get("include_na") for b in seg._resolve_bins_uncached("x_cont")[0])


def test_teto_alterado_nao_reaproveita_sorteio(captura, monkeypatch):
    seg = _seg(_base())
    for cap in (2000, 3500, 0, 2000):
        monkeypatch.setattr(_common, "OPTBINNING_MAX_ROWS", cap)
        _, fits = _compara(seg, captura, "x_cont")
        assert len(fits[0][0]) == (cap or len(seg._posicoes_ajuste()[1]))
    assert len(seg._ajuste_pos_cache) == 3


def test_des_acima_do_teto_real(captura):
    # teto padrão (300k): DES ~315k linhas, só o caminho numérico
    n = 450_000
    rng = np.random.default_rng(11)
    xc = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(xc - 1)))).astype(float)
    y[rng.random(n) < .02] = np.nan
    df = pd.DataFrame({"amostra": np.where(rng.random(n) < .7, "DES", "OOT"), "target": y,
                       "x": np.where(rng.random(n) < .1, np.nan, xc)})
    seg = _seg(df)
    _, fits = _compara(seg, captura, "x")
    assert len(fits[0][0]) == _common.OPTBINNING_MAX_ROWS


def test_ranking_inteiro_igual(cap_pequeno, monkeypatch):
    df = _base()
    novo = _seg(df.copy())
    velho = _seg(df.copy())
    monkeypatch.setattr(velho, "_resolve_bins_uncached",
                        lambda *a, **k: _old_resolve(velho, *a, **k))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b = velho.variable_iv(), novo.variable_iv()
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert velho._bins_cache.keys() == novo._bins_cache.keys()
    assert all(_iguais(velho._bins_cache[k], novo._bins_cache[k]) for k in velho._bins_cache)


def test_ajuste_pos_cache_no_registro_e_zerado_no_detached(cap_pequeno):
    seg = _seg(_base())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg._resolve_bins_uncached("x_cont")
    assert ("_ajuste_pos_cache", dict) in ModelSegmenter._ROW_CACHES
    assert seg._ajuste_pos_cache
    assert seg._detached_scorer()._ajuste_pos_cache == {}


# ───────────────────────── _bins_binaria: rejeição pelo prefixo ─────────────────────────
def _series_binaria():
    rng = np.random.default_rng(0)
    def nan_ini(k, niveis, n=20_000):
        v = np.full(k + n, np.nan)
        v[k:] = rng.choice(niveis, n)
        return pd.Series(v)
    obj = pd.Series(rng.choice(["a", "b"], 30_000).astype(object))
    obj[::7] = None
    dec = pd.Series([decimal.Decimal(rng.choice(["1.0", "2.5"])) for _ in range(5000)]
                    + [decimal.Decimal("NaN")] * 3, dtype=object)
    misto = pd.Series(["x", None, np.nan, pd.NA, "y"] * 3000, dtype=object)
    ib = pd.array(rng.integers(0, 2, 30_000), dtype="Int64")
    ib[::5] = pd.NA
    bb = pd.array(rng.random(30_000) < .5, dtype="boolean")
    bb[::3] = pd.NA
    idx_dup = pd.Series(rng.choice([0.0, 1.0, np.nan], 25_000),
                        index=rng.integers(0, 50, 25_000))
    return {
        "nan60k_2": (nan_ini(60_000, [0.0, 1.0]), "num"),
        "nan60k_3": (nan_ini(60_000, [0.0, 1.0, 2.0]), "num"),
        "nan300k_2": (nan_ini(300_000, [0.0, 1.0]), "num"),
        "nan300k_3": (nan_ini(300_000, [0.0, 1.0, 2.0]), "num"),
        "tarde_3": (pd.Series(np.r_[np.zeros(15_000), np.ones(15_000), [2.0]]), "num"),
        "continua": (pd.Series(rng.normal(size=30_000)), "num"),
        "flag_nan": (pd.Series(np.where(rng.random(30_000) < .1, np.nan,
                                        (rng.random(30_000) < .03) * 1.0)), "num"),
        "toda_nan": (pd.Series(np.full(30_000, np.nan)), "num"),
        "curta": (pd.Series([0.0, 1.0, np.nan, 1.0]), "num"),
        "Int64": (pd.Series(ib), "num"),
        "boolean": (pd.Series(bb), "cat"),
        "bool": (pd.Series(rng.random(30_000) < .5), "cat"),
        "obj2": (obj, "cat"),
        "obj3": (pd.Series(rng.choice(list("abc"), 30_000).astype(object)), "cat"),
        "decimal": (dec, "cat"),
        "misto_na": (misto, "cat"),
        "category": (pd.Series(rng.choice(["u", "v"], 30_000)).astype("category"), "cat"),
        "idx_dup": (idx_dup, "num"),
        "F64_00": (pd.Series([0.0, 1.0, 2.0] * 5000, dtype="Float64")
                   / pd.Series([0.0, 1.0, 1.0] * 5000, dtype="Float64"), "num"),
    }


@pytest.mark.parametrize("nome", list(_series_binaria()))
def test_bins_binaria_igual_ao_antigo(nome):
    s, kind = _series_binaria()[nome]
    seg = _seg(_base(n=500))
    rng = np.random.default_rng(1)
    fit = pd.DataFrame({"v": s.array,
                        "target": (rng.random(len(s)) < .3).astype(float)}, index=s.index)
    seg.var_meta.setdefault("v", {})
    with warnings.catch_warnings(record=True) as wo:
        warnings.simplefilter("always")
        old = _old_bins_binaria(seg, "v", fit, kind)
    with warnings.catch_warnings(record=True) as wn:
        warnings.simplefilter("always")
        new = seg._bins_binaria("v", fit, kind)
    assert _iguais(new, old)
    assert [str(w.message) for w in wn] == [str(w.message) for w in wo]


@pytest.mark.parametrize("nome", list(_series_binaria()))
def test_primeiros_observados_igual_dropna(nome):
    s, _ = _series_binaria()[nome]
    for n in (1, 10_000, 40_000):
        ref = s.dropna().iloc[:n]
        got = _primeiros_observados(s, n=n)
        assert got.dtype == ref.dtype
        assert pd.unique(got).size == pd.unique(ref).size
        pd.testing.assert_series_equal(got.reset_index(drop=True),
                                       ref.reset_index(drop=True), check_exact=True)
    mask = np.random.default_rng(2).random(len(s)) < .4
    ref = s[mask].dropna().iloc[:10_000]
    got = _primeiros_observados(s, mask)
    pd.testing.assert_series_equal(got.reset_index(drop=True),
                                   ref.reset_index(drop=True), check_exact=True)


def test_float64_pyarrow_nan_e_null(captura, cap_pequeno):
    pa = pytest.importorskip("pyarrow")
    df = _base()
    rng = np.random.default_rng(4)
    v = np.round(rng.normal(size=len(df)), 2).tolist()
    for i in range(0, len(v), 13):
        v[i] = None
    for i in range(5, len(v), 17):
        v[i] = float("nan")
    df["x_pa"] = pd.array(pa.array(v, from_pandas=False), dtype=pd.ArrowDtype(pa.float64()))
    seg = _seg(df)
    for sample in (None, "OOT"):
        _compara(seg, captura, "x_pa", sample=sample)
