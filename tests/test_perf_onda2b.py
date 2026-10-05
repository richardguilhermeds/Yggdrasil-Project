"""Onda 2B — ``_resolve_bins_uncached`` categórico com a coluna fatorada UMA vez
(faltante, linhas de ajuste e constância pelos códigos; texto só nas linhas
sorteadas).

Cada caso compara contra a versão ANTIGA copiada aqui: o ``xs``/``ys`` entregue a
``b.fit`` (``np.array_equal`` e mesmo dtype), os grupos/bins, o kind e os avisos."""
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

D = decimal.Decimal


# ───────────────────────── versão antiga (referência) ─────────────────────────
def _old_resolve(self, feature, max_n_bins=5, min_bin_size=0.05, splits=None, sample=None):
    """Caminho categórico de antes da 2B (o numérico/binária/derivada/manual
    seguem iguais e são delegados ao código atual pelo recorte)."""
    fos = segmod._fit_optbinning_splits
    if splits is None:
        splits = self.var_meta.get(feature, {}).get("splits")
    fit = self._frame(sample, cols=[feature, self.target])
    kind = self._detect_kind(feature, fit)
    if (kind == "num" or splits is not None or self._faixas_da_origem(feature) is not None):
        return ModelSegmenter._resolve_bins_uncached(self, feature, max_n_bins, min_bin_size,
                                                     splits, sample)
    binaria = self._bins_binaria(feature, fit, kind)
    if binaria is not None:
        return binaria, kind
    na_present = bool(fit[feature].isna().any())
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
    return self._aplica_destino_na(feature, bins, fit, False), kind


# ───────────────────────── captura do b.fit ─────────────────────────
@pytest.fixture()
def captura(monkeypatch):
    """(x, y, splits) de cada ``b.fit`` (cópias) e os grupos que ele devolveu."""
    vistos = []
    for cls in (OptimalBinning, ContinuousOptimalBinning):
        orig = cls.fit

        def fit(self, x, y, *a, _orig=orig, **k):
            r = _orig(self, x, y, *a, **k)
            vistos.append((np.array(x, copy=True), np.array(y, copy=True),
                           [list(g) for g in self.splits]))
            return r
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
    for (xa, ya, ga), (xb, yb, gb) in zip(a, b):
        assert xa.dtype == xb.dtype and ya.dtype == yb.dtype
        assert np.array_equal(xa, xb)
        assert np.array_equal(ya, yb, equal_nan=ya.dtype.kind == "f")
        assert [[str(c) for c in g] for g in ga] == [[str(c) for c in g] for g in gb]


def _compara(seg, captura, feature, sample=None, **kw):
    """Antigo × novo numa variável: bins, kind, (x, y, grupos) do fit e avisos."""
    old, fo, wo = _roda(captura, lambda: _old_resolve(seg, feature, sample=sample, **kw))
    new, fn, wn = _roda(captura, lambda: seg._resolve_bins_uncached(feature, sample=sample, **kw))
    assert new == old, (feature, sample)
    _mesmos_fits(fo, fn)
    assert wn == wo
    return new, fn


# ───────────────────────── base ─────────────────────────
class _Um:
    """Objeto que vira o texto "1" (hash por identidade: cada um é um nível)."""

    def __str__(self):
        return "1"


def _base(n=8000, seed=3, nan_inicio=0):
    rng = np.random.default_rng(seed)
    amostra = np.where(rng.random(n) < .7, "DES", "OOT")
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(z - 1)))).astype(float)
    y[rng.random(n) < .05] = np.nan                       # alvo com NaN na DES
    df = pd.DataFrame({"amostra": amostra, "target": y})
    # níveis ligados ao risco (z), para o optbinning formar grupos de fato
    nivel = np.clip(((z + 3) * 1.6).astype(int), 0, 9)
    letras = np.array(list("abcdefghij"), dtype=object)
    base = letras[nivel]
    nulos = np.array([None, np.nan, pd.NA], dtype=object)
    obj = base.copy()
    r = rng.random(n)
    obj[r < .06] = nulos[rng.integers(0, 3, n)][r < .06]
    df["c_obj_na"] = obj                                    # None / NaN / pd.NA
    lit = base.copy()
    lit[rng.random(n) < .08] = "nan"
    lit[rng.random(n) < .05] = "None"
    lit[rng.random(n) < .04] = None
    df["c_literais"] = lit                                  # textos "nan"/"None"
    df["c_string"] = pd.array(obj, dtype="string")
    df["c_string_pa"] = pd.array(obj, dtype="string[pyarrow]")
    try:
        import pyarrow as pa
        df["c_arrow"] = pd.array(lit.tolist(), dtype=pd.ArrowDtype(pa.string()))
    except ImportError:                                     # pragma: no cover
        pass
    df["c_categ"] = pd.Categorical(obj)
    df["c_categ_int"] = pd.Categorical(np.where(r < .05, np.nan, nivel.astype(float)))
    dec = np.array([D("1.0"), D("1.00"), D("2.5"), D("2.50"), D("3"), D("4.000")],
                   dtype=object)[np.clip(nivel // 2, 0, 5)]
    dec = dec.copy()
    dec[rng.random(n) < .05] = None
    df["c_decimal"] = dec                                   # 1.0/1.00 fundem no hash
    misto = np.array([1, 1.0, True, "1", "2", 2.0, D("3"), "x"], dtype=object)
    df["c_misto"] = misto[np.clip(nivel - 1, 0, 7)]         # 1/1.0/True/'1'
    df["c_bytes"] = np.array([b"a", b"b", "c", b"d", "e"], dtype=object)[nivel % 5]
    df["c_alta"] = np.array([f"k{i}" for i in range(2000)], dtype=object)[
        (nivel * 200 + rng.integers(0, 200, n)) % 2000]
    df["c_const"] = "z"
    df["c_const_na"] = np.where(rng.random(n) < .3, None, "z")
    df["c_datas"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(nivel, unit="D")
    df["c_bool"] = rng.random(n) < .3
    df["c_flag_txt"] = np.where(rng.random(n) < .1, "s", "n")
    df["c_3tarde"] = np.where(np.arange(n) < n // 2, "a", rng.choice(list("abc"), n))
    df["c_obs_so_alvo_nan"] = np.where(np.isnan(y), base, None)
    df["c_toda_na"] = pd.Series([None] * n, dtype=object)
    df["c_na_so_oot"] = np.where((amostra == "OOT") & (r < .1), None, base)
    um = [_Um() for _ in range(4)]
    df["c_texto_1"] = np.array(um + ["1", 1], dtype=object)[rng.integers(0, 6, n)]
    if nan_inicio:
        for nome, niveis in (("c_ini2", ["p", "q"]), ("c_ini3", list("pqr")),
                             ("c_ini_multi", list("abcdefgh"))):
            v = np.array(niveis, dtype=object)[rng.integers(0, len(niveis), n)]
            v[:nan_inicio] = None
            df[nome] = v
    return df


def _seg(df, **kw):
    kw.setdefault("sample_col", "amostra")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ModelSegmenter(df, target="target", verbose=False, **kw)


CATS = ["c_obj_na", "c_literais", "c_string", "c_string_pa", "c_arrow", "c_categ",
        "c_categ_int", "c_decimal", "c_misto", "c_bytes", "c_alta", "c_const",
        "c_const_na", "c_datas", "c_bool", "c_flag_txt", "c_3tarde", "c_obs_so_alvo_nan",
        "c_toda_na", "c_texto_1", "c_na_so_oot"]


@pytest.fixture()
def cap_pequeno(monkeypatch):
    monkeypatch.setattr(_common, "OPTBINNING_MAX_ROWS", 2000)   # DES (~5.3k) acima do teto


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_categoricas_iguais_ao_antigo(captura, cap_pequeno, task):
    df = _base()
    if task == "regression":
        df["target"] = np.where(df["target"].isna(), np.nan,
                                np.random.default_rng(1).random(len(df)))
    seg = _seg(df, task_type=task)
    feitos = {}
    for f in [c for c in CATS if c in df.columns]:
        for sample in (None, "OOT"):
            (bins, _kind), fits = _compara(seg, captura, f, sample=sample)
            feitos[(f, sample)] = (bins, fits)
    # o optbinning rodou de fato, no teto, e formou grupos nas colunas de texto
    for f in ("c_obj_na", "c_literais", "c_string", "c_string_pa", "c_categ", "c_decimal",
              "c_misto", "c_alta"):
        bins, fits = feitos[(f, None)]
        assert len(fits) == 1 and len(fits[0][0]) == 2000, f
        assert sum(b["kind"] == "cat" for b in bins) >= 2, f
    assert feitos[("c_const", None)][0] == []
    # faltante só fora da DES: sem faixa 'na' na DES, com ela na OOT
    assert [b["kind"] for b in feitos[("c_na_so_oot", None)][0]].count("na") == 0
    assert [b["kind"] for b in feitos[("c_na_so_oot", "OOT")][0]].count("na") == 1
    # textos literais "nan"/"None" não são faltantes, mas saem dos cats (regra antiga)
    assert not any(c in ("nan", "None") for b in feitos[("c_literais", None)][0]
                   for c in b.get("cats", ()))


def test_caminho_rapido_nao_copia_o_recorte(cap_pequeno, monkeypatch):
    seg = _seg(_base())
    chamadas = []
    orig = ModelSegmenter._frame
    monkeypatch.setattr(ModelSegmenter, "_frame",
                        lambda self, *a, **k: chamadas.append(a) or orig(self, *a, **k))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in ("c_obj_na", "c_literais", "c_string", "c_string_pa", "c_arrow", "c_categ",
                  "c_decimal", "c_misto", "c_bytes", "c_alta", "c_3tarde", "c_texto_1"):
            seg._resolve_bins_uncached(f)
        assert chamadas == []
        # fora do caminho fatorado: binária (e constante, que é candidata), category
        # de números e datetime
        for f in ("c_flag_txt", "c_const", "c_categ_int", "c_datas"):
            seg._resolve_bins_uncached(f)
        assert len(chamadas) == 4


def test_fatoracao_uma_vez_e_texto_so_nas_sorteadas(cap_pequeno, monkeypatch):
    seg = _seg(_base())
    fatora, astype_n = [], []
    orig_f = pd.factorize
    monkeypatch.setattr(segmod.pd, "factorize",
                        lambda v, *a, **k: fatora.append(len(v)) or orig_f(v, *a, **k))
    orig_a = pd.Series.astype

    def astype(self, dtype, *a, **k):
        if dtype is str:
            astype_n.append(len(self))
        return orig_a(self, dtype, *a, **k)
    monkeypatch.setattr(pd.Series, "astype", astype)
    n_des = int(seg._frame_mask().sum())                 # hash só nas linhas da DES
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg._resolve_bins_uncached("c_obj_na")
        assert fatora == [n_des] and astype_n == []
        fatora.clear()
        seg._resolve_bins_uncached("c_decimal")          # Decimal: str só nas 2000
        assert fatora == [n_des] and astype_n == [2000]
        fatora.clear()
        seg._resolve_bins_uncached("c_obj_na", sample="OOT")
        assert fatora == [len(seg.df) - n_des]


def _resultado(fn):
    """``("ok", valor)`` ou ``("erro", tipo da exceção)``."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return "ok", fn()
    except Exception as e:                                  # noqa: BLE001
        return "erro", type(e)


def test_niveis_nao_hasheaveis_fora_do_prefixo(captura, cap_pequeno, monkeypatch):
    """list/dict fora dos 10 mil primeiros observados da DES ou fora da amostra
    pedida: o recorte antigo só fazia hash do prefixo e seguia com o astype(str)
    — o caminho fatorado não pode passar a levantar ``TypeError``."""
    n = 30_000
    rng = np.random.default_rng(5)
    amostra = np.where(rng.random(n) < .7, "DES", "OOT")
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(z - 1)))).astype(float)
    base = np.array(list("abcdefghij"), dtype=object)[np.clip(((z + 3) * 1.6).astype(int), 0, 9)]
    listas = base.copy()
    oot = np.flatnonzero(amostra == "OOT")
    for i in oot[::50]:
        listas[i] = [1, 2]                                  # listas só na OOT
    dicts = base.copy()
    des = np.flatnonzero(amostra == "DES")
    for i in des[12_000::40]:
        dicts[i] = {"k": int(i % 3)}                        # depois do 10.000º da DES
    df = pd.DataFrame({"amostra": amostra, "target": y, "c_listas": listas,
                       "c_dicts": dicts})
    novo, velho = _seg(df.copy()), _seg(df.copy())
    monkeypatch.setattr(velho, "_resolve_bins_uncached",
                        lambda *a, **k: _old_resolve(velho, *a, **k))
    for f in ("c_listas", "c_dicts"):
        for sample in (None, "OOT"):
            a = _resultado(lambda: _old_resolve(velho, f, sample=sample))
            b = _resultado(lambda: novo._resolve_bins_uncached(f, sample=sample))
            assert a[0] == b[0] and a[1] == b[1], (f, sample, a, b)
            if a[0] == "ok":
                _compara(novo, captura, f, sample=sample)
    # o que funcionava antes continua funcionando (antes da correção: TypeError)
    assert _resultado(lambda: novo._resolve_bins_uncached("c_listas"))[0] == "ok"
    assert _resultado(lambda: novo._resolve_bins_uncached("c_dicts"))[0] == "ok"
    assert _resultado(lambda: novo._resolve_bins_uncached("c_dicts", sample="OOT"))[0] == "ok"
    for f in ("c_listas", "c_dicts"):
        a = _resultado(lambda: velho.variable_table(f, sample="DES"))
        b = _resultado(lambda: novo.variable_table(f, sample="DES"))
        # (dict na DES quebra adiante, fora do binning, igual nos dois caminhos)
        assert a[0] == b[0] and (a[0] == "ok" or a[1] == b[1]), (f, a, b)
        if a[0] == "ok":
            pd.testing.assert_frame_equal(a[1], b[1], check_exact=True)
    assert _resultado(lambda: novo.variable_table("c_listas", sample="DES"))[0] == "ok"


def test_sem_sample_col(captura, cap_pequeno):
    seg = _seg(_base().drop(columns="amostra"), sample_col=None)
    for f in CATS:
        _compara(seg, captura, f)


def test_nan_no_comeco_seguido_de_2_e_3_niveis(captura, cap_pequeno):
    seg = _seg(_base(n=90_000, nan_inicio=60_000))
    for f in ("c_ini2", "c_ini3", "c_ini_multi"):
        for sample in (None, "OOT"):
            _compara(seg, captura, f, sample=sample)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bins = seg._resolve_bins_uncached("c_ini2")[0]
    assert [b.get("cats") for b in bins][:2] == [["p"], ["q"]]          # binária


def test_amostra_constante_e_base_variando(captura, cap_pequeno):
    """Misto: o texto das 2000 sorteadas é todo "1", mas uma linha fora do sorteio
    vale "2" — a constância tem de olhar todas as linhas de ajuste."""
    df = _base()
    seg0 = _seg(df.copy())
    m = seg0._frame_mask() & df["target"].notna().to_numpy()
    lin = np.flatnonzero(m)
    idx = _common._amostra_optbinning(len(lin))
    fora = np.setdiff1d(lin, lin[idx])
    v = df["c_texto_1"].to_numpy(dtype=object).copy()
    v[fora[len(fora) // 2]] = "2"
    df["c_texto_2"] = v
    seg = _seg(df)
    _, fits = _compara(seg, captura, "c_texto_2")
    assert len(fits) == 1 and set(fits[0][0]) == {"1"}   # o ajuste rodou (base varia)
    _, fits = _compara(seg, captura, "c_texto_1")
    assert fits == []                                       # constante: sem ajuste


def test_bins_manuais_e_derivada_seguem_iguais(captura, cap_pequeno):
    seg = _seg(_base())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.set_manual_bins("c_obj_na", [["a", "b"], ["c", "d", "e"]], missing="pior")
        nova = seg.create_categorical("c_alta")
    for f in ("c_obj_na", nova, "c_literais"):
        for sample in (None, "OOT"):
            _compara(seg, captura, f, sample=sample)


def test_teto_alterado(captura, monkeypatch):
    seg = _seg(_base())
    for cap in (2000, 3500, 0, 2000):
        monkeypatch.setattr(_common, "OPTBINNING_MAX_ROWS", cap)
        for f in ("c_obj_na", "c_decimal"):
            _, fits = _compara(seg, captura, f)
            n_ok = int((seg._frame_mask() & seg.df[f].notna().to_numpy()
                        & seg.df["target"].notna().to_numpy()).sum())
            assert len(fits[0][0]) == (cap or n_ok)


def test_des_acima_do_teto_real(captura):
    # teto padrão (300k): base de 320k linhas, toda DES (sem passar muito de 300k
    # linhas), só duas categóricas — ~304k e 320k linhas de ajuste
    n = 320_000
    rng = np.random.default_rng(11)
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(z - 1)))).astype(float)
    nivel = np.clip(((z + 3) * 1.6).astype(int), 0, 9)
    c = np.array(list("abcdefghij"), dtype=object)[nivel]
    c[rng.random(n) < .05] = None
    dec = np.array([D("1.0"), D("1.00"), D("2.5"), D("2.50"), D("3")], dtype=object)[nivel // 2]
    df = pd.DataFrame({"amostra": "DES", "target": y, "c": c, "dec": dec})
    seg = _seg(df)
    for f in ("c", "dec"):
        _, fits = _compara(seg, captura, f)
        assert len(fits[0][0]) == _common.OPTBINNING_MAX_ROWS


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_ranking_inteiro_igual(cap_pequeno, monkeypatch, task):
    df = _base()
    if task == "regression":
        df["target"] = np.where(df["target"].isna(), np.nan,
                                np.random.default_rng(1).random(len(df)))
    df = df.drop(columns=["c_texto_1"])          # objetos sem igualdade de valor
    novo = _seg(df.copy(), task_type=task)
    velho = _seg(df.copy(), task_type=task)
    monkeypatch.setattr(velho, "_resolve_bins_uncached",
                        lambda *a, **k: _old_resolve(velho, *a, **k))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b = velho.variable_iv(), novo.variable_iv()
        for f in ("c_obj_na", "c_decimal", "c_literais"):
            for s in ("DES", "OOT"):
                pd.testing.assert_frame_equal(velho.variable_table(f, sample=s),
                                              novo.variable_table(f, sample=s),
                                              check_exact=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert velho._bins_cache.keys() == novo._bins_cache.keys()
    assert all(velho._bins_cache[k] == novo._bins_cache[k] for k in velho._bins_cache)
