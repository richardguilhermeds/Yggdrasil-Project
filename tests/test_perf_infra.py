"""Infraestrutura compartilhada das otimizações do ModelSegmenter (sem mudança
de comportamento): ``_particao_exata`` (quando contar por código = somar as
máscaras), ``_amostra_optbinning`` (mesmo sorteio do ``fit_optbinning_splits``),
registro dos caches por linha, ``_col_version`` e ``_fatias_por_safra(amostrar=)``."""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk import _common
from yggdrasil.credit_risk.model import ModelSegmenter
from yggdrasil.credit_risk.model.segmenter import (
    _bin_codes, _bin_masks, _particao_exata)

INF = np.inf
NUM3 = [{"kind": "num", "lo": -INF, "hi": 1.0}, {"kind": "num", "lo": 1.0, "hi": 5.0},
        {"kind": "num", "lo": 5.0, "hi": INF}]
NA = {"kind": "na"}


def _codigos_iguais_mascaras(series, bins) -> bool:
    """Contar por código (``_bin_codes`` + bincount) == somar as máscaras, e
    nenhuma linha em duas máscaras."""
    masks = _bin_masks(series, bins)
    soma = np.sum(np.vstack(masks).astype(int), axis=0) if masks else np.zeros(len(series))
    codes = _bin_codes(series, bins)
    cont = np.bincount(codes[codes >= 0], minlength=len(bins))
    return (bool((soma <= 1).all())
            and [int(m.sum()) for m in masks] == cont.tolist()
            and all(np.array_equal(codes == i, m) for i, m in enumerate(masks)))


# ───────────────────────── _particao_exata ─────────────────────────
def test_numerica_contigua_numpy_e_exata():
    s = pd.Series([-3.0, 0.5, 1.0, 2.0, 5.0, 9.0, INF, np.nan])
    for bins in (NUM3, NUM3 + [NA], [dict(NUM3[0], include_na=True)] + NUM3[1:]):
        assert _particao_exata(s, bins)
        assert _codigos_iguais_mascaras(s, bins)


def test_menos_inf_na_coluna_cai_no_fallback():
    s = pd.Series([-INF, 0.0, 3.0, np.nan])
    assert not _particao_exata(s, NUM3 + [NA])
    # o furo que a checagem cobre: searchsorted põe -inf na faixa 0, a máscara não
    assert not _codigos_iguais_mascaras(s, NUM3 + [NA])


def test_faixas_com_buraco_ou_sobrepostas_caem_no_fallback():
    s = pd.Series([0.0, 1.5, 3.0])
    buraco = [{"kind": "num", "lo": -INF, "hi": 1.0}, {"kind": "num", "lo": 2.0, "hi": INF}]
    sobre = [{"kind": "num", "lo": -INF, "hi": 2.0}, {"kind": "num", "lo": 1.0, "hi": INF}]
    assert not _particao_exata(s, buraco)
    assert not _particao_exata(s, sobre)
    assert not _codigos_iguais_mascaras(s, sobre)
    # fora de ordem, mas contíguas: continua exata
    assert _particao_exata(s, NUM3[::-1])


def test_grupos_categoricos_sobrepostos_caem_no_fallback():
    s = pd.Series(["a", "b", "c", "d", None], dtype=object)
    disj = [{"kind": "cat", "cats": ["a", "b", "b"]}, {"kind": "cat", "cats": ["c"]}, NA]
    sobre = [{"kind": "cat", "cats": ["a", "b"]}, {"kind": "cat", "cats": ["b", "c"]}]
    assert _particao_exata(s, disj) and _codigos_iguais_mascaras(s, disj)
    assert not _particao_exata(s, sobre)
    assert not _codigos_iguais_mascaras(s, sobre)       # 'b' contado duas vezes
    # cats da faixa de faltante (derivada) também entram na disjunção
    na_txt = [{"kind": "cat", "cats": ["a"]}, {"kind": "na", "cats": ["(faltante)"]}]
    na_sob = [{"kind": "cat", "cats": ["a", "(faltante)"]},
              {"kind": "na", "cats": ["(faltante)"]}]
    s2 = pd.Series(["a", "(faltante)", None], dtype=object)
    assert _particao_exata(s2, na_txt) and _codigos_iguais_mascaras(s2, na_txt)
    assert not _particao_exata(s2, na_sob)


def test_include_na_duplo_cai_no_fallback():
    s = pd.Series(["a", "b", None], dtype=object)
    dois = [{"kind": "cat", "cats": ["a"], "include_na": True},
            {"kind": "cat", "cats": ["b"], "include_na": True}]
    com_na = [{"kind": "cat", "cats": ["a"], "include_na": True},
              {"kind": "cat", "cats": ["b"]}, NA]
    assert not _particao_exata(s, dois)
    assert not _particao_exata(s, com_na)
    assert not _codigos_iguais_mascaras(s, dois)
    num = [dict(NUM3[0], include_na=True)] + NUM3[1:] + [NA]
    assert not _particao_exata(pd.Series([0.0, np.nan]), num)


def test_nullable():
    s = pd.Series([0.5, None, 3.0, 7.0], dtype="Float64")
    assert _particao_exata(s, NUM3 + [NA]) and _codigos_iguais_mascaras(s, NUM3 + [NA])
    si = pd.Series([0, None, 3, 7], dtype="Int64")
    assert _particao_exata(si, NUM3 + [NA]) and _codigos_iguais_mascaras(si, NUM3 + [NA])
    # NaN NÃO mascarado (0/0) no Float64: isna != isnan ⇒ fallback
    z = pd.Series([0.0, 1.0, None], dtype="Float64")
    z = z / pd.Series([0.0, 1.0, 1.0], dtype="Float64")
    difere = not np.array_equal(z.isna().to_numpy(), np.isnan(
        z.to_numpy(dtype="float64", na_value=np.nan)))
    assert _particao_exata(z, NUM3 + [NA]) is (not difere)
    if difere:
        assert not _codigos_iguais_mascaras(z, NUM3 + [NA])


def test_pyarrow():
    pa = pytest.importorskip("pyarrow")
    so_null = pd.Series(pd.array([0.5, None, 3.0], dtype="float64[pyarrow]"))
    assert _particao_exata(so_null, NUM3 + [NA])
    assert _codigos_iguais_mascaras(so_null, NUM3 + [NA])
    arr = pa.array([0.5, float("nan"), None, 3.0], from_pandas=False)
    nan_null = pd.Series(pd.arrays.ArrowExtensionArray(arr))
    assert nan_null.isna().tolist() == [False, False, True, False]
    assert not _particao_exata(nan_null, NUM3 + [NA])
    assert not _codigos_iguais_mascaras(nan_null, NUM3 + [NA])


def test_casos_degenerados():
    s = pd.Series([1.0, 2.0])
    assert not _particao_exata(s, [])
    assert not _particao_exata(s, NUM3 + [{"kind": "cat", "cats": ["1.0"]}])
    assert not _particao_exata(s, [{"kind": "num", "lo": np.nan, "hi": INF}])
    assert not _particao_exata(s, [{"kind": "outro"}])
    assert not _particao_exata(pd.Series(["x", "y"], dtype=object), NUM3)


# ───────────────────────── _amostra_optbinning ─────────────────────────
class _Captura:
    name = "cap"

    def __init__(self):
        self.x = self.y = None
        self.splits = [0.5]

    def fit(self, x, y):
        self.x, self.y = np.asarray(x), np.asarray(y)


def _sorteio_antigo(n, cap):
    return np.sort(np.random.default_rng(_common.OPTBINNING_SEED).choice(n, cap,
                                                                         replace=False))


def test_amostra_optbinning_mesmo_sorteio():
    assert _common._amostra_optbinning(100, 100) is None
    assert _common._amostra_optbinning(100, 0) is None
    assert np.array_equal(_common._amostra_optbinning(1000, 70), _sorteio_antigo(1000, 70))


def test_amostra_optbinning_le_teto_na_chamada(monkeypatch):
    assert _common._amostra_optbinning(1000) is None
    monkeypatch.setattr(_common, "OPTBINNING_MAX_ROWS", 50)
    assert np.array_equal(_common._amostra_optbinning(1000), _sorteio_antigo(1000, 50))


def test_fit_optbinning_splits_idx_retrocompativel(monkeypatch):
    monkeypatch.setattr(_common, "OPTBINNING_MAX_ROWS", 40)
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=300), (rng.random(300) < .3).astype(int)
    a, b, c = _Captura(), _Captura(), _Captura()
    assert _common.fit_optbinning_splits(a, x, y) == [0.5]          # sem idx: como antes
    ref = _sorteio_antigo(300, 40)
    assert np.array_equal(a.x, x[ref]) and np.array_equal(a.y, y[ref])
    _common.fit_optbinning_splits(b, x, y, idx=_common._amostra_optbinning(len(x)))
    assert np.array_equal(b.x, a.x) and np.array_equal(b.y, a.y)
    _common.fit_optbinning_splits(c, x[:30], y[:30])                # abaixo do teto
    assert np.array_equal(c.x, x[:30])


# ───────────────────────── caches por linha / _col_version ─────────────────────────
def _df(n=3000, seed=5):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x1[rng.random(n) < .05] = np.nan
    meses = pd.date_range("2023-01-01", periods=8, freq="MS")
    df = pd.DataFrame({"x1": x1, "c": rng.choice(list("ABCD"), n).astype(object),
                       "dt_ref": rng.choice(meses, n)})
    df["target"] = (rng.random(n) < 1 / (1 + np.exp(-np.nan_to_num(x1)))).astype(float)
    df["amostra"] = np.where(df["dt_ref"] >= meses[6], "OOT", "DES")
    return df


@pytest.fixture()
def seg():
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ModelSegmenter(_df(), target="target", sample_col="amostra",
                              date_col="dt_ref", verbose=False)


def test_caches_por_linha_nascem_no_init_e_zeram_no_detached(seg):
    for nome, fab in ModelSegmenter._ROW_CACHES:
        assert nome in seg.__dict__
        assert seg.__dict__[nome] == ({} if fab is dict else None)
    # toda coluna crua nasce com versão própria
    assert set(seg._col_version) == set(seg.df.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.variable_iv()
        seg._fatias_por_safra("dt_ref")
    assert seg._iv_row_cache and seg._fatias_cache and seg._safra_cache
    s = seg._detached_scorer()
    for nome, fab in ModelSegmenter._ROW_CACHES:
        assert s.__dict__[nome] == ({} if fab is dict else None)
    assert seg._iv_row_cache                      # o original não é tocado


def test_invalidate_bins_poda_so_a_variavel(seg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.variable_iv()
    seg._bincode_cache[("x1", "b")] = np.zeros(1)
    seg._bincode_cache[("c", "b")] = np.zeros(1)
    feats = {k[0] for k in seg._iv_row_cache}
    assert {"x1", "c"} <= feats
    seg._invalidate_bins("x1")
    assert {k[0] for k in seg._iv_row_cache} == feats - {"x1"}
    assert list(seg._bincode_cache) == [("c", "b")]
    assert seg._safra_cache is not None and "dt_ref" not in feats


def test_col_version_nas_derivadas(seg):
    v = seg._col_version
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = seg.create_categorical("x1", dummies=False)
        v1 = v[n]
        seg._bincode_cache[(n, "b")] = np.zeros(1)
        seg.remove_derived(n)
        assert v[n] > v1 and n not in seg.df.columns
        assert (n, "b") not in seg._bincode_cache
        seg.set_manual_bins("x1", [0.0])
        ds = seg.create_scorecard_dummies("x1")
        a = {d: v[d] for d in ds}
        assert len(set(a.values())) == len(ds)          # nenhuma versão repetida
        seg.df.drop(columns=ds[0], inplace=True)       # como o load com df cru
        seg._rebuild_derived()
        assert v[ds[0]] > a[ds[0]] and v[ds[-1]] == a[ds[-1]]
        b = {d: v[d] for d in ds}
        removidas = seg.clear_derived()
    assert set(removidas) == set(ds)
    assert all(v[d] > b[d] for d in ds)


def test_drop_derived_column_e_load_zera(seg, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = seg.create_categorical("c", dummies=False)
        v1 = seg._col_version[n]
        seg._drop_derived_column(n)
        assert n not in seg.df.columns and seg._col_version[n] > v1
        v2 = seg._col_version[n]
        seg._drop_derived_column(n)                   # ausente: só marca
        assert seg._col_version[n] > v2
        seg.variable_iv()
        seg.fit(features=["x1", "c"])
        p = str(tmp_path / "m.json")
        seg.save(p)
        assert seg._iv_row_cache
        seg.load(p)
    assert seg._iv_row_cache == {} and seg._bincode_cache == {}
    assert seg._raw_score_cache is None or seg._raw_score_cache[0] is seg.model


def test_col_version_nunca_repete_entre_loads(seg, tmp_path):
    """Memo que sobrevive ao load (UI) nunca casa por acaso: derivada recriada
    e coluna crua ganham versão nova com outra base — e com a mesma também."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = seg.create_categorical("x1", dummies=False)
        antes = {c: seg._col_key(c) for c in ("x1", "c", n)}
        p = str(tmp_path / "m.json")
        seg.save(p)
        seg.load(p, df=_df(seed=9))
        depois = {c: seg._col_key(c) for c in ("x1", "c", n)}
        assert n in seg.df.columns
        assert all(depois[c] != antes[c] for c in antes)
        assert depois[n] == seg._col_version.get(n)      # .get direto também serve
        # outra instância sobre a mesma base: versões próprias, sem colisão
        outro = ModelSegmenter(_df(), target="target", sample_col="amostra",
                               date_col="dt_ref", verbose=False)
    usadas = set(antes.values()) | set(depois.values())
    assert not usadas & set(outro._col_version.values())
    # coluna sem registro (posta no df por fora) ganha versão na 1ª leitura
    seg.df["extra"] = 1.0
    k = seg._col_key("extra")
    assert k is not None and seg._col_key("extra") == k and k not in usadas


def test_fatias_por_safra_amostrar(seg):
    seg.max_linhas_graficos = 1000
    fora = seg._fatias_por_safra("dt_ref", "DES")
    with seg._amostra_graficos():
        dentro = seg._fatias_por_safra("dt_ref", "DES")
        sem = seg._fatias_por_safra("dt_ref", "DES", amostrar=False)
    assert len(dentro[0]) < len(fora[0])
    assert all(np.array_equal(a, b) for a, b in zip(sem[:2], fora[:2]))
    assert sem[2] == fora[2]
    # amostragem efetiva num campo só: o recorte inteiro (fora do contexto ou
    # amostrar=False) é UMA entrada; a amostrada é chaveada por (cap, n, seed)
    assert sorted(map(repr, {k[-1] for k in seg._fatias_cache})) == sorted(
        [repr(None), repr((1000, len(seg.df), seg.random_state))])
    assert len(seg._fatias_cache) == 2
    assert seg._fatias_por_safra("dt_ref", "DES", amostrar=False) is fora


def test_fatias_e_mascara_mudam_com_teto_e_semente(seg):
    seg.max_linhas_graficos = 1000
    with seg._amostra_graficos():
        a = seg._fatias_por_safra("dt_ref", "DES")
        m_a = seg._mascara_amostra_graficos().copy()
        seg.max_linhas_graficos = 1500
        b = seg._fatias_por_safra("dt_ref", "DES")
        seg.random_state = 7
        m_c = seg._mascara_amostra_graficos()
        c = seg._fatias_por_safra("dt_ref", "DES")
    assert m_a.sum() == 1000 and m_c.sum() == 1500
    assert len(b[0]) > len(a[0])                    # teto novo: não reaproveita
    assert not np.array_equal(b[0], c[0])            # semente nova: não reaproveita
    ref = np.zeros(len(seg.df), dtype=bool)
    ref[np.random.default_rng(7).choice(len(seg.df), 1500, replace=False)] = True
    assert np.array_equal(m_c, ref)
