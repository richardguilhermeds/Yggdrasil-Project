"""Onda 3 — correções da revisão:

* 3B: ``_safra_codes`` com coluna objeto não hashável (list/dict) ou com
  escalares que o hash funde mas o ``to_datetime`` converte diferente
  (1/1.0/True/Decimal(1), 0/False, o mesmo instante em fusos distintos) cai no
  caminho antigo — igualdade exata (ou o mesmo erro) com a versão anterior.
* 3A: o gráfico "faixas do optimal binning" só LÊ o ``_bins_cache`` (hit
  puro); no miss não grava lá — quem grava é o ranking."""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optbinning")

from yggdrasil.credit_risk.model import ModelSegmenter  # noqa: E402


# ───────────────────────── 3B ─────────────────────────
def _old_safra_codes(col):
    per = pd.to_datetime(col, errors="coerce").dt.to_period("M")
    codes, uniq = pd.factorize(per, sort=True, use_na_sentinel=True)
    return np.asarray(codes, dtype=np.int32), [str(u) for u in uniq]


_UTC = dt.timezone.utc
_BRT = dt.timezone(dt.timedelta(hours=-3))

CASOS = {
    "lista": [[1], "2024-01-01"],
    "dict": ["2024-01-01", {"a": 1}, "2024-02-01"],
    "num_mistos": [1, 1.0, True, Decimal(1)],
    "zero_false": [0, False, 20240101],
    "np_bool": [np.bool_(True), "2024-01-01", 1],
    "decimal": [Decimal("20240101"), "2024-03-01"],
    "float_str": [1.5, "2024-01-01", "2024-02-01"],
    # mesmo instante, fusos distintos: iguais no hash, diferentes na conversão
    "fusos": [dt.datetime(2024, 1, 1, 0, tzinfo=_UTC),
              dt.datetime(2023, 12, 31, 21, tzinfo=_BRT)],
    "ts_fuso": [pd.Timestamp("2024-01-31 23:00", tz="UTC"),
                pd.Timestamp("2024-01-31 20:00", tz="America/Sao_Paulo")],
    "bytes": [b"2024-01-01", "2024-01-01"],
    # tipos aceitos no caminho pelos únicos (devem continuar idênticos)
    "texto": ["2024-01-05", None, "2024-01-05", "abc", "2024-02-10"],
    "datas": [dt.date(2024, 1, 5), None, dt.date(2024, 3, 1), dt.date(2024, 1, 5)],
    "datetime": [dt.datetime(2024, 1, 5, 3), pd.Timestamp("2024-02-01"), None],
    "d64_obj": [np.datetime64("2024-01-05"), pd.Timestamp("2024-01-05"), "2024-03-01"],
    "np_str": [np.str_("2024-01-05"), "2024-01-05", "2024-04-01"],
}


def _obj(df):
    o = ModelSegmenter.__new__(ModelSegmenter)
    o.df = df
    return o


@pytest.mark.parametrize("nome", list(CASOS))
@pytest.mark.parametrize("rep", [1, 300])
def test_safra_codes_objeto_igual_ao_antigo(nome, rep):
    vals = CASOS[nome] * rep
    col = pd.Series(np.array(vals + [None], dtype=object), dtype=object)
    assert col.dtype == object
    o = _obj(pd.DataFrame({"t": col}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ve = _old_safra_codes(col)
        except Exception as e:                          # mesmo erro nos dois caminhos
            with pytest.raises(type(e)):
                o._safra_codes("t")
            return
        no = o._safra_codes("t")
    assert no[0].dtype == np.int32
    assert np.array_equal(no[0], ve[0]), (nome, no, ve)
    assert no[1] == ve[1], nome


def test_safra_codes_casos_da_revisao():
    """Os dois exemplos reproduzidos na revisão, explícitos."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o = _obj(pd.DataFrame({"t": pd.Series([[1], "2024-01-01"], dtype=object)}))
        cod, rot = o._safra_codes("t")
        assert cod.tolist() == [-1, 0] and rot == ["2024-01"]
        o = _obj(pd.DataFrame({"t": pd.Series([1, 1.0, True, Decimal(1)], dtype=object)}))
        assert o._safra_codes("t")[0].tolist() == _old_safra_codes(o.df["t"])[0].tolist()
        o = _obj(pd.DataFrame({"t": pd.Series([0, False, 20240101], dtype=object)}))
        assert o._safra_codes("t")[0].tolist() == _old_safra_codes(o.df["t"])[0].tolist()


def test_safra_codes_fallback_sem_reuso_por_mascara():
    """No caminho antigo (objeto misto) o recorte não é marcado como exato:
    ``_safra_codes_linhas`` devolve None e o chamador converte o recorte."""
    o = _obj(pd.DataFrame({"t": pd.Series([dt.date(2024, 1, 1), 1, True] * 5,
                                          dtype=object)}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o._safra_codes("t")
    assert o._safra_cache[("t", "recorte_exato")] is False
    assert o._safra_codes_linhas("t") is None
    o = _obj(pd.DataFrame({"t": pd.Series([dt.date(2024, 1, 1), None,
                                           dt.date(2024, 2, 3)] * 5, dtype=object)}))
    o._safra_codes("t")
    assert o._safra_cache[("t", "recorte_exato")] is True


# ───────────────────────── 3A ─────────────────────────
def _base_raro(n=6000, seed=11):
    rng = np.random.default_rng(seed)
    xc = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-xc))).astype(float)
    df = pd.DataFrame({"amostra": np.where(rng.random(n) < .7, "DES", "OOT"),
                       "target": y, "x_cont": xc})
    flag = np.zeros(n)
    flag[rng.choice(n, int(n * .0035), replace=False)] = 1.0     # nível raro (0,35%)
    df["x_raro"] = flag
    return df


def _seg(df):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ModelSegmenter(df, target="target", verbose=False, sample_col="amostra")


def test_optbin_no_miss_nao_grava_no_bins_cache_e_ranking_grava():
    seg = _seg(_base_raro())
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in ("x_raro", "x_cont"):
            seg._optbin_numeric_bins(f, seg.ref_sample)
            seg._optbin_numeric_bins(f)
    assert not any(k[0] in ("x_raro", "x_cont") for k in seg._bins_cache)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.variable_iv()
    assert {"x_raro", "x_cont"} <= {k[0] for k in seg._bins_cache}
    # depois do ranking: hit puro do _bins_cache, mesmo resultado
    seg._optbin_cache.clear()
    n_cache = len(seg._bins_cache)
    for f in ("x_raro", "x_cont"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = seg._optbin_numeric_bins(f, seg.ref_sample)
        bins, kind = seg._resolve_bins(f, 5, 0.05, None, seg.ref_sample)
        assert repr(b) == repr(bins if kind == "num" else [])
    assert len(seg._bins_cache) == n_cache


def test_optbin_miss_igual_ao_hit():
    """Resultado do recorte (miss) == o do _bins_cache (hit) da mesma chave."""
    df = _base_raro()
    a, b = _seg(df.copy()), _seg(df.copy())
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b.variable_iv()
        for f in ("x_raro", "x_cont"):
            for s in (None, "DES", "OOT"):
                assert repr(a._optbin_numeric_bins(f, s)) == repr(b._optbin_numeric_bins(f, s))
