"""Onda 6A (plano_model): cópia RASA do df só com Copy-on-Write garantido.

``_common.copia_df`` substitui o ``df.copy()`` profundo no ``__init__`` do
ModelSegmenter, em ``assign()``, em ``_score_pandas`` e em ``_apply_derived``.
Sob CoW (pandas >= 3, ou ``mode.copy_on_write is True`` no 2.x) a cópia é rasa
— consolidada como a profunda e só quando cada bloco compartilhado já tem o
layout da cópia —; em qualquer outro caso, a cópia profunda de sempre. A lib
nunca liga o CoW.

Testes: (i) saídas idênticas (check_exact) entre profunda e rasa; (ii)
``np.shares_memory`` só no ramo com CoW; (iii) isolamento nos dois sentidos;
(iv) ``'warn'``/False/pandas 1.5 sem compartilhar; (v) guardas de layout
(fragmentada, F-order, só-leitura) caindo na profunda; (vi) memo de
``_samples()`` semeado com o mesmo tipo de elemento.
"""
from __future__ import annotations

import contextlib
import decimal
import io
import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest

import yggdrasil.credit_risk._common as common_mod
from yggdrasil.credit_risk._common import copia_df, copy_on_write_garantido
from yggdrasil.credit_risk.model import ModelSegmenter

PANDAS3 = int(pd.__version__.split(".")[0]) >= 3


@contextlib.contextmanager
def _cow(valor=True):
    """Liga (ou fixa) o CoW só dentro do bloco — no pandas 3 ele é o único modo."""
    if PANDAS3:
        if valor is not True:
            pytest.skip("pandas >= 3: CoW não pode ser desligado")
        yield
        return
    with pd.option_context("mode.copy_on_write", valor):
        yield


def _base(n=5000, seed=7):
    rng = np.random.default_rng(seed)
    meses = [date(2024, m, 1) for m in range(1, 9)]
    mes = rng.integers(0, len(meses), n)
    amostra = np.where(mes < 5, "DES", "OOT").astype(object)
    num_00 = rng.normal(size=n)
    num_00[rng.random(n) < 0.03] = np.nan
    num_01 = 0.7 * num_00 + 0.5 * rng.normal(size=n)
    num_01[rng.random(n) < 0.02] = np.nan
    num_02 = rng.integers(0, 8, n).astype(float)
    int_na = pd.array(rng.integers(0, 30, n), dtype="Int64")
    int_na[rng.random(n) < 0.05] = pd.NA
    D = decimal.Decimal
    niveis = [D(f"{v:.2f}") for v in np.arange(6) * 0.5]
    dec = np.array([niveis[k] for k in rng.integers(0, 6, n)], dtype=object)
    dec[rng.random(n) < 0.02] = None
    low = ["A", "B", "C", "D"]
    k = rng.integers(0, 4, n)
    cat = np.array([low[i] for i in k], dtype=object)
    cat[rng.random(n) < 0.02] = None
    flag = rng.random(n) < 0.3
    z = np.nan_to_num(num_00) * 0.9 + 0.3 * (k - 1.5) + 0.6 * flag
    target = (rng.random(n) < 1 / (1 + np.exp(-(z - 1.2)))).astype(float)
    return pd.DataFrame({
        "num_00": num_00, "num_01": num_01, "num_02": num_02, "int_na": int_na,
        "dec": dec, "cat": cat, "flag": flag,
        "dt_ref": np.array([meses[i] for i in mes], dtype=object),
        "amostra": amostra, "target": target})


def _seg(df, **kw):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ModelSegmenter(df, target="target", sample_col="amostra", ref_sample="DES",
                              date_col="dt_ref", verbose=False, **kw)


def _compartilha(a: pd.DataFrame, b: pd.DataFrame, col="num_00") -> bool:
    return bool(np.shares_memory(a[col].to_numpy(), b[col].to_numpy()))


def _saidas(df):
    """Fluxo completo: ranking, tabelas/PSI, redundância, derivada, WoE, ratings,
    SQL, escoragem (predict/score_table) e assign."""
    out = {}
    seg = _seg(df)
    out["compartilha_base"] = _compartilha(seg.df, df)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["variable_iv"] = seg.variable_iv()
        amostras = list(seg._samples())
        out["amostras"] = amostras
        for f in list(seg.candidates):
            out[f"table[{f}]"] = seg.variable_table(f)
            out[f"psi[{f}]"] = seg._variable_psi(f, amostras)
        out["corr"] = seg.correlation_report(threshold=0.3, features=list(seg.candidates))
        d = seg.create_categorical("num_01")
        seg.set_manual_bins("num_00", [-0.5, 0.0, 0.8])
        out["variable_iv2"] = seg.variable_iv()
        out[f"table[{d}]"] = seg.variable_table(d)
        out["bins_cache"] = {k: repr(v) for k, v in seg._bins_cache.items()}
        seg.fit(features=["num_00", d, "num_02", "int_na", "dec", "cat", "flag"],
                transform="woe")
        out["coef"] = seg.model_coefficients()
        out["score_"] = np.asarray(seg.score_)
        out["logit_sql"] = seg.logit_sql()
        out["cat_sql"] = seg.categorization_sql()
        seg.build_ratings(n_ratings=5)
        out["rating_"] = seg.rating_
        out["rating_table"] = seg.rating_table()
        out["backtest"] = seg.backtest()
        novos = _base(1500, seed=99).drop(columns=["target"])
        novos.index = novos.index * 3 + 11
        out["predict"] = seg.predict(novos, col_value="valor_previsto")
        out["score_table"] = seg.score_table(novos, col_value="valor_previsto")
        a = seg.assign()
        out["assign"] = a
        out["assign_compartilha"] = _compartilha(a, seg.df)
        out["df_final"] = seg.df
    return out


def _compara(a, b):
    assert a.keys() == b.keys()
    for k in a:
        x, y = a[k], b[k]
        if k.endswith("compartilha") or k == "compartilha_base":
            continue
        if isinstance(x, pd.DataFrame):
            pd.testing.assert_frame_equal(x, y, check_exact=True, obj=k)
            assert x.attrs.keys() == y.attrs.keys(), k
        elif isinstance(x, pd.Series):
            pd.testing.assert_series_equal(x, y, check_exact=True, obj=k)
        elif isinstance(x, np.ndarray):
            assert x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True), k
        elif isinstance(x, list):
            assert [type(v) for v in x] == [type(v) for v in y], k
            assert x == y, k
        else:
            assert x == y, k


# ─────────────────────────── (i) equivalência ───────────────────────────
@pytest.mark.parametrize("origem", ["dict", "parquet"])
def test_saidas_identicas_profunda_x_rasa(monkeypatch, tmp_path, origem):
    with _cow(True):
        df = _base()
        if origem == "parquet":                    # blocos vindos do arrow (toPandas-like)
            pytest.importorskip("pyarrow")
            p = tmp_path / "b.parquet"
            df.to_parquet(p)
            df = pd.read_parquet(p)
        rasa = _saidas(df)
        monkeypatch.setattr(common_mod, "copy_on_write_garantido", lambda: False)
        profunda = _saidas(df)
    assert rasa["compartilha_base"] is True
    assert profunda["compartilha_base"] is False
    assert rasa["assign_compartilha"] is True       # bloco float não fundido com a derivada
    assert profunda["assign_compartilha"] is False
    _compara(rasa, profunda)


def test_sem_cow_continua_profunda():
    if PANDAS3:
        pytest.skip("pandas >= 3: CoW sempre ativo")
    with _cow(False):
        df = _base(800)
        seg = _seg(df)
        assert not _compartilha(seg.df, df)
        assert not _compartilha(seg.assign(), seg.df)


# ─────────────────────────── (ii)/(iv) condição ───────────────────────────
@pytest.mark.parametrize("valor,esperado", [(True, True), (False, False), ("warn", False)])
def test_condicao_da_opcao(valor, esperado):
    if PANDAS3:
        assert copy_on_write_garantido() is True
        return
    with _cow(valor):
        assert copy_on_write_garantido() is esperado
        df = _base(600)
        seg = _seg(df)
        assert _compartilha(seg.df, df) is esperado


@pytest.mark.parametrize("versao,esperado", [("1.5.3", False), ("2.3.3", True),
                                             ("3.0.0", True), ("3.1.0rc0", True),
                                             ("x.y", False)])
def test_condicao_por_versao(monkeypatch, versao, esperado):
    # pandas 1.5 com a opção ligada: CoW experimental ⇒ nunca rasa
    if PANDAS3 and versao.startswith("2."):
        pytest.skip("pandas >= 3: a opção mode.copy_on_write não é a do 2.x")
    with _cow(True) if not PANDAS3 else contextlib.nullcontext():
        monkeypatch.setattr(pd, "__version__", versao)
        assert copy_on_write_garantido() is esperado


def test_opcao_true_nao_basta_sem_using_copy_on_write(monkeypatch):
    if PANDAS3:
        pytest.skip("pandas >= 3")
    import pandas._config as cfg
    with _cow(True):
        monkeypatch.setattr(cfg, "using_copy_on_write", lambda: False)   # ex.: ArrayManager
        assert copy_on_write_garantido() is False


def test_lib_nao_liga_o_cow():
    if PANDAS3:
        pytest.skip("pandas >= 3")
    with _cow(False):
        df = _base(600)
        seg = _seg(df)
        seg.assign()
        assert pd.options.mode.copy_on_write is False


# ─────────────────────────── (iii) isolamento ───────────────────────────
def test_isolamento_nos_dois_sentidos():
    with _cow(True):
        df = _base(1500)
        colunas = list(df.columns)
        antes = df.copy()
        seg = _seg(df)
        assert _compartilha(seg.df, df)
        ref = seg.df.copy()
        # escrita do usuário no original não chega ao segmentador
        df.loc[0, "num_00"] = 999.0
        df["nova"] = 1
        df.loc[1, "cat"] = "ZZZ"
        pd.testing.assert_frame_equal(seg.df, ref, check_exact=True)
        assert "nova" not in seg.df.columns
        # mudanças estruturais do segmentador não chegam ao df do usuário
        df = df.drop(columns=["nova"])
        df.loc[0, "num_00"] = antes.loc[0, "num_00"]
        df.loc[1, "cat"] = antes.loc[1, "cat"]
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = seg.create_categorical("num_01")
            seg.create_categorical("num_02", new_name="num_02_grp")
            assert d in seg.df.columns
            assert list(df.columns) == colunas
            seg.clear_derived()
        assert list(df.columns) == colunas
        pd.testing.assert_frame_equal(df, antes, check_exact=True)
        # saída de assign mutada no lugar não muda seg.df
        out = seg.assign()
        out.fillna({"num_00": 0.0, "num_01": 0.0}, inplace=True)
        out.loc[2, "num_01"] = -5.0
        pd.testing.assert_frame_equal(seg.df[ref.columns], ref, check_exact=True)
        assert seg.df["num_00"].isna().any()


def test_score_table_isolado_do_df_do_usuario():
    with _cow(True):
        df = _base(2000)
        seg = _seg(df)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = seg.create_categorical("num_01")
            seg.fit(features=["num_00", d, "cat"], transform="woe")
            seg.build_ratings(n_ratings=4)
            novos = _base(700, seed=3).drop(columns=["target"])
            antes = novos.copy()
            res = seg.score_table(novos, col_value="valor_previsto")
        assert list(novos.columns) == list(antes.columns)
        res.loc[0, "num_00"] = 123.0
        res.fillna({"num_00": 0.0, "num_01": 0.0}, inplace=True)
        pd.testing.assert_frame_equal(novos, antes, check_exact=True)


# ─────────────────────────── (v) guardas de layout ───────────────────────────
def test_fragmentada_consolida_como_a_profunda():
    with _cow(True):
        df = _base(1000)
        df["extra"] = np.arange(1000.0)                # 2º bloco float ⇒ fragmentada
        rasa = copia_df(df)
        prof = df.copy()
    assert rasa._mgr.is_consolidated()
    estrut = [(type(b.values).__name__, b.mgr_locs.as_array.tolist())
              for b in rasa._mgr.blocks]
    assert estrut == [(type(b.values).__name__, b.mgr_locs.as_array.tolist())
                      for b in prof._mgr.blocks]
    pd.testing.assert_frame_equal(rasa, prof, check_exact=True)
    # o grupo fundido ganha memória nova; os blocos únicos continuam compartilhados
    assert not _compartilha(rasa, df, "num_00")
    assert _compartilha(rasa, df, "dec")


def test_buffer_em_f_order_cai_na_profunda():
    with _cow(True):
        arr = np.random.default_rng(1).normal(size=(500, 3))
        df = pd.DataFrame(arr, columns=["a", "b", "c"], copy=False)
        bloco = df._mgr.blocks[0].values
        if bloco.flags.c_contiguous:
            pytest.skip("pandas copiou o ndarray no construtor")
        out = copia_df(df)
    assert not np.shares_memory(out["a"].to_numpy(), arr)
    assert out._mgr.blocks[0].values.flags.c_contiguous
    pd.testing.assert_frame_equal(out, df, check_exact=True)


def test_buffer_so_leitura_cai_na_profunda():
    with _cow(True):
        df = _base(500)
        for b in df._mgr.blocks:
            v = b.values
            if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                v.flags.writeable = False
        out = copia_df(df)
    assert not _compartilha(out, df)
    pd.testing.assert_frame_equal(out, df, check_exact=True)


def test_buffer_como_copia_tipos():
    b = common_mod._buffer_como_copia
    assert b(np.zeros((2, 3)))
    assert not b(np.zeros((3, 2)).T)
    assert b(pd.array([1, None], dtype="Int64"))
    assert b(pd.Categorical(["a", "b"]))
    assert not b(pd.arrays.SparseArray([0, 1]))
    pa = pytest.importorskip("pyarrow")
    assert b(pd.arrays.ArrowExtensionArray(pa.array([1.0, None])))


def test_api_interna_ausente_cai_na_profunda(monkeypatch):
    with _cow(True):
        df = _base(400)

        def _quebra(self):
            raise AttributeError("sem _consolidate_inplace")
        monkeypatch.setattr(pd.DataFrame, "_consolidate_inplace", _quebra)
        out = copia_df(df)
    assert not _compartilha(out, df)
    pd.testing.assert_frame_equal(out, df, check_exact=True)


# ─────────────────────────── (vi) memo de _samples ───────────────────────────
@pytest.mark.parametrize("kind", ["str", "int", "date64"])
def test_samples_semeado_com_o_mesmo_tipo(kind):
    df = _base(900)
    if kind == "int":
        df["amostra"] = np.where(df["amostra"] == "DES", 1, 2).astype("int64")
        ref = 1
    elif kind == "date64":
        df["amostra"] = pd.to_datetime(np.where(df["amostra"] == "DES", "2024-01-01",
                                                "2024-06-01"))
        ref = df["amostra"].iloc[np.flatnonzero(
            (df["amostra"] == pd.Timestamp("2024-01-01")).to_numpy())[0]]
    else:
        ref = "DES"
    seg = _seg(df) if kind == "str" else ModelSegmenter(
        df, target="target", sample_col="amostra", ref_sample=ref, verbose=False)
    assert seg._samples_cache is not None             # semeado no __init__
    semeado = list(seg._samples())
    ams = list(seg.df["amostra"].dropna().unique())   # expressão de _samples()
    esperado = [ref] + [a for a in ams if a != ref]
    assert semeado == esperado
    assert [type(v) for v in semeado[1:]] == [type(v) for v in esperado[1:]]
    seg._samples_cache = None                         # recalcula pelo caminho antigo
    assert [type(v) for v in seg._samples()] == [type(v) for v in semeado]
    assert seg._samples() == semeado
