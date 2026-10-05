"""Onda 4 (redundância) do ModelSegmenter: equivalência com o caminho antigo.

4A — correlation_report sorteia as POSIÇÕES antes de copiar (mesmo sorteio do
     DataFrame.sample): o recorte é idêntico (check_exact, índice e dtypes).
4B — _spearman_pairwise com ranks por código + bincount: bit a bit igual à
     versão anterior (copiada aqui como referência).
4C — V de Cramér por códigos fatorados uma vez: tabela densa (até
     _CRAMER_DENSO_MAX células) e réplica em blocos (até _CRAMER_EXATO_MAX)
     idênticas ao crosstab + chi2_contingency — inclusive a ORDEM da tabela com
     pares empatados (colunas tipo ID); acima, qui-quadrado esparso (rtol 1e-9
     no chi2) truncado em 0.
4D — memo por par (Spearman/Cramér) e do VIF: resultados iguais aos de um
     segmentador novo, com invalidação por versão de coluna/modelo/linhas.
"""
from __future__ import annotations

import decimal
import time
import tracemalloc
import warnings

import numpy as np
import pandas as pd
import pytest

import yggdrasil.credit_risk.model.segmenter as segmod
from yggdrasil.credit_risk.model import ModelSegmenter
from yggdrasil.credit_risk.model.segmenter import (_cramers_v_codigos,
                                                   _spearman_pairwise)

D = decimal.Decimal


# ─────────────────────── referências (código anterior) ───────────────────────
def _spearman_antigo(frame: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import rankdata
    cols = list(frame.columns)
    X = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    k = len(cols)
    out = np.eye(k)
    tem_nan = np.isnan(X).any(axis=0)

    def _pearson(a, b):
        if a.size < 2:
            return np.nan
        a = a - a.mean(); b = b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den > 0 else np.nan

    completas = [i for i in range(k) if not tem_nan[i]]
    if len(completas) >= 2:
        R = np.column_stack([rankdata(X[:, i]) for i in completas])
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(R, rowvar=False)
        for a, i in enumerate(completas):
            for b, j in enumerate(completas):
                if i != j:
                    out[i, j] = C[a, b]
    ordens = {}

    def _rank_no_recorte(i, ok):
        if i not in ordens:
            ordens[i] = np.argsort(X[:, i], kind="mergesort")
        o = ordens[i]
        o = o[ok[o]]
        v = X[o, i]
        n = v.size
        if n == 0:
            return np.empty(0)
        novo = np.r_[True, v[1:] != v[:-1]]
        grupo = np.cumsum(novo) - 1
        inicio = np.flatnonzero(novo)
        tam = np.diff(np.r_[inicio, n])
        media = inicio + (tam + 1) / 2.0
        r = np.empty(len(X))
        r[o] = media[grupo]
        return r[ok]

    for i in range(k):
        for j in range(i + 1, k):
            if not (tem_nan[i] or tem_nan[j]):
                continue
            ok = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
            v = _pearson(_rank_no_recorte(i, ok), _rank_no_recorte(j, ok))
            out[i, j] = out[j, i] = v
    for i in range(k):
        v = X[:, i][~np.isnan(X[:, i])]
        if v.size and np.all(v == v[0]):
            out[i, :] = np.nan; out[:, i] = np.nan
            out[i, i] = np.nan
    return pd.DataFrame(out, index=cols, columns=cols)


def _cramers_v_antigo(x: pd.Series, y: pd.Series, chi2_out=None) -> float:
    from scipy.stats import chi2_contingency
    m = x.notna() & y.notna()
    if not bool(m.any()):
        return float("nan")
    tab = pd.crosstab(x[m].astype(str), y[m].astype(str))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return float("nan")
    chi2 = float(chi2_contingency(tab, correction=False)[0])
    if chi2_out is not None:
        chi2_out.append((chi2, float(tab.to_numpy().sum()), min(tab.shape) - 1))
    n = float(tab.to_numpy().sum())
    k = min(tab.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n > 0 else float("nan")


def _sub_antigo(seg, feats, sample=None):
    sub = seg._frame(sample, cols=feats)
    cap = seg.max_linhas_graficos
    if cap and len(sub) > int(cap):
        sub = sub.sample(n=int(cap), random_state=seg.random_state)
    return sub


def _correlation_report_antigo(seg, threshold=0.85, features=None, sample=None):
    base = (list(features) if features is not None
            else (seg.selected_features() or list(seg.candidates)))
    feats, vistos = [], set()
    for f in base:
        if f in seg.df.columns and f not in vistos:
            feats.append(f); vistos.add(f)
    sub = _sub_antigo(seg, feats, sample)
    num = [f for f in feats if seg._detect_kind(f, sub) == "num"]
    cat = [f for f in feats if seg._detect_kind(f, sub) == "cat"]
    corr_num = (_spearman_antigo(sub[num]) if len(num) >= 2
                else pd.DataFrame(np.eye(len(num)), index=num, columns=num))
    if len(cat) >= 2:
        vals = np.eye(len(cat))
        for i in range(len(cat)):
            for j in range(i + 1, len(cat)):
                v = _cramers_v_antigo(sub[cat[i]], sub[cat[j]])
                vals[i, j] = vals[j, i] = v
        corr_cat = pd.DataFrame(vals, index=cat, columns=cat)
    else:
        corr_cat = pd.DataFrame(np.eye(len(cat)), index=cat, columns=cat)
    pares = []
    for mat, metodo in ((corr_num, "spearman"), (corr_cat, "cramers_v")):
        cols = list(mat.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = float(mat.iloc[i, j])
                if np.isfinite(v) and abs(v) >= float(threshold):
                    pares.append((cols[i], cols[j], metodo, abs(v)))
    pares.sort(key=lambda t: -t[3])
    envolvidas = sorted({f for p in pares for f in p[:2]})
    iv_map = {}
    if envolvidas:
        rk = seg.variable_iv(features=envolvidas, with_psi=False)
        iv_map = dict(zip(rk["variavel"], rk["iv"]))
    removidas, rows = set(), []
    for f1, f2, metodo, a in pares:
        iv1 = float(iv_map.get(f1, np.nan)); iv2 = float(iv_map.get(f2, np.nan))
        if np.isnan(iv2) or (not np.isnan(iv1) and iv1 >= iv2):
            manter, remover = f1, f2
        else:
            manter, remover = f2, f1
        rows.append({"variavel_1": f1, "variavel_2": f2, "metodo": metodo,
                     "associacao": round(a, 4), "iv_1": iv1, "iv_2": iv2,
                     "manter": manter, "remover": remover})
        if f1 in removidas or f2 in removidas:
            continue
        removidas.add(remover)
    out = pd.DataFrame(rows, columns=["variavel_1", "variavel_2", "metodo",
                                      "associacao", "iv_1", "iv_2", "manter", "remover"])
    out.attrs.update(threshold=float(threshold), corr_num=corr_num, corr_cat=corr_cat,
                     poda_sugerida=[f for f in feats if f in removidas])
    return out


def _igual_relatorio(a, b):
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    pd.testing.assert_frame_equal(a.attrs["corr_num"], b.attrs["corr_num"], check_exact=True)
    pd.testing.assert_frame_equal(a.attrs["corr_cat"], b.attrs["corr_cat"], check_exact=True)
    assert a.attrs["poda_sugerida"] == b.attrs["poda_sugerida"]
    assert a.attrs["threshold"] == b.attrs["threshold"]


# ─────────────────────────────── fixture ───────────────────────────────
def _base(n=6000, seed=0, nao_range=False):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    df = pd.DataFrame({
        "amostra": np.where(rng.random(n) < 0.7, "DES", "OOT"),
        "target": (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int),
    })
    a = z + rng.normal(0, 0.3, n)
    a[rng.random(n) < 0.05] = np.nan
    df["num_a"] = a
    b = 0.95 * z + rng.normal(0, 0.1, n)
    b[rng.random(n) < 0.03] = np.nan
    df["num_b"] = b
    df["num_empate"] = np.round(rng.normal(size=n), 1)            # empates pesados
    c = rng.integers(0, 5, n).astype(float)
    c[rng.random(n) < 0.02] = np.nan
    df["num_poisson"] = c
    zz = rng.normal(size=n)
    zz[rng.random(n) < 0.2] = 0.0
    zz[rng.random(n) < 0.1] = -0.0
    zz[rng.random(n) < 0.01] = np.inf
    zz[rng.random(n) < 0.01] = -np.inf
    zz[rng.random(n) < 0.03] = np.nan
    df["num_inf"] = zz                                             # ±0, ±inf, NaN
    df["num_completa"] = rng.normal(size=n)
    df["num_completa2"] = df["num_completa"] * 2 + rng.normal(0, 0.5, n)
    df["num_const"] = 3.0
    iv = pd.array(rng.integers(0, 30, n), dtype="Int64")
    iv[rng.random(n) < 0.05] = pd.NA
    df["int_na"] = iv
    df["flag_bool"] = rng.random(n) < 0.3
    cats = np.array(["A", "B", "C", "D", "e", "É"], dtype=object)
    cl = cats[rng.integers(0, 6, n)].copy()
    cl[rng.random(n) < 0.04] = None
    df["cat_low"] = cl
    df["cat_espelho"] = pd.Series(cl).map({"A": "x", "B": "y", "C": "z", "D": "w",
                                           "e": "x", "É": "q"}).to_numpy(dtype=object)
    df["cat_hc"] = np.array([f"n{v}" for v in rng.integers(0, 800, n)], dtype=object)
    mist = np.array([1, "1", 1.0, True, D("1.00"), D("1"), "a", 2, None, np.nan,
                     D("NaN"), pd.NA], dtype=object)
    df["cat_misto"] = mist[rng.integers(0, len(mist), n)]
    df["cat_category"] = pd.Categorical(cats[rng.integers(0, 6, n)],
                                        categories=list(cats) + ["nunca"])
    s = pd.array(cats[rng.integers(0, 6, n)], dtype="string")
    s[rng.random(n) < 0.03] = pd.NA
    df["cat_string"] = s
    bo = pd.array(rng.random(n) < 0.5, dtype="boolean")
    bo[rng.random(n) < 0.05] = pd.NA
    df["cat_boolean"] = bo
    dec = np.array([D(f"{v:.2f}") for v in rng.integers(0, 12, n) * 0.25], dtype=object)
    df["cat_dec"] = dec
    # datetime (vira 'cat' pelo dtype): o texto depende do recorte → caminho por série
    datas = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 40, n), "D")
    dt = pd.Series(datas)
    dt[rng.random(n) < 0.3] = dt[rng.random(n) < 0.3] + pd.Timedelta(hours=3)
    df["cat_data"] = dt.to_numpy()
    if nao_range:
        df.index = pd.Index(rng.permutation(n) * 3 + 7)
    return df


NUMS = ["num_a", "num_b", "num_empate", "num_poisson", "num_inf", "num_completa",
        "num_completa2", "num_const", "int_na"]
CATS = ["flag_bool", "cat_low", "cat_espelho", "cat_hc", "cat_misto", "cat_category",
        "cat_string", "cat_boolean", "cat_dec", "cat_data"]


def _seg(df=None, sample_col="amostra", cap=1500, **kw):
    df = _base() if df is None else df
    seg = ModelSegmenter(df, target="target", task_type="classification",
                         sample_col=sample_col, ref_sample="DES", verbose=False,
                         features=NUMS + CATS, **kw)
    seg.max_linhas_graficos = cap
    return seg


# ────────────────────────────── 4A: recorte ──────────────────────────────
@pytest.mark.parametrize("sample_col", ["amostra", None])
@pytest.mark.parametrize("cap", [1500, 10**9, None, "igual"])
@pytest.mark.parametrize("rs", ["int", "none", "randomstate"])
@pytest.mark.parametrize("nao_range", [False, True])
def test_4a_recorte_por_posicao_identico(sample_col, cap, rs, nao_range):
    seg = _seg(_base(nao_range=nao_range), sample_col=sample_col, cap=None)
    feats = NUMS + CATS
    n_ref = len(seg._frame(None, cols=feats))
    seg.max_linhas_graficos = n_ref if cap == "igual" else cap
    for amostra in ([None, "OOT"] if sample_col else [None]):
        def _novo():
            pos, sorteia = seg._posicoes_correlacao(amostra)
            return seg._recorte_correlacao(feats, pos, sorteia)
        if rs == "int":
            seg.random_state = 7
            a, b = _sub_antigo(seg, feats, amostra), _novo()
        elif rs == "none":
            seg.random_state = None
            np.random.seed(123); a = _sub_antigo(seg, feats, amostra)
            st_a = np.random.get_state()[1].copy()
            np.random.seed(123); b = _novo()
            assert np.array_equal(st_a, np.random.get_state()[1])   # RNG global avança igual
        else:
            seg.random_state = np.random.RandomState(5); a = _sub_antigo(seg, feats, amostra)
            seg.random_state = np.random.RandomState(5); b = _novo()
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_4a_colunas_duplicadas_usa_fallback():
    df = _base(1200)
    seg = _seg(df, cap=500)
    seg.df = pd.concat([seg.df, seg.df[["num_a"]]], axis=1)      # 'num_a' duplicada
    seg.random_state = 0
    pos, sorteia = seg._posicoes_correlacao(None)
    sub = seg._recorte_correlacao(["num_b", "cat_low"], pos, sorteia)
    pd.testing.assert_frame_equal(sub, _sub_antigo(seg, ["num_b", "cat_low"]), check_exact=True)


# ───────────────────────────── 4B: Spearman ─────────────────────────────
def _frames_spearman():
    rng = np.random.default_rng(3)
    n = 4000
    z = rng.normal(size=n)
    base = pd.DataFrame({
        "a": z, "b": 0.7 * z + rng.normal(size=n), "c": np.round(z, 1),
        "flag": (rng.random(n) < 0.2).astype(float),
        "poi": rng.poisson(2, n).astype(float),
        "dec": [D(str(v)) for v in np.round(rng.normal(size=n), 2)],
    })
    nan_ind = base.copy()
    for c in ["a", "b", "c", "poi"]:
        nan_ind.loc[rng.random(n) < 0.08, c] = np.nan
    nan_cor = base.copy()
    mm = rng.random(n) < 0.1
    nan_cor.loc[mm, ["a", "b"]] = np.nan                         # NaN correlacionados
    nan_cor.loc[rng.random(n) < 0.05, "flag"] = np.nan
    zeros = pd.DataFrame({"z": np.where(rng.random(n) < 0.5, 0.0, -0.0),
                          "w": rng.normal(size=n),
                          "inf": np.where(rng.random(n) < 0.1, np.inf,
                                          np.where(rng.random(n) < 0.1, -np.inf,
                                                   rng.normal(size=n)))})
    zeros.loc[rng.random(n) < 0.1, "w"] = np.nan
    zeros.loc[:5, "z"] = 1.0
    bordas = pd.DataFrame({"const": 5.0, "um_valido": np.nan, "todo_nan": np.nan,
                           "x": rng.normal(size=n), "y": rng.normal(size=n)})
    bordas.loc[17, "um_valido"] = 2.0
    bordas.loc[: n // 2, "x"] = np.nan
    bordas.loc[n // 2:, "y"] = np.nan                            # par sem linhas em comum
    bordas.loc[3, "const"] = np.nan
    intna = pd.DataFrame({"i": pd.array(rng.integers(0, 4, n), dtype="Int64"),
                          "f": pd.array(rng.normal(size=n), dtype="Float64")})
    intna.loc[rng.random(n) < 0.1, "i"] = pd.NA
    um = pd.DataFrame({"a": [1.0], "b": [np.nan], "c": [2.0]})
    dois = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, 1.0], "c": [2.0, 1.0],
                         "d": [3.0, 3.0]})
    vazio = pd.DataFrame({"a": np.array([], float), "b": np.array([], float)})
    return [base, nan_ind, nan_cor, zeros, bordas, intna, um, dois, vazio]


@pytest.mark.parametrize("i", range(9))
def test_4b_spearman_bit_a_bit(i):
    df = _frames_spearman()[i]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # corrcoef com 0/1 linha
        novo = _spearman_pairwise(df)
        antigo = _spearman_antigo(df)
    pd.testing.assert_frame_equal(novo, antigo, check_exact=True)


def test_4b_spearman_igual_ao_pandas():
    df = _frames_spearman()[1][["a", "b", "c", "flag", "poi"]]
    ref = df.corr(method="spearman")
    got = _spearman_pairwise(df)
    np.testing.assert_allclose(got.to_numpy(), ref.to_numpy(), rtol=1e-12, atol=1e-12)


def test_4b_memo_parcial_bit_a_bit():
    """O núcleo com memo, chamado depois num subconjunto/ordem diferente (pares
    vindos do memo + bloco recalculado), dá o mesmo que a versão sem memo."""
    df = _frames_spearman()[2]
    cols = list(df.columns)
    memo = {}
    chaves = {c: (c, 1) for c in cols}
    r1 = segmod._spearman_memo(cols, lambda idx: df[[cols[i] for i in idx]], memo,
                               [chaves[c] for c in cols])
    pd.testing.assert_frame_equal(r1, _spearman_antigo(df), check_exact=True)
    sub = ["poi", "b", "dec", "a"]
    lidas = []

    def _obter(idx):
        lidas.append([sub[i] for i in idx])
        return df[[sub[i] for i in idx]]
    r2 = segmod._spearman_memo(sub, _obter, memo, [chaves[c] for c in sub])
    pd.testing.assert_frame_equal(r2, _spearman_antigo(df[sub]), check_exact=True)
    # diagnóstico e pares com NaN vêm do memo: só o bloco sem NaN (outra tupla
    # de colunas) é recalculado
    assert lidas == [["poi", "dec"]]


# ─────────────────────────────── 4C: Cramér ───────────────────────────────
def _pares_cramer():
    rng = np.random.default_rng(11)
    n = 3000
    out = []
    mist = np.array([1, "1", 1.0, True, D("1.00"), D("1"), "a", 2, None, np.nan,
                     D("NaN"), pd.NA, "b"], dtype=object)
    x = pd.Series(mist[rng.integers(0, len(mist), n)])
    y = pd.Series(mist[rng.integers(0, len(mist), n)])
    out.append((x, y))
    a = pd.Series(rng.integers(0, 3, n)).astype(str)
    b = pd.Series(rng.integers(0, 50, n)).map(lambda v: f"c{v}")
    a[rng.random(n) < 0.1] = None
    out.append((a, b))                                            # 3 × 50
    out.append((pd.Series(rng.integers(0, 2, n)), pd.Series(rng.integers(0, 2, n))))  # 2×2
    big = pd.Series(rng.integers(0, 300, 30000)).map(lambda v: f"r{v}")
    big2 = pd.Series(rng.integers(0, 1000, 30000)).map(lambda v: f"s{v}")
    out.append((big, big2))                                       # 300 × 1000
    um = pd.Series(["A"] * 50); out.append((um, pd.Series(list("ab") * 25)))  # 1 nível
    vaz1 = pd.Series([None] * 20, dtype=object); out.append((vaz1, pd.Series(list("ab") * 10)))
    # níveis que só existem fora do recorte do par
    p = pd.Series(list("abcd") * 100)
    q = pd.Series(list("xy") * 200, dtype=object)
    q[p == "d"] = None
    out.append((p, q))
    # independência EXATA, n não diádico: margens 100/200 × 150/150
    xi = pd.Series(["r0"] * 100 + ["r1"] * 200)
    yi = pd.Series((["c0"] * 50 + ["c1"] * 50) + (["c0"] * 100 + ["c1"] * 100))
    out.append((xi, yi))
    cat = pd.Series(pd.Categorical(rng.choice(list("pqr"), n), categories=list("pqrs")))
    out.append((cat, x))
    st = pd.Series(pd.array(rng.choice(["k", "l", None], n), dtype="string"))
    bo = pd.Series(pd.array(rng.choice([True, False, None], n), dtype="boolean"))
    out.append((st, bo))
    return out


@pytest.mark.parametrize("i", range(10))
def test_4c_cramer_denso_identico(i):
    x, y = _pares_cramer()[i]
    novo = ModelSegmenter._cramers_v(x, y)
    antigo = _cramers_v_antigo(x, y)
    assert (np.isnan(novo) and np.isnan(antigo)) or novo == antigo
    # e no sentido inverso (a tabela transposta também bate)
    novo2, antigo2 = ModelSegmenter._cramers_v(y, x), _cramers_v_antigo(y, x)
    assert (np.isnan(novo2) and np.isnan(antigo2)) or novo2 == antigo2


@pytest.mark.parametrize("i", range(10))
def test_4c_cramer_esparso_tolerancia(i, monkeypatch):
    x, y = _pares_cramer()[i]
    monkeypatch.setattr(segmod, "_CRAMER_DENSO_MAX", 0)          # força o esparso
    monkeypatch.setattr(segmod, "_CRAMER_EXATO_MAX", 0)
    novo = ModelSegmenter._cramers_v(x, y)
    info = []
    antigo = _cramers_v_antigo(x, y, chi2_out=info)
    assert np.isnan(novo) == np.isnan(antigo)
    if not np.isnan(antigo):
        chi2_a, n, k = info[0]
        chi2_n = novo ** 2 * n * k
        np.testing.assert_allclose(chi2_n, chi2_a, rtol=1e-9, atol=1e-9)
        assert abs(novo - antigo) <= 1e-6


def test_4c_independencia_exata_da_zero_e_nao_nan(monkeypatch):
    xi, yi = _pares_cramer()[7]
    monkeypatch.setattr(segmod, "_CRAMER_DENSO_MAX", 0)
    monkeypatch.setattr(segmod, "_CRAMER_EXATO_MAX", 0)
    v = ModelSegmenter._cramers_v(xi, yi)
    assert v == 0.0 or (np.isfinite(v) and v < 1e-7)
    a = pd.factorize(xi, sort=True)[0]; b = pd.factorize(yi, sort=True)[0]
    assert np.isfinite(_cramers_v_codigos(a, b))
    assert np.isnan(_cramers_v_codigos(np.array([], int), np.array([], int)))


def test_4c_par_gigante_sem_estouro_e_memoria():
    """30k × 100k níveis em 300k linhas (3e9 células): o crosstab denso +
    chi2_contingency pediam ~150 GB (falta de memória); agora sai valor, pelo
    esparso, com memória O(linhas)."""
    rng = np.random.default_rng(0)
    n = 300_000
    a = rng.integers(0, 30_000, n)
    b = rng.integers(0, 100_000, n)
    tracemalloc.start()
    try:
        v = _cramers_v_codigos(a, b)
        pico = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert np.isfinite(v) and 0 <= v <= 1
    assert pico < 200 * 2 ** 20


@pytest.mark.parametrize("i", range(10))
def test_4c_replica_em_blocos_identica(i, monkeypatch):
    """Réplica em blocos (_chi2_blocos) forçada com blocos minúsculos: bit a bit
    igual ao crosstab + chi2_contingency, nos dois sentidos."""
    monkeypatch.setattr(segmod, "_CRAMER_DENSO_MAX", 0)
    monkeypatch.setattr(segmod, "_CRAMER_BLOCO", 37)
    assert segmod._replica_chi2_confere()
    x, y = _pares_cramer()[i]
    for u, w in ((x, y), (y, x)):
        novo, antigo = ModelSegmenter._cramers_v(u, w), _cramers_v_antigo(u, w)
        assert (np.isnan(novo) and np.isnan(antigo)) or novo == antigo


def test_4c_replica_reprovada_cai_no_esparso(monkeypatch):
    """Se a autoverificação reprovar a réplica (numpy/scipy de outro layout),
    acima de _CRAMER_DENSO_MAX vale o esparso — nunca um 'quase exato'."""
    x, y = _pares_cramer()[3]
    a = pd.factorize(x, sort=True)[0]; b = pd.factorize(y, sort=True)[0]
    monkeypatch.setattr(segmod, "_CRAMER_DENSO_MAX", 0)
    monkeypatch.setattr(segmod, "_CRAMER_EXATO_MAX", 0)
    esparso = _cramers_v_codigos(a, b)
    monkeypatch.setattr(segmod, "_CRAMER_EXATO_MAX", 10**12)
    monkeypatch.setattr(segmod, "_CRAMER_REPLICA_OK", False)
    assert _cramers_v_codigos(a, b) == esparso


def test_4c_replica_memoria():
    """ID de 300k níveis × categórica de 20 (6e6 células, acima do denso): valor
    exato com ~8 B/célula (o denso pedia ~50 B/célula, ~300 MB)."""
    rng = np.random.default_rng(2)
    n, L = 300_000, 20
    a = np.arange(n)
    b = rng.integers(0, L, n); b[:L] = np.arange(L)
    assert n * L > segmod._CRAMER_DENSO_MAX
    segmod._replica_chi2_confere()
    tracemalloc.start()
    try:
        v = _cramers_v_codigos(a, b)
        pico = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert np.isfinite(v) and pico < 150 * 2 ** 20


def _base_ids(n=6000, seed=4):
    """Colunas tipo ID (V≈1 contra todas, muitos empates) com tabelas acima de
    _CRAMER_DENSO_MAX (id1 × id2: 6000 × 1100 = 6,6e6 células)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    df = pd.DataFrame({"target": (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)})
    df["id1"] = np.array([f"u{i}" for i in rng.permutation(n)], dtype=object)
    g = np.r_[np.arange(1100), rng.integers(0, 1100, n - 1100)]
    df["id2"] = np.array([f"g{v}" for v in rng.permutation(g)], dtype=object)
    mist = np.array([1, "1", 1.0, True, D("1.00"), "a", 2, None, D("NaN"), pd.NA, "b",
                     "c", 3, 4.5], dtype=object)
    df["obj_mix"] = mist[rng.integers(0, len(mist), n)]
    df["cat17"] = np.array([f"k{v}" for v in rng.integers(0, 17, n)], dtype=object)
    df["cat_hc"] = np.array([f"h{v}" for v in rng.integers(0, 800, n)], dtype=object)
    df["num"] = z + rng.normal(0, 0.5, n)
    return df


def test_4c_ids_acima_do_denso_ordem_igual_ao_antigo():
    """O caso que o esparso reordenava: pares tipo ID empatados em V≈1 com
    tabelas acima de _CRAMER_DENSO_MAX. A tabela (ORDEM das linhas), corr_cat
    e a poda saem idênticas ao caminho antigo, em vários limiares."""
    df = _base_ids()
    feats = ["id1", "id2", "obj_mix", "cat17", "cat_hc", "num"]
    mk = lambda: ModelSegmenter(df.copy(), target="target", task_type="classification",
                                verbose=False, features=feats)
    novo, velho = mk(), mk()
    novo.max_linhas_graficos = velho.max_linhas_graficos = None
    assert len(df) * df["id2"].nunique() > segmod._CRAMER_DENSO_MAX
    for thr in (0.3, 0.9, 0.99):
        a = novo.correlation_report(threshold=thr)
        b = _correlation_report_antigo(velho, threshold=thr)
        _igual_relatorio(a, b)
        assert list(zip(a["variavel_1"], a["variavel_2"])) ==             list(zip(b["variavel_1"], b["variavel_2"]))
    assert len(a) > 0


# ───────────────────── correlation_report: antigo × novo ─────────────────────
@pytest.mark.parametrize("cfg", ["padrao", "sem_amostra", "nao_range", "cap_grande",
                                 "rs_none", "rs_randomstate", "cap_none"])
def test_4_correlation_report_identico_ao_antigo(cfg):
    df = _base(nao_range=(cfg == "nao_range"))
    sample_col = None if cfg == "sem_amostra" else "amostra"
    cap = {"cap_grande": 10**9, "cap_none": None}.get(cfg, 1500)
    novo, velho = _seg(df.copy(), sample_col=sample_col, cap=cap), \
        _seg(df.copy(), sample_col=sample_col, cap=cap)
    for thr, smp in ((0.3, None), (0.85, None), (0.5, "OOT" if sample_col else None)):
        if cfg == "rs_none":
            novo.random_state = velho.random_state = None
            np.random.seed(9); a = novo.correlation_report(threshold=thr, sample=smp)
            np.random.seed(9); b = _correlation_report_antigo(velho, threshold=thr, sample=smp)
        elif cfg == "rs_randomstate":
            novo.random_state = np.random.RandomState(1)
            velho.random_state = np.random.RandomState(1)
            a = novo.correlation_report(threshold=thr, sample=smp)
            b = _correlation_report_antigo(velho, threshold=thr, sample=smp)
        else:
            a = novo.correlation_report(threshold=thr, sample=smp)
            b = _correlation_report_antigo(velho, threshold=thr, sample=smp)
        _igual_relatorio(a, b)
    assert len(a.attrs["corr_cat"]) == len(CATS)


# ─────────────────────────────── 4D: memo ───────────────────────────────
def _conta_calculos(monkeypatch):
    cont = {"cramer": 0, "spearman_lidas": 0}
    orig_c = segmod._cramers_v_codigos
    orig_s = segmod._spearman_memo

    def _c(a, b):
        cont["cramer"] += 1
        return orig_c(a, b)

    def _s(cols, obter, memo, chaves):
        def _o(idx):
            cont["spearman_lidas"] += len(idx)
            return obter(idx)
        return orig_s(cols, _o, memo, chaves)
    monkeypatch.setattr(segmod, "_cramers_v_codigos", _c)
    monkeypatch.setattr(segmod, "_spearman_memo", _s)
    return cont


def test_4d_limiar_e_subconjunto_iguais_a_segmentador_novo(monkeypatch):
    df = _base()
    seg = _seg(df.copy())
    cont = _conta_calculos(monkeypatch)
    r1 = seg.correlation_report(threshold=0.85)
    assert cont["cramer"] > 0 and cont["spearman_lidas"] > 0
    cont.update(cramer=0, spearman_lidas=0)
    r2 = seg.correlation_report(threshold=0.3)
    assert cont == {"cramer": 0, "spearman_lidas": 0}            # tudo do memo
    _igual_relatorio(r1, _correlation_report_antigo(_seg(df.copy()), threshold=0.85))
    _igual_relatorio(r2, _correlation_report_antigo(_seg(df.copy()), threshold=0.3))
    # "Excluir redundantes": subconjunto → pares do memo, bloco sem NaN recalculado
    sub = [f for f in NUMS + CATS if f not in ("num_b", "cat_espelho", "num_completa2")]
    r3 = seg.correlation_report(threshold=0.3, features=sub)
    assert cont["cramer"] == 0
    _igual_relatorio(r3, _correlation_report_antigo(_seg(df.copy()), threshold=0.3,
                                                    features=sub))


def test_4d_derivada_recriada_com_mesmo_nome_invalida():
    df = _base()
    seg = _seg(df.copy())
    nome = seg.create_categorical("num_a", new_name="num_a_cat")
    feats = NUMS + CATS + [nome]
    r1 = seg.correlation_report(threshold=0.3, features=feats)
    seg.remove_derived(nome)
    seg.set_manual_bins("num_a", [-0.5, 0.0, 0.5])
    assert seg.create_categorical("num_a", new_name="num_a_cat") == nome
    r2 = seg.correlation_report(threshold=0.3, features=feats)
    assert not r1.attrs["corr_cat"].equals(r2.attrs["corr_cat"])
    # igual a um segmentador novo com o mesmo estado
    novo = _seg(df.copy())
    novo.set_manual_bins("num_a", [-0.5, 0.0, 0.5])
    novo.create_categorical("num_a", new_name="num_a_cat")
    _igual_relatorio(r2, novo.correlation_report(threshold=0.3, features=feats))


def test_4d_mudar_teto_e_amostra_recalcula(monkeypatch):
    df = _base()
    seg = _seg(df.copy())
    seg.correlation_report(threshold=0.3)
    cont = _conta_calculos(monkeypatch)
    seg.max_linhas_graficos = 1200
    r = seg.correlation_report(threshold=0.3)
    assert cont["cramer"] > 0
    ref = _seg(df.copy(), cap=1200)
    _igual_relatorio(r, _correlation_report_antigo(ref, threshold=0.3))
    cont.update(cramer=0)
    r_oot = seg.correlation_report(threshold=0.3, sample="OOT")
    assert cont["cramer"] > 0
    _igual_relatorio(r_oot, _correlation_report_antigo(ref, threshold=0.3, sample="OOT"))
    # semente não inteira: sem memo (o sorteio muda a cada chamada)
    seg.random_state = None
    cont.update(cramer=0)
    seg.correlation_report(threshold=0.3)
    seg.correlation_report(threshold=0.3)
    # (os pares com cat_data, datetime, vão pelo _cramers_v por série, que
    # também termina em _cramers_v_codigos: 1 chamada por par nos dois caminhos)
    assert cont["cramer"] == 2 * (len(CATS) * (len(CATS) - 1) // 2)


def test_4d_memo_fora_do_broadcast_e_do_load(tmp_path):
    seg = _seg()
    seg.correlation_report(threshold=0.3)
    assert seg._corr_cache is not None
    s = seg._detached_scorer()
    assert s._corr_cache is None and s._vif_cache is None


def _seg_modelo(df):
    seg = _seg(df, cap=1500)
    seg.clear_features()
    for f in ("num_a", "num_b", "num_completa", "cat_low"):
        seg.include(f)
    return seg


def test_4d_vif_memo_e_invalidacao(monkeypatch):
    df = _base()
    seg = _seg_modelo(df.copy())
    seg.fit("logistica", transform="raw")
    v1 = seg.vif_table()
    chamadas = []
    orig = ModelSegmenter._qr_acumulado

    def _qr(blocks):
        chamadas.append(1)
        return orig(blocks)
    monkeypatch.setattr(ModelSegmenter, "_qr_acumulado", staticmethod(_qr))
    v2 = seg.vif_table()
    assert chamadas == []
    pd.testing.assert_frame_equal(v1, v2, check_exact=True)
    v2.loc[0, "vif"] = -1.0                                       # cópia: não vaza
    pd.testing.assert_frame_equal(seg.vif_table(), v1, check_exact=True)
    # rótulo novo sem refit: nomes na hora
    seg.feature_labels["num_a"] = "Renda nova"
    v3 = seg.vif_table()
    assert chamadas == []
    assert any("Renda nova" in t for t in v3["termo"])
    novo = _seg_modelo(df.copy()); novo.fit("logistica", transform="raw")
    novo.feature_labels["num_a"] = "Renda nova"
    pd.testing.assert_frame_equal(v3, novo.vif_table(), check_exact=True)
    pd.testing.assert_frame_equal(seg.vif_table(use_labels=False),
                                  novo.vif_table(use_labels=False), check_exact=True)
    # refit com outras variáveis → recalcula e bate com um segmentador novo
    seg.include("num_empate")
    seg.fit("logistica", transform="raw")
    n0 = len(chamadas)
    v4 = seg.vif_table()
    assert len(chamadas) == n0 + 1
    novo.include("num_empate"); novo.fit("logistica", transform="raw")
    pd.testing.assert_frame_equal(v4, novo.vif_table(), check_exact=True)
    # (o segmentador novo também passa pelo _qr_acumulado espiado)
    n0 = len(chamadas)
    seg.vif_table()
    assert len(chamadas) == n0
    # mudar o teto de linhas → recalcula
    seg.max_linhas_graficos = 900
    seg.vif_table()
    assert len(chamadas) == n0 + 1
    # set_model com o MESMO objeto (reajustado por fora) → recalcula
    seg.set_model(seg.model, features=list(seg.model_features))
    seg.vif_table()
    assert len(chamadas) == n0 + 2


def test_4d_vif_two_stage():
    rng = np.random.default_rng(4)
    df = _base()
    df["target"] = np.clip(rng.normal(0.3, 0.3, len(df)), 0, 1)
    def _mk():
        s = ModelSegmenter(df.copy(), target="target", task_type="regression",
                           sample_col="amostra", ref_sample="DES", verbose=False,
                           features=NUMS + CATS)
        s.max_linhas_graficos = 1500
        s.clear_features()
        for f in ("num_a", "num_b", "num_completa"):
            s.include(f)
        return s
    seg = _mk()
    seg.fit("linear", transform="raw")
    seg.vif_table()
    seg.fit_two_stage(threshold=0.3)
    try:
        v = seg.vif_table()
    except Exception as e:  # noqa: BLE001 — o mesmo erro do segmentador novo
        novo = _mk(); novo.fit_two_stage(threshold=0.3)
        with pytest.raises(type(e)):
            novo.vif_table()
        return
    novo = _mk(); novo.fit_two_stage(threshold=0.3)
    pd.testing.assert_frame_equal(v, novo.vif_table(), check_exact=True)


def test_4d_segundo_clique_rapido():
    df = _base(30_000, seed=5)
    seg = _seg(df, cap=20_000)
    t0 = time.perf_counter(); seg.correlation_report(threshold=0.85)
    t1 = time.perf_counter(); seg.correlation_report(threshold=0.3)
    t2 = time.perf_counter()
    assert t2 - t1 < 1.5
    assert t2 - t1 < (t1 - t0)
