"""Onda 3 (aba Análise) — reuso do ranking no gráfico do optimal binning,
prebins da pré-binagem CART como ``user_splits`` (3A), safra pelos valores
únicos e reuso por máscara (3B) e a caixa de grupos categóricos só com 2
colunas (3D). O cache de figuras da UI (3C) está em ``test_perf_onda3_ui.py``.

Cada caso compara contra a versão ANTIGA (copiada aqui ou forçada por
monkeypatch) com igualdade EXATA."""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import warnings

import numpy as np
import pandas as pd
import pytest

optbinning = pytest.importorskip("optbinning")
from optbinning import ContinuousOptimalBinning, OptimalBinning  # noqa: E402

from yggdrasil.credit_risk import _common  # noqa: E402
from yggdrasil.credit_risk.model import ModelSegmenter  # noqa: E402
from yggdrasil.credit_risk.model import segmenter as segmod  # noqa: E402

VERSAO_020 = str(optbinning.__version__).startswith("0.20.")


# ───────────────────────── base ─────────────────────────
def _base(n=7000, seed=5, regressao=False):
    rng = np.random.default_rng(seed)
    amostra = np.where(rng.random(n) < .7, "DES", "OOT")
    xc = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(xc - .5)))
    if regressao:
        y = np.clip(p + rng.normal(0, .1, n), 0, 1)
    else:
        y = (rng.random(n) < p).astype(float)
    y[rng.random(n) < .04] = np.nan
    df = pd.DataFrame({"amostra": amostra, "target": y})
    df["x_cont"] = np.where(rng.random(n) < .1, np.nan, xc)
    df["x_int"] = rng.integers(0, 40, n) + (xc > 1) * 5
    xi = pd.array(rng.integers(0, 25, n) + (xc > 0) * 3, dtype="Int64")
    xi[rng.random(n) < .1] = pd.NA
    df["x_Int64"] = xi
    df["x_F64"] = pd.array(np.where(rng.random(n) < .1, np.nan, xc * 2), dtype="Float64")
    df["x_flag"] = (rng.random(n) < .3).astype(float)
    df["x_conc"] = np.where(rng.random(n) < .985, 1.0, rng.normal(size=n))   # prebins vazios
    # prebin puro (só eventos) ⇒ refinamento do optbinning na classificação
    df["x_puro"] = np.where((y >= .5) & (xc > 1.2), 50 + xc, rng.normal(size=n))
    df["x_tarde"] = np.where(np.arange(n) < 6000, (xc > 0).astype(float), xc)   # recorte
    df["cat"] = rng.choice(list("ABCD"), n).astype(object)
    meses = [dt.date(2023 + m // 12, m % 12 + 1, 1) for m in range(14)]
    dts = np.array(meses, dtype=object)[rng.integers(0, 14, n)]
    dts[rng.random(n) < .02] = None
    df["dt_ref"] = dts
    return df


NUMS = ["x_cont", "x_int", "x_Int64", "x_F64", "x_flag", "x_conc", "x_puro", "x_tarde"]


def _seg(df, **kw):
    kw.setdefault("sample_col", "amostra")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ModelSegmenter(df, target="target", verbose=False, date_col="dt_ref", **kw)


@pytest.fixture()
def cap_pequeno(monkeypatch):
    monkeypatch.setattr(_common, "OPTBINNING_MAX_ROWS", 2500)   # DES (~4,7k) acima do teto


def _quieto(fn):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


# ───────────────────────── 3A: prebins como user_splits ─────────────────────────
def test_reuso_fixado_na_versao():
    """O reuso só liga no optbinning 0.20.x com sklearn < 1.8 (detalhe interno)."""
    import sklearn
    maior, menor = (int(p) for p in sklearn.__version__.split(".")[:2])
    assert segmod._PREBINS_REUSO == (VERSAO_020 and (maior, menor) < (1, 8))


def _prebins_e_cortes(cls, cap_cls, x, y, mnb, max_rows=None):
    b = cap_cls(name="v", dtype="numerical", max_n_bins=mnb, min_bin_size=.05,
                monotonic_trend="auto_asc_desc")
    full = _common.fit_optbinning_splits(b, x, y, max_rows=max_rows)
    return b.__dict__.get("_yg_prebins"), full, getattr(b, "_is_fitted", False)


@pytest.mark.skipif(not segmod._PREBINS_REUSO, reason="reuso fixado no optbinning 0.20.x")
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("col", ["x_cont", "x_int", "x_conc", "x_puro", "x_F64"])
def test_user_splits_iguais_ao_ajuste_completo(task, col):
    df = _base(regressao=task == "regression")
    ok = df["target"].notna().to_numpy()
    x = df[col].to_numpy(dtype="float64")[ok]
    y = df["target"].to_numpy(dtype="float64")[ok]
    clf = task == "classification"
    yy = y.astype(int) if clf else y
    cls = OptimalBinning if clf else ContinuousOptimalBinning
    cap_cls = segmod._OptimalBinningCaptura if clf else segmod._ContinuousOptimalBinningCaptura
    for max_rows in (None, 3000):                       # sem e com sorteio do teto
        pre, _, fitted = _prebins_e_cortes(cls, cap_cls, x, yy, 5, max_rows)
        assert fitted and pre is not None
        for mnb in (5, 6, 10, 20):
            b = cls(name="v", dtype="numerical", max_n_bins=mnb, min_bin_size=.05,
                    monotonic_trend="auto_asc_desc")
            full = _common.fit_optbinning_splits(b, x, yy, max_rows=max_rows)
            # o captador não altera o ajuste
            bc = cap_cls(name="v", dtype="numerical", max_n_bins=mnb, min_bin_size=.05,
                         monotonic_trend="auto_asc_desc")
            assert np.array_equal(_common.fit_optbinning_splits(bc, x, yy, max_rows=max_rows),
                                  full)
            if len(pre) == 0:
                assert full == []                        # prebins vazios ⇒ sem corte
                continue
            bu = cls(name="v", dtype="numerical", max_n_bins=mnb, min_bin_size=.05,
                     monotonic_trend="auto_asc_desc", user_splits=pre.copy())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                user = _common.fit_optbinning_splits(bu, x, yy, max_rows=max_rows)
            assert np.array_equal(np.asarray(user), np.asarray(full)), (col, mnb, max_rows)
            assert [type(c) for c in user] == [type(c) for c in full]


def _old_optbin_numeric_bins(self, feature, sample=None, max_n_bins=5, min_bin_size=0.05):
    fit = self._frame(sample, cols=[feature, self.target])
    if self._detect_kind(feature, fit) != "num":
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        binaria = self._bins_binaria(feature, fit, "num")
    if binaria is not None:
        return binaria
    x = fit[feature].to_numpy(dtype="float64")
    y = fit[self.target].to_numpy(dtype="float64")
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    x_obs = x[~np.isnan(x)]
    if len(y) < 4 or x_obs.size == 0 or x_obs.min() == x_obs.max():
        return []
    if self.task_type == "classification":
        b = OptimalBinning(name=feature, dtype="numerical", max_n_bins=max_n_bins,
                           min_bin_size=min_bin_size, monotonic_trend="auto_asc_desc")
        cortes = _common.fit_optbinning_splits(b, x, y.astype(int))
    else:
        b = ContinuousOptimalBinning(name=feature, dtype="numerical", max_n_bins=max_n_bins,
                                     min_bin_size=min_bin_size, monotonic_trend="auto_asc_desc")
        cortes = _common.fit_optbinning_splits(b, x, y)
    if not cortes:
        return []
    edges = [-np.inf, *cortes, np.inf]
    bins = [{"kind": "num", "lo": edges[i], "hi": edges[i + 1]} for i in range(len(edges) - 1)]
    if fit[feature].isna().any():
        bins.append({"kind": "na"})
    return bins


def _iguais(a, b):
    return repr(a) == repr(b)


def _compara_optbin(seg, feats, samples=(None, "DES", "OOT")):
    for f in feats:
        for s in samples:
            for mnb in (5, 6):
                novo = _quieto(lambda: seg._optbin_numeric_bins(f, s, mnb))
                velho = _quieto(lambda: _old_optbin_numeric_bins(seg, f, s, mnb))
                assert _iguais(novo, velho), (f, s, mnb)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_optbin_numeric_bins_igual_ao_antigo(task, cap_pequeno):
    seg = _seg(_base(regressao=task == "regression"), task_type=task)
    _quieto(seg.variable_iv)
    _compara_optbin(seg, NUMS + ["cat"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.set_manual_bins("x_cont", [-0.5, 0.5])
        seg.set_missing_bin("x_cont", "pior")
        seg.set_manual_bins("x_int", [10, 20])
        seg.set_missing_bin("x_int", "melhor")
        seg.clear_manual_bins("x_int")                     # (destino que sobrar não vale)
        seg.set_manual_bins("x_Int64", [5, 12])
        nova = seg.create_categorical("x_Int64")
        seg.set_manual_bins("x_F64", [-1.0, 1.0])
        dums = seg.create_scorecard_dummies("x_F64")
    _compara_optbin(seg, ["x_cont", "x_int", "x_Int64", nova] + list(dums))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.clear_manual_bins("x_cont")
        seg.remove_derived(dums[0])
        # recria com o mesmo nome e outras faixas: versão nova da coluna ⇒ sem memo velho
        seg.set_manual_bins("x_F64", [-0.5, 0.5])
        dums2 = seg.create_scorecard_dummies("x_F64")
    assert dums[0] in dums2
    _compara_optbin(seg, ["x_cont"] + list(dums2))


def test_optbin_numeric_bins_sem_amostra(cap_pequeno):
    seg = _seg(_base(), sample_col=None)
    _quieto(seg.variable_iv)
    _compara_optbin(seg, NUMS, samples=(None, "DES"))


class _Contador:
    def __init__(self, monkeypatch):
        self.fits, self.arvores = 0, 0
        for cls in (OptimalBinning, ContinuousOptimalBinning):
            orig = cls.fit

            def fit(b, *a, _orig=orig, **k):
                self.fits += 1
                return _orig(b, *a, **k)
            monkeypatch.setattr(cls, "fit", fit)
        from optbinning.binning.prebinning import PreBinning
        orig_pb = PreBinning.fit

        def pb_fit(pb, *a, **k):
            self.arvores += 1
            return orig_pb(pb, *a, **k)
        monkeypatch.setattr(PreBinning, "fit", pb_fit)

    def zera(self):
        self.fits = self.arvores = 0


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_analise_reusa_ranking_e_prebins(task, cap_pequeno, monkeypatch):
    seg = _seg(_base(regressao=task == "regression"), task_type=task)
    cont = _Contador(monkeypatch)
    _quieto(seg.variable_iv)
    assert cont.arvores > 0
    for f in ("x_cont", "x_int", "x_F64"):
        cont.zera()
        # gráfico "faixas do optimal binning": cache hit do ranking (nenhum ajuste)
        _quieto(lambda: seg._optbin_numeric_bins(f, seg.ref_sample))
        assert (cont.fits, cont.arvores) == (0, 0), f
        # tabela (6) e PSI por safra (10): ajustam, mas sem a árvore CART
        _quieto(lambda: seg.variable_table(f))
        _quieto(lambda: seg._resolve_bins(f, 10, 0.05))
        if segmod._PREBINS_REUSO:
            assert cont.arvores == 0, f
            assert cont.fits == 2, f
        # bins manuais + novo gráfico: o memo do optbin sobrevive (0 ajustes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seg.set_manual_bins(f, [0.5, 3.0])
        cont.zera()
        _quieto(lambda: seg._optbin_numeric_bins(f, seg.ref_sample))
        assert (cont.fits, cont.arvores) == (0, 0), f
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seg.clear_manual_bins(f)
        cont.zera()
        # sem manual de novo: o ranking (5) é refeito só com o CP-SAT
        _quieto(lambda: seg._resolve_bins(f))
        if segmod._PREBINS_REUSO:
            assert cont.arvores == 0


def test_prebins_vazios_sem_ajuste_e_sem_aviso(cap_pequeno, monkeypatch):
    seg = _seg(_base())
    _quieto(seg.variable_iv)
    if segmod._PREBINS_REUSO:
        chaves = [k for k in seg._prebin_cache if k[0] == "x_conc"]
        assert chaves and all(len(seg._prebin_cache[k]) == 0 for k in chaves)
    cont = _Contador(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # nenhum aviso espúrio
        r6 = seg._resolve_bins("x_conc", 6, 0.05)
    assert r6 == ([], "num")
    if segmod._PREBINS_REUSO:
        assert cont.fits == 0                           # [] direto, sem optbinning


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_ranking_tabelas_e_inversao_iguais_sem_reuso(task, cap_pequeno, monkeypatch):
    """Tudo que a aba usa sai igual com e sem o reuso dos prebins."""
    df = _base(regressao=task == "regression")
    novo, velho = _seg(df.copy(), task_type=task), _seg(df.copy(), task_type=task)

    def roda(seg):
        out = {"iv": _quieto(seg.variable_iv)}
        for f in NUMS:
            out[f] = _quieto(lambda: (seg.variable_table(f), seg.variable_inversion(f),
                                      seg.variable_faixa_share_by_safra(f, max_n_bins=10),
                                      seg.variable_psi_by_safra(f),
                                      seg._resolve_bins(f, 10, 0.05),
                                      seg.variable_table(f, sample="OOT", max_n_bins=20)))
        return out
    a = roda(novo)
    monkeypatch.setattr(segmod, "_PREBINS_REUSO", False)
    b = roda(velho)
    assert not velho._prebin_cache
    pd.testing.assert_frame_equal(a["iv"], b["iv"], check_exact=True)
    for f in NUMS:
        for x, y in zip(a[f], b[f]):
            if isinstance(x, pd.DataFrame):
                pd.testing.assert_frame_equal(x, y, check_exact=True)
                assert repr(x.attrs) == repr(y.attrs)
            else:
                assert repr(x) == repr(y), f


def test_falha_com_user_splits_refaz_ajuste_completo(cap_pequeno, monkeypatch):
    if not segmod._PREBINS_REUSO:
        pytest.skip("reuso fixado no optbinning 0.20.x")
    seg = _seg(_base())
    _quieto(seg.variable_iv)
    esperado = _quieto(lambda: _old_optbin_numeric_bins(seg, "x_cont", None, 6))
    orig = OptimalBinning.fit

    def fit(b, *a, **k):
        if b.user_splits is not None:
            raise RuntimeError("boom")
        return orig(b, *a, **k)
    monkeypatch.setattr(OptimalBinning, "fit", fit)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # o aviso da falha não vaza
        bins, _ = seg._resolve_bins("x_cont", 6, 0.05)
    assert repr(bins) == repr(esperado)


def test_caches_novos_fora_do_scorer_destacado():
    seg = _seg(_base(n=2000))
    _quieto(seg.variable_iv)
    _quieto(lambda: seg._optbin_numeric_bins("x_cont"))
    assert seg._optbin_cache
    s = seg._detached_scorer()
    assert s._prebin_cache == {} and s._optbin_cache == {}


# ───────────────────────── 3B: safra pelos valores únicos ─────────────────────────
def _old_safra_codes(col):
    per = pd.to_datetime(col, errors="coerce").dt.to_period("M")
    codes, uniq = pd.factorize(per, sort=True, use_na_sentinel=True)
    return np.asarray(codes, dtype=np.int32), [str(u) for u in uniq]


def _obj(n=0):
    o = ModelSegmenter.__new__(ModelSegmenter)
    return o


def _colunas_data(n=1500, seed=2):
    rng = np.random.default_rng(seed)
    meses = [dt.date(2022 + m // 12, m % 12 + 1, 1 + m % 27) for m in range(20)]
    datas = np.array(meses, dtype=object)[rng.integers(0, 20, n)]
    d_none = datas.copy()
    d_none[rng.random(n) < .05] = None
    d_none[rng.random(n) < .03] = np.nan
    iso = np.array([d.isoformat() for d in datas], dtype=object)
    iso_ruim = iso.copy()
    iso_ruim[rng.random(n) < .03] = "abc"
    iso_ruim[rng.random(n) < .03] = "NaT"
    iso_ruim[rng.random(n) < .03] = None
    ddmm = np.array([d.strftime("%d/%m/%Y") for d in datas], dtype=object)
    misto = iso.copy()
    misto[rng.random(n) < .2] = ddmm[rng.random(n) < .2][:1][0] if n else None
    diarias = np.array([(dt.date(2023, 1, 1) + dt.timedelta(days=int(i))).isoformat()
                        for i in rng.integers(0, 400, n)], dtype=object)
    horas = np.array([dt.datetime(d.year, d.month, d.day, int(h))
                      for d, h in zip(datas, rng.integers(0, 24, n))], dtype=object)
    d64 = pd.to_datetime(pd.Series(datas)).to_numpy()
    d64_nat = d64.copy()
    d64_nat[rng.random(n) < .05] = np.datetime64("NaT")
    cols = {
        "date_none": d_none,
        "date_puro": datas.copy(),
        "iso": iso, "iso_ruim": iso_ruim, "ddmm": ddmm, "misto": misto,
        "diarias": diarias, "horas": horas,
        "sentinela": np.where(rng.random(n) < .1, dt.date(9999, 12, 31), datas),
        "d64": d64, "d64_nat": d64_nat,
        "tz": pd.Series(d64_nat).dt.tz_localize("America/Sao_Paulo"),
        "cat_date": pd.Categorical(d_none),
        "cat_iso": pd.Categorical(iso_ruim),
        "str_iso": pd.array(iso_ruim, dtype="string"),
        "toda_none": np.array([None] * n, dtype=object),
        "int": rng.integers(0, 10**12, n),
    }
    out = pd.DataFrame({k: v for k, v in cols.items()})
    out["horas"] = pd.Series(horas, dtype=object)          # datetime puro, sem virar d64
    assert out["horas"].dtype == object and out["date_puro"].dtype == object
    try:
        import pyarrow as pa
        out["arrow_date"] = pd.array(d_none, dtype=pd.ArrowDtype(pa.date32()))
        out["arrow_str"] = pd.array(iso_ruim, dtype=pd.ArrowDtype(pa.string()))
    except ImportError:  # pragma: no cover
        pass
    return out


def test_safra_codes_igual_ao_antigo():
    df = _colunas_data()
    o = _obj()
    o.df = df
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in df.columns:
            try:
                ve = _old_safra_codes(df[c])
            except Exception as e:                      # mesmo erro nos dois caminhos
                with pytest.raises(type(e)):
                    o._safra_codes(c)
                continue
            no = o._safra_codes(c)
            assert no[0].dtype == np.int32
            assert np.array_equal(no[0], ve[0]), c
            assert no[1] == ve[1], c


def test_safra_codes_df_vazio_e_memo():
    o = _obj()
    o.df = _colunas_data().iloc[0:0]
    for c in ("date_none", "iso", "toda_none", "cat_date"):
        cod, rot = o._safra_codes(c)
        assert cod.dtype == np.int32 and len(cod) == 0 and rot == []
    o2 = _obj()
    o2.df = _colunas_data()
    a = o2._safra_codes("date_none")
    assert o2._safra_codes("date_none") is a              # memoizado


def _old_recorte(col):
    per = pd.to_datetime(col, errors="coerce").dt.to_period("M")
    cod, pers = pd.factorize(per, sort=True, use_na_sentinel=True)
    return np.asarray(cod), [str(p) for p in pers]


def test_safra_codes_linhas_igual_ao_recorte():
    df = _colunas_data()
    o = _obj()
    o.df = df
    rng = np.random.default_rng(9)
    mascaras = [None, rng.random(len(df)) < .3, np.zeros(len(df), bool),
                (np.arange(len(df)) % 97) == 0]
    exatas = {"date_none", "date_puro", "sentinela", "d64", "d64_nat", "toda_none",
              "arrow_date", "cat_date"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in df.columns:
            for m in mascaras:
                r = o._safra_codes_linhas(c, m)
                if c not in exatas or c not in df.columns:
                    assert r is None, c                 # formato/fuso: recorte convertido
                    continue
                assert r is not None, c
                sub = df[c] if m is None else df[c][m]
                vc, vr = _old_recorte(sub)
                assert np.array_equal(r[0], vc), c
                assert r[1] == vr, c


def _linhas_do_grafico(fig):
    ax = fig.axes[0]
    return ([(ln.get_label(), list(map(float, ln.get_xdata())),
              [float(v) for v in ln.get_ydata()]) for ln in ax.lines],
            [t.get_text() for t in ax.get_xticklabels()])


@pytest.mark.parametrize("dtipo", ["date", "d64", "iso"])
def test_risco_por_safra_e_perfil_iguais(dtipo, monkeypatch):
    df = _base(n=4000)
    if dtipo == "d64":
        df["dt_ref"] = pd.to_datetime(df["dt_ref"])
    elif dtipo == "iso":
        df["dt_ref"] = [d.isoformat() if d is not None else None for d in df["dt_ref"]]
    seg = _seg(df)
    _quieto(seg.variable_iv)
    feats = ["x_cont", "cat", "x_flag"]

    def roda():
        out = []
        for f in feats:
            for s in (None, "DES", "OOT"):
                fig = _quieto(lambda: seg.plot_variable_risk_by_safra(f, sample=s))
                out.append(_linhas_do_grafico(fig))
        for todas in (True, False):
            out.append(_quieto(lambda: seg.variables_profile_by_safra(
                features=feats, all_samples=todas)))
        return out
    novo = roda()
    monkeypatch.setattr(ModelSegmenter, "_safra_codes_linhas", lambda *a, **k: None)
    velho = roda()
    for a, b in zip(novo, velho):
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        else:
            assert repr(a) == repr(b)


# ───────────────────────── 3D: caixa de grupos categóricos ─────────────────────────
def test_cat_box_so_duas_colunas(monkeypatch):
    pytest.importorskip("ipywidgets")
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    df = _base(n=3000)
    df["cat_manual"] = df["cat"]
    with contextlib.redirect_stdout(io.StringIO()):
        ui = ModelSegmenterUI(df, target="target", task_type="classification",
                              sample_col="amostra", ref_sample="DES", date_col="dt_ref")
    ui.seg.set_manual_bins("cat_manual", [["A", "B"], ["C", "D"]])

    def arvore(w):
        return (type(w).__name__, repr(getattr(w, "value", None)),
                repr(list(getattr(w, "options", ()))),
                [arvore(f) for f in getattr(w, "children", ()) or ()])

    def estado():
        return ([(c, w.value, list(w.options)) for c, w in ui._an_cat_widgets.items()],
                [arvore(w) for w in ui.an_cat_box.children])
    res = {}
    for modo in ("novo", "velho"):
        if modo == "velho":
            orig = ui.seg._frame
            monkeypatch.setattr(ui.seg, "_frame",
                                lambda sample=None, cols=None: orig(sample))
        for f in ("cat", "cat_manual"):
            ui.dd_var2.value = f
            ui._rebuild_an_cat_box(force=True)
            res[(modo, f)] = estado()
    for f in ("cat", "cat_manual"):
        assert res[("novo", f)] == res[("velho", f)]


# ───────────────────────── 3E: séries da inversão (memo + posições) ─────────────────────────
def _old_risco_por_grupo(self, bin_codes, grupo_codes, n_grupos, n_bins, linhas):
    y = self.df[self.target].to_numpy(dtype="float64")
    ok = linhas & (bin_codes >= 0) & (grupo_codes >= 0) & ~np.isnan(y)
    idx = grupo_codes[ok].astype(np.int64) * n_bins + bin_codes[ok]
    tam = n_grupos * n_bins
    soma = np.bincount(idx, weights=y[ok], minlength=tam)
    cont = np.bincount(idx, minlength=tam)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(cont > 0, soma / np.maximum(cont, 1), np.nan)
    return r.reshape(n_grupos, n_bins)


def _old_variable_bin_series(self, feature, bins, time_col=None, sample=None, min_n=20):
    labels = [self._bin_label(feature, b) for b in bins]
    nb = len(bins)
    bc = self._feature_bin_codes(feature, bins)
    zeros = np.zeros(len(self.df), dtype=np.int32)
    ref_risco = list(_old_risco_por_grupo(
        self, bc, zeros, 1, nb, self._rows_mask(self.ref_sample))[0])
    order = sorted(range(len(bins)),
                   key=lambda i: (np.inf if pd.isna(ref_risco[i]) else ref_risco[i]))
    xs_s = self._samples()
    cod_a = np.full(len(self.df), -1, dtype=np.int32)
    for j, a in enumerate(xs_s):
        cod_a[self._rows_mask(a)] = j
    mat_s = _old_risco_por_grupo(self, bc, cod_a, len(xs_s), nb,
                                 np.ones(len(self.df), dtype=bool))
    ser_s = {i: list(mat_s[:, i]) for i in range(nb)}
    xs_t, ser_t = [], {i: [] for i in range(nb)}
    tcol = time_col or self.date_col
    if tcol is not None and tcol in self.df.columns:
        linhas = self._rows_mask(sample, all_rows=not sample)
        cod_t, rot_t = self._safra_codes(tcol)
        n_t = np.bincount(cod_t[linhas & (cod_t >= 0)], minlength=len(rot_t))
        mat_t = _old_risco_por_grupo(self, bc, cod_t, len(rot_t), nb, linhas)
        for k, rot in enumerate(rot_t):
            if n_t[k] == 0 or n_t[k] < min_n:
                continue
            xs_t.append(rot)
            for i in range(nb):
                ser_t[i].append(mat_t[k, i])
    return {"ordered": order, "labels": labels, "ref_risco": ref_risco,
            "xs_sample": xs_s, "ser_sample": ser_s,
            "xs_safra": xs_t, "ser_safra": ser_t}


def _inv_igual(a, b):
    """Igualdade exata (NaN nas mesmas posições) de dicts/listas de float."""
    if isinstance(a, dict):
        return (isinstance(b, dict) and list(a) == list(b)
                and all(_inv_igual(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(_inv_igual(x, y) for x, y in zip(a, b)))
    if isinstance(a, (float, np.floating)) and np.isnan(a):
        return isinstance(b, (float, np.floating)) and np.isnan(b) and type(a) is type(b)
    return type(a) is type(b) and a == b


def _inversoes(seg, feats):
    out = []
    for f in feats:
        for tc in (None, "dt_ref"):
            for s in (None, "DES", "OOT"):
                out.append(_quieto(lambda: seg.variable_inversion(f, tc, s)))
                out.append(_quieto(lambda: seg.variable_inversion(f, tc, s, min_n=200)))
    return out


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_inversao_igual_ao_antigo(task, monkeypatch):
    df = _base(n=6000, regressao=task == "regression")
    seg = _seg(df, task_type=task)
    _quieto(seg.variable_iv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.set_manual_bins("x_Int64", [5, 12])
        seg.set_missing_bin("x_Int64", "pior")
        seg.set_manual_bins("cat", [["A", "B"], ["C"], ["D"]])
        der = seg.create_categorical("x_cont")
    feats = ["x_cont", "x_int", "x_Int64", "x_flag", "cat", der]
    monkeypatch.setattr(ModelSegmenter, "max_linhas_graficos", 2500)   # base > teto

    def roda():
        fora = _inversoes(seg, feats)
        with seg._amostra_graficos():
            dentro = _inversoes(seg, feats)
            # mudar o teto dentro do bloco não pode reaproveitar memo antigo
            seg.max_linhas_graficos = 1800
            dentro2 = _inversoes(seg, feats[:2])
            del seg.max_linhas_graficos
        return fora, dentro, dentro2

    novo = roda()
    novo_de_novo = roda()                               # 2ª passada: tudo do memo
    monkeypatch.setattr(ModelSegmenter, "_variable_bin_series", _old_variable_bin_series)
    velho = roda()
    for a, b, c in zip(novo, novo_de_novo, velho):
        assert _inv_igual(a, c)
        assert _inv_igual(b, c)
    assert not _inv_igual(novo[1], novo[0])             # a amostra dos gráficos muda algo


def test_inversao_memo_invalida_e_devolve_copias():
    seg = _seg(_base(n=4000))
    _quieto(seg.variable_iv)
    r1 = _quieto(lambda: seg.variable_inversion("x_cont"))
    r1["series"]["ser_sample"][0].append(123.0)          # o chamador muta o resultado
    r1["ordered"].append(99)
    r2 = _quieto(lambda: seg.variable_inversion("x_cont"))
    bins, _ = seg._resolve_bins("x_cont", 6, 0.05)
    assert _inv_igual(r2["series"], _old_variable_bin_series(seg, "x_cont", bins))
    assert any(k[0] == "x_cont" for k in seg._bin_series_cache)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.set_manual_bins("x_cont", [-0.3, 0.4])         # poda o memo da variável
    assert not any(k[0] == "x_cont" for k in seg._bin_series_cache)
    r3 = _quieto(lambda: seg.variable_inversion("x_cont"))
    bins, _ = seg._resolve_bins("x_cont", 6, 0.05)
    assert _inv_igual(r3["series"], _old_variable_bin_series(seg, "x_cont", bins))
    # derivada removida e recriada com o mesmo nome e outras faixas
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        der = seg.create_categorical("x_cont", new_name="x_der")
        _quieto(lambda: seg.variable_inversion(der))
        seg.remove_derived(der)
        seg.set_manual_bins("x_cont", [0.0, 1.0])
        assert seg.create_categorical("x_cont", new_name="x_der") == der
    r4 = _quieto(lambda: seg.variable_inversion(der))
    bins, _ = seg._resolve_bins(der, 6, 0.05)
    assert _inv_igual(r4["series"], _old_variable_bin_series(seg, der, bins))
    s = seg._detached_scorer()
    assert not s._bin_series_cache
