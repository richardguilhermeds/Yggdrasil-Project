"""Onda 1 de desempenho do ModelSegmenter: cada caminho novo contra a versão
ANTIGA copiada aqui como referência (igualdade exata). Cobre ``plot_roc``/
``plot_roc_compare`` (AUC pelo trapézio da mesma curva), ``_decimal_columns``
(busca em blocos), ``variable_faixa_share_by_safra`` (bincount),
``metrics_by_safra`` (só a curva ROC) + ``plot_metrics_by_safra(table=)``,
``backtest`` (fatias sobre arrays) e ``hosmer_lemeshow`` (numpy)."""
from __future__ import annotations

import contextlib
import decimal
import io
import tracemalloc
import warnings

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.model import ModelSegmenter
from yggdrasil.credit_risk.model import segmenter as segmod
from yggdrasil.metrics import classification_metrics, regression_metrics

INF = np.inf
D = decimal.Decimal


def _df(n=8000, seed=3, task="classification"):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 0.9 * x1 + 0.4 * rng.normal(size=n)
    x1[rng.random(n) < .05] = np.nan
    c = rng.choice(list("ABCDE"), n).astype(object)
    p = 1 / (1 + np.exp(-(-1.5 + np.nan_to_num(x1))))
    y = ((rng.random(n) < p).astype(float) if task == "classification"
         else np.clip(p + 0.1 * rng.normal(size=n), 0, 1))
    meses = pd.date_range("2023-01-01", periods=10, freq="MS")
    df = pd.DataFrame({"x1": x1, "x2": x2, "c": c, "target": y,
                       "dt_ref": rng.choice(meses, n)})
    df.loc[:4, "dt_ref"] = pd.NaT
    # uma safra (jan/24) só existe na DES: com sample='OOT' fica vazia
    df.loc[5:30, "dt_ref"] = pd.Timestamp("2024-01-01")
    df["amostra"] = np.where(df["dt_ref"] >= meses[7], "OOT", "DES")
    df.loc[5:30, "amostra"] = "DES"
    return df


def _quieto():
    pilha = contextlib.ExitStack()
    pilha.enter_context(contextlib.redirect_stdout(io.StringIO()))
    pilha.enter_context(warnings.catch_warnings())
    warnings.simplefilter("ignore")
    return pilha


def _seg(task="classification", fit=True, **kw):
    with _quieto():
        s = ModelSegmenter(_df(task=task, **kw), target="target", task_type=task,
                           sample_col="amostra", ref_sample="DES", date_col="dt_ref",
                           verbose=False)
        if fit:
            s.fit("logistica" if task == "classification" else "linear",
                  features=["x1", "x2", "c"])
    return s


@pytest.fixture(scope="module", params=["classification", "regression"])
def seg_fit(request):
    return _seg(request.param)


# ─────────────────────────── 1A. plot_roc ───────────────────────────
@pytest.mark.parametrize("seed", range(6))
def test_auc_trapezio_igual_roc_auc_score(seed):
    from sklearn.metrics import auc as _auc_trap
    from sklearn.metrics import roc_auc_score, roc_curve
    rng = np.random.default_rng(seed)
    for n in (2, 7, 500, 20000):
        y = (rng.random(n) < (0.03 if seed % 2 else 0.4)).astype("float64")
        y[0], y[-1] = 0.0, 1.0
        for sc in (rng.random(n), np.round(rng.random(n), 2),
                   rng.integers(0, 4, n).astype(float)):
            assert _auc_trap(*roc_curve(y, sc)[:2]) == roc_auc_score(y, sc)


def test_plot_roc_e_compare_legenda_e_curva():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    s = _seg()
    with _quieto():
        snap = s.snapshot("base")
        s.fit("logistica", features=["x1", "c"])
    for amostra in ("DES", "OOT"):
        y, sc = s._sample_scores(amostra)
        fig = s.plot_roc(amostra)
        ax = fig.axes[0]
        auc = roc_auc_score(y, sc)
        assert ax.get_legend().get_texts()[0].get_text() == \
            f"AUC={auc:.3f} · Gini={2*auc-1:.3f}"
        fpr, tpr, _ = roc_curve(y, sc)
        assert np.array_equal(ax.lines[0].get_xydata(), np.column_stack([fpr, tpr]))
        plt.close(fig)
        fig = s.plot_roc_compare(snap, sample=amostra)
        textos = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        mask = (s.df["amostra"] == amostra).to_numpy()
        y_all = s.df.loc[mask, "target"].to_numpy("float64")
        esperado = []
        for nome, ser in (("base", snap["score"]), ("atual", s.score_)):
            v = ser.reindex(s.df.index)[mask].to_numpy("float64")
            ok = ~np.isnan(y_all) & ~np.isnan(v)
            esperado.append(f"{nome} · AUC={roc_auc_score(y_all[ok], v[ok]):.3f}")
        assert textos == esperado
        plt.close(fig)


# ─────────────────────────── 1B. _decimal_columns ───────────────────────────
def _decimal_columns_antigo(df, cols):
    alvo = set(cols)
    achadas = []
    for j in range(df.shape[1]):
        if df.columns[j] not in alvo:
            continue
        col = df.iloc[:, j]
        if col.dtype != object:
            continue
        validos = np.flatnonzero(col.notna().to_numpy())
        if len(validos) and isinstance(col.iloc[int(validos[0])], decimal.Decimal):
            achadas.append(df.columns[j])
    return list(dict.fromkeys(achadas))


def _col_obj(n, preenche):
    v = np.empty(n, dtype=object)
    v[:] = None
    for p, x in preenche.items():
        v[p] = x
    return v


@pytest.mark.parametrize("indice", ["padrao", "duplicado", "texto"])
def test_decimal_columns_igual_ao_antigo(indice):
    n = 10000
    cols = {
        "dec0": _col_obj(n, {0: D("1.5"), 1: D("2")}),
        "dec_fim": _col_obj(n, {n - 1: D("3")}),
        "dec4095": _col_obj(n, {4095: D("1")}),
        "dec4096": _col_obj(n, {4096: D("1")}),
        "dec4097": _col_obj(n, {4097: D("1")}),
        "tudo_none": _col_obj(n, {}),
        "tudo_nan": np.full(n, np.nan, dtype=object),
        "tudo_pdna": np.array([pd.NA] * n, dtype=object),
        "dec_nan_antes": _col_obj(n, {0: D("NaN"), 1: D("1")}),
        "pdna_antes": _col_obj(n, {0: pd.NA, 1: pd.NaT, 2: np.nan, 3: D("1")}),
        "str_antes": _col_obj(n, {0: "a", 1: D("1")}),
        "str_tarde": _col_obj(n, {9000: "a", 9001: D("1")}),
    }
    df = pd.DataFrame(cols)
    df["int64"] = pd.array([1] * (n - 1) + [None], dtype="Int64")
    df["boolean"] = pd.array([True] * (n - 1) + [None], dtype="boolean")
    df["category"] = pd.Categorical(["a"] * n)
    # nomes repetidos: um object com Decimal e outro float
    dup = pd.DataFrame({"rep": _col_obj(n, {7: D("1")})})
    dup2 = pd.DataFrame({"rep": np.zeros(n)})
    df = pd.concat([df, dup, dup2], axis=1)
    if indice == "duplicado":
        df.index = [0] * n
    elif indice == "texto":
        df.index = [f"r{i % 50}" for i in range(n)]
    nomes = list(dict.fromkeys(df.columns))
    novo = segmod._decimal_columns(df, nomes)
    assert novo == _decimal_columns_antigo(df, nomes)
    assert novo == ["dec0", "dec_fim", "dec4095", "dec4096", "dec4097",
                    "dec_nan_antes", "pdna_antes", "rep"]
    # subconjunto de colunas e DataFrame vazio
    assert segmod._decimal_columns(df, ["dec_fim", "str_antes"]) == ["dec_fim"]
    vazio = df.iloc[:0]
    assert segmod._decimal_columns(vazio, nomes) == _decimal_columns_antigo(vazio, nomes) == []


def test_decimal_columns_aviso_identico(monkeypatch):
    df = _df(n=3000)
    df["valor"] = [D(f"{v:.2f}") for v in df["x2"] * 1000]
    df.loc[df.index[:5000], "valor"] = None
    df.loc[df.index[-1], "valor"] = D("1")
    df["texto"] = "a"

    def avisos(func):
        monkeypatch.setattr(segmod, "_decimal_columns", func)
        with warnings.catch_warnings(record=True) as w, \
                contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("always")
            s = ModelSegmenter(df, target="target", sample_col="amostra",
                               date_col="dt_ref", verbose=False)
        return s.decimal_cols_, [str(x.message) for x in w if "DecimalType" in str(x.message)]

    novo = avisos(segmod._decimal_columns)
    antigo = avisos(_decimal_columns_antigo)
    assert novo == antigo and novo[0] == ["valor"] and len(novo[1]) == 1


# ─────────────────── 1C. variable_faixa_share_by_safra ───────────────────
def _faixa_share_antigo(self, feature, time_col=None, sample=None, max_n_bins=6,
                        min_bin_size=0.05, bins=None, all_samples=False):
    time_col = time_col or self.date_col
    if bins is None:
        bins, _kind = self._resolve_bins(feature, max_n_bins, min_bin_size, None, sample)
    ordem = [self._bin_label(feature, b) for b in (bins or [])]
    if not ordem:
        return pd.DataFrame(columns=["safra"])
    linhas = self._rows_mask(sample, all_rows=all_samples)
    cod, rot = self._safra_codes(time_col)
    bc = self._feature_bin_codes(feature, bins)
    ok = linhas & (cod >= 0)
    if not ok.any():
        return pd.DataFrame(columns=["safra"])
    rot_faixa = np.asarray(ordem + ["(faltante)"], dtype=object)
    tab = pd.crosstab(np.asarray(rot, dtype=object)[cod[ok]],
                      rot_faixa[np.where(bc[ok] >= 0, bc[ok], len(ordem))])
    pct = tab.div(tab.sum(axis=1), axis=0) * 100
    cols = [c for c in dict.fromkeys(ordem) if c in pct.columns]
    if "(faltante)" in pct.columns and "(faltante)" not in cols:
        cols.append("(faltante)")
    pct = pct[cols].round(1).sort_index()
    pct.index.name = "safra"
    return pct.reset_index()


BINS_FAIXA = {
    # buraco entre 0 e 0.5: código -1 vira "(faltante)" (caminho geral)
    "num_buraco": ("x1", [{"kind": "num", "lo": -INF, "hi": 0.0},
                          {"kind": "num", "lo": 0.5, "hi": INF}]),
    "num_include_na": ("x1", [{"kind": "num", "lo": -INF, "hi": 0.0, "include_na": True},
                              {"kind": "num", "lo": 0.0, "hi": INF}]),
    "num_com_na": ("x1", [{"kind": "num", "lo": -INF, "hi": 0.0},
                          {"kind": "num", "lo": 0.0, "hi": INF}, {"kind": "na"}]),
    # rótulo repetido ({A, B} duas vezes) e faixa "na" rotulada "(faltante)"
    "cat_rotulo_repetido": ("c", [{"kind": "cat", "cats": ["A", "B"]},
                                  {"kind": "cat", "cats": ["A", "B"]},
                                  {"kind": "cat", "cats": ["C"]}, {"kind": "na"}]),
    # faixa que nenhuma linha usa (ck descarta a coluna)
    "cat_faixa_vazia": ("c", [{"kind": "cat", "cats": ["A", "B", "C"]},
                              {"kind": "cat", "cats": ["Z"]}]),
}


@pytest.fixture(scope="module")
def seg_faixa():
    df = _df()
    rng = np.random.default_rng(9)
    df["alta"] = np.array([f"nivel_{i:04d}_" + "x" * 30 for i in rng.integers(0, 2000, len(df))],
                          dtype=object)
    with _quieto():
        s = ModelSegmenter(df, target="target", sample_col="amostra", ref_sample="DES",
                           date_col="dt_ref", verbose=False)
    return s


def _todos_os_recortes(s, feature, bins):
    s.max_linhas_graficos = 3000
    for amostra in (None, "OOT"):
        for todas in (False, True):
            for graficos in (False, True):
                ctx = s._amostra_graficos() if graficos else contextlib.nullcontext()
                with ctx:
                    yield (amostra, todas, graficos,
                           s.variable_faixa_share_by_safra(feature, sample=amostra, bins=bins,
                                                           all_samples=todas),
                           _faixa_share_antigo(s, feature, sample=amostra, bins=bins,
                                               all_samples=todas))


@pytest.mark.parametrize("caso", sorted(BINS_FAIXA))
def test_faixa_share_bins_explicitos(seg_faixa, caso):
    feature, bins = BINS_FAIXA[caso]
    for amostra, todas, graficos, novo, antigo in _todos_os_recortes(seg_faixa, feature, bins):
        pd.testing.assert_frame_equal(novo, antigo, check_exact=True)
        assert novo.columns.name == antigo.columns.name


def test_faixa_share_alta_cardinalidade_explicita(seg_faixa):
    niveis = sorted(seg_faixa.df["alta"].unique())
    bins = [{"kind": "cat", "cats": niveis[i:i + 400]} for i in range(0, len(niveis), 400)]
    bins[-1] = dict(bins[-1], include_na=True)
    for *_, novo, antigo in _todos_os_recortes(seg_faixa, "alta", bins):
        pd.testing.assert_frame_equal(novo, antigo, check_exact=True)


def test_faixa_share_bins_resolvidos(seg_faixa):
    """Faixas da própria análise (optbinning): numérica no caminho rápido,
    categórica de 2000 níveis e uma derivada (create_categorical)."""
    pytest.importorskip("optbinning")
    s = seg_faixa
    with _quieto():
        der = s.create_categorical("x1", new_name="x1_cat")
    for feature in ("x1", "x2", "c", "alta", der):
        for amostra in (None, "OOT"):
            for todas in (False, True):
                novo = s.variable_faixa_share_by_safra(feature, sample=amostra,
                                                       all_samples=todas)
                antigo = _faixa_share_antigo(s, feature, sample=amostra, all_samples=todas)
                pd.testing.assert_frame_equal(novo, antigo, check_exact=True)


def test_faixa_share_sem_safra_valida():
    df = _df(n=500)
    df["dt_ref"] = pd.NaT
    with _quieto():
        s = ModelSegmenter(df, target="target", sample_col="amostra", date_col="dt_ref",
                           verbose=False)
    bins = BINS_FAIXA["num_com_na"][1]
    novo = s.variable_faixa_share_by_safra("x1", bins=bins)
    pd.testing.assert_frame_equal(novo, _faixa_share_antigo(s, "x1", bins=bins))


# ─────────────────────────── 1D. metrics_by_safra ───────────────────────────
def _metrics_by_safra_antigo(self, sample=None, time_col=None):
    time_col = time_col or self.date_col
    is_clf = self.task_type == "classification"
    met_cols = (["taxa_evento", "auc", "ks", "gini"] if is_clf
                else ["previsto_medio", "realizado_medio", "mae", "rmse", "r2"])
    idx, limites, rot = self._fatias_por_safra(time_col, sample, all_rows=not sample)
    y_all = pd.to_numeric(self.df[self.target], errors="coerce").to_numpy(dtype="float64")
    sc_all = self.score_.reindex(self.df.index).to_numpy(dtype="float64")
    rows = []
    for j, per in enumerate(rot):
        ii = idx[limites[j]:limites[j + 1]]
        if ii.size == 0:
            continue
        y, sc = y_all[ii], sc_all[ii]
        ok = ~np.isnan(y) & ~np.isnan(sc)
        y, sc = y[ok], sc[ok]
        row = {"safra": per, "n": int(y.size)}
        row.update({c: float("nan") for c in met_cols})
        if is_clf:
            row["taxa_evento"] = self._risco(y)
            if y.size >= 2 and len(np.unique(y)) == 2:
                try:
                    m = classification_metrics(y, sc)
                    row.update({c: m.get(c, float("nan")) for c in ("auc", "ks", "gini")})
                except Exception:  # noqa: BLE001
                    pass
        else:
            row["previsto_medio"] = float(np.mean(sc)) if sc.size else float("nan")
            row["realizado_medio"] = self._risco(y)
            if y.size >= 2:
                try:
                    m = regression_metrics(y, sc)
                    row.update({c: m.get(c, float("nan")) for c in ("mae", "rmse", "r2")})
                except Exception:  # noqa: BLE001
                    pass
        rows.append(row)
    return (pd.DataFrame(rows, columns=["safra", "n"] + met_cols)
            .sort_values("safra").reset_index(drop=True))


def _cenarios_metricas(s):
    """Muta alvo/score do ``s`` em cenários de borda; devolve o nome de cada um."""
    df, rng = s.df, np.random.default_rng(1)
    y0, sc0 = df["target"].copy(), s.score_.copy()
    cod, rot = s._safra_codes("dt_ref")
    safra0 = cod == 0
    safra1 = np.flatnonzero(cod == 1)
    yield "base"
    y = y0.copy()
    y[safra0] = 1.0                                   # safra de uma classe só
    y.iloc[safra1[1:]] = np.nan                       # safra com 1 linha útil
    df["target"] = y
    sc = sc0.copy()
    sc[rng.random(len(sc)) < 0.1] = np.nan            # score com NaN
    s.score_ = sc
    yield "uma_classe_1_linha_nan"
    s.score_ = sc0 * 3 - 1                            # score fora de [0,1]
    yield "score_fora_01"
    if s.task_type == "classification":
        s.score_ = sc0
        df["target"] = y0 * 2                         # alvo {0,2}
        yield "alvo_0_2"
        df["target"] = y0 * 2 - 1                     # alvo {-1,1}
        yield "alvo_m1_1"
        sc = sc0.copy()
        sc.iloc[safra1[0]] = np.inf                   # inf: roc_curve levanta
        s.score_, df["target"] = sc, y0
        yield "score_inf"
    df["target"], s.score_ = y0, sc0


def test_metrics_by_safra_igual_ao_antigo(seg_fit):
    s = seg_fit
    for cenario in _cenarios_metricas(s):
        for amostra in (None, "DES", "OOT"):
            novo = s.metrics_by_safra(sample=amostra)
            antigo = _metrics_by_safra_antigo(s, sample=amostra)
            pd.testing.assert_frame_equal(novo, antigo, check_exact=True, obj=cenario)
        if cenario == "alvo_m1_1":                    # caminho antigo: f1 levanta ⇒ NaN
            assert novo["auc"].isna().all()


def test_metrics_by_safra_chama_roc_pack_uma_vez_por_safra(monkeypatch):
    s = _seg()
    chamadas = []
    orig = segmod._roc_pack
    monkeypatch.setattr(segmod, "_roc_pack", lambda y, sc: chamadas.append(1) or orig(y, sc))
    ms = s.metrics_by_safra()
    assert len(chamadas) == int(ms["auc"].notna().sum()) == len(ms)
    # plot com a tabela pronta não recalcula
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = s.plot_metrics_by_safra(table=ms)
    assert len(chamadas) == len(ms)
    fig2 = s.plot_metrics_by_safra()
    assert len(chamadas) == 2 * len(ms)
    linhas = [l.get_ydata() for l in fig.axes[0].lines]
    linhas2 = [l.get_ydata() for l in fig2.axes[0].lines]
    assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(linhas, linhas2))
    plt.close(fig); plt.close(fig2)


def test_ui_msafra_um_calculo_por_clique(monkeypatch):
    pytest.importorskip("ipywidgets")
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    with _quieto():
        ui = ModelSegmenterUI(_df(n=3000), target="target", sample_col="amostra",
                              ref_sample="DES", date_col="dt_ref")
        ui.seg.fit("logistica", features=["x1", "x2"])
    n_ms, n_pack = [], []
    orig_ms, orig_pack = ui.seg.metrics_by_safra, segmod._roc_pack
    monkeypatch.setattr(ui.seg, "metrics_by_safra",
                        lambda *a, **k: n_ms.append(1) or orig_ms(*a, **k))
    monkeypatch.setattr(segmod, "_roc_pack",
                        lambda y, sc: n_pack.append(1) or orig_pack(y, sc))
    with _quieto():
        ui._on_adv_msafra(None)
    assert len(n_ms) == 1
    assert len(n_pack) == len(ui.seg._safra_codes("dt_ref")[1])
    assert "<img" in ui.out_adv_msafra_fig.value and ui.out_adv_msafra_tab.value


# ─────────────────────────── 1E. backtest ───────────────────────────
def _backtest_antigo(self, time_col=None, sample=None, tol=None):
    time_col = time_col or self.date_col
    base = self._frame(sample) if sample else self.df
    sc = self.score_.reindex(base.index)
    safra = pd.to_datetime(base[time_col], errors="coerce").dt.to_period("M")
    rows = []
    for per, g in base.groupby(safra):
        sg = sc.reindex(g.index)
        prev = float(sg.mean(skipna=True))
        real = self._risco(g[self.target])
        gap = real - prev if (np.isfinite(prev) and np.isfinite(real)) else np.nan
        t = self._calib_test(g[self.target], sg)
        if tol is not None:
            status = "ok" if (np.isfinite(gap) and abs(gap) <= tol) else "alerta"
        else:
            status = t["status"]
        rows.append({"safra": str(per), "n": len(g),
                     "previsto_medio": round(prev, 4) if np.isfinite(prev) else np.nan,
                     "realizado_medio": round(real, 4) if np.isfinite(real) else np.nan,
                     "gap": round(gap, 4) if np.isfinite(gap) else np.nan,
                     "ic_low": round(t["ic_low"], 4) if np.isfinite(t["ic_low"]) else np.nan,
                     "ic_high": round(t["ic_high"], 4) if np.isfinite(t["ic_high"]) else np.nan,
                     "p_valor": round(t["p_valor"], 4) if np.isfinite(t["p_valor"]) else np.nan,
                     "status": status})
    return pd.DataFrame(rows).sort_values("safra").reset_index(drop=True)


def test_backtest_igual_ao_antigo(seg_fit):
    s = seg_fit
    s.max_linhas_graficos = 3000
    y0, sc0 = s.df["target"].copy(), s.score_.copy()
    alvos = {"float": y0}
    if s.task_type == "classification":
        alvos["Int64_na"] = pd.array([None if i % 37 == 0 else int(v)
                                      for i, v in enumerate(y0)], dtype="Int64")
        alvos["bool"] = y0.astype(bool)
        alvos["decimal"] = pd.Series([D(int(v)) for v in y0], index=y0.index, dtype=object)
    sc_nan = sc0.copy()
    sc_nan.iloc[::23] = np.nan
    try:
        for nome, alvo in alvos.items():
            s.df["target"] = alvo
            for score in (sc0, sc_nan):
                s.score_ = score
                for amostra in (None, "DES", "OOT"):
                    for tol in (None, 0.03):
                        antigo = _backtest_antigo(s, sample=amostra, tol=tol)
                        novo = s.backtest(sample=amostra, tol=tol)
                        pd.testing.assert_frame_equal(novo, antigo, check_exact=True,
                                                      obj=f"{nome}/{amostra}/{tol}")
                        with s._amostra_graficos() as ativa:
                            assert ativa
                            dentro = s.backtest(sample=amostra, tol=tol)
                        pd.testing.assert_frame_equal(dentro, antigo, check_exact=True)
            # a safra só-DES não aparece com sample='OOT'
            assert "2024-01" not in set(s.backtest(sample="OOT")["safra"])
            assert "2024-01" in set(s.backtest()["safra"])
    finally:
        s.df["target"], s.score_ = y0, sc0


def test_backtest_pico_de_memoria_abaixo_da_base():
    s = _seg(n=40000)
    for c in range(30):                               # base larga (100 colunas na vida real)
        s.df[f"extra_{c}"] = np.arange(len(s.df), dtype="float64")
    s.backtest()                                      # aquece caches (safras, fatias)
    tracemalloc.start()
    try:
        s.backtest()
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert pico < s.df.memory_usage(deep=False).sum()


def test_backtest_indice_duplicado_passa_a_responder():
    s = _seg(n=3000)
    s.df.index = [0] * len(s.df)
    s.score_.index = s.df.index
    with pytest.raises(ValueError):
        _backtest_antigo(s, sample="OOT")
    bt = s.backtest(sample="OOT")
    assert int(bt["n"].sum()) == int(((s.df["amostra"] == "OOT")
                                      & s.df["dt_ref"].notna()).sum())


# ─────────────────────────── 1F. hosmer_lemeshow ───────────────────────────
def _hosmer_antigo(self, sample=None, n_groups=10):
    from scipy.stats import chi2
    base = self._frame(sample) if sample else self.df
    sc = self.score_.reindex(base.index)
    y = base[self.target]
    ok = sc.notna() & y.notna()
    sc, y = sc[ok], y[ok].astype("float64")
    if len(sc) == 0:
        raise ValueError("Sem linhas válidas (score e alvo) para o teste.")
    grp = pd.qcut(sc, q=int(n_groups), duplicates="drop")
    rows, stat = [], 0.0
    for faixa, idx in sc.groupby(grp, observed=True).groups.items():
        s_g = sc.loc[idx].to_numpy(dtype="float64")
        y_g = y.loc[idx].to_numpy(dtype="float64")
        n = int(s_g.size)
        obs = float(y_g.sum())
        esp = float(s_g.sum())
        p_bar = esp / n if n else float("nan")
        den = esp * (1.0 - p_bar)
        contrib = ((obs - esp) ** 2 / den) if den > 0 else 0.0
        stat += contrib
        rows.append({"faixa": str(faixa), "n": n, "eventos_obs": int(round(obs)),
                     "eventos_esp": round(esp, 1),
                     "taxa_obs": round(obs / n, 4) if n else np.nan,
                     "score_medio": round(float(s_g.mean()), 4),
                     "contrib": round(contrib, 3)})
    g_eff = len(rows)
    df_hl = max(g_eff - 2, 1)
    return {"statistic": float(round(stat, 4)), "p_value": float(chi2.sf(stat, df_hl)),
            "df": int(df_hl), "n_groups": int(g_eff), "table": pd.DataFrame(rows)}


def _hl_igual(novo, antigo):
    for k in ("statistic", "p_value", "df", "n_groups"):
        assert novo[k] == antigo[k], k
    pd.testing.assert_frame_equal(novo["table"], antigo["table"], check_exact=True)


def _score_em_degraus(n, rng):
    """Patamar 0,1 que termina exatamente na posição do decil 4 (0,4·(n−1),
    com parte fracionária): a aresta seguinte é interpolada entre 0,1 e o
    próximo valor, e a faixa (0,1, e₄] fica SEM linhas (observed=True a
    descarta)."""
    k = int(np.floor(0.4 * (n - 1))) + 1              # nº de valores ≤ 0,1
    baixo = n // 8
    v = np.concatenate([np.sort(rng.random(baixo)) * 0.05, np.full(k - baixo, 0.10),
                        0.2 + rng.random(n - k) * 0.7])
    return v[rng.permutation(n)]


def test_hosmer_lemeshow_igual_ao_antigo():
    s = _seg()
    rng = np.random.default_rng(5)
    y0, sc0 = s.df["target"].copy(), s.score_.copy()
    n = len(s.df)
    degraus = pd.Series(_score_em_degraus(n, rng), index=s.df.index)
    cat = pd.qcut(degraus, 10, duplicates="drop")
    assert (cat.value_counts() == 0).any()            # o caso existe de fato
    sc_nan = sc0.copy()
    sc_nan.iloc[::31] = np.nan
    int64 = pd.Series(pd.array([None if i % 29 == 0 else int(v) for i, v in enumerate(y0)],
                               dtype="Int64"), index=y0.index)
    try:
        for alvo in (y0, int64):
            s.df["target"] = alvo
            for score in (sc0, sc_nan, degraus):
                s.score_ = score
                for amostra in (None, "OOT"):
                    for g in (10, 4):
                        _hl_igual(s.hosmer_lemeshow(sample=amostra, n_groups=g),
                                  _hosmer_antigo(s, sample=amostra, n_groups=g))
    finally:
        s.df["target"], s.score_ = y0, sc0


def test_hosmer_lemeshow_indice_embaralhado_e_texto():
    s = _seg(n=4000)
    perm = np.random.default_rng(2).permutation(len(s.df))
    for novo_idx in (s.df.index[perm], pd.Index([f"k{i}" for i in perm])):
        s.df.index = novo_idx
        s.score_ = pd.Series(s.score_.to_numpy(), index=novo_idx)
        s._mask_cache.clear()
        for amostra in (None, "OOT"):
            _hl_igual(s.hosmer_lemeshow(sample=amostra), _hosmer_antigo(s, sample=amostra))
        # score com índice na outra ordem: cai no reindex (alinha por rótulo)
        s.score_ = s.score_.iloc[::-1]
        _hl_igual(s.hosmer_lemeshow(), _hosmer_antigo(s))


def test_hosmer_lemeshow_indice_duplicado_mantem_o_antigo():
    s = _seg(n=3000)
    s.df.index = [i // 2 for i in range(len(s.df))]
    s.score_.index = s.df.index
    _hl_igual(s.hosmer_lemeshow(), _hosmer_antigo(s))
