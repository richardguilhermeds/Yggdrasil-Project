"""
Testes do subpacote yggdrasil.credit_risk.model (segmentação orientada a modelo,
unificando classificação e regressão via ``task_type``).

Cobrem, para AMBOS os task_types: construção e detecção de variáveis, análise
univariada (IV/logodds/tabela/inversão), seleção/categorização, ajuste (logística/
linear e random_forest), métricas, score→ratings (nº escolhido pelo usuário),
tabela e inversão de ratings, validação (monotonia/PSI/backtest), predict em dados
novos, save/load round-trip, SHAP best-effort e a UI (gated por ipywidgets).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from yggdrasil.credit_risk.model import ModelSegmenter


def _synthetic(task: str, n: int = 2000, seed: int = 0, com_na: bool = False,
               com_cat: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    meses = pd.date_range("2023-01-01", periods=10, freq="MS")
    if task == "classification":
        X, y = make_classification(n_samples=n, n_features=6, n_informative=4,
                                   weights=[0.75], random_state=seed)
    else:
        X, y = make_regression(n_samples=n, n_features=6, n_informative=4,
                               noise=10.0, random_state=seed)
        y = (y - y.min()) / (y.max() - y.min())
    df = pd.DataFrame(X, columns=[f"feat_{i:02d}" for i in range(6)])
    df["target"] = y
    df["dt_ref"] = rng.choice(meses, size=n)
    df["amostra"] = np.where(df["dt_ref"] >= meses[7], "OOT", "DES")
    if com_cat:
        df["feat_cat"] = rng.choice(["A", "B", "C", "D"], size=n, p=[.4, .3, .2, .1])
    if com_na:
        df.loc[df.sample(frac=0.08, random_state=1).index, "feat_00"] = np.nan
    return df


@pytest.fixture(params=["classification", "regression"])
def task(request):
    return request.param


@pytest.fixture
def seg(task):
    df = _synthetic(task, com_cat=True)
    return ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                          ref_sample="DES", date_col="dt_ref", verbose=False)


def _default_algo(task):
    return "logistica" if task == "classification" else "linear"


# ----------------------------------------------------------------------
# Pacote / construção
# ----------------------------------------------------------------------
def test_import_pacote():
    import yggdrasil
    from yggdrasil.credit_risk import ModelSegmenter as C
    assert C is ModelSegmenter
    assert isinstance(yggdrasil.__version__, str)


def test_construcao_e_candidatas(seg):
    # alvo, amostra e a coluna de data (datetime) ficam fora das candidatas
    assert "target" not in seg.candidates
    assert "amostra" not in seg.candidates and "dt_ref" not in seg.candidates
    assert "feat_00" in seg.candidates and "feat_cat" in seg.candidates
    assert seg.included == set(seg.candidates)


def test_task_type_invalido():
    df = _synthetic("classification")
    with pytest.raises(ValueError):
        ModelSegmenter(df, target="target", task_type="foo")


def test_ref_sample_inexistente():
    df = _synthetic("classification")
    with pytest.raises(ValueError, match="referência"):
        ModelSegmenter(df, target="target", task_type="classification",
                       sample_col="amostra", ref_sample="ZZZ", verbose=False)


# ----------------------------------------------------------------------
# Análise univariada
# ----------------------------------------------------------------------
def test_variable_iv_ranking(seg):
    rk = seg.variable_iv()
    assert {"variavel", "tipo", "n_bins", "iv", "forca", "tendencia",
            "n_inversoes", "incluida"}.issubset(rk.columns)
    assert "psi_OOT" in rk.columns and "estabilidade" in rk.columns
    # ordenado por IV desc
    ivs = rk["iv"].dropna().to_numpy()
    assert (np.diff(ivs) <= 1e-9).all()


def test_variable_table_logodds_por_task(seg):
    vt = seg.variable_table("feat_00")
    assert {"faixa", "n", "repr_%"}.issubset(vt.columns)
    if seg.task_type == "classification":
        assert {"woe", "logodds", "event_rate", "iv_parcial"}.issubset(vt.columns)
    else:
        assert {"alvo_medio", "iv_parcial"}.issubset(vt.columns)
        assert "logodds" not in vt.columns
    assert np.isfinite(vt.attrs["iv"])


def test_variable_summary(seg):
    s = seg.variable_summary("feat_00")
    assert s["tipo"] == "num" and "pct_missing" in s
    assert s["iv"] is not None and s["forca"] != "—"
    s_cat = seg.variable_summary("feat_cat")
    assert s_cat["tipo"] == "cat" and "top_categorias" in s_cat


def test_variable_inversion(seg):
    inv = seg.variable_inversion("feat_00")
    assert inv["status"] in ("green", "yellow", "red")
    assert any(r["amostra"] == "DES" for r in inv["samples"])
    # DES (a própria referência) não inverte vs. si mesma
    des = [r for r in inv["samples"] if r["amostra"] == "DES"][0]
    assert des["n_inv"] == 0


def test_variable_psi_by_safra(seg):
    ps = seg.variable_psi_by_safra("feat_00")
    assert {"safra", "n", "psi", "classificacao"}.issubset(ps.columns)
    assert len(ps) >= 1


# ----------------------------------------------------------------------
# Seleção / categorização
# ----------------------------------------------------------------------
def test_include_exclude_categoria(seg):
    seg.clear_features()
    assert seg.selected_features() == []
    seg.include("feat_00").include("feat_01")
    assert set(seg.selected_features()) == {"feat_00", "feat_01"}
    seg.exclude("feat_00")
    assert seg.selected_features() == ["feat_01"]
    seg.set_category("feat_01", "manter")
    assert seg.var_meta["feat_01"]["categoria"] == "manter"
    with pytest.raises(ValueError):
        seg.include("nao_existe")


def test_auto_select(seg):
    rk = seg.auto_select(min_iv=0.0)
    assert not rk.empty
    # com min_iv=0 todas as features com IV finito entram
    assert len(seg.selected_features()) >= 1
    cats = {seg.var_meta[f]["categoria"] for f in seg.candidates}
    assert cats <= {"manter", "descartar"}


# ----------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------
def test_fit_linear_e_score(seg):
    seg.fit(_default_algo(seg.task_type))
    assert seg.score_ is not None and len(seg.score_) == len(seg.df)
    assert seg.score_.notna().all()
    if seg.task_type == "classification":
        assert seg.score_.between(0, 1).all()


def test_fit_random_forest_e_metrics(seg):
    seg.fit("random_forest", hyperparams={"n_estimators": 40})
    m = seg.metrics()
    assert "amostra" in m.columns and len(m) >= 1
    if seg.task_type == "classification":
        assert {"auc", "gini", "ks", "f1"}.issubset(m.columns)
        des = m[m["amostra"] == "DES"].iloc[0]
        assert abs(des["gini"] - (2 * des["auc"] - 1)) < 1e-6
        assert des["auc"] > 0.6                       # discrimina acima do acaso
    else:
        assert {"rmse", "mae", "r2"}.issubset(m.columns)
        assert m[m["amostra"] == "DES"].iloc[0]["r2"] > 0.3


def test_set_model_externo(seg):
    from sklearn.dummy import DummyClassifier, DummyRegressor
    feats = ["feat_00", "feat_01"]
    dev = seg.df[seg.df["amostra"] == "DES"]
    if seg.task_type == "classification":
        mdl = DummyClassifier(strategy="prior").fit(dev[feats], dev["target"].astype(int))
    else:
        mdl = DummyRegressor().fit(dev[feats], dev["target"])
    seg.set_model(mdl, features=feats)
    assert seg.score_ is not None and len(seg.score_) == len(seg.df)


def test_metric_shifts(seg):
    seg.fit(_default_algo(seg.task_type))
    sh = seg.metric_shifts()
    assert isinstance(sh, dict) and len(sh) >= 1


def test_metrics_exige_modelo(seg):
    with pytest.raises(RuntimeError):
        seg.metrics()


# ----------------------------------------------------------------------
# Score → Ratings
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ["decis", "quantil", "arvore", "optbin"])
def test_build_ratings_metodos(seg, method):
    seg.fit(_default_algo(seg.task_type))
    seg.build_ratings(method=method, n_ratings=6)
    assert seg.rating_ is not None and len(seg.rating_labels_) >= 1
    assert len(seg.rating_labels_) <= 10
    # rótulos cobrem todas as linhas com score
    assert seg.rating_.notna().sum() >= int(0.9 * len(seg.df))


def test_n_ratings_controla_numero(seg):
    seg.fit(_default_algo(seg.task_type))
    seg.build_ratings(method="decis", n_ratings=5)
    assert len(seg.rating_labels_) == 5         # decis puros: sem fusão
    seg.build_ratings(method="decis", n_ratings=8)
    assert len(seg.rating_labels_) == 8


def test_rating_table_e_inversao(seg):
    seg.fit("random_forest", hyperparams={"n_estimators": 40})
    seg.build_ratings(method="quantil", n_ratings=8)
    tab = seg.rating_table()
    assert {"rating", "n", "repr_%"}.issubset(tab.columns)
    prefix = "event_rate" if seg.task_type == "classification" else "alvo"
    assert f"{prefix}_DES" in tab.columns and f"{prefix}_OOT" in tab.columns
    inv = seg.rating_inversion()
    assert inv["status"] in ("green", "yellow", "red")
    assert len(inv["ordered"]) == len(seg.rating_labels_)


def test_build_ratings_exige_score(seg):
    with pytest.raises(RuntimeError):
        seg.build_ratings()


# ----------------------------------------------------------------------
# Validação / estabilidade
# ----------------------------------------------------------------------
def test_monotonicity_report(seg):
    seg.fit(_default_algo(seg.task_type))            # score contínuo (decis íntegros)
    seg.build_ratings(method="decis", n_ratings=10)
    mr = seg.monotonicity_report()
    assert {"amostra", "monotonico", "tendencia", "n_inversoes"}.issubset(mr.columns)
    assert (mr["amostra"] == "DES").any() and (mr["amostra"] == "OOT").any()
    des = mr[mr["amostra"] == "DES"].iloc[0]
    assert int(des["n_inversoes"]) >= 0
    assert des["tendencia"] in ("crescente", "decrescente", "não-monotônica", "—")
    # consistência: 'monotonico' equivale a 'sem inversões'
    assert bool(des["monotonico"]) == (int(des["n_inversoes"]) == 0)


def test_monotonicity_detecta_inversao():
    """Constrói um score com ordem de risco INVERTIDA no OOT e confirma a detecção."""
    rng = np.random.default_rng(4)
    n = 6000
    x = rng.uniform(0, 1, n)
    amostra = rng.choice(["DES", "OOT"], n, p=[.6, .4])
    p = np.where(x < 0.5, 0.15, 0.55)
    oot = amostra == "OOT"
    p = np.where(oot & (x < 0.5), 0.6, p)            # inverte os níveis no OOT
    p = np.where(oot & (x >= 0.5), 0.2, p)
    target = (rng.uniform(0, 1, n) < p).astype(int)
    df = pd.DataFrame({"feat_x": x, "target": target, "amostra": amostra})
    seg = ModelSegmenter(df, target="target", task_type="classification",
                         sample_col="amostra", ref_sample="DES", verbose=False)
    seg.set_model(_IdentityModel("feat_x"), features=["feat_x"])  # score = x
    seg.build_ratings(method="decis", n_ratings=4)
    mr = seg.monotonicity_report().set_index("amostra")
    assert bool(mr.loc["DES", "monotonico"]) is True
    assert bool(mr.loc["OOT", "monotonico"]) is False
    assert int(mr.loc["OOT", "n_inversoes"]) >= 1


class _IdentityModel:
    """Modelo trivial: o 'score' é a própria feature (para testes determinísticos)."""
    def __init__(self, col):
        self.col = col

    def predict_proba(self, X):
        s = np.asarray(X[self.col], dtype=float)
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return np.asarray(X[self.col], dtype=float)


def test_psi_ratings(seg):
    seg.fit(_default_algo(seg.task_type))
    seg.build_ratings(method="decis", n_ratings=10)
    psi = seg.psi()
    assert {"amostra", "psi", "classificacao"}.issubset(psi.columns)
    assert (psi["amostra"] == "OOT").any()


def test_backtest(seg):
    seg.fit(_default_algo(seg.task_type))
    bt = seg.backtest("dt_ref")
    assert {"safra", "n", "previsto_medio", "realizado_medio",
            "gap", "status"}.issubset(bt.columns)
    assert int(bt["n"].sum()) == len(seg.df)
    with pytest.raises(ValueError):
        seg.backtest("coluna_inexistente")


# ----------------------------------------------------------------------
# Predict / assign / persistência
# ----------------------------------------------------------------------
def test_predict_dados_novos(seg):
    seg.fit(_default_algo(seg.task_type))
    seg.build_ratings(method="quantil", n_ratings=6)
    novos = _synthetic(seg.task_type, n=300, seed=9, com_cat=True)
    pr = seg.predict(novos)
    assert {"score", "rating"}.issubset(pr.columns)
    assert pr["score"].notna().all()
    assert pr["rating"].notna().mean() > 0.9


def test_assign(seg):
    seg.fit(_default_algo(seg.task_type))
    seg.build_ratings(method="decis", n_ratings=5)
    out = seg.assign()
    assert {"score", "rating"}.issubset(out.columns)
    assert len(out) == len(seg.df)


def test_save_load_roundtrip(seg, tmp_path):
    seg.fit("random_forest", hyperparams={"n_estimators": 40})
    seg.build_ratings(method="quantil", n_ratings=8)
    p = tmp_path / "modelo.json"
    seg.save(str(p))
    assert '"yggdrasil.credit_risk.model' in p.read_text(encoding="utf-8")
    assert (tmp_path / "modelo.json.model.joblib").exists()
    seg2 = ModelSegmenter(seg.df, target="target", task_type=seg.task_type,
                          sample_col="amostra", ref_sample="DES",
                          date_col="dt_ref", verbose=False).load(str(p), seg.df)
    assert np.allclose(seg.score_.values, seg2.score_.values, equal_nan=True)
    assert seg2.rating_labels_ == seg.rating_labels_
    a = seg.predict(seg.df.head(200))["score"].to_numpy()
    b = seg2.predict(seg.df.head(200))["score"].to_numpy()
    assert np.allclose(a, b, equal_nan=True)


# ----------------------------------------------------------------------
# SHAP (best-effort) e sem sample_col
# ----------------------------------------------------------------------
def test_shap_importance(seg):
    seg.fit("random_forest", hyperparams={"n_estimators": 40})
    imp = seg.shap_importance(sample_size=300)
    assert {"feature", "mean_abs_shap"}.issubset(imp.columns)
    assert len(imp) >= 1 and (imp["mean_abs_shap"] >= 0).all()


def test_sem_sample_col():
    df = _synthetic("classification").drop(columns=["amostra"])
    seg = ModelSegmenter(df, target="target", task_type="classification",
                         date_col="dt_ref", verbose=False)
    seg.fit("logistica")
    seg.build_ratings(method="decis", n_ratings=5)
    assert len(seg.rating_labels_) == 5
    assert not seg.metrics().empty


# ----------------------------------------------------------------------
# Plots (matplotlib Agg)
# ----------------------------------------------------------------------
def test_plots_modelo(seg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seg.fit("random_forest", hyperparams={"n_estimators": 40})
    if seg.task_type == "classification":
        for f in (seg.plot_roc(), seg.plot_ks(), seg.plot_score_distribution(),
                  seg.plot_calibration()):
            assert f is not None; plt.close(f)
    else:
        for f in (seg.plot_score_distribution(), seg.plot_calibration(),
                  seg.plot_residuals()):
            assert f is not None; plt.close(f)


def test_plots_variavel(seg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for f in (seg.plot_variable_logodds("feat_00"),
              seg.plot_variable_distribution("feat_00"),
              seg.plot_variable_inversion_by_sample("feat_00"),
              seg.plot_variable_timeseries("feat_00")):
        assert f is not None; plt.close(f)


def test_plots_rating(seg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seg.fit(_default_algo(seg.task_type))
    seg.build_ratings(method="quantil", n_ratings=8)
    for f in (seg.plot_rating_badrate(), seg.plot_rating_distribution(),
              seg.plot_rating_inversion_by_sample()):
        assert f is not None; plt.close(f)


# ----------------------------------------------------------------------
# UI (gated por ipywidgets)
# ----------------------------------------------------------------------
def test_ui_layout_abas(task):
    pytest.importorskip("ipywidgets")
    import contextlib
    import io
    import ipywidgets as W
    from yggdrasil.credit_risk.model import ModelSegmenterUI

    df = _synthetic(task, com_cat=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ui = ModelSegmenterUI(df, target="target", task_type=task,
                              sample_col="amostra", ref_sample="DES", date_col="dt_ref")
    ch = list(ui.panel.children)
    tabs = next(c for c in ch if isinstance(c, W.Tab))
    titulos = [tabs.get_title(i) for i in range(len(tabs.children))]
    assert len(titulos) == 5
    assert any("Variáveis" in t or "variáveis" in t for t in titulos)
    assert any("Modelo" in t for t in titulos)
    assert any("Rating" in t for t in titulos)


def test_ui_fluxo_treina_e_ratings(task):
    pytest.importorskip("ipywidgets")
    import contextlib
    import io
    from yggdrasil.credit_risk.model import ModelSegmenterUI

    df = _synthetic(task, com_cat=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ui = ModelSegmenterUI(df, target="target", task_type=task,
                              sample_col="amostra", ref_sample="DES", date_col="dt_ref")
        ui._on_fit(None)
        assert ui.seg.score_ is not None
        ui._on_build_ratings(None)
        assert ui.seg.rating_ is not None
