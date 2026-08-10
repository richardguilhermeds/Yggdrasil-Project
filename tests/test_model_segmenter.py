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


def test_variable_faixa_share_by_safra_num_e_cat(seg):
    # % de cada faixa/categoria ao longo do tempo — numérica (faixas) e cat (grupos)
    shn = seg.variable_faixa_share_by_safra("feat_00")
    assert "safra" in shn.columns and shn.shape[1] >= 2
    faixas = [c for c in shn.columns if c != "safra"]
    # cada safra soma ~100% entre as faixas
    soma = shn[faixas].sum(axis=1)
    assert ((soma - 100).abs() < 0.5).all()
    shc = seg.variable_faixa_share_by_safra("feat_cat")   # 'seg' já tem feat_cat
    assert "safra" in shc.columns and shc.shape[1] >= 2


def test_plot_variable_faixa_share_timeseries_render(seg):
    import matplotlib
    matplotlib.use("Agg")
    fig = seg.plot_variable_faixa_share_timeseries("feat_00")
    ax = fig.axes[0]
    # uma linha por faixa e eixo X em mmm/aa (padrão do repo)
    assert len(ax.get_lines()) >= 2
    labs = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert any("/" in s for s in labs)


def test_optbin_share_numerica_ignora_bins_manuais(seg):
    # Gráfico das faixas do OPTIMAL BINNING no tempo: sempre optbin, mesmo com
    # bins manuais definidos (só numéricas).
    import matplotlib
    matplotlib.use("Agg")
    n_opt = len(seg._optbin_numeric_bins("feat_00"))
    assert n_opt >= 2
    seg.set_manual_bins("feat_00", [0.0])              # 2 faixas manuais
    assert len(seg._optbin_numeric_bins("feat_00")) == n_opt   # optbin não muda
    fig = seg.plot_variable_optbin_share_timeseries("feat_00")
    ax = fig.axes[0]
    assert len(ax.get_lines()) == len([b for b in seg._optbin_numeric_bins("feat_00")
                                       if b["kind"] == "num"]) or len(ax.get_lines()) >= 2


def test_optbin_share_categorica_placeholder(seg):
    import matplotlib
    matplotlib.use("Agg")
    fig = seg.plot_variable_optbin_share_timeseries("feat_cat")
    txts = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "numéric" in txts.lower()


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


def test_manual_bins_numerico(seg):
    # bins manuais sobrepõem o binning ótimo em toda a análise univariada
    seg.set_manual_bins("feat_00", "-0.5, 0.5")
    assert seg.manual_bins("feat_00") == [-0.5, 0.5]
    assert seg.manual_bins_spec("feat_00") == "-0.5, 0.5"
    vt = seg.variable_table("feat_00")
    faixas = [f for f in vt["faixa"] if f != "(faltante)"]
    assert faixas == ["(-inf, -0.5]", "(-0.5, 0.5]", "(0.5, inf]"]
    # marcado no ranking
    rk = seg.variable_iv().set_index("variavel")
    assert bool(rk.loc["feat_00", "bins_manuais"]) is True
    # limpar volta ao ótimo
    seg.clear_manual_bins("feat_00")
    assert seg.manual_bins("feat_00") is None
    assert seg.manual_bins_spec("feat_00") == ""


def test_manual_bins_categorico(seg):
    seg.set_manual_bins("feat_cat", "A,B; C,D")
    assert seg.manual_bins("feat_cat") == [["A", "B"], ["C", "D"]]
    vt = seg.variable_table("feat_cat")
    faixas = [f for f in vt["faixa"] if f != "(faltante)"]
    assert faixas == ["{A, B}", "{C, D}"]
    # texto vazio limpa
    seg.set_manual_bins("feat_cat", "")
    assert seg.manual_bins("feat_cat") is None
    with pytest.raises(ValueError):
        seg.set_manual_bins("nao_existe", "1,2")


def test_manual_bins_persistencia(seg, tmp_path):
    seg.set_manual_bins("feat_00", "-0.5, 0.5")
    seg.set_manual_bins("feat_cat", "A; B,C,D")
    p = tmp_path / "m.json"
    seg.save(str(p))
    seg2 = ModelSegmenter(seg.df, target="target", task_type=seg.task_type,
                          sample_col="amostra", ref_sample="DES",
                          date_col="dt_ref", verbose=False).load(str(p), seg.df)
    assert seg2.manual_bins("feat_00") == [-0.5, 0.5]
    assert seg2.manual_bins("feat_cat") == [["A"], ["B", "C", "D"]]


def test_model_formula(seg):
    seg.fit(_default_algo(seg.task_type))
    coefs = seg.model_coefficients()
    assert {"termo", "coef"}.issubset(coefs.columns)
    assert "intercept" in coefs.attrs
    # ordenado por |coef| desc
    abs_coef = coefs["coef"].abs().to_numpy()
    assert (np.diff(abs_coef) <= 1e-9).all()
    fm = seg.model_formula()
    assert {"intercept", "coef", "z_expr", "text", "latex"}.issubset(fm)
    assert isinstance(fm["z_expr"], str) and fm["z_expr"]
    if seg.task_type == "classification":
        assert "odds_ratio" in coefs.columns
        assert "logit" in fm["latex"]
    else:
        assert "odds_ratio" not in coefs.columns


def test_model_formula_nao_linear(seg):
    seg.fit("random_forest", hyperparams={"n_estimators": 30})
    with pytest.raises(ValueError, match="Logística|Linear"):
        seg.model_coefficients()


def test_auto_select(seg):
    rk = seg.auto_select(min_iv=0.0)
    assert not rk.empty
    # com min_iv=0 todas as features com IV finito entram
    assert len(seg.selected_features()) >= 1
    cats = {seg.var_meta[f]["categoria"] for f in seg.candidates}
    assert cats <= {"manter", "descartar"}


def test_auto_categorize(seg):
    rk = seg.auto_categorize()
    assert {"categoria", "motivo"}.issubset(rk.columns)
    assert set(rk["categoria"]) <= {"manter", "revisar", "descartar"}
    assert (rk["motivo"].astype(bool)).all()                 # toda linha tem justificativa
    # categoria é só triagem: não altera a seleção por padrão
    assert seg.included == set(seg.candidates)
    # 'motivo' passa a aparecer no ranking exibido
    assert "motivo" in seg.variable_iv().columns
    # set_category manual limpa o motivo automático
    f0 = seg.candidates[0]
    seg.set_category(f0, "manter")
    assert "motivo" not in seg.var_meta[f0]


def test_auto_categorize_apply_selection(seg):
    seg.auto_categorize(min_iv=0.0, apply_selection=True)
    mantidas = {f for f in seg.candidates if seg.var_meta[f]["categoria"] == "manter"}
    assert seg.included == mantidas


# ----------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------
def test_fit_linear_e_score(seg):
    seg.fit(_default_algo(seg.task_type))
    assert seg.score_ is not None and len(seg.score_) == len(seg.df)
    assert seg.score_.notna().all()
    if seg.task_type == "classification":
        assert seg.score_.between(0, 1).all()


def test_fit_woe_transform(seg):
    """Treino com variáveis transformadas (WoE/risco por bins), reaproveitando os
    bins da análise univariada; score válido e fórmula com termos WoE(...)/bin(...)."""
    algo = _default_algo(seg.task_type)
    seg.fit(algo, transform="woe")
    assert seg.feature_transform == "woe"
    assert seg.score_ is not None and seg.score_.notna().all()
    if seg.task_type == "classification":
        assert seg.score_.between(0, 1).all()
    # termos da fórmula vêm embrulhados conforme o task_type
    wrap = "WoE(" if seg.task_type == "classification" else "bin("
    coefs = seg.model_coefficients()
    assert all(t.startswith(wrap) and t.endswith(")") for t in coefs["termo"])
    # re-binagem de dados novos no predict (score sem NaN)
    novo = seg.df[seg.model_features].head(20).copy()
    pr = seg.predict(novo)
    assert "score" in pr.columns and pr["score"].notna().all()


def test_woe_transform_reusa_bins_manuais(seg):
    if seg.task_type != "classification":
        pytest.skip("WoE manual-bin check focado em classificação")
    feat = "feat_00"
    seg.set_manual_bins(feat, "0.0")              # 2 faixas: (-inf,0] e (0,inf]
    seg.included = {feat}
    enc = seg._bin_encoding(feat)
    # os bins usados na transformação são os manuais (2 faixas numéricas)
    faixas = [b for b in enc["bins"] if b[0]["kind"] == "num"]
    assert len(faixas) == 2
    seg.fit("logistica", transform="woe")
    assert seg._compute_score(seg.df).notna().all()


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


# algoritmos nativos do scikit-learn (sempre disponíveis)
_SKLEARN_ML_ALGOS = ["extra_trees", "hist_gradient_boosting"]
# algoritmos de pacotes opcionais (gated por importorskip)
_OPTIONAL_ALGOS = [("lightgbm", "lightgbm"), ("xgboost", "xgboost"),
                   ("catboost", "catboost")]


@pytest.mark.parametrize("algo", _SKLEARN_ML_ALGOS)
def test_fit_algoritmos_sklearn(seg, algo):
    seg.fit(algo, hyperparams={"n_estimators": 40})
    assert seg.score_ is not None and seg.score_.notna().all()
    des = seg.metrics().query("amostra == 'DES'").iloc[0]
    if seg.task_type == "classification":
        assert seg.score_.between(0, 1).all()
        assert des["auc"] > 0.6
    else:
        assert des["r2"] > 0.3


@pytest.mark.parametrize("algo,module", _OPTIONAL_ALGOS)
def test_fit_algoritmos_boosting_opcionais(seg, algo, module):
    pytest.importorskip(module)
    seg.fit(algo, hyperparams={"n_estimators": 50, "learning_rate": 0.1})
    assert seg.score_ is not None and seg.score_.notna().all()
    des = seg.metrics().query("amostra == 'DES'").iloc[0]
    if seg.task_type == "classification":
        assert seg.score_.between(0, 1).all()
        assert des["auc"] > 0.6
    else:
        assert des["r2"] > 0.3


def test_algoritmo_opcional_ausente_erro_amigavel(seg):
    # sem o pacote instalado, o erro orienta a instalação do extra correto
    import importlib.util
    if importlib.util.find_spec("lightgbm") is not None:
        pytest.skip("lightgbm instalado; erro de ausência não se aplica")
    with pytest.raises(ImportError, match=r"yggdrasil\[lgbm\]"):
        seg.fit("lightgbm", hyperparams={"n_estimators": 10})


def test_registry_algoritmos():
    from yggdrasil.credit_risk.model.segmenter import ALGORITHMS, BOOSTING_ALGORITHMS
    for a in ["random_forest", "extra_trees", "gradient_boosting",
              "hist_gradient_boosting", "lightgbm", "xgboost", "catboost"]:
        assert a in ALGORITHMS
        assert ALGORITHMS[a]["tasks"] == ("classification", "regression")
    assert ALGORITHMS["lightgbm"]["extra"] == "lgbm"
    assert ALGORITHMS["xgboost"]["extra"] == "xgboost"
    assert ALGORITHMS["catboost"]["extra"] == "catboost"
    assert set(BOOSTING_ALGORITHMS) <= set(ALGORITHMS)
    # os modelos de boosting do sklearn não exigem pacote externo
    assert ALGORITHMS["hist_gradient_boosting"]["extra"] is None


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


def test_suggest_n_ratings(seg):
    seg.fit(_default_algo(seg.task_type))
    seg.build_ratings(method="decis", n_ratings=7)
    antes = list(seg.rating_labels_)
    sug = seg.suggest_n_ratings(method="quantil", n_min=3, n_max=12)
    assert {"best", "table", "reason"}.issubset(sug)
    assert 3 <= sug["best"] <= 12
    assert {"n_alvo", "n_efetivo", "inv_amostra", "repr_min_%", "ok"}.issubset(sug["table"].columns)
    # é não-destrutivo: a régua atual permanece intacta
    assert list(seg.rating_labels_) == antes


def test_build_ratings_auto(seg):
    seg.fit(_default_algo(seg.task_type))
    seg.build_ratings(method="quantil", n_ratings="auto")
    assert seg.rating_ is not None and len(seg.rating_labels_) >= 2
    # a sugestão usada fica registrada
    assert seg._last_auto_suggestion["best"] >= 2


def test_build_ratings_exige_score(seg):
    with pytest.raises(RuntimeError):
        seg.build_ratings()


def test_adjacent_rating_tests(seg):
    """API: DataFrame par×amostra×p-valor×veredito para TODOS os pares de
    ratings vizinhos, em todas as amostras (ou só na pedida via ``sample``)."""
    seg.fit(_default_algo(seg.task_type))
    seg.build_ratings(method="quantil", n_ratings=5)
    t = seg.adjacent_rating_tests()
    assert list(t.columns) == ["par", "amostra", "n_a", "n_b", "teste",
                               "p_valor", "veredito"]
    assert t["par"].nunique() == len(seg.rating_labels_) - 1
    assert set(t["amostra"]) == {"DES", "OOT"}
    assert t["veredito"].isin(["separa", "nao_separa", "n/d"]).all()
    esperado = ("proporções (z)" if seg.task_type == "classification"
                else "Mann-Whitney")
    assert (t["teste"] == esperado).all()
    # filtro por amostra + validação de amostra desconhecida
    t_des = seg.adjacent_rating_tests(sample="DES")
    assert set(t_des["amostra"]) == {"DES"}
    assert len(t_des) == t["par"].nunique()
    with pytest.raises(ValueError, match="Amostra"):
        seg.adjacent_rating_tests(sample="ZZZ")


def test_adjacent_rating_tests_separacao(task):
    """Dois ratings com alvo de distribuição idêntica → 'nao_separa' (a fusão
    monotônica garante a ordem, não a significância); bem separados → 'separa'."""
    df = _synthetic(task)
    seg = ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                         ref_sample="DES", date_col="dt_ref", verbose=False)
    n = len(seg.df)
    seg.rating_labels_ = ["R1", "R2"]
    seg.rating_ = pd.Series(np.where(np.arange(n) < n // 2, "R1", "R2"),
                            index=seg.df.index)
    # alvo alternado por paridade (independente da faixa) → distribuição idêntica
    if task == "classification":
        seg.df["target"] = (np.arange(n) % 2).astype(float)
    else:
        seg.df["target"] = np.where(np.arange(n) % 2 == 0, 0.2, 0.8)
    t = seg.adjacent_rating_tests()
    assert len(t) and (t["veredito"] == "nao_separa").all()
    # faixas BEM separadas (R1 baixo, R2 alto) → separa em todas as amostras
    if task == "classification":
        seg.df["target"] = (np.arange(n) >= n // 2).astype(float)
    else:
        rng = np.random.default_rng(0)
        seg.df["target"] = (np.where(np.arange(n) < n // 2, 0.1, 0.9)
                            + rng.normal(0, 0.01, n))
    t2 = seg.adjacent_rating_tests()
    assert len(t2) and (t2["veredito"] == "separa").all()


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


def test_psi_rating_detalhe_des_oot_estab(task):
    # PSI por rating comparando DES × OOT e DES × ESTABILIDADE (Task 3).
    df = _synthetic(task, n=1800, seed=3)
    rng = np.random.default_rng(3)
    df["amostra"] = rng.choice(["DES", "OOT", "ESTABILIDADE"], size=len(df),
                               p=[0.5, 0.3, 0.2])
    seg = ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                         ref_sample="DES", date_col="dt_ref", verbose=False)
    seg.fit(_default_algo(task))
    seg.build_ratings(method="quantil", n_ratings=5)
    det = seg.psi_rating_detalhe()
    # colunas de contribuição por amostra de comparação
    assert "PSI OOT" in det.columns and "PSI ESTABILIDADE" in det.columns
    assert "%DES" in det.columns
    # última linha é o TOTAL e bate com o PSI agregado de psi()
    assert det.iloc[-1]["rating"] == "TOTAL"
    agg = seg.psi().set_index("amostra")["psi"]
    tot = det.iloc[-1]
    assert abs(float(tot["PSI OOT"]) - float(agg["OOT"])) < 1e-3
    assert abs(float(tot["PSI ESTABILIDADE"]) - float(agg["ESTABILIDADE"])) < 1e-3


def test_advanced_hyperparams_fit(task):
    # Task 2: hiperparâmetros avançados chegam ao estimador sem quebrar o fit.
    df = _synthetic(task, n=1200, seed=7)
    seg = ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                         ref_sample="DES", verbose=False)
    hp = {"n_estimators": 40, "min_samples_leaf": 10, "max_features": "sqrt"}
    seg.fit("random_forest", hyperparams=hp)
    est = seg.model.named_steps["est"]
    assert est.get_params()["min_samples_leaf"] == 10
    assert est.get_params()["max_features"] == "sqrt"


def test_score_table_progress_callback(seg):
    # Task 4: score_table dispara eventos de progresso (carregando/escorando).
    seg.fit(_default_algo(seg.task_type))
    seg.build_ratings(method="quantil", n_ratings=5)
    eventos = []
    out = seg.score_table(seg.df.head(200),
                          progress_callback=lambda *a: eventos.append(a))
    assert len(out) == 200
    chaves = {e[0] for e in eventos}
    assert "score" in chaves and "done" in chaves
    # cada etapa segue o contrato (key, label, status, detail)
    assert all(len(e) == 4 for e in eventos)
    assert any(e[2] == "ok" for e in eventos)


def test_backtest(seg):
    seg.fit(_default_algo(seg.task_type))
    bt = seg.backtest("dt_ref")
    assert {"safra", "n", "previsto_medio", "realizado_medio",
            "gap", "ic_low", "ic_high", "p_valor", "status"}.issubset(bt.columns)
    assert int(bt["n"].sum()) == len(seg.df)
    assert set(bt["status"].unique()) <= {"ok", "atencao", "alerta"}
    with pytest.raises(ValueError):
        seg.backtest("coluna_inexistente")


def test_backtest_jeffreys_n_pequeno_vs_grande():
    """Semáforo estatístico (Jeffreys) do backtest em classificação:

    * safra PEQUENA com gap grande (4/25 = 16% vs previsto 10%, gap 0,06) →
      IC largo → previsto DENTRO do IC 95% → ``ok`` (o critério antigo de
      tolerância fixa 0,03 marcaria alerta);
    * safra GRANDE com gap pequeno (11% vs 10%, gap 0,01, n=20000) → IC
      estreito → previsto FORA do IC 99% → ``alerta`` (o critério antigo
      marcaria ok)."""
    y_peq = [1.0] * 4 + [0.0] * 21                      # taxa 0.16, n=25
    y_grd = [1.0] * 2200 + [0.0] * 17800                # taxa 0.11, n=20000
    df = pd.DataFrame({"sc": 0.10,
                       "target": y_peq + y_grd,
                       "dt_ref": (["2024-01-15"] * len(y_peq)
                                  + ["2024-02-15"] * len(y_grd))})
    seg = ModelSegmenter(df, target="target", task_type="classification",
                         date_col="dt_ref", verbose=False)
    seg.set_model(_IdentityModel("sc"), features=["sc"])
    bt = seg.backtest("dt_ref").set_index("safra")
    assert bt.loc["2024-01", "status"] == "ok"          # n pequeno: gap é ruído
    assert bt.loc["2024-02", "status"] == "alerta"      # n grande: gap significa
    # IC 95% contém a taxa observada e p_valor acompanha o semáforo
    assert bt.loc["2024-01", "ic_low"] <= 0.16 <= bt.loc["2024-01", "ic_high"]
    assert bt.loc["2024-01", "p_valor"] >= 0.05
    assert bt.loc["2024-02", "p_valor"] < 0.01
    # compat: tol explícito volta ao critério antigo de tolerância fixa do gap
    bt_tol = seg.backtest("dt_ref", tol=0.03).set_index("safra")
    assert bt_tol.loc["2024-01", "status"] == "alerta"  # 0.06 > 0.03
    assert bt_tol.loc["2024-02", "status"] == "ok"      # 0.01 <= 0.03


def test_backtest_teste_t_regressao():
    """Regressão: teste t pareado do gap — safra pequena/ruidosa fica ``ok``;
    safra grande com viés pequeno mas sistemático dispara ``alerta``."""
    rng = np.random.default_rng(5)
    n1, n2 = 8, 5000
    noise1 = rng.normal(0, 0.30, n1)
    noise2 = rng.normal(0, 0.05, n2)
    df = pd.DataFrame({
        "sc": 0.5,
        "target": np.concatenate([0.5 + 0.10 + (noise1 - noise1.mean()),
                                  0.5 + 0.02 + (noise2 - noise2.mean())]),
        "dt_ref": ["2024-01-15"] * n1 + ["2024-02-15"] * n2})
    seg = ModelSegmenter(df, target="target", task_type="regression",
                         date_col="dt_ref", verbose=False)
    seg.set_model(_IdentityModel("sc"), features=["sc"])
    bt = seg.backtest("dt_ref").set_index("safra")
    assert bt.loc["2024-01", "status"] == "ok"          # se grande → t pequeno
    assert bt.loc["2024-02", "status"] == "alerta"      # viés 0.02 com n=5000
    assert bt.loc["2024-02", "p_valor"] < 0.01


def test_rating_table_status_teste(seg):
    seg.fit(_default_algo(seg.task_type))
    seg.build_ratings(method="quantil", n_ratings=5)
    rt = seg.rating_table()
    assert {"ic_low", "ic_high", "status_teste"}.issubset(rt.columns)
    assert set(rt["status_teste"].unique()) <= {"ok", "atencao", "alerta"}
    fin = rt.dropna(subset=["ic_low", "ic_high"])
    assert len(fin) and (fin["ic_low"] <= fin["ic_high"]).all()


def test_hosmer_lemeshow():
    df = _synthetic("classification")
    seg = ModelSegmenter(df, target="target", task_type="classification",
                         sample_col="amostra", ref_sample="DES",
                         date_col="dt_ref", verbose=False)
    seg.fit("logistica")
    hl = seg.hosmer_lemeshow()
    assert hl["statistic"] >= 0.0
    assert 0.0 <= hl["p_value"] <= 1.0
    assert hl["df"] == max(hl["n_groups"] - 2, 1)
    assert len(hl["table"]) == hl["n_groups"]
    assert int(hl["table"]["n"].sum()) == int(seg.df["target"].notna().sum())
    # restrita a uma amostra: usa só as linhas dela
    hl_oot = seg.hosmer_lemeshow(sample="OOT")
    assert int(hl_oot["table"]["n"].sum()) == int((seg.df["amostra"] == "OOT").sum())


def test_hosmer_lemeshow_erro_na_regressao():
    df = _synthetic("regression")
    seg = ModelSegmenter(df, target="target", task_type="regression",
                         sample_col="amostra", ref_sample="DES", verbose=False)
    seg.fit("linear")
    with pytest.raises(ValueError):
        seg.hosmer_lemeshow()


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


def test_score_table_pandas_so_variaveis_originais(seg):
    seg.fit(_default_algo(seg.task_type))
    seg.build_ratings(method="quantil", n_ratings=6)
    novos = _synthetic(seg.task_type, n=200, seed=11, com_cat=True)
    # a tabela só precisa ter as variáveis do modelo
    entrada = novos[seg.model_features].copy()
    out = seg.score_table(entrada, col_value="valor_previsto")
    assert {"score", "rating", "valor_previsto"}.issubset(out.columns)
    assert out["score"].notna().all()
    # faltando uma variável do modelo → erro claro
    if len(seg.model_features) > 1:
        with pytest.raises(ValueError):
            seg.score_table(entrada[seg.model_features[:-1]])


def test_create_categorical_agrupa_e_modela(seg):
    if seg.task_type != "classification":
        pytest.skip("foco em classificação")
    # agrupa categorias na mão (como na árvore) e materializa numa nova variável
    seg.set_manual_bins("feat_cat", "A,B; C,D")
    nome = seg.create_categorical("feat_cat")
    assert nome in seg.candidates and nome in seg.df.columns
    assert seg.df[nome].dropna().nunique() == 2
    assert seg.var_meta[nome]["derived_from"] == "feat_cat"
    # variável numérica vira faixas
    seg.set_manual_bins("feat_00", "0.0")
    nome_num = seg.create_categorical("feat_00")
    assert seg.df[nome_num].dropna().nunique() == 2
    # treina usando as derivadas e escora tabela só com as variáveis ORIGINAIS
    seg.included = {nome, nome_num}
    seg.fit("logistica")
    seg.build_ratings(method="quantil", n_ratings=6)
    novos = _synthetic("classification", n=200, seed=21, com_cat=True)[["feat_cat", "feat_00"]]
    out = seg.score_table(novos)                 # recria as derivadas a partir da origem
    assert {"score", "rating"}.issubset(out.columns)
    assert out["score"].notna().all()
    assert nome in out.columns and nome_num in out.columns


def test_clear_derived_reseta_variaveis_criadas(seg):
    if seg.task_type != "classification":
        pytest.skip("foco em classificação")
    originais = list(seg.candidates)
    seg.set_manual_bins("feat_cat", "A,B; C,D")
    n1 = seg.create_categorical("feat_cat")
    seg.set_manual_bins("feat_00", "0.0")
    n2 = seg.create_categorical("feat_00")
    seg.include(n1)
    assert set(seg.derived_features()) == {n1, n2}
    removidas = seg.clear_derived()
    assert set(removidas) == {n1, n2}
    assert seg.derived_features() == []
    assert list(seg.candidates) == originais
    assert n1 not in seg.df.columns and n2 not in seg.df.columns
    assert n1 not in seg.included
    # idempotente: sem nada a remover
    assert seg.clear_derived() == []


def test_create_categorical_save_load(seg):
    if seg.task_type != "classification":
        pytest.skip("foco em classificação")
    import tempfile
    import os
    seg.set_manual_bins("feat_cat", "A,B; C,D")
    nome = seg.create_categorical("feat_cat")
    seg.included = {nome, "feat_03"}
    seg.fit("logistica")
    p = os.path.join(tempfile.mkdtemp(), "m.json")
    seg.save(p)
    base = seg.df.drop(columns=[nome])           # df "cru", sem a derivada
    seg2 = ModelSegmenter(base, target="target", task_type="classification",
                          sample_col="amostra", ref_sample="DES",
                          date_col="dt_ref", verbose=False).load(p)
    assert nome in seg2.df.columns               # derivada recriada no load
    assert np.allclose(seg.score_.values, seg2.score_.values, equal_nan=True)


def test_recreate_categories_usa_bins_do_modelo(seg):
    if seg.task_type != "classification":
        pytest.skip("recreate em classificação")
    seg.included = {"feat_00", "feat_cat"}
    seg.set_manual_bins("feat_00", "0.0")                  # 2 faixas numéricas
    seg.fit("logistica", transform="woe")
    seg.build_ratings(method="quantil", n_ratings=6)
    novos = _synthetic("classification", n=150, seed=12, com_cat=True)[seg.model_features]
    out = seg.score_table(novos)                            # woe ⇒ recria faixas por padrão
    assert "feat_00_faixa" in out.columns
    # a faixa numérica recriada respeita o corte 0.0 definido no treino
    faixas = set(out["feat_00_faixa"].dropna().unique())
    assert any("0" in f for f in faixas)
    # cada linha caiu em exatamente uma faixa (sem nulos onde há valor)
    assert out.loc[novos["feat_00"].notna(), "feat_00_faixa"].notna().all()


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


def test_plot_cap_lift_gating(seg):
    """CAP e Lift/Gains: exclusivos de classificação (ValueError em regressão);
    em classificação retornam Figure e a legenda da CAP traz o AR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seg.fit(_default_algo(seg.task_type))
    if seg.task_type == "classification":
        fig = seg.plot_cap()
        assert fig is not None
        leg = fig.axes[0].get_legend()
        assert leg is not None and any("AR=" in t.get_text() for t in leg.get_texts())
        plt.close(fig)
        fig2 = seg.plot_lift(n_bins=10)
        assert fig2 is not None
        plt.close(fig2)
    else:
        with pytest.raises(ValueError):
            seg.plot_cap()
        with pytest.raises(ValueError):
            seg.plot_lift()


def test_metrics_by_safra(seg):
    """metrics_by_safra: DataFrame por safra com as colunas do task; exige date_col."""
    seg.fit(_default_algo(seg.task_type))
    ms = seg.metrics_by_safra()
    assert not ms.empty
    assert {"safra", "n"} <= set(ms.columns)
    if seg.task_type == "classification":
        assert {"taxa_evento", "auc", "ks", "gini"} <= set(ms.columns)
    else:
        assert {"previsto_medio", "realizado_medio", "mae", "rmse", "r2"} <= set(ms.columns)
    # sem date_col configurado → erro amigável
    df = _synthetic(seg.task_type)
    s2 = ModelSegmenter(df, target="target", task_type=seg.task_type,
                        sample_col="amostra", ref_sample="DES")
    s2.fit(_default_algo(seg.task_type))
    with pytest.raises(ValueError):
        s2.metrics_by_safra()


def test_plot_metrics_by_safra_e_backtest(seg):
    """Gráficos por safra: métricas ao longo do tempo e backtest previsto×realizado."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seg.fit(_default_algo(seg.task_type))
    fig = seg.plot_metrics_by_safra()
    assert fig is not None
    plt.close(fig)
    figb = seg.plot_backtest(tolerancia=0.25)
    assert figb is not None
    plt.close(figb)


def test_plot_metric_shift(seg):
    # barra horizontal do shift DES→OOT das principais métricas do task
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seg.fit(_default_algo(seg.task_type))
    fig = seg.plot_metric_shift()
    ax = fig.axes[0]
    bars = [p for c in ax.containers for p in c.patches]
    assert len(bars) >= 3                                   # ao menos 3 métricas
    labs = {t.get_text() for t in ax.get_yticklabels() if t.get_text()}
    esperado = ({"AUC", "Gini", "KS", "F1"} if seg.task_type == "classification"
                else {"R²", "RMSE", "MAE", "sMAPE"})
    assert labs & esperado                                  # rótulos das métricas
    plt.close(fig)


def test_plot_metric_shift_sem_oot():
    # sem amostra OOT (sem sample_col) → mensagem amigável, sem barras
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = _synthetic("regression").drop(columns=["amostra"])
    seg = ModelSegmenter(df, target="target", task_type="regression",
                         date_col="dt_ref", verbose=False)
    seg.fit("linear")
    ax = seg.plot_metric_shift().axes[0]
    assert not any(ax.containers)
    assert any("OOT" in t.get_text() for t in ax.texts)
    plt.close(ax.figure)


def test_plot_metric_comparison(seg):
    # barras agrupadas: 3 métricas × (DES, OOT) lado a lado. Em regressão o R² vai
    # num 2º eixo (twinx), então as barras se espalham por mais de um Axes.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seg.fit(_default_algo(seg.task_type))
    fig = seg.plot_metric_comparison()
    bars = [p for a in fig.axes for c in a.containers for p in c.patches]
    assert len(bars) == 6                                   # 3 métricas × 2 amostras
    base = {t.get_text().split()[0] for t in fig.axes[0].get_xticklabels()}  # tira a seta
    esperado = ({"AUC", "Gini", "KS"} if seg.task_type == "classification"
                else {"RMSE", "MAE", "R²"})
    assert base == esperado
    legs = list(fig.legends) + [a.get_legend() for a in fig.axes if a.get_legend() is not None]
    leg_txt = {t.get_text() for lg in legs for t in lg.get_texts()}
    assert {"DES", "OOT"} <= leg_txt                        # DES e OOT lado a lado
    plt.close(fig)


def test_plot_metric_comparison_sem_oot():
    # sem amostra OOT (sem sample_col) → mostra só a DES (3 barras), sem quebrar
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = _synthetic("regression").drop(columns=["amostra"])
    seg = ModelSegmenter(df, target="target", task_type="regression",
                         date_col="dt_ref", verbose=False)
    seg.fit("linear")
    fig = seg.plot_metric_comparison()
    bars = [p for a in fig.axes for c in a.containers for p in c.patches]
    assert len(bars) == 3                                   # 1 amostra × 3 métricas
    plt.close(fig)


def test_plots_variavel(seg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for f in (seg.plot_variable_logodds("feat_00"),
              seg.plot_variable_distribution("feat_00"),
              seg.plot_variable_distribution_badrate("feat_00"),
              seg.plot_variable_inversion_by_sample("feat_00"),
              seg.plot_variable_timeseries("feat_00"),
              seg.plot_variable_risk_by_safra("feat_00", "dt_ref"),   # numérica → bins
              seg.plot_variable_risk_by_safra("feat_cat", "dt_ref")):  # cat → PD/categoria
        assert f is not None; plt.close(f)


def test_psi_plot_mostra_linhas_de_alerta(seg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = seg.plot_variable_psi_by_safra("feat_00", "dt_ref")
    ax = fig.axes[0]
    # guia do PSI sempre visível: linhas em 0,10 e 0,25 e eixo que as alcança
    ys = {round(line.get_ydata()[0], 2) for line in ax.get_lines()}
    assert {0.10, 0.25}.issubset(ys)
    assert ax.get_ylim()[1] >= 0.25
    plt.close(fig)


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
    assert len(titulos) == 7
    assert any("Variáveis" in t or "variáveis" in t for t in titulos)
    assert any("Modelo" in t for t in titulos)
    assert any("Backward" in t for t in titulos)
    assert any("Rating" in t for t in titulos)
    assert any("Avançado" in t for t in titulos)
    # abas sem numeração (①–⑥), alinhado às UIs lgd/pd
    assert all(not t[:1] in "①②③④⑤⑥" for t in titulos)


def test_ui_bins_categoria_caixas(task):
    """Aba Análise: variável CATEGÓRICA no modo Manual mostra uma caixa (Dropdown de
    grupo) por categoria; alocar categorias ao mesmo grupo define os bins manuais
    (como no TreeSegmenter). Numérica segue no campo de cortes."""
    pytest.importorskip("ipywidgets")
    import contextlib
    import io
    import ipywidgets as W
    from yggdrasil.credit_risk.model import ModelSegmenterUI

    df = _synthetic(task, com_cat=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ui = ModelSegmenterUI(df, target="target", task_type=task,
                              sample_col="amostra", ref_sample="DES", date_col="dt_ref")
    # categórica no modo Manual → caixas visíveis, campo de cortes oculto
    ui.dd_var2.value = "feat_cat"
    ui.tg_binmode.value = "Manual"
    assert ui.an_cat_box.layout.display == "" and ui.tx_cuts.layout.display == "none"
    assert set(ui._an_cat_widgets) == {"A", "B", "C", "D"}
    assert all(isinstance(dd, W.Dropdown) for dd in ui._an_cat_widgets.values())
    # aloca A,B -> grupo 1 e C,D -> grupo 2, aplica e confere os bins manuais
    for c, g in {"A": 1, "B": 1, "C": 2, "D": 2}.items():
        ui._an_cat_widgets[c].value = g
    ui._on_apply_bins(None)
    grupos = ui.seg.manual_bins("feat_cat")
    assert sorted(sorted(g) for g in grupos) == [["A", "B"], ["C", "D"]]
    # numérica no Manual → campo de cortes visível, caixas ocultas
    ui.dd_var2.value = "feat_00"
    ui.tg_binmode.value = "Manual"
    assert ui.tx_cuts.layout.display == "" and ui.an_cat_box.layout.display == "none"
    # volta à categórica: as caixas remontam com o MESMO particionamento salvo
    ui.dd_var2.value = "feat_cat"
    ui.tg_binmode.value = "Manual"
    vals = {c: ui._an_cat_widgets[c].value for c in ui._an_cat_widgets}
    part = sorted(sorted(c for c in vals if vals[c] == g) for g in set(vals.values()))
    assert part == [["A", "B"], ["C", "D"]]
    # limpar volta ao ótimo e reseta as caixas (cada categoria no seu grupo)
    ui._on_clear_bins(None)
    assert ui.seg.manual_bins("feat_cat") is None
    ui.tg_binmode.value = "Manual"
    v2 = {c: ui._an_cat_widgets[c].value for c in ui._an_cat_widgets}
    assert len(set(v2.values())) == len(v2)


def test_ui_save_overwrite_confirma(task, tmp_path):
    """Salvar o modelo (.json) num caminho que já existe não grava direto: o gate
    executa do_save só quando não há conflito e aguarda confirmação se já existir."""
    pytest.importorskip("ipywidgets")
    import contextlib
    import io
    from yggdrasil.credit_risk.model import ModelSegmenterUI

    df = _synthetic(task, com_cat=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ui = ModelSegmenterUI(df, target="target", task_type=task,
                              sample_col="amostra", ref_sample="DES", date_col="dt_ref")
    p = str(tmp_path / "modelo.json")
    chamadas = []
    ui._confirm_overwrite(p, lambda: chamadas.append(1))   # não existe -> executa
    assert chamadas == [1]
    open(p, "w").close()                                   # passa a existir
    chamadas.clear()
    ui._confirm_overwrite(p, lambda: chamadas.append(1))   # existe -> aguarda confirmação
    assert chamadas == []


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


def test_ui_metrics_centralizada_e_comparacao(task):
    """A tabela de métricas por amostra fica com as células centralizadas (a regra
    `td` center vence a de direita herdada) e o gráfico de comparação das principais
    métricas por amostra (DES vs OOT) aparece."""
    pytest.importorskip("ipywidgets")
    import contextlib
    import io
    import re
    from yggdrasil.credit_risk.model import ModelSegmenterUI

    df = _synthetic(task, com_cat=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ui = ModelSegmenterUI(df, target="target", task_type=task,
                              sample_col="amostra", ref_sample="DES", date_col="dt_ref")
        ui._on_fit(None)
    html = ui.out_metrics.value
    # célula (td) centralizada e a regra vem depois da de direita (vence por ordem)
    assert re.search(r"td\s*\{[^}]*text-align:\s*center", html)
    assert html.rindex("text-align: center") > html.index("text-align: right")
    # gráfico de comparação das métricas (DES vs OOT) renderizado junto às métricas
    assert "img" in ui.out_metric_compare.value


def test_ui_bins_manuais_e_formula(task):
    pytest.importorskip("ipywidgets")
    import contextlib
    import io
    from yggdrasil.credit_risk.model import ModelSegmenterUI

    df = _synthetic(task, com_cat=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ui = ModelSegmenterUI(df, target="target", task_type=task,
                              sample_col="amostra", ref_sample="DES", date_col="dt_ref")
        # bins manuais via os controles da Aba 2
        ui.dd_var2.value = "feat_00"
        ui._sync_bin_controls()
        ui.tg_binmode.value = "Manual"
        ui.tx_cuts.value = "-0.5, 0.5"
        ui._on_apply_bins(None)
        assert ui.seg.manual_bins("feat_00") == [-0.5, 0.5]
        ui._on_clear_bins(None)
        assert ui.seg.manual_bins("feat_00") is None

        # visibilidade dos hiperparâmetros por algoritmo + fórmula
        algo_lin = _default_algo(task)
        ui.dd_algo.value = algo_lin
        assert ui.formula_card.layout.display != "none"
        ui.dd_algo.value = "random_forest"
        assert ui.box_ensemble.layout.display != "none"
        assert ui.formula_card.layout.display == "none"
        ui.dd_algo.value = algo_lin
        ui._on_fit(None)
        # a fórmula é renderizada automaticamente ao treinar (sem botão dedicado);
        # a tabela de coeficientes é o elemento principal
        assert "mseg-coef" in ui.out_formula.value


def test_ui_controles_boosting(task):
    pytest.importorskip("ipywidgets")
    import contextlib
    import io
    from yggdrasil.credit_risk.model import ModelSegmenterUI

    df = _synthetic(task, com_cat=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ui = ModelSegmenterUI(df, target="target", task_type=task,
                              sample_col="amostra", ref_sample="DES", date_col="dt_ref")
        # extra_trees: ensemble, sem learning_rate
        ui.dd_algo.value = "extra_trees"
        assert ui.box_ensemble.layout.display != "none"
        assert ui.box_lr.layout.display == "none"
        hp = ui._collect_hyperparams("extra_trees")
        assert "n_estimators" in hp and "learning_rate" not in hp
        # hist_gradient_boosting: boosting → learning_rate visível e coletado
        ui.dd_algo.value = "hist_gradient_boosting"
        assert ui.box_lr.layout.display != "none"
        assert ui.formula_card.layout.display == "none"
        hp = ui._collect_hyperparams("hist_gradient_boosting")
        assert "learning_rate" in hp
        # treina de fato um motor nativo do sklearn pela UI
        ui._on_fit(None)
        assert ui.seg.score_ is not None


def test_rating_ruler_e_predict_valor(seg):
    seg.fit(_default_algo(seg.task_type))
    seg.build_ratings(method="quantil", n_ratings=5)
    ruler = seg.rating_ruler(col_value="valor_previsto")
    assert list(ruler.columns) == ["rating", "n", "valor_previsto"]
    assert len(ruler) == len(seg.rating_labels_)
    assert ruler["n"].sum() > 0
    # predict com col_value anexa o valor previsto do alvo daquele rating
    novo = seg.df[seg.df["amostra"] == "OOT"].copy()
    out = seg.predict(novo, col_value="valor_previsto")
    assert {"score", "rating", "valor_previsto"}.issubset(out.columns)
    assert len(out) == len(novo)
    # cada valor_previsto bate exatamente com a régua para aquele rating
    mapa = dict(zip(ruler["rating"], ruler["valor_previsto"]))
    esperado = out["rating"].map(mapa)
    mask = out["rating"].notna()
    assert (out.loc[mask, "valor_previsto"].fillna(-999.0)
            == esperado.loc[mask].fillna(-999.0)).all()
    # sem col_value mantém o comportamento antigo (sem coluna de valor)
    out2 = seg.predict(novo)
    assert "valor_previsto" not in out2.columns
    assert {"score", "rating"}.issubset(out2.columns)


def test_rating_ruler_sem_ratings_erro(seg):
    seg.fit(_default_algo(seg.task_type))
    with pytest.raises(RuntimeError):
        seg.rating_ruler()


def test_ui_escorar_base(task):
    pytest.importorskip("ipywidgets")
    import contextlib
    import io
    from yggdrasil.credit_risk.model import ModelSegmenterUI

    df = _synthetic(task, com_cat=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ui = ModelSegmenterUI(df, target="target", task_type=task,
                              sample_col="amostra", ref_sample="DES", date_col="dt_ref")
        ui.seg.fit(_default_algo(task))
        ui.seg.build_ratings(method="quantil", n_ratings=5)
        # régua rating -> valor
        ui._on_ruler(None)
        assert "<table" in ui.out_ruler.value
        # escora a base carregada → ui.result com score/rating/valor_previsto
        ui._on_score(None)
        assert ui.result is not None
        assert {"score", "rating", "valor_previsto"}.issubset(ui.result.columns)
        # escora base externa via ui.score_df
        ui.score_df = df[df["amostra"] == "OOT"].copy()
        ui._on_score(None)
        assert len(ui.result) == int((df["amostra"] == "OOT").sum())
        assert "valor_previsto" in ui.result.columns


# ----------------------------------------------------------------------
# Tuning bayesiano (Optuna) — gated por optuna instalado
# ----------------------------------------------------------------------
def test_tune_optuna(task):
    pytest.importorskip("optuna")
    df = _synthetic(task, n=1500, seed=2)
    seg = ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                         ref_sample="DES", verbose=False)
    seg.auto_select(min_iv=0.0)
    res = seg.tune_optuna(algorithm="random_forest", n_trials=5, fit_best=True)
    assert res["algorithm"] == "random_forest"
    assert res["metric"] == ("auc" if task == "classification" else "r2")
    assert res["n_trials"] == 5 and isinstance(res["best_params"], dict)
    assert seg.score_ is not None              # fit_best treinou o modelo
    assert seg.tuning_["best_value"] == res["best_value"]


def test_tune_optuna_agrupa_metricas_nos_trials(task):
    # Task 7: cada trial guarda os grupos 'modelagem' e 'monitoramento'.
    pytest.importorskip("optuna")
    df = _synthetic(task, n=1200, seed=4)
    seg = ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                         ref_sample="DES", verbose=False)
    seg.auto_select(min_iv=0.0)
    seg.tune_optuna(algorithm="random_forest", n_trials=4, fit_best=False)
    tr = seg.study_.trials[0]
    assert "modelagem" in tr.user_attrs and "monitoramento" in tr.user_attrs
    chave = "auc" if task == "classification" else "r2"
    assert chave in tr.user_attrs["modelagem"]
    assert "psi_score_des_val" in tr.user_attrs["monitoramento"]


def test_tune_optuna_log_mlflow_nested(task, tmp_path):
    # Task 7: com log_mlflow=True, cada trial vira um run aninhado com métricas
    # agrupadas por 'modelagem/…' e 'monitoramento/…' sob um run-pai.
    pytest.importorskip("optuna")
    pytest.importorskip("mlflow")
    import mlflow
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    df = _synthetic(task, n=1200, seed=6)
    seg = ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                         ref_sample="DES", verbose=False)
    seg.auto_select(min_iv=0.0)
    exp = f"optuna_{task}"
    seg.tune_optuna(algorithm="random_forest", n_trials=3, fit_best=False,
                    log_mlflow=True, mlflow_experiment=exp)
    experiment = mlflow.get_experiment_by_name(exp)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                              output_format="list")
    trial_runs = [r for r in runs if r.data.tags.get("grupo") == "trial"]
    assert len(trial_runs) == 3
    # métricas agrupadas presentes em ao menos um trial
    metric_keys = set().union(*(r.data.metrics.keys() for r in trial_runs))
    assert any(k.startswith("modelagem/") for k in metric_keys)
    assert any(k.startswith("monitoramento/") for k in metric_keys)


def test_tune_optuna_algoritmo_nao_tunavel(task):
    pytest.importorskip("optuna")
    df = _synthetic(task, n=800, seed=2)
    seg = ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                         ref_sample="DES", verbose=False)
    seg.auto_select(min_iv=0.0)
    alvo = "linear" if task == "regression" else "logistica"
    if alvo == "linear":                       # linear não é tunável
        with pytest.raises(ValueError, match="tunável"):
            seg.tune_optuna(algorithm="linear", n_trials=3)


def test_ui_tune_optuna(task):
    pytest.importorskip("optuna")
    pytest.importorskip("ipywidgets")
    import contextlib
    import io
    from yggdrasil.credit_risk.model import ModelSegmenterUI

    df = _synthetic(task, n=1500, seed=2)
    with contextlib.redirect_stdout(io.StringIO()):
        ui = ModelSegmenterUI(df, target="target", task_type=task,
                              sample_col="amostra", ref_sample="DES", date_col="dt_ref")
        ui.seg.auto_select(min_iv=0.0)
        ui.dd_algo.value = "random_forest"
        ui.sl_trials.value = 5
        ui._on_tune(None)
        # o tuning roda numa thread de fundo (para o botão "Cancelar" responder);
        # aguarda a conclusão antes de checar o resultado.
        assert ui._tune_thread is not None
        ui._tune_thread.join(timeout=180)
    assert "Optuna" in ui.out_tune.value and "Erro" not in ui.out_tune.value
    assert ui.seg.score_ is not None
    assert "<table" in ui.out_metrics.value
    # barra de progresso preenchida até o fim e marcada como concluída
    assert ui.pb_tune.max == 5 and ui.pb_tune.value == 5
    assert ui.pb_tune.bar_style == "success" and ui.btn_tune.disabled is False
    assert ui.btn_cancel_tune.disabled is True


def test_ratings_manuais(task):
    df = _synthetic(task, n=2000, seed=1)
    seg = ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                         ref_sample="DES", verbose=False)
    seg.auto_select(min_iv=0.0)
    seg.fit(_default_algo(task))
    sc = seg.score_
    cuts = [float(np.quantile(sc, q)) for q in (0.3, 0.6, 0.85)]
    seg.build_ratings(method="manual_score", cuts=cuts)
    assert len(seg.rating_labels_) == 4                # 3 cortes → 4 faixas
    seg.build_ratings(method="manual_percentil", percentiles=[20, 40, 60, 80])
    assert len(seg.rating_labels_) == 5
    with pytest.raises(ValueError, match="cuts"):
        seg.build_ratings(method="manual_score")       # sem cortes → erro


def test_logistica_pvalores_na_formula():
    df = _synthetic("classification", n=2500, seed=2)
    seg = ModelSegmenter(df, target="target", task_type="classification",
                         sample_col="amostra", ref_sample="DES", verbose=False)
    seg.auto_select(min_iv=0.0)
    seg.fit("logistica")
    co = seg.model_coefficients()
    assert "p_valor" in co.columns and "signif" in co.columns
    assert (co["p_valor"].fillna(0) >= 0).all()


def test_report_pdf_model(task, tmp_path):
    df = _synthetic(task, n=1500, seed=3)
    seg = ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                         ref_sample="DES", verbose=False)
    seg.auto_select(min_iv=0.0)
    seg.fit(_default_algo(task))
    seg.build_ratings(method="quantil", n_ratings=5)
    p = str(tmp_path / "rel_model.pdf")
    seg.report_pdf(p)
    import os
    assert os.path.exists(p) and os.path.getsize(p) > 1000


def test_report_markdown_model(task, tmp_path):
    import os
    df = _synthetic(task, n=1500, seed=3)
    seg = ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                         ref_sample="DES", date_col="dt_ref", verbose=False)
    seg.auto_select(min_iv=0.0)
    seg.fit(_default_algo(task))
    seg.build_ratings(method="quantil", n_ratings=5)
    p = str(tmp_path / "rel_model.md")
    out = seg.report_markdown(p, time_col="dt_ref")
    assert out == p and os.path.exists(p) and os.path.getsize(p) > 200
    txt = open(p, encoding="utf-8").read()
    # seções esperadas + uma tabela markdown
    assert "## Visão geral" in txt and "## Métricas por amostra" in txt
    assert "## Régua de ratings" in txt and "## Backtest por safra" in txt
    assert "| amostra |" in txt or "| rating |" in txt
    # fórmula (linear/logística) ou importância SHAP, conforme o algoritmo
    assert ("## Fórmula do modelo" in txt) or ("## Importância das variáveis (SHAP)" in txt)


def test_report_markdown_requires_fit(task):
    df = _synthetic(task, n=400)
    seg = ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                         ref_sample="DES", verbose=False)
    with pytest.raises(RuntimeError):
        seg.report_markdown("nao_deve_gerar.md")


def test_ui_tema_escuro(task):
    pytest.importorskip("ipywidgets")
    import contextlib
    import io
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    df = _synthetic(task, n=800)
    with contextlib.redirect_stdout(io.StringIO()):
        ui = ModelSegmenterUI(df, target="target", task_type=task,
                              sample_col="amostra", ref_sample="DES")
        ui.cb_dark.value = True
    assert "dark" in ui.panel._dom_classes


def test_ui_avancado_tab_handlers(task):
    """Aba Avançado: os handlers rodam com modelo treinado e preenchem as saídas
    (CAP/Lift só em classificação; métricas por safra e backtest nos dois)."""
    pytest.importorskip("ipywidgets")
    import contextlib
    import io
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    df = _synthetic(task, com_cat=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ui = ModelSegmenterUI(df, target="target", task_type=task,
                              sample_col="amostra", ref_sample="DES", date_col="dt_ref")
        ui.seg.fit(_default_algo(task))
        ui._on_adv_msafra(None)
        ui._on_adv_backtest(None)
        if task == "classification":
            ui._on_adv_cap(None)
            ui._on_adv_lift(None)
    assert ui.out_adv_msafra_tab.value                    # tabela de métricas por safra
    assert "<img" in ui.out_adv_backtest.value            # figura do backtest
    if task == "classification":
        assert "<img" in ui.out_adv_cap.value
        assert "<img" in ui.out_adv_lift.value
    else:                                                 # regressão: CAP/Lift barrados
        assert ui.out_adv_cap.value == ""


def test_ui_modelo_desatualizado_e_reset(task):
    """Mexer nas variáveis DEPOIS de treinar marca o modelo como desatualizado;
    re-treinar (ou limpar a flag) volta ao estado 'em dia'."""
    pytest.importorskip("ipywidgets")
    import contextlib
    import io
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    df = _synthetic(task, com_cat=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ui = ModelSegmenterUI(df, target="target", task_type=task,
                              sample_col="amostra", ref_sample="DES", date_col="dt_ref")
        ui.seg.fit(_default_algo(task))
        ui._clear_dirty()
        assert ui._dirty_since_fit is False
        feat = next(iter(ui.seg.included))                # included é um set
        ui.dd_var.value = feat
        ui._on_exclude_var(None)                          # muda a seleção pós-treino
    assert ui._dirty_since_fit is True
    assert "desatualizado" in ui.bar.value


def test_tune_optuna_progress_callback(task):
    pytest.importorskip("optuna")
    df = _synthetic(task, n=1200, seed=4)
    seg = ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                         ref_sample="DES", verbose=False)
    seg.auto_select(min_iv=0.0)
    calls = []
    seg.tune_optuna(algorithm="random_forest", n_trials=4, fit_best=False,
                    progress_callback=lambda d, t, b: calls.append((d, t)))
    assert len(calls) == 4                       # um callback por trial
    assert calls[-1] == (4, 4)                   # último reporta conclusão


# ----------------------------------------------------------------------
# Two-Stage (hurdle de LGD): classificação P(y≥t) + regressão em y≥t
# ----------------------------------------------------------------------
def _synthetic_lgd(n=2500, seed=3):
    """Regressão com massa em 0 (curados) — cenário típico de LGD hurdle."""
    df = _synthetic("regression", n=n, seed=seed)
    zero = df.sample(frac=0.4, random_state=seed).index
    df.loc[zero, "target"] = 0.0
    return df


def test_two_stage_fit_metrics_e_rating():
    df = _synthetic_lgd()
    seg = ModelSegmenter(df, target="target", task_type="regression", sample_col="amostra",
                         ref_sample="DES", date_col="dt_ref", verbose=False)
    seg.fit_two_stage(threshold=0.2, clf_algorithm="logistica", reg_algorithm="linear")
    assert seg.two_stage and abs(seg.two_stage_threshold - 0.2) < 1e-9
    assert seg.score_ is not None and bool(seg.score_.notna().all())
    # a resposta combinada respeita a âncora do grupo abaixo do threshold
    assert float(seg.score_.min()) >= -1e-6
    # três visões de métricas: classificador, regressão (grupo ≥ t), combinado
    mc, mr, mm = seg.metrics_classifier(), seg.metrics_regressor(), seg.metrics()
    assert {"amostra", "n", "taxa_1", "auc", "ks"} <= set(mc.columns)
    assert {"amostra", "n", "rmse", "r2"} <= set(mr.columns)
    assert {"amostra", "n", "rmse", "r2"} <= set(mm.columns)
    # a 2ª etapa só conta o grupo ≥ t (n menor que o total da amostra)
    n_des_reg = int(mr.loc[mr["amostra"] == "DES", "n"].iloc[0])
    n_des_tot = int(mm.loc[mm["amostra"] == "DES", "n"].iloc[0])
    assert 0 < n_des_reg < n_des_tot
    # rating construído sobre a resposta combinada
    seg.build_ratings(method="quantil", n_ratings=6)
    assert len(seg.rating_labels_) >= 2


def test_two_stage_gating():
    df = _synthetic("regression", n=800, seed=4)
    seg = ModelSegmenter(df, target="target", task_type="regression", sample_col="amostra",
                         ref_sample="DES", verbose=False)
    with pytest.raises(ValueError):                 # threshold acima do máximo → 1 classe
        seg.fit_two_stage(threshold=float(df["target"].max()) + 1.0)
    with pytest.raises(RuntimeError):               # métricas exigem o modo two-stage
        seg.metrics_classifier()
    dfc = _synthetic("classification", n=800, seed=4)
    segc = ModelSegmenter(dfc, target="target", task_type="classification",
                          sample_col="amostra", ref_sample="DES", verbose=False)
    with pytest.raises(ValueError):                 # classificação não tem two-stage
        segc.fit_two_stage(threshold=0.5)


def test_two_stage_persistencia(tmp_path):
    df = _synthetic_lgd(n=1500, seed=5)
    seg = ModelSegmenter(df, target="target", task_type="regression", sample_col="amostra",
                         ref_sample="DES", date_col="dt_ref", verbose=False)
    seg.fit_two_stage(threshold=0.25)
    p = str(tmp_path / "ts.json")
    seg.save(p)
    seg2 = ModelSegmenter(df, target="target", task_type="regression", sample_col="amostra",
                          ref_sample="DES", date_col="dt_ref", verbose=False).load(p)
    assert seg2.two_stage and abs(seg2.two_stage_threshold - 0.25) < 1e-9
    assert np.allclose(seg.score_.to_numpy(), seg2.score_.to_numpy(), equal_nan=True)


def test_ui_two_stage_regressao():
    pytest.importorskip("ipywidgets")
    import contextlib
    import io
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    df = _synthetic_lgd(n=1800, seed=6)
    with contextlib.redirect_stdout(io.StringIO()):
        ui = ModelSegmenterUI(df, target="target", task_type="regression",
                              sample_col="amostra", ref_sample="DES", date_col="dt_ref")
        assert ui.row_twostage.layout.display != "none"     # a opção existe em regressão
        ui.cb_twostage.value = True
        ui.sl_ts_threshold.value = 0.2
        ui._on_fit(None)
    assert ui.seg.two_stage
    assert ui.box_twostage.layout.display == ""
    assert ui.row_algo.layout.display == "none"             # modelo único fica escondido
    assert "① Classificador" in ui.out_metrics.value
    assert "③ Resposta combinada" in ui.out_metrics.value
    assert ui.seg.rating_labels_                            # rating trazido automaticamente
    assert bool(ui.out_rating_table.value)


def test_ui_two_stage_oculto_em_classificacao():
    pytest.importorskip("ipywidgets")
    import contextlib
    import io
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    df = _synthetic("classification", n=800, seed=6)
    with contextlib.redirect_stdout(io.StringIO()):
        ui = ModelSegmenterUI(df, target="target", task_type="classification",
                              sample_col="amostra", ref_sample="DES")
    assert ui.row_twostage.layout.display == "none"         # sem two-stage em classificação


# ----------------------------------------------------------------------
# Balanceamento de classes (fit/tune) + métricas e ratings por segmento
# ----------------------------------------------------------------------
def _seg_com_contexto(task, seed=0):
    """Segmenter com uma coluna categórica de CONTEXTO ('canal', fora das
    features via ``features=``) para as quebras por grupo — inclui um grupo
    minúsculo ('RARO') para exercitar o ``min_n``."""
    df = _synthetic(task, seed=seed)
    rng = np.random.default_rng(seed)
    df["canal"] = rng.choice(["APP", "AGENCIA"], size=len(df), p=[0.6, 0.4]).astype(object)
    df.loc[df.index[:5], "canal"] = "RARO"
    feats = [c for c in df.columns if c.startswith("feat_")]
    return ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                          ref_sample="DES", date_col="dt_ref", features=feats,
                          verbose=False)


def test_metrics_by_group(task):
    seg = _seg_com_contexto(task)
    seg.fit(_default_algo(task))
    mg = seg.metrics_by_group("canal")
    assert {"grupo", "n", "nota"} <= set(mg.columns)
    if task == "classification":
        assert {"taxa_evento", "auc", "ks", "gini"} <= set(mg.columns)
    else:
        assert {"previsto_medio", "realizado_medio", "mae", "rmse", "r2"} <= set(mg.columns)
    assert set(mg["grupo"]) == {"APP", "AGENCIA", "RARO"}
    met = "auc" if task == "classification" else "rmse"
    idx = mg.set_index("grupo")
    # grupo minúsculo (< min_n): métricas suprimidas (NaN) com nota explicando
    assert np.isnan(idx.loc["RARO", met]) and "n <" in idx.loc["RARO", "nota"]
    # grupos grandes têm métrica reportada
    assert np.isfinite(idx.loc["APP", met])
    # restrição por amostra: menos linhas que a base toda
    mg_des = seg.metrics_by_group("canal", sample="DES")
    assert int(mg_des["n"].sum()) < int(mg["n"].sum())


def test_metrics_by_group_validacao(task):
    seg = _seg_com_contexto(task)
    seg.fit(_default_algo(task))
    with pytest.raises(ValueError, match="não existe"):
        seg.metrics_by_group("nao_existe")
    with pytest.raises(ValueError, match="candidata"):
        seg.metrics_by_group("feat_00")     # feature do modelo não vale
    with pytest.raises(ValueError, match="data"):
        seg.metrics_by_group("dt_ref")      # coluna de data → metrics_by_safra
    s2 = _seg_com_contexto(task, seed=1)    # sem fit → erro amigável
    with pytest.raises(RuntimeError):
        s2.metrics_by_group("canal")


def test_rating_distribution_by_group(task):
    seg = _seg_com_contexto(task)
    seg.fit(_default_algo(task))
    seg.build_ratings(method="quantil", n_ratings=4)
    rd = seg.rating_distribution_by_group("canal")
    assert list(rd.columns[:2]) == ["grupo", "n"]
    pct_cols = [c for c in rd.columns if c.startswith("%_")]
    assert len(pct_cols) == len(seg.rating_labels_)
    soma = rd[pct_cols].sum(axis=1)
    assert ((soma > 99.0) & (soma < 101.0)).all()   # cada grupo fecha ~100%
    # sem ratings construídos → erro amigável
    seg2 = _seg_com_contexto(task, seed=1)
    seg2.fit(_default_algo(task))
    with pytest.raises(RuntimeError):
        seg2.rating_distribution_by_group("canal")


def test_fit_class_balance_traducao_sklearn():
    df = _synthetic("classification")
    seg = ModelSegmenter(df, target="target", task_type="classification",
                         sample_col="amostra", ref_sample="DES", verbose=False)
    seg.fit("logistica", class_balance=True)
    assert seg.class_balance is True
    assert seg.model.named_steps["est"].class_weight == "balanced"
    assert seg.score_ is not None
    # balancear sobe a média do score (upweight da classe minoritária ~25%)
    m_bal = float(seg.score_.mean())
    seg.fit("logistica", class_balance=False)
    assert seg.class_balance is False
    assert seg.model.named_steps["est"].class_weight is None
    assert m_bal > float(seg.score_.mean())
    # florestas e HistGB também ganham class_weight='balanced'
    seg.fit("random_forest", hyperparams={"n_estimators": 20}, class_balance=True)
    assert seg.model.named_steps["est"].class_weight == "balanced"
    seg.fit("hist_gradient_boosting", hyperparams={"n_estimators": 30},
            class_balance=True)
    assert seg.model.named_steps["est"].class_weight == "balanced"


def test_fit_class_balance_gradient_boosting_sample_weight():
    """GradientBoosting clássico não tem class_weight — o balanceamento entra
    como sample_weight no fit e não pode quebrar o treino."""
    df = _synthetic("classification")
    seg = ModelSegmenter(df, target="target", task_type="classification",
                         sample_col="amostra", ref_sample="DES", verbose=False)
    seg.fit("gradient_boosting", hyperparams={"n_estimators": 30, "max_depth": 2},
            class_balance=True)
    assert seg.class_balance is True and seg.score_ is not None
    assert "class_weight" not in seg.model.named_steps["est"].get_params()


def test_fit_class_balance_via_hyperparams_e_persistencia():
    """'class_balance' embutido nos hyperparams (ex.: best_params do Optuna) é
    extraído — não vaza para o estimador — e a flag persiste no to_dict."""
    df = _synthetic("classification")
    seg = ModelSegmenter(df, target="target", task_type="classification",
                         sample_col="amostra", ref_sample="DES", verbose=False)
    seg.fit("logistica", hyperparams={"class_balance": True, "C": 0.5})
    assert seg.class_balance is True
    assert "class_balance" not in seg.hyperparams
    assert seg.model.named_steps["est"].class_weight == "balanced"
    d = seg.to_dict()
    assert d["class_balance"] is True
    seg2 = ModelSegmenter.from_dict(d, df)
    assert seg2.class_balance is True


def test_class_balance_regressao_erro():
    df = _synthetic("regression")
    seg = ModelSegmenter(df, target="target", task_type="regression",
                         sample_col="amostra", ref_sample="DES", verbose=False)
    with pytest.raises(ValueError, match="classification"):
        seg.fit("linear", class_balance=True)


def test_tune_optuna_class_balance_no_espaco():
    """class_balance=None (default) inclui o balanceamento no espaço de busca;
    True/False fixa a opção (não vira parâmetro do estudo)."""
    pytest.importorskip("optuna")
    df = _synthetic("classification", n=1200)
    seg = ModelSegmenter(df, target="target", task_type="classification",
                         sample_col="amostra", ref_sample="DES", verbose=False)
    res = seg.tune_optuna(algorithm="logistica", n_trials=4, verbose=False)
    assert any("class_balance" in t.params for t in seg.study_.trials)
    assert res["n_trials"] >= 1
    res2 = seg.tune_optuna(algorithm="logistica", n_trials=2, verbose=False,
                           class_balance=True)
    assert all("class_balance" not in t.params for t in seg.study_.trials)
    assert res2["n_trials"] >= 1
    assert seg.class_balance is True          # fit_best repassa o valor fixado

# ----------------------------------------------------------------------
# Restrições de monotonicidade (fit/tune: monotone='auto' | dict)
# ----------------------------------------------------------------------
def _df_monotonico(n=1600, seed=7):
    """Relação monotônica CLARA com o alvo: risco cresce com x_up e decresce
    com x_down; 'seg_cat' é categórica (fica fora das restrições)."""
    rng = np.random.default_rng(seed)
    x_up = rng.uniform(0, 1, n)
    x_down = rng.uniform(0, 1, n)
    p = np.clip(0.05 + 0.55 * x_up + 0.30 * (1.0 - x_down), 0.01, 0.95)
    meses = pd.date_range("2023-01-01", periods=6, freq="MS")
    df = pd.DataFrame({"x_up": x_up, "x_down": x_down,
                       "seg_cat": rng.choice(list("AB"), n).astype(object),
                       "target": (rng.uniform(0, 1, n) < p).astype(int)})
    df["dt_ref"] = rng.choice(meses, size=n)
    df["amostra"] = np.where(df["dt_ref"] >= meses[4], "OOT", "DES")
    return df


def _seg_monotonico(df=None):
    df = _df_monotonico() if df is None else df
    return ModelSegmenter(df, target="target", task_type="classification",
                          sample_col="amostra", ref_sample="DES",
                          date_col="dt_ref", verbose=False)


def test_monotone_directions_pela_tendencia_univariada():
    seg = _seg_monotonico()
    dirs = seg.monotone_directions(["x_up", "x_down", "seg_cat"])
    assert dirs["x_up"] == 1 and dirs["x_down"] == -1
    assert "seg_cat" not in dirs                     # categórica fica fora


def test_monotone_auto_hist_gradient_boosting():
    """monotone='auto': monotonic_cst chega como dict por NOME de coluna e as
    predições parciais respeitam a direção univariada."""
    df = _df_monotonico()
    seg = _seg_monotonico(df)
    seg.fit("hist_gradient_boosting", hyperparams={"max_iter": 60}, monotone="auto")
    cst = seg.model.named_steps["est"].monotonic_cst
    assert cst == {"num__x_up": 1, "num__x_down": -1}
    assert seg.monotone == "auto"
    assert seg.monotone_dirs_ == {"x_up": 1, "x_down": -1}
    # varrendo x_up (resto fixo), a média das probabilidades nunca cai
    base = df[seg.model_features]
    medias = []
    for q in np.linspace(0.05, 0.95, 7):
        g = base.copy()
        g["x_up"] = float(np.quantile(df["x_up"], q))
        medias.append(float(seg.model.predict_proba(g)[:, 1].mean()))
    assert all(b >= a - 1e-9 for a, b in zip(medias, medias[1:]))
    # persistência: escolha + direções aplicadas fazem round-trip no to_dict
    d = seg.to_dict()
    assert d["monotone"] == "auto"
    assert d["monotone_dirs"] == {"x_up": 1, "x_down": -1}
    seg2 = ModelSegmenter.from_dict(d, df, verbose=False)
    assert seg2.monotone == "auto"
    assert seg2.monotone_dirs_ == {"x_up": 1, "x_down": -1}


def test_monotone_dict_override_e_casos_sem_suporte():
    df = _df_monotonico()
    seg = _seg_monotonico(df)
    # dict sobrescreve a direção automática por variável (0 = libera)
    seg.fit("hist_gradient_boosting", hyperparams={"max_iter": 30},
            monotone={"x_down": 0})
    assert seg.model.named_steps["est"].monotonic_cst == {"num__x_up": 1}
    assert seg.monotone_dirs_ == {"x_up": 1}
    # valor inválido no dict
    with pytest.raises(ValueError, match="monotone"):
        seg.fit("hist_gradient_boosting", hyperparams={"max_iter": 20},
                monotone={"x_up": 2})
    # algoritmo sem suporte: aviso e opção ignorada (modelo treina livre)
    with pytest.warns(UserWarning, match="monotonicidade"):
        seg.fit("random_forest", hyperparams={"n_estimators": 20}, monotone="auto")
    assert seg.monotone is None and seg.monotone_dirs_ == {}
    # WoE: só transform='raw' — aviso e opção ignorada
    with pytest.warns(UserWarning, match="raw"):
        seg.fit("hist_gradient_boosting", hyperparams={"max_iter": 20},
                transform="woe", monotone="auto")
    assert seg.monotone is None and seg.monotone_dirs_ == {}


def test_monotone_vetor_posicional_lightgbm():
    pytest.importorskip("lightgbm")
    seg = _seg_monotonico()
    seg.fit("lightgbm", hyperparams={"n_estimators": 30}, monotone="auto")
    vec = seg.model.named_steps["est"].get_params()["monotone_constraints"]
    names = list(seg.model.named_steps["pre"].get_feature_names_out())
    # vetor alinhado às colunas pós-transformação: bloco num primeiro, dummies=0
    esperado = [{"num__x_up": 1, "num__x_down": -1}.get(nm, 0) for nm in names]
    assert list(vec) == esperado and len(vec) == len(names)


def test_monotone_vetor_posicional_xgboost():
    pytest.importorskip("xgboost")
    seg = _seg_monotonico()
    seg.fit("xgboost", hyperparams={"n_estimators": 30}, monotone="auto")
    vec = seg.model.named_steps["est"].get_params()["monotone_constraints"]
    names = list(seg.model.named_steps["pre"].get_feature_names_out())
    esperado = tuple({"num__x_up": 1, "num__x_down": -1}.get(nm, 0) for nm in names)
    assert tuple(vec) == esperado


def test_monotone_no_tune_optuna():
    """monotone='auto' no tuning: os trials e o re-ajuste final saem restritos."""
    pytest.importorskip("optuna")
    seg = _seg_monotonico(_df_monotonico(n=900))
    res = seg.tune_optuna(algorithm="hist_gradient_boosting", n_trials=2,
                          verbose=False, monotone="auto",
                          search_space={"max_iter": {"type": "int", "low": 20,
                                                     "high": 30, "step": 10}})
    assert res["n_trials"] >= 1
    cst = seg.model.named_steps["est"].monotonic_cst
    assert cst == {"num__x_up": 1, "num__x_down": -1}
    assert seg.monotone == "auto"
