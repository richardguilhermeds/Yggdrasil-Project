"""
Testes da árvore de segmentação unificada ``yggdrasil.credit_risk.tree``.

Uma só classe (:class:`TreeSegmenter`) atende **classificação** (PD, alvo binário)
e **regressão** (LGD, alvo contínuo) via ``task_type``. A maioria dos testes é
**parametrizada nos dois tipos** (prova de que a unificação preserva o
comportamento das antigas classes separadas); os checks de valor específicos de
cada tarefa (KS/AUC vs MAE/RMSE; IV WoE vs IV contínuo) ficam isolados.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.tree import TreeSegmenter

TASKS = ["classification", "regression"]


# ----------------------------------------------------------------------
# Geradores de dados: alvo binário (clf) ou contínuo (reg), mesmo desenho
# de features (score/ltv numérico + garantia categórica), com DES/OOT e safra.
# ----------------------------------------------------------------------
def make_df(task, n=4000, seed=0, com_na=False, com_oot=False):
    rng = np.random.default_rng(seed)
    x = rng.beta(2.5, 3, n) * 1.4 + 0.3                       # feature numérica
    gar = rng.choice(["A", "B", "C", "D"], n, p=[0.5, 0.22, 0.18, 0.1]).astype(object)
    if com_na:
        x[rng.random(n) < 0.08] = np.nan
        gar[rng.random(n) < 0.06] = np.nan
    lg = {"A": 0.0, "B": 0.10, "C": 0.16, "D": 0.30}
    risco = (0.1 + 0.4 * np.nan_to_num(x - 0.5, nan=0.35)
             + np.array([lg.get(g, 0.2) for g in gar]))
    if task == "classification":
        p = np.clip(risco, 0.01, 0.95)
        target = (rng.uniform(0, 1, n) < p).astype(float)
    else:
        target = np.clip(risco + rng.normal(0, 0.07, n), 0, 1)
    df = pd.DataFrame({"score": x, "garantia": gar, "target": target})
    if com_oot:
        meses = pd.date_range("2023-01-01", periods=10, freq="MS")
        df["dt_ref"] = rng.choice(meses, size=n)
        df["amostra"] = np.where(df["dt_ref"] >= meses[7], "OOT", "DES")
    else:
        df["amostra"] = "DES"
    return df


def _mk(task, **kw):
    df = kw.pop("df", None)
    if df is None:
        df = make_df(task, **{k: kw.pop(k) for k in ("n", "seed", "com_na", "com_oot")
                              if k in kw})
    return TreeSegmenter(df, target="target", task_type=task,
                         sample_col="amostra", ref_sample="DES", verbose=False, **kw)


@pytest.fixture(params=TASKS)
def task(request):
    return request.param


@pytest.fixture
def seg(task):
    return _mk(task)


def _cobertura_total(seg):
    cob = sum(s["mask"].sum() for s in seg.segments.values() if s["is_leaf"])
    return cob == len(seg.df)


# ----------------------------------------------------------------------
# Construção / validação
# ----------------------------------------------------------------------
def test_import_pacote():
    import yggdrasil
    from yggdrasil.credit_risk import TreeSegmenter as C
    assert C is TreeSegmenter
    assert isinstance(yggdrasil.__version__, str)


def test_task_type_invalido():
    df = make_df("classification", n=200)
    with pytest.raises(ValueError, match="task_type"):
        TreeSegmenter(df, target="target", task_type="binario", verbose=False)


def test_target_ausente_erro():
    df = make_df("classification", n=200).drop(columns=["target"])
    with pytest.raises(ValueError, match="alvo"):
        TreeSegmenter(df, target="target", task_type="classification", verbose=False)


def test_construtor_sample_col_ausente_erro(task):
    df = make_df(task, n=300)
    with pytest.raises(ValueError):
        TreeSegmenter(df, target="target", task_type=task,
                      sample_col="nao_existe", verbose=False)


# ----------------------------------------------------------------------
# Crescimento (manual num/cat, automático) + cobertura
# ----------------------------------------------------------------------
def test_grow_numerico(seg):
    seg.grow("score", splits=[0.8])
    assert len(seg.leaves()) >= 2
    assert _cobertura_total(seg)


def test_grow_categorico(seg):
    seg.grow("garantia", splits=[["A"], ["B", "C"], ["D"]])
    assert len(seg.leaves()) >= 2
    assert _cobertura_total(seg)


def test_grow_grupos_cat_repetidos_erro(seg):
    with pytest.raises(ValueError, match="repetida"):
        seg.grow("garantia", splits=[["A", "B"], ["B", "C"]])


def test_fit_auto_e_predict(seg):
    seg.fit_auto(max_depth=2, verbose=False)
    assert sum(s["is_leaf"] for s in seg.segments.values()) >= 2
    pred = seg.predict(make_df(seg.task_type, n=500, seed=9))
    assert {"segmento", "nota", "valor_regua"}.issubset(pred.columns)
    # nomes antigos task-específicos não existem mais
    assert "nota_pd" not in pred.columns and "nota_lgd" not in pred.columns


def test_faltantes_viram_bin_propria(task):
    seg = _mk(task, com_na=True, n=4000, seed=1)
    seg.grow("score", splits=[0.8])
    tem_na = any(v["is_leaf"] and v["conditions"][-1]["kind"] == "na"
                 for v in seg.segments.values())
    assert tem_na
    assert _cobertura_total(seg)


def test_grow_nao_cria_split_degenerado(task):
    df = make_df(task, n=500, seed=2)
    df["const"] = 1.0
    seg = TreeSegmenter(df, target="target", task_type=task,
                        sample_col="amostra", ref_sample="DES", verbose=False)
    seg.grow("const")
    assert len(seg.leaves()) == 1            # constante não separa


# ----------------------------------------------------------------------
# Métricas — específicas por tarefa (prova da ramificação)
# ----------------------------------------------------------------------
def test_metrics_classificacao():
    seg = _mk("classification", com_oot=True, n=6000, seed=3)
    seg.fit_auto(max_depth=3, verbose=False)
    m = seg.metrics()
    assert {"amostra", "taxa_default", "KS", "AUC", "Gini", "Acuracia", "F1"}.issubset(m.columns)
    row = m[m["amostra"] == "DES"].iloc[0]
    assert row["AUC"] > 0.6                    # régua discrimina
    assert np.isclose(row["Gini"], 2 * row["AUC"] - 1, atol=1e-6)


def test_metrics_regressao():
    seg = _mk("regression", com_oot=True, n=6000, seed=3)
    seg.fit_auto(max_depth=3, verbose=False)
    m = seg.metrics()
    assert {"amostra", "MAE", "RMSE", "R2"}.issubset(m.columns)
    row = m[m["amostra"] == "DES"].iloc[0]
    assert 0.0 <= row["MAE"] <= 1.0 and row["R2"] <= 1.0
    assert "KS" not in m.columns               # sem métricas de classificação


# ----------------------------------------------------------------------
# IV — escala específica por tarefa
# ----------------------------------------------------------------------
def test_variable_iv(seg):
    iv = seg.variable_iv("root")
    assert {"variavel", "iv", "forca", "n_bins"}.issubset(iv.columns)
    assert "score" in set(iv["variavel"])
    assert iv["iv"].max() > 0                  # alguma variável informativa


def test_variable_iv_forca_escala_por_task():
    iv_clf = _mk("classification", n=5000, seed=4).variable_iv("root")
    iv_reg = _mk("regression", n=5000, seed=4).variable_iv("root")
    # IV contínuo é numericamente bem menor que o WoE binário p/ o mesmo desenho
    assert iv_clf["iv"].max() > iv_reg["iv"].max()


def test_csi_requer_sample_col(task):
    df = make_df(task, n=500).drop(columns=["amostra"])
    seg = TreeSegmenter(df, target="target", task_type=task, verbose=False)
    seg.grow("score", splits=[0.8])
    with pytest.raises(Exception):
        seg.csi()


# ----------------------------------------------------------------------
# Persistência (inclui task_type) + régua
# ----------------------------------------------------------------------
def test_save_load_roundtrip(seg, tmp_path):
    seg.fit_auto(max_depth=2, verbose=False)
    p = str(tmp_path / "arvore.json")
    seg.save(p)
    novo = TreeSegmenter.load(p, seg.df)
    assert novo.task_type == seg.task_type
    a = seg.predict(seg.df)["segmento"]
    b = novo.predict(seg.df)["segmento"]
    assert (a.fillna("∅") == b.fillna("∅")).all()


def test_to_dict_preserva_task_type(seg):
    d = seg.to_dict()
    assert d["meta"]["task_type"] == seg.task_type
    assert d["schema"] == "yggdrasil.credit_risk.tree/1"


def test_regua_features(seg):
    seg.fit_auto(max_depth=2, verbose=False)
    feats = seg.regua_features()
    assert isinstance(feats, list) and len(feats) >= 1


def test_to_pyspark_compila(seg):
    seg.grow("score", splits=[0.8])
    code = seg.to_pyspark()
    compile(code, "<regua>", "exec")          # gera código válido
    assert "valor" in code and "def aplicar_regua" in code


# ----------------------------------------------------------------------
# Poda / fusão / estabilidade
# ----------------------------------------------------------------------
def test_auto_merge_funde_irmas_indistinguiveis(task):
    # dois patamares bem separados → após cortes finos, irmãs indistinguíveis fundem
    df = make_df(task, n=8000, seed=5, com_oot=True)
    seg = TreeSegmenter(df, target="target", task_type=task,
                        sample_col="amostra", ref_sample="DES", verbose=False)
    seg.grow("score", splits=[0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    n0 = len(seg.leaves())
    seg.auto_merge()
    assert len(seg.leaves()) <= n0


def test_prune_respeita_protect(seg):
    seg.grow("score", splits=[0.6, 0.8, 1.0])
    folhas = [s for s, v in seg.segments.items() if v["is_leaf"]]
    protect = set(folhas)
    seg.prune(min_repr=99.0, protect=protect)   # tudo violaria, mas está protegido
    assert set(s for s, v in seg.segments.items() if v["is_leaf"]) == protect


def test_psi_usa_segmentos_como_bins(task):
    seg = _mk(task, com_oot=True, n=6000, seed=6)
    seg.fit_auto(max_depth=2, verbose=False)
    psi = seg.psi()
    assert "psi" in psi.columns or "PSI" in psi.columns or len(psi) >= 1


# ----------------------------------------------------------------------
# Faltantes: bin própria + merge
# ----------------------------------------------------------------------
def test_merge_missing_numerico(task):
    seg = _mk(task, com_na=True, n=4000, seed=7)
    seg.grow("score", splits=[1.0])
    nums = [s for s, v in seg.segments.items()
            if v["is_leaf"] and v["conditions"][-1]["kind"] == "num"]
    assert any(v["is_leaf"] and v["conditions"][-1]["kind"] == "na"
               for v in seg.segments.values())
    seg.merge_missing(nums[-1])
    assert _cobertura_total(seg)


# ----------------------------------------------------------------------
# Backtest / calibração / monotonicidade
# ----------------------------------------------------------------------
def test_backtest_por_safra(task):
    seg = _mk(task, com_oot=True, n=6000, seed=8)
    seg.fit_auto(max_depth=2, verbose=False)
    bt = seg.backtest("dt_ref")
    assert {"periodo", "valor_previsto", "valor_realizado", "gap", "status"}.issubset(bt.columns)


def test_calibration_table(task):
    seg = _mk(task, com_oot=True, n=6000, seed=8)
    seg.fit_auto(max_depth=2, verbose=False)
    ct = seg.calibration_table()
    assert {"valor_previsto", "valor_realizado", "gap"}.issubset(ct.columns)


def test_monotonicity_report(task):
    seg = _mk(task, com_oot=True, n=6000, seed=8)
    seg.fit_auto(max_depth=3, verbose=False)
    mr = seg.monotonicity_report()
    assert {"amostra", "monotonico", "n_inversoes"}.issubset(mr.columns)


# ----------------------------------------------------------------------
# Plots — comuns e específicos por tarefa (gated por matplotlib)
# ----------------------------------------------------------------------
def test_plot_tree_gera_imagem(seg):
    pytest.importorskip("matplotlib")
    seg.fit_auto(max_depth=2, verbose=False)
    fig = seg.plot_tree()
    assert fig is not None


def test_plots_especificos_classificacao():
    pytest.importorskip("matplotlib")
    seg = _mk("classification", com_oot=True, n=4000, seed=3)
    seg.fit_auto(max_depth=2, verbose=False)
    assert seg.plot_roc() is not None
    assert seg.plot_ks() is not None
    assert seg.plot_score_distribution() is not None


def test_plots_especificos_regressao():
    pytest.importorskip("matplotlib")
    seg = _mk("regression", com_oot=True, n=4000, seed=3)
    seg.fit_auto(max_depth=2, verbose=False)
    assert seg.plot_leaf_boxplots() is not None
    assert seg.plot_target_hist() is not None
    assert seg.plot_leaf_value_hist() is not None


def test_plot_feature_value(seg):
    pytest.importorskip("matplotlib")
    fig = seg.plot_feature_value("score")
    assert fig is not None


# ----------------------------------------------------------------------
# PySpark (gated)
# ----------------------------------------------------------------------
def test_log_to_mlflow_metrricas_por_task(task, tmp_path):
    pytest.importorskip("mlflow")
    import mlflow
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    seg = _mk(task, com_oot=True, n=4000, seed=3)
    seg.fit_auto(max_depth=3, verbose=False)
    rid = seg.log_to_mlflow(experiment=f"t_{task}", verbose=False)
    run = mlflow.get_run(rid)
    # variáveis + profundidade nos params; n_variaveis nas métricas
    assert "variaveis" in run.data.params and "profundidade" in run.data.params
    assert "n_variaveis" in run.data.metrics
    # PSI por amostra (OOT) + métricas conforme o task
    assert any(k.startswith("psi_") for k in run.data.metrics)
    want = ["ks_DES", "auc_DES", "gini_DES"] if task == "classification" else ["mae_DES", "rmse_DES", "r2_DES"]
    assert all(k in run.data.metrics for k in want)


def test_to_sql_case_when(seg):
    seg.fit_auto(max_depth=2, verbose=False)
    sql = seg.to_sql(table="carteira")
    assert "CASE" in sql and "carteira" in sql
    assert "AS segmento" in sql and "AS folha" in sql and "AS valor_previsto" in sql
    # uma cláusula WHEN por folha
    assert sql.count("WHEN") >= 2 * len(seg.leaves())


def test_feature_importance(seg):
    seg.fit_auto(max_depth=3, verbose=False)
    fi = seg.feature_importance()
    assert {"variavel", "n_splits", "importancia"}.issubset(fi.columns)
    # só lista variáveis que entraram na árvore
    assert len(fi) >= 1 and (fi["n_splits"] >= 1).all()
    if "importancia_%" in fi.columns and fi["importancia_%"].sum() > 0:
        assert abs(fi["importancia_%"].sum() - 100.0) < 1.0


def test_suggest_splits(task):
    seg = _mk(task, com_oot=True, n=6000, seed=3)
    sug = seg.suggest_splits(top=3)
    assert {"variavel", "n_bins", "iv", "passa_teste", "p_valor"}.issubset(sug.columns)
    assert any(c.startswith("psi_") for c in sug.columns)     # PSI por amostra (OOT)
    assert len(sug) >= 1 and sug["passa_teste"].dtype == bool


CRITERIOS = {
    "classification": ["gini", "entropy", "ks", "iv", "chi2"],
    "regression": ["variance", "mae", "ftest"],
}


def test_fit_auto_por_criterio(task):
    for crit in CRITERIOS[task]:
        seg = _mk(task, n=5000, seed=4)
        seg.fit_auto(max_depth=3, criterion=crit, verbose=False)
        nf = sum(s["is_leaf"] for s in seg.segments.values())
        assert nf >= 2, f"critério {crit} não dividiu"
        assert _cobertura_total(seg)


def test_grow_por_criterio_e_binario(seg):
    # critério != optbin faz split BINÁRIO (2 filhos) por folha numérica
    seg.grow("score", criterion="gini" if seg.task_type == "classification" else "variance")
    assert len(seg.leaves()) == 2


def test_diff_trees(task):
    a = _mk(task, n=6000, seed=5)
    a.fit_auto(max_depth=3, verbose=False)
    b = _mk(task, df=a.df.copy())
    b.fit_auto(max_depth=1, verbose=False)
    d = a.diff_trees(b)
    assert 0.0 <= d["concordancia"] <= 1.0
    assert d["migracao"].shape[0] >= 1 and d["migracao"].shape[1] >= 1
    assert {"métrica", "árvore A", "árvore B"}.issubset(d["resumo"].columns)


def test_diff_trees_task_incompativel():
    a = _mk("classification", n=1000)
    b = _mk("regression", df=make_df("regression", n=1000))
    with pytest.raises(ValueError, match="task_type"):
        a.diff_trees(b)


def test_apply_spark_roundtrip(task):
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master("local[1]").appName("t").getOrCreate()
    try:
        seg = _mk(task, n=3000, seed=9)
        seg.grow("score", splits=[0.8])
        sdf = spark.createDataFrame(seg.df[["score", "garantia"]])
        out = seg.apply_spark(sdf).toPandas()
        assert {"segmento", "nota", "valor_regua"}.issubset(out.columns)
    finally:
        spark.stop()
