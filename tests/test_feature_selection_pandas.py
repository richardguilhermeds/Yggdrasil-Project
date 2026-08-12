"""Testes do backend **pandas** da seleção de features.

Rodam sem pyspark/Java: exercitam o mesmo ``run_feature_selection`` dos testes Spark
(``tests/test_feature_selection.py``), mas alimentado com um ``pandas.DataFrame``.

Três blocos:

* **primitivas**: cada função de ``pandas_stats`` isolada;
* **ponta a ponta**: pipeline completo em pandas (classificação e regressão);
* **paridade**: mesmas decisões nos dois backends sobre a mesma tabela — este roda
  só se pyspark estiver disponível (``importorskip``).
"""

import numpy as np
import pandas as pd
import pytest

from yggdrasil import ColumnConfig
from yggdrasil.feature_selection import (
    FeatureSelectionConfig,
    FeatureSelectionReport,
    backend_name,
    is_pandas,
    run_feature_selection,
)
from yggdrasil.feature_selection import pandas_stats as ps
from yggdrasil.feature_selection.boruta import boruta_select
from yggdrasil.feature_selection.importance import importance_indicators


def _make_pdf(problem: str = "classification", n: int = 1200, seed: int = 0,
              ruido: float = 2.0) -> pd.DataFrame:
    """Tabela sintética com books externo/mercado + colunas problemáticas para os filtros.

    ``ruido`` calibra a força do sinal: no default (2.0) as features boas ficam em
    IV ≈ 0.3 / AUC ≈ 0.64 — forte, mas abaixo de ``iv_leakage`` (0.50), que é a faixa
    onde o pipeline de fato *seleciona*. Valores baixos (ex.: 0.5) levam IV a ~2.0 e
    tudo é marcado como suspeita de leakage.
    """
    rng = np.random.default_rng(seed)
    s1 = rng.normal(size=n)
    s2 = rng.normal(size=n)
    df = pd.DataFrame({
        "feat_externo_score": s1,
        "feat_externo_atraso": s1 * 0.97 + rng.normal(0, 0.05, n),   # redundante c/ score
        "feat_externo_const": 1.0,                                    # sem variância
        "feat_mercado_renda": s2,
        "feat_mercado_util": rng.normal(size=n),
        "feat_mercado_miss": rng.normal(size=n),                         # alto missing
    })
    linear = 0.9 * s1 - 0.7 * s2 + rng.normal(0, ruido, n)
    df["target"] = (linear > 0).astype(int) if problem == "classification" else linear
    df.loc[df.sample(frac=0.8, random_state=1).index, "feat_mercado_miss"] = np.nan
    df["dt_ref"] = pd.Timestamp("2024-01-01")
    df["amostra"] = "DES"
    return df


@pytest.fixture
def fs_cfg_rapido():
    return FeatureSelectionConfig(boruta_max_iter=8, rf_n_estimators=40, rf_max_depth=5,
                                  top_k_book=10, top_k_overall=10)


# ───────────────────────── backend ─────────────────────────
def test_backend_detecta_pandas():
    pdf = _make_pdf()
    assert is_pandas(pdf) and backend_name(pdf) == "pandas"
    # qualquer coisa que não seja pandas cai no caminho Spark (comportamento de hoje)
    assert not is_pandas(object()) and backend_name(object()) == "spark"


# ───────────────────────── primitivas ─────────────────────────
def test_numeric_columns_ignora_texto_e_bool():
    pdf = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "c": [True, False]})
    assert ps.numeric_columns(pdf, ["a", "b", "c"]) == ["a"]


def test_missing_rate():
    pdf = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0], "b": [1, 2, 3, 4]})
    miss = ps.missing_rate(pdf, ["a", "b"])
    assert miss["a"] == pytest.approx(0.5)
    assert miss["b"] == pytest.approx(0.0)
    vazio = ps.missing_rate(pdf.iloc[:0], ["a"])
    assert np.isnan(vazio["a"])


def test_variance_flags_constante_e_quase_constante():
    n = 500
    pdf = pd.DataFrame({
        "const": np.ones(n),
        # 99% de valor modal, mas com P1 != P99 — o caso que near_constante existe p/ pegar
        "quase": np.r_[np.ones(n - 5), [2.0, 3.0, 4.0, 5.0, 6.0]],
        "ok": np.linspace(0, 1, n),
    })
    out = ps.variance_flags(pdf, ["const", "quase", "ok"]).set_index("feature")
    assert bool(out.loc["const", "sem_variancia"])
    assert not bool(out.loc["quase", "sem_variancia"])
    assert out.loc["quase", "top1_share"] == pytest.approx(0.99)
    assert bool(out.loc["quase", "near_constante"])
    assert not bool(out.loc["ok", "sem_variancia"])
    assert not bool(out.loc["ok", "near_constante"])
    # sem variância faz curto-circuito: o share modal nem é calculado (igual ao Spark)
    assert np.isnan(out.loc["const", "top1_share"])


def test_correlation_matrices_e_corr_with_target():
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    pdf = pd.DataFrame({"a": x, "b": x * 0.98 + rng.normal(0, 0.05, 400),
                        "c": rng.normal(size=400), "target": x})
    cm = ps.correlation_matrices(pdf, ["a", "b", "c"])
    assert abs(cm["spearman"].loc["a", "b"]) > 0.9
    assert abs(cm["pearson"].loc["a", "c"]) < 0.2

    ct = ps.corr_with_target(pdf, ["a", "c"], "target")
    assert ct["a"] > 0.95 and abs(ct["c"]) < 0.2

    # menos de 2 numéricas => matrizes vazias (mesmo contrato do Spark)
    assert ps.correlation_matrices(pdf, ["a"])["spearman"].empty


def test_univariate_metrics_ordena_por_poder():
    rng = np.random.default_rng(7)
    n = 2000
    sinal = rng.normal(size=n)
    pdf = pd.DataFrame({"forte": sinal, "ruido": rng.normal(size=n)})
    pdf["target"] = (sinal + rng.normal(0, 0.3, n) > 0).astype(int)
    out = ps.univariate_metrics(pdf, ["forte", "ruido"], "target",
                                FeatureSelectionConfig()).set_index("feature")
    assert out.loc["forte", "iv"] > out.loc["ruido", "iv"]
    assert out.loc["forte", "auc"] > 0.7
    assert out.loc["ruido", "auc"] < 0.6
    assert out.loc["forte", "gini"] == pytest.approx(2 * out.loc["forte", "auc"] - 1, abs=1e-3)


def test_rf_importances_pega_o_sinal():
    rng = np.random.default_rng(11)
    n = 800
    sinal = rng.normal(size=n)
    pdf = pd.DataFrame({"forte": sinal, "ruido": rng.normal(size=n)})
    pdf["target"] = (sinal + rng.normal(0, 0.3, n) > 0).astype(int)
    imp = ps.rf_importances(pdf, ["forte", "ruido"], "target", "classification",
                            FeatureSelectionConfig(rf_n_estimators=40, rf_max_depth=5))
    assert imp["forte"] > imp["ruido"]


def test_infer_problem_type():
    cfg = ColumnConfig()
    bin_df = pd.DataFrame({"target": [0, 1, 1, 0, np.nan]})
    con_df = pd.DataFrame({"target": [0.3, 1.7, -2.1, 5.0]})
    assert ps.infer_problem_type(bin_df, cfg) == "classification"
    assert ps.infer_problem_type(con_df, cfg) == "regression"


def test_maybe_sample_respeita_limite():
    pdf = _make_pdf(n=500)
    cfg = FeatureSelectionConfig(sample_size=100)
    assert len(ps.maybe_sample(pdf, cfg)) == 100
    assert len(ps.maybe_sample(pdf, FeatureSelectionConfig())) == 500


# ───────────────────────── despacho ─────────────────────────
def test_dispatch_das_primitivas_publicas():
    """As funções públicas de spark_stats/importance/boruta aceitam pandas sem pyspark."""
    from yggdrasil.feature_selection import spark_stats

    pdf = _make_pdf(n=400)
    feats = ["feat_externo_score", "feat_mercado_renda"]
    assert spark_stats.numeric_columns(pdf, feats) == feats
    assert spark_stats.missing_rate(pdf, feats).notna().all()
    assert not spark_stats.variance_flags(pdf, feats).empty
    assert not spark_stats.correlation_matrices(pdf, feats)["spearman"].empty
    assert not spark_stats.corr_with_target(pdf, feats, "target").empty

    ind = importance_indicators(pdf, feats, "target", "classification",
                                FeatureSelectionConfig(rf_n_estimators=20, rf_max_depth=4))
    assert set(["rf_importance", "iv", "ks", "auc", "gini", "score"]) <= set(ind.columns)

    # backend="spark" no config é ignorado quando a entrada é pandas
    cfg = FeatureSelectionConfig(boruta_max_iter=5, rf_n_estimators=20, rf_max_depth=4)
    assert cfg.backend == "spark"
    dec = boruta_select(pdf, feats, "target", "classification", cfg)
    assert set(dec["decisao"]) <= {"confirmada", "tentativa", "rejeitada"}


# ───────────────────────── ponta a ponta ─────────────────────────
def test_pandas_missing_variancia_e_redundancia(fs_cfg_rapido):
    rep = run_feature_selection(_make_pdf("classification"), ColumnConfig(), fs_cfg_rapido,
                                books=["externo", "mercado"], with_panels=False)
    tab = rep.selection_table.set_index("feature")
    assert tab.loc["feat_externo_const", "motivo"] == "sem variância"
    assert tab.loc["feat_mercado_miss", "motivo"] == "alto missing"
    assert not bool(tab.loc["feat_externo_const", "selecionada"])
    assert not bool(tab.loc["feat_mercado_miss", "selecionada"])
    motivos = {tab.loc["feat_externo_score", "motivo"], tab.loc["feat_externo_atraso", "motivo"]}
    assert any(isinstance(m, str) and m.startswith("redundante") for m in motivos)


def test_pandas_end_to_end_classificacao(fs_cfg_rapido):
    rep = run_feature_selection(_make_pdf("classification"), ColumnConfig(), fs_cfg_rapido,
                                books=["externo", "mercado"])
    assert isinstance(rep, FeatureSelectionReport)
    assert rep.problem_type == "classification"
    assert set(rep.selected_features) == {"externo", "mercado"}
    assert len(rep.selected_overall) >= 1
    assert not rep.overall_importance.empty
    assert "overall_importance" in rep.panels and "book::externo" in rep.panels
    assert "Seleção de Features" in rep.to_html(embed_panels=False)


def test_pandas_barra_feature_com_leakage(fs_cfg_rapido):
    """Sinal quase perfeito (IV >> iv_leakage) não pode ser selecionado."""
    pdf = _make_pdf("classification", ruido=0.5)   # IV ~2.0 nas features boas
    rep = run_feature_selection(pdf, ColumnConfig(), fs_cfg_rapido,
                                books=["externo"], with_panels=False)
    tab = rep.selection_table.set_index("feature")
    assert bool(tab.loc["feat_externo_score", "leakage_flag"])
    assert tab.loc["feat_externo_score", "motivo"] == "suspeita de leakage (revisar)"
    assert "feat_externo_score" not in rep.selected_overall


def test_pandas_regressao(fs_cfg_rapido):
    rep = run_feature_selection(_make_pdf("regression"), ColumnConfig(), fs_cfg_rapido,
                                books=["externo", "mercado"], with_panels=False)
    assert rep.problem_type == "regression"
    # em regressão não há IV/KS — as colunas nem entram no ranking global
    assert "iv" not in rep.overall_importance.columns
    assert not rep.overall_importance.empty


def test_pandas_nao_muta_a_entrada(fs_cfg_rapido):
    pdf = _make_pdf("classification", n=400)
    antes = pdf.copy()
    run_feature_selection(pdf, ColumnConfig(), fs_cfg_rapido,
                          books=["externo"], with_panels=False)
    pd.testing.assert_frame_equal(pdf, antes)


def test_pandas_filtra_amostra_de_desenvolvimento(fs_cfg_rapido):
    pdf = _make_pdf("classification", n=800)
    pdf.loc[pdf.index[400:], "amostra"] = "OOT"
    # o alvo do OOT é lixo: se ele vazasse na seleção, a tabela mudaria
    pdf.loc[pdf.index[400:], "target"] = 1
    rep = run_feature_selection(pdf, ColumnConfig(), fs_cfg_rapido,
                                books=["externo"], with_panels=False)
    so_des = run_feature_selection(pdf.iloc[:400], ColumnConfig(), fs_cfg_rapido,
                                   books=["externo"], with_panels=False)
    pd.testing.assert_series_equal(
        rep.selection_table.set_index("feature")["selecionada"],
        so_des.selection_table.set_index("feature")["selecionada"],
    )


def test_pandas_sem_coluna_de_alvo(fs_cfg_rapido):
    pdf = _make_pdf().drop(columns=["target"])
    with pytest.raises(ValueError, match="alvo"):
        run_feature_selection(pdf, ColumnConfig(), fs_cfg_rapido, with_panels=False)


# ───────────────────────── paridade pandas × Spark ─────────────────────────
def test_paridade_com_spark(fs_cfg_rapido):
    """Mesma tabela nos dois backends => mesmas features selecionadas."""
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    # getOrCreate devolve a sessão do fixture de tests/test_feature_selection.py quando
    # os dois arquivos rodam juntos. Só paramos a sessão se fomos nós que a criamos —
    # senão este teste derrubaria a sessão dos testes Spark que viessem depois.
    ja_existia = SparkSession.getActiveSession() is not None
    spark = (SparkSession.builder.master("local[2]").appName("ygg-fsel-paridade")
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        pdf = _make_pdf("classification")
        rep_pd = run_feature_selection(pdf, ColumnConfig(), fs_cfg_rapido,
                                       books=["externo", "mercado"], with_panels=False)
        rep_sp = run_feature_selection(spark.createDataFrame(pdf), ColumnConfig(), fs_cfg_rapido,
                                       books=["externo", "mercado"], with_panels=False)
    finally:
        if not ja_existia:
            spark.stop()

    assert rep_pd.problem_type == rep_sp.problem_type
    # os filtros duros são determinísticos nos dois backends
    for f in ("feat_externo_const", "feat_mercado_miss"):
        m_pd = rep_pd.selection_table.set_index("feature").loc[f, "motivo"]
        m_sp = rep_sp.selection_table.set_index("feature").loc[f, "motivo"]
        assert m_pd == m_sp
    assert set(rep_pd.selected_overall) == set(rep_sp.selected_overall)
