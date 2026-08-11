"""Testes da esteira de seleção de features (PySpark, por books).

Dois blocos:

* **puro** (sempre roda): resolução de books, decisão do Boruta, clusters de
  redundância, lógica de consenso e validação de config — tudo em pandas/numpy.
* **Spark** (``pytest.importorskip("pyspark")``): pipeline ponta a ponta sobre um
  Spark DataFrame local. Pula limpo se pyspark/Java não estiverem disponíveis
  (mesmo padrão de ``tests/test_lgd_segmenter.py``).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from yggdrasil import ColumnConfig
from yggdrasil.feature_selection import FeatureSelectionConfig, resolve_books
from yggdrasil.feature_selection.boruta import boruta_decision, boruta_driver
from yggdrasil.feature_selection.selector import _consensus
from yggdrasil.feature_selection.spark_stats import redundancy_clusters


# ───────────────────────── bloco puro ─────────────────────────
def test_resolve_books_keyword_dict_auto():
    cfg = ColumnConfig()
    cols = ["feat_bureau_score", "feat_bureau_atraso", "feat_bvs_renda",
            "feat_bvs_score", "target", "dt_ref", "amostra"]
    sdf = SimpleNamespace(columns=cols)

    kw = {b.name: b.features for b in resolve_books(sdf, cfg, ["bureau", "bvs"])}
    assert kw["bureau"] == ["feat_bureau_score", "feat_bureau_atraso"]
    assert kw["bvs"] == ["feat_bvs_renda", "feat_bvs_score"]

    auto = {b.name: b.features for b in resolve_books(sdf, cfg, None)}
    assert set(auto) == {"bureau", "bvs"}

    dic = resolve_books(sdf, cfg, {"bureau": ["feat_bureau_score", "feat_bvs_score"]})
    assert dic[0].name == "bureau" and len(dic[0].features) == 2


def test_resolve_books_erros():
    cfg = ColumnConfig()
    sdf = SimpleNamespace(columns=["feat_a", "target"])
    with pytest.raises(ValueError):                 # keyword sem match -> nenhum book
        resolve_books(sdf, cfg, ["inexistente"])
    with pytest.raises(ValueError):                 # coluna inexistente no dict
        resolve_books(sdf, cfg, {"x": ["feat_naoexiste"]})


def test_boruta_decision_classifica():
    hits = pd.Series({"forte": 50, "meio": 25, "ruido": 1}, dtype=int)
    dec = boruta_decision(hits, 50, 0.05).set_index("feature")["decisao"].to_dict()
    assert dec["forte"] == "confirmada"
    assert dec["ruido"] == "rejeitada"
    assert dec["meio"] == "tentativa"


def test_redundancy_clusters_representante_por_importancia():
    corr = pd.DataFrame(
        [[1.0, 0.95, 0.1], [0.95, 1.0, 0.12], [0.1, 0.12, 1.0]],
        index=["x1", "x2", "x3"], columns=["x1", "x2", "x3"],
    )
    imp = pd.Series({"x1": 0.2, "x2": 0.9, "x3": 0.5})
    out = redundancy_clusters(corr, 0.8, imp).set_index("feature")
    assert out.loc["x1", "cluster"] == out.loc["x2", "cluster"]   # x1,x2 redundantes
    assert out.loc["x2", "representante"] and not out.loc["x1", "representante"]
    assert out.loc["x1", "redundante_com"] == "x2"                # mantém a de maior importância
    assert out.loc["x3", "representante"]                         # x3 isolada


def test_consensus_seleciona_e_descarta():
    fscfg = FeatureSelectionConfig()
    hi = {"imp_norm": 0.9, "boruta_hit_rate": 0.95, "corr_target": 0.4,
          "boruta_decisao": "confirmada", "leakage_flag": False}
    lo = {"imp_norm": 0.05, "boruta_hit_rate": 0.05, "corr_target": 0.01,
          "boruta_decisao": "rejeitada", "leakage_flag": False}
    leak = {"imp_norm": 0.99, "boruta_hit_rate": 1.0, "corr_target": 0.99,
            "boruta_decisao": "confirmada", "leakage_flag": True}
    assert _consensus(hi, fscfg)[1] is True
    assert _consensus(lo, fscfg)[1] is False
    assert _consensus(leak, fscfg)[1] is False                   # leakage não entra


def test_boruta_driver_separa_sinal_de_ruido():
    rng = np.random.default_rng(0)
    n = 600
    sinal = rng.normal(size=n)
    X = pd.DataFrame({
        "feat_sinal": sinal,
        "feat_ruido1": rng.normal(size=n),
        "feat_ruido2": rng.normal(size=n),
    })
    y = (sinal + rng.normal(0, 0.3, n) > 0).astype(int)
    cfg = FeatureSelectionConfig(boruta_max_iter=15, rf_n_estimators=60, rf_max_depth=5)
    dec = boruta_driver(X, pd.Series(y), "classification", cfg).set_index("feature")["decisao"]
    assert dec["feat_sinal"] == "confirmada"


def test_config_validacao():
    with pytest.raises(ValueError):
        FeatureSelectionConfig(backend="dask")
    with pytest.raises(ValueError):
        FeatureSelectionConfig(var_p_low=0.9, var_p_high=0.1)


# ───────────────────────── bloco Spark ─────────────────────────
@pytest.fixture(scope="session")
def spark():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession
    sess = (
        SparkSession.builder.master("local[2]").appName("ygg-fsel-tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    sess.sparkContext.setLogLevel("ERROR")
    yield sess
    sess.stop()


def _make_pdf(problem: str = "classification", n: int = 1200, seed: int = 0,
              ruido: float = 2.0) -> pd.DataFrame:
    """Tabela sintética com books bureau/bvs + colunas problemáticas para os filtros.

    ``ruido`` calibra a força do sinal. No default (2.0) as features boas ficam em
    IV ≈ 0.3 / AUC ≈ 0.64 — forte, mas abaixo de ``iv_leakage`` (0.50), que é a faixa
    onde o pipeline de fato *seleciona*. Com ruído baixo (0.5) o IV vai a ~2.0 e o
    consenso barra tudo por suspeita de leakage: use esse valor para exercitar essa
    trava, nunca para testar o caminho feliz. Mesma convenção de
    ``tests/test_feature_selection_pandas.py``.
    """
    rng = np.random.default_rng(seed)
    s1 = rng.normal(size=n)
    s2 = rng.normal(size=n)
    df = pd.DataFrame({
        "feat_bureau_score": s1,
        "feat_bureau_atraso": s1 * 0.97 + rng.normal(0, 0.05, n),  # redundante c/ score
        "feat_bureau_const": 1.0,                                   # sem variância
        "feat_bvs_renda": s2,
        "feat_bvs_util": rng.normal(size=n),
        "feat_bvs_miss": rng.normal(size=n),                        # alto missing
    })
    linear = 0.9 * s1 - 0.7 * s2 + rng.normal(0, ruido, n)
    df["target"] = (linear > 0).astype(int) if problem == "classification" else linear
    df.loc[df.sample(frac=0.8, random_state=1).index, "feat_bvs_miss"] = np.nan
    df["dt_ref"] = pd.Timestamp("2024-01-01")
    df["amostra"] = "DES"
    return df


@pytest.fixture
def fs_cfg_rapido():
    # Config leve p/ os testes rodarem rápido.
    return FeatureSelectionConfig(boruta_max_iter=8, rf_n_estimators=40, rf_max_depth=5,
                                  top_k_book=10, top_k_overall=10)


def test_spark_missing_e_variancia(spark, fs_cfg_rapido):
    from yggdrasil.feature_selection import run_feature_selection
    sdf = spark.createDataFrame(_make_pdf("classification"))
    rep = run_feature_selection(sdf, ColumnConfig(), fs_cfg_rapido,
                                books=["bureau", "bvs"], with_panels=False)
    tab = rep.selection_table.set_index("feature")
    assert tab.loc["feat_bureau_const", "motivo"] == "sem variância"
    assert tab.loc["feat_bvs_miss", "motivo"] == "alto missing"
    assert not bool(tab.loc["feat_bureau_const", "selecionada"])
    assert not bool(tab.loc["feat_bvs_miss", "selecionada"])


def test_spark_redundancia(spark, fs_cfg_rapido):
    from yggdrasil.feature_selection import run_feature_selection
    sdf = spark.createDataFrame(_make_pdf("classification"))
    rep = run_feature_selection(sdf, ColumnConfig(), fs_cfg_rapido,
                                books=["bureau"], with_panels=False)
    tab = rep.selection_table.set_index("feature")
    # uma das duas features altamente correlacionadas deve sair como redundante
    motivos = {tab.loc["feat_bureau_score", "motivo"], tab.loc["feat_bureau_atraso", "motivo"]}
    assert any(isinstance(m, str) and m.startswith("redundante") for m in motivos)


def test_spark_end_to_end_classificacao(spark, fs_cfg_rapido):
    from yggdrasil.feature_selection import run_feature_selection, FeatureSelectionReport
    sdf = spark.createDataFrame(_make_pdf("classification"))
    rep = run_feature_selection(sdf, ColumnConfig(), fs_cfg_rapido, books=["bureau", "bvs"])
    assert isinstance(rep, FeatureSelectionReport)
    assert rep.problem_type == "classification"
    assert set(rep.selected_features) == {"bureau", "bvs"}
    assert len(rep.selected_overall) >= 1
    assert not rep.overall_importance.empty
    assert "overall_importance" in rep.panels and "book::bureau" in rep.panels
    html = rep.to_html(embed_panels=False)
    assert "Seleção de Features" in html


def test_spark_barra_feature_com_leakage(spark, fs_cfg_rapido):
    """Sinal quase perfeito (IV >> iv_leakage) não pode ser selecionado."""
    from yggdrasil.feature_selection import run_feature_selection
    sdf = spark.createDataFrame(_make_pdf("classification", ruido=0.5))  # IV ~2.0
    rep = run_feature_selection(sdf, ColumnConfig(), fs_cfg_rapido,
                                books=["bureau"], with_panels=False)
    tab = rep.selection_table.set_index("feature")
    # o representante do cluster bureau varia com o backend (os scores empatam);
    # o que importa é que quem sobreviveu à redundância foi barrado por leakage.
    vivos = tab[tab["motivo"] == "suspeita de leakage (revisar)"]
    assert not vivos.empty and bool(vivos["leakage_flag"].all())
    assert rep.selected_overall == []


def test_spark_regressao(spark, fs_cfg_rapido):
    from yggdrasil.feature_selection import run_feature_selection
    sdf = spark.createDataFrame(_make_pdf("regression"))
    rep = run_feature_selection(sdf, ColumnConfig(), fs_cfg_rapido,
                                books=["bureau", "bvs"], with_panels=False)
    assert rep.problem_type == "regression"
    # em regressão não há IV/KS; corr_target/RF devem orientar a seleção
    assert rep.overall_importance is not None
