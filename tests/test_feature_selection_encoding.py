"""Features não numéricas na seleção: aviso, motivo explícito e codificação opt-in."""

import logging

import numpy as np
import pandas as pd
import pytest

from yggdrasil import ColumnConfig
from yggdrasil.feature_selection import (
    FeatureSelectionConfig,
    apply_encodings,
    fit_encodings,
    run_feature_selection,
)
from yggdrasil.feature_selection.encoding import MISSING, classify_columns

CFG = ColumnConfig(feature_prefix="", target_col="mau", sample_col="amostra")
FEATS = ["score", "uf", "uf_cat", "flag", "ruido"]


def _make_pdf(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """``uf`` (texto) carrega risco forte; ``flag`` (bool) também; ``ruido`` não."""
    rng = np.random.default_rng(seed)
    uf = rng.choice(["SP", "RJ", "MG", "BA"], n)
    flag = rng.random(n) < 0.3
    score = rng.normal(size=n)
    logit = -1.5 - 0.6 * score + np.select([uf == "BA", uf == "RJ"], [2.0, 1.0], 0) + 1.2 * flag
    return pd.DataFrame({
        "mau": rng.binomial(1, 1 / (1 + np.exp(-logit))),
        "amostra": "DES",
        "score": score,
        "uf": uf,
        "uf_cat": pd.Categorical(uf),
        "flag": flag,
        "ruido": rng.normal(size=n),
    })


@pytest.fixture
def fs_rapido():
    return dict(boruta_max_iter=10, rf_n_estimators=40, rf_max_depth=5)


# ───────────────────────── tipos ─────────────────────────
def test_classify_columns_pandas():
    bools, cats = classify_columns(_make_pdf(50), FEATS)
    assert bools == ["flag"]
    assert cats == ["uf", "uf_cat"]


# ───────────────────────── sem codificação ─────────────────────────
def test_sem_codificacao_motivo_explicito_e_aviso(fs_rapido, caplog):
    with caplog.at_level(logging.WARNING, logger="yggdrasil.feature_selection"):
        rep = run_feature_selection(_make_pdf(), CFG, FeatureSelectionConfig(**fs_rapido),
                                    books={"todas": FEATS}, with_panels=False)
    tab = rep.selection_table.set_index("feature")
    for f in ("uf", "uf_cat", "flag"):
        assert tab.loc[f, "motivo"] == "não numérica (não avaliada)"
        assert not tab.loc[f, "selecionada"]
        assert pd.isna(tab.loc[f, "encoding"])
    assert "encode_categoricals=True" in caplog.text
    assert rep.encodings == {}


# ───────────────────────── com codificação ─────────────────────────
def test_codificacao_seleciona_texto_e_bool(fs_rapido):
    pdf = _make_pdf()
    rep = run_feature_selection(pdf, CFG, FeatureSelectionConfig(encode_categoricals=True, **fs_rapido),
                                books={"todas": FEATS}, with_panels=False)
    tab = rep.selection_table.set_index("feature")
    # uf e uf_cat são a mesma informação: uma é selecionada, a outra sai por redundância
    assert tab.loc[["uf", "uf_cat"], "selecionada"].sum() == 1
    assert tab.loc[["uf", "uf_cat"], "iv"].notna().all()
    assert tab.loc["flag", "selecionada"]
    assert tab.loc["uf", "encoding"] == "target_encoding"
    assert tab.loc["flag", "encoding"] == "bool_0_1"
    assert pd.isna(tab.loc["score", "encoding"])
    # a entrada do usuário não é alterada
    assert pdf["uf"].dtype == object and pdf["flag"].dtype == bool


def test_target_encoding_e_a_taxa_por_categoria():
    pdf = _make_pdf()
    enc = fit_encodings(pdf, ["uf"], "mau", FeatureSelectionConfig())["uf"]
    taxa = pdf.groupby("uf")["mau"].mean()
    for uf, v in taxa.items():
        assert enc["mapa"][uf] == pytest.approx(v, abs=1e-6)
    assert enc["prior"] == pytest.approx(pdf["mau"].mean(), abs=1e-6)
    assert enc["n_raras"] == 0 and enc["outros"] == enc["prior"]


def test_raras_viram_outros_e_nulo_e_categoria():
    pdf = pd.DataFrame({
        "mau": [1, 0] * 50 + [1, 1, 1, 0, 0],
        "c": ["A"] * 50 + ["B"] * 50 + ["X", "Y", "Z", None, None],
    })
    enc = fit_encodings(pdf, ["c"], "mau", FeatureSelectionConfig(encoding_min_share=0.03))["c"]
    assert set(enc["mapa"]) == {"A", "B"}          # X/Y/Z/(vazio) (<3%) viram OUTROS
    assert enc["n_raras"] == 4
    assert enc["outros"] == pytest.approx(0.6)     # X, Y, Z maus; os 2 nulos bons
    novo = apply_encodings(pd.DataFrame({"c": ["A", "Z", "NUNCA_VISTA", None]}), {"c": enc})
    assert novo["c"].tolist() == pytest.approx([0.5, 0.6, 0.6, 0.6])


def test_nulo_frequente_vira_categoria_propria():
    pdf = pd.DataFrame({"mau": [1] * 40 + [0] * 60, "c": [None] * 40 + ["A"] * 60})
    enc = fit_encodings(pdf, ["c"], "mau", FeatureSelectionConfig())["c"]
    assert enc["mapa"][MISSING] == pytest.approx(1.0)
    assert enc["mapa"]["A"] == pytest.approx(0.0)


def test_alta_cardinalidade_nao_e_codificada(fs_rapido, caplog):
    pdf = _make_pdf(600)
    pdf["cpf"] = [f"{i:011d}" for i in range(len(pdf))]
    cfg = FeatureSelectionConfig(encode_categoricals=True, encoding_max_categories=100, **fs_rapido)
    with caplog.at_level(logging.WARNING, logger="yggdrasil.feature_selection"):
        rep = run_feature_selection(pdf, CFG, cfg, books={"todas": ["score", "cpf"]}, with_panels=False)
    assert "cpf" not in rep.encodings
    assert "encoding_max_categories" in caplog.text
    assert rep.selection_table.set_index("feature").loc["cpf", "motivo"] == "não numérica (não avaliada)"


def test_report_reaplica_codificacao_em_outra_base(fs_rapido):
    rep = run_feature_selection(_make_pdf(), CFG, FeatureSelectionConfig(encode_categoricals=True, **fs_rapido),
                                books={"todas": FEATS}, with_panels=False)
    oot = _make_pdf(200, seed=9)
    out = rep.apply_encodings(oot)
    assert out["flag"].tolist() == oot["flag"].astype(float).tolist()
    assert out["uf"].map(lambda v: isinstance(v, float)).all()
    assert out["uf"].iloc[0] == rep.encodings["uf"]["mapa"][oot["uf"].iloc[0]]
    assert oot["uf"].dtype == object  # não muta


def test_config_valida_parametros_de_encoding():
    with pytest.raises(ValueError):
        FeatureSelectionConfig(encoding_min_share=1.0)
    with pytest.raises(ValueError):
        FeatureSelectionConfig(encoding_max_categories=0)


# ───────────────────────── andamento (verbose) ─────────────────────────
def test_verbose_imprime_cada_etapa_e_false_silencia(fs_rapido, capsys):
    pdf = _make_pdf(800)
    fs = FeatureSelectionConfig(encode_categoricals=True, **fs_rapido)
    rep = run_feature_selection(pdf, CFG, fs, books={"todas": FEATS}, with_panels=False)
    out = capsys.readouterr().out
    for i, etapa in enumerate(["Missing", "Variância", "Importância", "Redundância",
                               "Boruta", "Consenso"], 1):
        assert f"[{i}/6] {etapa}" in out
    assert "Codificação:" in out and "Seleção concluída" in out
    assert f"{len(rep.selected_overall)} de {len(FEATS)} feature(s) selecionada(s)" in out

    run_feature_selection(pdf, CFG, fs, books={"todas": FEATS}, with_panels=False, verbose=False)
    assert capsys.readouterr().out == ""


# ───────────────────────── paridade pandas × Spark ─────────────────────────
def test_paridade_encoding_com_spark():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    ja_existia = SparkSession.getActiveSession() is not None
    spark = (SparkSession.builder.master("local[2]").appName("ygg-fsel-encoding")
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        pdf = _make_pdf(800).drop(columns=["uf_cat"])
        pdf.loc[:30, "uf"] = None
        sdf = spark.createDataFrame(pdf)
        cfg = FeatureSelectionConfig(encoding_min_share=0.05)
        feats = ["score", "uf", "flag"]
        assert classify_columns(sdf, feats) == (["flag"], ["uf"])
        enc_pd = fit_encodings(pdf, feats, "mau", cfg)
        enc_sp = fit_encodings(sdf, feats, "mau", cfg)
        out_sp = apply_encodings(sdf, enc_sp).toPandas()
    finally:
        if not ja_existia:
            spark.stop()

    assert enc_pd == enc_sp
    out_pd = apply_encodings(pdf, enc_pd)
    np.testing.assert_allclose(out_sp["uf"].to_numpy(float), out_pd["uf"].to_numpy(float))
    np.testing.assert_allclose(out_sp["flag"].to_numpy(float), out_pd["flag"].to_numpy(float))
