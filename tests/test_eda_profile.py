"""Testes do perfil univariado (missing, percentis, cardinalidade, outliers)."""

import numpy as np

from yggdrasil import ColumnConfig
from yggdrasil.eda import EDAConfig
from yggdrasil.eda import profile


def test_missing_summary(df_eda):
    cfg, ec = ColumnConfig(), EDAConfig()
    ms = profile.missing_summary(df_eda, cfg, ec).set_index("feature")
    assert ms.loc["feat_missing", "pct_missing"] > 0.4          # ~60% missing
    assert ms.loc["feat_missing", "flag"] == "descartar"
    assert ms.loc["feat_bom", "pct_missing"] == 0.0


def test_missing_over_time(df_eda):
    cfg, ec = ColumnConfig(), EDAConfig()
    tab = profile.missing_over_time(df_eda, "feat_missing", cfg, ec)
    assert {"periodo", "pct_missing", "n"}.issubset(tab.columns)
    assert len(tab) >= 1
    assert "delta_max" in tab.attrs


def test_percentile_table_has_geral_and_samples(df_eda):
    cfg, ec = ColumnConfig(), EDAConfig()
    pt = profile.percentile_table(df_eda, "feat_bom", cfg, ec)
    assert "GERAL" in pt.index and "DES" in pt.index and "OOT" in pt.index
    assert "p50" in pt.columns and "skew" in pt.columns


def test_cardinality_flags_constant(df_eda):
    cfg, ec = ColumnConfig(), EDAConfig()
    card = profile.cardinality_summary(df_eda, cfg.feature_columns(df_eda), ec).set_index("feature")
    assert bool(card.loc["feat_const", "constante"]) is True
    assert bool(card.loc["feat_bom", "constante"]) is False


def test_outlier_summary_keys(df_eda):
    o = profile.outlier_summary(df_eda["feat_bom"])
    for k in ("pct_outlier_iqr", "pct_outlier_mad", "kurtosis"):
        assert k in o


def test_dataset_overview(df_eda):
    cfg, ec = ColumnConfig(), EDAConfig()
    ov = profile.dataset_overview(df_eda, cfg, ec)
    assert ov["tem_target"] is True
    assert ov["problem_type"] == "classification"
    assert ov["n_features"] == len(cfg.feature_columns(df_eda))
