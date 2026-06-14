"""Testes de binning de feature e análise bivariada (WoE/IV, monotonicidade)."""

import numpy as np

from yggdrasil import ColumnConfig
from yggdrasil.eda import EDAConfig
from yggdrasil.eda import binning, bivariate


def test_bin_feature_numeric_has_missing_bin(df_eda):
    cfg, ec = ColumnConfig(), EDAConfig()
    bins, binner = binning.bin_feature(df_eda, "feat_missing", cfg, ec, "classification")
    assert bins.notna().all()
    assert "MISSING" in set(bins.astype(str))           # há missing nessa feature


def test_binning_table_woe_iv(df_eda):
    cfg, ec = ColumnConfig(), EDAConfig()
    tab = binning.binning_table(df_eda, "feat_bom", cfg, ec, "classification")
    assert {"bin", "n", "pct", "event_rate", "woe", "iv_parcial"}.issubset(tab.columns)
    assert tab.attrs["iv"] > 0
    assert np.isclose(tab["pct"].sum(), 1.0, atol=1e-3)


def test_categorical_groups_rare_levels(df_eda):
    cfg = ColumnConfig()
    ec = EDAConfig(rare_level_pct=0.10)                 # 'D' (~5%) vira OUTROS
    bins, _ = binning.bin_feature(df_eda, "feat_cat", cfg, ec, "classification")
    assert "OUTROS" in set(bins.astype(str))


def test_event_rate_by_bin_has_ci(df_eda):
    cfg, ec = ColumnConfig(), EDAConfig()
    tab = bivariate.event_rate_by_bin(df_eda, "feat_bom", cfg, ec, "classification")
    assert "event_rate" in tab.columns
    assert "lower_ci" in tab.columns and "upper_ci" in tab.columns


def test_monotonicity_strong_feature(df_eda):
    cfg, ec = ColumnConfig(), EDAConfig()
    tab = bivariate.event_rate_by_bin(df_eda, "feat_bom", cfg, ec, "classification")
    mono = bivariate.monotonicity_diagnostic(tab)
    # feat_bom determina o target -> relação fortemente monotônica
    assert mono["trend"] in ("crescente", "decrescente")
    assert abs(mono["spearman"]) > 0.9


def test_iv_power_bands():
    assert bivariate.iv_power(0.005) == "inutil"
    assert bivariate.iv_power(0.2) == "medio"
    assert bivariate.iv_power(0.9) == "suspeito_leakage"
