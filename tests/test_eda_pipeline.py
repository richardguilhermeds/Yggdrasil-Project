"""Smoke da esteira de EDA de features end-to-end."""

import numpy as np
import pandas as pd

from yggdrasil import ColumnConfig
from yggdrasil.eda import EDAConfig, FeatureEDAReport, run_feature_eda


def test_run_feature_eda_classification(df_eda):
    rep = run_feature_eda(df_eda, ColumnConfig(), EDAConfig(), with_panels=True)
    assert isinstance(rep, FeatureEDAReport)
    assert rep.problem_type == "classification"

    prof = rep.feature_profile.set_index("feature")
    # uma linha por feature
    assert set(prof.index) == set(ColumnConfig().feature_columns(df_eda))
    # vereditos válidos
    assert set(prof["veredito"]).issubset({"manter", "revisar", "descartar"})
    # diagnósticos esperados
    assert prof.loc["feat_const", "veredito"] == "descartar"      # constante
    assert prof.loc["feat_missing", "veredito"] == "descartar"    # 60% missing
    assert prof.loc["feat_instavel", "veredito"] == "descartar"   # PSI alto
    assert bool(prof.loc["feat_leakage", "leakage"]) is True

    # HTML e painéis
    assert len(rep.to_html()) > 100
    assert len(rep.panels) >= 1


def test_run_feature_eda_regression():
    rng = np.random.default_rng(1)
    n = 1200
    meses = pd.date_range("2023-01-01", periods=8, freq="MS")
    df = pd.DataFrame({"feat_a": rng.normal(size=n), "feat_b": rng.normal(size=n)})
    df["target"] = df["feat_a"] * 2 + rng.normal(0, 0.5, n)      # contínuo
    df["dt_ref"] = rng.choice(meses, size=n)
    df["amostra"] = np.where(df["dt_ref"] >= meses[6], "OOT", "DES")

    rep = run_feature_eda(df, ColumnConfig(), EDAConfig(), with_panels=False)
    assert rep.problem_type == "regression"
    assert "psi_oot" in rep.feature_profile.columns
    assert len(rep.feature_profile) == 2


def test_run_feature_eda_sem_target():
    """EDA pura: sem target, capacidades dependentes do alvo são puladas."""
    rng = np.random.default_rng(2)
    n = 800
    meses = pd.date_range("2023-01-01", periods=6, freq="MS")
    df = pd.DataFrame({"feat_a": rng.normal(size=n), "feat_b": rng.normal(size=n)})
    df["dt_ref"] = rng.choice(meses, size=n)
    df["amostra"] = np.where(df["dt_ref"] >= meses[4], "OOT", "DES")

    rep = run_feature_eda(df, ColumnConfig(), EDAConfig(), with_panels=False)
    assert rep.problem_type is None
    # ainda produz perfil com missing/PSI
    assert "pct_missing" in rep.feature_profile.columns
    assert "psi_oot" in rep.feature_profile.columns
