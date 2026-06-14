"""Fixtures de dados sintéticos para os testes da esteira."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # sem display nos testes

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.datasets import make_classification, make_regression  # noqa: E402

from yggdrasil import ColumnConfig  # noqa: E402


def _synthetic(problem: str, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    meses = pd.date_range("2023-01-01", periods=10, freq="MS")
    if problem == "classification":
        X, y = make_classification(
            n_samples=n, n_features=6, n_informative=4, weights=[0.8], random_state=seed
        )
    else:
        X, y = make_regression(
            n_samples=n, n_features=6, n_informative=4, noise=10.0, random_state=seed
        )
        y = (y - y.min()) / (y.max() - y.min())

    df = pd.DataFrame(X, columns=[f"feat_{i:02d}" for i in range(6)])
    df["target"] = y
    df["dt_ref"] = rng.choice(meses, size=n)
    df["amostra"] = np.where(df["dt_ref"] >= meses[7], "OOT", "DES")
    # Algumas linhas scoring-only (não devem entrar na análise).
    df.loc[df.sample(frac=0.05, random_state=1).index, "amostra"] = "SIMUL"
    return df


@pytest.fixture
def cfg() -> ColumnConfig:
    return ColumnConfig()


@pytest.fixture
def df_clf() -> pd.DataFrame:
    return _synthetic("classification")


@pytest.fixture
def df_reg() -> pd.DataFrame:
    return _synthetic("regression")


@pytest.fixture
def scored_clf(df_clf, cfg):
    """DataFrame de classificação já scorado e com os 4 ratings (sem MLflow)."""
    from sklearn.ensemble import RandomForestClassifier

    from yggdrasil import MLPipeline

    feats = cfg.feature_columns(df_clf)
    dev = df_clf[df_clf["amostra"] == "DES"]
    model = RandomForestClassifier(n_estimators=60, random_state=0, n_jobs=-1).fit(
        dev[feats], dev["target"]
    )
    pipe = MLPipeline(cfg, problem_type="classification",
                      ratings=["decis", "quantil", "arvore", "optbin"])
    return pipe.run(df_clf, model=model, log_mlflow=False, log_shap=False)


@pytest.fixture
def df_eda() -> pd.DataFrame:
    """Dataset de EDA com features variadas: boa, fraca, categórica, constante,
    instável (drift), alto-missing e com leakage."""
    rng = np.random.default_rng(0)
    n = 1500
    meses = pd.date_range("2023-01-01", periods=10, freq="MS")
    df = pd.DataFrame({
        "feat_bom": rng.normal(size=n),
        "feat_fraca": rng.normal(size=n),
        "feat_cat": rng.choice(["A", "B", "C", "D"], size=n, p=[0.5, 0.3, 0.15, 0.05]),
        "feat_const": 1.0,
        "feat_instavel": rng.normal(size=n),
        "feat_missing": rng.normal(size=n),
    })
    df["target"] = (df["feat_bom"] * 1.2 + rng.normal(0, 0.6, n) > 0).astype(int)
    df["feat_leakage"] = df["target"] + rng.normal(0, 0.01, n)
    df["dt_ref"] = rng.choice(meses, size=n)
    df["amostra"] = np.where(df["dt_ref"] >= meses[7], "OOT", "DES")
    df.loc[df["amostra"] == "OOT", "feat_instavel"] += 2.5            # drift
    df.loc[df.sample(frac=0.6, random_state=2).index, "feat_missing"] = np.nan
    df.loc[df.sample(frac=0.04, random_state=3).index, "amostra"] = "SIMUL"
    return df
