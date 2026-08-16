"""Fixtures de dados sintéticos para os testes da esteira."""

import os
import sys

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


@pytest.fixture(autouse=True)
def _isola_estado_mlflow():
    """Impede que um teste que troca o *tracking* do MLflow contamine os seguintes.

    Vários testes apontam o tracking para um ``tmp_path`` e alguns não restauram o
    valor anterior. Pior: ``set_experiment()`` fixa o experimento ativo em DOIS
    lugares globais — a variável de ambiente ``MLFLOW_EXPERIMENT_ID`` e o cache de
    módulo ``fluent._active_experiment_id``. Um ``start_run()`` sem experimento
    explícito, num teste posterior, tenta reusar esse ID, que só existia no
    diretório temporário (já apagado) do teste anterior, e falha com
    ``Could not find experiment with ID`` / ``does not exist in the tracking
    server``. Restaurar tracking + experimento ativo deixa a suíte independente da
    ordem de execução.

    Só age quando o MLflow já foi importado — quem não usa MLflow não paga nada.
    """
    _VARS = ("MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_TRACKING_URI")
    mlflow = sys.modules.get("mlflow")
    uri = mlflow.get_tracking_uri() if mlflow is not None else None
    fluent = sys.modules.get("mlflow.tracking.fluent")
    exp = getattr(fluent, "_active_experiment_id", None) if fluent is not None else None
    env = {k: os.environ.get(k) for k in _VARS}
    try:
        yield
    finally:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        mod = sys.modules.get("mlflow")
        if mod is None:
            return
        if uri is not None:
            mod.set_tracking_uri(uri)
        fl = sys.modules.get("mlflow.tracking.fluent")
        if fl is not None:
            # None = "sem experimento ativo": o próximo start_run() resolve o
            # experimento no store VIGENTE em vez de reusar o ID do anterior.
            try:
                fl._active_experiment_id = exp
            except Exception:                     # pragma: no cover - API interna
                pass


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


# ----------------------------------------------------------------------
# Fixtures do subpacote de ECL (yggdrasil.credit_risk.ecl)
# ----------------------------------------------------------------------
def _painel_credito(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Painel longo de contratos com **DGP conhecido** — a base dos testes de ECL.

    Dois produtos com níveis de risco bem separados e **maturação** (o *hazard*
    cresce com a idade), rating correlacionado ao risco, prazo remanescente e as
    colunas de rotativo (sacado/limite) com aceleração do saque perto do
    *default* — o que dá um CCF positivo e não degenerado.
    """
    rng = np.random.default_rng(seed)
    base_h = {"cartao": 0.014, "consignado": 0.004}
    linhas = []
    for i in range(n):
        produto = "cartao" if rng.uniform() < 0.5 else "consignado"
        rating = str(rng.choice(["A", "B", "C"], p=[0.5, 0.3, 0.2]))
        origem = pd.Timestamp("2020-01-01") + pd.DateOffset(months=int(rng.integers(0, 18)))
        prazo = int(rng.integers(12, 49))
        limite = float(rng.lognormal(8.5, 0.5))
        sacado = limite * float(rng.beta(2.0, 3.0))
        mult = {"A": 0.6, "B": 1.0, "C": 1.8}[rating]
        for t in range(36):
            h = base_h[produto] * mult * (1.0 + t / 20.0)
            quebrou = int(rng.uniform() < min(h, 0.9))
            linhas.append((f"C{i:05d}", origem + pd.DateOffset(months=t), origem, quebrou,
                           produto, rating, max(prazo - t, 0), sacado, limite))
            if quebrou:
                break
            # Uso do limite sobe devagar e acelera quando o risco corrente é alto.
            passo = 1.0 + rng.normal(0.012 + 2.0 * h, 0.05)
            sacado = float(np.clip(sacado * passo, 0.0, limite))
    return pd.DataFrame(linhas, columns=[
        "id_contrato", "dt_ref", "safra_origem", "default", "produto", "rating",
        "prazo", "sacado", "limite",
    ])


@pytest.fixture
def df_credito() -> pd.DataFrame:
    """Painel longo de contratos (produto, rating, prazo, sacado/limite)."""
    return _painel_credito()


@pytest.fixture
def painel(df_credito):
    """O mesmo painel já embrulhado num ``ContractPanel``."""
    from yggdrasil.credit_risk.ecl import ContractPanel

    return ContractPanel(df_credito, origin_col="safra_origem", segment_col="produto",
                         term_col="prazo")


@pytest.fixture
def carteira_viva(df_credito) -> pd.DataFrame:
    """Última observação de cada contrato + saldo, LGD, taxa, estágio e ELBE."""
    rng = np.random.default_rng(7)
    d = df_credito.groupby("id_contrato").tail(1).copy()
    d["idade"] = (pd.PeriodIndex(d["dt_ref"], freq="M").asi8
                  - pd.PeriodIndex(d["safra_origem"], freq="M").asi8)
    d["saldo"] = rng.lognormal(9.0, 0.5, len(d))
    d["lgd"] = np.where(d["produto"] == "cartao", 0.75, 0.35)
    d["taxa_efetiva"] = np.where(d["produto"] == "cartao", 2.5, 0.25)
    d["estagio"] = rng.choice([1, 2, 3], len(d), p=[0.7, 0.2, 0.1])
    d["elbe"] = 0.8
    d["meses_em_default"] = rng.integers(0, 30, len(d))
    return d.reset_index(drop=True)


@pytest.fixture
def df_defaults_lgd() -> pd.DataFrame:
    """Contratos em *default* no formato **largo** (``exposicao_inicial`` + ``lgd_m*``).

    Recuperação exponencial com assíntota por segmento e **censura à direita**
    (os *defaults* recentes ainda não chegaram aos meses altos) — a situação em
    que a coorte variável distorce a curva se não for encadeada.
    """
    rng = np.random.default_rng(11)
    n, meses = 500, 37
    segmento = np.where(rng.uniform(size=n) < 0.6, "cartao", "consignado")
    ead0 = rng.lognormal(8.5, 0.6, n)
    assintota = np.clip(np.where(segmento == "cartao", 0.25, 0.65)
                        * rng.uniform(0.8, 1.2, n), 0.0, 0.95)
    velocidade = np.where(segmento == "cartao", 0.10, 0.18)
    t = np.arange(meses)
    r = assintota[:, None] * (1.0 - np.exp(-velocidade[:, None] * t[None, :]))
    observado = t[None, :] < rng.integers(3, meses + 1, n)[:, None]
    r = np.where(observado, r, np.nan)

    df = pd.DataFrame({"id_contrato": [f"D{i:04d}" for i in range(n)],
                       "segmento": segmento, "exposicao_inicial": ead0})
    for k in range(meses):
        df[f"lgd_m{k}"] = 1.0 - r[:, k]
    return df


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
