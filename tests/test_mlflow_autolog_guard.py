"""
Guard de autologging do MLflow (:mod:`yggdrasil.utils.mlflow_guard`).

Contexto do bug: nos runtimes ML do Databricks o autologging vem ligado por
padrão e abre um run MLflow a cada ``fit`` do sklearn — inclusive os de
bastidor. Abrir a ``ModelSegmenterUI`` roda o pré-binning do optbinning (uma
``DecisionTree`` por candidata), então o ranking de N variáveis virava N runs:
a interface só aparecia quando o último terminava, e o experimento enchia de
runs sem tags nem métricas de validação.

Os testes cobrem o comportamento do context manager (no-op, reentrância,
escape) e o efeito ponta-a-ponta com o autolog REAL ligado: nenhum run nos
ajustes internos, e o registro explícito (``tune_optuna(log_mlflow=True)``)
intacto.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from yggdrasil.utils import mlflow_autolog_status, sem_autolog, set_mlflow_autolog
from yggdrasil.utils import mlflow_guard


def _df(n: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    meses = pd.date_range("2023-01-01", periods=6, freq="MS")
    score = rng.beta(2.5, 3, n)
    renda = rng.gamma(2.0, 1500, n)
    idade = rng.integers(18, 75, n).astype(float)
    risco = np.clip(0.10 + 0.45 * (score - 0.5) + 1.5e-5 * renda, 0.02, 0.95)
    df = pd.DataFrame({"score": score, "renda": renda, "idade": idade,
                       "target": (rng.uniform(0, 1, n) < risco).astype(float)})
    df["dt_ref"] = rng.choice(meses, size=n)
    df["amostra"] = np.where(df["dt_ref"] >= meses[4], "OOT", "DES")
    return df


@pytest.fixture
def restaura_escolha():
    """Devolve o guard ao padrão (decidir pelo ambiente) ao fim do teste."""
    yield
    mlflow_guard._autolog_permitido = None


# ---------------------------------------------------------------- unidade
def test_noop_sem_mlflow_na_sessao(monkeypatch):
    """Sem mlflow carregado não há patch a suprimir — e o guard NÃO importa o
    mlflow só para checar (o import custa segundos na abertura da UI)."""
    sem_mlflow = {k: v for k, v in sys.modules.items() if not k.startswith("mlflow")}
    monkeypatch.setattr(sys, "modules", sem_mlflow)
    with sem_autolog() as suprimido:
        assert suprimido is False
    assert mlflow_autolog_status()["mlflow_carregado"] is False


def test_reentrante_restaura_o_valor_anterior():
    """Blocos aninhados restauram o valor de FORA, não um ``False`` fixo — o
    tuning roda numa thread de fundo com o guard já ligado por quem chamou."""
    au = pytest.importorskip("mlflow.utils.autologging_utils")
    with sem_autolog() as externo:
        assert externo is True
        assert au._AUTOLOGGING_GLOBALLY_DISABLED is True
        with sem_autolog():
            assert au._AUTOLOGGING_GLOBALLY_DISABLED is True
        assert au._AUTOLOGGING_GLOBALLY_DISABLED is True, "bloco interno vazou"
    assert au._AUTOLOGGING_GLOBALLY_DISABLED is False


def test_escape_por_codigo_e_por_ambiente(monkeypatch, restaura_escolha):
    pytest.importorskip("mlflow.utils.autologging_utils")
    set_mlflow_autolog(True)
    with sem_autolog() as suprimido:
        assert suprimido is False
    set_mlflow_autolog(False)
    with sem_autolog() as suprimido:
        assert suprimido is True

    mlflow_guard._autolog_permitido = None            # volta a decidir pelo ambiente
    monkeypatch.setenv(mlflow_guard.ENV_AUTOLOG, "1")
    with sem_autolog() as suprimido:
        assert suprimido is False


def test_status_reporta_integracoes_ativas():
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("sklearn")
    mlflow.sklearn.autolog()
    try:
        st = mlflow_autolog_status()
        assert "sklearn" in st["integracoes"]
        assert st["suprimido"] is True
    finally:
        mlflow.sklearn.autolog(disable=True)


# ------------------------------------------------------------ ponta-a-ponta
def _runs(mlflow, experimento: str) -> int:
    exp = mlflow.get_experiment_by_name(experimento)
    if exp is None:
        return 0
    return len(mlflow.search_runs([exp.experiment_id]))


def _esteira(df, permitir_autolog: bool):
    """O caminho da abertura da UI + treino + ratings: tudo binning/sklearn."""
    from yggdrasil.credit_risk.model import ModelSegmenter
    if permitir_autolog:
        set_mlflow_autolog(True)
    try:
        seg = ModelSegmenter(df, target="target", task_type="classification",
                             sample_col="amostra", ref_sample="DES", date_col="dt_ref")
        seg.variable_iv()                       # optbinning de TODAS as candidatas
        seg.fit()
        seg.build_ratings(method="arvore", n_ratings=4)
    finally:
        mlflow_guard._autolog_permitido = None


def test_ajustes_internos_nao_abrem_run(tmp_path, monkeypatch, restaura_escolha):
    """O bug: com o autolog do ambiente ligado, o ranking/treino/ratings abriam
    um run por ``fit``. Com o guard, nenhum."""
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("optbinning")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")   # mlflow 3.x: file store
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")
    df = _df()

    mlflow.sklearn.autolog()
    try:
        mlflow.set_experiment("sem_guard")       # controle: o autolog pega mesmo?
        _esteira(df, permitir_autolog=True)
        n_sem_guard = _runs(mlflow, "sem_guard")
        if n_sem_guard == 0:
            pytest.skip("autolog do mlflow não instrumenta este ambiente")

        mlflow.set_experiment("com_guard")
        _esteira(df, permitir_autolog=False)
        assert _runs(mlflow, "com_guard") == 0, (
            f"{_runs(mlflow, 'com_guard')} run(s) automáticos nos ajustes internos "
            f"(sem o guard eram {n_sem_guard})")
    finally:
        mlflow.sklearn.autolog(disable=True)


def test_registro_explicito_continua_funcionando(tmp_path, monkeypatch):
    """O guard desliga só o AUTOlog: ``mlflow.start_run``/``log_*`` explícitos
    seguem valendo — o tuning com ``log_mlflow=True`` registra pai + trials."""
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("optbinning")
    pytest.importorskip("optuna")
    from yggdrasil.credit_risk.model import ModelSegmenter
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")

    mlflow.sklearn.autolog()
    try:
        seg = ModelSegmenter(_df(), target="target", task_type="classification",
                             sample_col="amostra", ref_sample="DES", date_col="dt_ref")
        seg.tune_optuna(algorithm="logistica", n_trials=2, fit_best=True,
                        log_mlflow=True, mlflow_experiment="tuning_explicito")
    finally:
        mlflow.sklearn.autolog(disable=True)

    exp = mlflow.get_experiment_by_name("tuning_explicito")
    runs = mlflow.search_runs([exp.experiment_id])
    assert len(runs) == 3, "esperado 1 run pai + 1 por trial"
    assert (runs["tags.grupo"] == "tuning-optuna").any()
