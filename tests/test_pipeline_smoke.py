"""Smoke test end-to-end da esteira com logging real no MLflow (file store local)."""

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier

from yggdrasil import ColumnConfig, MLPipeline


def test_pipeline_end_to_end_loga_no_mlflow(df_clf, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # backend de arquivos no mlflow 3.x
    cfg = ColumnConfig()
    tracking_uri = f"file:{(tmp_path / 'mlruns').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    feats = cfg.feature_columns(df_clf)
    dev = df_clf[df_clf["amostra"] == "DES"]
    model = RandomForestClassifier(n_estimators=60, random_state=0, n_jobs=-1).fit(
        dev[feats], dev["target"]
    )

    pipe = MLPipeline(cfg, problem_type="classification",
                      ratings=["decis", "quantil", "arvore", "optbin"])
    res = pipe.run(
        df_clf, model=model, experiment="smoke_test", run_name="t",
        log_shap=True,
    )

    # resultado em memória
    assert res.run_id is not None
    assert len(res.reports) == 4
    assert set(res.metrics_by_sample) == {"DES", "OOT"}
    assert "ks" in res.metrics_by_sample["OOT"]
    assert any(k.endswith("_shift_abs") for k in res.shifts)
    assert "psi_score_oot" in res.psi_metrics
    # scoring-only preservado no df final, com score e rating
    assert (res.df_scored["amostra"] == "SIMUL").any()
    assert res.df_scored[cfg.score_col].notna().all()

    # run efetivamente registrado com métricas DES/OOT
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(res.run_id)
    metricas = run.data.metrics
    assert "ks_des" in metricas and "ks_oot" in metricas
    assert "auc_oot" in metricas
    assert any(k.startswith("psi_rating_") for k in metricas)
    # modelo logado sem erro (no MLflow 3.x vira um "Logged Model", não artefato do run)
    assert "model_log_error" not in run.data.tags

    # artefatos esperados do run
    artefatos = {a.path for a in client.list_artifacts(res.run_id)}
    assert "reports" in artefatos
    assert "psi" in artefatos
    assert "dashboard" in artefatos


def test_pipeline_exige_modelo_ou_trainer(df_clf):
    pipe = MLPipeline(ColumnConfig(), problem_type="classification", ratings=["decis"])
    try:
        pipe.run(df_clf, log_mlflow=False)
    except ValueError as exc:
        assert "trainer" in str(exc).lower() or "model" in str(exc).lower()
    else:
        raise AssertionError("esperava ValueError sem modelo/trainer")
