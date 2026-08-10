"""Smoke test end-to-end da esteira com logging real no MLflow (file store local)."""

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier

from yggdrasil import ColumnConfig, MLPipeline
from yggdrasil.data import validate_input_report


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
    flags = {k: v for k, v in res.shifts.items() if k.endswith("_shift_flag")}
    assert flags and set(flags.values()) <= {"ok", "atencao", "degradado"}
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
    # flags de degradação dos shifts registradas como tags do run
    assert any(k.endswith("_shift_flag") for k in run.data.tags)
    # flags não entram como métrica (são strings)
    assert not any(k.endswith("_shift_flag") for k in metricas)
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


def test_validate_input_report_sem_findings(df_clf, cfg):
    assert validate_input_report(df_clf, cfg, problem_type="classification") == []


def test_validate_input_report_findings(df_clf, cfg):
    df = df_clf.copy()
    df[cfg.score_col] = 0.5                              # score pré-existente
    df["feat_texto"] = "x"                               # feature não numérica
    df["target"] = df["target"].astype(float)            # permite NaN sem upcast
    idx_des = df.index[df["amostra"] == "DES"][:3]
    df.loc[idx_des, "target"] = np.nan                   # NaN no target (análise)
    df["dt_ref"] = df["dt_ref"].astype(str)
    df.loc[df.index[0], "dt_ref"] = "não-é-data"        # data inválida

    findings = validate_input_report(df, cfg, problem_type="regression")
    sev_col = {(sev, col) for sev, col, _ in findings}
    assert ("erro", cfg.date_col) in sev_col             # data não conversível
    assert ("erro", cfg.target_col) in sev_col           # NaN no target
    assert ("atencao", "feat_texto") in sev_col          # feature não numérica
    assert ("atencao", cfg.score_col) in sev_col         # score será sobrescrito
    # target binário com problem_type='regression' => divergência apontada
    assert any(col == cfg.target_col and "diverge" in msg for _, col, msg in findings)


def test_validate_input_report_amostra_nan_e_indice_duplicado(df_clf, cfg):
    import pandas as pd

    df = pd.concat([df_clf, df_clf.iloc[[0]]])           # índice duplicado
    df.loc[df.index[:2], "amostra"] = None               # amostra NaN
    findings = validate_input_report(df, cfg)
    colunas = {col for _, col, _ in findings}
    assert cfg.sample_col in colunas
    assert "(índice)" in colunas


def test_pipeline_imprime_resumo_findings(df_clf, cfg, capsys):
    from sklearn.linear_model import LogisticRegression

    df = df_clf.copy()
    df[cfg.score_col] = 0.5                              # gera 1 finding de atenção
    feats = cfg.feature_columns(df)
    dev = df[df["amostra"] == "DES"]
    model = LogisticRegression(max_iter=500).fit(dev[feats], dev["target"])

    pipe = MLPipeline(cfg, problem_type="classification", ratings=["decis"])
    pipe.run(df, model=model, log_mlflow=False, log_shap=False)
    saida = capsys.readouterr().out
    assert "finding" in saida and cfg.score_col in saida
