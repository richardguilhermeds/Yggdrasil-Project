"""Testes da matriz de migração de ratings entre safras (monitoring.migration)."""

import numpy as np
import pandas as pd
import pytest

from yggdrasil.config import ColumnConfig
from yggdrasil.monitoring import plot_migration_matrix, rating_migration_matrix


def _painel_2_safras():
    """Painel sintético de 2 safras (DES em jan, OOT em fev) com id de entidade.

    Migrações esperadas (labels A < B < C): id1 A→A, id2 A→B (downgrade),
    id3 B→A (upgrade), id4 B→B, id5 C→B (upgrade), id6 C→C; id7 só na safra 1
    (saída) e id8 só na safra 2 (entrada).
    """
    linhas = [
        (1, "2024-01-01", "DES", "A"), (1, "2024-02-01", "OOT", "A"),
        (2, "2024-01-01", "DES", "A"), (2, "2024-02-01", "OOT", "B"),
        (3, "2024-01-01", "DES", "B"), (3, "2024-02-01", "OOT", "A"),
        (4, "2024-01-01", "DES", "B"), (4, "2024-02-01", "OOT", "B"),
        (5, "2024-01-01", "DES", "C"), (5, "2024-02-01", "OOT", "B"),
        (6, "2024-01-01", "DES", "C"), (6, "2024-02-01", "OOT", "C"),
        (7, "2024-01-01", "DES", "B"),
        (8, "2024-02-01", "OOT", "C"),
    ]
    df = pd.DataFrame(linhas, columns=["id_ctr", "dt_ref", "amostra", "rating_teste"])
    df["dt_ref"] = pd.to_datetime(df["dt_ref"])
    return df


@pytest.fixture
def cfg_id():
    return ColumnConfig(id_col="id_ctr")


def test_matriz_contagem_pct_e_resumo(cfg_id):
    res = rating_migration_matrix(
        _painel_2_safras(), "rating_teste", cfg_id, "DES", "OOT")
    assert res.labels == ["A", "B", "C"]
    # contagens célula a célula (só entidades presentes nos dois períodos)
    esperado = pd.DataFrame(
        [[1, 1, 0], [1, 1, 0], [0, 1, 1]],
        index=pd.Index(["A", "B", "C"], name="de"),
        columns=pd.Index(["A", "B", "C"], name="para"),
    )
    pd.testing.assert_frame_equal(res.counts, esperado, check_dtype=False)
    assert res.counts.to_numpy().sum() == 6
    # % por linha de origem: cada linha soma 100
    assert res.pct.sum(axis=1).to_numpy() == pytest.approx([100.0] * 3)
    assert res.pct.loc["C", "B"] == pytest.approx(50.0)
    # resumo: diagonal 3/6, upgrade 2/6 (B→A, C→B), downgrade 1/6 (A→B)
    assert res.summary["n_comum"] == 6
    assert res.summary["pct_diagonal"] == pytest.approx(50.0)
    assert res.summary["pct_upgrade"] == pytest.approx(100 * 2 / 6, abs=1e-3)
    assert res.summary["pct_downgrade"] == pytest.approx(100 * 1 / 6, abs=1e-3)
    # nota sobre entradas/saídas da base
    assert res.summary["n_saidas"] == 1 and res.summary["n_entradas"] == 1
    assert "saíram" in res.nota and "entraram" in res.nota


def test_periodos_por_safra_equivalem_a_amostras(cfg_id):
    df = _painel_2_safras()
    por_amostra = rating_migration_matrix(df, "rating_teste", cfg_id, "DES", "OOT")
    por_safra = rating_migration_matrix(df, "rating_teste", cfg_id, "2024-01", "2024-02")
    pd.testing.assert_frame_equal(por_amostra.counts, por_safra.counts)
    assert por_safra.period_a == "jan/24" and por_safra.period_b == "fev/24"


def test_duplicata_no_periodo_usa_ultima_observacao(cfg_id):
    df = _painel_2_safras()
    # id1 ganha uma observação mais recente dentro da safra 1, com rating B
    extra = pd.DataFrame(
        [(1, pd.Timestamp("2024-01-15"), "DES", "B")], columns=df.columns)
    res = rating_migration_matrix(
        pd.concat([df, extra], ignore_index=True), "rating_teste", cfg_id,
        "DES", "OOT")
    assert res.counts.loc["B", "A"] == 2      # id1 vira B→A (junto do id3)
    assert res.counts.loc["A", "A"] == 0


def test_ordem_dos_labels_define_direcao(cfg_id):
    # ordem invertida: C < B < A ⇒ upgrades/downgrades trocam de papel
    res = rating_migration_matrix(
        _painel_2_safras(), "rating_teste", cfg_id, "DES", "OOT",
        labels=["C", "B", "A"])
    assert res.summary["pct_upgrade"] == pytest.approx(100 * 1 / 6, abs=1e-3)
    assert res.summary["pct_downgrade"] == pytest.approx(100 * 2 / 6, abs=1e-3)
    assert res.summary["pct_diagonal"] == pytest.approx(50.0)


def test_labels_incompletos_erro(cfg_id):
    with pytest.raises(ValueError, match="labels"):
        rating_migration_matrix(
            _painel_2_safras(), "rating_teste", cfg_id, "DES", "OOT",
            labels=["A", "B"])


def test_id_col_nao_configurado_erro():
    with pytest.raises(ValueError, match="id_col"):
        rating_migration_matrix(
            _painel_2_safras(), "rating_teste", ColumnConfig(), "DES", "OOT")


def test_id_col_inexistente_no_df_erro():
    cfg = ColumnConfig(id_col="nao_existe")
    with pytest.raises(ValueError, match="nao_existe"):
        rating_migration_matrix(
            _painel_2_safras(), "rating_teste", cfg, "DES", "OOT")


def test_periodo_invalido_e_vazio_erro(cfg_id):
    df = _painel_2_safras()
    with pytest.raises(ValueError, match="reconhec"):
        rating_migration_matrix(df, "rating_teste", cfg_id, "não-é-período", "OOT")
    with pytest.raises(ValueError, match="Nenhuma linha"):
        rating_migration_matrix(df, "rating_teste", cfg_id, "2050-01", "OOT")


def test_sem_entidade_em_comum_erro(cfg_id):
    df = _painel_2_safras()
    # desloca os ids da safra 2 → nenhum id repetido entre períodos
    ids = np.where(df["amostra"] == "OOT", df["id_ctr"] + 100, df["id_ctr"])
    df["id_ctr"] = pd.Series(ids, index=df.index)
    with pytest.raises(ValueError, match="em comum"):
        rating_migration_matrix(df, "rating_teste", cfg_id, "DES", "OOT")


def test_plot_migration_matrix(cfg_id, tmp_path):
    import matplotlib.pyplot as plt

    res = rating_migration_matrix(
        _painel_2_safras(), "rating_teste", cfg_id, "DES", "OOT")
    caminho = tmp_path / "mig.png"
    fig = plot_migration_matrix(res, save_path=str(caminho))
    assert fig is not None and caminho.exists()
    plt.close(fig)
    fig2 = plot_migration_matrix(res, values="counts")
    plt.close(fig2)
    with pytest.raises(ValueError):
        plot_migration_matrix(res, values="xxx")


# ── integração com a esteira/MLflow ────────────────────────────────────────


def _painel_pipeline(n_ids=250, seed=0):
    """Painel com as mesmas entidades em DES (jan/23) e OOT (ago/23)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_ids, 6))
    frames = []
    for amostra, data in (("DES", "2023-01-01"), ("OOT", "2023-08-01")):
        X = base + rng.normal(0, 0.3, size=base.shape)      # entidade estável + ruído
        df = pd.DataFrame(X, columns=[f"feat_{i:02d}" for i in range(6)])
        df["id_ctr"] = np.arange(n_ids)
        df["dt_ref"] = pd.Timestamp(data)
        df["amostra"] = amostra
        lin = X[:, :3].sum(axis=1)
        df["target"] = (lin + rng.normal(0, 1.0, n_ids) > 0).astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_pipeline_loga_artefato_de_migracao(tmp_path, monkeypatch):
    import mlflow
    from mlflow.tracking import MlflowClient
    from sklearn.ensemble import RandomForestClassifier

    from yggdrasil import MLPipeline

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = f"file:{(tmp_path / 'mlruns').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    cfg = ColumnConfig(id_col="id_ctr")
    df = _painel_pipeline()
    feats = cfg.feature_columns(df)
    dev = df[df["amostra"] == "DES"]
    model = RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=-1).fit(
        dev[feats], dev["target"])

    pipe = MLPipeline(cfg, problem_type="classification", ratings=["decis"])
    res = pipe.run(df, model=model, experiment="migration_test", run_name="t",
                   log_shap=False)

    client = MlflowClient(tracking_uri=tracking_uri)
    artefatos = {a.path for a in client.list_artifacts(res.run_id)}
    assert "migration" in artefatos
    arquivos = {a.path for a in client.list_artifacts(res.run_id, "migration")}
    assert "migration/migration_decis_counts.csv" in arquivos
    assert "migration/migration_decis_pct.csv" in arquivos
    assert "migration/migration_decis.png" in arquivos
    assert "migration/migration_decis_resumo.json" in arquivos
