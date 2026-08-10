"""Testes das metodologias de grupos homogêneos (ratings)."""

import json

import numpy as np
import pandas as pd
import pytest

from yggdrasil.ratings import (
    DecileRating,
    ManualScoreRating,
    OptBinningRating,
    PercentileRating,
    QuantileMonotonicRating,
    TreeRating,
    apply_ratings,
    build_ratings,
    rating_from_dict,
    ratings_to_dict,
)


def test_build_ratings_padrao_tem_quatro():
    estrategias = build_ratings()
    nomes = {e.name for e in estrategias}
    assert nomes == {"decis", "quantil", "arvore", "optbin"}


def test_decis_gera_ate_dez_grupos(df_clf, cfg):
    df = df_clf.copy()
    # score sintético = primeira feature (monotônico arbitrário)
    df[cfg.score_col] = df["feat_00"]
    strat = DecileRating()
    serie = strat.fit_transform(df, cfg, "classification")
    assert serie.notna().all()
    assert 1 < serie.nunique() <= 10
    assert all(str(v).startswith("R") for v in serie.unique())


def test_transform_rotula_todas_as_linhas(df_reg, cfg):
    df = df_reg.copy()
    df[cfg.score_col] = df["feat_00"]
    for strat in [DecileRating(), QuantileMonotonicRating(), TreeRating()]:
        strat.fit(df, cfg, "regression")
        serie = strat.transform(df, cfg)
        assert serie.notna().all()
        assert len(serie) == len(df)


def test_fusao_reduz_ou_mantem_grupos(df_reg, cfg):
    """A fusão monotônica nunca aumenta o número de grupos do quantil."""
    df = df_reg.copy()
    df[cfg.score_col] = df["feat_00"] + np.random.default_rng(0).normal(0, 0.01, len(df))
    strat = QuantileMonotonicRating(step=0.1)
    strat.fit(df, cfg, "regression")
    n_brutos = len(strat.edges_) - 1
    n_finais = len(strat.labels_)
    assert n_finais <= n_brutos


def test_monotonicidade_apos_fusao_no_oot(df_reg, cfg):
    """Com target alinhado ao score, a média por grupo no OOT é monotônica."""
    df = df_reg.copy()
    rng = np.random.default_rng(0)
    df[cfg.score_col] = df["feat_00"]
    # target = função monotônica do score => relação determinística
    df[cfg.target_col] = df["feat_00"] + rng.normal(0, 1e-3, len(df))
    strat = QuantileMonotonicRating(step=0.1)
    df[strat.column] = strat.fit_transform(df, cfg, "regression")
    oot = df[df[cfg.sample_col] == cfg.oot_sample]
    medias = oot.groupby(strat.column)[cfg.target_col].mean()
    medias = medias.reindex(sorted(medias.index))  # rótulos A,B,C crescentes
    assert np.all(np.diff(medias.values) >= -1e-6)


# ---------------------------------------------------------------------------
# Serialização (to_dict/from_dict, apply_ratings, ratings.json)
# ---------------------------------------------------------------------------
def _todas_estrategias():
    return [
        DecileRating(),
        QuantileMonotonicRating(step=0.1),
        TreeRating(max_leaf_nodes=6),
        OptBinningRating(max_n_bins=6),
        ManualScoreRating(cuts=[-0.5, 0.0, 0.5]),
        PercentileRating(percentiles=[25, 50, 75]),
    ]


def test_roundtrip_to_dict_from_dict_todas_estrategias(df_clf, cfg):
    """to_dict -> JSON estrito -> from_dict reproduz o transform original."""
    df = df_clf.copy()
    df[cfg.score_col] = df["feat_00"]
    for strat in _todas_estrategias():
        strat.fit(df, cfg, "classification")
        original = strat.transform(df, cfg)
        # allow_nan=False garante JSON estrito (sem literal Infinity nos edges)
        payload = json.loads(json.dumps(strat.to_dict(), allow_nan=False))
        novo = rating_from_dict(payload)
        pd.testing.assert_series_equal(novo.transform(df, cfg), original)
        assert novo.labels_ == strat.labels_
        assert novo.raw_to_label_ == strat.raw_to_label_
        # fronteiras preservadas bit a bit
        for attr in ("edges_", "splits_", "thresholds_"):
            if hasattr(strat, attr):
                assert np.array_equal(
                    np.asarray(getattr(novo, attr)), np.asarray(getattr(strat, attr))
                ), f"{type(strat).__name__}.{attr} alterado no roundtrip"


def test_roundtrip_scores_exatamente_na_borda(df_reg, cfg):
    """Empates no corte seguem a mesma convenção de borda após o roundtrip."""
    df = df_reg.copy()
    df[cfg.score_col] = df["feat_00"]
    for strat in [DecileRating(), TreeRating(max_leaf_nodes=6)]:
        strat.fit(df, cfg, "regression")
        cortes = getattr(strat, "thresholds_", None)
        if cortes is None:
            cortes = strat.edges_[np.isfinite(strat.edges_)]
        bordas = np.concatenate([cortes, np.nextafter(cortes, np.inf)])
        df_borda = pd.DataFrame({cfg.score_col: bordas})
        original = strat.transform(df_borda, cfg)
        rec = apply_ratings(bordas, strat.to_dict(), cfg)[strat.column]
        pd.testing.assert_series_equal(rec, original)


def test_tree_intervalos_equivalem_a_apply(df_clf, cfg):
    """O particionamento por limiares reproduz tree.apply + leaf_to_rank_."""
    df = df_clf.copy()
    df[cfg.score_col] = df["feat_00"]
    strat = TreeRating()
    strat.fit(df, cfg, "classification")
    scores = np.concatenate([df[cfg.score_col].to_numpy(dtype=float), strat.thresholds_])
    via_intervalos = strat._raw_groups(scores)
    leaves = strat.tree_.apply(scores.reshape(-1, 1))
    via_arvore = np.array([strat.leaf_to_rank_[int(l)] for l in leaves])
    assert np.array_equal(via_intervalos, via_arvore)


def test_apply_ratings_reaplica_sem_refit(df_clf, cfg):
    df = df_clf.copy()
    df[cfg.score_col] = df["feat_00"]
    estrategias = [DecileRating(), TreeRating(max_leaf_nodes=6)]
    esperado = {}
    for s in estrategias:
        s.fit(df, cfg, "classification")
        esperado[s.column] = s.transform(df, cfg)
    payload = json.loads(json.dumps(ratings_to_dict(estrategias), allow_nan=False))

    out = apply_ratings(df, payload, cfg)                 # via DataFrame
    assert list(out.columns) == [s.column for s in estrategias]
    for s in estrategias:
        pd.testing.assert_series_equal(out[s.column], esperado[s.column])

    out2 = apply_ratings(df[cfg.score_col].to_numpy(), payload, cfg)  # via vetor
    for s in estrategias:
        assert list(out2[s.column]) == list(esperado[s.column])


def test_to_dict_exige_fit_e_classe_desconhecida_falha():
    with pytest.raises(RuntimeError):
        DecileRating().to_dict()
    with pytest.raises(ValueError):
        rating_from_dict({"classe": "NaoExiste"})


def test_ratings_json_logado_no_run(df_clf, cfg, tmp_path, monkeypatch):
    """O artefato ratings.json aparece no run e reconstrói os mesmos grupos."""
    import mlflow
    from mlflow.tracking import MlflowClient
    from sklearn.linear_model import LogisticRegression

    from yggdrasil import MLPipeline

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = f"file:{(tmp_path / 'mlruns').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    feats = cfg.feature_columns(df_clf)
    dev = df_clf[df_clf["amostra"] == "DES"]
    model = LogisticRegression(max_iter=500).fit(dev[feats], dev["target"])

    pipe = MLPipeline(cfg, problem_type="classification", ratings=["decis"])
    res = pipe.run(df_clf, model=model, experiment="ratings_json", run_name="t",
                   log_shap=False)

    client = MlflowClient(tracking_uri=tracking_uri)
    artefatos = {a.path for a in client.list_artifacts(res.run_id)}
    assert "ratings.json" in artefatos

    local = mlflow.artifacts.download_artifacts(
        run_id=res.run_id, artifact_path="ratings.json", tracking_uri=tracking_uri
    )
    with open(local, encoding="utf-8") as fh:
        payload = json.load(fh)
    out = apply_ratings(res.df_scored, payload, cfg)
    pd.testing.assert_series_equal(out["rating_decis"], res.df_scored["rating_decis"])
