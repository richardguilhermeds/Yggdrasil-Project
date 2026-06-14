"""Testes das metodologias de grupos homogêneos (ratings)."""

import numpy as np

from yggdrasil.ratings import (
    DecileRating,
    QuantileMonotonicRating,
    TreeRating,
    build_ratings,
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
