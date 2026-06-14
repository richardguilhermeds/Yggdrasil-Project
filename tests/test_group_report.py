"""Testes do relatório por grupo homogêneo e dos shifts."""

import numpy as np

from yggdrasil.metrics import metric_by_sample, metric_shifts
from yggdrasil.reporting import group_report, is_monotonic


def test_group_report_estrutura(scored_clf, cfg):
    df = scored_clf.df_scored
    col = scored_clf.rating_cols[0]
    # restringe às amostras de análise (como a esteira faz)
    df_an = df[df[cfg.sample_col].isin(cfg.analysis_samples)]
    rep = group_report(df_an, col, cfg, "classification")

    for c in ["rating", "volume", "pct_volume", "score_medio", "target_medio"]:
        assert c in rep.columns
    # colunas por amostra de análise
    assert "target_medio_DES" in rep.columns
    assert "target_medio_OOT" in rep.columns
    # representatividade global soma ~100%
    assert np.isclose(rep["pct_volume"].sum(), 100.0, atol=1e-6)
    # uma linha por grupo
    assert len(rep) == df_an[col].nunique()


def test_group_report_ignora_scoring_only(scored_clf, cfg):
    """As amostras scoring-only (SIMUL) não devem aparecer como coluna."""
    df = scored_clf.df_scored
    col = scored_clf.rating_cols[0]
    df_an = df[df[cfg.sample_col].isin(cfg.analysis_samples)]
    rep = group_report(df_an, col, cfg, "classification")
    assert "target_medio_SIMUL" not in rep.columns


def test_is_monotonic():
    import pandas as pd
    cresc = pd.DataFrame({"target_medio": [0.1, 0.2, 0.3]})
    quebra = pd.DataFrame({"target_medio": [0.1, 0.05, 0.3]})
    assert is_monotonic(cresc)
    assert not is_monotonic(quebra)


def test_metric_shifts_chaves():
    ref = {"ks": 0.40, "auc": 0.80}
    cmp = {"ks": 0.30, "auc": 0.75}
    sh = metric_shifts(ref, cmp)
    assert sh["ks_shift_abs"] == -0.10
    assert "ks_shift_rel" in sh and "auc_shift_abs" in sh


def test_metric_by_sample_so_amostras_de_analise(scored_clf, cfg):
    df = scored_clf.df_scored
    df_an = df[df[cfg.sample_col].isin(cfg.analysis_samples)]
    met = metric_by_sample(df_an, cfg, "classification")
    assert set(met.keys()) == {"DES", "OOT"}
    assert "SIMUL" not in met
