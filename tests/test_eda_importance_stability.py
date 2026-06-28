"""Testes de importância de features e estabilidade (PSI)."""

import numpy as np

from yggdrasil import ColumnConfig
from yggdrasil.eda import EDAConfig
from yggdrasil.eda import importance, stability
from yggdrasil.monitoring import psi


def test_univariate_ks_gini_strong(df_eda):
    cfg = ColumnConfig()
    m = importance.univariate_ks_gini(df_eda, "feat_bom", cfg, "classification")
    assert m["gini_univ"] > 0.5
    fraca = importance.univariate_ks_gini(df_eda, "feat_fraca", cfg, "classification")
    assert fraca["gini_univ"] < 0.2


def test_importance_ranking_flags_leakage(df_eda):
    cfg, ec = ColumnConfig(), EDAConfig()
    rk = importance.importance_ranking(df_eda, cfg, ec, "classification")
    assert {"feature", "iv", "score", "leakage_flag"}.issubset(rk.columns)
    suspeitos = importance.leakage_suspects(rk)
    assert "feat_leakage" in suspeitos


def test_feature_psi_stable_vs_unstable(df_eda):
    cfg, ec = ColumnConfig(), EDAConfig()
    estavel = stability.feature_psi(df_eda, "feat_bom", cfg, ec)
    instavel = stability.feature_psi(df_eda, "feat_instavel", cfg, ec)
    assert estavel < 0.10
    assert instavel > 0.25


def test_feature_psi_sem_missing_igual_ao_monitoring(df_eda):
    """Propriedade: sem faltantes, o PSI numérico da feature coincide com
    monitoring.psi (o bin de faltantes não contribui)."""
    cfg, ec = ColumnConfig(), EDAConfig()
    dev = df_eda[df_eda["amostra"] == "DES"]["feat_bom"]   # feat_bom não tem NaN
    oot = df_eda[df_eda["amostra"] == "OOT"]["feat_bom"]
    esperado = psi(dev, oot, bins=ec.n_bins)
    obtido = stability.feature_psi(df_eda, "feat_bom", cfg, ec)
    assert np.isclose(esperado, obtido, atol=1e-9)


def test_feature_psi_detecta_migracao_de_missing(df_eda):
    """Regressão (A1.4): o PSI numérico NÃO pode ignorar faltantes via dropna —
    uma feature que passa a faltar no OOT deve acusar instabilidade."""
    cfg, ec = ColumnConfig(), EDAConfig()
    d = df_eda.copy()
    d["feat_bom"] = d["feat_bom"].astype(float)            # DES permanece sem NaN
    oot_idx = d.index[d["amostra"] == "OOT"]
    rng = np.random.default_rng(0)
    falta = rng.choice(oot_idx, size=int(0.5 * len(oot_idx)), replace=False)
    d.loc[falta, "feat_bom"] = np.nan                      # 50% de missing só no OOT
    val = stability.feature_psi(d, "feat_bom", cfg, ec)
    assert val > 0.25                                      # migração de missing = instável


def test_stability_summary_shape(df_eda):
    cfg, ec = ColumnConfig(), EDAConfig()
    ss = stability.stability_summary(df_eda, cfg, ec)
    assert set(ss["feature"]) == set(cfg.feature_columns(df_eda))
    assert {"psi_oot", "flag", "psi_max_safra"}.issubset(ss.columns)


def test_correlation_robusta_a_constante(df_eda):
    """VIF e clusters não devem quebrar com feature constante (corr indefinida)."""
    from yggdrasil.eda import correlation
    cfg, ec = ColumnConfig(), EDAConfig()
    vif = correlation.vif_table(df_eda, cfg, ec)
    clusters = correlation.redundancy_clusters(df_eda, cfg, ec)
    assert "feat_const" not in set(vif["feature"])          # constante excluída
    assert len(clusters) >= 1
