"""Testes das métricas de discriminação/erro ao longo do tempo (monitoring)."""

import numpy as np
import pandas as pd
import pytest

from yggdrasil.config import ColumnConfig
from yggdrasil.monitoring import metric_over_time, plot_metric_over_time


def _df_clf(n_mes=400, meses=8, seed=0, degradar_ultimo=False):
    """Base multi-safra de classificação: score informativo do alvo; DES nos
    primeiros meses e OOT nos 3 últimos. ``degradar_ultimo`` embaralha o score
    do último mês (discriminação ~ zero)."""
    rng = np.random.default_rng(seed)
    datas = pd.date_range("2023-01-01", periods=meses, freq="MS")
    frames = []
    for i, data in enumerate(datas):
        sc = rng.uniform(size=n_mes)
        y = (sc + rng.normal(0, 0.35, size=n_mes) > 0.6).astype(int)
        if degradar_ultimo and i == meses - 1:
            sc = rng.permutation(sc)
        frames.append(pd.DataFrame({
            "dt_ref": data, "target": y, "prediction": sc,
            "amostra": "DES" if i < meses - 3 else "OOT",
        }))
    return pd.concat(frames, ignore_index=True)


def _df_reg(n_mes=300, meses=6, seed=1, degradar_ultimo=False):
    """Base multi-safra de regressão com alvo em [0, 1]."""
    rng = np.random.default_rng(seed)
    datas = pd.date_range("2024-01-01", periods=meses, freq="MS")
    frames = []
    for i, data in enumerate(datas):
        base = rng.uniform(size=n_mes)
        y = np.clip(base + rng.normal(0, 0.05, size=n_mes), 0, 1)
        sc = rng.permutation(base) if (degradar_ultimo and i == meses - 1) else base
        frames.append(pd.DataFrame({
            "dt_ref": data, "target": y, "prediction": sc,
            "amostra": "DES" if i < meses - 2 else "OOT",
        }))
    return pd.concat(frames, ignore_index=True)


def test_estrutura_e_ordenacao_clf():
    out = metric_over_time(_df_clf(), ColumnConfig(), "classification")
    assert {"periodo", "n", "taxa_evento", "auc", "gini", "ks",
            "flag", "nota"}.issubset(out.columns)
    assert len(out) == 8
    assert list(out["periodo"]) == sorted(out["periodo"])
    assert out["auc"].notna().all() and out["ks"].notna().all()
    assert (out["nota"] == "").all()
    assert set(out["flag"]) <= {"ok", "atencao", "degradado", "n/a"}
    # métrica e referência da flag ficam nos attrs (para o plot)
    assert out.attrs["flag_metric"] == "ks"
    assert np.isfinite(out.attrs["flag_ref"])


def test_grupo_pequeno_nan_com_nota():
    df = _df_clf()
    per = pd.to_datetime(df["dt_ref"]).dt.to_period("M")
    fev = per == pd.Period("2023-02", "M")
    df = pd.concat([df[~fev], df[fev].head(50)], ignore_index=True)
    out = metric_over_time(df, ColumnConfig(), "classification", min_n=200)
    linha = out[out["periodo"] == "2023-02"].iloc[0]
    assert linha["n"] == 50
    assert np.isnan(linha["auc"]) and np.isnan(linha["ks"])
    assert "n <" in linha["nota"]
    assert linha["flag"] == "n/a"
    assert np.isfinite(linha["taxa_evento"])  # descritivo continua saindo


def test_classe_unica_nan_com_nota():
    df = _df_clf()
    per = pd.to_datetime(df["dt_ref"]).dt.to_period("M")
    df.loc[per == pd.Period("2023-03", "M"), "target"] = 0
    out = metric_over_time(df, ColumnConfig(), "classification")
    linha = out[out["periodo"] == "2023-03"].iloc[0]
    assert np.isnan(linha["auc"])
    assert linha["nota"] == "classe única"
    # demais meses seguem computáveis
    assert out[out["periodo"] != "2023-03"]["auc"].notna().all()


def test_flag_degradacao_no_periodo_degradado():
    out = metric_over_time(_df_clf(n_mes=600, degradar_ultimo=True),
                           ColumnConfig(), "classification")
    assert out.iloc[-1]["flag"] == "degradado"          # score embaralhado
    assert (out.iloc[:-1]["flag"] != "degradado").all()  # meses saudáveis


def test_regressao_colunas_e_flag_rmse():
    out = metric_over_time(_df_reg(degradar_ultimo=True), ColumnConfig(),
                           "regression", min_n=100)
    assert {"periodo", "n", "realizado_medio", "previsto_medio",
            "rmse", "mae", "r2", "flag", "nota"}.issubset(out.columns)
    assert out.attrs["flag_metric"] == "rmse"
    assert out.iloc[-1]["flag"] == "degradado"           # rmse saltou vs média DES
    assert (out.iloc[:-1]["flag"] == "ok").all()


def test_freq_trimestral_agrupa():
    df = _df_clf(meses=6)
    out = metric_over_time(df, ColumnConfig(), "classification", freq="Q")
    assert len(out) == 2
    assert out["n"].sum() == len(df)


def test_plot_metric_over_time():
    import matplotlib.pyplot as plt

    out = metric_over_time(_df_clf(meses=6), ColumnConfig(), "classification")
    fig = plot_metric_over_time(out, "ks")
    assert fig is not None
    plt.close(fig)
    with pytest.raises(ValueError):
        plot_metric_over_time(out, "nao_existe")


def test_problem_type_invalido():
    with pytest.raises(ValueError):
        metric_over_time(_df_clf(meses=2), ColumnConfig(), "cluster")
