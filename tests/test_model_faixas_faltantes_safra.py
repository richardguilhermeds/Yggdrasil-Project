"""Gráficos por safra de variável criada a partir de faixas quando a origem tem
faltantes: a categoria literal "(faltante)" não pode duplicar a coluna do share
e a distribuição acumulada deve empilhar as faixas (não "apenas numéricas")."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.model import ModelSegmenter


@pytest.fixture
def seg():
    rng = np.random.default_rng(0)
    n = 4000
    renda = rng.gamma(2, 1500, n)
    y = (rng.random(n) < 1 / (1 + np.exp(1.5 + 0.0004 * (renda - 3000)))).astype(float)
    renda[rng.random(n) < 0.12] = np.nan
    meses = pd.date_range("2023-01-01", periods=6, freq="MS")
    df = pd.DataFrame({"renda": renda, "target": y, "dt_ref": rng.choice(meses, n)})
    df["amostra"] = np.where(df["dt_ref"] >= meses[4], "OOT", "DES")
    s = ModelSegmenter(df, target="target", sample_col="amostra", date_col="dt_ref")
    s.set_manual_bins("renda", [1500, 3000])
    s.create_categorical("renda", new_name="renda_bin")
    return s


def test_share_por_safra_sem_coluna_duplicada(seg):
    sh = seg.variable_share_by_safra("renda_bin", "dt_ref", all_samples=True)
    assert not sh.columns.duplicated().any()
    assert list(sh.columns).count("(faltante)") == 1
    assert np.allclose(sh.drop(columns="safra").sum(axis=1), 100, atol=0.5)
    fig = seg.plot_variable_timeseries("renda_bin", "dt_ref", all_samples=True)
    plt.close(fig)


def test_acumulada_categorica_usa_faixas_na_ordem_da_origem(seg):
    fig = seg.plot_variable_optbin_cumshare_timeseries("renda_bin", "dt_ref",
                                                       all_samples=True)
    ax = fig.axes[0]
    textos = [t.get_text() for t in ax.texts]
    assert "apenas para variáveis numéricas" not in textos
    leg = [t.get_text() for t in ax.get_legend().get_texts()]
    assert leg == ["(-inf, 1500]", "(1500, 3000]", "(3000, inf]", "(faltante)"]
    plt.close(fig)
