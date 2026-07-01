"""Testes de PSI (numérico, categórico e ao longo do tempo)."""

import numpy as np
import pandas as pd

from yggdrasil.config import ColumnConfig
from yggdrasil.monitoring import (
    classify_psi,
    psi,
    psi_categorical,
    psi_rating_by_pairs,
    psi_rating_over_time,
)
from yggdrasil.reporting.style import fmt_month_year


def test_psi_zero_para_mesma_distribuicao():
    rng = np.random.default_rng(0)
    base = rng.normal(size=5000)
    igual = rng.normal(size=5000)
    assert psi(base, igual) < 0.05


def test_psi_alto_para_shift_grande():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, size=5000)
    deslocado = rng.normal(3, 1, size=5000)  # média muito diferente
    assert psi(base, deslocado) > 0.25


def test_psi_categorical_identico_e_zero():
    labels = ["A", "B", "C"] * 100
    assert psi_categorical(labels, labels) < 1e-6


def test_classify_psi_faixas():
    assert classify_psi(0.05) == "estavel"
    assert classify_psi(0.15) == "atencao"
    assert classify_psi(0.40) == "instavel"


def test_psi_rating_over_time_estrutura(scored_clf, cfg):
    df = scored_clf.df_scored
    col = scored_clf.rating_cols[0]
    ts = psi_rating_over_time(df, col, cfg)
    assert {"mes", "psi", "n", "flag"}.issubset(ts.columns)
    assert len(ts) >= 1


def test_psi_rating_by_pairs_des_oot_estab():
    rng = np.random.default_rng(1)
    n = 1200
    df = pd.DataFrame({
        "amostra": (["DES"] * 600 + ["OOT"] * 400 + ["ESTABILIDADE"] * 200),
        "rating": rng.choice(list("ABCDE"), size=n),
    })
    cfg = ColumnConfig(sample_col="amostra",
                       analysis_samples=("DES", "OOT", "ESTABILIDADE"))
    out = psi_rating_by_pairs(df, "rating", cfg)
    # uma linha por comparação (OOT e ESTABILIDADE), na ordem do analysis_samples
    assert list(out["comparacao"]) == ["OOT", "ESTABILIDADE"]
    assert (out["baseline"] == "DES").all()
    assert {"psi", "n_baseline", "n_comparacao", "flag"}.issubset(out.columns)


def test_fmt_month_year_formatos():
    # string 'AAAA-MM', Timestamp e Period → 'mmm/aa'
    assert fmt_month_year(["2022-01", "2023-12"]) == ["jan/22", "dez/23"]
    assert fmt_month_year([pd.Timestamp("2024-06-15")]) == ["jun/24"]
    assert fmt_month_year([pd.Period("2025-03", freq="M")]) == ["mar/25"]
    # valor fora do padrão volta como string, sem quebrar
    assert fmt_month_year(["total"]) == ["total"]
