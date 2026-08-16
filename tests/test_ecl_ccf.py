"""
Testes de CCF/EAD (``yggdrasil.credit_risk.ecl.ccf``).

As propriedades que ancoram a suíte:

* as **quatro medidas ex-post** (CCF/LEQ, EADF, AUF, EAD direto) reconstroem a
  exposição realizada **por identidade** — cada uma é só uma reparametrização da
  mesma observação;
* os **desenhos de base** divergem no sentido esperado: quanto mais cedo a data
  de referência, mais tempo o cliente teve para sacar, e maior o CCF. É o
  resultado que justifica documentar o desenho escolhido;
* o **backtest de EAD** acusa viés injetado, em moeda — o teste que decide se o
  parâmetro serve;
* a higiene do dado (limite nulo, não sacado nulo, *over limit*) é **contada**,
  não silenciosa.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.capital.parameters import ccf_downturn as ccf_downturn_capital
from yggdrasil.credit_risk.ecl import (
    MEASURES,
    backtest_ead,
    ccf_downturn,
    ccf_psi,
    compare_measures,
    ead_from_ccf,
    ead_from_measure,
    pooled_ccf,
    reference_dataset,
)


# ----------------------------------------------------------------------
# Base de referência
# ----------------------------------------------------------------------
def test_desenhos_de_base_divergem_no_sentido_esperado(df_credito):
    bases = {m: reference_dataset(df_credito, method=m, segment_col="produto")
             for m in ("cohort", "fixed_horizon", "variable")}
    dist = {m: b.frame["meses_ate_default"].mean() for m, b in bases.items()}
    assert dist["fixed_horizon"] == pytest.approx(12.0)            # sempre 12 meses antes
    assert dist["cohort"] < dist["fixed_horizon"]                  # a coorte olha mais perto
    # mais tempo até o default = mais limite convertido
    assert bases["fixed_horizon"].values.mean() > bases["cohort"].values.mean()
    # o horizonte variável multiplica a amostra (uma obs. por mês da janela)
    assert len(bases["variable"]) > len(bases["fixed_horizon"])
    assert bases["variable"].frame["id"].nunique() <= len(bases["variable"])


def test_as_quatro_medidas_reconstroem_o_ead(df_credito):
    b = reference_dataset(df_credito, method="fixed_horizon", clip=False)
    for medida in MEASURES:
        rec = ead_from_measure(b.frame["sacado_ref"], b.frame["limite_ref"],
                               b.frame[medida], measure=medida)
        assert np.allclose(rec, b.frame["ead"], rtol=1e-9, atol=1e-6), medida
    assert np.allclose(ead_from_ccf(b.frame["sacado_ref"], b.frame["limite_ref"],
                                    b.frame["ccf"]), b.frame["ead"], atol=1e-6)
    with pytest.raises(ValueError):
        ead_from_measure(1.0, 2.0, 0.5, measure="qualquer")


def test_piso_no_sacado_e_opcional():
    """O piso é política de uso: ligado na projeção, desligado na validação."""
    # CCF negativo = o contrato amortizou antes de quebrar; a fórmula respeita.
    assert ead_from_measure(100.0, 200.0, -0.5, measure="ccf") == pytest.approx(50.0)
    assert ead_from_measure(100.0, 200.0, -0.5, measure="ccf",
                            floor_at_drawn=True) == pytest.approx(100.0)
    assert ead_from_ccf(100.0, 200.0, -0.5, floor_at_drawn=True) == pytest.approx(100.0)
    # exposição negativa nunca sai, com ou sem piso
    assert ead_from_measure(100.0, 200.0, -5.0, measure="ccf") == pytest.approx(0.0)


def test_higiene_do_dado_e_contada(df_credito):
    ruim = df_credito.copy()
    idx = ruim.index[:200]
    ruim.loc[idx, "limite"] = 0.0                                  # sem limite
    idx2 = ruim.index[200:400]
    ruim.loc[idx2, "sacado"] = ruim.loc[idx2, "limite"] * 1.2      # over limit

    b = reference_dataset(ruim, method="variable", drop_over_limit=True)
    exc = b.excluded_frame().set_index("motivo")["n"]
    assert exc["limite_abaixo_do_piso"] > 0
    assert exc["sacado_acima_do_limite"] > 0
    assert (b.frame["limite_ref"] > 0).all()
    assert (b.frame["nao_sacado_ref"] > 0).all()
    assert not b.frame["over_limit_ref"].any()
    assert b.summary().loc[0, "n_excluidos"] == int(exc.sum())


def test_referencia_precisa_estar_adimplente(df_credito):
    """Uma data de referência já em *default* não informa conversão de limite."""
    b = reference_dataset(df_credito, method="variable")
    assert b.excluded["ja_em_default_na_referencia"] >= 0
    assert (b.frame["meses_ate_default"] > 0).all()


def test_clip_e_winsorizacao(df_credito):
    cru = reference_dataset(df_credito, method="variable", clip=False)
    recortado = reference_dataset(df_credito, method="variable", clip=True)
    assert recortado.values.between(0.0, 1.0).all()
    assert "ccf_bruto" in recortado.frame.columns
    if (cru.values < 0).any() or (cru.values > 1).any():
        assert recortado.frame["recortado"].any()

    wins = reference_dataset(df_credito, method="variable", winsorize=0.05, clip=False)
    assert wins.values.min() >= cru.values.quantile(0.05) - 1e-9
    assert wins.values.max() <= cru.values.quantile(0.95) + 1e-9
    with pytest.raises(ValueError):
        reference_dataset(df_credito, winsorize=0.7)


def test_distribuicao_expoe_as_massas_nos_extremos(df_credito):
    b = reference_dataset(df_credito, method="variable")
    dist = b.distribution(bins=10)
    assert dist["faixa"].iloc[0] == "= 0" and dist["faixa"].iloc[-1] == "= 1"
    assert dist["n"].sum() == len(b)
    resumo = b.summary().iloc[0]
    assert resumo["massa_em_0"] == pytest.approx(float((b.values <= 1e-9).mean()))
    assert resumo["massa_em_1"] == pytest.approx(float((b.values >= 1 - 1e-9).mean()))


# ----------------------------------------------------------------------
# Estimação agrupada
# ----------------------------------------------------------------------
def test_pooled_ccf_por_segmento(df_credito):
    b = reference_dataset(df_credito, method="variable", segment_col="produto")
    agrupado = pooled_ccf(b, by="produto")
    assert set(agrupado["grupo"]) == {"cartao", "consignado"}
    assert (agrupado["ic_inf"] <= agrupado["ccf"]).all()
    assert (agrupado["ccf"] <= agrupado["ic_sup"]).all()
    assert (agrupado["n"] > 0).all()

    global_ = pooled_ccf(b)
    assert len(global_) == 1 and global_["grupo"].iloc[0] == "__global__"
    assert global_["ccf"].iloc[0] == pytest.approx(float(b.values.mean()))
    assert pooled_ccf(b, stat="median")["ccf"].iloc[0] == pytest.approx(
        float(b.values.median()))
    ponderado = pooled_ccf(b, stat="weighted")["ccf"].iloc[0]
    assert 0.0 <= ponderado <= 1.0
    with pytest.raises(ValueError):
        pooled_ccf(b, stat="qualquer")


def test_downturn_reusa_a_formula_do_capital(df_credito):
    b = reference_dataset(df_credito, method="variable")
    assert ccf_downturn is ccf_downturn_capital                    # fonte única no repo
    assert b.downturn(0.9) == pytest.approx(
        ccf_downturn(b.values.to_numpy(dtype=float), quantile=0.9))
    assert b.downturn(0.9) >= b.downturn(0.5)                      # mais conservador


# ----------------------------------------------------------------------
# Backtest de EAD
# ----------------------------------------------------------------------
def test_backtest_ead_acusa_vies_injetado(df_credito):
    b = reference_dataset(df_credito, method="fixed_horizon", segment_col="produto")
    justo = float(pooled_ccf(b, stat="weighted")["ccf"].iloc[0])
    bt = backtest_ead(b, justo)
    assert abs(bt["vies_relativo"].iloc[0]) < 0.05                 # ponderado ≈ neutro

    alto, baixo = backtest_ead(b, 0.999), backtest_ead(b, 0.0)
    assert alto["vies"].iloc[0] > 0 > baixo["vies"].iloc[0]
    assert alto["rmse"].iloc[0] > bt["rmse"].iloc[0]
    assert alto["ead_realizado"].iloc[0] == pytest.approx(baixo["ead_realizado"].iloc[0])


def test_backtest_ead_por_segmento_e_por_coluna(df_credito):
    b = reference_dataset(df_credito, method="fixed_horizon", segment_col="produto")
    estimativas = dict(zip(*pooled_ccf(b, by="produto")[["grupo", "ccf"]].to_numpy().T))
    bt = backtest_ead(b, {k: float(v) for k, v in estimativas.items()}, by="produto")
    assert set(bt["grupo"]) == {"__global__", "cartao", "consignado"}

    # estimativa vinda de uma coluna: é assim que se testa um modelo já escorado.
    # Com o CCF BRUTO a reconstrução é exata; com o recortado em [0, 1] sobra o
    # viés que o próprio recorte introduz — o que o backtest tem de mostrar.
    perfeito = backtest_ead(b.frame, "ccf_bruto")
    assert perfeito["vies"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert perfeito["rmse"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    recortado = backtest_ead(b.frame, "ccf")
    assert recortado["vies"].iloc[0] >= perfeito["vies"].iloc[0]

    with pytest.raises(ValueError, match="exige `by`"):
        backtest_ead(b, {"cartao": 0.3})
    with pytest.raises(ValueError, match="sem estimativa"):
        backtest_ead(b, {"cartao": 0.3}, by="produto")


def test_compare_measures_ordena_pelo_erro(df_credito):
    b = reference_dataset(df_credito, method="fixed_horizon")
    comp = compare_measures(b)
    assert set(comp["medida"]) == set(MEASURES)
    assert comp["erro_absoluto_relativo"].is_monotonic_increasing  # melhor primeiro


# ----------------------------------------------------------------------
# Monitoramento
# ----------------------------------------------------------------------
def test_ccf_psi_entre_safras(df_credito):
    d = df_credito.assign(safra=df_credito["dt_ref"].dt.year)
    b = reference_dataset(d, method="variable", extra_cols=["safra"])
    psi = ccf_psi(b, by="safra")
    assert psi["referencia"].sum() == 1
    assert psi.loc[psi["referencia"], "psi"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert (psi["psi"] >= -1e-12).all()
    assert set(psi["classificacao"]) <= {"estável", "atenção", "instável", "—"}
    with pytest.raises(ValueError, match="ao menos 2 grupos"):
        ccf_psi(b.frame.assign(unico=1), by="unico")


# ----------------------------------------------------------------------
# Validações
# ----------------------------------------------------------------------
def test_reference_dataset_valida_uso(df_credito):
    with pytest.raises(ValueError):
        reference_dataset(df_credito, method="qualquer")
    with pytest.raises(ValueError):
        reference_dataset(df_credito, measure="qualquer")
    with pytest.raises(ValueError, match="ausentes"):
        reference_dataset(df_credito.drop(columns=["limite"]))
    sem_default = df_credito.assign(default=0)
    with pytest.raises(ValueError, match="nenhum default"):
        reference_dataset(sem_default)
    with pytest.raises(ValueError, match="ficou vazia"):
        reference_dataset(df_credito, min_limit=1e12)
