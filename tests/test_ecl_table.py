"""
Testes da montagem do ECL, dos gráficos e do registro no MLflow
(``yggdrasil.credit_risk.ecl.ecl``, ``.report``, ``.tracking``).

As propriedades que ancoram a suíte:

* a soma das perdas descontadas por período **é** o ECL *lifetime*, e os 12
  primeiros períodos **são** o ECL de 12 meses;
* o estágio só corta o horizonte: estágio 1 = ECL 12 meses, estágio 2 =
  *lifetime*, estágio 3 = ELBE sobre o saldo;
* desconto e amortização **reduzem** o ECL, cada um pelo seu motivo;
* os cenários ordenam pelo ciclo (adverso > base > otimista) e o cenário ``z = 0``
  reproduz o ECL não condicionado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.ecl import (
    ECLResult,
    LifetimePD,
    ead_schedule,
    ecl_scenarios,
    ecl_table,
    elbe_table,
    kaplan_meier,
    reference_dataset,
    pooled_ccf,
)


@pytest.fixture
def modelo(painel) -> LifetimePD:
    return LifetimePD(method="vintage", horizon=48).fit(painel, by="produto")


@pytest.fixture
def resultado(modelo, carteira_viva) -> ECLResult:
    return ecl_table(carteira_viva, model=modelo, lgd="lgd", ead="saldo",
                     stage_col="estagio", elbe="elbe", discount_rate="taxa_efetiva",
                     age_col="idade", term_col="prazo", detail=True)


# ----------------------------------------------------------------------
# ead_schedule
# ----------------------------------------------------------------------
def test_ead_schedule_modos():
    constante = ead_schedule(1000.0, term=24, horizon=36, method="constant")
    assert constante.shape == (1, 36)
    assert np.allclose(constante[0, :24], 1000.0)
    assert np.allclose(constante[0, 24:], 0.0)                     # depois do prazo, zero

    linear = ead_schedule(1000.0, term=24, horizon=24, method="linear")
    assert linear[0, 0] == pytest.approx(1000.0 * 23 / 24)
    assert linear[0, -1] == pytest.approx(0.0)
    assert np.all(np.diff(linear[0]) <= 0)                         # amortiza sempre

    price = ead_schedule(1000.0, term=24, horizon=24, method="annuity", rate=0.30)
    assert price[0, -1] == pytest.approx(0.0, abs=1e-9)
    assert (price[0, :-1] > linear[0, :-1]).all()                  # Price amortiza mais devagar
    assert ead_schedule(1000.0, horizon=6)[0].tolist() == [1000.0] * 6   # sem prazo

    with pytest.raises(ValueError, match="annuity"):
        ead_schedule(1000.0, term=12, horizon=12, method="annuity")
    with pytest.raises(ValueError):
        ead_schedule(1000.0, horizon=0)


# ----------------------------------------------------------------------
# ecl_table
# ----------------------------------------------------------------------
def test_soma_das_perdas_por_periodo_e_o_ecl(resultado):
    f = resultado.frame
    todas = [f"ecl_h{t}" for t in range(1, resultado.horizon + 1)]
    assert np.allclose(f["ecl_lifetime"], f[todas].sum(axis=1))
    assert np.allclose(f["ecl_12m"], f[todas[:12]].sum(axis=1))
    assert (f["ecl_lifetime"] >= f["ecl_12m"] - 1e-12).all()
    assert resultado.total == pytest.approx(float(f["ecl"].sum()))
    assert resultado.coverage == pytest.approx(resultado.total / f["exposicao"].sum())


def test_estagios_cortam_o_horizonte(resultado):
    f = resultado.frame
    e1, e2, e3 = (f[f["estagio"] == s] for s in (1, 2, 3))
    assert len(e1) and len(e2) and len(e3)
    assert np.allclose(e1["ecl"], e1["ecl_12m"])
    assert np.allclose(e2["ecl"], e2["ecl_lifetime"])
    assert np.allclose(e3["ecl"], e3["elbe"] * e3["exposicao"])     # ELBE sobre o saldo
    resumo = resultado.summary()
    assert set(resumo["estagio"]) == {1, 2, 3}
    assert resumo["ecl"].sum() == pytest.approx(resultado.total)


def test_regras_de_estagio_configuraveis(modelo, carteira_viva):
    tudo_lifetime = ecl_table(carteira_viva, model=modelo, lgd="lgd", ead="saldo",
                              stage_col="estagio", stage_rules={1: "lifetime", 2: "lifetime",
                                                                3: "lifetime"},
                              age_col="idade", term_col="prazo")
    assert np.allclose(tudo_lifetime.frame["ecl"], tudo_lifetime.frame["ecl_lifetime"])
    with pytest.raises(ValueError, match="sem regra"):
        ecl_table(carteira_viva, model=modelo, lgd="lgd", ead="saldo",
                  stage_col="estagio", stage_rules={1: "12m"}, age_col="idade")
    with pytest.raises(ValueError, match="stage_rules"):
        ecl_table(carteira_viva, model=modelo, lgd="lgd", ead="saldo",
                  stage_col="estagio", stage_rules={1: "qualquer", 2: "12m", 3: "elbe"},
                  age_col="idade")


def test_sem_estagio_a_carteira_toda_e_lifetime(modelo, carteira_viva):
    r = ecl_table(carteira_viva, model=modelo, lgd="lgd", ead="saldo",
                  age_col="idade", term_col="prazo")
    assert (r.frame["estagio"] == 2).all()
    assert np.allclose(r.frame["ecl"], r.frame["ecl_lifetime"])


def test_desconto_e_amortizacao_reduzem_o_ecl(modelo, carteira_viva):
    comum = dict(model=modelo, lgd="lgd", ead="saldo", age_col="idade", term_col="prazo")
    sem = ecl_table(carteira_viva, **comum)
    com_desconto = ecl_table(carteira_viva, discount_rate=0.5, **comum)
    assert com_desconto.total < sem.total
    assert ecl_table(carteira_viva, discount_rate=0.0, **comum).total == pytest.approx(sem.total)

    cronograma = ead_schedule(carteira_viva["saldo"].to_numpy(),
                              carteira_viva["prazo"].to_numpy(),
                              horizon=modelo.horizon, method="linear")
    amortizado = ecl_table(carteira_viva, model=modelo, lgd="lgd", ead=cronograma,
                           exposure_col="saldo", age_col="idade", term_col="prazo")
    assert amortizado.total < sem.total


def test_lgd_e_ead_aceitam_escalar_coluna_e_matriz(modelo, carteira_viva):
    n, H = len(carteira_viva), modelo.horizon
    escalar = ecl_table(carteira_viva, model=modelo, lgd=0.5, ead=1000.0,
                        age_col="idade", term_col="prazo")
    assert escalar.frame["exposicao"].nunique() == 1
    matriz = ecl_table(carteira_viva, model=modelo, lgd=np.full((n, H), 0.5),
                       ead=np.full((n, H), 1000.0), age_col="idade", term_col="prazo")
    assert matriz.total == pytest.approx(escalar.total)
    # LGD crescente no horizonte (vetor por horizonte)
    por_horizonte = ecl_table(carteira_viva, model=modelo, lgd=np.linspace(0.2, 0.8, H),
                              ead=1000.0, age_col="idade", term_col="prazo")
    assert por_horizonte.total != pytest.approx(escalar.total)

    with pytest.raises(ValueError, match="lgd"):
        ecl_table(carteira_viva, model=modelo, lgd=1.5, ead="saldo", age_col="idade")
    with pytest.raises(ValueError, match="não encontrada"):
        ecl_table(carteira_viva, model=modelo, lgd="inexistente", ead="saldo",
                  age_col="idade")
    with pytest.raises(ValueError):
        ecl_table(carteira_viva, model=modelo, lgd=0.5, ead=np.ones(3), age_col="idade")


def test_pd_marginal_pronta_e_colunas_escoradas(modelo, carteira_viva):
    referencia = ecl_table(carteira_viva, model=modelo, lgd="lgd", ead="saldo",
                           age_col="idade", term_col="prazo")
    marg = modelo.marginal_matrix(carteira_viva, age_col="idade", term_col="prazo")
    por_matriz = ecl_table(carteira_viva, pd_marginal=marg, lgd="lgd", ead="saldo")
    assert por_matriz.total == pytest.approx(referencia.total)

    escorada = modelo.apply(carteira_viva, age_col="idade", term_col="prazo")
    por_colunas = ecl_table(escorada, lgd="lgd", ead="saldo")
    assert por_colunas.total == pytest.approx(referencia.total)
    with pytest.raises(ValueError, match="nenhuma coluna"):
        ecl_table(carteira_viva, lgd="lgd", ead="saldo")


def test_agregacoes(resultado):
    por_produto = resultado.by("produto")
    assert set(por_produto["produto"]) == {"cartao", "consignado"}
    assert por_produto["ecl"].sum() == pytest.approx(resultado.total)
    assert np.allclose(por_produto["taxa_provisao"],
                       por_produto["ecl"] / por_produto["exposicao"])
    assert len(resultado.by("produto", "estagio")) <= 6
    assert resultado.by()["ecl"].iloc[0] == pytest.approx(resultado.total)
    with pytest.raises(ValueError, match="ausentes"):
        resultado.by("inexistente")
    d = resultado.to_dict()
    assert d["ecl_total"] == pytest.approx(resultado.total)


# ----------------------------------------------------------------------
# Cenários
# ----------------------------------------------------------------------
def test_cenarios_ordenam_pelo_ciclo(modelo, carteira_viva):
    comum = dict(lgd="lgd", ead="saldo", stage_col="estagio", elbe="elbe",
                 age_col="idade", term_col="prazo")
    base = ecl_table(carteira_viva, model=modelo, **comum)
    cen = ecl_scenarios(carteira_viva, modelo,
                        {"otimista": {"z": 1.0, "peso": 0.2},
                         "base": {"z": 0.0, "peso": 0.5},
                         "adverso": {"z": -1.5, "peso": 0.3, "decay": 0.1}},
                        rho=0.10, **comum)
    por_cenario = cen["por_cenario"].set_index("cenario")["ecl"]
    assert por_cenario["adverso"] > por_cenario["base"] > por_cenario["otimista"]
    assert por_cenario["base"] == pytest.approx(base.total)         # z = 0 é idempotente
    assert cen["ponderado"] == pytest.approx(
        float((cen["por_cenario"]["ecl"] * cen["por_cenario"]["peso"]).sum()))
    assert set(cen["resultados"]) == {"otimista", "base", "adverso"}


def test_cenarios_exigem_pesos_que_somam_um(modelo, carteira_viva):
    comum = dict(lgd="lgd", ead="saldo", age_col="idade", term_col="prazo")
    with pytest.raises(ValueError, match="sem peso"):
        ecl_scenarios(carteira_viva, modelo, {"base": {"z": 0.0}}, rho=0.1, **comum)
    with pytest.raises(ValueError, match="somar 1"):
        ecl_scenarios(carteira_viva, modelo,
                      {"base": {"z": 0.0, "peso": 0.5}, "adverso": {"z": -1.0, "peso": 0.2}},
                      rho=0.1, **comum)
    with pytest.raises(ValueError, match="ao menos um cenário"):
        ecl_scenarios(carteira_viva, modelo, {}, rho=0.1, **comum)


# ----------------------------------------------------------------------
# Ponta a ponta: os três parâmetros na mesma conta
# ----------------------------------------------------------------------
def test_ponta_a_ponta_pd_elbe_e_ccf(painel, df_credito, carteira_viva, df_defaults_lgd):
    from yggdrasil.credit_risk.ecl import apply_elbe

    pd_lt = LifetimePD(method="vintage", horizon=48).fit(painel, by="produto")
    tabela_elbe = elbe_table(df_defaults_lgd, addon=0.05)
    base_ccf = reference_dataset(df_credito, method="fixed_horizon", segment_col="produto")
    ccf_por_produto = dict(zip(*pooled_ccf(base_ccf, by="produto")[["grupo", "ccf"]]
                               .to_numpy().T))

    carteira = apply_elbe(carteira_viva.drop(columns=["elbe"]), tabela_elbe,
                          months_col="meses_em_default")
    # a exposição dos rotativos incorpora o CCF sobre o limite disponível
    carteira["ccf"] = carteira["produto"].map({k: float(v) for k, v in ccf_por_produto.items()})
    carteira["ead"] = carteira["saldo"] * (1.0 + 0.2 * carteira["ccf"])

    res = ecl_table(carteira, model=pd_lt, lgd="lgd", ead="ead", exposure_col="saldo",
                    stage_col="estagio", elbe="elbe", discount_rate="taxa_efetiva",
                    age_col="idade", term_col="prazo")
    assert res.total > 0 and 0 < res.coverage < 1.5
    assert res.frame["elbe"].between(0, 1).all()
    assert set(res.summary()["estagio"]) == {1, 2, 3}


# ----------------------------------------------------------------------
# Relatório e MLflow (smoke)
# ----------------------------------------------------------------------
def test_todos_os_graficos_geram_figura(painel, modelo, resultado, df_credito,
                                        df_defaults_lgd):
    import matplotlib.pyplot as plt

    from yggdrasil.credit_risk.ecl import report

    _, km = kaplan_meier(painel, return_table=True)
    tabela_elbe = elbe_table(df_defaults_lgd, addon=0.05)
    base_ccf = reference_dataset(df_credito, method="variable", segment_col="produto")
    bt = modelo.backtest(painel, horizons=(12, 24))
    cen = ecl_scenarios(resultado.frame, modelo,
                        {"base": {"z": 0.0, "peso": 0.6}, "adverso": {"z": -1.0, "peso": 0.4}},
                        rho=0.1, lgd="lgd", ead="saldo", age_col="idade", term_col="prazo")

    figuras = [
        modelo.curve("cartao").plot("cumulative"),
        modelo.plot("hazard"),
        report.plot_survival_ci(km),
        report.plot_vintage_heatmap(painel, cohort_freq="Q"),
        report.plot_backtest(bt),
        tabela_elbe.plot(),
        base_ccf.plot(),
        report.plot_ccf_by_horizon(base_ccf),
        report.plot_ecl_scenarios(cen),
        report.plot_ecl_by(resultado, "produto"),
    ]
    for fig in figuras:
        assert fig.axes
        plt.close(fig)


def test_registro_no_mlflow(tmp_path, painel, modelo, resultado, df_credito,
                            df_defaults_lgd):
    import mlflow

    from yggdrasil.credit_risk.ecl import log_ccf, log_ecl_run, log_elbe, log_lifetime_pd

    mlflow.set_tracking_uri(f"file:{tmp_path.as_posix()}")
    experimento = "/tests/ecl"

    bt = modelo.backtest(painel, horizons=(12,))
    base_ccf = reference_dataset(df_credito, method="fixed_horizon", segment_col="produto")
    ids = [
        log_lifetime_pd(modelo, backtest=bt, experiment=experimento),
        log_elbe(elbe_table(df_defaults_lgd, addon=0.05), experiment=experimento),
        log_ccf(base_ccf, pooled=pooled_ccf(base_ccf, by="produto"), experiment=experimento),
        log_ecl_run(resultado, model=modelo, by=["produto"], experiment=experimento),
    ]
    assert all(isinstance(i, str) and i for i in ids)
    cliente = mlflow.tracking.MlflowClient()
    run = cliente.get_run(ids[-1])
    assert run.data.metrics["ecl_total"] == pytest.approx(resultado.total, rel=1e-6)
    assert run.data.tags["model_type"] == "ecl"
    assert run.data.tags["risk_parameter"] == "ecl"
