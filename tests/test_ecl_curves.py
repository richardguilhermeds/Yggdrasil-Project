"""
Testes do contrato de dados e da álgebra da curva
(``yggdrasil.credit_risk.ecl.panel`` e ``.curves``).

A propriedade central testada aqui é a **identidade entre as quatro
representações** da curva de PD: condicional, marginal, acumulada e
sobrevivência carregam a mesma informação, e a ida e a volta entre elas tem de
ser exata. É essa identidade que sustenta o resto do subpacote — o ECL soma a
marginal, o SICR olha a acumulada, o motor estima a condicional.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.ecl import (
    ContractPanel,
    PDCurve,
    constant_hazard,
    curve_frame,
    periods_per_year,
    vintage_curve,
)


# ----------------------------------------------------------------------
# Álgebra da curva
# ----------------------------------------------------------------------
def _curva_qualquer(seed: int = 0, n: int = 24) -> PDCurve:
    rng = np.random.default_rng(seed)
    return PDCurve.from_hazard(rng.uniform(0.001, 0.03, n), label="teste")


def test_roundtrip_das_quatro_representacoes():
    c = _curva_qualquer()
    assert np.allclose(PDCurve.from_marginal(c.marginal()).hazard_, c.hazard_)
    assert np.allclose(PDCurve.from_cumulative(c.cumulative()).hazard_, c.hazard_)
    assert np.allclose(PDCurve.from_survival(c.survival()).hazard_, c.hazard_)


def test_identidades_entre_representacoes():
    c = _curva_qualquer(seed=1)
    s, f, m, h = c.survival(), c.cumulative(), c.marginal(), c.hazard()
    assert np.allclose(f, 1.0 - s)                                 # F = 1 - S
    assert np.allclose(m.cumsum(), f)                              # Σ marginais = acumulada
    s_ant = np.concatenate(([1.0], s.to_numpy()[:-1]))
    assert np.allclose(m, s_ant * h)                               # m(t) = S(t-1)·h(t)
    assert s.is_monotonic_decreasing and f.is_monotonic_increasing


def test_forward_entre_horizontes():
    c = _curva_qualquer(seed=2)
    s = c.survival()
    assert c.forward(0, 12) == pytest.approx(c.pd_lifetime(12))
    assert c.forward(12, 24) == pytest.approx(1.0 - s.iloc[23] / s.iloc[11])
    assert c.forward(5, 5) == 0.0
    with pytest.raises(ValueError):
        c.forward(10, 5)


def test_to_frame_e_json_roundtrip(tmp_path):
    c = _curva_qualquer(seed=3)
    frame = c.to_frame()
    assert list(frame.columns) == ["pd_condicional", "pd_marginal", "pd_acumulada",
                                   "sobrevivencia"]
    assert frame.index.name == "horizonte" and frame.index[0] == 1

    caminho = tmp_path / "curva.json"
    c.to_json(str(caminho))
    assert np.allclose(PDCurve.from_dict(c.to_dict()).hazard_, c.hazard_)


def test_truncate_e_extend():
    c = _curva_qualquer(seed=4, n=12)
    assert len(c.truncate(5)) == 5
    est = c.extend(20)
    assert len(est) == 20
    assert np.allclose(est.hazard_[12:], c.hazard_[-1])            # extrapolação plana
    assert est.meta["extrapolada_apos"] == 12
    assert len(c.extend(6)) == 6                                   # extend menor = truncate


def test_validacoes_da_curva():
    with pytest.raises(ValueError):
        PDCurve.from_hazard([0.1, 1.5])                            # fora de [0, 1]
    with pytest.raises(ValueError):
        PDCurve.from_cumulative([0.2, 0.1])                        # acumulada decrescente
    with pytest.raises(ValueError):
        PDCurve.from_survival([0.5, 0.8])                          # sobrevivência crescente
    with pytest.raises(ValueError):
        PDCurve.from_marginal([0.6, 0.6])                          # Σ marginais > 1
    with pytest.raises(ValueError):
        PDCurve.from_hazard([])


# ----------------------------------------------------------------------
# constant_hazard
# ----------------------------------------------------------------------
@pytest.mark.parametrize("pd_12m", [0.01, 0.05, 0.30])
def test_constant_hazard_bate_a_pd_de_12_meses(pd_12m):
    c = constant_hazard(pd_12m, horizon=36)
    assert c.pd_12m() == pytest.approx(pd_12m)
    assert c.cumulative().iloc[11] == pytest.approx(pd_12m)
    assert np.allclose(c.hazard_, c.hazard_[0])                    # é constante mesmo
    assert c.pd_lifetime() > pd_12m                                # 36 meses > 12 meses


def test_constant_hazard_respeita_a_frequencia():
    anual = constant_hazard(0.05, horizon=5, freq="A")
    assert anual.hazard_[0] == pytest.approx(0.05)                 # 1 período = 1 ano
    assert periods_per_year("A") == 1 and periods_per_year("Q") == 4


# ----------------------------------------------------------------------
# calibrate_to — a ponte com o eixo transversal
# ----------------------------------------------------------------------
@pytest.mark.parametrize("alvo", [0.02, 0.09, 0.25])
def test_calibrate_to_bate_o_alvo_e_preserva_o_formato(alvo):
    c = _curva_qualquer(seed=5)
    cal = c.calibrate_to(pd_12m=alvo)
    assert cal.pd_12m() == pytest.approx(alvo, abs=1e-9)
    # o formato (ordem relativa dos hazards) não muda: é um shift no logit
    assert np.array_equal(np.argsort(cal.hazard_), np.argsort(c.hazard_))
    assert c.pd_12m() != pytest.approx(alvo)                       # a original não muda


def test_calibrate_to_com_horizonte_explicito():
    c = _curva_qualquer(seed=6)
    cal = c.calibrate_to(target=0.30, horizon=24)
    assert cal.pd_lifetime(24) == pytest.approx(0.30, abs=1e-9)
    assert cal.meta["calibracao"]["horizonte"] == 24
    with pytest.raises(ValueError):
        c.calibrate_to(target=0.3)                                 # target exige horizon
    with pytest.raises(ValueError):
        c.calibrate_to(pd_12m=0.1, target=0.2)                     # os dois juntos


def test_calibrate_to_recusa_alvo_inatingivel():
    c = PDCurve.from_hazard(np.zeros(12))                          # nada para deslocar
    with pytest.raises(ValueError, match="hazard positivo"):
        c.calibrate_to(pd_12m=0.05)


# ----------------------------------------------------------------------
# ContractPanel
# ----------------------------------------------------------------------
def test_painel_deriva_idade_e_valida(df_credito, painel):
    assert painel.n_contracts == df_credito["id_contrato"].nunique()
    assert painel.max_age <= 35 and painel.df[painel.age_col].min() == 0
    assert set(painel.segments()) == {"cartao", "consignado"}
    resumo = painel.summary()
    assert resumo.loc[0, "n_defaults"] > 0
    assert resumo.loc[0, "n_contratos"] == painel.n_contracts


def test_painel_recusa_dados_invalidos(df_credito):
    with pytest.raises(ValueError, match="duplicad"):
        ContractPanel(pd.concat([df_credito, df_credito.head(1)]),
                      origin_col="safra_origem")
    ruim = df_credito.copy()
    ruim.loc[ruim.index[0], "default"] = 2
    with pytest.raises(ValueError, match="binária"):
        ContractPanel(ruim, origin_col="safra_origem")
    faltante = df_credito.drop(columns=["default"])
    with pytest.raises(ValueError, match="ausentes"):
        ContractPanel(faltante, origin_col="safra_origem")


def test_painel_descarta_observacoes_pos_default():
    linhas = [("A", "2024-01-01", 0), ("A", "2024-02-01", 1), ("A", "2024-03-01", 1),
              ("B", "2024-01-01", 0), ("B", "2024-02-01", 0)]
    df = pd.DataFrame(linhas, columns=["id_contrato", "dt_ref", "default"])
    p = ContractPanel(df)
    assert len(p) == 4                                             # a 3ª de A saiu
    assert p.summary().loc[0, "obs_pos_default_descartadas"] == 1
    mantido = ContractPanel(df, drop_post_default=False)
    assert len(mantido) == 5


def test_at_risk_conta_censura_e_base(painel):
    vida = painel.at_risk()
    assert vida.index.name == "idade" and vida.index[0] == 0
    assert vida["n_em_risco"].iloc[0] == painel.n_contracts
    assert vida["n_em_risco"].is_monotonic_decreasing
    # todo contrato sai por default ou por censura, exatamente uma vez
    total = int(vida["n_default"].sum() + vida["n_censurado"].sum())
    assert total == painel.n_contracts
    assert (vida["hazard"].dropna() >= 0).all()


def test_at_risk_ponderado_por_exposicao(df_credito):
    p = ContractPanel(df_credito, origin_col="safra_origem", exposure_col="sacado")
    ponderado = p.at_risk(weighted=True)
    assert ponderado["n_em_risco"].iloc[0] == pytest.approx(
        df_credito.loc[df_credito["dt_ref"] == df_credito.groupby("id_contrato")["dt_ref"]
                       .transform("min"), "sacado"].sum())
    with pytest.raises(ValueError, match="exposure_col"):
        ContractPanel(df_credito, origin_col="safra_origem").at_risk(weighted=True)


def test_by_quebra_o_painel(painel):
    partes = painel.by()
    assert set(partes) == {"cartao", "consignado"}
    assert sum(len(p) for p in partes.values()) == len(painel)
    assert all(isinstance(p, ContractPanel) for p in partes.values())


def test_spells_e_features(painel):
    s = painel.spells()
    assert {painel.id_col, painel.age_col, painel.default_col} <= set(s.columns)
    assert len(s) == len(painel)
    with pytest.raises(ValueError, match="Features ausentes"):
        painel.spells(features=["nao_existe"])


# ----------------------------------------------------------------------
# vintage_curve
# ----------------------------------------------------------------------
def test_vintage_curve_recupera_a_maturacao(painel):
    curva, tabela = vintage_curve(painel, return_table=True, min_at_risk=30)
    assert len(curva) == len(tabela)
    assert curva.pd_lifetime() > curva.pd_12m() > 0
    assert (tabela["hazard_ic_inf"] <= tabela["hazard"].fillna(0)).all()
    assert (tabela["hazard"].fillna(0) <= tabela["hazard_ic_sup"]).all()
    # o DGP tem maturação: o hazard da segunda metade é maior que o da primeira
    meio = len(curva) // 2
    assert curva.hazard_[meio:].mean() > curva.hazard_[:meio].mean()


def test_vintage_curve_from_age_encurta_a_curva(painel):
    cheia = vintage_curve(painel)
    a_partir_de_6 = vintage_curve(painel, from_age=6)
    assert len(a_partir_de_6) == len(cheia) - 6
    assert np.allclose(a_partir_de_6.hazard_, cheia.hazard_[6:])


def test_vintage_curve_fill_e_min_at_risk(painel):
    # Limiar tirado do próprio painel: um a mais que a menor base observada, de
    # modo que as idades finais fiquem inválidas (não depende do DGP da fixture).
    limiar = int(painel.at_risk()["n_em_risco"].iloc[-1]) + 1
    livre = vintage_curve(painel, min_at_risk=1)
    ffill = vintage_curve(painel, min_at_risk=limiar, fill="ffill")
    zero = vintage_curve(painel, min_at_risk=limiar, fill="zero")
    corta = vintage_curve(painel, min_at_risk=limiar, fill="drop")

    assert len(corta) < len(livre)                                 # 'drop' trunca a cauda
    assert len(ffill) == len(zero) == len(livre)
    assert np.allclose(ffill.hazard_[len(corta):], ffill.hazard_[len(corta) - 1])
    assert np.allclose(zero.hazard_[len(corta):], 0.0)
    assert ffill.pd_lifetime() > zero.pd_lifetime()                # repetir > zerar

    with pytest.raises(ValueError, match="min_at_risk"):
        vintage_curve(painel, min_at_risk=10**6)                   # nada válido em nenhuma idade
    with pytest.raises(ValueError):
        vintage_curve(painel, fill="qualquer")


def test_curve_frame_empilha(painel):
    c1 = vintage_curve(painel, label="a")
    c2 = constant_hazard(0.05, len(c1))
    frame = curve_frame({"vintage": c1, "constante": c2})
    assert list(frame.columns) == ["vintage", "constante"]
    assert frame.index.name == "horizonte"
    assert np.allclose(frame["constante"], c2.cumulative())
    with pytest.raises(ValueError):
        curve_frame({"a": c1}, kind="inexistente")
