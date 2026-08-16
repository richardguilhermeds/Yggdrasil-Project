"""
Testes dos motores de PD *lifetime* e da fachada ``LifetimePD``
(``yggdrasil.credit_risk.ecl.survival``, ``.markov``, ``.lifetime_pd``).

As propriedades de sanidade que ancoram a suíte:

* Kaplan-Meier e a curva de safra coincidem no **ponto estimado** (em tempo
  discreto são o mesmo produto-limite); o que o KM acrescenta é a incerteza;
* a cadeia de Markov sobre a matriz de dois estados reproduz **hazard
  constante** — o caso em que a resposta é conhecida em forma fechada;
* o condicionamento ao ciclo é **idempotente em ``z = 0``** e tem o sinal certo,
  tanto pela curva quanto pela matriz de migração;
* ``apply`` respeita **idade** e **prazo**, e a soma das marginais é a PD
  *lifetime* de cada contrato;
* calibração e ciclo aplicados ao objeto **propagam** para a carteira escorada,
  inclusive no motor por contrato (``hazard``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from yggdrasil.credit_risk.capital.asrf import conditional_pd
from yggdrasil.credit_risk.capital.migration import two_state_matrix
from yggdrasil.credit_risk.ecl import (
    ContractPanel,
    DiscreteHazard,
    LifetimePD,
    MarkovPD,
    kaplan_meier,
    pd_curve_from_matrix,
    pit_from_ttc,
    vintage_curve,
)


# ----------------------------------------------------------------------
# Kaplan-Meier
# ----------------------------------------------------------------------
def test_km_coincide_com_a_curva_de_safra(painel):
    km = kaplan_meier(painel)
    vt = vintage_curve(painel, fill="zero")
    assert np.allclose(km.hazard_, vt.hazard_)


def test_km_traz_greenwood_e_ic_valido(painel):
    curva, tab = kaplan_meier(painel, return_table=True)
    assert {"se_greenwood", "sobrevivencia_ic_inf", "sobrevivencia_ic_sup"} <= set(tab.columns)
    assert (tab["sobrevivencia_ic_inf"] <= tab["sobrevivencia"] + 1e-12).all()
    assert (tab["sobrevivencia"] <= tab["sobrevivencia_ic_sup"] + 1e-12).all()
    assert (tab["se_greenwood"] >= 0).all()
    assert np.allclose(tab["pd_acumulada"], curva.cumulative().to_numpy())
    # o IC aperta quando a base é grande e abre na cauda
    largura = tab["sobrevivencia_ic_sup"] - tab["sobrevivencia_ic_inf"]
    assert largura.iloc[-1] > largura.iloc[0]


def test_km_recupera_a_verdade_de_um_dgp_conhecido():
    """Com hazard constante conhecido e censura aleatória, o KM tem de acertar."""
    rng = np.random.default_rng(0)
    h_verdadeiro, n = 0.02, 4000
    linhas = []
    for i in range(n):
        censura = int(rng.integers(6, 25))
        for t in range(24):
            quebrou = int(rng.uniform() < h_verdadeiro)
            linhas.append((f"C{i}", pd.Timestamp("2024-01-01") + pd.DateOffset(months=t), quebrou))
            if quebrou or t >= censura:
                break
    p = ContractPanel(pd.DataFrame(linhas, columns=["id_contrato", "dt_ref", "default"]))
    curva = kaplan_meier(p, horizon=6)
    assert curva.hazard_.mean() == pytest.approx(h_verdadeiro, abs=0.006)


# ----------------------------------------------------------------------
# Hazard em tempo discreto
# ----------------------------------------------------------------------
@pytest.mark.parametrize("baseline", ["dummies", "spline", "linear", "log"])
def test_discrete_hazard_ajusta_em_todos_os_baselines(painel, baseline):
    df = painel.df.copy()
    df["feat_score"] = np.where(df["rating"] == "A", -1.0,
                                np.where(df["rating"] == "B", 0.0, 1.0))
    p = ContractPanel(df, age_col=painel.age_col, segment_col="produto",
                      drop_post_default=False)
    mh = DiscreteHazard(baseline=baseline).fit(p, features=["feat_score"])
    curva = mh.baseline_curve(horizon=36)
    assert 0 < curva.pd_12m() < curva.pd_lifetime() < 1
    coef = mh.coef_frame()
    assert coef.iloc[0]["termo"] == "(intercepto)" and "odds_ratio" in coef.columns
    # o score entra com sinal positivo (rating pior = mais risco)
    assert float(coef[coef["termo"] == "feat_score"]["coeficiente"].iloc[0]) > 0


def test_discrete_hazard_separa_perfis_de_risco(painel):
    df = painel.df.copy()
    df["feat_score"] = np.where(df["rating"] == "A", -1.0,
                                np.where(df["rating"] == "B", 0.0, 1.0))
    p = ContractPanel(df, age_col=painel.age_col, drop_post_default=False)
    mh = DiscreteHazard(baseline="spline").fit(p, features=["feat_score"])
    assert (mh.predict_curve([1.0], horizon=24).pd_12m()
            > mh.predict_curve([-1.0], horizon=24).pd_12m())
    matriz = mh.predict_curves(df.head(20), horizon=12, age_col=p.age_col)
    assert matriz.shape == (20, 12) and np.all((matriz >= 0) & (matriz <= 1))


def test_discrete_hazard_valida_uso(painel):
    mh = DiscreteHazard()
    with pytest.raises(RuntimeError, match="ainda não foi ajustado"):
        mh.predict_curve(horizon=12)
    with pytest.raises(ValueError):
        DiscreteHazard(baseline="inexistente")
    with pytest.raises(ValueError):
        DiscreteHazard(link="inexistente")
    with pytest.raises(ValueError, match="Features ausentes"):
        DiscreteHazard().fit(painel, features=["nao_existe"])


# ----------------------------------------------------------------------
# Markov
# ----------------------------------------------------------------------
def test_markov_sobre_dois_estados_reproduz_hazard_constante():
    tm, ratings, _ = two_state_matrix(0.02)
    curva = pd_curve_from_matrix(tm, ratings, horizon=24, from_rating="P", freq="A")
    assert np.allclose(curva.hazard_, 0.02)
    assert curva.cumulative().iloc[0] == pytest.approx(0.02)
    # o estado absorvente não gera curva própria
    todas = pd_curve_from_matrix(tm, ratings, horizon=6)
    assert set(todas) == {"P"}


def test_markov_ordena_os_ratings(painel):
    mk = MarkovPD().fit(painel, rating_col="rating")
    curvas = mk.curves(horizon=36)
    assert curvas["A"].pd_12m() < curvas["B"].pd_12m() < curvas["C"].pd_12m()
    tabela = mk.pd_by_rating((12, 24, 36))
    assert list(tabela.columns) == ["rating", "pd_12p", "pd_24p", "pd_36p"]
    matriz = mk.matrix_frame()
    assert np.allclose(matriz.to_numpy().sum(axis=1), 1.0)
    assert matriz.loc["D", "D"] == pytest.approx(1.0)              # default absorvente


def test_markov_condition_desloca_a_matriz(painel):
    mk = MarkovPD().fit(painel, rating_col="rating")
    base = mk.curve("A", 24).pd_12m()
    assert mk.condition(-1.5, 0.08).curve("A", 24).pd_12m() > base
    assert mk.condition(+1.5, 0.08).curve("A", 24).pd_12m() < base
    assert np.allclose(mk.condition(0.0, 0.08).matrix_, mk.matrix_, atol=1e-9)


def test_markov_exige_default_por_ultimo(painel):
    with pytest.raises(ValueError, match="ÚLTIMO"):
        MarkovPD(ratings=["D", "A", "B", "C"], default_state="D").fit(painel, rating_col="rating")
    with pytest.raises(ValueError, match="rating_col"):
        MarkovPD().fit(ContractPanel(painel.df, age_col=painel.age_col,
                                     drop_post_default=False))


# ----------------------------------------------------------------------
# Condicionamento de Vasicek
# ----------------------------------------------------------------------
def test_pit_from_ttc_modos_e_sinal():
    rho, q = 0.12, 0.9
    # 'conditional' é a lei do fator único — bate com o motor de capital
    assert pit_from_ttc(0.05, rho, -norm.ppf(q), mode="conditional") == pytest.approx(
        float(conditional_pd(0.05, rho, q)))
    # 'shift' é idempotente em z = 0 (e é o padrão)
    assert pit_from_ttc(0.05, rho, 0.0) == pytest.approx(0.05)
    assert pit_from_ttc(0.05, rho, -1.0) > 0.05 > pit_from_ttc(0.05, rho, 1.0)
    # rho = 0 desliga o ciclo nos dois modos
    assert pit_from_ttc(0.05, 0.0, -3.0) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        pit_from_ttc(0.05, 1.0, 0.0)
    with pytest.raises(ValueError):
        pit_from_ttc(0.05, 0.1, 0.0, mode="qualquer")


# ----------------------------------------------------------------------
# Fachada
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ["constant", "vintage", "km", "survival"])
def test_fachada_ajusta_por_grupo(painel, method):
    lt = LifetimePD(method=method, horizon=48).fit(painel, by="produto")
    assert set(lt.curves_) == {"cartao", "consignado"}
    resumo = lt.summary()
    assert len(resumo) == 2 and (resumo["pd_lifetime"] >= resumo["pd_12m"]).all()
    # o DGP separa os produtos por um fator ~3,5 no hazard
    assert lt.curve("cartao").pd_12m() > lt.curve("consignado").pd_12m()
    assert lt.frame("cumulative").shape == (48, 2)


def test_fachada_survival_resolve_o_apelido(painel):
    assert LifetimePD(method="survival").fit(painel).method_ == "km"
    df = painel.df.assign(feat=1.0 * (painel.df["rating"] == "C"))
    p = ContractPanel(df, age_col=painel.age_col, drop_post_default=False)
    assert LifetimePD(method="survival").fit(p, features=["feat"]).method_ == "hazard"
    with pytest.raises(ValueError, match="exige features"):
        LifetimePD(method="hazard").fit(painel)


def test_fachada_markov_e_from_pd_12m(painel):
    mk = LifetimePD(method="markov", horizon=36).fit(painel, rating_col="rating")
    assert set(mk.curves_) == {"A", "B", "C"}
    direto = LifetimePD.from_pd_12m({"cartao": 0.08, "consignado": 0.02}, horizon=24)
    assert direto.curve("cartao").pd_12m() == pytest.approx(0.08)
    unica = LifetimePD.from_pd_12m(0.05, horizon=24)
    assert unica.curve().pd_12m() == pytest.approx(0.05)


def test_apply_respeita_idade_e_prazo(painel, carteira_viva):
    lt = LifetimePD(method="vintage", horizon=48).fit(painel, by="produto")
    out = lt.apply(carteira_viva, age_col="idade", term_col="prazo", detail=True)
    cols = [f"pd_marg_h{t}" for t in range(1, 49)]
    assert np.allclose(out[cols].sum(axis=1), out["pd_lifetime"])
    assert np.allclose(out[cols[:12]].sum(axis=1), out["pd_12m"])
    assert (out["pd_lifetime"] >= out["pd_12m"] - 1e-12).all()

    curtos = out[out["prazo"] <= 5]
    assert len(curtos) and np.allclose(curtos[[f"pd_marg_h{t}" for t in range(6, 49)]], 0.0)
    zerados = out[out["prazo"] == 0]
    if len(zerados):
        assert np.allclose(zerados["pd_lifetime"], 0.0)

    resumo = lt.apply(carteira_viva, age_col="idade", detail=False)
    assert not [c for c in resumo.columns if c.startswith("pd_marg_h")]


def test_apply_exige_a_coluna_de_grupo(painel, carteira_viva):
    lt = LifetimePD(method="vintage", horizon=24).fit(painel, by="produto")
    with pytest.raises(ValueError, match="segment_col"):
        lt.apply(carteira_viva.drop(columns=["produto"]), age_col="idade")
    desconhecido = carteira_viva.assign(produto="outro")
    with pytest.raises(ValueError, match="sem curva ajustada"):
        lt.apply(desconhecido, age_col="idade")


def test_calibrate_to_por_grupo(painel):
    lt = LifetimePD(method="vintage", horizon=36).fit(painel, by="produto")
    cal = lt.calibrate_to({"cartao": 0.12, "consignado": 0.03})
    assert cal.curve("cartao").pd_12m() == pytest.approx(0.12, abs=1e-9)
    assert cal.curve("consignado").pd_12m() == pytest.approx(0.03, abs=1e-9)
    assert lt.curve("cartao").pd_12m() != pytest.approx(0.12)      # o original não muda
    escalar = lt.calibrate_to(0.05)
    assert all(c.pd_12m() == pytest.approx(0.05, abs=1e-9) for c in escalar.curves_.values())


def test_condition_sinal_idempotencia_e_decay(painel):
    lt = LifetimePD(method="vintage", horizon=36).fit(painel, by="produto")
    base = lt.curve("cartao").pd_12m()
    assert np.allclose(lt.condition(0.0, 0.10).curve("cartao").hazard_,
                       lt.curve("cartao").hazard_)
    assert lt.condition(-1.5, 0.10).curve("cartao").pd_12m() > base
    assert lt.condition(+1.5, 0.10).curve("cartao").pd_12m() < base
    # a reversão à média dissipa o choque ao longo do horizonte
    sem = lt.condition(-1.5, 0.10).curve("cartao").pd_lifetime()
    com = lt.condition(-1.5, 0.10, decay=0.2).curve("cartao").pd_lifetime()
    assert sem > com > lt.curve("cartao").pd_lifetime()
    # z por horizonte (a ponte com o modelo satélite)
    vetorial = lt.condition(np.linspace(-2.0, 0.0, 36), 0.10)
    assert vetorial.curve("cartao").pd_12m() > base


def test_ajustes_propagam_para_a_carteira_no_motor_hazard(painel, carteira_viva):
    df = painel.df.assign(feat_score=np.where(painel.df["rating"] == "C", 1.0, -1.0))
    p = ContractPanel(df, age_col=painel.age_col, segment_col="produto",
                      drop_post_default=False)
    lt = LifetimePD(method="hazard", horizon=36, baseline="spline").fit(
        p, by="produto", features=["feat_score"])
    carteira = carteira_viva.assign(
        feat_score=np.where(carteira_viva["rating"] == "C", 1.0, -1.0))

    base = lt.apply(carteira, age_col="idade", detail=False)
    cal = lt.calibrate_to({"cartao": 0.30, "consignado": 0.01})
    apl = cal.apply(carteira, age_col="idade", detail=False)
    for produto, sentido in (("cartao", 1), ("consignado", -1)):
        m_base = base.loc[base["produto"] == produto, "pd_12m"].mean()
        m_cal = apl.loc[apl["produto"] == produto, "pd_12m"].mean()
        assert sentido * (m_cal - m_base) > 0

    idem = cal.condition(0.0, 0.10).apply(carteira, age_col="idade", detail=False)
    assert np.allclose(idem["pd_lifetime"], apl["pd_lifetime"])
    adverso = cal.condition(-1.5, 0.10).apply(carteira, age_col="idade", detail=False)
    assert adverso["pd_lifetime"].mean() > apl["pd_lifetime"].mean()


def test_backtest_confronta_previsto_e_observado(painel):
    lt = LifetimePD(method="vintage", horizon=36).fit(painel, by="produto")
    bt = lt.backtest(painel, horizons=(12, 24))
    assert set(bt["grupo"]) == {"cartao", "consignado"}
    # em amostra, previsto == observado e tudo cai dentro do IC
    assert np.allclose(bt["pd_prevista"], bt["pd_observada"])
    assert bt["dentro_do_ic"].all()
    assert (bt["ic_inf"] <= bt["pd_observada"]).all()

    deslocado = lt.calibrate_to({"cartao": 0.60, "consignado": 0.55})
    fora = deslocado.backtest(painel, horizons=(12,))
    assert not fora["dentro_do_ic"].any()                          # o backtest acusa


def test_json_roundtrip_preserva_curvas_e_ajustes(painel, tmp_path):
    lt = LifetimePD(method="vintage", horizon=24).fit(painel, by="produto")
    cal = lt.calibrate_to(0.07).condition(-1.0, 0.10)
    caminho = tmp_path / "pd.json"
    cal.to_json(str(caminho))
    lido = LifetimePD.from_json(str(caminho))
    assert np.allclose(lido.curve("cartao").hazard_, cal.curve("cartao").hazard_)
    assert [a["tipo"] for a in lido.adjustments_] == ["logit_shift", "vasicek"]
    assert lido.method_ == cal.method_ and lido.horizon == cal.horizon


def test_fachada_valida_uso(painel):
    with pytest.raises(ValueError):
        LifetimePD(method="inexistente")
    with pytest.raises(ValueError):
        LifetimePD(horizon=0)
    with pytest.raises(RuntimeError, match="chame .fit"):
        LifetimePD().curve()
    with pytest.raises(TypeError):
        LifetimePD().fit(painel.df)                                # DataFrame cru
    lt = LifetimePD(method="vintage", horizon=12).fit(painel, by="produto")
    with pytest.raises(ValueError, match="informe o rótulo"):
        lt.curve()
    with pytest.raises(KeyError):
        lt.curve("inexistente")
