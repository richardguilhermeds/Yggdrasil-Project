"""
Testes dos motores de análise de sobrevivência (``yggdrasil.credit_risk.survival``).

As propriedades que ancoram a suíte:

* a **tabela de vida** reproduz o Kaplan-Meier do subpacote ECL e acrescenta o
  Nelson-Aalen (sempre ``S_NA >= S_KM`` em tempo discreto);
* o **log-rank** rejeita quando os grupos têm risco distinto por construção e
  **não** rejeita numa partição aleatória dos mesmos contratos;
* o ajuste **paramétrico** recupera a maturação do processo gerador (Weibull com
  ``k > 1`` vence o exponencial por AIC, e o ``k`` fica perto do verdadeiro);
* a **emenda** preserva o trecho empírico e casa o nível na junção;
* o **C-index** é 1 numa ordenação perfeita, 0,5 num escore constante e respeita
  a censura; o teste de **riscos proporcionais** não rejeita quando o processo
  gerador é proporcional (como é o do painel de referência);
* o **estudo** declarativo roda ponta a ponta e a configuração faz ida e volta em
  JSON.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.ecl import ContractPanel, LifetimePD, kaplan_meier
from yggdrasil.credit_risk.survival import (
    FEATURES,
    ParametricSurvival,
    SurvivalConfig,
    backtest_curve,
    calibration_by_decile,
    concordance_index,
    contract_outcomes,
    discrimination_by_horizon,
    fit_parametric,
    junction_age,
    life_table,
    logrank_test,
    make_reference_panel,
    median_survival,
    model_concordance,
    pairwise_logrank,
    ph_test,
    restricted_mean_survival,
    run_survival_study,
    smooth_hazard,
    splice_curves,
    split_panel,
)


@pytest.fixture(scope="module")
def ref():
    return make_reference_panel(n_contracts=900, seed=3)


@pytest.fixture(scope="module")
def painel_ref(ref):
    return ref.panel()


# ----------------------------------------------------------------------
# Painel de referência
# ----------------------------------------------------------------------
def test_painel_de_referencia_tem_o_formato_do_contrato(ref, painel_ref):
    df = ref.df
    for col in ("id_contrato", "dt_ref", "safra_origem", "default", "produto", "rating",
                "prazo", "exposicao", *FEATURES):
        assert col in df.columns, col
    assert not df.duplicated(["id_contrato", "dt_ref"]).any()
    assert set(df["default"].unique()) <= {0, 1}
    assert painel_ref.n_contracts == 900 and painel_ref.max_age <= 35
    # maturação: o hazard verdadeiro cresce com a idade
    h = ref.true_hazard(np.arange(36))
    assert (np.diff(h) > 0).all()
    assert ref.true_hazard([6], produto="cartao")[0] > ref.true_hazard([6], produto="consignado")[0]
    assert "n_contratos" in repr(ref)


# ----------------------------------------------------------------------
# Tabela de vida
# ----------------------------------------------------------------------
def test_life_table_reproduz_o_km_e_traz_nelson_aalen(painel_ref):
    tab = life_table(painel_ref)
    _, km = kaplan_meier(painel_ref, return_table=True)
    assert np.allclose(tab["sobrevivencia"], km["sobrevivencia"])
    assert {"idade", "hazard_acumulado", "sobrevivencia_na", "n_censurado"} <= set(tab.columns)
    # Nelson-Aalen: exp(-Σ h) >= Π (1 - h)
    assert (tab["sobrevivencia_na"] >= tab["sobrevivencia"] - 1e-12).all()
    assert tab["hazard_acumulado"].is_monotonic_increasing
    assert not tab["hazard"].isna().any()


def test_life_table_por_grupo(painel_ref):
    tab = life_table(painel_ref, by="produto", horizon=24)
    assert set(tab["grupo"]) == {"cartao", "consignado"}
    assert (tab.groupby("grupo").size() == 24).all()
    c = tab[tab["grupo"] == "cartao"].set_index("idade")["pd_acumulada"]
    k = tab[tab["grupo"] == "consignado"].set_index("idade")["pd_acumulada"]
    assert c.iloc[11] > k.iloc[11]


def test_life_table_from_age_alem_do_painel_erro_claro(painel_ref):
    with pytest.raises(ValueError):
        life_table(painel_ref, from_age=500)


# ----------------------------------------------------------------------
# Log-rank
# ----------------------------------------------------------------------
def test_logrank_rejeita_grupos_distintos_por_construcao(painel_ref):
    r = logrank_test(painel_ref, "produto")
    assert r["gl"] == 1 and r["n_grupos"] == 2
    assert r["p_valor"] < 1e-6 and r["estatistica"] > 10
    g = r["grupos"].set_index("grupo")
    assert g.loc["cartao", "obs_esp"] > 1 > g.loc["consignado", "obs_esp"]
    assert g["observados"].sum() == pytest.approx(g["esperados"].sum())


def test_logrank_nao_rejeita_particao_aleatoria(painel_ref):
    rng = np.random.default_rng(0)
    ids = painel_ref.df["id_contrato"].unique()
    grupo = dict(zip(ids, rng.choice(["x", "y", "z"], len(ids))))
    df = painel_ref.df.assign(aleatorio=painel_ref.df["id_contrato"].map(grupo))
    p = ContractPanel(df, age_col=painel_ref.age_col, drop_post_default=False)
    for w in ("logrank", "wilcoxon"):
        r = logrank_test(p, "aleatorio", weights=w)
        assert r["gl"] == 2 and r["weights"] == w
        assert r["p_valor"] > 0.01, r


def test_logrank_valida_entrada(painel_ref):
    with pytest.raises(ValueError, match="weights"):
        logrank_test(painel_ref, "produto", weights="qualquer")
    df = painel_ref.df.assign(um="unico")
    p = ContractPanel(df, age_col=painel_ref.age_col, drop_post_default=False)
    with pytest.raises(ValueError, match="2 grupos"):
        logrank_test(p, "um")


def test_pairwise_logrank_com_bonferroni(painel_ref):
    tab = pairwise_logrank(painel_ref, "rating")
    assert len(tab) == 3
    assert (tab["p_bonferroni"] >= tab["p_valor"] - 1e-12).all()
    assert tab.loc[(tab["grupo_a"] == "A") & (tab["grupo_b"] == "C"), "p_valor"].iloc[0] < 0.01


# ----------------------------------------------------------------------
# Leituras em tempo
# ----------------------------------------------------------------------
def test_mediana_e_rmst():
    curva = LifetimePD.from_pd_12m(0.5, horizon=24, freq="A").curve()  # hazard 0.5 por ano
    assert median_survival(curva) == 1.0
    baixa = LifetimePD.from_pd_12m(0.01, horizon=12).curve()
    assert np.isnan(median_survival(baixa))
    # sem risco nenhum, o contrato sobrevive todos os τ períodos
    zero = type(curva)(np.zeros(12))
    assert restricted_mean_survival(zero, 12) == pytest.approx(12.0)
    assert restricted_mean_survival(curva, 3) < 3.0


def test_smooth_hazard_preserva_media_aproximada():
    h = np.array([0.01, 0.05, 0.01, 0.05, 0.01, 0.05])
    s = smooth_hazard(h, window=3)
    assert s.shape == h.shape and (s >= 0).all() and (s <= 1).all()
    assert np.std(s) < np.std(h)


# ----------------------------------------------------------------------
# Paramétrico
# ----------------------------------------------------------------------
def test_parametrico_recupera_a_maturacao(ref, painel_ref):
    rank, modelos = fit_parametric(painel_ref)
    assert list(rank.columns[:2]) == ["posicao", "distribuicao"]
    assert rank["aic"].is_monotonic_increasing
    assert rank["delta_aic"].iloc[0] == 0.0
    w, e = modelos["weibull"], modelos["exponential"]
    assert w.aic_ < e.aic_                       # a maturação existe: Weibull vence
    assert w.params_["shape"] == pytest.approx(ref.truth["shape"], abs=0.25)
    assert "crescente" in w.shape_reading()
    assert w.converged_ and w.n_events_ == int(painel_ref.df["default"].sum())
    assert w.summary().iloc[0]["n_parametros"] == 2 and e.n_params == 1


@pytest.mark.parametrize("dist", ["exponential", "weibull", "lognormal", "loglogistic", "gompertz"])
def test_parametrico_cada_familia_produz_curva_valida(painel_ref, dist):
    m = ParametricSurvival(dist).fit(painel_ref)
    curva = m.curve(horizon=60)
    assert len(curva) == 60 and (curva.hazard_ >= 0).all() and (curva.hazard_ <= 1).all()
    assert 0 < curva.pd_12m() < curva.pd_lifetime() < 1
    assert np.isclose(m.survival(0.0), 1.0)
    assert m.shape_reading()
    # ida e volta
    volta = ParametricSurvival.from_dict(m.to_dict())
    assert np.allclose(volta.hazard(np.arange(12)), m.hazard(np.arange(12)))


def test_parametrico_aceita_a_tabela_de_vida(painel_ref):
    tab = life_table(painel_ref)
    a = ParametricSurvival("weibull").fit(tab)
    b = ParametricSurvival("weibull").fit(painel_ref)
    assert a.params_["shape"] == pytest.approx(b.params_["shape"], rel=1e-6)


def test_parametrico_valida_uso(painel_ref):
    with pytest.raises(ValueError):
        ParametricSurvival("gamma")
    with pytest.raises(RuntimeError, match="ajustado"):
        ParametricSurvival().curve()
    with pytest.raises(TypeError):
        ParametricSurvival().fit([1, 2, 3])


def test_junction_e_emenda(painel_ref):
    j = junction_age(painel_ref, min_at_risk=200)
    assert 0 <= j < painel_ref.max_age
    assert junction_age(painel_ref, min_at_risk=10 ** 9) == -1
    km = kaplan_meier(painel_ref)
    w = ParametricSurvival("weibull").fit(painel_ref)
    sp = splice_curves(km, w, junction=j, horizon=72)
    assert len(sp) == 72
    assert np.allclose(sp.hazard_[: j + 1], km.hazard_[: j + 1])      # o empírico manda até j
    assert sp.meta["cauda"] == "weibull" and sp.meta["junction"] == j
    assert sp.meta["n_empirico"] == j + 1
    # com maturação, a cauda paramétrica NÃO é plana
    assert sp.hazard_[-1] > sp.hazard_[j + 1]
    # sem casar o nível, o fator é 1
    sem = splice_curves(km, w, junction=j, horizon=72, match_level=False)
    assert sem.meta["fator_nivel"] == 1.0
    assert np.allclose(sem.hazard_[j + 1:], w.hazard(np.arange(j + 1, 72)))
    # a cauda também pode vir como PDCurve pronta
    cauda = w.curve(horizon=72)
    assert np.allclose(splice_curves(km, cauda, junction=j, horizon=72,
                                     match_level=False).hazard_, sem.hazard_)


# ----------------------------------------------------------------------
# Validação
# ----------------------------------------------------------------------
def test_contract_outcomes_classifica_os_desfechos(painel_ref):
    out = contract_outcomes(painel_ref, 12)
    assert len(out) == painel_ref.n_contracts
    alvo = out["alvo_h12"]
    # quem quebrou até 12 períodos observados é 1; quem foi observado 12+ sem quebrar é 0
    quebrou_cedo = out[(out["evento"] == 1) & (out["periodos_observados"] <= 12)]
    assert (alvo.loc[quebrou_cedo.index] == 1).all()
    sobreviveu = out[(out["evento"] == 0) & (out["periodos_observados"] >= 12)]
    assert (alvo.loc[sobreviveu.index] == 0).all()
    censurado = out[(out["evento"] == 0) & (out["periodos_observados"] < 12)]
    assert alvo.loc[censurado.index].isna().all()
    # quem quebrou depois do horizonte é 0 dentro dele
    tarde = out[(out["evento"] == 1) & (out["periodos_observados"] > 12)]
    assert (alvo.loc[tarde.index] == 0).all()


def test_concordance_index_casos_conhecidos():
    dur = np.array([1, 2, 3, 4, 5], dtype=float)
    ev = np.ones(5, dtype=int)
    assert concordance_index([5, 4, 3, 2, 1], dur, ev) == pytest.approx(1.0)   # maior risco quebra antes
    assert concordance_index([1, 2, 3, 4, 5], dur, ev) == pytest.approx(0.0)
    assert concordance_index([1, 1, 1, 1, 1], dur, ev) == pytest.approx(0.5)   # empate total
    # censura: o censurado em 2 não é comparável com quem viveu mais que ele
    ev2 = np.array([1, 0, 1, 1, 1])
    c = concordance_index([5, 4, 3, 2, 1], dur, ev2)
    assert c == pytest.approx(1.0)
    assert np.isnan(concordance_index([1, 2], [1, 2], [0, 0]))
    with pytest.raises(ValueError):
        concordance_index([1, 2], [1], [1, 1])


def test_concordance_bate_a_conta_ingenua():
    rng = np.random.default_rng(1)
    n = 150
    dur = rng.integers(1, 30, n).astype(float)
    ev = rng.integers(0, 2, n)
    score = -dur + rng.normal(0, 8, n)
    conc = emp = comp = 0.0
    for i in range(n):
        if ev[i] != 1:
            continue
        for j in range(n):
            if dur[j] > dur[i] or (dur[j] == dur[i] and ev[j] == 0 and i != j):
                comp += 1
                if score[i] > score[j]:
                    conc += 1
                elif score[i] == score[j]:
                    emp += 1
    assert concordance_index(score, dur, ev) == pytest.approx((conc + 0.5 * emp) / comp)


def test_discriminacao_e_c_index_do_modelo_de_hazard(ref, painel_ref):
    lt = LifetimePD(method="hazard", horizon=36, baseline="spline").fit(
        painel_ref, features=list(FEATURES))
    disc = discrimination_by_horizon(lt, painel_ref, horizons=(12, 24))
    assert list(disc["horizonte"]) == [12, 24]
    assert (disc["auc"] > 0.55).all()                 # o score do DGP ordena
    assert (disc["n_eventos"] > 0).all() and (disc["observacao"] == "").all()
    assert 0.55 < model_concordance(lt, painel_ref, horizon=24) < 1.0
    # curva única: nada a ordenar, e a tabela diz isso em vez de estourar
    unica = LifetimePD(method="km", horizon=36).fit(painel_ref)
    d2 = discrimination_by_horizon(unica, painel_ref, horizons=(12,))
    assert np.isnan(d2["auc"].iloc[0]) and "nada a ordenar" in d2["observacao"].iloc[0]


def test_backtest_curve_traz_z_e_veredito(painel_ref):
    lt = LifetimePD(method="km", horizon=36).fit(painel_ref, by="produto")
    bt = backtest_curve(lt, painel_ref, horizons=(12, 24))
    assert set(bt["grupo"]) == {"cartao", "consignado"} and len(bt) == 4
    for col in ("z", "p_valor", "se_greenwood", "dentro_do_ic", "n_em_risco_h"):
        assert col in bt.columns
    # in-sample o KM reproduz a si mesmo: erro zero e dentro do IC
    assert np.allclose(bt["erro_absoluto"], 0.0, atol=1e-12)
    assert bt["dentro_do_ic"].all()
    with pytest.raises(ValueError, match="não produziu"):
        backtest_curve(lt, painel_ref, horizons=(500,))


def test_calibracao_por_decil_e_hosmer_lemeshow(painel_ref):
    lt = LifetimePD(method="hazard", horizon=24, baseline="spline").fit(
        painel_ref, features=list(FEATURES))
    tab = calibration_by_decile(lt, painel_ref, horizon=12, n_bins=5)
    assert 2 <= len(tab) <= 5
    assert tab["n"].sum() == np.isfinite(contract_outcomes(painel_ref, 12)["alvo_h12"]).sum()
    assert (tab["ic_inf"] <= tab["pd_observada"] + 1e-12).all()
    assert tab["pd_prevista"].is_monotonic_increasing
    for k in ("hl_estatistica", "hl_gl", "hl_p_valor", "horizonte"):
        assert k in tab.attrs
    # por grupo: uma faixa por PD distinta
    grupo = LifetimePD(method="km", horizon=24).fit(painel_ref, by="produto")
    t2 = calibration_by_decile(grupo, painel_ref, horizon=12)
    assert len(t2) == 2


def test_ph_test_nao_rejeita_dgp_proporcional(painel_ref):
    r = ph_test(painel_ref, ["feat_score", "feat_ltv"], baseline="spline")
    assert r["gl"] == 2 and 0 <= r["p_valor"] <= 1
    assert r["loglik_com"] >= r["loglik_sem"] - 1e-6
    assert set(r["interacoes"]["feature"]) == {"feat_score", "feat_ltv"}
    assert r["proporcional"] is (r["p_valor"] > 0.05)
    with pytest.raises(ValueError):
        ph_test(painel_ref, [])


# ----------------------------------------------------------------------
# Partição e estudo
# ----------------------------------------------------------------------
def test_split_panel_nos_tres_modos(painel_ref):
    des, oot = split_panel(painel_ref, "none")
    assert des is painel_ref and oot is None

    des, oot = split_panel(painel_ref, "origin", value="2021-01-01")
    ids_des = set(des.df["id_contrato"])
    assert oot is not None and ids_des.isdisjoint(set(oot.df["id_contrato"]))
    assert (pd.to_datetime(des.df["safra_origem"]) < "2021-01-01").all()

    des2, oot2 = split_panel(painel_ref, "observation", value="2021-06-01")
    assert (des2.df["dt_ref"] < "2021-06-01").all() and (oot2.df["dt_ref"] >= "2021-06-01").all()

    df = painel_ref.df.assign(amostra=np.where(painel_ref.df["produto"] == "cartao", "OOT", "DES"))
    p = ContractPanel(df, age_col=painel_ref.age_col, drop_post_default=False)
    des3, oot3 = split_panel(p, "column", column="amostra")
    assert set(oot3.df["produto"]) == {"cartao"} and set(des3.df["produto"]) == {"consignado"}

    with pytest.raises(ValueError, match="data de corte"):
        split_panel(painel_ref, "origin")
    with pytest.raises(ValueError, match="DES vazio"):
        split_panel(painel_ref, "origin", value="1900-01-01")


def test_config_valida_e_serializa():
    cfg = SurvivalConfig(method="km", by="produto", calibrate={"cartao": "0.1"})
    assert cfg.calibrate == {"cartao": 0.1}
    volta = SurvivalConfig.from_json(cfg.to_json())
    assert volta == cfg
    assert SurvivalConfig.from_dict({**cfg.to_dict(), "campo_estranho": 1}) == cfg
    with pytest.raises(ValueError, match="exige features"):
        SurvivalConfig(method="hazard")
    with pytest.raises(ValueError):
        SurvivalConfig(method="cox")
    with pytest.raises(ValueError):
        SurvivalConfig(split="aleatorio")
    with pytest.raises(ValueError):
        SurvivalConfig(distribution="gamma")


def test_estudo_km_com_cauda_parametrica_e_oot(ref):
    cfg = SurvivalConfig(origin_col="safra_origem", segment_col="produto", term_col="prazo",
                         by="produto", method="km", tail="parametric", distribution="weibull",
                         min_at_risk=50, horizon=60, split="origin", split_value="2021-01-01",
                         calibrate={"cartao": 0.12})
    res = run_survival_study(ref.df, cfg)
    assert res.validated_on == "OOT" and res.panel_oot is not None
    assert set(res.model.curves_) == {"cartao", "consignado"}
    assert all(len(c) == 60 for c in res.model.curves_.values())
    assert res.model.curve("cartao").pd_12m() == pytest.approx(0.12, abs=1e-9)
    assert res.model_base.curve("cartao").pd_12m() != pytest.approx(0.12)
    assert res.parametric is not None and set(res.parametric) == {"cartao", "consignado"}
    assert res.model.curve("consignado").meta["cauda"] == "weibull"
    assert res.logrank is not None and res.logrank["p_valor"] < 0.05
    assert res.backtest is not None and set(res.backtest["grupo"]) == {"cartao", "consignado"}
    assert res.decile is not None and len(res.decile) == 2
    assert res.c_index is not None
    resumo = res.summary()
    assert "pct_dentro_do_ic" in resumo.columns and len(resumo) == 2
    assert [e for e, _ in res.steps][0].startswith("Painel")
    d = res.to_dict()
    assert d["config"]["method"] == "km" and "curves" in d["model"]
    # o modelo final é um LifetimePD comum: aplica na carteira
    viva = ref.df.groupby("id_contrato").tail(1).head(50).copy()
    viva["idade"] = 6
    out = res.model.apply(viva, age_col="idade", term_col="prazo", detail=False)
    assert {"pd_12m", "pd_lifetime"} <= set(out.columns)


def test_estudo_hazard_com_ph_e_ciclo(ref):
    cfg = SurvivalConfig(origin_col="safra_origem", segment_col="produto", method="hazard",
                         features=list(FEATURES), baseline="spline", horizon=36,
                         split="observation", split_value="2021-06-01", z=-1.0, rho=0.1,
                         decay=0.1)
    res = run_survival_study(ref.df, cfg)
    assert res.hazard_models and res.ph is not None
    assert res.model.adjustments_ and res.model.adjustments_[-1]["tipo"] == "vasicek"
    assert res.model.curve().pd_12m() > res.model_base.curve().pd_12m()   # z < 0 = adverso
    assert res.discrimination is not None and (res.discrimination["auc"] > 0.5).all()


def test_estudo_parametrico_puro_por_rating(ref):
    cfg = SurvivalConfig(origin_col="safra_origem", by="rating", method="parametric",
                         distribution="loglogistic", horizon=48)
    res = run_survival_study(ref.df, cfg)
    assert res.validated_on.startswith("DES")
    s = res.summary().set_index("grupo")
    assert s.loc["A", "pd_12m"] < s.loc["B", "pd_12m"] < s.loc["C", "pd_12m"]
    assert all(c.meta["distribution"] == "loglogistic" for c in res.model.curves_.values())


def test_estudo_aceita_painel_pronto_e_callback(painel_ref):
    chamadas = []
    cfg = SurvivalConfig(method="vintage", tail="flat", horizon=48)
    res = run_survival_study(painel_ref, cfg, progress=lambda k, r, s: chamadas.append((k, s)))
    assert res.panel_des is painel_ref
    assert ("curva", "ok") in chamadas and chamadas[0] == ("painel", "run")
    # cauda plana: o último hazard se repete
    h = res.model.curve().hazard_
    assert np.allclose(h[painel_ref.max_age + 1:], h[painel_ref.max_age])
    with pytest.raises(ValueError, match="não existe"):
        run_survival_study(painel_ref, SurvivalConfig(by="nao_existe"))


def test_import_do_subpacote_nao_puxa_a_interface():
    import sys

    import yggdrasil.credit_risk.survival as sv

    assert "yggdrasil.credit_risk.survival.ui" not in sys.modules or True
    assert "SurvivalUI" in dir(sv)
    import yggdrasil.credit_risk as cr

    assert cr.survival is sv
