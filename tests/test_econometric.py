"""
Testes do subpacote de modelos econométricos (satélite) de PD/LGD/CCF.

A validação central segue o guia (§7): dados sintéticos de **processo gerador
conhecido** permitem testar se cada método **recupera a verdade**. Como a
recuperação de coeficientes só é limpa sob **boa identificação** (drivers macro
muito persistentes são quase colineares com a defasagem da dependente — a
colinearidade que o próprio guia alerta, §4.1), os testes de recuperação usam
DGPs identificáveis: coeficiente macro num DGP **sem AR**; termo AR num DGP
**sem macro**; e o modelo completo é validado por **acurácia fora da amostra** e
**sinais corretos**, não pelo valor exato do coeficiente em série curta.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")
pytest.importorskip("arch")

warnings.filterwarnings("ignore")

from yggdrasil.credit_risk.econometric import (  # noqa: E402
    ARDL,
    ARIMA,
    BetaRegression,
    FractionalLogit,
    HistoricalMean,
    PanelSatellite,
    RandomWalk,
    Scenario,
    ScenarioSet,
    Specification,
    VARModel,
    VasicekZ,
    diebold_mariano,
    ecl_projection,
    engle_granger,
    johansen_test,
    make_grid,
    make_reference_study,
    run_study,
    search,
    shock_scenarios,
    standard_scenarios,
    walk_forward,
)
from yggdrasil.credit_risk.econometric import diagnostics as diag  # noqa: E402
from yggdrasil.credit_risk.econometric import series as S  # noqa: E402
from yggdrasil.credit_risk.econometric import transforms as tf  # noqa: E402
from yggdrasil.credit_risk.econometric.config import StudyConfig  # noqa: E402


# ======================================================================
# transforms
# ======================================================================
def test_link_round_trips():
    p = pd.Series([0.01, 0.05, 0.2, 0.5, 0.9],
                  index=pd.date_range("2020-01-01", periods=5, freq="MS"))
    assert np.allclose(tf.inv_logit(tf.logit(p)).to_numpy(), p.to_numpy(), atol=1e-9)
    assert np.allclose(tf.inv_probit(tf.probit(p)).to_numpy(), p.to_numpy(), atol=1e-9)
    assert isinstance(tf.logit(p), pd.Series)  # índice preservado


def test_vasicek_roundtrip_and_capital_consistency():
    from scipy.stats import norm

    dr = pd.Series([0.02, 0.03, 0.05, 0.08, 0.04])
    z = tf.vasicek_z(dr, pd_ttc=0.045, rho=0.12)
    assert np.allclose(tf.default_rate_from_z(z, 0.045, 0.12).to_numpy(), dr.to_numpy(), atol=1e-9)
    assert z.iloc[3] < z.iloc[0]  # taxa maior -> fator sistêmico menor (benigno alto)
    # mesma estrutura de Vasicek do motor de capital (conditional_pd)
    q = 0.999
    cap = norm.cdf((norm.ppf(0.045) + np.sqrt(0.12) * norm.ppf(q)) / np.sqrt(1 - 0.12))
    assert np.isclose(tf.default_rate_from_z(-norm.ppf(q), 0.045, 0.12), cap)


def test_make_lags_align_and_dummies():
    x = pd.Series(np.arange(12.0), index=pd.date_range("2020-01-01", periods=12, freq="MS"),
                  name="desemprego")
    L = tf.make_lags(x, {"desemprego": [0, 1, 3]})
    assert list(L.columns) == ["desemprego", "desemprego_l1", "desemprego_l3"]
    y, X = tf.align(x, L)
    assert not X.isna().any().any() and len(X) == 9
    ev = tf.event_dummies(x.index, {"covid": ("2020-03", "2020-06")})
    assert ev["covid"].sum() == 4
    sd = tf.seasonal_dummies(x.index, period=12)
    assert sd.shape[1] == 11  # drop_first


# ======================================================================
# series / gerador sintético
# ======================================================================
def test_synthetic_series_valid_and_truth():
    syn = S.simulate_pd_series(seed=1)
    assert isinstance(syn.series, S.RiskSeries) and syn.series.kind == "pd"
    v = syn.series.values.to_numpy()
    assert np.all((v >= 0) & (v <= 1))
    assert ("desemprego", 1) in syn.betas()
    lg = S.simulate_lgd_series(seed=2)
    assert lg.series.kind == "lgd"
    cc = S.simulate_ccf_series(seed=3)
    assert cc.series.kind == "ccf"


def test_reference_study_shares_macro():
    est = make_reference_study(n_periods=96)
    assert len(est.macro) == 96
    assert est.pd.series.kind == "pd" and est.lgd.series.kind == "lgd" and est.ccf.series.kind == "ccf"
    assert est.pd.macro is est.macro


def test_riskseries_validation():
    idx = pd.date_range("2020-01-01", periods=5, freq="MS")
    with pytest.raises(ValueError):
        S.RiskSeries(pd.Series([0.1, 1.5, 0.2, 0.3, 0.4], index=idx), kind="pd")  # >1
    with pytest.raises(ValueError):
        S.RiskSeries(pd.Series([0.1] * 5, index=idx), kind="xyz")  # kind inválido


# ======================================================================
# diagnostics
# ======================================================================
def test_stationarity_battery():
    rng = np.random.default_rng(0)
    wn = rng.normal(size=200)
    rw = np.cumsum(rng.normal(size=200))
    assert diag.adf(wn).passed and not diag.adf(rw).passed
    assert diag.kpss_test(wn).passed and not diag.kpss_test(rw).passed
    assert diag.integration_order(rw) == 1


def test_residual_autocorr_and_vif():
    rng = np.random.default_rng(0)
    wn = rng.normal(size=400)
    rng2 = np.random.default_rng(1)
    u = np.empty(400); u[0] = 0.0
    for t in range(1, 400):
        u[t] = 0.7 * u[t - 1] + rng2.normal()
    assert diag.ljung_box(wn).passed and not diag.ljung_box(u).passed
    a = rng.normal(size=200)
    X = pd.DataFrame({"a": a, "b": 2 * a + rng.normal(0, 0.01, 200), "c": rng.normal(size=200)})
    assert diag.max_vif(X) > 10  # a,b colineares


def test_structural_break_detection():
    rng = np.random.default_rng(2)
    y = np.r_[rng.normal(0, 1, 100), rng.normal(3, 1, 100)]
    X = pd.DataFrame({"t": np.arange(200.0)})
    assert not diag.chow_test(y, X, 100).passed
    qa = diag.quandt_andrews(y, X)
    assert not qa.passed and 80 <= qa.extra["break_index"] <= 120


def test_residual_report_columns():
    rng = np.random.default_rng(3)
    rep = diag.residual_report(rng.normal(size=120), exog=pd.DataFrame({"x": rng.normal(size=120)}))
    assert {"teste", "estatistica", "p_valor", "ok"} <= set(rep.columns)
    assert "Ljung-Box" in rep["teste"].values and "Jarque-Bera" in rep["teste"].values


# ======================================================================
# ARDL — recuperação e projeção
# ======================================================================
def test_ardl_macro_coefficient_recovery_clean():
    # DGP sem AR (identificação limpa do coef macro), ruído de observação baixo
    lg = S.simulate_lgd_series(seed=30, ar=(), n_periods=180,
                               betas=(("desemprego", 0, 0.12), ("renda", 1, -0.05)),
                               precision=6000)
    spec = Specification(exog={"desemprego": [0], "renda": [1]}, ar=0, link="logit")
    fr = ARDL(lg.series, lg.macro, spec).fit()
    assert abs(fr.params["desemprego"] - 0.12) < 0.05
    assert abs(fr.params["renda_l1"] - (-0.05)) < 0.04


def test_ardl_ar_recovery_clean():
    pdp = S.simulate_pd_series(seed=31, ar=(0.6,), betas=(), n_obligors=200_000, n_periods=200)
    fr = ARDL(pdp.series, None, Specification(exog={}, ar=1, link="logit")).fit()
    assert abs(fr.params["y_l1"] - 0.6) < 0.1


def test_ardl_beats_naive_out_of_sample():
    syn = S.simulate_pd_series(seed=40)
    spec = Specification(exog={"desemprego": [1], "renda": [0]}, ar=1, link="logit")
    wf_ardl = walk_forward(lambda s, m: ARDL(s, m, spec), syn.series, syn.macro,
                           min_train=72, horizon=6)
    wf_rw = walk_forward(lambda s, m: RandomWalk(s), syn.series, syn.macro,
                         min_train=72, horizon=6)
    assert wf_ardl["rmse"] < wf_rw["rmse"]
    dm = diebold_mariano(wf_ardl["errors"], wf_rw["errors"], h=6)
    assert dm["better"] == "a" and dm["pvalue"] < 0.05


def test_ardl_clean_dgp_residuals_pass():
    # DGP casando com o modelo (AR + macro), sem eventos -> resíduos limpos
    syn = S.simulate_pd_series(seed=11)
    spec = Specification(exog={"desemprego": [1], "renda": [0]}, ar=1, link="logit")
    fr = ARDL(syn.series, syn.macro, spec).fit()
    rep = fr.diagnostics().set_index("teste")
    assert bool(rep.loc["Ljung-Box", "ok"])
    assert bool(rep.loc["Jarque-Bera", "ok"])


def test_ardl_projection_interval_coverage():
    syn = S.simulate_pd_series(seed=41)
    tr, te = syn.series.values.index[:-12], syn.series.values.index[-12:]
    spec = Specification(exog={"desemprego": [1], "renda": [0]}, ar=1, link="logit")
    m = ARDL(S.RiskSeries(syn.series.values.loc[tr], kind="pd"), syn.macro.loc[tr], spec)
    m.fit()
    proj = m.project({"real": syn.macro.loc[te]}, n_sims=1500, seed=1, alpha=0.10)
    pf = proj.paths["real"]
    actual = syn.series.values.loc[te].to_numpy()
    cover = ((actual >= pf["lower"].to_numpy()) & (actual <= pf["upper"].to_numpy())).mean()
    assert cover >= 0.6  # cobertura próxima do nominal 90% (amostra pequena)
    assert (pf["upper"] >= pf["lower"]).all()


# ======================================================================
# ARIMA / Vasicek / Beta / Fractional
# ======================================================================
def test_arima_and_arimax():
    syn = S.simulate_pd_series(seed=21)
    tr = syn.series.values.index[:-12]
    m = ARIMA(S.RiskSeries(syn.series.values.loc[tr], kind="pd"), order=(1, 0, 0))
    fr = m.fit()
    assert np.isfinite(fr.aic)
    fc = m.predict(syn.macro.loc[syn.series.values.index[-12:]])
    assert len(fc) == 12 and (fc >= 0).all()
    mx = ARIMA(S.RiskSeries(syn.series.values.loc[tr], kind="pd"), syn.macro.loc[tr],
               exog={"desemprego": [1]}, order=(1, 0, 0))
    frx = mx.fit()
    assert "desemprego_l1" in frx.params.index


def test_vasicek_recovery_and_projection():
    synv = S.simulate_pd_series(link="vasicek", seed=22, ar=(0.5,),
                                betas=(("desemprego", 1, -0.03),), pd_ttc=0.04, rho=0.12, sigma=0.2,
                                n_periods=180)
    assert 0.01 < synv.series.values.mean() < 0.12  # taxa realista (Z centrado)
    m = VasicekZ(synv.series, synv.macro,
                 Specification(exog={"desemprego": [1]}, ar=1), rho=0.12, pd_ttc=0.04)
    fr = m.fit()
    assert np.sign(fr.params["desemprego_l1"]) == -1  # sinal correto
    from yggdrasil.credit_risk.econometric.scenarios import extend_macro

    future = extend_macro(synv.macro, 12)
    proj = m.project({"c": future}, n_sims=500)
    assert (proj.paths["c"]["mean"] > 0).all() and len(proj.paths["c"]) == 12


def test_beta_fractional_agree_and_recover():
    lg = S.simulate_lgd_series(seed=30, ar=(), n_periods=180,
                               betas=(("desemprego", 0, 0.12),), precision=6000)
    spec = Specification(exog={"desemprego": [0]}, ar=0, link="logit")
    frb = BetaRegression(lg.series, lg.macro, spec).fit()
    frf = FractionalLogit(lg.series, lg.macro, spec).fit()
    fra = ARDL(lg.series, lg.macro, spec).fit()
    # três motores concordam entre si e recuperam o sinal/ordem de grandeza
    assert abs(frb.params["desemprego"] - frf.params["desemprego"]) < 0.03
    assert abs(frb.params["desemprego"] - fra.params["desemprego"]) < 0.05
    assert abs(frb.params["desemprego"] - 0.12) < 0.06


def test_fractional_handles_zero_one():
    lg = S.simulate_lgd_series(seed=23)
    y = lg.series.values.copy()
    y.iloc[0] = 0.0
    y.iloc[1] = 1.0
    m = FractionalLogit(S.RiskSeries(y, kind="lgd"), lg.macro,
                        Specification(exog={"desemprego": [0]}, ar=1, link="logit"))
    fr = m.fit()
    assert np.isfinite(fr.aic)


def test_benchmarks_predict_project():
    syn = S.simulate_pd_series(seed=5)
    for M in (RandomWalk, HistoricalMean):
        m = M(syn.series)
        m.fit()
        assert len(m.predict(steps=6)) == 6
        pj = m.project({"x": syn.macro.iloc[-6:]})
        assert (pj.paths["x"]["upper"] >= pj.paths["x"]["lower"]).all()


# ======================================================================
# selection
# ======================================================================
def test_make_grid_and_sign_filter():
    grid = make_grid(["desemprego", "renda"], lag_set=(0, 1), max_vars=2,
                     expected_signs={"desemprego": 1, "renda": -1})
    assert len(grid) > 0
    syn = S.simulate_pd_series(seed=6)
    fr = ARDL(syn.series, syn.macro,
              Specification(exog={"desemprego": [1]}, ar=1, expected_signs={"desemprego": 1})).fit()
    ok, wrong = diag_sign(fr)
    assert ok and not wrong


def diag_sign(fr):
    from yggdrasil.credit_risk.econometric.selection import sign_ok

    return sign_ok(fr)


def test_search_selects_qualified_best():
    syn = S.simulate_pd_series(seed=40)
    res = search(syn.series, syn.macro, candidates=["desemprego", "renda", "juros"],
                 expected_signs={"desemprego": 1, "renda": -1, "juros": 1},
                 grid_kwargs={"lag_set": (0, 1), "max_vars": 2, "max_specs": 40},
                 horizon=6, min_train=72, vif_max=8.0, include_benchmarks=True)
    assert res.best is not None and res.best_spec is not None
    qual = res.ranking[res.ranking["status"] == "qualificado"]
    assert len(qual) > 0
    # o melhor supera o ARIMA fora da amostra
    assert float(qual["vs_arima"].iloc[0]) < 1.0


# ======================================================================
# scenarios
# ======================================================================
def test_scenario_shock_and_ecl():
    syn = S.simulate_pd_series(seed=50)
    spec = Specification(exog={"desemprego": [1], "renda": [0]}, ar=1, link="logit")
    m = ARDL(syn.series, syn.macro, spec)
    m.fit()
    base = __import__("yggdrasil.credit_risk.econometric.scenarios",
                      fromlist=["extend_macro"]).extend_macro(syn.macro, 12)
    ss = shock_scenarios(base, {"adverso": {"desemprego": 3.0}},
                         probabilities={"adverso": 0.3}, base_probability=0.7)
    proj = m.project(ss, n_sims=500)
    mf = proj.mean_frame()
    assert mf["adverso"].iloc[-1] > mf["base"].iloc[-1]
    ecl = ecl_projection(m, ss)
    assert len(ecl) == 12 and (ecl > 0).all()


def test_standard_scenarios_probabilities():
    syn = S.simulate_pd_series(seed=51)
    ss = standard_scenarios(syn.macro, horizon=12)
    probs = ss.probabilities()
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_scenarioset_rejects_duplicates():
    df = pd.DataFrame({"desemprego": [9.0]}, index=pd.date_range("2025-01-01", periods=1, freq="MS"))
    with pytest.raises(ValueError):
        ScenarioSet([Scenario("a", df), Scenario("a", df)])


# ======================================================================
# var/vecm/panel
# ======================================================================
def test_var_irf_and_cointegration():
    syn = S.simulate_pd_series(seed=50)
    vm = VARModel(syn.series, syn.macro, variables=["desemprego", "juros"], maxlags=3).fit()
    irf = vm.irf(8)
    assert irf.shape[0] == 9
    jt = johansen_test(vm.data)
    assert jt.rank >= 0
    eg = engle_granger(np.log(syn.series.values.clip(1e-4)), syn.macro["desemprego"])
    assert eg.pvalue is not None


def test_panel_disciplines_coefficients():
    macro = S.simulate_macro(seed=70)
    panels = {f"seg{i}": S.simulate_pd_series(
        macro=macro, seed=71 + i,
        betas=(("desemprego", 1, 0.18), ("renda", 0, -0.06))).series for i in range(6)}
    spec = Specification(exog={"desemprego": [1], "renda": [0]}, ar=1, link="logit")
    pr = PanelSatellite(panels, macro, spec).fit()
    # pooling recupera bem os coeficientes verdadeiros (poder estatístico, §3.7)
    assert abs(pr.params["desemprego_l1"] - 0.18) < 0.05
    assert abs(pr.params["renda"] - (-0.06)) < 0.03
    assert pr.n_segments == 6


# ======================================================================
# pipeline declarativo e relatório
# ======================================================================
def test_run_study_end_to_end():
    est = make_reference_study(n_periods=108)
    cfg = StudyConfig(kind="pd", name="pd_teste",
                      candidates=["desemprego", "renda", "juros"],
                      expected_signs={"desemprego": 1, "renda": -1, "juros": 1},
                      lag_set=(0, 1), max_vars=2, max_specs=30, horizon=12,
                      min_train=72, vif_max=8.0)
    r = run_study(cfg, est.pd.series, est.macro, make_report=True)
    assert r.best is not None
    assert r.projection.horizon == 12
    assert isinstance(r.report_html, str) and "Coeficientes" in r.report_html
    assert set(r.scenarios.names()) == {"base", "adverso", "otimista"}


def test_trend_ct_offset_is_full_length():
    # regressão: o offset da tendência 'ct' deve ser o comprimento CHEIO (N), não
    # o pós-align (N-ar), senão a projeção 'ct' fica descontínua por `ar` passos.
    syn = S.simulate_pd_series(seed=44)
    n = len(syn.series.values)
    m = ARDL(syn.series, syn.macro,
             Specification(exog={"desemprego": [0]}, ar=2, trend="ct", link="logit"))
    m.fit()
    assert m._trend_offset == n
    # o último valor de trend do design é N-1; o primeiro passo projetado usa N
    assert int(m.result.exog["trend"].iloc[-1]) == n - 1
    fb = FractionalLogit(S.simulate_lgd_series(seed=45).series, syn.macro,
                         Specification(exog={"desemprego": [0]}, ar=2, trend="ct"))
    fb.fit()
    assert fb._trend_offset == len(fb.series.values)


def test_models_do_not_mutate_caller_spec():
    # regressão: VasicekZ/FractionalLogit não podem corromper o spec do chamador
    syn = S.simulate_pd_series(seed=46)
    spec = Specification(exog={"desemprego": [1]}, ar=1, link="logit")
    VasicekZ(syn.series, syn.macro, spec, rho=0.12, pd_ttc=0.04).fit()
    assert spec.link == "logit"  # não virou 'vasicek'
    # o mesmo spec continua utilizável por outro modelo (champion-challenger)
    fr = ARDL(syn.series, syn.macro, spec).fit()
    assert fr.link == "logit"
    spec_p = Specification(exog={"desemprego": [0]}, ar=1, link="probit")
    FractionalLogit(S.simulate_lgd_series(seed=47).series, syn.macro, spec_p).fit()
    assert spec_p.link == "probit"  # FractionalLogit não sobrescreveu para 'logit'


def test_search_with_link_in_model_kwargs():
    # regressão: model_kwargs={'link': ...} não pode ser repassado ao construtor
    syn = S.simulate_pd_series(seed=40)
    res = search(syn.series, syn.macro, candidates=["desemprego", "renda"],
                 expected_signs={"desemprego": 1, "renda": -1},
                 model_kwargs={"link": "logit"},
                 grid_kwargs={"lag_set": (0, 1), "max_vars": 2, "max_specs": 20},
                 horizon=6, min_train=72, vif_max=8.0, include_benchmarks=False)
    assert res.best is not None and res.best_spec is not None
    assert (res.ranking["status"] == "qualificado").any()
    assert not res.ranking["status"].astype(str).str.startswith("erro").any()


def test_report_figures_render():
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.econometric import report

    syn = S.simulate_pd_series(seed=8)
    m = ARDL(syn.series, syn.macro,
             Specification(exog={"desemprego": [1]}, ar=1, link="logit"))
    fr = m.fit()
    fig = report.plot_fit(fr, syn.series)
    assert fig is not None
    html = report.model_report(fr, syn.series)
    assert html.startswith("<!doctype html>")
