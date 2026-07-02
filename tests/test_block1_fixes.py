"""
Regressões dos bugs do "bloco 1" (relatório de melhorias) — TreeSegmenter,
ModelSegmenter e o helper de rótulo de safra.

Cada teste referencia o item do relatório que valida (1.x).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.tree import TreeSegmenter


# ----------------------------------------------------------------------
# Geradores de dados
# ----------------------------------------------------------------------
def _tree_df(task="classification", n=4000, seed=0, com_na=False):
    rng = np.random.default_rng(seed)
    x = rng.beta(2.5, 3, n) * 1.4 + 0.3
    gar = rng.choice(["A", "B", "C", "D"], n, p=[0.5, 0.22, 0.18, 0.1]).astype(object)
    if com_na:
        x[rng.random(n) < 0.08] = np.nan
        gar[rng.random(n) < 0.06] = np.nan
    lg = {"A": 0.0, "B": 0.10, "C": 0.16, "D": 0.30}
    risco = 0.1 + 0.4 * np.nan_to_num(x - 0.5, nan=0.35) + np.array([lg.get(g, 0.2) for g in gar])
    target = ((rng.uniform(0, 1, n) < np.clip(risco, 0.01, 0.95)).astype(float)
              if task == "classification" else np.clip(risco + rng.normal(0, 0.07, n), 0, 1))
    return pd.DataFrame({"score": x, "garantia": gar, "target": target, "amostra": "DES"})


def _tree(task="classification", **kw):
    df = kw.pop("df", None)
    if df is None:
        df = _tree_df(task, **{k: kw.pop(k) for k in ("n", "seed", "com_na") if k in kw})
    return TreeSegmenter(df, target="target", task_type=task,
                         sample_col="amostra", ref_sample="DES", verbose=False, **kw)


def _model_df(n=2500, seed=0, datas_ruins=False):
    rng = np.random.default_rng(seed)
    score = rng.beta(2.5, 3, n) * 1.4 + 0.3
    renda = rng.gamma(2.0, 1500, n)
    gar = rng.choice(["A", "B", "C", "D"], n, p=[0.5, 0.22, 0.18, 0.1]).astype(object)
    lg = {"A": 0.0, "B": 0.10, "C": 0.16, "D": 0.30}
    risco = 0.1 + 0.35 * (score - 0.5) + 2e-5 * renda + np.array([lg[g] for g in gar])
    target = (rng.uniform(0, 1, n) < np.clip(risco, 0.02, 0.95)).astype(float)
    meses = [f"2023-{m:02d}" for m in range(1, 9)]
    dt = rng.choice(meses, size=n).astype(object)          # object p/ aceitar datas inválidas
    if datas_ruins:
        dt[:12] = "data-invalida"
    amostra = np.where(np.isin(dt, meses[6:]), "OOT", "DES")
    return pd.DataFrame({"score": score, "renda": renda, "garantia": gar,
                         "target": target, "dt_ref": dt, "amostra": amostra})


def _model(**kw):
    pytest.importorskip("optbinning")
    from yggdrasil.credit_risk.model import ModelSegmenter
    df = kw.pop("df", None)
    if df is None:
        df = _model_df()
    return ModelSegmenter(df, target="target", task_type="classification",
                          sample_col="amostra", ref_sample="DES", date_col="dt_ref",
                          verbose=False, **kw)


# ======================================================================
# 1.10 — fmt_month_year: mês fora de 1–12 volta como string bruta
# ======================================================================
def test_fmt_month_year_mes_invalido():
    from yggdrasil.reporting.style import fmt_month_year
    assert fmt_month_year(["2022-01"]) == ["jan/22"]
    assert fmt_month_year(["2022-00"]) == ["2022-00"]     # antes virava 'dez/22'
    assert fmt_month_year(["2022-13"]) == ["2022-13"]


# ======================================================================
# 1.9 — _assign_na_to_worst: rota de faltante na irmã de PIOR risco
# ======================================================================
def test_assign_na_to_worst_marca_include_na_na_pior():
    seg = _tree("classification", com_na=False)              # sem missing no ajuste → sem nó na
    seg.grow("score", splits=[0.6, 0.9], dtype="num")
    folhas = [s for s in seg.segments.values() if s["is_leaf"]]
    com_na = [f for f in folhas if f["conditions"][-1].get("include_na")]
    assert len(com_na) == 1                                  # exatamente uma rota de faltante
    pior = max(folhas, key=lambda f: seg._node_value(
        next(k for k, v in seg.segments.items() if v is f)))
    assert com_na[0] is pior                                 # e é a folha de maior risco


# ======================================================================
# 1.3 — predict (régua por condições, = SQL/Spark) reproduz as máscaras,
#        e a cobertura é total no treino (na_to_worst padrão)
# ======================================================================
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_predict_equivale_as_mascaras_e_cobre_treino(task):
    seg = _tree(task, com_na=True)
    seg.grow("score", splits=[0.8], dtype="num")
    seg.grow("garantia", splits=[["A", "B"], ["C", "D"]])
    out = seg.predict(seg.df)
    seg_por_folha = {}
    for sid, s in seg.segments.items():
        if s["is_leaf"]:
            vals = out.loc[s["mask"].to_numpy(), "segmento"].unique()
            assert len(vals) == 1                            # máscara → 1 segmento só
            seg_por_folha[sid] = vals[0]
    assert len(set(seg_por_folha.values())) == len(seg_por_folha)   # bijeção folha↔segmento
    assert out["segmento"].notna().all()                     # cobertura total no treino


# ======================================================================
# 1.2 — scoring sem rota (na_to_worst=False + faltante) AVISA, não some calado
# ======================================================================
def test_predict_avisa_linhas_sem_segmento():
    seg = _tree("classification", com_na=False)
    seg.grow("garantia", splits=[["A"], ["B", "C"], ["D"]], na_to_worst=False)
    df_novo = seg.df.copy()
    df_novo.loc[df_novo.index[:8], "garantia"] = np.nan      # faltante só no scoring
    with pytest.warns(UserWarning, match="sem segmento"):
        out = seg.predict(df_novo)
    assert out["segmento"].isna().sum() >= 8


# ======================================================================
# ModelSegmenter
# ======================================================================
# 1.11 — _parse_bin_spec: corte numérico inválido → ValueError claro
def test_parse_bin_spec_corte_invalido():
    seg = _model()
    with pytest.raises(ValueError, match="corte inválido"):
        seg._parse_bin_spec("score", "0.7, abc")


# 1.5 — _compute_score valida colunas ausentes com mensagem clara
def test_compute_score_colunas_ausentes():
    from sklearn.linear_model import LogisticRegression
    seg = _model()
    m = LogisticRegression(max_iter=200).fit(seg.df[["score", "renda"]], seg.df["target"])
    seg.set_model(m, features=["score", "renda"])
    with pytest.raises(KeyError, match="ausentes"):
        seg._compute_score(seg.df.drop(columns=["renda"]))


# 1.6 — set_model reseta feature_transform p/ "raw" (evita WoE residual)
def test_set_model_reseta_feature_transform():
    from sklearn.linear_model import LogisticRegression
    seg = _model()
    seg.fit("logistica", transform="woe", features=["score", "renda", "garantia"])
    assert seg.feature_transform == "woe"
    m = LogisticRegression(max_iter=200).fit(seg.df[["score"]], seg.df["target"])
    seg.set_model(m, features=["score"])
    assert seg.feature_transform == "raw"


# 1.8 — variable_by_safra ignora datas inválidas (NaT), sem linha "NaT"
def test_variable_by_safra_ignora_nat():
    seg = _model(df=_model_df(datas_ruins=True))
    out = seg.variable_by_safra("score", time_col="dt_ref")
    assert "NaT" not in set(out["safra"].astype(str))


# 1.7 — plot_calibration não quebra com score constante (classificação)
def test_plot_calibration_score_constante():
    import matplotlib
    matplotlib.use("Agg")
    from sklearn.dummy import DummyClassifier
    seg = _model()
    d = DummyClassifier(strategy="prior").fit(seg.df[["score"]], seg.df["target"])
    seg.set_model(d, features=["score"])
    assert seg.score_.nunique() == 1                         # score de fato constante
    fig = seg.plot_calibration()
    assert fig is not None


# 1.4 — metric_shifts robusto (não estoura em np.isfinite) e devolve dict
def test_metric_shifts_smoke():
    seg = _model()
    seg.fit("logistica", features=["score", "renda", "garantia"])
    ms = seg.metric_shifts()
    assert isinstance(ms, dict)
