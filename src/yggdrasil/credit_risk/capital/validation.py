"""
Validação, benchmark e testes do modelo de capital econômico
============================================================
Bloco F do guia. Capital econômico é um modelo de **alto impacto** e a
validação independente é obrigatória (Resolução CMN 4.557/2017). O backtesting
direto do quantil 99,9% é impossível por construção — valida-se o **corpo** da
distribuição (EL, quantis moderados), a **qualidade dos parâmetros** e a
**razoabilidade das correlações** por benchmark e estresse. Este módulo reúne:

* :func:`sensitivity` — sensibilidades univariadas (PD, LGD, ρ ±x%).
* :func:`correlation_stress` — estresse das correlações (correlações sobem em
  crise); mede a sensibilidade do capital.
* :func:`benchmark` — concilia ASRF, Monte Carlo e CreditRisk+.
* :func:`pillar1_comparison` — capital econômico × capital regulatório (Pilar 1).
* :func:`backtest_expected_loss` — realizado vs. previsto da EL por safra/segmento.
* :func:`convergence` — nº de cenários × estabilidade do VaR/ES no quantil-alvo.

Todos os motores são importados **sob demanda** (o módulo importa mesmo que um
motor irmão tenha problema pontual).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd


# ======================================================================
# Sensibilidades univariadas (PD, LGD, rho ±x%)
# ======================================================================
def sensitivity(
    portfolio,
    q: float = 0.999,
    shocks: Sequence[float] = (-0.20, -0.10, 0.10, 0.20),
    params: Sequence[str] = ("pd", "lgd", "rho"),
    engine: str = "asrf",
    n_scenarios: int = 100_000,
    seed: Optional[int] = 0,
    rho_default: float = 0.15,
) -> pd.DataFrame:
    """Choca multiplicativamente cada parâmetro e mede o capital econômico.

    ``engine="asrf"`` (padrão) usa o motor analítico — barato e determinístico,
    ideal para varrer sensibilidades. ``engine="mc"`` re-simula (mais caro).

    Retorna um ``DataFrame`` com ``parametro``, ``choque``, ``CE``, ``delta_CE``
    (variação vs. base) e ``elasticidade`` (``%ΔCE / %Δparâmetro``).
    """
    from .portfolio import Portfolio, Segment

    def _ec(port) -> float:
        if engine == "asrf":
            return port.asrf_capital(q=q, rho_default=rho_default).economic_capital
        if engine == "mc":
            return port.simulate(n_scenarios=n_scenarios, q=q, seed=seed,
                                 rho_default=rho_default).economic_capital(metric="var")
        raise ValueError("engine deve ser 'asrf' ou 'mc'.")

    base_ec = _ec(portfolio)
    linhas = [{"parametro": "base", "choque": 0.0, "CE": base_ec,
               "delta_CE": 0.0, "elasticidade": np.nan}]

    for p in params:
        if p not in ("pd", "lgd", "rho"):
            raise ValueError(f"parâmetro '{p}' não suportado (use pd/lgd/rho).")
        for sh in shocks:
            novos = []
            for s in portfolio.segments:
                atual = getattr(s, p)
                if p == "rho" and atual is None:
                    atual = rho_default
                novo = atual * (1.0 + sh)
                if p in ("pd", "lgd"):
                    novo = float(np.clip(novo, 0.0, 1.0))
                else:  # rho
                    novo = float(np.clip(novo, 0.0, 0.999))
                novos.append(s.with_params(**{p: novo}))
            port2 = Portfolio(novos, factor_corr=portfolio.factor_corr,
                              factor_names=list(portfolio.factor_names), name=portfolio.name)
            ec = _ec(port2)
            elast = ((ec / base_ec - 1.0) / sh) if (sh != 0 and base_ec != 0) else np.nan
            linhas.append({"parametro": p, "choque": sh, "CE": ec,
                           "delta_CE": ec - base_ec, "elasticidade": elast})
    return pd.DataFrame(linhas)


# ======================================================================
# Estresse de correlações (correlações sobem em crise)
# ======================================================================
def correlation_stress(
    portfolio,
    deltas: Sequence[float] = (0.25, 0.50),
    n_scenarios: int = 100_000,
    q: float = 0.999,
    seed: Optional[int] = 0,
    stress_asset_rho: bool = False,
) -> pd.DataFrame:
    """Escala as correlações **entre fatores** por ``(1+δ)`` e re-simula.

    Correlações sobem em crise, engordando a cauda. Opcionalmente também escala a
    correlação de ativos intra-segmento (``stress_asset_rho=True``). A matriz
    estressada é projetada para a correlação positiva-definida mais próxima.

    Retorna ``DataFrame`` com ``delta``, ``CE``, ``VaR``, ``ES`` e a variação %
    do capital vs. base.
    """
    from .portfolio import Portfolio

    def _run(port) -> "tuple":
        sim = port.simulate(n_scenarios=n_scenarios, q=q, seed=seed)
        return sim.economic_capital(metric="var"), sim.var(), sim.es()

    base_ce, base_var, base_es = _run(portfolio)
    linhas = [{"delta": 0.0, "CE": base_ce, "VaR": base_var, "ES": base_es, "var_CE_pct": 0.0}]

    C0 = np.asarray(portfolio.factor_corr, dtype=float)
    off = ~np.eye(C0.shape[0], dtype=bool)
    for d in deltas:
        C = C0.copy()
        C[off] = np.clip(C0[off] * (1.0 + d), -0.999, 0.999)
        C = (C + C.T) / 2.0
        np.fill_diagonal(C, 1.0)
        try:
            from .correlation import nearest_correlation
            C = nearest_correlation(C)
        except Exception:                                   # pragma: no cover
            vals, vecs = np.linalg.eigh(C)
            C = vecs @ np.diag(np.clip(vals, 1e-8, None)) @ vecs.T
            dd = np.sqrt(np.clip(np.diag(C), 1e-12, None))
            C = C / np.outer(dd, dd)
            np.fill_diagonal(C, 1.0)

        segs = portfolio.segments
        if stress_asset_rho:
            segs = [s.with_params(rho=None if s.rho is None
                                  else float(np.clip(s.rho * (1.0 + d), 0.0, 0.999)))
                    for s in segs]
        port2 = Portfolio(segs, factor_corr=C, factor_names=list(portfolio.factor_names),
                          name=portfolio.name)
        ce, var, es = _run(port2)
        linhas.append({"delta": d, "CE": ce, "VaR": var, "ES": es,
                       "var_CE_pct": (ce / base_ce - 1.0) if base_ce else np.nan})
    return pd.DataFrame(linhas)


# ======================================================================
# Benchmark: ASRF × Monte Carlo × CreditRisk+
# ======================================================================
def benchmark(
    portfolio,
    q: float = 0.999,
    n_scenarios: int = 200_000,
    seed: Optional[int] = 0,
    rho_default: float = 0.15,
    sigma_crp: float = 0.5,
) -> pd.DataFrame:
    """Concilia os três motores num quadro único (EL, VaR, ES, CE).

    O ASRF é aditivo (sem diversificação) e serve de teto/benchmark; o Monte
    Carlo multifatorial captura a diversificação; o CreditRisk+ é o desafio
    atuarial independente. Diferenças entre eles devem ser **explicadas**.
    """
    linhas = []

    # ASRF analítico
    a = portfolio.asrf_capital(q=q, rho_default=rho_default)
    linhas.append({"metodo": "ASRF/Vasicek", "EL": a.expected_loss, "VaR": a.value_at_risk,
                   "ES": np.nan, "CE": a.economic_capital})

    # Monte Carlo multifatorial
    sim = portfolio.simulate(n_scenarios=n_scenarios, q=q, seed=seed, rho_default=rho_default)
    d = sim.distribution()
    linhas.append({"metodo": "Monte Carlo", "EL": d.el, "VaR": d.var(q), "ES": d.es(q),
                   "CE": d.economic_capital(q, metric="var")})

    # CreditRisk+ (benchmark atuarial) — best-effort
    try:
        from .creditrisk_plus import creditrisk_plus
        crp = creditrisk_plus(portfolio, sigma=sigma_crp)
        linhas.append({"metodo": "CreditRisk+", "EL": crp.el, "VaR": crp.var(q),
                       "ES": crp.es(q), "CE": crp.economic_capital(q, metric="var")})
    except Exception as exc:                                 # pragma: no cover
        linhas.append({"metodo": "CreditRisk+", "EL": np.nan, "VaR": np.nan,
                       "ES": np.nan, "CE": np.nan, "erro": str(exc)[:120]})

    return pd.DataFrame(linhas)


# ======================================================================
# Comparação com o capital regulatório (Pilar 1 / IRB)
# ======================================================================
def pillar1_comparison(
    portfolio,
    exposure_class: Union[str, Dict[str, str]],
    q: float = 0.999,
    n_scenarios: int = 200_000,
    seed: Optional[int] = 0,
    engine: str = "mc",
) -> pd.DataFrame:
    """Capital econômico × capital regulatório de Pilar 1 (IRB) por segmento.

    ``exposure_class`` é uma classe única (str) ou um mapa ``{segmento: classe}``
    entre ``revolving``/``other_retail``/``mortgage``/``corporate``.
    """
    from .parameters import basel_capital_portfolio

    reg = basel_capital_portfolio(portfolio, exposure_class)   # DataFrame por segmento

    if engine == "asrf":
        econ = portfolio.asrf_capital(q=q).economic_capital
    else:
        econ = portfolio.simulate(n_scenarios=n_scenarios, q=q, seed=seed).economic_capital(metric="var")

    reg_total = float(reg["capital"].sum()) if "capital" in reg else np.nan
    return pd.DataFrame([{
        "capital_economico": econ,
        "capital_regulatorio_pilar1": reg_total,
        "razao_econ_reg": econ / reg_total if reg_total else np.nan,
    }])


# ======================================================================
# Backtesting da perda esperada (corpo da distribuição)
# ======================================================================
def backtest_expected_loss(
    realized: pd.DataFrame,
    *,
    loss_col: str = "perda_realizada",
    predicted_col: str = "EL_prevista",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compara a perda **realizada** com a EL **prevista** por safra/segmento.

    O quantil de cauda não é backtestável; o **corpo** (a EL) é. Para cada grupo
    (ou no agregado) devolve realizado, previsto, erro, erro % e um ``z`` simples
    (erro padronizado pelo desvio das perdas realizadas do grupo), útil como
    sinalizador — não um teste formal de cauda.
    """
    if group_col is None:
        grupos = [("total", realized)]
    else:
        grupos = list(realized.groupby(group_col))

    linhas = []
    for nome, g in grupos:
        obs = float(g[loss_col].sum())
        prev = float(g[predicted_col].sum())
        n = len(g)
        sd = float(g[loss_col].std(ddof=1)) if n > 1 else np.nan
        z = ((g[loss_col].mean() - g[predicted_col].mean()) / (sd / np.sqrt(n))
             if (sd and sd > 0 and n > 1) else np.nan)
        linhas.append({
            "grupo": nome, "n": n, "realizado": obs, "previsto": prev,
            "erro": obs - prev, "erro_pct": (obs / prev - 1.0) if prev else np.nan,
            "z": z,
        })
    return pd.DataFrame(linhas)


# ======================================================================
# Convergência do Monte Carlo (nº de cenários × estabilidade do quantil)
# ======================================================================
def convergence(
    portfolio,
    n_grid: Sequence[int] = (1_000, 5_000, 10_000, 50_000, 100_000, 250_000),
    q: float = 0.999,
    seed: Optional[int] = 0,
    n_repeats: int = 1,
) -> pd.DataFrame:
    """Estabilidade do VaR/ES no quantil-alvo conforme cresce o nº de cenários.

    Com ``n_repeats > 1`` cada tamanho é rodado com sementes diferentes e o
    desvio-padrão do VaR entre repetições é reportado (``VaR_sd``) — a medida
    direta do ruído de Monte Carlo na cauda.
    """
    linhas = []
    for n in n_grid:
        vars_, ess_ = [], []
        for r in range(max(n_repeats, 1)):
            s = None if seed is None else seed + r
            sim = portfolio.simulate(n_scenarios=int(n), q=q, seed=s,
                                     store_segment_losses=False)
            vars_.append(sim.var())
            ess_.append(sim.es())
        linhas.append({
            "n_cenarios": int(n),
            "VaR": float(np.mean(vars_)),
            "ES": float(np.mean(ess_)),
            "EL": portfolio.expected_loss(),
            "VaR_sd": float(np.std(vars_, ddof=1)) if len(vars_) > 1 else np.nan,
        })
    return pd.DataFrame(linhas)


__all__ = [
    "sensitivity",
    "correlation_stress",
    "benchmark",
    "pillar1_comparison",
    "backtest_expected_loss",
    "convergence",
]
