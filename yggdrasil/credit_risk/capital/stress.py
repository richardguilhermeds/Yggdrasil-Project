"""
Estresse de cenário macro — o elo entre os satélites e o capital
================================================================
Fecha o circuito prometido pelos modelos satélite (:mod:`..econometric`): o
cenário macro projetado — em particular o fator sistêmico ``Z`` de Vasicek —
alimenta a carteira de capital econômico, e o capital é recalculado sob cada
cenário. O fluxo tem três passos:

1. **Cenário → choque de parâmetros.** Cada cenário vira multiplicadores (ou
   níveis absolutos) das PDs/LGDs dos segmentos. Aceita-se:

   * um ``dict`` simples ``{cenario: {pd_mult|pd_abs, lgd_mult|lgd_abs}}`` —
     totalmente desacoplado dos satélites (funciona sem o subpacote
     econometric);
   * uma :class:`~yggdrasil.credit_risk.econometric.base.Projection` (ou uma
     lista delas, ex.: uma de PD e outra de LGD) — a trajetória projetada de
     cada cenário é reduzida a um nível (média/último/máximo no horizonte) e
     normalizada pelo cenário-base, virando multiplicador;
   * um :class:`~yggdrasil.credit_risk.econometric.scenarios.ScenarioSet` mais
     ``model=`` (modelo satélite ajustado, ex.: ``VasicekZ``) — projeta o
     modelo sobre os cenários e segue o caminho anterior.

2. **Choque → carteira estressada.** :func:`apply_scenario` aplica o choque
   segmento a segmento via :meth:`Segment.with_params`, preservando fatores,
   correlações e granularidade.

3. **Carteira estressada → capital.** O motor escolhido (``asrf`` analítico ou
   ``monte_carlo``) recomputa EL/VaR/ES/CE; :func:`scenario_capital` devolve o
   quadro cenário × métricas com os deltas vs. a carteira-base.

Modo ``conditional_z`` — a ponte formal com o satélite
:class:`~yggdrasil.credit_risk.econometric.vasicek.VasicekZ`: em vez do
multiplicador direto, o choque **agregado** de PD é invertido para o ``Z``
implicado (``vasicek_z``) e o **mesmo** ``Z`` é propagado a cada segmento pela
sua própria correlação de ativos ``rho`` (``default_rate_from_z``) — segmentos
mais cíclicos estressam mais. As duas funções são importadas **sob demanda** do
econometric, para não criar dependência dura entre os subpacotes: o caminho do
``dict`` simples funciona sem ele.

Uso típico::

    from yggdrasil.credit_risk.capital import scenario_capital

    quadro = scenario_capital(
        carteira,
        {"adverso": {"pd_mult": 1.8, "lgd_mult": 1.1},
         "severo":  {"pd_mult": 2.5, "lgd_mult": 1.2}},
        engine="monte_carlo", n_scenarios=200_000, seed=42,
    )
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import numpy as np
import pandas as pd

from .measures import DEFAULT_CONFIDENCE
from .portfolio import Portfolio

#: Chaves de choque aceitas por cenário (multiplicador OU nível absoluto).
SHOCK_KEYS = frozenset({"pd_mult", "pd_abs", "lgd_mult", "lgd_abs"})


# ======================================================================
# Ponte com o econometric (import sob demanda — sem dependência dura)
# ======================================================================
def _vasicek_funcs():
    """Importa ``vasicek_z``/``default_rate_from_z`` do econometric, sob demanda."""
    try:
        from ..econometric.transforms import default_rate_from_z, vasicek_z
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise ImportError(
            "conditional_z=True usa o fator Z do subpacote econometric "
            "(yggdrasil.credit_risk.econometric), indisponível neste ambiente: "
            f"{exc}. Use conditional_z=False (estresse direto por multiplicador) "
            "ou instale as dependências dos modelos satélite (ex.: statsmodels)."
        ) from exc
    return vasicek_z, default_rate_from_z


# ======================================================================
# Validação e aplicação de um choque a uma carteira
# ======================================================================
def _validate_shock(nome: str, shock) -> None:
    if not isinstance(shock, Mapping):
        raise TypeError(
            f"[{nome}] o choque deve ser um dict com chaves em {sorted(SHOCK_KEYS)}; "
            f"recebido {type(shock).__name__}.")
    desconhecidas = set(shock) - SHOCK_KEYS
    if desconhecidas:
        raise ValueError(
            f"[{nome}] chaves de choque desconhecidas: {sorted(desconhecidas)}; "
            f"válidas: {sorted(SHOCK_KEYS)}.")
    if "pd_mult" in shock and "pd_abs" in shock:
        raise ValueError(f"[{nome}] use pd_mult OU pd_abs, não ambos.")
    if "lgd_mult" in shock and "lgd_abs" in shock:
        raise ValueError(f"[{nome}] use lgd_mult OU lgd_abs, não ambos.")
    for k in ("pd_mult", "lgd_mult"):
        if k in shock and not (float(shock[k]) >= 0.0):   # nega NaN também
            raise ValueError(f"[{nome}] {k} deve ser >= 0; recebido {shock[k]!r}.")
    for k in ("pd_abs", "lgd_abs"):
        if k in shock and not (0.0 <= float(shock[k]) <= 1.0):
            raise ValueError(f"[{nome}] {k} deve estar em [0, 1]; recebido {shock[k]!r}.")


def _stressed_params(portfolio: Portfolio, shock: Mapping) -> tuple:
    """Vetores (PD, LGD) por segmento após o choque direto (mult/abs, recortados)."""
    pds = portfolio.pds()
    lgds = portfolio.lgds()
    if "pd_abs" in shock:
        new_pd = np.full_like(pds, float(shock["pd_abs"]))
    elif "pd_mult" in shock:
        new_pd = np.clip(pds * float(shock["pd_mult"]), 0.0, 1.0)
    else:
        new_pd = pds
    if "lgd_abs" in shock:
        new_lgd = np.full_like(lgds, float(shock["lgd_abs"]))
    elif "lgd_mult" in shock:
        new_lgd = np.clip(lgds * float(shock["lgd_mult"]), 0.0, 1.0)
    else:
        new_lgd = lgds
    return new_pd, new_lgd


def _stress(
    portfolio: Portfolio,
    shock: Mapping,
    *,
    conditional_z: bool = False,
    rho_default: float = 0.15,
    z_rho: Optional[float] = None,
    name: Optional[str] = None,
    nome_cenario: str = "cenario",
) -> tuple:
    """Núcleo: carteira estressada + ``Z`` implicado (ou ``None``)."""
    _validate_shock(nome_cenario, shock)
    new_pd, new_lgd = _stressed_params(portfolio, shock)

    z: Optional[float] = None
    if conditional_z and ("pd_mult" in shock or "pd_abs" in shock):
        vasicek_z, default_rate_from_z = _vasicek_funcs()
        eads = portfolio.eads()
        rhos = portfolio.rhos(default=rho_default)
        pds = portfolio.pds()
        pd_base = float(np.average(pds, weights=eads))       # PD agregada da carteira
        pd_stress = float(np.average(new_pd, weights=eads))  # PD agregada estressada
        if not (0.0 < pd_base < 1.0):
            raise ValueError(
                "conditional_z exige PD agregada da carteira em (0, 1) para inverter "
                f"o fator Z; a carteira tem PD agregada {pd_base!r}.")
        rho_agg = float(z_rho) if z_rho is not None else float(np.average(rhos, weights=eads))
        if not (0.0 < rho_agg < 1.0):
            raise ValueError(
                f"a correlação usada na inversão do Z deve estar em (0, 1); obtido {rho_agg!r} "
                "(defina z_rho= ou rho nos segmentos).")
        # Z implicado pelo cenário: qual fator sistêmico produziria a taxa agregada
        # estressada, dado o nível de longo prazo da carteira (inversão de Vasicek).
        z = float(np.asarray(vasicek_z(np.array([pd_stress]), pd_ttc=pd_base, rho=rho_agg))[0])
        # O MESMO Z propagado a cada segmento pela sua própria carga rho: segmentos
        # mais cíclicos (rho maior) estressam mais. Segmento degenerado (PD 0/1 ou
        # rho fora de (0,1)) mantém o choque direto.
        cond = new_pd.copy()
        for i, (p, r) in enumerate(zip(pds, rhos)):
            if 0.0 < p < 1.0 and 0.0 < r < 1.0:
                cond[i] = float(np.asarray(default_rate_from_z(np.array([z]), pd_ttc=p, rho=r))[0])
        new_pd = np.clip(cond, 0.0, 1.0)

    segs = [
        s.with_params(pd=float(p), lgd=float(l))
        for s, p, l in zip(portfolio.segments, new_pd, new_lgd)
    ]
    port2 = Portfolio(
        segs,
        factor_corr=portfolio.factor_corr,
        factor_names=list(portfolio.factor_names),
        name=name or portfolio.name,
    )
    return port2, z


def apply_scenario(
    portfolio: Portfolio,
    shock: Mapping,
    *,
    conditional_z: bool = False,
    rho_default: float = 0.15,
    z_rho: Optional[float] = None,
    name: Optional[str] = None,
) -> Portfolio:
    """Constrói a **carteira estressada** por um choque de cenário.

    Parameters
    ----------
    portfolio:
        A carteira-base (:class:`~yggdrasil.credit_risk.capital.portfolio.Portfolio`).
    shock:
        Dict com chaves em ``{"pd_mult", "pd_abs", "lgd_mult", "lgd_abs"}`` —
        multiplicador ou nível absoluto por parâmetro (mult e abs do mesmo
        parâmetro são mutuamente exclusivos). Dict vazio = identidade.
    conditional_z:
        Se ``True``, o choque agregado de PD é invertido para o ``Z`` implicado
        (``vasicek_z``) e propagado a cada segmento pela sua própria ``rho``
        (``default_rate_from_z``) — a mesma estrutura do satélite ``VasicekZ``.
        Requer o subpacote econometric (import sob demanda).
    rho_default:
        ``rho`` assumida nos segmentos com ``rho=None`` (só no modo ``conditional_z``).
    z_rho:
        Correlação usada na **inversão agregada** do ``Z``. Padrão: média das
        ``rho`` dos segmentos ponderada por EAD.
    name:
        Nome da carteira estressada (padrão: o nome da carteira-base).

    Returns
    -------
    Portfolio
        Nova carteira (a original não é modificada), com os mesmos fatores e
        correlações.
    """
    port2, _ = _stress(
        portfolio, shock, conditional_z=conditional_z, rho_default=rho_default,
        z_rho=z_rho, name=name,
    )
    return port2


# ======================================================================
# Cenários → choques (dict simples, Projection, ScenarioSet + model)
# ======================================================================
def _level(path, horizon: Optional[int], reduce: str) -> float:
    """Reduz a trajetória projetada de um cenário a um nível escalar."""
    if isinstance(path, pd.DataFrame):
        if "mean" not in path.columns:
            raise ValueError(
                "cada trajetória da projeção deve ter a coluna 'mean' "
                "(contrato de Projection do econometric).")
        serie = path["mean"]
    else:
        serie = pd.Series(np.asarray(path, dtype=float).ravel())
    if horizon is not None:
        h = int(horizon)
        if h < 1:
            raise ValueError(f"horizon deve ser >= 1; recebido {horizon!r}.")
        serie = serie.iloc[:h]
    serie = serie.dropna()
    if serie.empty:
        raise ValueError("trajetória vazia na projeção (nada a reduzir).")
    if reduce == "mean":
        return float(serie.mean())
    if reduce == "last":
        return float(serie.iloc[-1])
    if reduce == "max":
        return float(serie.max())
    raise ValueError(f"reduce deve ser 'mean', 'last' ou 'max'; recebido {reduce!r}.")


def _shocks_from_projection(proj, *, horizon, reduce, base_scenario) -> dict:
    """``Projection`` → ``{cenario: {pd_mult|lgd_mult: nivel_s / nivel_base}}``."""
    kind = str(getattr(proj, "kind", "")).lower()
    chave = {"pd": "pd_mult", "lgd": "lgd_mult"}.get(kind)
    if chave is None:
        raise ValueError(
            f"projeção de kind {kind!r} não é suportada no estresse de capital "
            "(use 'pd' ou 'lgd'; para outros parâmetros, monte o dict simples "
            "{cenario: {pd_mult|pd_abs, lgd_mult|lgd_abs}}).")
    paths = proj.paths
    if base_scenario not in paths:
        raise ValueError(
            f"o cenário-base {base_scenario!r} não está na projeção "
            f"(cenários: {sorted(paths)}); ajuste base_scenario=.")
    nivel_base = _level(paths[base_scenario], horizon, reduce)
    if not (nivel_base > 0.0):
        raise ValueError(
            f"o nível do cenário-base deve ser > 0 para normalizar (obtido {nivel_base!r}).")
    return {
        str(nome): {chave: _level(df, horizon, reduce) / nivel_base}
        for nome, df in paths.items()
    }


def _coerce_scenarios(scenarios, *, model, horizon, reduce, base_scenario) -> dict:
    """Normaliza a entrada de cenários para ``{nome: choque}``."""
    # 1) dict simples nome → choque (desacoplado do econometric)
    if isinstance(scenarios, Mapping):
        return {str(nome): dict(sh) for nome, sh in scenarios.items()}

    # 2) uma Projection (duck-typing: paths + kind)
    if hasattr(scenarios, "paths") and hasattr(scenarios, "kind"):
        return _shocks_from_projection(
            scenarios, horizon=horizon, reduce=reduce, base_scenario=base_scenario)

    # 3) várias Projections (ex.: satélite de PD + satélite de LGD)
    if (isinstance(scenarios, (list, tuple)) and scenarios
            and all(hasattr(p, "paths") and hasattr(p, "kind") for p in scenarios)):
        merged: dict = {}
        for proj in scenarios:
            parcial = _shocks_from_projection(
                proj, horizon=horizon, reduce=reduce, base_scenario=base_scenario)
            for nome, sh in parcial.items():
                destino = merged.setdefault(nome, {})
                repetidas = set(sh) & set(destino)
                if repetidas:
                    raise ValueError(
                        f"duas projeções estressam o mesmo parâmetro "
                        f"({sorted(repetidas)}) no cenário {nome!r}.")
                destino.update(sh)
        return merged

    # 4) ScenarioSet (trajetórias macro) — precisa do modelo satélite p/ projetar
    if hasattr(scenarios, "scenarios"):
        if model is None:
            raise ValueError(
                "scenarios parece ser um ScenarioSet (trajetórias macro): passe também "
                "model= (modelo satélite ajustado, ex.: VasicekZ) para projetar, ou "
                "use um dict simples {cenario: {pd_mult|pd_abs, lgd_mult|lgd_abs}}.")
        if not hasattr(model, "project"):
            raise TypeError(
                f"model deve expor .project(scenarios, horizon=...); "
                f"recebido {type(model).__name__}.")
        proj = model.project(scenarios, horizon=horizon, n_sims=0)
        return _shocks_from_projection(
            proj, horizon=horizon, reduce=reduce, base_scenario=base_scenario)

    raise TypeError(
        "scenarios deve ser um dict {cenario: choque}, uma Projection (ou lista "
        f"delas) ou um ScenarioSet + model=; recebido {type(scenarios).__name__}.")


# ======================================================================
# Motores: métricas de uma carteira sob um motor
# ======================================================================
def _metrics(port, *, engine, q, rho_default, n_scenarios, seed, engine_kwargs) -> dict:
    if engine == "asrf":
        r = port.asrf_capital(q=q, rho_default=rho_default, **engine_kwargs)
        return {"EL": r.expected_loss, "VaR": r.value_at_risk,
                "ES": float("nan"), "CE": r.economic_capital}
    kw = dict(store_segment_losses=False)
    kw.update(engine_kwargs)
    sim = port.simulate(n_scenarios=n_scenarios, q=q, seed=seed,
                        rho_default=rho_default, **kw)
    d = sim.distribution()
    return {"EL": d.el, "VaR": d.var(q), "ES": d.es(q),
            "CE": d.economic_capital(q, metric="var")}


# ======================================================================
# API principal: cenário × métricas de capital + delta vs. base
# ======================================================================
def scenario_capital(
    portfolio: Portfolio,
    scenarios,
    horizon: Optional[int] = None,
    engine: str = "asrf",
    *,
    q: float = DEFAULT_CONFIDENCE,
    model=None,
    base_scenario: str = "base",
    reduce: str = "mean",
    conditional_z: bool = False,
    z_rho: Optional[float] = None,
    rho_default: float = 0.15,
    n_scenarios: int = 100_000,
    seed: Optional[int] = 0,
    **engine_kwargs,
) -> pd.DataFrame:
    """Capital econômico sob cenários macro: quadro cenário × métricas + deltas.

    Para cada cenário, constrói a carteira estressada (:func:`apply_scenario`)
    e recomputa EL/VaR/ES/CE no motor escolhido. A linha de referência é a
    **carteira-base sem choque** (adicionada como ``base`` quando os cenários
    não trazem um com esse nome); os deltas são sempre contra ela.

    Parameters
    ----------
    portfolio:
        A carteira-base.
    scenarios:
        Uma das formas:

        * ``dict`` ``{cenario: {pd_mult|pd_abs, lgd_mult|lgd_abs}}`` — caminho
          simples, sem dependência do econometric;
        * ``Projection`` (ou lista de ``Projection`` de kinds distintos) — a
          trajetória por cenário vira multiplicador vs. o cenário-base;
        * ``ScenarioSet`` + ``model=`` — projeta o modelo satélite e segue como
          acima.
    horizon:
        Nº de períodos iniciais da trajetória considerados na redução (``None``
        = trajetória inteira). Também repassado a ``model.project`` no caminho
        ``ScenarioSet``.
    engine:
        ``"asrf"`` (analítico, barato e determinístico; ``ES`` sai ``NaN``) ou
        ``"monte_carlo"``/``"mc"`` (re-simula por cenário; a mesma ``seed`` em
        todos os cenários dá números aleatórios comuns — deltas estáveis).
    q:
        Nível de confiança das métricas de cauda.
    model:
        Modelo satélite ajustado (obrigatório apenas com ``ScenarioSet``).
    base_scenario:
        Nome do cenário-base nas projeções (normalização do multiplicador).
    reduce:
        Redução da trajetória a um nível: ``"mean"`` (padrão), ``"last"`` ou
        ``"max"`` (conservador).
    conditional_z:
        Propaga o choque de PD via ``Z`` implicado por segmento — ver
        :func:`apply_scenario`. Adiciona a coluna ``z_implicito`` ao quadro.
    z_rho, rho_default:
        Parâmetros do modo ``conditional_z`` (ver :func:`apply_scenario`).
    n_scenarios, seed:
        Parâmetros do motor ``monte_carlo``.
    **engine_kwargs:
        Opções extras repassadas ao motor (ex.: ``copula="t"`` no Monte Carlo).

    Returns
    -------
    pandas.DataFrame
        Uma linha por cenário com ``EL``, ``VaR``, ``ES``, ``CE``, os deltas
        ``delta_EL``/``delta_VaR``/``delta_ES``/``delta_CE`` vs. a base e
        ``delta_CE_pct``; com ``conditional_z=True``, também ``z_implicito``.
    """
    engine = str(engine).lower()
    if engine in ("mc", "montecarlo", "monte_carlo"):
        engine = "monte_carlo"
    if engine not in ("asrf", "monte_carlo"):
        raise ValueError(f"engine deve ser 'asrf' ou 'monte_carlo'; recebido {engine!r}.")

    shocks = _coerce_scenarios(
        scenarios, model=model, horizon=horizon, reduce=reduce, base_scenario=base_scenario)
    if not shocks:
        raise ValueError("nenhum cenário informado.")

    ref = _metrics(portfolio, engine=engine, q=q, rho_default=rho_default,
                   n_scenarios=n_scenarios, seed=seed, engine_kwargs=engine_kwargs)

    linhas = []
    if base_scenario not in shocks:
        # A carteira-base entra como linha de referência explícita.
        linhas.append({"cenario": base_scenario, "z_implicito": np.nan, **ref})
    for nome, shock in shocks.items():
        port2, z = _stress(
            portfolio, shock, conditional_z=conditional_z, rho_default=rho_default,
            z_rho=z_rho, name=f"{portfolio.name}|{nome}", nome_cenario=nome)
        met = _metrics(port2, engine=engine, q=q, rho_default=rho_default,
                       n_scenarios=n_scenarios, seed=seed, engine_kwargs=engine_kwargs)
        linhas.append({"cenario": nome,
                       "z_implicito": np.nan if z is None else z, **met})

    df = pd.DataFrame(linhas)
    for col in ("EL", "VaR", "ES", "CE"):
        df[f"delta_{col}"] = df[col] - ref[col]
    df["delta_CE_pct"] = df["delta_CE"] / ref["CE"] if ref["CE"] else np.nan
    if not conditional_z:
        df = df.drop(columns=["z_implicito"])
    return df


__all__ = ["SHOCK_KEYS", "apply_scenario", "scenario_capital"]
