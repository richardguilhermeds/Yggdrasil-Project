"""
Motor interno de design e projeção multi-passo (Guia §3.1, §5)
==============================================================
Núcleo compartilhado pelos modelos lineares-no-*link* (ARDL, fator ``Z`` de
Vasicek, regressão beta, fractional logit): a construção da **matriz de
defasagens** (design) usada na estimação e a **propagação multi-passo** usada na
projeção condicional a cenários.

Dois cuidados do guia estão implementados aqui de uma vez (§5): (i) a **propagação
correta das defasagens** — em projeções multi-passo com termo autorregressivo, a
projeção do período anterior alimenta a seguinte; (ii) os **intervalos por
simulação de resíduos**, em que os erros se **acumulam** ao longo do horizonte
(reamostragem *bootstrap* dos resíduos in-sample, que preserva caudas gordas
melhor que a hipótese gaussiana).

Não é API pública: os modelos expõem ``fit``/``predict``/``project``; este módulo
é o encanamento por baixo.
"""
from __future__ import annotations

import re
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .transforms import make_lags, seasonal_dummies, event_dummies

_AR_RE = re.compile(r"^y_l(\d+)$")
_LAG_RE = re.compile(r"^(.+)_l(\d+)$")


# ======================================================================
# Construção do design (matriz de regressores)
# ======================================================================
def build_design(endog_link: pd.Series, macro: Optional[pd.DataFrame], spec) -> pd.DataFrame:
    """Monta o design ``X`` do ARDL a partir da dependente (na escala do *link*)
    e da macro, segundo a :class:`Specification`.

    Colunas, nesta ordem: ``const`` (e ``trend`` se ``trend='ct'``), defasagens
    próprias ``y_l1..y_l{ar}``, termos macro ``{var}`` / ``{var}_l{lag}``, dummies
    sazonais ``s02..`` e dummies de evento. Linhas iniciais com ``NaN`` (bordas das
    defasagens) **não** são removidas — use :func:`~...transforms.align`.
    """
    idx = endog_link.index
    cols: dict[str, pd.Series] = {}
    if spec.trend in ("c", "ct"):
        cols["const"] = pd.Series(1.0, index=idx)
    if spec.trend == "ct":
        cols["trend"] = pd.Series(np.arange(len(idx), dtype=float), index=idx)

    # defasagens próprias (AR)
    for i in range(1, int(spec.ar) + 1):
        cols[f"y_l{i}"] = endog_link.shift(i)

    # termos macro
    if spec.exog:
        if macro is None:
            raise ValueError("spec.exog exige uma macro (DataFrame) não nula.")
        lag_frame = make_lags(macro[list(spec.exog.keys())], dict(spec.exog))
        for c in lag_frame.columns:
            cols[str(c)] = lag_frame[c]

    X = pd.DataFrame(cols, index=idx)

    # dummies sazonais e de evento (não entram em NaN de burn-in)
    if spec.seasonal:
        X = pd.concat([X, seasonal_dummies(idx, period=spec.seasonal_period)], axis=1)
    if spec.events:
        X = pd.concat([X, event_dummies(idx, dict(spec.events))], axis=1)
    return X


def classify_terms(names, macro_cols) -> list:
    """Classifica cada nome de coeficiente para a recursão de projeção.

    Devolve uma lista de tuplas: ``("const",)``, ``("trend",)``, ``("ar", i)``,
    ``("exog", var, lag)``, ``("seasonal", p)`` ou ``("event", name)`` (0 no futuro,
    salvo cenário que reative o evento).
    """
    macro_cols = set(macro_cols or [])
    terms = []
    for name in names:
        if name == "const":
            terms.append(("const",))
        elif name == "trend":
            terms.append(("trend",))
        elif _AR_RE.match(name):
            terms.append(("ar", int(_AR_RE.match(name).group(1))))
        elif name in macro_cols:
            terms.append(("exog", name, 0))
        elif _LAG_RE.match(name) and _LAG_RE.match(name).group(1) in macro_cols:
            m = _LAG_RE.match(name)
            terms.append(("exog", m.group(1), int(m.group(2))))
        elif re.match(r"^s\d+$", name):
            terms.append(("seasonal", int(name[1:])))
        else:
            terms.append(("event", name))
    return terms


# ======================================================================
# Propagação multi-passo (projeção)
# ======================================================================
def _season_number(ts: pd.Timestamp, period: int) -> int:
    return ts.month if period == 12 else (ts.quarter if period == 4 else ts.month)


def _step_paths(
    params: pd.Series,
    terms: list,
    macro_full: Optional[pd.DataFrame],
    future_index: pd.Index,
    hist_link: np.ndarray,
    trend_offset: int,
    seasonal_period: int,
    innovations: np.ndarray,
    event_flags: Optional[Mapping[str, np.ndarray]] = None,
) -> np.ndarray:
    """Recursão vetorizada nas simulações. ``innovations`` tem forma
    ``(n_sims, horizonte)``. Devolve as trajetórias na **escala do link**
    ``(n_sims, horizonte)``.
    """
    coef = params.to_numpy(dtype=float)
    names = list(params.index)
    n_sims, H = innovations.shape
    n_hist = len(hist_link)
    seq = np.empty((n_sims, n_hist + H), dtype=float)
    seq[:, :n_hist] = hist_link[None, :]

    # pré-computa as séries macro defasadas para lookup por timestamp
    shifted: dict[tuple, pd.Series] = {}
    if macro_full is not None:
        for t, term in enumerate(terms):
            if term[0] == "exog":
                _, var, lag = term
                key = (var, lag)
                if key not in shifted:
                    shifted[key] = macro_full[var].shift(lag)

    ts_future = pd.DatetimeIndex(
        future_index.to_timestamp() if isinstance(future_index, pd.PeriodIndex) else future_index
    )
    for j in range(H):
        ts = ts_future[j]
        pos = n_hist + j
        base = np.zeros(n_sims)
        ar_val = np.zeros(n_sims)
        for k, term in enumerate(terms):
            c = coef[k]
            kind = term[0]
            if kind == "const":
                base += c
            elif kind == "trend":
                base += c * (trend_offset + j)
            elif kind == "ar":
                ar_val += c * seq[:, pos - term[1]]
            elif kind == "exog":
                _, var, lag = term
                val = shifted[(var, lag)].get(future_index[j], np.nan)
                if not np.isfinite(val):
                    val = float(macro_full[var].iloc[-1])  # fallback: último observado
                base += c * val
            elif kind == "seasonal":
                base += c * (1.0 if _season_number(ts, seasonal_period) == term[1] else 0.0)
            elif kind == "event":
                flag = 0.0
                if event_flags is not None and term[1] in event_flags:
                    flag = float(event_flags[term[1]][j])
                base += c * flag
        seq[:, pos] = base + ar_val + innovations[:, j]
    return seq[:, n_hist:]


def forecast_paths(
    params: pd.Series,
    spec,
    inverse_link,
    hist_link: pd.Series,
    macro_full: Optional[pd.DataFrame],
    future_index: pd.Index,
    resid_pool: np.ndarray,
    *,
    trend_offset: int,
    n_sims: int = 2000,
    alpha: float = 0.10,
    seed: int = 0,
    event_flags: Optional[Mapping[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """Projeta ``horizonte = len(future_index)`` passos à frente com intervalos.

    Devolve um DataFrame indexado por ``future_index`` com ``mean`` e ``lower`` /
    ``upper`` (escala **original** da taxa) e ``mean_link`` (escala do *link*). A
    média é o *point forecast* determinístico (inovação zero); a banda vem da
    simulação *bootstrap* de resíduos propagada pela dinâmica AR (os erros se
    acumulam — Guia §5).
    """
    macro_cols = list(macro_full.columns) if macro_full is not None else []
    terms = classify_terms(list(params.index), macro_cols)
    hist = np.asarray(hist_link.to_numpy(dtype=float))
    H = len(future_index)

    # trajetória determinística (média) — inovação zero
    zero = np.zeros((1, H))
    mean_link = _step_paths(params, terms, macro_full, future_index, hist, trend_offset,
                            spec.seasonal_period, zero, event_flags)[0]
    mean_rate = np.asarray(inverse_link(mean_link), dtype=float)

    out = pd.DataFrame({"mean": mean_rate, "mean_link": mean_link}, index=future_index)

    # intervalos por simulação de resíduos (bootstrap)
    pool = np.asarray(resid_pool, dtype=float)
    pool = pool[np.isfinite(pool)]
    if n_sims and pool.size >= 2:
        rng = np.random.default_rng(seed)
        innov = rng.choice(pool, size=(n_sims, H), replace=True)
        sim_link = _step_paths(params, terms, macro_full, future_index, hist, trend_offset,
                               spec.seasonal_period, innov, event_flags)
        sim_rate = np.asarray(inverse_link(sim_link), dtype=float)
        lo = np.quantile(sim_rate, alpha / 2.0, axis=0)
        hi = np.quantile(sim_rate, 1.0 - alpha / 2.0, axis=0)
        out["lower"] = lo
        out["upper"] = hi
    else:
        out["lower"] = np.nan
        out["upper"] = np.nan
    return out[["mean", "lower", "upper", "mean_link"]]


def normalize_scenarios(scenarios) -> list[tuple]:
    """Normaliza a entrada de projeção em ``[(nome, macro_futura, probabilidade), ...]``.

    Aceita, por *duck typing*: um :class:`~...scenarios.ScenarioSet` (atributo
    ``scenarios``), um :class:`~...scenarios.Scenario` isolado, um ``Mapping``
    ``{nome: DataFrame|Scenario}``, um único ``DataFrame`` (vira ``"cenario"``) ou
    uma sequência de :class:`Scenario`.
    """
    if scenarios is None:
        raise ValueError("nenhum cenário fornecido para a projeção.")
    if isinstance(scenarios, pd.DataFrame):
        return [("cenario", scenarios, None)]
    if hasattr(scenarios, "scenarios"):  # ScenarioSet
        return [(s.name, s.macro, getattr(s, "probability", None)) for s in scenarios.scenarios]
    if hasattr(scenarios, "macro") and hasattr(scenarios, "name"):  # Scenario isolado
        return [(scenarios.name, scenarios.macro, getattr(scenarios, "probability", None))]
    if isinstance(scenarios, Mapping):
        out = []
        for k, v in scenarios.items():
            macro = v.macro if hasattr(v, "macro") else v
            out.append((k, macro, getattr(v, "probability", None)))
        return out
    # sequência de Scenario
    return [(s.name, s.macro, getattr(s, "probability", None)) for s in scenarios]


def future_index(index: pd.Index, freq: str, n: int) -> pd.Index:
    """Gera ``n`` períodos futuros a partir do fim de ``index`` na frequência ``freq``."""
    ts = pd.DatetimeIndex(index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index)
    off = pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start=ts[-1] + off, periods=n, freq=freq)


def concat_macro(hist_macro: Optional[pd.DataFrame], future_macro: Optional[pd.DataFrame]):
    """Concatena macro histórica + futura (para lookups de defasagem no futuro)."""
    if hist_macro is None:
        return future_macro
    if future_macro is None:
        return hist_macro
    faltando = [c for c in hist_macro.columns if c not in future_macro.columns]
    if faltando:
        raise ValueError(
            f"o cenário não traz as variáveis do modelo: faltam {faltando}. "
            "Guia §5: as variáveis do cenário precisam ser exatamente as do modelo."
        )
    combined = pd.concat([hist_macro, future_macro[hist_macro.columns]], axis=0)
    # se o futuro sobrepõe o histórico (mesmos carimbos de tempo), o cenário
    # vence — mantém rótulos únicos para o lookup por timestamp na projeção.
    return combined[~combined.index.duplicated(keep="last")]


__all__ = [
    "build_design",
    "classify_terms",
    "forecast_paths",
    "normalize_scenarios",
    "future_index",
    "concat_macro",
]
