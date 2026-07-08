"""
Projeção condicional a cenários e integração com os usos (Guia §5, §6.2 ``scenarios``)
====================================================================================
Com o modelo estimado, a projeção é mecânica: alimentam-se as **trajetórias das
variáveis macro** de cada cenário (base, otimista, pessimista, adversos de
estresse) e o modelo devolve a trajetória da PD/LGD/CCF, reconvertida à escala
original. Este módulo fornece o **contrato de cenários** e a **fonte única de
projeção** que o guia exige para eliminar "a pior patologia desse processo:
números diferentes para a mesma pergunta em áreas diferentes" (§5).

Os três cuidados do guia estão cobertos: a **propagação multi-passo** das
defasagens e os **intervalos por simulação de resíduos** vêm do motor de cada
modelo (:meth:`SatelliteModel.project`); a **coerência entre cenário e modelo** é
checada aqui (as variáveis do cenário precisam ser exatamente as do modelo); e a
**reconciliação com o uso** aparece em :func:`ecl_projection` (ponderação por
probabilidade para o ECL forward-looking de IFRS 9 / CMN 4.966) e no uso da
trajetória adversa direta para estresse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import _engine
from .base import Projection


# ======================================================================
# Contrato de cenários
# ======================================================================
@dataclass
class Scenario:
    """Um cenário macroeconômico: trajetórias **futuras** das variáveis + peso.

    Parameters
    ----------
    name:
        Rótulo (``"base"``, ``"adverso"``, ``"severo"``…).
    macro:
        DataFrame das trajetórias **futuras** das variáveis macro, indexado pelos
        períodos de projeção (à frente do fim da amostra), com **as mesmas
        colunas** que o modelo usa.
    probability:
        Peso do cenário para o ECL (opcional). Numa fonte única, os pesos dos
        cenários somam 1.
    """

    name: str
    macro: pd.DataFrame
    probability: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.macro, pd.DataFrame):
            raise TypeError("macro deve ser um pandas.DataFrame.")
        if self.macro.empty:
            raise ValueError(f"cenário {self.name!r} com macro vazia.")
        if self.probability is not None and not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"probability deve estar em [0, 1]; recebido {self.probability!r}.")

    @property
    def horizon(self) -> int:
        return len(self.macro)


@dataclass
class ScenarioSet:
    """Conjunto de cenários — a **fonte única** para todos os usos (Guia §5)."""

    scenarios: list = field(default_factory=list)

    def __post_init__(self) -> None:
        names = [s.name for s in self.scenarios]
        if len(names) != len(set(names)):
            raise ValueError("nomes de cenário duplicados no ScenarioSet.")

    def __iter__(self):
        return iter(self.scenarios)

    def __len__(self) -> int:
        return len(self.scenarios)

    def names(self) -> list[str]:
        return [s.name for s in self.scenarios]

    def probabilities(self) -> Optional[dict]:
        probs = {s.name: s.probability for s in self.scenarios}
        if any(p is None for p in probs.values()):
            return None
        total = float(sum(probs.values()))
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"as probabilidades dos cenários devem somar 1; somam {total:.4f}.")
        return probs

    def get(self, name: str) -> Scenario:
        for s in self.scenarios:
            if s.name == name:
                return s
        raise KeyError(name)

    @classmethod
    def from_frames(cls, frames: Mapping[str, pd.DataFrame],
                    probabilities: Optional[Mapping[str, float]] = None) -> "ScenarioSet":
        """Constrói a partir de ``{nome: DataFrame_futuro}`` e pesos opcionais."""
        probs = probabilities or {}
        return cls([Scenario(name=k, macro=v, probability=probs.get(k)) for k, v in frames.items()])


# ======================================================================
# Projeção — a fonte única
# ======================================================================
def project(model, scenarios, horizon: Optional[int] = None, alpha: float = 0.10,
            n_sims: int = 2000, seed: int = 0) -> Projection:
    """Projeta um modelo ajustado sobre um :class:`ScenarioSet` (ou dict/DataFrame).

    Fina camada de conveniência sobre :meth:`SatelliteModel.project` — a
    **fonte única de projeção** do guia. Todos os usos (ECL, estresse, capital)
    passam por aqui para não divergirem."""
    return model.project(scenarios, horizon=horizon, alpha=alpha, n_sims=n_sims, seed=seed)


def ecl_projection(model, scenarios: ScenarioSet, horizon: Optional[int] = None,
                   n_sims: int = 0, seed: int = 0) -> pd.Series:
    """Projeção **ponderada por probabilidade** — o insumo forward-looking do ECL.

    ``parâmetro_ECL_t = Σ_s w_s · parâmetro_{s,t}`` (Guia §5; IFRS 9 / CMN 4.966).
    Exige que **todos** os cenários tenham ``probability`` e que somem 1.
    """
    probs = scenarios.probabilities()
    if probs is None:
        raise ValueError(
            "ECL exige probabilidade em todos os cenários (e soma 1). "
            "Defina Scenario.probability para cada cenário."
        )
    proj = model.project(scenarios, horizon=horizon, n_sims=n_sims, seed=seed)
    return proj.weighted(probs)


# ======================================================================
# Construtores de trajetórias macro (demos, estresse e testes)
# ======================================================================
def extend_macro(macro: pd.DataFrame, horizon: int, freq: str = "MS",
                 method: str = "revert", revert_speed: float = 0.15) -> pd.DataFrame:
    """Gera uma trajetória macro **futura** de ``horizon`` períodos — o cenário
    **base** de referência.

    Métodos: ``"hold"`` (mantém o último valor), ``"trend"`` (extrapola a última
    variação) ou ``"revert"`` (reversão gradual à média histórica, à velocidade
    ``revert_speed`` — o comportamento default, mais realista para variáveis
    cíclicas).
    """
    idx = _engine.future_index(macro.index, freq, horizon)
    out = {}
    for col in macro.columns:
        s = macro[col].to_numpy(dtype=float)
        last, mean = float(s[-1]), float(np.mean(s))
        if method == "hold":
            path = np.full(horizon, last)
        elif method == "trend":
            step = float(s[-1] - s[-2]) if len(s) > 1 else 0.0
            path = last + step * np.arange(1, horizon + 1)
        elif method == "revert":
            path = np.empty(horizon)
            cur = last
            for h in range(horizon):
                cur = cur + revert_speed * (mean - cur)
                path[h] = cur
        else:
            raise ValueError("method deve ser 'hold', 'trend' ou 'revert'.")
        out[col] = path
    return pd.DataFrame(out, index=idx)


def shock_scenarios(
    base_future: pd.DataFrame,
    shocks: Mapping[str, Mapping[str, float]],
    probabilities: Optional[Mapping[str, float]] = None,
    include_base: bool = True,
    base_probability: Optional[float] = None,
) -> ScenarioSet:
    """Constrói um :class:`ScenarioSet` aplicando **choques aditivos** a um cenário base.

    Cada entrada de ``shocks`` é ``nome_cenario → {variavel: delta}``: o cenário
    soma ``delta`` (constante ao longo do horizonte) à trajetória base da variável
    — o modo usual de derivar adverso/severo a partir do base (ex.: ``desemprego
    +3`` p.p., ``renda −2``).

    Parameters
    ----------
    base_future:
        Trajetória base (saída de :func:`extend_macro`).
    shocks:
        ``{"adverso": {"desemprego": 3.0, "renda": -2.0}, "severo": {...}}``.
    probabilities, base_probability:
        Pesos opcionais dos cenários de choque e do base (para o ECL).
    include_base:
        Inclui o cenário ``"base"`` (sem choque) no conjunto.
    """
    probs = dict(probabilities or {})
    scenarios = []
    if include_base:
        scenarios.append(Scenario("base", base_future.copy(), probability=base_probability))
    for name, deltas in shocks.items():
        mf = base_future.copy()
        for var, delta in deltas.items():
            if var not in mf.columns:
                raise ValueError(f"choque em variável ausente do base: {var!r}.")
            mf[var] = mf[var] + float(delta)
        scenarios.append(Scenario(name, mf, probability=probs.get(name)))
    return ScenarioSet(scenarios)


def standard_scenarios(
    macro: pd.DataFrame,
    horizon: int = 12,
    freq: str = "MS",
    stress_var: str = "desemprego",
    probabilities: Sequence[float] = (0.5, 0.3, 0.2),
) -> ScenarioSet:
    """Atalho: base (reversão à média) + **otimista**/**adverso** por choque em
    ``stress_var`` (±), com pesos padrão — um ponto de partida para demonstração
    e estresse (a governança define os cenários reais)."""
    base = extend_macro(macro, horizon, freq=freq, method="revert")
    sd = float(macro[stress_var].std())
    return shock_scenarios(
        base,
        shocks={"adverso": {stress_var: 2.0 * sd}, "otimista": {stress_var: -1.0 * sd}},
        probabilities={"adverso": probabilities[2], "otimista": probabilities[1]},
        include_base=True,
        base_probability=probabilities[0],
    )


__all__ = [
    "Scenario",
    "ScenarioSet",
    "project",
    "ecl_projection",
    "extend_macro",
    "shock_scenarios",
    "standard_scenarios",
]
