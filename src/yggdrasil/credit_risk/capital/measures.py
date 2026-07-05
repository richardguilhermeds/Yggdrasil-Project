"""
Medidas de risco e a distribuição de perdas da carteira
=======================================================
O objetivo central de todo modelo de capital econômico é estimar a
**distribuição de probabilidade das perdas agregadas** da carteira em um
horizonte definido (tipicamente 1 ano). Uma vez que essa distribuição existe,
todas as métricas derivam dela (Seção 2 do guia):

* **Perda esperada (EL)** — a média da distribuição; é um custo previsível do
  negócio, coberto por provisão (ECL, sob IFRS 9 / Resolução CMN 4.966) e
  precificado no *spread*.
* **VaR de crédito** — o quantil no nível de confiança escolhido ``q`` (ex.: 99,9%).
* **Capital econômico (CE)** — a distância entre o quantil e a média,
  ``CE = VaR_q(L) − EL``; é a perda **inesperada (UL)**, que só pode ser
  absorvida por capital.
* **Expected Shortfall (ES)** — a média das perdas além do quantil; é
  subaditivo e mais sensível ao formato da cauda, cada vez mais usado como
  métrica principal ou de controle.

Este módulo trabalha sobre uma **amostra empírica de perdas** (uma perda por
cenário, vinda da simulação de Monte Carlo) ou sobre uma **distribuição
discreta ponderada** (pares valor/probabilidade, como a que o CreditRisk+
produz analiticamente). A classe :class:`LossDistribution` unifica os dois
casos e expõe as métricas de forma consistente.

Contexto regulatório: Resolução CMN 4.557/2017 (ICAAP) e 4.966/2021 (ponto de
partida dos parâmetros de perda esperada).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

# Nível de confiança de referência de Basileia para capital econômico.
DEFAULT_CONFIDENCE = 0.999

ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


# ======================================================================
# Helpers de validação
# ======================================================================
def _as_1d(losses: ArrayLike) -> np.ndarray:
    """Converte a entrada em vetor 1-D de float, sem NaN."""
    arr = np.asarray(losses, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("A distribuição de perdas está vazia.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("A distribuição de perdas contém valores não finitos (NaN/inf).")
    return arr


def _check_q(q: float) -> float:
    q = float(q)
    if not (0.0 < q < 1.0):
        raise ValueError(f"O nível de confiança q deve estar em (0, 1); recebido {q!r}.")
    return q


# ======================================================================
# Funções de medida sobre uma amostra empírica de perdas
# ======================================================================
def expected_loss(losses: ArrayLike) -> float:
    """Perda esperada (EL) — a média da distribuição de perdas."""
    return float(np.mean(_as_1d(losses)))


def loss_volatility(losses: ArrayLike) -> float:
    """Volatilidade das perdas (desvio-padrão) — a "perda inesperada" no sentido
    de 1 desvio-padrão. Não é o capital econômico, que usa um quantil de cauda."""
    return float(np.std(_as_1d(losses), ddof=1)) if len(_as_1d(losses)) > 1 else 0.0


def value_at_risk(losses: ArrayLike, q: float = DEFAULT_CONFIDENCE) -> float:
    """VaR de crédito: o quantil ``q`` da distribuição de perdas.

    Usa o método de interpolação linear (``numpy.quantile``), padrão para uma
    amostra grande de Monte Carlo. Em amostras pequenas o quantil de cauda é
    ruidoso — ver :func:`~yggdrasil.credit_risk.capital.validation.convergence`.
    """
    return float(np.quantile(_as_1d(losses), _check_q(q)))


def expected_shortfall(losses: ArrayLike, q: float = DEFAULT_CONFIDENCE) -> float:
    """Expected Shortfall (ES / CVaR): a média das perdas **iguais ou acima** do
    VaR no nível ``q``.

    Definido como ``E[L | L >= VaR_q(L)]``. É subaditivo (coerente) e mais
    estável que o VaR para alocação de capital (alocação de Euler)."""
    arr = _as_1d(losses)
    var = value_at_risk(arr, q)
    tail = arr[arr >= var]
    # A cauda nunca fica vazia porque o próprio VaR pertence à amostra (ou é
    # interpolado entre pontos), mas protegemos contra o caso degenerado.
    if tail.size == 0:
        return var
    return float(np.mean(tail))


def unexpected_loss(losses: ArrayLike, q: float = DEFAULT_CONFIDENCE) -> float:
    """Perda inesperada = capital econômico pela métrica VaR: ``VaR_q(L) − EL``."""
    arr = _as_1d(losses)
    return value_at_risk(arr, q) - expected_loss(arr)


def economic_capital(
    losses: ArrayLike, q: float = DEFAULT_CONFIDENCE, metric: str = "var"
) -> float:
    """Capital econômico: a distância entre a métrica de cauda e a perda esperada.

    * ``metric="var"`` → ``VaR_q(L) − EL`` (padrão de mercado).
    * ``metric="es"``  → ``ES_q(L) − EL`` (métrica coerente, mais estável para alocação).
    """
    arr = _as_1d(losses)
    el = expected_loss(arr)
    if metric == "var":
        return value_at_risk(arr, q) - el
    if metric == "es":
        return expected_shortfall(arr, q) - el
    raise ValueError(f"metric deve ser 'var' ou 'es'; recebido {metric!r}.")


# ======================================================================
# Distribuição de perdas (amostra empírica OU distribuição discreta ponderada)
# ======================================================================
class LossDistribution:
    """Envelope sobre a distribuição de perdas agregadas da carteira.

    Aceita dois formatos, cobrindo os dois tipos de motor de cálculo:

    * **Amostra empírica** (Monte Carlo): ``losses`` é um vetor com uma perda
      por cenário simulado; ``weights=None``.
    * **Distribuição discreta ponderada** (CreditRisk+ / analítica): ``losses``
      são os valores de perda possíveis e ``weights`` as probabilidades
      associadas (somam ~1).

    Parameters
    ----------
    losses:
        Vetor de perdas (por cenário) ou de valores de perda (grade discreta).
    weights:
        Probabilidades associadas a cada valor de ``losses`` (distribuição
        discreta). Se ``None``, trata ``losses`` como amostra equiponderada.
    expected:
        Perda esperada "teórica" (``Σ PD·LGD·EAD``), quando conhecida
        analiticamente. Se ``None``, usa a média da distribuição. Útil porque a
        EL analítica é exata, enquanto a média amostral do Monte Carlo tem ruído.
    name:
        Rótulo (ex.: nome da carteira / do motor) para relatórios.
    """

    def __init__(
        self,
        losses: ArrayLike,
        weights: Optional[ArrayLike] = None,
        expected: Optional[float] = None,
        name: str = "carteira",
    ) -> None:
        self.losses = _as_1d(losses)
        if weights is None:
            self.weights: Optional[np.ndarray] = None
        else:
            w = np.asarray(weights, dtype=float).ravel()
            if w.shape != self.losses.shape:
                raise ValueError("weights e losses devem ter o mesmo tamanho.")
            if np.any(w < 0):
                raise ValueError("weights não podem ser negativos.")
            total = w.sum()
            if total <= 0:
                raise ValueError("A soma dos weights deve ser positiva.")
            self.weights = w / total                    # normaliza para somar 1
        self._expected = None if expected is None else float(expected)
        self.name = name

    # ------------------------------------------------------------------
    # Núcleo: quantil respeitando pesos (empírico OU ponderado)
    # ------------------------------------------------------------------
    def _weighted_quantile(self, q: float) -> float:
        """Quantil ``q`` de uma distribuição discreta ponderada (inversa da CDF)."""
        order = np.argsort(self.losses)
        v = self.losses[order]
        w = self.weights[order]
        cdf = np.cumsum(w)
        # Menor valor cuja CDF acumulada atinge q (quantil tipo "lower").
        idx = int(np.searchsorted(cdf, q, side="left"))
        idx = min(idx, len(v) - 1)
        return float(v[idx])

    @property
    def el(self) -> float:
        """Perda esperada. Usa a EL analítica se fornecida; senão a média."""
        if self._expected is not None:
            return self._expected
        if self.weights is None:
            return float(np.mean(self.losses))
        return float(np.sum(self.losses * self.weights))

    def mean(self) -> float:
        """Média da distribuição (ignora a EL analítica passada em ``expected``)."""
        if self.weights is None:
            return float(np.mean(self.losses))
        return float(np.sum(self.losses * self.weights))

    def std(self) -> float:
        """Desvio-padrão da distribuição de perdas."""
        if self.weights is None:
            return float(np.std(self.losses, ddof=1)) if len(self.losses) > 1 else 0.0
        m = self.mean()
        var = float(np.sum(self.weights * (self.losses - m) ** 2))
        return float(np.sqrt(max(var, 0.0)))

    def var(self, q: float = DEFAULT_CONFIDENCE) -> float:
        """VaR de crédito no nível ``q``."""
        q = _check_q(q)
        if self.weights is None:
            return float(np.quantile(self.losses, q))
        return self._weighted_quantile(q)

    def es(self, q: float = DEFAULT_CONFIDENCE) -> float:
        """Expected Shortfall no nível ``q`` (média das perdas ≥ VaR)."""
        q = _check_q(q)
        var = self.var(q)
        mask = self.losses >= var
        if not np.any(mask):
            return var
        if self.weights is None:
            return float(np.mean(self.losses[mask]))
        w = self.weights[mask]
        return float(np.sum(self.losses[mask] * w) / w.sum())

    def economic_capital(self, q: float = DEFAULT_CONFIDENCE, metric: str = "var") -> float:
        """Capital econômico: ``VaR−EL`` (``metric='var'``) ou ``ES−EL`` (``'es'``)."""
        if metric == "var":
            return self.var(q) - self.el
        if metric == "es":
            return self.es(q) - self.el
        raise ValueError(f"metric deve ser 'var' ou 'es'; recebido {metric!r}.")

    def quantile(self, q: float) -> float:
        """Alias de :meth:`var` — o quantil ``q`` da distribuição de perdas."""
        return self.var(q)

    # ------------------------------------------------------------------
    # Resumo tabular
    # ------------------------------------------------------------------
    def summary(
        self,
        confidence_levels: Sequence[float] = (0.99, 0.995, 0.999, 0.9997),
        metric: str = "var",
    ) -> pd.DataFrame:
        """Tabela de EL, VaR, ES e CE em vários níveis de confiança."""
        el = self.el
        linhas = []
        for q in confidence_levels:
            var = self.var(q)
            es = self.es(q)
            linhas.append(
                {
                    "nivel_confianca": q,
                    "EL": el,
                    "VaR": var,
                    "ES": es,
                    "CE_var": var - el,
                    "CE_es": es - el,
                    "CE": (var - el) if metric == "var" else (es - el),
                }
            )
        return pd.DataFrame(linhas)

    def __repr__(self) -> str:  # pragma: no cover - conveniência
        n = len(self.losses)
        tipo = "ponderada" if self.weights is not None else "amostral"
        return (f"LossDistribution({self.name!r}, {tipo}, n={n}, EL={self.el:,.2f}, "
                f"VaR99.9={self.var(0.999):,.2f})")


__all__ = [
    "DEFAULT_CONFIDENCE",
    "expected_loss",
    "loss_volatility",
    "value_at_risk",
    "expected_shortfall",
    "unexpected_loss",
    "economic_capital",
    "LossDistribution",
]
