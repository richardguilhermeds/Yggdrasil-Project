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


def _check_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"A significância alpha deve estar em (0, 1); recebido {alpha!r}.")
    return alpha


def _weighted_quantile(values: ArrayLike, weights: ArrayLike, q: float) -> float:
    """Quantil ``q`` de valores com pesos (inversa da CDF, tipo "lower").

    Base comum do quantil ponderado: distribuições discretas analíticas
    (CreditRisk+) e amostras de Monte Carlo com pesos de verossimilhança
    (*importance sampling*). Os pesos são normalizados internamente — não
    precisam somar 1.
    """
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    order = np.argsort(v)
    v = v[order]
    cdf = np.cumsum(w[order])
    total = cdf[-1]
    if not (total > 0):
        raise ValueError("A soma dos pesos deve ser positiva.")
    # Menor valor cuja CDF acumulada atinge q (quantil tipo "lower").
    idx = int(np.searchsorted(cdf / total, q, side="left"))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


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
        self._sorted: Optional[np.ndarray] = None      # cache p/ estatísticas de ordem

    # ------------------------------------------------------------------
    # Núcleo: quantil respeitando pesos (empírico OU ponderado)
    # ------------------------------------------------------------------
    def _weighted_quantile(self, q: float) -> float:
        """Quantil ``q`` de uma distribuição discreta ponderada (inversa da CDF)."""
        return _weighted_quantile(self.losses, self.weights, q)

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
    # Incerteza amostral: IC e erro-padrão do VaR e do ES
    # ------------------------------------------------------------------
    def _sorted_losses(self) -> np.ndarray:
        """Perdas ordenadas (memoizado) — base das estatísticas de ordem."""
        if self._sorted is None:
            self._sorted = np.sort(self.losses)
        return self._sorted

    def var_ci(self, q: float = DEFAULT_CONFIDENCE, alpha: float = 0.05) -> tuple:
        """Intervalo de confiança não-paramétrico do VaR por estatísticas de ordem.

        Numa amostra empírica de tamanho ``n``, o número de observações abaixo
        do quantil verdadeiro segue ``Binomial(n, q)``; os *ranks* que deixam
        ``alpha/2`` de probabilidade em cada cauda delimitam o intervalo
        ``[x_(l), x_(u)]`` com cobertura ≥ ``1 − alpha``, sem hipótese sobre a
        forma da cauda. Em ``q`` muito alto com poucos cenários os *ranks* são
        truncados nos extremos da amostra (a banda "cola" no máximo observado)
        — sinal de que faltam cenários, ver
        :func:`~yggdrasil.credit_risk.capital.validation.convergence`.

        Distribuições **ponderadas** (grade analítica, ex.: recursão de
        Panjer) não têm erro amostral a quantificar: retorna ``(nan, nan)``
        mantendo a interface única.

        Parameters
        ----------
        q:
            Nível de confiança do VaR.
        alpha:
            Significância do intervalo (``0.05`` → IC de 95%).

        Returns
        -------
        tuple
            ``(lo, hi)`` na mesma unidade monetária de ``losses``.
        """
        q = _check_q(q)
        alpha = _check_alpha(alpha)
        if self.weights is not None:                   # analítica: banda não se aplica
            return (float("nan"), float("nan"))
        from scipy.stats import binom                  # import tardio (padrão do pacote)

        v = self._sorted_losses()
        n = v.size
        # Ranks 1..n pela binomial exata: P(B < l) < alpha/2 e P(B > u−1) ≤ alpha/2.
        lo_rank = int(binom.ppf(alpha / 2.0, n, q))
        hi_rank = int(binom.ppf(1.0 - alpha / 2.0, n, q)) + 1
        lo_idx = min(max(lo_rank, 1), n) - 1           # ranks 1..n → índices 0..n−1
        hi_idx = min(max(hi_rank, 1), n) - 1
        return (float(v[lo_idx]), float(v[hi_idx]))

    def _es_tail_boot(self, q: float, n_boot: int, seed: Optional[int]) -> np.ndarray:
        """Médias *bootstrap* da cauda empírica (perdas ≥ VaR no nível ``q``)."""
        var = self.var(q)
        tail = self.losses[self.losses >= var]
        if tail.size == 0:                             # degenerado: cauda vazia
            return np.array([var])
        rng = np.random.default_rng(seed)
        m = tail.size
        boot = np.empty(int(n_boot))
        for b in range(int(n_boot)):                   # laço evita matriz n_boot×m em memória
            boot[b] = float(np.mean(tail[rng.integers(0, m, size=m)]))
        return boot

    def es_ci(
        self,
        q: float = DEFAULT_CONFIDENCE,
        alpha: float = 0.05,
        n_boot: int = 500,
        seed: Optional[int] = None,
    ) -> tuple:
        """Intervalo de confiança do ES por *bootstrap* da cauda empírica.

        Reamostra com reposição as perdas ``≥ VaR_q`` e recalcula a média da
        cauda ``n_boot`` vezes; a banda é o intervalo percentílico
        ``[alpha/2, 1 − alpha/2]`` das médias reamostradas. Condiciona no VaR
        pontual (não propaga a incerteza de localização do quantil), o que
        tende a subestimar levemente a banda — suficiente como diagnóstico do
        ruído de Monte Carlo.

        Distribuições **ponderadas** (analíticas) retornam ``(nan, nan)``,
        mantendo a interface única.

        Parameters
        ----------
        q:
            Nível de confiança do ES.
        alpha:
            Significância do intervalo (``0.05`` → IC de 95%).
        n_boot:
            Número de reamostragens *bootstrap*.
        seed:
            Semente do gerador (reprodutibilidade).

        Returns
        -------
        tuple
            ``(lo, hi)`` na mesma unidade monetária de ``losses``.
        """
        q = _check_q(q)
        alpha = _check_alpha(alpha)
        n_boot = int(n_boot)
        if n_boot < 1:
            raise ValueError(f"n_boot deve ser >= 1; recebido {n_boot!r}.")
        if self.weights is not None:                   # analítica: banda não se aplica
            return (float("nan"), float("nan"))
        boot = self._es_tail_boot(q, n_boot, seed)
        return (float(np.quantile(boot, alpha / 2.0)),
                float(np.quantile(boot, 1.0 - alpha / 2.0)))

    def var_se(self, q: float = DEFAULT_CONFIDENCE, alpha: float = 0.05) -> float:
        """Erro-padrão aproximado do VaR: meia-largura do IC de :meth:`var_ci`
        dividida pelo quantil normal ``z_{1−alpha/2}``. ``nan`` para
        distribuições ponderadas (analíticas)."""
        lo, hi = self.var_ci(q, alpha)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return float("nan")
        from scipy.stats import norm                   # import tardio (padrão do pacote)
        z = float(norm.ppf(1.0 - _check_alpha(alpha) / 2.0))
        return float((hi - lo) / (2.0 * z))

    def es_se(
        self,
        q: float = DEFAULT_CONFIDENCE,
        n_boot: int = 500,
        seed: Optional[int] = None,
    ) -> float:
        """Erro-padrão do ES: desvio-padrão das médias *bootstrap* da cauda
        (ver :meth:`es_ci`). ``nan`` para distribuições ponderadas (analíticas)."""
        if self.weights is not None:
            return float("nan")
        boot = self._es_tail_boot(_check_q(q), max(int(n_boot), 2), seed)
        return float(np.std(boot, ddof=1)) if boot.size > 1 else 0.0

    # ------------------------------------------------------------------
    # Resumo tabular
    # ------------------------------------------------------------------
    def summary(
        self,
        confidence_levels: Sequence[float] = (0.99, 0.995, 0.999, 0.9997),
        metric: str = "var",
        alpha: float = 0.05,
        n_boot: int = 500,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Tabela de EL, VaR, ES e CE em vários níveis de confiança.

        Inclui as bandas de incerteza amostral ``VaR_lo``/``VaR_hi``
        (estatísticas de ordem, :meth:`var_ci`) e ``ES_lo``/``ES_hi``
        (*bootstrap* da cauda, :meth:`es_ci`), no nível ``1 − alpha``. Em
        distribuições ponderadas (analíticas) as bandas são ``NaN`` — não há
        ruído amostral a quantificar.
        """
        el = self.el
        linhas = []
        for q in confidence_levels:
            var = self.var(q)
            es = self.es(q)
            var_lo, var_hi = self.var_ci(q, alpha)
            es_lo, es_hi = self.es_ci(q, alpha, n_boot=n_boot, seed=seed)
            linhas.append(
                {
                    "nivel_confianca": q,
                    "EL": el,
                    "VaR": var,
                    "ES": es,
                    "CE_var": var - el,
                    "CE_es": es - el,
                    "CE": (var - el) if metric == "var" else (es - el),
                    "VaR_lo": var_lo,
                    "VaR_hi": var_hi,
                    "ES_lo": es_lo,
                    "ES_hi": es_hi,
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
