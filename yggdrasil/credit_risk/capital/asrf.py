"""
Motor v1 — Modelo assintótico de fator único (ASRF / Vasicek)
=============================================================
O modelo estrutural de Merton levado ao limite de carteira (Seção 3.1 do guia):
cada devedor entra em *default* quando o valor latente dos seus ativos cai
abaixo de um limiar, e todos compartilham a exposição a um único fator
sistêmico. Sob a hipótese de carteira **assintoticamente granular** (nenhuma
exposição individual é relevante), a perda condicional ao fator no quantil ``q``
tem forma fechada. O capital por unidade de exposição é:

    K = LGD × [ N( (N⁻¹(PD) + √ρ · N⁻¹(q)) / √(1 − ρ) ) − PD ]

onde ``N`` é a normal padrão acumulada, ``N⁻¹`` sua inversa, ``ρ`` a correlação
de ativos e ``q`` o nível de confiança. Esta é exatamente a fórmula do IRB de
Basileia — no capital econômico, com ``ρ`` estimado internamente.

Vantagens: transparência, custo computacional nulo e **aditividade** (o capital
da carteira é a soma dos capitais individuais). Limitações: um único fator não
captura diversificação entre produtos, e a granularidade infinita ignora
concentrações — por isso o ASRF é o **ponto de partida** e o benchmark, e a
simulação de Monte Carlo multifatorial (:mod:`.monte_carlo`) é o modelo
principal.

Este é o motor da **"versão 1"** recomendada pelo guia: simples, completo e
auditável, calculado antes de qualquer sofisticação.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

if TYPE_CHECKING:
    from .portfolio import Portfolio


# ======================================================================
# Núcleo analítico (vetorizável)
# ======================================================================
def conditional_pd(pd: float | np.ndarray, rho: float | np.ndarray,
                   q: float = 0.999) -> float | np.ndarray:
    """PD **condicional** ao cenário sistêmico ruim no nível ``q`` (Vasicek).

    ``p(q) = N( (N⁻¹(PD) + √ρ · N⁻¹(q)) / √(1 − ρ) )``.

    É a taxa de *default* que se materializa quando o fator sistêmico atinge o
    seu quantil adverso ``q``. Para ``q = 0.5`` retorna a própria PD.
    """
    pd = np.asarray(pd, dtype=float)
    rho = np.asarray(rho, dtype=float)
    # PD 0 ou 1 são pontos-limite: a inversa da normal diverge, mas o resultado
    # é bem-definido (0 e 1). Trata-se via clip para evitar ±inf.
    pd_c = np.clip(pd, 1e-12, 1 - 1e-12)
    num = norm.ppf(pd_c) + np.sqrt(rho) * norm.ppf(q)
    den = np.sqrt(1.0 - rho)
    out = norm.cdf(num / den)
    return out


def capital_ratio(pd: float | np.ndarray, lgd: float | np.ndarray,
                  rho: float | np.ndarray, q: float = 0.999) -> float | np.ndarray:
    """Capital por unidade de EAD (a perda **inesperada** por real exposto):

    ``K = LGD × [ p(q) − PD ]``  — já subtrai a perda esperada ``PD × LGD``.
    """
    lgd = np.asarray(lgd, dtype=float)
    pd = np.asarray(pd, dtype=float)
    return lgd * (conditional_pd(pd, rho, q) - pd)


# ======================================================================
# Resultado
# ======================================================================
@dataclass
class AsrfResult:
    """Resultado do cálculo ASRF por segmento e agregado."""

    q: float
    per_segment: pd.DataFrame  # segmento, PD, LGD, EAD, rho, cond_pd, K, EL, capital, VaR
    expected_loss: float
    economic_capital: float    # Σ capital_i (perda inesperada além da EL)
    value_at_risk: float       # EL + EC (perda condicional total no quantil q)
    metric: str = "var"

    def summary(self) -> pd.DataFrame:
        """Uma linha agregada com EL, VaR, CE."""
        return pd.DataFrame([{
            "nivel_confianca": self.q,
            "EL": self.expected_loss,
            "VaR": self.value_at_risk,
            "CE": self.economic_capital,
        }])

    def __repr__(self) -> str:  # pragma: no cover
        return (f"AsrfResult(q={self.q}, EL={self.expected_loss:,.2f}, "
                f"CE={self.economic_capital:,.2f}, VaR={self.value_at_risk:,.2f})")


# ======================================================================
# API principal
# ======================================================================
def asrf_capital(
    portfolio: "Portfolio",
    q: float = 0.999,
    rho_default: float = 0.15,
) -> AsrfResult:
    """Capital econômico analítico (ASRF/Vasicek) por segmento e agregado.

    Parameters
    ----------
    portfolio:
        A carteira (:class:`~yggdrasil.credit_risk.capital.portfolio.Portfolio`).
    q:
        Nível de confiança (padrão 99,9%, referência de Basileia).
    rho_default:
        Correlação de ativos usada nos segmentos cujo ``rho`` é ``None``.

    Returns
    -------
    AsrfResult
        Com o detalhe por segmento (``per_segment``) e os agregados. O capital
        econômico da carteira é a **soma** dos capitais dos segmentos — o ASRF é
        aditivo por construção e, portanto, **não** captura diversificação entre
        produtos (para isso, use :meth:`Portfolio.simulate`).
    """
    if not (0.0 < q < 1.0):
        raise ValueError(f"q deve estar em (0, 1); recebido {q!r}.")

    pds = portfolio.pds()
    lgds = portfolio.lgds()
    eads = portfolio.eads()
    rhos = portfolio.rhos(default=rho_default)

    cond = conditional_pd(pds, rhos, q)
    k = lgds * (cond - pds)                 # capital por unidade de EAD
    el_seg = pds * lgds * eads
    cap_seg = k * eads                      # capital econômico do segmento
    var_seg = el_seg + cap_seg              # perda condicional total no quantil

    per_segment = pd.DataFrame({
        "segmento": portfolio.segment_names(),
        "produto": [s.product for s in portfolio.segments],
        "PD": pds,
        "LGD": lgds,
        "EAD": eads,
        "rho": rhos,
        "cond_pd": cond,
        "K": k,
        "EL": el_seg,
        "capital": cap_seg,
        "VaR": var_seg,
    })

    el = float(el_seg.sum())
    ec = float(cap_seg.sum())
    return AsrfResult(
        q=float(q), per_segment=per_segment,
        expected_loss=el, economic_capital=ec, value_at_risk=el + ec,
    )


__all__ = ["conditional_pd", "capital_ratio", "asrf_capital", "AsrfResult"]
