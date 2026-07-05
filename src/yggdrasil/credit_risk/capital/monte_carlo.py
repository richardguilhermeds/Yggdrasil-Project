"""
Motor v2 — Simulação de Monte Carlo multifatorial
=================================================
A extensão natural do ASRF e o **padrão de mercado para capital econômico
interno** (Seção 3.2 do guia). A lógica:

1. Define-se um conjunto de **fatores sistêmicos correlacionados** (um por
   produto/segmento — cartão, consignado, veículos), com uma matriz de
   correlação entre fatores.
2. Em cada cenário simulado, sorteiam-se os fatores; condicionam-se as PDs de
   cada segmento ao cenário (fórmula de Vasicek); sorteiam-se (ou tomam-se pelo
   limite de grandes números) os *defaults* e, se desejado, as **LGDs
   estocásticas** correlacionadas ao ciclo; agrega-se a perda total.
3. Repetindo dezenas/centenas de milhares de vezes, obtém-se a **distribuição
   empírica completa** de perdas, da qual se extraem VaR, ES e as contribuições
   de cada produto.

Por que multifatorial importa: com um fator por produto e a matriz de
correlação entre fatores, o modelo captura o **benefício de diversificação** —
cartão, consignado e veículos não estressam ao mesmo tempo com a mesma
intensidade —, que o Pilar 1 ignora por construção.

Validação de sanidade (guia, bloco E): com **um único fator** e carteira
**granular** (``granular=True``), a simulação reproduz o ASRF analítico.

Em ambiente Databricks/PySpark a simulação paraleliza bem (cada cenário é
independente); aqui a implementação é vetorizada em NumPy, adequada ao nível de
**segmento homogêneo** (não de contrato).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .measures import DEFAULT_CONFIDENCE, LossDistribution

if TYPE_CHECKING:
    from .portfolio import Portfolio


# ======================================================================
# Álgebra: raiz de Cholesky robusta a matrizes quase-singulares
# ======================================================================
def _safe_cholesky(corr: np.ndarray) -> np.ndarray:
    """Fator ``L`` tal que ``L @ L.T ≈ corr``.

    Tenta Cholesky; se a matriz não for positiva-definida (comum em matrizes de
    correlação estimadas), projeta para a correlação positiva-definida mais
    próxima por *clipping* de autovalores e refatoriza. Ver
    :func:`yggdrasil.credit_risk.capital.correlation.nearest_correlation`.
    """
    corr = np.asarray(corr, dtype=float)
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Projeção espectral: zera autovalores negativos e renormaliza a diagonal.
        vals, vecs = np.linalg.eigh(corr)
        vals = np.clip(vals, 1e-10, None)
        A = vecs @ np.diag(vals) @ vecs.T
        d = np.sqrt(np.clip(np.diag(A), 1e-12, None))
        A = A / np.outer(d, d)
        A = (A + A.T) / 2.0
        try:
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:  # pragma: no cover - último recurso
            return np.linalg.cholesky(A + 1e-8 * np.eye(A.shape[0]))


# ======================================================================
# Resultado da simulação
# ======================================================================
@dataclass
class SimulationResult:
    """Saída da simulação: distribuição de perdas + perdas por segmento/cenário.

    ``segment_losses`` (``n_scenarios × n_segments``) é o insumo da **alocação de
    Euler** (contribuição condicional à cauda) — ver :mod:`.allocation`.
    """

    losses: np.ndarray                    # (n_scenarios,) perda total por cenário
    segment_losses: Optional[np.ndarray]  # (n_scenarios, n_segments) ou None
    segment_names: List[str]
    q: float
    expected_loss: float                  # EL analítica (Σ PD·LGD·EAD)
    n_scenarios: int
    seed: Optional[int] = None
    metric: str = "var"
    _dist: Optional[LossDistribution] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def distribution(self) -> LossDistribution:
        """A distribuição de perdas (usa a EL analítica exata como ``el``)."""
        if self._dist is None:
            self._dist = LossDistribution(
                self.losses, expected=self.expected_loss, name="monte_carlo")
        return self._dist

    def _q(self, q: Optional[float]) -> float:
        return self.q if q is None else float(q)

    def var(self, q: Optional[float] = None) -> float:
        return self.distribution().var(self._q(q))

    def es(self, q: Optional[float] = None) -> float:
        return self.distribution().es(self._q(q))

    def economic_capital(self, q: Optional[float] = None, metric: Optional[str] = None) -> float:
        return self.distribution().economic_capital(self._q(q), metric or self.metric)

    # ------------------------------------------------------------------
    def allocate(self, q: Optional[float] = None, metric: str = "es",
                 alpha: float = 0.05) -> pd.DataFrame:
        """Alocação de Euler do capital pelos segmentos (contribuição à cauda).

        Ver :func:`yggdrasil.credit_risk.capital.allocation.euler_allocation`.
        """
        from .allocation import euler_allocation
        return euler_allocation(self, q=self._q(q), metric=metric, alpha=alpha)

    def diversification_benefit(self, q: Optional[float] = None) -> dict:
        """Benefício de diversificação: capital **isolado** (soma dos CE de cada
        segmento como se estivesse sozinho) menos o capital **integrado** da
        carteira. É a informação gerencial que o Pilar 1 não dá.
        """
        if self.segment_losses is None:
            raise ValueError("segment_losses não foi armazenado (store_segment_losses=False).")
        qq = self._q(q)
        standalone = 0.0
        for j in range(self.segment_losses.shape[1]):
            col = self.segment_losses[:, j]
            standalone += float(np.quantile(col, qq) - col.mean())
        integrated = self.economic_capital(qq, metric="var")
        return {
            "capital_isolado": standalone,
            "capital_integrado": integrated,
            "beneficio_diversificacao": standalone - integrated,
            "beneficio_pct": (standalone - integrated) / standalone if standalone > 0 else np.nan,
        }

    def summary(self) -> pd.DataFrame:
        d = self.distribution()
        return pd.DataFrame([{
            "nivel_confianca": self.q,
            "n_cenarios": self.n_scenarios,
            "EL": d.el,
            "VaR": d.var(self.q),
            "ES": d.es(self.q),
            "CE_var": d.var(self.q) - d.el,
            "CE_es": d.es(self.q) - d.el,
        }])

    def __repr__(self) -> str:  # pragma: no cover
        d = self.distribution()
        return (f"SimulationResult(n={self.n_scenarios}, q={self.q}, "
                f"EL={d.el:,.2f}, VaR={d.var(self.q):,.2f}, ES={d.es(self.q):,.2f})")


# ======================================================================
# Motor
# ======================================================================
def simulate(
    portfolio: "Portfolio",
    n_scenarios: int = 100_000,
    q: float = DEFAULT_CONFIDENCE,
    seed: Optional[int] = None,
    *,
    granular: bool = True,
    stochastic_lgd: bool = False,
    pd_lgd_corr: float = 0.0,
    rho_default: float = 0.15,
    antithetic: bool = False,
    store_segment_losses: bool = True,
    block_size: int = 50_000,
) -> SimulationResult:
    """Simula a distribuição de perdas da carteira por Monte Carlo multifatorial.

    Parameters
    ----------
    portfolio:
        A carteira. Cada segmento carrega no seu fator sistêmico com ``√rho``; a
        dependência entre fatores vem de ``portfolio.factor_corr``.
    n_scenarios:
        Número de cenários. O quantil de cauda (99,9%) exige muitos cenários —
        ver :func:`yggdrasil.credit_risk.capital.validation.convergence`.
    q:
        Nível de confiança de referência do resultado.
    seed:
        Semente do gerador (reprodutibilidade).
    granular:
        ``True`` (padrão) — usa a **PD condicional** diretamente como fração de
        *default* do segmento (limite de grandes números: carteira granular).
        Reproduz o ASRF quando há um único fator. ``False`` — sorteia o número de
        *defaults* por ``Binomial(n_obligors, p)``, capturando risco
        idiossincrático/de **concentração** de nomes.
    stochastic_lgd:
        Se ``True``, a LGD é estocástica nos segmentos com ``lgd_vol > 0``.
    pd_lgd_corr:
        Correlação **adversa** PD–LGD em ``[0, 1)`` (relevante em veículos): em
        cenários ruins, a severidade sobe junto com os *defaults*. Só tem efeito
        com ``stochastic_lgd=True``.
    rho_default:
        ``rho`` usado nos segmentos com ``rho=None``.
    antithetic:
        Variáveis **antitéticas** (redução de variância): metade dos sorteios do
        fator usa ``+z`` e a outra metade ``−z``.
    store_segment_losses:
        Armazena a matriz perda-por-segmento (necessária para alocação de Euler
        e benefício de diversificação). Desligue para poupar memória.
    block_size:
        Tamanho do bloco de cenários processados por vez (controle de memória).

    Returns
    -------
    SimulationResult
    """
    from .portfolio import Portfolio  # noqa: F401 (garante o tipo em runtime)

    if n_scenarios < 1:
        raise ValueError("n_scenarios deve ser >= 1.")
    if not (0.0 < q < 1.0):
        raise ValueError(f"q deve estar em (0, 1); recebido {q!r}.")
    if not (0.0 <= pd_lgd_corr < 1.0):
        raise ValueError("pd_lgd_corr deve estar em [0, 1).")

    rng = np.random.default_rng(seed)

    pds = portfolio.pds()
    lgds = portfolio.lgds()
    eads = portfolio.eads()
    rhos = portfolio.rhos(default=rho_default)
    n_obl = portfolio.n_obligors()
    lgd_vols = portfolio.lgd_vols()
    fac_of = portfolio.factor_of()                      # (n_seg,) índice do fator
    n_seg = portfolio.n_segments
    F = portfolio.n_factors

    L = _safe_cholesky(portfolio.factor_corr)           # (F, F)
    inv_pd = norm.ppf(np.clip(pds, 1e-12, 1 - 1e-12))   # limiar de default por segmento
    sqrt_rho = np.sqrt(rhos)
    sqrt_1mrho = np.sqrt(1.0 - rhos)
    ead_per_obl = eads / np.maximum(n_obl, 1)

    total_losses = np.empty(n_scenarios, dtype=float)
    seg_losses = (np.empty((n_scenarios, n_seg), dtype=float)
                  if store_segment_losses else None)

    start = 0
    while start < n_scenarios:
        m = min(block_size, n_scenarios - start)
        # ---- fatores sistêmicos correlacionados (m, F) --------------------
        if antithetic:
            half = (m + 1) // 2
            z0 = rng.standard_normal((half, F))
            z = np.vstack([z0, -z0])[:m]
        else:
            z = rng.standard_normal((m, F))
        Y = z @ L.T                                     # cov(Y) = factor_corr
        Y_seg = Y[:, fac_of]                            # (m, n_seg) fator de cada segmento

        # ---- PD condicional ao cenário (Vasicek): baixo Y = ruim ----------
        # p = N( (N⁻¹(PD) − √rho · Y) / √(1−rho) )
        cond_pd = norm.cdf((inv_pd[None, :] - sqrt_rho[None, :] * Y_seg) / sqrt_1mrho[None, :])

        # ---- fração/nº de defaults ---------------------------------------
        if granular:
            default_frac = cond_pd                       # limite de grandes números
        else:
            draws = rng.binomial(n_obl[None, :].repeat(m, axis=0), cond_pd)
            default_frac = draws / np.maximum(n_obl[None, :], 1)

        # ---- LGD (determinística ou estocástica, correlacionada ao ciclo) -
        if stochastic_lgd and np.any(lgd_vols > 0):
            zeta = rng.standard_normal((m, n_seg))
            # latente da LGD: componente sistêmica (−Y: cenário ruim → LGD alta)
            # + idiossincrática. corr(latente_LGD, fator) = pd_lgd_corr.
            lgd_lat = (-pd_lgd_corr * Y_seg
                       + np.sqrt(1.0 - pd_lgd_corr ** 2) * zeta)
            lgd_eff = np.clip(lgds[None, :] + lgd_vols[None, :] * lgd_lat, 0.0, 1.0)
        else:
            lgd_eff = lgds[None, :]

        # ---- perda por segmento e total ----------------------------------
        block = default_frac * lgd_eff * eads[None, :]   # (m, n_seg)
        if seg_losses is not None:
            seg_losses[start:start + m, :] = block
        total_losses[start:start + m] = block.sum(axis=1)
        start += m

    el = portfolio.expected_loss()
    return SimulationResult(
        losses=total_losses,
        segment_losses=seg_losses,
        segment_names=portfolio.segment_names(),
        q=float(q),
        expected_loss=el,
        n_scenarios=int(n_scenarios),
        seed=seed,
    )


__all__ = ["simulate", "SimulationResult"]
