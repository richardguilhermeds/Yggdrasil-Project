"""
Contrato de dados do modelo de capital econômico: segmento e carteira
=====================================================================
A **segmentação é a espinha dorsal do modelo** (Seção 5 do guia): os segmentos
devem ser homogêneos em comportamento de risco e ter volume suficiente para
estimação estável. Cada motor de cálculo (ASRF analítico, simulação de Monte
Carlo multifatorial, CreditRisk+) consome a mesma estrutura:

* :class:`Segment` — um segmento homogêneo com seus parâmetros de capital
  econômico: PD **TTC**, LGD **downturn**, EAD/CCF **downturn**, correlação de
  ativos ``rho`` e o fator sistêmico ao qual o segmento está ligado.
* :class:`Portfolio` — a coleção de segmentos mais a **matriz de correlação
  entre fatores**, que produz o benefício de diversificação entre produtos
  (cartão, consignado, veículos) que o Pilar 1 ignora por construção.

Atenção à calibração (Tabela 2 do guia): os parâmetros aqui são os de **capital
econômico** — PD *through-the-cycle*, LGD/CCF *downturn* —, e **não** os
*point-in-time* de provisão. Usar parâmetros PIT diretamente subestima o risco
no topo do ciclo e o superestima no fundo.

O objeto :class:`Portfolio` é um contêiner de dados com métodos finos que
**delegam** aos motores (importados sob demanda para evitar dependência
circular): :meth:`~Portfolio.asrf_capital`, :meth:`~Portfolio.simulate` e
:meth:`~Portfolio.creditrisk_plus`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # evita import em runtime (sem custo, só para type-checkers)
    from .asrf import AsrfResult
    from .measures import LossDistribution
    from .monte_carlo import SimulationResult


# ======================================================================
# Segmento homogêneo
# ======================================================================
@dataclass
class Segment:
    """Um segmento homogêneo de risco com parâmetros de **capital econômico**.

    Parameters
    ----------
    name:
        Identificador do segmento (ex.: ``"cartao_score_alto_revolver"``).
    pd:
        Probabilidade de *default* em 1 ano, calibrada **TTC** (média de longo
        prazo do ciclo), em ``[0, 1]``.
    lgd:
        *Loss given default* esperada, calibrada **downturn**, em ``[0, 1]``.
    ead:
        Exposição no *default* (unidades monetárias), já com CCF/LEQ *downturn*
        aplicado ao limite não utilizado nos rotativos. Deve ser ``> 0``.
    rho:
        Correlação de ativos **intra-segmento** (parâmetro de Vasicek), em
        ``[0, 1)``. É o parâmetro que mais move o resultado. Se ``None``, o
        motor pode derivá-la da fórmula regulatória do produto
        (:mod:`~yggdrasil.credit_risk.capital.parameters`).
    n_obligors:
        Número de devedores no segmento. Governa a **granularidade**: com muitos
        devedores o risco idiossincrático se dilui e sobra o risco sistêmico
        (hipótese do ASRF); com poucos, há risco de concentração de nomes que só
        a simulação captura.
    factor:
        Nome do **fator sistêmico** ao qual o segmento está exposto (ex.:
        ``"cartao"``). Segmentos que compartilham o fator co-movem
        perfeitamente no componente sistêmico; a diversificação entre fatores
        vem da matriz de correlação da carteira. Se ``None``, usa ``product`` ou
        ``name``.
    product:
        Produto ao qual o segmento pertence (``"cartao"``, ``"consignado"``,
        ``"veiculos"``, ...). Usado para agregação de reporte e para presets.
    lgd_vol:
        Desvio-padrão da LGD para **LGD estocástica** na simulação (``0`` =
        LGD determinística). Relevante em produtos com garantia (veículos), onde
        a severidade é correlacionada ao ciclo.
    weight:
        Fator de carga (``loading``) do segmento sobre o seu fator sistêmico,
        em ``[0, 1]``. Por padrão ``1.0`` (a correlação de ativos é ``rho``).
        Reservado para estruturas multifator mais ricas.
    """

    name: str
    pd: float
    lgd: float
    ead: float
    rho: Optional[float] = None
    n_obligors: int = 1
    factor: Optional[str] = None
    product: Optional[str] = None
    lgd_vol: float = 0.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.pd <= 1.0):
            raise ValueError(f"[{self.name}] pd deve estar em [0, 1]; recebido {self.pd!r}.")
        if not (0.0 <= self.lgd <= 1.0):
            raise ValueError(f"[{self.name}] lgd deve estar em [0, 1]; recebido {self.lgd!r}.")
        if self.ead <= 0:
            raise ValueError(f"[{self.name}] ead deve ser > 0; recebido {self.ead!r}.")
        if self.rho is not None and not (0.0 <= self.rho < 1.0):
            raise ValueError(f"[{self.name}] rho deve estar em [0, 1); recebido {self.rho!r}.")
        if self.n_obligors < 1:
            raise ValueError(f"[{self.name}] n_obligors deve ser >= 1; recebido {self.n_obligors!r}.")
        if self.lgd_vol < 0:
            raise ValueError(f"[{self.name}] lgd_vol deve ser >= 0; recebido {self.lgd_vol!r}.")
        # Preenche o fator com uma escolha sensata quando omitido.
        if self.factor is None:
            self.factor = self.product or self.name

    @property
    def expected_loss(self) -> float:
        """Perda esperada do segmento: ``PD × LGD × EAD``."""
        return self.pd * self.lgd * self.ead

    def with_params(self, **overrides) -> "Segment":
        """Cópia do segmento com parâmetros substituídos (para sensibilidades)."""
        base = {
            "name": self.name, "pd": self.pd, "lgd": self.lgd, "ead": self.ead,
            "rho": self.rho, "n_obligors": self.n_obligors, "factor": self.factor,
            "product": self.product, "lgd_vol": self.lgd_vol, "weight": self.weight,
        }
        base.update(overrides)
        return Segment(**base)


# ======================================================================
# Carteira
# ======================================================================
@dataclass
class Portfolio:
    """Carteira = segmentos homogêneos + estrutura de fatores sistêmicos.

    Parameters
    ----------
    segments:
        Lista de :class:`Segment`.
    factor_corr:
        Matriz de correlação **entre fatores** (``F × F``, simétrica, diagonal
        1). É ela que produz o benefício de diversificação entre produtos e é
        ela que deve ser estressada na validação (correlações sobem em crise).
        Se ``None``, assume fatores independentes (identidade) — o que **anula**
        a diversificação e serve de caso-base conservador.
    factor_names:
        Nomes dos fatores na ordem das linhas/colunas de ``factor_corr``. Se
        ``None``, são inferidos dos segmentos (ordem de primeira aparição).
    name:
        Rótulo da carteira.
    """

    segments: List[Segment]
    factor_corr: Optional[np.ndarray] = None
    factor_names: Optional[List[str]] = None
    name: str = "carteira"

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("A carteira precisa de ao menos um segmento.")
        nomes = [s.name for s in self.segments]
        if len(set(nomes)) != len(nomes):
            raise ValueError("Nomes de segmento duplicados na carteira.")
        # Fatores presentes, na ordem de primeira aparição.
        inferidos: List[str] = []
        for s in self.segments:
            if s.factor not in inferidos:
                inferidos.append(s.factor)
        if self.factor_names is None:
            self.factor_names = inferidos
        else:
            faltando = set(inferidos) - set(self.factor_names)
            if faltando:
                raise ValueError(
                    f"factor_names não cobre todos os fatores usados: faltam {sorted(faltando)}.")
        F = len(self.factor_names)
        if self.factor_corr is None:
            self.factor_corr = np.eye(F)
        else:
            M = np.asarray(self.factor_corr, dtype=float)
            if M.shape != (F, F):
                raise ValueError(
                    f"factor_corr deve ser {F}×{F} (um fator por nome); recebido {M.shape}.")
            if not np.allclose(M, M.T, atol=1e-8):
                raise ValueError("factor_corr deve ser simétrica.")
            if not np.allclose(np.diag(M), 1.0, atol=1e-6):
                raise ValueError("factor_corr deve ter diagonal 1 (é matriz de correlação).")
            self.factor_corr = M

    # ------------------------------------------------------------------
    # Consultas básicas
    # ------------------------------------------------------------------
    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def n_factors(self) -> int:
        return len(self.factor_names)

    def factor_index(self) -> Dict[str, int]:
        """Mapa ``nome do fator → índice`` na matriz de correlação."""
        return {f: i for i, f in enumerate(self.factor_names)}

    def factor_of(self) -> np.ndarray:
        """Vetor (por segmento) com o índice do fator sistêmico de cada segmento."""
        idx = self.factor_index()
        return np.array([idx[s.factor] for s in self.segments], dtype=int)

    def expected_loss(self) -> float:
        """Perda esperada da carteira: ``Σ PD_i × LGD_i × EAD_i``."""
        return float(sum(s.expected_loss for s in self.segments))

    def total_ead(self) -> float:
        """Exposição total (soma das EADs)."""
        return float(sum(s.ead for s in self.segments))

    # Vetores paralelos aos segmentos (úteis para os motores vetorizados) -----
    def pds(self) -> np.ndarray:
        return np.array([s.pd for s in self.segments], dtype=float)

    def lgds(self) -> np.ndarray:
        return np.array([s.lgd for s in self.segments], dtype=float)

    def eads(self) -> np.ndarray:
        return np.array([s.ead for s in self.segments], dtype=float)

    def rhos(self, default: float = 0.0) -> np.ndarray:
        """Correlações de ativos por segmento; ``default`` onde ``rho`` é ``None``."""
        return np.array([default if s.rho is None else s.rho for s in self.segments], dtype=float)

    def n_obligors(self) -> np.ndarray:
        return np.array([s.n_obligors for s in self.segments], dtype=int)

    def lgd_vols(self) -> np.ndarray:
        return np.array([s.lgd_vol for s in self.segments], dtype=float)

    def segment_names(self) -> List[str]:
        return [s.name for s in self.segments]

    # ------------------------------------------------------------------
    # Delegações aos motores (import sob demanda: evita ciclo de import)
    # ------------------------------------------------------------------
    def asrf_capital(self, q: float = 0.999, **kwargs) -> "AsrfResult":
        """Capital econômico **analítico** (ASRF/Vasicek) — a "versão 1".

        Ver :func:`yggdrasil.credit_risk.capital.asrf.asrf_capital`.
        """
        from .asrf import asrf_capital
        return asrf_capital(self, q=q, **kwargs)

    def simulate(
        self, n_scenarios: int = 100_000, q: float = 0.999, seed: Optional[int] = None, **kwargs
    ) -> "SimulationResult":
        """Simulação de **Monte Carlo multifatorial** — a "versão 2".

        Ver :func:`yggdrasil.credit_risk.capital.monte_carlo.simulate`.
        """
        from .monte_carlo import simulate
        return simulate(self, n_scenarios=n_scenarios, q=q, seed=seed, **kwargs)

    def creditrisk_plus(self, **kwargs) -> "LossDistribution":
        """Distribuição de perdas **analítica** (CreditRisk+, benchmark atuarial).

        Ver :func:`yggdrasil.credit_risk.capital.creditrisk_plus.creditrisk_plus`.
        """
        from .creditrisk_plus import creditrisk_plus
        return creditrisk_plus(self, **kwargs)

    # ------------------------------------------------------------------
    # Resumo tabular
    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """Uma linha por segmento com parâmetros e perda esperada."""
        linhas = []
        for s in self.segments:
            linhas.append(
                {
                    "segmento": s.name,
                    "produto": s.product,
                    "fator": s.factor,
                    "PD": s.pd,
                    "LGD": s.lgd,
                    "EAD": s.ead,
                    "rho": s.rho,
                    "n_obligors": s.n_obligors,
                    "lgd_vol": s.lgd_vol,
                    "EL": s.expected_loss,
                }
            )
        df = pd.DataFrame(linhas)
        df["EL_share"] = df["EL"] / df["EL"].sum() if df["EL"].sum() > 0 else np.nan
        return df

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        *,
        name_col: str = "segmento",
        pd_col: str = "pd",
        lgd_col: str = "lgd",
        ead_col: str = "ead",
        rho_col: Optional[str] = "rho",
        n_obligors_col: Optional[str] = "n_obligors",
        factor_col: Optional[str] = "fator",
        product_col: Optional[str] = "produto",
        lgd_vol_col: Optional[str] = "lgd_vol",
        factor_corr: Optional[np.ndarray] = None,
        factor_names: Optional[List[str]] = None,
        name: str = "carteira",
    ) -> "Portfolio":
        """Constrói a carteira a partir de um ``DataFrame`` de segmentos.

        Colunas ausentes (quando o parâmetro aponta para uma coluna inexistente)
        caem no default do :class:`Segment`.
        """
        def _get(row, col, default=None):
            if col is None or col not in df.columns:
                return default
            val = row[col]
            return default if pd.isna(val) else val

        segments: List[Segment] = []
        for _, row in df.iterrows():
            segments.append(
                Segment(
                    name=str(row[name_col]),
                    pd=float(row[pd_col]),
                    lgd=float(row[lgd_col]),
                    ead=float(row[ead_col]),
                    rho=_get(row, rho_col),
                    n_obligors=int(_get(row, n_obligors_col, 1)),
                    factor=_get(row, factor_col),
                    product=_get(row, product_col),
                    lgd_vol=float(_get(row, lgd_vol_col, 0.0)),
                )
            )
        return cls(segments, factor_corr=factor_corr, factor_names=factor_names, name=name)


__all__ = ["Segment", "Portfolio"]
