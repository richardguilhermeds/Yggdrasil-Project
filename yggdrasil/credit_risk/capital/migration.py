"""
Motor CreditMetrics — migração de rating / estágio (mark-to-market)
===================================================================
Extensão do modelo estrutural que captura perdas por **migração**, não só por
*default* (Seção 3.3 do guia). No ASRF/Vasicek uma contraparte só gera perda se
quebrar; no CreditMetrics ela também perde valor ao **piorar de rating** (o
*spread* exigido sobe e o título é reprecificado para baixo), e ganha ao
melhorar. Cada contraparte migra entre classes de risco segundo uma **matriz de
transição**, e a carteira é **revalorizada** em cada cenário.

A mecânica é uma **cópula gaussiana de fator único** (o mesmo motor estrutural
do ASRF, generalizado para vários estados de destino):

1. O retorno latente do ativo de cada contraparte é
   ``X = √ρ · Y + √(1−ρ) · ε``, com ``Y`` o fator sistêmico comum (um por
   cenário) e ``ε`` o ruído idiossincrático — ambos ``N(0, 1)``.
2. Cada **linha** da matriz de transição (o rating de origem) é convertida em
   **limiares** ``z`` pela inversa da normal acumulada das probabilidades
   acumuladas, na convenção CreditMetrics: o *default* é o estado extremo
   (o pior), e ratings melhores ocupam a cauda superior de ``X``.
3. O valor de ``X`` frente aos limiares determina o **rating de destino**; a
   contraparte é revalorizada para o ``value`` daquele destino e a perda é a
   queda de valor face à referência (o valor no rating de origem).

Aplicabilidade (guia): indicado para **títulos e grandes corporativos**, onde a
reprecificação por migração é material. Para varejo massificado mantido até o
vencimento, o modo *default-only* costuma bastar — e é reproduzido aqui como
caso particular (:func:`two_state_matrix`), útil também para conectar **capital
e provisão** via migração entre **estágios** (Stage 1/2/3, IFRS 9 /
Resolução CMN 4.966/2021).

Contexto regulatório: Resolução CMN 4.557/2017 (ICAAP), 4.966/2021 (estágios) e
o arcabouço IRB de Basileia (modelo estrutural de fator único subjacente).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np
from scipy.stats import norm

from .measures import DEFAULT_CONFIDENCE, LossDistribution

ArrayLike = Union[Sequence[float], np.ndarray]
ArrayLikeInt = Union[Sequence[int], np.ndarray]


# ======================================================================
# Limiares da cópula gaussiana (convenção CreditMetrics)
# ======================================================================
def migration_thresholds(transition_matrix: np.ndarray) -> np.ndarray:
    """Converte cada linha da matriz de transição em **limiares** ``z`` da cópula.

    Convenção CreditMetrics: as colunas estão ordenadas do **melhor** rating de
    destino (coluna 0) ao **pior** (última coluna, tipicamente o *default*), e o
    retorno latente do ativo ``X ~ N(0, 1)`` mapeia estados assim — quanto maior
    ``X``, melhor o rating de destino:

    * a probabilidade de cair no **default** (estado extremo inferior) é a massa
      de ``X`` **abaixo** do limiar mais baixo ``z_D = Φ⁻¹(p_D)``;
    * a probabilidade do penúltimo pior estado é a massa entre ``z_D`` e o
      próximo limiar; e assim por diante, acumulando a partir do pior estado.

    Formalmente, para a linha ``i`` com probabilidades ``p_{i,·}`` (do melhor ao
    pior), define-se a acumulada **a partir do pior** e toma-se a inversa normal.
    Retornam-se ``n_estados − 1`` limiares por linha (as fronteiras entre os
    ``n_estados`` intervalos); um valor de ``X`` **abaixo** do primeiro limiar
    corresponde ao pior estado (default), e **acima** do último, ao melhor.

    Parameters
    ----------
    transition_matrix:
        Matriz ``(n_origem, n_destino)`` de probabilidades de migração. Cada
        linha deve somar ~1 e ter entradas em ``[0, 1]``. As colunas seguem a
        ordem do **melhor** ao **pior** rating de destino (default por último).

    Returns
    -------
    np.ndarray
        Matriz ``(n_origem, n_destino − 1)`` de limiares ``z``, crescentes ao
        longo de cada linha. Estados de destino com probabilidade nula na origem
        recebem limiares degenerados (``±inf``), consistentes com massa zero.

    Notes
    -----
    Os limiares dependem apenas da **linha** (rating de origem): a correlação de
    ativos ``ρ`` e o fator sistêmico entram na simulação, não aqui.
    """
    tm = np.asarray(transition_matrix, dtype=float)
    if tm.ndim != 2:
        raise ValueError("transition_matrix deve ser uma matriz 2-D (n_origem × n_destino).")
    n_from, n_to = tm.shape
    if n_to < 2:
        raise ValueError("A matriz de transição precisa de ao menos 2 estados de destino.")
    if np.any(tm < -1e-9) or np.any(tm > 1 + 1e-9):
        raise ValueError("As probabilidades da matriz de transição devem estar em [0, 1].")
    row_sums = tm.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(
            "Cada linha da matriz de transição deve somar 1; "
            f"somas encontradas: {np.round(row_sums, 6)}."
        )

    tm = np.clip(tm, 0.0, 1.0)
    # Acumula a massa a partir do PIOR estado (última coluna) em direção ao melhor:
    #   cum[:, k] = P(destino pertence aos k+1 piores estados)
    #            = p_pior + p_{pior-1} + … (k+1 termos).
    # As fronteiras da cópula em X são Φ⁻¹ dessas acumuladas, para as primeiras
    # n_to−1 (a última acumulada é 1 e não gera fronteira). O resultado é
    # **crescente** ao longo da linha: a primeira fronteira (baixa) separa o
    # default dos demais; a última (alta) separa o melhor rating do resto.
    cum_from_worst = np.cumsum(tm[:, ::-1], axis=1)            # (n_from, n_to)
    cdf_boundaries = cum_from_worst[:, : n_to - 1]            # (n_from, n_to-1), crescente
    # Φ⁻¹: pouca massa nos piores estados → fronteira baixa (X mais negativo).
    thresholds = norm.ppf(np.clip(cdf_boundaries, 0.0, 1.0))
    return thresholds


def _bucket_from_return(x: np.ndarray, thresholds_row: np.ndarray) -> np.ndarray:
    """Mapeia retornos do ativo ``x`` para o índice do rating de destino.

    ``thresholds_row`` são as ``n_to−1`` fronteiras crescentes (limiares) de uma
    linha. Um retorno ``x`` **abaixo** de todos → pior estado (índice
    ``n_to−1``); **acima** de todos → melhor estado (índice ``0``). Como as
    colunas vão do melhor (0) ao pior (n_to−1), o índice de destino é
    ``n_to − 1 − #{limiares ≤ x}``.
    """
    n_to = thresholds_row.size + 1
    # Nº de limiares que x supera (searchsorted em vetor crescente).
    n_above = np.searchsorted(thresholds_row, x, side="right")
    return (n_to - 1) - n_above


# ======================================================================
# Modelo de migração
# ======================================================================
@dataclass
class MigrationModel:
    """Motor CreditMetrics de migração via cópula gaussiana de fator único.

    Parameters
    ----------
    transition_matrix:
        Matriz ``(n_ratings, n_ratings)`` de probabilidades de migração em 1 ano.
        Colunas do **melhor** rating (0) ao **pior** (último, tipicamente 'D').
        Cada linha soma ~1.
    ratings:
        Rótulos dos ratings, na mesma ordem das colunas (melhor → pior). O
        tamanho deve casar com a dimensão da matriz.
    values:
        Valor (PV) de 1 unidade de exposição em cada rating de **destino**, na
        mesma ordem. O último rating (default) tem tipicamente ``1 − LGD``. A
        **referência** para a perda é o valor no rating de **origem** de cada
        contraparte (mark-to-market puro): perda = valor_origem − valor_destino.
    rho:
        Correlação de ativos ``ρ`` (peso do fator sistêmico comum). Padrão 0,15.
    """

    transition_matrix: np.ndarray
    ratings: List[str]
    values: np.ndarray
    rho: float = 0.15
    _thresholds: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.transition_matrix = np.asarray(self.transition_matrix, dtype=float)
        self.values = np.asarray(self.values, dtype=float).ravel()
        n = self.transition_matrix.shape[0]
        if self.transition_matrix.shape[1] != n:
            raise ValueError(
                "transition_matrix deve ser quadrada (n_ratings × n_ratings); "
                f"recebida {self.transition_matrix.shape}."
            )
        if len(self.ratings) != n:
            raise ValueError(
                f"ratings tem {len(self.ratings)} rótulos, mas a matriz tem {n} estados."
            )
        if self.values.size != n:
            raise ValueError(
                f"values tem {self.values.size} entradas, mas a matriz tem {n} estados."
            )
        if not (0.0 <= self.rho < 1.0):
            raise ValueError(f"rho deve estar em [0, 1); recebido {self.rho!r}.")
        # Limiares por linha (rating de origem) — calculados uma única vez.
        self._thresholds = migration_thresholds(self.transition_matrix)

    # ------------------------------------------------------------------
    @property
    def n_ratings(self) -> int:
        return self.transition_matrix.shape[0]

    @property
    def thresholds(self) -> np.ndarray:
        """Limiares ``z`` da cópula por rating de origem (``n_ratings × n_ratings−1``)."""
        return self._thresholds

    # ------------------------------------------------------------------
    def simulate(
        self,
        exposures: ArrayLike,
        ratings_idx: ArrayLikeInt,
        n_scenarios: int = 100_000,
        q: float = DEFAULT_CONFIDENCE,
        seed: Optional[int] = None,
        lgd: Union[float, ArrayLike, None] = None,
    ) -> LossDistribution:
        """Simula a distribuição de perdas por **migração** (mark-to-market).

        Para cada cenário sorteia-se um fator sistêmico ``Y ~ N(0, 1)`` comum a
        todas as contrapartes; o retorno do ativo de cada exposição é
        ``X = √ρ · Y + √(1−ρ) · ε`` com ``ε ~ N(0, 1)`` idiossincrático. Os
        limiares da **linha do rating de origem** definem o rating de destino de
        cada exposição, que é então **revalorizada** pelo ``values`` do destino.
        A perda do cenário é a soma, sobre as exposições, de
        ``exposição × (valor_referência − valor_destino)``, onde o valor de
        referência é o valor no rating de **origem** (ou o **maior** valor da
        curva, quando ``lgd`` sobrepõe o valor de default — ver abaixo).

        Parameters
        ----------
        exposures:
            Vetor ``(n_exp,)`` de exposições (EAD). Um valor por contraparte.
        ratings_idx:
            Vetor ``(n_exp,)`` de índices inteiros do rating de **origem** de
            cada exposição (0 = melhor … n_ratings−1 = pior).
        n_scenarios:
            Número de cenários de Monte Carlo. O quantil de cauda (99,9%) exige
            muitos cenários para estabilizar.
        q:
            Nível de confiança de referência anexado à distribuição.
        seed:
            Semente do gerador (reprodutibilidade).
        lgd:
            **Opcional.** Se fornecido (escalar ou vetor ``(n_exp,)``), sobrepõe
            o valor do rating de **default** (último) por ``1 − lgd`` **apenas
            no cálculo da perda** — conveniente no modo *default-only* para
            reproduzir exatamente ``PD·LGD·EAD`` sem reescrever ``values``. Não
            afeta os demais ratings de destino.

        Returns
        -------
        LossDistribution
            Amostra de ``n_scenarios`` perdas agregadas, com a EL analítica
            (``Σ P(migração)·(valor_ref − valor_destino)·exposição``) fixada como
            ``expected`` para reduzir o ruído amostral na média.
        """
        exposures = np.asarray(exposures, dtype=float).ravel()
        ratings_idx = np.asarray(ratings_idx).ravel().astype(int)
        n_exp = exposures.size
        if ratings_idx.size != n_exp:
            raise ValueError("exposures e ratings_idx devem ter o mesmo tamanho.")
        if n_exp == 0:
            raise ValueError("É preciso ao menos uma exposição.")
        if np.any(ratings_idx < 0) or np.any(ratings_idx >= self.n_ratings):
            raise ValueError(
                f"ratings_idx deve estar em [0, {self.n_ratings - 1}]."
            )
        if n_scenarios < 1:
            raise ValueError("n_scenarios deve ser >= 1.")
        if not (0.0 < q < 1.0):
            raise ValueError(f"q deve estar em (0, 1); recebido {q!r}.")

        # ---- vetor de valores de destino por exposição -------------------
        # values_dest[e, k] = valor de 1 unidade da exposição e no rating destino k.
        # Por padrão todas usam a mesma curva `values`; `lgd` só reescreve o
        # último estado (default) por exposição.
        n_ratings = self.n_ratings
        values_dest = np.broadcast_to(self.values, (n_exp, n_ratings)).copy()
        if lgd is not None:
            lgd_arr = np.asarray(lgd, dtype=float).ravel()
            if lgd_arr.size == 1:
                lgd_arr = np.full(n_exp, float(lgd_arr))
            elif lgd_arr.size != n_exp:
                raise ValueError("lgd deve ser escalar ou ter o tamanho de exposures.")
            if np.any(lgd_arr < 0.0) or np.any(lgd_arr > 1.0):
                raise ValueError("lgd deve estar em [0, 1].")
            values_dest[:, -1] = 1.0 - lgd_arr        # valor no default = 1 − LGD

        # ---- valor de referência (mark-to-market) por exposição ----------
        # Referência = valor no rating de ORIGEM (perda relativa à posição atual).
        # Referência mark-to-market: o valor de manutenção no rating de ORIGEM
        # de cada exposição. Perda = exp · (valor_origem − valor_cenário) — assim
        # migrações para baixo geram perda e migrações para cima geram ganho
        # (perda negativa), como no CreditMetrics. No modo default-only (origem =
        # melhor rating) as perdas ficam sempre >= 0.
        value_ref = values_dest[np.arange(n_exp), ratings_idx]

        # ---- EL analítica: Σ_e exp_e · Σ_k P(orig→k) · (ref_e − val_e,k) ---
        # (fixa a média da distribuição no valor exato, reduzindo ruído amostral)
        probs = self.transition_matrix[ratings_idx, :]           # (n_exp, n_ratings)
        loss_per_state = (value_ref[:, None] - values_dest)      # (n_exp, n_ratings)
        el = float(np.sum(exposures[:, None] * probs * loss_per_state))

        # ---- limiares por exposição (linha do rating de origem) ----------
        thr = self._thresholds[ratings_idx, :]                   # (n_exp, n_ratings-1)
        sqrt_rho = float(np.sqrt(self.rho))
        sqrt_1mrho = float(np.sqrt(1.0 - self.rho))

        rng = np.random.default_rng(seed)
        losses = np.empty(n_scenarios, dtype=float)
        exp_idx = np.arange(n_exp)                               # p/ indexar values_dest

        # Bloco de cenários para controlar memória (n_scenarios × n_exp pode ser grande).
        block = max(1, min(n_scenarios, max(1, 5_000_000 // max(n_exp, 1))))
        start = 0
        while start < n_scenarios:
            m = min(block, n_scenarios - start)
            # fator sistêmico comum (m, 1) e ruído idiossincrático (m, n_exp)
            Y = rng.standard_normal((m, 1))
            eps = rng.standard_normal((m, n_exp))
            X = sqrt_rho * Y + sqrt_1mrho * eps                  # (m, n_exp)

            # rating de destino por (cenário, exposição): quantos limiares X supera.
            # dest_idx = (n_ratings-1) - #{limiares <= X}, com thr[e] crescente.
            n_above = (X[:, :, None] >= thr[None, :, :]).sum(axis=2)   # (m, n_exp)
            dest_idx = (n_ratings - 1) - n_above                       # (m, n_exp)

            # valor revalorizado: val[s, e] = values_dest[e, dest_idx[s, e]].
            # Indexação avançada (exp_idx broadcast em (m, n_exp)) resolve a curva
            # de valor específica de cada exposição sem materializar (m, n_exp, k).
            val = values_dest[exp_idx[None, :], dest_idx]             # (m, n_exp)
            loss_block = exposures[None, :] * (value_ref[None, :] - val)
            losses[start:start + m] = loss_block.sum(axis=1)
            start += m

        return LossDistribution(losses, expected=el, name="creditmetrics")


# ======================================================================
# Função de conveniência
# ======================================================================
def simulate_creditmetrics(
    transition_matrix: np.ndarray,
    ratings: List[str],
    values: ArrayLike,
    exposures: ArrayLike,
    ratings_idx: ArrayLikeInt,
    rho: float = 0.15,
    n_scenarios: int = 100_000,
    q: float = DEFAULT_CONFIDENCE,
    seed: Optional[int] = None,
    lgd: Union[float, ArrayLike, None] = None,
) -> LossDistribution:
    """Atalho equivalente a ``MigrationModel(...).simulate(...)``.

    Constrói o :class:`MigrationModel` e chama :meth:`MigrationModel.simulate`
    de uma vez, para quem quer o resultado sem manter o objeto. Ver as duas
    classes para o significado de cada parâmetro.
    """
    model = MigrationModel(
        transition_matrix=np.asarray(transition_matrix, dtype=float),
        ratings=list(ratings),
        values=np.asarray(values, dtype=float),
        rho=rho,
    )
    return model.simulate(
        exposures=exposures,
        ratings_idx=ratings_idx,
        n_scenarios=n_scenarios,
        q=q,
        seed=seed,
        lgd=lgd,
    )


# ======================================================================
# Caso particular default-only (ponte para o modo Bernoulli/ASRF)
# ======================================================================
def two_state_matrix(pd: float) -> tuple[np.ndarray, List[str], np.ndarray]:
    """Matriz de 2 estados (performing / default) para validação *default-only*.

    Devolve ``(transition_matrix, ratings, values)`` com:

    * ``transition_matrix = [[1−PD, PD], [0, 1]]`` — o estado 'P' migra para 'D'
      com probabilidade ``PD`` e o 'D' é absorvente;
    * ``ratings = ['P', 'D']`` (melhor → pior, como manda a convenção);
    * ``values = [1.0, 0.0]`` — 1 unidade de valor no estado performing e 0 no
      default (perda cheia). Para ``LGD < 1`` passe ``lgd`` a
      :meth:`MigrationModel.simulate` (ou use ``values = [1, 1−LGD]``), o que
      reproduz a perda ``PD·LGD·EAD`` do modo Bernoulli/ASRF.

    Parameters
    ----------
    pd:
        Probabilidade de *default* em 1 ano, em ``[0, 1]``.
    """
    pd = float(pd)
    if not (0.0 <= pd <= 1.0):
        raise ValueError(f"pd deve estar em [0, 1]; recebido {pd!r}.")
    transition_matrix = np.array([[1.0 - pd, pd], [0.0, 1.0]], dtype=float)
    ratings = ["P", "D"]
    values = np.array([1.0, 0.0], dtype=float)
    return transition_matrix, ratings, values


__all__ = [
    "migration_thresholds",
    "MigrationModel",
    "simulate_creditmetrics",
    "two_state_matrix",
]
