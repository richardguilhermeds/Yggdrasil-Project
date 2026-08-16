"""
Motor de migração: cadeia de Markov sobre a matriz de transição
===============================================================
A terceira família de PD *lifetime* (ao lado de safra e sobrevivência) e a mais
usada quando a carteira já tem uma **régua de rating** funcionando: estima-se a
matriz de transição de um período e projeta-se o horizonte por
**Chapman-Kolmogorov**, ``P(t) = M^t``. Com o *default* como estado absorvente
(a última coluna), a PD acumulada em ``t`` períodos de quem está hoje no rating
``i`` é simplesmente ``(M^t)[i, default]`` — e daí sai a curva inteira.

A vantagem sobre a curva de safra é que a estrutura a termo passa a depender do
**rating de hoje**, não só da idade: um contrato bom e um ruim com a mesma idade
recebem curvas diferentes, com a convergência natural das cadeias (no longo
prazo as curvas se aproximam, porque o bom também migra). A limitação é a
hipótese de Markov — o próximo estado depende só do estado atual, não da
trajetória — e a homogeneidade no tempo, que este módulo permite relaxar de duas
formas: matriz por período ou condicionamento ao ciclo.

**Reuso, não reimplementação.** A estimação e o condicionamento já existem no
motor de capital econômico e são chamados daqui:

* :func:`~yggdrasil.credit_risk.capital.migration.estimate_transition_matrix` —
  estimadores de **coorte** (frequência relativa dos pares consecutivos) e de
  **duração** (gerador em tempo contínuo, ``expm(Λ)``), com suavização aditiva e
  tratamento do *default* absorvente;
* :func:`~yggdrasil.credit_risk.capital.migration.zshift_transition_matrix` —
  o deslocamento dos limiares pela carga sistêmica ``√ρ·z``, que transforma a
  matriz TTC numa matriz *point-in-time* condicionada ao ciclo. A convenção de
  sinal é a do repositório: ``z > 0`` = ciclo **benigno**.

Isso mantém uma única fórmula de migração no pacote — a mesma matriz que
alimenta a simulação de capital alimenta a curva de PD do ECL.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..capital.migration import estimate_transition_matrix, zshift_transition_matrix
from .curves import PDCurve
from .panel import ContractPanel


def pd_curve_from_matrix(
    transition_matrix,
    ratings: Sequence,
    horizon: int,
    from_rating=None,
    freq: str = "M",
) -> Union[PDCurve, Dict[object, PDCurve]]:
    """Curva de PD implícita numa matriz de transição, por Chapman-Kolmogorov.

    ``F_i(t) = (M^t)[i, default]`` — a probabilidade acumulada de estar no estado
    de *default* após ``t`` períodos, partindo do rating ``i``. O *default* é a
    **última** posição da régua (convenção do módulo de capital: estados do
    melhor ao pior).

    Parameters
    ----------
    transition_matrix:
        Matriz ``(n, n)`` de um período, linhas somando 1.
    ratings:
        Rótulos dos estados, do melhor ao pior (o último é o *default*).
    horizon:
        Nº de períodos da curva.
    from_rating:
        Rating de partida. ``None`` devolve ``{rating: PDCurve}`` para todos os
        estados **não absorventes**.
    freq:
        Frequência dos períodos da matriz.

    Returns
    -------
    PDCurve | dict[object, PDCurve]
    """
    tm = np.asarray(transition_matrix, dtype=float)
    if tm.ndim != 2 or tm.shape[0] != tm.shape[1]:
        raise ValueError(f"transition_matrix deve ser quadrada; recebida {tm.shape}.")
    ratings = list(ratings)
    if len(ratings) != tm.shape[0]:
        raise ValueError(
            f"ratings tem {len(ratings)} rótulos e a matriz é {tm.shape[0]}×{tm.shape[0]}."
        )
    h = int(horizon)
    if h < 1:
        raise ValueError(f"horizon deve ser >= 1; recebido {horizon!r}.")
    if np.any(tm < -1e-9) or not np.allclose(tm.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("as linhas da matriz de transição devem ser não-negativas e somar 1.")

    idx_default = len(ratings) - 1
    # Acumuladas por potência sucessiva (evita n matrizes independentes).
    acumuladas = np.empty((tm.shape[0], h), dtype=float)
    passo = np.eye(tm.shape[0])
    for t in range(h):
        passo = passo @ tm
        acumuladas[:, t] = passo[:, idx_default]
    acumuladas = np.clip(acumuladas, 0.0, 1.0)
    # Higiene: a acumulada de um estado absorvente pode oscilar no último dígito.
    acumuladas = np.maximum.accumulate(acumuladas, axis=1)

    def _curva(i: int) -> PDCurve:
        return PDCurve.from_cumulative(
            acumuladas[i], label=str(ratings[i]), freq=freq,
            metodo="markov", rating=str(ratings[i]),
        )

    if from_rating is not None:
        if from_rating not in ratings:
            raise ValueError(f"rating {from_rating!r} fora da régua {ratings}.")
        return _curva(ratings.index(from_rating))
    # Todos, menos os estados absorventes (a curva de quem já quebrou é trivial).
    absorvente = np.isclose(np.diag(tm), 1.0)
    return {ratings[i]: _curva(i) for i in range(len(ratings)) if not absorvente[i]}


class MarkovPD:
    """PD *lifetime* por cadeia de Markov sobre a matriz de transição de rating.

    Parameters
    ----------
    method:
        Estimador da matriz: ``'cohort'`` (padrão) ou ``'duration'`` — repassado
        a :func:`~yggdrasil.credit_risk.capital.migration.estimate_transition_matrix`.
    smoothing:
        Suavização aditiva das contagens (preenche células vazias). ``None``
        desliga.
    ratings:
        Ordem dos estados, do **melhor ao pior**. ``None`` usa a ordem crescente
        dos valores observados, com ``default_state`` empurrado para o fim.
    default_state:
        Rótulo do estado de *default*. Ao ajustar sobre um
        :class:`~yggdrasil.credit_risk.ecl.panel.ContractPanel`, as linhas com a
        flag de *default* recebem este rótulo — é assim que o evento de quebra
        entra na régua de rating.

    Examples
    --------
    >>> mk = MarkovPD().fit(painel, rating_col="rating")
    >>> mk.curve("A", horizon=60).pd_lifetime()
    >>> mk.condition(z=-1.5, rho=0.08).curve("A", horizon=60).pd_12m()   # ciclo adverso
    """

    def __init__(self, method: str = "cohort", smoothing: Optional[float] = None,
                 ratings: Optional[Sequence] = None, default_state: object = "D") -> None:
        self.method = method
        self.smoothing = smoothing
        self.ratings_init = list(ratings) if ratings is not None else None
        self.default_state = default_state

        self.matrix_: Optional[np.ndarray] = None
        self.ratings_: List = []
        self.freq: str = "M"
        self.meta: dict = {}

    # -- construção -------------------------------------------------------
    @classmethod
    def from_matrix(cls, transition_matrix, ratings: Sequence, freq: str = "M") -> "MarkovPD":
        """Constrói a partir de uma matriz **já estimada** (ex.: matriz de agência)."""
        obj = cls(ratings=list(ratings), default_state=list(ratings)[-1])
        tm = np.asarray(transition_matrix, dtype=float)
        if tm.ndim != 2 or tm.shape[0] != tm.shape[1] or tm.shape[0] != len(ratings):
            raise ValueError("matriz e régua de ratings incompatíveis.")
        obj.matrix_ = tm
        obj.ratings_ = list(ratings)
        obj.freq = freq
        obj.meta = {"origem": "matriz_informada"}
        return obj

    def fit(self, source: Union[ContractPanel, pd.DataFrame], rating_col: Optional[str] = None,
            id_col: Optional[str] = None, period_col: Optional[str] = None) -> "MarkovPD":
        """Estima a matriz de transição de um período.

        ``source`` pode ser um :class:`ContractPanel` — e aí ``id_col``/
        ``period_col`` vêm do próprio painel, ``rating_col`` cai em
        ``segment_col`` se omitido, e as linhas com *default* recebem
        ``default_state`` — ou um DataFrame longo com as três colunas
        explícitas."""
        if isinstance(source, ContractPanel):
            painel = source
            rating_col = rating_col or painel.segment_col
            if rating_col is None:
                raise ValueError(
                    "informe rating_col (ou defina segment_col no painel) — a cadeia de "
                    "Markov precisa de uma régua de estados."
                )
            if rating_col not in painel.df.columns:
                raise ValueError(f"Coluna {rating_col!r} não encontrada no painel.")
            d = painel.df[[painel.id_col, painel.date_col, rating_col, painel.default_col]].copy()
            # A quebra vira o estado de default (é o evento que a cadeia absorve).
            d[rating_col] = d[rating_col].astype(object).where(
                d[painel.default_col] == 0, self.default_state
            )
            id_col, period_col = painel.id_col, painel.date_col
            self.freq = painel.freq
        else:
            d = pd.DataFrame(source).copy()
            faltando = [c for c in (id_col, rating_col, period_col) if c is None]
            if faltando:
                raise ValueError(
                    "com DataFrame é preciso informar id_col, rating_col e period_col."
                )

        estados = self.ratings_init
        if estados is None:
            observados = [v for v in pd.unique(d[rating_col].dropna())]
            demais = sorted((v for v in observados if v != self.default_state), key=str)
            estados = demais + ([self.default_state] if self.default_state in observados else [])
        if len(estados) < 2:
            raise ValueError(
                "são necessários ao menos 2 estados (um vivo e o default) na régua de rating."
            )
        if estados[-1] != self.default_state:
            raise ValueError(
                f"o estado de default ({self.default_state!r}) deve ser o ÚLTIMO da régua "
                f"(pior estado); régua recebida: {estados}."
            )

        tm, ratings = estimate_transition_matrix(
            d, id_col=id_col, rating_col=rating_col, period_col=period_col,
            method=self.method, smoothing=self.smoothing, ratings=estados,
        )
        self.matrix_, self.ratings_ = tm, list(ratings)
        self.meta = {"metodo": self.method, "smoothing": self.smoothing,
                     "n_estados": len(ratings), "origem": "estimada"}
        return self

    def _check_fit(self) -> None:
        if self.matrix_ is None:
            raise RuntimeError("a matriz ainda não foi estimada — chame .fit(...) antes.")

    # -- curvas -----------------------------------------------------------
    def curve(self, rating, horizon: int = 60) -> PDCurve:
        """Curva de PD de quem está hoje no rating informado."""
        self._check_fit()
        return pd_curve_from_matrix(self.matrix_, self.ratings_, horizon,
                                    from_rating=rating, freq=self.freq)

    def curves(self, horizon: int = 60) -> Dict[object, PDCurve]:
        """``{rating: PDCurve}`` para todos os estados não absorventes."""
        self._check_fit()
        return pd_curve_from_matrix(self.matrix_, self.ratings_, horizon, freq=self.freq)

    # -- condicionamento ao ciclo ------------------------------------------
    def condition(self, z: float, rho: float) -> "MarkovPD":
        """Matriz *point-in-time*: desloca os limiares pela carga sistêmica ``√ρ·z``.

        ``z > 0`` é ciclo **benigno** (menos migração para pior) e ``z < 0`` é
        adverso — a mesma convenção de
        :func:`~yggdrasil.credit_risk.econometric.transforms.vasicek_z`. Devolve
        um novo :class:`MarkovPD`; o original não muda."""
        self._check_fit()
        novo = MarkovPD.from_matrix(
            zshift_transition_matrix(self.matrix_, z=z, rho=rho), self.ratings_, freq=self.freq
        )
        novo.default_state = self.default_state
        novo.meta = {**self.meta, "condicionada": {"z": float(z), "rho": float(rho)}}
        return novo

    # -- inspeção -----------------------------------------------------------
    def matrix_frame(self, steps: int = 1) -> pd.DataFrame:
        """A matriz de ``steps`` períodos (``M^steps``) rotulada pelos ratings."""
        self._check_fit()
        if int(steps) < 1:
            raise ValueError(f"steps deve ser >= 1; recebido {steps!r}.")
        m = np.linalg.matrix_power(self.matrix_, int(steps))
        return pd.DataFrame(m, index=pd.Index(self.ratings_, name="de"),
                            columns=pd.Index(self.ratings_, name="para"))

    def pd_by_rating(self, horizons: Sequence[int] = (12, 24, 36, 60)) -> pd.DataFrame:
        """PD acumulada por rating em vários horizontes — a tabela de leitura rápida."""
        self._check_fit()
        maior = int(max(horizons))
        curvas = self.curves(horizon=maior)
        linhas = []
        for rating, curva in curvas.items():
            linha = {"rating": rating}
            for h in horizons:
                linha[f"pd_{int(h)}p"] = curva.pd_lifetime(int(h))
            linhas.append(linha)
        return pd.DataFrame(linhas)

    def __repr__(self) -> str:
        if self.matrix_ is None:
            return f"MarkovPD(method={self.method!r}, não ajustado)"
        return (f"MarkovPD(method={self.method!r}, estados={len(self.ratings_)}, "
                f"freq={self.freq!r})")


__all__ = ["MarkovPD", "pd_curve_from_matrix"]
