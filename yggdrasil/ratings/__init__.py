"""Metodologias de grupos homogêneos (ratings).

Quatro estratégias disponíveis, registradas por nome curto:

* ``decis``   — :class:`DecileRating` (decis puros, obrigatório);
* ``quantil`` — :class:`QuantileMonotonicRating` (quantis finos + fusão);
* ``arvore``  — :class:`TreeRating` (árvore score->target + fusão);
* ``optbin``  — :class:`OptBinningRating` (binning ótimo monotônico).

Serialização: toda estratégia ajustada expõe ``to_dict()``;
:func:`rating_from_dict` reconstrói a instância (dispatch pelo nome da classe)
e :func:`apply_ratings` reaplica os grupos a novos scores **sem refit** — é o
mesmo conteúdo gravado como artefato ``ratings.json`` pelo logger da esteira.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Type, Union

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from .base import RatingStrategy
from .decile import DecileRating
from .manual import ManualScoreRating, PercentileRating
from .optbinning import OptBinningRating
from .quantile import QuantileMonotonicRating
from .tree import TreeRating

RATING_REGISTRY: Dict[str, Type[RatingStrategy]] = {
    "decis": DecileRating,
    "quantil": QuantileMonotonicRating,
    "arvore": TreeRating,
    "optbin": OptBinningRating,
    "manual_score": ManualScoreRating,
    "manual_percentil": PercentileRating,
}

DEFAULT_RATINGS: List[str] = ["decis", "quantil", "arvore", "optbin"]

#: versão do payload de :func:`ratings_to_dict` (artefato ``ratings.json``)
RATINGS_JSON_VERSION = 1


def build_ratings(names: Sequence[str] | None = None) -> List[RatingStrategy]:
    """Instancia estratégias de rating a partir de seus nomes curtos."""
    names = list(names) if names else list(DEFAULT_RATINGS)
    estrategias: List[RatingStrategy] = []
    for nome in names:
        if nome not in RATING_REGISTRY:
            raise ValueError(
                f"Rating desconhecido: {nome!r}. Opções: {sorted(RATING_REGISTRY)}"
            )
        estrategias.append(RATING_REGISTRY[nome]())
    return estrategias


# ---------------------------------------------------------------------------
# Serialização (ratings.json) e reaplicação sem refit
# ---------------------------------------------------------------------------
def ratings_to_dict(strategies: Sequence[RatingStrategy]) -> Dict:
    """Serializa estratégias **ajustadas** no payload do ``ratings.json``.

    O payload é JSON-estrito (bordas ``±inf`` viram strings) e reversível por
    :func:`rating_from_dict` / :func:`apply_ratings`.
    """
    return {
        "versao": RATINGS_JSON_VERSION,
        "estrategias": [s.to_dict() for s in strategies],
    }


def rating_from_dict(data: Mapping) -> RatingStrategy:
    """Factory: reconstrói uma estratégia ajustada a partir de ``to_dict()``.

    Resolve a classe pelo campo ``"classe"`` (nome da classe), com *fallback*
    no nome curto ``"name"`` do registro, e delega ao ``from_dict`` da classe.
    """
    por_classe = {c.__name__: c for c in RATING_REGISTRY.values()}
    cls = por_classe.get(str(data.get("classe")))
    if cls is None:
        cls = RATING_REGISTRY.get(str(data.get("name")))
    if cls is None:
        raise ValueError(
            f"Estratégia de rating desconhecida: classe={data.get('classe')!r}, "
            f"name={data.get('name')!r}. Opções: {sorted(por_classe)}"
        )
    return cls.from_dict(dict(data))


def _as_strategies(ratings_dict) -> List[RatingStrategy]:
    """Normaliza os formatos aceitos por :func:`apply_ratings` em instâncias."""
    if isinstance(ratings_dict, RatingStrategy):
        return [ratings_dict]
    if isinstance(ratings_dict, Mapping):
        if "estrategias" in ratings_dict:              # payload completo (ratings.json)
            itens = ratings_dict["estrategias"]
        elif "classe" in ratings_dict or "name" in ratings_dict:
            itens = [ratings_dict]                     # to_dict() de uma estratégia
        else:
            raise ValueError(
                "ratings_dict inválido: esperado to_dict() de uma estratégia, "
                "lista de to_dict() ou payload com chave 'estrategias'."
            )
    else:
        itens = list(ratings_dict)
    return [
        i if isinstance(i, RatingStrategy) else rating_from_dict(i) for i in itens
    ]


def apply_ratings(
    scores: Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[float]],
    ratings_dict,
    cfg: ColumnConfig | None = None,
) -> pd.DataFrame:
    """Reaplica grupos homogêneos serializados a novos scores, **sem refit**.

    Parameters
    ----------
    scores:
        Vetor de scores (array/Series) **ou** DataFrame que contenha a coluna
        de score (``cfg.score_col``).
    ratings_dict:
        Conteúdo do ``ratings.json`` (:func:`ratings_to_dict`), uma lista de
        ``to_dict()`` ou o ``to_dict()`` de uma única estratégia. Instâncias já
        ajustadas também são aceitas.
    cfg:
        Configuração de colunas; usa :class:`~yggdrasil.config.ColumnConfig`
        padrão quando omitida.

    Returns
    -------
    pandas.DataFrame
        Uma coluna por estratégia (``rating_<name>``), alinhada ao índice da
        entrada — pronta para ``df.join(...)``.
    """
    cfg = cfg or ColumnConfig()
    if isinstance(scores, pd.DataFrame):
        df = scores
    else:
        valores = np.asarray(scores, dtype=float)
        index = scores.index if isinstance(scores, pd.Series) else None
        df = pd.DataFrame({cfg.score_col: valores}, index=index)
    out = pd.DataFrame(index=df.index)
    for strat in _as_strategies(ratings_dict):
        out[strat.column] = strat.transform(df, cfg)
    return out


__all__ = [
    "RatingStrategy",
    "DecileRating",
    "QuantileMonotonicRating",
    "TreeRating",
    "OptBinningRating",
    "ManualScoreRating",
    "PercentileRating",
    "RATING_REGISTRY",
    "DEFAULT_RATINGS",
    "RATINGS_JSON_VERSION",
    "build_ratings",
    "ratings_to_dict",
    "rating_from_dict",
    "apply_ratings",
]
