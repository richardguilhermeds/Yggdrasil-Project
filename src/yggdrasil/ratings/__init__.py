"""Metodologias de grupos homogêneos (ratings).

Quatro estratégias disponíveis, registradas por nome curto:

* ``decis``   — :class:`DecileRating` (decis puros, obrigatório);
* ``quantil`` — :class:`QuantileMonotonicRating` (quantis finos + fusão);
* ``arvore``  — :class:`TreeRating` (árvore score->target + fusão);
* ``optbin``  — :class:`OptBinningRating` (binning ótimo monotônico).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Type

from .base import RatingStrategy
from .decile import DecileRating
from .optbinning import OptBinningRating
from .quantile import QuantileMonotonicRating
from .tree import TreeRating

RATING_REGISTRY: Dict[str, Type[RatingStrategy]] = {
    "decis": DecileRating,
    "quantil": QuantileMonotonicRating,
    "arvore": TreeRating,
    "optbin": OptBinningRating,
}

DEFAULT_RATINGS: List[str] = ["decis", "quantil", "arvore", "optbin"]


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


__all__ = [
    "RatingStrategy",
    "DecileRating",
    "QuantileMonotonicRating",
    "TreeRating",
    "OptBinningRating",
    "RATING_REGISTRY",
    "DEFAULT_RATINGS",
    "build_ratings",
]
