"""Resolução de *books* — grupos de features analisados em conjunto.

Um **book** representa uma origem/bloco de variáveis (ex.: serasa, bvs). A seleção
de features roda book a book e depois consolida uma visão global. Os books podem
ser definidos de três formas (ver :func:`resolve_books`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

from ..config import ColumnConfig
from ..utils import get_logger

_logger = get_logger("yggdrasil.feature_selection")

# Aceita: lista de palavras-chave, dict {book: [colunas]}, ou None (auto).
BooksSpec = Union[Sequence[str], Dict[str, Sequence[str]], None]


@dataclass
class Book:
    """Um book: um nome e a lista de features que pertencem a ele."""

    name: str
    features: List[str]

    def __len__(self) -> int:
        return len(self.features)


def feature_columns(sdf, cfg: ColumnConfig) -> List[str]:
    """Lista as colunas de features (prefixo ``feature_prefix``) de um DF (pandas/Spark).

    Funciona tanto com pandas quanto com Spark, pois ambos expõem ``.columns``.
    """
    cols = [c for c in sdf.columns if c.startswith(cfg.feature_prefix)]
    if not cols:
        raise ValueError(
            f"Nenhuma coluna de feature encontrada com o prefixo '{cfg.feature_prefix}'."
        )
    return cols


def _auto_token(col: str, prefix: str) -> str:
    """Deriva o token do book a partir do nome da coluna (1º segmento após o prefixo)."""
    resto = col[len(prefix):] if col.startswith(prefix) else col
    partes = [p for p in resto.split("_") if p]
    return partes[0] if partes else resto


def resolve_books(sdf, cfg: ColumnConfig, books: BooksSpec = None) -> List[Book]:
    """Resolve a especificação de books para uma lista de :class:`Book`.

    Parameters
    ----------
    sdf:
        DataFrame (Spark ou pandas) — usado apenas para listar as colunas.
    cfg:
        Contrato de colunas (define o prefixo das features).
    books:
        * ``Sequence[str]`` — lista de palavras-chave. Cada palavra vira um book com
          as features cujo nome **contém** o token (case-insensitive).
          Ex.: ``["serasa", "bvs"]``.
        * ``Dict[str, Sequence[str]]`` — colunas explícitas por book.
        * ``None`` — auto-deriva o book do 1º segmento após o prefixo
          (``feat_serasa_score`` → book ``"serasa"``).
    """
    cols = feature_columns(sdf, cfg)
    cols_set = set(sdf.columns)

    if books is None:
        grupos: Dict[str, List[str]] = {}
        for c in cols:
            grupos.setdefault(_auto_token(c, cfg.feature_prefix), []).append(c)
        resolved = [Book(name, feats) for name, feats in grupos.items()]

    elif isinstance(books, dict):
        resolved = []
        for name, feats in books.items():
            feats = list(dict.fromkeys(feats))  # dedup preservando ordem
            faltando = [f for f in feats if f not in cols_set]
            if faltando:
                raise ValueError(
                    f"Book '{name}': colunas inexistentes no DataFrame: {faltando}"
                )
            nao_feat = [f for f in feats if not f.startswith(cfg.feature_prefix)]
            if nao_feat:
                _logger.warning(
                    "Book '%s': colunas sem o prefixo '%s' incluídas: %s",
                    name, cfg.feature_prefix, nao_feat,
                )
            resolved.append(Book(name, feats))

    else:  # Sequence[str] de palavras-chave
        resolved = []
        for kw in books:
            token = str(kw).lower()
            feats = [c for c in cols if token in c.lower()]
            if not feats:
                _logger.warning(
                    "Book '%s': nenhuma feature contém o token '%s' — book vazio ignorado.",
                    kw, kw,
                )
                continue
            resolved.append(Book(str(kw), feats))

    resolved = [b for b in resolved if b.features]
    if not resolved:
        raise ValueError("Nenhum book com features foi resolvido. Verifique 'books'.")
    return resolved


__all__ = ["Book", "BooksSpec", "resolve_books", "feature_columns"]
