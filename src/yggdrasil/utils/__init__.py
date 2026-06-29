"""Utilidades transversais do pacote ``yggdrasil``."""

from __future__ import annotations

import logging
from typing import Optional  # noqa: F401  (mantido p/ compat de imports externos)

from .keepalive import ClusterKeepAlive, keep_cluster_alive, stop_keep_alive


def idx_para_letra(idx: int) -> str:
    """Converte um índice 0-based em rótulo alfabético: 0->A, 1->B, ..., 26->AA.

    Usado para nomear grupos homogêneos (ratings) de forma legível.
    """
    if idx < 0:
        raise ValueError("idx deve ser >= 0")
    letras = ""
    idx += 1
    while idx > 0:
        idx, resto = divmod(idx - 1, 26)
        letras = chr(65 + resto) + letras
    return letras


def get_logger(name: str = "yggdrasil", level: int = logging.INFO) -> logging.Logger:
    """Logger configurado com um handler de console (idempotente)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


__all__ = ["idx_para_letra", "get_logger",
           "ClusterKeepAlive", "keep_cluster_alive", "stop_keep_alive"]
