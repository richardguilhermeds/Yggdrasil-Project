"""Utilidades transversais do pacote ``yggdrasil``."""

from __future__ import annotations

import contextlib
import logging
import warnings
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


_AVISO_SOLVER_CVXPY = "Encountered unexpected exception importing solver"


class _FiltroSolverCvxpy(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith(_AVISO_SOLVER_CVXPY)


@contextlib.contextmanager
def sem_aviso_solver_cvxpy():
    """Cala, só durante o bloco, o aviso que o cvxpy emite ao ser importado
    quando um solver opcional está instalado mas quebrado (o caso típico é o
    ``highspy`` compilado contra outro numpy no cluster)::

        (CVXPY) Encountered unexpected exception importing solver HIGHS: ...

    O cvxpy chega pelo ``optbinning`` (optbinning → ropwr → cvxpy) e sonda TODOS
    os solvers no import. A yggdrasil não usa o cvxpy (o optbinning ajusta com o
    CP-SAT do OR-Tools), então o aviso é ruído. Envolve só o import do
    optbinning: um filtro no logger ``__cvxpy__`` descarta só essa mensagem (o
    nível não serve — o cvxpy faz ``setLevel(INFO)`` no próprio import) e o mesmo
    texto é ignorado em ``warnings`` (versões antigas). Ao sair, o filtro sai e
    o logger fica como o cvxpy o deixou. Quem importar o cvxpy por conta própria
    antes continua vendo o aviso."""
    lg = logging.getLogger("__cvxpy__")
    filtro = _FiltroSolverCvxpy()
    lg.addFilter(filtro)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_AVISO_SOLVER_CVXPY)
            yield
    finally:
        lg.removeFilter(filtro)

__all__ = ["idx_para_letra", "get_logger", "sem_aviso_solver_cvxpy",
           "ClusterKeepAlive", "keep_cluster_alive", "stop_keep_alive"]
