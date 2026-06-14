"""Motores de treino (opcionais). O núcleo da esteira é agnóstico ao treino."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from ..config import ColumnConfig


@runtime_checkable
class Trainer(Protocol):
    """Contrato mínimo de um treinador: recebe a amostra DES e devolve o modelo."""

    def train(self, df_dev: pd.DataFrame, cfg: ColumnConfig) -> Any:  # pragma: no cover
        ...


__all__ = ["Trainer"]
