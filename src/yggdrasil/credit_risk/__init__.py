"""
yggdrasil.credit_risk
=====================
Raiz de domínio para **risco de crédito**: parâmetros regulatórios (PD, LGD, EAD),
segmentações e réguas sob CMN 4.966/2021 e IFRS 9.

Hoje abriga o subpacote :mod:`yggdrasil.credit_risk.lgd` (segmentação de LGD).
Reservado para crescer com PD e EAD.
"""
from __future__ import annotations

from .lgd import SequentialLGDSegmenter

__all__ = ["SequentialLGDSegmenter", "lgd"]
