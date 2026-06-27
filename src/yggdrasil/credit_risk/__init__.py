"""
yggdrasil.credit_risk
=====================
Raiz de domínio para **risco de crédito**: parâmetros regulatórios (PD, LGD, EAD),
segmentações e réguas sob CMN 4.966/2021 e IFRS 9.

Abriga os subpacotes :mod:`yggdrasil.credit_risk.lgd` (segmentação de LGD,
modelo de regressão) e :mod:`yggdrasil.credit_risk.pd` (segmentação de PD,
modelo de classificação), ambos por árvore de bins, e
:mod:`yggdrasil.credit_risk.model` (segmentação **orientada a modelo**, que
unifica classificação e regressão via ``task_type``). Reservado para crescer com EAD.
"""
from __future__ import annotations

from .lgd import SequentialLGDSegmenter
from .model import ModelSegmenter
from .pd import SequentialPDSegmenter

__all__ = ["SequentialLGDSegmenter", "SequentialPDSegmenter", "ModelSegmenter",
           "lgd", "pd", "model"]
