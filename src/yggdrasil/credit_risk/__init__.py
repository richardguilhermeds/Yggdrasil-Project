"""
yggdrasil.credit_risk
=====================
Raiz de domínio para **risco de crédito**: parâmetros regulatórios (PD, LGD, EAD),
segmentações e réguas sob CMN 4.966/2021 e IFRS 9.

Abriga :mod:`yggdrasil.credit_risk.tree` — a **árvore de segmentação unificada**
(:class:`TreeSegmenter`), que atende classificação (PD) e regressão (LGD) via
``task_type`` (substitui as antigas classes separadas de PD e LGD) — e
:mod:`yggdrasil.credit_risk.model` (segmentação **orientada a modelo**, que também
unifica classificação e regressão via ``task_type``).
"""
from __future__ import annotations

from .model import ModelSegmenter
from .tree import TreeSegmenter

__all__ = ["TreeSegmenter", "ModelSegmenter", "tree", "model"]


def __getattr__(name):
    # UIs interativas carregadas sob demanda (dependem de ipywidgets/IPython).
    if name == "TreeSegmenterUI":
        from .tree import TreeSegmenterUI

        return TreeSegmenterUI
    if name == "ModelSegmenterUI":
        from .model import ModelSegmenterUI

        return ModelSegmenterUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__ + ["TreeSegmenterUI", "ModelSegmenterUI"])
