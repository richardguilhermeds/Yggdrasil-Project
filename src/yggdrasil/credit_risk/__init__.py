"""
yggdrasil.credit_risk
=====================
Raiz de domínio para **risco de crédito**: parâmetros regulatórios (PD, LGD, EAD),
segmentações, réguas e **capital** sob CMN 4.966/2021, 4.557/2017 e IFRS 9.

Abriga:

* :mod:`yggdrasil.credit_risk.tree` — a **árvore de segmentação unificada**
  (:class:`TreeSegmenter`), que atende classificação (PD) e regressão (LGD) via
  ``task_type`` (substitui as antigas classes separadas de PD e LGD).
* :mod:`yggdrasil.credit_risk.model` — segmentação **orientada a modelo**, que
  também unifica classificação e regressão via ``task_type``.
* :mod:`yggdrasil.credit_risk.capital` — **capital econômico** de carteira:
  distribuição de perdas, ASRF/Vasicek analítico, simulação de Monte Carlo
  multifatorial, CreditRisk+, correlações, alocação de Euler/RAROC e validação
  (:class:`~yggdrasil.credit_risk.capital.Portfolio`,
  :class:`~yggdrasil.credit_risk.capital.Segment`).
* :mod:`yggdrasil.credit_risk.econometric` — **modelos econométricos (satélite)**
  de PD, LGD e CCF: ligam as séries agregadas dos parâmetros de risco às variáveis
  macro (ARDL, ARIMA/ARIMAX, fator ``Z`` de Vasicek, beta/fractional logit,
  VAR/VECM, painel), com seleção champion-challenger, projeção por cenários e
  integração com ECL/estresse/capital. Carregado **sob demanda** — requer o extra
  ``econometric`` (``statsmodels``, ``arch``), que o restante do pacote não exige.
"""
from __future__ import annotations

from . import capital
from .model import ModelSegmenter
from .tree import TreeSegmenter

__all__ = ["TreeSegmenter", "ModelSegmenter", "capital", "tree", "model", "econometric"]


def __getattr__(name):
    # UIs interativas carregadas sob demanda (dependem de ipywidgets/IPython).
    if name == "TreeSegmenterUI":
        from .tree import TreeSegmenterUI

        return TreeSegmenterUI
    if name == "ModelSegmenterUI":
        from .model import ModelSegmenterUI

        return ModelSegmenterUI
    # Modelos econométricos carregados sob demanda: só quem os usa precisa de
    # statsmodels/arch (extra 'econometric'); capital/segmentadores não os exigem.
    if name == "econometric":
        import importlib

        return importlib.import_module(f"{__name__}.econometric")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__ + ["TreeSegmenterUI", "ModelSegmenterUI"])
