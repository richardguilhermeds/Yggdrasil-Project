"""Yggdrasil â€” ferramentas e esteiras de Machine Learning para projetos em dados.

Esteira governada de ML (estilo risco de crÃ©dito) com MLflow: mÃ©tricas por
amostra, grupos homogÃªneos (ratings), PSI ao longo do tempo, shifts DES/OOT,
SHAP e relatÃ³rios por grupo.

Uso rÃ¡pido
----------
>>> from yggdrasil import MLPipeline, ColumnConfig
>>> cfg = ColumnConfig()                      # feat_, dt_ref, amostra, target
>>> pipe = MLPipeline(cfg, problem_type="classification",
...                   ratings=["decis", "quantil", "arvore", "optbin"])
>>> resultado = pipe.run(df, model=modelo_treinado, experiment="/Shared/Yggdrasil/pd_pf")
"""

from __future__ import annotations

from .config import ColumnConfig, feature_columns
from .pipeline import MLPipeline, PipelineResult
from .ratings import build_ratings

__version__ = "0.0.20"

__all__ = [
    "MLPipeline",
    "PipelineResult",
    "ColumnConfig",
    "feature_columns",
    "build_ratings",
    "__version__",
]
