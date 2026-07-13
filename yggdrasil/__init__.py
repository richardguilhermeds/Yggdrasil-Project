"""Yggdrasil — ferramentas e esteiras de Machine Learning para projetos em dados.

Esteira governada de ML (estilo risco de crédito) com MLflow: métricas por
amostra, grupos homogêneos (ratings), PSI ao longo do tempo, shifts DES/OOT,
SHAP e relatórios por grupo.

Uso rápido
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

__version__ = "0.0.6"

__all__ = [
    "MLPipeline",
    "PipelineResult",
    "ColumnConfig",
    "feature_columns",
    "build_ratings",
    "__version__",
]
