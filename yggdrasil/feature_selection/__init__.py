"""Seleção de features para modelos de classificação/regressão via PySpark.

Esteira **independente** (não entra no pipeline de modelo) que seleciona features
organizadas por *book* (origem de dados — ex.: serasa, bvs). Cada book passa por
filtros duros (missing, variância, redundância) e por uma avaliação de importância
(RandomForest + univariadas) e **Boruta**, consolidadas num consenso. No fim,
produz uma tabela e painéis por book, além de um ranking global das variáveis mais
fortes para entrar nos modelos.

Uso típico (Spark)::

    from yggdrasil import ColumnConfig
    from yggdrasil.feature_selection import run_feature_selection

    report = run_feature_selection(sdf, ColumnConfig(), books=["serasa", "bvs"])
    report.selected_features      # {"serasa": [...], "bvs": [...]}
    report.overall_importance     # ranking global das selecionadas
    report.panels["overall_importance"]   # figura

O ``pyspark`` é um extra opcional (``pip install 'yggdrasil[spark]'``); o import
deste pacote funciona sem ele e só falha — com mensagem clara — ao executar.
"""

from __future__ import annotations

from .books import Book, resolve_books
from .config import FeatureSelectionConfig
from .selector import FeatureSelectionReport, run_feature_selection

__all__ = [
    "run_feature_selection",
    "FeatureSelectionReport",
    "FeatureSelectionConfig",
    "resolve_books",
    "Book",
]
