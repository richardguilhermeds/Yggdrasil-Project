"""Seleção de features para modelos de classificação/regressão (pandas ou PySpark).

Esteira **independente** (não entra no pipeline de modelo) que seleciona features
organizadas por *book* (origem de dados — ex.: externo, mercado). Cada book passa por
filtros duros (missing, variância, redundância) e por uma avaliação de importância
(RandomForest + univariadas) e **Boruta**, consolidadas num consenso. No fim,
produz uma tabela e painéis por book, além de um ranking global das variáveis mais
fortes para entrar nos modelos.

``run_feature_selection`` aceita **pandas ou Spark** e escolhe o backend pelo tipo do
DataFrame — a chamada e o relatório são idênticos nos dois casos::

    from yggdrasil import ColumnConfig
    from yggdrasil.feature_selection import run_feature_selection

    report = run_feature_selection(df, ColumnConfig(), books=["externo", "mercado"])
    report.selected_features      # {"externo": [...], "mercado": [...]}
    report.overall_importance     # ranking global das selecionadas
    report.panels["overall_importance"]   # figura

Com pandas tudo roda no driver (sklearn, percentis exatos) e o ``pyspark`` nem precisa
estar instalado — bom para dataset que cabe em memória, notebook local e teste. Com
Spark tudo é distribuído (``pyspark.ml``, ``approxQuantile``). O ``pyspark`` é um extra
opcional (``pip install 'yggdrasil[spark]'``); o import deste pacote funciona sem ele e
só falha — com mensagem clara — ao executar algo distribuído.
"""

from __future__ import annotations

from .backend import backend_name, is_pandas
from .books import Book, resolve_books
from .config import FeatureSelectionConfig
from .selector import FeatureSelectionReport, run_feature_selection

__all__ = [
    "run_feature_selection",
    "FeatureSelectionReport",
    "FeatureSelectionConfig",
    "resolve_books",
    "Book",
    "backend_name",
    "is_pandas",
]
