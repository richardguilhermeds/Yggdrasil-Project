"""Detecção do backend de execução da seleção de features.

O módulo aceita **pandas** (tudo no driver, sem cluster) ou **Spark** (distribuído).
A escolha é feita pelo tipo do DataFrame recebido, não por configuração: o mesmo
``run_feature_selection`` atende os dois e produz o mesmo relatório.

O teste é deliberadamente *positivo para pandas* e não para Spark. Um DataFrame do
``pyspark.sql`` clássico expõe ``_jdf``, mas o do Spark Connect (Databricks serverless,
DBR 14+) não — qualquer heurística baseada em atributos ou no nome do módulo erraria
em algum desses ambientes. Tratando "não-pandas" como Spark, o comportamento de hoje
fica preservado byte a byte para toda entrada que já funcionava, incluindo as
mensagens de erro do import *gated* do pyspark.
"""

from __future__ import annotations

import pandas as pd


def is_pandas(df) -> bool:
    """``True`` se ``df`` for um ``pandas.DataFrame`` (backend local)."""
    return isinstance(df, pd.DataFrame)


def backend_name(df) -> str:
    """Nome do backend que será usado para ``df`` — ``"pandas"`` ou ``"spark"``."""
    return "pandas" if is_pandas(df) else "spark"


__all__ = ["is_pandas", "backend_name"]
