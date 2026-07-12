"""
yggdrasil.credit_risk.tree
==========================
Árvore de segmentação **unificada** para risco de crédito: uma única classe
(:class:`TreeSegmenter`) que atende **classificação** e **regressão**
escolhendo o comportamento por ``task_type``. Substitui as antigas
classes separadas por tarefa.

Binning ótimo (OptBinning binário ou contínuo) ou manual, notas por folha, IV
(WoE binário ou contínuo), PSI por amostra (DES/OOT), IC bootstrap, faltantes em
bin própria, régua aplicável em pandas (``predict``) e Spark
(``to_pyspark``/``apply_spark``) e registro no MLflow (``log_to_mlflow``).

Uso típico::

    from yggdrasil.credit_risk.tree import TreeSegmenter

    # classificação
    seg = TreeSegmenter(df, target="target", task_type="classification",
                        sample_col="amostra", ref_sample="DES")
    seg.fit_auto(max_depth=3)
    seg.metrics()                 # KS, AUC, Gini, Acurácia, F1 por amostra

    # regressão — mesma API, só troca o task_type
    seg = TreeSegmenter(df, target="target", task_type="regression",
                        sample_col="amostra", ref_sample="DES")
    seg.fit_auto(max_depth=3)
    seg.metrics()                 # MAE, RMSE, R² por amostra

A interface interativa (ipywidgets, dentro do Jupyter/Databricks) é opcional e
carregada sob demanda — instale com ``pip install yggdrasil[ui]``::

    from yggdrasil.credit_risk.tree import TreeSegmenterUI
    ui = TreeSegmenterUI(df, target="target", task_type="classification",
                         sample_col="amostra", ref_sample="DES")
    ui
"""
from __future__ import annotations

from .segmenter import TreeSegmenter

__all__ = ["TreeSegmenter", "TreeSegmenterUI"]


def __getattr__(name):
    # Carrega a UI só quando pedida (depende de ipywidgets/IPython).
    if name == "TreeSegmenterUI":
        from .ui import TreeSegmenterUI

        return TreeSegmenterUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
