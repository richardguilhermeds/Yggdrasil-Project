"""
yggdrasil.credit_risk.pd
========================
Construção interativa e auditável de segmentações de **PD** (Probability of
Default) — modelos de **classificação** — para risco de crédito, sob
CMN 4.966/2021 e IFRS 9.

O núcleo é o :class:`SequentialPDSegmenter`: binning ótimo (OptBinning, alvo
binário) ou manual, notas por folha, IV (WoE binário), PSI por amostra
(DES/OOT), IC bootstrap da taxa de default, faltantes em bin própria, métricas
de discriminação (**KS, ROC/AUC, Gini, Acurácia, F1**), régua aplicável em
pandas (``predict``) e em Spark (``to_pyspark``/``apply_spark``) e registro no
MLflow / Unity Catalog (``log_to_mlflow``).

Uso típico::

    from yggdrasil.credit_risk.pd import SequentialPDSegmenter

    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES")
    seg.fit_auto(max_depth=3)
    seg.leaves()
    seg.metrics()          # KS, AUC, Gini, Acurácia, F1 por amostra
    regua = seg.predict(df_novos)

A interface interativa (ipywidgets, dentro do Jupyter/Databricks) é opcional e
carregada sob demanda — instale com ``pip install yggdrasil[ui]``::

    from yggdrasil.credit_risk.pd import PDSegmenterUI
    ui = PDSegmenterUI(df, target="target", sample_col="amostra", ref_sample="DES")
    ui
"""
from __future__ import annotations

from .segmenter import SequentialPDSegmenter

__all__ = ["SequentialPDSegmenter", "PDSegmenterUI"]


def __getattr__(name):
    # Carrega a UI só quando pedida (depende de ipywidgets/IPython).
    if name == "PDSegmenterUI":
        from .ui import PDSegmenterUI

        return PDSegmenterUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
