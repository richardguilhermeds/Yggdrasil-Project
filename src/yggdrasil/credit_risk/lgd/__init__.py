"""
yggdrasil.credit_risk.lgd
=========================
Construção interativa e auditável de segmentações de **LGD** (Loss Given Default)
para risco de crédito, sob CMN 4.966/2021 e IFRS 9.

O núcleo é o :class:`SequentialLGDSegmenter`: binning ótimo (OptBinning) ou manual,
notas por folha, IV, PSI por amostra (DES/OOT), IC bootstrap, faltantes em bin
própria, régua aplicável em pandas (``predict``) e em Spark (``to_pyspark``), e
registro no MLflow / Unity Catalog (``log_to_mlflow``).

Uso típico::

    from yggdrasil.credit_risk.lgd import SequentialLGDSegmenter

    seg = SequentialLGDSegmenter(df, target="lgd", sample_col="amostra",
                                 ref_sample="DES")
    seg.fit_auto(max_depth=3)
    seg.leaves()
    regua = seg.predict(df_novos)

A interface interativa (ipywidgets, dentro do Jupyter/Databricks) é opcional e
carregada sob demanda — instale com ``pip install yggdrasil[ui]``::

    from yggdrasil.credit_risk.lgd import LGDSegmenterUI
    ui = LGDSegmenterUI(df, target="lgd", sample_col="amostra", ref_sample="DES")
    ui
"""
from __future__ import annotations

from .segmenter import SequentialLGDSegmenter

__all__ = ["SequentialLGDSegmenter", "LGDSegmenterUI"]


def __getattr__(name):
    # Carrega a UI só quando pedida (depende de ipywidgets/IPython).
    if name == "LGDSegmenterUI":
        from .ui import LGDSegmenterUI

        return LGDSegmenterUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
