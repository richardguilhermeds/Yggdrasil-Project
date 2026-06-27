"""
yggdrasil.credit_risk.model
===========================
Segmentação **orientada a modelo** para risco de crédito — unifica
**classificação** (ex.: PD) e **regressão** (ex.: LGD) num único objeto via o
parâmetro ``task_type``.

O núcleo é o :class:`ModelSegmenter`: análise univariada das variáveis (logodds/
WoE, IV, distribuição e inversão entre amostras/safras) para decidir o que entra
no modelo; ajuste de Regressão Logística/Linear ou ML (RandomForest,
GradientBoosting) — treinado aqui ou recebido pronto; métricas do modelo e
gráficos **SHAP**; e a segmentação do **score** em **ratings** (decis/quantil/
árvore/optbin), com o número de ratings escolhido pelo usuário.

Uso típico::

    from yggdrasil.credit_risk.model import ModelSegmenter

    seg = ModelSegmenter(df, target="target", task_type="classification",
                         sample_col="amostra", ref_sample="DES")
    seg.variable_iv()                 # ranking para seleção de variáveis
    seg.auto_select(min_iv=0.02)      # categoriza/decide o que entra
    seg.fit("logistica")              # treina o modelo
    seg.metrics()                     # KS/AUC/Gini/Acc/F1 por amostra
    seg.build_ratings("quantil", n_ratings=10)
    seg.rating_table()

A interface interativa (ipywidgets, dentro do Jupyter/Databricks) é opcional e
carregada sob demanda — instale com ``pip install yggdrasil[ui]``::

    from yggdrasil.credit_risk.model import ModelSegmenterUI
    ui = ModelSegmenterUI(df, target="target", task_type="classification",
                          sample_col="amostra", ref_sample="DES")
    ui
"""
from __future__ import annotations

from .segmenter import ModelSegmenter

__all__ = ["ModelSegmenter", "ModelSegmenterUI"]


def __getattr__(name):
    # Carrega a UI só quando pedida (depende de ipywidgets/IPython).
    if name == "ModelSegmenterUI":
        from .ui import ModelSegmenterUI

        return ModelSegmenterUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
