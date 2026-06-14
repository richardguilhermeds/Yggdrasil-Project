"""Configuração de colunas e amostras da esteira de ML.

A esteira opera sobre uma única tabela (pandas) contendo:

* colunas de *features* com um prefixo comum (padrão ``feat_``);
* uma coluna de data/safra (padrão ``dt_ref``);
* uma coluna de amostra (padrão ``amostra``);
* uma coluna com a variável resposta (padrão ``target``).

A coluna de amostra define o *papel* de cada linha. As amostras listadas em
``analysis_samples`` (por padrão ``DES`` e ``OOT``) recebem a análise completa —
métricas, *shifts*, PSI, SHAP e relatórios. Qualquer outra amostra presente na
coluna (ex.: ``SIMUL``, ``BACKTEST``) é tratada como *scoring-only*: a esteira
apenas gera a predição e atribui o grupo homogêneo, sem nenhuma análise extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

PROBLEM_TYPES = ("classification", "regression")


@dataclass
class ColumnConfig:
    """Mapeia os nomes de colunas e os papéis de amostra usados pela esteira.

    Parameters
    ----------
    feature_prefix:
        Prefixo que identifica as colunas de features (ex.: ``feat_``).
    date_col:
        Coluna de data/safra usada nas análises temporais (PSI ao longo do tempo).
    sample_col:
        Coluna que identifica a amostra de cada linha.
    target_col:
        Coluna com a variável resposta observada.
    score_col:
        Nome da coluna onde a predição do modelo é gravada.
    dev_sample, oot_sample:
        Rótulos das amostras de desenvolvimento e *out-of-time*.
    analysis_samples:
        Conjunto de amostras que recebem análise completa. Amostras fora desse
        conjunto são *scoring-only*.
    """

    feature_prefix: str = "feat_"
    date_col: str = "dt_ref"
    sample_col: str = "amostra"
    target_col: str = "target"
    score_col: str = "prediction"
    dev_sample: str = "DES"
    oot_sample: str = "OOT"
    analysis_samples: Tuple[str, ...] = ("DES", "OOT")

    def __post_init__(self) -> None:
        # Garante que dev/oot estejam sempre entre as amostras de análise.
        amostras = list(self.analysis_samples)
        for s in (self.dev_sample, self.oot_sample):
            if s not in amostras:
                amostras.append(s)
        self.analysis_samples = tuple(amostras)

    # ------------------------------------------------------------------
    def feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Lista as colunas de features (que começam com ``feature_prefix``)."""
        cols = [c for c in df.columns if c.startswith(self.feature_prefix)]
        if not cols:
            raise ValueError(
                f"Nenhuma coluna de feature encontrada com o prefixo "
                f"'{self.feature_prefix}'."
            )
        return cols

    def is_analysis_sample(self, sample: str) -> bool:
        """Indica se a amostra recebe análise completa (True) ou é scoring-only."""
        return sample in self.analysis_samples


def feature_columns(df: pd.DataFrame, cfg: ColumnConfig) -> List[str]:
    """Atalho funcional para :meth:`ColumnConfig.feature_columns`."""
    return cfg.feature_columns(df)
