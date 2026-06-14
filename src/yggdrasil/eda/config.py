"""Configuração da esteira de EDA de features.

`EDAConfig` agrupa os parâmetros que NÃO pertencem ao `ColumnConfig` (que cuida
dos nomes de colunas e amostras). Os defaults seguem a prática de risco de
crédito, mas todos são ajustáveis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

# Grade de percentis padrão (inclui caudas P1/P99 para detectar outliers/drift).
DEFAULT_PERCENTILES: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

# Sentinelas/códigos especiais comuns — NÃO aplicados por padrão (opt-in).
DEFAULT_MISSING_CODES: Tuple = (-1, -999, -9999, 9999, "", "NULL", "NA", "SEM INFORMACAO", "SEM INFORMAÇÃO")


@dataclass
class EDAConfig:
    """Parâmetros da EDA de features.

    Por padrão, apenas NaN/None/NaT contam como ausentes. Para tratar sentinelas
    (ex.: -999) como missing, passe ``missing_codes=DEFAULT_MISSING_CODES`` ou a
    sua própria lista.
    """

    # ── missing / sentinelas ───────────────────────────────────────────
    missing_codes: Tuple = ()          # vazio = só NaN real conta como missing
    missing_warn: float = 0.20         # %missing p/ flag de atenção
    missing_drop: float = 0.50         # %missing p/ sugerir descarte

    # ── poder preditivo / qualidade ────────────────────────────────────
    iv_min: float = 0.02               # IV abaixo disso = feature fraca
    iv_leakage: float = 0.50           # IV acima disso = suspeita de leakage
    leakage_auc: float = 0.95          # AUC univariado acima = suspeita de leakage
    near_constant: float = 0.99        # share do valor modal p/ quase-constante
    corr_high: float = 0.80            # |corr| p/ redundância
    vif_high: float = 5.0              # VIF p/ multicolinearidade

    # ── binning / categóricas ──────────────────────────────────────────
    n_bins: int = 10
    binning_method: str = "quantile"   # "quantile" | "tree" | "optbinning"
    rare_level_pct: float = 0.01       # nível categórico abaixo disso vira 'OUTROS'
    cat_max_unique: int = 20           # numérica com nunique <= isso pode virar categórica
    max_levels_plot: int = 15          # top-k níveis exibidos em gráficos categóricos

    # ── temporal / amostragem ──────────────────────────────────────────
    time_freq: str = "M"               # granularidade das análises por safra
    top_k: int = 20                    # nº de features p/ painéis pesados (por importância)
    sample_size: int = 5000            # subamostra p/ SHAP / mutual info / permutation

    percentiles: Tuple[float, ...] = field(default_factory=lambda: DEFAULT_PERCENTILES)


__all__ = ["EDAConfig", "DEFAULT_PERCENTILES", "DEFAULT_MISSING_CODES"]
