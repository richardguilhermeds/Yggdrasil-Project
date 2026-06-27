"""Configuração da esteira de seleção de features (PySpark, por *books*).

`FeatureSelectionConfig` agrupa os parâmetros que NÃO pertencem ao `ColumnConfig`
(que cuida dos nomes de colunas e amostras). Os defaults seguem a prática de
risco de crédito e espelham os limiares já usados na EDA de features
(:class:`yggdrasil.eda.config.EDAConfig`), mas todos são ajustáveis.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeatureSelectionConfig:
    """Parâmetros da seleção de features sobre um Spark DataFrame.

    A seleção é feita por *book* (grupo de features de mesma origem, ex.: serasa,
    bvs). Cada feature passa por filtros duros (missing, variância, redundância) e
    depois por uma avaliação de importância + Boruta, consolidadas num **consenso**.
    """

    # ── missing ────────────────────────────────────────────────────────
    missing_max: float = 0.50          # %missing acima disso => descartar

    # ── variância (teste p_low == p_high; por padrão P1 vs P99) ─────────
    var_p_low: float = 0.01            # percentil inferior do teste de variância
    var_p_high: float = 0.99           # percentil superior do teste de variância
    var_tol: float = 0.0               # sem variância se (p_high - p_low) <= var_tol
    near_constant: float = 0.99        # share do valor modal p/ quase-constante

    # ── correlação / redundância ────────────────────────────────────────
    corr_high: float = 0.80            # |corr| p/ considerar duas features redundantes
    corr_target_min: float = 0.0       # |corr| mínimo com o alvo (0 = sem exigência)

    # ── poder preditivo / qualidade (classificação) ─────────────────────
    iv_min: float = 0.02               # IV abaixo disso = feature fraca
    iv_leakage: float = 0.50           # IV acima disso = suspeita de leakage
    leakage_auc: float = 0.95          # AUC univariado acima = suspeita de leakage

    # ── RandomForest (pyspark.ml) ───────────────────────────────────────
    rf_n_estimators: int = 100
    rf_max_depth: int = 6
    rf_subsampling: float = 1.0
    rf_seed: int = 42

    # ── Boruta ──────────────────────────────────────────────────────────
    boruta_enable: bool = True
    boruta_max_iter: int = 50
    boruta_alpha: float = 0.05         # nível de significância do teste binomial
    boruta_perc: float = 100.0         # percentil das shadows usado como limiar (100 = máximo)

    # ── backend / amostragem ────────────────────────────────────────────
    backend: str = "spark"             # "spark" (pyspark.ml) | "driver" (sklearn)
    sample_size: int = 0               # >0 amostra N linhas p/ as etapas de modelo (0 = full)
    approx_rel_error: float = 0.01     # erro relativo do approxQuantile (0 = exato, caro)

    # ── binning (IV/KS univariado em classificação) ─────────────────────
    n_bins: int = 10

    # ── consenso ────────────────────────────────────────────────────────
    consensus_threshold: float = 0.50  # score_consenso >= isso (ou Boruta confirmada) => selecionar
    peso_importancia: float = 0.50     # peso do rank de importância no consenso
    peso_boruta: float = 0.35          # peso da taxa de hits do Boruta no consenso
    peso_alvo: float = 0.15            # peso do sinal de relação com o alvo no consenso

    # ── plots / relatório ───────────────────────────────────────────────
    top_k_book: int = 15               # nº de features por book nos painéis
    top_k_overall: int = 25            # nº de features no ranking geral

    def __post_init__(self) -> None:
        if self.backend not in ("spark", "driver"):
            raise ValueError("backend deve ser 'spark' ou 'driver'.")
        if not (0.0 <= self.var_p_low < self.var_p_high <= 1.0):
            raise ValueError("Exige 0 <= var_p_low < var_p_high <= 1.")


__all__ = ["FeatureSelectionConfig"]
