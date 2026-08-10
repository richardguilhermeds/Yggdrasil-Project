"""Métricas de avaliação de modelos (classificação e regressão), calibração,
lift/CAP, shifts e incerteza (IC bootstrap)."""

from .calibration import (
    binomial_ci,
    calibration_in_the_large,
    calibration_slope_intercept,
    reliability_table,
)
from .classification import (
    classification_metrics,
    ks_optimal_cutoff,
    ks_statistic,
)
from .lift import accuracy_ratio, cap_curve, lift_table
from .regression import regression_metrics, robust_mape, smape
from .shift import (
    compute_metrics,
    metric_by_sample,
    metric_shifts,
    sample_shifts,
    shift_significance,
)
from .uncertainty import bootstrap_metric_ci

__all__ = [
    "classification_metrics",
    "ks_statistic",
    "ks_optimal_cutoff",
    "lift_table",
    "cap_curve",
    "accuracy_ratio",
    "regression_metrics",
    "robust_mape",
    "smape",
    "compute_metrics",
    "metric_by_sample",
    "metric_shifts",
    "sample_shifts",
    "shift_significance",
    "bootstrap_metric_ci",
    "binomial_ci",
    "calibration_in_the_large",
    "calibration_slope_intercept",
    "reliability_table",
]
