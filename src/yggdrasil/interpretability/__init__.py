"""Interpretabilidade de modelos (SHAP)."""

from .shap_explain import (
    compute_shap,
    save_shap_plots,
    shap_feature_importance,
    shap_report,
)

__all__ = [
    "compute_shap",
    "shap_feature_importance",
    "save_shap_plots",
    "shap_report",
]
