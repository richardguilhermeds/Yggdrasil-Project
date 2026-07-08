"""Orquestrador da esteira de EDA de features.

`run_feature_eda` valida (tolerante a target ausente), aplica sentinelas (opt-in),
infere o tipo de problema, consolida o perfil mestre de features com flags e
veredito, e monta um `FeatureEDAReport` (tabelas + figuras), exportável em
HTML/CSV e logável no MLflow. É um entrypoint INDEPENDENTE — não entra no
pipeline de modelo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from ..data import infer_problem_type
from ..monitoring.psi import PSI_SIGNIFICANT, PSI_STABLE
from . import bivariate, correlation, importance, plots, profile, stability
from .config import EDAConfig
from .dtypes import (
    apply_missing_codes,
    classify_features,
    has_target,
    validate_input_eda,
)


def verdict(row: dict, eda_cfg: EDAConfig) -> str:
    """Veredito automático por feature: manter | revisar | descartar."""
    psi = row.get("psi_oot", np.nan)
    if (row.get("constante")
            or (np.isfinite(row.get("pct_missing", np.nan)) and row["pct_missing"] >= eda_cfg.missing_drop)
            or (np.isfinite(psi) and psi > PSI_SIGNIFICANT)):
        return "descartar"
    motivos = (
        bool(row.get("leakage"))
        or row.get("quase_constante")
        or (np.isfinite(row.get("iv", np.nan)) and row["iv"] < eda_cfg.iv_min)
        or (np.isfinite(psi) and psi > PSI_STABLE)
        or row.get("missing_quebra")
    )
    return "revisar" if motivos else "manter"


def build_feature_profile(
    df: pd.DataFrame, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None,
    problem_type: Optional[str] = None,
) -> pd.DataFrame:
    """Tabela mestra: uma linha por feature consolidando todos os diagnósticos."""
    eda_cfg = eda_cfg or EDAConfig()
    feats = cfg.feature_columns(df)
    kinds = classify_features(df, cfg, eda_cfg)
    miss = profile.missing_summary(df, cfg, eda_cfg).set_index("feature")
    card = profile.cardinality_summary(df, feats, eda_cfg).set_index("feature")
    stab = stability.stability_summary(df, cfg, eda_cfg).set_index("feature")
    rk = importance.importance_ranking(df, cfg, eda_cfg, problem_type)
    rk_i = rk.set_index("feature") if len(rk) else pd.DataFrame()
    tem_target = has_target(df, cfg)

    rows: List[dict] = []
    for c in feats:
        mot = profile.missing_over_time(df, c, cfg, eda_cfg)
        r = {
            "feature": c, "tipo": kinds[c],
            "pct_missing": float(miss.loc[c, "pct_missing"]),
            "pct_missing_max_safra": round(float(mot["pct_missing"].max()), 4) if len(mot) else np.nan,
            "missing_quebra": bool(mot.attrs.get("quebra", False)),
            "nunique": int(card.loc[c, "nunique"]),
            "top1_share": float(card.loc[c, "top1_share"]) if np.isfinite(card.loc[c, "top1_share"]) else np.nan,
            "constante": bool(card.loc[c, "constante"]),
            "quase_constante": bool(card.loc[c, "quase_constante"]),
            "psi_oot": float(stab.loc[c, "psi_oot"]) if np.isfinite(stab.loc[c, "psi_oot"]) else np.nan,
            "psi_max_safra": float(stab.loc[c, "psi_max_safra"]) if np.isfinite(stab.loc[c, "psi_max_safra"]) else np.nan,
        }
        for m in ("iv", "ks_univ", "gini_univ", "score", "leakage_flag"):
            r[m if m != "leakage_flag" else "leakage"] = (
                rk_i.loc[c, m] if (len(rk_i) and m in rk_i.columns and c in rk_i.index) else (False if m == "leakage_flag" else np.nan)
            )
        if kinds[c] in ("numeric", "binary"):
            r["pct_outlier"] = profile.outlier_summary(df[c]).get("pct_outlier_iqr")
        else:
            r["pct_outlier"] = np.nan
        if tem_target and problem_type == "classification" and kinds[c] != "categorical":
            mono = bivariate.monotonicity_diagnostic(
                bivariate.event_rate_by_bin(df, c, cfg, eda_cfg, problem_type))
            r["trend"] = mono["trend"]
            r["n_inversoes"] = mono["n_inversoes"]
        rows.append(r)

    prof = pd.DataFrame(rows)
    prof["veredito"] = prof.apply(lambda row: verdict(row.to_dict(), eda_cfg), axis=1)
    if "score" in prof.columns and prof["score"].notna().any():
        prof = prof.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
    return prof


@dataclass
class FeatureEDAReport:
    """Resultado da esteira de EDA de features."""

    feature_profile: pd.DataFrame
    dataset_overview: Dict
    importance_ranking: pd.DataFrame
    correlation: pd.DataFrame
    panels: Dict[str, object] = field(default_factory=dict)
    problem_type: Optional[str] = None
    cfg: Optional[ColumnConfig] = None

    def to_csv(self, path: str) -> str:
        self.feature_profile.to_csv(path, index=False)
        return path

    def to_html(self, embed_panels: bool = False) -> str:
        return _to_html(self, embed_panels=embed_panels)


def run_feature_eda(
    df: pd.DataFrame,
    cfg: Optional[ColumnConfig] = None,
    eda_cfg: Optional[EDAConfig] = None,
    problem_type: Optional[str] = None,
    features: Optional[List[str]] = None,
    with_panels: bool = True,
    mlflow_experiment: Optional[str] = None,
    run_name: Optional[str] = None,
) -> FeatureEDAReport:
    """Roda a EDA de features ponta a ponta e retorna um ``FeatureEDAReport``."""
    cfg = cfg or ColumnConfig()
    eda_cfg = eda_cfg or EDAConfig()
    validate_input_eda(df, cfg)

    df = apply_missing_codes(df, cfg.feature_columns(df), eda_cfg.missing_codes)
    if problem_type is None and has_target(df, cfg):
        problem_type = infer_problem_type(df, cfg)

    overview = profile.dataset_overview(df, cfg, eda_cfg)
    prof = build_feature_profile(df, cfg, eda_cfg, problem_type)
    ranking = importance.importance_ranking(df, cfg, eda_cfg, problem_type) if has_target(df, cfg) else pd.DataFrame()
    corr = correlation.correlation_matrix(df, cfg, eda_cfg)

    panels: Dict[str, object] = {}
    if with_panels:
        alvo = features if features is not None else prof["feature"].head(eda_cfg.top_k).tolist()
        for c in alvo:
            linha = prof[prof["feature"] == c].iloc[0]
            resumo = {"tipo": linha["tipo"], "%miss": f"{linha['pct_missing']:.0%}",
                      "PSI": f"{linha['psi_oot']:.2f}" if np.isfinite(linha["psi_oot"]) else "—",
                      "veredito": linha["veredito"]}
            if np.isfinite(linha.get("iv", np.nan)):
                resumo["IV"] = f"{linha['iv']:.3f}"
            panels[c] = plots.plot_feature_panel(df, c, cfg, eda_cfg, problem_type, resumo)

    report = FeatureEDAReport(feature_profile=prof, dataset_overview=overview,
                              importance_ranking=ranking, correlation=corr,
                              panels=panels, problem_type=problem_type, cfg=cfg)
    if mlflow_experiment:
        _log_mlflow(report, mlflow_experiment, run_name)
    return report


# ── HTML / MLflow ────────────────────────────────────────────────────────
def _to_html(report: FeatureEDAReport, embed_panels: bool = False) -> str:
    import base64
    from io import BytesIO

    ov = "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in report.dataset_overview.items())
    partes = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:Arial,Helvetica,sans-serif;margin:24px;}",
        "table{border-collapse:collapse;margin-bottom:24px;font-size:12px;}",
        "th,td{border:1px solid #ddd;padding:5px 8px;text-align:right;}",
        "th{background:steelblue;color:#fff;}h1,h2{color:#2c3e50;}ul{font-size:13px;}",
        "</style></head><body>",
        "<h1>EDA de Features — Yggdrasil</h1>",
        f"<h2>Overview</h2><ul>{ov}</ul>",
        "<h2>Perfil de features (veredito)</h2>",
        report.feature_profile.to_html(index=False, border=0),
    ]
    if embed_panels and report.panels:
        partes.append("<h2>Painéis por feature</h2>")
        for c, fig in report.panels.items():
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
            b64 = base64.b64encode(buf.getvalue()).decode()
            partes.append(f"<h3>{c}</h3><img src='data:image/png;base64,{b64}' style='max-width:100%'/>")
    partes.append("</body></html>")
    return "\n".join(partes)


def _log_mlflow(report: FeatureEDAReport, experiment: str, run_name: Optional[str]) -> None:
    import tempfile
    import mlflow

    tmp = tempfile.mkdtemp(prefix="yggdrasil_eda_")
    report.to_csv(os.path.join(tmp, "feature_profile.csv"))
    with open(os.path.join(tmp, "eda_report.html"), "w", encoding="utf-8") as fh:
        fh.write(report.to_html(embed_panels=False))
    pdir = os.path.join(tmp, "panels")
    os.makedirs(pdir, exist_ok=True)
    for c, fig in report.panels.items():
        safe = "".join(ch if ch.isalnum() else "_" for ch in c)
        fig.savefig(os.path.join(pdir, f"{safe}.png"), dpi=90, bbox_inches="tight")

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"framework": "yggdrasil-eda", "stage": "feature_eda"})
        prof = report.feature_profile
        mlflow.log_metric("n_features", len(prof))
        for v in ("manter", "revisar", "descartar"):
            mlflow.log_metric(f"n_{v}", int((prof["veredito"] == v).sum()))
        mlflow.log_artifacts(tmp, artifact_path="eda")


__all__ = ["run_feature_eda", "build_feature_profile", "FeatureEDAReport", "verdict"]
