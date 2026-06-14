"""Relatório por grupo homogêneo (rating).

Para cada metodologia de rating, produz uma tabela com volume, representatividade
(global e por amostra), média prevista (score) e média observada (target) por
grupo, faixa de score e verificação de monotonicidade.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from ..data import analysis_samples_present


def group_report(
    df: pd.DataFrame,
    rating_col: str,
    cfg: ColumnConfig,
    problem_type: str = "regression",
    samples: Optional[Sequence[str]] = None,
    digits: int = 6,
) -> pd.DataFrame:
    """Tabela analítica por grupo homogêneo de um rating.

    Linhas ordenadas pelo score médio crescente (mesma ordem dos rótulos).
    """
    samples = list(samples) if samples is not None else analysis_samples_present(df, cfg)
    sc, tg = cfg.score_col, cfg.target_col

    g = df.groupby(rating_col, observed=True)
    rep = pd.DataFrame(
        {
            "volume": g.size(),
            "score_medio": g[sc].mean(),
            "target_medio": g[tg].mean(),
            "score_min": g[sc].min(),
            "score_max": g[sc].max(),
        }
    )
    rep["pct_volume"] = 100 * rep["volume"] / rep["volume"].sum()

    for s in samples:
        sub = df[df[cfg.sample_col] == s]
        if len(sub) == 0:
            continue
        gs = sub.groupby(rating_col, observed=True)
        rep[f"vol_{s}"] = gs.size()
        rep[f"pct_{s}"] = 100 * gs.size() / len(sub)
        rep[f"target_medio_{s}"] = gs[tg].mean()
        rep[f"score_medio_{s}"] = gs[sc].mean()

    rep = rep.sort_values("score_medio").reset_index().rename(columns={rating_col: "rating"})
    # ordem de colunas amigável
    front = ["rating", "volume", "pct_volume", "score_medio", "target_medio",
             "score_min", "score_max"]
    cols = front + [c for c in rep.columns if c not in front]
    rep = rep[cols]
    num_cols = rep.select_dtypes(include="number").columns
    rep[num_cols] = rep[num_cols].round(digits)
    return rep


def is_monotonic(report: pd.DataFrame, col: str = "target_medio") -> bool:
    """Indica se a média observada é monotonicamente crescente entre grupos."""
    valores = report[col].dropna().to_numpy()
    return bool(np.all(np.diff(valores) >= 0))


def group_reports_all(
    df: pd.DataFrame,
    rating_cols: Sequence[str],
    cfg: ColumnConfig,
    problem_type: str = "regression",
) -> Dict[str, pd.DataFrame]:
    """Gera um relatório por metodologia de rating (``{rating_col: DataFrame}``)."""
    return {col: group_report(df, col, cfg, problem_type) for col in rating_cols}


def group_reports_to_html(
    reports: Dict[str, pd.DataFrame],
    title: str = "Relatório de Grupos Homogêneos",
) -> str:
    """Concatena os relatórios em um HTML simples (para artefato no MLflow)."""
    partes: List[str] = [
        "<html><head><meta charset='utf-8'>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;}"
        "table{border-collapse:collapse;margin-bottom:28px;font-size:13px;}"
        "th,td{border:1px solid #ddd;padding:6px 10px;text-align:right;}"
        "th{background:#4C72B0;color:#fff;}h2{color:#2c3e50;}"
        "caption{font-weight:bold;margin-bottom:6px;}</style></head><body>",
        f"<h1>{title}</h1>",
    ]
    for col, rep in reports.items():
        metodo = col.replace("rating_", "")
        mono = "monotônico OK" if is_monotonic(rep) else "NÃO monotônico"
        partes.append(f"<h2>Rating: {metodo} &nbsp;<small>({mono})</small></h2>")
        partes.append(rep.to_html(index=False, border=0))
    partes.append("</body></html>")
    return "\n".join(partes)
