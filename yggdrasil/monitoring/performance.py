"""Métricas de discriminação/erro ao longo do tempo (por período de referência).

Complementa o PSI: enquanto o PSI acompanha a estabilidade das distribuições,
aqui acompanhamos a **performance do modelo período a período** (mês, trimestre
etc.), sinalizando degradação de cada período em relação à média dos períodos
da amostra de desenvolvimento. Reutiliza o mesmo pacote de métricas de
:mod:`yggdrasil.metrics` (classificação e regressão) e a mesma régua de flags
de :mod:`yggdrasil.metrics.shift` (``ok`` / ``atencao`` / ``degradado``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from ..metrics import compute_metrics
from ..metrics.classification import ks_optimal_cutoff
from ..metrics.shift import FLAG_ATENCAO, FLAG_DEGRADADO, shift_flag

# Colunas de métricas exibidas por tipo de problema. Sem 'ks_cutoff': o corte
# não é métrica de performance (mesmo critério de metrics/shift.py).
_METRIC_COLS: Dict[str, List[str]] = {
    "classification": ["auc", "gini", "ks", "accuracy", "f1", "precision",
                       "recall", "brier", "logloss"],
    "regression": ["rmse", "mae", "mape", "smape", "medae", "r2", "mean_bias"],
}

# Métrica que dirige a flag de degradação por padrão.
_FLAG_METRIC_DEFAULT = {"classification": "ks", "regression": "rmse"}


def _group_metrics(
    y: np.ndarray,
    sc: np.ndarray,
    problem_type: str,
    min_n: int,
    cutoff: Optional[float],
) -> Tuple[Dict[str, float], str]:
    """Pacote de métricas de um período: ``(dict, nota)``.

    Período pequeno (``n < min_n``) ou de classe única não quebra — as métricas
    ficam NaN e o motivo vai na ``nota`` (mesmo espírito do ``metrics_by_safra``
    dos segmentadores)."""
    met = {c: float("nan") for c in _METRIC_COLS[problem_type]}
    if y.size < min_n:
        return met, f"n < {min_n}"
    if problem_type == "classification" and len(np.unique(y)) < 2:
        return met, "classe única"
    try:
        calc = compute_metrics(y, sc, problem_type, cutoff=cutoff)
    except Exception:  # noqa: BLE001 - período degenerado ⇒ NaN com nota
        return met, "métricas não computáveis"
    met.update({c: calc.get(c, float("nan")) for c in met})
    return met, ""


def metric_over_time(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    problem_type: str,
    freq: str = "M",
    min_n: int = 200,
    *,
    flag_metric: Optional[str] = None,
    flag_atencao: float = FLAG_ATENCAO,
    flag_degradado: float = FLAG_DEGRADADO,
) -> pd.DataFrame:
    """Métricas do modelo por período de ``date_col`` (safra), com flag de degradação.

    Agrupa as linhas por período de ``cfg.date_col`` na frequência ``freq``
    (``'M'`` mensal, ``'Q'`` trimestral...) e calcula o pacote de métricas de
    :func:`~yggdrasil.metrics.compute_metrics` em cada período. Para
    classificação, o corte de classe é o KS-ótimo estimado na amostra de
    desenvolvimento inteira e reaplicado a todos os períodos (comparabilidade,
    como em :func:`~yggdrasil.metrics.metric_by_sample`).

    Colunas do retorno — classificação:
    ``[periodo, n, taxa_evento, auc, gini, ks, accuracy, f1, precision,
    recall, brier, logloss, flag, nota]``; regressão:
    ``[periodo, n, realizado_medio, previsto_medio, rmse, mae, mape, smape,
    medae, r2, mean_bias, flag, nota]``.

    * ``nota`` explica métricas NaN (``n < min_n``, classe única...); os campos
      descritivos (``n``, ``taxa_evento``/médias) saem mesmo nesses períodos.
    * ``flag`` compara a métrica ``flag_metric`` (default: ``ks`` na
      classificação, ``rmse`` na regressão) de cada período com a **média dos
      períodos da amostra de desenvolvimento**, usando a régua de
      :func:`~yggdrasil.metrics.shift.shift_flag` (``ok``/``atencao``/
      ``degradado``; ``n/a`` quando não computável — ex.: sem linhas da amostra
      de desenvolvimento). A métrica e a referência usadas ficam em
      ``df.attrs['flag_metric']`` e ``df.attrs['flag_ref']``.

    O ``df`` deve conter ``date_col``, ``target_col`` e ``score_col``;
    normalmente são as amostras de análise já scoradas (linhas de outras
    amostras presentes entram no cálculo do período em que caem).
    """
    if problem_type not in _METRIC_COLS:
        raise ValueError(f"problem_type inválido: {problem_type!r}")
    if cfg.date_col not in df.columns:
        raise ValueError(f"Coluna de data '{cfg.date_col}' não existe no DataFrame.")
    for col in (cfg.target_col, cfg.score_col):
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não existe no DataFrame.")
    flag_metric = flag_metric or _FLAG_METRIC_DEFAULT[problem_type]
    is_clf = problem_type == "classification"

    des_mask = (df[cfg.sample_col] == cfg.dev_sample
                if cfg.sample_col in df.columns
                else pd.Series(False, index=df.index))

    # Corte de classe estimado no DES inteiro e reaplicado a cada período.
    cutoff = None
    if is_clf and des_mask.any():
        des = df[des_mask]
        ok = des[cfg.target_col].notna() & des[cfg.score_col].notna()
        if ok.any() and des.loc[ok, cfg.target_col].nunique() > 1:
            cutoff = ks_optimal_cutoff(des.loc[ok, cfg.target_col],
                                       des.loc[ok, cfg.score_col])

    periodos = pd.to_datetime(df[cfg.date_col], errors="coerce").dt.to_period(freq)
    rows: List[dict] = []
    ref_vals: List[float] = []  # métrica da flag nos períodos do DES
    for per, g in df.groupby(periodos):        # groupby descarta período NaT
        y = g[cfg.target_col].to_numpy(dtype="float64")
        sc = g[cfg.score_col].to_numpy(dtype="float64")
        ok = ~np.isnan(y) & ~np.isnan(sc)
        y, sc = y[ok], sc[ok]
        row = {"periodo": str(per), "n": int(y.size)}
        if is_clf:
            row["taxa_evento"] = float(np.mean(y)) if y.size else float("nan")
        else:
            row["realizado_medio"] = float(np.mean(y)) if y.size else float("nan")
            row["previsto_medio"] = float(np.mean(sc)) if sc.size else float("nan")
        met, nota = _group_metrics(y, sc, problem_type, min_n, cutoff)
        row.update(met)
        row["nota"] = nota
        rows.append(row)

        # Referência da flag: mesmo cálculo restrito às linhas DES do período.
        gmask = des_mask.loc[g.index]
        if gmask.all():                        # período inteiramente DES: reusa
            v = met.get(flag_metric, float("nan"))
        elif gmask.any():
            gd = g[gmask]
            yd = gd[cfg.target_col].to_numpy(dtype="float64")
            scd = gd[cfg.score_col].to_numpy(dtype="float64")
            okd = ~np.isnan(yd) & ~np.isnan(scd)
            metd, _ = _group_metrics(yd[okd], scd[okd], problem_type, min_n, cutoff)
            v = metd.get(flag_metric, float("nan"))
        else:
            v = float("nan")
        if np.isfinite(v):
            ref_vals.append(float(v))

    ref = float(np.mean(ref_vals)) if ref_vals else float("nan")
    for row in rows:
        flag = shift_flag(flag_metric, ref, row.get(flag_metric, float("nan")),
                          atencao=flag_atencao, degradado=flag_degradado)
        row["flag"] = flag if flag is not None else "n/a"

    base_cols = ["periodo", "n"] + (["taxa_evento"] if is_clf
                                    else ["realizado_medio", "previsto_medio"])
    cols = base_cols + _METRIC_COLS[problem_type] + ["flag", "nota"]
    out = (pd.DataFrame(rows, columns=cols)
           .sort_values("periodo").reset_index(drop=True))
    out.attrs["flag_metric"] = flag_metric
    out.attrs["flag_ref"] = ref
    return out


def plot_metric_over_time(
    df_result: pd.DataFrame,
    metric: str,
    *,
    figsize: Tuple[float, float] = (10, 4),
    dpi: int = 110,
    ax=None,
    save_path: Optional[str] = None,
):
    """Evolução de uma métrica de :func:`metric_over_time` (linha por período).

    Quando ``metric`` é a métrica que dirigiu a flag (``attrs['flag_metric']``),
    destaca os períodos em ``atencao``/``degradado`` e traça a linha de
    referência (média dos períodos da amostra de desenvolvimento)."""
    import matplotlib.pyplot as plt

    from ..reporting.style import (COR_NEUTRA, COR_PRIMARIA, COR_SECUNDARIA,
                                   month_year_axis)

    if metric not in df_result.columns or metric in ("periodo", "n", "flag", "nota"):
        raise ValueError(f"métrica '{metric}' não existe no resultado")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    x = range(len(df_result))
    ax.plot(x, df_result[metric], marker="o", color=COR_PRIMARIA, linewidth=2)

    if "flag" in df_result.columns and df_result.attrs.get("flag_metric") == metric:
        # '#caa000' = âmbar de atenção já usado nos gráficos do repositório
        for rotulo, cor in (("atencao", "#caa000"), ("degradado", COR_SECUNDARIA)):
            m = (df_result["flag"] == rotulo).to_numpy()
            if m.any():
                ax.scatter(np.flatnonzero(m), df_result.loc[m, metric],
                           color=cor, zorder=3, s=45, label=rotulo)
        ref = df_result.attrs.get("flag_ref")
        if ref is not None and np.isfinite(ref):
            ax.axhline(ref, color=COR_NEUTRA, ls="--", lw=1, label="média DES")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)

    ax.set_title(f"{metric.upper()} ao longo do tempo", fontweight="bold")
    ax.set_ylabel(metric)
    ax.set_xlabel("Período de referência")
    month_year_axis(ax, df_result["periodo"])    # eixo X em mmm/aa (padrão do repo)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45); lbl.set_ha("right")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


__all__ = ["metric_over_time", "plot_metric_over_time"]
