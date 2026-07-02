"""Gráficos da EDA de features.

Todas as figuras são criadas pela API orientada a objetos (``Figure`` + canvas
Agg) com ``_repr_png_`` próprio — mesmo padrão anti-duplicação do dashboard de
modelo. Tema steelblue/crimson de :mod:`yggdrasil.reporting.style`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from ..reporting.style import (COR_NEUTRA, COR_PRIMARIA, COR_SECUNDARIA, colormap,
                               colormap_divergente, gradient, month_year_axis)
from .bivariate import event_rate_by_bin
from .config import EDAConfig
from .dtypes import as_numeric, infer_feature_kind
from .profile import _period_series, missing_over_time, percentiles_over_time
from .stability import feature_psi_over_time


def _figure(figsize):
    """Figura OO que sempre renderiza como PNG (sem duplicar no Jupyter)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    class _F(Figure):
        def _repr_png_(self):
            from io import BytesIO
            buf = BytesIO()
            self.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            return buf.getvalue()

    fig = _F(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig


def _hist_bins(x: pd.Series, cap: int = 60):
    x = x.dropna()
    if len(x) < 10:
        return 10
    lo, hi = x.quantile(0.005), x.quantile(0.995)
    core = x[(x >= lo) & (x <= hi)]
    try:
        edges = np.histogram_bin_edges(core, bins="fd")
    except Exception:
        return 30
    return 30 if len(edges) > cap or len(edges) < 2 else edges


def plot_histogram(df, col, cfg, eda_cfg=None, problem_type=None, ax=None):
    eda_cfg = eda_cfg or EDAConfig()
    fig = None
    if ax is None:
        fig = _figure((8, 4))
        ax = fig.add_subplot(111)
    if infer_feature_kind(df[col]) == "categorical":
        vc = df[col].astype(object).value_counts(normalize=True).head(eda_cfg.max_levels_plot) * 100
        ax.bar(range(len(vc)), vc.values, color=COR_PRIMARIA, edgecolor="white")
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels([str(i) for i in vc.index], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% frequência")
    else:
        des = as_numeric(df[df[cfg.sample_col] == cfg.dev_sample][col])
        oot = as_numeric(df[df[cfg.sample_col] == cfg.oot_sample][col])
        bins = _hist_bins(as_numeric(df[col]))
        ax.hist(des.dropna(), bins=bins, density=True, alpha=0.55, color=COR_PRIMARIA, label="DES")
        ax.hist(oot.dropna(), bins=bins, density=True, alpha=0.55, color=COR_SECUNDARIA, label="OOT")
        ax.legend(fontsize=8)
        ax.set_ylabel("densidade")
    ax.set_title(f"Distribuição · {col}", fontsize=11, fontweight="bold")
    ax.set_xlabel(col)
    return fig


def plot_missing_over_time(df, col, cfg, eda_cfg=None, ax=None):
    eda_cfg = eda_cfg or EDAConfig()
    tab = missing_over_time(df, col, cfg, eda_cfg)
    fig = None
    if ax is None:
        fig = _figure((8, 4))
        ax = fig.add_subplot(111)
    x = range(len(tab))
    ax.plot(x, tab["pct_missing"] * 100, marker="o", color=COR_PRIMARIA, lw=2)
    if tab.attrs.get("quebra"):
        ax.set_title(f"% missing no tempo · {col}  (QUEBRA detectada)", fontsize=11,
                     fontweight="bold", color=COR_SECUNDARIA)
    else:
        ax.set_title(f"% missing no tempo · {col}", fontsize=11, fontweight="bold")
    ax.set_ylabel("% missing")
    ax.set_xlabel("safra")
    month_year_axis(ax, tab["periodo"])          # eixo X em mmm/aa (padrão do repo)
    ax.tick_params(axis="x", rotation=45)
    return fig


def plot_percentile_fan(df, col, cfg, eda_cfg=None, ax=None):
    eda_cfg = eda_cfg or EDAConfig()
    tab = percentiles_over_time(df, col, cfg, eda_cfg)
    fig = None
    if ax is None:
        fig = _figure((8, 4))
        ax = fig.add_subplot(111)
    x = range(len(tab))
    pares = [("p10", "p90"), ("p25", "p75")]
    for lo, hi in pares:
        if lo in tab.columns and hi in tab.columns:
            ax.fill_between(list(x), tab[lo], tab[hi], color=COR_PRIMARIA, alpha=0.18)
    if "p50" in tab.columns:
        ax.plot(list(x), tab["p50"], marker="o", color=COR_SECUNDARIA, lw=2, label="mediana")
        ax.legend(fontsize=8)
    ax.set_title(f"Percentis no tempo · {col}", fontsize=11, fontweight="bold")
    ax.set_ylabel(col)
    ax.set_xlabel("safra")
    month_year_axis(ax, tab["periodo"])          # eixo X em mmm/aa (padrão do repo)
    ax.tick_params(axis="x", rotation=45)
    return fig


def plot_bivariate(df, col, cfg, eda_cfg=None, problem_type=None, ax=None):
    eda_cfg = eda_cfg or EDAConfig()
    tab = event_rate_by_bin(df, col, cfg, eda_cfg, problem_type)
    fig = None
    if ax is None:
        fig = _figure((9, 4.5))
        ax = fig.add_subplot(111)
    x = range(len(tab))
    ax.bar(x, tab["pct"] * 100, color=COR_PRIMARIA, alpha=0.45, label="% volume")
    ax.set_ylabel("% volume", color=COR_PRIMARIA)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(b) for b in tab["bin"]], rotation=45, ha="right", fontsize=8)
    val_col = "event_rate" if "event_rate" in tab.columns else ("target_medio" if "target_medio" in tab.columns else None)
    if val_col:
        ax2 = ax.twinx()
        ax2.plot(list(x), tab[val_col], color=COR_SECUNDARIA, marker="o", lw=2)
        if "lower_ci" in tab.columns:
            ax2.fill_between(list(x), tab["lower_ci"], tab["upper_ci"], color=COR_SECUNDARIA, alpha=0.15)
        ax2.set_ylabel("event rate / target médio", color=COR_SECUNDARIA)
        ax2.grid(False)
    ax.set_title(f"Bivariada · {col}", fontsize=11, fontweight="bold")
    ax.set_xlabel("bin")
    return fig


def plot_woe(table: pd.DataFrame, title: str = "WoE por bin", ax=None):
    fig = None
    if ax is None:
        fig = _figure((8, 4))
        ax = fig.add_subplot(111)
    if "woe" not in table.columns:
        ax.text(0.5, 0.5, "WoE indisponível (sem target binário)", ha="center", va="center")
        ax.axis("off")
        return fig
    w = table["woe"].values
    cores = [COR_PRIMARIA if v >= 0 else COR_SECUNDARIA for v in w]
    ax.bar(range(len(w)), w, color=cores, edgecolor="white")
    ax.axhline(0, color=COR_NEUTRA, lw=1)
    ax.set_xticks(range(len(w)))
    ax.set_xticklabels([str(b) for b in table["bin"]], rotation=45, ha="right", fontsize=8)
    iv = table.attrs.get("iv")
    ax.set_title(f"{title}" + (f"  (IV={iv:.3f})" if iv is not None else ""), fontsize=11, fontweight="bold")
    ax.set_ylabel("WoE")
    return fig


def plot_feature_psi_over_time(df, col, cfg, eda_cfg=None, ax=None):
    eda_cfg = eda_cfg or EDAConfig()
    from ..monitoring.psi import PSI_SIGNIFICANT, PSI_STABLE
    tab = feature_psi_over_time(df, col, cfg, eda_cfg)
    fig = None
    if ax is None:
        fig = _figure((8, 4))
        ax = fig.add_subplot(111)
    x = range(len(tab))
    ax.plot(x, tab["psi"], marker="o", color=COR_PRIMARIA, lw=2)
    ax.axhline(PSI_STABLE, color=COR_NEUTRA, ls="--", lw=1)
    ax.axhline(PSI_SIGNIFICANT, color=COR_SECUNDARIA, ls="--", lw=1)
    ax.set_title(f"PSI no tempo · {col}", fontsize=11, fontweight="bold")
    ax.set_ylabel("PSI")
    ax.set_xlabel("safra")
    month_year_axis(ax, tab["periodo"])          # eixo X em mmm/aa (padrão do repo)
    ax.tick_params(axis="x", rotation=45)
    return fig


def plot_correlation_heatmap(matrix: pd.DataFrame, title: str = "Correlação", ax=None):
    fig = None
    if ax is None:
        fig = _figure((1.0 + 0.5 * len(matrix), 0.8 + 0.5 * len(matrix)))
        ax = fig.add_subplot(111)
    if matrix.empty:
        ax.text(0.5, 0.5, "sem features suficientes", ha="center")
        ax.axis("off")
        return fig
    # Divergente centrado no branco: correlação 0 fica neutra (o sequencial
    # mapeava 0 num tom intermediário sem significado).
    im = ax.imshow(matrix.values, cmap=colormap_divergente(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    if len(matrix) <= 15:  # anota os valores quando a matriz é legível
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                v = matrix.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > 0.6 else "#333333")
    if fig is not None:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11, fontweight="bold")
    return fig


def plot_importance_ranking(ranking: pd.DataFrame, score_col: str = "score", top_n: int = 20, ax=None):
    fig = None
    if ax is None:
        fig = _figure((8, 0.4 * min(len(ranking), top_n) + 1.5))
        ax = fig.add_subplot(111)
    col = score_col if score_col in ranking.columns else ("iv" if "iv" in ranking.columns else None)
    if col is None:
        ax.text(0.5, 0.5, "sem métrica de importância", ha="center")
        ax.axis("off")
        return fig
    d = ranking.dropna(subset=[col]).sort_values(col, ascending=True).tail(top_n)
    cores = gradient(len(d))
    ax.barh(range(len(d)), d[col].values, color=cores, edgecolor="white")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["feature"], fontsize=8)
    ax.set_xlabel(f"importância ({col})")
    ax.set_title("Ranking de importância de features", fontsize=11, fontweight="bold")
    return fig


def plot_feature_panel(df, col, cfg, eda_cfg=None, problem_type=None, resumo: Optional[dict] = None):
    """Painel consolidado por feature (histograma + bivariada + missing/PSI + percentis)."""
    import matplotlib.gridspec as gridspec
    eda_cfg = eda_cfg or EDAConfig()
    fig = _figure((16, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.22,
                           top=1 - 0.9 / 8, bottom=0.07)
    titulo = f"Feature: {col}"
    if resumo:
        partes = [f"{k}={v}" for k, v in resumo.items()]
        titulo += "   |   " + "  ·  ".join(partes)
    fig.suptitle(titulo, fontsize=13, fontweight="bold", y=1 - 0.28 / 8)

    plot_histogram(df, col, cfg, eda_cfg, problem_type, ax=fig.add_subplot(gs[0, 0]))
    plot_bivariate(df, col, cfg, eda_cfg, problem_type, ax=fig.add_subplot(gs[0, 1]))
    plot_missing_over_time(df, col, cfg, eda_cfg, ax=fig.add_subplot(gs[1, 0]))
    plot_feature_psi_over_time(df, col, cfg, eda_cfg, ax=fig.add_subplot(gs[1, 1]))
    return fig


__all__ = [
    "plot_histogram", "plot_missing_over_time", "plot_percentile_fan", "plot_bivariate",
    "plot_woe", "plot_feature_psi_over_time", "plot_correlation_heatmap",
    "plot_importance_ranking", "plot_feature_panel",
]
