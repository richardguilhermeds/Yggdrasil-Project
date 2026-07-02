"""Gráficos da seleção de features.

Mesmo padrão visual da EDA (:mod:`yggdrasil.eda.plots`): figuras pela API orientada
a objetos (``Figure`` + canvas Agg) com ``_repr_png_`` próprio (anti-duplicação no
Jupyter) e tema steelblue/crimson de :mod:`yggdrasil.reporting.style`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..reporting.style import (COR_NEUTRA, COR_PRIMARIA, COR_SECUNDARIA, colormap,
                               colormap_divergente, gradient)


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


def _empty(ax, msg: str):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10)
    ax.axis("off")


def plot_book_selection(book_table: pd.DataFrame, book_name: str, top_k: int = 15):
    """Barras horizontais de importância por feature do book, destacando selecionadas.

    Selecionadas em steelblue, descartadas em cinza; o motivo do descarte é anotado.
    Este é o gráfico "variáveis selecionadas por book".
    """
    fig = _figure((9, 0.45 * min(len(book_table), top_k) + 1.8))
    ax = fig.add_subplot(111)
    if book_table is None or book_table.empty:
        _empty(ax, "book vazio")
        return fig

    d = book_table.copy()
    sel = d["selecionada"].fillna(False) if "selecionada" in d.columns else pd.Series(False, index=d.index)
    val = d["score"].astype(float) if "score" in d.columns else pd.Series(np.nan, index=d.index)
    # ordena por selecionada e score; mantém top_k mais relevantes
    d = d.assign(_sel=sel.values, _val=val.fillna(-np.inf).values)
    d = d.sort_values(["_sel", "_val"]).tail(top_k)

    y = range(len(d))
    cores = [COR_PRIMARIA if s else COR_NEUTRA for s in d["_sel"]]
    larguras = np.where(np.isfinite(d["_val"]), d["_val"], 0.0)
    ax.barh(list(y), larguras, color=cores, edgecolor="white")
    ax.set_yticks(list(y))
    ax.set_yticklabels(d["feature"], fontsize=8)
    ax.set_xlabel("importância (score de rank médio)")
    n_sel = int(sel.sum())
    ax.set_title(f"Book '{book_name}' · {n_sel}/{len(book_table)} selecionadas",
                 fontsize=11, fontweight="bold")

    if "motivo" in d.columns:
        xmax = max(larguras.max(), 0.0) if len(larguras) else 0.0
        for i, (_, r) in enumerate(d.iterrows()):
            if not r["_sel"] and isinstance(r["motivo"], str):
                ax.text(xmax * 0.02, i, f"  {r['motivo']}", va="center", fontsize=7, color=COR_SECUNDARIA)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COR_PRIMARIA, label="selecionada"),
                       Patch(color=COR_NEUTRA, label="descartada")], fontsize=8, loc="lower right")
    return fig


def plot_book_overview(selection_table: pd.DataFrame):
    """Barras empilhadas: selecionadas × descartadas por book."""
    fig = _figure((8, 4.5))
    ax = fig.add_subplot(111)
    if selection_table is None or selection_table.empty or "book" not in selection_table.columns:
        _empty(ax, "sem dados de seleção")
        return fig

    sel = selection_table["selecionada"].fillna(False)
    g = selection_table.assign(_sel=sel.values).groupby("book")["_sel"].agg(["sum", "count"])
    g["desc"] = g["count"] - g["sum"]
    x = range(len(g))
    ax.bar(list(x), g["sum"], color=COR_PRIMARIA, label="selecionadas")
    ax.bar(list(x), g["desc"], bottom=g["sum"], color=COR_NEUTRA, label="descartadas")
    ax.set_xticks(list(x))
    ax.set_xticklabels(g.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("nº de features")
    ax.set_title("Seleção por book", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    return fig


def plot_overall_importance(overall: pd.DataFrame, top_k: int = 25):
    """Top-K features mais importantes no geral (para entrar nos modelos)."""
    fig = _figure((9, 0.4 * min(len(overall), top_k) + 1.8))
    ax = fig.add_subplot(111)
    if overall is None or overall.empty:
        _empty(ax, "nenhuma feature selecionada")
        return fig

    col = "rf_importance" if ("rf_importance" in overall.columns and overall["rf_importance"].notna().any()) else "score"
    if col not in overall.columns or overall[col].notna().sum() == 0:
        _empty(ax, "sem métrica de importância")
        return fig
    d = overall.dropna(subset=[col]).sort_values(col, ascending=True).tail(top_k)
    cores = gradient(len(d))
    ax.barh(range(len(d)), d[col].values, color=cores, edgecolor="white")
    rotulos = (d["feature"] + "  [" + d["book"].astype(str) + "]") if "book" in d.columns else d["feature"]
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(rotulos, fontsize=8)
    ax.set_xlabel(f"importância ({col})")
    ax.set_title("Features mais importantes (global)", fontsize=11, fontweight="bold")
    return fig


def plot_corr_heatmap(matrix: pd.DataFrame, title: str = "Correlação"):
    """Heatmap de correlação (Spearman) entre as features do book."""
    n = len(matrix) if matrix is not None else 0
    fig = _figure((1.2 + 0.5 * n, 1.0 + 0.5 * n))
    ax = fig.add_subplot(111)
    if matrix is None or matrix.empty:
        _empty(ax, "sem features suficientes")
        return fig
    # Divergente centrado no branco: correlação 0 fica neutra (o sequencial
    # mapeava 0 num tom intermediário sem significado).
    im = ax.imshow(matrix.values, cmap=colormap_divergente(), vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(matrix.index, fontsize=8)
    if n <= 15:  # anota os valores quando a matriz é legível
        for i in range(n):
            for j in range(n):
                v = matrix.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > 0.6 else "#333333")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11, fontweight="bold")
    return fig


__all__ = [
    "plot_book_selection", "plot_book_overview", "plot_overall_importance", "plot_corr_heatmap",
]
