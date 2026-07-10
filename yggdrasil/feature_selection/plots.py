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


def _style_ax(ax):
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, linewidth=0.8)


def _short(feat) -> str:
    return str(feat).replace("feat_", "")


def _num(d, col):
    return pd.to_numeric(d.get(col), errors="coerce") if col in getattr(d, "columns", []) \
        else pd.Series(np.nan, index=d.index)


# Estágios do funil, na ordem real do pipeline.
_STAGE_ORDER = [
    ("alto missing", "Missing"),
    ("variância", "Variância"),
    ("redundante", "Redundância"),
    ("leakage", "Leakage"),
    ("Boruta/consenso", "Boruta / Consenso"),
]


def _stage_of(motivo) -> str:
    if not isinstance(motivo, str):
        return "?"
    m = motivo.lower()
    if "alto missing" in m:
        return "alto missing"
    if "variância" in m or "variancia" in m or "constante" in m:
        return "variância"
    if m.startswith("redundante"):
        return "redundante"
    if "leakage" in m:
        return "leakage"
    if "boruta rejeitada" in m or "consenso abaixo" in m:
        return "Boruta/consenso"
    return "selecionada"


def _signal_series(d: pd.DataFrame, problem_type=None):
    """Sinal univariado por feature: KS (classif) → IV → |corr_target|. Retorna (serie, rótulo)."""
    if problem_type != "regression" and "ks" in d.columns and _num(d, "ks").notna().any():
        return _num(d, "ks"), "KS (sinal univariado)"
    if problem_type != "regression" and "iv" in d.columns and _num(d, "iv").notna().any():
        return _num(d, "iv"), "IV (sinal univariado)"
    return _num(d, "corr_target").abs(), "|correlação com o alvo|"


# ── 1) Funil de atrito ───────────────────────────────────────────────────
def _draw_funnel(ax, selection_table: pd.DataFrame):
    if selection_table is None or selection_table.empty or "motivo" not in selection_table.columns:
        _empty(ax, "sem dados de seleção")
        return
    stages = selection_table["motivo"].map(_stage_of)
    total = len(selection_table)
    drops = {key: int((stages == key).sum()) for key, _ in _STAGE_ORDER}
    n_sel = int((stages == "selecionada").sum())

    labels = ["Candidatas"] + [lab for _, lab in _STAGE_ORDER] + ["Selecionadas"]
    remaining, cur = [total], total
    for key, _ in _STAGE_ORDER:
        cur -= drops[key]
        remaining.append(cur)
    remaining.append(n_sel)

    y = np.arange(len(labels))[::-1]
    cores = gradient(len(labels))
    half = total / 2 if total else 1
    for i, (lab, rem) in enumerate(zip(labels, remaining)):
        yi = y[i]
        ax.barh(yi, rem, left=-rem / 2, color=cores[i], edgecolor="white", height=0.66)
        ax.text(0, yi, f"{rem}", ha="center", va="center", fontsize=10,
                fontweight="bold", color="white")
        pct = 100 * rem / total if total else 0
        ax.text(half * 1.10, yi, f"{pct:.0f}%", ha="left", va="center", fontsize=8, color="#666")
        if 1 <= i <= len(_STAGE_ORDER):
            d = drops[_STAGE_ORDER[i - 1][0]]
            if d:
                ax.text(half * 1.38, yi, f"−{d}", ha="left", va="center", fontsize=8,
                        color=COR_SECUNDARIA, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks([])
    ax.set_xlim(-half * 1.15, half * 1.7)
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    taxa = 100 * n_sel / total if total else 0
    ax.set_title(f"Funil de seleção · {n_sel}/{total} sobreviveram ({taxa:.0f}%)",
                 fontsize=12, fontweight="bold")


def plot_selection_funnel(selection_table: pd.DataFrame):
    """Funil de atrito: quantas features restam após cada etapa do pipeline.

    Parte do total de candidatas e subtrai, na ordem real (missing → variância →
    redundância → leakage → Boruta/consenso), quantas caíram em cada etapa (do
    ``motivo``). Barras centradas = sobreviventes; ``−k`` em crimson = descartes.
    """
    fig = _figure((9, 4.8))
    _draw_funnel(fig.add_subplot(111), selection_table)
    return fig


# ── 2) Mapa de decisão ───────────────────────────────────────────────────
def _draw_decision_map(ax, selection_table: pd.DataFrame, problem_type=None, annotate_top=6):
    _style_ax(ax)
    if selection_table is None or selection_table.empty:
        _empty(ax, "sem dados de seleção")
        return
    d = selection_table.copy()
    x = _num(d, "rf_importance")
    ysig, ylab = _signal_series(d, problem_type)
    d = d.assign(_x=x, _y=ysig)
    d = d[np.isfinite(d["_x"]) & np.isfinite(d["_y"])]
    if d.empty:
        _empty(ax, "sem métricas numéricas para o mapa")
        return
    sel = d["selecionada"].fillna(False).astype(bool)
    leak = d["leakage_flag"].fillna(False).astype(bool) if "leakage_flag" in d else pd.Series(False, index=d.index)
    sz = _num(d, "score_consenso").fillna(0.3)
    sizes = 55 + 320 * np.clip(sz, 0, 1)

    ax.scatter(d.loc[sel & ~leak, "_x"], d.loc[sel & ~leak, "_y"], s=sizes[sel & ~leak],
               c=COR_PRIMARIA, alpha=0.75, edgecolor="white", linewidth=0.8, label="selecionada", zorder=3)
    ax.scatter(d.loc[~sel & ~leak, "_x"], d.loc[~sel & ~leak, "_y"], s=sizes[~sel & ~leak],
               c=COR_NEUTRA, alpha=0.55, edgecolor="white", linewidth=0.6, label="descartada", zorder=2)
    if leak.any():
        ax.scatter(d.loc[leak, "_x"], d.loc[leak, "_y"], s=230, marker="*", c=COR_SECUNDARIA,
                   edgecolor="white", linewidth=0.8, label="leakage (barrada)", zorder=4)
    mx, my = d["_x"].median(), d["_y"].median()
    ax.axvline(mx, color="#ccc", lw=1, ls="--", zorder=1)
    ax.axhline(my, color="#ccc", lw=1, ls="--", zorder=1)

    top = d.sort_values("_x", ascending=False).head(annotate_top)
    for k, (_, r) in enumerate(top.iterrows()):
        to_left = r["_x"] >= mx
        ax.annotate(_short(r["feature"]), (r["_x"], r["_y"]), fontsize=7,
                    xytext=(-6 if to_left else 6, 8 if k % 2 else -9),
                    textcoords="offset points", ha="right" if to_left else "left",
                    color="#333", zorder=5)
    ax.set_xlabel("importância multivariada (rf_importance)", fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title("Mapa de decisão · importância × sinal (tamanho = consenso)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.75)


def plot_decision_map(selection_table: pd.DataFrame, problem_type=None, annotate_top: int = 6):
    """Dispersão importância multivariada (x=rf_importance) × sinal univariado (y).

    Azul = selecionada, cinza = descartada, estrela crimson = leakage. Tamanho ∝
    consenso. Linhas de mediana formam quadrantes: alto-alto = âncoras; alto-x /
    baixo-y = revisar (interação/leakage); descartada de alto sinal = ver ``motivo``.
    """
    fig = _figure((8.5, 6))
    _draw_decision_map(fig.add_subplot(111), selection_table, problem_type, annotate_top)
    return fig


# ── 3) Boruta: hits × bandas binomiais ───────────────────────────────────
def _draw_boruta(ax, selection_table: pd.DataFrame, n_iter: int, alpha=0.05):
    from scipy.stats import binom

    if selection_table is None or "boruta_hits" not in selection_table.columns:
        _empty(ax, "Boruta não executado")
        return
    d = selection_table.copy()
    d = d[_num(d, "boruta_hits").notna()]
    if d.empty:
        _empty(ax, "Boruta não executado (sem hits)")
        return
    d = d.assign(_h=_num(d, "boruta_hits").astype(int)).sort_values("_h")
    cor_dec = {"confirmada": COR_PRIMARIA, "tentativa": COR_NEUTRA, "rejeitada": COR_SECUNDARIA}
    cores = [cor_dec.get(v, COR_NEUTRA) for v in d.get("boruta_decisao", pd.Series(index=d.index))]
    y = np.arange(len(d))

    n_feat = max(len(d), 1)
    alpha_c = alpha / n_feat
    ks = np.arange(0, n_iter + 1)
    accept = next((int(k) for k in ks if binom.sf(k - 1, n_iter, 0.5) <= alpha_c), n_iter)
    reject = next((int(k) for k in ks[::-1] if binom.cdf(k, n_iter, 0.5) <= alpha_c), 0)
    ax.axvspan(reject, accept, color=COR_NEUTRA, alpha=0.08, zorder=0)  # zona de tentativa
    ax.hlines(y, 0, d["_h"], color="#ccc", lw=1.2, zorder=1)
    ax.scatter(d["_h"], y, c=cores, s=70, edgecolor="white", linewidth=0.8, zorder=3)

    # anel nas que entraram por consenso apesar da rejeição do Boruta
    mot = d.get("motivo", pd.Series("", index=d.index)).fillna("")
    ring = mot.str.contains("Boruta rejeitou", case=False)
    if ring.any():
        ax.scatter(d.loc[ring, "_h"], y[ring.values], s=200, facecolors="none",
                   edgecolors="#222", linewidth=1.4, zorder=4, label="entrou apesar do Boruta")
    ax.set_yticks(y)
    ax.set_yticklabels([_short(f) for f in d["feature"]], fontsize=8)
    ax.axvline(n_iter * 0.5, color="#999", ls=":", lw=1.2, label="esperado sob H0")
    ax.axvline(accept, color=COR_PRIMARIA, ls="--", lw=1.4, label=f"confirma ≥ {accept}")
    ax.axvline(reject, color=COR_SECUNDARIA, ls="--", lw=1.4, label=f"rejeita ≤ {reject}")
    ax.set_xlabel(f"hits (de {n_iter} iterações)", fontsize=10)
    # nunca clipa pontos, mesmo se n_iter for menor que o máximo de hits observado
    ax.set_xlim(-0.5, max(int(n_iter), int(d["_h"].max())) + 0.5)
    ax.set_title("Boruta · hits por feature vs. bandas de decisão", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.8)
    _style_ax(ax)


def plot_boruta_significance(selection_table: pd.DataFrame, n_iter: int = 50, alpha: float = 0.05):
    """Lollipop dos hits do Boruta por feature, com bandas de decisão binomiais.

    Sob H0, hits ~ Binomial(n_iter, 0.5). Linhas de confirmação/rejeição por
    Bonferroni; zona de tentativa sombreada; cor do ponto = ``boruta_decisao``.
    Anel = feature que entrou pelo consenso apesar de o Boruta ter rejeitado.
    ``n_iter``/``alpha`` vêm por parâmetro (não estão na selection_table).
    """
    _n = len(selection_table) if selection_table is not None else 0
    fig = _figure((9, 0.42 * min(_n, 30) + 2))
    _draw_boruta(fig.add_subplot(111), selection_table, n_iter, alpha)
    return fig


# ── 4) Contribuição por book ─────────────────────────────────────────────
def _draw_book_power(ax, selection_table: pd.DataFrame, value="rf_importance"):
    if selection_table is None or selection_table.empty or "book" not in selection_table.columns:
        _empty(ax, "sem dados de seleção")
        return
    d = selection_table[selection_table["selecionada"].fillna(False)].copy()
    if d.empty:
        _empty(ax, "nenhuma feature selecionada")
        return
    col = value if _num(d, value).notna().any() else "score"
    d = d.assign(_v=_num(d, col).fillna(0.0))
    g = d.groupby("book").agg(poder=("_v", "sum"), n=("feature", "count"))
    tot_p, tot_n = g["poder"].sum(), g["n"].sum()
    g["poder_share"] = 100 * g["poder"] / tot_p if tot_p > 0 else 0.0
    g["n_share"] = 100 * g["n"] / tot_n if tot_n > 0 else 0.0
    g = g.sort_values("poder_share", ascending=True)

    y = np.arange(len(g)); h = 0.38
    ax.barh(y + h / 2, g["poder_share"], height=h, color=COR_PRIMARIA,
            edgecolor="white", label=f"% do poder (Σ {col})")
    ax.barh(y - h / 2, g["n_share"], height=h, color=COR_PRIMARIA, alpha=0.38,
            edgecolor="white", label="% das features")
    for i, (_, r) in enumerate(g.iterrows()):
        ax.text(r["poder_share"] + 0.8, i + h / 2, f"{r['poder_share']:.0f}%",
                va="center", fontsize=8, color="#333")
        ax.text(r["n_share"] + 0.8, i - h / 2, f"{int(r['n'])} feat",
                va="center", fontsize=8, color="#777")
    ax.set_yticks(y)
    ax.set_yticklabels(g.index, fontsize=9)
    ax.set_xlabel("% do total (features selecionadas)", fontsize=10)
    ax.set_xlim(0, max(float(g["poder_share"].max()), float(g["n_share"].max())) * 1.22)
    hhi = float(((g["poder_share"] / 100) ** 2).sum())
    ax.set_title(f"Contribuição por book · HHI de concentração = {hhi:.2f}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.8)
    _style_ax(ax)


def plot_book_power_contribution(selection_table: pd.DataFrame, value: str = "rf_importance"):
    """De onde vem o poder: share de importância e de features SELECIONADAS por book.

    Barra escura = % da importância total (Σ rf_importance, fallback ``score``);
    barra clara = % das features. HHI no título alerta concentração numa origem.
    """
    fig = _figure((9, 4.8))
    _draw_book_power(fig.add_subplot(111), selection_table, value)
    return fig


# ── 5) Redundância por cluster ───────────────────────────────────────────
def _redundancy_groups(d: pd.DataFrame, valcol: str, min_cluster_size: int, top_clusters: int):
    """Grupos de redundância como ``[(rótulo, sub-DataFrame)]``, ordenados por importância.

    Agrupa por ``(book, cluster)``: os ids de cluster **reiniciam em 1 em cada book**
    (a redundância do pipeline é intra-book), então agrupar só por ``cluster`` misturaria
    features de books diferentes que nunca foram comparadas. Mantém apenas grupos com
    ``>= min_cluster_size`` membros e ao menos um redundante (não-representante).
    """
    has_book = "book" in d.columns
    grouper = d.groupby(["book", "cluster"]) if has_book else d.groupby("cluster")
    grupos = []
    for key, g in grouper:
        if len(g) < min_cluster_size or g["representante"].fillna(True).all():
            continue
        if has_book:
            bk, cl = key
            rotulo = f"{bk}·{int(cl)}"
        else:
            rotulo = f"cl {int(key)}"
        grupos.append((rotulo, g))
    grupos.sort(key=lambda lg: np.nan_to_num(_num(lg[1], valcol).max(), nan=-np.inf), reverse=True)
    return grupos[:top_clusters]


def plot_cluster_redundancy(selection_table: pd.DataFrame, book=None,
                            min_cluster_size: int = 2, top_clusters: int = 12):
    """Clusters de redundância: representante mantido × redundantes descartados.

    Para cada cluster com ≥ ``min_cluster_size`` membros e ao menos um redundante,
    barras por feature (comprimento = ``score``/``rf_importance``): representante em
    azul (★), redundantes em cinza anotados ``→ <rep>``. 100% da tabela (sem matriz).
    """
    d = selection_table.copy() if selection_table is not None else pd.DataFrame()
    if book is not None and "book" in d.columns:
        d = d[d["book"] == book]
    fig = _figure((9, 5))
    ax = fig.add_subplot(111)
    if d.empty or "cluster" not in d.columns:
        _empty(ax, "sem informação de cluster")
        return fig
    d = d[_num(d, "cluster").notna()]
    valcol = "score" if _num(d, "score").notna().any() else "rf_importance"
    grupos = _redundancy_groups(d, valcol, min_cluster_size, top_clusters)
    if not grupos:
        _empty(ax, "nenhum cluster de redundância (features isoladas)")
        return fig

    linhas = []  # (feature, val, is_rep, redundante_com)
    for _rotulo, g in grupos:
        g = g.assign(_v=_num(g, valcol)).sort_values(["representante", "_v"], ascending=[False, False])
        for _, r in g.iterrows():
            linhas.append((r["feature"], r["_v"], bool(r.get("representante", True)),
                           r.get("redundante_com")))
    n = len(linhas)
    y = np.arange(n)[::-1]  # topo = 1º cluster
    vmax = max((v for _, v, _, _ in linhas if np.isfinite(v)), default=1.0)
    vmax = vmax if vmax > 0 else 1.0  # evita xlim degenerado se todos os scores == 0
    for i, (feat, val, is_rep, red_com) in enumerate(linhas):
        yi = y[i]
        w = val if np.isfinite(val) else 0.0
        ax.barh(yi, w, color=COR_PRIMARIA if is_rep else COR_NEUTRA,
                alpha=0.9 if is_rep else 0.55, edgecolor="white")
        if is_rep:
            ax.text(w, yi, "  ★ rep", va="center", fontsize=8, color=COR_PRIMARIA, fontweight="bold")
        else:
            ax.text(w, yi, f"  → {_short(red_com)}", va="center", fontsize=7, color=COR_SECUNDARIA)
    # separadores/sombras + rótulo (book·cluster) por grupo
    idx = 0
    for k, (rotulo, g) in enumerate(grupos):
        m = len(g)
        top_y = y[idx]; bot_y = y[idx + m - 1]
        if k % 2 == 0:
            ax.axhspan(bot_y - 0.5, top_y + 0.5, color="#f2f2f2", zorder=0)
        ax.text(-vmax * 0.02, (top_y + bot_y) / 2, rotulo, ha="right", va="center",
                fontsize=7, color="#999")
        idx += m
    ax.set_yticks(y)
    ax.set_yticklabels([_short(f) for f, *_ in linhas], fontsize=8)
    ax.set_xlabel(f"importância ({valcol})", fontsize=10)
    ax.set_xlim(-vmax * 0.12, vmax * 1.28)
    ax.set_title("Redundância por cluster · representante (★) × descartados",
                 fontsize=12, fontweight="bold")
    _style_ax(ax)
    return fig


# ── 6) Quadrante de poder IV × KS (classificação) ────────────────────────
def plot_power_quadrant_iv_ks(selection_table: pd.DataFrame, iv_min: float = 0.02,
                              ks_min: float = 0.10, iv_leakage: float = 0.50):
    """Quadrante IV × KS das features numéricas (só classificação), com faixas de IV.

    Faixas de Information Value (Siddiqi): <0.02 inútil · 0.02–0.1 fraco · 0.1–0.3
    médio · 0.3–0.5 forte · >0.5 zona de leakage (sombreada). Linha ``ks_min``
    horizontal. Azul = selecionada, cinza = descartada, estrela crimson = leakage.
    """
    fig = _figure((8.5, 6))
    ax = fig.add_subplot(111)
    _style_ax(ax)
    if selection_table is None or "iv" not in selection_table.columns or not _num(selection_table, "iv").notna().any():
        _empty(ax, "quadrante IV × KS disponível só em classificação")
        return fig
    d = selection_table.assign(_iv=_num(selection_table, "iv"), _ks=_num(selection_table, "ks"))
    d = d[np.isfinite(d["_iv"]) & np.isfinite(d["_ks"])]
    if d.empty:
        _empty(ax, "sem IV/KS para o quadrante")
        return fig
    xmax = max(float(d["_iv"].max()) * 1.18, iv_leakage * 1.1)
    ymax = max(float(d["_ks"].max()) * 1.18, ks_min * 1.5)
    ax.axvspan(iv_leakage, xmax, color=COR_SECUNDARIA, alpha=0.06)
    for b, lab in [(iv_min, "fraco"), (0.1, "médio"), (0.3, "forte"), (iv_leakage, "leakage")]:
        ax.axvline(b, color="#ccc", ls=":", lw=1)
        ax.text(b, ymax, f" {lab}", fontsize=7, color="#999", va="top", rotation=90)
    ax.axhline(ks_min, color="#ccc", ls=":", lw=1)
    ax.text(xmax, ks_min, f"ks_min={ks_min:g} ", fontsize=7, color="#999", ha="right", va="bottom")

    sel = d["selecionada"].fillna(False).astype(bool)
    leak = d["leakage_flag"].fillna(False).astype(bool) if "leakage_flag" in d else pd.Series(False, index=d.index)
    ax.scatter(d.loc[sel & ~leak, "_iv"], d.loc[sel & ~leak, "_ks"], s=90, c=COR_PRIMARIA,
               alpha=0.8, edgecolor="white", label="selecionada", zorder=3)
    ax.scatter(d.loc[~sel & ~leak, "_iv"], d.loc[~sel & ~leak, "_ks"], s=70, c=COR_NEUTRA,
               alpha=0.6, edgecolor="white", label="descartada", zorder=2)
    if leak.any():
        ax.scatter(d.loc[leak, "_iv"], d.loc[leak, "_ks"], s=240, marker="*", c=COR_SECUNDARIA,
                   edgecolor="white", label="leakage", zorder=4)
    for _, r in d.sort_values("_iv", ascending=False).head(9).iterrows():
        ax.annotate(_short(r["feature"]), (r["_iv"], r["_ks"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points", color="#333")
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    ax.set_xlabel("Information Value (IV)", fontsize=10)
    ax.set_ylabel("KS", fontsize=10)
    ax.set_title("Quadrante de poder preditivo · IV × KS", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.8)
    return fig


# ── 7) Auditoria de leakage ──────────────────────────────────────────────
def plot_leakage_audit(selection_table: pd.DataFrame, problem_type=None,
                       leakage_auc: float = 0.95, iv_leakage: float = 0.50, top_k: int = 15):
    """Watchlist de leakage: features ranqueadas pelo sinal, marcando as flagradas.

    Sinal = Gini (classif) → IV → |corr_target|. Barra crimson = ``leakage_flag``
    (barrada); azul = selecionada; cinza = descartada. Linha de referência no limiar
    de leakage. Serve para revisar de perto quem tem sinal implausivelmente alto.
    """
    _n = len(selection_table) if selection_table is not None else 0
    fig = _figure((9, 0.45 * min(_n, top_k) + 2))
    ax = fig.add_subplot(111)
    if selection_table is None or selection_table.empty:
        _empty(ax, "sem dados de seleção")
        return fig
    d = selection_table.copy()
    if problem_type != "regression" and "gini" in d.columns and _num(d, "gini").notna().any():
        sig, thr, lab = _num(d, "gini").abs(), 2 * leakage_auc - 1, "Gini univariado"
    elif problem_type != "regression" and "iv" in d.columns and _num(d, "iv").notna().any():
        sig, thr, lab = _num(d, "iv"), iv_leakage, "IV"
    else:
        sig, thr, lab = _num(d, "corr_target").abs(), None, "|correlação com o alvo|"
    d = d.assign(_s=sig)
    d = d[np.isfinite(d["_s"])].sort_values("_s", ascending=False).head(top_k)
    if d.empty:
        _empty(ax, "sem sinal univariado para auditar")
        return fig
    d = d.iloc[::-1]  # maior no topo
    leak = d["leakage_flag"].fillna(False).astype(bool) if "leakage_flag" in d else pd.Series(False, index=d.index)
    sel = d["selecionada"].fillna(False).astype(bool)
    cores = [COR_SECUNDARIA if lk else (COR_PRIMARIA if s else COR_NEUTRA)
             for lk, s in zip(leak, sel)]
    y = np.arange(len(d))
    ax.barh(y, d["_s"], color=cores, edgecolor="white")
    for i, (_, r) in enumerate(d.iterrows()):
        tag = "  ⚠ barrada" if (bool(r.get("leakage_flag")) and not bool(r.get("selecionada"))) else ""
        book = f" [{r.get('book')}]" if "book" in d.columns else ""
        ax.text(r["_s"], i, f"{book}{tag}", va="center", fontsize=7, color="#555")
    if thr is not None:
        ax.axvline(thr, color=COR_SECUNDARIA, ls="--", lw=1.3, label=f"limiar de leakage ({thr:.2f})")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(f) for f in d["feature"]], fontsize=8)
    ax.set_xlabel(lab, fontsize=10)
    n_leak = int((selection_table["leakage_flag"].fillna(False)).sum()) if "leakage_flag" in selection_table else 0
    ax.set_title(f"Auditoria de leakage · {n_leak} feature(s) flagrada(s)", fontsize=12, fontweight="bold")
    _style_ax(ax)
    return fig


# ── 8) Scorecard de qualidade dos sobreviventes ──────────────────────────
def plot_survivor_scorecard(selection_table: pd.DataFrame, top_k: int = 20):
    """Heatmap-tabela de qualidade das features SELECIONADAS (auditoria de uma olhada).

    Linhas = sobreviventes (por ``score``), colunas = métricas normalizadas por
    coluna para [0,1] com 1 = melhor (missing e dominância invertidos). Azul = bom,
    crimson = ruim; valores crus anotados. Cinza = métrica ausente (ex.: não-numérica).
    """
    _n = len(selection_table) if selection_table is not None else 0
    fig = _figure((9, 0.5 * min(_n, top_k) + 2.5))
    ax = fig.add_subplot(111)
    if selection_table is None or selection_table.empty:
        _empty(ax, "sem dados de seleção")
        return fig
    d = selection_table[selection_table["selecionada"].fillna(False)].copy()
    if d.empty:
        _empty(ax, "nenhuma feature selecionada")
        return fig
    if "score" in d.columns:
        d = d.sort_values("score", ascending=False)
    d = d.head(top_k)
    poder_col = "iv" if ("iv" in d.columns and _num(d, "iv").notna().any()) else "corr_target"
    especs = [
        ("missing", _num(d, "pct_missing"), True),
        ("dominância", _num(d, "top1_share"), True),
        (poder_col, _num(d, poder_col).abs(), False),
        ("rf_imp", _num(d, "rf_importance"), False),
        ("boruta", _num(d, "boruta_hits"), False),
        ("consenso", _num(d, "score_consenso"), False),
    ]
    raw = np.full((len(d), len(especs)), np.nan)
    qual = np.full((len(d), len(especs)), np.nan)
    for j, (_, serie, invert) in enumerate(especs):
        v = serie.to_numpy(dtype=float)
        raw[:, j] = v
        if not np.isfinite(v).any():
            continue
        lo, hi = np.nanmin(v), np.nanmax(v)
        norm = (v - lo) / (hi - lo) if hi > lo else np.where(np.isfinite(v), 0.5, np.nan)
        qual[:, j] = (1 - norm) if invert else norm
    cmap = colormap_divergente().reversed()  # crimson(ruim) → branco → steelblue(bom)
    cmap.set_bad("#eeeeee")
    ax.imshow(np.ma.masked_invalid(qual), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(especs)))
    ax.set_xticklabels([e[0] for e in especs], fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([_short(f) for f in d["feature"]], fontsize=8)
    for i in range(len(d)):
        for j in range(len(especs)):
            v, q = raw[i, j], qual[i, j]
            if np.isfinite(v):
                txt = f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if (np.isfinite(q) and (q < 0.22 or q > 0.85)) else "#333")
    ax.set_title("Scorecard de qualidade dos sobreviventes (azul = bom)",
                 fontsize=12, fontweight="bold")
    return fig


# ── 9) Correlação dos sobreviventes (cross-book) ─────────────────────────
def plot_survivor_corr_heatmap(matrix: pd.DataFrame, selection_table: pd.DataFrame = None,
                               corr_high: float = 0.80,
                               title: str = "Redundância entre sobreviventes (cross-book)"):
    """Heatmap Spearman das features SELECIONADAS, destacando redundância residual.

    Contorna em crimson pares com |corr| ≥ ``corr_high`` — a deduplicação do
    pipeline é POR BOOK, então redundância cross-book só aparece aqui. Se
    ``selection_table`` vier, marca o representante (★) e o book no rótulo.
    """
    from matplotlib.patches import Rectangle

    n = len(matrix) if matrix is not None else 0
    fig = _figure((1.6 + 0.6 * n, 1.4 + 0.6 * n))
    ax = fig.add_subplot(111)
    if matrix is None or matrix.empty or n < 2:
        _empty(ax, "poucos sobreviventes numéricos p/ correlação")
        return fig
    vals = matrix.values
    im = ax.imshow(vals, cmap=colormap_divergente(), vmin=-1, vmax=1)

    def _lab(f):
        s = _short(f)
        if selection_table is not None and "feature" in selection_table.columns:
            row = selection_table[selection_table["feature"] == f]
            if not row.empty:
                bk = row["book"].iloc[0] if "book" in row else None
                rep = bool(row["representante"].iloc[0]) if "representante" in row else False
                s = f"{'★ ' if rep else ''}{s}" + (f"  [{bk}]" if bk is not None else "")
        return s

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([_short(c) for c in matrix.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([_lab(c) for c in matrix.index], fontsize=8)
    for i in range(n):
        for j in range(n):
            v = vals[i, j]
            if np.isfinite(v) and n <= 16:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else "#333")
            if i < j and np.isfinite(v) and abs(v) >= corr_high:
                for (a, b) in ((j, i), (i, j)):
                    ax.add_patch(Rectangle((a - 0.5, b - 0.5), 1, 1, fill=False,
                                           edgecolor=COR_SECUNDARIA, lw=2))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11, fontweight="bold")
    return fig


# ── 10) Dashboard pós-seleção (2×2) ──────────────────────────────────────
def plot_post_selection_dashboard(selection_table: pd.DataFrame, problem_type=None,
                                  n_iter: int = 50, title: str = "Painel pós-seleção — Yggdrasil"):
    """Painel 2×2 que responde, numa tela, as 4 perguntas pós-seleção:
    (1) onde perdi features (funil) · (2) de onde vem o poder e o que revisar (mapa)
    · (3) robustez estatística (Boruta) · (4) dependência de origem (books)."""
    import matplotlib.gridspec as gridspec

    fig = _figure((17, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.22)
    _draw_funnel(fig.add_subplot(gs[0, 0]), selection_table)
    _draw_decision_map(fig.add_subplot(gs[0, 1]), selection_table, problem_type)
    _draw_boruta(fig.add_subplot(gs[1, 0]), selection_table, n_iter)
    _draw_book_power(fig.add_subplot(gs[1, 1]), selection_table)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    return fig


__all__ = [
    # gráficos originais
    "plot_book_selection", "plot_book_overview", "plot_overall_importance", "plot_corr_heatmap",
    # análises pós-seleção
    "plot_selection_funnel", "plot_decision_map", "plot_boruta_significance",
    "plot_book_power_contribution", "plot_cluster_redundancy", "plot_power_quadrant_iv_ks",
    "plot_leakage_audit", "plot_survivor_scorecard", "plot_survivor_corr_heatmap",
    "plot_post_selection_dashboard",
]
