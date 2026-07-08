"""Dashboard visual da esteira (generalizado do protótipo).

Painel com: cards de métrica (adaptados ao tipo de problema), uma seção de
interpretabilidade SHAP (importância por feature + beeswarm) e, para cada
metodologia de rating, uma linha com quatro gráficos — média do target e
volumetria por grupo, **distribuição dos ratings ao longo das safras**, série
temporal do target e dispersão.

A figura é criada pela API orientada a objetos (``matplotlib.figure.Figure``),
**fora** do gerenciador global do pyplot. Isso evita o problema clássico de
*duplicação* no Jupyter (backend inline auto-exibe figuras abertas e, ao ecoar
``fig``, o gráfico apareceria duas vezes).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from ..data import analysis_samples_present
from ..interpretability import compute_shap, shap_feature_importance
from .style import COR_PRIMARIA, COR_SECUNDARIA, colormap, fmt_month_year, gradient

# Cards exibidos por tipo de problema: (rótulo, chave da métrica, formato).
_CARDS = {
    "regression": [
        ("RMSE", "rmse", "{:.4f}"),
        ("MAE", "mae", "{:.4f}"),
        ("R²", "r2", "{:.4f}"),
        ("MAPE", "mape", "{:.1f}%"),
    ],
    "classification": [
        ("KS", "ks", "{:.4f}"),
        ("AUC", "auc", "{:.4f}"),
        ("F1", "f1", "{:.4f}"),
        ("Acurácia", "accuracy", "{:.4f}"),
    ],
}

# Tema padrão (ver yggdrasil.reporting.style): steelblue + crimson.
_COR_BARRA = COR_PRIMARIA       # barras / métrica principal
_COR_LINHA = COR_SECUNDARIA     # linhas de série / medianas
_COR_VOL = COR_SECUNDARIA       # linha de volumetria / acento
# Cards de métrica: cinza claro neutro (sem alternância de cor).
_COR_CARD_FILL = "#ECECEC"
_COR_CARD_EDGE = "#CCCCCC"
_COR_CARD_LABEL = "#666666"


def _style_ax(ax) -> None:
    """Grade discreta por eixo (substitui o antigo ``sns.set_theme`` global,
    que mutava os rcParams do notebook inteiro do usuário)."""
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, linewidth=0.8)


def _draw_cards(fig, gs, problem_type: str, metrics: Dict[str, float]) -> None:
    from matplotlib.patches import Rectangle

    for j, (nome, chave, fmt) in enumerate(_CARDS[problem_type]):
        ax = fig.add_subplot(gs[0, j])
        ax.axis("off")
        val = metrics.get(chave, float("nan"))
        ax.add_patch(Rectangle((0.04, 0.08), 0.92, 0.84, facecolor=_COR_CARD_FILL,
                               edgecolor=_COR_CARD_EDGE, linewidth=1.5,
                               transform=ax.transAxes))
        ax.text(0.5, 0.66, nome, ha="center", va="center", fontsize=16,
                fontweight="bold", color=_COR_CARD_LABEL, transform=ax.transAxes)
        txt = fmt.format(val) if np.isfinite(val) else "—"
        ax.text(0.5, 0.32, txt, ha="center", va="center", fontsize=27,
                fontweight="bold", color="#2c2c2a", transform=ax.transAxes)


def _shap_importance_bar(ax, importance: pd.DataFrame, max_display: int) -> None:
    _style_ax(ax)
    d = importance.head(max_display).iloc[::-1]  # mais importante no topo
    ax.barh(range(len(d)), d["mean_abs_shap"], color=_COR_BARRA, alpha=0.85,
            edgecolor="white")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["feature"])
    ax.set_xlabel("Importância média |SHAP|", fontsize=11)
    ax.set_title("SHAP · Importância por feature", fontsize=12, fontweight="bold")


def _shap_beeswarm(fig, ax, shap_values: np.ndarray, X: pd.DataFrame,
                   max_display: int) -> None:
    _style_ax(ax)
    feature_names = list(X.columns)
    imp = np.abs(shap_values).mean(axis=0)
    ordem = np.argsort(imp)[::-1][:max_display][::-1]  # mais importante no topo
    cmap = colormap()
    rng = np.random.default_rng(0)
    sc = None
    for yi, fidx in enumerate(ordem):
        sv = shap_values[:, fidx]
        fv = np.asarray(X.iloc[:, fidx], dtype=float)
        lo, hi = np.nanpercentile(fv, 5), np.nanpercentile(fv, 95)
        cval = np.clip((fv - lo) / (hi - lo + 1e-12), 0, 1)
        jitter = (rng.random(len(sv)) - 0.5) * 0.6
        sc = ax.scatter(sv, yi + jitter, c=cval, cmap=cmap, s=8, alpha=0.55,
                        linewidths=0)
    ax.axvline(0, color="#999999", lw=1)
    ax.set_yticks(range(len(ordem)))
    ax.set_yticklabels([feature_names[i] for i in ordem])
    ax.set_xlabel("Valor SHAP (impacto na predição)", fontsize=11)
    ax.set_title("SHAP · Beeswarm (cor = valor da feature)", fontsize=12, fontweight="bold")
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["baixo", "alto"])
        cbar.set_label("valor da feature", fontsize=9)


def _safra_distribution(ax, df, rating_col, cfg, ratings, cores) -> None:
    _style_ax(ax)
    comp = (df.groupby(["_mes", rating_col], observed=True).size()
              .unstack(rating_col).reindex(columns=ratings).fillna(0))
    comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
    ax.stackplot(comp_pct.index, [comp_pct[r].values for r in ratings],
                 labels=[str(r) for r in ratings], colors=cores, alpha=0.9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% da safra", fontsize=10)
    ax.set_xlabel("Mês de referência", fontsize=10)
    if len(ratings) <= 6:  # legenda só quando cabe sem poluir
        ax.legend(title="Rating", fontsize=6, ncol=len(ratings), loc="lower center",
                  framealpha=0.6)
    ax.tick_params(axis="x", rotation=45, labelsize=8)


def _vintage_chart(ax, df, cfg, y_label) -> None:
    """Carteira: volumetria por safra em BARRAS + alvo médio por safra em LINHA."""
    _style_ax(ax)
    g = (df.groupby("_mes", observed=True)
           .agg(vol=(cfg.target_col, "size"), alvo=(cfg.target_col, "mean")))
    x = list(range(len(g)))
    ax.bar(x, g["vol"].values, color=_COR_BARRA, alpha=0.55, edgecolor="white", width=0.7)
    ax.set_ylabel("Volumetria (nº)", fontsize=11, color=_COR_BARRA, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(fmt_month_year(g.index), rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Mês de referência", fontsize=11)
    ax.tick_params(axis="y", labelcolor=_COR_BARRA)
    axb = ax.twinx()
    axb.plot(x, g["alvo"].values, color=_COR_LINHA, marker="o", markersize=6, linewidth=2.5)
    axb.set_ylabel(y_label, fontsize=11, color=_COR_LINHA, fontweight="bold")
    axb.tick_params(axis="y", labelcolor=_COR_LINHA)
    axb.grid(False)
    ax.set_title(f"Volumetria e {y_label.lower()} por safra (carteira)",
                 fontsize=13, fontweight="bold")


def build_dashboard(
    df: pd.DataFrame,
    rating_cols: Sequence[str],
    cfg: ColumnConfig,
    problem_type: str = "regression",
    metrics: Optional[Dict[str, float]] = None,
    title: str = "Dashboard de Modelo — Yggdrasil",
    eval_sample: Optional[str] = None,
    model=None,
    X_shap: Optional[pd.DataFrame] = None,
    shap_values: Optional[np.ndarray] = None,
    shap_max_display: int = 15,
):
    """Constrói e retorna a ``Figure`` do dashboard.

    Passe ``model`` + ``X_shap`` (ou ``shap_values`` + ``X_shap``) para incluir a
    seção de interpretabilidade SHAP.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    class _DashFigure(Figure):
        """Figura que sempre se renderiza como PNG no Jupyter.

        Garante a exibição mesmo quando o notebook ainda não importou o pyplot
        (sem isso, uma figura OO ecoaria como texto). Como não é registrada no
        ``pyplot.Gcf``, também não duplica com o auto-display do backend inline.
        """

        def _repr_png_(self):
            from io import BytesIO

            buf = BytesIO()
            self.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            return buf.getvalue()

    eval_sample = eval_sample or cfg.oot_sample
    metrics = metrics or {}
    rating_cols = list(rating_cols)
    samples = analysis_samples_present(df, cfg)
    tg = cfg.target_col

    df = df.copy()
    df["_mes"] = pd.to_datetime(df[cfg.date_col]).dt.to_period("M").dt.to_timestamp()
    y_label = "Taxa de evento" if problem_type == "classification" else "Target médio"

    # ── SHAP (opcional): calcula se model+X (ou shap_values+X) vierem ─────
    Xs = None
    if X_shap is not None and shap_values is None and model is not None:
        try:
            shap_values, Xs = compute_shap(model, X_shap, problem_type, sample_size=1000)
        except Exception:  # SHAP é best-effort; sem ele só não desenha a seção
            shap_values = None
    elif X_shap is not None and shap_values is not None:
        Xs = X_shap
    has_shap = shap_values is not None and Xs is not None

    # ── layout ───────────────────────────────────────────────────────────
    shap_rows = 1 if has_shap else 0
    n_rows = 2 + shap_rows + len(rating_cols)   # cards + vintage (+ shap) + ratings
    altura = 2.6 + 3.0 + (4.2 if has_shap else 0) + 4.0 * len(rating_cols)
    fig = _DashFigure(figsize=(22, altura))
    FigureCanvasAgg(fig)  # canvas Agg => savefig e _repr_png_ funcionam
    height_ratios = [0.5, 0.85] + ([1.15] if has_shap else []) + [1] * len(rating_cols)
    # Cabeçalho com folgas em polegadas (constantes): título, subtítulo e cards
    # ficam separados o suficiente para não se sobreporem em nenhuma altura.
    top = 1 - 1.55 / altura
    # wspace alto evita o rótulo do eixo gêmeo (target/volumetria) colidir com o vizinho.
    gs = gridspec.GridSpec(n_rows, 4, figure=fig, height_ratios=height_ratios,
                           hspace=0.5, wspace=0.46, top=top, bottom=0.40 / altura)

    fig.suptitle(title, fontsize=22, fontweight="bold", y=1 - 0.38 / altura)
    fig.text(0.5, 1 - 1.18 / altura, f"Métricas avaliadas na amostra {eval_sample}",
             ha="center", fontsize=11, style="italic", color="#666")

    # ── Linha 0 — cards de métrica ───────────────────────────────────────
    _draw_cards(fig, gs, problem_type, metrics)

    # ── Linha 1 — vintage de carteira (volumetria barras + alvo linha) ───
    _vintage_chart(fig.add_subplot(gs[1, 0:4]), df, cfg, y_label)

    # ── Linha SHAP (interpretabilidade global do modelo) ─────────────────
    if has_shap:
        importance = shap_feature_importance(shap_values, Xs.columns)
        ax_imp = fig.add_subplot(gs[2, 0:2])
        _shap_importance_bar(ax_imp, importance, shap_max_display)
        ax_bee = fig.add_subplot(gs[2, 2:4])
        _shap_beeswarm(fig, ax_bee, shap_values, Xs, shap_max_display)

    # ── Uma linha por metodologia de rating ──────────────────────────────
    base = 2 + shap_rows
    for offset, rating_col in enumerate(rating_cols):
        row = base + offset
        ratings = sorted(df[rating_col].dropna().unique())
        titulo = rating_col.replace("rating_", "Rating ").title()
        cores = gradient(len(ratings))  # azul (menor risco) -> vermelho (maior)

        agg = (df.groupby(rating_col, observed=True)
                 .agg(target_medio=(tg, "mean"), volume=(tg, "size"))
                 .reindex(ratings))
        agg["pct_vol"] = 100 * agg["volume"] / agg["volume"].sum()

        # G1 — barras de VOLUMETRIA + linha do alvo médio, por rating (invertido)
        ax1 = fig.add_subplot(gs[row, 0])
        _style_ax(ax1)
        ax1.bar(range(len(ratings)), agg["pct_vol"], color=_COR_BARRA, alpha=0.6,
                edgecolor="white", linewidth=1.0)
        ax1.set_xticks(range(len(ratings)))
        ax1.set_xticklabels(ratings, rotation=0, fontsize=8)
        ax1.set_ylabel("% volumetria", fontsize=10, color=_COR_BARRA, fontweight="bold")
        ax1.set_xlabel("Rating", fontsize=10)
        ax1.tick_params(axis="y", labelcolor=_COR_BARRA, labelsize=8)
        ax1.set_title(f"{titulo} · volume e {y_label.lower()}", fontsize=11, fontweight="bold")
        ax1b = ax1.twinx()
        ax1b.plot(range(len(ratings)), agg["target_medio"], color=_COR_LINHA, marker="o",
                  markersize=6, linewidth=2.2)
        ax1b.set_ylabel(y_label, fontsize=10, color=_COR_LINHA, fontweight="bold", labelpad=1)
        ax1b.tick_params(axis="y", labelcolor=_COR_LINHA, labelsize=8)
        ax1b.grid(False)

        # G2 — distribuição dos ratings ao longo das safras (composição %)
        ax2 = fig.add_subplot(gs[row, 1])
        _safra_distribution(ax2, df, rating_col, cfg, ratings, cores)
        ax2.set_title(f"{titulo} · distribuição por safra", fontsize=11, fontweight="bold")

        # G3 — série temporal do alvo médio por rating
        ax3 = fig.add_subplot(gs[row, 2])
        _style_ax(ax3)
        serie = (df.groupby(["_mes", rating_col], observed=True)[tg].mean()
                   .unstack(rating_col).reindex(columns=ratings))
        for k, rt in enumerate(ratings):
            ax3.plot(serie.index, serie[rt], marker="o", markersize=3.5, linewidth=1.8,
                     color=cores[k], label=str(rt))
        ax3.set_title(f"{titulo} · {y_label.lower()} no tempo", fontsize=11, fontweight="bold")
        ax3.set_ylabel(y_label, fontsize=10)
        ax3.set_xlabel("Mês de referência", fontsize=10)
        if len(ratings) <= 6:
            ax3.legend(title="Rating", fontsize=6, ncol=2, loc="best", framealpha=0.6)
        ax3.tick_params(axis="x", rotation=45, labelsize=8)

        # G4 — dispersão do alvo por rating (boxplot)
        ax4 = fig.add_subplot(gs[row, 3])
        _style_ax(ax4)
        dados = [df.loc[df[rating_col] == rt, tg].dropna().values for rt in ratings]
        rotulos = [str(r) for r in ratings]
        try:  # matplotlib >= 3.9 renomeou 'labels' para 'tick_labels'
            bp = ax4.boxplot(dados, tick_labels=rotulos, showfliers=False, patch_artist=True)
        except TypeError:
            bp = ax4.boxplot(dados, labels=rotulos, showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(_COR_BARRA)
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color(_COR_LINHA)
            med.set_linewidth(1.5)
        ax4.set_title(f"{titulo} · dispersão do alvo", fontsize=11, fontweight="bold")
        ax4.set_ylabel("Alvo observado", fontsize=10)
        ax4.set_xlabel("Rating", fontsize=10)
        ax4.tick_params(axis="x", rotation=0, labelsize=8)

    return fig


def save_dashboard(
    df: pd.DataFrame,
    rating_cols: Sequence[str],
    cfg: ColumnConfig,
    problem_type: str,
    path: str,
    metrics: Optional[Dict[str, float]] = None,
    title: str = "Dashboard de Modelo — Yggdrasil",
    model=None,
    X_shap: Optional[pd.DataFrame] = None,
) -> str:
    """Constrói o dashboard e salva em ``path`` (PNG). Retorna o caminho."""
    fig = build_dashboard(df, rating_cols, cfg, problem_type, metrics=metrics,
                          title=title, model=model, X_shap=X_shap)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return path
