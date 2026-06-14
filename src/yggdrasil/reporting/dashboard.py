"""Dashboard visual da esteira (generalizado do protótipo).

Monta um painel com cards de métrica (adaptados ao tipo de problema) e, para
cada metodologia de rating, uma linha com quatro gráficos: média do target e
volumetria por grupo, distribuição por amostra, série temporal e dispersão.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from ..data import analysis_samples_present

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

_COR_BARRA, _COR_LINHA, _COR_VOL = "#4C72B0", "#C44E52", "#55A868"
_CARD_CORES = [_COR_BARRA, _COR_VOL, "#8172B3", _COR_LINHA]


def build_dashboard(
    df: pd.DataFrame,
    rating_cols: Sequence[str],
    cfg: ColumnConfig,
    problem_type: str = "regression",
    metrics: Optional[Dict[str, float]] = None,
    title: str = "Dashboard de Modelo — Yggdrasil",
    eval_sample: Optional[str] = None,
):
    """Constrói e retorna a ``Figure`` do dashboard."""
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="notebook")
        palette = sns.color_palette("viridis", 12)
    except Exception:  # seaborn é opcional para o estilo
        sns = None
        palette = plt.cm.viridis(np.linspace(0, 1, 12))

    eval_sample = eval_sample or cfg.oot_sample
    metrics = metrics or {}
    rating_cols = list(rating_cols)
    samples = analysis_samples_present(df, cfg)
    tg, sc = cfg.target_col, cfg.score_col

    df = df.copy()
    df["_mes"] = pd.to_datetime(df[cfg.date_col]).dt.to_period("M").dt.to_timestamp()
    y_label = "Taxa de evento" if problem_type == "classification" else "Target médio"

    n_rows = 1 + len(rating_cols)
    fig = plt.figure(figsize=(22, 3.5 + 4.0 * len(rating_cols)))
    gs = gridspec.GridSpec(
        n_rows, 4, figure=fig,
        height_ratios=[0.45] + [1] * len(rating_cols), hspace=0.55, wspace=0.30,
    )
    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.995)
    fig.text(0.5, 0.95, f"Métricas avaliadas na amostra {eval_sample}",
             ha="center", fontsize=12, style="italic", color="#666")

    # ── Linha 0 — cards de métrica ───────────────────────────────────────
    for j, (nome, chave, fmt) in enumerate(_CARDS[problem_type]):
        ax = fig.add_subplot(gs[0, j])
        ax.axis("off")
        cor = _CARD_CORES[j % len(_CARD_CORES)]
        val = metrics.get(chave, float("nan"))
        ax.add_patch(plt.Rectangle((0.04, 0.05), 0.92, 0.85, facecolor=cor, alpha=0.12,
                                   edgecolor=cor, linewidth=2, transform=ax.transAxes))
        ax.text(0.5, 0.64, nome, ha="center", va="center", fontsize=17,
                fontweight="bold", color=cor, transform=ax.transAxes)
        txt = fmt.format(val) if np.isfinite(val) else "—"
        ax.text(0.5, 0.30, txt, ha="center", va="center", fontsize=28,
                fontweight="bold", color="#2c2c2a", transform=ax.transAxes)

    # ── Uma linha por metodologia de rating ──────────────────────────────
    for row, rating_col in enumerate(rating_cols, start=1):
        ratings = sorted(df[rating_col].dropna().unique())
        titulo = rating_col.replace("rating_", "Rating ").title()

        agg = (df.groupby(rating_col, observed=True)
                 .agg(target_medio=(tg, "mean"), volume=(tg, "size"))
                 .reindex(ratings))
        agg["pct_vol"] = 100 * agg["volume"] / agg["volume"].sum()

        # G1 — barras target médio + linha de volumetria
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.bar(range(len(ratings)), agg["target_medio"], color=_COR_BARRA, alpha=0.85,
                edgecolor="white", linewidth=1.2)
        ax1.set_xticks(range(len(ratings)))
        ax1.set_xticklabels(ratings, rotation=0)
        ax1.set_ylabel(y_label, fontsize=11, color=_COR_BARRA, fontweight="bold")
        ax1.set_xlabel("Rating", fontsize=11)
        ax1.set_title(f"{titulo} · {y_label} e volumetria", fontsize=12, fontweight="bold")
        ax1b = ax1.twinx()
        ax1b.plot(range(len(ratings)), agg["pct_vol"], color=_COR_VOL, marker="o",
                  markersize=7, linewidth=2.5)
        ax1b.set_ylabel("% volumetria", fontsize=11, color=_COR_VOL, fontweight="bold")
        ax1b.grid(False)

        # G2 — distribuição % por amostra
        ax2 = fig.add_subplot(gs[row, 1])
        vol = (df.groupby([rating_col, cfg.sample_col], observed=True).size()
                 .unstack(cfg.sample_col).reindex(ratings).reindex(columns=samples))
        (vol.div(vol.sum(axis=0), axis=1) * 100).plot(
            kind="bar", ax=ax2, width=0.78, edgecolor="white")
        ax2.set_title(f"{titulo} · Distribuição (%) por amostra", fontsize=12, fontweight="bold")
        ax2.set_ylabel("% dentro da amostra", fontsize=11)
        ax2.set_xlabel("Rating", fontsize=11)
        ax2.legend(title="Amostra", fontsize=9)
        ax2.tick_params(axis="x", rotation=0)

        # G3 — série temporal do target médio por rating
        ax3 = fig.add_subplot(gs[row, 2])
        serie = (df.groupby(["_mes", rating_col], observed=True)[tg].mean()
                   .unstack(rating_col).reindex(columns=ratings))
        for k, rt in enumerate(ratings):
            ax3.plot(serie.index, serie[rt], marker="o", markersize=3.5, linewidth=1.8,
                     color=palette[k % len(palette)], label=str(rt))
        ax3.set_title(f"{titulo} · {y_label} no tempo", fontsize=12, fontweight="bold")
        ax3.set_ylabel(y_label, fontsize=11)
        ax3.set_xlabel("Mês de referência", fontsize=11)
        ax3.legend(title="Rating", fontsize=8, ncol=2, loc="best")
        ax3.tick_params(axis="x", rotation=45)

        # G4 — dispersão do target por rating (boxplot)
        ax4 = fig.add_subplot(gs[row, 3])
        dados = [df.loc[df[rating_col] == rt, tg].dropna().values for rt in ratings]
        rotulos = [str(r) for r in ratings]
        try:  # matplotlib >= 3.9 renomeou 'labels' para 'tick_labels'
            ax4.boxplot(dados, tick_labels=rotulos, showfliers=False)
        except TypeError:
            ax4.boxplot(dados, labels=rotulos, showfliers=False)
        ax4.set_title(f"{titulo} · Dispersão do target", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Target observado", fontsize=11)
        ax4.set_xlabel("Rating", fontsize=11)
        ax4.tick_params(axis="x", rotation=0)

    return fig


def save_dashboard(
    df: pd.DataFrame,
    rating_cols: Sequence[str],
    cfg: ColumnConfig,
    problem_type: str,
    path: str,
    metrics: Optional[Dict[str, float]] = None,
    title: str = "Dashboard de Modelo — Yggdrasil",
) -> str:
    """Constrói o dashboard e salva em ``path`` (PNG). Retorna o caminho."""
    import matplotlib.pyplot as plt

    fig = build_dashboard(df, rating_cols, cfg, problem_type, metrics=metrics, title=title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
