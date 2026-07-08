"""
Relatório e gráficos padronizados (Guia §5, §6.1 (iv), §6.2 ``reporting``)
=========================================================================
O guia trata o **relatório como cidadão de primeira classe** (§6.1): a saída de um
estudo inclui as tabelas de teste e os gráficos padronizados prontos para a
governança. Este módulo desenha, no estilo visual do ``yggdrasil`` (paleta
*steelblue* + *crimson* de :mod:`yggdrasil.reporting.style`):

* **Ajuste** (:func:`plot_fit`): série observada × ajustada (escala original) e os
  resíduos — o cartão de sanidade do modelo.
* **Diagnóstico de resíduos** (:func:`plot_residual_diagnostics`): resíduo no
  tempo, ACF, histograma e QQ-plot (a leitura visual da bateria §4.2).
* **Projeção em leque** (:func:`plot_projection`): a **Figura 1** do guia — a série
  histórica e a projeção condicional aos cenários, com o leque de incerteza.
* **Relatório de modelo** (:func:`model_report`): um HTML com a especificação, a
  tabela de coeficientes, as métricas e a bateria de diagnóstico — o documento de
  governança.

``matplotlib`` é importado **tardiamente**, dentro de cada função (o pacote só o
usa em relatório/tracking, não no cálculo).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # apenas type hints
    from matplotlib.figure import Figure

    from .base import FitResult, Projection
    from .series import RiskSeries


# ======================================================================
# Paleta (importada tardiamente de reporting.style, com fallback)
# ======================================================================
def _palette() -> dict:
    cores = {"primaria": "#4682b4", "secundaria": "#dc143c", "neutra": "#888888"}
    try:
        from ...reporting.style import COR_NEUTRA, COR_PRIMARIA, COR_SECUNDARIA

        cores.update(primaria=COR_PRIMARIA, secundaria=COR_SECUNDARIA, neutra=COR_NEUTRA)
    except Exception:  # pragma: no cover
        pass
    return cores


def _to_ts(index):
    return index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index


# ======================================================================
# Ajuste
# ======================================================================
def plot_fit(fit: "FitResult", series=None) -> "Figure":
    """Observado × ajustado (escala original) e resíduos ao longo do tempo."""
    import matplotlib.pyplot as plt

    c = _palette()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1], sharex=True)
    idx = _to_ts(fit.fitted.index)
    if series is not None:
        obs = series.values if hasattr(series, "values") and hasattr(series, "kind") else series
        ax1.plot(_to_ts(obs.index), obs.to_numpy(dtype=float), color=c["neutra"],
                 lw=1.4, label="observado", zorder=1)
    ax1.plot(idx, fit.fitted.to_numpy(dtype=float), color=c["primaria"], lw=2.0,
             label="ajustado", zorder=2)
    ax1.set_title(f"Ajuste — {fit.model_name} ({fit.kind.upper()}, link {fit.link})")
    ax1.set_ylabel(fit.kind.upper())
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.axhline(0, color=c["neutra"], lw=0.8)
    ax2.plot(idx, fit.resid.to_numpy(dtype=float), color=c["secundaria"], lw=1.0)
    ax2.set_ylabel("resíduo\n(link)")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ======================================================================
# Diagnóstico de resíduos
# ======================================================================
def plot_residual_diagnostics(fit: "FitResult") -> "Figure":
    """Painel 2×2: resíduo no tempo, ACF, histograma e QQ-plot (Guia §4.2)."""
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf
    from scipy import stats

    c = _palette()
    r = fit.resid.to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    axes[0, 0].axhline(0, color=c["neutra"], lw=0.8)
    axes[0, 0].plot(_to_ts(fit.resid.index), r, color=c["secundaria"], lw=1.0)
    axes[0, 0].set_title("Resíduo no tempo")
    axes[0, 0].grid(alpha=0.25)

    lags = min(20, len(r) // 2 - 1)
    plot_acf(r, ax=axes[0, 1], lags=max(1, lags), color=c["primaria"])
    axes[0, 1].set_title("ACF do resíduo (Ljung-Box §4.2)")

    axes[1, 0].hist(r, bins=min(25, max(6, len(r) // 5)), color=c["primaria"],
                    alpha=0.75, edgecolor="white")
    axes[1, 0].set_title("Histograma (Jarque-Bera §4.2)")
    axes[1, 0].grid(alpha=0.25)

    stats.probplot(r, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("QQ-plot normal")
    axes[1, 1].get_lines()[0].set_color(c["primaria"])
    axes[1, 1].get_lines()[1].set_color(c["secundaria"])
    fig.suptitle(f"Diagnóstico de resíduos — {fit.model_name}", fontsize=12)
    fig.tight_layout()
    return fig


# ======================================================================
# Projeção em leque (Figura 1 do guia)
# ======================================================================
def plot_projection(projection: "Projection", history=None,
                    title: Optional[str] = None) -> "Figure":
    """Gráfico em **leque**: histórico + projeção por cenário com intervalos (§5)."""
    import matplotlib.pyplot as plt

    c = _palette()
    cores_cenario = [c["primaria"], c["secundaria"], "#2ca02c", "#9467bd", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=(11, 5.5))

    if history is not None:
        hobs = history.values if hasattr(history, "kind") else history
        ax.plot(_to_ts(hobs.index), hobs.to_numpy(dtype=float), color=c["neutra"],
                lw=1.6, label="histórico", zorder=1)

    for i, (name, df) in enumerate(projection.paths.items()):
        col = cores_cenario[i % len(cores_cenario)]
        x = _to_ts(df.index)
        ax.plot(x, df["mean"].to_numpy(dtype=float), color=col, lw=2.0,
                label=f"{name} (média)", zorder=3)
        if df["lower"].notna().all():
            ax.fill_between(x, df["lower"].to_numpy(dtype=float),
                            df["upper"].to_numpy(dtype=float), color=col, alpha=0.15,
                            label=f"{name} ({int((1-projection.alpha)*100)}%)", zorder=2)
    ax.set_title(title or f"Projeção condicional — {projection.kind.upper()} (leque {int((1-projection.alpha)*100)}%)")
    ax.set_ylabel(projection.kind.upper())
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ======================================================================
# Relatório de modelo (HTML)
# ======================================================================
def _fig_to_base64(fig) -> str:
    import base64
    import io

    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def model_report(fit: "FitResult", series=None, projection: "Projection" = None,
                 title: Optional[str] = None) -> str:
    """Gera o **relatório de modelo** em HTML (Guia §6.1 (iv)): especificação,
    coeficientes, métricas, bateria de diagnóstico e (se dado) o gráfico de
    projeção. Devolve a string HTML (use :func:`save_report` para gravar)."""
    spec_desc = fit.spec.describe() if fit.spec else "—"
    coef_html = fit.coef_frame().round(4).to_html(classes="tbl", border=0)
    metrics_html = pd.DataFrame([fit.metrics()]).round(4).to_html(
        classes="tbl", border=0, index=False)
    diag_html = fit.diagnostics().round(4).to_html(classes="tbl", border=0, index=False)

    figs_html = ""
    if series is not None:
        figs_html += f'<h3>Ajuste</h3><img src="data:image/png;base64,{_fig_to_base64(plot_fit(fit, series))}"/>'
        figs_html += f'<h3>Diagnóstico de resíduos</h3><img src="data:image/png;base64,{_fig_to_base64(plot_residual_diagnostics(fit))}"/>'
    if projection is not None:
        hist = series if series is not None else None
        figs_html += f'<h3>Projeção condicional</h3><img src="data:image/png;base64,{_fig_to_base64(plot_projection(projection, hist))}"/>'

    ttl = title or f"Relatório de modelo satélite — {fit.model_name} ({fit.kind.upper()})"
    return f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<title>{ttl}</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#222;max-width:960px}}
h1{{color:#4682b4}} h3{{color:#333;border-bottom:1px solid #eee;padding-bottom:4px}}
.tbl{{border-collapse:collapse;font-size:13px;margin:8px 0}}
.tbl th,.tbl td{{padding:5px 10px;border-bottom:1px solid #eee;text-align:right}}
.tbl th{{background:#f4f7fb;color:#4682b4}}
img{{max-width:100%;height:auto;border:1px solid #eee;border-radius:6px;margin:6px 0}}
.spec{{background:#f4f7fb;padding:8px 12px;border-radius:6px;font-family:monospace}}
</style></head><body>
<h1>{ttl}</h1>
<p class="spec">{spec_desc} &nbsp;|&nbsp; kind={fit.kind} &nbsp;|&nbsp; link={fit.link} &nbsp;|&nbsp; n={fit.nobs}</p>
<h3>Coeficientes</h3>{coef_html}
<h3>Métricas in-sample</h3>{metrics_html}
<h3>Diagnóstico de resíduos (Guia §4.2)</h3>{diag_html}
{figs_html}
<p style="color:#888;font-size:11px;margin-top:20px">Gerado por yggdrasil.credit_risk.econometric —
modelos satélite de PD/LGD/CCF. Validação independente exigida antes do uso em provisão/estresse/capital.</p>
</body></html>"""


def save_report(html: str, path: str) -> str:
    """Grava o HTML do relatório em ``path`` e devolve o caminho."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


__all__ = [
    "plot_fit",
    "plot_residual_diagnostics",
    "plot_projection",
    "model_report",
    "save_report",
]
