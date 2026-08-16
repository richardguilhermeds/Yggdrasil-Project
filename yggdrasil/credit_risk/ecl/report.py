"""
Visualizações do subpacote de ECL (matplotlib, carregado sob demanda)
=====================================================================
Um gráfico por pergunta, na paleta única do repositório
(:mod:`yggdrasil.reporting.style`): *steelblue* como primária, *crimson* como
secundária, e o gradiente entre as duas para séries ordenadas por risco.

Todas as funções **devolvem** a ``Figure`` (ou usam o ``ax`` recebido) e nenhuma
grava em disco — quem grava é o :mod:`~yggdrasil.credit_risk.ecl.tracking`, com
``savefig``. Esse é o mesmo contrato do :mod:`yggdrasil.credit_risk.capital.report`.

O módulo é importado **sob demanda** (``__getattr__`` do pacote e imports tardios
nos métodos ``.plot()``): o núcleo do subpacote roda sem matplotlib.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd

from ...reporting.style import COR_NEUTRA, COR_PRIMARIA, COR_SECUNDARIA, gradient

#: Rótulo de eixo Y de cada representação da curva.
_ROTULO = {
    "cumulative": "PD acumulada",
    "marginal": "PD marginal",
    "hazard": "PD condicional (hazard)",
    "survival": "Sobrevivência",
}


def _ax(ax, figsize=(9, 5)):
    """Devolve ``(fig, ax)``, criando a figura quando ``ax`` é ``None``."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.figure, ax


def _serie(curve, kind: str) -> pd.Series:
    if kind not in _ROTULO:
        raise ValueError(f"kind deve ser um de {sorted(_ROTULO)}; recebido {kind!r}.")
    return getattr(curve, kind)()


# ======================================================================
# Curvas de PD
# ======================================================================
def plot_pd_curve(curve, kind: str = "cumulative", ax=None):
    """Uma curva de PD, na representação pedida."""
    fig, ax = _ax(ax)
    s = _serie(curve, kind)
    ax.plot(s.index, s.to_numpy(), color=COR_PRIMARIA, lw=2,
            label=curve.label or _ROTULO[kind])
    ax.set_xlabel("Horizonte (períodos)")
    ax.set_ylabel(_ROTULO[kind])
    ax.set_title(f"{_ROTULO[kind]} — {curve.label}" if curve.label else _ROTULO[kind])
    ax.grid(alpha=0.25)
    if kind != "survival":
        ax.set_ylim(bottom=0)
    return fig


def plot_curves(curves: Union[Mapping, Iterable], kind: str = "cumulative", ax=None,
                max_curves: int = 12):
    """Várias curvas no mesmo eixo, coloridas pelo gradiente do repositório.

    Com mais de ``max_curves`` grupos, plota os ``max_curves`` de **maior PD
    lifetime** e anota quantos ficaram de fora — um gráfico com 40 linhas não é
    um gráfico."""
    fig, ax = _ax(ax)
    itens = (list(curves.items()) if isinstance(curves, Mapping)
             else [(c.label or f"curva_{i}", c) for i, c in enumerate(curves)])
    if not itens:
        raise ValueError("nenhuma curva informada.")
    itens.sort(key=lambda kv: kv[1].pd_lifetime(), reverse=True)
    ocultas = max(0, len(itens) - int(max_curves))
    itens = itens[: int(max_curves)]

    cores = gradient(len(itens))[::-1]        # menor risco em azul, maior em vermelho
    for (rot, curva), cor in zip(itens, cores):
        s = _serie(curva, kind)
        ax.plot(s.index, s.to_numpy(), lw=2, color=cor, label=str(rot))
    ax.set_xlabel("Horizonte (períodos)")
    ax.set_ylabel(_ROTULO[kind])
    ax.set_title(f"{_ROTULO[kind]} por grupo")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    if ocultas:
        ax.annotate(f"+{ocultas} curva(s) omitida(s)", xy=(0.99, 0.02),
                    xycoords="axes fraction", ha="right", fontsize=8, color=COR_NEUTRA)
    return fig


def plot_survival_ci(table: pd.DataFrame, ax=None):
    """Curva de sobrevivência de Kaplan-Meier com a banda de Greenwood.

    ``table`` é a saída de
    :func:`~yggdrasil.credit_risk.ecl.survival.kaplan_meier` com
    ``return_table=True``."""
    fig, ax = _ax(ax)
    faltando = [c for c in ("sobrevivencia", "sobrevivencia_ic_inf", "sobrevivencia_ic_sup")
                if c not in table.columns]
    if faltando:
        raise ValueError(f"a tabela não tem as colunas {faltando} — use kaplan_meier(..., "
                         "return_table=True).")
    x = table["horizonte"] if "horizonte" in table.columns else table.index
    ax.plot(x, table["sobrevivencia"], color=COR_PRIMARIA, lw=2, label="Sobrevivência")
    ax.fill_between(x, table["sobrevivencia_ic_inf"], table["sobrevivencia_ic_sup"],
                    color=COR_PRIMARIA, alpha=0.18, label="IC (Greenwood)")
    ax.set_xlabel("Horizonte (períodos)")
    ax.set_ylabel("S(t)")
    ax.set_title("Kaplan-Meier com intervalo de confiança")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    return fig


def plot_vintage_heatmap(panel, cohort_freq: str = "Q", max_age: Optional[int] = None,
                         ax=None):
    """Mapa de calor da PD acumulada por **safra de originação × idade**.

    O gráfico que mostra os dois eixos de uma vez: a **maturação** (subir na
    horizontal) e a **qualidade da safra** (mudar de linha). Um bloco de safras
    mais escuro na mesma idade é deterioração de originação — não de ciclo."""
    fig, ax = _ax(ax, figsize=(10, 5.5))
    from ...reporting.style import colormap
    from .curves import vintage_curve

    if panel.origin_col is None and "safra_origem" not in panel.df.columns:
        # Sem originação explícita, aproxima pela primeira safra observada do contrato.
        primeiro = panel.df.groupby(panel.id_col, sort=False)[panel.date_col].transform("min")
    else:
        col = panel.origin_col or "safra_origem"
        primeiro = pd.to_datetime(panel.df[col])
    coorte = pd.PeriodIndex(primeiro, freq=cohort_freq).astype(str)

    d = panel.df.assign(_coorte=coorte.to_numpy())
    topo = int(max_age if max_age is not None else panel.max_age)
    linhas, rotulos = [], []
    for rot, parte in d.groupby("_coorte", sort=True):
        sub = type(panel)(parte.reset_index(drop=True), id_col=panel.id_col,
                          date_col=panel.date_col, default_col=panel.default_col,
                          age_col=panel.age_col, freq=panel.freq, drop_post_default=False)
        try:
            curva = vintage_curve(sub, horizon=topo + 1, fill="ffill", min_at_risk=1)
        except ValueError:
            continue
        linhas.append(curva.cumulative().to_numpy())
        rotulos.append(rot)
    if not linhas:
        raise ValueError("nenhuma coorte com dados suficientes para o mapa de calor.")

    m = np.vstack(linhas)
    im = ax.imshow(m, aspect="auto", cmap=colormap(), origin="upper")
    ax.set_yticks(range(len(rotulos)))
    ax.set_yticklabels(rotulos, fontsize=8)
    ax.set_xlabel("Idade (períodos desde a originação)")
    ax.set_ylabel("Safra de originação")
    ax.set_title("PD acumulada por safra e idade")
    fig.colorbar(im, ax=ax, label="PD acumulada")
    return fig


def plot_backtest(bt: pd.DataFrame, ax=None):
    """Previsto × observado por horizonte, com a banda do IC do observado."""
    fig, ax = _ax(ax)
    faltando = [c for c in ("horizonte", "pd_prevista", "pd_observada", "ic_inf", "ic_sup")
                if c not in bt.columns]
    if faltando:
        raise ValueError(f"a tabela de backtest não tem as colunas {faltando}.")
    grupos = list(bt["grupo"].unique()) if "grupo" in bt.columns else [None]
    cores = gradient(len(grupos))[::-1]
    for grupo, cor in zip(grupos, cores):
        g = bt if grupo is None else bt[bt["grupo"] == grupo]
        g = g.sort_values("horizonte")
        rot = "" if grupo is None else f" — {grupo}"
        ax.plot(g["horizonte"], g["pd_observada"], "o-", color=cor, lw=2,
                label=f"observado{rot}")
        ax.plot(g["horizonte"], g["pd_prevista"], "s--", color=cor, lw=1.6, alpha=0.75,
                label=f"previsto{rot}")
        ax.fill_between(g["horizonte"], g["ic_inf"], g["ic_sup"], color=cor, alpha=0.13)
    ax.set_xlabel("Horizonte (períodos)")
    ax.set_ylabel("PD acumulada")
    ax.set_title("Backtest da curva: previsto × observado")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    return fig


# ======================================================================
# ELBE
# ======================================================================
def plot_elbe(table, ax=None):
    """Recuperação acumulada e ELBE por mês em *default*, em eixos gêmeos.

    A leitura é a história inteira do *workout* num quadro: a recuperação sobe e
    achata; a ELBE sobe junto, porque o que sobra a recuperar encolhe mais
    rápido que a perda. A linha vertical marca o horizonte de *workout*."""
    fig, ax = _ax(ax)
    f = table.frame
    ax.plot(f.index, f["recuperacao_acumulada"], color=COR_PRIMARIA, lw=2,
            label="Recuperação acumulada")
    ax.set_xlabel("Meses desde o default")
    ax.set_ylabel("Recuperação acumulada", color=COR_PRIMARIA)
    ax.tick_params(axis="y", labelcolor=COR_PRIMARIA)
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(f.index, f["elbe"], color=COR_SECUNDARIA, lw=2, label="ELBE")
    if "lgd_in_default" in f.columns and table.addon > 0:
        ax2.plot(f.index, f["lgd_in_default"], color=COR_SECUNDARIA, lw=1.4, ls="--",
                 label="LGD in default")
    ax2.set_ylabel("ELBE (sobre o saldo remanescente)", color=COR_SECUNDARIA)
    ax2.tick_params(axis="y", labelcolor=COR_SECUNDARIA)
    ax2.set_ylim(0, 1.02)

    ax.axvline(table.workout, color=COR_NEUTRA, ls=":", lw=1.5)
    ax.annotate(f"workout = {table.workout}", xy=(table.workout, 0.02),
                xytext=(4, 0), textcoords="offset points", fontsize=8, color=COR_NEUTRA)
    ax.set_title(f"Curva de recuperação e ELBE — LGD do ciclo = {table.lgd:.1%}")

    linhas = ax.get_lines()[:1] + ax2.get_lines()
    ax.legend(linhas, [l.get_label() for l in linhas], fontsize=9, loc="center right")
    return fig


# ======================================================================
# CCF
# ======================================================================
def plot_ccf_distribution(data, bins: int = 25, ax=None):
    """Histograma do CCF com as **massas em 0 e em 1** destacadas.

    A bimodalidade é o fato central do CCF, e é ela que decide o estimador: se
    quase tudo está nos extremos, a média agrupada descreve mal a carteira e um
    modelo de resposta fracionária (ou de mistura) passa a valer o custo."""
    fig, ax = _ax(ax)
    # Aceita CCFDataset (``.values`` é a Series da medida), Series ou array cru.
    bruto = data.values if hasattr(data, "measure") else data
    v = np.asarray(bruto, dtype=float).ravel()
    em_zero, em_um = v <= 1e-9, v >= 1.0 - 1e-9
    meio = v[~(em_zero | em_um)]
    if meio.size:
        ax.hist(meio, bins=int(bins), range=(0, 1), color=COR_PRIMARIA, alpha=0.75,
                label="interior (0, 1)")
    altura = max(1.0, (np.histogram(meio, bins=int(bins), range=(0, 1))[0].max()
                       if meio.size else 1.0))
    for massa, x, rot in ((em_zero.sum(), 0.0, "massa em 0"), (em_um.sum(), 1.0, "massa em 1")):
        if massa:
            ax.bar([x], [massa], width=0.03, color=COR_SECUNDARIA, label=rot)
    ax.set_xlabel(getattr(data, "measure", "ccf"))
    ax.set_ylabel("Nº de observações")
    ax.set_title(
        f"Distribuição do {getattr(data, 'measure', 'CCF')} — "
        f"{em_zero.mean():.1%} em 0 · {em_um.mean():.1%} em 1"
    )
    ax.set_ylim(0, altura * 1.15)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9)
    return fig


def plot_ccf_by_horizon(data, ax=None):
    """CCF médio por **meses até o *default*** — a curva de aceleração do saque.

    Faz sentido com ``method='variable'``, que gera uma observação por mês da
    janela: mostra o quanto do limite disponível vai sendo consumido à medida que
    o *default* se aproxima, e é o argumento visual para a escolha do horizonte."""
    fig, ax = _ax(ax)
    frame = data.frame if hasattr(data, "frame") else pd.DataFrame(data)
    medida = getattr(data, "measure", "ccf")
    g = frame.groupby("meses_ate_default")[medida].agg(["mean", "median", "size"])
    ax.plot(g.index, g["mean"], "o-", color=COR_PRIMARIA, lw=2, label="média")
    ax.plot(g.index, g["median"], "s--", color=COR_SECUNDARIA, lw=1.6, label="mediana")
    ax.set_xlabel("Meses entre a data de referência e o default")
    ax.set_ylabel(medida)
    ax.set_title(f"{medida} por distância até o default")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    return fig


# ======================================================================
# ECL
# ======================================================================
def plot_ecl_scenarios(scenarios: Mapping, ax=None):
    """ECL por cenário, com a linha do valor ponderado.

    ``scenarios`` é a saída de
    :func:`~yggdrasil.credit_risk.ecl.ecl.ecl_scenarios`."""
    fig, ax = _ax(ax, figsize=(8, 4.5))
    df = scenarios["por_cenario"].sort_values("ecl")
    cores = gradient(len(df))
    ax.bar(df["cenario"], df["ecl"], color=cores, alpha=0.85)
    ax.axhline(scenarios["ponderado"], color=COR_NEUTRA, ls="--", lw=1.6,
               label=f"ponderado = {scenarios['ponderado']:,.0f}")
    for x, (v, p) in enumerate(zip(df["ecl"], df["peso"])):
        ax.annotate(f"peso {p:.0%}", xy=(x, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color=COR_NEUTRA)
    ax.set_ylabel("ECL")
    ax.set_title("ECL por cenário macroeconômico")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9)
    return fig


def plot_ecl_by(result, *cols: str, ax=None):
    """ECL e taxa de provisão por grupo (estágio, produto, safra…)."""
    fig, ax = _ax(ax, figsize=(9, 4.5))
    g = result.by(*cols)
    rotulos = (g[list(cols)].astype(str).agg(" · ".join, axis=1) if cols
               else pd.Series(["carteira"]))
    ax.bar(rotulos, g["ecl"], color=COR_PRIMARIA, alpha=0.85, label="ECL")
    ax.set_ylabel("ECL")
    ax.tick_params(axis="x", rotation=30)
    ax2 = ax.twinx()
    ax2.plot(rotulos, g["taxa_provisao"], "o-", color=COR_SECUNDARIA, lw=2,
             label="taxa de provisão")
    ax2.set_ylabel("Taxa de provisão", color=COR_SECUNDARIA)
    ax2.tick_params(axis="y", labelcolor=COR_SECUNDARIA)
    ax.set_title("ECL e cobertura por " + (" · ".join(cols) if cols else "carteira"))
    ax.grid(alpha=0.25, axis="y")
    return fig


__all__ = [
    "plot_pd_curve", "plot_curves", "plot_survival_ci", "plot_vintage_heatmap",
    "plot_backtest", "plot_elbe", "plot_ccf_distribution", "plot_ccf_by_horizon",
    "plot_ecl_scenarios", "plot_ecl_by",
]
