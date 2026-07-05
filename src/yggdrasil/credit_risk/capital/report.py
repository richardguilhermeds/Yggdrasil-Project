"""
Visualizações de capital econômico — figuras do guia
====================================================
Este módulo desenha, no estilo visual do ``yggdrasil`` (paleta *steelblue* +
*crimson* de :mod:`yggdrasil.reporting.style`), as figuras que traduzem os
resultados dos motores de capital em leitura gerencial e regulatória:

* **Figura 1 — distribuição de perdas** (:func:`plot_loss_distribution`): a
  imagem central do guia. A **provisão** cobre a perda esperada (EL); o
  **capital econômico** cobre a perda inesperada de ``EL`` até o quantil
  ``VaR_q``; a **cauda** além do VaR é o risco residual aceito pelo apetite de
  risco. As três regiões aparecem sombreadas.
* **Alocação por produto** (:func:`plot_allocation`): capital *isolado* vs
  *alocado* por segmento — o desenho do benefício de diversificação na ponta.
* **Benefício de diversificação** (:func:`plot_diversification`): *waterfall*
  do capital isolado → benefício → capital integrado da carteira.
* **Convergência** (:func:`plot_convergence`, bloco E do guia): estabilização
  do VaR/ES conforme cresce o nº de cenários de Monte Carlo.
* **Comparação de métodos** (:func:`plot_capital_comparison`): capital econômico
  por metodologia (ASRF × Monte Carlo × CreditRisk+ × Pilar 1).

Contexto regulatório: Resolução CMN 4.557/2017 (ICAAP), 4.966/2021 (perda
esperada / provisão) e Basileia/IRB (fórmula ASRF do Pilar 1). matplotlib é
sempre importado **tardiamente**, dentro de cada função, porque o pacote só o
usa em relatório/tracking — não é dependência de tempo de cálculo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .measures import DEFAULT_CONFIDENCE, LossDistribution

if TYPE_CHECKING:  # apenas para type hints; sem custo em runtime
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .monte_carlo import SimulationResult


# ======================================================================
# Paleta — importada TARDIAMENTE de reporting.style, com fallback neutro
# ======================================================================
def _palette() -> dict:
    """Cores do tema do repositório (steelblue/crimson) com *fallback* seguro.

    O import é tardio e protegido: se o módulo de estilo mudar de forma ou não
    estiver disponível, as figuras ainda saem com uma paleta hex neutra em vez
    de quebrar.
    """
    cores = {
        "primaria": "#4682b4",    # steelblue
        "secundaria": "#dc143c",  # crimson
        "neutra": "#888888",
    }
    try:
        from ...reporting.style import (  # import tardio
            COR_NEUTRA,
            COR_PRIMARIA,
            COR_SECUNDARIA,
        )

        cores["primaria"] = COR_PRIMARIA
        cores["secundaria"] = COR_SECUNDARIA
        cores["neutra"] = COR_NEUTRA
    except Exception:  # pragma: no cover - só quando o estilo some/muda
        pass
    return cores


def _style_ax(ax: "Axes") -> None:
    """Grade discreta por eixo (mesmo padrão do dashboard do repositório)."""
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, linewidth=0.8)


def _new_fig_ax(ax: Optional["Axes"], figsize):
    """Devolve ``(fig, ax)``. Se ``ax`` vier, respeita-o; senão cria uma figura
    OO (fora do ``pyplot.Gcf``) para não duplicar no auto-display do Jupyter."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if ax is not None:
        return ax.get_figure(), ax
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)  # canvas Agg => savefig funciona sem backend interativo
    ax = fig.add_subplot(111)
    return fig, ax


def _fmt_moeda(x: float) -> str:
    """Formata um valor monetário de forma compacta (mil/mi/bi)."""
    ax = abs(float(x))
    if ax >= 1e9:
        return f"{x / 1e9:,.2f} bi"
    if ax >= 1e6:
        return f"{x / 1e6:,.2f} mi"
    if ax >= 1e3:
        return f"{x / 1e3:,.1f} mil"
    return f"{x:,.2f}"


# ======================================================================
# Normalização da entrada de plot_loss_distribution
# ======================================================================
def _as_loss_dist(
    dist: Union["LossDistribution", "SimulationResult", np.ndarray, Sequence[float]],
) -> "LossDistribution":
    """Normaliza a entrada para uma :class:`LossDistribution`.

    Aceita:

    * :class:`~yggdrasil.credit_risk.capital.measures.LossDistribution` — usada
      diretamente;
    * :class:`~yggdrasil.credit_risk.capital.monte_carlo.SimulationResult` — via
      ``.distribution()`` (carrega a EL analítica exata);
    * ``numpy.ndarray`` / sequência de perdas — envelopada como amostra empírica.
    """
    # LossDistribution: pato — tem os métodos que usamos.
    if isinstance(dist, LossDistribution):
        return dist
    # SimulationResult (import tardio p/ evitar ciclo): expõe .distribution().
    if hasattr(dist, "distribution") and callable(getattr(dist, "distribution")):
        d = dist.distribution()
        if isinstance(d, LossDistribution):
            return d
    # Array/sequência de perdas.
    arr = np.asarray(dist, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("A distribuição de perdas está vazia.")
    return LossDistribution(arr, name="carteira")


# ======================================================================
# Figura 1 — distribuição de perdas (provisão / capital / cauda)
# ======================================================================
def plot_loss_distribution(
    dist: Union["LossDistribution", "SimulationResult", np.ndarray, Sequence[float]],
    q: float = DEFAULT_CONFIDENCE,
    bins: int = 120,
    ax: Optional["Axes"] = None,
    title: Optional[str] = None,
) -> "Figure":
    """Figura 1 do guia — a **distribuição de perdas** da carteira.

    Desenha o histograma (densidade) das perdas e marca as três zonas de
    cobertura, com linhas verticais em ``EL``, ``VaR_q`` e ``ES_q``:

    * ``[0, EL]`` — **provisão** (perda esperada, custo previsível: ECL/IFRS 9,
      Resolução CMN 4.966);
    * ``[EL, VaR_q]`` — **capital econômico** (perda inesperada; só absorvível por
      capital, Resolução CMN 4.557/ICAAP);
    * ``> VaR_q`` — **cauda / apetite de risco** (perda residual aceita acima do
      nível de confiança).

    Parameters
    ----------
    dist:
        :class:`LossDistribution`, :class:`SimulationResult` ou vetor de perdas.
    q:
        Nível de confiança do VaR/ES (padrão 99,9%).
    bins:
        Número de classes do histograma.
    ax:
        Eixo onde desenhar; se ``None``, cria uma figura própria.
    title:
        Título; se ``None``, usa um padrão.

    Returns
    -------
    matplotlib.figure.Figure
    """
    q = float(q)
    if not (0.0 < q < 1.0):
        raise ValueError(f"q deve estar em (0, 1); recebido {q!r}.")

    d = _as_loss_dist(dist)
    cores = _palette()
    fig, ax = _new_fig_ax(ax, figsize=(11, 6))
    _style_ax(ax)

    losses = np.asarray(d.losses, dtype=float)
    el = float(d.el)
    var = float(d.var(q))
    es = float(d.es(q))

    # Histograma como densidade. Distribuição discreta ponderada respeita os pesos.
    weights = d.weights if d.weights is not None else None
    ax.hist(losses, bins=bins, density=True, weights=weights,
            color=cores["primaria"], alpha=0.45, edgecolor="white", linewidth=0.4,
            zorder=2)

    # Eixo de cor das três regiões de cobertura (sombreamento vertical).
    ymin, ymax = ax.get_ylim()
    x_max = float(np.max(losses))
    # A cauda vai do VaR até um pouco além da maior perda observada.
    x_tail = max(x_max, var, el) * 1.02
    if x_tail <= 0:                       # carteira que nunca perde: evita xlim (0, 0)
        x_tail = 1.0

    ax.axvspan(0.0, el, color=cores["neutra"], alpha=0.16, zorder=1,
               label=f"Provisão · EL = {_fmt_moeda(el)}")
    ax.axvspan(el, var, color=cores["primaria"], alpha=0.20, zorder=1,
               label=f"Capital econômico · CE = {_fmt_moeda(var - el)}")
    ax.axvspan(var, x_tail, color=cores["secundaria"], alpha=0.16, zorder=1,
               label=f"Cauda / apetite (> VaR)")

    # Linhas verticais nas métricas.
    ax.axvline(el, color=cores["neutra"], linestyle="--", linewidth=2.0, zorder=4)
    ax.axvline(var, color=cores["secundaria"], linestyle="-", linewidth=2.2, zorder=4,
               label=f"VaR {q:.3%} = {_fmt_moeda(var)}")
    ax.axvline(es, color=cores["secundaria"], linestyle=":", linewidth=2.0, zorder=4,
               label=f"ES {q:.3%} = {_fmt_moeda(es)}")

    ax.set_xlim(0.0, x_tail)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Perda da carteira", fontsize=11)
    ax.set_ylabel("Densidade", fontsize=11)
    ax.set_title(title or "Distribuição de perdas da carteira",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7, loc="upper right")
    return fig


# ======================================================================
# Alocação — capital isolado vs alocado por segmento
# ======================================================================
def plot_allocation(allocation_df: pd.DataFrame, ax: Optional["Axes"] = None) -> "Figure":
    """Barras de **capital isolado × capital alocado** por segmento.

    A diferença entre a barra isolada (CE do segmento sozinho) e a alocada
    (contribuição ao risco conjunto, via Euler) é o **benefício de
    diversificação** que chega a cada produto (Seção 6 do guia).

    Parameters
    ----------
    allocation_df:
        Saída de :func:`~yggdrasil.credit_risk.capital.allocation.euler_allocation`
        (precisa das colunas ``segmento``, ``capital_isolado``,
        ``capital_alocado``).
    ax:
        Eixo onde desenhar; se ``None``, cria uma figura própria.

    Returns
    -------
    matplotlib.figure.Figure
    """
    obrig = {"segmento", "capital_isolado", "capital_alocado"}
    faltando = obrig - set(allocation_df.columns)
    if faltando:
        raise ValueError(
            f"allocation_df não tem as colunas exigidas: {sorted(faltando)}. "
            "Use a saída de euler_allocation().")

    cores = _palette()
    segs = allocation_df["segmento"].astype(str).tolist()
    isolado = allocation_df["capital_isolado"].to_numpy(dtype=float)
    alocado = allocation_df["capital_alocado"].to_numpy(dtype=float)

    n = len(segs)
    fig, ax = _new_fig_ax(ax, figsize=(max(8, 1.6 * n + 3), 6))
    _style_ax(ax)

    x = np.arange(n)
    w = 0.4
    b_iso = ax.bar(x - w / 2, isolado, width=w, color=cores["neutra"], alpha=0.75,
                   edgecolor="white", label="Capital isolado (standalone)")
    b_aloc = ax.bar(x + w / 2, alocado, width=w, color=cores["primaria"], alpha=0.85,
                    edgecolor="white", label="Capital alocado (Euler)")

    # Rótulos numéricos no topo das barras.
    for barras in (b_iso, b_aloc):
        ax.bar_label(barras, labels=[_fmt_moeda(v) for v in barras.datavalues],
                     fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(segs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Capital econômico", fontsize=11)
    ax.set_xlabel("Segmento", fontsize=11)
    ax.set_title("Capital isolado × alocado por segmento (benefício de diversificação)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.7)
    return fig


# ======================================================================
# Benefício de diversificação — waterfall
# ======================================================================
def plot_diversification(sim_result: "SimulationResult", ax: Optional["Axes"] = None) -> "Figure":
    """*Waterfall* do benefício de diversificação da carteira.

    Parte do **capital isolado** (soma dos CE de cada segmento sozinho),
    desconta o **benefício de diversificação** e chega ao **capital integrado**
    da carteira. É a informação gerencial que o Pilar 1 não entrega (Seção 6).

    Parameters
    ----------
    sim_result:
        :class:`~yggdrasil.credit_risk.capital.monte_carlo.SimulationResult`
        simulado com ``store_segment_losses=True`` (usa
        ``.diversification_benefit()``).
    ax:
        Eixo onde desenhar; se ``None``, cria uma figura própria.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not hasattr(sim_result, "diversification_benefit"):
        raise ValueError(
            "sim_result deve ser um SimulationResult com diversification_benefit().")
    info = sim_result.diversification_benefit()
    isolado = float(info["capital_isolado"])
    beneficio = float(info["beneficio_diversificacao"])
    integrado = float(info["capital_integrado"])
    beneficio_pct = info.get("beneficio_pct", np.nan)

    cores = _palette()
    fig, ax = _new_fig_ax(ax, figsize=(8, 6))
    _style_ax(ax)

    # Três barras: isolado (cheia) → benefício (queda) → integrado (cheia).
    rotulos = ["Capital\nisolado", "Benefício de\ndiversificação", "Capital\nintegrado"]
    x = np.arange(3)

    # Barra 1: do zero ao isolado.
    ax.bar(0, isolado, width=0.6, color=cores["neutra"], alpha=0.8, edgecolor="white")
    # Barra 2 (flutuante): a queda do isolado até o integrado.
    ax.bar(1, -beneficio, bottom=isolado, width=0.6, color=cores["secundaria"],
           alpha=0.75, edgecolor="white")
    # Barra 3: do zero ao integrado.
    ax.bar(2, integrado, width=0.6, color=cores["primaria"], alpha=0.85, edgecolor="white")

    # Linha-guia ligando o topo do isolado ao início da queda e ao integrado.
    ax.plot([0.3, 0.7], [isolado, isolado], color=cores["neutra"], linewidth=1.0,
            linestyle="--")
    ax.plot([1.3, 1.7], [integrado, integrado], color=cores["neutra"], linewidth=1.0,
            linestyle="--")

    # Rótulos numéricos.
    ax.text(0, isolado, f"  {_fmt_moeda(isolado)}", ha="center", va="bottom", fontsize=10,
            fontweight="bold")
    pct_txt = "" if not np.isfinite(beneficio_pct) else f"\n({beneficio_pct:.1%})"
    ax.text(1, isolado - beneficio / 2, f"−{_fmt_moeda(beneficio)}{pct_txt}", ha="center",
            va="center", fontsize=9, fontweight="bold", color="white")
    ax.text(2, integrado, f"  {_fmt_moeda(integrado)}", ha="center", va="bottom", fontsize=10,
            fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(rotulos, fontsize=10)
    ax.set_ylabel("Capital econômico", fontsize=11)
    ax.set_ylim(0, isolado * 1.15 if isolado > 0 else None)
    ax.set_title("Benefício de diversificação da carteira", fontsize=13, fontweight="bold")
    return fig


# ======================================================================
# Convergência — VaR/ES vs nº de cenários
# ======================================================================
def plot_convergence(convergence_df: pd.DataFrame, ax: Optional["Axes"] = None) -> "Figure":
    """Estabilização do **VaR/ES** conforme cresce o nº de cenários (bloco E).

    O quantil de cauda (99,9%) é ruidoso com poucos cenários; o gráfico mostra
    quando a estimativa estabiliza — evidência de que a simulação convergiu.

    Parameters
    ----------
    convergence_df:
        DataFrame com colunas ``n_cenarios``, ``VaR`` e ``ES`` (saída de
        :func:`yggdrasil.credit_risk.capital.validation.convergence`).
    ax:
        Eixo onde desenhar; se ``None``, cria uma figura própria.

    Returns
    -------
    matplotlib.figure.Figure
    """
    obrig = {"n_cenarios", "VaR", "ES"}
    faltando = obrig - set(convergence_df.columns)
    if faltando:
        raise ValueError(
            f"convergence_df não tem as colunas exigidas: {sorted(faltando)}. "
            "Esperado ['n_cenarios', 'VaR', 'ES'].")

    cores = _palette()
    d = convergence_df.sort_values("n_cenarios")
    n = d["n_cenarios"].to_numpy(dtype=float)
    var = d["VaR"].to_numpy(dtype=float)
    es = d["ES"].to_numpy(dtype=float)

    fig, ax = _new_fig_ax(ax, figsize=(10, 6))
    _style_ax(ax)

    ax.plot(n, var, color=cores["primaria"], marker="o", markersize=5, linewidth=2.0,
            label="VaR")
    ax.plot(n, es, color=cores["secundaria"], marker="s", markersize=5, linewidth=2.0,
            label="ES")

    # Faixa de referência ±2% em torno da estimativa mais fina (maior n): visualiza
    # a estabilização — a curva deve entrar e ficar dentro da faixa.
    var_ref = float(var[-1])
    if np.isfinite(var_ref) and var_ref != 0:
        ax.axhspan(var_ref * 0.98, var_ref * 1.02, color=cores["primaria"], alpha=0.10,
                   label="±2% do VaR final")

    ax.set_xscale("log")
    ax.set_xlabel("Nº de cenários (escala log)", fontsize=11)
    ax.set_ylabel("Métrica de risco", fontsize=11)
    ax.set_title("Convergência do VaR/ES com o nº de cenários",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.7)
    return fig


# ======================================================================
# Comparação de capital por método
# ======================================================================
def plot_capital_comparison(
    labels: Sequence[str],
    capitals: Sequence[float],
    ax: Optional["Axes"] = None,
) -> "Figure":
    """Barras comparando o **capital econômico por metodologia**.

    Ex.: ASRF (analítico) × Monte Carlo (empírico) × CreditRisk+ (analítico) ×
    Pilar 1 (regulatório). Contrastar os métodos evidencia o efeito da
    diversificação multifatorial e a distância para o piso regulatório.

    Parameters
    ----------
    labels:
        Nomes dos métodos (eixo X).
    capitals:
        Capital econômico de cada método, na mesma ordem de ``labels``.
    ax:
        Eixo onde desenhar; se ``None``, cria uma figura própria.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = list(labels)
    capitals = np.asarray(list(capitals), dtype=float)
    if len(labels) != len(capitals):
        raise ValueError("labels e capitals devem ter o mesmo tamanho.")
    if len(labels) == 0:
        raise ValueError("Informe ao menos um método para comparar.")

    cores = _palette()
    # Gradiente steelblue → crimson quando houver estilo; senão, cor primária.
    try:
        from ...reporting.style import gradient  # import tardio

        paleta = gradient(len(labels))
    except Exception:  # pragma: no cover
        paleta = [cores["primaria"]] * len(labels)

    n = len(labels)
    fig, ax = _new_fig_ax(ax, figsize=(max(7, 1.5 * n + 2), 6))
    _style_ax(ax)

    x = np.arange(n)
    barras = ax.bar(x, capitals, color=paleta, alpha=0.85, edgecolor="white")
    ax.bar_label(barras, labels=[_fmt_moeda(v) for v in capitals], fontsize=9, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Capital econômico", fontsize=11)
    ax.set_title("Capital econômico por metodologia", fontsize=13, fontweight="bold")
    return fig


__all__ = [
    "plot_loss_distribution",
    "plot_allocation",
    "plot_diversification",
    "plot_convergence",
    "plot_capital_comparison",
]
