"""Paleta padrão das análises visuais do ``yggdrasil``.

Cores padrão: **steelblue** (primária) e **crimson** (secundária). Este é o
ponto único de configuração do tema — altere aqui para mudar a cor de todos os
dashboards, gráficos de PSI e relatórios.
"""

from __future__ import annotations

from typing import List

COR_PRIMARIA = "steelblue"
COR_SECUNDARIA = "crimson"
COR_NEUTRA = "#888888"

# Abreviações de mês em PT-BR (independem do locale do SO; Windows não traz pt_BR).
MESES_PT = ("jan", "fev", "mar", "abr", "mai", "jun",
            "jul", "ago", "set", "out", "nov", "dez")


def fmt_month_year(values) -> List[str]:
    """Rótulos de data → **'mmm/aa'** (ex.: ``'2022-01'`` → ``'jan/22'``).

    Formato **padrão de mês/ano** dos gráficos temporais do repositório (eixo X
    com data/safra). Aceita ``str`` no padrão ``'AAAA-MM[-DD]'`` e objetos com
    ``.month``/``.year`` (``datetime``, ``pandas.Timestamp``, ``pandas.Period``).
    Valores fora do padrão voltam como ``str``, sem alteração."""
    out: List[str] = []
    for v in values:
        try:
            if hasattr(v, "month") and hasattr(v, "year"):     # datetime/Timestamp/Period
                out.append(f"{MESES_PT[int(v.month) - 1]}/{str(int(v.year))[-2:]}")
                continue
            s = str(v)
            ano, mes = s.split("-")[:2]
            out.append(f"{MESES_PT[int(mes) - 1]}/{ano[-2:]}")
        except (ValueError, IndexError, TypeError):
            out.append(str(v))
    return out


def month_year_axis(ax, values=None) -> None:
    """Formata o eixo X de ``ax`` como **mmm/aa** (padrão do repositório).

    Com ``values`` (as datas/posições dos ticks), fixa os ticks em ``0..n-1`` e
    aplica :func:`fmt_month_year`. Sem ``values``, reescreve os rótulos atuais do
    eixo (útil quando já são textos 'AAAA-MM')."""
    if values is not None:
        vals = list(values)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(fmt_month_year(vals))
    else:
        labels = [t.get_text() for t in ax.get_xticklabels()]
        if any(labels):
            ax.set_xticklabels(fmt_month_year(labels))


def colormap():
    """Colormap contínuo steelblue → crimson (para beeswarm/heatmaps)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("yggdrasil", [COR_PRIMARIA, COR_SECUNDARIA])


def gradient(n: int) -> List:
    """Retorna ``n`` cores interpoladas de steelblue a crimson.

    Útil para séries com várias categorias — ex.: ratings ordenados do menor ao
    maior risco (azul → vermelho).
    """
    if n <= 0:
        return []
    if n == 1:
        return [COR_PRIMARIA]
    cmap = colormap()
    return [cmap(i / (n - 1)) for i in range(n)]


__all__ = ["COR_PRIMARIA", "COR_SECUNDARIA", "COR_NEUTRA", "colormap", "gradient",
           "MESES_PT", "fmt_month_year", "month_year_axis"]
