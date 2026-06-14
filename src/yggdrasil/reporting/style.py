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


__all__ = ["COR_PRIMARIA", "COR_SECUNDARIA", "COR_NEUTRA", "colormap", "gradient"]
