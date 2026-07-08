"""Fusão monotônica de grupos por inversão (generalizada do protótipo).

A ideia: grupos são ordenados de forma crescente pelo score previsto, de modo
que a média observada do target deveria ser monotonicamente crescente. Quando
um grupo de score maior apresenta média **menor** que o anterior na amostra OOT
(uma *inversão*), testamos se a diferença é estatisticamente significativa. Se
**não** for, os dois grupos são fundidos — preservando a monotonicidade sem
descartar separações reais.

O teste de significância é escolhido pelo tipo de problema:

* regressão  -> Mann-Whitney U (target contínuo);
* classificação -> qui-quadrado de independência sobre a tabela 2x2 de
  evento/não-evento (comparação de duas proporções).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..utils import idx_para_letra

# NOTA DE DESEMPENHO: scipy.stats é importado **lazy** (dentro da função), não no
# topo — este módulo é puxado por `import yggdrasil` (via ratings) e o import no
# topo anulava o padrão lazy documentado em metrics/classification.py.


def _p_value_inversao(v_prev: np.ndarray, v_curr: np.ndarray, problem_type: str) -> float:
    """p-valor da diferença entre dois grupos adjacentes (na amostra OOT)."""
    from scipy.stats import chi2_contingency, mannwhitneyu

    if len(v_prev) == 0 or len(v_curr) == 0:
        return 1.0

    if problem_type == "classification":
        a1, a0 = int((v_prev == 1).sum()), int((v_prev == 0).sum())
        b1, b0 = int((v_curr == 1).sum()), int((v_curr == 0).sum())
        table = np.array([[a1, a0], [b1, b0]])
        # Sem variação em uma das margens não há teste possível => não funde.
        # (p <= alpha sinaliza "não fundir" no chamador, que funde só quando p > alpha.)
        if table.sum(axis=1).min() == 0 or table.sum(axis=0).min() == 0:
            return 0.0
        try:
            _, p, _, _ = chi2_contingency(table, correction=True)
        except ValueError:
            p = 1.0
        return float(p)

    # regressão
    try:
        _, p = mannwhitneyu(v_prev, v_curr, alternative="two-sided")
    except ValueError:
        p = 1.0
    return float(p)


def fundir_por_inversao(
    raw_groups,
    target,
    sample,
    *,
    oot_sample: str = "OOT",
    alpha: float = 0.05,
    problem_type: str = "regression",
    verbose: bool = False,
) -> Dict[int, str]:
    """Funde grupos crescentes com inversão não significativa no OOT.

    Parameters
    ----------
    raw_groups:
        Array de grupos brutos (inteiros), já ordenados de forma crescente pelo
        score (0 = menor score).
    target, sample:
        Arrays alinhados a ``raw_groups`` com o target observado e a amostra.
    oot_sample:
        Rótulo da amostra usada para avaliar inversões.

    Returns
    -------
    dict
        Mapeamento ``{grupo_bruto -> rótulo}`` (rótulos = letras A, B, C, ...).
    """
    raw_groups = np.asarray(raw_groups)
    target = np.asarray(target, dtype=float)
    sample = np.asarray(sample)

    grupos = sorted(int(g) for g in np.unique(raw_groups))

    oot_mask = sample == oot_sample
    g_oot = raw_groups[oot_mask]
    t_oot = target[oot_mask]

    # Sem OOT não há como avaliar inversões: rotula sem fundir.
    if oot_mask.sum() == 0:
        return {g: idx_para_letra(i) for i, g in enumerate(grupos)}

    def media_oot(cluster):
        m = np.isin(g_oot, cluster)
        return np.nan if m.sum() == 0 else float(t_oot[m].mean())

    def valores_oot(cluster):
        return t_oot[np.isin(g_oot, cluster)]

    clusters = [[g] for g in grupos]

    fundiu = True
    while fundiu:
        fundiu = False
        i = 1
        while i < len(clusters):
            prev, curr = clusters[i - 1], clusters[i]
            m_prev, m_curr = media_oot(prev), media_oot(curr)
            if np.isfinite(m_prev) and np.isfinite(m_curr) and m_curr < m_prev:
                p = _p_value_inversao(valores_oot(prev), valores_oot(curr), problem_type)
                if p > alpha:
                    clusters[i - 1] = prev + curr
                    clusters.pop(i)
                    fundiu = True
                    if verbose:
                        print(f"  Fusão {prev}+{curr} (inversão p={p:.4f} > {alpha})")
                    continue
                elif verbose:
                    print(f"  Inversão significativa entre {prev} e {curr} "
                          f"(p={p:.4f}) — não fundido")
            i += 1

    mapa: Dict[int, str] = {}
    for novo_rating, cluster in enumerate(clusters):
        letra = idx_para_letra(novo_rating)
        for g in cluster:
            mapa[int(g)] = letra
    return mapa
