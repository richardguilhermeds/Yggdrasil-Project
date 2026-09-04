"""
Tabela de vida, Nelson-Aalen e o teste de log-rank
==================================================
As ferramentas **não paramétricas** da análise de sobrevivência aplicadas ao
painel de contratos: o que a carteira mostrou, sem hipótese funcional.

:func:`life_table`
    A tabela de vida atuarial por idade (uma linha por idade, opcionalmente por
    grupo): base em risco, quebras, censura, *hazard*, sobrevivência de
    Kaplan-Meier com o erro padrão de Greenwood e o IC log-log, mais o *hazard*
    acumulado de **Nelson-Aalen** e a sobrevivência que ele implica. É o anexo
    de documentação de qualquer curva *lifetime* e o insumo dos ajustes
    paramétricos.

:func:`logrank_test`
    O teste de **log-rank** (Mantel-Cox) para ``k`` grupos: as curvas de
    sobrevivência dos segmentos são estatisticamente distintas? É a pergunta
    que decide se vale ajustar uma curva **por segmento** ou uma única para a
    carteira. A variante de **Wilcoxon** (Gehan-Breslow) pesa as idades baixas,
    onde a base é maior.

:func:`median_survival`, :func:`restricted_mean_survival`
    Leituras da curva em unidades de **tempo**: em quantos períodos metade da
    coorte quebra, e quantos períodos um contrato sobrevive em média dentro de
    um horizonte (a *restricted mean survival time*).

Tudo devolve ``pandas``/``float`` puros e roda sobre o
:class:`~yggdrasil.credit_risk.ecl.panel.ContractPanel`, com as mesmas
convenções de idade e horizonte do resto do eixo *lifetime*.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..ecl.curves import PDCurve
from ..ecl.panel import ContractPanel
from ..ecl.survival import kaplan_meier

#: Rótulo da tabela quando o painel não é quebrado por grupo.
GLOBAL = "__global__"

#: Pesos suportados pelo teste de log-rank.
LOGRANK_WEIGHTS = ("logrank", "wilcoxon")


# ======================================================================
# Tabela de vida
# ======================================================================
def _uma_tabela(panel: ContractPanel, from_age: int, horizon: Optional[int],
                alpha: float, weighted: bool) -> pd.DataFrame:
    _, tab = kaplan_meier(panel, from_age=from_age, horizon=horizon, weighted=weighted,
                          alpha=alpha, return_table=True)
    n = tab["n_em_risco"].to_numpy(dtype=float)
    d = tab["n_default"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        incremento = np.where(n > 0, d / n, 0.0)
    tab = tab.copy()
    # Nelson-Aalen: H(t) = Σ d_k / n_k; a sobrevivência implicada é exp(-H)
    tab["hazard_acumulado"] = np.cumsum(incremento)
    tab["sobrevivencia_na"] = np.exp(-tab["hazard_acumulado"])
    tab["hazard"] = tab["hazard"].fillna(0.0)
    tab.index.name = "idade"
    return tab.reset_index()


def life_table(panel: ContractPanel, by: Optional[str] = None, from_age: int = 0,
               horizon: Optional[int] = None, alpha: float = 0.05,
               weighted: bool = False) -> pd.DataFrame:
    """Tabela de vida por idade (e por grupo, com ``by``).

    Colunas: ``idade``, ``horizonte``, ``n_em_risco``, ``n_default``,
    ``n_censurado``, ``hazard``, ``sobrevivencia``, ``se_greenwood``,
    ``sobrevivencia_ic_inf/sup``, ``pd_acumulada`` (e os seus ICs),
    ``hazard_acumulado`` (Nelson-Aalen) e ``sobrevivencia_na``. Com ``by``, a
    coluna ``grupo`` vem na frente.

    Parameters
    ----------
    panel:
        O painel de contratos.
    by:
        Coluna de agrupamento (produto, rating...). ``None`` = carteira inteira.
    from_age, horizon, alpha, weighted:
        Repassados a :func:`~yggdrasil.credit_risk.ecl.survival.kaplan_meier`.
    """
    if by is None:
        return _uma_tabela(panel, from_age, horizon, alpha, weighted)
    partes = []
    for rot, parte in panel.by(by).items():
        try:
            t = _uma_tabela(parte, from_age, horizon, alpha, weighted)
        except ValueError:
            continue
        t.insert(0, "grupo", rot)
        partes.append(t)
    if not partes:
        raise ValueError(f"nenhum grupo de {by!r} tem observações a partir de from_age={from_age}.")
    return pd.concat(partes, ignore_index=True)


# ======================================================================
# Log-rank
# ======================================================================
def _at_risk_por_grupo(panel: ContractPanel, by: str, from_age: int,
                       horizon: Optional[int]) -> Dict[object, pd.DataFrame]:
    partes = panel.by(by)
    topo = max(p.max_age for p in partes.values())
    if horizon is not None:
        topo = min(topo, int(from_age) + int(horizon) - 1)
    saida = {}
    for rot, parte in partes.items():
        vida = parte.at_risk(max_age=topo)
        saida[rot] = vida[vida.index >= int(from_age)]
    return saida


def logrank_test(panel: ContractPanel, by: str, from_age: int = 0,
                 horizon: Optional[int] = None, weights: str = "logrank") -> dict:
    """Teste de log-rank (Mantel-Cox) para ``k`` grupos do painel.

    H0: as funções de sobrevivência dos grupos são iguais. Em cada idade ``t``
    com quebras, o nº esperado no grupo ``g`` sob H0 é ``e_gt = n_gt · d_t / n_t``
    e a estatística compara observado e esperado acumulados, com a covariância
    hipergeométrica. Distribuição de referência: qui-quadrado com ``k − 1``
    graus de liberdade.

    Parameters
    ----------
    panel:
        O painel de contratos (contagem, não ponderado por exposição).
    by:
        Coluna que define os grupos.
    from_age, horizon:
        Janela de idades considerada.
    weights:
        ``'logrank'`` (peso 1 em todas as idades) ou ``'wilcoxon'``
        (Gehan-Breslow, peso ``n_t``: mais sensível a diferenças nas idades
        baixas, onde a base é maior).

    Returns
    -------
    dict
        ``estatistica``, ``gl``, ``p_valor``, ``n_grupos``, ``weights`` e a tabela
        ``grupos`` (uma linha por grupo: base em risco inicial, observados, esperados e
        a razão ``observado/esperado``, que é a leitura de risco relativo).
    """
    if weights not in LOGRANK_WEIGHTS:
        raise ValueError(f"weights deve ser um de {LOGRANK_WEIGHTS}; recebido {weights!r}.")
    vidas = _at_risk_por_grupo(panel, by, from_age, horizon)
    grupos = list(vidas)
    k = len(grupos)
    if k < 2:
        raise ValueError(f"o teste exige ao menos 2 grupos; {by!r} tem {k}.")

    idades = sorted(set().union(*[set(v.index) for v in vidas.values()]))
    N = np.array([[float(vidas[g]["n_em_risco"].get(t, 0)) for g in grupos] for t in idades])
    D = np.array([[float(vidas[g]["n_default"].get(t, 0)) for g in grupos] for t in idades])
    n_t, d_t = N.sum(axis=1), D.sum(axis=1)
    valido = (n_t > 1) & (d_t > 0)
    N, D, n_t, d_t = N[valido], D[valido], n_t[valido], d_t[valido]
    if not len(n_t):
        raise ValueError("nenhuma idade com quebras e base > 1 na janela pedida.")

    w = np.ones_like(n_t) if weights == "logrank" else n_t
    E = N * (d_t / n_t)[:, None]
    U = (w[:, None] * (D - E)).sum(axis=0)
    V = np.zeros((k, k))
    fator = w ** 2 * d_t * (n_t - d_t) / (n_t ** 2 * (n_t - 1.0))
    for g in range(k):
        for h in range(k):
            delta = 1.0 if g == h else 0.0
            V[g, h] = np.sum(fator * (N[:, g] * n_t * delta - N[:, g] * N[:, h]))
    Ur, Vr = U[:-1], V[:-1, :-1]
    estat = float(Ur @ np.linalg.pinv(Vr) @ Ur)
    from scipy.stats import chi2

    gl = k - 1
    p = float(chi2.sf(estat, gl))
    tabela = pd.DataFrame({
        "grupo": grupos,
        "n_em_risco_inicial": [int(vidas[g]["n_em_risco"].iloc[0]) if len(vidas[g]) else 0
                               for g in grupos],
        "observados": D.sum(axis=0),
        "esperados": E.sum(axis=0),
    })
    with np.errstate(divide="ignore", invalid="ignore"):
        tabela["obs_esp"] = np.where(tabela["esperados"] > 0,
                                     tabela["observados"] / tabela["esperados"], np.nan)
    return {"estatistica": estat, "gl": gl, "p_valor": p, "n_grupos": k,
            "weights": weights, "grupos": tabela}


def pairwise_logrank(panel: ContractPanel, by: str, from_age: int = 0,
                     horizon: Optional[int] = None, weights: str = "logrank") -> pd.DataFrame:
    """Log-rank **par a par** entre os grupos, com correção de Bonferroni.

    Quando o teste global rejeita, esta tabela diz **quais** pares diferem: é o
    argumento para fundir dois segmentos cuja curva não se distingue."""
    grupos = panel.segments() if by == panel.segment_col else sorted(
        pd.unique(panel.df[by].dropna()).tolist())
    linhas = []
    pares = [(a, b) for i, a in enumerate(grupos) for b in grupos[i + 1:]]
    for a, b in pares:
        sub = panel.df[panel.df[by].isin([a, b])]
        p_sub = ContractPanel(sub.reset_index(drop=True), id_col=panel.id_col,
                              date_col=panel.date_col, default_col=panel.default_col,
                              age_col=panel.age_col, segment_col=by, freq=panel.freq,
                              drop_post_default=False)
        try:
            r = logrank_test(p_sub, by, from_age=from_age, horizon=horizon, weights=weights)
            linhas.append({"grupo_a": a, "grupo_b": b, "estatistica": r["estatistica"],
                           "p_valor": r["p_valor"]})
        except ValueError:
            linhas.append({"grupo_a": a, "grupo_b": b, "estatistica": np.nan, "p_valor": np.nan})
    out = pd.DataFrame(linhas)
    if len(out):
        out["p_bonferroni"] = np.clip(out["p_valor"] * len(pares), 0.0, 1.0)
    return out


# ======================================================================
# Leituras em tempo
# ======================================================================
def median_survival(curve: PDCurve) -> float:
    """Primeiro horizonte em que a sobrevivência cai a 50% (NaN se não chega lá
    dentro do horizonte da curva)."""
    s = curve.survival().to_numpy()
    idx = np.flatnonzero(s <= 0.5)
    return float(idx[0] + 1) if idx.size else float("nan")


def restricted_mean_survival(curve: PDCurve, tau: Optional[int] = None) -> float:
    """Tempo médio de sobrevivência **restrito** a ``tau`` períodos.

    ``E[min(T, τ)] = Σ_{t=0}^{τ−1} S(t)`` com ``S(0) = 1``: quantos períodos, em
    média, um contrato passa vivo dentro do horizonte. Em unidades de período
    da curva."""
    s = curve.survival().to_numpy()
    n = len(s) if tau is None else int(min(max(tau, 1), len(s)))
    return float(1.0 + s[: n - 1].sum())


def smooth_hazard(hazard, window: int = 3, weights=None) -> np.ndarray:
    """Média móvel **centrada** do *hazard* (ponderada, se ``weights`` vier).

    Serve para ler a forma da maturação quando a curva bruta é serrilhada por
    base pequena; não substitui o ajuste paramétrico para extrapolar."""
    h = np.asarray(hazard, dtype=float)
    w = np.ones_like(h) if weights is None else np.asarray(weights, dtype=float)
    k = max(int(window), 1)
    meio = k // 2
    out = np.empty_like(h)
    for i in range(len(h)):
        lo, hi = max(0, i - meio), min(len(h), i + meio + 1)
        peso = w[lo:hi]
        out[i] = (np.sum(h[lo:hi] * peso) / peso.sum()) if peso.sum() > 0 else h[i]
    return np.clip(out, 0.0, 1.0)


__all__ = ["life_table", "logrank_test", "pairwise_logrank", "median_survival",
           "restricted_mean_survival", "smooth_hazard", "LOGRANK_WEIGHTS", "GLOBAL"]
