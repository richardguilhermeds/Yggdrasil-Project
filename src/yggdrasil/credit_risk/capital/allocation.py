"""
Alocação de capital, contribuição de risco e RAROC
==================================================
Calculado o capital econômico da carteira total, é preciso **devolvê-lo aos
produtos e segmentos** de forma coerente (Seção 6 do guia): a soma das parcelas
alocadas deve igualar o total, e cada parcela deve refletir a contribuição do
segmento ao **risco conjunto**, não seu risco isolado.

O método padrão é a **alocação de Euler**: a contribuição de cada segmento é a
derivada do capital total em relação à sua exposição, que em simulação se
estima pela **perda média do segmento condicionada aos cenários da cauda** (os
cenários em que a perda total fica próxima do VaR, ou além dele, no caso do ES).
Segmentos que perdem muito justamente nos cenários ruins da carteira recebem
mais capital; segmentos descorrelacionados recebem menos — é assim que o
benefício de diversificação chega ao produto. Com **ES** como métrica, a
alocação de Euler é particularmente estável e bem definida.

Com o capital alocado, fecham-se os usos gerenciais: **RAROC** (resultado
ajustado ao risco ÷ capital econômico alocado) para comparar produtos em base
única, precificação mínima e limites de apetite por consumo de capital.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .monte_carlo import SimulationResult


# ======================================================================
# Alocação de Euler
# ======================================================================
def euler_allocation(
    result: "SimulationResult",
    q: Optional[float] = None,
    metric: str = "es",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Aloca o capital econômico pelos segmentos via **contribuição de Euler**.

    Parameters
    ----------
    result:
        Resultado de :func:`~yggdrasil.credit_risk.capital.monte_carlo.simulate`
        com ``store_segment_losses=True``.
    q:
        Nível de confiança. Se ``None``, usa ``result.q``.
    metric:
        * ``"es"`` (padrão) — contribuição = perda média do segmento nos cenários
          com perda total **≥ VaR** (a cauda). As contribuições somam o ES; é a
          alocação coerente e estável.
        * ``"var"`` — contribuição = perda média do segmento nos cenários com
          perda total **numa faixa** de largura ``alpha`` em torno do VaR. As
          contribuições somam ~VaR.
    alpha:
        Largura (fração de cenários) da faixa em torno do VaR na métrica ``var``.

    Returns
    -------
    pandas.DataFrame
        Colunas: ``segmento``, ``EL``, ``contribuicao_risco`` (contribuição ao
        VaR/ES), ``capital_alocado`` (contribuição − EL), ``capital_isolado``
        (CE do segmento sozinho), ``beneficio_diversificacao`` e
        ``share_capital``. A soma de ``capital_alocado`` iguala o capital
        econômico da carteira (aditividade de Euler).
    """
    if result.segment_losses is None:
        raise ValueError(
            "A alocação de Euler exige as perdas por segmento; rode simulate(..., "
            "store_segment_losses=True).")
    qq = result.q if q is None else float(q)
    if not (0.0 < qq < 1.0):
        raise ValueError(f"q deve estar em (0, 1); recebido {qq!r}.")

    losses = result.losses
    seg = result.segment_losses                       # (n_scenarios, n_seg)
    n = len(losses)
    var = float(np.quantile(losses, qq))

    if metric == "es":
        mask = losses >= var
        if not np.any(mask):                          # degenerado
            mask = losses >= np.quantile(losses, qq - 1e-6)
        contrib = seg[mask, :].mean(axis=0)           # E[L_i | L >= VaR]
    elif metric == "var":
        # Faixa de cenários em torno do quantil (janela de ~alpha·n cenários).
        lo = max(qq - alpha / 2.0, 0.0)
        hi = min(qq + alpha / 2.0, 1.0)
        v_lo, v_hi = np.quantile(losses, [lo, hi])
        mask = (losses >= v_lo) & (losses <= v_hi)
        if not np.any(mask):                          # janela vazia → vizinhança do VaR
            k = max(int(alpha * n), 1)
            idx = np.argsort(np.abs(losses - var))[:k]
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
        contrib = seg[mask, :].mean(axis=0)
    else:
        raise ValueError(f"metric deve ser 'es' ou 'var'; recebido {metric!r}.")

    el_seg = seg.mean(axis=0)                          # EL empírica por segmento
    capital_aloc = contrib - el_seg                    # capital econômico alocado

    # Capital isolado (standalone): CE do segmento como se estivesse sozinho.
    standalone = np.array([
        np.quantile(seg[:, j], qq) - el_seg[j] for j in range(seg.shape[1])
    ])

    total_cap = float(capital_aloc.sum())
    df = pd.DataFrame({
        "segmento": result.segment_names,
        "EL": el_seg,
        "contribuicao_risco": contrib,
        "capital_alocado": capital_aloc,
        "capital_isolado": standalone,
        "beneficio_diversificacao": standalone - capital_aloc,
        "share_capital": capital_aloc / total_cap if total_cap != 0 else np.nan,
    })
    return df


# ======================================================================
# RAROC e precificação ajustada ao risco
# ======================================================================
def raroc(
    receita: float,
    custo: float,
    perda_esperada: float,
    capital: float,
) -> float:
    """RAROC = (receita − custo − perda esperada) ÷ capital econômico alocado.

    Resultado ajustado ao risco por unidade de capital — permite comparar
    produtos e safras em base única. Retorna ``nan`` se o capital for zero.
    """
    if capital == 0:
        return float("nan")
    return (receita - custo - perda_esperada) / capital


def raroc_table(
    allocation: pd.DataFrame,
    receitas: Union[Sequence[float], dict],
    custos: Union[Sequence[float], dict],
    hurdle_rate: Optional[float] = None,
) -> pd.DataFrame:
    """RAROC por segmento a partir da tabela de alocação.

    Parameters
    ----------
    allocation:
        Saída de :func:`euler_allocation` (precisa de ``segmento``, ``EL``,
        ``capital_alocado``).
    receitas, custos:
        Receita e custo por segmento — lista alinhada à ordem de ``allocation``
        ou dicionário ``{segmento: valor}``.
    hurdle_rate:
        Custo de capital exigido pelo acionista. Se fornecido, adiciona
        ``cria_valor`` (RAROC ≥ hurdle) e a ``precificacao_minima`` (receita que
        zera o EVA: ``custo + EL + hurdle·capital``).

    Returns
    -------
    pandas.DataFrame
        ``allocation`` acrescido de ``receita``, ``custo``, ``RAROC`` e, se houver
        ``hurdle_rate``, ``cria_valor`` e ``precificacao_minima``.
    """
    df = allocation.copy()
    segs = df["segmento"].tolist()

    def _align(x):
        if isinstance(x, dict):
            return np.array([x.get(s, 0.0) for s in segs], dtype=float)
        x = np.asarray(list(x), dtype=float)
        if len(x) != len(segs):
            raise ValueError("receitas/custos devem ter um valor por segmento.")
        return x

    rec = _align(receitas)
    cst = _align(custos)
    df["receita"] = rec
    df["custo"] = cst
    cap = df["capital_alocado"].to_numpy(dtype=float)
    el = df["EL"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["RAROC"] = np.where(cap != 0, (rec - cst - el) / cap, np.nan)
    if hurdle_rate is not None:
        df["cria_valor"] = df["RAROC"] >= hurdle_rate
        df["precificacao_minima"] = cst + el + hurdle_rate * cap
    return df


__all__ = ["euler_allocation", "raroc", "raroc_table"]
