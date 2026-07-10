"""
yggdrasil.credit_risk._common
==============================
Helpers PUROS compartilhados entre os segmentadores de **árvore**
(:mod:`yggdrasil.credit_risk.tree`) e de **modelo**
(:mod:`yggdrasil.credit_risk.model`). Fonte ÚNICA para formatação de faixas,
classificação de PSI/IV, contagem de inversões e o ajuste do optbinning — antes
essas funções eram copiadas nos dois módulos e já haviam começado a **divergir**
(guard de NaN no PSI, default de ``task_type`` no IV). Centralizá-las aqui elimina
o drift.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def fmt(x: float) -> str:
    """Formata limites de faixa de forma legível."""
    if x == -np.inf:
        return "-inf"
    if x == np.inf:
        return "inf"
    return f"{x:.4g}"


def classifica_psi(psi) -> str:
    """Classificação usual de PSI para monitoramento de estabilidade.

    ``None``/``NaN`` → ``"—"`` (um PSI indefinido não é 'instável')."""
    if psi is None or (isinstance(psi, float) and np.isnan(psi)):
        return "—"
    if psi < 0.10:
        return "estável"
    if psi < 0.25:
        return "atenção"
    return "instável"


def classifica_iv(iv, task_type: str) -> str:
    """Faixas de força do IV, conforme o tipo de alvo (``task_type`` OBRIGATÓRIO —
    sem default, para nunca classificar IV de regressão pela escala binária por
    engano).

    classification → IV **binário** (WoE/Siddiqi): < 0.02 inútil · 0.02–0.10 fraco
        · 0.10–0.30 médio · 0.30–0.50 forte · ≥ 0.50 suspeito.
    regression → IV **contínuo** (desvio absoluto médio ponderado do alvo por
        faixa, escala menor): < 0.01 inútil · 0.01–0.03 fraco · 0.03–0.10 médio ·
        0.10–0.35 forte · ≥ 0.35 suspeito."""
    if iv is None or (isinstance(iv, float) and np.isnan(iv)):
        return "—"
    faixas = ((0.02, 0.10, 0.30, 0.50) if task_type == "classification"
              else (0.01, 0.03, 0.10, 0.35))
    for lim, rot in zip(faixas, ("inútil", "fraco", "médio", "forte")):
        if iv < lim:
            return rot
    return "suspeito"


def count_inversions(ordered, values) -> tuple:
    """Nº de pares invertidos vs. a ordem de referência e nº de pares comparáveis.

    ``ordered`` = chaves na ordem de risco de referência (crescente); ``values`` =
    dict chave→risco num ponto (amostra/safra). Par (i<j na ref.) inverte quando
    risco_i > risco_j. Pares com valor faltante (NaN) são ignorados."""
    n_inv = n_pairs = 0
    for a in range(len(ordered)):
        va = values.get(ordered[a], float("nan"))
        if pd.isna(va):
            continue
        for b in range(a + 1, len(ordered)):
            vb = values.get(ordered[b], float("nan"))
            if pd.isna(vb):
                continue
            n_pairs += 1
            if va > vb:
                n_inv += 1
    return n_inv, n_pairs


def fmt_safras(safras) -> list:
    """Rótulos de safra → 'mmm/aa' (padrão de mês/ano do repositório). Delega ao
    helper único :func:`yggdrasil.reporting.style.fmt_month_year`."""
    from ..reporting.style import fmt_month_year
    return fmt_month_year(safras)


def fit_optbinning_splits(b, x, y) -> list:
    """Roda ``b.fit(x, y)`` e devolve ``list(b.splits)``.

    Silencia os ``RuntimeWarning`` de "divide by zero" benignos do optbinning
    (em ``auto_monotonic``, quando algum prebin fica com 0 registros) — o ajuste
    ainda produz cortes válidos. Devolve ``[]`` se o ajuste falhar.

    ``ValueError`` (problema inviável / sem corte) é o caminho esperado e fica
    silencioso. Qualquer outra exceção (ex.: incompatibilidade de versão de
    dependência) é **avisada** em vez de mascarada como "sem corte válido"."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(divide="ignore", invalid="ignore"):
                b.fit(x, y)
        return list(b.splits)
    except ValueError:
        return []
    except Exception as e:
        warnings.warn(
            f"optbinning falhou inesperadamente em '{getattr(b, 'name', '?')}': "
            f"{type(e).__name__}: {e}", RuntimeWarning)
        return []
