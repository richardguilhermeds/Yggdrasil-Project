"""
yggdrasil.credit_risk._common
==============================
Helpers PUROS compartilhados entre os segmentadores de **árvore**
(:mod:`yggdrasil.credit_risk.tree`) e de **modelo**
(:mod:`yggdrasil.credit_risk.model`). Fonte ÚNICA para formatação de faixas,
a fórmula do PSI (:func:`psi_from_shares`), classificação de PSI/IV, contagem de
inversões e o ajuste do optbinning — antes essas funções eram copiadas nos dois
módulos e já haviam começado a **divergir** (guard de NaN no PSI, default de
``task_type`` no IV). Centralizá-las aqui elimina o drift.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# limiares únicos do repositório (< PSI_STABLE estável · < PSI_SIGNIFICANT
# atenção · acima instável) — import leve: monitoring.psi só usa numpy/pandas.
from ..monitoring.psi import PSI_SIGNIFICANT, PSI_STABLE


def fmt(x: float) -> str:
    """Formata limites de faixa de forma legível."""
    if x == -np.inf:
        return "-inf"
    if x == np.inf:
        return "inf"
    return f"{x:.4g}"


def psi_from_shares(p_ref, p_cur, eps: float = 1e-6, return_contrib: bool = False):
    """PSI entre duas distribuições de participação por faixa (*shares*).

    Fórmula clássica, por faixa: ``(p_cur − p_ref)·ln(p_cur/p_ref)``, com cada
    participação truncada por baixo em ``eps`` (faixa vazia não gera ``log(0)``
    nem divisão por zero — mesmo tratamento de todos os segmentadores). A soma é
    **sequencial na ordem das faixas**, preservando o comportamento numérico dos
    laços originais que esta função substitui.

    Parameters
    ----------
    p_ref, p_cur:
        Sequências de participações (mesmo comprimento e mesma ordem de faixa).
        Valores já truncados em ``eps`` na origem não mudam (``max`` idempotente).
    eps:
        Piso de cada participação antes do log.
    return_contrib:
        Se ``True``, devolve ``(psi_total, contribuicoes)`` com a parcela de cada
        faixa — para tabelas de decomposição (``psi_detalhe``/``csi_detalhe``).

    Returns
    -------
    float | tuple[float, list[float]]
    """
    contribs = []
    for r, c in zip(p_ref, p_cur):
        r = max(float(r), eps)
        c = max(float(c), eps)
        contribs.append(float((c - r) * np.log(c / r)))
    total = float(sum(contribs))
    return (total, contribs) if return_contrib else total


def classifica_psi(psi) -> str:
    """Classificação usual de PSI para monitoramento de estabilidade.

    Limiares únicos do repositório (:mod:`yggdrasil.monitoring.psi`):
    ``< PSI_STABLE`` estável · ``< PSI_SIGNIFICANT`` atenção · acima, instável.
    ``None``/``NaN`` → ``"—"`` (um PSI indefinido não é 'instável')."""
    if psi is None or (isinstance(psi, float) and np.isnan(psi)):
        return "—"
    if psi < PSI_STABLE:
        return "estável"
    if psi < PSI_SIGNIFICANT:
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
