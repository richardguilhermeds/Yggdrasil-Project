"""Migração de ratings entre períodos — matriz de transição observada.

Acompanha, no nível da **entidade** (``ColumnConfig.id_col``), como os grupos
homogêneos mudam entre dois períodos: duas safras de ``date_col`` ou dois
rótulos de amostra (ex.: desenvolvimento → *out-of-time*). Para as entidades
presentes em ambos os períodos monta-se o *crosstab* rating origem × destino
(contagem e % por linha de origem), com resumo de permanência na diagonal,
*upgrades* e *downgrades* segundo a ordem dos rótulos, e nota sobre entradas e
saídas da base entre os períodos.

Não confundir com :mod:`yggdrasil.credit_risk.capital.migration`, que **simula**
migrações; aqui a matriz é **observada** na base scorada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from ..reporting.style import fmt_month_year


@dataclass
class MigrationResult:
    """Resultado de :func:`rating_migration_matrix`.

    Attributes
    ----------
    counts:
        Crosstab rating no período A (linhas) × rating no período B (colunas),
        em contagem de entidades presentes em ambos os períodos.
    pct:
        Mesma matriz normalizada **por linha de origem** (cada linha soma 100%;
        linhas sem entidade ficam NaN) — convenção de matriz de transição.
    summary:
        ``n_comum``, ``n_saidas``, ``n_entradas``, ``pct_diagonal``,
        ``pct_upgrade`` e ``pct_downgrade`` (percentuais 0–100 sobre as
        entidades em comum).
    nota:
        Texto sobre entradas/saídas da base (entidades fora da matriz).
    period_a, period_b:
        Rótulos resolvidos dos períodos (amostra ou safra em ``mmm/aa``).
    labels:
        Ordem dos rótulos usada nos eixos e na direção de upgrade/downgrade.
    """

    counts: pd.DataFrame
    pct: pd.DataFrame
    summary: Dict[str, float]
    nota: str
    period_a: str
    period_b: str
    labels: List


def _resolve_period(df: pd.DataFrame, cfg: ColumnConfig, period) -> Tuple[pd.Series, str]:
    """Máscara booleana das linhas do período + rótulo de exibição.

    ``period`` pode ser um rótulo de amostra presente em ``sample_col`` (tem
    prioridade — ex.: ``'DES'`` → ``'OOT'``) ou uma data/safra de ``date_col``
    (comparada no nível do mês; ex.: ``'2024-01'``).
    """
    if cfg.sample_col in df.columns:
        mask = df[cfg.sample_col] == period
        if mask.any():
            return mask, str(period)
    if cfg.date_col not in df.columns:
        raise ValueError(
            f"Período {period!r} não é um rótulo de amostra presente em "
            f"'{cfg.sample_col}' e a coluna de data '{cfg.date_col}' não existe "
            f"no DataFrame."
        )
    try:
        alvo = pd.to_datetime(str(period)).to_period("M")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Período {period!r} não é um rótulo de amostra presente em "
            f"'{cfg.sample_col}' nem uma data reconhecível (ex.: '2024-01')."
        ) from exc
    meses = pd.to_datetime(df[cfg.date_col], errors="coerce").dt.to_period("M")
    mask = meses == alvo
    if not mask.any():
        raise ValueError(
            f"Nenhuma linha no período {period!r} (coluna '{cfg.date_col}')."
        )
    return mask, fmt_month_year([alvo])[0]


def _snapshot(
    df: pd.DataFrame, mask: pd.Series, id_col: str, rating_col: str, date_col: str
) -> pd.Series:
    """Rating por entidade dentro do período (última observação no tempo).

    Entidades com rating NaN no período são descartadas (contam como fora da
    base naquele período).
    """
    cols = [id_col, rating_col] + ([date_col] if date_col in df.columns else [])
    sub = df.loc[mask, cols].dropna(subset=[id_col, rating_col])
    if date_col in sub.columns:
        sub = sub.sort_values(date_col, kind="mergesort")
    sub = sub.drop_duplicates(subset=[id_col], keep="last")
    return sub.set_index(id_col)[rating_col]


def rating_migration_matrix(
    df: pd.DataFrame,
    rating_col: str,
    cfg: ColumnConfig,
    period_a,
    period_b,
    *,
    labels: Optional[Sequence] = None,
) -> MigrationResult:
    """Matriz de migração de um rating entre dois períodos, por entidade.

    Para as entidades (``cfg.id_col``) presentes em **ambos** os períodos,
    cruza o rating no período A com o rating no período B (contagem e % por
    linha de origem). Entidades presentes em só um período ficam fora da
    matriz e são reportadas na ``nota`` (saídas/entradas da base).

    Parameters
    ----------
    df:
        Base scorada com ``id_col``, ``rating_col`` e ``sample_col``/``date_col``.
    rating_col:
        Coluna do rating (ex.: ``rating_decis``).
    cfg:
        Configuração de colunas; exige ``id_col`` configurado.
    period_a, period_b:
        Rótulo de amostra (ex.: ``cfg.dev_sample`` → ``cfg.oot_sample``) ou
        data/safra de ``date_col`` (comparação no nível do mês). Rótulo de
        amostra tem prioridade quando ambíguo.
    labels:
        Ordem dos rótulos de rating (default: união observada, ordenada). A
        ordem define a direção: mover-se para um rótulo **anterior** conta como
        *upgrade*; para um posterior, *downgrade*.

    Returns
    -------
    MigrationResult
        Matrizes ``counts``/``pct``, ``summary`` (% diagonal/upgrade/downgrade
        sobre as entidades em comum), ``nota`` e rótulos dos períodos.
    """
    if not cfg.id_col:
        raise ValueError(
            "ColumnConfig.id_col não configurado — defina id_col com a coluna "
            "identificadora da entidade (ex.: ColumnConfig(id_col='id_contrato')) "
            "para calcular a migração de ratings entre períodos."
        )
    for col in (cfg.id_col, rating_col):
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não existe no DataFrame.")

    mask_a, rotulo_a = _resolve_period(df, cfg, period_a)
    mask_b, rotulo_b = _resolve_period(df, cfg, period_b)
    s_a = _snapshot(df, mask_a, cfg.id_col, rating_col, cfg.date_col)
    s_b = _snapshot(df, mask_b, cfg.id_col, rating_col, cfg.date_col)

    comuns = s_a.index.intersection(s_b.index)
    n_saidas = int(len(s_a.index.difference(s_b.index)))
    n_entradas = int(len(s_b.index.difference(s_a.index)))
    if len(comuns) == 0:
        raise ValueError(
            f"Nenhuma entidade em comum entre os períodos {rotulo_a!r} e "
            f"{rotulo_b!r} (coluna '{cfg.id_col}')."
        )
    ra, rb = s_a.loc[comuns], s_b.loc[comuns]

    observados = sorted(set(ra.unique()) | set(rb.unique()))
    if labels is None:
        labels = observados
    else:
        labels = list(labels)
        faltantes = [lab for lab in observados if lab not in labels]
        if faltantes:
            raise ValueError(
                f"Rótulos observados fora de 'labels': {faltantes} — inclua "
                f"todos os rótulos presentes nos dois períodos."
            )

    counts = (
        pd.crosstab(ra, rb)
        .reindex(index=labels, columns=labels, fill_value=0)
        .astype(int)
    )
    counts.index.name = "de"
    counts.columns.name = "para"
    pct = counts.div(counts.sum(axis=1), axis=0).mul(100.0)

    # Direção segundo a ordem dos rótulos: destino antes da origem => upgrade.
    pos = {lab: i for i, lab in enumerate(labels)}
    ia = ra.map(pos).to_numpy(dtype=int)
    ib = rb.map(pos).to_numpy(dtype=int)
    n = int(len(comuns))
    summary: Dict[str, float] = {
        "n_comum": n,
        "n_saidas": n_saidas,
        "n_entradas": n_entradas,
        "pct_diagonal": round(100.0 * float(np.mean(ia == ib)), 4),
        "pct_upgrade": round(100.0 * float(np.mean(ib < ia)), 4),
        "pct_downgrade": round(100.0 * float(np.mean(ib > ia)), 4),
    }
    nota = (
        f"{n} entidade(s) em comum entre {rotulo_a} e {rotulo_b}; "
        f"{n_saidas} presente(s) só em {rotulo_a} (saíram da base) e "
        f"{n_entradas} só em {rotulo_b} (entraram) — fora da matriz."
    )
    return MigrationResult(
        counts=counts,
        pct=pct,
        summary=summary,
        nota=nota,
        period_a=rotulo_a,
        period_b=rotulo_b,
        labels=list(labels),
    )


def plot_migration_matrix(
    result: MigrationResult,
    *,
    values: str = "pct",
    annot: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 110,
    ax=None,
    save_path: Optional[str] = None,
):
    """Heatmap da matriz de migração (padrão visual de ``reporting.style``).

    ``values='pct'`` (default) usa a matriz % por linha de origem; ``'counts'``
    usa a contagem. Linhas de origem sem entidade (NaN em ``pct``) ficam sem
    preenchimento.
    """
    import matplotlib.pyplot as plt

    from ..reporting.style import colormap

    if values not in ("pct", "counts"):
        raise ValueError(f"values inválido: {values!r} (use 'pct' ou 'counts')")
    mat = result.pct if values == "pct" else result.counts
    data = mat.to_numpy(dtype=float)
    n = len(result.labels)

    if ax is None:
        lado = max(4.5, 1.5 + 0.55 * n)
        fig, ax = plt.subplots(figsize=figsize or (lado + 1.2, lado))
    else:
        fig = ax.figure

    vmax = 100.0 if values == "pct" else max(1.0, float(np.nanmax(data)))
    im = ax.imshow(data, cmap=colormap(), vmin=0.0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% da linha de origem" if values == "pct" else "contagem",
                   fontsize=8)

    ax.set_xticks(range(n))
    ax.set_xticklabels(result.labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(result.labels)
    ax.set_xlabel(f"Rating em {result.period_b}")
    ax.set_ylabel(f"Rating em {result.period_a}")
    s = result.summary
    ax.set_title(
        f"Migração de rating — {result.period_a} → {result.period_b}\n"
        f"diagonal {s['pct_diagonal']:.0f}% · upgrade {s['pct_upgrade']:.0f}% · "
        f"downgrade {s['pct_downgrade']:.0f}%",
        fontweight="bold", fontsize=10,
    )

    if annot:
        for i in range(n):
            for j in range(n):
                v = data[i, j]
                if not np.isfinite(v):
                    continue
                txt = f"{v:.0f}%" if values == "pct" else f"{int(v)}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


__all__ = ["MigrationResult", "rating_migration_matrix", "plot_migration_matrix"]
