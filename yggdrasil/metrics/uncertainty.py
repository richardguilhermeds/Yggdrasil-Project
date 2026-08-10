"""Incerteza de métricas via *bootstrap*: IC de AUC/KS/Gini/R² e afins.

Um ponto único de AUC ou KS não diz se a diferença entre duas amostras é
sinal ou ruído amostral. Este módulo estima o intervalo de confiança (IC)
de uma métrica reamostrando os pares ``(y_true, y_score)`` com reposição
(*bootstrap* percentil). Para classificação a reamostragem é **estratificada
por classe** — preserva a proporção de eventos em cada réplica, evitando
réplicas degeneradas (uma classe só) em bases desbalanceadas.

O IC alimenta :func:`yggdrasil.metrics.shift.shift_significance`, que decide
se um *shift* DES→OOT está dentro do ruído ou é degradação real.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Union

import numpy as np

from .classification import _roc_pack

# NOTA DE DESEMPENHO: sklearn é importado lazy (dentro das funções), mesmo
# padrão dos demais módulos de metrics/ — este pacote é puxado por
# `import yggdrasil` e o import no topo encareceria a 1ª célula do notebook.

# Métricas nomeadas suportadas. As de classificação saem todas do mesmo
# `_roc_pack` (uma única ordenação por réplica).
_METRICAS_CLF = ("auc", "gini", "ks")
_METRICAS_REG = ("r2",)


def _avaliar_metrica(
    metric: Union[str, Callable], y_true: np.ndarray, y_score: np.ndarray
) -> float:
    """Avalia a métrica em um par de arrays; ``NaN`` quando não computável."""
    if callable(metric):
        return float(metric(y_true, y_score))
    if metric in _METRICAS_CLF:
        pack = _roc_pack(y_true, y_score)  # roc_curve UMA vez → auc/gini/ks
        if pack is None:  # réplica com uma classe só
            return float("nan")
        auc, gini, ks, _ = pack
        return {"auc": auc, "gini": gini, "ks": ks}[metric]
    if metric == "r2":
        from sklearn.metrics import r2_score
        if len(y_true) < 2 or np.unique(y_true).size < 2:
            return float("nan")  # alvo constante: R² indefinido
        return float(r2_score(y_true, y_score))
    raise ValueError(
        f"metric inválida: {metric!r}. Use "
        f"{_METRICAS_CLF + _METRICAS_REG} ou um callable(y_true, y_score)."
    )


def bootstrap_metric_ci(
    y_true,
    y_score,
    metric: Union[str, Callable] = "auc",
    n_boot: int = 200,
    alpha: float = 0.05,
    stratified: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """IC por *bootstrap* percentil de uma métrica de performance.

    Parameters
    ----------
    y_true, y_score:
        Alvo observado e predição do modelo (array-like, mesmo tamanho).
    metric:
        ``'auc'``, ``'gini'``, ``'ks'``, ``'r2'`` ou um
        ``callable(y_true, y_score) -> float``.
    n_boot:
        Número de réplicas bootstrap.
    alpha:
        Nível de significância — IC de ``100·(1−alpha)%`` (default 95%).
    stratified:
        Reamostra dentro de cada classe quando o alvo é binário {0,1},
        preservando a proporção de eventos por réplica. Ignorado quando o
        alvo não é binário (regressão).
    seed:
        Semente do gerador — mesmo ``seed`` reproduz exatamente o mesmo IC.

    Returns
    -------
    dict
        ``{'valor', 'ic_low', 'ic_high', 'se'}`` — estimativa pontual na
        amostra completa, limites percentis do IC e erro-padrão bootstrap
        (desvio das réplicas). Réplicas não computáveis (ex.: métrica NaN)
        são descartadas; sem réplica válida, IC e ``se`` saem ``NaN``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    if n == 0 or n_boot <= 0:
        return {"valor": float("nan"), "ic_low": float("nan"),
                "ic_high": float("nan"), "se": float("nan")}

    valor = _avaliar_metrica(metric, y_true, y_score)

    # Estratifica apenas em alvo binário {0,1} com as duas classes presentes.
    classes = np.unique(y_true)
    estratificar = (
        stratified and classes.size == 2 and set(classes) <= {0.0, 1.0}
    )
    if estratificar:
        idx_pos = np.flatnonzero(y_true == 1)
        idx_neg = np.flatnonzero(y_true == 0)

    rng = np.random.default_rng(seed)
    replicas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        if estratificar:
            idx = np.concatenate([
                rng.choice(idx_pos, size=idx_pos.size, replace=True),
                rng.choice(idx_neg, size=idx_neg.size, replace=True),
            ])
        else:
            idx = rng.integers(0, n, size=n)
        replicas[b] = _avaliar_metrica(metric, y_true[idx], y_score[idx])

    validas = replicas[np.isfinite(replicas)]
    if validas.size == 0:
        ic_low = ic_high = se = float("nan")
    else:
        ic_low = float(np.percentile(validas, 100 * alpha / 2))
        ic_high = float(np.percentile(validas, 100 * (1 - alpha / 2)))
        se = float(np.std(validas, ddof=1)) if validas.size > 1 else float("nan")

    def _r(v: float) -> float:
        return round(float(v), 6) if np.isfinite(v) else float("nan")

    return {"valor": _r(valor), "ic_low": _r(ic_low),
            "ic_high": _r(ic_high), "se": _r(se)}
