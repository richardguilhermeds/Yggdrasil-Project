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

from typing import Callable, Dict, List, Optional, Sequence, Union

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


def _roc_por_contagem(y_true: np.ndarray, y_score: np.ndarray) -> Callable:
    """ROC de réplicas bootstrap a partir das CONTAGENS de cada linha.

    Ordena o score uma única vez; cada réplica vira um vetor de multiplicidades
    (quantas vezes cada linha foi sorteada) e a curva sai de somas acumuladas
    nessa ordem fixa, em O(n). O ``roc_curve`` por réplica reordenava as n
    linhas (mais o ``np.unique`` das checagens) a cada réplica e dominava o custo
    em base grande. Os pontos são os mesmos do ``roc_curve`` sobre
    ``(y[idx], score[idx])`` (empates agrupados por valor distinto de score);
    os pontos intermediários colineares que o sklearn descarta não mudam a
    área nem o máximo de TPR − FPR. Devolve ``pack(contagens) -> (auc, gini,
    ks)`` ou ``None`` quando a réplica não tem as duas classes."""
    ordem = np.argsort(-y_score, kind="mergesort")
    s = y_score[ordem]
    fins = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    positivo = y_true[ordem] == 1

    def pack(contagens: np.ndarray):
        c = contagens[ordem]
        tps = np.cumsum(np.where(positivo, c, 0))[fins]
        fps = np.cumsum(c)[fins] - tps
        if tps[-1] == 0 or fps[-1] == 0:
            return None
        tpr = np.r_[0.0, tps / tps[-1]]
        fpr = np.r_[0.0, fps / fps[-1]]
        auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
        ks = float(np.max(tpr - fpr))
        return auc, 2 * auc - 1, ks

    return pack


def bootstrap_metrics_ci(
    y_true,
    y_score,
    metrics: Sequence[Union[str, Callable]] = ("auc",),
    n_boot: int = 200,
    alpha: float = 0.05,
    stratified: bool = True,
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    """IC bootstrap de VÁRIAS métricas sobre as MESMAS réplicas.

    Mesma reamostragem de :func:`bootstrap_metric_ci` (mesma ``seed`` ⇒ mesmos
    índices), avaliando todas as ``metrics`` em cada réplica: AUC, Gini e KS
    saem de uma única curva ROC por réplica, calculada por contagem
    (:func:`_roc_por_contagem`). Devolve uma lista alinhada a ``metrics``, cada
    item no formato ``{'valor', 'ic_low', 'ic_high', 'se'}``."""
    metrics = list(metrics)
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    vazio = {"valor": float("nan"), "ic_low": float("nan"),
             "ic_high": float("nan"), "se": float("nan")}
    if n == 0 or n_boot <= 0:
        return [dict(vazio) for _ in metrics]

    # ponto na amostra completa pelo caminho de referência (valida os nomes)
    valores = [_avaliar_metrica(m, y_true, y_score) for m in metrics]

    # Estratifica apenas em alvo binário {0,1} com as duas classes presentes.
    classes = np.unique(y_true)
    binario = classes.size == 2 and set(classes) <= {0.0, 1.0}
    estratificar = stratified and binario
    if estratificar:
        idx_pos = np.flatnonzero(y_true == 1)
        idx_neg = np.flatnonzero(y_true == 0)

    usa_contagem = [isinstance(m, str) and m in _METRICAS_CLF for m in metrics]
    pack = (_roc_por_contagem(y_true, y_score)
            if any(usa_contagem) and binario and np.isfinite(y_score).all()
            else None)
    rng = np.random.default_rng(seed)
    replicas = np.empty((n_boot, len(metrics)), dtype=float)
    for b in range(n_boot):
        if estratificar:
            idx = np.concatenate([
                rng.choice(idx_pos, size=idx_pos.size, replace=True),
                rng.choice(idx_neg, size=idx_neg.size, replace=True),
            ])
        else:
            idx = rng.integers(0, n, size=n)
        roc = None
        if pack is not None:
            roc = pack(np.bincount(idx, minlength=n))
        yb = sb = None
        for j, m in enumerate(metrics):
            if pack is not None and usa_contagem[j]:
                replicas[b, j] = (float("nan") if roc is None
                                  else roc[_METRICAS_CLF.index(m)])
                continue
            if yb is None:
                yb, sb = y_true[idx], y_score[idx]
            replicas[b, j] = _avaliar_metrica(m, yb, sb)

    def _r(v: float) -> float:
        return round(float(v), 6) if np.isfinite(v) else float("nan")

    out = []
    for j, valor in enumerate(valores):
        validas = replicas[:, j][np.isfinite(replicas[:, j])]
        if validas.size == 0:
            ic_low = ic_high = se = float("nan")
        else:
            ic_low = float(np.percentile(validas, 100 * alpha / 2))
            ic_high = float(np.percentile(validas, 100 * (1 - alpha / 2)))
            se = float(np.std(validas, ddof=1)) if validas.size > 1 else float("nan")
        out.append({"valor": _r(valor), "ic_low": _r(ic_low),
                    "ic_high": _r(ic_high), "se": _r(se)})
    return out


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
        Para várias métricas sobre as mesmas réplicas, use
        :func:`bootstrap_metrics_ci`.
    """
    return bootstrap_metrics_ci(y_true, y_score, metrics=(metric,), n_boot=n_boot,
                                alpha=alpha, stratified=stratified, seed=seed)[0]
