"""Métricas de calibração do previsto vs. observado.

Inclui calibration-in-the-large (razão observado/previsto), slope/intercept de
calibração (regressão logística do observado no logit do previsto), tabela de
confiabilidade por faixa de score e intervalo de confiança binomial (Jeffreys
ou Clopper-Pearson) para taxas observadas.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

# NOTA DE DESEMPENHO: scipy.stats e sklearn são importados **lazy** (dentro das
# funções), não no topo — mesmo padrão de metrics/classification.py. Este módulo
# é puxado por `import yggdrasil`; o import no topo encareceria a 1ª célula do
# notebook sem necessidade.


def _finite_pair(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
    """Converte para arrays float e descarta pares com valor não finito."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    return y[mask], p[mask]


def calibration_in_the_large(y_true, y_pred) -> float:
    """Razão observado/previsto agregada (calibration-in-the-large).

    Valor 1 indica calibração perfeita no agregado; > 1 significa que o modelo
    subestima a média observada, < 1 que superestima. Devolve NaN quando não há
    dados válidos ou a média prevista é zero.
    """
    y, p = _finite_pair(y_true, y_pred)
    if len(y) == 0:
        return float("nan")
    media_prevista = float(np.mean(p))
    if media_prevista == 0:
        return float("nan")
    return float(np.mean(y) / media_prevista)


def calibration_slope_intercept(y_true, y_pred, eps: float = 1e-6) -> Tuple[float, float]:
    """Slope e intercept de calibração (Cox): logística do observado no logit do previsto.

    Ajusta ``y ~ logit(p)`` sem regularização; um modelo bem calibrado tem
    slope ≈ 1 e intercept ≈ 0. ``y_pred`` é clipado a ``[eps, 1-eps]`` antes do
    logit para estabilidade numérica. Requer alvo binário com as duas classes
    presentes — caso contrário devolve ``(nan, nan)``.
    """
    from sklearn.linear_model import LogisticRegression

    y, p = _finite_pair(y_true, y_pred)
    classes = np.unique(y)
    if len(classes) != 2 or not np.isin(classes, (0.0, 1.0)).all():
        return float("nan"), float("nan")

    p = np.clip(p, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit_p.reshape(-1, 1), y.astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def binomial_ci(k, n, alpha: float = 0.05, method: str = "jeffreys"):
    """Intervalo de confiança binomial para a taxa ``k/n``.

    Parameters
    ----------
    k, n:
        Sucessos e tentativas (escalares ou array-like; aceita ``k`` fracionário
        para taxas de alvo contínuo em [0, 1], tratado como pseudo-binomial).
    alpha:
        Nível de significância (0.05 → IC de 95%).
    method:
        ``'jeffreys'`` (padrão, prior Beta(1/2, 1/2)) ou ``'clopper-pearson'``
        (exato). Ambos via ``scipy.stats.beta``.

    Returns
    -------
    Tupla ``(ic_inf, ic_sup)`` — floats para entrada escalar, ``np.ndarray``
    para entrada vetorial. Entradas inválidas (``n <= 0`` ou não finitas)
    devolvem NaN.
    """
    from scipy.stats import beta

    k_arr = np.asarray(k, dtype=float)
    escalar = k_arr.ndim == 0 and np.asarray(n, dtype=float).ndim == 0
    k_arr, n_arr = np.broadcast_arrays(np.atleast_1d(k_arr),
                                       np.atleast_1d(np.asarray(n, dtype=float)))

    inf = np.full(k_arr.shape, np.nan)
    sup = np.full(k_arr.shape, np.nan)
    ok = np.isfinite(k_arr) & np.isfinite(n_arr) & (n_arr > 0)
    kk, nn = k_arr[ok], n_arr[ok]

    if method == "jeffreys":
        a, b = kk + 0.5, nn - kk + 0.5
        inf[ok] = np.where(kk <= 0, 0.0, beta.ppf(alpha / 2, a, b))
        sup[ok] = np.where(kk >= nn, 1.0, beta.ppf(1 - alpha / 2, a, b))
    elif method == "clopper-pearson":
        inf[ok] = np.where(kk <= 0, 0.0, beta.ppf(alpha / 2, kk, nn - kk + 1))
        sup[ok] = np.where(kk >= nn, 1.0, beta.ppf(1 - alpha / 2, kk + 1, nn - kk))
    else:
        raise ValueError(f"method desconhecido: {method!r} "
                         "(use 'jeffreys' ou 'clopper-pearson')")

    inf = np.clip(inf, 0.0, 1.0)
    sup = np.clip(sup, 0.0, 1.0)
    if escalar:
        return float(inf[0]), float(sup[0])
    return inf, sup


def reliability_table(
    y_true,
    y_pred,
    n_bins: int = 10,
    alpha: float = 0.05,
    method: str = "jeffreys",
) -> pd.DataFrame:
    """Tabela de confiabilidade por faixa (quantis) do previsto.

    Para cada faixa devolve volume, média prevista, taxa observada e o IC
    binomial (:func:`binomial_ci`) da taxa observada, além da flag ``calibrado``
    (média prevista dentro do IC). Faixas com previsto constante são fundidas
    (``duplicates='drop'`` no ``qcut``).
    """
    y, p = _finite_pair(y_true, y_pred)
    dados = pd.DataFrame({"y": y, "p": p})
    dados["faixa"] = pd.qcut(dados["p"], q=n_bins, duplicates="drop")

    g = dados.groupby("faixa", observed=True)
    tab = pd.DataFrame(
        {
            "n": g.size(),
            "p_medio": g["p"].mean(),
            "taxa_observada": g["y"].mean(),
        }
    )
    ic_inf, ic_sup = binomial_ci(
        g["y"].sum().to_numpy(dtype=float), tab["n"].to_numpy(dtype=float),
        alpha=alpha, method=method,
    )
    tab["ic_inf"] = ic_inf
    tab["ic_sup"] = ic_sup
    tab["calibrado"] = (tab["p_medio"] >= tab["ic_inf"]) & (tab["p_medio"] <= tab["ic_sup"])
    return tab.reset_index()


__all__ = [
    "calibration_in_the_large",
    "calibration_slope_intercept",
    "binomial_ci",
    "reliability_table",
]
