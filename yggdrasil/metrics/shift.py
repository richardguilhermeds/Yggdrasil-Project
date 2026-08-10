"""Métricas por amostra e cálculo de *shifts* entre DES e OOT.

O *shift* mede a degradação (ou ganho) de uma métrica entre a amostra de
desenvolvimento e a *out-of-time*, atendendo ao requisito de acompanhar
deslocamentos de KS, AUC, RMSE, MAE etc. no experimento. Cada shift ganha
também uma *flag* de degradação (``{m}_shift_flag``) que interpreta a variação
na direção ruim da métrica via :data:`HIGHER_IS_BETTER`.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from ..data import analysis_samples_present
from .classification import HIGHER_IS_BETTER as _HIB_CLASSIFICATION
from .classification import classification_metrics, ks_optimal_cutoff
from .regression import HIGHER_IS_BETTER as _HIB_REGRESSION
from .regression import regression_metrics
from .uncertainty import bootstrap_metric_ci

# Direção "boa" de cada métrica (classificação + regressão): ``True`` = maior é
# melhor (AUC, KS, R²...), ``False`` = menor é melhor (RMSE, Brier...) e
# ``None`` = viés, avaliado pela magnitude ``|valor|`` (ideal perto de zero).
HIGHER_IS_BETTER: Dict[str, Optional[bool]] = {
    **_HIB_CLASSIFICATION,
    **_HIB_REGRESSION,
}

# Limiares default das flags: degradação relativa na direção ruim da métrica.
FLAG_ATENCAO = 0.10    # > 10% de queda relativa => 'atencao'
FLAG_DEGRADADO = 0.20  # > 20% de queda relativa => 'degradado'


def compute_metrics(
    y_true,
    y_score,
    problem_type: str,
    cutoff: Optional[float] = None,
) -> Dict[str, float]:
    """Despacha para o pacote de métricas conforme o tipo de problema."""
    if problem_type == "classification":
        return classification_metrics(y_true, y_score, cutoff=cutoff)
    if problem_type == "regression":
        return regression_metrics(y_true, y_score)
    raise ValueError(f"problem_type inválido: {problem_type!r}")


# Métricas que ganham IC bootstrap em ``metric_by_sample(with_ci=True)``.
_CI_METRICS = {"classification": ("auc", "gini", "ks"), "regression": ("r2",)}
# Sufixos das colunas de IC anexadas — ignorados por ``metric_shifts``.
_CI_SUFFIXES = ("_ic_low", "_ic_high", "_se")


def metric_by_sample(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    problem_type: str,
    *,
    with_ci: bool = False,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Calcula métricas para cada amostra de análise presente.

    Para classificação, o limiar KS-ótimo é estimado **na amostra de
    desenvolvimento** e reaplicado às demais, garantindo comparabilidade
    de Acurácia/F1 entre DES e OOT.

    Com ``with_ci=True`` (aditivo — o default não muda nada), anexa colunas
    de IC bootstrap (:func:`~yggdrasil.metrics.uncertainty.bootstrap_metric_ci`)
    às métricas de discriminação — ``{m}_ic_low``/``{m}_ic_high``/``{m}_se``
    para AUC/Gini/KS (classificação) ou R² (regressão). ``n_boot``, ``alpha``
    e ``seed`` são repassados ao bootstrap.
    """
    resultado: Dict[str, Dict[str, float]] = {}
    cutoff = None
    if problem_type == "classification":
        dev = df[df[cfg.sample_col] == cfg.dev_sample]
        if len(dev):
            cutoff = ks_optimal_cutoff(dev[cfg.target_col], dev[cfg.score_col])

    for amostra in analysis_samples_present(df, cfg):
        sub = df[df[cfg.sample_col] == amostra]
        if len(sub) == 0:
            continue
        resultado[amostra] = compute_metrics(
            sub[cfg.target_col], sub[cfg.score_col], problem_type, cutoff=cutoff
        )
        if with_ci:
            for m in _CI_METRICS.get(problem_type, ()):
                ci = bootstrap_metric_ci(
                    sub[cfg.target_col], sub[cfg.score_col], metric=m,
                    n_boot=n_boot, alpha=alpha, seed=seed,
                )
                resultado[amostra][f"{m}_ic_low"] = ci["ic_low"]
                resultado[amostra][f"{m}_ic_high"] = ci["ic_high"]
                resultado[amostra][f"{m}_se"] = ci["se"]
    return resultado


def shift_flag(
    metric: str,
    ref: float,
    cmp: float,
    atencao: float = FLAG_ATENCAO,
    degradado: float = FLAG_DEGRADADO,
) -> Optional[str]:
    """Classifica o shift de uma métrica em ``'ok'``/``'atencao'``/``'degradado'``.

    A degradação é a variação relativa na direção *ruim* da métrica, segundo
    :data:`HIGHER_IS_BETTER`:

    * maior-é-melhor (AUC, KS, R²...): queda ``(ref - cmp) / |ref|``;
    * menor-é-melhor (RMSE, Brier...): aumento ``(cmp - ref) / |ref|``;
    * viés (``mean_bias``): crescimento da magnitude ``(|cmp| - |ref|) / |ref|``.

    Devolve ``None`` quando a direção da métrica é desconhecida ou a degradação
    relativa não é computável (referência zero ou valores não finitos).
    """
    if metric not in HIGHER_IS_BETTER:
        return None
    ref, cmp = float(ref), float(cmp)
    if not (np.isfinite(ref) and np.isfinite(cmp)) or ref == 0:
        return None
    sentido = HIGHER_IS_BETTER[metric]
    if sentido is True:
        degradacao = (ref - cmp) / abs(ref)
    elif sentido is False:
        degradacao = (cmp - ref) / abs(ref)
    else:  # viés: importa o afastamento de zero, não o sinal
        degradacao = (abs(cmp) - abs(ref)) / abs(ref)
    if degradacao > degradado:
        return "degradado"
    if degradacao > atencao:
        return "atencao"
    return "ok"


def metric_shifts(
    metrics_ref: Dict[str, float],
    metrics_cmp: Dict[str, float],
    *,
    flag_atencao: float = FLAG_ATENCAO,
    flag_degradado: float = FLAG_DEGRADADO,
) -> Dict[str, Union[float, str]]:
    """Shifts absoluto/relativo e flag de degradação de ``ref`` (DES) para ``cmp`` (OOT).

    ``{m}_shift_abs = cmp - ref`` e ``{m}_shift_rel = (cmp - ref) / |ref|``.
    ``{m}_shift_flag`` classifica a degradação (:func:`shift_flag`) conforme os
    limiares ``flag_atencao``/``flag_degradado``.
    """
    shifts: Dict[str, Union[float, str]] = {}
    for m, ref in metrics_ref.items():
        if m == "ks_cutoff":  # corte não é métrica de performance
            continue
        if m.endswith(_CI_SUFFIXES):  # colunas de IC não são métricas
            continue
        cmp = metrics_cmp.get(m, np.nan)
        if not (np.isfinite(ref) and np.isfinite(cmp)):
            continue
        shifts[f"{m}_shift_abs"] = round(float(cmp - ref), 6)
        shifts[f"{m}_shift_rel"] = (
            round(float((cmp - ref) / abs(ref)), 6) if ref != 0 else float("nan")
        )
        flag = shift_flag(m, ref, cmp, atencao=flag_atencao, degradado=flag_degradado)
        if flag is not None:
            shifts[f"{m}_shift_flag"] = flag
    return shifts


def sample_shifts(
    metrics_by_sample: Dict[str, Dict[str, float]],
    cfg: ColumnConfig,
    *,
    flag_atencao: float = FLAG_ATENCAO,
    flag_degradado: float = FLAG_DEGRADADO,
) -> Dict[str, Union[float, str]]:
    """Atalho: shifts entre as amostras dev e OOT de um dict por amostra."""
    ref = metrics_by_sample.get(cfg.dev_sample)
    cmp = metrics_by_sample.get(cfg.oot_sample)
    if ref is None or cmp is None:
        return {}
    return metric_shifts(
        ref, cmp, flag_atencao=flag_atencao, flag_degradado=flag_degradado
    )


def shift_significance(
    y_ref,
    score_ref,
    y_cmp,
    score_cmp,
    metric: Union[str, Callable] = "auc",
    *,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    higher_is_better: Optional[bool] = None,
) -> Dict[str, Union[float, str, None]]:
    """Significância estatística do shift de uma métrica entre duas amostras.

    Complementa :func:`shift_flag` — que classifica o shift pela magnitude da
    estimativa pontual — com a pergunta que a flag sozinha não responde: a
    queda observada é maior do que o ruído amostral? Compara os ICs bootstrap
    (:func:`~yggdrasil.metrics.uncertainty.bootstrap_metric_ci`) da referência
    (tip. DES) e da comparação (tip. OOT):

    * ICs **disjuntos na direção ruim** da métrica → ``'degradacao_real'``;
    * ICs sobrepostos, ou movimento na direção boa → ``'dentro_do_ruido'``.

    Parameters
    ----------
    y_ref, score_ref:
        Alvo e score da amostra de referência (tip. desenvolvimento).
    y_cmp, score_cmp:
        Alvo e score da amostra de comparação (tip. *out-of-time*).
    metric:
        ``'auc'``/``'gini'``/``'ks'``/``'r2'`` ou callable — repassado ao
        bootstrap.
    n_boot, alpha, seed:
        Parâmetros do bootstrap (``alpha`` também define o nível do IC).
    higher_is_better:
        Direção "boa" da métrica; obrigatório para callable cujo nome não
        esteja em :data:`HIGHER_IS_BETTER`. Default: consulta a tabela.

    Returns
    -------
    dict
        ``valor_ref``/``valor_cmp`` e respectivos ``ic_low``/``ic_high``,
        ``shift_abs``, ``flag`` (:func:`shift_flag` da estimativa pontual;
        ``None`` p/ métrica sem direção conhecida) e ``significancia``
        (``'dentro_do_ruido'`` ou ``'degradacao_real'``).
    """
    nome = metric if isinstance(metric, str) else getattr(metric, "__name__", "")
    if higher_is_better is None:
        higher_is_better = HIGHER_IS_BETTER.get(nome)
    if higher_is_better is None:
        raise ValueError(
            f"Direção da métrica {nome!r} desconhecida — informe "
            "higher_is_better=True/False."
        )

    ci_ref = bootstrap_metric_ci(
        y_ref, score_ref, metric=metric, n_boot=n_boot, alpha=alpha, seed=seed
    )
    ci_cmp = bootstrap_metric_ci(
        y_cmp, score_cmp, metric=metric, n_boot=n_boot, alpha=alpha, seed=seed
    )

    out: Dict[str, Union[float, str, None]] = {
        "metric": nome,
        "valor_ref": ci_ref["valor"],
        "ic_low_ref": ci_ref["ic_low"],
        "ic_high_ref": ci_ref["ic_high"],
        "valor_cmp": ci_cmp["valor"],
        "ic_low_cmp": ci_cmp["ic_low"],
        "ic_high_cmp": ci_cmp["ic_high"],
        "shift_abs": (
            round(float(ci_cmp["valor"] - ci_ref["valor"]), 6)
            if np.isfinite(ci_ref["valor"]) and np.isfinite(ci_cmp["valor"])
            else float("nan")
        ),
        "flag": shift_flag(nome, ci_ref["valor"], ci_cmp["valor"]),
    }

    # Degradação real = IC da comparação inteiro do lado ruim do IC de
    # referência. ICs não computáveis (NaN) contam como sobreposição
    # (veredicto conservador: dentro do ruído).
    limites = (ci_ref["ic_low"], ci_ref["ic_high"],
               ci_cmp["ic_low"], ci_cmp["ic_high"])
    if not all(np.isfinite(v) for v in limites):
        disjunto_ruim = False
    elif higher_is_better:
        disjunto_ruim = ci_cmp["ic_high"] < ci_ref["ic_low"]
    else:
        disjunto_ruim = ci_cmp["ic_low"] > ci_ref["ic_high"]
    out["significancia"] = "degradacao_real" if disjunto_ruim else "dentro_do_ruido"
    return out
