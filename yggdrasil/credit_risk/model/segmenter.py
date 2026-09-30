"""
ModelSegmenter
==============
Segmentador **orientado a modelo** para risco de crédito, unificando
**classificação** e **regressão** num único objeto via o
parâmetro ``task_type``.

Diferente dos irmãos :class:`~yggdrasil.credit_risk.lgd.SequentialLGDSegmenter`
e :class:`~yggdrasil.credit_risk.pd.SequentialPDSegmenter` — que constroem uma
árvore de bins sobre o espaço de features — aqui o fluxo é:

1. **Análise univariada** de cada variável candidata (logodds/WoE, IV, distribuição
   e *inversão* da ordem de risco entre amostras/safras), para **categorizar** e
   **decidir o que entra no modelo** (``include`` / ``exclude`` / ``auto_select``).
2. **Ajuste de um modelo** — Regressão Logística/Linear ou ML (RandomForest,
   ExtraTrees, GradientBoosting, HistGradientBoosting e — via pacotes opcionais —
   LightGBM, XGBoost, CatBoost), treinado na própria interface (``fit``) ou
   recebido pronto (``set_model``). Registry extensível em :data:`ALGORITHMS`.
3. **Métricas** do modelo por amostra (KS/AUC/Gini/Acc/F1 na classificação;
   RMSE/MAE/R² na regressão) e **SHAP** do modelo criado. Opcionalmente, uma
   **calibração pós-treino** do score cru (``calibrate``: intercepto/Platt/
   isotônica) sem re-treinar o modelo.
4. **Score → ratings**: a resposta do modelo é segmentada em faixas homogêneas
   ordenadas, reaproveitando :mod:`yggdrasil.ratings` (decis/quantil/árvore/optbin),
   com o número de ratings escolhido pelo usuário.

Contexto: parâmetros de risco de crédito sob Resolução CMN 4.966/2021 e IFRS 9.
Reaproveita :mod:`yggdrasil.metrics`, :mod:`yggdrasil.ratings` e
:mod:`yggdrasil.interpretability.shap_explain`.
"""
from __future__ import annotations

import json
import threading
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd

from ...config import ColumnConfig
from ...metrics import bootstrap_metrics_ci, classification_metrics, regression_metrics
from ...metrics.shift import HIGHER_IS_BETTER as _HIGHER_IS_BETTER
from ...ratings import RATING_REGISTRY
# helpers puros compartilhados com o TreeSegmenter (fonte única — sem drift)
from .._common import (
    fmt as _fmt,
    fmt_safras as _fmt_safras,
    classifica_psi as _classifica_psi,
    classifica_iv as _classifica_iv,
    count_inversions as _count_inversions,
    fit_optbinning_splits as _fit_optbinning_splits,
    psi_from_shares as _psi_from_shares,
)

try:  # optbinning é dependência core, mas degradamos com elegância.
    from optbinning import ContinuousOptimalBinning, OptimalBinning
except ImportError:  # pragma: no cover
    ContinuousOptimalBinning = OptimalBinning = None

try:  # sklearn é dependência core; degradamos se ausente (só p/ importar o módulo).
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:  # pragma: no cover
    BaseEstimator = TransformerMixin = object

SCHEMA = "yggdrasil.credit_risk.model/1"

#: Algoritmos suportados (registry extensível). Cada entrada indica em quais
#: ``task_type`` é válido, o rótulo amigável para a UI e o ``extra`` de instalação
#: (pacote opcional via ``pip install "yggdrasil[<extra>]"``; ``None`` = só sklearn).
#: Plugar um novo algoritmo é adicionar uma entrada aqui e o ramo correspondente
#: em :func:`_build_estimator`.
_BOTH = ("classification", "regression")
ALGORITHMS: dict[str, dict] = {
    "logistica": {"label": "Regressão Logística", "tasks": ("classification",),
                  "extra": None},
    "linear": {"label": "Regressão Linear", "tasks": ("regression",), "extra": None},
    "random_forest": {"label": "Random Forest", "tasks": _BOTH, "extra": None},
    "extra_trees": {"label": "Extra Trees", "tasks": _BOTH, "extra": None},
    "gradient_boosting": {"label": "Gradient Boosting", "tasks": _BOTH, "extra": None},
    "hist_gradient_boosting": {"label": "Hist Gradient Boosting", "tasks": _BOTH,
                               "extra": None},
    "lightgbm": {"label": "LightGBM", "tasks": _BOTH, "extra": "lgbm"},
    "xgboost": {"label": "XGBoost", "tasks": _BOTH, "extra": "xgboost"},
    "catboost": {"label": "CatBoost", "tasks": _BOTH, "extra": "catboost"},
}

#: Algoritmos de boosting (expõem ``learning_rate`` na UI).
BOOSTING_ALGORITHMS = ("gradient_boosting", "hist_gradient_boosting",
                       "lightgbm", "xgboost", "catboost")

#: Algoritmos com suporte nativo a **restrições de monotonicidade** (ver
#: ``fit(monotone=...)``): ``monotonic_cst`` no HistGradientBoosting (dict por
#: NOME de coluna) e ``monotone_constraints`` no LightGBM/XGBoost (vetor
#: posicional alinhado às colunas pós-transformação). Nos demais algoritmos a
#: opção é ignorada com aviso.
MONOTONE_ALGORITHMS = ("hist_gradient_boosting", "lightgbm", "xgboost")

#: Algoritmos com espaço de busca para tuning bayesiano (Optuna).
TUNABLE_ALGORITHMS = ("logistica", "random_forest", "extra_trees", "gradient_boosting",
                      "hist_gradient_boosting", "lightgbm", "xgboost", "catboost")

#: Hiperparâmetros AVANÇADOS (opcionais) que a UI expõe por algoritmo, além dos
#: básicos (``C``/``n_estimators``/``max_depth``/``learning_rate``). Cada nome é
#: passado diretamente ao estimador em :func:`_build_estimator` — mantê-los 1:1
#: com o parâmetro real do sklearn/boosting evita remapeamentos. Só entram no
#: ``hyperparams`` quando o usuário os habilita explicitamente (ver a UI).
ADVANCED_HYPERPARAMS: dict[str, tuple] = {
    "random_forest": ("min_samples_leaf", "max_features"),
    "extra_trees": ("min_samples_leaf", "max_features"),
    "gradient_boosting": ("min_samples_leaf", "max_features", "subsample"),
    "hist_gradient_boosting": ("min_samples_leaf", "l2_regularization"),
    "lightgbm": ("num_leaves", "subsample", "colsample_bytree", "reg_lambda"),
    "xgboost": ("subsample", "colsample_bytree", "reg_lambda"),
    "catboost": ("subsample", "l2_leaf_reg"),
}

#: Espaço de busca do tuning bayesiano (Optuna) por algoritmo — fonte única de
#: verdade consumida por :func:`_optuna_space` e exposta para edição na UI
#: (``ModelSegmenterUI``: quais hiperparâmetros tunar e seus intervalos). Cada
#: parâmetro traz o ``type`` (``int``/``float``/``categorical``) e a faixa
#: PADRÃO (``low``/``high``; ``step`` p/ inteiros, ``log`` p/ floats em escala
#: logarítmica, ``choices`` p/ categóricos). Os nomes são 1:1 com o parâmetro
#: real do estimador (ver :func:`_build_estimator`).
OPTUNA_SEARCH_SPACE: dict[str, dict[str, dict]] = {
    "logistica": {
        "C": {"type": "float", "low": 1e-3, "high": 1e2, "log": True},
    },
    "random_forest": {
        "n_estimators": {"type": "int", "low": 100, "high": 600, "step": 50},
        "max_depth": {"type": "int", "low": 3, "high": 16},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 80},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
    },
    "extra_trees": {
        "n_estimators": {"type": "int", "low": 100, "high": 600, "step": 50},
        "max_depth": {"type": "int", "low": 3, "high": 16},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 80},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
    },
    "gradient_boosting": {
        "n_estimators": {"type": "int", "low": 100, "high": 600, "step": 50},
        "max_depth": {"type": "int", "low": 2, "high": 6},
        "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    },
    "hist_gradient_boosting": {
        "max_iter": {"type": "int", "low": 100, "high": 600, "step": 50},
        "max_depth": {"type": "int", "low": 2, "high": 12},
        "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
        "l2_regularization": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    },
    "lightgbm": {
        "n_estimators": {"type": "int", "low": 100, "high": 800, "step": 50},
        "num_leaves": {"type": "int", "low": 15, "high": 255},
        "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
    },
    "xgboost": {
        "n_estimators": {"type": "int", "low": 100, "high": 800, "step": 50},
        "max_depth": {"type": "int", "low": 2, "high": 12},
        "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
    },
    "catboost": {
        "iterations": {"type": "int", "low": 100, "high": 800, "step": 50},
        "depth": {"type": "int", "low": 2, "high": 10},
        "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
    },
}

_EPS = 1e-6


def _sem_prefixo_da_variavel(labels, feature):
    """Rótulos de bin sem o prefixo ``"<feature>: "``.

    O nome da variável já aparece no TÍTULO do gráfico; repeti-lo em cada item
    da legenda só rouba espaço e faz o rótulo ser truncado no meio do número
    (ex.: "comprometimento_renda: (0.3287, 0.56…"). Sobra a faixa, que é o que
    distingue as séries."""
    pref = f"{feature}: "
    return [l[len(pref):] if isinstance(l, str) and l.startswith(pref) else l
            for l in labels]


def _optuna_space(trial, algorithm: str, space: dict | None = None) -> dict:
    """Sugere um conjunto de hiperparâmetros para o Optuna, a partir do catálogo
    :data:`OPTUNA_SEARCH_SPACE` do algoritmo.

    ``space`` (opcional): sobrescreve o catálogo — dict ``{nome: {type, low,
    high, log?, step?, choices?}}`` (ex.: o que a UI monta a partir dos limites
    escolhidos). Só os parâmetros presentes em ``space`` **e** válidos para o
    algoritmo são sugeridos; os ausentes ficam no default do estimador. ``space``
    vazio (ou sem interseção) cai de volta no catálogo padrão."""
    base = OPTUNA_SEARCH_SPACE.get(algorithm)
    if base is None:
        raise ValueError(
            f"O algoritmo {algorithm!r} não tem espaço de tuning. "
            f"Tunáveis: {TUNABLE_ALGORITHMS}.")
    if not space:
        space = base
    else:                                   # só nomes válidos p/ o algoritmo
        space = {k: v for k, v in space.items() if k in base}
        if not space:
            space = base
    out = {}
    for name, spec in space.items():
        t = spec.get("type")
        if t == "int":
            step = int(spec.get("step") or 1)
            out[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]),
                                          step=step)
        elif t == "categorical":
            out[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:                               # float (log opcional)
            out[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]),
                                            log=bool(spec.get("log", False)))
    return out


# ======================================================================
# Helpers de módulo
# ======================================================================
# _fmt / _classifica_psi / _classifica_iv vêm de credit_risk._common (import acima)


def _trend(values) -> tuple:
    """(tendência, nº de inversões) de uma sequência de valores de risco.

    tendência ∈ {crescente, decrescente, não-monotônica}; nº de inversões =
    mudanças de sinal nas diferenças consecutivas."""
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                      dtype="float64")
    if vals.size < 2:
        return "—", 0
    diffs = np.diff(vals)
    if (diffs >= 0).all():
        trend = "crescente"
    elif (diffs <= 0).all():
        trend = "decrescente"
    else:
        trend = "não-monotônica"
    # inversões = mudanças de sinal APENAS entre diferenças não-nulas: um platô
    # (empate entre pontos adjacentes, np.sign(0)==0) não é uma inversão e não deve
    # contradizer a tendência nem o _count_inversions (que usa '>' estrito).
    nz = diffs[diffs != 0]
    n_inv = int((np.sign(nz[:-1]) != np.sign(nz[1:])).sum()) if nz.size > 1 else 0
    return trend, n_inv


# _count_inversions vem de credit_risk._common (import acima)


def _inverted_pairs(ordered, values) -> list:
    """Pares ``(a, b)`` cuja ordem de risco inverte vs. a referência: ``a`` vem
    ANTES de ``b`` em ``ordered`` (risco crescente na referência), mas tem risco
    MAIOR que ``b`` em ``values`` (dict rótulo→risco). Ex.: a régua diz C < D na
    DES, mas na OOT C > D → devolve ``[("C", "D")]``."""
    pairs = []
    for a in range(len(ordered)):
        va = values.get(ordered[a], float("nan"))
        if pd.isna(va):
            continue
        for b in range(a + 1, len(ordered)):
            vb = values.get(ordered[b], float("nan"))
            if pd.isna(vb):
                continue
            if va > vb:
                pairs.append((ordered[a], ordered[b]))
    return pairs


# _fit_optbinning_splits vem de credit_risk._common (import acima)


def _new_ax(figsize, dpi, ax):
    """Figura SEM pyplot (não entra no Gcf) — evita o backend inline re-exibir."""
    if ax is not None:
        return ax.figure, ax
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    return fig, fig.subplots()


def _pct_axis(ax, axis="y", xmax=1.0):
    """Formata o(s) eixo(s) como percentual — só exibição, não altera os dados.
    Use ``xmax=1.0`` quando os valores estão em fração [0,1] (risco/score) e
    ``xmax=100`` quando já estão em 0-100 (ex.: % da amostra)."""
    from matplotlib.ticker import PercentFormatter
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=xmax, decimals=None))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=xmax, decimals=None))


def _fit_labels_x(fig, ax, texts, pad_frac=0.04, passes=2):
    """Alarga os limites do eixo-x até que todos os ``texts`` (rótulos no fim de
    barras horizontais) caibam dentro da área de plotagem. O texto tem largura
    fixa em pixels; com um ``span`` pequeno ele estoura o ``xlim`` — mede a
    extensão real de cada rótulo e amplia os limites (2 passadas convergem).
    Best-effort: nunca derruba o plot."""
    if not texts:
        return
    try:
        for _ in range(passes):
            fig.canvas.draw()
            rnd = fig.canvas.get_renderer()
            inv = ax.transData.inverted()
            x0, x1 = ax.get_xlim()
            nx0, nx1 = x0, x1
            for t in texts:
                bb = t.get_window_extent(renderer=rnd)
                nx0 = min(nx0, inv.transform((bb.x0, 0))[0])
                nx1 = max(nx1, inv.transform((bb.x1, 0))[0])
            pad = pad_frac * (nx1 - nx0)
            new0, new1 = nx0 - pad, nx1 + pad
            if abs(new0 - x0) < 1e-9 and abs(new1 - x1) < 1e-9:
                break
            ax.set_xlim(new0, new1)
    except Exception:  # noqa: BLE001 - ajuste cosmético; nunca derruba o plot
        pass


#: Pontos exibidos nas nuvens de dispersão (calibração/resíduos da regressão).
_MAX_PONTOS_DISPERSAO = 50_000
#: Pontos mais extremos desenhados em destaque, por ponta e por eixo, quando a
#: nuvem é amostrada.
_EXTREMOS_POR_PONTA = 500


def _pontos_dispersao(n: int, seed, *eixos) -> tuple:
    """Pontos de uma nuvem de dispersão em duas camadas: ``(amostra, extremos)``.

    Até :data:`_MAX_PONTOS_DISPERSAO` pontos: ``(slice(None), [])``, a nuvem
    inteira. Acima disso, ``amostra`` é uma amostra UNIFORME fixa (``seed``),
    fiel à densidade por construção, e ``extremos`` são os
    :data:`_EXTREMOS_POR_PONTA` maiores e menores valores de cada eixo de
    ``eixos`` (empates no corte sorteados, para não depender da ordem das
    linhas; um empate maior que o necessário é massa, não extremo, e fica só
    na amostra), que o gráfico desenha numa camada à parte, com legenda
    própria (:func:`_legenda_nuvem`).

    Com milhões de pontos o ``scatter`` levava ~10 s por gráfico e cada Figure
    guardada pela UI (troca de tema) retinha ~50 MB; só a amostra, porém,
    sumia com os resíduos extremos (LGD acima de 100%, por exemplo), que são o
    que a validação procura. Misturar os extremos à amostra na mesma camada
    cria degraus de densidade (blocos e vazios que não existem nos dados); em
    camada separada o extremo aparece sem se passar por densidade. Curvas,
    bandas e cobertura seguem calculadas em todas as observações."""
    if n <= _MAX_PONTOS_DISPERSAO:
        return slice(None), np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    amostra = np.sort(rng.choice(n, _MAX_PONTOS_DISPERSAO, replace=False))
    extremos = []
    for v in eixos:
        v = np.asarray(v, dtype="float64")
        finitos = np.flatnonzero(np.isfinite(v))
        m = min(_EXTREMOS_POR_PONTA, len(finitos))
        if not m:
            continue
        vf = v[finitos]
        for w in (vf, -vf):                      # maiores e menores
            corte = np.partition(w, len(w) - m)[len(w) - m]
            acima = np.flatnonzero(w > corte)
            iguais = np.flatnonzero(w == corte)
            if len(iguais) > m:
                # empate maciço no corte (grade de rating, zeros da LGD, modelo
                # constante): é massa da distribuição, não extremo; a amostra
                # uniforme já a mostra
                extremos.append(finitos[acima])
                continue
            sorteio = rng.choice(iguais, m - len(acima), replace=False)
            extremos.append(finitos[np.concatenate([acima, sorteio])])
    ext = np.unique(np.concatenate(extremos)) if extremos else np.array([], dtype=int)
    return amostra, ext


def _desenha_nuvem(ax, x, y, amostra, extremos, n, **kw) -> bool:
    """Desenha a nuvem de :func:`_pontos_dispersao`: a amostra com o estilo de
    ``kw`` e, quando amostrada, os extremos em destaque, ambos rotulados para
    a legenda (que o gráfico monta no fim). Devolve se a nuvem foi amostrada."""
    if isinstance(amostra, slice):             # nuvem inteira, sem amostragem
        ax.scatter(x[amostra], y[amostra], edgecolors="none", **kw)
        return False
    fmt = lambda v: f"{v:,}".replace(",", ".")  # noqa: E731
    ax.scatter(x[amostra], y[amostra], edgecolors="none",
               label=f"amostra uniforme: {fmt(len(amostra))} de {fmt(n)}", **kw)
    if len(extremos):
        ax.scatter(x[extremos], y[extremos], s=12, alpha=0.75, color="#e08a2b",
                   edgecolors="none", zorder=kw.get("zorder", 1) + 0.5,
                   label=f"extremos: {_EXTREMOS_POR_PONTA} maiores e menores por eixo")
    return True


def _legenda_nuvem(ax) -> None:
    """Legenda da nuvem amostrada ABAIXO do eixo x, fora da área de dados: no
    canto ela cobria justamente os resíduos extremos (em LGD/CCF os mais
    negativos ficam no canto inferior direito). As chaves saem opacas; com o
    alpha da nuvem (0,16) a da amostra ficava invisível."""
    leg = ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                    ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.2)
    for h in (getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])):
        h.set_alpha(1.0)


def _is_stability_sample(name) -> bool:
    """Heurística: a *safra de estabilidade* é a amostra cujo nome remete a
    estabilidade (ex.: ``ESTABILIDADE``, ``ESTAB``) — convenção do repositório
    (ver PSI por rating: DES × OOT e ESTABILIDADE)."""
    return "estab" in str(name).strip().lower()


def _jeffreys_ci(k, n, conf=0.95) -> tuple:
    """Intervalo de credibilidade de **Jeffreys** para uma taxa binomial:
    quantis equiláteros da posteriori Beta(k+½, n−k+½) (priori de Jeffreys).
    Convenção usual (Brown–Cai–DasGupta): com ``k=0`` o limite inferior é 0 e
    com ``k=n`` o superior é 1. Devolve ``(lo, hi)``; ``(nan, nan)`` sem
    observações válidas."""
    from scipy.stats import beta as _beta_dist
    n = int(n)
    if n <= 0 or k is None or not np.isfinite(k):
        return (float("nan"), float("nan"))
    k = float(min(max(k, 0.0), n))
    a = 0.5 * (1.0 - float(conf))
    lo = 0.0 if k <= 0 else float(_beta_dist.ppf(a, k + 0.5, n - k + 0.5))
    hi = 1.0 if k >= n else float(_beta_dist.ppf(1.0 - a, k + 0.5, n - k + 0.5))
    return (lo, hi)


# _fmt_safras vem de credit_risk._common (import acima)

# Métricas limitadas a [0,1] (discriminação/classificação): os eixos que só as
# contêm devem ficar FIXOS em 0–1, para leitura estável e comparável — a autoescala
# "dá zoom" e exagera variações pequenas. As fora de [0,1] (regressão: rmse/mae ·
# logloss · r2, que pode ser negativo) mantêm autoescala.
_UNIT_METRICS = frozenset({"ks", "auc", "gini", "accuracy", "f1", "precision",
                           "recall", "brier"})


def _emit_progress(cb, key: str, label: str, status: str, detail: str = "") -> None:
    """Dispara um evento de progresso (escoragem) para a UI, se houver callback.

    ``status``: ``"run"`` (iniciando a etapa), ``"ok"`` (concluída) ou ``"err"``.
    Nunca derruba a ação — o progresso é cosmético (mesma política do tuning)."""
    if cb is None:
        return
    try:
        cb(key, label, status, detail)
    except Exception:  # noqa: BLE001 - progresso é cosmético
        pass


def _require(module: str, algorithm: str):
    """Importa um pacote opcional (LightGBM/XGBoost/CatBoost) com mensagem de
    instalação amigável quando ausente."""
    import importlib
    extra = ALGORITHMS.get(algorithm, {}).get("extra") or module
    try:
        return importlib.import_module(module)
    except ImportError as e:  # pragma: no cover - depende do ambiente
        raise ImportError(
            f"O algoritmo {algorithm!r} requer o pacote opcional '{module}'. "
            f"Instale com: pip install \"yggdrasil[{extra}]\"  (ou pip install {module})."
        ) from e


def _balanced_sample_weight(y) -> np.ndarray:
    """Pesos amostrais no esquema ``class_weight='balanced'`` do sklearn —
    ``w_c = n / (k · n_c)`` — para algoritmos SEM ``class_weight`` no construtor
    (GradientBoosting clássico): o chamador passa o resultado como
    ``sample_weight`` no ``fit``."""
    arr = np.asarray(y, dtype="float64")
    classes, counts = np.unique(arr, return_counts=True)
    w = {c: arr.size / (classes.size * n) for c, n in zip(classes, counts)}
    return np.asarray([w[v] for v in arr], dtype="float64")


def _build_estimator(algorithm: str, task_type: str, hyperparams: dict | None,
                     random_state: int | None = None, class_counts=None):
    """Instancia o estimador do algoritmo escolhido (registry extensível).

    sklearn é sempre disponível; LightGBM/XGBoost/CatBoost são pacotes opcionais
    importados sob demanda (ver :func:`_require`).

    ``random_state`` semeia os estimadores estocásticos (florestas/boosting) para
    reprodutibilidade; ``None`` cai no default histórico 42. Via ``setdefault``, uma
    seed explícita em ``hyperparams`` (usuário/Optuna) vence. Logística/linear não
    aceitam seed (não são estocásticas); CatBoost usa ``random_seed``.

    ``class_counts`` (só classificação): tupla ``(n_neg, n_pos)`` da amostra de
    treino — quando informada, liga o **balanceamento de classes**, traduzido para
    o parâmetro próprio de cada algoritmo (``setdefault``: um valor explícito em
    ``hyperparams`` vence): ``class_weight='balanced'`` (logística, florestas,
    HistGB), ``scale_pos_weight=n_neg/n_pos`` (XGBoost, LightGBM) e
    ``auto_class_weights='Balanced'`` (CatBoost). O GradientBoosting clássico não
    tem ``class_weight`` no construtor — o chamador aplica ``sample_weight``
    balanceado no ``fit`` (ver :func:`_balanced_sample_weight`)."""
    hp = dict(hyperparams or {})
    seed = 42 if random_state is None else int(random_state)
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algoritmo desconhecido: {algorithm!r}. "
                         f"Opções: {sorted(ALGORITHMS)}")
    if task_type not in ALGORITHMS[algorithm]["tasks"]:
        raise ValueError(
            f"Algoritmo {algorithm!r} não suporta task_type={task_type!r} "
            f"(suporta {ALGORITHMS[algorithm]['tasks']}).")
    is_clf = task_type == "classification"
    balance = is_clf and class_counts is not None
    spw = 1.0
    if balance:
        n_neg, n_pos = int(class_counts[0]), int(class_counts[1])
        spw = n_neg / max(n_pos, 1)

    if algorithm == "logistica":
        from sklearn.linear_model import LogisticRegression
        hp.setdefault("max_iter", 1000)
        if balance:
            hp.setdefault("class_weight", "balanced")
        return LogisticRegression(**hp)
    if algorithm == "linear":
        from sklearn.linear_model import LinearRegression
        # o X que chega aqui é a matriz TEMPORÁRIA do pré-processador do pipeline:
        # centralizar no lugar evita uma cópia do tamanho da matriz de desenho
        hp.setdefault("copy_X", False)
        return LinearRegression(**hp)
    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        RF = RandomForestClassifier if is_clf else RandomForestRegressor
        hp.setdefault("n_estimators", 200)
        hp.setdefault("random_state", seed)
        if balance:
            hp.setdefault("class_weight", "balanced")
        return RF(**hp)
    if algorithm == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        ET = ExtraTreesClassifier if is_clf else ExtraTreesRegressor
        hp.setdefault("n_estimators", 200)
        hp.setdefault("random_state", seed)
        if balance:
            hp.setdefault("class_weight", "balanced")
        return ET(**hp)
    if algorithm == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        GB = GradientBoostingClassifier if is_clf else GradientBoostingRegressor
        hp.setdefault("random_state", seed)
        # sem class_weight no construtor: o balanceamento é via sample_weight no
        # fit (responsabilidade do chamador — ver _balanced_sample_weight).
        return GB(**hp)
    if algorithm == "hist_gradient_boosting":
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      HistGradientBoostingRegressor)
        HGB = HistGradientBoostingClassifier if is_clf else HistGradientBoostingRegressor
        if "n_estimators" in hp:                       # nome unificado na UI → max_iter
            hp["max_iter"] = hp.pop("n_estimators")
        hp.setdefault("random_state", seed)
        if balance:                                    # class_weight no sklearn >= 1.2
            hp.setdefault("class_weight", "balanced")
        return HGB(**hp)
    if algorithm == "lightgbm":
        lgb = _require("lightgbm", algorithm)
        Est = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor
        hp.setdefault("n_estimators", 300)
        hp.setdefault("learning_rate", 0.05)
        hp.setdefault("random_state", seed)
        hp.setdefault("verbose", -1)
        if balance:
            hp.setdefault("scale_pos_weight", spw)
        return Est(**hp)
    if algorithm == "xgboost":
        xgb = _require("xgboost", algorithm)
        Est = xgb.XGBClassifier if is_clf else xgb.XGBRegressor
        hp.setdefault("n_estimators", 300)
        hp.setdefault("learning_rate", 0.05)
        hp.setdefault("random_state", seed)
        hp.setdefault("verbosity", 0)
        hp.setdefault("tree_method", "hist")
        if balance:
            hp.setdefault("scale_pos_weight", spw)
        return Est(**hp)
    if algorithm == "catboost":
        cb = _require("catboost", algorithm)
        Est = cb.CatBoostClassifier if is_clf else cb.CatBoostRegressor
        if "n_estimators" in hp:                        # nomes próprios do CatBoost
            hp["iterations"] = hp.pop("n_estimators")
        if "max_depth" in hp:
            hp["depth"] = min(int(hp.pop("max_depth")), 16)  # teto do CatBoost
        if "subsample" in hp:            # subsample só vale com bootstrap amostral
            hp.setdefault("bootstrap_type", "Bernoulli")
        if balance:
            hp.setdefault("auto_class_weights", "Balanced")
        hp.setdefault("iterations", 300)
        hp.setdefault("learning_rate", 0.05)
        hp.setdefault("random_seed", seed)
        hp.setdefault("verbose", False)
        hp.setdefault("allow_writing_files", False)     # não polui o diretório
        return Est(**hp)
    raise ValueError(algorithm)  # pragma: no cover


def _decimal_columns(df: pd.DataFrame, cols) -> list:
    """Colunas de ``cols`` (por posição: nomes podem se repetir) cujo 1º valor
    não nulo é ``decimal.Decimal``, que é como o ``toPandas()`` do Spark entrega
    ``DecimalType``. Só detecta: a conversão fica com quem monta a base, no
    Spark (``.cast("double")``), para valer igual no treino e na escoragem
    distribuída, que recebe os ``Decimal`` crus."""
    import decimal

    alvo = set(cols)
    achadas = []
    for j in range(df.shape[1]):
        if df.columns[j] not in alvo:
            continue
        col = df.iloc[:, j]
        if col.dtype != object:
            continue
        validos = np.flatnonzero(col.notna().to_numpy())
        if len(validos) and isinstance(col.iloc[int(validos[0])], decimal.Decimal):
            achadas.append(df.columns[j])
    return list(dict.fromkeys(achadas))


def _cgroup_free_bytes(raiz: str = "/sys/fs/cgroup") -> int | None:
    """Folga até o limite de memória do cgroup (container), descontando todo o
    cache de arquivo (``active_file`` + ``inactive_file``), que o kernel
    recupera antes de matar o processo: a conta erra para o lado de deixar o
    fit rodar, não de recusá-lo. ``None`` sem limite configurado ou fora de
    Linux."""
    for lim_p, uso_p, stat_p, chaves in (
            (f"{raiz}/memory.max", f"{raiz}/memory.current",
             f"{raiz}/memory.stat", ("active_file", "inactive_file")),
            (f"{raiz}/memory/memory.limit_in_bytes",
             f"{raiz}/memory/memory.usage_in_bytes",
             f"{raiz}/memory/memory.stat",
             ("total_active_file", "total_inactive_file"))):
        try:
            with open(lim_p) as fh:
                lim = fh.read().strip()
            with open(uso_p) as fh:
                uso = int(fh.read().strip())
        except (OSError, ValueError):
            continue
        if lim == "max" or int(lim) >= 1 << 60:      # v2 "max" / v1 sem limite
            return None
        cache = 0
        try:
            with open(stat_p) as fh:
                for linha in fh:
                    nome, _, valor = linha.partition(" ")
                    if nome in chaves:
                        cache += int(valor)
        except (OSError, ValueError):
            pass
        return max(int(lim) - (uso - cache), 0)
    return None


def _available_memory_bytes() -> int | None:
    """RAM disponível agora: ``MemAvailable`` do Linux (``psutil`` se houver),
    limitada pela folga do cgroup quando o processo roda num container com
    limite de memória. ``None`` quando não dá para medir (a checagem de
    memória vira no-op)."""
    livre = None
    try:
        with open("/proc/meminfo") as fh:
            for linha in fh:
                if linha.startswith("MemAvailable:"):
                    livre = int(linha.split()[1]) * 1024
                    break
    except OSError:
        pass
    if livre is None:
        try:
            import psutil
            livre = int(psutil.virtual_memory().available)
        except Exception:  # noqa: BLE001 (sem psutil ou plataforma sem suporte)
            livre = None
    cg = _cgroup_free_bytes()
    if cg is not None:
        livre = cg if livre is None else min(livre, cg)
    return livre


def _make_ohe():
    """OneHotEncoder denso e robusto a versões do sklearn."""
    from sklearn.preprocessing import OneHotEncoder
    try:  # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _bin_mask_series(series: pd.Series, b: dict) -> pd.Series:
    """Máscara das linhas que caem no bin ``b`` (faltante / faixa numérica /
    grupo categórico). Espelha ``ModelSegmenter._mask_in`` para uso fora da classe
    (no transformador serializável).

    ``b["include_na"]`` (categorização manual com faltantes alocados numa faixa —
    ver :meth:`ModelSegmenter.set_missing_bin`): a faixa também recebe os NaN."""
    if b["kind"] == "na":
        return series.isna()
    if b["kind"] == "num":
        m = series.between(b["lo"], b["hi"], inclusive="right")
    else:
        m = pd.Series(_cat_group_masks(series, [b["cats"]])[0], index=series.index)
    return (m | series.isna()) if b.get("include_na") else m


def _cat_group_masks(series: pd.Series, grupos) -> list:
    """Máscaras (numpy bool) de grupos categóricos — mesma regra de
    ``series.astype(str).isin(grupo)`` (compara como TEXTO; faltante nunca casa),
    mas via ``pd.factorize``: o ``str()`` roda só nos valores DISTINTOS, não em
    cada linha. Em coluna de texto com milhões de linhas o ``astype(str)`` por
    linha era o custo dominante de tabela/IV/PSI/WoE das categóricas."""
    codes, uniq = pd.factorize(series, use_na_sentinel=True)     # NA → -1
    ustr = np.array([str(u) for u in uniq], dtype=object)
    validos = codes >= 0
    out = []
    for cats in grupos:
        sel = np.isin(ustr, np.asarray(list(cats), dtype=object))
        out.append(np.where(validos, sel[np.where(validos, codes, 0)], False)
                   if len(ustr) else np.zeros(len(codes), dtype=bool))
    return out


def _bin_masks(series: pd.Series, bins) -> list:
    """Máscaras (numpy) de todos os ``bins`` sobre ``series``, com UMA
    fatoração por chamada para os grupos categóricos (ver
    :func:`_cat_group_masks`) em vez de uma por bin. Mesmas máscaras de
    :func:`_bin_mask_series`, com ``<NA>`` como fora do bin: em colunas
    nullable/pyarrow (``Float64``, ``Int64``, ``float64[pyarrow]``) o ``between``
    devolve NA nos faltantes, e sem isso a soma das máscaras virava ``pd.NA``
    (PSI NaN)."""
    cat_idx = [i for i, b in enumerate(bins) if b["kind"] == "cat"]
    cat_m = (dict(zip(cat_idx, _cat_group_masks(series, [bins[i]["cats"] for i in cat_idx])))
             if cat_idx else {})
    na = (series.isna().to_numpy(dtype=bool, na_value=True)
          if any(b.get("include_na") for b in bins) else None)
    # numéricas: a coluna vira float64 UMA vez e cada faixa é comparação numpy
    # (NaN nunca casa com (lo, hi]) — antes, um between() do pandas por faixa
    x = None
    if any(b["kind"] == "num" for b in bins):
        try:
            x = series.to_numpy(dtype="float64", na_value=np.nan)
        except (TypeError, ValueError):
            x = None
    out = []
    for i, b in enumerate(bins):
        if b["kind"] == "cat":
            m = cat_m[i]
            out.append((m | na) if b.get("include_na") else m)
        elif b["kind"] == "num" and x is not None:
            with np.errstate(invalid="ignore"):
                m = (x > b["lo"]) & (x <= b["hi"])
            out.append((m | na) if b.get("include_na") else m)
        else:
            out.append(_bin_mask_series(series, b).to_numpy(dtype=bool, na_value=False))
    return out


def _spearman_pairwise(frame: pd.DataFrame) -> pd.DataFrame:
    """Spearman par a par com a MESMA regra de ``DataFrame.corr("spearman")``
    (linhas completas do par, ranks médios nos empates), mais rápido: colunas
    sem NaN são ranqueadas UMA vez e correlacionadas numa operação de matriz; só
    os pares que envolvem NaN são re-ranqueados no recorte completo do par."""
    from scipy.stats import rankdata
    cols = list(frame.columns)
    X = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    k = len(cols)
    out = np.eye(k)
    tem_nan = np.isnan(X).any(axis=0)

    def _pearson(a, b):
        if a.size < 2:
            return np.nan
        a = a - a.mean(); b = b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den > 0 else np.nan

    completas = [i for i in range(k) if not tem_nan[i]]
    if len(completas) >= 2:
        R = np.column_stack([rankdata(X[:, i]) for i in completas])
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(R, rowvar=False)
        for a, i in enumerate(completas):
            for b, j in enumerate(completas):
                if i != j:
                    out[i, j] = C[a, b]
    # pares com NaN: ranks médios no recorte completo do par, SEM re-ordenar —
    # a ordem de cada coluna é calculada uma vez e só filtrada pelo recorte
    ordens = {}

    def _rank_no_recorte(i, ok):
        if i not in ordens:
            ordens[i] = np.argsort(X[:, i], kind="mergesort")   # NaN vão ao fim
        o = ordens[i]
        o = o[ok[o]]                                    # linhas válidas, já ordenadas
        v = X[o, i]
        n = v.size
        if n == 0:
            return np.empty(0)
        novo = np.r_[True, v[1:] != v[:-1]]             # início de cada grupo de empate
        grupo = np.cumsum(novo) - 1
        inicio = np.flatnonzero(novo)
        tam = np.diff(np.r_[inicio, n])
        media = inicio + (tam + 1) / 2.0                # rank médio (1-based) do grupo
        r = np.empty(len(X))
        r[o] = media[grupo]
        return r[ok]

    for i in range(k):
        for j in range(i + 1, k):
            if not (tem_nan[i] or tem_nan[j]):
                continue
            ok = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
            v = _pearson(_rank_no_recorte(i, ok), _rank_no_recorte(j, ok))
            out[i, j] = out[j, i] = v
    # coluna constante: o pandas devolve NaN (desvio zero) — mantém
    for i in range(k):
        v = X[:, i][~np.isnan(X[:, i])]
        if v.size and np.all(v == v[0]):
            out[i, :] = np.nan; out[:, i] = np.nan
            out[i, i] = np.nan
    return pd.DataFrame(out, index=cols, columns=cols)


def _bin_codes(series: pd.Series, bins) -> np.ndarray:
    """Índice da faixa de cada linha (``-1`` = fora de todas), na regra da 1ª faixa
    que casa — a mesma de :class:`WoeBinEncoder`. Base das contagens vetorizadas
    por safra/amostra (``np.bincount``) no lugar de uma máscara por faixa × safra."""
    rapido = _bin_codes_numericos(series, bins)
    if rapido is not None:
        return rapido
    codes = np.full(len(series), -1, dtype=np.int32)
    for i, m in enumerate(_bin_masks(series, bins)):
        codes[m & (codes < 0)] = i
    return codes


def _bin_codes_numericos(series: pd.Series, bins):
    """Caminho rápido de :func:`_bin_codes` para faixas NUMÉRICAS contíguas
    ``(-inf, c1], (c1, c2], ..., (ck, inf]`` (+ faixa de faltante e/ou
    ``include_na``): uma única busca binária (``np.searchsorted``) sobre os cortes
    no lugar de uma máscara booleana por faixa. ``None`` quando as faixas não têm
    esse formato (grupos categóricos, faixas manuais com buraco) — aí vale o
    caminho geral, com o mesmo resultado."""
    num = [(i, b) for i, b in enumerate(bins) if b["kind"] == "num"]
    outros = [(i, b) for i, b in enumerate(bins) if b["kind"] != "num"]
    if not num or any(b["kind"] != "na" for _i, b in outros):
        return None
    if [i for i, _b in num] != list(range(len(num))):     # num primeiro, em ordem
        return None
    los = [b["lo"] for _i, b in num]
    his = [b["hi"] for _i, b in num]
    if los[0] != -np.inf or his[-1] != np.inf or any(
            los[k] != his[k - 1] for k in range(1, len(num))):
        return None
    try:
        x = series.to_numpy(dtype="float64", na_value=np.nan)
    except (TypeError, ValueError):
        return None
    codes = np.searchsorted(np.asarray(his[:-1], dtype="float64"), x,
                            side="left").astype(np.int32)   # x <= hi_k ⇒ faixa k
    nan = np.isnan(x)
    alvo_na = next((i for i, b in num if b.get("include_na")), None)
    if alvo_na is None:
        alvo_na = next((i for i, _b in outros), -1)          # faixa "(faltante)" ou fora
    codes[nan] = alvo_na
    return codes


class WoeBinEncoder(BaseEstimator, TransformerMixin):
    """Transforma cada variável no **valor do seu bin** — WoE (classificação) ou
    risco médio do bin (regressão) — usando bins/grupos já ajustados na amostra de
    referência (faixas para contínuas, grupos para categóricas, como nas árvores
    de alvo). Serve para alimentar os modelos com variáveis transformadas, no
    estilo *scorecard*.

    ``encodings``: ``{feature: {"kind", "bins": [(bin_dict, valor), ...],
    "fallback": float}}``. Valores fora de qualquer bin (categoria nova, faltante
    sem bin próprio) recebem ``fallback`` (0 = WoE neutro).

    ``prefixes`` (opcional): ``{feature: prefixo}`` que sobrepõe ``name_prefix``
    no nome de saída — ex.: ``ord`` para as variáveis em codificação ordinal de
    scorecard."""

    def __init__(self, encodings=None, features=None, name_prefix="WoE", prefixes=None):
        self.encodings = encodings
        self.features = features
        self.name_prefix = name_prefix
        self.prefixes = prefixes

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        feats = self.features or []
        pref = self.prefixes or {}
        return np.asarray([f"{pref.get(f, self.name_prefix)}({f})" for f in feats],
                          dtype=object)

    def transform(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        feats = self.features or []
        out = np.empty((len(X), len(feats)), dtype="float64")
        for j, f in enumerate(feats):
            enc = self.encodings[f]
            # índice da faixa (1ª que casa) numa passada + consulta do valor
            codes = _bin_codes(X[f], [b for b, _v in enc["bins"]])
            tabela = np.asarray([float(v) for _b, v in enc["bins"]] + [enc["fallback"]],
                                dtype="float64")
            out[:, j] = tabela[np.where(codes >= 0, codes, len(tabela) - 1)]
        return out


class ScorecardDummyEncoder(BaseEstimator, TransformerMixin):
    """Dummies de *scorecard*: cada faixa (bins já ajustados na referência) vira
    uma coluna 0/1, **exceto a referência** (pior faixa), que é omitida — com alvo
    1 = mau os coeficientes medem o quanto cada faixa é melhor que a pior.

    ``specs``: ``{feature: {"bins": [...], "labels": [...], "ref": int}}``. Valor
    fora de qualquer faixa (categoria nova) fica 0 em todas = referência (pior),
    conservador. Nomes de saída: ``"<feature>=<faixa>"``."""

    def __init__(self, specs=None, features=None):
        self.specs = specs
        self.features = features

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        nomes = []
        for f in self.features or []:
            sp = self.specs[f]
            nomes += [f"{f}={lbl}" for i, lbl in enumerate(sp["labels"]) if i != sp["ref"]]
        return np.asarray(nomes, dtype=object)

    def transform(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        cols = []
        for f in self.features or []:
            sp = self.specs[f]
            codes = _bin_codes(X[f], sp["bins"])     # 1ª faixa que casa (como no WoE)
            for i in range(len(sp["bins"])):
                if i != sp["ref"]:
                    cols.append((codes == i).astype("float64"))
        return (np.column_stack(cols) if cols
                else np.empty((len(X), 0), dtype="float64"))


class _TwoStageModel:
    """Modelo *hurdle* de duas etapas para regressão (típico de alvo).

    Combina um **classificador** que estima ``P(y ≥ threshold)`` com uma
    **regressão** treinada apenas no grupo ``y ≥ threshold``; a resposta final é
    o valor esperado

        E[y | x] = P(≥t|x)·reg(x) + (1 − P(≥t|x))·âncora₀,

    onde ``âncora₀`` é a média observada do grupo abaixo do threshold (≈ 0 em
    alvo). Expõe ``predict`` (a resposta combinada) para se comportar como
    qualquer estimador de regressão no restante do pipeline (``score_``, ratings,
    backtest, escoragem). Definido no nível do módulo para ser *picklable*
    (joblib) junto dos dois sub-pipelines em :meth:`ModelSegmenter.save`."""

    def __init__(self, clf, reg, threshold: float, anchor0: float):
        self.clf = clf
        self.reg = reg
        self.threshold = float(threshold)
        self.anchor0 = float(anchor0)

    def proba(self, X) -> np.ndarray:
        """P(y ≥ threshold | x) da etapa de classificação."""
        p = np.asarray(self.clf.predict_proba(X))
        return p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else np.ravel(p)

    def reg_predict(self, X) -> np.ndarray:
        """Previsão crua da etapa de regressão (treinada em y ≥ threshold)."""
        return np.ravel(self.reg.predict(X))

    def predict(self, X) -> np.ndarray:
        """Resposta combinada E[y|x] = p·reg(x) + (1−p)·âncora₀."""
        p = self.proba(X)
        return p * self.reg_predict(X) + (1.0 - p) * self.anchor0


def _logit_np(p, eps=1e-12) -> np.ndarray:
    """Logito seguro: clipa ``p`` a ``[eps, 1−eps]`` antes de ``log(p/(1−p))`` —
    evita ±inf com probabilidades 0/1 na camada de calibração."""
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid_np(z) -> np.ndarray:
    """Sigmoide numericamente estável (sem overflow do ``exp`` em |z| grande)."""
    z = np.asarray(z, dtype="float64")
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# ======================================================================
# Classe principal
# ======================================================================
def _scorer_broadcast_getter(spark, scorer):
    """Getter de zero-args que entrega o ``scorer`` aos executores Spark.

    Usa ``spark.sparkContext.broadcast`` quando disponível (cluster clássico —
    evita reenviar o modelo por task). Em **Spark Connect** (Databricks
    serverless/shared, DBR 14.3+, ou Databricks Connect) a sessão NÃO expõe
    ``sparkContext`` e o acesso/``.broadcast`` levanta ``PySparkAttributeError``;
    nesse caso captura o scorer por **closure** (o ``mapInPandas`` serializa a
    função 1× para os executores). O getter da via broadcast captura só o objeto
    ``Broadcast`` (não o modelo), preservando a economia de rede."""
    try:
        bc = spark.sparkContext.broadcast(scorer)
        return lambda: bc.value
    except Exception:                      # Spark Connect: sem sparkContext/broadcast
        return lambda: scorer


class ModelSegmenter:
    """Segmentador orientado a modelo (classificação **ou** regressão).

    Notas de memória (bases grandes): o score da base inteira, o VIF e os
    p-valores de Wald passam pela matriz de desenho em blocos de
    :attr:`_CHUNK_ROWS` linhas; o ``fit`` recusa (``MemoryError``) uma matriz
    one-hot densa que não caberia na RAM livre, em vez de derrubar o processo.

    Parameters
    ----------
    df:
        Tabela com alvo, features e (opcionalmente) amostra e data de referência.
    target:
        Coluna com a variável resposta (binária na classificação; contínua na
        regressão).
    task_type:
        ``"classification"`` ou ``"regression"`` — chave que unifica o comportamento.
    sample_col, ref_sample:
        Coluna de amostra (DES/OOT/…) e a amostra de referência (desenvolvimento).
    feature_labels:
        Rótulos amigáveis por variável (exibição).
    features:
        Restringe as variáveis candidatas (default: todas que não são alvo/amostra/data).
    date_col:
        Coluna de data/safra (fora da modelagem; usada nas análises temporais).
    """

    #: Linhas por bloco nas passagens em lote sobre a base (escoragem, VIF,
    #: p-valores): o pico da matriz de desenho densa fica em ~bloco × colunas em
    #: vez de linhas × colunas. Bases menores que isso rodam num bloco só.
    _CHUNK_ROWS = 250_000

    def __init__(
        self,
        df: pd.DataFrame,
        target: str = "target",
        task_type: str = "classification",
        sample_col: str | None = None,
        ref_sample: str = "DES",
        feature_labels: dict[str, str] | None = None,
        problem_label: str | None = None,
        features: list | None = None,
        date_col: str | None = None,
        verbose: bool = True,
        score_scale: float = 1000.0,
        random_state: int | None = 42,
    ):
        if task_type not in ("classification", "regression"):
            raise ValueError("task_type deve ser 'classification' ou 'regression'.")
        if features is not None:          # gerador/iterador: lido uma vez só
            features = list(features)
        if target not in df.columns:
            raise ValueError(f"Alvo '{target}' não está no DataFrame.")

        self.df = df.copy()
        # Decimal (DecimalType do toPandas) no alvo/candidatas é lido como
        # CATEGÓRICO: cada valor distinto vira um nível do one-hot denso (é o
        # que derruba o driver em base grande). Avisa em vez de converter: a
        # conversão implícita aqui não chegaria à escoragem (Spark/predict
        # recebem os Decimal crus) nem aos modelos já salvos.
        conv = ([target, *features] if features is not None else
                [c for c in self.df.columns if c not in (sample_col, date_col)])
        self.decimal_cols_: list = _decimal_columns(self.df, conv)
        if self.decimal_cols_:
            warnings.warn(
                f"Colunas com decimal.Decimal (DecimalType do Spark): "
                f"{self.decimal_cols_}. Elas serão tratadas como CATEGÓRICAS (um "
                "nível por valor distinto no one-hot). Se forem numéricas, "
                "converta na origem, ex.: sdf.withColumn(c, F.col(c).cast('double')) "
                "antes do toPandas(), para treino e escoragem receberem o mesmo tipo.",
                stacklevel=2)
        # caches de performance (memoização): binning ótimo por variável (caro —
        # solver CP-SAT do optbinning) e máscara de linhas por amostra. O cache de
        # bins é invalidado SÓ NA VARIÁVEL editada em set/clear_manual_bins e nas
        # derivadas em clear_derived; a máscara é invariante (linhas e sample_col
        # não mudam após a construção).
        self._bins_cache: dict = {}
        self._mask_cache: dict = {}
        # cache do RANKING caro de variable_iv (binning+IV+PSI por variável). A
        # parte mutável barata (incluida/categoria/motivo) é reanexada a cada
        # chamada, então include/exclude/set_category NÃO recomputam o ranking;
        # _rank_version sobe só quando bins/derivadas/amostra mudam de fato.
        self._rank_cache: dict = {}
        self._rank_version: int = 0
        # lista de amostras (constante após a construção) memoizada — _samples()
        # era recalculado (dropna().unique()) dezenas de vezes por clique.
        self._samples_cache: list | None = None
        # cache das métricas do modelo por identidade do score_ (muda só em
        # fit/set_model); evita recomputar AUC/KS/ROC por amostra em cada render +
        # metric_shifts no mesmo clique.
        self._metrics_cache: tuple | None = None
        # cache dos ICs bootstrap (metrics_ci) — mesmo padrão por identidade do
        # score_, com um dict interno por parâmetros (n_boot/métricas/alpha/seed):
        # o bootstrap custa segundos e é pedido 2× no mesmo clique (tabela da UI +
        # qualificação do shift em metric_shifts).
        self._metrics_ci_cache: tuple | None = None
        self.target = target
        self.task_type = task_type
        self.sample_col = sample_col
        self.ref_sample = ref_sample
        self.date_col = date_col
        self.feature_labels = feature_labels or {}
        # rótulo do alvo nos gráficos/relatórios: `problem_label` se informado,
        # senão o nome da coluna alvo (nunca rótulo fixo).
        self.problem_label = problem_label
        self._risk_word = problem_label or target

        if date_col is not None and date_col not in self.df.columns:
            raise ValueError(f"Coluna de data '{date_col}' não está no DataFrame.")
        if sample_col is not None:
            if sample_col not in self.df.columns:
                raise ValueError(f"Coluna de amostra '{sample_col}' não está no DataFrame.")
            amostras = self.df[sample_col].dropna().unique().tolist()
            if ref_sample not in amostras:
                raise ValueError(
                    f"Amostra de referência '{ref_sample}' não encontrada em "
                    f"'{sample_col}'. Disponíveis: {amostras}")
            if verbose:
                print(f"[init] amostras: {amostras} | referência = {ref_sample} "
                      f"| task_type = {task_type}")

        # variáveis candidatas e estado de seleção/categorização
        self.candidates: list = (list(features) if features is not None
                                 else [c for c in self.df.columns
                                       if c not in self._nonfeature_cols()])
        self.included: set = set(self.candidates)      # começa com todas; usuário poda
        self.var_meta: dict[str, dict] = {c: {"categoria": None} for c in self.candidates}
        # última esteira de seleção (ver :meth:`select_features`): o resultado
        # completo vive só nesta sessão (memória) e a POLÍTICA — etapas +
        # parâmetros efetivos, tudo JSON — persiste em to_dict/save, para que a
        # seleção possa ser reproduzida a partir de um modelo salvo.
        self.selection_ = None
        self.selection_policy_: dict | None = None

        # estado de modelo / score / rating
        self.model = None
        self.algorithm: str | None = None
        self.hyperparams: dict = {}
        self.feature_transform: str = "raw"   # "raw" | "woe" (binagem + WoE/risco do bin)
        # balanceamento de classes do último fit (só classificação; ver fit) —
        # traduzido por algoritmo em _build_estimator.
        self.class_balance: bool = False
        # restrições de monotonicidade do último fit: a ESCOLHA ('auto' | dict |
        # None) e as direções efetivamente aplicadas {variável: ±1} (ver
        # fit(monotone=...) e MONOTONE_ALGORITHMS).
        self.monotone = None
        self.monotone_dirs_: dict = {}
        self.model_features: list = []
        # modelo Two-Stage (hurdle de alvo): classificação P(y≥t) + regressão em
        # y≥t, combinadas em E[y]. Desligado por padrão; ligado por fit_two_stage.
        self.two_stage: bool = False
        self.two_stage_threshold: float | None = None
        # escala de negócio do score: o modelo produz uma predição crua (prob. em
        # [0,1] na classificação; alvo previsto na regressão) guardada em ``score_``
        # — usada pelas MÉTRICAS, calibração e ratings (fidelidade numérica). O que
        # o negócio consome (escoragem :meth:`predict`/:meth:`assign` e os eixos de
        # score dos gráficos) é ``score_ * score_scale`` — 0–1000 por padrão.
        self.score_scale: float = float(score_scale)
        # seed global de reprodutibilidade na elaboração de modelos: default 42
        # (retrocompatível — números já publicados não mudam). Herdada como default
        # por _build_pipeline/_build_estimator, tune_optuna, backward_elimination e
        # SHAP; persistida em to_dict/from_dict. None também vira 42.
        self.random_state: int = 42 if random_state is None else int(random_state)
        self.score_: pd.Series | None = None
        # camada de CALIBRAÇÃO pós-treino do score CRU (ver :meth:`calibrate`):
        # dict serializável {method, sample, params, …} aplicado no fluxo único
        # de score (_predict_score_array) — None = sem camada. Um novo
        # fit/set_model a descarta; persiste em to_dict/save e volta no load.
        self.calibration_: dict | None = None
        self.rating_strategy = None
        self.rating_col_: str | None = None
        self.rating_: pd.Series | None = None
        self.rating_labels_: list = []
        self.rating_config: dict = {}
        self._shap_cache: dict = {}
        # snapshots champion × challenger EM MEMÓRIA (ver :meth:`snapshot`):
        # nome → foto {config, métricas por amostra, Series score_}. Vivem só
        # nesta sessão (não entram em to_dict/save) e sobrevivem a re-fits —
        # justamente para comparar o desafiante com o baseline congelado.
        self.snapshots_: dict = {}
        # sinalizador de cancelamento do tuning (Optuna): setado por
        # :meth:`cancel_tuning` e observado por um callback do estudo, que chama
        # ``study.stop()`` para interromper após o trial em andamento.
        self._tuning_cancel = threading.Event()

    # ------------------------------------------------------------------
    # Colunas / kinds / amostras
    # ------------------------------------------------------------------
    def _nonfeature_cols(self) -> set:
        skip = {self.target, self.sample_col, self.date_col}
        skip.discard(None)
        for c in self.df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(self.df[c]):
                    skip.add(c)
            except Exception:
                pass
        return skip

    def _detect_kind(self, feature, sub=None) -> str:
        col = (sub if sub is not None else self.df)[feature]
        if pd.api.types.is_bool_dtype(col):
            return "cat"
        return "num" if pd.api.types.is_numeric_dtype(col) else "cat"

    def _samples(self) -> list:
        """Amostras presentes, referência primeiro (memoizado — invariante após a
        construção; era recomputado dezenas de vezes por clique nos loops de
        tabela/inversão/variável)."""
        if self.sample_col is None:
            return [self.ref_sample]
        if self._samples_cache is None:
            ams = list(self.df[self.sample_col].dropna().unique())
            self._samples_cache = [self.ref_sample] + [a for a in ams if a != self.ref_sample]
        return self._samples_cache

    def _nonref_samples(self) -> list:
        return [a for a in self._samples() if a != self.ref_sample]

    def _oot_sample(self) -> str:
        nr = self._nonref_samples()
        return nr[0] if nr else self.ref_sample

    def _frame(self, sample=None, cols=None) -> pd.DataFrame:
        """Recorte do df por amostra (default DES quando há sample_col).

        Memoiza a máscara booleana por amostra (a comparação numa coluna de objeto
        é cara e se repete centenas de vezes); devolve sempre uma cópia fresca
        ``df[mask]`` — segura contra mutação. Linhas e ``sample_col`` não mudam
        após a construção, então a máscara nunca precisa ser invalidada.

        ``cols`` restringe a cópia às colunas pedidas. Em base grande a cópia da
        amostra INTEIRA (todas as colunas, inclusive as de texto) a cada
        variável binada dominava tempo e memória de variable_iv/fit; os
        caminhos quentes pedem só o que leem (variável + alvo)."""
        if cols is not None:
            cols = list(dict.fromkeys(cols))     # dedup (variável == alvo, etc.)
        if self.sample_col is None:
            return self.df if cols is None else self.df[cols]
        if sample is None:
            sample = self.ref_sample
        mask = self._frame_mask(sample)
        return self.df[mask] if cols is None else self.df.loc[mask, cols]

    def _fit_mask(self, sample=None) -> np.ndarray:
        """Máscara (numpy) das linhas da amostra (default: referência) com alvo
        observado: a base de ajuste de fit/tuning/p-valores/VIF, sem copiar o
        recorte inteiro só para filtrar ``target.notna()``."""
        ok = self.df[self.target].notna().to_numpy()
        if self.sample_col is None:
            return ok
        return self._frame_mask(sample) & ok

    def _row_blocks(self, mask, cols, size=None):
        """Gera ``df.loc[mask, cols]`` em blocos de ``size`` linhas (default
        :attr:`_CHUNK_ROWS`), sem materializar o recorte inteiro. Base das
        passagens em lote (escoragem, VIF, p-valores de Wald)."""
        size = int(size or self._CHUNK_ROWS)
        cols = list(cols)
        pos = np.flatnonzero(mask)
        try:
            cidx = self.df.columns.get_indexer(cols)
            unicas = bool((cidx >= 0).all())
        except Exception:  # noqa: BLE001 (colunas duplicadas no df)
            unicas = False
        for i in range(0, len(pos), size):
            if unicas:
                yield self.df.iloc[pos[i:i + size], cidx]
            else:
                yield self.df.iloc[pos[i:i + size]][cols]

    def _frame_mask(self, sample=None):
        """Máscara booleana (numpy) das linhas da amostra, memoizada (invariante
        após a construção). Reusada por _frame/metrics em vez de recriar a
        comparação `df[sample_col]==a` full-length a cada chamada."""
        if sample is None:
            sample = self.ref_sample
        mask = self._mask_cache.get(sample)
        if mask is None:
            mask = (self.df[self.sample_col] == sample).to_numpy()
            self._mask_cache[sample] = mask
        return mask

    # ---- caches por linha p/ as análises por safra/amostra (vetorizadas) ----
    # Teto de linhas dos GRÁFICOS da aba Análise de variáveis (ver
    # :meth:`_amostra_graficos`): acima disso eles usam uma amostra aleatória fixa.
    # IV, tabela por faixa, PSI do resumo e o ranking seguem na base inteira.
    # ``None``/0 desliga.
    max_linhas_graficos = 300_000

    @contextmanager
    def _amostra_graficos(self):
        """Dentro do bloco, as análises por safra/amostra que alimentam os
        GRÁFICOS (:meth:`_rows_mask`) usam uma amostra aleatória (semente fixa)
        de até :attr:`max_linhas_graficos` linhas — só quando a base é maior que
        isso. Fora do bloco (API, ranking, relatórios) tudo segue na base inteira."""
        anterior = self.__dict__.get("_amostra_graficos_on", False)
        self._amostra_graficos_on = True
        try:
            yield self.amostra_graficos_ativa()
        finally:
            self._amostra_graficos_on = anterior

    def amostra_graficos_ativa(self) -> bool:
        """``True`` quando os gráficos da análise usam amostra (base maior que o teto)."""
        cap = self.max_linhas_graficos
        return bool(cap) and len(self.df) > int(cap)

    def _mascara_amostra_graficos(self):
        """Máscara (memoizada) da amostra aleatória dos gráficos, ou ``None``."""
        if not (self.__dict__.get("_amostra_graficos_on") and self.amostra_graficos_ativa()):
            return None
        cap, n = int(self.max_linhas_graficos), len(self.df)
        hit = self.__dict__.get("_amostra_graficos_cache")
        if hit is None or hit[0] != (cap, n):
            m = np.zeros(n, dtype=bool)
            m[np.random.default_rng(self.random_state).choice(n, cap, replace=False)] = True
            hit = ((cap, n), m)
            self._amostra_graficos_cache = hit
        return hit[1]

    def _rows_mask(self, sample=None, all_rows=False, amostrar=True) -> np.ndarray:
        """Máscara das linhas de ``sample`` (referência por padrão); ``all_rows``
        ou sem ``sample_col`` ⇒ todas — mesmo recorte de :meth:`_frame`. Dentro
        de :meth:`_amostra_graficos`, restrita à amostra dos gráficos (exceto com
        ``amostrar=False`` — números de decisão, como o PSI do ranking)."""
        if all_rows or self.sample_col is None:
            base = np.ones(len(self.df), dtype=bool)
        else:
            base = self._frame_mask(sample)
        amostra = self._mascara_amostra_graficos() if amostrar else None
        return base if amostra is None else (base & amostra)

    def _safra_codes(self, time_col):
        """``(codigos, rotulos)``: índice da safra mensal de cada linha do ``df``
        (``-1`` = data não parseável) e os rótulos ``'AAAA-MM'`` em ordem.
        Memoizado por coluna — o ``to_datetime``/``to_period`` em milhões de
        linhas era refeito em cada gráfico por safra."""
        cache = self.__dict__.setdefault("_safra_cache", {})
        hit = cache.get(time_col)
        if hit is None:
            per = pd.to_datetime(self.df[time_col], errors="coerce").dt.to_period("M")
            codes, uniq = pd.factorize(per, sort=True, use_na_sentinel=True)
            hit = (np.asarray(codes, dtype=np.int32), [str(u) for u in uniq])
            cache[time_col] = hit
        return hit

    def _fatias_por_safra(self, time_col, sample=None, all_rows=False):
        """``(idx, limites, rotulos)``: posições das linhas do recorte ordenadas
        por safra (estável) e os limites de cada safra em ``idx`` — a safra ``k``
        é ``idx[limites[k]:limites[k+1]]``. Memoizado: vale p/ qualquer variável."""
        cache = self.__dict__.setdefault("_fatias_cache", {})
        key = (time_col, sample, bool(all_rows) or self.sample_col is None,
               self._mascara_amostra_graficos() is not None)
        hit = cache.get(key)
        if hit is None:
            cod, rot = self._safra_codes(time_col)
            idx = np.flatnonzero(self._rows_mask(sample, all_rows=all_rows) & (cod >= 0))
            idx = idx[np.argsort(cod[idx], kind="stable")]
            limites = np.searchsorted(cod[idx], np.arange(len(rot) + 1))
            hit = (idx, limites, rot)
            cache[key] = hit
        return hit

    def _feature_bin_codes(self, feature, bins) -> np.ndarray:
        """:func:`_bin_codes` da variável no ``df`` inteiro, memoizado por
        (variável, faixas) — as análises da aba reusam o mesmo vetor."""
        cache = self.__dict__.setdefault("_bincode_cache", {})
        key = (feature, repr(bins))
        hit = cache.get(key)
        if hit is None:
            if len(cache) > 16:
                cache.clear()
            hit = _bin_codes(self.df[feature], bins)
            cache[key] = hit
        return hit

    def _risco_por_grupo(self, bin_codes, grupo_codes, n_grupos, n_bins, linhas):
        """Risco (média do alvo, NaN-safe — igual a :meth:`_risco`) por grupo ×
        faixa numa passada: matriz ``(n_grupos, n_bins)``, NaN onde não há alvo."""
        y = self.df[self.target].to_numpy(dtype="float64")
        ok = linhas & (bin_codes >= 0) & (grupo_codes >= 0) & ~np.isnan(y)
        idx = grupo_codes[ok].astype(np.int64) * n_bins + bin_codes[ok]
        tam = n_grupos * n_bins
        soma = np.bincount(idx, weights=y[ok], minlength=tam)
        cont = np.bincount(idx, minlength=tam)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(cont > 0, soma / np.maximum(cont, 1), np.nan)
        return r.reshape(n_grupos, n_bins)

    def label(self, feature) -> str:
        return self.feature_labels.get(feature, feature)

    # ------------------------------------------------------------------
    # Binning de uma variável (classificação OU regressão)
    # ------------------------------------------------------------------
    def _mask_in(self, frame, feature, b):
        return _bin_mask_series(frame[feature], b)

    def _bin_label(self, feature, b) -> str:
        if b["kind"] == "na":
            return "(faltante)"
        if b["kind"] == "num":
            lbl = f"({_fmt(b['lo'])}, {_fmt(b['hi'])}]"
        elif (len(b["cats"]) == 1
              and self.var_meta.get(feature, {}).get("derived_from")):
            lbl = str(b["cats"][0])        # derivada: a categoria JÁ é o rótulo da faixa
        else:
            lbl = "{" + ", ".join(map(str, b["cats"])) + "}"
        return lbl + (" + faltante" if b.get("include_na") else "")

    @staticmethod
    def _splits_key(sp):
        """Chave hashável dos splits (cortes numéricos ou grupos categóricos)."""
        if not sp:
            return None
        if isinstance(sp[0], (list, tuple)):
            return tuple(tuple(g) for g in sp)
        return tuple(sp)

    def _invalidate_bins(self, *features):
        """Invalida o cache de binning APENAS das variáveis dadas (as chaves do
        _bins_cache começam pela feature) e marca o ranking como obsoleto — em vez
        de limpar o cache inteiro e re-rodar o optbinning de TODAS as candidatas."""
        feats = set(features)
        for k in [k for k in self._bins_cache if k[0] in feats]:
            self._bins_cache.pop(k, None)
        self._rank_version += 1

    def _resolve_bins(self, feature, max_n_bins=5, min_bin_size=0.05, splits=None,
                      sample=None):
        """Memoiza :meth:`_resolve_bins_uncached` (caro: roda o solver CP-SAT do
        optbinning). A chave cobre tudo que altera o resultado; o cache é
        invalidado POR VARIÁVEL em set/clear_manual_bins e clear_derived."""
        eff_splits = splits if splits is not None else self.var_meta.get(feature, {}).get("splits")
        sample_key = sample if sample is not None else self.ref_sample
        ck = (feature, max_n_bins, min_bin_size, sample_key, self._splits_key(eff_splits),
              self.var_meta.get(feature, {}).get("na_destino"))
        hit = self._bins_cache.get(ck)
        if hit is not None:
            return hit
        res = self._resolve_bins_uncached(feature, max_n_bins, min_bin_size, eff_splits, sample)
        self._bins_cache[ck] = res
        return res

    def _resolve_bins_uncached(self, feature, max_n_bins=5, min_bin_size=0.05, splits=None,
                               sample=None):
        """Resolve os bins de uma variável na amostra de ajuste (DES por padrão).
        Usa ``OptimalBinning`` (classificação) ou ``ContinuousOptimalBinning``
        (regressão). Devolve (bins, kind).

        Se ``splits`` não for informado e a variável tiver **bins manuais**
        (``var_meta[feature]['splits']``, definidos via :meth:`set_manual_bins`),
        eles são usados — sobrepondo o binning ótimo em toda a análise univariada."""
        if OptimalBinning is None:
            raise ImportError("optbinning não instalado. Rode: pip install optbinning")
        if splits is None:
            splits = self.var_meta.get(feature, {}).get("splits")
        fit = self._frame(sample, cols=[feature, self.target])
        kind = self._detect_kind(feature, fit)

        if splits is None:
            # binária (2 níveis): um nível por faixa, sem optbinning — o mínimo de
            # 5% por faixa apagava o IV das flags raras (ver _bins_binaria)
            binaria = self._bins_binaria(feature, fit, kind)
            if binaria is not None:
                return binaria, kind

        if kind == "num":
            if splits is not None:
                lo, hi = fit[feature].min(), fit[feature].max()
                cortes = [s for s in sorted(splits) if lo < s < hi]
            else:
                x = fit[feature].to_numpy(dtype="float64")
                y = fit[self.target].to_numpy(dtype="float64")
                ok = ~np.isnan(y)
                x, y = x[ok], y[ok]
                x_obs = x[~np.isnan(x)]
                if len(y) < 4 or x_obs.size == 0 or x_obs.min() == x_obs.max():
                    cortes = []
                else:
                    if self.task_type == "classification":
                        b = OptimalBinning(name=feature, dtype="numerical",
                                           max_n_bins=max_n_bins, min_bin_size=min_bin_size,
                                           monotonic_trend="auto_asc_desc")
                        cortes = _fit_optbinning_splits(b, x, y.astype(int))
                    else:
                        b = ContinuousOptimalBinning(
                            name=feature, dtype="numerical", max_n_bins=max_n_bins,
                            min_bin_size=min_bin_size, monotonic_trend="auto_asc_desc")
                        cortes = _fit_optbinning_splits(b, x, y)
            if not cortes:
                return [], kind
            edges = [-np.inf, *cortes, np.inf]
            bins = [{"kind": "num", "lo": edges[i], "hi": edges[i + 1]}
                    for i in range(len(edges) - 1)]
            if fit[feature].isna().any():
                bins.append({"kind": "na"})
            return self._aplica_destino_na(feature, bins, fit, splits is not None), kind

        # categórico
        na_present = bool(fit[feature].isna().any())
        if splits is not None:
            grupos = [list(g) for g in splits]
        else:
            f2 = fit[fit[feature].notna() & fit[self.target].notna()]
            xs = f2[feature].astype(str).to_numpy()
            ys = f2[self.target].to_numpy(dtype="float64")
            if len(ys) < 4 or not (xs != xs[0]).any():
                grupos = []
            else:
                if self.task_type == "classification":
                    b = OptimalBinning(name=feature, dtype="categorical",
                                       max_n_bins=max_n_bins, min_bin_size=min_bin_size,
                                       monotonic_trend="auto_asc_desc")
                    grupos = [list(a) for a in _fit_optbinning_splits(b, xs, ys.astype(int))]
                else:
                    b = ContinuousOptimalBinning(
                        name=feature, dtype="categorical", max_n_bins=max_n_bins,
                        min_bin_size=min_bin_size, monotonic_trend="auto_asc_desc")
                    grupos = [list(a) for a in _fit_optbinning_splits(b, xs, ys)]
        _NA_TOK = {"nan", "NaN", "<NA>", "None"}
        bins = []
        for g in grupos:
            cats = [str(c) for c in g if str(c) not in _NA_TOK]
            if cats:
                bins.append({"kind": "cat", "cats": cats})
        if bins and na_present:
            bins.append({"kind": "na"})
        return self._aplica_destino_na(feature, bins, fit, splits is not None), kind

    # níveis da flag abaixo destes limites deixam o IV instável (aviso, não bloqueio)
    _BINARIA_MIN_SHARE = 0.01
    _BINARIA_MIN_EVENTOS = 30

    def _bins_binaria(self, feature, fit, kind):
        """Bins de uma variável **binária** (exatamente 2 valores distintos fora os
        faltantes): um nível por faixa, sem optbinning. ``None`` se não for binária.

        O binning ótimo exige ``min_bin_size`` (5%) por faixa; numa flag rara
        (ex.: "teve restrição" em 3% da base) ele não consegue separar os dois
        níveis e o IV sai NaN/0 — a variável parecia inútil sendo forte. Aqui cada
        nível vira a sua faixa (numérica: corte no ponto médio; categórica/bool:
        um grupo por nível) e os faltantes seguem em faixa própria."""
        col = fit[feature]
        obs = col.dropna()
        # rejeição rápida: >2 níveis já nas primeiras linhas ⇒ não é binária
        # (evita varrer/converter milhões de valores em cada variável contínua)
        if pd.unique(obs.iloc[:10_000]).size > 2:
            return None
        if kind == "num":
            vals = np.sort(pd.unique(obs.to_numpy(dtype="float64")))
            if vals.size != 2:
                return None
            bins = [{"kind": "num", "lo": -np.inf, "hi": float(vals.mean())},
                    {"kind": "num", "lo": float(vals.mean()), "hi": np.inf}]
        else:
            if obs.nunique() > 2:                          # hash, sem str() por linha
                return None
            niveis = sorted({str(v) for v in pd.unique(obs)} - {"nan", "NaN", "<NA>", "None"})
            if len(niveis) != 2:
                return None
            bins = [{"kind": "cat", "cats": [v]} for v in niveis]
        if col.isna().any():
            bins.append({"kind": "na"})
        self._avisa_binaria_rara(feature, fit, bins)
        return bins

    def _avisa_binaria_rara(self, feature, fit, bins) -> None:
        """Avisa quando o nível minoritário da flag é pequeno demais para um IV
        estável (share < 1% ou, na classificação, < 30 eventos)."""
        y = fit[self.target].to_numpy(dtype="float64")
        tot = int(np.sum(~np.isnan(y)))
        if not tot:
            return
        fracos = []
        for b, m in zip(bins[:2], _bin_masks(fit[feature], bins[:2])):
            yi = y[m]
            yi = yi[~np.isnan(yi)]
            share = len(yi) / tot
            eventos = int((yi == 1).sum()) if self.task_type == "classification" else None
            if share < self._BINARIA_MIN_SHARE or (
                    eventos is not None and eventos < self._BINARIA_MIN_EVENTOS):
                txt = f"{self._bin_label(feature, b)}: {100 * share:.2f}% da base"
                fracos.append(txt + (f", {eventos} eventos" if eventos is not None else ""))
        if fracos:
            warnings.warn(
                f"'{self.label(feature)}' é binária com nível raro ({'; '.join(fracos)}) — "
                "o IV é calculado nível a nível, mas fica instável com tão poucos casos.")

    def _aplica_destino_na(self, feature, bins, fit, manual) -> list:
        """Categorização manual: move os faltantes para a faixa escolhida em
        :meth:`set_missing_bin` (a faixa ganha ``include_na`` e a faixa
        "(faltante)" some). Vale mesmo sem faltante na referência — assim os NaN
        da OOT/escoragem já têm destino. Binning ótimo: nada muda."""
        destino = self.var_meta.get(feature, {}).get("na_destino") if manual else None
        faixas = [b for b in bins if b["kind"] != "na"]
        if destino in (None, "separado") or not faixas:
            return bins
        if destino in ("pior", "melhor"):
            y = fit[self.target].to_numpy(dtype="float64")
            riscos = [self._risco(y[m]) for m in _bin_masks(fit[feature], faixas)]
            validos = [i for i, r in enumerate(riscos) if np.isfinite(r)]
            if not validos:
                return bins
            escolha = max if destino == "pior" else min
            idx = escolha(validos, key=lambda i: riscos[i])
        else:
            idx = int(destino)
            if not 0 <= idx < len(faixas):
                warnings.warn(
                    f"'{self.label(feature)}': faltantes apontavam para a faixa {idx + 1}, "
                    f"mas a categorização atual tem {len(faixas)} faixas — faltantes "
                    "voltam para faixa própria. Escolha o destino de novo.")
                return bins
        faixas = [dict(b) for b in faixas]
        faixas[idx]["include_na"] = True
        return faixas

    def _risco(self, y) -> float:
        """Valor de risco de um conjunto de alvos: event_rate (classificação)
        ou média (regressão). NaN-safe."""
        y = np.asarray(y, dtype="float64")
        y = y[~np.isnan(y)]
        return float(y.mean()) if y.size else float("nan")

    # ------------------------------------------------------------------
    # A) Análise univariada: tabela de bins com logodds/WoE/IV
    # ------------------------------------------------------------------
    def variable_table(self, feature, sample=None, max_n_bins=6, min_bin_size=0.05,
                       splits=None) -> pd.DataFrame:
        """Tabela por faixa de uma variável (na amostra de referência):
        ``faixa, n, repr_%, risco`` e — na **classificação** — ``woe, logodds`` e
        ``iv_parcial`` (escala WoE/IV de Siddiqi). Na **regressão**, ``iv_parcial``
        é o desvio absoluto ponderado do alvo. IV total em ``.attrs['iv']``."""
        bins, kind = self._resolve_bins(feature, max_n_bins, min_bin_size, splits, sample)
        sub = self._frame(sample, cols=[feature, self.target])
        n_tot = max(len(sub), 1)
        risco_label = "event_rate" if self.task_type == "classification" else "alvo_medio"
        if not bins:
            out = pd.DataFrame(columns=["faixa", "n", "repr_%", risco_label])
            out.attrs.update(iv=float("nan"), mono_ok=True, kind=kind,
                             risco_label=risco_label)
            return out

        y_all = sub[self.target].to_numpy(dtype="float64")
        mean_global = self._risco(y_all)
        n_evt_tot = float(np.nansum(y_all == 1)) if self.task_type == "classification" else 0.0
        n_non_tot = float(np.nansum(y_all == 0)) if self.task_type == "classification" else 0.0
        n_base = float(np.sum(~np.isnan(y_all)))

        rows, iv_total, is_na = [], 0.0, []
        for b, m in zip(bins, _bin_masks(sub[feature], bins)):
            yi = y_all[m]
            yi_ok = yi[~np.isnan(yi)]
            n_i = int(m.sum())
            if n_i == 0:
                continue
            is_na.append(b.get("kind") == "na")
            risco = self._risco(yi)
            row = {"faixa": self._bin_label(feature, b), "n": n_i,
                   "repr_%": round(100 * n_i / n_tot, 1),
                   risco_label: round(risco, 4) if np.isfinite(risco) else np.nan}
            if self.task_type == "classification":
                n_evt = float((yi_ok == 1).sum())
                n_non = float((yi_ok == 0).sum())
                d_evt = n_evt / max(n_evt_tot, _EPS)
                d_non = n_non / max(n_non_tot, _EPS)
                woe = float(np.log((d_non + _EPS) / (d_evt + _EPS)))
                logodds = (float(np.log((risco + _EPS) / (1 - risco + _EPS)))
                           if np.isfinite(risco) else np.nan)
                ivp = (d_non - d_evt) * woe
                row.update(woe=round(woe, 4),
                           logodds=round(logodds, 4) if np.isfinite(logodds) else np.nan,
                           iv_parcial=round(ivp, 4))
                iv_total += ivp
            else:
                ivp = ((yi_ok.size / max(n_base, _EPS)) *
                       abs(risco - mean_global)) if np.isfinite(risco) else 0.0
                row["iv_parcial"] = round(ivp, 4)
                iv_total += ivp
            rows.append(row)

        out = pd.DataFrame(rows)
        # a faixa de faltantes (NA) não pertence à sequência ordenável pela variável —
        # excluída do teste de monotonicidade (sua média fica fixa no fim e
        # quebraria/mascararia o mono_ok das faixas reais).
        if risco_label in out and len(out):
            rcol = out.loc[~np.asarray(is_na), risco_label]
        else:
            rcol = pd.Series(dtype=float)
        mono = bool(rcol.is_monotonic_increasing or rcol.is_monotonic_decreasing) \
            if len(rcol) else True
        out.attrs.update(iv=round(float(iv_total), 4), mono_ok=mono, kind=kind,
                         risco_label=risco_label, mean_global=round(mean_global, 4),
                         # risco das faixas ORDENÁVEIS (sem o NA) p/ tendência/inversões
                         risco_ordenavel=[float(x) for x in rcol])
        return out

    def variable_iv(self, features=None, sample=None, max_n_bins=5, min_bin_size=0.05,
                    with_psi=True) -> pd.DataFrame:
        """Ranking das variáveis candidatas para apoiar a seleção: ``variavel,
        tipo, n_bins, iv, forca, tendencia, n_inversoes, psi_<amostra>, estabilidade,
        incluida, categoria``. IV binário (Siddiqi) na classificação; IV contínuo
        na regressão. PSI calculado nos mesmos bins (DES × cada amostra).

        A parte CARA (binning + IV + PSI por variável) é memoizada por assinatura de
        bins/amostra (``_rank_version``); a parte MUTÁVEL barata (incluida/categoria/
        motivo) é reanexada a cada chamada. Assim include/exclude/set_category e a 2ª
        chamada após auto_select/auto_categorize NÃO recomputam o ranking inteiro."""
        features = list(features) if features is not None else list(self.candidates)
        base = self._variable_iv_base(tuple(features), sample, max_n_bins,
                                      min_bin_size, bool(with_psi))
        df = base.copy()
        feats = df["variavel"].tolist()
        # estado mutável (barato) reanexado fresco — não invalida o cache caro
        df["incluida"] = [f in self.included for f in feats]
        df["categoria"] = [self.var_meta.get(f, {}).get("categoria") for f in feats]
        df["motivo"] = [self.var_meta.get(f, {}).get("motivo", "") for f in feats]
        df["bins_manuais"] = [bool(self.var_meta.get(f, {}).get("splits")) for f in feats]
        if not df["motivo"].astype(bool).any():
            df = df.drop(columns="motivo")   # só aparece após auto-categorizar
        return df

    def _variable_iv_base(self, features, sample, max_n_bins, min_bin_size,
                          with_psi) -> pd.DataFrame:
        """Parte CARA do ranking (binning/IV/PSI), memoizada por _rank_version.
        Não inclui incluida/categoria/motivo (estado mutável, reanexado em
        :meth:`variable_iv`)."""
        key = (features, sample, max_n_bins, min_bin_size, with_psi, self._rank_version)
        hit = self._rank_cache.get(key)
        if hit is not None:
            return hit
        nonref = self._nonref_samples() if (with_psi and self.sample_col) else []
        # cache POR VARIÁVEL (assinatura = bins manuais, destino dos faltantes e
        # derivação): mudar os bins de UMA variável invalidava o ranking inteiro
        # (_rank_version) e re-calculava IV/PSI de todas — agora só a dela
        row_cache = self.__dict__.setdefault("_iv_row_cache", {})
        rows = []
        for feat in features:
            meta = self.var_meta.get(feat, {})
            rkey = (feat, sample, max_n_bins, min_bin_size, with_psi, tuple(nonref),
                    self._splits_key(meta.get("splits")), repr(meta.get("na_destino")),
                    repr(meta.get("derived_bins")), meta.get("derived_from"),
                    feat in self.df.columns)
            hit_row = row_cache.get(rkey)
            if hit_row is not None:
                rows.append(dict(hit_row))
                continue
            iv, nb, kind, trend, n_inv = np.nan, 0, "—", "—", 0
            psi_vals = {a: np.nan for a in nonref}
            try:
                vt = self.variable_table(feat, sample=sample, max_n_bins=max_n_bins,
                                         min_bin_size=min_bin_size)
                kind = vt.attrs.get("kind", "—")
                nb = len(vt)
                iv = vt.attrs.get("iv", np.nan)
                if nb:
                    # usa o risco das faixas ORDENÁVEIS (exclui o NA, que não entra
                    # na tendência/contagem de inversões)
                    vals = vt.attrs.get("risco_ordenavel")
                    if vals is None:
                        vals = vt[vt.attrs.get("risco_label")].tolist()
                    trend, n_inv = _trend(vals)
                if nonref:
                    psi_vals = self._variable_psi(feat, nonref, max_n_bins, min_bin_size)
            except Exception:
                pass
            row = {"variavel": feat, "tipo": kind, "n_bins": nb,
                   "iv": round(float(iv), 4) if np.isfinite(iv) else np.nan,
                   "forca": _classifica_iv(iv, self.task_type),
                   "tendencia": trend, "n_inversoes": n_inv}
            if nonref:
                for a in nonref:
                    row[f"psi_{a}"] = psi_vals[a]
                validos = [v for v in psi_vals.values() if np.isfinite(v)]
                pior = max(validos) if validos else np.nan
                row["pior_psi"] = round(float(pior), 4) if np.isfinite(pior) else np.nan
                row["estabilidade"] = _classifica_psi(pior)
            if len(row_cache) > 4096:          # backstop de memória
                row_cache.clear()
            row_cache[rkey] = dict(row)
            rows.append(row)
        base = (pd.DataFrame(rows)
                .sort_values("iv", ascending=False, na_position="last")
                .reset_index(drop=True))
        if len(self._rank_cache) > 8:      # backstop de memória (versões antigas)
            self._rank_cache.clear()
        self._rank_cache[key] = base
        return base

    def _variable_psi(self, feature, samples, max_n_bins=5, min_bin_size=0.05,
                      eps=1e-6) -> dict:
        """PSI da variável (bins fixados na DES) entre DES e cada amostra."""
        bins, _kind = self._resolve_bins(feature, max_n_bins, min_bin_size)
        out = {a: np.nan for a in samples}
        if not bins:
            return out
        # faixa de cada linha numa passada na coluna inteira + contagem por
        # amostra (np.bincount): sem copiar a referência e cada amostra
        nb = len(bins)
        bc = _bin_codes(self.df[feature], bins)
        ref_m = self._rows_mask(self.ref_sample, amostrar=False)   # ranking: base inteira
        n_ref = max(int(ref_m.sum()), 1)
        ref_cont = np.bincount(bc[ref_m & (bc >= 0)], minlength=nb)
        ref_pct = [max(int(c) / n_ref, eps) for c in ref_cont]
        for a in samples:
            m = self._rows_mask(a, amostrar=False)
            n_cur = int(m.sum())
            if n_cur == 0:
                continue
            cont = np.bincount(bc[m & (bc >= 0)], minlength=nb)
            cur_pct = [int(c) / n_cur for c in cont]
            out[a] = round(_psi_from_shares(ref_pct, cur_pct, eps), 4)
        return out

    def variable_summary(self, feature, sample=None) -> dict:
        """Resumo de uma variável: %missing, estatísticas/top-categorias, IV,
        força, tendência e PSI por amostra."""
        sub = self._frame(sample, cols=[feature])
        col = sub[feature]
        kind = self._detect_kind(feature, sub)
        n = int(len(col)); n_miss = int(col.isna().sum())
        res = {"variavel": feature, "tipo": kind, "n": n, "n_missing": n_miss,
               "pct_missing": round(100 * n_miss / n, 2) if n else float("nan"),
               "incluida": feature in self.included,
               "categoria": self.var_meta.get(feature, {}).get("categoria")}
        if kind == "num":
            x = col.to_numpy(dtype="float64"); x = x[~np.isnan(x)]
            if x.size:
                res.update(media=round(float(np.mean(x)), 4),
                           mediana=round(float(np.median(x)), 4),
                           desvio=round(float(np.std(x, ddof=1)) if x.size > 1 else 0.0, 4),
                           min=round(float(np.min(x)), 4),
                           p5=round(float(np.percentile(x, 5)), 4),
                           p95=round(float(np.percentile(x, 95)), 4),
                           max=round(float(np.max(x)), 4))
        else:
            vc = col.dropna().astype(str).value_counts(normalize=True)
            res["top_categorias"] = [(c, round(100 * p, 1)) for c, p in vc.head(8).items()]
        res.update(iv=None, forca="—", tendencia="—", n_inversoes=0, psi={}, pior_psi=None)
        try:
            # max_n_bins=6 casa com a tabela/plots da aba (variable_table usa 6 por
            # default) → mesma chave de _bins_cache → cache hit, evita re-rodar o
            # optbinning da MESMA variável a cada clique em "Analisar variável".
            ivt = self.variable_iv(features=[feature], sample=sample, max_n_bins=6)
            if len(ivt):
                r0 = ivt.iloc[0]
                res["iv"] = None if pd.isna(r0["iv"]) else float(r0["iv"])
                res["forca"] = r0["forca"]
                res["tendencia"] = r0["tendencia"]
                res["n_inversoes"] = int(r0["n_inversoes"])
                for c in ivt.columns:
                    if c.startswith("psi_"):
                        res["psi"][c[4:]] = None if pd.isna(r0[c]) else float(r0[c])
                if "pior_psi" in ivt and not pd.isna(r0["pior_psi"]):
                    res["pior_psi"] = float(r0["pior_psi"])
        except Exception:
            pass
        return res

    def variable_by_safra(self, feature, time_col=None, sample=None,
                          all_samples=False) -> pd.DataFrame:
        """Percentis (min, p5, média, p95, max) e %missing de variável NUMÉRICA por safra.

        ``all_samples=True`` usa **todas as safras da base** (todas as amostras),
        não só a DES — útil para ver o comportamento no tempo em toda a série."""
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col.")
        # fatias por safra sobre arrays (ordem por safra memoizada) + percentil do
        # numpy por fatia — o groupby().quantile do pandas custava ~1 s/variável
        idx, limites, rot = self._fatias_por_safra(time_col, sample, all_samples)
        xall = pd.to_numeric(self.df[feature], errors="coerce").to_numpy(dtype="float64")
        rows = []
        for j, per in enumerate(rot):
            ii = idx[limites[j]:limites[j + 1]]
            n = int(ii.size)
            if n == 0:
                continue
            x = xall[ii]
            x = x[~np.isnan(x)]
            row = {"safra": per, "n": n,
                   "pct_missing": round(100 * (n - x.size) / n, 1)}
            if x.size:
                p5, p95 = np.percentile(x, [5, 95])
                row.update(min=round(float(x.min()), 3), p5=round(float(p5), 3),
                           media=round(float(x.mean()), 3), p95=round(float(p95), 3),
                           max=round(float(x.max()), 3))
            else:
                row.update({c: float("nan") for c in ("min", "p5", "media", "p95", "max")})
            rows.append(row)
        if not rows:
            return pd.DataFrame(columns=["safra", "n", "pct_missing", "min", "p5",
                                         "media", "p95", "max"])
        return pd.DataFrame(rows).sort_values("safra").reset_index(drop=True)

    def variable_share_by_safra(self, feature, time_col=None, sample=None, top=8,
                                all_samples=False) -> pd.DataFrame:
        """Representatividade (%) de cada categoria por safra (variável CATEGÓRICA).
        ``all_samples=True`` usa todas as safras da base (todas as amostras)."""
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col.")
        # rótulo por VALOR DISTINTO (factorize), não por linha — o antigo
        # cat.map(lab) chamava uma função Python milhões de vezes
        cod_s, rot_s = self._safra_codes(time_col)
        linhas = self._rows_mask(sample, all_rows=all_samples)
        cat = self.df[feature][linhas]
        codes, uniq = pd.factorize(cat, use_na_sentinel=True)
        ustr = np.array([str(u) for u in uniq], dtype=object)
        cont = pd.Series(np.bincount(codes[codes >= 0], minlength=len(ustr)),
                         index=ustr).groupby(level=0, sort=False).sum()
        keep = list(cont.sort_values(ascending=False, kind="stable").head(top).index)
        lab_u = np.array([s if s in keep else "outras" for s in ustr], dtype=object)
        rotulo = np.where(codes >= 0, lab_u[np.where(codes >= 0, codes, 0)]
                          if len(lab_u) else "outras", "(faltante)")
        saf = cod_s[linhas]
        ok = saf >= 0
        tab = pd.crosstab(np.asarray(rot_s, dtype=object)[saf[ok]], rotulo[ok])
        tab.columns.name = feature
        if tab.empty:
            return pd.DataFrame(columns=["safra"])
        pct = tab.div(tab.sum(axis=1), axis=0) * 100
        # "outras"/"(faltante)" vão ao fim UMA vez só — uma variável criada a partir
        # de faixas já traz "(faltante)" como categoria literal; repetir o rótulo
        # duplicava a coluna e quebrava o gráfico de share (stackplot 2-D)
        order = [c for c in keep if c in pct.columns and c not in ("outras", "(faltante)")]
        order += [c for c in ("outras", "(faltante)") if c in pct.columns]
        pct = pct[order].round(1).sort_index()
        pct.index.name = "safra"
        return pct.reset_index()

    def variable_psi_by_safra(self, feature, time_col=None, max_n_bins=10,
                              min_bin_size=0.05, eps=1e-6) -> pd.DataFrame:
        """PSI da variável por safra vs DES (bins fixados na DES)."""
        if self.sample_col is None:
            raise ValueError("PSI por safra requer sample_col (referência DES).")
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col.")
        bins, _kind = self._resolve_bins(feature, max_n_bins, min_bin_size)
        if not bins:
            return pd.DataFrame(columns=["safra", "n", "psi", "classificacao"])
        # contagens faixa × safra numa passada (np.bincount sobre a faixa de cada linha)
        nb = len(bins)
        bc = self._feature_bin_codes(feature, bins)
        ref_m = self._rows_mask(self.ref_sample)
        n_ref = max(int(ref_m.sum()), 1)
        ref_cont = np.bincount(bc[ref_m & (bc >= 0)], minlength=nb)
        ref_pct = [max(int(c) / n_ref, eps) for c in ref_cont]
        cod, rot = self._safra_codes(time_col)
        n_t = np.bincount(cod[cod >= 0], minlength=len(rot))
        ok = (cod >= 0) & (bc >= 0)
        cont = np.bincount(cod[ok].astype(np.int64) * nb + bc[ok],
                           minlength=len(rot) * nb).reshape(len(rot), nb)
        rows = []
        for k, r in enumerate(rot):
            n_g = int(n_t[k])
            if n_g == 0:
                continue
            cur_pct = [int(c) / n_g for c in cont[k]]
            psi = _psi_from_shares(ref_pct, cur_pct, eps)
            rows.append({"safra": r, "n": n_g, "psi": round(psi, 4),
                         "classificacao": _classifica_psi(psi)})
        if not rows:
            return pd.DataFrame(columns=["safra", "n", "psi", "classificacao"])
        return pd.DataFrame(rows).sort_values("safra").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Inversão da ordem de risco dos BINS de uma variável (entre amostras/safras)
    # ------------------------------------------------------------------
    def _variable_bin_series(self, feature, bins, time_col=None, sample=None,
                             min_n=20):
        """Risco de cada bin por amostra e por safra. Devolve dict com chaves
        ``ordered`` (bins na ordem de risco DES), ``labels``, ``samples`` (xs +
        series por bin) e ``safras`` (xs + series por bin)."""
        # vetorizado: a faixa de cada linha sai UMA vez (_feature_bin_codes) e o
        # risco por faixa × amostra/safra vem de np.bincount — antes era uma
        # máscara por faixa × safra sobre recortes copiados da base
        labels = [self._bin_label(feature, b) for b in bins]
        nb = len(bins)
        bc = self._feature_bin_codes(feature, bins)
        zeros = np.zeros(len(self.df), dtype=np.int32)
        ref_risco = list(self._risco_por_grupo(
            bc, zeros, 1, nb, self._rows_mask(self.ref_sample))[0])
        order = sorted(range(len(bins)),
                       key=lambda i: (np.inf if pd.isna(ref_risco[i]) else ref_risco[i]))

        # por amostra
        xs_s = self._samples()
        cod_a = np.full(len(self.df), -1, dtype=np.int32)
        for j, a in enumerate(xs_s):
            cod_a[self._rows_mask(a)] = j
        mat_s = self._risco_por_grupo(bc, cod_a, len(xs_s), nb,
                                      np.ones(len(self.df), dtype=bool))
        ser_s = {i: list(mat_s[:, i]) for i in range(nb)}

        # por safra
        xs_t, ser_t = [], {i: [] for i in range(nb)}
        tcol = time_col or self.date_col
        if tcol is not None and tcol in self.df.columns:
            linhas = self._rows_mask(sample, all_rows=not sample)
            cod_t, rot_t = self._safra_codes(tcol)
            n_t = np.bincount(cod_t[linhas & (cod_t >= 0)], minlength=len(rot_t))
            mat_t = self._risco_por_grupo(bc, cod_t, len(rot_t), nb, linhas)
            for k, rot in enumerate(rot_t):
                if n_t[k] == 0 or n_t[k] < min_n:   # safra ausente ou pequena fica fora
                    continue
                xs_t.append(rot)
                for i in range(nb):
                    ser_t[i].append(mat_t[k, i])
        return {"ordered": order, "labels": labels, "ref_risco": ref_risco,
                "xs_sample": xs_s, "ser_sample": ser_s,
                "xs_safra": xs_t, "ser_safra": ser_t}

    def variable_inversion(self, feature, time_col=None, sample=None,
                           max_n_bins=6, min_bin_size=0.05, min_n=20) -> dict:
        """Diagnóstico de inversão da ordem de risco dos bins de uma variável,
        entre amostras e entre safras. Veredito verde/amarelo/vermelho — análoga
        à inversão entre folhas-irmãs do alvo, mas sobre os bins de UMA variável."""
        bins, _kind = self._resolve_bins(feature, max_n_bins, min_bin_size, sample=sample)
        if len(bins) < 2:
            return {"status": "green", "samples": [], "safras": [], "ordered": [],
                    "labels": [], "ref_risco": [], "sample_inv": 0, "n_safras": 0,
                    "safras_inv": 0, "safra_rate": 0.0,
                    "msg": "menos de 2 faixas — sem ordem para inverter"}
        s = self._variable_bin_series(feature, bins, time_col, sample, min_n)
        ordered = s["ordered"]

        sample_rows = []
        for j, xlab in enumerate(s["xs_sample"]):
            vals = {i: s["ser_sample"][i][j] for i in range(len(bins))}
            n_inv, npp = _count_inversions(ordered, vals)
            sample_rows.append({"amostra": xlab, "n_inv": n_inv, "n_pares": npp})
        safra_rows = []
        for j, xlab in enumerate(s["xs_safra"]):
            vals = {i: s["ser_safra"][i][j] for i in range(len(bins))}
            n_inv, npp = _count_inversions(ordered, vals)
            if npp == 0:
                continue
            safra_rows.append({"safra": xlab, "n_inv": n_inv, "n_pares": npp})

        sample_inv = sum(r["n_inv"] for r in sample_rows if r["amostra"] != self.ref_sample)
        n_safras = len(safra_rows)
        safras_inv = sum(1 for r in safra_rows if r["n_inv"] > 0)
        safra_rate = (safras_inv / n_safras) if n_safras else 0.0
        status = ("red" if (sample_inv > 0 or safra_rate > 0.25)
                  else "yellow" if safras_inv > 0 else "green")
        return {"status": status, "samples": sample_rows, "safras": safra_rows,
                "ordered": ordered, "labels": s["labels"], "ref_risco": s["ref_risco"],
                "sample_inv": sample_inv, "n_safras": n_safras, "safras_inv": safras_inv,
                "safra_rate": safra_rate, "series": s}

    # ------------------------------------------------------------------
    # Plots de variável
    # ------------------------------------------------------------------
    def plot_variable_logodds(self, feature, sample=None, max_n_bins=6, min_bin_size=0.05,
                              figsize=(7.6, 3.4), dpi=150, save_path=None, ax=None,
                              target_ylim01=False):
        """Barras de representatividade (%) + linha de **logodds/WoE** (classificação)
        ou **alvo médio** (regressão) por faixa — leitura de monotonicidade.

        ``target_ylim01=True`` fixa o eixo do **alvo médio** em [0, 1] na regressão
        (padroniza a leitura em alvos limitados, ex.: LGD); ignorado na classificação."""
        vt = self.variable_table(feature, sample, max_n_bins, min_bin_size)
        fig, ax = _new_ax(figsize, dpi, ax)
        if vt.empty:
            ax.text(0.5, 0.5, "sem faixas", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        rcol = vt.attrs["risco_label"]
        labels = vt["faixa"].tolist()
        reprs = vt["repr_%"].to_numpy()
        xs = list(range(len(labels)))
        cols = ["#c98a8a" if "faltante" in f else "#9db8cf" for f in labels]
        ax.bar(xs, reprs, color=cols, edgecolor="#2f5d82", alpha=0.85, width=0.7)
        ax.set_ylabel("% da amostra"); ax.set_ylim(0, float(np.nanmax(reprs)) * 1.2 + 1)
        ax2 = ax.twinx()
        if self.task_type == "classification" and "logodds" in vt:
            yline = vt["logodds"].to_numpy(); ylabel = "logodds"
        else:
            yline = vt[rcol].to_numpy(); ylabel = ("event_rate"
                                                   if self.task_type == "classification"
                                                   else "alvo médio")
        ax2.plot(xs, yline, color="#15324a", lw=2.3, marker="o", ms=5,
                 markeredgecolor="#fff", markeredgewidth=0.6, label=ylabel)
        ax2.set_ylabel(ylabel)
        if target_ylim01 and self.task_type != "classification":
            ax2.set_ylim(0.0, 1.0)
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_xlim(-0.7, len(labels) - 0.3)
        mono = "monotônica" if vt.attrs.get("mono_ok") else "NÃO monotônica"
        ax.set_title(f"'{self.label(feature)}' · {ylabel} por faixa  ·  IV={vt.attrs['iv']}"
                     f"  ·  {mono}", fontsize=10.5, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.12)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_variable_distribution(self, feature, sample=None, max_n_bins=8,
                                   min_bin_size=0.03, figsize=(7.2, 3.1),
                                   dpi=150, save_path=None, ax=None):
        """Distribuição da variável por faixa (faltantes em destaque)."""
        vt = self.variable_table(feature, sample, max_n_bins, min_bin_size)
        fig, ax = _new_ax(figsize, dpi, ax)
        if vt.empty:
            sub = self._frame(sample, cols=[feature])
            # categórica com 1 categoria / poucos dados ⇒ vt vazio. NÃO coagir para
            # float (quebra em strings); ramificar por tipo, como plot_*_badrate.
            if self._detect_kind(feature, sub) == "num":
                x = sub[feature].to_numpy(dtype="float64"); x = x[~np.isnan(x)]
                if x.size:
                    ax.hist(x, bins=20, color="steelblue", alpha=0.85, edgecolor="#2f5d82")
            else:
                vc = sub[feature].astype("object").value_counts().head(20)
                n_tot = max(int(sub[feature].notna().sum()), 1)
                if len(vc):
                    xs = list(range(len(vc)))
                    ax.bar(xs, 100 * vc.to_numpy() / n_tot, color="steelblue",
                           edgecolor="#2f5d82", alpha=0.9, width=0.72)
                    ax.set_xticks(xs)
                    ax.set_xticklabels([str(k) for k in vc.index], rotation=25,
                                       ha="right", fontsize=8)
                else:
                    ax.text(0.5, 0.5, "sem faixas", ha="center", va="center",
                            transform=ax.transAxes, color="#889")
        else:
            labels = vt["faixa"].tolist(); reprs = vt["repr_%"].to_numpy()
            cols = ["#c98a8a" if "faltante" in f else "steelblue" for f in labels]
            xs = list(range(len(labels)))
            ax.bar(xs, reprs, color=cols, edgecolor="#2f5d82", alpha=0.9, width=0.72)
            for x0, rp in zip(xs, reprs):
                ax.text(x0, rp, f"{rp:.0f}%", ha="center", va="bottom", fontsize=7.5,
                        color="#15324a")
            ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
            ax.set_xlim(-0.75, len(labels) - 0.25)
            ax.set_ylim(0, float(np.nanmax(reprs)) * 1.16 + 1)
        ax.set_ylabel("% da amostra")
        ax.set_title(f"Distribuição de '{self.label(feature)}'",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_variable_distribution_badrate(self, feature, sample=None, max_n_bins=6,
                                           min_bin_size=0.05, figsize=(9.0, 3.8), dpi=150,
                                           save_path=None, ax=None, target_ylim01=False):
        """Um único gráfico: barras de **distribuição** (% da amostra por faixa) +
        linha do risco por faixa — **% de maus** (event_rate) na classificação ou
        **alvo médio** na regressão. Faltantes destacados.

        ``target_ylim01=True`` fixa o eixo do **alvo médio** em [0, 1] na regressão
        (padroniza a leitura em alvos limitados, ex.: LGD); ignorado na classificação."""
        vt = self.variable_table(feature, sample, max_n_bins, min_bin_size)
        # com muitas faixas (>8) o gráfico fica apertado — aumenta a altura
        if ax is None and len(vt) > 8:
            figsize = (figsize[0], figsize[1] + 0.30 * (len(vt) - 8))
        fig, ax = _new_ax(figsize, dpi, ax)
        if vt.empty:
            ax.text(0.5, 0.5, "sem faixas", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        rcol = vt.attrs["risco_label"]
        labels = vt["faixa"].tolist()
        reprs = vt["repr_%"].to_numpy()
        xs = list(range(len(labels)))
        cols = ["#c98a8a" if "faltante" in f else "steelblue" for f in labels]
        ax.bar(xs, reprs, color=cols, edgecolor="#2f5d82", alpha=0.85, width=0.7,
               label="% da amostra")
        for x0, rp in zip(xs, reprs):
            ax.text(x0, rp, f"{rp:.0f}%", ha="center", va="bottom", fontsize=7.5,
                    color="#15324a")
        ax.set_ylabel("% da amostra"); ax.set_ylim(0, float(np.nanmax(reprs)) * 1.2 + 1)

        is_clf = self.task_type == "classification"
        risco = vt[rcol].to_numpy(dtype="float64")
        yline = risco * 100 if is_clf else risco
        ylabel = "% de maus" if is_clf else "alvo médio"
        ax2 = ax.twinx()
        ax2.plot(xs, yline, color="crimson", lw=2.3, marker="o", ms=5.5,
                 markeredgecolor="#fff", markeredgewidth=0.7, label=ylabel)
        for x0, yv in zip(xs, yline):
            if np.isfinite(yv):
                ax2.text(x0, yv, (f"{yv:.1f}%" if is_clf else f"{yv:.3f}"),
                         ha="center", va="bottom", fontsize=7.5, color="crimson")
        ax2.set_ylabel(ylabel, color="crimson"); ax2.tick_params(axis="y", labelcolor="crimson")
        finite = yline[np.isfinite(yline)]
        if target_ylim01 and not is_clf:
            ax2.set_ylim(0.0, 1.0)
        elif finite.size:
            ax2.set_ylim(0, float(np.nanmax(finite)) * 1.25 + (1 if is_clf else 1e-9))
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_xlim(-0.7, len(labels) - 0.3)
        mono = "monotônica" if vt.attrs.get("mono_ok") else "NÃO monotônica"
        ax.set_title(f"'{self.label(feature)}' · distribuição & {ylabel}  ·  "
                     f"IV={vt.attrs['iv']}  ·  {mono}",
                     fontsize=10.5, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.12)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_variable_timeseries(self, feature, time_col=None, sample=None,
                                 figsize=(8.6, 3.4), dpi=150, save_path=None, ax=None,
                                 all_samples=False):
        """Numérica: percentis por safra. Categórica: área empilhada de share.
        ``all_samples=True`` considera todas as safras da base (todas as amostras)."""
        frame = self.df if all_samples else self._frame(sample)
        if self._detect_kind(feature, frame) == "cat":
            return self._plot_share_timeseries(feature, time_col, sample, figsize,
                                               dpi, save_path, ax, all_samples)
        bs = self.variable_by_safra(feature, time_col, sample, all_samples=all_samples)
        fig, ax = _new_ax(figsize, dpi, ax)
        if bs.empty or bs["media"].notna().sum() == 0:
            ax.text(0.5, 0.5, "sem dados por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        x = list(range(len(bs)))
        ax.fill_between(x, bs["min"], bs["max"], color="steelblue", alpha=0.07)
        ax.plot(x, bs["min"], color="#9bb7c9", lw=1.0)
        ax.plot(x, bs["max"], color="#9bb7c9", lw=1.0, label="min / max")
        ax.plot(x, bs["p5"], color="#6f93ad", lw=1.3, ls="--")
        ax.plot(x, bs["p95"], color="#6f93ad", lw=1.3, ls="--", label="p5 / p95")
        ax.plot(x, bs["media"], color="#15324a", lw=2.4, marker="o", ms=4, label="média")
        ax.margins(x=0)                                   # sem respiro lateral (eixo x)
        ax.set_xticks(x)
        ax.set_xticklabels(_fmt_safras(bs["safra"]), rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8, ncol=3, framealpha=0.9, loc="upper left")
        ax.set_title(f"'{self.label(feature)}' ao longo do tempo — percentis por safra",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(alpha=0.12)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def _plot_share_timeseries(self, feature, time_col, sample, figsize, dpi, save_path, ax,
                               all_samples=False):
        import matplotlib.colors as mcolors
        sh = self.variable_share_by_safra(feature, time_col, sample, all_samples=all_samples)
        fig, ax = _new_ax((figsize[0], 3.6), dpi, ax)
        cats = [c for c in sh.columns if c != "safra"]
        if sh.empty or not cats:
            ax.text(0.5, 0.5, "sem dados por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        x = list(range(len(sh)))
        cmap = mcolors.LinearSegmentedColormap.from_list("sc", ["steelblue", "crimson"])
        base = [c for c in cats if c not in ("outras", "(faltante)")]
        colors = []
        for c in cats:
            if c == "(faltante)":
                colors.append("#c98a8a")
            elif c == "outras":
                colors.append("#b9c0cb")
            else:
                colors.append(cmap(base.index(c) / max(len(base) - 1, 1)))
        ys = [sh[c].fillna(0).to_numpy() for c in cats]
        ax.stackplot(x, ys, labels=cats, colors=colors, alpha=0.92)
        ax.set_ylim(0, 100); ax.margins(x=0)
        ax.set_xticks(x)
        ax.set_xticklabels(_fmt_safras(sh["safra"]), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% da safra")
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.9, title="categoria")
        ax.set_title(f"'{self.label(feature)}' — share por categoria no tempo",
                     fontsize=11, fontweight="bold", color="#15324a")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def variable_faixa_share_by_safra(self, feature, time_col=None, sample=None,
                                      max_n_bins=6, min_bin_size=0.05, bins=None,
                                      all_samples=False) -> pd.DataFrame:
        """% de cada **faixa/categoria** da variável por safra (mês).

        Usa as MESMAS faixas da análise (:meth:`_resolve_bins`) — vale para
        variáveis numéricas (faixas) e categóricas (grupos). Colunas: ``safra`` +
        uma por faixa (% da safra). Linhas sem faixa (ex.: faltantes fora dos
        bins) entram em ``(faltante)``.

        ``bins`` (opcional): lista de bins já resolvida a usar no lugar de
        :meth:`_resolve_bins` — útil para forçar as faixas do optimal binning
        (ver :meth:`plot_variable_optbin_share_timeseries`).
        ``all_samples=True`` usa **toda a base** (todas as amostras/safras), não só
        a amostra de referência."""
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col.")
        if time_col not in self.df.columns:
            raise ValueError(f"Coluna de tempo '{time_col}' não existe no DataFrame.")
        if bins is None:
            bins, _kind = self._resolve_bins(feature, max_n_bins, min_bin_size, None, sample)
        ordem = [self._bin_label(feature, b) for b in (bins or [])]
        if not ordem:
            return pd.DataFrame(columns=["safra"])
        # faixa de cada linha numa passada; sem faixa → "(faltante)"
        linhas = self._rows_mask(sample, all_rows=all_samples)
        cod, rot = self._safra_codes(time_col)
        bc = self._feature_bin_codes(feature, bins)
        ok = linhas & (cod >= 0)
        if not ok.any():
            return pd.DataFrame(columns=["safra"])
        rot_faixa = np.asarray(ordem + ["(faltante)"], dtype=object)
        tab = pd.crosstab(np.asarray(rot, dtype=object)[cod[ok]],
                          rot_faixa[np.where(bc[ok] >= 0, bc[ok], len(ordem))])
        pct = tab.div(tab.sum(axis=1), axis=0) * 100
        cols = [c for c in dict.fromkeys(ordem) if c in pct.columns]
        if "(faltante)" in pct.columns and "(faltante)" not in cols:
            cols.append("(faltante)")
        pct = pct[cols].round(1).sort_index()
        pct.index.name = "safra"
        return pct.reset_index()

    def plot_variable_faixa_share_timeseries(self, feature, time_col=None, sample=None,
                                             max_n_bins=6, min_bin_size=0.05,
                                             figsize=(8.6, 3.4), dpi=150,
                                             save_path=None, ax=None, bins=None,
                                             titulo=None, legend_title="faixa"):
        """**% de cada faixa/categoria da variável ao longo do tempo** — uma LINHA
        por faixa. Complementa o gráfico de comportamento (percentis/share): mostra
        como a composição da variável migra entre as faixas ao longo das safras.

        ``bins``/``titulo`` (opcionais): forçam as faixas e o título — usados por
        :meth:`plot_variable_optbin_share_timeseries`."""
        import matplotlib.colors as mcolors
        sh = self.variable_faixa_share_by_safra(feature, time_col, sample,
                                                max_n_bins, min_bin_size, bins=bins)
        fig, ax = _new_ax((figsize[0], 3.6), dpi, ax)
        cats = [c for c in sh.columns if c != "safra"]
        if sh.empty or not cats:
            ax.text(0.5, 0.5, "sem dados por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        x = list(range(len(sh)))
        cmap = mcolors.LinearSegmentedColormap.from_list("sc", ["steelblue", "crimson"])
        base = [c for c in cats if c not in ("outras", "(faltante)")]
        for c in cats:
            if c == "(faltante)":
                cor = "#c98a8a"
            elif c == "outras":
                cor = "#b9c0cb"
            else:
                cor = cmap(base.index(c) / max(len(base) - 1, 1)) if c in base else "#889"
            ax.plot(x, sh[c].fillna(0).to_numpy(), marker="o", ms=3.5, lw=1.8,
                    color=cor, label=c)
        ax.set_ylim(bottom=0); ax.margins(x=0)
        ax.set_xticks(x)
        ax.set_xticklabels(_fmt_safras(sh["safra"]), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% da safra")
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.9, title=legend_title)
        ax.set_title(titulo or f"'{self.label(feature)}' — % de cada faixa ao longo do tempo",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(alpha=0.12)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def _optbin_numeric_bins(self, feature, sample=None, max_n_bins=5, min_bin_size=0.05):
        """Faixas do **OPTIMAL BINNING** de uma variável NUMÉRICA — sempre roda o
        optbinning na amostra ``sample`` (default: referência/DES), IGNORANDO eventuais
        bins manuais. Retorna a lista de bins numéricos (+ ``na`` se houver faltantes)
        ou ``[]`` quando não dá para binar. (:meth:`_resolve_bins` respeita bins
        manuais; este não.)"""
        if OptimalBinning is None:
            raise ImportError("optbinning não instalado. Rode: pip install optbinning")
        fit = self._frame(sample, cols=[feature, self.target])
        if self._detect_kind(feature, fit) != "num":
            return []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # o aviso de nível raro já sai no ranking
            binaria = self._bins_binaria(feature, fit, "num")
        if binaria is not None:
            return binaria
        x = fit[feature].to_numpy(dtype="float64")
        y = fit[self.target].to_numpy(dtype="float64")
        ok = ~np.isnan(y)
        x, y = x[ok], y[ok]
        x_obs = x[~np.isnan(x)]
        if len(y) < 4 or x_obs.size == 0 or x_obs.min() == x_obs.max():
            return []
        if self.task_type == "classification":
            b = OptimalBinning(name=feature, dtype="numerical", max_n_bins=max_n_bins,
                               min_bin_size=min_bin_size, monotonic_trend="auto_asc_desc")
            cortes = _fit_optbinning_splits(b, x, y.astype(int))
        else:
            b = ContinuousOptimalBinning(name=feature, dtype="numerical", max_n_bins=max_n_bins,
                                         min_bin_size=min_bin_size, monotonic_trend="auto_asc_desc")
            cortes = _fit_optbinning_splits(b, x, y)
        if not cortes:
            return []
        edges = [-np.inf, *cortes, np.inf]
        bins = [{"kind": "num", "lo": edges[i], "hi": edges[i + 1]}
                for i in range(len(edges) - 1)]
        if fit[feature].isna().any():
            bins.append({"kind": "na"})
        return bins

    def plot_variable_optbin_share_timeseries(self, feature, time_col=None, sample=None,
                                              max_n_bins=5, min_bin_size=0.05,
                                              figsize=(8.6, 3.4), dpi=150,
                                              save_path=None, ax=None):
        """**Distribuição das categorias do OPTIMAL BINNING ao longo do tempo**
        (só variáveis NUMÉRICAS): % de cada faixa gerada pelo binning ótimo por
        safra, uma linha por faixa. Sempre usa o optbinning (ignora bins manuais),
        para acompanhar a estabilidade das faixas do algoritmo no tempo."""
        if self._detect_kind(feature) != "num":
            fig, ax = _new_ax((figsize[0], 3.6), dpi, ax)
            ax.text(0.5, 0.5, "apenas para variáveis numéricas", ha="center",
                    va="center", transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        bins = self._optbin_numeric_bins(feature, sample, max_n_bins, min_bin_size)
        titulo = f"'{self.label(feature)}' — faixas do optimal binning ao longo do tempo"
        return self.plot_variable_faixa_share_timeseries(
            feature, time_col, sample, max_n_bins, min_bin_size, figsize, dpi,
            save_path, ax, bins=bins, titulo=titulo, legend_title="faixa (optbin)")

    def _faixas_categoricas_ordenadas(self, feature, sample, max_n_bins, min_bin_size):
        """Faixas/grupos de uma variável CATEGÓRICA para o gráfico acumulado. Numa
        variável criada a partir das faixas de outra (``derived_bins``), segue a ordem
        das faixas de origem (menor → maior, ``(faltante)`` no fim) em vez da ordem de
        risco; nas demais, a ordem de :meth:`_resolve_bins`."""
        bins, _kind = self._resolve_bins(feature, max_n_bins, min_bin_size, None, sample)
        bins = list(bins or [])
        meta = self.var_meta.get(feature) or {}
        src, dbins = meta.get("derived_from"), meta.get("derived_bins")
        if src and dbins:
            pos = {self._bin_label(src, b): i for i, b in enumerate(dbins)}
            fim = len(pos) + 1

            def chave(b):
                cats = [str(c) for c in b.get("cats", [])] if b.get("kind") == "cat" else []
                return min((pos.get(c, fim) for c in cats), default=fim + 1)
            bins.sort(key=chave)
        return bins

    def plot_variable_optbin_cumshare_timeseries(self, feature, time_col=None, sample=None,
                                                 max_n_bins=5, min_bin_size=0.05,
                                                 figsize=(11.5, 4.2), dpi=150,
                                                 save_path=None, ax=None, all_samples=False):
        """**Distribuição ACUMULADA das faixas do OPTIMAL BINNING ao longo do tempo**
        (só variáveis NUMÉRICAS): área EMPILHADA das %s de cada faixa por safra, da
        **primeira faixa (base) até a última (topo)**, somando 100%. Deixa ver como
        a composição da variável migra entre as faixas no tempo. Sempre usa o
        optbinning (ignora bins manuais).

        ``all_samples=True`` (padrão na UI) = análise de **estabilidade**: as faixas do
        optbin são **fixadas na DES** (referência — yardstick estável, como no PSI) e a
        **distribuição** é observada sobre **toda a população** (todas as amostras/safras).
        Com ``all_samples=False``, faixas e distribuição usam a amostra ``sample``."""
        import matplotlib.colors as mcolors
        # Estabilidade: as faixas do optbin são um YARDSTICK. Quando o gráfico cobre
        # toda a população (all_samples), elas são FIXADAS na referência (DES) e só a
        # DISTRIBUIÇÃO (abaixo) varre todas as safras/amostras; fora disso, faixas e
        # distribuição saem da mesma amostra.
        ref = self.ref_sample if all_samples else sample
        categorica = self._detect_kind(feature) == "cat"
        if categorica:
            # categórica (ex.: variável criada a partir das faixas de outra, com a
            # faixa "(faltante)"): empilha as próprias faixas/grupos da análise
            bins = self._faixas_categoricas_ordenadas(feature, ref, max_n_bins, min_bin_size)
        else:
            bins = self._optbin_numeric_bins(feature, ref, max_n_bins, min_bin_size)
        sh = self.variable_faixa_share_by_safra(feature, time_col, sample, max_n_bins,
                                                min_bin_size, bins=bins, all_samples=all_samples)
        fig, ax = _new_ax(figsize, dpi, ax)
        cats = [c for c in sh.columns if c != "safra"]
        if sh.empty or not cats:
            ax.text(0.5, 0.5, "sem dados por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        x = list(range(len(sh)))
        # cores em gradiente (steelblue→crimson) na ordem das faixas; (faltante) cinza
        base = [c for c in cats if c not in ("outras", "(faltante)")]
        cmap = mcolors.LinearSegmentedColormap.from_list("sc", ["steelblue", "crimson"])
        cores = []
        for c in cats:
            if c == "(faltante)":
                cores.append("#c98a8a")
            elif c == "outras":
                cores.append("#b9c0cb")
            else:
                cores.append(cmap(base.index(c) / max(len(base) - 1, 1)) if c in base else "#889")
        Y = [sh[c].fillna(0).to_numpy() for c in cats]
        ax.stackplot(x, *Y, labels=cats, colors=cores, alpha=0.9,
                     edgecolor="white", linewidth=0.3)
        ax.set_ylim(0, 100); ax.margins(x=0)
        ax.set_xticks(x)
        ax.set_xticklabels(_fmt_safras(sh["safra"]), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% acumulado da safra")
        # legenda no canto superior esquerdo, logo abaixo do título (dentro do gráfico);
        # caixa branca semiopaca p/ ficar legível sobre a área empilhada.
        leg = ax.legend(fontsize=8, loc="upper left", framealpha=0.92,
                        title="faixa" if categorica else "faixa (optbin)",
                        labelspacing=0.3, borderpad=0.5)
        leg.get_frame().set_edgecolor("#cccccc")
        ax.set_title(f"'{self.label(feature)}' — distribuição acumulada das faixas"
                     f"{'' if categorica else ' do optimal binning'} ao longo do tempo",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(alpha=0.12, axis="y")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_variable_psi_by_safra(self, feature, time_col=None, figsize=(9.6, 4.4),
                                   dpi=150, save_path=None, ax=None):
        """PSI da variável por safra vs DES (barras coloridas)."""
        ps = self.variable_psi_by_safra(feature, time_col)
        fig, ax = _new_ax(figsize, dpi, ax)
        if ps.empty:
            ax.text(0.5, 0.5, "sem PSI por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        x = list(range(len(ps)))
        cor = ["#1aa64b" if p < 0.10 else "#caa000" if p < 0.25 else "#d6453e"
               for p in ps["psi"]]
        ax.bar(x, ps["psi"], color=cor, alpha=0.92, width=0.78)
        for x0, p in zip(x, ps["psi"]):
            ax.text(x0, p, f"{p:.2f}", ha="center", va="bottom", fontsize=7, color="#555")
        # guia de alerta do PSI (sempre visível, mesmo com PSI pequeno)
        ax.axhline(0.10, color="#caa000", lw=1.2, ls="--", label="alerta (0,10)")
        ax.axhline(0.25, color="#d6453e", lw=1.2, ls="--", label="crítico (0,25)")
        ax.set_xticks(x); ax.set_xticklabels(_fmt_safras(ps["safra"]), rotation=45, ha="right", fontsize=8)
        ax.set_xlim(-0.7, len(ps) - 0.3)
        ax.set_ylim(0, max(float(np.nanmax(ps["psi"])) * 1.16 + 0.02, 0.28))
        ax.set_ylabel("PSI")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
        ax.set_title(f"PSI de '{self.label(feature)}' por safra vs DES",
                     fontsize=11, fontweight="bold", color="#15324a")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_variable_inversion_by_sample(self, feature, max_n_bins=6, min_bin_size=0.05,
                                          figsize=(7.6, 4.0), dpi=150, save_path=None, ax=None,
                                          target_ylim01=False):
        """Risco de cada faixa por amostra; cruzamentos = inversão da ordem de risco.

        ``target_ylim01=True`` fixa o eixo do risco em [0,1] na regressão (LGD)."""
        inv = self.variable_inversion(feature, max_n_bins=max_n_bins, min_bin_size=min_bin_size)
        fig, ax = _new_ax(figsize, dpi, ax)
        s = inv.get("series")
        if not s or not inv["ordered"]:
            ax.text(0.5, 0.5, "menos de 2 faixas", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        xs = s["xs_sample"]; x = list(range(len(xs)))
        cmap = _cmap("RdYlGn_r"); k = len(inv["ordered"])
        _rot = _sem_prefixo_da_variavel(s["labels"], feature)
        for rank, i in enumerate(inv["ordered"]):
            ax.plot(x, s["ser_sample"][i], marker="o", lw=1.9, ms=5.5,
                    color=cmap(rank / (k - 1) if k > 1 else 0.5),
                    markeredgecolor="#33424f", markeredgewidth=0.6,
                    label=_rot[i])
        ax.set_xticks(x); ax.set_xticklabels(xs, fontsize=9)
        ax.set_ylabel("risco médio"); ax.set_xlabel("amostra")
        if target_ylim01 and self.task_type != "classification":
            ax.set_ylim(0.0, 1.0)
        ax.set_title(f"'{self.label(feature)}' — risco das faixas por amostra",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        ax.legend(fontsize=7.5, ncol=max(1, min(k, 3)), loc="best", framealpha=0.85)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_variable_inversion_by_safra(self, feature, time_col=None, sample=None,
                                         max_n_bins=6, min_bin_size=0.05, min_n=20,
                                         figsize=(9.6, 4.0), dpi=150, save_path=None, ax=None,
                                         target_ylim01=False):
        """Risco de cada faixa por safra; safras com inversão ficam sombreadas.

        ``target_ylim01=True`` fixa o eixo do risco em [0,1] na regressão (LGD)."""
        inv = self.variable_inversion(feature, time_col, sample, max_n_bins,
                                      min_bin_size, min_n)
        fig, ax = _new_ax(figsize, dpi, ax)
        s = inv.get("series")
        if not s or not s["xs_safra"]:
            ax.text(0.5, 0.5, "sem dados por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        xs = s["xs_safra"]; x = list(range(len(xs))); ordered = inv["ordered"]
        for j in x:
            vals = {i: s["ser_safra"][i][j] for i in range(len(s["labels"]))}
            n_inv, npp = _count_inversions(ordered, vals)
            if npp and n_inv:
                ax.axvspan(j - 0.5, j + 0.5, color="#d6453e", alpha=0.08, lw=0)
        cmap = _cmap("RdYlGn_r"); k = len(ordered)
        _rot = _sem_prefixo_da_variavel(s["labels"], feature)
        for rank, i in enumerate(ordered):
            ax.plot(x, s["ser_safra"][i], marker="o", lw=1.7, ms=4.5,
                    color=cmap(rank / (k - 1) if k > 1 else 0.5),
                    markeredgecolor="#33424f", markeredgewidth=0.5, label=_rot[i])
        ax.set_xticks(x); ax.set_xticklabels(_fmt_safras(xs), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("risco médio"); ax.set_xlabel("safra")
        if target_ylim01 and self.task_type != "classification":
            ax.set_ylim(0.0, 1.0)
        ax.set_title(f"'{self.label(feature)}' — risco das faixas por safra"
                     "  ·  faixas vermelhas = inversão",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        ax.legend(fontsize=7.5, ncol=max(1, min(k, 3)), loc="best", framealpha=0.85)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_variable_risk_by_safra(self, feature, time_col=None, sample=None,
                                    max_n_bins=5, min_bin_size=0.05, min_n=20,
                                    figsize=(9.0, 3.8), dpi=150, save_path=None, ax=None):
        """Comportamento da variável ao longo do tempo: o **risco** (event_rate/alvo na
        classificação, alvo médio na regressão) de cada bin/categoria por safra.

        - **Numérica**: usa os mesmos bins do ranking (``n_bins``) — risco de cada
          faixa por safra.
        - **Categórica**: traz a **alvo por categoria** por safra, sem reagrupar (top
          categorias; respeita os grupos manuais se definidos). Sem ordem de risco
          imposta (categorias não têm ordem intrínseca)."""
        time_col = time_col or self.date_col
        fig, ax = _new_ax(figsize, dpi, ax)
        base = self._frame(sample) if sample else self.df
        if not time_col or time_col not in base.columns:
            ax.text(0.5, 0.5, "defina uma coluna de safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        kind = self._detect_kind(feature, base)
        ordered_by_risk = True
        groups = []   # [(label, máscara booleana sobre 'base')]
        if kind == "num" or self.manual_bins(feature):
            bins, _k = self._resolve_bins(feature, max_n_bins, min_bin_size, sample=sample)
            for b in bins:
                groups.append((self._bin_label(feature, b), self._mask_in(base, feature, b)))
            ordered_by_risk = (kind == "num")
        else:
            vc = base[feature].dropna().astype(str).value_counts()
            top = list(vc.index[:8])
            for c in top:
                groups.append((str(c), base[feature].astype(str) == str(c)))
            if len(vc) > len(top):
                groups.append(("(outras)", base[feature].astype(str).isin(vc.index[len(top):])))
            if base[feature].isna().any():
                groups.append(("(faltante)", base[feature].isna()))
            ordered_by_risk = False
        if not groups:
            ax.text(0.5, 0.5, "sem bins/categorias", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig

        safra = pd.to_datetime(base[time_col], errors="coerce").dt.to_period("M")
        cod, pers = pd.factorize(safra, sort=True, use_na_sentinel=True)   # NaT → -1
        pers = list(pers)
        if not pers:
            ax.text(0.5, 0.5, "sem dados por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        xs = [str(p) for p in pers]; x = list(range(len(xs)))
        # risco grupo × safra por contagem (np.bincount), sem máscara por safra
        y_b = pd.to_numeric(base[self.target], errors="coerce").to_numpy(dtype="float64")
        P = len(pers)
        series = []
        for label, gmask in groups:
            gm = np.asarray(gmask.to_numpy(dtype=bool, na_value=False)
                            if hasattr(gmask, "to_numpy") else gmask, dtype=bool)
            sel = gm & (cod >= 0)
            n_p = np.bincount(cod[sel], minlength=P)
            ok = sel & ~np.isnan(y_b)
            soma = np.bincount(cod[ok], weights=y_b[ok], minlength=P)
            cont = np.bincount(cod[ok], minlength=P)
            ys = [float(soma[k] / cont[k]) if (n_p[k] >= min_n and cont[k] > 0) else np.nan
                  for k in range(P)]
            series.append((label, ys))

        is_clf = self.task_type == "classification"
        ylabel = self._risk_word if is_clf else "alvo médio"
        k = len(series)
        if ordered_by_risk:
            cmap = _cmap("RdYlGn_r")
            means = [np.nanmean(ys) if np.any(np.isfinite(ys)) else np.inf
                     for _, ys in series]
            order = sorted(range(k), key=lambda i: means[i])
            colors = {i: cmap(rank / (k - 1) if k > 1 else 0.5)
                      for rank, i in enumerate(order)}
        else:
            cmap = _cmap("tab10")
            colors = {i: cmap((i % 10) / 9) for i in range(k)}
        for i, (label, ys) in enumerate(series):
            ax.plot(x, ys, marker="o", lw=1.7, ms=4.5, color=colors[i],
                    markeredgecolor="#33424f", markeredgewidth=0.5, label=label)
        ax.set_xticks(x); ax.set_xticklabels(_fmt_safras(xs), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel); ax.set_xlabel("safra")
        titulo = "risco das faixas" if kind == "num" else f"{ylabel} por categoria"
        ax.set_title(f"'{self.label(feature)}' — {titulo} por safra",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        ax.legend(fontsize=7.5, ncol=max(1, min(k, 3)), loc="best", framealpha=0.85)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # B) Seleção / categorização de variáveis
    # ------------------------------------------------------------------
    def include(self, feature):
        if feature not in self.candidates:
            raise ValueError(f"'{feature}' não é variável candidata.")
        self.included.add(feature)
        return self

    def exclude(self, feature):
        self.included.discard(feature)
        return self

    def include_all(self):
        self.included = set(self.candidates)
        return self

    def clear_features(self):
        self.included = set()
        return self

    def derived_features(self) -> list:
        """Variáveis categóricas criadas via :meth:`create_categorical`."""
        return [n for n, m in self.var_meta.items() if m.get("derived_from")]

    def remove_derived(self, name) -> str:
        """Exclui UMA variável criada (:meth:`create_categorical` /
        :meth:`create_scorecard_dummies`) **da base**: tira do DataFrame, das
        candidatas, da seleção, do ``var_meta`` e dos rótulos. Só vale para
        variáveis criadas — as colunas originais da base não são apagadas.

        Recusa (``ValueError``) quando a variável está no **modelo treinado** (a
        escoragem dependeria dela — re-treine sem ela antes) ou quando outra
        variável criada foi derivada dela (exclua aquela primeiro)."""
        meta = self.var_meta.get(name, {})
        if not meta.get("derived_from"):
            raise ValueError(f"'{name}' não é uma variável criada — só as variáveis "
                             "criadas na categorização podem ser excluídas da base.")
        if self.model is not None and name in (self.model_features or []):
            raise ValueError(f"'{self.label(name)}' está no modelo treinado — re-treine "
                             "sem ela antes de excluí-la da base.")
        filhas = [n for n, m in self.var_meta.items() if m.get("derived_from") == name]
        if filhas:
            raise ValueError(f"'{self.label(name)}' é a origem de {filhas} — exclua "
                             "essas variáveis antes.")
        if name in self.df.columns:
            self.df.drop(columns=name, inplace=True)
        if name in self.candidates:
            self.candidates.remove(name)
        self.included.discard(name)
        self.var_meta.pop(name, None)
        self.feature_labels.pop(name, None)
        self._invalidate_bins(name)
        return name

    def clear_derived(self) -> list:
        """Remove **todas** as variáveis criadas via :meth:`create_categorical`
        (reset): tira do DataFrame, das candidatas, da seleção e do ``var_meta``.
        Devolve os nomes removidos."""
        removidas = self.derived_features()
        for n in removidas:
            if n in self.df.columns:
                self.df.drop(columns=n, inplace=True)
            if n in self.candidates:
                self.candidates.remove(n)
            self.included.discard(n)
            self.var_meta.pop(n, None)
            self.feature_labels.pop(n, None)
        if removidas:
            self._invalidate_bins(*removidas)   # invalida só as derivadas removidas
        return removidas

    def set_category(self, feature, categoria):
        """Categoriza a variável (ex.: 'manter', 'revisar', 'descartar')."""
        meta = self.var_meta.setdefault(feature, {})
        meta["categoria"] = categoria
        meta.pop("motivo", None)   # 'motivo' só vale para categorização automática
        return self

    # ---- bins manuais ("categorizar na mão", como nos projetos de árvore) ----
    def _parse_bin_spec(self, feature, text):
        """Interpreta a especificação de bins manuais digitada na UI.

        * **Numérica** — lista de cortes: ``"0.7, 0.9"`` → ``[0.7, 0.9]``
          (gera as faixas ``(-inf,0.7] (0.7,0.9] (0.9,inf]``).
        * **Categórica** — grupos separados por ``;`` e categorias por ``,``:
          ``"a, b; c"`` → ``[["a", "b"], ["c"]]``.

        Devolve ``None`` quando o texto é vazio (volta ao binning ótimo)."""
        text = (text or "").strip()
        if not text:
            return None
        if self._detect_kind(feature) == "num":
            cuts = []
            for tok in text.replace(";", ",").split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    cuts.append(float(tok))
                except ValueError:
                    raise ValueError(
                        f"corte inválido: {tok!r}. Use números com ponto decimal "
                        f"separados por vírgula (ex.: 0.7, 0.9).") from None
            return sorted(set(cuts)) or None
        grupos = []
        for grp in text.split(";"):
            cats = [c.strip() for c in grp.split(",") if c.strip()]
            if cats:
                grupos.append(cats)
        return grupos or None

    def set_manual_bins(self, feature, spec, missing=None):
        """Define **bins manuais** para a variável, sobrepondo o binning ótimo em
        toda a análise univariada (tabela, IV, logodds/WoE, PSI, inversão).

        ``spec`` pode ser o texto da UI (ver :meth:`_parse_bin_spec`), uma lista
        já parseada (cortes numéricos ou grupos categóricos), ou ``None``/``""``
        para limpar e voltar ao binning ótimo.

        ``missing`` (opcional): destino dos faltantes — ver :meth:`set_missing_bin`
        (``None`` mantém o destino atual)."""
        if feature not in self.candidates:
            raise ValueError(f"'{feature}' não é variável candidata.")
        splits = self._parse_bin_spec(feature, spec) if isinstance(spec, (str, type(None))) \
            else (list(spec) or None)
        meta = self.var_meta.setdefault(feature, {})
        if splits:
            meta["splits"] = splits
        else:
            meta.pop("splits", None)
            meta.pop("ordinal_scorecard", None)   # dummies/ordinal só existem sobre bins manuais
            meta.pop("scorecard_dummies", None)
            meta.pop("na_destino", None)          # idem o destino dos faltantes
        self._invalidate_bins(feature)   # só ESTA variável re-bina; demais ficam quentes
        if splits and missing is not None:
            self.set_missing_bin(feature, missing)
        return self

    def clear_manual_bins(self, feature):
        """Remove os bins manuais da variável (volta ao binning ótimo) — e, com
        eles, a codificação ordinal de scorecard e o destino dos faltantes, que
        dependem das faixas manuais."""
        meta = self.var_meta.get(feature, {})
        meta.pop("splits", None)
        meta.pop("ordinal_scorecard", None)
        meta.pop("scorecard_dummies", None)
        meta.pop("na_destino", None)
        self._invalidate_bins(feature)   # só ESTA variável volta ao ótimo
        return self

    # ---- faltantes na categorização manual ----
    def set_missing_bin(self, feature, destino="separado"):
        """Escolhe **onde os faltantes (NaN) ficam** na categorização manual.

        ``destino``:

        * ``"separado"`` (padrão) — faixa própria "(faltante)", como no binning ótimo;
        * ``"pior"`` / ``"melhor"`` — juntam-se à faixa de maior / menor risco
          na referência (recalculado a cada binning);
        * ``int`` — juntam-se à faixa de índice ``destino`` (0 = primeira, na
          ordem de :meth:`manual_bins_faixas`).

        A faixa escolhida vira ``"<faixa> + faltante"`` em toda a análise (tabela,
        IV, PSI, WoE, ordinal de scorecard) e na escoragem — inclusive para NaN
        que só aparecem fora da referência. Só com bins manuais."""
        if not self.manual_bins(feature):
            raise ValueError(
                f"'{self.label(feature)}' não tem categorização manual — o destino "
                "dos faltantes só se escolhe sobre bins manuais.")
        if isinstance(destino, (bool, np.bool_)):
            raise ValueError("destino deve ser 'separado', 'pior', 'melhor' ou o índice da faixa.")
        if isinstance(destino, (int, np.integer)):
            n = len(self.manual_bins_faixas(feature))
            if not 0 <= int(destino) < n:
                raise ValueError(f"faixa {int(destino)} inexistente — a variável tem "
                                 f"{n} faixas (índices 0 a {n - 1}).")
            destino = int(destino)
        elif destino not in ("separado", "pior", "melhor"):
            raise ValueError("destino deve ser 'separado', 'pior', 'melhor' ou o índice "
                             f"da faixa (recebi {destino!r}).")
        meta = self.var_meta.setdefault(feature, {})
        if destino == "separado":
            meta.pop("na_destino", None)
        else:
            meta["na_destino"] = destino
        self._invalidate_bins(feature)
        return self

    def missing_bin(self, feature):
        """Destino atual dos faltantes: ``"separado"``, ``"pior"``, ``"melhor"`` ou índice."""
        return self.var_meta.get(feature, {}).get("na_destino", "separado")

    def manual_bins_faixas(self, feature) -> list:
        """Rótulos das faixas manuais (sem a de faltantes e sem "+ faltante"), na
        ordem dos índices aceitos por :meth:`set_missing_bin`."""
        if not self.manual_bins(feature):
            return []
        bins, _ = self._resolve_bins(feature, sample=self.ref_sample)
        return [self._bin_label(feature, {k: v for k, v in b.items() if k != "include_na"})
                for b in bins if b["kind"] != "na"]

    def missing_info(self, feature) -> dict:
        """Resumo dos faltantes da variável na referência e de onde eles caem:
        ``n``, ``pct`` (da referência), ``taxa`` (risco dos faltantes), ``destino``
        (como configurado) e ``faixa`` (rótulo da faixa que os recebe)."""
        ref = self._frame(self.ref_sample, cols=[feature, self.target])
        na = ref[feature].isna().to_numpy()
        n, tot = int(na.sum()), len(ref)
        taxa = self._risco(ref[self.target].to_numpy(dtype="float64")[na]) if n else float("nan")
        if not self.manual_bins(feature):
            # binning ótimo: faltante sempre em faixa própria — sem rodar o optbinning
            return {"n": n, "pct": (n / tot if tot else float("nan")), "taxa": taxa,
                    "destino": "separado", "faixa": "(faltante)" if n else None}
        bins, _ = self._resolve_bins(feature, sample=self.ref_sample)
        alvo = next((b for b in bins if b.get("include_na")), None)
        if alvo is not None:
            faixa = self._bin_label(feature, alvo)
        elif any(b["kind"] == "na" for b in bins):
            faixa = "(faltante)"
        else:
            faixa = None
        return {"n": n, "pct": (n / tot if tot else float("nan")), "taxa": taxa,
                "destino": self.missing_bin(feature), "faixa": faixa}

    # ---- dummies de scorecard (só categorização manual) ----
    def set_scorecard_dummies(self, feature, ativo=True):
        """Liga (ou desliga) as **dummies de scorecard** na variável.

        Cada faixa manual vira uma coluna 0/1, e a **pior faixa** (maior risco
        na referência/DES) é a **referência** — a coluna omitida. Com alvo 1 = mau
        (PD), cada coeficiente mede o quanto a faixa é melhor que a pior e sai
        **negativo**: no scorecard a pior faixa vale 0 pontos e as demais somam
        pontos. Vale nos dois ``transform`` (``raw`` e ``woe``); os termos aparecem
        como ``variável = faixa`` nos coeficientes.

        Só para variáveis com **bins manuais** (:meth:`set_manual_bins`). Limpar
        os bins manuais desliga a opção. A pior faixa é recalculada a cada treino;
        valor fora das faixas (categoria nova) cai na referência (0 em todas)."""
        if feature not in self.candidates:
            raise ValueError(f"'{feature}' não é variável candidata.")
        meta = self.var_meta.setdefault(feature, {})
        meta.pop("ordinal_scorecard", None)            # chave da 0.0.14 (ordinal)
        if ativo:
            if not self.manual_bins(feature):
                raise ValueError(
                    f"'{self.label(feature)}' não tem categorização manual — as "
                    "dummies de scorecard só valem sobre bins manuais "
                    "(defina-os com set_manual_bins / modo Manual da aba Análise).")
            meta["scorecard_dummies"] = True
        else:
            meta.pop("scorecard_dummies", None)
        return self

    def scorecard_dummies(self, feature) -> bool:
        """``True`` se a variável entra no modelo como dummies de scorecard.
        (Aceita a chave ``ordinal_scorecard`` de modelos salvos na 0.0.14.)"""
        meta = self.var_meta.get(feature, {})
        return bool((meta.get("scorecard_dummies") or meta.get("ordinal_scorecard"))
                    and self.manual_bins(feature))

    def scorecard_dummy_features(self, features=None) -> list:
        """Variáveis (de ``features`` ou das candidatas) com dummies de scorecard."""
        feats = list(features) if features is not None else list(self.candidates)
        return [f for f in feats if self.scorecard_dummies(f)]

    def _faixas_por_risco(self, feature) -> list:
        """``[(bin, risco, n, ordem), ...]`` na ordem das faixas: ordem 0 = maior
        risco na referência (a referência das dummies). Faixa sem observação na
        referência (risco NaN) vai para o topo — conservador."""
        ref = self._frame(self.ref_sample, cols=[feature, self.target])
        bins, _kind = self._resolve_bins(feature, sample=self.ref_sample)
        y_all = ref[self.target].to_numpy(dtype="float64")
        riscos, ns = [], []
        for m in _bin_masks(ref[feature], bins):
            riscos.append(self._risco(y_all[m]))
            ns.append(int(m.sum()))
        # pior primeiro: NaN (-inf na chave) empata no topo; empate mantém a ordem
        ordem = sorted(range(len(bins)),
                       key=lambda i: -(riscos[i] if np.isfinite(riscos[i]) else np.inf))
        pos = {i: c for c, i in enumerate(ordem)}
        return [(bins[i], riscos[i], ns[i], pos[i]) for i in range(len(bins))]

    def _dummy_spec(self, feature) -> dict:
        """Especificação p/ o :class:`ScorecardDummyEncoder`: faixas, rótulos e o
        índice da referência (pior faixa)."""
        faixas = self._faixas_por_risco(feature)
        return {"bins": [b for b, _r, _n, _o in faixas],
                "labels": [self._bin_label(feature, b) for b, _r, _n, _o in faixas],
                "ref": next(i for i, (_b, _r, _n, o) in enumerate(faixas) if o == 0)}

    def scorecard_table(self, feature) -> pd.DataFrame:
        """Tabela das dummies de scorecard da variável: ``faixa``, ``n`` e risco na
        referência e o ``papel`` no modelo (``referência`` = pior faixa, omitida;
        ``dummy`` = coluna 0/1), ordenada da pior para a melhor faixa."""
        if not self.manual_bins(feature):
            raise ValueError(f"'{self.label(feature)}' não tem categorização manual.")
        col_risco = "taxa_maus" if self.task_type == "classification" else "alvo_medio"
        rows = [{"faixa": self._bin_label(feature, b), "n": n,
                 col_risco: (round(float(r), 6) if np.isfinite(r) else np.nan),
                 "papel": "referência" if o == 0 else "dummy", "_o": o}
                for b, r, n, o in self._faixas_por_risco(feature)]
        return (pd.DataFrame(rows).sort_values("_o").drop(columns="_o")
                .reset_index(drop=True))

    # aliases da 0.0.14 (codificação ORDINAL, substituída pelas dummies)
    def set_scorecard_ordinal(self, feature, ativo=True):
        """Descontinuado: use :meth:`set_scorecard_dummies` (a opção de scorecard
        agora gera dummies com a pior faixa como referência)."""
        warnings.warn("set_scorecard_ordinal foi substituído por set_scorecard_dummies "
                      "(dummies com a pior faixa como referência).", DeprecationWarning,
                      stacklevel=2)
        return self.set_scorecard_dummies(feature, ativo)

    def scorecard_ordinal(self, feature) -> bool:
        """Descontinuado: use :meth:`scorecard_dummies`."""
        return self.scorecard_dummies(feature)

    def scorecard_ordinal_features(self, features=None) -> list:
        """Descontinuado: use :meth:`scorecard_dummy_features`."""
        return self.scorecard_dummy_features(features)

    def scorecard_ordinal_table(self, feature) -> pd.DataFrame:
        """Descontinuado: use :meth:`scorecard_table`."""
        return self.scorecard_table(feature)

    def manual_bins(self, feature):
        """Bins manuais da variável (cortes ou grupos), ou ``None`` se ótimo."""
        return self.var_meta.get(feature, {}).get("splits")

    def manual_bins_spec(self, feature) -> str:
        """Texto da UI equivalente aos bins manuais atuais (vazio se ótimo)."""
        splits = self.manual_bins(feature)
        if not splits:
            return ""
        if self._detect_kind(feature) == "num":
            return ", ".join(_fmt(s) for s in splits)
        return "; ".join(", ".join(map(str, g)) for g in splits)

    def selected_features(self) -> list:
        return [c for c in self.candidates if c in self.included]

    def auto_select(self, min_iv=0.02, max_psi=0.25, require_monotonic=False,
                    max_n_bins=5) -> pd.DataFrame:
        """Inclui em lote as variáveis que satisfazem os critérios e exclui as demais;
        marca a categoria ('manter'/'descartar'). Devolve o ranking usado."""
        rk = self.variable_iv(max_n_bins=max_n_bins)
        for _, r in rk.iterrows():
            feat = r["variavel"]
            iv = r["iv"]
            psi = r.get("pior_psi", np.nan)
            mono_ok = (not require_monotonic) or (r["tendencia"] in ("crescente", "decrescente"))
            ok = (np.isfinite(iv) and iv >= min_iv
                  and (not np.isfinite(psi) or psi <= max_psi) and mono_ok)
            if ok:
                self.include(feat); self.set_category(feat, "manter")
            else:
                self.exclude(feat); self.set_category(feat, "descartar")
        return rk

    def auto_categorize(self, min_iv=0.02, max_psi=0.25, require_monotonic=True,
                        psi_warn=0.10, max_n_bins=5, apply_selection=False) -> pd.DataFrame:
        """Categoriza em lote **todas** as candidatas em ``manter``/``revisar``/
        ``descartar`` por uma regra transparente, pensada para **Regressão
        Logística** (scorecard de crédito).

        Diferente de :meth:`auto_select`, **não altera a seleção** por padrão — a
        categoria é só triagem/documentação. Use ``apply_selection=True`` para
        também incluir as ``manter`` e excluir o resto.

        Regra (avaliada nesta ordem, por variável)::

            descartar  IV < min_iv (sem poder)  ou  pior_psi > max_psi (instável)
            revisar    força 'suspeito' (IV alto demais → possível vazamento)
                       ou IV fraco (min_iv ≤ IV < piso de 'médio': 0.10 clf / 0.03 reg)
                       ou pior_psi em atenção (psi_warn ≤ PSI ≤ max_psi)
                       ou (require_monotonic) tendência não-monotônica / com inversões
            manter     o restante (IV médio/forte, estável e monotônica)

        Devolve o ranking de :meth:`variable_iv` com ``categoria`` e ``motivo``
        (justificativa curta) preenchidos; ``motivo`` também passa a aparecer no
        ranking da UI.
        """
        rk = self.variable_iv(max_n_bins=max_n_bins)
        weak_ceiling = 0.10 if self.task_type == "classification" else 0.03
        cats, motivos = [], []
        for _, r in rk.iterrows():
            feat = r["variavel"]
            iv = r["iv"]; psi = r.get("pior_psi", np.nan)
            nao_mono = (r["tendencia"] == "não-monotônica") or (int(r.get("n_inversoes", 0)) > 0)
            iv_txt = "—" if not np.isfinite(iv) else f"{iv:.3f}"
            if not np.isfinite(iv) or iv < min_iv:
                cat, motivo = "descartar", f"IV {iv_txt} < mín. {min_iv:g} (sem poder)"
            elif np.isfinite(psi) and psi > max_psi:
                cat, motivo = "descartar", f"PSI {psi:.3f} > máx. {max_psi:g} (instável)"
            elif r["forca"] == "suspeito":
                cat, motivo = "revisar", f"IV {iv_txt} alto demais (possível vazamento)"
            elif iv < weak_ceiling:
                cat, motivo = "revisar", f"IV {iv_txt} fraco"
            elif np.isfinite(psi) and psi >= psi_warn:
                cat, motivo = "revisar", f"PSI {psi:.3f} em atenção"
            elif require_monotonic and nao_mono:
                cat, motivo = "revisar", "não-monotônica / com inversões"
            else:
                cat, motivo = "manter", f"IV {iv_txt}, estável e monotônica"
            self.set_category(feat, cat)
            self.var_meta[feat]["motivo"] = motivo
            if apply_selection:
                (self.include if cat == "manter" else self.exclude)(feat)
            cats.append(cat); motivos.append(motivo)
        rk = rk.copy()
        rk["categoria"] = cats
        rk["motivo"] = motivos
        return rk

    # ==================================================================
    # ESTEIRA DE SELEÇÃO DE VARIÁVEIS
    #   Porta de entrada única: escolha as etapas, rode e leve o relatório.
    #   A lógica vive em `selection` (esteira) e `selection_report`
    #   (apresentação) — importados LAZY, dentro dos métodos.
    # ==================================================================
    def select_features(self, steps=None, apply=True, progress_callback=None,
                        **params):
        """Roda a **esteira de seleção de variáveis** e devolve a trilha de auditoria.

        É a porta de entrada da seleção: você escolhe **quais** etapas quer, em
        **qual ordem**, e recebe — para cada candidata — a decisão
        (``selecionada``/``revisar``/``excluida``), **onde** ela saiu e **por
        quê**, em texto apresentável. O resultado fica guardado em
        :attr:`selection_` (e a política em :attr:`selection_policy_`), de onde o
        relatório, os gráficos e a interface o reaproveitam sem re-rodar.

        Etapas disponíveis (``steps``), na ordem canônica:

        * ``missing`` — exclui quem tem faltantes acima de ``max_missing``;
        * ``constante`` — exclui valor único, variância ~nula ou categoria
          dominante demais;
        * ``categoricas`` — cardinalidade, agrupamento de categorias raras e
          faltantes como categoria;
        * ``iv`` — poder discriminante: IV mínimo; IV altíssimo vira *revisar*
          (suspeita de vazamento);
        * ``psi`` — estabilidade entre amostras pelo pior PSI da variável;
        * ``monotonia`` — tendência da ordem de risco entre as faixas (só numéricas);
        * ``correlacao`` — redundância entre pares: sai a de menor IV;
        * ``vif`` — multicolinearidade pelo VIF do desenho do modelo vigente
          (exige modelo ajustado);
        * ``backward`` — *backward elimination* por importância, aplicando o passo
          escolhido (treina dezenas de modelos).

        ``steps=None`` usa a sequência default
        ``("missing", "constante", "categoricas", "iv", "psi", "monotonia",
        "correlacao")`` — filtros baratos primeiro, o tratamento das categóricas
        antes do IV (o agrupamento das raras muda a binagem e, portanto, o IV) e a
        redundância no fim. ``vif`` e ``backward`` ficam de fora do default por
        custo/pré-requisito.

        Parâmetros mais usados (todos opcionais; a lista completa e os defaults
        estão em :data:`~yggdrasil.credit_risk.model.selection.PARAMS_DEFAULT`):

        * ``min_iv`` — IV mínimo (``None`` → 0,02 na classificação · 0,01 na regressão);
        * ``max_psi`` — pior PSI tolerado entre amostras (0,25);
        * ``max_corr`` — associação máxima entre um par de variáveis (0,85);
        * ``max_missing`` — fração de faltantes tolerada (0,60);
        * ``max_categorias`` — cardinalidade máxima de uma categórica (30);
        * ``min_freq_categoria`` — abaixo disso a categoria é "rara" (0,01).

        Parameters
        ----------
        steps:
            Etapas a executar, na ordem desejada. Nome desconhecido/repetido
            levanta ``ValueError`` listando os válidos.
        apply:
            ``True`` (default) grava a decisão no segmentador (``include``/
            ``exclude``, ``set_category`` e o ``motivo`` do ``var_meta``).
            ``False`` **simula**: o estado volta exatamente como estava.
        progress_callback:
            ``cb(key, label, status, detail)`` — mesmo contrato de progresso das
            demais rotinas longas (``status`` ∈ ``"run"``/``"ok"``/``"err"``).
        **params:
            Réguas da esteira (ver acima).

        Returns
        -------
        SelectionResult
            Com ``tabela`` (uma linha por candidata), ``funil`` (por etapa),
            ``politica`` (parâmetros efetivos, em JSON) e ``historico``.

        Examples
        --------
        Chamada mínima — a sequência default, já aplicada no segmentador::

            res = seg.select_features()
            res.resumo()
            seg.selection_report("selecao.html")

        Escolhendo as etapas e apertando as réguas::

            res = seg.select_features(steps=["missing", "categoricas", "iv", "psi",
                                             "correlacao"],
                                      min_iv=0.05, max_psi=0.10, max_corr=0.80)
            res.tabela[["variavel", "decisao", "etapa_saida", "motivo"]]

        Simulando antes de aplicar (nada muda no segmentador)::

            simulado = seg.select_features(apply=False, min_iv=0.10)
            simulado.funil
        """
        from .selection import run_selection

        res = run_selection(self, steps=steps, apply=apply,
                            progress_callback=progress_callback, **params)
        self.selection_ = res
        self.selection_policy_ = dict(res.politica)
        return res

    def _selection_result(self, result=None):
        """Resultado de seleção a usar: o informado ou o último de
        :meth:`select_features` — com erro claro quando não há nenhum."""
        res = self.selection_ if result is None else result
        if res is None:
            raise RuntimeError(
                "Nenhuma seleção disponível: rode seg.select_features(...) antes "
                "(ou passe result=... com um SelectionResult já obtido).")
        return res

    def selection_report(self, path=None, result=None, **kw):
        """Relatório da última seleção como página HTML **autocontida**.

        Sem ``path`` devolve o HTML (``str``) — pronto para ``display(HTML(...))``
        no notebook; com ``path`` grava o arquivo e devolve o caminho. Usa
        ``result`` ou, na falta dele, a última :meth:`select_features`.

        ``**kw`` segue
        :func:`~yggdrasil.credit_risk.model.selection_report.build_selection_report_html`
        (``title``, ``subtitle``, ``top_iv``, ``annotate_top``, ``dpi``,
        ``incluir_graficos``); o próprio segmentador entra como contexto do
        cabeçalho."""
        from .selection_report import build_selection_report_html

        res = self._selection_result(result)
        kw.setdefault("seg", self)
        html_doc = build_selection_report_html(res, **kw)
        if path is None:
            return html_doc
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html_doc)
        return str(path)

    def selection_xlsx(self, path, result=None):
        """Exporta a última seleção para um Excel multi-abas (``Decisoes``,
        ``Funil``, ``Politica``) e devolve o caminho.

        Requer o pacote **opcional** ``openpyxl`` — sem ele, sobe um
        :class:`ImportError` com a instrução de instalação (a biblioteca não ganha
        dependência nova por causa do relatório)."""
        from .selection_report import export_selection_xlsx

        return export_selection_xlsx(self._selection_result(result), path)

    def plot_selection_funil(self, result=None, figsize=None, dpi=150,
                             save_path=None, ax=None):
        """**Funil da seleção**: quantas variáveis seguiram depois de cada etapa —
        o gráfico-síntese da apresentação. Usa ``result`` ou a última
        :meth:`select_features`."""
        from .selection_report import plot_funil

        return plot_funil(self._selection_result(result), figsize=figsize, dpi=dpi,
                          save_path=save_path, ax=ax)

    def plot_selection_motivos(self, result=None, top=10, por="causa", figsize=None,
                               dpi=150, save_path=None, ax=None):
        """**Por que perdemos variáveis**: quantas cada causa de exclusão levou
        embora (a leitura executiva do funil). ``por`` ∈ ``"causa"``/``"etapa"``/
        ``"motivo"``."""
        from .selection_report import plot_motivos

        return plot_motivos(self._selection_result(result), top=top, por=por,
                            figsize=figsize, dpi=dpi, save_path=save_path, ax=ax)

    def plot_selection_iv(self, result=None, top=20, figsize=None, dpi=150,
                          save_path=None, ax=None):
        """**Ranking de IV** das candidatas, colorido pela decisão, com o corte de
        IV efetivamente usado marcado no eixo."""
        from .selection_report import plot_iv_ranking

        return plot_iv_ranking(self._selection_result(result), top=top,
                               figsize=figsize, dpi=dpi, save_path=save_path, ax=ax)

    def plot_selection_iv_psi(self, result=None, annotate_top=8, figsize=(7.8, 5.6),
                              dpi=150, save_path=None, ax=None):
        """**Poder × estabilidade** (IV × pior PSI) em quadrantes formados pelos
        cortes da política — a matriz de decisão da seleção."""
        from .selection_report import plot_iv_psi

        return plot_iv_psi(self._selection_result(result), annotate_top=annotate_top,
                           figsize=figsize, dpi=dpi, save_path=save_path, ax=ax)

    # ---- multicolinearidade (redundância entre variáveis selecionadas) ----
    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> float:
        """V de Cramér entre duas séries categóricas (0 = independentes · 1 =
        associação perfeita), via qui-quadrado da tabela de contingência (sem
        correção de continuidade). ``NaN`` quando alguma delas tem menos de 2
        categorias observadas."""
        from scipy.stats import chi2_contingency
        m = x.notna() & y.notna()
        if not bool(m.any()):
            return float("nan")
        tab = pd.crosstab(x[m].astype(str), y[m].astype(str))
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            return float("nan")
        chi2 = float(chi2_contingency(tab, correction=False)[0])
        n = float(tab.to_numpy().sum())
        k = min(tab.shape) - 1
        return float(np.sqrt(chi2 / (n * k))) if n > 0 else float("nan")

    def correlation_report(self, threshold=0.85, features=None, sample=None) -> pd.DataFrame:
        """Relatório de **multicolinearidade** entre as variáveis selecionadas
        (default: as da lista de incluídas; sem seleção, todas as candidatas):
        **Spearman** entre as numéricas e **V de Cramér** entre as categóricas,
        na amostra de referência (ou em ``sample``).

        Devolve um par redundante por linha — associação ≥ ``threshold`` (no
        Spearman vale o |ρ|) — com a sugestão de poda: ``manter`` = a variável
        de **maior IV** do par e ``remover`` = a redundante. Colunas: ``variavel_1,
        variavel_2, metodo, associacao, iv_1, iv_2, manter, remover``.

        ``.attrs`` traz as matrizes completas (``corr_num``/``corr_cat``), o
        ``threshold`` e ``poda_sugerida`` — a lista de variáveis a excluir por
        uma poda **gulosa** (pares em ordem decrescente de associação; par já
        resolvido por uma remoção anterior não remove ninguém), pronta para
        aplicar via :meth:`exclude`. Heatmap: :meth:`plot_correlation_heatmap`."""
        base = (list(features) if features is not None
                else (self.selected_features() or list(self.candidates)))
        feats, vistos = [], set()
        for f in base:                       # dedup preservando a ordem
            if f in self.df.columns and f not in vistos:
                feats.append(f); vistos.add(f)
        sub = self._frame(sample, cols=feats)
        # associação numa amostra de até max_linhas_graficos linhas (semente
        # fixa): p/ triagem de redundância (corte ~0,85) o erro da correlação
        # fica na 3ª casa — e o Spearman do pandas com NaN re-ranqueava a base
        # inteira POR PAR (~150 s em 3,5M linhas × 18 numéricas)
        cap = self.max_linhas_graficos
        if cap and len(sub) > int(cap):
            sub = sub.sample(n=int(cap), random_state=self.random_state)
        num = [f for f in feats if self._detect_kind(f, sub) == "num"]
        cat = [f for f in feats if self._detect_kind(f, sub) == "cat"]
        corr_num = (_spearman_pairwise(sub[num]) if len(num) >= 2
                    else pd.DataFrame(np.eye(len(num)), index=num, columns=num))
        if len(cat) >= 2:
            vals = np.eye(len(cat))
            for i in range(len(cat)):
                for j in range(i + 1, len(cat)):
                    v = self._cramers_v(sub[cat[i]], sub[cat[j]])
                    vals[i, j] = vals[j, i] = v
            corr_cat = pd.DataFrame(vals, index=cat, columns=cat)
        else:
            corr_cat = pd.DataFrame(np.eye(len(cat)), index=cat, columns=cat)
        pares = []
        for mat, metodo in ((corr_num, "spearman"), (corr_cat, "cramers_v")):
            cols = list(mat.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    v = float(mat.iloc[i, j])
                    if np.isfinite(v) and abs(v) >= float(threshold):
                        pares.append((cols[i], cols[j], metodo, abs(v)))
        pares.sort(key=lambda t: -t[3])
        # IV só das variáveis ENVOLVIDAS em algum par (evita binar as demais)
        envolvidas = sorted({f for p in pares for f in p[:2]})
        iv_map = {}
        if envolvidas:
            rk = self.variable_iv(features=envolvidas, with_psi=False)
            iv_map = dict(zip(rk["variavel"], rk["iv"]))
        removidas, rows = set(), []
        for f1, f2, metodo, a in pares:
            iv1 = float(iv_map.get(f1, np.nan)); iv2 = float(iv_map.get(f2, np.nan))
            # mantém a de MAIOR IV (empate ou ambos sem IV: mantém a 1ª)
            if np.isnan(iv2) or (not np.isnan(iv1) and iv1 >= iv2):
                manter, remover = f1, f2
            else:
                manter, remover = f2, f1
            rows.append({"variavel_1": f1, "variavel_2": f2, "metodo": metodo,
                         "associacao": round(a, 4), "iv_1": iv1, "iv_2": iv2,
                         "manter": manter, "remover": remover})
            # poda gulosa: par já resolvido por remoção anterior não remove mais
            if f1 in removidas or f2 in removidas:
                continue
            removidas.add(remover)
        out = pd.DataFrame(rows, columns=["variavel_1", "variavel_2", "metodo",
                                          "associacao", "iv_1", "iv_2",
                                          "manter", "remover"])
        out.attrs.update(threshold=float(threshold), corr_num=corr_num,
                         corr_cat=corr_cat,
                         poda_sugerida=[f for f in feats if f in removidas])
        return out

    def plot_correlation_heatmap(self, threshold=0.85, features=None, sample=None,
                                 report=None, figsize=None, dpi=150, save_path=None):
        """Heatmap das associações do :meth:`correlation_report`: Spearman entre
        as numéricas (−1..+1) e V de Cramér entre as categóricas (0..1), lado a
        lado. Células fora da diagonal com |associação| ≥ ``threshold`` ganham
        contorno (par redundante). ``report`` reaproveita um relatório já
        computado (evita recalcular as matrizes)."""
        rep = report if report is not None else self.correlation_report(
            threshold=threshold, features=features, sample=sample)
        thr = float(rep.attrs.get("threshold", threshold))
        mats = []
        cn = rep.attrs.get("corr_num"); cc = rep.attrs.get("corr_cat")
        if cn is not None and len(cn.columns) >= 2:
            mats.append(("Spearman (numéricas)", cn, "RdBu_r", -1.0, 1.0))
        if cc is not None and len(cc.columns) >= 2:
            mats.append(("V de Cramér (categóricas)", cc, "Blues", 0.0, 1.0))
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle
        if not mats:
            fig = Figure(figsize=figsize or (6.4, 2.0), dpi=dpi)
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            ax.text(0.5, 0.5, "menos de 2 variáveis por tipo — sem matriz a exibir",
                    ha="center", va="center", transform=ax.transAxes, color="#889")
            ax.axis("off"); fig.tight_layout()
            return fig
        larguras = [len(m.columns) for _, m, *_ in mats]
        if figsize is None:
            figsize = (min(2.4 + 0.62 * sum(larguras), 13.5),
                       min(2.0 + 0.5 * max(larguras), 8.0))
        fig = Figure(figsize=figsize, dpi=dpi)
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, len(mats), squeeze=False,
                            gridspec_kw={"width_ratios": larguras})[0]
        for ax, (titulo, mat, cmap, vmin, vmax) in zip(axes, mats):
            vals = mat.to_numpy(dtype="float64")
            im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            labels = [self.label(c) for c in mat.columns]
            k = len(labels)
            ax.set_xticks(range(k))
            ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5)
            ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=7.5)
            for i in range(k):
                for j in range(k):
                    v = vals[i, j]
                    if not np.isfinite(v):
                        continue
                    if k <= 12:              # anota só quando a matriz é legível
                        # a cor casa com a CÉLULA (preto nas claras, branco nas
                        # saturadas) — e a célula não muda com o tema da UI, por
                        # isso o gid "keep-ink" pede ao _dark_fig para NÃO trocar
                        # a tinta (trocado, o preto virava branco invisível
                        # sobre célula clara no tema escuro)
                        txt = ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                      fontsize=7,
                                      color="#fff" if abs(v) >= 0.65 else "#111")
                        txt.set_gid("keep-ink")
                    if i != j and abs(v) >= thr:   # par redundante em destaque
                        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="#b3392f", lw=1.6))
            ax.set_title(titulo, fontsize=10, fontweight="bold", color="#15324a")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        fig.suptitle(f"Associação entre variáveis · limiar {thr:g}",
                     fontsize=10.5, fontweight="bold", color="#15324a")
        fig.tight_layout(rect=(0, 0, 1, 0.94))           # reserva a faixa do suptitle
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # C) Modelo
    # ------------------------------------------------------------------
    def _bin_encoding(self, feature) -> dict:
        """Bins (ajustados na referência) + valor de codificação por bin: **WoE**
        (classificação) ou **risco médio do bin** (regressão). Reaproveita os bins
        manuais/ótimos da análise univariada (:meth:`_resolve_bins`). Usado pela
        transformação WoE que alimenta o modelo."""
        ref = self._frame(self.ref_sample, cols=[feature, self.target])
        bins, kind = self._resolve_bins(feature, sample=self.ref_sample)
        y_all = ref[self.target].to_numpy(dtype="float64")
        enc_bins = []
        if self.task_type == "classification":
            n_evt_tot = float(np.nansum(y_all == 1))
            n_non_tot = float(np.nansum(y_all == 0))
            for b, m in zip(bins, _bin_masks(ref[feature], bins)):
                yi = y_all[m]; yi = yi[~np.isnan(yi)]
                d_evt = float((yi == 1).sum()) / max(n_evt_tot, _EPS)
                d_non = float((yi == 0).sum()) / max(n_non_tot, _EPS)
                enc_bins.append((b, float(np.log((d_non + _EPS) / (d_evt + _EPS)))))
            fallback = 0.0   # WoE neutro p/ valores fora dos bins vistos na referência
        else:
            mean_global = self._risco(y_all)
            for b, m in zip(bins, _bin_masks(ref[feature], bins)):
                r = self._risco(y_all[m])
                enc_bins.append((b, float(r) if np.isfinite(r) else mean_global))
            fallback = float(mean_global) if np.isfinite(mean_global) else 0.0
        return {"kind": kind, "bins": enc_bins, "fallback": fallback}

    def _build_raw_preprocessor(self, features):
        """``ColumnTransformer`` do ``transform='raw'``: imputação (mediana) nas
        numéricas + imputação (moda) e one-hot nas categóricas. A ordem
        pós-transformação é o bloco ``num`` primeiro (na ordem de ``features``) e
        depois as *dummies* do bloco ``cat`` — premissa dos vetores posicionais
        de monotonicidade (ver :meth:`_monotone_est_params`)."""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        # dummies de scorecard: bloco próprio (0/1 por faixa, pior = referência),
        # fora do num/cat — vem depois deles, então as posições de num/cat não mudam
        ordinais = self.scorecard_dummy_features(features)
        num = [f for f in features if self._detect_kind(f) == "num" and f not in ordinais]
        cat = [f for f in features if self._detect_kind(f) == "cat" and f not in ordinais]
        transformers = []
        if num:
            transformers.append(("num", SimpleImputer(strategy="median"), num))
        if cat:
            cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                 ("ohe", _make_ohe())])
            transformers.append(("cat", cat_pipe, cat))
        if ordinais:
            enc = ScorecardDummyEncoder(specs={f: self._dummy_spec(f) for f in ordinais},
                                        features=list(ordinais))
            transformers.append(("dum", enc, list(ordinais)))
        return ColumnTransformer(transformers, remainder="drop")

    def _check_design_memory(self, X, features) -> None:
        """Pré-checagem do one-hot DENSO do ``transform='raw'``.

        A matriz de desenho tem linhas × (numéricas + Σ níveis das categóricas)
        × 8 bytes, e o ``ColumnTransformer`` a materializa duas vezes (bloco
        one-hot + ``hstack``). Uma categórica de alta cardinalidade (ID, código,
        número lido como texto) faz isso passar da RAM, e o OOM killer derruba o
        driver no meio do fit, sem traceback. Aqui a conta é feita antes e,
        quando não cabe na memória livre, levanta ``MemoryError`` nomeando as
        variáveis responsáveis. No-op quando não há categórica ou quando a
        memória livre não é mensurável."""
        ordinais = set(self.scorecard_dummy_features(features))
        cat = [f for f in features if self._detect_kind(f) == "cat" and f not in ordinais]
        if not cat or any(isinstance(X[f], pd.DataFrame) for f in cat):
            return      # nome repetido no df: o sklearn recusa com mensagem clara
        livre = _available_memory_bytes()
        if livre is None:
            return
        niveis = {f: int(X[f].nunique(dropna=True)) for f in cat}
        largura = (len(features) - len(cat)) + sum(niveis.values())
        estimado = 2 * len(X) * largura * 8
        if estimado <= livre:
            return
        top = sorted(niveis.items(), key=lambda kv: -kv[1])[:5]
        dec = set(getattr(self, "decimal_cols_", []))
        lista = ", ".join(f"{self.label(f)} ({n:,} níveis"
                          f"{', Decimal: converta para float' if f in dec else ''})"
                          for f, n in top)
        raise MemoryError(
            f"O one-hot denso das categóricas pediria ~{estimado / 1024 ** 3:.1f} GB "
            f"({len(X):,} linhas × {largura:,} colunas, 2 cópias) e há "
            f"~{livre / 1024 ** 3:.1f} GB livres. Maiores cardinalidades: {lista}. "
            "Exclua IDs/códigos, converta números lidos como texto para numérico "
            "ou use transform='woe', que agrupa as categorias nos bins.")

    def _build_pipeline(self, features, algorithm, hyperparams, transform="raw", task=None,
                        class_counts=None):
        from sklearn.pipeline import Pipeline

        # ``task`` permite forçar classificação/regressão (usado pelo Two-Stage,
        # que constrói um classificador e uma regressão a partir de um segmenter
        # de regressão); por padrão segue o task_type do segmenter.
        # ``class_counts=(n_neg, n_pos)`` liga o balanceamento de classes no
        # estimador (ver :func:`_build_estimator`).
        task = task or self.task_type
        est = _build_estimator(algorithm, task, hyperparams, random_state=self.random_state,
                               class_counts=class_counts)
        if transform == "woe":
            # variáveis transformadas no estilo scorecard (binagem + WoE/risco do bin);
            # as marcadas com dummies de scorecard entram como 0/1 por faixa
            dums = self.scorecard_dummy_features(features)
            resto = [f for f in features if f not in dums]
            prefix = "WoE" if task == "classification" else "bin"
            woe = WoeBinEncoder(encodings={f: self._bin_encoding(f) for f in resto},
                                features=resto, name_prefix=prefix)
            if not dums:
                return Pipeline([("pre", woe), ("est", est)])
            from sklearn.compose import ColumnTransformer
            blocos = [("woe", woe, resto)] if resto else []
            blocos.append(("dum", ScorecardDummyEncoder(
                specs={f: self._dummy_spec(f) for f in dums}, features=dums), dums))
            return Pipeline([("pre", ColumnTransformer(blocos, remainder="drop")),
                             ("est", est)])

        return Pipeline([("pre", self._build_raw_preprocessor(features)), ("est", est)])

    # ------------------------------------------------------------------
    # Restrições de monotonicidade (fit/tune_optuna: monotone='auto' | dict)
    # ------------------------------------------------------------------
    def monotone_directions(self, features=None) -> dict:
        """Direção de monotonicidade sugerida para cada variável **numérica**, a
        partir da tendência univariada do risco por faixa na referência (mesma
        leitura da coluna ``tendencia`` de :meth:`variable_iv`): ``+1`` = risco
        cresce com a variável (predição não-decrescente), ``-1`` = decresce e
        ``0`` = não-monotônica (fica livre). Categóricas ficam de fora (o one-hot
        não tem ordem). É a base do ``monotone='auto'`` de :meth:`fit` e
        :meth:`tune_optuna`."""
        feats = (list(features) if features is not None
                 else (self.selected_features() or list(self.candidates)))
        out = {}
        for f in feats:
            if f not in self.df.columns or self._detect_kind(f) != "num":
                continue
            direction = 0
            try:
                vt = self.variable_table(f)
                trend, _ = _trend(vt.attrs.get("risco_ordenavel") or [])
                direction = {"crescente": 1, "decrescente": -1}.get(trend, 0)
            except Exception:  # noqa: BLE001 — variável problemática fica livre
                pass
            out[f] = direction
        return out

    def _resolve_monotone(self, monotone, feats, algorithm, transform):
        """Resolve a opção ``monotone`` de :meth:`fit`/:meth:`tune_optuna` num
        dict ``{variável: ±1}`` só com as direções efetivas (sem zeros), ou
        ``None`` quando não há restrição a aplicar. ``'auto'``/``True`` usa
        :meth:`monotone_directions`; um dict sobrescreve a direção automática
        por variável (``0`` libera a variável). Situações sem suporte
        (``transform!='raw'``, algoritmo fora de :data:`MONOTONE_ALGORITHMS`)
        são ignoradas com aviso."""
        if not monotone:
            return None
        if transform != "raw":
            warnings.warn(
                "Restrições de monotonicidade só se aplicam a transform='raw' — "
                "com a transformação por bins cada variável entra pelo valor do "
                "seu bin e a relação com a resposta já segue a direção do próprio "
                "WoE/risco do bin. Opção ignorada.")
            return None
        if algorithm not in MONOTONE_ALGORITHMS:
            warnings.warn(
                f"O algoritmo {algorithm!r} não suporta restrições de "
                f"monotonicidade (suportados: {', '.join(MONOTONE_ALGORITHMS)}). "
                "Opção ignorada.")
            return None
        if not (monotone in ("auto", True) or isinstance(monotone, dict)):
            raise ValueError("monotone deve ser 'auto', um dict {variável: -1|0|+1} "
                             f"ou None (recebi {monotone!r}).")
        dirs = self.monotone_directions(feats)      # base automática (numéricas)
        if isinstance(monotone, dict):              # override manual por variável
            for f, d in monotone.items():
                if f not in dirs:
                    warnings.warn(f"monotone: variável {f!r} ignorada (não é "
                                  "numérica ou não está entre as variáveis do "
                                  "modelo).")
                    continue
                d = int(d)
                if d not in (-1, 0, 1):
                    raise ValueError(f"monotone[{f!r}] deve ser -1, 0 ou +1 "
                                     f"(recebi {d}).")
                dirs[f] = d
        dirs = {f: d for f, d in dirs.items() if d != 0}
        if not dirs:
            warnings.warn("Nenhuma variável numérica com direção monotônica "
                          "definida — modelo treinado sem restrições de "
                          "monotonicidade.")
            return None
        return dirs

    def _monotone_est_params(self, algorithm, dirs, X, features) -> tuple:
        """Traduz as direções por variável (``{variável: ±1}``) no parâmetro
        nativo de monotonicidade do algoritmo. Devolve ``(params, pandas_out)``:
        ``params`` entra no estimador via ``set_params`` e ``pandas_out=True``
        indica que o pré-processador precisa emitir DataFrame (nomes de coluna
        visíveis ao estimador).

        Ajusta um pré-processador DESCARTÁVEL em ``X`` só para descobrir os
        nomes/ordem REAIS das colunas pós-transformação (``num__<var>`` primeiro,
        depois as *dummies* ``cat__<var>_<valor>``) — assim um eventual descarte
        de coluna toda-faltante pelo imputador não desalinha o vetor.

        * ``hist_gradient_boosting``: ``monotonic_cst`` como **dict por nome**
          (robusto à ordem; exige entrada nomeada, daí o
          ``set_output(transform='pandas')`` no pré-processador);
        * ``lightgbm``/``xgboost``: ``monotone_constraints`` **posicional**,
          alinhado às colunas pós-transformação (0 nas *dummies* one-hot)."""
        probe = self._build_raw_preprocessor(features)
        probe.fit(X)
        names = list(probe.get_feature_names_out())
        if algorithm == "hist_gradient_boosting":
            cst = {n: dirs[n[len("num__"):]] for n in names
                   if n.startswith("num__") and n[len("num__"):] in dirs}
            return {"monotonic_cst": cst}, True
        vec = [dirs.get(n[len("num__"):], 0) if n.startswith("num__") else 0
               for n in names]
        if algorithm == "lightgbm":
            return {"monotone_constraints": vec}, False
        if algorithm == "xgboost":
            return {"monotone_constraints": tuple(vec)}, False
        raise ValueError(algorithm)  # pragma: no cover — guardado por MONOTONE_ALGORITHMS

    @staticmethod
    def _set_monotone_on_pipe(pipe, params, pandas_out):
        """Aplica no pipeline AINDA não ajustado os parâmetros produzidos por
        :meth:`_monotone_est_params` (estimador + saída pandas do pré-processador
        quando o parâmetro é um dict por nome)."""
        pipe.named_steps["est"].set_params(**params)
        if pandas_out:      # dict por NOME exige que o estimador veja um DataFrame
            pipe.named_steps["pre"].set_output(transform="pandas")

    def fit(self, algorithm=None, hyperparams=None, features=None, transform="raw",
            class_balance=False, monotone=None):
        """Treina um modelo na amostra de referência (DES) com as variáveis
        selecionadas (ou ``features``). ``algorithm`` default: logística
        (classificação) / linear (regressão). Calcula ``score_`` para todas as linhas.

        ``transform``: ``"raw"`` usa os valores originais (numéricas + one-hot das
        categóricas); ``"woe"`` transforma cada variável no WoE do seu bin
        (classificação) ou no risco médio do bin (regressão), reaproveitando os
        bins/grupos definidos na análise univariada — estilo *scorecard*.

        ``class_balance=True`` (só classificação) compensa o desbalanceio do alvo
        no treino, traduzido por algoritmo em :func:`_build_estimator`:
        ``class_weight='balanced'`` (logística, florestas, HistGB),
        ``scale_pos_weight=n_neg/n_pos`` (XGBoost, LightGBM),
        ``auto_class_weights='Balanced'`` (CatBoost); o GradientBoosting clássico
        (sem ``class_weight``) recebe ``sample_weight`` balanceado no ``fit``.
        Um ``"class_balance"`` dentro de ``hyperparams`` (ex.: ``best_params`` do
        Optuna) tem precedência sobre o argumento.

        ``monotone`` (opcional): impõe **restrições de monotonicidade** nas
        variáveis numéricas dos algoritmos que as suportam
        (:data:`MONOTONE_ALGORITHMS` — ``monotonic_cst`` no HistGradientBoosting,
        ``monotone_constraints`` no LightGBM/XGBoost). ``'auto'`` usa a direção
        da tendência univariada de cada variável (risco por faixa na referência
        — ver :meth:`monotone_directions`): risco crescente → predição
        não-decrescente na variável; decrescente → não-crescente; variáveis
        não-monotônicas e categóricas ficam livres. Um dict
        ``{variável: -1|0|+1}`` sobrescreve a direção automática por variável
        (``0`` libera). Só se aplica a ``transform='raw'`` — com WoE cada
        variável entra pelo valor do seu bin e a relação já segue a direção do
        próprio WoE/risco do bin; algoritmos sem suporte ignoram a opção com
        aviso. As direções aplicadas ficam em ``monotone_dirs_`` e a escolha é
        persistida em :meth:`to_dict`."""
        if algorithm is None:
            algorithm = "logistica" if self.task_type == "classification" else "linear"
        hp = dict(hyperparams or {})
        # "class_balance" pode vir embutido nos hyperparams (pseudo-parâmetro
        # nosso, não do estimador — ex.: best_params do tuning): extrai antes.
        class_balance = bool(hp.pop("class_balance", class_balance))
        if class_balance and self.task_type != "classification":
            raise ValueError("Balanceamento de classes só se aplica a "
                             "task_type='classification'.")
        feats = list(features) if features is not None else self.selected_features()
        if not feats:
            feats = list(self.candidates)
        # restrições de monotonicidade (opcional): direções por variável numérica
        mono_dirs = self._resolve_monotone(monotone, feats, algorithm, transform)
        m_fit = self._fit_mask()
        X = self.df.loc[m_fit, feats]
        y = self.df.loc[m_fit, self.target]
        if transform == "raw":                 # antes de tocar no estado: recusa limpa
            self._check_design_memory(X, feats)
        self.model_features = feats
        self.algorithm = algorithm
        self.hyperparams = dict(hp)
        self.feature_transform = transform
        self.class_balance = class_balance
        self.monotone = monotone if mono_dirs else None
        self.monotone_dirs_ = dict(mono_dirs or {})
        self.two_stage = False                 # fit "normal" desliga o modo hurdle
        self.two_stage_threshold = None

        if self.task_type == "classification":
            y = y.astype(int)
        counts = None
        fit_kwargs = {}
        if class_balance:
            if algorithm == "gradient_boosting":   # sem class_weight no construtor
                fit_kwargs["est__sample_weight"] = _balanced_sample_weight(y)
            else:
                n_pos = int((y == 1).sum())
                counts = (int(len(y)) - n_pos, n_pos)
        self.model = self._build_pipeline(feats, algorithm, hp, transform=transform,
                                          class_counts=counts)
        if mono_dirs:
            self._set_monotone_on_pipe(
                self.model, *self._monotone_est_params(algorithm, mono_dirs, X, feats))
        self.model.fit(X, y, **fit_kwargs)
        self.calibration_ = None           # modelo novo ⇒ a camada antiga não vale
        self.score_ = self._compute_score(self.df)
        # sem calibração, o score_ recém-calculado É o score cru deste modelo
        self._raw_score_cache = (self.model, len(self.df), self.score_)
        self._shap_cache = {}
        self._avisa_sinal_scorecard()      # ordinais de scorecard: coeficiente < 0?
        return self

    def fit_two_stage(self, threshold, clf_algorithm="logistica", reg_algorithm="linear",
                      clf_hyperparams=None, reg_hyperparams=None, features=None,
                      transform="raw"):
        """Ajusta um modelo **Two-Stage (hurdle)** para regressão (alvo): binariza o
        alvo em ``y ≥ threshold`` e treina, na referência (DES):

        * **etapa 1 — classificação**: ``P(y ≥ threshold)`` (``clf_algorithm``);
        * **etapa 2 — regressão**: prevê ``y`` no grupo ``y ≥ threshold``
          (``reg_algorithm``).

        A resposta final combina as duas — ``E[y] = P(≥t)·reg(x) + (1−P)·âncora₀``,
        com ``âncora₀`` = média do grupo abaixo do threshold — e alimenta
        ``score_``, as métricas combinadas e os ratings, exatamente como um modelo
        de regressão comum (ver :class:`_TwoStageModel`). Métricas de cada etapa
        ficam em :meth:`metrics_classifier` e :meth:`metrics_regressor`; a resposta
        combinada, em :meth:`metrics`.

        Só se aplica a ``task_type='regression'``. ``transform='raw'`` (o Two-Stage
        usa os valores originais das variáveis; o WoE por etapa não é suportado)."""
        if self.task_type != "regression":
            raise ValueError("Two-Stage é exclusivo de problemas de regressão "
                             "(task_type='regression').")
        if transform != "raw":
            raise ValueError("Two-Stage suporta apenas transform='raw'.")
        feats = list(features) if features is not None else self.selected_features()
        if not feats:
            feats = list(self.candidates)
        t = float(threshold)

        m_fit = self._fit_mask()
        X = self.df.loc[m_fit, feats]
        y = self.df.loc[m_fit, self.target].astype(float)
        self._check_design_memory(X, feats)
        ybin = (y >= t).astype(int)
        if ybin.nunique() < 2:
            lado = "≥" if int(ybin.iloc[0]) == 1 else "<"
            raise ValueError(f"O threshold {t:g} deixa uma única classe (todos "
                             f"{lado} t) na referência. Ajuste o threshold.")
        mask1 = ybin == 1
        if int(mask1.sum()) < 10:
            raise ValueError(f"Poucas observações acima do threshold ({int(mask1.sum())}) "
                             "para treinar a regressão da 2ª etapa (mínimo 10). "
                             "Reduza o threshold.")

        clf = self._build_pipeline(feats, clf_algorithm, clf_hyperparams,
                                   transform="raw", task="classification")
        clf.fit(X, ybin)
        reg = self._build_pipeline(feats, reg_algorithm, reg_hyperparams,
                                   transform="raw", task="regression")
        reg.fit(X[mask1], y[mask1])
        anchor0 = float(y[~mask1].mean()) if bool((~mask1).any()) else 0.0

        self.model = _TwoStageModel(clf, reg, t, anchor0)
        self.model_features = feats
        self.two_stage = True
        self.two_stage_threshold = t
        self.algorithm = f"two_stage:{clf_algorithm}+{reg_algorithm}"
        self.hyperparams = {"threshold": t, "clf_algorithm": clf_algorithm,
                            "reg_algorithm": reg_algorithm,
                            "clf_hyperparams": dict(clf_hyperparams or {}),
                            "reg_hyperparams": dict(reg_hyperparams or {}),
                            "anchor0": anchor0}
        self.feature_transform = "raw"
        self.class_balance = False             # hurdle não usa o balanceamento do fit
        self.monotone = None                   # hurdle não aplica monotonicidade
        self.monotone_dirs_ = {}
        self.calibration_ = None               # modelo novo ⇒ a camada antiga não vale
        self.score_ = self._compute_score(self.df)
        self._shap_cache = {}
        self._metrics_cache = None
        self._metrics_ci_cache = None
        return self

    def _two_stage_sample_mask(self, a):
        """Máscara booleana da amostra ``a`` (all-True sem sample_col)."""
        return (pd.Series(True, index=self.df.index) if self.sample_col is None
                else self._frame_mask(a))

    def metrics_classifier(self) -> pd.DataFrame:
        """(Two-Stage) Métricas de classificação da 1ª etapa por amostra —
        ``y ≥ threshold`` (real) vs. ``P(≥t)`` prevista: taxa_1, auc, ks, gini…"""
        if not self.two_stage:
            raise RuntimeError("Disponível apenas no modo Two-Stage (fit_two_stage).")
        t = self.model.threshold
        rows = []
        for a in self._samples():
            mask = self._two_stage_sample_mask(a)
            sub = self.df.loc[mask]
            y = sub[self.target].to_numpy(dtype="float64")
            ok = ~np.isnan(y)
            if not ok.any():
                continue
            Xa = sub.loc[ok, self.model_features]
            ybin = (y[ok] >= t).astype(int)
            p = self.model.proba(Xa)
            row = {"amostra": a, "n": int(ybin.size), "taxa_1": round(float(ybin.mean()), 6)}
            if len(np.unique(ybin)) == 2:
                row.update(classification_metrics(ybin, p))
            rows.append(row)
        return pd.DataFrame(rows)

    def metrics_regressor(self) -> pd.DataFrame:
        """(Two-Stage) Métricas de regressão da 2ª etapa por amostra, restritas ao
        grupo ``y ≥ threshold`` — ``y`` real vs. ``reg(x)``: rmse, mae, r2…"""
        if not self.two_stage:
            raise RuntimeError("Disponível apenas no modo Two-Stage (fit_two_stage).")
        t = self.model.threshold
        rows = []
        for a in self._samples():
            mask = self._two_stage_sample_mask(a)
            sub = self.df.loc[mask]
            y = sub[self.target].to_numpy(dtype="float64")
            ok = ~np.isnan(y) & (y >= t)
            row = {"amostra": a, "n": int(ok.sum())}
            if int(ok.sum()) >= 2:
                Xa = sub.loc[ok, self.model_features]
                row.update(regression_metrics(y[ok], self.model.reg_predict(Xa)))
            rows.append(row)
        return pd.DataFrame(rows)

    def tune_optuna(self, algorithm=None, n_trials=30, transform="raw", features=None,
                    timeout=None, random_state=None, fit_best=True, verbose=False,
                    progress_callback=None, log_mlflow=False, mlflow_experiment=None,
                    mlflow_run_name=None, search_space=None, register_model=False,
                    mlflow_model_name=None, class_balance=None, monotone=None,
                    cv=None, time_aware=False, stability_penalty=None, pruner=None):
        """Otimização bayesiana de hiperparâmetros com **Optuna** (dependência
        core). Treina na referência (DES) e avalia no OOT (se houver alvo;
        senão, num split 75/25 do DES), maximizando **AUC** (classificação) ou
        **R²** (regressão). Guarda o resultado em ``self.tuning_`` (e o estudo em
        ``self.study_``); com ``fit_best=True`` reajusta o modelo com os melhores
        hiperparâmetros. Algoritmos tunáveis: :data:`TUNABLE_ALGORITHMS`.

        ``cv`` (opcional): ``None`` (default) mantém a validação única acima;
        um inteiro ``k ≥ 2`` valida cada trial por **validação cruzada na
        referência (DES)** — ``StratifiedKFold`` na classificação, ``KFold`` na
        regressão — e o objetivo vira a **média dos folds**. Com
        ``time_aware=True`` (requer ``cv`` e ``date_col``), os folds são
        **temporais por safra**: as safras (``date_col``) são partidas em ``k``
        blocos contíguos e cada bloco (do 2º em diante) é validado com treino em
        todos os anteriores (janela expansiva — ``k−1`` folds).

        ``stability_penalty`` (opcional): ``λ ≥ 0`` **penaliza instabilidade**
        no objetivo de cada trial::

            objetivo = métrica_val − λ·max(0, PSI_des→val − 0,10)
                                   − λ·max(0, métrica_treino − métrica_val − 0,05)

        onde o PSI é o do score treino→validação (média dos folds com CV) e o
        segundo termo é o **gap de overfit** (AUC na classificação; R² na
        regressão). As componentes ficam em ``trial.user_attrs['objetivo']``
        (``metric_val``, ``metric_treino``, ``gap_treino_val``,
        ``psi_score_des_val``, ``pen_psi``, ``pen_gap``, ``lambda``, ``valor``).

        ``pruner`` (opcional): ``'median'`` liga o ``MedianPruner`` do Optuna —
        com CV, a média parcial dos folds é reportada por fold e trials pouco
        promissores são **podados** antes de completar todos os folds (também
        aceita uma instância de ``optuna.pruners.BasePruner``).

        ``search_space`` (opcional): sobrescreve quais hiperparâmetros são
        buscados e seus intervalos — dict ``{nome: {type, low, high, log?, step?,
        choices?}}`` (ver :data:`OPTUNA_SEARCH_SPACE` e :func:`_optuna_space`).
        ``None`` usa o catálogo padrão do algoritmo.

        ``class_balance`` (só classificação): ``None`` (default) inclui o
        **balanceamento de classes** no espaço de busca — o Optuna testa ligado ×
        desligado como um parâmetro categórico ``class_balance``; ``True``/``False``
        fixa a opção em todos os trials. O melhor valor é aplicado no re-ajuste
        final (``fit_best``). Ignorado na regressão. Ver :meth:`fit`.

        ``monotone`` (opcional): restrições de monotonicidade aplicadas a TODOS
        os trials (e ao re-ajuste final) — mesma semântica de :meth:`fit`
        (``'auto'`` = direção da tendência univariada; dict = override por
        variável; só ``transform='raw'`` e algoritmos de
        :data:`MONOTONE_ALGORITHMS`; sem suporte → aviso e opção ignorada).
        As direções são resolvidas UMA vez, fora do loop de trials.

        Cada trial guarda, em ``trial.user_attrs``, dois grupos de métricas —
        ``modelagem`` (AUC/KS/Gini ou RMSE/MAE/R² na validação) e ``monitoramento``
        (PSI do score DES→validação, volumetria). Com ``log_mlflow=True`` cada
        trial vira um **run aninhado** no MLflow (params + métricas agrupadas por
        ``modelagem/…`` e ``monitoramento/…``), sob um run-pai com o resumo do
        estudo (melhores hiperparâmetros e gráficos do Optuna). Ver também
        :meth:`log_optuna_to_mlflow` para logar um estudo já concluído.

        ``register_model`` (só com ``log_mlflow=True`` e ``fit_best=True``): loga
        também o **modelo re-treinado com os melhores hiperparâmetros** no run-pai
        (``mlflow.sklearn``). Com ``mlflow_model_name``, registra no Model Registry
        evitando colidir com um nome já existente — vira ``nome_v2``, ``nome_v3``…
        (ou, sem acesso ao registry, ganha um carimbo de tempo). Ver
        :meth:`_unique_registered_model_name`.

        ``progress_callback`` (opcional): chamado após CADA trial com
        ``(n_concluidos, n_total, melhor_valor)`` — útil p/ barra de progresso na
        UI; com CV, também após cada fold do trial em curso (mesma assinatura).
        Exceções no callback são ignoradas (não derrubam o tuning). O
        cancelamento (:meth:`cancel_tuning`) também responde por fold."""
        optuna = _require("optuna", "optuna")
        from sklearn.model_selection import train_test_split

        # herda a seed do segmenter quando não especificada (reprodutibilidade)
        if random_state is None:
            random_state = self.random_state
        if algorithm is None:
            algorithm = "logistica" if self.task_type == "classification" else "hist_gradient_boosting"
        if algorithm not in TUNABLE_ALGORITHMS:
            raise ValueError(f"Algoritmo {algorithm!r} não é tunável. "
                             f"Use um de {TUNABLE_ALGORITHMS}.")
        is_clf = self.task_type == "classification"
        feats = list(features) if features is not None else self.selected_features()
        if not feats:
            feats = list(self.candidates)
        # --- validação: única (OOT/split) ou cruzada (cv=k; temporal opcional) --
        if cv is not None:
            cv = int(cv)
            if cv < 2:
                raise ValueError("cv deve ser um inteiro ≥ 2 (ou None para a "
                                 "validação única OOT/split).")
        if time_aware and cv is None:
            raise ValueError("time_aware=True requer cv=k (folds temporais "
                             "contíguos por safra).")
        if time_aware and self.date_col is None:
            raise ValueError("time_aware=True requer um segmenter com date_col "
                             "(coluna de safra).")
        lam = 0.0 if stability_penalty is None else float(stability_penalty)
        if lam < 0:
            raise ValueError("stability_penalty (λ) deve ser ≥ 0.")

        cols_tune = list(dict.fromkeys(
            [*feats, self.target] + ([self.date_col] if self.date_col else [])))
        tr = self.df.loc[self._fit_mask(), cols_tune]
        if transform == "raw":
            self._check_design_memory(tr, feats)
        # fold_data: lista de (Xtr, ytr, Xva, yva) — 1 tupla na validação única;
        # k folds (ou k−1 temporais) com cv. O objetivo do trial é a média.
        fold_data = []
        if cv is not None:
            Xd = tr[feats]
            yd = tr[self.target].astype(int) if is_clf else tr[self.target]
            if time_aware:
                # folds TEMPORAIS por safra: safras ordenadas partidas em k blocos
                # contíguos; treino = blocos anteriores, validação = bloco corrente
                # (janela expansiva ⇒ k−1 folds, sem vazamento de futuro).
                datas = tr[self.date_col]
                safras = np.sort(datas.dropna().unique())
                grupos = [g for g in np.array_split(safras, cv) if len(g)]
                if len(grupos) < 2:
                    raise ValueError(f"Safras insuficientes ({len(safras)}) para "
                                     f"cv={cv} temporal (time_aware=True).")
                d_arr = datas.to_numpy()
                for i in range(1, len(grupos)):
                    m_tr = np.isin(d_arr, np.concatenate(grupos[:i]))
                    m_va = np.isin(d_arr, grupos[i])
                    fold_data.append((Xd[m_tr], yd[m_tr], Xd[m_va], yd[m_va]))
                val_sample = f"cv{cv}_temporal"
            else:
                from sklearn.model_selection import KFold, StratifiedKFold
                splitter = (StratifiedKFold if is_clf else KFold)(
                    n_splits=cv, shuffle=True, random_state=random_state)
                for idx_tr, idx_va in splitter.split(Xd, yd):
                    fold_data.append((Xd.iloc[idx_tr], yd.iloc[idx_tr],
                                      Xd.iloc[idx_va], yd.iloc[idx_va]))
                val_sample = f"cv{cv}"
            Xtr = Xd                       # p/ resolução de monotonicidade abaixo
        else:
            va = None
            oot = self._oot_sample()
            if oot and oot != self.ref_sample:
                vf = self.df.loc[self._fit_mask(oot), cols_tune]
                if len(vf) >= 50:
                    va = vf
            used_oot = va is not None          # o OOT de fato virou a validação?
            if va is None:                     # sem OOT com alvo → split do DES
                strat = tr[self.target].astype(int) if is_clf else None
                tr, va = train_test_split(tr, test_size=0.25, random_state=random_state,
                                          stratify=strat)
            ytr = tr[self.target].astype(int) if is_clf else tr[self.target]
            yva = va[self.target].astype(int) if is_clf else va[self.target]
            Xtr, Xva = tr[feats], va[feats]
            # `va is not None` é SEMPRE verdadeiro após o split acima — sem used_oot,
            # um OOT de <50 linhas cairia no holdout do DES mas seria rotulado como OOT
            # (tag MLflow enganosa para governança).
            val_sample = oot if used_oot else "split"
            fold_data.append((Xtr, ytr, Xva, yva))
        # restrições de monotonicidade (opcional): resolvidas UMA vez fora do
        # objective (as direções univariadas e os nomes pós-transformação não
        # mudam entre trials) e injetadas no estimador de CADA trial.
        mono_dirs = self._resolve_monotone(monotone, feats, algorithm, transform)
        mono_params, mono_pandas = ({}, False)
        if mono_dirs:
            mono_params, mono_pandas = self._monotone_est_params(algorithm, mono_dirs,
                                                                 Xtr, feats)

        metric_key = "auc" if is_clf else "r2"

        def _nanmean(vals):
            """Média ignorando NaN; NaN se nada sobrar (sem RuntimeWarning)."""
            arr = np.asarray([v for v in vals if v == v], dtype="float64")
            return float(arr.mean()) if arr.size else float("nan")

        def _round6(v):
            return round(v, 6) if v == v else float("nan")

        def objective(trial):
            from ...monitoring import psi as _psi_num
            hp = _optuna_space(trial, algorithm, search_space)
            # balanceamento de classes: entra no espaço de busca (class_balance
            # =None) ou fica fixo em todos os trials (True/False). Pseudo-
            # parâmetro nosso — traduzido em _build_estimator, não vai direto ao
            # estimador (GradientBoosting usa sample_weight balanceado no fit).
            cb_flag = False
            if is_clf:
                cb_flag = (bool(trial.suggest_categorical("class_balance", [False, True]))
                           if class_balance is None else bool(class_balance))
            # um passo por fold: métricas de validação, métrica de treino (p/ o
            # gap de overfit) e PSI treino→validação. Na validação única (cv=None)
            # há 1 fold e o comportamento é idêntico ao histórico.
            fold_mods, fold_vals, fold_trs, fold_psis = [], [], [], []
            n_tr_l, n_va_l = [], []
            last_err = None
            for k_i, (Xtr_f, ytr_f, Xva_f, yva_f) in enumerate(fold_data):
                try:
                    counts, fit_kw = None, {}
                    if cb_flag:
                        if algorithm == "gradient_boosting":
                            fit_kw["est__sample_weight"] = _balanced_sample_weight(ytr_f)
                        else:
                            n_pos_f = int((ytr_f == 1).sum())
                            counts = (int(len(ytr_f)) - n_pos_f, n_pos_f)
                    pipe = self._build_pipeline(feats, algorithm, hp, transform=transform,
                                                class_counts=counts)
                    if mono_params:
                        self._set_monotone_on_pipe(pipe, mono_params, mono_pandas)
                    pipe.fit(Xtr_f, ytr_f, **fit_kw)
                    s = self._predict_score_array(pipe, Xva_f)
                    s_tr = self._predict_score_array(pipe, Xtr_f)
                    m_va = (classification_metrics(yva_f, s) if is_clf
                            else regression_metrics(yva_f, s))
                    m_tr = (classification_metrics(ytr_f, s_tr) if is_clf
                            else regression_metrics(ytr_f, s_tr))
                    try:              # PSI é MONITORAMENTO — não pode derrubar o trial
                        _psi_val = round(float(_psi_num(s_tr, s)), 6)
                    except Exception:
                        _psi_val = float("nan")
                    fold_mods.append(m_va)
                    fold_vals.append(float(m_va.get(metric_key, float("nan"))))
                    fold_trs.append(float(m_tr.get(metric_key, float("nan"))))
                    fold_psis.append(_psi_val)
                    n_tr_l.append(int(len(Xtr_f)))
                    n_va_l.append(int(len(Xva_f)))
                except Exception as e:     # fold degenerado (ex.: classe única) — pula
                    last_err = e
                    continue
                if len(fold_data) > 1 and k_i < len(fold_data) - 1:
                    # progresso por FOLD (mantém a UI viva em trials longos de CV)
                    if progress_callback is not None:
                        try:
                            try:
                                best = float(study.best_value)
                            except Exception:  # noqa: BLE001 - ainda sem trial completo
                                best = float("nan")
                            progress_callback(trial.number, n_trials, best)
                        except Exception:  # noqa: BLE001 - progresso é cosmético
                            pass
                    # pruning (opcional): reporta a média parcial dos folds
                    if pruner is not None:
                        trial.report(_nanmean(fold_vals), step=k_i)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                if self._tuning_cancel.is_set():   # cancelamento responde por fold
                    break
            if not fold_mods:
                if last_err is not None:           # TODOS os folds falharam → trial FAILED
                    raise last_err
                return float("-1e9")               # cancelado antes do 1º fold
            # agregação: média dos folds (1 fold ⇒ valores idênticos ao histórico)
            if len(fold_mods) == 1:
                modelagem = fold_mods[0]
            else:
                chaves = sorted(set().union(*(m.keys() for m in fold_mods)))
                modelagem = {k: _round6(_nanmean([float(m.get(k, float("nan")))
                                                  for m in fold_mods]))
                             for k in chaves}
            valor_base = _nanmean(fold_vals)
            metric_tr = _nanmean(fold_trs)
            gap = metric_tr - valor_base if (metric_tr == metric_tr
                                             and valor_base == valor_base) else float("nan")
            psi_val = _nanmean(fold_psis)
            monitoramento = {
                "psi_score_des_val": _round6(psi_val),
                "n_treino": int(np.mean(n_tr_l)), "n_validacao": int(np.mean(n_va_l)),
            }
            if len(fold_data) > 1:
                monitoramento["cv_folds"] = int(len(fold_mods))
            # objetivo penalizado por instabilidade: excesso de PSI (>0,10) e gap
            # de overfit treino−validação (>0,05), ambos escalados por λ. NaN em
            # PSI/gap não penaliza (monitoramento nunca derruba o trial).
            pen_psi = lam * max(0.0, psi_val - 0.10) if (lam > 0 and psi_val == psi_val) else 0.0
            pen_gap = lam * max(0.0, gap - 0.05) if (lam > 0 and gap == gap) else 0.0
            valor = valor_base - pen_psi - pen_gap
            trial.set_user_attr("modelagem", modelagem)
            trial.set_user_attr("monitoramento", monitoramento)
            trial.set_user_attr("objetivo", {
                "metric_val": _round6(valor_base), "metric_treino": _round6(metric_tr),
                "gap_treino_val": _round6(gap), "psi_score_des_val": _round6(psi_val),
                "lambda": lam, "pen_psi": _round6(pen_psi), "pen_gap": _round6(pen_gap),
                "valor": _round6(valor),
            })
            trial.set_user_attr("val_sample", val_sample)
            return float(valor) if np.isfinite(valor) else float("-1e9")

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        # NÃO limpamos a flag de cancelamento aqui: quando o tuning roda numa thread
        # de fundo (UI), a preparação acima leva tempo e o botão "Cancelar" já está
        # ativo — um clear aqui apagaria um cancelamento pedido nessa janela. O
        # chamador (UI) limpa a flag na main thread ANTES de habilitar o botão; e
        # este método a deixa limpa NO FIM (para o próximo tuning).
        # pruner opcional: 'median' → MedianPruner; também aceita uma instância.
        # Só é acionado quando há passos intermediários (CV com ≥2 folds).
        pruner_obj = None
        if pruner is not None:
            if isinstance(pruner, str):
                if pruner.lower() != "median":
                    raise ValueError("pruner: use None, 'median' ou uma instância "
                                     "de optuna.pruners.BasePruner.")
                pruner_obj = optuna.pruners.MedianPruner()
            else:
                pruner_obj = pruner
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=random_state),
                                    pruner=pruner_obj)
        callbacks = []

        def _cancel_cb(study, trial):           # cancelamento pedido pela UI/usuário
            if self._tuning_cancel.is_set():
                study.stop()                      # para após o trial em andamento
        callbacks.append(_cancel_cb)

        if progress_callback is not None:
            def _progress_cb(study, trial):     # chamado após cada trial
                try:
                    best = float(study.best_value) if study.best_trial else float("nan")
                    progress_callback(len(study.trials), n_trials, best)
                except Exception:
                    pass                          # progresso é cosmético; nunca derruba o tuning
            callbacks.append(_progress_cb)

        def _n_complete():
            return sum(1 for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE)

        def _finish_best():
            """Fecha o estudo: grava ``study_``/``tuning_`` e, com ``fit_best``,
            reajusta o modelo com os melhores hiperparâmetros. Se o tuning foi
            **cancelado**, preserva o modelo vigente (não reajusta) — "cancelar"
            significa "não altere meu modelo"."""
            self.study_ = study
            cancelled = self._tuning_cancel.is_set()
            n_ok = _n_complete()
            n_failed = sum(1 for t in study.trials
                           if t.state == optuna.trial.TrialState.FAIL)
            best_val = float(study.best_value) if n_ok else float("nan")
            # trials DEGENERADOS (métrica não-finita: AUC/R² NaN) retornam a sentinela
            # -1e9 e ficam COMPLETE. Se o MELHOR ainda é a sentinela, NENHUM trial
            # produziu métrica válida — não reajustar com hiperparâmetros de um trial
            # degenerado (o "best" seria arbitrário).
            degenerate = bool(n_ok) and np.isfinite(best_val) and best_val <= -1e8
            n_pruned = sum(1 for t in study.trials
                           if t.state == optuna.trial.TrialState.PRUNED)
            self.tuning_ = {"algorithm": algorithm, "metric": "auc" if is_clf else "r2",
                            "n_trials": n_ok, "n_failed": n_failed,
                            "n_pruned": n_pruned,
                            "best_value": (round(best_val, 6)
                                           if (n_ok and not degenerate) else float("nan")),
                            "best_params": (dict(study.best_params)
                                            if (n_ok and not degenerate) else {}),
                            "degenerate": degenerate, "cancelled": cancelled,
                            # escolhas de validação/estabilidade (persistidas também
                            # nos params do run-pai do MLflow — governança)
                            "cv": cv, "time_aware": bool(time_aware and cv),
                            "stability_penalty": (lam if stability_penalty is not None
                                                  else None),
                            "pruner": (pruner if isinstance(pruner, (str, type(None)))
                                       else type(pruner).__name__)}
            if degenerate and verbose:
                print("[tune_optuna] aviso: todos os trials produziram métrica inválida "
                      "(AUC/R² não-finito) — modelo NÃO reajustado.")
            if fit_best and n_ok and not cancelled and not degenerate:
                # "class_balance" (se buscado) vem dentro de best_params e o fit
                # o extrai; se foi FIXADO por argumento, repassa o valor fixo.
                cb_fixed = bool(class_balance) if (is_clf and class_balance is not None) else False
                # monotone repassado só se foi de fato aplicado nos trials (evita
                # um segundo aviso quando a opção foi ignorada por falta de suporte)
                self.fit(algorithm=algorithm, hyperparams=study.best_params,
                         features=feats, transform=transform, class_balance=cb_fixed,
                         monotone=(monotone if mono_dirs else None))

        # --- MLflow: run-pai + um run aninhado por trial (opcional) ----------
        if log_mlflow:
            import mlflow
            if mlflow_experiment:                       # explícito vence; senão usa o experimento ativo da sessão
                mlflow.set_experiment(mlflow_experiment)
            run_name = mlflow_run_name or f"optuna_{algorithm}"
            with mlflow.start_run(run_name=run_name):
                def _mlflow_cb(study, trial):
                    self._log_optuna_trial(mlflow, trial, algorithm, transform)
                study.optimize(objective, n_trials=n_trials, timeout=timeout,
                               callbacks=callbacks + [_mlflow_cb], catch=(Exception,))
                # Persiste o resultado + refit ANTES do logging cosmético; e o log no
                # MLflow (resumo do estudo + registro do modelo) vira best-effort: um
                # problema/lentidão no MLflow ou no Model Registry não descarta o tuning
                # nem deixa a run "presa" sem concluir.
                _finish_best()
                try:
                    if _n_complete():            # cancelamento cedo pode não ter trial concluído
                        self._log_optuna_parent(mlflow, study, algorithm, transform, is_clf, feats)
                    if (register_model and fit_best and not self._tuning_cancel.is_set()
                            and _n_complete()):
                        self._log_fitted_model(mlflow, registered_model_name=mlflow_model_name,
                                               verbose=verbose)
                except Exception as _e:          # noqa: BLE001 — logging é best-effort
                    if verbose:
                        print(f"[tune_optuna] aviso: log no MLflow falhou "
                              f"({type(_e).__name__}: {_e}); resultado do tuning preservado.")
        else:
            # catch=(Exception,): um trial que falhe (combo de hiperparâmetros inválido,
            # métrica/predição degenerada, etc.) é marcado FAILED e o estudo CONTINUA. Sem
            # isso, uma ÚNICA falha aborta o tuning inteiro (o modelo "para de treinar" ou
            # a barra de progresso trava sem concluir).
            study.optimize(objective, n_trials=n_trials, timeout=timeout,
                           callbacks=callbacks, catch=(Exception,))
            _finish_best()
        # deixa a flag limpa para o PRÓXIMO tuning (já foi lida em _finish_best).
        self._tuning_cancel.clear()
        return self.tuning_

    def cancel_tuning(self):
        """Sinaliza o **cancelamento** de um :meth:`tune_optuna` em andamento. O
        estudo Optuna para após o trial atual (via ``study.stop()``) e o modelo
        vigente é **preservado** (não é reajustado com os melhores hiperparâmetros).
        No-op se não houver tuning rodando — a flag é limpa no início do próximo
        tuning. Pensado para ser chamado de outra thread (ex.: um botão "Cancelar"
        na UI enquanto o tuning roda numa thread de fundo)."""
        self._tuning_cancel.set()

    @staticmethod
    def _unique_registered_model_name(base: str) -> str:
        """Nome de modelo para o MLflow Model Registry que **não colida** com um
        já existente. Se ``base`` estiver livre, usa-o; senão tenta ``base_v2``,
        ``base_v3``… (o próprio ``base`` conta como v1). Sem acesso ao registry
        (offline/backend sem suporte), anexa um carimbo de tempo
        (``base_AAAAMMDD_HHMMSS``)."""
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()

            def _exists(nome):
                try:
                    client.get_registered_model(nome)
                    return True
                except Exception:
                    return False

            if not _exists(base):
                return base
            for v in range(2, 1000):
                cand = f"{base}_v{v}"
                if not _exists(cand):
                    return cand
        except Exception:
            pass
        import datetime as _dt
        return f"{base}_{_dt.datetime.now():%Y%m%d_%H%M%S}"

    def _log_fitted_model(self, mlflow, registered_model_name=None,
                          artifact_path="modelo", verbose=False):
        """Loga o modelo ajustado no run MLflow **ativo** (``mlflow.sklearn``).
        Com ``registered_model_name``, registra no Model Registry usando um nome
        único (:meth:`_unique_registered_model_name`) para não sobrescrever/colidir
        com um modelo já existente. Retorna o nome efetivamente registrado (ou
        ``None`` se só foi logado como artefato). Best-effort."""
        if getattr(self, "model", None) is None:
            return None
        name = (self._unique_registered_model_name(registered_model_name)
                if registered_model_name else None)
        try:
            import mlflow.sklearn
            # cloudpickle: default histórico e compatível com mlflow 2.9→3.x — o
            # 3.x passou a serializar sklearn via 'skops', que rejeita tipos como
            # numpy.dtype (comum em RF/GBM) e derrubaria o log do modelo.
            mlflow.sklearn.log_model(self.model, artifact_path,
                                     registered_model_name=name,
                                     serialization_format="cloudpickle")
            mlflow.set_tag("modelo_registrado", name or "(artefato, sem registry)")
            if verbose:
                print(f"[mlflow] modelo logado"
                      + (f" e registrado como '{name}'." if name else " (artefato)."))
            return name
        except Exception as e:
            if verbose:
                print(f"[mlflow] modelo não logado: {e}")
            return None

    # ------------------------------------------------------------------
    # MLflow: logging dos trials do Optuna (agrupado por finalidade)
    # ------------------------------------------------------------------
    @staticmethod
    def _log_optuna_trial(mlflow, trial, algorithm, transform) -> None:
        """Loga UM trial do Optuna como run aninhado no MLflow, com os parâmetros
        e as métricas agrupadas por finalidade (``modelagem/…`` e
        ``monitoramento/…``). Best-effort — nunca derruba o tuning."""
        try:
            with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
                mlflow.set_tags({
                    "grupo": "trial",
                    "algoritmo": algorithm,
                    "transform": transform,
                    "optuna_trial": trial.number,
                    "optuna_state": getattr(trial.state, "name", str(trial.state)),
                    "val_sample": trial.user_attrs.get("val_sample", "?"),
                })
                # parâmetros do trial (os hiperparâmetros sugeridos)
                mlflow.log_params({f"hp/{k}": v for k, v in trial.params.items()})
                # métricas agrupadas: prefixo por finalidade -> "abas" na leitura
                # ('objetivo' traz as componentes do valor penalizado, se houver)
                for grupo in ("modelagem", "monitoramento", "objetivo"):
                    for nome, valor in (trial.user_attrs.get(grupo) or {}).items():
                        try:
                            v = float(valor)
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(v):
                            mlflow.log_metric(f"{grupo}/{nome}", v)
                if trial.value is not None and np.isfinite(trial.value):
                    mlflow.log_metric("modelagem/objetivo", float(trial.value))
        except Exception:  # noqa: BLE001 - logging é best-effort
            pass

    def _log_optuna_parent(self, mlflow, study, algorithm, transform, is_clf, feats) -> None:
        """Loga no run-pai o resumo do estudo: config, melhores hiperparâmetros/
        métricas e os gráficos do Optuna (história e importância)."""
        import os
        import tempfile
        try:
            mlflow.set_tags({"framework": "yggdrasil-ml", "grupo": "tuning-optuna",
                             "algoritmo": algorithm, "trained_by": "richard-guilherme"})
            mlflow.log_params({
                "algoritmo": algorithm, "transform": transform,
                "n_trials": len(study.trials), "n_features": len(feats),
                "metric": "auc" if is_clf else "r2", "direction": "maximize",
            })
            # escolhas de validação/estabilidade do tuning (governança): CV,
            # split temporal, λ da penalização e pruner — vindas de self.tuning_.
            tun = getattr(self, "tuning_", None) or {}
            mlflow.log_params({
                "cv": tun.get("cv") or "nenhum (OOT/split)",
                "time_aware": bool(tun.get("time_aware")),
                "stability_penalty": (tun.get("stability_penalty")
                                      if tun.get("stability_penalty") is not None
                                      else "nenhuma"),
                "pruner": tun.get("pruner") or "nenhum",
            })
            mlflow.log_params({f"melhor_hp/{k}": v for k, v in study.best_params.items()})
            if study.best_value is not None and np.isfinite(study.best_value):
                mlflow.log_metric("melhor/objetivo", float(study.best_value))
            best = study.best_trial
            for grupo in ("modelagem", "monitoramento", "objetivo"):
                for nome, valor in (best.user_attrs.get(grupo) or {}).items():
                    try:
                        v = float(valor)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(v):
                        mlflow.log_metric(f"melhor/{grupo}/{nome}", v)
            mlflow.log_dict(dict(study.best_params), "optuna/best_params.json")
            # gráficos do Optuna (best-effort: requerem matplotlib e ≥2 trials)
            tmp = tempfile.mkdtemp(prefix="optuna_viz_")
            try:
                import matplotlib.pyplot as plt
                from optuna.visualization.matplotlib import (
                    plot_optimization_history, plot_param_importances)
                for fn, nome in ((plot_optimization_history, "optimization_history"),
                                 (plot_param_importances, "param_importances")):
                    try:
                        ax = fn(study)
                        fig = ax.figure
                        p = os.path.join(tmp, f"{nome}.png")
                        fig.savefig(p, dpi=110, bbox_inches="tight")
                        plt.close(fig)
                        mlflow.log_artifact(p, artifact_path="optuna")
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001 - viz é opcional
                pass
        except Exception:  # noqa: BLE001 - logging é best-effort
            pass

    def log_optuna_to_mlflow(self, experiment=None, run_name=None):
        """Loga no MLflow um estudo do Optuna **já concluído** (``self.study_``,
        produzido por :meth:`tune_optuna`) — run-pai com o resumo + um run aninhado
        por trial, com métricas agrupadas por ``modelagem/…`` e ``monitoramento/…``.
        Use quando o tuning rodou sem ``log_mlflow=True`` e você quer registrá-lo
        depois. Retorna o ``run_id`` do run-pai."""
        if getattr(self, "study_", None) is None:
            raise RuntimeError("Rode tune_optuna antes (não há self.study_).")
        import mlflow
        algorithm = self.tuning_.get("algorithm", "?")
        is_clf = self.task_type == "classification"
        transform = getattr(self, "feature_transform", "raw")
        feats = list(self.model_features or self.selected_features() or self.candidates)
        if experiment:                                  # explícito vence; senão usa o experimento ativo da sessão
            mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name or f"optuna_{algorithm}") as run:
            for trial in self.study_.trials:
                self._log_optuna_trial(mlflow, trial, algorithm, transform)
            self._log_optuna_parent(mlflow, self.study_, algorithm, transform, is_clf, feats)
            return run.info.run_id

    def set_model(self, model, features=None):
        """Recebe um modelo já ajustado (sklearn/pipeline). ``features`` indica as
        colunas de entrada (default: as selecionadas). Calcula ``score_``."""
        self.model = model
        self.algorithm = self.algorithm or "externo"
        # um _TwoStageModel injetado por fora também ativa o modo hurdle
        self.two_stage = isinstance(model, _TwoStageModel)
        self.two_stage_threshold = model.threshold if self.two_stage else None
        self.model_features = (list(features) if features is not None
                               else (self.model_features or self.selected_features()
                                     or list(self.candidates)))
        # modelo externo é opaco e recebe as colunas CRUAS em _compute_score; um
        # feature_transform="woe" residual de um fit anterior faria predict/assign
        # aplicarem WoE antes do modelo (≠ do score_ aqui). Volta a "raw" p/ manter
        # escoragem consistente. (Se o modelo externo já espera WoE, passe um
        # pipeline que faça isso internamente.)
        self.feature_transform = "raw"
        # modelo externo: não sabemos se veio balanceado nem com restrições de
        # monotonicidade — limpa as flags do fit.
        self.class_balance = False
        self.monotone = None
        self.monotone_dirs_ = {}
        self.calibration_ = None           # modelo novo ⇒ a camada antiga não vale
        self.score_ = self._compute_score(self.df)
        self._shap_cache = {}
        return self

    @property
    def score_points_(self):
        """Score em **escala de negócio** (0–``score_scale``, i.e. 0–1000 por
        padrão): o ``score_`` cru multiplicado por ``score_scale``. É a escala
        apresentada na escoragem (:meth:`predict`/:meth:`assign`) e nos eixos de
        score dos gráficos. ``score_`` continua cru (probabilidade/predição) para
        métricas e calibração. ``None`` se o modelo ainda não foi ajustado."""
        return None if self.score_ is None else self.score_ * self.score_scale

    # ---- fórmula do modelo linear/logístico (coeficientes) ----
    def _design_feature_names(self, pre, use_labels=True) -> list:
        """Nomes dos termos do desenho (saída do ``ColumnTransformer``), sem o
        prefixo ``num__``/``cat__``. Aplica ``feature_labels`` quando possível."""
        raw = None
        if pre is not None and hasattr(pre, "get_feature_names_out"):
            try:
                raw = list(pre.get_feature_names_out())
            except Exception:
                raw = None
        if raw is None:
            raw = list(self.model_features)
        return [self._display_feature_name(nm, use_labels) for nm in raw]

    def _display_feature_name(self, nm, use_labels=True) -> str:
        """Nome de exibição de UM termo do desenho/SHAP: remove o prefixo
        ``num__``/``cat__``, desembrulha ``WoE(...)``/``bin(...)`` e aplica o alias
        de ``feature_labels`` quando houver — mesma convenção da fórmula."""
        dummy = nm.startswith("dum__")
        for p in ("num__", "cat__", "ord__", "woe__", "dum__"):
            if nm.startswith(p):
                nm = nm[len(p):]
                break
        if dummy and "=" in nm:                 # dummy de scorecard: 'var=faixa'
            f, faixa = nm.split("=", 1)
            lbl = self.feature_labels.get(f, f) if use_labels else f
            return f"{lbl} = {faixa}"
        if dummy:                               # dummies agrupadas (SHAP): a variável
            lbl = self.feature_labels.get(nm, nm) if use_labels else nm
            return f"{lbl} (faixas)"
        # termos transformados vêm como 'WoE(feat)'/'bin(feat)'/'ord(feat)':
        # rotula o miolo
        wrap = None
        for w in ("WoE", "bin", "ord"):
            if nm.startswith(f"{w}(") and nm.endswith(")"):
                wrap, nm = w, nm[len(w) + 1:-1]
                break
        if use_labels and nm in self.feature_labels:
            nm = self.feature_labels[nm]
        return f"{wrap}({nm})" if wrap else nm

    def _logit_wald_pvalues(self, est, pre, names) -> list:
        """p-valores de Wald (aprox.) da **logística**, como LISTA alinhada por
        POSIÇÃO a ``names``/``coef`` (não um dict por nome — dois termos com o mesmo
        rótulo de exibição colapsariam e receberiam o p-valor errado): z = coef/EP,
        EP da diagonal de inv(Xᵀ W X), W = p(1−p). Aproximação (a logística do
        sklearn é regularizada); serve como indicação de significância."""
        from scipy.stats import norm
        H = None
        # Xᵀ W X acumulada em blocos de linhas: a matriz de desenho densa da
        # referência inteira (e as duas cópias com intercepto/peso) não chega a
        # existir de uma vez.
        for bloco in self._row_blocks(self._fit_mask(), self.model_features):
            Xd = self._design_block(pre, bloco)
            p = np.clip(est.predict_proba(Xd)[:, 1], 1e-6, 1 - 1e-6)
            w = p * (1 - p)
            Xf = np.column_stack([np.ones(len(Xd)), Xd])      # intercepto + design
            Hb = Xf.T @ (Xf * w[:, None])
            H = Hb if H is None else H + Hb
        if H is None:                                         # referência sem alvo
            return [float("nan")] * len(names)
        cov = np.linalg.pinv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        beta = np.concatenate([np.ravel(est.intercept_), np.ravel(est.coef_)])
        z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
        pv = 2.0 * (1.0 - norm.cdf(np.abs(z)))
        return [float(pv[i + 1]) for i in range(len(names))]   # alinhado a names/coef

    @staticmethod
    def _signif_stars(p) -> str:
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return ""
        return ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05
                else "." if p < 0.10 else "n.s.")

    def scorecard_sign_check(self) -> pd.DataFrame:
        """Sinal dos coeficientes das **dummies de scorecard** no modelo
        linear/logístico ajustado: ``variavel``, ``faixa``, ``coef`` e ``sinal_ok``
        (``coef < 0``). Com a pior faixa como referência, toda dummy deve sair
        NEGATIVA (faixa melhor que a pior); positiva indica que, no multivariado,
        a faixa inverteu — em geral por correlação com outra variável. Vazio se
        não há dummies de scorecard no modelo ou o algoritmo não é linear."""
        cols = ["variavel", "faixa", "coef", "sinal_ok"]
        dums = self.scorecard_dummy_features(self.model_features or [])
        if (not dums or self.model is None or self.two_stage
                or self.algorithm not in ("logistica", "linear")
                or not hasattr(self.model, "named_steps")):
            return pd.DataFrame(columns=cols)
        est = self.model.named_steps["est"]
        pre = self.model.named_steps.get("pre")
        coef = np.ravel(np.asarray(getattr(est, "coef_", []), dtype="float64"))
        try:
            nomes = list(pre.get_feature_names_out())
        except Exception:                           # noqa: BLE001 — sem nomes, sem checagem
            return pd.DataFrame(columns=cols)
        rows = []
        for nm, c in zip(nomes, coef):
            if nm.startswith("dum__") and "=" in nm:
                f, faixa = nm[len("dum__"):].split("=", 1)
                rows.append({"variavel": f, "faixa": faixa, "coef": round(float(c), 6),
                             "sinal_ok": bool(c < 0)})
        return pd.DataFrame(rows, columns=cols)

    def _avisa_sinal_scorecard(self) -> None:
        """Pós-treino: avisa as dummies de scorecard que ficaram com coeficiente
        ≥ 0 (o scorecard exige todas negativas)."""
        chk = self.scorecard_sign_check()
        ruins = chk[~chk["sinal_ok"]] if not chk.empty else chk
        if ruins.empty:
            return
        lista = ", ".join(f"{self.label(r.variavel)} = {r.faixa} (coef {r.coef:+.4f})"
                          for r in ruins.itertuples())
        warnings.warn(
            f"Scorecard: {len(ruins)} dummy(ies) com coeficiente ≥ 0 — {lista}. Com a "
            "pior faixa como referência o esperado é negativo; isso costuma vir de "
            "correlação com outra variável do modelo ou de faixas com risco parecido. "
            "Revise: funda faixas, retire a variável correlacionada ou a própria variável.")

    def model_coefficients(self, use_labels=True, ordem="blocos") -> pd.DataFrame:
        """Coeficientes do modelo **linear/logístico** ajustado: ``termo``, ``coef``
        e — na classificação — ``odds_ratio`` (``exp(coef)``). Na **logística**
        inclui também ``p_valor`` (Wald aprox.) e ``signif`` (estrelas). O intercepto
        fica em ``.attrs['intercept']``. Erro para modelos não-lineares (use SHAP).

        ``variavel``/``variavel_label``: a variável ORIGINAL de cada termo (dummies
        de scorecard, one-hot e WoE apontam para a variável de origem) e
        ``termo_curto``: o termo sem o nome da variável (a faixa/categoria).

        ``ordem``: ``"blocos"`` (padrão) agrupa os termos por variável — blocos
        ordenados pela maior |coef| do bloco e, dentro dele, por |coef| —;
        ``"magnitude"`` ordena todos os termos por |coef| (comportamento antigo)."""
        if self.model is None:
            raise RuntimeError("Ajuste o modelo antes (fit / set_model).")
        if self.algorithm not in ("logistica", "linear"):
            raise ValueError(
                "Fórmula de coeficientes disponível apenas para Regressão "
                f"Logística/Linear (algoritmo atual: {self.algorithm!r}). "
                "Para modelos não-lineares use os gráficos SHAP.")
        est = self.model.named_steps["est"] if hasattr(self.model, "named_steps") else self.model
        pre = (self.model.named_steps.get("pre")
               if hasattr(self.model, "named_steps") else None)
        coef = np.ravel(np.asarray(getattr(est, "coef_", []), dtype="float64"))
        intercept = float(np.ravel(np.asarray(getattr(est, "intercept_", [0.0])))[0])
        names = self._design_feature_names(pre, use_labels=use_labels)
        try:
            crus = [str(n) for n in pre.get_feature_names_out()]
        except Exception:                                 # noqa: BLE001
            crus = list(names)
        if len(names) != len(coef):                       # robustez a divergências
            names = [f"x{i}" for i in range(len(coef))]
            crus = list(names)
        rows = [{"termo": nm, "coef": round(float(c), 6)} for nm, c in zip(names, coef)]
        out = pd.DataFrame(rows, columns=["termo", "coef"])
        if not out.empty:
            out["variavel"] = [self._original_feature_of(c) for c in crus]
            out["variavel_label"] = out["variavel"].map(
                lambda f: self.label(f) if use_labels else f)
            out["termo_curto"] = [self._termo_curto(t, v, lv)
                                  for t, v, lv in zip(out["termo"], out["variavel"],
                                                      out["variavel_label"])]
        if self.task_type == "classification" and not out.empty:
            out["odds_ratio"] = np.exp(out["coef"]).round(4)
        if self.algorithm == "logistica" and not out.empty:
            try:                                   # p-valor de Wald (aprox.), por POSIÇÃO
                pvals = self._logit_wald_pvalues(est, pre, names)   # lista alinhada a coef
                if len(pvals) == len(out):
                    out["p_valor"] = [round(float(p), 4) for p in pvals]
                    out["signif"] = out["p_valor"].map(self._signif_stars)
            except Exception:
                pass
        if out.empty:
            pass
        elif ordem == "magnitude":
            out = out.reindex(out["coef"].abs().sort_values(ascending=False).index)
        else:
            # blocos por variável, na ordem da maior |coef| do bloco; dentro do
            # bloco, por |coef| — a leitura fica "variável a variável"
            absc = out["coef"].abs()
            peso = absc.groupby(out["variavel"]).transform("max")
            out = (out.assign(_p=peso, _a=absc)
                   .sort_values(["_p", "variavel", "_a"], ascending=[False, True, False])
                   .drop(columns=["_p", "_a"]))
        out = out.reset_index(drop=True)
        out.attrs["intercept"] = round(intercept, 6)
        return out

    @staticmethod
    def _termo_curto(termo, var, var_label) -> str:
        """O termo sem o nome da variável: ``renda = (1500, 3000]`` → ``(1500, 3000]``,
        ``uf_SP`` → ``SP``; termos 1:1 (numérica, WoE) ficam como estão."""
        termo = str(termo)
        for pref in (f"{var_label} = ", f"{var} = ", f"{var_label}_", f"{var}_"):
            if termo.startswith(pref) and len(termo) > len(pref):
                return termo[len(pref):]
        return termo

    def vif_table(self, use_labels=True) -> pd.DataFrame:
        """VIF (fator de inflação de variância) de cada termo da **matriz de
        desenho pós-transformação** do modelo vigente — a mesma dos coeficientes
        (:meth:`model_coefficients`): WoE/risco do bin por variável em
        ``transform='woe'`` ou imputação + one-hot em ``transform='raw'``.
        Leitura clássica de multicolinearidade para modelos lineares/logísticos:
        ``VIF = 1/(1−R²)`` da regressão de cada termo sobre os demais (com
        intercepto). Regra de bolso: < 5 ok · 5–10 atenção · > 10 alto.

        Calculado a partir do fator R de um QR da matriz de desenho acumulado em
        blocos de linhas (ver :meth:`_qr_acumulado`): os mesmos valores do
        ``statsmodels``/``LinearRegression`` sem rodar uma regressão sobre a base
        inteira por termo. Devolve ``termo, vif, avaliacao`` ordenado do maior
        para o menor VIF (``inf`` = colinearidade perfeita; termo constante sai
        ``NaN``)."""
        if self.model is None:
            raise RuntimeError("Ajuste o modelo antes (fit / set_model).")
        pre = (self.model.named_steps.get("pre")
               if hasattr(self.model, "named_steps") else None)
        # diagnóstico: acima de max_linhas_graficos linhas de treino, o QR roda
        # numa amostra aleatória (semente fixa) — o VIF estimado não muda de
        # leitura (<5 · 5–10 · >10) e a transformação da base inteira sai do caminho
        linhas = np.asarray(self._fit_mask(), dtype=bool)
        cap = self.max_linhas_graficos
        if cap and int(linhas.sum()) > int(cap):
            pos = np.flatnonzero(linhas)
            escolha = np.random.default_rng(self.random_state).choice(
                pos.size, int(cap), replace=False)
            linhas = np.zeros_like(linhas)
            linhas[pos[np.sort(escolha)]] = True
        n, R, constante = self._qr_acumulado(
            self._design_block(pre, bloco)
            for bloco in self._row_blocks(linhas, self.model_features))
        names = self._design_feature_names(pre, use_labels=use_labels)
        k = R.shape[1] - 1 if R is not None else len(names)
        if len(names) != k:                              # robustez a divergências
            names = [f"x{i}" for i in range(k)]
        vifs = (self._vif_from_qr(R, constante) if R is not None and n >= 2
                else [float("nan")] * len(names))
        out = pd.DataFrame({
            "termo": names,
            "vif": [round(v, 3) if np.isfinite(v) else v for v in vifs]})
        out["avaliacao"] = ["—" if np.isnan(v)
                            else "alto" if v > 10 else "atenção" if v >= 5 else "ok"
                            for v in vifs]
        return (out.sort_values("vif", ascending=False, na_position="last")
                .reset_index(drop=True))

    @staticmethod
    def _design_block(pre, X) -> np.ndarray:
        """Bloco denso (float64) da matriz de desenho para as linhas cruas ``X``."""
        Xd = pre.transform(X) if pre is not None else X.to_numpy(dtype="float64")
        return Xd.toarray() if hasattr(Xd, "toarray") else np.asarray(Xd, dtype="float64")

    @staticmethod
    def _qr_acumulado(blocks) -> tuple:
        """Fator R de ``[1, X − x₀]`` por QR em blocos de linhas (TSQR):
        ``R ← qr([R; bloco]).R``. Uma passada e memória O(bloco × k). Ao
        contrário de ``XᵀX``, o QR não eleva ao quadrado o número de condição
        de X: colunas quase duplicadas (saldo e saldo + encargos, correlação
        1 − 1e-10) seguem distinguíveis da colinearidade exata, como no OLS do
        statsmodels. ``x₀`` é a média do 1º bloco (condiciona colunas de média
        alta frente ao intercepto). Linhas com NaN ou ±inf saem. Devolve
        ``(n, R, constante)``, com ``constante[j]`` = coluna sem variação;
        ``(0, None, None)`` sem linhas."""
        n, desloc, R, mn, mx = 0, None, None, None, None
        for B in blocks:
            B = np.asarray(B, dtype="float64")
            B = B[np.isfinite(B).all(axis=1)]
            if not len(B):
                continue
            if desloc is None:
                desloc = B.mean(axis=0)
                mn, mx = B.min(axis=0), B.max(axis=0)
            else:
                mn, mx = np.minimum(mn, B.min(axis=0)), np.maximum(mx, B.max(axis=0))
            topo = 0 if R is None else R.shape[0]
            A = np.empty((topo + len(B), B.shape[1] + 1))
            if topo:
                A[:topo] = R
            A[topo:, 0] = 1.0
            A[topo:, 1:] = B - desloc
            R = np.linalg.qr(A, mode="r")
            n += len(B)
        if n == 0:
            return 0, None, None
        return n, R, mn == mx

    @staticmethod
    def _vif_from_qr(R, constante) -> list:
        """VIF de cada coluna a partir do fator R de ``[1, X]``.

        Como ``[1, X] = QR`` com Q ortonormal, a soma de quadrados dos resíduos
        da regressão da coluna i nas demais (com intercepto) é a mesma do
        mínimos quadrados sobre as colunas de R, uma matriz (k+1) × k: é o OLS
        do ``statsmodels.variance_inflation_factor``/:meth:`_vif_values_sklearn`
        sem voltar às n linhas. ``VIF = SQT/SQR``, com a SQT centrada vinda da
        regressão só no intercepto. Coluna constante → NaN (e fica fora dos
        regressores); R² ≥ 1−1e-12 → inf.

        Os regressores entram com norma unitária e corte relativo 1e-10 no
        ``lstsq``: numa colinearidade EXATA (one-hot completo + intercepto) o
        valor singular que sobra é ruído (~1e-13, maior quanto mais blocos o
        QR acumulou) e, sem o corte, vira uma direção espúria que absorve parte
        do resíduo e infla o VIF dos termos comuns. Quase duplicatas reais
        (correlação 1 − 1e-10 ⇒ valor singular ~1e-5) ficam bem acima do corte."""
        k = R.shape[1] - 1
        if k == 1:
            return [1.0]

        def _sqr(cols, y):
            A = R[:, cols]
            normas = np.linalg.norm(A, axis=0)
            A = A / np.where(normas > 0, normas, 1.0)
            coef = np.linalg.lstsq(A, y, rcond=1e-10)[0]
            res = y - A @ coef
            return float(res @ res)

        vivas = [j for j in range(k) if not constante[j]]
        out = []
        for i in range(k):
            if constante[i]:
                out.append(float("nan"))
                continue
            y = R[:, i + 1]
            sqt = _sqr([0], y)
            sqr = _sqr([0] + [j + 1 for j in vivas if j != i], y)
            r2 = 1.0 - sqr / sqt if sqt > 0 else float("nan")
            if not np.isfinite(r2):
                out.append(float("nan"))
            else:
                out.append(float("inf") if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2))
        return out

    @staticmethod
    def _vif_values(X) -> list:
        """VIF por coluna de ``X`` (ver :meth:`_vif_from_qr`)."""
        X = np.asarray(X, dtype="float64")
        n_linhas, k = X.shape
        if k == 0:
            return []
        passo = ModelSegmenter._CHUNK_ROWS
        n, R, constante = ModelSegmenter._qr_acumulado(
            X[i:i + passo] for i in range(0, n_linhas, passo))
        if n < 2:
            return []
        if k == 1:
            return [1.0]
        return ModelSegmenter._vif_from_qr(R, constante)

    @staticmethod
    def _vif_values_sklearn(X) -> list:
        """Fallback do VIF sem ``statsmodels``: ``1/(1−R²)`` da regressão de cada
        coluna sobre as demais via ``sklearn.LinearRegression`` (com intercepto —
        equivalente ao VIF clássico do statsmodels)."""
        from sklearn.linear_model import LinearRegression
        X = np.asarray(X, dtype="float64")
        X = X[~np.isnan(X).any(axis=1)]
        n, k = X.shape
        if k == 0 or n < 2:
            return []
        if k == 1:
            return [1.0]
        out = []
        for i in range(k):
            y = X[:, i]
            if float(np.std(y)) == 0.0:                  # termo constante: indefinido
                out.append(float("nan"))
                continue
            resto = np.delete(X, i, axis=1)
            r2 = float(LinearRegression().fit(resto, y).score(resto, y))
            out.append(float("inf") if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2))
        return out

    def model_formula(self, use_labels=True) -> dict:
        """Fórmula legível do modelo linear/logístico. Devolve um ``dict`` com:
        ``intercept``, ``coef`` (DataFrame ordenado por |coef|), ``z_expr`` (o
        preditor linear como texto), ``text`` (forma completa) e ``latex``."""
        coefs = self.model_coefficients(use_labels=use_labels)
        intercept = float(coefs.attrs.get("intercept", 0.0))
        parts = [f"{intercept:+.4f}"]
        for _, r in coefs.iterrows():
            parts.append(f"{r['coef']:+.4f}·[{r['termo']}]")
        z_expr = "  ".join(parts)
        if self.task_type == "classification":
            text = (f"z = {z_expr}\n"
                    "p = 1 / (1 + exp(−z))   ·   odds(p) = exp(z)")
            latex = (r"\operatorname{logit}(p)=\ln\frac{p}{1-p}=z,\qquad "
                     r"p=\dfrac{1}{1+e^{-z}}")
        else:
            text = f"ŷ = {z_expr}"
            latex = r"\hat{y}=\beta_0+\sum_i \beta_i\,x_i"
        return {"intercept": intercept, "coef": coefs, "z_expr": z_expr,
                "text": text, "latex": latex}

    def _predict_score_array(self, model, X) -> np.ndarray:
        if self.task_type == "classification":
            if hasattr(model, "predict_proba"):
                p = np.asarray(model.predict_proba(X))
                sc = p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else np.ravel(p)
            elif hasattr(model, "decision_function"):
                sc = np.ravel(model.decision_function(X))
            else:
                sc = np.ravel(model.predict(X))
        else:
            sc = np.ravel(model.predict(X))
        # camada de calibração pós-treino (calibrate): entra AQUI, no fluxo único
        # de score — score_, métricas, ratings, predict e a escoragem Spark saem
        # calibrados. Só sobre o modelo VIGENTE: pipes candidatos (CV do fit /
        # backward elimination) seguem crus.
        if getattr(self, "calibration_", None) is not None and model is self.model:
            sc = self._apply_calibration(sc)
        return sc

    def _compute_score(self, df) -> pd.Series:
        faltando = [f for f in self.model_features if f not in df.columns]
        if faltando:
            raise KeyError(
                f"Colunas ausentes para escorar: {faltando}. O modelo espera "
                f"{list(self.model_features)}.")
        n, passo = len(df), int(self._CHUNK_ROWS)
        if n <= passo:
            X = df[self.model_features]
            return pd.Series(self._predict_score_array(self.model, X), index=df.index,
                             name="score", dtype="float64")
        # escoragem em blocos: o pré-processador densifica o one-hot, então
        # escorar a base inteira de uma vez criava linhas × colunas-do-desenho
        # float64 (mais as cópias internas) só para produzir um vetor.
        out = np.empty(n, dtype="float64")
        for i in range(0, n, passo):
            X = df.iloc[i:i + passo][self.model_features]
            out[i:i + passo] = self._predict_score_array(self.model, X)
        return pd.Series(out, index=df.index, name="score", dtype="float64")

    # ------------------------------------------------------------------
    # CALIBRAÇÃO pós-treino: camada sobre o score CRU (antes de score_scale)
    # ------------------------------------------------------------------
    def _apply_calibration(self, sc) -> np.ndarray:
        """Aplica a camada vigente (:attr:`calibration_`) a um array de score
        CRU. Só numpy + parâmetros serializáveis (sem objeto sklearn ajustado):
        a mesma camada roda idêntica no driver e nos executores Spark."""
        cal = self.calibration_
        sc = np.asarray(sc, dtype="float64")
        if not cal:
            return sc
        method = cal.get("method")
        p = cal.get("params") or {}
        if self.task_type == "classification":
            if method == "intercept":
                return _sigmoid_np(_logit_np(sc) + float(p["delta"]))
            if method == "platt":
                return _sigmoid_np(float(p["a"]) * _logit_np(sc) + float(p["b"]))
            if method == "isotonic":
                # interpolação linear entre os degraus da isotônica (mesma regra
                # do sklearn com out_of_bounds='clip'); np.interp já clipa nas pontas
                return np.interp(sc, np.asarray(p["x"], dtype="float64"),
                                 np.asarray(p["y"], dtype="float64"))
        else:
            if method == "intercept":
                if p.get("mode") == "multiplicativo":
                    return sc * float(p["factor"])
                return sc + float(p["shift"])
            if method == "platt":                  # recalibração LINEAR a·ŷ + b
                return float(p["a"]) * sc + float(p["b"])
            if method == "isotonic":
                return np.interp(sc, np.asarray(p["x"], dtype="float64"),
                                 np.asarray(p["y"], dtype="float64"))
        raise ValueError(f"Camada de calibração desconhecida: {method!r}.")

    def _base_score_series(self, df=None) -> pd.Series:
        """Score CRU do modelo — SEM a camada de calibração — na base ``df``
        (default: o df de treino). É sobre ele que :meth:`calibrate` ajusta:
        re-calibrar SUBSTITUI a camada (nunca empilha uma sobre a outra)."""
        if df is None:
            return self._raw_score_df()
        cal, self.calibration_ = self.calibration_, None
        try:
            return self._compute_score(df)
        finally:
            self.calibration_ = cal

    def _raw_score_df(self) -> pd.Series:
        """Score CRU (sem calibração) do ``df`` de treino, memoizado pela
        identidade do modelo — calibrar/comparar/atualizar ratings re-escorava a
        base inteira várias vezes com o MESMO modelo."""
        hit = self.__dict__.get("_raw_score_cache")
        if hit is not None and hit[0] is self.model and hit[1] == len(self.df):
            return hit[2]
        cal, self.calibration_ = self.calibration_, None
        try:
            raw = self._compute_score(self.df)
        finally:
            self.calibration_ = cal
        self._raw_score_cache = (self.model, len(self.df), raw)
        return raw

    def _calibration_xy(self, sample=None):
        """``(y, score cru, score calibrado)`` da amostra, alinhados e sem NaN —
        base do antes×depois (:meth:`calibration_compare` /
        :meth:`plot_calibration_compare`)."""
        raw = self._base_score_series()
        mask = (np.ones(len(self.df), dtype=bool) if self.sample_col is None
                else self._frame_mask(sample))
        y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
        r = raw[mask].to_numpy(dtype="float64")
        ok = ~np.isnan(y) & ~np.isnan(r)
        y, r = y[ok], r[ok]
        c = self._apply_calibration(r) if self.calibration_ is not None else r
        return y, r, np.asarray(c, dtype="float64")

    def calibrate(self, method="intercept", sample=None, target_rate=None,
                  mode="aditivo"):
        """Ajusta uma camada de **calibração pós-treino** sobre o score CRU
        (0–1, ANTES da escala de negócio ``score_scale``), sem re-treinar.

        Classificação (camada sobre a probabilidade prevista):

        * ``'intercept'`` — desloca o intercepto no LOGITO (``p' = σ(logit(p)+δ)``)
          para a média calibrada casar a tendência central: ``target_rate`` se
          informado, senão a taxa observada da amostra do ajuste. Preserva o
          ranking (AUC/KS/Gini inalterados).
        * ``'platt'`` — regressão logística sobre o logito
          (``p' = σ(a·logit(p)+b)``): corrige nível E inclinação
          (sobre/subconfiança). Preserva o ranking quando ``a > 0``.
        * ``'isotonic'`` — regressão isotônica (monotônica, não-paramétrica) do
          observado sobre o previsto. Preserva a ORDENAÇÃO (com possíveis empates).

        Regressão: ``'intercept'`` vira ajuste da média do previsto —
        ``mode='aditivo'`` (``ŷ + shift``) ou ``'multiplicativo'`` (``ŷ ×
        fator``); ``'platt'`` vira a recalibração LINEAR ``a·ŷ + b`` (mínimos
        quadrados); ``'isotonic'`` aplica a isotônica ao previsto.

        A camada entra no **fluxo único de score** (:meth:`_predict_score_array`):
        ``score_``, métricas, ratings, :meth:`predict`/:meth:`score_table` e a
        escoragem Spark (:meth:`apply_spark`) passam a usar o score calibrado.
        Persiste em :meth:`to_dict`/:meth:`save` e volta no :meth:`load`;
        um novo ``fit``/``set_model`` a descarta; :meth:`decalibrate` remove.
        Ratings existentes são reprojetados sobre o novo score (mesmos cortes).

        ``sample``: amostra do ajuste (default: a referência). ``target_rate``:
        tendência central alvo — só com ``method='intercept'``. Devolve ``self``."""
        if self.model is None:
            raise RuntimeError("Ajuste/defina o modelo antes (fit / set_model / load).")
        if method not in ("intercept", "platt", "isotonic"):
            raise ValueError("method deve ser 'intercept', 'platt' ou 'isotonic'.")
        if target_rate is not None and method != "intercept":
            raise ValueError("target_rate só se aplica a method='intercept' "
                             "(platt/isotonic ajustam contra o alvo observado).")
        if mode not in ("aditivo", "multiplicativo"):
            raise ValueError("mode deve ser 'aditivo' ou 'multiplicativo'.")
        sample = sample or self.ref_sample
        if self.sample_col is not None and sample not in self._samples():
            raise ValueError(f"Amostra '{sample}' não encontrada. "
                             f"Disponíveis: {self._samples()}")
        raw = self._base_score_series()
        mask = (np.ones(len(self.df), dtype=bool) if self.sample_col is None
                else self._frame_mask(sample))
        y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
        sc = raw[mask].to_numpy(dtype="float64")
        ok = ~np.isnan(y) & ~np.isnan(sc)
        y, sc = y[ok], sc[ok]
        if y.size == 0:
            raise ValueError(f"Sem observações válidas (alvo e score) na amostra "
                             f"'{sample}' para ajustar a calibração.")
        is_clf = self.task_type == "classification"
        params: dict
        if method == "intercept":
            alvo = float(target_rate) if target_rate is not None else float(np.mean(y))
            if is_clf:
                if not (0.0 < alvo < 1.0):
                    raise ValueError("Na classificação a tendência central alvo deve "
                                     "estar em (0, 1).")
                # δ por bisseção: média(σ(z+δ)) é crescente em δ ⇒ raiz única
                z = _logit_np(sc)
                lo, hi = -30.0, 30.0
                for _ in range(100):
                    mid = 0.5 * (lo + hi)
                    if float(np.mean(_sigmoid_np(z + mid))) < alvo:
                        lo = mid
                    else:
                        hi = mid
                params = {"delta": 0.5 * (lo + hi), "target": alvo}
            elif mode == "multiplicativo":
                media = float(np.mean(sc))
                if abs(media) < 1e-12:
                    raise ValueError("Previsto com média ~0: o ajuste multiplicativo "
                                     "é indefinido — use mode='aditivo'.")
                params = {"mode": "multiplicativo", "factor": alvo / media,
                          "target": alvo}
            else:
                params = {"mode": "aditivo", "shift": alvo - float(np.mean(sc)),
                          "target": alvo}
        elif method == "platt":
            if is_clf:
                if np.unique(y).size < 2:
                    raise ValueError(f"A amostra '{sample}' tem uma única classe — "
                                     "Platt requer as duas.")
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
                lr.fit(_logit_np(sc).reshape(-1, 1), y.astype(int))
                params = {"a": float(np.ravel(lr.coef_)[0]),
                          "b": float(np.ravel(lr.intercept_)[0])}
            else:
                if float(np.std(sc)) < 1e-12:
                    raise ValueError("Previsto (quase) constante — a recalibração "
                                     "linear é indefinida; use method='intercept'.")
                a, b = np.polyfit(sc, y, 1)
                params = {"a": float(a), "b": float(b)}
        else:                                       # isotonic
            from sklearn.isotonic import IsotonicRegression
            kw = {"y_min": 0.0, "y_max": 1.0} if is_clf else {}
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip", **kw)
            iso.fit(sc, y)
            # guarda só os degraus (listas JSON): a aplicação é np.interp — idem
            # sklearn — e a camada fica serializável sem o objeto ajustado
            params = {"x": [float(v) for v in np.ravel(iso.X_thresholds_)],
                      "y": [float(v) for v in np.ravel(iso.y_thresholds_)]}
        self.calibration_ = {
            "method": method, "sample": sample,
            "target_rate": float(target_rate) if target_rate is not None else None,
            "params": params,
            "ajustada_em": pd.Timestamp.now().isoformat(timespec="seconds"),
        }
        self._refresh_after_calibration()
        return self

    def decalibrate(self):
        """Remove a camada de calibração (:meth:`calibrate`) e volta ao score CRU
        do modelo — recalcula ``score_``, invalida caches dependentes e reprojeta
        a régua de ratings (mesmos cortes). No-op sem camada. Devolve ``self``."""
        if self.calibration_ is None:
            return self
        self.calibration_ = None
        self._refresh_after_calibration()
        return self

    def _refresh_after_calibration(self):
        """Após aplicar/remover a camada: recalcula ``score_`` e invalida o que
        depende dele — caches de métricas/IC e os ratings, cuja régua existente é
        **reprojetada** sobre o novo score (mesmos cortes; padrão do load e do
        retreino na UI). Se a reprojeção falhar, os ratings são limpos para não
        refletirem o score antigo."""
        # score calibrado = camada sobre o score CRU memoizado (mesmo resultado de
        # re-escorar a base com o modelo vigente, sem passar pelo pipeline de novo)
        raw = self._raw_score_df()
        vals = raw.to_numpy(dtype="float64")
        if self.calibration_ is not None:
            vals = self._apply_calibration(vals)
        self.score_ = pd.Series(np.asarray(vals, dtype="float64"), index=raw.index,
                                name="score", dtype="float64")
        self._metrics_cache = None
        self._metrics_ci_cache = None
        if self.rating_strategy is not None:
            try:
                self.rating_ = self.rating_strategy.transform(
                    self._rating_frame(), self._make_cfg("_amostra"))
            except Exception:
                self.rating_ = None
                self.rating_strategy = None
                self.rating_labels_ = []
                self.rating_config = {}

    def calibration_compare(self, sample=None) -> pd.DataFrame:
        """Efeito da camada vigente na amostra (default: a do ajuste): linhas
        ``sem calibração`` (score cru) × ``com calibração``, com a média prevista,
        a média observada do alvo e as métricas de calibração recomputadas —
        ``brier``/``logloss`` na classificação; ``mean_bias``/``rmse``/``mae`` na
        regressão. Requer :meth:`calibrate` aplicado."""
        if self.calibration_ is None:
            raise RuntimeError("Nenhuma camada de calibração vigente (use calibrate).")
        sample = sample or self.calibration_.get("sample") or self.ref_sample
        y, raw, cal = self._calibration_xy(sample)
        rows = []
        for nome, sc in (("sem calibração", raw), ("com calibração", cal)):
            if self.task_type == "classification":
                m = classification_metrics(y, sc)
                rows.append({"camada": nome, "media_score": float(np.mean(sc)),
                             "taxa_observada": float(np.mean(y)),
                             "brier": m.get("brier"), "logloss": m.get("logloss")})
            else:
                m = regression_metrics(y, sc)
                rows.append({"camada": nome, "media_prevista": float(np.mean(sc)),
                             "media_observada": float(np.mean(y)),
                             "mean_bias": m.get("mean_bias"), "rmse": m.get("rmse"),
                             "mae": m.get("mae")})
        out = pd.DataFrame(rows)
        out.attrs["sample"] = sample
        out.attrs["method"] = self.calibration_["method"]
        return out

    def plot_calibration_compare(self, sample=None, n_bins=10, figsize=(6.6, 4.8),
                                 dpi=150, save_path=None, ax=None):
        """Calibração **antes × depois** da camada (:meth:`calibrate`): curvas
        previsto×observado por faixa de previsto (quantis) do score CRU e do
        CALIBRADO na mesma figura, com a diagonal ideal. Eixos na unidade do
        alvo (não na escala de negócio 0–1000). Requer camada vigente."""
        if self.calibration_ is None:
            raise RuntimeError("Nenhuma camada de calibração vigente (use calibrate).")
        sample = sample or self.calibration_.get("sample") or self.ref_sample
        y, raw, cal = self._calibration_xy(sample)
        fig, ax = _new_ax(figsize, dpi, ax)
        if y.size == 0:
            ax.axis("off"); fig.tight_layout(); return fig

        def _curva(sc):
            q = np.unique(np.quantile(sc, np.linspace(0, 1, n_bins + 1)))
            if len(q) < 2:                          # previsto (quase) constante
                return np.asarray([float(np.mean(sc))]), np.asarray([float(np.mean(y))])
            idx = np.clip(np.searchsorted(q, sc, side="right") - 1, 0, len(q) - 2)
            px, py = [], []
            for g in range(len(q) - 1):
                m = idx == g
                if m.sum():
                    px.append(float(sc[m].mean())); py.append(float(y[m].mean()))
            return np.asarray(px), np.asarray(py)

        for nome, sc, cor, ls in (("sem calibração", raw, "#8aa4bf", "--"),
                                  ("com calibração", cal, "#15324a", "-")):
            px, py = _curva(sc)
            ax.plot(px, py, marker="o", ms=5, lw=1.8, ls=ls, color=cor, label=nome)
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
               max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, color="#bbb", ls=":", lw=1, zorder=0)
        ax.set_xlabel("previsto"); ax.set_ylabel("observado")
        _pct_axis(ax, "both")                       # unidade do alvo, em %
        met = {"intercept": "intercepto", "platt": "Platt",
               "isotonic": "isotônica"}.get(self.calibration_["method"],
                                            self.calibration_["method"])
        ax.set_title(f"Calibração antes × depois ({met}) · {sample}", fontsize=11,
                     fontweight="bold", color="#15324a")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def metrics(self) -> pd.DataFrame:
        """Métricas do modelo por amostra: classificação (auc, gini, ks, ks_cutoff,
        accuracy, f1, precision, recall, brier, logloss) ou regressão (rmse, mae,
        mape, smape, medae, r2, mean_bias)."""
        if self.score_ is None:
            raise RuntimeError("Ajuste o modelo antes (fit / set_model / load).")
        # cache por identidade do score_ (invalida em fit/set_model, que criam um
        # novo score_): _render_metrics + metric_shifts pediam metrics() 2× por clique.
        if self._metrics_cache is not None and self._metrics_cache[0] is self.score_:
            return self._metrics_cache[1].copy()
        rows = []
        for a in self._samples():
            mask = (pd.Series(True, index=self.df.index) if self.sample_col is None
                    else self._frame_mask(a))
            y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
            sc = self.score_[mask].to_numpy(dtype="float64")
            ok = ~np.isnan(y) & ~np.isnan(sc)
            y, sc = y[ok], sc[ok]
            if y.size == 0:
                continue
            m = (classification_metrics(y, sc) if self.task_type == "classification"
                 else regression_metrics(y, sc))
            rows.append({"amostra": a, "n": int(y.size), **m})
        out = pd.DataFrame(rows)
        # ks_cutoff é um LIMIAR na escala do score → apresenta na mesma escala de
        # negócio (0–1000). As demais métricas são invariantes à escala (rank) ou
        # calculadas sobre o score CRU (brier/logloss, RMSE/R²), então não mudam.
        if "ks_cutoff" in out.columns:
            out["ks_cutoff"] = out["ks_cutoff"] * self.score_scale
        self._metrics_cache = (self.score_, out)
        return out.copy()

    def metrics_ci(self, n_boot=200, metrics=None, alpha=0.05, seed=None) -> pd.DataFrame:
        """IC bootstrap das métricas de discriminação por amostra — reamostra o
        par ``(y, score_)`` JÁ computado (sem re-treino: barato). Na classificação
        a reamostragem é **estratificada por classe** (preserva a taxa de evento em
        cada réplica); ver :func:`yggdrasil.metrics.bootstrap_metric_ci`.

        Devolve um DataFrame métrica × amostra com ``valor`` (estimativa pontual),
        ``ic_low``/``ic_high`` (IC percentil de ``100·(1−alpha)%``) e ``se``
        (erro-padrão bootstrap). ``metrics`` default: ``('auc', 'ks', 'gini')`` na
        classificação e ``('r2', 'rmse')`` na regressão. ``seed=None`` herda a seed
        do segmenter (``random_state``) — mesma seed reproduz o mesmo IC. Cache por
        identidade do ``score_`` + parâmetros (invalida em fit/set_model, que criam
        um novo ``score_``)."""
        if self.score_ is None:
            raise RuntimeError("Ajuste o modelo antes (fit / set_model / load).")
        if metrics is None:
            metrics = (("auc", "ks", "gini") if self.task_type == "classification"
                       else ("r2", "rmse"))
        metrics = tuple(metrics)
        if seed is None:                          # herda a seed do segmenter
            seed = self.random_state
        key = (metrics, int(n_boot), float(alpha), seed)
        if self._metrics_ci_cache is None or self._metrics_ci_cache[0] is not self.score_:
            self._metrics_ci_cache = (self.score_, {})
        cached = self._metrics_ci_cache[1].get(key)
        if cached is not None:
            return cached.copy()

        def _rmse(yt, ys):                        # métrica sem nome nativo no bootstrap
            return float(np.sqrt(np.mean((yt - ys) ** 2)))

        rows = []
        for a in self._samples():
            mask = (pd.Series(True, index=self.df.index) if self.sample_col is None
                    else self._frame_mask(a))
            y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
            sc = self.score_[mask].to_numpy(dtype="float64")
            ok = ~np.isnan(y) & ~np.isnan(sc)
            y, sc = y[ok], sc[ok]
            if y.size == 0:
                continue
            # uma reamostragem por amostra para todas as métricas (mesma seed ⇒
            # mesmas réplicas que o cálculo métrica a métrica de antes)
            cis = bootstrap_metrics_ci(
                y, sc, metrics=[_rmse if m == "rmse" else m for m in metrics],
                n_boot=int(n_boot), alpha=alpha, seed=seed)
            for mname, ci in zip(metrics, cis):
                rows.append({"metrica": mname, "amostra": a, "n": int(y.size),
                             "valor": ci["valor"], "ic_low": ci["ic_low"],
                             "ic_high": ci["ic_high"], "se": ci["se"]})
        out = pd.DataFrame(rows, columns=["metrica", "amostra", "n", "valor",
                                          "ic_low", "ic_high", "se"])
        self._metrics_ci_cache[1][key] = out
        return out.copy()

    def metric_shifts(self, qualify=False, n_boot=200, alpha=0.05, seed=None) -> dict:
        """Variação de cada métrica DES→OOT (oot − des).

        Com ``qualify=True``, anexa ``{m}_significancia`` às métricas com IC
        bootstrap (:meth:`metrics_ci`), qualificando a queda como maior que o
        ruído amostral ou não: ``'degradacao_real'`` quando o IC do OOT cai
        **inteiro do lado ruim** do IC do DES (ICs disjuntos na direção ruim da
        métrica) e ``'dentro_do_ruido'`` quando os ICs se sobrepõem — ou o
        movimento é na direção boa. ICs não computáveis contam como sobreposição
        (veredicto conservador). ``n_boot``/``alpha``/``seed`` vão para o
        bootstrap (cacheado por identidade do ``score_``)."""
        m = self.metrics().set_index("amostra")
        oot = self._oot_sample()
        if self.ref_sample not in m.index or oot not in m.index or oot == self.ref_sample:
            return {}
        # 'ks_cutoff' é o limiar de score onde o KS é máximo (escala do score),
        # não uma métrica de desempenho — seu "shift" não é comparável aos demais.
        cols = [c for c in m.columns if c not in ("n", "ks_cutoff")]

        def _num(x):     # evita np.isfinite(None)/str → TypeError (não está em try)
            return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x)

        out = {c: round(float(m.loc[oot, c] - m.loc[self.ref_sample, c]), 6)
               for c in cols if _num(m.loc[oot, c]) and _num(m.loc[self.ref_sample, c])}
        if not qualify or not out:
            return out
        ci = self.metrics_ci(n_boot=n_boot, alpha=alpha, seed=seed)
        if len(ci) == 0:
            return out
        ci_idx = ci.set_index(["metrica", "amostra"])
        for c in list(out):
            sentido = _HIGHER_IS_BETTER.get(c)
            if sentido is None:                   # direção desconhecida/viés: não qualifica
                continue
            try:
                ref, cmp_ = ci_idx.loc[(c, self.ref_sample)], ci_idx.loc[(c, oot)]
            except KeyError:                      # métrica sem IC calculado
                continue
            lims = [float(ref["ic_low"]), float(ref["ic_high"]),
                    float(cmp_["ic_low"]), float(cmp_["ic_high"])]
            if not all(np.isfinite(v) for v in lims):
                out[f"{c}_significancia"] = "dentro_do_ruido"
                continue
            piora = (float(cmp_["ic_high"]) < float(ref["ic_low"]) if sentido
                     else float(cmp_["ic_low"]) > float(ref["ic_high"]))
            out[f"{c}_significancia"] = "degradacao_real" if piora else "dentro_do_ruido"
        return out

    def backward_elimination(self, sample=None, min_features=1, features=None, algorithm=None,
                             transform=None, hyperparams=None, n_repeats=3,
                             importance_sample_size=5000, random_state=None,
                             progress_callback=None) -> pd.DataFrame:
        """**Backward elimination** por importância: reajusta o modelo removendo, a
        cada passo, a variável **menos importante** e mede as métricas do modelo com
        o conjunto restante. Começa com ``model_features`` (ou as selecionadas),
        treina na referência (DES) e avalia na amostra ``sample`` (OOT quando
        existir, senão DES). A importância de cada variável vem de **permutation
        importance** (agnóstica ao algoritmo) sobre o modelo do passo.

        **Não altera o modelo vigente** (``self.model``/``score_``/rating): treina
        modelos temporários. Devolve um DataFrame — uma linha por passo, do conjunto
        cheio ao mínimo — com ``n_variaveis``, ``removida`` (a menos importante do
        passo, retirada no passo seguinte), ``importancia`` (dela) e TODAS as
        métricas na amostra de avaliação (classificação: auc, gini, ks, ks_cutoff,
        …; regressão: rmse, mae, mape, smape, medae, r2, mean_bias). ``features``
        força o conjunto inicial (default: model_features/selecionadas). Em ``.attrs``
        guarda ``eval_sample``, ``algorithm`` e ``feats0`` (o conjunto inicial).

        ``progress_callback(done, total, n_variaveis)`` (opcional) — barra de
        progresso na UI. ``n_repeats``/``importance_sample_size`` controlam o custo
        da permutação."""
        from sklearn.inspection import permutation_importance
        if random_state is None:                          # herda a seed do segmenter
            random_state = self.random_state
        if self.task_type == "classification":
            algorithm = algorithm or self.algorithm or "logistica"
        else:
            algorithm = algorithm or self.algorithm or "linear"
        transform = transform if transform is not None else (self.feature_transform or "raw")
        hyperparams = dict(hyperparams if hyperparams is not None else (self.hyperparams or {}))
        feats = (list(features) if features is not None
                 else list(self.model_features or self.selected_features() or self.candidates))
        if len(feats) < 2:
            raise ValueError("Backward elimination requer ao menos 2 variáveis no modelo.")
        min_features = max(1, int(min_features))
        is_clf = self.task_type == "classification"

        eval_sample = sample or self._oot_sample()
        cols_be = list(dict.fromkeys([*feats, self.target]))
        tr = self.df.loc[self._fit_mask(), cols_be]
        ev = self.df.loc[self._fit_mask(eval_sample), cols_be]
        if len(ev) < 20:                          # avaliação insuficiente → usa a própria DES
            ev, eval_sample = tr, self.ref_sample
        ytr = tr[self.target].astype(int) if is_clf else tr[self.target].astype("float64")
        yev = ev[self.target].astype(int) if is_clf else ev[self.target].astype("float64")
        scoring = "roc_auc" if is_clf else "r2"
        # subamostra para a permutação (limita o custo em bases grandes)
        if importance_sample_size and len(ev) > importance_sample_size:
            ev_imp = ev.sample(importance_sample_size, random_state=random_state)
        else:
            ev_imp = ev
        yev_imp = (ev_imp[self.target].astype(int) if is_clf
                   else ev_imp[self.target].astype("float64"))

        def _r(v):
            try:
                v = float(v)
            except Exception:
                return v
            return round(v, 6) if np.isfinite(v) else np.nan

        cur = list(feats)
        total = len(feats) - min_features + 1
        rows, done = [], 0
        while len(cur) >= min_features:
            pipe = self._build_pipeline(cur, algorithm, hyperparams, transform=transform)
            pipe.fit(tr[cur], ytr)
            s_ev = self._predict_score_array(pipe, ev[cur])
            met = (classification_metrics(yev.to_numpy(dtype="float64"), s_ev) if is_clf
                   else regression_metrics(yev.to_numpy(dtype="float64"), s_ev))
            removida, imp_val = "—", float("nan")
            if len(cur) > min_features:
                try:
                    pi = permutation_importance(pipe, ev_imp[cur], yev_imp, scoring=scoring,
                                                n_repeats=n_repeats, random_state=random_state)
                    j = int(np.argsort(pi.importances_mean)[0])     # menos importante
                    removida, imp_val = cur[j], float(pi.importances_mean[j])
                except Exception:
                    removida = cur[-1]                              # fallback determinístico
            rows.append({"n_variaveis": len(cur), "removida": removida,
                         "importancia": _r(imp_val), **{k: _r(v) for k, v in met.items()}})
            done += 1
            if progress_callback is not None:
                try:
                    progress_callback(done, total, len(cur))
                except Exception:
                    pass
            if len(cur) <= min_features:
                break
            cur = [f for f in cur if f != removida]
        out = pd.DataFrame(rows)
        out.attrs["eval_sample"] = eval_sample
        out.attrs["algorithm"] = algorithm
        out.attrs["transform"] = transform     # transform/hyperparams que geraram a ordem
        out.attrs["hyperparams"] = dict(hyperparams)   # de remoção e as métricas do backward
        out.attrs["feats0"] = list(feats)      # conjunto inicial (identidade p/ apply)
        return out

    def backward_optimal_step(self, result, criterion="parsimony", tol=0.01,
                              metric=None) -> dict:
        """Escolhe o passo **ótimo** de uma :meth:`backward_elimination` — **sem aplicar**.

        No DataFrame ``result`` devolvido por ela, seleciona o nº de variáveis
        recomendado e reconstrói o subconjunto correspondente (ancorado no
        ``attrs['feats0']``, por identidade). ``criterion``:

        * ``"parsimony"`` (default): o **menor** nº de variáveis cuja métrica fica
          dentro de ``tol`` (relativo, ``|melhor|·tol``) da melhor — parcimônia/cotovelo;
        * ``"best"``: o passo de **melhor** métrica.

        ``metric`` default: ``ks``→``auc``→``gini`` (classificação, maior é melhor)
        ou ``rmse`` (regressão, menor é melhor). Devolve ``{metric, criterion,
        target_n, best, features, removed}`` **sem tocar no modelo** — usado por
        :meth:`apply_backward_selection` (antes de reajustar) e pela UI, que destaca
        a linha ótima e habilita o botão de retreino."""
        cols = list(getattr(result, "columns", []))
        if result is None or len(result) == 0 or "n_variaveis" not in cols:
            raise ValueError("Resultado de backward_elimination vazio ou inválido.")
        is_clf = self.task_type == "classification"
        if metric is None:
            prefs = (["ks", "auc", "gini"] if is_clf else ["rmse", "mae", "smape"])
            metric = next((m for m in prefs if m in cols), None)
            if metric is None:
                raise ValueError(f"Nenhuma métrica conhecida em result: {cols}.")
        # direção pela MÉTRICA, não pelo task_type: erro/perda = menor melhor; o resto
        # (ks/auc/gini/r2/accuracy/f1/...) = maior melhor. Evita inverter com metric='r2'.
        _lower_better = {"rmse", "mae", "mape", "smape", "medae", "brier", "logloss"}
        higher_better = str(metric) not in _lower_better
        vals = pd.to_numeric(result[metric], errors="coerce")
        ns = result["n_variaveis"].astype(int)
        ok = vals.notna()
        if not ok.any():
            raise ValueError(f"Métrica '{metric}' sem valores válidos no resultado.")
        best = float(vals[ok].max() if higher_better else vals[ok].min())
        if criterion == "best":
            idx = vals[ok].idxmax() if higher_better else vals[ok].idxmin()
            target_n = int(ns.loc[idx])
        else:                                      # parcimônia / cotovelo
            thr = tol * abs(best)
            elig = ok & ((vals >= best - thr) if higher_better else (vals <= best + thr))
            target_n = int(ns[elig].min())         # menor nº de variáveis na tolerância
        # subconjunto do passo target_n — reconstruído por IDENTIDADE (ancorado em
        # attrs['feats0'], não no estado atual): ver _backward_subset.
        _, subset, removed = self._backward_subset(result, target_n)
        return {"metric": metric, "criterion": criterion, "target_n": target_n,
                "best": best, "features": subset, "removed": sorted(removed)}

    def _backward_subset(self, result, target_n):
        """Reconstrói ``(feats0, subset, removed)`` do passo com ``target_n`` variáveis
        de uma :meth:`backward_elimination`, ancorado em ``attrs['feats0']`` (identidade,
        não no estado atual). Base comum do passo ÓTIMO (:meth:`backward_optimal_step`)
        e da escolha MANUAL (:meth:`backward_subset_at`)."""
        ns = result["n_variaveis"].astype(int)
        feats0 = list(result.attrs.get("feats0")
                      or (self.model_features or self.selected_features() or self.candidates))
        n_full = int(ns.max())
        if len(feats0) != n_full:
            raise RuntimeError(
                f"A seleção/modelo vigente ({len(feats0)} variáveis) diverge do topo "
                f"do backward ({n_full}). Rode o backward elimination de novo antes "
                f"de aplicar (a seleção mudou desde então).")
        removed = set(result.loc[ns > int(target_n), "removida"]) - {"—"}
        subset = [f for f in feats0 if f not in removed]
        if len(subset) != int(target_n):
            raise RuntimeError(
                f"Reconstrução inconsistente do subconjunto: esperado {target_n} "
                f"variáveis, obtido {len(subset)}.")
        return feats0, subset, removed

    def backward_subset_at(self, result, n_variaveis, metric=None) -> dict:
        """Subconjunto de variáveis do passo com ``n_variaveis`` de uma
        :meth:`backward_elimination` — **escolha MANUAL** do nº de variáveis (em vez do
        ótimo). Reconstrói o subconjunto ancorado em ``attrs['feats0']`` (identidade),
        **sem tocar no modelo**. Devolve o mesmo formato de
        :meth:`backward_optimal_step` (``criterion='manual'``; ``best`` = a métrica no
        passo escolhido) — usado pela UI e por :meth:`apply_backward_selection`
        (via ``n_variaveis=...``)."""
        cols = list(getattr(result, "columns", []))
        if result is None or len(result) == 0 or "n_variaveis" not in cols:
            raise ValueError("Resultado de backward_elimination vazio ou inválido.")
        ns = result["n_variaveis"].astype(int)
        target_n = int(n_variaveis)
        if target_n not in set(ns.tolist()):
            raise ValueError(f"n_variaveis={target_n} não existe no backward "
                             f"(disponíveis: {sorted(set(ns.tolist()))}).")
        if metric is None:
            prefs = (["ks", "auc", "gini"] if self.task_type == "classification"
                     else ["rmse", "mae", "smape"])
            metric = next((m for m in prefs if m in cols), None)
        best = None
        if metric is not None and metric in cols:
            v = pd.to_numeric(result.loc[ns == target_n, metric], errors="coerce")
            if len(v) and pd.notna(v.iloc[0]):
                best = float(v.iloc[0])
        _, subset, removed = self._backward_subset(result, target_n)
        return {"metric": metric, "criterion": "manual", "target_n": target_n,
                "best": best, "features": subset, "removed": sorted(removed)}

    def apply_backward_selection(self, result, criterion="parsimony", tol=0.01,
                                 metric=None, refit=True, rebuild_ratings=False,
                                 n_variaveis=None):
        """Aplica à seleção vigente (``included``) o subconjunto de variáveis de um
        passo de uma :meth:`backward_elimination` já executada: o passo **ótimo**
        (default) ou, se ``n_variaveis`` for dado, o passo com esse **nº de variáveis**
        (escolha MANUAL do usuário — ignora ``criterion``/``tol``/``metric``).

        ``result`` é o DataFrame devolvido por ela. ``criterion``:

        * ``"parsimony"`` (default): o **menor** nº de variáveis cuja métrica fica
          dentro de ``tol`` (relativo, ``|melhor|·tol``) da melhor — parcimônia /
          cotovelo, alinhado à cultura de risco (menos variáveis, estável);
        * ``"best"``: o passo de **melhor** métrica, independentemente do tamanho.

        ``metric`` força a métrica de escolha (default: ``ks`` senão ``auc`` na
        classificação — maior é melhor; ``rmse`` na regressão — menor é melhor).
        Com ``refit`` (default) reajusta o modelo no subconjunto preservando
        algoritmo/hyperparams/transform vigentes; com ``rebuild_ratings`` regenera
        os ratings com a config atual — **réguas manuais são preservadas** (não são
        regeradas silenciosamente). Devolve um dict-resumo (não ``self``). A escolha
        do passo ótimo é delegada a :meth:`backward_optimal_step`."""
        if n_variaveis is not None:                 # escolha MANUAL do nº de variáveis
            pick = self.backward_subset_at(result, n_variaveis, metric=metric)
        else:                                       # passo ÓTIMO (parcimônia/best)
            pick = self.backward_optimal_step(result, criterion=criterion, tol=tol, metric=metric)
        target_n = pick["target_n"]
        subset = pick["features"]
        removed = set(pick["removed"])
        metric = pick["metric"]
        best = pick["best"]
        criterion = pick["criterion"]
        self.clear_features()
        for f in subset:
            if f in self.candidates:
                self.included.add(f)
        rebuilt = reprojected = False
        if refit:
            # reajusta com o algoritmo/transform/hyperparams que GERARAM o backward
            # (a ordem de remoção e as métricas vêm deles); fallback p/ os vigentes
            # em resultados antigos que não gravaram esses attrs.
            algo = result.attrs.get("algorithm") or self.algorithm
            xform = result.attrs.get("transform", self.feature_transform or "raw")
            hp = result.attrs.get("hyperparams", self.hyperparams)
            self.fit(algorithm=algo, hyperparams=hp, features=subset,
                     transform=xform or "raw")
            method = (self.rating_config or {}).get("method")
            if (rebuild_ratings and self.score_ is not None and method
                    and method not in ("manual_score", "manual_percentil")):
                cfg = self.rating_config
                self.build_ratings(method=method, n_ratings=cfg.get("n_ratings", 10),
                                   monotonic_fusion=cfg.get("monotonic_fusion", True),
                                   alpha=cfg.get("alpha", 0.05))
                rebuilt = True
            elif self.rating_strategy is not None and self.score_ is not None:
                # o fit trocou score_ mas a régua (ex.: cortes manuais) não foi
                # regenerada → REPROJETA a estratégia vigente sobre o novo score,
                # mantendo rating_ coerente sem re-ajustar os cortes do usuário.
                self.rating_ = self.rating_strategy.transform(
                    self._rating_frame(), self._make_cfg("_amostra"))
                reprojected = True
        return {"metric": metric, "criterion": criterion, "target_n": target_n,
                "best": best, "features": subset, "removed": sorted(removed),
                "refit": bool(refit), "ratings_rebuilt": rebuilt,
                "ratings_reprojected": reprojected}

    def plot_backward_elimination(self, result, metrics=None, figsize=(9.4, 4.6),
                                  dpi=150, save_path=None, ax=None, y_range=None):
        """Curva das **métricas vs nº de variáveis** da :meth:`backward_elimination`
        (o modelo encolhendo à medida que a variável menos importante sai). ``result``
        é o DataFrame devolvido por ela; ``metrics`` escolhe quais métricas plotar
        (default: KS+AUC na classificação, RMSE+MAE na regressão). A 1ª métrica vai
        no eixo esquerdo e as demais no direito (escalas diferentes).

        ``y_range=(min, max)`` fixa AMBOS os eixos y nesse intervalo — útil em
        regressão, onde as métricas não estão em 0–1. Sem ele, métricas de
        discriminação (0–1) usam eixos fixos em 0–1 e as demais, autoescala."""
        if metrics is None:
            metrics = (["ks", "auc"] if self.task_type == "classification"
                       else ["rmse", "mae"])
        metrics = [m for m in metrics if m in getattr(result, "columns", [])]
        fig, ax = _new_ax(figsize, dpi, ax)
        if result is None or len(result) == 0 or not metrics:
            ax.text(0.5, 0.5, "sem resultado de backward elimination", ha="center",
                    va="center", transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            return fig
        x = result["n_variaveis"].to_numpy()
        palette = ["#15324a", "#b23a2a", "#157a52", "#9a6f12", "#6b46c1"]
        # métricas limitadas a [0,1] (discriminação) → eixos FIXOS em 0–1 (constante
        # de módulo _UNIT_METRICS); as fora de [0,1] (regressão/logloss) autoescala.
        _UNIT = _UNIT_METRICS
        # range manual (y_range) tem prioridade sobre o 0–1 automático das métricas 0–1
        _yr = (tuple(y_range) if (y_range is not None and None not in y_range
                                  and float(y_range[1]) > float(y_range[0])) else None)
        ln = ax.plot(x, result[metrics[0]].to_numpy(dtype="float64"), color=palette[0],
                     lw=2.0, marker="o", ms=4, label=metrics[0].upper())
        ax.set_ylabel(metrics[0].upper(), color=palette[0])
        if _yr is not None:
            ax.set_ylim(*_yr)
        elif metrics[0] in _UNIT:
            ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("nº de variáveis no modelo")
        ax.invert_xaxis()                    # cheio (esq.) → mínimo (dir.)
        ax.grid(axis="both", alpha=0.12)
        handles = list(ln)
        if len(metrics) > 1:
            ax2 = ax.twinx()
            for i, mt in enumerate(metrics[1:], start=1):
                handles += ax2.plot(x, result[mt].to_numpy(dtype="float64"),
                                    color=palette[i % len(palette)], lw=1.8,
                                    marker="s", ms=3, label=mt.upper())
            ax2.set_ylabel(" · ".join(m.upper() for m in metrics[1:]))
            if _yr is not None:
                ax2.set_ylim(*_yr)
            elif all(m in _UNIT for m in metrics[1:]):
                ax2.set_ylim(0.0, 1.0)
        ax.legend(handles, [h.get_label() for h in handles], fontsize=8, loc="best",
                  framealpha=0.9)
        ax.set_title("Backward elimination — métricas × nº de variáveis", fontsize=11,
                     fontweight="bold", color="#15324a")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ---- plots do modelo ----
    def _sample_scores(self, sample=None):
        if sample is None:
            sample = self.ref_sample
        mask = (np.ones(len(self.df), dtype=bool) if self.sample_col is None
                else self._frame_mask(sample))
        y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
        sc = self.score_[mask].to_numpy(dtype="float64")
        ok = ~np.isnan(y) & ~np.isnan(sc)
        return y[ok], sc[ok]

    def plot_roc(self, sample=None, figsize=(5.4, 5.0), dpi=150, save_path=None, ax=None):
        if self.task_type != "classification":
            raise ValueError("plot_roc é exclusivo de classificação (eventos "
                             "binários); em regressão use plot_calibration/"
                             "plot_residuals.")
        from sklearn.metrics import roc_curve, roc_auc_score
        y, sc = self._sample_scores(sample)
        fig, ax = _new_ax(figsize, dpi, ax)
        if len(np.unique(y)) < 2:
            ax.text(0.5, 0.5, "amostra com 1 classe", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        fpr, tpr, _ = roc_curve(y, sc); auc = roc_auc_score(y, sc)
        ax.plot(fpr, tpr, color="#15324a", lw=2.2, label=f"AUC={auc:.3f} · Gini={2*auc-1:.3f}")
        ax.plot([0, 1], [0, 1], color="#bbb", ls="--", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        _pct_axis(ax, "both")
        ax.set_title(f"Curva ROC · {sample or self.ref_sample}", fontsize=11,
                     fontweight="bold", color="#15324a")
        ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_ks(self, sample=None, figsize=(6.4, 4.2), dpi=150, save_path=None, ax=None):
        if self.task_type != "classification":
            raise ValueError("plot_ks é exclusivo de classificação (KS entre as CDFs "
                             "de evento/não-evento); em regressão use plot_calibration/"
                             "plot_residuals.")
        y, sc = self._sample_scores(sample)
        sc = sc * self.score_scale                          # score na escala de negócio
        fig, ax = _new_ax(figsize, dpi, ax)
        if len(np.unique(y)) < 2:
            ax.text(0.5, 0.5, "amostra com 1 classe", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        grid = np.linspace(np.nanmin(sc), np.nanmax(sc), 200)
        pos, neg = np.sort(sc[y == 1]), np.sort(sc[y == 0])
        cdf_pos = np.searchsorted(pos, grid, side="right") / max(len(pos), 1)
        cdf_neg = np.searchsorted(neg, grid, side="right") / max(len(neg), 1)
        diff = np.abs(cdf_pos - cdf_neg); j = int(np.argmax(diff))
        ax.plot(grid, cdf_neg, color="#1aa64b", lw=2, label="não-evento (0)")
        ax.plot(grid, cdf_pos, color="#d6453e", lw=2, label="evento (1)")
        ax.vlines(grid[j], cdf_pos[j], cdf_neg[j], color="#15324a", lw=2,
                  label=f"KS={diff[j]:.3f}")
        ax.set_xlabel(f"score (0–{self.score_scale:.0f})"); ax.set_ylabel("CDF acumulada")
        # x é o score (0–score_scale, plain); a CDF (y) segue como antes
        ax.set_title(f"Curva KS · {sample or self.ref_sample}", fontsize=11,
                     fontweight="bold", color="#15324a")
        ax.legend(fontsize=9, loc="center right"); ax.grid(alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_cap(self, samples=None, figsize=(5.8, 5.2), dpi=150, save_path=None,
                 ax=None):
        """Curva CAP (Cumulative Accuracy Profile / Lorenz): % acumulado de
        **eventos** capturados × % acumulado da **carteira** ordenada do pior
        para o melhor score. Sobrepõe múltiplas amostras na mesma figura
        (default: referência + demais, como o ``plot_roc`` da família tree),
        com o **AR** (accuracy ratio) de cada amostra na legenda, a diagonal
        (modelo aleatório) e a curva do modelo perfeito (da referência).

        Somente classificação — o CAP é definido sobre eventos binários."""
        if self.task_type != "classification":
            raise ValueError("plot_cap é exclusivo de classificação (eventos "
                             "binários); em regressão use plot_calibration/"
                             "plot_residuals.")
        if self.score_ is None:
            raise RuntimeError("Ajuste o modelo antes (fit / set_model / load).")
        grupos = self._samples() if samples is None else [a for a in self._samples()
                                                          if a in samples]
        fig, ax = _new_ax(figsize, dpi, ax)
        ax.plot([0, 1], [0, 1], color="#bbb", ls="--", lw=1, label="aleatório")
        cores = ["#15324a", "#d6453e", "#1aa64b", "#caa000", "#6b3fa0", "#2a9d8f"]
        perfeito_feito = False
        alguma = False
        for i, a in enumerate(grupos):
            y, sc = self._sample_scores(a)
            if y.size == 0 or len(np.unique(y)) < 2:
                continue
            # carteira ordenada do PIOR para o MELHOR score (maior prob. de
            # evento primeiro) → % acumulado de eventos capturados
            order = np.argsort(-sc, kind="mergesort")
            y_ord = y[order]
            n = y_ord.size
            cum_port = np.arange(1, n + 1) / n
            cum_ev = np.cumsum(y_ord) / max(float(y_ord.sum()), 1.0)
            x_cap = np.concatenate(([0.0], cum_port))
            y_cap = np.concatenate(([0.0], cum_ev))
            # AR = (área do modelo − 0,5) / (área do perfeito − 0,5)
            tx_ev = float(y.mean())
            # Área sob a CAP pela regra do trapézio (manual: independe da versão
            # do numpy — trapz foi deprecado na 2.0 e trapezoid não existe <2.0).
            area_mod = float(np.sum(np.diff(x_cap) * (y_cap[:-1] + y_cap[1:]) / 2.0))
            area_perf = 1.0 - tx_ev / 2.0
            ar = ((area_mod - 0.5) / (area_perf - 0.5)
                  if area_perf > 0.5 else float("nan"))
            if not perfeito_feito:                       # perfeito da 1ª amostra útil
                ax.plot([0, tx_ev, 1], [0, 1, 1], color="#8891a0", ls=":", lw=1.4,
                        label="modelo perfeito")
                perfeito_feito = True
            ax.plot(x_cap, y_cap, color=cores[i % len(cores)], lw=2.0,
                    label=f"{a} · AR={ar:.3f}")
            alguma = True
        if not alguma:
            ax.text(0.5, 0.5, "sem as duas classes para a curva CAP", ha="center",
                    va="center", transform=ax.transAxes, color="#889")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("% acumulado da carteira (pior → melhor score)")
        ax.set_ylabel("% acumulado de eventos capturados")
        _pct_axis(ax, "both")
        ax.set_title("Curva CAP (Lorenz)", fontsize=11, fontweight="bold",
                     color="#15324a")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_lift(self, sample=None, n_bins=10, figsize=(7.2, 4.2), dpi=150,
                  save_path=None, ax=None):
        """Lift por decil de score (barras; decil 1 = piores scores) + linha de
        **gains** acumulado (% de eventos capturados até o decil) num segundo
        eixo. Linha de referência em lift = 1 (modelo aleatório).

        Somente classificação — lift/gains são definidos sobre eventos binários."""
        if self.task_type != "classification":
            raise ValueError("plot_lift é exclusivo de classificação (eventos "
                             "binários); em regressão use plot_calibration/"
                             "plot_residuals.")
        if self.score_ is None:
            raise RuntimeError("Ajuste o modelo antes (fit / set_model / load).")
        y, sc = self._sample_scores(sample)
        fig, ax = _new_ax(figsize, dpi, ax)
        if y.size == 0 or len(np.unique(y)) < 2:
            ax.text(0.5, 0.5, "amostra com 1 classe", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        # decil 1 = PIORES scores (maior prob. de evento) — ordena descendente e
        # fatia em n_bins grupos de tamanho ~igual (robusto a empates de score).
        order = np.argsort(-sc, kind="mergesort")
        y_ord = y[order]
        n = y_ord.size
        n_bins = max(2, min(int(n_bins), n))
        idx = np.minimum((np.arange(n) * n_bins) // n, n_bins - 1)
        tx_geral = float(y_ord.mean())
        tot_ev = max(float(y_ord.sum()), 1.0)
        lifts, gains = [], []
        acum = 0.0
        for g in range(n_bins):
            m = idx == g
            tx_g = float(y_ord[m].mean()) if m.any() else float("nan")
            lifts.append(tx_g / tx_geral if tx_geral > 0 else float("nan"))
            acum += float(y_ord[m].sum())
            gains.append(acum / tot_ev)
        x = np.arange(1, n_bins + 1)
        ax.bar(x, lifts, color="#3b6ea5", edgecolor="#2f5d82", alpha=0.9,
               width=0.72, label="lift do decil")
        ax.axhline(1.0, color="#d6453e", lw=1.2, ls="--", label="lift = 1 (aleatório)")
        for x0, lf in zip(x, lifts):
            if np.isfinite(lf):
                ax.text(x0, lf, f"{lf:.2f}", ha="center", va="bottom", fontsize=7.5,
                        color="#15324a")
        ax.set_xticks(list(x))
        ax.set_xlabel("decil de score (1 = piores scores)")
        ax.set_ylabel("lift (taxa do decil / taxa geral)")
        ax.set_ylim(0, max([l for l in lifts if np.isfinite(l)] + [1.0]) * 1.18)
        # gains acumulado no eixo secundário (% de eventos capturados)
        ax2 = ax.twinx()
        ax2.plot(x, gains, color="#15324a", lw=2.0, marker="o", ms=4.5,
                 label="gains acumulado")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("% de eventos capturados (acum.)")
        _pct_axis(ax2, "y")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right", framealpha=0.9)
        ax.set_title(f"Lift e gains por decil de score · {sample or self.ref_sample}",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_score_distribution(self, sample=None, bins=30, figsize=(6.6, 3.8),
                                dpi=150, save_path=None, ax=None):
        y, sc = self._sample_scores(sample)
        sc = sc * self.score_scale                          # score na escala de negócio
        fig, ax = _new_ax(figsize, dpi, ax)
        if self.task_type == "classification" and len(np.unique(y)) == 2:
            ax.hist(sc[y == 0], bins=bins, color="#1aa64b", alpha=0.55, label="não-evento (0)",
                    density=True, edgecolor="white", linewidth=0.3)
            ax.hist(sc[y == 1], bins=bins, color="#d6453e", alpha=0.55, label="evento (1)",
                    density=True, edgecolor="white", linewidth=0.3)
        else:
            ax.hist(sc, bins=bins, color="steelblue", alpha=0.85, edgecolor="#2f5d82")
        # linhas de referência (tracejadas, na legenda): quartis e média do score.
        # Percentis em PRETO (mediana em traço mais grosso) e média em CRIMSON —
        # alto contraste sobre o histograma (o azul/laranja anterior sumia).
        if sc.size:
            refs = (("p25", float(np.nanpercentile(sc, 25)), "#111111", 1.3),
                    ("mediana (p50)", float(np.nanpercentile(sc, 50)), "#111111", 2.1),
                    ("p75", float(np.nanpercentile(sc, 75)), "#111111", 1.3),
                    ("média", float(np.nanmean(sc)), "#dc143c", 1.9))
            for lab, v, col, lw in refs:
                ax.axvline(v, ls="--", lw=lw, color=col, alpha=0.95, label=f"{lab} = {v:.0f}")
        # eixo x na faixa CHEIA do score (0–score_scale) p/ ver a distribuição no
        # contexto geral, não só onde caem os dados; estende se houver valor fora.
        if sc.size:
            ax.set_xlim(min(0.0, float(np.nanmin(sc))),
                        max(float(self.score_scale), float(np.nanmax(sc))))
        else:
            ax.set_xlim(0.0, float(self.score_scale))
        ax.set_xlabel(f"score (0–{self.score_scale:.0f})"); ax.set_ylabel("densidade")
        ax.set_title(f"Distribuição do score · {sample or self.ref_sample}",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        if ax.get_legend_handles_labels()[0]:            # classes (clf) + quartis/média
            ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_calibration(self, sample=None, n_bins=10, figsize=(5.6, 5.2), dpi=150,
                         save_path=None, ax=None):
        """Classificação: previsto×observado por decil. Regressão: previsto×observado
        com uma **banda de 95%** em torno da curva de calibração (média observada ±
        1,96·desvio, por faixa de previsto) e a **cobertura** — % das observações
        (pontos) que caem dentro da banda.

        Eixos na **unidade do alvo** (valor previsto vs. observado), NÃO na escala
        de score 0–1000: a calibração é sobre o risco previsto casar com o realizado
        (mesma família de ``valor_previsto``/``backtest``). Distribuição e KS é que
        usam a escala de negócio (ranking). Na regressão, acima de 50 mil
        observações a nuvem exibe uma amostra uniforme e, em destaque, os
        pontos extremos; curva, banda e cobertura usam todas (ver
        :func:`_pontos_dispersao`)."""
        y, sc = self._sample_scores(sample)                 # previsto/observado CRUS (alvo)
        fig, ax = _new_ax(figsize, dpi, ax)
        if y.size == 0:
            ax.axis("off"); fig.tight_layout(); return fig
        if self.task_type == "classification":
            q = np.quantile(sc, np.linspace(0, 1, n_bins + 1))
            q = np.unique(q)
            if len(q) < 2:                     # score (quase) constante → sem faixas
                ax.text(0.5, 0.5, "score constante — sem calibração por faixa",
                        ha="center", va="center", transform=ax.transAxes, color="#889")
                ax.axis("off"); fig.tight_layout(); return fig
            idx = np.clip(np.searchsorted(q, sc, side="right") - 1, 0, len(q) - 2)
            pred, obs = [], []
            for g in range(len(q) - 1):
                m = idx == g
                if m.sum():
                    pred.append(sc[m].mean()); obs.append(y[m].mean())
            ax.plot(pred, obs, marker="o", color="#15324a", lw=2, ms=6)
        else:
            # nuvem bruta + curva de calibração por faixa de previsto com BANDA de
            # 95% (média observada ± 1,96·desvio) e a cobertura: % das observações
            # que caem dentro da banda.
            vis, ext = _pontos_dispersao(len(sc), self.random_state, sc, y)
            amostrada = _desenha_nuvem(ax, sc, y, vis, ext, len(sc), s=10, alpha=0.16,
                                       color="#9db8d2", zorder=1)
            q = np.unique(np.quantile(sc, np.linspace(0, 1, n_bins + 1)))
            cov_txt = None
            if len(q) >= 2:
                idx = np.clip(np.searchsorted(q, sc, side="right") - 1, 0, len(q) - 2)
                nb = len(q) - 1
                cx = np.full(nb, np.nan); cy = np.full(nb, np.nan)
                lo = np.full(nb, np.nan); hi = np.full(nb, np.nan)
                for g in range(nb):
                    m = idx == g
                    n = int(m.sum())
                    if n == 0:
                        continue
                    yg = y[m]
                    cx[g] = sc[m].mean(); cy[g] = yg.mean()
                    sd = yg.std(ddof=1) if n > 1 else 0.0
                    lo[g] = cy[g] - 1.96 * sd; hi[g] = cy[g] + 1.96 * sd
                ok = ~np.isnan(cx)
                if ok.any():
                    order = np.argsort(cx[ok])           # banda contígua por previsto
                    bx = cx[ok][order]
                    ax.fill_between(bx, lo[ok][order], hi[ok][order], color="#8aa4bf",
                                    alpha=0.28, zorder=2, label="IC 95%")
                    ax.plot(bx, cy[ok][order], color="#15324a", lw=1.8, marker="o",
                            ms=5, zorder=3)
                    inside = (y >= lo[idx]) & (y <= hi[idx])
                    cov_txt = f"{100 * inside.mean():.0f}% das observações dentro do IC 95%"
            if cov_txt:
                ax.text(0.03, 0.97, cov_txt, transform=ax.transAxes, ha="left",
                        va="top", fontsize=8.5, color="#15324a",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="#c9d4df", alpha=0.85))
            if amostrada:
                _legenda_nuvem(ax)
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, color="#bbb", ls="--", lw=1)
        ax.set_xlabel("previsto"); ax.set_ylabel("observado")
        _pct_axis(ax, "both")                               # unidade do alvo, em %
        ax.set_title(f"Calibração · {sample or self.ref_sample}", fontsize=11,
                     fontweight="bold", color="#15324a")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_residuals(self, sample=None, figsize=(6.6, 4.0), dpi=150, save_path=None, ax=None):
        """Regressão: resíduo (observado − previsto) vs. previsto, na **unidade do
        alvo** (alvo previsto), não na escala de score 0–1000. Acima de 50 mil
        observações a nuvem exibe uma amostra uniforme e, em destaque, os pontos
        extremos (ver :func:`_pontos_dispersao`)."""
        y, sc = self._sample_scores(sample)                 # previsto/observado CRUS (alvo)
        fig, ax = _new_ax(figsize, dpi, ax)
        res = y - sc
        vis, ext = _pontos_dispersao(len(sc), self.random_state, sc, res)
        amostrada = _desenha_nuvem(ax, sc, res, vis, ext, len(sc), s=10, alpha=0.35,
                                   color="#3b6ea5")
        ax.axhline(0, color="#d6453e", lw=1)
        if amostrada:
            _legenda_nuvem(ax)
        ax.set_xlabel("previsto"); ax.set_ylabel("resíduo (obs − prev)")
        _pct_axis(ax, "both")                               # unidade do alvo, em %
        ax.set_title(f"Resíduos · {sample or self.ref_sample}", fontsize=11,
                     fontweight="bold", color="#15324a")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_metric_shift(self, figsize=(7.4, 4.0), dpi=150, save_path=None, ax=None):
        """Shift das principais métricas da referência (DES) para o OOT, como
        **variação relativa (%)** em barras horizontais, colorida por melhora
        (verde) / piora (vermelho) conforme a direção de cada métrica. O rótulo
        traz também o Δ absoluto (OOT − DES). Usa :meth:`metric_shifts`."""
        fig, ax = _new_ax(figsize, dpi, ax)
        oot = self._oot_sample()
        shifts = self.metric_shifts()
        if not shifts or oot == self.ref_sample:
            ax.text(0.5, 0.5, "sem amostra OOT para comparar", ha="center",
                    va="center", transform=ax.transAxes, color="#8891a0")
            ax.axis("off"); fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            return fig
        names = {"auc": "AUC", "gini": "Gini", "ks": "KS", "f1": "F1", "r2": "R²",
                 "rmse": "RMSE", "mae": "MAE", "smape": "sMAPE"}
        order = (["auc", "gini", "ks", "f1"] if self.task_type == "classification"
                 else ["r2", "rmse", "mae", "smape"])
        better_up = {"auc", "gini", "ks", "f1", "r2"}      # maior = melhor
        m = self.metrics().set_index("amostra")
        labels, rels, deltas, cols = [], [], [], []
        for c in order:
            if c not in shifts:
                continue
            des = float(m.loc[self.ref_sample, c])
            if not np.isfinite(des) or abs(des) < 1e-9:
                continue                                    # % relativo instável
            delta = shifts[c]
            improve = (delta > 0) if c in better_up else (delta < 0)
            labels.append(names.get(c, c))
            rels.append(100.0 * delta / abs(des)); deltas.append(delta)
            cols.append("#1aa64b" if improve else "#d6453e")
        if not labels:
            ax.text(0.5, 0.5, "sem métricas comparáveis", ha="center", va="center",
                    transform=ax.transAxes, color="#8891a0")
            ax.axis("off"); fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            return fig
        yp = np.arange(len(labels))[::-1]                   # 1ª métrica no topo
        ax.barh(yp, rels, color=cols, edgecolor="#33424f", alpha=0.9, height=0.62)
        ax.axvline(0, color="#33424f", lw=1)
        span = max((abs(r) for r in rels), default=1.0) or 1.0
        txts = []
        for yi, r, d in zip(yp, rels, deltas):
            off = span * 0.02
            t = ax.text(r + (off if r >= 0 else -off), yi, f"{r:+.1f}%  (Δ{d:+.3f})",
                        ha="left" if r >= 0 else "right", va="center", fontsize=8,
                        color="#15324a")
            txts.append(t)
        ax.set_yticks(yp); ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(-span * 1.35, span * 1.35)
        ax.set_xlabel("variação DES → OOT (%)")
        _pct_axis(ax, "x", xmax=100)
        ax.set_title(f"Shift das principais métricas · DES → {oot}", fontsize=11,
                     fontweight="bold", color="#15324a")
        ax.grid(axis="x", alpha=0.15)
        # legenda melhora/piora FORA do eixo (não colide com os rótulos das barras)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#1aa64b", label="melhora"),
                           Patch(color="#d6453e", label="piora")],
                  fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=2, framealpha=0.85, columnspacing=1.4, handlelength=1.2)
        fig.tight_layout()
        # alarga o eixo-x p/ que os rótulos no fim das barras não saiam da caixa
        # (o texto tem largura fixa em px; com span pequeno ele estourava o xlim)
        _fit_labels_x(fig, ax, txts)
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_metric_comparison(self, figsize=(6.6, 4.8), dpi=150, save_path=None, ax=None):
        """Compara as principais métricas do modelo **entre amostras** (DES vs OOT
        lado a lado), em barras agrupadas — uma métrica por grupo, uma barra por
        amostra, com os valores **em %**. Sobre cada grupo, anota a **variação
        DES→OOT** (seta + %; verde = melhora, vermelho = piora). Classificação:
        **AUC, Gini, KS** (↑ maior = melhor). Regressão: **RMSE, MAE** (↓ menor =
        melhor) e **R²** (↑ maior = melhor) — com alvo em [0,1] (ex.: LGD) as três
        ficam no MESMO eixo em %, idêntico à classificação; só um alvo de magnitude
        grande leva o R² a um eixo à direita. Referência (DES) em steelblue e
        comparação (OOT) em crimson (safra de estabilidade em teal). Usa
        :meth:`metrics`."""
        from matplotlib.patches import Patch
        from matplotlib.ticker import PercentFormatter

        fig, ax = _new_ax(figsize, dpi, ax)
        try:
            m = self.metrics().set_index("amostra")
        except Exception:
            m = pd.DataFrame()
        samples = [a for a in self._samples() if a in m.index] if not m.empty else []

        def _val(x):
            return (float(x) if isinstance(x, (int, float, np.integer, np.floating))
                    and np.isfinite(x) else np.nan)

        # plano: (coluna, rótulo, maior_é_melhor?, no_eixo_direito?)
        if self.task_type == "classification":
            plano = [("auc", "AUC", True, False), ("gini", "Gini", True, False),
                     ("ks", "KS", True, False)]
            unit_scale = True
        else:
            # Eixo ÚNICO, idêntico à classificação, quando o alvo é unitário: em LGD/[0,1]
            # o RMSE/MAE ficam ~[0,1] e cabem no mesmo eixo do R² (tudo em %). Só quando o
            # alvo tem magnitude grande (erro ≫ 1) o R² vai a um eixo à direita — senão
            # sumiria esmagado pela escala do erro.
            _errs = [v for c in ("rmse", "mae") if c in m.columns
                     for v in (_val(m.loc[a, c]) for a in samples)]
            _errs = [v for v in _errs if np.isfinite(v)]
            unit_scale = (not _errs) or all(abs(v) <= 1.5 for v in _errs)
            plano = [("rmse", "RMSE", False, False), ("mae", "MAE", False, False),
                     ("r2", "R²", True, not unit_scale)]
        plano = [p for p in plano if p[0] in m.columns]
        if not plano or not samples:
            ax.text(0.5, 0.5, "sem métricas para comparar", ha="center", va="center",
                    transform=ax.transAxes, color="#8891a0", fontsize=12)
            ax.axis("off"); fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            return fig

        # cor por amostra: DES=steelblue, OOT=crimson, safra de estabilidade=teal.
        palette = ["steelblue", "crimson"]; stab_color = "#2a9d8f"; base_i = 0
        cores = {}
        for a in samples:
            if _is_stability_sample(a):
                cores[a] = stab_color
            else:
                cores[a] = palette[base_i % len(palette)]; base_i += 1

        w = 0.8 / max(len(samples), 1)
        ax2 = ax.twinx() if any(right for *_, right in plano) else None
        ref, oot = self.ref_sample, self._oot_sample()
        tem_oot = (ref in samples and oot in samples and oot != ref)

        left_vals, right_vals = [], []
        for gi, (col, _lab, up, right) in enumerate(plano):
            axis = ax2 if right else ax
            vals_g = {}
            for k, a in enumerate(samples):
                v = _val(m.loc[a, col]); xi = gi + k * w; vals_g[a] = v
                axis.bar(xi, v if np.isfinite(v) else 0.0, width=w, color=cores[a],
                         alpha=0.9, edgecolor="#33424f", linewidth=0.5)
                if np.isfinite(v):
                    (right_vals if right else left_vals).append(v)
                    lbl = f"{v * 100:.1f}%" if (unit_scale or right) else f"{v:,.3g}"
                    axis.text(xi, v, lbl, ha="center",
                              va="bottom" if v >= 0 else "top", fontsize=9.5,
                              color="#15324a", fontweight="bold")
            # variação DES→OOT sobre o grupo (topo do eixo; verde melhora / vermelho piora)
            dref, doot = vals_g.get(ref, np.nan), vals_g.get(oot, np.nan)
            if tem_oot and np.isfinite(dref) and np.isfinite(doot) and abs(dref) > 1e-9:
                delta = doot - dref
                rel = 100.0 * delta / abs(dref)
                piora = (delta < 0) if up else (delta > 0)
                seta = "▼" if delta < 0 else "▲"
                cor = "#d6453e" if piora else "#1aa64b"
                xc = gi + (len(samples) - 1) * w / 2
                axis.text(xc, 0.965, f"{seta} {abs(rel):.1f}%",
                          transform=axis.get_xaxis_transform(), ha="center", va="top",
                          fontsize=10.5, fontweight="bold", color=cor)

        # folga no topo p/ os rótulos de valor e a variação não colidirem com as barras
        lmax = max(left_vals + [0.0]); lmin = min(left_vals + [0.0])
        ax.set_ylim(lmin * 1.12 if lmin < 0 else 0.0, lmax * 1.30 if lmax > 0 else 1.0)
        # eixo em %: 1 casa quando os valores são pequenos (erro de regressão ~0.01),
        # senão inteiro (AUC/Gini/KS/R²). Alvo não-unitário (magnitude grande) usa
        # escala numérica simples no eixo esquerdo.
        if unit_scale:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1 if lmax < 0.1 else 0))
        else:
            from matplotlib.ticker import ScalarFormatter
            ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(axis="y", labelsize=10)
        if ax2 is not None:
            rmax = max(right_vals + [0.0]); rmin = min(right_vals + [0.0])
            ax2.set_ylim(rmin * 1.12 if rmin < 0 else 0.0, max(rmax * 1.30, 1.0))
            ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])   # R² ≤ 100%: não rotula acima
            ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax2.tick_params(axis="y", labelsize=10)
            ax2.set_ylabel("R² (eixo direito)", fontsize=11, color="#15324a")

        labels = [f"{lab} {'↑' if up else '↓'}" for _c, lab, up, _r in plano]
        ax.set_xticks(np.arange(len(plano)) + (len(samples) - 1) * w / 2)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel("erro (RMSE · MAE)" if (self.task_type == "regression"
                      and not unit_scale) else "métrica (%)", fontsize=11)
        ax.set_title("Principais métricas por amostra", fontsize=13,
                     fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        # legenda por amostra ABAIXO do gráfico (não colide com a variação no topo nem
        # com as barras, e serve aos dois eixos). Reserva a faixa inferior no layout.
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.legend(handles=[Patch(color=cores[a], label=a) for a in samples],
                   loc="lower center", ncol=min(len(samples), 3), fontsize=10,
                   framealpha=0.85, bbox_to_anchor=(0.5, 0.0))
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ---- discriminação por safra ----
    def metrics_by_safra(self, sample=None, time_col=None) -> pd.DataFrame:
        """Métricas do modelo **por safra** (mês de ``date_col``/``time_col``).

        Classificação: ``safra, n, taxa_evento, auc, ks, gini``. Regressão:
        ``safra, n, previsto_medio, realizado_medio, mae, rmse, r2``. Reutiliza
        :func:`~yggdrasil.metrics.classification_metrics` /
        :func:`~yggdrasil.metrics.regression_metrics` (mesmo pacote de
        :meth:`metrics`). Safras com poucas linhas ou classe única não quebram —
        as métricas ficam NaN.

        ``sample=None`` usa toda a base (as safras normalmente já separam
        DES/OOT); informe uma amostra para restringir."""
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col.")
        if self.score_ is None:
            raise RuntimeError("Ajuste o modelo antes (fit / set_model / load).")
        if time_col not in self.df.columns:
            raise ValueError(f"Coluna de tempo '{time_col}' não existe no DataFrame.")
        is_clf = self.task_type == "classification"
        met_cols = (["taxa_evento", "auc", "ks", "gini"] if is_clf
                    else ["previsto_medio", "realizado_medio", "mae", "rmse", "r2"])
        # fatias por safra sobre ARRAYS (uma ordenação), sem recortar o DataFrame
        # inteiro safra a safra (o groupby da base copiava todas as colunas)
        idx, limites, rot = self._fatias_por_safra(time_col, sample, all_rows=not sample)
        y_all = pd.to_numeric(self.df[self.target], errors="coerce").to_numpy(dtype="float64")
        sc_all = self.score_.reindex(self.df.index).to_numpy(dtype="float64")
        rows = []
        for j, per in enumerate(rot):
            ii = idx[limites[j]:limites[j + 1]]
            if ii.size == 0:                          # safra ausente neste recorte
                continue
            y, sc = y_all[ii], sc_all[ii]
            ok = ~np.isnan(y) & ~np.isnan(sc)
            y, sc = y[ok], sc[ok]
            row = {"safra": per, "n": int(y.size)}
            row.update({c: float("nan") for c in met_cols})
            if is_clf:
                row["taxa_evento"] = self._risco(y)
                if y.size >= 2 and len(np.unique(y)) == 2:
                    try:
                        m = classification_metrics(y, sc)
                        row.update({c: m.get(c, float("nan"))
                                    for c in ("auc", "ks", "gini")})
                    except Exception:  # noqa: BLE001 - safra degenerada ⇒ NaN
                        pass
            else:
                row["previsto_medio"] = float(np.mean(sc)) if sc.size else float("nan")
                row["realizado_medio"] = self._risco(y)
                if y.size >= 2:
                    try:
                        m = regression_metrics(y, sc)
                        row.update({c: m.get(c, float("nan"))
                                    for c in ("mae", "rmse", "r2")})
                    except Exception:  # noqa: BLE001 - safra degenerada ⇒ NaN
                        pass
            rows.append(row)
        return (pd.DataFrame(rows, columns=["safra", "n"] + met_cols)
                .sort_values("safra").reset_index(drop=True))

    def plot_metrics_by_safra(self, sample=None, metrics=("ks", "auc"),
                              time_col=None, figsize=(9.6, 4.2), dpi=150,
                              save_path=None, ax=None, ylim=None):
        """Evolução das métricas do modelo por safra (linhas), a partir de
        :meth:`metrics_by_safra`. ``metrics`` que não existirem para o
        ``task_type`` são ignoradas (default de regressão: ``mae``/``rmse``).
        ``ylim=(lo, hi)`` fixa o eixo vertical (ex.: ``(0, 1)`` para padronizar a
        leitura quando o alvo está em [0,1]); tem precedência sobre a heurística
        que fixa [0,1] para métricas de discriminação."""
        ms = self.metrics_by_safra(sample, time_col)
        cols = [m for m in metrics if m in ms.columns and m not in ("safra", "n")]
        if not cols:                        # default de clf pedido em regressão
            cols = [c for c in ("mae", "rmse") if c in ms.columns]
        fig, ax = _new_ax(figsize, dpi, ax)
        if ms.empty or not cols:
            ax.text(0.5, 0.5, "sem métricas por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        x = list(range(len(ms)))
        cores = ["#15324a", "#d6453e", "#1aa64b", "#caa000", "#6b3fa0", "#2a9d8f"]
        for i, c in enumerate(cols):
            ax.plot(x, ms[c], marker="o", lw=2.0, ms=4.5, color=cores[i % len(cores)],
                    markeredgecolor="#33424f", markeredgewidth=0.5, label=c.upper())
        # eixo Y: override explícito (ylim) tem precedência; senão, métricas de
        # discriminação (KS/AUC/Gini/…) ⇒ 0–1 comparável; regressão autoescala.
        # Todas as plotadas partilham o MESMO eixo.
        if ylim is not None:
            ax.set_ylim(*ylim)
        elif all(c in _UNIT_METRICS for c in cols):
            ax.set_ylim(0.0, 1.0)
        # rótulos mmm/aa; com muitas safras, afina os ticks p/ não sobrepor
        labels = _fmt_safras(ms["safra"])
        step = max(1, len(ms) // 18)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("safra"); ax.set_ylabel("métrica")
        ax.set_title(f"Métricas por safra · {sample or 'todas as amostras'}",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ---- discriminação / ratings por segmento arbitrário (coluna de contexto) ----
    def _check_group_col(self, col) -> None:
        """Valida a coluna de quebra POR GRUPO (:meth:`metrics_by_group` /
        :meth:`rating_distribution_by_group`): precisa existir no df e ser uma
        coluna de **contexto** — o alvo, as variáveis candidatas do modelo e
        colunas de data (use :meth:`metrics_by_safra`) não valem."""
        if col not in self.df.columns:
            raise ValueError(f"Coluna '{col}' não existe no DataFrame.")
        if col == self.target:
            raise ValueError("A coluna de grupo não pode ser o próprio alvo.")
        if col in set(self.candidates):
            raise ValueError(
                f"'{col}' é variável candidata do modelo — a quebra por grupo é "
                f"para colunas de CONTEXTO (não-features), ex.: produto/canal/região.")
        if pd.api.types.is_datetime64_any_dtype(self.df[col]):
            raise ValueError(f"'{col}' é uma coluna de data — para quebras "
                             f"temporais use metrics_by_safra.")

    def metrics_by_group(self, col, sample=None, min_n=30) -> pd.DataFrame:
        """Métricas do modelo **por grupo** de uma coluna categórica de contexto
        do df (que **não** é variável do modelo — ex.: produto, canal, região):
        a mesma leitura de :meth:`metrics_by_safra`, trocando a safra pelo grupo.

        Classificação: ``grupo, n, taxa_evento, auc, ks, gini``. Regressão:
        ``grupo, n, previsto_medio, realizado_medio, mae, rmse, r2``. Grupos com
        menos de ``min_n`` linhas válidas **não** têm as métricas de
        discriminação/erro reportadas (ficam NaN) — a coluna ``nota`` explica;
        taxas/médias descritivas continuam reportadas. Grupos de classe única
        também ganham nota (AUC/KS/Gini indefinidos).

        ``sample=None`` usa toda a base; informe uma amostra (ex.: OOT) para
        restringir. Saída ordenada por ``n`` decrescente."""
        if self.score_ is None:
            raise RuntimeError("Ajuste o modelo antes (fit / set_model / load).")
        self._check_group_col(col)
        base = self._frame(sample) if sample else self.df
        is_clf = self.task_type == "classification"
        sc_full = self.score_.reindex(base.index)
        met_cols = (["taxa_evento", "auc", "ks", "gini"] if is_clf
                    else ["previsto_medio", "realizado_medio", "mae", "rmse", "r2"])
        min_n = max(int(min_n), 2)
        rows = []
        for gname, g in base.groupby(base[col], observed=True):  # dropna: NaN fora
            y = g[self.target].to_numpy(dtype="float64")
            sc = sc_full.reindex(g.index).to_numpy(dtype="float64")
            ok = ~np.isnan(y) & ~np.isnan(sc)
            y, sc = y[ok], sc[ok]
            row = {"grupo": str(gname), "n": int(y.size), "nota": ""}
            row.update({c: float("nan") for c in met_cols})
            if is_clf:
                row["taxa_evento"] = self._risco(y)
            else:
                row["previsto_medio"] = float(np.mean(sc)) if sc.size else float("nan")
                row["realizado_medio"] = self._risco(y)
            if y.size < min_n:
                row["nota"] = f"n < {min_n} — métricas não reportadas"
            elif is_clf:
                if len(np.unique(y)) == 2:
                    try:
                        m = classification_metrics(y, sc)
                        row.update({c: m.get(c, float("nan"))
                                    for c in ("auc", "ks", "gini")})
                    except Exception:  # noqa: BLE001 - grupo degenerado ⇒ NaN
                        row["nota"] = "grupo degenerado — métricas indisponíveis"
                else:
                    row["nota"] = "classe única no grupo — AUC/KS/Gini indefinidos"
            else:
                try:
                    m = regression_metrics(y, sc)
                    row.update({c: m.get(c, float("nan"))
                                for c in ("mae", "rmse", "r2")})
                except Exception:  # noqa: BLE001 - grupo degenerado ⇒ NaN
                    row["nota"] = "grupo degenerado — métricas indisponíveis"
            rows.append(row)
        return (pd.DataFrame(rows, columns=["grupo", "n"] + met_cols + ["nota"])
                .sort_values("n", ascending=False, kind="stable")
                .reset_index(drop=True))

    def rating_distribution_by_group(self, col, sample=None) -> pd.DataFrame:
        """Distribuição (%) dos ratings **por grupo** de uma coluna categórica de
        contexto (não-feature): uma linha por grupo — ``n`` e o % do grupo em cada
        rating, na ordem da régua (colunas ``%_<rating>``). Complementa
        :meth:`plot_rating_distribution` (por amostra) para segmentações
        arbitrárias (produto/canal/região). Requer os ratings construídos
        (:meth:`build_ratings`); ``sample=None`` usa toda a base. Saída ordenada
        por ``n`` decrescente."""
        rating = self._rating_series()
        self._check_group_col(col)
        base = self._frame(sample) if sample else self.df
        r = rating.reindex(base.index)
        labels = list(self.rating_labels_)
        rows = []
        for gname, g in base.groupby(base[col], observed=True):
            rr = r.reindex(g.index)
            n = int(len(g))
            row = {"grupo": str(gname), "n": n}
            for lab in labels:
                row[f"%_{lab}"] = round(100.0 * int((rr == lab).sum()) / max(n, 1), 1)
            rows.append(row)
        return (pd.DataFrame(rows, columns=["grupo", "n"] + [f"%_{l}" for l in labels])
                .sort_values("n", ascending=False, kind="stable")
                .reset_index(drop=True))

    def variables_profile_by_safra(self, time_col=None, features=None,
                                    all_samples=True) -> pd.DataFrame:
        """Perfil das variáveis do modelo **por safra (dbase)**: para cada safra e
        cada variável, ``% missing``, ``média`` (numéricas) e ``moda`` (valor mais
        frequente). Tabela LONGA — uma linha por (safra, variável), ordenada por
        safra e pela ordem das variáveis no modelo —, para acompanhar a estabilidade
        das variáveis ao longo do tempo numa única tabela. ``features`` default:
        ``model_features`` (senão selecionadas/candidatas); ``all_samples=True`` usa
        toda a população (todas as amostras/safras)."""
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col.")
        feats = (list(features) if features is not None
                 else list(self.model_features or self.selected_features() or self.candidates))
        feats = [f for f in feats if f in self.df.columns]
        if not feats:
            raise ValueError("Nenhuma variável do modelo disponível (treine ou selecione).")
        sub = self.df if all_samples else self._frame(None)
        if time_col not in sub.columns:
            raise ValueError(f"Coluna de tempo '{time_col}' não existe no DataFrame.")
        safra = pd.to_datetime(sub[time_col], errors="coerce").dt.to_period("M").astype(str)
        rows = []
        for f in feats:
            col = sub[f]
            is_num = self._detect_kind(f, sub) == "num"
            for per, s in col.groupby(safra):
                if per == "NaT":
                    continue
                n = int(len(s)); m = s.dropna()
                moda = None
                if not m.empty:
                    moda = m.value_counts().index[0]
                    if is_num:
                        moda = round(float(moda), 4)
                rows.append({
                    "safra": per, "variável": self.label(f),
                    "% missing": round(100 * (n - len(m)) / n, 1) if n else float("nan"),
                    "média": (round(float(m.mean()), 4) if (is_num and not m.empty)
                              else float("nan")),
                    "moda": moda,
                })
        out = pd.DataFrame(rows, columns=["safra", "variável", "% missing", "média", "moda"])
        ordem = {self.label(f): i for i, f in enumerate(feats)}
        return (out.assign(_o=out["variável"].map(ordem))
                   .sort_values(["safra", "_o"]).drop(columns="_o").reset_index(drop=True))

    def _profile_feats(self, features=None) -> list:
        """Variáveis do modelo p/ o perfil por safra: as que ENTRARAM no modelo
        (senão as selecionadas/candidatas), restritas às colunas presentes."""
        feats = (list(features) if features is not None
                 else list(self.model_features or self.selected_features() or self.candidates))
        return [f for f in feats if f in self.df.columns]

    @staticmethod
    def _profile_grid(n, ncols, dpi):
        """Cria uma grade (fig, axes 2D, nrows, ncols) com ``ncols`` colunas para
        ``n`` subplots (um por variável)."""
        import matplotlib.pyplot as plt
        ncols = max(1, min(int(ncols), n))
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 2.55 * nrows),
                                 dpi=dpi, squeeze=False)
        return fig, axes, nrows, ncols

    @staticmethod
    def _short(txt, n=28):
        txt = str(txt)
        return txt if len(txt) <= n else txt[:n - 1] + "…"

    def _amostra_dominante_por_safra(self, time_col) -> dict:
        """``{safra 'AAAA-MM': amostra mais frequente}`` — a moda de ``sample_col``
        por safra (empate → o menor valor, como ``Series.mode``). Memoizado por
        coluna: não depende da variável, e era recalculado a cada subplot com um
        ``to_period().astype(str)`` + moda Python por safra em milhões de linhas."""
        cache = self.__dict__.setdefault("_amostra_safra_cache", {})
        if time_col in cache:
            return cache[time_col]
        cod, rot = self._safra_codes(time_col)
        s_codes, s_uniq = pd.factorize(self.df[self.sample_col], use_na_sentinel=True)
        ok = (cod >= 0) & (s_codes >= 0)
        ns = len(s_uniq)
        cont = np.bincount(cod[ok].astype(np.int64) * ns + s_codes[ok],
                           minlength=len(rot) * ns).reshape(len(rot), ns)
        out = {}
        for k, per in enumerate(rot):
            linha = cont[k]
            if not linha.any():
                continue
            mx = linha.max()
            out[per] = sorted(s_uniq[j] for j in np.flatnonzero(linha == mx))[0]
        cache[time_col] = out
        return out

    def _sample_boundaries(self, safras, time_col=None):
        """Índices no eixo X (= ``range(len(safras))``) onde a AMOSTRA dominante
        muda entre safras consecutivas. ``safras`` é a sequência de safras (Period
        ou str 'YYYY-MM') na MESMA ordem do subplot. Retorna ``[]`` se não houver
        ``sample_col`` (usado p/ marcar a troca de amostra com linha pontilhada)."""
        if self.sample_col is None:
            return []
        time_col = time_col or self.date_col
        if time_col is None or time_col not in self.df.columns:
            return []
        samp_by = self._amostra_dominante_por_safra(time_col)
        seq = [samp_by.get(str(p)) for p in safras]
        return [i for i in range(1, len(seq))
                if seq[i] is not None and seq[i - 1] is not None and seq[i] != seq[i - 1]]

    def plot_variables_missing_by_safra(self, time_col=None, features=None, ncols=3,
                                        dpi=150, save_path=None):
        """Grade (``ncols`` colunas) com a **% de missing por safra** de CADA variável
        do modelo — numéricas e categóricas — sobre toda a população. Um subplot por
        variável: leitura rápida de buracos/coleta instável ao longo do tempo. Só as
        variáveis que entraram no modelo. Requer ``date_col``/``time_col``."""
        import matplotlib.pyplot as plt
        time_col = time_col or self.date_col
        if time_col is None or time_col not in self.df.columns:
            raise ValueError("Informe time_col ou configure date_col.")
        feats = self._profile_feats(features)
        if not feats:
            raise ValueError("Nenhuma variável do modelo disponível (treine ou selecione).")
        cod, pers = self._safra_codes(time_col)              # memoizado
        xs = _fmt_safras(list(pers)); x = list(range(len(pers)))
        n_p = np.bincount(cod[cod >= 0], minlength=len(pers))
        fig, axes, nrows, ncols = self._profile_grid(len(feats), ncols, dpi)
        for idx, f in enumerate(feats):
            ax = axes[idx // ncols][idx % ncols]
            # % de missing por safra numa contagem (sem máscara por safra)
            na = self.df[f].isna().to_numpy(dtype=bool, na_value=True) & (cod >= 0)
            miss = np.bincount(cod[na], minlength=len(pers))
            ys = [100.0 * int(miss[k]) / int(n_p[k]) if n_p[k] else np.nan
                  for k in range(len(pers))]
            ax.fill_between(x, 0, ys, color="#c0392b", alpha=0.12)
            ax.plot(x, ys, marker="o", lw=1.7, ms=4, color="#c0392b",
                    markeredgecolor="#33424f", markeredgewidth=0.4)
            ax.set_title(self._short(self.label(f)), fontsize=9.5, fontweight="bold",
                         color="#15324a")
            ax.set_ylabel("% missing", fontsize=8)
            ax.set_ylim(0, 100)
            ax.set_xticks(x); ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=7)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", alpha=0.15)
        for j in range(len(feats), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle("% de missing por safra — variáveis do modelo",
                     fontsize=12, fontweight="bold", color="#15324a")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        return fig

    def plot_variables_stats_by_safra(self, time_col=None, features=None, ncols=3,
                                      top_cat=6, dpi=150, save_path=None):
        """Grade (``ncols`` colunas) da **dispersão por safra** de cada variável do
        modelo: **numéricas** mostram **p5 · média · p95** (banda p5–p95 + linha da
        média); **categóricas** mostram a **proporção das categorias** ao longo do
        tempo num gráfico de linhas preenchido (área empilhada). Só as variáveis que
        entraram no modelo. Requer ``date_col``/``time_col``."""
        import matplotlib.pyplot as plt
        time_col = time_col or self.date_col
        if time_col is None or time_col not in self.df.columns:
            raise ValueError("Informe time_col ou configure date_col.")
        feats = self._profile_feats(features)
        if not feats:
            raise ValueError("Nenhuma variável do modelo disponível (treine ou selecione).")
        fig, axes, nrows, ncols = self._profile_grid(len(feats), ncols, dpi)
        num_legend_done = False              # legenda p5/média: no 1º subplot NUMÉRICO
        for idx, f in enumerate(feats):
            ax = axes[idx // ncols][idx % ncols]
            if self._detect_kind(f, self.df) == "num":
                t = self.variable_by_safra(f, time_col=time_col, all_samples=True)
                safras_sub = list(t["safra"])
                xs = _fmt_safras(safras_sub); x = list(range(len(t)))
                if len(t):
                    ax.fill_between(x, t["p5"], t["p95"], color="#4c78a8", alpha=0.16,
                                    label="p5–p95")
                    ax.plot(x, t["p5"], color="#4c78a8", lw=1.0, ls="--")
                    ax.plot(x, t["p95"], color="#4c78a8", lw=1.0, ls="--")
                    ax.plot(x, t["media"], color="#c0392b", lw=1.8, marker="o", ms=3.5,
                            label="média")
                    if not num_legend_done:              # legenda p5/média no 1º subplot numérico
                        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.6)
                        num_legend_done = True
            else:
                sh = self.variable_share_by_safra(f, time_col=time_col, top=top_cat,
                                                  all_samples=True)
                cats = [c for c in sh.columns if c != "safra"]
                safras_sub = list(sh["safra"])
                xs = _fmt_safras(safras_sub); x = list(range(len(sh)))
                if cats:
                    cmap = _cmap("tab10")
                    colors = [cmap((i % 10) / 9) for i in range(len(cats))]
                    ax.stackplot(x, *[sh[c].to_numpy() for c in cats], labels=cats,
                                 colors=colors, alpha=0.85)
                    ax.set_ylim(0, 100)
                    ax.set_ylabel("% categoria", fontsize=8)
                    # legenda em CADA subplot categórico: as categorias mudam por
                    # variável, então cada gráfico precisa identificar as suas.
                    ax.legend(fontsize=6, loc="upper left", ncol=2, framealpha=0.85)
            # faixas verticais pontilhadas onde a AMOSTRA muda ao longo das safras
            for bx in self._sample_boundaries(safras_sub, time_col):
                ax.axvline(bx - 0.5, ls=":", lw=1.0, color="#33424f", alpha=0.7)
            ax.set_title(self._short(self.label(f)), fontsize=9.5, fontweight="bold",
                         color="#15324a")
            ax.set_xticks(x); ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=7)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", alpha=0.12)
        for j in range(len(feats), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle("Dispersão por safra — p5 · média · p95 (num.) · proporção (cat.)",
                     fontsize=12, fontweight="bold", color="#15324a")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        return fig

    # ---- SHAP ----
    def _shap_transform(self, X) -> tuple:
        """(estimador_final, X_transformado_df) — aplica o pré-processador do
        pipeline (quando houver) a linhas CRUAS de ``X`` e devolve o estimador
        final, para alimentar o SHAP. Compartilhado por :meth:`_shap_inputs`
        (amostra inteira) e pelos caminhos por linha/lote (reason codes)."""
        est, pre = self.model, None
        try:
            if hasattr(self.model, "named_steps") and "est" in self.model.named_steps:
                pre = self.model[:-1]
                est = self.model.named_steps["est"]
        except Exception:
            pre = None
        if pre is not None:
            Xt = pre.transform(X)
            try:
                names = list(pre.get_feature_names_out())
            except Exception:
                names = [f"f{i}" for i in range(np.asarray(Xt).shape[1])]
            Xt = pd.DataFrame(np.asarray(Xt), columns=names, index=X.index)
        else:
            Xt = X
        return est, Xt

    def _shap_inputs(self, sample=None, sample_size=2000):
        """(estimador, X_transformado_df, nomes) — para SHAP. Em pipelines,
        transforma com o pré-processador e usa o estimador final."""
        mask = (np.ones(len(self.df), dtype=bool) if self.sample_col is None
                else self._frame_mask(sample))
        pos = pd.Series(np.flatnonzero(mask), index=self.df.index[mask])
        # sorteia as POSIÇÕES (mesmo tamanho e seed ⇒ mesmas linhas que sortear
        # o recorte transformado) e só então copia e transforma essas linhas: a
        # referência inteira não é copiada nem densificada para usar 2 mil.
        if sample_size and len(pos) > sample_size:
            pos = pos.sample(sample_size, random_state=self.random_state)
        sub = self.df.iloc[pos.to_numpy()][list(self.model_features)]
        est, Xt = self._shap_transform(sub)
        names = list(Xt.columns)
        return est, Xt, names

    def shap_values(self, sample=None, sample_size=2000):
        """Calcula (e cacheia) os valores SHAP do modelo criado."""
        key = (sample, sample_size)
        if key in self._shap_cache:
            return self._shap_cache[key]
        from ...interpretability.shap_explain import compute_shap
        est, Xt, _names = self._shap_inputs(sample, sample_size)
        sv, Xs = compute_shap(est, Xt, problem_type=self.task_type, sample_size=None)
        self._shap_cache[key] = (sv, Xs)
        return sv, Xs

    def shap_importance(self, sample=None, sample_size=2000) -> pd.DataFrame:
        from ...interpretability.shap_explain import shap_feature_importance
        sv, Xs = self.shap_values(sample, sample_size)
        return shap_feature_importance(sv, Xs.columns)

    def _original_feature_of(self, name: str) -> str:
        """Mapeia UM nome de coluna transformada (``num__idade``, ``cat__uf_SP``,
        ``WoE(x)``) de volta para a variável ORIGINAL do modelo (``idade``, ``uf``,
        ``x``). Casa pelo prefixo mais longo em ``model_features`` — desambigua
        nomes de variáveis que contêm ``_`` (ex.: ``uf`` vs ``uf_regiao``)."""
        raw = str(name)
        dummy = raw.startswith("dum__")
        for p in ("num__", "cat__", "ord__", "woe__", "dum__"):
            if raw.startswith(p):
                raw = raw[len(p):]
                break
        if dummy and "=" in raw:                 # dummy de scorecard: 'var=faixa'
            return raw.split("=", 1)[0]
        for w in ("WoE", "bin", "ord"):
            if raw.startswith(f"{w}(") and raw.endswith(")"):
                raw = raw[len(w) + 1:-1]
                break
        feats = list(self.model_features or [])
        if raw in feats:                         # numérica / WoE (1:1)
            return raw
        cand = [f for f in feats if raw == f or raw.startswith(f + "_")]
        if cand:
            return max(cand, key=len)            # dummy de categórica → variável de origem
        return raw

    def shap_importance_grouped(self, sample=None, sample_size=2000) -> pd.DataFrame:
        """Importância SHAP **agregada por variável original**: soma a
        ``média(|SHAP|)`` de todas as colunas geradas por cada variável — as
        *dummies* de uma categórica entram numa **única** barra. Devolve
        ``[variavel, variavel_label, importancia, pct]`` ordenado do maior para o
        menor; ``pct`` é a importância RELATIVA (0–100%). Ver :meth:`shap_importance`
        (por coluna transformada) e :meth:`plot_shap_importance_relative`."""
        imp = self.shap_importance(sample, sample_size)     # feature, mean_abs_shap
        grp: dict = {}
        for _, r in imp.iterrows():
            orig = self._original_feature_of(r["feature"])
            grp[orig] = grp.get(orig, 0.0) + float(r["mean_abs_shap"])
        out = (pd.DataFrame({"variavel": list(grp.keys()),
                             "importancia": list(grp.values())})
               .sort_values("importancia", ascending=False).reset_index(drop=True))
        total = float(out["importancia"].sum())
        out["pct"] = (100.0 * out["importancia"] / total) if total > 0 else 0.0
        out["variavel_label"] = out["variavel"].map(self.label)
        return out[["variavel", "variavel_label", "importancia", "pct"]]

    def plot_shap_importance_relative(self, sample=None, sample_size=2000, max_display=20,
                                      figsize=(7.8, 4.8), dpi=150, save_path=None, ax=None):
        """Barras horizontais da importância **relativa (%)** de TODAS as variáveis
        que entraram no modelo — com as *dummies* de cada categórica somadas numa
        única barra (ver :meth:`shap_importance_grouped`). Complementa o beeswarm e a
        importância global (por coluna) com uma leitura por VARIÁVEL."""
        imp = self.shap_importance_grouped(sample, sample_size)
        fig, ax = _new_ax(figsize, dpi, ax)
        if imp.empty:
            ax.text(0.5, 0.5, "sem importância SHAP", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            return fig
        imp = imp.head(max_display).iloc[::-1]           # maior no topo do barh
        labels = [self._truncate_label(s) for s in imp["variavel_label"]]
        y = list(range(len(imp)))
        pct = imp["pct"].to_numpy()
        ax.barh(y, pct, color="#3b6ea5", alpha=0.9, edgecolor="#27324a", linewidth=0.4)
        for yi, p in zip(y, pct):
            ax.text(p, yi, f" {p:.1f}%", va="center", ha="left", fontsize=8, color="#333")
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("importância relativa (%)")
        ax.set_xlim(0, min(100.0, float(np.nanmax(pct)) * 1.18 + 3))
        ax.grid(axis="x", alpha=0.12)
        ax.set_title("SHAP — importância relativa por variável (categóricas agregadas)",
                     fontsize=11, fontweight="bold", color="#15324a")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    @staticmethod
    def _truncate_label(s, n=28) -> str:
        """Corta rótulos muito longos (evita que os nomes dominem e 'espalhem' o
        gráfico SHAP, espremendo o swarm). Corta pelo MEIO — preserva início e fim
        — para não confundir nomes com prefixo comum (ex.: ``..._var_02`` vs
        ``..._var_03``)."""
        s = str(s)
        if len(s) <= n:
            return s
        head = (n - 1) // 2
        tail = n - 1 - head
        return s[:head] + "…" + s[-tail:]

    def _shap_feature_names(self, cols) -> list:
        """Rótulos das variáveis para os gráficos SHAP: alias (``feature_labels``)
        + corte de nomes muito longos."""
        return [self._truncate_label(self._display_feature_name(c)) for c in cols]

    def _shap_agrupa_dummies(self, sv, Xs):
        """Junta as colunas de **dummies de scorecard** (``dum__var=faixa``) numa
        coluna só por variável, para os gráficos SHAP tratarem a variável como UMA:
        SHAP = soma das dummies; valor = posição da faixa por risco (0 = pior ...
        maior = melhor; linha com todas as dummies em 0 = referência = 0). As
        demais colunas passam intactas. Devolve ``(sv, Xs)`` novos."""
        cols = [str(c) for c in Xs.columns]
        grupos: dict = {}
        for j, c in enumerate(cols):
            if c.startswith("dum__") and "=" in c:
                f, faixa = c[len("dum__"):].split("=", 1)
                grupos.setdefault(f, []).append((j, faixa))
        if not grupos:
            return sv, Xs
        sv = np.asarray(sv)
        novas_sv, novas_x, nomes, usados = [], [], [], set()
        for j, c in enumerate(cols):
            if j in usados:
                continue
            f = (c[len("dum__"):].split("=", 1)[0]
                 if c.startswith("dum__") and "=" in c else None)
            if f is None or f not in grupos:
                novas_sv.append(sv[:, j]); novas_x.append(Xs.iloc[:, j].to_numpy()); nomes.append(c)
                continue
            membros = grupos.pop(f)
            idx = [m for m, _ in membros]
            usados.update(idx)
            ordem = {self._bin_label(f, b): o
                     for b, _r, _n, o in self._faixas_por_risco(f)}
            valor = np.zeros(len(Xs), dtype="float64")          # referência = pior = 0
            for m, faixa in membros:
                valor = np.where(Xs.iloc[:, m].to_numpy() > 0.5, ordem.get(faixa, 0), valor)
            novas_sv.append(sv[:, idx].sum(axis=1)); novas_x.append(valor)
            nomes.append(f"dum__{f}")
        return (np.column_stack(novas_sv),
                pd.DataFrame(np.column_stack(novas_x), columns=nomes, index=Xs.index))

    def plot_shap_beeswarm(self, sample=None, sample_size=2000, max_display=15):
        """Beeswarm SHAP do modelo (usa pyplot; devolve a figura). Usa o alias das
        variáveis (``feature_labels``) e corta nomes longos no eixo Y. Dummies de
        scorecard aparecem como UMA variável (ver :meth:`_shap_agrupa_dummies`)."""
        import matplotlib.pyplot as plt
        import shap
        sv, Xs = self._shap_agrupa_dummies(*self.shap_values(sample, sample_size))
        names = self._shap_feature_names(Xs.columns)     # alias + corte (não muta Xs/cache)
        plt.figure()
        shap.summary_plot(sv, Xs, feature_names=names, show=False, max_display=max_display)
        fig = plt.gcf()
        try:                                   # eixos/legenda em português
            fig.axes[0].set_xlabel("valor SHAP (impacto na saída do modelo)")
            if len(fig.axes) > 1:              # colorbar ("heatmap") à direita
                cb = fig.axes[-1]
                cb.set_ylabel("valor da variável")
                cb.set_yticklabels(["baixo", "alto"])
        except Exception:
            pass
        fig.suptitle("SHAP — contribuição por variável", fontsize=11,
                     fontweight="bold", color="#15324a")
        fig.tight_layout()
        plt.close(fig)          # tira do Gcf: evita re-exibição inline e vazamento de figuras
        return fig

    def plot_shap_bar(self, sample=None, sample_size=2000, max_display=15):
        """Importância global SHAP (barras). Usa o alias das variáveis
        (``feature_labels``) e corta nomes longos no eixo Y. Dummies de scorecard
        aparecem como UMA variável."""
        import matplotlib.pyplot as plt
        import shap
        sv, Xs = self._shap_agrupa_dummies(*self.shap_values(sample, sample_size))
        names = self._shap_feature_names(Xs.columns)     # alias + corte (não muta Xs/cache)
        plt.figure()
        shap.summary_plot(sv, Xs, plot_type="bar", feature_names=names, show=False,
                          max_display=max_display)
        fig = plt.gcf()
        try:
            fig.axes[0].set_xlabel("média(|valor SHAP|) — impacto médio na saída do modelo")
        except Exception:
            pass
        fig.suptitle("SHAP — importância global (|valor| médio)", fontsize=11,
                     fontweight="bold", color="#15324a")
        fig.tight_layout()
        plt.close(fig)          # tira do Gcf: evita re-exibição inline e vazamento de figuras
        return fig

    # ---- SHAP local: dependence · explicação por linha · reason codes ----
    def _group_shap_matrix(self, sv, Xs) -> pd.DataFrame:
        """Agrega a matriz de SHAP ``(linhas × colunas transformadas)`` por
        variável ORIGINAL: soma as colunas geradas por cada variável (as
        *dummies* de uma categórica entram juntas — ver
        :meth:`_original_feature_of`). Devolve DataFrame ``(linhas × variáveis)``
        alinhado ao índice de ``Xs``."""
        sv = np.asarray(sv)
        groups: dict = {}
        for j, c in enumerate(Xs.columns):
            groups.setdefault(self._original_feature_of(str(c)), []).append(j)
        data = {orig: sv[:, cols].sum(axis=1) for orig, cols in groups.items()}
        return pd.DataFrame(data, index=Xs.index)

    def shap_contributions(self, sample=None, sample_size=2000) -> pd.DataFrame:
        """Contribuições SHAP **por variável original**, linha a linha: DataFrame
        ``(linhas × variáveis do modelo)`` com a soma das colunas transformadas de
        cada variável (dummies de categórica agregadas), alinhado à amostra usada
        no cache de :meth:`shap_values`. Base para o dependence
        (:meth:`plot_shap_dependence`) e para os reason codes
        (:meth:`reason_codes`)."""
        sv, Xs = self.shap_values(sample, sample_size)
        return self._group_shap_matrix(sv, Xs)

    def plot_shap_dependence(self, feature, sample=None, sample_size=2000,
                             figsize=(7.0, 4.4), dpi=150, save_path=None, ax=None):
        """Dependence SHAP por variável ORIGINAL: dispersão do valor da variável
        (eixo X) × contribuição SHAP agregada pela variável (eixo Y — dummies de
        categórica somadas). Numéricas viram dispersão contínua; categóricas,
        dispersão por categoria (com *jitter*) e a média por categoria marcada.
        Matplotlib puro — não depende dos gráficos nativos do ``shap``."""
        M = self.shap_contributions(sample, sample_size)
        if feature not in M.columns:
            raise ValueError(f"'{feature}' não é variável do modelo. "
                             f"Opções: {sorted(M.columns)}")
        contrib = M[feature].to_numpy(dtype="float64")
        vals = self.df.loc[M.index, feature]
        fig, ax = _new_ax(figsize, dpi, ax)
        if self._detect_kind(feature) == "num":
            v = pd.to_numeric(vals, errors="coerce").to_numpy(dtype="float64")
            ok = np.isfinite(v)
            ax.scatter(v[ok], contrib[ok], s=14, alpha=0.45, color="#3b6ea5",
                       edgecolors="none")
            if (~ok).any():                     # faltantes não têm posição no eixo X
                ax.text(0.02, 0.03,
                        f"faltantes: n={int((~ok).sum())} · contribuição média "
                        f"{float(np.mean(contrib[~ok])):+.4f}",
                        transform=ax.transAxes, fontsize=8, color="#889")
            ax.set_xlabel(self.label(feature), fontsize=9)
        else:
            s = pd.Series(np.where(vals.isna(), "(faltante)", vals.astype(str)),
                          index=vals.index)
            medias = (pd.Series(contrib, index=s.index).groupby(s).mean()
                      .sort_values())
            cats = list(medias.index)
            rng = np.random.default_rng(self.random_state)
            for i, c in enumerate(cats):
                m = (s == c).to_numpy()
                x = i + rng.uniform(-0.18, 0.18, int(m.sum()))
                ax.scatter(x, contrib[m], s=12, alpha=0.4, color="#3b6ea5",
                           edgecolors="none")
                mu = float(medias[c])           # média da categoria em destaque
                ax.plot([i - 0.3, i + 0.3], [mu, mu], color="#d6453e", lw=2.2,
                        solid_capstyle="round", zorder=3)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels([self._truncate_label(c, 16) for c in cats],
                               rotation=30, ha="right", fontsize=8)
            ax.set_xlabel(f"{self.label(feature)} (média por categoria em vermelho)",
                          fontsize=9)
        ax.axhline(0.0, color="#889", lw=0.8, ls="--")
        ax.set_ylabel("contribuição SHAP (agregada pela variável)", fontsize=9)
        ax.grid(alpha=0.12)
        ax.set_title(f"SHAP — dependence · {self._truncate_label(self.label(feature))}",
                     fontsize=11, fontweight="bold", color="#15324a")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def _shap_row(self, index, sample=None, sample_size=2000, background=100):
        """Valores SHAP (por coluna transformada) de UMA linha do ``df``. Reusa o
        cache de :meth:`shap_values` quando a linha caiu na amostra; senão calcula
        só para ela, com um pequeno ``background`` da própria amostra (necessário
        aos explicadores que precisam de base). Devolve ``(sv_linha, x_linha)``."""
        if index not in self.df.index:
            raise KeyError(f"Índice {index!r} não existe no DataFrame do segmenter.")
        sv, Xs = self.shap_values(sample, sample_size)
        try:
            pos = int(Xs.index.get_indexer([index])[0])
        except Exception:
            pos = -1
        if pos >= 0:
            return np.asarray(sv)[pos], Xs.iloc[pos]
        from ...interpretability.shap_explain import compute_shap
        rows = [index] + [i for i in Xs.index[:background] if i != index]
        est, Xt = self._shap_transform(self.df.loc[rows, self.model_features])
        sv2, _ = compute_shap(est, Xt, problem_type=self.task_type, sample_size=None)
        return np.asarray(sv2)[0], Xt.iloc[0]

    def row_contributions(self, index, sample=None, sample_size=2000) -> pd.DataFrame:
        """Contribuições SHAP de UMA observação, agregadas por variável original.
        Devolve ``[variavel, variavel_label, valor, contribuicao]`` ordenado por
        |contribuição| decrescente; ``valor`` é o valor CRU da variável na linha.
        Em ``.attrs``: ``score`` (score cru da linha, 0–1 na classificação) e
        ``base_value`` — obtido por diferença (``score − Σ contribuições``), o que
        garante que o gráfico feche EXATAMENTE no score da linha mesmo quando o
        explicador trabalha noutra escala (ex.: log-odds)."""
        if self.model is None:
            raise RuntimeError("Ajuste/defina o modelo antes (fit / set_model).")
        sv_row, x_row = self._shap_row(index, sample, sample_size)
        contrib: dict = {}
        for j, c in enumerate(x_row.index):
            orig = self._original_feature_of(str(c))
            contrib[orig] = contrib.get(orig, 0.0) + float(sv_row[j])
        rows = []
        for var, v in contrib.items():
            raw = self.df.loc[index, var] if var in self.df.columns else None
            if raw is None or pd.isna(raw):
                valor = "(faltante)"
            elif isinstance(raw, (int, float, np.floating, np.integer)):
                valor = _fmt(raw)
            else:
                valor = str(raw)
            rows.append({"variavel": var, "variavel_label": self.label(var),
                         "valor": valor, "contribuicao": v})
        out = pd.DataFrame(rows)
        out = (out.reindex(out["contribuicao"].abs().sort_values(ascending=False).index)
               .reset_index(drop=True))
        score = float(self._predict_score_array(
            self.model, self.df.loc[[index], self.model_features])[0])
        out.attrs["score"] = score
        out.attrs["base_value"] = score - float(out["contribuicao"].sum())
        out.attrs["index"] = index
        return out

    def explain_row(self, index, top_n=10, sample=None, sample_size=2000,
                    figsize=(7.6, 4.8), dpi=150, save_path=None, ax=None):
        """Waterfall horizontal (barh) da observação ``index``: parte do valor-base
        e acumula as top-N contribuições SHAP (por variável original, dummies
        agregadas) até o score da linha; as demais variáveis entram somadas numa
        única barra. Rótulos usam ``feature_labels`` + o valor cru da variável.
        Vermelho empurra o score para CIMA; azul, para baixo. Dados em
        :meth:`row_contributions`."""
        rc = self.row_contributions(index, sample=sample, sample_size=sample_size)
        base = float(rc.attrs["base_value"])
        score = float(rc.attrs["score"])
        itens = [(f"{self._truncate_label(r['variavel_label'], 24)} = "
                  f"{self._truncate_label(r['valor'], 14)}", float(r["contribuicao"]))
                 for _, r in rc.head(top_n).iterrows()]
        if len(rc) > top_n:
            resto = float(rc.iloc[top_n:]["contribuicao"].sum())
            itens.append((f"(demais {len(rc) - top_n} variáveis)", resto))
        fig, ax = _new_ax(figsize, dpi, ax)
        cum = base
        for i, (_lab, c) in enumerate(itens):
            left = cum if c >= 0 else cum + c
            ax.barh(i, abs(c), left=left,
                    color=("#d6453e" if c >= 0 else "#3b6ea5"),
                    alpha=0.9, edgecolor="#27324a", linewidth=0.4, height=0.62)
            ax.text(left + abs(c), i, f" {c:+.4f}", va="center", ha="left",
                    fontsize=8, color="#33424f")
            cum += c
        ax.axvline(base, color="#889", lw=1.0, ls="--")
        ax.axvline(score, color="#15324a", lw=1.2, ls=":")
        ax.set_yticks(range(len(itens)))
        ax.set_yticklabels([lab for lab, _ in itens], fontsize=8)
        ax.invert_yaxis()                        # maior contribuição no topo
        ax.set_xlabel(f"base {base:+.4f}  →  score {score:.4f} (escala crua)",
                      fontsize=9)
        ax.grid(axis="x", alpha=0.12)
        ax.set_title(f"SHAP — explicação da linha {index!r}",
                     fontsize=11, fontweight="bold", color="#15324a")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def reason_codes(self, X=None, top_n=3, use_labels=False, prefix="motivo_",
                     sample=None, sample_size=2000) -> pd.DataFrame:
        """Top-N motivos (*reason codes*) por linha: as variáveis de maior
        |contribuição SHAP| agregada por variável original, com o sinal —
        ``(+)`` empurra o score para cima, ``(-)`` para baixo. Sem ``X``, usa a
        amostra do cache de :meth:`shap_values` (``sample``/``sample_size``); com
        ``X`` (lote a escorar), calcula SHAP para TODAS as linhas de ``X`` —
        caminho pandas (na escoragem Spark distribuída os motivos não são
        anexados; ver :meth:`apply_spark`). Devolve DataFrame com as colunas
        ``{prefix}1..{prefix}N`` (ex.: ``motivo_1``), alinhado ao índice das
        linhas explicadas. Requer o pacote opcional ``shap``."""
        try:
            import shap                                    # noqa: F401
        except ImportError as e:
            raise ImportError("reason_codes requer o pacote opcional 'shap' — "
                              "pip install shap") from e
        if self.model is None:
            raise RuntimeError("Ajuste/defina o modelo antes (fit / set_model).")
        if X is None:
            M = self.shap_contributions(sample, sample_size)
        else:
            from ...interpretability.shap_explain import compute_shap
            X = self._apply_derived(pd.DataFrame(X))
            est, Xt = self._shap_transform(X[self.model_features])
            sv, _ = compute_shap(est, Xt, problem_type=self.task_type,
                                 sample_size=None)
            M = self._group_shap_matrix(sv, Xt)
        A = M.to_numpy(dtype="float64")
        names = [str(self.label(c)) if use_labels else str(c) for c in M.columns]
        ordem = np.argsort(-np.abs(A), axis=1)             # |contribuição| desc
        k = max(1, min(int(top_n), A.shape[1]))
        linhas = np.arange(len(A))
        out = {}
        for j in range(k):
            idx = ordem[:, j]
            sinal = np.where(A[linhas, idx] >= 0, "(+)", "(-)")
            out[f"{prefix}{j + 1}"] = [f"{names[i]} {s}" for i, s in zip(idx, sinal)]
        return pd.DataFrame(out, index=M.index)

    # ------------------------------------------------------------------
    # D) Score → Ratings
    # ------------------------------------------------------------------
    def _make_cfg(self, sample_col):
        return ColumnConfig(date_col=self.date_col or "dt_ref", sample_col=sample_col,
                            target_col=self.target, score_col="score",
                            dev_sample=self.ref_sample, oot_sample=self._oot_sample())

    def _rating_frame(self):
        wf = pd.DataFrame(index=self.df.index)
        wf["score"] = self.score_
        wf[self.target] = self.df[self.target]
        if self.sample_col is not None:
            wf["_amostra"] = self.df[self.sample_col].astype(object)
        else:
            wf["_amostra"] = self.ref_sample
        return wf

    def build_ratings(self, method="quantil", n_ratings=10, monotonic_fusion=True,
                      alpha=0.05, label_style=None, cuts=None, percentiles=None):
        """Segmenta o score em ratings ordenados. ``method`` ∈ {decis, quantil,
        arvore, optbin, manual_score, manual_percentil}; ``n_ratings`` é o
        número-alvo de faixas (a fusão monotônica pode reduzi-lo). Para os métodos
        manuais: ``manual_score`` usa ``cuts`` (lista de cortes de score) e
        ``manual_percentil`` usa ``percentiles`` (lista 0–100). Reaproveita
        :mod:`yggdrasil.ratings`."""
        if self.score_ is None:
            raise RuntimeError("Gere o score antes (fit / set_model / load).")
        if method not in RATING_REGISTRY:
            raise ValueError(f"Método de rating desconhecido: {method!r}. "
                             f"Opções: {sorted(RATING_REGISTRY)}")
        if isinstance(n_ratings, str) and n_ratings.lower() == "auto":
            sug = self.suggest_n_ratings(method=method, monotonic_fusion=monotonic_fusion,
                                         alpha=alpha)
            n_ratings = sug["best"]
            self._last_auto_suggestion = sug
        n = int(n_ratings)
        if method == "decis":
            strat = RATING_REGISTRY[method](n=n)
        elif method == "quantil":
            strat = RATING_REGISTRY[method](step=1.0 / max(n, 1), alpha=alpha)
        elif method == "arvore":
            strat = RATING_REGISTRY[method](max_leaf_nodes=n, alpha=alpha,
                                            random_state=self.random_state)
        elif method == "manual_score":
            if not cuts:
                raise ValueError("manual_score requer 'cuts' (lista de cortes de score).")
            strat = RATING_REGISTRY[method](cuts=cuts)
        elif method == "manual_percentil":
            if not percentiles:
                raise ValueError("manual_percentil requer 'percentiles' (lista 0–100).")
            strat = RATING_REGISTRY[method](percentiles=percentiles)
        else:  # optbin
            strat = RATING_REGISTRY[method](max_n_bins=n)
        # respeita as flags quando a estratégia as expõe (manuais não fundem)
        if method not in ("decis", "manual_score", "manual_percentil"):
            strat.monotonic_fusion = bool(monotonic_fusion)
        if label_style:
            strat.label_style = label_style

        wf = self._rating_frame()
        cfg = self._make_cfg("_amostra")
        fit_df = wf[wf["score"].notna() & wf[self.target].notna()]
        strat.fit(fit_df, cfg, problem_type=self.task_type)
        self.rating_ = strat.transform(wf, cfg)
        self.rating_strategy = strat
        self.rating_col_ = strat.column
        self.rating_labels_ = list(strat.labels_)
        self.rating_config = {"method": method, "n_ratings": n,
                              "monotonic_fusion": bool(monotonic_fusion), "alpha": alpha}
        return self

    def _rating_gini(self) -> float:
        """Discriminação retida pela régua atual na referência (DES): Gini usando o
        risco médio de cada rating como score. ``NaN`` fora da classificação."""
        if self.task_type != "classification" or self.rating_ is None:
            return float("nan")
        rating = self.rating_
        ref_mask = (pd.Series(True, index=self.df.index) if self.sample_col is None
                    else self.df[self.sample_col] == self.ref_sample)
        risco_by = {lab: self._risco(self.df.loc[(rating == lab) & ref_mask, self.target])
                    for lab in self.rating_labels_}
        sub = self.df[ref_mask]
        y = sub[self.target].to_numpy(dtype="float64")
        sc = rating[ref_mask].map(risco_by).to_numpy(dtype="float64")
        ok = ~np.isnan(y) & ~np.isnan(sc)
        if ok.sum() < 2 or np.unique(y[ok]).size < 2:
            return float("nan")
        return float(classification_metrics(y[ok], sc[ok]).get("gini", np.nan))

    def suggest_n_ratings(self, method="quantil", n_min=3, n_max=15,
                          monotonic_fusion=True, alpha=0.05, min_repr=0.02) -> dict:
        """Deixa o algoritmo escolher o nº de ratings. Testa de ``n_max`` a ``n_min``
        e recomenda a régua **mais granular** que mantém a ordem de risco monotônica
        entre amostras (sem inversões) e com volume mínimo por faixa (``min_repr``,
        fração da DES). Se nenhuma zera as inversões, escolhe a de menor inversão e
        maior granularidade.

        Método **não-destrutivo**: restaura a régua atual ao final. Devolve
        ``{'best': int, 'table': DataFrame, 'reason': str}``."""
        if self.score_ is None:
            raise RuntimeError("Gere o score antes (fit / set_model).")
        snap = (self.rating_, self.rating_strategy, self.rating_col_,
                list(self.rating_labels_), dict(self.rating_config))
        n_top = max(int(n_min), min(int(n_max), max(2, int(self.score_.nunique()))))
        rows, evals = [], {}
        try:
            for n in range(n_top, int(n_min) - 1, -1):
                try:
                    self.build_ratings(method=method, n_ratings=n,
                                       monotonic_fusion=monotonic_fusion, alpha=alpha)
                except Exception:
                    continue
                eff = len(self.rating_labels_)
                inv = self.rating_inversion()
                rt = self.rating_table()
                repr_min = (float(rt["repr_%"].min()) / 100) if len(rt) else 0.0
                mono_ok = inv["sample_inv"] == 0
                vol_ok = repr_min >= min_repr
                gini = self._rating_gini()
                rows.append({"n_alvo": n, "n_efetivo": eff,
                             "inv_amostra": int(inv["sample_inv"]),
                             "safras_inv_%": round(100 * inv["safra_rate"], 0),
                             "repr_min_%": round(100 * repr_min, 1),
                             "gini": round(gini, 4) if np.isfinite(gini) else np.nan,
                             "ok": bool(mono_ok and vol_ok)})
                evals[n] = (mono_ok, vol_ok, eff, inv["safra_rate"], gini,
                            int(inv["sample_inv"]))
        finally:
            (self.rating_, self.rating_strategy, self.rating_col_,
             self.rating_labels_, self.rating_config) = snap

        if not evals:
            raise RuntimeError("Não foi possível avaliar nenhuma régua de ratings.")
        table = pd.DataFrame(rows).sort_values("n_alvo").reset_index(drop=True)
        passa = [n for n, e in evals.items() if e[0] and e[1]]
        if passa:
            ginis = {n: evals[n][4] for n in passa}
            gmax = max((g for g in ginis.values() if np.isfinite(g)), default=float("nan"))
            if np.isfinite(gmax) and gmax > 0:
                # parcimônia: a MENOR régua que retém ≥ 99% do Gini máximo possível
                # (ponto de cotovelo — mais faixas quase não agregam discriminação)
                keep = [n for n in passa if np.isfinite(ginis[n]) and ginis[n] >= 0.99 * gmax]
                best = min(keep or passa, key=lambda n: (evals[n][2], evals[n][3]))
                reason = (f"{best} ratings — menor régua que mantém monotonia entre amostras, "
                          f"≥ {min_repr * 100:.0f}% da base por faixa e ≥ 99% do Gini máximo "
                          f"({gmax:.3f}); mais faixas quase não agregam discriminação.")
            else:  # sem métrica de discriminação (regressão): mais granular válida
                best = max(passa, key=lambda n: (evals[n][2], -evals[n][3]))
                reason = (f"{best} ratings — régua mais granular que mantém a monotonia entre "
                          f"amostras (0 inversões) e ≥ {min_repr * 100:.0f}% da base por faixa.")
        else:
            # nenhuma zerou inversões: prioriza a MENOR contagem de inversões entre
            # amostras (inteiro, não só o flag ==0), depois volume adequado, menos
            # inversão de safra e maior granularidade
            best = min(evals, key=lambda n: (evals[n][5], not evals[n][1],
                                             evals[n][3], -evals[n][2]))
            reason = (f"{best} ratings — nenhuma régua zerou as inversões; escolhida a de "
                      f"MENOR nº de inversões entre amostras ({evals[best][5]}), volume "
                      f"adequado e maior granularidade.")
        return {"best": int(best), "table": table, "reason": reason,
                "n_recomendado": int(best)}

    def _rating_series(self) -> pd.Series:
        if self.rating_ is None:
            raise RuntimeError("Gere os ratings antes (build_ratings).")
        return self.rating_

    def _rating_codes(self) -> np.ndarray:
        """Índice do rating (na ordem de ``rating_labels_``) de cada linha do
        ``df``, ``-1`` = sem rating. Memoizado pela identidade de ``rating_`` —
        comparar a coluna de texto com cada rótulo (``rating == lab``) por
        amostra/safra custava ~550 varreduras de milhões de linhas."""
        rating = self._rating_series()
        hit = self.__dict__.get("_rating_codes_cache")
        if hit is not None and hit[0] is rating and hit[1] == list(self.rating_labels_):
            return hit[2]
        codes = pd.Categorical(rating, categories=list(self.rating_labels_)).codes
        codes = np.asarray(codes, dtype=np.int32)
        self._rating_codes_cache = (rating, list(self.rating_labels_), codes)
        return codes

    def _sample_codes(self):
        """``(codigos, amostras)``: índice da amostra de cada linha (``-1`` = fora),
        na ordem de :meth:`_samples`."""
        amostras = self._samples()
        cod = np.full(len(self.df), -1, dtype=np.int32)
        for j, a in enumerate(amostras):
            cod[self._rows_mask(a)] = j
        return cod, amostras

    def rating_table(self) -> pd.DataFrame:
        """Por rating (na ordem dos rótulos): n, repr_% (na DES) e o risco
        (event_rate/alvo médio) em **cada amostra** — leitura de monotonicidade e
        estabilidade da régua entre amostras.

        Inclui também o **teste de calibração** de cada rating na amostra de
        referência (ver :meth:`_calib_test`): ``ic_low``/``ic_high`` = IC 95% do
        realizado (Jeffreys na classificação; t pareado na regressão, dado o n
        do rating) e ``status_teste`` = semáforo ``ok``/``atencao``/``alerta``
        conforme o score médio do rating cai dentro do IC 95%, só do 99%, ou
        fora de ambos."""
        rating = self._rating_series()
        labels = self.rating_labels_
        prefix = "event_rate" if self.task_type == "classification" else "alvo"
        # teste de calibração por rating (na referência) num ÚNICO groupby —
        # sem máscara full-length por rótulo (ver notas de performance).
        tests = None
        if self.score_ is not None:
            sub_ts = pd.DataFrame({"y": self.df[self.target], "s": self.score_,
                                   "r": rating})
            if self.sample_col is not None:
                sub_ts = sub_ts[self._frame_mask(self.ref_sample)]
            tests = {lab: self._calib_test(g["y"], g["s"])
                     for lab, g in sub_ts.groupby("r", observed=True)}

        def _test_cols(row, lab):
            if tests is None:
                return row
            t = tests.get(lab) or {"ic_low": np.nan, "ic_high": np.nan,
                                   "status": "alerta"}
            row["ic_low"] = (round(float(t["ic_low"]), 4)
                             if np.isfinite(t["ic_low"]) else np.nan)
            row["ic_high"] = (round(float(t["ic_high"]), 4)
                              if np.isfinite(t["ic_high"]) else np.nan)
            row["status_teste"] = t["status"]
            return row

        # risco (= nanmean, ver _risco) por (rating, amostra) num ÚNICO groupby, em
        # vez de recriar `(rating==lab) & (sample==a)` full-length por célula.
        if self.sample_col is None:
            risk = self.df.groupby(rating, observed=True)[self.target].mean()
            n_by = rating.value_counts()
            n_ref_tot = max(len(self.df), 1)
            rows = []
            for lab in labels:
                n = int(n_by.get(lab, 0))
                v = risk.get(lab, np.nan)
                row = {"rating": lab, "n": n,
                       "repr_%": round(100 * n / n_ref_tot, 1),
                       f"{prefix}_{self.ref_sample}":
                           round(float(v), 4) if pd.notna(v) else np.nan}
                rows.append(_test_cols(row, lab))
            return pd.DataFrame(rows)

        risk = self.df.groupby([rating, self.df[self.sample_col]],
                               observed=True)[self.target].mean()
        ref_mask = self._frame_mask(self.ref_sample)
        n_ref = rating[ref_mask].value_counts()
        n_ref_tot = max(int(ref_mask.sum()), 1)
        rows = []
        for lab in labels:
            n = int(n_ref.get(lab, 0))
            row = {"rating": lab, "n": n,
                   "repr_%": round(100 * n / n_ref_tot, 1)}
            for a in self._samples():
                v = risk.get((lab, a), np.nan)
                row[f"{prefix}_{a}"] = round(float(v), 4) if pd.notna(v) else np.nan
            rows.append(_test_cols(row, lab))
        return pd.DataFrame(rows)

    def rating_inversion(self, time_col=None, sample=None, min_n=20) -> dict:
        """Inversão da ordem de risco ENTRE ratings (entre amostras e safras) —
        o estudo de folhas-irmãs do alvo, aplicado às faixas de rating."""
        rating = self._rating_series()
        labels = self.rating_labels_
        risco_by = {}

        # risco (= nanmean) por (rating, amostra) num único groupby — substitui o
        # `(rating==lab) & (sample==a)` full-length por célula (rating × amostra).
        if self.sample_col is None:
            _risk_overall = self.df.groupby(rating, observed=True)[self.target].mean()

            def _risk_of(lab, a):
                v = _risk_overall.get(lab, np.nan)
                return float(v) if pd.notna(v) else float("nan")
        else:
            _risk = self.df.groupby([rating, self.df[self.sample_col]],
                                    observed=True)[self.target].mean()

            def _risk_of(lab, a):
                v = _risk.get((lab, a), np.nan)
                return float(v) if pd.notna(v) else float("nan")

        # ordem de referência: pela média de risco na DES
        ref_risco = {lab: _risk_of(lab, self.ref_sample) for lab in labels}
        ordered = sorted(labels, key=lambda l: (np.inf if pd.isna(ref_risco[l])
                                                else ref_risco[l]))
        # por amostra
        sample_rows = []
        for a in self._samples():
            vals = {lab: _risk_of(lab, a) for lab in labels}
            risco_by[a] = vals
            n_inv, npp = _count_inversions(ordered, vals)
            sample_rows.append({"amostra": a, "n_inv": n_inv, "n_pares": npp})
        # por safra
        safra_rows, safra_series = [], {}
        tcol = time_col or self.date_col
        if tcol is not None and tcol in self.df.columns:
            # risco rating × safra numa passada (np.bincount sobre códigos)
            linhas = self._rows_mask(sample, all_rows=not sample)
            cod_t, rot_t = self._safra_codes(tcol)
            rc = self._rating_codes()
            n_t = np.bincount(cod_t[linhas & (cod_t >= 0)], minlength=len(rot_t))
            mat = self._risco_por_grupo(rc, cod_t, len(rot_t), len(labels), linhas)
            for k, per in enumerate(rot_t):
                if n_t[k] == 0 or n_t[k] < min_n:
                    continue
                vals = {lab: float(mat[k, i]) for i, lab in enumerate(labels)}
                safra_series[per] = vals
                n_inv, npp = _count_inversions(ordered, vals)
                if npp == 0:
                    continue
                safra_rows.append({"safra": per, "n_inv": n_inv, "n_pares": npp})

        sample_inv = sum(r["n_inv"] for r in sample_rows if r["amostra"] != self.ref_sample)
        n_safras = len(safra_rows)
        safras_inv = sum(1 for r in safra_rows if r["n_inv"] > 0)
        safra_rate = (safras_inv / n_safras) if n_safras else 0.0
        status = ("red" if (sample_inv > 0 or safra_rate > 0.25)
                  else "yellow" if safras_inv > 0 else "green")
        return {"status": status, "ordered": ordered, "ref_risco": ref_risco,
                "samples": sample_rows, "safras": safra_rows, "risco_by_sample": risco_by,
                "safra_series": safra_series, "sample_inv": sample_inv,
                "n_safras": n_safras, "safras_inv": safras_inv, "safra_rate": safra_rate}

    # ------------------------------------------------------------------
    # Separação estatística entre ratings adjacentes
    # ------------------------------------------------------------------
    @staticmethod
    def _two_proportion_pvalue(k1, n1, k2, n2) -> float:
        """p-valor (bicaudal) do teste z de duas proporções, com proporção
        agrupada (aproximação normal). ``NaN`` quando indefinido; 1,0 quando as
        duas faixas são degeneradas iguais (todas as linhas na mesma classe)."""
        try:
            from scipy.stats import norm
        except ImportError:  # pragma: no cover
            return float("nan")
        if n1 <= 0 or n2 <= 0:
            return float("nan")
        p1, p2 = k1 / n1, k2 / n2
        pool = (k1 + k2) / (n1 + n2)
        se = float(np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2)))
        if se == 0.0:                        # pool ∈ {0,1} ⇒ p1 == p2
            return 1.0 if p1 == p2 else 0.0
        return float(2 * norm.sf(abs((p1 - p2) / se)))

    def _rating_pair_pvalue(self, va, vb, min_n: int = 8) -> float:
        """p-valor do teste de igualdade do alvo entre dois ratings vizinhos
        (arrays já sem NaN): teste z de duas proporções na classificação e
        Mann-Whitney bicaudal na regressão. ``NaN`` com volume < ``min_n``."""
        if len(va) < min_n or len(vb) < min_n:
            return float("nan")
        if self.task_type == "classification":
            return self._two_proportion_pvalue(float(np.sum(va)), len(va),
                                               float(np.sum(vb)), len(vb))
        try:
            from scipy.stats import mannwhitneyu
            return float(mannwhitneyu(va, vb, alternative="two-sided").pvalue)
        except Exception:
            return float("nan")

    def adjacent_rating_tests(self, sample=None, alpha=0.05, min_n=8) -> pd.DataFrame:
        """Testa a SEPARAÇÃO ESTATÍSTICA do alvo entre cada par de ratings
        VIZINHOS (na ordem da régua), por amostra — o análogo, para a régua de
        ratings, do teste entre folhas-irmãs da árvore. A fusão monotônica
        garante a **ordem** do risco, não a significância: dois ratings podem
        ficar ordenados e ainda assim estatisticamente indistinguíveis —
        candidatos a fusão (reduzir ``n_ratings`` ou usar cortes manuais).

        Teste z de duas proporções na classificação; Mann-Whitney (bicaudal)
        na regressão. ``sample=None`` avalia todas as amostras (referência
        primeiro); pares com volume < ``min_n`` em algum lado saem como ``n/d``.

        Retorna DataFrame com ``par``, ``amostra``, ``n_a``, ``n_b``, ``teste``,
        ``p_valor`` e ``veredito`` (``separa`` se p < ``alpha``, senão
        ``nao_separa``; ``n/d`` sem volume ou sem p-valor)."""
        rating = self._rating_series()
        labels = list(self.rating_labels_)
        if sample is not None and sample not in self._samples():
            raise ValueError(f"Amostra desconhecida: {sample!r}. "
                             f"Opções: {self._samples()}")
        samples = self._samples() if sample is None else [sample]
        teste = ("proporções (z)" if self.task_type == "classification"
                 else "Mann-Whitney")
        cols = ["par", "amostra", "n_a", "n_b", "teste", "p_valor", "veredito"]
        rows, vazio = [], np.empty(0)
        for a in samples:
            m = (np.ones(len(self.df), dtype=bool) if self.sample_col is None
                 else self._frame_mask(a))
            sub = pd.DataFrame({"y": self.df[self.target], "r": rating})[m]
            sub = sub[sub["y"].notna() & sub["r"].notna()]
            grp = {lab: g["y"].to_numpy(dtype="float64")
                   for lab, g in sub.groupby("r", observed=True)}
            for la, lb in zip(labels[:-1], labels[1:]):
                va, vb = grp.get(la, vazio), grp.get(lb, vazio)
                p = self._rating_pair_pvalue(va, vb, min_n=min_n)
                verd = ("n/d" if not np.isfinite(p)
                        else "separa" if p < alpha else "nao_separa")
                rows.append({"par": f"{la} × {lb}", "amostra": a,
                             "n_a": int(len(va)), "n_b": int(len(vb)),
                             "teste": teste,
                             "p_valor": round(p, 4) if np.isfinite(p) else np.nan,
                             "veredito": verd})
        return pd.DataFrame(rows, columns=cols)

    def plot_rating_badrate(self, sample=None, figsize=(7.4, 4.0), dpi=150,
                            save_path=None, ax=None):
        """Risco médio (event_rate/alvo) por rating, na amostra escolhida."""
        rating = self._rating_series()
        labels = self.rating_labels_
        mask = (pd.Series(True, index=self.df.index) if self.sample_col is None
                else self.df[self.sample_col] == (sample or self.ref_sample))
        xs = list(range(len(labels)))
        risco = [self._risco(self.df.loc[(rating == l) & mask, self.target]) for l in labels]
        fig, ax = _new_ax(figsize, dpi, ax)
        cmap = _cmap("RdYlGn_r"); k = len(labels)
        cols = [cmap(i / (k - 1) if k > 1 else 0.5) for i in range(k)]
        ax.bar(xs, risco, color=cols, edgecolor="#33424f", alpha=0.9, width=0.72)
        for x0, r in zip(xs, risco):
            if np.isfinite(r):
                ax.annotate(f"{r*100:.1f}%", (x0, r), textcoords="offset points",
                            xytext=(0, 4), ha="center", va="bottom", fontsize=8.5,
                            color="#15324a")
        finite = [r for r in risco if np.isfinite(r)]
        if finite:
            ax.set_ylim(top=max(finite) * 1.12)      # espaço p/ o rótulo afastado
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("event_rate" if self.task_type == "classification" else "alvo médio")
        _pct_axis(ax, "y")
        ax.set_title(f"Risco por rating · {sample or self.ref_sample}", fontsize=11,
                     fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_rating_distribution(self, figsize=(7.4, 4.0), dpi=150, save_path=None, ax=None):
        """Distribuição (%) dos ratings por amostra (barras agrupadas)."""
        rating = self._rating_series()
        labels = self.rating_labels_
        samples = self._samples()
        fig, ax = _new_ax(figsize, dpi, ax)
        x = np.arange(len(labels)); w = 0.8 / max(len(samples), 1)
        palette = ["steelblue", "crimson"]       # base: referência × comparação
        stab_color = "#2a9d8f"                    # 3ª cor (teal) p/ a safra de estabilidade
        base_i = 0
        for k, a in enumerate(samples):
            am = (pd.Series(True, index=self.df.index) if self.sample_col is None
                  else self.df[self.sample_col] == a)
            n_a = max(int(am.sum()), 1)
            pct = [100 * int(((rating == l) & am).sum()) / n_a for l in labels]
            if _is_stability_sample(a):
                color = stab_color
            else:
                color = palette[base_i % len(palette)]; base_i += 1
            ax.bar(x + k * w, pct, width=w, label=a, alpha=0.9, color=color)
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("% da amostra"); ax.legend(fontsize=8)
        _pct_axis(ax, "y", xmax=100)
        ax.set_title("Distribuição dos ratings por amostra", fontsize=11,
                     fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    @staticmethod
    def _tight_ylim(values, pad_frac: float = 0.03, floor: float = 0.003):
        """Faixa de Y justa em torno dos valores finitos — para "dar zoom" e revelar
        cruzamentos/inversões que ficam comprimidos no eixo cheio. Aperta mais que a
        folga padrão do matplotlib (~5%). Retorna ``(baixo, alto)`` em fração, ou
        ``None`` se não houver valor finito."""
        v = np.asarray([x for x in np.ravel(list(values))
                        if x is not None and np.isfinite(x)], dtype=float)
        if v.size == 0:
            return None
        lo, hi = float(v.min()), float(v.max())
        pad = max((hi - lo) * pad_frac, floor)
        return (max(0.0, lo - pad), hi + pad)

    @staticmethod
    def _apply_inversion_ylim(ax, values, ylim=None, auto_zoom=False):
        """Aplica limites de Y ao gráfico de inversão. ``ylim=(baixo, alto)`` em
        fração tem precedência; lados ``None`` (ou ``auto_zoom=True``) são
        preenchidos pelo zoom automático justo aos dados. Sem nada, mantém o eixo
        padrão (começando em 0)."""
        if ylim is None and not auto_zoom:
            return
        lo = hi = None
        if ylim is not None:
            lo, hi = ylim
        if auto_zoom or lo is None or hi is None:
            tight = ModelSegmenter._tight_ylim(values)
            if tight is not None:
                lo = tight[0] if lo is None else lo
                hi = tight[1] if hi is None else hi
        if lo is not None or hi is not None:
            ax.set_ylim(bottom=lo, top=hi)

    def plot_rating_inversion_by_sample(self, figsize=(7.6, 4.0), dpi=150,
                                        save_path=None, ax=None, ylim=None,
                                        auto_zoom=False):
        inv = self.rating_inversion()
        fig, ax = _new_ax(figsize, dpi, ax)
        labels, samples = self.rating_labels_, self._samples()
        x = list(range(len(samples)))
        cmap = _cmap("RdYlGn_r"); k = len(inv["ordered"])
        for rank, lab in enumerate(inv["ordered"]):
            ys = [inv["risco_by_sample"][a].get(lab, np.nan) for a in samples]
            ax.plot(x, ys, marker="o", lw=1.9, ms=5.5,
                    color=cmap(rank / (k - 1) if k > 1 else 0.5),
                    markeredgecolor="#33424f", markeredgewidth=0.6, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(samples, fontsize=9)
        ax.set_ylabel("risco médio"); ax.set_xlabel("amostra")
        _pct_axis(ax, "y")
        self._apply_inversion_ylim(
            ax, [inv["risco_by_sample"][a].get(lab, np.nan)
                 for a in samples for lab in inv["ordered"]], ylim, auto_zoom)
        ax.set_title("Risco dos ratings por amostra (cruzamento = inversão)",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        ax.legend(fontsize=7.5, ncol=max(1, min(k, 4)), loc="best", framealpha=0.85)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_rating_inversion_by_safra(self, time_col=None, sample=None, min_n=20,
                                       figsize=(9.6, 4.0), dpi=150, save_path=None, ax=None,
                                       ylim=None, auto_zoom=False):
        inv = self.rating_inversion(time_col, sample, min_n)
        fig, ax = _new_ax(figsize, dpi, ax)
        ss = inv["safra_series"]
        if not ss:
            ax.text(0.5, 0.5, "sem dados por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        xs = list(ss.keys()); x = list(range(len(xs))); ordered = inv["ordered"]
        for j, per in enumerate(xs):
            vals = ss[per]
            n_inv, npp = _count_inversions(ordered, vals)
            if npp and n_inv:
                ax.axvspan(j - 0.5, j + 0.5, color="#d6453e", alpha=0.08, lw=0)
        cmap = _cmap("RdYlGn_r"); k = len(ordered)
        for rank, lab in enumerate(ordered):
            ys = [ss[per].get(lab, np.nan) for per in xs]
            ax.plot(x, ys, marker="o", lw=1.7, ms=4.5,
                    color=cmap(rank / (k - 1) if k > 1 else 0.5),
                    markeredgecolor="#33424f", markeredgewidth=0.5, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(_fmt_safras(xs), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("risco médio"); ax.set_xlabel("safra")
        _pct_axis(ax, "y")
        self._apply_inversion_ylim(
            ax, [ss[per].get(lab, np.nan) for per in xs for lab in ordered],
            ylim, auto_zoom)
        ax.set_title("Risco dos ratings por safra  ·  faixas vermelhas = inversão",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        ax.legend(fontsize=7.5, ncol=max(1, min(k, 4)), loc="best", framealpha=0.85)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # E) Validação / estabilidade dos ratings
    # ------------------------------------------------------------------
    def psi(self, eps: float = 1e-6) -> pd.DataFrame:
        """PSI da distribuição de RATINGS por amostra (DES como referência)."""
        if self.sample_col is None:
            raise ValueError("PSI requer sample_col.")
        labels = self.rating_labels_
        rc = self._rating_codes()
        nl = len(labels)
        dist = {}
        for a in self.df[self.sample_col].dropna().unique():
            am = self._rows_mask(a)
            # denominador = ratings NÃO-NaN da amostra (score inválido ⇒ rating NaN
            # não entra na distribuição), para as proporções somarem 1 e o PSI
            # coincidir com rating_psi_by_safra (que já normaliza por não-NaN).
            cont = np.bincount(rc[am & (rc >= 0)], minlength=nl)
            n_a = max(int(cont.sum()), 1)
            dist[a] = {l: int(cont[i]) / n_a for i, l in enumerate(labels)}
        ref = dist[self.ref_sample]
        rows = []
        for a, pct in dist.items():
            if a == self.ref_sample:
                continue
            psi = _psi_from_shares([ref[l] for l in labels],
                                   [pct[l] for l in labels], eps)
            rows.append({"amostra": a, "psi": round(psi, 4),
                         "classificacao": _classifica_psi(psi)})
        return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)

    def psi_rating_detalhe(self, comparison_samples=None, eps: float = 1e-6) -> pd.DataFrame:
        """PSI **por rating** decomposto: distribuição de cada rating na referência
        (DES) vs. cada amostra de comparação — por padrão **OOT e ESTABILIDADE**
        (todas as não-referência) — com a contribuição de PSI de cada rating.

        Para cada rating: ``%<ref>`` e, por amostra comparada, ``%<amostra>`` e
        ``PSI <amostra>`` (a parcela daquele rating no PSI). A última linha
        (``TOTAL``) traz o PSI agregado por amostra e a classificação
        (estável/atenção/instável). Complementa :meth:`psi` (que dá só o agregado)
        mostrando de ONDE vem a instabilidade — quais ratings mais deslocaram."""
        if self.sample_col is None:
            raise ValueError("PSI requer sample_col.")
        rating = self._rating_series()
        labels = self.rating_labels_
        ref = self.ref_sample
        if comparison_samples is None:
            comparison_samples = self._nonref_samples()
        comparison_samples = [a for a in comparison_samples if a != ref]
        valid = rating.notna()

        def _dist(sample):
            am = self.df[self.sample_col] == sample
            n = max(int((valid & am).sum()), 1)        # só ratings não-NaN (ver psi())
            return {l: int(((rating == l) & am).sum()) / n for l in labels}

        ref_dist = _dist(ref)
        comp_dists = {a: _dist(a) for a in comparison_samples}
        ref_shares = [ref_dist[l] for l in labels]
        decomp = {a: _psi_from_shares(ref_shares, [comp_dists[a][l] for l in labels],
                                      eps, return_contrib=True)
                  for a in comparison_samples}
        rows = []
        totais = {a: tot for a, (tot, _c) in decomp.items()}
        for i, l in enumerate(labels):
            row = {"rating": l, f"%{ref}": round(100 * ref_dist[l], 2)}
            for a in comparison_samples:
                row[f"%{a}"] = round(100 * comp_dists[a][l], 2)
                row[f"PSI {a}"] = round(decomp[a][1][i], 4)
            rows.append(row)
        total = {"rating": "TOTAL", f"%{ref}": round(100 * sum(ref_dist.values()), 1)}
        for a in comparison_samples:
            total[f"%{a}"] = round(100 * sum(comp_dists[a].values()), 1)
            total[f"PSI {a}"] = round(float(totais[a]), 4)
        rows.append(total)
        return pd.DataFrame(rows)

    def rating_psi_by_safra(self, time_col=None, eps: float = 1e-6) -> pd.DataFrame:
        """PSI da distribuição de RATINGS **por safra** vs a referência (DES).

        Mede a estabilidade da régua **ao longo do tempo**: para cada safra (mês
        de ``time_col``/``date_col``) compara a distribuição dos ratings com a da
        amostra de referência (``ref_sample``/DES) — o mesmo PSI de :meth:`psi`,
        porém período a período. Requer ratings gerados (:meth:`build_ratings`).

        Devolve ``safra``, ``n``, ``psi`` e ``classificacao`` (estável/atenção/
        instável, ver :func:`_classifica_psi`), ordenado por safra."""
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col.")
        labels = self.rating_labels_
        nl = len(labels)
        rc = self._rating_codes()
        # distribuição de referência: ratings na DES (ou toda a base, sem sample_col).
        # Denominador = ratings NÃO-NaN (score inválido ⇒ rating NaN não entra na
        # distribuição); assim as proporções somam 1, como o value_counts por safra.
        ref_mask = self._rows_mask(self.ref_sample)
        ref_cont = np.bincount(rc[ref_mask & (rc >= 0)], minlength=nl)
        n_ref = max(int(ref_cont.sum()), 1)
        ref_pct = {l: max(int(ref_cont[i]) / n_ref, eps) for i, l in enumerate(labels)}
        # contagens rating × safra numa passada (safra NaT e rating NaN ficam fora)
        cod, rot = self._safra_codes(time_col)
        ok = (cod >= 0) & (rc >= 0)
        cont = np.bincount(cod[ok].astype(np.int64) * nl + rc[ok],
                           minlength=len(rot) * nl).reshape(len(rot), nl)
        rows = []
        for k, per in enumerate(rot):
            n_g = int(cont[k].sum())                  # ignora ratings NaN no denominador
            if n_g == 0:
                continue
            psi = _psi_from_shares([ref_pct[l] for l in labels],
                                   [int(cont[k, i]) / n_g for i in range(nl)], eps)
            rows.append({"safra": per, "n": int(n_g), "psi": round(psi, 4),
                         "classificacao": _classifica_psi(psi)})
        return (pd.DataFrame(rows, columns=["safra", "n", "psi", "classificacao"])
                .sort_values("safra").reset_index(drop=True))

    def plot_rating_psi_by_safra(self, time_col=None, figsize=(9.6, 4.4), dpi=150,
                                 save_path=None, ax=None):
        """PSI da distribuição de RATINGS por safra vs DES (barras coloridas) — a
        estabilidade da régua ao longo do tempo. Ver :meth:`rating_psi_by_safra`."""
        ps = self.rating_psi_by_safra(time_col)
        fig, ax = _new_ax(figsize, dpi, ax)
        if ps.empty:
            ax.text(0.5, 0.5, "sem PSI por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            return fig
        x = list(range(len(ps)))
        cor = ["#1aa64b" if p < 0.10 else "#caa000" if p < 0.25 else "#d6453e"
               for p in ps["psi"]]
        ax.bar(x, ps["psi"], color=cor, alpha=0.92, width=0.78)
        for x0, p in zip(x, ps["psi"]):
            ax.text(x0, p, f"{p:.2f}", ha="center", va="bottom", fontsize=7, color="#555")
        # guia de alerta do PSI (sempre visível, mesmo com PSI pequeno)
        ax.axhline(0.10, color="#caa000", lw=1.2, ls="--", label="alerta (0,10)")
        ax.axhline(0.25, color="#d6453e", lw=1.2, ls="--", label="crítico (0,25)")
        ax.set_xticks(x)
        ax.set_xticklabels(_fmt_safras(ps["safra"]), rotation=45, ha="right", fontsize=8)
        ax.set_xlim(-0.7, len(ps) - 0.3)
        ax.set_ylim(0, max(float(np.nanmax(ps["psi"])) * 1.16 + 0.02, 0.28))
        ax.set_ylabel("PSI")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
        ax.set_title(f"PSI dos ratings por safra vs {self.ref_sample}", fontsize=11,
                     fontweight="bold", color="#15324a")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_rating_psi_by_sample(self, figsize=(9.6, 4.2), dpi=150, save_path=None, ax=None):
        """PSI da distribuição de RATINGS por **amostra** vs a referência (DES) —
        barras coloridas (DES × OOT, ESTABILIDADE, …). É a leitura de estabilidade
        da régua ENTRE AMOSTRAS; complementa :meth:`plot_rating_psi_by_safra` (ao
        longo do tempo). Reaproveita :meth:`psi` (que já usa DES como base e devolve
        uma linha por amostra de comparação)."""
        fig, ax = _new_ax(figsize, dpi, ax)
        try:
            ps = self.psi()
        except Exception:
            ps = None
        if ps is None or ps.empty:
            ax.text(0.5, 0.5, "sem PSI por amostra (requer amostras além da DES)",
                    ha="center", va="center", transform=ax.transAxes, color="#889")
            ax.axis("off"); fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            return fig
        # ordena OOT/ESTABILIDADE (e demais) na ordem em que aparecem na base
        order = {a: i for i, a in enumerate(self._nonref_samples())}
        ps = (ps.assign(_o=ps["amostra"].map(lambda a: order.get(a, 999)))
                .sort_values("_o").reset_index(drop=True))
        x = list(range(len(ps)))
        cor = ["#1aa64b" if p < 0.10 else "#caa000" if p < 0.25 else "#d6453e"
               for p in ps["psi"]]
        ax.bar(x, ps["psi"], color=cor, alpha=0.92, width=0.6)
        for x0, p in zip(x, ps["psi"]):
            ax.text(x0, p, f"{p:.3f}", ha="center", va="bottom", fontsize=8, color="#555")
        ax.axhline(0.10, color="#caa000", lw=1.2, ls="--", label="alerta (0,10)")
        ax.axhline(0.25, color="#d6453e", lw=1.2, ls="--", label="crítico (0,25)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in ps["amostra"]], fontsize=9)
        ax.set_xlim(-0.7, len(ps) - 0.3)
        ax.set_ylim(0, max(float(np.nanmax(ps["psi"])) * 1.18 + 0.02, 0.28))
        ax.set_ylabel("PSI")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
        ax.set_title(f"PSI dos ratings por amostra vs {self.ref_sample}", fontsize=11,
                     fontweight="bold", color="#15324a")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def monotonicity_report(self) -> pd.DataFrame:
        """Verifica se o risco médio é monotônico ao longo dos ratings, por amostra.

        A coluna ``inverte_vs_DES`` lista os ratings cuja ordem de risco se inverte
        em relação à referência (DES) — ex.: ``C↔D`` quando a régua diz C < D na DES
        mas a amostra (p.ex. OOT) traz C > D. ``—`` = nenhuma inversão."""
        rating = self._rating_series()
        labels = self.rating_labels_
        # risco (= nanmean) por (rating, amostra) num único groupby (ver rating_table)
        if self.sample_col is None:
            _risk_overall = self.df.groupby(rating, observed=True)[self.target].mean()

            def _risk_of(lab, a):
                v = _risk_overall.get(lab, np.nan)
                return float(v) if pd.notna(v) else float("nan")
        else:
            _risk = self.df.groupby([rating, self.df[self.sample_col]],
                                    observed=True)[self.target].mean()

            def _risk_of(lab, a):
                v = _risk.get((lab, a), np.nan)
                return float(v) if pd.notna(v) else float("nan")

        # ordem de referência: ratings ordenados pelo risco na DES (crescente)
        ref_risco = {l: _risk_of(l, self.ref_sample) for l in labels}
        ordered = sorted(labels, key=lambda l: (np.inf if pd.isna(ref_risco[l])
                                                 else ref_risco[l]))
        rows = []
        for a in self._samples():
            vals = {l: _risk_of(l, a) for l in labels}
            seq = [vals[l] for l in ordered]           # risco na ordem esperada (DES asc.)
            trend, _ = _trend(seq)                     # tendência (rótulo) via _trend
            # n_inversoes = DESCIDAS ADJACENTES na escada esperada — a MESMA definição de
            # TreeSegmenter.monotonicity_report (antes usava mudanças de sinal do _trend,
            # que dava número diferente para o mesmo conceito).
            descidas = [(ordered[i], ordered[i + 1]) for i in range(len(seq) - 1)
                        if np.isfinite(seq[i]) and np.isfinite(seq[i + 1])
                        and seq[i + 1] < seq[i]]
            n_inv = len(descidas)
            if self.sample_col is not None and a == self.ref_sample:
                inv_txt = "(referência)"
            else:
                pares = _inverted_pairs(ordered, vals)   # diagnóstico complementar (vs DES)
                inv_txt = ", ".join(f"{x}↔{y}" for x, y in pares) if pares else "—"
            rows.append({"amostra": a, "monotonico": n_inv == 0,
                         "tendencia": trend, "n_inversoes": n_inv,
                         "inverte_vs_DES": inv_txt})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # BOOTSTRAP_CI: intervalo de confiança do alvo médio por RATING, via
    #   reamostragem bootstrap na amostra `sample` (default = referência/DES).
    #   Se houver `check_sample` (default = 1ª não-referência, ex. OOT), traz o
    #   alvo dela por rating e verifica a ADERÊNCIA: se o realizado cai dentro
    #   do IC bootstrap da referência (estável) ou fora (acima/abaixo = alerta).
    # ------------------------------------------------------------------
    def bootstrap_ci(self, n_boot: int = 1000, ci: float = 0.95,
                     sample: str | None = None, check_sample: str | None = None,
                     seed: int = 42) -> pd.DataFrame:
        """IC bootstrap do alvo médio por **rating** + aderência em outra amostra.

        Premissa de custo: o score e o rating de cada linha são **fixos** (já
        atribuídos pela régua vigente) — cada réplica reamostra apenas os
        ÍNDICES das linhas do rating e reagrega o alvo, **sem reescorar** o
        modelo. Isso mede a incerteza amostral do realizado por faixa a custo
        baixo (nenhum refit/re-predição por réplica); a incerteza do próprio
        modelo/régua fica fora do escopo.

        Devolve uma linha por rating: ``n``, ``valor_<sample>`` (alvo médio),
        ``ic_low``/``ic_high``/``amplitude`` (IC de ``ci``) e — quando há
        ``check_sample`` — ``valor_<check>``, ``aderente`` e ``status``
        (``dentro`` do IC = estável; ``acima``/``abaixo`` = o alvo deslocou
        além da incerteza amostral)."""
        rating = self._rating_series()
        rng = np.random.default_rng(seed)
        alpha = (1 - ci) / 2
        if self.sample_col is not None:
            if sample is None:
                sample = self.ref_sample
            if check_sample is None:
                nonref = self._nonref_samples()
                check_sample = nonref[0] if nonref else None
        else:
            sample = check_sample = None

        # valores do alvo por rating num ÚNICO groupby por amostra (sem máscara
        # full-length por rótulo — ver notas de performance em rating_table)
        base = pd.DataFrame({"y": self.df[self.target], "r": rating})
        sub = base if sample is None else base[self._frame_mask(sample)]
        vals_by = {lab: g["y"].to_numpy(dtype="float64")
                   for lab, g in sub.groupby("r", observed=True)}
        chk_mean = None
        if check_sample is not None:
            chk_mean = (base[self._frame_mask(check_sample)]
                        .groupby("r", observed=True)["y"].mean())

        rows = []
        for lab in self.rating_labels_:
            vals = vals_by.get(lab, np.empty(0, dtype="float64"))
            vals = vals[~np.isnan(vals)]
            n = len(vals)
            if n >= 2:
                niveis, freq = np.unique(vals, return_counts=True)
                if len(niveis) <= 256:
                    # poucos valores distintos (alvo 0/1 da PD, notas): a média de
                    # uma reamostra com reposição = Σ valor × contagem/n, com as
                    # contagens ~ Multinomial(n, frequências) — MESMA distribuição
                    # da reamostragem por índice, em O(n_boot × níveis) e não
                    # O(n_boot × n) (ratings com centenas de milhares de linhas)
                    cont = rng.multinomial(n, freq / n, size=n_boot)
                    means = cont @ niveis.astype("float64") / n
                else:
                    # bootstrap em BLOCOS de n_boot: limita a matriz de
                    # reamostragem a ~4M elementos (em vez de n_boot×n inteiro de
                    # uma vez — um rating com n=100k estouraria a memória do driver).
                    means = np.empty(n_boot, dtype="float64")
                    passo = max(1, min(n_boot, 4_000_000 // max(n, 1)))
                    feito = 0
                    while feito < n_boot:
                        b = min(passo, n_boot - feito)
                        idx = rng.integers(0, n, size=(b, n))
                        means[feito:feito + b] = vals[idx].mean(axis=1)
                        feito += b
                lo, hi = np.quantile(means, [alpha, 1 - alpha])
                pt = float(vals.mean())
            elif n == 1:
                lo = hi = pt = float(vals[0])
            else:
                lo = hi = pt = np.nan

            row = {
                "rating": lab,
                "n": int(n),
                f"valor_{sample or 'todos'}": round(pt, 4) if not np.isnan(pt) else np.nan,
                "ic_low": round(float(lo), 4) if not np.isnan(lo) else np.nan,
                "ic_high": round(float(hi), 4) if not np.isnan(hi) else np.nan,
                "amplitude": round(float(hi - lo), 4) if not np.isnan(hi) else np.nan,
            }
            if check_sample is not None:
                v_c = chk_mean.get(lab, np.nan)
                v_c = float(v_c) if pd.notna(v_c) else np.nan
                row[f"valor_{check_sample}"] = (round(v_c, 4)
                                                if not np.isnan(v_c) else np.nan)
                if np.isnan(v_c) or np.isnan(lo):
                    row["aderente"] = None
                    row["status"] = "—"
                else:
                    dentro = bool(lo <= v_c <= hi)
                    row["aderente"] = dentro
                    row["status"] = ("dentro" if dentro
                                     else "acima" if v_c > hi else "abaixo")
            rows.append(row)

        out = pd.DataFrame(rows)
        out.attrs.update(sample=sample, check_sample=check_sample, ci=ci, n_boot=n_boot)
        return out

    def plot_bootstrap_forest(self, bc: pd.DataFrame | None = None, n_boot: int = 1000,
                              ci: float = 0.95, sample: str | None = None,
                              check_sample: str | None = None, seed: int = 42,
                              figsize=None, dpi=150, save_path=None, ax=None):
        """Forest plot dos ICs bootstrap por rating (:meth:`bootstrap_ci`): barra
        horizontal = IC do alvo médio na referência; traço vertical = alvo médio
        da referência; ponto = realizado na amostra de comparação (verde dentro
        do IC / vermelho fora). Aceita um ``bc`` já calculado (evita refazer o
        bootstrap) ou os parâmetros para calculá-lo."""
        if bc is None:
            bc = self.bootstrap_ci(n_boot=n_boot, ci=ci, sample=sample,
                                   check_sample=check_sample, seed=seed)
        smp = bc.attrs.get("sample") or "todos"
        chk = bc.attrs.get("check_sample")
        ci_pct = int(round(bc.attrs.get("ci", ci) * 100))
        ref_col, chk_col = f"valor_{smp}", f"valor_{chk}"
        k = len(bc)
        if figsize is None:
            figsize = (8.4, max(2.8, 0.46 * k + 1.4))
        fig, ax = _new_ax(figsize, dpi, ax)
        if k == 0:
            ax.text(0.5, 0.5, "sem ratings para o forest plot", ha="center",
                    va="center", transform=ax.transAxes, color="#889")
            ax.axis("off"); fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            return fig

        ys = np.arange(k)[::-1]                     # 1º rating no topo
        for y0, (_, r) in zip(ys, bc.iterrows()):
            lo, hi = r["ic_low"], r["ic_high"]
            if pd.notna(lo):
                ax.barh(y0, max(float(hi) - float(lo), 1e-12), left=float(lo),
                        height=0.5, color="steelblue", alpha=0.35,
                        edgecolor="#33424f", linewidth=0.6, zorder=2)
                if pd.notna(r[ref_col]):
                    ax.plot([r[ref_col]], [y0], marker="|", ms=15, mew=2.2,
                            color="#15324a", zorder=3)
            # marcador da amostra de comparação só quando há IC para comparar
            # (aderente=None ⇒ status "—": sem CI ou sem realizado — nada a pintar)
            if (chk and chk_col in bc.columns and pd.notna(r.get(chk_col, np.nan))
                    and r.get("aderente") is not None):
                cor = "#1aa64b" if r.get("aderente") else "#d6453e"
                ax.plot([r[chk_col]], [y0], marker="o", ms=6.5, color=cor,
                        markeredgecolor="#33424f", markeredgewidth=0.6, zorder=4)
        ax.set_yticks(ys)
        ax.set_yticklabels([str(v) for v in bc["rating"]], fontsize=9)
        ax.set_ylim(-0.7, k - 0.3)
        ax.set_xlabel(f"{self._risk_word} médio")
        _pct_axis(ax, "x")
        ax.grid(axis="x", alpha=0.15)
        # legenda com artistas proxy (as marcas são desenhadas por rating)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [Patch(facecolor="steelblue", alpha=0.35, edgecolor="#33424f",
                         label=f"IC {ci_pct}% ({smp})"),
                   Line2D([], [], marker="|", ls="", ms=11, mew=2.2,
                          color="#15324a", label=f"média ({smp})")]
        if chk:
            handles += [Line2D([], [], marker="o", ls="", ms=6.5, color="#1aa64b",
                               markeredgecolor="#33424f", label=f"{chk} dentro do IC"),
                        Line2D([], [], marker="o", ls="", ms=6.5, color="#d6453e",
                               markeredgecolor="#33424f", label=f"{chk} fora do IC")]
        ax.legend(handles=handles, fontsize=7.5, loc="best", framealpha=0.9)
        titulo = f"IC {ci_pct}% bootstrap por rating · {smp}" + (f" × {chk}" if chk else "")
        ax.set_title(titulo, fontsize=11, fontweight="bold", color="#15324a")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def _calib_test(self, y, score) -> dict:
        """Teste estatístico de calibração de uma célula (safra ou rating):
        compara o **previsto** (score médio) com o **realizado** (alvo médio).

        * **classificação** — IC de credibilidade de **Jeffreys** da taxa
          observada (posteriori Beta(k+½, n−k+½), dado o nº de eventos e o n da
          célula); o previsto é confrontado com os ICs de 95% e 99%.
        * **regressão** — teste **t pareado** do gap (alvo − score por linha);
          os ICs de 95%/99% do realizado usam o erro-padrão das diferenças
          (previsto fora do IC ⇔ |t| acima do quantil correspondente).

        Semáforo em ``status``: ``ok`` (previsto dentro do IC 95%), ``atencao``
        (fora do 95%, dentro do 99%), ``alerta`` (fora do 99%, ou célula não
        testável). Devolve ``{'ic_low','ic_high','p_valor','status'}`` — o IC
        reportado é o de 95% do realizado; ``p_valor`` é a cauda equilátera da
        posteriori (classificação, aprox.) ou o p bicaudal do teste t."""
        y = np.asarray(y, dtype="float64")
        s = np.asarray(score, dtype="float64")
        ok = np.isfinite(y) & np.isfinite(s)
        y, s = y[ok], s[ok]
        n = int(y.size)
        out = {"ic_low": float("nan"), "ic_high": float("nan"),
               "p_valor": float("nan"), "status": "alerta"}
        if n == 0:
            return out
        prev = float(s.mean())
        if self.task_type == "classification":
            from scipy.stats import beta as beta_dist
            k = float(y.sum())
            lo95, hi95 = _jeffreys_ci(k, n, 0.95)
            lo99, hi99 = _jeffreys_ci(k, n, 0.99)
            post = beta_dist(k + 0.5, n - k + 0.5)
            out["p_valor"] = float(min(1.0, 2.0 * min(post.cdf(prev),
                                                      post.sf(prev))))
        else:
            from scipy.stats import t as t_dist
            if n < 2:
                return out
            real = float(y.mean())
            d = y - s
            se = float(np.std(d, ddof=1)) / float(np.sqrt(n))
            if se == 0.0:                       # gap constante: sem incerteza
                out.update(ic_low=real, ic_high=real,
                           p_valor=1.0 if float(d.mean()) == 0.0 else 0.0,
                           status="ok" if float(d.mean()) == 0.0 else "alerta")
                return out
            t95 = float(t_dist.ppf(0.975, n - 1))
            t99 = float(t_dist.ppf(0.995, n - 1))
            lo95, hi95 = real - t95 * se, real + t95 * se
            lo99, hi99 = real - t99 * se, real + t99 * se
            tstat = float(d.mean()) / se
            out["p_valor"] = float(2.0 * t_dist.sf(abs(tstat), n - 1))
        out["ic_low"], out["ic_high"] = lo95, hi95
        if not np.isfinite(prev):
            return out
        out["status"] = ("ok" if lo95 <= prev <= hi95
                         else "atencao" if lo99 <= prev <= hi99 else "alerta")
        return out

    def backtest(self, time_col=None, sample=None, tol=None) -> pd.DataFrame:
        """Risco previsto (score médio) vs realizado (alvo médio) por safra, com
        **teste estatístico de calibração** por safra.

        ``gap = realizado − previsto`` (mesma convenção do ``TreeSegmenter``):
        gap positivo = risco realizado ACIMA do previsto (sub-provisionamento).

        O ``status`` é um semáforo estatístico (ver :meth:`_calib_test`):

        * **classificação** — IC de credibilidade de **Jeffreys** da taxa
          observada da safra (Beta(k+½, n−k+½)); ``ok`` se o previsto cai dentro
          do IC 95%, ``atencao`` se fora do 95% mas dentro do 99%, ``alerta``
          se fora do 99%. Safra pequena tem IC largo (gap grande pode ser só
          ruído); safra grande tem IC estreito (gap pequeno já é significativo).
        * **regressão** — teste **t pareado** do gap (alvo − score por linha),
          com os mesmos cortes (5%/1%) expressos como IC do realizado.

        Colunas: ``ic_low``/``ic_high`` (IC 95% do realizado) e ``p_valor``,
        além das já existentes. **Compat**: informe ``tol`` (número) para voltar
        ao critério antigo de tolerância fixa — ``status`` binário ok/alerta com
        ``|gap| > tol`` (os defaults antigos eram 0,03/0,10 por task_type);
        ``tol=None`` (default) usa o teste estatístico."""
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col.")
        if self.score_ is None:
            raise RuntimeError("Gere o score antes (fit / set_model).")
        base = self._frame(sample) if sample else self.df
        if time_col not in base.columns:
            raise ValueError(f"Coluna de tempo '{time_col}' não existe no DataFrame.")
        sc = self.score_.reindex(base.index)
        safra = pd.to_datetime(base[time_col], errors="coerce").dt.to_period("M")
        rows = []
        for per, g in base.groupby(safra):
            sg = sc.reindex(g.index)
            prev = float(sg.mean(skipna=True))
            real = self._risco(g[self.target])
            gap = real - prev if (np.isfinite(prev) and np.isfinite(real)) else np.nan
            t = self._calib_test(g[self.target], sg)
            if tol is not None:               # fallback/compat: tolerância fixa
                status = "ok" if (np.isfinite(gap) and abs(gap) <= tol) else "alerta"
            else:
                status = t["status"]
            rows.append({"safra": str(per), "n": len(g),
                         "previsto_medio": round(prev, 4) if np.isfinite(prev) else np.nan,
                         "realizado_medio": round(real, 4) if np.isfinite(real) else np.nan,
                         "gap": round(gap, 4) if np.isfinite(gap) else np.nan,
                         "ic_low": (round(t["ic_low"], 4)
                                    if np.isfinite(t["ic_low"]) else np.nan),
                         "ic_high": (round(t["ic_high"], 4)
                                     if np.isfinite(t["ic_high"]) else np.nan),
                         "p_valor": (round(t["p_valor"], 4)
                                     if np.isfinite(t["p_valor"]) else np.nan),
                         "status": status})
        return pd.DataFrame(rows).sort_values("safra").reset_index(drop=True)

    def hosmer_lemeshow(self, sample=None, n_groups=10) -> dict:
        """Teste de **Hosmer–Lemeshow** global da calibração (classificação):
        agrupa as linhas em ``n_groups`` faixas de score (decis por padrão) e
        compara eventos observados vs esperados por faixa:
        ``χ² = Σ (O − E)² / (E·(1 − p̄))``, com ``df = g − 2``. p-valor alto =
        sem evidência de má calibração global; p-valor baixo = o previsto se
        afasta do realizado em alguma região do score.

        ``sample=None`` usa todas as linhas (passe uma amostra para restringir,
        ex.: a fora-do-tempo). Devolve ``{'statistic', 'p_value', 'df',
        'n_groups', 'table'}`` — a ``table`` tem, por faixa, ``n``, eventos
        observados/esperados, taxa observada, score médio e a contribuição ao
        χ². Erro na regressão (use o teste t do :meth:`backtest`)."""
        if self.task_type != "classification":
            raise ValueError("hosmer_lemeshow avalia calibração de alvo binário "
                             "(classificação); na regressão use o backtest "
                             "(teste t do gap).")
        if self.score_ is None:
            raise RuntimeError("Gere o score antes (fit / set_model).")
        from scipy.stats import chi2
        base = self._frame(sample) if sample else self.df
        sc = self.score_.reindex(base.index)
        y = base[self.target]
        ok = sc.notna() & y.notna()
        sc, y = sc[ok], y[ok].astype("float64")
        if len(sc) == 0:
            raise ValueError("Sem linhas válidas (score e alvo) para o teste.")
        grp = pd.qcut(sc, q=int(n_groups), duplicates="drop")
        rows, stat = [], 0.0
        for faixa, idx in sc.groupby(grp, observed=True).groups.items():
            s_g = sc.loc[idx].to_numpy(dtype="float64")
            y_g = y.loc[idx].to_numpy(dtype="float64")
            n = int(s_g.size)
            obs = float(y_g.sum())
            esp = float(s_g.sum())
            p_bar = esp / n if n else float("nan")
            den = esp * (1.0 - p_bar)
            contrib = ((obs - esp) ** 2 / den) if den > 0 else 0.0
            stat += contrib
            rows.append({"faixa": str(faixa), "n": n,
                         "eventos_obs": int(round(obs)),
                         "eventos_esp": round(esp, 1),
                         "taxa_obs": round(obs / n, 4) if n else np.nan,
                         "score_medio": round(float(s_g.mean()), 4),
                         "contrib": round(contrib, 3)})
        g_eff = len(rows)
        df_hl = max(g_eff - 2, 1)
        return {"statistic": float(round(stat, 4)),
                "p_value": float(chi2.sf(stat, df_hl)),
                "df": int(df_hl), "n_groups": int(g_eff),
                "table": pd.DataFrame(rows)}

    def plot_backtest(self, sample=None, tolerancia=0.2, time_col=None,
                      figsize=(9.6, 4.4), dpi=150, save_path=None, ax=None):
        """Backtest gráfico: previsto × realizado por safra (duas linhas) com uma
        **banda visual** sombreada (``previsto ± tolerancia·previsto``; ``0.2`` =
        ±20% — só referência visual). Os **marcadores** destacam as safras pelo
        semáforo estatístico da tabela :meth:`backtest` (âmbar = previsto fora do
        IC 95% do realizado; vermelho = fora do IC 99%), para o gráfico e a
        tabela nunca discordarem. Funciona nos dois ``task_type`` (unidade do
        alvo)."""
        bt = self.backtest(time_col=time_col, sample=sample)
        fig, ax = _new_ax(figsize, dpi, ax)
        if bt.empty:
            ax.text(0.5, 0.5, "sem dados por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        x = list(range(len(bt)))
        prev = bt["previsto_medio"].to_numpy(dtype="float64")
        real = bt["realizado_medio"].to_numpy(dtype="float64")
        lo = prev * (1.0 - float(tolerancia))
        hi = prev * (1.0 + float(tolerancia))
        ax.fill_between(x, lo, hi, color="#8aa4bf", alpha=0.25,
                        label=f"banda visual ±{100 * tolerancia:.0f}%")
        ax.plot(x, prev, color="#15324a", lw=2.2, marker="o", ms=4.5,
                label="previsto (médio)")
        ax.plot(x, real, color="#1aa64b", lw=2.0, marker="o", ms=4.5,
                label="realizado (médio)")
        # marcadores destacados = semáforo estatístico da tabela backtest (teste de
        # calibração por safra), NÃO a banda relativa visual — assim os pontos
        # âmbar/vermelhos batem com o 'status' da tabela.
        finito = np.isfinite(real) & np.isfinite(prev)
        status = bt["status"].to_numpy()
        atencao = (status == "atencao") & finito
        alerta = (status == "alerta") & finito
        if atencao.any():
            xs = [x0 for x0, f in zip(x, atencao) if f]
            ax.plot(xs, real[atencao], "o", ms=9, mfc="none", mec="#caa000", mew=2.0,
                    label="atenção (fora do IC 95%)")
        if alerta.any():
            xs = [x0 for x0, f in zip(x, alerta) if f]
            ax.plot(xs, real[alerta], "o", ms=9, mfc="none", mec="#d6453e", mew=2.0,
                    label="alerta (fora do IC 99%)")
        labels = _fmt_safras(bt["safra"])
        step = max(1, len(bt) // 18)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("safra")
        ax.set_ylabel(self._risk_word if self.task_type == "classification" else "alvo médio")
        _pct_axis(ax, "y")                                  # unidade do alvo, em %
        ax.set_title(f"Backtest · previsto × realizado por safra · "
                     f"{sample or 'todas as amostras'}", fontsize=11,
                     fontweight="bold", color="#15324a")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # Export / predict / persistência
    # ------------------------------------------------------------------
    def assign(self, col_score="score", col_rating="rating") -> pd.DataFrame:
        """Cópia do df com o score e o rating de cada linha."""
        out = self.df.copy()
        if self.score_ is not None:
            out[col_score] = self.score_ * self.score_scale     # escala de negócio
        if self.rating_ is not None:
            out[col_rating] = self.rating_
        return out

    def rating_ruler(self, sample=None, col_rating="rating",
                     col_value="valor_previsto") -> pd.DataFrame:
        """Régua **rating → valor previsto do alvo**: o valor representativo de cada
        rating, definido como a média do alvo observado na amostra de referência
        (``ref_sample``/DES por padrão). É o "valor previsto daquele rating", usado
        para escorar uma base. Requer :meth:`build_ratings` já chamado.

        Devolve um DataFrame ordenado pelos rótulos, com ``col_rating``, ``n`` e
        ``col_value``."""
        # régua pré-computada e injetada (escoragem distribuída em executores Spark,
        # onde ``self.df`` foi esvaziado): devolve-a direto, sem tocar no df.
        override = getattr(self, "_ruler_override", None)
        if override is not None:
            return override
        rating = self._rating_series()
        sample = sample or self.ref_sample
        if self.sample_col is None:
            mask = pd.Series(True, index=self.df.index)
        else:
            mask = self.df[self.sample_col] == sample
        rows = []
        for lab in self.rating_labels_:
            m = (rating == lab) & mask
            rows.append({col_rating: lab, "n": int(m.sum()),
                         col_value: round(self._risco(self.df.loc[m, self.target]), 6)})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # TO_SQL: a régua de ratings como CASE WHEN sobre o score materializado
    # ------------------------------------------------------------------
    def _rating_score_cuts(self) -> tuple:
        """Cortes de score da régua vigente + a **convenção de borda** usada.

        Devolve ``(cuts, borda)``: ``cuts`` são os limites FINITOS que separam as
        faixas de score (crescentes, na escala CRUA do score) e ``borda`` diz de
        que lado o valor exatamente no limite cai, conforme a estratégia ajustada:

        * cortes por quantil/decis/percentil/manuais e binning ótimo usam
          ``searchsorted(..., side='right')`` ⇒ o valor no limite entra na faixa de
          CIMA ⇒ ``'esquerda'`` (faixa fechada à esquerda: ``lo <= score < hi``);
        * a estratégia de árvore compara ``x <= limiar`` ⇒ o valor no limite fica
          na faixa de BAIXO ⇒ ``'direita'`` (``lo < score <= hi``).
        """
        st = self.rating_strategy
        if st is None or not self.rating_labels_:
            raise RuntimeError("Gere os ratings antes (build_ratings).")
        thr = getattr(st, "thresholds_", None)          # estratégia de árvore
        if thr is not None:
            cuts = np.asarray(thr, dtype="float64")
            return np.unique(cuts[np.isfinite(cuts)]), "direita"
        cortes = getattr(st, "splits_", None)           # binning ótimo
        if cortes is None:
            cortes = getattr(st, "edges_", None)        # quantil/decis/manuais
        if cortes is None:
            raise RuntimeError(
                f"A estratégia de rating {type(st).__name__!r} não expõe cortes de "
                f"score — não dá para reproduzi-la como CASE WHEN.")
        cuts = np.asarray(cortes, dtype="float64")
        return np.unique(cuts[np.isfinite(cuts)]), "esquerda"

    def _rating_score_ranges(self) -> list:
        """Faixas de score de cada rating, na ordem de ``rating_labels_``:
        ``[{'rating': 'A', 'faixas': [(lo, hi), …]}, …]`` — ``None`` nas pontas
        abertas. Um rating pode ter MAIS de uma faixa (estratégia de árvore, onde
        intervalos de score não contíguos podem receber o mesmo rótulo).

        O rótulo de cada intervalo vem da própria estratégia (um ponto interno é
        reclassificado por ela), então a régua em SQL nunca diverge da aplicada em
        Python — só as fronteiras são lidas dos cortes."""
        st = self.rating_strategy
        cuts, _borda = self._rating_score_cuts()
        k = len(cuts)
        por_rating = {lab: [] for lab in self.rating_labels_}
        for i in range(k + 1):
            lo = float(cuts[i - 1]) if i > 0 else None
            hi = float(cuts[i]) if i < k else None
            # ponto REPRESENTATIVO no MEIO do intervalo (longe das bordas: a
            # estratégia de árvore compara em float32 e um valor exatamente no
            # limiar poderia ser reclassificado do outro lado)
            if lo is None and hi is None:
                rep = 0.0
            elif lo is None:
                rep = hi - max(1.0, abs(hi))
            elif hi is None:
                rep = lo + max(1.0, abs(lo))
            else:
                rep = 0.5 * (lo + hi)
            raw = int(np.asarray(st._raw_groups(np.array([rep], dtype="float64")))[0])
            lab = st.raw_to_label_.get(raw)
            if lab in por_rating:
                por_rating[lab].append((lo, hi))
        out = []
        for lab in self.rating_labels_:
            faixas = []
            for lo, hi in por_rating[lab]:
                if faixas and lo is not None and faixas[-1][1] == lo:
                    faixas[-1] = (faixas[-1][0], hi)     # intervalos colados: une
                else:
                    faixas.append((lo, hi))
            out.append({"rating": lab, "faixas": faixas})
        return out

    def to_sql(self, table: str = "minha_tabela", score_col: str = "score",
               col_rating: str = "rating", col_value=None, ruler_sample=None,
               score_scale=None) -> str:
        """Gera SQL ANSI com ``CASE WHEN`` que reproduz a **régua de ratings** sobre
        uma coluna de score JÁ materializada. Pronto p/ copiar.

        ``table`` é a tabela/CTE de origem e ``score_col`` a coluna de score dela —
        esperada na **escala de negócio** (0–``score_scale``, i.e. 0–1000 por
        padrão), a mesma devolvida por :meth:`predict`/:meth:`score_table`/
        :meth:`apply_spark`. Passe ``score_scale=1`` se a coluna guardar o score
        CRU (0–1). Cada rating de :meth:`rating_ruler` vira um ramo do CASE, na
        ordem da régua, com o volume e o valor previsto do alvo no comentário.

        **Fronteiras e convenção de borda** (ver :meth:`_rating_score_cuts`): são
        os cortes EXATOS aprendidos pela estratégia de rating, escritos com toda a
        precisão. O valor exatamente no limite segue o mesmo lado que em Python —
        ``score >= lo AND score < hi`` para cortes por quantil/decis/percentil/
        manuais e binning ótimo; ``score > lo AND score <= hi`` na estratégia de
        árvore. A convenção usada sai como comentário no cabeçalho do SQL.

        ``col_value`` (ex.: ``'valor_previsto'``) anexa uma segunda coluna com o
        valor previsto do alvo daquele rating — a régua de :meth:`rating_ruler`
        calibrada em ``ruler_sample`` (referência/DES por padrão).

        Linhas com ``score`` NULL caem no ``ELSE NULL`` (rating nulo); qualquer
        score não-nulo cai em alguma faixa (as pontas são abertas)."""
        ranges = self._rating_score_ranges()
        esc = float(self.score_scale if score_scale is None else score_scale)
        _cuts, borda = self._rating_score_cuts()
        ge, lt = (">=", "<") if borda == "esquerda" else (">", "<=")
        ruler = self.rating_ruler(sample=ruler_sample)
        info = {r["rating"]: (int(r["n"]), float(r["valor_previsto"]))
                for _, r in ruler.iterrows()}

        def _n(v):                       # literal numérico com precisão de ida-e-volta
            return repr(float(v))

        def cond(faixas):
            partes = []
            for lo, hi in faixas:
                sub = []
                if lo is not None:
                    sub.append(f"{score_col} {ge} {_n(lo * esc)}")
                if hi is not None:
                    sub.append(f"{score_col} {lt} {_n(hi * esc)}")
                partes.append(" AND ".join(sub) if sub
                              else f"{score_col} IS NOT NULL")
            if len(partes) == 1:
                return partes[0]
            return " OR ".join(f"({p})" for p in partes)

        borda_txt = (f"[lo, hi) — {score_col} >= lo AND {score_col} < hi (valor no "
                     f"limite entra na faixa de CIMA)" if borda == "esquerda"
                     else f"(lo, hi] — {score_col} > lo AND {score_col} <= hi (valor no "
                          f"limite fica na faixa de BAIXO)")
        _amostra = ruler_sample or self.ref_sample
        cab = [f"-- Régua de ratings ({self.task_type}) gerada por ModelSegmenter "
               f"· {len(ranges)} faixas",
               f"-- método: {self.rating_config.get('method', '—')} · coluna de score "
               f"'{score_col}' na escala 0–{_fmt(esc)} (score cru × {_fmt(esc)})",
               f"-- convenção de borda: {borda_txt}",
               f"-- score NULL ⇒ {col_rating} NULL (ELSE)"]
        if col_value is not None:
            cab.append(f"-- {col_value}: alvo médio do rating na amostra '{_amostra}' "
                       f"(régua de rating_ruler)")

        def case(valfn, alias, comentar=True):
            linhas = ["  CASE"]
            for r in ranges:
                n_r, v_r = info.get(r["rating"], (0, float("nan")))
                com = f"  -- n={n_r} · valor previsto {v_r:.6f}" if comentar else ""
                linhas.append(f"    WHEN {cond(r['faixas'])} THEN "
                              f"{valfn(r['rating'])}{com}")
            linhas.append("    ELSE NULL")
            linhas.append(f"  END AS {alias}")
            return "\n".join(linhas)

        def _q(v):                       # literal de texto com escape de aspas
            return "'" + str(v).replace("'", "''") + "'"

        corpo = [case(_q, col_rating)]
        if col_value is not None:
            corpo[-1] += ","
            corpo.append(case(lambda lab: _n(info.get(lab, (0, float("nan")))[1]),
                              col_value, comentar=False))
        return "\n".join(cab + ["SELECT", "  *,"] + corpo + [f"FROM {table};"])

    def predict(self, X: pd.DataFrame, col_score="score", col_rating="rating",
                col_value=None, ruler_sample=None, col_reasons=None,
                reasons_top_n=3) -> pd.DataFrame:
        """Aplica modelo (+rating) a novos dados. Devolve score e rating por linha.

        Se ``col_value`` for informado (ex.: ``"valor_previsto"``), anexa também o
        **valor previsto do alvo daquele rating** — a régua de :meth:`rating_ruler`
        calibrada na amostra ``ruler_sample`` (DES por padrão), i.e. o alvo
        previsto por rating.

        Se ``col_reasons`` for informado (ex.: ``"motivo"``), anexa as colunas
        ``motivo_1..motivo_N`` (``N = reasons_top_n``) com os principais motivos
        (*reason codes*) de cada linha via :meth:`reason_codes` — a variável de
        maior |contribuição SHAP| e o sinal. Requer o pacote opcional ``shap`` e
        custa um cálculo de SHAP sobre TODO o ``X`` (caminho pandas; a escoragem
        Spark distribuída não anexa motivos — ver :meth:`apply_spark`)."""
        if self.model is None:
            raise RuntimeError("Ajuste/defina o modelo antes (fit / set_model).")
        X = self._apply_derived(X)        # recria variáveis derivadas a partir da origem
        sc = self._compute_score(X)                              # predição CRUA (em blocos)
        # o negócio recebe o score na escala 0–score_scale (0–1000); os ratings,
        # porém, seguem a estratégia salva na escala CRUA (bins definidos no fit).
        out = pd.DataFrame({col_score: sc * self.score_scale}, index=X.index)
        if self.rating_strategy is not None:
            wf = pd.DataFrame({"score": sc}, index=X.index)      # rating na escala crua
            cfg = self._make_cfg("_amostra")
            wf["_amostra"] = self.ref_sample
            out[col_rating] = self.rating_strategy.transform(wf, cfg).values
            if col_value is not None:
                ruler = self.rating_ruler(sample=ruler_sample, col_rating=col_rating,
                                          col_value=col_value)
                mapping = dict(zip(ruler[col_rating], ruler[col_value]))
                out[col_value] = out[col_rating].map(mapping)
        if col_reasons:                   # motivos SHAP por linha (opcional)
            rc = self.reason_codes(X=X, top_n=int(reasons_top_n),
                                   prefix=f"{col_reasons}_")
            out = out.join(rc)
        return out

    def _labels_from_bins(self, col, bins, feature, others="(outros)") -> pd.Series:
        """Rótulo do bin (faixa/grupo) de cada valor de ``col`` segundo ``bins``.
        Valores fora de todos os bins (categoria nova) e não-nulos viram ``others``."""
        col = pd.Series(col)
        labels = pd.Series([pd.NA] * len(col), index=col.index, dtype="object")
        assigned = np.zeros(len(col), dtype=bool)
        for b, m in zip(bins, _bin_masks(col, bins)):
            m = m & ~assigned
            labels.iloc[m] = self._bin_label(feature, b)
            assigned |= m
        miss = (~assigned) & col.notna().to_numpy()
        if miss.any():
            labels.iloc[miss] = others
        return labels

    def recreate_categories(self, X, suffix="_faixa", features=None) -> pd.DataFrame:
        """Recria, para cada variável, a **faixa/grupo** (categoria) a que cada linha
        pertence — com os mesmos bins do modelo (manuais ou ótimos, ajustados na
        referência). Numéricas viram faixas ``(lo, hi]``; categóricas viram grupos
        ``{A, B}``; faltantes ``(faltante)``. Devolve só as colunas recriadas
        (``<feature><suffix>``). É como a binagem/WoE "vê" cada linha ao escorar.
        Pula variáveis já derivadas (criadas via :meth:`create_categorical`)."""
        feats = (list(features) if features is not None
                 else (list(self.model_features) or self.selected_features()
                       or list(self.candidates)))
        X = pd.DataFrame(X)
        out = {}
        for f in feats:
            if f not in X.columns or self.var_meta.get(f, {}).get("derived_from"):
                continue
            try:
                bins, _kind = self._resolve_bins(f, sample=self.ref_sample)
            except Exception:
                continue
            if not bins:
                continue
            out[f"{f}{suffix}"] = self._labels_from_bins(X[f], bins, f)
        return pd.DataFrame(out, index=X.index)

    # ---- categorização como SQL ----
    def _sql_cond(self, col, b, is_bool=False) -> str:
        """Condição SQL de UM bin sobre a coluna ``col`` — mesma regra das máscaras
        Python: numérica ``(lo, hi]``, grupo categórico por igualdade de texto,
        faltante ``IS NULL`` e ``include_na`` (faltantes alocados na faixa)."""
        def _n(v):
            return repr(float(v))
        if b["kind"] == "na":
            return f"{col} IS NULL"
        if b["kind"] == "num":
            partes = []
            if np.isfinite(b["lo"]):
                partes.append(f"{col} > {_n(b['lo'])}")
            if np.isfinite(b["hi"]):
                partes.append(f"{col} <= {_n(b['hi'])}")
            cond = " AND ".join(partes) if partes else f"{col} IS NOT NULL"
        elif is_bool:
            vals = sorted({c for c in b["cats"] if c in ("True", "False")})
            cond = (f"{col} = {vals[0].upper()}" if len(vals) == 1
                    else f"{col} IS NOT NULL")
        else:
            lits = ", ".join("'" + str(c).replace("'", "''") + "'" for c in b["cats"])
            cond = f"{col} IN ({lits})"
        if b.get("include_na"):
            cond = f"({cond} OR {col} IS NULL)"
        return cond

    def categorization_sql(self, features=None, table: str = "minha_tabela",
                           woe=None, suffix_faixa: str = "_faixa",
                           suffix_woe: str = "_woe") -> str:
        """Gera SQL (``CASE WHEN``) que reproduz a **categorização das variáveis**
        — faixas numéricas, grupos categóricos, faltantes (inclusive alocados numa
        faixa), dummies de scorecard e variáveis derivadas — para rodar direto no
        banco/Databricks. Pronto p/ copiar.

        Por variável sai:

        * ``<var><suffix_faixa>``: rótulo da faixa (``(lo, hi]``, ``{A, B}``,
          ``(faltante)``; valor fora das faixas = ``'(outros)'``, NULL sem faixa
          própria = NULL — igual a :meth:`recreate_categories`);
        * ``<var><suffix_woe>``: o WoE da faixa (classificação) ou o risco médio da
          faixa (regressão), quando ``woe=True`` (padrão: se o modelo usa
          ``transform='woe'``) e a variável não é dummy de scorecard;
        * dummies de scorecard: uma coluna 0/1 por faixa (``<var>=<faixa>`` do
          modelo, aqui ``<var>__d<k>`` — dois ``_`` p/ não colidir com as variáveis
          de :meth:`create_scorecard_dummies`), sem a pior faixa (referência);
        * variáveis derivadas (:meth:`create_categorical` /
          :meth:`create_scorecard_dummies`): recriadas a partir da variável de origem.

        ``features`` padrão: as variáveis do modelo (ou as selecionadas, antes do
        treino). As faixas são as da referência (manuais ou ótimas), com os
        valores de WoE/risco aprendidos nela."""
        feats = (list(features) if features is not None
                 else (list(self.model_features) or self.selected_features()
                       or list(self.candidates)))
        if woe is None:
            woe = (self.feature_transform == "woe" and self.model is not None)
        dums = set(self.scorecard_dummy_features(feats))
        linhas, avisos = [], []
        for f in feats:
            meta = self.var_meta.get(f, {})
            src = meta.get("derived_from")
            if src:                                          # derivada → sai da origem
                col = src
                bins = meta.get("derived_bins") or []
                is_bool = src in self.df.columns and pd.api.types.is_bool_dtype(self.df[src])
                if meta.get("derived_dummy"):
                    expr = f"CASE WHEN {self._sql_cond(col, bins[0], is_bool)} THEN 1 ELSE 0 END"
                    linhas.append(f"  -- {self.label(f)} (dummy de '{src}'; referência = "
                                  f"{meta.get('dummy_ref', '—')})")
                    linhas.append(f"  {expr} AS {f},")
                else:
                    expr = self._sql_label_expr(col, bins, is_bool)
                    linhas += self._sql_case_labels(f, col, bins, f, is_bool,
                                                    comentario=f"derivada de '{src}'")
                if woe and f in (self.model_features or []):
                    # o modelo usa o WoE da derivada: sai da expressão sobre a origem
                    linhas += self._sql_case_woe(f, f"({expr})", False, f"{f}{suffix_woe}")
                continue
            if f not in self.df.columns:
                avisos.append(f"-- '{f}' ignorada: não está no DataFrame.")
                continue
            bins, _kind = self._resolve_bins(f, sample=self.ref_sample)
            if not bins:
                avisos.append(f"-- '{f}' ignorada: sem faixas (binning não separou níveis).")
                continue
            is_bool = pd.api.types.is_bool_dtype(self.df[f])
            origem = "manual" if self.manual_bins(f) else "binning ótimo"
            linhas += self._sql_case_labels(f, f, bins, f"{f}{suffix_faixa}", is_bool,
                                            comentario=origem)
            if f in dums:
                spec = self._dummy_spec(f)
                linhas.append(f"  -- dummies de scorecard de {self.label(f)} · referência "
                              f"(omitida, 0 pts) = {spec['labels'][spec['ref']]}")
                for k, (b, lbl) in enumerate(zip(spec["bins"], spec["labels"])):
                    if k == spec["ref"]:
                        continue
                    linhas.append(f"  CASE WHEN {self._sql_cond(f, b, is_bool)} THEN 1 "
                                  f"ELSE 0 END AS {f}__d{k + 1},  -- {lbl}")
            elif woe:
                linhas += self._sql_case_woe(f, f, is_bool, f"{f}{suffix_woe}")
        # a última expressão do SELECT não leva vírgula
        corpo = self._sql_tira_virgula_final("\n".join(linhas).rstrip())
        cab = [f"-- Categorização das variáveis ({self.task_type}) gerada por ModelSegmenter",
               f"-- faixas ajustadas na amostra '{self.ref_sample}' · numéricas (lo, hi] · "
               "a 1ª faixa que casa vence (mesma regra do Python)",
               "-- valor fora das faixas: '(outros)' no rótulo, WoE neutro/risco médio no "
               "WoE, 0 nas dummies (= referência)"]
        return "\n".join(cab + avisos + ["SELECT", "  *,", corpo, f"FROM {table};"])

    def _sql_case_woe(self, f, col, is_bool, alias) -> list:
        """``CASE`` do WoE (classificação) / risco médio (regressão) da faixa de
        ``f`` sobre a expressão ``col`` — os mesmos valores do :class:`WoeBinEncoder`."""
        enc = self._bin_encoding(f)
        nome = "WoE" if self.task_type == "classification" else "risco médio"
        linhas = [f"  CASE  -- {nome} da faixa ({self.label(f)})"]
        for b, v in enc["bins"]:
            linhas.append(f"    WHEN {self._sql_cond(col, b, is_bool)} THEN {v!r}"
                          f"  -- {self._bin_label(f, b)}")
        linhas.append(f"    ELSE {float(enc['fallback'])!r}  -- fora das faixas")
        linhas.append(f"  END AS {alias},")
        return linhas

    def _sql_label_expr(self, col, bins, is_bool) -> str:
        """Expressão (uma linha) do rótulo da faixa — p/ embutir em outro CASE."""
        whens = " ".join(
            f"WHEN {self._sql_cond(col, b, is_bool)} THEN "
            f"'{self._bin_label(col, b).replace(chr(39), chr(39) * 2)}'" for b in bins)
        return f"CASE {whens} WHEN {col} IS NOT NULL THEN '(outros)' ELSE NULL END"

    # ---- fórmula da logística como SQL ----
    def _sql_valor(self, f) -> str:
        """Expressão SQL do valor CRU da variável ``f`` do modelo: a coluna, ou —
        para variável derivada — a recriação a partir da coluna de origem."""
        meta = self.var_meta.get(f, {})
        src = meta.get("derived_from")
        if not src:
            return f
        bins = meta.get("derived_bins") or []
        is_bool = src in self.df.columns and pd.api.types.is_bool_dtype(self.df[src])
        if meta.get("derived_dummy"):
            return f"(CASE WHEN {self._sql_cond(src, bins[0], is_bool)} THEN 1 ELSE 0 END)"
        return f"({self._sql_label_expr(src, bins, is_bool)})"

    @staticmethod
    def _sql_lit(v) -> str:
        """Literal SQL de um valor Python (texto com aspas escapadas, bool, número)."""
        if isinstance(v, (bool, np.bool_)):
            return "TRUE" if v else "FALSE"
        if isinstance(v, (int, float, np.integer, np.floating)):
            return repr(float(v))
        return "'" + str(v).replace("'", "''") + "'"

    def _sql_termos_logit(self) -> dict:
        """``{nome_da_coluna_do_desenho: expressão SQL}`` para cada coluna que o
        pré-processador ajustado entrega ao estimador — mesma transformação do
        pipeline (imputação, one-hot, WoE, dummies de scorecard, derivadas)."""
        pre = self.model.named_steps.get("pre")
        termos: dict = {}

        def _cond(f, b):
            col = self._sql_valor(f)
            is_bool = (not self.var_meta.get(f, {}).get("derived_from")
                       and pd.api.types.is_bool_dtype(self.df[f]))
            return self._sql_cond(col, b, is_bool)

        def _woe(enc_obj, prefixo):
            for f in enc_obj.features or []:
                enc = enc_obj.encodings[f]
                whens = " ".join(f"WHEN {_cond(f, b)} THEN {float(v)!r}" for b, v in enc["bins"])
                pref = (enc_obj.prefixes or {}).get(f, enc_obj.name_prefix)
                termos[f"{prefixo}{pref}({f})"] = (
                    f"(CASE {whens} ELSE {float(enc['fallback'])!r} END)")

        def _dum(enc_obj, prefixo):
            for f in enc_obj.features or []:
                sp = enc_obj.specs[f]
                for i, (b, lbl) in enumerate(zip(sp["bins"], sp["labels"])):
                    if i != sp["ref"]:
                        termos[f"{prefixo}{f}={lbl}"] = (
                            f"(CASE WHEN {_cond(f, b)} THEN 1 ELSE 0 END)")

        if isinstance(pre, WoeBinEncoder):
            _woe(pre, "")
            return termos
        for nome, trans, cols in getattr(pre, "transformers_", []):
            if nome == "num":
                for col, med in zip(cols, trans.statistics_):
                    if np.isfinite(med):
                        termos[f"num__{col}"] = f"COALESCE({self._sql_valor(col)}, {float(med)!r})"
            elif nome == "cat":
                imp, ohe = trans.named_steps["imp"], trans.named_steps["ohe"]
                todos = iter(ohe.get_feature_names_out(list(cols)))   # na ordem col × cat
                for col, moda, cats in zip(cols, imp.statistics_, ohe.categories_):
                    v = self._sql_valor(col)
                    # None em coluna de texto NÃO é imputado (o SimpleImputer só vê
                    # NaN): vira categoria própria → NULL casa com ela; sem essa
                    # categoria, o NULL recebe a moda (como no treino)
                    tem_none = any(c is None for c in cats)
                    base = v if tem_none else f"COALESCE({v}, {self._sql_lit(moda)})"
                    for c in cats:
                        cond = f"{v} IS NULL" if c is None else f"{base} = {self._sql_lit(c)}"
                        termos[f"cat__{next(todos)}"] = f"(CASE WHEN {cond} THEN 1 ELSE 0 END)"
            elif nome == "woe":
                _woe(trans, "woe__")
            elif nome == "dum":
                _dum(trans, "dum__")
        return termos

    def logit_sql(self, table: str = "minha_tabela", col_logit: str = "logit",
                  col_prob: str = "probabilidade", col_score: str = "score") -> str:
        """Gera SQL com a **fórmula da regressão logística** ajustada, sobre as
        colunas CRUAS da tabela: o logito (``col_logit`` = intercepto + Σ coef ×
        termo), a **probabilidade** (``1/(1+e^-z)``, com a camada de calibração
        vigente, se houver) e o **score** na escala de negócio (probabilidade ×
        ``score_scale`` — 0 a 1000 por padrão). Cada termo reproduz o
        pré-processamento do pipeline: imputação (mediana/moda), one-hot, WoE da
        faixa, dummies de scorecard e variáveis derivadas.

        Só para ``algorithm='logistica'`` (classificação, sem Two-Stage)."""
        if self.model is None or self.algorithm != "logistica" or self.two_stage:
            raise ValueError("A fórmula em SQL exige um modelo de regressão logística "
                             "treinado (algorithm='logistica').")
        est = self.model.named_steps["est"]
        pre = self.model.named_steps.get("pre")
        nomes = list(pre.get_feature_names_out())
        coef = np.ravel(np.asarray(est.coef_, dtype="float64"))
        intercepto = float(np.ravel(np.asarray(est.intercept_))[0])
        termos = self._sql_termos_logit()
        faltam = [n for n in nomes if n not in termos]
        if faltam:
            raise ValueError(f"Termos sem tradução para SQL: {faltam[:5]}")
        linhas = [f"    {intercepto!r}  -- intercepto"]
        for nm, c in zip(nomes, coef):
            linhas.append(f"    + ({float(c)!r}) * {termos[nm]}"
                          f"  -- {self._display_feature_name(nm)}")
        esc = float(self.score_scale)
        cal = self.calibration_ or {}
        metodo, par = cal.get("method"), cal.get("params") or {}
        z = col_logit
        if metodo == "intercept":
            prob = f"1.0 / (1.0 + EXP(-({z} + {float(par['delta'])!r})))"
        elif metodo == "platt":
            prob = (f"1.0 / (1.0 + EXP(-({float(par['a'])!r} * {z} + "
                    f"{float(par['b'])!r})))")
        else:
            prob = f"1.0 / (1.0 + EXP(-{z}))"
        cab = [f"-- Fórmula da regressão logística gerada por ModelSegmenter · "
               f"{len(nomes)} termos",
               f"-- {col_logit} = intercepto + Σ coef × termo · {col_prob} = 1 / (1 + e^-"
               f"{col_logit})" + (f" (calibração '{metodo}')" if metodo else ""),
               f"-- {col_score} = {col_prob} × {_fmt(esc)} (escala de negócio, 0–{_fmt(esc)})",
               "-- termos: imputação do treino (mediana/moda), one-hot, WoE da faixa e "
               "dummies de scorecard sobre as colunas CRUAS"]
        if metodo == "isotonic":
            xs = [float(v) for v in par["x"]]
            ys = [float(v) for v in par["y"]]
            p0 = f"(1.0 / (1.0 + EXP(-{z})))"
            partes = [f"WHEN {p0} <= {xs[0]!r} THEN {ys[0]!r}"]
            for i in range(len(xs) - 1):
                x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
                if x1 == x0:
                    continue
                partes.append(f"WHEN {p0} <= {x1!r} THEN {y0!r} + ({p0} - {x0!r}) * "
                              f"{(y1 - y0) / (x1 - x0)!r}")
            partes.append(f"ELSE {ys[-1]!r}")
            prob = "CASE " + " ".join(partes) + " END"
            cab.append("-- calibração isotônica: interpolação linear entre os degraus")
        return "\n".join(cab + [
            "WITH _ygg_logit AS (",
            "  SELECT",
            "    *,",
            "\n".join(linhas) + f"\n    AS {col_logit}",
            f"  FROM {table}",
            "), _ygg_prob AS (",
            f"  SELECT *, {prob} AS {col_prob} FROM _ygg_logit",
            ")",
            f"SELECT *, {col_prob} * {esc!r} AS {col_score}",
            "FROM _ygg_prob;"])

    def _sql_case_labels(self, f, col, bins, alias, is_bool, comentario="") -> list:
        """``CASE`` do rótulo da faixa (mesma saída de :meth:`_labels_from_bins`)."""
        linhas = [f"  CASE  -- faixa de {self.label(f)}" + (f" ({comentario})" if comentario else "")]
        for b in bins:
            lbl = self._bin_label(col, b).replace("'", "''")
            linhas.append(f"    WHEN {self._sql_cond(col, b, is_bool)} THEN '{lbl}'")
        linhas.append(f"    WHEN {col} IS NOT NULL THEN '(outros)'")
        linhas.append("    ELSE NULL")
        linhas.append(f"  END AS {alias},")
        return linhas

    @staticmethod
    def _sql_tira_virgula_final(corpo: str) -> str:
        """Remove a vírgula da ÚLTIMA expressão do SELECT (preserva comentário)."""
        linhas = corpo.split("\n")
        for i in range(len(linhas) - 1, -1, -1):
            ln = linhas[i]
            codigo, sep, com = ln.partition("  --")
            if codigo.rstrip().endswith(","):
                linhas[i] = codigo.rstrip()[:-1] + (sep + com if sep else "")
                break
        return "\n".join(linhas)

    def create_categorical(self, feature, new_name=None, dummies=None) -> str:
        """Materializa a binagem atual de ``feature`` (faixas numéricas ou grupos
        categóricos — **manuais** quando definidos, senão o **ótimo**) como uma NOVA
        variável categórica no DataFrame, candidata ao modelo. É o equivalente a
        "agrupar bins" da árvore de decisão, persistido numa coluna: junte categorias
        na mão (ex.: ``A,B; C,D``) e gere a variável agrupada.

        A derivação fica registrada (origem + bins), então a variável é **recriada
        automaticamente** ao escorar uma base que tenha só as variáveis originais.
        ``dummies`` (default: segue :meth:`scorecard_dummies` da variável): a nova
        variável já nasce com **uma categoria por faixa** como bins manuais e com
        as **dummies de scorecard** ligadas — UMA variável (uma linha no ranking,
        IV/gráficos das faixas) que entra no modelo como 0/1 por faixa, com a
        pior faixa de referência. Para materializar as dummies como colunas
        separadas use :meth:`create_scorecard_dummies`.

        Devolve o nome da nova variável."""
        if feature not in self.df.columns:
            raise ValueError(f"'{feature}' não está no DataFrame.")
        bins, kind = self._resolve_bins(feature, sample=self.ref_sample)
        if not bins:
            raise ValueError(f"Sem bins para '{feature}'. Defina cortes/grupos (ou rode o "
                             "binning ótimo) antes de criar a variável.")
        name = self._nome_livre(new_name or f"{feature}_cat")
        meta = {"categoria": None, "derived_from": feature,
                "derived_kind": kind, "derived_bins": bins}
        self.df[name] = self._derived_values(self.df[feature], meta, feature)
        if name not in self.candidates:
            self.candidates.append(name)
        self.var_meta[name] = meta
        if dummies is None:
            dummies = self.scorecard_dummies(feature)
        # rótulo distinto por versão: o nome digitado vira o rótulo; nomes
        # automáticos repetidos (<var>_cat_2, _3...) ganham "v2", "v3" — senão a
        # aba Variáveis mostrava várias linhas iguais "var (dummies)"
        tipo = "dummies" if dummies else "cat."
        if new_name:
            rotulo_nova = name
        else:
            base_auto = f"{feature}_cat"
            versao = name[len(base_auto) + 1:] if name != base_auto else ""
            rotulo_nova = (f"{self.label(feature)} ({tipo} v{versao})" if versao
                           else f"{self.label(feature)} ({tipo})")
        self.feature_labels[name] = rotulo_nova
        self._rank_version += 1   # nova candidata → o ranking de IV precisa recalcular
        if dummies:
            # uma categoria por faixa (lista, não texto: rótulos têm vírgula) e as
            # dummies de scorecard ligadas na nova variável
            rotulos = [self._bin_label(feature, b) for b in bins]
            self.set_manual_bins(name, [[r] for r in rotulos])
            self.set_scorecard_dummies(name)
        return name

    def create_scorecard_dummies(self, feature, prefix=None) -> list:
        """Materializa as **dummies de scorecard** de ``feature`` como variáveis
        0/1 no DataFrame, candidatas ao modelo: uma por faixa, **exceto a pior**
        (maior risco na referência), que é a referência — assim, com alvo 1 = mau,
        cada coeficiente sai negativo. Nomes ``<prefixo>_d<k>`` (``k`` = posição
        da faixa, 1 = primeira) e rótulo ``"<variável> = <faixa>"``.

        Usa as faixas atuais (manuais, com o destino dos faltantes). Cada dummy é
        registrada como derivada e **recriada automaticamente** na escoragem.
        Devolve a lista de nomes criados."""
        if feature not in self.df.columns:
            raise ValueError(f"'{feature}' não está no DataFrame.")
        if not self.manual_bins(feature):
            raise ValueError(f"'{self.label(feature)}' não tem categorização manual — "
                             "as dummies de scorecard usam as faixas manuais.")
        bins, kind = self._resolve_bins(feature, sample=self.ref_sample)
        spec = self._dummy_spec(feature)
        base = prefix or feature
        criadas = []
        for k, (b, lbl) in enumerate(zip(spec["bins"], spec["labels"])):
            if k == spec["ref"]:
                continue
            name = self._nome_livre(f"{base}_d{k + 1}")
            meta = {"categoria": None, "derived_from": feature, "derived_kind": kind,
                    "derived_bins": [b], "derived_dummy": True,
                    "dummy_ref": spec["labels"][spec["ref"]]}
            self.df[name] = self._derived_values(self.df[feature], meta, feature)
            if name not in self.candidates:
                self.candidates.append(name)
            self.var_meta[name] = meta
            self.feature_labels.setdefault(name, f"{self.label(feature)} = {lbl}")
            criadas.append(name)
        self._rank_version += 1
        return criadas

    def _nome_livre(self, name) -> str:
        """``name`` ou ``name_2``, ``name_3``... — o primeiro livre no DataFrame."""
        base, k = name, 2
        while name in self.df.columns:
            name = f"{base}_{k}"; k += 1
        return name

    def _derived_values(self, series, meta, src) -> np.ndarray:
        """Valores de uma variável derivada a partir da origem: rótulo da faixa
        (``create_categorical``) ou 0/1 da faixa (dummy de scorecard)."""
        bins = meta.get("derived_bins") or []
        if meta.get("derived_dummy"):
            return _bin_masks(series, bins)[0].astype("int64")
        return self._labels_from_bins(series, bins, src).to_numpy()

    def _apply_derived(self, X):
        """Recria, em ``X``, as variáveis derivadas (criadas via
        :meth:`create_categorical`) que o modelo usa e que ainda não estão presentes,
        a partir da variável de origem — para escorar bases com só as colunas crus."""
        need = [n for n in self.model_features
                if self.var_meta.get(n, {}).get("derived_from") and n not in X.columns]
        if not need:
            return X
        X = X.copy()
        for n in need:
            meta = self.var_meta[n]; src = meta["derived_from"]
            if src not in X.columns:
                raise ValueError(f"Para recriar a variável derivada '{n}', a tabela "
                                 f"precisa conter a variável de origem '{src}'.")
            X[n] = self._derived_values(X[src], meta, src)
        return X

    def _rebuild_derived(self):
        """Recria no ``self.df`` as variáveis derivadas registradas em ``var_meta``
        (usado após load, quando o df vem só com as colunas originais)."""
        for name, meta in self.var_meta.items():
            src = meta.get("derived_from")
            if not src or name in self.df.columns or src not in self.df.columns:
                continue
            self.df[name] = self._derived_values(self.df[src], meta, src)

    def _score_pandas(self, pdf, col_score="score", col_rating="rating", col_value=None,
                      ruler_sample=None, recreate_categories=None, cat_suffix="_faixa",
                      progress_callback=None):
        """Escora um pandas DataFrame: colunas originais + score + rating (+ valor
        previsto) e, quando o modelo usou variáveis categorizadas, as faixas
        recriadas. A tabela só precisa ter as variáveis originais do modelo.

        ``progress_callback`` (opcional): ``cb(key, label, status, detail)`` por
        etapa — ver :func:`_emit_progress`."""
        pdf = self._apply_derived(pd.DataFrame(pdf))    # recria derivadas (se houver)
        missing = [f for f in self.model_features if f not in pdf.columns]
        if missing:
            raise ValueError(
                f"Faltam variáveis do modelo na tabela: {missing}. "
                f"Ela precisa conter: {list(self.model_features)}.")
        n = len(pdf)
        _emit_progress(progress_callback, "score", "Escorar (score + rating)", "run",
                       f"{n:,} linhas".replace(",", "."))
        scored = self.predict(pdf, col_score=col_score, col_rating=col_rating,
                              col_value=col_value, ruler_sample=ruler_sample)
        out = pdf.copy()
        for c in scored.columns:
            out[c] = scored[c].to_numpy()
        _emit_progress(progress_callback, "score", "Escorar (score + rating)", "ok",
                       f"{n:,} linhas".replace(",", "."))
        if recreate_categories is None:
            recreate_categories = (self.feature_transform == "woe")
        if recreate_categories:
            _emit_progress(progress_callback, "categories", "Recriar categorias (faixas)", "run")
            cats = self.recreate_categories(pdf, suffix=cat_suffix)
            for c in cats.columns:
                out[c] = cats[c].to_numpy()
            _emit_progress(progress_callback, "categories", "Recriar categorias (faixas)", "ok",
                           f"{cats.shape[1]} coluna(s)")
        return out

    def _detached_scorer(self, col_rating="rating", col_value=None, ruler_sample=None,
                         recreate=False):
        """Cópia LEVE e serializável do segmenter, pronta para **broadcast** aos
        executores Spark: mantém o modelo/estratégia/bins, mas **descarta o
        DataFrame de treino** e os caches por-linha (que referenciam todas as
        linhas). Pré-computa no DRIVER — onde ``self.df`` ainda existe — a régua
        ``rating→valor`` (:meth:`rating_ruler`) e os bins de ``recreate_categories``
        (via ``_bins_cache``), de modo que escorar uma partição nos executores
        **não toque em ``df``**. Ver :meth:`_apply_spark_distributed`."""
        import copy
        if recreate:                              # pré-aquece o cache de bins (usa df)
            for f in (self.model_features or []):
                if f in self.df.columns and not self.var_meta.get(f, {}).get("derived_from"):
                    try:
                        self._resolve_bins(f, sample=self.ref_sample)
                    except Exception:
                        pass
        ruler_df = None
        if col_value is not None:                 # pré-computa a régua (usa df)
            ruler_df = self.rating_ruler(sample=ruler_sample, col_rating=col_rating,
                                         col_value=col_value)
        s = copy.copy(self)                       # rasa: compartilha model/var_meta/bins
        # a cópia rasa também leva ``calibration_`` (dict de parâmetros puros —
        # picklável): a UDF distribuída escora com a MESMA camada de calibração.
        s.df = self.df.iloc[0:0].copy()           # só o schema (sem as linhas de treino)
        s._mask_cache = {}
        s._samples_cache = None
        s._rank_cache = {}
        s._metrics_cache = None
        # caches por LINHA da base (safra/faixa) e por variável: nada disso pode
        # ir no pickle para os executores
        s._safra_cache = {}
        s._bincode_cache = {}
        s._iv_row_cache = {}
        s._fatias_cache = {}
        s._amostra_safra_cache = {}
        s._rating_codes_cache = None
        s._raw_score_cache = None
        s._amostra_graficos_cache = None
        s._metrics_ci_cache = None
        s._shap_cache = {}
        s.score_ = None
        s.rating_ = None
        s._tuning_cancel = None                   # Event tem lock → não é picklável (broadcast)
        # o estudo/resumo do Optuna não são usados na escoragem: descarta-os para não
        # inflar o broadcast (um estudo com muitos trials seria enviado a cada executor).
        s.study_ = None
        s.tuning_ = None
        s._ruler_override = ruler_df
        return s

    def _apply_spark_distributed(self, sdf, col_score, col_rating, col_value, ruler_sample,
                                 recreate, cat_suffix, n_partitions=None,
                                 progress_callback=None):
        """Escoragem **distribuída** de um Spark DataFrame via ``mapInPandas``: cada
        partição é escorada NO EXECUTOR (o modelo/estratégia/régua vão por broadcast),
        sem coletar a tabela no driver (``toPandas``) — o que evita OOM do driver e a
        queda do cluster em tabelas grandes. Devolve um Spark DataFrame LAZY (a
        computação ocorre na ação seguinte: ``saveAsTable`` / preview)."""
        from pyspark.sql import SparkSession
        from pyspark.sql.types import (BooleanType, DoubleType, LongType, StringType,
                                        StructField, StructType)
        spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        _emit_progress(progress_callback, "prepare",
                       "Preparar escoragem distribuída (broadcast do modelo)", "run")
        scorer = self._detached_scorer(col_rating=col_rating, col_value=col_value,
                                       ruler_sample=ruler_sample, recreate=recreate)
        # deriva o schema de saída escorando uma amostra pequena NO DRIVER (também
        # valida a ponta-a-ponta antes de despachar aos executores).
        sample_pdf = sdf.limit(50).toPandas()
        out_sample = scorer._score_pandas(sample_pdf, col_score, col_rating, col_value,
                                          ruler_sample, recreate, cat_suffix)
        out_cols = list(out_sample.columns)
        in_types = {f.name: f.dataType for f in sdf.schema.fields}

        # colunas que a escoragem SEMPRE (sobre)escreve — precedem o passthrough
        # (o tipo de saída vence o da coluna de mesmo nome que já exista na tabela,
        # ex.: re-escorar uma base que já tem 'score'/'rating' de uma rodada anterior).
        written_str = {col_rating}
        written_str |= {c for c in out_cols if c.endswith(cat_suffix) and c not in in_types}
        written_dbl = {col_score} | ({col_value} if col_value is not None else set())

        def _spark_type(colname):
            if colname in written_dbl:
                return DoubleType()                # score/valor: sempre double
            if colname in written_str:
                return StringType()                # rating/faixas recriadas: sempre texto
            if colname in in_types:                # passthrough: preserva o tipo de origem
                return in_types[colname]
            dt = out_sample[colname].dtype
            if pd.api.types.is_bool_dtype(dt):
                return BooleanType()
            if pd.api.types.is_integer_dtype(dt):
                return LongType()
            if pd.api.types.is_float_dtype(dt):
                return DoubleType()
            return StringType()                    # derivadas / demais texto

        out_schema = StructType([StructField(c, _spark_type(c), True) for c in out_cols])
        _emit_progress(progress_callback, "prepare",
                       "Preparar escoragem distribuída (broadcast do modelo)", "ok",
                       f"{len(out_cols)} colunas de saída")
        # getter do scorer nos executores: broadcast no cluster clássico; em Spark
        # Connect (sem sparkContext) cai para closure. Ver _scorer_broadcast_getter.
        _get_scorer = _scorer_broadcast_getter(spark, scorer)
        _cs, _cr, _cv, _rs, _rec, _suf = (col_score, col_rating, col_value, ruler_sample,
                                          recreate, cat_suffix)

        def _score_iter(it):
            sc = _get_scorer()
            for part in it:
                res = sc._score_pandas(part, _cs, _cr, _cv, _rs, _rec, _suf)
                for c in out_cols:                 # garante todas as colunas do schema
                    if c not in res.columns:
                        res[c] = None
                yield res[out_cols]                # e a MESMA ordem do schema

        if n_partitions:
            sdf = sdf.repartition(int(n_partitions))
        _emit_progress(progress_callback, "score",
                       "Escorar por partição (distribuído, sem coletar no driver)", "ok",
                       "plano distribuído montado")
        return sdf.mapInPandas(_score_iter, schema=out_schema)

    def apply_spark(self, sdf, col_score="score", col_rating="rating", col_value=None,
                    ruler_sample=None, recreate_categories=None, cat_suffix="_faixa",
                    progress_callback=None, distributed=None, n_partitions=None,
                    probe=True):
        """Escora um **Spark DataFrame** (ex.: tabela do Databricks/Unity Catalog) e
        devolve um Spark DataFrame com ``score`` + ``rating`` (+ valor previsto) e,
        quando o modelo usou variáveis categorizadas (WoE/bins), as faixas recriadas.
        A tabela só precisa ter as variáveis originais do modelo (``model_features``);
        a binagem/WoE é refeita internamente.

        Por padrão escora de forma **distribuída** (``mapInPandas``): cada partição é
        escorada no executor, sem coletar a tabela no driver — evita OOM do driver e
        a queda do cluster em tabelas grandes. ``distributed=False`` força o caminho
        **legado** (``toPandas`` no driver, para bases pequenas ou fora do cluster).
        ``n_partitions`` reparticiona antes de escorar (paralelismo).

        Robustez: com ``probe=True`` (padrão) uma linha é escorada nos executores já
        na montagem — se o caminho distribuído falhar (erro na montagem **ou** no
        executor, ex.: a lib ``yggdrasil`` não instalada no cluster), **cai
        automaticamente no legado** em vez de quebrar depois, na ação. ``probe=False``
        pula essa sonda (retorna o DataFrame 100% lazy; erros de executor só
        aparecerão na ação, sem fallback).

        Requisito do caminho distribuído: o pacote ``yggdrasil`` deve estar
        **instalado nos executores** (biblioteca do cluster) — o normal no Databricks
        — pois o modelo/estratégia de rating são desserializados lá. Sem isso (com
        ``probe=True``) a escoragem cai no legado automaticamente.

        Nota: os *reason codes* (motivos SHAP por linha — :meth:`reason_codes`)
        NÃO são anexados na escoragem Spark: o custo do SHAP dentro da UDF
        distribuída é proibitivo (fica para uma evolução futura). Para obtê-los,
        use :meth:`predict` com ``col_reasons`` no caminho pandas."""
        if self.model is None:
            raise RuntimeError("Ajuste/defina o modelo antes (fit / set_model).")
        try:
            from pyspark.sql import SparkSession
        except ImportError as e:  # pragma: no cover
            raise ImportError("apply_spark requer pyspark — no Databricks já vem no "
                              "cluster; fora dele: pip install pyspark") from e
        missing = [f for f in self.model_features if f not in sdf.columns]
        if missing:
            raise ValueError(
                f"Colunas ausentes no Spark DataFrame: {missing}. A tabela precisa ter "
                f"as variáveis do modelo: {list(self.model_features)}.")
        if recreate_categories is None:
            recreate_categories = (self.feature_transform == "woe")
        recreate_categories = bool(recreate_categories)
        if distributed is None:
            distributed = True
        if distributed:
            try:
                sout = self._apply_spark_distributed(
                    sdf, col_score, col_rating, col_value, ruler_sample,
                    recreate_categories, cat_suffix, n_partitions=n_partitions,
                    progress_callback=progress_callback)
                if probe:
                    # SONDA: força a execução de 1 linha nos executores AGORA, dentro
                    # do try. Como ``mapInPandas`` é lazy, erros que só aparecem no
                    # executor (ex.: a lib yggdrasil não instalada no cluster, ou um
                    # erro de escoragem) surgiriam apenas na AÇÃO (saveAsTable) — fora
                    # deste try. A sonda os antecipa para cá e permite cair no legado.
                    sout.take(1)
                return sout
            except Exception as e:                # robustez: cai no legado sem derrubar
                warnings.warn(f"Escoragem distribuída indisponível ({type(e).__name__}: "
                              f"{e}); usando o caminho legado (toPandas no driver).")
                _emit_progress(progress_callback, "prepare",
                               "Escoragem distribuída indisponível — caminho legado", "ok",
                               type(e).__name__)
        # --- legado: coleta a tabela no driver (bases pequenas / fora do cluster) ---
        _emit_progress(progress_callback, "collect", "Coletar no driver (toPandas)", "run")
        pdf = sdf.toPandas()
        _emit_progress(progress_callback, "collect", "Coletar no driver (toPandas)", "ok",
                       f"{len(pdf):,} linhas".replace(",", "."))
        out = self._score_pandas(pdf, col_score, col_rating, col_value, ruler_sample,
                                 recreate_categories, cat_suffix,
                                 progress_callback=progress_callback)
        _emit_progress(progress_callback, "build_spark", "Montar Spark DataFrame", "run")
        spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        sout = spark.createDataFrame(out)
        _emit_progress(progress_callback, "build_spark", "Montar Spark DataFrame", "ok")
        return sout

    def score_table(self, data, col_score="score", col_rating="rating", col_value=None,
                    ruler_sample=None, recreate_categories=None, cat_suffix="_faixa",
                    output_table=None, mode="overwrite", spark=None,
                    progress_callback=None, distributed=None, n_partitions=None,
                    probe=True):
        """Escora uma tabela e devolve as **notas (score)** e os **ratings**.

        ``data`` pode ser o **nome** de uma tabela do Databricks (``catalog.schema.
        tabela``), um **Spark DataFrame** ou um **pandas DataFrame**. A tabela só
        precisa conter as variáveis originais do modelo; se uma variável foi
        categorizada (faixas/grupos via WoE), a categoria é recriada na saída.

        - nome de tabela ou Spark DataFrame → devolve um **Spark DataFrame** (e grava
          em ``output_table`` quando informado). Escora **distribuído** por padrão
          (ver :meth:`apply_spark`); ``distributed=False`` força o caminho legado e
          ``n_partitions`` controla o paralelismo;
        - pandas DataFrame → devolve **pandas**.

        ``progress_callback`` (opcional): ``cb(key, label, status, detail)`` chamado
        a cada etapa (carregar/coletar/escorar/recriar/salvar) — útil p/ uma tabela
        de progresso na UI. Ver :func:`_emit_progress`."""
        if self.model is None:
            raise RuntimeError("Ajuste/defina o modelo antes (fit / set_model).")
        if isinstance(data, str):
            from pyspark.sql import SparkSession
            _emit_progress(progress_callback, "load", f"Carregar tabela '{data}'", "run")
            spark = spark or SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
            data = spark.table(data)
            _emit_progress(progress_callback, "load", f"Carregar tabela", "ok")
        if hasattr(data, "toPandas"):                       # Spark DataFrame
            out = self.apply_spark(data, col_score=col_score, col_rating=col_rating,
                                   col_value=col_value, ruler_sample=ruler_sample,
                                   recreate_categories=recreate_categories,
                                   cat_suffix=cat_suffix, progress_callback=progress_callback,
                                   distributed=distributed, n_partitions=n_partitions,
                                   probe=probe)
            if output_table:
                _emit_progress(progress_callback, "save", f"Salvar em '{output_table}'", "run")
                # mergeSchema: evolui o schema (colunas novas: score/rating/valor) ao
                # sobrescrever (mode='overwrite' por padrão) se a base já existir.
                (out.write.mode(mode).option("mergeSchema", "true")
                    .saveAsTable(output_table))
                _emit_progress(progress_callback, "save", f"Salvar em '{output_table}'", "ok")
            _emit_progress(progress_callback, "done", "Escoragem concluída", "ok")
            return out
        out = self._score_pandas(pd.DataFrame(data), col_score, col_rating, col_value,
                                 ruler_sample, recreate_categories, cat_suffix,
                                 progress_callback=progress_callback)
        _emit_progress(progress_callback, "done", "Escoragem concluída", "ok")
        return out

    # ------------------------------------------------------------------
    # CHAMPION × CHALLENGER: snapshot em memória + comparação com modelo salvo
    #   snapshot/compare_to → foto nomeada (config + métricas + score_) versus o
    #   modelo VIGENTE; diff_models → outro ModelSegmenter salvo (save/load JSON)
    #   escorado sobre as MESMAS linhas, com a matriz de migração de ratings.
    # ------------------------------------------------------------------
    def snapshot(self, name: str = "baseline") -> dict:
        """Congela o modelo vigente como um **snapshot nomeado em memória**.

        Guarda a configuração (algoritmo, hiperparâmetros, transformação,
        variáveis do modelo e derivadas), as métricas por amostra e uma cópia da
        ``Series`` ``score_`` — o suficiente para :meth:`compare_to` medir deltas,
        sobrepor ROC e calcular o PSI entre os scores depois de um re-fit. A foto
        vive só nesta sessão (memória do kernel; não entra em :meth:`to_dict`)
        e **sobrevive a re-fits** — persistência de verdade é :meth:`save`."""
        if self.score_ is None:
            raise RuntimeError("Ajuste o modelo antes (fit / set_model / load).")
        snap = {
            "nome": str(name),
            "task_type": self.task_type,
            "algorithm": self.algorithm,
            "hyperparams": dict(self.hyperparams or {}),
            "feature_transform": self.feature_transform,
            "class_balance": bool(self.class_balance),
            "monotone": self.monotone,
            "monotone_dirs": dict(self.monotone_dirs_ or {}),
            "model_features": list(self.model_features),
            "derived": {n: dict(self.var_meta.get(n) or {})
                        for n in self.derived_features()},
            "two_stage": bool(self.two_stage),
            "two_stage_threshold": self.two_stage_threshold,
            "metrics": self.metrics(),
            "score": self.score_.copy(),
            "criado_em": pd.Timestamp.now().isoformat(timespec="seconds"),
        }
        self.snapshots_[str(name)] = snap
        return snap

    def _resolve_snapshot(self, ref) -> dict:
        """Snapshot a partir do nome (em ``snapshots_``) ou do próprio dict."""
        if isinstance(ref, dict):
            return ref
        snap = self.snapshots_.get(ref)
        if snap is None:
            raise KeyError(f"Snapshot '{ref}' não encontrado. Disponíveis: "
                           f"{sorted(self.snapshots_)} (use snapshot(nome) antes).")
        return snap

    def score_psi(self, baseline_score, n_bins: int = 10, eps: float = 1e-6) -> pd.DataFrame:
        """PSI entre a distribuição de um score **baseline** e a do score vigente,
        por amostra — mede o quanto o modelo novo desloca os scores sobre as
        MESMAS linhas. Faixas = decis do score baseline dentro da amostra
        (``n_bins``); PSI clássico via :func:`psi_from_shares`."""
        if self.score_ is None:
            raise RuntimeError("Ajuste o modelo antes (fit / set_model / load).")
        base = pd.Series(baseline_score).reindex(self.df.index).astype("float64")
        rows = []
        for a in self._samples():
            mask = (np.ones(len(self.df), dtype=bool) if self.sample_col is None
                    else self._frame_mask(a))
            b = base.to_numpy()[mask]
            c = self.score_.to_numpy(dtype="float64")[mask]
            b, c = b[~np.isnan(b)], c[~np.isnan(c)]
            if b.size == 0 or c.size == 0:
                continue
            # cortes internos = quantis do baseline na amostra (dedup p/ score
            # discreto/constante); bin i = (q_{i-1}, q_i] via searchsorted.
            qs = np.unique(np.quantile(b, np.linspace(0, 1, n_bins + 1)[1:-1]))
            cb = np.bincount(np.searchsorted(qs, b, side="left"), minlength=len(qs) + 1)
            cc = np.bincount(np.searchsorted(qs, c, side="left"), minlength=len(qs) + 1)
            psi = _psi_from_shares((cb / b.size).tolist(), (cc / c.size).tolist(), eps)
            rows.append({"amostra": a, "psi": round(float(psi), 4),
                         "classificacao": _classifica_psi(psi)})
        return pd.DataFrame(rows, columns=["amostra", "psi", "classificacao"])

    def _baseline_config_frame(self, snap: dict) -> tuple:
        """Tabelas de diff de CONFIG e de HIPERPARÂMETROS (baseline × atual):
        DataFrames ``item/parametro · baseline · atual · mudou`` — compartilhadas
        por :meth:`compare_to` e :meth:`diff_models`."""
        cfg_pairs = [
            ("algoritmo", snap.get("algorithm"), self.algorithm),
            ("transformação", snap.get("feature_transform"), self.feature_transform),
            ("balanceamento de classes", bool(snap.get("class_balance", False)),
             bool(self.class_balance)),
            ("monotonicidade", snap.get("monotone"), self.monotone),
            ("two-stage", bool(snap.get("two_stage", False)), bool(self.two_stage)),
            ("nº de variáveis", len(snap.get("model_features") or []),
             len(self.model_features or [])),
        ]
        config = pd.DataFrame([{"item": i, "baseline": b, "atual": a,
                                "mudou": bool(b != a)} for i, b, a in cfg_pairs])
        hp_base = dict(snap.get("hyperparams") or {})
        hp_cur = dict(self.hyperparams or {})
        hp_rows = [{"parametro": k, "baseline": hp_base.get(k, "—"),
                    "atual": hp_cur.get(k, "—"),
                    "mudou": bool(hp_base.get(k, "—") != hp_cur.get(k, "—"))}
                   for k in sorted(set(hp_base) | set(hp_cur))]
        hyper = pd.DataFrame(hp_rows,
                             columns=["parametro", "baseline", "atual", "mudou"])
        return config, hyper

    def _feature_diff(self, base_feats, cur_feats) -> dict:
        """Variáveis que entraram/saíram/permaneceram entre dois conjuntos."""
        b, c = set(base_feats or []), set(cur_feats or [])
        return {"entraram": sorted(c - b), "sairam": sorted(b - c),
                "mantidas": sorted(b & c)}

    def _score_ci_map(self, score: pd.Series, metrics, n_boot, alpha, seed) -> dict:
        """IC bootstrap ``(metrica, amostra) → (ic_low, ic_high)`` para um score
        qualquer (ex.: o congelado num snapshot) — mesmo protocolo de
        :meth:`metrics_ci`, sem cache (usado uma vez por comparação)."""
        def _rmse(yt, ys):
            return float(np.sqrt(np.mean((yt - ys) ** 2)))

        sc_all = pd.Series(score).reindex(self.df.index).astype("float64")
        out = {}
        for a in self._samples():
            mask = (np.ones(len(self.df), dtype=bool) if self.sample_col is None
                    else self._frame_mask(a))
            y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
            sc = sc_all.to_numpy()[mask]
            ok = ~np.isnan(y) & ~np.isnan(sc)
            y, sc = y[ok], sc[ok]
            if y.size == 0:
                continue
            fns = [_rmse if m == "rmse" else m for m in metrics]
            try:
                cis = bootstrap_metrics_ci(y, sc, metrics=fns, n_boot=int(n_boot),
                                           alpha=alpha, seed=seed)
            except Exception:               # alguma métrica não computável: uma a uma
                cis = []
                for fn in fns:
                    try:
                        cis.append(bootstrap_metrics_ci(y, sc, metrics=[fn],
                                                        n_boot=int(n_boot),
                                                        alpha=alpha, seed=seed)[0])
                    except Exception:
                        cis.append(None)
            for m, ci in zip(metrics, cis):
                if ci is not None:
                    out[(m, a)] = (float(ci["ic_low"]), float(ci["ic_high"]))
        return out

    def compare_to(self, baseline, n_boot: int = 200, alpha: float = 0.05,
                   seed=None) -> dict:
        """Compara o modelo VIGENTE com um snapshot (:meth:`snapshot`) — o coração
        do champion × challenger em memória.

        ``baseline`` é o **nome** do snapshot ou o próprio dict devolvido por
        :meth:`snapshot`. Devolve um dict com:

        * ``deltas`` — tabela métrica × amostra com ``baseline``, ``atual``,
          ``delta`` (atual − baseline) e ``veredicto`` (semáforo): *melhorou* /
          *piorou* pela direção boa da métrica; **empate** quando os ICs
          bootstrap (:meth:`metrics_ci`; mesmas métricas/protocolo) do baseline
          e do atual se sobrepõem — diferença dentro do ruído amostral — ou o
          delta é zero. Métricas sem IC usam só o sinal do delta.
        * ``psi_score`` — PSI entre o score do baseline e o vigente por amostra
          (:meth:`score_psi`): o quanto a "régua de scores" deslocou.
        * ``variaveis`` — ``{entraram, sairam, mantidas}`` entre os dois modelos.
        * ``config`` / ``hyperparams`` — diffs de configuração e de HPs.
        * ``metrics_baseline`` / ``metrics_atual`` — tabelas completas.

        A ROC sobreposta (classificação) fica em :meth:`plot_roc_compare`."""
        snap = self._resolve_snapshot(baseline)
        if self.score_ is None:
            raise RuntimeError("Ajuste o modelo antes (fit / set_model / load).")
        if snap.get("task_type") not in (None, self.task_type):
            raise ValueError(f"Snapshot de task_type diferente: "
                             f"{snap.get('task_type')} vs {self.task_type}.")
        base_m = snap["metrics"].set_index("amostra")
        cur_m = self.metrics().set_index("amostra")
        # ICs p/ qualificar o empate (dentro do ruído): atual via metrics_ci
        # (cacheado por identidade do score_); baseline re-bootstrapado sobre o
        # score congelado. Mesmo protocolo/seed ⇒ comparação justa.
        ci_metrics = (("auc", "ks", "gini") if self.task_type == "classification"
                      else ("r2", "rmse"))
        if seed is None:
            seed = self.random_state
        ci_cur, ci_base = {}, {}
        try:
            cur_ci_df = self.metrics_ci(n_boot=n_boot, metrics=ci_metrics,
                                        alpha=alpha, seed=seed)
            ci_cur = {(r["metrica"], r["amostra"]): (float(r["ic_low"]),
                                                     float(r["ic_high"]))
                      for _, r in cur_ci_df.iterrows()}
            ci_base = self._score_ci_map(snap["score"], ci_metrics, n_boot, alpha, seed)
        except Exception:                   # sem IC ⇒ veredicto só pelo sinal
            ci_cur, ci_base = {}, {}

        # 'n' não é métrica; 'ks_cutoff' é um limiar na escala do score (não
        # comparável em qualidade) — mesmos descartes de metric_shifts.
        cols = [c for c in cur_m.columns if c in base_m.columns
                and c not in ("n", "ks_cutoff")]
        rows = []
        for a in [s for s in self._samples() if s in base_m.index and s in cur_m.index]:
            for m in cols:
                vb, vc = base_m.loc[a, m], cur_m.loc[a, m]
                if not (isinstance(vb, (int, float, np.integer, np.floating))
                        and isinstance(vc, (int, float, np.integer, np.floating))):
                    continue
                vb, vc = float(vb), float(vc)
                if not (np.isfinite(vb) and np.isfinite(vc)):
                    continue
                delta = vc - vb
                sentido = _HIGHER_IS_BETTER.get(m)
                if sentido is None:                     # viés: magnitude perto de 0
                    melhor = abs(vc) < abs(vb)
                else:
                    melhor = (delta > 0) if sentido else (delta < 0)
                lb, lc = ci_base.get((m, a)), ci_cur.get((m, a))
                if delta == 0:
                    veredicto = "empate"
                elif (lb is not None and lc is not None
                        and all(np.isfinite(v) for v in (*lb, *lc))):
                    disjuntos = lc[0] > lb[1] or lc[1] < lb[0]
                    veredicto = (("melhorou" if melhor else "piorou")
                                 if disjuntos else "empate")
                else:
                    veredicto = "melhorou" if melhor else "piorou"
                rows.append({"metrica": m, "amostra": a, "baseline": round(vb, 6),
                             "atual": round(vc, 6), "delta": round(delta, 6),
                             "veredicto": veredicto})
        deltas = pd.DataFrame(rows, columns=["metrica", "amostra", "baseline",
                                             "atual", "delta", "veredicto"])
        config, hyper = self._baseline_config_frame(snap)
        return {
            "baseline": snap.get("nome"),
            "criado_em": snap.get("criado_em"),
            "deltas": deltas,
            "psi_score": self.score_psi(snap["score"]),
            "variaveis": self._feature_diff(snap.get("model_features"),
                                            self.model_features),
            "config": config,
            "hyperparams": hyper,
            "metrics_baseline": snap["metrics"].copy(),
            "metrics_atual": self.metrics(),
        }

    def plot_roc_compare(self, baseline, sample=None, figsize=(5.4, 5.0), dpi=150,
                         save_path=None, ax=None):
        """ROC **sobreposta** baseline × modelo vigente na mesma amostra
        (classificação): o desafiante domina quando sua curva envolve a do
        baseline. ``baseline`` = nome do snapshot ou o dict de :meth:`snapshot`."""
        if self.task_type != "classification":
            raise ValueError("plot_roc_compare é exclusivo de classificação; em "
                             "regressão compare pelas métricas (compare_to).")
        from sklearn.metrics import roc_curve, roc_auc_score
        snap = self._resolve_snapshot(baseline)
        if sample is None:
            sample = self.ref_sample
        mask = (pd.Series(True, index=self.df.index) if self.sample_col is None
                else self.df[self.sample_col] == sample)
        y_all = self.df.loc[mask, self.target].to_numpy(dtype="float64")
        base_sc = snap["score"].reindex(self.df.index)[mask].to_numpy(dtype="float64")
        cur_sc = self.score_[mask].to_numpy(dtype="float64")
        fig, ax = _new_ax(figsize, dpi, ax)
        rotulo_base = snap.get("nome") or "baseline"
        for nome, sc, cor, ls in ((rotulo_base, base_sc, "#889", "--"),
                                  ("atual", cur_sc, "#15324a", "-")):
            ok = ~np.isnan(y_all) & ~np.isnan(sc)
            y, s = y_all[ok], sc[ok]
            if len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, s)
            auc = roc_auc_score(y, s)
            ax.plot(fpr, tpr, color=cor, ls=ls, lw=2.0,
                    label=f"{nome} · AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], color="#bbb", ls=":", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        _pct_axis(ax, "both")
        ax.set_title(f"ROC · baseline × atual · {sample}", fontsize=11,
                     fontweight="bold", color="#15324a")
        ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def diff_models(self, other, eps: float = 1e-6) -> dict:
        """Compara este modelo (A, vigente) com **outro ModelSegmenter salvo** (B)
        sobre as MESMAS linhas (``self.df``).

        ``other`` pode ser o **caminho do .json** gerado por :meth:`save` (o
        ``.model.joblib`` ao lado é carregado junto) ou um ``ModelSegmenter`` já
        construído. Devolve dict com:

        * ``resumo`` — nº de variáveis + métrica-chave (AUC/R²) por amostra,
          A vs B vs Δ (B−A);
        * ``migracao`` — **matriz de migração** ``crosstab(rating_A, rating_B)``
          sobre as mesmas linhas (o número que o comitê pede) e ``concordancia``
          (fração com o MESMO rating) — ``None``/NaN quando algum dos modelos
          está sem régua de ratings;
        * ``variaveis`` (entraram/saíram/mantidas), ``config``/``hyperparams``
          (diffs), ``psi_score`` (PSI score B × A por amostra) e as tabelas
          completas ``metrics_a``/``metrics_b``."""
        import os
        if isinstance(other, (str, os.PathLike)):
            # instância "oca" só para reusar load() (correção de sufixo, erros
            # amigáveis, score + ratings reaplicados) sem clonar df antes da hora:
            # com df= explícito, load não toca nenhum outro atributo do objeto.
            alvo = ModelSegmenter.__new__(ModelSegmenter)
            other = alvo.load(str(other), df=self.df)
        if not isinstance(other, ModelSegmenter):
            raise TypeError("other deve ser um caminho de .json salvo ou um "
                            "ModelSegmenter.")
        if other.task_type != self.task_type:
            raise ValueError(f"Modelos de task_type diferentes: {self.task_type} "
                             f"vs {other.task_type}.")
        if self.score_ is None or other.score_ is None:
            raise RuntimeError("Os dois modelos precisam estar ajustados (o salvo "
                               "deve ter o .model.joblib ao lado do .json).")
        ma, mb = self.metrics(), other.metrics()
        key = "auc" if self.task_type == "classification" else "r2"
        rows = [{"métrica": "nº de variáveis",
                 "modelo A": len(self.model_features or []),
                 "modelo B": len(other.model_features or [])}]
        rows[0]["Δ (B−A)"] = rows[0]["modelo B"] - rows[0]["modelo A"]
        for am in ma["amostra"]:
            va = ma.loc[ma["amostra"] == am, key]
            vb = mb.loc[mb["amostra"] == am, key]
            if len(va) and len(vb):
                a, b = float(va.iloc[0]), float(vb.iloc[0])
                rows.append({"métrica": f"{key.upper()} · {am}",
                             "modelo A": round(a, 4), "modelo B": round(b, 4),
                             "Δ (B−A)": round(b - a, 4)})
        resumo = pd.DataFrame(rows)

        # matriz de migração de ratings A × B nas MESMAS linhas (crosstab)
        migracao, concordancia, notas = None, float("nan"), []
        if self.rating_ is not None and other.rating_ is not None:
            ra, rb = self._rating_series(), other._rating_series()
            valid = ra.notna() & rb.notna()
            if bool(valid.any()):
                migracao = pd.crosstab(ra[valid], rb[valid], dropna=False)
                migracao.index.name, migracao.columns.name = "rating_A", "rating_B"
                concordancia = float((ra[valid].astype(str)
                                      == rb[valid].astype(str)).mean())
        else:
            notas.append("migração indisponível: gere os ratings nos dois modelos "
                         "(build_ratings) para a matriz de migração.")
        snap_b = {"nome": "modelo B", "task_type": other.task_type,
                  "algorithm": other.algorithm, "hyperparams": other.hyperparams,
                  "feature_transform": other.feature_transform,
                  "class_balance": other.class_balance, "monotone": other.monotone,
                  "two_stage": other.two_stage,
                  "model_features": other.model_features}
        config, hyper = self._baseline_config_frame(snap_b)
        # no diff A×B o "baseline" das tabelas é o modelo B (salvo) — renomeia
        config = config.rename(columns={"baseline": "modelo B", "atual": "modelo A"})
        hyper = hyper.rename(columns={"baseline": "modelo B", "atual": "modelo A"})
        return {"resumo": resumo, "migracao": migracao, "concordancia": concordancia,
                "variaveis": self._feature_diff(other.model_features,
                                                self.model_features),
                "config": config, "hyperparams": hyper,
                "psi_score": self.score_psi(other.score_, eps=eps),
                "metrics_a": ma, "metrics_b": mb, "notas": notas}

    def to_dict(self) -> dict:
        """Configuração serializável (sem o modelo binário — ver :meth:`save`)."""
        return {
            "schema": SCHEMA,
            "meta": {"target": self.target, "task_type": self.task_type,
                     "sample_col": self.sample_col, "ref_sample": self.ref_sample,
                     "date_col": self.date_col, "feature_labels": self.feature_labels,
                     "problem_label": self.problem_label,
                     "score_scale": self.score_scale, "random_state": self.random_state},
            "candidates": list(self.candidates),
            "included": sorted(self.included),
            "var_meta": self.var_meta,
            "algorithm": self.algorithm,
            "hyperparams": self.hyperparams,
            "feature_transform": self.feature_transform,
            "class_balance": self.class_balance,
            # monotonicidade: a escolha ('auto' | dict | None) e as direções
            # efetivamente aplicadas no último fit (governança/reprodutibilidade)
            "monotone": self.monotone,
            "monotone_dirs": self.monotone_dirs_,
            "model_features": list(self.model_features),
            "rating_config": self.rating_config,
            "two_stage": self.two_stage,
            "two_stage_threshold": self.two_stage_threshold,
            # camada de calibração pós-treino (calibrate): parâmetros 100% JSON —
            # o load a reaplica no fluxo de score sem depender do joblib
            "calibration": self.calibration_,
            # política da última esteira de seleção (select_features): etapas +
            # parâmetros efetivos, para reproduzir a seleção. JSONs antigos não
            # têm a chave ⇒ None no from_dict.
            "selection_policy": self.selection_policy_,
        }

    def save(self, path: str):
        """Salva a configuração em JSON e o modelo+estratégia de rating em
        ``<path>.model.joblib`` (joblib)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        try:
            import joblib
            joblib.dump({"model": self.model, "rating_strategy": self.rating_strategy},
                        path + ".model.joblib")
        except Exception as e:  # pragma: no cover
            print(f"[save] modelo não serializado: {e}")
        return path

    @classmethod
    def from_dict(cls, data: dict, df: pd.DataFrame, verbose: bool = False):
        meta = data["meta"]
        seg = cls(df, target=meta["target"], task_type=meta["task_type"],
                  sample_col=meta.get("sample_col"), ref_sample=meta.get("ref_sample", "DES"),
                  feature_labels=meta.get("feature_labels"),
                  problem_label=meta.get("problem_label"),
                  features=data.get("candidates"), date_col=meta.get("date_col"),
                  verbose=verbose, score_scale=meta.get("score_scale", 1000.0),
                  random_state=meta.get("random_state", 42))
        seg.included = set(data.get("included", seg.candidates))
        seg.var_meta = data.get("var_meta", seg.var_meta)
        seg.algorithm = data.get("algorithm")
        seg.hyperparams = data.get("hyperparams", {})
        seg.feature_transform = data.get("feature_transform", "raw")
        seg.class_balance = bool(data.get("class_balance", False))
        seg.monotone = data.get("monotone")
        seg.monotone_dirs_ = {k: int(v) for k, v in
                              (data.get("monotone_dirs") or {}).items()}
        seg.model_features = data.get("model_features", [])
        seg.rating_config = data.get("rating_config", {})
        seg.two_stage = bool(data.get("two_stage", False))
        seg.two_stage_threshold = data.get("two_stage_threshold")
        # camada de calibração: setada ANTES do load recomputar o score_, para o
        # score carregado já sair calibrado (JSONs antigos: sem a chave ⇒ None)
        seg.calibration_ = data.get("calibration")
        # política da última seleção (JSONs antigos: sem a chave ⇒ None). Só a
        # política volta — a trilha completa (selection_) é da sessão que rodou.
        seg.selection_policy_ = data.get("selection_policy")
        seg._rebuild_derived()      # recria colunas categóricas derivadas no df
        return seg

    def load(self, path: str, df: pd.DataFrame = None):
        """Carrega configuração + modelo **no próprio objeto** (in-place) e o
        devolve (``return self``), para que tanto ``seg.load(path)`` quanto
        ``seg = ModelSegmenter(...).load(path)`` funcionem. Se ``df`` for dado,
        usa-o (senão, o ``df`` atual); recalcula o score e, se havia rating,
        reaproveita a estratégia salva para reaplicar os ratings — deixando o
        modelo pronto para métricas, ratings e escoragem sem re-treinar.

        Espera o **.json de configuração** gerado por :meth:`save` (o modelo
        binário fica ao lado em ``<arquivo>.model.joblib`` e é carregado sozinho).
        Se você apontar por engano para o próprio ``.model.joblib`` (ou outro
        ``.joblib``/``.pkl``), o load corrige o sufixo automaticamente e, se ainda
        assim o arquivo não for um JSON de texto, levanta um erro explicando."""
        # Engano comum: apontar para o binário '<cfg>.model.joblib' em vez do
        # .json — dá "utf-8 can't decode byte 0x80" (0x80 = início de pickle).
        # Corrige o sufixo para o .json de configuração correspondente.
        if path.endswith(".model.joblib"):
            path = path[: -len(".model.joblib")]
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(
                f"'{path}' não é um JSON de configuração válido do ModelSegmenter "
                f"({type(e).__name__}: {e}). O load espera o arquivo .json gerado por "
                f"save(); o modelo binário fica ao lado em '<arquivo>.model.joblib' e "
                f"é carregado automaticamente. Se você apontou para um .joblib/.pkl "
                f"(binário — começa com o byte 0x80 do pickle), passe o .json no lugar."
            ) from e
        seg = ModelSegmenter.from_dict(data, self.df if df is None else df, verbose=False)
        try:
            import joblib
            blob = joblib.load(path + ".model.joblib")
            seg.model = blob.get("model")
            seg.rating_strategy = blob.get("rating_strategy")
            if isinstance(seg.model, _TwoStageModel):   # robustez p/ JSONs sem a flag
                seg.two_stage = True
                seg.two_stage_threshold = seg.model.threshold
        except Exception as e:  # pragma: no cover
            print(f"[load] modelo não carregado: {e}")
        if seg.model is not None and seg.model_features:
            seg.score_ = seg._compute_score(seg.df)
            if seg.rating_strategy is not None:
                cfg = seg._make_cfg("_amostra")
                wf = seg._rating_frame()
                seg.rating_ = seg.rating_strategy.transform(wf, cfg)
                seg.rating_col_ = seg.rating_strategy.column
                seg.rating_labels_ = list(seg.rating_strategy.labels_)
        # adota o estado carregado no próprio objeto: evita o erro clássico de
        # ``seg.load(path)`` "não fazer nada" (antes só o retorno vinha carregado).
        self.__dict__.update(seg.__dict__)
        return self

    # ------------------------------------------------------------------
    # REPORT_PDF: relatório do modelo em PDF (capa + métricas + fórmula/SHAP +
    #   ratings). Usa matplotlib (sem dependência extra).
    # ------------------------------------------------------------------
    def report_pdf(self, path: str) -> str:
        """Gera um relatório PDF do modelo em ``path`` e o devolve. Páginas: capa
        (algoritmo, variáveis), métricas por amostra, fórmula (logística/linear)
        ou importância SHAP (não-lineares), e a régua de ratings (se houver)."""
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        feats = list(self.model_features or self.selected_features() or self.candidates)

        def _trunc(v, n=46):
            s = str(v)
            return s if len(s) <= n else s[:n - 1] + "…"

        def _table_fig(df, titulo, fs=8):
            df = df.copy()
            for c in df.columns:
                if df[c].dtype.kind in "fc":
                    df[c] = df[c].round(4)
            cell = [[_trunc(v) for v in row] for row in df.astype(object).values]
            fig = plt.figure(figsize=(11, max(2.0, 0.42 * (len(df) + 2))))
            ax = fig.add_subplot(111); ax.axis("off")
            ax.set_title(titulo, fontsize=13, fontweight="bold", color="#15324a", loc="left")
            if cell:
                t = ax.table(cellText=cell, colLabels=list(df.columns),
                             loc="center", cellLoc="center")
                t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, 1.3)
            fig.tight_layout()
            return fig

        with PdfPages(path) as pdf:
            fig = plt.figure(figsize=(11, 8.5)); fig.patch.set_facecolor("white")
            fig.text(0.06, 0.9, f"Relatório — ModelSegmenter ({self.task_type})", fontsize=20,
                     fontweight="bold", color="#15324a")
            info = (f"algoritmo: {self.algorithm}     ·     alvo: {self.target}\n"
                    f"amostra de referência: {self.ref_sample}\n"
                    f"variáveis no modelo ({len(feats)}):\n{', '.join(feats) or '—'}")
            fig.text(0.06, 0.8, info, fontsize=12, color="#33424f", va="top")
            pdf.savefig(fig); plt.close(fig)
            try:
                f = _table_fig(self.metrics(), "Métricas por amostra"); pdf.savefig(f); plt.close(f)
            except Exception:
                plt.close("all")
            if self.algorithm in ("logistica", "linear"):
                try:
                    co = self.model_coefficients()
                    f = _table_fig(co, "Fórmula — coeficientes"); pdf.savefig(f); plt.close(f)
                except Exception:
                    plt.close("all")
            else:
                try:
                    f = self.plot_shap_bar(sample_size=800); pdf.savefig(f); plt.close(f)
                except Exception:
                    plt.close("all")
            if self.rating_strategy is not None:
                try:
                    f = _table_fig(self.rating_table(), "Régua de ratings"); pdf.savefig(f); plt.close(f)
                except Exception:
                    plt.close("all")
        return path

    @staticmethod
    def _df_to_md(df: pd.DataFrame) -> str:
        """DataFrame → tabela Markdown (GFM), sem depender de `tabulate`."""
        def cell(v):
            try:
                if pd.isna(v):                  # cobre None, NaN, pd.NA, pd.NaT
                    return "—"
            except (TypeError, ValueError):
                pass
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v).replace("|", "\\|")
        cols = list(df.columns)
        linhas = ["| " + " | ".join(str(c) for c in cols) + " |",
                  "| " + " | ".join("---" for _ in cols) + " |"]
        for _, r in df.iterrows():
            linhas.append("| " + " | ".join(cell(r[c]) for c in cols) + " |")
        return "\n".join(linhas)

    def report_markdown(self, path: str = "relatorio_modelo.md",
                        time_col: str | None = None, title: str | None = None,
                        stamp: str | None = None) -> str:
        """Gera um relatório do modelo em **Markdown** (.md) e o devolve — alternativa
        ao :meth:`report_pdf` em formato de texto versionável (abre no Jupyter, VS Code
        e GitHub).

        Seções: visão geral (algoritmo, variáveis, hiperparâmetros), métricas por
        amostra, fórmula (logística/linear) ou importância SHAP (não-lineares),
        discriminação & calibração, régua de ratings, PSI dos ratings, monotonicidade
        e backtest por safra (se ``time_col``/``date_col``). As imagens são salvas como
        PNG ao lado do .md e referenciadas por caminho relativo. Não exige dependência
        extra (usa matplotlib)."""
        import os
        if self.score_ is None:
            raise RuntimeError("Ajuste o modelo antes (fit / set_model / load).")
        import matplotlib.pyplot as plt

        base = os.path.dirname(os.path.abspath(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        if title is None:
            title = f"Relatório do Modelo — {self.algorithm} ({self.task_type})"

        def _fig(maker):
            try:
                return maker()
            except Exception:
                plt.close("all")
                return None

        def _save(fig, name):
            """Salva a figura ao lado do .md; devolve o nome do arquivo (ou None)."""
            if fig is None:
                return None
            try:
                fig.savefig(os.path.join(base, name), dpi=150, bbox_inches="tight")
                return name
            except Exception:
                return None
            finally:
                plt.close(fig)

        feats = list(self.model_features or self.selected_features() or self.candidates)
        L = [f"# {title}", ""]
        if stamp:
            L.append(f"_Gerado em {stamp}._\n")
        L += ["## Visão geral", "",
              f"- **Algoritmo:** `{self.algorithm}`",
              f"- **Tarefa:** `{self.task_type}`",
              f"- **Target:** `{self.target}`",
              f"- **Amostra de referência:** `{self.ref_sample}`"
              + ("" if self.sample_col is None else f" (coluna `{self.sample_col}`)"),
              f"- **Variáveis no modelo ({len(feats)}):** "
              + (", ".join(f"`{f}`" for f in feats) or "(nenhuma)"),
              f"- **Linhas:** {len(self.df):,}".replace(",", "."), ""]
        if self.hyperparams:
            L += ["**Hiperparâmetros:** "
                  + ", ".join(f"`{k}={v}`" for k, v in self.hyperparams.items()), ""]

        try:
            L += ["## Métricas por amostra", "", self._df_to_md(self.metrics()), ""]
        except Exception as e:
            L += ["## Métricas por amostra", "", f"_não geradas: {e}_", ""]

        if self.algorithm in ("logistica", "linear"):
            try:
                frm = self.model_formula()
                L += ["## Fórmula do modelo", "",
                      "```", frm["text"], "```", "",
                      "Coeficientes (ordenados por |coef|):", "",
                      self._df_to_md(frm["coef"]),
                      f"\n_Intercepto:_ `{frm['intercept']:+.6f}`", ""]
            except Exception as e:
                L += ["## Fórmula do modelo", "", f"_não gerada: {e}_", ""]
        else:
            L += ["## Importância das variáveis (SHAP)", ""]
            try:
                L += [self._df_to_md(self.shap_importance(sample_size=800)), ""]
            except Exception as e:
                L += [f"_tabela SHAP não gerada: {e}_", ""]
            img = _save(_fig(lambda: self.plot_shap_bar(sample_size=800)),
                        f"{stem}_shap.png")
            if img:
                L += [f"![shap]({img})", ""]

        if self.task_type == "classification":
            imgs = [("Curva ROC", _save(_fig(self.plot_roc), f"{stem}_roc.png")),
                    ("Curva KS", _save(_fig(self.plot_ks), f"{stem}_ks.png")),
                    ("Calibração", _save(_fig(self.plot_calibration), f"{stem}_calibracao.png"))]
        else:
            imgs = [("Resíduos", _save(_fig(self.plot_residuals), f"{stem}_residuos.png")),
                    ("Calibração", _save(_fig(self.plot_calibration), f"{stem}_calibracao.png"))]
        imgs = [(t, n) for t, n in imgs if n]
        if imgs:
            L += ["## Discriminação & calibração", ""]
            for t, n in imgs:
                L += [f"**{t}**", "", f"![{t}]({n})", ""]

        if self.rating_strategy is not None:
            try:
                L += ["## Régua de ratings", "", self._df_to_md(self.rating_table()), ""]
            except Exception as e:
                L += ["## Régua de ratings", "", f"_não gerada: {e}_", ""]
            if self.sample_col is not None:
                try:
                    L += ["### PSI dos ratings (estabilidade entre amostras)", "",
                          self._df_to_md(self.psi()), ""]
                except Exception:
                    pass
            try:
                L += ["### Monotonicidade do risco por rating", "",
                      self._df_to_md(self.monotonicity_report()), ""]
            except Exception:
                pass

        if time_col is not None or self.date_col is not None:
            try:
                L += ["## Backtest por safra (previsto × realizado no tempo)", "",
                      self._df_to_md(self.backtest(time_col)), ""]
            except Exception as e:
                L += ["## Backtest por safra", "", f"_não gerado: {e}_", ""]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(L))
        return path

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    def log_to_mlflow(self, experiment=None, run_name=None, registered_model_name=None,
                      artifact_path="modelo", registry_uri=None, save_base=False,
                      save_scored=False, verbose=True):
        """Registra o modelo (mlflow.sklearn), métricas por amostra, a régua de
        ratings e os gráficos SHAP como artefatos. Best-effort.

        ``save_base`` loga as amostras DES/OOT cruas; ``save_scored`` loga as
        mesmas amostras JÁ ESCORADAS (colunas ``score`` e ``rating`` via
        :meth:`assign` — nada é re-escorado) como ``base_*_escorada``."""
        import os
        import tempfile
        import mlflow
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        if experiment:
            mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({"task_type": self.task_type, "algorithm": self.algorithm,
                               "n_features": len(self.model_features),
                               "ref_sample": self.ref_sample,
                               **{f"hp_{k}": v for k, v in self.hyperparams.items()},
                               **{f"rating_{k}": v for k, v in self.rating_config.items()}})
            try:
                for _, r in self.metrics().iterrows():
                    for c, v in r.items():
                        if c == "amostra" or not (
                                isinstance(v, (int, float, np.integer, np.floating))
                                and np.isfinite(v)):
                            continue
                        mlflow.log_metric(f"{r['amostra']}_{c}", float(v))
            except Exception:
                pass
            try:
                import mlflow.sklearn
                # cloudpickle: compatível com mlflow 2.9→3.x (o 3.x passou a usar
                # 'skops' por padrão, que rejeita numpy.dtype e quebra RF/GBM).
                mlflow.sklearn.log_model(self.model, artifact_path,
                                         registered_model_name=registered_model_name,
                                         serialization_format="cloudpickle")
            except Exception as e:
                if verbose:
                    print(f"[mlflow] modelo não logado: {e}")
            with tempfile.TemporaryDirectory() as d:
                try:
                    self.rating_table().to_csv(os.path.join(d, "ratings.csv"), index=False)
                    mlflow.log_artifact(os.path.join(d, "ratings.csv"), "regua")
                except Exception:
                    pass
                try:
                    from ...interpretability.shap_explain import shap_report
                    est, Xt, names = self._shap_inputs()
                    shap_report(est, Xt, names, self.task_type, d, sample_size=None)
                    for fn in ("shap_beeswarm.png", "shap_importance_bar.png",
                               "shap_importance.csv"):
                        fp = os.path.join(d, fn)
                        if os.path.exists(fp):
                            mlflow.log_artifact(fp, "shap")
                except Exception as e:
                    if verbose:
                        print(f"[mlflow] SHAP não logado: {e}")
            # relatório em abas (Resumo/Métricas/Estabilidade) + base opcional
            try:
                from .._mlflow_report import log_tabbed_report
                m_df = self.metrics()
                p_df = self.psi() if self.sample_col is not None else None
                val_sample = self._oot_sample() if self.sample_col is not None else None
                stab = []
                if p_df is not None:
                    stab.append(("PSI dos ratings por amostra", p_df))
                try:
                    stab.append(("Ratings", self.rating_table()))
                except Exception:
                    pass
                dev_df = oot_df = None
                if save_base and self.sample_col is not None:
                    dev_df = self._frame(self.ref_sample)
                    if val_sample is not None:
                        oot_df = self._frame(val_sample)
                sc_dev = sc_oot = None
                if save_scored and self.sample_col is not None:
                    # base escorada SEM re-escorar: score_/rating_ já ficaram
                    # prontos no fit — assign() só anexa as colunas
                    scored = self.assign()
                    sc_dev = scored[self._frame_mask(self.ref_sample)]
                    if val_sample is not None:
                        sc_oot = scored[self._frame_mask(val_sample)]
                log_tabbed_report(
                    mlflow, run, title=f"ModelSegmenter — {self.task_type}",
                    subtitle=(f"alvo '{self.target}' · algoritmo {self.algorithm} · "
                              f"ref. {self.ref_sample}"),
                    val_sample=val_sample, metrics_df=m_df, psi_df=p_df,
                    stability_blocks=stab, save_base=save_base,
                    dev_df=dev_df, oot_df=oot_df,
                    save_scored=save_scored, scored_dev_df=sc_dev,
                    scored_oot_df=sc_oot, verbose=verbose)
            except Exception as e:  # pragma: no cover
                if verbose:
                    print(f"[mlflow] relatório em abas não gerado: {type(e).__name__}: {e}")
            if verbose:
                print(f"[mlflow] run_id = {run.info.run_id}")
            return run.info.run_id


def _cmap(name):
    """Colormap sem depender de pyplot estar configurado."""
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except Exception:  # pragma: no cover - matplotlib < 3.6
        import matplotlib.cm as cm
        return cm.get_cmap(name)
