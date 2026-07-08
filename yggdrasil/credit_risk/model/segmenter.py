"""
ModelSegmenter
==============
Segmentador **orientado a modelo** para risco de crédito, unificando
**classificação** (ex.: PD) e **regressão** (ex.: LGD) num único objeto via o
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
   RMSE/MAE/R² na regressão) e **SHAP** do modelo criado.
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

import numpy as np
import pandas as pd

from ...config import ColumnConfig
from ...metrics import classification_metrics, regression_metrics
from ...ratings import RATING_REGISTRY

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
def _fmt(x: float) -> str:
    """Formata limites de faixa de forma legível."""
    if x == -np.inf:
        return "-inf"
    if x == np.inf:
        return "inf"
    return f"{x:.4g}"


def _classifica_psi(psi: float) -> str:
    """Classificação usual de PSI para monitoramento de estabilidade."""
    if psi is None or (isinstance(psi, float) and np.isnan(psi)):
        return "—"
    if psi < 0.10:
        return "estável"
    if psi < 0.25:
        return "atenção"
    return "instável"


def _classifica_iv(iv, task_type: str) -> str:
    """Força do IV. Na **classificação** usa a escala WoE/IV de Siddiqi
    (0.02/0.1/0.3/0.5); na **regressão** usa a escala do IV contínuo (desvio
    absoluto médio do alvo por faixa: 0.01/0.03/0.10/0.35), bem menor."""
    if iv is None or (isinstance(iv, float) and np.isnan(iv)):
        return "—"
    if task_type == "classification":
        if iv < 0.02:
            return "inútil"
        if iv < 0.10:
            return "fraco"
        if iv < 0.30:
            return "médio"
        if iv < 0.50:
            return "forte"
        return "suspeito"
    if iv < 0.01:
        return "inútil"
    if iv < 0.03:
        return "fraco"
    if iv < 0.10:
        return "médio"
    if iv < 0.35:
        return "forte"
    return "suspeito"


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
    n_inv = int((np.sign(diffs[:-1]) != np.sign(diffs[1:])).sum()) if len(diffs) > 1 else 0
    return trend, n_inv


def _count_inversions(ordered, values) -> tuple:
    """Nº de pares invertidos vs. a ordem de referência e nº de pares comparáveis.
    `ordered` = lista de chaves na ordem de risco de referência (crescente);
    `values` = dict chave->risco num ponto (amostra/safra). Par (i<j na ref.)
    inverte quando risco_i > risco_j."""
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


def _fit_optbinning_splits(b, x, y) -> list:
    """Roda ``b.fit(x, y)`` e devolve ``list(b.splits)``.

    Silencia os ``RuntimeWarning`` de "divide by zero" benignos do optbinning
    (em ``auto_monotonic``, quando algum prebin fica com 0 registros) — o ajuste
    ainda produz cortes válidos. Devolve ``[]`` se o ajuste falhar.

    ``ValueError`` (problema inviável / sem corte) é o caminho esperado e fica
    silencioso. Qualquer outra exceção (ex.: incompatibilidade de versão de
    dependência) é **avisada** em vez de mascarada como "sem corte válido".
    """
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


def _is_stability_sample(name) -> bool:
    """Heurística: a *safra de estabilidade* é a amostra cujo nome remete a
    estabilidade (ex.: ``ESTABILIDADE``, ``ESTAB``) — convenção do repositório
    (ver PSI por rating: DES × OOT e ESTABILIDADE)."""
    return "estab" in str(name).strip().lower()


def _fmt_safras(safras) -> list:
    """Rótulos de safra → 'mmm/aa' (padrão de mês/ano do repositório).

    Delega ao helper único :func:`yggdrasil.reporting.style.fmt_month_year`."""
    from ...reporting.style import fmt_month_year
    return fmt_month_year(safras)


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


def _build_estimator(algorithm: str, task_type: str, hyperparams: dict | None,
                     random_state: int | None = None):
    """Instancia o estimador do algoritmo escolhido (registry extensível).

    sklearn é sempre disponível; LightGBM/XGBoost/CatBoost são pacotes opcionais
    importados sob demanda (ver :func:`_require`).

    ``random_state`` semeia os estimadores estocásticos (florestas/boosting) para
    reprodutibilidade; ``None`` cai no default histórico 42. Via ``setdefault``, uma
    seed explícita em ``hyperparams`` (usuário/Optuna) vence. Logística/linear não
    aceitam seed (não são estocásticas); CatBoost usa ``random_seed``."""
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

    if algorithm == "logistica":
        from sklearn.linear_model import LogisticRegression
        hp.setdefault("max_iter", 1000)
        return LogisticRegression(**hp)
    if algorithm == "linear":
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**hp)
    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        RF = RandomForestClassifier if is_clf else RandomForestRegressor
        hp.setdefault("n_estimators", 200)
        hp.setdefault("random_state", seed)
        return RF(**hp)
    if algorithm == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        ET = ExtraTreesClassifier if is_clf else ExtraTreesRegressor
        hp.setdefault("n_estimators", 200)
        hp.setdefault("random_state", seed)
        return ET(**hp)
    if algorithm == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        GB = GradientBoostingClassifier if is_clf else GradientBoostingRegressor
        hp.setdefault("random_state", seed)
        return GB(**hp)
    if algorithm == "hist_gradient_boosting":
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      HistGradientBoostingRegressor)
        HGB = HistGradientBoostingClassifier if is_clf else HistGradientBoostingRegressor
        if "n_estimators" in hp:                       # nome unificado na UI → max_iter
            hp["max_iter"] = hp.pop("n_estimators")
        hp.setdefault("random_state", seed)
        return HGB(**hp)
    if algorithm == "lightgbm":
        lgb = _require("lightgbm", algorithm)
        Est = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor
        hp.setdefault("n_estimators", 300)
        hp.setdefault("learning_rate", 0.05)
        hp.setdefault("random_state", seed)
        hp.setdefault("verbose", -1)
        return Est(**hp)
    if algorithm == "xgboost":
        xgb = _require("xgboost", algorithm)
        Est = xgb.XGBClassifier if is_clf else xgb.XGBRegressor
        hp.setdefault("n_estimators", 300)
        hp.setdefault("learning_rate", 0.05)
        hp.setdefault("random_state", seed)
        hp.setdefault("verbosity", 0)
        hp.setdefault("tree_method", "hist")
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
        hp.setdefault("iterations", 300)
        hp.setdefault("learning_rate", 0.05)
        hp.setdefault("random_seed", seed)
        hp.setdefault("verbose", False)
        hp.setdefault("allow_writing_files", False)     # não polui o diretório
        return Est(**hp)
    raise ValueError(algorithm)  # pragma: no cover


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
    (no transformador serializável)."""
    if b["kind"] == "na":
        return series.isna()
    if b["kind"] == "num":
        return series.between(b["lo"], b["hi"], inclusive="right")
    return series.astype(str).isin(b["cats"])


class WoeBinEncoder(BaseEstimator, TransformerMixin):
    """Transforma cada variável no **valor do seu bin** — WoE (classificação) ou
    risco médio do bin (regressão) — usando bins/grupos já ajustados na amostra de
    referência (faixas para contínuas, grupos para categóricas, como nas árvores
    de PD/LGD). Serve para alimentar os modelos com variáveis transformadas, no
    estilo *scorecard*.

    ``encodings``: ``{feature: {"kind", "bins": [(bin_dict, valor), ...],
    "fallback": float}}``. Valores fora de qualquer bin (categoria nova, faltante
    sem bin próprio) recebem ``fallback`` (0 = WoE neutro)."""

    def __init__(self, encodings=None, features=None, name_prefix="WoE"):
        self.encodings = encodings
        self.features = features
        self.name_prefix = name_prefix

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        feats = self.features or []
        return np.asarray([f"{self.name_prefix}({f})" for f in feats], dtype=object)

    def transform(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        feats = self.features or []
        out = np.empty((len(X), len(feats)), dtype="float64")
        for j, f in enumerate(feats):
            enc = self.encodings[f]
            col = X[f]
            vals = np.full(len(X), enc["fallback"], dtype="float64")
            assigned = np.zeros(len(X), dtype=bool)
            for b, v in enc["bins"]:
                m = _bin_mask_series(col, b).to_numpy() & ~assigned
                vals[m] = v
                assigned |= m
            out[:, j] = vals
        return out


class _TwoStageModel:
    """Modelo *hurdle* de duas etapas para regressão (típico de LGD).

    Combina um **classificador** que estima ``P(y ≥ threshold)`` com uma
    **regressão** treinada apenas no grupo ``y ≥ threshold``; a resposta final é
    o valor esperado

        E[y | x] = P(≥t|x)·reg(x) + (1 − P(≥t|x))·âncora₀,

    onde ``âncora₀`` é a média observada do grupo abaixo do threshold (≈ 0 em
    LGD). Expõe ``predict`` (a resposta combinada) para se comportar como
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

    def __init__(
        self,
        df: pd.DataFrame,
        target: str = "target",
        task_type: str = "classification",
        sample_col: str | None = None,
        ref_sample: str = "DES",
        feature_labels: dict[str, str] | None = None,
        features: list | None = None,
        date_col: str | None = None,
        verbose: bool = True,
        score_scale: float = 1000.0,
        random_state: int | None = 42,
    ):
        if task_type not in ("classification", "regression"):
            raise ValueError("task_type deve ser 'classification' ou 'regression'.")
        if target not in df.columns:
            raise ValueError(f"Alvo '{target}' não está no DataFrame.")

        self.df = df.copy()
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
        self.target = target
        self.task_type = task_type
        self.sample_col = sample_col
        self.ref_sample = ref_sample
        self.date_col = date_col
        self.feature_labels = feature_labels or {}

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

        # estado de modelo / score / rating
        self.model = None
        self.algorithm: str | None = None
        self.hyperparams: dict = {}
        self.feature_transform: str = "raw"   # "raw" | "woe" (binagem + WoE/risco do bin)
        self.model_features: list = []
        # modelo Two-Stage (hurdle de LGD): classificação P(y≥t) + regressão em
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
        self.rating_strategy = None
        self.rating_col_: str | None = None
        self.rating_: pd.Series | None = None
        self.rating_labels_: list = []
        self.rating_config: dict = {}
        self._shap_cache: dict = {}
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

    def _frame(self, sample=None) -> pd.DataFrame:
        """Recorte do df por amostra (default DES quando há sample_col).

        Memoiza a máscara booleana por amostra (a comparação numa coluna de objeto
        é cara e se repete centenas de vezes); devolve sempre uma cópia fresca
        ``df[mask]`` — segura contra mutação. Linhas e ``sample_col`` não mudam
        após a construção, então a máscara nunca precisa ser invalidada."""
        if self.sample_col is None:
            return self.df
        if sample is None:
            sample = self.ref_sample
        return self.df[self._frame_mask(sample)]

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

    def label(self, feature) -> str:
        return self.feature_labels.get(feature, feature)

    # ------------------------------------------------------------------
    # Binning de uma variável (classificação OU regressão)
    # ------------------------------------------------------------------
    def _mask_in(self, frame, feature, b):
        if b["kind"] == "na":
            return frame[feature].isna()
        if b["kind"] == "num":
            return frame[feature].between(b["lo"], b["hi"], inclusive="right")
        return frame[feature].astype(str).isin(b["cats"])

    def _bin_label(self, feature, b) -> str:
        if b["kind"] == "na":
            return "(faltante)"
        if b["kind"] == "num":
            return f"({_fmt(b['lo'])}, {_fmt(b['hi'])}]"
        return "{" + ", ".join(map(str, b["cats"])) + "}"

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
        ck = (feature, max_n_bins, min_bin_size, sample_key, self._splits_key(eff_splits))
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
        fit = self._frame(sample)
        kind = self._detect_kind(feature, fit)

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
                if len(y) < 4 or x_obs.size == 0 or np.unique(x_obs).size < 2:
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
            return bins, kind

        # categórico
        na_present = bool(fit[feature].isna().any())
        if splits is not None:
            grupos = [list(g) for g in splits]
        else:
            f2 = fit[fit[feature].notna() & fit[self.target].notna()]
            xs = f2[feature].astype(str).to_numpy()
            ys = f2[self.target].to_numpy(dtype="float64")
            if len(ys) < 4 or np.unique(xs).size < 2:
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
        return bins, kind

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
        sub = self._frame(sample)
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

        rows, iv_total = [], 0.0
        for b in bins:
            m = self._mask_in(sub, feature, b).to_numpy()
            yi = y_all[m]
            yi_ok = yi[~np.isnan(yi)]
            n_i = int(m.sum())
            if n_i == 0:
                continue
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
        rcol = out[risco_label] if risco_label in out else pd.Series(dtype=float)
        mono = bool(rcol.is_monotonic_increasing or rcol.is_monotonic_decreasing) \
            if len(rcol) else True
        out.attrs.update(iv=round(float(iv_total), 4), mono_ok=mono, kind=kind,
                         risco_label=risco_label, mean_global=round(mean_global, 4))
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
        rows = []
        for feat in features:
            iv, nb, kind, trend, n_inv = np.nan, 0, "—", "—", 0
            psi_vals = {a: np.nan for a in nonref}
            try:
                vt = self.variable_table(feat, sample=sample, max_n_bins=max_n_bins,
                                         min_bin_size=min_bin_size)
                kind = vt.attrs.get("kind", "—")
                nb = len(vt)
                iv = vt.attrs.get("iv", np.nan)
                if nb:
                    rcol = vt.attrs.get("risco_label")
                    trend, n_inv = _trend(vt[rcol].tolist())
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
        ref = self._frame(self.ref_sample)
        n_ref = max(len(ref), 1)
        ref_pct = [max(int(self._mask_in(ref, feature, b).sum()) / n_ref, eps) for b in bins]
        for a in samples:
            cur = self._frame(a)
            n_cur = len(cur)
            if n_cur == 0:
                continue
            psi = 0.0
            for b, p_ref in zip(bins, ref_pct):
                p_cur = max(int(self._mask_in(cur, feature, b).sum()) / n_cur, eps)
                psi += (p_cur - p_ref) * np.log(p_cur / p_ref)
            out[a] = round(float(psi), 4)
        return out

    def variable_summary(self, feature, sample=None) -> dict:
        """Resumo de uma variável: %missing, estatísticas/top-categorias, IV,
        força, tendência e PSI por amostra."""
        sub = self._frame(sample)
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
        sub = self.df if all_samples else self._frame(sample)
        safra = pd.to_datetime(sub[time_col], errors="coerce").dt.to_period("M")
        rows = []
        for per, g in sub.groupby(safra):
            if pd.isna(per):        # data não parseável (NaT) — não vira safra "NaT"
                continue
            col = g[feature]
            x = col.to_numpy(dtype="float64"); x = x[~np.isnan(x)]
            n = int(len(col)); n_miss = int(col.isna().sum())
            row = {"safra": str(per), "n": n,
                   "pct_missing": round(100 * n_miss / n, 1) if n else float("nan")}
            if x.size:
                row.update(min=round(float(np.min(x)), 3),
                           p5=round(float(np.percentile(x, 5)), 3),
                           media=round(float(np.mean(x)), 3),
                           p95=round(float(np.percentile(x, 95)), 3),
                           max=round(float(np.max(x)), 3))
            else:
                row.update({k: float("nan") for k in ("min", "p5", "media", "p95", "max")})
            rows.append(row)
        return pd.DataFrame(rows).sort_values("safra").reset_index(drop=True)

    def variable_share_by_safra(self, feature, time_col=None, sample=None, top=8,
                                all_samples=False) -> pd.DataFrame:
        """Representatividade (%) de cada categoria por safra (variável CATEGÓRICA).
        ``all_samples=True`` usa todas as safras da base (todas as amostras)."""
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col.")
        sub = self.df if all_samples else self._frame(sample)
        safra = pd.to_datetime(sub[time_col], errors="coerce").dt.to_period("M").astype(str)
        cat = sub[feature]
        keep = list(cat.dropna().astype(str).value_counts().head(top).index)

        def lab(v):
            if pd.isna(v):
                return "(faltante)"
            s = str(v)
            return s if s in keep else "outras"

        tab = pd.crosstab(safra, cat.map(lab))
        tab = tab[tab.index != "NaT"]
        if tab.empty:
            return pd.DataFrame(columns=["safra"])
        pct = tab.div(tab.sum(axis=1), axis=0) * 100
        order = [c for c in keep if c in pct.columns]
        if "outras" in pct.columns:
            order.append("outras")
        if "(faltante)" in pct.columns:
            order.append("(faltante)")
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
        ref = self._frame(self.ref_sample)
        n_ref = max(len(ref), 1)
        ref_pct = [max(int(self._mask_in(ref, feature, b).sum()) / n_ref, eps) for b in bins]
        safra = pd.to_datetime(self.df[time_col], errors="coerce").dt.to_period("M")
        rows = []
        for per, g in self.df.groupby(safra):
            n_g = len(g)
            if n_g == 0:
                continue
            psi = 0.0
            for b, p_ref in zip(bins, ref_pct):
                p_cur = max(int(self._mask_in(g, feature, b).sum()) / n_g, eps)
                psi += (p_cur - p_ref) * np.log(p_cur / p_ref)
            rows.append({"safra": str(per), "n": n_g, "psi": round(float(psi), 4),
                         "classificacao": _classifica_psi(psi)})
        return pd.DataFrame(rows).sort_values("safra").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Inversão da ordem de risco dos BINS de uma variável (entre amostras/safras)
    # ------------------------------------------------------------------
    def _variable_bin_series(self, feature, bins, time_col=None, sample=None,
                             min_n=20):
        """Risco de cada bin por amostra e por safra. Devolve dict com chaves
        ``ordered`` (bins na ordem de risco DES), ``labels``, ``samples`` (xs +
        series por bin) e ``safras`` (xs + series por bin)."""
        labels = [self._bin_label(feature, b) for b in bins]
        ref = self._frame(self.ref_sample)
        ref_risco = [self._risco(ref.loc[self._mask_in(ref, feature, b), self.target])
                     for b in bins]
        order = sorted(range(len(bins)),
                       key=lambda i: (np.inf if pd.isna(ref_risco[i]) else ref_risco[i]))

        # por amostra
        xs_s = self._samples()
        ser_s = {i: [] for i in range(len(bins))}
        for a in xs_s:
            fa = self._frame(a)
            for i, b in enumerate(bins):
                ser_s[i].append(self._risco(fa.loc[self._mask_in(fa, feature, b), self.target]))

        # por safra
        xs_t, ser_t = [], {i: [] for i in range(len(bins))}
        tcol = time_col or self.date_col
        if tcol is not None and tcol in self.df.columns:
            base = self._frame(sample) if sample else self.df
            safra = pd.to_datetime(base[tcol], errors="coerce").dt.to_period("M")
            for per, g in base.groupby(safra):
                if len(g) < min_n:
                    continue
                xs_t.append(str(per))
                for i, b in enumerate(bins):
                    ser_t[i].append(self._risco(g.loc[self._mask_in(g, feature, b), self.target]))
        return {"ordered": order, "labels": labels, "ref_risco": ref_risco,
                "xs_sample": xs_s, "ser_sample": ser_s,
                "xs_safra": xs_t, "ser_safra": ser_t}

    def variable_inversion(self, feature, time_col=None, sample=None,
                           max_n_bins=6, min_bin_size=0.05, min_n=20) -> dict:
        """Diagnóstico de inversão da ordem de risco dos bins de uma variável,
        entre amostras e entre safras. Veredito verde/amarelo/vermelho — análoga
        à inversão entre folhas-irmãs do PD/LGD, mas sobre os bins de UMA variável."""
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
                              figsize=(7.6, 3.4), dpi=150, save_path=None, ax=None):
        """Barras de representatividade (%) + linha de **logodds/WoE** (classificação)
        ou **alvo médio** (regressão) por faixa — leitura de monotonicidade."""
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
            sub = self._frame(sample)
            x = sub[feature].to_numpy(dtype="float64"); x = x[~np.isnan(x)]
            if x.size:
                ax.hist(x, bins=20, color="steelblue", alpha=0.85, edgecolor="#2f5d82")
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
                                           save_path=None, ax=None):
        """Um único gráfico: barras de **distribuição** (% da amostra por faixa) +
        linha do risco por faixa — **% de maus** (event_rate) na classificação ou
        **alvo médio** na regressão. Faltantes destacados."""
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
        if finite.size:
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
        sub = self.df if all_samples else self._frame(sample)
        if time_col not in sub.columns:
            raise ValueError(f"Coluna de tempo '{time_col}' não existe no DataFrame.")
        if bins is None:
            bins, _kind = self._resolve_bins(feature, max_n_bins, min_bin_size, None, sample)
        safra = pd.to_datetime(sub[time_col], errors="coerce").dt.to_period("M").astype(str)
        faixa = pd.Series("(faltante)", index=sub.index, dtype=object)
        ordem = []
        for b in bins or []:
            lab = self._bin_label(feature, b)
            ordem.append(lab)
            m = self._mask_in(sub, feature, b).to_numpy()
            faixa.loc[sub.index[m]] = lab
        tmp = pd.DataFrame({"safra": safra.to_numpy(), "faixa": faixa.to_numpy()})
        tmp = tmp[tmp["safra"] != "NaT"]
        if tmp.empty or not ordem:
            return pd.DataFrame(columns=["safra"])
        tab = pd.crosstab(tmp["safra"], tmp["faixa"])
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

    def _optbin_numeric_bins(self, feature, sample=None, max_n_bins=5, min_bin_size=0.05,
                             all_samples=False):
        """Faixas do **OPTIMAL BINNING** de uma variável NUMÉRICA — sempre roda o
        optbinning na amostra de ajuste, IGNORANDO eventuais bins manuais. Retorna
        a lista de bins numéricos (+ ``na`` se houver faltantes) ou ``[]`` quando
        não dá para binar. (:meth:`_resolve_bins` respeita bins manuais; este não.)

        ``all_samples=True`` ajusta o binning em **toda a base** (todas as amostras),
        não só na amostra de referência."""
        if OptimalBinning is None:
            raise ImportError("optbinning não instalado. Rode: pip install optbinning")
        fit = self.df if all_samples else self._frame(sample)
        if self._detect_kind(feature, fit) != "num":
            return []
        x = fit[feature].to_numpy(dtype="float64")
        y = fit[self.target].to_numpy(dtype="float64")
        ok = ~np.isnan(y)
        x, y = x[ok], y[ok]
        x_obs = x[~np.isnan(x)]
        if len(y) < 4 or x_obs.size == 0 or np.unique(x_obs).size < 2:
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
        if self._detect_kind(feature, self._frame(sample)) != "num":
            fig, ax = _new_ax((figsize[0], 3.6), dpi, ax)
            ax.text(0.5, 0.5, "apenas para variáveis numéricas", ha="center",
                    va="center", transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        bins = self._optbin_numeric_bins(feature, sample, max_n_bins, min_bin_size)
        titulo = f"'{self.label(feature)}' — faixas do optimal binning ao longo do tempo"
        return self.plot_variable_faixa_share_timeseries(
            feature, time_col, sample, max_n_bins, min_bin_size, figsize, dpi,
            save_path, ax, bins=bins, titulo=titulo, legend_title="faixa (optbin)")

    def plot_variable_optbin_cumshare_timeseries(self, feature, time_col=None, sample=None,
                                                 max_n_bins=5, min_bin_size=0.05,
                                                 figsize=(11.5, 4.2), dpi=150,
                                                 save_path=None, ax=None, all_samples=False):
        """**Distribuição ACUMULADA das faixas do OPTIMAL BINNING ao longo do tempo**
        (só variáveis NUMÉRICAS): área EMPILHADA das %s de cada faixa por safra, da
        **primeira faixa (base) até a última (topo)**, somando 100%. Deixa ver como
        a composição da variável migra entre as faixas no tempo. Sempre usa o
        optbinning (ignora bins manuais).

        ``all_samples=True`` (padrão na UI) calcula sobre **toda a base** — todas as
        amostras e safras —, não só na DES/amostra de referência."""
        import matplotlib.colors as mcolors
        if self._detect_kind(feature, self.df if all_samples else self._frame(sample)) != "num":
            fig, ax = _new_ax(figsize, dpi, ax)
            ax.text(0.5, 0.5, "apenas para variáveis numéricas", ha="center",
                    va="center", transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        bins = self._optbin_numeric_bins(feature, sample, max_n_bins, min_bin_size,
                                         all_samples=all_samples)
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
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.9, title="faixa (optbin)")
        ax.set_title(f"'{self.label(feature)}' — distribuição acumulada das faixas do "
                     f"optimal binning ao longo do tempo", fontsize=11,
                     fontweight="bold", color="#15324a")
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
                                          figsize=(7.6, 4.0), dpi=150, save_path=None, ax=None):
        """Risco de cada faixa por amostra; cruzamentos = inversão da ordem de risco."""
        inv = self.variable_inversion(feature, max_n_bins=max_n_bins, min_bin_size=min_bin_size)
        fig, ax = _new_ax(figsize, dpi, ax)
        s = inv.get("series")
        if not s or not inv["ordered"]:
            ax.text(0.5, 0.5, "menos de 2 faixas", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        xs = s["xs_sample"]; x = list(range(len(xs)))
        cmap = _cmap("RdYlGn_r"); k = len(inv["ordered"])
        for rank, i in enumerate(inv["ordered"]):
            ax.plot(x, s["ser_sample"][i], marker="o", lw=1.9, ms=5.5,
                    color=cmap(rank / (k - 1) if k > 1 else 0.5),
                    markeredgecolor="#33424f", markeredgewidth=0.6,
                    label=s["labels"][i])
        ax.set_xticks(x); ax.set_xticklabels(xs, fontsize=9)
        ax.set_ylabel("risco médio"); ax.set_xlabel("amostra")
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
                                         figsize=(9.6, 4.0), dpi=150, save_path=None, ax=None):
        """Risco de cada faixa por safra; safras com inversão ficam sombreadas."""
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
        for rank, i in enumerate(ordered):
            ax.plot(x, s["ser_safra"][i], marker="o", lw=1.7, ms=4.5,
                    color=cmap(rank / (k - 1) if k > 1 else 0.5),
                    markeredgecolor="#33424f", markeredgewidth=0.5, label=s["labels"][i])
        ax.set_xticks(x); ax.set_xticklabels(_fmt_safras(xs), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("risco médio"); ax.set_xlabel("safra")
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
        """Comportamento da variável ao longo do tempo: o **risco** (event_rate/PD na
        classificação, alvo médio na regressão) de cada bin/categoria por safra.

        - **Numérica**: usa os mesmos bins do ranking (``n_bins``) — risco de cada
          faixa por safra.
        - **Categórica**: traz a **PD por categoria** por safra, sem reagrupar (top
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
        pers = sorted(p for p in safra.dropna().unique())
        if not pers:
            ax.text(0.5, 0.5, "sem dados por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        xs = [str(p) for p in pers]; x = list(range(len(xs)))
        series = []
        for label, gmask in groups:
            gm = gmask.to_numpy()
            ys = []
            for p in pers:
                m = gm & (safra == p).to_numpy()
                ys.append(self._risco(base.loc[m, self.target]) if int(m.sum()) >= min_n
                          else np.nan)
            series.append((label, ys))

        is_clf = self.task_type == "classification"
        ylabel = "PD" if is_clf else "alvo médio"
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

    def set_manual_bins(self, feature, spec):
        """Define **bins manuais** para a variável, sobrepondo o binning ótimo em
        toda a análise univariada (tabela, IV, logodds/WoE, PSI, inversão).

        ``spec`` pode ser o texto da UI (ver :meth:`_parse_bin_spec`), uma lista
        já parseada (cortes numéricos ou grupos categóricos), ou ``None``/``""``
        para limpar e voltar ao binning ótimo."""
        if feature not in self.candidates:
            raise ValueError(f"'{feature}' não é variável candidata.")
        splits = self._parse_bin_spec(feature, spec) if isinstance(spec, (str, type(None))) \
            else (list(spec) or None)
        meta = self.var_meta.setdefault(feature, {})
        if splits:
            meta["splits"] = splits
        else:
            meta.pop("splits", None)
        self._invalidate_bins(feature)   # só ESTA variável re-bina; demais ficam quentes
        return self

    def clear_manual_bins(self, feature):
        """Remove os bins manuais da variável (volta ao binning ótimo)."""
        self.var_meta.get(feature, {}).pop("splits", None)
        self._invalidate_bins(feature)   # só ESTA variável volta ao ótimo
        return self

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

    # ------------------------------------------------------------------
    # C) Modelo
    # ------------------------------------------------------------------
    def _bin_encoding(self, feature) -> dict:
        """Bins (ajustados na referência) + valor de codificação por bin: **WoE**
        (classificação) ou **risco médio do bin** (regressão). Reaproveita os bins
        manuais/ótimos da análise univariada (:meth:`_resolve_bins`). Usado pela
        transformação WoE que alimenta o modelo."""
        ref = self._frame(self.ref_sample)
        bins, kind = self._resolve_bins(feature, sample=self.ref_sample)
        y_all = ref[self.target].to_numpy(dtype="float64")
        enc_bins = []
        if self.task_type == "classification":
            n_evt_tot = float(np.nansum(y_all == 1))
            n_non_tot = float(np.nansum(y_all == 0))
            for b in bins:
                m = self._mask_in(ref, feature, b).to_numpy()
                yi = y_all[m]; yi = yi[~np.isnan(yi)]
                d_evt = float((yi == 1).sum()) / max(n_evt_tot, _EPS)
                d_non = float((yi == 0).sum()) / max(n_non_tot, _EPS)
                enc_bins.append((b, float(np.log((d_non + _EPS) / (d_evt + _EPS)))))
            fallback = 0.0   # WoE neutro p/ valores fora dos bins vistos na referência
        else:
            mean_global = self._risco(y_all)
            for b in bins:
                m = self._mask_in(ref, feature, b).to_numpy()
                r = self._risco(y_all[m])
                enc_bins.append((b, float(r) if np.isfinite(r) else mean_global))
            fallback = float(mean_global) if np.isfinite(mean_global) else 0.0
        return {"kind": kind, "bins": enc_bins, "fallback": fallback}

    def _build_pipeline(self, features, algorithm, hyperparams, transform="raw", task=None):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        # ``task`` permite forçar classificação/regressão (usado pelo Two-Stage,
        # que constrói um classificador e uma regressão a partir de um segmenter
        # de regressão); por padrão segue o task_type do segmenter.
        task = task or self.task_type
        est = _build_estimator(algorithm, task, hyperparams, random_state=self.random_state)
        if transform == "woe":
            # variáveis transformadas no estilo scorecard (binagem + WoE/risco do bin)
            encodings = {f: self._bin_encoding(f) for f in features}
            prefix = "WoE" if task == "classification" else "bin"
            pre = WoeBinEncoder(encodings=encodings, features=list(features),
                                name_prefix=prefix)
            return Pipeline([("pre", pre), ("est", est)])

        num = [f for f in features if self._detect_kind(f) == "num"]
        cat = [f for f in features if self._detect_kind(f) == "cat"]
        transformers = []
        if num:
            transformers.append(("num", SimpleImputer(strategy="median"), num))
        if cat:
            from sklearn.pipeline import Pipeline as P
            cat_pipe = P([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", _make_ohe())])
            transformers.append(("cat", cat_pipe, cat))
        pre = ColumnTransformer(transformers, remainder="drop")
        return Pipeline([("pre", pre), ("est", est)])

    def fit(self, algorithm=None, hyperparams=None, features=None, transform="raw"):
        """Treina um modelo na amostra de referência (DES) com as variáveis
        selecionadas (ou ``features``). ``algorithm`` default: logística
        (classificação) / linear (regressão). Calcula ``score_`` para todas as linhas.

        ``transform``: ``"raw"`` usa os valores originais (numéricas + one-hot das
        categóricas); ``"woe"`` transforma cada variável no WoE do seu bin
        (classificação) ou no risco médio do bin (regressão), reaproveitando os
        bins/grupos definidos na análise univariada — estilo *scorecard*."""
        if algorithm is None:
            algorithm = "logistica" if self.task_type == "classification" else "linear"
        feats = list(features) if features is not None else self.selected_features()
        if not feats:
            feats = list(self.candidates)
        self.model_features = feats
        self.algorithm = algorithm
        self.hyperparams = dict(hyperparams or {})
        self.feature_transform = transform
        self.two_stage = False                 # fit "normal" desliga o modo hurdle
        self.two_stage_threshold = None

        fit_df = self._frame(self.ref_sample)
        fit_df = fit_df[fit_df[self.target].notna()]
        X = fit_df[feats]
        y = fit_df[self.target]
        if self.task_type == "classification":
            y = y.astype(int)
        self.model = self._build_pipeline(feats, algorithm, hyperparams, transform=transform)
        self.model.fit(X, y)
        self.score_ = self._compute_score(self.df)
        self._shap_cache = {}
        return self

    def fit_two_stage(self, threshold, clf_algorithm="logistica", reg_algorithm="linear",
                      clf_hyperparams=None, reg_hyperparams=None, features=None,
                      transform="raw"):
        """Ajusta um modelo **Two-Stage (hurdle)** para regressão (LGD): binariza o
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

        fit_df = self._frame(self.ref_sample)
        fit_df = fit_df[fit_df[self.target].notna()]
        X = fit_df[feats]
        y = fit_df[self.target].astype(float)
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
        self.score_ = self._compute_score(self.df)
        self._shap_cache = {}
        self._metrics_cache = None
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
                    mlflow_model_name=None):
        """Otimização bayesiana de hiperparâmetros com **Optuna** (dependência
        core). Treina na referência (DES) e avalia no OOT (se houver alvo;
        senão, num split 75/25 do DES), maximizando **AUC** (classificação) ou
        **R²** (regressão). Guarda o resultado em ``self.tuning_`` (e o estudo em
        ``self.study_``); com ``fit_best=True`` reajusta o modelo com os melhores
        hiperparâmetros. Algoritmos tunáveis: :data:`TUNABLE_ALGORITHMS`.

        ``search_space`` (opcional): sobrescreve quais hiperparâmetros são
        buscados e seus intervalos — dict ``{nome: {type, low, high, log?, step?,
        choices?}}`` (ver :data:`OPTUNA_SEARCH_SPACE` e :func:`_optuna_space`).
        ``None`` usa o catálogo padrão do algoritmo.

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
        UI. Exceções no callback são ignoradas (não derrubam o tuning)."""
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

        tr = self._frame(self.ref_sample)
        tr = tr[tr[self.target].notna()]
        va = None
        oot = self._oot_sample()
        if oot and oot != self.ref_sample:
            vf = self._frame(oot)
            vf = vf[vf[self.target].notna()]
            if len(vf) >= 50:
                va = vf
        if va is None:                     # sem OOT com alvo → split do DES
            strat = tr[self.target].astype(int) if is_clf else None
            tr, va = train_test_split(tr, test_size=0.25, random_state=random_state,
                                      stratify=strat)
        ytr = tr[self.target].astype(int) if is_clf else tr[self.target]
        yva = va[self.target].astype(int) if is_clf else va[self.target]
        Xtr, Xva = tr[feats], va[feats]
        val_sample = oot if (oot and oot != self.ref_sample and va is not None) else "split"

        def objective(trial):
            from ...monitoring import psi as _psi_num
            hp = _optuna_space(trial, algorithm, search_space)
            pipe = self._build_pipeline(feats, algorithm, hp, transform=transform)
            pipe.fit(Xtr, ytr)
            s = self._predict_score_array(pipe, Xva)
            s_tr = self._predict_score_array(pipe, Xtr)
            # grupo MODELAGEM (qualidade na validação) + o valor-objetivo
            if is_clf:
                modelagem = classification_metrics(yva, s)
                valor = modelagem.get("auc", float("nan"))
            else:
                modelagem = regression_metrics(yva, s)
                valor = modelagem.get("r2", float("nan"))
            # grupo MONITORAMENTO (estabilidade/volumetria)
            monitoramento = {
                "psi_score_des_val": round(float(_psi_num(s_tr, s)), 6),
                "n_treino": int(len(Xtr)), "n_validacao": int(len(Xva)),
            }
            trial.set_user_attr("modelagem", modelagem)
            trial.set_user_attr("monitoramento", monitoramento)
            trial.set_user_attr("val_sample", val_sample)
            return float(valor) if np.isfinite(valor) else float("-1e9")

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        # NÃO limpamos a flag de cancelamento aqui: quando o tuning roda numa thread
        # de fundo (UI), a preparação acima leva tempo e o botão "Cancelar" já está
        # ativo — um clear aqui apagaria um cancelamento pedido nessa janela. O
        # chamador (UI) limpa a flag na main thread ANTES de habilitar o botão; e
        # este método a deixa limpa NO FIM (para o próximo tuning).
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=random_state))
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
            self.tuning_ = {"algorithm": algorithm, "metric": "auc" if is_clf else "r2",
                            "n_trials": n_ok,
                            "best_value": (round(float(study.best_value), 6)
                                           if n_ok else float("nan")),
                            "best_params": (dict(study.best_params) if n_ok else {}),
                            "cancelled": cancelled}
            if fit_best and n_ok and not cancelled:
                self.fit(algorithm=algorithm, hyperparams=study.best_params,
                         features=feats, transform=transform)

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
                               callbacks=callbacks + [_mlflow_cb])
                if _n_complete():                # cancelamento cedo pode não ter trial concluído
                    self._log_optuna_parent(mlflow, study, algorithm, transform, is_clf, feats)
                # re-treina o melhor modelo DENTRO do run-pai e o registra junto
                # dos trials (evita colisão de nome no Model Registry).
                _finish_best()
                if register_model and fit_best and not self._tuning_cancel.is_set() and _n_complete():
                    self._log_fitted_model(mlflow, registered_model_name=mlflow_model_name,
                                           verbose=verbose)
        else:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=callbacks)
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
                for grupo in ("modelagem", "monitoramento"):
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
            mlflow.log_params({f"melhor_hp/{k}": v for k, v in study.best_params.items()})
            if study.best_value is not None and np.isfinite(study.best_value):
                mlflow.log_metric("melhor/objetivo", float(study.best_value))
            best = study.best_trial
            for grupo in ("modelagem", "monitoramento"):
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
        for p in ("num__", "cat__"):
            if nm.startswith(p):
                nm = nm[len(p):]
                break
        # termos transformados vêm como 'WoE(feat)'/'bin(feat)': rotula o miolo
        wrap = None
        for w in ("WoE", "bin"):
            if nm.startswith(f"{w}(") and nm.endswith(")"):
                wrap, nm = w, nm[len(w) + 1:-1]
                break
        if use_labels and nm in self.feature_labels:
            nm = self.feature_labels[nm]
        return f"{wrap}({nm})" if wrap else nm

    def _logit_wald_pvalues(self, est, pre, names) -> dict:
        """p-valores de Wald (aprox.) por termo da **logística**: z = coef/EP,
        EP da diagonal de inv(Xᵀ W X), W = p(1−p). Aproximação (a logística do
        sklearn é regularizada); serve como indicação de significância."""
        from scipy.stats import norm
        fit_df = self._frame(self.ref_sample)
        fit_df = fit_df[fit_df[self.target].notna()]
        Xd = pre.transform(fit_df[self.model_features]) if pre is not None \
            else fit_df[self.model_features].to_numpy(dtype="float64")
        Xd = Xd.toarray() if hasattr(Xd, "toarray") else np.asarray(Xd, dtype="float64")
        p = np.clip(est.predict_proba(Xd)[:, 1], 1e-6, 1 - 1e-6)
        w = p * (1 - p)
        Xf = np.column_stack([np.ones(len(Xd)), Xd])          # intercepto + design
        H = Xf.T @ (Xf * w[:, None])
        cov = np.linalg.pinv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        beta = np.concatenate([np.ravel(est.intercept_), np.ravel(est.coef_)])
        z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
        pv = 2.0 * (1.0 - norm.cdf(np.abs(z)))
        return {nm: float(pv[i + 1]) for i, nm in enumerate(names)}

    @staticmethod
    def _signif_stars(p) -> str:
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return ""
        return ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05
                else "." if p < 0.10 else "n.s.")

    def model_coefficients(self, use_labels=True) -> pd.DataFrame:
        """Coeficientes do modelo **linear/logístico** ajustado: ``termo``, ``coef``
        e — na classificação — ``odds_ratio`` (``exp(coef)``). Na **logística**
        inclui também ``p_valor`` (Wald aprox.) e ``signif`` (estrelas). O intercepto
        fica em ``.attrs['intercept']``. Erro para modelos não-lineares (use SHAP)."""
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
        if len(names) != len(coef):                       # robustez a divergências
            names = [f"x{i}" for i in range(len(coef))]
        rows = [{"termo": nm, "coef": round(float(c), 6)} for nm, c in zip(names, coef)]
        out = pd.DataFrame(rows, columns=["termo", "coef"])
        if self.task_type == "classification" and not out.empty:
            out["odds_ratio"] = np.exp(out["coef"]).round(4)
        if self.algorithm == "logistica" and not out.empty:
            try:                                   # p-valor de Wald (aprox.) por termo
                pvals = self._logit_wald_pvalues(est, pre, names)
                out["p_valor"] = out["termo"].map(lambda t: round(pvals.get(t, np.nan), 4))
                out["signif"] = out["p_valor"].map(self._signif_stars)
            except Exception:
                pass
        out = out.reindex(out["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        out.attrs["intercept"] = round(intercept, 6)
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
                return p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else np.ravel(p)
            if hasattr(model, "decision_function"):
                return np.ravel(model.decision_function(X))
        return np.ravel(model.predict(X))

    def _compute_score(self, df) -> pd.Series:
        faltando = [f for f in self.model_features if f not in df.columns]
        if faltando:
            raise KeyError(
                f"Colunas ausentes para escorar: {faltando}. O modelo espera "
                f"{list(self.model_features)}.")
        X = df[self.model_features]
        return pd.Series(self._predict_score_array(self.model, X), index=df.index,
                         name="score", dtype="float64")

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

    def metric_shifts(self) -> dict:
        """Variação de cada métrica DES→OOT (oot − des)."""
        m = self.metrics().set_index("amostra")
        oot = self._oot_sample()
        if self.ref_sample not in m.index or oot not in m.index or oot == self.ref_sample:
            return {}
        # 'ks_cutoff' é o limiar de score onde o KS é máximo (escala do score),
        # não uma métrica de desempenho — seu "shift" não é comparável aos demais.
        cols = [c for c in m.columns if c not in ("n", "ks_cutoff")]

        def _num(x):     # evita np.isfinite(None)/str → TypeError (não está em try)
            return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x)

        return {c: round(float(m.loc[oot, c] - m.loc[self.ref_sample, c]), 6)
                for c in cols if _num(m.loc[oot, c]) and _num(m.loc[self.ref_sample, c])}

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
        tr = self._frame(self.ref_sample); tr = tr[tr[self.target].notna()]
        ev = self._frame(eval_sample); ev = ev[ev[self.target].notna()]
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
        # reconstrói o subconjunto do passo target_n: as features iniciais menos
        # todas as 'removida' dos passos com n_variaveis > target_n (mesma ordem do
        # backward). A coluna 'removida' é a variável retirada NO passo seguinte.
        # reconstrói a partir do conjunto inicial ANCORADO no resultado (identidade),
        # não do estado atual — evita aplicar um subconjunto de um backward de OUTRA
        # seleção. Fallback ao estado atual só p/ resultados legados sem 'feats0'.
        feats0 = list(result.attrs.get("feats0")
                      or (self.model_features or self.selected_features() or self.candidates))
        n_full = int(ns.max())
        if len(feats0) != n_full:
            raise RuntimeError(
                f"A seleção/modelo vigente ({len(feats0)} variáveis) diverge do topo "
                f"do backward ({n_full}). Rode o backward elimination de novo antes "
                f"de aplicar (a seleção mudou desde então).")
        removed = set(result.loc[ns > target_n, "removida"]) - {"—"}
        subset = [f for f in feats0 if f not in removed]
        if len(subset) != target_n:
            raise RuntimeError(
                f"Reconstrução inconsistente do subconjunto: esperado {target_n} "
                f"variáveis, obtido {len(subset)}.")
        return {"metric": metric, "criterion": criterion, "target_n": target_n,
                "best": best, "features": subset, "removed": sorted(removed)}

    def apply_backward_selection(self, result, criterion="parsimony", tol=0.01,
                                 metric=None, refit=True, rebuild_ratings=False):
        """Aplica à seleção vigente (``included``) o subconjunto de variáveis do
        passo **ótimo** de uma :meth:`backward_elimination` já executada.

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
        pick = self.backward_optimal_step(result, criterion=criterion, tol=tol, metric=metric)
        target_n = pick["target_n"]
        subset = pick["features"]
        removed = set(pick["removed"])
        metric = pick["metric"]
        best = pick["best"]
        self.clear_features()
        for f in subset:
            if f in self.candidates:
                self.included.add(f)
        rebuilt = reprojected = False
        if refit:
            self.fit(algorithm=self.algorithm, hyperparams=self.hyperparams,
                     features=subset, transform=self.feature_transform or "raw")
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
                                  dpi=150, save_path=None, ax=None):
        """Curva das **métricas vs nº de variáveis** da :meth:`backward_elimination`
        (o modelo encolhendo à medida que a variável menos importante sai). ``result``
        é o DataFrame devolvido por ela; ``metrics`` escolhe quais métricas plotar
        (default: KS+AUC na classificação, RMSE+MAE na regressão). A 1ª métrica vai
        no eixo esquerdo e as demais no direito (escalas diferentes)."""
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
        ln = ax.plot(x, result[metrics[0]].to_numpy(dtype="float64"), color=palette[0],
                     lw=2.0, marker="o", ms=4, label=metrics[0].upper())
        ax.set_ylabel(metrics[0].upper(), color=palette[0])
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
        mask = (pd.Series(True, index=self.df.index) if self.sample_col is None
                else self.df[self.sample_col] == sample)
        y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
        sc = self.score_[mask].to_numpy(dtype="float64")
        ok = ~np.isnan(y) & ~np.isnan(sc)
        return y[ok], sc[ok]

    def plot_roc(self, sample=None, figsize=(5.4, 5.0), dpi=150, save_path=None, ax=None):
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

        Eixos na **unidade do alvo** (LGD/PD previsto vs. observado), NÃO na escala
        de score 0–1000: a calibração é sobre o risco previsto casar com o realizado
        (mesma família de ``valor_previsto``/``backtest``). Distribuição e KS é que
        usam a escala de negócio (ranking)."""
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
            ax.scatter(sc, y, s=10, alpha=0.16, color="#9db8d2", edgecolors="none",
                       zorder=1)
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
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, color="#bbb", ls="--", lw=1)
        ax.set_xlabel("previsto"); ax.set_ylabel("observado")
        _pct_axis(ax, "both")                               # unidade do alvo (LGD/PD), em %
        ax.set_title(f"Calibração · {sample or self.ref_sample}", fontsize=11,
                     fontweight="bold", color="#15324a")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_residuals(self, sample=None, figsize=(6.6, 4.0), dpi=150, save_path=None, ax=None):
        """Regressão: resíduo (observado − previsto) vs. previsto, na **unidade do
        alvo** (LGD previsto), não na escala de score 0–1000."""
        y, sc = self._sample_scores(sample)                 # previsto/observado CRUS (alvo)
        fig, ax = _new_ax(figsize, dpi, ax)
        res = y - sc
        ax.scatter(sc, res, s=10, alpha=0.35, color="#3b6ea5", edgecolors="none")
        ax.axhline(0, color="#d6453e", lw=1)
        ax.set_xlabel("previsto"); ax.set_ylabel("resíduo (obs − prev)")
        _pct_axis(ax, "both")                               # unidade do alvo (LGD/PD), em %
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
        base = self._frame(sample) if sample else self.df
        if time_col not in base.columns:
            raise ValueError(f"Coluna de tempo '{time_col}' não existe no DataFrame.")
        is_clf = self.task_type == "classification"
        sc_full = self.score_.reindex(base.index)
        safra = pd.to_datetime(base[time_col], errors="coerce").dt.to_period("M")
        met_cols = (["taxa_evento", "auc", "ks", "gini"] if is_clf
                    else ["previsto_medio", "realizado_medio", "mae", "rmse", "r2"])
        rows = []
        for per, g in base.groupby(safra):            # groupby dropa safra NaT
            y = g[self.target].to_numpy(dtype="float64")
            sc = sc_full.reindex(g.index).to_numpy(dtype="float64")
            ok = ~np.isnan(y) & ~np.isnan(sc)
            y, sc = y[ok], sc[ok]
            row = {"safra": str(per), "n": int(y.size)}
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
                              save_path=None, ax=None):
        """Evolução das métricas do modelo por safra (linhas), a partir de
        :meth:`metrics_by_safra`. ``metrics`` que não existirem para o
        ``task_type`` são ignoradas (default de regressão: ``mae``/``rmse``)."""
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

    # ---- SHAP ----
    def _shap_inputs(self, sample=None, sample_size=2000):
        """(estimador, X_transformado_df, nomes) — para SHAP. Em pipelines,
        transforma com o pré-processador e usa o estimador final."""
        sub = self._frame(sample)
        X = sub[self.model_features]
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
            names = list(self.model_features)
        if sample_size and len(Xt) > sample_size:
            Xt = Xt.sample(sample_size, random_state=self.random_state)
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
        for p in ("num__", "cat__"):
            if raw.startswith(p):
                raw = raw[len(p):]
                break
        for w in ("WoE", "bin"):
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

    def plot_shap_beeswarm(self, sample=None, sample_size=2000, max_display=15):
        """Beeswarm SHAP do modelo (usa pyplot; devolve a figura). Usa o alias das
        variáveis (``feature_labels``) e corta nomes longos no eixo Y."""
        import matplotlib.pyplot as plt
        import shap
        sv, Xs = self.shap_values(sample, sample_size)
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
        return fig

    def plot_shap_bar(self, sample=None, sample_size=2000, max_display=15):
        """Importância global SHAP (barras). Usa o alias das variáveis
        (``feature_labels``) e corta nomes longos no eixo Y."""
        import matplotlib.pyplot as plt
        import shap
        sv, Xs = self.shap_values(sample, sample_size)
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
        return fig

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
                evals[n] = (mono_ok, vol_ok, eff, inv["safra_rate"], gini)
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
            # nenhuma zerou inversões: prioriza menos inversão entre amostras, depois
            # volume adequado, menos inversão de safra e maior granularidade
            best = min(evals, key=lambda n: (0 if evals[n][0] else 1, not evals[n][1],
                                             evals[n][3], -evals[n][2]))
            reason = (f"{best} ratings — nenhuma régua zerou as inversões; escolhida a de "
                      f"menor inversão, volume adequado e maior granularidade.")
        return {"best": int(best), "table": table, "reason": reason,
                "n_recomendado": int(best)}

    def _rating_series(self) -> pd.Series:
        if self.rating_ is None:
            raise RuntimeError("Gere os ratings antes (build_ratings).")
        return self.rating_

    def rating_table(self) -> pd.DataFrame:
        """Por rating (na ordem dos rótulos): n, repr_% (na DES) e o risco
        (event_rate/alvo médio) em **cada amostra** — leitura de monotonicidade e
        estabilidade da régua entre amostras."""
        rating = self._rating_series()
        labels = self.rating_labels_
        prefix = "event_rate" if self.task_type == "classification" else "alvo"
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
                rows.append({"rating": lab, "n": n,
                             "repr_%": round(100 * n / n_ref_tot, 1),
                             f"{prefix}_{self.ref_sample}":
                                 round(float(v), 4) if pd.notna(v) else np.nan})
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
            rows.append(row)
        return pd.DataFrame(rows)

    def rating_inversion(self, time_col=None, sample=None, min_n=20) -> dict:
        """Inversão da ordem de risco ENTRE ratings (entre amostras e safras) —
        o estudo de folhas-irmãs do PD/LGD, aplicado às faixas de rating."""
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
            base = self._frame(sample) if sample else self.df
            r2 = rating.reindex(base.index)
            safra = pd.to_datetime(base[tcol], errors="coerce").dt.to_period("M")
            for per, g in base.groupby(safra):
                if len(g) < min_n:
                    continue
                rr = r2.reindex(g.index)
                vals = {lab: self._risco(g.loc[rr == lab, self.target]) for lab in labels}
                safra_series[str(per)] = vals
                n_inv, npp = _count_inversions(ordered, vals)
                if npp == 0:
                    continue
                safra_rows.append({"safra": str(per), "n_inv": n_inv, "n_pares": npp})

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
        stab_color = "#7e57c2"                    # 3ª cor (roxo) p/ a safra de estabilidade
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

    def plot_rating_inversion_by_sample(self, figsize=(7.6, 4.0), dpi=150,
                                        save_path=None, ax=None):
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
        ax.set_title("Risco dos ratings por amostra (cruzamento = inversão)",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        ax.legend(fontsize=7.5, ncol=max(1, min(k, 4)), loc="best", framealpha=0.85)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_rating_inversion_by_safra(self, time_col=None, sample=None, min_n=20,
                                       figsize=(9.6, 4.0), dpi=150, save_path=None, ax=None):
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
        rating = self._rating_series()
        labels = self.rating_labels_
        dist = {}
        for a in self.df[self.sample_col].dropna().unique():
            am = self.df[self.sample_col] == a
            n_a = max(int(am.sum()), 1)
            dist[a] = {l: int(((rating == l) & am).sum()) / n_a for l in labels}
        ref = dist[self.ref_sample]
        rows = []
        for a, pct in dist.items():
            if a == self.ref_sample:
                continue
            psi = sum((max(pct[l], eps) - max(ref[l], eps)) *
                      np.log(max(pct[l], eps) / max(ref[l], eps)) for l in labels)
            rows.append({"amostra": a, "psi": round(float(psi), 4),
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

        def _dist(sample):
            am = self.df[self.sample_col] == sample
            n = max(int(am.sum()), 1)
            return {l: int(((rating == l) & am).sum()) / n for l in labels}

        ref_dist = _dist(ref)
        comp_dists = {a: _dist(a) for a in comparison_samples}
        rows, totais = [], {a: 0.0 for a in comparison_samples}
        for l in labels:
            row = {"rating": l, f"%{ref}": round(100 * ref_dist[l], 2)}
            for a in comparison_samples:
                pct_a = comp_dists[a][l]
                contrib = ((max(pct_a, eps) - max(ref_dist[l], eps)) *
                           np.log(max(pct_a, eps) / max(ref_dist[l], eps)))
                totais[a] += contrib
                row[f"%{a}"] = round(100 * pct_a, 2)
                row[f"PSI {a}"] = round(float(contrib), 4)
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
        rating = self._rating_series()
        labels = self.rating_labels_
        # distribuição de referência: ratings na DES (ou toda a base, sem sample_col).
        # Denominador = ratings NÃO-NaN (score inválido ⇒ rating NaN não entra na
        # distribuição); assim as proporções somam 1, como o value_counts por safra.
        ref_mask = (self._frame_mask(self.ref_sample) if self.sample_col is not None
                    else pd.Series(True, index=self.df.index))
        valid = rating.notna()
        n_ref = max(int((valid & ref_mask).sum()), 1)
        ref_pct = {l: max(int(((rating == l) & ref_mask).sum()) / n_ref, eps) for l in labels}
        safra = pd.to_datetime(self.df[time_col], errors="coerce").dt.to_period("M")
        rows = []
        for per, r_g in rating.groupby(safra):        # groupby dropa safra NaT
            n_g = int(r_g.notna().sum())              # ignora ratings NaN no denominador
            if n_g == 0:
                continue
            vc = r_g.value_counts()
            psi = 0.0
            for l in labels:
                p_cur = max(int(vc.get(l, 0)) / n_g, eps)
                p_ref = ref_pct[l]
                psi += (p_cur - p_ref) * np.log(p_cur / p_ref)
            rows.append({"safra": str(per), "n": int(n_g), "psi": round(float(psi), 4),
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
            risco = [vals[l] for l in labels]
            trend, n_inv = _trend(risco)
            if self.sample_col is not None and a == self.ref_sample:
                inv_txt = "(referência)"
            else:
                pares = _inverted_pairs(ordered, vals)
                inv_txt = ", ".join(f"{x}↔{y}" for x, y in pares) if pares else "—"
            rows.append({"amostra": a, "monotonico": n_inv == 0,
                         "tendencia": trend, "n_inversoes": n_inv,
                         "inverte_vs_DES": inv_txt})
        return pd.DataFrame(rows)

    def backtest(self, time_col=None, sample=None, tol=0.10) -> pd.DataFrame:
        """Risco previsto (score médio) vs realizado (alvo médio) por safra."""
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
            prev = float(sc.reindex(g.index).mean(skipna=True))
            real = self._risco(g[self.target])
            gap = prev - real if (np.isfinite(prev) and np.isfinite(real)) else np.nan
            rows.append({"safra": str(per), "n": len(g),
                         "previsto_medio": round(prev, 4) if np.isfinite(prev) else np.nan,
                         "realizado_medio": round(real, 4) if np.isfinite(real) else np.nan,
                         "gap": round(gap, 4) if np.isfinite(gap) else np.nan,
                         "status": ("ok" if (np.isfinite(gap) and abs(gap) <= tol)
                                    else "alerta")})
        return pd.DataFrame(rows).sort_values("safra").reset_index(drop=True)

    def plot_backtest(self, sample=None, tolerancia=0.2, time_col=None,
                      figsize=(9.6, 4.4), dpi=150, save_path=None, ax=None):
        """Backtest gráfico: previsto × realizado por safra (duas linhas) com
        **banda de tolerância** sombreada em torno do previsto
        (``previsto ± tolerancia·previsto``; ``tolerancia=0.2`` = ±20%) e
        marcadores destacados nos meses em que o realizado sai da banda.
        Funciona nos dois ``task_type`` (unidade do alvo: PD/LGD). Usa a mesma
        agregação por safra de :meth:`backtest`."""
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
                        label=f"tolerância ±{100 * tolerancia:.0f}%")
        ax.plot(x, prev, color="#15324a", lw=2.2, marker="o", ms=4.5,
                label="previsto (médio)")
        ax.plot(x, real, color="#1aa64b", lw=2.0, marker="o", ms=4.5,
                label="realizado (médio)")
        # meses FORA da banda: marcador destacado em vermelho sobre o realizado
        fora = np.isfinite(real) & np.isfinite(prev) & ((real < lo) | (real > hi))
        if fora.any():
            xs = [x0 for x0, f in zip(x, fora) if f]
            ax.plot(xs, real[fora], "o", ms=9, mfc="none", mec="#d6453e", mew=2.0,
                    label="fora da banda")
        labels = _fmt_safras(bt["safra"])
        step = max(1, len(bt) // 18)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("safra")
        ax.set_ylabel("PD" if self.task_type == "classification" else "alvo médio")
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
        (``ref_sample``/DES por padrão). É o "LGD/PD previsto daquele rating", usado
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

    def predict(self, X: pd.DataFrame, col_score="score", col_rating="rating",
                col_value=None, ruler_sample=None) -> pd.DataFrame:
        """Aplica modelo (+rating) a novos dados. Devolve score e rating por linha.

        Se ``col_value`` for informado (ex.: ``"valor_previsto"``), anexa também o
        **valor previsto do alvo daquele rating** — a régua de :meth:`rating_ruler`
        calibrada na amostra ``ruler_sample`` (DES por padrão), i.e. o LGD/PD
        previsto por rating."""
        if self.model is None:
            raise RuntimeError("Ajuste/defina o modelo antes (fit / set_model).")
        X = self._apply_derived(X)        # recria variáveis derivadas a partir da origem
        sc = pd.Series(self._predict_score_array(self.model, X[self.model_features]),
                       index=X.index, dtype="float64")           # predição CRUA
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
        return out

    def _labels_from_bins(self, col, bins, feature, others="(outros)") -> pd.Series:
        """Rótulo do bin (faixa/grupo) de cada valor de ``col`` segundo ``bins``.
        Valores fora de todos os bins (categoria nova) e não-nulos viram ``others``."""
        col = pd.Series(col)
        labels = pd.Series([pd.NA] * len(col), index=col.index, dtype="object")
        assigned = np.zeros(len(col), dtype=bool)
        for b in bins:
            m = _bin_mask_series(col, b).to_numpy() & ~assigned
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

    def create_categorical(self, feature, new_name=None) -> str:
        """Materializa a binagem atual de ``feature`` (faixas numéricas ou grupos
        categóricos — **manuais** quando definidos, senão o **ótimo**) como uma NOVA
        variável categórica no DataFrame, candidata ao modelo. É o equivalente a
        "agrupar bins" da árvore de decisão, persistido numa coluna: junte categorias
        na mão (ex.: ``A,B; C,D``) e gere a variável agrupada.

        A derivação fica registrada (origem + bins), então a variável é **recriada
        automaticamente** ao escorar uma base que tenha só as variáveis originais.
        Devolve o nome da nova variável."""
        if feature not in self.df.columns:
            raise ValueError(f"'{feature}' não está no DataFrame.")
        bins, kind = self._resolve_bins(feature, sample=self.ref_sample)
        if not bins:
            raise ValueError(f"Sem bins para '{feature}'. Defina cortes/grupos (ou rode o "
                             "binning ótimo) antes de criar a variável.")
        name = new_name or f"{feature}_cat"
        base, k = name, 2
        while name in self.df.columns:
            name = f"{base}_{k}"; k += 1
        self.df[name] = self._labels_from_bins(self.df[feature], bins, feature).to_numpy()
        if name not in self.candidates:
            self.candidates.append(name)
        self.var_meta[name] = {"categoria": None, "derived_from": feature,
                               "derived_kind": kind, "derived_bins": bins}
        self.feature_labels.setdefault(name, f"{self.label(feature)} (cat.)")
        self._rank_version += 1   # nova candidata → o ranking de IV precisa recalcular
        return name

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
            X[n] = self._labels_from_bins(X[src], meta.get("derived_bins") or [],
                                          src).to_numpy()
        return X

    def _rebuild_derived(self):
        """Recria no ``self.df`` as variáveis derivadas registradas em ``var_meta``
        (usado após load, quando o df vem só com as colunas originais)."""
        for name, meta in self.var_meta.items():
            src = meta.get("derived_from")
            if not src or name in self.df.columns or src not in self.df.columns:
                continue
            self.df[name] = self._labels_from_bins(
                self.df[src], meta.get("derived_bins") or [], src).to_numpy()

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
        s.df = self.df.iloc[0:0].copy()           # só o schema (sem as linhas de treino)
        s._mask_cache = {}
        s._samples_cache = None
        s._rank_cache = {}
        s._metrics_cache = None
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
        ``probe=True``) a escoragem cai no legado automaticamente."""
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

    def to_dict(self) -> dict:
        """Configuração serializável (sem o modelo binário — ver :meth:`save`)."""
        return {
            "schema": SCHEMA,
            "meta": {"target": self.target, "task_type": self.task_type,
                     "sample_col": self.sample_col, "ref_sample": self.ref_sample,
                     "date_col": self.date_col, "feature_labels": self.feature_labels,
                     "score_scale": self.score_scale, "random_state": self.random_state},
            "candidates": list(self.candidates),
            "included": sorted(self.included),
            "var_meta": self.var_meta,
            "algorithm": self.algorithm,
            "hyperparams": self.hyperparams,
            "model_features": list(self.model_features),
            "rating_config": self.rating_config,
            "two_stage": self.two_stage,
            "two_stage_threshold": self.two_stage_threshold,
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
                  features=data.get("candidates"), date_col=meta.get("date_col"),
                  verbose=verbose, score_scale=meta.get("score_scale", 1000.0),
                  random_state=meta.get("random_state", 42))
        seg.included = set(data.get("included", seg.candidates))
        seg.var_meta = data.get("var_meta", seg.var_meta)
        seg.algorithm = data.get("algorithm")
        seg.hyperparams = data.get("hyperparams", {})
        seg.model_features = data.get("model_features", [])
        seg.rating_config = data.get("rating_config", {})
        seg.two_stage = bool(data.get("two_stage", False))
        seg.two_stage_threshold = data.get("two_stage_threshold")
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
                      artifact_path="modelo", registry_uri=None, verbose=True):
        """Registra o modelo (mlflow.sklearn), métricas por amostra, a régua de
        ratings e os gráficos SHAP como artefatos. Best-effort."""
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
