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

_EPS = 1e-6


def _optuna_space(trial, algorithm: str) -> dict:
    """Espaço de busca de hiperparâmetros por algoritmo para o Optuna."""
    if algorithm == "logistica":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if algorithm in ("random_forest", "extra_trees"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 80),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    if algorithm == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    if algorithm == "hist_gradient_boosting":
        return {
            "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        }
    if algorithm == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if algorithm == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if algorithm == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 800, step=50),
            "depth": trial.suggest_int("depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        }
    raise ValueError(
        f"O algoritmo {algorithm!r} não tem espaço de tuning. "
        f"Tunáveis: {TUNABLE_ALGORITHMS}.")


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
    """Roda ``b.fit(x, y)`` e devolve ``list(b.splits)``; silencia warnings
    benignos e devolve ``[]`` se o ajuste falhar."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(divide="ignore", invalid="ignore"):
                b.fit(x, y)
        return list(b.splits)
    except Exception:
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


def _build_estimator(algorithm: str, task_type: str, hyperparams: dict | None):
    """Instancia o estimador do algoritmo escolhido (registry extensível).

    sklearn é sempre disponível; LightGBM/XGBoost/CatBoost são pacotes opcionais
    importados sob demanda (ver :func:`_require`)."""
    hp = dict(hyperparams or {})
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
        hp.setdefault("random_state", 42)
        return RF(**hp)
    if algorithm == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        ET = ExtraTreesClassifier if is_clf else ExtraTreesRegressor
        hp.setdefault("n_estimators", 200)
        hp.setdefault("random_state", 42)
        return ET(**hp)
    if algorithm == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        GB = GradientBoostingClassifier if is_clf else GradientBoostingRegressor
        hp.setdefault("random_state", 42)
        return GB(**hp)
    if algorithm == "hist_gradient_boosting":
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      HistGradientBoostingRegressor)
        HGB = HistGradientBoostingClassifier if is_clf else HistGradientBoostingRegressor
        if "n_estimators" in hp:                       # nome unificado na UI → max_iter
            hp["max_iter"] = hp.pop("n_estimators")
        hp.setdefault("random_state", 42)
        return HGB(**hp)
    if algorithm == "lightgbm":
        lgb = _require("lightgbm", algorithm)
        Est = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor
        hp.setdefault("n_estimators", 300)
        hp.setdefault("learning_rate", 0.05)
        hp.setdefault("random_state", 42)
        hp.setdefault("verbose", -1)
        return Est(**hp)
    if algorithm == "xgboost":
        xgb = _require("xgboost", algorithm)
        Est = xgb.XGBClassifier if is_clf else xgb.XGBRegressor
        hp.setdefault("n_estimators", 300)
        hp.setdefault("learning_rate", 0.05)
        hp.setdefault("random_state", 42)
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
        hp.setdefault("random_seed", 42)
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


# ======================================================================
# Classe principal
# ======================================================================
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
        self.score_: pd.Series | None = None
        self.rating_strategy = None
        self.rating_col_: str | None = None
        self.rating_: pd.Series | None = None
        self.rating_labels_: list = []
        self.rating_config: dict = {}
        self._shap_cache: dict = {}

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
            ivt = self.variable_iv(features=[feature], sample=sample)
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

    def variable_by_safra(self, feature, time_col=None, sample=None) -> pd.DataFrame:
        """Percentis (min, p5, média, p95, max) e %missing de variável NUMÉRICA por safra."""
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col.")
        sub = self._frame(sample)
        safra = pd.to_datetime(sub[time_col], errors="coerce").dt.to_period("M")
        rows = []
        for per, g in sub.groupby(safra):
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

    def variable_share_by_safra(self, feature, time_col=None, sample=None, top=8) -> pd.DataFrame:
        """Representatividade (%) de cada categoria por safra (variável CATEGÓRICA)."""
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col.")
        sub = self._frame(sample)
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
                                 figsize=(8.6, 3.4), dpi=150, save_path=None, ax=None):
        """Numérica: percentis por safra. Categórica: área empilhada de share."""
        if self._detect_kind(feature, self._frame(sample)) == "cat":
            return self._plot_share_timeseries(feature, time_col, sample, figsize,
                                               dpi, save_path, ax)
        bs = self.variable_by_safra(feature, time_col, sample)
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

    def _plot_share_timeseries(self, feature, time_col, sample, figsize, dpi, save_path, ax):
        import matplotlib.colors as mcolors
        sh = self.variable_share_by_safra(feature, time_col, sample)
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
                                      max_n_bins=6, min_bin_size=0.05, bins=None) -> pd.DataFrame:
        """% de cada **faixa/categoria** da variável por safra (mês).

        Usa as MESMAS faixas da análise (:meth:`_resolve_bins`) — vale para
        variáveis numéricas (faixas) e categóricas (grupos). Colunas: ``safra`` +
        uma por faixa (% da safra). Linhas sem faixa (ex.: faltantes fora dos
        bins) entram em ``(faltante)``.

        ``bins`` (opcional): lista de bins já resolvida a usar no lugar de
        :meth:`_resolve_bins` — útil para forçar as faixas do optimal binning
        (ver :meth:`plot_variable_optbin_share_timeseries`)."""
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col.")
        sub = self._frame(sample)
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

    def _optbin_numeric_bins(self, feature, sample=None, max_n_bins=5, min_bin_size=0.05):
        """Faixas do **OPTIMAL BINNING** de uma variável NUMÉRICA — sempre roda o
        optbinning na amostra de ajuste, IGNORANDO eventuais bins manuais. Retorna
        a lista de bins numéricos (+ ``na`` se houver faltantes) ou ``[]`` quando
        não dá para binar. (:meth:`_resolve_bins` respeita bins manuais; este não.)"""
        if OptimalBinning is None:
            raise ImportError("optbinning não instalado. Rode: pip install optbinning")
        fit = self._frame(sample)
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
                cuts.append(float(tok))
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

    def _build_pipeline(self, features, algorithm, hyperparams, transform="raw"):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        est = _build_estimator(algorithm, self.task_type, hyperparams)
        if transform == "woe":
            # variáveis transformadas no estilo scorecard (binagem + WoE/risco do bin)
            encodings = {f: self._bin_encoding(f) for f in features}
            prefix = "WoE" if self.task_type == "classification" else "bin"
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

    def tune_optuna(self, algorithm=None, n_trials=30, transform="raw", features=None,
                    timeout=None, random_state=42, fit_best=True, verbose=False,
                    progress_callback=None, log_mlflow=False, mlflow_experiment=None,
                    mlflow_run_name=None):
        """Otimização bayesiana de hiperparâmetros com **Optuna** (dependência
        core). Treina na referência (DES) e avalia no OOT (se houver alvo;
        senão, num split 75/25 do DES), maximizando **AUC** (classificação) ou
        **R²** (regressão). Guarda o resultado em ``self.tuning_`` (e o estudo em
        ``self.study_``); com ``fit_best=True`` reajusta o modelo com os melhores
        hiperparâmetros. Algoritmos tunáveis: :data:`TUNABLE_ALGORITHMS`.

        Cada trial guarda, em ``trial.user_attrs``, dois grupos de métricas —
        ``modelagem`` (AUC/KS/Gini ou RMSE/MAE/R² na validação) e ``monitoramento``
        (PSI do score DES→validação, volumetria). Com ``log_mlflow=True`` cada
        trial vira um **run aninhado** no MLflow (params + métricas agrupadas por
        ``modelagem/…`` e ``monitoramento/…``), sob um run-pai com o resumo do
        estudo (melhores hiperparâmetros e gráficos do Optuna). Ver também
        :meth:`log_optuna_to_mlflow` para logar um estudo já concluído.

        ``progress_callback`` (opcional): chamado após CADA trial com
        ``(n_concluidos, n_total, melhor_valor)`` — útil p/ barra de progresso na
        UI. Exceções no callback são ignoradas (não derrubam o tuning)."""
        optuna = _require("optuna", "optuna")
        from sklearn.model_selection import train_test_split

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
            hp = _optuna_space(trial, algorithm)
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
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=random_state))
        callbacks = []
        if progress_callback is not None:
            def _progress_cb(study, trial):     # chamado após cada trial
                try:
                    progress_callback(len(study.trials), n_trials, float(study.best_value))
                except Exception:
                    pass                          # progresso é cosmético; nunca derruba o tuning
            callbacks.append(_progress_cb)

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
                self._log_optuna_parent(mlflow, study, algorithm, transform, is_clf, feats)
        else:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=callbacks)

        self.study_ = study
        self.tuning_ = {"algorithm": algorithm, "metric": "auc" if is_clf else "r2",
                        "n_trials": len(study.trials), "best_value": round(float(study.best_value), 6),
                        "best_params": dict(study.best_params)}
        if fit_best:
            self.fit(algorithm=algorithm, hyperparams=study.best_params,
                     features=feats, transform=transform)
        return self.tuning_

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
        self.model_features = (list(features) if features is not None
                               else (self.model_features or self.selected_features()
                                     or list(self.candidates)))
        self.score_ = self._compute_score(self.df)
        self._shap_cache = {}
        return self

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
        out = []
        for nm in raw:
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
            out.append(f"{wrap}({nm})" if wrap else nm)
        return out

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
        X = df[self.model_features]
        return pd.Series(self._predict_score_array(self.model, X), index=df.index,
                         name="score", dtype="float64")

    def metrics(self) -> pd.DataFrame:
        """Métricas do modelo por amostra: classificação (auc, gini, ks, ks_cutoff,
        accuracy, f1, precision, recall, brier, logloss) ou regressão (rmse, mae,
        mape, smape, medae, r2, mean_bias)."""
        if self.score_ is None:
            raise RuntimeError("Ajuste o modelo antes (fit / set_model).")
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
        return {c: round(float(m.loc[oot, c] - m.loc[self.ref_sample, c]), 6)
                for c in cols if np.isfinite(m.loc[oot, c]) and np.isfinite(m.loc[self.ref_sample, c])}

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
        ax.set_xlabel("score"); ax.set_ylabel("CDF acumulada")
        _pct_axis(ax, "x")
        ax.set_title(f"Curva KS · {sample or self.ref_sample}", fontsize=11,
                     fontweight="bold", color="#15324a")
        ax.legend(fontsize=9, loc="center right"); ax.grid(alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_score_distribution(self, sample=None, bins=30, figsize=(6.6, 3.8),
                                dpi=150, save_path=None, ax=None):
        y, sc = self._sample_scores(sample)
        fig, ax = _new_ax(figsize, dpi, ax)
        if self.task_type == "classification" and len(np.unique(y)) == 2:
            ax.hist(sc[y == 0], bins=bins, color="#1aa64b", alpha=0.55, label="não-evento (0)",
                    density=True, edgecolor="white", linewidth=0.3)
            ax.hist(sc[y == 1], bins=bins, color="#d6453e", alpha=0.55, label="evento (1)",
                    density=True, edgecolor="white", linewidth=0.3)
            ax.legend(fontsize=9)
        else:
            ax.hist(sc, bins=bins, color="steelblue", alpha=0.85, edgecolor="#2f5d82")
        ax.set_xlabel("score"); ax.set_ylabel("densidade")
        _pct_axis(ax, "x")
        ax.set_title(f"Distribuição do score · {sample or self.ref_sample}",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_calibration(self, sample=None, n_bins=10, figsize=(5.6, 5.2), dpi=150,
                         save_path=None, ax=None):
        """Classificação: previsto×observado por decil de score. Regressão: previsto×observado."""
        y, sc = self._sample_scores(sample)
        fig, ax = _new_ax(figsize, dpi, ax)
        if y.size == 0:
            ax.axis("off"); fig.tight_layout(); return fig
        if self.task_type == "classification":
            q = np.quantile(sc, np.linspace(0, 1, n_bins + 1))
            q = np.unique(q)
            idx = np.clip(np.searchsorted(q, sc, side="right") - 1, 0, len(q) - 2)
            pred, obs = [], []
            for g in range(len(q) - 1):
                m = idx == g
                if m.sum():
                    pred.append(sc[m].mean()); obs.append(y[m].mean())
            ax.plot(pred, obs, marker="o", color="#15324a", lw=2, ms=6)
        else:
            ax.scatter(sc, y, s=10, alpha=0.35, color="#3b6ea5", edgecolors="none")
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, color="#bbb", ls="--", lw=1)
        ax.set_xlabel("previsto"); ax.set_ylabel("observado")
        _pct_axis(ax, "both")
        ax.set_title(f"Calibração · {sample or self.ref_sample}", fontsize=11,
                     fontweight="bold", color="#15324a")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_residuals(self, sample=None, figsize=(6.6, 4.0), dpi=150, save_path=None, ax=None):
        """Regressão: resíduo (observado − previsto) vs. previsto."""
        y, sc = self._sample_scores(sample)
        fig, ax = _new_ax(figsize, dpi, ax)
        res = y - sc
        ax.scatter(sc, res, s=10, alpha=0.35, color="#3b6ea5", edgecolors="none")
        ax.axhline(0, color="#d6453e", lw=1)
        ax.set_xlabel("previsto"); ax.set_ylabel("resíduo (obs − prev)")
        _pct_axis(ax, "both")
        ax.set_title(f"Resíduos · {sample or self.ref_sample}", fontsize=11,
                     fontweight="bold", color="#15324a")
        ax.grid(alpha=0.15)
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
            Xt = Xt.sample(sample_size, random_state=42)
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

    def plot_shap_beeswarm(self, sample=None, sample_size=2000, max_display=15):
        """Beeswarm SHAP do modelo (usa pyplot; devolve a figura)."""
        import matplotlib.pyplot as plt
        import shap
        sv, Xs = self.shap_values(sample, sample_size)
        plt.figure()
        shap.summary_plot(sv, Xs, show=False, max_display=max_display)
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
        """Importância global SHAP (barras)."""
        import matplotlib.pyplot as plt
        import shap
        sv, Xs = self.shap_values(sample, sample_size)
        plt.figure()
        shap.summary_plot(sv, Xs, plot_type="bar", show=False, max_display=max_display)
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
            raise RuntimeError("Gere o score antes (fit / set_model).")
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
            strat = RATING_REGISTRY[method](max_leaf_nodes=n, alpha=alpha)
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
                ax.text(x0, r, f"{r*100:.1f}%", ha="center", va="bottom", fontsize=7.5,
                        color="#15324a")
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
        palette = ["steelblue", "crimson"]
        for k, a in enumerate(samples):
            am = (pd.Series(True, index=self.df.index) if self.sample_col is None
                  else self.df[self.sample_col] == a)
            n_a = max(int(am.sum()), 1)
            pct = [100 * int(((rating == l) & am).sum()) / n_a for l in labels]
            ax.bar(x + k * w, pct, width=w, label=a, alpha=0.9,
                   color=palette[k % len(palette)])
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

    # ------------------------------------------------------------------
    # Export / predict / persistência
    # ------------------------------------------------------------------
    def assign(self, col_score="score", col_rating="rating") -> pd.DataFrame:
        """Cópia do df com o score e o rating de cada linha."""
        out = self.df.copy()
        if self.score_ is not None:
            out[col_score] = self.score_
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
                       index=X.index, name=col_score, dtype="float64")
        out = pd.DataFrame({col_score: sc}, index=X.index)
        if self.rating_strategy is not None:
            wf = pd.DataFrame({"score": sc}, index=X.index)
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

    def apply_spark(self, sdf, col_score="score", col_rating="rating", col_value=None,
                    ruler_sample=None, recreate_categories=None, cat_suffix="_faixa",
                    progress_callback=None):
        """Escora um **Spark DataFrame** (ex.: tabela do Databricks/Unity Catalog) e
        devolve um Spark DataFrame com ``score`` + ``rating`` (+ valor previsto) e,
        quando o modelo usou variáveis categorizadas (WoE/bins), as faixas recriadas.
        A tabela só precisa ter as variáveis originais do modelo (``model_features``);
        a binagem/WoE é refeita internamente.

        Aplica o modelo sklearn coletando a tabela no driver (``toPandas``); para
        tabelas muito grandes, filtre/particione antes de escorar."""
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
                    progress_callback=None):
        """Escora uma tabela e devolve as **notas (score)** e os **ratings**.

        ``data`` pode ser o **nome** de uma tabela do Databricks (``catalog.schema.
        tabela``), um **Spark DataFrame** ou um **pandas DataFrame**. A tabela só
        precisa conter as variáveis originais do modelo; se uma variável foi
        categorizada (faixas/grupos via WoE), a categoria é recriada na saída.

        - nome de tabela ou Spark DataFrame → devolve um **Spark DataFrame** (e grava
          em ``output_table`` quando informado);
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
                                   cat_suffix=cat_suffix, progress_callback=progress_callback)
            if output_table:
                _emit_progress(progress_callback, "save", f"Salvar em '{output_table}'", "run")
                out.write.mode(mode).saveAsTable(output_table)
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
                     "date_col": self.date_col, "feature_labels": self.feature_labels},
            "candidates": list(self.candidates),
            "included": sorted(self.included),
            "var_meta": self.var_meta,
            "algorithm": self.algorithm,
            "hyperparams": self.hyperparams,
            "model_features": list(self.model_features),
            "rating_config": self.rating_config,
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
                  verbose=verbose)
        seg.included = set(data.get("included", seg.candidates))
        seg.var_meta = data.get("var_meta", seg.var_meta)
        seg.algorithm = data.get("algorithm")
        seg.hyperparams = data.get("hyperparams", {})
        seg.model_features = data.get("model_features", [])
        seg.rating_config = data.get("rating_config", {})
        seg._rebuild_derived()      # recria colunas categóricas derivadas no df
        return seg

    def load(self, path: str, df: pd.DataFrame = None):
        """Carrega configuração + modelo. Se ``df`` for dado, recalcula score e
        (se havia rating) reaproveita a estratégia salva para reaplicar os ratings."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        seg = ModelSegmenter.from_dict(data, self.df if df is None else df, verbose=False)
        try:
            import joblib
            blob = joblib.load(path + ".model.joblib")
            seg.model = blob.get("model")
            seg.rating_strategy = blob.get("rating_strategy")
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
        return seg

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
            raise RuntimeError("Ajuste o modelo antes (fit / set_model).")
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
                        if c in ("amostra",) or not np.isfinite(v):
                            continue
                        mlflow.log_metric(f"{r['amostra']}_{c}", float(v))
            except Exception:
                pass
            try:
                import mlflow.sklearn
                mlflow.sklearn.log_model(self.model, artifact_path,
                                         registered_model_name=registered_model_name)
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
