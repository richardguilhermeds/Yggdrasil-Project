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

_EPS = 1e-6


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
        """Amostras presentes, referência primeiro."""
        if self.sample_col is None:
            return [self.ref_sample]
        ams = list(self.df[self.sample_col].dropna().unique())
        return [self.ref_sample] + [a for a in ams if a != self.ref_sample]

    def _nonref_samples(self) -> list:
        return [a for a in self._samples() if a != self.ref_sample]

    def _oot_sample(self) -> str:
        nr = self._nonref_samples()
        return nr[0] if nr else self.ref_sample

    def _frame(self, sample=None) -> pd.DataFrame:
        """Recorte do df por amostra (default DES quando há sample_col)."""
        if self.sample_col is None:
            return self.df
        if sample is None:
            sample = self.ref_sample
        return self.df[self.df[self.sample_col] == sample]

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

    def _resolve_bins(self, feature, max_n_bins=5, min_bin_size=0.05, splits=None,
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
        na regressão. PSI calculado nos mesmos bins (DES × cada amostra)."""
        features = list(features) if features is not None else list(self.candidates)
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
            row["incluida"] = feat in self.included
            row["categoria"] = self.var_meta.get(feat, {}).get("categoria")
            row["bins_manuais"] = bool(self.var_meta.get(feat, {}).get("splits"))
            rows.append(row)
        return (pd.DataFrame(rows)
                .sort_values("iv", ascending=False, na_position="last")
                .reset_index(drop=True))

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
        ax.set_xticks(x); ax.set_xticklabels(bs["safra"], rotation=45, ha="right", fontsize=8)
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
        ax.set_xticks(x); ax.set_xticklabels(sh["safra"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% da safra")
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.9, title="categoria")
        ax.set_title(f"'{self.label(feature)}' — share por categoria no tempo",
                     fontsize=11, fontweight="bold", color="#15324a")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_variable_psi_by_safra(self, feature, time_col=None, figsize=(9.6, 3.4),
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
        ax.axhline(0.10, color="#caa000", lw=0.8, ls="--")
        ax.axhline(0.25, color="#d6453e", lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(ps["safra"], rotation=45, ha="right", fontsize=8)
        ax.set_xlim(-0.7, len(ps) - 0.3)
        ax.set_ylim(0, float(np.nanmax(ps["psi"])) * 1.16 + 0.05)
        ax.set_ylabel("PSI")
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
        ax.set_xticks(x); ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=8)
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

    def set_category(self, feature, categoria):
        """Categoriza a variável (ex.: 'manter', 'revisar', 'descartar')."""
        self.var_meta.setdefault(feature, {})["categoria"] = categoria
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
        return self

    def clear_manual_bins(self, feature):
        """Remove os bins manuais da variável (volta ao binning ótimo)."""
        self.var_meta.get(feature, {}).pop("splits", None)
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

    # ------------------------------------------------------------------
    # C) Modelo
    # ------------------------------------------------------------------
    def _build_pipeline(self, features, algorithm, hyperparams):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

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
        est = _build_estimator(algorithm, self.task_type, hyperparams)
        return Pipeline([("pre", pre), ("est", est)])

    def fit(self, algorithm=None, hyperparams=None, features=None):
        """Treina um modelo na amostra de referência (DES) com as variáveis
        selecionadas (ou ``features``). ``algorithm`` default: logística
        (classificação) / linear (regressão). Calcula ``score_`` para todas as linhas."""
        if algorithm is None:
            algorithm = "logistica" if self.task_type == "classification" else "linear"
        feats = list(features) if features is not None else self.selected_features()
        if not feats:
            feats = list(self.candidates)
        self.model_features = feats
        self.algorithm = algorithm
        self.hyperparams = dict(hyperparams or {})

        fit_df = self._frame(self.ref_sample)
        fit_df = fit_df[fit_df[self.target].notna()]
        X = fit_df[feats]
        y = fit_df[self.target]
        if self.task_type == "classification":
            y = y.astype(int)
        self.model = self._build_pipeline(feats, algorithm, hyperparams)
        self.model.fit(X, y)
        self.score_ = self._compute_score(self.df)
        self._shap_cache = {}
        return self

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
            if use_labels and nm in self.feature_labels:
                nm = self.feature_labels[nm]
            out.append(nm)
        return out

    def model_coefficients(self, use_labels=True) -> pd.DataFrame:
        """Coeficientes do modelo **linear/logístico** ajustado: ``termo``, ``coef``
        e — na classificação — ``odds_ratio`` (``exp(coef)``). O intercepto fica em
        ``.attrs['intercept']``. Erro para modelos não-lineares (use SHAP)."""
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
        rows = []
        for a in self._samples():
            mask = (pd.Series(True, index=self.df.index) if self.sample_col is None
                    else self.df[self.sample_col] == a)
            y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
            sc = self.score_[mask].to_numpy(dtype="float64")
            ok = ~np.isnan(y) & ~np.isnan(sc)
            y, sc = y[ok], sc[ok]
            if y.size == 0:
                continue
            m = (classification_metrics(y, sc) if self.task_type == "classification"
                 else regression_metrics(y, sc))
            rows.append({"amostra": a, "n": int(y.size), **m})
        return pd.DataFrame(rows)

    def metric_shifts(self) -> dict:
        """Variação de cada métrica DES→OOT (oot − des)."""
        m = self.metrics().set_index("amostra")
        oot = self._oot_sample()
        if self.ref_sample not in m.index or oot not in m.index or oot == self.ref_sample:
            return {}
        cols = [c for c in m.columns if c != "n"]
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
        fig.suptitle("SHAP — contribuição por feature", fontsize=11,
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
                      alpha=0.05, label_style=None):
        """Segmenta o score em ratings ordenados. ``method`` ∈ {decis, quantil,
        arvore, optbin}; ``n_ratings`` é o número-alvo de faixas (a fusão monotônica
        pode reduzi-lo). Reaproveita :mod:`yggdrasil.ratings`."""
        if self.score_ is None:
            raise RuntimeError("Gere o score antes (fit / set_model).")
        if method not in RATING_REGISTRY:
            raise ValueError(f"Método de rating desconhecido: {method!r}. "
                             f"Opções: {sorted(RATING_REGISTRY)}")
        n = int(n_ratings)
        if method == "decis":
            strat = RATING_REGISTRY[method](n=n)
        elif method == "quantil":
            strat = RATING_REGISTRY[method](step=1.0 / max(n, 1), alpha=alpha)
        elif method == "arvore":
            strat = RATING_REGISTRY[method](max_leaf_nodes=n, alpha=alpha)
        else:  # optbin
            strat = RATING_REGISTRY[method](max_n_bins=n)
        # respeita as flags quando a estratégia as expõe
        if method != "decis":
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
        ref_mask = (pd.Series(True, index=self.df.index) if self.sample_col is None
                    else self.df[self.sample_col] == self.ref_sample)
        n_ref_tot = max(int(ref_mask.sum()), 1)
        rows = []
        for lab in labels:
            m_ref = (rating == lab) & ref_mask
            row = {"rating": lab, "n": int(m_ref.sum()),
                   "repr_%": round(100 * int(m_ref.sum()) / n_ref_tot, 1)}
            for a in self._samples():
                am = (pd.Series(True, index=self.df.index) if self.sample_col is None
                      else self.df[self.sample_col] == a)
                row[f"{prefix}_{a}"] = round(self._risco(self.df.loc[(rating == lab) & am,
                                                                     self.target]), 4)
            rows.append(row)
        return pd.DataFrame(rows)

    def rating_inversion(self, time_col=None, sample=None, min_n=20) -> dict:
        """Inversão da ordem de risco ENTRE ratings (entre amostras e safras) —
        o estudo de folhas-irmãs do PD/LGD, aplicado às faixas de rating."""
        rating = self._rating_series()
        labels = self.rating_labels_
        risco_by = {}

        def risco(mask):
            return self._risco(self.df.loc[mask, self.target])

        # ordem de referência: pela média de risco na DES
        ref_mask = (pd.Series(True, index=self.df.index) if self.sample_col is None
                    else self.df[self.sample_col] == self.ref_sample)
        ref_risco = {lab: risco((rating == lab) & ref_mask) for lab in labels}
        ordered = sorted(labels, key=lambda l: (np.inf if pd.isna(ref_risco[l])
                                                else ref_risco[l]))
        # por amostra
        sample_rows = []
        for a in self._samples():
            am = (pd.Series(True, index=self.df.index) if self.sample_col is None
                  else self.df[self.sample_col] == a)
            vals = {lab: risco((rating == lab) & am) for lab in labels}
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
                ax.text(x0, r, f"{r:.3f}", ha="center", va="bottom", fontsize=7.5,
                        color="#15324a")
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("event_rate" if self.task_type == "classification" else "alvo médio")
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
        for k, a in enumerate(samples):
            am = (pd.Series(True, index=self.df.index) if self.sample_col is None
                  else self.df[self.sample_col] == a)
            n_a = max(int(am.sum()), 1)
            pct = [100 * int(((rating == l) & am).sum()) / n_a for l in labels]
            ax.bar(x + k * w, pct, width=w, label=a, alpha=0.9)
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("% da amostra"); ax.legend(fontsize=8)
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
        ax.set_xticks(x); ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("risco médio"); ax.set_xlabel("safra")
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

    def monotonicity_report(self) -> pd.DataFrame:
        """Verifica se o risco médio é monotônico ao longo dos ratings, por amostra."""
        rating = self._rating_series()
        labels = self.rating_labels_
        rows = []
        for a in self._samples():
            am = (pd.Series(True, index=self.df.index) if self.sample_col is None
                  else self.df[self.sample_col] == a)
            risco = [self._risco(self.df.loc[(rating == l) & am, self.target]) for l in labels]
            trend, n_inv = _trend(risco)
            rows.append({"amostra": a, "monotonico": n_inv == 0,
                         "tendencia": trend, "n_inversoes": n_inv})
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

    def predict(self, X: pd.DataFrame, col_score="score", col_rating="rating") -> pd.DataFrame:
        """Aplica modelo (+rating) a novos dados. Devolve score e rating por linha."""
        if self.model is None:
            raise RuntimeError("Ajuste/defina o modelo antes (fit / set_model).")
        sc = pd.Series(self._predict_score_array(self.model, X[self.model_features]),
                       index=X.index, name=col_score, dtype="float64")
        out = pd.DataFrame({col_score: sc}, index=X.index)
        if self.rating_strategy is not None:
            wf = pd.DataFrame({"score": sc}, index=X.index)
            cfg = self._make_cfg("_amostra")
            wf["_amostra"] = self.ref_sample
            out[col_rating] = self.rating_strategy.transform(wf, cfg).values
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
