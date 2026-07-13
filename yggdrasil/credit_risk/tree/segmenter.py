"""
TreeSegmenter
=============
Construtor sequencial e híbrido de **árvore de segmentação** para risco de
crédito, **unificado por** ``task_type``:

- ``task_type="classification"`` — alvo **binário** (ex.: 0 = adimplente,
  1 = default). Binning ótimo binário (``OptimalBinning``), IV WoE (Siddiqi),
  métricas de discriminação **KS, ROC/AUC, Gini, Acurácia, F1**.
- ``task_type="regression"`` — alvo **contínuo** em [0, 1]. Binning
  ótimo contínuo (``ContinuousOptimalBinning``), IV contínuo (desvio médio
  ponderado), métricas **MAE, RMSE, R²**.

Em ambos os modos: cresce em camadas (`grow`) por OPTIMAL BINNING ou CORTES
MANUAIS, com preview (`show_grow`), poda por representatividade/separação
(`prune`), PSI entre amostras (`psi`), régua aplicável em pandas (`predict`) e
Spark (`to_pyspark`/`apply_spark`) e registro no MLflow (`log_to_mlflow`).

Substitui as antigas classes ``SequentialPDSegmenter``/``SequentialLGDSegmenter``
(uma só classe parametrizada pelo tipo de problema).

Contexto: parâmetros de risco de crédito sob Resolução CMN 4.966/2021 e IFRS 9.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from optbinning import ContinuousOptimalBinning, OptimalBinning
except ImportError:  # pragma: no cover
    OptimalBinning = None
    ContinuousOptimalBinning = None

# scipy.stats no topo (e não dentro de _pair_pvalue): os testes de irmãs rodam por
# par a cada rodada de prune/auto_merge e em todo _refresh — pagar o lookup do
# import por chamada era desperdício no hot path. import único, resolvido uma vez.
try:
    from scipy.stats import mannwhitneyu, ttest_ind
except Exception:  # pragma: no cover
    mannwhitneyu = None
    ttest_ind = None

# helpers puros compartilhados com o ModelSegmenter (fonte única — sem drift)
from .._common import (
    fmt as _fmt,
    fmt_safras as _fmt_safras,
    classifica_psi as _classifica_psi,
    classifica_iv as _classifica_iv,
    count_inversions as _count_inversions,
    fit_optbinning_splits as _fit_optbinning_splits,
)

TASK_TYPES = ("classification", "regression")


# ======================================================================
# Helpers (formatação/classificação/optbinning vêm de credit_risk._common)
# ======================================================================
def _intervalo_por_extenso(lo: float, hi: float) -> str:
    """Converte um intervalo (lo, hi] em texto legível."""
    if lo == -np.inf:
        return f"até {_fmt(hi)}"
    if hi == np.inf:
        return f"acima de {_fmt(lo)}"
    return f"entre {_fmt(lo)} e {_fmt(hi)}"


# ======================================================================
# Critérios de split (alternativos ao binning ótimo do optbinning).
# Cada critério avalia uma partição BINÁRIA (CART-style): dado o alvo nos dois
# filhos, devolve um score (maior = melhor). Usados quando o usuário escolhe o
# critério em grow/fit_auto (criterion != "optbin").
# ======================================================================
CRITERIA_CLASSIFICATION = ("gini", "entropy", "ks", "iv", "chi2")
CRITERIA_REGRESSION = ("variance", "mae", "ftest")


def _gini(y):
    p = y.mean() if len(y) else 0.0
    return 1.0 - p * p - (1 - p) * (1 - p)


def _entropy(y):
    p = y.mean() if len(y) else 0.0
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _split_score(yl, yr, criterion, is_clf):
    """Score (maior = melhor) de dividir o alvo do pai em filhos `yl`/`yr`."""
    nl, nr = len(yl), len(yr)
    n = nl + nr
    if nl == 0 or nr == 0:
        return -np.inf
    yp = np.concatenate([yl, yr])
    wl, wr = nl / n, nr / n
    if is_clf:
        if criterion == "gini":
            return _gini(yp) - (wl * _gini(yl) + wr * _gini(yr))
        if criterion == "entropy":
            return _entropy(yp) - (wl * _entropy(yl) + wr * _entropy(yr))
        if criterion == "ks":
            tb, tg = yp.sum(), (yp == 0).sum()
            if tb == 0 or tg == 0:
                return -np.inf
            return abs(yl.sum() / tb - (yl == 0).sum() / tg)      # separação good/bad
        if criterion == "iv":
            tb, tg = yp.sum(), (yp == 0).sum()
            if tb == 0 or tg == 0:
                return -np.inf
            iv = 0.0
            for g in (yl, yr):
                db, dg = g.sum() / tb, (g == 0).sum() / tg
                if db > 0 and dg > 0:
                    iv += (dg - db) * np.log(dg / db)
            return iv
        if criterion == "chi2":
            from scipy.stats import chi2_contingency
            tab = np.array([[yl.sum(), (yl == 0).sum()], [yr.sum(), (yr == 0).sum()]])
            if (tab.sum(0) == 0).any() or (tab.sum(1) == 0).any():
                return -np.inf
            return float(chi2_contingency(tab)[0])                 # estatística qui-quadrado
    else:
        if criterion == "variance":
            return yp.var() - (wl * yl.var() + wr * yr.var())
        if criterion == "mae":
            def _mae(a):
                return np.mean(np.abs(a - np.median(a))) if len(a) else 0.0
            return _mae(yp) - (wl * _mae(yl) + wr * _mae(yr))
        if criterion == "ftest":
            from scipy.stats import f_oneway
            if np.unique(yl).size < 2 and np.unique(yr).size < 2:
                return -np.inf
            try:
                return float(f_oneway(yl, yr)[0])
            except Exception:
                return -np.inf
    raise ValueError(f"Critério de split desconhecido: {criterion!r}")


def _best_numeric_cut(x, y, criterion, is_clf, min_n):
    """Melhor corte único em `x` (numérico) pelo critério. Devolve o corte ou None."""
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[ok], y[ok]
    if x.size < 2 * min_n or np.unique(x).size < 2:
        return None
    cand = np.unique(np.quantile(x, np.linspace(0.02, 0.98, 49)))
    best_t, best_s = None, -np.inf
    for t in cand:
        left = x <= t
        nl = int(left.sum()); nr = x.size - nl
        if nl < min_n or nr < min_n:
            continue
        s = _split_score(y[left], y[~left], criterion, is_clf)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t if (best_t is not None and np.isfinite(best_s)) else None


def _best_categorical_split(xs, y, criterion, is_clf, min_n):
    """Melhor partição binária das categorias pelo critério (ordena por alvo médio
    e testa cada ponto de corte). Devolve (cats_esq, cats_dir) ou None."""
    cats = list(np.unique(xs))
    if len(cats) < 2:
        return None
    medias = {c: y[xs == c].mean() for c in cats}
    ordem = sorted(cats, key=lambda c: medias[c])
    best, best_s = None, -np.inf
    for k in range(1, len(ordem)):
        left_cats = set(ordem[:k])
        left = np.array([c in left_cats for c in xs])
        nl = int(left.sum()); nr = xs.size - nl
        if nl < min_n or nr < min_n:
            continue
        s = _split_score(y[left], y[~left], criterion, is_clf)
        if s > best_s:
            best_s, best = s, (ordem[:k], ordem[k:])
    return best if (best is not None and np.isfinite(best_s)) else None


def _match_conditions_pandas(df, conditions):
    """Máscara das linhas de `df` que satisfazem todas as condições do caminho.

    Condições numéricas/categóricas podem trazer ``include_na=True``: nesse caso
    a faixa também captura os faltantes (bin populado **OU** faltante).
    """
    m = pd.Series(True, index=df.index)
    for c in conditions:
        feat = c["feature"]
        if c["kind"] == "na":
            m &= df[feat].isna()
        elif c["kind"] == "num":
            lo, hi = c.get("lo"), c.get("hi")
            sub = pd.Series(True, index=df.index)
            if lo is not None:
                sub &= df[feat] > lo
            if hi is not None:
                sub &= df[feat] <= hi
            if c.get("include_na"):
                sub |= df[feat].isna()
            m &= sub
        else:
            cats = [str(x) for x in c["cats"]]
            sub = df[feat].astype(str).isin(cats)
            if c.get("include_na"):
                sub |= df[feat].isna()
            m &= sub
    return m


def _aplicar_regua_pandas(regua, df, col_seg="segmento",
                          col_nota="nota", col_valor="valor_regua"):
    """Aplica uma régua (dict de folhas) a um DataFrame pandas."""
    seg = pd.Series(pd.NA, index=df.index, dtype="object")
    nota = pd.Series(pd.NA, index=df.index, dtype="Int64")
    pdcol = pd.Series(np.nan, index=df.index, dtype="float64")
    for leaf in regua["leaves"]:
        m = _match_conditions_pandas(df, leaf["conditions"])
        seg[m] = leaf["id"]
        nota[m] = leaf["nota"]
        pdcol[m] = leaf["pd"]
    # cobertura no scoring: linhas que não caíram em NENHUM segmento ficam nulas.
    # Não é silencioso — avisa com a causa provável (faltante sem rota quando o
    # split usou na_to_worst=False, ou categoria não vista no ajuste).
    n_orfas = int(seg.isna().sum())
    if n_orfas:
        import warnings
        warnings.warn(
            f"{n_orfas} linha(s) sem segmento na régua (segmento nulo): valor "
            f"faltante numa variável sem rota de faltantes (grow com "
            f"na_to_worst=False) ou categoria não vista no ajuste.",
            stacklevel=2)
    return pd.DataFrame({col_seg: seg, col_nota: nota, col_valor: pdcol}, index=df.index)


# ======================================================================
# Classe principal
# ======================================================================
class TreeSegmenter:
    """Árvore de segmentação unificada (classificação/regressão) com binning
    ótimo/manual, poda e PSI. O comportamento por tipo de alvo é escolhido por
    ``task_type`` ("classification" p/ alvo · "regression" p/ alvo)."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str = "target",
        task_type: str = "classification",
        sample_col: str | None = None,
        ref_sample: str = "DES",
        feature_labels: dict[str, str] | None = None,
        problem_label: str | None = None,
        min_leaf_rows: int = 50,
        date_col: str | None = None,
        verbose: bool = True,
    ):
        if task_type not in TASK_TYPES:
            raise ValueError(
                f"task_type inválido: {task_type!r}. Use um de {TASK_TYPES}.")
        self.task_type = task_type
        self._is_clf = task_type == "classification"
        # rótulo do alvo nos gráficos/relatórios: `problem_label` se informado,
        # senão o nome da coluna alvo. Nunca rótulo fixo — o tipo de alvo
        # (binário/contínuo) segue vindo de `task_type`.
        self.problem_label = problem_label
        self._risk_word = problem_label or target
        self._risk_mean = f"média de {self._risk_word}"
        # cache do binning ótimo por folha (o solver do optbinning é o gargalo) —
        # ver _resolve_bins_cached; chaveado por id() da máscara da folha
        self._bins_cache: dict = {}
        # ---- caches de agregados por VERSÃO da árvore (desempenho) -------------
        # Toda mutação estrutural (grow/prune/collapse/merge_*/auto_merge/load/
        # fit_auto) chama _bump_version(). leaves()/_leaf_order()/_grade_map()/
        # metrics()/psi() são reconstruídos ~4-5× por _refresh da UI varrendo
        # máscaras full-length por folha; memoizá-los por versão elimina esse
        # recompute redundante (o maior gargalo da UI da árvore no Databricks).
        self._tree_version: int = 0
        self._agg_cache: dict = {}        # (nome, *args, version) -> resultado
        # alvo por folha restrito à referência — chaveado por id(mask),
        # invalida sozinho quando a folha é refeita (mesma ideia do _bins_cache)
        self._leaf_target_cache: dict = {}
        # máscara por ASSINATURA DE CONDIÇÕES: como (condições → máscara) é função
        # PURA do df (que não muda), o cache nunca fica stale. Acelera undo/redo —
        # _load_segments reconstruía toda máscara via _match_conditions_pandas a
        # cada desfazer/refazer; agora os segmentos inalterados são cache-hit.
        self._mask_cache_by_conds: dict = {}
        # classe de optbinning e nome do kwarg de diferença mínima por tipo de alvo
        self._OptBin = OptimalBinning if self._is_clf else ContinuousOptimalBinning
        self._diff_kwarg = "min_event_rate_diff" if self._is_clf else "min_mean_diff"
        if self._OptBin is None:
            raise ImportError("optbinning não instalado. Rode: pip install optbinning")
        if target not in df.columns:
            raise ValueError(f"Coluna alvo '{target}' não está no DataFrame "
                             f"(colunas: {list(df.columns)}).")

        self.df = df.copy()
        self.target = target
        self.sample_col = sample_col
        self.ref_sample = ref_sample
        # coluna de DATA/safra: NÃO entra na modelagem (fica fora das features);
        # serve só para os gráficos no tempo (alvo/variável por safra, PSI por safra).
        self.date_col = date_col
        if date_col is not None and date_col not in self.df.columns:
            raise ValueError(f"Coluna de data '{date_col}' não está no DataFrame "
                             f"(colunas: {list(self.df.columns)}).")
        # rótulos amigáveis por variável para a descrição por extenso
        self.feature_labels = feature_labels or {}
        # mínimo de linhas (na amostra de ajuste) para tentar binning ótimo
        self.min_leaf_rows = min_leaf_rows

        if sample_col is not None:
            if sample_col not in self.df.columns:
                raise ValueError(
                    f"Coluna de amostra '{sample_col}' não está no DataFrame "
                    f"(colunas: {list(self.df.columns)}).")
            amostras = self.df[sample_col].dropna().unique().tolist()
            if ref_sample not in amostras:
                raise ValueError(
                    f"Amostra de referência '{ref_sample}' não encontrada em "
                    f"'{sample_col}'. Disponíveis: {amostras}"
                )
            if verbose:
                print(f"[init] amostras: {amostras} | referência PSI = {ref_sample}")

        # máscaras booleanas por amostra, pré-computadas UMA vez (linhas e
        # sample_col não mudam após a construção). PSI/CSI/metrics recriavam
        # `df[sample_col]==a` full-length a cada chamada — caro e repetido por
        # folha×amostra. Aqui vira lookup. _mask_ref é a máscara da referência.
        if sample_col is not None:
            self._sample_masks = {a: (self.df[sample_col] == a)
                                  for a in self.df[sample_col].dropna().unique()}
            self._mask_ref = self._sample_masks.get(ref_sample)
        else:
            self._sample_masks = {}
            self._mask_ref = None

        # cada segmento: id -> dict(mask, label, depth, is_leaf, path, parent, conditions)
        self.segments: dict[str, dict] = {
            "root": {
                "mask": pd.Series(True, index=df.index),
                "label": "root",
                "depth": 0,
                "is_leaf": True,
                "path": [],
                "parent": None,
                "conditions": [],   # lista de (feature, lo, hi) acumulada no caminho
            }
        }
        self.history: list[dict] = []
        self._psi_detalhe: list[dict] = []
        self._csi_detalhe: list[dict] = []

    # ------------------------------------------------------------------
    # Versão da árvore + memoização de agregados (desempenho)
    # ------------------------------------------------------------------
    def _bump_version(self):
        """Marca a árvore como alterada: invalida os caches de agregado por versão.
        Chamado por TODA mutação estrutural (grow/prune/collapse/merge_*/
        auto_merge/_load_segments/fit_auto)."""
        self._tree_version += 1
        if len(self._agg_cache) > 64:        # backstop: só guardamos a versão atual
            self._agg_cache.clear()

    def _agg_memo(self, key, build):
        """Memoiza ``build()`` por (key, versão-da-árvore). Devolve sempre uma
        cópia defensiva (DataFrame/Series/dict/list) para o chamador não corromper
        o cache. ``build`` é uma função sem argumentos."""
        import copy
        ck = (key, self._tree_version)
        hit = self._agg_cache.get(ck)
        if hit is None:
            hit = build()
            self._agg_cache[ck] = hit
        if isinstance(hit, (pd.DataFrame, pd.Series)):
            return hit.copy()
        if isinstance(hit, (dict, list)):
            # deepcopy: valores ANINHADOS mutáveis (ex.: grade_map = {'nota':dict,
            # 'desc':dict}) não podem vazar por referência e contaminar o cache.
            return copy.deepcopy(hit)
        return hit

    def _nonfeature_cols(self) -> set:
        """Colunas que NÃO entram na modelagem (não viram variáveis candidatas):
        o alvo, a coluna de amostra, a coluna de data de referência (``date_col``)
        e **qualquer coluna datetime** — uma data nunca é variável do modelo aqui
        (e o optbinning não a bina). Para datas que não sejam datetime (ex.: safra
        yyyymm inteira), passe ``date_col`` no construtor para excluí-la também."""
        skip = {self.target, self.sample_col, self.date_col}
        skip.discard(None)
        for c in self.df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(self.df[c]):
                    skip.add(c)
            except Exception:
                pass
        return skip

    # ------------------------------------------------------------------
    # Helpers de bin genéricos (numérico OU categórico)
    # ------------------------------------------------------------------
    def _detect_kind(self, sub, feature, dtype):
        """Decide se a variável é tratada como 'num' ou 'cat'."""
        if dtype in ("num", "cat"):
            return dtype
        col = sub[feature]
        if pd.api.types.is_bool_dtype(col):
            return "cat"
        return "num" if pd.api.types.is_numeric_dtype(col) else "cat"

    def _fit_frame(self, sub, min_bin_size):
        """Amostra usada para AJUSTAR o binning ótimo (só DES, se houver)."""
        if self.sample_col is None:
            return sub
        fit_data = sub[sub[self.sample_col] == self.ref_sample]
        if len(fit_data) < 2 * int(min_bin_size * len(self.df)):
            return sub  # ramo com pouco DES: usa o ramo inteiro
        return fit_data

    def _mask_in(self, frame, feature, b):
        """Máscara booleana das linhas de `frame` que caem no bin `b`."""
        if b["kind"] == "na":
            return frame[feature].isna()
        if b["kind"] == "num":
            return frame[feature].between(b["lo"], b["hi"], inclusive="right")
        return frame[feature].astype(str).isin(b["cats"])

    @staticmethod
    def _bin_label(feature, b):
        if b["kind"] == "na":
            return f"{feature}: (faltante)"
        if b["kind"] == "num":
            return f"{feature}: ({_fmt(b['lo'])}, {_fmt(b['hi'])}]"
        return f"{feature}: {{{', '.join(map(str, b['cats']))}}}"

    @staticmethod
    def _bin_condition(feature, b):
        if b["kind"] == "na":
            return {"feature": feature, "kind": "na"}
        if b["kind"] == "num":
            return {"feature": feature, "kind": "num", "lo": b["lo"], "hi": b["hi"]}
        return {"feature": feature, "kind": "cat", "cats": list(b["cats"])}

    @staticmethod
    def _splits_key(sp):
        """Chave hashável dos splits (cortes numéricos ou grupos categóricos)."""
        if sp is None:
            return None
        if sp and isinstance(sp[0], (list, tuple)):
            return tuple(tuple(g) for g in sp)
        return tuple(sp)

    def _resolve_bins_cached(self, sid, feature, splits, dtype, max_n_bins, min_bin_size,
                             max_bin_size=None, relax_max=False, min_mean_diff=0.0,
                             criterion="optbin"):
        """Memoiza :meth:`_resolve_bins` para o SUBFRAME COMPLETO da folha ``sid``
        (o solver CP-SAT do optbinning é o gargalo das trocas de variável/sugestões).

        A chave inclui o ``id`` do objeto-máscara da folha — que é trocado sempre
        que a folha é refeita (grow/merge/prune/collapse/auto_merge/load/fit_auto),
        então o cache se invalida sozinho, POR FOLHA, sem precisar versionar a
        árvore nem caçar todos os pontos de mutação. O ``mask`` é guardado no valor
        para o seu ``id`` não ser reciclado pelo GC. Use só com o subframe completo
        da folha (o fit do optbin já restringe à DES internamente)."""
        nsid = sid if (sid in self.segments) else "root"
        mask = self.segments[nsid]["mask"]
        key = (id(mask), feature, self._splits_key(splits), dtype, max_n_bins,
               min_bin_size, max_bin_size, relax_max, round(float(min_mean_diff), 9),
               criterion)
        hit = self._bins_cache.get(key)
        if hit is not None:
            return hit[0], hit[1], hit[2]
        res = self._resolve_bins(self.df[mask], feature, splits, dtype, max_n_bins,
                                 min_bin_size, max_bin_size, relax_max, min_mean_diff,
                                 criterion)
        if len(self._bins_cache) > 6000:        # backstop de memória
            self._bins_cache.clear()
        self._bins_cache[key] = (res[0], res[1], res[2], mask)
        return res[0], res[1], res[2]

    def _resolve_bins(self, sub, feature, splits, dtype, max_n_bins, min_bin_size,
                      max_bin_size=None, relax_max=False, min_mean_diff=0.0,
                      criterion="optbin"):
        """Resolve os bins de um ramo. Devolve (bins, modo, kind).
        - num: bins = [{'kind':'num','lo','hi'}, ...]
        - cat: bins = [{'kind':'cat','cats':[...]}, ...]
        No modo ótimo (``criterion="optbin"``), o binning é ajustado só na amostra
        de referência (DES) — binário (``OptimalBinning``) p/ classificação,
        contínuo (``ContinuousOptimalBinning``) p/ regressão. Com outro
        ``criterion`` (gini/entropy/ks/iv/chi2 · variance/mae/ftest) faz um split
        BINÁRIO no melhor corte por esse critério. No modo manual: splits numérico
        = lista de cortes; splits categórico = lista de grupos.
        min_mean_diff: diferença mínima do alvo médio (taxa de default em
        classificação) exigida entre bins consecutivas no binning ótimo; 0 = sem
        restrição.
        """
        kind = self._detect_kind(sub, feature, dtype)

        if kind == "num":
            if splits is not None:
                lo, hi = sub[feature].min(), sub[feature].max()
                cortes = [s for s in sorted(splits) if lo < s < hi]
                modo = "manual"
            else:
                fit = self._fit_frame(sub, min_bin_size)
                x = fit[feature].to_numpy(dtype="float64")
                y = fit[self.target].to_numpy(dtype="float64")
                ok = ~np.isnan(y)
                x, y = x[ok], y[ok]                         # alvo NaN não entra no ajuste
                yfit = y.astype(int) if self._is_clf else y    # binário 0/1 só na classif.
                x_obs = x[~np.isnan(x)]
                # classificação exige as 2 classes presentes; regressão não
                degenerado = (len(yfit) < 4 or x_obs.size == 0
                              or np.unique(x_obs).size < 2
                              or (self._is_clf and np.unique(yfit).size < 2))
                if degenerado:
                    cortes = []                            # dados degenerados → sem corte
                elif criterion != "optbin":
                    min_n = max(1, int(min_bin_size * len(yfit)))
                    t = _best_numeric_cut(x, yfit, criterion, self._is_clf, min_n)
                    cortes = [t] if t is not None else []
                else:
                    def _opt(mnb, mbs):
                        b = self._OptBin(
                            name=feature, dtype="numerical", max_n_bins=mnb,
                            min_bin_size=min_bin_size, max_bin_size=mbs,
                            monotonic_trend="auto_asc_desc",
                            **{self._diff_kwarg: min_mean_diff})
                        return _fit_optbinning_splits(b, x, yfit)
                    cortes = _opt(max_n_bins, max_bin_size)
                    if not cortes and max_bin_size is not None and relax_max:
                        # o máximo pode deixar o problema INFEASIBLE p/ este nº de
                        # bins/amostra: tenta mais bins e, por fim, relaxa o máximo
                        for k in range(max_n_bins + 1, 9):
                            cortes = _opt(k, max_bin_size)
                            if cortes:
                                break
                        if not cortes:
                            cortes = _opt(max_n_bins, None)
                modo = "ótimo" if criterion == "optbin" else criterion
            if not cortes:
                return [], modo, kind
            edges = [-np.inf, *cortes, np.inf]
            bins = [{"kind": "num", "lo": edges[i], "hi": edges[i + 1]}
                    for i in range(len(edges) - 1)]
            if sub[feature].isna().any():
                bins.append({"kind": "na"})
            return bins, modo, kind

        # ---- categórico ----
        na_present = bool(sub[feature].isna().any())
        if splits is not None:
            grupos = [list(g) for g in splits]   # lista de grupos de categorias
            flat = [str(c) for g in grupos for c in g]
            if len(flat) != len(set(flat)):      # grupos devem ser disjuntos
                dup = sorted({c for c in flat if flat.count(c) > 1})
                raise ValueError(
                    f"Grupos categóricos manuais têm categoria(s) repetida(s): {dup}. "
                    "Cada categoria deve estar em um único grupo.")
            modo = "manual"
        else:
            fit = self._fit_frame(sub, min_bin_size)
            fit = fit[fit[feature].notna() & fit[self.target].notna()]  # NaN fora do ajuste
            xs = fit[feature].astype(str).to_numpy()
            yf = fit[self.target].to_numpy(dtype="float64")
            ys = yf.astype(int) if self._is_clf else yf       # binário só na classif.
            degenerado = (len(ys) < 4 or np.unique(xs).size < 2
                          or (self._is_clf and np.unique(ys).size < 2))
            if degenerado:
                grupos = []                          # dados degenerados → sem grupos
            elif criterion != "optbin":
                min_n = max(1, int(min_bin_size * len(ys)))
                par = _best_categorical_split(xs, ys, criterion, self._is_clf, min_n)
                grupos = [list(par[0]), list(par[1])] if par is not None else []
            else:
                b = self._OptBin(
                    name=feature, dtype="categorical", max_n_bins=max_n_bins,
                    min_bin_size=min_bin_size, max_bin_size=max_bin_size,
                    monotonic_trend="auto_asc_desc",
                    **{self._diff_kwarg: min_mean_diff})
                grupos = [list(arr) for arr in _fit_optbinning_splits(b, xs, ys)]
            modo = "ótimo" if criterion == "optbin" else criterion
        _NA_TOK = {"nan", "NaN", "<NA>", "None"}
        bins = []
        for g in grupos:
            cats = [str(c) for c in g if str(c) not in _NA_TOK]
            if cats:
                bins.append({"kind": "cat", "cats": cats})
        if bins and na_present:
            bins.append({"kind": "na"})
        return bins, modo, kind

    # ------------------------------------------------------------------
    # Tabela de bins: taxa de default (alvo) + representatividade (num ou cat)
    # ------------------------------------------------------------------
    def _bin_table(self, sub, feature, bins, n_ref):
        linhas, is_na = [], []
        for b in bins:
            m = self._mask_in(sub, feature, b)
            s = sub.loc[m, self.target]
            if len(s) == 0:
                continue
            linhas.append({
                "faixa": self._bin_label(feature, b),
                "n": len(s),
                "repr_%": round(100 * len(s) / n_ref, 1),
                "valor_medio": round(s.mean(), 4),
                "valor_std": round(s.std(), 4),
            })
            is_na.append(b.get("kind") == "na")
        tbl = pd.DataFrame(linhas)
        if tbl.empty:                       # todos os bins vazios
            tbl.attrs["mono_ok"] = True
            return tbl
        # a bin de faltantes (NA) não pertence à sequência ordenada pela variável — a
        # média dos ausentes fica fixa no fim e quebraria/mascararia a monotonicidade
        # das faixas reais; por isso é EXCLUÍDA do teste de mono_ok.
        means = tbl.loc[~np.asarray(is_na), "valor_medio"]
        tbl.attrs["mono_ok"] = bool(
            len(means) < 2 or means.is_monotonic_increasing or means.is_monotonic_decreasing
        )
        return tbl

    # ------------------------------------------------------------------
    # SHOW_GROW: preview do split (não altera estado)
    #   dtype: None=auto-detecta, 'num' ou 'cat' para forçar
    # ------------------------------------------------------------------
    def show_grow(self, feature, splits=None, dtype=None, max_n_bins=4,
                  min_bin_size=0.05, only_segments=None, max_bin_size=None,
                  min_mean_diff=0.0, criterion="optbin"):
        targets = {
            sid: s for sid, s in self.segments.items()
            if s["is_leaf"] and (only_segments is None or sid in only_segments)
        }
        n_total = len(self.df)
        previews = {}
        for sid, seg in targets.items():
            sub = self.df[seg["mask"]]
            # cache por id(mask): o Preview popula o solver e o .grow() seguinte
            # (mesma folha/feature/args) reaproveita, sem rodar optbinning 2×.
            bins, modo, kind = self._resolve_bins_cached(
                sid, feature, splits, dtype, max_n_bins, min_bin_size, max_bin_size,
                min_mean_diff=min_mean_diff, criterion=criterion)
            if not bins:
                print(f"[{sid}] sem corte válido em '{feature}' ({modo})")
                continue
            tbl = self._bin_table(sub, feature, bins, n_total)
            previews[sid] = tbl
            print(f"\n┌─ PREVIEW: dividir '{sid}'")
            print(f"│  feature = {feature} | tipo = {kind} | modo = {modo}")
            print(f"│  monotonicidade de {self._risk_word} respeitada: {tbl.attrs['mono_ok']}")
            print(tbl.to_string(index=False))
        print("\n→ se aprovado, repita como .grow(...) com os mesmos argumentos")
        return previews

    def _assign_na_to_worst(self, filhos: dict):
        """Quando um split NÃO gera nó de faltantes (a variável não tinha missing
        no ajuste), roteia eventuais missings do scoring para a folha-irmã de PIOR
        risco — maior alvo (classificação) ou maior alvo (regressão) na referência
        (DES) — marcando ``include_na=True`` na condição dessa irmã. Atribuição
        conservadora; não faz nada se já houver um nó de faltantes entre as irmãs."""
        if not filhos or any(f["conditions"][-1].get("kind") == "na"
                             or f["conditions"][-1].get("include_na")
                             for f in filhos.values()):
            return
        ref = ((self.df[self.sample_col] == self.ref_sample).to_numpy()
               if self.sample_col is not None else np.ones(len(self.df), dtype=bool))
        y_all = self.df[self.target].to_numpy(dtype="float64")

        def _risco(f):
            yy = y_all[f["mask"].to_numpy() & ref]
            yy = yy[~np.isnan(yy)]
            return float(yy.mean()) if yy.size else -np.inf

        # desempate quando o risco é indefinido (folha sem dados na referência,
        # _risco == -inf): cai para a folha de MAIOR tamanho em vez de escolher a
        # primeira arbitrariamente — rota de missing mais representativa.
        pior = max(filhos, key=lambda cid: (_risco(filhos[cid]),
                                            int(filhos[cid]["mask"].sum())))
        filhos[pior]["conditions"][-1]["include_na"] = True

    # ------------------------------------------------------------------
    # GROW: efetiva o split (manual ou ótimo, num ou cat) em cada folha-alvo
    # ------------------------------------------------------------------
    def grow(self, feature, splits=None, dtype=None, max_n_bins=4,
             min_bin_size=0.05, only_segments=None, max_bin_size=None, relax_max=False,
             min_mean_diff=0.0, criterion="optbin", na_to_worst=True):
        self._bump_version()      # estrutura vai mudar → invalida agregados memoizados
        targets = {
            sid: s for sid, s in self.segments.items()
            if s["is_leaf"] and (only_segments is None or sid in only_segments)
        }
        novos, modo_usado = {}, None
        for sid, seg in targets.items():
            sub = self.df[seg["mask"]]
            if splits is None:
                n_fit = len(self._fit_frame(sub, min_bin_size))
                if n_fit < self.min_leaf_rows:
                    print(f"[{sid}] poucas linhas para binning ótimo "
                          f"({n_fit} < {self.min_leaf_rows}) — folha mantida")
                    continue
            bins, modo, kind = self._resolve_bins_cached(
                sid, feature, splits, dtype, max_n_bins, min_bin_size, max_bin_size,
                relax_max=relax_max, min_mean_diff=min_mean_diff, criterion=criterion)
            modo_usado = modo
            if not bins:
                print(f"[{sid}] sem corte válido em '{feature}' ({modo}) — folha mantida")
                continue
            filhos_sid = {}
            for b in bins:
                child_mask = seg["mask"] & self._mask_in(self.df, feature, b)
                if child_mask.sum() == 0:
                    continue
                child_label = self._bin_label(feature, b)
                child_id = f"{sid} | {child_label}" if sid != "root" else child_label
                filhos_sid[child_id] = {
                    "mask": child_mask,
                    "label": child_label,
                    "depth": seg["depth"] + 1,
                    "is_leaf": True,
                    "path": seg["path"] + [child_label],
                    "parent": sid,
                    "conditions": seg["conditions"] + [self._bin_condition(feature, b)],
                }
            if len(filhos_sid) < 2:        # 0/1 filho não-vazio = não separou de fato
                print(f"[{sid}] '{feature}' não separou ({len(filhos_sid)} filho) "
                      "— folha mantida")
                continue
            if na_to_worst:               # sem nó de faltantes → missings vão p/ a pior irmã
                self._assign_na_to_worst(filhos_sid)
            novos.update(filhos_sid)
            self.segments[sid]["is_leaf"] = False
        self.segments.update(novos)
        self.history.append({"feature": feature, "modo": modo_usado, "splits": splits})
        n_folhas = sum(s["is_leaf"] for s in self.segments.values())
        print(f"[grow] '{feature}' ({modo_usado}) criou {len(novos)} segmentos. "
              f"Folhas atuais: {n_folhas}")
        return self

    # ------------------------------------------------------------------
    # PRUNE: poda por FUSÃO de folhas-IRMÃS (mesmo pai). Em cada rodada funde um
    #   par de irmãs adjacentes que viole um dos critérios:
    #     - diferença de alvo médio entre as duas irmãs < `min_valor_gap` (ex.: 0.02
    #       = 2 p.p.) → separam pouco, devem ser unidas;
    #     - alguma das duas tem representatividade < `min_repr` % (imaterial) →
    #       é unida à irmã adjacente mais próxima em alvo.
    #   Prioriza o par de menor diferença de alvo. Só funde irmãs (o nó de
    #   faltantes não entra). Itera até nenhum par violar os critérios.
    # ------------------------------------------------------------------
    def prune(self, min_repr: float = 2.0, min_valor_gap: float | None = None,
              protect: set | None = None, verbose: bool = True,
              max_rounds: int = 1000):
        if min_valor_gap is None:
            min_valor_gap = 0.02 if self._is_clf else 0.03
        protect = set(protect or [])
        n_total = len(self.df)
        n_merges = 0
        for _ in range(max_rounds):
            folhas_por_pai: dict[str, list[str]] = {}
            for sid, s in self.segments.items():
                if s["is_leaf"]:
                    folhas_por_pai.setdefault(s["parent"], []).append(sid)

            melhor = None  # (gap, sid_direita_do_par, motivo)
            for pai in folhas_por_pai:
                if pai is None:
                    continue
                # Só pares de folhas TERMINAIS ADJACENTES (mesma fonte canônica de
                # auto_merge/plot_tree/teste de hipótese): um nó expandido no meio
                # QUEBRA a adjacência, então nunca pareamos folhas separadas por uma
                # irmã não-folha — o que faria merge_leaf abortar sem mudar nada e a
                # salvaguarda encerrar TODA a poda.
                for a, b in self._adjacent_sibling_pairs(pai):
                    if a in protect or b in protect:          # respeita folhas travadas
                        continue
                    ta, tb = self._leaf_target(a), self._leaf_target(b)
                    pda = float(ta.mean()) if len(ta) else np.nan     # alvo na referência (DES)
                    pdb = float(tb.mean()) if len(tb) else np.nan
                    gap = (abs(pdb - pda)
                           if not (pd.isna(pda) or pd.isna(pdb)) else np.inf)
                    repr_a = (100 * int(self.segments[a]["mask"].sum()) / n_total
                              if n_total else 0.0)
                    repr_b = (100 * int(self.segments[b]["mask"].sum()) / n_total
                              if n_total else 0.0)
                    viola_gap = gap < min_valor_gap
                    viola_repr = (repr_a < min_repr) or (repr_b < min_repr)
                    if not (viola_gap or viola_repr):
                        continue
                    motivo = "ΔPD" if viola_gap else "repr."
                    if melhor is None or gap < melhor[0]:
                        melhor = (gap, b, motivo)
            if melhor is None:
                break
            _, sid_dir, motivo = melhor
            antes = set(self.segments)
            self.merge_leaf(sid_dir, side="left", verbose=False)
            if set(self.segments) == antes:
                break  # salvaguarda contra laço infinito
            n_merges += 1
            if verbose:
                print(f"[prune] folhas-irmãs unidas ({motivo})")

        n_folhas = sum(s["is_leaf"] for s in self.segments.values())
        if verbose:
            print(f"[prune] {n_merges} fusão(ões) (repr<{min_repr}% ou ΔPD<{min_valor_gap}). "
                  f"Folhas finais: {n_folhas}")
        return self

    # ------------------------------------------------------------------
    # COLLAPSE: recolhe a subárvore enraizada em `sid`, transformando o nó
    #   de volta em folha (remove todos os descendentes). Desfaz um split.
    # ------------------------------------------------------------------
    def collapse(self, sid: str, verbose: bool = True):
        self._bump_version()      # estrutura vai mudar → invalida agregados memoizados
        if sid not in self.segments:
            if verbose:
                print(f"[collapse] segmento '{sid}' não existe")
            return self
        if self.segments[sid]["is_leaf"]:
            if verbose:
                print(f"[collapse] '{sid}' já é folha — nada a recolher")
            return self
        # coleta todos os descendentes (BFS pelos ponteiros de pai)
        descendentes, frente = [], [c for c, s in self.segments.items() if s["parent"] == sid]
        while frente:
            atual = frente.pop()
            descendentes.append(atual)
            frente.extend([c for c, s in self.segments.items() if s["parent"] == atual])
        for d in descendentes:
            self.segments.pop(d, None)
        self.segments[sid]["is_leaf"] = True
        if verbose:
            print(f"[collapse] '{sid}' recolhido — {len(descendentes)} segmento(s) removido(s)")
        return self

    # ------------------------------------------------------------------
    # MERGE_LEAF: funde uma folha com a folha-irmã adjacente (mesmo pai).
    #   side="left"  -> vizinha de menor corte (num) / menor alvo (cat)
    #   side="right" -> vizinha de maior corte (num) / maior alvo (cat)
    #   Numérico: os dois intervalos viram um só (lo=min, hi=max).
    #   Categórico: as categorias dos dois grupos são unidas.
    #   Se o pai ficar com uma única folha, a fusão equivale a recolhê-lo.
    # ------------------------------------------------------------------
    def merge_leaf(self, sid: str, side: str = "left", verbose: bool = True):
        self._bump_version()      # estrutura vai mudar → invalida agregados memoizados
        if sid not in self.segments:
            if verbose:
                print(f"[merge] segmento '{sid}' não existe")
            return self
        seg = self.segments[sid]
        if not seg["is_leaf"]:
            if verbose:
                print(f"[merge] '{sid}' não é folha")
            return self
        pai = seg["parent"]
        if pai is None:
            if verbose:
                print("[merge] a raiz não tem folhas vizinhas para fundir")
            return self

        cond = seg["conditions"][-1]
        feat, kind = cond["feature"], cond["kind"]
        if kind == "na":
            if verbose:
                print("[merge] o nó de faltantes não funde por vizinhança — use "
                      "merge_missing(<nó populado>) para juntá-lo a um bin da variável")
            return self

        # irmãos que dá para fundir: mesmo tipo de corte, excluindo o nó de faltantes
        irmaos = [c for c, s in self.segments.items()
                  if s["parent"] == pai and s["conditions"][-1]["kind"] != "na"]
        if kind == "num" and all(self.segments[c]["conditions"][-1]["kind"] == "num"
                                 for c in irmaos):
            irmaos.sort(key=lambda c: self.segments[c]["conditions"][-1]["lo"])
        else:
            # mesma chave (alvo na referência/DES) usada por prune/auto_merge/teste —
            # senão a ordem das irmãs diverge e a fusão erra o par ou aborta
            irmaos.sort(key=lambda c: (self._leaf_target(c).mean()
                                       if len(self._leaf_target(c)) else np.inf))

        i = irmaos.index(sid)
        j = i - 1 if side == "left" else i + 1
        lado = "à esquerda" if side == "left" else "à direita"
        if j < 0 or j >= len(irmaos):
            if verbose:
                print(f"[merge] '{sid}' não tem folha {lado} para fundir")
            return self
        viz = irmaos[j]
        if not self.segments[viz]["is_leaf"]:
            if verbose:
                print(f"[merge] a folha {lado} foi subdividida — recolha-a antes de fundir")
            return self

        cond_viz = self.segments[viz]["conditions"][-1]
        inc_na = bool(cond.get("include_na") or cond_viz.get("include_na"))
        if kind == "num":
            new_cond = {"feature": feat, "kind": "num",
                        "lo": min(cond["lo"], cond_viz["lo"]),
                        "hi": max(cond["hi"], cond_viz["hi"])}
        else:
            cats = list(dict.fromkeys(list(cond["cats"]) + list(cond_viz["cats"])))
            new_cond = {"feature": feat, "kind": "cat", "cats": cats}
        if inc_na:
            new_cond["include_na"] = True

        new_mask = seg["mask"] | self.segments[viz]["mask"]
        new_label = self._bin_label(feat, new_cond) + (" + faltante" if inc_na else "")
        new_id = new_label if pai == "root" else f"{pai} | {new_label}"
        pai_seg = self.segments[pai]
        merged = {
            "mask": new_mask, "label": new_label, "depth": seg["depth"],
            "is_leaf": True, "path": pai_seg["path"] + [new_label],
            "parent": pai, "conditions": pai_seg["conditions"] + [new_cond],
        }
        self.segments.pop(sid, None)
        self.segments.pop(viz, None)
        self.segments[new_id] = merged

        filhos_pai = [c for c, s in self.segments.items() if s["parent"] == pai]
        if len(filhos_pai) == 1:
            self.segments.pop(new_id, None)
            self.segments[pai]["is_leaf"] = True
            if verbose:
                print(f"[merge] folhas fundidas {lado}; pai '{pai}' voltou a ser folha")
            return self
        if verbose:
            print(f"[merge] '{sid}' fundida com a folha {lado} → '{new_id}'")
        return self

    # ------------------------------------------------------------------
    # MERGE_MISSING: junta o nó de FALTANTES (na) do mesmo split DENTRO de um
    #   nó POPULADO da variável. A regra do destino passa a ser "<bin> OU
    #   faltante" (condição com include_na=True). Selecione sempre o nó populado
    #   de destino; o nó de faltantes é localizado automaticamente entre os
    #   irmãos. Diferente de merge_leaf (vizinhança), aqui a fusão é sempre
    #   válida porque faltante é disjunto de qualquer bin populado.
    # ------------------------------------------------------------------
    def merge_missing(self, sid: str, verbose: bool = True):
        if sid not in self.segments:
            if verbose:
                print(f"[merge_missing] segmento '{sid}' não existe")
            return self
        seg = self.segments[sid]
        if not seg["is_leaf"]:
            if verbose:
                print(f"[merge_missing] '{sid}' não é folha")
            return self
        if not seg["conditions"]:
            if verbose:
                print("[merge_missing] a raiz não tem nó de faltantes para juntar")
            return self
        if seg["conditions"][-1]["kind"] == "na":
            if verbose:
                print("[merge_missing] selecione o nó POPULADO de destino "
                      "(não o próprio nó de faltantes)")
            return self
        pai = seg["parent"]
        na_irmaos = [c for c, s in self.segments.items()
                     if s["parent"] == pai and s["is_leaf"] and s["conditions"]
                     and s["conditions"][-1]["kind"] == "na"]
        if not na_irmaos:
            if verbose:
                print("[merge_missing] este split não tem nó de faltantes "
                      "(ou ele já foi juntado)")
            return self
        return self._merge_missing_into(na_irmaos[0], sid, verbose=verbose)

    def _merge_missing_into(self, na_sid: str, target_sid: str, verbose: bool = True):
        """Funde o nó de faltantes `na_sid` no nó populado `target_sid` (mesmo pai)."""
        self._bump_version()      # estrutura vai mudar → invalida agregados memoizados
        if na_sid not in self.segments or target_sid not in self.segments:
            return self
        na_seg, tgt = self.segments[na_sid], self.segments[target_sid]
        pai = tgt["parent"]
        if na_seg["parent"] != pai:
            if verbose:
                print("[merge_missing] o nó de faltantes é de outro split")
            return self

        cond = dict(tgt["conditions"][-1])      # cópia da última condição do destino
        cond["include_na"] = True
        if cond["kind"] == "num":
            base = self._bin_label(cond["feature"],
                                   {"kind": "num", "lo": cond["lo"], "hi": cond["hi"]})
        else:
            base = self._bin_label(cond["feature"],
                                   {"kind": "cat", "cats": cond["cats"]})
        new_label = base + " + faltante"
        new_id = new_label if pai == "root" else f"{pai} | {new_label}"
        pai_seg = self.segments[pai]
        merged = {
            "mask": tgt["mask"] | na_seg["mask"], "label": new_label,
            "depth": tgt["depth"], "is_leaf": True,
            "path": pai_seg["path"] + [new_label], "parent": pai,
            "conditions": pai_seg["conditions"] + [cond],
        }
        self.segments.pop(na_sid, None)
        self.segments.pop(target_sid, None)
        self.segments[new_id] = merged

        filhos_pai = [c for c, s in self.segments.items() if s["parent"] == pai]
        if len(filhos_pai) == 1:
            self.segments.pop(new_id, None)
            self.segments[pai]["is_leaf"] = True
            if verbose:
                print(f"[merge_missing] faltantes juntados; pai '{pai}' voltou a ser folha")
            return self
        if verbose:
            print(f"[merge_missing] faltantes juntados em → '{new_id}'")
        return self

    # ------------------------------------------------------------------
    # AUTO_MERGE: funde automaticamente pares de folhas-IRMÃS (mesmo pai)
    #   adjacentes que NÃO se distinguem em alvo. Em cada rodada, escolhe o par
    #   de irmãs vizinhas mais parecido e o funde se:
    #     - o teste de hipótese não rejeita a igualdade de alvo (p > alpha), OU
    #     - a diferença de alvo médio entre elas é < min_valor_gap.
    #   Só funde irmãs (a única fusão válida na árvore). Por padrão o nó de
    #   faltantes NÃO entra; com `include_missing=True` ele também é juntado ao
    #   bin populado irmão estatisticamente mais próximo. `protect` = ids de
    #   folhas a preservar (ex.: travadas na UI). Itera até nenhum par qualificar.
    # ------------------------------------------------------------------
    def auto_merge(self, alpha: float = 0.05, min_valor_gap: float = 0.0,
                   test: str = "mannwhitney", min_n: int = 8,
                   protect: set | None = None, include_missing: bool = False,
                   max_rounds: int = 200, verbose: bool = True):
        protect = set(protect or [])
        n_merges = 0
        for _ in range(max_rounds):
            # agrupa folhas por pai
            folhas_por_pai: dict[str, list[str]] = {}
            for sid, s in self.segments.items():
                if s["is_leaf"]:
                    folhas_por_pai.setdefault(s["parent"], []).append(sid)

            melhor = None  # (prioridade, ação, *args)
            for pai, folhas in folhas_por_pai.items():
                if pai is None or len(folhas) < 2:
                    continue
                populadas = [c for c in folhas
                             if self.segments[c]["conditions"]
                             and self.segments[c]["conditions"][-1]["kind"] != "na"]
                # --- fusão por vizinhança entre irmãs ADJACENTES ---
                # Só pares realmente adjacentes (mesma run de folhas terminais);
                # uma irmã que se expandiu (nó não-terminal) quebra a adjacência,
                # então folhas de lados opostos dela nunca são fundidas entre si.
                for a, b in self._adjacent_sibling_pairs(pai):
                    if a in protect or b in protect:
                        continue
                    va, vb = self._leaf_target(a), self._leaf_target(b)
                    gap = (abs(va.mean() - vb.mean())
                           if len(va) and len(vb) else np.inf)
                    p = self._pair_pvalue(a, b, test=test, min_n=min_n)
                    if not ((not np.isnan(p) and p > alpha) or gap < min_valor_gap):
                        continue
                    prio = p if not np.isnan(p) else (2.0 - min(gap, 1.0))
                    if melhor is None or prio > melhor[0]:
                        melhor = (prio, "adj", b)
                # --- fusão do nó de faltantes no bin populado mais próximo ---
                if include_missing:
                    na_leaves = [c for c in folhas
                                 if self.segments[c]["conditions"]
                                 and self.segments[c]["conditions"][-1]["kind"] == "na"]
                    for na_sid in na_leaves:
                        if na_sid in protect:
                            continue
                        for tgt in populadas:
                            if tgt in protect:
                                continue
                            va, vb = self._leaf_target(na_sid), self._leaf_target(tgt)
                            gap = (abs(va.mean() - vb.mean())
                                   if len(va) and len(vb) else np.inf)
                            p = self._pair_pvalue(na_sid, tgt, test=test, min_n=min_n)
                            if not ((not np.isnan(p) and p > alpha) or gap < min_valor_gap):
                                continue
                            prio = p if not np.isnan(p) else (2.0 - min(gap, 1.0))
                            if melhor is None or prio > melhor[0]:
                                melhor = (prio, "miss", na_sid, tgt)
            if melhor is None:
                break
            antes = set(self.segments)
            if melhor[1] == "adj":
                self.merge_leaf(melhor[2], side="left", verbose=False)
            else:
                self._merge_missing_into(melhor[2], melhor[3], verbose=False)
            if set(self.segments) == antes:
                break  # salvaguarda: nenhuma mudança → evita laço infinito
            n_merges += 1

        n_folhas = sum(s["is_leaf"] for s in self.segments.values())
        if verbose:
            print(f"[auto_merge] {n_merges} fusão(ões) automática(s) "
                  f"(alpha={alpha}, min_valor_gap={min_valor_gap}, "
                  f"include_missing={include_missing}). Folhas finais: {n_folhas}")
        return self

    # ------------------------------------------------------------------
    # Descrição por extenso de um segmento a partir das suas condições
    # ------------------------------------------------------------------
    def _descrever(self, conditions: list) -> str:
        if not conditions:
            return "população total"
        partes = []
        for c in conditions:
            rotulo = self.feature_labels.get(c["feature"], c["feature"].replace("_", " "))
            na = c.get("include_na")
            if c["kind"] == "na":
                partes.append(f"{rotulo} faltante")
            elif c["kind"] == "num":
                base = f"{rotulo} {_intervalo_por_extenso(c['lo'], c['hi'])}"
                partes.append(f"({base} ou faltante)" if na else base)
            else:
                cats = " ou ".join(map(str, c["cats"]))
                base = f"{rotulo} em {{{cats}}}"
                partes.append(f"({base} ou faltante)" if na else base)
        return " e ".join(partes)

    # ------------------------------------------------------------------
    # Ordenação ESQUERDA→DIREITA das folhas (posição na árvore construída).
    #   A nota passa a ser essa posição: 1, 2, 3, … da esquerda p/ a direita.
    #   Em cada nó os filhos são ordenados pela MENOR alvo (DES) de folha do ramo
    #   (asc. = ramo de menor risco à esquerda), espelhando o layout de plot_tree.
    #   Para um split único, posição = ordem de alvo (idêntico ao comportamento
    #   anterior); em árvores profundas, as notas leem 1, 2, 3 de fato.
    # ------------------------------------------------------------------
    def _node_value(self, sid: str) -> float:
        """alvo médio do nó na amostra de referência (DES), com fallback p/ todas.

        Lê apenas a COLUNA-ALVO (não materializa o subframe inteiro) e reusa a
        máscara de referência pré-computada — chamado por folha em _leaf_order."""
        mask = self.segments[sid]["mask"]
        if self._mask_ref is not None:
            sr = self.df.loc[mask & self._mask_ref, self.target]
            if len(sr):
                return float(sr.mean())
        sub = self.df.loc[mask, self.target]
        return float(sub.mean()) if len(sub) else float("nan")

    def _leaf_order(self, ascending: bool = True) -> list:
        """sids das folhas na ordem esquerda→direita (memoizado por versão da árvore)."""
        return self._agg_memo(("leaf_order", ascending),
                              lambda: self._compute_leaf_order(ascending))

    def _compute_leaf_order(self, ascending: bool = True) -> list:
        filhos: dict = {}
        for sid, s in self.segments.items():
            filhos.setdefault(s["parent"], []).append(sid)
        INF = float("inf")
        leaf_pd = {sid: self._node_value(sid)
                   for sid, s in self.segments.items() if s["is_leaf"]}
        _submin: dict = {}

        def submin(sid):
            if sid not in _submin:
                if self.segments[sid]["is_leaf"]:
                    v = leaf_pd.get(sid, INF)
                    _submin[sid] = INF if (v is None or pd.isna(v)) else v
                else:
                    _submin[sid] = min((submin(c) for c in filhos.get(sid, [])),
                                       default=INF)
            return _submin[sid]

        sgn = 1 if ascending else -1
        order: list = []

        def dfs(sid):
            if self.segments[sid]["is_leaf"]:
                order.append(sid)
                return
            for c in sorted(filhos.get(sid, []), key=lambda c: (sgn * submin(c), str(c))):
                dfs(c)

        dfs("root")
        for sid, s in self.segments.items():
            if s["is_leaf"] and sid not in order:
                order.append(sid)
        return order

    # ------------------------------------------------------------------
    # LEAVES: segmentos-folha finais, com nota de alvo e descrição
    #   nota: 1..N pela POSIÇÃO esquerda→direita na árvore
    #   with_psi:  adiciona a contribuição de PSI de cada folha por amostra
    #              (psi_<amostra>), tendo a referência (DES) como base
    #   with_test: adiciona p_vs_prox = p-valor do teste de alvo entre a folha e a
    #              IRMÃ adjacente (mesmo pai — só irmãs são comparáveis/fundíveis).
    #              p alto ⇒ irmãs não distinguíveis (candidatas a fusão); folhas sem
    #              irmã à frente ficam com NaN. test: 'mannwhitney' (default)/'welch'
    # ------------------------------------------------------------------
    def leaves(self, ascending: bool = True, with_psi: bool = False,
               with_test: bool = False, test: str = "mannwhitney") -> pd.DataFrame:
        """Tabela de folhas (memoizada por versão da árvore + parâmetros). A UI
        chama isto ~5× por _refresh (direto e via _grade_map); a memoização evita
        revarrer as máscaras full-length por folha a cada chamada."""
        return self._agg_memo(
            ("leaves", ascending, with_psi, with_test, test),
            lambda: self._compute_leaves(ascending, with_psi, with_test, test))

    def _compute_leaves(self, ascending: bool = True, with_psi: bool = False,
                        with_test: bool = False, test: str = "mannwhitney") -> pd.DataFrame:
        linhas, n_total = [], len(self.df)
        for sid, seg in self.segments.items():
            if not seg["is_leaf"]:
                continue
            sub = self.df[seg["mask"]]
            # valor_medio na referência (DES) = base da régua; assim nota é
            # monotônica no alvo que entra em predict/apply_spark (com fallback)
            if self.sample_col is not None:
                ref_t = sub.loc[sub[self.sample_col] == self.ref_sample, self.target]
                pd_m = ref_t.mean() if len(ref_t) else sub[self.target].mean()
            else:
                pd_m = sub[self.target].mean()
            row = {
                "segmento": sid,
                "descricao": self._descrever(seg["conditions"]),
                "profundidade": seg["depth"],
                "n": len(sub),
                "repr_%": round(100 * len(sub) / n_total, 1) if n_total else 0.0,
                "valor_medio": round(pd_m, 4) if not pd.isna(pd_m) else np.nan,
                "valor_std": round(sub[self.target].std(), 4),
            }
            if self.sample_col is not None:
                for amostra in self.df[self.sample_col].dropna().unique():
                    s_am = sub.loc[sub[self.sample_col] == amostra, self.target]
                    row[f"valor_{amostra}"] = round(s_am.mean(), 4) if len(s_am) else np.nan
            linhas.append(row)
        # nota = POSIÇÃO esquerda→direita na árvore (ver _leaf_order); assim os
        # números sempre leem 1, 2, 3 da esquerda p/ a direita no plot_tree.
        ordem = {sid: i for i, sid in enumerate(self._leaf_order(ascending=ascending))}
        out = (
            pd.DataFrame(linhas)
            .sort_values("segmento", key=lambda s: s.map(ordem), kind="stable")
            .reset_index(drop=True)
        )
        out.insert(1, "nota", range(1, len(out) + 1))

        if with_psi and self.sample_col is not None:
            out = self._append_psi_cols(out)
        if with_test:
            out = self._append_adjacency_test(out, test=test)
        return out

    # contribuição de PSI por folha para cada amostra (≠ referência)
    def _append_psi_cols(self, out, eps: float = 1e-6):
        # Vetorizado: um único crosstab(folha, amostra) substitui os laços
        # (mask_folha & mask_amostra).sum() full-length por folha×amostra. O
        # denominador continua o tamanho TOTAL da amostra (máscaras pré-computadas).
        leaf_ids = out["segmento"].tolist()
        leaf_id = self._leaf_id_series()
        ct = pd.crosstab(leaf_id, self.df[self.sample_col])

        def fracs(amostra):
            n = float(self._sample_masks[amostra].sum())
            col = ct[amostra] if amostra in ct.columns else None
            return {sid: (float(col.get(sid, 0.0)) / n if (col is not None and n) else 0.0)
                    for sid in leaf_ids}

        ref_pct = fracs(self.ref_sample)
        # representatividade (% da folha DENTRO da amostra) — começa pela referência;
        # as demais amostras saem no laço abaixo, ao lado do respectivo PSI.
        out[f"repr_{self.ref_sample}_%"] = [round(100 * ref_pct[sid], 1)
                                            for sid in leaf_ids]
        for amostra in self.df[self.sample_col].dropna().unique():
            if amostra == self.ref_sample:
                continue
            cur_pct = fracs(amostra)
            psi_col, repr_col = [], []
            for sid in leaf_ids:
                frac = cur_pct[sid]
                p_ref = max(ref_pct[sid], eps)
                p_cur = max(frac, eps)
                psi_col.append(round((p_cur - p_ref) * np.log(p_cur / p_ref), 4))
                repr_col.append(round(100 * frac, 1))
            out[f"psi_{amostra}"] = psi_col
            out[f"repr_{amostra}_%"] = repr_col
        return out

    # alvo de uma folha, restrito à amostra de referência quando houver
    # (sem NaN, para os testes/fusões não ficarem cegos por valores ausentes)
    def _leaf_target(self, sid: str) -> np.ndarray:
        # Cache por id() da máscara da folha: prune/auto_merge e o teste de
        # adjacência re-extraem o alvo da MESMA folha dezenas de vezes por rodada.
        # A máscara da referência (DES) é pré-computada (self._mask_ref) em vez de
        # recriar `df[sample_col]==ref` full-length a cada chamada.
        mask = self.segments[sid]["mask"]
        ck = id(mask)
        hit = self._leaf_target_cache.get(ck)
        if hit is not None and hit[1] is mask:    # confirma que o id não foi reciclado
            return hit[0]
        m = mask if self._mask_ref is None else (mask & self._mask_ref)
        vals = self.df.loc[m, self.target].to_numpy(dtype="float64")
        vals = vals[~np.isnan(vals)]
        if len(self._leaf_target_cache) > 4000:   # backstop de memória
            self._leaf_target_cache.clear()
        self._leaf_target_cache[ck] = (vals, mask)   # guarda mask p/ o id não reciclar
        return vals

    # p-valor do teste de igualdade de alvo entre duas folhas (na referência)
    def _pair_pvalue(self, sid_a: str, sid_b: str, test: str = "mannwhitney",
                     min_n: int = 8) -> float:
        if mannwhitneyu is None:        # scipy ausente (importado no topo do módulo)
            return np.nan
        a, b = self._leaf_target(sid_a), self._leaf_target(sid_b)
        if len(a) < min_n or len(b) < min_n:
            return np.nan
        try:
            if test == "welch":
                return float(ttest_ind(a, b, equal_var=False).pvalue)
            return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except Exception:
            return np.nan

    # p-valor do teste de alvo entre uma folha e a IRMÃ adjacente (mesmo pai).
    #   Só irmãs são diretamente comparáveis (e fundíveis), por isso o teste é
    #   restrito a elas — folhas de pais diferentes não se comparam. A ordem das
    #   irmãs é a mesma da fusão (por corte, no numérico; por alvo, no categórico).
    #   A última irmã do grupo e o nó de faltantes ficam com NaN (sem par à frente).
    def _append_adjacency_test(self, out, test="mannwhitney", min_n=8):
        leaf_ids = out["segmento"].tolist()
        # 'próxima irmã' = a vizinha à frente numa mesma run de folhas terminais
        # adjacentes; uma irmã que se expandiu (nó não-terminal) QUEBRA a
        # adjacência, então folhas em lados opostos dela não são comparadas.
        prox: dict = {}
        for pai in {self.segments[sid]["parent"] for sid in leaf_ids}:
            if pai is None:
                continue
            for a, b in self._adjacent_sibling_pairs(pai):
                prox[a] = b
        pvals = []
        for sid in leaf_ids:
            nxt = prox.get(sid)
            if nxt is None:
                pvals.append(np.nan)
                continue
            p = self._pair_pvalue(sid, nxt, test=test, min_n=min_n)
            pvals.append(round(p, 4) if not np.isnan(p) else np.nan)
        out["p_vs_prox"] = pvals
        return out

    # mapeia segmento -> (nota, descrição) para uso no assign
    def _grade_map(self, ascending: bool = True):
        # memoizado por versão: a UI chama _grade_map 4× por _refresh (tree_html,
        # leaf_header, leaf_chips, ordered_leaf_options)
        d = self._agg_memo(("grade_map", ascending),
                           lambda: self._compute_grade_map(ascending))
        return d["nota"], d["desc"]

    def _compute_grade_map(self, ascending: bool = True):
        lv = self.leaves(ascending=ascending)
        nota = dict(zip(lv["segmento"], lv["nota"]))
        desc = dict(zip(lv["segmento"], lv["descricao"]))
        return {"nota": nota, "desc": desc}

    def _leaf_id_series(self, leaf_ids=None) -> pd.Series:
        """Série (index = df.index) com o id da folha de cada linha (NA se nenhuma).
        Como as folhas são mutuamente exclusivas, isso permite trocar os laços
        ``(mask_folha & mask_amostra).sum()`` por um único ``pd.crosstab``.
        Memoizada por versão da árvore."""
        def build():
            ids = leaf_ids if leaf_ids is not None else \
                [sid for sid, s in self.segments.items() if s["is_leaf"]]
            # array numpy gravável e DEPOIS embrulhado em Series: sob pandas ≥3.0
            # (copy-on-write) ``Series.values`` é read-only e a atribuição in-place
            # ``s.values[mask] = sid`` levantaria ValueError.
            arr = np.full(len(self.df), pd.NA, dtype=object)
            for sid in ids:
                arr[self.segments[sid]["mask"].values] = sid
            return pd.Series(arr, index=self.df.index, dtype=object)
        # a identidade do conjunto entra na chave: uma chamada com subconjunto de
        # folhas não pode devolver a série do conjunto completo do cache.
        cache_key = ("leaf_id_series", tuple(leaf_ids) if leaf_ids is not None else None)
        return self._agg_memo(cache_key, build)

    # ------------------------------------------------------------------
    # TREE: desenha a árvore hierárquica (nós internos + folhas) em texto.
    #   Cada nó mostra n, representatividade e alvo médio; folhas trazem [nota N].
    #   Filhos são ordenados por alvo (ascendente por padrão).
    # ------------------------------------------------------------------
    def tree(self, ascending: bool = True) -> str:
        # mapa pai -> filhos
        filhos: dict = {}
        for sid, seg in self.segments.items():
            filhos.setdefault(seg["parent"], []).append(sid)

        nota_map, _ = self._grade_map(ascending=ascending)
        n_total = len(self.df)
        linhas: list[str] = []

        def stats(sid):
            sub = self.df[self.segments[sid]["mask"]]
            pdv = sub[self.target].mean() if len(sub) else float("nan")
            return len(sub), 100 * len(sub) / n_total, pdv

        def rotulo(sid):
            seg = self.segments[sid]
            if seg["parent"] is None:
                return "TODA A CARTEIRA"
            return self._descrever([seg["conditions"][-1]])

        # ordena os filhos pela MENOR nota do ramo (esquerda→direita), igual ao
        # plot_tree — assim o texto lê as folhas 1, 2, 3 na mesma ordem
        _min_nota: dict = {}

        def min_nota(sid):
            if sid not in _min_nota:
                if self.segments[sid]["is_leaf"]:
                    _min_nota[sid] = nota_map.get(sid, 10 ** 9)
                else:
                    _min_nota[sid] = min((min_nota(c) for c in filhos.get(sid, [])),
                                         default=10 ** 9)
            return _min_nota[sid]

        def rec(sid, prefix, is_last, is_root=False):
            n, rep, pdv = stats(sid)
            seg = self.segments[sid]
            conn = "" if is_root else ("└─ " if is_last else "├─ ")
            tag = f"  [nota {nota_map.get(sid, '?')}]" if seg["is_leaf"] else ""
            linhas.append(f"{prefix}{conn}{rotulo(sid)}  "
                          f"(n={n}, {rep:.1f}%, {self._risk_word}={pdv:.4f}){tag}")
            ch = sorted(filhos.get(sid, []), key=min_nota)
            child_prefix = "" if is_root else prefix + ("   " if is_last else "│  ")
            for i, c in enumerate(ch):
                rec(c, child_prefix, i == len(ch) - 1)

        rec("root", "", True, is_root=True)
        n_folhas = sum(s["is_leaf"] for s in self.segments.values())
        linhas.append(f"\n({n_folhas} folhas | profundidade máxima "
                      f"{max(s['depth'] for s in self.segments.values())})")
        out = "\n".join(linhas)
        print(out)
        return out

    # ------------------------------------------------------------------
    # _value_color_range: faixa (vmin, vmax) da escala de cor do alvo, a partir das
    #   valores observados nos nós. Diferente do alvo (fixa 0–1), o alvo costuma ser
    #   pequena (ex.: 0–0.3), então a escala é dinâmica para as cores
    #   discriminarem — ancorada em 0 (sem default = verde).
    # ------------------------------------------------------------------
    def _value_color_range(self):
        """Faixa de cor do alvo na árvore. Regressão (alvo em [0,1]) usa escala
        fixa 0–1; classificação (alvo, taxas pequenas) usa escala dinâmica até a
        maior taxa observada para dar contraste."""
        if not self._is_clf:
            return 0.0, 1.0
        # só FOLHAS (a média de um nó interno é média ponderada das folhas, logo
        # ≤ a maior média de folha → o vmax é o mesmo) e via um único groupby
        # vetorizado, em vez de materializar o subframe de cada segmento.
        leaf_ids = [sid for sid, s in self.segments.items() if s["is_leaf"]]
        leaf_id = self._leaf_id_series(leaf_ids)
        # vmax na REFERÊNCIA (DES) quando há sample_col: as cores dos nós e a colorbar
        # usam a média na DES (colorbar rotulada "(DES)"); usar a média all-samples faria
        # uma folha com alvo DES > vmax saturar e a escala ficar incoerente com o rótulo.
        if self.sample_col is not None:
            ref = self._mask_ref
            means = self.df.loc[ref, self.target].groupby(leaf_id[ref]).mean()
        else:
            means = self.df[self.target].groupby(leaf_id).mean()
        vmax = float(means.max()) if len(means) and not means.isna().all() else 0.01
        return 0.0, (vmax if vmax > 1e-9 else 0.01)

    # ------------------------------------------------------------------
    # PLOT_TREE: desenha a árvore como IMAGEM (matplotlib). Cada nó mostra o
    #   rótulo da condição, n, % e alvo médio; folhas trazem a nota. A cor do nó
    #   reflete o alvo (verde = baixa → vermelho = alta). Devolve a Figure e,
    #   se `save_path` for dado, salva a imagem (PNG/SVG/PDF…).
    # ------------------------------------------------------------------
    def plot_tree(self, ascending: bool = True, figsize=None, cmap: str = "RdYlGn_r",
                  show_samples: bool = False, title: str | None = "auto",
                  save_path: str | None = None, dpi: int = 150, ax=None,
                  highlight: str | None = None):
        # "auto" → título conforme o rótulo do alvo (self._risk_word)
        if title == "auto":
            title = f"Segmentação de {self._risk_word}"
        try:
            import textwrap

            import matplotlib.pyplot as plt
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            from matplotlib.patches import FancyBboxPatch
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_tree requer matplotlib — use: pip install matplotlib") from e

        filhos: dict = {}
        for sid, s in self.segments.items():
            filhos.setdefault(s["parent"], []).append(sid)

        n_total = len(self.df)
        nota_map, _ = self._grade_map(ascending=ascending)
        ref = self.ref_sample if self.sample_col is not None else None

        def stats(sid):
            m = self.segments[sid]["mask"]
            n = int(m.sum())
            sub = self.df[m]
            if ref is not None:
                sr = sub.loc[sub[self.sample_col] == ref, self.target]
                pdv = sr.mean() if len(sr) else (sub[self.target].mean() if n else float("nan"))
            else:
                pdv = sub[self.target].mean() if n else float("nan")
            return n, (100 * n / n_total if n_total else 0.0), pdv

        def sample_value(sid, a):
            m = self.segments[sid]["mask"] & (self.df[self.sample_col] == a)
            sub = self.df[m]
            return sub[self.target].mean() if len(sub) else float("nan")

        # menor nota entre as folhas de cada ramo — usada para ordenar os
        # ramos da esquerda (menor nota) para a direita (maior nota)
        _min_nota: dict = {}

        def min_nota(sid):
            if sid not in _min_nota:
                if self.segments[sid]["is_leaf"]:
                    _min_nota[sid] = nota_map.get(sid, 10 ** 9)
                else:
                    _min_nota[sid] = min((min_nota(c) for c in filhos.get(sid, [])),
                                         default=10 ** 9)
            return _min_nota[sid]

        # --- layout: folhas em x sequencial na ORDEM DE nota (esq.→dir.);
        #     nós internos centralizados sobre o intervalo dos filhos ---
        X_GAP, Y_GAP = 2.4, 2.15        # espaçamento entre folhas / entre níveis
        bw, bh = 0.97, 0.74             # meia-largura / meia-altura do box (dados)
        pos: dict = {}
        counter = [0]
        max_depth = [0]

        def place(sid, depth):
            max_depth[0] = max(max_depth[0], depth)
            ch = sorted(filhos.get(sid, []), key=min_nota)   # ramos por menor nota
            if self.segments[sid]["is_leaf"] or not ch:
                x = counter[0] * X_GAP
                counter[0] += 1
            else:
                cxs = [place(c, depth + 1) for c in ch]
                x = 0.5 * (min(cxs) + max(cxs))              # centra sobre os filhos
            pos[sid] = (x, -depth * Y_GAP)
            return x

        place("root", 0)
        md = max_depth[0]
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]

        if figsize is None:
            figsize = (max(7.0, (max(xs) - min(xs) + 2 * bw + 1.0) * 0.95),
                       max(3.5, (md + 1) * Y_GAP * 0.95))
        fig, ax = self._new_ax(figsize, dpi, ax)

        vmin, vmax = self._value_color_range()      # escala de cor do alvo (dinâmica)
        norm = Normalize(vmin, vmax)
        cmap_obj = plt.get_cmap(cmap)

        def rotulo(sid):
            s = self.segments[sid]
            if s["parent"] is None:
                return "TODA A CARTEIRA"
            return self._descrever([s["conditions"][-1]])

        # arestas (atrás dos nós)
        for sid, (x, y) in pos.items():
            for c in filhos.get(sid, []):
                cx, cy = pos[c]
                ax.plot([x, cx], [y - bh, cy + bh], color="#9aa7b2", lw=1.1, zorder=1)

        base_fs = 8.6
        fit_items = []            # (texto, meia-largura, meia-altura) p/ ajuste de fonte
        for sid, (x, y) in pos.items():
            _, rep, pdv = stats(sid)
            color = cmap_obj(norm(pdv)) if not pd.isna(pdv) else (0.88, 0.88, 0.88, 1.0)
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            txt_color = "#15324a" if lum > 0.6 else "#ffffff"
            is_leaf = self.segments[sid]["is_leaf"]
            selecionada = (highlight is not None and sid == highlight)
            # folha selecionada: contorno destacado (âmbar grosso) + leve glow
            if selecionada:
                ax.add_patch(FancyBboxPatch(
                    (x - bw - 0.06, y - bh - 0.06), 2 * bw + 0.12, 2 * bh + 0.12,
                    boxstyle="round,pad=0.02,rounding_size=0.16",
                    linewidth=0, facecolor="#f5a623", alpha=0.30, zorder=1.5))
            # gid=sid identifica a caixa do nó no canvas — é o que permite ao
            # plot_tree_hitmap devolver a região clicável de cada segmento.
            ax.add_patch(FancyBboxPatch(
                (x - bw, y - bh), 2 * bw, 2 * bh,
                boxstyle="round,pad=0.02,rounding_size=0.14",
                linewidth=(3.2 if selecionada else 1.3),
                edgecolor=("#e8870b" if selecionada else "#33424f"),
                facecolor=color, zorder=2, gid=sid))
            # 1) QUEBRA (condição do nó) em NEGRITO, no topo da caixa
            cab = "\n".join(textwrap.wrap(rotulo(sid), 18)[:3])
            t_split = ax.text(x, y + 0.34 * bh, cab, ha="center", va="center",
                              fontsize=base_fs, color=txt_color, zorder=3,
                              fontweight="bold", linespacing=1.12, clip_on=True)
            fit_items.append((t_split, bw * 0.94, bh * 0.58))
            # 2) representatividade e alvo (em %, 2 casas) na MESMA linha, separados por barra
            pd_txt = (f"{self._risk_word} {pdv * 100:.2f}%" if not pd.isna(pdv)
                      else f"{self._risk_word} —")
            metr = f"repr. {rep:.1f}%  |  {pd_txt}"
            if show_samples and self.sample_col is not None:
                amostras = list(self.df[self.sample_col].dropna().unique())
                metr += "\n" + " | ".join(f"{a} {sample_value(sid, a) * 100:.2f}%"
                                          for a in amostras)
            t_metr = ax.text(x, y - 0.42 * bh, metr, ha="center", va="center",
                             fontsize=base_fs - 0.8, color=txt_color, zorder=3,
                             linespacing=1.12, clip_on=True)
            fit_items.append((t_metr, bw * 0.94, bh * 0.34))
            # 3) número da folha no CANTO INFERIOR DIREITO
            if is_leaf:
                ax.text(x + bw * 0.93, y - bh * 0.9, f"folha {nota_map.get(sid, '?')}",
                        ha="right", va="bottom", fontsize=base_fs - 1.8, color=txt_color,
                        zorder=3, fontweight="bold", clip_on=True)

        ax.set_xlim(min(xs) - bw - 0.4, max(xs) + bw + 0.4)
        ax.set_ylim(min(ys) - bh - 0.4, max(ys) + bh + 0.6)
        ax.axis("off")
        if title:
            n_folhas = sum(s["is_leaf"] for s in self.segments.values())
            ax.set_title(f"{title}  ·  {n_folhas} folhas",
                         fontsize=12.5, fontweight="bold", color="#15324a")
        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
        from matplotlib.ticker import PercentFormatter
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
        cbar.set_label(f"{self._risk_mean}{' (DES)' if ref else ''} "
                       f"(escala 0–{vmax * 100:.1f}%)", fontsize=9)
        fig.tight_layout()

        # força cada texto a caber na sua sub-região, encolhendo a fonte se preciso
        self._fit_texts_to_boxes(fig, ax, fit_items)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    @staticmethod
    def _fit_texts_to_boxes(fig, ax, items, pad: float = 0.92, min_fs: float = 4.0):
        """Encolhe a fonte de cada texto até caber na sua sub-região.

        ``items``: lista de ``(texto, meia_largura, meia_altura)`` em coordenadas de
        dados, centradas na posição do texto.
        """
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        except Exception:        # pragma: no cover - backend sem renderer
            return
        for t, hw, hh in items:
            x, y = t.get_position()
            px0, py0 = ax.transData.transform((x - hw, y - hh))
            px1, py1 = ax.transData.transform((x + hw, y + hh))
            box_w, box_h = abs(px1 - px0) * pad, abs(py1 - py0) * pad
            ext = t.get_window_extent(renderer)
            if ext.width <= box_w and ext.height <= box_h:
                continue
            scale = min(box_w / max(ext.width, 1e-6), box_h / max(ext.height, 1e-6))
            t.set_fontsize(max(min_fs, t.get_fontsize() * scale))

    # ------------------------------------------------------------------
    # PLOT_TREE_HITMAP: PNG da árvore + "mapa de cliques" (caixa de cada nó
    #   em pixels) — base do preview INTERATIVO da TreeSegmenterUI.
    # ------------------------------------------------------------------
    def plot_tree_hitmap(self, dpi: int = 110, ascending: bool = True,
                         cmap: str = "RdYlGn_r", show_samples: bool = False,
                         title: str | None = "auto") -> dict:
        """Renderiza a árvore em PNG e devolve as regiões clicáveis por nó.

        Devolve ``{"png": bytes, "width": int, "height": int, "nodes": dict}``,
        onde ``nodes`` mapeia ``sid → {x0, y0, x1, y1, is_leaf}`` em PIXELS do
        PNG com origem no canto SUPERIOR esquerdo (convenção HTML/CSS — o eixo
        y do matplotlib é invertido aqui). O PNG é salvo SEM ``bbox_inches=
        'tight'``: o recorte mudaria a geometria e desalinharia as caixas.
        """
        import io as _io
        fig = self.plot_tree(ascending=ascending, cmap=cmap,
                             show_samples=show_samples, title=title, dpi=dpi)
        canvas = fig.canvas
        canvas.draw()
        renderer = canvas.get_renderer()
        width, height = canvas.get_width_height()
        ax = fig.axes[0]                      # eixo principal (axes[1] é a colorbar)
        nodes: dict = {}
        for p in ax.patches:
            sid = p.get_gid()
            if not sid or sid not in self.segments:
                continue                      # ignora patches sem gid (ex.: glow do realce)
            ext = p.get_window_extent(renderer)
            nodes[sid] = {
                "x0": round(float(ext.x0), 2), "y0": round(float(height - ext.y1), 2),
                "x1": round(float(ext.x1), 2), "y1": round(float(height - ext.y0), 2),
                "is_leaf": bool(self.segments[sid]["is_leaf"]),
            }
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=fig.dpi)
        return {"png": buf.getvalue(), "width": int(width), "height": int(height),
                "nodes": nodes}

    # ------------------------------------------------------------------
    # PSI: estabilidade populacional, segmentos-folha como bins
    # ------------------------------------------------------------------
    def psi(self, eps: float = 1e-6) -> pd.DataFrame:
        if self.sample_col is None:
            raise ValueError("PSI requer sample_col definido no construtor.")
        return self._agg_memo(("psi", round(float(eps), 12)),
                              lambda: self._compute_psi(eps))

    def _compute_psi(self, eps: float = 1e-6) -> pd.DataFrame:
        leaf_ids = [sid for sid, s in self.segments.items() if s["is_leaf"]]
        if not leaf_ids:
            raise ValueError("Nenhuma folha — cresça a segmentação antes do PSI.")

        # distribuição (% por folha) por amostra via UM crosstab vetorizado, em vez
        # de (mask_folha & mask_amostra).sum() full-length por folha×amostra.
        # Denominador = tamanho TOTAL da amostra (igual ao código original), via
        # máscaras de amostra pré-computadas.
        leaf_id_por_linha = self._leaf_id_series(leaf_ids)
        ct = pd.crosstab(leaf_id_por_linha, self.df[self.sample_col])
        dist = {}
        for amostra in self.df[self.sample_col].dropna().unique():
            n_am = float(self._sample_masks[amostra].sum())
            col = ct[amostra] if amostra in ct.columns else None
            dist[amostra] = {
                sid: (float(col.get(sid, 0.0)) / n_am if (col is not None and n_am) else 0.0)
                for sid in leaf_ids
            }

        ref = dist[self.ref_sample]
        self._psi_detalhe = []
        linhas = []
        for amostra, pct in dist.items():
            if amostra == self.ref_sample:
                continue
            psi_total = 0.0
            for sid in leaf_ids:
                p_ref = max(ref[sid], eps)
                p_cur = max(pct[sid], eps)
                contrib = (p_cur - p_ref) * np.log(p_cur / p_ref)
                psi_total += contrib
                self._psi_detalhe.append({
                    "amostra": amostra, "segmento": sid,
                    f"%_{self.ref_sample}": round(100 * ref[sid], 2),
                    "%_atual": round(100 * pct[sid], 2),
                    "psi_bin": round(contrib, 4),
                })
            linhas.append({
                "amostra": amostra,
                "psi": round(psi_total, 4),
                "classificacao": _classifica_psi(psi_total),
            })
        return pd.DataFrame(linhas).sort_values("psi", ascending=False).reset_index(drop=True)

    def psi_detalhe(self) -> pd.DataFrame:
        """Contribuição de cada segmento para o PSI (chame após .psi())."""
        return pd.DataFrame(self._psi_detalhe)

    # ------------------------------------------------------------------
    # CSI: Characteristic Stability Index POR VARIÁVEL.
    #   Enquanto o `psi()` mede a estabilidade da SEGMENTAÇÃO (folhas como
    #   bins), o CSI mede a estabilidade da DISTRIBUIÇÃO DE CADA VARIÁVEL DE
    #   ENTRADA entre a referência (DES) e cada outra amostra (OOT, ...). Os
    #   bins são fixados na referência (mesmo binning ótimo/manual do split) e
    #   a fórmula é a do PSI aplicada à variável. Útil para detectar QUAL
    #   característica está migrando antes mesmo de entrar na árvore.
    #   `csi_detalhe()` traz a contribuição de cada faixa (decomposição).
    # ------------------------------------------------------------------
    def csi(self, features: list | None = None, max_n_bins: int = 10,
            min_bin_size: float = 0.05, eps: float = 1e-6) -> pd.DataFrame:
        if self.sample_col is None:
            raise ValueError("CSI requer sample_col definido (precisa de amostras "
                             "para comparar a referência com as demais).")
        if features is None:
            features = [c for c in self.df.columns if c not in self._nonfeature_cols()]

        nonref = [a for a in self.df[self.sample_col].dropna().unique()
                  if a != self.ref_sample]
        ref_mask = self._mask_ref
        n_ref = int(ref_mask.sum())
        self._csi_detalhe = []

        rows = []
        for feat in features:
            try:
                bins, _modo, kind = self._resolve_bins(
                    self.df, feat, None, None, max_n_bins, min_bin_size)
            except Exception:
                bins, kind = [], "—"
            row = {"variavel": feat, "tipo": kind, "n_bins": len(bins)}
            if not bins or n_ref == 0:
                for a in nonref:
                    row[f"csi_{a}"] = np.nan
                row["pior_csi"], row["classificacao"] = np.nan, "—"
                rows.append(row)
                continue

            # máscara de cada bin sobre o df calculada UMA vez (inclui o .astype(str)
            # caro das categóricas) e reusada para a referência e todas as amostras —
            # antes era recomputada 1+n_amostras vezes por bin.
            bin_masks = [self._mask_in(self.df, feat, b) for b in bins]
            ref_pct = [(int((bm & ref_mask).sum()) / n_ref) for bm in bin_masks]
            pior = 0.0
            for a in nonref:
                a_mask = self._sample_masks[a]
                n_a = int(a_mask.sum())
                csi_tot = 0.0
                for b, p_ref, bm in zip(bins, ref_pct, bin_masks):
                    p_cur = (int((bm & a_mask).sum()) / n_a
                             if n_a else 0.0)
                    pr, pc = max(p_ref, eps), max(p_cur, eps)
                    contrib = (pc - pr) * np.log(pc / pr)
                    csi_tot += contrib
                    self._csi_detalhe.append({
                        "variavel": feat, "amostra": a,
                        "faixa": self._bin_label(feat, b),
                        f"%_{self.ref_sample}": round(100 * p_ref, 2),
                        "%_atual": round(100 * p_cur, 2),
                        "csi_bin": round(contrib, 4),
                    })
                row[f"csi_{a}"] = round(csi_tot, 4)
                pior = max(pior, csi_tot)
            row["pior_csi"] = round(pior, 4)
            row["classificacao"] = _classifica_psi(pior)
            rows.append(row)

        return (pd.DataFrame(rows)
                .sort_values("pior_csi", ascending=False, na_position="last")
                .reset_index(drop=True))

    def csi_detalhe(self) -> pd.DataFrame:
        """Contribuição de cada faixa para o CSI por variável (chame após .csi())."""
        return pd.DataFrame(self._csi_detalhe)

    # ------------------------------------------------------------------
    # METRICS: avalia a régua como um modelo de alvo (classificação).
    #   Score de cada linha = alvo médio do seu segmento na amostra de referência
    #   (DES). Retorna, por amostra (DES, OOT, ...): taxa de default observada,
    #   **KS**, **AUC** (ROC), **Gini**, **Acurácia** e **F1**. Acurácia/F1 usam
    #   o corte KS-ótimo por amostra (ou `cutoff`, se informado).
    # ------------------------------------------------------------------
    def metrics(self, cutoff: float | None = None) -> pd.DataFrame:
        """Métricas da régua como modelo, por amostra (memoizada por versão).

        classification → taxa de default + KS/AUC/Gini/Acurácia/F1.
        regression → MAE/RMSE/R² (``cutoff`` é ignorado)."""
        ck = cutoff if cutoff is None else round(float(cutoff), 12)
        return self._agg_memo(("metrics", ck), lambda: self._compute_metrics(cutoff))

    def _compute_metrics(self, cutoff: float | None = None) -> pd.DataFrame:
        if self._is_clf:
            from yggdrasil.metrics.classification import classification_metrics

        leaf_ids = [sid for sid, s in self.segments.items() if s["is_leaf"]]
        if self.sample_col is not None:
            ref_mask = self._mask_ref
        else:
            ref_mask = pd.Series(True, index=self.df.index)
        overall = self.df.loc[ref_mask, self.target].mean()

        # score por linha = alvo médio do segmento na referência (régua). Em vez de
        # `pred[m.values] = v` full-length por folha, agrega o alvo da referência
        # por folha num ÚNICO groupby e mapeia de volta via leaf_id por linha.
        leaf_id = self._leaf_id_series(leaf_ids)
        ref_pd = self.df.loc[ref_mask, self.target].groupby(leaf_id[ref_mask]).mean()
        # fallback IGUAL ao da régua real (_predicted_series/_compute_leaves): folha sem
        # linhas na referência ⇒ média da PRÓPRIA folha em TODAS as amostras (não o alvo
        # global), para metrics avaliar o mesmo score que a régua produz em produção;
        # só se a folha for totalmente vazia é que cai no alvo global (overall).
        all_pd = self.df[self.target].groupby(leaf_id).mean()
        pd_map = {}
        for sid in leaf_ids:
            v = ref_pd.get(sid, np.nan)
            if pd.isna(v):
                v = all_pd.get(sid, np.nan)
            pd_map[sid] = float(v) if pd.notna(v) else float(overall)
        pred = leaf_id.map(pd_map).astype("float64")

        # ordem das amostras: referência primeiro
        if self.sample_col is not None:
            todas = list(self.df[self.sample_col].dropna().unique())
            ordem = ([self.ref_sample] + [a for a in todas if a != self.ref_sample])
            grupos = [(a, self._sample_masks[a]) for a in ordem]
        else:
            grupos = [("todos", pd.Series(True, index=self.df.index))]

        linhas = []
        for nome, mask in grupos:
            y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
            yhat = pred[mask.values].to_numpy(dtype="float64")
            valid = ~(np.isnan(y) | np.isnan(yhat))      # ignora alvo/score NaN
            y, yhat = y[valid], yhat[valid]
            base = {"amostra": nome, "n": int(mask.sum())}
            if self._is_clf:
                if len(y) == 0:
                    base.update({"taxa_default": np.nan, "KS": np.nan, "AUC": np.nan,
                                 "Gini": np.nan, "Acuracia": np.nan, "F1": np.nan})
                else:
                    cm = classification_metrics(y, yhat, cutoff=cutoff)
                    base.update({
                        "taxa_default": round(float(np.mean(y)), 4),
                        "KS": cm["ks"], "AUC": cm["auc"], "Gini": cm["gini"],
                        "Acuracia": cm["accuracy"], "F1": cm["f1"],
                    })
            else:
                if len(y) == 0:
                    base.update({"MAE": np.nan, "RMSE": np.nan, "R2": np.nan})
                else:
                    err = y - yhat
                    ss_res = float(np.sum(err ** 2))
                    ss_tot = float(np.sum((y - y.mean()) ** 2))
                    base.update({
                        "MAE": round(float(np.mean(np.abs(err))), 4),
                        "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 4),
                        "R2": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else np.nan,
                    })
            linhas.append(base)
        return pd.DataFrame(linhas)

    # ------------------------------------------------------------------
    # BOOTSTRAP_CI: intervalo de confiança do alvo (taxa de default) por folha,
    #   via reamostragem bootstrap na amostra `sample` (default = referência/DES).
    #   Se houver `check_sample` (default = 1ª não-referência, ex. OOT), traz a
    #   alvo dela por folha e verifica a ADERÊNCIA: se o alvo de OOT cai dentro do
    #   IC bootstrap do DES (estável) ou fora (acima/abaixo = alerta).
    # ------------------------------------------------------------------
    def bootstrap_ci(self, n_boot: int = 1000, ci: float = 0.95,
                     sample: str | None = None, check_sample: str | None = None,
                     seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        alpha = (1 - ci) / 2
        if self.sample_col is not None:
            if sample is None:
                sample = self.ref_sample
            if check_sample is None:
                nonref = [a for a in self.df[self.sample_col].dropna().unique()
                          if a != self.ref_sample]
                check_sample = nonref[0] if nonref else None
        else:
            sample = check_sample = None

        sample_mask = (self._sample_masks.get(sample)
                       if (self.sample_col is not None and sample is not None) else None)
        lv = self.leaves()  # ordem de nota
        rows = []
        for _, r in lv.iterrows():
            sid = r["segmento"]
            m = self.segments[sid]["mask"]
            m_s = (m & sample_mask) if sample_mask is not None else m
            vals = self.df.loc[m_s, self.target].to_numpy(dtype="float64")
            vals = vals[~np.isnan(vals)]
            n = len(vals)
            if n >= 2:
                # bootstrap em BLOCOS de n_boot: limita a matriz de reamostragem a
                # ~4M elementos (em vez de n_boot×n inteiro de uma vez — uma folha
                # com n=100k estouraria a memória do driver no Databricks).
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
                "nota": r["nota"],
                "descricao": r["descricao"],
                "n": int(n),
                f"valor_{sample or 'todos'}": round(pt, 4) if not np.isnan(pt) else np.nan,
                "ic_low": round(float(lo), 4) if not np.isnan(lo) else np.nan,
                "ic_high": round(float(hi), 4) if not np.isnan(hi) else np.nan,
                "amplitude": round(float(hi - lo), 4) if not np.isnan(hi) else np.nan,
            }
            if check_sample is not None:
                m_c = m & self._sample_masks[check_sample]
                cvals = self.df.loc[m_c, self.target].to_numpy(dtype="float64")
                cvals = cvals[~np.isnan(cvals)]
                pd_c = float(cvals.mean()) if len(cvals) else np.nan
                row[f"valor_{check_sample}"] = round(pd_c, 4) if not np.isnan(pd_c) else np.nan
                if np.isnan(pd_c) or np.isnan(lo):
                    row["aderente"] = None
                    row["status_oot"] = "—"
                else:
                    dentro = bool(lo <= pd_c <= hi)
                    row["aderente"] = dentro
                    row["status_oot"] = ("dentro" if dentro
                                         else "acima" if pd_c > hi else "abaixo")
            rows.append(row)

        out = pd.DataFrame(rows)
        out.attrs.update(sample=sample, check_sample=check_sample, ci=ci, n_boot=n_boot)
        return out

    # ==================================================================
    # VALIDAÇÃO: backtesting por safra, monotonicidade e calibração
    # ==================================================================
    def _predicted_series(self) -> pd.Series:
        """alvo previsto por linha = alvo médio do segmento na referência (régua)."""
        pred = pd.Series(np.nan, index=self.df.index, dtype="float64")
        for sid, seg in self.segments.items():
            if not seg["is_leaf"]:
                continue
            sub = self.df[seg["mask"]]
            if self.sample_col is not None:
                ref = sub.loc[sub[self.sample_col] == self.ref_sample, self.target]
                pdv = float(ref.mean()) if len(ref) else float(sub[self.target].mean())
            else:
                pdv = float(sub[self.target].mean()) if len(sub) else np.nan
            pred[seg["mask"]] = pdv
        return pred

    # ------------------------------------------------------------------
    # BACKTEST: alvo previsto (régua) vs realizada ao longo do tempo (safra).
    #   `time_col` = coluna de período (ex.: dt_ref). Em cada período traz n,
    #   alvo previsto, taxa de default realizada, o gap e um status (ok/alerta
    #   por `tol`).
    # ------------------------------------------------------------------
    def backtest(self, time_col: str, sample: str | None = None,
                 tol: float | None = None) -> pd.DataFrame:
        if tol is None:                       # tolerância de gap default por tipo de alvo
            tol = 0.03 if self._is_clf else 0.10
        if time_col not in self.df.columns:
            raise ValueError(f"Coluna de tempo '{time_col}' não está no DataFrame.")
        pred = self._predicted_series()
        mask = pd.Series(True, index=self.df.index)
        if sample is not None and self.sample_col is not None:
            mask = self.df[self.sample_col] == sample
        base = pd.DataFrame({
            "periodo": self.df.loc[mask, time_col].values,
            "real": self.df.loc[mask, self.target].values,
            "prev": pred[mask].values,
        })
        g = base.groupby("periodo", dropna=False)
        out = g.agg(n=("real", "size"),
                    valor_previsto=("prev", "mean"),
                    valor_realizado=("real", "mean")).reset_index()
        out["gap"] = out["valor_realizado"] - out["valor_previsto"]
        out["status"] = out["gap"].abs().map(lambda d: "ok" if d <= tol else "alerta")
        for c in ("valor_previsto", "valor_realizado", "gap"):
            out[c] = out[c].round(4)
        out = out.sort_values("periodo").reset_index(drop=True)
        out.attrs.update(time_col=time_col, sample=sample, tol=tol)
        return out

    def plot_backtest(self, time_col: str, sample: str | None = None,
                      tol: float | None = None, figsize=(7.2, 4.2),
                      save_path: str | None = None, dpi: int = 150, ax=None):
        """Backtest **gráfico**: alvo previsto (régua) × realizado por safra, com a
        banda de tolerância do gap e as safras em **alerta** destacadas. Leitura
        visual do :meth:`backtest` (sem tabela)."""
        import matplotlib.pyplot as plt  # noqa: F401
        from matplotlib.ticker import PercentFormatter
        bt = self.backtest(time_col, sample=sample, tol=tol)
        tol = bt.attrs.get("tol", 0.03 if self._is_clf else 0.10)
        fig, ax = self._new_ax(figsize, dpi, ax)
        if bt.empty:
            ax.text(0.5, 0.5, "sem dados para backtest", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            return fig
        labels = _fmt_safras([str(p) for p in bt["periodo"]])
        x = list(range(len(bt)))
        prev = bt["valor_previsto"].to_numpy(dtype="float64")
        real = bt["valor_realizado"].to_numpy(dtype="float64")
        ax.fill_between(x, prev - tol, prev + tol, color="#9bb7c9", alpha=0.22,
                        label=f"tolerância ±{tol}")
        ax.plot(x, prev, color="#0f3d57", lw=1.8, marker="o", ms=4.5,
                label="previsto (régua)")
        ax.plot(x, real, color="#d6453e", lw=1.8, marker="s", ms=4.5, label="realizado")
        alert = (bt["status"].to_numpy() == "alerta")
        if alert.any():
            xa = np.asarray(x)[alert]
            ax.scatter(xa, real[alert], s=130, facecolors="none", edgecolors="#d6453e",
                       linewidths=1.7, zorder=4, label="alerta (|gap| > tol)")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylabel(self._risk_word); ax.set_xlabel("safra")
        ax.set_title(f"Backtest da régua — {self._risk_word} previsto × realizado por safra",
                     fontsize=12, fontweight="bold", color="#15324a")
        ax.legend(fontsize=8, loc="best"); ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # MONOTONICITY_REPORT: verifica se o alvo cresce ao longo das folhas 1..N
    #   (posição esquerda→direita), em cada amostra — tanto na referência (DES)
    #   quanto nas demais (estabilidade). Lista as inversões (pares de notas
    #   consecutivas onde o alvo cai).
    # ------------------------------------------------------------------
    def monotonicity_report(self) -> pd.DataFrame:
        lv = self.leaves()
        order = lv["segmento"].tolist()
        if self.sample_col is None:
            amostras = [None]
        else:
            todas = list(self.df[self.sample_col].dropna().unique())
            amostras = [self.ref_sample] + [a for a in todas if a != self.ref_sample]
        rows = []
        for a in amostras:
            vals = []
            for sid in order:
                m = self.segments[sid]["mask"]
                if a is not None:
                    m = m & (self.df[self.sample_col] == a)
                s = self.df.loc[m, self.target]
                vals.append(float(s.mean()) if len(s) else np.nan)
            inv = []
            for i in range(len(vals) - 1):
                v0, v1 = vals[i], vals[i + 1]
                if not (np.isnan(v0) or np.isnan(v1)) and v1 < v0:
                    inv.append((int(lv.iloc[i]["nota"]), int(lv.iloc[i + 1]["nota"])))
            rows.append({"amostra": a if a is not None else "todos",
                         "monotonico": len(inv) == 0,
                         "n_inversoes": len(inv), "inversoes": inv})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # CALIBRATION_TABLE: por folha, alvo previsto (régua/DES) vs realizada na
    #   amostra de verificação (default = 1ª não-referência, ex.: OOT).
    # ------------------------------------------------------------------
    def calibration_table(self, check_sample: str | None = None) -> pd.DataFrame:
        if self.sample_col is not None and check_sample is None:
            nonref = [a for a in self.df[self.sample_col].dropna().unique()
                      if a != self.ref_sample]
            check_sample = nonref[0] if nonref else self.ref_sample
        lv = self.leaves()
        rows = []
        for _, r in lv.iterrows():
            sid = r["segmento"]
            prev_vals = self._leaf_target(sid)               # régua = DES (ou todos)
            prev = float(prev_vals.mean()) if len(prev_vals) else np.nan
            m = self.segments[sid]["mask"]
            if check_sample is not None and self.sample_col is not None:
                m = m & (self.df[self.sample_col] == check_sample)
            real_vals = self.df.loc[m, self.target]
            real = float(real_vals.mean()) if len(real_vals) else np.nan
            rows.append({
                "nota": int(r["nota"]), "descricao": r["descricao"],
                "n": int(len(real_vals)),
                "valor_previsto": round(prev, 4) if not np.isnan(prev) else np.nan,
                "valor_realizado": round(real, 4) if not np.isnan(real) else np.nan,
                "gap": (round(real - prev, 4)
                        if not (np.isnan(real) or np.isnan(prev)) else np.nan),
            })
        out = pd.DataFrame(rows)
        out.attrs["check_sample"] = check_sample
        return out

    # ------------------------------------------------------------------
    # PLOT_CALIBRATION: dispersão alvo previsto (x) vs realizada (y) por folha,
    #   com a diagonal y=x (calibração perfeita) e faixa de tolerância.
    # ------------------------------------------------------------------
    def plot_calibration(self, check_sample: str | None = None, tol: float | None = None,
                         figsize=(6.0, 6.0), save_path: str | None = None,
                         dpi: int = 150, ax=None):
        if tol is None:
            tol = 0.02 if self._is_clf else 0.05
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_calibration requer matplotlib.") from e
        ct = self.calibration_table(check_sample)
        chk = ct.attrs.get("check_sample")
        fig, ax = self._new_ax(figsize, dpi, ax)
        x = ct["valor_previsto"].to_numpy(dtype="float64")
        y = ct["valor_realizado"].to_numpy(dtype="float64")
        lim_hi = float(np.nanmax([np.nanmax(x) if len(x) else 0,
                                  np.nanmax(y) if len(y) else 0, 0.05])) * 1.15
        diag = np.linspace(0, lim_hi, 50)
        ax.fill_between(diag, diag - tol, diag + tol, color="#9bb7c9", alpha=0.25,
                        label=f"tolerância ±{tol}")
        ax.plot(diag, diag, color="#0f3d57", lw=1.3, label="calibração perfeita")
        for _, r in ct.iterrows():
            if np.isnan(r["valor_previsto"]) or np.isnan(r["valor_realizado"]):
                continue
            ok = abs(r["gap"]) <= tol
            ax.scatter(r["valor_previsto"], r["valor_realizado"], s=70,
                       color=("#1aa64b" if ok else "#d6453e"),
                       edgecolor="#33424f", zorder=3)
            ax.annotate(str(r["nota"]),
                        (r["valor_previsto"], r["valor_realizado"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_xlim(0, lim_hi)
        ax.set_ylim(0, lim_hi)
        _prev = f"{self._risk_word} previsto"
        _real = f"{self._risk_word} realizado"
        ax.set_xlabel(f"{_prev} (régua · {self.ref_sample if self.sample_col else 'todos'})")
        ax.set_ylabel(f"{_real} ({chk if chk else 'todos'})")
        ax.set_title(f"Calibração da régua de {self._risk_word} por folha", fontsize=12,
                     fontweight="bold", color="#15324a")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # VALIDATION_REPORT: gera um relatório de validação em Markdown reunindo
    #   árvore (imagem), folhas, monotonicidade, PSI, CSI, métricas, calibração
    #   (tabela + imagem) e backtest por safra (se `time_col`). As imagens são
    #   salvas ao lado do .md. Documento único de governança (CMN 4.966/IFRS 9).
    # ------------------------------------------------------------------
    @staticmethod
    def _df_to_md(df: pd.DataFrame) -> str:
        def cell(v):
            try:
                if pd.isna(v):                  # cobre None, NaN, pd.NA, pd.NaT
                    return "—"
            except (TypeError, ValueError):
                pass
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        cols = list(df.columns)
        linhas = ["| " + " | ".join(str(c) for c in cols) + " |",
                  "| " + " | ".join("---" for _ in cols) + " |"]
        for _, r in df.iterrows():
            linhas.append("| " + " | ".join(cell(r[c]) for c in cols) + " |")
        return "\n".join(linhas)

    # ------------------------------------------------------------------
    # Plots de DISTRIBUIÇÃO do alvo (só fazem sentido em regressão)
    # ------------------------------------------------------------------
    def plot_leaf_boxplots(self, sample: str | None = None, ascending: bool = True,
                           cmap: str = "RdYlGn_r", figsize=None,
                           save_path: str | None = None, dpi: int = 150, ax=None):
        """Boxplot do alvo por folha (regressão). Mostra a dispersão dentro de
        cada folha, não só a média."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_leaf_boxplots requer matplotlib.") from e
        lv = self.leaves(ascending=ascending)
        ids, notas, descr, vals = [], [], [], []
        for _, r in lv.iterrows():
            sid = r["segmento"]
            m = self.segments[sid]["mask"]
            if sample is not None and self.sample_col is not None:
                m = m & (self.df[self.sample_col] == sample)
            v = self.df.loc[m, self.target].to_numpy(dtype="float64")
            v = v[~np.isnan(v)]
            if len(v) == 0:
                continue
            ids.append(sid); notas.append(int(r["nota"]))
            descr.append(r["descricao"]); vals.append(v)
        if not vals:
            raise ValueError("Sem dados para o boxplot.")
        if figsize is None:                       # mais largo e ALTO (legibilidade)
            figsize = (max(10.0, len(vals) * 1.6), 5.6)
        fig, ax = self._new_ax(figsize, dpi, ax)
        # [0,1] como piso quando os dados residem no intervalo, ESTENDIDO quando há
        # alvo fora de [0,1] (recuperações líquidas) — sem recortar caixas nem saturar
        # a cor (que mapeia a média de cada folha).
        means = np.asarray([float(np.mean(v)) for v in vals], dtype="float64")
        ylo, yhi = self._hist_domain(np.concatenate(vals))
        clo, chi = self._hist_domain(means)
        norm = Normalize(clo, chi)
        cmap_obj = plt.get_cmap(cmap)
        bp = ax.boxplot(vals, positions=range(1, len(vals) + 1), widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="#15324a", lw=1.4))
        for patch, v in zip(bp["boxes"], vals):
            patch.set_facecolor(cmap_obj(norm(float(np.mean(v)))))
            patch.set_edgecolor("#33424f")
            patch.set_alpha(0.92)
        ax.set_xticks(range(1, len(vals) + 1))
        ax.set_xticklabels([f"folha {n}" for n in notas], rotation=0, fontsize=9)
        ax.set_ylabel("alvo")
        sfx = f" · {sample}" if sample else ""
        ax.set_title(f"Dispersão do alvo por folha{sfx}", fontsize=12,
                     fontweight="bold", color="#15324a")
        ax.set_ylim(ylo, yhi)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    @staticmethod
    def _hist_domain(y, base_lo: float = 0.0, base_hi: float = 1.0):
        """Faixa do eixo para histogramas do alvo: ``[0,1]`` por padrão, mas ESTENDIDA
        para incluir massa fora de ``[0,1]`` (ex.: recuperações líquidas). O
        ``hist`` descarta valores fora de ``range``; sem estender, alvo<0 ou >1 sumiria
        do gráfico e a contagem deixaria de somar n (a média, calculada sobre todo y,
        cairia fora do canvas)."""
        if len(y) == 0:
            return (base_lo, base_hi)
        return (min(base_lo, float(np.min(y))), max(base_hi, float(np.max(y))))

    def plot_target_hist(self, sample: str | None = None, bins: int = 30,
                         color: str = "#2a9d8f", figsize=(12.0, 3.8),
                         save_path: str | None = None, dpi: int = 150, ax=None):
        """Histograma do alvo na carteira (regressão) — revela bimodalidade
        (massa em 0 e/ou 1) e concentração."""
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_target_hist requer matplotlib.") from e
        fig, ax = self._new_ax(figsize, dpi, ax)
        if sample is None and self.sample_col is not None:
            sample = self.ref_sample
        if sample is not None and self.sample_col is not None:
            y = self.df.loc[self.df[self.sample_col] == sample, self.target].to_numpy(dtype="float64")
            sfx = f" — {sample}"
        else:
            y = self.df[self.target].to_numpy(dtype="float64")
            sfx = ""
        y = y[~np.isnan(y)]
        lo, hi = self._hist_domain(y)
        ax.hist(y, bins=bins, range=(lo, hi), color=color, alpha=0.9,
                edgecolor="#15324a")
        if len(y):
            ax.axvline(float(np.mean(y)), color="#d6453e", lw=1.6, ls="--",
                       label=f"média {np.mean(y):.3f}")
            ax.legend(fontsize=8)
        ax.set_xlabel("alvo")
        ax.set_ylabel("frequência")
        ax.set_title(f"Distribuição do alvo{sfx}", fontsize=12,
                     fontweight="bold", color="#15324a")
        ax.set_xlim(lo, hi)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_leaf_value_hist(self, sid: str | None = None, sample: str | None = None,
                             bins: int = 24, figsize=(5.2, 2.9),
                             save_path: str | None = None, dpi: int = 150, ax=None):
        """Histograma do alvo numa folha (regressão)."""
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_leaf_value_hist requer matplotlib.") from e
        mask = (pd.Series(True, index=self.df.index)
                if sid is None or sid not in self.segments
                else self.segments[sid]["mask"].copy())
        if sample is None and self.sample_col is not None:
            sample = self.ref_sample
        sfx = ""
        if sample is not None and self.sample_col is not None:
            mask = mask & (self.df[self.sample_col] == sample)
            sfx = f" · {sample}"
        y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
        y = y[~np.isnan(y)]
        fig, ax = self._new_ax(figsize, dpi, ax)
        if y.size == 0:
            ax.text(0.5, 0.5, "sem dados nesta folha/amostra", ha="center", va="center",
                    transform=ax.transAxes, color="#889")
            ax.axis("off"); fig.tight_layout()
            return fig
        lo, hi = self._hist_domain(y)
        ax.hist(y, bins=bins, range=(lo, hi), color="steelblue", alpha=0.85,
                edgecolor="#2f5d82")
        m = float(np.mean(y))
        ax.axvline(m, color="crimson", lw=1.8, ls="--", label=f"média {m:.3f}")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.set_xlabel("alvo"); ax.set_ylabel("freq.")
        ax.set_title(f"Alvo médio da folha{sfx} (n={y.size})", fontsize=10.5,
                     fontweight="bold", color="#15324a")
        ax.set_xlim(lo, hi); ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def validation_report(self, path: str = "relatorio_validacao.md",
                          time_col: str | None = None,
                          title: str = "Relatório de Validação — Segmentação",
                          tol_backtest: float | None = None, tol_calib: float | None = None,
                          stamp: str | None = None) -> str:
        import os
        if tol_backtest is None:
            tol_backtest = 0.03 if self._is_clf else 0.10
        if tol_calib is None:
            tol_calib = 0.02 if self._is_clf else 0.05

        base = os.path.dirname(os.path.abspath(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        img_arvore = f"{stem}_arvore.png"
        img_calib = f"{stem}_calibracao.png"

        import matplotlib.pyplot as plt
        try:
            fig = self.plot_tree(save_path=os.path.join(base, img_arvore))
            plt.close(fig)
            tem_arvore = True
        except Exception:
            plt.close("all")                    # não vaza figura se o plot falhar
            tem_arvore = False
        tem_calib = False
        if self.sample_col is not None:
            try:
                fig = self.plot_calibration(save_path=os.path.join(base, img_calib),
                                            tol=tol_calib)
                plt.close(fig)
                tem_calib = True
            except Exception:
                plt.close("all")
                tem_calib = False

        regua = self._regua_dict()
        feats = self.regua_features()
        n_folhas = len(regua["leaves"])
        prof = max(s["depth"] for s in self.segments.values())

        L = [f"# {title}", ""]
        if stamp:
            L.append(f"_Gerado em {stamp}._\n")
        L += ["## Visão geral", "",
              f"- **Target:** `{self.target}`",
              f"- **Amostra de referência:** `{self.ref_sample}`"
              + ("" if self.sample_col is None else f" (coluna `{self.sample_col}`)"),
              f"- **Folhas:** {n_folhas} · **profundidade máxima:** {prof}",
              f"- **Variáveis usadas:** {', '.join(feats) or '(nenhuma)'}",
              f"- **Linhas:** {len(self.df):,}".replace(",", "."), ""]

        if tem_arvore:
            L += ["## Árvore de segmentação", "", f"![arvore]({img_arvore})", ""]

        L += ["## Folhas", "",
              self._df_to_md(self.leaves(with_psi=(self.sample_col is not None))), ""]

        mr = self.monotonicity_report()
        ok = bool(mr["monotonico"].all())
        L += [f"## Monotonicidade de {self._risk_word} por nota", "",
              (f"✅ {self._risk_word} cresce monotonicamente em todas as amostras."
               if ok else "⚠️ Há inversões de monotonicidade — ver tabela."),
              "", self._df_to_md(mr[["amostra", "monotonico", "n_inversoes"]]), ""]

        if self.sample_col is not None:
            try:
                L += ["## PSI (estabilidade da segmentação)", "",
                      self._df_to_md(self.psi()), ""]
            except Exception:
                pass
            try:
                L += ["## CSI por variável (estabilidade das entradas)", "",
                      self._df_to_md(self.csi()), ""]
            except Exception:
                pass

        try:
            L += [f"## Discriminação (régua como modelo de {self._risk_word})", "",
                  self._df_to_md(self.metrics()), ""]
        except Exception:
            pass

        if self.sample_col is not None:
            ct = self.calibration_table()
            L += [f"## Calibração (prevista DES × realizada {ct.attrs.get('check_sample')})",
                  "", self._df_to_md(ct[["nota", "n", "valor_previsto",
                                         "valor_realizado", "gap"]]), ""]
            if tem_calib:
                L += [f"![calibracao]({img_calib})", ""]

        if time_col is not None:
            try:
                bt = self.backtest(time_col, tol=tol_backtest)
                L += ["## Backtest por safra (prevista × realizada no tempo)", "",
                      self._df_to_md(bt), ""]
            except Exception as e:
                L += ["## Backtest por safra", "", f"_não gerado: {e}_", ""]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(L))
        return path

    # ==================================================================
    # QUALIDADE DOS SEGMENTOS / DISCRIMINAÇÃO: gráficos
    # ==================================================================
    @staticmethod
    def _new_ax(figsize, dpi, ax):
        # Figura SEM pyplot (não entra no Gcf): evita o backend inline
        # re-exibir o gráfico (duplicação) além do display explícito do widget.
        if ax is not None:
            return ax.figure, ax
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(figsize=figsize, dpi=dpi)
        FigureCanvasAgg(fig)
        return fig, fig.subplots()

    @staticmethod
    def _wilson_ci(k, n, z=1.96):
        """Intervalo de Wilson para uma proporção (taxa de default por folha)."""
        if n == 0:
            return (np.nan, np.nan, np.nan)
        p = k / n
        denom = 1 + z * z / n
        centro = (p + z * z / (2 * n)) / denom
        meio = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
        return p, max(0.0, centro - meio), min(1.0, centro + meio)

    def _samples_with_both_classes(self):
        """Amostras (referência primeiro) que têm as DUAS classes — usáveis em
        ROC/KS. Sem sample_col, devolve [(None, máscara total)]."""
        if self.sample_col is None:
            y = self.df[self.target].to_numpy(dtype="float64")
            y = y[~np.isnan(y)]
            return [(None, pd.Series(True, index=self.df.index))] if np.unique(y).size >= 2 else []
        todas = list(self.df[self.sample_col].dropna().unique())
        ordem = [self.ref_sample] + [a for a in todas if a != self.ref_sample]
        out = []
        for a in ordem:
            mask = self.df[self.sample_col] == a
            y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
            y = y[~np.isnan(y)]
            if np.unique(y).size >= 2:
                out.append((a, mask))
        return out

    # ------------------------------------------------------------------
    # PLOT_ROC: curva ROC da régua (score = alvo previsto) por amostra, com a AUC
    #   na legenda. Mede o poder de ORDENAÇÃO de risco da segmentação.
    # ------------------------------------------------------------------
    def plot_roc(self, samples: list | None = None, figsize=(6.0, 6.0),
                 save_path: str | None = None, dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
            from sklearn.metrics import roc_auc_score, roc_curve
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_roc requer matplotlib e scikit-learn.") from e
        pred = self._predicted_series()
        grupos = self._samples_with_both_classes()
        if samples is not None:
            grupos = [(a, m) for a, m in grupos if a in samples]
        fig, ax = self._new_ax(figsize, dpi, ax)
        ax.plot([0, 1], [0, 1], color="#9aa7b2", lw=1.0, ls="--")
        cores = ["#0f3d57", "#d6453e", "#1aa64b", "#caa000", "#6b3fa0", "#2a9d8f"]
        if not grupos:
            ax.text(0.5, 0.5, "sem as duas classes para a curva ROC", ha="center",
                    va="center", transform=ax.transAxes, color="#889")
        for i, (a, mask) in enumerate(grupos):
            y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
            s = pred[mask.values].to_numpy(dtype="float64")
            ok = ~(np.isnan(y) | np.isnan(s))
            y, s = y[ok], s[ok]
            if np.unique(y).size < 2:
                continue
            fpr, tpr, _ = roc_curve(y, s)
            auc = roc_auc_score(y, s)
            nome = a if a is not None else "todos"
            ax.plot(fpr, tpr, color=cores[i % len(cores)], lw=2.0,
                    label=f"{nome} · AUC {auc:.3f} · Gini {2*auc-1:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Falso positivo (1 − especificidade)")
        ax.set_ylabel("Verdadeiro positivo (sensibilidade)")
        ax.set_title(f"Curva ROC da régua de {self._risk_word}", fontsize=12, fontweight="bold",
                     color="#15324a")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # PLOT_KS: curva KS — distribuições acumuladas de bons (alvo 0) e maus
    #   (alvo 1) ao longo do score (alvo previsto), com a distância KS marcada.
    # ------------------------------------------------------------------
    def plot_ks(self, sample: str | None = None, figsize=(7.0, 4.6),
                save_path: str | None = None, dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_ks requer matplotlib.") from e
        pred = self._predicted_series()
        if sample is None and self.sample_col is not None:
            sample = self.ref_sample
        mask = pd.Series(True, index=self.df.index)
        if sample is not None and self.sample_col is not None:
            mask = self.df[self.sample_col] == sample
        y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
        s = pred[mask.values].to_numpy(dtype="float64")
        ok = ~(np.isnan(y) | np.isnan(s))
        y, s = y[ok], s[ok]
        fig, ax = self._new_ax(figsize, dpi, ax)
        if np.unique(y).size < 2:
            ax.text(0.5, 0.5, "sem as duas classes para o KS", ha="center",
                    va="center", transform=ax.transAxes, color="#889")
            ax.axis("off"); fig.tight_layout()
            return fig
        order = np.argsort(s, kind="mergesort")
        s_ord, y_ord = s[order], y[order]
        tot_bad = y_ord.sum()
        tot_good = (1 - y_ord).sum()
        cum_bad = np.cumsum(y_ord) / tot_bad
        cum_good = np.cumsum(1 - y_ord) / tot_good
        diff = cum_good - cum_bad           # bons acumulam mais rápido nos scores baixos
        j = int(np.argmax(diff))
        ks = float(diff[j])
        x = s_ord
        ax.plot(x, cum_good, color="#1aa64b", lw=2.0, label="acum. bons (alvo 0)")
        ax.plot(x, cum_bad, color="#d6453e", lw=2.0, label="acum. maus (alvo 1)")
        ax.vlines(x[j], cum_bad[j], cum_good[j], color="#0f3d57", lw=1.8, ls="--")
        ax.annotate(f"KS = {ks:.3f}", (x[j], (cum_good[j] + cum_bad[j]) / 2),
                    textcoords="offset points", xytext=(8, 0), fontsize=10,
                    fontweight="bold", color="#0f3d57")
        sfx = f" · {sample}" if sample else ""
        ax.set_xlabel(f"score ({self._risk_word} previsto)")
        ax.set_ylabel("proporção acumulada")
        ax.set_title(f"Curva KS da régua de {self._risk_word}{sfx}", fontsize=12, fontweight="bold",
                     color="#15324a")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.2)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # PLOT_LEAF_BADRATE: taxa de default (alvo) por folha (barras na ordem da
    #   nota), com IC de Wilson — mostra a separação de risco entre folhas e a
    #   incerteza amostral de cada alvo. Substitui o boxplot (alvo binário).
    # ------------------------------------------------------------------
    def plot_leaf_badrate(self, sample: str | None = None, ascending: bool = True,
                          cmap: str = "RdYlGn_r", figsize=None,
                          save_path: str | None = None, dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_leaf_badrate requer matplotlib.") from e
        if sample is None and self.sample_col is not None:
            sample = self.ref_sample
        lv = self.leaves(ascending=ascending)
        notas, rates, los, his = [], [], [], []
        for _, r in lv.iterrows():
            sid = r["segmento"]
            m = self.segments[sid]["mask"]
            if sample is not None and self.sample_col is not None:
                m = m & (self.df[self.sample_col] == sample)
            v = self.df.loc[m, self.target].to_numpy(dtype="float64")
            v = v[~np.isnan(v)]
            if len(v) == 0:
                continue
            p, lo, hi = self._wilson_ci(int(v.sum()), len(v))
            notas.append(int(r["nota"])); rates.append(p); los.append(lo); his.append(hi)
        if not rates:
            raise ValueError("Sem dados para a taxa de default por folha.")
        if figsize is None:
            figsize = (max(6.0, len(rates) * 1.1), 4.6)
        fig, ax = self._new_ax(figsize, dpi, ax)
        vmax = max(his) * 1.1 if his else 0.1
        norm = Normalize(0.0, vmax if vmax > 1e-9 else 0.01)
        cmap_obj = plt.get_cmap(cmap)
        xs = list(range(1, len(rates) + 1))
        yerr = [[p - lo for p, lo in zip(rates, los)],
                [hi - p for p, hi in zip(rates, his)]]
        bars = ax.bar(xs, rates, width=0.62, color=[cmap_obj(norm(p)) for p in rates],
                      edgecolor="#33424f", alpha=0.92,
                      yerr=yerr, capsize=4, error_kw=dict(ecolor="#33424f", lw=1.1))
        for x, p in zip(xs, rates):
            ax.text(x, p, f"{p:.3f}", ha="center", va="bottom", fontsize=8.5,
                    color="#15324a", fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"folha {n}" for n in notas], rotation=0, fontsize=9)
        ax.set_ylabel(f"Taxa de default ({self._risk_word})")
        sfx = f" · {sample}" if sample else ""
        ax.set_title(f"Taxa de default por folha{sfx} (IC de Wilson 95%)", fontsize=12,
                     fontweight="bold", color="#15324a")
        ax.set_ylim(0, vmax if vmax > 1e-9 else 0.05)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # PLOT_SCORE_DISTRIBUTION: distribuição do score (alvo previsto) separada por
    #   classe (bons × maus) — quanto mais separadas as massas, melhor o KS/AUC.
    # ------------------------------------------------------------------
    def plot_score_distribution(self, sample: str | None = None, bins: int = 30,
                                figsize=(7.0, 4.2), save_path: str | None = None,
                                dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_score_distribution requer matplotlib.") from e
        pred = self._predicted_series()
        if sample is None and self.sample_col is not None:
            sample = self.ref_sample
        mask = pd.Series(True, index=self.df.index)
        sfx = ""
        if sample is not None and self.sample_col is not None:
            mask = self.df[self.sample_col] == sample
            sfx = f" — {sample}"
        y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
        s = pred[mask.values].to_numpy(dtype="float64")
        ok = ~(np.isnan(y) | np.isnan(s))
        y, s = y[ok], s[ok]
        fig, ax = self._new_ax(figsize, dpi, ax)
        if s.size == 0:
            ax.text(0.5, 0.5, "sem score para exibir", ha="center", va="center",
                    transform=ax.transAxes, color="#889")
            ax.axis("off"); fig.tight_layout()
            return fig
        rng = (float(np.min(s)), float(np.max(s)) if np.max(s) > np.min(s)
               else float(np.min(s)) + 0.01)
        good, bad = s[y == 0], s[y == 1]
        ax.hist(good, bins=bins, range=rng, color="#1aa64b", alpha=0.55,
                edgecolor="#157a52", label=f"bons (n={good.size})", density=True)
        ax.hist(bad, bins=bins, range=rng, color="#d6453e", alpha=0.55,
                edgecolor="#b23a2a", label=f"maus (n={bad.size})", density=True)
        ax.set_xlabel(f"score ({self._risk_word} previsto)")
        ax.set_ylabel("densidade")
        ax.set_title(f"Distribuição do score por classe{sfx}", fontsize=12,
                     fontweight="bold", color="#15324a")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # PLOT_FEATURE_PD: preview da variável candidata DENTRO de uma folha —
    #   taxa de default (alvo) por faixa da variável (com a representatividade),
    #   nos mesmos bins do split. Mostra a FORMA da relação antes de dividir e
    #   se é monotônica. Use antes do `grow` para escolher a variável/corte.
    # ------------------------------------------------------------------
    def plot_feature_value(self, feature: str, sid: str | None = None, splits=None,
                        dtype=None, max_n_bins: int = 6, min_bin_size: float = 0.05,
                        max_bin_size=None, min_mean_diff=0.0, figsize=None,
                        save_path: str | None = None, dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_feature_value requer matplotlib.") from e
        if sid is None or sid not in self.segments:
            sid, sub = "root", self.df
        else:
            sub = self.df[self.segments[sid]["mask"]]
        # `splits` (mesmos do preview/grow) garante que o gráfico bata com a tabela
        bins, modo, kind = self._resolve_bins_cached(sid, feature, splits, dtype,
                                                     max_n_bins, min_bin_size, max_bin_size,
                                                     min_mean_diff=min_mean_diff)
        if not bins:
            raise ValueError(f"Sem faixas válidas para '{feature}' nesta folha.")
        tbl = self._bin_table(sub, feature, bins, len(self.df))
        if tbl.empty:
            raise ValueError(f"Sem dados de '{feature}' nesta folha.")
        labels = [s.split(": ", 1)[-1] for s in tbl["faixa"]]
        pds = tbl["valor_medio"].to_numpy()
        reprs = tbl["repr_%"].to_numpy()
        if figsize is None:
            figsize = (max(6.0, len(labels) * 1.25), 4.6)
        fig, ax = self._new_ax(figsize, dpi, ax)
        xs = list(range(len(labels)))

        # BARRAS = representatividade/volumetria (%), eixo esquerdo (steelblue)
        # LINHA  = taxa de default (alvo), eixo direito (crimson) — tema padrão da lib
        col_bar, col_line = "steelblue", "crimson"
        bars = ax.bar(xs, reprs, color=col_bar, edgecolor="#2f5d82", alpha=0.85,
                      width=0.62, label="Representatividade (%)")
        ax.set_ylabel("Representatividade (%)", color=col_bar, fontweight="bold")
        ax.tick_params(axis="y", labelcolor=col_bar)
        rmax = float(np.nanmax(reprs)) or 1.0
        ax.set_ylim(0, rmax * 1.25 + 1)
        for x, rp in zip(xs, reprs):
            dentro = rp > rmax * 0.18          # barra alta: rótulo branco dentro
            ax.text(x, rp * 0.5 if dentro else rp + rmax * 0.02, f"{rp:.1f}%",
                    ha="center", va="center" if dentro else "bottom", fontsize=8,
                    color="white" if dentro else col_bar, fontweight="bold")

        # LINHA = risco médio por faixa, eixo da direita (vermelho)
        _lab = f"Taxa de default ({self._risk_word})" if self._is_clf else self._risk_word
        ax2 = ax.twinx()
        line, = ax2.plot(xs, pds, color=col_line, marker="o", lw=2.2,
                         markersize=7, markeredgecolor="#fff", zorder=5,
                         label=_lab)
        ax2.set_ylabel(_lab, color=col_line, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor=col_line)
        pmax = float(np.nanmax(pds)) if pds.size else 0.0
        ax2.set_ylim(0, max(pmax * 1.3 + 0.02, 0.02))
        for x, v in zip(xs, pds):
            ax2.text(x, v + max(pmax, 0.01) * 0.04, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8, color=col_line,
                     fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
        rot = self.feature_labels.get(feature, feature)
        mono = "monotônico" if tbl.attrs.get("mono_ok") else "NÃO monotônico"
        ax.set_title(f"'{rot}': representatividade (barra) × {self._risk_word} (linha) — "
                     f"{modo}, {mono}", fontsize=11.5, fontweight="bold", color="#15324a")
        ax.legend(handles=[bars, line], loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # PLOT_FEATURE_HIST: histograma de uma variável NUMÉRICA dentro de uma
    #   folha, com linhas verticais nos cortes propostos (ótimo ou manual).
    #   Mostra a forma da distribuição da variável antes de dividir. Para
    #   variável categórica cai no plot_feature_value (barras × alvo).
    # ------------------------------------------------------------------
    def plot_feature_hist(self, feature: str, sid: str | None = None, splits=None,
                          dtype=None, max_n_bins: int = 6, min_bin_size: float = 0.05,
                          max_bin_size=None, min_mean_diff=0.0, bins_hist: int = 30,
                          figsize=None, save_path: str | None = None, dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_feature_hist requer matplotlib.") from e
        if sid is None or sid not in self.segments:
            sid, sub = "root", self.df
        else:
            sub = self.df[self.segments[sid]["mask"]]
        kind = self._detect_kind(sub, feature, dtype)
        if kind != "num":
            # categórica não tem histograma — barras de representatividade × alvo
            return self.plot_feature_value(feature, sid=sid, splits=splits, dtype=dtype,
                                        max_n_bins=max_n_bins, min_bin_size=min_bin_size,
                                        max_bin_size=max_bin_size,
                                        min_mean_diff=min_mean_diff,
                                        figsize=figsize, save_path=save_path, dpi=dpi, ax=ax)
        x = sub[feature].to_numpy(dtype="float64")
        x = x[~np.isnan(x)]
        if x.size == 0:
            raise ValueError(f"Sem valores numéricos de '{feature}' nesta folha.")
        # cortes (linhas verticais) a partir dos bins resolvidos
        cortes, modo = [], "—"
        try:
            bins, modo, _ = self._resolve_bins_cached(sid, feature, splits, dtype,
                                                      max_n_bins, min_bin_size, max_bin_size,
                                                      min_mean_diff=min_mean_diff)
            cortes = sorted({e for b in bins if b["kind"] == "num"
                             for e in (b["lo"], b["hi"]) if np.isfinite(e)})
        except Exception:
            pass
        if figsize is None:
            figsize = (7.0, 4.4)
        fig, ax = self._new_ax(figsize, dpi, ax)
        # barras = volumetria da variável (steelblue) · cortes = crimson
        ax.hist(x, bins=bins_hist, color="steelblue", alpha=0.85, edgecolor="#2f5d82")
        for c in cortes:
            ax.axvline(c, color="crimson", lw=1.8, ls="--")
        if cortes:
            ax.plot([], [], color="crimson", lw=1.8, ls="--",
                    label=f"cortes propostos ({len(cortes)})")
            ax.legend(fontsize=9, framealpha=0.9)
        rot = self.feature_labels.get(feature, feature)
        ax.set_xlabel(rot)
        ax.set_ylabel("frequência")
        ax.set_title(f"Distribuição de '{rot}' na folha — {modo}", fontsize=11.5,
                     fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # PLOT_LEAF_TARGET_HIST: alvo da folha (referência DES por padrão) como uma
    #   barra horizontal com IC de Wilson, comparada à alvo da carteira — leitura
    #   rápida do nível de risco da folha. (Alvo binário não tem histograma.)
    # ------------------------------------------------------------------
    def plot_leaf_target_hist(self, sid: str | None = None, sample: str | None = None,
                              bins: int = 24, figsize=(5.2, 2.0),
                              save_path: str | None = None, dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_leaf_target_hist requer matplotlib.") from e
        mask = (pd.Series(True, index=self.df.index)
                if sid is None or sid not in self.segments
                else self.segments[sid]["mask"].copy())
        if sample is None and self.sample_col is not None:
            sample = self.ref_sample
        sfx = ""
        if sample is not None and self.sample_col is not None:
            mask = mask & (self.df[self.sample_col] == sample)
            sfx = f" · {sample}"
        y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
        y = y[~np.isnan(y)]
        fig, ax = self._new_ax(figsize, dpi, ax)
        if y.size == 0:
            ax.text(0.5, 0.5, "sem alvo nesta folha/amostra", ha="center", va="center",
                    transform=ax.transAxes, color="#889")
            ax.axis("off"); fig.tight_layout()
            return fig
        if self._is_clf:
            p, lo, hi = self._wilson_ci(int(y.sum()), len(y))
        else:
            # alvo contínuo (alvo): ponto = média e IC = IC da média (normal). O IC
            # binomial de Wilson e o int(y.sum()) só valem para proporção 0/1 e
            # truncariam/distorceriam a estatística do alvo contínuo.
            p = float(np.mean(y))
            if y.size > 1:
                se = float(np.std(y, ddof=1)) / np.sqrt(y.size)
                lo, hi = p - 1.96 * se, p + 1.96 * se
            else:
                lo = hi = p
        # alvo da carteira (mesma amostra) como referência
        cart = self.df
        if sample is not None and self.sample_col is not None:
            cart = self.df[self.df[self.sample_col] == sample]
        yc = cart[self.target].to_numpy(dtype="float64")
        yc = yc[~np.isnan(yc)]
        pc = float(np.mean(yc)) if yc.size else np.nan
        xmax = max(hi, pc if not np.isnan(pc) else 0, 0.02) * 1.25
        ax.barh([0], [p], height=0.5, color="#d6453e" if (not np.isnan(pc) and p > pc)
                else "#1aa64b", alpha=0.85, edgecolor="#33424f",
                xerr=[[p - lo], [hi - p]], capsize=5, error_kw=dict(ecolor="#33424f", lw=1.1))
        ax.text(p, 0, f"  {self._risk_word} {p:.3f}", va="center", ha="left", fontsize=10,
                fontweight="bold", color="#15324a")
        if not np.isnan(pc):
            ax.axvline(pc, color="#0f3d57", lw=1.6, ls="--",
                       label=f"{self._risk_word} carteira {pc:.3f}")
            ax.legend(fontsize=8, loc="lower right")
        ax.set_yticks([])
        ax.set_xlim(0, xmax)
        ax.set_xlabel(f"Taxa de default ({self._risk_word})" if self._is_clf else self._risk_word)
        if self._is_clf:
            ax.set_title(f"{self._risk_word} da folha{sfx} (n={y.size}; maus={int(y.sum())})",
                         fontsize=10.5, fontweight="bold", color="#15324a")
        else:
            ax.set_title(f"{self._risk_word} da folha{sfx} (n={y.size})",
                         fontsize=10.5, fontweight="bold", color="#15324a")
        ax.grid(axis="x", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ==================================================================
    # ANÁLISE DE VARIÁVEIS: perfil estatístico, distribuição e estabilidade
    #   (PSI atual e por safra) de UMA variável de entrada, numa folha.
    # ==================================================================
    def _leaf_mask(self, sid):
        if sid is None or sid not in self.segments:
            return pd.Series(True, index=self.df.index)
        return self.segments[sid]["mask"].copy()

    def label(self, feature) -> str:
        """Rótulo de exibição da variável (alias de ``feature_labels``)."""
        return self.feature_labels.get(feature, feature)

    def variable_summary(self, feature, sid=None, sample=None) -> dict:
        """Resumo de uma variável numa folha (referência DES por padrão):
        %missing, média/mediana/desvio, n, percentis (min, p5, p95, max), o IV
        binário e o PSI atual por amostra (nos bins do IV)."""
        sid = sid if (sid in self.segments) else "root"
        mask = self._leaf_mask(sid)
        if sample is None and self.sample_col is not None:
            sample = self.ref_sample
        if sample is not None and self.sample_col is not None:
            mask = mask & (self.df[self.sample_col] == sample)
        col = self.df.loc[mask, feature]
        kind = self._detect_kind(self.df[mask], feature, None)
        n = int(len(col)); n_miss = int(col.isna().sum())
        res = {"variavel": feature, "tipo": kind, "amostra": sample, "n": n,
               "n_missing": n_miss,
               "pct_missing": round(100 * n_miss / n, 2) if n else float("nan")}
        if kind == "num":
            x = col.to_numpy(dtype="float64"); x = x[~np.isnan(x)]
            if x.size:
                res.update({
                    "media": round(float(np.mean(x)), 4),
                    "mediana": round(float(np.median(x)), 4),
                    "desvio": round(float(np.std(x, ddof=1)) if x.size > 1 else 0.0, 4),
                    "min": round(float(np.min(x)), 4),
                    "p5": round(float(np.percentile(x, 5)), 4),
                    "p95": round(float(np.percentile(x, 95)), 4),
                    "max": round(float(np.max(x)), 4)})
        else:
            vc = col.dropna().astype(str).value_counts(normalize=True)
            res["top_categorias"] = [(c, round(100 * p, 1)) for c, p in vc.head(8).items()]
        # IV binário + PSI atual (nos mesmos bins) desta variável na folha
        res["iv"], res["forca"], res["psi"], res["pior_psi"] = None, "—", {}, None
        try:
            ivt = self.variable_iv(sid, features=[feature], with_psi=True)
            if len(ivt):
                r0 = ivt.iloc[0]
                res["iv"] = None if pd.isna(r0["iv"]) else float(r0["iv"])
                res["forca"] = r0["forca"]
                for c in ivt.columns:
                    if c.startswith("psi_") and c != "psi_classificacao":
                        res["psi"][c[4:]] = None if pd.isna(r0[c]) else float(r0[c])
                if "pior_psi" in ivt.columns and not pd.isna(r0["pior_psi"]):
                    res["pior_psi"] = float(r0["pior_psi"])
        except Exception:
            pass
        return res

    def variable_table(self, feature, sid=None, sample=None, max_n_bins=6,
                       min_bin_size=0.05, splits=None) -> pd.DataFrame:
        """Tabela por FAIXA de uma variável numa folha (referência DES por padrão):
        ``faixa, n, repr_%, risco`` (event_rate/alvo médio) e — na **classificação** —
        ``woe, logodds`` e ``iv_parcial`` (escala WoE/IV de Siddiqi). Na **regressão**,
        ``iv_parcial`` é o desvio absoluto ponderado do alvo. IV total, monotonicidade
        e ``risco_ordenavel`` (sem a faixa NA) em ``.attrs``.

        Espelha :meth:`ModelSegmenter.variable_table` — base comum dos gráficos de
        distribuição+badrate e logodds na aba unificada de Análise de variáveis."""
        _EPS = 1e-6
        sid = sid if (sid in self.segments) else "root"
        mask = self._leaf_mask(sid)
        if sample is None and self.sample_col is not None:
            sample = self.ref_sample
        if sample is not None and self.sample_col is not None:
            mask = mask & (self.df[self.sample_col] == sample)
        sub = self.df[mask]
        n_tot = max(len(sub), 1)
        risco_label = "event_rate" if self._is_clf else "alvo_medio"
        bins, _modo, kind = self._resolve_bins(sub, feature, splits, None,
                                               max_n_bins, min_bin_size)
        if not bins:
            out = pd.DataFrame(columns=["faixa", "n", "repr_%", risco_label])
            out.attrs.update(iv=float("nan"), mono_ok=True, kind=kind,
                             risco_label=risco_label, risco_ordenavel=[])
            return out
        y_all = sub[self.target].to_numpy(dtype="float64")
        mean_global = float(np.nanmean(y_all)) if np.isfinite(y_all).any() else float("nan")
        n_evt_tot = float(np.nansum(y_all == 1)) if self._is_clf else 0.0
        n_non_tot = float(np.nansum(y_all == 0)) if self._is_clf else 0.0
        n_base = float(np.sum(~np.isnan(y_all)))
        rows, iv_total, is_na = [], 0.0, []
        for b in bins:
            m = self._mask_in(sub, feature, b).to_numpy()
            yi = y_all[m]; yi_ok = yi[~np.isnan(yi)]
            n_i = int(m.sum())
            if n_i == 0:
                continue
            risco = float(np.nanmean(yi)) if yi_ok.size else float("nan")
            row = {"faixa": self._bin_label(feature, b), "n": n_i,
                   "repr_%": round(100 * n_i / n_tot, 1),
                   risco_label: round(risco, 4) if np.isfinite(risco) else np.nan}
            if self._is_clf:
                n_evt = float((yi_ok == 1).sum()); n_non = float((yi_ok == 0).sum())
                d_evt = n_evt / max(n_evt_tot, _EPS); d_non = n_non / max(n_non_tot, _EPS)
                woe = float(np.log((d_non + _EPS) / (d_evt + _EPS)))
                logodds = (float(np.log((risco + _EPS) / (1 - risco + _EPS)))
                           if np.isfinite(risco) else np.nan)
                ivp = (d_non - d_evt) * woe
                row.update(woe=round(woe, 4),
                           logodds=round(logodds, 4) if np.isfinite(logodds) else np.nan,
                           iv_parcial=round(ivp, 4))
                iv_total += ivp
            else:
                ivp = ((yi_ok.size / max(n_base, _EPS)) * abs(risco - mean_global)
                       if np.isfinite(risco) else 0.0)
                row["iv_parcial"] = round(ivp, 4)
                iv_total += ivp
            rows.append(row)
            is_na.append(b.get("kind") == "na")
        out = pd.DataFrame(rows)
        if risco_label in out and len(out):
            rcol = out.loc[~np.asarray(is_na), risco_label]   # exclui a faixa NA
        else:
            rcol = pd.Series(dtype=float)
        mono = bool(rcol.is_monotonic_increasing or rcol.is_monotonic_decreasing) \
            if len(rcol) else True
        out.attrs.update(iv=round(float(iv_total), 4), mono_ok=mono, kind=kind,
                         risco_label=risco_label, mean_global=round(mean_global, 4),
                         risco_ordenavel=[float(x) for x in rcol])
        return out

    def variable_by_safra(self, feature, time_col=None, sid=None, sample=None) -> pd.DataFrame:
        """Percentis (min, p5, média, p95, max) e %missing da variável NUMÉRICA
        por safra (mês de `time_col`), dentro da folha. `time_col` default = date_col."""
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col no segmenter.")
        mask = self._leaf_mask(sid)
        if sample is not None and self.sample_col is not None:
            mask = mask & (self.df[self.sample_col] == sample)
        sub = self.df[mask]
        if time_col not in sub.columns:
            raise ValueError(f"Coluna de tempo '{time_col}' não existe no DataFrame.")
        safra = pd.to_datetime(sub[time_col], errors="coerce").dt.to_period("M")
        rows = []
        for per, g in sub.groupby(safra):
            col = g[feature]
            x = col.to_numpy(dtype="float64"); x = x[~np.isnan(x)]
            n = int(len(col)); n_miss = int(col.isna().sum())
            row = {"safra": str(per), "n": n,
                   "pct_missing": round(100 * n_miss / n, 1) if n else float("nan")}
            if x.size:
                row.update({"min": round(float(np.min(x)), 3),
                            "p5": round(float(np.percentile(x, 5)), 3),
                            "media": round(float(np.mean(x)), 3),
                            "p95": round(float(np.percentile(x, 95)), 3),
                            "max": round(float(np.max(x)), 3)})
            else:
                row.update({k: float("nan") for k in ("min", "p5", "media", "p95", "max")})
            rows.append(row)
        return pd.DataFrame(rows).sort_values("safra").reset_index(drop=True)

    def variable_psi_by_safra(self, feature, time_col=None, sid=None, max_n_bins=10,
                              min_bin_size=0.05, eps=1e-6) -> pd.DataFrame:
        """PSI da variável por safra vs. a referência (DES), com os bins
        fixados na DES da folha. `time_col` default = date_col."""
        if self.sample_col is None:
            raise ValueError("PSI por safra requer sample_col (referência DES).")
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col no segmenter.")
        mask = self._leaf_mask(sid)
        leaf = self.df[mask]
        if time_col not in leaf.columns:
            raise ValueError(f"Coluna de tempo '{time_col}' não existe no DataFrame.")
        ref = leaf[leaf[self.sample_col] == self.ref_sample]
        if len(ref) == 0:
            raise ValueError("Sem dados da referência (DES) nesta folha.")
        bins, _modo, _kind = self._resolve_bins_cached(sid, feature, None, None,
                                                       max_n_bins, min_bin_size)
        if not bins:
            return pd.DataFrame(columns=["safra", "n", "psi", "classificacao"])
        n_ref = len(ref)
        ref_pct = [max(int(self._mask_in(ref, feature, b).sum()) / n_ref, eps) for b in bins]
        safra = pd.to_datetime(leaf[time_col], errors="coerce").dt.to_period("M")
        rows = []
        for per, g in leaf.groupby(safra):
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

    def plot_variable_distribution(self, feature, sid=None, sample=None, max_n_bins=8,
                                   min_bin_size=0.03, figsize=(7.2, 3.1),
                                   save_path=None, dpi=150, ax=None):
        """Distribuição da variável na folha (barras por faixa; a barra de
        faltantes aparece em destaque). Numérica usa os bins ótimos."""
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_variable_distribution requer matplotlib.") from e
        mask = self._leaf_mask(sid)
        if sample is None and self.sample_col is not None:
            sample = self.ref_sample
        if sample is not None and self.sample_col is not None:
            mask = mask & (self.df[self.sample_col] == sample)
        sub = self.df[mask]
        rot = self.feature_labels.get(feature, feature)
        fig, ax = self._new_ax(figsize, dpi, ax)
        try:
            bins, _modo, _kind = self._resolve_bins(sub, feature, None, None,
                                                    max_n_bins, min_bin_size)
        except Exception:
            bins = []
        if bins:
            tbl = self._bin_table(sub, feature, bins, max(len(sub), 1))
        else:
            tbl = pd.DataFrame()
        if tbl.empty:                         # fallback quando não há bins válidos
            # categórica com 1 categoria na folha, ou folha pura (clf) ⇒ bins=[]. NÃO
            # coagir para float (quebra em coluna de strings); ramificar por tipo.
            if self._detect_kind(sub, feature, None) == "num":
                x = sub[feature].to_numpy(dtype="float64"); x = x[~np.isnan(x)]
                if x.size:
                    ax.hist(x, bins=20, color="steelblue", alpha=0.85, edgecolor="#2f5d82")
            else:                             # categórica: barras por categoria (% da folha)
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
                    ax.text(0.5, 0.5, "sem dados por faixa", ha="center", va="center",
                            transform=ax.transAxes, color="#889")
        else:
            labels = [s.split(": ", 1)[-1] for s in tbl["faixa"]]
            reprs = tbl["repr_%"].to_numpy()
            cols = ["#c98a8a" if "faltante" in f else "steelblue" for f in tbl["faixa"]]
            xs = list(range(len(labels)))
            ax.bar(xs, reprs, color=cols, edgecolor="#2f5d82", alpha=0.9, width=0.72)
            for x0, rp in zip(xs, reprs):
                ax.text(x0, rp, f"{rp:.0f}%", ha="center", va="bottom", fontsize=7.5,
                        color="#15324a")
            ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
            ax.set_xlim(-0.75, len(labels) - 0.25)               # respiro nas bordas (eixo x)
            ax.set_ylim(0, float(np.nanmax(reprs)) * 1.16 + 1)   # espaço p/ os rótulos %
        ax.set_ylabel("% da folha")
        ax.set_title(f"Distribuição de '{rot}'" + (f" · {sample}" if sample else ""),
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_variable_logodds(self, feature, sid=None, sample=None, max_n_bins=6,
                              min_bin_size=0.05, figsize=(7.6, 3.4), dpi=150,
                              save_path=None, ax=None):
        """Barras de representatividade (%) + linha de **logodds/WoE** (classificação)
        ou **alvo médio** (regressão) por faixa — leitura de monotonicidade, na folha.
        Espelha :meth:`ModelSegmenter.plot_variable_logodds`."""
        vt = self.variable_table(feature, sid=sid, sample=sample,
                                 max_n_bins=max_n_bins, min_bin_size=min_bin_size)
        fig, ax = self._new_ax(figsize, dpi, ax)
        if vt.empty:
            ax.text(0.5, 0.5, "sem faixas", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        rcol = vt.attrs["risco_label"]
        labels = vt["faixa"].tolist(); reprs = vt["repr_%"].to_numpy()
        xs = list(range(len(labels)))
        cols = ["#c98a8a" if "faltante" in f else "#9db8cf" for f in labels]
        ax.bar(xs, reprs, color=cols, edgecolor="#2f5d82", alpha=0.85, width=0.7)
        ax.set_ylabel("% da amostra"); ax.set_ylim(0, float(np.nanmax(reprs)) * 1.2 + 1)
        ax2 = ax.twinx()
        if self._is_clf and "logodds" in vt:
            yline = vt["logodds"].to_numpy(); ylabel = "logodds"
        else:
            yline = vt[rcol].to_numpy()
            ylabel = "event_rate" if self._is_clf else "alvo médio"
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

    def plot_variable_distribution_badrate(self, feature, sid=None, sample=None,
                                           max_n_bins=6, min_bin_size=0.05,
                                           figsize=(9.0, 3.8), dpi=150,
                                           save_path=None, ax=None):
        """Distribuição (% por faixa) + linha do risco por faixa (**% de maus** na
        classificação / **alvo médio** na regressão), faltantes destacados, na folha.
        Espelha :meth:`ModelSegmenter.plot_variable_distribution_badrate`."""
        vt = self.variable_table(feature, sid=sid, sample=sample,
                                 max_n_bins=max_n_bins, min_bin_size=min_bin_size)
        if ax is None and len(vt) > 8:
            figsize = (figsize[0], figsize[1] + 0.30 * (len(vt) - 8))
        fig, ax = self._new_ax(figsize, dpi, ax)
        if vt.empty:
            ax.text(0.5, 0.5, "sem faixas", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        rcol = vt.attrs["risco_label"]
        labels = vt["faixa"].tolist(); reprs = vt["repr_%"].to_numpy()
        xs = list(range(len(labels)))
        cols = ["#c98a8a" if "faltante" in f else "steelblue" for f in labels]
        ax.bar(xs, reprs, color=cols, edgecolor="#2f5d82", alpha=0.85, width=0.7,
               label="% da amostra")
        for x0, rp in zip(xs, reprs):
            ax.text(x0, rp, f"{rp:.0f}%", ha="center", va="bottom", fontsize=7.5,
                    color="#15324a")
        ax.set_ylabel("% da amostra"); ax.set_ylim(0, float(np.nanmax(reprs)) * 1.2 + 1)
        is_clf = self._is_clf
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

    def variable_inversion(self, feature, sid=None, time_col=None, sample=None,
                           max_n_bins=6, min_bin_size=0.05, min_n=20) -> dict:
        """Inversão da ordem de risco das FAIXAS de uma variável, entre amostras e
        safras, DENTRO de uma folha — o estudo de folhas-irmãs aplicado às faixas de
        UMA variável. Veredito verde/amarelo/vermelho. Espelha
        :meth:`ModelSegmenter.variable_inversion`."""
        mask0 = self._leaf_mask(sid)
        ref_mask = (mask0 & (self.df[self.sample_col] == self.ref_sample)
                    if self.sample_col is not None else mask0)
        ref = self.df[ref_mask]
        bins, _modo, _kind = self._resolve_bins(ref, feature, None, None,
                                                max_n_bins, min_bin_size)
        empty = {"status": "green", "samples": [], "safras": [], "ordered": [],
                 "labels": [], "series": None, "sample_inv": 0, "n_safras": 0,
                 "safras_inv": 0, "safra_rate": 0.0}
        if len(bins) < 2:
            return empty
        labels = [self._bin_label(feature, b) for b in bins]

        def _risco(frame):
            y = frame[self.target].to_numpy(dtype="float64")
            y = y[~np.isnan(y)]
            return float(np.mean(y)) if y.size else float("nan")

        ref_risco = [_risco(ref[self._mask_in(ref, feature, b)]) for b in bins]
        order = sorted(range(len(bins)),
                       key=lambda i: (np.inf if pd.isna(ref_risco[i]) else ref_risco[i]))
        # por amostra
        xs_s = self._ordered_samples_with_value() or [None]
        xs_s_lab = [str(a) if a is not None else "todos" for a in xs_s]
        ser_s = {i: [] for i in range(len(bins))}
        for a in xs_s:
            fa = (self.df[mask0 & (self.df[self.sample_col] == a)]
                  if a is not None else self.df[mask0])
            for i, b in enumerate(bins):
                ser_s[i].append(_risco(fa[self._mask_in(fa, feature, b)]))
        # por safra
        xs_t, ser_t = [], {i: [] for i in range(len(bins))}
        tcol = time_col or self.date_col
        if tcol is not None and tcol in self.df.columns:
            base = (self.df[mask0 & (self.df[self.sample_col] == sample)]
                    if (sample and self.sample_col is not None) else self.df[mask0])
            safra = pd.to_datetime(base[tcol], errors="coerce").dt.to_period("M")
            for per, g in base.groupby(safra):
                if len(g) < min_n:
                    continue
                xs_t.append(str(per))
                for i, b in enumerate(bins):
                    ser_t[i].append(_risco(g[self._mask_in(g, feature, b)]))
        series = {"ordered": order, "labels": labels, "ref_risco": ref_risco,
                  "xs_sample": xs_s_lab, "ser_sample": ser_s,
                  "xs_safra": xs_t, "ser_safra": ser_t}
        sample_rows = []
        for j, xlab in enumerate(xs_s_lab):
            vals = {i: ser_s[i][j] for i in range(len(bins))}
            n_inv, npp = _count_inversions(order, vals)
            sample_rows.append({"amostra": xlab, "n_inv": n_inv, "n_pares": npp})
        safra_rows = []
        for j, xlab in enumerate(xs_t):
            vals = {i: ser_t[i][j] for i in range(len(bins))}
            n_inv, npp = _count_inversions(order, vals)
            if npp == 0:
                continue
            safra_rows.append({"safra": xlab, "n_inv": n_inv, "n_pares": npp})
        sample_inv = sum(r["n_inv"] for r in sample_rows
                         if r["amostra"] != str(self.ref_sample))
        n_safras = len(safra_rows)
        safras_inv = sum(1 for r in safra_rows if r["n_inv"] > 0)
        safra_rate = (safras_inv / n_safras) if n_safras else 0.0
        status = ("red" if (sample_inv > 0 or safra_rate > 0.25)
                  else "yellow" if safras_inv > 0 else "green")
        return {"status": status, "samples": sample_rows, "safras": safra_rows,
                "ordered": order, "labels": labels, "ref_risco": ref_risco,
                "sample_inv": sample_inv, "n_safras": n_safras, "safras_inv": safras_inv,
                "safra_rate": safra_rate, "series": series}

    def plot_variable_inversion_by_sample(self, feature, sid=None, max_n_bins=6,
                                          min_bin_size=0.05, figsize=(7.6, 4.0),
                                          dpi=150, save_path=None, ax=None):
        """Risco de cada faixa por amostra; cruzamentos = inversão da ordem de risco.
        Espelha :meth:`ModelSegmenter.plot_variable_inversion_by_sample`."""
        import matplotlib.pyplot as plt
        inv = self.variable_inversion(feature, sid=sid, max_n_bins=max_n_bins,
                                      min_bin_size=min_bin_size)
        fig, ax = self._new_ax(figsize, dpi, ax)
        s = inv.get("series")
        if not s or not inv["ordered"]:
            ax.text(0.5, 0.5, "menos de 2 faixas", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        xs = s["xs_sample"]; x = list(range(len(xs)))
        cmap = plt.get_cmap("RdYlGn_r"); k = len(inv["ordered"])
        for rank, i in enumerate(inv["ordered"]):
            ax.plot(x, s["ser_sample"][i], marker="o", lw=1.9, ms=5.5,
                    color=cmap(rank / (k - 1) if k > 1 else 0.5),
                    markeredgecolor="#33424f", markeredgewidth=0.6, label=s["labels"][i])
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

    def plot_variable_inversion_by_safra(self, feature, sid=None, time_col=None,
                                         sample=None, max_n_bins=6, min_bin_size=0.05,
                                         min_n=20, figsize=(9.6, 4.0), dpi=150,
                                         save_path=None, ax=None):
        """Risco de cada faixa por safra; safras com inversão ficam sombreadas.
        Espelha :meth:`ModelSegmenter.plot_variable_inversion_by_safra`."""
        import matplotlib.pyplot as plt
        inv = self.variable_inversion(feature, sid=sid, time_col=time_col, sample=sample,
                                      max_n_bins=max_n_bins, min_bin_size=min_bin_size,
                                      min_n=min_n)
        fig, ax = self._new_ax(figsize, dpi, ax)
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
        cmap = plt.get_cmap("RdYlGn_r"); k = len(ordered)
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

    def _optbin_numeric_bins(self, feature, sid=None, sample=None, max_n_bins=5,
                             min_bin_size=0.05):
        """Faixas do OPTIMAL BINNING de uma variável NUMÉRICA na folha (amostra de
        referência por default), SEMPRE rodando o optbinning (ignora cortes manuais)."""
        if OptimalBinning is None:
            raise ImportError("optbinning não instalado. Rode: pip install optbinning")
        mask0 = self._leaf_mask(sid)
        m = (mask0 & (self.df[self.sample_col] == (sample or self.ref_sample))
             if self.sample_col is not None else mask0)
        fit = self.df[m]
        if self._detect_kind(fit, feature, None) != "num":
            return []
        x = fit[feature].to_numpy(dtype="float64")
        y = fit[self.target].to_numpy(dtype="float64")
        ok = ~np.isnan(y); x, y = x[ok], y[ok]
        x_obs = x[~np.isnan(x)]
        if len(y) < 4 or x_obs.size == 0 or np.unique(x_obs).size < 2:
            return []
        if self._is_clf:
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

    def plot_variable_optbin_cumshare_timeseries(self, feature, sid=None, time_col=None,
                                                 sample=None, max_n_bins=5, min_bin_size=0.05,
                                                 figsize=(11.5, 4.2), dpi=150,
                                                 save_path=None, ax=None):
        """Distribuição ACUMULADA das faixas do OPTIMAL BINNING (numéricas) ao longo do
        tempo, na folha — área empilhada das %s por safra, faixas FIXADAS na DES
        (yardstick estável). Espelha
        :meth:`ModelSegmenter.plot_variable_optbin_cumshare_timeseries`."""
        import matplotlib.colors as mcolors
        fig, ax = self._new_ax(figsize, dpi, ax)
        mask0 = self._leaf_mask(sid)
        ref_mask = (mask0 & (self.df[self.sample_col] == self.ref_sample)
                    if self.sample_col is not None else mask0)
        if self._detect_kind(self.df[ref_mask], feature, None) != "num":
            ax.text(0.5, 0.5, "apenas para variáveis numéricas", ha="center",
                    va="center", transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        tcol = time_col or self.date_col
        if tcol is None or tcol not in self.df.columns:
            ax.text(0.5, 0.5, "sem coluna de tempo", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        bins = self._optbin_numeric_bins(feature, sid=sid, max_n_bins=max_n_bins,
                                         min_bin_size=min_bin_size)
        if len(bins) < 2:
            ax.text(0.5, 0.5, "sem faixas do optbin", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        labels = [self._bin_label(feature, b) for b in bins]
        base_all = self.df[mask0]                 # distribuição sobre toda a folha
        safra = pd.to_datetime(base_all[tcol], errors="coerce").dt.to_period("M")
        xs, Y = [], [[] for _ in bins]
        for per, g in base_all.groupby(safra):
            n_g = len(g)
            if n_g == 0:
                continue
            xs.append(str(per))
            for i, b in enumerate(bins):
                Y[i].append(100 * int(self._mask_in(g, feature, b).sum()) / n_g)
        if not xs:
            ax.text(0.5, 0.5, "sem dados por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        x = list(range(len(xs)))
        cmapc = mcolors.LinearSegmentedColormap.from_list("sc", ["steelblue", "crimson"])
        n_real = sum(1 for b in bins if b.get("kind") != "na"); ri = 0
        cores = []
        for b in bins:
            if b.get("kind") == "na":
                cores.append("#c98a8a")
            else:
                cores.append(cmapc(ri / max(n_real - 1, 1))); ri += 1
        ax.stackplot(x, *Y, labels=labels, colors=cores, alpha=0.9,
                     edgecolor="white", linewidth=0.3)
        ax.set_ylim(0, 100); ax.margins(x=0); ax.set_xticks(x)
        ax.set_xticklabels(_fmt_safras(xs), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% acumulado da safra")
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.9, title="faixa (optbin)")
        ax.set_title(f"'{self.label(feature)}' — distribuição acumulada das faixas do "
                     "optimal binning ao longo do tempo", fontsize=11,
                     fontweight="bold", color="#15324a")
        ax.grid(alpha=0.12, axis="y")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def variable_share_by_safra(self, feature, time_col=None, sid=None, sample=None,
                                top=8) -> pd.DataFrame:
        """Representatividade (% por safra) de cada categoria de uma variável
        CATEGÓRICA ao longo do tempo. Linhas = safra; colunas = categorias (as
        `top` mais frequentes; o resto vira 'outras') + '(faltante)'."""
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col no segmenter.")
        mask = self._leaf_mask(sid)
        if sample is not None and self.sample_col is not None:
            mask = mask & (self.df[self.sample_col] == sample)
        sub = self.df[mask]
        if time_col not in sub.columns:
            raise ValueError(f"Coluna de tempo '{time_col}' não existe no DataFrame.")
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

    def plot_variable_share_timeseries(self, feature, time_col=None, sid=None, sample=None,
                                       figsize=(8.6, 3.6), save_path=None, dpi=150, ax=None):
        """Área empilhada da representatividade (%) de cada categoria por safra —
        mostra como a distribuição da variável categórica migra no tempo."""
        import matplotlib.colors as mcolors
        sh = self.variable_share_by_safra(feature, time_col, sid=sid, sample=sample)
        fig, ax = self._new_ax(figsize, dpi, ax)
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
                i = base.index(c)
                colors.append(cmap(i / max(len(base) - 1, 1)))
        ys = [sh[c].fillna(0).to_numpy() for c in cats]
        ax.stackplot(x, ys, labels=cats, colors=colors, alpha=0.92)
        ax.set_ylim(0, 100); ax.margins(x=0)
        ax.set_xticks(x)
        ax.set_xticklabels(_fmt_safras(sh["safra"]), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% da safra")
        # legenda à DIREITA (vertical) — não colide com as datas do eixo x
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.9, title="categoria")
        rot = self.feature_labels.get(feature, feature)
        ax.set_title(f"'{rot}' ao longo do tempo — representatividade por categoria",
                     fontsize=11, fontweight="bold", color="#15324a")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_variable_timeseries(self, feature, time_col=None, sid=None, sample=None,
                                 figsize=(12.0, 3.4), save_path=None, dpi=150, ax=None):
        """Variável NUMÉRICA: percentis por safra (min–max, p5–p95, média).
        Variável CATEGÓRICA: representatividade de cada categoria por safra."""
        if self._detect_kind(self.df[self._leaf_mask(sid)], feature, None) == "cat":
            return self.plot_variable_share_timeseries(
                feature, time_col, sid=sid, sample=sample, figsize=(figsize[0], 3.6),
                save_path=save_path, dpi=dpi, ax=ax)
        bs = self.variable_by_safra(feature, time_col, sid=sid, sample=sample)
        fig, ax = self._new_ax(figsize, dpi, ax)
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
        ax.plot(x, bs["media"], color="#15324a", lw=2.4, marker="o", markersize=4,
                label="média")
        ax.margins(x=0)                                   # sem respiro lateral (eixo x)
        ax.set_xticks(x)
        ax.set_xticklabels(_fmt_safras(bs["safra"]), rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8, ncol=3, framealpha=0.9, loc="upper left")
        rot = self.feature_labels.get(feature, feature)
        ax.set_title(f"'{rot}' ao longo do tempo — percentis por safra",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(alpha=0.12)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_variable_psi_by_safra(self, feature, time_col=None, sid=None,
                                   figsize=(9.6, 4.4), save_path=None, dpi=150, ax=None):
        """PSI da variável por safra vs DES (barras coloridas por faixa de PSI)."""
        ps = self.variable_psi_by_safra(feature, time_col, sid=sid)
        fig, ax = self._new_ax(figsize, dpi, ax)
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
        ax.set_xticks(x); ax.set_xticklabels(_fmt_safras(ps["safra"]), rotation=45, ha="right", fontsize=8)
        ax.set_xlim(-0.7, len(ps) - 0.3)                          # respiro nas bordas (eixo x)
        ax.set_ylim(0, float(np.nanmax(ps["psi"])) * 1.16 + 0.05)  # espaço p/ os rótulos
        ax.set_ylabel("PSI")
        rot = self.feature_labels.get(feature, feature)
        ax.set_title(f"PSI de '{rot}' por safra vs DES (bins fixados na DES)",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.12)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ==================================================================
    # FOLHAS-IRMÃS: comparação da alvo médio entre folhas de MESMO PAI, por
    #   amostra e por safra, com detecção de INVERSÃO de ordenação. Duas irmãs
    #   "invertem" quando a ordem da alvo médio observada numa amostra/safra
    #   contradiz a ordem de referência (alvo na DES) — sinal de instabilidade
    #   da segmentação (o ranking de risco não se sustenta no tempo/fora da
    #   amostra de desenvolvimento).
    # ==================================================================
    def _ordered_samples_with_value(self) -> list:
        """Amostras que têm alvo observada, com a referência (DES) à frente."""
        if self.sample_col is None:
            return []
        samples = list(self.df[self.sample_col].dropna().unique())
        com_pd = [a for a in samples
                  if self.df.loc[self.df[self.sample_col] == a, self.target].notna().any()]
        ref = [self.ref_sample] if self.ref_sample in com_pd else []
        return ref + [a for a in com_pd if a != self.ref_sample]

    def _sibling_meta(self, parent_sid, leaves=None):
        """Metadados das folhas-irmãs de `parent_sid`:
        (ordenadas_por_PD_DES, nota_por_sid, descrição_curta_por_sid,
        pd_ref_por_sid). A ORDEM DE REFERÊNCIA é a alvo médio na DES (asc.)."""
        lv = self.leaves()
        nota = dict(zip(lv["segmento"], lv["nota"]))
        if leaves is None:
            leaves = [sid for sid, s in self.segments.items()
                      if s["parent"] == parent_sid and s["is_leaf"]]
        pd_ref = {}
        for sid in leaves:
            v = self._leaf_target(sid)          # alvo na referência (DES), sem NaN
            pd_ref[sid] = float(v.mean()) if len(v) else float("nan")
        ordered = sorted(leaves, key=lambda c: (np.inf if pd.isna(pd_ref[c])
                                                else pd_ref[c], str(c)))
        desc = {sid: self._descrever(self.segments[sid]["conditions"][-1:])
                for sid in leaves}
        return ordered, nota, desc, pd_ref

    def _ordered_direct_children(self, parent_sid) -> list:
        """Todos os filhos DIRETOS de `parent_sid` na ordem do split, com a flag
        de folha. Numérico: por `lo` (ordem da variável); categórico/misto: pelo
        risco na referência (DES). O nó de faltantes (na) vai sempre por último.
        Retorna ``[(sid, is_leaf), ...]``."""
        kids = [c for c, s in self.segments.items() if s["parent"] == parent_sid]

        def _key(c):
            conds = self.segments[c]["conditions"]
            cond = conds[-1] if conds else {}
            kind = cond.get("kind")
            if kind == "na":
                return (2, 0.0)                       # faltantes por último
            if kind == "num":
                lo = cond.get("lo")
                return (0, lo if lo is not None else float("-inf"))
            v = self._node_value(c)                   # risco na referência
            return (1, v if not pd.isna(v) else float("inf"))

        kids.sort(key=_key)
        return [(c, self.segments[c]["is_leaf"]) for c in kids]

    def _sibling_runs(self, parent_sid) -> list:
        """Sequências MAXIMAIS de folhas terminais irmãs ADJACENTES (mesmo pai
        direto). Um filho NÃO-terminal (irmã subdividida) QUEBRA a adjacência:
        folhas terminais de lados opostos dele caem em runs distintas e não são
        comparadas entre si. Retorna ``[[sid, ...], ...]`` (cada run = folhas)."""
        runs, atual = [], []
        for sid, is_leaf in self._ordered_direct_children(parent_sid):
            if is_leaf:
                atual.append(sid)
            elif atual:                               # nó não-terminal encerra a run
                runs.append(atual)
                atual = []
        if atual:
            runs.append(atual)
        return runs

    def _adjacent_sibling_pairs(self, parent_sid) -> list:
        """Pares ``(a, b)`` de folhas-irmãs TERMINAIS ADJACENTES de ``parent_sid``:
        consecutivas dentro de uma MESMA run de :meth:`_sibling_runs` (nenhum nó
        expandido/intermediário entre elas) e ambas populadas — o nó de faltantes
        (na) não entra na comparação. Definição canônica de 'irmãs comparáveis/
        fundíveis'; usada pelo teste ``p (irmãs)`` e pela fusão automática para
        NÃO comparar folhas separadas por uma irmã que se expandiu."""
        pairs = []
        for run in self._sibling_runs(parent_sid):
            seq = [c for c in run
                   if self.segments[c]["conditions"]
                   and self.segments[c]["conditions"][-1]["kind"] != "na"]
            pairs.extend(zip(seq, seq[1:]))
        return pairs

    def _adjacent_sibling_neighbors(self, sid):
        """``(esq, dir)``: as folhas-irmãs terminais imediatamente adjacentes a
        ``sid`` (mesma run — sem nó expandido entre elas; ignora o nó de
        faltantes). Cada lado é o sid vizinho ou ``None``."""
        s = self.segments.get(sid)
        if s is None or not s["is_leaf"] or s["parent"] is None:
            return None, None
        for run in self._sibling_runs(s["parent"]):
            seq = [c for c in run
                   if self.segments[c]["conditions"]
                   and self.segments[c]["conditions"][-1]["kind"] != "na"]
            if sid in seq:
                i = seq.index(sid)
                return (seq[i - 1] if i > 0 else None,
                        seq[i + 1] if i + 1 < len(seq) else None)
        return None, None

    def sibling_leaf_groups(self, min_leaves: int = 2) -> list:
        """Grupos de folhas-irmãs TERMINAIS ADJACENTES (mesmo pai direto,
        ≥`min_leaves` folhas). Uma irmã subdividida (nó não-terminal) quebra a
        adjacência — folhas terminais em lados opostos dela NÃO são agrupadas.
        Um mesmo pai pode gerar mais de um grupo (uma run por trecho contíguo de
        folhas). Cada item: ``{'parent','leaves','notas','feature','label','key'}``
        — pronto para um seletor na UI. Ordenado pela menor nota do grupo."""
        pais = {s["parent"] for s in self.segments.values()
                if s["is_leaf"] and s["parent"] is not None}
        out = []
        for pai in pais:
            for run in self._sibling_runs(pai):
                if len(run) < min_leaves:
                    continue
                ordered, nota, _desc, _pd = self._sibling_meta(pai, run)
                feat = None
                for c in ordered:                    # variável do split (cond. não-na)
                    conds = self.segments[c]["conditions"]
                    if conds and conds[-1]["kind"] != "na":
                        feat = conds[-1]["feature"]
                        break
                rot = self.feature_labels.get(feat, feat) if feat else "?"
                notas = [nota.get(c) for c in ordered]
                ns = [n for n in notas if n is not None]
                faixa = f"{min(ns)}–{max(ns)}" if ns else "?"
                label = f"split '{rot}' · folhas {faixa} ({len(ordered)})"
                pai_desc = self._descrever(self.segments[pai]["conditions"])
                if self.segments[pai]["conditions"]:
                    label += f" · pai: {pai_desc}"
                out.append({"parent": pai, "leaves": ordered, "notas": notas,
                            "feature": feat, "label": label, "key": tuple(ordered)})
        out.sort(key=lambda g: min([n for n in g["notas"] if n is not None] or [1e9]))
        return out

    def _sibling_sample_series(self, parent_sid, leaves=None):
        """Série da alvo médio por AMOSTRA para cada folha-irmã.
        Retorna (ordered, nota, desc, xs, series) com series[sid] = [médias]."""
        ordered, nota, desc, _pd = self._sibling_meta(parent_sid, leaves)
        samples = self._ordered_samples_with_value()
        usar = samples if samples else [None]
        xs = samples if samples else ["todas"]
        series: dict = {}
        for sid in ordered:
            m = self.segments[sid]["mask"]
            vals = []
            for a in usar:
                mm = m & (self.df[self.sample_col] == a) if a is not None else m
                v = self.df.loc[mm, self.target].to_numpy(dtype="float64")
                v = v[~np.isnan(v)]
                vals.append(float(v.mean()) if v.size else float("nan"))
            series[sid] = vals
        return ordered, nota, desc, xs, series

    def _sibling_safra_series(self, parent_sid, time_col=None, sample=None,
                              leaves=None, min_n: int = 1):
        """Série da alvo médio por SAFRA (mês de `time_col`) para cada folha-irmã.
        `sample` restringe a uma amostra (None = todas). Safras com < `min_n`
        observações na folha viram NaN. Retorna (ordered, nota, desc, xs, series)."""
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col no segmenter.")
        if time_col not in self.df.columns:
            raise ValueError(f"Coluna de tempo '{time_col}' não existe no DataFrame.")
        ordered, nota, desc, _pd = self._sibling_meta(parent_sid, leaves)
        base = pd.Series(True, index=self.df.index)
        if sample is not None and self.sample_col is not None:
            base = base & (self.df[self.sample_col] == sample)
        per_all = pd.to_datetime(self.df[time_col], errors="coerce").dt.to_period("M")
        safras = sorted(per_all[base].dropna().unique())
        xs = [str(p) for p in safras]
        series: dict = {}
        for sid in ordered:
            m = self.segments[sid]["mask"] & base
            per = per_all[m]
            tgt = self.df.loc[m, self.target]
            mu = tgt.groupby(per).mean()
            cnt = tgt.groupby(per).count()
            vals = []
            for p in safras:
                n = int(cnt.get(p, 0))
                val = mu.get(p, float("nan"))
                vals.append(float(val) if (n >= min_n and not pd.isna(val))
                            else float("nan"))
            series[sid] = vals
        return ordered, nota, desc, xs, series

    def sibling_inversion_summary(self, parent_sid, time_col=None, sample=None,
                                  min_n: int = 20, leaves=None) -> dict:
        """Diagnóstico de inversão das folhas-irmãs de `parent_sid`. Compara a
        ordem de alvo de referência (DES) com a observada em cada AMOSTRA e em
        cada SAFRA. Retorna contagens por amostra/safra e um veredito
        (verde/amarelo/vermelho). `leaves` restringe a comparação a um trecho
        contíguo de folhas terminais adjacentes (uma run de `sibling_leaf_groups`);
        omitido, usa todas as folhas terminais diretas do pai."""
        ordered, nota, desc, pd_ref = self._sibling_meta(parent_sid, leaves)
        k = len(ordered)
        n_pairs = k * (k - 1) // 2

        _o, _n, _d, xs_s, ser_s = self._sibling_sample_series(parent_sid, leaves)
        sample_rows = []
        for j, xlab in enumerate(xs_s):
            vals = {sid: ser_s[sid][j] for sid in ordered}
            n_inv, npp = _count_inversions(ordered, vals)
            sample_rows.append({"amostra": xlab, "n_inv": n_inv, "n_pares": npp})

        safra_rows, safra_err = [], None
        try:
            _o, _n, _d, xs_t, ser_t = self._sibling_safra_series(
                parent_sid, time_col, sample, leaves, min_n=min_n)
            for j, xlab in enumerate(xs_t):
                vals = {sid: ser_t[sid][j] for sid in ordered}
                n_inv, npp = _count_inversions(ordered, vals)
                if npp == 0:
                    continue
                safra_rows.append({"safra": xlab, "n_inv": n_inv, "n_pares": npp})
        except Exception as e:                   # noqa: BLE001
            safra_err = f"{type(e).__name__}: {e}"

        sample_inv = sum(r["n_inv"] for r in sample_rows
                         if r["amostra"] != self.ref_sample)
        n_safras = len(safra_rows)
        safras_inv = sum(1 for r in safra_rows if r["n_inv"] > 0)
        safra_rate = (safras_inv / n_safras) if n_safras else 0.0
        if sample_inv > 0 or safra_rate > 0.25:
            status = "red"
        elif safras_inv > 0:
            status = "yellow"
        else:
            status = "green"
        return {"ordered": ordered, "nota": nota, "desc": desc, "pd_ref": pd_ref,
                "n_pairs": n_pairs, "samples": sample_rows, "safras": safra_rows,
                "sample_inv": sample_inv, "n_safras": n_safras,
                "safras_inv": safras_inv, "safra_rate": safra_rate,
                "status": status, "ref_sample": self.ref_sample,
                "safra_err": safra_err}

    def sibling_value_by_sample(self, parent_sid, leaves=None) -> pd.DataFrame:
        """Tabela tidy: alvo médio de cada folha-irmã por amostra."""
        ordered, nota, desc, xs, series = self._sibling_sample_series(parent_sid, leaves)
        rows = []
        for sid in ordered:
            row = {"segmento": sid, "nota": nota.get(sid), "descricao": desc[sid]}
            for x, v in zip(xs, series[sid]):
                row[x] = round(v, 4) if not pd.isna(v) else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def sibling_value_by_safra(self, parent_sid, time_col=None, sample=None,
                            leaves=None, min_n: int = 1) -> pd.DataFrame:
        """Tabela tidy: alvo médio de cada folha-irmã por safra (linhas = safra)."""
        ordered, nota, desc, xs, series = self._sibling_safra_series(
            parent_sid, time_col, sample, leaves, min_n=min_n)
        data = {"safra": xs}
        for sid in ordered:
            col = f"folha {nota.get(sid)}"
            data[col] = [round(v, 4) if not pd.isna(v) else np.nan
                         for v in series[sid]]
        return pd.DataFrame(data)

    def _sibling_colors(self, ordered):
        """Cor por folha (verde→vermelho na ordem de alvo) p/ os gráficos."""
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("RdYlGn_r")
        k = len(ordered)
        return {sid: cmap(i / (k - 1) if k > 1 else 0.5)
                for i, sid in enumerate(ordered)}

    def plot_sibling_value_by_sample(self, parent_sid, leaves=None,
                                  figsize=(7.6, 4.0), dpi=150,
                                  save_path=None, ax=None, title=None):
        """Linhas da alvo médio das folhas-irmãs por amostra (DES, OOT, …). Onde
        as linhas se cruzam há INVERSÃO da ordem de risco entre as irmãs."""
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_sibling_value_by_sample requer matplotlib.") from e
        ordered, nota, desc, xs, series = self._sibling_sample_series(parent_sid, leaves)
        fig, ax = self._new_ax(figsize, dpi, ax)
        if not ordered or not xs:
            ax.text(0.5, 0.5, "sem folhas-irmãs", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        cores = self._sibling_colors(ordered)
        x = list(range(len(xs)))
        for sid in ordered:
            ax.plot(x, series[sid], marker="o", lw=1.9, ms=5.5,
                    color=cores[sid], markeredgecolor="#33424f", markeredgewidth=0.6,
                    label=f"folha {nota.get(sid)}")
        ax.set_xticks(x); ax.set_xticklabels(xs, fontsize=9)
        ax.set_xlim(-0.25, len(xs) - 0.75 + 0.5)
        ax.set_ylabel(self._risk_mean); ax.set_xlabel("amostra")
        ax.set_title(title or f"{self._risk_mean} das folhas-irmãs por amostra",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        # headroom no topo p/ a legenda FIXA (canto superior esquerdo) não cobrir linhas
        _y0, _y1 = ax.get_ylim(); ax.set_ylim(_y0, _y1 + (_y1 - _y0) * 0.22)
        ax.legend(fontsize=8, ncol=max(1, min(len(ordered), 4)),
                  loc="upper left", framealpha=0.9)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_sibling_value_by_safra(self, parent_sid, time_col=None, sample=None,
                                 leaves=None, min_n: int = 1,
                                 figsize=(9.6, 4.0), dpi=150,
                                 save_path=None, ax=None, title=None):
        """Linhas da alvo médio das folhas-irmãs por safra ao longo do tempo. As
        safras em que a ordem de risco inverte (vs. DES) ficam sombreadas em
        vermelho — leitura rápida de instabilidade temporal."""
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_sibling_value_by_safra requer matplotlib.") from e
        ordered, nota, desc, xs, series = self._sibling_safra_series(
            parent_sid, time_col, sample, leaves, min_n=min_n)
        fig, ax = self._new_ax(figsize, dpi, ax)
        if not ordered or not xs:
            ax.text(0.5, 0.5, "sem dados por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        cores = self._sibling_colors(ordered)
        x = list(range(len(xs)))
        # sombreia as safras com inversão (≥1 par fora de ordem vs. referência)
        for j in x:
            vals = {sid: series[sid][j] for sid in ordered}
            n_inv, npp = _count_inversions(ordered, vals)
            if npp and n_inv:
                ax.axvspan(j - 0.5, j + 0.5, color="#d6453e", alpha=0.08, lw=0)
        for sid in ordered:
            ax.plot(x, series[sid], marker="o", lw=1.7, ms=4.5,
                    color=cores[sid], markeredgecolor="#33424f", markeredgewidth=0.5,
                    label=f"folha {nota.get(sid)}")
        ax.set_xticks(x); ax.set_xticklabels(_fmt_safras(xs), rotation=45, ha="right", fontsize=8)
        ax.set_xlim(-0.7, len(xs) - 0.3)
        ax.set_ylabel(self._risk_mean); ax.set_xlabel("safra")
        sfx = f" · {sample}" if sample else " · todas as amostras"
        ax.set_title(title or (f"{self._risk_mean} das folhas-irmãs por safra{sfx}"
                     "  ·  faixas vermelhas = inversão"),
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        # headroom no topo p/ a legenda FIXA (canto superior esquerdo) não cobrir linhas
        _y0, _y1 = ax.get_ylim(); ax.set_ylim(_y0, _y1 + (_y1 - _y0) * 0.22)
        ax.legend(fontsize=8, ncol=max(1, min(len(ordered), 4)),
                  loc="upper left", framealpha=0.9)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # DIAGNÓSTICO POR AMOSTRA/SAFRA: métricas por amostra, PSI da segmentação
    #   ao longo do tempo e concentração das folhas entre amostras — análogos
    #   aos gráficos do ModelSegmenter (barras).
    # ------------------------------------------------------------------
    def _sample_bar_color(self, a) -> str:
        """Cor da barra por amostra: referência = azul · estabilidade = teal ·
        demais = vermelho."""
        if a == self.ref_sample:
            return "#4c78a8"
        if str(a).upper().startswith("ESTAB"):
            return "#2a9d8f"
        return "#c0392b"

    def psi_by_safra(self, time_col=None, eps: float = 1e-6) -> pd.DataFrame:
        """PSI da **segmentação** (folhas como bins) por safra vs a referência
        (DES): a distribuição das linhas entre as folhas em cada mês comparada à
        distribuição na DES. Colunas: ``safra, n, psi, classificacao``. Requer
        ``sample_col`` e uma coluna de tempo (``time_col``/``date_col``)."""
        if self.sample_col is None:
            raise ValueError("PSI por safra requer sample_col (referência DES).")
        time_col = time_col or self.date_col
        if time_col is None or time_col not in self.df.columns:
            raise ValueError("Informe time_col ou configure date_col.")
        leaf_ids = [sid for sid, s in self.segments.items() if s["is_leaf"]]
        if not leaf_ids:
            raise ValueError("Nenhuma folha — cresça a segmentação antes do PSI.")
        leaf_id = self._leaf_id_series(leaf_ids)
        ref_mask = self._sample_masks.get(self.ref_sample)
        ref_series = leaf_id[ref_mask] if ref_mask is not None else leaf_id
        n_ref = float(len(ref_series))
        vc_ref = ref_series.value_counts()
        ref_pct = {sid: max((float(vc_ref.get(sid, 0.0)) / n_ref) if n_ref else 0.0, eps)
                   for sid in leaf_ids}
        safra = pd.to_datetime(self.df[time_col], errors="coerce").dt.to_period("M")
        rows = []
        for per, g in leaf_id.groupby(safra):
            if pd.isna(per):
                continue
            n_g = int(len(g)); vc = g.value_counts()
            psi = 0.0
            for sid in leaf_ids:
                p_cur = max((float(vc.get(sid, 0.0)) / n_g) if n_g else 0.0, eps)
                psi += (p_cur - ref_pct[sid]) * np.log(p_cur / ref_pct[sid])
            rows.append({"safra": str(per), "n": n_g, "psi": round(psi, 4),
                         "classificacao": _classifica_psi(psi)})
        return (pd.DataFrame(rows, columns=["safra", "n", "psi", "classificacao"])
                .sort_values("safra").reset_index(drop=True))

    def plot_psi_by_safra(self, time_col=None, figsize=(8.4, 4.0), dpi=150,
                          save_path=None, ax=None):
        """Gráfico de **barras** do PSI da segmentação ao longo do tempo (vs DES).
        Verde &lt; 0.10 (estável) · amarelo &lt; 0.25 (atenção) · vermelho ≥ 0.25
        (instável); linhas-guia em 0.10 e 0.25."""
        import matplotlib.pyplot as plt  # noqa: F401
        ps = self.psi_by_safra(time_col)
        fig, ax = self._new_ax(figsize, dpi, ax)
        if ps.empty:
            ax.text(0.5, 0.5, "sem dados por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        x = list(range(len(ps)))
        cor = ["#1aa64b" if p < 0.10 else "#caa000" if p < 0.25 else "#d6453e"
               for p in ps["psi"]]
        ax.bar(x, ps["psi"], color=cor, width=0.78, edgecolor="#33424f", linewidth=0.4)
        for xi, p in zip(x, ps["psi"]):
            ax.annotate(f"{p:.2f}", (xi, p), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7, color="#33424f")
        ax.axhline(0.10, color="#caa000", lw=1.0, ls="--", alpha=0.7)
        ax.axhline(0.25, color="#d6453e", lw=1.0, ls="--", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(_fmt_safras(ps["safra"]), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("PSI"); ax.set_xlabel("safra"); ax.set_ylim(bottom=0)
        ax.set_title(f"PSI da segmentação por safra vs {self.ref_sample}",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ---- perfil das variáveis (que entraram na árvore) por safra ----
    def _profile_feats(self, features=None) -> list:
        """Variáveis p/ o perfil por safra: as que ENTRARAM na árvore (splits),
        restritas às colunas presentes."""
        feats = list(features) if features is not None else list(self.regua_features())
        return [f for f in feats if f in self.df.columns]

    @staticmethod
    def _profile_grid(n, ncols, dpi):
        """Grade (fig, axes 2D, nrows, ncols) com ``ncols`` colunas para ``n``
        subplots (um por variável)."""
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
        saf = pd.to_datetime(self.df[time_col], errors="coerce").dt.to_period("M").astype(str)
        samp_by = (self.df.assign(_saf=saf)
                   .dropna(subset=[self.sample_col])
                   .groupby("_saf")[self.sample_col]
                   .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None))
        seq = [samp_by.get(str(p)) for p in safras]
        return [i for i in range(1, len(seq))
                if seq[i] is not None and seq[i - 1] is not None and seq[i] != seq[i - 1]]

    def plot_variables_missing_by_safra(self, time_col=None, features=None, ncols=3,
                                        dpi=150, save_path=None):
        """Grade (``ncols`` colunas) com a **% de missing por safra** de CADA variável
        que entrou na árvore — numéricas e categóricas — sobre toda a população. Um
        subplot por variável (eixo fixo 0–100%). Requer ``date_col``/``time_col``."""
        import matplotlib.pyplot as plt
        time_col = time_col or self.date_col
        if time_col is None or time_col not in self.df.columns:
            raise ValueError("Informe time_col ou configure date_col.")
        feats = self._profile_feats(features)
        if not feats:
            raise ValueError("Nenhuma variável entrou na árvore (crie ao menos um split).")
        safra = pd.to_datetime(self.df[time_col], errors="coerce").dt.to_period("M")
        pers = sorted(p for p in safra.dropna().unique())
        xs = _fmt_safras([str(p) for p in pers]); x = list(range(len(pers)))
        fig, axes, nrows, ncols = self._profile_grid(len(feats), ncols, dpi)
        for idx, f in enumerate(feats):
            ax = axes[idx // ncols][idx % ncols]
            col = self.df[f]
            ys = []
            for p in pers:
                m = (safra == p).to_numpy()
                nn = int(m.sum())
                ys.append(100.0 * int(col[m].isna().sum()) / nn if nn else np.nan)
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
        fig.suptitle("% de missing por safra — variáveis da árvore",
                     fontsize=12, fontweight="bold", color="#15324a")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        return fig

    def plot_variables_stats_by_safra(self, time_col=None, features=None, ncols=3,
                                      top_cat=6, dpi=150, save_path=None):
        """Grade (``ncols`` colunas) da **dispersão por safra** de cada variável que
        entrou na árvore: **numéricas** mostram **p5 · média · p95** (banda + linha da
        média); **categóricas** mostram a **proporção das categorias** ao longo do
        tempo (área empilhada, com legenda em CADA subplot). Faixas verticais
        pontilhadas marcam a troca de amostra. Requer ``date_col``/``time_col``."""
        import matplotlib.pyplot as plt
        time_col = time_col or self.date_col
        if time_col is None or time_col not in self.df.columns:
            raise ValueError("Informe time_col ou configure date_col.")
        feats = self._profile_feats(features)
        if not feats:
            raise ValueError("Nenhuma variável entrou na árvore (crie ao menos um split).")
        fig, axes, nrows, ncols = self._profile_grid(len(feats), ncols, dpi)
        num_legend_done = False              # legenda p5/média: no 1º subplot NUMÉRICO
        for idx, f in enumerate(feats):
            ax = axes[idx // ncols][idx % ncols]
            safras_sub = []
            if self._detect_kind(self.df, f, None) == "num":
                t = self.variable_by_safra(f, time_col=time_col)
                if len(t):
                    safras_sub = list(t["safra"])
                    xs = _fmt_safras(safras_sub); x = list(range(len(t)))
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
                    xs = []; x = []
            else:
                sh = self.variable_share_by_safra(f, time_col=time_col, top=top_cat)
                cats = [c for c in sh.columns if c != "safra"]
                safras_sub = list(sh["safra"]) if len(sh) else []
                xs = _fmt_safras(safras_sub); x = list(range(len(sh)))
                if cats:
                    cmap = plt.get_cmap("tab10")
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

    def plot_metrics_comparison(self, figsize=(6.6, 4.6), dpi=150, save_path=None, ax=None):
        """Compara as **principais métricas entre amostras** (DES vs OOT lado a lado)
        em barras agrupadas — uma métrica por grupo, uma barra por amostra, com os
        valores anotados e a **variação DES→OOT** (seta + %) no topo de cada grupo.
        clf: **KS · AUC · Gini** (eixo único, em %). reg: **RMSE · MAE** (erro, na
        escala do alvo, eixo esquerdo) e **R²** (eixo direito próprio, em %).
        Referência em steelblue, comparação em crimson (estabilidade em teal).
        Mesmo padrão visual do ModelSegmenter; valores negativos (Gini de OOT
        invertida, R² pior que a média) aparecem abaixo do zero, nunca somem."""
        import matplotlib.pyplot as plt  # noqa: F401
        from matplotlib.patches import Patch
        from matplotlib.ticker import PercentFormatter
        fig, ax = self._new_ax(figsize, dpi, ax)
        m = self.metrics()
        m = m.set_index("amostra") if "amostra" in getattr(m, "columns", []) else pd.DataFrame()
        samples = ([self.ref_sample] if self.ref_sample in m.index else []) \
            + [a for a in m.index if a != self.ref_sample]
        # plano: (coluna, rótulo, maior_é_melhor?, eixo_direito?, em_%?)
        if self._is_clf:
            plano = [("KS", "KS", True, False, True), ("AUC", "AUC", True, False, True),
                     ("Gini", "Gini", True, False, True)]
        else:
            plano = [("RMSE", "RMSE", False, False, False),
                     ("MAE", "MAE", False, False, False), ("R2", "R²", True, True, True)]
        plano = [p for p in plano if p[0] in m.columns]
        if not plano or not samples:
            ax.text(0.5, 0.5, "sem métricas para comparar", ha="center", va="center",
                    transform=ax.transAxes, color="#8891a0", fontsize=12)
            ax.axis("off"); fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            return fig
        cores = {a: self._sample_bar_color(a) for a in samples}
        w = 0.8 / max(len(samples), 1)
        ax2 = ax.twinx() if any(p[3] for p in plano) else None
        ref = self.ref_sample
        oot = next((a for a in samples if a != ref), None)

        def _val(x):
            try:
                v = float(x)
                return v if np.isfinite(v) else np.nan
            except (TypeError, ValueError):
                return np.nan

        left_vals, right_vals = [], []
        for gi, (col, lab, up, right, pct) in enumerate(plano):
            axis = ax2 if right else ax
            vg = {}
            for k, a in enumerate(samples):
                v = _val(m.loc[a, col]); xi = gi + k * w; vg[a] = v
                axis.bar(xi, v if np.isfinite(v) else 0.0, width=w, color=cores[a],
                         alpha=0.9, edgecolor="#33424f", linewidth=0.5)
                if np.isfinite(v):
                    (right_vals if right else left_vals).append(v)
                    axis.text(xi, v, f"{v * 100:.1f}%" if pct else f"{v:.3g}",
                              ha="center", va="bottom" if v >= 0 else "top",
                              fontsize=9, fontweight="bold", color="#15324a")
            dref, doot = vg.get(ref, np.nan), vg.get(oot, np.nan)
            if oot is not None and np.isfinite(dref) and np.isfinite(doot) and abs(dref) > 1e-9:
                delta = doot - dref; rel = 100.0 * delta / abs(dref)
                piora = (delta < 0) if up else (delta > 0)
                xc = gi + (len(samples) - 1) * w / 2
                axis.text(xc, 0.96, f"{'▼' if delta < 0 else '▲'} {abs(rel):.1f}%",
                          transform=axis.get_xaxis_transform(), ha="center", va="top",
                          fontsize=10, fontweight="bold",
                          color="#d6453e" if piora else "#1aa64b")

        # folga no topo (32%) p/ os rótulos de valor e a variação não colidirem
        lmax = max(left_vals + [0.0]); lmin = min(left_vals + [0.0])
        ax.set_ylim(lmin * 1.15 if lmin < 0 else 0.0, lmax * 1.32 if lmax > 0 else 1.0)
        ax.axhline(0, color="#33424f", lw=0.7, alpha=0.5)
        if self._is_clf:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.tick_params(axis="y", labelsize=10)
        if ax2 is not None:
            rmax = max(right_vals + [0.0]); rmin = min(right_vals + [0.0])
            ax2.set_ylim(rmin * 1.15 if rmin < 0 else 0.0, max(rmax * 1.32, 1.0))
            ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax2.tick_params(axis="y", labelsize=10)
            ax2.set_ylabel("R² (eixo direito)", fontsize=10.5, color="#15324a")
        ax.set_xticks(np.arange(len(plano)) + (len(samples) - 1) * w / 2)
        ax.set_xticklabels([f"{lab} {'↑' if up else '↓'}" for _c, lab, up, _r, _p in plano],
                           fontsize=11.5)
        ax.set_ylabel("erro (escala do alvo)" if not self._is_clf else "métrica (%)",
                      fontsize=10.5)
        ax.set_title("Principais métricas por amostra", fontsize=12.5,
                     fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        # legenda por amostra ABAIXO (não colide com a variação no topo nem com as barras)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.legend(handles=[Patch(color=cores[a], label=str(a)) for a in samples],
                   loc="lower center", ncol=min(len(samples), 3), fontsize=9.5,
                   framealpha=0.85, bbox_to_anchor=(0.5, 0.0))
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_leaf_concentration(self, figsize=(8.4, 4.2), dpi=150, save_path=None, ax=None):
        """Barras agrupadas da **concentração das folhas entre amostras**:
        representatividade (%) de cada folha DENTRO de cada amostra (DES, OOT, …).
        Análogo à distribuição dos ratings por amostra do ModelSegmenter."""
        import matplotlib.pyplot as plt  # noqa: F401
        fig, ax = self._new_ax(figsize, dpi, ax)
        if self.sample_col is None:
            ax.text(0.5, 0.5, "requer coluna de amostra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        lv = self.leaves(with_psi=True)
        if len(lv) == 0:
            ax.text(0.5, 0.5, "sem folhas", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        notas = lv["nota"].tolist()
        all_samples = [self.ref_sample] + [a for a in self.df[self.sample_col].dropna().unique()
                                           if a != self.ref_sample]
        cols = [(a, f"repr_{a}_%") for a in all_samples if f"repr_{a}_%" in lv.columns]
        x = np.arange(len(notas)); n_s = max(len(cols), 1); w = 0.8 / n_s
        for k, (a, col) in enumerate(cols):
            ax.bar(x + (k - (n_s - 1) / 2.0) * w, lv[col].to_numpy(dtype="float64"),
                   width=w, label=str(a), color=self._sample_bar_color(a),
                   edgecolor="#33424f", linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels([str(n) for n in notas])
        ax.set_ylabel("% da amostra"); ax.set_xlabel("folha")
        ax.set_ylim(0, 100)
        ax.set_title("Concentração das folhas entre amostras", fontsize=11,
                     fontweight="bold", color="#15324a")
        if cols:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9,
                      ncol=max(1, min(len(cols), 3)))
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # VARIABLE_IV: Information Value (WoE binário) de cada variável candidata em
    #   relação à folha `sid`, para indicar qual variável usar no próximo split.
    #   O alvo é binário, então o IV é o clássico do optimal binning:
    #   IV = Σ (dist_bons_i − dist_maus_i)·WoE_i, WoE_i = ln(dist_bons/dist_maus).
    #   Calculado na amostra de referência (DES), sobre os MESMOS bins ótimos do
    #   split. Quando há amostras, o PSI de cada variável é calculado sobre ESSES
    #   MESMOS bins (DES × cada amostra, dentro da folha): colunas
    #   ``psi_<amostra>``, ``pior_psi`` e ``psi_classificacao``.
    # ------------------------------------------------------------------
    def variable_iv(self, sid: str | None = None, features: list | None = None,
                    max_n_bins: int = 5, min_bin_size: float = 0.05,
                    cutoff="median", with_psi: bool = True) -> pd.DataFrame:
        if features is None:
            features = [c for c in self.df.columns if c not in self._nonfeature_cols()]

        if sid is None or sid == "root" or sid not in self.segments:
            leaf_mask = pd.Series(True, index=self.df.index)
        else:
            leaf_mask = self.segments[sid]["mask"]
        if self.sample_col is not None:
            ref_mask = self.df[self.sample_col] == self.ref_sample
            nonref = [a for a in self.df[self.sample_col].dropna().unique()
                      if a != self.ref_sample]
        else:
            ref_mask = pd.Series(True, index=self.df.index)
            nonref = []
        psi_on = with_psi and bool(nonref)

        leaf = self.df[leaf_mask]
        base = self.df[leaf_mask & ref_mask]            # porção DES da folha (referência)
        yb = base[self.target].to_numpy(dtype="float64")
        yb = yb[~np.isnan(yb)]
        if self._is_clf:
            ybin = yb.astype(int)
            tot_bad = int(ybin.sum())
            tot_good = int(ybin.size - tot_bad)
            # IV binário precisa das DUAS classes presentes na folha (DES)
            ok_base = ybin.size >= 4 and tot_bad > 0 and tot_good > 0
        else:
            tot_bad = tot_good = 0
            # IV contínuo exige variação do alvo na folha (DES)
            ok_base = yb.size >= 4 and np.unique(yb).size > 1
        mean_global = float(yb.mean()) if yb.size else np.nan
        n_base = int(yb.size)

        # frames e contagens por amostra DENTRO da folha (para o PSI nos bins do IV)
        ref_frame = cur_frames = None
        n_ref_leaf = 0
        if psi_on:
            ref_frame = self.df[leaf_mask & (self.df[self.sample_col] == self.ref_sample)]
            n_ref_leaf = len(ref_frame)
            cur_frames = {a: self.df[leaf_mask & (self.df[self.sample_col] == a)]
                          for a in nonref}

        rows = []
        for feat in features:
            iv, nb, kind = np.nan, 0, "—"
            psi_vals = {a: np.nan for a in nonref}
            if ok_base:
                try:
                    bins, _modo, kind = self._resolve_bins_cached(
                        sid, feat, None, None, max_n_bins, min_bin_size)
                    nb = len(bins)
                    if bins:
                        # IV por faixa: classificação = WoE binário (Siddiqi)
                        # Σ(dist_bons−dist_maus)·ln(dist_bons/dist_maus); regressão =
                        # IV contínuo Σ (n_i/N)·|média_bin − média_global|.
                        iv = 0.0
                        yb_feat = base[self.target].to_numpy(dtype="float64")
                        for b in bins:
                            m = self._mask_in(base, feat, b).to_numpy()
                            yi = yb_feat[m]
                            yi = yi[~np.isnan(yi)]
                            if yi.size == 0:
                                continue
                            if self._is_clf:
                                yi = yi.astype(int)
                                bad_i = int(yi.sum()); good_i = int(yi.size - bad_i)
                                dg = good_i / tot_good
                                db = bad_i / tot_bad
                                if dg > 0 and db > 0:
                                    iv += (dg - db) * np.log(dg / db)
                            else:
                                iv += (yi.size / n_base) * abs(float(yi.mean()) - mean_global)
                        # PSI sobre os MESMOS bins (DES × amostra, na folha)
                        if psi_on and n_ref_leaf > 0:
                            for a in nonref:
                                n_a = len(cur_frames[a])
                                if n_a == 0:
                                    continue
                                psi = 0.0
                                for b in bins:
                                    p_ref = max(int(self._mask_in(ref_frame, feat, b).sum())
                                                / n_ref_leaf, 1e-6)
                                    p_cur = max(int(self._mask_in(cur_frames[a], feat, b).sum())
                                                / n_a, 1e-6)
                                    psi += (p_cur - p_ref) * np.log(p_cur / p_ref)
                                psi_vals[a] = round(float(psi), 4)
                except Exception:
                    iv = np.nan
            row = {
                "variavel": feat, "tipo": kind, "n_bins": nb,
                "iv": round(float(iv), 4) if not (iv is None or np.isnan(iv)) else np.nan,
                "forca": _classifica_iv(iv, self.task_type),
            }
            if psi_on:
                for a in nonref:
                    row[f"psi_{a}"] = psi_vals[a]
                validos = [v for v in psi_vals.values() if not pd.isna(v)]
                pior = max(validos) if validos else np.nan
                row["pior_psi"] = round(float(pior), 4) if not pd.isna(pior) else np.nan
                row["psi_classificacao"] = _classifica_psi(pior) if not pd.isna(pior) else "—"
            rows.append(row)
        out = (pd.DataFrame(rows)
               .sort_values("iv", ascending=False, na_position="last")
               .reset_index(drop=True))
        out.attrs["valor_medio"] = round(mean_global, 4) if not np.isnan(mean_global) else np.nan
        out.attrs["cutoff"] = out.attrs["valor_medio"]      # compat (média do alvo na folha)
        return out

    # ------------------------------------------------------------------
    # BEST_BINNING: binning ótimo de UMA variável numa folha — nº de bins e os
    #   cortes (num) / grupos (cat) que o optbinning escolhe. Alimenta o botão
    #   "sugerir cortes" dos controles de split (folha × variável específica).
    # ------------------------------------------------------------------
    def best_binning(self, sid: str | None = None, feature: str | None = None,
                     max_n_bins: int = 8, min_bin_size: float = 0.05) -> dict:
        """Sugere o binning ótimo de ``feature`` na folha ``sid`` (raiz por padrão).

        Devolve ``{kind, n_bins, cuts, groups}``: ``n_bins`` = nº de faixas/grupos
        ideais (sem o bin de faltantes), ``cuts`` = cortes numéricos sugeridos
        (para preencher 'Cortes' no modo Manual) e ``groups`` = grupos de
        categorias (categórico). Útil para indicar o melhor 'máx. bins' e os
        cortes ao dividir a folha por aquela variável."""
        sid = sid if (sid in self.segments) else "root"
        sub = self.df[self._leaf_mask(sid)]
        bins, _modo, kind = self._resolve_bins_cached(
            sid, feature, None, None, max_n_bins, min_bin_size, relax_max=True)
        cuts, groups = [], []
        for b in bins:
            if b["kind"] == "num" and np.isfinite(b.get("hi", np.inf)):
                cuts.append(round(float(b["hi"]), 6))
            elif b["kind"] == "cat":
                groups.append([str(c) for c in b["cats"]])
        n_bins = sum(1 for b in bins if b["kind"] != "na")
        return {"kind": kind, "n_bins": int(n_bins), "cuts": sorted(cuts), "groups": groups}

    # ------------------------------------------------------------------
    # FEATURE_IMPORTANCE: importância das variáveis que ENTRARAM na árvore.
    #   Para cada nó interno, contribuição = (representatividade do nó) × (IV da
    #   variável do split, medido nesse nó). Soma por variável e normaliza.
    # ------------------------------------------------------------------
    def feature_importance(self, normalize: bool = True) -> pd.DataFrame:
        """Importância de cada variável usada nos splits da árvore.

        A importância de uma variável é a soma, sobre os nós internos divididos
        por ela, de ``representatividade_do_nó × IV_da_variável_no_nó`` (ganho de
        IV ponderado pela população). Só lista variáveis que entraram na árvore;
        ``normalize=True`` devolve em % do total."""
        n_total = len(self.df)
        filhos: dict = {}
        for sid, s in self.segments.items():
            filhos.setdefault(s["parent"], []).append(sid)
        contrib: dict = {}
        usos: dict = {}
        for sid, s in self.segments.items():
            if s["is_leaf"]:
                continue
            kids = filhos.get(sid, [])
            if not kids:
                continue
            conds = self.segments[kids[0]]["conditions"]
            if not conds:
                continue
            feat = conds[-1]["feature"]               # variável do split deste nó
            try:
                ivt = self.variable_iv(sid, features=[feat], with_psi=False)
                iv = float(ivt["iv"].iloc[0]) if len(ivt) else np.nan
            except Exception:
                iv = np.nan
            iv = 0.0 if (iv is None or np.isnan(iv)) else iv
            w = (int(s["mask"].sum()) / n_total) if n_total else 0.0
            contrib[feat] = contrib.get(feat, 0.0) + w * iv
            usos[feat] = usos.get(feat, 0) + 1
        rows = [{"variavel": f, "n_splits": usos[f],
                 "importancia": round(v, 6)} for f, v in contrib.items()]
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        total = out["importancia"].sum()
        if normalize and total > 0:
            out["importancia_%"] = (100 * out["importancia"] / total).round(2)
        out["variavel"] = out["variavel"].map(lambda v: self.feature_labels.get(v, v))
        return out.sort_values("importancia", ascending=False).reset_index(drop=True)

    def plot_importance_bar(self, figsize=None, save_path=None, dpi: int = 150, ax=None):
        """Barras horizontais da **importância relativa (%)** de cada variável da
        árvore — a importância de :meth:`feature_importance` normalizada para 100%
        do total, ordenada da mais para a menos importante (topo → base).

        Pensado para ficar ao lado da tabela de importância: mostra de relance
        quanto cada variável pesa **em relação a todas as outras**."""
        try:
            import matplotlib.pyplot as plt  # noqa: F401  (garante backend p/ savefig)
            from matplotlib.colors import LinearSegmentedColormap, Normalize
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_importance_bar requer matplotlib.") from e
        fi = self.feature_importance(normalize=True)
        if fi.empty:
            raise ValueError("A árvore ainda não tem splits — sem importância para o gráfico.")
        col = "importancia_%" if "importancia_%" in fi.columns else "importancia"
        names = fi["variavel"].tolist()
        vals = fi[col].to_numpy(dtype="float64")
        if figsize is None:
            figsize = (6.4, max(2.6, len(names) * 0.46))     # cresce com o nº de variáveis
        fig, ax = self._new_ax(figsize, dpi, ax)
        # degradê pela MAGNITUDE da importância: mais importante → steelblue,
        # menos importante → crimson (extremos = a menor e a maior da árvore).
        vmin, vmax = float(vals.min()), float(vals.max())
        norm = (Normalize(vmin, vmax) if (vmax - vmin) > 1e-9 else Normalize(vmin - 1.0, vmax))
        cmap_obj = LinearSegmentedColormap.from_list("imp_crimson_steelblue",
                                                     ["crimson", "steelblue"])
        y = list(range(len(names)))
        ax.barh(y, vals, color=[cmap_obj(norm(v)) for v in vals],
                edgecolor="#33424f", alpha=0.95, height=0.72)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()                                     # mais importante no topo
        is_pct = col == "importancia_%"
        for yi, v in zip(y, vals):
            ax.text(v, yi, (f" {v:.1f}%" if is_pct else f" {v:.4f}"),
                    va="center", ha="left", fontsize=8.5, fontweight="bold", color="#15324a")
        # sem eixo x: deixamos só a importância relativa rotulada em cada barra
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_xlim(0, (vmax * 1.16) if vmax > 1e-9 else 1.0)
        for _sp in ("top", "right", "bottom"):
            ax.spines[_sp].set_visible(False)
        ax.set_title("Importância relativa das variáveis na árvore", fontsize=12,
                     fontweight="bold", color="#15324a")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # SUGGEST_SPLITS: ranqueia as melhores variáveis para dividir uma folha,
    #   com nº de bins, PSI por amostra (OOT/ESTABILIDADE), se passa no teste de
    #   hipótese entre os bins, e o IV. Reusa variable_iv (IV + PSI nos bins).
    # ------------------------------------------------------------------
    def suggest_splits(self, sid: str | None = None, top: int = 3,
                       max_n_bins: int = 5, min_bin_size: float = 0.05,
                       alpha: float = 0.05) -> pd.DataFrame:
        """TOP-``top`` variáveis sugeridas para dividir a folha ``sid`` (raiz por
        padrão). Colunas: variável, nº de bins, IV, força, PSI por amostra,
        ``passa_teste`` (separação de risco entre os bins é significativa,
        p < ``alpha``) e ``p_valor`` do teste (qui-quadrado p/ classificação,
        Kruskal-Wallis p/ regressão)."""
        from scipy.stats import chi2_contingency, kruskal

        sid = sid if (sid in self.segments) else "root"
        iv = self.variable_iv(sid, max_n_bins=max_n_bins, min_bin_size=min_bin_size,
                              with_psi=True)
        iv = iv[iv["iv"].notna()].head(top).copy()
        leaf = self.df[self._leaf_mask(sid)]
        if self.sample_col is not None:
            leaf = leaf[leaf[self.sample_col] == self.ref_sample]

        pvals, passa = [], []
        # mapeia o rótulo amigável de volta ao nome real da coluna
        inv = {self.feature_labels.get(c, c): c
               for c in self.df.columns}
        for _, r in iv.iterrows():
            feat = inv.get(r["variavel"], r["variavel"])
            p = np.nan
            try:
                bins, _modo, kind = self._resolve_bins(
                    leaf, feat, None, None, max_n_bins, min_bin_size)
                grupos = []
                for b in bins:
                    y = leaf.loc[self._mask_in(leaf, feat, b), self.target].to_numpy(dtype="float64")
                    y = y[~np.isnan(y)]
                    if y.size:
                        grupos.append(y)
                if len(grupos) >= 2:
                    if self._is_clf:
                        tab = np.array([[int((g == 1).sum()), int((g == 0).sum())] for g in grupos])
                        if (tab.sum(0) > 0).all() and (tab.sum(1) > 0).all():
                            _, p, _, _ = chi2_contingency(tab)
                    else:
                        if all(np.unique(g).size > 1 for g in grupos) or len(grupos) > 1:
                            _, p = kruskal(*grupos)
            except Exception:
                p = np.nan
            pvals.append(round(float(p), 6) if p == p else np.nan)
            passa.append(bool(p == p and p < alpha))
        iv["p_valor"] = pvals
        iv["passa_teste"] = passa
        cols = ["variavel", "n_bins", "iv", "forca"]
        cols += [c for c in iv.columns if c.startswith("psi_")]
        cols += ["passa_teste", "p_valor"]
        return iv[[c for c in cols if c in iv.columns]].reset_index(drop=True)

    # ------------------------------------------------------------------
    # DIFF_TREES: compara DUAS árvores (versões) — migração de notas entre as
    #   segmentações, concordância e métricas lado a lado.
    # ------------------------------------------------------------------
    def diff_trees(self, other: "TreeSegmenter", df: pd.DataFrame | None = None) -> dict:
        """Compara esta árvore (A) com `other` (B) sobre o mesmo DataFrame.

        Devolve dict com: ``migracao`` (crosstab nota_A × nota_B — para onde as
        linhas migram), ``concordancia`` (fração com a MESMA nota), ``resumo``
        (nº de folhas + métricas-chave por amostra, A vs B vs Δ) e as tabelas de
        métricas completas ``metrics_a``/``metrics_b``."""
        if other.task_type != self.task_type:
            raise ValueError(
                f"Árvores de task_type diferentes: {self.task_type} vs {other.task_type}.")
        df = self.df if df is None else df
        na = self.predict(df)["nota"].astype("Int64")
        nb = other.predict(df)["nota"].astype("Int64")
        valid = na.notna() & nb.notna()
        migracao = pd.crosstab(na[valid], nb[valid], dropna=False)
        migracao.index.name, migracao.columns.name = "nota_A", "nota_B"
        concord = float((na[valid] == nb[valid]).mean()) if valid.any() else float("nan")

        ma, mb = self.metrics(), other.metrics()
        key = "AUC" if self._is_clf else "R2"
        rows = [{"métrica": "nº de folhas",
                 "árvore A": int(sum(s["is_leaf"] for s in self.segments.values())),
                 "árvore B": int(sum(s["is_leaf"] for s in other.segments.values()))}]
        rows[0]["Δ (B−A)"] = rows[0]["árvore B"] - rows[0]["árvore A"]
        for am in ma["amostra"]:
            va = ma.loc[ma["amostra"] == am, key]
            vb = mb.loc[mb["amostra"] == am, key]
            if len(va) and len(vb):
                a, b = float(va.iloc[0]), float(vb.iloc[0])
                rows.append({"métrica": f"{key} · {am}", "árvore A": round(a, 4),
                             "árvore B": round(b, 4), "Δ (B−A)": round(b - a, 4)})
        resumo = pd.DataFrame(rows)
        return {"migracao": migracao, "concordancia": concord, "resumo": resumo,
                "metrics_a": ma, "metrics_b": mb}

    # ------------------------------------------------------------------
    # ASSIGN: rotula cada linha com seu segmento-folha
    #   Por padrão adiciona também a nota de alvo (1..N) e a descrição
    #   por extenso. Colunas: <col>, <col>_nota, <col>_desc
    # ------------------------------------------------------------------
    def assign(
        self,
        col_name: str = "segmento",
        add_grade: bool = True,
        add_desc: bool = True,
        ascending: bool = True,
    ) -> pd.DataFrame:
        out = self.df.copy()
        out[col_name] = pd.Series(pd.NA, index=out.index, dtype="object")

        nota_map, desc_map = self._grade_map(ascending=ascending)
        if add_grade:
            out[f"{col_name}_nota"] = pd.Series(pd.NA, index=out.index, dtype="object")
        if add_desc:
            out[f"{col_name}_desc"] = pd.Series(pd.NA, index=out.index, dtype="object")

        for sid, seg in self.segments.items():
            if not seg["is_leaf"]:
                continue
            out.loc[seg["mask"], col_name] = sid
            if add_grade:
                out.loc[seg["mask"], f"{col_name}_nota"] = nota_map[sid]
            if add_desc:
                out.loc[seg["mask"], f"{col_name}_desc"] = desc_map[sid]

        if add_grade:
            out[f"{col_name}_nota"] = out[f"{col_name}_nota"].astype("Int64")
        return out

    # ==================================================================
    # PERSISTÊNCIA: salvar / carregar a ÁRVORE inteira em JSON
    #   Serializa a estrutura (segmentos + condições + metadados), não as
    #   máscaras (que dependem dos dados). Ao carregar, as máscaras são
    #   reconstruídas a partir das condições sobre o DataFrame fornecido — por
    #   isso `load`/`from_dict` exigem um `df`. Permite versionar a segmentação
    #   e reaplicá-la (mesmos dados ou novos) em qualquer máquina.
    # ==================================================================
    SCHEMA = "yggdrasil.credit_risk.tree/1"

    @staticmethod
    def _conditions_from_json(conditions):
        """Inverso de `_conditions_json`: restaura lo/hi ilimitados como ±inf."""
        out = []
        for c in conditions:
            if c["kind"] == "na":
                out.append({"feature": c["feature"], "kind": "na"})
            elif c["kind"] == "num":
                lo = float("-inf") if c.get("lo") is None else float(c["lo"])
                hi = float("inf") if c.get("hi") is None else float(c["hi"])
                d = {"feature": c["feature"], "kind": "num", "lo": lo, "hi": hi}
                if c.get("include_na"):
                    d["include_na"] = True
                out.append(d)
            else:
                d = {"feature": c["feature"], "kind": "cat",
                     "cats": [str(x) for x in c["cats"]]}
                if c.get("include_na"):
                    d["include_na"] = True
                out.append(d)
        return out

    def to_dict(self) -> dict:
        """Estrutura serializável da árvore (segmentos + metadados, sem máscaras)."""
        segs = {}
        for sid, s in self.segments.items():
            segs[sid] = {
                "label": s["label"],
                "depth": s["depth"],
                "is_leaf": s["is_leaf"],
                "parent": s["parent"],
                "path": list(s["path"]),
                "conditions": self._conditions_json(s["conditions"]),
            }
        return {
            "schema": self.SCHEMA,
            "meta": {
                "target": self.target,
                "task_type": self.task_type,
                "sample_col": self.sample_col,
                "ref_sample": self.ref_sample,
                "date_col": self.date_col,
                "min_leaf_rows": self.min_leaf_rows,
                "feature_labels": dict(self.feature_labels),
                "problem_label": self.problem_label,
            },
            "segments": segs,
        }

    def _conds_key(self, conds_json):
        """Assinatura hashável das condições serializadas (chave do cache de máscara)."""
        import json
        return json.dumps(conds_json, sort_keys=True, default=str)

    def _mask_from_conds(self, conds_json, conds_runtime):
        """Máscara das condições, reusando o cache (condições→máscara é função pura
        do df). Só recomputa via _match_conditions_pandas no 1º encontro."""
        key = self._conds_key(conds_json)
        m = self._mask_cache_by_conds.get(key)
        if m is None:
            m = _match_conditions_pandas(self.df, conds_runtime)
            if len(self._mask_cache_by_conds) > 4000:      # backstop de memória
                self._mask_cache_by_conds.clear()
            self._mask_cache_by_conds[key] = m
        return m

    def _prime_mask_cache(self):
        """Registra as máscaras VIVAS (por condições) no cache, para um restore
        seguinte (undo/redo) reusar as inalteradas em vez de recomputá-las."""
        cache = self._mask_cache_by_conds
        if len(cache) > 4000:
            cache.clear()
        for s in self.segments.values():
            cache.setdefault(self._conds_key(self._conditions_json(s["conditions"])),
                             s["mask"])

    def _load_segments(self, segs: dict):
        """Reconstrói self.segments a partir da forma serializada. Reusa máscaras
        já calculadas (cache por condições) — chave p/ undo/redo barato."""
        novo = {}
        for sid, s in segs.items():
            conds = self._conditions_from_json(s["conditions"])
            novo[sid] = {
                "mask": self._mask_from_conds(s["conditions"], conds),
                "label": s["label"],
                "depth": int(s["depth"]),
                "is_leaf": bool(s["is_leaf"]),
                "path": list(s.get("path", [])),
                "parent": s["parent"],
                "conditions": conds,
            }
        if "root" not in novo:
            raise ValueError("Árvore inválida: segmento 'root' ausente.")
        self.segments = novo
        self.history = []
        self._bump_version()      # árvore recarregada → invalida agregados memoizados
        return self

    def save(self, path: str) -> str:
        """Salva a árvore em um arquivo JSON. Devolve o caminho."""
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def from_dict(cls, data: dict, df: pd.DataFrame, verbose: bool = False):
        """Reconstrói um segmentador a partir de `to_dict()` + um DataFrame."""
        meta = data.get("meta", {})
        # date_col entrou no meta depois (JSONs antigos não têm a chave → None);
        # se o df fornecido não tiver a coluna, degrada para None em vez de
        # falhar no construtor — a coluna é só para os gráficos no tempo.
        date_col = meta.get("date_col")
        if date_col is not None and date_col not in df.columns:
            date_col = None
        seg = cls(df, target=meta.get("target", "target"),
                  task_type=meta.get("task_type", "classification"),
                  sample_col=meta.get("sample_col"),
                  ref_sample=meta.get("ref_sample", "DES"),
                  date_col=date_col,
                  feature_labels=meta.get("feature_labels"),
                  problem_label=meta.get("problem_label"),
                  min_leaf_rows=meta.get("min_leaf_rows", 50),
                  verbose=verbose)
        seg._load_segments(data["segments"])
        return seg

    @classmethod
    def load(cls, path: str, df: pd.DataFrame, verbose: bool = False):
        """Carrega uma árvore salva em JSON e a aplica ao DataFrame fornecido."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, df, verbose=verbose)

    # ==================================================================
    # RÉGUA / SCORING
    # ==================================================================
    @staticmethod
    def _conditions_json(conditions):
        """Converte condições para forma serializável (lo/hi=None se ilimitado)."""
        out = []
        for c in conditions:
            if c["kind"] == "na":
                out.append({"feature": c["feature"], "kind": "na"})
            elif c["kind"] == "num":
                d = {"feature": c["feature"], "kind": "num",
                     "lo": None if c["lo"] == float("-inf") else float(c["lo"]),
                     "hi": None if c["hi"] == float("inf") else float(c["hi"])}
                if c.get("include_na"):
                    d["include_na"] = True
                out.append(d)
            else:
                d = {"feature": c["feature"], "kind": "cat",
                     "cats": [str(x) for x in c["cats"]]}
                if c.get("include_na"):
                    d["include_na"] = True
                out.append(d)
        return out

    def _regua_dict(self) -> dict:
        """Régua final: folhas com nota, condições e alvo (na referência/DES)."""
        nota_map, _ = self._grade_map()
        leaves = []
        for sid, seg in self.segments.items():
            if not seg["is_leaf"]:
                continue
            sub = self.df[seg["mask"]]
            if self.sample_col is not None:
                ref = sub.loc[sub[self.sample_col] == self.ref_sample, self.target]
                pdv = float(ref.mean()) if len(ref) else float(sub[self.target].mean())
            else:
                pdv = float(sub[self.target].mean())
            if np.isnan(pdv):                       # folha vazia → alvo global (sem NaN na régua)
                if self.sample_col is not None:
                    g = self.df.loc[self.df[self.sample_col] == self.ref_sample,
                                    self.target].mean()
                else:
                    g = self.df[self.target].mean()
                pdv = float(g) if not pd.isna(g) else 0.0
            leaves.append({"id": sid, "nota": int(nota_map[sid]),
                           "conditions": self._conditions_json(seg["conditions"]),
                           "pd": round(pdv, 6)})
        leaves.sort(key=lambda x: x["nota"])
        return {"target": self.target, "ref_sample": self.ref_sample, "leaves": leaves}

    def predict(self, X: pd.DataFrame, col_seg="segmento",
                col_nota="nota", col_valor="valor_regua") -> pd.DataFrame:
        """Aplica a régua a um DataFrame pandas novo: segmento, nota e alvo."""
        return _aplicar_regua_pandas(self._regua_dict(), X, col_seg, col_nota, col_valor)

    # ------------------------------------------------------------------
    # TO_PYSPARK: gera o código da régua como F.when().otherwise() para
    #   aplicar a segmentação (segmento + nota + alvo) em escala no Spark.
    # ------------------------------------------------------------------
    def to_pyspark(self, func_name: str = "aplicar_regua") -> str:
        """Gera o código PySpark (string) que reproduz a régua (segmento, nota e alvo).

        Contrato de faltantes: as colunas numéricas devem usar **NULL** (não ``NaN``)
        para ausente. No Spark um ``NaN`` numérico é ordenado como o MAIOR valor e
        ``isNull()`` não o captura, então um ``NaN`` literal cairia no bin mais alto —
        divergindo do pandas. Se a tabela puder conter ``NaN`` literal, normalize antes
        (``F.when(F.isnan(col), None).otherwise(col)``) ou use :meth:`apply_spark`, que
        já trata ``NaN`` como faltante internamente."""
        regua = self._regua_dict()

        def cond_expr(conds):
            parts = []
            for c in conds:
                feat = c["feature"]
                if c["kind"] == "na":
                    parts.append(f'F.col("{feat}").isNull()')
                    continue
                if c["kind"] == "num":
                    sub = []
                    if c["lo"] is not None:
                        sub.append(f'(F.col("{feat}") > {c["lo"]})')
                    if c["hi"] is not None:
                        sub.append(f'(F.col("{feat}") <= {c["hi"]})')
                    expr = " & ".join(sub) if sub else "F.lit(True)"
                else:
                    cats = ", ".join(repr(x) for x in c["cats"])
                    # cast p/ string espelha o astype(str) do pandas/pyfunc
                    expr = f'F.col("{feat}").cast("string").isin({cats})'
                if c.get("include_na"):
                    expr = f'(({expr}) | F.col("{feat}").isNull())'
                parts.append(expr)
            return " & ".join(parts) if parts else "F.lit(True)"

        def chain(valfn):
            out = []
            for i, leaf in enumerate(regua["leaves"], 1):
                head = "  F.when" if i == 1 else "   .when"
                out.append(f'        {head}(c{i}, {valfn(leaf)})')
            out.append("           .otherwise(F.lit(None))")
            return "\n".join(out)

        L = ["from pyspark.sql import functions as F", "",
             f"def {func_name}(df, col_seg='segmento', col_nota='nota', col_valor='valor_regua'):",
             f'    """Régua de {self._risk_word} gerada por TreeSegmenter (segmento, nota e {self._risk_word} por folha)."""']
        for i, leaf in enumerate(regua["leaves"], 1):
            L.append(f'    c{i} = {cond_expr(leaf["conditions"])}')
        L.append("    seg = (")
        L.append(chain(lambda lf: f'F.lit({lf["id"]!r})'))
        L.append("    )")
        L.append("    nota = (")
        L.append(chain(lambda lf: f'F.lit({lf["nota"]})'))
        L.append("    )")
        L.append("    pd_val = (")
        L.append(chain(lambda lf: f'F.lit({lf["pd"]})'))
        L.append("    )")
        L.append("    return (df.withColumn(col_seg, seg)")
        L.append("              .withColumn(col_nota, nota)")
        L.append("              .withColumn(col_valor, pd_val))")
        return "\n".join(L)

    # ------------------------------------------------------------------
    # TO_SQL: régua como CASE WHEN (copiar e colar no SQL). Gera 3 colunas:
    #   segmento (id da folha), nota (1..N) e o valor previsto do alvo.
    # ------------------------------------------------------------------
    def to_sql(self, table: str = "minha_tabela", col_seg: str = "segmento",
               col_nota: str = "folha", col_valor: str = "valor_previsto") -> str:
        """Gera SQL ANSI com ``CASE WHEN`` que reproduz a régua. Pronto p/ copiar.

        ``table`` é o nome da tabela/CTE de origem. Cada folha vira um ramo do
        CASE (na ordem da nota); a condição final usa as MESMAS regras de
        pandas/Spark (faixas (lo, hi], ``IN`` para categóricas, ``IS NULL`` para
        faltantes, ``include_na`` quando o nó de faltantes foi fundido).

        Contrato de faltantes: colunas numéricas devem usar **NULL** (não ``NaN``)
        para ausente — no Spark um ``NaN`` é o MAIOR valor e ``IS NULL`` não o captura,
        então cairia no bin mais alto. Normalize ``NaN``→``NULL`` antes (ex.:
        ``nanvl``/``CASE WHEN isnan(x) THEN NULL``) ou use :meth:`apply_spark`."""
        regua = self._regua_dict()

        def _q(v):                       # literal de categoria com escape de aspas
            return "'" + str(v).replace("'", "''") + "'"

        def cond_sql(conds):
            parts = []
            for c in conds:
                feat = c["feature"]
                if c["kind"] == "na":
                    parts.append(f"{feat} IS NULL")
                    continue
                if c["kind"] == "num":
                    sub = []
                    if c.get("lo") is not None:
                        sub.append(f"{feat} > {c['lo']}")
                    if c.get("hi") is not None:
                        sub.append(f"{feat} <= {c['hi']}")
                    expr = " AND ".join(sub) if sub else "1=1"
                else:
                    cats = ", ".join(_q(x) for x in c["cats"])
                    # STRING (não VARCHAR): no Spark/Databricks `CAST(x AS VARCHAR)`
                    # sem tamanho falha ("VARCHAR requires a length parameter"); STRING
                    # é o tipo canônico e equivale a VARCHAR(n) sem exigir comprimento.
                    expr = f"CAST({feat} AS STRING) IN ({cats})"
                if len(conds) > 1 or c["kind"] != "na":
                    expr = f"({expr})"
                if c.get("include_na"):
                    expr = f"({expr} OR {feat} IS NULL)"
                parts.append(expr)
            return " AND ".join(parts) if parts else "1=1"

        def case(valfn, alias):
            linhas = [f"  CASE"]
            for leaf in regua["leaves"]:
                linhas.append(f"    WHEN {cond_sql(leaf['conditions'])} THEN {valfn(leaf)}")
            linhas.append(f"    ELSE NULL")
            linhas.append(f"  END AS {alias}")
            return "\n".join(linhas)

        L = [f"-- Régua de segmentação ({self.task_type}) gerada por TreeSegmenter",
             "SELECT",
             "  *,",
             case(lambda lf: _q(lf["id"]), col_seg) + ",",
             case(lambda lf: str(lf["nota"]), col_nota) + ",",
             case(lambda lf: str(lf["pd"]), col_valor),
             f"FROM {table};"]
        return "\n".join(L)

    # ------------------------------------------------------------------
    # REGUA_FEATURES: colunas usadas pela árvore (necessárias na tabela).
    # ------------------------------------------------------------------
    def regua_features(self) -> list:
        feats = []
        for leaf in self._regua_dict()["leaves"]:
            for c in leaf["conditions"]:
                if c["feature"] not in feats:
                    feats.append(c["feature"])
        return feats

    # ------------------------------------------------------------------
    # REPORT_PDF: relatório do modelo de árvore em PDF (capa + métricas +
    #   imagem da árvore + folhas + calibração + backtest). Usa matplotlib.
    # ------------------------------------------------------------------
    def report_pdf(self, path: str, time_col: str | None = None) -> str:
        """Gera um relatório PDF da árvore no caminho ``path`` e o devolve.

        Páginas: capa (parâmetros), métricas por amostra, imagem da árvore,
        tabela de folhas, calibração (se houver amostra) e backtest (se
        ``time_col``). Não exige dependência extra (usa matplotlib)."""
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        rl = self._risk_word
        n_folhas = sum(s["is_leaf"] for s in self.segments.values())
        prof = max(s["depth"] for s in self.segments.values())
        feats = self.regua_features()

        def _trunc(v, n=46):
            s = str(v)
            return s if len(s) <= n else s[:n - 1] + "…"

        def _table_fig(df, titulo, fs=8):
            df = df.copy()
            for c in df.columns:                 # arredonda numéricos, trunca textos
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
            fig.text(0.06, 0.9, f"Relatório — Segmentação de {rl}", fontsize=22,
                     fontweight="bold", color="#15324a")
            info = (f"task_type: {self.task_type}     ·     alvo: {self.target}\n"
                    f"amostra de referência: {self.ref_sample}\n"
                    f"folhas: {n_folhas}     ·     profundidade máxima: {prof}\n\n"
                    f"variáveis na árvore ({len(feats)}):\n{', '.join(feats) or '—'}")
            fig.text(0.06, 0.8, info, fontsize=12, color="#33424f", va="top")
            pdf.savefig(fig); plt.close(fig)
            for builder in (
                lambda: _table_fig(self.metrics(), "Métricas por amostra"),
                lambda: self.plot_tree(figsize=(11, 7)),
                lambda: _table_fig(self.leaves(with_psi=(self.sample_col is not None)), "Folhas"),
            ):
                try:
                    f = builder(); pdf.savefig(f); plt.close(f)
                except Exception:
                    plt.close("all")
            if self.sample_col is not None:
                try:
                    f = self.plot_calibration(); pdf.savefig(f); plt.close(f)
                except Exception:
                    plt.close("all")
            if time_col:
                try:
                    f = _table_fig(self.backtest(time_col), "Backtest por safra")
                    pdf.savefig(f); plt.close(f)
                except Exception:
                    plt.close("all")
        return path

    # ------------------------------------------------------------------
    # APPLY_SPARK: aplica a régua DIRETAMENTE num Spark DataFrame, devolvendo-o
    #   com as colunas de segmento, nota e alvo ("reconstrói as folhas" na
    #   tabela). Diferente de `to_pyspark` (que só gera o código), aqui a régua
    #   é executada. Exige pyspark e que as colunas usadas na árvore existam no
    #   `sdf` com o MESMO nome (senão levanta erro listando as que faltam).
    # ------------------------------------------------------------------
    def apply_spark(self, sdf, col_seg: str = "segmento",
                    col_nota: str = "nota", col_valor: str = "valor_regua"):
        try:
            from pyspark.sql import functions as F
        except ImportError as e:  # pragma: no cover
            raise ImportError("apply_spark requer pyspark — use: pip install pyspark") from e

        regua = self._regua_dict()
        if not regua["leaves"]:
            raise ValueError("Árvore sem folhas — cresça a segmentação antes de aplicar.")

        faltando = [f for f in self.regua_features() if f not in sdf.columns]
        if faltando:
            raise ValueError(
                f"Colunas ausentes no Spark DataFrame: {faltando}. A tabela precisa "
                f"ter as mesmas colunas usadas na árvore: {self.regua_features()}.")

        # Paridade com pandas: no Spark um NaN literal numérico é ordenado como o MAIOR
        # valor (NaN > lo = True) e isNull() NÃO o captura — então uma linha NaN cairia
        # no bin mais alto no Spark, enquanto no pandas (que usa .isna()) fica faltante.
        # Tratamos NaN como faltante (= NULL) nas colunas float, via uma expressão
        # derivada, SEM alterar as colunas do DataFrame de saída.
        float_cols = {name for name, t in sdf.dtypes if t in ("float", "double")}

        def col_of(feat):
            c = F.col(feat)
            return (F.when(F.isnan(c), F.lit(None)).otherwise(c)
                    if feat in float_cols else c)

        def cond_col(conds):
            expr = F.lit(True)
            for c in conds:
                feat = c["feature"]
                fc = col_of(feat)
                if c["kind"] == "na":
                    part = fc.isNull()
                elif c["kind"] == "num":
                    part = F.lit(True)
                    if c.get("lo") is not None:
                        part = part & (fc > c["lo"])
                    if c.get("hi") is not None:
                        part = part & (fc <= c["hi"])
                    if c.get("include_na"):
                        part = part | fc.isNull()
                else:
                    part = fc.cast("string").isin([str(x) for x in c["cats"]])
                    if c.get("include_na"):
                        part = part | fc.isNull()
                expr = expr & part
            return expr

        seg_col = nota_col = pd_col = None
        for leaf in regua["leaves"]:
            cond = cond_col(leaf["conditions"])
            if seg_col is None:
                seg_col = F.when(cond, F.lit(leaf["id"]))
                nota_col = F.when(cond, F.lit(leaf["nota"]))
                pd_col = F.when(cond, F.lit(leaf["pd"]))
            else:
                seg_col = seg_col.when(cond, F.lit(leaf["id"]))
                nota_col = nota_col.when(cond, F.lit(leaf["nota"]))
                pd_col = pd_col.when(cond, F.lit(leaf["pd"]))

        return (sdf.withColumn(col_seg, seg_col.otherwise(F.lit(None)))
                   .withColumn(col_nota, nota_col.otherwise(F.lit(None)))
                   .withColumn(col_valor, pd_col.otherwise(F.lit(None))))

    # ------------------------------------------------------------------
    # SUGGEST_SPLIT: recomenda a melhor variável para dividir uma folha,
    #   pelo maior Information Value, e devolve o ranking completo.
    # ------------------------------------------------------------------
    def suggest_split(self, sid: str | None = None, features: list | None = None,
                      max_n_bins: int = 4, min_bin_size: float = 0.05) -> dict:
        if sid is None:
            sid = "root"
        iv = self.variable_iv(sid, features=features, max_n_bins=max_n_bins,
                              min_bin_size=min_bin_size, with_psi=False)
        if len(iv) == 0 or pd.isna(iv.iloc[0]["iv"]):
            return {"sid": sid, "feature": None, "iv": None, "forca": None,
                    "ranking": iv, "msg": "nenhuma variável informativa para esta folha"}
        top = iv.iloc[0]
        return {"sid": sid, "feature": top["variavel"], "tipo": top["tipo"],
                "iv": float(top["iv"]), "forca": top["forca"], "ranking": iv,
                "msg": f"dividir por '{top['variavel']}' (IV={top['iv']:.4f}, {top['forca']})"}

    def _is_descendant_or_self(self, sid, ancestor) -> bool:
        """True se `sid` é o próprio `ancestor` ou um descendente dele."""
        cur, guard = sid, 0
        while cur is not None and cur in self.segments and guard < 10000:
            if cur == ancestor:
                return True
            cur = self.segments[cur].get("parent")
            guard += 1
        return False

    # ------------------------------------------------------------------
    # FIT_AUTO: constrói uma árvore de forma gulosa — em cada folha escolhe a
    #   variável de maior IV e divide com binning ótimo, até a profundidade
    #   máxima ou IV abaixo do mínimo. Ponto de partida para refinar à mão.
    #   Se `subtree` (id de folha) for dado, cresce APENAS a subárvore daquela
    #   folha (até `max_depth` níveis abaixo dela), sem reiniciar nem tocar no
    #   resto da árvore — útil para "auto-ajustar só esta folha".
    # ------------------------------------------------------------------
    def _concentration_bins(self, sub, feature, min_leaf_repr, max_bin_repr, dtype=None):
        """Bins de `feature` na folha `sub` que respeitam a concentração GLOBAL
        (fração da carteira inteira, todas as amostras) — usado pelo auto-fit.

        Uma restrição de concentração é de POPULAÇÃO, então partimos de cortes
        finos por QUANTIL GLOBAL (numérico) ou de uma categoria por grupo ordenada
        por alvo em DES (categórico) e fundimos os bins adjacentes pequenos —
        preferindo não estourar o máximo — até cada faixa/grupo ter ≥ `min_leaf_repr`
        da carteira. (O optbinning não serve aqui: devolve o nº ótimo por IV, não
        granularidade fina, e seu `min_bin_size` é relativo a DES, não global.)
        Devolve (bins, kind) ou (None, kind) se a variável não respeita o mín/máx
        (ex.: grupo de faltantes abaixo do mínimo, ou bin que não cabe entre mín/máx)."""
        N = len(self.df)
        if not N:
            return None, "num"
        kind = self._detect_kind(sub, feature, dtype)
        na_present = bool(sub[feature].isna().any())

        if kind == "num":
            x = sub[feature].dropna().to_numpy(dtype="float64")
            if x.size == 0 or np.unique(x).size < 2:
                return None, kind
            qs = np.quantile(x, np.linspace(0.0, 1.0, 25)[1:-1])   # ~24 faixas finas
            cuts = sorted({round(float(c), 10) for c in qs})
            edges = [-np.inf, *cuts, np.inf]
            core = [{"kind": "num", "lo": edges[i], "hi": edges[i + 1]}
                    for i in range(len(edges) - 1)]
        else:
            # uma categoria por grupo, ordenadas por alvo em DES (contiguidade de risco)
            ref = self._fit_frame(sub, 0.0).dropna(subset=[feature, self.target])
            means = (ref.groupby(ref[feature].astype(str))[self.target].mean()
                     .sort_values())
            present = [str(c) for c in pd.unique(sub[feature].dropna().astype(str))]
            ordered = [c for c in means.index if c in present]
            ordered += [c for c in present if c not in ordered]
            if len(ordered) < 2:
                return None, kind
            core = [{"kind": "cat", "cats": [c]} for c in ordered]

        def grepr(b):
            return int(self._mask_in(sub, feature, b).sum()) / N

        na = [{"kind": "na"}] if na_present else []
        if na and min_leaf_repr is not None and grepr(na[0]) < min_leaf_repr - 1e-9:
            return None, kind

        while len(core) > 1 and min_leaf_repr is not None:
            reprs = [grepr(b) for b in core]
            i = int(np.argmin(reprs))
            if reprs[i] >= min_leaf_repr - 1e-9:
                break
            cands = [k for k in (i - 1, i + 1) if 0 <= k < len(core)]
            ok = ([k for k in cands if reprs[i] + reprs[k] <= max_bin_repr + 1e-9]
                  if max_bin_repr is not None else cands)
            j = min(ok or cands, key=lambda k: reprs[k])
            a, b2 = sorted((i, j))
            if kind == "num":
                core[a] = {"kind": "num", "lo": core[a]["lo"], "hi": core[b2]["hi"]}
            else:
                core[a] = {"kind": "cat", "cats": core[a]["cats"] + core[b2]["cats"]}
            del core[b2]

        if len(core) < 2:
            return None, kind
        out = core + na
        if min_leaf_repr is not None and any(grepr(b) < min_leaf_repr - 1e-9 for b in out):
            return None, kind
        if max_bin_repr is not None and any(grepr(b) > max_bin_repr + 1e-9 for b in out):
            return None, kind
        return out, kind

    def _best_criterion_feature(self, sid, features, criterion, min_bin_size):
        """Feature com o melhor split BINÁRIO pelo `criterion` na folha `sid`
        (avaliado na referência/DES). Devolve (feature, score). Usado por fit_auto
        quando o usuário escolhe um critério (≠ optbin)."""
        leaf = self.df[self._leaf_mask(sid)]
        if self.sample_col is not None:
            leaf = leaf[leaf[self.sample_col] == self.ref_sample]
        if features is None:
            features = [c for c in self.df.columns if c not in self._nonfeature_cols()]
        y_all = leaf[self.target].to_numpy(dtype="float64")
        min_n = max(1, int(min_bin_size * len(leaf)))
        best_f, best_s = None, 0.0
        for feat in features:
            try:
                kind = self._detect_kind(leaf, feat, None)
                if kind == "num":
                    x = leaf[feat].to_numpy(dtype="float64")
                    ok = ~np.isnan(x) & ~np.isnan(y_all)
                    xx, yy = x[ok], y_all[ok]
                    yy = yy.astype(int) if self._is_clf else yy
                    t = _best_numeric_cut(xx, yy, criterion, self._is_clf, min_n)
                    if t is None:
                        continue
                    left = xx <= t
                else:
                    xs = leaf[feat].astype(str).to_numpy()
                    ok = leaf[feat].notna().to_numpy() & ~np.isnan(y_all)
                    xs, yy = xs[ok], y_all[ok]
                    yy = yy.astype(int) if self._is_clf else yy
                    par = _best_categorical_split(xs, yy, criterion, self._is_clf, min_n)
                    if par is None:
                        continue
                    lc = set(par[0])
                    left = np.array([c in lc for c in xs])
                s = _split_score(yy[left], yy[~left], criterion, self._is_clf)
                if np.isfinite(s) and s > best_s:
                    best_s, best_f = float(s), feat
            except Exception:
                continue
        return best_f, best_s

    def fit_auto(self, features: list | None = None, max_depth: int = 3,
                 min_iv: float = 0.02, max_n_bins: int = 2, min_bin_size: float = 0.05,
                 from_scratch: bool = True, subtree: str | None = None,
                 min_leaf_repr: float | None = None, max_bin_repr: float | None = None,
                 criterion: str = "optbin", verbose: bool = True):
        """Cresce a árvore de forma gulosa (maior IV por nível).

        ``min_leaf_repr`` e ``max_bin_repr`` são **representatividades GLOBAIS**
        (fração da carteira inteira), não da folha-mãe:

        * ``min_leaf_repr`` — cada folha terminal deve reter ao menos essa fração
          da carteira. Uma folha pequena demais para gerar dois filhos ≥
          ``min_leaf_repr`` não é dividida (vira terminal). Corrige o efeito de o
          ``min_bin_size`` do optbinning ser relativo à folha-mãe e, ao compor por
          nível, deixar folhas profundas com representatividade ínfima.
        * ``max_bin_repr`` — nenhuma quebra pode concentrar mais que essa fração da
          carteira (força granularidade em segmentos dominantes).

        Cada um é traduzido por folha para a fração local exigida pelo optbinning
        (``repr_global · N / n_folha``).
        """
        import io
        import math
        import contextlib
        self._bump_version()      # reconstrói a árvore → invalida agregados memoizados
        if subtree is not None and subtree not in self.segments:
            raise ValueError(f"Folha '{subtree}' não existe na árvore atual.")
        if subtree is None:                       # árvore inteira (comportamento padrão)
            if from_scratch:
                root = self.segments["root"]
                root["is_leaf"] = True
                self.segments = {"root": root}
            base_depth = 0
            def in_scope(_sid):
                return True
        else:                                     # só a subárvore da folha escolhida
            base_depth = self.segments[subtree]["depth"]
            def in_scope(_sid):
                return self._is_descendant_or_self(_sid, subtree)
        for depth in range(base_depth, base_depth + max_depth):
            atuais = [sid for sid, s in self.segments.items()
                      if s["is_leaf"] and s["depth"] == depth and in_scope(sid)]
            for sid in atuais:
                if sid in self.segments and not self.segments[sid]["is_leaf"]:
                    continue
                sub = self.df[self.segments[sid]["mask"]]
                n_leaf = len(sub)
                # ---- critério escolhido (≠ optbin): split BINÁRIO guloso pelo
                # melhor (feature, corte) por esse critério ----
                if criterion != "optbin":
                    if len(self._fit_frame(sub, min_bin_size)) < self.min_leaf_rows:
                        continue
                    feat, score = self._best_criterion_feature(
                        sid, features, criterion, min_bin_size)
                    if feat is not None and score > 0:
                        with contextlib.redirect_stdout(io.StringIO()):
                            self.grow(feat, only_segments=[sid], min_bin_size=min_bin_size,
                                      criterion=criterion)
                    continue
                # concentrações GLOBAIS → fração local (da folha) p/ o optbinning
                loc_min = min_bin_size
                if min_leaf_repr is not None and n_leaf:
                    loc_min = min_leaf_repr * len(self.df) / n_leaf
                    if loc_min >= 0.5:        # folha pequena p/ 2 filhos ≥ min_leaf_repr
                        continue              # → não divide (vira terminal)
                # a concentração MÁXIMA por quebra pode exigir mais bins do que o
                # split binário padrão. k bins cada ≤ loc_max precisa de k ≥ 1/loc_max;
                # com monotonicidade no alvo o optbinning fica INFEASIBLE no
                # limite exato, então pedimos 1 bin de folga (e limitamos a 6).
                loc_max, mnb = None, max_n_bins
                if max_bin_repr is not None and n_leaf:
                    loc_max = min(0.999, max_bin_repr * len(self.df) / n_leaf)
                    if loc_max <= loc_min:    # restrição de máx incompatível → ignora
                        loc_max = None
                    else:
                        mnb = max(max_n_bins, min(math.ceil(1.0 / loc_max) + 1, 6))
                        if loc_max * mnb < 1.0:   # inviável mesmo com 6 bins → ignora máx
                            loc_max, mnb = None, max_n_bins
                if len(self._fit_frame(sub, loc_min)) < self.min_leaf_rows:
                    continue
                iv = self.variable_iv(sid, features=features, max_n_bins=mnb,
                                      min_bin_size=loc_min, with_psi=False)
                if len(iv) == 0:
                    continue
                # Sem restrição de concentração: cresce a melhor variável por IV.
                if min_leaf_repr is None and max_bin_repr is None:
                    top = iv.iloc[0]
                    if pd.isna(top["iv"]) or top["iv"] < min_iv:
                        continue
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.grow(top["variavel"], only_segments=[sid], max_n_bins=mnb,
                                  min_bin_size=loc_min, max_bin_size=loc_max, relax_max=True)
                    continue
                # Com restrição: a concentração é medida GLOBALMENTE (todas as
                # amostras); ajusta os cortes à concentração global e cresce a 1ª
                # variável (por IV) que respeite o mín/máx por folha; senão terminal.
                for _, row in iv.iterrows():
                    if pd.isna(row["iv"]) or row["iv"] < min_iv:
                        break
                    var = row["variavel"]
                    try:
                        bins_fit, kind = self._concentration_bins(
                            sub, var, min_leaf_repr, max_bin_repr)
                    except Exception:
                        continue
                    if bins_fit is None:
                        continue
                    if kind == "num":
                        splits = sorted({b["hi"] for b in bins_fit
                                         if b["kind"] == "num" and np.isfinite(b["hi"])})
                    else:
                        splits = [b["cats"] for b in bins_fit if b["kind"] == "cat"]
                    if not splits:
                        continue
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.grow(var, splits=splits, only_segments=[sid])
                    break
        n_folhas = sum(s["is_leaf"] for s in self.segments.values())
        prof = max(s["depth"] for s in self.segments.values())
        if verbose:
            escopo = "subárvore" if subtree is not None else "árvore gulosa"
            print(f"[fit_auto] {escopo} construída: profundidade {prof} "
                  f"(máx +{max_depth}), IV mínimo {min_iv} → {n_folhas} folhas")
        return self

    # ------------------------------------------------------------------
    # LOG_TO_MLFLOW: salva a segmentação no MLflow — parâmetros, métricas,
    #   artefatos (folhas, árvore, régua PySpark, régua JSON) e o MODELO
    #   pyfunc (régua aplicável via .predict para scoring).
    # ------------------------------------------------------------------
    def log_to_mlflow(self, experiment: str | None = None,
                      run_name: str | None = None, registered_model_name: str | None = None,
                      artifact_path: str = "modelo", extra_params: dict | None = None,
                      registry_uri: str | None = None, save_base: bool = False,
                      verbose: bool = True) -> str:
        try:
            import mlflow
            import mlflow.pyfunc
            from mlflow.models import infer_signature
        except ImportError as e:  # pragma: no cover
            raise ImportError("mlflow não está instalado — use: pip install mlflow") from e
        import json
        import os
        import tempfile

        regua = self._regua_dict()
        feats = []
        for leaf in regua["leaves"]:
            for c in leaf["conditions"]:
                if c["feature"] not in feats:
                    feats.append(c["feature"])
        n_folhas = len(regua["leaves"])
        prof = max(s["depth"] for s in self.segments.values())
        params = {"target": self.target, "ref_sample": self.ref_sample,
                  "sample_col": self.sample_col, "n_folhas": n_folhas,
                  "profundidade": prof, "min_leaf_rows": self.min_leaf_rows,
                  "variaveis": ", ".join(feats) or "(nenhuma)"}
        if extra_params:
            params.update(extra_params)

        metrics = {"n_folhas": float(n_folhas), "profundidade": float(prof),
                   "n_variaveis": float(len(feats))}
        for leaf in regua["leaves"]:
            metrics[f"valor_nota_{leaf['nota']}"] = leaf["pd"]
        # métricas do modelo por amostra, conforme o tipo de problema
        # (classificação: KS/AUC/Gini/Acurácia/F1 · regressão: MAE/RMSE/R²)
        metric_keys = (("taxa_default", "KS", "AUC", "Gini", "Acuracia", "F1")
                       if self._is_clf else ("MAE", "RMSE", "R2"))
        try:
            for _, r in self.metrics().iterrows():
                a = r["amostra"]
                for k in metric_keys:
                    if k in r and not pd.isna(r[k]):
                        metrics[f"{k.lower()}_{a}"] = float(r[k])
        except Exception:
            pass
        # PSI da segmentação por amostra (cobre OOT, ESTABILIDADE, ...)
        if self.sample_col is not None:
            try:
                for _, r in self.psi().iterrows():
                    metrics[f"psi_{r['amostra']}"] = float(r["psi"])
            except Exception:
                pass

        class _ReguaModel(mlflow.pyfunc.PythonModel):
            def __init__(self, regua):
                self.regua = regua

            def predict(self, context, model_input):
                import numpy as _np
                import pandas as _pd
                df = model_input
                seg = _pd.Series(_pd.NA, index=df.index, dtype="object")
                nota = _pd.Series(_pd.NA, index=df.index, dtype="Int64")
                pdcol = _pd.Series(_np.nan, index=df.index, dtype="float64")
                for leaf in self.regua["leaves"]:
                    m = _pd.Series(True, index=df.index)
                    for c in leaf["conditions"]:
                        feat = c["feature"]
                        if c["kind"] == "na":
                            m &= df[feat].isna()
                            continue
                        if c["kind"] == "num":
                            sub = _pd.Series(True, index=df.index)
                            if c.get("lo") is not None:
                                sub &= df[feat] > c["lo"]
                            if c.get("hi") is not None:
                                sub &= df[feat] <= c["hi"]
                        else:
                            sub = df[feat].astype(str).isin([str(x) for x in c["cats"]])
                        if c.get("include_na"):
                            sub = sub | df[feat].isna()
                        m &= sub
                    seg[m] = leaf["id"]
                    nota[m] = leaf["nota"]
                    pdcol[m] = leaf["pd"]
                return _pd.DataFrame({"segmento": seg, "nota": nota, "valor_regua": pdcol},
                                     index=df.index)

        # Unity Catalog exige set_registry_uri('databricks-uc') e assinatura do modelo
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        if experiment:
            mlflow.set_experiment(experiment)

        model = _ReguaModel(regua)
        cols = feats if feats else [c for c in self.df.columns
                                    if c not in (self.target, self.sample_col)]
        input_example = self.df[cols].head(5).reset_index(drop=True).copy()
        try:
            output_example = model.predict(None, input_example)
            signature = infer_signature(input_example, output_example)
        except Exception:
            signature = None

        def _log_model():
            kw = dict(python_model=model, registered_model_name=registered_model_name,
                      signature=signature, input_example=input_example,
                      pip_requirements=["pandas", "numpy"])
            try:
                return mlflow.pyfunc.log_model(name=artifact_path, **kw)
            except TypeError:
                return mlflow.pyfunc.log_model(artifact_path=artifact_path, **kw)

        version = None
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            with tempfile.TemporaryDirectory() as d:
                self.leaves(with_psi=(self.sample_col is not None)).to_csv(
                    os.path.join(d, "folhas.csv"), index=False)
                with open(os.path.join(d, "arvore.txt"), "w", encoding="utf-8") as f:
                    f.write(self.tree())
                with open(os.path.join(d, "regua_pyspark.py"), "w", encoding="utf-8") as f:
                    f.write(self.to_pyspark())
                with open(os.path.join(d, "regua.sql"), "w", encoding="utf-8") as f:
                    f.write(self.to_sql())
                with open(os.path.join(d, "regua.json"), "w", encoding="utf-8") as f:
                    json.dump(regua, f, ensure_ascii=False, indent=2)
                # árvore completa (estrutura) para recarregar via .load(...)
                with open(os.path.join(d, "arvore.json"), "w", encoding="utf-8") as f:
                    json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
                # CSI por variável (estabilidade das entradas), se houver amostras
                if self.sample_col is not None:
                    try:
                        self.csi().to_csv(os.path.join(d, "csi_variaveis.csv"), index=False)
                    except Exception:
                        pass
                mlflow.log_artifacts(d, artifact_path="regua")
            model_info = _log_model()
            version = getattr(model_info, "registered_model_version", None)
            # relatório em abas (Resumo/Métricas/Estabilidade) + base opcional
            try:
                from .._mlflow_report import log_tabbed_report
                m_df = self.metrics()
                p_df = self.psi() if self.sample_col is not None else None
                _amostras = list(p_df["amostra"]) if p_df is not None else []
                val_sample = ("OOT" if "OOT" in _amostras
                              else (_amostras[0] if _amostras else None))
                stab = []
                if p_df is not None:
                    stab.append(("PSI da segmentação por amostra", p_df))
                stab.append(("Folhas da régua",
                             self.leaves(with_psi=(self.sample_col is not None))))
                dev_df = oot_df = None
                if save_base and self.sample_col is not None:
                    dev_df = self.df[self.df[self.sample_col] == self.ref_sample]
                    if val_sample is not None:
                        oot_df = self.df[self.df[self.sample_col] == val_sample]
                log_tabbed_report(
                    mlflow, run, title=f"TreeSegmenter — {self.task_type}",
                    subtitle=f"alvo '{self.target}' · {n_folhas} folhas · ref. {self.ref_sample}",
                    val_sample=val_sample, metrics_df=m_df, psi_df=p_df,
                    stability_blocks=stab, save_base=save_base,
                    dev_df=dev_df, oot_df=oot_df, verbose=verbose)
            except Exception as e:                       # pragma: no cover
                if verbose:
                    print(f"[mlflow] relatório em abas não gerado: {type(e).__name__}: {e}")
            run_id = run.info.run_id

        if verbose:
            msg = (f"[mlflow] modelo salvo (run {run_id[:8]}…) — {n_folhas} folhas. "
                   f"Artefatos: 'regua/' + modelo pyfunc '{artifact_path}/'.")
            if registered_model_name:
                v = f" v{version}" if version is not None else ""
                msg += f" Registrado em '{registered_model_name}'{v}."
            print(msg)
        return run_id
