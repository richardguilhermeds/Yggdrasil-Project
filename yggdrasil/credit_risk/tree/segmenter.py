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

import os
import sys
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ...utils import sem_aviso_solver_cvxpy as _sem_aviso_solver_cvxpy

try:
    with _sem_aviso_solver_cvxpy():     # cala a sondagem de solvers do cvxpy
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
    _amostra_optbinning,
    _cpsat_um_worker,
    checa_status_optbinning as _checa_status_optbinning,
    psi_from_shares as _psi_from_shares,
    sql_texto as _sql_texto,
    eh_booleana as _eh_booleana,
)

TASK_TYPES = ("classification", "regression")

# Métricas limitadas a [0,1] (discriminação): eixos que só as contêm ficam FIXOS
# em 0–1 p/ leitura estável e comparável entre safras — a autoescala "dá zoom" e
# exagera variações pequenas. As demais (regressão: rmse/mae · r2, que pode ser
# negativo) mantêm autoescala. Mesma convenção do ModelSegmenter.
_UNIT_METRICS = frozenset({"ks", "auc", "gini"})


# ======================================================================
# Helpers (formatação/classificação/optbinning vêm de credit_risk._common)
# ======================================================================
def _copy_on_write_ativo() -> bool:
    """True quando o pandas garante copy-on-write: pandas >= 3 (sempre ligado)
    ou ``mode.copy_on_write is True`` no 2.x. Nunca testar a opção como
    truthy: no 2.2+ ela pode valer ``"warn"`` (truthy), que NÃO é CoW — só
    avisa. Com CoW, ``df.copy(deep=False)`` já isola o segmentador do df do
    chamador, então a cópia profunda é dispensável."""
    try:
        if int(str(pd.__version__).split(".")[0]) >= 3:
            return True
    except ValueError:
        pass
    try:
        return pd.options.mode.copy_on_write is True
    except Exception:            # opção ausente (pandas antigo) → sem CoW
        return False


def _normaliza_n_jobs(n_jobs) -> int:
    """Nº de threads dos ajustes do optbinning (convenção do sklearn): ``None``
    ou 1 = em série (padrão); -1 = todos os núcleos; k ≥ 1 = k threads."""
    if n_jobs is None:
        return 1
    n = int(n_jobs)
    if n == -1:
        return os.cpu_count() or 1
    if n < 1:
        raise ValueError(f"n_jobs inválido: {n_jobs!r}. Use None, -1 ou um inteiro >= 1.")
    return n


def _fit_splits_na_thread(b, x, y, falhas) -> list:
    """Variante do ``fit_optbinning_splits`` para as threads do lote paralelo do
    :meth:`TreeSegmenter.variable_iv` — SEM ``warnings.catch_warnings``.

    Até o Python 3.13 (e nos runtimes do Databricks) o ``catch_warnings`` salva e
    restaura o ``warnings.filters`` GLOBAL: com várias threads entrando e saindo,
    uma restaura a lista que a outra já trocou. Por isso o "ignore" fica num
    ÚNICO ``catch_warnings`` na thread principal, em volta do lote (as threads
    leem os filtros globais). ATENÇÃO: as dependências do fit (sklearn
    ``check_array``/``type_of_target``, pandas ``pandas_dtype``…) abrem
    ``catch_warnings`` + ``simplefilter`` DENTRO dos workers — trocam a lista
    global e, ao sair, repõem a lista que salvaram (que contém o "ignore" do
    lote). Isso só é inofensivo enquanto o ``catch_warnings`` da thread principal
    estiver aberto (ele restaura a lista original ao sair, por cima de tudo); por
    isso NENHUM ajuste pode seguir rodando depois que o lote sai — nem na
    interrupção (ver :meth:`TreeSegmenter._bins_iv_resolvedor`). Com avisos por
    contexto (``sys.flags.context_aware_warnings``, 3.14+) os filtros da thread
    principal não chegam aqui — aí o ``catch_warnings`` local é seguro e é usado.
    ``np.errstate`` é por thread e fica — assim como o CP-SAT com 1 worker
    (:func:`_cpsat_um_worker`, contador por thread: tem de entrar NO worker).

    x/y já vêm amostrados (``_resolve_bins_preparar(amostrar=True)``). Mesma
    política de erro: ``ValueError`` (inviável / sem corte) → ``[]`` calado;
    outra exceção → ``[]`` e ``(nome, exceção)`` em ``falhas`` — o aviso é
    emitido depois, na thread principal (:class:`_RefazFalha`)."""
    try:
        if getattr(sys.flags, "context_aware_warnings", False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(divide="ignore", invalid="ignore"), _cpsat_um_worker():
                    b.fit(x, y)
        else:
            with np.errstate(divide="ignore", invalid="ignore"), _cpsat_um_worker():
                b.fit(x, y)
        _checa_status_optbinning(b)
        return list(b.splits)
    except ValueError:
        return []
    except Exception as e:
        falhas.append((getattr(b, "name", "?"), e))
        return []


class _RefazFalha:
    """Binning de mentira que relança, na thread principal, a exceção que um
    ajuste sofreu numa thread do lote. Passado ao ``fit_optbinning_splits``, o
    aviso "optbinning falhou inesperadamente" sai pelo MESMO código (mesma
    mensagem, categoria e origem) que sairia no ajuste em série."""

    def __init__(self, name, exc):
        self.name = name
        self._exc = exc

    def fit(self, x, y):
        raise self._exc


def _sem_prefixo_da_variavel(labels, feature):
    """Rótulos de bin sem o prefixo ``"<feature>: "``.

    O nome da variável já aparece no TÍTULO do gráfico; repeti-lo em cada item
    da legenda só rouba espaço e faz o rótulo ser truncado no meio do número
    (ex.: "comprometimento_renda: (0.3287, 0.56…"). Sobra a faixa, que é o que
    distingue as séries."""
    pref = f"{feature}: "
    return [l[len(pref):] if isinstance(l, str) and l.startswith(pref) else l
            for l in labels]


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


def _folha_fallback(regua, fallback):
    """Resolve a folha de fallback da régua para linhas órfãs (sem rota).

    ``fallback``: ``None`` mantém o comportamento atual (linha órfã fica nula);
    ``'pior_nota'`` devolve a folha de pior risco segundo a ordenação vigente
    (maior nota — a nota é a posição esquerda→direita da árvore, então a última
    é a de maior alvo na referência); um ``int`` devolve a folha daquela nota.
    Levanta ``ValueError`` para valor inválido ou nota inexistente.
    """
    if fallback is None:
        return None
    leaves = regua.get("leaves") or []
    if not leaves:
        return None
    if fallback == "pior_nota":
        return max(leaves, key=lambda lf: lf["nota"])
    if isinstance(fallback, (int, np.integer)) and not isinstance(fallback, bool):
        alvo = int(fallback)
        for lf in leaves:
            if lf["nota"] == alvo:
                return lf
        raise ValueError(
            f"fallback inválido: nota {alvo} não existe na régua "
            f"(notas válidas: {[lf['nota'] for lf in leaves]}).")
    raise ValueError(
        f"fallback inválido: {fallback!r}. Use 'pior_nota', uma nota (int) ou None.")


def _aplicar_regua_pandas(regua, df, col_seg="segmento",
                          col_nota="nota", col_valor="valor_regua",
                          fallback=None):
    """Aplica uma régua (dict de folhas) a um DataFrame pandas.

    ``fallback`` roteia as linhas órfãs (que não caem em folha nenhuma):
    ``None`` mantém nulo; ``'pior_nota'``/nota (int) atribui a folha escolhida
    — ver :func:`_folha_fallback`. O warning com a contagem é mantido."""
    fb = _folha_fallback(regua, fallback)   # valida cedo, mesmo sem órfãs
    seg = pd.Series(pd.NA, index=df.index, dtype="object")
    nota = pd.Series(pd.NA, index=df.index, dtype="Int64")
    pdcol = pd.Series(np.nan, index=df.index, dtype="float64")
    for leaf in regua["leaves"]:
        m = _match_conditions_pandas(df, leaf["conditions"])
        seg[m] = leaf["id"]
        nota[m] = leaf["nota"]
        pdcol[m] = leaf["pd"]
    # cobertura no scoring: linhas que não caíram em NENHUM segmento ficam nulas
    # (ou vão para a folha de fallback, quando configurado). Não é silencioso —
    # avisa com a causa provável (faltante sem rota quando o split usou
    # na_to_worst=False, ou categoria não vista no ajuste).
    orfas = seg.isna()
    n_orfas = int(orfas.sum())
    if n_orfas:
        import warnings
        causa = ("valor faltante numa variável sem rota de faltantes (grow com "
                 "na_to_worst=False) ou categoria não vista no ajuste")
        if fb is not None:
            seg[orfas] = fb["id"]
            nota[orfas] = fb["nota"]
            pdcol[orfas] = fb["pd"]
            warnings.warn(
                f"{n_orfas} linha(s) sem segmento na régua atribuída(s) ao "
                f"fallback '{fb['id']}' (nota {fb['nota']}): {causa}.",
                stacklevel=2)
        else:
            warnings.warn(
                f"{n_orfas} linha(s) sem segmento na régua (segmento nulo): "
                f"{causa}.", stacklevel=2)
    return pd.DataFrame({col_seg: seg, col_nota: nota, col_valor: pdcol}, index=df.index)


# sentinela p/ distinguir "parâmetro omitido" (usa o fallback persistido em
# ``self.fallback``) de "explicitamente None" (sem fallback nesta chamada)
_FALLBACK_OMITIDO = object()

# sentinela p/ o ``time_col`` do variable_inversion: pula o laço por safra (o
# gráfico por amostra não lê nada dele). ``time_col=False`` não serve: o
# ``time_col or self.date_col`` o transformaria em date_col.
_SEM_SAFRAS = object()


def _emit_progress(cb, key: str, label: str, status: str, detail: str = "") -> None:
    """Dispara um evento de progresso (aplicação da régua) para a UI, se houver
    callback.

    ``status``: ``"run"`` (iniciando a etapa), ``"ok"`` (concluída) ou ``"err"``.
    Nunca derruba a ação — o progresso é cosmético (mesma política do
    ``ModelSegmenter``)."""
    if cb is None:
        return
    try:
        cb(key, label, status, detail)
    except Exception:  # noqa: BLE001 - progresso é cosmético
        pass


# ======================================================================
# Classe principal
# ======================================================================
class TreeSegmenter:
    """Árvore de segmentação unificada (classificação/regressão) com binning
    ótimo/manual, poda e PSI. O comportamento por tipo de alvo é escolhido por
    ``task_type`` ("classification" p/ alvo · "regression" p/ alvo).

    ``weight_col`` (opcional): coluna de PESO/EXPOSIÇÃO (ex.: saldo do contrato)
    que habilita a **visão dupla contratos × saldo** — a tabela de folhas e o
    cartão da folha passam a mostrar, além de ``n`` e % de contratos, o % do
    SALDO (soma dos pesos da folha / total) e o alvo médio PONDERADO pelos pesos
    (``saldo_%`` · ``valor_medio_pond`` · ``valor_pond_<amostra>``). **Fase 1 —
    exibição apenas**: o critério de split e o binning ótimo continuam NÃO
    ponderados (contrato a contrato); ponderar os cortes é fase 2 futura."""

    # PREVIEW interativo (plot_tree_hitmap): a figura da árvore cresce ∝ nº de
    # folhas (X_GAP por folha), então uma árvore grande vira um PNG gigantesco
    # (~80 folhas ≈ 180" → 27000 px @150 dpi) cujo *encode* domina a renderização
    # a CADA _refresh. Capamos a largura do PNG do preview reduzindo o dpi efetivo
    # (mantendo um piso de legibilidade): a geometria — e o hit-map, que escala
    # junto — fica idêntica, e o front reamostra a imagem de qualquer forma.
    _PREVIEW_MAX_PX = 5200        # teto de largura do PNG do preview (px)
    _PREVIEW_MIN_DPI = 40.0       # piso de dpi (não deixa o texto virar borrão)

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
        weight_col: str | None = None,
        verbose: bool = True,
        copy: bool = True,
        n_jobs: int | None = 1,
    ):
        # copy: True (padrão) guarda uma cópia PROFUNDA do df — o segmentador
        # fica isolado de qualquer escrita posterior no df do chamador. False
        # guarda uma cópia RASA (df.copy(deep=False)): DataFrame novo (incluir
        # ou remover colunas no original não o afeta) que COMPARTILHA os dados,
        # sem duplicar a base na RAM. O segmentador nunca escreve em self.df,
        # então os resultados são idênticos — DESDE QUE o chamador também não
        # altere o df in-place (df.loc[...] = ...) enquanto usa o segmentador:
        # no pandas 2.x sem copy-on-write essa escrita passaria a ser vista aqui
        # e deixaria velhos os caches por máscara. Com copy-on-write ligado
        # (pandas >= 3, ou mode.copy_on_write=True) a cópia rasa já é isolada,
        # e é usada mesmo com copy=True. Não é persistido em to_dict().
        # n_jobs: threads dos ajustes do optbinning no variable_iv (uma variável
        # por thread; o CART e o CP-SAT soltam a GIL). 1 (padrão) = em série,
        # como antes; -1 = todos os núcleos. Os cortes são idênticos aos do
        # ajuste em série, inclusive em empate de ótimo: o CP-SAT de cada ajuste
        # roda com 1 worker (_cpsat_um_worker, por thread). Não é persistido em
        # to_dict(); pode ser trocado depois em ``seg.n_jobs``.
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
        # fallback de scoring p/ linhas órfãs (categoria não vista / faltante sem
        # rota): None mantém nulo; 'pior_nota' roteia p/ a folha de pior risco
        # (maior nota); int roteia p/ a nota específica. É o PADRÃO usado por
        # predict/to_sql/to_pyspark/apply_spark quando o parâmetro `fallback` é
        # omitido, e é persistido em to_dict/from_dict (viaja no JSON salvo).
        self.fallback: str | int | None = None
        # cache do binning ótimo por folha (o solver do optbinning é o gargalo) —
        # ver _resolve_bins_cached; chaveado por id() da máscara da folha
        self._bins_cache: dict = {}
        self.n_jobs = _normaliza_n_jobs(n_jobs)
        # ---- caches de agregados por VERSÃO da árvore (desempenho) -------------
        # Toda mutação estrutural (grow/prune/collapse/merge_*/auto_merge/load/
        # fit_auto) chama _bump_version(). leaves()/_leaf_order()/_grade_map()/
        # metrics()/psi() são reconstruídos ~4-5× por _refresh da UI varrendo
        # máscaras full-length por folha; memoizá-los por versão elimina esse
        # recompute redundante (o maior gargalo da UI da árvore no Databricks).
        self._tree_version: int = 0
        self._agg_cache: dict = {}        # (nome, *args, version) -> resultado
        # 2º nível: LRU por ASSINATURA ESTRUTURAL (ver _agg_assinatura). O undo/
        # redo volta a um estado já visto com os MESMOS objetos-máscara, mas numa
        # versão nova — o cache por versão dava miss em tudo. assinatura -> {
        # 'mascaras' (referência forte, conferida com `is`), 'rotulos', 'vals'}.
        self._agg_lru: OrderedDict = OrderedDict()
        # alvo por folha restrito à referência — chaveado por id(mask),
        # invalida sozinho quando a folha é refeita (mesma ideia do _bins_cache)
        self._leaf_target_cache: dict = {}
        # p-valor do teste de alvo entre duas folhas (_pair_pvalue), chaveado por
        # (id da máscara a, id da máscara b, teste, min_n) — mesma ideia: as duas
        # máscaras ficam no valor e são conferidas com `is`. Limpo junto com o
        # _leaf_target_cache (não retém máscaras que aquele cache já soltou).
        self._pair_p_cache: dict = {}
        # máscara por ASSINATURA DE CONDIÇÕES: como (condições → máscara) é função
        # PURA do df (que não muda), o cache nunca fica stale. Acelera undo/redo —
        # _load_segments reconstruía toda máscara via _match_conditions_pandas a
        # cada desfazer/refazer; agora os segmentos inalterados são cache-hit.
        self._mask_cache_by_conds: dict = {}
        # amostra dominante por safra (moda) — função PURA do df (imutável): as
        # linhas/bandas DES/OOT/… por safra são desenhadas em MUITOS gráficos (e por
        # subplot nos grids); cacheado por time_col p/ não regroupar a df toda a cada.
        self._samp_by_cache: dict = {}
        # classe de optbinning e nome do kwarg de diferença mínima por tipo de alvo
        self._OptBin = OptimalBinning if self._is_clf else ContinuousOptimalBinning
        self._diff_kwarg = "min_event_rate_diff" if self._is_clf else "min_mean_diff"
        if self._OptBin is None:
            raise ImportError("optbinning não instalado. Rode: pip install optbinning")
        if target not in df.columns:
            raise ValueError(f"Coluna alvo '{target}' não está no DataFrame "
                             f"(colunas: {list(df.columns)}).")

        self.df = df.copy(deep=bool(copy) and not _copy_on_write_ativo())
        self.target = target
        self.sample_col = sample_col
        self.ref_sample = ref_sample
        # coluna de DATA/safra: NÃO entra na modelagem (fica fora das features);
        # serve só para os gráficos no tempo (alvo/variável por safra, PSI por safra).
        self.date_col = date_col
        if date_col is not None and date_col not in self.df.columns:
            raise ValueError(f"Coluna de data '{date_col}' não está no DataFrame "
                             f"(colunas: {list(self.df.columns)}).")
        # coluna de PESO/EXPOSIÇÃO (ex.: saldo): NUNCA entra na modelagem (fica
        # fora das variáveis candidatas) e habilita a visão dupla contratos ×
        # saldo em leaves()/UI/exports. FASE 1: exibição apenas — split e binning
        # ótimo seguem NÃO ponderados (fase 2 futura). Pesos NaN são tratados
        # como ausentes (fora das somas/médias ponderadas).
        self.weight_col = weight_col
        if weight_col is not None:
            if weight_col not in self.df.columns:
                raise ValueError(
                    f"Coluna de peso '{weight_col}' não está no DataFrame "
                    f"(colunas: {list(self.df.columns)}).")
            if not pd.api.types.is_numeric_dtype(self.df[weight_col]):
                raise ValueError(
                    f"Coluna de peso '{weight_col}' precisa ser numérica "
                    f"(dtype atual: {self.df[weight_col].dtype}).")
            w = self.df[weight_col].astype("float64")
            if (w.dropna() < 0).any():
                raise ValueError(
                    f"Coluna de peso '{weight_col}' tem valor NEGATIVO — pesos "
                    "de exposição devem ser ≥ 0.")
            self._weights = w
        else:
            self._weights = None
        # rótulos amigáveis por variável para a descrição por extenso
        self.feature_labels = feature_labels or {}
        # APELIDOS DE NEGÓCIO por folha: sid -> texto livre (ex.: "Baixo risco
        # com garantia"). Os sids derivam dos RÓTULOS DE CONDIÇÃO, então mudam
        # quando a estrutura muda: fundir/dividir renomeia a folha e o apelido do
        # sid antigo fica ÓRFÃO — órfãos são ignorados na leitura (leaf_name) e
        # descartados em silêncio na serialização (to_dict). Um collapse que
        # restaura um sid antigo "revive" o apelido correspondente. Persistido em
        # to_dict/from_dict (JSONs antigos sem a chave carregam com {}).
        self.leaf_names: dict[str, str] = {}
        # mínimo de linhas (na amostra de ajuste) para tentar binning ótimo
        self.min_leaf_rows = min_leaf_rows

        uniq = None
        if sample_col is not None:
            if sample_col not in self.df.columns:
                raise ValueError(
                    f"Coluna de amostra '{sample_col}' não está no DataFrame "
                    f"(colunas: {list(self.df.columns)}).")
            # UMA passada de unique: a validação usa a lista (.tolist()) e as
            # máscaras abaixo iteram o MESMO array (antes eram duas passadas)
            uniq = self.df[sample_col].dropna().unique()
            amostras = uniq.tolist()
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
            self._sample_masks = {a: (self.df[sample_col] == a) for a in uniq}
            self._mask_ref = self._sample_masks.get(ref_sample)
        else:
            self._sample_masks = {}
            self._mask_ref = None
        # amostras na ordem de dropna().unique() (= ordem das chaves acima). Os
        # métodos liam `df[sample_col].dropna().unique()` de novo a cada chamada;
        # agora leem esta lista (use list(self._samples) se for alterá-la).
        self._samples = list(self._sample_masks)
        # código da amostra por linha (int32; -1 = sem amostra: NaN/None/pd.NA),
        # com k = posição em self._samples. Derivado das PRÓPRIAS máscaras (mesma
        # partição que define os denominadores do PSI). None se alguma linha
        # casar com duas amostras (só com objetos de igualdade não transitiva) —
        # aí os caminhos por código caem no caminho antigo.
        self._sample_codes = self._build_sample_codes()
        # _ordered_samples_with_value: função pura do df → memo preguiçoso (tupla)
        self._osv_memo: tuple | None = None

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

    def __setstate__(self, state):
        """Despickle (pickle/joblib/``copy``) tolerante a versão.

        Os memos de agregado/binning começam VAZIOS: são caches puros, alguns
        chaveados por ``id()`` de máscara (ids de outro processo podem coincidir
        com objetos novos) e o formato deles muda entre versões (o ``psi`` da
        0.0.21 guardava só o DataFrame). Um segmentador salvo por uma versão
        anterior também não tem os atributos DERIVADOS que vieram depois
        (amostras/códigos por linha, ``n_jobs``) — recompostos aqui a partir do
        df e das máscaras, em vez de estourar ``AttributeError`` em
        ``leaves``/``metrics``/``psi``."""
        self.__dict__.update(state)
        d = self.__dict__
        d["_agg_cache"] = {}
        d["_agg_lru"] = OrderedDict()
        d["_bins_cache"] = {}
        d["_leaf_target_cache"] = {}
        d["_pair_p_cache"] = {}
        d["_samp_by_cache"] = {}
        d["_osv_memo"] = None
        d.setdefault("n_jobs", 1)
        d.setdefault("_tree_version", 0)
        if "_samples" not in d:
            d["_samples"] = list(d.get("_sample_masks") or {})
        if "_sample_codes" not in d:
            d["_sample_codes"] = self._build_sample_codes()

    # ------------------------------------------------------------------
    # Versão da árvore + memoização de agregados (desempenho)
    # ------------------------------------------------------------------
    def _bump_version(self):
        """Marca a árvore como alterada: invalida os caches de agregado por versão.
        Chamado por TODA mutação estrutural (grow/prune/collapse/merge_*/
        auto_merge/_load_segments/fit_auto)."""
        self._tree_version += 1
        # a versão só cresce: as entradas das versões anteriores nunca mais são
        # lidas. Limpar já (em vez do antigo backstop de 64 entradas) solta os
        # arrays por linha (código/rótulo da folha) das versões mortas.
        self._agg_cache.clear()

    # agregados que também ficam na LRU por assinatura: resultados PEQUENOS e
    # funções puras de (estrutura, máscaras, df imutável, parâmetros da chave).
    # Os arrays por linha (leaf_codes/leaf_id_arr, N posições) ficam só no
    # cache por versão — não seguram memória de estados antigos.
    _AGG_LRU_NOMES = frozenset({"leaves", "leaf_order", "grade_map", "metrics",
                                "psi", "feature_importance"})
    _AGG_LRU_SIGS = 8          # últimas assinaturas retidas (0 desliga a LRU)
    _AGG_LRU_MASCARAS = 256    # teto de máscaras DISTINTAS retidas pela LRU

    def _agg_assinatura(self):
        """``(chave, máscaras)`` da estrutura ATUAL da árvore, ou ``None`` se não
        der para montar uma chave hashável. A chave cobre, por segmento e na
        ordem de ``segments``: sid, id da máscara, is_leaf, pai, rótulo,
        profundidade, caminho e as condições na forma serializada (a mesma do
        undo/redo e do cache de máscaras). ``máscaras`` são os objetos, para a
        LRU segurar a referência (o id não é reciclado) e conferir com ``is``.

        Calculada na LEITURA (a cada miss do cache por versão), nunca em
        _bump_version: a versão sobe ANTES da mutação, e uma assinatura
        congelada ali gravaria o resultado pós-mutação sob a estrutura antiga."""
        try:
            partes, mascaras = [], []
            for sid, s in self.segments.items():
                m = s["mask"]
                partes.append((sid, id(m), bool(s["is_leaf"]), s["parent"],
                               s["label"], s["depth"], tuple(s["path"]),
                               self._sig_conds(s["conditions"])))
                mascaras.append(m)
            chave = tuple(partes)
            hash(chave)
        except Exception:
            return None
        return chave, tuple(mascaras)

    def _sig_conds(self, conds):
        """Condições na forma serializada (``_conds_key(_conditions_json)``), que
        é a forma que o undo/redo restaura — o estado vivo e o restaurado têm a
        mesma chave. Um limite numérico que o ``float()`` do JSON não representa
        exatamente (int > 2**53, Decimal) ganha o ``repr`` cru: aí o estado vivo
        e o restaurado diferem na chave (miss, seguro), porque a ordem e o texto
        poderiam diferir."""
        k = self._conds_key(self._conditions_json(conds))
        for c in conds:
            if c["kind"] == "num" and any(not (b == float(b))
                                          for b in (c["lo"], c["hi"])):
                return k + "|" + repr(conds)
        return k

    def _agg_contexto(self):
        """O que os agregados leem ALÉM da estrutura: os objetos fixos da
        construção (conferidos com ``is``) e a configuração (com ``==``), com
        os rótulos de variável (``feature_labels``, lidos pela descrição das
        folhas). Uma entrada da LRU só vale com o mesmo contexto."""
        return ((self.df, self._weights, self._mask_ref, self._sample_codes),
                (self.target, self.sample_col, self.ref_sample, self.task_type,
                 self.weight_col, tuple(self._samples), dict(self.feature_labels)))

    def _agg_contexto_ok(self, ctx):
        objs, vals = self._agg_contexto()
        return all(a is b for a, b in zip(ctx[0], objs)) and ctx[1] == vals

    def _agg_lru_valida(self, ent, sig):
        """A entrada vale para ``sig``: as MESMAS máscaras (``is``) e o mesmo
        contexto."""
        return (all(a is b for a, b in zip(ent["mascaras"], sig[1]))
                and self._agg_contexto_ok(ent["ctx"]))

    def _agg_lru_ler(self, key, sig):
        """Valor de ``key`` na entrada da assinatura ``sig`` ou ``None``."""
        ent = self._agg_lru.get(sig[0])
        if ent is None or not self._agg_lru_valida(ent, sig):
            return None
        self._agg_lru.move_to_end(sig[0])
        return ent["vals"].get(key)

    def _agg_lru_gravar(self, key, sig, valor):
        """Grava ``valor`` sob ``sig`` (substitui a entrada se as máscaras ou o
        contexto não baterem) e despeja as mais antigas além de
        ``_AGG_LRU_SIGS`` assinaturas ou ``_AGG_LRU_MASCARAS`` máscaras
        distintas. A entrada recém-gravada nunca é despejada (as máscaras dela
        são as da árvore atual, vivas em ``segments`` de qualquer forma)."""
        lru = self._agg_lru
        ent = lru.get(sig[0])
        if ent is None or not self._agg_lru_valida(ent, sig):
            ent = {"mascaras": sig[1], "ctx": self._agg_contexto(), "vals": {}}
            lru[sig[0]] = ent
        ent["vals"][key] = valor
        lru.move_to_end(sig[0])
        while len(lru) > 1:
            if len(lru) <= self._AGG_LRU_SIGS:
                n_masc = len({id(m) for e in lru.values() for m in e["mascaras"]})
                if n_masc <= self._AGG_LRU_MASCARAS:
                    break
            lru.popitem(last=False)

    def _agg_memo(self, key, build):
        """Memoiza ``build()`` por (key, versão-da-árvore). Devolve sempre uma
        cópia defensiva (DataFrame/Series/dict/list) para o chamador não corromper
        o cache. ``build`` é uma função sem argumentos.

        No miss da versão, os agregados de ``_AGG_LRU_NOMES`` consultam a LRU
        por assinatura estrutural: voltar (undo/redo, collapse + o mesmo split
        com as máscaras reaproveitadas) a uma árvore já vista reaproveita o
        resultado em vez de revarrer as máscaras."""
        import copy
        ck = (key, self._tree_version)
        hit = self._agg_cache.get(ck)
        if hit is None:
            sig = None
            if self._AGG_LRU_SIGS > 0 and key[0] in self._AGG_LRU_NOMES:
                # assinatura ANTES do build: o resultado é gravado sob a
                # estrutura em que foi calculado (build só lê a árvore)
                sig = self._agg_assinatura()
                if sig is not None:
                    try:
                        hit = self._agg_lru_ler(key, sig)
                    except Exception:       # ex.: feature_labels trocado por
                        sig = hit = None    # algo não comparável → sem LRU
            if hit is None:
                hit = build()
                if sig is not None and hit is not None:
                    try:
                        self._agg_lru_gravar(key, sig, hit)
                    except Exception:
                        pass
            self._agg_cache[ck] = hit
        if isinstance(hit, (pd.DataFrame, pd.Series)):
            return hit.copy()
        if isinstance(hit, (dict, list)):
            # deepcopy: valores ANINHADOS mutáveis (ex.: grade_map = {'nota':dict,
            # 'desc':dict}) não podem vazar por referência e contaminar o cache.
            return copy.deepcopy(hit)
        return hit

    # ------------------------------------------------------------------
    # Máscaras/códigos de amostra (fixos após a construção)
    # ------------------------------------------------------------------
    def _build_sample_codes(self):
        """Código da amostra por linha (int32 read-only; -1 = sem amostra),
        montado a partir de ``_sample_masks`` (NA da máscara anulável = fora).
        ``None`` sem sample_col ou se alguma linha cair em duas máscaras."""
        if not self._sample_masks:
            return None
        codes = np.full(len(self.df), -1, dtype=np.int32)
        for k, m in enumerate(self._sample_masks.values()):
            mb = m.to_numpy(dtype=bool, na_value=False)
            if (codes[mb] >= 0).any():
                return None
            codes[mb] = k
        codes.flags.writeable = False
        return codes

    def _smask(self, a):
        """Máscara da amostra ``a`` sobre a base INTEIRA — a mesma Series de
        ``df[sample_col] == a``, lida das máscaras pré-computadas em vez de
        recomparar a coluna (object) full-length. Rótulo fora das amostras
        (inexistente, NaN, não hasheável) cai no ``==`` literal, com o mesmo
        resultado de antes. A Series é COMPARTILHADA: não altere in-place
        (``&=``/``|=``); quem a devolve ao chamador entrega ``.copy()``."""
        try:
            m = self._sample_masks.get(a)
        except TypeError:
            m = None
        if m is None:
            return self.df[self.sample_col] == a
        return m

    def _leaf_codes(self, leaf_ids=None):
        """``(sids, código)``: código int32 (read-only) da folha de cada linha —
        posição em ``sids``, -1 = sem folha (órfã). ``leaf_ids=None`` = todas as
        folhas na ordem de ``segments`` (mesma chave da tupla explícita).
        Memoizado por versão; substitui o array object de rótulos por linha."""
        ids = (tuple(leaf_ids) if leaf_ids is not None
               else tuple(sid for sid, s in self.segments.items() if s["is_leaf"]))

        def build():
            code = np.full(len(self.df), -1, dtype=np.int32)
            for k, sid in enumerate(ids):
                code[self.segments[sid]["mask"].values] = k
            code.flags.writeable = False
            return ids, code
        return self._agg_memo(("leaf_codes", ids), build)

    def _leaf_sample_counts(self, leaf_ids=None):
        """Contagem de linhas por (folha, amostra) como função ``cnt(sid, amostra)``
        → int: o MESMO número do antigo ``pd.crosstab(leaf_id, df[sample_col])``
        (órfãs e amostra NaN fora), via UM bincount de códigos inteiros. As
        linhas inválidas (folha -1 ou amostra -1) saem ANTES de combinar os
        códigos — senão folha k com amostra -1 cairia na célula da folha k-1."""
        ids, code = self._leaf_codes(leaf_ids)
        sc = self._sample_codes
        if sc is None or not self.df.index.is_unique:
            # caminho antigo, literal: crosstab sobre uma CÓPIA da série de
            # rótulos (o memo antigo devolvia .copy(), com outro objeto de
            # índice). Com índice DUPLICADO o crosstab reindexa e levanta
            # ValueError — é o comportamento atual de psi()/leaves(with_psi),
            # preservado aqui; deixar de levantar é decisão à parte.
            ct = pd.crosstab(self._leaf_id_series(leaf_ids).copy(),
                             self.df[self.sample_col])

            def cnt(sid, amostra):
                col = ct[amostra] if amostra in ct.columns else None
                return col.get(sid, 0) if col is not None else 0
            return cnt
        n_s, n_l = len(self._samples), len(ids)
        valid = (code >= 0) & (sc >= 0)
        flat = np.bincount(code[valid].astype(np.int64) * n_s + sc[valid],
                           minlength=n_l * n_s)
        mat = flat.reshape(n_l, n_s)
        # sid → linha por DICIONÁRIO (não por posição: quem pergunta pode estar
        # noutra ordem, ex. out["segmento"] ordenado por nota)
        linha = {sid: k for k, sid in enumerate(ids)}
        coluna = {a: j for j, a in enumerate(self._samples)}

        def cnt(sid, amostra):
            k, j = linha.get(sid), coluna.get(amostra)
            return int(mat[k, j]) if (k is not None and j is not None) else 0
        return cnt

    def _nonfeature_cols(self) -> set:
        """Colunas que NÃO entram na modelagem (não viram variáveis candidatas):
        o alvo, a coluna de amostra, a coluna de data de referência (``date_col``)
        e **qualquer coluna datetime** — uma data nunca é variável do modelo aqui
        (e o optbinning não a bina). Para datas que não sejam datetime (ex.: safra
        yyyymm inteira), passe ``date_col`` no construtor para excluí-la também.
        A coluna de peso (``weight_col``) também fica fora — peso não é variável."""
        skip = {self.target, self.sample_col, self.date_col, self.weight_col}
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
    def _sub(self, mask, *cols):
        """Recorte de LINHAS só com as colunas usadas: as pedidas (variável, safra)
        + alvo, amostra e peso. ``self.df[mask]`` copiava TODAS as colunas da base a
        cada recorte (nó da árvore, gráfico, amostra) — em milhões de linhas × dezenas
        de colunas era o grosso do tempo e do pico de memória da UI."""
        keep = [c for c in dict.fromkeys((*cols, self.target, self.sample_col,
                                          self.weight_col))
                if c is not None and c in self.df.columns]
        return self.df.loc[mask, keep]

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
        if sub is self.df:
            # base INTEIRA (ex.: csi, por variável): a máscara da referência já
            # existe — é a mesma `self.df[sample_col] == ref_sample`. Recortes
            # (sub ≠ self.df) seguem comparando no próprio recorte.
            fit_data = sub[self._mask_ref]
        else:
            fit_data = sub[sub[self.sample_col] == self.ref_sample]
        if len(fit_data) < 2 * int(min_bin_size * len(self.df)):
            return sub  # ramo com pouco DES: usa o ramo inteiro
        return fit_data

    def _mask_in(self, frame, feature, b):
        """Máscara booleana das linhas de `frame` que caem no bin `b`."""
        return self._mask_in_col(frame[feature], b)

    @staticmethod
    def _mask_in_col(col, b):
        """:meth:`_mask_in` sobre a própria coluna (Series) — mesmo resultado de
        ``_mask_in(frame, feature, b)`` com ``col = frame[feature]``."""
        if b["kind"] == "na":
            return col.isna()
        if b["kind"] == "num":
            return col.between(b["lo"], b["hi"], inclusive="right")
        return col.astype(str).isin(b["cats"])

    @staticmethod
    def _bin_ids(col, bins):
        """Código da faixa por linha (int32; -1 = nenhuma faixa) numa passada só,
        ou ``None`` quando não dá para garantir ``(ids == k)`` idêntico a
        ``_mask_in_col(col, bins[k]).to_numpy()`` — aí o chamador volta ao
        ``_mask_in`` por faixa (caminho antigo). Troca K ``astype(str)``/``between``
        da coluna (um por faixa, e por amostra/safra) por um só.

        - num (faixas contíguas (-inf, c1], …, (ck, inf] + 'na'): só para coluna
          numpy float64, ou inteira com |x| ≤ 2**53 (conversão exata). ``searchsorted
          (cortes, x, 'left')`` dá o i com c[i-1] < x ≤ c[i], a pertinência do
          ``between(lo, hi, 'right')``; -inf não cai em faixa nenhuma e NaN vai para
          'na'. float32/anuláveis/object/Decimal: ``None`` (o ``between`` compara em
          outra precisão ou devolve NA, e o IV hoje vira NaN nesse caso).
        - cat: ``astype(str)`` UMA vez e busca exata do texto nas categorias (as
          faixas são disjuntas); NA vai para 'na'. Se uma linha NA também casar com
          uma categoria (ex.: ``pd.NaT`` → 'NaT' literal) ela hoje conta nas DUAS
          faixas — não cabe num código único, então ``None``. Nunca ``factorize``
          dos valores brutos: fundiria 1/'1'/True e Decimal('1.0')/('1.00')."""
        if not isinstance(col, pd.Series) or not bins:
            return None
        kinds = [b.get("kind") for b in bins]
        na_pos = [k for k, kd in enumerate(kinds) if kd == "na"]
        if len(na_pos) > 1:
            return None
        na_i = na_pos[0] if na_pos else -1
        num_pos = [k for k, kd in enumerate(kinds) if kd == "num"]
        cat_pos = [k for k, kd in enumerate(kinds) if kd == "cat"]
        if len(num_pos) + len(cat_pos) + len(na_pos) != len(bins):
            return None
        n = len(col)

        if num_pos and not cat_pos:
            dt = col.dtype
            if not isinstance(dt, np.dtype) or not (dt == np.float64 or dt.kind in "iu"):
                return None

            def _exato(v):                  # corte representável em float64 exato
                if isinstance(v, (bool, np.bool_)):
                    return False
                if isinstance(v, np.floating):      # longdouble não cabe em float64
                    return v.dtype.itemsize <= 8
                if isinstance(v, float):
                    return True
                if isinstance(v, (int, np.integer)):
                    return abs(int(v)) <= 2 ** 53
                return False
            los = [bins[k]["lo"] for k in num_pos]
            his = [bins[k]["hi"] for k in num_pos]
            if not all(_exato(v) for v in (*los, *his)):
                return None
            los = [float(v) for v in los]
            his = [float(v) for v in his]
            if not (los[0] == -np.inf and his[-1] == np.inf
                    and all(his[j] == los[j + 1] for j in range(len(his) - 1))):
                return None
            cuts = np.asarray(his[:-1], dtype=np.float64)
            if np.isnan(cuts).any() or (cuts.size > 1 and (np.diff(cuts) < 0).any()):
                return None
            x = col.to_numpy()
            if dt.kind in "iu":
                if n and (int(x.min()) < -2 ** 53 or int(x.max()) > 2 ** 53):
                    return None
                x = x.astype(np.float64)
            mapa = np.asarray(num_pos, dtype=np.int32)
            ids = mapa[np.searchsorted(cuts, x, side="left")]
            ids[x == -np.inf] = -1                 # (-inf, c1] exclui o -inf
            ids[np.isnan(x)] = na_i                # NaN: só a faixa 'na'
            return ids

        if cat_pos and not num_pos:
            flat, grp = [], []
            for k in cat_pos:
                for c in bins[k]["cats"]:
                    flat.append(c)
                    grp.append(k)
            if not all(type(c) is str for c in flat):
                return None
            idx = pd.Index(np.asarray(flat, dtype=object), dtype=object)
            if not idx.is_unique:                  # faixas não disjuntas
                return None
            vals = np.asarray(col.astype(str).to_numpy(), dtype=object)
            onde = idx.get_indexer(vals)           # -1 = texto fora das faixas
            ids = np.asarray([*grp, -1], dtype=np.int32)[onde]
            if na_i >= 0:
                isna = col.isna().to_numpy(dtype=bool)
                if isna.any():
                    if (ids[isna] >= 0).any():     # NA que casa com categoria
                        return None
                    ids[isna] = na_i
            return ids
        return None

    def _bin_counts(self, col, bins, ids=None):
        """Nº de linhas de ``col`` em cada faixa (lista de int, ordem de
        ``bins``) — o mesmo ``int(_mask_in(...).sum())`` por faixa, contado de uma
        vez pelo código de faixa (:meth:`_bin_ids`), com fallback literal."""
        if ids is None:
            ids = self._bin_ids(col, bins)
        if ids is None:
            return [int(self._mask_in_col(col, b).sum()) for b in bins]
        return [int(c) for c in np.bincount(ids[ids >= 0], minlength=len(bins))]

    def _bin_masks(self, frame, feature, bins):
        """Máscara de cada faixa sobre ``frame`` (lista, ordem de ``bins``):
        ``ids == k`` (ndarray bool, mesmos valores do ``_mask_in``) quando há código
        de faixa por linha; senão o ``_mask_in`` de cada faixa (caminho antigo)."""
        if not bins:
            return []
        ids = self._bin_ids(frame[feature], bins)
        if ids is None:
            return [self._mask_in(frame, feature, b) for b in bins]
        return [ids == k for k in range(len(bins))]

    def _pos_of(self, mask):
        """Posições (ordem crescente) das linhas de ``self.df.loc[mask]`` — o
        recorte por COLUNA (``self.df[col].iloc[pos]``) dá os mesmos valores, na
        mesma ordem, que a coluna do frame ``self.df.loc[mask, cols]``. Máscara
        bool/boolean alinhada à base: ``flatnonzero`` (NA conta como False, como no
        ``.loc``). Fora disso, o próprio ``.loc`` sobre uma Series de posições."""
        if isinstance(mask, pd.Series) and (mask.index is self.df.index
                                            or mask.index.equals(self.df.index)):
            if mask.dtype == bool:
                return np.flatnonzero(mask.to_numpy())
            if isinstance(mask.dtype, pd.BooleanDtype):
                return np.flatnonzero(mask.to_numpy(dtype=bool, na_value=False))
        marc = pd.Series(np.arange(len(self.df)), index=self.df.index)
        return marc.loc[mask].to_numpy()

    @staticmethod
    def _bin_label(feature, b):
        if b["kind"] == "na":
            return f"{feature}: (faltante)"
        if b["kind"] == "num":
            return f"{feature}: ({_fmt(b['lo'])}, {_fmt(b['hi'])}]"
        return f"{feature}: {{{', '.join(map(str, b['cats']))}}}"

    @staticmethod
    def _bin_label_short(b):
        """Rótulo do bin SEM o prefixo da variável — p/ legendas onde o nome da
        variável já está no título (evita repetir 'feat: ' em cada item e o
        truncamento no meio do número). Ex.: '(-inf, 0.62]', '(faltante)'."""
        if b["kind"] == "na":
            return "(faltante)"
        if b["kind"] == "num":
            return f"({_fmt(b['lo'])}, {_fmt(b['hi'])}]"
        return "{" + ", ".join(map(str, b["cats"])) + "}"

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
        key = self._bins_key(mask, feature, splits, dtype, max_n_bins, min_bin_size,
                             max_bin_size, relax_max, min_mean_diff, criterion)
        hit = self._bins_cache.get(key)
        if hit is not None:
            return hit[0], hit[1], hit[2]
        res = self._resolve_bins(self._sub(mask, feature), feature, splits, dtype, max_n_bins,
                                 min_bin_size, max_bin_size, relax_max, min_mean_diff,
                                 criterion)
        self._bins_cache_put(key, res, mask)
        return res[0], res[1], res[2]

    def _bins_key(self, mask, feature, splits, dtype, max_n_bins, min_bin_size,
                  max_bin_size=None, relax_max=False, min_mean_diff=0.0,
                  criterion="optbin"):
        """Chave do :attr:`_bins_cache` (ver :meth:`_resolve_bins_cached`)."""
        return (id(mask), feature, self._splits_key(splits), dtype, max_n_bins,
                min_bin_size, max_bin_size, relax_max, round(float(min_mean_diff), 9),
                criterion)

    def _bins_cache_put(self, key, res, mask):
        """Grava ``res`` = (bins, modo, kind) no :attr:`_bins_cache` (thread
        principal; com o backstop de memória de sempre)."""
        if len(self._bins_cache) > 6000:        # backstop de memória
            self._bins_cache.clear()
        self._bins_cache[key] = (res[0], res[1], res[2], mask)

    def _bins_iv_resolvedor(self, sid, features, max_n_bins, min_bin_size):
        """Bins do :meth:`variable_iv` por variável: devolve ``(obter, fechar)``.

        ``obter(i, feat)`` (``feat = features[i]``) = ``_resolve_bins_cached(sid,
        feat, None, None, max_n_bins, min_bin_size)`` — mesmo valor, mesmas exceções e o MESMO
        estado final do :attr:`_bins_cache`. Com ``n_jobs`` = 1 (padrão) é
        literalmente essa chamada. Com ``n_jobs`` > 1, os ajustes do optbinning
        rodam em threads, em LOTES de ``2·n_jobs`` variáveis (memória limitada:
        cada ajuste pendente guarda só x/y já amostrados):

        1. preparar (thread principal, na ordem): consulta o cache e roda o
           :meth:`_resolve_bins_preparar` das que faltam — exceção fica guardada;
        2. resolver: ``map`` ORDENADO num ``ThreadPoolExecutor`` (criado por
           chamada; nada de processos), dentro de UM ``catch_warnings`` +
           ``simplefilter("ignore")`` na thread principal; o nosso código nos
           workers não mexe nos filtros, mas as dependências do fit (sklearn,
           pandas) mexem — por isso o lote só sai do ``catch_warnings`` com
           TODOS os ajustes terminados, inclusive numa interrupção: aí cancela
           os pendentes e espera os em curso (no máximo 1 por worker) antes de
           relançar (ver :func:`_fit_splits_na_thread`). Com 1 só ajuste no
           lote, roda em série, como antes;
        3. finalizar (thread principal): ``obter(i)`` repete a lógica do cache
           do ajuste em série — hit usa o cache; o 1º miss emite os avisos de
           falha daquela variável e relança a exceção guardada ou grava o
           resultado (com o backstop de 6000); um 2º miss do mesmo ajuste (o
           backstop limpou o cache no meio do lote) recalcula em série, como o
           ajuste em série faria.

        ``fechar(erro)`` encerra o executor no fim do :meth:`variable_iv`; com
        ``erro`` (ex.: KeyboardInterrupt/Cancel no laço do IV), sem esperar e
        com ``cancel_futures`` — fora do lote não há ajuste em curso (o lote
        sempre termina ou espera os seus)."""
        # getattr: segmentador despicklado de antes do n_jobs (sem o atributo)
        n_jobs = _normaliza_n_jobs(getattr(self, "n_jobs", 1))
        if (n_jobs <= 1 or not isinstance(features, (list, tuple))
                or len(features) < 2):
            def obter_serie(i, feat):
                return self._resolve_bins_cached(sid, feat, None, None,
                                                 max_n_bins, min_bin_size)
            return obter_serie, (lambda erro=False: None)

        nsid = sid if (sid in self.segments) else "root"
        mask = self.segments[nsid]["mask"]
        tam_lote = 2 * n_jobs
        estado = {"ini": None, "itens": None, "ex": None}

        def _executor():
            if estado["ex"] is None:
                estado["ex"] = ThreadPoolExecutor(max_workers=n_jobs,
                                                  thread_name_prefix="ygg-optbin")
            return estado["ex"]

        def _tarefa(val):
            # roda numa thread do executor: só o ajuste (numpy + optbinning)
            falhas = []
            try:
                aj = self._resolve_bins_resolver(
                    val["prep"],
                    ajusta=lambda b, x, y: _fit_splits_na_thread(b, x, y, falhas))
            except Exception as e:
                return falhas, None, e
            return falhas, aj, None

        def _lote(ini):
            itens, pend, preps = [], {}, []
            for feat in features[ini:ini + tam_lote]:
                key = self._bins_key(mask, feat, None, None, max_n_bins, min_bin_size)
                try:
                    hit = self._bins_cache.get(key)
                    repetida = key in pend
                except Exception:
                    # chave não hashável (ex.: lista no lugar do nome): fica fora
                    # do lote — o obter repete a consulta e falha SÓ nesta
                    # variável, como no ajuste em série (antes derrubava o lote)
                    itens.append((key, {"usado": True}))
                    continue
                if hit is not None:
                    itens.append((key, {"usado": True}))
                    continue
                if repetida:                        # variável repetida no lote
                    itens.append((key, pend[key]))
                    continue
                val = {}
                try:
                    prep = self._resolve_bins_preparar(
                        self._sub(mask, feat), feat, None, None, max_n_bins,
                        min_bin_size, amostrar=True)
                except Exception as e:
                    val["exc"] = e
                else:
                    val["prep"] = prep
                    if prep["fit"] is not None:
                        preps.append(val)
                pend[key] = val
                itens.append((key, val))

            if len(preps) == 1:
                # 1 ajuste só: em série, exatamente como o _resolve_bins
                val = preps[0]
                try:
                    val["ajuste"] = self._resolve_bins_resolver(val["prep"])
                except Exception as e:
                    val["exc"] = e
            elif preps:
                ex = _executor()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        saidas = list(ex.map(_tarefa, preps))
                    except BaseException:
                        # KeyboardInterrupt/Cancel: cancela os pendentes e ESPERA os
                        # ajustes em curso (no máximo 1 por worker) ainda DENTRO do
                        # catch_warnings — as dependências do fit (sklearn/pandas)
                        # abrem catch_warnings nos workers; se a thread principal
                        # restaurasse os filtros antes, os ajustes que seguem
                        # despejariam avisos e um worker, ao sair do próprio
                        # catch_warnings, reporia a lista com o "ignore" do lote
                        # (vazaria para a sessão inteira)
                        estado["ex"] = None
                        ex.shutdown(wait=True, cancel_futures=True)
                        raise
                for val, (falhas, aj, exc) in zip(preps, saidas):
                    val["falhas"] = falhas
                    if exc is not None:
                        val["exc"] = exc
                    else:
                        val["ajuste"] = aj
            for val in pend.values():
                if "prep" in val:
                    if "exc" not in val:
                        try:
                            val["res"] = self._resolve_bins_finalizar(
                                val["prep"], val.get("ajuste"))
                        except Exception as e:
                            val["exc"] = e
                    del val["prep"]                 # solta x/y do ajuste
                    val.pop("ajuste", None)
            estado["ini"], estado["itens"] = ini, itens

        def obter(i, feat):
            ini = estado["ini"]
            if ini is None or not (ini <= i < ini + tam_lote):
                _lote(i - i % tam_lote)
                ini = estado["ini"]
            key, val = estado["itens"][i - ini]
            hit = self._bins_cache.get(key)
            if hit is not None:
                return hit[0], hit[1], hit[2]
            if val.get("usado"):
                # miss de novo (era hit no preparo e o backstop limpou o cache, ou
                # variável repetida): o ajuste em série recalcularia — recalcula
                return self._resolve_bins_cached(sid, feat, None, None,
                                                 max_n_bins, min_bin_size)
            val["usado"] = True
            # 1º miss: os avisos "falhou inesperadamente" saem agora, na ordem em
            # que o ajuste em série os emitiria
            for nome, e in val.pop("falhas", ()):
                _fit_optbinning_splits(_RefazFalha(nome, e), np.empty(0), np.empty(0))
            if "exc" in val:
                raise val.pop("exc")
            res = val.pop("res")
            self._bins_cache_put(key, res, mask)
            return res[0], res[1], res[2]

        def fechar(erro=False):
            ex, estado["ex"] = estado["ex"], None
            if ex is not None:
                if erro:
                    ex.shutdown(wait=False, cancel_futures=True)
                else:
                    ex.shutdown(wait=True)

        return obter, fechar

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

        Dividido em preparar → resolver → finalizar: só o resolver (o ajuste do
        optbinning) pode rodar fora da thread principal — ver o lote paralelo do
        :meth:`variable_iv` (``n_jobs``). Aqui os três rodam em sequência, com o
        mesmo resultado de antes.
        """
        prep = self._resolve_bins_preparar(sub, feature, splits, dtype, max_n_bins,
                                           min_bin_size, max_bin_size, relax_max,
                                           min_mean_diff, criterion)
        return self._resolve_bins_finalizar(prep, self._resolve_bins_resolver(prep))

    def _resolve_bins_preparar(self, sub, feature, splits, dtype, max_n_bins,
                               min_bin_size, max_bin_size=None, relax_max=False,
                               min_mean_diff=0.0, criterion="optbin", amostrar=False):
        """1ª etapa do :meth:`_resolve_bins` (thread principal: lê o pandas).

        Decide o tipo, aplica cortes/grupos manuais, o critério alternativo e o
        teste de degenerado. Quando falta o ajuste do optbinning, guarda em
        ``prep['fit']`` só os arrays do ajuste (x/y já filtrados) e os parâmetros —
        nada do recorte ``sub``. ``amostrar=True`` (lote paralelo) já sorteia aqui
        a amostra de ajuste (:func:`_amostra_optbinning`, o MESMO sorteio que o
        ``fit_optbinning_splits`` faria): x/y ficam com ≤ ``OPTBINNING_MAX_ROWS``
        linhas enquanto esperam a vez no lote, e o ajuste dá o mesmo resultado."""
        kind = self._detect_kind(sub, feature, dtype)
        prep = {"kind": kind, "fit": None}

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
                # classificação exige as 2 classes presentes; regressão não.
                # "Menos de 2 distintos" por min == max (O(n), sem ordenar como o
                # np.unique): sem NaN em x_obs/yfit, equivale a unique().size < 2
                # (inclusive -0.0 == 0.0 e ±inf). O curto-circuito protege o min()
                # de array vazio. Nunca max-min == 0: inf - inf dá NaN.
                degenerado = (len(yfit) < 4 or x_obs.size == 0
                              or x_obs.min() == x_obs.max()
                              or (self._is_clf and yfit.min() == yfit.max()))
                if degenerado:
                    cortes = []                            # dados degenerados → sem corte
                elif criterion != "optbin":
                    min_n = max(1, int(min_bin_size * len(yfit)))
                    t = _best_numeric_cut(x, yfit, criterion, self._is_clf, min_n)
                    cortes = [t] if t is not None else []
                else:
                    cortes = None                          # pendente: ajuste no resolver
                    if amostrar:
                        idx = _amostra_optbinning(len(x))
                        if idx is not None:
                            x, yfit = x[idx], yfit[idx]
                    prep["fit"] = {"feature": feature, "dtype": "numerical",
                                   "x": x, "y": yfit, "max_n_bins": max_n_bins,
                                   "min_bin_size": min_bin_size,
                                   "max_bin_size": max_bin_size, "relax_max": relax_max,
                                   "min_mean_diff": min_mean_diff}
                modo = "ótimo" if criterion == "optbin" else criterion
            prep["cortes"], prep["modo"] = cortes, modo
            # a bin de faltantes só entra com corte — ou com ajuste pendente (o
            # isna é puro: lido aqui, o resultado é o mesmo de lê-lo no fim)
            prep["na"] = bool(sub[feature].isna().any()) if (cortes is None or cortes) else False
            return prep

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
            # ≥2 textos distintos por comparação com o 1º (O(n), sem ordenar
            # strings Python como o np.unique); len(ys) < 4 vem antes e garante
            # xs[0]. Sempre sobre xs JÁ em str: factorize/nunique na coluna bruta
            # fundiria 1/1.0/True e Decimal('1.0')/Decimal('1.00').
            degenerado = (len(ys) < 4 or not (xs[1:] != xs[0]).any()
                          or (self._is_clf and ys.min() == ys.max()))
            if degenerado:
                grupos = []                          # dados degenerados → sem grupos
            elif criterion != "optbin":
                min_n = max(1, int(min_bin_size * len(ys)))
                par = _best_categorical_split(xs, ys, criterion, self._is_clf, min_n)
                grupos = [list(par[0]), list(par[1])] if par is not None else []
            else:
                grupos = None                        # pendente: ajuste no resolver
                if amostrar:
                    idx = _amostra_optbinning(len(xs))
                    if idx is not None:
                        xs, ys = xs[idx], ys[idx]
                prep["fit"] = {"feature": feature, "dtype": "categorical",
                               "x": xs, "y": ys, "max_n_bins": max_n_bins,
                               "min_bin_size": min_bin_size,
                               "max_bin_size": max_bin_size, "relax_max": relax_max,
                               "min_mean_diff": min_mean_diff}
            modo = "ótimo" if criterion == "optbin" else criterion
        prep.update(grupos=grupos, modo=modo, na_present=na_present)
        return prep

    def _resolve_bins_resolver(self, prep, ajusta=None):
        """2ª etapa do :meth:`_resolve_bins`: o ajuste do optbinning (o gargalo).

        Só numpy + optbinning — não lê o pandas nem escreve em ``self`` —, então
        pode rodar numa thread do lote paralelo. Devolve os splits do optbinning
        (lista, possivelmente vazia) ou ``None`` quando ``prep`` não tem ajuste
        pendente. ``ajusta`` (padrão: ``fit_optbinning_splits``, resolvido na
        hora da chamada) faz o ``b.fit`` e devolve ``list(b.splits)``."""
        f = prep["fit"]
        if f is None:
            return None
        if ajusta is None:
            ajusta = _fit_optbinning_splits
        if f["dtype"] == "numerical":
            def _opt(mnb, mbs):
                b = self._OptBin(
                    name=f["feature"], dtype="numerical", max_n_bins=mnb,
                    min_bin_size=f["min_bin_size"], max_bin_size=mbs,
                    monotonic_trend="auto_asc_desc",
                    **{self._diff_kwarg: f["min_mean_diff"]})
                return ajusta(b, f["x"], f["y"])
            max_n_bins, max_bin_size = f["max_n_bins"], f["max_bin_size"]
            cortes = _opt(max_n_bins, max_bin_size)
            if not cortes and max_bin_size is not None and f["relax_max"]:
                # o máximo pode deixar o problema INFEASIBLE p/ este nº de
                # bins/amostra: tenta mais bins e, por fim, relaxa o máximo
                for k in range(max_n_bins + 1, 9):
                    cortes = _opt(k, max_bin_size)
                    if cortes:
                        break
                if not cortes:
                    cortes = _opt(max_n_bins, None)
            return cortes
        b = self._OptBin(
            name=f["feature"], dtype="categorical", max_n_bins=f["max_n_bins"],
            min_bin_size=f["min_bin_size"], max_bin_size=f["max_bin_size"],
            monotonic_trend="auto_asc_desc",
            **{self._diff_kwarg: f["min_mean_diff"]})
        return ajusta(b, f["x"], f["y"])

    @staticmethod
    def _resolve_bins_finalizar(prep, ajuste):
        """3ª etapa do :meth:`_resolve_bins` (thread principal): cortes → faixas
        (+ bin de faltantes) ou grupos → faixas (sem os rótulos de NA)."""
        kind, modo = prep["kind"], prep["modo"]
        if kind == "num":
            cortes = prep["cortes"] if prep["fit"] is None else ajuste
            if not cortes:
                return [], modo, kind
            edges = [-np.inf, *cortes, np.inf]
            bins = [{"kind": "num", "lo": edges[i], "hi": edges[i + 1]}
                    for i in range(len(edges) - 1)]
            if prep["na"]:
                bins.append({"kind": "na"})
            return bins, modo, kind

        # ---- categórico ----
        if prep["fit"] is None:
            grupos = prep["grupos"]
        else:
            grupos = [list(arr) for arr in ajuste]
        _NA_TOK = {"nan", "NaN", "<NA>", "None"}
        bins = []
        for g in grupos:
            cats = [str(c) for c in g if str(c) not in _NA_TOK]
            if cats:
                bins.append({"kind": "cat", "cats": cats})
        if bins and prep["na_present"]:
            bins.append({"kind": "na"})
        return bins, modo, kind

    # ------------------------------------------------------------------
    # Tabela de bins: taxa de default (alvo) + representatividade (num ou cat)
    # ------------------------------------------------------------------
    def _bin_table(self, sub, feature, bins, n_ref):
        linhas, is_na = [], []
        # código de faixa por linha calculado UMA vez (ids == k é a mesma máscara do
        # _mask_in); sem ele, o _mask_in por faixa de antes
        ids = self._bin_ids(sub[feature], bins) if bins else None
        for k, b in enumerate(bins):
            m = self._mask_in(sub, feature, b) if ids is None else (ids == k)
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
            sub = self._sub(seg["mask"], feature)
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
        ref = (self._smask(self.ref_sample).to_numpy()
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
        # A versão só sobe na PRIMEIRA mutação real (logo antes do 1º
        # is_leaf=False), não no início: um grow que não divide nada (variável
        # constante, sem corte válido, 0/1 filho) deixa a árvore idêntica e não
        # deve invalidar os agregados memoizados. Bump antes da escrita (e não
        # "só no fim") mantém o cache coerente mesmo com exceção numa folha
        # seguinte de um grow com várias folhas-alvo.
        mudou = False
        targets = {
            sid: s for sid, s in self.segments.items()
            if s["is_leaf"] and (only_segments is None or sid in only_segments)
        }
        novos, modo_usado = {}, None
        for sid, seg in targets.items():
            sub = self._sub(seg["mask"])
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
            if not mudou:          # 1ª mutação real → invalida agregados memoizados
                self._bump_version()
                mudou = True
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
    # MOVE_CUT: move o corte numérico entre a folha e a irmã adjacente à
    #   DIREITA (mesma variável, mesmo pai) SEM recolher o pai: o `hi` da
    #   folha e o `lo` da irmã passam a ser o novo corte. A irmã pode ser um
    #   NÓ INTERNO (com sub-splits): cada descendente carrega a condição do
    #   pai no caminho, então o novo limite é propagado às conditions de TODO
    #   o subtree dela e as máscaras são recalculadas.
    # ------------------------------------------------------------------
    def _resolve_move_cut(self, sid: str, new_threshold):
        """Valida o :meth:`move_cut` e localiza a irmã adjacente à direita.
        Devolve ``(seg, sib_id, t, cond, cond_sib)``; levanta ``ValueError``
        quando a folha não veio de split numérico, é a última faixa da variável
        ou o novo corte está fora do intervalo válido — estritamente entre o
        ``lo`` da folha e o ``hi`` da irmã (exclusivo nos dois extremos)."""
        if sid not in self.segments:
            raise ValueError(f"segmento '{sid}' não existe.")
        seg = self.segments[sid]
        if not seg["is_leaf"]:
            raise ValueError(f"'{sid}' não é folha — selecione a folha à esquerda do corte.")
        if not seg["conditions"] or seg["conditions"][-1]["kind"] != "num":
            raise ValueError("mover corte só vale para folha originada de split numérico.")
        cond = seg["conditions"][-1]
        feat, hi = cond["feature"], cond["hi"]
        if not np.isfinite(hi):
            raise ValueError(
                f"a folha é a última faixa de '{feat}' (sem corte à direita) — "
                "selecione a folha à ESQUERDA do corte que quer mover.")
        pai = seg["parent"]
        sib_id = None
        for c, s in self.segments.items():
            if c == sid or s["parent"] != pai or not s["conditions"]:
                continue
            lc = s["conditions"][-1]
            lo_c = lc.get("lo")
            if lc["kind"] == "num" and lc["feature"] == feat and lo_c is not None \
                    and (lo_c == hi or np.isclose(lo_c, hi, rtol=1e-12, atol=0.0)):
                sib_id = c
                break
        if sib_id is None:
            raise ValueError(
                f"não há irmã adjacente à direita do corte {_fmt(hi)} em '{feat}'.")
        cond_sib = self.segments[sib_id]["conditions"][-1]
        try:
            t = float(new_threshold)
        except (TypeError, ValueError):
            raise ValueError(f"novo corte inválido: {new_threshold!r}.") from None
        if not np.isfinite(t):
            raise ValueError(f"novo corte inválido: {new_threshold!r}.")
        if not (cond["lo"] < t < cond_sib["hi"]):
            raise ValueError(
                f"novo corte fora do intervalo válido: {_fmt(t)} deve ficar entre "
                f"{_fmt(cond['lo'])} e {_fmt(cond_sib['hi'])} (exclusivo).")
        return seg, sib_id, t, cond, cond_sib

    def movable_cut(self, sid: str) -> dict | None:
        """Corte numérico MÓVEL da folha ``sid`` (o ``hi`` dela, compartilhado
        com a irmã adjacente à direita) — ``None`` quando a folha não veio de
        split numérico ou é a última faixa da variável. Devolve
        ``{'feature','cut','lo','hi_sib','sibling'}``: o corte vigente, os
        limites do intervalo VÁLIDO (exclusivo) de :meth:`move_cut` e o sid da
        irmã. Pronto para a UI preencher o campo editável."""
        seg = self.segments.get(sid)
        if seg is None or not seg["is_leaf"] or not seg["conditions"]:
            return None
        cond = seg["conditions"][-1]
        if cond["kind"] != "num" or not np.isfinite(cond["hi"]):
            return None
        try:
            _, sib_id, _, cond, cond_sib = self._resolve_move_cut(sid, cond["hi"])
        except ValueError:
            return None
        return {"feature": cond["feature"], "cut": float(cond["hi"]),
                "lo": float(cond["lo"]), "hi_sib": float(cond_sib["hi"]),
                "sibling": sib_id}

    def preview_move_cut(self, sid: str, new_threshold) -> pd.DataFrame:
        """PREVIEW do :meth:`move_cut` (não altera estado): ``n`` e alvo médio —
        total e por amostra, quando houver — dos DOIS lados do corte proposto:
        a folha (esquerda) e a irmã à direita (o subtree inteiro dela). Levanta
        ``ValueError`` se o corte for inválido (ver :meth:`_resolve_move_cut`)."""
        seg, sib_id, t, cond, cond_sib = self._resolve_move_cut(sid, new_threshold)
        linhas = []
        for nome, base, c_new in (
                ("folha (esq.)", seg, {**cond, "hi": t}),
                ("irmã (dir.)", self.segments[sib_id], {**cond_sib, "lo": t})):
            conds = base["conditions"][:-1] + [c_new]
            sub = self.df[_match_conditions_pandas(self.df, conds)]
            s_t = sub[self.target]
            row = {"lado": f"{nome} {self._bin_label_short(c_new)}",
                   "n": int(len(sub)),
                   "valor_medio": round(float(s_t.mean()), 4) if len(s_t) else np.nan}
            if self.sample_col is not None:
                for amostra in self._samples:
                    s_am = sub.loc[sub[self.sample_col] == amostra, self.target]
                    row[f"n_{amostra}"] = int(len(s_am))
                    row[f"valor_{amostra}"] = (round(float(s_am.mean()), 4)
                                               if len(s_am) else np.nan)
            linhas.append(row)
        return pd.DataFrame(linhas)

    def move_cut(self, sid: str, new_threshold, verbose: bool = True):
        """Move o corte numérico entre a folha ``sid`` e a irmã adjacente à
        direita (mesma variável, mesmo pai) sem recolher o pai: o ``hi`` da
        folha e o ``lo`` da irmã passam a ser ``new_threshold`` — que deve
        ficar ESTRITAMENTE entre o ``lo`` da folha e o ``hi`` da irmã.

        A irmã pode ser nó INTERNO: o novo limite é propagado às conditions de
        TODO o subtree dela (cada descendente carrega a condição do pai no
        caminho) e as máscaras são recalculadas. A operação é ATÔMICA: se algum
        segmento afetado ficar sem linhas com o novo corte, nada muda e um
        ``ValueError`` é levantado. Rótulos/ids afetados são reescritos e os
        apelidos de negócio acompanham a renomeação."""
        seg, sib_id, t, cond, cond_sib = self._resolve_move_cut(sid, new_threshold)
        if t == cond["hi"]:
            if verbose:
                print(f"[move_cut] novo corte igual ao vigente ({_fmt(t)}) — nada a mudar")
            return self
        feat, hi_antigo = cond["feature"], cond["hi"]
        pai = seg["parent"]
        pai_seg = self.segments[pai]

        # novas condições dos dois lados (preservam o include_na de cada um)
        cond_leaf_new = dict(cond); cond_leaf_new["hi"] = t
        cond_sib_new = dict(cond_sib); cond_sib_new["lo"] = t

        # rótulo novo preserva o sufixo de faltantes quando o lado já o exibia
        def _label(c, old_label):
            suf = " + faltante" if old_label.endswith(" + faltante") else ""
            return self._bin_label(feat, c) + suf

        # ---- STAGING (nada é mutado ainda): folha + subtree da irmã ----------
        # subtree em ordem pai-antes-do-filho (BFS) p/ reconstruir path/conditions
        # de cada descendente a partir do pai já reconstruído
        subtree, frente = [sib_id], [sib_id]
        while frente:
            atual = frente.pop(0)
            filhos = [c for c, s in self.segments.items() if s["parent"] == atual]
            subtree.extend(filhos)
            frente.extend(filhos)

        novos: dict[str, dict] = {}
        id_map: dict[str, str] = {}
        new_leaf_label = _label(cond_leaf_new, seg["label"])
        new_leaf_id = new_leaf_label if pai == "root" else f"{pai} | {new_leaf_label}"
        conds_leaf = pai_seg["conditions"] + [cond_leaf_new]
        novos[new_leaf_id] = {
            "mask": self._mask_from_conds(self._conditions_json(conds_leaf), conds_leaf),
            "label": new_leaf_label, "depth": seg["depth"], "is_leaf": True,
            "path": pai_seg["path"] + [new_leaf_label], "parent": pai,
            "conditions": conds_leaf,
        }
        id_map[sid] = new_leaf_id
        for old in subtree:
            s = self.segments[old]
            if old == sib_id:
                own_cond = cond_sib_new
                label = _label(cond_sib_new, s["label"])
                parent_new, parent_seg = pai, pai_seg
            else:
                own_cond = dict(s["conditions"][-1])   # cópia: quebra o compartilhamento
                label = s["label"]
                parent_new = id_map[s["parent"]]
                parent_seg = novos[parent_new]
            new_id = label if parent_new == "root" else f"{parent_new} | {label}"
            conds = parent_seg["conditions"] + [own_cond]
            novos[new_id] = {
                "mask": self._mask_from_conds(self._conditions_json(conds), conds),
                "label": label, "depth": s["depth"], "is_leaf": s["is_leaf"],
                "path": parent_seg["path"] + [label], "parent": parent_new,
                "conditions": conds,
            }
            id_map[old] = new_id

        # atomicidade: nenhum segmento afetado pode ficar vazio com o novo corte
        # (ex.: o corte cruzou um sub-split da irmã na MESMA variável)
        vazios = [nid for nid, s in novos.items() if int(s["mask"].sum()) == 0]
        if vazios:
            raise ValueError(
                f"novo corte {_fmt(t)} deixaria segmento(s) sem linhas: "
                f"{vazios}. Recolha o sub-split da irmã ou escolha outro valor.")

        # ---- COMMIT ----------------------------------------------------------
        self._bump_version()      # estrutura vai mudar → invalida agregados memoizados
        for old in [sid, *subtree]:
            self.segments.pop(old, None)
        self.segments.update(novos)
        # apelidos de negócio acompanham a renomeação (a folha é conceitualmente
        # a MESMA — só o limite mudou; sem isto o apelido viraria órfão)
        for old, novo_id in id_map.items():
            if old in self.leaf_names and old != novo_id:
                self.leaf_names[novo_id] = self.leaf_names.pop(old)
        if verbose:
            n_desc = len(subtree) - 1
            extra = (f" (+{n_desc} segmento(s) do subtree da irmã atualizado(s))"
                     if n_desc else "")
            print(f"[move_cut] corte de '{feat}' movido: {_fmt(hi_antigo)} → "
                  f"{_fmt(t)}{extra}")
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

    # ------------------------------------------------------------------
    # PESOS DE EXPOSIÇÃO (weight_col) — visão dupla contratos × saldo.
    #   FASE 1: exibição apenas — split/binning seguem NÃO ponderados.
    # ------------------------------------------------------------------
    def _wmean(self, mask) -> float:
        """Alvo médio PONDERADO pelos pesos nas linhas de ``mask``. Linhas com
        alvo ou peso faltante ficam fora; ``NaN`` sem par (alvo, peso) válido."""
        y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
        w = self._weights[mask].to_numpy(dtype="float64")
        ok = ~np.isnan(y) & ~np.isnan(w)
        y, w = y[ok], w[ok]
        if not len(y) or w.sum() <= 0:
            return float("nan")
        return float(np.average(y, weights=w))

    def weight_share(self, sid: str) -> float:
        """% do SALDO da carteira no segmento ``sid``: soma dos pesos
        (``weight_col``) das linhas do segmento ÷ soma total, ×100. ``NaN`` sem
        ``weight_col`` configurada ou com soma total nula."""
        if self.weight_col is None:
            return float("nan")
        total = float(self._weights.sum())
        if not total:
            return float("nan")
        return float(100.0 * self._weights[self.segments[sid]["mask"]].sum() / total)

    def weighted_value(self, sid: str, sample: str | None = None) -> float:
        """Alvo médio PONDERADO pelos pesos (``weight_col``) no segmento ``sid``,
        restrito à amostra ``sample`` quando informada. ``NaN`` sem
        ``weight_col`` ou sem par (alvo, peso) válido no recorte."""
        if self.weight_col is None:
            return float("nan")
        mask = self.segments[sid]["mask"]
        if sample is not None and sample in self._sample_masks:
            mask = mask & self._sample_masks[sample]
        return self._wmean(mask)

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
        # Composição sobre a tabela BASE memoizada: _compute_leaves monta as
        # colunas base sem ler with_psi/with_test/test e só no fim acrescenta PSI
        # e teste de adjacência — refazer as folhas inteiras por combinação de
        # parâmetros (placar, tabela de folhas) era redundante. A base usa a MESMA
        # chave que leaves()/_grade_map já preenchem; 'test' só entra na chave com
        # with_test (senão é ignorado), para não duplicar a base com 'welch'.
        test_k = test if with_test else "mannwhitney"
        base_key = ("leaves", ascending, False, False, "mannwhitney")
        key = ("leaves", ascending, with_psi, with_test, test_k)

        def _composta():
            # _agg_memo direto (não via leaves(), que inseriria 'apelido') e sobre
            # a CÓPIA que ele devolve: o objeto em cache da base nunca é mutado.
            # Mesma ordem de _compute_leaves: base → PSI → teste de adjacência.
            out = self._agg_memo(base_key, lambda: self._compute_leaves(ascending))
            if with_psi and self.sample_col is not None:
                out = self._append_psi_cols(out)
            if with_test:
                out = self._append_adjacency_test(out, test=test)
            return out

        if key == base_key:
            out = self._agg_memo(base_key, lambda: self._compute_leaves(ascending))
        else:
            out = self._agg_memo(key, _composta)
        # coluna 'apelido' SÓ quando há apelido definido — e FORA do memo: os
        # apelidos mudam sem alterar a versão da árvore, então não podem ficar
        # presos no cache por versão (o memo devolve cópia defensiva).
        nomes = self._leaf_names_validos()
        if nomes:
            out.insert(1, "apelido", out["segmento"].map(lambda s: nomes.get(s, "")))
        return out

    def _compute_leaves(self, ascending: bool = True, with_psi: bool = False,
                        with_test: bool = False, test: str = "mannwhitney") -> pd.DataFrame:
        linhas, n_total = [], len(self.df)
        # total de SALDO (soma dos pesos) p/ a visão dupla contratos × saldo
        w_total = float(self._weights.sum()) if self.weight_col is not None else 0.0
        amostras = list(self._sample_masks) if self.sample_col is not None else []
        # código da amostra por linha (pré-computado): o alvo por amostra DENTRO
        # da folha vira comparação de int, sem recomparar a coluna (object) da
        # amostra a cada folha×amostra. Sem os códigos, o caminho antigo.
        sc = self._sample_codes if self.sample_col is not None else None
        k_am = {a: j for j, a in enumerate(self._samples)}

        def _alvo_am(sub, sc_sub, amostra):
            if sc_sub is not None:
                return sub.loc[sc_sub == k_am[amostra], self.target]
            return sub.loc[sub[self.sample_col] == amostra, self.target]

        for sid, seg in self.segments.items():
            if not seg["is_leaf"]:
                continue
            sub = self._sub(seg["mask"])
            # códigos das linhas de `sub`, na MESMA ordem: sub = df.loc[máscara]
            # é recorte posicional (máscara com o índice do df), então o gather
            # pela máscara numpy alinha linha a linha (vale com índice duplicado)
            m_np = seg["mask"].values
            sc_sub = (sc[m_np] if (sc is not None and isinstance(m_np, np.ndarray)
                                   and m_np.dtype == bool) else None)
            # valor_medio na referência (DES) = base da régua; assim nota é
            # monotônica no alvo que entra em predict/apply_spark (com fallback)
            if self.sample_col is not None:
                ref_t = _alvo_am(sub, sc_sub, self.ref_sample)
                pd_m = ref_t.mean() if len(ref_t) else sub[self.target].mean()
            else:
                pd_m = sub[self.target].mean()
            row = {
                "segmento": sid,
                "descricao": self._descrever(seg["conditions"]),
                "profundidade": seg["depth"],
                "n": len(sub),
                "repr_%": round(100 * len(sub) / n_total, 1) if n_total else 0.0,
            }
            # visão dupla (weight_col): % do SALDO da folha, ao lado do % de
            # contratos — as colunas SÓ existem quando weight_col foi definida
            if self.weight_col is not None:
                w_leaf = float(self._weights[seg["mask"]].sum())
                row["saldo_%"] = round(100 * w_leaf / w_total, 1) if w_total else 0.0
            row["valor_medio"] = round(pd_m, 4) if not pd.isna(pd_m) else np.nan
            if self.weight_col is not None:
                # alvo médio PONDERADO na referência (mesma base do valor_medio)
                if self.sample_col is not None:
                    vp = self._wmean(seg["mask"] & self._mask_ref)
                    if pd.isna(vp):
                        vp = self._wmean(seg["mask"])
                else:
                    vp = self._wmean(seg["mask"])
                row["valor_medio_pond"] = round(vp, 4) if not pd.isna(vp) else np.nan
            row["valor_std"] = round(sub[self.target].std(), 4)
            if self.sample_col is not None:
                for amostra in amostras:
                    s_am = _alvo_am(sub, sc_sub, amostra)
                    row[f"valor_{amostra}"] = round(s_am.mean(), 4) if len(s_am) else np.nan
                    if self.weight_col is not None:   # alvo PONDERADO por amostra
                        vpa = self._wmean(seg["mask"] & self._sample_masks[amostra])
                        row[f"valor_pond_{amostra}"] = (round(vpa, 4)
                                                        if not pd.isna(vpa) else np.nan)
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
        # Vetorizado: uma única contagem (folha, amostra) por bincount substitui
        # os laços (mask_folha & mask_amostra).sum() full-length por folha×amostra.
        # O denominador continua o tamanho TOTAL da amostra (máscaras
        # pré-computadas). leaf_ids vem ordenado por nota: a contagem é lida
        # por sid (dicionário), não por posição.
        leaf_ids = out["segmento"].tolist()
        cnt = self._leaf_sample_counts()

        def fracs(amostra):
            n = float(self._sample_masks[amostra].sum())
            return {sid: (float(cnt(sid, amostra)) / n if n else 0.0)
                    for sid in leaf_ids}

        ref_pct = fracs(self.ref_sample)
        # representatividade (% da folha DENTRO da amostra) — começa pela referência;
        # as demais amostras saem no laço abaixo, ao lado do respectivo PSI.
        out[f"repr_{self.ref_sample}_%"] = [round(100 * ref_pct[sid], 1)
                                            for sid in leaf_ids]
        for amostra in self._samples:
            if amostra == self.ref_sample:
                continue
            cur_pct = fracs(amostra)
            _, contribs = _psi_from_shares([ref_pct[sid] for sid in leaf_ids],
                                           [cur_pct[sid] for sid in leaf_ids],
                                           eps, return_contrib=True)
            out[f"psi_{amostra}"] = [round(c, 4) for c in contribs]
            out[f"repr_{amostra}_%"] = [round(100 * cur_pct[sid], 1)
                                        for sid in leaf_ids]
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
            self.__dict__.get("_pair_p_cache", {}).clear()
        self._leaf_target_cache[ck] = (vals, mask)   # guarda mask p/ o id não reciclar
        return vals

    # p-valor do teste de igualdade de alvo entre duas folhas (na referência)
    def _pair_pvalue(self, sid_a: str, sid_b: str, test: str = "mannwhitney",
                     min_n: int = 8) -> float:
        if mannwhitneyu is None:        # scipy ausente (importado no topo do módulo)
            return np.nan
        # memo por (máscara a, máscara b, teste, min_n): o p-valor é função pura
        # dos alvos DES das duas folhas, e a máscara é TROCADA sempre que a folha é
        # refeita — o cabeçalho/canvas da UI e o prune/auto_merge repetem os
        # mesmos pares várias vezes por rodada. O par NÃO é ordenado (o teste
        # recebe a e b nessa ordem) e as duas máscaras ficam no valor, conferidas
        # com `is`, para o id não ser reciclado.
        ma, mb = self.segments[sid_a]["mask"], self.segments[sid_b]["mask"]
        cache = self.__dict__.setdefault("_pair_p_cache", {})
        key = (id(ma), id(mb), test, min_n)
        hit = cache.get(key)
        if hit is not None and hit[0] is ma and hit[1] is mb:
            return hit[2]
        a, b = self._leaf_target(sid_a), self._leaf_target(sid_b)
        if len(a) < min_n or len(b) < min_n:
            p = np.nan
        else:
            try:
                if test == "welch":
                    p = float(ttest_ind(a, b, equal_var=False).pvalue)
                else:
                    p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
            except Exception:
                p = np.nan
        if len(cache) > 4000:                    # backstop de memória
            cache.clear()
        cache[key] = (ma, mb, p)
        return p

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
        Memoizada por versão da árvore.

        Derivada PREGUIÇOSAMENTE do código int32 da folha (``_leaf_codes``): o
        array object só é montado quando alguém pede os rótulos, e fica no cache
        read-only. Cada chamada devolve um wrapper Series NOVO sobre esse array
        (O(1), sem a cópia profunda de N ponteiros); só leia — escrever nos
        valores levanta ValueError."""
        ids, code = self._leaf_codes(leaf_ids)

        def build():
            # rótulo por código; o código -1 (órfã) indexa a ÚLTIMA posição = NA
            lab = np.empty(len(ids) + 1, dtype=object)
            for k, sid in enumerate(ids):
                lab[k] = sid
            lab[len(ids)] = pd.NA
            arr = lab[code]
            arr.flags.writeable = False
            return arr
        # a identidade do conjunto entra na chave: uma chamada com subconjunto de
        # folhas não pode devolver a série do conjunto completo do cache. None
        # vira a tupla das folhas na ordem de segments (uma entrada só).
        arr = self._agg_memo(("leaf_id_arr", ids), build)
        return pd.Series(arr, index=self.df.index, dtype=object, copy=False)

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

        y_all = self.df[self.target]

        def stats(sid):
            sub = y_all[self.segments[sid]["mask"]]      # só o alvo (não a base toda)
            pdv = sub.mean() if len(sub) else float("nan")
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
            sub = self._sub(m)                   # alvo + amostra (não a base toda)
            if ref is not None:
                sr = sub.loc[sub[self.sample_col] == ref, self.target]
                pdv = sr.mean() if len(sr) else (sub[self.target].mean() if n else float("nan"))
            else:
                pdv = sub[self.target].mean() if n else float("nan")
            return n, (100 * n / n_total if n_total else 0.0), pdv

        def sample_value(sid, a):
            m = self.segments[sid]["mask"] & self._smask(a)
            sub = self.df.loc[m, self.target]
            return sub.mean() if len(sub) else float("nan")

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
                amostras = list(self._samples)
                metr += "\n" + " | ".join(f"{a} {sample_value(sid, a) * 100:.2f}%"
                                          for a in amostras)
            t_metr = ax.text(x, y - 0.42 * bh, metr, ha="center", va="center",
                             fontsize=base_fs - 0.8, color=txt_color, zorder=3,
                             linespacing=1.12, clip_on=True)
            fit_items.append((t_metr, bw * 0.94, bh * 0.34))
            # 3) número da folha (+ apelido de negócio, quando definido) no
            #    CANTO INFERIOR DIREITO
            if is_leaf:
                ax.text(x + bw * 0.93, y - bh * 0.9,
                        self._leaf_tag(sid, nota_map.get(sid, "?"), max_len=14),
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
        # ``get_renderer()`` (Agg) basta para medir texto — não precisa rasterizar a
        # figura inteira com ``canvas.draw()``. Evitar esse render completo corta um
        # terço do custo de plot_tree/plot_tree_hitmap em árvores grandes; as
        # extents medidas são idênticas às do draw (só afetam o fontsize, não a
        # geometria das caixas).
        try:
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
        # capa a largura do PNG do preview (ver _PREVIEW_MAX_PX): reduz o dpi efetivo
        # quando a árvore é larga demais, mantendo a geometria (o hit-map abaixo
        # escala junto) e um piso de legibilidade — o encode do PNG deixa de dominar.
        w_in = fig.get_size_inches()[0]
        if w_in > 0 and w_in * fig.dpi > self._PREVIEW_MAX_PX:
            fig.set_dpi(max(self._PREVIEW_MIN_DPI, self._PREVIEW_MAX_PX / w_in))
        canvas = fig.canvas
        # sem canvas.draw() aqui: as caixas (patches) têm extent estável e
        # get_renderer() já as posiciona; o único render real é o savefig abaixo.
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

        def build():
            # o detalhe por folha (efeito colateral de _compute_psi) vai JUNTO
            # no valor memoizado, como cópia: um hit (inclusive da LRU, noutro
            # estado já visto) reatribui o detalhe do próprio resultado
            out = self._compute_psi(eps)
            return out, [dict(d) for d in self._psi_detalhe]
        out, det = self._agg_memo(("psi", round(float(eps), 12)), build)
        # psi_detalhe() sempre corresponde ao ÚLTIMO psi() chamado. Antes só um
        # recálculo o trocava: psi(eps=a) → psi(eps=b) → psi(eps=a) devolvia o
        # detalhe de b no último passo (corrigido aqui — não é paridade bit a
        # bit com o antigo nesse caso de eps alternado na mesma versão).
        self._psi_detalhe = [dict(d) for d in det]
        return out.copy()

    def _compute_psi(self, eps: float = 1e-6) -> pd.DataFrame:
        leaf_ids = [sid for sid, s in self.segments.items() if s["is_leaf"]]
        if not leaf_ids:
            raise ValueError("Nenhuma folha — cresça a segmentação antes do PSI.")

        # distribuição (% por folha) por amostra via UM bincount de códigos
        # inteiros (folha × amostra) — as mesmas contagens do antigo crosstab de
        # duas colunas object. Denominador = tamanho TOTAL da amostra (igual ao
        # código original), via máscaras de amostra pré-computadas.
        cnt = self._leaf_sample_counts(leaf_ids)
        dist = {}
        for amostra in self._samples:
            n_am = float(self._sample_masks[amostra].sum())
            dist[amostra] = {
                sid: (float(cnt(sid, amostra)) / n_am if n_am else 0.0)
                for sid in leaf_ids
            }

        ref = dist[self.ref_sample]
        self._psi_detalhe = []
        linhas = []
        for amostra, pct in dist.items():
            if amostra == self.ref_sample:
                continue
            psi_total, contribs = _psi_from_shares(
                [ref[sid] for sid in leaf_ids], [pct[sid] for sid in leaf_ids],
                eps, return_contrib=True)
            for sid, contrib in zip(leaf_ids, contribs):
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

        nonref = [a for a in self._samples
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
            # Com código de faixa por linha (uma passada na coluna), as contagens
            # amostra × faixa saem de um bincount sobre as linhas da amostra.
            ids = self._bin_ids(self.df[feat], bins)
            if ids is not None and not all(m.dtype == bool for m in (
                    ref_mask, *(self._sample_masks[a] for a in nonref))):
                ids = None                       # máscara anulável: caminho antigo
            if ids is None:
                bin_masks = [self._mask_in(self.df, feat, b) for b in bins]

            def _cnt_csi(msk):
                if ids is None:
                    return [int((bm & msk).sum()) for bm in bin_masks]
                ia = ids[msk.to_numpy()]
                return [int(c) for c in np.bincount(ia[ia >= 0], minlength=len(bins))]
            ref_pct = [(c / n_ref) for c in _cnt_csi(ref_mask)]
            pior = 0.0
            for a in nonref:
                a_mask = self._sample_masks[a]
                n_a = int(a_mask.sum())
                cur_pct = [(c / n_a if n_a else 0.0) for c in _cnt_csi(a_mask)]
                csi_tot, contribs = _psi_from_shares(ref_pct, cur_pct, eps,
                                                     return_contrib=True)
                for b, p_ref, p_cur, contrib in zip(bins, ref_pct, cur_pct,
                                                    contribs):
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
        leaf_ids = [sid for sid, s in self.segments.items() if s["is_leaf"]]
        if self._is_clf:
            # classificação: caminho por contagens folha×amostra×classe (mesmo
            # resultado, sem ordenar as N linhas por amostra). None = alguma
            # pré-condição não vale → caminho antigo, literal, logo abaixo.
            rapido = self._metrics_clf_contagens(cutoff, leaf_ids)
            if rapido is not None:
                return rapido
            from yggdrasil.metrics.classification import classification_metrics

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
            todas = list(self._samples)
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

    # dtypes do alvo em que o groupby.mean do caminho antigo acumula e divide em
    # float64 (0/1 → soma inteira exata) — aí pos/(pos+neg) é o MESMO float.
    # float32/Float32 dividem em float32; object/pyarrow têm outro caminho.
    _METRICS_DTYPES_ANULAVEIS = (
        pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype,
        pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, pd.UInt64Dtype,
        pd.Float64Dtype,
    )

    def _metrics_clf_contagens(self, cutoff, leaf_ids):
        """``metrics()`` da classificação a partir de contagens por folha.

        O score de uma linha é o pd_map da sua folha, então dentro de cada
        (folha, amostra) só importa quantos 1 e quantos 0 há. Um bincount
        folha×amostra×classe (``_metrics_clf_tabela``) substitui o groupby/map
        por linha, e as métricas saem de 2L linhas sintéticas (y=1 com peso pos,
        y=0 com peso neg) com ``sample_weight`` no roc_curve/accuracy/f1 do
        sklearn: as somas cumulativas ponderadas são os MESMOS inteiros do
        caminho por linha, logo fpr/tpr/limiares/AUC/KS/corte e acurácia/F1
        saem iguais bit a bit.

        Devolve ``None`` (o chamador usa o caminho antigo, literal) quando a
        tabela não pode ser montada (ver ``_metrics_clf_tabela``) ou em qualquer
        exceção no cálculo (o caminho antigo reproduz a mesma)."""
        tab = self._metrics_clf_tabela(leaf_ids)
        if tab is None:
            return None
        try:
            return self._metrics_clf_de_tabela(tab, cutoff)
        except Exception:
            return None

    def _metrics_clf_tabela(self, leaf_ids):
        """``(ids, pos, neg, score, coluna)`` do caminho por contagens, ou
        ``None`` quando alguma pré-condição não vale.

        ``pos``/``neg``: contagem de alvo 1/0 por folha (linha, na ordem de
        ``ids``) × amostra (coluna ``coluna[a]``), com uma coluna EXTRA no fim
        para as linhas SEM amostra (sample_col NaN/None/pd.NA).
        ``score``: o pd_map do caminho antigo, folha a folha, no MESMO float.

        ``None`` quando: códigos de amostra indisponíveis; alvo de dtype fora
        da lista acima; alvo com valor fora de {0, 1} (o caminho antigo levanta
        o ValueError do roc_curve) ou -0.0; NaN não mascarado em alvo Float64 (o
        antigo o propaga); linha válida fora das contagens; ou qualquer exceção
        na montagem (o caminho antigo reproduz a mesma)."""
        if self.sample_col is not None and self._sample_codes is None:
            return None
        dt = self.df[self.target].dtype
        if isinstance(dt, np.dtype):
            if not (dt.kind in "biu" or dt == np.float64):
                return None
        elif not isinstance(dt, self._METRICS_DTYPES_ANULAVEIS):
            return None
        try:
            return self._metrics_clf_tabela_calc(leaf_ids)
        except Exception:
            return None

    def _metrics_clf_tabela_calc(self, leaf_ids):
        # MESMA conversão do caminho antigo (lá feita por amostra)
        y = self.df[self.target].to_numpy(dtype="float64")
        ok_y = ~np.isnan(y)
        # Float64 anulável pode carregar NaN NÃO mascarado (ex.: 0/0): isna() dá
        # False e o groupby/mean antigo PROPAGA o NaN (média da folha NaN,
        # overall pd.NA). Aqui ele viraria "ausente" — então só segue quando o
        # NaN da conversão é exatamente o NA do pandas (numpy float64: isna ≡
        # isnan; Int*/UInt*: só o NA mascarado).
        if int((~ok_y).sum()) != int(self.df[self.target].isna().sum()):
            return None
        yv = y[ok_y]
        if not ((yv == 0) | (yv == 1)).all() or np.signbit(yv).any():
            return None
        if self.sample_col is not None:
            ref_mask = self._mask_ref
        else:
            ref_mask = pd.Series(True, index=self.df.index)
        # overall: média do alvo de TODA a referência (órfãs incluídas), pela
        # mesma expressão de antes (inclusive o tipo do NA de alvo anulável)
        overall = self.df.loc[ref_mask, self.target].mean()

        ids, code = self._leaf_codes(leaf_ids)
        n_l = len(ids)
        if self.sample_col is not None:
            n_s = len(self._samples)
            # coluna por amostra pela MESMA chave de dicionário do caminho
            # antigo (_sample_masks[a] e _mask_ref = _sample_masks.get(ref))
            coluna = {a: j for j, a in enumerate(self._samples)}
            j_ref = coluna[self.ref_sample]
            # linhas SEM amostra (sample_col NaN/None/pd.NA → código -1) vão
            # para a coluna EXTRA n_s, não somem: o all_pd antigo é o groupby
            # da folha sobre TODAS as linhas do df, com ou sem amostra
            sc = np.where(self._sample_codes >= 0, self._sample_codes, n_s)
        else:
            n_s = 1
            coluna = {}
            j_ref = 0
            sc = np.zeros(len(self.df), dtype=np.int32)
        nb = n_s + 1
        # órfãs (código -1) e alvo NaN saem ANTES de combinar os códigos
        ok = (code >= 0) & ok_y
        comb = code.astype(np.int64) * nb + sc
        pos = np.bincount(comb[ok & (y == 1)], minlength=n_l * nb).reshape(n_l, nb)
        neg = np.bincount(comb[ok & (y == 0)], minlength=n_l * nb).reshape(n_l, nb)
        # garantia: TODA linha com folha e alvo válido (as sem amostra também)
        # caiu numa célula — senão o fallback all_pd perderia linhas
        if int(pos.sum()) + int(neg.sum()) != int(ok.sum()):
            return None

        # pd_map com o MESMO fallback do caminho antigo: referência → todas as
        # linhas da folha (TODAS as colunas, inclusive a extra das sem amostra)
        # → overall. Média de 0/1 = pos/(pos+neg), exata.
        ref_pos, ref_cnt = pos[:, j_ref], pos[:, j_ref] + neg[:, j_ref]
        all_pos = pos.sum(axis=1)
        all_cnt = all_pos + neg.sum(axis=1)
        score = np.empty(n_l, dtype="float64")
        for k in range(n_l):
            if ref_cnt[k] > 0:
                v = ref_pos[k] / ref_cnt[k]
            elif all_cnt[k] > 0:
                v = all_pos[k] / all_cnt[k]
            else:
                v = np.nan
            score[k] = float(v) if pd.notna(v) else float(overall)
        return ids, pos, neg, score, coluna

    def _metrics_clf_de_tabela(self, tab, cutoff):
        from sklearn.metrics import (accuracy_score, auc as _auc, f1_score,
                                     roc_curve)
        ids, pos, neg, score, coluna = tab
        # folhas com score NaN (overall NaN) ficam fora, como o filtro `valid`
        com_score = ~np.isnan(score)

        if self.sample_col is not None:
            ordem = ([self.ref_sample]
                     + [a for a in self._samples if a != self.ref_sample])
            grupos = [(a, self._sample_masks[a], coluna[a]) for a in ordem]
        else:
            grupos = [("todos", pd.Series(True, index=self.df.index), 0)]

        def _r6(v):
            # mesmo arredondamento de classification_metrics (digits=6)
            return round(float(v), 6) if np.isfinite(v) else float("nan")

        def _metrics_from_table(p, q, s):
            # métricas de UMA amostra a partir das contagens por folha
            n_pos, n_neg = int(p.sum()), int(q.sum())
            if n_pos + n_neg == 0:
                return {"taxa_default": np.nan, "KS": np.nan, "AUC": np.nan,
                        "Gini": np.nan, "Acuracia": np.nan, "F1": np.nan}
            # linhas sintéticas SÓ com peso > 0 (classe sem linhas não vira
            # rótulo; o sklearn também descartaria, mas sem gerar limiar)
            m1, m0 = p > 0, q > 0
            y_g = np.r_[np.ones(int(m1.sum())), np.zeros(int(m0.sum()))]
            s_g = np.r_[s[m1], s[m0]]
            w_g = np.r_[p[m1], q[m0]].astype("float64")
            # classe única: olha os TOTAIS (como o np.unique(y)<2 do _roc_pack)
            if n_pos > 0 and n_neg > 0:
                fpr, tpr, thr = roc_curve(y_g, s_g, sample_weight=w_g)
                auc_v = float(_auc(fpr, tpr))
                jj = int(np.argmax(tpr - fpr))
                ks = float(tpr[jj] - fpr[jj])
                corte_ks = float(thr[jj]) if np.isfinite(thr[jj]) else 1.0
                gini = 2 * auc_v - 1
            else:
                auc_v = gini = ks = float("nan")
                corte_ks = 0.5
            corte = corte_ks if cutoff is None else float(cutoff)
            y_pred = (s_g >= corte).astype(int)
            return {
                "taxa_default": round(float(n_pos / (n_pos + n_neg)), 4),
                "KS": _r6(ks), "AUC": _r6(auc_v), "Gini": _r6(gini),
                "Acuracia": _r6(accuracy_score(y_g, y_pred, sample_weight=w_g)),
                "F1": _r6(f1_score(y_g, y_pred, zero_division=0,
                                   sample_weight=w_g)),
            }

        linhas = []
        for nome, mask, j in grupos:
            base = {"amostra": nome, "n": int(mask.sum())}
            base.update(_metrics_from_table(pos[com_score, j], neg[com_score, j],
                                            score[com_score]))
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
                nonref = [a for a in self._samples
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
            sub = self._sub(seg["mask"])
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
            mask = self._smask(sample)
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
            todas = list(self._samples)
            amostras = [self.ref_sample] + [a for a in todas if a != self.ref_sample]
        rows = []
        for a in amostras:
            vals = []
            for sid in order:
                m = self.segments[sid]["mask"]
                if a is not None:
                    m = m & self._smask(a)
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
            nonref = [a for a in self._samples
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
                m = m & self._smask(check_sample)
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
                m = m & self._smask(sample)
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
        ax.set_xticklabels([self._leaf_tag(i, n, max_len=16)
                            for i, n in zip(ids, notas)], rotation=0, fontsize=9)
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
            y = self.df.loc[self._smask(sample), self.target].to_numpy(dtype="float64")
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
            mask = mask & self._smask(sample)
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
        todas = list(self._samples)
        ordem = [self.ref_sample] + [a for a in todas if a != self.ref_sample]
        out = []
        for a in ordem:
            mask = self._smask(a)
            y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
            y = y[~np.isnan(y)]
            if np.unique(y).size >= 2:
                # .copy(): a máscara vai ao chamador — não pode ser a Series
                # compartilhada de _sample_masks (um `&=` lá fora a corromperia)
                out.append((a, mask.copy()))
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
            mask = self._smask(sample)
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
    # PLOT_CAP: curva CAP (Cumulative Accuracy Profile / Lorenz) da régua —
    #   % acumulado do ALVO capturado × % acumulado da carteira ordenada do
    #   pior para o melhor score (alvo previsto da folha), com o AR (accuracy
    #   ratio) por amostra na legenda, a diagonal (aleatório) e a curva do
    #   modelo perfeito. Paridade clf×reg: em classificação captura EVENTOS;
    #   em regressão captura a MASSA do alvo contínuo (mesma ordenação
    #   previsto→alvo do ModelSegmenter).
    # ------------------------------------------------------------------
    def plot_cap(self, samples: list | None = None, figsize=(6.0, 5.6),
                 save_path: str | None = None, dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
            from matplotlib.ticker import PercentFormatter
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_cap requer matplotlib.") from e
        pred = self._predicted_series()
        # amostras (referência primeiro): clf exige as duas classes; reg usa todas
        if self._is_clf:
            grupos = self._samples_with_both_classes()
        elif self.sample_col is None:
            grupos = [(None, pd.Series(True, index=self.df.index))]
        else:
            todas = list(self._samples)
            ordem = [self.ref_sample] + [a for a in todas if a != self.ref_sample]
            grupos = [(a, self._sample_masks[a]) for a in ordem]
        if samples is not None:
            grupos = [(a, m) for a, m in grupos if a in samples]
        fig, ax = self._new_ax(figsize, dpi, ax)
        ax.plot([0, 1], [0, 1], color="#9aa7b2", lw=1.0, ls="--", label="aleatório")
        cores = ["#0f3d57", "#d6453e", "#1aa64b", "#caa000", "#6b3fa0", "#2a9d8f"]
        perfeito_feito = False
        alguma = False
        for i, (a, mask) in enumerate(grupos):
            y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
            s = pred[mask.values].to_numpy(dtype="float64")
            ok = ~(np.isnan(y) | np.isnan(s))
            y, s = y[ok], s[ok]
            tot = float(y.sum())
            if y.size == 0 or tot <= 0 or (self._is_clf and np.unique(y).size < 2):
                continue
            # carteira ordenada do PIOR para o MELHOR score → % do alvo capturado
            order = np.argsort(-s, kind="mergesort")
            y_ord = y[order]
            n = y_ord.size
            x_cap = np.concatenate(([0.0], np.arange(1, n + 1) / n))
            y_cap = np.concatenate(([0.0], np.cumsum(y_ord) / tot))
            # área sob a CAP pela regra do trapézio (manual: independe da versão
            # do numpy — trapz foi deprecado na 2.0 e trapezoid não existe <2.0)
            area_mod = float(np.sum(np.diff(x_cap) * (y_cap[:-1] + y_cap[1:]) / 2.0))
            # modelo perfeito = ordenar pelo PRÓPRIO alvo: clf sobe reto até a
            # taxa de evento; reg é a curva de Lorenz do alvo decrescente.
            y_perf = np.concatenate(([0.0], np.cumsum(np.sort(y)[::-1]) / tot))
            area_perf = float(np.sum(np.diff(x_cap) * (y_perf[:-1] + y_perf[1:]) / 2.0))
            # AR = (área do modelo − 0,5) / (área do perfeito − 0,5)
            ar = ((area_mod - 0.5) / (area_perf - 0.5)
                  if area_perf > 0.5 else float("nan"))
            if not perfeito_feito:                       # perfeito da 1ª amostra útil
                ax.plot(x_cap, y_perf, color="#8891a0", ls=":", lw=1.4,
                        label="modelo perfeito")
                perfeito_feito = True
            nome = a if a is not None else "todos"
            ax.plot(x_cap, y_cap, color=cores[i % len(cores)], lw=2.0,
                    label=f"{nome} · AR={ar:.3f}")
            alguma = True
        if not alguma:
            msg = ("sem as duas classes para a curva CAP" if self._is_clf
                   else "sem massa de alvo para a curva CAP")
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, color="#889")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("% acumulado da carteira (pior → melhor score)")
        ax.set_ylabel("% acumulado de eventos capturados" if self._is_clf
                      else f"% acumulado de {self._risk_word} capturado")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=None))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=None))
        ax.set_title(f"Curva CAP (Lorenz) da régua de {self._risk_word}",
                     fontsize=12, fontweight="bold", color="#15324a")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # PLOT_LIFT: lift por decil de score (barras; decil 1 = piores scores) +
    #   linha de gains acumulado (% do alvo capturado até o decil) num segundo
    #   eixo, com a referência em lift = 1 (aleatório). Paridade clf×reg: em
    #   classificação a taxa é de eventos; em regressão é a média do alvo
    #   contínuo (mesma ordenação previsto→alvo do ModelSegmenter).
    # ------------------------------------------------------------------
    def plot_lift(self, sample: str | None = None, n_bins: int = 10,
                  figsize=(7.2, 4.2), save_path: str | None = None,
                  dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
            from matplotlib.ticker import PercentFormatter
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_lift requer matplotlib.") from e
        pred = self._predicted_series()
        if sample is None and self.sample_col is not None:
            sample = self.ref_sample
        mask = pd.Series(True, index=self.df.index)
        if sample is not None and self.sample_col is not None:
            mask = self._smask(sample)
        y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
        s = pred[mask.values].to_numpy(dtype="float64")
        ok = ~(np.isnan(y) | np.isnan(s))
        y, s = y[ok], s[ok]
        fig, ax = self._new_ax(figsize, dpi, ax)
        tx_geral = float(np.mean(y)) if y.size else 0.0
        if y.size == 0 or tx_geral <= 0 or (self._is_clf and np.unique(y).size < 2):
            msg = ("amostra com 1 classe" if self._is_clf
                   else "alvo sem massa para o lift")
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, color="#889")
            ax.axis("off")
            fig.tight_layout()
            return fig
        # decil 1 = PIORES scores (maior alvo previsto) — ordena descendente e
        # fatia em n_bins grupos de tamanho ~igual (robusto a empates de score).
        order = np.argsort(-s, kind="mergesort")
        y_ord = y[order]
        n = y_ord.size
        n_bins = max(2, min(int(n_bins), n))
        idx = np.minimum((np.arange(n) * n_bins) // n, n_bins - 1)
        tot = float(y_ord.sum())                       # > 0 (guard do tx_geral)
        lifts, gains = [], []
        acum = 0.0
        for g in range(n_bins):
            m = idx == g
            tx_g = float(y_ord[m].mean()) if m.any() else float("nan")
            lifts.append(tx_g / tx_geral)
            acum += float(y_ord[m].sum())
            gains.append(acum / tot)
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
        ax.set_ylabel("lift (taxa do decil / taxa geral)" if self._is_clf
                      else "lift (média do decil / média geral)")
        ax.set_ylim(0, max([lf for lf in lifts if np.isfinite(lf)] + [1.0]) * 1.18)
        # gains acumulado no eixo secundário (% do alvo capturado)
        ax2 = ax.twinx()
        ax2.plot(x, gains, color="#0f3d57", lw=2.0, marker="o", ms=4.5,
                 label="gains acumulado")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("% de eventos capturados (acum.)" if self._is_clf
                       else f"% de {self._risk_word} capturado (acum.)")
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=None))
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right", framealpha=0.9)
        ax.set_title(f"Lift e gains por decil de score · {sample or 'todos'}",
                     fontsize=12, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # METRICS_BY_SAFRA: métricas da régua como modelo POR SAFRA (mês de
    #   date_col/time_col), com o alvo previsto da folha como score — mesma
    #   leitura do metrics(), trocando a amostra pela safra.
    # ------------------------------------------------------------------
    def metrics_by_safra(self, sample: str | None = None,
                         time_col: str | None = None) -> pd.DataFrame:
        """Métricas da régua **por safra** (mês de ``date_col``/``time_col``).

        classification → ``safra, n, taxa_evento, auc, ks, gini``.
        regression → ``safra, n, previsto_medio, realizado_medio, mae, rmse, r2``.
        Reutiliza :func:`~yggdrasil.metrics.classification_metrics` /
        :func:`~yggdrasil.metrics.regression_metrics` (mesmo pacote de
        :meth:`metrics`). Safras pequenas ou de classe única não quebram — as
        métricas ficam NaN.

        ``sample=None`` usa toda a base (as safras normalmente já separam
        DES/OOT); informe uma amostra para restringir."""
        time_col = time_col or self.date_col
        if time_col is None or time_col not in self.df.columns:
            raise ValueError("Informe time_col ou configure date_col.")
        from yggdrasil.metrics import classification_metrics, regression_metrics
        pred = self._predicted_series()
        mask = pd.Series(True, index=self.df.index)
        if sample is not None and self.sample_col is not None:
            mask = self._smask(sample)
        base = self._sub(mask, time_col)
        sc_full = pred[mask.values]
        safra = pd.to_datetime(base[time_col], errors="coerce").dt.to_period("M")
        met_cols = (["taxa_evento", "auc", "ks", "gini"] if self._is_clf
                    else ["previsto_medio", "realizado_medio", "mae", "rmse", "r2"])
        rows = []
        for per, g in base.groupby(safra):            # groupby dropa safra NaT
            y = g[self.target].to_numpy(dtype="float64")
            sc = sc_full.reindex(g.index).to_numpy(dtype="float64")
            ok = ~np.isnan(y) & ~np.isnan(sc)
            y, sc = y[ok], sc[ok]
            row = {"safra": str(per), "n": int(y.size)}
            row.update({c: float("nan") for c in met_cols})
            if self._is_clf:
                row["taxa_evento"] = float(np.mean(y)) if y.size else float("nan")
                if y.size >= 2 and len(np.unique(y)) == 2:
                    try:
                        m = classification_metrics(y, sc)
                        row.update({c: m.get(c, float("nan"))
                                    for c in ("auc", "ks", "gini")})
                    except Exception:  # noqa: BLE001 - safra degenerada ⇒ NaN
                        pass
            else:
                row["previsto_medio"] = float(np.mean(sc)) if sc.size else float("nan")
                row["realizado_medio"] = float(np.mean(y)) if y.size else float("nan")
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

    def plot_metrics_by_safra(self, sample: str | None = None, metrics=None,
                              time_col: str | None = None, figsize=(9.6, 4.2),
                              dpi: int = 150, save_path: str | None = None,
                              ax=None, ylim=None):
        """Evolução das métricas da régua por safra (linhas), a partir de
        :meth:`metrics_by_safra`. ``metrics=None`` usa o default do ``task_type``
        (clf: ``ks``/``auc`` · reg: ``rmse``/``r2``); métricas que não existirem
        para o tipo de alvo são ignoradas. ``ylim=(lo, hi)`` fixa o eixo vertical;
        tem precedência sobre a heurística que fixa [0,1] para métricas de
        discriminação."""
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_metrics_by_safra requer matplotlib.") from e
        default = ("ks", "auc") if self._is_clf else ("rmse", "r2")
        if metrics is None:
            metrics = default
        ms = self.metrics_by_safra(sample, time_col)
        cols = [m for m in metrics if m in ms.columns and m not in ("safra", "n")]
        if not cols:                        # métricas do outro task_type pedidas
            cols = [c for c in default if c in ms.columns]
        fig, ax = self._new_ax(figsize, dpi, ax)
        if ms.empty or not cols:
            ax.text(0.5, 0.5, "sem métricas por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889")
            ax.axis("off")
            fig.tight_layout()
            return fig
        x = list(range(len(ms)))
        cores = ["#0f3d57", "#d6453e", "#1aa64b", "#caa000", "#6b3fa0", "#2a9d8f"]
        for i, c in enumerate(cols):
            ax.plot(x, ms[c], marker="o", lw=2.0, ms=4.5, color=cores[i % len(cores)],
                    markeredgecolor="#33424f", markeredgewidth=0.5, label=c.upper())
        # eixo Y: override explícito (ylim) tem precedência; senão, métricas de
        # discriminação (KS/AUC/Gini) ⇒ 0–1 comparável; regressão autoescala.
        if ylim is not None:
            ax.set_ylim(*ylim)
        elif all(c in _UNIT_METRICS for c in cols):
            ax.set_ylim(0.0, 1.0)
        # rótulos mmm/aa; com muitas safras, afina os ticks p/ não sobrepor
        labels = _fmt_safras(ms["safra"])
        step = max(1, len(ms) // 18)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("safra")
        ax.set_ylabel("métrica")
        # sem restrição de amostra, marca a troca DES→OOT… (linhas pontilhadas)
        if sample is None:
            self._draw_sample_dividers(ax, list(ms["safra"]),
                                       time_col or self.date_col)
        ax.set_title(f"Métricas por safra · {sample or 'todas as amostras'}",
                     fontsize=12, fontweight="bold", color="#15324a")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
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
        ids, notas, rates, los, his = [], [], [], [], []
        for _, r in lv.iterrows():
            sid = r["segmento"]
            m = self.segments[sid]["mask"]
            if sample is not None and self.sample_col is not None:
                m = m & self._smask(sample)
            v = self.df.loc[m, self.target].to_numpy(dtype="float64")
            v = v[~np.isnan(v)]
            if len(v) == 0:
                continue
            p, lo, hi = self._wilson_ci(int(v.sum()), len(v))
            ids.append(sid); notas.append(int(r["nota"]))
            rates.append(p); los.append(lo); his.append(hi)
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
        ax.set_xticklabels([self._leaf_tag(i, n, max_len=16)
                            for i, n in zip(ids, notas)], rotation=0, fontsize=9)
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
            mask = self._smask(sample)
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
            sub = self._sub(self.segments[sid]["mask"], feature)
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
            sub = self._sub(self.segments[sid]["mask"], feature)
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
            mask = mask & self._smask(sample)
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
            cart = self.df[self._smask(sample)]
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
            mask = mask & self._smask(sample)
        col = self.df.loc[mask, feature]
        kind = self._detect_kind(self.df, feature, None)
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
            mask = mask & self._smask(sample)
        sub = self._sub(mask, feature)
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
        ids = self._bin_ids(sub[feature], bins)   # 1 passada; None = _mask_in por faixa
        for k, b in enumerate(bins):
            m = self._mask_in(sub, feature, b).to_numpy() if ids is None else (ids == k)
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
            mask = mask & self._smask(sample)
        sub = self._sub(mask, feature, time_col)
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
        leaf = self._sub(mask, feature, time_col)
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
        # contagens por faixa num passo por recorte (código de faixa por linha),
        # em vez de um _mask_in (astype(str) da coluna) por faixa × safra
        ref_pct = [max(c / n_ref, eps) for c in self._bin_counts(ref[feature], bins)]
        safra = pd.to_datetime(leaf[time_col], errors="coerce").dt.to_period("M")
        rows = []
        for per, g in leaf.groupby(safra):
            n_g = len(g)
            if n_g == 0:
                continue
            cur_pct = [c / n_g for c in self._bin_counts(g[feature], bins)]
            psi = _psi_from_shares(ref_pct, cur_pct, eps)
            rows.append({"safra": str(per), "n": n_g, "psi": round(psi, 4),
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
            mask = mask & self._smask(sample)
        sub = self._sub(mask, feature)
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
        # rótulos do eixo X = só as FAIXAS (sem repetir "feature: " em cada uma)
        labels = [s.split(": ", 1)[-1] for s in vt["faixa"]]; reprs = vt["repr_%"].to_numpy()
        xs = list(range(len(labels)))
        cols = ["#c98a8a" if "faltante" in f else "steelblue" for f in labels]
        ax.bar(xs, reprs, color=cols, edgecolor="#2f5d82", alpha=0.85, width=0.7,
               label="% da amostra")
        for x0, rp in zip(xs, reprs):
            ax.text(x0, rp, f"{rp:.0f}%", ha="center", va="bottom", fontsize=9.5,
                    fontweight="semibold", color="#15324a")
        ax.set_ylabel("% da amostra"); ax.set_ylim(0, float(np.nanmax(reprs)) * 1.2 + 1)
        is_clf = self._is_clf
        risco = vt[rcol].to_numpy(dtype="float64")
        yline = risco * 100 if is_clf else risco
        # eixo secundário: "% de maus" (clf) ou o rótulo do problema (reg, ex.: LGD)
        ylabel = "% de maus" if is_clf else self._risk_word
        ax2 = ax.twinx()
        ax2.plot(xs, yline, color="crimson", lw=2.3, marker="o", ms=5.5,
                 markeredgecolor="#fff", markeredgewidth=0.7, label=ylabel)
        for x0, yv in zip(xs, yline):
            if np.isfinite(yv):
                ax2.text(x0, yv, (f"{yv:.1f}%" if is_clf else f"{yv:.3f}"),
                         ha="center", va="bottom", fontsize=9.5, fontweight="semibold",
                         color="crimson")
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
        ref_mask = (mask0 & self._smask(self.ref_sample)
                    if self.sample_col is not None else mask0)
        ref = self._sub(ref_mask, feature)
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

        ref_risco = [_risco(ref[m]) for m in self._bin_masks(ref, feature, bins)]
        order = sorted(range(len(bins)),
                       key=lambda i: (np.inf if pd.isna(ref_risco[i]) else ref_risco[i]))
        # por amostra
        xs_s = self._ordered_samples_with_value() or [None]
        xs_s_lab = [str(a) if a is not None else "todos" for a in xs_s]
        ser_s = {i: [] for i in range(len(bins))}
        for a in xs_s:
            fa = (self._sub(mask0 & self._smask(a), feature)
                  if a is not None else self._sub(mask0, feature))
            for i, m in enumerate(self._bin_masks(fa, feature, bins)):
                ser_s[i].append(_risco(fa[m]))
        # por safra
        xs_t, ser_t = [], {i: [] for i in range(len(bins))}
        # _SEM_SAFRAS (uso interno do gráfico por amostra): sem laço por safra
        tcol = None if time_col is _SEM_SAFRAS else (time_col or self.date_col)
        if tcol is not None and tcol in self.df.columns:
            base = (self._sub(mask0 & self._smask(sample), feature, tcol)
                    if (sample and self.sample_col is not None)
                    else self._sub(mask0, feature, tcol))
            safra = pd.to_datetime(base[tcol], errors="coerce").dt.to_period("M")
            for per, g in base.groupby(safra):
                if len(g) < min_n:
                    continue
                xs_t.append(str(per))
                for i, m in enumerate(self._bin_masks(g, feature, bins)):
                    ser_t[i].append(_risco(g[m]))
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
                                          dpi=150, save_path=None, ax=None, inv=None):
        """Risco de cada faixa por amostra; cruzamentos = inversão da ordem de risco.
        Espelha :meth:`ModelSegmenter.plot_variable_inversion_by_sample`.

        ``inv``: resultado de :meth:`variable_inversion` já calculado (mesmos
        ``feature``/``sid``/``max_n_bins``/``min_bin_size``; o ``time_col`` não
        importa, este gráfico só lê as faixas e a série por amostra) — evita
        refazer o cálculo quando a mesma análise também desenha o gráfico por
        safra. Sem ele, o cálculo pula o laço por safra, que aqui seria descartado."""
        import matplotlib.pyplot as plt
        if inv is None:
            inv = self.variable_inversion(feature, sid=sid, time_col=_SEM_SAFRAS,
                                          max_n_bins=max_n_bins, min_bin_size=min_bin_size)
        fig, ax = self._new_ax(figsize, dpi, ax)
        s = inv.get("series")
        if not s or not inv["ordered"]:
            ax.text(0.5, 0.5, "menos de 2 faixas", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        xs = s["xs_sample"]; x = list(range(len(xs)))
        cmap = plt.get_cmap("RdYlGn_r"); k = len(inv["ordered"])
        _rot = _sem_prefixo_da_variavel(s["labels"], feature)
        for rank, i in enumerate(inv["ordered"]):
            ax.plot(x, s["ser_sample"][i], marker="o", lw=1.9, ms=5.5,
                    color=cmap(rank / (k - 1) if k > 1 else 0.5),
                    markeredgecolor="#33424f", markeredgewidth=0.6, label=_rot[i])
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
                                         save_path=None, ax=None, inv=None):
        """Risco de cada faixa por safra; safras com inversão ficam sombreadas.
        Espelha :meth:`ModelSegmenter.plot_variable_inversion_by_safra`.

        ``inv``: resultado de :meth:`variable_inversion` já calculado com os
        MESMOS parâmetros (inclusive ``time_col``) — evita refazer o cálculo."""
        import matplotlib.pyplot as plt
        if inv is None:
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
        _rot = _sem_prefixo_da_variavel(s["labels"], feature)
        for rank, i in enumerate(ordered):
            ax.plot(x, s["ser_safra"][i], marker="o", lw=1.7, ms=4.5,
                    color=cmap(rank / (k - 1) if k > 1 else 0.5),
                    markeredgecolor="#33424f", markeredgewidth=0.5, label=_rot[i])
        ax.set_xticks(x); ax.set_xticklabels(_fmt_safras(xs), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("risco médio"); ax.set_xlabel("safra")
        self._draw_sample_dividers(ax, list(xs), time_col)
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
        m = (mask0 & self._smask(sample or self.ref_sample)
             if self.sample_col is not None else mask0)
        fit = self._sub(m, feature)
        if self._detect_kind(fit, feature, None) != "num":
            return []
        x = fit[feature].to_numpy(dtype="float64")
        y = fit[self.target].to_numpy(dtype="float64")
        ok = ~np.isnan(y); x, y = x[ok], y[ok]
        x_obs = x[~np.isnan(x)]
        if len(y) < 4 or x_obs.size == 0 or x_obs.min() == x_obs.max():
            return []                     # min == max ⇔ <2 distintos (sem NaN)
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
        if self._detect_kind(self.df, feature, None) != "num":
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
        labels = [self._bin_label_short(b) for b in bins]   # sem repetir 'feat:' na legenda
        base_all = self._sub(mask0, feature, tcol)  # distribuição sobre toda a folha
        safra = pd.to_datetime(base_all[tcol], errors="coerce").dt.to_period("M")
        xs, Y = [], [[] for _ in bins]
        for per, g in base_all.groupby(safra):
            n_g = len(g)
            if n_g == 0:
                continue
            xs.append(str(per))
            for i, c in enumerate(self._bin_counts(g[feature], bins)):
                Y[i].append(100 * c / n_g)
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
        ax.set_xticklabels(_fmt_safras(xs), rotation=45, ha="right", fontsize=9.5)
        ax.set_ylabel("% acumulado da safra", fontsize=10.5)
        ax.tick_params(axis="y", labelsize=9.5)
        self._draw_sample_dividers(ax, list(xs), time_col)
        ax.set_title(f"'{self.label(feature)}' — distribuição acumulada das faixas do "
                     "optimal binning ao longo do tempo", fontsize=11,
                     fontweight="bold", color="#15324a")
        ax.grid(alpha=0.12, axis="y")
        fig.tight_layout()
        # legenda ABAIXO por ÚLTIMO (reserva o espaço sem o tight_layout desfazer);
        # fonte maior — as faixas do optbin são o eixo de leitura do gráfico
        self._legend_below(ax, title=f"faixa de '{self.label(feature)}' (optbin)",
                           fontsize=10)
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
            mask = mask & self._smask(sample)
        sub = self._sub(mask, feature, time_col)
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
        self._draw_sample_dividers(ax, list(sh["safra"]), time_col)
        rot = self.feature_labels.get(feature, feature)
        ax.set_title(f"'{rot}' ao longo do tempo — representatividade por categoria",
                     fontsize=11, fontweight="bold", color="#15324a")
        fig.tight_layout()
        # legenda ABAIXO por ÚLTIMO: reserva o espaço embaixo (subplots_adjust) sem o
        # tight_layout desfazer a reserva (categorias de nome longo achatavam o gráfico)
        self._legend_below(ax, title="categoria")
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_variable_timeseries(self, feature, time_col=None, sid=None, sample=None,
                                 figsize=(12.0, 3.4), save_path=None, dpi=150, ax=None,
                                 bs=None):
        """Variável NUMÉRICA: percentis por safra (min–max, p5–p95, média).
        Variável CATEGÓRICA: representatividade de cada categoria por safra.

        ``bs``: resultado de :meth:`variable_by_safra` já calculado com os MESMOS
        ``feature``/``time_col``/``sid``/``sample`` — evita refazer o cálculo
        (só vale para a numérica; a categórica o ignora)."""
        if self._detect_kind(self.df, feature, None) == "cat":
            return self.plot_variable_share_timeseries(
                feature, time_col, sid=sid, sample=sample, figsize=figsize,
                save_path=save_path, dpi=dpi, ax=ax)
        if bs is None:
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
        self._draw_sample_dividers(ax, list(bs["safra"]), time_col)
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
            if p < 0.98:      # barra (e o rótulo acima) que passaria de 100% fica sem texto
                ax.text(x0, p, f"{p * 100:.1f}%", ha="center", va="bottom", fontsize=7, color="#555")
        ax.set_xticks(x); ax.set_xticklabels(_fmt_safras(ps["safra"]), rotation=45, ha="right", fontsize=8)
        ax.set_xlim(-0.7, len(ps) - 0.3)                          # respiro nas bordas (eixo x)
        self._psi_pct_axis(ax)                                    # PSI em %, eixo 0–100% + linhas
        self._draw_sample_dividers(ax, list(ps["safra"]), time_col)
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
        # função pura do df (amostras e alvo não mudam): calculada uma vez e
        # guardada como TUPLA; o chamador recebe uma lista nova (pode alterá-la)
        if self._osv_memo is None:
            com_pd = [a for a in self._samples
                      if self.df.loc[self._sample_masks[a], self.target].notna().any()]
            ref = [self.ref_sample] if self.ref_sample in com_pd else []
            self._osv_memo = tuple(ref + [a for a in com_pd if a != self.ref_sample])
        return list(self._osv_memo)

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
                mm = m & self._smask(a) if a is not None else m
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
            base = base & self._smask(sample)
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

    @staticmethod
    def _tight_ylim(values, pad_frac: float = 0.03, floor: float = 0.003):
        """Faixa de Y JUSTA aos valores finitos (zoom): revela cruzamentos/diferenças
        que o eixo cheio comprime. ``(baixo, alto)`` em fração, ou ``None`` sem dados."""
        v = np.asarray([x for x in np.ravel(list(values))
                        if x is not None and np.isfinite(x)], dtype=float)
        if v.size == 0:
            return None
        lo, hi = float(v.min()), float(v.max())
        pad = max((hi - lo) * pad_frac, floor)
        return (max(0.0, lo - pad), hi + pad)

    @classmethod
    def _apply_zoom_ylim(cls, ax, values, ylim=None, auto_zoom=False) -> bool:
        """Aplica limites de Y: ``ylim=(lo, hi)`` (fração) tem precedência; lados
        ``None`` (ou ``auto_zoom=True``) são preenchidos pelo zoom justo aos dados.
        Sem ``ylim`` nem ``auto_zoom``, não mexe. Devolve ``True`` se alterou o eixo."""
        if ylim is None and not auto_zoom:
            return False
        lo = hi = None
        if ylim is not None:
            lo, hi = ylim
        if auto_zoom or lo is None or hi is None:
            tight = cls._tight_ylim(values)
            if tight is not None:
                lo = tight[0] if lo is None else lo
                hi = tight[1] if hi is None else hi
        if lo is not None or hi is not None:
            ax.set_ylim(bottom=lo, top=hi)
            return True
        return False

    def _finish_sibling_axes(self, ax, n_lines, values, ylim, auto_zoom, pct):
        """Fecha o eixo Y dos gráficos de folhas-irmãs (estabilidade): aplica o zoom
        (auto/manual) ou, sem zoom, dá headroom no topo p/ a legenda não cobrir as
        linhas; formata em % se ``pct``; posiciona a legenda."""
        zoomed = self._apply_zoom_ylim(ax, values, ylim, auto_zoom)
        if not zoomed:                          # eixo padrão: espaço p/ a legenda fixa
            _y0, _y1 = ax.get_ylim(); ax.set_ylim(_y0, _y1 + (_y1 - _y0) * 0.22)
        if pct:
            from matplotlib.ticker import PercentFormatter
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax.set_ylabel(f"{self._risk_mean} (%)")
        ax.legend(fontsize=8, ncol=max(1, min(n_lines, 4)),
                  loc="best" if zoomed else "upper left", framealpha=0.9)

    def plot_sibling_value_by_sample(self, parent_sid, leaves=None,
                                  figsize=(7.6, 4.0), dpi=150,
                                  save_path=None, ax=None, title=None,
                                  ylim=None, auto_zoom=False, pct=False):
        """Linhas da alvo médio das folhas-irmãs por amostra (DES, OOT, …). Onde
        as linhas se cruzam há INVERSÃO da ordem de risco entre as irmãs.
        ``auto_zoom``/``ylim`` dão zoom no eixo Y (revela cruzamentos comprimidos);
        ``pct=True`` mostra o eixo em % (alvo em [0,1], ex.: LGD)."""
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
                    label=self._leaf_tag(sid, nota.get(sid), max_len=16))
        ax.set_xticks(x); ax.set_xticklabels(xs, fontsize=9)
        ax.set_xlim(-0.25, len(xs) - 0.75 + 0.5)
        ax.set_ylabel(self._risk_mean); ax.set_xlabel("amostra")
        ax.set_title(title or f"{self._risk_mean} das folhas-irmãs por amostra",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        self._finish_sibling_axes(ax, len(ordered),
                                  [series[sid] for sid in ordered], ylim, auto_zoom, pct)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_sibling_value_by_safra(self, parent_sid, time_col=None, sample=None,
                                 leaves=None, min_n: int = 1,
                                 figsize=(9.6, 4.0), dpi=150,
                                 save_path=None, ax=None, title=None,
                                 ylim=None, auto_zoom=False, pct=False):
        """Linhas da alvo médio das folhas-irmãs por safra ao longo do tempo. As
        safras em que a ordem de risco inverte (vs. DES) ficam sombreadas em
        vermelho — leitura rápida de instabilidade temporal.
        ``auto_zoom``/``ylim`` dão zoom no eixo Y; ``pct=True`` mostra-o em %."""
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
                    label=self._leaf_tag(sid, nota.get(sid), max_len=16))
        ax.set_xticks(x); ax.set_xticklabels(_fmt_safras(xs), rotation=45, ha="right", fontsize=8)
        ax.set_xlim(-0.7, len(xs) - 0.3)
        ax.set_ylabel(self._risk_mean); ax.set_xlabel("safra")
        self._draw_sample_dividers(ax, list(xs), time_col)
        sfx = f" · {sample}" if sample else " · todas as amostras"
        ax.set_title(title or (f"{self._risk_mean} das folhas-irmãs por safra{sfx}"
                     "  ·  faixas vermelhas = inversão"),
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        self._finish_sibling_axes(ax, len(ordered),
                                  [series[sid] for sid in ordered], ylim, auto_zoom, pct)
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
            cur_pct = [(float(vc.get(sid, 0.0)) / n_g) if n_g else 0.0
                       for sid in leaf_ids]
            psi = _psi_from_shares([ref_pct[sid] for sid in leaf_ids], cur_pct, eps)
            rows.append({"safra": str(per), "n": n_g, "psi": round(psi, 4),
                         "classificacao": _classifica_psi(psi)})
        return (pd.DataFrame(rows, columns=["safra", "n", "psi", "classificacao"])
                .sort_values("safra").reset_index(drop=True))

    def plot_psi_by_safra(self, time_col=None, figsize=(8.4, 4.0), dpi=150,
                          save_path=None, ax=None, ylim=None, auto_zoom=False):
        """Gráfico de **barras** do PSI da segmentação ao longo do tempo (vs DES).
        Verde &lt; 0.10 (estável) · amarelo &lt; 0.25 (atenção) · vermelho ≥ 0.25
        (instável); linhas-guia em 0.10 e 0.25. ``auto_zoom``/``ylim`` dão zoom no
        eixo Y."""
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
        ax.set_xticks(x)
        ax.set_xticklabels(_fmt_safras(ps["safra"]), rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("safra")
        self._psi_pct_axis(ax)                                    # PSI em %, eixo 0–100% + linhas
        self._apply_zoom_ylim(ax, ps["psi"].to_numpy(dtype="float64"), ylim, auto_zoom)
        for xi, p in zip(x, ps["psi"]):
            if p < ax.get_ylim()[1] * 0.98:   # rótulo só se não passar do topo do eixo
                ax.annotate(f"{p * 100:.1f}%", (xi, p), textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=7, color="#33424f")
        self._draw_sample_dividers(ax, list(ps["safra"]), time_col)
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

    def _sample_dominante_por_safra(self, time_col=None):
        """Amostra dominante (moda) por safra (mês) — ``Series`` indexada por
        'YYYY-MM'. Função PURA do df: cacheada por ``time_col`` (o groupby varre a df
        inteira, e as linhas/bandas por safra são pedidas em muitos gráficos)."""
        time_col = time_col or self.date_col
        if self.sample_col is None or time_col is None or time_col not in self.df.columns:
            return None
        if time_col not in self._samp_by_cache:
            sc = self.sample_col
            saf = pd.to_datetime(self.df[time_col], errors="coerce").dt.to_period("M").astype(str)
            # frame de 2 colunas (Series alinhadas pelo MESMO índice do df, sem
            # realinhar mesmo com índice duplicado) em vez de assign + dropna sobre
            # a base inteira: o groupby só lê '_saf' e a amostra, e as duas cópias
            # das ~100 colunas eram um pico do tamanho da base (≈16 GB a 10M).
            # sample_col == '_saf' colidiria no dict: mantém o caminho antigo.
            src = (self.df.assign(_saf=saf) if sc == "_saf"
                   else pd.DataFrame({"_saf": saf, sc: self.df[sc]}))
            self._samp_by_cache[time_col] = (
                src.dropna(subset=[sc])
                .groupby("_saf")[sc]
                .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None))
        return self._samp_by_cache[time_col]

    def _sample_seq(self, safras, time_col=None):
        """Amostra dominante de cada safra em ``safras``, na MESMA ordem — lista com
        o nome da amostra por posição (ou ``None``). ``[]`` sem ``sample_col``/tempo.
        Base das linhas e bandas DES/OOT/… por safra."""
        samp_by = self._sample_dominante_por_safra(time_col)
        if samp_by is None:
            return []
        return [samp_by.get(str(p)) for p in safras]

    def _sample_boundaries(self, safras, time_col=None):
        """Índices no eixo X (= ``range(len(safras))``) onde a AMOSTRA dominante
        muda entre safras consecutivas. ``safras`` é a sequência de safras (Period
        ou str 'YYYY-MM') na MESMA ordem do subplot. Retorna ``[]`` se não houver
        ``sample_col`` (usado p/ marcar a troca de amostra com linha pontilhada)."""
        seq = self._sample_seq(safras, time_col)
        return [i for i in range(1, len(seq))
                if seq[i] is not None and seq[i - 1] is not None and seq[i] != seq[i - 1]]

    def _sample_bands(self, safras, time_col=None):
        """Trechos contíguos de safras com a MESMA amostra dominante:
        ``[(i_ini, i_fim, amostra), ...]`` (índices inclusivos no eixo X) — base dos
        rótulos DES/OOT/ESTAB por período do tempo."""
        seq = self._sample_seq(safras, time_col)
        bands, i, n = [], 0, len(seq)
        while i < n:
            if seq[i] is None:
                i += 1
                continue
            j = i
            while j + 1 < n and seq[j + 1] == seq[i]:
                j += 1
            bands.append((i, j, seq[i]))
            i = j + 1
        return bands

    def _draw_sample_dividers(self, ax, safras, time_col=None, label=True,
                              fontsize=7.5):
        """Num gráfico com safras no eixo X, desenha as **linhas verticais que
        separam as amostras** (DES · OOT · ESTABILIDADE …) e, com ``label=True``,
        rotula cada trecho com o nome da amostra no topo. Best-effort: sem
        ``sample_col``/tempo não faz nada. Compartilhado por todos os gráficos por
        safra (análise de variáveis e estabilidade das folhas)."""
        for bx in self._sample_boundaries(safras, time_col):
            ax.axvline(bx - 0.5, ls=":", lw=1.1, color="#33424f", alpha=0.75, zorder=1.6)
        if not label:
            return
        for i0, i1, a in self._sample_bands(safras, time_col):
            ab = "ESTAB" if str(a).upper().startswith("ESTAB") else str(a)
            ax.annotate(ab, xy=((i0 + i1) / 2.0, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, -2), textcoords="offset points",
                        ha="center", va="top", fontsize=fontsize, fontweight="bold",
                        color="#33424f", zorder=4, clip_on=True,
                        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none",
                                  alpha=0.6))

    @staticmethod
    def _legend_below(ax, title=None, max_label=20, fontsize=8):
        """Legenda ABAIXO do eixo, RESERVADA dentro da figura (``subplots_adjust``).
        Vantagens sobre a legenda vertical à direita (que, com rótulos longos,
        achatava o gráfico) e sobre a legenda flutuante embaixo:
        - o eixo mantém a largura cheia — nunca é espremido;
        - fica DENTRO do ``figsize``, então não é recortada mesmo salvando sem
          ``bbox_inches='tight'`` (necessário p/ alinhar alturas de gráficos lado a
          lado — ver o par percentis×PSI da aba de variáveis);
        - usa QUANTAS COLUNAS couberem na largura da figura → legenda baixa e o
          gráfico continua visível (grande).
        Trunca rótulos longos p/ caber mais colunas."""
        import math
        fig = ax.figure
        handles, labels = ax.get_legend_handles_labels()
        if not labels:
            return
        short = [s if len(s) <= max_label else s[:max_label - 1] + "…" for s in labels]
        mx = max((len(s) for s in short), default=1)
        fw, fh = fig.get_size_inches()
        # nº de colunas que cabem: largura útil / largura aprox. de um item (rótulo +
        # marcador), proporcional à fonte → mais colunas = legenda baixa e gráfico maior
        item_w = mx * 0.0076 * fontsize + 0.55
        ncol = max(1, min(len(short), int((fw - 0.2) / item_w)))
        nrows = math.ceil(len(short) / ncol)
        # espaço reservado embaixo (polegadas): datas do eixo x + linhas (∝ fonte) + título
        dates_in = 0.62
        leg_in = 0.021 * fontsize * nrows + (0.26 if title else 0.10)
        frac = min(0.66, (dates_in + leg_in) / fh)
        fig.subplots_adjust(bottom=frac)
        # âncora em coords de FIGURA: topo da legenda logo abaixo das datas do eixo x
        ax.legend(handles, short, loc="upper center",
                  bbox_to_anchor=(0.5, max(0.0, frac - dates_in / fh)),
                  bbox_transform=fig.transFigure, ncol=ncol, fontsize=fontsize,
                  framealpha=0.9, title=title, borderaxespad=0.2,
                  columnspacing=1.1, handlelength=1.3)

    @staticmethod
    def _psi_pct_axis(ax):
        """Eixo Y do PSI em PERCENTUAL, FIXO em 0–100%, com as linhas-guia de
        **atenção** (10%, amarelo) e **crítico** (25%, vermelho). O PSI segue na
        escala fração (0–1) nos dados; só a EXIBIÇÃO vira %. Escala fixa dá leitura
        comparável entre variáveis/safras e garante que as duas linhas apareçam. As
        linhas não são rotuladas (a cor indica) — o texto batia nas barras."""
        from matplotlib.ticker import PercentFormatter
        ax.set_ylim(0, 1.0)
        # decimals=None: casas automáticas conforme a faixa — no eixo cheio mostra
        # "0%..100%", e com zoom (faixa estreita) mostra "1.8%", "2.0%", … sem colapsar
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=None))
        ax.set_ylabel("PSI (%)")
        ax.axhline(0.10, color="#caa000", lw=1.1, ls="--", alpha=0.85)   # atenção
        ax.axhline(0.25, color="#d6453e", lw=1.1, ls="--", alpha=0.85)   # crítico

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
            # linhas verticais onde a amostra muda no tempo (grid: só linhas)
            self._draw_sample_dividers(ax, [str(p) for p in pers], time_col, label=False)
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
            # linhas verticais onde a amostra muda no tempo (grid: sem rótulo p/ não
            # colidir com a legenda de categorias dos subplots)
            self._draw_sample_dividers(ax, safras_sub, time_col, label=False)
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

    @staticmethod
    def metrics_comparison_options(task_type: str) -> list:
        """Métricas disponíveis no gráfico 'principais métricas por amostra', na
        ordem, para o tipo de alvo — alimenta o seletor da UI (nomes = colunas de
        :meth:`metrics`)."""
        return (["KS", "AUC", "Gini"] if task_type == "classification"
                else ["RMSE", "MAE", "R2"])

    def plot_metrics_comparison(self, figsize=(6.6, 4.6), dpi=150, save_path=None, ax=None,
                                metrics=None, pct=None):
        """Compara as **principais métricas entre amostras** (DES vs OOT lado a lado)
        em barras agrupadas — uma métrica por grupo, uma barra por amostra, com os
        valores anotados e a **variação DES→OOT** (seta + %) no topo de cada grupo.
        clf: **KS · AUC · Gini** (eixo único, em %). reg: **RMSE · MAE** (erro, na
        escala do alvo, eixo esquerdo) e **R²** (eixo direito próprio, em %).
        Referência em steelblue, comparação em crimson (estabilidade em teal).

        ``metrics``: subconjunto/ordem das métricas a exibir (nomes de coluna, ex.:
        ``["KS", "AUC"]``); ``None`` = todas. ``pct``: ``True`` força TODAS em %
        (útil p/ alvo em [0,1]), ``False`` força valor bruto, ``None`` = padrão por
        métrica. Valores negativos (Gini de OOT invertida, R²<0) aparecem abaixo do
        zero, nunca somem."""
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
        # subconjunto/ordem escolhido pelo usuário (por nome de coluna, sem caixa)
        if metrics is not None:
            order = [str(x).upper() for x in metrics]
            plano = sorted((p for p in plano if p[0].upper() in order),
                           key=lambda p: order.index(p[0].upper()))
        # toggle de %: força todas em % (True) ou todas em bruto (False)
        if pct is not None:
            plano = [(c, l, up, right, bool(pct)) for (c, l, up, right, _p) in plano]
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
        # cada eixo é formatado em % se TODAS as suas métricas estão em % (ver `pct`)
        left_pct = any(not p[3] for p in plano) and all(p[4] for p in plano if not p[3])
        right_pct = any(p[3] for p in plano) and all(p[4] for p in plano if p[3])
        if left_pct:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.tick_params(axis="y", labelsize=10)
        if ax2 is not None:
            rmax = max(right_vals + [0.0]); rmin = min(right_vals + [0.0])
            ax2.set_ylim(rmin * 1.15 if rmin < 0 else 0.0, max(rmax * 1.32, 1.0))
            if right_pct:
                ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax2.tick_params(axis="y", labelsize=10)
            ax2.set_ylabel("R² (eixo direito)", fontsize=10.5, color="#15324a")
        ax.set_xticks(np.arange(len(plano)) + (len(samples) - 1) * w / 2)
        ax.set_xticklabels([f"{lab} {'↑' if up else '↓'}" for _c, lab, up, _r, _p in plano],
                           fontsize=11.5)
        ax.set_ylabel("métrica (%)" if left_pct
                      else ("erro (escala do alvo)" if not self._is_clf else "métrica"),
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

    def plot_psi_by_sample(self, figsize=(7.0, 3.8), dpi=150, save_path=None, ax=None,
                           ylim=None, auto_zoom=False):
        """PSI da SEGMENTAÇÃO por AMOSTRA vs DES (barras verticais): estabilidade da
        distribuição das folhas ENTRE amostras (OOT, ESTABILIDADE, …). Amostra no
        eixo X, PSI (%) no eixo Y — verde &lt;10% · amarelo &lt;25% · vermelho ≥25%;
        linhas-guia de atenção (10%) e crítico (25%). ``auto_zoom``/``ylim`` dão zoom
        no eixo Y. Complementa o PSI por safra (ao longo do tempo)."""
        import matplotlib.pyplot as plt  # noqa: F401
        fig, ax = self._new_ax(figsize, dpi, ax)
        if self.sample_col is None:
            ax.text(0.5, 0.5, "requer coluna de amostra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        try:
            ps = self.psi()
        except Exception:
            ps = None
        if ps is None or ps.empty:
            ax.text(0.5, 0.5, "sem PSI por amostra (requer amostras além da DES)",
                    ha="center", va="center", transform=ax.transAxes, color="#889")
            ax.axis("off"); fig.tight_layout(); return fig
        amostras = ps["amostra"].tolist()
        vals = ps["psi"].to_numpy(dtype="float64")
        x = np.arange(len(amostras))
        cor = ["#1aa64b" if p < 0.10 else "#caa000" if p < 0.25 else "#d6453e" for p in vals]
        ax.bar(x, vals, color=cor, width=0.6, edgecolor="#33424f", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels([str(a) for a in amostras], fontsize=9)
        ax.set_xlim(-0.7, len(amostras) - 0.3); ax.set_xlabel("amostra")
        self._psi_pct_axis(ax)                             # eixo Y em %, 0–100% + linhas
        zoomed = self._apply_zoom_ylim(ax, vals, ylim, auto_zoom)  # zoom sobrescreve
        for xi, p in zip(x, vals):
            if p < ax.get_ylim()[1] * 0.98:                # não passar do topo do eixo
                ax.annotate(f"{p * 100:.1f}%", (xi, p), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=8, fontweight="bold",
                            color="#33424f")
        ax.set_title(f"PSI da segmentação por amostra vs {self.ref_sample}", fontsize=11,
                     fontweight="bold", color="#15324a", loc="left")
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_leaf_concentration(self, figsize=(8.4, 4.2), dpi=150, save_path=None, ax=None):
        """Barras HORIZONTAIS agrupadas da **concentração das folhas entre amostras**:
        representatividade (%) de cada folha DENTRO de cada amostra (DES, OOT, …).
        Folhas no eixo Y, concentração no eixo X — a figura CRESCE em altura com o nº
        de folhas (em vez de espremer as barras), então os rótulos de % ficam
        legíveis mesmo com muitas folhas. Análogo à distribuição dos ratings por
        amostra do ModelSegmenter."""
        import matplotlib.pyplot as plt  # noqa: F401
        from matplotlib.ticker import PercentFormatter
        if self.sample_col is None:
            fig, ax = self._new_ax(figsize, dpi, ax)
            ax.text(0.5, 0.5, "requer coluna de amostra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        lv = self.leaves(with_psi=True)
        if len(lv) == 0:
            fig, ax = self._new_ax(figsize, dpi, ax)
            ax.text(0.5, 0.5, "sem folhas", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        notas = lv["nota"].tolist()
        all_samples = [self.ref_sample] + [a for a in self._samples
                                           if a != self.ref_sample]
        cols = [(a, f"repr_{a}_%") for a in all_samples if f"repr_{a}_%" in lv.columns]
        n_leaves = len(notas); n_s = max(len(cols), 1)
        # ALTURA adaptativa ao nº de folhas (≈0.44" por folha, piso = figsize[1],
        # teto p/ não explodir): a figura cresce e as barras/rótulos mantêm o tamanho
        if ax is None:
            h = min(46.0, max(figsize[1], 1.4 + 0.44 * n_leaves))
            fig, ax = self._new_ax((figsize[0], h), dpi, ax)
        else:
            fig = ax.figure
        y = np.arange(n_leaves); bw = 0.82 / n_s
        conts, vmax = [], 0.0
        for k, (a, col) in enumerate(cols):
            vals = lv[col].to_numpy(dtype="float64")
            vmax = max(vmax, float(np.nanmax(vals)) if vals.size else 0.0)
            conts.append(ax.barh(y + (k - (n_s - 1) / 2.0) * bw, vals,
                                 height=bw, label=str(a), color=self._sample_bar_color(a),
                                 edgecolor="#33424f", linewidth=0.3))
        ax.set_yticks(y)
        ax.set_yticklabels([self._leaf_tag(s, n, max_len=18, com_prefixo=False)
                            for s, n in zip(lv["segmento"], lv["nota"])], fontsize=8)
        ax.invert_yaxis()                          # folha de menor nota no topo
        ax.set_xlabel("concentração (% da amostra)"); ax.set_ylabel("folha")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.set_xlim(0, max(vmax * 1.18, 1.0))      # folga à direita p/ os rótulos
        # título à ESQUERDA p/ dar lugar à legenda (topo-direito, fora do eixo)
        ax.set_title("Concentração das folhas entre amostras", fontsize=11,
                     fontweight="bold", color="#15324a", loc="left")
        # rótulo de % à DIREITA de cada barra (só oculta as nulas — sem massa naquela
        # amostra), em PRETO e NEGRITO; fonte fixa e legível (as barras não se
        # espremem mais — a figura é que cresce)
        for cont in conts:
            labels = [f"{v:.1f}%" if v > 0 else "" for v in (r.get_width() for r in cont)]
            ax.bar_label(cont, labels=labels, fontsize=7, padding=2,
                         color="black", fontweight="bold")
        if cols:
            # legenda ACIMA do eixo (topo-direito) — não cobre nenhuma barra
            ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.003),
                      ncol=max(1, min(len(cols), 3)), framealpha=0.9, borderaxespad=0.2)
        ax.grid(axis="x", alpha=0.15)
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
            ref_mask = self._smask(self.ref_sample)
            nonref = [a for a in self._samples
                      if a != self.ref_sample]
        else:
            ref_mask = pd.Series(True, index=self.df.index)
            nonref = []
        psi_on = with_psi and bool(nonref)

        # porção DES da folha (referência) recortada por POSIÇÃO e por coluna: o
        # frame folha∩DES com todas as variáveis (≈6 GB na raiz a 10M) não é mais
        # montado — cada variável recorta só a própria coluna dentro do laço
        pos_base = self._pos_of(leaf_mask & ref_mask)
        yb_all = self.df[self.target].iloc[pos_base].to_numpy(dtype="float64")
        ok_y = ~np.isnan(yb_all)
        yb = yb_all[ok_y]
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

        # posições por amostra DENTRO da folha (para o PSI nos bins do IV)
        pos_cur = {}
        n_ref_leaf = 0
        if psi_on:
            n_ref_leaf = len(pos_base)           # mesma máscara: folha ∩ DES
            pos_cur = {a: self._pos_of(leaf_mask & self._smask(a)) for a in nonref}
        # maus por faixa por contagem inteira quando o alvo (truncado) é 0/1
        bin01 = bool(self._is_clf and ok_base
                     and ((ybin == 0) | (ybin == 1)).all())

        # bins por variável: _resolve_bins_cached (em série) ou o lote paralelo
        # dos ajustes do optbinning com n_jobs > 1 — mesmos bins e mesmo cache
        obter_bins, fechar_bins = (
            self._bins_iv_resolvedor(sid, features, max_n_bins, min_bin_size)
            if ok_base else (None, None))
        rows = []
        erro = True
        try:
            for i, feat in enumerate(features):
                iv, nb, kind = np.nan, 0, "—"
                psi_vals = {a: np.nan for a in nonref}
                if ok_base:
                    try:
                        bins, _modo, kind = obter_bins(i, feat)
                        nb = len(bins)
                        if bins:
                            # IV por faixa: classificação = WoE binário (Siddiqi)
                            # Σ(dist_bons−dist_maus)·ln(dist_bons/dist_maus); regressão =
                            # IV contínuo Σ (n_i/N)·|média_bin − média_global|.
                            iv = 0.0
                            col_b = self.df[feat].iloc[pos_base]   # SÓ esta coluna
                            ids_b = self._bin_ids(col_b, bins)
                            if ids_b is None:
                                # caminho antigo: uma máscara (_mask_in) por faixa
                                for b in bins:
                                    m = self._mask_in_col(col_b, b).to_numpy()
                                    yi = yb_all[m]
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
                            else:
                                # código de faixa por linha: contagens só das linhas com
                                # alvo válido, na mesma ordem de faixa (faixa vazia pula)
                                ids_v = ids_b[ok_y]
                                if self._is_clf:
                                    dentro = ids_v >= 0
                                    n_k = np.bincount(ids_v[dentro], minlength=nb)
                                    bad_k = (np.bincount(ids_v[dentro & (ybin == 1)],
                                                         minlength=nb) if bin01 else None)
                                    for k in range(nb):
                                        if n_k[k] == 0:
                                            continue
                                        bad_i = (int(bad_k[k]) if bin01
                                                 else int(ybin[ids_v == k].sum()))
                                        good_i = int(n_k[k] - bad_i)
                                        dg = good_i / tot_good
                                        db = bad_i / tot_bad
                                        if dg > 0 and db > 0:
                                            iv += (dg - db) * np.log(dg / db)
                                else:
                                    # média por faixa sobre o MESMO subarray, na mesma
                                    # ordem (soma pairwise idêntica); nunca bincount
                                    # com pesos (soma sequencial ≠ pairwise)
                                    for k in range(nb):
                                        yi = yb[ids_v == k]
                                        if yi.size == 0:
                                            continue
                                        iv += (yi.size / n_base) * abs(float(yi.mean()) - mean_global)
                            # PSI sobre os MESMOS bins (DES × amostra, na folha): TODAS as
                            # linhas do recorte contam, inclusive as de alvo NaN
                            if psi_on and n_ref_leaf > 0:
                                ref_shares = [c / n_ref_leaf
                                              for c in self._bin_counts(col_b, bins, ids_b)]
                                for a in nonref:
                                    n_a = len(pos_cur[a])
                                    if n_a == 0:
                                        continue
                                    col_a = self.df[feat].iloc[pos_cur[a]]
                                    cur_shares = [c / n_a
                                                  for c in self._bin_counts(col_a, bins)]
                                    psi_vals[a] = round(_psi_from_shares(
                                        ref_shares, cur_shares, eps=1e-6), 4)
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
            erro = False
        finally:
            if fechar_bins is not None:
                fechar_bins(erro)       # encerra o executor (com erro: cancela)
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
        ``normalize=True`` devolve em % do total.

        Memoizado por versão da árvore: recalcula o IV (optbinning) de cada nó
        interno, então reabrir o Diagnóstico sem mexer na árvore não paga de novo."""
        return self._agg_memo(("feature_importance", bool(normalize)),
                              lambda: self._compute_feature_importance(normalize))

    def _compute_feature_importance(self, normalize: bool = True) -> pd.DataFrame:
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
                for m in self._bin_masks(leaf, feat, bins):
                    y = leaf.loc[m, self.target].to_numpy(dtype="float64")
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
    # ==================================================================
    # APELIDOS DE NEGÓCIO POR FOLHA (leaf_names)
    #   sid -> texto livre. Como o sid deriva dos rótulos de condição, ele MUDA
    #   quando a estrutura muda: merge/split renomeia a folha e o apelido do sid
    #   antigo fica órfão — órfãos são ignorados na leitura e descartados em
    #   silêncio no to_dict; um collapse que restaura o sid antigo revive o
    #   apelido. Quando definidos, os apelidos aparecem na tabela de folhas,
    #   no assign() (coluna extra), no to_sql (comentário por ramo do CASE) e
    #   nos gráficos com rótulo de folha.
    # ==================================================================
    def set_leaf_name(self, sid: str, name: str | None):
        """Define (ou remove, com ``name=None``/vazio) o apelido de negócio da
        folha ``sid``. O texto é normalizado (espaços/quebras colapsados).
        Devolve ``self`` para encadear."""
        if sid not in self.segments or not self.segments[sid]["is_leaf"]:
            raise ValueError(f"'{sid}' não é uma folha atual da árvore.")
        nome = " ".join(str(name).split()) if name is not None else ""
        if nome:
            self.leaf_names[sid] = nome
        else:
            self.leaf_names.pop(sid, None)
        return self

    def leaf_name(self, sid) -> str | None:
        """Apelido de negócio da folha ``sid`` — ``None`` sem apelido ou quando o
        sid não é mais uma folha (apelidos órfãos são ignorados)."""
        s = self.segments.get(sid)
        if s is None or not s["is_leaf"]:
            return None
        return self.leaf_names.get(sid) or None

    def _leaf_names_validos(self) -> dict:
        """Apelidos das folhas ATUAIS (descarta os órfãos de sids extintos)."""
        return {sid: nome for sid, nome in self.leaf_names.items()
                if nome and sid in self.segments and self.segments[sid]["is_leaf"]}

    def _leaf_tag(self, sid, nota, max_len: int = 24,
                  com_prefixo: bool = True) -> str:
        """Rótulo curto de folha p/ os gráficos: ``"folha N"`` (+ `` · apelido``
        truncado quando a folha tem apelido de negócio)."""
        base = f"folha {nota}" if com_prefixo else f"{nota}"
        nome = self.leaf_name(sid)
        if not nome:
            return base
        if max_len and len(nome) > max_len:
            nome = nome[:max_len - 1] + "…"
        return f"{base} · {nome}"

    def assign(
        self,
        col_name: str = "segmento",
        add_grade: bool = True,
        add_desc: bool = True,
        ascending: bool = True,
        add_name: bool = True,
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
        # coluna com o APELIDO de negócio da folha (NA onde não definido), logo
        # após a coluna do segmento — só quando há apelido em alguma folha atual
        nomes = self._leaf_names_validos() if add_name else {}
        if nomes:
            ap = out[col_name].map(lambda s: nomes.get(s, pd.NA))
            out.insert(out.columns.get_loc(col_name) + 1, f"{col_name}_apelido", ap)
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
                "weight_col": self.weight_col,
                "min_leaf_rows": self.min_leaf_rows,
                "feature_labels": dict(self.feature_labels),
                "problem_label": self.problem_label,
                "fallback": self.fallback,
                # apelidos de negócio das folhas ATUAIS — apelidos órfãos (o sid
                # deixou de existir após merge/split) são descartados em silêncio
                "leaf_names": self._leaf_names_validos(),
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
        seguinte (undo/redo) reusar as inalteradas em vez de recomputá-las.
        Devolve o próprio cache — pode ser passado a :meth:`from_dict` via
        ``mask_cache`` para amortecer reconstruções em memória (cenários)."""
        cache = self._mask_cache_by_conds
        if len(cache) > 4000:
            cache.clear()
        for s in self.segments.values():
            cache.setdefault(self._conds_key(self._conditions_json(s["conditions"])),
                             s["mask"])
        return cache

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
    def from_dict(cls, data: dict, df: pd.DataFrame, verbose: bool = False,
                  mask_cache: dict | None = None, copy: bool = True,
                  n_jobs: int | None = 1):
        """Reconstrói um segmentador a partir de `to_dict()` + um DataFrame.

        ``copy`` e ``n_jobs`` são repassados ao construtor (cópia profunda × rasa
        do ``df`` e threads do optbinning — ver ``TreeSegmenter.__init__``); não
        fazem parte do JSON.

        ``mask_cache`` (opcional): cache condições→máscara de OUTRA árvore sobre
        o **mesmo** ``df`` (o dict devolvido por :meth:`_prime_mask_cache`). O
        cache passa a ser COMPARTILHADO com a nova instância: segmentos cujas
        condições coincidem reusam a máscara pronta em vez de recomputá-la —
        amortiza reconstruções em memória (ex.: comparação de cenários na UI).
        Só passe um cache construído sobre o MESMO DataFrame."""
        meta = data.get("meta", {})
        # date_col entrou no meta depois (JSONs antigos não têm a chave → None);
        # se o df fornecido não tiver a coluna, degrada para None em vez de
        # falhar no construtor — a coluna é só para os gráficos no tempo.
        date_col = meta.get("date_col")
        if date_col is not None and date_col not in df.columns:
            date_col = None
        # weight_col entrou no meta depois (JSONs antigos → None); se o df
        # fornecido não tiver a coluna, degrada para None — a visão dupla é só
        # exibição (a estrutura da árvore não depende dos pesos).
        weight_col = meta.get("weight_col")
        if weight_col is not None and weight_col not in df.columns:
            weight_col = None
        seg = cls(df, target=meta.get("target", "target"),
                  task_type=meta.get("task_type", "classification"),
                  sample_col=meta.get("sample_col"),
                  ref_sample=meta.get("ref_sample", "DES"),
                  date_col=date_col,
                  weight_col=weight_col,
                  feature_labels=meta.get("feature_labels"),
                  problem_label=meta.get("problem_label"),
                  min_leaf_rows=meta.get("min_leaf_rows", 50),
                  verbose=verbose, copy=copy, n_jobs=n_jobs)
        # fallback persistido (JSONs antigos não têm a chave → None = sem fallback)
        seg.fallback = meta.get("fallback")
        if mask_cache is not None:      # cache compartilhado (mesmo df) — ver docstring
            seg._mask_cache_by_conds = mask_cache
        seg._load_segments(data["segments"])
        # apelidos de negócio por folha (JSONs antigos não têm a chave → {});
        # entradas cujo sid não é folha desta estrutura são descartadas
        nomes = meta.get("leaf_names") or {}
        seg.leaf_names = {sid: str(n) for sid, n in nomes.items()
                          if n and sid in seg.segments
                          and seg.segments[sid]["is_leaf"]}
        return seg

    @classmethod
    def load(cls, path: str, df: pd.DataFrame, verbose: bool = False,
             copy: bool = True, n_jobs: int | None = 1):
        """Carrega uma árvore salva em JSON e a aplica ao DataFrame fornecido.
        ``copy`` e ``n_jobs`` são repassados ao construtor (ver
        ``TreeSegmenter.__init__``)."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, df, verbose=verbose, copy=copy, n_jobs=n_jobs)

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
            sub = self._sub(seg["mask"])
            if self.sample_col is not None:
                ref = sub.loc[sub[self.sample_col] == self.ref_sample, self.target]
                pdv = float(ref.mean()) if len(ref) else float(sub[self.target].mean())
            else:
                pdv = float(sub[self.target].mean())
            if np.isnan(pdv):                       # folha vazia → alvo global (sem NaN na régua)
                if self.sample_col is not None:
                    g = self.df.loc[self._smask(self.ref_sample),
                                    self.target].mean()
                else:
                    g = self.df[self.target].mean()
                pdv = float(g) if not pd.isna(g) else 0.0
            leaves.append({"id": sid, "nota": int(nota_map[sid]),
                           "conditions": self._conditions_json(seg["conditions"]),
                           "pd": round(pdv, 6)})
        leaves.sort(key=lambda x: x["nota"])
        return {"target": self.target, "ref_sample": self.ref_sample, "leaves": leaves}

    def _fallback_resolvido(self, fallback):
        """Fallback efetivo de uma chamada de scoring/export: o parâmetro quando
        informado (inclusive ``None`` explícito = sem fallback), senão o
        persistido em ``self.fallback``."""
        return self.fallback if fallback is _FALLBACK_OMITIDO else fallback

    def n_orfas(self) -> int:
        """Linhas da base atual SEM folha na régua vigente (sem rota): valor
        faltante numa variável sem rota de faltantes ou categoria fora de toda
        folha. Esperado 0 na base de desenvolvimento; >0 antecipa linhas nulas
        no scoring (use ``fallback`` para roteá-las)."""
        # código -1 = sem folha (o mesmo NA da série de rótulos, sem montá-la)
        _ids, code = self._leaf_codes()
        return int((code < 0).sum())

    def predict(self, X: pd.DataFrame, col_seg="segmento",
                col_nota="nota", col_valor="valor_regua",
                fallback=_FALLBACK_OMITIDO) -> pd.DataFrame:
        """Aplica a régua a um DataFrame pandas novo: segmento, nota e alvo.

        ``fallback`` roteia as linhas órfãs (categoria não vista / faltante sem
        rota): ``None`` mantém nulo (comportamento padrão), ``'pior_nota'`` usa a
        folha de pior risco, um ``int`` usa a folha daquela nota. Omitido, vale o
        persistido em ``self.fallback``."""
        return _aplicar_regua_pandas(self._regua_dict(), X, col_seg, col_nota,
                                     col_valor,
                                     fallback=self._fallback_resolvido(fallback))

    # ------------------------------------------------------------------
    # TO_PYSPARK: gera o código da régua como F.when().otherwise() para
    #   aplicar a segmentação (segmento + nota + alvo) em escala no Spark.
    # ------------------------------------------------------------------
    def _bool_vals(self, feat, cats):
        """Para variável BOOLEANA na base (bool/boolean), os valores do grupo como
        literais booleanos (``[True]``, ``[False]`` ou os dois); ``None`` se a
        variável não for booleana. No pandas a faixa compara ``astype(str)`` com
        "True"/"False", mas no Spark ``CAST(bool AS STRING)`` dá "true"/"false"
        (minúsculo) — comparar como texto deixava TODA linha sem folha."""
        if feat not in self.df.columns or not _eh_booleana(self.df[feat]):
            return None
        return [v == "True" for v in ("True", "False") if v in {str(c) for c in cats}]

    def to_pyspark(self, func_name: str = "aplicar_regua",
                   fallback=_FALLBACK_OMITIDO) -> str:
        """Gera o código PySpark (string) que reproduz a régua (segmento, nota e alvo).

        ``fallback`` define o ``otherwise`` das linhas que não caem em folha
        nenhuma: ``None`` mantém nulo; ``'pior_nota'``/nota (int) grava a folha
        escolhida. Omitido, vale o persistido em ``self.fallback``.

        Contrato de faltantes: as colunas numéricas devem usar **NULL** (não ``NaN``)
        para ausente. No Spark um ``NaN`` numérico é ordenado como o MAIOR valor e
        ``isNull()`` não o captura, então um ``NaN`` literal cairia no bin mais alto —
        divergindo do pandas. Se a tabela puder conter ``NaN`` literal, normalize antes
        (``F.when(F.isnan(col), None).otherwise(col)``) ou use :meth:`apply_spark`, que
        já trata ``NaN`` como faltante internamente."""
        regua = self._regua_dict()
        fb = _folha_fallback(regua, self._fallback_resolvido(fallback))

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
                    vb = self._bool_vals(feat, c["cats"])
                    if vb is not None:
                        # booleana: compara com literal booleano (ver _bool_vals)
                        b = f'F.col("{feat}").cast("boolean")'
                        expr = (f'({b} == F.lit({vb[0]}))' if len(vb) == 1
                                else f'{b}.isNotNull()' if vb else "F.lit(False)")
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
            # linhas órfãs (sem rota): nulo por padrão; com fallback, a folha escolhida
            otherwise = "F.lit(None)" if fb is None else valfn(fb)
            out.append(f"           .otherwise({otherwise})")
            return "\n".join(out)

        fb_doc = ("" if fb is None else
                  f" Linhas sem rota vão p/ o fallback '{fb['id']}' (nota {fb['nota']}).")
        L = ["from pyspark.sql import functions as F", "",
             f"def {func_name}(df, col_seg='segmento', col_nota='nota', col_valor='valor_regua'):",
             f'    """Régua de {self._risk_word} gerada por TreeSegmenter (segmento, nota e {self._risk_word} por folha).{fb_doc}"""']
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
               col_nota: str = "folha", col_valor: str = "valor_previsto",
               fallback=_FALLBACK_OMITIDO) -> str:
        """Gera SQL com ``CASE WHEN`` que reproduz a régua. Pronto p/ copiar.

        Dialeto Spark/Databricks (também roda em Postgres/Oracle): variável booleana
        vira ``CAST(x AS BOOLEAN) = TRUE`` e texto com apóstrofo/barra invertida usa
        ``chr(39)``/``chr(92)`` concatenado com ``||`` (ver ``_common.sql_texto``).

        ``table`` é o nome da tabela/CTE de origem. Cada folha vira um ramo do
        CASE (na ordem da nota); a condição final usa as MESMAS regras de
        pandas/Spark (faixas (lo, hi], ``IN`` para categóricas, ``IS NULL`` para
        faltantes, ``include_na`` quando o nó de faltantes foi fundido).

        ``fallback`` define o ``ELSE`` das linhas que não caem em folha nenhuma
        (categoria não vista / faltante sem rota): ``None`` mantém ``ELSE NULL``;
        ``'pior_nota'``/nota (int) grava a folha escolhida. Omitido, vale o
        persistido em ``self.fallback``.

        Contrato de faltantes: colunas numéricas devem usar **NULL** (não ``NaN``)
        para ausente — no Spark um ``NaN`` é o MAIOR valor e ``IS NULL`` não o captura,
        então cairia no bin mais alto. Normalize ``NaN``→``NULL`` antes (ex.:
        ``nanvl``/``CASE WHEN isnan(x) THEN NULL``) ou use :meth:`apply_spark`."""
        regua = self._regua_dict()
        fb = _folha_fallback(regua, self._fallback_resolvido(fallback))

        def _q(v):                       # literal de categoria (ver _common.sql_texto)
            return _sql_texto(v)

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
                    vb = self._bool_vals(feat, c["cats"])
                    if vb is not None:
                        # booleana: literal booleano (CAST(bool AS STRING) dá "true"
                        # minúsculo no Spark e nunca casaria com 'True')
                        # CAST AS BOOLEAN: aceita BOOLEAN, int 0/1 e texto True/true
                        # (comparar int com TRUE falha no modo ANSI do Databricks)
                        b = f"CAST({feat} AS BOOLEAN)"
                        expr = (f"{b} = {str(vb[0]).upper()}" if len(vb) == 1
                                else f"{b} IS NOT NULL" if vb else "1=0")
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

        # apelido de negócio por folha → comentário no ramo do CASE (documenta a
        # régua no próprio SQL sem alterar a semântica)
        nomes = self._leaf_names_validos()

        def case(valfn, alias):
            linhas = [f"  CASE"]
            for leaf in regua["leaves"]:
                com = f"  -- {nomes[leaf['id']]}" if leaf["id"] in nomes else ""
                linhas.append(f"    WHEN {cond_sql(leaf['conditions'])} "
                              f"THEN {valfn(leaf)}{com}")
            # linhas órfãs (sem rota): NULL por padrão; com fallback, a folha escolhida
            linhas.append("    ELSE NULL" if fb is None else f"    ELSE {valfn(fb)}")
            linhas.append(f"  END AS {alias}")
            return "\n".join(linhas)

        cab = f"-- Régua de segmentação ({self.task_type}) gerada por TreeSegmenter"
        if fb is not None:
            cab += (f"\n-- fallback p/ linhas sem rota (não classificadas): "
                    f"'{fb['id']}' (nota {fb['nota']})")
        L = [cab,
             "SELECT",
             "  *,",
             case(lambda lf: _q(lf["id"]), col_seg) + ",",
             case(lambda lf: str(lf["nota"]), col_nota) + ",",
             case(lambda lf: str(lf["pd"]), col_valor),
             f"FROM {table};"]
        return "\n".join(L)

    # ------------------------------------------------------------------
    # TO_EXCEL: relatório multi-abas (.xlsx) — requer openpyxl (opcional).
    # ------------------------------------------------------------------
    def to_excel(self, path: str, table: str = "minha_tabela") -> str:
        """Exporta o modelo para um Excel multi-abas (``.xlsx``) em ``path``.

        Requer o pacote OPCIONAL **openpyxl** — o import é lazy (só aqui) e,
        sem ele instalado, sobe um :class:`ImportError` com instrução amigável.

        Abas geradas (uma aba indisponível é pulada — ex.: PSI sem ``sample_col``):

        - **Folhas** — tabela completa numérica das folhas (com PSI e teste de
          hipótese entre folhas adjacentes, quando aplicável; com ``weight_col``
          configurada, sai também a visão dupla contratos × saldo — ``saldo_%``,
          ``valor_medio_pond`` e ``valor_pond_<amostra>``);
        - **Métricas por amostra** — :meth:`metrics`;
        - **PSI** — resumo por amostra (:meth:`psi`) e, logo abaixo, o detalhe
          da contribuição por folha (:meth:`psi_detalhe`);
        - **IV por variável** — :meth:`variable_iv` na raiz (carteira inteira);
        - **Calibração** — :meth:`calibration_table`;
        - **Régua SQL** — o ``CASE WHEN`` de :meth:`to_sql` como texto
          (``table`` é o nome da tabela de origem no SQL).

        Formatação mínima: cabeçalho CONGELADO em todas as abas e colunas de
        percentual como % de verdade no Excel — representatividade vira fração
        0–1 com formato ``0.00%`` (e, na classificação, as colunas de alvo/taxa
        também ganham o formato, já que são frações).

        Retorna o próprio ``path``.
        """
        try:
            import openpyxl  # noqa: F401  — dependência OPCIONAL (import lazy)
        except ImportError as e:
            raise ImportError(
                "A exportação para Excel requer o pacote opcional 'openpyxl' "
                "(instale com: pip install openpyxl).") from e
        from openpyxl.utils import get_column_letter

        # colunas exibidas como % no Excel: {coluna: divisor} — representativi-
        # dade e % do saldo (visão dupla de weight_col) vêm 0–100 (÷100); na
        # classificação, alvo/taxa (inclusive as ponderadas) já são frações (÷1)
        def _pct_cols(df: pd.DataFrame) -> dict:
            cols = {}
            for c in df.columns:
                cs = str(c)
                if (cs in ("repr_%", "saldo_%")
                        or (cs.startswith("repr_") and cs.endswith("_%"))):
                    cols[c] = 100.0
                elif self._is_clf and (cs.startswith("valor_") or cs == "valor_medio"
                                       or cs == "taxa_default"):
                    cols[c] = 1.0
            return cols

        # cada aba é montada de forma independente: uma falha pontual (ex.:
        # métrica sem alvo na amostra) pula a aba em vez de derrubar o arquivo
        def _tenta(build):
            try:
                return build()
            except Exception:
                return None

        lv = self.leaves(with_psi=self.sample_col is not None, with_test=True)
        met = _tenta(self.metrics)
        psi_df = psi_det = None
        if self.sample_col is not None:
            psi_df = _tenta(self.psi)
            if psi_df is not None:
                psi_det = _tenta(self.psi_detalhe)
        ivt = _tenta(self.variable_iv)
        ct = _tenta(self.calibration_table)
        sql = _tenta(lambda: self.to_sql(table=table))

        with pd.ExcelWriter(path, engine="openpyxl") as writer:

            def _write(nome, df, startrow=0, freeze=True):
                df = df.copy()
                pct = _pct_cols(df)
                for c, den in pct.items():
                    df[c] = pd.to_numeric(df[c], errors="coerce") / den
                df.to_excel(writer, sheet_name=nome, index=False, startrow=startrow)
                ws = writer.sheets[nome]
                if freeze:
                    ws.freeze_panes = "A2"          # congela o cabeçalho
                idx = {str(c): i + 1 for i, c in enumerate(df.columns)}
                for c in pct:                        # formato de % nas células
                    letra = get_column_letter(idx[str(c)])
                    ini = startrow + 2               # 1-based, abaixo do cabeçalho
                    for row in range(ini, ini + len(df)):
                        ws[f"{letra}{row}"].number_format = "0.00%"
                return ws

            _write("Folhas", lv)
            if met is not None:
                _write("Métricas por amostra", met)
            if psi_df is not None:
                ws = _write("PSI", psi_df)
                if psi_det is not None and len(psi_det):
                    start = len(psi_df) + 3          # 2 linhas em branco após o resumo
                    ws.cell(row=start, column=1,
                            value="Detalhe — contribuição de cada folha p/ o PSI")
                    _write("PSI", psi_det, startrow=start, freeze=False)
            if ivt is not None:
                _write("IV por variável", ivt)
            if ct is not None:
                _write("Calibração", ct)
            if sql:
                ws = writer.book.create_sheet("Régua SQL")
                for i, ln in enumerate(str(sql).split("\n"), start=1):
                    ws.cell(row=i, column=1, value=ln)
        return path

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
                    col_nota: str = "nota", col_valor: str = "valor_regua",
                    fallback=_FALLBACK_OMITIDO, progress_callback=None):
        """Aplica a régua num Spark DataFrame (segmento, nota e alvo).

        ``fallback`` roteia as linhas órfãs (sem rota) para a folha escolhida
        (``'pior_nota'``/nota int) em vez de deixá-las nulas; omitido, vale o
        persistido em ``self.fallback``.

        ``progress_callback`` (opcional): ``cb(key, label, status, detail)``
        chamado ao montar o plano — ver :func:`_emit_progress`."""
        try:
            from pyspark.sql import functions as F
        except ImportError as e:  # pragma: no cover
            raise ImportError("apply_spark requer pyspark — use: pip install pyspark") from e

        _emit_progress(progress_callback, "aplicar",
                       "Aplicar a régua (montar o plano Spark)", "run")
        regua = self._regua_dict()
        if not regua["leaves"]:
            raise ValueError("Árvore sem folhas — cresça a segmentação antes de aplicar.")
        fb = _folha_fallback(regua, self._fallback_resolvido(fallback))

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
                    vb = self._bool_vals(feat, c["cats"])
                    if vb is not None:        # booleana: literal booleano (ver _bool_vals)
                        fb = fc.cast("boolean")
                        part = ((fb == F.lit(vb[0])) if len(vb) == 1
                                else fb.isNotNull() if vb else F.lit(False))
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

        # linhas órfãs (sem rota): nulo por padrão; com fallback, a folha escolhida
        if fb is None:
            o_seg = o_nota = o_val = F.lit(None)
        else:
            o_seg, o_nota, o_val = (F.lit(fb["id"]), F.lit(fb["nota"]),
                                    F.lit(fb["pd"]))
        out = (sdf.withColumn(col_seg, seg_col.otherwise(o_seg))
                  .withColumn(col_nota, nota_col.otherwise(o_nota))
                  .withColumn(col_valor, pd_col.otherwise(o_val)))
        _emit_progress(progress_callback, "aplicar",
                       "Aplicar a régua (montar o plano Spark)", "ok",
                       f"{len(regua['leaves'])} folhas")
        return out

    # ------------------------------------------------------------------
    # APPLY_TABLE: aplica a régua numa TABELA (nome Databricks, Spark DataFrame
    #   ou pandas em memória) com progresso por etapa e resumo por folha —
    #   fachada usada pelo card "Reconstruir folhas" da UI.
    # ------------------------------------------------------------------
    def apply_table(self, data, col_seg: str = "segmento",
                    col_nota: str = "nota", col_valor: str = "valor_regua",
                    fallback=_FALLBACK_OMITIDO, output_table=None,
                    mode: str = "overwrite", spark=None,
                    progress_callback=None):
        """Aplica a régua numa tabela e devolve ``(saida, resumo)``.

        ``data`` pode ser o **nome** de uma tabela do Databricks
        (``catalogo.schema.tabela``), um **Spark DataFrame** ou um **pandas
        DataFrame** em memória:

        - nome de tabela ou Spark DataFrame → ``saida`` é um **Spark DataFrame**
          com as colunas da régua (e grava em ``output_table`` quando
          informado, ``mode='overwrite'`` por padrão);
        - pandas DataFrame → ``saida`` é **pandas** (a base + segmento, nota e
          valor), sem exigir Spark.

        ``resumo`` é um ``pandas.DataFrame`` com a distribuição por folha
        (``col_nota`` · ``linhas`` · ``pct``), calculado SEM reexecutar o plano
        Spark: quando a saída é gravada, ele é lido da própria tabela já
        materializada; sem gravação, a contagem é a única ação executada.

        ``progress_callback`` (opcional): ``cb(key, label, status, detail)``
        chamado a cada etapa (ler tabela → aplicar régua → gravar → resumo) —
        útil p/ uma tabela de progresso na UI. Ver :func:`_emit_progress`."""
        cb = progress_callback
        if isinstance(data, str):                       # nome de tabela Databricks
            try:
                from pyspark.sql import SparkSession
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "aplicar por nome de tabela requer pyspark — no Databricks "
                    "já vem no cluster; fora dele: pip install pyspark") from e
            _emit_progress(cb, "load", f"Ler a tabela '{data}'", "run")
            spark = (spark or SparkSession.getActiveSession()
                     or SparkSession.builder.getOrCreate())
            data = spark.table(data)
            _emit_progress(cb, "load", "Ler a tabela", "ok")
        if hasattr(data, "toPandas"):                   # Spark DataFrame
            out = self.apply_spark(data, col_seg=col_seg, col_nota=col_nota,
                                   col_valor=col_valor, fallback=fallback,
                                   progress_callback=cb)
            base_resumo = out
            if output_table:
                _emit_progress(cb, "save", f"Gravar em '{output_table}'", "run")
                # mergeSchema: evolui o schema (colunas novas: segmento/nota/
                # valor) ao sobrescrever a base se ela já existir.
                (out.write.mode(mode).option("mergeSchema", "true")
                    .saveAsTable(output_table))
                _emit_progress(cb, "save", f"Gravar em '{output_table}'", "ok")
                # resumo sobre a tabela GRAVADA (já materializada) — evita
                # reexecutar o plano lazy inteiro só para contar por folha
                sess = spark or getattr(out, "sparkSession", None)
                if sess is None:  # pragma: no cover - Spark antigo sem o atributo
                    from pyspark.sql import SparkSession
                    sess = SparkSession.getActiveSession()
                if sess is not None:
                    base_resumo = sess.table(output_table)
            _emit_progress(cb, "resumo", "Resumir a distribuição por folha", "run")
            resumo = (base_resumo.groupBy(col_nota).count()
                      .orderBy(col_nota).toPandas()
                      .rename(columns={"count": "linhas"}))
            resumo["pct"] = resumo["linhas"] / max(int(resumo["linhas"].sum()), 1)
            _emit_progress(cb, "resumo", "Resumir a distribuição por folha", "ok",
                           f"{int(resumo['linhas'].sum()):,} linhas".replace(",", "."))
            _emit_progress(cb, "done", "Aplicação concluída", "ok")
            return out, resumo
        # --- pandas puro (sem Spark): base em memória ---
        pdf = pd.DataFrame(data)
        _emit_progress(cb, "aplicar", "Aplicar a régua (pandas)", "run")
        extra = self.predict(pdf, col_seg=col_seg, col_nota=col_nota,
                             col_valor=col_valor, fallback=fallback)
        out = pdf.copy()
        for c in extra.columns:            # sobrescreve colisões, como no Spark
            out[c] = extra[c]
        _emit_progress(cb, "aplicar", "Aplicar a régua (pandas)", "ok",
                       f"{len(out):,} linhas".replace(",", "."))
        _emit_progress(cb, "resumo", "Resumir a distribuição por folha", "run")
        vc = out[col_nota].value_counts(dropna=False).sort_index()
        resumo = pd.DataFrame({col_nota: vc.index.to_numpy(),
                               "linhas": vc.to_numpy()})
        resumo["pct"] = resumo["linhas"] / max(int(resumo["linhas"].sum()), 1)
        _emit_progress(cb, "resumo", "Resumir a distribuição por folha", "ok")
        _emit_progress(cb, "done", "Aplicação concluída", "ok")
        return out, resumo

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
        import heapq

        N = len(self.df)
        if not N:
            return None, "num"
        kind = self._detect_kind(sub, feature, dtype)
        col = sub[feature]
        na_cnt = int(col.isna().sum())
        na_present = na_cnt > 0
        # o grupo de faltantes é checado ANTES de montar os bins finos: abaixo do
        # mínimo a variável é descartada de qualquer jeito (mesmo (None, kind) que
        # o retorno lá embaixo), sem pagar quantil/groupby à toa.
        if na_present and min_leaf_repr is not None and na_cnt / N < min_leaf_repr - 1e-9:
            return None, kind

        # `cnt` = contagem (int64) de linhas por bin fino. Com ela a concentração
        # de um bin fundido é a SOMA das contagens das partes, sem varrer a folha
        # de novo a cada fusão (o laço antigo refazia K máscaras por iteração,
        # O(K²·n) — travava o autofit com categórica de alta cardinalidade, ex.:
        # Decimal com ~100k níveis). None = cai no caminho antigo (grepr).
        cnt = None
        if kind == "num":
            x = col.dropna().to_numpy(dtype="float64")
            if x.size == 0 or np.unique(x).size < 2:
                return None, kind
            qs = np.quantile(x, np.linspace(0.0, 1.0, 25)[1:-1])   # ~24 faixas finas
            cuts = sorted({round(float(c), 10) for c in qs})
            edges = [-np.inf, *cuts, np.inf]
            core = [{"kind": "num", "lo": edges[i], "hi": edges[i + 1]}
                    for i in range(len(edges) - 1)]
            # contagem aditiva só quando ela é EXATAMENTE a do between por bin:
            # cortes estritamente crescentes e sem NaN (±inf repetidos fazem o
            # np.quantile devolver corte NaN, que o sorted deixa fora de ordem e o
            # between conta diferente) e coluna float64/inteira (numpy ou anulável),
            # cuja comparação com o limite float é feita em float64 — float32
            # compararia em float32 e divergiria; dtype pyarrow compara pelo
            # pyarrow. Fora disso, caminho antigo.
            cuts_arr = np.asarray(cuts, dtype="float64")
            mascarado = isinstance(col.array, (pd.arrays.IntegerArray,
                                               pd.arrays.FloatingArray))
            base_dt = col.dtype.numpy_dtype if mascarado else col.dtype
            if (isinstance(base_dt, np.dtype)
                    and (base_dt == np.float64 or base_dt.kind in "iu")
                    and not np.isnan(cuts_arr).any()
                    and bool(np.all(np.diff(cuts_arr) > 0))):
                # (lo, hi] do between = searchsorted 'left'; -inf não cai em bin
                # nenhum (lo=-inf é exclusivo) e +inf vai para o último; NaN que
                # sobreviva ao dropna (Float64 anulável com 0/0) também fica de fora
                xs = x[~np.isnan(x) & (x != -np.inf)]
                cnt = np.bincount(np.searchsorted(cuts_arr, xs, side="left"),
                                  minlength=len(core)).astype(np.int64)
        else:
            # uma categoria por grupo, ordenadas por alvo em DES (contiguidade de risco)
            ref = self._fit_frame(sub, 0.0).dropna(subset=[feature, self.target])
            means = (ref.groupby(ref[feature].astype(str))[self.target].mean()
                     .sort_values())
            present = [str(c) for c in pd.unique(col.dropna().astype(str))]
            # pertinência por set (por lista eram O(K²) comparações de string em
            # alta cardinalidade); a ordem resultante é a mesma
            present_set = set(present)
            ordered = [c for c in means.index if c in present_set]
            seen = set(ordered)
            ordered += [c for c in present if c not in seen]
            if len(ordered) < 2:
                return None, kind
            core = [{"kind": "cat", "cats": [c]} for c in ordered]
            # contagem por categoria sobre a coluna INTEIRA convertida a texto — a
            # mesma base do astype(str).isin do _mask_in, inclusive o 'nan'/'None'
            # literal que colide com o nulo convertido; repr do grupo = soma
            vc = col.astype(str).value_counts(dropna=True)
            vcd = dict(zip(vc.index, vc.to_numpy()))
            cnt = np.array([int(vcd.get(c, 0)) for c in ordered], dtype=np.int64)

        na = [{"kind": "na"}] if na_present else []

        if cnt is None:
            # caminho antigo: concentração medida por máscara a cada iteração
            def grepr(b):
                return int(self._mask_in(sub, feature, b).sum()) / N

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

        # caminho aditivo: mesma sequência de fusões do laço antigo, em O(K log K).
        # Cada bin vivo é um nó de lista ligada identificado pelo índice do seu 1º
        # bin fino (a ordem dos ids = a ordem das faixas) e cobre o trecho
        # [no, fim[no]]. O heap guarda (contagem, id): o topo válido é o argmin da
        # concentração com desempate pelo PRIMEIRO índice, como o np.argmin — a
        # contagem/N é estritamente crescente na contagem. Entradas velhas (nó
        # removido ou contagem desatualizada) são descartadas ao sair do topo.
        n_core = len(core)
        cl = cnt.tolist()
        fim = list(range(n_core))
        ant = list(range(-1, n_core - 1))
        prox = [*range(1, n_core), -1]
        morto = [False] * n_core
        vivos = n_core
        heap = [(cl[i], i) for i in range(n_core)]
        heapq.heapify(heap)
        while vivos > 1 and min_leaf_repr is not None:
            c, i = heap[0]
            if morto[i] or cl[i] != c:
                heapq.heappop(heap)
                continue
            ri = cl[i] / N
            if ri >= min_leaf_repr - 1e-9:
                break
            cands = [k for k in (ant[i], prox[i]) if k >= 0]
            # teto na MESMA forma do laço antigo: soma de duas frações (nunca
            # (a+b)/N, que pode diferir no último ulp)
            ok = ([k for k in cands if ri + cl[k] / N <= max_bin_repr + 1e-9]
                  if max_bin_repr is not None else cands)
            j = min(ok or cands, key=lambda k: cl[k] / N)
            a, b2 = (i, j) if i < j else (j, i)
            cl[a] += cl[b2]
            fim[a] = fim[b2]
            prox[a] = prox[b2]
            if prox[b2] >= 0:
                ant[prox[b2]] = a
            morto[b2] = True
            vivos -= 1
            heapq.heappush(heap, (cl[a], a))

        if vivos < 2:
            return None, kind
        # materializa os bins só no fim (o nó 0 nunca é removido: não tem vizinho
        # à esquerda, então é sempre o `a` da fusão)
        nos, no = [], 0
        while no >= 0:
            nos.append(no)
            no = prox[no]
        if kind == "num":
            out = [{"kind": "num", "lo": edges[no], "hi": edges[fim[no] + 1]} for no in nos]
        else:
            out = [{"kind": "cat", "cats": ordered[no:fim[no] + 1]} for no in nos]
        out += na
        reps = [cl[no] / N for no in nos] + ([na_cnt / N] if na else [])
        if min_leaf_repr is not None and any(r < min_leaf_repr - 1e-9 for r in reps):
            return None, kind
        if max_bin_repr is not None and any(r > max_bin_repr + 1e-9 for r in reps):
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
                sub = self._sub(self.segments[sid]["mask"])
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
                            self._sub(self.segments[sid]["mask"], var), var,
                            min_leaf_repr, max_bin_repr)
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
                    dev_df = self.df[self._smask(self.ref_sample)]
                    if val_sample is not None:
                        oot_df = self.df[self._smask(val_sample)]
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
