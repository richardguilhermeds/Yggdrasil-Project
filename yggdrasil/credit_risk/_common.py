"""
yggdrasil.credit_risk._common
==============================
Helpers PUROS compartilhados entre os segmentadores de **árvore**
(:mod:`yggdrasil.credit_risk.tree`) e de **modelo**
(:mod:`yggdrasil.credit_risk.model`). Fonte ÚNICA para formatação de faixas,
a fórmula do PSI (:func:`psi_from_shares`), classificação de PSI/IV, contagem de
inversões e o ajuste do optbinning — antes essas funções eram copiadas nos dois
módulos e já haviam começado a **divergir** (guard de NaN no PSI, default de
``task_type`` no IV). Centralizá-las aqui elimina o drift.
"""
from __future__ import annotations

import contextlib
import sys
import threading
import types
import warnings

import numpy as np
import pandas as pd

# limiares únicos do repositório (< PSI_STABLE estável · < PSI_SIGNIFICANT
# atenção · acima instável) — import leve: monitoring.psi só usa numpy/pandas.
from ..monitoring.psi import PSI_SIGNIFICANT, PSI_STABLE


def fmt(x: float) -> str:
    """Formata limites de faixa de forma legível."""
    if x == -np.inf:
        return "-inf"
    if x == np.inf:
        return "inf"
    return f"{x:.4g}"


def psi_from_shares(p_ref, p_cur, eps: float = 1e-6, return_contrib: bool = False):
    """PSI entre duas distribuições de participação por faixa (*shares*).

    Fórmula clássica, por faixa: ``(p_cur − p_ref)·ln(p_cur/p_ref)``, com cada
    participação truncada por baixo em ``eps`` (faixa vazia não gera ``log(0)``
    nem divisão por zero — mesmo tratamento de todos os segmentadores). A soma é
    **sequencial na ordem das faixas**, preservando o comportamento numérico dos
    laços originais que esta função substitui.

    Parameters
    ----------
    p_ref, p_cur:
        Sequências de participações (mesmo comprimento e mesma ordem de faixa).
        Valores já truncados em ``eps`` na origem não mudam (``max`` idempotente).
    eps:
        Piso de cada participação antes do log.
    return_contrib:
        Se ``True``, devolve ``(psi_total, contribuicoes)`` com a parcela de cada
        faixa — para tabelas de decomposição (``psi_detalhe``/``csi_detalhe``).

    Returns
    -------
    float | tuple[float, list[float]]
    """
    contribs = []
    for r, c in zip(p_ref, p_cur):
        r = max(float(r), eps)
        c = max(float(c), eps)
        contribs.append(float((c - r) * np.log(c / r)))
    total = float(sum(contribs))
    return (total, contribs) if return_contrib else total


def classifica_psi(psi) -> str:
    """Classificação usual de PSI para monitoramento de estabilidade.

    Limiares únicos do repositório (:mod:`yggdrasil.monitoring.psi`):
    ``< PSI_STABLE`` estável · ``< PSI_SIGNIFICANT`` atenção · acima, instável.
    ``None``/``NaN`` → ``"—"`` (um PSI indefinido não é 'instável')."""
    if psi is None or (isinstance(psi, float) and np.isnan(psi)):
        return "—"
    if psi < PSI_STABLE:
        return "estável"
    if psi < PSI_SIGNIFICANT:
        return "atenção"
    return "instável"


def classifica_iv(iv, task_type: str) -> str:
    """Faixas de força do IV, conforme o tipo de alvo (``task_type`` OBRIGATÓRIO —
    sem default, para nunca classificar IV de regressão pela escala binária por
    engano).

    classification → IV **binário** (WoE/Siddiqi): < 0.02 inútil · 0.02–0.10 fraco
        · 0.10–0.30 médio · 0.30–0.50 forte · ≥ 0.50 suspeito.
    regression → IV **contínuo** (desvio absoluto médio ponderado do alvo por
        faixa, escala menor): < 0.01 inútil · 0.01–0.03 fraco · 0.03–0.10 médio ·
        0.10–0.35 forte · ≥ 0.35 suspeito."""
    if iv is None or (isinstance(iv, float) and np.isnan(iv)):
        return "—"
    faixas = ((0.02, 0.10, 0.30, 0.50) if task_type == "classification"
              else (0.01, 0.03, 0.10, 0.35))
    for lim, rot in zip(faixas, ("inútil", "fraco", "médio", "forte")):
        if iv < lim:
            return rot
    return "suspeito"


def count_inversions(ordered, values) -> tuple:
    """Nº de pares invertidos vs. a ordem de referência e nº de pares comparáveis.

    ``ordered`` = chaves na ordem de risco de referência (crescente); ``values`` =
    dict chave→risco num ponto (amostra/safra). Par (i<j na ref.) inverte quando
    risco_i > risco_j. Pares com valor faltante (NaN) são ignorados."""
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


def fmt_safras(safras) -> list:
    """Rótulos de safra → 'mmm/aa' (padrão de mês/ano do repositório). Delega ao
    helper único :func:`yggdrasil.reporting.style.fmt_month_year`."""
    from ..reporting.style import fmt_month_year
    return fmt_month_year(safras)


# Teto de linhas do AJUSTE do optbinning. Acima disso os cortes são aprendidos
# numa amostra aleatória (semente fixa ⇒ determinístico); IV, tabelas e PSI
# continuam calculados na base inteira com esses cortes. A pré-binagem do
# optbinning (árvore CART sobre todas as linhas) era ~70% do tempo de abrir a UI
# em bases de milhões de linhas; com 300 mil linhas (≥15 mil por faixa com o
# mínimo de 5%) os cortes são estáveis e bases menores não mudam nada. 0/None
# desliga.
OPTBINNING_MAX_ROWS = 300_000
OPTBINNING_SEED = 42


def _amostra_optbinning(n, max_rows=None):
    """Posições (ordenadas) da amostra de ajuste do optbinning sobre ``n``
    linhas, ou ``None`` quando não há amostragem (``n`` dentro do teto ou teto
    0/None). Mesmo sorteio de :func:`fit_optbinning_splits` — quem já tem as
    linhas válidas em mãos sorteia aqui e entrega só as escolhidas.
    :data:`OPTBINNING_MAX_ROWS` é lido na hora da chamada."""
    cap = OPTBINNING_MAX_ROWS if max_rows is None else max_rows
    if cap and n > cap:
        return np.sort(np.random.default_rng(OPTBINNING_SEED).choice(n, cap,
                                                                    replace=False))
    return None


# ---- CP-SAT do optbinning com 1 worker (determinismo) ----
# O optbinning cria o solver em ``optbinning.binning.cp`` (``BinningCP.solve``,
# herdado pelo ContinuousBinningCP) com ``cp_model.CpSolver()`` fixando só o
# tempo máximo: o CP-SAT usa então TODOS os núcleos em paralelo e, quando há
# EMPATE de ótimo, devolve soluções diferentes entre execuções (ex.: grupos
# {2.00}/{0.1, 0.10} × {2.00, 0.1}/{0.10} numa categórica). Com 1 worker a busca
# é determinística — mesmo (x, y) ⇒ mesmos cortes — e, nestes modelos pequenos
# (≤ max_n_prebins faixas), até mais rápida; os núcleos ficam livres para
# paralelizar ENTRE variáveis (n_jobs).
#
# Desenho: só o nome ``cp_model`` DENTRO de ``optbinning.binning.cp`` passa a
# apontar para um proxy que delega tudo ao módulo real do OR-Tools, exceto
# ``CpSolver``: uma subclasse que fixa 1 worker SÓ quando a thread corrente está
# dentro de um ajuste da lib (contador POR THREAD, :func:`_cpsat_um_worker`).
# Fora disso — ajustes do próprio usuário, outras threads — o solver nasce igual
# ao original, e o ``cp_model`` global do OR-Tools fica intocado. Preferido a
# trocar/restaurar o atributo a cada ajuste com contador global + lock porque
# (a) a troca valeria também para ajustes de OUTRAS threads do usuário enquanto
# a lib ajusta e (b) nada é trocado por ajuste: as threads do lote paralelo só
# mexem no próprio contador, sem corrida. A instalação é única (lock + flag) e
# idempotente (inclusive após reload deste módulo); se o layout do optbinning
# mudar (sem ``binning.cp.cp_model``), não instala e o ajuste segue como antes.
#
# Os workers entram na hora do ``Solve``, e SÓ se ninguém escolheu antes: o
# optbinning >= 1.0 já fixa ``num_search_workers = 1`` (e ``random_seed = 0``)
# logo depois de criar o solver, e o CP-SAT recusa o modelo (``MODEL_INVALID``,
# "Do not specify both num_search_workers and num_workers") com os DOIS campos
# preenchidos — o optbinning então devolve um bin só, calado. Nessas versões o
# proxy não mexe em nada; nas antigas (0.19/0.20) fixa o 1 worker.
# OPTBINNING_CPSAT_WORKERS é lido a cada ``Solve``; 0/None devolve o padrão do
# optbinning (no 0.19/0.20: todos os núcleos, não determinístico em empate).
OPTBINNING_CPSAT_WORKERS = 1

_CPSAT_TLS = threading.local()        # .n = profundidade de ajustes da lib nesta thread
_CPSAT_LOCK = threading.Lock()
_CPSAT_INSTALADO = False


class _CpModelProxy(types.ModuleType):
    """Fachada do ``ortools.sat.python.cp_model`` vista só pelo
    ``optbinning.binning.cp``: tudo vem do módulo real (``__getattr__``), menos
    ``CpSolver`` — subclasse que, dentro de :func:`_cpsat_um_worker`, resolve com
    :data:`OPTBINNING_CPSAT_WORKERS` workers (se quem criou o solver não fixou
    os workers por conta própria).

    A subclasse é montada sobre o ``CpSolver`` que o módulo real tem NA HORA da
    leitura (e reaproveitada enquanto ele não muda): um patch do usuário em
    ``cp_model.CpSolver`` continua chegando ao optbinning, e desfazê-lo também —
    nada fica congelado no que existia quando o proxy foi instalado."""

    def __init__(self, real):
        super().__init__(real.__name__, real.__doc__)
        self._yg_real = real
        self._yg_sub = (None, None)       # (CpSolver base, subclasse) da última leitura

    def __getattr__(self, nome):
        if nome == "CpSolver":
            return self._yg_cpsolver()
        return getattr(self._yg_real, nome)

    def _yg_cpsolver(self):
        base = self._yg_real.CpSolver
        visto, sub = self._yg_sub
        if visto is base:
            return sub
        sub = _subclasse_cpsolver(base)
        self._yg_sub = (base, sub)
        return sub


def _subclasse_cpsolver(base):
    """``CpSolver`` que fixa :data:`OPTBINNING_CPSAT_WORKERS` workers no
    ``Solve`` de um ajuste da lib (ver :class:`_CpModelProxy`)."""

    class CpSolver(base):
        def _yg_fixa_workers(self):
            w = OPTBINNING_CPSAT_WORKERS
            if not (w and getattr(_CPSAT_TLS, "n", 0)):
                return
            p = self.parameters
            # quem criou o solver já escolheu (optbinning >= 1.0): os dois
            # campos juntos dão MODEL_INVALID — respeita a escolha.
            if getattr(p, "num_workers", 0) or getattr(p, "num_search_workers", 0):
                return
            try:
                p.num_workers = int(w)
            except AttributeError:   # OR-Tools antigo: só o nome velho
                p.num_search_workers = int(w)

        # ``Solve`` (o que o optbinning chama) e ``solve`` (o nome novo — no
        # OR-Tools recente ``Solve`` só delega a ele) separados, cada um pelo
        # ``super()``: um alias ``solve = Solve`` entraria em recursão.
        def Solve(self, *args, **kwargs):
            self._yg_fixa_workers()
            return super().Solve(*args, **kwargs)

        if hasattr(base, "solve"):
            def solve(self, *args, **kwargs):
                self._yg_fixa_workers()
                return super().solve(*args, **kwargs)

    CpSolver.__qualname__ = "CpSolver"
    return CpSolver


def _instala_cpsat_um_worker() -> bool:
    """Instala (uma vez) o :class:`_CpModelProxy` em ``optbinning.binning.cp``.
    Devolve ``False`` se o optbinning não tiver o layout esperado."""
    global _CPSAT_INSTALADO
    if _CPSAT_INSTALADO:
        return True
    with _CPSAT_LOCK:
        if _CPSAT_INSTALADO:
            return True
        try:
            import optbinning.binning.cp as ob_cp
            atual = ob_cp.cp_model
            real = atual.__dict__.get("_yg_real", atual)   # proxy de um reload anterior
            if not isinstance(real, types.ModuleType) or not hasattr(real, "CpSolver"):
                return False
            ob_cp.cp_model = _CpModelProxy(real)
        except Exception:
            return False
        _CPSAT_INSTALADO = True
        return True


@contextlib.contextmanager
def _cpsat_um_worker():
    """Escopo de um ajuste da lib: os ``CpSolver`` que o optbinning usar NESTA
    thread resolvem com :data:`OPTBINNING_CPSAT_WORKERS` workers — se o próprio
    optbinning não fixou os workers (o >= 1.0 já fixa 1). Reentrante e por
    thread (seguro no ``ThreadPoolExecutor`` dos lotes paralelos)."""
    _instala_cpsat_um_worker()
    tls = _CPSAT_TLS
    tls.n = getattr(tls, "n", 0) + 1
    try:
        yield
    finally:
        tls.n -= 1


def fit_optbinning_splits(b, x, y, max_rows=None, idx=None) -> list:
    """Roda ``b.fit(x, y)`` e devolve ``list(b.splits)``.

    Silencia os ``RuntimeWarning`` de "divide by zero" benignos do optbinning
    (em ``auto_monotonic``, quando algum prebin fica com 0 registros) — o ajuste
    ainda produz cortes válidos. Devolve ``[]`` se o ajuste falhar.

    ``ValueError`` (problema inviável / sem corte) é o caminho esperado e fica
    silencioso. Qualquer outra exceção (ex.: incompatibilidade de versão de
    dependência) é **avisada** em vez de mascarada como "sem corte válido".

    ``max_rows`` (padrão :data:`OPTBINNING_MAX_ROWS`): com mais linhas, ajusta
    numa amostra aleatória de ``max_rows`` (semente fixa). ``idx`` (opcional):
    posições já sorteadas (:func:`_amostra_optbinning`) — com ``None`` o sorteio
    é feito aqui, exatamente como antes. O CP-SAT do ajuste roda com 1 worker
    (:func:`_cpsat_um_worker`): mesmo (x, y) ⇒ mesmos cortes, inclusive em empate."""
    x, y = _recorta_amostra_optbinning(x, y, max_rows, idx)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(divide="ignore", invalid="ignore"), _cpsat_um_worker():
                b.fit(x, y)
        checa_status_optbinning(b)
        return list(b.splits)
    except ValueError:
        return []
    except Exception as e:
        avisa_falha_optbinning(b, e)
        return []


def _recorta_amostra_optbinning(x, y, max_rows=None, idx=None):
    """``(x, y)`` nas linhas de ajuste: ``idx`` dado, o sorteio de
    :func:`_amostra_optbinning` (``idx=None``) ou tudo (sem amostragem)."""
    if idx is None:
        idx = _amostra_optbinning(len(x), max_rows)
    if idx is not None:
        x = np.asarray(x)[idx]
        y = np.asarray(y)[idx]
    return x, y


def avisa_falha_optbinning(b, e) -> None:
    """Aviso de falha INESPERADA do optbinning (qualquer exceção que não seja o
    ``ValueError`` de problema inviável) — mesmo texto nos dois caminhos."""
    warnings.warn(
        f"optbinning falhou inesperadamente em '{getattr(b, 'name', '?')}': "
        f"{type(e).__name__}: {e}", RuntimeWarning)


def checa_status_optbinning(b) -> None:
    """Levanta ``RuntimeError`` se o solver RECUSOU o modelo (``MODEL_INVALID``).

    O optbinning não levanta nesse caso: devolve um bin só, que viraria um "sem
    corte válido" calado (foi assim que o conflito ``num_workers`` ×
    ``num_search_workers`` do CP-SAT passou despercebido). Não é o "inviável"
    das restrições (esse é ``INFEASIBLE`` e segue calado): é parâmetro do
    solver ou um alvo contínuo de magnitude enorme (~1e12), que estoura a escala
    inteira com que o optbinning monta o modelo — por isso vira o aviso de falha
    inesperada, com a dica, e não o ``[]`` silencioso."""
    try:
        status = b.status
    except Exception:
        return
    if status == "MODEL_INVALID":
        raise RuntimeError(
            "o solver recusou o modelo (status MODEL_INVALID) — com alvo contínuo "
            "de magnitude muito grande, reescale o alvo (ex.: em milhares/milhões)")


def optbinning_terminou(b) -> bool:
    """O ajuste ``b`` terminou de verdade: ajustado e sem o ``MODEL_INVALID``
    (ver :func:`checa_status_optbinning`), em que o optbinning se dá por
    ajustado com um bin só."""
    return bool(getattr(b, "_is_fitted", False)) and getattr(b, "_status", None) != "MODEL_INVALID"


def ajusta_optbinning_em_thread(b, x, y, max_rows=None, idx=None):
    """Núcleo de :func:`fit_optbinning_splits` para rodar numa THREAD de
    trabalho (lote paralelo do ranking, ``n_jobs``): mesma amostra, mesmo
    ``np.errstate`` e mesmo CP-SAT com 1 worker (:func:`_cpsat_um_worker` —
    ambos por thread: têm de entrar AQUI), mas SEM ``warnings.catch_warnings``
    — até o Python 3.13 os filtros são globais e o enter/exit concorrente
    restauraria listas trocadas por outra thread; quem cria o pool filtra uma
    vez só, na thread principal, e só sai do filtro com o pool encerrado. Com
    avisos por contexto (``sys.flags.context_aware_warnings``, 3.14+) o filtro
    da thread principal não chega aqui — aí o ``catch_warnings`` local é seguro
    e é usado. Devolve ``(cortes, exceção)`` (``[]`` na exceção): nada é avisado
    aqui — a thread principal decide depois, já fora do filtro (``ValueError``
    ⇒ ``[]`` calado, como em :func:`fit_optbinning_splits`; outra ⇒
    :func:`avisa_falha_optbinning`)."""
    x, y = _recorta_amostra_optbinning(x, y, max_rows, idx)
    try:
        if getattr(sys.flags, "context_aware_warnings", False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(divide="ignore", invalid="ignore"), _cpsat_um_worker():
                    b.fit(x, y)
        else:
            with np.errstate(divide="ignore", invalid="ignore"), _cpsat_um_worker():
                b.fit(x, y)
        checa_status_optbinning(b)
        return list(b.splits), None
    except Exception as e:
        return [], e


def sql_texto(v) -> str:
    """Literal de TEXTO SQL que vale igual no Spark/Databricks e no ANSI.

    No Spark SQL ``'O''Neil'`` NÃO é um apóstrofo escapado: são duas strings
    adjacentes, que o parser concatena ("ONeil") — a categoria nunca casava. E a
    barra invertida é caractere de escape no Spark. Os dois viram ``chr(39)`` /
    ``chr(92)`` concatenados com ``||``; texto sem eles fica como sempre foi."""
    s = str(v)
    if "'" not in s and "\\" not in s:
        return "'" + s + "'"
    partes, atual = [], ""
    for ch in s:
        if ch in "'\\":
            if atual:
                partes.append("'" + atual + "'")
                atual = ""
            partes.append(f"chr({ord(ch)})")
        else:
            atual += ch
    if atual:
        partes.append("'" + atual + "'")
    return "(" + " || ".join(partes) + ")"


def eh_booleana(serie) -> bool:
    """A coluna é BOOLEANA? ``bool``/``boolean`` ou ``object`` só com True/False (e
    nulos) — é assim que uma booleana com faltantes chega do ``toPandas()``. Usada
    pelos geradores de SQL/Spark, que comparam booleana com literal booleano (o
    ``CAST(bool AS STRING)`` do Spark dá "true" minúsculo e não casa com 'True')."""
    if pd.api.types.is_bool_dtype(serie):
        return True
    return serie.dtype == object and pd.api.types.infer_dtype(serie, skipna=True) == "boolean"



def copy_on_write_garantido() -> bool:
    """O Copy-on-Write do pandas está ATIVO agora? ``True`` só com pandas >= 3
    (CoW é o único modo) ou pandas 2.x com ``pd.options.mode.copy_on_write is
    True`` — identidade, como o próprio ``using_copy_on_write`` do pandas: o
    ``'warn'`` do 2.2 é string truthy mas NÃO é CoW, e o ArrayManager do 2.0/2.1
    também não o aplica. pandas 1.x: nunca (CoW experimental e incompleto).

    Só LÊ a opção — a lib nunca liga o CoW (é um efeito global na sessão do
    usuário: atribuição encadeada deixa de valer e ``.values`` vira só-leitura)."""
    try:
        maior = int(str(pd.__version__).split(".")[0])
    except (TypeError, ValueError):
        return False
    if maior >= 3:
        return True
    if maior < 2:
        return False
    try:
        if pd.options.mode.copy_on_write is not True:
            return False
        from pandas._config import using_copy_on_write
        return using_copy_on_write() is True
    except Exception:  # noqa: BLE001 - opção/API ausente: trata como sem CoW
        return False


def _buffer_como_copia(v) -> bool:
    """O buffer de um bloco já tem o layout que ``.copy()`` (profundo) produziria?
    Contíguo em C e gravável — ndarray do bloco, ``_ndarray`` dos NDArrayBacked
    (Categorical, datetime/period, string[python], NumpyExtensionArray) ou
    ``_data``/``_mask`` dos mascarados (Int64/Float64/boolean). O Arrow é imutável
    e o ``.copy()`` dele devolve o mesmo ChunkedArray. Qualquer outro tipo:
    ``False`` (a cópia rasa não é garantidamente idêntica)."""
    if isinstance(v, np.ndarray):
        return bool(v.flags.c_contiguous and v.flags.writeable)
    if hasattr(v, "_pa_array"):
        return True
    nd = getattr(v, "_ndarray", None)
    if isinstance(nd, np.ndarray):
        return bool(nd.flags.c_contiguous and nd.flags.writeable)
    dados, mascara = getattr(v, "_data", None), getattr(v, "_mask", None)
    if isinstance(dados, np.ndarray) and isinstance(mascara, np.ndarray):
        return all(bool(a.flags.c_contiguous and a.flags.writeable) for a in (dados, mascara))
    return False


def copia_df(df: pd.DataFrame) -> pd.DataFrame:
    """``df.copy()`` que, sob Copy-on-Write GARANTIDO (:func:`copy_on_write_garantido`),
    é RASA — sem duplicar a base (~8 GB a 10M linhas × 100 colunas).

    Com CoW o pandas copia o bloco na 1ª escrita de qualquer um dos lados, então
    o isolamento entre a cópia e o original continua total. Para o resultado
    ficar IDÊNTICO ao da cópia profunda (que consolida os blocos e devolve
    buffers contíguos em C e graváveis), a cópia rasa é consolidada do mesmo
    jeito — só os grupos fragmentados são fundidos (memória nova só para eles;
    sem isso, inserir derivadas numa base fragmentada emitiria o
    PerformanceWarning "highly fragmented") — e cada bloco compartilhado precisa
    já ter o layout da cópia (:func:`_buffer_como_copia`): sum/BLAS sobre
    coluna com passo ≠ 1 arredondam diferente. Falhou qualquer guarda (ou
    API interna ausente): cópia profunda, como sempre. Sem CoW: cópia profunda —
    desligar o CoW DEPOIS de copiar com ele ligado quebra o isolamento."""
    if not copy_on_write_garantido():
        return df.copy()
    try:
        out = df.copy(deep=False)
        out._consolidate_inplace()
        if all(_buffer_como_copia(b.values) for b in out._mgr.blocks):
            return out
    except Exception:  # noqa: BLE001 - API interna do pandas mudou: caminho antigo
        pass
    return df.copy()
