"""
Estimação da correlação de ativos (Vasicek) e da matriz entre fatores
=====================================================================
A **correlação de ativos** ``ρ`` é, no capital econômico de crédito, o parâmetro
que **mais move o resultado** e, ao mesmo tempo, aquele para o qual **menos
dados** temos (Seção 4.1 e bloco D do guia). No modelo de Vasicek de fator
único, o valor latente dos ativos de cada devedor é

    A_i = √ρ · Y + √(1 − ρ) · ε_i ,   Y ~ N(0,1) (fator sistêmico),  ε_i ~ N(0,1)

e o *default* ocorre quando ``A_i < N⁻¹(PD)``. Como todos os devedores
compartilham o mesmo ``Y``, ``ρ`` governa quanto as taxas de *default* oscilam
em conjunto ao longo do ciclo econômico — e, portanto, o tamanho da cauda da
distribuição de perdas.

Este módulo reúne os três caminhos que a literatura de varejo usa para estimar
``ρ`` a partir de **séries históricas de taxa de default** (o dado que costuma
existir), mais a construção da **matriz de correlação entre fatores/produtos**:

1. **Método dos momentos** (:func:`asset_correlation_moments`) — a variância
   temporal da taxa de *default* implica, sob Vasicek, um valor de ``ρ``.
   Simples, robusto e sem otimização iterativa cara.
2. **Máxima verossimilhança** (:func:`asset_correlation_mle`,
   :func:`asset_params_mle`) — ajusta a mistura binomial de Vasicek sobre a
   contagem de *defaults* por período; usa a informação da amostra de forma
   completa (tamanho de cada safra), ao custo de uma quadratura + otimização.
3. **Modelos de fatores macro** (:func:`macro_factor_correlation`) — regride a
   PD (na escala probit) contra variáveis macroeconômicas; o ``R²`` mede a
   fração da variância sistêmica explicada por fatores observáveis.

A :func:`factor_correlation_matrix` recupera o fator sistêmico latente de cada
produto por período e correlaciona-os, produzindo a matriz de correlação entre
fatores exigida pela simulação multifatorial. :func:`nearest_correlation` e
:func:`is_positive_definite` garantem que essa matriz seja **positiva-definida**
(pré-requisito para a fatoração de Cholesky usada no Monte Carlo).

Recomendação prática do guia: **comparar** os ``ρ`` estimados com os valores
regulatórios do IRB de Basileia (varejo: 0,03–0,16; corporativo: 0,12–0,24)
como teste de sanidade — divergências grandes sinalizam série curta, mistura de
safras heterogêneas ou quebra estrutural. A **incerteza** de ``ρ`` é tratada
por :func:`asset_correlation_ci` (bootstrap em blocos, que preserva a
autocorrelação da série) e o :func:`rho_sanity_report` consolida, por
segmento, estimativas, IC e o confronto com as faixas IRB
(:data:`IRB_RETAIL_RHO`, :data:`IRB_CORPORATE_RHO`).

Contexto regulatório: Resolução CMN 4.557/2017 (ICAAP) e 4.966/2021; Basileia
II/III, abordagem IRB (correlação de ativos regulatória).
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq, minimize
from scipy.stats import norm

ArrayLike = Union[list, tuple, np.ndarray, pd.Series]

# Faixa numérica segura: evita ±inf ao aplicar N⁻¹ em taxas iguais a 0 ou 1.
_EPS = 1e-9
# Piso/teto para ρ. O teto < 1 evita divisões por √(1−ρ) → 0.
_RHO_FLOOR = 0.0
_RHO_CEIL = 0.999
# Faixa regulatória do IRB de varejo, útil como referência de sanidade.
IRB_RETAIL_RHO = (0.03, 0.16)
IRB_CORPORATE_RHO = (0.12, 0.24)


# ======================================================================
# Helpers internos
# ======================================================================
def _as_1d(x: ArrayLike, *, name: str) -> np.ndarray:
    """Converte para vetor 1-D de float, validando dimensão e finitude."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} está vazio.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contém valores não-finitos (NaN/inf).")
    return arr


def _clip_rate(rates: np.ndarray) -> np.ndarray:
    """Prende taxas em (0, 1) para que ``N⁻¹`` seja finito."""
    return np.clip(rates, _EPS, 1.0 - _EPS)


def _bivariate_normal_cdf(z: float, rho: float) -> float:
    """``Φ₂(z, z; ρ)`` — normal bivariada padrão com correlação ``ρ`` em ``(z, z)``.

    É a probabilidade **conjunta** de dois devedores idênticos entrarem em
    *default* simultaneamente sob Vasicek com correlação de ativos ``ρ``.
    """
    rho = float(np.clip(rho, -0.999999, 0.999999))
    if abs(rho) < 1e-12:
        return float(norm.cdf(z)) ** 2
    if rho > 0.0:
        # Representação de Vasicek: Φ₂(z, z; ρ) = E_Y[ p(Y)² ] com
        # p(y) = Φ((z − √ρ·y)/√(1−ρ)). Quadratura de Gauss–Hermite cacheada —
        # ordens de grandeza mais rápida que `quad`, o que viabiliza o
        # bootstrap de ρ (milhares de reestimações por chamada).
        nodes, weights = _gauss_hermite_nodes(96)
        p_y = norm.cdf((z - np.sqrt(rho) * nodes) / np.sqrt(1.0 - rho))
        val = float(np.sum(weights * p_y * p_y))
    else:
        # ρ < 0 não tem a representação de fator único; integral 1-D
        # **determinística** (evita o ruído quasi-Monte-Carlo do
        # ``multivariate_normal.cdf`` 2-D, que muda entre chamadas nos mesmos
        # dados): Φ₂(z, z; ρ) = ∫_{-∞}^{z} φ(x)·Φ((z − ρx)/√(1 − ρ²)) dx.
        s = np.sqrt(1.0 - rho * rho)
        val, _ = quad(lambda x: norm.pdf(x) * norm.cdf((z - rho * x) / s),
                      -np.inf, z, limit=200)
    return float(np.clip(val, 0.0, 1.0))


# ======================================================================
# (i) Método dos momentos
# ======================================================================
def asset_correlation_moments(default_rates: ArrayLike) -> float:
    """Correlação de ativos ``ρ`` pelo **método dos momentos** (Vasicek).

    Ideia. Seja ``DR_t`` a taxa de *default* observada no período ``t``. Sob o
    modelo de Vasicek, a média ``m`` estima a própria PD incondicional e a
    variância temporal da série carrega a assinatura de ``ρ``:

    * PD ≈ ``m = média(DR)`` ;
    * a probabilidade **conjunta** de *default* de dois devedores é
      ``Φ₂(N⁻¹(PD), N⁻¹(PD); ρ)`` (normal bivariada padrão);
    * assintoticamente ``Var[DR] = Φ₂(N⁻¹(PD), N⁻¹(PD); ρ) − PD²``.

    Resolve-se então, para ``ρ`` em ``[0, 1)``, a equação

        ``Φ₂(N⁻¹(m), N⁻¹(m); ρ) = m² + v`` ,   com ``v = Var(DR)`` amostral.

    Como ``Φ₂`` é monótona crescente em ``ρ`` (para ``z`` fixo), a solução é
    única e obtida por :func:`scipy.optimize.brentq`.

    Parameters
    ----------
    default_rates:
        Série histórica de taxas de *default* por período (proporções em
        ``[0, 1]``), tipicamente uma observação por ano ou safra.

    Returns
    -------
    float
        ``ρ`` estimado, prendido em ``[0, 0.999]``.

    Notes
    -----
    Séries curtas produzem estimativa instável — a variância amostral é ela
    própria ruidosa. Com menos de ~5 observações emite-se um aviso. Use a faixa
    regulatória do IRB (:data:`IRB_RETAIL_RHO`) como teste de sanidade.
    """
    dr = _as_1d(default_rates, name="default_rates")
    if np.any((dr < 0) | (dr > 1)):
        raise ValueError("default_rates deve conter proporções em [0, 1].")
    if dr.size < 5:
        import warnings
        warnings.warn(
            f"Série curta ({dr.size} observações): a estimativa de ρ pelo "
            "método dos momentos é instável. Interprete com cautela.",
            stacklevel=2,
        )

    m = float(np.mean(dr))
    # Variância amostral (ddof=1); com 1 ponto não há dispersão a explicar.
    v = float(np.var(dr, ddof=1)) if dr.size > 1 else 0.0

    # Casos-limite: sem PD ou sem dispersão ⇒ ρ = 0.
    if m <= _EPS or m >= 1.0 - _EPS or v <= 0.0:
        return _RHO_FLOOR

    z = float(norm.ppf(m))          # N⁻¹(PD)
    alvo = m * m + v                # E[DR²] = Var + média²

    # f(ρ) = Φ₂(z, z; ρ) − alvo, crescente em ρ. Buscamos a raiz.
    def f(rho: float) -> float:
        return _bivariate_normal_cdf(z, rho) - alvo

    lo, hi = _RHO_FLOOR, _RHO_CEIL
    f_lo, f_hi = f(lo), f(hi)
    # Se o alvo excede o que ρ→1 alcança, a variância observada é grande demais
    # para o modelo: satura no teto. Se já é negativo em ρ=0, satura no piso.
    if f_lo >= 0.0:
        return _RHO_FLOOR
    if f_hi <= 0.0:
        return _RHO_CEIL
    rho = brentq(f, lo, hi, xtol=1e-8, rtol=1e-8, maxiter=200)
    return float(np.clip(rho, _RHO_FLOOR, _RHO_CEIL))


# ======================================================================
# (ii) Máxima verossimilhança — mistura binomial de Vasicek
# ======================================================================
_GH_CACHE: dict = {}


def _gauss_hermite_nodes(n_nodes: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Nós/pesos de Gauss–Hermite *probabilístico* para integrar contra ``φ(y)``.

    ``hermegauss`` (Hermite_e) usa peso ``exp(−y²/2)``; dividindo os pesos por
    ``√(2π)`` obtém-se ``∫ φ(y) g(y) dy ≈ Σ w_i g(y_i)`` com ``φ`` a densidade
    normal padrão. Assim não é preciso mudar variável. O resultado é cacheado
    por ``n_nodes`` (os nós são reutilizados milhares de vezes no bootstrap).
    """
    if n_nodes not in _GH_CACHE:
        nodes, weights = np.polynomial.hermite_e.hermegauss(n_nodes)
        _GH_CACHE[n_nodes] = (nodes, weights / np.sqrt(2.0 * np.pi))
    return _GH_CACHE[n_nodes]


def _neg_loglik_vasicek(
    params: np.ndarray,
    defaults: np.ndarray,
    exposures: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
    log_binom: np.ndarray,
) -> float:
    """Log-verossimilhança negativa da mistura binomial de Vasicek.

    ``params = (a, b)`` são os parâmetros **transformados** (irrestritos):
    ``pd = Φ(a)`` e ``ρ = sigmoide(b)`` mantêm ``pd ∈ (0,1)`` e ``ρ ∈ (0,1)``
    sem barreiras, o que estabiliza o otimizador.

    Para cada período ``t`` com ``k_t`` *defaults* em ``n_t`` obligores:

        ``L_t = ∫ φ(y) · C(n_t,k_t) · p(y)^{k_t} · (1−p(y))^{n_t−k_t} dy`` ,

    com ``p(y) = Φ( (N⁻¹(pd) − √ρ · y) / √(1−ρ) )``. A integral em ``y`` é
    aproximada por Gauss–Hermite. Tudo em log para evitar *underflow*.
    """
    a, b = params
    pd = norm.cdf(a)
    rho = 1.0 / (1.0 + np.exp(-b))
    pd = min(max(pd, _EPS), 1.0 - _EPS)
    rho = min(max(rho, _EPS), 1.0 - _EPS)

    inv_pd = norm.ppf(pd)
    sqrt_rho = np.sqrt(rho)
    sqrt_1m = np.sqrt(1.0 - rho)

    # p(y) por nó de quadratura: forma (n_nodes,).
    p_y = norm.cdf((inv_pd - sqrt_rho * nodes) / sqrt_1m)
    p_y = np.clip(p_y, _EPS, 1.0 - _EPS)
    log_p = np.log(p_y)
    log_1mp = np.log1p(-p_y)

    k = defaults[:, None]           # (T, 1)
    n = exposures[:, None]          # (T, 1)
    # log da binomial condicional a cada nó: (T, n_nodes).
    log_cond = log_binom[:, None] + k * log_p[None, :] + (n - k) * log_1mp[None, :]

    # log ∫ = logsumexp sobre os nós, deslocando por log(peso).
    log_terms = log_cond + np.log(weights)[None, :]
    max_lt = np.max(log_terms, axis=1, keepdims=True)
    log_lik_t = (max_lt[:, 0]
                 + np.log(np.sum(np.exp(log_terms - max_lt), axis=1)))
    total = float(np.sum(log_lik_t))
    if not np.isfinite(total):
        return 1e12
    return -total


def asset_params_mle(
    defaults: ArrayLike,
    exposures: ArrayLike,
    *,
    n_nodes: int = 96,
    rho_starts: tuple = (0.02, 0.08, 0.15, 0.25),
    return_details: bool = False,
) -> Union[tuple[float, float], dict]:
    """Estima ``(pd, ρ)`` por **máxima verossimilhança** (mistura de Vasicek).

    A cada período ``t`` observam-se ``k_t`` *defaults* em ``n_t`` obligores. A
    verossimilhança marginaliza o fator sistêmico ``Y ~ N(0,1)``:

        ``L = Π_t ∫ φ(y) · C(n_t,k_t) · p(y)^{k_t} (1−p(y))^{n_t−k_t} dy`` ,

    com ``p(y) = Φ((N⁻¹(pd) − √ρ·y)/√(1−ρ))``. A integral usa quadratura de
    Gauss–Hermite (``n_nodes`` nós) e a otimização roda em espaço transformado
    (``pd = Φ(a)``, ``ρ = sigmoide(b)``) via
    :func:`scipy.optimize.minimize` (Nelder–Mead, robusto e sem gradiente).

    **Multi-start.** A superfície da log-verossimilhança tem curvatura fraca em
    ``ρ`` (platôs) e um único chute pode deixar o Nelder–Mead estacionar longe
    do ótimo. Varre-se uma grade de ``ρ`` inicial (``rho_starts``) e fica-se
    com a melhor log-verossimilhança entre os chutes. Se **nenhum** chute
    melhora a própria verossimilhança inicial (otimizador estacionado) ou todos
    divergem, aplica-se um *fallback* declarado: ``ρ`` pelo
    :func:`asset_correlation_moments` sobre a série ``k/n`` e ``pd`` pela
    fração agregada de *defaults* — com aviso e flag ``fallback_momentos`` no
    retorno detalhado.

    Parameters
    ----------
    defaults:
        Número de *defaults* por período (inteiros ``k_t ≥ 0``).
    exposures:
        Número de obligores em risco por período (inteiros ``n_t ≥ k_t``).
    n_nodes:
        Número de nós de Gauss–Hermite (padrão 96; mais nós = mais precisão,
        mas ``hermegauss`` fica instável acima de ~120 nós).
    rho_starts:
        Grade de valores iniciais de ``ρ`` (em ``(0, 1)``) para o multi-start.
    return_details:
        Se ``True``, retorna um ``dict`` com diagnóstico da otimização em vez
        da tupla ``(pd, rho)``.

    Returns
    -------
    (pd, rho): tuple[float, float]
        PD incondicional e correlação de ativos, ambos em ``(0, 1)`` e ``ρ``
        prendido em ``[0, 0.999]`` (padrão, ``return_details=False``).
    dict
        Com ``return_details=True``: ``{'pd', 'rho', 'loglik',
        'fallback_momentos', 'convergiu', 'rho_starts'}``, onde ``loglik`` é a
        log-verossimilhança no ponto retornado e ``fallback_momentos`` indica
        que a MLE falhou e o resultado veio do método dos momentos.

    Notes
    -----
    A quadratura de Gauss–Hermite resolve bem o integrando para coortes de
    tamanho **realista** (dezenas a poucos milhares de obligores por período),
    regime em que a MLE é praticamente não-enviesada. Para ``n_t`` muito grande
    (dezenas de milhares), a binomial condicional vira quase uma delta em
    ``p(y)`` e a grade fixa de nós subestima ``ρ``; nesse limite de carteira
    grande, prefira o :func:`asset_correlation_moments` sobre a série de taxas
    de *default* (que independe de ``n_t``). Emite-se aviso quando a razão
    sinal/ruído indica esse regime.
    """
    k = _as_1d(defaults, name="defaults")
    n = _as_1d(exposures, name="exposures")
    if k.shape != n.shape:
        raise ValueError(
            f"defaults {k.shape} e exposures {n.shape} têm tamanhos distintos.")
    if np.any(k < 0) or np.any(n <= 0):
        raise ValueError("Exija defaults ≥ 0 e exposures > 0 em cada período.")
    if np.any(k > n):
        raise ValueError("Há período com mais defaults do que obligores (k > n).")
    if k.size < 5:
        import warnings
        warnings.warn(
            f"Série curta ({k.size} períodos): a MLE de ρ é instável. "
            "Interprete com cautela.",
            stacklevel=2,
        )
    if float(np.median(n)) > 20000:
        import warnings
        warnings.warn(
            "Coortes muito grandes (mediana de obligores > 20 mil): no limite "
            "de carteira grande a binomial vira quase uma delta e a quadratura "
            "de Gauss–Hermite tende a subestimar ρ. Considere "
            "asset_correlation_moments sobre a série de taxas de default.",
            stacklevel=2,
        )

    rho_starts = tuple(float(r) for r in rho_starts)
    if len(rho_starts) == 0 or any(not (0.0 < r < 1.0) for r in rho_starts):
        raise ValueError("rho_starts deve conter ao menos um valor em (0, 1).")

    from scipy.special import gammaln
    # log C(n, k) constante em (pd, ρ) — pré-calculado uma vez.
    log_binom = gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
    nodes, weights = _gauss_hermite_nodes(n_nodes)

    # Chute de PD comum a todos os starts: fração agregada de defaults.
    pd0 = float(np.clip(np.sum(k) / np.sum(n), _EPS, 1.0 - _EPS))
    a0 = float(norm.ppf(pd0))
    args = (k, n, nodes, weights, log_binom)

    # Multi-start sobre a grade de ρ inicial: guarda, por chute, o resultado e
    # o valor da função objetivo NO chute (p/ detectar otimizador estacionado).
    ensaios: list[tuple] = []
    for rho0 in rho_starts:
        b0 = float(np.log(rho0 / (1.0 - rho0)))          # sigmoide⁻¹(rho0)
        x0 = np.array([a0, b0])
        f0 = float(_neg_loglik_vasicek(x0, *args))
        res = minimize(
            _neg_loglik_vasicek,
            x0=x0,
            args=args,
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000},
        )
        ensaios.append((res, f0))

    validos = [(res, f0) for res, f0 in ensaios
               if np.isfinite(res.fun) and res.fun < 1e11]
    # Falha declarada: divergiu (nenhum resultado finito) ou estacionou no
    # chute (nenhum start melhorou a própria verossimilhança inicial).
    divergiu = len(validos) == 0
    estacionou = (not divergiu
                  and not any(res.fun < f0 - 1e-9 for res, f0 in validos))

    if divergiu or estacionou:
        import warnings
        motivo = "divergiu" if divergiu else "estacionou no chute inicial"
        warnings.warn(
            f"MLE de (pd, ρ) {motivo}; aplicando fallback pelo método dos "
            "momentos sobre a série de taxas de default (k/n).",
            stacklevel=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # evita aviso duplicado
            rho_hat = float(asset_correlation_moments(k / n))
        pd_hat = pd0
        b_fb = float(np.log(max(rho_hat, _EPS) / max(1.0 - rho_hat, _EPS)))
        loglik = -float(_neg_loglik_vasicek(np.array([a0, b_fb]), *args))
        fallback, convergiu = True, False
    else:
        best, _ = min(validos, key=lambda rf: rf[0].fun)
        a_hat, b_hat = best.x
        pd_hat = float(np.clip(norm.cdf(a_hat), _EPS, 1.0 - _EPS))
        rho_hat = float(np.clip(1.0 / (1.0 + np.exp(-b_hat)),
                                _RHO_FLOOR, _RHO_CEIL))
        loglik = -float(best.fun)
        fallback, convergiu = False, bool(best.success)

    if not return_details:
        return pd_hat, rho_hat
    return {
        "pd": pd_hat,
        "rho": rho_hat,
        "loglik": loglik,
        "fallback_momentos": fallback,
        "convergiu": convergiu,
        "rho_starts": rho_starts,
    }


def asset_correlation_mle(
    defaults: ArrayLike,
    exposures: ArrayLike,
    *,
    n_nodes: int = 64,
) -> float:
    """Correlação de ativos ``ρ`` por MLE da mistura de Vasicek.

    Conveniência que retorna apenas ``ρ`` de :func:`asset_params_mle` (a PD
    também é estimada internamente; use ``asset_params_mle`` para obtê-la).
    """
    _, rho = asset_params_mle(defaults, exposures, n_nodes=n_nodes)
    return rho


# ======================================================================
# Incerteza de ρ — bootstrap em blocos e relatório de sanidade
# ======================================================================
def asset_correlation_ci(
    default_rates: ArrayLike,
    n_boot: int = 1000,
    block: int = 4,
    alpha: float = 0.05,
    method: str = "moments",
    seed: Optional[int] = None,
    *,
    exposures: Optional[ArrayLike] = None,
    n_nodes: int = 64,
) -> dict:
    """Intervalo de confiança de ``ρ`` por **bootstrap em blocos**.

    Séries de taxa de *default* são curtas e **autocorrelacionadas** (o ciclo
    econômico persiste por vários períodos); o bootstrap i.i.d. clássico
    embaralharia essa dependência e subestimaria a incerteza. Usa-se aqui o
    *moving block bootstrap*: sorteiam-se blocos contíguos de ``block``
    períodos (com reposição), concatenados até recompor o tamanho original da
    série, e reestima-se ``ρ`` em cada réplica. O IC é o intervalo
    **percentílico** ``[α/2, 1−α/2]`` da distribuição bootstrap.

    Parameters
    ----------
    default_rates:
        Série histórica de taxas de *default* por período (proporções em
        ``[0, 1]``).
    n_boot:
        Número de réplicas bootstrap.
    block:
        Comprimento do bloco (períodos contíguos preservados por sorteio).
        Blocos maiores preservam mais autocorrelação, mas reduzem a
        diversidade das réplicas; 4 é um padrão razoável para séries anuais.
    alpha:
        Nível de significância do IC bilateral (``0.05`` ⇒ IC de 95%).
    method:
        ``'moments'`` reestima via :func:`asset_correlation_moments` (rápido);
        ``'mle'`` via :func:`asset_params_mle` (exige ``exposures`` e é
        **caro**: uma otimização multi-start por réplica — prefira ``n_boot``
        menor).
    seed:
        Semente do gerador (reprodutibilidade).
    exposures:
        Obligores em risco por período, alinhados a ``default_rates``.
        Obrigatório com ``method='mle'`` (as contagens de *default* são
        reconstruídas como ``k_t = round(DR_t · n_t)``).
    n_nodes:
        Nós de Gauss–Hermite repassados à MLE (só com ``method='mle'``).

    Returns
    -------
    dict
        ``{'rho', 'ic_inferior', 'ic_superior', 'alpha', 'metodo', 'n_boot',
        'block', 'rhos_boot'}`` — estimativa pontual na série original, limites
        do IC e o vetor bootstrap completo (p/ histogramas ou ICs em outro
        nível).
    """
    dr = _as_1d(default_rates, name="default_rates")
    if np.any((dr < 0) | (dr > 1)):
        raise ValueError("default_rates deve conter proporções em [0, 1].")
    T = dr.size
    if T < 2:
        raise ValueError("São necessários ao menos 2 períodos para o bootstrap.")
    if n_boot < 1:
        raise ValueError("n_boot deve ser >= 1.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha deve estar em (0, 1).")
    if block < 1:
        raise ValueError("block deve ser >= 1.")
    block = int(min(block, T))
    if method not in ("moments", "mle"):
        raise ValueError("method deve ser 'moments' ou 'mle'.")

    n_arr: Optional[np.ndarray] = None
    if method == "mle":
        if exposures is None:
            raise ValueError(
                "method='mle' exige `exposures` (obligores por período) para "
                "reconstruir as contagens de default.")
        n_arr = _as_1d(exposures, name="exposures")
        if n_arr.shape != dr.shape:
            raise ValueError(
                f"exposures {n_arr.shape} e default_rates {dr.shape} têm "
                "tamanhos distintos.")
        if np.any(n_arr <= 0):
            raise ValueError("exposures deve ser > 0 em todos os períodos.")

    def _rho(idx: np.ndarray) -> float:
        """Reestima ρ no reamostrado ``idx`` pelo método escolhido."""
        if method == "moments":
            return float(asset_correlation_moments(dr[idx]))
        k_b = np.rint(dr[idx] * n_arr[idx])
        _, rho_b = asset_params_mle(k_b, n_arr[idx], n_nodes=n_nodes)
        return float(rho_b)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # avisos de série curta por réplica
        rho_ponto = _rho(np.arange(T))
        rng = np.random.default_rng(seed)
        n_blocks = -(-T // block)           # ceil(T / block)
        rhos = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            starts = rng.integers(0, T - block + 1, size=n_blocks)
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T]
            rhos[i] = _rho(idx)

    lo, hi = np.quantile(rhos, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "rho": rho_ponto,
        "ic_inferior": float(lo),
        "ic_superior": float(hi),
        "alpha": float(alpha),
        "metodo": method,
        "n_boot": int(n_boot),
        "block": int(block),
        "rhos_boot": rhos,
    }


# Faixas IRB por tipo de segmento (aceita os nomes em inglês e em pt-BR).
_BANDAS_IRB = {
    "retail": IRB_RETAIL_RHO,
    "varejo": IRB_RETAIL_RHO,
    "corporate": IRB_CORPORATE_RHO,
    "corporativo": IRB_CORPORATE_RHO,
}


def _banda_irb(tipo) -> tuple[str, tuple[float, float]]:
    """Resolve o tipo de segmento na faixa IRB correspondente."""
    chave = str(tipo).strip().lower()
    if chave not in _BANDAS_IRB:
        raise ValueError(
            f"segment_type {tipo!r} inválido; use 'retail'/'varejo' ou "
            "'corporate'/'corporativo'.")
    faixa = _BANDAS_IRB[chave]
    nome = "retail" if faixa is IRB_RETAIL_RHO else "corporate"
    return nome, faixa


def rho_sanity_report(
    default_rates_by_segment: Union[dict, pd.DataFrame],
    segment_type: Union[str, dict] = "retail",
    *,
    exposures_by_segment: Optional[dict] = None,
    n_boot: int = 500,
    block: int = 4,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Relatório de sanidade de ``ρ`` por segmento contra as faixas IRB.

    Consolida, para cada segmento, as estimativas de correlação de ativos
    (momentos e, quando houver exposições, MLE), o IC por bootstrap em blocos
    (:func:`asset_correlation_ci`, método dos momentos) e o confronto com a
    faixa de referência do IRB (:data:`IRB_RETAIL_RHO` ou
    :data:`IRB_CORPORATE_RHO`). É a materialização do teste de sanidade
    recomendado no guia: ``ρ`` muito fora da faixa sinaliza série curta,
    mistura de safras heterogêneas ou quebra estrutural.

    Parameters
    ----------
    default_rates_by_segment:
        ``dict`` nome → série de taxas de *default* (proporções em ``[0, 1]``)
        ou ``DataFrame`` com uma coluna por segmento (linhas = períodos).
    segment_type:
        ``'retail'``/``'varejo'`` ou ``'corporate'``/``'corporativo'`` — a
        faixa IRB usada no confronto. Aceita também ``dict`` nome → tipo para
        misturar segmentos de faixas distintas (ausentes caem em ``'retail'``).
    exposures_by_segment:
        ``dict`` opcional nome → obligores por período; segmentos presentes
        ganham a coluna ``rho_mle`` (contagens reconstruídas como
        ``round(DR_t · n_t)``); ausentes ficam com ``NaN``.
    n_boot, block, alpha, seed:
        Parâmetros repassados a :func:`asset_correlation_ci`.

    Returns
    -------
    pd.DataFrame
        Uma linha por segmento com colunas ``segmento``, ``tipo``,
        ``rho_momentos``, ``rho_mle``, ``ic_inferior``, ``ic_superior``,
        ``faixa_irb_min``, ``faixa_irb_max`` e ``fora_da_faixa`` (``True`` se
        **alguma** estimativa pontual disponível cair fora da faixa IRB).
    """
    if isinstance(default_rates_by_segment, pd.DataFrame):
        if default_rates_by_segment.shape[1] == 0:
            raise ValueError("default_rates_by_segment não tem colunas.")
        series = {str(c): default_rates_by_segment[c]
                  for c in default_rates_by_segment.columns}
    elif isinstance(default_rates_by_segment, dict):
        if len(default_rates_by_segment) == 0:
            raise ValueError("default_rates_by_segment está vazio.")
        series = {str(k): v for k, v in default_rates_by_segment.items()}
    else:
        raise ValueError(
            "default_rates_by_segment deve ser dict ou pandas.DataFrame.")

    import warnings
    linhas = []
    for nome, serie in series.items():
        dr = _as_1d(serie, name=f"segmento {nome!r}")
        if np.any((dr < 0) | (dr > 1)):
            raise ValueError(f"Segmento {nome!r} tem taxas fora de [0, 1].")
        tipo = (segment_type.get(nome, "retail")
                if isinstance(segment_type, dict) else segment_type)
        tipo_nome, (faixa_lo, faixa_hi) = _banda_irb(tipo)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # avisos de série curta por segmento
            rho_mom = float(asset_correlation_moments(dr))
            ci = asset_correlation_ci(dr, n_boot=n_boot, block=block,
                                      alpha=alpha, method="moments", seed=seed)
            rho_mle = float("nan")
            if exposures_by_segment is not None and nome in exposures_by_segment:
                n_seg = _as_1d(exposures_by_segment[nome],
                               name=f"exposures do segmento {nome!r}")
                if n_seg.shape != dr.shape:
                    raise ValueError(
                        f"exposures do segmento {nome!r} não alinham com a "
                        "série de taxas.")
                _, rho_mle = asset_params_mle(np.rint(dr * n_seg), n_seg)
                rho_mle = float(rho_mle)

        estimativas = [rho_mom] + ([rho_mle] if np.isfinite(rho_mle) else [])
        fora = any(not (faixa_lo <= e <= faixa_hi) for e in estimativas)
        linhas.append({
            "segmento": nome,
            "tipo": tipo_nome,
            "rho_momentos": rho_mom,
            "rho_mle": rho_mle,
            "ic_inferior": ci["ic_inferior"],
            "ic_superior": ci["ic_superior"],
            "faixa_irb_min": float(faixa_lo),
            "faixa_irb_max": float(faixa_hi),
            "fora_da_faixa": bool(fora),
        })
    return pd.DataFrame(linhas)


# ======================================================================
# (iii) Modelo de fatores macro
# ======================================================================
def macro_factor_correlation(
    default_rate_series: ArrayLike,
    macro: pd.DataFrame,
) -> dict:
    """Regressão OLS da PD (escala **probit**) contra variáveis macro.

    Transforma a taxa de *default* pela inversa da normal — ``x_t = N⁻¹(DR_t)``
    — que é a escala natural do modelo de Vasicek (o *default* é acionado por um
    limiar normal) e ajusta ``x_t = β₀ + Σ_j β_j · macro_{t,j} + u_t`` por
    mínimos quadrados ordinários.

    Interpretação do ``R²``. Sob Vasicek, ``x_t = N⁻¹(PD) − √(ρ/(1−ρ))·(−Y_t)``,
    ou seja, a variação de ``x_t`` **é** a variação sistêmica. A fração dessa
    variância explicada por fatores macro observáveis (o ``R²``) é uma leitura
    empírica de **quanto do risco sistêmico é macro-atribuível** — reportada
    aqui como ``rho_implicado``.

    Ressalvas (importantes). Este ``R²`` **não é** a correlação de ativos ``ρ``
    do Vasicek: é a fração da variância *sistêmica* explicada, não a fração da
    variância *total* dos ativos. Trate-o como diagnóstico de atribuição a
    fatores macro e como *cross-check* qualitativo dos coeficientes (sinais
    econômicos: desemprego ↑ ⇒ PD ↑), não como substituto de
    :func:`asset_correlation_moments`/:func:`asset_correlation_mle`.

    Parameters
    ----------
    default_rate_series:
        Série de taxas de *default* por período (proporções em ``[0, 1]``).
    macro:
        ``DataFrame`` de variáveis macro alinhado por período (uma linha por
        período, uma coluna por fator: desemprego, renda, juros, ...).

    Returns
    -------
    dict
        ``{'r2', 'rho_implicado', 'coef', 'intercept', 'n_obs'}``, onde
        ``rho_implicado == r2`` (fração da variância sistêmica explicada) e
        ``coef`` mapeia nome da variável ao coeficiente OLS.
    """
    dr = _as_1d(default_rate_series, name="default_rate_series")
    if np.any((dr < 0) | (dr > 1)):
        raise ValueError("default_rate_series deve conter proporções em [0, 1].")
    if not isinstance(macro, pd.DataFrame):
        raise ValueError("macro deve ser um pandas.DataFrame.")
    if macro.shape[0] != dr.size:
        raise ValueError(
            f"macro tem {macro.shape[0]} linhas mas a série tem {dr.size} "
            "períodos; alinhe-os por período.")
    if macro.shape[1] == 0:
        raise ValueError("macro não tem colunas (nenhuma variável explicativa).")

    y = norm.ppf(_clip_rate(dr))            # PD na escala probit
    X_vars = np.asarray(macro.to_numpy(), dtype=float)
    if not np.all(np.isfinite(X_vars)):
        raise ValueError("macro contém valores não-finitos (NaN/inf).")

    # Matriz de desenho com intercepto.
    X = np.column_stack([np.ones(dr.size), X_vars])
    # OLS por mínimos quadrados (robusto a colinearidade via lstsq/SVD).
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    r2 = float(np.clip(r2, 0.0, 1.0))

    coef = {str(name): float(b) for name, b in zip(macro.columns, beta[1:])}
    return {
        "r2": r2,
        "rho_implicado": r2,
        "coef": coef,
        "intercept": float(beta[0]),
        "n_obs": int(dr.size),
    }


# ======================================================================
# Matriz de correlação entre fatores/produtos
# ======================================================================
def _latent_factor_series(dr: np.ndarray, pd_p: float, rho_p: float) -> np.ndarray:
    """Recupera o fator sistêmico latente ``Y_t`` de um produto por período.

    Invertendo ``DR_t = Φ((N⁻¹(pd) − √ρ·Y_t)/√(1−ρ))`` do modelo de Vasicek:

        ``Y_t = (N⁻¹(pd) − √(1−ρ)·N⁻¹(DR_t)) / √ρ`` .

    Com ``ρ → 0`` o fator não é identificável; usa-se ``N⁻¹(DR_t)`` como
    *proxy* (a série ainda carrega o sinal sistêmico, apenas em outra escala,
    o que não altera a correlação de Pearson subsequente).
    """
    inv_dr = norm.ppf(_clip_rate(dr))
    if rho_p <= _EPS:
        return -inv_dr
    inv_pd = norm.ppf(min(max(pd_p, _EPS), 1.0 - _EPS))
    return (inv_pd - np.sqrt(1.0 - rho_p) * inv_dr) / np.sqrt(rho_p)


def factor_correlation_matrix(
    default_rate_frame: pd.DataFrame,
    rho: Optional[Union[float, dict]] = None,
) -> np.ndarray:
    """Matriz de correlação entre os **fatores sistêmicos** de cada produto.

    Uma coluna por produto/fator (``default_rate_frame``), uma linha por
    período. Para cada coluna recupera-se a série do fator sistêmico latente
    ``Y_t`` (via :func:`_latent_factor_series`) e retorna-se a **correlação de
    Pearson** entre essas séries. A matriz alimenta a simulação multifatorial —
    é ela que introduz diversificação entre produtos que o ASRF ignora.

    Parameters
    ----------
    default_rate_frame:
        ``DataFrame`` com taxas de *default* por período (linhas) e produto
        (colunas).
    rho:
        Correlação de ativos por produto. Aceita:

        * ``None`` — estima ``ρ_p`` por :func:`asset_correlation_moments` em
          cada coluna;
        * ``float`` — mesmo ``ρ`` para todos os produtos;
        * ``dict`` — mapeia nome de coluna a ``ρ_p`` (colunas ausentes caem no
          método dos momentos).

    Returns
    -------
    np.ndarray
        Matriz ``(n_produtos, n_produtos)`` simétrica, diagonal 1, projetada em
        positiva-definida por :func:`nearest_correlation`.
    """
    if not isinstance(default_rate_frame, pd.DataFrame):
        raise ValueError("default_rate_frame deve ser um pandas.DataFrame.")
    if default_rate_frame.shape[1] == 0:
        raise ValueError("default_rate_frame não tem colunas (produtos).")
    if default_rate_frame.shape[0] < 2:
        raise ValueError(
            "São necessários ao menos 2 períodos para estimar correlação.")

    cols = list(default_rate_frame.columns)
    latents = []
    for col in cols:
        dr = _as_1d(default_rate_frame[col], name=f"coluna {col!r}")
        if np.any((dr < 0) | (dr > 1)):
            raise ValueError(f"Coluna {col!r} tem valores fora de [0, 1].")
        # Resolve ρ_p conforme a forma de `rho`.
        if rho is None:
            rho_p = asset_correlation_moments(dr)
        elif isinstance(rho, dict):
            rho_p = float(rho.get(col, asset_correlation_moments(dr)))
        else:
            rho_p = float(rho)
        rho_p = float(np.clip(rho_p, _RHO_FLOOR, _RHO_CEIL))
        pd_p = float(np.mean(dr))
        latents.append(_latent_factor_series(dr, pd_p, rho_p))

    Y = np.column_stack(latents)            # (T, n_produtos)
    # Correlação de Pearson; colunas constantes ⇒ correlação indefinida → 0.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(Y, rowvar=False)
    corr = np.atleast_2d(corr)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    np.fill_diagonal(corr, 1.0)
    corr = 0.5 * (corr + corr.T)            # força simetria exata
    # Projeta em positiva-definida (pré-requisito da fatoração de Cholesky).
    return nearest_correlation(corr)


# ======================================================================
# Positividade / projeção para correlação PD (Higham)
# ======================================================================
def is_positive_definite(matrix: ArrayLike) -> bool:
    """``True`` se ``matrix`` é positiva-definida (todos autovalores > 0).

    Testa via fatoração de Cholesky, que só existe para matrizes simétricas
    positivas-definidas — o teste mais barato e numericamente estável. Uma
    matriz não-simétrica retorna ``False``.
    """
    M = np.asarray(matrix, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("matrix deve ser quadrada (2-D).")
    if not np.allclose(M, M.T, atol=1e-10):
        return False
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_correlation(matrix: ArrayLike, max_iter: int = 100) -> np.ndarray:
    """Correlação **positiva-definida** mais próxima (aproximação de Higham).

    Uma matriz de correlação empírica (Pearson entre séries com faltantes,
    janelas distintas ou ``ρ`` impostos manualmente) pode não ser
    positiva-definida, o que **impede** a fatoração de Cholesky usada na
    simulação. Este é um passo de saneamento.

    Método (projeções alternadas de Higham, 2002, simplificado):

    1. **Eigen-clip** — zera os autovalores negativos (projeta no cone das
       matrizes positivas-semidefinidas);
    2. **Renormaliza a diagonal** para 1 (projeta no conjunto das matrizes com
       diagonal unitária), reescalando ``C_ij / √(C_ii C_jj)``;

    itera até convergir ou atingir ``max_iter``. É uma **aproximação** da
    projeção de Higham (que alterna projeções de Frobenius), suficiente na
    prática para produzir uma matriz PD próxima da original. Ao final aplica um
    pequeno *jitter* na diagonal caso ainda reste autovalor marginalmente ≤ 0.

    Parameters
    ----------
    matrix:
        Matriz simétrica aproximadamente de correlação (diagonal ~1).
    max_iter:
        Máximo de iterações das projeções alternadas.

    Returns
    -------
    np.ndarray
        Matriz de correlação simétrica, diagonal 1 e positiva-definida.
    """
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("matrix deve ser quadrada (2-D).")
    n = A.shape[0]
    X = 0.5 * (A + A.T)                      # simetriza a entrada
    np.fill_diagonal(X, 1.0)

    for _ in range(max_iter):
        # (1) projeta no cone PSD: clip dos autovalores em ≥ 0.
        eigvals, eigvecs = np.linalg.eigh(X)
        eigvals_clipped = np.clip(eigvals, 0.0, None)
        X_psd = (eigvecs * eigvals_clipped) @ eigvecs.T
        X_psd = 0.5 * (X_psd + X_psd.T)

        # (2) renormaliza a diagonal para 1 (unit-diagonal).
        d = np.sqrt(np.clip(np.diag(X_psd), _EPS, None))
        X_new = X_psd / np.outer(d, d)
        np.fill_diagonal(X_new, 1.0)

        if np.max(np.abs(X_new - X)) < 1e-10:
            X = X_new
            break
        X = X_new

    # Garantia final de PD: se ainda houver autovalor ≤ 0, aplica jitter.
    if not is_positive_definite(X):
        eigvals = np.linalg.eigvalsh(X)
        min_eig = float(eigvals.min())
        if min_eig <= 0.0:
            jitter = (-min_eig + 1e-8)
            X = X + jitter * np.eye(n)
            d = np.sqrt(np.diag(X))
            X = X / np.outer(d, d)
            np.fill_diagonal(X, 1.0)
    return X


__all__ = [
    "asset_correlation_moments",
    "asset_correlation_mle",
    "asset_params_mle",
    "asset_correlation_ci",
    "rho_sanity_report",
    "macro_factor_correlation",
    "factor_correlation_matrix",
    "nearest_correlation",
    "is_positive_definite",
    "IRB_RETAIL_RHO",
    "IRB_CORPORATE_RHO",
]
