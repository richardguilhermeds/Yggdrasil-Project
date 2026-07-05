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
safras heterogêneas ou quebra estrutural.

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
    # Integral 1-D **determinística** (evita o ruído quasi-Monte-Carlo do
    # ``multivariate_normal.cdf`` em 2-D, que muda entre chamadas nos mesmos
    # dados): Φ₂(z, z; ρ) = ∫_{-∞}^{z} φ(x)·Φ((z − ρx)/√(1 − ρ²)) dx.
    rho = float(np.clip(rho, -0.999999, 0.999999))
    if abs(rho) < 1e-12:
        return float(norm.cdf(z)) ** 2
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
def _gauss_hermite_nodes(n_nodes: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Nós/pesos de Gauss–Hermite *probabilístico* para integrar contra ``φ(y)``.

    ``hermegauss`` (Hermite_e) usa peso ``exp(−y²/2)``; dividindo os pesos por
    ``√(2π)`` obtém-se ``∫ φ(y) g(y) dy ≈ Σ w_i g(y_i)`` com ``φ`` a densidade
    normal padrão. Assim não é preciso mudar variável.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_nodes)
    weights = weights / np.sqrt(2.0 * np.pi)
    return nodes, weights


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
) -> tuple[float, float]:
    """Estima ``(pd, ρ)`` por **máxima verossimilhança** (mistura de Vasicek).

    A cada período ``t`` observam-se ``k_t`` *defaults* em ``n_t`` obligores. A
    verossimilhança marginaliza o fator sistêmico ``Y ~ N(0,1)``:

        ``L = Π_t ∫ φ(y) · C(n_t,k_t) · p(y)^{k_t} (1−p(y))^{n_t−k_t} dy`` ,

    com ``p(y) = Φ((N⁻¹(pd) − √ρ·y)/√(1−ρ))``. A integral usa quadratura de
    Gauss–Hermite (``n_nodes`` nós) e a otimização roda em espaço transformado
    (``pd = Φ(a)``, ``ρ = sigmoide(b)``) via
    :func:`scipy.optimize.minimize` (Nelder–Mead, robusto e sem gradiente).

    Parameters
    ----------
    defaults:
        Número de *defaults* por período (inteiros ``k_t ≥ 0``).
    exposures:
        Número de obligores em risco por período (inteiros ``n_t ≥ k_t``).
    n_nodes:
        Número de nós de Gauss–Hermite (padrão 96; mais nós = mais precisão,
        mas ``hermegauss`` fica instável acima de ~120 nós).

    Returns
    -------
    (pd, rho): tuple[float, float]
        PD incondicional e correlação de ativos, ambos em ``(0, 1)`` e ``ρ``
        prendido em ``[0, 0.999]``.

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

    from scipy.special import gammaln
    # log C(n, k) constante em (pd, ρ) — pré-calculado uma vez.
    log_binom = gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
    nodes, weights = _gauss_hermite_nodes(n_nodes)

    # Chute inicial: PD = fração média de defaults; ρ moderado (0.1).
    pd0 = float(np.clip(np.sum(k) / np.sum(n), _EPS, 1.0 - _EPS))
    a0 = float(norm.ppf(pd0))
    b0 = float(np.log(0.1 / 0.9))   # sigmoide⁻¹(0.10)

    res = minimize(
        _neg_loglik_vasicek,
        x0=np.array([a0, b0]),
        args=(k, n, nodes, weights, log_binom),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000},
    )
    a_hat, b_hat = res.x
    pd_hat = float(norm.cdf(a_hat))
    rho_hat = float(1.0 / (1.0 + np.exp(-b_hat)))
    rho_hat = float(np.clip(rho_hat, _RHO_FLOOR, _RHO_CEIL))
    pd_hat = float(np.clip(pd_hat, _EPS, 1.0 - _EPS))
    return pd_hat, rho_hat


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
    "macro_factor_correlation",
    "factor_correlation_matrix",
    "nearest_correlation",
    "is_positive_definite",
    "IRB_RETAIL_RHO",
    "IRB_CORPORATE_RHO",
]
