"""
Motor v3 — CreditRisk+ (benchmark atuarial, distribuição analítica)
====================================================================
Enquanto o ASRF (:mod:`.asrf`) parte do modelo estrutural de Merton e a
simulação de Monte Carlo (:mod:`.monte_carlo`) reamostra cenários, o
**CreditRisk+** (Credit Suisse First Boston, 1997) toma um caminho diferente,
de **matemática atuarial** (Seção 3.4 do guia): trata o número de *defaults* de
cada segmento como uma contagem aleatória e obtém a distribuição de perdas
agregadas de forma **analítica**, por recursão, sem simular um único cenário.

A mecânica em três ideias:

* **Contagem de Poisson.** Num segmento com ``n`` devedores e PD ``p``, o número
  esperado de *defaults* é ``μ = n·p``. Para PDs pequenas, o número de eventos é
  bem aproximado por uma **Poisson(μ)** — o mesmo salto que a atuária faz para
  contar sinistros raros e independentes.
* **Intensidade estocástica (mistura gama).** Se ``μ`` fosse fixo, as perdas
  seriam quase independentes e a cauda, fina demais. O CreditRisk+ deixa a
  intensidade ``μ`` **flutuar** com um fator de risco sistêmico ``Γ`` de média 1
  e volatilidade ``σ`` (coeficiente de variação). Marginalizar a Poisson sobre
  um fator **gama** produz uma **binomial negativa** para a contagem — cauda
  mais gorda e *defaults* positivamente correlacionados, exatamente o que uma
  crise faz.
* **Bandas de perda + recursão de Panjer.** As severidades ``LGD·EAD`` são
  discretizadas em múltiplos inteiros de uma **unidade de perda** ``L₀`` (as
  "bandas"). A função geradora de probabilidade da perda agregada tem forma
  fechada, e a **recursão de Panjer/Giese** extrai a distribuição inteira
  ``P(L = j·L₀)`` em tempo ``O(N²)``, sem simulação.

Insumos mínimos: taxas de *default* médias e a sua volatilidade sistêmica ``σ``
por carteira. É **rápido** e **barato**. Limitações (por construção): LGD e EAD
entram como **constantes por banda** (severidade determinística), e este módulo
implementa o caso **single-sector** (um único fator gama comum a toda a
carteira). Por isso o CreditRisk+ é usado aqui como **benchmark** para desafiar
o modelo principal — se ASRF, Monte Carlo e CreditRisk+ concordam na ordem de
grandeza do capital, ganha-se confiança; se divergem, há o que investigar.

Contexto regulatório: Resolução CMN 4.557/2017 (ICAAP) — a validação exige
mais de uma metodologia independente para o capital econômico; o CreditRisk+ é
o terceiro ângulo, atuarial, complementar ao estrutural (Basileia/IRB) e ao
simulado.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .measures import LossDistribution

if TYPE_CHECKING:  # evita import em runtime (só para type-checkers)
    from .portfolio import Portfolio


# ======================================================================
# Escolha da unidade de perda e discretização das severidades em bandas
# ======================================================================
def _severities(portfolio: "Portfolio") -> tuple[np.ndarray, np.ndarray]:
    """Severidade esperada e número de *defaults* esperado por segmento.

    Para cada segmento ``s`` com ``n`` devedores:

    * exposição por devedor ``v = EAD / n``;
    * **severidade esperada por default** ``e = LGD · v`` (a perda de um único
      *default*, tratada como constante — hipótese central do CreditRisk+);
    * **número esperado de defaults** ``μ = n · PD``.

    Quando ``n = 1`` o segmento é um "nome" único: ``e = LGD · EAD`` e
    ``μ = PD``, exatamente como esperado.

    Returns
    -------
    (sev, mu):
        Vetores paralelos aos segmentos com a severidade esperada e o ``μ``.
    """
    eads = portfolio.eads()
    lgds = portfolio.lgds()
    pds = portfolio.pds()
    n = portfolio.n_obligors().astype(float)

    v = eads / n                      # exposição por devedor
    sev = lgds * v                    # severidade esperada de um default
    mu = n * pds                      # número esperado de defaults do segmento
    return sev, mu


# Teto do nº de bandas do SUPORTE da distribuição de perdas. O custo da recursão
# de Panjer é O(N_bandas_distintas · N_max); manter N_max limitado aqui deixa o
# cálculo barato mesmo em carteiras granulares (dezenas de milhares de devedores,
# em que a perda agregada é milhares de vezes a severidade individual).
_MAX_GRID_BANDS = 8000
# Nº de desvios-padrão acima da média que o suporte precisa cobrir (a cauda de
# uma perda com mistura gama é pesada; 15σ dá folga para σ alto).
_TAIL_SIGMAS = 15.0


def _choose_loss_unit(sev: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    """Escolhe a unidade de perda ``L₀`` (a granularidade da grade).

    ``L₀`` controla o compromisso entre **precisão** (unidade menor → grade mais
    fina) e **custo** (unidade menor → mais bandas → recursão ``O(N·N_max)`` mais
    cara). O ponto delicado é que o **tamanho da grade** (``N_max``) não é ditado
    pela severidade individual e sim pela **perda agregada** da carteira: com
    varejo granular há milhares de *defaults* esperados, e a perda total fica em
    ordens de grandeza acima da severidade por devedor. Ancorar ``L₀`` apenas na
    severidade faria ``N_max = perda_total / L₀`` explodir (milhões de bandas).

    Por isso ``L₀`` é ancorado na **escala da perda agregada**: estima-se a média
    e o desvio-padrão da perda total (variância idiossincrática de Poisson
    ``Σ μ·sev²`` mais a componente sistêmica gama ``(σ·média)²``) e exige-se que o
    suporte relevante ``média + _TAIL_SIGMAS·desvio`` caiba em cerca de
    :data:`_MAX_GRID_BANDS` bandas:

        ``L₀ = max( (média + 15·desvio) / _MAX_GRID_BANDS ,  menor_sev / 20 )``

    O piso pela **menor severidade** (``/20``) preserva resolução em carteiras
    pequenas (poucos nomes), regime em que o primeiro termo é minúsculo e a grade
    já é pequena. A escolha de ``L₀`` afeta só a **cauda**: a EL é preservada
    **exatamente** pela reescala de intensidade em :func:`_discretize`
    (``μ_eff = μ·sev/(ν·L₀)``), independentemente de ``L₀``.
    """
    pos = sev[sev > 0]
    if pos.size == 0:
        raise ValueError(
            "Todas as severidades (LGD·EAD/n) são nulas: não há perda a modelar."
        )
    menor = float(np.min(pos))
    maior = float(np.max(pos))
    # Escala da perda AGREGADA (não da severidade individual).
    loss_mean = float(np.sum(mu * sev))
    idio_var = float(np.sum(mu * sev * sev))                 # Poisson: Var = Σ μ·sev²
    std = float(np.sqrt(idio_var + (float(sigma) * loss_mean) ** 2))
    loss_upper = loss_mean + _TAIL_SIGMAS * std
    l0_grid = loss_upper / _MAX_GRID_BANDS if loss_upper > 0 else 0.0
    # L₀ = maior entre o limite de custo (bounded grid) e a resolução da menor
    # severidade. Em carteira granular o primeiro domina e evita a grade explodir.
    l0 = max(l0_grid, menor / 20.0)
    if l0 <= 0:  # proteção numérica (severidades minúsculas)
        l0 = maior if maior > 0 else 1.0
    return float(l0)


def _discretize(
    sev: np.ndarray, mu: np.ndarray, loss_unit: float
) -> tuple[np.ndarray, np.ndarray]:
    """Discretiza severidades em bandas **preservando a perda esperada**.

    Duas operações acopladas (a técnica clássica de Gordy/CSFB para o CreditRisk+):

    1. **Banda inteira** ``ν = round(e/L₀)``, com ``ν ≥ 1`` para toda severidade
       positiva — uma severidade real nunca deve virar "banda zero" (perda nula),
       o que a excluiria do risco. Severidade nula → ``ν = 0`` (não contribui).
    2. **Reescala da intensidade** para corrigir o erro de arredondamento. A
       perda esperada de um segmento é ``μ · e``; ao discretizar ``e → ν·L₀``,
       ela viraria ``μ · ν·L₀`` e ficaria distorcida (grave quando severidades
       pequenas mas de ``μ`` alto são arredondadas para cima). Ajusta-se a
       intensidade efetiva para

           ``μ_eff = μ · e / (ν · L₀)``,

       de modo que ``μ_eff · ν·L₀ = μ · e`` **exatamente**. Assim a EL da
       distribuição discretizada bate com ``Σ PD·LGD·EAD`` a menos do ruído da
       recursão, **independentemente** da granularidade ``L₀`` escolhida. A
       variância muda ligeiramente (o "grão" da grade), mas isso é o erro de
       discretização de segunda ordem, pequeno e controlável por ``L₀``.

    Returns
    -------
    (nu, mu_eff):
        Bandas inteiras e intensidades reescaladas, paralelas aos segmentos.
    """
    sev = np.asarray(sev, dtype=float)
    mu = np.asarray(mu, dtype=float)
    nu = np.rint(sev / loss_unit).astype(int)
    nu[(nu < 1) & (sev > 0)] = 1
    nu[sev <= 0] = 0

    mu_eff = mu.copy().astype(float)
    ativos = nu >= 1
    # μ_eff = μ · e / (ν·L₀) — preserva a EL do segmento após o arredondamento.
    mu_eff[ativos] = mu[ativos] * sev[ativos] / (nu[ativos] * loss_unit)
    mu_eff[~ativos] = 0.0
    return nu, mu_eff


# ======================================================================
# Recursão de Panjer / Giese (single-sector, fator gama)
# ======================================================================
def panjer_recursion(
    band: np.ndarray,
    mu: np.ndarray,
    sigma: float = 0.5,
    max_loss_units: Optional[int] = None,
    *,
    tol: float = 1e-9,
) -> np.ndarray:
    """Distribuição de perdas do CreditRisk+ single-sector via recursão.

    Implementa a recursão de **Panjer generalizada (Giese, 2003)** para o
    CreditRisk+ com **um único fator sistêmico gama** de média 1 e variância
    ``σ²``. A perda agregada é medida em **bandas** (múltiplos inteiros da
    unidade de perda ``L₀``); esta função devolve o vetor de probabilidades
    ``P(L = j)``, ``j = 0, 1, …, N_max`` (em bandas).

    Ideia da dedução (sem simulação):

    * Cada segmento ``s`` contribui com ``μ_s`` *defaults* esperados, todos na
      banda ``ν_s`` (severidade discretizada). Agrega-se a **massa de defaults
      esperados por banda** ``A(j) = Σ_{s: ν_s = j} μ_s`` e o total
      ``μ_total = Σ_s μ_s``.
    * A perda agregada tem a função geradora de probabilidade (PGF)

          ``G(z) = ( 1 − σ² · (P(z) − 1) )^(−1/σ²)``,

      onde ``P(z) = (1/μ_total) · Σ_j A(j) · z^j`` é a PGF da severidade de um
      *default*. O expoente ``−1/σ²`` é a assinatura da mistura **gama**: ela
      transforma a Poisson em **binomial negativa**, engordando a cauda.
    * Diferenciar ``ln G`` e igualar coeficientes leva à recursão de convolução
      abaixo, que preenche ``P(j)`` a partir de ``P(0)`` em ``O(N²)``.

    O caso **Poisson puro** (sem risco sistêmico) é o limite ``σ → 0``; para
    ``σ`` muito pequeno cai-se automaticamente na recursão de Panjer clássica
    ``P(j) = (1/j) Σ_i i·A(i)·P(j−i)`` (numericamente estável).

    Parameters
    ----------
    band:
        Vetor (por segmento) com a banda inteira ``ν_s`` da severidade.
    mu:
        Vetor (por segmento) com o número esperado de *defaults* ``μ_s``.
    sigma:
        Coeficiente de variação do fator sistêmico gama (``σ = desvio/média``,
        com média 1). ``σ = 0`` recupera a Poisson pura (perdas quase
        independentes); ``σ`` maior engorda a cauda.
    max_loss_units:
        Teto da grade em bandas. Se ``None``, cresce automaticamente até a massa
        acumulada passar de ``1 − tol`` (ou um teto de segurança).
    tol:
        Massa residual tolerada na cauda para o corte automático da grade.

    Returns
    -------
    np.ndarray
        Vetor ``P`` de probabilidades por banda (``P[j] = P(L = j)``), já
        normalizado para somar ~1.
    """
    band = np.asarray(band, dtype=int)
    mu = np.asarray(mu, dtype=float)
    if band.shape != mu.shape:
        raise ValueError("band e mu devem ter o mesmo tamanho (um por segmento).")
    if np.any(mu < 0):
        raise ValueError("mu (número esperado de defaults) não pode ser negativo.")
    if sigma < 0:
        raise ValueError(f"sigma deve ser >= 0; recebido {sigma!r}.")

    # Considera apenas segmentos que efetivamente geram perda (banda >= 1).
    ativos = band >= 1
    band = band[ativos]
    mu = mu[ativos]
    if band.size == 0 or float(mu.sum()) <= 0.0:
        # Sem perda possível: toda a massa em L = 0.
        return np.array([1.0])

    N = int(band.max())                       # maior banda de severidade
    mu_total = float(mu.sum())                # defaults esperados totais

    # A(j) = massa de defaults esperados cuja severidade cai na banda j.
    A = np.zeros(N + 1)
    np.add.at(A, band, mu)                     # soma mu_s nos índices ν_s

    # -----------------------------------------------------------------
    # Tamanho da grade. A EL em bandas é Σ_j j·A(j); a UL sobe com sigma.
    # Um teto generoso (múltiplo da média + folga pela cauda) garante que a
    # massa acumulada ultrapasse 1 − tol antes do corte.
    # -----------------------------------------------------------------
    el_bands = float(np.sum(np.arange(N + 1) * A))     # perda esperada em bandas
    if max_loss_units is None:
        # Desvio-padrão aproximado da perda (em bandas) no modelo CreditRisk+:
        #   Var(L) ≈ Σ_j j²·A(j) · (1 + σ²·μ_total)   [aprox. de dimensionamento]
        segundo_momento = float(np.sum(np.arange(N + 1) ** 2 * A))
        var_aprox = segundo_momento * (1.0 + sigma * sigma * mu_total)
        sd = np.sqrt(max(var_aprox, 0.0))
        # Média + ~14 desvios cobre com folga o quantil 99,99% de caudas gordas.
        alvo = el_bands + 14.0 * sd + 10.0 * N
        Nmax = int(np.ceil(alvo)) + 1
        Nmax = max(Nmax, 4 * N + 8)
        # Teto absoluto de segurança para não explodir em memória.
        Nmax = min(Nmax, 20_000_000)
    else:
        if max_loss_units < 1:
            raise ValueError("max_loss_units deve ser >= 1.")
        Nmax = int(max_loss_units)

    P = _panjer_core(A, mu_total, N, sigma, Nmax, tol=tol, auto=(max_loss_units is None))

    s = P.sum()
    if s > 0:
        P = P / s
    return P


def _panjer_core(
    A: np.ndarray,
    mu_total: float,
    N: int,
    sigma: float,
    Nmax: int,
    *,
    tol: float,
    auto: bool,
) -> np.ndarray:
    """Núcleo da recursão (loop ``O(N·Nmax)``); ver :func:`panjer_recursion`.

    Implementa a recursão do CreditRisk+ single-sector em duas formas fechadas,
    ambas com o **mesmo esqueleto** ``P(j) = (1/j)·Σ_{k=1}^{min(j,N)} c(k,j)·P(j−k)``:

    * **Poisson** (``δ = σ² → 0``): ``c(k,j) = k·A(k)`` — a recursão de Panjer
      clássica ``P(j) = (1/j) Σ k·A(k)·P(j−k)``.
    * **Gama** (``δ > 0``): contagem binomial negativa com ``r = 1/δ`` e
      ``β = δ·μ_total/(1+δ·μ_total)``; ``c(k,j) = j·β·(1 + (r−1)·k/j)·a(k)``,
      i.e. ``P(j) = β·Σ (1 + (r−1)·k/j)·a(k)·P(j−k)`` (recursão de Giese).

    Estabilidade numérica (crítica em carteiras grandes)
    ----------------------------------------------------
    As duas recursões são **lineares e homogêneas** no valor semente ``P[0]``.
    O ``P[0]`` teórico (``exp(−μ_total)`` na Poisson, ``(1−β)^r`` na gama) sofre
    **underflow** para 0 quando ``μ_total`` é grande — matando toda a recursão.
    Por isso semeamos ``P[0] = 1`` e **normalizamos só no fim** (o fator de
    escala se cancela). Mas o pico não normalizado cresce como ``exp(μ_total)`` e
    estoura para ``+inf`` quando ``μ_total ≳ 700``. A solução é o
    **reescalonamento dinâmico**: sempre que os valores passam de um teto, todo
    o prefixo já calculado é dividido em bloco (operação que preserva todas as
    razões, porque a janela de convolução ``[j−N, j)`` está inteiramente dentro
    do prefixo reescalado). Assim a recursão fica estável para qualquer
    ``μ_total``, e o corte de cauda e a normalização final — ambos invariantes de
    escala — continuam corretos.
    """
    delta = sigma * sigma  # variância do fator gama (CV² = σ²)
    poisson = delta <= 1e-12

    # a(i) = A(i)/mu_total : PGF da severidade de um único default.
    a = A / mu_total                       # a[0] = 0 por construção (band >= 1)

    # Coeficientes k·A(k) (Poisson) ou a(k) (gama), pré-computados fora do laço.
    if poisson:
        coef = np.arange(N + 1, dtype=float) * A     # k·A(k), k = 0..N
    else:
        r = 1.0 / delta
        beta = (delta * mu_total) / (1.0 + delta * mu_total)
        k_idx = np.arange(N + 1, dtype=float)

    P = np.zeros(Nmax + 1)
    P[0] = 1.0                             # semente arbitrária (normalizada no fim)

    # Teto/piso de reescalonamento: mantém os valores longe de over/underflow.
    _CEIL = 1e250
    _FLOOR = 1e-250
    acc = 1.0                              # massa acumulada (na escala corrente)
    jmax = Nmax
    for j in range(1, Nmax + 1):
        kmax = min(j, N)
        janela = P[j - kmax:j][::-1]       # P(j−1), …, P(j−kmax)
        if poisson:
            s = float(np.dot(coef[1:kmax + 1], janela))
            P[j] = s / j
        else:
            # peso(k) = 1 + (r−1)·k/j ; contrib(k) = peso·a(k)
            peso = 1.0 + (r - 1.0) * (k_idx[1:kmax + 1] / j)
            s = float(np.dot(peso * a[1:kmax + 1], janela))
            P[j] = beta * s
        acc += P[j]

        # Corte de cauda por **janela** (avaliado ANTES do reescalonamento, na
        # mesma escala). Como um único default salta no máximo ``N`` bandas, só
        # há garantia de que não surgem novas "cristas" quando um bloco inteiro
        # de ``N`` bandas consecutivas soma massa desprezível ante ``acc``. Isso
        # respeita distribuições **multimodais** (carteiras de poucos nomes, com
        # vales de probabilidade zero entre picos), onde um corte banda-a-banda
        # truncaria cedo demais. Exige também ``j > 2N`` para já ter passado o
        # corpo mesmo que o modo esteja além da maior severidade.
        if auto and j > 2 * N:
            bloco = float(np.sum(P[j - N + 1:j + 1]))
            if bloco <= tol * acc:
                jmax = j
                break

        # Reescalonamento dinâmico: se o valor corrente saiu da faixa segura,
        # divide o prefixo inteiro por P[j] (preserva razões e a janela futura).
        val = P[j]
        if val > _CEIL or (0.0 < val < _FLOOR):
            P[: j + 1] /= val
            acc /= val

    return P[: jmax + 1]


# ======================================================================
# API principal
# ======================================================================
def creditrisk_plus(
    portfolio: "Portfolio",
    loss_unit: Optional[float] = None,
    sigma: float = 0.5,
    max_loss_units: Optional[int] = None,
) -> LossDistribution:
    """Distribuição de perdas **analítica** da carteira pelo CreditRisk+.

    Motor atuarial single-sector (um fator sistêmico gama comum). Obtém a
    distribuição inteira das perdas agregadas por recursão de Panjer/Giese, sem
    simulação — rápido e reprodutível, ideal como **benchmark** para desafiar o
    ASRF (:func:`.asrf.asrf_capital`) e o Monte Carlo (:func:`.monte_carlo.simulate`).

    Parameters
    ----------
    portfolio:
        A carteira (:class:`~yggdrasil.credit_risk.capital.portfolio.Portfolio`).
        Usa PD, LGD, EAD e ``n_obligors`` de cada segmento; ``rho`` e a estrutura
        de fatores **não** entram (o CreditRisk+ tem seu próprio fator gama). O
        risco sistêmico é governado por ``sigma``, único para toda a carteira.
    loss_unit:
        Unidade de perda ``L₀`` (granularidade da grade). Se ``None``, é
        escolhida automaticamente a partir da mediana das severidades — ver
        :func:`_choose_loss_unit`. Um ``L₀`` menor dá mais precisão ao custo de
        uma grade maior.
    sigma:
        Coeficiente de variação do fator sistêmico gama (média 1). ``sigma = 0``
        → *defaults* quase independentes (Poisson pura, cauda fina); ``sigma``
        maior → mais correlação sistêmica e **cauda mais gorda** (VaR sobe).
        Padrão 0,5, valor típico de calibração do CreditRisk+.
    max_loss_units:
        Número máximo de bandas da grade. Se ``None``, é dimensionado
        automaticamente para conter a cauda (massa acumulada > ``1 − 1e-9``).

    Returns
    -------
    LossDistribution
        Distribuição **discreta ponderada**: valores ``j·L₀`` (``j = 0..N_max``)
        e pesos ``P(L = j·L₀)``, com a ``expected`` fixada na EL analítica
        ``Σ PD·LGD·EAD`` da carteira. Dela derivam VaR, ES e capital econômico
        (ver :class:`~yggdrasil.credit_risk.capital.measures.LossDistribution`).

    Notes
    -----
    A EL da distribuição discretizada bate com ``Σ PD·LGD·EAD`` a menos do ruído
    numérico da recursão: a reescala de intensidade em :func:`_discretize`
    preserva a perda esperada **exatamente**, independentemente de ``loss_unit``.
    A granularidade afeta a **cauda** (VaR/ES), não a média; um ``loss_unit``
    menor refina a cauda ao custo de uma grade maior.
    """
    if sigma < 0:
        raise ValueError(f"sigma deve ser >= 0; recebido {sigma!r}.")
    if loss_unit is not None and loss_unit <= 0:
        raise ValueError(f"loss_unit deve ser > 0; recebido {loss_unit!r}.")

    sev, mu = _severities(portfolio)

    L0 = float(loss_unit) if loss_unit is not None else _choose_loss_unit(sev, mu, sigma)

    nu, mu_eff = _discretize(sev, mu, L0)

    P = panjer_recursion(nu, mu_eff, sigma=sigma, max_loss_units=max_loss_units)

    valores = np.arange(P.size, dtype=float) * L0
    el_analitica = portfolio.expected_loss()

    return LossDistribution(
        losses=valores,
        weights=P,
        expected=el_analitica,
        name=f"{portfolio.name} · CreditRisk+ (σ={sigma:g})",
    )


__all__ = [
    "creditrisk_plus",
    "panjer_recursion",
]
