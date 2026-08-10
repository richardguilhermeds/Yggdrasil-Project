"""
Motor v2 — Simulação de Monte Carlo multifatorial
=================================================
A extensão natural do ASRF e o **padrão de mercado para capital econômico
interno** (Seção 3.2 do guia). A lógica:

1. Define-se um conjunto de **fatores sistêmicos correlacionados** (um por
   produto/segmento — cartão, consignado, veículos), com uma matriz de
   correlação entre fatores.
2. Em cada cenário simulado, sorteiam-se os fatores; condicionam-se as PDs de
   cada segmento ao cenário (fórmula de Vasicek); sorteiam-se (ou tomam-se pelo
   limite de grandes números) os *defaults* e, se desejado, as **LGDs
   estocásticas** correlacionadas ao ciclo; agrega-se a perda total.
3. Repetindo dezenas/centenas de milhares de vezes, obtém-se a **distribuição
   empírica completa** de perdas, da qual se extraem VaR, ES e as contribuições
   de cada produto.

Por que multifatorial importa: com um fator por produto e a matriz de
correlação entre fatores, o modelo captura o **benefício de diversificação** —
cartão, consignado e veículos não estressam ao mesmo tempo com a mesma
intensidade —, que o Pilar 1 ignora por construção.

Extensões opcionais de :func:`simulate`: ``copula="t"`` adiciona **dependência
de cauda** entre os latentes (mistura qui-quadrado — choque de variância comum
por cenário) e ``lgd_dist="beta"`` troca a LGD normal-clipada por uma **Beta
com *moment matching***, de suporte natural em ``[0, 1]``. Já
``importance_sampling=True`` liga a **amostragem por importância**
(Glasserman–Li) na cauda: os fatores são sorteados com a média deslocada na
direção adversa e cada cenário carrega um peso de verossimilhança, propagado
às medidas de risco e à alocação — mesmo VaR/ES esperados, variância bem
menor no quantil de cauda. Os defaults (``copula="gaussian"``,
``lgd_dist="normal"``, ``importance_sampling=False``) preservam o
comportamento histórico exato.

Validação de sanidade (guia, bloco E): com **um único fator** e carteira
**granular** (``granular=True``), a simulação reproduz o ASRF analítico.

Em ambiente Databricks/PySpark a simulação paraleliza bem (cada cenário é
independente); aqui a implementação é vetorizada em NumPy, adequada ao nível de
**segmento homogêneo** (não de contrato).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, norm, t as student_t

from .measures import DEFAULT_CONFIDENCE, LossDistribution, _weighted_quantile

if TYPE_CHECKING:
    from .portfolio import Portfolio


# ======================================================================
# Álgebra: raiz de Cholesky robusta a matrizes quase-singulares
# ======================================================================
def _safe_cholesky(corr: np.ndarray) -> np.ndarray:
    """Fator ``L`` tal que ``L @ L.T ≈ corr``.

    Tenta Cholesky; se a matriz não for positiva-definida (comum em matrizes de
    correlação estimadas), projeta para a correlação positiva-definida mais
    próxima por *clipping* de autovalores e refatoriza. Ver
    :func:`yggdrasil.credit_risk.capital.correlation.nearest_correlation`.
    """
    corr = np.asarray(corr, dtype=float)
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Projeção espectral: zera autovalores negativos e renormaliza a diagonal.
        vals, vecs = np.linalg.eigh(corr)
        vals = np.clip(vals, 1e-10, None)
        A = vecs @ np.diag(vals) @ vecs.T
        d = np.sqrt(np.clip(np.diag(A), 1e-12, None))
        A = A / np.outer(d, d)
        A = (A + A.T) / 2.0
        try:
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:  # pragma: no cover - último recurso
            return np.linalg.cholesky(A + 1e-8 * np.eye(A.shape[0]))


# ======================================================================
# Importance sampling (Glasserman–Li): tilt exponencial do fator sistêmico
# ======================================================================
def _tilt_direction(
    L: np.ndarray,
    fac_of: np.ndarray,
    c_gauss: np.ndarray,
    sqrt_rho: np.ndarray,
    sqrt_1mrho: np.ndarray,
    lgds: np.ndarray,
    eads: np.ndarray,
    n_factors: int,
) -> Optional[np.ndarray]:
    """Direção adversa **unitária** no espaço dos choques independentes ``z``.

    É o gradiente da perda esperada condicional aos fatores (aproximação
    gaussiana granular, avaliado em ``Y = 0``), mapeado de volta para o espaço
    dos choques via ``Y = z @ L.T``. Como a perda cresce quando os fatores
    caem, a direção aponta para fatores negativos. Retorna ``None`` quando não
    há componente sistêmica (todos os ``rho`` nulos) — o tilt seria inócuo.
    """
    # sensibilidade da perda condicional de cada segmento ao seu fator, em Y=0
    sens = eads * lgds * norm.pdf(c_gauss / sqrt_1mrho) * sqrt_rho / sqrt_1mrho
    grad_y = np.zeros(n_factors)
    np.add.at(grad_y, fac_of, sens)                 # agrega por fator
    d_z = -(L.T @ grad_y)                           # perda cresce quando Y cai
    nrm = float(np.linalg.norm(d_z))
    if not np.isfinite(nrm) or nrm <= 0.0:
        return None
    return d_z / nrm


def _calibrate_tilt(
    target: float,
    d_hat: np.ndarray,
    L: np.ndarray,
    fac_of: np.ndarray,
    c_gauss: np.ndarray,
    sqrt_rho: np.ndarray,
    sqrt_1mrho: np.ndarray,
    lgds: np.ndarray,
    eads: np.ndarray,
    s_max: float = 12.0,
) -> float:
    """Magnitude ``μ`` do tilt mirando o quantil-alvo (heurística documentada).

    Resolve por bisseção o ``μ`` tal que a perda média do cenário **tiltado**
    — a perda esperada condicional avaliada na média deslocada dos fatores
    (``Y = μ·d̂ @ Lᵀ``, aproximação gaussiana granular) — iguale ``target``
    (o VaR de um piloto pequeno sem IS). Assim a massa da simulação se
    concentra em torno do quantil de interesse. A perda é monótona em ``μ``
    ao longo da direção adversa; quando o alvo não é atingível o resultado é
    truncado em ``s_max``.
    """
    def loss_at(s: float) -> float:
        y = ((s * d_hat) @ L.T)[fac_of]
        p = norm.cdf((c_gauss - sqrt_rho * y) / sqrt_1mrho)
        return float(np.sum(p * lgds * eads))

    lo, hi = 0.0, 1.0
    while loss_at(hi) < target and hi < s_max:      # expande até envolver o alvo
        hi = min(2.0 * hi, s_max)
    if loss_at(hi) < target:
        return float(hi)
    for _ in range(60):                             # bisseção (precisão ~1e-16·s_max)
        mid = 0.5 * (lo + hi)
        if loss_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


# ======================================================================
# Resultado da simulação
# ======================================================================
@dataclass
class SimulationResult:
    """Saída da simulação: distribuição de perdas + perdas por segmento/cenário.

    ``segment_losses`` (``n_scenarios × n_segments``) é o insumo da **alocação de
    Euler** (contribuição condicional à cauda) — ver :mod:`.allocation`.

    ``weights`` carrega os pesos de verossimilhança por cenário quando a
    simulação rodou com *importance sampling* (``None`` = amostra
    equiponderada). Os pesos são propagados para a distribuição de perdas,
    para a alocação de Euler e para o benefício de diversificação.
    """

    losses: np.ndarray                    # (n_scenarios,) perda total por cenário
    segment_losses: Optional[np.ndarray]  # (n_scenarios, n_segments) ou None
    segment_names: List[str]
    q: float
    expected_loss: float                  # EL analítica (Σ PD·LGD·EAD)
    n_scenarios: int
    seed: Optional[int] = None
    metric: str = "var"
    weights: Optional[np.ndarray] = None  # pesos de importância por cenário ou None
    _dist: Optional[LossDistribution] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def distribution(self) -> LossDistribution:
        """A distribuição de perdas (usa a EL analítica exata como ``el``).

        Com *importance sampling*, os pesos de verossimilhança seguem junto —
        VaR/ES saem do quantil **ponderado**.
        """
        if self._dist is None:
            self._dist = LossDistribution(
                self.losses, weights=self.weights,
                expected=self.expected_loss, name="monte_carlo")
        return self._dist

    def _q(self, q: Optional[float]) -> float:
        return self.q if q is None else float(q)

    def var(self, q: Optional[float] = None) -> float:
        return self.distribution().var(self._q(q))

    def es(self, q: Optional[float] = None) -> float:
        return self.distribution().es(self._q(q))

    def economic_capital(self, q: Optional[float] = None, metric: Optional[str] = None) -> float:
        return self.distribution().economic_capital(self._q(q), metric or self.metric)

    # ------------------------------------------------------------------
    def allocate(self, q: Optional[float] = None, metric: str = "es",
                 alpha: float = 0.05) -> pd.DataFrame:
        """Alocação de Euler do capital pelos segmentos (contribuição à cauda).

        Ver :func:`yggdrasil.credit_risk.capital.allocation.euler_allocation`.
        """
        from .allocation import euler_allocation
        return euler_allocation(self, q=self._q(q), metric=metric, alpha=alpha)

    def diversification_benefit(self, q: Optional[float] = None) -> dict:
        """Benefício de diversificação: capital **isolado** (soma dos CE de cada
        segmento como se estivesse sozinho) menos o capital **integrado** da
        carteira. É a informação gerencial que o Pilar 1 não dá.
        """
        if self.segment_losses is None:
            raise ValueError("segment_losses não foi armazenado (store_segment_losses=False).")
        qq = self._q(q)
        standalone = 0.0
        for j in range(self.segment_losses.shape[1]):
            col = self.segment_losses[:, j]
            if self.weights is None:
                standalone += float(np.quantile(col, qq) - col.mean())
            else:
                # amostra ponderada (importance sampling): quantil e média ponderados
                standalone += float(_weighted_quantile(col, self.weights, qq)
                                    - np.average(col, weights=self.weights))
        integrated = self.economic_capital(qq, metric="var")
        return {
            "capital_isolado": standalone,
            "capital_integrado": integrated,
            "beneficio_diversificacao": standalone - integrated,
            "beneficio_pct": (standalone - integrated) / standalone if standalone > 0 else np.nan,
        }

    def summary(self) -> pd.DataFrame:
        d = self.distribution()
        return pd.DataFrame([{
            "nivel_confianca": self.q,
            "n_cenarios": self.n_scenarios,
            "EL": d.el,
            "VaR": d.var(self.q),
            "ES": d.es(self.q),
            "CE_var": d.var(self.q) - d.el,
            "CE_es": d.es(self.q) - d.el,
        }])

    def __repr__(self) -> str:  # pragma: no cover
        d = self.distribution()
        return (f"SimulationResult(n={self.n_scenarios}, q={self.q}, "
                f"EL={d.el:,.2f}, VaR={d.var(self.q):,.2f}, ES={d.es(self.q):,.2f})")


# ======================================================================
# Motor
# ======================================================================
def simulate(
    portfolio: "Portfolio",
    n_scenarios: int = 100_000,
    q: float = DEFAULT_CONFIDENCE,
    seed: Optional[int] = None,
    *,
    granular: bool = True,
    stochastic_lgd: bool = False,
    pd_lgd_corr: float = 0.0,
    rho_default: float = 0.15,
    copula: str = "gaussian",
    t_dof: float = 8.0,
    lgd_dist: str = "normal",
    antithetic: bool = False,
    importance_sampling: bool = False,
    tilt: Union[str, float] = "auto",
    store_segment_losses: bool = True,
    block_size: int = 50_000,
) -> SimulationResult:
    """Simula a distribuição de perdas da carteira por Monte Carlo multifatorial.

    Parameters
    ----------
    portfolio:
        A carteira. Cada segmento carrega no seu fator sistêmico com ``√rho``; a
        dependência entre fatores vem de ``portfolio.factor_corr``.
    n_scenarios:
        Número de cenários. O quantil de cauda (99,9%) exige muitos cenários —
        ver :func:`yggdrasil.credit_risk.capital.validation.convergence`.
    q:
        Nível de confiança de referência do resultado.
    seed:
        Semente do gerador (reprodutibilidade).
    granular:
        ``True`` (padrão) — usa a **PD condicional** diretamente como fração de
        *default* do segmento (limite de grandes números: carteira granular).
        Reproduz o ASRF quando há um único fator. ``False`` — sorteia o número de
        *defaults* por ``Binomial(n_obligors, p)``, capturando risco
        idiossincrático/de **concentração** de nomes.
    stochastic_lgd:
        Se ``True``, a LGD é estocástica nos segmentos com ``lgd_vol > 0``.
    pd_lgd_corr:
        Correlação **adversa** PD–LGD em ``[0, 1)`` (relevante em veículos): em
        cenários ruins, a severidade sobe junto com os *defaults*. Só tem efeito
        com ``stochastic_lgd=True``.
    rho_default:
        ``rho`` usado nos segmentos com ``rho=None``.
    copula:
        Cópula dos latentes: ``"gaussian"`` (padrão — comportamento histórico)
        ou ``"t"`` (cópula t de Student com ``t_dof`` graus de liberdade). A
        cópula t introduz **dependência de cauda**: além dos fatores
        correlacionados, todos os latentes do cenário compartilham um choque de
        variância comum ``W = ν/χ²_ν``. Construção do latente de *default*:
        ``X = √W · (√ρ·Y + √(1−ρ)·ε) ~ t_ν``, com ``Y`` o fator gaussiano e
        ``ε`` idiossincrático normal — a mistura ``√W`` multiplica o par
        completo (fator **e** idiossincrático), o que caracteriza a cópula t
        completa. O *default* ocorre quando ``X ≤ t_ν⁻¹(PD)``: o limiar usa o
        **quantil da t** (não ``Φ⁻¹``) para preservar a PD incondicional — e
        portanto a EL — sob o latente t. Condicional a ``(Y, W)``, ``ε`` é
        normal padrão e a PD condicional fica
        ``Φ( (t_ν⁻¹(PD)/√W − √ρ·Y) / √(1−ρ) )``.
    t_dof:
        Graus de liberdade ``ν`` da cópula t (só usado com ``copula="t"``).
        ``ν`` baixo (ex.: 4) engorda a cauda; ``ν → ∞`` recupera a gaussiana.
    lgd_dist:
        Distribuição da LGD estocástica: ``"normal"`` (padrão — normal clipada
        em ``[0, 1]``, comportamento histórico, que concentra massa artificial
        nas bordas) ou ``"beta"`` (Beta com *moment matching*: ``α``/``β``
        derivados de média ``lgd`` e desvio ``lgd_vol``, suporte natural em
        ``[0, 1]``). A correlação adversa PD–LGD é preservada por transformação
        de quantil do mesmo latente sistêmico: ``u = F(latente)`` (CDF normal
        ou t, conforme a cópula) e ``LGD = Beta⁻¹(u; α, β)``. Casos degenerados
        (``lgd_vol = 0`` ou média em ``{0, 1}``) permanecem determinísticos;
        desvio inviável (``σ² ≥ μ(1−μ)``) é truncado ao máximo atingível.
    antithetic:
        Variáveis **antitéticas** (redução de variância): metade dos sorteios do
        fator usa ``+z`` e a outra metade ``−z``.
    importance_sampling:
        Liga a **amostragem por importância** (Glasserman–Li) no fator
        sistêmico: a média dos choques gaussianos independentes dos fatores é
        deslocada na direção adversa (*exponential tilting*, ``z ~ N(μ, I)``)
        e cada cenário ``i`` carrega o peso de verossimilhança
        ``w_i = exp(−μ·z_i + ‖μ‖²/2)`` (produto nas dimensões dos fatores),
        que o :class:`SimulationResult` propaga para a distribuição de
        perdas, para a alocação de Euler e para o benefício de
        diversificação. O estimador ponderado permanece **não-viesado**; a
        variância do quantil de cauda cai porque a região do quantil passa a
        ser visitada com frequência muito maior. ``False`` (padrão) preserva
        o comportamento histórico exato — mesmos sorteios, mesmos resultados.
    tilt:
        Magnitude ``μ`` do deslocamento ao longo da direção adversa unitária
        (só usado com ``importance_sampling=True``). ``"auto"`` (padrão)
        calibra ``μ`` com uma heurística simples: roda um **piloto pequeno**
        sem IS (até 10 mil cenários, semente derivada de ``seed``), toma o
        VaR\\ :sub:`q` piloto como alvo e resolve por bisseção o ``μ`` tal que
        a perda média do cenário tiltado (aproximação gaussiana granular)
        iguale esse alvo — ver :func:`_calibrate_tilt`. Um ``float >= 0`` usa
        a magnitude dada diretamente. A direção adversa é o gradiente da
        perda esperada condicional nos fatores, mapeado para o espaço dos
        choques independentes (:func:`_tilt_direction`); com ``rho = 0`` em
        todos os segmentos não há componente sistêmica e o tilt é nulo
        (pesos 1).
    store_segment_losses:
        Armazena a matriz perda-por-segmento (necessária para alocação de Euler
        e benefício de diversificação). Desligue para poupar memória.
    block_size:
        Tamanho do bloco de cenários processados por vez (controle de memória).

    Returns
    -------
    SimulationResult
    """
    from .portfolio import Portfolio  # noqa: F401 (garante o tipo em runtime)

    if n_scenarios < 1:
        raise ValueError("n_scenarios deve ser >= 1.")
    if not (0.0 < q < 1.0):
        raise ValueError(f"q deve estar em (0, 1); recebido {q!r}.")
    if not (0.0 <= pd_lgd_corr < 1.0):
        raise ValueError("pd_lgd_corr deve estar em [0, 1).")
    copula = str(copula).lower()
    if copula not in ("gaussian", "t"):
        raise ValueError(f"copula deve ser 'gaussian' ou 't'; recebido {copula!r}.")
    if copula == "t":
        t_dof = float(t_dof)
        if not np.isfinite(t_dof) or t_dof <= 0.0:
            raise ValueError(
                f"t_dof (graus de liberdade da cópula t) deve ser > 0; recebido {t_dof!r}.")
    lgd_dist = str(lgd_dist).lower()
    if lgd_dist not in ("normal", "beta"):
        raise ValueError(f"lgd_dist deve ser 'normal' ou 'beta'; recebido {lgd_dist!r}.")
    if importance_sampling:
        if isinstance(tilt, str):
            if tilt.lower() != "auto":
                raise ValueError(f"tilt deve ser 'auto' ou um float >= 0; recebido {tilt!r}.")
        else:
            tilt = float(tilt)
            if not np.isfinite(tilt) or tilt < 0.0:
                raise ValueError(f"tilt deve ser 'auto' ou um float >= 0; recebido {tilt!r}.")

    rng = np.random.default_rng(seed)

    pds = portfolio.pds()
    lgds = portfolio.lgds()
    eads = portfolio.eads()
    rhos = portfolio.rhos(default=rho_default)
    n_obl = portfolio.n_obligors()
    lgd_vols = portfolio.lgd_vols()
    fac_of = portfolio.factor_of()                      # (n_seg,) índice do fator
    n_seg = portfolio.n_segments
    F = portfolio.n_factors

    L = _safe_cholesky(portfolio.factor_corr)           # (F, F)
    pd_clip = np.clip(pds, 1e-12, 1 - 1e-12)
    if copula == "t":
        # Limiar no quantil da t_ν: como o latente é t_ν, é o t_ν⁻¹(PD) que
        # preserva a PD incondicional (e a EL) — ver docstring de ``copula``.
        inv_pd = student_t.ppf(pd_clip, df=t_dof)
    else:
        inv_pd = norm.ppf(pd_clip)                      # limiar de default por segmento
    sqrt_rho = np.sqrt(rhos)
    sqrt_1mrho = np.sqrt(1.0 - rhos)
    ead_per_obl = eads / np.maximum(n_obl, 1)

    # Parâmetros da Beta por segmento (moment matching de média/desvio) quando
    # ``lgd_dist='beta'``. Segmento degenerado (vol 0 ou média em {0, 1}) fica
    # determinístico; desvio inviável (σ² ≥ μ(1−μ)) é truncado ao máximo.
    if lgd_dist == "beta":
        mu = lgds
        beta_ok = (lgd_vols > 0) & (mu > 0.0) & (mu < 1.0)
        var_max = np.clip(mu * (1.0 - mu), 1e-300, None)
        sd_eff = np.where(beta_ok, np.minimum(lgd_vols, 0.995 * np.sqrt(var_max)), 1.0)
        conc = np.where(beta_ok, var_max / sd_eff ** 2 - 1.0, 2.0)   # α + β
        beta_a = np.clip(mu * conc, 1e-12, None)
        beta_b = np.clip((1.0 - mu) * conc, 1e-12, None)

    # ---- importance sampling: direção adversa + magnitude do tilt ---------
    # Todo o sorteio extra (piloto) usa um gerador próprio: com o IS desligado
    # o fluxo do RNG principal fica intocado (regressão bit a bit garantida).
    mu_z: Optional[np.ndarray] = None
    weights_arr: Optional[np.ndarray] = None
    mu_half_sq = 0.0
    if importance_sampling:
        c_gauss = norm.ppf(pd_clip)                 # direção na aprox. gaussiana
        d_hat = _tilt_direction(L, fac_of, c_gauss, sqrt_rho, sqrt_1mrho,
                                lgds, eads, F)
        if d_hat is None:
            mu_z = np.zeros(F)                      # sem sistêmico: pesos ficam 1
        elif isinstance(tilt, str):                 # 'auto': mira o VaR de um piloto
            n_pilot = int(min(10_000, n_scenarios))
            pilot_seed = None if seed is None else int(seed) + 202_406
            pilot = simulate(
                portfolio, n_scenarios=n_pilot, q=q, seed=pilot_seed,
                granular=granular, stochastic_lgd=stochastic_lgd,
                pd_lgd_corr=pd_lgd_corr, rho_default=rho_default,
                copula=copula, t_dof=t_dof, lgd_dist=lgd_dist,
                antithetic=antithetic, store_segment_losses=False,
                block_size=block_size)
            mu_z = _calibrate_tilt(pilot.var(q), d_hat, L, fac_of, c_gauss,
                                   sqrt_rho, sqrt_1mrho, lgds, eads) * d_hat
        else:
            mu_z = float(tilt) * d_hat
        mu_half_sq = 0.5 * float(mu_z @ mu_z)
        weights_arr = np.empty(n_scenarios, dtype=float)

    total_losses = np.empty(n_scenarios, dtype=float)
    seg_losses = (np.empty((n_scenarios, n_seg), dtype=float)
                  if store_segment_losses else None)

    start = 0
    while start < n_scenarios:
        m = min(block_size, n_scenarios - start)
        # ---- fatores sistêmicos correlacionados (m, F) --------------------
        if antithetic:
            half = (m + 1) // 2
            z0 = rng.standard_normal((half, F))
            z = np.vstack([z0, -z0])[:m]
        else:
            z = rng.standard_normal((m, F))
        if mu_z is not None:
            # tilt exponencial: z ~ N(0, I) → N(μ, I); o peso é a razão de
            # verossimilhança w = φ(z)/φ_μ(z) = exp(−μ·z + ‖μ‖²/2), produto
            # nas F dimensões dos choques independentes.
            z = z + mu_z[None, :]
            weights_arr[start:start + m] = np.exp(mu_half_sq - z @ mu_z)
        Y = z @ L.T                                     # cov(Y) = factor_corr
        Y_seg = Y[:, fac_of]                            # (m, n_seg) fator de cada segmento

        # ---- mistura da cópula t: choque de variância comum W = ν/χ²_ν ----
        # O fator misto é Y_t = √W·Y; a mistura entra na PD condicional pelo
        # limiar reescalado t_ν⁻¹(PD)/√W (equivalente, dividindo o latente
        # X = √W·(√rho·Y + √(1−rho)·ε) por √W).
        if copula == "t":
            w_chi = rng.chisquare(t_dof, size=m)
            sqrt_w = np.sqrt(t_dof / np.maximum(w_chi, 1e-300))
            thresh = inv_pd[None, :] / sqrt_w[:, None]
        else:
            thresh = inv_pd[None, :]

        # ---- PD condicional ao cenário (Vasicek): baixo Y = ruim ----------
        # gaussiana: p = N( (N⁻¹(PD) − √rho · Y) / √(1−rho) )
        # t:        p = N( (t_ν⁻¹(PD)/√W − √rho · Y) / √(1−rho) )
        cond_pd = norm.cdf((thresh - sqrt_rho[None, :] * Y_seg) / sqrt_1mrho[None, :])

        # ---- fração/nº de defaults ---------------------------------------
        if granular:
            default_frac = cond_pd                       # limite de grandes números
        else:
            draws = rng.binomial(n_obl[None, :].repeat(m, axis=0), cond_pd)
            default_frac = draws / np.maximum(n_obl[None, :], 1)

        # ---- LGD (determinística ou estocástica, correlacionada ao ciclo) -
        if stochastic_lgd and np.any(lgd_vols > 0):
            zeta = rng.standard_normal((m, n_seg))
            # latente da LGD: componente sistêmica (−Y: cenário ruim → LGD alta)
            # + idiossincrática. corr(latente_LGD, fator) = pd_lgd_corr. Na
            # cópula t, o latente recebe o MESMO √W do default (dependência de
            # cauda PD–LGD) e vira t_ν; a CDF da t devolve o uniforme.
            lgd_lat = (-pd_lgd_corr * Y_seg
                       + np.sqrt(1.0 - pd_lgd_corr ** 2) * zeta)
            u_lgd = (student_t.cdf(sqrt_w[:, None] * lgd_lat, df=t_dof)
                     if copula == "t" else None)
            if lgd_dist == "beta":
                # Transformação de quantil: u = F(latente) → Beta⁻¹(u; α, β).
                if u_lgd is None:
                    u_lgd = norm.cdf(lgd_lat)
                u_lgd = np.clip(u_lgd, 1e-12, 1.0 - 1e-12)
                lgd_eff = np.where(
                    beta_ok[None, :],
                    beta_dist.ppf(u_lgd, beta_a[None, :], beta_b[None, :]),
                    lgds[None, :])
            elif u_lgd is not None:
                # cópula t + marginal normal-clipada: o quantil normal do
                # uniforme preserva a marginal histórica sob o latente t.
                z_lgd = norm.ppf(np.clip(u_lgd, 1e-12, 1.0 - 1e-12))
                lgd_eff = np.clip(lgds[None, :] + lgd_vols[None, :] * z_lgd, 0.0, 1.0)
            else:
                # caminho histórico (gaussiana + normal-clipada), bit a bit.
                lgd_eff = np.clip(lgds[None, :] + lgd_vols[None, :] * lgd_lat, 0.0, 1.0)
        else:
            lgd_eff = lgds[None, :]

        # ---- perda por segmento e total ----------------------------------
        block = default_frac * lgd_eff * eads[None, :]   # (m, n_seg)
        if seg_losses is not None:
            seg_losses[start:start + m, :] = block
        total_losses[start:start + m] = block.sum(axis=1)
        start += m

    el = portfolio.expected_loss()
    return SimulationResult(
        losses=total_losses,
        segment_losses=seg_losses,
        segment_names=portfolio.segment_names(),
        q=float(q),
        expected_loss=el,
        n_scenarios=int(n_scenarios),
        seed=seed,
        weights=weights_arr,
    )


__all__ = ["simulate", "SimulationResult"]
