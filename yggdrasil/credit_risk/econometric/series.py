"""
Contrato de dados e gerador de séries sintéticas (Guia §2.1, §6.2 ``data``, §7)
==============================================================================
Duas responsabilidades:

Contrato de dados — :class:`RiskSeries`
    O objeto que percorre toda a biblioteca: a **série temporal agregada** de um
    parâmetro de risco (taxa de *default*, LGD ou CCF médios) de um segmento
    homogêneo, com o índice temporal, a frequência e o ``kind`` (``'pd'``,
    ``'lgd'`` ou ``'ccf'``) que decide a transformação padrão. É o eixo
    **temporal** dos modelos satélite — distinto do eixo transversal da escoragem
    (Guia §1).

Gerador de séries sintéticas — :func:`simulate_pd_series`, :func:`simulate_lgd_series`,
:func:`simulate_ccf_series`, :func:`make_reference_study`
    O guia chama os dados sintéticos de **"o investimento de melhor retorno do
    plano"** (§7): séries geradas a partir de um **processo gerador conhecido**
    (ARDL em logit/probit ou fator ``Z`` de Vasicek, dirigido por variáveis macro
    e um choque sistêmico) permitem a única validação forte possível para código
    econométrico — testar se cada método **recupera a verdade**. Cada gerador
    devolve a série observada, a macro que a dirigiu e o dicionário ``truth`` com
    os coeficientes verdadeiros.

Nada aqui depende de Spark/Delta: os conectores de dados internos (leitura das
bases contratuais, agregação por safra/estoque) são a parte **específica de cada
instituição**, mantida fora do pacote aberto. O que fica é o contrato e o
gerador — reprodutíveis e públicos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import transforms as _tf

#: Tipos de parâmetro de risco suportados. ``pd`` = taxa de *default*; ``lgd`` =
#: severidade; ``ccf`` = fator de conversão de crédito (utilização até o default).
RISK_KINDS = ("pd", "lgd", "ccf")

#: Transformação padrão por tipo. PD e taxas usam logit; a alternativa para PD é
#: o fator ``Z`` de Vasicek (modelo dedicado). LGD/CCF também usam logit por
#: padrão (ou regressão beta / fractional logit, que têm link próprio).
DEFAULT_LINK = {"pd": "logit", "lgd": "logit", "ccf": "logit"}


# ======================================================================
# Contrato de dados
# ======================================================================
@dataclass
class RiskSeries:
    """Série temporal de um parâmetro de risco de um segmento homogêneo.

    Parameters
    ----------
    values:
        A série (``pandas.Series``) da taxa por período, em ``[0, 1]``, com índice
        :class:`~pandas.DatetimeIndex` ou :class:`~pandas.PeriodIndex`.
    kind:
        Um de :data:`RISK_KINDS` (``'pd'``, ``'lgd'``, ``'ccf'``).
    segment:
        Nome do segmento homogêneo (ex.: ``"cartao_revolver"``).
    frequency:
        Frequência pandas da série (``"MS"`` mensal, ``"QS"`` trimestral).
    exposure:
        Série opcional de **base em risco** (nº de contratos/saldo para PD, nº de
        resoluções para LGD) alinhada a ``values`` — usada em ponderações e na
        modelagem da variância amostral (a taxa de um segmento pequeno é mais
        ruidosa).
    metadata:
        Dicionário livre (fonte, definição de *default*, janela, etc.) para a
        linhagem exigida pela governança (Guia §4.4).
    """

    values: pd.Series
    kind: str = "pd"
    segment: str = ""
    frequency: str = "MS"
    exposure: Optional[pd.Series] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in RISK_KINDS:
            raise ValueError(f"kind deve ser um de {RISK_KINDS}; recebido {self.kind!r}.")
        if not isinstance(self.values, pd.Series):
            self.values = pd.Series(self.values)
        v = self.values.to_numpy(dtype=float)
        if v.size == 0:
            raise ValueError("values não pode ser vazio.")
        if np.any(~np.isfinite(v)):
            raise ValueError(f"values do segmento {self.segment!r} contém NaN/inf.")
        if np.any((v < 0.0) | (v > 1.0)):
            raise ValueError(
                f"values do segmento {self.segment!r} deve estar em [0, 1] "
                "(taxa/severidade). Recorte CCF > 1 antes de construir a série."
            )
        if not isinstance(self.values.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            raise TypeError("o índice de values deve ser DatetimeIndex ou PeriodIndex.")
        if self.exposure is not None and len(self.exposure) != len(self.values):
            raise ValueError("exposure deve ter o mesmo comprimento de values.")

    # -- conveniências ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.values)

    @property
    def index(self) -> pd.Index:
        return self.values.index

    @property
    def default_link(self) -> str:
        """Transformação padrão para este ``kind`` (:data:`DEFAULT_LINK`)."""
        return DEFAULT_LINK[self.kind]

    def ttc(self, method: str = "mean") -> float:
        """Nível *through-the-cycle* (média/mediana de longo prazo) da série.

        Para PD, é a ``PD_TTC`` que alimenta a inversão de Vasicek
        (:func:`~yggdrasil.credit_risk.econometric.transforms.vasicek_z`) e o motor
        de capital. Delega a
        :func:`yggdrasil.credit_risk.capital.pit_to_ttc` quando disponível.
        """
        arr = self.values.to_numpy(dtype=float)
        return float(np.median(arr)) if method == "median" else float(np.mean(arr))

    def transformed(self, link: Optional[str] = None) -> pd.Series:
        """A série na escala de modelagem (aplica o *link*; padrão :attr:`default_link`)."""
        return _tf.get_link(link or self.default_link).forward(self.values)

    def to_frame(self) -> pd.DataFrame:
        """A série como DataFrame de uma coluna nomeada pelo ``kind`` do segmento."""
        col = f"{self.kind}_{self.segment}" if self.segment else self.kind
        return self.values.to_frame(col)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"RiskSeries(kind={self.kind!r}, segment={self.segment!r}, "
                f"n={len(self)}, freq={self.frequency!r}, "
                f"mean={self.values.mean():.4f})")


def as_risk_series(obj, kind: str = "pd", segment: str = "") -> RiskSeries:
    """Coage ``pandas.Series``/:class:`RiskSeries` em :class:`RiskSeries`."""
    if isinstance(obj, RiskSeries):
        return obj
    return RiskSeries(values=obj, kind=kind, segment=segment)


# ======================================================================
# Estruturas de saída do gerador sintético
# ======================================================================
@dataclass
class SyntheticSeries:
    """Uma série sintética com a **verdade** conhecida (para testes de recuperação)."""

    series: RiskSeries          # série observada (com ruído de observação)
    macro: pd.DataFrame         # variáveis macro que a dirigiram
    truth: dict                 # parâmetros verdadeiros do DGP
    latent: pd.Series           # taxa "verdadeira" antes do ruído de observação

    def betas(self) -> dict:
        """Dicionário ``{(variavel, defasagem): coef}`` dos coeficientes macro verdadeiros."""
        return dict(self.truth.get("betas", {}))


@dataclass
class ReferenceStudy:
    """O **estudo de referência** do guia (§7): um segmento de ponta a ponta.

    Reúne, sobre a **mesma** macro e o mesmo horizonte (com recessão e evento de
    pandemia), as três séries de parâmetros — PD, LGD e CCF — para servir de
    *smoke test* de integração vivo e de notebook de demonstração.
    """

    macro: pd.DataFrame
    pd: SyntheticSeries
    lgd: SyntheticSeries
    ccf: SyntheticSeries
    events: dict
    segment: str = "segmento_referencia"


# ======================================================================
# Gerador de macro
# ======================================================================
def simulate_macro(
    n_periods: int = 120,
    start: str = "2015-01-01",
    freq: str = "MS",
    seed: int = 0,
    recession: Optional[tuple] = ("2016-01", "2017-06"),
    covid: Optional[tuple] = ("2020-03", "2020-08"),
) -> pd.DataFrame:
    """Gera um painel de variáveis macro **plausíveis** e persistentes.

    Cada variável é um processo AR(1) com média e persistência realistas, mais uma
    **recessão** (pico de desemprego/juros, queda de renda) e um **evento de
    pandemia** injetados como bumps determinísticos — o cenário canônico do guia
    para testar dummies de evento (§2.3). As colunas:

    * ``desemprego`` — taxa de desemprego (%), persistente, sobe na recessão.
    * ``renda`` — variação da massa de renda real (%), cai na recessão.
    * ``juros`` — taxa básica (%), sobe na recessão.
    * ``cambio`` — variação cambial (%), ruidosa, salta na pandemia.
    * ``inadimplencia`` — inadimplência agregada do SFN (%), coincidente ao ciclo.

    Returns
    -------
    pandas.DataFrame
        Indexado por :class:`~pandas.DatetimeIndex` na frequência ``freq``.
    """
    if n_periods <= 0:
        raise ValueError("n_periods deve ser > 0.")
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n_periods, freq=freq)

    def ar1(mu: float, phi: float, sigma: float) -> np.ndarray:
        e = rng.normal(0.0, sigma, n_periods)
        x = np.empty(n_periods)
        x[0] = mu + e[0] / np.sqrt(max(1e-6, 1 - phi**2))
        for t in range(1, n_periods):
            x[t] = mu + phi * (x[t - 1] - mu) + e[t]
        return x

    desemprego = ar1(9.0, 0.92, 0.35)
    renda = ar1(1.5, 0.55, 0.9)
    juros = ar1(9.0, 0.95, 0.4)
    cambio = ar1(0.0, 0.3, 2.5)
    inadimplencia = ar1(3.2, 0.9, 0.18)

    def _mask(window: Optional[tuple]) -> np.ndarray:
        if window is None:
            return np.zeros(n_periods, dtype=bool)
        ini, fim = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        return (idx >= ini) & (idx <= fim)

    rec = _mask(recession)
    cov = _mask(covid)
    # Recessão: desemprego e juros sobem, renda cai, inadimplência sobe (com curva).
    if rec.any():
        ramp = np.linspace(0.0, 1.0, rec.sum()) ** 0.7
        desemprego[rec] += 3.5 * ramp
        juros[rec] += 3.0 * ramp
        renda[rec] -= 3.0 * ramp
        inadimplencia[rec] += 1.6 * ramp
    # Pandemia: choque abrupto e curto no câmbio e na inadimplência.
    if cov.any():
        cambio[cov] += 8.0
        inadimplencia[cov] += 0.8
        desemprego[cov] += 1.2

    return pd.DataFrame(
        {
            "desemprego": desemprego,
            "renda": renda,
            "juros": juros,
            "cambio": cambio,
            "inadimplencia": inadimplencia,
        },
        index=idx,
    )


# ======================================================================
# Núcleo do DGP: ARDL em escala de link
# ======================================================================
def _simulate_link_ardl(
    macro: pd.DataFrame,
    intercept: float,
    ar: Sequence[float],
    betas: Sequence[tuple],
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simula a série do **preditor linear** ``eta`` (escala de link) por ARDL.

    ``eta_t = c + Σ_i φ_i · eta_{t-i} + Σ_j β_j · x_{j, t-lag_j} + σ·ε_t``.

    ``betas`` é uma lista de tuplas ``(variavel, defasagem, coef)``. As primeiras
    ``max(len(ar), max_lag)`` observações usam a média incondicional como semente
    e ``NaN`` nos regressores defasados indisponíveis é tratado como 0 (burn-in).
    """
    n = len(macro)
    ar = np.asarray(ar, dtype=float)
    p = len(ar)
    # Contribuição macro por período (soma das variáveis defasadas).
    x_contrib = np.zeros(n)
    for var, lag, coef in betas:
        col = macro[var].shift(lag).to_numpy(dtype=float)
        col = np.nan_to_num(col, nan=np.nanmean(macro[var].to_numpy(dtype=float)))
        x_contrib += coef * col
    denom = 1.0 - ar.sum() if p else 1.0
    mean_eta = (intercept + x_contrib.mean()) / (denom if abs(denom) > 1e-6 else 1e-6)
    eta = np.full(n, mean_eta)
    innov = rng.normal(0.0, sigma, n)
    for t in range(n):
        ar_term = 0.0
        for i in range(1, p + 1):
            ar_term += ar[i - 1] * (eta[t - i] if t - i >= 0 else mean_eta)
        eta[t] = intercept + ar_term + x_contrib[t] + innov[t]
    return eta


def _observe_binomial(rate: np.ndarray, n_base, rng: np.random.Generator) -> np.ndarray:
    """Ruído de observação de PD: taxa = defaults/base, com defaults ~ Binomial."""
    n_base = np.asarray(n_base)
    counts = rng.binomial(np.round(n_base).astype(int), np.clip(rate, 0, 1))
    return counts / n_base


def _observe_beta(rate: np.ndarray, precision: float, rng: np.random.Generator) -> np.ndarray:
    """Ruído de observação de LGD/CCF: severidade média ~ Beta(μ·φ, (1−μ)·φ)."""
    mu = np.clip(rate, 1e-4, 1 - 1e-4)
    a = mu * precision
    b = (1.0 - mu) * precision
    return rng.beta(a, b)


# ======================================================================
# Geradores por tipo de parâmetro
# ======================================================================
def simulate_pd_series(
    macro: Optional[pd.DataFrame] = None,
    *,
    link: str = "logit",
    intercept: Optional[float] = None,
    ar: Sequence[float] = (0.5,),
    betas: Sequence[tuple] = (("desemprego", 1, 0.18), ("renda", 0, -0.06)),
    sigma: float = 0.15,
    pd_ttc: float = 0.04,
    rho: float = 0.12,
    n_obligors: Optional[int] = 50_000,
    segment: str = "pd_sintetico",
    seed: int = 0,
    **macro_kwargs,
) -> SyntheticSeries:
    """Gera uma série sintética de **taxa de *default*** de DGP conhecido.

    Dois DGPs, pelo ``link``:

    * ``"logit"`` / ``"probit"`` — ARDL na escala do *link*:
      ``link(DR_t) = c + Σφ·link(DR_{t-i}) + Σβ·macro``. O intercepto ``c``, se
      ``None``, é calibrado para a série orbitar ``pd_ttc``.
    * ``"vasicek"`` — o fator sistêmico ``Z`` segue o ARDL macro e a taxa vem de
      :func:`~yggdrasil.credit_risk.econometric.transforms.default_rate_from_z`
      com ``pd_ttc`` e ``rho``. É o DGP que casa com o modelo :class:`VasicekZ` e
      com o motor de capital. **Atenção ao sinal**: neste ramo os ``betas`` agem
      sobre ``Z`` (o fator de *saúde* sistêmica, alto = ciclo benigno), então têm
      sinal **oposto** à convenção logit — ex.: para o desemprego **elevar** a PD,
      passe um coeficiente **negativo** (os ``betas`` padrão são calibrados para o
      ramo logit; troque-os ao usar ``link="vasicek"``).

    Sobre a taxa "verdadeira" aplica-se o **ruído de observação binomial** (a taxa
    é ``defaults/base`` com ``n_obligors`` contratos em risco), a menos que
    ``n_obligors=None``. ``betas`` é uma lista de ``(variavel, defasagem, coef)``.

    Returns
    -------
    SyntheticSeries
        Com ``series`` (observada), ``macro``, ``latent`` (taxa verdadeira) e
        ``truth`` (``intercept``, ``ar``, ``betas``, ``sigma``, ``link``, e para
        Vasicek ``pd_ttc``/``rho``).
    """
    rng = np.random.default_rng(seed)
    if macro is None:
        macro = simulate_macro(seed=seed, **macro_kwargs)

    if link == "vasicek":
        if intercept is None:
            # centra o fator Z em ~0: mean_eta = (c + x̄)/(1−Σφ) = 0 ⇒ c = −x̄,
            # de modo que a taxa orbite pd_ttc (e não deriva a extremos).
            c = -sum(coef * float(macro[var].mean()) for var, lag, coef in betas)
        else:
            c = intercept
        z = _simulate_link_ardl(macro, c, ar, betas, sigma, rng)
        latent = _tf.default_rate_from_z(z, pd_ttc, rho)
        latent = np.clip(latent, 1e-5, 1 - 1e-5)
    else:
        link_obj = _tf.get_link(link)
        if intercept is None:
            # calibra o intercepto para a média incondicional bater com pd_ttc
            ar_arr = np.asarray(ar, dtype=float)
            x_mean = 0.0
            for var, lag, coef in betas:
                x_mean += coef * float(macro[var].mean())
            intercept = float(link_obj.forward(np.array([pd_ttc]))[0]) * (1 - ar_arr.sum()) - x_mean
        eta = _simulate_link_ardl(macro, intercept, ar, betas, sigma, rng)
        latent = np.asarray(link_obj.inverse(eta), dtype=float)
        latent = np.clip(latent, 1e-5, 1 - 1e-5)

    if n_obligors is not None:
        observed = _observe_binomial(latent, n_obligors, rng)
    else:
        observed = latent.copy()

    idx = macro.index
    latent_s = pd.Series(latent, index=idx, name="dr_latente")
    obs_s = pd.Series(np.clip(observed, 0.0, 1.0), index=idx, name="dr")
    truth = {
        "kind": "pd", "link": link, "intercept": intercept, "ar": tuple(ar),
        "betas": {(v, l): c for v, l, c in betas}, "sigma": sigma,
    }
    if link == "vasicek":
        truth.update({"pd_ttc": pd_ttc, "rho": rho})
    rs = RiskSeries(values=obs_s, kind="pd", segment=segment, metadata={"synthetic": True})
    return SyntheticSeries(series=rs, macro=macro, truth=truth, latent=latent_s)


def simulate_lgd_series(
    macro: Optional[pd.DataFrame] = None,
    *,
    intercept: Optional[float] = None,
    ar: Sequence[float] = (0.4,),
    betas: Sequence[tuple] = (("desemprego", 0, 0.10), ("cambio", 3, 0.02)),
    sigma: float = 0.12,
    lgd_mean: float = 0.35,
    precision: float = 400.0,
    segment: str = "lgd_sintetico",
    seed: int = 1,
    **macro_kwargs,
) -> SyntheticSeries:
    """Gera uma série sintética de **LGD** média de DGP logit-ARDL conhecido.

    ``logit(LGD_t) = c + Σφ·logit(LGD_{t-i}) + Σβ·macro``, com a severidade
    subindo no *downturn* (correlação adversa PD–LGD via, por ex., ``desemprego``
    positivo). O ruído de observação é **beta** com precisão ``precision`` (a LGD
    média de um período vem de poucas resoluções, logo é ruidosa) — o DGP que casa
    com :class:`BetaRegression` e :class:`FractionalLogit`.
    """
    rng = np.random.default_rng(seed)
    if macro is None:
        macro = simulate_macro(seed=seed, **macro_kwargs)
    link_obj = _tf.get_link("logit")
    ar_arr = np.asarray(ar, dtype=float)
    if intercept is None:
        x_mean = sum(coef * float(macro[var].mean()) for var, lag, coef in betas)
        intercept = float(link_obj.forward(np.array([lgd_mean]))[0]) * (1 - ar_arr.sum()) - x_mean
    eta = _simulate_link_ardl(macro, intercept, ar, betas, sigma, rng)
    latent = np.clip(np.asarray(link_obj.inverse(eta), dtype=float), 1e-4, 1 - 1e-4)
    observed = _observe_beta(latent, precision, rng)

    idx = macro.index
    obs_s = pd.Series(np.clip(observed, 0.0, 1.0), index=idx, name="lgd")
    truth = {
        "kind": "lgd", "link": "logit", "intercept": intercept, "ar": tuple(ar),
        "betas": {(v, l): c for v, l, c in betas}, "sigma": sigma, "precision": precision,
    }
    rs = RiskSeries(values=obs_s, kind="lgd", segment=segment, metadata={"synthetic": True})
    return SyntheticSeries(series=rs, macro=macro, truth=truth,
                           latent=pd.Series(latent, index=idx, name="lgd_latente"))


def simulate_ccf_series(
    macro: Optional[pd.DataFrame] = None,
    *,
    intercept: Optional[float] = None,
    ar: Sequence[float] = (0.6,),
    betas: Sequence[tuple] = (("juros", 2, 0.05), ("desemprego", 1, 0.06)),
    sigma: float = 0.10,
    ccf_mean: float = 0.55,
    precision: float = 300.0,
    segment: str = "ccf_sintetico",
    seed: int = 2,
    **macro_kwargs,
) -> SyntheticSeries:
    """Gera uma série sintética de **CCF/LEQ** (utilização até o *default*).

    Mesmo DGP logit-ARDL da LGD (o CCF também é uma fração em ``[0, 1]``), mas com
    persistência mais alta e dirigido por juros/desemprego: o cliente à beira do
    *default* **saca mais** o rotativo quando o custo de crédito sobe (β positivo).
    Fecha a tríade PD/LGD/CCF pedida para os modelos satélite.
    """
    out = simulate_lgd_series(
        macro=macro, intercept=intercept, ar=ar, betas=betas, sigma=sigma,
        lgd_mean=ccf_mean, precision=precision, segment=segment, seed=seed, **macro_kwargs,
    )
    # Reetiqueta como CCF (o DGP é idêntico ao da LGD, um parâmetro em [0,1]).
    out.series.kind = "ccf"
    out.latent.name = "ccf_latente"
    out.series.values.name = "ccf"
    out.truth["kind"] = "ccf"
    return out


def make_reference_study(
    n_periods: int = 120,
    start: str = "2015-01-01",
    freq: str = "MS",
    seed: int = 7,
    segment: str = "segmento_referencia",
) -> ReferenceStudy:
    """Monta o **estudo de referência** de ponta a ponta (Guia §7).

    Uma macro única (com recessão e pandemia) dirigindo PD (logit-ARDL), LGD e CCF
    (logit-ARDL beta), do mesmo segmento e horizonte — o *fio condutor* que serve
    de teste de integração vivo e de demonstração. As três séries compartilham a
    macro, então cenários e projeção são coerentes entre parâmetros.
    """
    macro = simulate_macro(n_periods=n_periods, start=start, freq=freq, seed=seed)
    events = {"recessao": ("2016-01", "2017-06"), "covid": ("2020-03", "2020-08")}
    pd_s = simulate_pd_series(macro=macro, segment=f"{segment}_pd", seed=seed)
    lgd_s = simulate_lgd_series(macro=macro, segment=f"{segment}_lgd", seed=seed + 1)
    ccf_s = simulate_ccf_series(macro=macro, segment=f"{segment}_ccf", seed=seed + 2)
    return ReferenceStudy(macro=macro, pd=pd_s, lgd=lgd_s, ccf=ccf_s,
                          events=events, segment=segment)


__all__ = [
    "RISK_KINDS",
    "DEFAULT_LINK",
    "RiskSeries",
    "as_risk_series",
    "SyntheticSeries",
    "ReferenceStudy",
    "simulate_macro",
    "simulate_pd_series",
    "simulate_lgd_series",
    "simulate_ccf_series",
    "make_reference_study",
]
