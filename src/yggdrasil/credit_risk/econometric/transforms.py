"""
Transformações e engenharia de variáveis (Guia §2.2, Tabela 2 — ``transforms``)
===============================================================================
Taxas de *default* e severidades vivem em ``(0, 1)``: modelá-las **em nível**
viola as hipóteses dos modelos lineares (previsões saem do intervalo, a variância
não é constante). Este módulo implementa as transformações padrão do guia e a
engenharia de defasagens/dummies que alimenta os modelos satélite:

Ligações (*links*) — levam a taxa para a reta real e a trazem de volta
    * **logit** ``ln(p/(1−p))`` — a transformação canônica; estabiliza a variância.
    * **probit** ``N⁻¹(p)`` — a inversa da normal padrão, base do arcabouço de
      Vasicek.
    * **identidade** — para séries já em escala real (ex.: a série ``Z``).

Fator ``Z`` de Vasicek (Guia §2.2 e §3.4)
    :func:`vasicek_z` inverte a fórmula de Vasicek e extrai da série de taxas de
    *default* o **fator sistêmico latente** ``Z`` (aproximadamente ``N(0,1)`` e
    estacionário por construção); :func:`default_rate_from_z` faz o caminho de
    volta. É a ponte formal com o motor de capital econômico
    (:mod:`yggdrasil.credit_risk.capital`), que usa a **mesma** estrutura de
    Vasicek — aqui, ``Z`` positivo = ciclo **benigno** (menos *default*).

Engenharia de séries
    :func:`lags`/:func:`make_lags` (defasagens), :func:`difference` (integração),
    :func:`seasonal_dummies`, :func:`event_dummies`, :func:`step_dummy`
    (quebra/evento — Guia §2.3) e :func:`standardize`.

Tudo opera sobre :class:`pandas.Series`/:class:`~pandas.DataFrame` preservando o
índice temporal, para que defasagens e junções com macro fiquem alinhadas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

# Recorte padrão para tirar a taxa das bordas 0/1 antes de aplicar logit/probit
# (a transformação diverge em 0 e 1). 1e-6 corresponde a ~±4,75 desvios no probit.
DEFAULT_EPS = 1e-6


# ======================================================================
# Ligações (links) genéricas: (0,1) → reta real e volta
# ======================================================================
def _clip01(p: np.ndarray, eps: float) -> np.ndarray:
    if not (0.0 < eps < 0.5):
        raise ValueError(f"eps deve estar em (0, 0.5); recebido {eps!r}.")
    return np.clip(p, eps, 1.0 - eps)


def logit(p, eps: float = DEFAULT_EPS):
    """Transformação **logit** ``ln(p/(1−p))`` de uma taxa em ``(0, 1)``.

    Recorta ``p`` em ``[eps, 1−eps]`` antes de transformar, para não divergir nas
    bordas. Aceita escalar, ``ndarray`` ou :class:`pandas.Series` (preserva o
    índice).
    """
    idx = p.index if isinstance(p, pd.Series) else None
    arr = _clip01(np.asarray(p, dtype=float), eps)
    out = np.log(arr / (1.0 - arr))
    return pd.Series(out, index=idx) if idx is not None else (float(out) if np.ndim(out) == 0 else out)


def inv_logit(x):
    """Inversa do logit (função logística) ``1/(1+e^{−x})`` — de volta a ``(0,1)``."""
    idx = x.index if isinstance(x, pd.Series) else None
    arr = np.asarray(x, dtype=float)
    # Forma numericamente estável (evita overflow de exp para x muito negativo).
    out = np.where(arr >= 0, 1.0 / (1.0 + np.exp(-arr)), np.exp(arr) / (1.0 + np.exp(arr)))
    return pd.Series(out, index=idx) if idx is not None else (float(out) if np.ndim(out) == 0 else out)


def probit(p, eps: float = DEFAULT_EPS):
    """Transformação **probit** ``N⁻¹(p)`` (inversa da normal padrão)."""
    idx = p.index if isinstance(p, pd.Series) else None
    arr = _clip01(np.asarray(p, dtype=float), eps)
    out = norm.ppf(arr)
    return pd.Series(out, index=idx) if idx is not None else (float(out) if np.ndim(out) == 0 else out)


def inv_probit(x):
    """Inversa do probit ``N(x)`` (normal padrão acumulada) — de volta a ``(0,1)``."""
    idx = x.index if isinstance(x, pd.Series) else None
    out = norm.cdf(np.asarray(x, dtype=float))
    return pd.Series(out, index=idx) if idx is not None else (float(out) if np.ndim(out) == 0 else out)


@dataclass(frozen=True)
class Link:
    """Uma ligação nomeada com sua inversa: ``forward`` mapeia ``(0,1)→ℝ``."""

    name: str
    forward: Callable
    inverse: Callable


#: Ligações genéricas disponíveis por nome (a ``vasicek_z`` é paramétrica e vive
#: nas funções dedicadas abaixo, pois depende de ``pd_ttc`` e ``rho``).
LINKS: dict[str, Link] = {
    "logit": Link("logit", logit, inv_logit),
    "probit": Link("probit", probit, inv_probit),
    "identity": Link("identity", lambda s: s, lambda s: s),
}


def get_link(name: str) -> Link:
    """Recupera uma :class:`Link` por nome (``'logit'``, ``'probit'``, ``'identity'``)."""
    try:
        return LINKS[name]
    except KeyError:
        raise ValueError(
            f"link desconhecido: {name!r}. Válidos: {', '.join(sorted(LINKS))} "
            f"(para PD via fator sistêmico, use o modelo VasicekZ)."
        ) from None


# ======================================================================
# Fator Z de Vasicek (Guia §2.2, §3.4) — ponte com o capital econômico
# ======================================================================
def vasicek_z(default_rate, pd_ttc: float, rho: float, eps: float = DEFAULT_EPS):
    """Extrai o **fator sistêmico latente ``Z``** de uma série de taxas de *default*.

    Inverte a fórmula de Vasicek (o mesmo modelo estrutural do motor de capital,
    :mod:`yggdrasil.credit_risk.capital.asrf`):

    ``Z = ( N⁻¹(PD_TTC) − √(1−ρ) · N⁻¹(DR) ) / √ρ``

    Por construção ``Z`` é aproximadamente ``N(0, 1)`` e estacionário, e é ele —
    não a taxa bruta — que se modela contra as variáveis macro (Guia §3.4). A
    **convenção de sinal** aqui é: ``Z`` **alto** = ciclo **benigno** (taxa de
    *default* **baixa**), consistente com o fator sistêmico do capital econômico,
    onde o cenário adverso é o fator caindo à cauda inferior.

    Parameters
    ----------
    default_rate:
        Série (ou array) de taxas de *default* observadas por período, em
        ``(0, 1)``. Recortada em ``[eps, 1−eps]`` antes da inversão.
    pd_ttc:
        PD *through-the-cycle* do segmento (o nível de longo prazo), em ``(0, 1)``.
        Tipicamente a média da própria série (ver
        :func:`yggdrasil.credit_risk.capital.pit_to_ttc`).
    rho:
        Correlação de ativos ``ρ`` do segmento, em ``(0, 1)`` (a mesma do capital).

    Returns
    -------
    pandas.Series | numpy.ndarray
        A série do fator ``Z`` (mesma forma/índice da entrada).
    """
    if not (0.0 < pd_ttc < 1.0):
        raise ValueError(f"pd_ttc deve estar em (0, 1); recebido {pd_ttc!r}.")
    if not (0.0 < rho < 1.0):
        raise ValueError(f"rho deve estar em (0, 1); recebido {rho!r}.")
    idx = default_rate.index if isinstance(default_rate, pd.Series) else None
    dr = _clip01(np.asarray(default_rate, dtype=float), eps)
    z = (norm.ppf(pd_ttc) - np.sqrt(1.0 - rho) * norm.ppf(dr)) / np.sqrt(rho)
    return pd.Series(z, index=idx) if idx is not None else z


def default_rate_from_z(z, pd_ttc: float, rho: float):
    """Reconverte o fator ``Z`` em taxa de *default* (inversa de :func:`vasicek_z`).

    ``DR = N( ( N⁻¹(PD_TTC) − √ρ · Z ) / √(1−ρ) )``.
    """
    if not (0.0 < pd_ttc < 1.0):
        raise ValueError(f"pd_ttc deve estar em (0, 1); recebido {pd_ttc!r}.")
    if not (0.0 < rho < 1.0):
        raise ValueError(f"rho deve estar em (0, 1); recebido {rho!r}.")
    idx = z.index if isinstance(z, pd.Series) else None
    zz = np.asarray(z, dtype=float)
    dr = norm.cdf((norm.ppf(pd_ttc) - np.sqrt(rho) * zz) / np.sqrt(1.0 - rho))
    return pd.Series(dr, index=idx) if idx is not None else dr


# ======================================================================
# Engenharia de séries: defasagens, diferenças, dummies, padronização
# ======================================================================
def lags(series: pd.Series, k: int) -> pd.Series:
    """Defasagem de ``k`` períodos (``series.shift(k)``), preservando o índice."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if k < 0:
        raise ValueError(f"k (defasagem) deve ser >= 0; recebido {k!r}.")
    out = series.shift(k)
    out.name = f"{series.name}_l{k}" if series.name is not None else f"l{k}"
    return out


def make_lags(
    data: pd.Series | pd.DataFrame,
    spec: int | Sequence[int] | Mapping[str, Sequence[int]],
) -> pd.DataFrame:
    """Monta a matriz de defasagens (Guia §3.1 — o "coração" do ARDL).

    Parameters
    ----------
    data:
        Uma :class:`~pandas.Series` (uma variável) ou :class:`~pandas.DataFrame`
        (várias). O índice temporal é preservado.
    spec:
        Como defasar. Três formas:

        * ``int`` ``p`` → defasagens ``1..p`` de **cada** coluna.
        * ``Sequence[int]`` → exatamente essas defasagens de cada coluna
          (ex.: ``[0, 3, 6]`` inclui o contemporâneo e as defasagens 3 e 6).
        * ``Mapping[str, Sequence[int]]`` → defasagens **por coluna**
          (ex.: ``{"desemprego": [0, 3], "juros": [6]}``).

    Returns
    -------
    pandas.DataFrame
        Colunas nomeadas ``{var}_l{k}`` (defasagem 0 mantém o nome ``{var}``).
        **Não** remove as linhas iniciais com ``NaN`` — use :func:`align` para
        alinhar com a variável dependente.
    """
    if isinstance(data, pd.Series):
        name = data.name if data.name is not None else "x"
        data = data.to_frame(name)

    def _lag_list(col: str) -> list[int]:
        if isinstance(spec, Mapping):
            return list(spec.get(col, []))
        if isinstance(spec, int):
            return list(range(1, spec + 1))
        return list(spec)

    cols: dict[str, pd.Series] = {}
    for col in data.columns:
        for k in _lag_list(col):
            if k < 0:
                raise ValueError(f"defasagem negativa não é válida: {k!r}.")
            key = str(col) if k == 0 else f"{col}_l{k}"
            cols[key] = data[col].shift(k)
    return pd.DataFrame(cols, index=data.index)


def align(y: pd.Series, X: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Alinha ``y`` e ``X`` e remove as linhas com ``NaN`` (bordas das defasagens).

    Retorna ``(y, X)`` restritos ao índice comum sem faltantes — o par pronto para
    a estimação.
    """
    df = X.copy()
    df["__y__"] = y
    df = df.dropna()
    y_clean = df.pop("__y__")
    return y_clean, df


def difference(series: pd.Series, d: int = 1) -> pd.Series:
    """Diferença de ordem ``d`` (para séries integradas — Guia §2.3)."""
    if d < 0:
        raise ValueError(f"d deve ser >= 0; recebido {d!r}.")
    out = series.copy()
    for _ in range(d):
        out = out.diff()
    return out


def _period_of(index: pd.Index) -> np.ndarray:
    """Número do período sazonal (mês 1..12 ou trimestre 1..4) do índice."""
    if isinstance(index, pd.PeriodIndex):
        return np.asarray(index.month if index.freqstr and index.freqstr.startswith(("M",)) else index.quarter)
    if isinstance(index, pd.DatetimeIndex):
        return np.asarray(index.month)
    raise TypeError("índice deve ser DatetimeIndex ou PeriodIndex para dummies sazonais.")


def seasonal_dummies(index: pd.Index, period: int = 12, drop_first: bool = True) -> pd.DataFrame:
    """Dummies sazonais (Guia §2.3): uma coluna por estação, ``drop_first`` evita
    colinearidade perfeita com a constante.

    ``period=12`` usa o **mês**; ``period=4`` usa o **trimestre**. As colunas saem
    ``s02..s12`` (ou ``s2..s4``), 0/1.
    """
    if period == 12:
        num = pd.DatetimeIndex(index).month if not isinstance(index, pd.PeriodIndex) else index.month
    elif period == 4:
        num = pd.DatetimeIndex(index).quarter if not isinstance(index, pd.PeriodIndex) else index.quarter
    else:
        raise ValueError("period deve ser 12 (mensal) ou 4 (trimestral).")
    num = np.asarray(num)
    start = 2 if drop_first else 1
    cols = {f"s{p:02d}": (num == p).astype(float) for p in range(start, period + 1)}
    return pd.DataFrame(cols, index=index)


def event_dummies(index: pd.Index, events: Mapping[str, object]) -> pd.DataFrame:
    """Dummies de **evento** (Guia §2.3 — a pandemia é o caso canônico).

    Cada evento vira uma coluna 0/1 marcando os períodos afetados. O valor de cada
    entrada em ``events`` pode ser:

    * uma data única (marca só aquele período);
    * uma lista de datas;
    * uma tupla ``(inicio, fim)`` marcando o intervalo fechado (janela do evento,
      ex.: ``("2020-03", "2020-12")`` para o auge da pandemia).
    """
    idx = pd.DatetimeIndex(index.to_timestamp()) if isinstance(index, pd.PeriodIndex) else pd.DatetimeIndex(index)
    cols: dict[str, np.ndarray] = {}
    for name, spec in events.items():
        mark = np.zeros(len(idx), dtype=float)
        if isinstance(spec, tuple) and len(spec) == 2:
            ini, fim = pd.Timestamp(spec[0]), pd.Timestamp(spec[1])
            mark[(idx >= ini) & (idx <= fim)] = 1.0
        else:
            datas = spec if isinstance(spec, (list, tuple)) else [spec]
            for d in datas:
                d = pd.Timestamp(d)
                # marca o período cujo carimbo casa no nível da frequência do índice
                mark[idx == d] = 1.0
                if not (idx == d).any():  # tolera datas dentro do período (mês)
                    mark[(idx.year == d.year) & (idx.month == d.month)] = 1.0
        cols[name] = mark
    return pd.DataFrame(cols, index=index)


def step_dummy(index: pd.Index, date, name: str = "quebra") -> pd.Series:
    """Dummy de **degrau** (0 antes, 1 a partir de ``date``) — quebra estrutural
    de nível (mudança de definição/política, Guia §2.3)."""
    idx = pd.DatetimeIndex(index.to_timestamp()) if isinstance(index, pd.PeriodIndex) else pd.DatetimeIndex(index)
    d = pd.Timestamp(date)
    return pd.Series((idx >= d).astype(float), index=index, name=name)


@dataclass
class Standardizer:
    """Padroniza uma série (``z = (x−μ)/σ``) e guarda ``μ, σ`` para reverter.

    Útil para deixar coeficientes macro comparáveis e melhorar o condicionamento
    numérico; ``inverse`` recompõe a escala original nas projeções.
    """

    mean: float
    std: float

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.mean) / self.std

    def inverse(self, z):
        return np.asarray(z, dtype=float) * self.std + self.mean


def standardize(series: pd.Series) -> tuple[pd.Series, Standardizer]:
    """Padroniza ``series`` e devolve ``(série_padronizada, Standardizer)``."""
    mu = float(np.nanmean(series))
    sd = float(np.nanstd(series, ddof=1))
    if sd == 0.0:
        raise ValueError("série constante não pode ser padronizada (desvio zero).")
    st = Standardizer(mean=mu, std=sd)
    out = (series - mu) / sd if isinstance(series, pd.Series) else st.transform(series)
    return out, st


__all__ = [
    "DEFAULT_EPS",
    "logit",
    "inv_logit",
    "probit",
    "inv_probit",
    "Link",
    "LINKS",
    "get_link",
    "vasicek_z",
    "default_rate_from_z",
    "lags",
    "make_lags",
    "align",
    "difference",
    "seasonal_dummies",
    "event_dummies",
    "step_dummy",
    "Standardizer",
    "standardize",
]
