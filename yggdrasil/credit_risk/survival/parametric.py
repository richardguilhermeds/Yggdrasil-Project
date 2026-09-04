"""
Sobrevivência paramétrica e a extrapolação da cauda
===================================================
A curva empírica (safra ou Kaplan-Meier) só vai até onde a carteira foi
observada. Um contrato de 60 meses numa carteira com 30 meses de histórico
precisa de **30 meses de curva que os dados não mostram**. A extensão plana
(repetir o último *hazard*) é a hipótese mínima; a **alternativa defensável** é
ajustar uma família paramétrica ao trecho observado e deixá-la dizer o que a
maturação faz depois.

:class:`ParametricSurvival`
    Ajusta por máxima verossimilhança uma distribuição de tempo até o *default*
    (exponencial, **Weibull**, log-normal, log-logística ou Gompertz) sobre a
    tabela de vida agregada, respeitando a censura à direita e a truncagem à
    esquerda pela própria construção em tempo discreto: cada idade contribui
    com ``d_t`` quebras em ``n_t`` em risco, e o *hazard* discreto da família é
    ``h_t = 1 − S(t+1)/S(t)``. Entrega AIC/BIC para comparar famílias e a
    leitura do formato (crescente, decrescente, em corcova).

:func:`fit_parametric`
    Ajusta várias famílias e devolve o ranking por AIC: o gráfico da curva
    ajustada contra o KM é o que decide; o AIC é o desempate.

:func:`splice_curves`
    Emenda a curva empírica até a idade de junção (onde a base em risco ainda
    sustenta a taxa) com a cauda paramétrica dali em diante, opcionalmente
    casando o **nível** na junção para não haver degrau.

Tudo em ``numpy``/``scipy``; o resultado é sempre a mesma
:class:`~yggdrasil.credit_risk.ecl.curves.PDCurve` que o resto do pacote usa.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from ..ecl.curves import PDCurve
from ..ecl.panel import ContractPanel

#: Famílias suportadas.
DISTRIBUTIONS = ("exponential", "weibull", "lognormal", "loglogistic", "gompertz")

#: Rótulos em português para tabelas e interface.
DISTRIBUTION_LABELS = {
    "exponential": "Exponencial (hazard constante)",
    "weibull": "Weibull (monótono)",
    "lognormal": "Log-normal (corcova)",
    "loglogistic": "Log-logística (corcova)",
    "gompertz": "Gompertz (exponencial na idade)",
}

_EPS = 1e-10


def _life_counts(source, max_age: Optional[int] = None, min_at_risk: int = 1
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(idades, n_em_risco, n_default)`` de um painel ou de uma tabela de vida."""
    if isinstance(source, ContractPanel):
        tab = source.at_risk(max_age=max_age)
        idades = tab.index.to_numpy(dtype=int)
    elif isinstance(source, pd.DataFrame):
        tab = source
        if "idade" in tab.columns:
            idades = tab["idade"].to_numpy(dtype=int)
        else:
            idades = np.asarray(tab.index, dtype=int)
        if max_age is not None:
            manter = idades <= int(max_age)
            tab, idades = tab[manter], idades[manter]
    else:
        raise TypeError("source deve ser um ContractPanel ou a tabela de vida (DataFrame).")
    faltando = [c for c in ("n_em_risco", "n_default") if c not in tab.columns]
    if faltando:
        raise ValueError(f"a tabela de vida não tem as colunas {faltando}.")
    n = tab["n_em_risco"].to_numpy(dtype=float)
    d = tab["n_default"].to_numpy(dtype=float)
    ok = n >= float(max(min_at_risk, 1))
    if not ok.any():
        raise ValueError("nenhuma idade com base em risco suficiente para o ajuste.")
    return idades[ok], n[ok], d[ok]


# ======================================================================
# As famílias (função de sobrevivência contínua em parâmetros naturais)
# ======================================================================
def _survival(dist: str, x: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if dist == "exponential":
        return np.exp(-p["rate"] * x)
    if dist == "weibull":
        return np.exp(-(x / p["scale"]) ** p["shape"])
    if dist == "lognormal":
        with np.errstate(divide="ignore"):
            z = (np.log(np.maximum(x, _EPS)) - p["mu"]) / p["sigma"]
        return np.where(x <= 0, 1.0, norm.sf(z))
    if dist == "loglogistic":
        return 1.0 / (1.0 + (x / p["scale"]) ** p["shape"])
    if dist == "gompertz":
        c = p["c"]
        if abs(c) < 1e-9:
            return np.exp(-p["b"] * x)
        return np.exp(-(p["b"] / c) * (np.exp(c * x) - 1.0))
    raise ValueError(f"distribution deve ser uma de {DISTRIBUTIONS}; recebido {dist!r}.")


def _unpack(dist: str, theta: np.ndarray) -> Dict[str, float]:
    """Parâmetros irrestritos do otimizador → escala natural."""
    if dist == "exponential":
        return {"rate": float(np.exp(theta[0]))}
    if dist == "weibull":
        return {"scale": float(np.exp(theta[0])), "shape": float(np.exp(theta[1]))}
    if dist == "lognormal":
        return {"mu": float(theta[0]), "sigma": float(np.exp(theta[1]))}
    if dist == "loglogistic":
        return {"scale": float(np.exp(theta[0])), "shape": float(np.exp(theta[1]))}
    if dist == "gompertz":
        return {"b": float(np.exp(theta[0])), "c": float(theta[1])}
    raise ValueError(f"distribution deve ser uma de {DISTRIBUTIONS}; recebido {dist!r}.")


def _theta0(dist: str, h_medio: float) -> np.ndarray:
    lam = float(np.clip(h_medio, 1e-6, 0.5))
    if dist == "exponential":
        return np.array([np.log(lam)])
    if dist in ("weibull", "loglogistic"):
        return np.array([np.log(1.0 / lam), 0.0])
    if dist == "lognormal":
        return np.array([np.log(1.0 / lam), 0.0])
    return np.array([np.log(lam), 0.01])         # gompertz


def _discrete_hazard(dist: str, ages: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    a = np.asarray(ages, dtype=float)
    s0 = _survival(dist, a, p)
    s1 = _survival(dist, a + 1.0, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(s0 > _EPS, 1.0 - s1 / s0, 1.0)
    return np.clip(np.nan_to_num(h, nan=1.0), 0.0, 1.0)


# ======================================================================
# O modelo
# ======================================================================
class ParametricSurvival:
    """Distribuição paramétrica do tempo até o *default*, em tempo discreto.

    Parameters
    ----------
    distribution:
        Uma de :data:`DISTRIBUTIONS`.

    Attributes
    ----------
    params_:
        Parâmetros na escala natural (``rate``; ``scale``/``shape``;
        ``mu``/``sigma``; ``b``/``c``).
    loglik_, aic_, bic_:
        Verossimilhança maximizada e critérios de informação (``n`` = pessoa-
        períodos observados).
    n_obs_, n_events_:
        Pessoa-períodos e quebras usados no ajuste.
    converged_:
        Se o otimizador convergiu.

    Examples
    --------
    >>> ps = ParametricSurvival("weibull").fit(painel)
    >>> ps.params_["shape"] > 1      # hazard crescente: maturação
    >>> ps.curve(horizon=60).pd_lifetime()
    """

    def __init__(self, distribution: str = "weibull") -> None:
        if distribution not in DISTRIBUTIONS:
            raise ValueError(
                f"distribution deve ser uma de {DISTRIBUTIONS}; recebido {distribution!r}.")
        self.distribution = distribution
        self.params_: Dict[str, float] = {}
        self.loglik_: float = float("nan")
        self.aic_: float = float("nan")
        self.bic_: float = float("nan")
        self.n_obs_: int = 0
        self.n_events_: int = 0
        self.converged_: bool = False
        self.freq: str = "M"
        self.ages_: np.ndarray = np.array([])

    @property
    def n_params(self) -> int:
        return 1 if self.distribution == "exponential" else 2

    # -- ajuste -------------------------------------------------------------
    def _nll(self, theta: np.ndarray, ages: np.ndarray, n: np.ndarray, d: np.ndarray) -> float:
        p = _unpack(self.distribution, theta)
        h = np.clip(_discrete_hazard(self.distribution, ages, p), _EPS, 1.0 - _EPS)
        ll = np.sum(d * np.log(h) + (n - d) * np.log(1.0 - h))
        return float(-ll) if np.isfinite(ll) else 1e30

    def fit(self, source: Union[ContractPanel, pd.DataFrame], max_age: Optional[int] = None,
            min_at_risk: int = 1) -> "ParametricSurvival":
        """Ajusta por máxima verossimilhança sobre a tabela de vida.

        ``source`` é o :class:`~yggdrasil.credit_risk.ecl.panel.ContractPanel`
        ou a tabela de vida (``n_em_risco``/``n_default`` por idade). Idades com
        base abaixo de ``min_at_risk`` ficam de fora do ajuste (não da curva)."""
        ages, n, d = _life_counts(source, max_age=max_age, min_at_risk=min_at_risk)
        if d.sum() <= 0:
            raise ValueError("a tabela não tem nenhum default; nada a ajustar.")
        if isinstance(source, ContractPanel):
            self.freq = source.freq
        h_medio = float(d.sum() / n.sum())
        theta0 = _theta0(self.distribution, h_medio)

        res = minimize(self._nll, theta0, args=(ages, n, d), method="L-BFGS-B")
        if not res.success or not np.isfinite(res.fun):
            alt = minimize(self._nll, theta0, args=(ages, n, d), method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10})
            if np.isfinite(alt.fun) and (not np.isfinite(res.fun) or alt.fun <= res.fun):
                res = alt
        self.params_ = _unpack(self.distribution, np.asarray(res.x, dtype=float))
        self.loglik_ = float(-res.fun)
        self.n_obs_ = int(round(n.sum()))
        self.n_events_ = int(round(d.sum()))
        k = self.n_params
        self.aic_ = 2.0 * k - 2.0 * self.loglik_
        self.bic_ = k * np.log(max(self.n_obs_, 1)) - 2.0 * self.loglik_
        self.converged_ = bool(res.success)
        self.ages_ = ages
        return self

    def _check_fit(self) -> None:
        if not self.params_:
            raise RuntimeError("o modelo ainda não foi ajustado; chame .fit(painel) antes.")

    # -- leitura --------------------------------------------------------------
    def survival(self, x) -> np.ndarray:
        """Função de sobrevivência **contínua** ``S(x)`` da família ajustada."""
        self._check_fit()
        return _survival(self.distribution, np.asarray(x, dtype=float), self.params_)

    def hazard(self, ages) -> np.ndarray:
        """*Hazard* **discreto** por idade: ``h_t = 1 − S(t+1)/S(t)``."""
        self._check_fit()
        return _discrete_hazard(self.distribution, np.asarray(ages), self.params_)

    def curve(self, horizon: int = 60, from_age: int = 0, label: str = "") -> PDCurve:
        """A :class:`PDCurve` da família: idades ``from_age .. from_age+horizon−1``."""
        self._check_fit()
        grade = np.arange(int(from_age), int(from_age) + int(horizon))
        return PDCurve(self.hazard(grade), label=label, freq=self.freq,
                       meta={"metodo": "parametric", "distribution": self.distribution,
                             "params": dict(self.params_), "from_age": int(from_age)})

    def shape_reading(self) -> str:
        """Leitura em texto do formato do *hazard* implicado pelos parâmetros."""
        self._check_fit()
        p = self.params_
        if self.distribution == "exponential":
            return "hazard constante na idade (sem maturação)"
        if self.distribution == "weibull":
            k = p["shape"]
            if k > 1.05:
                return f"hazard crescente com a idade (k = {k:.2f} > 1: maturação)"
            if k < 0.95:
                return f"hazard decrescente com a idade (k = {k:.2f} < 1: seleção/burn-out)"
            return f"hazard aproximadamente constante (k = {k:.2f})"
        if self.distribution == "lognormal":
            pico = float(np.exp(p["mu"] - p["sigma"] ** 2))
            return f"hazard em corcova (sobe e depois cai; moda perto de {pico:.0f} períodos)"
        if self.distribution == "loglogistic":
            b = p["shape"]
            if b > 1.0:
                pico = p["scale"] * ((b - 1.0) ** (1.0 / b))
                return f"hazard em corcova (β = {b:.2f} > 1; pico perto de {pico:.0f} períodos)"
            return f"hazard decrescente (β = {b:.2f} ≤ 1)"
        c = p["c"]
        if c > 1e-3:
            return f"hazard exponencial crescente na idade (c = {c:.3f} > 0)"
        if c < -1e-3:
            return f"hazard exponencial decrescente na idade (c = {c:.3f} < 0)"
        return "hazard aproximadamente constante (c ≈ 0)"

    def summary(self) -> pd.DataFrame:
        """Uma linha: família, parâmetros, log-verossimilhança, AIC, BIC e a leitura."""
        self._check_fit()
        linha = {"distribuicao": self.distribution,
                 "rotulo": DISTRIBUTION_LABELS[self.distribution],
                 "parametros": ", ".join(f"{k}={v:.4g}" for k, v in self.params_.items()),
                 "n_parametros": self.n_params, "loglik": self.loglik_,
                 "aic": self.aic_, "bic": self.bic_, "convergiu": self.converged_,
                 "leitura": self.shape_reading()}
        return pd.DataFrame([linha])

    def to_dict(self) -> dict:
        return {"distribution": self.distribution, "params": dict(self.params_),
                "loglik": self.loglik_, "aic": self.aic_, "bic": self.bic_,
                "n_obs": self.n_obs_, "n_events": self.n_events_, "freq": self.freq,
                "converged": self.converged_}

    @classmethod
    def from_dict(cls, d: dict) -> "ParametricSurvival":
        obj = cls(d["distribution"])
        obj.params_ = {k: float(v) for k, v in d.get("params", {}).items()}
        obj.loglik_ = float(d.get("loglik", np.nan))
        obj.aic_ = float(d.get("aic", np.nan))
        obj.bic_ = float(d.get("bic", np.nan))
        obj.n_obs_ = int(d.get("n_obs", 0))
        obj.n_events_ = int(d.get("n_events", 0))
        obj.freq = d.get("freq", "M")
        obj.converged_ = bool(d.get("converged", True))
        return obj

    def __repr__(self) -> str:
        if not self.params_:
            return f"ParametricSurvival({self.distribution!r}, não ajustado)"
        pars = ", ".join(f"{k}={v:.4g}" for k, v in self.params_.items())
        return f"ParametricSurvival({self.distribution!r}, {pars}, AIC={self.aic_:.1f})"


# ======================================================================
# Comparação de famílias
# ======================================================================
def fit_parametric(source: Union[ContractPanel, pd.DataFrame],
                   distributions: Sequence[str] = DISTRIBUTIONS,
                   max_age: Optional[int] = None, min_at_risk: int = 1
                   ) -> Tuple[pd.DataFrame, Dict[str, ParametricSurvival]]:
    """Ajusta várias famílias e devolve ``(ranking, modelos)``.

    O ranking é ordenado por **AIC** (menor melhor) e traz ``delta_aic`` em
    relação ao melhor: abaixo de 2 as famílias são indistinguíveis pelos dados,
    e a escolha deve cair na forma mais plausível para o produto."""
    modelos: Dict[str, ParametricSurvival] = {}
    linhas = []
    for dist in distributions:
        try:
            m = ParametricSurvival(dist).fit(source, max_age=max_age, min_at_risk=min_at_risk)
        except (ValueError, RuntimeError, FloatingPointError):
            continue
        modelos[dist] = m
        linhas.append(m.summary().iloc[0].to_dict())
    if not linhas:
        raise ValueError("nenhuma família pôde ser ajustada.")
    rank = pd.DataFrame(linhas).sort_values("aic").reset_index(drop=True)
    rank["delta_aic"] = rank["aic"] - rank["aic"].iloc[0]
    rank.insert(0, "posicao", np.arange(1, len(rank) + 1))
    return rank, modelos


# ======================================================================
# Emenda da cauda
# ======================================================================
def junction_age(source: Union[ContractPanel, pd.DataFrame], min_at_risk: int = 30) -> int:
    """Última idade cuja base em risco ainda é ``>= min_at_risk``.

    É a idade a partir da qual a curva empírica deixa de ser confiável e a
    cauda paramétrica assume. Devolve ``-1`` se nenhuma idade atinge o mínimo."""
    ages, n, _ = _life_counts(source, min_at_risk=1)
    ok = np.flatnonzero(n >= float(min_at_risk))
    return int(ages[ok.max()]) if ok.size else -1


def splice_curves(empirical: PDCurve, tail: Union[ParametricSurvival, PDCurve],
                  junction: int, horizon: int, match_level: bool = True,
                  window: int = 3, from_age: int = 0) -> PDCurve:
    """Emenda a curva **empírica** até ``junction`` com a **cauda** paramétrica.

    Parameters
    ----------
    empirical:
        A curva observada (safra ou KM), indexada por idade a partir de ``from_age``.
    tail:
        O modelo paramétrico ajustado (ou uma :class:`PDCurve` já na grade de
        idades ``from_age..``).
    junction:
        Idade (inclusive) até onde a empírica manda. Idades acima vêm da cauda.
    horizon:
        Nº de períodos da curva final.
    match_level:
        Casa o nível da cauda com a empírica na junção: multiplica os *hazards*
        paramétricos pela razão entre a média empírica e a média paramétrica
        nas últimas ``window`` idades antes da junção. Evita o degrau na emenda
        sem mudar o **formato** que a família traz.
    window:
        Nº de idades usadas para casar o nível.

    Returns
    -------
    PDCurve
        ``meta`` registra ``junction``, a família e o fator de nível.
    """
    H = int(horizon)
    if H < 1:
        raise ValueError(f"horizon deve ser >= 1; recebido {horizon!r}.")
    grade = np.arange(int(from_age), int(from_age) + H)
    if isinstance(tail, ParametricSurvival):
        h_tail = tail.hazard(grade)
        familia = tail.distribution
    else:
        h_tail = tail.extend(H).hazard_[:H]
        familia = tail.meta.get("distribution", "curva")
    h_emp = empirical.hazard_
    n_emp = int(min(len(h_emp), max(int(junction) - int(from_age) + 1, 0), H))

    fator = 1.0
    if match_level and n_emp > 0:
        w = int(max(min(window, n_emp), 1))
        num = float(np.mean(h_emp[n_emp - w: n_emp]))
        den = float(np.mean(h_tail[n_emp - w: n_emp]))
        if den > _EPS and num > _EPS:
            fator = num / den
    h = np.concatenate([h_emp[:n_emp], np.clip(h_tail[n_emp:] * fator, 0.0, 1.0)])
    return PDCurve(np.clip(h, 0.0, 1.0), label=empirical.label, freq=empirical.freq,
                   meta={**empirical.meta, "cauda": familia, "junction": int(junction),
                         "fator_nivel": float(fator), "n_empirico": n_emp,
                         "extrapolada_apos": n_emp})


__all__ = ["ParametricSurvival", "fit_parametric", "splice_curves", "junction_age",
           "DISTRIBUTIONS", "DISTRIBUTION_LABELS"]
