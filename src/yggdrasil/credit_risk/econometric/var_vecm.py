"""
VAR, VECM e cointegração (Guia §3.3, Tabela 1) — equilíbrio de longo prazo
==========================================================================
Quando a relação entre a série de risco e as variáveis macro é de **equilíbrio de
longo prazo entre séries integradas** (por exemplo, inadimplência e desemprego
caminhando juntos em nível), a técnica correta é a **cointegração**: teste de
Engle-Granger ou Johansen, e um modelo de **correção de erros (VECM)** em que a
série se ajusta ao desvio do equilíbrio (§3.3). O **VAR** trata todas as variáveis
como endógenas e é útil para (i) gerar **cenários internamente consistentes** e
(ii) **funções de impulso-resposta** (quanto a PD responde a um choque de juros ao
longo dos trimestres).

Custo: consomem muitos parâmetros; em séries curtas, restringir a poucas
variáveis (§3.3). Motor: :mod:`statsmodels.tsa` (VAR, VECM, Johansen).

Diferente dos modelos uni-equação, aqui a saída é o **sistema**: as classes
:class:`VARModel`/:class:`VECMModel` expõem ``fit``/``forecast``/``irf`` sobre o
quadro conjunto (risco transformado + macro), e não a interface
``SatelliteModel`` de projeção condicional.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .series import RiskSeries, as_risk_series
from .transforms import get_link


def joint_frame(series, macro: pd.DataFrame, link: Optional[str] = None,
                variables: Optional[Sequence[str]] = None) -> tuple[pd.DataFrame, str]:
    """Monta o quadro conjunto ``[risco_transformado, macro...]`` alinhado (dropna).

    Devolve ``(DataFrame, nome_da_coluna_de_risco)``. O risco entra na escala do
    *link* (logit por padrão), pois VAR/VECM assumem variáveis contínuas na reta.
    """
    rs: RiskSeries = as_risk_series(series)
    lk = get_link(link or rs.default_link)
    risk_col = f"{rs.kind}_link"
    cols = {risk_col: lk.forward(rs.values)}
    use = list(variables) if variables else list(macro.columns)
    for c in use:
        cols[c] = macro[c]
    df = pd.DataFrame(cols).dropna()
    return df, risk_col


# ======================================================================
# Testes de cointegração (Guia §3.3)
# ======================================================================
@dataclass
class CointegrationResult:
    method: str
    statistic: float
    pvalue: Optional[float]
    rank: Optional[int]
    detail: dict

    def __repr__(self) -> str:  # pragma: no cover
        p = f"{self.pvalue:.4f}" if self.pvalue is not None else "—"
        return f"CointegrationResult({self.method!r}, stat={self.statistic:.3f}, p={p}, rank={self.rank})"


def engle_granger(y, x, trend: str = "c") -> CointegrationResult:
    """**Engle-Granger**: testa cointegração entre ``y`` e ``x`` (H0: **não**
    cointegradas). ``pvalue < 0,05`` ⇒ há relação de equilíbrio de longo prazo."""
    from statsmodels.tsa.stattools import coint

    yy = np.asarray(y, dtype=float)
    xx = np.asarray(x, dtype=float)
    stat, pval, crit = coint(yy, xx, trend=trend)
    return CointegrationResult(
        method="Engle-Granger", statistic=float(stat), pvalue=float(pval),
        rank=int(pval < 0.05), detail={"crit_5%": float(crit[1])},
    )


def johansen_test(data: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1,
                  alpha: str = "5%") -> CointegrationResult:
    """**Johansen**: nº de relações de cointegração (posto) num sistema (§3.3).

    Usa a estatística do **traço** contra os valores críticos (coluna ``alpha``) e
    devolve o **posto de cointegração** estimado — 0 = sem cointegração, ``k`` =
    todas estacionárias. ``det_order``: −1 (sem determinístico), 0 (constante), 1
    (tendência); ``k_ar_diff``: defasagens em diferença.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    jres = coint_johansen(data, det_order, k_ar_diff)
    col = {"10%": 0, "5%": 1, "1%": 2}[alpha]
    trace = jres.lr1
    crit = jres.cvt[:, col]
    rank = int(np.sum(trace > crit))  # nº de hipóteses r<=i rejeitadas
    return CointegrationResult(
        method="Johansen (traço)", statistic=float(trace[0]), pvalue=None, rank=rank,
        detail={"trace": trace.tolist(), "crit": crit.tolist(), "alpha": alpha},
    )


# ======================================================================
# VAR — vetor autorregressivo
# ======================================================================
class VARModel:
    """VAR sobre ``[risco_transformado, macro...]`` (Guia §3.3).

    Serve a **impulso-resposta** (:meth:`irf` — resposta da PD a choques macro) e
    a geração de **cenários internamente consistentes** (:meth:`forecast`). Em
    séries curtas, restrinja a poucas variáveis.
    """

    name = "VAR"

    def __init__(self, series, macro: pd.DataFrame, *, link: Optional[str] = None,
                 variables: Optional[Sequence[str]] = None, maxlags: int = 4, ic: str = "aic") -> None:
        self.data, self.risk_col = joint_frame(series, macro, link, variables)
        self._link = get_link(link or as_risk_series(series).default_link)
        self.maxlags = maxlags
        self.ic = ic
        self.res = None

    def fit(self, maxlags: Optional[int] = None):
        from statsmodels.tsa.api import VAR

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = VAR(self.data).fit(maxlags=maxlags or self.maxlags, ic=self.ic)
        return self

    def _require(self):
        if self.res is None:
            raise RuntimeError("VARModel: chame .fit() primeiro.")
        return self.res

    def forecast(self, steps: int = 12) -> pd.DataFrame:
        """Previsão do sistema; a coluna de risco volta à escala original da taxa."""
        res = self._require()
        k = res.k_ar
        fc = res.forecast(self.data.values[-k:], steps=steps)
        from . import _engine
        idx = _engine.future_index(self.data.index, "MS", steps)
        out = pd.DataFrame(fc, columns=self.data.columns, index=idx)
        out[self.risk_col.replace("_link", "")] = self._link.inverse(out[self.risk_col].to_numpy())
        return out

    def irf(self, periods: int = 12, orth: bool = True) -> pd.DataFrame:
        """Funções de **impulso-resposta**: resposta do risco (na escala do *link*)
        a um choque de 1 desvio em cada variável, ao longo de ``periods`` (§3.3)."""
        res = self._require()
        irf = res.irf(periods)
        arr = irf.orth_irfs if orth else irf.irfs  # (periods+1, k, k)
        ridx = list(self.data.columns).index(self.risk_col)
        resp = arr[:, ridx, :]  # resposta do risco a cada impulso
        return pd.DataFrame(resp, columns=[f"choque_{c}" for c in self.data.columns])

    def granger_causality(self, causing) -> dict:
        """Teste de causalidade de Granger: ``causing`` ajuda a prever o risco?"""
        res = self._require()
        test = res.test_causality(self.risk_col, causing, kind="f")
        return {"statistic": float(test.test_statistic), "pvalue": float(test.pvalue),
                "conclusion": "causa Granger" if test.pvalue < 0.05 else "não causa"}


# ======================================================================
# VECM — modelo de correção de erros
# ======================================================================
class VECMModel:
    """VECM (correção de erros) para séries **cointegradas** (Guia §3.3).

    A série se ajusta ao **desvio do equilíbrio** de longo prazo (o termo de
    correção de erro). Estime ``coint_rank`` por :func:`johansen_test` antes de
    ajustar.
    """

    name = "VECM"

    def __init__(self, series, macro: pd.DataFrame, *, link: Optional[str] = None,
                 variables: Optional[Sequence[str]] = None, k_ar_diff: int = 1,
                 coint_rank: int = 1, deterministic: str = "ci") -> None:
        self.data, self.risk_col = joint_frame(series, macro, link, variables)
        self._link = get_link(link or as_risk_series(series).default_link)
        self.k_ar_diff = k_ar_diff
        self.coint_rank = coint_rank
        self.deterministic = deterministic
        self.res = None

    def fit(self):
        from statsmodels.tsa.vector_ar.vecm import VECM

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = VECM(self.data, k_ar_diff=self.k_ar_diff, coint_rank=self.coint_rank,
                            deterministic=self.deterministic).fit()
        return self

    def _require(self):
        if self.res is None:
            raise RuntimeError("VECMModel: chame .fit() primeiro.")
        return self.res

    def forecast(self, steps: int = 12) -> pd.DataFrame:
        """Previsão do sistema; a coluna de risco volta à escala original."""
        res = self._require()
        fc = res.predict(steps=steps)
        from . import _engine
        idx = _engine.future_index(self.data.index, "MS", steps)
        out = pd.DataFrame(fc, columns=self.data.columns, index=idx)
        out[self.risk_col.replace("_link", "")] = self._link.inverse(out[self.risk_col].to_numpy())
        return out

    def alpha_beta(self) -> dict:
        """Vetores de **ajuste** (``alpha``, velocidade de correção) e de
        **cointegração** (``beta``, a relação de equilíbrio de longo prazo)."""
        res = self._require()
        return {"alpha": np.asarray(res.alpha), "beta": np.asarray(res.beta),
                "colunas": list(self.data.columns)}


__all__ = [
    "joint_frame",
    "CointegrationResult",
    "engle_granger",
    "johansen_test",
    "VARModel",
    "VECMModel",
]
