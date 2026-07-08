"""
Benchmarks ingênuos (Guia §4.3) — o piso que o modelo macro precisa superar
===========================================================================
A validação fora da amostra compara o modelo candidato contra **benchmarks
ingênuos**: passeio aleatório, média histórica e sazonal-ingênuo (§4.3). "Um
modelo macro que não supera o ARIMA — nem os ingênuos — fora da amostra não está
pronto." Estes modelos conformam à mesma interface (:class:`Satellite_
Model`) para entrarem na comparação *walk-forward* e no teste de Diebold-Mariano
como qualquer outro candidato.

Operam na **escala original** da taxa (são referências propositalmente simples):

* :class:`RandomWalk` — a projeção é o **último valor observado** (a variância
  cresce com o horizonte, ``σ_Δ·√h``).
* :class:`HistoricalMean` — a projeção é a **média histórica** (a âncora de
  reversão à média).
* :class:`SeasonalNaive` — a projeção repete o valor da **mesma estação** do
  ciclo anterior (útil em séries mensais de varejo com sazonalidade forte).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .base import Projection, SatelliteModel, build_fit_result
from .series import RiskSeries, as_risk_series


def _identity(x):
    return np.asarray(x, dtype=float)


def _future_index(index: pd.Index, freq: str, n: int) -> pd.Index:
    """Gera ``n`` períodos futuros a partir do fim de ``index`` na frequência ``freq``."""
    ts = pd.DatetimeIndex(index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index)
    off = pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start=ts[-1] + off, periods=n, freq=freq)


class _NaiveModel(SatelliteModel):
    """Base dos benchmarks ingênuos (escala original, sem macro)."""

    name = "_Naive"

    def __init__(self, series, *, kind: Optional[str] = None) -> None:
        rs: RiskSeries = as_risk_series(series, kind=kind or "pd")
        super().__init__(kind=rs.kind, link="identity")
        self.series = rs

    def _insample(self) -> tuple[pd.Series, pd.Series]:
        """Devolve ``(ajustado, observado)`` in-sample (um passo à frente)."""
        raise NotImplementedError

    def fit(self):
        fitted, obs = self._insample()
        resid = (obs - fitted).dropna()
        fitted = fitted.reindex(resid.index)
        fr = build_fit_result(
            model_name=self.name, kind=self.kind, link="identity",
            params=pd.Series(dtype=float), fitted_link=fitted, resid=resid,
            observed_original=self.series.values, inverse_link=_identity,
            n_params=self._n_params(), exog=None, spec=None, raw=None,
        )
        self.result = fr
        self._sigma_step = float(np.nanstd(resid.to_numpy(dtype=float), ddof=1))
        return fr

    def _n_params(self) -> int:
        return 0

    def _mean_path(self, n: int) -> np.ndarray:
        raise NotImplementedError

    def _band_scale(self, h_arr: np.ndarray) -> np.ndarray:
        """Desvio-padrão da projeção por passo (subclasses ajustam a acumulação)."""
        return np.full(len(h_arr), self._sigma_step)

    def predict(self, exog_future: Optional[pd.DataFrame] = None,
                steps: Optional[int] = None) -> pd.Series:
        fr = self._require_fit()
        if exog_future is None and steps is None:
            return fr.fitted
        if exog_future is not None:
            idx = exog_future.index
            n = len(idx) if steps is None else min(steps, len(idx))
            idx = idx[:n]
        else:
            n = steps
            idx = _future_index(self.series.index, self.series.frequency, n)
        out = pd.Series(np.clip(self._mean_path(n), 0.0, 1.0), index=idx,
                        name=f"{self.kind}_previsto")
        return out

    def project(self, scenarios, horizon: Optional[int] = None, alpha: float = 0.10,
                n_sims: int = 2000, seed: int = 0) -> Projection:
        from . import _engine
        fr = self._require_fit()
        items = _engine.normalize_scenarios(scenarios)
        z = norm.ppf(1 - alpha / 2)
        paths: dict[str, pd.DataFrame] = {}
        probs: dict[str, float] = {}
        H = 0
        for name, macro_future, prob in items:
            idx = macro_future.index if macro_future is not None else \
                _future_index(self.series.index, self.series.frequency, horizon or 12)
            if horizon is not None:
                idx = idx[:horizon]
            H = len(idx)
            mean = np.clip(self._mean_path(H), 0.0, 1.0)
            scale = self._band_scale(np.arange(1, H + 1))
            paths[name] = pd.DataFrame({
                "mean": mean,
                "lower": np.clip(mean - z * scale, 0.0, 1.0),
                "upper": np.clip(mean + z * scale, 0.0, 1.0),
                "mean_link": mean,
            }, index=idx)
            probs[name] = prob
        probabilities = probs if all(p is not None for p in probs.values()) else None
        return Projection(paths=paths, kind=self.kind, link="identity", horizon=H,
                          alpha=alpha, probabilities=probabilities)


class RandomWalk(_NaiveModel):
    """Passeio aleatório: projeção = último valor; variância cresce como ``σ_Δ²·h``."""

    name = "RandomWalk"

    def _insample(self):
        y = self.series.values
        return y.shift(1), y

    def _mean_path(self, n: int) -> np.ndarray:
        return np.full(n, float(self.series.values.iloc[-1]))

    def _band_scale(self, h_arr: np.ndarray) -> np.ndarray:
        return self._sigma_step * np.sqrt(h_arr)


class HistoricalMean(_NaiveModel):
    """Média histórica: projeção = média da série (âncora de reversão à média)."""

    name = "HistoricalMean"

    def _n_params(self) -> int:
        return 1

    def _insample(self):
        y = self.series.values
        # média expansível (usa só o passado) para não vazar futuro no resíduo
        exp_mean = y.expanding(min_periods=2).mean().shift(1)
        return exp_mean, y

    def _mean_path(self, n: int) -> np.ndarray:
        return np.full(n, float(self.series.values.mean()))


class SeasonalNaive(_NaiveModel):
    """Sazonal-ingênuo: projeção = valor da mesma estação do ciclo anterior."""

    name = "SeasonalNaive"

    def __init__(self, series, *, period: int = 12, kind: Optional[str] = None) -> None:
        super().__init__(series, kind=kind)
        self.period = int(period)

    def _insample(self):
        y = self.series.values
        return y.shift(self.period), y

    def _mean_path(self, n: int) -> np.ndarray:
        y = self.series.values.to_numpy(dtype=float)
        tail = y[-self.period:]
        return np.array([tail[i % self.period] for i in range(n)])

    def _band_scale(self, h_arr: np.ndarray) -> np.ndarray:
        return np.full(len(h_arr), self._sigma_step)


#: Fábricas dos benchmarks para uso pela seleção (Guia §4.3).
NAIVE_MODELS = {
    "random_walk": RandomWalk,
    "media_historica": HistoricalMean,
    "sazonal_ingenuo": SeasonalNaive,
}


__all__ = ["RandomWalk", "HistoricalMean", "SeasonalNaive", "NAIVE_MODELS"]
