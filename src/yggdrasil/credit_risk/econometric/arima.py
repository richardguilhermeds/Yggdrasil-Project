"""
ARIMA / ARIMAX / SARIMA (Box-Jenkins) — o benchmark univariado (Guia §3.2)
==========================================================================
ARIMA modela a série apenas pela sua **própria dinâmica** (autorregressiva,
integração e médias móveis) e é o **benchmark univariado obrigatório**: qualquer
modelo com variáveis macro precisa **prever melhor que o ARIMA fora da amostra**
para justificar a complexidade (Guia §3.2, §4.3). ARIMAX acrescenta regressores
exógenos à estrutura ARIMA (aproximando-se do ARDL, com tratamento explícito do
erro por médias móveis); SARIMA acrescenta a camada sazonal.

Motor: :class:`statsmodels.tsa.statespace.sarimax.SARIMAX` (MLE por espaço de
estados) — a implementação de referência, com forecasting e intervalos analíticos
prontos. Como os demais modelos, opera na escala de um *link* (logit/probit) para
respeitar o suporte ``(0, 1)`` da taxa, e reconverte na projeção.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import _engine
from .base import Projection, SatelliteModel, build_fit_result
from .series import RiskSeries, as_risk_series
from .transforms import get_link, make_lags


def _build_exog(macro: Optional[pd.DataFrame], exog) -> Optional[pd.DataFrame]:
    """Constrói a matriz exógena a partir de ``exog`` (lista contemporânea ou dict
    ``{var: [lags]}``)."""
    if not exog or macro is None:
        return None
    if isinstance(exog, dict):
        return make_lags(macro[list(exog.keys())], exog)
    return macro[list(exog)].copy()


class ARIMA(SatelliteModel):
    """ARIMA/ARIMAX/SARIMA sobre a taxa transformada (escala de *link*).

    Parameters
    ----------
    series:
        A :class:`RiskSeries` (ou ``pandas.Series`` + ``kind``) da dependente.
    macro, exog:
        Para ARIMAX: ``macro`` é o DataFrame e ``exog`` diz quais colunas/defasagens
        usar — uma **lista** de nomes (contemporâneos) ou um **dict** ``{var:
        [lags]}``. ``exog=None`` ⇒ ARIMA puro (o benchmark).
    order:
        ``(p, d, q)`` — ordens AR, integração e MA.
    seasonal_order:
        ``(P, D, Q, s)`` — camada sazonal (``s=12`` mensal). ``(0,0,0,0)`` desliga.
    link:
        Transformação da dependente (padrão: o *link* do ``kind``). Use
        ``"identity"`` para modelar o nível diretamente.
    trend:
        Determinístico do SARIMAX (``"c"``, ``"n"``, ``"t"``, ``"ct"``). ``None``
        ⇒ ``"c"`` se ``d=0`` senão ``"n"`` (evita deriva sobre série diferenciada).
    """

    name = "ARIMA"

    def __init__(
        self,
        series,
        macro: Optional[pd.DataFrame] = None,
        *,
        exog=None,
        order: tuple = (1, 0, 0),
        seasonal_order: tuple = (0, 0, 0, 0),
        link: Optional[str] = None,
        trend: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> None:
        rs: RiskSeries = as_risk_series(series, kind=kind or "pd")
        super().__init__(kind=rs.kind, link=link or rs.default_link)
        self.series = rs
        self.macro = macro
        self.exog_spec = exog
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend if trend is not None else ("c" if order[1] == 0 else "n")
        self._link = get_link(self.link)
        self._res = None
        self._exog_cols: Optional[list] = None

    # ------------------------------------------------------------------
    def fit(self):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        endog_link = self._link.forward(self.series.values)
        endog_link.name = "y"
        exog = _build_exog(self.macro, self.exog_spec)
        if exog is not None:
            df = pd.concat([endog_link.rename("y"), exog], axis=1).dropna()
            y = df.pop("y")
            exog = df
            self._exog_cols = list(exog.columns)
        else:
            y = endog_link.dropna()

        model = SARIMAX(y, exog=exog, order=self.order, seasonal_order=self.seasonal_order,
                        trend=self.trend, enforce_stationarity=False, enforce_invertibility=False)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(disp=False)
        self._res = res

        fitted_link = res.fittedvalues
        resid = res.resid
        # descarta o transiente inicial do filtro (primeiras d+P*s obs)
        burn = max(1, self.order[1] + self.seasonal_order[3] * self.seasonal_order[1])
        fitted_link = fitted_link.iloc[burn:]
        resid = resid.iloc[burn:]

        params = res.params
        fr = build_fit_result(
            model_name=self.name, kind=self.kind, link=self.link,
            params=params, fitted_link=fitted_link, resid=resid,
            observed_original=self.series.values, inverse_link=self._link.inverse,
            n_params=int(len(params)), exog=exog.loc[resid.index] if exog is not None else None,
            spec=None, raw=res, llf=float(res.llf), aic=float(res.aic), bic=float(res.bic),
        )
        self.result = fr
        return fr

    # ------------------------------------------------------------------
    def _future_exog(self, macro_future: pd.DataFrame, future_index) -> Optional[pd.DataFrame]:
        if self.exog_spec is None:
            return None
        macro_full = _engine.concat_macro(self.macro, macro_future)
        exog_full = _build_exog(macro_full, self.exog_spec)
        return exog_full.loc[future_index]

    def predict(self, exog_future: Optional[pd.DataFrame] = None,
                steps: Optional[int] = None) -> pd.Series:
        """In-sample (``exog_future=None``) ou previsão fora da amostra.

        ``exog_future`` é a **macro futura** (mesmo se o modelo for ARIMA puro,
        pode passar o DataFrame para definir o horizonte); ``steps`` limita o nº de
        passos."""
        fr = self._require_fit()
        if exog_future is None and steps is None:
            return fr.fitted
        if exog_future is not None:
            fut_index = exog_future.index if steps is None else exog_future.index[:steps]
            ex = self._future_exog(exog_future, fut_index)
            fc = self._res.get_forecast(steps=len(fut_index), exog=ex)
            mean = pd.Series(np.asarray(self._link.inverse(fc.predicted_mean)), index=fut_index)
        else:
            fc = self._res.get_forecast(steps=steps)
            mean = pd.Series(np.asarray(self._link.inverse(fc.predicted_mean)),
                             index=fc.predicted_mean.index)
        mean.name = f"{self.kind}_previsto"
        return mean

    # ------------------------------------------------------------------
    def project(self, scenarios, horizon: Optional[int] = None, alpha: float = 0.10,
                n_sims: int = 2000, seed: int = 0) -> Projection:
        """Projeção com intervalos **analíticos** do espaço de estados
        (``get_forecast().conf_int``), reconvertidos à escala da taxa. ``n_sims`` é
        ignorado (o SARIMAX propaga a incerteza multi-passo analiticamente)."""
        fr = self._require_fit()
        items = _engine.normalize_scenarios(scenarios)
        paths: dict[str, pd.DataFrame] = {}
        probs: dict[str, float] = {}
        H = 0
        for name, macro_future, prob in items:
            if macro_future is None:
                raise ValueError(f"cenário {name!r} sem macro futura.")
            fut_index = macro_future.index if horizon is None else macro_future.index[:horizon]
            H = len(fut_index)
            ex = self._future_exog(macro_future, fut_index)
            fc = self._res.get_forecast(steps=H, exog=ex)
            mean_link = np.asarray(fc.predicted_mean)
            ci = fc.conf_int(alpha=alpha)
            lo_link = np.asarray(ci.iloc[:, 0])
            hi_link = np.asarray(ci.iloc[:, 1])
            paths[name] = pd.DataFrame({
                "mean": np.asarray(self._link.inverse(mean_link)),
                "lower": np.asarray(self._link.inverse(lo_link)),
                "upper": np.asarray(self._link.inverse(hi_link)),
                "mean_link": mean_link,
            }, index=fut_index)
            probs[name] = prob
        probabilities = probs if all(p is not None for p in probs.values()) else None
        return Projection(paths=paths, kind=self.kind, link=self.link, horizon=H,
                          alpha=alpha, probabilities=probabilities)


__all__ = ["ARIMA"]
