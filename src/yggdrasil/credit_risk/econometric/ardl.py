"""
ARDL — regressão com defasagens distribuídas (Guia §3.1, Tabela 1)
==================================================================
**O cavalo de batalha dos modelos satélite.** A transformação da PD (logit ou
``Z``) — ou da LGD/CCF — é regredida contra defasagens dela própria e das
variáveis macro:

    ``y_t = c + Σ_i φ_i · y_{t-i} + Σ_j β_j · x_{j, t-lag} + ε_t``

As defasagens próprias capturam a **inércia** do processo de inadimplência
(*defaults* respondem ao desemprego com atraso de meses) e as macro defasadas, a
dinâmica de ajuste. É interpretável, parcimonioso e projeta condicionalmente a
cenários de forma natural — o candidato a **modelo principal** na maioria dos
segmentos de varejo.

Implementação: a matriz de defasagens é montada explicitamente
(:func:`~..._engine.build_design`, controle exato de quais defasagens entram — o
guia recomenda percorrer defasagens de 0 a 6, §4.1) e estimada por **OLS**
(:mod:`statsmodels`), o que dá coeficientes nomeados por defasagem (``desemprego_l3``)
para a inspeção de **sinal econômico** que a seleção exige. A projeção multi-passo
e os intervalos por simulação de resíduos vêm do motor compartilhado.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import _engine
from .base import Projection, SatelliteModel, Specification, build_fit_result
from .series import RiskSeries, as_risk_series
from .transforms import align, get_link


class ARDL(SatelliteModel):
    """Modelo ARDL (OLS sobre a matriz de defasagens) na escala de um *link*.

    Parameters
    ----------
    series:
        A :class:`~yggdrasil.credit_risk.econometric.series.RiskSeries` (ou uma
        ``pandas.Series`` de taxa, com ``kind``) da dependente.
    macro:
        DataFrame das variáveis macro alinhado ao índice de ``series`` (mesma
        frequência). ``None`` ⇒ modelo univariado (só AR + determinísticos).
    spec:
        A :class:`Specification` (defasagens macro, ordem AR, *link*, sazonal,
        eventos). Se ``None``, usa AR(1) + o *link* padrão do ``kind`` sem macro.
    kind:
        Sobrescreve o ``kind`` quando ``series`` é uma ``pandas.Series`` crua.
    cov_type:
        Covariância dos erros: ``"nonrobust"`` (padrão) ou ``"HAC"`` (Newey-West,
        robusta a autocorrelação/heterocedasticidade — recomendada em séries de
        risco). ``hac_maxlags`` controla a janela do HAC (padrão: regra de
        Newey-West ``⌊4·(n/100)^{2/9}⌋``).
    """

    name = "ARDL"

    def __init__(
        self,
        series,
        macro: Optional[pd.DataFrame] = None,
        spec: Optional[Specification] = None,
        *,
        kind: Optional[str] = None,
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
    ) -> None:
        rs: RiskSeries = as_risk_series(series, kind=kind or "pd")
        if spec is None:
            spec = Specification(ar=1, link=rs.default_link)
        super().__init__(kind=rs.kind, link=spec.link)
        self.series = rs
        self.macro = macro
        self.spec = spec
        self.cov_type = cov_type
        self.hac_maxlags = hac_maxlags
        self._link = get_link(spec.link)
        self._endog_link: Optional[pd.Series] = None
        self._trend_offset: int = 0

    # ------------------------------------------------------------------
    def fit(self):
        import statsmodels.api as sm

        endog_link = self._link.forward(self.series.values)
        endog_link.name = "y"
        X = _engine.build_design(endog_link, self.macro, self.spec)
        y, Xa = align(endog_link, X)
        if Xa.shape[1] >= Xa.shape[0]:
            raise ValueError(
                f"{self.name}: especificação com {Xa.shape[1]} termos e apenas "
                f"{Xa.shape[0]} observações após defasar — reduza a complexidade "
                "(o guia recomenda 2 a 4 variáveis em séries curtas, §2.3)."
            )
        model = sm.OLS(y, Xa)
        if self.cov_type == "HAC":
            maxlags = self.hac_maxlags or int(np.floor(4 * (len(y) / 100.0) ** (2.0 / 9.0)))
            res = model.fit(cov_type="HAC", cov_kwds={"maxlags": max(1, maxlags)})
        else:
            res = model.fit(cov_type=self.cov_type)

        fr = build_fit_result(
            model_name=self.name, kind=self.kind, link=self.link,
            params=res.params, fitted_link=res.fittedvalues, resid=res.resid,
            observed_original=self.series.values, inverse_link=self._link.inverse,
            n_params=int(len(res.params)), exog=Xa, spec=self.spec, raw=res,
            llf=float(res.llf), aic=float(res.aic), bic=float(res.bic),
            rsquared=float(res.rsquared), bse=res.bse, tvalues=res.tvalues, pvalues=res.pvalues,
        )
        self.result = fr
        self._endog_link = endog_link
        # offset do contador de tendência determinística ('ct'): deve continuar do
        # ÚLTIMO valor de trend do design (= N-1, sobre o índice cheio de comprimento
        # N), independentemente das linhas de burn-in que o align() removeu. Logo o
        # primeiro passo projetado usa trend=N. Usar len(y) (pós-align = N-ar)
        # subestimaria o contador em `ar` a cada horizonte.
        self._trend_offset = int(len(self._endog_link))
        return fr

    # ------------------------------------------------------------------
    def predict(self, exog_future: Optional[pd.DataFrame] = None,
                steps: Optional[int] = None) -> pd.Series:
        """In-sample (``exog_future=None``) ou *point forecast* fora da amostra.

        Com ``exog_future`` (macro futura), devolve a trajetória média da taxa
        (escala original) por propagação multi-passo — sem intervalos (use
        :meth:`project` para o leque)."""
        fr = self._require_fit()
        if exog_future is None and steps is None:
            return fr.fitted
        if exog_future is not None:
            macro_full = _engine.concat_macro(self.macro, exog_future)
            fut_index = exog_future.index if steps is None else exog_future.index[:steps]
        else:  # passos sem macro (modelo puramente AR/determinístico)
            fut_index = _engine.future_index(self.series.index, self.series.frequency, steps)
            macro_full = self.macro
        df = _engine.forecast_paths(
            fr.params, self.spec, self._link.inverse, self._endog_link, macro_full,
            fut_index, resid_pool=fr.resid.to_numpy(dtype=float),
            trend_offset=self._trend_offset, n_sims=0,
        )
        out = df["mean"].copy()
        out.name = f"{self.kind}_previsto"
        return out

    # ------------------------------------------------------------------
    def project(self, scenarios, horizon: Optional[int] = None, alpha: float = 0.10,
                n_sims: int = 2000, seed: int = 0) -> Projection:
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
            macro_full = _engine.concat_macro(self.macro, macro_future)
            paths[name] = _engine.forecast_paths(
                fr.params, self.spec, self._link.inverse, self._endog_link, macro_full,
                fut_index, resid_pool=fr.resid.to_numpy(dtype=float),
                trend_offset=self._trend_offset, n_sims=n_sims, alpha=alpha, seed=seed,
            )
            probs[name] = prob
        probabilities = probs if all(p is not None for p in probs.values()) else None
        return Projection(paths=paths, kind=self.kind, link=self.link, horizon=H,
                          alpha=alpha, probabilities=probabilities)


__all__ = ["ARDL"]
