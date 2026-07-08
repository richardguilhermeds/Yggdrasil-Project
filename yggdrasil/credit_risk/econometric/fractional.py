"""
Modelos para variáveis limitadas: beta regression e fractional logit (Guia §3.5)
================================================================================
Para **LGD e CCF** (e taxas muito próximas das bordas), modelos que respeitam o
suporte ``(0, 1)`` **nativamente**, em vez de transformar e usar OLS:

* :class:`BetaRegression` — modela a **média e a dispersão** da distribuição beta
  em função dos regressores (Ferrari & Cribari-Neto, 2004). *Link* logit na média.
* :class:`FractionalLogit` — a **quase-verossimilhança** de Papke & Wooldridge
  (1996): GLM binomial com *link* logit e erros-padrão robustos (*sandwich*). Mais
  robusto a má especificação e aceita observações **exatamente em 0 e 1** — o caso
  da LGD com massa em cura total (0) e perda total (1).

Ambos partilham a mecânica dos modelos lineares-no-*link*: a média segue
``μ_t = logit⁻¹( c + Σφ·logit(y_{t-i}) + Σβ·macro )``, então a estimação muda
(MLE/QMLE em vez de OLS), mas o preditor linear, o diagnóstico de resíduos e a
**projeção multi-passo** (motor compartilhado) são os mesmos do ARDL. Para LGD com
massa em 0/1, o guia lembra a alternativa da **decomposição cura × severidade**,
geralmente preferível pela interpretabilidade (ver :func:`decompose_lgd`).
"""
from __future__ import annotations

import abc
from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

from . import _engine
from .base import Projection, SatelliteModel, Specification, build_fit_result
from .series import RiskSeries, as_risk_series
from .transforms import DEFAULT_EPS, align, get_link, inv_logit, logit


class _BoundedRegression(SatelliteModel):
    """Base dos modelos de resposta limitada (média com *link* logit).

    Subclasses implementam :meth:`_estimate` (o motor de estimação). O restante —
    montagem do design, resíduo na escala do *link*, projeção multi-passo — é comum
    e idêntico ao :class:`ARDL`.
    """

    name = "_BoundedRegression"

    def __init__(self, series, macro: Optional[pd.DataFrame] = None,
                 spec: Optional[Specification] = None, *, kind: Optional[str] = None) -> None:
        rs: RiskSeries = as_risk_series(series, kind=kind or "lgd")
        # a média é sempre logit aqui; copia o spec para não mutar o do chamador.
        if spec is None:
            spec = Specification(ar=1, link="logit")
        else:
            spec = replace(spec, link="logit")
        super().__init__(kind=rs.kind, link="logit")
        self.series = rs
        self.macro = macro
        self.spec = spec
        self._link = get_link("logit")
        self._endog_link: Optional[pd.Series] = None
        self._trend_offset = 0

    # -- hook de estimação ----------------------------------------------
    @abc.abstractmethod
    def _estimate(self, endog_raw: pd.Series, X: pd.DataFrame) -> dict:
        """Estima o modelo. Deve devolver um dict com ``params`` (coefs da média,
        índice = colunas de ``X``), ``llf``, ``aic``, ``bic``, ``bse``,
        ``tvalues``, ``pvalues``, ``n_params`` e ``raw``."""

    # ------------------------------------------------------------------
    def fit(self):
        endog_raw = self.series.values
        endog_link = logit(endog_raw)
        endog_link.name = "y"
        X = _engine.build_design(endog_link, self.macro, self.spec)
        # alinha endog (bruto e link) e o design, removendo bordas de defasagem
        combo = X.copy()
        combo["__yraw__"] = endog_raw
        combo["__ylink__"] = endog_link
        combo = combo.dropna()
        yraw = combo.pop("__yraw__")
        ylink = combo.pop("__ylink__")
        Xa = combo
        if Xa.shape[1] >= Xa.shape[0]:
            raise ValueError(
                f"{self.name}: {Xa.shape[1]} termos e só {Xa.shape[0]} observações — "
                "reduza a complexidade (Guia §2.3)."
            )

        est = self._estimate(yraw, Xa)
        params = pd.Series(est["params"]).reindex(Xa.columns)
        eta = pd.Series(Xa.to_numpy(dtype=float) @ params.to_numpy(dtype=float), index=Xa.index)
        resid = ylink - eta  # resíduo na escala do link (preditor linear)

        fr = build_fit_result(
            model_name=self.name, kind=self.kind, link="logit",
            params=params, fitted_link=eta, resid=resid,
            observed_original=self.series.values, inverse_link=inv_logit,
            n_params=int(est["n_params"]), exog=Xa, spec=self.spec, raw=est["raw"],
            llf=est.get("llf"), aic=est.get("aic"), bic=est.get("bic"),
            bse=est.get("bse"), tvalues=est.get("tvalues"), pvalues=est.get("pvalues"),
        )
        self.result = fr
        self._endog_link = endog_link
        # ver ARDL.fit: o offset da tendência 'ct' continua do índice cheio (N),
        # não do comprimento pós-align (len(Xa) = N-ar).
        self._trend_offset = int(len(self._endog_link))
        return fr

    # -- projeção (idêntica ao ARDL: motor compartilhado) ---------------
    def predict(self, exog_future: Optional[pd.DataFrame] = None,
                steps: Optional[int] = None) -> pd.Series:
        fr = self._require_fit()
        if exog_future is None:
            return fr.fitted
        macro_full = _engine.concat_macro(self.macro, exog_future)
        fut_index = exog_future.index if steps is None else exog_future.index[:steps]
        df = _engine.forecast_paths(
            fr.params, self.spec, self._link.inverse, self._endog_link, macro_full,
            fut_index, resid_pool=fr.resid.to_numpy(dtype=float),
            trend_offset=self._trend_offset, n_sims=0)
        out = df["mean"].copy()
        out.name = f"{self.kind}_previsto"
        return out

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
                trend_offset=self._trend_offset, n_sims=n_sims, alpha=alpha, seed=seed)
            probs[name] = prob
        probabilities = probs if all(p is not None for p in probs.values()) else None
        return Projection(paths=paths, kind=self.kind, link=self.link, horizon=H,
                          alpha=alpha, probabilities=probabilities)


class BetaRegression(_BoundedRegression):
    """Regressão **beta** (Ferrari & Cribari-Neto, 2004) para LGD/CCF.

    Modela a média (``link`` logit) e a precisão da distribuição beta. Exige
    resposta em ``(0, 1)`` **aberto** (recorta 0/1 por ``eps``); para massa exata
    em 0/1, prefira :class:`FractionalLogit` ou :func:`decompose_lgd`.
    """

    name = "BetaRegression"

    def _estimate(self, endog_raw: pd.Series, X: pd.DataFrame) -> dict:
        from statsmodels.othermod.betareg import BetaModel

        y = np.clip(endog_raw.to_numpy(dtype=float), DEFAULT_EPS, 1 - DEFAULT_EPS)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = BetaModel(pd.Series(y, index=endog_raw.index), X).fit()
        k = X.shape[1]  # os k primeiros params são os da média (depois vêm os de precisão)
        vals = np.asarray(res.params)[:k]
        bse = pd.Series(np.asarray(res.bse)[:k], index=X.columns)
        tv = pd.Series(np.asarray(res.tvalues)[:k], index=X.columns)
        pv = pd.Series(np.asarray(res.pvalues)[:k], index=X.columns)
        return {
            "params": pd.Series(vals, index=X.columns),
            "llf": float(res.llf), "aic": float(res.aic), "bic": float(res.bic),
            "bse": bse, "tvalues": tv, "pvalues": pv,
            "n_params": int(len(res.params)), "raw": res,
        }


class FractionalLogit(_BoundedRegression):
    """**Fractional logit** (Papke & Wooldridge, 1996): GLM binomial, *link* logit,
    quase-verossimilhança com erros-padrão robustos (*sandwich*, ``HC0``).

    Aceita resposta **exatamente em 0 e 1** (curas e perdas totais) e é robusto a
    má especificação da distribuição — a escolha padrão do guia para LGD com massa
    nas bordas quando não se decompõe.
    """

    name = "FractionalLogit"

    def _estimate(self, endog_raw: pd.Series, X: pd.DataFrame) -> dict:
        import statsmodels.api as sm

        y = np.clip(endog_raw.to_numpy(dtype=float), 0.0, 1.0)
        model = sm.GLM(pd.Series(y, index=endog_raw.index), X, family=sm.families.Binomial())
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(cov_type="HC0")
        # bic_llf: BIC pela log-verossimilhança (consistente com ARDL/Beta e estável
        # entre versões do statsmodels), no lugar do BIC por deviance (deprecado).
        # hasattr evita avaliar res.bic (que dispara o FutureWarning) quando há bic_llf.
        bic = float(res.bic_llf) if hasattr(res, "bic_llf") else float(res.bic)
        return {
            "params": pd.Series(np.asarray(res.params), index=X.columns),
            "llf": float(res.llf), "aic": float(res.aic), "bic": bic,
            "bse": pd.Series(np.asarray(res.bse), index=X.columns),
            "tvalues": pd.Series(np.asarray(res.tvalues), index=X.columns),
            "pvalues": pd.Series(np.asarray(res.pvalues), index=X.columns),
            "n_params": int(len(res.params)), "raw": res,
        }


# ======================================================================
# Decomposição cura × severidade (Guia §2.2, §3.5)
# ======================================================================
def decompose_lgd(
    cure_rate: pd.Series,
    severity_given_loss: pd.Series,
    macro: Optional[pd.DataFrame] = None,
    *,
    cure_spec: Optional[Specification] = None,
    severity_spec: Optional[Specification] = None,
) -> dict:
    """Modela a LGD pela **decomposição** em (probabilidade de cura) × (severidade
    dos não curados) — costuma ser mais estável que modelar a LGD média agregada
    (Guia §2.2, §3.5).

    ``LGD = (1 − p_cura) · severidade_dado_perda``. Ajusta um
    :class:`FractionalLogit` para cada componente e devolve os dois modelos
    ajustados mais uma função ``lgd(macro_futura)`` que recompõe a LGD projetada.

    Parameters
    ----------
    cure_rate:
        Série da **probabilidade de cura** por período, em ``[0, 1]``.
    severity_given_loss:
        Série da **severidade média dos casos não curados** por período, em
        ``[0, 1]``.
    macro, cure_spec, severity_spec:
        Macro e especificações de cada componente (padrão AR(1) logit).

    Returns
    -------
    dict
        ``{"cura": FractionalLogit, "severidade": FractionalLogit, "lgd_fn": fn}``,
        onde ``lgd_fn(macro_future)`` devolve a série de LGD projetada.
    """
    cure_spec = cure_spec or Specification(ar=1, link="logit")
    severity_spec = severity_spec or Specification(ar=1, link="logit")
    mc = FractionalLogit(RiskSeries(cure_rate, kind="lgd", segment="cura"), macro, cure_spec)
    mc.fit()
    ms = FractionalLogit(RiskSeries(severity_given_loss, kind="lgd", segment="severidade"),
                         macro, severity_spec)
    ms.fit()

    def lgd_fn(macro_future: pd.DataFrame) -> pd.Series:
        p_cura = mc.predict(macro_future)
        sev = ms.predict(macro_future)
        out = (1.0 - p_cura) * sev
        out.name = "lgd_decomposta"
        return out

    return {"cura": mc, "severidade": ms, "lgd_fn": lgd_fn}


__all__ = ["BetaRegression", "FractionalLogit", "decompose_lgd"]
