"""
Pipeline declarativo do estudo (Guia §6.1 (i), §6.3 — as "cinco chamadas")
==========================================================================
O guia pede um **pipeline declarativo**: um estudo completo — série,
transformação, candidatas, testes, seleção, validação, projeção — descrito por
**configuração versionável**, não por notebook ad hoc; rodar o mesmo YAML produz
o mesmo resultado (§6.1 (i)). E fixa a régua de sucesso do desenho: **cinco
chamadas** entre a base bruta e o relatório de governança (§6.3).

:class:`StudyConfig` é essa configuração (serializável de/para dict/YAML) e
:func:`run_study` é o orquestrador que encadeia seleção → ajuste → diagnóstico →
projeção por cenários → relatório, opcionalmente registrando tudo no MLflow —
mantendo uma **fonte única** de cenários e projeção para todos os usos (§5).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

import pandas as pd

from .ardl import ARDL
from .base import FitResult, Projection
from .fractional import BetaRegression, FractionalLogit
from .scenarios import ScenarioSet, standard_scenarios
from .selection import SearchResult, search
from .series import RiskSeries, as_risk_series
from .vasicek import VasicekZ

#: Modelos disponíveis por nome (para a configuração declarativa).
MODEL_REGISTRY = {
    "ardl": ARDL,
    "beta": BetaRegression,
    "fractional": FractionalLogit,
    "vasicek": VasicekZ,
}


@dataclass
class StudyConfig:
    """Configuração versionável de um estudo de modelo satélite (Guia §6.1)."""

    kind: str = "pd"
    model: str = "ardl"
    link: str = "logit"
    candidates: Sequence[str] = field(default_factory=list)
    expected_signs: Mapping[str, int] = field(default_factory=dict)
    lag_set: Sequence[int] = (0, 1, 3, 6)
    max_vars: int = 3
    ar_orders: Sequence[int] = (1,)
    seasonal: bool = False
    seasonal_period: int = 12
    horizon: int = 12
    min_train: Optional[int] = None
    vif_max: float = 5.0
    criterion: str = "oos_rmse"
    max_specs: int = 400
    # específicos do fator Z de Vasicek
    rho: Optional[float] = None
    pd_ttc: Optional[float] = None
    # cenários
    stress_var: str = "desemprego"
    scenario_probabilities: Sequence[float] = (0.5, 0.3, 0.2)
    name: str = "estudo"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "StudyConfig":
        campos = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in campos})


@dataclass
class StudyResult:
    """Saída de :func:`run_study` — o estudo completo, pronto para governança."""

    config: StudyConfig
    search: SearchResult
    best: object                 # SatelliteModel ajustado
    fit: FitResult
    diagnostics: pd.DataFrame
    scenarios: ScenarioSet
    projection: Projection
    report_html: Optional[str] = None
    mlflow_run_id: Optional[str] = None

    def summary(self) -> pd.DataFrame:
        """Uma linha com a especificação escolhida e as métricas-chave."""
        row = {"estudo": self.config.name, "modelo": self.fit.model_name,
               "spec": self.fit.spec.describe() if self.fit.spec else "—",
               "AIC": self.fit.aic, "BIC": self.fit.bic, "R2": self.fit.rsquared}
        qual = self.search.ranking[self.search.ranking["status"] == "qualificado"]
        if not qual.empty:
            row["oos_rmse"] = float(qual["oos_rmse"].iloc[0])
            if "vs_arima" in qual.columns:
                row["vs_arima"] = float(qual["vs_arima"].iloc[0])
        return pd.DataFrame([row])


def run_study(
    config: StudyConfig,
    series,
    macro: pd.DataFrame,
    *,
    make_report: bool = True,
    log_mlflow: bool = False,
    mlflow_kwargs: Optional[dict] = None,
) -> StudyResult:
    """Roda o estudo de ponta a ponta a partir da configuração (Guia §6.3).

    Encadeia: **seleção** champion-challenger sobre a grade de candidatas →
    **ajuste** do melhor modelo → **diagnóstico** de resíduos → **projeção**
    condicional aos cenários padrão (base/adverso/otimista) → **relatório** HTML,
    opcionalmente **registrado no MLflow**. As cinco chamadas do guia, numa só.

    Parameters
    ----------
    config:
        A :class:`StudyConfig`.
    series, macro:
        A :class:`RiskSeries` (ou ``pandas.Series`` + ``config.kind``) e a macro.
    make_report:
        Gera o HTML de governança (:func:`.report.model_report`).
    log_mlflow, mlflow_kwargs:
        Se ``True``, registra o *run* (:func:`.tracking.log_satellite_run`).

    Returns
    -------
    StudyResult
    """
    rs: RiskSeries = as_risk_series(series, kind=config.kind)
    model_cls = MODEL_REGISTRY.get(config.model)
    if model_cls is None:
        raise ValueError(f"model desconhecido: {config.model!r}. Válidos: {list(MODEL_REGISTRY)}.")

    model_kwargs = {}
    if config.model == "vasicek":
        if config.rho is None:
            raise ValueError("model='vasicek' exige config.rho (correlação de ativos).")
        model_kwargs = {"rho": config.rho, "pd_ttc": config.pd_ttc}

    grid_kwargs = {"lag_set": config.lag_set, "max_vars": config.max_vars,
                   "ar_orders": config.ar_orders, "seasonal": config.seasonal,
                   "seasonal_period": config.seasonal_period, "max_specs": config.max_specs}

    res = search(
        rs, macro, candidates=config.candidates, expected_signs=config.expected_signs,
        model_cls=model_cls, model_kwargs=model_kwargs, link=config.link,
        horizon=config.horizon, min_train=config.min_train, criterion=config.criterion,
        vif_max=config.vif_max, grid_kwargs=grid_kwargs,
    )
    if res.best is None:
        raise RuntimeError(
            "nenhuma especificação qualificada (todas reprovadas por sinal/VIF). "
            "Revise candidatas, sinais esperados ou o teto de VIF."
        )
    fit = res.best.result
    diag = fit.diagnostics()

    scen = standard_scenarios(macro, horizon=config.horizon, stress_var=config.stress_var,
                              probabilities=config.scenario_probabilities)
    projection = res.best.project(scen, n_sims=2000, seed=0)

    report_html = None
    if make_report:
        from . import report as _report
        report_html = _report.model_report(fit, rs, projection, title=f"Estudo: {config.name}")

    run_id = None
    if log_mlflow:
        from . import tracking as _tracking
        run_id = _tracking.log_satellite_run(
            fit, series=rs, projection=projection, search=res,
            run_name=config.name, **(mlflow_kwargs or {}))

    return StudyResult(config=config, search=res, best=res.best, fit=fit, diagnostics=diag,
                       scenarios=scen, projection=projection, report_html=report_html,
                       mlflow_run_id=run_id)


__all__ = ["MODEL_REGISTRY", "StudyConfig", "StudyResult", "run_study"]
