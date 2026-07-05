"""
Interface comum dos modelos e tipos de resultado (Guia §6.1 (iii), §6.2 ``models``)
==================================================================================
O guia exige **modelos plugáveis**: uma interface única — ``fit``, ``predict``,
``project``, ``diagnostics`` — para ARDL, ARIMAX, fator ``Z``, regressão beta e
os *challengers*, de modo que a comparação *champion-challenger* seja apenas uma
iteração sobre uma lista (§6.1). Este módulo define:

* :class:`Specification` — a descrição declarativa e versionável de um modelo
  (transformação, defasagens macro, ordem autorregressiva, sazonalidade,
  eventos), o "YAML" que torna um estudo reprodutível (§6.1 (i)).
* :class:`SatelliteModel` — a classe-base abstrata com a interface comum e a
  bateria de diagnóstico embutida (§6.1 (ii): *tudo é testado*).
* :class:`FitResult` — o objeto de resultados de um ajuste: coeficientes,
  ajustados/resíduos (nas duas escalas), métricas in-sample e o resultado
  estatístico bruto por baixo.
* :class:`Projection` — a projeção condicional a cenários: média e intervalos por
  cenário, com ponderação para ECL (§5).

Convenção de **escalas**: os modelos estimam na escala do *link* (logit/probit/Z
ou nível); ``*_link`` guarda a escala de modelagem e os campos sem sufixo já vêm
**reconvertidos** à escala original da taxa (``[0, 1]``) — a que interessa ao uso.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # evita ciclo em runtime (diagnostics é importado tardiamente)
    pass


# ======================================================================
# Especificação declarativa
# ======================================================================
@dataclass
class Specification:
    """Descrição versionável de um modelo satélite (o "YAML" do estudo).

    Parameters
    ----------
    exog:
        Defasagens das variáveis macro por nome: ``{"desemprego": [0, 1],
        "renda": [0]}`` inclui o contemporâneo e a defasagem 1 do desemprego e o
        contemporâneo da renda. Vazio ⇒ modelo univariado.
    ar:
        Ordem autorregressiva da dependente (nº de defasagens próprias). ``0``
        desliga o termo AR.
    trend:
        Determinístico: ``"c"`` (constante), ``"ct"`` (constante + tendência),
        ``"n"`` (nenhum).
    link:
        Transformação da dependente: ``"logit"``, ``"probit"``, ``"identity"`` ou
        ``"vasicek"`` (fator ``Z``, tratado pelo modelo :class:`VasicekZ`).
    seasonal, seasonal_period:
        Se ``seasonal``, inclui dummies sazonais de período ``seasonal_period``
        (12 mensal, 4 trimestral).
    events:
        Dummies de evento ``{nome: (inicio, fim) | data | [datas]}`` (§2.3).
    expected_signs:
        Sinal econômico esperado por variável (``+1``/``-1``) — o filtro **duro**
        de seleção (§4.1): coeficiente com sinal trocado é desqualificador.
    name:
        Rótulo do modelo (para relatórios e ranking).
    """

    exog: Mapping[str, Sequence[int]] = field(default_factory=dict)
    ar: int = 1
    trend: str = "c"
    link: str = "logit"
    seasonal: bool = False
    seasonal_period: int = 12
    events: Mapping[str, object] = field(default_factory=dict)
    expected_signs: Mapping[str, int] = field(default_factory=dict)
    name: str = ""

    def variables(self) -> list[str]:
        """Variáveis macro usadas (chaves de ``exog``)."""
        return list(self.exog.keys())

    def n_terms(self) -> int:
        """Nº de termos macro (soma das defasagens sobre as variáveis)."""
        return int(sum(len(v) for v in self.exog.values()))

    def describe(self) -> str:
        """Descrição legível (para rótulos de ranking/relatório)."""
        parts = []
        for var, ls in self.exog.items():
            parts.append(f"{var}[{','.join(map(str, ls))}]")
        macro = " + ".join(parts) if parts else "(univariado)"
        tag = f"AR({self.ar})·{self.link}"
        if self.seasonal:
            tag += f"·saz{self.seasonal_period}"
        return f"{tag}: {macro}"


# ======================================================================
# Resultado de um ajuste
# ======================================================================
@dataclass
class FitResult:
    """Resultado de um ajuste, agnóstico ao motor por baixo.

    Campos ``*_link`` estão na escala de modelagem; ``fitted``/``resid_original``
    estão na escala original da taxa (``[0, 1]``).
    """

    model_name: str
    kind: str
    link: str
    params: pd.Series                    # coeficientes nomeados
    fitted_link: pd.Series               # ajustados na escala do link
    resid: pd.Series                     # resíduos na escala do link (p/ diagnóstico)
    fitted: pd.Series                    # ajustados reconvertidos à escala original
    nobs: int
    n_params: int
    llf: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    rsquared: Optional[float] = None
    sigma: Optional[float] = None        # desvio-padrão residual (escala do link)
    bse: Optional[pd.Series] = None      # erros-padrão
    tvalues: Optional[pd.Series] = None
    pvalues: Optional[pd.Series] = None
    exog: Optional[pd.DataFrame] = None  # design usado (p/ BG/BP/White e VIF)
    spec: Optional[Specification] = None
    raw: object = None                   # resultado estatístico bruto (statsmodels)

    @property
    def resid_original(self) -> pd.Series:
        """Resíduo na escala original (``observado − ajustado``)."""
        # observado_original = fitted + resid, mas na escala do link; reconstruímos
        # o observado original a partir do ajustado e do resíduo original guardado
        # implicitamente: aqui devolvemos observado_orig − fitted quando possível.
        return self._resid_original if self._resid_original is not None else self.fitted * np.nan

    _resid_original: Optional[pd.Series] = field(default=None, repr=False)

    def coef_frame(self) -> pd.DataFrame:
        """Tabela de coeficientes: estimativa, erro-padrão, ``t``, ``p``."""
        out = pd.DataFrame({"coef": self.params})
        if self.bse is not None:
            out["std_err"] = self.bse
        if self.tvalues is not None:
            out["t"] = self.tvalues
        if self.pvalues is not None:
            out["p_valor"] = self.pvalues
        return out

    def metrics(self) -> dict:
        """Métricas in-sample num dicionário (AIC/BIC/logL/R²/σ/RMSE)."""
        rmse = float(np.sqrt(np.mean(self.resid.to_numpy(dtype=float) ** 2)))
        return {
            "nobs": self.nobs, "n_params": self.n_params,
            "logL": self.llf, "AIC": self.aic, "BIC": self.bic,
            "R2": self.rsquared, "sigma": self.sigma, "RMSE_link": rmse,
        }

    def metrics_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.metrics()])

    def diagnostics(self, alpha: float = 0.05) -> pd.DataFrame:
        """Roda a **bateria de diagnóstico de resíduos** (Guia §4.2).

        Ljung-Box e Breusch-Godfrey (autocorrelação), Breusch-Pagan/White
        (heterocedasticidade), Jarque-Bera (normalidade), Durbin-Watson e ARCH-LM.
        Import tardio de :mod:`.diagnostics` (evita ciclo de importação).
        """
        from . import diagnostics as diag  # import tardio

        return diag.residual_report(self.resid, exog=self.exog, alpha=alpha)

    def summary(self) -> pd.DataFrame:
        """Alias para :meth:`coef_frame` (compatível com o estilo dos motores)."""
        return self.coef_frame()

    def __repr__(self) -> str:  # pragma: no cover
        aic = f"{self.aic:.1f}" if self.aic is not None else "?"
        return (f"FitResult(model={self.model_name!r}, kind={self.kind!r}, "
                f"link={self.link!r}, nobs={self.nobs}, AIC={aic})")


# ======================================================================
# Projeção condicional a cenários
# ======================================================================
@dataclass
class Projection:
    """Projeção condicional a um ou mais cenários (Guia §5).

    ``paths`` mapeia ``nome_do_cenário → DataFrame`` indexado pelo horizonte
    futuro, com colunas ``mean``, ``lower``, ``upper`` (escala **original** da
    taxa) e ``mean_link``. ``probabilities`` (se dado) pondera os cenários para o
    ECL forward-looking.
    """

    paths: dict[str, pd.DataFrame]
    kind: str
    link: str
    horizon: int
    alpha: float = 0.10
    probabilities: Optional[Mapping[str, float]] = None

    def scenario_names(self) -> list[str]:
        return list(self.paths.keys())

    def mean_frame(self) -> pd.DataFrame:
        """Média por cenário (escala original) em formato largo (cenários nas colunas)."""
        return pd.DataFrame({name: df["mean"] for name, df in self.paths.items()})

    def weighted(self, probabilities: Optional[Mapping[str, float]] = None) -> pd.Series:
        """Projeção **ponderada** pelos pesos de cenário — o insumo do ECL (§5).

        ``PD_ECL_t = Σ_s w_s · PD_{s,t}``. Os pesos vêm de ``probabilities`` (ou do
        atributo homônimo); devem somar ~1.
        """
        w = dict(probabilities or self.probabilities or {})
        if not w:
            raise ValueError(
                "sem probabilidades de cenário: passe probabilities={cenario: peso}."
            )
        faltando = [s for s in self.paths if s not in w]
        if faltando:
            raise ValueError(f"probabilidades não cobrem os cenários: {faltando}.")
        total = float(sum(w[s] for s in self.paths))
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"as probabilidades devem somar 1; somam {total:.4f}.")
        acc = None
        for s, df in self.paths.items():
            term = w[s] * df["mean"]
            acc = term if acc is None else acc + term
        acc.name = f"{self.kind}_ecl"
        return acc

    def to_frame(self) -> pd.DataFrame:
        """Formato longo: uma linha por (cenário, período) com média e intervalo."""
        blocos = []
        for name, df in self.paths.items():
            b = df.copy()
            b.insert(0, "cenario", name)
            b = b.reset_index().rename(columns={"index": "periodo"})
            blocos.append(b)
        return pd.concat(blocos, ignore_index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Projection(kind={self.kind!r}, cenarios={self.scenario_names()}, "
                f"horizonte={self.horizon})")


# ======================================================================
# Classe-base abstrata
# ======================================================================
class SatelliteModel(abc.ABC):
    """Interface comum de um modelo satélite (Guia §6.1 (iii)).

    Subclasses implementam :meth:`fit`, :meth:`predict` e :meth:`project`. A
    base fornece :meth:`diagnostics`, :meth:`summary` e o protocolo de estado
    (``self.result`` após ``fit``).
    """

    #: nome legível do modelo (subclasses sobrescrevem)
    name: str = "SatelliteModel"

    def __init__(self, kind: str = "pd", link: str = "logit") -> None:
        self.kind = kind
        self.link = link
        self.result: Optional[FitResult] = None

    # -- interface obrigatória ------------------------------------------
    @abc.abstractmethod
    def fit(self) -> FitResult:
        """Estima o modelo e devolve (e armazena em ``self.result``) o :class:`FitResult`."""

    @abc.abstractmethod
    def predict(self, exog_future: Optional[pd.DataFrame] = None,
                steps: Optional[int] = None) -> pd.Series:
        """Previsão in-sample (``exog_future=None``) ou fora da amostra (dado ``exog_future``)."""

    @abc.abstractmethod
    def project(self, scenarios, horizon: Optional[int] = None,
                alpha: float = 0.10, n_sims: int = 2000, seed: int = 0) -> Projection:
        """Projeta condicionalmente a um :class:`ScenarioSet` (Guia §5)."""

    # -- comum ----------------------------------------------------------
    def _require_fit(self) -> FitResult:
        if self.result is None:
            raise RuntimeError(f"{self.name}: chame .fit() antes desta operação.")
        return self.result

    def diagnostics(self, alpha: float = 0.05) -> pd.DataFrame:
        """Bateria de diagnóstico de resíduos do modelo ajustado (Guia §4.2)."""
        return self._require_fit().diagnostics(alpha=alpha)

    def summary(self) -> pd.DataFrame:
        return self._require_fit().coef_frame()

    def __repr__(self) -> str:  # pragma: no cover
        est = "ajustado" if self.result is not None else "não ajustado"
        return f"{self.name}(kind={self.kind!r}, link={self.link!r}, {est})"


# ======================================================================
# Helpers compartilhados
# ======================================================================
def gaussian_llf(resid: np.ndarray, nobs: int) -> float:
    """Log-verossimilhança gaussiana de resíduos (para AIC/BIC de modelos OLS)."""
    resid = np.asarray(resid, dtype=float)
    sigma2 = float(np.mean(resid ** 2))
    if sigma2 <= 0:
        return np.inf
    return float(-0.5 * nobs * (np.log(2 * np.pi) + np.log(sigma2) + 1.0))


def information_criteria(llf: float, nobs: int, n_params: int) -> tuple[float, float]:
    """``(AIC, BIC)`` a partir da log-verossimilhança."""
    aic = -2.0 * llf + 2.0 * n_params
    bic = -2.0 * llf + n_params * np.log(nobs)
    return float(aic), float(bic)


def quantile_bands(sims: np.ndarray, alpha: float = 0.10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Média e banda ``[α/2, 1−α/2]`` de um conjunto de trajetórias simuladas.

    ``sims`` tem forma ``(n_sims, horizonte)``. Devolve ``(media, inferior,
    superior)``, cada um de comprimento ``horizonte`` — o **gráfico em leque** do
    guia (§5, Figura 1).
    """
    sims = np.asarray(sims, dtype=float)
    lo = float(alpha) / 2.0
    hi = 1.0 - lo
    media = sims.mean(axis=0)
    inferior = np.quantile(sims, lo, axis=0)
    superior = np.quantile(sims, hi, axis=0)
    return media, inferior, superior


def build_fit_result(
    *, model_name: str, kind: str, link: str, params: pd.Series,
    fitted_link: pd.Series, resid: pd.Series, observed_original: pd.Series,
    inverse_link, n_params: int, exog: Optional[pd.DataFrame] = None,
    spec: Optional[Specification] = None, raw: object = None,
    llf: Optional[float] = None, aic: Optional[float] = None, bic: Optional[float] = None,
    rsquared: Optional[float] = None, bse: Optional[pd.Series] = None,
    tvalues: Optional[pd.Series] = None, pvalues: Optional[pd.Series] = None,
) -> FitResult:
    """Monta um :class:`FitResult` calculando o que faltar (AIC/BIC/σ/escala original).

    ``inverse_link`` reconverte a escala do link à taxa; ``observed_original`` é a
    taxa observada (para o resíduo na escala original). Quando ``llf/aic/bic`` não
    são dados, usa a aproximação gaussiana sobre os resíduos.
    """
    resid = pd.Series(resid)
    nobs = int(resid.shape[0])
    sigma = float(np.std(resid.to_numpy(dtype=float), ddof=1)) if nobs > 1 else np.nan
    if llf is None:
        llf = gaussian_llf(resid.to_numpy(dtype=float), nobs)
    if aic is None or bic is None:
        aic_c, bic_c = information_criteria(llf, nobs, n_params)
        aic = aic if aic is not None else aic_c
        bic = bic if bic is not None else bic_c
    fitted_orig = pd.Series(np.asarray(inverse_link(fitted_link), dtype=float),
                            index=fitted_link.index, name=f"{kind}_ajustado")
    if rsquared is None:
        y = observed_original.reindex(fitted_orig.index).to_numpy(dtype=float)
        yhat = fitted_orig.to_numpy(dtype=float)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        rsquared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    resid_orig = (observed_original.reindex(fitted_orig.index) - fitted_orig)
    resid_orig.name = f"{kind}_resid"
    fr = FitResult(
        model_name=model_name, kind=kind, link=link, params=pd.Series(params),
        fitted_link=pd.Series(fitted_link), resid=resid, fitted=fitted_orig,
        nobs=nobs, n_params=n_params, llf=llf, aic=aic, bic=bic, rsquared=rsquared,
        sigma=sigma, bse=bse, tvalues=tvalues, pvalues=pvalues, exog=exog, spec=spec, raw=raw,
    )
    fr._resid_original = resid_orig
    return fr


__all__ = [
    "Specification",
    "FitResult",
    "Projection",
    "SatelliteModel",
    "gaussian_llf",
    "information_criteria",
    "quantile_bands",
    "build_fit_result",
]
