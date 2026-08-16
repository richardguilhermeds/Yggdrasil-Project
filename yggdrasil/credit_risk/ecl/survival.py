"""
Motores de sobrevivência: Kaplan-Meier e *hazard* em tempo discreto
===================================================================
Duas técnicas da família que a literatura de estrutura a termo sob IFRS 9 trata
como referência para PD *lifetime* — porque são as únicas que tratam **censura à
direita** de forma correta e que separam *quando* o risco acontece de *quanto*
risco existe.

:func:`kaplan_meier`
    O estimador **produto-limite**, não paramétrico: a curva de sobrevivência
    sai da contagem de quebras sobre a base em risco de cada idade, sem hipótese
    funcional nenhuma. É o retrato honesto do que a carteira mostrou, com o erro
    padrão de **Greenwood** e o intervalo de confiança log-log (que respeita os
    limites ``[0, 1]``). Serve de referência para julgar qualquer curva
    paramétrica: se a paramétrica se afasta do KM em idades com base grande, o
    problema é do modelo.

:class:`DiscreteHazard`
    A regressão de *hazard* em **tempo discreto**: expande o painel em
    pessoa-período e ajusta um modelo binário para "quebrou neste período, dado
    que chegou vivo nele". A idade entra como *baseline* (dummies, spline ou
    forma paramétrica) e as covariáveis deslocam essa linha de base. Duas
    virtudes práticas: (i) a curva passa a ser **por contrato** — cada perfil de
    risco tem a sua estrutura a termo, o que é o que o ECL por contrato exige; e
    (ii) covariáveis que mudam no tempo entram naturalmente, porque cada linha do
    painel é uma observação.

    O *link* padrão é o **logit** (via ``scikit-learn``, já no núcleo do
    pacote). O ``cloglog`` — o análogo em tempo discreto do modelo de riscos
    proporcionais de Cox — está disponível via ``statsmodels`` e é carregado
    **sob demanda**: sem o extra ``[econometric]`` instalado, o erro diz o que
    fazer em vez de quebrar o import do pacote.

Ambos devolvem uma :class:`~yggdrasil.credit_risk.ecl.curves.PDCurve` — a mesma
estrutura que o resto do subpacote consome.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .curves import PDCurve
from .panel import ContractPanel

#: *Links* suportados pela regressão de *hazard*.
HAZARD_LINKS = ("logit", "cloglog")

#: Formas do *baseline* em idade.
BASELINES = ("dummies", "spline", "linear", "log")


# ======================================================================
# Kaplan-Meier (produto-limite)
# ======================================================================
def kaplan_meier(
    panel: ContractPanel,
    from_age: int = 0,
    horizon: Optional[int] = None,
    weighted: bool = False,
    alpha: float = 0.05,
    label: str = "",
    return_table: bool = False,
) -> Union[PDCurve, Tuple[PDCurve, pd.DataFrame]]:
    """Estimador produto-limite de Kaplan-Meier sobre o painel.

    ``S(t) = Π_{k ≤ t} (1 − d_k / n_k)``, com ``d_k`` quebras e ``n_k`` a base em
    risco da idade ``k``. Em tempo discreto o *hazard* estimado coincide com o da
    curva de safra (:func:`~yggdrasil.credit_risk.ecl.curves.vintage_curve` com
    ``fill='zero'``) — a diferença está na **incerteza**, que aqui vem do erro
    padrão de Greenwood:

    ``Var[S(t)] ≈ S(t)² · Σ_{k ≤ t} d_k / (n_k (n_k − d_k))``

    e do IC **log-log**, que não estoura ``[0, 1]`` como o IC linear faria nas
    caudas.

    Parameters
    ----------
    panel:
        O painel de contratos.
    from_age:
        Idade de partida da curva (``0`` = curva de originação).
    horizon:
        Nº de períodos. ``None`` vai até a última idade observada.
    weighted:
        Pondera por exposição (exige ``exposure_col``). Nesse caso o IC de
        Greenwood é apenas indicativo — a variância binomial pressupõe contagem.
    alpha:
        Nível de significância do IC (``0.05`` → 95%).
    return_table:
        Devolve ``(curva, tabela)`` com a tabela de vida completa.

    Returns
    -------
    PDCurve | tuple[PDCurve, pandas.DataFrame]
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha deve estar em (0, 1); recebido {alpha!r}.")
    vida = panel.at_risk(weighted=weighted)
    vida = vida[vida.index >= int(from_age)]
    if vida.empty:
        raise ValueError(
            f"nenhuma idade >= from_age={from_age} no painel (idade máxima: {panel.max_age})."
        )
    if horizon is not None:
        vida = vida.iloc[: int(horizon)]

    n = vida["n_em_risco"].to_numpy(dtype=float)
    d = vida["n_default"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(n > 0, d / n, 0.0)
    h = np.clip(np.nan_to_num(h, nan=0.0), 0.0, 1.0)

    curva = PDCurve(h, label=label, freq=panel.freq,
                    meta={"metodo": "kaplan_meier", "from_age": int(from_age),
                          "weighted": bool(weighted), "n_contratos": panel.n_contracts})
    if horizon is not None and len(curva) < int(horizon):
        curva = curva.extend(int(horizon))
    if not return_table:
        return curva

    # --- Greenwood + IC log-log --------------------------------------
    from scipy.stats import norm

    s = np.cumprod(1.0 - h)
    with np.errstate(divide="ignore", invalid="ignore"):
        parcela = np.where((n > 0) & (n - d > 0), d / (n * (n - d)), 0.0)
    soma = np.cumsum(parcela)
    se_s = s * np.sqrt(soma)                       # erro padrão de Greenwood

    z = norm.ppf(1.0 - alpha / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # IC log-log: c = ln(−ln S) ± z · SE[ln(−ln S)]
        log_s = np.log(np.clip(s, 1e-12, 1.0))
        se_loglog = np.sqrt(soma) / np.abs(log_s)
        fator = np.exp(z * se_loglog)
        s_inf = np.clip(s ** fator, 0.0, 1.0)
        s_sup = np.clip(s ** (1.0 / fator), 0.0, 1.0)
    s_inf = np.where(s >= 1.0, 1.0, s_inf)
    s_sup = np.where(s >= 1.0, 1.0, s_sup)

    tabela = vida.copy()
    tabela["horizonte"] = np.arange(1, len(tabela) + 1)
    tabela["sobrevivencia"] = s
    tabela["se_greenwood"] = se_s
    tabela["sobrevivencia_ic_inf"] = s_inf
    tabela["sobrevivencia_ic_sup"] = s_sup
    tabela["pd_acumulada"] = 1.0 - s
    tabela["pd_acumulada_ic_inf"] = 1.0 - s_sup
    tabela["pd_acumulada_ic_sup"] = 1.0 - s_inf
    return curva, tabela


# ======================================================================
# Regressão de hazard em tempo discreto
# ======================================================================
class DiscreteHazard:
    """Regressão de *hazard* em tempo discreto sobre o painel pessoa-período.

    Modela ``P(quebra em t | vivo em t) = g⁻¹( f(idade) + xβ )``, onde
    ``f(idade)`` é o *baseline* e ``x`` as covariáveis. Cada linha do painel é
    uma observação binária; a censura entra sozinha (o contrato simplesmente
    deixa de ter linhas), sem tratamento especial.

    Parameters
    ----------
    baseline:
        Forma da dependência em idade: ``'dummies'`` (padrão — uma indicadora por
        idade, totalmente flexível), ``'spline'`` (B-spline em idade, suaviza a
        cauda onde a base rareia), ``'linear'`` ou ``'log'`` (``ln(1 + idade)``,
        a forma paramétrica mais econômica).
    link:
        ``'logit'`` (padrão, ``scikit-learn``) ou ``'cloglog'`` (o análogo
        discreto de Cox; exige ``statsmodels`` — extra ``[econometric]``).
    n_knots, degree:
        Parâmetros do ``'spline'``.
    C:
        Inverso da força de regularização L2 do logit. Padrão ``1e6``
        (praticamente sem penalização — a estimativa fica de máxima
        verossimilhança). Baixe quando houver idades com pouquíssima base.
    max_age:
        Trunca o painel nesta idade antes de ajustar. Útil para não deixar a
        cauda rala dominar o *baseline* de dummies.

    Examples
    --------
    >>> mh = DiscreteHazard(baseline="spline").fit(painel, features=["feat_score"])
    >>> mh.baseline_curve(horizon=60).pd_12m()
    >>> curvas = mh.predict_curves(carteira, horizon=60)      # (n_contratos, 60)
    """

    def __init__(self, baseline: str = "dummies", link: str = "logit",
                 n_knots: int = 6, degree: int = 3, C: float = 1e6,
                 max_age: Optional[int] = None) -> None:
        if baseline not in BASELINES:
            raise ValueError(f"baseline deve ser um de {BASELINES}; recebido {baseline!r}.")
        if link not in HAZARD_LINKS:
            raise ValueError(f"link deve ser um de {HAZARD_LINKS}; recebido {link!r}.")
        self.baseline = baseline
        self.link = link
        self.n_knots = int(n_knots)
        self.degree = int(degree)
        self.C = float(C)
        self.max_age = max_age

        self.features: List[str] = []
        self.freq: str = "M"
        self.ages_: np.ndarray = np.array([])
        self.feature_means_: np.ndarray = np.array([])
        self._spline = None
        self._model = None
        self._coef: Optional[np.ndarray] = None
        self._intercept: float = 0.0
        self._design_names: List[str] = []

    # -- desenho da matriz ----------------------------------------------
    def _baseline_matrix(self, ages: np.ndarray, fit: bool = False) -> Tuple[np.ndarray, List[str]]:
        a = np.asarray(ages, dtype=float).reshape(-1, 1)
        if self.baseline == "dummies":
            if fit:
                self.ages_ = np.unique(np.asarray(ages, dtype=int))
            grade = self.ages_
            # Primeira idade é a referência (absorvida pelo intercepto).
            cols = grade[1:]
            m = (np.asarray(ages, dtype=int).reshape(-1, 1) == cols.reshape(1, -1)).astype(float)
            return m, [f"idade={int(c)}" for c in cols]
        if self.baseline == "spline":
            from sklearn.preprocessing import SplineTransformer

            if fit:
                self._spline = SplineTransformer(
                    n_knots=self.n_knots, degree=self.degree, include_bias=False
                ).fit(a)
            m = self._spline.transform(a)
            return np.asarray(m, dtype=float), [f"spline_{i}" for i in range(m.shape[1])]
        if self.baseline == "linear":
            return a, ["idade"]
        return np.log1p(a), ["ln(1+idade)"]

    def _design(self, ages, X: Optional[np.ndarray], fit: bool = False) -> np.ndarray:
        base, nomes = self._baseline_matrix(np.asarray(ages), fit=fit)
        if X is None or (isinstance(X, np.ndarray) and X.size == 0):
            if fit:
                self._design_names = nomes
            return base
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        if Xa.shape[0] != base.shape[0]:
            raise ValueError(
                f"X tem {Xa.shape[0]} linhas e a idade tem {base.shape[0]} — devem casar."
            )
        if fit:
            self._design_names = nomes + list(self.features)
        return np.hstack([base, Xa])

    # -- ajuste -----------------------------------------------------------
    def fit(self, panel: ContractPanel, features: Optional[Sequence[str]] = None) -> "DiscreteHazard":
        """Ajusta o modelo sobre o painel pessoa-período.

        ``features`` são colunas do painel usadas como covariáveis (numéricas;
        categóricas devem vir já codificadas — o pacote não inventa codificação
        aqui, para que a política de categorização continue sendo a do
        :class:`~yggdrasil.credit_risk.model.ModelSegmenter`)."""
        self.features = list(features or [])
        self.freq = panel.freq
        d = panel.spells(features=self.features, max_age=self.max_age)
        idade = d[panel.age_col].to_numpy(dtype=int)
        y = d[panel.default_col].to_numpy(dtype=int)
        if y.sum() == 0:
            raise ValueError("o painel não tem nenhum default — nada a ajustar.")
        if len(np.unique(y)) < 2:
            raise ValueError("o painel só tem uma classe no evento de default.")

        X = d[self.features].to_numpy(dtype=float) if self.features else None
        if X is not None:
            if not np.all(np.isfinite(X)):
                raise ValueError("há NaN/inf nas features — trate os faltantes antes de ajustar.")
            self.feature_means_ = X.mean(axis=0)
        M = self._design(idade, X, fit=True)

        if self.link == "logit":
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(C=self.C, solver="lbfgs", max_iter=2000)
            lr.fit(M, y)
            self._model = lr
            self._coef = lr.coef_.ravel().astype(float)
            self._intercept = float(lr.intercept_[0])
        else:  # cloglog — statsmodels, sob demanda
            try:
                import statsmodels.api as sm
            except ImportError as e:  # pragma: no cover - depende do ambiente
                raise ImportError(
                    "link='cloglog' exige statsmodels. Instale o extra: "
                    'pip install "yggdrasil-project[econometric]" '
                    "(ou use link='logit', que roda no núcleo)."
                ) from e
            Mc = sm.add_constant(M, has_constant="add")
            glm = sm.GLM(y, Mc, family=sm.families.Binomial(sm.families.links.CLogLog()))
            res = glm.fit()
            self._model = res
            params = np.asarray(res.params, dtype=float)
            self._intercept = float(params[0])
            self._coef = params[1:]
        return self

    def _check_fit(self) -> None:
        if self._coef is None:
            raise RuntimeError("o modelo ainda não foi ajustado — chame .fit(painel) antes.")

    # -- predição ----------------------------------------------------------
    def _inverse_link(self, eta: np.ndarray) -> np.ndarray:
        if self.link == "logit":
            return 1.0 / (1.0 + np.exp(-eta))
        return 1.0 - np.exp(-np.exp(np.clip(eta, -50.0, 50.0)))   # cloglog

    def predict_hazard(self, ages, X=None) -> np.ndarray:
        """*Hazard* previsto para pares (idade, covariáveis) — vetor de ``[0, 1]``."""
        self._check_fit()
        eta = self._intercept + self._design(np.asarray(ages), X) @ self._coef
        return np.clip(self._inverse_link(eta), 0.0, 1.0)

    def _ages_grid(self, from_age: int, horizon: int) -> np.ndarray:
        grade = np.arange(int(from_age), int(from_age) + int(horizon))
        if self.baseline == "dummies" and self.ages_.size:
            # Idades fora da grade ajustada caem na última idade conhecida
            # (extrapolação plana explícita, em vez de virarem a referência).
            grade = np.clip(grade, self.ages_.min(), self.ages_.max())
        return grade

    def predict_curve(self, x=None, from_age: int = 0, horizon: int = 60,
                      label: str = "") -> PDCurve:
        """Curva de um contrato: covariáveis ``x`` fixas, idade correndo.

        ``x`` é um vetor com um valor por feature, na ordem de ``self.features``.
        ``None`` usa a **média** amostral de cada feature — a curva "do contrato
        médio", que é o *baseline* interpretável."""
        self._check_fit()
        grade = self._ages_grid(from_age, horizon)
        if self.features:
            vetor = self.feature_means_ if x is None else np.asarray(x, dtype=float).ravel()
            if vetor.size != len(self.features):
                raise ValueError(
                    f"x deve ter {len(self.features)} valores (as features do ajuste); "
                    f"recebido {vetor.size}."
                )
            X = np.tile(vetor, (len(grade), 1))
        else:
            X = None
        h = self.predict_hazard(grade, X)
        return PDCurve(h, label=label, freq=self.freq,
                       meta={"metodo": "discrete_hazard", "link": self.link,
                             "baseline": self.baseline, "from_age": int(from_age)})

    def baseline_curve(self, from_age: int = 0, horizon: int = 60) -> PDCurve:
        """Curva no ponto médio das covariáveis — a referência do modelo."""
        return self.predict_curve(None, from_age=from_age, horizon=horizon, label="baseline")

    def predict_curves(self, df: pd.DataFrame, horizon: int = 60,
                       age_col: Optional[str] = None) -> np.ndarray:
        """*Hazards* de vários contratos de uma vez → matriz ``(n_linhas, horizon)``.

        ``age_col`` dá a idade **atual** de cada contrato (a curva começa dali).
        Sem ela, todos partem da idade 0."""
        self._check_fit()
        n = len(df)
        idade0 = (df[age_col].to_numpy(dtype=int) if age_col else np.zeros(n, dtype=int))
        if self.features:
            faltando = [c for c in self.features if c not in df.columns]
            if faltando:
                raise ValueError(f"Features ausentes em df: {faltando}.")
            X0 = df[self.features].to_numpy(dtype=float)
        else:
            X0 = None

        out = np.empty((n, int(horizon)), dtype=float)
        for j in range(int(horizon)):
            grade = idade0 + j
            if self.baseline == "dummies" and self.ages_.size:
                grade = np.clip(grade, self.ages_.min(), self.ages_.max())
            out[:, j] = self.predict_hazard(grade, X0)
        return out

    # -- inspeção ----------------------------------------------------------
    def coef_frame(self) -> pd.DataFrame:
        """Coeficientes do modelo, com o intercepto na primeira linha."""
        self._check_fit()
        nomes = ["(intercepto)"] + list(self._design_names)
        valores = np.concatenate([[self._intercept], self._coef])
        out = pd.DataFrame({"termo": nomes, "coeficiente": valores})
        if self.link == "logit":
            out["odds_ratio"] = np.exp(out["coeficiente"])
        return out

    def __repr__(self) -> str:
        estado = "ajustado" if self._coef is not None else "não ajustado"
        return (f"DiscreteHazard(baseline={self.baseline!r}, link={self.link!r}, "
                f"features={len(self.features)}, {estado})")


__all__ = ["kaplan_meier", "DiscreteHazard", "HAZARD_LINKS", "BASELINES"]
