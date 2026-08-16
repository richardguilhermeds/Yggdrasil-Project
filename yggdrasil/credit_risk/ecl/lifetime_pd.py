"""
A fachada: :class:`LifetimePD` — uma classe, cinco motores, uma curva
=====================================================================
Este é o ponto de entrada do eixo *lifetime*. Segue o padrão que o resto do
pacote já usa (``task_type`` nos segmentadores): **uma** classe, e o
comportamento escolhido por argumento — ``method=``. Assim quem troca de método
não troca de API, e a comparação entre métodos vira uma linha de código.

=================  =========================================================
``method``         o que faz
=================  =========================================================
``"constant"``     *hazard* constante a partir da PD de 12 meses. A linha de
                   base: sem maturação, sem painel, sem desculpa para não ter
                   uma curva.
``"vintage"``      taxa marginal observada **por idade**, com a base em risco
                   recontada período a período (censura tratada).
``"km"``           Kaplan-Meier — mesmo ponto estimado da safra, mais o erro
                   padrão de Greenwood e o IC log-log.
``"hazard"``       regressão de *hazard* em tempo discreto: curva **por
                   contrato**, com covariáveis.
``"markov"``       cadeia de Markov sobre a matriz de transição: curva **por
                   rating**, com o *default* absorvente.
``"survival"``     apelido que resolve em ``"hazard"`` quando há ``features`` e
                   em ``"km"`` quando não há.
=================  =========================================================

Três coisas que a fachada faz além de embrulhar os motores:

**Curvas por grupo.** ``fit(painel, by="segmento")`` devolve uma curva por
segmento e todo o resto (``apply``, ``backtest``, ``frame``) passa a operar por
grupo, sem laço na mão.

**A ponte com o eixo transversal.** :meth:`LifetimePD.calibrate_to` recalibra o
**nível** de cada curva para bater com a PD de 12 meses que saiu do modelo de
escoragem, preservando o **formato** da maturação. É o desenho usual: o modelo
por cliente ordena e dá o nível; a curva empírica dá o tempo.

**O forward-looking.** :meth:`LifetimePD.condition` desloca a curva ao ciclo pelo
arcabouço de Vasicek, com a convenção de sinal do repositório (``z > 0`` = ciclo
benigno). Aceita ``z`` escalar, um ``z`` por horizonte (basta ligar a projeção de
um modelo satélite) ou um ``z`` inicial com **reversão à média**, que é como o
efeito de ciclo costuma ser tratado num horizonte *lifetime*: forte nos primeiros
períodos, dissipando depois.

O padrão é o **deslocamento puro do limiar**,
``PD_PIT = Φ(Φ⁻¹(PD_TTC) − √ρ·z)`` — a mesma escolha (e a mesma justificativa) de
:func:`~yggdrasil.credit_risk.capital.migration.zshift_transition_matrix`, que
garante idempotência em ``z = 0`` e faz o caminho da curva concordar com o
caminho da matriz de migração. A lei condicional exata do fator único, com o
reescalonamento por ``1/√(1−ρ)``, está em ``mode='conditional'`` e é a que fecha
com :func:`~yggdrasil.credit_risk.capital.asrf.conditional_pd` e com a inversa
:func:`~yggdrasil.credit_risk.econometric.transforms.vasicek_z`. A escolha está
documentada em :func:`pit_from_ttc` — as duas são úteis, e trocá-las por engano
muda o nível da provisão.

A fórmula é implementada aqui sobre ``scipy`` — de propósito. O subpacote
:mod:`yggdrasil.credit_risk.econometric`, que traz a mesma transformação, importa
``statsmodels`` no seu ``__init__``, e o condicionamento ao ciclo é básico demais
para exigir um extra.
"""
from __future__ import annotations

import json
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from .curves import PDCurve, constant_hazard, curve_frame, vintage_curve
from .markov import MarkovPD
from .panel import ContractPanel, periods_per_year
from .survival import DiscreteHazard, kaplan_meier

#: Motores disponíveis (``"survival"`` é apelido resolvido no ``fit``).
METHODS = ("constant", "vintage", "km", "hazard", "markov", "survival")

#: Métodos cuja curva é indexada por **idade** — em ``apply`` a curva de um
#: contrato vivo começa na idade atual dele, não no horizonte 1.
AGE_INDEXED = ("vintage", "km")


# ======================================================================
# Condicionamento ao ciclo (Vasicek)
# ======================================================================
def pit_from_ttc(pd_ttc, rho: float, z, mode: str = "shift") -> np.ndarray:
    """Converte PD *through-the-cycle* em *point-in-time* pelo arcabouço de Vasicek.

    **Convenção de sinal** (a mesma de
    :func:`~yggdrasil.credit_risk.econometric.transforms.vasicek_z` e de
    :func:`~yggdrasil.credit_risk.capital.migration.zshift_transition_matrix`):
    ``z > 0`` = ciclo **benigno** (PD cai), ``z < 0`` = adverso (PD sobe).

    Duas leituras do mesmo arcabouço, e a diferença **importa**:

    ``mode='shift'`` (padrão) — deslocamento puro do limiar
        ``PD_PIT = Φ( Φ⁻¹(PD_TTC) − √ρ · z )``

        Mantém a escala unitária do retorno latente e é **idempotente em
        ``z = 0``**: sem ciclo, a curva não muda. É exatamente a escolha (e a
        justificativa) de
        :func:`~yggdrasil.credit_risk.capital.migration.zshift_transition_matrix`,
        e é o que faz o caminho da curva e o caminho da matriz de migração
        concordarem dentro de :class:`LifetimePD`.

    ``mode='conditional'`` — lei condicional exata do fator único
        ``PD_PIT = Φ( (Φ⁻¹(PD_TTC) − √ρ · z) / √(1 − ρ) )``

        A PD condicional a ``Z = z`` do modelo ASRF, com o reescalonamento por
        ``1/√(1−ρ)``. **Não** é idempotente em ``z = 0`` — e não deveria ser: a
        PD condicional à realização *mediana* do fator é maior que a PD média
        (para ``PD < 0,5``), porque a média é tomada sobre toda a distribuição de
        ``Z``. É a mesma fórmula de
        :func:`~yggdrasil.credit_risk.capital.asrf.conditional_pd` e de
        :func:`~yggdrasil.credit_risk.econometric.transforms.default_rate_from_z`,
        valendo a identidade
        ``pit_from_ttc(p, ρ, −Φ⁻¹(q), mode='conditional') == conditional_pd(p, ρ, q)``.
        Use quando o ``z`` vier de :func:`~...econometric.transforms.vasicek_z`,
        que é a **inversa exata** desta forma — só assim a ida e a volta fecham.

    Parameters
    ----------
    pd_ttc:
        PD (escalar ou array) em ``[0, 1]``.
    rho:
        Correlação de ativos em ``[0, 1)``. Com ``ρ = 0`` o ciclo não carrega e a
        PD não muda em nenhum dos modos.
    z:
        Fator sistêmico (escalar ou array *broadcastável* com ``pd_ttc``).
    mode:
        ``'shift'`` ou ``'conditional'``.
    """
    if mode not in ("shift", "conditional"):
        raise ValueError(f"mode deve ser 'shift' ou 'conditional'; recebido {mode!r}.")
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho deve estar em [0, 1); recebido {rho!r}.")
    if rho == 0.0:
        return np.asarray(pd_ttc, dtype=float)
    p = np.clip(np.asarray(pd_ttc, dtype=float), 1e-12, 1.0 - 1e-12)
    eta = norm.ppf(p) - np.sqrt(rho) * np.asarray(z, dtype=float)
    if mode == "conditional":
        eta = eta / np.sqrt(1.0 - rho)
    return norm.cdf(eta)


def _z_path(z, horizon: int, decay: Optional[float] = None) -> np.ndarray:
    """Trajetória de ``z`` ao longo do horizonte.

    Escalar → constante (ou revertendo a zero à taxa ``decay`` por período);
    sequência → usada como veio, estendida pelo último valor se for curta."""
    arr = np.atleast_1d(np.asarray(z, dtype=float))
    if arr.size == 1:
        z0 = float(arr[0])
        if decay is None:
            return np.full(horizon, z0)
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"decay deve estar em [0, 1]; recebido {decay!r}.")
        return z0 * (1.0 - decay) ** np.arange(horizon)
    if arr.size < horizon:
        arr = np.concatenate([arr, np.full(horizon - arr.size, arr[-1])])
    return arr[:horizon]


# ======================================================================
# A fachada
# ======================================================================
class LifetimePD:
    """Estrutura a termo de PD por contrato — a fachada dos motores *lifetime*.

    Parameters
    ----------
    method:
        Um de :data:`METHODS`.
    horizon:
        Horizonte da curva em períodos (padrão ``60``). É o teto; o prazo
        remanescente de cada contrato trunca em :meth:`apply`.
    engine_kwargs:
        Repassados ao motor: ``min_at_risk``/``fill``/``weighted`` (safra),
        ``alpha`` (KM), ``baseline``/``link``/``C`` (*hazard*),
        ``method``/``smoothing``/``default_state`` (Markov).

    Attributes
    ----------
    curves_:
        ``{rótulo: PDCurve}``. Sem ``by``, a única curva fica sob a chave
        :data:`GLOBAL`.

    Examples
    --------
    >>> lt = LifetimePD(method="vintage", horizon=60).fit(painel, by="produto")
    >>> lt.curve("consignado").to_frame().head()
    >>> lt.calibrate_to({"consignado": 0.021, "cartao": 0.078})
    >>> lt.apply(carteira, age_col="idade", term_col="prazo")
    """

    #: Rótulo da curva única quando o ajuste não é por grupo.
    GLOBAL = "__global__"

    def __init__(self, method: str = "vintage", horizon: int = 60, **engine_kwargs) -> None:
        if method not in METHODS:
            raise ValueError(f"method deve ser um de {METHODS}; recebido {method!r}.")
        if int(horizon) < 1:
            raise ValueError(f"horizon deve ser >= 1; recebido {horizon!r}.")
        self.method = method
        self.method_ = method            # resolvido no fit (apelido "survival")
        self.horizon = int(horizon)
        self.engine_kwargs = dict(engine_kwargs)

        self.curves_: Dict[object, PDCurve] = {}
        self.by: Optional[str] = None
        self.features: List[str] = []
        self.freq: str = "M"
        self.hazard_models_: Dict[object, DiscreteHazard] = {}
        self.markov_: Optional[MarkovPD] = None
        self.tables_: Dict[object, pd.DataFrame] = {}
        # Ajustes aplicados às curvas depois do ajuste (calibração, ciclo). Só o
        # motor 'hazard' precisa deles: nos demais motores as curvas já são a
        # fonte de verdade de `marginal_matrix`, mas ali a matriz vem do modelo
        # por contrato e precisa receber a MESMA transformação — senão a carteira
        # sairia escorada com a curva não calibrada.
        self.adjustments_: List[dict] = []
        self.meta: dict = {}

    # ------------------------------------------------------------------
    # Construtores diretos (sem painel)
    # ------------------------------------------------------------------
    @classmethod
    def from_pd_12m(cls, pd_12m: Union[float, Mapping], horizon: int = 60,
                    freq: str = "M") -> "LifetimePD":
        """Curva(s) de *hazard* constante a partir da PD de 12 meses — sem painel.

        Aceita escalar (curva única) ou ``{segmento: pd_12m}``."""
        obj = cls(method="constant", horizon=horizon)
        obj.freq = freq
        obj.method_ = "constant"
        alvos = pd_12m if isinstance(pd_12m, Mapping) else {cls.GLOBAL: pd_12m}
        obj.curves_ = {
            k: constant_hazard(float(v), horizon, freq=freq, label=str(k))
            for k, v in alvos.items()
        }
        obj.by = None if not isinstance(pd_12m, Mapping) else "informado"
        obj.meta = {"origem": "from_pd_12m"}
        return obj

    @classmethod
    def from_curves(cls, curves: Mapping[object, PDCurve], method: str = "vintage") -> "LifetimePD":
        """Empacota curvas já construídas (ex.: vindas de outra fonte)."""
        obj = cls(method=method, horizon=max(len(c) for c in curves.values()))
        obj.curves_ = dict(curves)
        obj.method_ = method
        obj.freq = next(iter(curves.values())).freq
        obj.by = None if set(curves) == {cls.GLOBAL} else "informado"
        obj.meta = {"origem": "from_curves"}
        return obj

    # ------------------------------------------------------------------
    # Ajuste
    # ------------------------------------------------------------------
    def fit(self, panel: ContractPanel, by: Optional[str] = None,
            features: Optional[Sequence[str]] = None,
            rating_col: Optional[str] = None) -> "LifetimePD":
        """Ajusta o motor sobre o painel.

        Parameters
        ----------
        panel:
            O :class:`~yggdrasil.credit_risk.ecl.panel.ContractPanel`.
        by:
            Coluna de agrupamento (produto, segmento, rating). ``None`` ajusta
            uma curva única. Ignorado em ``method='markov'``, onde o agrupamento
            **é** o rating.
        features:
            Covariáveis para ``method='hazard'`` (e o que resolve o apelido
            ``'survival'``).
        rating_col:
            Coluna de rating para ``method='markov'``. Sem ela, usa o
            ``segment_col`` do painel.
        """
        if not isinstance(panel, ContractPanel):
            raise TypeError("panel deve ser um ContractPanel.")
        self.by = by
        self.features = list(features or [])
        self.freq = panel.freq
        self.curves_, self.tables_, self.hazard_models_ = {}, {}, {}

        metodo = self.method
        if metodo == "survival":
            metodo = "hazard" if self.features else "km"
        self.method_ = metodo

        if metodo == "markov":
            self.markov_ = MarkovPD(**{k: v for k, v in self.engine_kwargs.items()
                                       if k in ("method", "smoothing", "ratings", "default_state")}
                                    ).fit(panel, rating_col=rating_col)
            self.curves_ = self.markov_.curves(horizon=self.horizon)
            self.by = rating_col or panel.segment_col
        elif metodo == "hazard":
            if not self.features:
                raise ValueError(
                    "method='hazard' exige features (use 'km'/'vintage' para a curva "
                    "não paramétrica)."
                )
            kw = {k: v for k, v in self.engine_kwargs.items()
                  if k in ("baseline", "link", "n_knots", "degree", "C", "max_age")}
            # Um modelo POR GRUPO — mesmo sentido de `by` dos demais motores: o
            # grupo tem curva própria porque foi estimado nos dados dele, e não
            # por avaliar um modelo comum na média das features do grupo.
            partes = {self.GLOBAL: panel} if by is None else panel.by(by)
            for rot, parte in partes.items():
                modelo = DiscreteHazard(**kw).fit(parte, features=self.features)
                self.hazard_models_[rot] = modelo
                self.curves_[rot] = modelo.baseline_curve(horizon=self.horizon)
                self.curves_[rot].label = "" if rot == self.GLOBAL else str(rot)
        else:
            partes = {self.GLOBAL: panel} if by is None else panel.by(by)
            for rot, parte in partes.items():
                self.curves_[rot] = self._fit_uma(parte, rot)

        if not self.curves_:
            raise ValueError("o ajuste não produziu nenhuma curva.")
        self.meta = {"metodo": self.method_, "by": self.by, "horizon": self.horizon,
                     "freq": self.freq, "n_curvas": len(self.curves_),
                     "n_contratos": panel.n_contracts, "features": list(self.features)}
        return self

    def _fit_uma(self, parte: ContractPanel, chave) -> PDCurve:
        """Ajusta o motor não paramétrico (ou constante) num único grupo.

        ``chave`` é o rótulo bruto do grupo — a mesma chave usada em
        ``curves_`` e em ``tables_``, para que as duas coleções casem."""
        rotulo = "" if chave == self.GLOBAL else str(chave)
        if self.method_ == "constant":
            base = vintage_curve(parte, horizon=self.horizon,
                                 **{k: v for k, v in self.engine_kwargs.items()
                                    if k in ("min_at_risk", "fill", "weighted")})
            return constant_hazard(base.pd_12m(), self.horizon, freq=parte.freq, label=rotulo)
        if self.method_ == "vintage":
            kw = {k: v for k, v in self.engine_kwargs.items()
                  if k in ("from_age", "weighted", "min_at_risk", "fill", "alpha")}
            curva, tabela = vintage_curve(parte, horizon=self.horizon, label=rotulo,
                                          return_table=True, **kw)
        else:
            kw = {k: v for k, v in self.engine_kwargs.items()
                  if k in ("from_age", "weighted", "alpha")}
            curva, tabela = kaplan_meier(parte, horizon=self.horizon, label=rotulo,
                                         return_table=True, **kw)
        self.tables_[chave] = tabela
        return curva

    def _check_fit(self) -> None:
        if not self.curves_:
            raise RuntimeError("nenhuma curva construída — chame .fit(painel) antes.")

    # ------------------------------------------------------------------
    # Leitura
    # ------------------------------------------------------------------
    def curve(self, label=None) -> PDCurve:
        """A curva de um grupo. Com curva única, ``label`` pode ser omitido."""
        self._check_fit()
        if label is None:
            if len(self.curves_) == 1:
                return next(iter(self.curves_.values()))
            raise ValueError(
                f"há {len(self.curves_)} curvas — informe o rótulo "
                f"({sorted(map(str, self.curves_))[:6]})."
            )
        if label not in self.curves_:
            raise KeyError(f"curva {label!r} não encontrada; disponíveis: {sorted(map(str, self.curves_))}.")
        return self.curves_[label]

    def frame(self, kind: str = "cumulative") -> pd.DataFrame:
        """As curvas lado a lado (uma coluna por grupo)."""
        self._check_fit()
        return curve_frame(self.curves_, kind=kind)

    def summary(self) -> pd.DataFrame:
        """Uma linha por curva: horizonte, PD de 12 meses e PD *lifetime*."""
        self._check_fit()
        n_ano = periods_per_year(self.freq)
        linhas = []
        for rot, c in self.curves_.items():
            linhas.append({
                "grupo": rot, "metodo": self.method_, "horizonte": len(c),
                "pd_12m": c.pd_12m(),
                f"pd_{min(2 * n_ano, len(c))}p": c.pd_lifetime(min(2 * n_ano, len(c))),
                "pd_lifetime": c.pd_lifetime(),
            })
        return pd.DataFrame(linhas)

    def plot(self, kind: str = "cumulative", ax=None):
        """Todas as curvas num gráfico só (matplotlib sob demanda)."""
        from .report import plot_curves
        return plot_curves(self.curves_, kind=kind, ax=ax)

    # ------------------------------------------------------------------
    # Calibração e ciclo
    # ------------------------------------------------------------------
    def calibrate_to(self, pd_12m: Union[float, Mapping], horizon: Optional[int] = None
                     ) -> "LifetimePD":
        """Recalibra o **nível** das curvas preservando o **formato**.

        ``pd_12m`` escalar aplica o mesmo alvo a todas as curvas;
        ``{grupo: alvo}`` aplica por grupo (grupos ausentes ficam como estão).
        Devolve **novo** objeto — o original não muda."""
        self._check_fit()
        alvos = dict(pd_12m) if isinstance(pd_12m, Mapping) else {k: pd_12m for k in self.curves_}
        novas, deltas = {}, {}
        for rot, c in self.curves_.items():
            alvo = alvos.get(rot)
            if alvo is None:
                novas[rot] = c
                continue
            novas[rot] = c.calibrate_to(pd_12m=float(alvo), horizon=horizon)
            deltas[rot] = float(novas[rot].meta["calibracao"]["delta"])
        return self._replace(novas, {"calibrada": alvos},
                             ajuste={"tipo": "logit_shift", "por_grupo": deltas})

    def condition(self, z, rho: float, decay: Optional[float] = None,
                  mode: str = "shift") -> "LifetimePD":
        """Condiciona as curvas ao ciclo (TTC → PIT) pelo arcabouço de Vasicek.

        Aplica :func:`pit_from_ttc` ao *hazard* de cada período — e não à PD
        acumulada — de modo que a sobrevivência continue sendo o produto das
        condicionais deslocadas.

        Parameters
        ----------
        z:
            Fator sistêmico. Escalar (mesmo ciclo em todo o horizonte), ou uma
            **sequência** com um valor por horizonte — que é onde se encaixa a
            projeção de um modelo satélite
            (:mod:`yggdrasil.credit_risk.econometric`). Sequências curtas são
            estendidas pelo último valor.
        rho:
            Correlação de ativos em ``[0, 1)`` — a sensibilidade ao ciclo.
        decay:
            Só para ``z`` escalar: **reversão à média** por período
            (``z_t = z·(1−decay)^t``). É o tratamento usual num horizonte
            *lifetime*, onde o choque de hoje não vale para sempre — ``0.1``
            dissipa ~65% do choque em 10 períodos.
        mode:
            ``'shift'`` (padrão, idempotente em ``z = 0``) ou ``'conditional'``
            (lei condicional exata do fator único). Ver :func:`pit_from_ttc`.

        Returns
        -------
        LifetimePD
            Novo objeto condicionado. Em ``method='markov'`` com ``z`` escalar,
            ``decay=None`` e ``mode='shift'``, delega ao
            :meth:`~yggdrasil.credit_risk.ecl.markov.MarkovPD.condition`, que
            desloca a **matriz inteira** (todas as migrações, não só a coluna de
            *default*) — mais fiel, e a mesma fórmula de deslocamento.
        """
        self._check_fit()
        escalar = np.atleast_1d(np.asarray(z, dtype=float)).size == 1
        if (self.method_ == "markov" and self.markov_ is not None
                and escalar and decay is None and mode == "shift"):
            mk = self.markov_.condition(z=float(np.ravel(z)[0]), rho=rho)
            novo = self._replace(mk.curves(horizon=self.horizon),
                                 {"condicionada": {"z": float(np.ravel(z)[0]), "rho": rho,
                                                   "mode": "shift", "via": "matriz"}})
            novo.markov_ = mk
            return novo

        novas = {}
        for rot, c in self.curves_.items():
            zt = _z_path(z, len(c), decay=decay)
            h = pit_from_ttc(c.hazard_, rho, zt, mode=mode)
            novas[rot] = PDCurve(np.clip(h, 0.0, 1.0), label=c.label, freq=c.freq,
                                 meta={**c.meta, "condicionada": {"rho": rho, "decay": decay,
                                                                  "mode": mode,
                                                                  "z0": float(zt[0])}})
        ajuste = {"tipo": "vasicek", "rho": float(rho), "mode": mode, "decay": decay,
                  "z": np.ravel(np.asarray(z, dtype=float)).tolist()}
        return self._replace(novas, {"condicionada": {"rho": rho, "decay": decay, "mode": mode}},
                             ajuste=ajuste)

    def _replace(self, curves: Dict[object, PDCurve], meta_extra: dict,
                 ajuste: Optional[dict] = None) -> "LifetimePD":
        novo = LifetimePD(method=self.method, horizon=self.horizon, **self.engine_kwargs)
        novo.curves_ = curves
        novo.method_, novo.by, novo.features = self.method_, self.by, list(self.features)
        novo.freq, novo.tables_ = self.freq, self.tables_
        novo.hazard_models_, novo.markov_ = dict(self.hazard_models_), self.markov_
        novo.adjustments_ = list(self.adjustments_) + ([ajuste] if ajuste else [])
        novo.meta = {**self.meta, **meta_extra}
        return novo

    def _apply_adjustments(self, haz: np.ndarray, chaves: np.ndarray) -> np.ndarray:
        """Reaplica calibração e ciclo à matriz de *hazards* vinda do modelo.

        Percorre :attr:`adjustments_` na ordem em que foram feitos, para que a
        carteira escorada pelo motor ``'hazard'`` receba exatamente as mesmas
        transformações que as curvas já receberam."""
        out = haz
        H = out.shape[1]
        for aj in self.adjustments_:
            if aj["tipo"] == "logit_shift":
                delta = np.zeros(len(out))
                for chave, d in aj["por_grupo"].items():
                    delta[chaves == chave] = float(d)
                p = np.clip(out, 1e-12, 1.0 - 1e-12)
                out = 1.0 / (1.0 + np.exp(-(np.log(p / (1.0 - p)) + delta[:, None])))
            elif aj["tipo"] == "vasicek":
                zt = _z_path(np.asarray(aj["z"], dtype=float), H, decay=aj.get("decay"))
                out = pit_from_ttc(out, aj["rho"], zt[None, :], mode=aj.get("mode", "shift"))
        return np.clip(out, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Aplicação à carteira
    # ------------------------------------------------------------------
    def marginal_matrix(self, df: pd.DataFrame, horizon: Optional[int] = None,
                        age_col: Optional[str] = None, term_col: Optional[str] = None,
                        segment_col: Optional[str] = None) -> np.ndarray:
        """Matriz ``(n_linhas, horizon)`` de PDs **marginais** por contrato.

        É o insumo direto do ECL: a linha ``i``, coluna ``t`` é a probabilidade
        de o contrato ``i`` quebrar exatamente no período ``t`` visto de hoje.

        * **Idade** (``age_col``) — nos métodos indexados por idade
          (:data:`AGE_INDEXED`) a curva do contrato começa na idade atual dele.
        * **Prazo** (``term_col``) — zera as marginais além do prazo
          remanescente: um contrato com 8 meses de prazo não acumula PD no 9º.
        * **Grupo** (``segment_col``) — casa cada linha com a sua curva. Sem ele,
          usa a coluna que foi usada no ``fit`` (``self.by``), e cai na curva
          única se o ajuste não foi por grupo.
        """
        self._check_fit()
        H = int(horizon or self.horizon)
        if H < 1:
            raise ValueError(f"horizon deve ser >= 1; recebido {horizon!r}.")
        n = len(df)
        if n == 0:
            return np.zeros((0, H))

        idade = (pd.to_numeric(df[age_col], errors="coerce").fillna(0).to_numpy(dtype=int)
                 if age_col else np.zeros(n, dtype=int))
        if np.any(idade < 0):
            raise ValueError(f"{age_col!r} tem idades negativas.")

        # --- de cada linha para a sua curva ---------------------------------
        grupo_col = segment_col or (self.by if (self.by and self.by in df.columns) else None)
        if grupo_col is None:
            if len(self.curves_) > 1:
                raise ValueError(
                    f"o modelo tem {len(self.curves_)} curvas mas df não traz a coluna de "
                    f"grupo ({self.by!r}) — informe segment_col."
                )
            chaves = np.full(n, next(iter(self.curves_)), dtype=object)
        else:
            chaves = df[grupo_col].to_numpy(dtype=object)
        desconhecidos = {k for k in pd.unique(chaves) if k not in self.curves_}
        if desconhecidos:
            raise ValueError(f"grupos sem curva ajustada: {sorted(map(str, desconhecidos))[:6]}.")

        # --- hazards por contrato -------------------------------------------
        haz = np.empty((n, H), dtype=float)
        if self.method_ == "hazard" and self.hazard_models_:
            for chave in pd.unique(chaves):
                mask = chaves == chave
                haz[mask] = self.hazard_models_[chave].predict_curves(
                    df[mask], horizon=H, age_col=age_col)
            haz = self._apply_adjustments(haz, chaves)
        else:
            desloca = self.method_ in AGE_INDEXED
            for chave in pd.unique(chaves):
                mask = chaves == chave
                curva = self.curves_[chave]
                if not desloca:
                    haz[mask] = curva.extend(H).hazard_[:H]
                    continue
                base = curva.hazard_
                for a in np.unique(idade[mask]):
                    sub = mask & (idade == a)
                    trecho = base[a: a + H]
                    if trecho.size < H:                    # extrapolação plana
                        ultimo = trecho[-1] if trecho.size else base[-1]
                        trecho = np.concatenate([trecho, np.full(H - trecho.size, ultimo)])
                    haz[sub] = trecho

        # --- hazard → marginal, com o corte de prazo -----------------------
        haz = np.clip(haz, 0.0, 1.0)
        if term_col is not None:
            prazo = pd.to_numeric(df[term_col], errors="coerce").fillna(H).to_numpy(dtype=float)
            prazo = np.clip(prazo, 0, H)
            fora = np.arange(1, H + 1)[None, :] > prazo[:, None]
            haz = np.where(fora, 0.0, haz)
        sobrev = np.cumprod(1.0 - haz, axis=1)
        sobrev_ant = np.concatenate([np.ones((len(haz), 1)), sobrev[:, :-1]], axis=1)
        return sobrev_ant * haz

    def apply(self, df: pd.DataFrame, horizon: Optional[int] = None,
              age_col: Optional[str] = None, term_col: Optional[str] = None,
              segment_col: Optional[str] = None, prefix: str = "pd_marg_h",
              detail: bool = True) -> pd.DataFrame:
        """Cola a estrutura a termo em cada contrato do DataFrame.

        Devolve uma **cópia** de ``df`` com ``pd_12m`` e ``pd_lifetime`` sempre, e
        com as marginais por horizonte (``pd_marg_h1``…) quando ``detail=True``.
        Com horizonte de 60 períodos são 60 colunas — use ``detail=False`` para
        só o resumo."""
        marg = self.marginal_matrix(df, horizon=horizon, age_col=age_col,
                                    term_col=term_col, segment_col=segment_col)
        H = marg.shape[1]
        out = df.copy()
        n_ano = min(periods_per_year(self.freq), H)
        out["pd_12m"] = marg[:, :n_ano].sum(axis=1)
        out["pd_lifetime"] = marg.sum(axis=1)
        if detail:
            novas = pd.DataFrame(
                marg, index=out.index, columns=[f"{prefix}{t}" for t in range(1, H + 1)]
            )
            out = pd.concat([out, novas], axis=1)
        return out

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------
    def backtest(self, panel: ContractPanel, horizons: Sequence[int] = (12, 24, 36),
                 alpha: float = 0.05) -> pd.DataFrame:
        """Previsto × observado por horizonte, com o IC de Greenwood do observado.

        Para cada grupo, estima a curva de Kaplan-Meier do painel informado (a
        janela *out-of-time*, tipicamente) e confronta com a PD acumulada que o
        modelo prevê. A coluna ``dentro_do_ic`` é o veredito por horizonte."""
        self._check_fit()
        partes = ({self.GLOBAL: panel} if (self.by is None or self.by not in panel.df.columns)
                  else panel.by(self.by))
        linhas = []
        for rot, parte in partes.items():
            chave = rot if rot in self.curves_ else (
                self.GLOBAL if self.GLOBAL in self.curves_ else None)
            if chave is None:
                continue
            curva = self.curves_[chave]
            _, tab = kaplan_meier(parte, alpha=alpha, return_table=True)
            for h in horizons:
                h = int(h)
                if h > len(tab) or h > len(curva):
                    continue
                obs = tab.iloc[h - 1]
                prev = curva.pd_lifetime(h)
                inf, sup = float(obs["pd_acumulada_ic_inf"]), float(obs["pd_acumulada_ic_sup"])
                linhas.append({
                    "grupo": rot, "horizonte": h,
                    "n_em_risco_inicial": int(tab.iloc[0]["n_em_risco"]),
                    "pd_prevista": prev,
                    "pd_observada": float(obs["pd_acumulada"]),
                    "ic_inf": inf, "ic_sup": sup,
                    "erro_absoluto": prev - float(obs["pd_acumulada"]),
                    "erro_relativo": (prev / float(obs["pd_acumulada"]) - 1.0
                                      if obs["pd_acumulada"] > 0 else np.nan),
                    "dentro_do_ic": bool(inf <= prev <= sup),
                })
        if not linhas:
            raise ValueError(
                "o backtest não produziu linhas — confira se os grupos do painel batem "
                "com os do ajuste e se o horizonte cabe na janela observada."
            )
        return pd.DataFrame(linhas)

    # ------------------------------------------------------------------
    # Persistência e rastreio
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Representação serializável (as curvas + a linhagem do ajuste)."""
        self._check_fit()
        return {
            "method": self.method, "method_resolvido": self.method_,
            "horizon": self.horizon, "freq": self.freq, "by": self.by,
            "features": list(self.features), "meta": self.meta,
            "adjustments": self.adjustments_,
            "curves": {str(k): v.to_dict() for k, v in self.curves_.items()},
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """Serializa em JSON; grava em ``path`` se informado."""
        txt = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(txt)
        return txt

    @classmethod
    def from_dict(cls, d: Mapping) -> "LifetimePD":
        """Reconstrói a partir de :meth:`to_dict` (as curvas, não os motores)."""
        obj = cls(method=d.get("method", "vintage"), horizon=int(d.get("horizon", 60)))
        obj.method_ = d.get("method_resolvido", obj.method)
        obj.freq = d.get("freq", "M")
        obj.by = d.get("by")
        obj.features = list(d.get("features") or [])
        obj.meta = dict(d.get("meta") or {})
        obj.adjustments_ = list(d.get("adjustments") or [])
        obj.curves_ = {k: PDCurve.from_dict(v) for k, v in (d.get("curves") or {}).items()}
        if cls.GLOBAL in obj.curves_ and len(obj.curves_) == 1:
            obj.by = None
        return obj

    @classmethod
    def from_json(cls, path_or_text: str) -> "LifetimePD":
        """Lê de um caminho de arquivo ou de uma string JSON."""
        texto = path_or_text
        if not str(path_or_text).lstrip().startswith("{"):
            with open(path_or_text, "r", encoding="utf-8") as fh:
                texto = fh.read()
        return cls.from_dict(json.loads(texto))

    def log_to_mlflow(self, experiment: Optional[str] = None, run_name: Optional[str] = None,
                      backtest: Optional[pd.DataFrame] = None, **kwargs):
        """Registra as curvas e o backtest no MLflow (módulo carregado sob demanda)."""
        from .tracking import log_lifetime_pd
        return log_lifetime_pd(self, experiment=experiment, run_name=run_name,
                               backtest=backtest, **kwargs)

    def __repr__(self) -> str:
        if not self.curves_:
            return f"LifetimePD(method={self.method!r}, não ajustado)"
        return (f"LifetimePD(method={self.method_!r}, curvas={len(self.curves_)}, "
                f"H={self.horizon}, freq={self.freq!r})")


__all__ = ["LifetimePD", "pit_from_ttc", "METHODS", "AGE_INDEXED"]
