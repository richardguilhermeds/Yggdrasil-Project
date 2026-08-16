"""
A álgebra da curva de PD (:class:`PDCurve`) — quatro nomes para a mesma coisa
============================================================================
A maior fonte de confusão em PD *lifetime* não é o motor de estimação: é o
vocabulário. A mesma curva é pedida de quatro formas diferentes, e cada área usa
uma:

============================  ======================================================
representação                 o que responde
============================  ======================================================
**condicional** (*hazard*)    "sobreviveu até ``t−1``; qual a chance de quebrar em ``t``?"
**marginal**                  "qual a chance de quebrar **exatamente** em ``t``?" (vista de hoje)
**acumulada**                 "qual a chance de ter quebrado **até** ``t``?"
**sobrevivência**             "qual a chance de **não** ter quebrado até ``t``?"
============================  ======================================================

As quatro carregam **a mesma informação** e se convertem por identidades exatas:

``S(t) = Π_{k≤t} (1 − h_k)`` · ``F(t) = 1 − S(t)`` ·
``m(t) = S(t−1) · h(t)`` · ``h(t) = m(t) / S(t−1)``

:class:`PDCurve` guarda internamente o *hazard* — a representação **canônica**,
por ser a única que não depende do ponto de partida — e entrega as outras três
sob demanda. Aceita ser construída a partir de qualquer uma das quatro
(:meth:`~PDCurve.from_hazard`, :meth:`~PDCurve.from_marginal`,
:meth:`~PDCurve.from_cumulative`, :meth:`~PDCurve.from_survival`), o que
dispensa quem chama de fazer a conversão na mão — a fonte mais comum de erro de
sinal e de defasagem de um período.

Além da álgebra, o módulo traz os dois construtores de entrada:

* :func:`constant_hazard` — o caminho mais curto: **só** a PD de 12 meses que já
  saiu do modelo transversal (:class:`~yggdrasil.credit_risk.model.ModelSegmenter`)
  vira uma curva de *hazard* constante ``h = 1 − (1 − PD₁₂)^{1/n}``;
* :func:`vintage_curve` — a curva empírica por **safra/idade**, contada direto do
  painel com a base em risco correta (censura tratada).

E a ponte entre os dois eixos: :meth:`PDCurve.calibrate_to` desloca o *hazard*
no logit até a PD acumulada bater com a PD de 12 meses do modelo de escoragem,
**preservando o formato** da maturação. É a prática de mercado — o **nível** vem
do modelo transversal, a **forma** vem da curva.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .panel import ContractPanel, periods_per_year

#: Recorte das probabilidades antes do logit (a transformação diverge em 0 e 1).
EPS = 1e-12

#: Nomes das colunas de :meth:`PDCurve.to_frame` — fonte única para report e testes.
CURVE_COLUMNS = ("pd_condicional", "pd_marginal", "pd_acumulada", "sobrevivencia")


def _as_array(x, nome: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{nome} não pode ser vazio.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{nome} contém NaN/inf.")
    return arr


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ======================================================================
# A curva
# ======================================================================
@dataclass
class PDCurve:
    """Estrutura a termo da PD de um contrato/segmento.

    Guarda o *hazard* ``h(1..H)`` (PD condicional por período) e deriva as
    demais representações. Prefira os construtores nomeados
    (:meth:`from_hazard`, :meth:`from_marginal`, :meth:`from_cumulative`,
    :meth:`from_survival`) a instanciar direto.

    Parameters
    ----------
    hazard:
        Vetor de PDs **condicionais** por horizonte, em ``[0, 1]``. A posição
        ``i`` é o horizonte ``i + 1``.
    label:
        Rótulo da curva (segmento, rating, produto) — usado nos gráficos e nas
        tabelas comparativas.
    freq:
        Frequência dos períodos (``'M'``, ``'Q'``, ``'A'``). Define quantos
        períodos formam um ano em :meth:`pd_12m`.
    meta:
        Dicionário livre de linhagem (método, amostra, data de construção) para
        a documentação do modelo.
    """

    hazard_: np.ndarray
    label: str = ""
    freq: str = "M"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        h = _as_array(self.hazard_, "hazard")
        if np.any((h < 0.0) | (h > 1.0)):
            raise ValueError("hazard deve estar em [0, 1] (é uma probabilidade condicional).")
        periods_per_year(self.freq)  # valida a frequência
        self.hazard_ = h

    # -- construtores ---------------------------------------------------
    @classmethod
    def from_hazard(cls, hazard, label: str = "", freq: str = "M", **meta) -> "PDCurve":
        """A partir da PD **condicional** por período (a forma canônica)."""
        return cls(_as_array(hazard, "hazard"), label=label, freq=freq, meta=dict(meta))

    @classmethod
    def from_marginal(cls, marginal, label: str = "", freq: str = "M", **meta) -> "PDCurve":
        """A partir da PD **marginal** — ``h(t) = m(t) / S(t−1)``.

        Exige ``m ≥ 0`` e ``Σ m ≤ 1`` (a soma das marginais é a PD acumulada, que
        não pode passar de 1)."""
        m = _as_array(marginal, "marginal")
        if np.any(m < 0.0):
            raise ValueError("marginal não pode ter valores negativos.")
        acum = np.cumsum(m)
        if acum[-1] > 1.0 + 1e-9:
            raise ValueError(
                f"a soma das PDs marginais é {acum[-1]:.6f} > 1 — a curva acumulada "
                "ultrapassaria a certeza de default."
            )
        sobrev_ant = np.concatenate(([1.0], 1.0 - acum[:-1]))    # S(t−1)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(sobrev_ant > EPS, m / sobrev_ant, 0.0)
        return cls(np.clip(h, 0.0, 1.0), label=label, freq=freq, meta=dict(meta))

    @classmethod
    def from_cumulative(cls, cumulative, label: str = "", freq: str = "M", **meta) -> "PDCurve":
        """A partir da PD **acumulada** — diferencia e cai em :meth:`from_marginal`.

        Exige curva não-decrescente em ``[0, 1]``."""
        f = _as_array(cumulative, "cumulative")
        if np.any((f < 0.0) | (f > 1.0)):
            raise ValueError("cumulative deve estar em [0, 1].")
        if np.any(np.diff(f) < -1e-9):
            raise ValueError("cumulative deve ser não-decrescente (é uma acumulada).")
        m = np.diff(np.concatenate(([0.0], f)))
        return cls.from_marginal(np.clip(m, 0.0, 1.0), label=label, freq=freq, **meta)

    @classmethod
    def from_survival(cls, survival, label: str = "", freq: str = "M", **meta) -> "PDCurve":
        """A partir da curva de **sobrevivência** — ``h(t) = 1 − S(t)/S(t−1)``.

        Exige curva não-crescente em ``[0, 1]``."""
        s = _as_array(survival, "survival")
        if np.any((s < 0.0) | (s > 1.0)):
            raise ValueError("survival deve estar em [0, 1].")
        if np.any(np.diff(s) > 1e-9):
            raise ValueError("survival deve ser não-crescente (é uma sobrevivência).")
        s_ant = np.concatenate(([1.0], s[:-1]))
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(s_ant > EPS, 1.0 - s / s_ant, 0.0)
        return cls(np.clip(h, 0.0, 1.0), label=label, freq=freq, meta=dict(meta))

    # -- as quatro representações ---------------------------------------
    def _index(self) -> pd.Index:
        return pd.Index(np.arange(1, len(self) + 1), name="horizonte")

    def hazard(self) -> pd.Series:
        """PD **condicional** por horizonte: quebrar em ``t`` tendo chegado vivo a ``t``."""
        return pd.Series(self.hazard_.copy(), index=self._index(), name="pd_condicional")

    #: Sinônimo de :meth:`hazard` — o nome usado em sobrevivência.
    conditional = hazard

    def survival(self) -> pd.Series:
        """Probabilidade de **não** ter quebrado até ``t``: ``S(t) = Π (1 − h_k)``."""
        s = np.cumprod(1.0 - self.hazard_)
        return pd.Series(s, index=self._index(), name="sobrevivencia")

    def cumulative(self) -> pd.Series:
        """PD **acumulada** até ``t``: ``F(t) = 1 − S(t)``."""
        return pd.Series(
            1.0 - np.cumprod(1.0 - self.hazard_), index=self._index(), name="pd_acumulada"
        )

    def marginal(self) -> pd.Series:
        """PD **marginal** de ``t`` (vista de hoje): ``m(t) = S(t−1) · h(t)``.

        É a parcela que entra na soma do ECL *lifetime* — cada período contribui
        com a sua marginal, não com a sua condicional."""
        s = np.cumprod(1.0 - self.hazard_)
        s_ant = np.concatenate(([1.0], s[:-1]))
        return pd.Series(s_ant * self.hazard_, index=self._index(), name="pd_marginal")

    # -- leituras derivadas ---------------------------------------------
    def forward(self, t0: int, t1: int) -> float:
        """PD **condicional entre horizontes**: quebrar em ``(t0, t1]`` tendo chegado a ``t0``.

        ``P = 1 − S(t1) / S(t0)``. É o insumo natural de SICR/estágio 2: compara
        a PD *lifetime* remanescente de hoje com a de quando o contrato entrou."""
        n = len(self)
        if not (0 <= t0 <= t1 <= n):
            raise ValueError(f"exige 0 <= t0 <= t1 <= {n}; recebido t0={t0}, t1={t1}.")
        if t0 == t1:
            return 0.0
        s = np.cumprod(1.0 - self.hazard_)
        s0 = 1.0 if t0 == 0 else float(s[t0 - 1])
        s1 = float(s[t1 - 1])
        if s0 <= EPS:
            return 1.0
        return float(np.clip(1.0 - s1 / s0, 0.0, 1.0))

    def pd_12m(self) -> float:
        """PD de **12 meses** — a acumulada em um ano da frequência da curva.

        Com menos de um ano de horizonte, devolve a acumulada do horizonte
        disponível (e o ``meta`` da curva registra o truncamento)."""
        n_ano = min(periods_per_year(self.freq), len(self))
        return float(self.cumulative().iloc[n_ano - 1])

    def pd_lifetime(self, prazo: Optional[int] = None) -> float:
        """PD **lifetime**: a acumulada até ``prazo`` (ou até o fim da curva).

        ``prazo`` é o prazo **remanescente** em períodos. Valores acima do
        horizonte da curva são truncados no horizonte."""
        n = len(self)
        h = n if prazo is None else int(min(max(prazo, 0), n))
        if h == 0:
            return 0.0
        return float(self.cumulative().iloc[h - 1])

    # -- manipulação -----------------------------------------------------
    def truncate(self, horizon: int) -> "PDCurve":
        """Curva cortada em ``horizon`` períodos (não altera a original)."""
        h = int(horizon)
        if h < 1:
            raise ValueError(f"horizon deve ser >= 1; recebido {horizon!r}.")
        return PDCurve(self.hazard_[:h].copy(), label=self.label, freq=self.freq,
                       meta={**self.meta, "truncada_em": h})

    def extend(self, horizon: int) -> "PDCurve":
        """Estende a curva até ``horizon`` repetindo o último *hazard* observado.

        Extrapolação **plana** — a hipótese mínima quando o painel não observa
        idades altas. Registra em ``meta`` a partir de onde a curva é
        extrapolada, para que o relatório sinalize."""
        h = int(horizon)
        n = len(self)
        if h <= n:
            return self.truncate(h)
        extra = np.full(h - n, self.hazard_[-1])
        return PDCurve(np.concatenate([self.hazard_, extra]), label=self.label,
                       freq=self.freq, meta={**self.meta, "extrapolada_apos": n})

    def shift_logit(self, delta: float) -> "PDCurve":
        """Desloca o *hazard* no logit: ``h' = expit(logit(h) + δ)``.

        Preserva a **ordem** e as razões de chance entre horizontes — muda o
        nível da curva sem mudar o seu formato. É a primitiva de
        :meth:`calibrate_to` e do condicionamento ao ciclo."""
        return PDCurve(_expit(_logit(self.hazard_) + float(delta)), label=self.label,
                       freq=self.freq, meta={**self.meta, "shift_logit": float(delta)})

    def calibrate_to(self, pd_12m: Optional[float] = None, *,
                     target: Optional[float] = None, horizon: Optional[int] = None) -> "PDCurve":
        """Recalibra o **nível** da curva preservando o **formato**.

        Resolve o deslocamento ``δ`` no logit do *hazard* tal que a PD acumulada
        no horizonte-alvo iguale o valor pedido. O uso típico é colar a curva
        empírica (que dá a forma da maturação) no nível de PD que saiu do modelo
        transversal por cliente:

        >>> curva.calibrate_to(pd_12m=0.037)      # bate a PD de 12 meses do scorecard

        Parameters
        ----------
        pd_12m:
            PD-alvo de **12 meses**. Atalho para ``target=pd_12m`` com
            ``horizon`` = um ano da frequência da curva.
        target, horizon:
            Alvo e horizonte explícitos (ex.: ``target=0.22, horizon=36`` para
            calibrar a PD *lifetime* de 3 anos).

        Returns
        -------
        PDCurve
            Nova curva; a original não é alterada. ``meta['calibracao']`` guarda
            alvo, horizonte e ``δ`` para a documentação do modelo.
        """
        if (pd_12m is None) == (target is None):
            raise ValueError("informe exatamente um de pd_12m ou target.")
        if pd_12m is not None:
            alvo = float(pd_12m)
            h = min(periods_per_year(self.freq), len(self)) if horizon is None else int(horizon)
        else:
            alvo = float(target)
            if horizon is None:
                raise ValueError("target exige horizon explícito.")
            h = int(horizon)
        if not (0.0 < alvo < 1.0):
            raise ValueError(f"o alvo de calibração deve estar em (0, 1); recebido {alvo!r}.")
        if not (1 <= h <= len(self)):
            raise ValueError(f"horizon deve estar em [1, {len(self)}]; recebido {h}.")

        base = self.hazard_[:h]
        if np.all(base <= EPS):
            raise ValueError(
                "não há hazard positivo até o horizonte pedido — a curva não pode "
                "ser recalibrada por deslocamento (nada para deslocar)."
            )

        def acumulada(delta: float) -> float:
            return 1.0 - float(np.prod(1.0 - _expit(_logit(base) + delta)))

        from scipy.optimize import brentq

        lo, hi = -40.0, 40.0
        f_lo, f_hi = acumulada(lo) - alvo, acumulada(hi) - alvo
        if f_lo > 0 or f_hi < 0:
            raise ValueError(
                f"alvo {alvo:.6f} fora do alcance do deslocamento no logit "
                f"(acumulada em {h} períodos varia de {acumulada(lo):.2e} a "
                f"{acumulada(hi):.6f}); revise o alvo ou o horizonte."
            )
        delta = float(brentq(lambda d: acumulada(d) - alvo, lo, hi, xtol=1e-12, rtol=1e-14))
        nova = self.shift_logit(delta)
        nova.meta = {**self.meta, "calibracao": {"alvo": alvo, "horizonte": h, "delta": delta}}
        return nova

    # -- saída -----------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """As quatro representações lado a lado, uma linha por horizonte."""
        out = pd.concat(
            [self.hazard(), self.marginal(), self.cumulative(), self.survival()], axis=1
        )
        return out[list(CURVE_COLUMNS)]

    def to_dict(self) -> dict:
        """Representação serializável (JSON) da curva."""
        return {
            "hazard": [float(x) for x in self.hazard_],
            "label": self.label,
            "freq": self.freq,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "PDCurve":
        """Reconstrói a curva a partir de :meth:`to_dict`."""
        return cls.from_hazard(d["hazard"], label=d.get("label", ""),
                               freq=d.get("freq", "M"), **dict(d.get("meta") or {}))

    def to_json(self, path: Optional[str] = None) -> str:
        """Serializa em JSON; grava em ``path`` se informado."""
        txt = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(txt)
        return txt

    def plot(self, kind: str = "cumulative", ax=None):
        """Gráfico da curva (matplotlib carregado sob demanda).

        ``kind``: ``'cumulative'``, ``'marginal'``, ``'hazard'`` ou ``'survival'``."""
        from .report import plot_pd_curve
        return plot_pd_curve(self, kind=kind, ax=ax)

    # -- protocolo --------------------------------------------------------
    def __len__(self) -> int:
        return int(self.hazard_.size)

    def __repr__(self) -> str:
        rot = f", label={self.label!r}" if self.label else ""
        return (f"PDCurve(H={len(self)}, freq={self.freq!r}, "
                f"PD12m={self.pd_12m():.4%}, PDlife={self.pd_lifetime():.4%}{rot})")


# ======================================================================
# Construtores de entrada
# ======================================================================
def constant_hazard(pd_12m: float, horizon: int, freq: str = "M",
                    label: str = "") -> PDCurve:
    """Curva de *hazard* **constante** a partir da PD de 12 meses.

    O caminho de entrada mais curto do pacote: quem já tem a PD de um ano vinda
    do modelo transversal ganha uma estrutura a termo completa distribuindo esse
    risco uniformemente pelos períodos:

    ``h = 1 − (1 − PD₁₂)^{1/n}``, com ``n`` = períodos por ano.

    A hipótese embutida é forte — risco constante ao longo da vida, ou seja
    **sem maturação** — e por isso serve como *linha de base* e como plano B
    quando não há histórico por safra. Assim que houver painel, prefira
    :func:`vintage_curve` (ou os motores de :mod:`~yggdrasil.credit_risk.ecl.survival`)
    e use :meth:`PDCurve.calibrate_to` para trazer o nível de volta a este mesmo
    ``pd_12m``.

    Parameters
    ----------
    pd_12m:
        PD acumulada de 12 meses, em ``[0, 1)``.
    horizon:
        Nº de períodos da curva.
    freq:
        Frequência (``'M'``, ``'Q'``, ``'A'``).
    """
    p = float(pd_12m)
    if not (0.0 <= p < 1.0):
        raise ValueError(f"pd_12m deve estar em [0, 1); recebido {pd_12m!r}.")
    h_total = int(horizon)
    if h_total < 1:
        raise ValueError(f"horizon deve ser >= 1; recebido {horizon!r}.")
    n_ano = periods_per_year(freq)
    h = 1.0 - (1.0 - p) ** (1.0 / n_ano)
    return PDCurve(np.full(h_total, h), label=label, freq=freq,
                   meta={"metodo": "constant_hazard", "pd_12m": p})


def vintage_curve(
    panel: ContractPanel,
    from_age: int = 0,
    horizon: Optional[int] = None,
    weighted: bool = False,
    min_at_risk: int = 1,
    fill: str = "ffill",
    label: str = "",
    alpha: float = 0.05,
    return_table: bool = False,
):
    """Curva empírica por **safra/idade** (*vintage*), contada direto do painel.

    O *hazard* de cada horizonte é a taxa de *default* observada na idade
    correspondente, sobre a **base em risco daquela idade** — não sobre o total
    de contratos. É essa recontagem que trata a censura à direita: contratos
    jovens entram nas idades baixas e simplesmente não aparecem nas altas, em vez
    de contarem como sobreviventes.

    Parameters
    ----------
    panel:
        O :class:`~yggdrasil.credit_risk.ecl.panel.ContractPanel`.
    from_age:
        Idade de partida. ``0`` (padrão) é a curva de originação; ``from_age=6``
        dá a curva de quem **já** está há 6 meses na carteira — o que se usa
        para contratos vivos.
    horizon:
        Nº de períodos da curva. ``None`` vai até a última idade com base em
        risco suficiente.
    weighted:
        Pondera as taxas por exposição (exige ``exposure_col`` no painel).
    min_at_risk:
        Base mínima para a idade contar. Idades com menos que isso são tratadas
        conforme ``fill`` — evita que uma cauda com 3 contratos vire *hazard* de
        33%.
    fill:
        Tratamento das idades sem base suficiente: ``'ffill'`` (padrão, repete o
        último *hazard* válido), ``'zero'`` ou ``'drop'`` (trunca a curva na
        última idade válida).
    alpha:
        Nível do IC binomial (Jeffreys) da tabela auxiliar.
    return_table:
        Se ``True``, devolve ``(curva, tabela)`` com a tabela de vida por idade
        (base em risco, quebras, censura, *hazard* e IC) — o anexo de
        documentação da curva.

    Returns
    -------
    PDCurve | tuple[PDCurve, pandas.DataFrame]
    """
    if fill not in ("ffill", "zero", "drop"):
        raise ValueError(f"fill deve ser 'ffill', 'zero' ou 'drop'; recebido {fill!r}.")
    if from_age < 0:
        raise ValueError(f"from_age deve ser >= 0; recebido {from_age!r}.")

    vida = panel.at_risk(weighted=weighted)
    vida = vida[vida.index >= int(from_age)]
    if vida.empty:
        raise ValueError(
            f"nenhuma idade >= from_age={from_age} no painel (idade máxima "
            f"observada: {panel.max_age})."
        )

    n_risco = vida["n_em_risco"].to_numpy(dtype=float)
    valido = n_risco >= float(min_at_risk)
    if not valido.any():
        raise ValueError(
            f"nenhuma idade com base em risco >= min_at_risk={min_at_risk}; "
            "reduza o limiar ou amplie a janela do painel."
        )

    h = vida["hazard"].to_numpy(dtype=float).copy()
    h[~valido] = np.nan
    if fill == "drop":
        ultimo = int(np.max(np.flatnonzero(valido)))
        h = h[: ultimo + 1]
        vida = vida.iloc[: ultimo + 1]
    h_series = pd.Series(h)
    h = (h_series.ffill().fillna(0.0) if fill == "ffill" else h_series.fillna(0.0)).to_numpy()

    if horizon is not None:
        alvo = int(horizon)
        if alvo < 1:
            raise ValueError(f"horizon deve ser >= 1; recebido {horizon!r}.")
        if alvo <= len(h):
            h, vida = h[:alvo], vida.iloc[:alvo]
        else:
            h = np.concatenate([h, np.full(alvo - len(h), h[-1])])

    curva = PDCurve(
        np.clip(h, 0.0, 1.0), label=label, freq=panel.freq,
        meta={"metodo": "vintage", "from_age": int(from_age), "weighted": bool(weighted),
              "min_at_risk": int(min_at_risk), "fill": fill,
              "n_contratos": panel.n_contracts},
    )
    if not return_table:
        return curva

    from ...metrics.calibration import binomial_ci

    tabela = vida.copy()
    ic_inf, ic_sup = binomial_ci(tabela["n_default"].to_numpy(dtype=float),
                                 tabela["n_em_risco"].to_numpy(dtype=float), alpha=alpha)
    tabela["hazard_ic_inf"] = ic_inf
    tabela["hazard_ic_sup"] = ic_sup
    tabela["hazard_usado"] = curva.hazard_[: len(tabela)]
    tabela["horizonte"] = np.arange(1, len(tabela) + 1)
    return curva, tabela


def curve_frame(curves: Union[Mapping[object, PDCurve], Iterable[PDCurve]],
                kind: str = "cumulative") -> pd.DataFrame:
    """Empilha várias curvas numa tabela — uma coluna por curva, linhas = horizonte.

    ``kind``: ``'cumulative'`` (padrão), ``'marginal'``, ``'hazard'`` ou
    ``'survival'``. Aceita ``{rótulo: curva}`` ou uma sequência (aí o rótulo vem
    de ``curve.label``, com o índice como reserva)."""
    if isinstance(curves, Mapping):
        itens = list(curves.items())
    else:
        itens = [(c.label or f"curva_{i}", c) for i, c in enumerate(curves)]
    if not itens:
        raise ValueError("nenhuma curva informada.")
    getter = {"cumulative": "cumulative", "marginal": "marginal",
              "hazard": "hazard", "survival": "survival"}.get(kind)
    if getter is None:
        raise ValueError(
            f"kind deve ser 'cumulative', 'marginal', 'hazard' ou 'survival'; recebido {kind!r}."
        )
    return pd.DataFrame({str(rot): getattr(c, getter)() for rot, c in itens})


__all__ = ["PDCurve", "constant_hazard", "vintage_curve", "curve_frame", "CURVE_COLUMNS"]
