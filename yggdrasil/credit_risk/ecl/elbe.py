"""
A tabela ELBE: perda esperada dos contratos **já em *default***
===============================================================
Para o contrato que ainda não quebrou, a perda esperada é ``PD × LGD × EAD``.
Para o que **já quebrou**, não há PD — o evento aconteceu. O que resta é
perguntar: *do saldo que ainda está lá, quanto ainda vai virar perda?* Essa é a
**ELBE** (*expected loss best estimate*), e ela muda conforme o tempo em
*default*: um contrato que quebrou ontem tem quase tudo a recuperar; um que está
há três anos em cobrança já entregou o que tinha para entregar, e o que sobrou é
quase todo perda.

A construção parte de duas coisas — e só delas, que é como a área costuma
receber o dado:

* a **exposição inicial** (o EAD na data do *default*);
* as **colunas de LGD por mês em *default*** — ``lgd_m0``, ``lgd_m1``, …,
  ``lgd_mN`` — que são a LGD que se realizaria se a cobrança parasse naquele mês,
  isto é, ``lgd(t) = 1 − r(t)``, com ``r(t)`` a **recuperação acumulada** até
  ``t`` como fração da exposição inicial.

Daí sai tudo:

============================  ==========================================================
grandeza                      fórmula
============================  ==========================================================
recuperação acumulada         ``r̄(t) = Σᵢ EAD₀ᵢ·rᵢ(t) / Σᵢ EAD₀ᵢ`` (ponderada por exposição)
exposição remanescente        ``Σᵢ EAD₀ᵢ·(1 − rᵢ(t))``
LGD do ciclo completo         ``LGD = 1 − r̄(T*)``, no horizonte de *workout* ``T*``
**ELBE(t)**                   ``(1 − r̄(T*)) / (1 − r̄(t))``
LGD *in default*              ``ELBE(t) + add-on`` de perda inesperada
============================  ==========================================================

A ELBE é a perda que resta **sobre o saldo remanescente** — e não sobre a
exposição original. É por isso que ela é uma razão, e é por isso que ela **sobe**
com o tempo em *default* mesmo quando a recuperação total é boa: o denominador
encolhe mais rápido que o numerador. Um erro comum é reportar ``1 − r̄(t)`` como
se fosse a ELBE; isso é a LGD acumulada até ``t``, uma grandeza retrospectiva,
não a melhor estimativa do que ainda falta.

Três cuidados que o módulo trata explicitamente:

**Ponderação por exposição.** A média simples das LGDs individuais não é a LGD da
carteira. Tudo aqui é ponderado por ``EAD₀``.

**Censura.** Um *default* de três meses atrás não tem coluna de mês 24 — ele
ainda não chegou lá. Cada mês ``t`` usa **só** os contratos efetivamente
observados até ``t``, em vez de tratar o não-observado como zero recuperação (o
que despencaria a curva na cauda).

**Horizonte de *workout*.** A recuperação não é infinita: em algum mês a curva
achata. :func:`workout_horizon` localiza esse ponto — o critério do *ECB Guide to
Internal Models* (§6.4), "o momento após o *default* a partir do qual a evolução
das taxas de recuperação acumuladas é praticamente nula" — e é lá que se lê a LGD
do ciclo completo.

**Desconto.** Sob a Resolução CMN 4.966/2021 e o IFRS 9, as recuperações futuras
entram a **valor presente**. Passar ``discount_rate`` desconta cada recuperação
marginal pela taxa efetiva antes de reacumular — a única forma correta, já que a
acumulada bruta não se desconta em bloco.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

#: Como interpretar as colunas informadas.
LGD_KINDS = ("lgd", "recuperacao")

#: Unidade das colunas: fração da exposição inicial ou valor monetário.
LGD_UNITS = ("taxa", "valor")


# ======================================================================
# Leitura das colunas
# ======================================================================
def detect_month_columns(df: pd.DataFrame, prefix: str = "lgd_m") -> Tuple[List[str], np.ndarray]:
    """Localiza as colunas ``<prefix><t>`` e devolve ``(colunas, meses)`` ordenados.

    O sufixo numérico é o **mês em *default***, e é ele que ordena — não a ordem
    das colunas no DataFrame, que costuma vir de um ``pivot`` desordenado
    (``lgd_m10`` antes de ``lgd_m2``)."""
    padrao = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    achados = []
    for col in df.columns:
        m = padrao.match(str(col))
        if m:
            achados.append((int(m.group(1)), str(col)))
    if not achados:
        raise ValueError(
            f"nenhuma coluna com o prefixo {prefix!r} seguido de número "
            f"(ex.: '{prefix}0', '{prefix}1'). Colunas disponíveis: "
            f"{list(df.columns)[:10]}."
        )
    achados.sort()
    meses = np.array([m for m, _ in achados], dtype=int)
    return [c for _, c in achados], meses


def _recovery_matrix(df: pd.DataFrame, cols: Sequence[str], ead0: np.ndarray,
                     kind: str, unit: str) -> np.ndarray:
    """Matriz ``(n_contratos, n_meses)`` de **recuperação acumulada** em fração de EAD₀.

    Normaliza os quatro formatos de entrada (LGD ou recuperação × taxa ou valor)
    numa representação única, e trata as lacunas: um buraco **no meio** da série é
    falha de reporte e é preenchido para a frente; ``NaN`` no **fim** é censura e
    permanece ``NaN`` (o contrato ainda não chegou àquele mês)."""
    bruto = df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if unit == "valor":
        with np.errstate(divide="ignore", invalid="ignore"):
            bruto = bruto / ead0[:, None]
    rec = bruto if kind == "recuperacao" else 1.0 - bruto

    # Lacuna interior → preenche para a frente; NaN de cauda → censura.
    observado = ~np.isnan(rec)
    tem_obs = observado.any(axis=1)
    ultimo = np.where(tem_obs, observado.shape[1] - 1 - np.argmax(observado[:, ::-1], axis=1), -1)
    idx = np.arange(rec.shape[1])[None, :]
    interior = (~observado) & (idx <= ultimo[:, None]) & (idx >= np.argmax(observado, axis=1)[:, None])
    if interior.any():
        n_lacunas = int(interior.sum())
        preenchido = pd.DataFrame(rec).ffill(axis=1).to_numpy()
        rec = np.where(interior, preenchido, rec)
        warnings.warn(
            f"{n_lacunas} lacuna(s) no meio da série de recuperação foram preenchidas "
            "com o último valor observado (buraco de reporte, não censura).",
            RuntimeWarning,
        )
    return rec


# ======================================================================
# Curva de recuperação e horizonte de workout
# ======================================================================
def recovery_curve(
    df: pd.DataFrame,
    exposure_col: str = "exposicao_inicial",
    lgd_prefix: str = "lgd_m",
    kind: str = "lgd",
    unit: str = "taxa",
    discount_rate: Optional[float] = None,
    periods_per_year: int = 12,
    monotonic: str = "none",
    cohort: str = "chained",
) -> pd.DataFrame:
    """Curva de recuperação acumulada por mês em *default*, ponderada por exposição.

    O ponto delicado é a **coorte variável**: como os *defaults* recentes ainda
    não chegaram aos meses altos, o conjunto de contratos observados muda de mês
    para mês. Fazer a média direta da recuperação acumulada de cada mês mistura
    coortes diferentes e produz uma curva que pode até **cair** — o que não faz
    sentido para uma acumulada. O tratamento correto é **encadear a recuperação
    marginal**, o análogo do estimador produto-limite usado em sobrevivência:

    ``Δr̄(t) = Σ_{i ∈ obs(t)} EAD₀ᵢ·(rᵢ(t) − rᵢ(t−1)) / Σ_{i ∈ obs(t)} EAD₀ᵢ``
    e ``r̄(t) = Σ_{k ≤ t} Δr̄(k)``

    Cada mês é estimado nos contratos que **efetivamente chegaram** naquele mês, e
    a curva fica monotônica por construção (dadas recuperações marginais
    não-negativas).

    Parameters
    ----------
    df:
        Uma linha por contrato em *default*, no formato **largo**.
    exposure_col:
        Exposição na data do *default* (EAD₀). Deve ser ``> 0``.
    lgd_prefix:
        Prefixo das colunas por mês (``'lgd_m'`` → ``lgd_m0``, ``lgd_m1``, …).
    kind:
        ``'lgd'`` (padrão — as colunas são a LGD remanescente ``1 − r(t)``) ou
        ``'recuperacao'`` (as colunas já são ``r(t)``).
    unit:
        ``'taxa'`` (padrão — valores como fração de EAD₀) ou ``'valor'``
        (monetário; é dividido por ``exposure_col``).
    discount_rate:
        Taxa efetiva **anual** para trazer as recuperações a valor presente. Cada
        recuperação **marginal** do mês ``t`` é descontada por
        ``(1 + i)^(−t/periods_per_year)`` antes de agregar — a acumulada bruta
        não se desconta em bloco. ``None`` mantém os valores nominais.
    periods_per_year:
        Períodos por ano das colunas (``12`` para meses).
    monotonic:
        Tratamento de recuperação **marginal negativa** (estorno, revisão de
        garantia): ``'none'`` (padrão, respeita o dado e apenas conta as
        reversões), ``'clip'`` (zera as marginais negativas) ou ``'error'``.
    cohort:
        ``'chained'`` (padrão, o encadeamento acima) ou ``'complete'`` — a
        **coorte fechada**, que usa só os contratos observados até o último mês.
        A coorte fechada é mais simples de auditar, mas descarta os *defaults*
        recentes e, com isso, só enxerga o passado distante.

    Returns
    -------
    pandas.DataFrame
        Indexado por ``mes_default``, com ``n_contratos`` e
        ``exposicao_observada`` (a coorte que chegou àquele mês),
        ``exposicao_inicial`` (a coorte inteira, constante),
        ``recuperacao_acumulada``, ``recuperacao_marginal``,
        ``exposicao_remanescente``, ``lgd_acumulada`` e ``n_reversoes``.
    """
    if kind not in LGD_KINDS:
        raise ValueError(f"kind deve ser um de {LGD_KINDS}; recebido {kind!r}.")
    if unit not in LGD_UNITS:
        raise ValueError(f"unit deve ser um de {LGD_UNITS}; recebido {unit!r}.")
    if monotonic not in ("none", "clip", "error"):
        raise ValueError(f"monotonic deve ser 'none', 'clip' ou 'error'; recebido {monotonic!r}.")
    if cohort not in ("chained", "complete"):
        raise ValueError(f"cohort deve ser 'chained' ou 'complete'; recebido {cohort!r}.")
    if exposure_col not in df.columns:
        raise ValueError(f"Coluna de exposição {exposure_col!r} não encontrada.")
    if df.empty:
        raise ValueError("df não pode ser vazio.")

    cols, meses = detect_month_columns(df, lgd_prefix)
    if cohort == "complete":
        completo = df[cols[-1]].notna().to_numpy()
        if not completo.any():
            raise ValueError(
                f"cohort='complete' exige contratos observados até {cols[-1]!r}, e não há "
                "nenhum. Use cohort='chained' (padrão), que aproveita as observações parciais."
            )
        df = df.loc[completo]

    ead0 = pd.to_numeric(df[exposure_col], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(ead0)) or np.any(ead0 <= 0):
        raise ValueError(f"{exposure_col!r} deve ser finita e > 0 em todas as linhas.")
    rec = _recovery_matrix(df, cols, ead0, kind, unit)
    observado = ~np.isnan(rec)

    # --- recuperação MARGINAL individual (base do encadeamento) ---------
    acum = np.concatenate([np.zeros((len(rec), 1)), np.nan_to_num(rec, nan=0.0)], axis=1)
    marg = np.diff(acum, axis=1)
    marg = np.where(observado, marg, 0.0)      # censura não contribui

    reversoes = ((marg < -1e-12) & observado).sum(axis=0).astype(int)
    if reversoes.sum():
        if monotonic == "error":
            raise ValueError(
                f"{int(reversoes.sum())} recuperação(ões) marginal(is) negativa(s) "
                "(a acumulada cai de um mês para o outro). Use monotonic='clip' para "
                "zerá-las ou 'none' para respeitar o dado."
            )
        if monotonic == "clip":
            marg = np.maximum(marg, 0.0)

    # --- desconto das marginais a valor presente -------------------------
    if discount_rate is not None:
        i = float(discount_rate)
        if i <= -1.0:
            raise ValueError(f"discount_rate deve ser > -1; recebido {discount_rate!r}.")
        marg = marg * ((1.0 + i) ** (-meses / float(periods_per_year)))[None, :]

    # --- encadeamento ponderado por exposição ---------------------------
    exp_obs = np.where(observado, ead0[:, None], 0.0).sum(axis=0)
    if exp_obs[0] <= 0:
        raise ValueError(f"nenhum contrato observado no primeiro mês ({cols[0]!r}).")
    with np.errstate(divide="ignore", invalid="ignore"):
        marg_barra = np.where(exp_obs > 0, (marg * ead0[:, None]).sum(axis=0) / exp_obs, np.nan)
    # Meses sem base ficam fora da curva (a censura é sufixo: exp_obs não volta a subir).
    sem_base = exp_obs <= 0
    r_barra = np.cumsum(np.where(sem_base, 0.0, marg_barra))
    r_barra = np.where(sem_base, np.nan, r_barra)
    marg_barra = np.where(sem_base, np.nan, marg_barra)
    ead_total = float(ead0.sum())

    return pd.DataFrame(
        {
            "n_contratos": observado.sum(axis=0).astype(int),
            "exposicao_observada": exp_obs,
            "exposicao_inicial": np.full(len(meses), ead_total),
            "recuperacao_acumulada": r_barra,
            "recuperacao_marginal": marg_barra,
            "exposicao_remanescente": ead_total * (1.0 - r_barra),
            "lgd_acumulada": 1.0 - r_barra,
            "n_reversoes": reversoes,
        },
        index=pd.Index(meses, name="mes_default"),
    )


def workout_horizon(curve: pd.DataFrame, tol: float = 0.005,
                    min_contratos: int = 1) -> int:
    """Mês em que a curva de recuperação **achata** — o horizonte de *workout*.

    Devolve o **menor** ``t`` tal que o que ainda se recupera depois dele é
    menor ou igual a ``tol``:

    ``r̄(T_max) − r̄(t) ≤ tol``

    É a leitura operacional do critério do *ECB Guide to Internal Models* (§6.4):
    a partir daí a evolução da recuperação acumulada é praticamente nula, e é ali
    que se lê a LGD do ciclo completo. Meses com base abaixo de ``min_contratos``
    são ignorados — não se declara o fim da recuperação olhando para três
    contratos.

    Parameters
    ----------
    curve:
        Saída de :func:`recovery_curve`.
    tol:
        Recuperação adicional tolerada (em fração de EAD₀). ``0.005`` = meio
        ponto percentual.
    min_contratos:
        Base mínima para o mês contar.
    """
    if tol < 0:
        raise ValueError(f"tol deve ser >= 0; recebido {tol!r}.")
    valido = curve[(curve["n_contratos"] >= int(min_contratos))
                   & curve["recuperacao_acumulada"].notna()]
    if valido.empty:
        raise ValueError(
            f"nenhum mês com base >= min_contratos={min_contratos} e recuperação observada."
        )
    r = valido["recuperacao_acumulada"].to_numpy(dtype=float)
    final = float(r[-1])
    atingiu = np.flatnonzero(final - r <= tol)
    return int(valido.index[atingiu[0] if atingiu.size else -1])


# ======================================================================
# A tabela
# ======================================================================
@dataclass
class ELBETable:
    """Resultado de :func:`elbe_table` — a tabela ELBE por mês em *default*.

    Attributes
    ----------
    frame:
        A tabela, indexada por ``mes_default``.
    lgd:
        LGD do ciclo completo, ``1 − r̄(T*)``.
    workout:
        Horizonte de *workout* ``T*`` (mês em que a curva achata).
    ultimate_recovery:
        ``r̄(T*)``.
    addon:
        Add-on de perda inesperada somado à ELBE para formar a LGD *in default*.
    meta:
        Linhagem (parâmetros da construção) para a documentação do modelo.
    """

    frame: pd.DataFrame
    lgd: float
    workout: int
    ultimate_recovery: float
    addon: float = 0.0
    meta: dict = field(default_factory=dict)

    def elbe_at(self, mes: int) -> float:
        """ELBE de um contrato com ``mes`` meses em *default*.

        Meses além da tabela recebem a ELBE do último mês disponível (a
        estimativa não melhora depois do *workout*)."""
        idx = self.frame.index
        m = int(np.clip(mes, int(idx.min()), int(idx.max())))
        return float(self.frame.loc[m, "elbe"])

    def summary(self) -> pd.DataFrame:
        """Uma linha com os números que vão para a documentação."""
        return pd.DataFrame([{
            "n_contratos": int(self.frame["n_contratos"].iloc[0]),
            "exposicao_inicial": float(self.frame["exposicao_inicial"].iloc[0]),
            "horizonte_workout": self.workout,
            "recuperacao_no_workout": self.ultimate_recovery,
            "lgd_ciclo_completo": self.lgd,
            "elbe_mes_0": float(self.frame["elbe"].iloc[0]),
            "elbe_no_workout": self.elbe_at(self.workout),
            "addon_lgd_in_default": self.addon,
        }])

    def plot(self, ax=None):
        """Curva de recuperação e ELBE por mês em *default* (matplotlib sob demanda)."""
        from .report import plot_elbe
        return plot_elbe(self, ax=ax)

    def to_dict(self) -> dict:
        """Representação serializável (a tabela vira lista de registros)."""
        return {
            "lgd": self.lgd, "workout": self.workout,
            "ultimate_recovery": self.ultimate_recovery, "addon": self.addon,
            "meta": self.meta,
            "frame": self.frame.reset_index().to_dict(orient="records"),
        }

    def __repr__(self) -> str:
        return (f"ELBETable(T*={self.workout}, LGD={self.lgd:.2%}, "
                f"ELBE(0)={float(self.frame['elbe'].iloc[0]):.2%}, "
                f"ELBE(T*)={self.elbe_at(self.workout):.2%})")


def elbe_table(
    df: pd.DataFrame,
    exposure_col: str = "exposicao_inicial",
    lgd_prefix: str = "lgd_m",
    kind: str = "lgd",
    unit: str = "taxa",
    ultimate: Union[str, float] = "workout",
    addon: float = 0.0,
    floor: Optional[float] = None,
    discount_rate: Optional[float] = None,
    periods_per_year: int = 12,
    monotonic: str = "none",
    cohort: str = "chained",
    tol: float = 0.005,
    min_contratos: int = 1,
    by: Optional[str] = None,
) -> Union[ELBETable, Dict[object, ELBETable]]:
    """Constrói a tabela ELBE a partir das colunas de LGD e da exposição inicial.

    Para cada mês em *default* ``t``:

    * ``recuperacao_acumulada`` ``r̄(t)`` — ponderada por EAD₀, só sobre os
      contratos efetivamente observados em ``t``;
    * ``exposicao_remanescente`` ``Σ EAD₀·(1 − r(t))``;
    * ``elbe`` ``= (1 − r̄(T*)) / (1 − r̄(t))`` — a perda que ainda se espera,
      **sobre o saldo remanescente**;
    * ``lgd_in_default`` ``= min(1, ELBE + addon)``, com ``floor`` opcional.

    Parameters
    ----------
    df, exposure_col, lgd_prefix, kind, unit, discount_rate, periods_per_year, monotonic, cohort:
        Ver :func:`recovery_curve`.
    ultimate:
        Como fixar a recuperação do ciclo completo: ``'workout'`` (padrão — lê
        ``r̄`` no horizonte de :func:`workout_horizon`), ``'last'`` (último mês da
        tabela) ou um ``float`` em ``[0, 1)`` imposto pela política (ex.: a
        recuperação de longo prazo de um estudo próprio).

        Com ``'workout'``, a ELBE do próprio ``T*`` vale **1 por construção**: se
        a recuperação do ciclo completo é a que se observa até ``T*``, então quem
        chegou a ``T*`` sem recuperar não recupera mais, e o saldo remanescente é
        perda inteira. É o comportamento correto, e é o que faz a curva de ELBE
        subir do nível da LGD até 1 ao longo da cobrança. Use ``'last'`` para ler
        a recuperação no fim da janela (mais dado, base menor) ou um ``float``
        para impor a assíntota de um estudo de longo prazo.
    addon:
        Acréscimo de **perda inesperada** que separa a LGD *in default* da ELBE.
        As diretrizes de risco (EBA/GL/2017/16 §6) exigem que a LGD dos ativos em
        *default* reconheça a possibilidade de as recuperações virem piores que a
        melhor estimativa; ``0.0`` iguala as duas e precisa ser justificado.
    floor:
        Piso opcional aplicado à LGD *in default*.
    tol, min_contratos:
        Repassados a :func:`workout_horizon`.
    by:
        Coluna de segmentação. Com ela, devolve ``{segmento: ELBETable}`` — a
        recuperação de consignado e de cartão não é a mesma, e agregar as duas
        numa curva só é o erro clássico desta tabela.

    Returns
    -------
    ELBETable | dict[object, ELBETable]
    """
    if by is not None:
        if by not in df.columns:
            raise ValueError(f"Coluna de segmentação {by!r} não encontrada.")
        return {
            seg: elbe_table(parte, exposure_col=exposure_col, lgd_prefix=lgd_prefix,
                            kind=kind, unit=unit, ultimate=ultimate, addon=addon, floor=floor,
                            discount_rate=discount_rate, periods_per_year=periods_per_year,
                            monotonic=monotonic, cohort=cohort, tol=tol,
                            min_contratos=min_contratos)
            for seg, parte in df.groupby(by, sort=True)
        }

    if addon < 0.0:
        raise ValueError(f"addon deve ser >= 0 (a LGD in default não fica abaixo da ELBE); "
                         f"recebido {addon!r}.")
    curva = recovery_curve(df, exposure_col=exposure_col, lgd_prefix=lgd_prefix, kind=kind,
                           unit=unit, discount_rate=discount_rate,
                           periods_per_year=periods_per_year, monotonic=monotonic, cohort=cohort)

    if isinstance(ultimate, str):
        if ultimate == "workout":
            t_estrela = workout_horizon(curva, tol=tol, min_contratos=min_contratos)
        elif ultimate == "last":
            validos = curva[curva["recuperacao_acumulada"].notna()]
            t_estrela = int(validos.index[-1])
        else:
            raise ValueError(
                f"ultimate deve ser 'workout', 'last' ou um float; recebido {ultimate!r}."
            )
        r_final = float(curva.loc[t_estrela, "recuperacao_acumulada"])
    else:
        r_final = float(ultimate)
        if not (0.0 <= r_final < 1.0):
            raise ValueError(f"ultimate numérico deve estar em [0, 1); recebido {ultimate!r}.")
        t_estrela = workout_horizon(curva, tol=tol, min_contratos=min_contratos)

    lgd_total = 1.0 - r_final
    r = curva["recuperacao_acumulada"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        elbe = np.where(1.0 - r > 1e-12, lgd_total / (1.0 - r), 0.0)
    elbe = np.clip(elbe, 0.0, 1.0)

    lgd_in_default = np.clip(elbe + float(addon), 0.0, 1.0)
    if floor is not None:
        if not (0.0 <= floor <= 1.0):
            raise ValueError(f"floor deve estar em [0, 1]; recebido {floor!r}.")
        lgd_in_default = np.maximum(lgd_in_default, float(floor))

    tabela = curva.copy()
    tabela["elbe"] = elbe
    tabela["lgd_in_default"] = lgd_in_default
    tabela["perda_esperada_remanescente"] = tabela["exposicao_remanescente"] * elbe
    tabela["apos_workout"] = tabela.index > t_estrela

    return ELBETable(
        frame=tabela, lgd=lgd_total, workout=int(t_estrela), ultimate_recovery=r_final,
        addon=float(addon),
        meta={"kind": kind, "unit": unit, "ultimate": ultimate, "addon": float(addon),
              "floor": floor, "discount_rate": discount_rate, "monotonic": monotonic,
              "cohort": cohort, "tol": tol, "min_contratos": int(min_contratos),
              # a coorte EFETIVA (cohort='complete' descarta os defaults recentes)
              "n_contratos": int(curva["n_contratos"].iloc[0]),
              "n_linhas_entrada": int(len(df))},
    )


def elbe_frame(tables: Dict[object, ELBETable], segment_name: str = "segmento") -> pd.DataFrame:
    """Empilha ``{segmento: ELBETable}`` numa tabela longa única.

    É a forma que vai para o relatório e para o *join* com a carteira em
    *default*, quando a ELBE é estimada por segmento."""
    partes = []
    for seg, tab in tables.items():
        p = tab.frame.reset_index()
        p.insert(0, segment_name, seg)
        partes.append(p)
    if not partes:
        raise ValueError("nenhuma tabela informada.")
    return pd.concat(partes, ignore_index=True)


# ======================================================================
# Aplicação à carteira em default
# ======================================================================
def apply_elbe(
    df: pd.DataFrame,
    table: Union[ELBETable, Dict[object, ELBETable]],
    months_col: str = "meses_em_default",
    exposure_col: Optional[str] = None,
    segment_col: Optional[str] = None,
    prefix: str = "",
) -> pd.DataFrame:
    """Escora os contratos **vivos em *default*** pela tabela ELBE.

    Cada contrato recebe a ELBE e a LGD *in default* do seu tempo em *default*
    (e do seu segmento, se a tabela foi construída por segmento). Com
    ``exposure_col`` — o **saldo atual**, não a exposição na data do *default* —
    também sai a perda esperada em moeda, que é o que vai para o estágio 3 do
    ECL.

    Parameters
    ----------
    df:
        Carteira em *default*, uma linha por contrato.
    table:
        Uma :class:`ELBETable` ou ``{segmento: ELBETable}``.
    months_col:
        Coluna com os meses decorridos desde o *default*.
    exposure_col:
        Saldo atual do contrato (opcional).
    segment_col:
        Coluna de segmento — obrigatória quando ``table`` é um dicionário.
    prefix:
        Prefixo das colunas criadas (útil para comparar duas versões da tabela
        lado a lado).

    Returns
    -------
    pandas.DataFrame
        Cópia de ``df`` com ``elbe``, ``lgd_in_default`` e, se houver exposição,
        ``perda_esperada``.
    """
    if months_col not in df.columns:
        raise ValueError(f"Coluna {months_col!r} não encontrada.")
    meses = pd.to_numeric(df[months_col], errors="coerce")
    if meses.isna().any():
        raise ValueError(f"{months_col!r} contém valores não numéricos/NaN.")
    meses = meses.to_numpy(dtype=int)
    if np.any(meses < 0):
        raise ValueError(f"{months_col!r} tem valores negativos.")

    if isinstance(table, ELBETable):
        tabelas = {None: table}
        chaves = np.full(len(df), None, dtype=object)
    else:
        if segment_col is None:
            raise ValueError("com tabela por segmento é preciso informar segment_col.")
        if segment_col not in df.columns:
            raise ValueError(f"Coluna {segment_col!r} não encontrada.")
        tabelas = dict(table)
        chaves = df[segment_col].to_numpy(dtype=object)
        faltando = {k for k in pd.unique(chaves) if k not in tabelas}
        if faltando:
            raise ValueError(f"segmentos sem tabela ELBE: {sorted(map(str, faltando))[:6]}.")

    elbe = np.empty(len(df), dtype=float)
    lgd_id = np.empty(len(df), dtype=float)
    for chave in pd.unique(chaves):
        mask = chaves == chave
        tab = tabelas[chave]
        idx = tab.frame.index
        alvo = np.clip(meses[mask], int(idx.min()), int(idx.max()))
        elbe[mask] = tab.frame["elbe"].reindex(alvo).to_numpy(dtype=float)
        lgd_id[mask] = tab.frame["lgd_in_default"].reindex(alvo).to_numpy(dtype=float)

    out = df.copy()
    out[f"{prefix}elbe"] = elbe
    out[f"{prefix}lgd_in_default"] = lgd_id
    if exposure_col is not None:
        if exposure_col not in df.columns:
            raise ValueError(f"Coluna {exposure_col!r} não encontrada.")
        saldo = pd.to_numeric(df[exposure_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        out[f"{prefix}perda_esperada"] = saldo * elbe
    return out


__all__ = [
    "ELBETable", "elbe_table", "elbe_frame", "apply_elbe",
    "recovery_curve", "workout_horizon", "detect_month_columns",
    "LGD_KINDS", "LGD_UNITS",
]
