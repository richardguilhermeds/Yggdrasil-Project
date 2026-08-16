"""
A montagem: perda esperada de crédito (ECL)
===========================================
Os três parâmetros deste subpacote existem para virar um número só. A conta é
uma soma sobre o horizonte, e o único jeito de errá-la é trocar a PD **marginal**
pela condicional ou esquecer o desconto:

``ECL = Σ_t  PD_marginal(t) · LGD(t) · EAD(t) · (1 + i)^(−t/n)``

Cada fator vem de um lugar:

===================  ==========================================================
fator                de onde vem
===================  ==========================================================
``PD_marginal(t)``   :class:`~yggdrasil.credit_risk.ecl.lifetime_pd.LifetimePD`
                     — a probabilidade de quebrar **exatamente** em ``t``, vista
                     de hoje. É a marginal que soma, não a condicional: usar o
                     *hazard* aqui conta o mesmo contrato várias vezes.
``LGD(t)``           o modelo de severidade (o
                     :class:`~yggdrasil.credit_risk.model.ModelSegmenter` de
                     regressão, ou uma LGD por segmento). Pode variar no
                     horizonte.
``EAD(t)``           o saldo projetado — constante, amortizado ou vindo do CCF
                     (:mod:`~yggdrasil.credit_risk.ecl.ccf`) nos rotativos.
                     :func:`ead_schedule` gera as trajetórias usuais.
``(1+i)^(−t/n)``     desconto pela **taxa efetiva** do contrato, como exigem a
                     Resolução CMN 4.966/2021 e o IFRS 9.
===================  ==========================================================

**Estágios.** O corte de horizonte vem do estágio, e é só isso que ele faz aqui:

* **estágio 1** — ECL de **12 meses**: a soma para nos 12 primeiros períodos;
* **estágio 2** — ECL ***lifetime***: a soma vai até o prazo remanescente;
* **estágio 3** — ativo **já problemático**: não há PD a estimar, e a perda é a
  **ELBE** sobre o saldo atual (:mod:`~yggdrasil.credit_risk.ecl.elbe`).

A **regra de transferência entre estágios** (o SICR) **não** está aqui, e isso é
deliberado: o gatilho de aumento significativo de risco é política da
instituição — combinação de atraso, deterioração relativa da PD *lifetime*,
renegociação, carência, listas de observação — e cada uma documenta a sua. O
módulo recebe a coluna de estágio pronta e faz a conta; inventar o SICR aqui
transformaria uma decisão de governança em *default* de biblioteca.

O que ele oferece é o insumo quantitativo do SICR, que é a comparação de PD
*lifetime* remanescente entre a originação e hoje —
:meth:`~yggdrasil.credit_risk.ecl.curves.PDCurve.forward` calcula exatamente
isso.

**Cenários.** :func:`ecl_scenarios` roda a conta sob vários estados do ciclo e
pondera por probabilidade — o *forward-looking* de IFRS 9 / 4.966. Os ``z`` de
cada cenário podem vir de um julgamento (``+1`` benigno, ``−2`` severo) ou da
projeção de um modelo satélite de :mod:`yggdrasil.credit_risk.econometric`, que
já traz ``ScenarioSet`` e ``ecl_projection``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .lifetime_pd import LifetimePD

#: Tratamento de cada estágio. A chave é o valor da coluna de estágio.
DEFAULT_STAGE_RULES = {1: "12m", 2: "lifetime", 3: "elbe"}

#: Modos de :func:`ead_schedule`.
EAD_METHODS = ("constant", "linear", "annuity")


# ======================================================================
# Insumos: trajetória de exposição
# ======================================================================
def ead_schedule(exposure, term=None, horizon: int = 60, method: str = "constant",
                 rate: Optional[float] = None, periods_per_year: int = 12) -> np.ndarray:
    """Trajetória de exposição ``(n_contratos, horizon)`` ao longo do horizonte.

    Tratar o EAD como constante superestima a provisão de um contrato que
    amortiza — a perda do 40º mês incide sobre um saldo que já é uma fração do
    de hoje.

    Parameters
    ----------
    exposure:
        Saldo atual (escalar ou vetor por contrato).
    term:
        Prazo remanescente em períodos (escalar ou vetor). Depois dele a
        exposição é zero. ``None`` mantém a exposição por todo o horizonte.
    horizon:
        Nº de períodos.
    method:
        ``'constant'`` (padrão — rotativos e limites), ``'linear'`` (amortização
        constante do principal, a SAC) ou ``'annuity'`` (prestação fixa, a
        Price; exige ``rate``).
    rate:
        Taxa efetiva **anual** do contrato, para ``'annuity'``.
    periods_per_year:
        Períodos por ano.

    Returns
    -------
    numpy.ndarray
        Matriz ``(n, horizon)`` com o saldo devedor de cada período.
    """
    if method not in EAD_METHODS:
        raise ValueError(f"method deve ser um de {EAD_METHODS}; recebido {method!r}.")
    H = int(horizon)
    if H < 1:
        raise ValueError(f"horizon deve ser >= 1; recebido {horizon!r}.")
    e0 = np.atleast_1d(np.asarray(exposure, dtype=float)).reshape(-1, 1)
    n = e0.shape[0]
    t = np.arange(1, H + 1)[None, :]

    if term is None:
        prazo = np.full((n, 1), float(H))
    else:
        prazo = np.atleast_1d(np.asarray(term, dtype=float)).reshape(-1, 1)
        if prazo.shape[0] not in (1, n):
            raise ValueError(f"term tem {prazo.shape[0]} valores e exposure tem {n}.")
        prazo = np.broadcast_to(prazo, (n, 1))
    prazo = np.clip(prazo, 0.0, None)

    if method == "constant":
        saldo = np.broadcast_to(e0, (n, H)).copy()
    elif method == "linear":
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(prazo > 0, 1.0 - t / prazo, 0.0)
        saldo = e0 * np.clip(frac, 0.0, 1.0)
    else:  # annuity
        if rate is None:
            raise ValueError("method='annuity' exige rate (taxa efetiva anual).")
        i = (1.0 + float(rate)) ** (1.0 / float(periods_per_year)) - 1.0
        if abs(i) < 1e-12:
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(prazo > 0, 1.0 - t / prazo, 0.0)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(prazo > 0,
                                ((1.0 + i) ** prazo - (1.0 + i) ** t) / ((1.0 + i) ** prazo - 1.0),
                                0.0)
        saldo = e0 * np.clip(frac, 0.0, 1.0)

    return np.where(t > prazo, 0.0, saldo)


def _as_matrix(value, df: pd.DataFrame, H: int, nome: str) -> np.ndarray:
    """Normaliza escalar / nome de coluna / vetor / matriz numa matriz ``(n, H)``."""
    n = len(df)
    if isinstance(value, str):
        if value not in df.columns:
            raise ValueError(f"{nome}: coluna {value!r} não encontrada no DataFrame.")
        col = pd.to_numeric(df[value], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(col)):
            raise ValueError(f"{nome}: coluna {value!r} tem valores não numéricos/NaN.")
        return np.broadcast_to(col.reshape(-1, 1), (n, H)).copy()
    if isinstance(value, pd.Series):
        return np.broadcast_to(value.to_numpy(dtype=float).reshape(-1, 1), (n, H)).copy()
    if isinstance(value, pd.DataFrame):
        value = value.to_numpy(dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full((n, H), float(arr))
    if arr.ndim == 1:
        if arr.size == n and n != H:
            return np.broadcast_to(arr.reshape(-1, 1), (n, H)).copy()
        if arr.size == H:
            return np.broadcast_to(arr.reshape(1, -1), (n, H)).copy()
        if arr.size == n:                       # n == H: ambíguo, decide por contrato
            return np.broadcast_to(arr.reshape(-1, 1), (n, H)).copy()
        raise ValueError(
            f"{nome}: vetor de tamanho {arr.size} não casa nem com o nº de contratos "
            f"({n}) nem com o horizonte ({H})."
        )
    if arr.shape != (n, H):
        raise ValueError(f"{nome}: matriz {arr.shape} deveria ser ({n}, {H}).")
    return arr


def _detect_pd_matrix(df: pd.DataFrame, prefix: str) -> np.ndarray:
    """Lê as colunas ``<prefix>1..H`` já escoradas por :meth:`LifetimePD.apply`."""
    import re

    padrao = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    achados = sorted((int(m.group(1)), c) for c in map(str, df.columns)
                     if (m := padrao.match(c)))
    if not achados:
        raise ValueError(
            f"nenhuma coluna {prefix}<t> no DataFrame. Passe `model=` (um LifetimePD) "
            "ou `pd_marginal=` com a matriz de PDs marginais."
        )
    return df[[c for _, c in achados]].to_numpy(dtype=float)


# ======================================================================
# Resultado
# ======================================================================
@dataclass
class ECLResult:
    """Resultado de :func:`ecl_table`.

    Attributes
    ----------
    frame:
        Cópia do DataFrame de entrada com ``pd_12m``, ``pd_lifetime``,
        ``ecl_12m``, ``ecl_lifetime``, ``ecl_elbe`` (quando houver estágio 3),
        ``ecl`` (a que vale pelo estágio) e ``exposicao``.
    horizon:
        Horizonte usado.
    stage_rules:
        Regra aplicada por estágio.
    meta:
        Linhagem dos parâmetros.
    """

    frame: pd.DataFrame
    horizon: int
    stage_rules: Dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        """ECL total da carteira."""
        return float(self.frame["ecl"].sum())

    @property
    def coverage(self) -> float:
        """Taxa de provisão: ECL / exposição."""
        exp = float(self.frame["exposicao"].sum())
        return self.total / exp if exp else np.nan

    def by(self, *cols: str) -> pd.DataFrame:
        """Agrega o ECL por uma ou mais colunas, com a taxa de provisão."""
        chaves = list(cols)
        faltando = [c for c in chaves if c not in self.frame.columns]
        if faltando:
            raise ValueError(f"Colunas ausentes no resultado: {faltando}.")
        colunas = ["exposicao", "ecl", "ecl_12m", "ecl_lifetime"]
        g = (self.frame.groupby(chaves, sort=True)[colunas].sum()
             if chaves else self.frame[colunas].sum().to_frame().T)
        g["n_contratos"] = (self.frame.groupby(chaves, sort=True).size() if chaves
                            else len(self.frame))
        g["taxa_provisao"] = g["ecl"] / g["exposicao"].replace(0, np.nan)
        return g.reset_index() if chaves else g.reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        """Resumo por estágio — a visão que vai para a nota explicativa."""
        return self.by("estagio")

    def to_dict(self) -> dict:
        """Números agregados, serializáveis (sem a carteira inteira)."""
        return {"horizon": self.horizon, "ecl_total": self.total,
                "exposicao_total": float(self.frame["exposicao"].sum()),
                "taxa_provisao": self.coverage, "n_contratos": int(len(self.frame)),
                "stage_rules": {str(k): v for k, v in self.stage_rules.items()},
                "meta": self.meta}

    def __repr__(self) -> str:
        return (f"ECLResult(n={len(self.frame)}, H={self.horizon}, "
                f"ECL={self.total:,.2f}, cobertura={self.coverage:.2%})")


# ======================================================================
# A conta
# ======================================================================
def ecl_table(
    df: pd.DataFrame,
    model: Optional[LifetimePD] = None,
    pd_marginal=None,
    pd_prefix: str = "pd_marg_h",
    lgd: Union[float, str, np.ndarray] = 0.45,
    ead: Union[float, str, np.ndarray] = "saldo",
    horizon: Optional[int] = None,
    stage_col: Optional[str] = None,
    stage_rules: Optional[Mapping] = None,
    elbe: Union[float, str, None] = "elbe",
    discount_rate: Union[float, str, None] = None,
    periods_per_year: int = 12,
    age_col: Optional[str] = None,
    term_col: Optional[str] = None,
    segment_col: Optional[str] = None,
    exposure_col: Optional[str] = None,
    detail: bool = False,
) -> ECLResult:
    """Monta a tabela de perda esperada de crédito.

    Parameters
    ----------
    df:
        Carteira: uma linha por contrato.
    model:
        Um :class:`~yggdrasil.credit_risk.ecl.lifetime_pd.LifetimePD` ajustado —
        o caminho usual. As PDs marginais são calculadas na hora, respeitando
        idade e prazo.
    pd_marginal:
        Alternativa a ``model``: a matriz ``(n, H)`` de PDs marginais pronta.
        ``None`` em ambos faz o módulo procurar as colunas ``pd_prefix<t>``
        deixadas por :meth:`LifetimePD.apply`.
    lgd:
        Severidade: escalar, **nome de coluna** (LGD por contrato), vetor por
        contrato, vetor por horizonte ou matriz ``(n, H)``.
    ead:
        Exposição por período: mesma flexibilidade. Um nome de coluna vira
        exposição **constante**; para amortização, passe a saída de
        :func:`ead_schedule`.
    horizon:
        Horizonte em períodos. ``None`` usa o do ``model`` (ou o nº de colunas da
        matriz de PD).
    stage_col:
        Coluna com o estágio (1/2/3). ``None`` trata a carteira toda como
        estágio 2 (*lifetime*) e ainda assim reporta o ECL de 12 meses à parte.
    stage_rules:
        Sobrescreve :data:`DEFAULT_STAGE_RULES` — ``{valor_do_estagio: '12m' |
        'lifetime' | 'elbe'}``.
    elbe:
        ELBE do estágio 3: escalar ou nome de coluna (tipicamente a coluna que
        :func:`~yggdrasil.credit_risk.ecl.elbe.apply_elbe` deixou).
    discount_rate:
        Taxa efetiva **anual** (escalar ou nome de coluna). ``None`` não
        desconta — o que só é aceitável em horizontes curtos.
    periods_per_year:
        Períodos por ano (``12`` para painel mensal).
    age_col, term_col, segment_col:
        Repassados a :meth:`LifetimePD.marginal_matrix`.
    exposure_col:
        Coluna de exposição para a taxa de provisão. ``None`` usa a exposição do
        primeiro período de ``ead``.
    detail:
        Também devolve as colunas ``ecl_h1``…``ecl_hH`` (a perda esperada
        descontada de cada período), para conciliação.

    Returns
    -------
    ECLResult
    """
    if df.empty:
        raise ValueError("df não pode ser vazio.")
    regras = dict(DEFAULT_STAGE_RULES if stage_rules is None else stage_rules)
    validos = {"12m", "lifetime", "elbe"}
    if not set(regras.values()) <= validos:
        raise ValueError(f"stage_rules só aceita {sorted(validos)}; recebido {regras}.")

    # --- PD marginal ------------------------------------------------------
    if model is not None:
        H = int(horizon or model.horizon)
        marg = model.marginal_matrix(df, horizon=H, age_col=age_col, term_col=term_col,
                                     segment_col=segment_col)
    elif pd_marginal is not None:
        marg = np.asarray(pd_marginal, dtype=float)
        if marg.ndim != 2 or marg.shape[0] != len(df):
            raise ValueError(f"pd_marginal deve ser ({len(df)}, H); recebido {marg.shape}.")
        H = marg.shape[1] if horizon is None else int(horizon)
        marg = marg[:, :H]
    else:
        marg = _detect_pd_matrix(df, pd_prefix)
        H = marg.shape[1] if horizon is None else int(horizon)
        marg = marg[:, :H]
    if H < 1:
        raise ValueError(f"horizon deve ser >= 1; recebido {horizon!r}.")
    if np.any(marg < -1e-9) or np.any(marg > 1.0 + 1e-9):
        raise ValueError("as PDs marginais devem estar em [0, 1].")
    marg = np.clip(marg, 0.0, 1.0)

    # --- LGD, EAD e desconto ----------------------------------------------
    lgd_m = _as_matrix(lgd, df, H, "lgd")
    if np.any(lgd_m < 0.0) or np.any(lgd_m > 1.0):
        raise ValueError("lgd deve estar em [0, 1].")
    ead_m = _as_matrix(ead, df, H, "ead")
    if np.any(ead_m < 0.0):
        raise ValueError("ead não pode ser negativa.")

    if discount_rate is None:
        desconto = np.ones((len(df), H))
    else:
        taxa = _as_matrix(discount_rate, df, H, "discount_rate")[:, :1]
        if np.any(taxa <= -1.0):
            raise ValueError("discount_rate deve ser > -1.")
        expoente = np.arange(1, H + 1)[None, :] / float(periods_per_year)
        desconto = (1.0 + taxa) ** (-expoente)

    # --- a soma ------------------------------------------------------------
    perda = marg * lgd_m * ead_m * desconto
    n_ano = min(int(periods_per_year), H)
    ecl_12m = perda[:, :n_ano].sum(axis=1)
    ecl_life = perda.sum(axis=1)

    out = df.copy()
    out["pd_12m"] = marg[:, :n_ano].sum(axis=1)
    out["pd_lifetime"] = marg.sum(axis=1)
    out["ecl_12m"] = ecl_12m
    out["ecl_lifetime"] = ecl_life
    out["exposicao"] = (pd.to_numeric(df[exposure_col], errors="coerce").to_numpy(dtype=float)
                        if exposure_col else ead_m[:, 0])

    # --- estágios -----------------------------------------------------------
    if stage_col is None:
        estagio = np.full(len(df), 2, dtype=object)
    else:
        if stage_col not in df.columns:
            raise ValueError(f"Coluna de estágio {stage_col!r} não encontrada.")
        estagio = df[stage_col].to_numpy(dtype=object)
        desconhecidos = {s for s in pd.unique(estagio) if s not in regras}
        if desconhecidos:
            raise ValueError(
                f"estágios sem regra definida: {sorted(map(str, desconhecidos))} "
                f"(regras: {regras})."
            )
    out["estagio"] = estagio

    usa_elbe = np.array([regras.get(s) == "elbe" for s in estagio])
    if usa_elbe.any():
        if elbe is None:
            raise ValueError("há contratos em estágio 'elbe' mas `elbe` é None.")
        elbe_v = _as_matrix(elbe, df, 1, "elbe")[:, 0]
        if np.any(elbe_v < 0.0) or np.any(elbe_v > 1.0):
            raise ValueError("elbe deve estar em [0, 1].")
        out["ecl_elbe"] = elbe_v * out["exposicao"].to_numpy(dtype=float)

    escolha = np.array([regras.get(s, "lifetime") for s in estagio], dtype=object)
    ecl = np.where(escolha == "12m", ecl_12m, ecl_life)
    if usa_elbe.any():
        ecl = np.where(usa_elbe, out["ecl_elbe"].to_numpy(dtype=float), ecl)
    out["ecl"] = ecl

    if detail:
        detalhe = pd.DataFrame(perda, index=out.index,
                               columns=[f"ecl_h{t}" for t in range(1, H + 1)])
        out = pd.concat([out, detalhe], axis=1)

    return ECLResult(
        frame=out, horizon=H, stage_rules=regras,
        meta={"periods_per_year": int(periods_per_year),
              "discount_rate": discount_rate if isinstance(discount_rate, (int, float, str))
              else "matriz",
              "lgd": lgd if isinstance(lgd, (int, float, str)) else "matriz",
              "ead": ead if isinstance(ead, (int, float, str)) else "matriz",
              "modelo_pd": (model.meta if model is not None else None)},
    )


# ======================================================================
# Cenários — o forward-looking
# ======================================================================
def ecl_scenarios(
    df: pd.DataFrame,
    model: LifetimePD,
    scenarios: Mapping[str, Mapping],
    rho: float,
    weights: Optional[Mapping[str, float]] = None,
    **kwargs,
) -> Dict[str, object]:
    """ECL sob vários estados do ciclo, ponderado por probabilidade.

    O *forward-looking* de IFRS 9 / CMN 4.966 é uma média ponderada de cenários,
    não o cenário base com uma margem por cima. Cada cenário condiciona a curva
    de PD ao seu ``z`` (ver :meth:`LifetimePD.condition`) e a conta é refeita.

    Parameters
    ----------
    df, kwargs:
        A carteira e os demais argumentos de :func:`ecl_table` (LGD, EAD,
        desconto, estágios…), idênticos em todos os cenários.
    model:
        O modelo de PD *lifetime* ajustado, em base **TTC**.
    scenarios:
        ``{nome: {"z": ..., "peso": ..., "decay": ...}}``. ``z`` é escalar ou uma
        trajetória por horizonte — que é onde entra a projeção de um modelo
        satélite de :mod:`yggdrasil.credit_risk.econometric` (o ``Z`` de Vasicek
        projetado por cenário macro). ``decay`` é opcional.
    rho:
        Correlação de ativos usada no condicionamento.
    weights:
        Pesos alternativos ao ``"peso"`` de cada cenário. Devem somar 1.

    Returns
    -------
    dict
        ``{"por_cenario": DataFrame, "ponderado": float, "resultados":
        {nome: ECLResult}}``. O ``DataFrame`` traz ECL, exposição, taxa de
        provisão e peso de cada cenário.
    """
    if not scenarios:
        raise ValueError("informe ao menos um cenário.")
    pesos = {}
    for nome, cfg in scenarios.items():
        p = (weights or {}).get(nome, cfg.get("peso"))
        if p is None:
            raise ValueError(
                f"cenário {nome!r} sem peso — a ponderação do ECL exige probabilidade "
                "em todos os cenários (some 1)."
            )
        pesos[nome] = float(p)
    total = sum(pesos.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"os pesos dos cenários devem somar 1; somam {total:.6f}.")

    resultados, linhas = {}, []
    for nome, cfg in scenarios.items():
        condicionado = model.condition(cfg["z"], rho=rho, decay=cfg.get("decay"),
                                       mode=cfg.get("mode", "shift"))
        res = ecl_table(df, model=condicionado, **kwargs)
        resultados[nome] = res
        linhas.append({"cenario": nome, "peso": pesos[nome], "ecl": res.total,
                       "exposicao": float(res.frame["exposicao"].sum()),
                       "taxa_provisao": res.coverage,
                       "z": cfg["z"] if np.isscalar(cfg["z"]) else "trajetória"})
    por_cenario = pd.DataFrame(linhas)
    ponderado = float((por_cenario["ecl"] * por_cenario["peso"]).sum())
    por_cenario["contribuicao"] = por_cenario["ecl"] * por_cenario["peso"]
    return {"por_cenario": por_cenario, "ponderado": ponderado, "resultados": resultados}


__all__ = [
    "ecl_table", "ECLResult", "ecl_scenarios", "ead_schedule",
    "DEFAULT_STAGE_RULES", "EAD_METHODS",
]
