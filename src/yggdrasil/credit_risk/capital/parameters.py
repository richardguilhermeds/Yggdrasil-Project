"""
Calibração de parâmetros (PIT→TTC, LGD/CCF downturn) e o IRB de Basileia
=======================================================================
O capital econômico **reaproveita** os parâmetros de risco da provisão — PD, LGD
e EAD/CCF — mas com uma **calibração diferente** (Guia §4, Tabela 2, e §3.1). A
provisão contábil (IFRS 9 / Resolução CMN 4.966/2021) é *point-in-time* (PIT):
mede o risco **hoje**, seguindo o ciclo econômico. O capital econômico e o
regulatório (Basileia/IRB, Resolução CMN 4.557/2017 — ICAAP) olham o **ciclo
inteiro** e o **estresse**, para que a exigência de capital não desabe no topo
do ciclo (quando as PDs observadas estão baixas) nem exploda no fundo:

* **PD** — *through-the-cycle* (TTC): média de longo prazo do ciclo, e não a
  taxa corrente. Estabiliza a exigência de capital ao longo das fases.
* **LGD** — *downturn*: calibrada a períodos de **estresse**, porque a severidade
  sobe justamente quando os *defaults* sobem (garantias desvalorizam quando há
  mais execuções no mercado). É a correlação adversa PD–LGD.
* **EAD/CCF** — *downturn*: o fator de conversão do limite não sacado (CCF/LEQ)
  é conservador, pois o cliente tende a **sacar mais** o rotativo à beira do
  *default*.

A segunda metade do módulo traz as **fórmulas supervisoras do IRB de Basileia**
para o *benchmark* de **Pilar 1**: a correlação de ativos ``ρ`` é **fixada por
classe** de exposição (não estimada, como no capital econômico) e o nível de
confiança é ``q = 0,999``. O capital ``K`` por unidade de EAD sai da mesma
mecânica de Vasicek/ASRF (:mod:`.asrf`), sem o ajuste de maturidade (varejo).
Comparar o IRB com o capital econômico interno é um exercício central de
validação — o Pilar 1 ignora a diversificação entre produtos, então o capital
econômico bem construído tende a ser **menor** que a soma dos RWAs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .asrf import capital_ratio

if TYPE_CHECKING:  # só para type-checkers; evita import em runtime
    from .portfolio import Portfolio

# Classes de exposição suportadas pela fórmula de correlação do IRB (varejo +
# corporativo simplificado). Exportado para mensagens de erro e validação.
BASEL_EXPOSURE_CLASSES = ("revolving", "mortgage", "other_retail", "corporate")

# Multiplicador que converte requerimento de capital em ativo ponderado pelo
# risco (RWA): 1 / 0,08 = 12,5 (razão mínima de capital de Basileia de 8%).
RWA_MULTIPLIER = 12.5


# ======================================================================
# Calibração de PD: point-in-time → through-the-cycle
# ======================================================================
def pit_to_ttc(pd_pit_series: Sequence[float] | np.ndarray, method: str = "mean") -> float:
    """Converte uma série de PDs *point-in-time* em uma PD *through-the-cycle*.

    A PD TTC é a **média de longo prazo** da PD ao longo do ciclo econômico. A
    provisão usa a PD PIT (a taxa corrente, que sobe em recessão e cai em
    expansão); o capital usa a PD TTC para não pró-ciclar a exigência.

    Parameters
    ----------
    pd_pit_series:
        Série histórica de PDs observadas (ou taxas de *default*) por período,
        em ``[0, 1]``. **Deve cobrir ao menos um ciclo econômico completo**
        (idealmente incluindo uma recessão): uma média sobre só a parte boa do
        ciclo subestima a PD TTC e, portanto, o capital. Este é o ponto mais
        delicado da calibração TTC.
    method:
        ``"mean"`` (padrão) → média aritmética de longo prazo, a proxy usual de
        TTC. ``"median"`` → mediana, mais robusta a poucos períodos de estresse
        extremo na amostra.

    Returns
    -------
    float
        A PD TTC.
    """
    arr = np.asarray(pd_pit_series, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("pd_pit_series deve ser uma série 1-D não vazia.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("pd_pit_series contém valores não finitos (NaN/inf).")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("pd_pit_series deve estar em [0, 1] (são probabilidades).")

    if method == "mean":
        return float(np.mean(arr))
    if method == "median":
        return float(np.median(arr))
    raise ValueError(f"method deve ser 'mean' ou 'median'; recebido {method!r}.")


# ======================================================================
# Calibração de LGD: downturn
# ======================================================================
def lgd_downturn_addon(
    lgd: float, addon: float = 0.0, floor: Optional[float] = None
) -> float:
    """LGD *downturn* pelo método do **add-on supervisor** (BCBS §468).

    A abordagem mais simples do *downturn*: parte da LGD média de longo prazo e
    soma um acréscimo conservador que reflete o estresse. Onde o supervisor
    prescreve, também se aplica um **piso** (``floor``).

    ``LGD_downturn = min(1, max(0, LGD + addon))`` e, se ``floor`` dado,
    ``max(floor, ·)``.

    Parameters
    ----------
    lgd:
        LGD média (de longo prazo / esperada), em ``[0, 1]``.
    addon:
        Acréscimo de *downturn* (em pontos de LGD, ex.: ``0.10`` = +10 p.p.).
        Pode ser negativo, mas o resultado é sempre recortado em ``[0, 1]``.
    floor:
        Piso regulatório opcional para a LGD (ex.: pisos de LGD para varejo),
        em ``[0, 1]``. Aplicado **após** o *clip* em ``[0, 1]``.

    Returns
    -------
    float
        A LGD *downturn*, em ``[0, 1]``.
    """
    if not (0.0 <= lgd <= 1.0):
        raise ValueError(f"lgd deve estar em [0, 1]; recebido {lgd!r}.")
    if floor is not None and not (0.0 <= floor <= 1.0):
        raise ValueError(f"floor deve estar em [0, 1]; recebido {floor!r}.")

    out = float(np.clip(lgd + addon, 0.0, 1.0))
    if floor is not None:
        out = max(out, float(floor))
    return out


def lgd_downturn_from_series(
    lgd_series: Sequence[float] | np.ndarray,
    default_rate_series: Sequence[float] | np.ndarray,
    worst_frac: float = 0.2,
) -> float:
    """LGD *downturn* **empírica**: média da LGD nos períodos de pior *default*.

    Captura a **correlação adversa PD–LGD** diretamente dos dados: seleciona a
    fração ``worst_frac`` **superior** dos períodos por taxa de *default* (os
    anos de estresse) e devolve a LGD média observada nesses períodos. É o método
    de *downturn* preferido quando há histórico suficiente com ao menos uma
    recessão.

    Parameters
    ----------
    lgd_series:
        LGD média observada por período, em ``[0, 1]``. Deve estar **alinhada**
        (mesmo comprimento e ordem) a ``default_rate_series``.
    default_rate_series:
        Taxa de *default* observada por período, em ``[0, 1]``. Define quais
        períodos são de *downturn* (os de maior taxa).
    worst_frac:
        Fração superior dos períodos, por taxa de *default*, a considerar como
        *downturn*, em ``(0, 1]``. Ex.: ``0.2`` = os 20% piores anos. É garantido
        que **ao menos um** período entra no cálculo.

    Returns
    -------
    float
        A LGD *downturn* (média das LGDs nos períodos de pior *default*).
    """
    lgd = np.asarray(lgd_series, dtype=float)
    dr = np.asarray(default_rate_series, dtype=float)
    if lgd.ndim != 1 or dr.ndim != 1:
        raise ValueError("lgd_series e default_rate_series devem ser 1-D.")
    if lgd.size == 0:
        raise ValueError("As séries não podem ser vazias.")
    if lgd.shape != dr.shape:
        raise ValueError(
            "lgd_series e default_rate_series devem ter o mesmo comprimento; "
            f"recebido {lgd.shape} e {dr.shape}."
        )
    if np.any(~np.isfinite(lgd)) or np.any(~np.isfinite(dr)):
        raise ValueError("As séries contêm valores não finitos (NaN/inf).")
    if np.any((lgd < 0.0) | (lgd > 1.0)):
        raise ValueError("lgd_series deve estar em [0, 1].")
    if np.any((dr < 0.0) | (dr > 1.0)):
        raise ValueError("default_rate_series deve estar em [0, 1].")
    if not (0.0 < worst_frac <= 1.0):
        raise ValueError(f"worst_frac deve estar em (0, 1]; recebido {worst_frac!r}.")

    n = lgd.size
    # Ao menos 1 período; arredonda para cima para não perder o downturn em
    # amostras curtas.
    k = int(np.ceil(worst_frac * n))
    k = max(1, min(k, n))
    # Índices dos k períodos com MAIOR taxa de default.
    piores = np.argsort(dr)[::-1][:k]
    return float(np.mean(lgd[piores]))


# ======================================================================
# Calibração de EAD/CCF: downturn
# ======================================================================
def ccf_downturn(
    ccf_series: Sequence[float] | np.ndarray, quantile: float = 0.9
) -> float:
    """CCF/LEQ *downturn*: um **quantil alto** da utilização até o *default*.

    O fator de conversão de crédito (CCF, ou LEQ — *loan equivalent*) mede quanto
    do limite **não sacado** vira exposição quando o cliente entra em *default*.
    Como o cliente à beira do *default* tende a sacar mais o rotativo, o CCF de
    capital é calibrado a um **quantil alto** da distribuição histórica (ex.:
    90%), e não à média — uma escolha conservadora coerente com o *downturn*.

    Parameters
    ----------
    ccf_series:
        Série histórica de CCFs/LEQs observados até o *default*. Tipicamente em
        ``[0, 1]``, mas valores fora (saques além do limite) são **recortados**
        em ``[0, 1]`` no resultado.
    quantile:
        Quantil a extrair, em ``[0, 1]`` (padrão ``0.9``). Quanto maior, mais
        conservador.

    Returns
    -------
    float
        O CCF *downturn*, recortado em ``[0, 1]``.
    """
    arr = np.asarray(ccf_series, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("ccf_series deve ser uma série 1-D não vazia.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("ccf_series contém valores não finitos (NaN/inf).")
    if not (0.0 <= quantile <= 1.0):
        raise ValueError(f"quantile deve estar em [0, 1]; recebido {quantile!r}.")

    q_val = float(np.quantile(arr, quantile))
    # Mantém em [0, 1]: um CCF é uma fração do limite não sacado.
    return float(np.clip(q_val, 0.0, 1.0))


# ======================================================================
# Fórmulas regulatórias do IRB de Basileia (Pilar 1, benchmark)
# ======================================================================
def basel_correlation(pd: float, exposure_class: str) -> float:
    """Correlação de ativos ``ρ`` **prescrita** pelo IRB por classe de exposição.

    No IRB, ``ρ`` **não é estimado**: é fixado pelo supervisor por classe (é a
    diferença central para o capital econômico interno, onde ``ρ`` vem dos
    dados). As classes de varejo:

    * ``"revolving"`` (QRRE, rotativo qualificado — cartão) → ``ρ = 0,04``, fixo.
    * ``"mortgage"`` (imobiliário residencial) → ``ρ = 0,15``, fixo.
    * ``"other_retail"`` (demais varejo) → interpola entre ``0,03`` e ``0,16`` em
      função (decrescente) da PD:
      ``ρ = 0,03·w + 0,16·(1 − w)``, com ``w = (1 − e^{−35·PD}) / (1 − e^{−35})``.
    * ``"corporate"`` (atacado, versão sem ajuste de porte/PME) → interpola entre
      ``0,12`` e ``0,24``:
      ``ρ = 0,12·w + 0,24·(1 − w)``, com ``w = (1 − e^{−50·PD}) / (1 − e^{−50})``.

    A intuição da interpolação: PDs **baixas** têm correlação **alta** (o
    *default* de bons clientes é dominado pelo fator sistêmico); PDs altas têm
    correlação baixa (dominam fatores idiossincráticos).

    Parameters
    ----------
    pd:
        Probabilidade de *default* em ``[0, 1]``. Ignorada para ``"revolving"`` e
        ``"mortgage"`` (correlações fixas).
    exposure_class:
        Uma de :data:`BASEL_EXPOSURE_CLASSES`.

    Returns
    -------
    float
        A correlação de ativos ``ρ``.
    """
    if not (0.0 <= pd <= 1.0):
        raise ValueError(f"pd deve estar em [0, 1]; recebido {pd!r}.")

    cls = exposure_class
    if cls == "revolving":
        return 0.04
    if cls == "mortgage":
        return 0.15
    if cls == "other_retail":
        w = (1.0 - np.exp(-35.0 * pd)) / (1.0 - np.exp(-35.0))
        return float(0.03 * w + 0.16 * (1.0 - w))
    if cls == "corporate":
        w = (1.0 - np.exp(-50.0 * pd)) / (1.0 - np.exp(-50.0))
        return float(0.12 * w + 0.24 * (1.0 - w))
    raise ValueError(
        f"exposure_class desconhecida: {exposure_class!r}. "
        f"Classes válidas: {', '.join(BASEL_EXPOSURE_CLASSES)}."
    )


def basel_irb_capital(
    pd: float, lgd: float, exposure_class: str, q: float = 0.999
) -> float:
    """Capital ``K`` por unidade de EAD pela fórmula supervisora do IRB (varejo).

    Usa a correlação **regulatória** da classe (:func:`basel_correlation`) e a
    mesma mecânica de Vasicek/ASRF do capital econômico
    (:func:`~yggdrasil.credit_risk.capital.asrf.capital_ratio`), **sem** ajuste
    de maturidade — que no IRB só se aplica ao atacado, não ao varejo. O nível de
    confiança é fixo em ``q = 0,999`` por definição de Basileia.

    ``K = LGD × [ N((N⁻¹(PD) + √ρ · N⁻¹(q)) / √(1 − ρ)) − PD ]``.

    Parameters
    ----------
    pd:
        PD em ``[0, 1]`` (TTC, na calibração de capital).
    lgd:
        LGD em ``[0, 1]`` (*downturn*, na calibração de capital).
    exposure_class:
        Uma de :data:`BASEL_EXPOSURE_CLASSES`.
    q:
        Nível de confiança (padrão ``0,999``, referência de Basileia).

    Returns
    -------
    float
        O requerimento de capital ``K`` por real de EAD (já exclui a perda
        esperada).
    """
    if not (0.0 <= lgd <= 1.0):
        raise ValueError(f"lgd deve estar em [0, 1]; recebido {lgd!r}.")
    if not (0.0 < q < 1.0):
        raise ValueError(f"q deve estar em (0, 1); recebido {q!r}.")

    rho = basel_correlation(pd, exposure_class)
    return float(capital_ratio(pd, lgd, rho, q))


def basel_rwa(pd: float, lgd: float, ead: float, exposure_class: str) -> float:
    """Ativo ponderado pelo risco (RWA) do IRB: ``RWA = K × 12,5 × EAD``.

    O RWA reexpressa o capital ``K × EAD`` na escala de ativos ponderados, pelo
    multiplicador ``12,5 = 1 / 0,08`` (razão mínima de 8%). É a moeda do Pilar 1
    e a base de comparação com o capital econômico.

    Parameters
    ----------
    pd, lgd:
        PD e LGD em ``[0, 1]``.
    ead:
        Exposição no *default* (``> 0``), já com CCF *downturn* aplicado.
    exposure_class:
        Uma de :data:`BASEL_EXPOSURE_CLASSES`.

    Returns
    -------
    float
        O RWA da exposição.
    """
    if ead <= 0:
        raise ValueError(f"ead deve ser > 0; recebido {ead!r}.")
    k = basel_irb_capital(pd, lgd, exposure_class)
    return float(k * RWA_MULTIPLIER * ead)


def basel_capital_portfolio(
    portfolio: "Portfolio",
    exposure_class_map: Union[Mapping[str, str], str],
) -> pd.DataFrame:
    """Aplica o IRB de Basileia **segmento a segmento** e agrega a carteira.

    Produz o *benchmark* de **Pilar 1** para comparar com o capital econômico
    interno (validação, Resolução CMN 4.557/2017). Como o IRB é aditivo por
    construção (não há diversificação entre produtos), o capital regulatório
    tende a ser **maior** que o capital econômico de um modelo multifatorial —
    a diferença é justamente o benefício de diversificação que o Pilar 1 ignora.

    Parameters
    ----------
    portfolio:
        A carteira (:class:`~yggdrasil.credit_risk.capital.portfolio.Portfolio`).
    exposure_class_map:
        Ou um ``dict`` ``{nome_do_segmento: classe}`` mapeando cada segmento à
        sua classe de exposição do IRB, ou uma **string única** com a classe a
        aplicar a **todos** os segmentos. Cada classe deve estar em
        :data:`BASEL_EXPOSURE_CLASSES`.

    Returns
    -------
    pandas.DataFrame
        Uma linha por segmento com ``segmento``, ``classe``, ``PD``, ``LGD``,
        ``EAD``, ``rho``, ``K``, ``EL``, ``capital`` (``= K × EAD``) e ``RWA``,
        mais uma linha final ``"TOTAL"`` com os agregados (``rho`` e ``K`` ficam
        ``NaN`` no total, pois não são aditivos).
    """
    nomes = portfolio.segment_names()

    # Resolve a classe de cada segmento (dict ou string única).
    if isinstance(exposure_class_map, str):
        classes = {n: exposure_class_map for n in nomes}
    else:
        classes = dict(exposure_class_map)
        faltando = [n for n in nomes if n not in classes]
        if faltando:
            raise ValueError(
                "exposure_class_map não cobre todos os segmentos; faltam: "
                f"{faltando}."
            )

    pds = portfolio.pds()
    lgds = portfolio.lgds()
    eads = portfolio.eads()

    linhas = []
    for i, seg in enumerate(portfolio.segments):
        cls = classes[seg.name]
        rho = basel_correlation(float(pds[i]), cls)  # também valida a classe
        k = basel_irb_capital(float(pds[i]), float(lgds[i]), cls)
        cap = k * float(eads[i])
        el = float(pds[i]) * float(lgds[i]) * float(eads[i])
        rwa = k * RWA_MULTIPLIER * float(eads[i])
        linhas.append(
            {
                "segmento": seg.name,
                "classe": cls,
                "PD": float(pds[i]),
                "LGD": float(lgds[i]),
                "EAD": float(eads[i]),
                "rho": rho,
                "K": k,
                "EL": el,
                "capital": cap,
                "RWA": rwa,
            }
        )

    df = pd.DataFrame(linhas)
    total = {
        "segmento": "TOTAL",
        "classe": "",
        "PD": np.nan,
        "LGD": np.nan,
        "EAD": float(df["EAD"].sum()),
        "rho": np.nan,
        "K": np.nan,
        "EL": float(df["EL"].sum()),
        "capital": float(df["capital"].sum()),
        "RWA": float(df["RWA"].sum()),
    }
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


__all__ = [
    "BASEL_EXPOSURE_CLASSES",
    "RWA_MULTIPLIER",
    "pit_to_ttc",
    "lgd_downturn_addon",
    "lgd_downturn_from_series",
    "ccf_downturn",
    "basel_correlation",
    "basel_irb_capital",
    "basel_rwa",
    "basel_capital_portfolio",
]
