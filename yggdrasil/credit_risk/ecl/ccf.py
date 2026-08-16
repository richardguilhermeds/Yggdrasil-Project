"""
CCF e EAD: o fator de conversão do limite não sacado
====================================================
Em produto **rotativo** (cartão, cheque especial, capital de giro com limite) a
exposição não é conhecida: o cliente decide quanto saca, e quem está indo para o
*default* costuma sacar mais. A exposição na data do *default* é modelada por um
fator aplicado ao limite ainda disponível:

``EAD = sacado_ref + CCF · (limite_ref − sacado_ref)``

O ``CCF`` (*credit conversion factor*, também chamado LEQ, *loan equivalent*) é a
fração do **não sacado** que vira exposição. É um parâmetro estimado a partir de
histórico, e a literatura é bem clara sobre onde a estimação costuma dar errado.

Como se monta a base de referência
----------------------------------
O passo que mais move o resultado não é o modelo: é **de qual data de referência
se olha o contrato**. Três desenhos, e cada um responde a uma pergunta diferente
(Moral, 2011; Gürtler, Hibbeln & Usselmann, 2018):

``method='cohort'`` — **coorte**
    As datas de referência ficam numa grade de calendário espaçada de
    ``horizon``. Todo *default* do período é ligado ao **início da coorte**, de
    modo que o tempo até o *default* varia de 0 a ``horizon − 1``. É o desenho
    clássico e o mais próximo do que o supervisor imagina ao falar em "um ano
    antes do *default*".

``method='fixed_horizon'`` — **horizonte fixo**
    A referência é exatamente ``horizon`` períodos antes do *default*. Todo
    contrato contribui com uma observação, sempre à mesma distância — o desenho
    mais limpo para interpretar o CCF como "conversão em 12 meses", e o mais
    sensível à falta de histórico (o contrato precisa existir 12 meses antes).

``method='variable'`` — **horizonte variável**
    **Todas** as datas de referência dentro da janela ``[default − horizon,
    default)``. O mesmo contrato gera até ``horizon`` observações, o que
    multiplica a amostra e revela como o CCF cresce à medida que o *default* se
    aproxima — ao custo de observações correlacionadas dentro do contrato.

    *Nota*: a **coorte generalizada** de Gürtler, Hibbeln & Usselmann (2018) —
    coortes sobrepostas começando em todo período, em vez da grade fixa de
    calendário — produz exatamente este mesmo conjunto de pares
    (contrato, data de referência) quando a amostra é a dos **inadimplentes**. A
    generalização daquele artigo está no conjunto em risco dos **adimplentes**,
    que não entra na estimação do CCF. Por isso o pacote expõe três desenhos, e
    não quatro.

As quatro medidas ex-post
-------------------------
A mesma exposição realizada pode ser resumida de quatro formas, e a escolha muda
a estabilidade da estimativa (Tong, Mues, Brown & Thomas, 2016):

=================  ==========================================  ==================================
medida             definição                                   reconstrução do EAD
=================  ==========================================  ==================================
``ccf`` (LEQ)      ``(EAD − sacado) / (limite − sacado)``       ``sacado + CCF·(limite − sacado)``
``eadf``           ``EAD / limite``                            ``EADF · limite``
``auf``            ``(EAD − sacado) / limite``                  ``sacado + AUF · limite``
``ead``            o próprio EAD (modelagem direta)             ``EAD``
=================  ==========================================  ==================================

O ``ccf`` tem o denominador mais instável — quando o cliente já está quase no
limite, ``limite − sacado`` tende a zero e o fator explode. ``eadf`` e ``auf``
trocam esse denominador pelo limite, que é estável. A evidência empírica ainda
assim favorece o ``ccf`` para acurácia de estimação, e é ele o padrão aqui; as
outras três ficam disponíveis para o *benchmark* que a validação vai pedir.

A distribuição é **bimodal**
----------------------------
O CCF observado se concentra em **0** (o cliente não mexeu no limite) e em **1**
(sacou tudo), com uma massa achatada no meio. Ajustar uma média a isso esconde
mais do que revela — por isso :meth:`CCFDataset.distribution` traz explicitamente
as massas nos extremos, e é o que justifica, na documentação do modelo, a escolha
entre média agrupada e um modelo de resposta fracionária.

O que este módulo faz e o que ele **não** faz
---------------------------------------------
Faz o que não existia: montar a base de referência, calcular as quatro medidas,
limpar o dado (limite nulo, não sacado nulo, saque acima do limite,
winsorização), agrupar por segmento, calibrar o *downturn* e **testar** o
resultado contra o EAD realizado.

Não reimplementa motor de regressão: o ajuste **preditivo** do CCF em função de
características do cliente é um alvo contínuo em ``[0, 1]``, exatamente o caso do
:class:`~yggdrasil.credit_risk.model.ModelSegmenter` com
``task_type='regression'``. A base que sai de :func:`reference_dataset` já está no
formato que ele consome. E o *downturn* reaproveita
:func:`~yggdrasil.credit_risk.capital.parameters.ccf_downturn`, que já existia no
motor de capital — uma única fórmula no repositório.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

# Fonte ÚNICA do CCF downturn no repositório (quantil alto da distribuição).
from ..capital.parameters import ccf_downturn

#: Desenhos de base de referência suportados.
REFERENCE_METHODS = ("cohort", "fixed_horizon", "variable")

#: Medidas ex-post da exposição realizada.
MEASURES = ("ccf", "eadf", "auf", "ead")

#: Denominador natural de cada medida — é ele que pondera a média agrupada.
MEASURE_WEIGHT = {"ccf": "nao_sacado_ref", "eadf": "limite_ref",
                  "auf": "limite_ref", "ead": "limite_ref"}


def _month_ordinal(s: pd.Series) -> np.ndarray:
    """Ordinal mensal de uma coluna de data (diferença = nº exato de meses)."""
    return pd.PeriodIndex(pd.to_datetime(s), freq="M").asi8.astype("int64")


# ======================================================================
# Base de referência
# ======================================================================
@dataclass
class CCFDataset:
    """Base de referência do CCF — o resultado de :func:`reference_dataset`.

    Attributes
    ----------
    frame:
        Uma linha por par (contrato, data de referência) **elegível**, com as
        quatro medidas ex-post e as colunas de origem.
    method, horizon, measure:
        O desenho usado.
    excluded:
        Contagem das exclusões por motivo — a tabela que a validação pede para
        entender de onde saiu a amostra final.
    meta:
        Linhagem dos parâmetros de construção.
    """

    frame: pd.DataFrame
    method: str
    horizon: int
    measure: str = "ccf"
    excluded: Dict[str, int] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def values(self) -> pd.Series:
        """A série da medida escolhida (o que se agrupa, modela ou winsoriza)."""
        return self.frame[self.measure]

    def summary(self) -> pd.DataFrame:
        """Uma linha com o retrato da base: volume, medida e estatísticas."""
        v = self.values.to_numpy(dtype=float)
        return pd.DataFrame([{
            "method": self.method, "horizon": self.horizon, "measure": self.measure,
            "n_observacoes": len(self.frame),
            "n_contratos": int(self.frame["id"].nunique()),
            "meses_ate_default_medio": float(self.frame["meses_ate_default"].mean()),
            "media": float(np.mean(v)), "mediana": float(np.median(v)),
            "desvio": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
            "p10": float(np.quantile(v, 0.10)), "p90": float(np.quantile(v, 0.90)),
            "massa_em_0": float(np.mean(v <= 1e-9)),
            "massa_em_1": float(np.mean(v >= 1.0 - 1e-9)),
            "n_excluidos": int(sum(self.excluded.values())),
        }])

    def excluded_frame(self) -> pd.DataFrame:
        """As exclusões por motivo, em contagem e percentual do bruto."""
        total = int(sum(self.excluded.values())) + len(self.frame)
        return pd.DataFrame(
            [{"motivo": k, "n": v, "pct": (v / total if total else np.nan)}
             for k, v in self.excluded.items()]
        ).sort_values("n", ascending=False, ignore_index=True)

    def distribution(self, bins: int = 20) -> pd.DataFrame:
        """Distribuição da medida, com as **massas em 0 e em 1** destacadas.

        O CCF observado é bimodal: um pico em 0 (o cliente não tocou no limite) e
        outro em 1 (sacou tudo). A tabela separa esses dois pontos das faixas
        internas — é o que se leva para a documentação ao justificar o estimador
        escolhido."""
        v = self.values.to_numpy(dtype=float)
        em_zero, em_um = v <= 1e-9, v >= 1.0 - 1e-9
        meio = v[~(em_zero | em_um)]
        linhas = [{"faixa": "= 0", "n": int(em_zero.sum()), "pct": float(em_zero.mean())}]
        if meio.size:
            cortes = np.linspace(0.0, 1.0, int(bins) + 1)
            idx = np.clip(np.digitize(meio, cortes) - 1, 0, int(bins) - 1)
            for b in range(int(bins)):
                n = int((idx == b).sum())
                if n:
                    linhas.append({"faixa": f"({cortes[b]:.2f}, {cortes[b + 1]:.2f}]",
                                   "n": n, "pct": n / len(v)})
        linhas.append({"faixa": "= 1", "n": int(em_um.sum()), "pct": float(em_um.mean())})
        return pd.DataFrame(linhas)

    def pooled(self, by: Optional[str] = None, stat: str = "mean",
               alpha: float = 0.05) -> pd.DataFrame:
        """Atalho para :func:`pooled_ccf` sobre esta base."""
        return pooled_ccf(self, by=by, stat=stat, alpha=alpha)

    def downturn(self, quantile: float = 0.9) -> float:
        """CCF *downturn* — quantil alto da distribuição observada.

        Delega a :func:`~yggdrasil.credit_risk.capital.parameters.ccf_downturn`,
        a fonte única desta calibração no repositório."""
        return ccf_downturn(self.values.to_numpy(dtype=float), quantile=quantile)

    def plot(self, ax=None):
        """Histograma da medida com as massas nos extremos (matplotlib sob demanda)."""
        from .report import plot_ccf_distribution
        return plot_ccf_distribution(self, ax=ax)

    def __repr__(self) -> str:
        return (f"CCFDataset(method={self.method!r}, horizon={self.horizon}, "
                f"measure={self.measure!r}, n={len(self.frame)}, "
                f"media={float(self.values.mean()):.4f})")


def reference_dataset(
    df: pd.DataFrame,
    id_col: str = "id_contrato",
    date_col: str = "dt_ref",
    drawn_col: str = "sacado",
    limit_col: str = "limite",
    default_col: str = "default",
    method: str = "cohort",
    horizon: int = 12,
    measure: str = "ccf",
    ead_col: Optional[str] = None,
    segment_col: Optional[str] = None,
    extra_cols: Optional[Sequence[str]] = None,
    min_limit: float = 0.0,
    min_undrawn: float = 0.0,
    drop_over_limit: bool = False,
    clip: bool = True,
    winsorize: Optional[float] = None,
    cohort_anchor=None,
) -> CCFDataset:
    """Monta a base de referência do CCF a partir do painel de contratos rotativos.

    Parameters
    ----------
    df:
        Painel longo: uma linha por contrato × safra, com sacado, limite e a flag
        de *default*.
    id_col, date_col, drawn_col, limit_col, default_col:
        Nomes das colunas.
    method:
        Um de :data:`REFERENCE_METHODS` — ver a discussão no topo do módulo.
    horizon:
        Janela em períodos (``12`` = um ano).
    measure:
        Medida principal da base (:data:`MEASURES`). As quatro são sempre
        calculadas; ``measure`` define qual é a "a" medida em
        :attr:`CCFDataset.values`, no *clip*, na winsorização e no agrupamento.
    ead_col:
        Coluna com a exposição na data do *default*, se ela vier pronta. Sem
        ela, o EAD é o ``drawn_col`` observado na data do *default*.
    segment_col, extra_cols:
        Colunas do painel a carregar para a base (medidas na **data de
        referência**), para segmentar ou para alimentar um modelo preditivo.
    min_limit, min_undrawn:
        Pisos de elegibilidade. ``min_undrawn`` é o filtro decisivo do ``ccf``:
        sem limite disponível na referência o fator não é definível (divisão por
        zero) e a observação **não informa nada** sobre conversão.
    drop_over_limit:
        Descarta as referências em que o sacado já excede o limite (excesso
        autorizado, encargos capitalizados). Sem isso, a utilização passa de 100%
        e o não sacado fica negativo.
    clip:
        Recorta a medida principal em ``[0, 1]``. O CCF cru pode passar de 1
        (saque acima do limite disponível, encargos) ou ficar negativo
        (amortização antes do *default*); o recorte é a convenção usual, e a base
        guarda a versão crua em ``<measure>_bruto`` e a marca em
        ``recortado``.
    winsorize:
        Fração winsorizada em cada cauda **antes** do recorte (ex.: ``0.01`` para
        1%). ``None`` desliga.
    cohort_anchor:
        Data que ancora a grade de coortes em ``method='cohort'``. Sem ela, usa a
        primeira safra do painel.

    Returns
    -------
    CCFDataset
    """
    if method not in REFERENCE_METHODS:
        raise ValueError(f"method deve ser um de {REFERENCE_METHODS}; recebido {method!r}.")
    if measure not in MEASURES:
        raise ValueError(f"measure deve ser um de {MEASURES}; recebido {measure!r}.")
    if int(horizon) < 1:
        raise ValueError(f"horizon deve ser >= 1; recebido {horizon!r}.")
    obrigatorias = [id_col, date_col, drawn_col, limit_col, default_col]
    extras = [c for c in ([segment_col] + list(extra_cols or [])) if c]
    faltando = [c for c in obrigatorias + extras + ([ead_col] if ead_col else [])
                if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes no painel: {faltando}.")

    d = df[obrigatorias + extras + ([ead_col] if ead_col else [])].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    if d.duplicated(subset=[id_col, date_col]).any():
        raise ValueError("Há pares (contrato, safra) duplicados no painel.")
    d["_ord"] = _month_ordinal(d[date_col])
    d = d.sort_values([id_col, "_ord"], kind="mergesort")

    # --- data e exposição no default -------------------------------------
    quebras = d[pd.to_numeric(d[default_col], errors="coerce").fillna(0) == 1]
    if quebras.empty:
        raise ValueError("o painel não tem nenhum default — não há base de CCF a montar.")
    primeiro = quebras.groupby(id_col, sort=False).head(1)
    ead = pd.to_numeric(primeiro[ead_col if ead_col else drawn_col], errors="coerce")
    defaults = pd.DataFrame({
        id_col: primeiro[id_col].to_numpy(),
        "_ord_default": primeiro["_ord"].to_numpy(),
        "data_default": primeiro[date_col].to_numpy(),
        "ead": ead.to_numpy(dtype=float),
    })
    n_bruto_contratos = len(defaults)
    defaults = defaults[np.isfinite(defaults["ead"].to_numpy())]

    # --- datas de referência por desenho ----------------------------------
    H = int(horizon)
    if method == "fixed_horizon":
        pares = defaults.assign(_ord_ref=defaults["_ord_default"] - H)
    elif method == "cohort":
        ancora = (int(d["_ord"].min()) if cohort_anchor is None
                  else int(_month_ordinal(pd.Series([pd.to_datetime(cohort_anchor)]))[0]))
        passo = ((defaults["_ord_default"].to_numpy() - ancora) // H)
        pares = defaults.assign(_ord_ref=ancora + passo * H)
    else:  # variable — todas as referências da janela
        offsets = np.arange(1, H + 1)
        pares = defaults.loc[defaults.index.repeat(H)].copy()
        pares["_ord_ref"] = (pares["_ord_default"].to_numpy()
                             - np.tile(offsets, len(defaults)))
    pares = pares[pares["_ord_ref"] < pares["_ord_default"]]

    # --- junta os saldos na data de referência ----------------------------
    ref = d.rename(columns={"_ord": "_ord_ref", drawn_col: "sacado_ref",
                            limit_col: "limite_ref", date_col: "data_referencia"})
    base = pares.merge(
        ref[[id_col, "_ord_ref", "data_referencia", "sacado_ref", "limite_ref",
             default_col] + extras],
        on=[id_col, "_ord_ref"], how="inner",
    )
    n_sem_referencia = len(pares) - len(base)

    # A referência precisa ser de um momento ADIMPLENTE: se o contrato já estava
    # em default naquela data, o "não sacado" não é mais limite disponível.
    antes = len(base)
    base = base[pd.to_numeric(base[default_col], errors="coerce").fillna(0) == 0]
    n_ja_em_default = antes - len(base)
    base = base.drop(columns=[default_col])

    # --- higiene ------------------------------------------------------------
    for col in ("sacado_ref", "limite_ref"):
        base[col] = pd.to_numeric(base[col], errors="coerce")
    antes = len(base)
    base = base[np.isfinite(base["sacado_ref"]) & np.isfinite(base["limite_ref"])]
    n_nao_numerico = antes - len(base)

    base["nao_sacado_ref"] = base["limite_ref"] - base["sacado_ref"]
    base["over_limit_ref"] = base["nao_sacado_ref"] < 0

    antes = len(base)
    base = base[base["limite_ref"] > float(min_limit)]
    n_sem_limite = antes - len(base)

    n_over_limit = 0
    if drop_over_limit:
        antes = len(base)
        base = base[~base["over_limit_ref"]]
        n_over_limit = antes - len(base)

    antes = len(base)
    base = base[base["nao_sacado_ref"] > float(min_undrawn)]
    n_sem_nao_sacado = antes - len(base)

    if base.empty:
        raise ValueError(
            "a base de referência ficou vazia após os filtros — revise horizon/method "
            "(o contrato precisa existir e estar adimplente na data de referência) e "
            "os pisos min_limit/min_undrawn."
        )

    # --- as quatro medidas ex-post -------------------------------------------
    base["meses_ate_default"] = (base["_ord_default"] - base["_ord_ref"]).astype(int)
    base["utilizacao_ref"] = base["sacado_ref"] / base["limite_ref"]
    base["utilizacao_default"] = base["ead"] / base["limite_ref"]
    base["ccf"] = (base["ead"] - base["sacado_ref"]) / base["nao_sacado_ref"]
    base["eadf"] = base["ead"] / base["limite_ref"]
    base["auf"] = (base["ead"] - base["sacado_ref"]) / base["limite_ref"]
    base = base.rename(columns={id_col: "id"})
    base = base.drop(columns=["_ord_ref", "_ord_default"])

    # --- winsorização e recorte da medida principal ---------------------------
    base[f"{measure}_bruto"] = base[measure]
    if winsorize is not None:
        w = float(winsorize)
        if not (0.0 <= w < 0.5):
            raise ValueError(f"winsorize deve estar em [0, 0.5); recebido {winsorize!r}.")
        if w > 0:
            lo, hi = np.quantile(base[measure].to_numpy(dtype=float), [w, 1.0 - w])
            base[measure] = base[measure].clip(lo, hi)
    if clip and measure != "ead":
        base[measure] = base[measure].clip(0.0, 1.0)
    base["recortado"] = ~np.isclose(base[measure].to_numpy(dtype=float),
                                    base[f"{measure}_bruto"].to_numpy(dtype=float))

    excluidos = {
        "sem_observacao_na_referencia": int(n_sem_referencia),
        "ja_em_default_na_referencia": int(n_ja_em_default),
        "saldo_nao_numerico": int(n_nao_numerico),
        "limite_abaixo_do_piso": int(n_sem_limite),
        "sacado_acima_do_limite": int(n_over_limit),
        "nao_sacado_abaixo_do_piso": int(n_sem_nao_sacado),
    }
    return CCFDataset(
        frame=base.reset_index(drop=True), method=method, horizon=H, measure=measure,
        excluded=excluidos,
        meta={"min_limit": float(min_limit), "min_undrawn": float(min_undrawn),
              "drop_over_limit": bool(drop_over_limit), "clip": bool(clip),
              "winsorize": winsorize, "segment_col": segment_col,
              "n_contratos_com_default": int(n_bruto_contratos)},
    )


# ======================================================================
# Estimação agrupada
# ======================================================================
def pooled_ccf(data: Union[CCFDataset, pd.DataFrame], by: Optional[str] = None,
               stat: str = "mean", measure: Optional[str] = None,
               alpha: float = 0.05) -> pd.DataFrame:
    """CCF agrupado por segmento, com intervalo de confiança.

    Parameters
    ----------
    data:
        Um :class:`CCFDataset` ou o seu ``frame``.
    by:
        Coluna de segmentação. ``None`` devolve uma linha (a carteira toda).
    stat:
        ``'mean'`` (padrão), ``'median'`` ou ``'weighted'`` — média ponderada
        pelo denominador natural da medida (:data:`MEASURE_WEIGHT`: o **não
        sacado** para o CCF, o **limite** para EADF/AUF). A ponderada é a que
        preserva a soma do EAD da carteira; a simples trata cada contrato como
        uma observação.
    measure:
        Medida a agrupar. ``None`` usa a do :class:`CCFDataset`.
    alpha:
        Nível do IC (``0.05`` → 95%), pela aproximação normal da média.

    Returns
    -------
    pandas.DataFrame
        ``grupo``, ``n``, ``ccf`` (a estimativa), ``ic_inf``, ``ic_sup``,
        ``desvio``, ``mediana``, ``massa_em_0``, ``massa_em_1``.
    """
    if stat not in ("mean", "median", "weighted"):
        raise ValueError(f"stat deve ser 'mean', 'median' ou 'weighted'; recebido {stat!r}.")
    if isinstance(data, CCFDataset):
        frame, medida = data.frame, measure or data.measure
    else:
        frame, medida = pd.DataFrame(data), measure or "ccf"
    if medida not in frame.columns:
        raise ValueError(f"medida {medida!r} não está na base.")
    if by is not None and by not in frame.columns:
        raise ValueError(f"Coluna de segmentação {by!r} não encontrada na base.")

    from scipy.stats import norm
    z = float(norm.ppf(1.0 - alpha / 2.0))
    peso_col = MEASURE_WEIGHT[medida] if medida in MEASURE_WEIGHT else None

    def _uma(g: pd.DataFrame, rotulo) -> dict:
        v = g[medida].to_numpy(dtype=float)
        n = len(v)
        sd = float(np.std(v, ddof=1)) if n > 1 else np.nan
        if stat == "median":
            est = float(np.median(v))
        elif stat == "weighted":
            w = g[peso_col].to_numpy(dtype=float) if peso_col in g.columns else np.ones(n)
            w = np.clip(w, 0.0, None)
            est = float(np.average(v, weights=w)) if w.sum() > 0 else float(np.mean(v))
        else:
            est = float(np.mean(v))
        erro = sd / np.sqrt(n) if n > 1 else np.nan
        return {
            "grupo": rotulo, "n": n, "ccf": est,
            "ic_inf": est - z * erro if np.isfinite(erro) else np.nan,
            "ic_sup": est + z * erro if np.isfinite(erro) else np.nan,
            "desvio": sd, "mediana": float(np.median(v)),
            "massa_em_0": float(np.mean(v <= 1e-9)),
            "massa_em_1": float(np.mean(v >= 1.0 - 1e-9)),
        }

    if by is None:
        linhas = [_uma(frame, "__global__")]
    else:
        linhas = [_uma(g, rot) for rot, g in frame.groupby(by, sort=True)]
    out = pd.DataFrame(linhas)
    out["estatistica"] = stat
    out["medida"] = medida
    return out


# ======================================================================
# Reconstrução do EAD e backtest
# ======================================================================
def ead_from_ccf(drawn, limit, ccf, floor_at_drawn: bool = False) -> np.ndarray:
    """``EAD = sacado + CCF · (limite − sacado)`` — a fórmula de uso do parâmetro."""
    return ead_from_measure(drawn, limit, ccf, measure="ccf", floor_at_drawn=floor_at_drawn)


def ead_from_measure(drawn, limit, value, measure: str = "ccf",
                     floor_at_drawn: bool = False) -> np.ndarray:
    """Reconstrói o EAD a partir de qualquer uma das medidas de :data:`MEASURES`.

    ``ccf`` → ``sacado + valor·(limite − sacado)`` · ``eadf`` → ``valor·limite`` ·
    ``auf`` → ``sacado + valor·limite`` · ``ead`` → o próprio valor.

    Aplicada à **medida observada** de uma linha da base de referência, qualquer
    uma das quatro devolve exatamente o EAD realizado — elas são
    reparametrizações da mesma observação, e é essa identidade que permite
    compará-las no mesmo *backtest*.

    Parameters
    ----------
    drawn, limit, value:
        Sacado e limite na data de referência, e o valor da medida.
    measure:
        Uma de :data:`MEASURES`.
    floor_at_drawn:
        Piso no valor já sacado (``False`` por padrão). É uma escolha de
        **política de uso**, não da fórmula: impede que a exposição prevista
        fique abaixo do que o cliente já devia, o que costuma ser conservador e
        razoável na projeção — mas **quebra a identidade** com o EAD realizado
        quando o contrato amortizou antes de quebrar, e aí viesa a estimativa
        para cima. Ligue conscientemente, na projeção; deixe desligado na
        validação. O resultado é sempre recortado por baixo em zero.
    """
    if measure not in MEASURES:
        raise ValueError(f"measure deve ser um de {MEASURES}; recebido {measure!r}.")
    s = np.asarray(drawn, dtype=float)
    l = np.asarray(limit, dtype=float)
    v = np.asarray(value, dtype=float)
    if measure == "ead":
        bruto = v
    elif measure == "ccf":
        bruto = s + v * (l - s)
    elif measure == "eadf":
        bruto = v * l
    else:
        bruto = s + v * l
    if floor_at_drawn:
        bruto = np.maximum(bruto, s)
    return np.maximum(bruto, 0.0)


def backtest_ead(data: Union[CCFDataset, pd.DataFrame], value: Union[float, Mapping, str],
                 by: Optional[str] = None, measure: Optional[str] = None) -> pd.DataFrame:
    """EAD previsto × realizado — o teste que decide se o parâmetro serve.

    Aplica a estimativa do parâmetro à base de referência, reconstrói o EAD e
    compara com o realizado. O viés em **moeda** é o número que importa: um CCF
    com erro médio zero por contrato pode subestimar a carteira inteira se errar
    justamente nos contratos grandes.

    Parameters
    ----------
    data:
        Um :class:`CCFDataset` ou o seu ``frame``.
    value:
        A estimativa: um escalar, um ``{segmento: valor}`` (exige ``by``) ou o
        **nome de uma coluna** da base — que é como se testa um modelo preditivo
        já escorado.
    by:
        Coluna de segmentação para a tabela de resultados (e para casar o
        dicionário de estimativas).
    measure:
        Medida usada na reconstrução. ``None`` usa a do :class:`CCFDataset`.

    Returns
    -------
    pandas.DataFrame
        Por grupo (e uma linha ``__global__``): ``n``, ``ead_previsto``,
        ``ead_realizado``, ``vies`` (moeda), ``vies_relativo``, ``mae``,
        ``rmse``, ``erro_medio`` e ``erro_absoluto_relativo``.
    """
    if isinstance(data, CCFDataset):
        frame, medida = data.frame.copy(), measure or data.measure
    else:
        frame, medida = pd.DataFrame(data).copy(), measure or "ccf"
    for col in ("sacado_ref", "limite_ref", "ead"):
        if col not in frame.columns:
            raise ValueError(f"coluna {col!r} ausente — a base não veio de reference_dataset?")

    if isinstance(value, str):
        if value not in frame.columns:
            raise ValueError(f"coluna de estimativa {value!r} não encontrada na base.")
        est = frame[value].to_numpy(dtype=float)
    elif isinstance(value, Mapping):
        if by is None:
            raise ValueError("estimativa por segmento exige `by`.")
        if by not in frame.columns:
            raise ValueError(f"Coluna de segmentação {by!r} não encontrada na base.")
        chaves = frame[by].to_numpy(dtype=object)
        faltando = {k for k in pd.unique(chaves) if k not in value}
        if faltando:
            raise ValueError(f"segmentos sem estimativa: {sorted(map(str, faltando))[:6]}.")
        est = np.array([float(value[k]) for k in chaves], dtype=float)
    else:
        est = np.full(len(frame), float(value))

    frame["ead_previsto"] = ead_from_measure(
        frame["sacado_ref"].to_numpy(dtype=float),
        frame["limite_ref"].to_numpy(dtype=float), est, measure=medida,
    )
    frame["_erro"] = frame["ead_previsto"] - frame["ead"]

    def _uma(g: pd.DataFrame, rotulo) -> dict:
        prev, real = float(g["ead_previsto"].sum()), float(g["ead"].sum())
        erro = g["_erro"].to_numpy(dtype=float)
        return {
            "grupo": rotulo, "n": len(g),
            "ead_previsto": prev, "ead_realizado": real,
            "vies": prev - real,
            "vies_relativo": (prev / real - 1.0) if real else np.nan,
            "erro_medio": float(np.mean(erro)),
            "mae": float(np.mean(np.abs(erro))),
            "rmse": float(np.sqrt(np.mean(erro ** 2))),
            "erro_absoluto_relativo": (float(np.sum(np.abs(erro))) / real) if real else np.nan,
        }

    linhas = [_uma(frame, "__global__")]
    if by is not None:
        linhas += [_uma(g, rot) for rot, g in frame.groupby(by, sort=True)]
    out = pd.DataFrame(linhas)
    out["medida"] = medida
    return out


def compare_measures(data: CCFDataset, by: Optional[str] = None,
                     stat: str = "mean") -> pd.DataFrame:
    """Compara as quatro medidas no mesmo *backtest* de EAD.

    É o quadro que a validação pede: estimar por CCF, EADF, AUF e EAD direto,
    reconstruir a exposição com cada um e ver qual erra menos em moeda. A
    literatura encontra o ``ccf`` na frente na maioria das carteiras — mas isso
    é resultado empírico, não teorema, e depende da carteira."""
    linhas = []
    for medida in MEASURES:
        if medida not in data.frame.columns and medida != "ead":
            continue
        agrupado = pooled_ccf(data, by=by, stat=stat, measure=medida)
        estimativa = (dict(zip(agrupado["grupo"], agrupado["ccf"])) if by is not None
                      else float(agrupado["ccf"].iloc[0]))
        bt = backtest_ead(data, estimativa, by=by, measure=medida)
        linhas.append(bt[bt["grupo"] == "__global__"].assign(medida=medida))
    out = pd.concat(linhas, ignore_index=True)
    return out.sort_values("erro_absoluto_relativo", ignore_index=True)


def ccf_psi(data: Union[CCFDataset, pd.DataFrame], by: str, reference=None,
            bins: int = 10, measure: Optional[str] = None) -> pd.DataFrame:
    """PSI da distribuição do CCF entre safras (ou entre quaisquer grupos).

    Usa as faixas do grupo de referência e a fórmula única do repositório
    (:func:`~yggdrasil.credit_risk._common.psi_from_shares`). Serve ao
    monitoramento: se a distribuição do CCF muda, a estimativa em produção
    envelheceu."""
    from .._common import classifica_psi, psi_from_shares

    if isinstance(data, CCFDataset):
        frame, medida = data.frame, measure or data.measure
    else:
        frame, medida = pd.DataFrame(data), measure or "ccf"
    if by not in frame.columns:
        raise ValueError(f"Coluna {by!r} não encontrada na base.")
    grupos = list(pd.unique(frame[by]))
    if len(grupos) < 2:
        raise ValueError(f"são necessários ao menos 2 grupos em {by!r} para calcular PSI.")
    ref = grupos[0] if reference is None else reference
    if ref not in grupos:
        raise ValueError(f"grupo de referência {ref!r} ausente em {by!r}.")

    base_ref = frame.loc[frame[by] == ref, medida].to_numpy(dtype=float)
    cortes = np.unique(np.quantile(base_ref, np.linspace(0, 1, int(bins) + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    p_ref = np.histogram(base_ref, bins=cortes)[0] / len(base_ref)

    linhas = []
    for g in grupos:
        v = frame.loc[frame[by] == g, medida].to_numpy(dtype=float)
        p = np.histogram(v, bins=cortes)[0] / max(len(v), 1)
        psi = psi_from_shares(p_ref, p)
        linhas.append({"grupo": g, "n": len(v), "psi": psi,
                       "classificacao": classifica_psi(psi),
                       "media": float(np.mean(v)) if len(v) else np.nan,
                       "referencia": g == ref})
    return pd.DataFrame(linhas)


__all__ = [
    "CCFDataset", "reference_dataset", "pooled_ccf", "ead_from_ccf", "ead_from_measure",
    "backtest_ead", "compare_measures", "ccf_psi", "ccf_downturn",
    "REFERENCE_METHODS", "MEASURES", "MEASURE_WEIGHT",
]
