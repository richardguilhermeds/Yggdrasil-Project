"""
Contrato de dados: o painel de contratos (:class:`ContractPanel`)
=================================================================
Todo o eixo **lifetime** parte do mesmo objeto: um painel **longo** com uma linha
por contrato × safra de observação, onde cada linha diz *quantos meses o contrato
tem de vida* (a idade, ou *months on book*) e *se ele quebrou naquele período*.

É esse formato — e não a base transversal de escoragem — que sustenta as três
famílias de motores de PD lifetime:

* **safra/vintage** — a taxa marginal de *default* por idade, direto da contagem;
* **sobrevivência** — Kaplan-Meier e o *hazard* em tempo discreto, que exigem o
  formato pessoa-período e o tratamento correto de **censura à direita**;
* **migração** — a matriz de transição entre ratings, que exige a trajetória
  ordenada de cada contrato.

O ponto delicado é a **censura**: um contrato originado há 6 meses não é um
contrato que sobreviveu 60 meses; ele apenas ainda não foi observado até lá.
Somar os dois na mesma taxa subestima o risco das idades altas. Por isso a base
em risco (:meth:`ContractPanel.at_risk`) é recontada **idade a idade**, e não
fixada no total de contratos.

O segundo ponto é o *default* ser **absorvente**: depois da primeira quebra o
contrato sai do conjunto em risco. Observações posteriores ao primeiro *default*
são descartadas por padrão (``drop_post_default=True``) — mantê-las contaria a
mesma quebra várias vezes e achataria a curva.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

#: Períodos por ano de cada frequência suportada — converte horizonte em anos
#: (``pd_12m`` é "12 meses", não "12 períodos").
PERIODS_PER_YEAR = {"M": 12, "Q": 4, "S": 2, "A": 1, "Y": 1}


def periods_per_year(freq: str) -> int:
    """Nº de períodos por ano da frequência (``'M'`` → 12, ``'Q'`` → 4, ``'A'`` → 1)."""
    f = str(freq).upper()[:1]
    if f not in PERIODS_PER_YEAR:
        raise ValueError(
            f"freq deve ser uma de {sorted(set(PERIODS_PER_YEAR))}; recebido {freq!r}."
        )
    return PERIODS_PER_YEAR[f]


def _month_ordinal(s: pd.Series) -> np.ndarray:
    """Ordinal mensal de uma coluna de data — a base da idade em meses.

    Usa ``PeriodIndex.asi8`` (o ordinal inteiro do período) em vez de subtrair
    datas: a diferença de ordinais mensais é exatamente o nº de meses, sem o
    arredondamento de 30/31 dias que a subtração de ``Timestamp`` produz.
    """
    return pd.PeriodIndex(pd.to_datetime(s), freq="M").asi8.astype("int64")


@dataclass
class ContractPanel:
    """Painel longo de contratos — o contrato de dados do eixo *lifetime*.

    Parameters
    ----------
    df:
        DataFrame com uma linha por contrato × safra de observação.
    id_col:
        Coluna identificadora do contrato (ou do cliente, se a unidade de risco
        for o cliente).
    date_col:
        Safra de observação (data). Define a ordem da trajetória e as safras do
        *backtest*.
    default_col:
        Flag ``0/1`` de **entrada em *default*** no período (evento, não estado).
        Se a base traz o *estado* (permanece 1 enquanto o contrato está em
        *default*), o corte por ``drop_post_default`` já reduz ao evento.
    age_col:
        Idade do contrato em períodos (*months on book*). Se ``None``, é
        derivada: de ``origin_col`` quando informado, senão da posição da
        observação dentro da trajetória (0, 1, 2, ...) — a saída fica na coluna
        :data:`AGE_DERIVED`.
    origin_col:
        Safra de originação (data). Usada só para derivar a idade.
    term_col:
        Prazo **remanescente** em períodos (opcional). Trunca a curva *lifetime*
        contrato a contrato: um contrato com 8 meses de prazo não acumula PD por
        60 meses.
    segment_col:
        Segmento, produto ou rating — a chave das curvas por grupo.
    exposure_col:
        Saldo/exposição, para ponderar as taxas e montar o ECL.
    freq:
        Frequência do painel (``'M'`` mensal, ``'Q'`` trimestral, ``'A'`` anual).
    drop_post_default:
        Descarta as observações **posteriores** ao primeiro *default* de cada
        contrato (padrão ``True`` — *default* absorvente).

    Notes
    -----
    O painel **não** precisa ser balanceado nem contínuo: lacunas de safra são
    aceitas (a idade é que ordena o risco). Pares ``(id, data)`` duplicados são
    erro — a trajetória ficaria ambígua.
    """

    df: pd.DataFrame
    id_col: str = "id_contrato"
    date_col: str = "dt_ref"
    default_col: str = "default"
    age_col: Optional[str] = None
    origin_col: Optional[str] = None
    term_col: Optional[str] = None
    segment_col: Optional[str] = None
    exposure_col: Optional[str] = None
    freq: str = "M"
    drop_post_default: bool = True

    #: Nome da coluna de idade criada quando ``age_col`` não é informada.
    AGE_DERIVED = "idade"

    _n_post_default: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError("df deve ser um pandas.DataFrame.")
        periods_per_year(self.freq)  # valida a frequência

        obrigatorias = [self.id_col, self.date_col, self.default_col]
        opcionais = [self.age_col, self.origin_col, self.term_col,
                     self.segment_col, self.exposure_col]
        faltando = [c for c in obrigatorias + [o for o in opcionais if o] if c not in self.df.columns]
        if faltando:
            raise ValueError(f"Colunas ausentes no painel: {faltando}.")

        data = self.df.copy()
        data[self.date_col] = pd.to_datetime(data[self.date_col])
        if data.duplicated(subset=[self.id_col, self.date_col]).any():
            raise ValueError(
                "Há pares (contrato, safra) duplicados — a trajetória fica ambígua. "
                "Agregue ou remova as duplicatas antes de montar o painel."
            )

        # --- flag de default: precisa ser binária -----------------------
        alvo = pd.to_numeric(data[self.default_col], errors="coerce")
        if alvo.isna().any():
            raise ValueError(f"{self.default_col!r} contém valores não numéricos/NaN.")
        valores = set(np.unique(alvo.to_numpy()))
        if not valores <= {0.0, 1.0}:
            raise ValueError(
                f"{self.default_col!r} deve ser binária (0/1); valores encontrados: "
                f"{sorted(valores)[:6]}."
            )
        data[self.default_col] = alvo.astype(int)

        data = data.sort_values([self.id_col, self.date_col], kind="mergesort")

        # --- idade -------------------------------------------------------
        if self.age_col is None:
            self.age_col = self.AGE_DERIVED
            if self.origin_col is not None:
                data[self.age_col] = (
                    _month_ordinal(data[self.date_col]) - _month_ordinal(data[self.origin_col])
                )
                if self.freq.upper()[:1] != "M":
                    # meses → períodos da frequência do painel
                    passo = 12 // periods_per_year(self.freq)
                    data[self.age_col] = data[self.age_col] // passo
            else:
                # Sem originação: a idade é a posição na própria trajetória.
                data[self.age_col] = data.groupby(self.id_col, sort=False).cumcount()
        idade = pd.to_numeric(data[self.age_col], errors="coerce")
        if idade.isna().any():
            raise ValueError(f"{self.age_col!r} contém valores não numéricos/NaN.")
        if (idade < 0).any():
            raise ValueError(
                f"{self.age_col!r} tem idades negativas — a safra de observação é "
                "anterior à de originação em alguma linha."
            )
        data[self.age_col] = idade.astype(int)

        # --- default absorvente ------------------------------------------
        if self.drop_post_default:
            ja_quebrou = (
                data.groupby(self.id_col, sort=False)[self.default_col].cumsum()
                - data[self.default_col]
            )
            manter = ja_quebrou == 0
            self._n_post_default = int((~manter).sum())
            data = data[manter]

        if data.empty:
            raise ValueError("O painel ficou vazio após as validações.")
        self.df = data.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Conveniências
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return (
            f"ContractPanel(n_obs={len(self.df)}, n_contratos={self.n_contracts}, "
            f"idade_max={self.max_age}, freq={self.freq!r})"
        )

    @property
    def n_contracts(self) -> int:
        """Nº de contratos distintos no painel."""
        return int(self.df[self.id_col].nunique())

    @property
    def max_age(self) -> int:
        """Maior idade observada (o horizonte máximo suportado pelos dados)."""
        return int(self.df[self.age_col].max())

    @property
    def periods_per_year(self) -> int:
        """Períodos por ano da frequência do painel."""
        return periods_per_year(self.freq)

    def segments(self) -> List:
        """Segmentos distintos (lista vazia se o painel não tem ``segment_col``)."""
        if self.segment_col is None:
            return []
        return sorted(pd.unique(self.df[self.segment_col].dropna()).tolist())

    def by(self, segment_col: Optional[str] = None) -> Dict[object, "ContractPanel"]:
        """Quebra o painel por segmento, devolvendo ``{segmento: ContractPanel}``.

        Cada sub-painel já vem validado; a quebra não recalcula idade nem
        reaplica o corte pós-*default* (o painel-mãe já os resolveu)."""
        col = segment_col or self.segment_col
        if col is None:
            raise ValueError("Informe segment_col (no painel ou na chamada) para quebrar por grupo.")
        if col not in self.df.columns:
            raise ValueError(f"Coluna {col!r} não encontrada no painel.")
        saida: Dict[object, "ContractPanel"] = {}
        for valor, parte in self.df.groupby(col, sort=True):
            saida[valor] = ContractPanel(
                parte.reset_index(drop=True),
                id_col=self.id_col, date_col=self.date_col, default_col=self.default_col,
                age_col=self.age_col, origin_col=None, term_col=self.term_col,
                segment_col=col, exposure_col=self.exposure_col, freq=self.freq,
                drop_post_default=False,   # já aplicado no painel-mãe
            )
        return saida

    # ------------------------------------------------------------------
    # Base em risco por idade — o insumo das curvas empíricas
    # ------------------------------------------------------------------
    def at_risk(self, max_age: Optional[int] = None, weighted: bool = False) -> pd.DataFrame:
        """Tabela de vida: nº em risco, quebras e censura **por idade**.

        Esta é a tabela que o resto do pacote consome. Para cada idade ``t``:

        ``n_em_risco``
            contratos observados naquela idade **ainda sem *default***;
        ``n_default``
            quebras ocorridas naquela idade;
        ``n_censurado``
            contratos cuja última observação foi naquela idade **sem** quebrar —
            saíram do risco por fim de janela, quitação ou baixa, e não por
            *default*. É a censura à direita;
        ``hazard``
            ``n_default / n_em_risco`` — a PD **condicional** da idade.

        Parameters
        ----------
        max_age:
            Trunca a tabela nesta idade. ``None`` usa a idade máxima observada.
        weighted:
            Pondera por ``exposure_col`` em vez de contar contratos (``hazard``
            passa a ser a taxa de *default* em saldo). Exige ``exposure_col``.

        Returns
        -------
        pandas.DataFrame
            Indexado por ``idade`` (0 .. ``max_age``), sem lacunas — idades sem
            observação aparecem com ``n_em_risco = 0`` e ``hazard`` NaN.
        """
        if weighted and self.exposure_col is None:
            raise ValueError("weighted=True exige exposure_col no painel.")

        d = self.df
        idades = d[self.age_col].to_numpy()
        eventos = d[self.default_col].to_numpy()
        peso = (
            pd.to_numeric(d[self.exposure_col], errors="coerce").fillna(0.0).to_numpy()
            if weighted else np.ones(len(d), dtype=float)
        )

        topo = int(self.max_age if max_age is None else max_age)
        if topo < 0:
            raise ValueError(f"max_age deve ser >= 0; recebido {max_age!r}.")
        eixo = np.arange(topo + 1)

        dentro = idades <= topo
        em_risco = np.bincount(idades[dentro], weights=peso[dentro], minlength=topo + 1)[: topo + 1]
        quebras = np.bincount(idades[dentro], weights=(peso * eventos)[dentro],
                              minlength=topo + 1)[: topo + 1]

        # Censura: última observação do contrato, sem default nela.
        ultima = d.groupby(self.id_col, sort=False).tail(1)
        sem_quebra = ultima[ultima[self.default_col] == 0]
        idade_fim = sem_quebra[self.age_col].to_numpy()
        peso_fim = (
            pd.to_numeric(sem_quebra[self.exposure_col], errors="coerce").fillna(0.0).to_numpy()
            if weighted else np.ones(len(sem_quebra), dtype=float)
        )
        dentro_fim = idade_fim <= topo
        censura = np.bincount(idade_fim[dentro_fim], weights=peso_fim[dentro_fim],
                              minlength=topo + 1)[: topo + 1]

        with np.errstate(divide="ignore", invalid="ignore"):
            hazard = np.where(em_risco > 0, quebras / em_risco, np.nan)

        out = pd.DataFrame(
            {
                "n_em_risco": em_risco,
                "n_default": quebras,
                "n_censurado": censura,
                "hazard": hazard,
            },
            index=pd.Index(eixo, name="idade"),
        )
        if not weighted:
            for col in ("n_em_risco", "n_default", "n_censurado"):
                out[col] = out[col].astype(int)
        return out

    # ------------------------------------------------------------------
    # Formato pessoa-período — o insumo do hazard em tempo discreto
    # ------------------------------------------------------------------
    def spells(self, features: Optional[List[str]] = None,
               max_age: Optional[int] = None) -> pd.DataFrame:
        """Painel pessoa-período pronto para a regressão de *hazard*.

        Cada linha é um período **em risco** de um contrato, com a idade, o
        evento e as covariáveis pedidas. Como o painel já vem truncado no
        primeiro *default*, o próprio ``df`` é o formato pessoa-período — este
        método apenas seleciona colunas, trunca a idade e valida as features.
        """
        cols = [self.id_col, self.date_col, self.age_col, self.default_col]
        for extra in (self.segment_col, self.exposure_col, self.term_col):
            if extra and extra not in cols:
                cols.append(extra)
        if features:
            faltando = [c for c in features if c not in self.df.columns]
            if faltando:
                raise ValueError(f"Features ausentes no painel: {faltando}.")
            cols += [c for c in features if c not in cols]

        out = self.df[cols]
        if max_age is not None:
            out = out[out[self.age_col] <= int(max_age)]
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """Uma linha com o retrato do painel (para o relatório e a governança)."""
        d = self.df
        n_def = int(d[self.default_col].sum())
        linha = {
            "n_obs": len(d),
            "n_contratos": self.n_contracts,
            "n_defaults": n_def,
            "taxa_default_contratos": n_def / max(self.n_contracts, 1),
            "idade_min": int(d[self.age_col].min()),
            "idade_max": self.max_age,
            "safra_min": d[self.date_col].min(),
            "safra_max": d[self.date_col].max(),
            "n_segmentos": len(self.segments()),
            "obs_pos_default_descartadas": self._n_post_default,
            "freq": self.freq,
        }
        return pd.DataFrame([linha])


__all__ = ["ContractPanel", "PERIODS_PER_YEAR", "periods_per_year"]
