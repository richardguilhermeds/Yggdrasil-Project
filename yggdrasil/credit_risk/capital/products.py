"""
Presets por produto — particularidades de capital econômico (Guia §5, Tabela 3)
===============================================================================
A segmentação é a espinha dorsal do modelo, mas cada **produto** de varejo tem
uma física de risco distinta que precisa ser respeitada na calibração dos
parâmetros de capital econômico (Seção 5 e Tabela 3 do guia). Este módulo
**codifica** essas particularidades como *presets* reutilizáveis, de modo que a
montagem de uma carteira (:class:`~yggdrasil.credit_risk.capital.portfolio.Segment`)
já parta de valores plausíveis e regulatoriamente coerentes para cada produto.

O que muda de produto para produto (o miolo da Tabela 3):

* **Cartão de crédito** — o parâmetro crítico é a **EAD/CCF** do limite rotativo
  (o cliente saca limite *antes* do default); PD alta e ciclo-sensível, LGD alta
  (sem garantia), correlação com o ciclo **alta**. Classe Basileia:
  ``revolving``. Segmentar por *score* × *revolver/transactor* × faixa de limite.
* **Consignado** — o crítico é a **PD por convênio** e eventos discretos (perda
  de margem, troca do ente pagador, regulação); PD baixa/estável, LGD baixa a
  média (margem consignável + seguro prestamista), correlação baixa a média.
  Classe ``other_retail``. Segmentar por convênio (INSS, público
  federal/estadual/municipal, privado) × *score* × prazo.
* **Veículos (CDC)** — o crítico é a **LGD *downturn*** (o colateral é
  procíclico: em recessão o preço do usado cai junto com a inadimplência); PD
  média, LGD média (garantia real), correlação média porém com **correlação
  adversa PD–LGD** forte via preço de usados. Classe ``other_retail``.
  Segmentar por *score* × LTV × idade do bem.
* **Demais** — crédito pessoal comporta-se como cartão (rotativo sem garantia);
  imobiliário como veículos, porém com **horizonte > 1 ano** (a garantia amortece
  a LGD, mas o prazo longo quebra a hipótese de horizonte anual); cheque especial
  é rotativo com CCF ainda **mais instável** que o cartão.

Cada preset carrega um ``rho_sugerido`` de correlação de **ativos** (o ``rho`` de
Vasicek usado pelos motores ASRF e Monte Carlo), calibrado na ordem de grandeza
típica do produto, e as recomendações de LGD estocástica / correlação PD–LGD
que a simulação deve usar. São **pontos de partida** para a calibração interna,
não números regulatórios de Pilar 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd


# ======================================================================
# Preset de produto
# ======================================================================
@dataclass(frozen=True)
class ProductPreset:
    """Particularidades de capital econômico de um produto de varejo.

    Reproduz uma linha da Tabela 3 do guia mais as recomendações da §5. Os
    campos numéricos (``rho_sugerido``, ``lgd_vol_sugerido``,
    ``pd_lgd_corr_sugerido``) são **pontos de partida** plausíveis para a
    calibração interna, e não parâmetros regulatórios fechados.

    Parameters
    ----------
    nome:
        Chave curta do produto (ex.: ``"cartao"``).
    descricao:
        Descrição pedagógica do produto e da sua física de risco.
    basel_class:
        Classe de exposição de varejo de Basileia associada — um de
        ``"revolving"`` (rotativo qualificado), ``"other_retail"`` (demais
        varejo), ``"mortgage"`` (imobiliário residencial) ou ``"corporate"``.
        Governa a fórmula regulatória de correlação de referência.
    rho_sugerido:
        Correlação de **ativos** (parâmetro ``rho`` de Vasicek) sugerida para o
        produto, em ``[0, 1)``. Ordem de grandeza típica: cartão ``~0.08–0.12``,
        consignado ``~0.03–0.05``, veículos ``~0.06–0.09``.
    parametro_critico:
        O parâmetro que mais move o capital do produto (onde concentrar a
        validação): ``"EAD/CCF"``, ``"PD por convênio"``, ``"LGD downturn"``...
    nivel_pd:
        Nível qualitativo da PD (``"baixa"``, ``"média"``, ``"alta"``...).
    nivel_lgd:
        Nível qualitativo da LGD.
    correlacao_ciclo:
        Sensibilidade ao ciclo econômico (governa ``rho`` e o estresse de
        correlação): ``"baixa"``, ``"média"``, ``"alta"``.
    segmentacao_recomendada:
        Eixos de segmentação recomendados pela §5 (lista de dimensões).
    risco_especifico:
        O risco idiossincrático do produto a monitorar (ex.: saque de limite
        pré-default, perda de margem, correlação adversa PD–LGD).
    lgd_estocastica:
        Se a simulação de Monte Carlo deve tratar a LGD como **estocástica**
        (``True`` quando a severidade é materialmente incerta / procíclica).
    lgd_vol_sugerido:
        Desvio-padrão sugerido da LGD para a simulação (``0`` quando
        ``lgd_estocastica`` é ``False``).
    pd_lgd_corr_sugerido:
        Correlação adversa PD–LGD sugerida (``> 0`` onde recessão eleva *ambos*,
        como em veículos e imobiliário via preço do colateral).
    horizonte_1ano_ok:
        Se o horizonte anual do modelo é adequado (``False`` para imobiliário,
        cujo prazo longo exige horizonte plurianual).
    observacoes:
        Notas de calibração e ressalvas específicas do produto.
    """

    nome: str
    descricao: str
    basel_class: str
    rho_sugerido: float
    parametro_critico: str
    nivel_pd: str
    nivel_lgd: str
    correlacao_ciclo: str
    segmentacao_recomendada: List[str]
    risco_especifico: str
    lgd_estocastica: bool
    lgd_vol_sugerido: float
    pd_lgd_corr_sugerido: float
    horizonte_1ano_ok: bool
    observacoes: str

    def __post_init__(self) -> None:
        classes_validas = {"revolving", "other_retail", "mortgage", "corporate"}
        if self.basel_class not in classes_validas:
            raise ValueError(
                f"[{self.nome}] basel_class inválida: {self.basel_class!r}. "
                f"Use uma de {sorted(classes_validas)}."
            )
        if not (0.0 <= self.rho_sugerido < 1.0):
            raise ValueError(
                f"[{self.nome}] rho_sugerido deve estar em [0, 1); "
                f"recebido {self.rho_sugerido!r}."
            )
        if self.lgd_vol_sugerido < 0.0:
            raise ValueError(
                f"[{self.nome}] lgd_vol_sugerido deve ser >= 0; "
                f"recebido {self.lgd_vol_sugerido!r}."
            )
        if not (-1.0 <= self.pd_lgd_corr_sugerido <= 1.0):
            raise ValueError(
                f"[{self.nome}] pd_lgd_corr_sugerido deve estar em [-1, 1]; "
                f"recebido {self.pd_lgd_corr_sugerido!r}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Representação em dicionário (útil para montar a Tabela 3 tabular)."""
        return {
            "produto": self.nome,
            "descricao": self.descricao,
            "basel_class": self.basel_class,
            "rho_sugerido": self.rho_sugerido,
            "parametro_critico": self.parametro_critico,
            "nivel_pd": self.nivel_pd,
            "nivel_lgd": self.nivel_lgd,
            "correlacao_ciclo": self.correlacao_ciclo,
            "segmentacao_recomendada": ", ".join(self.segmentacao_recomendada),
            "risco_especifico": self.risco_especifico,
            "lgd_estocastica": self.lgd_estocastica,
            "lgd_vol_sugerido": self.lgd_vol_sugerido,
            "pd_lgd_corr_sugerido": self.pd_lgd_corr_sugerido,
            "horizonte_1ano_ok": self.horizonte_1ano_ok,
            "observacoes": self.observacoes,
        }


# ======================================================================
# A Tabela 3 codificada — presets por produto
# ======================================================================
PRESETS: Dict[str, ProductPreset] = {
    "cartao": ProductPreset(
        nome="cartao",
        descricao=(
            "Cartão de crédito rotativo. Produto sem garantia, de alta rotatividade "
            "e forte sensibilidade ao ciclo: em deterioração, o cliente saca o limite "
            "disponível antes de inadimplir, inflando a exposição efetiva no default."
        ),
        basel_class="revolving",
        rho_sugerido=0.10,
        parametro_critico="EAD/CCF (fator de conversão do limite rotativo)",
        nivel_pd="alta e ciclo-sensível",
        nivel_lgd="alta (sem garantia)",
        correlacao_ciclo="alta",
        segmentacao_recomendada=[
            "score",
            "revolver vs. transactor",
            "faixa de limite",
        ],
        risco_especifico=(
            "saque de limite pré-default: o CCF sobe justamente quando o risco piora, "
            "elevando a EAD downturn."
        ),
        lgd_estocastica=False,
        lgd_vol_sugerido=0.0,
        pd_lgd_corr_sugerido=0.0,
        horizonte_1ano_ok=True,
        observacoes=(
            "Concentre a validação na EAD/CCF downturn e na PD por safra. Correlação de "
            "ativos no topo da faixa de varejo rotativo (~0.08–0.12) pela alta co-movência "
            "com o ciclo."
        ),
    ),
    "consignado": ProductPreset(
        nome="consignado",
        descricao=(
            "Crédito consignado com desconto em folha/benefício. Risco baixo e estável: "
            "a parcela é retida na fonte (margem consignável) e há seguro prestamista, "
            "mas o risco migra para eventos discretos ligados ao ente pagador e à regulação."
        ),
        basel_class="other_retail",
        rho_sugerido=0.04,
        parametro_critico="PD por convênio (e eventos discretos de margem/ente pagador)",
        nivel_pd="baixa e estável",
        nivel_lgd="baixa a média (margem + seguro prestamista)",
        correlacao_ciclo="baixa a média",
        segmentacao_recomendada=[
            "convênio (INSS, público fed/est/mun, privado)",
            "score",
            "prazo",
        ],
        risco_especifico=(
            "perda de margem consignável, troca/inadimplência do ente pagador e mudanças "
            "regulatórias — riscos discretos e específicos por convênio, não capturados pela PD contínua."
        ),
        lgd_estocastica=False,
        lgd_vol_sugerido=0.0,
        pd_lgd_corr_sugerido=0.0,
        horizonte_1ano_ok=True,
        observacoes=(
            "Correlação de ativos na base da faixa de varejo (~0.03–0.05) pela baixa "
            "ciclicidade. Modele o risco de convênio de forma discreta (por ente pagador), "
            "não só pela PD média."
        ),
    ),
    "veiculos": ProductPreset(
        nome="veiculos",
        descricao=(
            "Financiamento de veículos (CDC com alienação fiduciária). Garantia real "
            "reduz a severidade em condições normais, mas o colateral é procíclico: em "
            "recessão o preço do usado cai junto com o aumento da inadimplência."
        ),
        basel_class="other_retail",
        rho_sugerido=0.075,
        parametro_critico="LGD downturn (valor do colateral em estresse)",
        nivel_pd="média",
        nivel_lgd="média (garantia real, mas procíclica)",
        correlacao_ciclo="média (forte via preço de usados)",
        segmentacao_recomendada=[
            "score",
            "LTV (loan-to-value)",
            "idade do bem",
        ],
        risco_especifico=(
            "correlação adversa PD–LGD: recessão eleva simultaneamente a frequência de "
            "default e a severidade (queda do preço do usado), penalizando a cauda."
        ),
        lgd_estocastica=True,
        lgd_vol_sugerido=0.20,
        pd_lgd_corr_sugerido=0.30,
        horizonte_1ano_ok=True,
        observacoes=(
            "Use LGD estocástica correlacionada ao fator sistêmico (pd_lgd_corr > 0) para "
            "capturar a cauda; a LGD downturn é o parâmetro mais crítico. rho intermediário "
            "(~0.06–0.09)."
        ),
    ),
    "credito_pessoal": ProductPreset(
        nome="credito_pessoal",
        descricao=(
            "Crédito pessoal sem garantia (empréstimo direto ao consumidor). "
            "Comporta-se como o cartão — sem colateral, PD alta e ciclo-sensível — "
            "porém tipicamente parcelado, com EAD mais previsível que a do rotativo."
        ),
        basel_class="other_retail",
        rho_sugerido=0.09,
        parametro_critico="PD (sem garantia; alta e ciclo-sensível)",
        nivel_pd="alta e ciclo-sensível",
        nivel_lgd="alta (sem garantia)",
        correlacao_ciclo="alta",
        segmentacao_recomendada=[
            "score",
            "prazo",
            "faixa de renda/comprometimento",
        ],
        risco_especifico=(
            "ausência de garantia com forte sensibilidade ao ciclo, similar ao cartão, mas "
            "sem o componente de saque de limite (produto parcelado, EAD mais estável)."
        ),
        lgd_estocastica=False,
        lgd_vol_sugerido=0.0,
        pd_lgd_corr_sugerido=0.0,
        horizonte_1ano_ok=True,
        observacoes=(
            "Trate como cartão sem o problema de CCF: EAD ~ saldo devedor. rho alto "
            "(~0.08–0.10) pela ciclicidade da PD."
        ),
    ),
    "imobiliario": ProductPreset(
        nome="imobiliario",
        descricao=(
            "Financiamento imobiliário residencial. Garantia real forte (o imóvel) mantém "
            "a LGD baixa, à semelhança de veículos, mas o **prazo longo** rompe a hipótese "
            "de horizonte anual: o risco relevante se acumula em vários anos."
        ),
        basel_class="mortgage",
        rho_sugerido=0.15,
        parametro_critico="LGD downturn e horizonte plurianual (prazo longo)",
        nivel_pd="baixa",
        nivel_lgd="baixa (garantia forte, mas procíclica)",
        correlacao_ciclo="média a alta (via preço de imóveis)",
        segmentacao_recomendada=[
            "score",
            "LTV (loan-to-value)",
            "prazo/maturidade",
        ],
        risco_especifico=(
            "horizonte > 1 ano: o modelo anual subestima o risco; e correlação adversa "
            "PD–LGD via preço de imóveis, com concentração geográfica."
        ),
        lgd_estocastica=True,
        lgd_vol_sugerido=0.15,
        pd_lgd_corr_sugerido=0.30,
        horizonte_1ano_ok=False,
        observacoes=(
            "Único produto com horizonte_1ano_ok=False: ajuste o horizonte (ou aplique fator "
            "de maturidade) antes de comparar com os demais. Correlação de ativos alta "
            "(~0.15, à la mortgage de Basileia)."
        ),
    ),
    "cheque_especial": ProductPreset(
        nome="cheque_especial",
        descricao=(
            "Cheque especial (limite de crédito em conta). Rotativo sem garantia como o "
            "cartão, porém com CCF ainda **mais instável**: o uso do limite oscila com o "
            "fluxo de caixa do cliente e dispara na deterioração."
        ),
        basel_class="revolving",
        rho_sugerido=0.11,
        parametro_critico="EAD/CCF (ainda mais instável que o cartão)",
        nivel_pd="alta e ciclo-sensível",
        nivel_lgd="alta (sem garantia)",
        correlacao_ciclo="alta",
        segmentacao_recomendada=[
            "score",
            "padrão de uso do limite",
            "faixa de limite",
        ],
        risco_especifico=(
            "CCF altamente instável: o saldo utilizado sobe abruptamente pré-default, "
            "com EAD downturn ainda menos previsível que a do cartão."
        ),
        lgd_estocastica=False,
        lgd_vol_sugerido=0.0,
        pd_lgd_corr_sugerido=0.0,
        horizonte_1ano_ok=True,
        observacoes=(
            "Rotativo com o CCF mais volátil do portfólio de varejo; estresse a EAD "
            "agressivamente. rho no topo da faixa rotativa (~0.10–0.12)."
        ),
    ),
}


# ======================================================================
# API de consulta
# ======================================================================
def preset(nome: str) -> ProductPreset:
    """Retorna o :class:`ProductPreset` do produto ``nome``.

    Parameters
    ----------
    nome:
        Chave do produto (ex.: ``"cartao"``, ``"consignado"``, ``"veiculos"``).

    Raises
    ------
    ValueError
        Se ``nome`` não for um produto conhecido — a mensagem lista os
        disponíveis.
    """
    try:
        return PRESETS[nome]
    except KeyError:
        disponiveis = ", ".join(list_presets())
        raise ValueError(
            f"Produto {nome!r} não tem preset. Disponíveis: {disponiveis}."
        ) from None


def list_presets() -> List[str]:
    """Lista as chaves de produto com preset disponível (na ordem canônica)."""
    return list(PRESETS.keys())


def presets_frame() -> pd.DataFrame:
    """A **Tabela 3** reproduzida: uma linha por produto, para documentação.

    Returns
    -------
    pandas.DataFrame
        Uma linha por produto (na ordem canônica), colunas com todos os campos
        do preset. Útil para exibição em relatórios/notebooks.
    """
    linhas = [PRESETS[nome].to_dict() for nome in list_presets()]
    return pd.DataFrame(linhas)


# ======================================================================
# Ponte com o contrato de dados (Segment)
# ======================================================================
def apply_preset(product: str, **overrides: Any) -> Dict[str, Any]:
    """Devolve ``kwargs`` prontos para construir um
    :class:`~yggdrasil.credit_risk.capital.portfolio.Segment` a partir do preset.

    Devolve **apenas** chaves aceitas por :class:`Segment` (``rho``, ``product``,
    ``factor`` = ``product``, ``lgd_vol``), de modo que
    ``Segment(name=..., pd=..., lgd=..., ead=..., **apply_preset(product))``
    funciona diretamente. As recomendações para a **simulação** (usar LGD
    estocástica e a correlação PD–LGD) ficam nos atributos do próprio preset —
    ``preset(product).lgd_estocastica`` e ``.pd_lgd_corr_sugerido`` —, fora deste
    dicionário, para não colidir com o construtor do segmento.

    Qualquer chave passada em ``overrides`` sobrescreve o valor do preset — por
    exemplo, ``apply_preset("veiculos", rho=0.06, lgd_vol=0.25)``.

    Parameters
    ----------
    product:
        Chave do produto (validada contra :func:`list_presets`).
    **overrides:
        Substituições pontuais (ex.: ``rho``, ``factor``, ``lgd_vol``).

    Returns
    -------
    dict
        ``kwargs`` diretamente aceitos por :class:`Segment` (``rho``, ``product``,
        ``factor``, ``lgd_vol``).

    Examples
    --------
    >>> from yggdrasil.credit_risk.capital.portfolio import Segment
    >>> seg = Segment(name="cartao_score_baixo", pd=0.12, lgd=0.75, ead=1_000_000,
    ...               **apply_preset("cartao"))
    """
    p = preset(product)
    kwargs: Dict[str, Any] = {
        "rho": p.rho_sugerido,
        "product": p.nome,
        "factor": p.nome,
        "lgd_vol": p.lgd_vol_sugerido,
    }
    kwargs.update(overrides)
    return kwargs


__all__ = [
    "ProductPreset",
    "PRESETS",
    "preset",
    "list_presets",
    "presets_frame",
    "apply_preset",
]
