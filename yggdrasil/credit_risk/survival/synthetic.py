"""
Painel de referência com processo gerador conhecido
===================================================
A única forma de saber se um motor de sobrevivência acerta é dar a ele um painel
cuja verdade se conhece. :func:`make_reference_panel` gera uma carteira de
crédito sintética com os três fenômenos que a estrutura a termo precisa
recuperar:

* **maturação**: o *hazard* cresce com a idade do contrato (forma de Weibull com
  ``k > 1``), o que uma curva de *hazard* constante não captura;
* **heterogeneidade**: dois produtos com níveis de risco distintos, um *score*
  contínuo e um LTV que deslocam o *hazard* de cada contrato (o insumo da
  regressão de *hazard* com covariáveis);
* **censura à direita**: fim de janela, quitação antecipada e prazo do contrato
  tiram contratos do risco sem *default*. A censura é informativa por desenho
  (quem quita mais cedo tende a ser menos arriscado), como na carteira real.

O objeto devolvido (:class:`ReferencePanel`) traz o DataFrame no formato longo
que o :class:`~yggdrasil.credit_risk.ecl.panel.ContractPanel` consome, a função
de *hazard* verdadeira (para comparar com o estimado) e um atalho para montar o
painel com os nomes de coluna certos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

#: Colunas de covariáveis do painel de referência.
FEATURES = ("feat_score", "feat_ltv", "feat_atraso_prev")


@dataclass
class ReferencePanel:
    """O painel sintético e a verdade que o gerou.

    Attributes
    ----------
    df:
        Painel longo (uma linha por contrato x mês) com ``id_contrato``,
        ``dt_ref``, ``safra_origem``, ``default``, ``produto``, ``rating``,
        ``prazo`` (remanescente), ``exposicao`` e as covariáveis de :data:`FEATURES`.
    truth:
        Parâmetros do processo gerador: ``base`` por produto, ``shape`` (o ``k``
        de Weibull), ``beta`` por covariável e a taxa mensal de quitação.
    """

    df: pd.DataFrame
    truth: Dict[str, object] = field(default_factory=dict)
    features: List[str] = field(default_factory=lambda: list(FEATURES))

    def true_hazard(self, ages, produto: str = "cartao", score: float = 0.0,
                    ltv: float = 0.5, atraso: int = 0) -> np.ndarray:
        """*Hazard* verdadeiro por idade para um perfil de contrato."""
        t = np.asarray(ages, dtype=float)
        base = float(self.truth["base"][produto])
        k = float(self.truth["shape"])
        beta = self.truth["beta"]
        eta = (beta["feat_score"] * score + beta["feat_ltv"] * (ltv - 0.5) * 2.0
               + beta["feat_atraso_prev"] * atraso)
        return np.clip(base * np.exp(eta) * ((t + 1.0) / 12.0) ** (k - 1.0), 0.0, 0.95)

    def panel(self, **kwargs):
        """Monta o :class:`~yggdrasil.credit_risk.ecl.panel.ContractPanel` com os
        nomes de coluna deste painel (os ``kwargs`` sobrescrevem)."""
        from ..ecl.panel import ContractPanel

        base = dict(id_col="id_contrato", date_col="dt_ref", default_col="default",
                    origin_col="safra_origem", segment_col="produto", term_col="prazo",
                    exposure_col="exposicao")
        base.update(kwargs)
        return ContractPanel(self.df, **base)

    def __repr__(self) -> str:
        return (f"ReferencePanel(n_obs={len(self.df)}, "
                f"n_contratos={self.df['id_contrato'].nunique()}, "
                f"n_defaults={int(self.df['default'].sum())})")


def make_reference_panel(n_contracts: int = 1500, months: int = 36, seed: int = 7,
                         start: str = "2020-01-01", n_origin_months: int = 18,
                         prepay_rate: float = 0.012, shape: float = 1.35,
                         base: Optional[Dict[str, float]] = None) -> ReferencePanel:
    """Gera o painel de referência.

    Parameters
    ----------
    n_contracts:
        Nº de contratos originados.
    months:
        Nº máximo de meses observados por contrato (a janela de observação).
    seed:
        Semente do gerador.
    start, n_origin_months:
        As safras de originação vão de ``start`` até ``start + n_origin_months``;
        a janela de observação fecha em ``start + n_origin_months + months``, o
        que censura as safras recentes nas idades altas (o fenômeno que a base em
        risco recontada trata).
    prepay_rate:
        Taxa mensal de quitação antecipada; contratos de menor risco quitam mais.
    shape:
        O ``k`` da maturação (``> 1`` = *hazard* crescente com a idade).
    base:
        *Hazard* base mensal por produto (padrão: cartão 1,4%, consignado 0,4%).
    """
    rng = np.random.default_rng(seed)
    base = dict(base or {"cartao": 0.014, "consignado": 0.004})
    beta = {"feat_score": 0.55, "feat_ltv": 0.35, "feat_atraso_prev": 0.25}
    produtos = list(base)
    inicio = pd.Timestamp(start)
    fim_janela = inicio + pd.DateOffset(months=n_origin_months + months)

    linhas = []
    for i in range(int(n_contracts)):
        produto = produtos[int(rng.integers(0, len(produtos)))]
        score = float(rng.normal(0.0, 1.0))
        # rating ordenado pelo score (a régua transversal que já existe na casa)
        rating = "A" if score < -0.45 else ("B" if score < 0.45 else "C")
        ltv = float(np.clip(rng.beta(4.0, 4.0), 0.0, 1.0))
        atraso = int(rng.poisson(0.35 + 0.35 * max(score, 0.0)))
        origem = inicio + pd.DateOffset(months=int(rng.integers(0, n_origin_months)))
        prazo = int(rng.integers(12, 61))
        exposicao = float(rng.lognormal(8.8 if produto == "cartao" else 9.6, 0.45))
        eta = (beta["feat_score"] * score + beta["feat_ltv"] * (ltv - 0.5) * 2.0
               + beta["feat_atraso_prev"] * atraso)
        # quem tem menos risco quita mais: censura informativa, como na carteira real
        p_quita = prepay_rate * float(np.exp(-0.4 * score))
        for t in range(int(months)):
            data = origem + pd.DateOffset(months=t)
            if data >= fim_janela or t >= prazo:
                break
            h = min(base[produto] * np.exp(eta) * ((t + 1.0) / 12.0) ** (shape - 1.0), 0.95)
            quebrou = int(rng.uniform() < h)
            linhas.append((f"C{i:05d}", data, origem, quebrou, produto, rating,
                           max(prazo - t, 0), exposicao, score, ltv, atraso))
            if quebrou or rng.uniform() < p_quita:
                break
            exposicao = float(max(exposicao * (1.0 - 1.0 / max(prazo, 1)), 0.0))

    df = pd.DataFrame(linhas, columns=[
        "id_contrato", "dt_ref", "safra_origem", "default", "produto", "rating",
        "prazo", "exposicao", "feat_score", "feat_ltv", "feat_atraso_prev",
    ])
    truth = {"base": base, "shape": float(shape), "beta": beta,
             "prepay_rate": float(prepay_rate), "seed": int(seed),
             "months": int(months), "n_origin_months": int(n_origin_months)}
    return ReferencePanel(df=df, truth=truth, features=list(FEATURES))


__all__ = ["ReferencePanel", "make_reference_panel", "FEATURES"]
