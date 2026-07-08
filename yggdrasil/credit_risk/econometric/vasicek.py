"""
Modelo de fator ``Z`` de Vasicek (Guia §3.4, Tabela 1) — a ponte com o capital
=============================================================================
A abordagem que **nasce do próprio arcabouço de crédito**: em vez de modelar a
taxa transformada por um *link* genérico, inverte-se a fórmula de Vasicek para
extrair da série de taxas de *default* o **fator sistêmico latente ``Z``** — que é
aproximadamente ``N(0, 1)`` e estacionário por construção — e modela-se ``Z``
contra as variáveis macro por ARDL. As projeções de ``Z`` são reconvertidas em PD
pela fórmula direta.

Vantagens (Tabela 1): **consistência total com o modelo de capital** (o mesmo
``Z`` alimenta a simulação multifatorial de
:mod:`yggdrasil.credit_risk.capital`), tratamento natural do piso/teto da taxa e
separação limpa entre o **nível TTC** e o **ciclo**. Cuidado: depende de
``PD_TTC`` e ``ρ`` **bem estimados** — os mesmos insumos do capital.

Implementação: é um :class:`~yggdrasil.credit_risk.econometric.ardl.ARDL` cujo
*link* é o par (:func:`~...transforms.vasicek_z`,
:func:`~...transforms.default_rate_from_z`) parametrizado por ``PD_TTC`` e ``ρ``.
Toda a maquinaria de estimação, diagnóstico e projeção multi-passo é herdada.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .ardl import ARDL
from .base import Specification
from .series import RiskSeries, as_risk_series
from .transforms import Link, default_rate_from_z, vasicek_z


def vasicek_link(pd_ttc: float, rho: float) -> Link:
    """Constrói a :class:`~...transforms.Link` de Vasicek para ``(PD_TTC, ρ)``.

    ``forward`` = inversão (DR → ``Z``); ``inverse`` = reconversão (``Z`` → DR).
    """
    return Link(
        name="vasicek",
        forward=lambda s: vasicek_z(s, pd_ttc, rho),
        inverse=lambda z: default_rate_from_z(z, pd_ttc, rho),
    )


class VasicekZ(ARDL):
    """ARDL sobre o fator sistêmico ``Z`` de Vasicek (modelo satélite de PD).

    Parameters
    ----------
    series:
        A :class:`RiskSeries` de **taxa de *default*** (``kind='pd'``).
    macro, spec:
        Como no :class:`ARDL`. O ``spec.link`` é ignorado (o *link* é sempre o de
        Vasicek); o restante (defasagens, AR, sazonal, eventos) vale igual. Se
        ``spec`` é ``None``, usa AR(1).
    rho:
        Correlação de ativos ``ρ`` do segmento, em ``(0, 1)`` — **o mesmo ``ρ`` do
        capital econômico**. Obrigatório (não é estimado aqui).
    pd_ttc:
        PD *through-the-cycle*. Se ``None``, usa a média de longo prazo da série
        (:meth:`RiskSeries.ttc`), a proxy usual de TTC.
    """

    name = "VasicekZ"

    def __init__(
        self,
        series,
        macro=None,
        spec: Optional[Specification] = None,
        *,
        rho: float,
        pd_ttc: Optional[float] = None,
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
    ) -> None:
        rs: RiskSeries = as_risk_series(series, kind="pd")
        if rs.kind != "pd":
            raise ValueError("VasicekZ é específico de PD (taxa de default); use ARDL/beta para LGD/CCF.")
        if not (0.0 < rho < 1.0):
            raise ValueError(f"rho deve estar em (0, 1); recebido {rho!r}.")
        pd_ttc = float(pd_ttc) if pd_ttc is not None else rs.ttc()
        if not (0.0 < pd_ttc < 1.0):
            raise ValueError(f"pd_ttc deve estar em (0, 1); recebido {pd_ttc!r}.")

        # Cópia do spec (não muta o objeto do chamador — ele pode reusá-lo em
        # outro modelo do champion-challenger). 'identity' satisfaz get_link no
        # __init__ do ARDL; o link real de Vasicek é injetado logo abaixo.
        if spec is None:
            spec = Specification(ar=1, link="identity")
        else:
            spec = replace(spec, link="identity")
        super().__init__(rs, macro, spec, kind="pd", cov_type=cov_type, hac_maxlags=hac_maxlags)

        # injeta o link de Vasicek (sobrepõe o identity herdado do ARDL)
        self.pd_ttc = pd_ttc
        self.rho = rho
        self._link = vasicek_link(pd_ttc, rho)
        self.link = "vasicek"
        self.spec.link = "vasicek"  # self.spec é a cópia; seguro p/ describe()/relatório

    def z_series(self):
        """A série extraída do fator sistêmico ``Z`` (a variável efetivamente modelada)."""
        return vasicek_z(self.series.values, self.pd_ttc, self.rho)


__all__ = ["VasicekZ", "vasicek_link"]
