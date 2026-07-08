"""
Painel de segmentos (Guia §3.7, Tabela 1) — poder estatístico e sinais disciplinados
===================================================================================
Quando há **muitos segmentos com séries curtas**, empilhá-los em painel
(segmentos × tempo) com **efeitos fixos de segmento** e **coeficientes macro
comuns** aumenta o poder estatístico e disciplina os sinais: "é mais difícil um
coeficiente de desemprego trocar de sinal quando estimado com 20 segmentos
juntos" (§3.7). É frequentemente a melhor escolha para varejo com segmentação
fina.

Implementação: estimador de **variáveis dummy de menor quadrado** (LSDV) — um
intercepto por segmento (o efeito fixo) e **inclinações macro/AR comuns** —, com
**erros-padrão agrupados por segmento** (*cluster-robust*), o padrão para painéis
onde os resíduos de um segmento são correlacionados no tempo. Em NumPy puro (não
exige ``linearmodels``), consistente com o resto do pacote.

Cuidado do guia: em painéis **dinâmicos** (com defasagem da dependente e ``T``
curto) há o **viés de Nickell**; o intercepto por segmento não o remove
totalmente — trate o termo AR com cautela em ``T`` pequeno.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from . import _engine
from .base import Specification
from .series import RiskSeries, as_risk_series
from .transforms import align, get_link, inv_logit


@dataclass
class PanelResult:
    """Resultado do painel: efeitos fixos por segmento + inclinações comuns."""

    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    segment_names: list
    macro_terms: list
    nobs: int
    n_segments: int
    sigma: float
    resid: np.ndarray

    def coef_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"coef": self.params, "std_err": self.bse,
                             "t": self.tvalues, "p_valor": self.pvalues})

    def macro_frame(self) -> pd.DataFrame:
        """Só as inclinações **comuns** (macro/AR) — o que interessa ao ciclo."""
        return self.coef_frame().loc[self.macro_terms]

    def segment_intercepts(self) -> pd.Series:
        """O efeito fixo (intercepto) de cada segmento — o nível TTC relativo."""
        return pd.Series({s: self.params[f"fe::{s}"] for s in self.segment_names})

    def __repr__(self) -> str:  # pragma: no cover
        return (f"PanelResult(segmentos={self.n_segments}, obs={self.nobs}, "
                f"termos_comuns={self.macro_terms})")


class PanelSatellite:
    """Painel de segmentos com efeitos fixos e inclinações macro comuns (§3.7).

    Parameters
    ----------
    panels:
        ``{nome_do_segmento: RiskSeries}`` — uma série por segmento (mesmo ``kind``).
    macro:
        Um :class:`~pandas.DataFrame` comum a todos os segmentos, **ou** um dict
        ``{segmento: DataFrame}`` com a macro específica de cada um.
    spec:
        A :class:`Specification` **comum** (defasagens macro, ordem AR, *link*). O
        intercepto vem dos efeitos fixos (a constante do ``spec`` é ignorada).
    """

    name = "PanelSatellite"

    def __init__(self, panels: Mapping[str, object], macro: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],
                 spec: Optional[Specification] = None) -> None:
        if len(panels) < 2:
            raise ValueError("painel exige ao menos 2 segmentos.")
        self.panels = {k: as_risk_series(v, kind="pd") for k, v in panels.items()}
        self.macro = macro
        self.spec = spec or Specification(ar=1, link="logit")
        self.link = self.spec.link
        self._link = get_link(self.link)
        self.result: Optional[PanelResult] = None
        self._fits: dict = {}

    def _macro_for(self, seg: str) -> Optional[pd.DataFrame]:
        if isinstance(self.macro, Mapping):
            return self.macro[seg]
        return self.macro

    def fit(self) -> PanelResult:
        seg_spec = replace(self.spec, trend="n")  # sem const global: os FE são os interceptos
        Xblocks, yblocks, seg_of_row, macro_terms = [], [], [], None
        seg_names = list(self.panels.keys())
        for name in seg_names:
            rs = self.panels[name]
            ylink = self._link.forward(rs.values)
            ylink.name = "y"
            X = _engine.build_design(ylink, self._macro_for(name), seg_spec)
            y, Xa = align(ylink, X)
            if macro_terms is None:
                macro_terms = list(Xa.columns)
            Xblocks.append(Xa[macro_terms].to_numpy(dtype=float))
            yblocks.append(y.to_numpy(dtype=float))
            seg_of_row.append(np.full(len(y), name, dtype=object))

        Xc = np.vstack(Xblocks)                     # inclinações comuns
        yv = np.concatenate(yblocks)
        seg_row = np.concatenate(seg_of_row)
        # dummies de efeito fixo (uma por segmento, sem drop → sem const global)
        D = np.column_stack([(seg_row == s).astype(float) for s in seg_names])
        Xfull = np.column_stack([D, Xc])
        names = [f"fe::{s}" for s in seg_names] + list(macro_terms)

        XtX = Xfull.T @ Xfull
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ (Xfull.T @ yv)
        resid = yv - Xfull @ beta
        nobs, k = Xfull.shape

        # erros-padrão AGRUPADOS por segmento (cluster-robust)
        meat = np.zeros((k, k))
        for s in seg_names:
            m = seg_row == s
            Xg = Xfull[m]
            ug = resid[m]
            sg = Xg.T @ ug
            meat += np.outer(sg, sg)
        G = len(seg_names)
        dof = (G / (G - 1)) * ((nobs - 1) / (nobs - k)) if G > 1 and nobs > k else 1.0
        V = dof * (XtX_inv @ meat @ XtX_inv)
        bse = np.sqrt(np.clip(np.diag(V), 0, np.inf))
        tvals = beta / np.where(bse > 0, bse, np.nan)
        pvals = 2.0 * stats.t.sf(np.abs(tvals), df=max(1, G - 1))
        sigma = float(np.std(resid, ddof=1))

        idx = pd.Index(names)
        self.result = PanelResult(
            params=pd.Series(beta, index=idx), bse=pd.Series(bse, index=idx),
            tvalues=pd.Series(tvals, index=idx), pvalues=pd.Series(pvals, index=idx),
            segment_names=seg_names, macro_terms=list(macro_terms), nobs=nobs,
            n_segments=G, sigma=sigma, resid=resid,
        )
        return self.result

    def predict(self, segment: str, macro_future: pd.DataFrame) -> pd.Series:
        """Projeção de um segmento: inclinações **comuns** + o efeito fixo do
        segmento + a dinâmica AR da própria série (Guia §3.7)."""
        if self.result is None:
            raise RuntimeError("PanelSatellite: chame .fit() primeiro.")
        if segment not in self.panels:
            raise KeyError(segment)
        rs = self.panels[segment]
        # params na forma que o motor espera: o FE do segmento vira 'const'
        pred_spec = replace(self.spec, trend="c")
        params = {"const": float(self.result.params[f"fe::{segment}"])}
        for t in self.result.macro_terms:
            params[t] = float(self.result.params[t])
        params = pd.Series(params)
        hist_link = self._link.forward(rs.values)
        macro_full = _engine.concat_macro(self._macro_for(segment), macro_future)
        df = _engine.forecast_paths(
            params, pred_spec, self._link.inverse, hist_link, macro_full,
            macro_future.index, resid_pool=self.result.resid,
            trend_offset=len(hist_link), n_sims=0)
        out = df["mean"].copy()
        out.name = f"{rs.kind}_{segment}_previsto"
        return out


__all__ = ["PanelResult", "PanelSatellite"]
