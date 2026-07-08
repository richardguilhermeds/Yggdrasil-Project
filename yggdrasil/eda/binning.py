"""Binning de FEATURE (numérica e categórica) com tabela de WoE/IV.

Espelha a ideia das estratégias de `yggdrasil.ratings` (fit no DES, transform em
qualquer base), mas aplicada à feature em vez do score. Missing vira um bin
próprio; níveis categóricos raros são agrupados em 'OUTROS'.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from .config import EDAConfig
from .dtypes import as_numeric, has_target, infer_feature_kind

_EPS = 1e-6
MISSING = "MISSING"
OUTROS = "OUTROS"


def _numeric_edges(method, xnum, target, problem_type, n_bins):
    """Cortes para feature numérica conforme o método (com fallbacks)."""
    x = xnum.dropna()
    if x.nunique() < 2:
        return np.array([-np.inf, np.inf])

    if method in ("tree", "optbinning") and target is not None:
        mask = xnum.notna() & target.notna()
        xt = as_numeric(xnum[mask]).values
        yt = np.asarray(target[mask].values, dtype=float)
        if len(np.unique(xt)) >= 2 and len(np.unique(yt)) >= 2:
            try:
                if method == "tree":
                    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                    Tree = DecisionTreeClassifier if problem_type == "classification" else DecisionTreeRegressor
                    tree = Tree(max_leaf_nodes=n_bins,
                                min_samples_leaf=max(int(0.05 * len(xt)), 50),
                                random_state=42).fit(xt.reshape(-1, 1), yt)
                    thr = sorted(t for t in tree.tree_.threshold if t != -2)
                    edges = np.unique(np.concatenate([[-np.inf], thr, [np.inf]]))
                else:
                    if problem_type == "classification":
                        from optbinning import OptimalBinning
                        ob = OptimalBinning(dtype="numerical", max_n_bins=n_bins)
                        ob.fit(xt, yt.astype(int))
                    else:
                        from optbinning import ContinuousOptimalBinning
                        ob = ContinuousOptimalBinning(dtype="numerical", max_n_bins=n_bins)
                        ob.fit(xt, yt)
                    edges = np.unique(np.concatenate([[-np.inf], np.asarray(ob.splits, float), [np.inf]]))
                if len(edges) >= 2:
                    return edges
            except Exception:
                pass  # cai para quantil

    # quantil (default e fallback)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x, qs)).astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges if len(edges) >= 2 else np.array([-np.inf, np.inf])


class FeatureBinner:
    """Aprende bins de uma feature (no DES) e os aplica em qualquer amostra."""

    def __init__(self, method: str = "quantile", n_bins: int = 10, rare_level_pct: float = 0.01):
        self.method = method
        self.n_bins = n_bins
        self.rare_level_pct = rare_level_pct
        self.kind_: Optional[str] = None
        self.edges_: Optional[np.ndarray] = None
        self.faixas_: List[str] = []
        self.keep_levels_: List[str] = []

    def fit(self, df: pd.DataFrame, col: str, cfg: ColumnConfig, problem_type: Optional[str] = None):
        self.kind_ = infer_feature_kind(df[col])
        base = df[df[cfg.sample_col] == cfg.dev_sample]
        if len(base) == 0:
            base = df
        if self.kind_ == "categorical":
            freq = base[col].astype(object).value_counts(normalize=True)
            self.keep_levels_ = [str(l) for l in freq[freq >= self.rare_level_pct].index]
        else:
            xnum = as_numeric(base[col])
            target = df[cfg.target_col] if has_target(df, cfg) else None
            tbase = base[cfg.target_col] if (target is not None) else None
            self.edges_ = _numeric_edges(self.method, xnum, tbase, problem_type, self.n_bins)
            self.faixas_ = [f"({self.edges_[i]:.4g}, {self.edges_[i + 1]:.4g}]"
                            for i in range(len(self.edges_) - 1)]
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        miss = series.isna().values
        if self.kind_ == "categorical":
            s = series.astype(object)
            codes = np.where(pd.Series(s).isin(self.keep_levels_), s.astype(str), OUTROS).astype(object)
            codes[miss] = MISSING
            cats = list(self.keep_levels_)
            if (codes == OUTROS).any():
                cats.append(OUTROS)
        else:
            xnum = as_numeric(series)
            idx = np.searchsorted(self.edges_, xnum.values, side="right") - 1
            idx = np.clip(idx, 0, len(self.edges_) - 2)
            codes = np.array([f"{i + 1:02d}" for i in idx], dtype=object)
            codes[miss | xnum.isna().values] = MISSING
            cats = [f"{i + 1:02d}" for i in range(len(self.edges_) - 1)]
        if (codes == MISSING).any():
            cats = cats + [MISSING]
        return pd.Series(pd.Categorical(codes, categories=cats, ordered=True),
                         index=series.index, name=f"bin_{series.name}")

    def fit_transform(self, df, col, cfg, problem_type=None) -> pd.Series:
        return self.fit(df, col, cfg, problem_type).transform(df[col])


def bin_feature(df, col, cfg, eda_cfg=None, problem_type=None, method=None):
    """Atalho: cria e ajusta um FeatureBinner, retornando (bins, binner)."""
    eda_cfg = eda_cfg or EDAConfig()
    binner = FeatureBinner(method or eda_cfg.binning_method, eda_cfg.n_bins, eda_cfg.rare_level_pct)
    bins = binner.fit_transform(df, col, cfg, problem_type)
    return bins, binner


def binning_table(
    df: pd.DataFrame, col: str, cfg: ColumnConfig, eda_cfg: Optional[EDAConfig] = None,
    problem_type: Optional[str] = None, binner: Optional[FeatureBinner] = None,
) -> pd.DataFrame:
    """Tabela por bin: volume, %, event_rate/target_medio e (binário) WoE/IV.

    O IV total fica em ``tabela.attrs['iv']``.
    """
    eda_cfg = eda_cfg or EDAConfig()
    if binner is None:
        binner = FeatureBinner(eda_cfg.binning_method, eda_cfg.n_bins, eda_cfg.rare_level_pct)
        binner.fit(df, col, cfg, problem_type)
    bins = binner.transform(df[col])

    tab = pd.DataFrame({"bin": bins.values})
    tem_target = has_target(df, cfg)
    if tem_target:
        tab["target"] = df[cfg.target_col].values

    g = tab.groupby("bin", observed=True)
    out = pd.DataFrame({"n": g.size()})
    out["pct"] = (out["n"] / out["n"].sum()).round(4)

    if tem_target and problem_type == "classification":
        out["event_rate"] = g["target"].mean().round(6)
        # Contagens vetorizadas (o apply por bin era uma passada Python por grupo).
        maus = (tab["target"] == 1).groupby(tab["bin"], observed=True).sum().astype(float)
        bons = (tab["target"] == 0).groupby(tab["bin"], observed=True).sum().astype(float)
        dist_mau = maus / max(maus.sum(), _EPS)
        dist_bom = bons / max(bons.sum(), _EPS)
        woe = np.log((dist_bom + _EPS) / (dist_mau + _EPS))
        out["woe"] = woe.round(6)
        out["iv_parcial"] = ((dist_bom - dist_mau) * woe).round(6)
    elif tem_target:
        out["target_medio"] = g["target"].mean().round(6)

    # ordena pelos bins crescentes (ordem da feature) e injeta a faixa numérica
    ordem = [c for c in bins.cat.categories if c in out.index]
    out = out.reindex(ordem)
    if binner.kind_ != "categorical" and binner.faixas_:
        mapa = {f"{i + 1:02d}": fx for i, fx in enumerate(binner.faixas_)}
        mapa[MISSING] = MISSING
        out.insert(0, "faixa", [mapa.get(b, b) for b in out.index])
    out = out.reset_index().rename(columns={"index": "bin"})
    if "iv_parcial" in out.columns:
        out.attrs["iv"] = round(float(out["iv_parcial"].sum()), 6)
    return out


def group_rare_levels(series: pd.Series, rare_pct: float = 0.01) -> pd.Series:
    """Agrupa níveis categóricos com frequência < rare_pct em 'OUTROS'."""
    freq = series.astype(object).value_counts(normalize=True)
    keep = set(freq[freq >= rare_pct].index)
    return series.astype(object).where(series.isin(keep), other=OUTROS)


__all__ = ["FeatureBinner", "bin_feature", "binning_table", "group_rare_levels", "MISSING", "OUTROS"]
