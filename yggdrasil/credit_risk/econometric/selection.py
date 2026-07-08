"""
Seleção, validação e ranking champion-challenger (Guia §4, §6.2 ``selection``)
=============================================================================
Em séries curtas, o ajuste in-sample engana: ``R²`` alto é barato e não prova
nada (§4). O processo de seleção pune complexidade e premia **desempenho fora da
amostra** e **coerência econômica**. Este módulo entrega:

Grade de especificações — :func:`make_grid`
    Percorre combinações **parcimoniosas** (2 a 4 variáveis, defasagens de 0 a 6),
    o espaço que o guia recomenda varrer (§4.1).

Filtros duros — :func:`sign_ok`, VIF
    **Sinal econômico coerente** (coeficiente com sinal trocado é *desqualificador*
    — a projeção de estresse iria na direção errada, §4.1) e **VIF controlado**
    (colinearidade macro).

Validação fora da amostra — :func:`walk_forward`, :func:`diebold_mariano`
    O padrão-ouro em séries temporais (§4.3): estima até ``t``, projeta ``h``
    passos, avança a janela; compara RMSE/MAE/MAPE na escala original contra os
    **benchmarks ingênuos** e o ARIMA, com o teste de **Diebold-Mariano** para a
    significância da diferença de acurácia.

Ranking — :func:`search`, :func:`compare`
    Reúne tudo num ranking champion-challenger reprodutível por configuração.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import diagnostics as _diag
from .ardl import ARDL
from .arima import ARIMA
from .base import SatelliteModel, Specification
from .benchmarks import HistoricalMean, RandomWalk, SeasonalNaive
from .series import RiskSeries, as_risk_series


# ======================================================================
# Grade de especificações (Guia §4.1)
# ======================================================================
def make_grid(
    candidates: Sequence[str],
    *,
    lag_set: Sequence[int] = (0, 1, 3, 6),
    min_vars: int = 1,
    max_vars: int = 3,
    ar_orders: Sequence[int] = (1,),
    link: str = "logit",
    expected_signs: Optional[Mapping[str, int]] = None,
    seasonal: bool = False,
    seasonal_period: int = 12,
    max_specs: Optional[int] = 400,
) -> list[Specification]:
    """Gera a grade de :class:`Specification` parcimoniosas (Guia §4.1).

    Para cada tamanho de ``min_vars..max_vars``, cada combinação de variáveis, cada
    atribuição de **uma** defasagem por variável (de ``lag_set``) e cada ordem AR,
    cria uma especificação. ``max_specs`` limita o total (trunca com aviso via
    ``attrs['truncated']``).
    """
    signs = dict(expected_signs or {})
    specs: list[Specification] = []
    for k in range(max(1, min_vars), max_vars + 1):
        for combo in itertools.combinations(candidates, k):
            for lags in itertools.product(lag_set, repeat=k):
                exog = {var: [lag] for var, lag in zip(combo, lags)}
                for ar in ar_orders:
                    specs.append(Specification(
                        exog=exog, ar=int(ar), link=link, seasonal=seasonal,
                        seasonal_period=seasonal_period,
                        expected_signs={v: signs[v] for v in combo if v in signs},
                    ))
    truncated = False
    if max_specs is not None and len(specs) > max_specs:
        specs = specs[:max_specs]
        truncated = True
    if truncated:
        # o guia pede que limites de cobertura sejam explícitos (não truncar em silêncio)
        import warnings
        warnings.warn(
            f"make_grid: grade truncada em max_specs={max_specs} de um total maior; "
            "aumente max_specs ou reduza lag_set/max_vars para cobertura completa.",
            stacklevel=2,
        )
    return specs


# ======================================================================
# Filtros: sinal econômico e VIF
# ======================================================================
def variable_effects(fit) -> dict:
    """Efeito (soma dos coeficientes das defasagens) de cada variável macro do ajuste."""
    spec = fit.spec
    if spec is None or not spec.exog:
        return {}
    out = {}
    for var, lag_list in spec.exog.items():
        s = 0.0
        for lag in lag_list:
            name = var if lag == 0 else f"{var}_l{lag}"
            if name in fit.params.index:
                s += float(fit.params[name])
        out[var] = s
    return out


def sign_ok(fit, expected_signs: Optional[Mapping[str, int]] = None) -> tuple[bool, list]:
    """Verifica o **sinal econômico** de cada variável (Guia §4.1).

    Compara o sinal do **efeito líquido** (soma das defasagens) de cada variável
    com o esperado (``+1``/``−1``). Devolve ``(ok, [variáveis_com_sinal_errado])``.
    Coeficiente com sinal trocado é *desqualificador*.
    """
    exp = dict(expected_signs or (fit.spec.expected_signs if fit.spec else {}) or {})
    if not exp:
        return True, []
    effects = variable_effects(fit)
    wrong = [v for v, s in exp.items() if v in effects and np.sign(effects[v]) != np.sign(s) and effects[v] != 0]
    return (len(wrong) == 0), wrong


def _macro_columns(fit) -> list[str]:
    spec = fit.spec
    if spec is None or not spec.exog:
        return []
    cols = []
    for var, lag_list in spec.exog.items():
        for lag in lag_list:
            cols.append(var if lag == 0 else f"{var}_l{lag}")
    return [c for c in cols if fit.exog is not None and c in fit.exog.columns]


def spec_max_vif(fit) -> float:
    """Maior VIF entre as colunas macro do ajuste (Guia §4.1)."""
    cols = _macro_columns(fit)
    if len(cols) < 2 or fit.exog is None:
        return 1.0
    return _diag.max_vif(fit.exog[cols])


# ======================================================================
# Validação fora da amostra (Guia §4.3)
# ======================================================================
def _oos_metrics(actuals: np.ndarray, preds: np.ndarray) -> dict:
    e = actuals - preds
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    denom = np.maximum(np.abs(actuals), 1e-6)
    mape = float(np.mean(np.abs(e) / denom))
    return {"rmse": rmse, "mae": mae, "mape": mape}


def walk_forward(
    build: Callable[[RiskSeries, Optional[pd.DataFrame]], SatelliteModel],
    series,
    macro: Optional[pd.DataFrame] = None,
    *,
    min_train: Optional[int] = None,
    horizon: int = 1,
    step: int = 1,
) -> dict:
    """Validação **walk-forward** (expanding window) — o padrão-ouro (Guia §4.3).

    Estima o modelo até ``t`` (a partir de ``min_train``), projeta ``horizon``
    passos usando a macro observada do período de teste, avança ``step`` e repete.
    ``build(series_treino, macro_treino)`` devolve um modelo **não ajustado** (é
    reajustado a cada janela).

    Returns
    -------
    dict
        ``rmse``/``mae``/``mape`` (escala original), ``errors`` (vetor de erros por
        (origem, passo), para o Diebold-Mariano), ``n_windows`` e ``horizon``.
    """
    rs: RiskSeries = as_risk_series(series)
    y = rs.values
    n = len(y)
    min_train = min_train or max(24, n // 2)
    preds_all, act_all, errors = [], [], []
    n_windows = 0
    for t in range(min_train, n - horizon + 1, step):
        idx_tr = y.index[:t]
        idx_te = y.index[t:t + horizon]
        s_tr = RiskSeries(y.loc[idx_tr], kind=rs.kind, segment=rs.segment, frequency=rs.frequency)
        m_tr = macro.loc[idx_tr] if macro is not None else None
        try:
            model = build(s_tr, m_tr)
            model.fit()
            fc = model.predict(macro.loc[idx_te] if macro is not None else None,
                               steps=horizon)
            p = np.asarray(fc.to_numpy(dtype=float))[:horizon]
        except Exception:  # noqa: BLE001 - janela problemática não derruba a validação
            continue
        a = y.loc[idx_te].to_numpy(dtype=float)[:horizon]
        if len(p) != len(a):
            continue
        preds_all.append(p)
        act_all.append(a)
        errors.append(a - p)
        n_windows += 1
    if not preds_all:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan,
                "errors": np.array([]), "n_windows": 0, "horizon": horizon}
    actuals = np.concatenate(act_all)
    preds = np.concatenate(preds_all)
    out = _oos_metrics(actuals, preds)
    out.update({"errors": np.concatenate(errors), "n_windows": n_windows, "horizon": horizon})
    return out


def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray, *, h: int = 1,
                    loss: str = "mse", harvey: bool = True) -> dict:
    """Teste de **Diebold-Mariano** (1995) de igualdade de acurácia preditiva.

    ``H0``: os dois modelos têm a mesma acurácia. Estatística sobre o diferencial
    de perda ``d_t = L(e_{a,t}) − L(e_{b,t})`` com variância de longo prazo (HAC,
    ``h−1`` defasagens); ``harvey`` aplica a correção de Harvey-Leybourne-Newbold
    para amostras pequenas e usa a ``t`` de Student. ``stat < 0`` ⇒ o modelo **a**
    é mais acurado (menor perda).
    """
    ea = np.asarray(errors_a, dtype=float)
    eb = np.asarray(errors_b, dtype=float)
    m = min(len(ea), len(eb))
    ea, eb = ea[:m], eb[:m]
    if m < 8:
        return {"stat": np.nan, "pvalue": np.nan, "n": m, "better": None}
    if loss == "mae":
        d = np.abs(ea) - np.abs(eb)
    else:
        d = ea ** 2 - eb ** 2
    dbar = float(np.mean(d))
    dc = d - dbar
    gamma0 = float(np.mean(dc ** 2))
    lrv = gamma0
    for k in range(1, h):
        gk = float(np.mean(dc[k:] * dc[:-k]))
        lrv += 2.0 * gk
    if lrv <= 0:
        return {"stat": np.nan, "pvalue": np.nan, "n": m, "better": None}
    dm = dbar / np.sqrt(lrv / m)
    if harvey:
        corr = np.sqrt(max(1e-9, (m + 1 - 2 * h + h * (h - 1) / m) / m))
        dm *= corr
        pval = 2.0 * float(stats.t.sf(abs(dm), df=m - 1))
    else:
        pval = 2.0 * float(stats.norm.sf(abs(dm)))
    better = "a" if dbar < 0 else "b"
    return {"stat": float(dm), "pvalue": pval, "n": m, "better": better}


# ======================================================================
# Busca e ranking champion-challenger (Guia §4.4)
# ======================================================================
@dataclass
class SearchResult:
    """Resultado da busca: ranking, melhor modelo (reajustado) e benchmarks."""

    ranking: pd.DataFrame
    best: Optional[SatelliteModel]
    best_spec: Optional[Specification]
    benchmarks: dict = field(default_factory=dict)

    def top(self, n: int = 10) -> pd.DataFrame:
        return self.ranking.head(n)

    def __repr__(self) -> str:  # pragma: no cover
        b = self.best_spec.describe() if self.best_spec else "—"
        return f"SearchResult(n_avaliados={len(self.ranking)}, melhor={b!r})"


def _benchmark_builds(kind: str, seasonal_period: int) -> dict:
    return {
        "random_walk": lambda s, m: RandomWalk(s),
        "media_historica": lambda s, m: HistoricalMean(s),
        "sazonal_ingenuo": lambda s, m: SeasonalNaive(s, period=seasonal_period),
        "ARIMA(1,0,0)": lambda s, m: ARIMA(s, order=(1, 0, 0)),
    }


def search(
    series,
    macro: Optional[pd.DataFrame],
    grid: Optional[Sequence[Specification]] = None,
    *,
    model_cls=ARDL,
    model_kwargs: Optional[dict] = None,
    candidates: Optional[Sequence[str]] = None,
    expected_signs: Optional[Mapping[str, int]] = None,
    link: Optional[str] = None,
    horizon: int = 6,
    min_train: Optional[int] = None,
    criterion: str = "oos_rmse",
    vif_max: float = 5.0,
    require_signs: bool = True,
    include_benchmarks: bool = True,
    grid_kwargs: Optional[dict] = None,
) -> SearchResult:
    """Executa a busca champion-challenger sobre uma grade (Guia §4).

    Para cada especificação: ajusta na amostra cheia (AIC/BIC, VIF, sinais), aplica
    os **filtros duros** (sinal econômico e VIF) e — para as que passam — roda a
    **validação walk-forward**. Ranqueia as qualificadas por ``criterion``
    (``"oos_rmse"`` padrão, ou ``"aic"``/``"bic"``) e reajusta a melhor na amostra
    cheia. Se ``include_benchmarks``, adiciona ARIMA e ingênuos ao ranking e a
    razão ``vs_arima`` (RMSE relativo).

    ``grid`` pode ser dado direto; senão é construído de ``candidates`` (+
    ``expected_signs``) via :func:`make_grid` (parâmetros extra em ``grid_kwargs``).
    """
    rs: RiskSeries = as_risk_series(series)
    model_kwargs = dict(model_kwargs or {})
    # 'link' serve só para montar a grade (o modelo deriva o link de spec.link e não
    # aceita 'link' no construtor); consome-o de model_kwargs para não ser repassado
    # (via **) ao construtor do modelo e quebrar todo ajuste.
    link = link or model_kwargs.pop("link", None) or rs.default_link
    if grid is None:
        if not candidates:
            raise ValueError("forneça 'grid' ou 'candidates' para montar a grade.")
        grid = make_grid(candidates, expected_signs=expected_signs, link=link,
                         **(grid_kwargs or {}))

    rows = []
    qualified: list[tuple] = []  # (spec, oos_rmse, errors)
    for spec in grid:
        if expected_signs and not spec.expected_signs:
            spec.expected_signs = {v: expected_signs[v] for v in spec.variables()
                                   if v in expected_signs}
        try:
            model = model_cls(rs, macro, spec, **model_kwargs)
            fr = model.fit()
        except Exception as exc:  # noqa: BLE001
            rows.append({"modelo": spec.describe(), "status": f"erro: {str(exc)[:40]}",
                         "n_vars": spec.n_terms(), "AIC": np.nan, "BIC": np.nan,
                         "max_vif": np.nan, "sinais_ok": None, "oos_rmse": np.nan})
            continue
        ok_sign, wrong = sign_ok(fr, spec.expected_signs)
        mvif = spec_max_vif(fr)
        status = "qualificado"
        if require_signs and not ok_sign:
            status = f"reprovado: sinal {wrong}"
        elif mvif > vif_max:
            status = f"reprovado: VIF={mvif:.1f}>{vif_max}"
        row = {"modelo": spec.describe(), "status": status, "n_vars": spec.n_terms(),
               "AIC": fr.aic, "BIC": fr.bic, "max_vif": mvif, "sinais_ok": ok_sign,
               "oos_rmse": np.nan, "oos_mae": np.nan}
        if status == "qualificado":
            wf = walk_forward(lambda s, m: model_cls(s, m, spec, **model_kwargs),
                              rs, macro, min_train=min_train, horizon=horizon)
            row["oos_rmse"] = wf["rmse"]
            row["oos_mae"] = wf["mae"]
            qualified.append((spec, wf["rmse"], wf["errors"]))
        rows.append(row)

    benchmarks = {}
    arima_rmse = np.nan
    if include_benchmarks:
        for name, build in _benchmark_builds(rs.kind, 12).items():
            try:
                wf = walk_forward(build, rs, macro if "ARIMA" not in name else None,
                                  min_train=min_train, horizon=horizon)
                benchmarks[name] = wf
                rows.append({"modelo": name, "status": "benchmark", "n_vars": 0,
                             "AIC": np.nan, "BIC": np.nan, "max_vif": np.nan,
                             "sinais_ok": None, "oos_rmse": wf["rmse"], "oos_mae": wf["mae"]})
                if name.startswith("ARIMA"):
                    arima_rmse = wf["rmse"]
            except Exception:  # noqa: BLE001
                pass

    ranking = pd.DataFrame(rows)
    if not ranking.empty and np.isfinite(arima_rmse):
        ranking["vs_arima"] = ranking["oos_rmse"] / arima_rmse

    # escolhe a melhor qualificada pelo critério
    best = best_spec = None
    if qualified:
        if criterion in ("aic", "bic"):
            key_col = criterion.upper()
            qual_rows = ranking[ranking["status"] == "qualificado"]
            best_desc = qual_rows.sort_values(key_col).iloc[0]["modelo"]
            best_spec = next(s for s, _, _ in qualified if s.describe() == best_desc)
        else:
            best_spec = min(qualified, key=lambda t: (np.inf if not np.isfinite(t[1]) else t[1]))[0]
        best = model_cls(rs, macro, best_spec, **model_kwargs)
        best.fit()

    # ordena o ranking: qualificados por critério, depois benchmarks, depois reprovados
    sort_col = "oos_rmse" if criterion == "oos_rmse" else criterion.upper()
    ranking["_ord"] = ranking["status"].map(
        lambda s: 0 if s == "qualificado" else (1 if s == "benchmark" else 2))
    ranking = ranking.sort_values(["_ord", sort_col], na_position="last").drop(columns="_ord")
    ranking = ranking.reset_index(drop=True)
    return SearchResult(ranking=ranking, best=best, best_spec=best_spec, benchmarks=benchmarks)


def compare(
    builds: Mapping[str, Callable],
    series,
    macro: Optional[pd.DataFrame] = None,
    *,
    horizon: int = 6,
    min_train: Optional[int] = None,
    champion: Optional[str] = None,
    h_dm: Optional[int] = None,
) -> pd.DataFrame:
    """Compara modelos nomeados por walk-forward + Diebold-Mariano vs o *champion*.

    ``builds`` é ``{nome: build(series, macro) -> modelo}``. O primeiro (ou
    ``champion``) é a referência; a coluna ``dm_pvalue`` testa se cada *challenger*
    difere dele em acurácia. Estrutura champion-challenger do guia (§4.4).
    """
    names = list(builds.keys())
    champ = champion or names[0]
    results = {name: walk_forward(build, series, macro, min_train=min_train, horizon=horizon)
               for name, build in builds.items()}
    champ_err = results[champ]["errors"]
    rows = []
    for name in names:
        wf = results[name]
        dm = diebold_mariano(wf["errors"], champ_err, h=h_dm or horizon) if name != champ else \
            {"stat": np.nan, "pvalue": np.nan}
        rows.append({"modelo": name, "rmse": wf["rmse"], "mae": wf["mae"],
                     "mape": wf["mape"], "n_windows": wf["n_windows"],
                     "dm_stat_vs_champ": dm["stat"], "dm_pvalue": dm["pvalue"],
                     "champion": name == champ})
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


__all__ = [
    "make_grid",
    "variable_effects",
    "sign_ok",
    "spec_max_vif",
    "walk_forward",
    "diebold_mariano",
    "SearchResult",
    "search",
    "compare",
]
