"""
Validação da estrutura a termo: calibração, discriminação e hipóteses
=====================================================================
O que a validação independente pergunta de uma curva de PD *lifetime*, em
quatro blocos, todos sobre um painel **fora do tempo** (a safra recente, ou a
janela de observação posterior à do ajuste):

**Calibração por horizonte** (:func:`backtest_curve`)
    A PD acumulada prevista em 12, 24, 36 períodos contra a observada por
    Kaplan-Meier, com o erro padrão de Greenwood: um ``z`` por horizonte e o
    veredito de estar dentro do IC.

**Discriminação por horizonte** (:func:`discrimination_by_horizon`,
:func:`concordance_index`)
    Se a curva é por contrato (covariáveis) ou por grupo, ela ordena o risco?
    AUC/KS/Gini do *default* em ``H`` períodos contra a PD acumulada prevista, e
    o **C-index de Harrell**, a métrica de ordenação própria de sobrevivência,
    que usa o tempo até o evento e respeita a censura.

**Calibração por decil** (:func:`calibration_by_decile`)
    Prevista × observada em faixas de PD prevista no horizonte escolhido, com
    o IC binomial de cada faixa e a estatística de Hosmer-Lemeshow.

**Riscos proporcionais** (:func:`ph_test`)
    A regressão de *hazard* assume que as covariáveis deslocam a linha de base
    pela **mesma** razão em todas as idades. O teste de razão de verossimilhança
    contra o modelo com interações covariável × ``ln(1 + idade)`` diz se essa
    hipótese sobrevive aos dados; quando não sobrevive, a curva do contrato
    jovem e a do maduro pedem inclinações diferentes.

Os contratos são avaliados **a partir da primeira observação** de cada um no
painel de validação: quem entra já com 10 meses tem a PD prevista para os
próximos ``H`` períodos a partir da idade 10 (a mesma convenção de
:meth:`~yggdrasil.credit_risk.ecl.lifetime_pd.LifetimePD.apply`). Contratos
censurados antes de ``H`` períodos **sem** *default* saem do denominador da
discriminação e da calibração por decil (não há como saber o desfecho deles);
o backtest por Kaplan-Meier, esse sim, os aproveita até onde foram observados.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..ecl.lifetime_pd import LifetimePD
from ..ecl.panel import ContractPanel
from ..ecl.survival import DiscreteHazard, kaplan_meier


# ======================================================================
# Desfechos por contrato
# ======================================================================
def contract_outcomes(panel: ContractPanel, horizon: int) -> pd.DataFrame:
    """Uma linha por contrato: idade de entrada, períodos observados, evento e o
    desfecho em ``horizon`` períodos (``NaN`` = censurado antes do horizonte).

    Devolve também todas as colunas da **primeira** observação do contrato (as
    covariáveis na entrada), para que a previsão seja feita a partir dali."""
    H = int(horizon)
    if H < 1:
        raise ValueError(f"horizon deve ser >= 1; recebido {horizon!r}.")
    d = panel.df
    g = d.groupby(panel.id_col, sort=False)
    primeiro = g.head(1).set_index(panel.id_col)
    ultimo = g.tail(1).set_index(panel.id_col)
    out = primeiro.copy()
    out["idade_entrada"] = primeiro[panel.age_col].to_numpy()
    out["idade_saida"] = ultimo[panel.age_col].reindex(out.index).to_numpy()
    out["periodos_observados"] = out["idade_saida"] - out["idade_entrada"] + 1
    out["evento"] = ultimo[panel.default_col].reindex(out.index).to_numpy().astype(int)
    alvo = np.where((out["evento"] == 1) & (out["periodos_observados"] <= H), 1.0,
                    np.where(out["periodos_observados"] >= H, 0.0, np.nan))
    # quem quebrou DEPOIS do horizonte conta como não-evento dentro dele
    alvo = np.where((out["evento"] == 1) & (out["periodos_observados"] > H), 0.0, alvo)
    out[f"alvo_h{H}"] = alvo
    return out.reset_index()


def _predicted_cumulative(model: LifetimePD, first: pd.DataFrame, panel: ContractPanel,
                          horizon: int) -> np.ndarray:
    seg = model.by if (model.by and model.by in first.columns) else None
    marg = model.marginal_matrix(first, horizon=int(horizon), age_col=panel.age_col,
                                 segment_col=seg)
    return marg.sum(axis=1)


# ======================================================================
# Calibração por horizonte (backtest)
# ======================================================================
def backtest_curve(model: LifetimePD, panel: ContractPanel,
                   horizons: Sequence[int] = (12, 24, 36), alpha: float = 0.05) -> pd.DataFrame:
    """Previsto × observado (Kaplan-Meier) por horizonte, com ``z`` e p-valor.

    Para cada grupo do modelo presente no painel: a PD acumulada que a curva
    prevê a partir da idade 0 contra a acumulada observada até ``h`` no painel
    de validação, com o erro padrão de Greenwood. ``z = (prevista − observada) /
    se``; ``dentro_do_ic`` usa o IC log-log do KM."""
    from scipy.stats import norm

    model._check_fit()
    partes = ({LifetimePD.GLOBAL: panel} if (model.by is None or model.by not in panel.df.columns)
              else panel.by(model.by))
    linhas = []
    for rot, parte in partes.items():
        chave = rot if rot in model.curves_ else (
            LifetimePD.GLOBAL if LifetimePD.GLOBAL in model.curves_ else None)
        if chave is None:
            continue
        curva = model.curves_[chave]
        try:
            _, tab = kaplan_meier(parte, alpha=alpha, return_table=True)
        except ValueError:
            continue
        for h in horizons:
            h = int(h)
            if h > len(tab) or h > len(curva):
                continue
            obs = tab.iloc[h - 1]
            prev = curva.pd_lifetime(h)
            se = float(obs["se_greenwood"])
            z = (prev - float(obs["pd_acumulada"])) / se if se > 0 else np.nan
            linhas.append({
                "grupo": rot, "horizonte": h,
                "n_em_risco_inicial": int(tab.iloc[0]["n_em_risco"]),
                "n_em_risco_h": int(obs["n_em_risco"]),
                "pd_prevista": prev, "pd_observada": float(obs["pd_acumulada"]),
                "se_greenwood": se,
                "ic_inf": float(obs["pd_acumulada_ic_inf"]),
                "ic_sup": float(obs["pd_acumulada_ic_sup"]),
                "erro_absoluto": prev - float(obs["pd_acumulada"]),
                "erro_relativo": (prev / float(obs["pd_acumulada"]) - 1.0
                                  if obs["pd_acumulada"] > 0 else np.nan),
                "z": z,
                "p_valor": float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else np.nan,
                "dentro_do_ic": bool(obs["pd_acumulada_ic_inf"] <= prev <= obs["pd_acumulada_ic_sup"]),
            })
    if not linhas:
        raise ValueError(
            "o backtest não produziu linhas: confira se os grupos do painel batem com os do "
            "ajuste e se os horizontes cabem na janela observada."
        )
    return pd.DataFrame(linhas)


# ======================================================================
# Discriminação
# ======================================================================
def _ks(y: np.ndarray, s: np.ndarray) -> float:
    ordem = np.argsort(-s, kind="mergesort")
    y = y[ordem]
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    tpr = np.cumsum(y) / pos
    fpr = np.cumsum(1 - y) / neg
    return float(np.max(np.abs(tpr - fpr)))


def discrimination_by_horizon(model: LifetimePD, panel: ContractPanel,
                              horizons: Sequence[int] = (12, 24, 36)) -> pd.DataFrame:
    """AUC, Gini e KS do *default* em ``h`` períodos contra a PD acumulada prevista.

    Um modelo com curva única (sem grupos nem covariáveis) dá a mesma PD a todos
    e não ordena: AUC sai ``NaN`` e a coluna ``observacao`` explica. Contratos
    censurados antes de ``h`` sem *default* ficam de fora."""
    from sklearn.metrics import roc_auc_score

    model._check_fit()
    linhas = []
    for h in horizons:
        h = int(h)
        out = contract_outcomes(panel, h)
        alvo = out[f"alvo_h{h}"].to_numpy(dtype=float)
        ok = np.isfinite(alvo)
        if ok.sum() == 0:
            linhas.append({"horizonte": h, "n": 0, "n_eventos": 0, "taxa_observada": np.nan,
                           "pd_prevista_media": np.nan, "auc": np.nan, "gini": np.nan,
                           "ks": np.nan, "observacao": "nenhum contrato observado até o horizonte"})
            continue
        prev = _predicted_cumulative(model, out[ok], panel, h)
        y = alvo[ok].astype(int)
        n_ev = int(y.sum())
        obs = ""
        if n_ev == 0 or n_ev == len(y):
            auc = gini = ks = np.nan
            obs = "só uma classe observada no horizonte"
        elif np.ptp(prev) < 1e-12:
            auc = gini = ks = np.nan
            obs = "PD prevista idêntica para todos (curva única): nada a ordenar"
        else:
            auc = float(roc_auc_score(y, prev))
            gini = 2.0 * auc - 1.0
            ks = _ks(y, prev)
        linhas.append({"horizonte": h, "n": int(ok.sum()), "n_eventos": n_ev,
                       "taxa_observada": float(y.mean()),
                       "pd_prevista_media": float(np.mean(prev)),
                       "auc": auc, "gini": gini, "ks": ks, "observacao": obs})
    return pd.DataFrame(linhas)


class _Fenwick:
    """Árvore de Fenwick (soma de prefixos) para contar postos já inseridos."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.t = np.zeros(n + 1, dtype=np.int64)

    def add(self, i: int) -> None:
        i += 1
        while i <= self.n:
            self.t[i] += 1
            i += i & (-i)

    def prefix(self, i: int) -> int:
        """Nº de elementos com posto ``< i``."""
        s = 0
        while i > 0:
            s += int(self.t[i])
            i -= i & (-i)
        return s


def concordance_index(score, duration, event) -> float:
    """C-index de Harrell: pares comparáveis em que o **maior** ``score`` quebrou
    **antes**, com censura à direita.

    Um par ``(i, j)`` é comparável quando ``i`` sofreu o evento e ``j`` foi
    observado por mais tempo (``duração_j > duração_i``, ou censurado exatamente
    em ``duração_i``). Empate de ``score`` vale meio ponto. ``0,5`` = ordenação
    aleatória; ``1`` = perfeita. Implementação em ``O(n log n)``."""
    s = np.asarray(score, dtype=float).ravel()
    t = np.asarray(duration, dtype=float).ravel()
    e = np.asarray(event).astype(int).ravel()
    if not (len(s) == len(t) == len(e)):
        raise ValueError("score, duration e event devem ter o mesmo comprimento.")
    ok = np.isfinite(s) & np.isfinite(t)
    s, t, e = s[ok], t[ok], e[ok]
    if e.sum() == 0:
        return float("nan")
    postos_unicos = np.unique(s)
    posto = np.searchsorted(postos_unicos, s)
    arvore = _Fenwick(len(postos_unicos))
    concordantes = empates = comparaveis = 0.0
    inseridos = 0
    for T in np.unique(t)[::-1]:
        idx = np.flatnonzero(t == T)
        censurados = idx[e[idx] == 0]
        eventos = idx[e[idx] == 1]
        for i in censurados:            # sobreviveram além (ou até) de T: comparáveis
            arvore.add(int(posto[i]))
            inseridos += 1
        for i in eventos:
            if inseridos == 0:
                continue
            menores = arvore.prefix(int(posto[i]))
            ate_igual = arvore.prefix(int(posto[i]) + 1)
            concordantes += menores
            empates += ate_igual - menores
            comparaveis += inseridos
        for i in eventos:
            arvore.add(int(posto[i]))
            inseridos += 1
    if comparaveis == 0:
        return float("nan")
    return float((concordantes + 0.5 * empates) / comparaveis)


def model_concordance(model: LifetimePD, panel: ContractPanel, horizon: int = 36) -> float:
    """C-index do modelo no painel: ``score`` = PD acumulada prevista em
    ``horizon`` a partir da entrada; tempo = períodos observados; evento = *default*."""
    out = contract_outcomes(panel, horizon)
    prev = _predicted_cumulative(model, out, panel, horizon)
    return concordance_index(prev, out["periodos_observados"].to_numpy(),
                             out["evento"].to_numpy())


# ======================================================================
# Calibração por decil
# ======================================================================
def calibration_by_decile(model: LifetimePD, panel: ContractPanel, horizon: int = 12,
                          n_bins: int = 10, alpha: float = 0.05) -> pd.DataFrame:
    """Prevista × observada por faixa de PD prevista em ``horizon`` períodos.

    Faixas por quantis da PD prevista (com poucas PDs distintas, uma faixa por
    valor; faixas vazias colapsam). ``attrs`` da
    tabela trazem ``hl_estatistica``, ``hl_gl`` e ``hl_p_valor`` (Hosmer-
    Lemeshow, ``gl = faixas − 2``)."""
    from scipy.stats import chi2

    from ...metrics.calibration import binomial_ci

    out = contract_outcomes(panel, horizon)
    alvo = out[f"alvo_h{int(horizon)}"].to_numpy(dtype=float)
    ok = np.isfinite(alvo)
    if ok.sum() == 0:
        raise ValueError("nenhum contrato observado até o horizonte pedido.")
    prev = _predicted_cumulative(model, out[ok], panel, horizon)
    y = alvo[ok]
    unicos = np.unique(prev)
    if len(unicos) <= int(n_bins):
        # poucas PDs distintas (curva por grupo): uma faixa por valor previsto
        faixa = np.searchsorted(unicos, prev)
    else:
        try:
            faixa = pd.qcut(prev, q=int(n_bins), labels=False, duplicates="drop")
        except ValueError:
            faixa = np.zeros(len(prev), dtype=int)
    faixa = np.asarray(pd.Series(faixa).fillna(0).astype(int))
    linhas = []
    for f in np.unique(faixa):
        m = faixa == f
        n = int(m.sum())
        ev = float(y[m].sum())
        p_med = float(prev[m].mean())
        inf, sup = binomial_ci(ev, n, alpha=alpha)
        linhas.append({"faixa": int(f) + 1, "n": n, "n_eventos": int(ev),
                       "pd_prevista": p_med, "pd_observada": ev / n,
                       "ic_inf": float(inf), "ic_sup": float(sup),
                       "pd_min": float(prev[m].min()), "pd_max": float(prev[m].max()),
                       "dentro_do_ic": bool(inf <= p_med <= sup)})
    tab = pd.DataFrame(linhas)
    esperado = tab["pd_prevista"] * tab["n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        termo = (tab["n_eventos"] - esperado) ** 2 / (esperado * (1.0 - tab["pd_prevista"]))
    termo = termo.replace([np.inf, -np.inf], np.nan).dropna()
    hl = float(termo.sum())
    gl = max(len(tab) - 2, 1)
    tab.attrs["hl_estatistica"] = hl
    tab.attrs["hl_gl"] = gl
    tab.attrs["hl_p_valor"] = float(chi2.sf(hl, gl))
    tab.attrs["horizonte"] = int(horizon)
    return tab


# ======================================================================
# Riscos proporcionais
# ======================================================================
def _loglik(model: DiscreteHazard, panel: ContractPanel, features: Sequence[str],
            max_age: Optional[int]) -> float:
    d = panel.spells(features=list(features), max_age=max_age)
    X = d[list(features)].to_numpy(dtype=float) if features else None
    h = np.clip(model.predict_hazard(d[panel.age_col].to_numpy(dtype=int), X), 1e-12, 1 - 1e-12)
    y = d[panel.default_col].to_numpy(dtype=float)
    return float(np.sum(y * np.log(h) + (1.0 - y) * np.log(1.0 - h)))


def ph_test(panel: ContractPanel, features: Sequence[str], baseline: str = "spline",
            link: str = "logit", C: float = 1e6, max_age: Optional[int] = None,
            n_knots: int = 6, degree: int = 3, alpha: float = 0.05) -> dict:
    """Teste de **riscos proporcionais** por razão de verossimilhança.

    Compara a regressão de *hazard* com as ``features`` contra a mesma regressão
    acrescida das interações ``feature × ln(1 + idade)``. Sob H0 (efeitos
    proporcionais, isto é, constantes na idade) a estatística
    ``2·(LL_com − LL_sem)`` é qui-quadrado com ``len(features)`` graus de
    liberdade. Devolve a estatística, o p-valor, o veredito e o coeficiente de
    cada interação (o sinal diz se o efeito **cresce** ou **encolhe** com a
    idade)."""
    from scipy.stats import chi2

    feats = list(features)
    if not feats:
        raise ValueError("informe ao menos uma feature para testar.")
    kw = dict(baseline=baseline, link=link, C=C, max_age=max_age, n_knots=n_knots, degree=degree)
    base = DiscreteHazard(**kw).fit(panel, features=feats)
    ll0 = _loglik(base, panel, feats, max_age)

    d = panel.spells(features=feats, max_age=max_age).copy()
    inter = []
    ln_idade = np.log1p(d[panel.age_col].to_numpy(dtype=float))
    for f in feats:
        nome = f"{f}:ln_idade"
        d[nome] = d[f].to_numpy(dtype=float) * ln_idade
        inter.append(nome)
    p_aug = ContractPanel(d, id_col=panel.id_col, date_col=panel.date_col,
                          default_col=panel.default_col, age_col=panel.age_col,
                          segment_col=panel.segment_col, exposure_col=panel.exposure_col,
                          term_col=panel.term_col, freq=panel.freq, drop_post_default=False)
    aug = DiscreteHazard(**kw).fit(p_aug, features=feats + inter)
    ll1 = _loglik(aug, p_aug, feats + inter, None)

    estat = max(2.0 * (ll1 - ll0), 0.0)
    gl = len(feats)
    p = float(chi2.sf(estat, gl))
    coef = aug.coef_frame()
    tab = coef[coef["termo"].isin(inter)][["termo", "coeficiente"]].reset_index(drop=True)
    tab["feature"] = [t.split(":")[0] for t in tab["termo"]]
    tab["leitura"] = np.where(tab["coeficiente"] > 0, "efeito cresce com a idade",
                              "efeito encolhe com a idade")
    return {"estatistica": float(estat), "gl": gl, "p_valor": p, "alpha": alpha,
            "proporcional": bool(p > alpha), "loglik_sem": ll0, "loglik_com": ll1,
            "interacoes": tab}


__all__ = ["contract_outcomes", "backtest_curve", "discrimination_by_horizon",
           "concordance_index", "model_concordance", "calibration_by_decile", "ph_test"]
