"""
Bateria de diagnóstico de séries e resíduos (Guia §2.3, §4.2, §6.2 ``diagnostics``)
=================================================================================
O **ritual obrigatório** do guia, com saída **tabular padronizada**. Antes de
qualquer regressão, cada série passa pelos testes de estacionariedade,
sazonalidade e quebra (§2.3); depois do ajuste, os resíduos passam pela bateria
de autocorrelação, heterocedasticidade, normalidade e estabilidade (§4.2). Todo
teste devolve o mesmo objeto :class:`DiagnosticResult`, e os relatórios
(:func:`stationarity_report`, :func:`residual_report`) empilham vários numa única
tabela — a "saída tabular única" que o guia pede (§6.2, marco do Bloco 3).

Motores: :mod:`statsmodels` (ADF, KPSS, Ljung-Box, Breusch-Pagan/White,
Jarque-Bera, CUSUM) e :mod:`arch` (Phillips-Perron). Onde a hipótese nula importa
para a leitura, o campo :attr:`DiagnosticResult.h0` documenta-a — ADF e KPSS têm
**nulas opostas** e por isso são lidos **em conjunto** (§2.3).

Convenção de ``passed``: ``True`` = resultado **desejável** para um bom modelo
(série estacionária; resíduos sem autocorrelação/heterocedasticidade; normais;
coeficientes estáveis). ``None`` quando o teste não conclui (ex.: White sem graus
de liberdade suficientes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch,
    het_breuschpagan,
    het_white,
)
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss


# ======================================================================
# Resultado padronizado
# ======================================================================
@dataclass
class DiagnosticResult:
    """Resultado de um teste de diagnóstico, num formato uniforme."""

    test: str
    statistic: float
    pvalue: Optional[float] = None
    passed: Optional[bool] = None
    h0: str = ""
    conclusion: str = ""
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        row = {
            "teste": self.test,
            "estatistica": self.statistic,
            "p_valor": self.pvalue,
            "ok": self.passed,
            "H0": self.h0,
            "conclusao": self.conclusion,
        }
        row.update(self.extra)
        return row

    def __repr__(self) -> str:  # pragma: no cover
        p = f"{self.pvalue:.4f}" if self.pvalue is not None else "—"
        return f"DiagnosticResult({self.test!r}, stat={self.statistic:.4f}, p={p}, ok={self.passed})"


def _frame(results: Sequence[DiagnosticResult]) -> pd.DataFrame:
    """Empilha :class:`DiagnosticResult` numa tabela (colunas homogêneas)."""
    return pd.DataFrame([r.to_row() for r in results])


def _as_1d(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("série vazia após remover não-finitos.")
    return a


def _with_const(exog: pd.DataFrame) -> np.ndarray:
    """Garante uma coluna constante no design (para os testes de heterocedasticidade)."""
    X = np.asarray(exog, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if not np.allclose(X[:, 0], 1.0) and not any(np.allclose(X[:, j], 1.0) for j in range(X.shape[1])):
        X = np.column_stack([np.ones(X.shape[0]), X])
    return X


# ======================================================================
# Estacionariedade (Guia §2.3): ADF, KPSS, Phillips-Perron
# ======================================================================
def adf(series, regression: str = "c", autolag: str = "AIC", alpha: float = 0.05) -> DiagnosticResult:
    """**ADF** (Dickey-Fuller aumentado). H0: **raiz unitária** (não estacionária).

    ``passed=True`` (estacionária) quando ``p < alpha`` — rejeita a raiz unitária.
    """
    x = _as_1d(series)
    stat, pval, usedlag, nobs, crit, _ = adfuller(x, regression=regression, autolag=autolag)
    est = pval < alpha
    return DiagnosticResult(
        test="ADF", statistic=float(stat), pvalue=float(pval), passed=bool(est),
        h0="raiz unitária (não estacionária)",
        conclusion="estacionária" if est else "não estacionária (raiz unitária)",
        extra={"lags": int(usedlag), "n": int(nobs), "crit_5%": float(crit["5%"])},
    )


def kpss_test(series, regression: str = "c", nlags="auto", alpha: float = 0.05) -> DiagnosticResult:
    """**KPSS**. H0: **estacionária** (nula **oposta** à do ADF).

    ``passed=True`` (estacionária) quando ``p > alpha`` — **não** rejeita a nula.
    Lê-se em conjunto com o ADF (§2.3): ambos apontando estacionariedade ⇒ I(0);
    ADF não rejeita e KPSS rejeita ⇒ forte indício de I(1).
    """
    x = _as_1d(series)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # InterpolationWarning quando p fora da tabela
        stat, pval, lags, crit = kpss(x, regression=regression, nlags=nlags)
    est = pval > alpha
    return DiagnosticResult(
        test="KPSS", statistic=float(stat), pvalue=float(pval), passed=bool(est),
        h0="estacionária",
        conclusion="estacionária" if est else "não estacionária",
        extra={"lags": int(lags), "crit_5%": float(crit["5%"])},
    )


def phillips_perron(series, trend: str = "c", alpha: float = 0.05) -> DiagnosticResult:
    """**Phillips-Perron**. H0: **raiz unitária**. Correção não-paramétrica de
    autocorrelação/heterocedasticidade (robusto onde o ADF exige muitos lags).

    Usa :mod:`arch` (import tardio); se ``arch`` não estiver instalado, devolve um
    resultado inconclusivo (``passed=None``) em vez de quebrar.
    """
    x = _as_1d(series)
    try:
        from arch.unitroot import PhillipsPerron  # import tardio (dep. opcional)
    except ImportError:
        return DiagnosticResult(
            test="Phillips-Perron", statistic=np.nan, pvalue=None, passed=None,
            h0="raiz unitária", conclusion="indisponível (instale 'arch')",
        )
    pp = PhillipsPerron(x, trend=trend)
    est = pp.pvalue < alpha
    return DiagnosticResult(
        test="Phillips-Perron", statistic=float(pp.stat), pvalue=float(pp.pvalue),
        passed=bool(est), h0="raiz unitária",
        conclusion="estacionária" if est else "não estacionária (raiz unitária)",
        extra={"lags": int(pp.lags)},
    )


def integration_order(series, alpha: float = 0.05, max_d: int = 2) -> int:
    """Ordem de integração ``d`` estimada: diferencia até ADF **e** KPSS
    concordarem em estacionariedade (ou até ``max_d``)."""
    s = pd.Series(np.asarray(series, dtype=float))
    for d in range(max_d + 1):
        a = adf(s, alpha=alpha)
        try:
            k = kpss_test(s, alpha=alpha)
            ok = a.passed and k.passed
        except Exception:  # noqa: BLE001 - KPSS pode falhar em séries curtas
            ok = a.passed
        if ok:
            return d
        s = s.diff().dropna()
    return max_d


def stationarity_report(series, alpha: float = 0.05) -> pd.DataFrame:
    """Tabela ADF + KPSS + PP + veredito de ordem de integração (Guia §2.3)."""
    res = [adf(series, alpha=alpha)]
    try:
        res.append(kpss_test(series, alpha=alpha))
    except Exception as exc:  # noqa: BLE001
        res.append(DiagnosticResult("KPSS", np.nan, None, None, "estacionária",
                                    f"indisponível: {str(exc)[:60]}"))
    res.append(phillips_perron(series, alpha=alpha))
    df = _frame(res)
    df.attrs["ordem_integracao"] = integration_order(series, alpha=alpha)
    return df


# ======================================================================
# Autocorrelação dos resíduos (Guia §4.2): Ljung-Box, Breusch-Godfrey, DW
# ======================================================================
def ljung_box(resid, lags: Optional[int] = None, alpha: float = 0.05) -> DiagnosticResult:
    """**Ljung-Box**. H0: **sem autocorrelação** até ``lags``.

    ``passed=True`` (resíduo limpo) quando ``p > alpha``. ``lags`` padrão é
    ``min(10, n//4)``.
    """
    x = _as_1d(resid)
    n = x.size
    L = lags or max(1, min(10, n // 4))
    out = acorr_ljungbox(x, lags=[L], return_df=True)
    stat = float(out["lb_stat"].iloc[0])
    pval = float(out["lb_pvalue"].iloc[0])
    ok = pval > alpha
    return DiagnosticResult(
        test="Ljung-Box", statistic=stat, pvalue=pval, passed=bool(ok),
        h0="sem autocorrelação",
        conclusion="resíduo sem autocorrelação" if ok else "autocorrelação residual",
        extra={"lags": int(L)},
    )


def breusch_godfrey(resid, exog: pd.DataFrame, nlags: int = 4, alpha: float = 0.05) -> DiagnosticResult:
    """**Breusch-Godfrey** (LM de autocorrelação de ordem ``nlags``). H0: sem
    autocorrelação. Regressão auxiliar de ``u_t`` sobre os regressores e ``u_{t-1..p}``;
    ``LM = n·R² ~ χ²(p)``. Preferível ao Ljung-Box na presença de regressores/AR."""
    u = np.asarray(resid, dtype=float)
    n = u.size
    Z = _with_const(exog) if exog is not None else np.ones((n, 1))
    if Z.shape[0] != n:
        raise ValueError("exog e resid devem ter o mesmo nº de observações.")
    # colunas de resíduo defasado (preenche o início com 0)
    lagged = np.zeros((n, nlags))
    for k in range(1, nlags + 1):
        lagged[k:, k - 1] = u[:-k]
    A = np.column_stack([Z, lagged])
    beta, *_ = np.linalg.lstsq(A, u, rcond=None)
    resid_aux = u - A @ beta
    ss_res = float(np.sum(resid_aux ** 2))
    ss_tot = float(np.sum((u - u.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    lm = n * r2
    pval = float(stats.chi2.sf(lm, nlags))
    ok = pval > alpha
    return DiagnosticResult(
        test="Breusch-Godfrey", statistic=float(lm), pvalue=pval, passed=bool(ok),
        h0="sem autocorrelação",
        conclusion="sem autocorrelação" if ok else "autocorrelação residual",
        extra={"lags": int(nlags)},
    )


def durbin_watson_stat(resid) -> DiagnosticResult:
    """**Durbin-Watson** (autocorrelação de 1ª ordem). ~2 ⇒ sem autocorrelação;
    <1,5 positiva; >2,5 negativa. Sem p-valor por construção."""
    x = _as_1d(resid)
    dw = float(durbin_watson(x))
    ok = 1.5 <= dw <= 2.5
    return DiagnosticResult(
        test="Durbin-Watson", statistic=dw, pvalue=None, passed=bool(ok),
        h0="sem autocorrelação de 1ª ordem",
        conclusion="~2 (ok)" if ok else ("autocorr. positiva" if dw < 1.5 else "autocorr. negativa"),
    )


# ======================================================================
# Heterocedasticidade (Guia §4.2): Breusch-Pagan, White, ARCH-LM
# ======================================================================
def breusch_pagan(resid, exog: pd.DataFrame, alpha: float = 0.05) -> DiagnosticResult:
    """**Breusch-Pagan**. H0: **homocedasticidade**. ``passed=True`` quando ``p > alpha``."""
    u = np.asarray(resid, dtype=float)
    X = _with_const(exog)
    lm, lm_p, f, f_p = het_breuschpagan(u, X)
    ok = lm_p > alpha
    return DiagnosticResult(
        test="Breusch-Pagan", statistic=float(lm), pvalue=float(lm_p), passed=bool(ok),
        h0="homocedasticidade",
        conclusion="homocedástico" if ok else "heterocedástico",
    )


def white_test(resid, exog: pd.DataFrame, alpha: float = 0.05) -> DiagnosticResult:
    """**White** (heterocedasticidade geral, com termos cruzados). H0: homocedástico.

    Exige ``n`` grande relativo ao nº de termos cruzados; se não houver graus de
    liberdade, devolve ``passed=None`` (inconclusivo)."""
    u = np.asarray(resid, dtype=float)
    X = _with_const(exog)
    try:
        lm, lm_p, f, f_p = het_white(u, X)
    except Exception as exc:  # noqa: BLE001 - típico: poucos graus de liberdade
        return DiagnosticResult(
            test="White", statistic=np.nan, pvalue=None, passed=None,
            h0="homocedasticidade", conclusion=f"inconclusivo: {str(exc)[:50]}")
    ok = lm_p > alpha
    return DiagnosticResult(
        test="White", statistic=float(lm), pvalue=float(lm_p), passed=bool(ok),
        h0="homocedasticidade",
        conclusion="homocedástico" if ok else "heterocedástico",
    )


def arch_lm(resid, nlags: int = 4, alpha: float = 0.05) -> DiagnosticResult:
    """**ARCH-LM** (Engle): clusterização de volatilidade nos resíduos. H0: sem
    efeito ARCH. Relevante para intervalos de projeção (§4.2)."""
    u = _as_1d(resid)
    lm, lm_p, f, f_p = het_arch(u, nlags=nlags)
    ok = lm_p > alpha
    return DiagnosticResult(
        test="ARCH-LM", statistic=float(lm), pvalue=float(lm_p), passed=bool(ok),
        h0="sem efeito ARCH",
        conclusion="sem ARCH" if ok else "volatilidade condicional (ARCH)",
        extra={"lags": int(nlags)},
    )


# ======================================================================
# Normalidade (Guia §4.2): Jarque-Bera
# ======================================================================
def jarque_bera_test(resid, alpha: float = 0.05) -> DiagnosticResult:
    """**Jarque-Bera**. H0: **normalidade** (relevante para os intervalos de
    projeção, mais que para os pontos, §4.2)."""
    x = _as_1d(resid)
    jb, jb_p, skew, kurt = jarque_bera(x)
    ok = jb_p > alpha
    return DiagnosticResult(
        test="Jarque-Bera", statistic=float(jb), pvalue=float(jb_p), passed=bool(ok),
        h0="normalidade",
        conclusion="normal" if ok else "não normal",
        extra={"assimetria": float(skew), "curtose": float(kurt)},
    )


# ======================================================================
# Colinearidade (Guia §4.1): VIF
# ======================================================================
def vif(exog: pd.DataFrame) -> pd.DataFrame:
    """**VIF** (fator de inflação da variância) por variável (Guia §4.1).

    ``VIF_j = 1/(1−R²_j)``, onde ``R²_j`` é o ajuste da regressão da variável ``j``
    nas demais. Regra prática: VIF > 5 (ou 10) sinaliza colinearidade — o filtro de
    seleção usa isso para descartar especificações com macro redundante
    (desemprego e renda disputando o mesmo papel). A constante é ignorada.
    """
    X = exog.copy()
    # remove colunas constantes (const/intercepto e degeneradas)
    const_cols = [c for c in X.columns if np.allclose(X[c].to_numpy(dtype=float),
                                                       X[c].to_numpy(dtype=float)[0])]
    X = X.drop(columns=const_cols)
    cols = list(X.columns)
    if len(cols) < 2:
        return pd.DataFrame({"variavel": cols, "VIF": [1.0] * len(cols)})
    Xm = X.to_numpy(dtype=float)
    rows = []
    for j, c in enumerate(cols):
        y = Xm[:, j]
        others = np.column_stack([np.ones(Xm.shape[0]), np.delete(Xm, j, axis=1)])
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        resid = y - others @ beta
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        v = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
        rows.append({"variavel": c, "VIF": float(v)})
    return pd.DataFrame(rows)


def max_vif(exog: pd.DataFrame) -> float:
    """O maior VIF do design (atalho para o filtro de seleção)."""
    tab = vif(exog)
    return float(tab["VIF"].max()) if len(tab) else 1.0


# ======================================================================
# Estabilidade e quebras (Guia §2.3, §4.2): Chow, Quandt-Andrews, CUSUM
# ======================================================================
def _ols_rss(y: np.ndarray, X: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(np.sum(r ** 2))


def chow_test(y, exog: pd.DataFrame, break_index: int, alpha: float = 0.05) -> DiagnosticResult:
    """**Chow** para quebra em **data conhecida** ``break_index`` (Guia §2.3).

    H0: **sem quebra** (coeficientes iguais nos dois subperíodos). ``passed=True``
    (estável) quando ``p > alpha``."""
    y = np.asarray(y, dtype=float)
    X = _with_const(exog)
    n, k = X.shape
    i = int(break_index)
    if not (k < i < n - k):
        raise ValueError(f"break_index {i} sem graus de liberdade suficientes (n={n}, k={k}).")
    rss_p = _ols_rss(y, X)
    rss_1 = _ols_rss(y[:i], X[:i])
    rss_2 = _ols_rss(y[i:], X[i:])
    num = (rss_p - (rss_1 + rss_2)) / k
    den = (rss_1 + rss_2) / (n - 2 * k)
    f = num / den if den > 0 else np.inf
    pval = float(stats.f.sf(f, k, n - 2 * k))
    ok = pval > alpha
    return DiagnosticResult(
        test="Chow", statistic=float(f), pvalue=pval, passed=bool(ok),
        h0="sem quebra estrutural",
        conclusion="estável" if ok else "quebra na data",
        extra={"break_index": i},
    )


#: Valores críticos assintóticos do sup-F de Quandt-Andrews (trimming 15%),
#: aproximados de Andrews (1993, Tabela 1) por nº de parâmetros que quebram.
_QANDREWS_CRIT_5 = {1: 8.85, 2: 11.79, 3: 14.15, 4: 16.30, 5: 18.30, 6: 20.26,
                    7: 22.16, 8: 24.01, 9: 25.83, 10: 27.63}


def quandt_andrews(y, exog: pd.DataFrame, trim: float = 0.15) -> DiagnosticResult:
    """**Quandt-Andrews sup-F** para quebra em **data desconhecida** (§2.3).

    Ponto de entrada para o espírito **Bai-Perron** (a quebra dominante): varre os
    pontos de quebra candidatos no miolo ``[trim, 1−trim]`` da amostra, calcula o
    Chow-F em cada um e reporta o **máximo** (``sup-F``) e a data que o atinge. A
    comparação usa o valor crítico assintótico de Andrews (:data:`_QANDREWS_CRIT_5`,
    trimming 15%): ``passed=False`` (há quebra) quando ``sup-F`` excede o crítico.
    """
    y = np.asarray(y, dtype=float)
    X = _with_const(exog)
    n, k = X.shape
    lo = max(k + 1, int(np.floor(trim * n)))
    hi = min(n - k - 1, int(np.ceil((1 - trim) * n)))
    if lo >= hi:
        raise ValueError("amostra curta demais para o sup-F com este trimming.")
    rss_p = _ols_rss(y, X)
    best_f, best_i = -np.inf, lo
    for i in range(lo, hi + 1):
        rss_1 = _ols_rss(y[:i], X[:i])
        rss_2 = _ols_rss(y[i:], X[i:])
        den = (rss_1 + rss_2) / (n - 2 * k)
        f = ((rss_p - (rss_1 + rss_2)) / k) / den if den > 0 else np.inf
        if f > best_f:
            best_f, best_i = f, i
    crit = _QANDREWS_CRIT_5.get(min(k, 10), _QANDREWS_CRIT_5[10])
    ok = best_f < crit  # abaixo do crítico ⇒ estável
    return DiagnosticResult(
        test="Quandt-Andrews sup-F", statistic=float(best_f), pvalue=None, passed=bool(ok),
        h0="sem quebra (coeficientes estáveis)",
        conclusion="estável" if ok else f"quebra provável em t={best_i}",
        extra={"break_index": int(best_i), "crit_5%": float(crit), "n_restricoes": int(k)},
    )


def cusum(resid, ddof: int = 0, alpha: float = 0.05) -> DiagnosticResult:
    """**CUSUM** dos resíduos de OLS (Ploberger-Krämer). H0: **coeficientes
    estáveis**. Falha de estabilidade é o alerta mais sério do guia (§4.2) —
    sugere quebra/mudança de regime não capturada pela especificação."""
    u = _as_1d(resid)
    from statsmodels.stats.diagnostic import breaks_cusumolsresid
    stat, pval, crit = breaks_cusumolsresid(u, ddof=ddof)
    ok = pval > alpha
    return DiagnosticResult(
        test="CUSUM", statistic=float(stat), pvalue=float(pval), passed=bool(ok),
        h0="coeficientes estáveis",
        conclusion="estável" if ok else "instabilidade / quebra",
    )


# ======================================================================
# Relatórios consolidados
# ======================================================================
def residual_report(resid, exog: Optional[pd.DataFrame] = None, alpha: float = 0.05) -> pd.DataFrame:
    """Bateria de **resíduos** numa tabela (Guia §4.2).

    Sempre roda Ljung-Box, Durbin-Watson, ARCH-LM, Jarque-Bera e CUSUM. Se
    ``exog`` for dado (modelos de regressão), acrescenta Breusch-Godfrey,
    Breusch-Pagan e White — que precisam dos regressores.
    """
    res: list[DiagnosticResult] = [ljung_box(resid, alpha=alpha), durbin_watson_stat(resid)]
    if exog is not None and len(exog.columns) > 0:
        res.append(breusch_godfrey(resid, exog, alpha=alpha))
        res.append(breusch_pagan(resid, exog, alpha=alpha))
        res.append(white_test(resid, exog, alpha=alpha))
    res.append(arch_lm(resid, alpha=alpha))
    res.append(jarque_bera_test(resid, alpha=alpha))
    try:
        res.append(cusum(resid, alpha=alpha))
    except Exception as exc:  # noqa: BLE001 - CUSUM pode falhar em amostras muito curtas
        res.append(DiagnosticResult("CUSUM", np.nan, None, None,
                                    "coeficientes estáveis", f"indisponível: {str(exc)[:50]}"))
    return _frame(res)


def series_report(series, alpha: float = 0.05) -> pd.DataFrame:
    """Relatório de **propriedades da série** (Guia §2.3, marco do Bloco 3):
    estacionariedade (ADF/KPSS/PP) — o cartão de entrada antes de modelar."""
    return stationarity_report(series, alpha=alpha)


__all__ = [
    "DiagnosticResult",
    # estacionariedade
    "adf", "kpss_test", "phillips_perron", "integration_order",
    "stationarity_report", "series_report",
    # autocorrelação
    "ljung_box", "breusch_godfrey", "durbin_watson_stat",
    # heterocedasticidade
    "breusch_pagan", "white_test", "arch_lm",
    # normalidade
    "jarque_bera_test",
    # colinearidade
    "vif", "max_vif",
    # estabilidade / quebras
    "chow_test", "quandt_andrews", "cusum",
    # relatórios
    "residual_report",
]
