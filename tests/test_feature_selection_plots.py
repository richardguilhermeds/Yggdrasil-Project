"""Testes das análises visuais PÓS-seleção (yggdrasil.feature_selection.plots).

Bloco 100% puro (pandas/numpy/matplotlib) — NÃO requer pyspark: os gráficos
operam sobre a ``selection_table`` (pandas) já materializada no driver. Cada teste
fabrica uma tabela com todos os "motivos" do pipeline e exercita o caminho de
desenho de cada figura via ``savefig`` (backend Agg embutido no ``_figure``).
"""

from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from yggdrasil.feature_selection import plots
from yggdrasil.feature_selection.selector import _COLS


def _fake_table(problem: str = "classification") -> pd.DataFrame:
    """selection_table sintética cobrindo todos os motivos/etapas do pipeline."""
    classif = problem == "classification"

    def linha(feature, book, *, sel, motivo, score=np.nan, rf=np.nan, corr=np.nan,
              iv=np.nan, ks=np.nan, auc=np.nan, gini=np.nan, hits=np.nan,
              dec=None, cons=np.nan, cluster=1, rep=True, red=None, leak=False,
              miss=0.05, top1=0.3, semvar=False, nearc=False):
        return {
            "book": book, "feature": feature, "pct_missing": miss, "p_low": np.nan,
            "p_high": np.nan, "top1_share": top1, "sem_variancia": semvar,
            "near_constante": nearc, "rf_importance": rf,
            "iv": iv if classif else np.nan, "ks": ks if classif else np.nan,
            "auc": auc if classif else np.nan, "gini": gini if classif else np.nan,
            "corr_target": corr, "score": score, "leakage_flag": leak,
            "cluster": cluster, "representante": rep, "redundante_com": red,
            "boruta_hits": hits, "boruta_decisao": dec, "score_consenso": cons,
            "selecionada": sel, "motivo": motivo,
        }

    rows = [
        linha("feat_serasa_score", "serasa", sel=True, motivo="selecionada (consenso)",
              score=20, rf=0.30, corr=0.8, iv=0.55, ks=0.5, auc=0.9, gini=0.8,
              hits=11, dec="confirmada", cons=0.9, cluster=1),
        linha("feat_serasa_atraso_max", "serasa", sel=False,
              motivo="redundante c/ feat_serasa_score", score=18, rf=0.25, corr=0.78,
              iv=0.5, ks=0.48, cluster=1, rep=False, red="feat_serasa_score"),
        linha("feat_cadastral_renda", "cadastral", sel=True, motivo="selecionada (consenso)",
              score=17, rf=0.28, corr=0.75, iv=0.5, ks=0.45, auc=0.88, gini=0.76,
              hits=10, dec="confirmada", cons=0.85, cluster=2),
        linha("feat_bvs_miss", "bvs", sel=False, motivo="alto missing", miss=0.82,
              cluster=3),
        linha("feat_serasa_const", "serasa", sel=False, motivo="sem variância",
              semvar=True, top1=1.0, cluster=4),
        linha("feat_cadastral_flag_pep", "cadastral", sel=False, motivo="quase-constante",
              nearc=True, top1=0.995, cluster=5),
        linha("feat_openfinance_vazamento", "openfinance", sel=False,
              motivo="suspeita de leakage (revisar)", score=25, rf=0.2, corr=0.985,
              iv=0.95, ks=0.82, auc=0.985, gini=0.97, hits=12, dec="confirmada",
              leak=True, cluster=6),
        linha("feat_serasa_qt_negativacoes", "serasa", sel=False, motivo="Boruta rejeitada",
              score=3, rf=0.05, corr=0.1, iv=0.03, ks=0.05, hits=1, dec="rejeitada",
              cons=0.2, cluster=7),
        linha("feat_openfinance_qt_vinculos", "openfinance", sel=False,
              motivo="consenso abaixo do limiar", score=3, rf=0.06, corr=0.12, iv=0.04,
              ks=0.06, hits=4, dec="tentativa", cons=0.31, cluster=8),
        linha("feat_transacional_qt_saques", "transacional", sel=True,
              motivo="selecionada (consenso; Boruta rejeitou)", score=6, rf=0.12,
              corr=0.4, iv=0.2, ks=0.2, hits=1, dec="rejeitada", cons=0.58, cluster=9),
        # não-numérica: métricas de modelo ausentes (NaN), entra como representante
        linha("feat_cadastral_uf", "cadastral", sel=True, motivo="selecionada (consenso)",
              cluster=10),
    ]
    return pd.DataFrame(rows).reindex(columns=_COLS)


def _renderiza(fig) -> int:
    """Desenha a figura em PNG (Agg) e devolve o nº de bytes — 0 = falhou a render."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=70, bbox_inches="tight")
    return buf.getbuffer().nbytes


# ── mapeamento de estágio do funil (núcleo puro) ─────────────────────────
@pytest.mark.parametrize("motivo,esperado", [
    ("alto missing", "alto missing"),
    ("sem variância", "variância"),
    ("quase-constante", "variância"),
    ("redundante c/ feat_x", "redundante"),
    ("suspeita de leakage (revisar)", "leakage"),
    ("Boruta rejeitada", "Boruta/consenso"),
    ("consenso abaixo do limiar", "Boruta/consenso"),
    ("selecionada (consenso)", "selecionada"),
    ("selecionada (Boruta confirmada)", "selecionada"),
    ("selecionada (consenso; Boruta rejeitou)", "selecionada"),
])
def test_stage_of_mapeia_motivos(motivo, esperado):
    assert plots._stage_of(motivo) == esperado


# ── render de todos os gráficos ──────────────────────────────────────────
def _todos(tab, problem):
    return {
        "funnel": plots.plot_selection_funnel(tab),
        "decision_map": plots.plot_decision_map(tab, problem),
        "boruta": plots.plot_boruta_significance(tab, n_iter=12),
        "book_power": plots.plot_book_power_contribution(tab),
        "cluster": plots.plot_cluster_redundancy(tab),
        "quadrant": plots.plot_power_quadrant_iv_ks(tab),
        "leakage": plots.plot_leakage_audit(tab, problem),
        "scorecard": plots.plot_survivor_scorecard(tab),
        "dashboard": plots.plot_post_selection_dashboard(tab, problem, n_iter=12),
    }


@pytest.mark.parametrize("problem", ["classification", "regression"])
def test_todos_os_graficos_renderizam(problem):
    tab = _fake_table(problem)
    for nome, fig in _todos(tab, problem).items():
        assert _renderiza(fig) > 0, f"{nome} não renderizou"


def test_quadrante_degrada_em_regressao():
    """Sem IV/KS (regressão) o quadrante retorna figura com mensagem, sem exceção."""
    fig = plots.plot_power_quadrant_iv_ks(_fake_table("regression"))
    assert _renderiza(fig) > 0
    assert not fig.axes[0].collections  # nenhum scatter desenhado


def test_graficos_com_tabela_vazia_nao_quebram():
    vazia = pd.DataFrame(columns=_COLS)
    figs = [
        plots.plot_selection_funnel(vazia),
        plots.plot_decision_map(vazia),
        plots.plot_boruta_significance(vazia, n_iter=10),
        plots.plot_book_power_contribution(vazia),
        plots.plot_cluster_redundancy(vazia),
        plots.plot_power_quadrant_iv_ks(vazia),
        plots.plot_leakage_audit(vazia),
        plots.plot_survivor_scorecard(vazia),
        plots.plot_post_selection_dashboard(vazia),
    ]
    for fig in figs:
        assert _renderiza(fig) > 0


# ── asserções de conteúdo ────────────────────────────────────────────────
def test_funil_conta_sobreviventes():
    tab = _fake_table("classification")
    n_sel = int(tab["selecionada"].fillna(False).sum())
    fig = plots.plot_selection_funnel(tab)
    titulo = fig.axes[0].get_title()
    assert f"{n_sel}/{len(tab)}" in titulo


def test_leakage_audit_marca_flag():
    fig = plots.plot_leakage_audit(_fake_table("classification"))
    assert "1 feature" in fig.axes[0].get_title()


def test_survivor_corr_heatmap_destaca_redundancia_residual():
    from matplotlib.patches import Rectangle

    feats = ["feat_a", "feat_b", "feat_c"]
    m = np.array([[1.0, 0.88, 0.1], [0.88, 1.0, 0.2], [0.1, 0.2, 1.0]])
    corr = pd.DataFrame(m, index=feats, columns=feats)
    fig = plots.plot_survivor_corr_heatmap(corr, corr_high=0.80)
    caixas = [p for p in fig.axes[0].patches if isinstance(p, Rectangle)
              and p.get_fill() is False]
    assert len(caixas) >= 2  # par a-b contornado nos dois triângulos
    assert _renderiza(fig) > 0


def test_cluster_redundancy_sem_cluster_multi_retorna_figura():
    """Tabela só com features isoladas (clusters de 1) → figura com mensagem."""
    tab = _fake_table("classification")
    tab = tab[tab["representante"].fillna(True)]  # remove os redundantes
    tab = tab.assign(cluster=range(1, len(tab) + 1))
    fig = plots.plot_cluster_redundancy(tab)
    assert _renderiza(fig) > 0


def test_redundancy_groups_separa_books_com_mesmo_cluster_id():
    """Os ids de cluster reiniciam em 1 por book — o agrupamento é por (book, cluster),
    então dois books com cluster=1 NÃO podem virar um cluster visual único."""
    d = pd.DataFrame([
        {"book": "A", "feature": "feat_A_rep", "cluster": 1, "representante": True,
         "redundante_com": None, "score": 9},
        {"book": "A", "feature": "feat_A_red", "cluster": 1, "representante": False,
         "redundante_com": "feat_A_rep", "score": 8},
        {"book": "B", "feature": "feat_B_rep", "cluster": 1, "representante": True,
         "redundante_com": None, "score": 7},
        {"book": "B", "feature": "feat_B_red", "cluster": 1, "representante": False,
         "redundante_com": "feat_B_rep", "score": 6},
    ])
    grupos = plots._redundancy_groups(d, "score", 2, 12)
    assert len(grupos) == 2                                   # não mistura os books
    for _rotulo, g in grupos:
        assert int(g["representante"].sum()) == 1            # um representante por cluster real
    assert _renderiza(plots.plot_cluster_redundancy(d)) > 0


def test_graficos_none_nao_quebram():
    """Chamada direta com selection_table=None não deve levantar (figsize None-safe)."""
    figs = [
        plots.plot_boruta_significance(None, n_iter=10),
        plots.plot_leakage_audit(None),
        plots.plot_survivor_scorecard(None),
        plots.plot_selection_funnel(None),
    ]
    for fig in figs:
        assert _renderiza(fig) > 0
