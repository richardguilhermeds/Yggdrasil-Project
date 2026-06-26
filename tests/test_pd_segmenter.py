"""
Testes do subpacote yggdrasil.credit_risk.pd (segmentação de PD — classificação).

Cobrem: import do pacote, construção da árvore (manual e automática), faltantes
em bin própria, predict, geração de régua PySpark, robustez do merge, métricas de
discriminação (KS/AUC/Gini/Acurácia/F1) e gráficos (ROC, KS, taxa de default).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.pd import SequentialPDSegmenter


def _amostra(n=4000, seed=0, com_na=False):
    """Carteira DES com alvo binário (default) dirigido por `score` e `garantia`."""
    rng = np.random.default_rng(seed)
    score = rng.beta(2.5, 3, n) * 1.4 + 0.3
    gar = rng.choice(["A", "B", "C", "D"], n, p=[0.5, 0.22, 0.18, 0.1]).astype(object)
    if com_na:
        score[rng.random(n) < 0.08] = np.nan
        gar[rng.random(n) < 0.06] = np.nan
    lg = {"A": 0.0, "B": 0.08, "C": 0.14, "D": 0.30}
    p = (0.05 + 0.35 * np.nan_to_num(score - 0.5, nan=0.35)
         + np.array([lg.get(g, 0.2) for g in gar]))
    p = np.clip(p, 0.01, 0.95)
    target = (rng.uniform(0, 1, n) < p).astype(int)
    return pd.DataFrame({"score": score, "garantia": gar,
                         "target": target, "amostra": "DES"})


@pytest.fixture
def seg():
    df = _amostra()
    return SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                 ref_sample="DES")


def test_import_pacote():
    import yggdrasil
    from yggdrasil.credit_risk import SequentialPDSegmenter as C
    assert C is SequentialPDSegmenter
    assert isinstance(yggdrasil.__version__, str)


def test_grow_numerico(seg):
    seg.grow("score", splits=[1.0])
    folhas = seg.leaves()
    assert len(folhas) >= 2
    cob = sum(s["mask"].sum() for s in seg.segments.values() if s["is_leaf"])
    assert cob == len(seg.df)


def test_fit_auto_e_predict(seg):
    seg.fit_auto(max_depth=2, verbose=False)
    assert sum(s["is_leaf"] for s in seg.segments.values()) >= 2
    pred = seg.predict(_amostra(n=500, seed=9))
    assert {"segmento_pd", "nota_pd", "pd_regua"}.issubset(pred.columns)
    assert pred["pd_regua"].notna().all()


def test_metrics_classificacao(seg):
    """A régua é avaliada como modelo de PD: KS/AUC/Gini/Acurácia/F1 por amostra."""
    seg.fit_auto(max_depth=3, verbose=False)
    m = seg.metrics()
    assert {"amostra", "n", "taxa_default", "KS", "AUC", "Gini",
            "Acuracia", "F1"}.issubset(m.columns)
    row = m[m["amostra"] == "DES"].iloc[0]
    # discriminação SUBSTANTIVA acima do acaso: pisos que uma régua constante
    # (KS=0, AUC=0.5) ou invertida (defaults nos scores baixos) NÃO passam.
    # Valores observados nesta fixture: KS≈0.27, AUC≈0.68.
    assert row["AUC"] > 0.6
    assert row["KS"] > 0.15
    assert abs(row["Gini"] - (2 * row["AUC"] - 1)) < 1e-6   # identidade Gini=2·AUC−1
    assert 0.0 <= row["F1"] <= 1.0


def test_fit_auto_concentracao_global():
    """min_leaf_repr/max_bin_repr são REPRESENTATIVIDADE GLOBAL (% da carteira)."""
    df = _amostra(n=6000, seed=1)
    N = len(df)

    def reprs(seg):
        return [s["mask"].sum() / N for s in seg.segments.values() if s["is_leaf"]]

    s_min = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                  ref_sample="DES", verbose=False)
    s_min.fit_auto(max_depth=5, min_iv=0.0, min_leaf_repr=0.10, verbose=False)
    assert min(reprs(s_min)) >= 0.07            # folha terminal respeita o piso global

    # concentração MÁXIMA global (variável contínua divisível)
    rng = np.random.default_rng(7)
    nc = 6000
    x = rng.normal(0, 1, nc)
    p = np.clip(0.15 + 0.12 * x, 0.01, 0.95)
    dfc = pd.DataFrame({"x": x, "amostra": "DES",
                        "target": (rng.uniform(0, 1, nc) < p).astype(int)})
    s_max = SequentialPDSegmenter(dfc, target="target", sample_col="amostra",
                                  ref_sample="DES", verbose=False)
    s_max.fit_auto(max_depth=4, min_iv=0.0, min_leaf_repr=0.04,
                   max_bin_repr=0.25, verbose=False)
    rs = [s["mask"].sum() / nc for s in s_max.segments.values() if s["is_leaf"]]
    assert len(rs) > 1                          # a restrição não impede a divisão
    assert max(rs) <= 0.25 + 0.03               # respeita o teto de concentração


def test_faltantes_viram_bin_propria():
    df = _amostra(com_na=True)
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES")
    seg.grow("score", splits=[1.0])
    cob = sum(s["mask"].sum() for s in seg.segments.values() if s["is_leaf"])
    assert cob == len(df)  # nada é descartado
    desc = [seg._descrever(s["conditions"]) for s in seg.segments.values()
            if s["is_leaf"]]
    assert any("faltante" in d for d in desc)
    # predict roteia NaN para a folha de faltantes
    pred = seg.predict(pd.DataFrame({"score": [np.nan], "garantia": ["A"]}))
    assert pred["segmento_pd"].notna().all()


def test_to_pyspark_compila(seg):
    import ast
    seg.grow("score", splits=[1.0])
    code = seg.to_pyspark()
    ast.parse(code)  # não levanta SyntaxError
    assert "def aplicar_regua_pd" in code


def test_to_pyspark_faltante_usa_isnull():
    df = _amostra(com_na=True)
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES")
    seg.grow("score", splits=[1.0])
    assert "isNull()" in seg.to_pyspark()


def test_merge_nao_funde_no_de_faltante():
    df = _amostra(com_na=True)
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES")
    seg.grow("garantia")
    na = [s for s, v in seg.segments.items()
          if v["is_leaf"] and v["conditions"][-1]["kind"] == "na"]
    if na:  # nó de faltante não deve se fundir; a árvore segue íntegra
        seg.merge_leaf(na[0], side="left", verbose=False)
        cob = sum(s["mask"].sum() for s in seg.segments.values() if s["is_leaf"])
        assert cob == len(df)


# ----------------------------------------------------------------------
# CSI por variável, persistência (JSON), auto-merge e UI (undo/redo)
# ----------------------------------------------------------------------
def _amostra_shift(n=6000, seed=5):
    """DES + OOT com SHIFT de população no score (garantia estável)."""
    rng = np.random.default_rng(seed)
    amostra = rng.choice(["DES", "OOT"], n, p=[0.7, 0.3]).astype(object)
    score = rng.uniform(0.3, 1.0, n)
    score[amostra == "OOT"] += 0.25      # score migra no OOT
    gar = rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]).astype(object)
    lg = {"A": 0.05, "B": 0.15, "C": 0.30}
    p = np.clip(0.05 + 0.35 * (score - 0.5) + np.array([lg[g] for g in gar]), 0.01, 0.95)
    target = (rng.uniform(0, 1, n) < p).astype(int)
    return pd.DataFrame({"score": score, "garantia": gar,
                         "target": target, "amostra": amostra})


def _dois_patamares(n=8000, seed=3):
    """score com DOIS patamares de PD: bins-irmãs vizinhos são indistinguíveis."""
    rng = np.random.default_rng(seed)
    score = rng.uniform(0, 1, n)
    p = np.where(score < 0.5, 0.15, 0.55)
    target = (rng.uniform(0, 1, n) < p).astype(int)
    return pd.DataFrame({"score": score, "target": target, "amostra": "DES"})


def test_csi_por_variavel():
    df = _amostra_shift()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    csi = seg.csi()
    assert {"variavel", "tipo", "n_bins", "pior_csi", "classificacao"}.issubset(csi.columns)
    assert "csi_OOT" in csi.columns
    by = csi.set_index("variavel")
    # score migrou (CSI alto / instável); garantia é estável (CSI baixo)
    assert by.loc["score", "csi_OOT"] > 0.25
    assert by.loc["score", "csi_OOT"] > by.loc["garantia", "csi_OOT"]
    assert not seg.csi_detalhe().empty


def test_csi_requer_sample_col():
    df = _amostra_shift().drop(columns=["amostra"])
    seg = SequentialPDSegmenter(df, target="target", verbose=False)
    with pytest.raises(ValueError):
        seg.csi()


def test_variable_iv_binario_e_psi_mesmos_bins():
    """O IV é o do optbinning para alvo BINÁRIO (WoE: Σ(dist_bons−dist_maus)·WoE),
    e o PSI por variável é calculado sobre os MESMOS bins do IV (DES × OOT)."""
    import contextlib
    import io

    from optbinning import OptimalBinning

    df = _amostra_shift()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    iv = seg.variable_iv("root")
    # PSI vem junto do IV (mesmos bins), não do csi() global
    assert {"psi_OOT", "pior_psi", "psi_classificacao"}.issubset(iv.columns)
    by = iv.set_index("variavel")

    # IV(score) == IV binário que o optbinning calcula (mesmos bins, na DES)
    des = df[df["amostra"] == "DES"]
    cob = OptimalBinning(name="score", dtype="numerical", max_n_bins=5,
                         min_bin_size=0.05, monotonic_trend="auto_asc_desc")
    with contextlib.redirect_stderr(io.StringIO()):
        cob.fit(des["score"].to_numpy(), des["target"].to_numpy())
        cob.binning_table.build()
        cob.binning_table.analysis(print_output=False)
    assert abs(float(by.loc["score", "iv"]) - cob.binning_table.iv) < 0.03

    # o PSI (nos bins do IV) capta o shift: score migrou ≫ garantia estável
    assert by.loc["score", "pior_psi"] > by.loc["garantia", "pior_psi"]

    # with_psi=False (usado por fit_auto/suggest_split) não traz colunas de PSI
    assert "pior_psi" not in seg.variable_iv("root", with_psi=False).columns


def test_save_load_roundtrip(tmp_path):
    df = _amostra()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.fit_auto(max_depth=2, verbose=False)
    p = tmp_path / "arvore.json"
    seg.save(str(p))
    assert '"yggdrasil.credit_risk.pd.tree' in p.read_text(encoding="utf-8")
    seg2 = SequentialPDSegmenter.load(str(p), df, verbose=False)
    cols = ["nota_pd", "descricao", "pd_medio"]
    assert seg.leaves()[cols].equals(seg2.leaves()[cols])
    p1 = seg.predict(df.head(300))["pd_regua"].fillna(-1).to_numpy()
    p2 = seg2.predict(df.head(300))["pd_regua"].fillna(-1).to_numpy()
    assert np.allclose(p1, p2)
    cob = sum(s["mask"].sum() for s in seg2.segments.values() if s["is_leaf"])
    assert cob == len(df)


def test_auto_merge_funde_irmas_indistinguiveis():
    df = _dois_patamares()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.grow("score", splits=[0.2, 0.35, 0.5, 0.7, 0.85])
    n0 = sum(s["is_leaf"] for s in seg.segments.values())
    seg.auto_merge(alpha=0.05, verbose=False)
    n1 = sum(s["is_leaf"] for s in seg.segments.values())
    assert n0 == 6 and n1 == 2                       # recupera os 2 patamares
    cob = sum(s["mask"].sum() for s in seg.segments.values() if s["is_leaf"])
    assert cob == len(df)


def test_auto_merge_respeita_protect():
    df = _dois_patamares()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.grow("score", splits=[0.2, 0.35, 0.5, 0.7, 0.85])
    folhas = [s for s, v in seg.segments.items() if v["is_leaf"]]
    prot = {folhas[0], folhas[1]}
    seg.auto_merge(alpha=0.05, protect=prot, verbose=False)
    assert prot.issubset(set(seg.segments))          # travadas preservadas


def test_ui_undo_redo_automerge_e_json(tmp_path):
    pytest.importorskip("ipywidgets")
    import contextlib
    import io
    from yggdrasil.credit_risk.pd import PDSegmenterUI

    df = _dois_patamares(n=5000, seed=7)
    with contextlib.redirect_stdout(io.StringIO()):
        ui = PDSegmenterUI(df, target="target", sample_col="amostra", ref_sample="DES")
        nleaf = lambda: sum(s["is_leaf"] for s in ui.seg.segments.values())
        iv_root = ui.seg.variable_iv("root")
        assert "iv" in iv_root.columns and not iv_root.empty
        ui.dd_leaf.value = "root"
        ui.dd_feature.value = "score"
        ui.tg_mode.value = "Manual"
        ui.tx_cuts.value = "0.2,0.35,0.5,0.7,0.85"
        ui._on_preview(None)
        ui._on_split(None)
        n_split = nleaf()
        ui._on_undo(None); n_undo = nleaf()
        ui._on_redo(None); n_redo = nleaf()
        ui.sl_alpha.value = 0.05
        ui._on_automerge(None); n_am = nleaf()
        p = str(tmp_path / "ui_arvore.json")
        ui.tx_json_path.value = p
        ui._on_save_json(None); n_saved = nleaf()
        ui._on_reset(None); n_reset = nleaf()
        ui._on_load_json(None); n_loaded = nleaf()
    assert n_split == 6 and n_undo == 1 and n_redo == 6
    assert n_am == 2                                  # auto-merge volta aos 2 patamares
    assert n_reset == 1 and n_loaded == n_saved


# ----------------------------------------------------------------------
# Juntar nó de faltantes (na) com nó populado da variável (include_na)
# ----------------------------------------------------------------------
def _regua_bate_mascaras(seg):
    """A régua (predict) reproduz exatamente as máscaras em memória das folhas?"""
    pred = seg.predict(seg.df)["segmento_pd"]
    for sid, v in seg.segments.items():
        if v["is_leaf"] and not (((pred == sid).values) == v["mask"].values).all():
            return False
    return bool(pred.notna().all())   # cobertura total (faltantes roteados)


def test_merge_missing_numerico():
    df = _amostra(com_na=True)
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.grow("score", splits=[1.0])
    nums = [s for s, v in seg.segments.items()
            if v["is_leaf"] and v["conditions"][-1]["kind"] == "num"]
    assert any(v["is_leaf"] and v["conditions"][-1]["kind"] == "na"
               for v in seg.segments.values())          # há nó de faltantes
    seg.merge_missing(nums[-1])
    assert not any(v["is_leaf"] and v["conditions"][-1]["kind"] == "na"
                   for v in seg.segments.values())
    merged = [s for s, v in seg.segments.items()
              if v["is_leaf"] and v["conditions"][-1].get("include_na")]
    assert merged
    assert "faltante" in seg._descrever(seg.segments[merged[0]]["conditions"])
    assert _regua_bate_mascaras(seg)


def test_merge_missing_pyspark_e_save_load(tmp_path):
    import ast
    df = _amostra(com_na=True)
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.grow("score", splits=[1.0])
    nums = [s for s, v in seg.segments.items()
            if v["is_leaf"] and v["conditions"][-1]["kind"] == "num"]
    seg.merge_missing(nums[-1])
    code = seg.to_pyspark()
    ast.parse(code)
    assert "isNull()" in code
    p = tmp_path / "t.json"
    seg.save(str(p))
    assert '"include_na": true' in p.read_text(encoding="utf-8")
    seg2 = SequentialPDSegmenter.load(str(p), df, verbose=False)
    a = seg.predict(df.head(400))["pd_regua"].fillna(-1).to_numpy()
    b = seg2.predict(df.head(400))["pd_regua"].fillna(-1).to_numpy()
    assert np.allclose(a, b)
    assert _regua_bate_mascaras(seg2)


def test_merge_missing_exige_no_populado():
    df = _amostra(com_na=True)
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.grow("score", splits=[1.0])
    na = [s for s, v in seg.segments.items()
          if v["is_leaf"] and v["conditions"][-1]["kind"] == "na"]
    n0 = sum(s["is_leaf"] for s in seg.segments.values())
    seg.merge_missing(na[0], verbose=False)     # apontar o nó de faltantes não muda nada
    assert sum(s["is_leaf"] for s in seg.segments.values()) == n0


def test_auto_merge_include_missing_funde_na():
    # nó de faltantes com MESMA PD de um bin populado → include_missing junta
    rng = np.random.default_rng(11)
    n = 8000
    score = rng.uniform(0, 1, n)
    miss = rng.random(n) < 0.15
    score[miss] = np.nan
    p = np.where(score < 0.5, 0.15, 0.55)
    p[miss] = 0.55                                  # faltante ~ bin alto
    target = (rng.uniform(0, 1, n) < p).astype(int)
    df = pd.DataFrame({"score": score, "target": target, "amostra": "DES"})
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.grow("score", splits=[0.5])
    assert any(v["is_leaf"] and v["conditions"][-1]["kind"] == "na"
               for v in seg.segments.values())
    seg.auto_merge(alpha=0.05, include_missing=True, verbose=False)
    assert not any(v["is_leaf"] and v["conditions"][-1]["kind"] == "na"
                   for v in seg.segments.values())       # faltante foi juntado
    assert _regua_bate_mascaras(seg)


def test_binning_sem_runtimewarning():
    """CSI/IV/fit_auto não devem vazar o RuntimeWarning de divisão por zero do
    optbinning (auto_monotonic), mesmo com feature constante e faltantes."""
    df = _amostra(com_na=True)
    df["amostra"] = np.where(np.arange(len(df)) % 10 < 7, "DES", "OOT")
    df["feat_const"] = 1.0
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            seg.csi()
            seg.variable_iv("root")
            seg.fit_auto(max_depth=2, verbose=False)


def test_ui_merge_missing_e_layout(tmp_path):
    pytest.importorskip("ipywidgets")
    import contextlib
    import io

    import ipywidgets as W

    from yggdrasil.credit_risk.pd import PDSegmenterUI

    df = _amostra(com_na=True)
    df["amostra"] = np.where(np.arange(len(df)) % 10 < 7, "DES", "OOT")
    with contextlib.redirect_stdout(io.StringIO()):
        ui = PDSegmenterUI(df, target="target", sample_col="amostra", ref_sample="DES")
        ui.seg.grow("score", splits=[1.0])
        ui._refresh()
        alvo = [s for s, v in ui.seg.segments.items()
                if v["is_leaf"] and v["conditions"][-1]["kind"] == "num"][-1]
        ui.dd_leaf.value = alvo
        n_na0 = sum(1 for s, v in ui.seg.segments.items()
                    if v["is_leaf"] and v["conditions"][-1]["kind"] == "na")
        ui._on_merge_missing(None)
        n_na1 = sum(1 for s, v in ui.seg.segments.items()
                    if v["is_leaf"] and v["conditions"][-1]["kind"] == "na")
    assert n_na0 == 1 and n_na1 == 0          # botão juntou o nó de faltantes

    # layout em ABAS (workbench): banner · faixa de KPIs · Tab(5 abas) · console
    ch = list(ui.panel.children)
    tabs = next(c for c in ch if isinstance(c, W.Tab))
    titulos = [tabs.get_title(i) for i in range(len(tabs.children))]
    assert titulos == ["① Construir", "② Análise de variáveis", "③ Diagnóstico",
                       "④ Validar & Exportar", "⑤ Histórico"]

    def _all(w):                       # achata a subárvore de widgets
        acc = [w]
        for c in getattr(w, "children", ()) or ():
            acc.extend(_all(c))
        return acc

    def _titles_in(w):                 # cabeçalhos (1º HTML de cada VBox) na subárvore
        return [x.children[0].value for x in _all(w)
                if isinstance(x, W.VBox) and x.children
                and isinstance(x.children[0], W.HTML)]

    # ① Construir = "Cockpit em T": topo = Árvore & quebras AO LADO do IV; detalhe
    # em 3 colunas (folha · dividir · ações+auto-fit); preview separado; placar no
    # Diagnóstico.
    construir = tabs.children[0]
    A = _all(construir)
    top_cols = next(h for h in A if isinstance(h, W.HBox)
                    and ui.out_tree in _all(h) and ui.out_iv in _all(h))
    tree_card = next(c for c in top_cols.children if ui.out_tree in _all(c))
    iv_card = next(c for c in top_cols.children if ui.out_iv in _all(c))
    assert tree_card.layout.width and iv_card.layout.width   # lado a lado
    det_row = next(h for h in A if isinstance(h, W.HBox) and len(h.children) == 3
                   and ui.leaf_header in _all(h))
    c1, c2, c3 = det_row.children
    assert ui.leaf_header in _all(c1)               # folha (detalhe)
    assert ui.dd_feature in _all(c2)                # dividir a folha
    assert ui.out_preview_seg in _all(c2)           # segmentação proposta dentro de "Dividir"
    assert ui.btn_lock in _all(c3) and ui.btn_merge_l in _all(c3)   # ações
    assert ui.sl_depth in _all(c3)                  # auto-fit embaixo das ações
    assert ui.leaf_chips in A                        # régua da folha ativa
    # distribuição+cortes (preview) ao lado do histograma da PD, num mesmo HBox
    assert any(isinstance(h, W.HBox) and len(h.children) == 2
               and ui.out_preview_chart in _all(h) and ui.out_leaf_hist in _all(h)
               for h in A)

    assert ui.out_var_dist in _all(tabs.children[1])

    # ③ Diagnóstico: placar de saúde (out_diag) + folhas; IV vive no Construir
    diag = tabs.children[2]
    assert ui.out_diag in _all(diag) and ui.btn_diag in _all(diag)
    assert any("Folhas criadas" in t for t in _titles_in(diag))
    assert ui.out_iv not in _all(diag)

    assert any(ui.out_log in _all(c) for c in ch if c is not tabs)


# ----------------------------------------------------------------------
# Visualização gráfica da árvore (imagem matplotlib)
# ----------------------------------------------------------------------
def test_plot_tree_gera_imagem(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    df = _amostra(com_na=True)
    df["amostra"] = np.where(np.arange(len(df)) % 10 < 7, "DES", "OOT")
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.grow("score", splits=[1.0])

    fig = seg.plot_tree()
    boxes = [p for p in fig.axes[0].patches if isinstance(p, FancyBboxPatch)]
    assert len(boxes) == len(seg.segments)              # uma caixa por segmento
    txt = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "repr." in txt and "PD" in txt and "folha" in txt
    assert "n=" not in txt                              # n removido (só repr. e PD)
    import re
    ordem = [n for _, n in sorted(
        (t.get_position()[0], int(re.search(r"folha (\d+)", t.get_text()).group(1)))
        for t in fig.axes[0].texts if "folha" in t.get_text())]
    assert ordem == sorted(ordem)
    plt.close(fig)

    p = tmp_path / "arvore.png"
    seg.plot_tree(save_path=str(p))
    assert p.exists() and p.stat().st_size > 0
    plt.close("all")


def test_plot_tree_raiz_unica_nao_quebra():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _amostra()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    fig = seg.plot_tree()                                # só a raiz, sem split
    assert fig is not None
    plt.close(fig)


def test_ui_plot_tree(tmp_path):
    pytest.importorskip("ipywidgets")
    import matplotlib
    matplotlib.use("Agg")
    import contextlib
    import io
    import os

    from yggdrasil.credit_risk.pd import PDSegmenterUI

    df = _dois_patamares(n=4000, seed=4)
    p = str(tmp_path / "ui_tree.png")
    with contextlib.redirect_stdout(io.StringIO()):
        ui = PDSegmenterUI(df, target="target", sample_col="amostra", ref_sample="DES")
        ui.seg.grow("score", splits=[0.5])
        ui._refresh()
        ui.tx_img_path.value = p
        ui._on_plot(None)
        valor_apos_plot = ui.out_plot.value
        ui._on_plot_hide(None)
        valor_apos_hide = ui.out_plot.value
    assert os.path.exists(p) and os.path.getsize(p) > 0
    assert "<img" in valor_apos_plot
    assert valor_apos_hide == ""


# ----------------------------------------------------------------------
# Discriminação: curva ROC e curva KS
# ----------------------------------------------------------------------
def test_plot_roc_e_ks():
    import re

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _amostra_shift()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.fit_auto(max_depth=3, verbose=False)
    m = seg.metrics().set_index("amostra")

    # ROC: a AUC desenhada (legenda) deve bater com metrics() e ficar acima do acaso
    fig = seg.plot_roc()
    assert len(fig.axes[0].lines) >= 2          # 1 curva por amostra + diagonal
    leg = " ".join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
    auc_des = float(re.search(r"DES · AUC ([\d.]+)", leg).group(1))
    assert abs(auc_des - m.loc["DES", "AUC"]) < 1e-2
    assert auc_des > 0.6                          # discrimina, não é a diagonal
    plt.close(fig)

    # KS: o valor anotado deve bater com metrics() e ser substantivo (> acaso)
    figk = seg.plot_ks()
    txt = " ".join(t.get_text() for t in figk.axes[0].texts)
    ks_val = float(re.search(r"KS = ([\d.]+)", txt).group(1))
    assert abs(ks_val - m.loc["DES", "KS"]) < 1e-2
    assert ks_val > 0.15
    plt.close(figk)


# ----------------------------------------------------------------------
# Aplicar a régua numa tabela Spark (apply_spark / "reconstruir as folhas")
# ----------------------------------------------------------------------
def test_regua_features():
    df = _amostra(com_na=True)
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.grow("score", splits=[1.0])
    feats = seg.regua_features()
    assert "score" in feats
    assert seg.target not in feats and seg.sample_col not in feats


def test_apply_spark_valida_colunas():
    pytest.importorskip("pyspark")
    df = _amostra(com_na=True)
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.fit_auto(max_depth=2, verbose=False)

    class _FakeDF:
        def __init__(self, cols):
            self.columns = cols

    with pytest.raises(ValueError, match="score"):
        seg.apply_spark(_FakeDF(["target", "amostra"]))


# ----------------------------------------------------------------------
# Validação regulatória: backtest, monotonicidade, calibração e relatório
# ----------------------------------------------------------------------
def _amostra_safra(n=8000, seed=1):
    """DES/OOT por safra (dt_ref), com faltantes em score."""
    rng = np.random.default_rng(seed)
    score = rng.uniform(0.3, 1.3, n)
    score[rng.random(n) < 0.10] = np.nan
    gar = rng.choice(["A", "B", "C", "D"], n, p=[.4, .3, .2, .1]).astype(object)
    lg = {"A": .05, "B": .12, "C": .22, "D": .40}
    p = np.clip(0.05 + 0.25 * np.nan_to_num(score - 0.6, nan=.4)
                + np.array([lg[g] for g in gar]), 0.01, 0.95)
    target = (rng.uniform(0, 1, n) < p).astype(int)
    safra = rng.choice(["2023Q1", "2023Q2", "2023Q3", "2023Q4"], n)
    amostra = np.where(np.isin(safra, ["2023Q1", "2023Q2", "2023Q3"]), "DES", "OOT")
    return pd.DataFrame({"score": score, "garantia": gar, "dt_ref": safra,
                         "target": target, "amostra": amostra})


def test_backtest_por_safra():
    df = _amostra_safra()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.fit_auto(max_depth=2, verbose=False)
    bt = seg.backtest("dt_ref")
    assert {"periodo", "n", "pd_prevista", "pd_realizada", "gap", "status"}.issubset(bt.columns)
    assert int(bt["n"].sum()) == len(df)
    assert set(bt["status"]) <= {"ok", "alerta"}
    with pytest.raises(ValueError):
        seg.backtest("coluna_inexistente")


def test_monotonicity_report_detecta_inversao():
    rng = np.random.default_rng(4)
    n = 6000
    x = rng.uniform(0, 1, n)
    amostra = rng.choice(["DES", "OOT"], n, p=[.6, .4])
    p = np.where(x < 0.5, 0.15, 0.55)
    oot = amostra == "OOT"
    p = np.where(oot & (x < 0.5), 0.6, p)        # inverte os níveis no OOT
    p = np.where(oot & (x >= 0.5), 0.2, p)
    target = (rng.uniform(0, 1, n) < p).astype(int)
    df = pd.DataFrame({"x": x, "target": target, "amostra": amostra})
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.grow("x", splits=[0.5])
    mr = seg.monotonicity_report().set_index("amostra")
    assert bool(mr.loc["DES", "monotonico"]) is True          # DES monotônico por construção
    assert bool(mr.loc["OOT", "monotonico"]) is False         # OOT invertido
    assert int(mr.loc["OOT", "n_inversoes"]) >= 1


def test_calibration_table():
    df = _amostra_safra()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.fit_auto(max_depth=2, verbose=False)
    ct = seg.calibration_table()
    assert {"nota_pd", "n", "pd_prevista", "pd_realizada", "gap"}.issubset(ct.columns)
    assert ct.attrs["check_sample"] == "OOT"
    assert len(ct) == sum(s["is_leaf"] for s in seg.segments.values())


def test_plot_calibration():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.collections import PathCollection

    df = _amostra_safra()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.fit_auto(max_depth=2, verbose=False)
    fig = seg.plot_calibration()
    ax = fig.axes[0]
    # a diagonal y=x (calibração perfeita) existe...
    assert len(ax.lines) >= 1
    # ...e há um ponto por folha com previsto E realizado não-nulos
    n_pts = sum(len(c.get_offsets()) for c in ax.collections
                if isinstance(c, PathCollection))
    ct = seg.calibration_table()
    esperado = int(((ct["pd_prevista"].notna()) & (ct["pd_realizada"].notna())).sum())
    assert n_pts == esperado and n_pts >= 1
    plt.close(fig)


def test_validation_report(tmp_path):
    import os

    import matplotlib
    matplotlib.use("Agg")

    df = _amostra_safra()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.fit_auto(max_depth=2, verbose=False)
    p = tmp_path / "rel.md"
    out = seg.validation_report(str(p), time_col="dt_ref", stamp="2026-06-25")
    assert os.path.exists(out)
    txt = p.read_text(encoding="utf-8")
    for sec in ["## Visão geral", "## Folhas", "## Monotonicidade",
                "## Discriminação", "## Calibração", "## Backtest"]:
        assert sec in txt
    assert (tmp_path / "rel_arvore.png").exists()
    assert (tmp_path / "rel_calibracao.png").exists()


def test_p_vs_prox_apenas_entre_irmas():
    """O teste de hipótese (p_vs_prox) só compara folhas-IRMÃS (mesmo pai)."""
    rng = np.random.default_rng(1)
    n = 12000
    score = rng.uniform(0.3, 1.3, n)
    gar = rng.choice(["A", "B", "C", "D"], n, p=[.4, .3, .2, .1]).astype(object)
    lg = {"A": .05, "B": .15, "C": .25, "D": .45}
    p = np.clip(0.05 + 0.25 * (score - 0.6) + np.array([lg[g] for g in gar]), 0.01, 0.95)
    target = (rng.uniform(0, 1, n) < p).astype(int)
    df = pd.DataFrame({"score": score, "garantia": gar,
                       "target": target, "amostra": "DES"})
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.grow("garantia", splits=[["A", "B", "C"], ["D"]])
    abc = [s for s, v in seg.segments.items()
           if v["is_leaf"] and v["conditions"][-1]["kind"] == "cat"
           and set(v["conditions"][-1]["cats"]) == {"A", "B", "C"}]
    seg.grow("score", splits=[0.6, 0.9], only_segments=abc)

    lv = seg.leaves(with_test=True).set_index("segmento")
    pai = lambda s: seg.segments[s]["parent"]
    d_leaf = [s for s in lv.index
              if seg.segments[s]["conditions"][-1].get("cats") == ["D"]][0]
    assert pd.isna(lv.loc[d_leaf, "p_vs_prox"])
    for sid, r in lv.iterrows():
        if not pd.isna(r["p_vs_prox"]):
            irmas = [s for s in lv.index if pai(s) == pai(sid)
                     and seg.segments[s]["conditions"][-1]["kind"] != "na"]
            assert len(irmas) >= 2
    assert int(lv["p_vs_prox"].notna().sum()) == 2     # 2 pares de irmãs em {A,B,C}


def test_prune_funde_irmas_por_pd_e_repr():
    """Poda funde irmãs com ΔPD < min_pd_gap ou repr% < min_repr."""
    rng = np.random.default_rng(2)
    n = 12000
    score = rng.uniform(0, 1, n)

    def base(x):
        if x < 0.3:
            return 0.30
        if x < 0.6:
            return 0.32          # ΔPD ~0.02 vs faixa A -> deve fundir
        if x < 0.62:
            return 0.50          # repr ~2% -> imaterial, deve fundir
        return 0.65

    p = np.array([base(x) for x in score])
    target = (rng.uniform(0, 1, n) < p).astype(int)
    df = pd.DataFrame({"score": score, "target": target, "amostra": "DES"})
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.grow("score", splits=[0.3, 0.6, 0.62])             # 4 folhas-irmãs
    assert sum(s["is_leaf"] for s in seg.segments.values()) == 4

    seg.prune(min_repr=3.0, min_pd_gap=0.03, verbose=False)
    cob = sum(s["mask"].sum() for s in seg.segments.values() if s["is_leaf"])
    assert cob == len(df)                                # cobertura preservada

    n_total = len(df)
    fp = {}
    for sid, s in seg.segments.items():
        if s["is_leaf"]:
            fp.setdefault(s["parent"], []).append(sid)
    for pai, fs in fp.items():
        if pai is None:
            continue
        comp = [c for c in fs if seg.segments[c]["conditions"][-1]["kind"] != "na"]
        comp.sort(key=lambda c: seg.segments[c]["conditions"][-1].get("lo", 0))
        reprs = [100 * seg.segments[c]["mask"].sum() / n_total for c in comp]
        pds = [seg.df[seg.segments[c]["mask"]][seg.target].mean() for c in comp]
        for i in range(len(comp) - 1):
            assert abs(pds[i + 1] - pds[i]) >= 0.03
            assert reprs[i] >= 3.0 and reprs[i + 1] >= 3.0


# ----------------------------------------------------------------------
# Qualidade dos segmentos: taxa de default por folha, score, preview da variável
# ----------------------------------------------------------------------
def test_plot_leaf_badrate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _amostra_safra()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.fit_auto(max_depth=2, verbose=False)
    fig = seg.plot_leaf_badrate()
    n_folhas = sum(s["is_leaf"] for s in seg.segments.values())
    assert len(fig.axes[0].get_xticklabels()) == n_folhas
    plt.close(fig)


def test_plot_score_distribution():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.container as mcont
    import matplotlib.pyplot as plt

    df = _amostra_safra()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.fit_auto(max_depth=2, verbose=False)
    fig = seg.plot_score_distribution()
    # dois histogramas: um de bons (alvo 0) e um de maus (alvo 1)
    bars = [c for c in fig.axes[0].containers if isinstance(c, mcont.BarContainer)]
    assert len(bars) == 2
    leg = " ".join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
    assert "bons" in leg and "maus" in leg          # ambas as classes presentes
    plt.close(fig)


def test_plot_feature_pd():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _amostra_safra()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    fig = seg.plot_feature_pd("score", sid="root")
    assert len(fig.axes[0].patches) >= 2          # uma barra por faixa
    plt.close(fig)
    seg.grow("garantia")
    folha = [s for s, v in seg.segments.items() if v["is_leaf"]][0]
    fig2 = seg.plot_feature_pd("score", sid=folha)
    assert fig2 is not None
    plt.close(fig2)


# ----------------------------------------------------------------------
# Correções da auditoria de bugs
# ----------------------------------------------------------------------
def test_grow_grupos_cat_repetidos_erro():
    df = _amostra()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    with pytest.raises(ValueError, match="repetida"):
        seg.grow("garantia", splits=[["A", "B"], ["A"], ["C"]])   # 'A' em 2 grupos


def test_grow_nao_cria_split_degenerado():
    df = _amostra()                       # garantia A/B/C/D, sem NaN
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    n0 = sum(s["is_leaf"] for s in seg.segments.values())
    seg.grow("garantia", splits=[["A", "B", "C", "D"]])           # 1 grupo cobre tudo
    n1 = sum(s["is_leaf"] for s in seg.segments.values())
    assert n1 == n0                       # nenhum split degenerado de 1 filho
    assert seg.segments["root"]["is_leaf"]


def test_prune_respeita_protect():
    df = _dois_patamares()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.grow("score", splits=[0.2, 0.35, 0.5, 0.7, 0.85])           # 6 folhas-irmãs
    folhas = [s for s, v in seg.segments.items() if v["is_leaf"]]
    prot = {folhas[0], folhas[1]}
    seg.prune(min_repr=0.0, min_pd_gap=0.03, protect=prot, verbose=False)
    assert prot.issubset(set(seg.segments))                       # travadas preservadas


def test_to_pyspark_categorico_cast_string():
    df = _amostra()
    seg = SequentialPDSegmenter(df, target="target", sample_col="amostra",
                                ref_sample="DES", verbose=False)
    seg.grow("garantia")
    assert 'cast("string").isin' in seg.to_pyspark()              # paridade com pandas


def test_construtor_sample_col_ausente_erro():
    df = _amostra().drop(columns=["amostra"])
    with pytest.raises(ValueError, match="amostra"):
        SequentialPDSegmenter(df, target="target", sample_col="amostra",
                              ref_sample="DES", verbose=False)
