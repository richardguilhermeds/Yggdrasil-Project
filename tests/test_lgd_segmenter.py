"""
Testes do subpacote yggdrasil.credit_risk.lgd.

Cobrem: import do pacote, construção da árvore (manual e automática), faltantes em
bin própria, predict, geração de régua PySpark e robustez do merge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.lgd import SequentialLGDSegmenter


def _amostra(n=4000, seed=0, com_na=False):
    rng = np.random.default_rng(seed)
    ltv = rng.beta(2.5, 3, n) * 1.4 + 0.3
    gar = rng.choice(["A", "B", "C", "D"], n, p=[0.5, 0.22, 0.18, 0.1]).astype(object)
    if com_na:
        ltv[rng.random(n) < 0.08] = np.nan
        gar[rng.random(n) < 0.06] = np.nan
    lg = {"A": 0.0, "B": 0.10, "C": 0.14, "D": 0.28}
    base = (0.1 + 0.4 * np.nan_to_num(ltv - 0.5, nan=0.35)
            + np.array([lg.get(g, 0.2) for g in gar]) + rng.normal(0, 0.07, n))
    return pd.DataFrame({"ltv": ltv, "garantia": gar,
                         "lgd": np.clip(base, 0, 1), "amostra": "DES"})


@pytest.fixture
def seg():
    df = _amostra()
    return SequentialLGDSegmenter(df, target="lgd", sample_col="amostra",
                                  ref_sample="DES")


def test_import_pacote():
    import yggdrasil
    from yggdrasil.credit_risk import SequentialLGDSegmenter as C
    assert C is SequentialLGDSegmenter
    assert isinstance(yggdrasil.__version__, str)


def test_grow_numerico(seg):
    seg.grow("ltv", splits=[0.8])
    folhas = seg.leaves()
    assert len(folhas) >= 2
    cob = sum(s["mask"].sum() for s in seg.segments.values() if s["is_leaf"])
    assert cob == len(seg.df)


def test_fit_auto_e_predict(seg):
    seg.fit_auto(max_depth=2, verbose=False)
    assert sum(s["is_leaf"] for s in seg.segments.values()) >= 2
    pred = seg.predict(_amostra(n=500, seed=9))
    assert {"segmento_lgd", "nota_lgd", "lgd_regua"}.issubset(pred.columns)
    assert pred["lgd_regua"].notna().all()


def test_faltantes_viram_bin_propria():
    df = _amostra(com_na=True)
    seg = SequentialLGDSegmenter(df, target="lgd", sample_col="amostra",
                                 ref_sample="DES")
    seg.grow("ltv", splits=[0.8])
    cob = sum(s["mask"].sum() for s in seg.segments.values() if s["is_leaf"])
    assert cob == len(df)  # nada é descartado
    desc = [seg._descrever(s["conditions"]) for s in seg.segments.values()
            if s["is_leaf"]]
    assert any("faltante" in d for d in desc)
    # predict roteia NaN para a folha de faltantes
    pred = seg.predict(pd.DataFrame({"ltv": [np.nan], "garantia": ["A"]}))
    assert pred["segmento_lgd"].notna().all()


def test_to_pyspark_compila(seg):
    import ast
    seg.grow("ltv", splits=[0.8])
    code = seg.to_pyspark()
    ast.parse(code)  # não levanta SyntaxError
    assert "def aplicar_regua_lgd" in code


def test_to_pyspark_faltante_usa_isnull():
    df = _amostra(com_na=True)
    seg = SequentialLGDSegmenter(df, target="lgd", sample_col="amostra",
                                 ref_sample="DES")
    seg.grow("ltv", splits=[0.8])
    assert "isNull()" in seg.to_pyspark()


def test_merge_nao_funde_no_de_faltante():
    df = _amostra(com_na=True)
    seg = SequentialLGDSegmenter(df, target="lgd", sample_col="amostra",
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
def _amostra_shift(n=5000, seed=5):
    """DES + OOT com SHIFT de população no ltv (garantia estável)."""
    rng = np.random.default_rng(seed)
    amostra = rng.choice(["DES", "OOT"], n, p=[0.7, 0.3]).astype(object)
    ltv = rng.uniform(0.3, 1.0, n)
    ltv[amostra == "OOT"] += 0.25      # ltv migra no OOT
    gar = rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]).astype(object)
    lg = {"A": 0.05, "B": 0.15, "C": 0.25}
    base = (0.1 + 0.4 * (ltv - 0.5)
            + np.array([lg[g] for g in gar]) + rng.normal(0, 0.06, n))
    return pd.DataFrame({"ltv": ltv, "garantia": gar,
                         "lgd": np.clip(base, 0, 1), "amostra": amostra})


def _dois_patamares(n=6000, seed=3):
    """ltv com DOIS patamares de LGD: bins-irmãs vizinhos são indistinguíveis."""
    rng = np.random.default_rng(seed)
    ltv = rng.uniform(0, 1, n)
    lgd = np.where(ltv < 0.5, 0.2, 0.6) + rng.normal(0, 0.05, n)
    return pd.DataFrame({"ltv": ltv, "lgd": np.clip(lgd, 0, 1), "amostra": "DES"})


def test_csi_por_variavel():
    df = _amostra_shift()
    seg = SequentialLGDSegmenter(df, target="lgd", sample_col="amostra",
                                 ref_sample="DES", verbose=False)
    csi = seg.csi()
    assert {"variavel", "tipo", "n_bins", "pior_csi", "classificacao"}.issubset(csi.columns)
    assert "csi_OOT" in csi.columns
    by = csi.set_index("variavel")
    # ltv migrou (CSI alto / instável); garantia é estável (CSI baixo)
    assert by.loc["ltv", "csi_OOT"] > 0.25
    assert by.loc["ltv", "csi_OOT"] > by.loc["garantia", "csi_OOT"]
    assert not seg.csi_detalhe().empty


def test_csi_requer_sample_col():
    df = _amostra_shift().drop(columns=["amostra"])
    seg = SequentialLGDSegmenter(df, target="lgd", verbose=False)
    with pytest.raises(ValueError):
        seg.csi()


def test_save_load_roundtrip(tmp_path):
    df = _amostra()
    seg = SequentialLGDSegmenter(df, target="lgd", sample_col="amostra",
                                 ref_sample="DES", verbose=False)
    seg.fit_auto(max_depth=2, verbose=False)
    p = tmp_path / "arvore.json"
    seg.save(str(p))
    seg2 = SequentialLGDSegmenter.load(str(p), df, verbose=False)
    # folhas e régua idênticas após recarregar
    cols = ["nota_lgd", "descricao", "lgd_medio"]
    assert seg.leaves()[cols].equals(seg2.leaves()[cols])
    p1 = seg.predict(df.head(300))["lgd_regua"].fillna(-1).to_numpy()
    p2 = seg2.predict(df.head(300))["lgd_regua"].fillna(-1).to_numpy()
    assert np.allclose(p1, p2)
    # máscaras reconstruídas cobrem toda a base
    cob = sum(s["mask"].sum() for s in seg2.segments.values() if s["is_leaf"])
    assert cob == len(df)


def test_auto_merge_funde_irmas_indistinguiveis():
    df = _dois_patamares()
    seg = SequentialLGDSegmenter(df, target="lgd", sample_col="amostra",
                                 ref_sample="DES", verbose=False)
    seg.grow("ltv", splits=[0.2, 0.35, 0.5, 0.7, 0.85])
    n0 = sum(s["is_leaf"] for s in seg.segments.values())
    seg.auto_merge(alpha=0.05, verbose=False)
    n1 = sum(s["is_leaf"] for s in seg.segments.values())
    assert n0 == 6 and n1 == 2                       # recupera os 2 patamares
    cob = sum(s["mask"].sum() for s in seg.segments.values() if s["is_leaf"])
    assert cob == len(df)


def test_auto_merge_respeita_protect():
    df = _dois_patamares()
    seg = SequentialLGDSegmenter(df, target="lgd", sample_col="amostra",
                                 ref_sample="DES", verbose=False)
    seg.grow("ltv", splits=[0.2, 0.35, 0.5, 0.7, 0.85])
    folhas = [s for s, v in seg.segments.items() if v["is_leaf"]]
    prot = {folhas[0], folhas[1]}
    seg.auto_merge(alpha=0.05, protect=prot, verbose=False)
    assert prot.issubset(set(seg.segments))          # travadas preservadas


def test_ui_undo_redo_automerge_e_json(tmp_path):
    pytest.importorskip("ipywidgets")
    import contextlib
    import io
    from yggdrasil.credit_risk.lgd import LGDSegmenterUI

    df = _dois_patamares(n=4000, seed=7)
    # redireciona stdout (em ambiente headless o display() cai p/ print)
    with contextlib.redirect_stdout(io.StringIO()):
        ui = LGDSegmenterUI(df, target="lgd", sample_col="amostra", ref_sample="DES")
        nleaf = lambda: sum(s["is_leaf"] for s in ui.seg.segments.values())
        assert ui._csi_cache is not None             # CSI calculado uma vez
        ui.dd_leaf.value = "root"
        ui.dd_feature.value = "ltv"
        ui.tg_mode.value = "Manual"
        ui.tx_cuts.value = "0.2,0.35,0.5,0.7,0.85"
        ui._on_preview(None)
        ui._on_split(None)
        n_split = nleaf()
        ui._on_undo(None); n_undo = nleaf()
        ui._on_redo(None); n_redo = nleaf()
        ui.sl_alpha.value = 0.05
        ui._on_automerge(None); n_am = nleaf()
        # salva, reseta e recarrega
        p = str(tmp_path / "ui_arvore.json")
        ui.tx_json_path.value = p
        ui._on_save_json(None); n_saved = nleaf()
        ui._on_reset(None); n_reset = nleaf()
        ui._on_load_json(None); n_loaded = nleaf()
    assert n_split == 6 and n_undo == 1 and n_redo == 6
    assert n_am == 2                                  # auto-merge volta aos 2 patamares
    assert n_reset == 1 and n_loaded == n_saved
