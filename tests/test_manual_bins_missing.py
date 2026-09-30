"""Faltantes na categorização manual do ModelSegmenter: faixa própria (padrão),
pior/melhor faixa ou uma faixa específica — e a visão disso na aba Análise."""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.model import ModelSegmenter


def _df(n=8000, seed=0):
    rng = np.random.default_rng(seed)
    renda = rng.gamma(2, 1500, n)
    logit = -1.5 - 0.0004 * (renda - 3000)
    y = rng.binomial(1, 1 / (1 + np.exp(-logit))).astype(float)
    renda[rng.random(n) < .12] = np.nan
    gar = rng.choice(list("ABCD"), n).astype(object)
    gar[rng.random(n) < .05] = None
    df = pd.DataFrame({"renda": renda, "gar": gar, "sem_na": rng.normal(size=n), "target": y})
    df["amostra"] = np.where(rng.random(n) < .7, "DES", "OOT")
    return df


def _seg():
    with contextlib.redirect_stdout(io.StringIO()):
        return ModelSegmenter(_df(), target="target", sample_col="amostra")


def _faixas(seg, f):
    return list(seg.variable_table(f)["faixa"])


def test_padrao_e_faixa_propria():
    seg = _seg()
    seg.set_manual_bins("renda", "1500, 3000, 5000")
    assert seg.missing_bin("renda") == "separado"
    assert "(faltante)" in _faixas(seg, "renda")
    info = seg.missing_info("renda")
    assert info["faixa"] == "(faltante)" and info["n"] > 0 and 0.10 < info["pct"] < 0.14


def test_faixa_especifica_absorve_os_faltantes():
    seg = _seg()
    seg.set_manual_bins("renda", "1500, 3000, 5000")
    n_na = seg.missing_info("renda")["n"]
    n0 = int(seg.variable_table("renda").set_index("faixa").loc["(-inf, 1500]", "n"])
    seg.set_missing_bin("renda", 0)
    vt = seg.variable_table("renda").set_index("faixa")
    assert "(faltante)" not in vt.index
    assert int(vt.loc["(-inf, 1500] + faltante", "n"]) == n0 + n_na
    assert seg.missing_info("renda")["faixa"] == "(-inf, 1500] + faltante"
    assert seg.manual_bins_faixas("renda")[0] == "(-inf, 1500]"   # rótulo limpo p/ o índice


@pytest.mark.parametrize("destino,esperado", [("pior", "(-inf, 1500]"), ("melhor", "(5000, inf]")])
def test_pior_e_melhor_seguem_o_risco(destino, esperado):
    seg = _seg()
    seg.set_manual_bins("renda", "1500, 3000, 5000", missing=destino)
    assert seg.missing_info("renda")["faixa"] == esperado + " + faltante"


def test_categorica_e_nan_fora_da_referencia():
    seg = _seg()
    seg.set_manual_bins("gar", "A; B; C, D", missing=2)
    assert seg.missing_info("gar")["faixa"] == "{C, D} + faltante"
    seg.set_manual_bins("sem_na", "-1, 0, 1", missing="pior")    # sem NaN na DES
    info = seg.missing_info("sem_na")
    assert info["n"] == 0 and info["faixa"].endswith("+ faltante")


def test_valida_destino_e_exige_manual():
    seg = _seg()
    with pytest.raises(ValueError, match="categorização manual"):
        seg.set_missing_bin("renda", "pior")
    seg.set_manual_bins("renda", "1500, 3000, 5000")
    with pytest.raises(ValueError, match="inexistente"):
        seg.set_missing_bin("renda", 9)
    with pytest.raises(ValueError):
        seg.set_missing_bin("renda", "qualquer")


def test_limpar_bins_limpa_destino():
    seg = _seg()
    seg.set_manual_bins("renda", "1500, 3000, 5000", missing=1)
    seg.clear_manual_bins("renda")
    assert seg.missing_bin("renda") == "separado"
    assert "na_destino" not in seg.var_meta["renda"]


@pytest.mark.parametrize("transform", ["raw", "woe"])
def test_modelo_e_escoragem_seguem_o_destino(transform):
    seg = _seg()
    seg.set_manual_bins("renda", "1500, 3000, 5000", missing=0)
    seg.set_scorecard_ordinal("renda")
    tab = seg.scorecard_ordinal_table("renda")
    assert "(-inf, 1500] + faltante" in list(tab["faixa"])
    with contextlib.redirect_stdout(io.StringIO()):
        seg.fit(features=["renda"], transform=transform)
    X = pd.DataFrame({"renda": [np.nan, 100.0, 9000.0]})
    enc = seg.model.named_steps["pre"].transform(X)
    enc = np.asarray(enc, dtype=float)[:, 0]
    assert enc[0] == enc[1]                        # NaN codificado como a faixa (-inf, 1500]
    assert enc[2] != enc[1]


def test_persistencia_do_destino():
    seg = _seg()
    seg.set_manual_bins("renda", "1500, 3000, 5000", missing="melhor")
    with contextlib.redirect_stdout(io.StringIO()):
        seg2 = ModelSegmenter.from_dict(seg.to_dict(), _df())
    assert seg2.missing_bin("renda") == "melhor"
    assert seg2.missing_info("renda")["faixa"] == "(5000, inf] + faltante"


# ───────────────────────── UI ─────────────────────────
@pytest.fixture
def ui():
    pytest.importorskip("ipywidgets")
    import matplotlib
    matplotlib.use("Agg")
    from yggdrasil.credit_risk.model import ModelSegmenterUI
    df = _df()
    df["dt_ref"] = pd.Timestamp("2024-01-01")
    with contextlib.redirect_stdout(io.StringIO()):
        u = ModelSegmenterUI(df, target="target", task_type="classification",
                             sample_col="amostra", ref_sample="DES", date_col="dt_ref")
        u.dd_var2.value = "renda"
    return u


def _aplica(ui, cortes):
    ui.tg_binmode.value = "Manual"
    ui.tx_cuts.value = cortes
    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_apply_bins(None)


def test_ui_otimo_mostra_faltantes_sem_seletor(ui):
    assert "faltantes na referência" in ui.out_na_info.value
    assert "(faltante)" in ui.out_na_info.value
    assert ui.dd_na_dest.layout.display == "none"
    assert ui._passo_res.layout.display == "none"
    assert "binning ótimo" in ui.out_bin_status.value


def test_ui_manual_escolhe_faixa_e_atualiza_tudo(ui):
    _aplica(ui, "1500, 3000, 5000")
    assert ui.dd_na_dest.layout.display == ""
    assert ui._passo_res.layout.display == ""
    valores = [v for _, v in ui.dd_na_dest.options]
    assert valores[:3] == ["separado", "pior", "melhor"] and valores[3:] == [0, 1, 2, 3]
    assert "faltantes aqui" in ui.out_ord_table.value            # linha (faltante)

    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_na_dest.value = 3
    assert ui.seg.missing_bin("renda") == 3
    assert "(5000, inf] + faltante" in ui.out_na_info.value
    assert "(5000, inf] + faltante" in ui.out_ord_table.value
    assert "(5000, inf] + faltante" in ui.out_an_table.value     # análise re-renderizada

    with contextlib.redirect_stdout(io.StringIO()):
        ui._on_undo(None)
    assert ui.seg.missing_bin("renda") == "separado" and ui.dd_na_dest.value == "separado"


def test_ui_novos_cortes_invalidam_indice(ui):
    _aplica(ui, "1500, 3000, 5000")
    with contextlib.redirect_stdout(io.StringIO()):
        ui.dd_na_dest.value = 3
    _aplica(ui, "3000")                                         # agora só 2 faixas
    assert ui.seg.missing_bin("renda") == "separado"
    assert ui.dd_na_dest.value == "separado"


def test_ui_tabela_nao_troca_virgula_do_rotulo(ui):
    _aplica(ui, "1500, 3000, 5000")
    assert "(1500, 3000]" in ui.out_ord_table.value
