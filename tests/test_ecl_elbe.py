"""
Testes da tabela ELBE (``yggdrasil.credit_risk.ecl.elbe``).

As propriedades que ancoram a suíte:

* **ELBE(0) = LGD do ciclo completo** — no mês do *default* nada foi recuperado,
  então o que resta perder sobre o saldo remanescente é a LGD inteira;
* **ELBE(T*) = 1** com ``ultimate='workout'`` — quem chegou ao fim do *workout*
  sem recuperar não recupera mais (a consequência correta da definição);
* a curva de recuperação é **monotônica** mesmo com coorte variável, porque é
  encadeada pela recuperação marginal — a média direta cairia;
* a **censura** não é tratada como zero: um *default* de 3 meses não puxa a
  recuperação do mês 24 para baixo;
* o **desconto** das recuperações aumenta a LGD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.ecl import (
    apply_elbe,
    detect_month_columns,
    elbe_frame,
    elbe_table,
    recovery_curve,
    workout_horizon,
)


# ----------------------------------------------------------------------
# Leitura das colunas
# ----------------------------------------------------------------------
def test_detect_month_columns_ordena_pelo_sufixo():
    df = pd.DataFrame({"lgd_m10": [1.0], "lgd_m2": [1.0], "lgd_m0": [1.0], "outra": [1.0]})
    cols, meses = detect_month_columns(df)
    assert cols == ["lgd_m0", "lgd_m2", "lgd_m10"]                 # e não a ordem do frame
    assert list(meses) == [0, 2, 10]
    with pytest.raises(ValueError, match="nenhuma coluna"):
        detect_month_columns(df, prefix="rec_m")


# ----------------------------------------------------------------------
# Curva de recuperação
# ----------------------------------------------------------------------
def test_curva_encadeada_e_monotonica_com_coorte_variavel(df_defaults_lgd):
    curva = recovery_curve(df_defaults_lgd)
    assert curva["recuperacao_acumulada"].is_monotonic_increasing
    assert curva["n_contratos"].is_monotonic_decreasing            # censura à direita
    assert curva["n_contratos"].iloc[0] == len(df_defaults_lgd)
    assert curva["exposicao_inicial"].nunique() == 1               # a coorte inteira
    assert curva["recuperacao_acumulada"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(curva["lgd_acumulada"], 1.0 - curva["recuperacao_acumulada"])


def test_censura_nao_e_tratada_como_zero(df_defaults_lgd):
    """Zerar o não observado achataria a recuperação da cauda."""
    curva = recovery_curve(df_defaults_lgd)
    zerado = df_defaults_lgd.fillna(1.0)                           # LGD 1 = recuperação 0
    curva_zerada = recovery_curve(zerado)
    assert curva["recuperacao_acumulada"].iloc[-1] > \
        curva_zerada["recuperacao_acumulada"].iloc[-1]


def test_coorte_fechada_usa_so_os_defaults_completos(df_defaults_lgd):
    completos = df_defaults_lgd["lgd_m36"].notna().sum()
    curva = recovery_curve(df_defaults_lgd, cohort="complete")
    assert curva["n_contratos"].iloc[0] == completos
    assert curva["n_contratos"].nunique() == 1                     # sem censura
    with pytest.raises(ValueError, match="cohort='complete'"):
        recovery_curve(df_defaults_lgd.assign(lgd_m36=np.nan), cohort="complete")


def test_workout_horizon_localiza_o_achatamento(df_defaults_lgd):
    curva = recovery_curve(df_defaults_lgd)
    apertado = workout_horizon(curva, tol=0.001)
    frouxo = workout_horizon(curva, tol=0.05)
    assert frouxo <= apertado <= int(curva.index[-1])
    # depois do horizonte, sobra no máximo `tol` a recuperar
    r = curva["recuperacao_acumulada"]
    assert float(r.iloc[-1] - r.loc[apertado]) <= 0.001 + 1e-12
    with pytest.raises(ValueError):
        workout_horizon(curva, tol=-1)


# ----------------------------------------------------------------------
# Tabela ELBE
# ----------------------------------------------------------------------
def test_elbe_identidades_centrais(df_defaults_lgd):
    tab = elbe_table(df_defaults_lgd, addon=0.05)
    f = tab.frame
    assert f["elbe"].iloc[0] == pytest.approx(tab.lgd)             # ELBE(0) = LGD
    assert tab.elbe_at(tab.workout) == pytest.approx(1.0)          # ELBE(T*) = 1
    assert f.loc[:tab.workout, "elbe"].is_monotonic_increasing
    assert (f["elbe"].between(0.0, 1.0)).all()
    assert np.allclose(f["lgd_in_default"], np.clip(f["elbe"] + 0.05, 0, 1))
    assert tab.lgd == pytest.approx(1.0 - tab.ultimate_recovery)
    assert np.allclose(f["perda_esperada_remanescente"],
                       f["exposicao_remanescente"] * f["elbe"])


def test_elbe_ultimate_alternativos(df_defaults_lgd):
    workout = elbe_table(df_defaults_lgd, ultimate="workout")
    ultimo = elbe_table(df_defaults_lgd, ultimate="last")
    imposto = elbe_table(df_defaults_lgd, ultimate=0.5)
    assert ultimo.lgd <= workout.lgd                               # 'last' recupera mais
    assert imposto.lgd == pytest.approx(0.5)
    assert imposto.frame["elbe"].iloc[0] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        elbe_table(df_defaults_lgd, ultimate="qualquer")
    with pytest.raises(ValueError):
        elbe_table(df_defaults_lgd, ultimate=1.5)


def test_elbe_addon_e_floor(df_defaults_lgd):
    tab = elbe_table(df_defaults_lgd, addon=0.10, floor=0.9)
    assert (tab.frame["lgd_in_default"] >= 0.9 - 1e-12).all()
    assert (tab.frame["lgd_in_default"] >= tab.frame["elbe"]).all()
    with pytest.raises(ValueError, match="addon deve ser >= 0"):
        elbe_table(df_defaults_lgd, addon=-0.1)
    with pytest.raises(ValueError):
        elbe_table(df_defaults_lgd, floor=1.5)


def test_desconto_aumenta_a_lgd(df_defaults_lgd):
    nominal = elbe_table(df_defaults_lgd)
    descontada = elbe_table(df_defaults_lgd, discount_rate=0.18)
    assert descontada.lgd > nominal.lgd
    assert descontada.ultimate_recovery < nominal.ultimate_recovery
    assert elbe_table(df_defaults_lgd, discount_rate=0.0).lgd == pytest.approx(nominal.lgd)


def test_kind_e_unit_alternativos_dao_o_mesmo_resultado(df_defaults_lgd):
    cols, _ = detect_month_columns(df_defaults_lgd)
    ead0 = df_defaults_lgd["exposicao_inicial"].to_numpy()

    como_taxa = df_defaults_lgd[["exposicao_inicial"]].copy()
    como_valor = df_defaults_lgd[["exposicao_inicial"]].copy()
    for c in cols:
        rec = 1.0 - df_defaults_lgd[c]
        como_taxa[c.replace("lgd_m", "rec_m")] = rec
        como_valor[c.replace("lgd_m", "rec_m")] = rec * ead0

    referencia = elbe_table(df_defaults_lgd).lgd
    assert elbe_table(como_taxa, lgd_prefix="rec_m", kind="recuperacao").lgd == \
        pytest.approx(referencia)
    assert elbe_table(como_valor, lgd_prefix="rec_m", kind="recuperacao",
                      unit="valor").lgd == pytest.approx(referencia)


def test_monotonic_trata_reversoes(df_defaults_lgd):
    ruim = df_defaults_lgd.copy()
    alvo = ruim.index[:80]
    ruim.loc[alvo, "lgd_m5"] = ruim.loc[alvo, "lgd_m4"] + 0.05     # estorno
    assert int(recovery_curve(ruim)["n_reversoes"].sum()) > 0
    respeitando = elbe_table(ruim, monotonic="none")
    recortado = elbe_table(ruim, monotonic="clip")
    assert recortado.lgd < respeitando.lgd                         # zerar o estorno recupera mais
    with pytest.raises(ValueError, match="marginal"):
        elbe_table(ruim, monotonic="error")


def test_elbe_por_segmento(df_defaults_lgd):
    tabelas = elbe_table(df_defaults_lgd, by="segmento", addon=0.05)
    assert set(tabelas) == {"cartao", "consignado"}
    # o DGP dá recuperação bem melhor ao consignado
    assert tabelas["cartao"].lgd > tabelas["consignado"].lgd
    longo = elbe_frame(tabelas)
    assert {"segmento", "mes_default", "elbe"} <= set(longo.columns)
    assert len(longo) == sum(len(t.frame) for t in tabelas.values())


def test_apply_elbe_escora_a_carteira_em_default(df_defaults_lgd):
    tabelas = elbe_table(df_defaults_lgd, by="segmento", addon=0.05)
    vivos = pd.DataFrame({
        "id": ["A", "B", "C"],
        "segmento": ["cartao", "consignado", "cartao"],
        "meses_em_default": [0, 12, 999],                          # o 999 é clipado
        "saldo": [1000.0, 2000.0, 500.0],
    })
    out = apply_elbe(vivos, tabelas, exposure_col="saldo", segment_col="segmento")
    assert out["elbe"].between(0, 1).all()
    assert np.allclose(out["perda_esperada"], out["saldo"] * out["elbe"])
    assert out["elbe"].iloc[2] == pytest.approx(1.0)               # além do workout
    assert (out["lgd_in_default"] >= out["elbe"]).all()

    unica = apply_elbe(vivos, tabelas["cartao"], exposure_col="saldo")
    assert len(unica) == 3
    with pytest.raises(ValueError, match="segment_col"):
        apply_elbe(vivos, tabelas)
    with pytest.raises(ValueError, match="sem tabela"):
        apply_elbe(vivos.assign(segmento="outro"), tabelas, segment_col="segmento")


def test_elbe_summary_e_serializacao(df_defaults_lgd):
    tab = elbe_table(df_defaults_lgd, addon=0.05)
    resumo = tab.summary()
    assert resumo.loc[0, "lgd_ciclo_completo"] == pytest.approx(tab.lgd)
    assert resumo.loc[0, "horizonte_workout"] == tab.workout
    d = tab.to_dict()
    assert d["lgd"] == pytest.approx(tab.lgd) and len(d["frame"]) == len(tab.frame)
    assert "ELBETable" in repr(tab)


def test_elbe_valida_entradas(df_defaults_lgd):
    with pytest.raises(ValueError, match="não encontrada"):
        elbe_table(df_defaults_lgd, exposure_col="inexistente")
    with pytest.raises(ValueError, match="> 0"):
        elbe_table(df_defaults_lgd.assign(exposicao_inicial=0.0))
    with pytest.raises(ValueError, match="vazio"):
        elbe_table(df_defaults_lgd.head(0))
    with pytest.raises(ValueError):
        elbe_table(df_defaults_lgd, kind="qualquer")
    with pytest.raises(ValueError):
        elbe_table(df_defaults_lgd, unit="qualquer")
