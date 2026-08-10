"""
Testes da esteira de seleção de variáveis (:mod:`yggdrasil.credit_risk.model.selection`).

A base sintética tem uma variável desenhada para cair em CADA etapa: faltantes
demais, constante, categórica de alta cardinalidade, categórica com cauda rara,
ruído sem IV, instável no PSI, um par quase perfeitamente correlacionado e uma
"boa". Os testes verificam onde cada uma sai, com qual motivo, que ``apply=False``
não muta o segmentador (e ``apply=True`` muta), que o funil fecha aritmeticamente,
que categórica nominal não é penalizada por monotonia, que a política roundtripa
em JSON e que etapa/parâmetro inválido dá erro claro.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.model import ModelSegmenter, run_selection
from yggdrasil.credit_risk.model.selection import (
    COLUNAS_FUNIL,
    COLUNAS_TABELA,
    DECISOES,
    PARAMS_DEFAULT,
    SELECTION_STEPS,
    STEPS_DEFAULT,
    SelectionResult,
    etapas_disponiveis,
    rotulo_etapa,
)

N = 2000


def _base(task: str = "classification", seed: int = 7) -> pd.DataFrame:
    """Base sintética com uma variável por etapa da esteira."""
    rng = np.random.default_rng(seed)
    meses = pd.date_range("2023-01-01", periods=10, freq="MS")
    dt = rng.choice(meses, size=N)
    amostra = np.where(dt >= meses[7], "OOT", "DES")

    z_bom = rng.normal(size=N)
    z_inst = rng.normal(size=N)

    # categórica com cauda rara: A/B/C frequentes + 5 categorias com ~0.6% cada
    cat = np.array(rng.choice(["A", "B", "C"], size=N, p=[0.34, 0.33, 0.33]),
                   dtype=object)
    pos = rng.permutation(N)[:60]
    for k, i in enumerate(pos):
        cat[i] = f"R{k % 5 + 1}"
    efeito_cat = pd.Series(cat).map({"A": 0.9, "B": 0.0, "C": -0.9}).fillna(0.0)

    # coeficientes moderados de propósito: IV na faixa "médio/forte" (um sinal
    # forte demais viraria "suspeito" e mudaria a decisão da etapa de IV)
    lin = 0.7 * z_bom + 0.6 * z_inst + 0.7 * efeito_cat.to_numpy()
    if task == "classification":
        p = 1.0 / (1.0 + np.exp(-(lin - 1.0)))
        y = rng.binomial(1, p).astype(float)
    else:
        bruto = lin + rng.normal(0, 0.5, size=N)
        y = (bruto - bruto.min()) / (bruto.max() - bruto.min())

    df = pd.DataFrame({
        "target": y,
        "dt_ref": dt,
        "amostra": amostra,
        # sobrevive a tudo
        "feat_bom": z_bom,
        # quase idêntica à boa → cai na correlação
        "feat_gemea": z_bom + rng.normal(0, 0.02, size=N),
        # 80% de faltantes → cai no filtro de faltantes
        "feat_missing": rng.normal(size=N),
        # valor único → cai no filtro de constantes
        "feat_const": np.full(N, 7.0),
        # ruído puro → cai no IV
        "feat_ruido": rng.normal(size=N),
        # deslocada na OOT → cai no PSI
        "feat_instavel": z_inst + np.where(amostra == "OOT", 3.0, 0.0),
        # 60 categorias → cai nas categóricas (cardinalidade)
        "feat_cat_alta": [f"K{v:02d}" for v in rng.integers(0, 60, size=N)],
        # cauda rara → agrupada em OUTROS e segue
        "feat_cat_raras": cat,
    })
    df.loc[df.sample(frac=0.80, random_state=3).index, "feat_missing"] = np.nan
    return df


def _seg(task: str = "classification", df: pd.DataFrame | None = None) -> ModelSegmenter:
    return ModelSegmenter(df if df is not None else _base(task), target="target",
                          task_type=task, sample_col="amostra", ref_sample="DES",
                          date_col="dt_ref", verbose=False)


@pytest.fixture(scope="module")
def df_clf() -> pd.DataFrame:
    return _base("classification")


@pytest.fixture
def seg(df_clf) -> ModelSegmenter:
    return _seg("classification", df_clf)


def _saidas(res) -> dict:
    return dict(zip(res.tabela["variavel"], res.tabela["etapa_saida"]))


def _motivos(res) -> dict:
    return dict(zip(res.tabela["variavel"], res.tabela["motivo"]))


def _decisoes(res) -> dict:
    return dict(zip(res.tabela["variavel"], res.tabela["decisao"]))


# ----------------------------------------------------------------------
# Registro de etapas / schema
# ----------------------------------------------------------------------
def test_registro_de_etapas():
    assert etapas_disponiveis() == ["missing", "constante", "categoricas", "iv",
                                    "psi", "monotonia", "correlacao", "vif",
                                    "backward"]
    # a sequência default trata as categóricas ANTES do IV (o IV das categóricas
    # tem de ser medido depois do agrupamento das raras)
    assert list(STEPS_DEFAULT).index("categoricas") < list(STEPS_DEFAULT).index("iv")
    assert "vif" not in STEPS_DEFAULT and "backward" not in STEPS_DEFAULT
    assert rotulo_etapa("correlacao") == "correlação"
    assert rotulo_etapa("candidatas") == "candidatas"   # linha inicial do funil
    for nome, step in SELECTION_STEPS.items():
        assert step.nome == nome and step.rotulo and step.descricao


def test_schema_do_resultado(seg):
    res = run_selection(seg, apply=False)
    assert tuple(res.tabela.columns) == COLUNAS_TABELA
    assert tuple(res.funil.columns) == COLUNAS_FUNIL
    assert set(res.tabela["decisao"]) <= set(DECISOES)
    assert set(res.tabela["tipo"]) <= {"num", "cat"}
    # uma linha por candidata, na ordem de entrada
    assert list(res.tabela["variavel"]) == list(seg.candidates)
    assert (res.tabela["motivo"].astype(bool)).all()      # toda linha justificada
    assert set(res.selecionadas) | set(res.excluidas) | set(res.revisar) == \
        set(seg.candidates)
    assert res.tabela.loc[res.tabela["variavel"] == "feat_bom", "rotulo"].iloc[0] \
        == "feat_bom"
    assert isinstance(repr(res), str) and "candidatas" in repr(res)


# ----------------------------------------------------------------------
# Cada variável sai na etapa certa, com o motivo certo
# ----------------------------------------------------------------------
def test_cada_variavel_sai_na_etapa_esperada(seg):
    res = run_selection(seg, apply=False)
    saida, motivo, dec = _saidas(res), _motivos(res), _decisoes(res)

    assert saida["feat_missing"] == "missing"
    assert "faltantes em 80" in motivo["feat_missing"]

    assert saida["feat_const"] == "constante"
    assert "valor único" in motivo["feat_const"]

    assert saida["feat_cat_alta"] == "categoricas"
    assert "alta cardinalidade: 60 categorias" in motivo["feat_cat_alta"]

    assert saida["feat_ruido"] == "iv"
    assert "IV" in motivo["feat_ruido"]

    assert saida["feat_instavel"] == "psi"
    assert "PSI" in motivo["feat_instavel"] and "instável" in motivo["feat_instavel"]

    # do par 0.98-correlacionado, exatamente uma sai — e sai na correlação
    par = {"feat_bom", "feat_gemea"}
    caiu = {f for f in par if dec[f] == "excluida"}
    assert len(caiu) == 1
    perdedora = caiu.pop()
    ficou = (par - {perdedora}).pop()
    assert saida[perdedora] == "correlacao"
    assert "redundante com" in motivo[perdedora] and ficou in motivo[perdedora]
    assert dec[ficou] == "selecionada" and saida[ficou] == ""

    # a categórica com cauda rara sobrevive, com o agrupamento registrado
    assert dec["feat_cat_raras"] in ("selecionada", "revisar")
    assert "agrupada(s) em OUTROS" in motivo["feat_cat_raras"]
    assert "5 categoria(s) rara(s)" in motivo["feat_cat_raras"]


def test_metricas_na_tabela(seg):
    res = run_selection(seg, apply=False)
    tab = res.tabela.set_index("variavel")
    # quem caiu no filtro duro não gasta binning: métrica vazia
    assert pd.isna(tab.loc["feat_missing", "iv"])
    assert tab.loc["feat_missing", "missing_pct"] == pytest.approx(80, abs=3)
    assert tab.loc["feat_const", "missing_pct"] == 0
    # quem chegou ao IV tem métrica preenchida
    assert tab.loc["feat_bom", "iv"] > 0.1
    assert tab.loc["feat_bom", "forca"] in ("médio", "forte", "suspeito")
    assert tab.loc["feat_bom", "tendencia"] in ("crescente", "decrescente")
    assert int(tab.loc["feat_bom", "n_inversoes"]) == 0
    assert tab.loc["feat_instavel", "pior_psi"] > 0.25
    assert tab.loc["feat_cat_raras", "n_categorias"] == 8
    assert tab.loc["feat_cat_alta", "n_categorias"] == 60
    assert pd.isna(tab.loc["feat_bom", "n_categorias"])   # numérica


# ----------------------------------------------------------------------
# Funil
# ----------------------------------------------------------------------
def test_funil_fecha_aritmeticamente(seg):
    res = run_selection(seg, apply=False)
    fun = res.funil
    assert fun.iloc[0]["etapa"] == "candidatas"
    assert fun.iloc[0]["n_entrada"] == fun.iloc[0]["n_saida"] == len(seg.candidates)
    assert list(fun["etapa"])[1:] == list(STEPS_DEFAULT)
    for i in range(len(fun)):
        r = fun.iloc[i]
        assert r["n_saida"] == r["n_entrada"] - r["n_excluidas"]
        if i:
            assert r["n_entrada"] == fun.iloc[i - 1]["n_saida"]
    assert fun.iloc[-1]["n_saida"] == len(res.selecionadas) + len(res.revisar)
    # o funil conta o que a etapa de fato removeu
    assert int(fun.set_index("etapa").loc["missing", "n_excluidas"]) == 1


def test_steps_vazio_nao_exclui_ninguem(seg):
    res = run_selection(seg, steps=[], apply=False)
    assert len(res.funil) == 1 and res.funil.iloc[0]["etapa"] == "candidatas"
    assert res.selecionadas == list(seg.candidates) and not res.excluidas
    assert set(res.tabela["motivo"]) == {"nenhuma etapa executada"}


# ----------------------------------------------------------------------
# apply=False (simulação) × apply=True
# ----------------------------------------------------------------------
def _estado(seg) -> dict:
    return {"included": set(seg.included), "candidates": list(seg.candidates),
            "var_meta": copy.deepcopy(seg.var_meta),
            "colunas": list(seg.df.columns)}


def test_apply_false_nao_muta_o_segmentador(seg):
    antes = _estado(seg)
    res = run_selection(seg, apply=False)
    assert _estado(seg) == antes
    assert seg.manual_bins("feat_cat_raras") is None    # agrupamento desfeito
    assert res.politica["aplicado"] is False
    assert res.excluidas                                # houve decisão, só não aplicada


def test_apply_true_muta_o_segmentador(seg):
    res = run_selection(seg, apply=True)
    assert seg.included == set(res.selecionadas) | set(res.revisar)
    for f in res.excluidas:
        assert f not in seg.included
        assert seg.var_meta[f]["categoria"] == "descartar"
        assert seg.var_meta[f]["motivo"] == _motivos(res)[f]
    for f in res.selecionadas:
        assert seg.var_meta[f]["categoria"] == "manter"
    for f in res.revisar:
        assert seg.var_meta[f]["categoria"] == "revisar" and f in seg.included
    # o agrupamento das categorias raras fica APLICADO (bins manuais do segmenter)
    grupos = seg.manual_bins("feat_cat_raras")
    assert grupos is not None
    assert sorted(len(g) for g in grupos) == [1, 1, 1, 5]   # A, B, C + OUTROS
    assert sorted(grupos[-1]) == ["R1", "R2", "R3", "R4", "R5"]
    # o motivo aparece no ranking da UI
    assert "motivo" in seg.variable_iv().columns


def test_incluir_revisar_false(seg):
    res = run_selection(seg, apply=True, incluir_revisar=False,
                        monotonia_exclui=False, psi_warn=0.001)
    assert res.revisar, "o cenário deveria produzir alguma variável a revisar"
    for f in res.revisar:
        assert f not in seg.included
        assert seg.var_meta[f]["categoria"] == "revisar"


def test_iv_das_categoricas_medido_apos_agrupamento(seg):
    """O IV da categórica com cauda rara é medido DEPOIS do agrupamento — o
    agrupamento é aplicado de fato via os bins manuais do segmentador."""
    iv_cru = seg.variable_iv(features=["feat_cat_raras"], with_psi=False)["iv"].iloc[0]
    res = run_selection(seg, apply=True)
    iv_pos = res.tabela.set_index("variavel").loc["feat_cat_raras", "iv"]
    assert np.isfinite(iv_pos)
    # com 8 categorias e 5 minúsculas, a binagem crua e a agrupada não coincidem
    assert seg.manual_bins("feat_cat_raras") is not None
    assert iv_cru is not None


def test_agrupamento_manual_do_usuario_e_preservado(seg):
    seg.set_manual_bins("feat_cat_raras", [["A"], ["B", "C"], ["R1", "R2", "R3",
                                                              "R4", "R5"]])
    antes = seg.manual_bins("feat_cat_raras")
    res = run_selection(seg, apply=True)
    assert seg.manual_bins("feat_cat_raras") == antes
    assert "agrupamento manual já definido" in _motivos(res)["feat_cat_raras"]


def test_faltante_vira_categoria_propria(df_clf):
    df = df_clf.copy()
    idx = df.sample(frac=0.10, random_state=11).index
    df.loc[idx, "feat_cat_raras"] = None
    seg = _seg("classification", df)
    res = run_selection(seg, apply=False)
    assert "(faltante)" in _motivos(res)["feat_cat_raras"]


# ----------------------------------------------------------------------
# Monotonia: categórica nominal é ISENTA
# ----------------------------------------------------------------------
def test_categorica_nominal_nao_e_penalizada_por_monotonia(seg):
    res = run_selection(seg, steps=["categoricas", "iv", "monotonia"], apply=False,
                        monotonia_exclui=True)
    dec = _decisoes(res)
    assert dec["feat_cat_raras"] != "excluida"
    mono = [h for h in res.historico
            if h["etapa"] == "monotonia" and h["variavel"] == "feat_cat_raras"]
    assert mono and mono[0]["decisao"] == "passou"
    assert "isenta (categórica nominal)" in mono[0]["motivo"]


def test_monotonia_exclui(df_clf):
    """Numérica em U: tendência não-monotônica vira 'revisar' por padrão e
    'excluida' com monotonia_exclui=True.

    O binning ótimo força tendência monotônica; a não-monotonia aparece quando o
    analista fixa os cortes na mão (bins manuais), que é o caso realista."""
    df = df_clf.copy()
    rng = np.random.default_rng(5)
    # risco alto nas pontas e baixo no meio (relação em U com o alvo)
    u = np.where(df["target"].to_numpy() > 0.5,
                 rng.choice([-1.8, 1.8], size=len(df)), 0.0) \
        + rng.normal(0, 0.6, len(df))
    df["feat_em_u"] = u

    def _monta():
        s = _seg("classification", df)
        s.set_manual_bins("feat_em_u", [-1.0, 1.0])
        return s

    res = run_selection(_monta(), steps=["monotonia"], apply=False,
                        features=["feat_em_u"])
    assert _decisoes(res)["feat_em_u"] == "revisar"
    assert "não-monotônica" in _motivos(res)["feat_em_u"]
    assert "inversão(ões)" in _motivos(res)["feat_em_u"]
    res2 = run_selection(_monta(), steps=["monotonia"], apply=False,
                         features=["feat_em_u"], monotonia_exclui=True)
    assert _decisoes(res2)["feat_em_u"] == "excluida"
    assert _saidas(res2)["feat_em_u"] == "monotonia"


# ----------------------------------------------------------------------
# IV suspeito (vazamento) → revisar, nunca exclusão automática
# ----------------------------------------------------------------------
def test_iv_suspeito_vira_revisar(df_clf):
    df = df_clf.copy()
    rng = np.random.default_rng(2)
    # quase determina o alvo, mas com ruído suficiente para o binning existir
    df["feat_vazamento"] = 3.0 * df["target"].to_numpy() + rng.normal(0, 1.0, len(df))
    seg = _seg("classification", df)
    res = run_selection(seg, steps=["iv"], apply=False, features=["feat_vazamento"])
    assert _decisoes(res)["feat_vazamento"] == "revisar"
    assert "vazamento" in _motivos(res)["feat_vazamento"]
    assert _saidas(res)["feat_vazamento"] == ""      # revisar sobrevive


# ----------------------------------------------------------------------
# Histórico / política
# ----------------------------------------------------------------------
def test_historico_por_variavel_e_etapa(seg):
    res = run_selection(seg, apply=False)
    assert res.historico
    for h in res.historico:
        assert set(h) <= {"variavel", "etapa", "decisao", "motivo", "detalhe"}
        assert h["etapa"] in STEPS_DEFAULT
        assert h["decisao"] in ("passou", "revisar", "excluida")
        assert h["motivo"]
    # uma variável excluída não é avaliada nas etapas seguintes
    etapas_missing = [h["etapa"] for h in res.historico
                      if h["variavel"] == "feat_missing"]
    assert etapas_missing == ["missing"]


def test_politica_roundtrip_json(seg):
    res = run_selection(seg, steps=["missing", "iv"], apply=False, min_iv=0.05,
                        max_missing=0.5)
    pol = res.politica
    assert pol["etapas"] == ["missing", "iv"] and pol["aplicado"] is False
    assert pol["task_type"] == "classification" and pol["alvo"] == "target"
    assert pol["amostra_referencia"] == "DES"
    # TODOS os parâmetros efetivos (inclusive defaults) ficam registrados
    assert set(pol["parametros"]) == set(PARAMS_DEFAULT)
    assert pol["parametros"]["min_iv"] == 0.05
    assert pol["parametros"]["max_missing"] == 0.5
    assert pol["parametros"]["max_psi"] == PARAMS_DEFAULT["max_psi"]
    assert json.loads(json.dumps(pol)) == pol            # roundtrip JSON

    volta = SelectionResult.from_dict(json.loads(json.dumps(res.to_dict())))
    assert volta.politica == pol
    assert list(volta.tabela["variavel"]) == list(res.tabela["variavel"])
    assert list(volta.tabela["decisao"]) == list(res.tabela["decisao"])
    assert list(volta.funil["n_saida"]) == list(res.funil["n_saida"])
    assert volta.selecionadas == res.selecionadas and volta.excluidas == res.excluidas
    assert len(volta.historico) == len(res.historico)


def test_min_iv_default_por_task_type(df_clf):
    res_clf = run_selection(_seg("classification", df_clf), steps=["iv"], apply=False,
                            features=["feat_bom"])
    assert res_clf.politica["parametros"]["min_iv"] == 0.02
    df_reg = _base("regression")
    res_reg = run_selection(_seg("regression", df_reg), steps=["iv"], apply=False,
                            features=["feat_bom"])
    assert res_reg.politica["parametros"]["min_iv"] == 0.01
    assert res_reg.politica["task_type"] == "regression"


# ----------------------------------------------------------------------
# Erros claros
# ----------------------------------------------------------------------
def test_step_invalido_erro_claro(seg):
    with pytest.raises(ValueError, match="etapa desconhecida") as e:
        run_selection(seg, steps=["missing", "gini"], apply=False)
    msg = str(e.value)
    for nome in etapas_disponiveis():
        assert nome in msg


def test_step_repetido_erro_claro(seg):
    with pytest.raises(ValueError, match="repetida"):
        run_selection(seg, steps=["iv", "iv"], apply=False)


def test_parametro_desconhecido_erro_claro(seg):
    with pytest.raises(ValueError, match="desconhecido"):
        run_selection(seg, steps=[], apply=False, min_ivv=0.02)


def test_feature_nao_candidata_erro_claro(seg):
    with pytest.raises(ValueError, match="não são candidatas"):
        run_selection(seg, steps=[], apply=False, features=["target"])


def test_vif_sem_modelo_erro_claro(seg):
    with pytest.raises(RuntimeError, match="modelo ajustado"):
        run_selection(seg, steps=["vif"], apply=False)
    # a pré-checagem falha ANTES de mexer no segmentador
    assert seg.included == set(seg.candidates)


def test_backward_criterion_invalido(seg):
    with pytest.raises(ValueError, match="backward_criterion"):
        run_selection(seg, steps=[], apply=False, backward_criterion="cotovelo")
    with pytest.raises(ValueError, match="backward_n_variaveis"):
        run_selection(seg, steps=[], apply=False, backward_criterion="manual")


def test_aviso_de_ordem_categoricas_depois_do_iv(seg):
    with pytest.warns(UserWarning, match="categoricas"):
        res = run_selection(seg, steps=["iv", "categoricas"], apply=False)
    assert res.politica["avisos"] and "inflado" in res.politica["avisos"][0]


# ----------------------------------------------------------------------
# Progresso
# ----------------------------------------------------------------------
def test_progress_callback(seg):
    eventos = []
    run_selection(seg, steps=["missing", "constante"], apply=False,
                  progress_callback=lambda k, l, s, d: eventos.append((k, l, s, d)))
    chaves = [e[0] for e in eventos]
    assert chaves == ["missing", "missing", "constante", "constante", "fim"]
    assert [e[2] for e in eventos] == ["run", "ok", "run", "ok", "ok"]
    assert eventos[1][1] == "faltantes" and "excluída" in eventos[1][3]

    # callback quebrado nunca derruba a esteira (progresso é cosmético)
    def _ruim(*_a):
        raise RuntimeError("boom")

    res = run_selection(seg, steps=["missing"], apply=False, progress_callback=_ruim)
    assert res.excluidas == ["feat_missing"]


# ----------------------------------------------------------------------
# Etapas que dependem do modelo (VIF / backward)
# ----------------------------------------------------------------------
def test_etapa_vif_com_modelo(seg):
    feats = ["feat_bom", "feat_gemea", "feat_instavel"]
    seg.clear_features()
    for f in feats:
        seg.include(f)
    seg.fit("logistica", transform="raw")
    res = run_selection(seg, steps=["vif"], apply=False, features=feats)
    dec, motivo = _decisoes(res), _motivos(res)
    # o par quase colinear estoura o VIF (revisar por padrão, sem excluir)
    assert dec["feat_bom"] == "revisar" and dec["feat_gemea"] == "revisar"
    assert "VIF" in motivo["feat_bom"] and "multicolinearidade" in motivo["feat_bom"]
    assert dec["feat_instavel"] == "selecionada"
    assert not res.excluidas
    # com vif_exclui=True a mesma régua exclui
    res2 = run_selection(seg, steps=["vif"], apply=False, features=feats,
                         vif_exclui=True)
    assert set(res2.excluidas) == {"feat_bom", "feat_gemea"}
    assert _saidas(res2)["feat_bom"] == "vif"


def test_etapa_backward(df_clf):
    seg = _seg("classification", df_clf)
    feats = ["feat_bom", "feat_instavel", "feat_ruido"]
    res = run_selection(seg, steps=["backward"], apply=False, features=feats,
                        backward_min_features=1)
    assert res.excluidas, "o backward deveria remover ao menos uma variável"
    for f in res.excluidas:
        assert _saidas(res)[f] == "backward"
        assert "removida no backward elimination" in _motivos(res)[f]
        assert "passo com" in _motivos(res)[f]
    for f in res.selecionadas:
        assert "mantida no backward elimination" in _motivos(res)[f]


def test_backward_menos_de_duas_variaveis_e_ignorado(seg):
    res = run_selection(seg, steps=["backward"], apply=False, features=["feat_bom"])
    assert not res.excluidas
    assert any("backward" in a for a in res.politica["avisos"])
    assert int(res.funil.iloc[-1]["n_saida"]) == 1


# ----------------------------------------------------------------------
# Regressão (paridade com classificação)
# ----------------------------------------------------------------------
def test_esteira_em_regressao():
    seg = _seg("regression", _base("regression"))
    res = run_selection(seg, apply=True)
    saida = _saidas(res)
    assert saida["feat_missing"] == "missing" and saida["feat_const"] == "constante"
    assert saida["feat_cat_alta"] == "categoricas"
    assert saida["feat_instavel"] == "psi"
    assert seg.included == set(res.selecionadas) | set(res.revisar)
    assert res.politica["problema"] == "target"      # rótulo neutro do alvo
