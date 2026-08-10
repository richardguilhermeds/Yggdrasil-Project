"""
Testes do relatório de seleção de variáveis
(:mod:`yggdrasil.credit_risk.model.selection_report`).

A base sintética é a mesma ideia dos testes da esteira: uma variável desenhada
para cair em CADA etapa (faltantes, constante, cardinalidade alta, ruído sem IV,
instável no PSI e um par redundante), de modo que o relatório tenha as três
decisões (selecionada / a revisar / excluída) e vários motivos distintos.

Cobrem, em **classificação e regressão** (paridade): que cada gráfico devolve uma
``Figure`` sem exceção — inclusive quando falta informação (sem amostra de
comparação não há PSI) —, que as tabelas formatadas trazem os rótulos legíveis e
que ``numerico=True`` devolve os valores crus, que o HTML tem todas as seções e é
**autocontido** (sem URL externa e sem ``<script src=...>``), que
``save_selection_report`` grava um arquivo legível e que o ``.xlsx`` sai com as
três abas (openpyxl é opcional — ``importorskip``).
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from yggdrasil.credit_risk.model import ModelSegmenter, run_selection  # noqa: E402
from yggdrasil.credit_risk.model import selection_report as SR  # noqa: E402

N = 1200
STEPS = ["missing", "constante", "categoricas", "iv", "psi", "monotonia", "correlacao"]


# ----------------------------------------------------------------------
# Base / fixtures
# ----------------------------------------------------------------------
def _base(task: str = "classification", seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    meses = pd.date_range("2023-01-01", periods=10, freq="MS")
    dt = rng.choice(meses, size=N)
    amostra = np.where(dt >= meses[7], "OOT", "DES")

    z_bom = rng.normal(size=N)
    z_inst = rng.normal(size=N)
    cat = np.array(rng.choice(["A", "B", "C"], size=N, p=[0.34, 0.33, 0.33]),
                   dtype=object)
    for k, i in enumerate(rng.permutation(N)[:36]):
        cat[i] = f"R{k % 5 + 1}"
    efeito_cat = pd.Series(cat).map({"A": 0.9, "B": 0.0, "C": -0.9}).fillna(0.0)

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
        "feat_bom": z_bom,                                   # sobrevive
        "feat_gemea": z_bom + rng.normal(0, 0.02, size=N),   # cai na correlação
        "feat_missing": rng.normal(size=N),                  # cai nos faltantes
        "feat_const": np.full(N, 7.0),                       # cai nas constantes
        "feat_ruido": rng.normal(size=N),                    # cai no IV
        "feat_instavel": z_inst + np.where(amostra == "OOT", 3.0, 0.0),  # cai no PSI
        "feat_cat_alta": [f"K{v:02d}" for v in rng.integers(0, 60, size=N)],
        "feat_cat_raras": cat,
    })
    df.loc[df.sample(frac=0.80, random_state=3).index, "feat_missing"] = np.nan
    return df


def _seg(task: str, df: pd.DataFrame, **kw) -> ModelSegmenter:
    kw.setdefault("sample_col", "amostra")
    kw.setdefault("ref_sample", "DES")
    return ModelSegmenter(df, target="target", task_type=task, date_col="dt_ref",
                          problem_label="inadimplência 12m", verbose=False, **kw)


@pytest.fixture(scope="module")
def res_clf():
    seg = _seg("classification", _base("classification"))
    return run_selection(seg, steps=STEPS, apply=False), seg


@pytest.fixture(scope="module")
def res_reg():
    seg = _seg("regression", _base("regression"))
    return run_selection(seg, steps=STEPS, apply=False), seg


@pytest.fixture(params=["clf", "reg"])
def par(request, res_clf, res_reg):
    """Paridade classificação × regressão: o relatório é o mesmo nos dois."""
    return res_clf if request.param == "clf" else res_reg


# ----------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------
def test_plots_devolvem_figure_em_clf_e_reg(par):
    res, _ = par
    for fig in (SR.plot_funil(res), SR.plot_motivos(res),
                SR.plot_iv_ranking(res), SR.plot_iv_psi(res)):
        assert isinstance(fig, Figure)
        assert fig.axes, "a figura precisa ter ao menos um eixo"


def test_funil_tem_uma_barra_por_etapa_mais_candidatas_e_selecionadas(res_clf):
    res, _ = res_clf
    ax = SR.plot_funil(res).axes[0]
    rotulos = [t.get_text() for t in ax.get_yticklabels()]
    assert rotulos[0] == "candidatas" and rotulos[-1] == "selecionadas"
    # candidatas + uma linha por etapa executada + selecionadas
    assert len(rotulos) == len(res.funil) + 1
    assert "faltantes" in rotulos and "correlação" in rotulos   # rótulos em pt-BR


def test_iv_ranking_usa_o_corte_da_politica_e_limita_o_top(res_clf):
    res, _ = res_clf
    min_iv = float(res.politica["parametros"]["min_iv"])
    ax = SR.plot_iv_ranking(res, top=3).axes[0]
    assert len([p for p in ax.patches]) == 3                    # top respeitado
    linhas_x = [ln.get_xdata()[0] for ln in ax.lines]
    assert any(abs(float(x) - min_iv) < 1e-9 for x in linhas_x), "corte de IV ausente"
    textos = " ".join(t.get_text() for t in ax.texts)
    assert "fora do top 3" in textos


def test_iv_psi_marca_os_quadrantes(res_clf):
    res, _ = res_clf
    ax = SR.plot_iv_psi(res).axes[0]
    textos = " ".join(t.get_text() for t in ax.texts)
    for q in ("forte e estável", "forte mas instável", "fraca e estável", "descartar"):
        assert q in textos


def test_plots_sem_psi_dao_aviso_amigavel_em_vez_de_excecao():
    """Sem coluna de amostra não há PSI — o gráfico avisa, não levanta."""
    df = _base("classification").drop(columns=["amostra"])
    seg = _seg("classification", df, sample_col=None, ref_sample=None)
    res = run_selection(seg, steps=["missing", "constante", "iv"], apply=False)
    fig = SR.plot_iv_psi(res)
    assert isinstance(fig, Figure)
    assert "sem PSI calculado" in " ".join(t.get_text() for t in fig.axes[0].texts)


def test_plot_motivos_sem_exclusoes_avisa(res_clf):
    """Esteira sem etapas → ninguém sai; o gráfico de motivos avisa."""
    _, seg = res_clf
    vazio = run_selection(seg, steps=[], apply=False)
    fig = SR.plot_motivos(vazio)
    assert "nenhuma variável foi excluída" in " ".join(
        t.get_text() for t in fig.axes[0].texts)


def test_plots_aceitam_ax_e_save_path(res_clf, tmp_path):
    from matplotlib.figure import Figure as _F
    res, _ = res_clf
    fig = _F(figsize=(6, 4))
    ax = fig.subplots()
    assert SR.plot_funil(res, ax=ax) is fig
    destino = tmp_path / "funil.png"
    SR.plot_iv_ranking(res, save_path=str(destino))
    assert destino.exists() and destino.stat().st_size > 0


# ----------------------------------------------------------------------
# Tabelas
# ----------------------------------------------------------------------
def test_tabela_decisoes_formatada_e_numerica(par):
    res, _ = par
    fmt = SR.tabela_decisoes(res)
    for col in ("Variável", "Tipo", "Decisão", "Etapa", "Motivo", "IV", "PSI (pior)",
                "Faltantes"):
        assert col in fmt.columns
    assert set(fmt["Decisão"]) <= {"Selecionada", "A revisar", "Excluída"}
    assert set(fmt["Tipo"]) <= {"numérica", "categórica"}
    assert (fmt["Faltantes"].str.endswith("%")).all()
    assert (fmt["Faltantes"].str.contains(",")).all()          # vírgula decimal pt-BR
    ivs = [v for v in fmt["IV"] if v != "—"]
    assert ivs and all("," in v for v in ivs)
    assert (fmt["Motivo"].str.len() > 0).all()                 # motivo nunca vazio

    cru = SR.tabela_decisoes(res, numerico=True)
    assert list(cru.columns) == list(res.tabela.columns)
    assert pd.api.types.is_float_dtype(cru["iv"])
    assert pd.api.types.is_float_dtype(cru["missing_pct"])
    assert sorted(cru["variavel"]) == sorted(res.tabela["variavel"])
    # ordenada por decisão: selecionadas primeiro, excluídas por último
    assert cru["decisao"].iloc[0] == "selecionada"
    assert cru["decisao"].iloc[-1] == "excluida"
    # sem ordenar, mantém a ordem de entrada das candidatas
    entrada = SR.tabela_decisoes(res, numerico=True, ordenar=False)
    assert list(entrada["variavel"]) == list(res.tabela["variavel"])


def test_tabela_decisoes_filtra_por_decisao(res_clf):
    res, _ = res_clf
    so_exc = SR.tabela_decisoes(res, numerico=True, decisao="excluida")
    assert len(so_exc) == len(res.excluidas)
    assert set(so_exc["decisao"]) == {"excluida"}


def test_tabela_funil_formatada_e_numerica(par):
    res, _ = par
    fmt = SR.tabela_funil(res)
    assert list(fmt.columns) == ["Etapa", "Entraram", "Excluídas", "A revisar",
                                 "Seguiram", "% das candidatas"]
    assert fmt["Etapa"].iloc[0] == "candidatas"
    assert fmt["% das candidatas"].iloc[0] == "100%"
    cru = SR.tabela_funil(res, numerico=True)
    assert "retido_pct" in cru.columns and "n_entrada" in cru.columns
    assert pd.api.types.is_numeric_dtype(cru["n_saida"])


def test_tabela_politica_traz_todos_os_parametros(res_clf):
    res, _ = res_clf
    pol = SR.tabela_politica(res)
    assert list(pol.columns) == ["Item", "Valor"]
    itens = set(pol["Item"])
    for chave in res.politica["parametros"]:
        assert f"parâmetro · {chave}" in itens
    assert "Etapas executadas" in itens and "Amostra de referência" in itens


def test_causas_exclusao_conta_por_etapa(res_clf):
    res, _ = res_clf
    causas = SR.causas_exclusao(res)
    assert int(causas["n"].sum()) == len(res.excluidas)
    assert (causas["n"].diff().dropna() <= 0).all()            # da maior para a menor
    por_motivo = SR.causas_exclusao(res, por="motivo")
    assert int(por_motivo["n"].sum()) == len(res.excluidas)


# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------
def test_html_tem_as_secoes_e_e_autocontido(par):
    res, seg = par
    doc = SR.build_selection_report_html(res, seg=seg)
    assert doc.lstrip().lower().startswith("<!doctype html>")
    for anchor in ("id='sumario'", "id='funil'", "id='motivos'", "id='iv'",
                   "id='iv-psi'", "id='decisoes'", "id='politica'"):
        assert anchor in doc
    assert "Sumário executivo" in doc and "Política usada" in doc
    # autocontido: nenhuma requisição externa
    assert "http://" not in doc and "https://" not in doc
    assert "<script" not in doc.lower()
    assert doc.count("data:image/png;base64,") == 4            # os 4 gráficos
    # tema claro/escuro por variáveis CSS
    assert ":root{" in doc and "prefers-color-scheme: dark" in doc
    # conteúdo: motivos por extenso e o rótulo neutro do alvo
    assert "inadimplência 12m" in doc
    assert "feat_bom" in doc and "feat_ruido" in doc


def test_html_funciona_sem_segmenter_e_com_titulo_proprio(res_clf):
    res, _ = res_clf
    doc = SR.build_selection_report_html(res, seg=None, title="Comitê de modelos",
                                         subtitle="safra 2026-06")
    assert "Comitê de modelos" in doc and "safra 2026-06" in doc
    assert "data:image/png;base64," in doc


def test_html_sinaliza_simulacao_e_avisos(res_clf):
    res, _ = res_clf
    doc = SR.build_selection_report_html(res, incluir_graficos=False)
    assert "Simulação" in doc                                   # apply=False
    assert "data:image/png;base64," not in doc                  # sem gráficos


def test_save_selection_report_grava_arquivo_legivel(res_clf, tmp_path):
    res, seg = res_clf
    destino = tmp_path / "relatorio_selecao.html"
    saida = SR.save_selection_report(res, str(destino), seg=seg)
    assert saida == str(destino) and destino.exists()
    conteudo = destino.read_text(encoding="utf-8")
    assert conteudo.strip().endswith("</html>")
    assert "Relatório de seleção de variáveis" in conteudo
    assert len(conteudo) > 5000                                 # figuras embutidas


def test_html_roundtrip_do_resultado_serializado(res_clf):
    """O relatório também funciona com um resultado reconstruído do JSON."""
    from yggdrasil.credit_risk.model import SelectionResult
    res, _ = res_clf
    volta = SelectionResult.from_dict(res.to_dict())
    doc = SR.build_selection_report_html(volta)
    assert "id='decisoes'" in doc
    assert isinstance(SR.plot_funil(volta), Figure)


# ----------------------------------------------------------------------
# Excel (openpyxl é opcional)
# ----------------------------------------------------------------------
def test_export_selection_xlsx(res_clf, tmp_path):
    pytest.importorskip("openpyxl")
    res, _ = res_clf
    destino = tmp_path / "selecao.xlsx"
    saida = SR.export_selection_xlsx(res, str(destino))
    assert saida == str(destino) and destino.exists()
    abas = pd.read_excel(destino, sheet_name=None)
    assert set(abas) == {"Decisoes", "Funil", "Politica"}
    dec = abas["Decisoes"]
    assert len(dec) == len(res.tabela)
    for col in ("Variável", "Decisão", "Motivo", "IV"):
        assert col in dec.columns
    assert pd.api.types.is_numeric_dtype(dec["IV"])             # numérico p/ análise
    assert set(dec["Decisão"]) <= {"Selecionada", "A revisar", "Excluída"}
    assert list(abas["Funil"]["Etapa"])[0] == "candidatas"
    assert list(abas["Politica"].columns) == ["Item", "Valor"]


# ----------------------------------------------------------------------
# API pública
# ----------------------------------------------------------------------
def test_exportado_no_pacote_model():
    import yggdrasil.credit_risk.model as M
    for nome in ("plot_funil", "plot_motivos", "plot_iv_ranking", "plot_iv_psi",
                 "tabela_decisoes", "tabela_funil", "tabela_politica",
                 "causas_exclusao", "build_selection_report_html",
                 "save_selection_report", "export_selection_xlsx"):
        assert nome in M.__all__
        assert callable(getattr(M, nome))
