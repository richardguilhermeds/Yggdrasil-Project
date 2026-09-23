"""
Bases grandes no ModelSegmenter: processamento em blocos (score, VIF, p-valores
de Wald), VIF em forma fechada, bootstrap por contagem, recortes por coluna e as
salvaguardas de memória (Decimal do Spark, one-hot denso que não cabe na RAM).

Todos os caminhos otimizados são comparados com o cálculo direto: a otimização
não pode mudar número publicado.
"""
from __future__ import annotations

import decimal

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.model import ModelSegmenter
from yggdrasil.credit_risk.model import segmenter as segmod
from yggdrasil.metrics import bootstrap_metric_ci, bootstrap_metrics_ci
from yggdrasil.metrics.classification import _roc_pack
from yggdrasil.metrics.uncertainty import _roc_por_contagem


def _base(task: str, n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    df = pd.DataFrame(X, columns=["x0", "x1", "x2", "x3"])
    df.loc[rng.random(n) < 0.05, "x1"] = np.nan
    df["cat"] = rng.choice(["A", "B", "C", "D"], size=n)
    lin = 0.9 * df["x0"] - 0.6 * df["x2"] + (df["cat"] == "A") * 0.5
    if task == "classification":
        df["target"] = (rng.random(n) < 1 / (1 + np.exp(-(lin - 1.0)))).astype(int)
    else:
        df["target"] = 1 / (1 + np.exp(-lin)) + rng.normal(0, 0.05, n)
    df["amostra"] = np.where(rng.random(n) < 0.7, "DES", "OOT")
    return df


def _seg(df, task, **kw):
    return ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                          ref_sample="DES", verbose=False, **kw)


def _algo(task):
    return "logistica" if task == "classification" else "linear"


# ----------------------------------------------------------------------
# Processamento em blocos = processamento direto
# ----------------------------------------------------------------------
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_blocos_reproduzem_score_vif_e_pvalores(task):
    df = _base(task)
    inteiro = _seg(df, task)
    inteiro.fit(_algo(task))
    blocos = _seg(df, task)
    blocos._CHUNK_ROWS = 97                     # força dezenas de blocos
    blocos.fit(_algo(task))

    np.testing.assert_allclose(blocos.score_.to_numpy(), inteiro.score_.to_numpy(),
                               rtol=0, atol=1e-12)
    pd.testing.assert_index_equal(blocos.score_.index, inteiro.score_.index)
    pd.testing.assert_frame_equal(blocos.vif_table(), inteiro.vif_table(),
                                  check_exact=False, atol=1e-3)
    pd.testing.assert_frame_equal(blocos.model_coefficients(),
                                  inteiro.model_coefficients(),
                                  check_exact=False, atol=1e-4)
    # escorar uma base externa também passa pelos blocos
    novo = df.drop(columns=["target"]).sample(frac=1.0, random_state=3)
    np.testing.assert_allclose(blocos.predict(novo)["score"].to_numpy(),
                               inteiro.predict(novo)["score"].to_numpy(), atol=1e-9)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_shap_amostra_antes_de_transformar(task):
    """Amostrar as linhas cruas e depois transformar dá exatamente a mesma
    amostra de antes (transformar a referência inteira e amostrar)."""
    seg = _seg(_base(task), task)
    seg.fit(_algo(task))
    _est, Xt, names = seg._shap_inputs(sample_size=150)
    _e, completo = seg._shap_transform(seg._frame(None)[seg.model_features])
    esperado = completo.sample(150, random_state=seg.random_state)
    pd.testing.assert_frame_equal(Xt, esperado)
    assert names == list(completo.columns)


def test_frame_por_colunas_igual_ao_recorte_completo():
    seg = _seg(_base("classification"), "classification")
    for amostra in ("DES", "OOT"):
        pd.testing.assert_frame_equal(
            seg._frame(amostra, cols=["x1", "target"]),
            seg._frame(amostra)[["x1", "target"]])
    # coluna repetida (variável == alvo) não duplica no recorte
    assert list(seg._frame("DES", cols=["target", "target"]).columns) == ["target"]


def test_nuvem_de_pontos_amostrada_sem_mudar_a_cobertura(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.collections import PathCollection

    seg = _seg(_base("regression"), "regression")
    seg.fit("linear")

    def _pontos_e_textos(fig):
        ax = fig.axes[0]
        pts = sum(len(c.get_offsets()) for c in ax.collections
                  if isinstance(c, PathCollection))
        return pts, [t.get_text() for t in ax.texts]

    n_des = int(seg._fit_mask().sum())
    cheio_cal = _pontos_e_textos(seg.plot_calibration())
    cheio_res = _pontos_e_textos(seg.plot_residuals())
    monkeypatch.setattr(segmod, "_MAX_PONTOS_DISPERSAO", 300)
    amostra_cal = _pontos_e_textos(seg.plot_calibration())
    amostra_res = _pontos_e_textos(seg.plot_residuals())
    assert cheio_res[0] == n_des and amostra_res[0] == 300
    # a curva de calibração é um plot (Line2D), não entra na contagem de pontos
    assert amostra_cal[0] == 300 < cheio_cal[0]
    assert amostra_cal[1] == cheio_cal[1]          # cobertura da banda: todas as obs.
    import matplotlib.pyplot as plt
    plt.close("all")


# ----------------------------------------------------------------------
# VIF em forma fechada
# ----------------------------------------------------------------------
def test_vif_forma_fechada_igual_a_regressao():
    rng = np.random.default_rng(7)
    n = 4000
    a = rng.normal(size=n)
    X = np.column_stack([
        a,
        a + rng.normal(0, 0.05, n),                 # colinear: VIF alto
        rng.normal(size=n) * 1e4 + 5e6,             # média alta (cancelamento)
        rng.normal(size=n),
    ])
    np.testing.assert_allclose(ModelSegmenter._vif_values(X),
                               ModelSegmenter._vif_values_sklearn(X), rtol=1e-6)


def test_vif_degenerados():
    rng = np.random.default_rng(8)
    n = 2000
    dummies = np.eye(3)[rng.integers(0, 3, n)]      # somam 1: colinear c/ intercepto
    X = np.column_stack([rng.normal(size=n), np.full(n, 2.5), dummies])
    v = ModelSegmenter._vif_values(X)
    assert np.isfinite(v[0])
    assert np.isnan(v[1])                           # constante: indefinido
    assert all(np.isinf(x) for x in v[2:])          # colinearidade perfeita
    assert ModelSegmenter._vif_values(X[:, :1]) == [1.0]
    assert ModelSegmenter._vif_values(X[:1]) == []


def test_vif_nao_contaminado_por_dummies_completas():
    """One-hot completo (colinear com o intercepto) na mesma matriz não pode
    contaminar o VIF das colunas regulares: tem de bater com o cálculo sem as
    colunas degeneradas."""
    rng = np.random.default_rng(9)
    n = 20_000
    a = rng.normal(size=n)
    regulares = np.column_stack([a, a + rng.normal(0, 0.01, n),
                                 rng.normal(size=n) * 1e4 + 5e6])
    d4, d7 = np.eye(4)[rng.integers(0, 4, n)], np.eye(7)[rng.integers(0, 7, n)]
    completo = np.column_stack([regulares, np.full(n, 3.0), d4, d7])
    reduzido = np.column_stack([regulares, d4[:, :3], d7[:, :6]])
    np.testing.assert_allclose(ModelSegmenter._vif_values(completo)[:3],
                               ModelSegmenter._vif_values_sklearn(reduzido)[:3],
                               rtol=1e-8)


def test_vif_com_one_hot_completo_independe_dos_blocos(monkeypatch):
    """One-hot completo (colinear com o intercepto) + numéricas, acumulado em
    1, 4 ou 10 blocos: o VIF das numéricas tem de bater com a regressão direta
    no desenho sem a dummy de referência. Com o corte padrão do lstsq o ruído
    da colinearidade exata, que cresce com o nº de blocos, inflava esses VIFs
    (ex.: 1,59 no lugar de 1,23; 7,30 no lugar de 3,19). Varredura de 40
    bases da revisão pré-release, com a mesma sequência aleatória."""
    rng = np.random.default_rng(7)
    for _ in range(40):
        n = int(rng.choice([20000, 100000]))
        p = int(rng.integers(1, 4))
        L = int(rng.integers(2, 8))
        c = rng.integers(0, L, n)
        num = rng.normal(size=(n, p)) @ (np.eye(p) + 0.6 * rng.normal(size=(p, p)))
        num[:, 0] += 0.5 * c
        completo = np.column_stack([num, np.eye(L)[c]])
        ref = ModelSegmenter._vif_values_sklearn(
            np.column_stack([num, np.eye(L)[c][:, 1:]]))[:p]
        for bloco in (250_000, n // 4 + 1, n // 10 + 1):
            monkeypatch.setattr(ModelSegmenter, "_CHUNK_ROWS", bloco)
            v = ModelSegmenter._vif_values(completo)
            np.testing.assert_allclose(v[:p], ref, rtol=1e-6)
            assert all(np.isinf(x) for x in v[p:])


def test_vif_ignora_linhas_nao_finitas():
    rng = np.random.default_rng(3)
    X = rng.normal(500, 3, size=(3000, 3))
    X[10, 1] = np.inf
    X[20, 2] = np.nan
    ok = np.isfinite(X).all(axis=1)
    np.testing.assert_allclose(ModelSegmenter._vif_values(X),
                               ModelSegmenter._vif_values_sklearn(X[ok]), rtol=1e-8)


def test_vif_colunas_quase_duplicadas_de_credito():
    """Saldo e saldo + encargos (correlação ~1 − 1e-10) não podem esconder a
    colinearidade de encargos_mes: o VIF segue o da regressão direta (~50),
    não 1,0. É o caso em que uma solução via XᵀX perde a precisão."""
    rng = np.random.default_rng(0)
    n = 5000
    saldo = rng.lognormal(13, 1, n)
    encargos = rng.gamma(2.0, 5.0, n)
    X = np.column_stack([saldo, saldo + encargos, encargos + rng.normal(0, 1.0, n),
                         rng.normal(40, 10, n)])
    v = ModelSegmenter._vif_values(X)
    ref = ModelSegmenter._vif_values_sklearn(X)
    assert v[2] > 10                                   # encargos_mes: "alto"
    np.testing.assert_allclose(v[2:], ref[2:], rtol=1e-6)
    assert v[0] > 1e8 and v[1] > 1e8                   # o par quase duplicado
    # a mesma leitura pelo caminho completo do segmenter, em blocos pequenos
    df = pd.DataFrame(X, columns=["saldo", "saldo_total", "encargos_mes", "idade"])
    df["target"] = (rng.random(n) < 0.2).astype(int)
    df["amostra"] = "DES"
    seg = _seg(df, "classification")
    seg._CHUNK_ROWS = 700
    seg.fit("logistica", features=["saldo", "saldo_total", "encargos_mes", "idade"])
    vt = seg.vif_table(use_labels=False).set_index("termo")
    assert vt.loc["encargos_mes", "avaliacao"] == "alto"
    assert vt.loc["idade", "avaliacao"] == "ok"


# ----------------------------------------------------------------------
# Bootstrap: réplicas compartilhadas e ROC por contagem
# ----------------------------------------------------------------------
def _clf(n, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(float)
    s = np.round(1 / (1 + np.exp(-(1.5 * y + rng.normal(size=n)))), 2)  # com empates
    return y, s


def test_roc_por_contagem_igual_ao_roc_curve():
    y, s = _clf(3000)
    pack = _roc_por_contagem(y, s)
    rng = np.random.default_rng(1)
    for _ in range(5):
        idx = rng.integers(0, len(y), len(y))
        got = pack(np.bincount(idx, minlength=len(y)))
        ref = _roc_pack(y[idx], s[idx])
        np.testing.assert_allclose(got, ref[:3], rtol=0, atol=1e-12)


def test_bootstrap_multimetrica_igual_ao_calculo_legado():
    """Mesma seed ⇒ mesmos índices: o IC multi-métrica bate com o cálculo
    original (roc_curve por réplica, uma métrica por vez)."""
    y, s = _clf(2500, seed=4)
    novo = bootstrap_metrics_ci(y, s, metrics=("auc", "gini", "ks"), n_boot=80, seed=11)
    idx_pos, idx_neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for j, nome in enumerate(("auc", "gini", "ks")):
        rng = np.random.default_rng(11)
        reps = []
        for _ in range(80):
            idx = np.concatenate([rng.choice(idx_pos, size=idx_pos.size, replace=True),
                                  rng.choice(idx_neg, size=idx_neg.size, replace=True)])
            reps.append(_roc_pack(y[idx], s[idx])[j])
        assert novo[j]["ic_low"] == pytest.approx(np.percentile(reps, 2.5), abs=2e-6)
        assert novo[j]["ic_high"] == pytest.approx(np.percentile(reps, 97.5), abs=2e-6)
        assert novo[j]["se"] == pytest.approx(np.std(reps, ddof=1), abs=2e-6)
        assert novo[j] == bootstrap_metric_ci(y, s, metric=nome, n_boot=80, seed=11)


def test_bootstrap_multimetrica_com_callable_e_r2():
    rng = np.random.default_rng(2)
    y = rng.normal(size=600)
    p = y + rng.normal(0, 0.5, 600)
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))  # noqa: E731
    r2, r_rmse = bootstrap_metrics_ci(y, p, metrics=("r2", rmse), n_boot=50, seed=0)
    assert r2 == bootstrap_metric_ci(y, p, metric="r2", n_boot=50, seed=0)
    assert r_rmse == bootstrap_metric_ci(y, p, metric=rmse, n_boot=50, seed=0)


# ----------------------------------------------------------------------
# Salvaguardas: Decimal do Spark e one-hot denso que não cabe na RAM
# ----------------------------------------------------------------------
def test_decimal_do_spark_avisa_e_nao_converte():
    """Decimal (DecimalType do toPandas) em candidata: aviso explícito com o
    cast a fazer na origem, e nenhuma conversão implícita (treino, escoragem
    distribuída e modelos salvos continuam vendo o mesmo tipo)."""
    df = _base("classification")
    df["valor"] = [decimal.Decimal(f"{v:.2f}") for v in df["x3"] * 1000]
    df.loc[df.index[:5], "valor"] = None
    with pytest.warns(UserWarning, match=r"DecimalType.*\['valor'\].*cast\('double'\)"):
        seg = _seg(df, "classification")
    assert seg.decimal_cols_ == ["valor"]
    assert seg.df["valor"].dtype == object
    assert seg._detect_kind("valor") == "cat"
    seg.fit("logistica", features=["x0", "x1"])      # demais variáveis seguem normais
    assert seg.score_.notna().all()


def test_aviso_de_decimal_so_no_alvo_e_nas_candidatas(recwarn):
    """Chave, amostra e colunas fora do modelo não entram no aviso; nada é
    convertido, então a exportação (assign) devolve as chaves intactas."""
    D = decimal.Decimal
    df = _base("regression")
    n = len(df)
    df["contrato"] = [D(10 ** 19 + i) for i in range(n)]
    df["parcelas"] = [D(int(v)) for v in np.arange(n) % 48]
    df["lote"] = [D(i % 7) for i in range(n)]
    df["amostra"] = [D(1) if a == "DES" else D(2) for a in df["amostra"]]
    seg = ModelSegmenter(df, target="target", task_type="regression",
                         sample_col="amostra", ref_sample=D(1), verbose=False,
                         features=["x0", "x2", "parcelas"])
    assert seg.decimal_cols_ == ["parcelas"]
    avisos = [str(w.message) for w in recwarn.list if "DecimalType" in str(w.message)]
    assert len(avisos) == 1 and "contrato" not in avisos[0] and "lote" not in avisos[0]
    seg.fit("linear", features=["x0", "x2"])
    out = seg.assign()
    assert out["contrato"].tolist() == df["contrato"].tolist()
    assert list(seg.metrics()["amostra"]) == [D(1), D(2)]


def test_colunas_com_nome_duplicado_nao_quebram_a_construcao():
    """``join`` + ``toPandas()`` no Spark deixa nomes repetidos: o segmenter
    continua sendo construído (a detecção de Decimal percorre por posição)."""
    df = _base("classification")
    extra = pd.DataFrame({"cat": df["cat"],
                          "v": [decimal.Decimal("1.5")] * len(df)}, index=df.index)
    extra2 = pd.DataFrame({"v": [decimal.Decimal("2.5")] * len(df)}, index=df.index)
    df = pd.concat([df, extra, extra2], axis=1)
    assert list(df.columns).count("cat") == 2 and list(df.columns).count("v") == 2
    with pytest.warns(UserWarning, match="DecimalType"):
        seg = ModelSegmenter(df, target="target", task_type="classification",
                             sample_col="amostra", ref_sample="DES", verbose=False,
                             features=["x0", "x1", "v"])
    assert seg.decimal_cols_ == ["v"]
    seg.fit("logistica", features=["x0", "x1"])
    assert seg.score_.notna().all()


def test_texto_nao_e_convertido():
    df = _base("classification")
    seg = _seg(df, "classification")
    assert not pd.api.types.is_numeric_dtype(seg.df["cat"])   # object ou str (pandas 3)
    assert seg._detect_kind("cat") == "cat"
    assert seg.decimal_cols_ == []


def test_one_hot_que_nao_cabe_levanta_memory_error(monkeypatch):
    df = _base("classification")
    df["id_contrato"] = [f"c{i}" for i in range(len(df))]     # ID lido como categoria
    seg = _seg(df, "classification")
    monkeypatch.setattr(segmod, "_available_memory_bytes", lambda: 5 * 1024 ** 2)
    with pytest.raises(MemoryError, match="id_contrato"):
        seg.fit("logistica")
    assert seg.model is None
    # WoE agrupa as categorias nos bins: não passa pelo one-hot
    seg.fit("logistica", features=["x0", "cat"], transform="woe")
    # sem medição de memória a checagem vira no-op
    monkeypatch.setattr(segmod, "_available_memory_bytes", lambda: None)
    seg.fit("logistica", features=["x0", "cat"])


def test_folga_do_cgroup_v1_e_v2(tmp_path):
    gb = 1024 ** 3
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "memory.max").write_text(f"{8 * gb}\n")
    (v2 / "memory.current").write_text(f"{6 * gb}\n")
    (v2 / "memory.stat").write_text(
        f"anon {3 * gb}\nactive_file {2 * gb}\ninactive_file {1 * gb}\n")
    assert segmod._cgroup_free_bytes(str(v2)) == 5 * gb      # 8 − (6 − 2 − 1)
    (v2 / "memory.max").write_text("max\n")
    assert segmod._cgroup_free_bytes(str(v2)) is None        # sem limite
    v1 = tmp_path / "v1" / "memory"
    v1.mkdir(parents=True)
    (v1 / "memory.limit_in_bytes").write_text(f"{4 * gb}\n")
    (v1 / "memory.usage_in_bytes").write_text(f"{3 * gb}\n")
    (v1 / "memory.stat").write_text(
        f"total_active_file {gb // 2}\ntotal_inactive_file {gb // 2}\n")
    assert segmod._cgroup_free_bytes(str(tmp_path / "v1")) == 2 * gb
    assert segmod._cgroup_free_bytes(str(tmp_path / "nada")) is None


@pytest.mark.parametrize("transform", ["raw", "woe"])
def test_modelo_com_decimal_salvo_carrega_e_escora_igual(tmp_path, transform):
    """Sem conversão implícita, um modelo treinado com Decimal (categórica)
    volta idêntico do save/load, com o df cru ou com o df do próprio
    segmenter, e predict sobre a base crua bate com o score_."""
    rng = np.random.default_rng(5)
    n = 4000
    taxas = np.array([decimal.Decimal(v) for v in ("0.05", "0.10", "0.25", "0.50")],
                     dtype=object)
    df = pd.DataFrame({"x": rng.normal(size=n), "taxa": taxas[rng.integers(0, 4, n)]})
    df["target"] = (rng.random(n) < 0.3).astype(int)
    df["amostra"] = np.where(rng.random(n) < 0.7, "DES", "OOT")
    with pytest.warns(UserWarning, match="DecimalType"):
        seg = _seg(df, "classification")
    seg.fit("logistica", transform=transform)
    caminho = str(tmp_path / "m.json")
    seg.save(caminho)
    for base in (df, seg.df, None):
        with pytest.warns(UserWarning, match="DecimalType"):
            volta = _seg(df, "classification").load(caminho, base)
        np.testing.assert_allclose(volta.score_.to_numpy(), seg.score_.to_numpy(),
                                   atol=1e-12)
    np.testing.assert_allclose(seg.predict(df)["score"].to_numpy() / 1000,
                               seg.score_.to_numpy(), atol=1e-12)


def test_trava_de_memoria_aponta_decimal(monkeypatch):
    df = _base("classification")
    df["valor"] = [decimal.Decimal(f"{v:.3f}") for v in df["x3"]]
    with pytest.warns(UserWarning, match="DecimalType"):
        seg = _seg(df, "classification")
    monkeypatch.setattr(segmod, "_available_memory_bytes", lambda: 5 * 1024 ** 2)
    with pytest.raises(MemoryError, match=r"valor \([0-9.,]+ níveis, Decimal: converta"):
        seg.fit("logistica")
