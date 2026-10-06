"""Onda 2D — ranking (``variable_iv``) com o ajuste do optbinning em threads
(``n_jobs``): preparar → ajustar → finalizar sobre a MESMA lógica de
``_resolve_bins_etapas``.

``n_jobs=1`` × ``n_jobs=4`` com ``assert_frame_equal(check_exact=True)`` em
classificação e regressão (também no 2º ranking, que ajusta com os prebins
reaproveitados); ``_bins_cache`` e prebins iguais chave a chave;
``np.random.get_state()`` igual (a árvore CART saca do RNG global — mesmo número
de saques); ``warnings.filters`` restaurado e os MESMOS avisos; falha inesperada
do optbinning numa variável continua avisada (na thread principal, já fora do
filtro) e a linha sai com NaN; erro de preparação cai no laço de sempre.
Padrões (decisão do dono): ``ModelSegmenter(n_jobs=1)``; a UI usa
``min(núcleos, 8)``; ``n_jobs`` não vai para ``to_dict``."""
from __future__ import annotations

import collections
import decimal
import json
import threading
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optbinning")
pyarrow = pytest.importorskip("pyarrow")

from optbinning import OptimalBinning

from yggdrasil.credit_risk import _common as common_mod
from yggdrasil.credit_risk.model import ModelSegmenter
from yggdrasil.credit_risk.model import segmenter as segmod

D = decimal.Decimal
N = 20_000


def _base(task):
    rng = np.random.default_rng(7)
    i = np.arange(N)
    df = pd.DataFrame({"amostra": rng.choice(["DES", "OOT", "OOS"], N, p=[.6, .25, .15])})
    z = np.zeros(N)
    for k in range(12):
        x = rng.normal(size=N) * (k + 1) + k
        if k % 3 == 0:
            x[rng.random(N) < 0.08] = np.nan
        df[f"x_num{k}"] = x
        z += np.nan_to_num(x / (k + 1)) * (0.6 if k % 2 else -0.4) / (1 + k / 4)
    df["x_int"] = pd.array(np.where(rng.random(N) < .1, None, rng.integers(0, 40, N)),
                           dtype="Int64")
    f64 = pd.array(rng.gamma(2.0, 3.0, N), dtype="Float64")
    f64[rng.random(N) < .07] = pd.NA
    df["x_f64"] = f64
    df["x_const"] = 3.0
    df["x_flag_rara"] = (rng.random(N) < 0.004).astype(float)       # aviso de nível raro
    # 0/1 nos 10 mil primeiros observados da DES, contínua depois: optbinning
    # pelo caminho do recorte (numérico)
    df["x_tarde"] = np.where(i < 18_000, rng.integers(0, 2, N), rng.normal(5, 2, N))
    lv = np.array(list("ABCDEF"), dtype=object)
    c = lv[rng.integers(0, 6, N)]
    c[rng.random(N) < .05] = None
    df["c_str"] = c
    df["c_alta"] = np.array([f"k{v}" for v in rng.integers(0, 2000, N)], dtype=object)
    df["c_dec"] = np.array([D(str(v)) for v in rng.choice([1.5, 2.25, 3.0, 4.75], N)],
                           dtype=object)
    df["c_misto"] = np.array([[1, 1.0, "1", "a", True, "b"][v]
                              for v in rng.integers(0, 6, N)], dtype=object)
    df["c_bool"] = rng.random(N) < .3
    df["c_pa"] = pd.array(rng.choice(["u", "v", "w", "x"], N), dtype="string[pyarrow]")
    # categoria numérica: caminho do recorte (categórico)
    df["c_catnum"] = pd.Categorical(rng.integers(1, 7, N))
    df["x_manual"] = rng.normal(size=N)
    df.loc[rng.random(N) < .1, "x_manual"] = np.nan
    lvl = {"A": .3, "B": -.2, "C": .5, "D": 0, "E": -.5, "F": .1}
    z += pd.Series(c).map(lvl).fillna(0).to_numpy() + df["x_manual"].fillna(0).to_numpy() * .3
    z += pd.to_numeric(df["x_tarde"]).to_numpy() * .1
    if task == "classification":
        df["target"] = (rng.random(N) < 1 / (1 + np.exp(-(z - 1.5)))).astype(int)
    else:
        df["target"] = np.clip(1 / (1 + np.exp(-z)) + rng.normal(0, .1, N), 0, 1)
    return df


def _monta(task, n_jobs):
    df = _base(task)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                             ref_sample="DES", date_col=None, verbose=False, n_jobs=n_jobs)
        seg.set_manual_bins("x_manual", [-0.5, 0.0, 0.7], missing="pior")
        seg.create_categorical("x_num1")                        # variável derivada
    return seg


@pytest.fixture(autouse=True)
def _teto_menor(monkeypatch):
    # amostragem do optbinning em ação (teto lido na hora da chamada)
    monkeypatch.setattr(common_mod, "OPTBINNING_MAX_ROWS", 5_000)


def _roda(seg, seed=123, **kw):
    np.random.seed(seed)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        antes = list(warnings.filters)
        r = seg.variable_iv(**kw)
        assert list(warnings.filters) == antes           # filtros restaurados
    estado = np.random.get_state()
    avisos = collections.Counter((x.category.__name__, str(x.message)) for x in w)
    return r, estado, avisos


def _espia_pool(monkeypatch):
    """Conta os ajustes que foram ao pool e em quais threads rodaram."""
    vistos = []
    orig = segmod._ajusta_optbinning_em_thread

    def espia(b, x, y, max_rows=None, idx=None):
        vistos.append(threading.current_thread() is threading.main_thread())
        return orig(b, x, y, max_rows, idx)

    monkeypatch.setattr(segmod, "_ajusta_optbinning_em_thread", espia)
    return vistos


def _espia_cpsat(monkeypatch):
    """Conta (thread principal?, num_workers) de cada solver CP-SAT resolvido.

    ``ContinuousBinningCP`` (regressão) herda o ``solve`` de ``BinningCP``: um
    embrulho só cobre os dois. Sem ``_cpsat_um_worker()`` no worker do pool, o
    solver das threads nasce com ``num_workers=0`` (todos os núcleos) e deixa de
    ser determinístico em empate de ótimo — o que o ``assert_frame_equal`` destes
    dados (sem empate) não pegaria."""
    import optbinning.binning.cp as ob_cp
    vistos = collections.Counter()
    lock = threading.Lock()
    orig = ob_cp.BinningCP.solve

    def solve(self):
        out = orig(self)
        with lock:
            vistos[(threading.current_thread() is threading.main_thread(),
                    int(self.solver_.parameters.num_workers))] += 1
        return out

    monkeypatch.setattr(ob_cp.BinningCP, "solve", solve)
    return vistos


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_ranking_n_jobs_1_igual_a_4(task, monkeypatch):
    vistos = _espia_pool(monkeypatch)
    cpsat = _espia_cpsat(monkeypatch)
    s1, s4 = _monta(task, 1), _monta(task, 4)
    filtros = list(warnings.filters)
    r1, e1, w1 = _roda(s1)
    assert vistos == []                                  # n_jobs=1: em série
    assert cpsat and set(cpsat) == {(True, 1)}           # CP-SAT com 1 worker
    cpsat.clear()
    r4, e4, w4 = _roda(s4)
    assert warnings.filters == filtros
    assert len(vistos) >= 15 and not any(vistos)         # fits nas threads do pool
    # CP-SAT resolvido nas threads do pool, todos com 1 worker (_cpsat_um_worker
    # dentro do worker): nenhum solver com num_workers=0 (= todos os núcleos)
    assert cpsat[(False, 1)] >= 15
    assert {w for _p, w in cpsat} == {1}

    pd.testing.assert_frame_equal(r1, r4, check_exact=True)
    assert r1["variavel"].tolist() == r4["variavel"].tolist()
    # _bins_cache chave a chave (repr: -inf/inf, ordem das faixas e dos níveis)
    b1 = {k: repr(v) for k, v in s1._bins_cache.items()}
    b4 = {k: repr(v) for k, v in s4._bins_cache.items()}
    assert b1 == b4
    # prebins capturados nas threads = os do ajuste em série (sem as versões de
    # coluna da chave: o contador é global, difere entre os dois objetos)
    p1 = {k[:5]: v.tolist() for k, v in s1.__dict__.get("_prebin_cache", {}).items()}
    p4 = {k[:5]: v.tolist() for k, v in s4.__dict__.get("_prebin_cache", {}).items()}
    assert p1 == p4 and len(p1) >= 10
    # RNG global: a árvore saca dele, o nº de saques é o mesmo
    assert e1[0] == e4[0] and np.array_equal(e1[1], e4[1]) and e1[2:] == e4[2:]
    # os mesmos avisos (nível raro, Decimal...) e nenhum vazado das threads
    assert w1 == w4
    assert any("nível raro" in m for _c, m in w1)

    # 2º ranking noutro max_n_bins: ajustes com os prebins (pedido "calado")
    n_antes = len(vistos)
    q1, e1, w1 = _roda(s1, max_n_bins=6)
    q4, e4, w4 = _roda(s4, max_n_bins=6)
    assert len(vistos) > n_antes
    pd.testing.assert_frame_equal(q1, q4, check_exact=True)
    assert ({k: repr(v) for k, v in s1._bins_cache.items()}
            == {k: repr(v) for k, v in s4._bins_cache.items()})
    assert e1[0] == e4[0] and np.array_equal(e1[1], e4[1]) and e1[2:] == e4[2:]
    assert w1 == w4
    # sanidade: o ranking tem variáveis binadas pelo optbinning nos 4 caminhos
    nb = r1.set_index("variavel")["n_bins"]
    for f in ("x_num0", "x_tarde", "c_str", "c_alta", "c_catnum", "x_f64"):
        assert nb[f] >= 2, f
    assert nb["x_const"] == 0


def test_segunda_chamada_so_cache(monkeypatch):
    vistos = _espia_pool(monkeypatch)
    seg = _monta("classification", 4)
    r1 = seg.variable_iv()
    n = len(vistos)
    seg.set_manual_bins("x_num2", [0.0, 2.0])            # só ESTA variável re-bina
    r2 = seg.variable_iv()
    assert len(vistos) == n                              # 1 só faltava: sem pool
    assert r1.shape == r2.shape


def test_falha_inesperada_avisa_e_linha_nan(monkeypatch):
    orig = OptimalBinning.fit

    def fit(self, x, y, *a, **k):
        if self.name == "x_num2":
            raise RuntimeError("boom")
        return orig(self, x, y, *a, **k)

    monkeypatch.setattr(OptimalBinning, "fit", fit)
    onde = []
    aviso = common_mod.avisa_falha_optbinning

    def espia(b, e):
        onde.append((threading.current_thread() is threading.main_thread(),
                     any(getattr(f[3], "pattern", f[3]) == segmod._MODULOS_AJUSTE_OPTBIN
                         for f in warnings.filters)))
        aviso(b, e)

    # quem avisa é o ajuste em série (fit_optbinning_splits): no n_jobs=4 o
    # worker devolve (cortes, exceção), a falha não vai ao cache e a variável é
    # refeita no laço — aviso na thread principal, fora do filtro do lote
    monkeypatch.setattr(common_mod, "avisa_falha_optbinning", espia)
    linhas = []
    for nj in (1, 4):
        seg = _monta("classification", nj)
        with pytest.warns(RuntimeWarning, match="optbinning falhou inesperadamente em "
                                                "'x_num2': RuntimeError: boom") as rec:
            r = seg.variable_iv()
        assert sum("falhou inesperadamente" in str(x.message) for x in rec) == 1
        row = r.set_index("variavel").loc["x_num2"]
        assert np.isnan(row["iv"]) and row["n_bins"] == 0
        linhas.append(r)
    # nos dois: emitido uma vez, na thread principal, fora do filtro do lote
    assert onde == [(True, False), (True, False)]
    pd.testing.assert_frame_equal(linhas[0], linhas[1], check_exact=True)


def test_falha_inesperada_com_filtro_error(monkeypatch):
    """Aviso promovido a exceção (``simplefilter('error', RuntimeWarning)``,
    ``pytest -W error``): n_jobs=4 reage como o caminho em série — ranking
    completo, linha NaN, nada no cache para a variável (e nada de ``[]`` gravado
    pelo pool); MemoryError no fit concorrente idem (refeito em série)."""
    orig = OptimalBinning.fit

    def fit(self, x, y, *a, **k):
        if self.name == "x_num2":
            raise RuntimeError("boom")
        if self.name == "x_num5" and threading.current_thread() is not threading.main_thread():
            raise MemoryError("só no pool")
        return orig(self, x, y, *a, **k)

    monkeypatch.setattr(OptimalBinning, "fit", fit)
    res = []
    for nj in (1, 4):
        seg = _monta("classification", nj)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.simplefilter("error", RuntimeWarning)
            r = seg.variable_iv()
        row = r.set_index("variavel").loc["x_num2"]
        assert np.isnan(row["iv"]) and row["n_bins"] == 0
        assert not any(k[0] == "x_num2" for k in seg._bins_cache)
        assert r.set_index("variavel").loc["x_num5", "n_bins"] >= 2
        res.append((r, {k: repr(v) for k, v in seg._bins_cache.items()}))
    pd.testing.assert_frame_equal(res[0][0], res[1][0], check_exact=True)
    assert res[0][1] == res[1][1]


def test_pedidos_do_pool_ja_sorteados(monkeypatch):
    """Todo pedido do lote chega sorteado (``max_rows=0``, no máximo o teto de
    linhas), inclusive os do caminho do recorte (numérico com prefixo binário e
    Categorical numérica): o lote nunca segura x/y da DES inteira."""
    pedidos = []
    orig = segmod._ajusta_optbinning_em_thread

    def espia(b, x, y, max_rows=None, idx=None):
        pedidos.append((b.name, max_rows, len(x), len(y)))
        return orig(b, x, y, max_rows, idx)

    monkeypatch.setattr(segmod, "_ajusta_optbinning_em_thread", espia)
    seg = _monta("regression", 4)
    n_des = int((seg.df["amostra"] == "DES").sum())
    assert n_des > common_mod.OPTBINNING_MAX_ROWS
    seg.variable_iv()
    nomes = {n for n, *_ in pedidos}
    assert {"x_tarde", "c_catnum", "x_num0", "c_alta"} <= nomes
    for nome, mr, nx, ny in pedidos:
        assert mr == 0 and nx == ny <= common_mod.OPTBINNING_MAX_ROWS, nome


def test_erro_de_preparacao_cai_no_laco(monkeypatch):
    seg1 = _monta("regression", 1)
    r1 = seg1.variable_iv()
    orig = ModelSegmenter._resolve_bins_etapas
    chamadas = collections.Counter()

    def etapas(self, feature, *a, **k):
        chamadas[feature] += 1
        if feature == "c_alta" and chamadas[feature] == 1:
            raise MemoryError("preparação")             # só no pré-aquecimento
        return orig(self, feature, *a, **k)

    monkeypatch.setattr(ModelSegmenter, "_resolve_bins_etapas", etapas)
    seg4 = _monta("regression", 4)
    chamadas.clear()
    r4 = seg4.variable_iv()
    assert chamadas["c_alta"] == 2                       # pré-aquecimento + laço
    pd.testing.assert_frame_equal(r1, r4, check_exact=True)


def test_janela_limitada(monkeypatch):
    orig = ModelSegmenter._resolve_bins_etapas
    estado = {"ativos": 0, "pico": 0}

    def etapas(self, *a, **k):
        g = orig(self, *a, **k)
        try:
            pedido = next(g)
        except StopIteration as fim:
            return fim.value
        estado["ativos"] += 1
        estado["pico"] = max(estado["pico"], estado["ativos"])
        try:
            cortes = yield pedido
        finally:
            estado["ativos"] -= 1
        try:
            g.send(cortes)
        except StopIteration as fim:
            return fim.value
        estado["segundo"] = True                         # 1º ranking: 1 ajuste só
        raise AssertionError("2º pedido de ajuste")

    monkeypatch.setattr(ModelSegmenter, "_resolve_bins_etapas", etapas)
    seg = _monta("classification", 2)
    seg.variable_iv()
    assert estado["pico"] == 4                           # 2 × n_jobs, nunca mais
    assert "segundo" not in estado


def test_n_jobs_nao_persiste(tmp_path):
    seg = _monta("classification", 3)
    assert "n_jobs" not in json.dumps(seg.to_dict(), default=str)
    path = str(tmp_path / "m.json")
    seg.save(path)
    novo = ModelSegmenter(_base("classification"), target="target", sample_col="amostra",
                          verbose=False, n_jobs=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        novo.load(path)
    assert novo.n_jobs == 6                              # configuração da sessão
    assert ModelSegmenter.from_dict(seg.to_dict(), _base("classification")).n_jobs == 1


def test_n_jobs_normaliza():
    import os
    assert segmod._normaliza_n_jobs(None) == 1
    assert segmod._normaliza_n_jobs(1) == 1
    assert segmod._normaliza_n_jobs(5) == 5
    assert segmod._normaliza_n_jobs(-1) == (os.cpu_count() or 1)
    for ruim in (0, -2):
        with pytest.raises(ValueError, match="n_jobs inválido"):
            segmod._normaliza_n_jobs(ruim)
    with pytest.raises(ValueError, match="n_jobs inválido"):
        ModelSegmenter(_base("classification"), target="target", verbose=False, n_jobs=0)
    # padrão do segmentador: em série (modo regulatório)
    assert ModelSegmenter(_base("classification"), target="target", verbose=False).n_jobs == 1


class _Interrompe(BaseException):
    pass


def test_interrupcao_no_lote_restaura_filtros(monkeypatch):
    """Interrupção (BaseException) num ajuste do lote: o pool é encerrado dentro
    do filtro e os filtros de warnings voltam ao que eram."""
    orig = OptimalBinning.fit

    def fit(self, x, y, *a, **k):
        if self.name == "x_num4" and threading.current_thread() is not threading.main_thread():
            raise _Interrompe()
        return orig(self, x, y, *a, **k)

    monkeypatch.setattr(OptimalBinning, "fit", fit)
    seg = _monta("classification", 4)
    filtros = list(warnings.filters)
    with pytest.raises(_Interrompe):
        seg.variable_iv()
    assert list(warnings.filters) == filtros
    assert not [t for t in threading.enumerate() if t.name.startswith("ygg-optbin")]
    # a sessão segue: o ranking seguinte (em série) sai inteiro
    monkeypatch.setattr(OptimalBinning, "fit", orig)
    seg.n_jobs = 1
    assert len(seg.variable_iv()) > 15


def test_ui_default_n_jobs(monkeypatch):
    pytest.importorskip("ipywidgets")
    import os
    from yggdrasil.credit_risk.model import ui as uimod

    criados = []
    real = uimod.ModelSegmenter

    def falso(*a, **k):
        criados.append(k.get("n_jobs"))
        raise RuntimeError("parar aqui")

    monkeypatch.setattr(uimod, "ModelSegmenter", falso)
    # padrão da UI interativa = min(núcleos, 8); 1 (modo regulatório) por opção
    for kw, esperado in (({}, min(os.cpu_count() or 1, 8)), ({"n_jobs": None},
                         min(os.cpu_count() or 1, 8)), ({"n_jobs": 1}, 1),
                         ({"n_jobs": 4}, 4)):
        with pytest.raises(RuntimeError, match="parar aqui"):
            uimod.ModelSegmenterUI(pd.DataFrame({"target": [0, 1]}), **kw)
        assert criados[-1] == esperado
    assert real is not None
