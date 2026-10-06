"""VEL-02 — CP-SAT do optbinning com 1 worker nos ajustes da lib.

O optbinning cria ``cp_model.CpSolver()`` sem fixar ``num_workers``: o CP-SAT
usa todos os núcleos e, em EMPATE de ótimo, devolve soluções diferentes entre
execuções (o golden da árvore oscilava em ``variable_table[root|dec_mix]``).
``_common._cpsat_um_worker`` faz os solvers criados DENTRO de um ajuste da lib
(nesta thread) nascerem com 1 worker, sem tocar no ``cp_model`` global do
OR-Tools nem nos ajustes que o usuário faz direto no optbinning.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optbinning")

import optbinning.binning.cp as ob_cp  # noqa: E402
from optbinning import ContinuousOptimalBinning, OptimalBinning  # noqa: E402
from ortools.sat.python import cp_model  # noqa: E402

from yggdrasil.credit_risk import _common  # noqa: E402
from yggdrasil.credit_risk._common import (  # noqa: E402
    _cpsat_um_worker,
    _instala_cpsat_um_worker,
    fit_optbinning_splits,
)
from yggdrasil.credit_risk.tree import segmenter as treemod  # noqa: E402

CPSOLVER_ORIGINAL = cp_model.CpSolver          # capturado antes de qualquer ajuste


@pytest.fixture
def workers(monkeypatch):
    """Lista com o ``num_workers`` de cada solver que o optbinning usou."""
    vistos = []
    lock = threading.Lock()
    orig = ob_cp.BinningCP.solve

    def solve(self):
        out = orig(self)
        with lock:
            vistos.append(int(self.solver_.parameters.num_workers))
        return out

    monkeypatch.setattr(ob_cp.BinningCP, "solve", solve)
    return vistos


def _dados(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(x - 0.5)))).astype(int)
    return x, y


def _cats(n=6000, seed=5):
    # categórica de RUÍDO (alvo independente): taxas quase iguais entre níveis —
    # o terreno dos empates de ótimo
    rng = np.random.default_rng(seed)
    niveis = ["1.5", "1.50", "2", "2.00", "3.25", "0.1", "0.10"]
    x = np.array([niveis[k] for k in rng.integers(0, len(niveis), n)], dtype=object)
    y = (rng.random(n) < 0.2).astype(int)
    return x, y


# ----------------------------------------------------------------------
# solver dentro × fora do ajuste
# ----------------------------------------------------------------------
def test_ajuste_da_lib_usa_1_worker(workers):
    x, y = _dados()
    assert fit_optbinning_splits(OptimalBinning(name="v"), x, y)
    xc = x + np.random.default_rng(1).normal(size=len(x))
    assert fit_optbinning_splits(ContinuousOptimalBinning(name="c"), x, xc)
    xs, ys = _cats()
    fit_optbinning_splits(OptimalBinning(name="k", dtype="categorical"), xs, ys)
    assert len(workers) == 3 and set(workers) == {1}


def test_ajuste_direto_do_usuario_fica_com_o_padrao(workers):
    x, y = _dados()
    fit_optbinning_splits(OptimalBinning(name="v"), x, y)     # instala o proxy
    assert workers == [1]
    OptimalBinning(name="v").fit(x, y)                        # fora da lib
    assert workers == [1, 0]                                  # 0 = padrão do CP-SAT


def test_cp_model_global_do_ortools_fica_intocado():
    x, y = _dados()
    fit_optbinning_splits(OptimalBinning(name="v"), x, y)
    assert _instala_cpsat_um_worker()
    # o módulo e a classe globais são os originais e criam solver padrão
    assert cp_model.CpSolver is CPSOLVER_ORIGINAL
    assert cp_model.CpSolver().parameters.num_workers == 0
    # só o nome dentro de optbinning.binning.cp aponta para a fachada, que
    # delega todo o resto ao módulo real
    prox = ob_cp.cp_model
    assert prox is not cp_model and prox._yg_real is cp_model
    assert prox.CpModel is cp_model.CpModel
    assert prox.OPTIMAL == cp_model.OPTIMAL and prox.FEASIBLE == cp_model.FEASIBLE
    assert issubclass(prox.CpSolver, CPSOLVER_ORIGINAL)
    # fora do escopo a fachada também cria o solver padrão
    assert prox.CpSolver().parameters.num_workers == 0
    with _cpsat_um_worker():
        assert prox.CpSolver().parameters.num_workers == 1
        assert cp_model.CpSolver().parameters.num_workers == 0


def test_escopo_reentrante_e_instalacao_idempotente(monkeypatch):
    assert _instala_cpsat_um_worker()
    prox = ob_cp.cp_model
    with _cpsat_um_worker():
        with _cpsat_um_worker():
            assert _common._CPSAT_TLS.n == 2
        assert prox.CpSolver().parameters.num_workers == 1
    assert _common._CPSAT_TLS.n == 0
    assert prox.CpSolver().parameters.num_workers == 0
    # reinstalar (ex.: reload do _common) não empilha fachada sobre fachada
    monkeypatch.setattr(_common, "_CPSAT_INSTALADO", False)
    monkeypatch.setattr(ob_cp, "cp_model", prox)
    assert _instala_cpsat_um_worker()
    assert ob_cp.cp_model._yg_real is cp_model


def test_escopo_e_por_thread():
    """Enquanto uma thread está DENTRO de um ajuste da lib, outra thread (ex.:
    um ajuste do usuário) continua criando o solver padrão."""
    _instala_cpsat_um_worker()
    dentro, fora = threading.Barrier(2), threading.Barrier(2)
    res = {}

    def da_lib():
        with _cpsat_um_worker():
            dentro.wait()                        # a outra thread cria o solver agora
            fora.wait()
            res["lib"] = ob_cp.cp_model.CpSolver().parameters.num_workers

    def do_usuario():
        dentro.wait()
        res["usuario"] = ob_cp.cp_model.CpSolver().parameters.num_workers
        fora.wait()

    ts = [threading.Thread(target=da_lib), threading.Thread(target=do_usuario)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert res == {"lib": 1, "usuario": 0}


def test_knob_desligado_volta_ao_padrao(workers, monkeypatch):
    monkeypatch.setattr(_common, "OPTBINNING_CPSAT_WORKERS", 0)
    x, y = _dados()
    fit_optbinning_splits(OptimalBinning(name="v"), x, y)
    assert workers == [0]


def test_lote_paralelo_da_arvore_usa_1_worker(workers):
    """``_fit_splits_na_thread`` (lote do ThreadPoolExecutor) entra no escopo NO
    worker: todos os solvers com 1 worker e cortes iguais aos do ajuste em série."""
    rng = np.random.default_rng(11)
    lotes = []
    for k in range(8):
        x = rng.normal(size=3000)
        y = (rng.random(3000) < 1 / (1 + np.exp(-(x * (k + 1) / 4)))).astype(int)
        lotes.append((x, y))
    falhas = []
    with ThreadPoolExecutor(4) as ex:
        par = list(ex.map(lambda xy: treemod._fit_splits_na_thread(
            OptimalBinning(name="v"), xy[0], xy[1], falhas), lotes))
    serie = [fit_optbinning_splits(OptimalBinning(name="v"), x, y) for x, y in lotes]
    assert not falhas
    assert [list(map(float, s)) for s in par] == [list(map(float, s)) for s in serie]
    assert len(workers) == 16 and set(workers) == {1}


def test_variable_iv_da_arvore_com_n_jobs_usa_1_worker(workers):
    rng = np.random.default_rng(2)
    n = 3000
    s = rng.normal(size=n)
    df = pd.DataFrame({
        "a": s + rng.normal(scale=0.7, size=n),
        "b": rng.normal(size=n),
        "c": rng.integers(0, 30, n).astype(float),
        "k": np.array(["x", "y", "z"], dtype=object)[rng.integers(0, 3, n)],
        "target": (rng.random(n) < 1 / (1 + np.exp(-(s - 0.8)))).astype(float),
    })
    seg = treemod.TreeSegmenter(df, target="target", verbose=False, n_jobs=2)
    seg.variable_iv("root")
    assert workers and set(workers) == {1}


# ----------------------------------------------------------------------
# determinismo: mesmo ajuste 5× = mesmos cortes
# ----------------------------------------------------------------------
def _chave(splits):
    return tuple(tuple(sorted(map(str, g))) if isinstance(g, (list, np.ndarray))
                 else float(g) for g in splits)


@pytest.mark.parametrize("caso", ["numerica", "categorica_ruido", "continua"])
def test_mesmo_ajuste_5x_mesmos_cortes(caso, workers):
    if caso == "numerica":
        x, y = _dados(seed=9)
        mk = lambda: OptimalBinning(name="v")                          # noqa: E731
    elif caso == "categorica_ruido":
        x, y = _cats(seed=13)
        mk = lambda: OptimalBinning(name="v", dtype="categorical")     # noqa: E731
    else:
        x, _ = _dados(seed=9)
        y = np.round(x + np.random.default_rng(4).normal(size=len(x)), 1)
        mk = lambda: ContinuousOptimalBinning(name="v")                # noqa: E731
    cortes = {_chave(fit_optbinning_splits(mk(), x, y)) for _ in range(5)}
    assert len(cortes) == 1
    assert set(workers) == {1}


# Empate REAL: as contagens (não evento, evento) por categoria do problema que
# fazia o golden da árvore oscilar (``dec_mix`` na raiz, clf). O modelo CP-SAT
# tem dois ótimos — juntar as faixas 5-6 ou 6-7 da ordem por taxa —, e com o
# padrão (todos os núcleos) ~2,5% dos solves devolvem o outro (5 em 200 medidos).
EMPATE_NAO_EVENTO = [2399, 2382, 2284, 2331, 1568, 1599, 1590]
EMPATE_EVENTO = [947, 962, 943, 970, 665, 680, 680]


def _cp_do_empate():
    m = ob_cp.BinningCP("ascending", None, 6, 1000, None, None, None, None, None,
                        0.0, None, "consecutive", 0, None, 100)
    m.build_model("iv", np.array(EMPATE_NAO_EVENTO), np.array(EMPATE_EVENTO), None)
    return m


def test_empate_real_solver_1_worker_sempre_a_mesma_solucao():
    _instala_cpsat_um_worker()
    sols = set()
    for _ in range(40):
        m = _cp_do_empate()
        with _cpsat_um_worker():
            status, sol = m.solve()
        assert status == "OPTIMAL" and m.solver_.parameters.num_workers == 1
        sols.add(tuple(sol.astype(int)))
    assert sols == {(1, 1, 1, 1, 1, 0, 1)}


def test_empate_real_ajuste_categorico_5x_mesmos_grupos(workers):
    cats = [f"k{i}" for i in range(7)]
    x = np.concatenate([[c] * (a + b) for c, a, b in
                        zip(cats, EMPATE_NAO_EVENTO, EMPATE_EVENTO)]).astype(object)
    y = np.concatenate([[0] * a + [1] * b for a, b in zip(EMPATE_NAO_EVENTO, EMPATE_EVENTO)])
    grupos = {_chave(fit_optbinning_splits(
        OptimalBinning(name="v", dtype="categorical", max_n_bins=6, min_bin_size=0.05),
        x, y)) for _ in range(5)}
    assert len(grupos) == 1 and set(workers) == {1}


# ----------------------------------------------------------------------
# régua 'optbin' do ModelSegmenter (ajuste dentro de ratings/optbinning.py)
# ----------------------------------------------------------------------
class _Identidade:
    """Score = a própria feature (régua determinística, sem treino)."""
    def __init__(self, col):
        self.col = col

    def predict_proba(self, X):
        s = np.asarray(X[self.col], dtype=float)
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return np.asarray(X[self.col], dtype=float)


def _seg_score(task):
    from yggdrasil.credit_risk.model.segmenter import ModelSegmenter
    rng = np.random.default_rng(11)
    n = 4000
    x = rng.uniform(0, 1, n)
    if task == "classification":
        alvo = (rng.random(n) < 0.05 + 0.4 * x).astype(float)
    else:
        alvo = np.clip(0.2 + 0.5 * x + rng.normal(scale=0.1, size=n), 0, 1)
    df = pd.DataFrame({"feat_x": x, "target": alvo,
                       "amostra": rng.choice(["DES", "OOT"], n, p=[.7, .3])})
    seg = ModelSegmenter(df, target="target", task_type=task, sample_col="amostra",
                         ref_sample="DES", verbose=False)
    seg.set_model(_Identidade("feat_x"), features=["feat_x"])
    return seg


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_build_ratings_optbin_do_model_usa_1_worker(task, workers):
    seg = _seg_score(task)
    seg.build_ratings(method="optbin", n_ratings=5)
    assert seg.rating_labels_
    assert workers and set(workers) == {1}
    cortes = list(seg.rating_strategy.splits_)
    seg.build_ratings(method="optbin", n_ratings=5)          # mesmo score ⇒ mesmos cortes
    assert list(seg.rating_strategy.splits_) == cortes


def test_suggest_n_ratings_optbin_do_model_usa_1_worker(workers):
    seg = _seg_score("classification")
    sug = seg.suggest_n_ratings(method="optbin", n_min=3, n_max=5)
    assert sug["best"] >= 3
    assert len(workers) >= 3 and set(workers) == {1}


def test_optbin_rating_direto_do_usuario_fica_com_o_padrao(workers):
    from yggdrasil.ratings.optbinning import OptBinningRating
    seg = _seg_score("classification")
    seg.build_ratings(method="optbin", n_ratings=5)          # instala o proxy
    n_lib = len(workers)
    OptBinningRating(max_n_bins=5).fit(seg._rating_frame(), seg._make_cfg("_amostra"),
                                       problem_type="classification")
    assert set(workers[:n_lib]) == {1} and set(workers[n_lib:]) == {0}
