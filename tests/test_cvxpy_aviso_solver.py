"""Import da yggdrasil sem o aviso do cvxpy sobre solver quebrado.

O cvxpy (optbinning → ropwr → cvxpy) sonda todos os solvers no import e, se um
pacote opcional está instalado mas quebra ao importar (``highspy`` compilado
contra outro numpy no cluster), loga "Encountered unexpected exception importing
solver HIGHS". A yggdrasil não usa o cvxpy: o import do optbinning é envolvido
por ``sem_aviso_solver_cvxpy``. Cada caso roda num subprocesso com um
``highspy`` falso que levanta no import (o cvxpy só sonda uma vez por processo).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

pytest.importorskip("optbinning")
pytest.importorskip("cvxpy")

RAIZ = Path(__file__).resolve().parents[1]
AVISO = "Encountered unexpected exception importing solver"
# estado do logger do cvxpy depois do import: nível, filtros e handlers
ESTADO = ("import logging\n"
          "lg = logging.getLogger('__cvxpy__')\n"
          "print('ESTADO', lg.level, len(lg.filters), len(lg.handlers))\n")


def _roda(tmp_path, codigo):
    falso = tmp_path / "highspy"
    falso.mkdir(exist_ok=True)
    (falso / "__init__.py").write_text(
        'raise ImportError("numpy.core.multiarray failed to import (simulado)")\n')
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(RAIZ), env.get("PYTHONPATH", "")])
    env["PYTHONUTF8"] = "1"
    r = subprocess.run([sys.executable, "-c", codigo], env=env, capture_output=True,
                       text=True, encoding="utf-8", timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout + r.stderr


def _estado(saida):
    return [l for l in saida.splitlines() if l.startswith("ESTADO")]


@pytest.fixture(scope="module")
def cru(tmp_path_factory):
    """Import cru do optbinning com o highspy falso: tem de mostrar o aviso —
    senão os testes abaixo passariam sem provar nada."""
    saida = _roda(tmp_path_factory.mktemp("cru"), "import optbinning\n" + ESTADO)
    if AVISO not in saida:
        pytest.skip("este cvxpy não sonda o HiGHS via highspy no import")
    return saida


@pytest.mark.parametrize("modulo", [
    "yggdrasil.credit_risk",
    "yggdrasil.credit_risk.model.segmenter",
    "yggdrasil.credit_risk.tree.segmenter",
])
def test_import_da_yggdrasil_sem_aviso(tmp_path, cru, modulo):
    saida = _roda(tmp_path, f"import sys\nimport {modulo}\n"
                            "assert 'cvxpy' in sys.modules\n" + ESTADO)
    assert AVISO not in saida, saida
    # o logger do cvxpy fica igual ao de um import cru: mesmo nível/handlers que
    # o próprio cvxpy configura, sem filtro sobrando
    assert _estado(saida) == _estado(cru) and _estado(saida)


def test_imports_preguicosos_sem_aviso(tmp_path, cru):
    # ratings/optbinning (e eda/binning) importam o optbinning só no ajuste
    saida = _roda(tmp_path, "\n".join([
        "import numpy as np",
        "from yggdrasil.ratings.optbinning import OptBinningRating",
        "rng = np.random.default_rng(0)",
        "s = rng.normal(size=2000)",
        "y = (rng.random(2000) < 1 / (1 + np.exp(-s))).astype(int)",
        "r = OptBinningRating(max_n_bins=4)",
        "r._problem_type = 'classification'",
        "r._fit_binner(s, y)",
        "assert len(r.splits_) >= 1",
    ]))
    assert AVISO not in saida, saida


def test_contexto_cala_so_esse_texto_e_so_no_bloco():
    from yggdrasil.utils import sem_aviso_solver_cvxpy

    lg = logging.getLogger("__cvxpy__")
    filtros_log, filtros_w = list(lg.filters), list(warnings.filters)

    class _Coleta(logging.Handler):
        def __init__(self):
            super().__init__()
            self.msgs = []

        def emit(self, record):
            self.msgs.append(record.getMessage())

    h = _Coleta()
    lg.addHandler(h)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with sem_aviso_solver_cvxpy():
                lg.warning(AVISO + " HIGHS:\nImportError('x')")
                lg.warning("outro log do cvxpy")
                warnings.warn(AVISO + " HIGHS:\nImportError('x')")
                warnings.warn("outro aviso qualquer")
            lg.warning(AVISO + " depois do bloco")
    finally:
        lg.removeHandler(h)
    assert lg.filters == filtros_log and warnings.filters == filtros_w
    assert h.msgs == ["outro log do cvxpy", AVISO + " depois do bloco"]
    assert [str(x.message) for x in w] == ["outro aviso qualquer"]
