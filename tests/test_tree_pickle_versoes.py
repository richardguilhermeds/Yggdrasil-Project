"""TreeSegmenter salvo (pickle/joblib) por uma versão anterior continua usável.

A 0.0.22 ganhou atributos DERIVADOS na construção (amostras/códigos por linha,
memos, LRU de agregados, ``n_jobs``). Um segmentador despicklado de antes deles
estourava ``AttributeError`` em ``leaves``/``metrics``/``psi``/``variable_iv``;
o ``__setstate__`` os recompõe a partir do df e das máscaras.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.tree import TreeSegmenter

# atributos que não existiam na 0.0.21
NOVOS_0022 = ("_agg_lru", "_osv_memo", "_pair_p_cache", "_sample_codes", "_samples",
              "n_jobs")


def _seg():
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.normal(size=n)
    df = pd.DataFrame({
        "x": x,
        "c": np.array(list("ABC"), dtype=object)[rng.integers(0, 3, n)],
        "target": (rng.random(n) < 1 / (1 + np.exp(-2 * x))).astype(int),
        "amostra": rng.choice(["DES", "OOT"], n),
    })
    seg = TreeSegmenter(df, target="target", sample_col="amostra", ref_sample="DES",
                        verbose=False)
    seg.grow("x")
    return seg


def _como_versao_antiga(seg):
    """Despickle de um estado como o da 0.0.21: SEM os atributos novos e com o
    cache de agregados QUENTE no formato antigo (o ``psi`` guardava só o
    DataFrame; hoje guarda ``(tabela, detalhe)``)."""
    estado = pickle.loads(pickle.dumps(seg)).__dict__.copy()
    for k in NOVOS_0022:
        estado.pop(k, None)
    estado["_agg_cache"] = {(("psi", 1e-06), estado["_tree_version"]): seg.psi()}
    velho = TreeSegmenter.__new__(TreeSegmenter)
    velho.__setstate__(estado)
    return velho


@pytest.mark.parametrize("metodo", ["leaves", "metrics", "psi", "variable_iv"])
def test_despickle_de_versao_antiga_da_os_mesmos_resultados(metodo):
    seg = _seg()
    esperado = getattr(seg, metodo)()
    velho = _como_versao_antiga(seg)
    obtido = getattr(velho, metodo)()
    if isinstance(esperado, pd.DataFrame):
        pd.testing.assert_frame_equal(obtido, esperado)
    else:
        assert obtido == esperado


def test_despickle_de_versao_antiga_continua_editavel():
    seg = _seg()
    velho = _como_versao_antiga(seg)
    assert velho.n_jobs == 1
    assert velho._samples == seg._samples
    n_antes = len(velho.leaves())
    velho.grow("c")                       # divide as folhas atuais pela categórica
    assert len(velho.leaves()) > n_antes


def test_pickle_da_versao_atual_ida_e_volta():
    seg = _seg()
    volta = pickle.loads(pickle.dumps(seg))
    pd.testing.assert_frame_equal(volta.leaves(), seg.leaves())
    pd.testing.assert_frame_equal(volta.variable_iv(), seg.variable_iv())
