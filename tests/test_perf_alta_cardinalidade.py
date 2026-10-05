"""Categóricas de ALTA cardinalidade (CEP, IDs, Decimal do toPandas): as máscaras de
grupo do ModelSegmenter têm de ser lineares na cardinalidade — o ``np.isin`` em
arrays de objeto era O(K × G) e travava/derrubava o driver."""
import decimal
import time

import numpy as np
import pandas as pd

from yggdrasil.credit_risk.model.segmenter import _cat_group_masks


def _grupos(uniq, k=5):
    uniq = list(uniq)
    tam = len(uniq) // k + 1
    return [uniq[i:i + tam] for i in range(0, len(uniq), tam)]


def test_mascaras_iguais_a_regra_texto():
    rng = np.random.default_rng(0)
    vals = np.array(["a", "b", "c", "10", "x y"], dtype=object)[rng.integers(0, 5, 2000)]
    s = pd.Series(vals, dtype=object)
    s[rng.random(2000) < 0.1] = None
    grupos = [["a", "10"], ["b"], ["inexistente"], []]
    got = _cat_group_masks(s, grupos)
    for g, m in zip(grupos, got):
        esperado = (s.astype(str).isin(g) & s.notna()).to_numpy()
        assert np.array_equal(m, esperado)


def test_decimal_compara_como_texto():
    s = pd.Series([decimal.Decimal("1.50"), decimal.Decimal("2"), None], dtype=object)
    m = _cat_group_masks(s, [["1.50"], ["2", "1.5"]])
    assert m[0].tolist() == [True, False, False]
    assert m[1].tolist() == [False, True, False]


def test_linear_na_cardinalidade():
    n = 60_000                                      # 60 mil níveis distintos
    s = pd.Series(np.char.add("C", np.arange(n).astype(str)).astype(object))
    grupos = _grupos(s.unique())
    t0 = time.perf_counter()
    m = _cat_group_masks(s, grupos)
    dt = time.perf_counter() - t0
    assert np.array_equal(np.sum(m, axis=0), np.ones(n, dtype=int))   # partição
    # o caminho quadrático levava dezenas de segundos aqui; o por hash, < 1 s
    assert dt < 5, f"_cat_group_masks levou {dt:.1f}s com {n} categorias"
