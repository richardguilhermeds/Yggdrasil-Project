"""Literais de texto e booleanas no SQL gerado (dialeto Spark/Databricks).

- ``sql_texto``: apóstrofo e barra invertida viram chr(39)/chr(92) com ``||`` — no
  Spark SQL 'O''Neil' são DUAS strings coladas ("ONeil");
- ``eh_booleana``: booleana com faltantes chega do toPandas() como object.
"""
import contextlib
import io
import sqlite3
import warnings

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk._common import eh_booleana, sql_texto


@pytest.mark.parametrize("v", ["SP", "", "O'Neil", "'x", "x'", "''", "a\\b", "\\", "x'y\\z'",
                               "{A, O'B}", "ç'ã"])
def test_sql_texto_reconstroi_o_valor(v):
    lit = sql_texto(v)
    con = sqlite3.connect(":memory:")
    con.create_function("chr", 1, chr)
    try:
        got = con.execute(f"SELECT {lit}").fetchone()[0]
    finally:
        con.close()
    assert got == v
    if "'" not in v and "\\" not in v:
        assert lit == f"'{v}'"                 # texto comum: literal de sempre
    if v:                                      # nunca o escape ANSI '' (exceto o texto vazio)
        assert "''" not in lit


def test_eh_booleana():
    assert eh_booleana(pd.Series([True, False]))
    assert eh_booleana(pd.Series([True, None], dtype="boolean"))
    assert eh_booleana(pd.Series([True, False, None], dtype=object))
    assert not eh_booleana(pd.Series(["True", "False"], dtype=object))
    assert not eh_booleana(pd.Series([1, 0]))
    assert not eh_booleana(pd.Series([1.0, np.nan]))


def test_sql_do_model_sem_apostrofo_dobrado_em_derivada():
    """Rótulo da faixa de uma variável CRIADA (create_categorical) embutido no SQL do
    logit/categorização usa o mesmo literal do lado da comparação."""
    from yggdrasil.credit_risk.model import ModelSegmenter
    rng = np.random.default_rng(3)
    n = 3000
    g = rng.choice(["A", "B", "C", "D", "O'Neil"], n).astype(object)
    flag = rng.random(n) < 0.3
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.2 + 0.6 * (g == "C") + 0.5 * flag)))).astype(int)
    df = pd.DataFrame({"garantia": g, "flag": flag.astype(object), "x": rng.normal(size=n),
                       "target": y})
    df.loc[rng.random(n) < 0.05, "flag"] = None          # booleana com faltante (object)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = ModelSegmenter(df, target="target", verbose=False)
        seg.set_manual_bins("garantia", [["A"], ["B"], ["C", "D", "O'Neil"]])
        cat = seg.create_categorical("garantia")
        seg.fit("logistica", features=[cat, "x"], transform="woe")
        sqls = [seg.categorization_sql(table="base"), seg.logit_sql(table="base")]
    for sql in sqls:
        assert "O''Neil" not in sql
        assert "chr(39)" in sql
