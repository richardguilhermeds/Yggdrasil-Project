"""Paridade da régua da TreeSegmenter entre Python (predict) e os caminhos de
escoragem fora do pandas (to_sql, to_pyspark, apply_spark), linha a linha.

Regressões cobertas:
- variável BOOLEANA: o Spark faz CAST(bool AS STRING) = "true" (minúsculo), então a
  comparação como texto com 'True' deixava TODA linha sem folha;
- categoria com APÓSTROFO: no Spark SQL 'O''Neil' são duas strings coladas
  ("ONeil") — o literal agora usa chr(39).
O SQL da árvore é dialeto Spark/Databricks (CAST(x AS STRING)); a execução dele é
conferida no Spark local (testes pulados sem pyspark funcional).
"""
import contextlib
import decimal
import io
import warnings

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.tree import TreeSegmenter


def _base(n=4000, seed=7, flag_objeto=False):
    rng = np.random.default_rng(seed)
    renda = rng.gamma(2, 1500, n)
    renda[rng.random(n) < 0.10] = np.nan
    flag = rng.random(n) < 0.3
    uf = rng.choice(["SP", "RJ", "MG", "BA", "O'Neil"], n).astype(object)
    uf[rng.random(n) < 0.05] = None
    taxa = np.array([decimal.Decimal(f"{v:.2f}") for v in rng.choice([0.5, 1.25, 2.0, 3.75], n)],
                    dtype=object)
    logit = (-1.4 - 0.0004 * np.nan_to_num(renda, nan=3000) + 0.8 * flag
             + 0.4 * (uf == "BA") + 0.3 * (uf == "O'Neil") + 0.2 * (taxa.astype(float) > 1.5))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    df = pd.DataFrame({"renda": renda, "flag": flag, "uf": uf, "taxa": taxa, "target": y})
    df["amostra"] = np.where(rng.random(n) < 0.7, "DES", "OOT")
    if flag_objeto:
        # booleana com faltante como chega do toPandas(): object com True/False/None
        f = df["flag"].astype(object)
        f[rng.random(n) < 0.05] = None
        df["flag"] = f
    return df


def _sem_decimal(base):
    """Decimal → float (SQLite/createDataFrame não recebem decimal.Decimal misturado)."""
    for c in base.columns:
        if base[c].map(lambda v: isinstance(v, decimal.Decimal)).any():
            base[c] = base[c].map(lambda v: None if v is None or pd.isna(v) else float(v))
    return base


@pytest.fixture(scope="module", params=["bool", "objeto_com_none"])
def arvore(request):
    df = _base(flag_objeto=request.param == "objeto_com_none")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = TreeSegmenter(df, target="target", sample_col="amostra", ref_sample="DES")
        seg.grow("flag", splits=[["True"], ["False"]])     # raiz: booleana
        if request.param == "objeto_com_none":
            assert df["flag"].dtype == object and df["flag"].isna().any()
        folhas = [k for k, s in seg.segments.items() if s["is_leaf"]]
        seg.grow("renda", splits=[600.0, 2400.0], only_segments=[folhas[0]])
        folhas = [k for k, s in seg.segments.items() if s["is_leaf"]]
        seg.grow("uf", splits=[["SP", "RJ"], ["MG", "BA"], ["O'Neil"]],
                 only_segments=[folhas[-1]])
    feats = seg.regua_features()
    py = seg.predict(df[feats])["segmento"]
    return seg, df, feats, py.astype(object).where(py.notna(), None).tolist()


def test_sql_e_pyspark_usam_literal_booleano_e_chr39(arvore):
    seg, *_ = arvore
    sql = seg.to_sql(table="base")
    assert "CAST(flag AS STRING)" not in sql
    assert "CAST(flag AS BOOLEAN) = TRUE" in sql and "CAST(flag AS BOOLEAN) = FALSE" in sql
    assert "chr(39)" in sql and "'O''Neil'" not in sql
    code = seg.to_pyspark()
    assert 'F.col("flag").cast("string")' not in code
    assert "F.lit(True)" in code and "F.lit(False)" in code


@pytest.fixture(scope="module")
def spark():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession
    try:
        s = SparkSession.builder.master("local[1]").appName("ygg-paridade").getOrCreate()
    except Exception as e:  # noqa: BLE001 — ambiente sem Spark funcional
        pytest.skip(f"Spark local indisponível: {e}")
    # aquece o worker PYTHON (range().collect() fica na JVM): no Windows a 1ª
    # conexão do worker às vezes falha ("failed to connect back") — tenta de novo
    for tentativa in range(3):
        try:
            s.sparkContext.parallelize([1, 2], 1).map(lambda x: x + 1).collect()
            break
        except Exception as e:  # noqa: BLE001
            if tentativa == 2:
                s.stop()
                pytest.skip(f"worker Python do Spark local não sobe: {str(e)[:200]}")
    yield s
    s.stop()


def _spark_df(spark, df, feats, nan_literal=False):
    from pyspark.sql import functions as F
    base = _sem_decimal(df[feats].copy())
    base["_id"] = np.arange(len(df))
    sdf = spark.createDataFrame(base.astype(object).where(base.notna(), None))
    if nan_literal:
        sdf = sdf.withColumn("renda", F.when(F.col("renda").isNull(), F.lit(float("nan")))
                             .otherwise(F.col("renda")))
    return sdf


def _folhas(sdf_out):
    return sdf_out.select("_id", "segmento").toPandas().sort_values("_id")["segmento"].map(
        lambda v: None if pd.isna(v) else v).tolist()


@pytest.mark.parametrize("nan_literal", [False, True])
def test_apply_spark_bate_com_o_python(arvore, spark, nan_literal):
    seg, df, feats, py = arvore
    assert _folhas(seg.apply_spark(_spark_df(spark, df, feats, nan_literal))) == py


def test_to_pyspark_e_to_sql_no_spark_batem_com_o_python(arvore, spark):
    seg, df, feats, py = arvore
    sdf = _spark_df(spark, df, feats)
    ns = {}
    exec(seg.to_pyspark(func_name="aplicar_regua"), ns)
    assert _folhas(ns["aplicar_regua"](sdf)) == py
    sdf.createOrReplaceTempView("base_paridade")
    sql = seg.to_sql(table="base_paridade").rstrip().rstrip(";")
    assert _folhas(spark.sql(sql)) == py
