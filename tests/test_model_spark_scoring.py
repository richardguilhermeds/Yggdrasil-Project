"""
Regressão do getter de broadcast da escoragem distribuída (ModelSegmenter).

Em Spark Connect (Databricks serverless/shared, DBR 14.3+, ou Databricks Connect)
a sessão NÃO expõe ``sparkContext`` e ``spark.sparkContext.broadcast(...)`` levanta
``PySparkAttributeError`` — quebrava a etapa "preparar escoragem distribuída
(broadcast do modelo)". ``_scorer_broadcast_getter`` deve cair para closure nesse
caso e usar broadcast no cluster clássico.

Testes LEVES com sessões *fake* — não exigem pyspark nem um cluster.
"""
from __future__ import annotations

from yggdrasil.credit_risk.model.segmenter import _scorer_broadcast_getter


class _FakeBroadcast:
    def __init__(self, value):
        self.value = value


class _FakeSparkContext:
    def __init__(self):
        self.broadcasted = []

    def broadcast(self, obj):
        self.broadcasted.append(obj)
        return _FakeBroadcast(obj)


class _SparkClassico:
    """Sessão clássica: expõe sparkContext com broadcast."""
    def __init__(self):
        self.sparkContext = _FakeSparkContext()


class _SparkConnect:
    """Sessão Spark Connect: acessar sparkContext levanta (como no runtime real)."""
    @property
    def sparkContext(self):
        raise AttributeError(
            "[JVM_ATTRIBUTE_NOT_SUPPORTED] Attribute 'sparkContext' is not "
            "supported in Spark Connect")


def test_getter_usa_broadcast_no_cluster_classico():
    spark = _SparkClassico()
    scorer = object()
    get = _scorer_broadcast_getter(spark, scorer)
    assert spark.sparkContext.broadcasted == [scorer]   # broadcast FOI usado
    assert get() is scorer                               # e entrega o mesmo scorer


def test_getter_cai_para_closure_no_spark_connect():
    spark = _SparkConnect()
    scorer = object()
    get = _scorer_broadcast_getter(spark, scorer)        # não pode levantar AttributeError
    assert get() is scorer                               # closure entrega o scorer


def test_getter_fallback_tambem_quando_broadcast_falha():
    """Mesmo com sparkContext presente, se .broadcast falhar cai para closure."""
    class _SCquebrado:
        def broadcast(self, obj):
            raise RuntimeError("broadcast indisponível")

    class _Spark:
        sparkContext = _SCquebrado()

    scorer = object()
    get = _scorer_broadcast_getter(_Spark(), scorer)
    assert get() is scorer
