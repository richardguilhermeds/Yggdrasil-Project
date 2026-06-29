"""Testes do keepalive de cluster (yggdrasil.utils.keepalive).

Sem Spark real: usa um fake que conta os pings. Cobre has_spark (no-op fora do
Spark), ping com fake, ciclo start/stop e o context manager.
"""
from __future__ import annotations

import pytest

from yggdrasil.utils import ClusterKeepAlive, keep_cluster_alive, stop_keep_alive


class _FakeDF:
    def __init__(self, parent):
        self._parent = parent

    def count(self):
        self._parent.pings += 1
        return 1


class _FakeSpark:
    """Imita o mínimo de SparkSession usado pelo keepalive: spark.range(1).count()."""
    def __init__(self):
        self.pings = 0

    def range(self, n):
        return _FakeDF(self)


def test_sem_spark_eh_noop():
    ka = ClusterKeepAlive(spark=None, interval_seconds=20)
    if ka.has_spark():       # outra suíte pode ter deixado uma SparkSession ativa
        pytest.skip("há uma SparkSession ativa no ambiente de teste")
    # sem SparkSession ativa, has_spark é False e o ping falha sem levantar erro
    assert ka._ping() is False
    assert ka.last_ok is False and ka.last_error


def test_ping_com_fake_spark():
    fake = _FakeSpark()
    ka = ClusterKeepAlive(spark=fake, interval_seconds=20)
    assert ka.has_spark() is True
    assert ka._ping() is True
    assert fake.pings == 1 and ka.pings == 1 and ka.last_ok is True


def test_start_stop_lifecycle():
    fake = _FakeSpark()
    ka = ClusterKeepAlive(spark=fake, interval_seconds=20)
    assert ka.running is False
    ka.start()
    assert ka.running is True
    assert fake.pings >= 1            # ping imediato no start
    ka.stop()
    assert ka.running is False
    # idempotente
    ka.stop(); ka.start(); ka.start()
    assert ka.running is True
    ka.stop()


def test_context_manager():
    fake = _FakeSpark()
    with ClusterKeepAlive(spark=fake, interval_seconds=20) as ka:
        assert ka.running is True
    assert ka.running is False


def test_intervalo_minimo():
    # nunca pinga mais rápido que 20s (evita martelar o cluster)
    assert ClusterKeepAlive(interval_seconds=1).interval == 20
    assert ClusterKeepAlive(interval_seconds=300).interval == 300


def test_atalho_global():
    fake = _FakeSpark()
    ka = keep_cluster_alive(interval_seconds=45, spark=fake, verbose=False)
    try:
        assert ka.running is True and ka.interval == 45
        # reusa a mesma instância global
        assert keep_cluster_alive(interval_seconds=60, spark=fake, verbose=False) is ka
    finally:
        stop_keep_alive()
    assert ka.running is False
