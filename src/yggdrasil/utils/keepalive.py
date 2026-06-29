"""Mantém um cluster Databricks ativo enquanto uma interface/notebook está aberta.

O Databricks encerra o cluster após alguns minutos de **inatividade** (sem jobs
Spark em execução). Quando você está só interagindo com uma UI de ipywidgets — sem
rodar células — o cluster não vê atividade e desliga.

:class:`ClusterKeepAlive` resolve isso disparando, numa *thread* daemon, um **job
Spark mínimo** (``spark.range(1).count()``) a cada intervalo, reiniciando o
cronômetro de inatividade. Funciona dentro do Databricks (que injeta a
``SparkSession`` ativa) e em qualquer ambiente com Spark; fora disso, vira um
no-op seguro (``has_spark()`` devolve ``False``).

Uso rápido
----------
>>> from yggdrasil.utils import keep_cluster_alive
>>> ka = keep_cluster_alive(interval_seconds=120)   # liga (a cada 2 min)
>>> # ... trabalhe na interface ...
>>> ka.stop()                                        # desligue ao terminar

Ou como context manager:

>>> with ClusterKeepAlive(interval_seconds=120):
...     ...  # cluster mantido ativo dentro do bloco

**Dica:** use um intervalo MENOR que o tempo de auto-término do cluster (ex.: se o
cluster desliga após 10 min, pingue a cada 2–5 min).
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

__all__ = ["ClusterKeepAlive", "keep_cluster_alive", "stop_keep_alive"]


class ClusterKeepAlive:
    """Dispara um job Spark mínimo periodicamente para o cluster não desligar.

    Parameters
    ----------
    spark:
        ``SparkSession`` a usar. Se ``None``, é resolvida automaticamente (sessão
        ativa do PySpark ou o global ``spark`` que o Databricks injeta).
    interval_seconds:
        Intervalo entre pings, em segundos (mínimo 20). Use menos que o tempo de
        auto-término do cluster.
    ping_fn:
        Função opcional ``ping_fn(spark)`` para o "toque" de atividade. O padrão
        roda ``spark.range(1).count()`` — um job Spark real (não só driver-local),
        que é o que conta como atividade para o auto-término.
    verbose:
        Imprime cada ping (para depuração).
    """

    def __init__(self, spark=None, interval_seconds: int = 120,
                 ping_fn: Optional[Callable] = None, verbose: bool = False):
        self.interval = max(20, int(interval_seconds))
        self.verbose = verbose
        self._spark = spark
        self._ping_fn = ping_fn
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.pings = 0
        self.last_ok: Optional[bool] = None
        self.last_error: Optional[str] = None

    # -- resolução da SparkSession -------------------------------------
    def _resolve_spark(self):
        if self._spark is not None:
            return self._spark
        try:
            from pyspark.sql import SparkSession
        except Exception:
            return None
        try:
            s = SparkSession.getActiveSession()
        except Exception:
            s = None
        if s is None:                       # Databricks injeta `spark` como global
            import builtins
            s = getattr(builtins, "spark", None)
        return s

    def has_spark(self) -> bool:
        """``True`` se há uma SparkSession a manter ativa (senão é no-op)."""
        return self._resolve_spark() is not None

    # -- ping / loop ---------------------------------------------------
    def _ping(self) -> bool:
        spark = self._resolve_spark()
        if spark is None:
            self.last_ok, self.last_error = False, "sem SparkSession ativa"
            return False
        try:
            if self._ping_fn is not None:
                self._ping_fn(spark)
            else:
                spark.range(1).count()      # job Spark real → reseta a inatividade
            self.pings += 1
            self.last_ok, self.last_error = True, None
            return True
        except Exception as e:              # transitório: não derruba a thread
            self.last_ok = False
            self.last_error = f"{type(e).__name__}: {e}"
            return False

    def _loop(self):
        # wait() devolve True assim que stop é setado → para imediatamente
        while not self._stop.wait(self.interval):
            self._ping()
            if self.verbose:
                print(f"[keepalive] ping #{self.pings} ok={self.last_ok}")

    # -- controle ------------------------------------------------------
    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> "ClusterKeepAlive":
        """Inicia a thread de keepalive (idempotente). Faz um ping imediato para
        já zerar o cronômetro de inatividade."""
        if self.running:
            return self
        self._ping()                        # toque imediato
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="yggdrasil-keepalive")
        self._thread.start()
        if self.verbose:
            print(f"[keepalive] iniciado (a cada {self.interval}s)")
        return self

    def stop(self) -> "ClusterKeepAlive":
        """Para a thread de keepalive (idempotente)."""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None
        if self.verbose:
            print("[keepalive] parado")
        return self

    def __enter__(self) -> "ClusterKeepAlive":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# Instância única de conveniência para o atalho de módulo.
_GLOBAL: Optional[ClusterKeepAlive] = None


def keep_cluster_alive(interval_seconds: int = 120, spark=None,
                       verbose: bool = True) -> ClusterKeepAlive:
    """Liga um keepalive global (reutiliza o existente) e o devolve.

    Guarde o retorno e chame ``.stop()`` ao terminar — ou use
    :func:`stop_keep_alive`. Veja :class:`ClusterKeepAlive`."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = ClusterKeepAlive(spark=spark, interval_seconds=interval_seconds,
                                   verbose=verbose)
    else:
        _GLOBAL.interval = max(20, int(interval_seconds))
    if verbose and not _GLOBAL.has_spark():
        print("[keepalive] nenhuma SparkSession ativa — sem efeito fora do Databricks/Spark.")
    return _GLOBAL.start()


def stop_keep_alive() -> None:
    """Para o keepalive global iniciado por :func:`keep_cluster_alive`."""
    global _GLOBAL
    if _GLOBAL is not None:
        _GLOBAL.stop()
