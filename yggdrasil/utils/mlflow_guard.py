"""
yggdrasil.utils.mlflow_guard
============================
Impede que o **autologging** do MLflow abra um run a cada ``fit`` interno do
yggdrasil.

Por que existe
--------------
Nos runtimes ML do Databricks o *Databricks Autologging* vem **ligado por
padrão**: o MLflow aplica um patch nos estimadores do ``scikit-learn`` e abre um
run novo a cada chamada de ``fit`` — inclusive nas que não são "o modelo". O
yggdrasil ajusta muitos estimadores sklearn de bastidor:

* o ``optbinning`` roda uma ``DecisionTree`` por variável no pré-binning
  (:func:`yggdrasil.credit_risk._common.fit_optbinning_splits`) — abrir a
  ``ModelSegmenterUI`` com 80 candidatas dispara 80+ runs, cada um com ida e
  volta ao tracking server, e a interface só aparece quando o último termina;
* o tuning do Optuna ajusta um pipeline por trial (× folds), o backward
  elimination um por passo, o VIF uma regressão por coluna, a estratégia de
  ratings uma árvore.

Nenhum desses ajustes é um experimento: o registro de governança do pacote é
explícito (``log_to_mlflow``, ``tune_optuna(log_mlflow=True)``), com tags,
métricas por amostra e o relatório em abas. Os runs do autolog só poluem o
experimento e custam tempo de parede.

O que o guard faz
-----------------
:func:`sem_autolog` liga, enquanto o bloco roda, a mesma chave global que o
próprio MLflow usa para não se autologar
(``mlflow.utils.autologging_utils._AUTOLOGGING_GLOBALLY_DISABLED``): os patches
de autolog chamam a função original sem abrir run. O que é **explícito**
(``mlflow.start_run``, ``log_model``, ``log_metric``) não é afetado — o
``log_mlflow=True`` do tuning continua registrando trials e modelo.

É seguro por construção: no-op quando o ``mlflow`` não está carregado na sessão
(nunca importa o mlflow só para isso), quando a chave interna não existe na
versão instalada, e reentrante — restaura o valor anterior, não um ``False``
fixo, para blocos aninhados e para a thread de fundo do tuning.

Uso
---
>>> from yggdrasil.utils import sem_autolog
>>> with sem_autolog():          # bloco
...     modelo.fit(X, y)
>>> @sem_autolog()               # ou decorando o método que ajusta
... def fit(self, ...): ...

Voltar ao comportamento do ambiente (autolog do Databricks também nos ajustes
internos): ``YGGDRASIL_MLFLOW_AUTOLOG=1`` no ambiente, ou em código

>>> from yggdrasil.utils import set_mlflow_autolog
>>> set_mlflow_autolog(True)
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from typing import Optional

__all__ = ["sem_autolog", "set_mlflow_autolog", "mlflow_autolog_status"]

#: Variável de ambiente que devolve o autolog do ambiente aos ajustes internos.
ENV_AUTOLOG = "YGGDRASIL_MLFLOW_AUTOLOG"

_MODULO = "mlflow.utils.autologging_utils"
_FLAG = "_AUTOLOGGING_GLOBALLY_DISABLED"
_VERDADEIROS = {"1", "true", "yes", "sim", "on"}

#: ``None`` = decide pelo ambiente; ``True``/``False`` = escolha explícita do
#: usuário via :func:`set_mlflow_autolog` (vence a variável de ambiente).
_autolog_permitido: Optional[bool] = None

# A chave do MLflow é global (não é por thread) e o yggdrasil roda ajustes em
# threads de fundo (tuning, backward) enquanto a UI segue viva na principal.
# Um contador sob lock faz a chave voltar ao valor original só quando o ÚLTIMO
# bloco sai — com um simples salvar/restaurar por bloco, dois guards que
# terminassem fora de ordem deixariam o autolog do usuário desligado para
# sempre.
_lock = threading.Lock()
_profundidade = 0
_valor_original = False


def set_mlflow_autolog(ativo: bool) -> None:
    """Permite (``True``) ou suprime (``False``) o autologging do ambiente nos
    ajustes internos do yggdrasil. Vence a variável ``YGGDRASIL_MLFLOW_AUTOLOG``.
    """
    global _autolog_permitido
    _autolog_permitido = bool(ativo)


def _permitido() -> bool:
    if _autolog_permitido is not None:
        return _autolog_permitido
    return os.environ.get(ENV_AUTOLOG, "").strip().lower() in _VERDADEIROS


def _autologging_utils():
    """Módulo de autologging do MLflow — **só** se o mlflow já estiver na sessão.

    Importar o mlflow custa segundos e é o oposto do que este módulo quer; quando
    há autolog ativo (Databricks) o mlflow já está carregado, que é justamente o
    caso que importa. Devolve ``None`` quando não há nada a suprimir.
    """
    if "mlflow" not in sys.modules:
        return None
    mod = sys.modules.get(_MODULO)
    if mod is None:
        try:    # mlflow já em memória: o import do submódulo é barato
            import mlflow.utils.autologging_utils as mod  # noqa: PLC0415
        except Exception:                                 # noqa: BLE001
            return None
    return mod if hasattr(mod, _FLAG) else None


@contextmanager
def sem_autolog():
    """Suprime o autologging do MLflow dentro do bloco (ou do método decorado).

    Serve como context manager e como decorator (``@sem_autolog()``). Produz
    ``True`` quando a supressão de fato entrou em vigor e ``False`` quando é
    no-op (sem mlflow na sessão, versão sem a chave interna, ou autolog
    explicitamente permitido).
    """
    global _profundidade, _valor_original
    mod = None if _permitido() else _autologging_utils()
    if mod is None:
        yield False
        return
    with _lock:
        if _profundidade == 0:
            _valor_original = getattr(mod, _FLAG)
        _profundidade += 1
        setattr(mod, _FLAG, True)
    try:
        yield True
    finally:
        with _lock:
            _profundidade -= 1
            if _profundidade == 0:      # último bloco: devolve o valor do usuário
                setattr(mod, _FLAG, _valor_original)


def mlflow_autolog_status() -> dict:
    """Diagnóstico do autologging na sessão — ``{'mlflow_carregado',
    'integracoes', 'suprimido'}``.

    ``integracoes`` lista os *flavors* com autolog habilitado (ex.: ``sklearn``,
    ``xgboost`` — no Databricks vêm ligados por padrão) e ``suprimido`` diz se o
    yggdrasil vai desligá-los nos ajustes internos. Não importa o mlflow.
    """
    mod = _autologging_utils()
    integracoes: list = []
    if mod is not None:
        try:
            for nome, cfg in dict(getattr(mod, "AUTOLOGGING_INTEGRATIONS", {})).items():
                if isinstance(cfg, dict) and not cfg.get("disable", True):
                    integracoes.append(nome)
        except Exception:                                 # noqa: BLE001
            pass
    return {"mlflow_carregado": "mlflow" in sys.modules,
            "integracoes": sorted(integracoes),
            "suprimido": mod is not None and not _permitido()}
