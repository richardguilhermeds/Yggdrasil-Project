"""Trava de compatibilidade com a versão mínima de Python declarada.

O ``pyproject.toml`` declara ``requires-python = ">=3.9"``, mas a suíte roda em
3.13 — sintaxe nova demais passa despercebida. Já aconteceu: uma f-string com
aspas repetidas (PEP 701, válida só a partir do 3.12) derrubou o import de
``yggdrasil.credit_risk.model`` inteiro em 3.9–3.11, o que inclui os runtimes
LTS do Databricks até o 15.4. Como os segmentadores são importados de forma
preguiçosa, ``import yggdrasil`` continuava funcionando e nada acusou.

São duas travas porque nenhuma sozinha cobre todo mundo:

* :func:`test_pacote_compila_na_versao_minima` compila o pacote com um
  interpretador antigo DE VERDADE — a checagem completa, que pega qualquer
  sintaxe nova, não só f-string. Depende de haver um instalado; aponte qual em
  ``YGG_PYTHON_MINIMO`` se a descoberta automática não achar.
* :func:`test_sem_fstring_com_aspas_repetidas` roda sempre (em 3.12+) e pega
  especificamente o PEP 701 lendo os tokens da f-string. Cobre o caso concreto
  que já ocorreu mesmo numa máquina que só tem Python novo.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
PACOTE = RAIZ / "yggdrasil"

#: Rodado pelo interpretador antigo: compila todo ``.py`` do pacote e sai com
#: código 1 listando as falhas. Fica como texto porque o interpretador que roda
#: a suíte não é o mesmo que executa isto.
_VERIFICADOR = """
import pathlib, sys
raiz = pathlib.Path(sys.argv[1])
arquivos = sorted(raiz.rglob('*.py'))
falhas = []
for f in arquivos:
    try:
        compile(f.read_text(encoding='utf-8'), str(f), 'exec')
    except SyntaxError as e:
        falhas.append('%s:%s: %s' % (f.relative_to(raiz.parent), e.lineno, e.msg))
print('ARQUIVOS=%d' % len(arquivos))
print('\\n'.join(falhas))
sys.exit(1 if falhas else 0)
"""


def _versao_minima() -> tuple[int, int]:
    """``(maior, menor)`` do ``requires-python`` do ``pyproject.toml``."""
    txt = (RAIZ / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'requires-python\s*=\s*["\'][^0-9]*(\d+)\.(\d+)', txt)
    assert m, "não achei requires-python no pyproject.toml"
    return int(m.group(1)), int(m.group(2))


def _versao_de(cmd: list[str]) -> tuple[int, int] | None:
    """Versão que ``cmd`` de fato executa, ou ``None`` se não roda."""
    try:
        p = subprocess.run([*cmd, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    try:
        maior, menor = p.stdout.strip().split(".")[:2]
        return int(maior), int(menor)
    except ValueError:
        return None


def _interpretador_antigo():
    """Primeiro Python disponível entre a mínima declarada e o 3.11 (inclusive).

    Procura, nesta ordem: ``YGG_PYTHON_MINIMO``, os lançadores da mínima para
    cima e, por último, o ``.venv-spark`` do repositório. A versão é confirmada
    rodando o próprio interpretador — lançador que erra a versão é descartado
    em vez de dar falso positivo.
    """
    minimo = _versao_minima()
    tentativas: list[list[str]] = []
    if os.environ.get("YGG_PYTHON_MINIMO"):
        tentativas.append([os.environ["YGG_PYTHON_MINIMO"]])
    for menor in range(minimo[1], 12):
        if os.name == "nt":
            tentativas.append(["py", f"-3.{menor}"])
        tentativas.append([f"python3.{menor}"])
    venv = RAIZ / ".venv-spark" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if venv.exists():
        tentativas.append([str(venv)])

    for cmd in tentativas:
        v = _versao_de(cmd)
        if v is not None and minimo <= v < (3, 12):
            return cmd, v
    return None, None


def test_pacote_compila_na_versao_minima():
    """Todo ``.py`` do pacote precisa compilar num Python anterior ao 3.12."""
    cmd, versao = _interpretador_antigo()
    if cmd is None:
        minimo = _versao_minima()
        pytest.skip(
            f"nenhum Python entre {minimo[0]}.{minimo[1]} e 3.11 nesta máquina; "
            "aponte um em YGG_PYTHON_MINIMO para rodar a checagem completa "
            "(test_sem_fstring_com_aspas_repetidas ainda cobre o PEP 701)")

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run([*cmd, "-c", _VERIFICADOR, str(PACOTE)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env, timeout=600)
    assert "ARQUIVOS=0" not in p.stdout, f"nenhum arquivo varrido em {PACOTE}"
    assert p.returncode == 0, (
        f"o pacote não compila em Python {versao[0]}.{versao[1]}, dentro do "
        f"requires-python declarado:\n{p.stdout}\n{p.stderr}")


# ----------------------------------------------------------------------
# PEP 701 (aspas repetidas em f-string) — sem depender de Python antigo
# ----------------------------------------------------------------------
_ASPAS = ('"""', "'''", '"', "'")


def _aspa(literal: str) -> str:
    """Delimitador de um literal de string, já sem os prefixos (``f``/``r``…)."""
    i = 0
    while i < len(literal) and literal[i] not in "\"'":
        i += 1
    resto = literal[i:]
    for q in _ASPAS:
        if resto.startswith(q):
            return q
    return ""


def _aspas_repetidas(src: str) -> list[tuple[int, str]]:
    """``[(linha, delimitador)]`` de literais que reusam a aspa de uma f-string
    que os contém — sintaxe válida só a partir do 3.12.

    Depende dos tokens ``FSTRING_START``/``FSTRING_END``, que o próprio PEP 701
    introduziu: antes do 3.12 a f-string inteira vem num único token ``STRING``.
    Pega tanto f-string aninhada (``f"{f"x"}"``) quanto literal comum dentro da
    expressão (``f"{d["k"]}"``).
    """
    achados: list[tuple[int, str]] = []
    pilha: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.FSTRING_START:
            q = _aspa(tok.string)
            if q in pilha:
                achados.append((tok.start[0], q))
            pilha.append(q)
        elif tok.type == tokenize.FSTRING_END:
            if pilha:
                pilha.pop()
        elif tok.type == tokenize.STRING and pilha:
            q = _aspa(tok.string)
            if q in pilha:
                achados.append((tok.start[0], q))
    return achados


@pytest.mark.skipif(sys.version_info < (3, 12),
                    reason="sem tokens de f-string antes do 3.12 — e aqui o próprio "
                           "import do pacote já teria falhado")
def test_sem_fstring_com_aspas_repetidas():
    arquivos = sorted(PACOTE.rglob("*.py"))
    assert arquivos, f"nenhum arquivo varrido em {PACOTE}"
    achados = []
    for f in arquivos:
        for linha, q in _aspas_repetidas(f.read_text(encoding="utf-8")):
            achados.append(f"{f.relative_to(RAIZ)}:{linha}: aspa {q} repetida dentro "
                           "da f-string")
    assert not achados, (
        "f-string com aspas repetidas (PEP 701) — SyntaxError antes do 3.12; "
        "troque o tipo de aspa ou extraia o valor para uma variável:\n"
        + "\n".join(achados))


@pytest.mark.skipif(sys.version_info < (3, 12), reason="idem")
@pytest.mark.parametrize("codigo, ruim", [
    ("""x = f"{d['k']}" """, False),            # aspas alternadas: sempre valeu
    ("""x = f'{d["k"]}' """, False),
    ("""x = f"{f'{y}' if y else ''}" """, False),
    ('''x = f"""{d["k"]}""" ''', False),        # dupla simples não fecha a tripla
    ('''x = f"{f"{y}"}" ''', True),             # f-string aninhada, mesma aspa
    ('''x = f"{d["k"]}" ''', True),             # literal comum, mesma aspa
    ("""x = f'{f"{f'{y}'}"}' """, True),        # repete a aspa de dois níveis acima
])
def test_detector_de_aspas_repetidas(codigo, ruim):
    """O detector tem que concordar com o compilador antigo — os casos ``ruim``
    são exatamente os que o Python 3.11 rejeita."""
    assert bool(_aspas_repetidas(codigo)) is ruim
