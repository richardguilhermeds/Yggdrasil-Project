# Publicando a Yggdrasil no PyPI

Guia de release do pacote. Nome de **distribuição**: `yggdrasil-project`
(o `yggdrasil` já está registrado por outro projeto). Nome de **import**:
continua `yggdrasil` → `from yggdrasil import MLPipeline`.

> Rode tudo a partir da raiz do projeto (onde está o `pyproject.toml`), no seu
> ambiente local — o build precisa acessar `yggdrasil/` (na raiz do repo).

---

## 0. Pré-requisitos (uma vez só)

1. Crie contas e ative 2FA:
   - TestPyPI: https://test.pypi.org/account/register/
   - PyPI: https://pypi.org/account/register/
2. Gere um **API token** em cada uma (Account settings → API tokens). Guarde os
   dois — o de TestPyPI e o de PyPI são diferentes.
3. Instale/atualize as ferramentas de build:

```powershell
python -m pip install --upgrade build twine
```

---

## 1. Build dos artefatos

```powershell
# limpa builds antigos
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue

# gera sdist (.tar.gz) e wheel (.whl) em dist/
python -m build
```

Resultado esperado em `dist/`:

```
yggdrasil_project-0.0.1.tar.gz
yggdrasil_project-0.0.1-py3-none-any.whl
```

Valide os metadados antes de subir:

```powershell
twine check dist/*
```

Ambos devem dar `PASSED`.

---

## 2. Subir no TestPyPI (ensaio)

```powershell
twine upload --repository testpypi dist/*
```

- Usuário: `__token__`
- Senha: o token do **TestPyPI** (começa com `pypi-...`)

Página do projeto: https://test.pypi.org/project/yggdrasil-project/

### Testar a instalação num ambiente limpo

O TestPyPI não hospeda as dependências (numpy, pandas, mlflow...), por isso
aponte o `--extra-index-url` para o PyPI real:

```powershell
python -m venv .venv-test
.\.venv-test\Scripts\Activate.ps1

pip install --index-url https://test.pypi.org/simple/ `
            --extra-index-url https://pypi.org/simple/ `
            yggdrasil-project

python -c "import yggdrasil; print(yggdrasil.__version__)"
```

Saída esperada: `0.0.1`. Depois pode apagar a `.venv-test`.

---

## 3. Subir no PyPI de produção

Quando o ensaio no TestPyPI estiver ok:

```powershell
twine upload dist/*
```

- Usuário: `__token__`
- Senha: o token do **PyPI** (o de produção, não o de teste)

Pronto — qualquer pessoa instala com:

```powershell
pip install yggdrasil-project
```

Página do projeto: https://pypi.org/project/yggdrasil-project/

---

## 4. Releases seguintes

O PyPI **não** deixa reenviar uma versão já publicada. Para cada release:

1. Suba o `version` em dois lugares (mantenha iguais):
   - `pyproject.toml` → `version = "0.0.2"`
   - `yggdrasil/__init__.py` → `__version__ = "0.0.2"`
2. Repita os passos 1 a 3.

> Dica: dá para eliminar a duplicação usando
> `[tool.setuptools.dynamic]` com `version = {attr = "yggdrasil.__version__"}`,
> assim a versão vive só no `__init__.py`. Posso configurar isso se quiser.

---

## Guardar os tokens (opcional, evita colar a cada upload)

Crie `%USERPROFILE%\.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-SEU_TOKEN_DE_PRODUCAO

[testpypi]
username = __token__
password = pypi-SEU_TOKEN_DE_TESTE
```

---

## Alternativa moderna: Trusted Publishing (GitHub Actions)

Em vez de tokens, o PyPI aceita publicação direta de um workflow do GitHub
(sem senha, via OIDC). Bom para automatizar release por tag. Posso montar o
`.github/workflows/release.yml` se você quiser seguir por aí.
