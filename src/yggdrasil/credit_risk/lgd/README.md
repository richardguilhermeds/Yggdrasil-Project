# 🌳 Yggdrasil — `credit_risk.lgd`

Segmentação de **LGD** (Loss Given Default) para risco de crédito, sob
**CMN 4.966/2021** e **IFRS 9** — construída para rodar em **Spark/Databricks**.

Este é um módulo do repositório [Yggdrasil-Project](https://github.com/zheage/Yggdrasil-Project),
na raiz de domínio `yggdrasil.credit_risk`.

## Instalação

Na raiz do repositório (layout `src/`):

```bash
pip install -e .            # núcleo: numpy, pandas, optbinning
pip install -e ".[ui]"      # + interface interativa (ipywidgets)
pip install -e ".[all]"     # + ui, mlflow, pyspark
```

Ou direto do GitHub:

```bash
pip install "git+https://github.com/zheage/Yggdrasil-Project.git"
```

## Uso

```python
import pandas as pd
from yggdrasil.credit_risk.lgd import SequentialLGDSegmenter

seg = SequentialLGDSegmenter(
    df, target="lgd", sample_col="amostra", ref_sample="DES",
    feature_labels={"ltv": "LTV", "garantia": "garantia"},
)

seg.fit_auto(max_depth=3)        # árvore inicial gulosa por IV
seg.auto_merge(alpha=0.05)       # funde folhas-irmãs indistinguíveis (p > alpha)
seg.leaves()                     # folhas com nota, LGD, representatividade e PSI
print(seg.tree())                # árvore em texto, colorida por LGD no painel

seg.csi()                        # CSI por variável (estabilidade das entradas DES→OOT)
seg.csi_detalhe()                # contribuição de cada faixa ao CSI

seg.save("arvore_lgd.json")      # salva a árvore (estrutura) em JSON
seg = SequentialLGDSegmenter.load("arvore_lgd.json", df)   # recarrega e reaplica

regua = seg.predict(df_novos)    # aplica a régua em pandas
print(seg.to_pyspark())          # gera a régua como F.when().otherwise() p/ Spark
```

### Interface interativa (Jupyter/Databricks)

```python
from yggdrasil.credit_risk.lgd import LGDSegmenterUI
ui = LGDSegmenterUI(df, target="lgd", sample_col="amostra", ref_sample="DES",
                    feature_labels={"ltv": "LTV", "garantia": "garantia"})
ui  # painel: preview, criar/fundir/recolher folha, auto-fit, auto-fundir, podar,
    #         desfazer/refazer, salvar/carregar (JSON), CSI por variável, bootstrap, MLflow
```

### Registro no MLflow / Unity Catalog

```python
seg.log_to_mlflow(
    registered_model_name="catalogo.schema.lgd_segmentacao",
    registry_uri="databricks-uc",
)
```

## O que vem na caixa

- Binning **ótimo** (OptBinning) ou **manual**, numérico e categórico
- **Faltantes (NaN) em bin própria** — nada é descartado no split
- Notas por folha, **IV**, **PSI por amostra** (DES/OOT), **IC bootstrap**
- **CSI por variável** (`csi`/`csi_detalhe`) — estabilidade de cada característica de entrada
- **Salvar/carregar a árvore em JSON** (`save`/`load`, `to_dict`/`from_dict`) — portável entre máquinas
- **Auto-merge** (`auto_merge`) — funde automaticamente folhas-irmãs indistinguíveis (teste de hipótese)
- `predict` (pandas) e `to_pyspark` (Spark) com a mesma régua
- `fit_auto`, `suggest_split`, `prune`, `merge_leaf`, `collapse`
- UI com **desfazer/refazer** de splits, auto-fundir e persistência em JSON
- `log_to_mlflow` com assinatura e versão no Model Registry (Unity Catalog)

## Documentação

Tutorial completo em [`docs/credit-risk/lgd-segmenter.md`](../../docs/credit-risk/lgd-segmenter.md)
e notebook em [`notebooks/credit_risk/01_lgd_segmenter.ipynb`](../../notebooks/credit_risk/01_lgd_segmenter.ipynb).

## Testes

```bash
pip install -e ".[dev]"
pytest tests/credit_risk
```
