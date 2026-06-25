# 🌳 Yggdrasil — `credit_risk.pd`

Segmentação de **PD** (Probability of Default) — modelos de **classificação** —
para risco de crédito, sob **CMN 4.966/2021** e **IFRS 9**, construída para rodar
em **Spark/Databricks**.

É o gêmeo de classificação do [`credit_risk.lgd`](../lgd/README.md): mesma UI e
mesmo fluxo (binning ótimo, notas por folha, PSI, calibração, MLflow), mas o alvo
é **binário** (0 = adimplente, 1 = default) e a avaliação é de **discriminação**
— **KS, ROC/AUC, Gini, Acurácia e F1**.

## Instalação

```bash
pip install -e .                 # núcleo (numpy, pandas, optbinning, scikit-learn, mlflow, …)
pip install -e ".[ui]"           # + interface interativa (ipywidgets) → PDSegmenterUI
pip install -e ".[ui,spark,dev]" # tudo: UI + PySpark + testes/Jupyter
```

## Uso

```python
import pandas as pd
from yggdrasil.credit_risk.pd import SequentialPDSegmenter

seg = SequentialPDSegmenter(
    df, target="target", sample_col="amostra", ref_sample="DES",
    feature_labels={"score": "score externo", "garantia": "garantia"},
)

seg.fit_auto(max_depth=3)        # árvore inicial gulosa por IV (WoE binário)
seg.auto_merge(alpha=0.05)       # funde folhas-irmãs indistinguíveis (p > alpha)
seg.merge_missing(folha)         # junta o nó de faltantes (NaN) num bin populado
seg.leaves()                     # folhas com nota, PD (taxa de default), repr. e PSI
seg.metrics()                    # KS, AUC, Gini, Acurácia e F1 por amostra
print(seg.tree())                # árvore em texto, colorida por PD no painel
seg.plot_tree(save_path="arvore_pd.png")   # imagem da árvore (PD média e % por folha)

seg.plot_roc()                   # curva ROC por amostra (AUC + Gini)
seg.plot_ks()                    # curva KS (acumuladas de bons × maus pelo score)
seg.plot_leaf_badrate()          # taxa de default por folha (IC de Wilson)
seg.plot_score_distribution()    # distribuição do score por classe (separação)

seg.csi()                        # CSI por variável (estabilidade das entradas DES→OOT)
seg.backtest("dt_ref")           # PD prevista × realizada por safra
seg.monotonicity_report()        # monotonicidade da PD nas notas (por amostra)
seg.plot_calibration()           # calibração prevista (DES) × realizada (OOT) por folha
seg.validation_report("rel.md", time_col="dt_ref")   # relatório de validação (Markdown + imagens)

seg.save("arvore_pd.json")       # salva a árvore (estrutura) em JSON
seg = SequentialPDSegmenter.load("arvore_pd.json", df)   # recarrega e reaplica

regua = seg.predict(df_novos)    # aplica a régua em pandas (segmento, nota e PD)
print(seg.to_pyspark())          # gera a régua como F.when().otherwise() p/ Spark
sdf2 = seg.apply_spark(sdf)      # aplica a régua direto numa tabela Spark
```

### Interface interativa (Jupyter/Databricks)

```python
from yggdrasil.credit_risk.pd import PDSegmenterUI
ui = PDSegmenterUI(df, target="target", sample_col="amostra", ref_sample="DES",
                   feature_labels={"score": "score externo", "garantia": "garantia"})
ui  # painel: KS/AUC ao vivo, preview, criar/fundir/recolher folha, auto-fit,
    #         auto-fundir, podar, desfazer/refazer, ROC/KS, calibração, MLflow
```

### Registro no MLflow / Unity Catalog

```python
seg.log_to_mlflow(
    registered_model_name="catalogo.schema.pd_segmentacao",
    registry_uri="databricks-uc",
)
```

## O que vem na caixa

- Binning **ótimo** (OptBinning, alvo binário) ou **manual**, numérico e categórico
- **Faltantes (NaN) em bin própria** — nada é descartado no split
- Notas por folha, **IV (WoE binário)**, **PSI por amostra** (DES/OOT), **IC bootstrap** da taxa de default
- **Discriminação** (`metrics`): **KS, AUC (ROC), Gini, Acurácia e F1** por amostra · curvas **ROC** (`plot_roc`) e **KS** (`plot_ks`)
- **Validação regulatória**: backtesting por safra (`backtest`), **calibração** prevista×realizada (`plot_calibration`/`calibration_table`) e **monotonicidade** das notas (`monotonicity_report`)
- **Relatório de validação** (`validation_report`) — Markdown com árvore, folhas, PSI, CSI, discriminação, calibração e backtest (+ imagens)
- **CSI por variável** (`csi`/`csi_detalhe`) — estabilidade de cada característica de entrada
- **Salvar/carregar a árvore em JSON** (`save`/`load`, `to_dict`/`from_dict`) — portável entre máquinas
- **Auto-merge** (`auto_merge`) — funde automaticamente folhas-irmãs indistinguíveis (teste de hipótese)
- **Juntar faltantes** (`merge_missing`) — agrupa o nó de faltantes (NaN) com um bin populado (regra "bin OU faltante")
- **Imagem da árvore** (`plot_tree`) — figura matplotlib com PD média, % e nota por folha, colorida pela PD
- **Qualidade dos segmentos** — taxa de default por folha (`plot_leaf_badrate`, com IC de Wilson) e distribuição do score por classe (`plot_score_distribution`)
- `predict` (pandas), `to_pyspark` (gera o código) e **`apply_spark`** (aplica a régua direto numa tabela Spark) com a mesma régua
- `fit_auto`, `suggest_split`, `prune`, `merge_leaf`, `collapse`
- UI com **desfazer/refazer** de splits, auto-fundir e persistência em JSON
- `log_to_mlflow` com assinatura e versão no Model Registry (Unity Catalog)

## Diferenças em relação ao `credit_risk.lgd`

| Aspecto | `lgd` (regressão) | `pd` (classificação) |
| --- | --- | --- |
| Alvo | contínuo em [0, 1] (LGD) | binário 0/1 (default) |
| Binning ótimo | `ContinuousOptimalBinning` | `OptimalBinning` |
| Valor por folha | LGD médio | taxa de default (PD) |
| IV | contínuo (desvio absoluto ponderado) | WoE binário (escala de Siddiqi) |
| Métricas da régua | MAE, RMSE, R² | **KS, AUC, Gini, Acurácia, F1** |
| Gráficos próprios | boxplot do LGD, histograma do LGD | **ROC, KS**, taxa de default por folha, distribuição do score |
| Régua / colunas | `segmento_lgd`, `nota_lgd`, `lgd_regua` | `segmento_pd`, `nota_pd`, `pd_regua` |

## Documentação

Tutorial completo, executável e com saídas reais, em
[`docs/credit-risk/pd-segmenter.md`](../../../../docs/credit-risk/pd-segmenter.md) —
exemplo ponta a ponta de PD de crédito ao consumidor (binning manual + ótimo,
poda, notas, PSI, KS/AUC/Gini, ROC/KS, IV binário, scoring e validação).

## Testes

```bash
pip install -e ".[dev]"
pytest tests/test_pd_segmenter.py
```
