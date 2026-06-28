# Árvore de segmentação unificada — `TreeSegmenter`

`yggdrasil.credit_risk.tree` traz **uma única classe** (`TreeSegmenter`) e **uma
única UI** (`TreeSegmenterUI`) que atendem **PD (classificação)** e **LGD
(regressão)** escolhendo o comportamento por `task_type`. Substitui as antigas
classes separadas `SequentialPDSegmenter`/`SequentialLGDSegmenter` (removidas).

```python
from yggdrasil.credit_risk.tree import TreeSegmenter

seg = TreeSegmenter(df, target="target", task_type="classification",  # ou "regression"
                    sample_col="amostra", ref_sample="DES", date_col="dt_ref")
seg.fit_auto(max_depth=3, criterion="optbin")   # ou ks/gini/chi2 ... (clf) · variance/mae/ftest (reg)
seg.leaves(); seg.metrics(); seg.predict(df_novos)
```

## O que muda por `task_type`

| Aspecto | `classification` (PD) | `regression` (LGD) |
|---|---|---|
| Binning ótimo | `OptimalBinning` (binário) | `ContinuousOptimalBinning` |
| IV | WoE binário (Siddiqi) | IV contínuo (desvio médio) |
| Métricas (`metrics`) | KS/AUC/Gini/Acurácia/F1 | MAE/RMSE/R² |
| Cor da árvore | escala dinâmica até a maior taxa | fixa 0–1 |
| Gráficos | ROC/KS/taxa-default/distribuição | boxplot/histograma do alvo |

A maquinaria comum é a mesma: `grow`/`prune`/`merge_*`/`auto_merge`/`collapse`,
PSI/CSI por amostra, IC bootstrap, faltantes em bin própria, `backtest`,
calibração, `monotonicity_report`, persistência JSON (`save`/`load`),
`predict`/`to_pyspark`/`apply_spark` e `log_to_mlflow`.

## Recursos adicionais

- **Critério de split selecionável** (`criterion=` em `fit_auto`/`grow`): `optbin`
  (multi-bin por IV, padrão) ou um split binário CART/CHAID — clf: `gini`,
  `entropy`, `ks`, `iv`, `chi2`; reg: `variance`, `mae`, `ftest`.
- **`suggest_splits(sid, top)`** — TOP-N variáveis para dividir a folha, com nº de
  bins, IV, PSI por amostra (OOT/ESTABILIDADE), `passa_teste` (qui-quadrado / Kruskal)
  e p-valor. Na UI, também há **sugestão de cortes + máx. bins** por variável.
- **`feature_importance()`** — ganho de IV ponderado pela representatividade do nó,
  por variável que entrou na árvore.
- **`to_sql(table=...)`** — régua como `CASE WHEN` copiável.
- **`diff_trees(other)`** — migração de notas, concordância e métricas entre duas versões.
- **`report_pdf(path)`** — relatório PDF (capa, métricas, imagem da árvore, folhas, calibração).
- **`log_to_mlflow(...)`** — loga variáveis, profundidade, PSI por amostra e métricas
  por tarefa, e registra a régua como modelo `pyfunc` (+ artefatos `.txt/.sql/.py/.json`).

## UI interativa

`TreeSegmenterUI(df, task_type=..., ...)` — abas Construir · Análise de variáveis ·
Diagnóstico · Validar & Exportar · Avançado · Histórico, com **tema claro/escuro**
(toggle), discriminação ao vivo (KS/AUC ou R²), preview de split, auto-fit por
critério, auto-merge, e o relatório PDF na aba Histórico.

> Passo a passo executável: [`notebooks/tutoriais/04_tutorial_tree_segmenter.ipynb`](../../notebooks/tutoriais/04_tutorial_tree_segmenter.ipynb).
