# 🌳 Yggdrasil — `feature_selection`

Módulo Spark-first de **seleção de features** do Yggdrasil. Ele recebe um Spark DataFrame que segue o contrato de colunas do projeto (`feat_*`, alvo e, opcionalmente, coluna de amostra), organiza as candidatas em **books** (origens de dados, ex.: serasa, bvs) e roda, por book, um pipeline de filtros duros, indicadores de importância, **Boruta** e **consenso**. A esteira é **independente**: não faz parte da EDA nem do pipeline de treino do modelo — apenas seleciona variáveis e produz tabelas, painéis e um ranking global.

## Índice

1. [Visão geral](#1-visão-geral)
2. [Instalação](#2-instalação)
3. [Contrato de dados (`ColumnConfig`)](#3-contrato-de-dados-columnconfig)
4. [Quickstart mínimo](#4-quickstart-mínimo)
5. [Books: as 3 formas](#5-books-as-3-formas)
6. [O pipeline por book, passo a passo](#6-o-pipeline-por-book-passo-a-passo)
7. [Indicadores de importância](#7-indicadores-de-importância)
8. [Boruta](#8-boruta)
9. [Consenso](#9-consenso)
10. [Lendo o `FeatureSelectionReport`](#10-lendo-o-featureselectionreport)
11. [Gráficos](#11-gráficos)
12. [Referência de configuração (`FeatureSelectionConfig`)](#12-referência-de-configuração-featureselectionconfig)
13. [Backend e performance](#13-backend-e-performance)
14. [Logging no MLflow](#14-logging-no-mlflow)
15. [Exemplo ponta a ponta](#15-exemplo-ponta-a-ponta)
16. [Troubleshooting / FAQ](#16-troubleshooting--faq)

---

## 1. Visão geral

O `yggdrasil.feature_selection` responde a uma pergunta objetiva: **dado um conjunto grande de candidatas `feat_*`, quais variáveis vale a pena levar adiante?**

Características centrais:

- **Esteira independente.** Não entra no pipeline do modelo nem na EDA. É um **entrypoint próprio** (`run_feature_selection`) que recebe um Spark DataFrame e devolve um relatório.
- **Organização por book.** Features são agrupadas por **origem de dados** (book), ex.: `serasa`, `bvs`. Todo o pipeline roda **por book**, e ao final há um **ranking global** (`overall_importance`) das selecionadas.
- **Pipeline por book.** Filtros duros (missing, variância, redundância) → avaliação de importância (RandomForest + univariadas) e **Boruta** → consolidação num **consenso**.
- **Spark-first.** As operações pesadas usam `pyspark.ml` (ou, opcionalmente, `sklearn` no driver). Os resultados pequenos são coletados no driver como `pandas`.

Use o módulo quando precisar **reduzir e justificar** um conjunto de candidatas antes de modelar, mantendo rastreabilidade do **motivo** de cada descarte ou seleção.

---

## 2. Instalação

O `pyspark` é um **extra opcional**. O pacote **importa sem pyspark** — o `import` é *gated*: só falha, com mensagem clara, ao **executar** uma função distribuída.

```bash
pip install 'yggdrasil[spark]'   # instala o yggdrasil com o extra de Spark
```

> Sem o extra, ao executar a seleção você verá: `A seleção de features requer pyspark — instale com: pip install 'yggdrasil[spark]'`.

Imports usados ao longo deste tutorial:

```python
from yggdrasil import ColumnConfig                                  # contrato de colunas
from yggdrasil.feature_selection import run_feature_selection, FeatureSelectionConfig  # entrypoint + config
```

O pacote `yggdrasil.feature_selection` reexporta exatamente cinco nomes: `run_feature_selection`, `FeatureSelectionReport`, `FeatureSelectionConfig`, `resolve_books` e `Book`.

---

## 3. Contrato de dados (`ColumnConfig`)

A seleção depende do `ColumnConfig`, que define o **contrato de colunas** do DataFrame:

- **`feature_prefix`** — prefixo das colunas-candidatas. Só colunas que **começam com** esse prefixo (via `startswith`, **case-sensitive**) entram na seleção.
- **`target_col`** — coluna-alvo. Se ela **não estiver** em `sdf.columns`, `run_feature_selection` levanta `ValueError(f"Coluna de alvo '{cfg.target_col}' ausente no DataFrame.")`.
- **`sample_col` / `dev_sample`** — coluna de amostra e o rótulo da amostra de desenvolvimento.

**Uso da amostra DEV.** Se `cfg.sample_col` existir nas colunas, a seleção é **restrita à amostra de desenvolvimento**: `base = sdf.where(F.col(cfg.sample_col) == cfg.dev_sample)`. Esse filtro **só é aplicado** se a amostra DEV não for vazia; caso contrário, o DataFrame inteiro é mantido. Quando o filtro entra, registra-se `Seleção restrita à amostra de desenvolvimento '<dev_sample>'.` (com o rótulo da amostra DEV interpolado).

```python
cfg = ColumnConfig()                 # usa os defaults do contrato (feature_prefix, target_col, sample_col, dev_sample)
# run_feature_selection(sdf, cfg)    # passa o contrato explicitamente
```

> Se `cfg` não for informado, `run_feature_selection` usa `ColumnConfig()` (`cfg = cfg or ColumnConfig()`).

---

## 4. Quickstart mínimo

```python
from yggdrasil import ColumnConfig
from yggdrasil.feature_selection import run_feature_selection

report = run_feature_selection(sdf)              # sdf é o seu Spark DataFrame; cfg/books inferidos
print(report.selected_overall)                   # lista achatada de todas as features selecionadas
report.summary()                                 # tabela resumo: por book, n_features/selecionadas/descartadas
```

Inspecionando o resultado:

```python
report.selection_table                           # tabela completa (todas as colunas, todos os books)
report.selected_features                         # dict: {book: [features selecionadas]}
report.overall_importance                        # ranking global das selecionadas
```

Sem `books`, o módulo **auto-deriva** os books pelo 1º segmento após o prefixo (ver seção 5). Sem `problem_type`, ele é **inferido** do alvo (ver seção 6).

---

## 5. Books: as 3 formas

O parâmetro `books` (`BooksSpec = Union[Sequence[str], Dict[str, Sequence[str]], None]`) aceita **três formas**:

### Forma 1 — lista de palavras-chave (principal)

Uma `Sequence[str]` de palavras-chave. O match é **case-insensitive** e por **`contains`** (substring): `token = str(kw).lower()` e `feats = [c for c in cols if token in c.lower()]`. O **nome do book** preserva o caso original do `kw` (via `str(kw)`).

```python
report = run_feature_selection(sdf, ColumnConfig(), books=["serasa", "bvs"])  # dois books por palavra-chave
```

Se **nenhuma** feature contiver o token, emite-se um *warning* e o book vazio é **ignorado** (não levanta erro por book vazio individual).

### Forma 2 — dict explícito

Um `Dict[str, Sequence[str]]` mapeando nome do book → colunas explícitas:

```python
books = {
    "serasa": ["feat_serasa_score", "feat_serasa_pendencias"],   # colunas explícitas do book
    "bvs":    ["feat_bvs_consultas"],
}
report = run_feature_selection(sdf, ColumnConfig(), books=books)
```

Regras do modo dict:
- Se **alguma coluna listada não existir** no DataFrame (checado contra `set(sdf.columns)`), levanta `ValueError("Book '<name>': colunas inexistentes no DataFrame: <faltando>")`.
- Colunas **sem o prefixo** `feature_prefix` são **aceitas** (incluídas no book); apenas emitem *warning*.
- Há **dedup** preservando a ordem (`list(dict.fromkeys(feats))`).

### Forma 3 — `None` (auto)

Sem `books` (ou `books=None`), as colunas de feature são agrupadas pelo **1º segmento** após o prefixo, derivado por `_auto_token` (separador `_`):

```python
# feat_serasa_score  -> book "serasa"
# feat_bvs_consultas -> book "bvs"
report = run_feature_selection(sdf)   # books=None: auto pelo 1º segmento após o prefixo
```

A ordem dos books segue a **1ª ocorrência** de cada token.

> Em qualquer forma, `resolve_books` mantém apenas books com features não-vazias e levanta `ValueError("Nenhum book com features foi resolvido. Verifique 'books'.")` se a lista final ficar vazia. Se **nenhuma** coluna começar com o prefixo, `feature_columns` levanta `ValueError("Nenhuma coluna de feature encontrada com o prefixo '<prefix>'.")`.

---

## 6. O pipeline por book, passo a passo

Antes de processar os books, o `problem_type` é inferido (se não informado) por `_infer_problem_type`: coleta até 3 valores distintos não-nulos do alvo; se algum não for convertível para `float`, retorna `'classification'`; senão, retorna `'classification'` se houver **≤ 2** distintos **e** todos forem subconjunto de `{0.0, 1.0}`, caso contrário `'regression'`.

Para **cada book**, `_process_book` aplica esta ordem exata:

| # | Etapa | O que faz | Motivo registrado |
|---|-------|-----------|-------------------|
| 1 | **Missing** | Descarta a feature se `pct_missing` for finito e `> missing_max` | `alto missing` |
| 2 | **Variância** | `variance_flags`: descarta se `sem_variancia` (teste por percentis P1 vs P99) ou `near_constante` | `sem variancia` / `quase-constante` |
| 3 | **Importância** | `importance_indicators`: preenche `rf_importance`, `iv`, `ks`, `auc`, `gini`, `corr_target`, `score`, `leakage_flag` | — |
| 4 | **Redundância** | `correlation_matrices` (spearman) + `redundancy_clusters(corr_high)`: mantém o **representante** de cada cluster; não-representantes descartados | `redundante c/ <feat>` |
| 5 | **Boruta** | Só roda se `boruta_enable` **e** houver vivos numéricos; preenche `boruta_hits`, `boruta_decisao`, `boruta_hit_rate` | — |
| 6 | **Consenso** | `_consensus`: combina importância, Boruta e sinal com o alvo na decisão final | ver seção 9 |

**Detalhes do teste de variância** (etapa 2): `sem_variancia = (p_high - p_low) <= var_tol` para numéricas com percentis finitos; senão `nunique_approx <= 1`. Os percentis usam `var_p_low`/`var_p_high` via `approxQuantile` com `approx_rel_error`. `near_constante = top1_share >= near_constant`, onde `top1_share` (valor modal / linhas não-nulas) só é computado para features que **ainda têm variância**.

**Redundância** (etapa 4): a distância usada é `1 - |corr|`; o corte é `1.0 - corr_high`. O **representante** de cada cluster é a feature de maior `importance` (aqui, o `score` da etapa 3). Features **não-numéricas** ficam fora da matriz e são tratadas como `representante=True`.

A tabela do book é ordenada por `['selecionada', 'score']` (ambos descendente, `na_position='last'`) quando há coluna `score`.

---

## 7. Indicadores de importância

`importance_indicators` produz **uma linha por feature** com os indicadores abaixo. As métricas de modelo operam **só em colunas numéricas** (`numeric_columns`); features não-numéricas ficam `NaN` no merge `how='left'`.

- **`rf_importance`** — `featureImportances` de um RandomForest (`pyspark.ml` no backend `spark`), arredondado a 6 casas. Treinado com `numTrees=rf_n_estimators`, `maxDepth=rf_max_depth`, `subsamplingRate=rf_subsampling`, `seed=rf_seed`. Em classificação usa `RandomForestClassifier`; caso contrário, `RandomForestRegressor`.
- **`corr_target`** — correlação (com sinal) de cada feature numérica com o alvo. Calculada sobre o `sdf` **original** (não a amostra do RF).
- **`iv`, `ks`, `auc`, `gini`** — métricas univariadas **só em classificação** (`problem_type == 'classification'`). Em regressão, **essas 4 colunas não existem** no DataFrame.

**Binning das univariadas.** Usa `n_bins` para o número de bins do IV/KS univariado. WoE com suavização de Siddiqi; KS = `max|cumsum(dg) - cumsum(db)|`; AUC tornado direção-agnóstico (`max(auc, 1-auc)`) e `gini = 2*auc - 1`.

**Score composto.** É a **média dos RANKS** (não dos valores) das colunas disponíveis entre `('rf_importance', 'iv', 'ks', 'gini', 'corr_target')` — para `corr_target` usa-se o **valor absoluto**. **Atenção:** `auc` **não** entra no score. Maior `score` = mais importante.

**`leakage_flag`** (booleano por feature):

```
leakage_flag = (gini > 2*leakage_auc - 1) OR (iv > iv_leakage)
```

com `gini = gini.fillna(0).abs()` e `iv = iv.fillna(0)`. Em **regressão** (sem `gini`/`iv`), ambos viram série de zeros, logo `leakage_flag` fica **`False`** para todas.

---

## 8. Boruta

O Boruta compara a importância de cada feature real contra **shadows** (cópias embaralhadas das features). A cada iteração `t` (seed `rf_seed + t`), calcula-se o limiar `thr = percentil(shadow_imp, boruta_perc)` e conta-se um **hit** para cada feature real com `real_imp > thr` (estritamente maior).

**Decisão** (`boruta_decision`). Sob H0, `hits ~ Binomial(n_iter, 0.5)`. Com correção de **Bonferroni** (`alpha_c = boruta_alpha / n_features`):

- **`confirmada`** — se `p_accept <= alpha_c` **e** `p_accept <= p_reject` (tem prioridade);
- **`rejeitada`** — senão, se `p_reject <= alpha_c`;
- **`tentativa`** — caso contrário.

A saída traz `feature`, `hits`, `n_iter`, `hit_rate` (`= round(h/n_iter, 4)`) e `decisao`.

**Backend (`spark` vs `driver`).** Selecionado por `cfg.backend` em `boruta_select`:

- **`driver`** — amostra para o driver (`maybe_sample(...).toPandas()`) e roda `boruta_driver` (sklearn). Shadows permutadas **independentemente por coluna**.
- **qualquer outro valor** (ex.: `spark`) — roda `boruta_spark` sobre `maybe_sample(sdf, cfg)`. Shadows permutadas **em bloco** (uma reordenação de linhas por iteração via `row_number` sobre `Window.orderBy(F.rand(rf_seed+t))`).

Ambos só operam em **colunas numéricas** e aplicam amostragem via `maybe_sample`. Em `classification` usa `RandomForestClassifier`; qualquer outro valor cai no Regressor.

**Custo e sampling.** Boruta roda **até `boruta_max_iter`** iterações de RandomForest — é a etapa mais cara. Controle o custo com `sample_size` (amostra `N` linhas para as etapas de modelo) e, no backend `spark`, com `rf_*`. Desligue tudo com `boruta_enable=False`.

---

## 9. Consenso

O consenso (`_consensus`) consolida a decisão de seleção. Ele monta uma **média ponderada** de até três componentes (cada um só entra se for finito):

| Componente | Valor | Peso |
|------------|-------|------|
| Importância | `imp_norm` (rank percentil do `score` **dentro do book**) | `peso_importancia` |
| Boruta | `boruta_hit_rate` (só se `boruta_enable`) | `peso_boruta` |
| Alvo | `abs(corr_target)` limitado a `1.0` | `peso_alvo` |

```
score_consenso = sum(w*v) / sum(w)      # média ponderada; se W <= 0, score = NaN
```

> `imp_norm = score.reindex(vivos3).rank(pct=True)` — a normalização da importância (rank percentil) é **por book** e feita sobre as **representantes vivas pós-redundância** (`vivos3`), não sobre todas as que passaram no filtro de missing nem de forma global. Os defaults dos pesos somam 1.0 (`0.50 + 0.35 + 0.15`), mas **não há validação** forçando essa soma.

**Decisão de seleção** (ordem de precedência):

1. **`leakage_flag` verdadeiro** → **NÃO** selecionada — motivo `suspeita de leakage (revisar)`. *Leakage barra a seleção, antes de tudo.*
2. **Boruta `confirmada`** → selecionada — `selecionada (Boruta confirmada)`.
3. **Consenso** (`score_consenso` finito **e** `>= consensus_threshold`) → selecionada — `selecionada (consenso)` (ou `...; Boruta rejeitou` se a decisão Boruta foi `rejeitada`).
4. **Boruta `rejeitada`** → **NÃO** — `Boruta rejeitada`.
5. Caso contrário → **NÃO** — `consenso abaixo do limiar`.

---

## 10. Lendo o `FeatureSelectionReport`

`run_feature_selection` retorna um `FeatureSelectionReport` (`@dataclass`) com:

| Atributo | Tipo | Conteúdo |
|----------|------|----------|
| `selection_table` | `pd.DataFrame` | Concat de todas as `book_tables` (`ignore_index=True`) |
| `book_tables` | `Dict[str, pd.DataFrame]` | Tabela de seleção por book (chave = `book.name`) |
| `selected_features` | `Dict[str, List[str]]` | Por book, features com `selecionada == True` |
| `selected_overall` | `List[str]` | Flatten de `selected_features` (na ordem dos books) |
| `overall_importance` | `pd.DataFrame` | Ranking **global** recalculado sobre `selected_overall`, com coluna `book` na posição 1 |
| `panels` | `Dict[str, object]` | Figuras matplotlib (ver seção 11); vazio se `with_panels=False` |
| `problem_type` | `Optional[str]` | Tipo de problema usado |
| `cfg` | `Optional[ColumnConfig]` | O `ColumnConfig` usado |

**Colunas da `selection_table`** (ordem canônica `_COLS`):

```
book, feature, pct_missing, p_low, p_high, top1_share, sem_variancia, near_constante,
rf_importance, iv, ks, auc, gini, corr_target, score, leakage_flag, cluster,
representante, redundante_com, boruta_hits, boruta_decisao, score_consenso,
selecionada, motivo
```

`pct_missing` e `score_consenso` são arredondados a 4 casas (ou `NaN`).

**Métodos:**

```python
report.summary()                  # por book: book, n_features, selecionadas, descartadas
report.to_csv("selecao.csv")      # grava selection_table.to_csv(index=False); retorna o path
html = report.to_html()           # string HTML; to_html(embed_panels=True) embute os painéis como PNG base64
```

> `overall_importance`: se nada for selecionado (ou não houver numéricas), vira `DataFrame(columns=['feature','book','rf_importance','score'])`.

---

## 11. Gráficos

Com `with_panels=True` (default), o relatório traz figuras matplotlib em `report.panels`, indexadas por chave string:

| Chave | Figura |
|-------|--------|
| `"overview"` | `plot_book_overview` — selecionadas × descartadas por book (empilhado) |
| `"overall_importance"` | `plot_overall_importance(overall, top_k_overall)` — top global |
| `"book::<nome>"` | `plot_book_selection(t, nome, top_k_book)` — seleção do book, com `motivo` anotado nas não selecionadas |
| `"corr::<nome>"` | `plot_corr_heatmap` — heatmap de correlação do book (**só** se a matriz não for vazia e tiver ≥ 2 features) |

`top_k_book` (default 15) e `top_k_overall` (default 25) controlam quantas features aparecem nos painéis por book e no ranking global, respectivamente.

Exibindo e salvando (cada figura é uma `Figure` matplotlib com `_repr_png_` para render inline no Jupyter):

```python
report.panels["overview"]                      # render inline no Jupyter (sem duplicar)
fig = report.panels["overall_importance"]
fig.savefig("overall.png", dpi=110, bbox_inches="tight")   # salva o ranking global

report.panels["book::serasa"]                  # painel de seleção do book serasa
report.panels["corr::serasa"]                  # heatmap de correlação do book serasa (se existir)
```

> Em dados vazios/insuficientes, as funções de plot ainda retornam uma `Figure` (com mensagem como `book vazio` ou `nenhuma feature selecionada`), nunca `None` nem exceção.

---

## 12. Referência de configuração (`FeatureSelectionConfig`)

`FeatureSelectionConfig` é um `@dataclass` puro (apenas um container de parâmetros; sem métodos além da validação em `__post_init__`). Os defaults espelham os limiares da EDA de features.

| Parâmetro | Default | Significado |
|-----------|---------|-------------|
| `missing_max` | `0.50` | `%missing` acima disso ⇒ descartar feature |
| `var_p_low` | `0.01` | Percentil inferior do teste de variância (P1) |
| `var_p_high` | `0.99` | Percentil superior do teste de variância (P99) |
| `var_tol` | `0.0` | Sem variância se `(p_high - p_low) <= var_tol` |
| `near_constant` | `0.99` | Share do valor modal p/ quase-constante |
| `corr_high` | `0.80` | `|corr|` p/ considerar duas features redundantes |
| `corr_target_min` | `0.0` | **Reservado / não utilizado** pela lógica atual (default espelhado da EDA; não filtra features) |
| `iv_min` | `0.02` | **Reservado / não utilizado** pela lógica atual (default espelhado da EDA; não filtra features) |
| `iv_leakage` | `0.50` | IV acima disso = suspeita de leakage |
| `leakage_auc` | `0.95` | AUC univariado acima = suspeita de leakage |
| `rf_n_estimators` | `100` | Nº de árvores do RandomForest (`pyspark.ml`) |
| `rf_max_depth` | `6` | Profundidade máx. do RandomForest |
| `rf_subsampling` | `1.0` | Fração de subsampling do RandomForest |
| `rf_seed` | `42` | Seed do RandomForest |
| `boruta_enable` | `True` | Liga/desliga a etapa Boruta |
| `boruta_max_iter` | `50` | Nº máx. de iterações do Boruta |
| `boruta_alpha` | `0.05` | Nível de significância do teste binomial |
| `boruta_perc` | `100.0` | Percentil das shadows usado como limiar (100 = máximo) |
| `backend` | `"spark"` | `"spark"` (`pyspark.ml`) \| `"driver"` (`sklearn`) |
| `sample_size` | `0` | `>0` amostra N linhas p/ as etapas de modelo (0 = full) |
| `approx_rel_error` | `0.01` | Erro relativo do `approxQuantile` (0 = exato, caro) |
| `n_bins` | `10` | Nº de bins p/ IV/KS univariado em classificação |
| `consensus_threshold` | `0.50` | `score_consenso >=` isso (ou Boruta confirmada) ⇒ selecionar |
| `peso_importancia` | `0.50` | Peso do rank de importância no consenso |
| `peso_boruta` | `0.35` | Peso da taxa de hits do Boruta no consenso |
| `peso_alvo` | `0.15` | Peso do sinal de relação com o alvo no consenso |
| `top_k_book` | `15` | Nº de features por book nos painéis (plots) |
| `top_k_overall` | `25` | Nº de features no ranking geral (plots) |

> **Nota:** `corr_target_min` e `iv_min` estão definidos no `config.py`, mas **não são lidos em nenhum ponto do pipeline** de seleção — são parâmetros reservados (defaults espelhados da EDA) e não afetam quais features são selecionadas.

**Validações** (`__post_init__`):
- `backend` deve ser `'spark'` ou `'driver'`, senão `ValueError("backend deve ser 'spark' ou 'driver'.")`.
- Exige `0.0 <= var_p_low < var_p_high <= 1.0` (desigualdade **estrita** entre os dois), senão `ValueError("Exige 0 <= var_p_low < var_p_high <= 1.")`.

```python
fs_cfg = FeatureSelectionConfig(            # exemplo de override
    missing_max=0.30,                       # mais rígido com missing
    corr_high=0.90,                         # tolera mais correlação antes de chamar de redundante
    sample_size=500_000,                    # amostra p/ as etapas de modelo
    backend="spark",                        # pyspark.ml
)
```

---

## 13. Backend e performance

**`backend`** controla o motor de modelo:

- **`"spark"`** (default) — RandomForest e Boruta via `pyspark.ml`; tudo distribuído no cluster.
- **`"driver"`** — usa `sklearn` no driver; o Boruta amostra para o driver via `toPandas()`.

**Controles de custo:**

- **`sample_size`** — `>0` amostra `N` linhas **só para as etapas de modelo** (RF e Boruta) via `sdf.sample(False, N/n, seed=rf_seed)`; `0` usa o dataset completo. A correlação com o alvo e as univariadas usam sempre o `sdf` **completo**, não a amostra.
- **`approx_rel_error`** — erro relativo do `approxQuantile` (usado em variância, imputação por mediana, binning das univariadas e matriz de correlação). `0` = exato, porém caro. A cardinalidade aproximada (`approx_count_distinct`) **não** é afetada por este parâmetro — ela usa o `rsd` default do Spark (0.05).
- **`rf_n_estimators`, `rf_max_depth`, `rf_subsampling`** — tamanho/custo do RandomForest.
- **`boruta_enable=False`** — desliga a etapa mais cara; **`boruta_max_iter`** limita as iterações.

**Recomendações para Databricks:** prefira `backend="spark"` para escalar no cluster; use `sample_size > 0` para acelerar RF/Boruta em bases grandes mantendo a correlação/univariadas no dataset completo; mantenha `approx_rel_error` em `0.01` (ou maior) para evitar `approxQuantile` exato; reduza `boruta_max_iter` ou desligue o Boruta para iterações rápidas.

---

## 14. Logging no MLflow

Se `mlflow_experiment` for *truthy*, `run_feature_selection` chama o logger interno e registra a corrida (mlflow é importado de forma *lazy*):

```python
report = run_feature_selection(
    sdf,
    ColumnConfig(),
    books=["serasa", "bvs"],
    mlflow_experiment="/Shared/feature_selection",   # nome do experimento MLflow
    run_name="fs_serasa_bvs_v1",                      # nome do run
)
```

O que é registrado:
- **Artefatos** (em `artifact_path="feature_selection"`): `selection_table.csv`, `overall_importance.csv`, `feature_selection.html` (com `embed_panels=False`) e os PNGs dos painéis em `panels/`.
- **Tags:** `framework="yggdrasil-feature-selection"`, `stage="feature_selection"`.
- **Métricas:** `n_features`, `n_selecionadas`, `n_books`.

---

## 15. Exemplo ponta a ponta

Cenário realista: classificação, dois books (`serasa` e `bvs`), amostra DEV no contrato, log no MLflow.

```python
from yggdrasil import ColumnConfig
from yggdrasil.feature_selection import run_feature_selection, FeatureSelectionConfig

cfg = ColumnConfig()                              # contrato: feature_prefix, target_col, sample_col, dev_sample

fs_cfg = FeatureSelectionConfig(                  # ajustes da seleção
    missing_max=0.40,                             # descarta quem tem >40% missing
    corr_high=0.85,                               # redundância acima de |corr| 0.85
    sample_size=1_000_000,                        # amostra p/ RF e Boruta
    boruta_enable=True,                           # mantém Boruta ligado
    backend="spark",                              # pyspark.ml
)

report = run_feature_selection(
    sdf,                                          # Spark DataFrame com feat_*, alvo e coluna de amostra
    cfg,
    fs_cfg,
    books=["serasa", "bvs"],                      # dois books por palavra-chave
    problem_type="classification",                # explícito (poderia ser inferido)
    with_panels=True,                             # monta os painéis
    mlflow_experiment="/Shared/feature_selection",
    run_name="fs_serasa_bvs_v1",
)

# Inspeção
print(report.selected_features)                   # {"serasa": [...], "bvs": [...]}
report.summary()                                  # por book: n_features / selecionadas / descartadas
report.overall_importance.head(25)                # ranking global das selecionadas

# Painéis
report.panels["overview"]                         # selecionadas x descartadas por book
report.panels["overall_importance"]               # top global
report.panels["book::serasa"]                     # seleção detalhada do book serasa

# Exportações
report.to_csv("selecao_features.csv")             # CSV da selection_table
open("selecao_features.html", "w", encoding="utf-8").write(report.to_html())   # HTML do relatório
```

---

## 16. Troubleshooting / FAQ

**As métricas de modelo só aparecem para algumas features (numeric-only).**
`rf_importance`, `corr_target`, `iv`, `ks`, `auc`, `gini` e o Boruta operam **só em colunas numéricas** (filtradas por `numeric_columns`, prefixos de dtype: `int`, `bigint`, `smallint`, `tinyint`, `double`, `float`, `decimal`, `long`). Features não-numéricas ficam `NaN` nessas colunas (merge `how='left'`) e entram no pipeline como `representante=True` na redundância.

**`Nenhum book com features foi resolvido. Verifique 'books'.`**
Nenhum book resolveu features não-vazias. No modo palavra-chave, verifique se os tokens realmente aparecem (como substring, case-insensitive) nos nomes das colunas. Lembre que books vazios por palavra-chave são apenas **ignorados** com warning.

**`Nenhuma coluna de feature encontrada com o prefixo '<prefix>'.`**
Nenhuma coluna começa com `cfg.feature_prefix` (match **case-sensitive**, `startswith`). Confira o `feature_prefix` do `ColumnConfig` contra os nomes reais das colunas.

**`A seleção de features requer pyspark — instale com: pip install 'yggdrasil[spark]'`.**
O `pyspark` não está instalado. O import do pacote funciona sem ele, mas a **execução** falha. Instale o extra `[spark]`.

**Não tenho coluna de amostra.**
Tudo bem: se `cfg.sample_col` não existir nas colunas, a seleção roda no DataFrame **inteiro**. O filtro de amostra DEV também é ignorado se a amostra DEV estiver vazia.

**Regressão sem IV/KS/AUC/Gini.**
`iv`, `ks`, `auc`, `gini` e o binning (`iv_leakage`, `leakage_auc`, `n_bins`) são **exclusivos de classificação**. Em regressão essas colunas **não existem** no DataFrame; o `leakage_flag` fica `False` para todas, e o `score`/consenso se apoiam em `rf_importance` e `corr_target`.

**`Coluna de alvo '<target>' ausente no DataFrame.`**
O `cfg.target_col` não está em `sdf.columns`. Ajuste o `target_col` do `ColumnConfig` ou o DataFrame de entrada.

**Erro de validação na config.**
`backend` deve ser `'spark'` ou `'driver'`; e os percentis de variância exigem `0.0 <= var_p_low < var_p_high <= 1.0` (não podem ser iguais).
