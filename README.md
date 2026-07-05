# Yggdrasil-Project

![](https://cdn.pixabay.com/photo/2023/10/31/16/56/yggdrasil-8355580_1280.png)

> "Três raízes sustentam a Yggdrasil, e por elas correm as águas que dão vida aos mundos."

Na cosmologia nórdica, Yggdrasil é a árvore-mundo: um freixo imenso cujos galhos abrigam os céus e cujas raízes mergulham em três fontes sagradas:

- Poço de Urðr (das normas e do destino);
- Poço de Mímir (da sabedoria);
- Hvergelmir (de onde brotam todos os rios).

É ela que conecta os mundos, e é dela que o cosmos retira sua coerência.

Aqui a árvore vira metáfora de organização. O `Yggdrasil-Project` é um repositório pessoal de ciência de dados que cresce a partir de três raízes: estatística, machine learning e tutoriais. A ideia é manter os três num lugar só, onde um apoia o outro.

O foco aplicado é risco de crédito (PD, LGD, EAD) no estilo regulatório brasileiro (CMN 4.966/2021, IFRS 9). O núcleo é pandas puro e roda tanto localmente quanto no Databricks.

---

## 📦 O que o pacote `yggdrasil` contempla hoje

O código de produção vive em `src/yggdrasil/` (layout `src/`). São cinco esteiras isoladas que compartilham o mesmo contrato de dados, mas não interferem umas nas outras.

### 1. 🚂 Esteira de ML governada (`yggdrasil`)
Avaliação completa de um modelo já treinado, orquestrada por MLflow. A entrada é uma tabela com features `feat_*`, coluna de data (`dt_ref`), coluna de amostra (`amostra`) e a variável resposta (`target`), tudo configurável via `ColumnConfig`. As amostras `DES` e `OOT` recebem análise completa; `SIMUL` e `BACKTEST` são *scoring-only* (predição mais grupo homogêneo).

Registra no experimento:
- Métricas por amostra: KS, AUC, Gini, Acurácia, F1 (classificação) e RMSE, MAE, MAPE, R² (regressão);
- Shifts DES→OOT de cada métrica (absoluto e relativo);
- Grupos homogêneos (ratings) em 4 metodologias: `decis`, `quantil` (fusão monotônica por inversão / Mann-Whitney), `arvore` (DecisionTree) e `optbin` (OptBinning);
- PSI agregado (DES→OOT) e a série temporal do PSI de cada rating;
- SHAP (importância e beeswarm) e relatórios por grupo (média prevista/observada, representatividade, monotonicidade), além de um dashboard.

Módulos: `metrics/`, `ratings/`, `monitoring/psi.py`, `interpretability/shap_explain.py`, `reporting/`, `tracking/mlflow_logger.py`, `pipeline.py`. O treino é agnóstico; `training/pycaret_adapter.py` é opcional.

### 2. 🔎 Esteira de EDA de features (`yggdrasil.eda`)
Análise exploratória inicial das features: missing (global e por safra), percentis e variação no tempo, histograma, relação com o alvo, binning com WoE/IV, importância (univariada mais surrogate multivariado), estabilidade/PSI por feature e extras (monotonicidade, outliers, correlação/VIF/redundância, detecção de leakage). Consolida tudo num `feature_profile` (1 linha por feature) com veredito (manter, revisar ou descartar).

### 3. 🧮 Esteira de seleção de features em PySpark (`yggdrasil.feature_selection`)
Seleção por book (grupo de features por palavra-chave ou prefixo, ex.: `serasa`, `bvs`) sobre um Spark DataFrame. O pipeline por book vai de missing a variância, importância (RF `pyspark.ml` com IV/KS/AUC/Gini/corr_target), redundância (Pearson e Spearman), Boruta (Spark-native com shadows, e fallback driver/sklearn) até o consenso (`selecionada` e `motivo`). Saída: tabela e painéis por book, mais um ranking global. Backend `"spark"` ou `"driver"`.

### 4. 🌳 Árvore de segmentação de risco de crédito (`yggdrasil.credit_risk.tree`)
`TreeSegmenter` e `TreeSegmenterUI` são uma única classe/UI que atende PD e LGD, escolhendo o comportamento por `task_type` (substituem as antigas classes separadas `SequentialPDSegmenter` e `SequentialLGDSegmenter`). É uma régua sequencial com UI interativa (5 abas): binning ótimo/manual, faltantes em bin própria, notas por folha, IV, PSI/CSI, bootstrap, calibração, backtest, save/load JSON e `predict`/`to_pyspark`/`apply_spark`/`log_to_mlflow`.
- `task_type="classification"` (PD), alvo binário: binning binário, IV WoE (escala Siddiqi), KS/AUC/Gini/Acurácia/F1 e gráficos ROC/KS/taxa-default/distribuição.
- `task_type="regression"` (LGD), alvo contínuo: binning contínuo, IV contínuo, métricas MAE/RMSE/R² e boxplot/histograma do alvo.

A aba Avançado traz, entre outros: critério de split selecionável no Auto-fit e no split por folha (`criterion=` em `fit_auto`/`grow`, com `optbin`, mais `gini`/`entropy`/`ks`/`iv`/`chi2` na classificação e `variance`/`mae`/`ftest` na regressão); `suggest_splits()` (TOP-N variáveis com nº de bins, PSI por amostra, teste de hipótese e IV) e sugestão de cortes com máx. bins por variável na folha; `feature_importance()` das variáveis que entraram na árvore; auto-merge de folhas indistinguíveis (`auto_merge`); `to_sql()` (régua como `CASE WHEN` copiável); `diff_trees()` (migração de notas e métricas entre duas versões); `report_pdf()` (relatório do modelo em PDF) e tema escuro na UI.

```python
from yggdrasil.credit_risk.tree import TreeSegmenter
seg = TreeSegmenter(df, target="target", task_type="classification",  # ou "regression"
                    sample_col="amostra", ref_sample="DES")
seg.fit_auto(max_depth=3, criterion="ks")     # ou "optbin" (padrão), "gini", ...
seg.suggest_splits(top=3); seg.feature_importance(); seg.metrics()
print(seg.to_sql(table="carteira"))           # régua como CASE WHEN
```

### 5. 🤖 Segmentador orientado a modelo (`yggdrasil.credit_risk.model`)
`ModelSegmenter` e `ModelSegmenterUI` unificam classificação e regressão via `task_type`. O fluxo vai da análise univariada (logodds/WoE, IV, distribuição, inversão de bins entre amostras/safras, com opção de bins manuais) para a seleção/categorização de variáveis, depois o ajuste do modelo, as métricas com fórmula (coeficientes/odds-ratio nos modelos lineares) e SHAP, até o score que vira ratings (decis/quantil/arvore/optbin). A persistência fica em JSON (config) e `.model.joblib` (modelo e estratégia). A UI tem 5 abas: Variáveis, Análise de variáveis, Modelo (com SHAP), Ratings & Score e Validar & Exportar.

Algoritmos disponíveis (registry extensível em `ALGORITHMS`):

| Algoritmo | Tarefas | Dependência |
|---|---|---|
| Regressão Logística / Linear | clf / reg | scikit-learn (core) |
| Random Forest, Extra Trees | clf + reg | scikit-learn (core) |
| Gradient Boosting, Hist Gradient Boosting | clf + reg | scikit-learn (core) |
| LightGBM | clf + reg | core |
| XGBoost | clf + reg | core |
| CatBoost | clf + reg | extra `[catboost]` |

> Também aceita um modelo já treinado via `set_model(...)`. LightGBM e XGBoost vêm no core; o CatBoost é importado sob demanda e, sem ele, o erro orienta a instalar o extra `[catboost]`.

**Tuning bayesiano (Optuna):** `seg.tune_optuna(algorithm="lightgbm", n_trials=40)` busca os hiperparâmetros que maximizam AUC (clf) ou R² (reg) no OOT e re-treina com os melhores. Na UI há um slider de *trials* e o botão Tunar com Optuna (com barra de progresso) na aba Modelo. O Optuna já vem no core (o extra `[optuna]` existe só por compatibilidade).

Mais na UI e no segmentador: ratings em decis/quantil/árvore/optbin, e também manuais (`manual_score` por cortes de score, `manual_percentil` por lista de percentis); na regressão logística, a tabela da fórmula traz o p-valor (Wald) e estrelas de significância por coeficiente; relatório PDF do modelo (`report_pdf`) e tema escuro (toggle).

### 6. 🏛️ Capital econômico de carteira (`yggdrasil.credit_risk.capital`)
Estimativa do **capital para absorver perdas inesperadas** da carteira de crédito em 1 ano, no nível de confiança do apetite de risco (ex.: 99,9%) — a visão **interna** que complementa a provisão (ECL, IFRS 9 / CMN 4.966) e o capital regulatório de Pilar 1, capturando concentração e diversificação entre produtos (cartão, consignado, veículos) que o Pilar 1 ignora. Baseado no guia de construção (ASRF/Vasicek, Monte Carlo multifatorial, CreditMetrics e CreditRisk+) e organizado do contrato de dados ao uso gerencial.

- **Contrato**: `Segment` (PD TTC, LGD/CCF *downturn*, ρ, fator sistêmico) e `Portfolio` (matriz de correlação entre fatores).
- **Distribuição de perdas e medidas**: `LossDistribution`, `value_at_risk`, `expected_shortfall`, `economic_capital` (`CE = VaR_q − EL`).
- **Motores**: `asrf_capital` (v1, analítico e aditivo), `simulate` (v2, Monte Carlo multifatorial com LGD estocástica e correlação adversa PD–LGD), `creditrisk_plus` (benchmark atuarial por recursão de Panjer) e `MigrationModel` (CreditMetrics / migração de estágio).
- **Insumos**: `pit_to_ttc`, `lgd_downturn_from_series`, `ccf_downturn`; correlações `asset_correlation_moments`/`asset_correlation_mle`/`factor_correlation_matrix`/`nearest_correlation`; regulatório `basel_correlation`/`basel_irb_capital` (Pilar 1).
- **Alocação e uso**: `euler_allocation` (contribuição à cauda), `raroc`/`raroc_table`, benefício de diversificação.
- **Validação**: `sensitivity`, `correlation_stress`, `benchmark`, `pillar1_comparison`, `backtest_expected_loss`, `convergence`.
- **Produtos**: `preset`/`PRESETS` (particularidades de cartão, consignado, veículos e afins). Visualizações (`report`, matplotlib) e registro no MLflow (`log_capital_run`) carregados sob demanda.

```python
from yggdrasil.credit_risk.capital import Portfolio, Segment

carteira = Portfolio([
    Segment("cartao_revolver", pd=0.06, lgd=0.75, ead=8e6, rho=0.10, n_obligors=40_000,
            product="cartao", factor="cartao"),
    Segment("consig_inss", pd=0.01, lgd=0.30, ead=12e6, rho=0.04, n_obligors=60_000,
            product="consignado", factor="consignado"),
], factor_corr=[[1.0, 0.25], [0.25, 1.0]], factor_names=["cartao", "consignado"])

carteira.asrf_capital(q=0.999).summary()                 # v1 analítico (ASRF/Vasicek)
sim = carteira.simulate(n_scenarios=200_000, q=0.999, seed=42)   # v2 Monte Carlo
sim.economic_capital(); sim.allocate(metric="es")        # capital + alocação de Euler
```

---

## 🗂️ Estrutura de pastas

| Pasta | Conteúdo |
|---|---|
| `src/yggdrasil/` | Código-fonte principal (as seis esteiras acima). |
| `tests/` | Testes automatizados (`pytest`): suíte parametrizada (classificação/regressão), incluindo UI, Spark, boosting e Optuna (estes *gated* pela dependência). |
| `notebooks/tutoriais/` | Tutoriais passo a passo (índice abaixo). A lógica de produção não vive aqui. |
| `docs/` | Metodologia (o *porquê* dos métodos) e documentação dos segmentadores. |
| `conf/` | Configuração por ambiente (dev/homolog/prod). Nunca versionar segredos. |
| `dashboards/` | Acompanhamento de qualidade de dados, performance e drift. |
| `jobs/` | Definições de jobs para orquestração dos pipelines. |
| `references/` | Esquemas de tabelas, contratos de dados e papers de apoio. |

---

## ⚙️ Instalação

```bash
pip install -e ".[dev]"          # núcleo + ferramentas de teste/notebook
pip install -e ".[ui]"           # opcional: UIs interativas (ipywidgets)
pip install -e ".[spark]"        # opcional: geração/aplicação de régua em PySpark (fora do Databricks)
pip install -e ".[catboost]"     # opcional: CatBoost (LightGBM e XGBoost já vêm no core)
pip install -e ".[pycaret]"      # opcional: treino automatizado via PyCaret
```

> CatBoost é o único motor de boosting não incluído por padrão (`pip install -e ".[catboost]"`).

> Localmente, o MLflow 3.x exige `MLFLOW_ALLOW_FILE_STORE=true` para usar o backend `./mlruns` (os notebooks já definem isso). No Databricks, use o tracking do workspace.

## 🚀 Uso rápido

```python
from yggdrasil import MLPipeline, ColumnConfig

cfg = ColumnConfig()  # feat_, dt_ref, amostra, target  (ajustável)
pipe = MLPipeline(cfg, problem_type="classification",
                  ratings=["decis", "quantil", "arvore", "optbin"])
resultado = pipe.run(df, model=modelo_treinado, experiment="/Shared/Yggdrasil/pd_pf")

resultado.metrics_by_sample   # métricas por DES/OOT
resultado.shifts              # shifts DES->OOT
resultado.reports             # relatório por grupo homogêneo
```

```python
# EDA de features (subpacote isolado)
from yggdrasil import ColumnConfig
from yggdrasil.eda import run_feature_eda, EDAConfig

report = run_feature_eda(df, ColumnConfig(), EDAConfig())
report.feature_profile        # 1 linha por feature, com flags e veredito
```

---

## 📓 Tutoriais

Todos centralizados em **[`notebooks/tutoriais/`](notebooks/tutoriais/)** (passo a passo, prontos para Jupyter/Databricks):

| # | Tutorial |
|---|---|
| 00 | [Visão geral / PD](notebooks/tutoriais/00_tutorial_yggdrasil.ipynb) |
| 01 | [LGD / regressão (alvo [0,1] bimodal)](notebooks/tutoriais/01_tutorial_lgd.ipynb) |
| 02 | [EDA de features](notebooks/tutoriais/02_tutorial_eda_features.ipynb) |
| 03 | [Seleção de features (PySpark)](notebooks/tutoriais/03_tutorial_feature_selection.ipynb) |
| 04 | [Árvore de segmentação unificada (PD & LGD por `task_type`)](notebooks/tutoriais/04_tutorial_tree_segmenter.ipynb) |
| 06 | [Construtor de modelos (UI)](notebooks/tutoriais/06_tutorial_model_segmenter.ipynb) |
| 07 | [Esteira ML + MLflow](notebooks/tutoriais/07_tutorial_esteira_ml_mlflow.ipynb) |
| 08 | [Capital econômico (ASRF, Monte Carlo, alocação de Euler)](notebooks/tutoriais/08_tutorial_capital_economico.ipynb) |

> 📖 **Metodologia** (o *porquê* dos métodos, como KS, PSI/CSI, WoE/IV, ratings com fusão monotônica, SHAP e veredito de EDA): [`docs/metodologia.md`](docs/metodologia.md).
> 🌳 **Árvore de segmentação unificada (PD & LGD):** [`docs/credit-risk/tree-segmenter.md`](docs/credit-risk/tree-segmenter.md).

---

## 🖼️ Galeria

| Árvore de PD (classificação) | Importância SHAP (model) |
|---|---|
| ![Árvore de segmentação de PD](docs/img/tree_pd.png) | ![Importância SHAP](docs/img/shap_importance.png) |

| Dispersão do alvo por folha (LGD / regressão) |
|---|
| ![Boxplot do alvo por folha](docs/img/tree_lgd_boxplot.png) |

> As UIs interativas (`TreeSegmenterUI` e `ModelSegmenterUI`) têm tema claro e escuro (toggle 🌙), abas de construção/diagnóstico/validação, sugestão de splits, critério de split (Gini/Entropy/KS/IV/Chi²/Variância/MAE/F-test), export SQL, diff de versões e relatório PDF. Rode os tutoriais para ver ao vivo.

---

## 👤 Sobre o desenvolvedor

**Richard Guilherme**, Cientista de Dados com foco em risco de crédito (PD/LGD/EAD), modelagem regulatória (CMN 4.966/2021, IFRS 9) e MLOps em Databricks.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Richard%20Guilherme-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/richardguilhermeda/)

> 🔗 Conecte-se no LinkedIn para acompanhar projetos e conteúdos de ciência de dados e risco de crédito.
