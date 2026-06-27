# Yggdrasil-Project

![](https://cdn.pixabay.com/photo/2023/10/31/16/56/yggdrasil-8355580_1280.png)

> "Três raízes sustentam a Yggdrasil, e por elas correm as águas que dão vida aos mundos."

Na cosmologia nórdica, **Yggdrasil é a árvore-mundo:** um freixo imenso cujos galhos abrigam os céus e cujas raízes mergulham em três fontes sagradas:

- Poço de Urðr (das normas e do destino);
- Poço de Mímir (da sabedoria) ;
- Hvergelmir (de onde brotam todos os rios).

É ela que conecta os mundos, e é dela que o cosmos retira sua coerência.

Este projeto toma a árvore emprestada como metáfora de organização. `Yggdrasil-Project` é um repositório pessoal de ciência de dados que cresce a partir de três raízes — estatística, machine learning e tutoriais — e tem por ambição ser um lugar único onde esses três mundos se sustentam mutuamente.

O foco aplicado é **risco de crédito** (PD, LGD, EAD) no estilo regulatório brasileiro (CMN 4.966/2021, IFRS 9). O núcleo é **pandas puro** e roda tanto localmente quanto no **Databricks**.

---

## 📦 O que o pacote `yggdrasil` contempla hoje

O código de produção vive em `src/yggdrasil/` (layout `src/`). São **cinco esteiras isoladas** que compartilham o mesmo contrato de dados, mas não interferem umas nas outras:

### 1. 🚂 Esteira de ML governada (`yggdrasil`)
Avaliação completa de um modelo já treinado, orquestrada por **MLflow**. A entrada é uma tabela com features `feat_*`, coluna de data (`dt_ref`), coluna de amostra (`amostra`) e a variável resposta (`target`) — configurável via `ColumnConfig`. Amostras `DES`/`OOT` recebem análise completa; `SIMUL`/`BACKTEST` são *scoring-only* (predição + grupo homogêneo).

Registra no experimento:
- **Métricas** por amostra — KS, AUC, Gini, Acurácia, F1 (classificação) e RMSE, MAE, MAPE, R² (regressão);
- **Shifts** DES→OOT de cada métrica (absoluto e relativo);
- **Grupos homogêneos (ratings)** em 4 metodologias: `decis`, `quantil` (fusão monotônica por inversão / Mann-Whitney), `arvore` (DecisionTree) e `optbin` (OptBinning);
- **PSI** agregado (DES→OOT) e a série temporal do PSI de cada rating;
- **SHAP** (importância + beeswarm) e **relatórios por grupo** (média prevista/observada, representatividade, monotonicidade) + dashboard.

Módulos: `metrics/`, `ratings/`, `monitoring/psi.py`, `interpretability/shap_explain.py`, `reporting/`, `tracking/mlflow_logger.py`, `pipeline.py`. Treino é agnóstico; `training/pycaret_adapter.py` é opcional.

### 2. 🔎 Esteira de EDA de features (`yggdrasil.eda`)
Análise exploratória inicial das features: missing (global e por safra), percentis e variação no tempo, histograma, relação com o alvo, binning com **WoE/IV**, **importância** (univariada + surrogate multivariado), **estabilidade/PSI por feature** e extras (monotonicidade, outliers, correlação/VIF/redundância, **detecção de leakage**). Consolida tudo num `feature_profile` (1 linha por feature) com **veredito** (manter/revisar/descartar).

### 3. 🧮 Esteira de seleção de features em PySpark (`yggdrasil.feature_selection`)
Seleção **por book** (grupo de features por palavra-chave/prefixo, ex.: `serasa`, `bvs`) sobre um Spark DataFrame. Pipeline por book: missing → variância → importância (RF `pyspark.ml` + IV/KS/AUC/Gini/corr_target) → redundância (Pearson+Spearman) → **Boruta** (Spark-native com shadows; fallback driver/sklearn) → **consenso** (`selecionada` + `motivo`). Saída: tabela/painéis por book + ranking global. Backend `"spark"|"driver"`.

### 4. 🌳 Segmentadores de risco de crédito por árvore de bins (`yggdrasil.credit_risk.lgd` / `.pd`)
Réguas sequenciais com UI interativa (5 abas): binning ótimo/manual, faltantes em bin própria, notas por folha, IV, PSI/CSI, bootstrap, calibração, backtest, save/load JSON e `predict`/`to_pyspark`/`apply_spark`/`log_to_mlflow`.
- **`lgd/`** — `SequentialLGDSegmenter` + `LGDSegmenterUI`. Alvo **contínuo** (LGD): binning contínuo, IV contínuo, métricas MAE/RMSE/R².
- **`pd/`** — `SequentialPDSegmenter` + `PDSegmenterUI`. Alvo **binário** (default): IV WoE (escala Siddiqi), KS/AUC/Gini/Acurácia/F1, gráficos ROC/KS/taxa-default/distribuição.

### 5. 🤖 Segmentador orientado a modelo (`yggdrasil.credit_risk.model`)
`ModelSegmenter` + `ModelSegmenterUI` — **unifica classificação e regressão** via `task_type`. Fluxo: análise univariada (logodds/WoE, IV, distribuição, inversão de bins entre amostras/safras, com opção de **bins manuais**) → seleção/categorização de variáveis → ajuste do modelo → métricas + **fórmula** (coeficientes/odds-ratio nos modelos lineares) + SHAP → **score → ratings** (decis/quantil/arvore/optbin). Persistência em JSON (config) + `.model.joblib` (modelo + estratégia). UI em 5 abas: Variáveis · Análise de variáveis · Modelo (+SHAP) · Ratings & Score · Validar & Exportar.

Algoritmos disponíveis (registry extensível em `ALGORITHMS`):

| Algoritmo | Tarefas | Dependência |
|---|---|---|
| Regressão Logística / Linear | clf / reg | scikit-learn (core) |
| Random Forest · Extra Trees | clf + reg | scikit-learn (core) |
| Gradient Boosting · Hist Gradient Boosting | clf + reg | scikit-learn (core) |
| **LightGBM** | clf + reg | extra `[lgbm]` |
| **XGBoost** | clf + reg | extra `[xgboost]` |
| **CatBoost** | clf + reg | extra `[catboost]` |

> Também aceita um modelo já treinado via `set_model(...)`. Os motores de boosting opcionais são importados sob demanda — sem o pacote, o erro orienta a instalação do extra correto.

---

## 🗂️ Estrutura de pastas

| Pasta | Conteúdo |
|---|---|
| `src/yggdrasil/` | Código-fonte principal (as cinco esteiras acima). |
| `tests/` | Testes automatizados (`pytest`) — 206 testes, incluindo os UI e Spark (estes *gated*). |
| `notebooks/tutoriais/` | Tutoriais passo a passo (índice abaixo). Lógica de produção **não** vive aqui. |
| `docs/` | Metodologia (o *porquê* dos métodos) e documentação dos segmentadores. |
| `conf/` | Configuração por ambiente (dev/homolog/prod). Nunca versionar segredos. |
| `dashboards/` | Acompanhamento de qualidade de dados, performance e drift. |
| `jobs/` | Definições de jobs para orquestração dos pipelines. |
| `references/` | Esquemas de tabelas, contratos de dados, papers de apoio. |

---

## ⚙️ Instalação

```bash
pip install -e ".[dev]"          # núcleo + ferramentas de teste/notebook
pip install -e ".[ui]"           # opcional: UIs interativas (ipywidgets)
pip install -e ".[spark]"        # opcional: geração/aplicação de régua em PySpark (fora do Databricks)
pip install -e ".[boosting]"     # opcional: LightGBM + XGBoost + CatBoost para o ModelSegmenter
pip install -e ".[pycaret]"      # opcional: treino automatizado via PyCaret
```

> Os motores de boosting também podem ser instalados individualmente: `.[lgbm]`, `.[xgboost]` ou `.[catboost]`.

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
| 04 | [Segmentador LGD (UI)](notebooks/tutoriais/04_tutorial_lgd_segmenter.ipynb) |
| 05 | [Segmentador PD (UI)](notebooks/tutoriais/05_tutorial_pd_segmenter.ipynb) |
| 06 | [Construtor de modelos (UI)](notebooks/tutoriais/06_tutorial_model_segmenter.ipynb) |
| 07 | [Esteira ML + MLflow](notebooks/tutoriais/07_tutorial_esteira_ml_mlflow.ipynb) |
| 08 | [Relatórios de validação LGD](notebooks/tutoriais/08_tutorial_validacao_lgd.ipynb) |

> 📖 **Metodologia** (o *porquê* dos métodos — KS, PSI/CSI, WoE/IV, ratings com fusão monotônica, SHAP, veredito de EDA): [`docs/metodologia.md`](docs/metodologia.md).
> 📑 **Segmentadores:** [LGD](docs/credit-risk/lgd-segmenter.md) · [PD](docs/credit-risk/pd-segmenter.md).
