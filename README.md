# Yggdrasil-Project

![](https://cdn.pixabay.com/photo/2023/10/31/16/56/yggdrasil-8355580_1280.png)

> "Três raízes sustentam a Yggdrasil, e por elas correm as águas que dão vida aos mundos."

Na cosmologia nórdica, **Yggdrasil é a árvore-mundo:** um freixo imenso cujos galhos abrigam os céus e cujas raízes mergulham em três fontes sagradas:

- Poço de Urðr (das normas e do destino);
- Poço de Mímir (da sabedoria) ;
- Hvergelmir (de onde brotam todos os rios).

É ela que conecta os mundos, e é dela que o cosmos retira sua coerência.

Este projeto toma a árvore emprestada como metáfora de organização. `Yggdrasil-Project` é um repositório pessoal de ciência de dados que cresce a partir de três raízes — estatística, machine learning e tutoriais — e tem por ambição ser um lugar único onde esses três mundos se sustentam mutuamente.

---

## 🗂️ Descrição das Pastas

###  `conf/`
Contém arquivos de configuração separados por ambiente (desenvolvimento, homologação, produção)

Nunca versionar credenciais ou segredos — utilize variáveis de ambiente ou um secret manager.

### `dashboards/`
Dashboards para acompanhamento de:

Qualidade dos dados — _freshness_, completude, volume e distribuições.
Performance dos modelos — métricas de treino, validação e produção.
Drift — alertas de data drift e concept drift.

### `docs/`
Documentação técnica do projeto, incluindo:

Dicionário de features
Decisões de design (ADRs — Architecture Decision Records)
Guias de onboarding para novos colaboradores

### `jobs/`
Definições de jobs para orquestração dos pipelines. 

### `notebooks/`
Ambiente de exploração e prototipagem. Organizado em:

01_eda/ — Análise exploratória de dados
02_feature_engineering/ — Prototipagem de features
03_modeling/ — Experimentos de modelagem
04_evaluation/ — Análise de resultados e interpretabilidade

⚠️ Notebooks não devem conter lógica de produção. Código validado deve ser migrado para src/.

### `references/`
Materiais de apoio e referência:

Esquemas de tabelas e contratos de dados
Artigos e papers de referência

### `src/`
Código-fonte principal do projeto, organizado em módulos reutilizáveis:

### `tests/`
Módulo de testes automatizados cobrindo:

Testes unitários — funções e transformações individuais

---

## 🚂 Esteira de ML (`yggdrasil`)

Esteira governada de Machine Learning (estilo risco de crédito) orquestrada por **MLflow**.
A entrada é uma tabela com features `feat_*`, coluna de data, coluna de amostra
(`DES`/`OOT` para análise; `SIMUL`/`BACKTEST` apenas para score + rating) e a variável resposta.

A esteira registra no experimento:

- **Métricas** por amostra — KS, AUC, Gini, Acurácia, F1 (classificação) e RMSE, MAE, MAPE, R² (regressão);
- **Shifts** DES→OOT de cada métrica (absoluto e relativo);
- **Grupos homogêneos (ratings)** em 4 metodologias: `decis`, `quantil` (fusão monotônica por
  inversão / Mann-Whitney), `arvore` (DecisionTree) e `optbin` (OptBinning);
- **PSI** agregado (DES→OOT) e a série temporal do PSI de cada rating;
- **SHAP** (importância + beeswarm) e **relatórios por grupo** (média prevista/observada,
  representatividade, monotonicidade) + dashboard.

### Instalação

```bash
pip install -e ".[dev]"          # núcleo + ferramentas de teste/notebook
pip install -e ".[pycaret]"      # opcional: treino automatizado via PyCaret
```

### Uso

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

> 📖 **Metodologia** (o *porquê* dos métodos — KS, PSI/CSI, WoE/IV, ratings com fusão monotônica, SHAP, veredito de EDA): [`docs/metodologia.md`](docs/metodologia.md).
> 📓 **Tutoriais passo a passo** (cada módulo isolado):
> [PD/classificação](notebooks/tutoriais/00_tutorial_yggdrasil.ipynb) ·
> [LGD/regressão (alvo [0,1] bimodal)](notebooks/tutoriais/01_tutorial_lgd.ipynb) ·
> [EDA de features](notebooks/tutoriais/02_tutorial_eda_features.ipynb).
> Notebook orquestrador pronto para produção: [`notebooks/03_modeling/01_esteira_ml_mlflow.ipynb`](notebooks/03_modeling/01_esteira_ml_mlflow.ipynb).

### 🔎 Esteira de EDA de features (`yggdrasil.eda`)

Subpacote **isolado** (não interfere na esteira de modelo) para análise exploratória inicial das features:
missing (global e por safra), percentis e variação no tempo, histograma, relação com o alvo, binning com
**WoE/IV**, **importância** (univariada + surrogate multivariado), **estabilidade/PSI por feature** e extras
(monotonicidade, outliers, correlação/VIF/redundância, **detecção de leakage**). Consolida tudo num
`feature_profile` com **veredito** (manter/revisar/descartar).

```python
from yggdrasil import ColumnConfig
from yggdrasil.eda import run_feature_eda, EDAConfig

report = run_feature_eda(df, ColumnConfig(), EDAConfig())   # df: feat_*, dt_ref, amostra, target (opcional)
report.feature_profile        # 1 linha por feature, com flags e veredito
report.panels["feat_x"]       # painel consolidado da feature
```
> Localmente, o MLflow 3.x exige `MLFLOW_ALLOW_FILE_STORE=true` para usar o backend `./mlruns`
> (o notebook já define isso). No Databricks, use o tracking do workspace.


















