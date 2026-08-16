# Yggdrasil-Project

![](https://cdn.pixabay.com/photo/2023/10/31/16/56/yggdrasil-8355580_1280.png)

> "Três raízes sustentam a Yggdrasil, e por elas correm as águas que dão vida aos mundos."

Na cosmologia nórdica, Yggdrasil é a árvore-mundo: um freixo imenso cujos galhos abrigam os céus e cujas raízes mergulham em três fontes sagradas:

- Poço de Urðr (das normas e do destino);
- Poço de Mímir (da sabedoria);
- Hvergelmir (de onde brotam todos os rios).

É ela que conecta os mundos, e é dela que o cosmos retira sua coerência.

Aqui a árvore vira metáfora de organização. O `Yggdrasil-Project` é um repositório pessoal de ciência de dados que cresce a partir de três raízes: estatística, machine learning e tutoriais. A ideia é manter os três num lugar só, onde um apoia o outro.

O foco aplicado é o **crédito**, de forma ampla — cobrindo todo o ciclo: da **concessão** (aprovação, definição de limites e precificação), passando pela **recuperação** (cobrança e renegociação), até o **risco de crédito**. O núcleo é pandas puro e roda tanto localmente quanto no Databricks.

---

## 📦 O que o pacote `yggdrasil` contempla hoje

O código de produção vive em `yggdrasil/`, na raiz do repositório (layout *flat*): assim `import yggdrasil` funciona **sem `pip install`** — tanto no clone local (rodando da raiz) quanto no Databricks Repos, que adiciona a raiz do repo ao `sys.path`. São **oito módulos isolados** que não interferem uns nos outros: as esteiras de ML, EDA e seleção compartilham o contrato de dados `feat_*`/`dt_ref`/`amostra`/`target`; os de risco de crédito (segmentadores, capital, perda esperada e econométricos) têm contratos próprios.

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

### 3. 🧮 Esteira de triagem de features (`yggdrasil.feature_selection`)
Seleção por book (grupo de features por palavra-chave ou prefixo, ex.: `externo`, `mercado`) sobre um DataFrame **pandas ou Spark** — o backend vem do tipo do objeto, a chamada e o relatório são os mesmos. O pipeline por book vai de missing a variância, importância (RF com IV/KS/AUC/Gini/corr_target), redundância (Pearson e Spearman), Boruta (com shadows) até o consenso (`selecionada` e `motivo`). Saída: tabela e painéis por book, mais um ranking global. Com pandas roda tudo no driver via `sklearn` e dispensa o `pyspark`; com Spark distribui no cluster (`pyspark.ml`), e aí o backend `"spark"`/`"driver"` decide onde RF e Boruta rodam. É a peneira **anterior** ao modelo — a régua final da lista curta é a `ModelSegmenter.select_features`.

### 4. 🌳 Árvore de segmentação de risco de crédito (`yggdrasil.credit_risk.tree`)
`TreeSegmenter` e `TreeSegmenterUI` são uma única classe/UI que atende classificação e regressão, escolhendo o comportamento por `task_type` (substituem as antigas classes separadas por tarefa). É uma régua sequencial com UI interativa (5 abas): binning ótimo/manual, faltantes em bin própria, notas por folha, IV, PSI/CSI, bootstrap, calibração, backtest, save/load JSON e `predict`/`to_pyspark`/`apply_spark`/`log_to_mlflow`.
- `task_type="classification"`, alvo binário: binning binário, IV WoE (escala Siddiqi), KS/AUC/Gini/Acurácia/F1 e gráficos ROC/KS/taxa-default/distribuição.
- `task_type="regression"`, alvo contínuo: binning contínuo, IV contínuo, métricas MAE/RMSE/R² e boxplot/histograma do alvo.

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
Estimativa do **capital para absorver perdas inesperadas** da carteira de crédito em 1 ano, no nível de confiança do apetite de risco (ex.: 99,9%) — a visão **interna** que complementa a provisão (ECL) e o capital regulatório de Pilar 1, capturando concentração e diversificação entre produtos (cartão, consignado, veículos) que o Pilar 1 ignora. Baseado no guia de construção (ASRF/Vasicek, Monte Carlo multifatorial, CreditMetrics e CreditRisk+) e organizado do contrato de dados ao uso gerencial.

- **Contrato**: `Segment` (frequência TTC, perda/conversão *downturn*, ρ, fator sistêmico) e `Portfolio` (matriz de correlação entre fatores).
- **Distribuição de perdas e medidas**: `LossDistribution`, `value_at_risk`, `expected_shortfall`, `economic_capital` (`CE = VaR_q − EL`).
- **Motores**: `asrf_capital` (v1, analítico e aditivo), `simulate` (v2, Monte Carlo multifatorial com severidade estocástica e correlação adversa frequência–severidade), `creditrisk_plus` (benchmark atuarial por recursão de Panjer) e `MigrationModel` (CreditMetrics / migração de estágio).
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

### 7. 💧 Perda esperada: PD lifetime, ELBE e CCF (`yggdrasil.credit_risk.ecl`)
Os **parâmetros de risco da perda esperada** e a conta que os junta na provisão de IFRS 9 / Resolução CMN 4.966/2021. É a outra metade da pergunta que os segmentadores respondem: eles *ordenam* o risco entre clientes em 12 meses; aqui se decide **quando** a perda acontece ao longo da vida do contrato, **quanto** ainda sobra a perder no que já quebrou e **sobre qual exposição** ela incide. Núcleo em pandas/numpy/scipy/sklearn — não exige o extra `[econometric]`.

- **A curva (`PDCurve`)** — o objeto aceita e entrega as **quatro** representações (condicional/*hazard*, marginal, acumulada, sobrevivência) por identidades exatas, o que elimina a fonte nº 1 de erro de um período em projeto de ECL. Mais `forward(t0, t1)` (o insumo quantitativo do SICR), `truncate`/`extend` e persistência JSON.
- **Cinco motores, uma fachada** — `LifetimePD(method=...)`: `constant` (só a PD de 12m vira curva), `vintage` (taxa marginal por idade, com a base em risco recontada), `km` (Kaplan-Meier com Greenwood e IC log-log), `hazard` (regressão de *hazard* em tempo discreto — curva **por contrato**, com covariáveis) e `markov` (cadeia sobre a matriz de transição — curva **por rating**, reusando `capital.migration`).
- **As duas pontes** — `calibrate_to(pd_12m=...)` cola o **nível** da curva na PD do scorecard preservando o **formato** da maturação; `condition(z, rho)` desloca ao **ciclo** por Vasicek (`z > 0` = benigno), com `z` escalar, por horizonte (a projeção do satélite) ou com reversão à média.
- **ELBE** — `elbe_table(...)` sai de duas colunas: a **exposição inicial** e as **LGD por mês em *default***. Devolve a curva de recuperação (encadeada pela marginal, para a coorte variável não distorcer a comparação), o horizonte de *workout*, `ELBE = (1 − r̄(T*))/(1 − r̄(t))`, a LGD *in default* e o desconto pela taxa efetiva.
- **CCF/EAD** — `reference_dataset(...)` monta a base pelos três desenhos da literatura (coorte, horizonte fixo, horizonte variável), calcula as **quatro medidas ex-post** (CCF/LEQ, EADF, AUF, EAD direto), conta cada exclusão da higiene e expõe a **bimodalidade** em 0 e 1; `pooled_ccf`, `backtest_ead` (viés em moeda), `compare_measures`, `ccf_psi` e o `ccf_downturn` reexportado do capital.
- **A montagem** — `ecl_table(...)`: `ECL = Σ PD_marginal · LGD · EAD · desconto`, com o estágio cortando o horizonte (1 → 12 meses, 2 → *lifetime*, 3 → ELBE) e `ecl_scenarios` ponderando o *forward-looking*. A regra de **SICR não está aqui** de propósito: é política da instituição, e o módulo recebe a coluna pronta.
- **Governança** — gráficos (`report`) e MLflow (`log_lifetime_pd`/`log_elbe`/`log_ccf`/`log_ecl_run`) carregados sob demanda.

```python
from yggdrasil.credit_risk.ecl import ContractPanel, LifetimePD, elbe_table, apply_elbe, ecl_table

painel = ContractPanel(df, origin_col="safra_origem", segment_col="produto", term_col="prazo")
pd_lt  = LifetimePD(method="vintage", horizon=60).fit(painel, by="produto")
pd_lt  = pd_lt.calibrate_to({"cartao": 0.078, "consignado": 0.021})   # nível do scorecard

elbe = elbe_table(defaults, exposure_col="exposicao_inicial", lgd_prefix="lgd_m")
carteira = apply_elbe(carteira, elbe, months_col="meses_em_default")

res = ecl_table(carteira, model=pd_lt, lgd="lgd", ead="saldo", stage_col="estagio",
                elbe="elbe", discount_rate="taxa_efetiva", age_col="idade", term_col="prazo")
res.total, res.summary()
```

### 8. 📈 Modelos econométricos (satélite) das séries de risco (`yggdrasil.credit_risk.econometric`)
Modelos **satélite / macro** que ligam as **séries temporais agregadas** dos parâmetros de risco (taxa de *default*, perda e conversão por segmento) às **variáveis macroeconômicas** (desemprego, renda, juros, câmbio, inadimplência) e **projetam por cenário**. É o **eixo temporal**, complementar ao eixo transversal dos segmentadores: o transversal *ordena* o risco entre clientes, o satélite *desloca o nível* da curva conforme o ciclo. As projeções alimentam o *forward-looking* do ECL, os testes de estresse, o **capital econômico** (a ligação fator-macro) e o planejamento. Baseado no guia de construção (ARDL, ARIMAX, fator Z de Vasicek, beta/fractional logit, VAR/VECM, painel), organizado da série ao relatório de governança. Requer o extra `[econometric]` (`statsmodels` + `arch`) e é **carregado sob demanda** — o resto de `credit_risk` não o exige.

- **Séries + sintéticos**: `RiskSeries` (contrato com `kind` pd/lgd/ccf) e geradores de **DGP conhecido** (`simulate_pd/lgd/ccf_series`, `make_reference_study`) — a base dos testes de recuperação de parâmetros.
- **Transformações e diagnóstico**: `transforms` (logit/probit, **fator Z de Vasicek**, defasagens, dummies sazonais/evento/quebra); `diagnostics` (ADF/KPSS/PP, Ljung-Box, Breusch-Godfrey/Pagan/White, Jarque-Bera, ARCH-LM, VIF, Chow/Quandt-Andrews, CUSUM) com saída tabular padronizada.
- **Modelos** (interface comum `fit`/`predict`/`project`/`diagnostics`): `ARDL` (principal), `ARIMA`/ARIMAX (benchmark), `VasicekZ` (ponte com o capital), `BetaRegression`/`FractionalLogit` (alvos em [0,1]), ingênuos (`RandomWalk`/`HistoricalMean`/`SeasonalNaive`), `VARModel`/`VECMModel` + cointegração (Engle-Granger/Johansen) e `PanelSatellite`.
- **Seleção e cenários**: `search` champion-challenger (filtros de **sinal econômico** e **VIF**, *walk-forward*, Diebold-Mariano); `Scenario`/`ScenarioSet`, `project` e `ecl_projection` (ponderação de cenários para o ECL).
- **Governança**: relatório HTML (`report`), registro no MLflow (`log_satellite_run`) e o pipeline declarativo `StudyConfig`/`run_study` (as "cinco chamadas") — carregados sob demanda.

```python
from yggdrasil.credit_risk.econometric import make_reference_study, StudyConfig, run_study

est = make_reference_study()                       # macro + séries de risco sintéticas
cfg = StudyConfig(kind="pd", candidates=["desemprego", "renda", "juros"],
                  expected_signs={"desemprego": 1, "renda": -1, "juros": 1})
r = run_study(cfg, est.pd.series, est.macro)       # seleção → ajuste → diagnóstico → projeção → relatório
r.summary(); r.projection.mean_frame()             # ranking e projeção por cenário (base/adverso/otimista)
```

---

## 🗂️ Estrutura de pastas

| Pasta | Conteúdo |
|---|---|
| `yggdrasil/` | Código-fonte principal (os oito módulos acima), na raiz do repo (layout *flat*). |
| `tests/` | Testes automatizados (`pytest`): suíte parametrizada (classificação/regressão), incluindo UI, Spark, boosting, Optuna e econométricos (`statsmodels`/`arch`) — estes *gated* pela dependência. |
| `notebooks/tutoriais/` | Tutoriais passo a passo (índice abaixo). A lógica de produção não vive aqui. |
| `docs/` | Metodologia (o *porquê* dos métodos), documentação dos segmentadores e da perda esperada. |
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
pip install -e ".[econometric]"  # opcional: modelos econométricos satélite (statsmodels + arch)
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

Todos centralizados em **[`notebooks/tutoriais/`](https://github.com/richardguilhermeds/Yggdrasil-Project/tree/main/notebooks/tutoriais)** (passo a passo, prontos para Jupyter/Databricks):

| # | Tutorial | O que resolve |
|---|---|---|
| 00 | [Visão geral (classificação)](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/notebooks/tutoriais/00_tutorial_yggdrasil.ipynb) | Passeio pela lib peça a peça — métricas, ratings, PSI, SHAP — até o `MLPipeline` fazer tudo de uma vez |
| 01 | [Regressão (alvo [0,1] bimodal)](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/notebooks/tutoriais/01_tutorial_lgd.ipynb) | A mesma esteira quando o alvo é contínuo e preso em [0,1] |
| 02 | [EDA de features](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/notebooks/tutoriais/02_tutorial_eda_features.ipynb) | Conhecer a base antes de decidir qualquer coisa: perfil, estabilidade e veredito por feature |
| 03 | [Seleção de features e de variáveis, ponta a ponta](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/notebooks/tutoriais/03_tutorial_feature_selection.ipynb) | Em duas partes: **triagem do universo** por book (pandas **ou** Spark, Boruta e consenso → shortlist) e depois a **régua do modelo** (PSI, monotonia, VIF, funil, relatório e política em JSON) |
| 04 | [Árvore de segmentação unificada](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/notebooks/tutoriais/04_tutorial_tree_segmenter.ipynb) | Achar os segmentos da carteira (classificação & regressão por `task_type`) |
| 05 | [Instalação e carregamento das interfaces](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/notebooks/tutoriais/05_tutorial_instalacao_e_interfaces.ipynb) | Colocar a árvore e o model segmenter para rodar no seu ambiente |
| 06 | [Construtor de modelos (UI)](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/notebooks/tutoriais/06_tutorial_model_segmenter.ipynb) | Construir, diagnosticar e validar o modelo na interface |
| 07 | [Esteira ML + MLflow](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/notebooks/tutoriais/07_tutorial_esteira_ml_mlflow.ipynb) | Rastrear experimento, artefato e versão de ponta a ponta |
| 08 | [Capital econômico](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/notebooks/tutoriais/08_tutorial_capital_economico.ipynb) | ASRF, Monte Carlo multifatorial e alocação de Euler sobre a carteira |
| 09 | [Modelos econométricos satélite](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/notebooks/tutoriais/09_tutorial_modelos_econometricos.ipynb) | Ligar PD/LGD/CCF ao macro (ARDL, fator Z) e projetar por cenários |
| 11 | [Interface de séries temporais (`SatelliteUI`)](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/notebooks/tutoriais/11_tutorial_interface_series_temporais.ipynb) | 7 abas, da estacionariedade ao backtest de cobertura |
| 12 | [Perda esperada ponta a ponta](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/notebooks/tutoriais/12_tutorial_ecl.ipynb) | Os três parâmetros juntos: curva de PD, ELBE, CCF e a tabela de ECL com estágios e cenários |
| 13 | [PD lifetime](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/notebooks/tutoriais/13_tutorial_pd_lifetime.ipynb) | Aprofundamento: censura, as 4 representações da curva, os 5 motores, calibração de nível e condicionamento ao ciclo |
| 14 | [ELBE](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/notebooks/tutoriais/14_tutorial_elbe.ipynb) | Aprofundamento: por que a média direta da recuperação cai, horizonte de *workout*, desconto, add-on e LGD *in default* |
| 15 | [CCF / EAD](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/notebooks/tutoriais/15_tutorial_ccf.ipynb) | Aprofundamento: os 3 desenhos de base, as 4 medidas, higiene, bimodalidade, backtest de EAD e a ponte com o `ModelSegmenter` |

> O **03** cobre a seleção inteira em **duas partes que rodam em sequência**: a Parte 1
> (`yggdrasil.feature_selection`) é a peneira do **universo de features** — centenas de colunas agrupadas
> por book, filtros duros + importância + Boruta, devolvendo uma shortlist; a Parte 2
> (`ModelSegmenter.select_features`) pega essa lista curta e aplica a **régua do modelo** — categóricas,
> PSI, monotonia, VIF, backward — gerando a decisão por variável, a política reproduzível em JSON e o
> relatório que vai anexo à documentação. As partes são independentes: cada uma tem setup e base próprios.
>
> **Trilha sugerida:** `00` → `02` (conhecer a base) → `03` (selecionar) → `04`/`06` (modelar)
> → `07` (rastrear). Os de risco de crédito (`08`, `09`, `11`) são independentes. Na perda
> esperada, o `12` é a visão de conjunto e o `13`/`14`/`15` são os aprofundamentos por
> parâmetro — comece pelo `12` se quiser o mapa, ou vá direto ao módulo que você precisa.

> 📖 **Metodologia** (o *porquê* dos métodos, como KS, PSI/CSI, WoE/IV, ratings com fusão monotônica, SHAP e veredito de EDA): [`docs/metodologia.md`](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/docs/metodologia.md).
> 🌳 **Árvore de segmentação unificada (classificação & regressão):** [`docs/credit-risk/tree-segmenter.md`](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/docs/credit-risk/tree-segmenter.md).
> 💧 **PD lifetime, ELBE e CCF** (as quatro representações da curva, censura, coorte variável, desenhos de base de CCF e as referências da literatura): [`docs/credit-risk/ecl.md`](https://github.com/richardguilhermeds/Yggdrasil-Project/blob/main/docs/credit-risk/ecl.md).

---

## 🖼️ Galeria

| Árvore (classificação) | Importância SHAP (model) |
|---|---|
| ![Árvore de segmentação (classificação)](https://raw.githubusercontent.com/richardguilhermeds/Yggdrasil-Project/main/docs/img/tree_pd.png) | ![Importância SHAP](https://raw.githubusercontent.com/richardguilhermeds/Yggdrasil-Project/main/docs/img/shap_importance.png) |

| Dispersão do alvo por folha (regressão) |
|---|
| ![Boxplot do alvo por folha](https://raw.githubusercontent.com/richardguilhermeds/Yggdrasil-Project/main/docs/img/tree_lgd_boxplot.png) |

| Projeção condicional por cenário — modelo satélite (leque 90%) |
|---|
| ![Projeção econométrica em leque](https://raw.githubusercontent.com/richardguilhermeds/Yggdrasil-Project/main/docs/img/econometric_fanchart.png) |

> As UIs interativas (`TreeSegmenterUI` e `ModelSegmenterUI`) têm tema claro e escuro (toggle 🌙), abas de construção/diagnóstico/validação, sugestão de splits, critério de split (Gini/Entropy/KS/IV/Chi²/Variância/MAE/F-test), export SQL, diff de versões e relatório PDF. Rode os tutoriais para ver ao vivo.

---

## 👤 Sobre o desenvolvedor

**Richard Guilherme**, Cientista de Dados com foco em crédito e MLOps em Databricks.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Richard%20Guilherme-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/richard-guilherme-da/)

> 🔗 Conecte-se no LinkedIn para acompanhar projetos e conteúdos de ciência de dados e risco de crédito.
