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


















