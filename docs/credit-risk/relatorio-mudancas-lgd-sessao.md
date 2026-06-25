# Relatório de mudanças — módulo `credit_risk.lgd`

_Resumo do que foi construído nesta sessão de trabalho._

## Resumo executivo

Esta sessão evoluiu o subpacote **`yggdrasil.credit_risk.lgd`** (segmentação de LGD
para risco de crédito sob CMN 4.966/2021 e IFRS 9) de um construtor interativo para
uma ferramenta **ponta a ponta**: construção, edição assistida, persistência,
aplicação em escala (pandas/Spark), **visualização gráfica** e **validação
regulatória com relatório automático**.

Todas as mudanças concentram-se em três arquivos do pacote, mais os testes:

| Arquivo | Papel |
| --- | --- |
| `src/yggdrasil/credit_risk/lgd/segmenter.py` | núcleo (`SequentialLGDSegmenter`) |
| `src/yggdrasil/credit_risk/lgd/ui.py` | interface interativa (`LGDSegmenterUI`) |
| `src/yggdrasil/credit_risk/lgd/README.md` | documentação do módulo |
| `tests/test_lgd_segmenter.py` | testes (pytest) |

**Status de testes ao final:** `75 passed, 1 skipped` (o *skip* é o teste de
`apply_spark`, que exige pyspark e roda no Databricks/CI).

---

## Mudanças por tema

### 1. CSI por variável (estabilidade das entradas)
- **O quê:** `csi()` e `csi_detalhe()` — Characteristic Stability Index de cada
  variável de entrada (bins fixados no DES), apontando *qual* característica está
  migrando entre DES e as demais amostras (OOT/…), com decomposição por faixa.
- **Por quê:** complementa o `psi()` (que mede a estabilidade da segmentação) com a
  estabilidade das variáveis antes mesmo de entrarem na árvore.
- **UI:** card **"PSI por variável (CSI)"**.

### 2. Salvar e carregar a árvore em JSON
- **O quê:** `to_dict()`, `save()`, e os classmethods `from_dict()` / `load()`
  (mais `_load_segments` e `_conditions_from_json`). Serializa a **estrutura** da
  árvore (segmentos + condições + metadados); ao carregar, as máscaras são
  reconstruídas a partir das condições sobre o DataFrame fornecido.
- **Por quê:** versionar a segmentação e reaplicá-la (mesmos dados ou novos) em
  qualquer máquina.
- **UI:** botões **"Salvar/Carregar árvore (JSON)"**.

### 3. Undo/redo de splits (UI)
- **O quê:** pilhas de estado + botões **◀ Desfazer / Refazer ▶** cobrindo toda
  alteração estrutural (split, fusão, recolher, podar, auto-fit, auto-merge, reset).
- **Por quê:** edição manual segura, sem medo de errar um passo.

### 4. Auto-merge automático
- **O quê:** `auto_merge()` — funde recursivamente pares de folhas-**irmãs**
  adjacentes estatisticamente indistinguíveis (teste de hipótese `p > alpha` ou gap
  de LGD abaixo do limite). Respeita folhas travadas (`protect`).
- **Por quê:** poda estatística que recupera a estrutura real de LGD.
- **UI:** botão **"Auto-fundir folhas"** + slider de alpha.

### 5. Juntar nó de faltantes com bin populado (`merge_missing`)
- **O quê:** introduzido o flag **`include_na`** nas condições (regra "bin **OU**
  faltante"), propagado por **todos os 6 interpretadores** de condição: predict
  pandas, PySpark, pyfunc do MLflow, serialização JSON (ida e volta), descrição por
  extenso e reconstrução de máscara. Novo método `merge_missing(folha)` e opção
  `auto_merge(include_missing=True)`.
- **Por quê:** muitas vezes o nó de faltantes é pequeno ou deve seguir uma regra de
  negócio que o agrupa a um bin populado da variável.
- **UI:** botão **"Juntar faltante nesta folha"**.
- **Validação:** régua (`predict`) reproduz exatamente as máscaras em memória em
  todos os interpretadores; round-trip JSON preserva `include_na`; pyfunc do MLflow
  validado de ponta a ponta.

### 6. Correção do `RuntimeWarning` de divisão por zero (CSI/IV)
- **O quê:** guarda contra dados degenerados (poucas linhas, variável constante,
  alvo NaN) antes do ajuste do binning + supressão do warning ao redor do `.fit()`
  via `_fit_optbinning_splits`.
- **Causa raiz:** o OptBinning, em `auto_monotonic` (`mean = sums / n_records`),
  emite "divide by zero" quando um *prebin* fica com 0 registros — **benigno** (ainda
  produz cortes válidos), mas poluía a saída no Databricks.

### 7. Layout da interface
- **O quê:** **Information Value** e **PSI por variável** lado a lado, separados por
  uma **barra vertical**; folhas criadas posicionadas **logo abaixo da
  segmentação**; nova ordem do painel (segmentação → folhas → IV | PSI → métricas →
  bootstrap → validação).

### 8. Visualização gráfica da árvore (`plot_tree`)
- **O quê:** imagem matplotlib da árvore (PNG/SVG/PDF) — cada nó com rótulo da
  condição, **representatividade (%)** e **LGD médio (DES)**; folhas com a nota.
- **Refinamentos pedidos durante a sessão:**
  - heatmap com **escala de cor fixa de 0 a 1**;
  - conteúdo enxuto: **só representatividade e LGD de DES** (removidos `n=` e a
    linha por amostra);
  - **texto forçado a caber nas caixas** (auto-encolhe a fonte medindo o tamanho
    real do texto vs. a caixa);
  - rótulo das folhas como **"folha N"** (em vez de "nota N");
  - **folhas ordenadas por `nota_lgd` da esquerda para a direita** (estrito para
    árvores monotônicas; agrupado por ramo quando não-monotônica, limitação inerente
    a layout hierárquico).
- **UI:** card "Árvore atual" com campo de caminho, botão **"Ver / salvar árvore
  (imagem)"** e botão **"Recolher imagem"**.

### 9. Botão "Limpar log" (UI)
- **O quê:** botão ao lado de **"Salvar no MLflow"** que limpa a área de
  preview/log (`_on_clear_log` → `out_log.clear_output()`).

### 10. Aplicar a régua numa tabela Spark (`apply_spark`)
- **O quê:** `apply_spark(sdf)` — executa a régua direto num Spark DataFrame,
  devolvendo-o com `segmento_lgd`, `folha` e `lgd_regua` (constrói as expressões
  `F.when().otherwise()`, com `isNull` para faltantes e `include_na`). **Valida que
  as colunas existem com o mesmo nome** (senão, `ValueError` listando as ausentes).
  Também `regua_features()` (colunas usadas pela árvore).
- **Por quê:** "reconstruir as folhas" numa tabela em escala, sem copiar/colar o
  código gerado por `to_pyspark`.
- **UI:** seção **"Reconstruir folhas em tabela Spark"** (tabela de entrada/saída +
  botão), com o resultado em `ui.spark_result`.
- **Nota:** a validação de colunas foi testada localmente; a execução completa roda
  num cluster real (o Spark local não sobe neste ambiente Windows).

### 11. Validação regulatória + relatório automático
- **Backtesting por safra** — `backtest(time_col)`: LGD previsto (régua) ×
  realizado por período, com `n`, gap e status (ok/alerta).
- **Monotonicidade** — `monotonicity_report()`: verifica se o LGD é monotônico
  crescente nas notas em cada amostra (flagra inversões no OOT).
- **Calibração** — `calibration_table()` e `plot_calibration()`: previsto (DES) ×
  realizado (OOT) por folha, com diagonal de calibração perfeita e faixa de
  tolerância.
- **Relatório de validação** — `validation_report("rel.md", time_col="dt_ref")`:
  documento **Markdown** (sem dependências extras) reunindo visão geral, imagem da
  árvore, folhas (com PSI), monotonicidade, PSI, CSI, métricas, calibração (tabela +
  imagem) e backtest. As imagens são salvas ao lado do `.md`.
- **UI:** card **"Validação & relatório"** com botões "Validar" e "Gerar relatório
  de validação (MD)".

### 12. Teste de hipótese restrito a folhas-irmãs
- **O quê:** o `p_vs_prox` (teste de LGD entre folhas, em `leaves(with_test=True)`)
  passou a comparar cada folha apenas com a **irmã adjacente (mesmo pai)**, na
  mesma ordem da fusão. Folhas de pais diferentes não são comparadas; a última
  irmã do grupo e o nó de faltantes ficam com `NaN`.
- **Por quê:** só folhas-irmãs são **diretamente comparáveis e fundíveis** — antes
  o teste comparava com a próxima folha na ordem global de LGD, que podia ser de
  outro ramo (não fundível), gerando leitura enganosa.
- **Arquivos:** `segmenter.py` (`_append_adjacency_test`), `ui.py` (legenda da
  tabela de folhas).

### 13. Poda por fusão de irmãs + documentação do `p_vs_prox`
- **Poda (`prune`) reescrita:** em vez de colapsar o split inteiro pela amplitude
  de LGD, agora **funde pares de folhas-irmãs adjacentes** que violem:
  - **diferença de LGD entre as duas irmãs < `min_lgd_gap`** (ex.: 0,03 = 3% → unir), ou
  - **representatividade de uma folha < `min_repr` %** (imaterial → unir com a irmã
    mais próxima em LGD).
  Prioriza o par de menor ΔLGD e itera até nenhum par violar. Reusa `merge_leaf`
  (só irmãs; faltantes fora).
- **UI:** slider renomeado para **"ΔLGD mínimo"**, tooltips e legenda do card de poda
  explicando os critérios.
- **Documentação do `p_vs_prox` na UI:** legenda da tabela de folhas expandida —
  explica que é o **p-valor de um teste de hipótese** (Mann-Whitney U padrão ou t de
  Welch, conforme o seletor **Teste**) comparando a **distribuição de LGD** da folha
  com a da **irmã adjacente** na referência DES (H₀: mesmo LGD), e como ler p
  alto/baixo.

### 14. Qualidade dos segmentos (gráficos)
- **O quê:** três gráficos novos no núcleo, expostos num card **"Qualidade dos
  segmentos"** na UI (logo após o IC bootstrap):
  - `plot_leaf_boxplots()` — **dispersão do LGD dentro de cada folha** (boxplot por
    nota, cor por LGD); caixa estreita = folha homogênea, larga = separa pouco;
  - `plot_target_hist()` — **distribuição do LGD da carteira** (bimodalidade);
  - `plot_feature_lgd(feature, sid)` — **LGD médio por faixa** da variável numa
    folha (com repr% e flag de monotonicidade), para escolher o split antes do
    `grow`.
- **Por quê:** a média + IC não mostram a heterogeneidade intra-folha nem a forma
  da relação variável × LGD; estes gráficos preenchem essa lacuna.

### 15. Preview da variável no fluxo de split + imagem responsiva
- **Preview gráfico junto da tabela:** o gráfico de **LGD por faixa** da variável
  (`plot_feature_lgd`, agora com parâmetro `splits`) passou a ser renderizado
  **embaixo da tabela do 👁 Preview** (mesmos bins do split), em vez de um botão
  separado no card de qualidade.
- **Imagem da árvore que cabe no painel:** o display inline passou a escalar a
  figura (PNG em base64 com `max-width:100%`) — antes, com **muitas folhas**, a
  imagem larga não aparecia. O **arquivo salvo continua em tamanho real**. O mesmo
  display responsivo vale para boxplot, histograma e calibração (helper
  `_display_fig`).

### 16. Gráfico da variável: barras (representatividade) + linha (LGD médio)
- **O quê:** `plot_feature_lgd` virou um gráfico de binning de risco com **eixo
  duplo**: **barras = representatividade (%)** (eixo esquerdo, teal) e **linha =
  LGD médio por faixa** (eixo direito, vermelho), com legenda, eixos coloridos e
  título indicando qual métrica é qual. Antes as barras eram o próprio LGD.
- **Notebook de validação:** criado `notebooks/credit_risk/02_relatorios_validacao_lgd.ipynb`
  — só relatórios de validação (folhas/PSI/CSI/métricas/bootstrap/backtest/
  monotonicidade/calibração/qualidade/relatório consolidado), rodando de ponta a
  ponta com dados sintéticos.

### 17. Auditoria de bugs (multi-agente) + correções
Rodei uma auditoria de revisão multi-agente (7 dimensões → verificação adversarial
de cada achado): **33 achados, 23 confirmados** (0 high, 13 medium, 10 low). Corrigidos:

- **Ordenação de irmãs inconsistente (núcleo):** `merge_leaf` ordenava categóricas
  pela média de **todas** as amostras, enquanto `prune`/`auto_merge`/teste usavam só
  **DES** → a fusão errava o par ou abortava. Unificado para o LGD de **DES**
  (`_leaf_target`), inclusive na nota (`leaves().lgd_medio` agora é a base da régua).
- **Paridade pandas/Spark:** `to_pyspark`/`apply_spark` agora fazem `cast("string")`
  na coluna categórica (espelha o `astype(str)` do pandas/pyfunc).
- **`grow`:** valida grupos categóricos **disjuntos** (erro claro) e não cria mais
  **split degenerado de 1 filho** (só rebaixa o pai com ≥2 filhos).
- **`prune`** passou a respeitar **folhas travadas** (`protect`); a UI passa os locks.
- **Robustez a NaN/vazio:** `_leaf_target` dropa NaN; `metrics` ignora alvo NaN;
  `_bin_table` e `leaves` tratam tabela/`n_total` vazios; `_regua_dict` cai no LGD
  global se a folha estiver vazia (sem NaN na régua); construtor dá erro claro se
  faltar a `sample_col`; `_df_to_md` trata `pd.NA`; `validation_report` não vaza
  figuras; o no-op de "juntar faltante" não destrói mais a pilha de refazer.
- **Adicionados 5 testes de regressão.**

Deixados de fora (baixo valor/por design): empates exatos de LGD na monotonicidade,
clip de LGD fora de [0,1] nos gráficos (LGD ∈ [0,1] por definição), valor de feature
literalmente igual a -inf, e `tree()` em DataFrame vazio.

---

## Dependências

No `pyproject.toml` (já presentes): núcleo com `numpy`, `pandas`, `optbinning`,
`mlflow`, `matplotlib`; extras `ui` (ipywidgets), `spark` (pyspark, opcional —
no Databricks já vem no cluster).

## Pendências / próximos candidatos (não feitos)

- **Downturn LGD + Margem de Conservadorismo (MoC)** na régua.
- **Export do relatório em DOCX/PDF** (hoje é Markdown).
- **Champion-challenger** (diff entre duas árvores salvas em JSON).
- **Expansão de domínio:** PD (scorecard/segmentação), EAD/CCF e ECL IFRS 9
  (PD×LGD×EAD + staging) — direção declarada do pacote.
