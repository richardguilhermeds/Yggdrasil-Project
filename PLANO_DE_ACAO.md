# Plano de ação — melhorias nas interfaces de árvore e no ModelSegmenter

> Origem: revisão multi-agente com **verificação adversarial** (39 agentes, 19 fatias
> cobrindo `tree/segmenter.py`, `tree/ui.py`, `model/segmenter.py`, `model/ui.py` +
> paridade cruzada). 49 achados sobreviveram à verificação; consolidados aqui em 42
> itens, **ordenados da maior para a menor severidade**. Data: 2026-07-09.

**Como usar:** marque `[x]` ao concluir. Cada item traz `arquivo:linha`, o problema e a
correção sugerida. `(P)` = achado plausível (depende de contexto/uso), demais são
confirmados no código.

**Tema de raiz que atravessa vários itens:** centralizar os *helpers* de classificação
(`_classifica_psi`/`_classifica_iv`), contagem de inversões e a "régua" (score da folha /
tolerância de backtest) num módulo comum (`credit_risk/_common`) reusado pelas duas
famílias — resolve de uma vez P6, P7, P9, P12, P15, P24, P25, P26.

---

## ✅ Feito nesta sessão

- [x] **P0 — Zoom nos gráficos de inversão de rating** — `model/segmenter.py` (`plot_rating_inversion_by_sample`/`_by_safra`) + `model/ui.py` (`_render_ratings` e aba Ratings).
  Botão **🔍 Zoom** (aperta o eixo Y aos dados) + **↺** (volta ao eixo cheio) + campos **mín/máx %** opcionais, abaixo de cada gráfico de inversão (amostras e safras), para inspecionar cruzamentos/inversões entre safras sem fixar o eixo na mão. Novos parâmetros `ylim=(lo,hi)` e `auto_zoom=` nas duas funções de plot (retrocompatíveis). Verificado por smoke test + suíte de rating/inversão/UI (77 testes, 0 falhas).
- [x] **P1–P8 — os 8 achados de severidade ALTA corrigidos e verificados** (`tree/segmenter.py`, `tree/ui.py`, `model/segmenter.py`, `model/ui.py`). Verificação comportamental dedicada (P1 poda com irmã expandida; P2 platôs; P3 reprojeção de ratings; P5 IC de média na regressão; P6/P7 sinal e tol do backtest) + suítes tree/model/ratings/UI/spark reexecutadas, **0 falhas**.
- [x] **Bônus (pandas 3.0 · P41 parcial) — `_leaf_id_series` CoW-safe** — `tree/segmenter.py:1420`. Sob pandas ≥3.0 (copy-on-write, o caso do ambiente local) `Series.values` é read-only e a atribuição in-place `s.values[mask]=sid` quebrava `metrics()`/UI da árvore. Corrigido construindo o array numpy gravável antes de embrulhar em Series.
- [x] **P9–P23 — os 15 achados de severidade MÉDIA corrigidos e verificados** (`tree/segmenter.py`, `tree/ui.py`, `model/segmenter.py`, `model/ui.py`). Verificação comportamental dedicada (P9 bin NA fora do mono; P12 `n_inversoes` unificado como descidas adjacentes; P13 guards; P14 PSI não-NaN; P16 val_sample honesto; P17 transform persiste; P18 attrs do backward; P19 desempate por inversões) + **360 testes** das suítes reexecutados, **0 falhas**.
- [x] **P24–P26 — refactor de raiz**: helpers puros centralizados em `credit_risk/_common.py` (fonte única, sem drift); `_classifica_psi` com guard NaN (P25), `_classifica_iv` sem default (P26).
- [x] **P27–P42 — os 19 achados de severidade BAIXA corrigidos e verificados** (backtest/plot, SHAP `plt.close`, tuning degenerado, p-valores por posição, ranges de regressão, escala de cor DES, memoização; tokens de tema/PSI, gauge NaN, contraste WCAG; validação log-scale, reuso backward por amostra, clamp, dirty por algo/HP). Verificação comportamental (P27, P28, P31, P35, P40) + **360 testes**, **0 falhas**. **Revisão 100% endereçada.**

---

## 🔴 Severidade ALTA (resultado errado, silencioso)

- [x] **P1 — `prune` pode abortar a poda inteira** — `tree/segmenter.py:840`.
  Pareia folhas por índices consecutivos ignorando uma irmã *expandida* (não-folha) no meio; `merge_leaf` aborta e a salvaguarda `set(segments)==antes` faz `break`, encerrando todo o laço. → Iterar sobre `_adjacent_sibling_pairs(pai)` (fonte canônica de `auto_merge`/`plot_tree`) ou trocar `break`→`continue`.
- [x] **P2 — `_trend` conta inversões espúrias em platôs** — `model/segmenter.py:257`.
  `np.sign(0)==0`: um empate entre ratings adjacentes vira "mudança de sinal" → escada crescente marcada como não-monotônica (alimenta `monotonicity_report`). → Contar só sobre `diffs != 0`.
- [x] **P3 — Retreino deixa ratings do modelo antigo (stale)** — `model/ui.py:2642` (`_on_fit`/`_finish_tune`) + `model/segmenter.py:2188` (`fit`).
  `fit`/`tune` recalculam `score_` mas não tocam em `rating_`; a UI re-renderiza tudo menos os ratings, e `assign()`/export misturam score novo com ratings velhos. → Após retreino, reprojetar ratings sobre o novo score (ou limpá-los) e atualizar o pill/aba Ratings.
- [x] **P4 — `_on_export` lê coluna inexistente `segmento_pd_nota`** — `tree/ui.py:3548`.
  `assign()` gera `segmento_nota`/`segmento_desc`; o handler sempre dispara `KeyError` e a distribuição de notas nunca aparece. → `self.result['segmento_nota']` + try/except. *(one-liner, confiança 0,97)*
- [x] **P5 — `plot_leaf_target_hist`: IC de Wilson (binomial) sobre LGD contínuo** — `tree/segmenter.py:2875`.
  Ramo de regressão calcula ponto/IC via `_wilson_ci(int(y.sum()), len(y))` — `int()` trunca a soma e o IC binomial é inválido p/ alvo contínuo. → Bifurcar por `_is_clf`: média + IC de média na regressão.
- [x] **P6 — Backtest: `gap` com sinal OPOSTO entre árvore e modelo** — `tree/segmenter.py:2080` vs `model/segmenter.py:4545`.
  Árvore usa `realizado − previsto`; modelo, `previsto − realizado`. Mesmo portfólio → sinal invertido. → Padronizar (sugestão: `realizado − previsto`, alinhado a `calibration_table`).
- [x] **P7 — Backtest: tolerância default diverge por `task_type`** — `model/segmenter.py:4529` vs `tree/segmenter.py:2063`.
  Árvore: `0.03 clf / 0.10 reg`; modelo: `0.10` fixo. Para PD, gap de 0,06 é "alerta" na árvore e "ok" no modelo. → Default sensível ao `task_type` no modelo.
- [x] **P8 — Faltantes numéricos divergem pandas × Spark** — `tree/segmenter.py:4405`.
  (Corrigido em `apply_spark` via normalização NaN→NULL nas colunas float, sem mutar as colunas de saída; `to_pyspark`/`to_sql` documentam o contrato NULL-não-NaN; a pyfunc já usa pandas.)
  No Spark `NaN` é o maior valor e `isNull()` não o pega → linha `NaN` cai no bin mais alto (em pandas fica órfã); `include_na` também não recupera. → Normalizar `NaN→NULL` na entrada Spark e documentar o contrato.

---

## 🟡 Severidade MÉDIA

- [x] **P9 — Bin de faltantes (NA) entra na monotonicidade** — `tree/segmenter.py:675` + `model/segmenter.py:1007`.
  A bin NA fica fixa no fim e corrompe `mono_ok`/`n_inversoes`. → Excluir `kind=='na'` do teste (nos dois módulos).
- [x] **P10 — Histogramas de regressão descartam dados fora de [0,1]** — `tree/segmenter.py:2298` (+`2340`).
  `range=(0,1)` no `hist` descarta LGD<0 ou >1 (contagens não somam n); a média/axvline fica incoerente. → Derivar o range dos dados quando houver massa fora de [0,1].
- [x] **P11 — `plot_variable_distribution` coage categórica→float64 no fallback** — `tree/segmenter.py:3058` + `model/segmenter.py:1369`.
  Com binning vazio (categórica de 1 categoria / folha pura), `to_numpy(float64)` sobre strings levanta `ValueError`. → Checar `_detect_kind=='num'` antes; senão desenhar `value_counts` / "sem faixas".
- [x] **P12 — `n_inversoes` com semântica diferente entre árvore e modelo** — `tree/segmenter.py:2112` vs `model/segmenter.py:257`. (Unificado: ambos contam **descidas adjacentes** na escada esperada; o `inverte_vs_DES` do modelo segue como diagnóstico complementar.)
  `[1,2,1,2]` → árvore 1, modelo 2, na mesma coluna. → Uma única definição reusada nos dois relatórios.
- [x] **P13 — `plot_roc`/`plot_ks` do modelo sem guard de `task_type`** — `model/segmenter.py:3208`/`3229`.
  Em regressão, KS olha só alvo ==0/1 (sem sentido); ROC quebra com erro críptico. → Guard `if task_type != 'classification': raise`, como `plot_cap`/`plot_lift`.
- [x] **P14 — PSI de rating por amostra inclui NaN no denominador** — `model/segmenter.py:4312` (e `4346`).
  Normaliza por total (com rating NaN) enquanto `rating_psi_by_safra` usa não-NaN → divergência contra a própria docstring. → Padronizar denominador para ratings não-NaN.
- [x] **P15 — `metrics` usa fallback ≠ da régua real (folha só-OOT)** — `tree/segmenter.py:1911`.
  Pontua folha sem dados na DES com PD global, divergindo de `_predicted_series`/`backtest`. → Unificar no fallback da régua (média da folha em todas as amostras).
- [x] **P16 — `tune_optuna` rotula `val_sample` como OOT falsamente** — `model/segmenter.py:2376`.
  Guard morto: com OOT <50 linhas, valida em holdout 25% do DES mas marca `val_sample=OOT` (vai p/ tag MLflow). → Capturar `used_oot` antes do split e usar `'split'` caso contrário.
- [x] **P17 — `feature_transform` não é persistido em `to_dict`/`from_dict`** — `model/segmenter.py:5045`.
  Modelo `woe` volta a `raw` após save→load (some `<feat>_faixa`; refits pós-load erram). → Incluir/restaurar `feature_transform`.
- [x] **P18 — `apply_backward_selection` ignora algo/transform do backward** — `model/segmenter.py:3112`.
  Reajusta com os valores vigentes, não os que geraram a ordem de remoção. → Ler `result.attrs['algorithm']` (e persistir/ler `transform`) ou avisar na divergência.
- [x] **P19 — `suggest_n_ratings`: desempate ignora a contagem de inversões** — `model/segmenter.py:4066`.
  Guarda só o flag `mono_ok`, não `sample_inv`; 1 vs 15 inversões tratadas como iguais. → Guardar `int(sample_inv)` e usá-lo como 1º critério.
- [x] **P20 — `_checkpoint()` antes da mutação corrompe undo/redo** — `tree/ui.py:3496` (`_on_split`/`_on_prune`/`_on_collapse`/`_on_merge`). (Novo helper `_revert_checkpoint`; `_on_merge` também detecta no-op.)
  Em falha da mutação, fica snapshot espúrio e o redo é destruído. → Checkpoint só após sucesso, ou restaurar no `except` (padrão de `_on_merge_missing`).
- [x] **P21 — `_on_autofit` sem try/except** — `tree/ui.py:2812`.
  `fit_auto` sem proteção deixa estado dessincronizado em falha. → Envolver em try/except revertendo o checkpoint.
- [x] **P22 — Strings de dados interpoladas em HTML sem escape** — `tree/ui.py:3153` (+`3588`). (Escape em `_var_cards_html`, `_boot_forest_html` e `_rebuild_cat_box`.)
  `'ATACADO & VAREJO'`, `<`, `>` quebram o markup. → `html.escape()` nos fragmentos vindos de dados.
- [x] **P23 — PD do dropdown de folhas usa amostra cheia (chips/header usam DES)** — `tree/ui.py:2371`.
  Mesma folha exibe PD diferente no dropdown e nos painéis. → Reusar `_node_value(sid, ref_sample)`.

---

## 🟢 Severidade BAIXA (qualidade / cosmético / robustez menor)

- [x] **P24 — Duplicação de helpers entre tree e model** — `model/segmenter.py:194` (e correspondentes na árvore).
  `_fmt`, `_fmt_safras`, `_fit_optbinning_splits`, `_count_inversions` são cópias; `_classifica_psi`/`_iv` já divergiram. → Extrair para `credit_risk/_common` e importar nos dois.
- [x] **P25 — `_classifica_psi` da árvore não trata None/NaN** — `tree/segmenter.py:95`.
  Classifica PSI inválido como "instável". → Guard `if not np.isfinite(psi): return '—'` (alinha com o model).
- [x] **P26 — `_classifica_iv`: default de `task_type` diverge** — `model/segmenter.py:214` vs `tree/segmenter.py:113`.
  Árvore tem default (pode classificar IV de regressão pela escala binária); model é obrigatório. → Assinatura única (sem default) ao centralizar.
- [x] **P27 — `backtest` (tol absoluta) vs `plot_backtest` (banda relativa)** — `model/segmenter.py:4550`.
  Tabela e gráfico podem discordar sobre "fora da banda". → Unificar a definição de tolerância.
- [x] **P28 — `(P)` SHAP beeswarm/bar usam `plt.figure`/`gcf` sem `close`** — `model/segmenter.py:3877`.
  Possível display duplicado/vazamento de figuras no inline. → `plt.close(fig)` antes do return.
- [x] **P29 — Trials degenerados (métrica NaN) viram COMPLETE e `fit_best` reajusta** — `model/segmenter.py:2404`.
  Sentinela `-1e9` deixa o estudo "concluído" mesmo vazio. → Tratar `best_value<=-1e8` como degenerado (não reajustar, avisar).
- [x] **P30 — Forest plot do bootstrap clampa eixo em [0,1]** — `tree/ui.py:3565`.
  Quebra alvos de regressão (LGD) fora do intervalo. → Só clampar quando `_is_clf`.
- [x] **P31 — `plot_leaf_boxplots` fixa `ylim(0,1)`/`Normalize(0,1)`** — `tree/segmenter.py:2272`.
  Recorta caixas de LGD fora de [0,1]. → Ajustar aos dados observados.
- [x] **P32 — Tokens de tema de PSI divergentes** — `tree/ui.py:1871`.
  Árvore usa `--risk-*`; cartões usam `--ok/warn/bad-tx`. → Padronizar nos tokens semânticos.
- [x] **P33 — Gauge de PSI não trata NaN** — `tree/ui.py:3159`.
  `nan%` quebra a barra e vira classe "red". → Filtrar NaN / tratar como "—".
- [x] **P34 — Contraste WCAG no heatmap categórico (fallback)** — `tree/ui.py:476`.
  Texto branco sobre fundo claro (~2:1). → Escolher preto/branco pela luminância real.
- [x] **P35 — Escala de cor da árvore usa `vmax` de todas as amostras (rótulo diz DES)** — `tree/segmenter.py:1504`.
  Cor/rótulo incoerentes quando OOT puxa a média. → Restringir `vmax` à máscara de referência.
- [x] **P36 — Limite de params log-scale não validado (>0) no tuning** — `model/ui.py:2301`.
  `low<=0` com `log=True` faz todos os trials falharem. → Piso positivo + aviso.
- [x] **P37 — "Escolha ótima" reusa backward ignorando a amostra atual** — `model/ui.py:1861`.
  Usa métricas da amostra antiga. → Incluir amostra/algoritmo na condição de reuso.
- [x] **P38 — `(P)` `_on_backelim` não clampa `min_features`** — `model/ui.py:2940`.
  Slider com max obsoleto pode rodar poucos/nenhum passo. → Clampar como `_on_feat_optimal`.
- [x] **P39 — `(P)` Trocar algo/HP após treino não marca dirty** — `model/ui.py:1096`.
  Métricas/fórmula do modelo antigo sem aviso de obsolescência. → Encadear `_mark_dirty` (ou documentar por que não).
- [x] **P40 — p-valores de Wald mapeados por nome de exibição colidem** — `model/segmenter.py:2745`.
  Rótulos idênticos colapsam a dict → p-valor errado. → Mapear por posição/índice do termo.
- [x] **P41 — `_leaf_id_series`: chave de memo ignora `leaf_ids`** — `tree/segmenter.py:1426`.
  Crash de CoW (pandas 3.0) já resolvido antes; agora `leaf_ids` entra na chave de memo (subconjunto ≠ conjunto completo).
- [x] **P42 — `(P)` `_agg_memo` faz cópia rasa de dict/list** — `tree/segmenter.py:436`.
  Retorno mutável compartilhado pode contaminar o cache. → `deepcopy` p/ aninhados ou documentar como somente-leitura.
