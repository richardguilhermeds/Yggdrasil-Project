# Metodologia da Esteira Yggdrasil

O `yggdrasil` é um pacote Python para construção, validação e monitoramento de modelos de **risco de crédito** (PD, LGD, EAD e perda esperada). Este documento descreve a **metodologia** por trás de cada componente da esteira — o *porquê* de cada método, sua fórmula, como ele é computado no código (com referência às funções reais) e suas armadilhas — e não o uso de API. O público-alvo são **cientistas de dados e modeladores de risco** que precisam entender, auditar e defender as escolhas metodológicas perante validação independente e governança.

A esteira organiza-se em torno de um contrato de amostras: a base de **desenvolvimento (DES)** é a fotografia sobre a qual todos os cortes, cutoffs e referências de estabilidade são **aprendidos e congelados**; a base **out-of-time (OOT)** é uma janela temporal posterior, usada para testar generalização; safras de produção são monitoradas continuamente contra a referência DES. Todo este documento adota a convenção de que o **alvo (target) é binário com $y=1$ denotando o evento de interesse — o "mau" (inadimplência/default) — e $y=0$ o "bom"** (adimplente). Métricas de regressão aplicam-se a alvos contínuos (ex.: LGD, EAD), com $y$ o valor observado e $\hat{y}$ o previsto.

---

## Índice

1. [Métricas de discriminação, erro e shifts](#1-métricas-de-discriminação-erro-e-shifts)
2. [Grupos homogêneos (ratings) e binning monotônico](#2-grupos-homogêneos-ratings-e-binning-monotônico)
3. [Estabilidade e monitoramento no tempo (PSI/CSI)](#3-estabilidade-e-monitoramento-no-tempo-psicsi)
4. [Interpretabilidade (SHAP) e EDA de features](#4-interpretabilidade-shap-e-eda-de-features)
5. [Referências](#5-referências)

---

## 1. Métricas de discriminação, erro e shifts

Esta seção descreve as métricas que o `yggdrasil` usa para avaliar **poder de discriminação** (modelos de classificação/risco), **erro de calibração/previsão** (modelos de regressão) e **estabilidade temporal** das métricas entre as amostras de desenvolvimento (DES) e *out-of-time* (OOT). O fio condutor é: separar quem é bom de quem é mau (discriminação), errar pouco na magnitude prevista (erro) e garantir que o desempenho observado no desenvolvimento se sustente fora da janela temporal de treino (shifts).

### 1.1 KS (Kolmogorov–Smirnov)

**O que mede.** A máxima separação entre as distribuições acumuladas do score para as classes positiva ($y=1$, mau) e negativa ($y=0$, bom). É a métrica de discriminação mais usada em risco de crédito.

**Fórmula.** Sejam $F_{\text{bom}}(s)$ e $F_{\text{mau}}(s)$ as funções de distribuição acumulada (CDF) empíricas do score em cada classe:

$$KS = \max_{s}\; \left| F_{\text{bom}}(s) - F_{\text{mau}}(s) \right|$$

**Como o yggdrasil computa.** `ks_statistic` (em `metrics/classification.py`) aplica o teste de Kolmogorov–Smirnov entre os scores dos bons e dos maus, retornando a estatística $KS$. Também é reportada dentro de `classification_metrics`.

**Interpretação / limiares (orientativos).**

| Faixa de KS | Leitura |
|---|---|
| $< 0{,}20$ | discriminação fraca |
| $0{,}20$–$0{,}40$ | aceitável a bom |
| $> 0{,}40$ | forte |
| $> 0{,}75$ | suspeito de *leakage* ou vazamento de tempo |

**Armadilhas.** O KS captura a separação em **um único ponto** do score (onde a distância das CDFs é máxima) e ignora o resto da curva — dois modelos com KS idêntico podem ter perfis de ranqueamento bem diferentes. É sensível ao ponto de corte populacional e pode ser inflado por *target leakage*. KS muito alto raramente é boa notícia: quase sempre indica variável que carrega o próprio desfecho.

### 1.2 AUC e Gini

**O que medem.** A AUC (área sob a curva ROC) é a probabilidade de que um mau ($y=1$) sorteado ao acaso receba score maior que um bom ($y=0$) sorteado ao acaso — uma medida **global** de ordenação. O Gini é uma reescala linear da AUC, popular em crédito por variar em torno de zero.

**Fórmulas.**

$$\text{AUC} = \mathbb{P}\big(\text{score}_{\text{mau}} > \text{score}_{\text{bom}}\big), \qquad \text{Gini} = 2\,\text{AUC} - 1$$

**Como o yggdrasil computa.** Em `classification_metrics`: `auc = roc_auc_score(...)` e `gini = 2*auc - 1`.

**Interpretação / limiares.** AUC $= 0{,}5$ (Gini $= 0$) equivale ao acaso; AUC $= 1$ (Gini $= 1$) é separação perfeita. Em crédito, Gini típico de bons modelos fica na faixa $0{,}40$–$0{,}70$; Gini $> 0{,}90$ é forte indício de *leakage* (mesma bandeira usada pela EDA em `importance_ranking`).

**Armadilhas.** A AUC é **agnóstica à calibração** — mede só a ordenação, não se a probabilidade prevista bate com a frequência observada (para isso use Brier/log loss). É insensível a desbalanceamento extremo no sentido de poder dar uma falsa sensação de qualidade quando a classe positiva é rara. Diferente do KS, integra toda a curva, então é menos volátil mas também menos "diagnóstica" sobre onde o modelo separa melhor.

### 1.3 Acurácia, F1, precisão e recall

**O que medem.** Métricas calculadas **após binarizar** o score por um ponto de corte. A partir da matriz de confusão ($TP, FP, TN, FN$, com o evento $y=1$ como classe positiva):

$$\text{precisão} = \frac{TP}{TP + FP}, \qquad \text{recall} = \frac{TP}{TP + FN}$$

$$\text{F1} = 2\cdot\frac{\text{precisão}\cdot\text{recall}}{\text{precisão}+\text{recall}}, \qquad \text{acurácia} = \frac{TP + TN}{TP + FP + TN + FN}$$

Precisão = fração de acertos entre os classificados como mau; recall (sensibilidade) = fração de maus efetivamente capturados; F1 = média harmônica entre as duas; acurácia = fração total de acertos.

**Como o yggdrasil computa.** Todas saem de `classification_metrics`, avaliadas no **cutoff KS-ótimo** (ver §1.6): `accuracy`, `f1`, `precision`, `recall`.

**Interpretação.** Não há limiar universal — dependem da prevalência e do *trade-off* de negócio (custo de aprovar um mau vs. recusar um bom). Em populações de crédito, com maus raros, a acurácia tende a ser enganosamente alta (basta prever "todos bons"), por isso pondere precisão/recall e priorize métricas de ranqueamento (KS/Gini) na seleção.

**Armadilhas.** São **dependentes do corte** e, portanto, mudam se o cutoff muda. A acurácia é traiçoeira sob desbalanceamento. Como o `yggdrasil` fixa o corte no KS-ótimo (otimizado em DES), essas métricas herdam a sensibilidade desse ponto — ver a discussão de cutoff (§1.6) e de shifts (§1.8).

### 1.4 Brier score

**O que mede.** Erro quadrático médio entre a **probabilidade prevista** e o desfecho binário observado. É uma métrica de **calibração + discriminação** (regra de pontuação própria — *proper scoring rule*).

**Fórmula.**

$$\text{Brier} = \frac{1}{n}\sum_{i=1}^{n}\big(\hat{p}_i - y_i\big)^2$$

com $y_i \in \{0,1\}$ e $\hat{p}_i$ a probabilidade prevista do evento ($y=1$).

**Como o yggdrasil computa.** `classification_metrics`: `brier = brier_score_loss(...)`.

**Interpretação.** Menor é melhor; $0$ é perfeito. Diferente de AUC/KS, o Brier **penaliza probabilidades mal calibradas** mesmo que a ordenação esteja correta — útil quando a probabilidade prevista será usada diretamente (provisão, precificação).

**Armadilhas.** É afetado pela prevalência (um modelo trivial que preveja sempre a taxa-base já obtém um Brier não-trivial), então compare contra um *baseline*. Não decompõe sozinho calibração vs. resolução — para diagnosticar a fonte do erro, combine com curvas de calibração.

### 1.5 Log loss (entropia cruzada binária)

**O que mede.** Verossimilhança negativa média do modelo probabilístico; também uma regra de pontuação própria, porém com **penalização mais severa** para erros confiantes.

**Fórmula.**

$$\text{LogLoss} = -\frac{1}{n}\sum_{i=1}^{n}\Big[\,y_i\ln\hat{p}_i + (1-y_i)\ln(1-\hat{p}_i)\,\Big]$$

**Como o yggdrasil computa.** `classification_metrics`: `logloss = log_loss(...)`.

**Interpretação.** Menor é melhor. Comparada ao Brier, a log loss cresce **sem limite** quando o modelo atribui probabilidade próxima de $0$ a um evento que ocorre (ou vice-versa), penalizando duramente o excesso de confiança.

**Armadilhas.** É **muito sensível a probabilidades extremas** ($\hat{p}\to 0$ ou $1$) — um único caso confiante e errado pode dominar a métrica; exige *clipping*/regularização das probabilidades. Por isso é mais volátil que o Brier entre amostras pequenas.

### 1.6 Cutoff KS-ótimo

**O que é.** O ponto de corte usado para binarizar o score nas métricas dependentes de corte. O `yggdrasil` o define como o limiar que **maximiza a separação na curva ROC**, ou seja, o ponto de maior estatística KS:

$$\text{cutoff}^{*} = \arg\max_{s}\; \big(\text{TPR}(s) - \text{FPR}(s)\big)$$

Esse $\arg\max(\text{TPR}-\text{FPR})$ é exatamente o ponto onde as CDFs das classes mais se distanciam — daí o nome "KS-ótimo".

**Como o yggdrasil computa.** Internamente em `classification_metrics`, varrendo a curva ROC e tomando o limiar de máximo $\text{TPR}-\text{FPR}$. Em avaliações *out-of-time*, `metric_by_sample` (em `metrics/shift.py`) **estima o cutoff no DES e o reaplica no OOT**, garantindo que a amostra futura seja julgada pelo mesmo critério de decisão do desenvolvimento.

**Interpretação.** É um corte **estatístico**, que equilibra sensibilidade e especificidade — não necessariamente o corte **ótimo de negócio**, que dependeria da matriz de custos (aprovar mau vs. recusar bom).

**Armadilhas.** Por ser otimizado nos dados, sofre *overfitting* do ponto de corte: o KS-ótimo do DES quase nunca é o KS-ótimo do OOT. Reaplicar o corte do DES no OOT (como faz o `yggdrasil`) é o procedimento **correto** — recalibrar o corte na amostra de teste mascararia a degradação. Não confunda o cutoff KS-ótimo com um *threshold* de aprovação operacional.

### 1.7 Métricas de regressão: RMSE, MAE, MAPE robusto, sMAPE, MedAE, R² e viés

Para alvos contínuos (ex.: LGD, EAD, perda esperada), `regression_metrics` (em `metrics/regression.py`) calcula o conjunto abaixo. Seja $e_i = \hat{y}_i - y_i$ o resíduo.

**RMSE — raiz do erro quadrático médio.**
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} e_i^{\,2}}$$
Penaliza erros grandes quadraticamente; na mesma unidade do alvo. **Armadilha:** muito sensível a *outliers*.

**MAE — erro absoluto médio.**
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |e_i|$$
Robusto a *outliers* relativamente ao RMSE; interpretação direta. Se RMSE $\gg$ MAE, há poucos resíduos muito grandes.

**MAPE robusto — erro percentual absoluto médio.**
$$\text{MAPE} = \frac{1}{|\mathcal{I}|}\sum_{i\in\mathcal{I}} \left|\frac{e_i}{y_i}\right|, \qquad \mathcal{I} = \{\,i : |y_i| \ge \varepsilon\,\}$$
A versão do `yggdrasil` é **robusta**: ignora observações com $|y_i| < \varepsilon$, evitando a divisão por (quase) zero que explode o MAPE clássico. **Armadilha:** assimétrico — penaliza mais a superestimação; e ao descartar $|y_i|<\varepsilon$ deixa de avaliar justamente a região de alvos pequenos (registre quantos pontos foram excluídos).

**sMAPE — MAPE simétrico.**
$$\text{sMAPE} = \frac{1}{n}\sum_{i=1}^{n} \frac{|e_i|}{(|y_i| + |\hat{y}_i|)/2}$$
Corrige parte da assimetria do MAPE e é limitado, mas fica instável quando $|y_i|+|\hat{y}_i|\to 0$.

**MedAE — erro absoluto mediano.**
$$\text{MedAE} = \operatorname{mediana}\big(|e_1|,\dots,|e_n|\big)$$
Altamente robusto: ignora completamente o efeito de *outliers*; bom complemento ao RMSE.

**R² — coeficiente de determinação.**
$$R^{2} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$
Fração da variância explicada. $1$ = perfeito; $0$ = equivale a prever a média; **pode ser negativo** se o modelo for pior que a média. **Armadilha:** infla com o número de preditores e é sensível à variância do alvo na amostra.

**Viés médio (mean bias).**
$$\text{viés} = \frac{1}{n}\sum_{i=1}^{n} e_i = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)$$
Mede erro **sistemático** (sub/superestimação média). No `yggdrasil`, **viés $> 0$ indica modelo conservador** (previsão acima do observado — ex.: superestima perda). **Armadilha:** erros de sinais opostos se cancelam, então um viés próximo de zero **não** implica bom ajuste — leia-o sempre junto com MAE/RMSE.

### 1.8 Shifts DES → OOT (absoluto e relativo)

**O que medem.** A **degradação de desempenho** entre a amostra de desenvolvimento (DES) e a amostra *out-of-time* (OOT), para qualquer métrica das seções anteriores. É o teste-chave de generalização temporal.

**Fórmulas.** Para uma métrica $M$:

$$\text{shift}_{\text{abs}} = M_{\text{OOT}} - M_{\text{DES}}, \qquad \text{shift}_{\text{rel}} = \frac{M_{\text{OOT}} - M_{\text{DES}}}{\lvert M_{\text{DES}} \rvert}$$

**Como o yggdrasil computa.** `metric_by_sample` calcula cada métrica por amostra (com o **cutoff estimado no DES e reaplicado no OOT**, conforme §1.6); `metric_shifts` (ambas em `metrics/shift.py`) deriva `shift_abs = oot - des` e `shift_rel = (oot - des)/|des|`.

**Por que comparar amostras out-of-time.** A validação *in-sample* (ou mesmo *cross-validation* aleatório) compartilha a janela temporal do treino e **não captura desvios populacionais** que surgem com o tempo: mudança de perfil de clientes, sazonalidade, alterações de política de crédito, choques macroeconômicos. O OOT — uma janela **posterior** à de treino — é o análogo, em validação, do ambiente de produção. Uma queda forte de KS/Gini do DES para o OOT sinaliza modelo pouco robusto, mesmo que o desempenho *in-sample* seja excelente.

**Interpretação.** O sinal importa: para métricas em que **maior é melhor** (KS, AUC, Gini, R²), $\text{shift}_{\text{abs}} < 0$ é degradação; para métricas de **erro** (RMSE, MAE, log loss, Brier), $\text{shift}_{\text{abs}} > 0$ é piora. O `shift_rel` normaliza pela escala da métrica no DES, permitindo comparar magnitudes entre métricas heterogêneas. Como referência prática, quedas relativas de Gini/KS acima de $\sim 10\%$ merecem investigação; acima de $\sim 25\%$ costumam ser críticas.

**Armadilhas.** (i) O `shift_rel` **explode quando $M_{\text{DES}} \approx 0$** (denominador pequeno) — interprete com cautela métricas naturalmente próximas de zero (ex.: viés). (ii) O shift mede *o quê* mudou, não *por quê*: cruze-o sempre com o **PSI do score** e o **CSI/PSI das features** (§3) para distinguir degradação por mudança populacional de degradação por quebra do relacionamento feature→alvo. (iii) Shifts em métricas dependentes de corte (acurácia/F1) misturam degradação de ranqueamento com inadequação do cutoff do DES no OOT — leia-os junto com os shifts de KS/Gini, que são independentes de corte.

---

## 2. Grupos homogêneos (ratings) e binning monotônico

Um *rating* é uma partição do score contínuo em poucos grupos homogêneos de risco, ordenados de forma que a taxa de evento (PD, ou o alvo médio em regressão) cresça de faixa em faixa. O `yggdrasil` aprende todos os cortes na amostra de **desenvolvimento (DES)** e os reaplica inalterados nas demais amostras (OOT, OOS), preservando a comparabilidade temporal. Esta seção descreve as quatro metodologias disponíveis em `yggdrasil.ratings`, o motor de fusão monotônica, as métricas WoE/IV e o relatório por grupo. Mantém-se a convenção de que o evento ("mau") é $y=1$ e o não-evento ("bom") é $y=0$.

### 2.1 Por que monotonicidade importa em crédito

Em risco de crédito o rating é um objeto de **decisão e governança**, não apenas de discriminação. Espera-se que a relação score → risco observado seja monotônica: se o grupo $g+1$ tem score pior que $g$, sua inadimplência observada não pode ser menor. Uma inversão (faixa de score "pior" com risco realizado "melhor") quebra a interpretação de ordenação, inviabiliza precificação por faixa, dificulta a definição de políticas de corte e costuma ser ruído amostral em vez de sinal. Por isso as metodologias do `yggdrasil` ou impõem a monotonicidade na construção dos cortes (OptBinning) ou a impõem *a posteriori* fundindo grupos com inversão não significativa.

Importante: a monotonicidade é **avaliada no OOT**, não no DES. Cortes ajustados no DES quase sempre são monotônicos *no próprio DES* por construção (overfit da partição); o teste relevante é se a ordenação sobrevive fora da amostra.

### 2.2 Decis (`DecileRating`) — metodologia de referência

Particiona o score em $n=10$ faixas de igual frequência, usando os decis empíricos do score no DES. Os cortes são os quantis

$$q_k = Q_{\text{score}}\!\left(\tfrac{k}{n}\right),\quad k=0,\dots,n,$$

com as bordas extremas substituídas por $\pm\infty$ (`np.quantile` sobre `np.linspace(0,1,n+1)` em `_fit_binner`; alocação por `np.searchsorted` em `_raw_groups`). Rótulos `R01..R10` deixam a ordem explícita.

**Característica central:** decis **não** aplicam fusão (`monotonic_fusion=False`). São a referência exigida — fáceis de auditar, frequência fixa por faixa — mas justamente por não fundir podem exibir inversões no OOT. Limiares iguais de quantil colapsam quando o score tem massa concentrada (muitos empates): `np.unique` remove bordas duplicadas e você pode acabar com menos de 10 grupos. Use os decis como diagnóstico de ordenação bruta; para a partição final de produção prefira as metodologias com monotonicidade garantida.

### 2.3 Quantis finos + fusão monotônica (`QuantileMonotonicRating`)

Corta o score em faixas finas (passo `step=0.05`, i.e. ~20 grupos por quantil no DES) e em seguida **funde** grupos adjacentes cuja inversão no OOT não seja estatisticamente significativa (`monotonic_fusion=True`, `alpha=0.05`). A granularidade inicial alta permite que a fusão decida onde estão as fronteiras reais de risco, em vez de impô-las por uma grade fixa. Rótulos em letras (`A, B, C, ...`). A lógica de fusão é a descrita a seguir (§2.4).

### 2.4 Fusão por inversão (`fundir_por_inversao`)

É o motor de monotonicidade compartilhado por `QuantileMonotonicRating` e `TreeRating`. Os grupos brutos chegam **já ordenados de forma crescente pelo score**; sua média de target deveria, portanto, ser crescente. O algoritmo varre os pares adjacentes $(g_{i-1}, g_i)$ **na amostra OOT**:

1. Se $\bar t^{\,\text{oot}}_{i} < \bar t^{\,\text{oot}}_{i-1}$ há uma **inversão**.
2. Testa-se se a diferença entre os dois grupos é significativa:
   - **Regressão** — Mann-Whitney U bilateral sobre os valores do target ($U$ não-paramétrico, robusto à forma da distribuição, adequado a alvos como LGD em $[0,1]$);
   - **Classificação** — qui-quadrado de independência sobre a tabela $2\times2$ de evento/não-evento (com correção de continuidade), $\chi^2=\sum (O-E)^2/E$.
3. Se $p > \alpha$ (inversão **não** significativa), os grupos são fundidos; se $p \le \alpha$, a inversão é considerada real e preservada.

O laço repete até não haver mais fusões (varredura iterativa, pois fundir um par pode criar/eliminar inversões com o vizinho). Ao final, os clusters resultantes recebem rótulos $A, B, C, \dots$ via `idx_para_letra`.

**Interpretação de $\alpha$:** $\alpha$ é a tolerância a inversões. $\alpha$ baixo → funde mais (exige forte evidência para *manter* uma inversão) → menos grupos, mais robusto, menor granularidade; $\alpha$ alto → preserva mais separações, risco de manter inversões espúrias.

**Armadilhas:**
- Se a amostra OOT estiver ausente (`oot_mask.sum()==0`), **não há fusão** — os grupos são apenas rotulados na ordem. A monotonicidade fica sem garantia; confira sempre se o OOT existe.
- Grupos vazios no OOT ou margens nulas na tabela $2\times2$ retornam $p=1.0$ por convenção (sem teste possível ⇒ funde). Faixas muito finas com pouco volume no OOT tendem a fundir agressivamente — não confunda com "feature fraca".
- O teste é feito **só nos pares adjacentes com inversão**; a fusão corrige a ordenação, não maximiza separação. Para máxima discriminação com monotonicidade, OptBinning é mais indicado.

### 2.5 Árvore (`TreeRating`)

Uma `DecisionTreeRegressor` regride o **target sobre o score** (mesmo em classificação: regredir o alvo $0/1$ estima a taxa de evento por folha). Cada folha é um grupo bruto; as folhas são ordenadas pela média do target **no DES** (gerando o rank crescente) e em seguida passam pela mesma `fundir_por_inversao` (`monotonic_fusion=True`). A árvore escolhe os cortes do score que mais reduzem a variância do alvo, com guarda-corpos contra folhas minúsculas: `min_samples_leaf = max(0.05·n, 50)` e `max_leaf_nodes=10`.

**Armadilhas:** a árvore tende a colocar cortes onde há mais dados, podendo produzir faixas de tamanho muito desigual (baixa representatividade em alguns grupos). É sensível a `random_state` e aos hiperparâmetros de folha mínima — fixe-os para reprodutibilidade. Como aprende cortes do alvo no DES, há risco de sobreajuste da partição; a fusão no OOT atenua, mas não elimina.

### 2.6 OptBinning (`OptBinningRating`)

Resolve um problema de **binning ótimo** que maximiza a separação (poder preditivo) **sujeito à restrição de monotonicidade da tendência** (`monotonic_trend="auto_asc_desc"`). Usa `OptimalBinning` (classificação) ou `ContinuousOptimalBinning` (regressão), com `max_n_bins=10` e pré-binning mínimo `min_prebin_size=0.02`. Como a monotonicidade já é imposta na otimização, **não** se aplica a fusão por inversão (`monotonic_fusion=False`); a alocação usa diretamente `ob.splits` (`np.searchsorted`).

É, em geral, a metodologia preferida para a partição final: combina cortes ótimos, número parcimonioso de grupos e monotonicidade garantida na própria solução. **Armadilha:** a monotonicidade é imposta *no DES*; sob forte drift do score, a ordenação pode degradar no OOT mesmo assim — valide com o relatório por grupo e com PSI/CSI (§3).

### 2.7 WoE e IV (binning de feature, `binning.py`)

Para features (e, por extensão, para diagnosticar bins), o `yggdrasil` computa **Weight of Evidence** e **Information Value** sobre a tabela de bins (`binning_table`, problema de classificação). Por bin $i$, com $\text{dist\_bom}_i$ e $\text{dist\_mau}_i$ as proporções de bons (alvo $y=0$) e maus (alvo $y=1$) que caem no bin:

$$\text{WoE}_i = \ln\!\frac{\text{dist\_bom}_i}{\text{dist\_mau}_i}, \qquad \text{IV} = \sum_i \big(\text{dist\_bom}_i - \text{dist\_mau}_i\big)\,\text{WoE}_i .$$

Implementação com suavização $\varepsilon=10^{-6}$ no numerador, denominador e nas distribuições, evitando $\ln 0$ e divisão por zero (`woe = np.log((dist_bom+eps)/(dist_mau+eps))`; `iv_parcial = (dist_bom-dist_mau)*woe`; IV total em `out.attrs['iv']`). O `FeatureBinner` trata o **missing como bin próprio** (`MISSING`) e agrupa níveis categóricos raros (`< rare_level_pct`) em `OUTROS`.

**Interpretação do IV** (faixas usadas no subpacote EDA, `iv_power`):

| IV | Poder |
|---|---|
| $<0{,}02$ | inútil |
| $0{,}02$–$0{,}1$ | fraco |
| $0{,}1$–$0{,}3$ | médio |
| $0{,}3$–$0{,}5$ | forte |
| $>0{,}5$ | suspeito de *leakage* |

**Armadilhas:** WoE é sensível a bins com pouquíssimos maus (instável apesar do $\varepsilon$); IV é sempre não-negativo (não indica direção) e cresce mecanicamente com o número de bins — compare IVs apenas sob o mesmo esquema de binning. IV muito alto raramente é "feature excelente": quase sempre é vazamento de informação do alvo. A suavização por $\varepsilon$ enviesa levemente o WoE de bins vazios para zero.

### 2.8 Relatório por grupo (`group_report` / `is_monotonic`)

Consolida a auditoria de um rating numa tabela ordenada pelo **score médio crescente** (`group_report`), com, por grupo: `volume` e `pct_volume` (**representatividade**); `score_medio` (média **prevista**); `target_medio` (média **observada**); faixa de score (`score_min`/`score_max`); e, por amostra, `vol_{s}`, `pct_{s}`, `score_medio_{s}`, `target_medio_{s}`.

A leitura central confronta **previsto vs. observado** por grupo:

- **calibração** — `score_medio` ≈ `target_medio` em cada faixa indica score bem calibrado (não só bem ordenado);
- **monotonicidade** — `is_monotonic` verifica $\forall i:\ \bar t_{i+1} \ge \bar t_i$ (todos os incrementos da média observada não-negativos) na ordem de score; aplicada por amostra, é o teste prático de que a ordenação sobrevive ao OOT;
- **representatividade** — grupos com `pct_volume` muito baixo são instáveis e pouco úteis para política; reavalie a granularidade (`step`, `max_leaf_nodes`, `max_n_bins`) ou aumente $\alpha$ para fundir.

**Armadilha:** monotonicidade no DES é quase garantida por construção e não prova nada — sempre avalie `is_monotonic` por amostra, priorizando o OOT. E monotonicidade não implica calibração: um rating pode ordenar perfeitamente e ainda ter `score_medio` sistematicamente distante de `target_medio` (viés de nível), o que só a comparação previsto-vs-observado revela.

---

## 3. Estabilidade e monitoramento no tempo (PSI/CSI)

Modelos de risco são treinados sobre uma fotografia (a base de **desenvolvimento**, DES) e aplicados sobre populações futuras (**out-of-time**, OOT, e safras de produção). A premissa central de qualquer escore é que a distribuição da população se mantém razoavelmente estável: se o perfil dos solicitantes muda (mudança de mix, sazonalidade, nova política de aquisição, quebra de pipeline de dados), o escore calibrado no passado deixa de ser confiável mesmo que a relação preditiva siga válida. Esta seção descreve como o `yggdrasil` quantifica esse deslocamento populacional via **PSI** (Population Stability Index) e o decompõe via **CSI** (Characteristic Stability Index), tanto para o escore quanto para cada feature, e como acompanha esses indicadores ao longo das safras.

### 3.1 PSI numérico

O PSI mede a divergência entre a distribuição de uma variável na população esperada (DES) e na população atual (OOT/safra), discretizando-a em $B$ faixas. É uma **divergência de Jeffreys** (KL simetrizada) aplicada a histogramas:

$$\text{PSI} = \sum_{i=1}^{B} \left( a_i - e_i \right) \cdot \ln\!\left( \frac{a_i}{e_i} \right)$$

onde $e_i$ é a proporção de observações da base esperada no bin $i$ e $a_i$ a proporção da base atual no mesmo bin. Cada termo penaliza simultaneamente a magnitude do desvio ($a_i - e_i$) e sua razão logarítmica $\ln(a_i/e_i)$, de modo que migrações em faixas extremas pesam mais que oscilações em faixas densas.

**Escolha dos cortes (no DES).** Os limites dos bins são definidos por **quantis da base esperada** (DES) — tipicamente decis — e **congelados**. A base atual é então alocada nesses mesmos cortes. Isso é deliberado: medir estabilidade exige uma régua fixa. Se os cortes fossem recalculados na base atual, cada faixa conteria, por construção, aproximadamente a mesma fração, e o PSI tenderia artificialmente a zero — mascarando exatamente o deslocamento que se quer detectar.

**Estabilizador $\varepsilon$.** Bins vazios geram $a_i = 0$ ou $e_i = 0$, e o $\ln$ diverge. O `yggdrasil` soma um piso $\varepsilon = 10^{-6}$ a cada proporção antes do log. É uma escolha pragmática, mas tem efeito colateral: bins genuinamente vazios em uma das bases inflam o termo via $\ln(a_i/\varepsilon)$. Por isso, **bins pouco populosos no DES tornam o PSI volátil** e devem ser consolidados.

No `yggdrasil`, `psi` (em `monitoring/psi.py`) implementa o caso numérico: corta o `expected` por quantis, aplica os mesmos cortes ao `actual` e acumula a soma com `eps=1e-6`.

### 3.2 PSI categórico

Para variáveis nominais não há quantis a calcular: cada **categoria é o seu próprio bin**. A fórmula é idêntica, com $i$ percorrendo as categorias observadas na união das duas bases:

$$\text{PSI}_{\text{cat}} = \sum_{c \,\in\, \mathcal{C}} \left( a_c - e_c \right) \cdot \ln\!\left( \frac{a_c}{e_c} \right)$$

A função `psi_categorical` cobre esse caso. **Armadilhas** específicas: categorias presentes no OOT mas ausentes no DES (níveis novos) produzem $e_c \approx \varepsilon$ e dominam a soma; e variáveis de alta cardinalidade geram muitos bins ralos, com PSI ruidoso. Categorias raras devem ser agrupadas (ex.: `OUTROS`) antes do cálculo — o mesmo princípio aplicado pelo `FeatureBinner` no tratamento de raros.

### 3.3 Faixas de interpretação (0,10 / 0,25)

A classificação segue os limiares de mercado, implementada em `classify_psi`:

| Faixa de PSI | Classificação | Leitura |
|---|---|---|
| $\text{PSI} < 0{,}10$ | **estável** | sem deslocamento material |
| $0{,}10 \le \text{PSI} \le 0{,}25$ | **atenção** | deslocamento moderado; investigar a causa |
| $\text{PSI} > 0{,}25$ | **instável** | deslocamento severo; revisar/recalibrar |

Esses cortes são convenções, não testes de hipótese: não dependem do tamanho amostral. Em amostras pequenas, o PSI é instável e pode ultrapassar $0{,}10$ por ruído; em amostras enormes, deslocamentos irrelevantes podem ficar abaixo de $0{,}10$. Interprete sempre o número junto da decomposição por bin (CSI) e do volume.

### 3.4 CSI por bin (decomposição do PSI)

O PSI é um escalar que diz **quanto** a distribuição se moveu, mas não **onde**. O CSI abre essa soma: a contribuição de cada bin é exatamente uma parcela do PSI.

$$\text{CSI}_i = \left( a_i - e_i \right) \cdot \ln\!\left( \frac{a_i}{e_i} \right), \qquad \text{PSI} = \sum_{i=1}^{B} \text{CSI}_i$$

Cada $\text{CSI}_i \ge 0$ identifica as faixas responsáveis pelo deslocamento. Um CSI concentrado em uma única faixa aponta migração localizada (ex.: empilhamento de clientes em um decil de escore, ou um valor sentinela que virou missing); um CSI espalhado uniformemente sugere mudança global de nível. No `yggdrasil`, `csi_by_bin` (em `stability/`) retorna a contribuição por bin no formato $(a_i - e_i)\ln(a_i/e_i)$, permitindo ranquear as faixas que mais empurram o PSI. **Armadilha:** como cada termo é não-negativo, sinais opostos (um bin esvazia, outro enche) não se cancelam — o CSI mostra os dois como contribuições positivas, e é isso que se quer ao auditar a causa.

### 3.5 PSI ao longo das safras

Estabilidade é uma propriedade temporal: um PSI baixo no OOT agregado pode esconder deterioração monotônica safra a safra. O `yggdrasil` calcula o PSI de cada safra **sempre contra a mesma referência DES**, com os **cortes congelados do DES**, de modo que os valores sejam comparáveis na série. As funções relevantes:

- `psi_score_over_time` / `psi_summary` (em `monitoring/psi.py`) — evolução do PSI do **escore** por período, com sumarização.
- `feature_psi` e `feature_psi_over_time` (em `stability/`) — o mesmo para **features** individuais.

Na leitura da série, distinga **tendência** (crescimento sustentado do PSI → drift estrutural, candidato a recalibração) de **picos isolados** (uma safra anômala → sazonalidade ou incidente de dados pontual). Plote contra as faixas de referência $0{,}10$ e $0{,}25$ (paleta steelblue para a série, marcações nos limiares) para leitura imediata.

### 3.6 PSI de escore vs. PSI de feature

Os dois respondem a perguntas diferentes e devem ser lidos em conjunto:

- **PSI do escore** mede o deslocamento da distribuição da saída do modelo. É o indicador de topo: resume o efeito líquido de todas as mudanças nas features sobre a predição final.
- **PSI de feature** (via `feature_psi`) mede o deslocamento de cada insumo isoladamente. É diagnóstico: localiza qual variável está dirigindo o movimento do escore.

A combinação é informativa. **PSI de escore alto com PSI de features baixo** é um sinal de alerta — frequentemente indica problema de pipeline (feature recalculada, encoding alterado) ou interação entre variáveis. **PSI de features alto com PSI de escore baixo** sugere que os deslocamentos se compensaram na saída, mas a robustez está corroída e pode quebrar a qualquer momento. PSI elevado em ambos aponta drift populacional genuíno. Note ainda que PSI mede **distribuição**, não **desempenho**: um escore pode permanecer estável (PSI baixo) e ainda assim perder poder discriminante — por isso o monitoramento de estabilidade complementa, mas não substitui, o acompanhamento de KS/Gini ao longo do tempo (§1).

### 3.7 Missing por safra e quebras

O percentual de missing de uma feature é um dos sinais mais sensíveis de incidente de dados: uma fonte que cai, um join que falha ou uma mudança de contrato no upstream costumam se manifestar primeiro como um salto abrupto na taxa de nulos — antes mesmo de o PSI acusar. O `yggdrasil` trata missing como categoria de pleno direito: no `FeatureBinner` o **missing vira um bin próprio**, de modo que sua migração entra naturalmente no PSI/CSI em vez de ser silenciosamente descartada.

O monitoramento direto fica em `missing_over_time` (em `profile/`), que acompanha a taxa de nulos por safra e **sinaliza quebra quando $|\Delta| > 0{,}20$** entre safras consecutivas, além de estimar a **tendência via `polyfit`** (inclinação da reta ajustada à série). A leitura: uma quebra ($|\Delta| > 0{,}20$) é tipicamente um evento (incidente de pipeline) e exige investigação de origem; uma tendência suave de alta indica erosão gradual de cobertura da fonte. **Armadilha:** quando o missing é preenchido por imputação ou por um valor sentinela (ex.: $-999$) antes do cálculo, a quebra não aparece na taxa de nulos — ela se desloca para o PSI/CSI da feature, geralmente como um CSI concentrado no bin do sentinela. Por isso, missing por safra e PSI/CSI devem ser auditados em conjunto.

---

## 4. Interpretabilidade (SHAP) e EDA de features

Esta seção documenta os métodos de **interpretabilidade do modelo** (valores de Shapley) e a **EDA de features** (perfil, relação com o alvo, importância, estabilidade e redundância), que culminam num **veredito automático** por feature. O subpacote `yggdrasil.eda` é um *entrypoint* independente do pipeline de modelo: ele diagnostica candidatas a feature **antes** da modelagem, ao passo que o SHAP (`yggdrasil.interpretability`) explica o modelo **já treinado**. Mantém-se a convenção de que o evento ("mau") é $y=1$ e o não-evento ("bom") é $y=0$.

### 4.1 Valores de Shapley (SHAP)

**O que mede.** A contribuição marginal de cada feature para a predição de **cada observação**, atribuída de forma justa segundo a teoria de jogos cooperativos. O valor de Shapley da feature $j$ é a média ponderada do ganho que ela traz a todas as coalizões $S$ de features:

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\,\big[\,f(S \cup \{j\}) - f(S)\,\big]$$

Os valores são **aditivos** (propriedade de eficiência): a predição decompõe-se como $\hat{f}(x) = \mathbb{E}[\hat{f}] + \sum_{j} \phi_j(x)$. Assim, $\phi_j(x)$ é uma explicação **local** (por linha) com unidade na escala da saída do modelo (log-odds ou probabilidade, conforme o explainer).

**Como o yggdrasil computa.** `compute_shap` (em `interpretability/shap_explain.py`) tenta três caminhos em cascata, do mais rápido ao mais geral:

1. `shap.TreeExplainer` — Shapley **exato e eficiente** para modelos de árvore (a escolha padrão da esteira). Em saída multi-classe seleciona a contribuição da **classe positiva** (`sv[:, :, -1]`);
2. `shap.Explainer` (API unificada, cobre lineares e outros);
3. `shap.KernelExplainer` (*fallback* agnóstico ao modelo, lento, com *background* reduzido a ≤100 amostras).

Para escala, amostra até `sample_size=2000` linhas (`_sample_X`). Todo o `shap_report` é **best-effort**: se o SHAP não suportar o modelo, devolve DataFrame vazio em vez de quebrar a esteira.

**Armadilhas.**

- `check_additivity=False` é usado por robustez — não confunda a tolerância numérica do TreeExplainer com erro de modelo.
- O KernelExplainer aproxima Shapley por amostragem e **assume independência entre features**; com features muito correlacionadas as atribuições ficam enviesadas (idealmente usar `TreeExplainer`).
- A escala importa: para TreeExplainer de classificador a contribuição costuma estar em **log-odds**, não em probabilidade — não some $\phi_j$ ingenuamente com taxas de evento.

### 4.2 Importância global por |SHAP| e beeswarm

**O que mede.** A **importância global** de cada feature é a média do valor absoluto de SHAP sobre as amostras:

$$\text{imp}_j = \frac{1}{n}\sum_{i=1}^{n} \big|\phi_j(x_i)\big|$$

`shap_feature_importance` calcula `np.abs(shap_values).mean(axis=0)` e ordena de forma decrescente. Diferentemente do `feature_importances_` (impureza) das árvores, esta métrica é **consistente** e reflete o efeito real sobre as predições, não a estrutura interna do modelo.

**Visualizações** (`save_shap_plots`):

- **beeswarm** (`shap.summary_plot`): cada ponto é uma observação; o **eixo x** é $\phi_j$ (direção e magnitude do efeito) e a **cor** é o valor da feature. Revela direção, dispersão e **não-linearidades/interações** que a média absoluta esconde;
- **barras** (`plot_type="bar"`): ranking de $\text{imp}_j$.

`shap_surrogate_importance` (em `eda/importance.py`) aplica o mesmo cálculo sobre um *surrogate* RandomForest quando não há modelo final — útil já na fase de EDA.

**Armadilhas.** A média de $|\phi_j|$ colapsa direção e perde **interações**: uma feature com efeito forte mas bidirecional pode parecer pequena. Sempre cruze o ranking de barras com o beeswarm. Importância alta **não** implica causalidade nem ausência de *leakage* (ver §4.7).

### 4.3 EDA — *Missing* global e temporal

**O que mede.** A prevalência de ausentes por feature e sua **estabilidade ao longo das safras** (mudanças de coleta/sistema geram quebras).

`missing_summary` (`profile.py`) reporta `pct_missing` global e por amostra (DES/OOT), além de `pct_zero` para numéricas, com *flag* por limiar: `descartar` se $\text{pct\_missing} \ge$ `missing_drop` (padrão $0{,}50$), `atencao` se $\ge$ `missing_warn`, senão `ok`.

`missing_over_time` agrega o *missing* por período (`time_freq`, mês por padrão) e detecta:

- **quebra** se a maior variação absoluta entre safras consecutivas excede o limiar: $\max_t |\Delta \text{pct}_t| > 0{,}20$;
- **tendência** via inclinação de regressão linear simples (`np.polyfit` de grau 1) sobre a série de `pct_missing`.

**Armadilhas.** *Missing* não é necessariamente ruim — pode ser informativo. Por isso o binning trata **missing como bin próprio** (§4.5), preservando seu sinal em vez de imputar cegamente. Uma quebra de *missing* entre DES e OOT é forte indício de instabilidade de processo, não apenas de qualidade.

### 4.4 Percentis e *drift* de distribuição

**O que mede.** A forma da distribuição (percentis e momentos) globalmente e sua **migração no tempo**.

- `percentile_table`: percentis configuráveis + `mean, std, min, max, skew, kurtosis, iqr`, por amostra;
- `percentiles_over_time`: grade de percentis por safra (alimenta o *fan chart* e o diagnóstico de *drift*);
- `outlier_summary`: dois detectores complementares — **IQR de Tukey** (outlier se $x < Q_1 - 1{,}5\,\text{IQR}$ ou $x > Q_3 + 1{,}5\,\text{IQR}$) e **z-score robusto via MAD**:

$$z_i = \frac{x_i - \text{med}(x)}{1{,}4826 \cdot \text{MAD}}, \qquad \text{outlier se } |z_i| > 3{,}5$$

O fator $1{,}4826$ torna o MAD um estimador consistente do desvio-padrão sob normalidade; o corte $3{,}5$ é a regra de Iglewicz–Hoaglin.

**Armadilhas.** O *drift* de percentis antecipa instabilidade que o PSI (§3) só confirmará depois. A curtose alta sinaliza caudas pesadas que distorcem médias e qualquer modelo linear — combine com o `pct_outlier_iqr`.

### 4.5 WoE/IV univariado

**O que mede.** O poder discriminante de uma feature **binada** numa tarefa binária. Para cada bin define-se o *Weight of Evidence* e agrega-se o *Information Value*:

$$\text{WoE}_b = \ln\!\left(\frac{\text{dist\_bons}_b}{\text{dist\_maus}_b}\right), \qquad \text{IV} = \sum_b (\text{dist\_bons}_b - \text{dist\_maus}_b)\cdot \text{WoE}_b$$

onde $\text{dist\_bons}_b$ e $\text{dist\_maus}_b$ são as frações de não-eventos ($y=0$) e eventos ($y=1$) no bin $b$ (com `eps=1e-6` para estabilizar logaritmos e divisões). É a mesma definição usada nos ratings (§2.7), aqui aplicada à seleção de features.

**Como o yggdrasil computa.** `FeatureBinner` (`binning.py`) ajusta os cortes **no DES** (`quantile`/`tree`/`optbinning`) e os reaplica em qualquer amostra — **missing vira bin próprio** e níveis categóricos raros (< `rare_level_pct`) viram `'OUTROS'`. `binning_table` produz a tabela por bin com `woe`, `iv_parcial` e o IV total em `.attrs['iv']`; `woe_iv_table` e `iv_score` expõem esse valor.

**Interpretação (`iv_power`).** Faixas de mercado:

| IV | Poder |
|---|---|
| $< 0{,}02$ | inútil |
| $0{,}02$–$0{,}1$ | fraco |
| $0{,}1$–$0{,}3$ | médio |
| $0{,}3$–$0{,}5$ | forte |
| $> 0{,}5$ | **suspeito de leakage** |

`event_rate_by_bin` adiciona **IC de Wilson** à taxa de evento por bin (mais confiável que o IC normal em bins pequenos ou taxas extremas), e `monotonicity_diagnostic` classifica a tendência (`crescente`/`decrescente`/`nao_monotonica`), conta inversões e mede o Spearman bin-índice × taxa.

**Armadilhas.** IV depende do binning: bins demais inflam o IV artificialmente. IV muito alto ($>0{,}5$) raramente é "feature ótima" — quase sempre é vazamento. WoE/IV só fazem sentido em **classificação binária** (degrada para `nan` em regressão).

### 4.6 Importância univariada vs. multivariada

**Univariada (model-free).** Trata a feature como score isolado:

- `univariate_ks_gini`: **KS** (máxima separação entre as CDFs de bons e maus), **AUC direção-agnóstica** $\text{AUC} = \max(\text{AUC}, 1-\text{AUC})$ e **Gini** $= 2\,\text{AUC} - 1$. A correção direção-agnóstica permite ranquear features sem conhecer o sinal esperado;
- `mutual_information`: $I(X;Y)$ via `mutual_info_classif`/`mutual_info_regression`, capturando dependência **não-linear** que correlação e KS perdem (categóricas/binárias marcadas como `discrete_features`).

**Multivariada (surrogate).** `model_importance` ajusta um **RandomForest** (200 árvores) no DES e reporta duas visões:

- `rf_importance` — importância por impureza (`feature_importances_`);
- `permutation` — `permutation_importance` (5 repetições): queda da performance ao **embaralhar** a feature, medindo o efeito **condicional às demais**.

A diferença univariada × multivariada é diagnóstica: uma feature com alto KS univariado mas baixa permutação é **redundante** (informação já presente em outras); o oposto sugere efeito que só emerge em conjunto.

**Ranking consolidado** (`importance_ranking`): o `score` final é a **média dos ranks** das métricas disponíveis (`iv, ks_univ, gini_univ, mutual_info, rf_importance, permutation`), tornando o ranking robusto a escalas heterogêneas.

**Armadilhas.** `rf_importance` por impureza é **enviesada** para features de alta cardinalidade/contínuas — por isso a permutação é o contraponto. O *surrogate* opera sobre matriz com categóricas *ordinal-encoded* e *missing* imputado pela mediana (`_build_matrix`), o que pode subestimar categóricas.

### 4.7 Detecção de *leakage*

**O que mede.** Features que "conhecem o futuro" (vazamento do alvo), denunciadas por poder preditivo **implausivelmente alto**.

`importance_ranking` marca `leakage_flag` quando:

$$\text{gini\_univ} > 2\cdot\text{leakage\_auc} - 1 \quad\text{(equiv. AUC} > 0{,}9\text{)} \qquad \textbf{ou} \qquad \text{IV} > \text{iv\_leakage}\;(0{,}5)$$

`leakage_suspects` lista as marcadas. Coerente com a faixa `suspeito_leakage` do `iv_power` (§4.5).

**Armadilhas.** É um *flag* de **suspeita**, não prova: variáveis legitimamente fortes (ex.: *score* de bureau) podem disparar. Sempre investigue a **origem temporal** da feature antes de descartar. Inversamente, *leakage* sutil (Gini moderado) escapa a esses cortes — não confie só neles.

### 4.8 Estabilidade (PSI/CSI por feature)

**O que mede.** O quanto a distribuição de uma feature migrou do DES para o OOT (e entre safras). O **PSI** de uma feature usa a mesma fórmula do monitoramento de score (§3.1):

$$\text{PSI} = \sum_b (\text{act}_b - \text{exp}_b)\cdot \ln\!\left(\frac{\text{act}_b}{\text{exp}_b}\right)$$

`feature_psi` (DES→OOT) e `feature_psi_over_time` (por safra contra a baseline) usam cortes por quantil para numéricas e `psi_categorical` (com *missing* como categoria) para categóricas. **Classificação** (`classify_psi`): estável $< 0{,}10$, atenção $0{,}10$–$0{,}25$, instável $> 0{,}25$.

O **CSI** decompõe o PSI **por bin** — cada parcela é a contribuição do bin $b$ ao PSI total:

$$\text{CSI}_b = (\text{act}_b - \text{exp}_b)\cdot \ln\!\left(\frac{\text{act}_b}{\text{exp}_b}\right), \qquad \text{PSI} = \sum_b \text{CSI}_b$$

`csi_by_bin` (cortes fixados no DES) localiza **qual faixa** migrou, identificando a causa do *drift*.

**Armadilhas.** PSI é sensível ao número de bins e a bins vazios (mitigado por `eps`). PSI alto isolado não diz **se** o *drift* prejudica o modelo — cruze com `bivariate_over_time` (IV por safra com cortes fixos), que mede se o **poder preditivo** caiu.

### 4.9 Correlação, VIF e redundância

**O que mede.** Multicolinearidade e features redundantes que inflam variância de coeficientes e prejudicam interpretabilidade.

- `correlation_matrix`: **Spearman** (robusto a não-linearidade monotônica e outliers), excluindo constantes;
- `cramers_v_matrix`: associação entre **categóricas**:

$$V = \sqrt{\frac{\chi^2 / n}{\min(r-1,\,k-1)}}$$

- `vif_table`: **VIF** por feature numérica, regredindo cada uma nas demais:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

*flag* `alto` se $\text{VIF} > $ `vif_high`; $\text{VIF}>10$ é a regra usual de multicolinearidade grave;

- `redundancy_clusters`: **cluster hierárquico** (ligação média) sobre a distância $d = 1 - |\text{corr}|$, cortando em $|\text{corr}| > 0{,}8$, para agrupar redundantes e sugerir um representante por cluster.

**Armadilhas.** Spearman captura apenas associação **monotônica** — pares com dependência não-monotônica passam despercebidos. VIF é definido só para numéricas; correlação indefinida (constantes) é tratada como **distância máxima** no clustering. Alto VIF não condena a feature: prejudica a inferência sobre coeficientes, não necessariamente a predição de modelos não-lineares.

### 4.10 Veredito automático por feature

A função `verdict` (`report.py`), consolidada em `build_feature_profile`, sintetiza todos os diagnósticos numa decisão acionável. A lógica é **hierárquica** — `descartar` tem prioridade sobre `revisar`:

**`descartar`** se qualquer condição crítica ocorrer:

$$\text{constante} \;\;\lor\;\; \text{pct\_missing} \ge 0{,}50 \;\;\lor\;\; \text{PSI}_{oot} > 0{,}25$$

**`revisar`** (sinais de cautela, mas recuperável) se:

$$\text{leakage} \;\lor\; \text{quase\_constante} \;\lor\; \text{IV} < 0{,}02 \;\lor\; \text{PSI}_{oot} > 0{,}10 \;\lor\; \text{quebra de missing}$$

**`manter`** caso contrário. Os limiares vêm de `EDAConfig` e `monitoring.psi` (`PSI_STABLE=0,10`, `PSI_SIGNIFICANT=0,25`).

**Lógica de negócio.** Descarta-se o que é **inviável** (constante, vazada de *missing*, instável demais para generalizar); revisa-se o que é **fraco ou arriscado** (sem poder — IV inútil; *leakage* suspeito; *drift* moderado; quebra de coleta) — exigindo decisão humana, não automação. O veredito é **conservador por construção**: prefere sinalizar `revisar` a deixar passar uma feature problemática.

**Armadilhas.** O veredito é um **filtro de triagem**, não substitui julgamento de domínio: uma feature `revisar` por IV baixo pode ser valiosa em interação (capturada só pela permutação/SHAP), e uma `manter` ainda pode ser redundante (cruze com §4.9). *Leakage* e PSI dependem da correta marcação de DES/OOT e dos *missing codes* (`apply_missing_codes`); contrato de amostras incorreto contamina todo o veredito.

Arquivos de referência: `src/yggdrasil/interpretability/shap_explain.py`; `src/yggdrasil/eda/{profile,binning,bivariate,importance,stability,correlation,report}.py`.

---

## 5. Referências

- **Resolução CMN nº 4.966/2021** — Conselho Monetário Nacional. Dispõe sobre os critérios contábeis para instrumentos financeiros e provisão para perdas esperadas associadas ao risco de crédito nas instituições financeiras (convergência local ao IFRS 9). Fundamenta a exigência de estimativas de PD/LGD/EAD e seu monitoramento.
- **IFRS 9 — Financial Instruments** (IASB, 2014). Modelo de perda de crédito esperada (*Expected Credit Loss*, ECL), com a decomposição $\text{ECL} = \text{PD}\times\text{LGD}\times\text{EAD}$ que motiva os modelos de classificação (PD) e regressão (LGD/EAD) desta esteira.
- **Basileia II/III — *International Convergence of Capital Measurement and Capital Standards*** (BCBS) e a abordagem **IRB** (*Internal Ratings-Based*). Fundamenta a construção de *ratings* internos, a exigência de monotonicidade da PD por faixa e a validação de discriminação e estabilidade.
- **Siddiqi, N.** *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring* (Wiley, 2006). Referência para **Weight of Evidence (WoE)** e **Information Value (IV)**, binning de scorecards e as faixas de interpretação de IV.
- **Kolmogorov, A. (1933); Smirnov, N. (1948).** Fundamentos do teste de **Kolmogorov–Smirnov**, base da estatística KS e do cutoff KS-ótimo ($\arg\max(\text{TPR}-\text{FPR})$).
- **Population Stability Index (PSI) / Characteristic Stability Index (CSI).** Práticas de mercado de monitoramento de *scorecards* (relacionadas à **divergência de Jeffreys** / KL simetrizada). Faixas de referência $0{,}10$ e $0{,}25$. Ver também Siddiqi (op. cit.).
- **Lundberg, S. M.; Lee, S.-I.** *A Unified Approach to Interpreting Model Predictions* (NeurIPS, 2017) e **Lundberg et al.** *From Local Explanations to Global Understanding with Explainable AI for Trees* (Nature Machine Intelligence, 2020). Fundamento dos **valores de SHAP** e do `TreeExplainer`.
- **Iglewicz, B.; Hoaglin, D. C.** *How to Detect and Handle Outliers* (ASQC, 1993). Origem da regra do **z-score robusto via MAD** com corte $3{,}5$.

---

> **Nota.** Este documento cobre a *metodologia* (o "porquê" dos métodos). Para exemplos de uso prático da API — execução do pipeline, EDA de features e leitura dos relatórios — consulte os tutoriais em [`notebooks/tutoriais/`](../notebooks/tutoriais/), em especial `02_tutorial_eda_features.ipynb`.