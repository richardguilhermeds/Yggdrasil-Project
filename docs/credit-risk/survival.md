# Análise de sobrevivência para a PD lifetime: `yggdrasil.credit_risk.survival`

`yggdrasil.credit_risk.survival` é a **bancada de trabalho** em cima dos motores de
estrutura a termo de `yggdrasil.credit_risk.ecl`: as ferramentas que a análise de
sobrevivência usa para construir, escolher e defender uma curva de PD *lifetime*, e a
interface interativa `SurvivalUI` que conduz o estudo aba a aba.

```python
from yggdrasil.credit_risk.survival import SurvivalUI
ui = SurvivalUI(df, origin_col="safra_origem", segment_col="produto", term_col="prazo",
                features=["feat_score", "feat_ltv"], horizon=60)
ui                                      # Painel · Kaplan-Meier · Hazard · Paramétrico · Calibração & Ciclo · Validação · Exportar

# o mesmo estudo, declarativo e reproduzível
from yggdrasil.credit_risk.survival import SurvivalConfig, run_survival_study
cfg = SurvivalConfig(origin_col="safra_origem", segment_col="produto", by="produto",
                     method="km", tail="parametric", distribution="weibull",
                     calibrate={"cartao": 0.078, "consignado": 0.021},
                     split="origin", split_value="2023-01-01")
res = run_survival_study(df, cfg)
res.model.apply(carteira, age_col="idade", term_col="prazo")     # o LifetimePD que o ecl_table consome
```

O `ecl` responde *como* estimar a curva (safra, KM, hazard discreto, Markov) e *como* usá-la
(calibração, ciclo, `apply`, `ecl_table`). Este subpacote responde às perguntas que ficam entre
as duas: **as curvas dos segmentos diferem?** **qual é a forma da maturação?** **o que a curva faz
além da última idade observada?** **ela acerta o nível, ordena o risco e a hipótese do modelo
sobrevive aos dados?**

---

## 1. O que a análise de sobrevivência acrescenta

Uma curva de PD *lifetime* é uma função de sobrevivência com o *default* como evento. Três
fatos dessa literatura mudam a prática:

1. **Censura à direita é a regra, não a exceção.** Um contrato originado há 6 meses não é um
   contrato que sobreviveu 60 meses; ele só não foi observado até lá. A base em risco é
   recontada **idade a idade** (`ContractPanel.at_risk`), e a tabela separa `n_default` de
   `n_censurado`. A censura por quitação antecipada costuma ser **informativa** (quem quita
   antes tende a ser menos arriscado); o painel de referência reproduz isso de propósito.
2. **A incerteza da curva é estimável.** O erro padrão de Greenwood e o IC log-log do
   Kaplan-Meier são o que separa "a PD de 36 meses é 18%" de "a PD de 36 meses está entre 15% e
   22%, com 44 contratos em risco na idade 36". A validação usa exatamente essa banda.
3. **Formato e nível são coisas distintas.** O modelo transversal (scorecard) dá o nível da
   PD de 12 meses; a análise de sobrevivência dá o formato da maturação e a cauda. Misturar os
   dois sem separar é a fonte mais comum de curva indefensável.

---

## 2. Tabela de vida, Nelson-Aalen e log-rank (`lifetable`)

**`life_table(panel, by=None)`**: uma linha por idade (e por grupo): `n_em_risco`, `n_default`,
`n_censurado`, `hazard`, `sobrevivencia` (KM) com `se_greenwood` e IC log-log, `pd_acumulada`
com IC, `hazard_acumulado` (Nelson-Aalen, $\hat H(t) = \sum_{k \le t} d_k / n_k$) e a
sobrevivência que ele implica, $e^{-\hat H(t)}$. Em tempo discreto $e^{-\hat H} \ge \hat S_{KM}$
sempre; as duas se afastam onde a base rareia.

**`logrank_test(panel, by, weights="logrank")`**: o teste de Mantel-Cox para $k$ grupos. Em cada
idade $t$ com quebras, o esperado no grupo $g$ sob $H_0$ é $e_{gt} = n_{gt}\,d_t / n_t$; a
estatística $U^\top V^{-1} U$ (com $U_g = \sum_t w_t (d_{gt} - e_{gt})$ e a covariância
hipergeométrica) é $\chi^2_{k-1}$. `weights="wilcoxon"` (Gehan-Breslow, $w_t = n_t$) pesa as
idades baixas. A tabela `grupos` traz a razão `observado/esperado`, a leitura de risco
relativo de cada grupo. `pairwise_logrank` faz o par a par com Bonferroni: quando o teste
global rejeita, diz **quais** pares diferem, o argumento para fundir segmentos.

**`median_survival`** (período em que $S(t)$ cruza 50%) e **`restricted_mean_survival`**
($E[\min(T,\tau)] = \sum_{t=0}^{\tau-1} S(t)$) leem a curva em unidades de tempo.

### Armadilha

O log-rank tem mais poder quando as curvas **não se cruzam** (riscos proporcionais). Curvas
que se cruzam (um produto com risco alto no início e baixo depois) podem passar no teste global
e ainda assim pedir curvas separadas: olhe o gráfico antes de fundir.

---

## 3. Famílias paramétricas e a cauda (`parametric`)

A curva empírica só vai até onde a carteira foi observada. Um contrato de 60 meses numa
carteira com 30 meses de histórico precisa de **30 meses que os dados não mostram**. A extensão
plana (`PDCurve.extend`, repetir o último *hazard*) é a hipótese mínima; a alternativa
defensável é ajustar uma família ao trecho observado e deixá-la dizer o que a maturação faz
depois.

`ParametricSurvival(distribution).fit(panel)` ajusta por máxima verossimilhança em tempo
discreto: cada idade contribui com $d_t$ quebras em $n_t$ em risco, e o *hazard* discreto da
família é $h_t = 1 - S(t+1)/S(t)$. A censura à direita e a truncagem à esquerda (contratos que
entram no painel já com idade) entram pela própria construção, sem tratamento especial.

| família | $S(x)$ | forma do hazard | leitura |
|---|---|---|---|
| exponencial | $e^{-\lambda x}$ | constante | referência: sem maturação |
| **Weibull** | $e^{-(x/\lambda)^k}$ | monótono: $k>1$ cresce, $k<1$ decresce | maturação de originação; seleção/*burn-out* |
| log-normal | $1 - \Phi\!\left(\frac{\ln x - \mu}{\sigma}\right)$ | corcova | pico de risco e queda depois |
| log-logística | $\frac{1}{1 + (x/\alpha)^\beta}$ | corcova ($\beta > 1$) ou decrescente | crédito ao consumidor típico |
| Gompertz | $\exp\!\left(-\frac{b}{c}(e^{cx} - 1)\right)$ | exponencial na idade | crescimento acelerado |

`fit_parametric(panel)` ajusta todas e devolve o ranking por **AIC** com `delta_aic`; abaixo de 2
as famílias são indistinguíveis pelos dados e a escolha cai na forma mais plausível para o
produto. `shape_reading()` traduz os parâmetros em texto ("hazard crescente com a idade,
k = 1,33: maturação").

**`splice_curves(empirical, tail, junction, horizon, match_level=True)`**: até a **junção** (a
última idade cuja base em risco ainda é maior ou igual a `min_at_risk`, `junction_age`) manda a
curva empírica; dali em diante, a família. *Casar o nível* multiplica a cauda pela razão entre a
média empírica e a paramétrica nas últimas idades antes da junção: sem degrau, e o **formato**
continua o da família. O `meta` da curva registra `junction`, `cauda` e `fator_nivel` para a
documentação.

### Armadilha

Uma família ajustada a 30 meses de histórico extrapola 30 meses de maturação. Weibull com
$k = 1{,}35$ segue subindo; log-logística com $\beta > 1$ faz a corcova e desce. Ambas podem
ter AIC parecido no trecho observado e divergir muito na cauda. Documente a escolha e compare
a PD *lifetime* das duas: a diferença é a **incerteza de modelo** da cauda, e a validação vai
perguntar por ela.

---

## 4. Validação (`validation`)

Sempre num painel **fora do tempo** (`split_panel`): por **safra de originação** (contratos
que o ajuste nunca viu, com toda a sua maturação: a validação mais exigente), por **data de
observação** (mesmos contratos, janela posterior: testa a curva a partir das idades em que a
carteira viva está) ou por **coluna de amostra**. Sem partição a validação roda no DES e a
interface avisa que é *in-sample*.

Os contratos são avaliados a partir da **primeira observação** no painel de validação: quem
entra com 10 meses tem a PD prevista para os próximos $h$ períodos a partir da idade 10 (a
mesma convenção de `LifetimePD.apply`). Contratos censurados antes de $h$ sem *default* saem do
denominador da discriminação e do decil; o backtest por Kaplan-Meier, esse sim, os aproveita
até onde foram observados.

| função | pergunta | estatística |
|---|---|---|
| `backtest_curve` | a PD acumulada prevista em 12/24/36 bate com a observada? | $z = (\text{prevista} - \hat F_{KM}) / \text{se}_{Greenwood}$ por horizonte e grupo; `dentro_do_ic` |
| `discrimination_by_horizon` | a curva ordena o risco no horizonte $h$? | AUC, Gini, KS do *default* em $h$ contra a PD acumulada prevista |
| `concordance_index` / `model_concordance` | e ordena o **tempo** até o evento, respeitando a censura? | C-index de Harrell, $O(n \log n)$ |
| `calibration_by_decile` | prevista × observada por faixa de PD prevista | IC binomial por faixa e Hosmer-Lemeshow ($gl = \text{faixas} - 2$) |
| `ph_test` | o efeito das covariáveis é o mesmo em todas as idades? | razão de verossimilhança contra o modelo com interações `feature × ln(1 + idade)`, $\chi^2_{p}$ |

Leitura do backtest: $\lvert z \rvert > 1{,}96$ em todos os horizontes **com o mesmo sinal** é
erro de **nível** (recalibre); sinal **trocando** entre horizontes é erro de **formato** (revise a
cauda ou o *baseline*). Uma curva única (sem grupos nem covariáveis) dá a mesma PD a todos e não
ordena: a discriminação sai `NaN` com a observação "nada a ordenar", e isso é esperado: o
ordenamento vem do scorecard.

---

## 5. O estudo declarativo (`study`)

`SurvivalConfig` é a configuração completa (colunas do painel, motor, grupo, horizonte,
features e *baseline*, família e junção da cauda, alvos de calibração, ciclo, partição e
horizontes de validação), serializável em JSON. `run_survival_study(df_ou_painel, cfg)` encadeia:

1. **partição** DES/OOT;
2. **tabela de vida** e log-rank (quando há grupo);
3. **curva** no DES: `km`/`vintage` (por grupo), `hazard` (um modelo por grupo, com covariáveis)
   ou `parametric` (família pura);
4. **cauda** (para `km`/`vintage`): emenda paramétrica ou plana;
5. **ajustes**: `calibrate_to` (nível) e `condition` (ciclo), nessa ordem;
6. **validação** no OOT (ou no DES, sinalizado).

O `SurvivalResult` carrega o `LifetimePD` final (o objeto que `ecl_table` consome), o modelo
antes dos ajustes, as tabelas de cada etapa e o tempo de cada uma. É o mesmo desenho de
`StudyConfig`/`run_study` do subpacote econométrico: a interface e o pipeline falam a mesma
língua, e a configuração que gerou a curva viaja com ela.

---

## 6. A interface (`SurvivalUI`)

Sete abas, no desenho das demais UIs do `credit_risk` (cartões, console, tema claro/escuro, JSON
de configuração, MLflow):

| aba | o que faz |
|---|---|
| **Painel** | mapeamento das colunas (ou o painel de referência sintético), mosaicos, base em risco / censura / hazard por idade, mapa safra × idade, **partição DES/OOT**, **Rodar estudo completo** |
| **Kaplan-Meier** | curvas por grupo nas quatro representações com banda de Greenwood e Nelson-Aalen, tabela de vida, mediana e RMST, **log-rank** global e par a par; *Adotar* (cauda plana) |
| **Hazard** | regressão em tempo discreto com covariáveis: *baseline*, *link*, coeficientes e odds ratio, contrato médio × KM, perfis P10/P50/P90, **riscos proporcionais**; *Adotar* |
| **Paramétrico** | as cinco famílias por grupo com ranking por AIC, sobrepostas ao KM (hazard e acumulada com IC), a **emenda** com junção e fator de nível; *Adotar* (emenda ou pura) |
| **Calibração & Ciclo** | alvo de PD de 12 meses por grupo (`calibrate_to`), $z$/$\rho$/reversão/modo (`condition`), antes × depois |
| **Validação** | os quatro blocos num placar com leitura em texto, tabelas e gráficos (backtest, decil) |
| **Exportar** | o `LifetimePD` em JSON (salvar/carregar), a `SurvivalConfig` em JSON (ver/salvar/carregar/aplicar), tabelas em CSV, MLflow (`log_lifetime_pd`) |

Qualquer aba **adota** a curva que produziu como a curva do estudo; Calibração, Validação e
Exportar operam sobre ela. Mexer na calibração ou no ciclo invalida a validação anterior; trocar
a partição derruba a curva adotada (ela foi ajustada em outro DES). `ui.to_config()` e
`ui.from_config()` fazem a ida e volta com a `SurvivalConfig`; `ui.apply(carteira, ...)` cola a
curva vigente na carteira.

Sem dados, `SurvivalUI()` e o botão **Carregar painel de referência** geram uma carteira
sintética de processo gerador conhecido (`make_reference_panel`: maturação Weibull, três
covariáveis, dois produtos, censura informativa) para aprender o fluxo. Os testes de recuperação
de parâmetros da suíte usam o mesmo painel.

---

## 7. Sob a Resolução CMN 4.966/2021 e o IFRS 9

- **Censura tratada** é requisito, não refinamento: contar contratos jovens como sobreviventes
  subestima a PD *lifetime* e a provisão de estágio 2.
- **Cauda documentada.** A extrapolação além do histórico observado é uma **hipótese de
  modelo**; a família escolhida, a junção e a comparação com a alternativa (plana, outra família)
  ficam no `meta` da curva e no JSON da configuração.
- **Nível × ciclo.** Calibrar ao scorecard e depois condicionar ao ciclo é defensável quando o
  scorecard é TTC. Se a PD de 12 meses já é PIT, condicionar de novo conta o ciclo duas vezes.
- **Validação fora do tempo.** O backtest in-sample de uma curva empírica é trivialmente perfeito;
  a interface o sinaliza. A partição por safra de originação é a que a validação independente
  espera.
- **SICR não está aqui** de propósito: `PDCurve.forward(t0, t1)` dá o insumo quantitativo; a regra
  de transferência é política da instituição.

---

## Referências

- Kaplan, E. L. & Meier, P. (1958). *Nonparametric estimation from incomplete observations*. JASA.
- Greenwood, M. (1926). *The natural duration of cancer*. Reports on Public Health and Medical Subjects.
- Nelson, W. (1972). *Theory and applications of hazard plotting for censored failure data*. Technometrics; Aalen, O. (1978). *Nonparametric inference for a family of counting processes*. Annals of Statistics.
- Mantel, N. (1966). *Evaluation of survival data and two new rank order statistics arising in its consideration*. Cancer Chemotherapy Reports (log-rank); Gehan, E. A. (1965), Breslow, N. (1970) (Wilcoxon generalizado).
- Harrell, F. E. et al. (1982). *Evaluating the yield of medical tests*. JAMA (C-index).
- Singer, J. D. & Willett, J. B. (1993). *It's about time: using discrete-time survival analysis to study duration and the timing of events*. Journal of Educational Statistics (hazard em tempo discreto).
- Grambsch, P. M. & Therneau, T. M. (1994). *Proportional hazards tests and diagnostics based on weighted residuals*. Biometrika (riscos proporcionais; aqui via LR com interações na idade).
- Botha, A. et al. (2025). *Approaches for modelling the term-structure of default risk under IFRS 9: a tutorial using discrete-time survival analysis*. [arXiv:2507.15441](https://arxiv.org/abs/2507.15441).
- Bellotti, T. & Crook, J. (2013). *Forecasting and stress testing credit card default using dynamic models*. International Journal of Forecasting.
- Resolução CMN 4.966/2021 e IFRS 9: perda esperada, estágios, PD *lifetime*.

## Tutoriais

| # | notebook | o que cobre |
|---|---|---|
| 13 | [PD lifetime](../../notebooks/tutoriais/13_tutorial_pd_lifetime.ipynb) | os motores: censura, as 4 representações, os 5 métodos, calibração e ciclo |
| 16 | [Interface de análise de sobrevivência](../../notebooks/tutoriais/16_tutorial_interface_sobrevivencia.ipynb) | as 7 abas da `SurvivalUI`: log-rank, hazard com covariáveis, cauda paramétrica, validação OOT e o estudo declarativo |
