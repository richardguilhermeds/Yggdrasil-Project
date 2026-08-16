# PD lifetime, ELBE e CCF — `yggdrasil.credit_risk.ecl`

`yggdrasil.credit_risk.ecl` traz os **parâmetros de risco da perda esperada** e a
conta que os junta na provisão de IFRS 9 / Resolução CMN 4.966/2021.

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

É o eixo que faltava no pacote. `tree`/`model` **ordenam** o risco entre clientes
(transversal, 12 meses); `econometric` **desloca o nível** conforme o ciclo
(temporal); `capital` mede a perda **inesperada**. Este subpacote responde à outra
metade: *quando* a perda esperada acontece ao longo da vida do contrato, *quanto*
sobra a perder no que já quebrou, e *sobre qual exposição* ela incide.

---

## 1. A curva de PD — quatro nomes para a mesma coisa

A maior fonte de erro em PD *lifetime* não é o motor de estimação: é o
vocabulário. A mesma curva é pedida de quatro formas, e cada área usa uma.

| representação | responde |
|---|---|
| **condicional** (*hazard*) | "sobreviveu até `t−1`; qual a chance de quebrar em `t`?" |
| **marginal** | "qual a chance de quebrar **exatamente** em `t`?" (vista de hoje) |
| **acumulada** | "qual a chance de ter quebrado **até** `t`?" |
| **sobrevivência** | "qual a chance de **não** ter quebrado até `t`?" |

As quatro carregam a mesma informação e se convertem por identidades exatas:

$$S(t)=\prod_{k\le t}(1-h_k) \qquad F(t)=1-S(t) \qquad m(t)=S(t-1)\,h(t) \qquad h(t)=\frac{m(t)}{S(t-1)}$$

`PDCurve` guarda o *hazard* — a representação canônica, a única que não depende
do ponto de partida — e entrega as outras três. Aceita ser construída a partir de
qualquer uma (`from_hazard`, `from_marginal`, `from_cumulative`,
`from_survival`), o que dispensa quem chama de fazer a conversão na mão: é ali
que nascem os erros de um período de defasagem.

**O ECL soma a marginal.** Usar a condicional na soma conta o mesmo contrato
várias vezes; usar a acumulada conta a mesma perda repetidamente. Só a marginal
particiona a probabilidade entre os períodos.

`forward(t0, t1) = 1 − S(t1)/S(t0)` é a PD condicional entre horizontes — o
insumo quantitativo natural do SICR (comparar a PD *lifetime* remanescente de
hoje com a de quando o contrato foi originado).

---

## 2. Os cinco motores de PD lifetime

`LifetimePD(method=...)` — uma classe, o comportamento escolhido por argumento,
como o `task_type` dos segmentadores. Todos devolvem a mesma `PDCurve`.

| `method` | o que faz | quando usar |
|---|---|---|
| `"constant"` | *hazard* constante a partir da PD de 12 meses: `h = 1 − (1 − PD₁₂)^{1/12}` | linha de base; quando não há painel histórico |
| `"vintage"` | taxa marginal observada **por idade**, com a base em risco recontada período a período | há painel e a maturação importa |
| `"km"` | Kaplan-Meier — mesmo ponto estimado da safra, mais Greenwood e IC log-log | quando a **incerteza** da curva vai para a documentação |
| `"hazard"` | regressão de *hazard* em tempo discreto sobre pessoa-período | curva **por contrato**, com covariáveis |
| `"markov"` | cadeia de Markov sobre a matriz de transição, por Chapman-Kolmogorov | há régua de **rating** funcionando |
| `"survival"` | apelido: resolve em `"hazard"` com `features`, em `"km"` sem | quando tanto faz |

### Censura é o ponto delicado

Um contrato originado há 6 meses não é um contrato que sobreviveu 60 meses; ele
apenas ainda não foi observado até lá. Somar os dois na mesma taxa subestima o
risco das idades altas. Por isso a base em risco (`ContractPanel.at_risk`) é
recontada **idade a idade**, e a tabela separa `n_default` de `n_censurado`.

### *Default* é absorvente

Depois da primeira quebra o contrato sai do conjunto em risco. Observações
posteriores são descartadas por padrão (`drop_post_default=True`) — mantê-las
contaria a mesma quebra várias vezes e achataria a curva.

### Hazard em tempo discreto

Expande o painel em pessoa-período e ajusta um modelo binário para "quebrou neste
período, dado que chegou vivo nele". A idade entra como *baseline* (`dummies`,
`spline`, `linear` ou `log`) e as covariáveis deslocam essa linha. Duas virtudes
práticas: a curva passa a ser **por contrato** (o que o ECL por contrato exige) e
covariáveis que mudam no tempo entram naturalmente.

O *link* padrão é o **logit** (`scikit-learn`, núcleo do pacote). O `cloglog` — o
análogo em tempo discreto do modelo de Cox — está disponível via `statsmodels`,
carregado sob demanda.

### Markov reusa o motor de capital

A estimação da matriz (`estimate_transition_matrix`, coorte ou duração) e o
condicionamento ao ciclo (`zshift_transition_matrix`) já existiam em
`yggdrasil.credit_risk.capital.migration` e são chamados daqui. Uma única fórmula
de migração no repositório: a mesma matriz que alimenta a simulação de capital
alimenta a curva de PD do ECL.

---

## 3. As duas pontes: nível e ciclo

### `calibrate_to` — o nível vem do modelo transversal

A curva empírica dá a **forma** da maturação; o scorecard dá o **nível** por
cliente. `calibrate_to(pd_12m=...)` resolve o deslocamento `δ` no logit do
*hazard* tal que a PD acumulada iguale o alvo, preservando a ordem e as razões de
chance entre horizontes. É a prática de mercado, e deixa a linhagem em
`meta["calibracao"]`.

### `condition` — o ciclo desloca a curva

`condition(z, rho)` leva a curva de TTC a PIT pelo arcabouço de Vasicek, com a
convenção de sinal do repositório: **`z > 0` = ciclo benigno**.

| `mode` | fórmula | propriedade |
|---|---|---|
| `"shift"` (padrão) | `Φ(Φ⁻¹(PD) − √ρ·z)` | **idempotente em `z = 0`** |
| `"conditional"` | `Φ((Φ⁻¹(PD) − √ρ·z)/√(1−ρ))` | lei condicional exata do fator único |

O padrão é o **deslocamento puro do limiar** — a mesma escolha (e a mesma
justificativa) de `zshift_transition_matrix`, o que faz o caminho da curva
concordar com o caminho da matriz de migração. A lei condicional exata fecha com
`capital.asrf.conditional_pd` (com `z = −Φ⁻¹(q)`) e com a inversa
`econometric.transforms.vasicek_z` — use-a quando o `z` vier de lá, porque só
assim a ida e a volta fecham. **Trocá-las por engano muda o nível da provisão.**

`z` aceita escalar, uma trajetória por horizonte (que é onde entra a projeção de
um modelo satélite) ou um `z` inicial com `decay` — a **reversão à média** que se
usa num horizonte *lifetime*, onde o choque de hoje não vale para sempre.

---

## 4. ELBE — a perda dos contratos já em *default*

Para o contrato que não quebrou, a perda esperada é `PD × LGD × EAD`. Para o que
**já quebrou**, não há PD. O que resta é: *do saldo que ainda está lá, quanto
ainda vira perda?*

A construção parte de duas coisas — a **exposição inicial** (o EAD na data do
*default*) e as **colunas de LGD por mês em *default*** (`lgd_m0`, `lgd_m1`, …,
onde `lgd(t) = 1 − r(t)`, com `r(t)` a recuperação acumulada como fração da
exposição inicial).

| grandeza | fórmula |
|---|---|
| recuperação acumulada | `r̄(t)`, encadeada e ponderada por exposição |
| exposição remanescente | `Σ EAD₀·(1 − r(t))` |
| LGD do ciclo completo | `LGD = 1 − r̄(T*)`, no horizonte de *workout* `T*` |
| **ELBE(t)** | `(1 − r̄(T*)) / (1 − r̄(t))` |
| LGD *in default* | `ELBE(t) + add-on` de perda inesperada |

A ELBE é a perda que resta **sobre o saldo remanescente**, e não sobre a exposição
original. Por isso ela é uma razão, e por isso **sobe** com o tempo em *default*
mesmo quando a recuperação total é boa: o denominador encolhe mais rápido que o
numerador. Um erro comum é reportar `1 − r̄(t)` como se fosse a ELBE — isso é a
LGD acumulada até `t`, grandeza retrospectiva, não a melhor estimativa do que
ainda falta.

### Três cuidados que o módulo trata

**Coorte variável.** Como os *defaults* recentes ainda não chegaram aos meses
altos, o conjunto observado muda de mês para mês. A média direta da recuperação
acumulada mistura coortes e produz uma curva que pode até **cair**. O tratamento
correto é encadear a recuperação **marginal** — o análogo do produto-limite:

$$\Delta \bar r(t) = \frac{\sum_{i \in obs(t)} EAD_{0i}\,(r_i(t) - r_i(t-1))}{\sum_{i \in obs(t)} EAD_{0i}} \qquad \bar r(t) = \sum_{k \le t} \Delta \bar r(k)$$

Cada mês é estimado nos contratos que **chegaram** naquele mês, e a curva fica
monotônica por construção. `cohort="complete"` oferece a alternativa da coorte
fechada — mais simples de auditar, mas só enxerga o passado distante.

**Horizonte de *workout*.** `workout_horizon` localiza o menor `t` tal que
`r̄(T_max) − r̄(t) ≤ tol` — a leitura operacional do critério do *ECB Guide to
Internal Models* (§6.4): o momento a partir do qual a evolução da recuperação
acumulada é praticamente nula.

Com `ultimate="workout"` (padrão), a **ELBE do próprio `T*` vale 1 por
construção**: se a recuperação do ciclo completo é a observada até `T*`, quem
chegou lá sem recuperar não recupera mais. É o comportamento correto, e é o que
faz a curva de ELBE subir do nível da LGD até 1 ao longo da cobrança.

**Desconto.** Sob a 4.966 e o IFRS 9, as recuperações futuras entram a valor
presente. `discount_rate` desconta cada recuperação **marginal** antes de
reacumular — a acumulada bruta não se desconta em bloco.

---

## 5. CCF / EAD — a conversão do limite não sacado

Em produto rotativo a exposição não é conhecida: o cliente decide quanto saca, e
quem está indo para o *default* costuma sacar mais.

$$EAD = sacado_{ref} + CCF \cdot (limite_{ref} - sacado_{ref})$$

### O desenho da base é o que mais move o resultado

| `method` | data de referência | tempo até o *default* |
|---|---|---|
| `"cohort"` | grade de calendário espaçada de `horizon` | varia de 0 a `horizon − 1` |
| `"fixed_horizon"` | exatamente `horizon` períodos antes | sempre `horizon` |
| `"variable"` | **todas** as datas em `[default − horizon, default)` | 1 a `horizon` |

Quanto mais cedo a referência, mais tempo o cliente teve para sacar, e **maior o
CCF** — é por isso que o desenho tem de ser documentado junto do número.

> A **coorte generalizada** de Gürtler, Hibbeln & Usselmann (2018) — coortes
> sobrepostas começando em todo período, em vez da grade fixa de calendário —
> produz exatamente o mesmo conjunto de pares (contrato, data de referência) que
> `"variable"` quando a amostra é a dos inadimplentes. A generalização daquele
> artigo está no conjunto em risco dos **adimplentes**, que não entra na
> estimação do CCF. Por isso o pacote expõe três desenhos, e não quatro.

### As quatro medidas ex-post

| medida | definição | reconstrução do EAD |
|---|---|---|
| `ccf` (LEQ) | `(EAD − sacado) / (limite − sacado)` | `sacado + CCF·(limite − sacado)` |
| `eadf` | `EAD / limite` | `EADF · limite` |
| `auf` | `(EAD − sacado) / limite` | `sacado + AUF · limite` |
| `ead` | o próprio EAD | `EAD` |

O `ccf` tem o denominador mais instável — com o cliente quase no limite,
`limite − sacado` tende a zero e o fator explode. `eadf` e `auf` trocam esse
denominador pelo limite, que é estável. A evidência empírica ainda assim favorece
o `ccf` para acurácia; `compare_measures` roda as quatro no mesmo *backtest* de
EAD para a validação decidir na carteira concreta.

### A distribuição é bimodal

O CCF observado se concentra em **0** (o cliente não mexeu no limite) e em **1**
(sacou tudo), com massa achatada no meio. `distribution()` separa os dois pontos
das faixas internas — é o que justifica, na documentação, a escolha entre média
agrupada e um modelo de resposta fracionária.

### Higiene, agrupamento e uso

`reference_dataset` filtra limite nulo, não sacado nulo e *over limit*, winsoriza
e recorta em `[0, 1]` — e **conta** cada exclusão em `excluded_frame()`.
`pooled_ccf` agrupa (média, mediana ou ponderada pelo denominador natural da
medida), `backtest_ead` compara EAD previsto e realizado em **moeda** (um CCF com
erro médio zero por contrato pode subestimar a carteira se errar nos contratos
grandes), `ccf_psi` monitora a distribuição entre safras e `ccf_downturn` é
reexportado de `capital.parameters` — fonte única no repositório.

O ajuste **preditivo** (CCF em função de características) fica com o
`ModelSegmenter(task_type="regression")`: a base que sai de `reference_dataset` já
está no formato que ele consome.

> **Onde concentrar a validação.** Os presets de produto do motor de capital
> (`capital.presets_frame()`) já registram, por produto, qual parâmetro mais move o
> resultado: no cartão é `"EAD/CCF (fator de conversão do limite rotativo)"`, e o cheque
> especial é descrito como rotativo com CCF ainda **mais instável**. Essa é a orientação
> qualitativa; o número sai daqui, e a instabilidade prevista lá é visível na tabela de
> desvio por faixa de utilização do tutorial 15.

`ead_from_measure(..., floor_at_drawn=)` merece atenção: o piso no valor já sacado
é **política de uso**, não fórmula. Ligado, impede exposição prevista abaixo do
que o cliente já devia (conservador na projeção); mas quebra a identidade com o
EAD realizado quando o contrato amortizou antes de quebrar, e viesa a estimativa
para cima. Por isso vem **desligado** por padrão.

---

## 6. A montagem — `ecl_table`

$$ECL = \sum_t PD_{marginal}(t)\cdot LGD(t)\cdot EAD(t)\cdot (1+i)^{-t/n}$$

| estágio | tratamento |
|---|---|
| 1 | ECL de **12 meses** — a soma para nos 12 primeiros períodos |
| 2 | ECL ***lifetime*** — a soma vai até o prazo remanescente |
| 3 | ativo problemático: **ELBE** sobre o saldo atual |

**A regra de transferência entre estágios (SICR) não está aqui**, e isso é
deliberado: o gatilho é política da instituição — combinação de atraso,
deterioração relativa da PD *lifetime*, renegociação, carência, listas de
observação — e cada uma documenta a sua. O módulo recebe a coluna pronta e faz a
conta. O que ele oferece é o insumo quantitativo do SICR: `PDCurve.forward`.

`ead_schedule` gera a trajetória de exposição (`constant`, `linear` ou `annuity`):
tratar o EAD como constante superestima a provisão de um contrato que amortiza,
porque a perda do 40º mês incide sobre um saldo que já é fração do de hoje.

`ecl_scenarios` roda a conta sob vários estados do ciclo e pondera por
probabilidade — o *forward-looking* de IFRS 9 / 4.966 é uma média ponderada de
cenários, não o cenário base com uma margem por cima. Os `z` podem vir de
julgamento ou da projeção de um modelo satélite de `credit_risk.econometric`.

---

## 7. Governança

`report` (matplotlib) e `tracking` (MLflow) são carregados **sob demanda**; o
núcleo roda com `numpy`/`pandas`/`scipy`/`scikit-learn` e **não** exige o extra
`[econometric]`.

- **Gráficos**: curvas nas quatro representações, Kaplan-Meier com banda de
  Greenwood, mapa de calor safra × idade, backtest previsto × observado, curva de
  recuperação com ELBE, distribuição bimodal do CCF, ECL por cenário e por grupo.
- **MLflow**: `log_lifetime_pd`, `log_elbe`, `log_ccf` e `log_ecl_run` — o número
  precisa ser reconstruível a partir dos insumos versionados (qual curva, qual
  calibração, qual cenário, qual data). `LifetimePD.to_json()`/`from_json()`
  reconstrói o objeto.

---

## Referências

**PD lifetime / estrutura a termo**
- Bellotti, T. & Crook, J. (2013). *Forecasting and stress testing credit card default using dynamic models*. International Journal of Forecasting.
- Botha, A. et al. (2025). *Approaches for modelling the term-structure of default risk under IFRS 9: a tutorial using discrete-time survival analysis*. International Journal of Data Science and Analytics. [arXiv:2507.15441](https://arxiv.org/abs/2507.15441)
- Vasicek, O. (2002). *Loan portfolio value*. Risk.

**LGD in default / ELBE**
- EBA/GL/2017/16 — *Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures*, §6.
- ECB (2024). *Guide to internal models*, §6.4 — estimação de ELBE e LGD *in default*.

**CCF / EAD**
- Moral, G. (2011). *EAD estimates for facilities with explicit limits*, em Engelmann & Rauhmeier (eds.), *The Basel II Risk Parameters*, 2ª ed., Springer.
- Valvonis, V. (2008). *Estimating EAD for retail exposures for Basel II purposes*. Journal of Credit Risk.
- Tong, E. N. C., Mues, C., Brown, I. & Thomas, L. C. (2016). *Exposure at default models with and without the credit conversion factor*. European Journal of Operational Research, 252(3), 910–920.
- Gürtler, M., Hibbeln, M. & Usselmann, P. (2018). *Exposure at default modeling — a theoretical and empirical assessment of estimation approaches and parameter choice*. Journal of Banking & Finance.

**Regulação**
- Resolução CMN 4.966/2021 e IFRS 9 — perda esperada, estágios, desconto e ativo problemático.
- BCBS — CRE32 (parâmetros de risco no IRB).

## Tutoriais

| # | notebook | o que cobre |
|---|---|---|
| 12 | [Perda esperada ponta a ponta](../../notebooks/tutoriais/12_tutorial_ecl.ipynb) | os três parâmetros juntos até a tabela de ECL |
| 13 | [PD lifetime](../../notebooks/tutoriais/13_tutorial_pd_lifetime.ipynb) | censura, as 4 representações, os 5 motores, calibração e ciclo |
| 14 | [ELBE](../../notebooks/tutoriais/14_tutorial_elbe.ipynb) | coorte variável, *workout*, desconto, add-on e LGD *in default* |
| 15 | [CCF / EAD](../../notebooks/tutoriais/15_tutorial_ccf.ipynb) | desenhos de base, 4 medidas, higiene, bimodalidade e backtest |

Comece pelo **12** para o mapa; vá direto ao **13**, **14** ou **15** para o módulo que
você precisa.
