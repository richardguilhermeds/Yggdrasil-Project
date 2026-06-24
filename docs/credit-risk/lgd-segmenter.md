# Tutorial — `SequentialLGDSegmenter`

Construtor sequencial e híbrido de segmentações para modelos de **LGD**, pensado para uso sob **Resolução CMN 4.966/2021** e **IFRS 9**. Resolve o problema central da árvore de decisão padrão do `sklearn` em risco de crédito: cortes estatisticamente ótimos mas ininterpretáveis e sem aderência à lógica de negócio.

A ideia: você **cresce a segmentação em camadas**, e a cada camada divide cada folha por uma nova variável usando **optimal binning** (automático) **ou cortes manuais** (política de negócio). Os bins resultantes viram, ao mesmo tempo, a régua de LGD e o esquema de bins do PSI.

---

## O que a classe faz

| Método | Função |
|---|---|
| `show_grow(...)` | **Preview** do split (LGD médio + representatividade) sem alterar nada |
| `grow(...)` | Efetiva o split — optimal binning **ou** cortes manuais; variável **numérica ou categórica** |
| `prune(...)` | Poda *bottom-up*: colapsa splits sem materialidade ou sem separação de LGD |
| `leaves()` | Tabela final de segmentos-folha: **nota de LGD (1..N)**, **descrição por extenso**, LGD por amostra (DES/OOT/...) |
| `tree()` | Desenha a **árvore hierárquica** em texto (nós internos + folhas, com n, repr. e LGD) |
| `psi()` / `psi_detalhe()` | PSI de estabilidade contra DES, usando os segmentos como bins |
| `metrics()` | Avalia a régua como **modelo de LGD**: MAE, RMSE e R² por amostra (DES, OOT, ...) |
| `assign(...)` | Rotula cada linha: id do segmento, **nota de LGD** e **descrição por extenso** |

Princípios de design relevantes para validação independente:

- **Optimal binning ajustado só em DES** quando há `sample_col` — a janela OOT nunca influencia onde os cortes caem, então o PSI mede deslocamento genuíno de população, não vazamento.
- **`monotonic_trend="auto_asc_desc"`** — a OptBinning detecta sozinha se o LGD sobe ou desce em cada ramo.
- **Crescimento ramo a ramo** — os cortes de uma variável dentro do ramo "LTV baixo" podem ser diferentes dos cortes no ramo "LTV alto", refletindo o comportamento real do LGD.
- **`history`** guarda modo (ótimo/manual) e cortes de cada passo — documentação de auditoria direta.

---

## Instalação

O segmentador faz parte do pacote **Yggdrasil**, no subpacote de domínio
`yggdrasil.credit_risk.lgd`. Em modo de desenvolvimento (na raiz do repositório):

```bash
pip install -e .            # núcleo (numpy, pandas, optbinning)
pip install -e ".[ui]"      # + interface ipywidgets (Jupyter/Databricks)
pip install -e ".[all]"     # + ui, mlflow e pyspark
```

Ou direto do GitHub:

```bash
pip install "git+https://github.com/zheage/Yggdrasil-Project.git"
```

Importe com:

```python
from yggdrasil.credit_risk.lgd import SequentialLGDSegmenter
```

---

## Exemplo prático — LGD de financiamento de veículo

O exemplo abaixo é **executável e reproduz exatamente as saídas mostradas**. Gera uma base sintética com três amostras (DES, OOT, ESTABILIDADE) onde o LGD sobe com LTV e atraso, cai com relacionamento e idade do contrato — e injeta um **drift de originação na janela OOT** (LTV mais alto) para o PSI ter o que detectar.

### Passo 0 — Base sintética

```python
import numpy as np
import pandas as pd
from yggdrasil.credit_risk.lgd import SequentialLGDSegmenter

rng = np.random.default_rng(42)

def gera_amostra(n, drift=False):
    ltv   = rng.beta(2.5 if not drift else 3.5, 3.0, n) * 1.4 + 0.3
    idade = rng.integers(1, 60, n)
    relac = rng.integers(0, 120, n)
    atraso = rng.integers(30, 720, n)
    base = (0.05 + 0.45 * np.clip(ltv - 0.5, 0, 1)
            + 0.10 * (atraso > 360)
            - 0.0010 * relac - 0.0015 * idade)
    lgd = np.clip(base + rng.normal(0, 0.08, n), 0, 1)
    return pd.DataFrame({
        "ltv": ltv, "idade_contrato_meses": idade,
        "tempo_relacionamento_meses": relac, "dias_atraso": atraso, "lgd": lgd,
    })

des = gera_amostra(8000);              des["amostra"] = "DES"
oot = gera_amostra(3000, drift=True);  oot["amostra"] = "OOT"
est = gera_amostra(3000);              est["amostra"] = "ESTABILIDADE"
df = pd.concat([des, oot, est], ignore_index=True)
```

### Passo 1 — Instanciar com amostras e referência do PSI

O parâmetro opcional `feature_labels` define rótulos amigáveis por variável, usados na descrição por extenso de cada segmento (Passo 6.1). Sem ele, o nome técnico da coluna é usado (com `_` virando espaço).

```python
labels = {
    "ltv": "LTV",
    "idade_contrato_meses": "idade do contrato (meses)",
    "dias_atraso": "dias em atraso",
    "tempo_relacionamento_meses": "tempo de relacionamento (meses)",
}

seg = SequentialLGDSegmenter(
    df, target="lgd", sample_col="amostra", ref_sample="DES",
    feature_labels=labels,
)
```
```
[init] amostras: ['DES', 'OOT', 'ESTABILIDADE'] | referência PSI = DES
```

### Passo 2 — Preview do split manual de LTV

Antes de efetivar, veja LGD médio e representatividade dos cortes de política propostos. Nada é alterado.

```python
seg.show_grow("ltv", splits=[0.70, 0.90, 1.00])
```
```
┌─ PREVIEW: dividir 'root'
│  feature = ltv | modo = manual | cortes = [0.7, 0.9, 1.0]
│  monotonicidade do LGD respeitada: True
      faixa    n  repr_%  lgd_medio  lgd_std
(-inf, 0.7] 2667    19.1     0.0633   0.0742
 (0.7, 0.9] 3305    23.6     0.1377   0.0978
   (0.9, 1] 1835    13.1     0.2037   0.1008
   (1, inf] 6193    44.2     0.3168   0.1211
```

A monotonicidade ascendente está confirmada e nenhum bin é frágil. Pode efetivar.

### Passo 3 — Crescer: LTV manual, depois idade ótima

Os dois modos no mesmo segmentador. A idade é binada **dentro de cada faixa de LTV**, e os cortes são definidos só pela amostra DES.

```python
seg.grow("ltv", splits=[0.70, 0.90, 1.00])      # manual (política)
seg.grow("idade_contrato_meses", max_n_bins=3)  # ótimo (OptBinning, só DES)
```
```
[grow] 'ltv' (manual) criou 4 segmentos. Folhas atuais: 4
[grow] 'idade_contrato_meses' (ótimo) criou 12 segmentos. Folhas atuais: 12
```

### Passo 4 — Folhas antes da poda

```python
# leaves() agora traz também nota_lgd e descricao; aqui mostramos um recorte
cols = ["segmento", "profundidade", "n", "repr_%", "lgd_medio",
        "lgd_DES", "lgd_OOT", "lgd_ESTABILIDADE"]
print(seg.leaves()[cols].to_string(index=False))
```
```
                                            segmento  profundidade    n  repr_%  lgd_medio  lgd_DES  lgd_OOT  lgd_ESTABILIDADE
ltv: (-inf, 0.7] | idade_contrato_meses: (32.5, inf]             2 1206     8.6     0.0483   0.0447   0.0609            0.0522
ltv: (-inf, 0.7] | idade_contrato_meses: (3.5, 32.5]             2 1317     9.4     0.0739   0.0768   0.0683            0.0682
ltv: (-inf, 0.7] | idade_contrato_meses: (-inf, 3.5]             2  144     1.0     0.0921   0.0931   0.0960            0.0857
 ltv: (0.7, 0.9] | idade_contrato_meses: (50.5, inf]             2  497     3.5     0.1013   0.0963   0.1111            0.1061
 ltv: (0.7, 0.9] | idade_contrato_meses: (4.5, 50.5]             2 2568    18.3     0.1412   0.1393   0.1338            0.1515
   ltv: (0.9, 1] | idade_contrato_meses: (56.5, inf]             2   92     0.7     0.1609   0.1778   0.1485            0.1320
  ltv: (0.9, 1] | idade_contrato_meses: (52.5, 56.5]             2  111     0.8     0.1622   0.1504   0.2107            0.1390
 ltv: (0.7, 0.9] | idade_contrato_meses: (-inf, 4.5]             2  240     1.7     0.1766   0.1822   0.1735            0.1618
  ltv: (0.9, 1] | idade_contrato_meses: (-inf, 52.5]             2 1632    11.7     0.2089   0.2115   0.2005            0.2105
   ltv: (1, inf] | idade_contrato_meses: (55.5, inf]             2  426     3.0     0.2768   0.2649   0.2946            0.2762
   ltv: (1, inf] | idade_contrato_meses: (5.5, 55.5]             2 5258    37.6     0.3161   0.3115   0.3268            0.3130
   ltv: (1, inf] | idade_contrato_meses: (-inf, 5.5]             2  509     3.6     0.3575   0.3644   0.3635            0.3334
```

Repare nas folhas frágeis: vários segmentos de LTV baixo/médio têm menos de 2% da carteira (ex: `0.7%`, `0.8%`, `1.0%`). Para LGD, caudas pequenas geram médias erráticas — é o que a poda resolve.

### Passo 5 — Podar

`prune` é *bottom-up* e desfaz um split (pai com todos os filhos folha) quando **alguma folha tem `repr_%` abaixo do mínimo** (materialidade) **ou a amplitude de LGD entre os filhos é pequena demais** (sem separação que justifique segmentar). Itera até estabilizar, permitindo poda em cascata.

```python
seg.prune(min_repr=3.0, min_lgd_gap=0.03)
```
```
[prune] colapsado 'ltv: (-inf, 0.7]' (3 folhas) — folha com repr_=1.0% < 3.0%
[prune] colapsado 'ltv: (0.7, 0.9]' (3 folhas) — folha com repr_=1.7% < 3.0%
[prune] colapsado 'ltv: (0.9, 1]'  (3 folhas) — folha com repr_=0.7% < 3.0%
[prune] concluído em 4 rodada(s). Folhas finais: 6
```

A poda colapsou os três ramos de LTV baixo/médio (onde a divisão por idade criava folhas imateriais), mas **manteve o ramo de LTV > 100% subdividido por idade** — ali os segmentos são materiais (3.0%, 37.6%, 3.6%) e bem separados em LGD. É exatamente a granularidade onde há materialidade, como a 4.966 espera.

### Passo 6 — Folhas finais

```python
print(seg.leaves()[cols].to_string(index=False))
```
```
                                         segmento  profundidade    n  repr_%  lgd_medio  lgd_DES  lgd_OOT  lgd_ESTABILIDADE
                                 ltv: (-inf, 0.7]             1 2667    19.1     0.0633   0.0636   0.0669            0.0611
                                  ltv: (0.7, 0.9]             1 3305    23.6     0.1377   0.1362   0.1332            0.1451
                                    ltv: (0.9, 1]             1 1835    13.1     0.2037   0.2063   0.1979            0.2026
ltv: (1, inf] | idade_contrato_meses: (55.5, inf]             2  426     3.0     0.2768   0.2649   0.2946            0.2762
ltv: (1, inf] | idade_contrato_meses: (5.5, 55.5]             2 5258    37.6     0.3161   0.3115   0.3268            0.3130
ltv: (1, inf] | idade_contrato_meses: (-inf, 5.5]             2  509     3.6     0.3575   0.3644   0.3635            0.3334
```

Régua limpa, monotônica, com 6 segmentos. As colunas `lgd_DES` / `lgd_OOT` / `lgd_ESTABILIDADE` permitem checar a **estabilidade do próprio LGD** dentro de cada segmento entre as janelas — pergunta distinta do PSI (que mede deslocamento de população).

### Passo 6.1 — Nota de LGD e descrição por extenso

A régua só vira política de negócio quando cada segmento ganha uma **nota ordenada** e uma **descrição legível**. O `leaves()` já entrega as duas: `nota_lgd` numera as folhas de 1 a N por ordem de LGD (1 = menor LGD / melhor recuperação) e `descricao` traduz o caminho técnico usando os `feature_labels`.

```python
lv = seg.leaves()
print(lv[["nota_lgd", "descricao", "repr_%", "lgd_medio"]].to_string(index=False))
```
```
 nota_lgd                                                   descricao  repr_%  lgd_medio
        1                                                 LTV até 0.7    19.1     0.0633
        2                                         LTV entre 0.7 e 0.9    23.6     0.1377
        3                                           LTV entre 0.9 e 1    13.1     0.2037
        4    LTV acima de 1 e idade do contrato (meses) acima de 55.5     3.0     0.2768
        5 LTV acima de 1 e idade do contrato (meses) entre 5.5 e 55.5    37.6     0.3161
        6          LTV acima de 1 e idade do contrato (meses) até 5.5     3.6     0.3575
```

Para inverter a ordem (nota 1 = maior LGD, padrão de algumas áreas de severidade), use `seg.leaves(ascending=False)`. A descrição compõe múltiplas condições com " e " e converte cada intervalo automaticamente: `(-inf, x]` → "até x", `(x, inf]` → "acima de x", `(x, y]` → "entre x e y".

### Passo 6.2 — Visualizar a árvore (`tree`)

`leaves()` dá a visão **plana** das folhas; `tree()` desenha a **hierarquia inteira**, com os nós intermediários, n, representatividade e LGD médio em cada nó, e a nota nas folhas. Pode ser chamado a qualquer momento da construção (inclusive antes da poda, para decidir o que podar).

```python
seg.tree()
```
```
TODA A CARTEIRA  (n=14000, 100.0%, LGD=0.2854)
├─ tipo de garantia em {alienacao_fiduciaria}  (n=6904, 49.3%, LGD=0.2138)  [nota 1]
├─ tipo de garantia em {aval}  (n=3160, 22.6%, LGD=0.3058)  [nota 2]
├─ tipo de garantia em {fianca_bancaria}  (n=2537, 18.1%, LGD=0.3445)  [nota 4]
└─ tipo de garantia em {sem_garantia}  (n=1399, 10.0%, LGD=0.4859)
   ├─ LTV até 0.8  (n=419, 3.0%, LGD=0.3353)  [nota 3]
   ├─ LTV entre 0.8 e 1  (n=341, 2.4%, LGD=0.4579)  [nota 5]
   └─ LTV acima de 1  (n=639, 4.6%, LGD=0.5995)  [nota 6]

(6 folhas | profundidade máxima 2)
```

Os filhos são ordenados por LGD, então a árvore lê de cima (menor LGD) para baixo. Repare que as **notas interligam ramos diferentes** (nota 3 fica num ramo, nota 4 em outro): a numeração é global por LGD, não por ramo — exatamente o que você quer numa régua final. Use `seg.tree(ascending=False)` para inverter a ordem.

### Passo 7 — PSI de estabilidade (DES como referência)

```python
print(seg.psi().to_string(index=False))
```
```
     amostra    psi classificacao
         OOT 0.1885       atenção
ESTABILIDADE 0.0007       estável
```

O PSI capturou o drift injetado em OOT (`0.1885`, faixa de atenção) e confirmou ESTABILIDADE estável (`0.0007`).

### Passo 8 — De onde vem o PSI

```python
det = seg.psi_detalhe()
det = det.reindex(det["psi_bin"].abs().sort_values(ascending=False).index)
print(det.head(5).to_string(index=False))
```
```
amostra                                          segmento  %_DES  %_atual  psi_bin
    OOT                                  ltv: (-inf, 0.7]  21.82     9.60   0.1004
    OOT ltv: (1, inf] | idade_contrato_meses: (5.5, 55.5]  34.38    49.20   0.0532
    OOT                                   ltv: (0.7, 0.9]  24.90    18.60   0.0184
    OOT ltv: (1, inf] | idade_contrato_meses: (55.5, inf]   2.56     4.63   0.0123
    OOT ltv: (1, inf] | idade_contrato_meses: (-inf, 5.5]   3.29     4.57   0.0042
```

A decomposição mostra a causa: a faixa de **LTV baixo encolheu de 21.8% para 9.6%** e a massa migrou para **LTV alto** — exatamente o drift de originação injetado. É isso que torna o PSI acionável: você aponta *qual* segmento migrou, não só o número agregado.

### Passo 8.1 — Métricas da régua como modelo de LGD

Além de estabilidade (PSI), vale medir o **poder preditivo** da régua. O `metrics()` trata a segmentação como um modelo: a predição de cada contrato é o **LGD médio do seu segmento na amostra de referência (DES)**, e avalia MAE, RMSE e R² em cada amostra — in-sample (DES) e out-of-time (OOT).

```python
seg.metrics()
```
```
     amostra    n    MAE   RMSE     R2
         DES 6000 0.0902 0.1114 0.5357
         OOT 3000 0.0906 0.1124 0.4867
ESTABILIDADE 3000 0.0923 0.1130 0.5165
```

O R² no DES mede quanto da variação do LGD a segmentação explica; no OOT, o mesmo fora da janela de desenvolvimento. Uma queda grande de R² do DES para o OOT é sinal de sobreajuste da régua — assim como o PSI sinaliza drift de população, as métricas sinalizam degradação de poder preditivo. Antes de qualquer split, com uma folha só, o R² no DES é 0 (a régua prediz a média global): cada split bom o aumenta.

### Passo 9 — Materializar o segmento

```python
df_seg = seg.assign("segmento_lgd")
# colunas geradas: segmento_lgd, segmento_lgd_nota (Int64), segmento_lgd_desc
df_seg[["ltv", "idade_contrato_meses", "lgd",
        "segmento_lgd_nota", "segmento_lgd_desc"]].head(8)
```
```
     ltv  idade_contrato_meses      lgd  segmento_lgd_nota                                           segmento_lgd_desc
0.850093                    39 0.118082                  2                                         LTV entre 0.7 e 0.9
0.960940                    23 0.187781                  3                                           LTV entre 0.9 e 1
1.103200                    50 0.233051                  5 LTV acima de 1 e idade do contrato (meses) entre 5.5 e 55.5
0.956844                     5 0.197658                  3                                           LTV entre 0.9 e 1
1.155575                     3 0.265330                  6          LTV acima de 1 e idade do contrato (meses) até 5.5
1.265973                    20 0.374342                  5 LTV acima de 1 e idade do contrato (meses) entre 5.5 e 55.5
0.961937                    25 0.122714                  3                                           LTV entre 0.9 e 1
0.695287                    45 0.194604                  1                                                 LTV até 0.7
```

O `assign` adiciona três colunas: `segmento_lgd` (id da folha), `segmento_lgd_nota` (a nota de LGD, tipo `Int64`) e `segmento_lgd_desc` (descrição por extenso) — prontas para virar a régua de LGD, *feature* categórica do modelo, ou rótulo legível em relatório. Desligue qualquer extra com `add_grade=False` ou `add_desc=False`.

---

## Crescer apenas um segmento (`only_segments`)

Por padrão, `grow` divide **todas** as folhas atuais. Mas em geral você quer aprofundar **só onde há materialidade ou heterogeneidade** — por exemplo, abrir o ramo de maior risco por faixa de atraso e deixar os ramos de LTV baixo intactos, já que ali o LGD é baixo e homogêneo. Tanto `show_grow` quanto `grow` aceitam `only_segments` com a lista de IDs dos segmentos-folha a dividir.

Partindo da segmentação só com LTV (4 folhas):

```python
print(seg.leaves()[["segmento", "n", "repr_%", "lgd_medio"]].to_string(index=False))
```
```
        segmento    n  repr_%  lgd_medio
ltv: (-inf, 0.7] 2667    19.1     0.0633
 ltv: (0.7, 0.9] 3305    23.6     0.1377
   ltv: (0.9, 1] 1835    13.1     0.2037
   ltv: (1, inf] 6193    44.2     0.3168
```

O ramo `ltv: (1, inf]` concentra 44% da carteira e o maior LGD — vale abrir só ele por faixa de atraso. **O ID em `only_segments` precisa ser exatamente o que aparece na coluna `segmento`** (copie de `leaves()`):

```python
# preview apenas desse ramo
seg.show_grow("dias_atraso", max_n_bins=3, only_segments=["ltv: (1, inf]"])
```
```
┌─ PREVIEW: dividir 'ltv: (1, inf]'
│  feature = dias_atraso | modo = ótimo | cortes = [360.5, 668.5]
│  monotonicidade do LGD respeitada: True
         faixa    n  repr_%  lgd_medio  lgd_std
 (-inf, 360.5] 2923    20.9     0.2638   0.1096
(360.5, 668.5] 2797    20.0     0.3631   0.1101
  (668.5, inf]  473     3.4     0.3710   0.1142
```

```python
# efetiva só nesse ramo
seg.grow("dias_atraso", max_n_bins=3, only_segments=["ltv: (1, inf]"])
```
```
[grow] 'dias_atraso' (ótimo) criou 3 segmentos. Folhas atuais: 6
```

Resultado — os três ramos de LTV mais baixo continuam **intactos como folhas de nível 1**, e só o ramo de alto risco ganhou profundidade:

```python
print(seg.leaves()[["segmento", "n", "repr_%", "lgd_medio"]].to_string(index=False))
```
```
                                   segmento    n  repr_%  lgd_medio
                           ltv: (-inf, 0.7] 2667    19.1     0.0633
                            ltv: (0.7, 0.9] 3305    23.6     0.1377
                              ltv: (0.9, 1] 1835    13.1     0.2037
 ltv: (1, inf] | dias_atraso: (-inf, 360.5] 2923    20.9     0.2638
ltv: (1, inf] | dias_atraso: (360.5, 668.5] 2797    20.0     0.3631
  ltv: (1, inf] | dias_atraso: (668.5, inf]  473     3.4     0.3710
```

Pontos práticos:

- **Árvore assimétrica é uma vantagem aqui**, não um defeito: você concentra granularidade onde o risco mora e mantém a régua enxuta no resto.
- `only_segments` aceita **vários IDs** — passe uma lista para abrir alguns ramos no mesmo passo, mas com a mesma variável.
- Para abrir ramos diferentes com **variáveis diferentes**, faça chamadas `grow` separadas, cada uma com seu `only_segments` e sua feature.
- A última folha gerada (`(668.5, inf]`, com 3.4%) é candidata natural à poda se você considerar 3.4% pouco material — o `prune` cuida disso depois.

---

## Variáveis categóricas (agrupar categorias)

Um bin não precisa ser numérico. Variáveis como **tipo de garantia**, canal de originação, produto ou região são categóricas, e o `grow` as trata nativamente — agrupando categorias em bins (ótimo, pela relação com o LGD) ou manualmente (você define os grupos). O tipo é **auto-detectado** pelo dtype da coluna; force com `dtype="cat"` ou `dtype="num"` se precisar.

Suponha uma coluna `tipo_garantia` com categorias `alienacao_fiduciaria`, `aval`, `fianca_bancaria`, `sem_garantia`.

### Categórico ótimo — a OptBinning agrupa por LGD

```python
seg.show_grow("tipo_garantia")   # dtype auto-detectado como 'cat'
```
```
┌─ PREVIEW: dividir 'root'
│  feature = tipo_garantia | tipo = cat | modo = ótimo
│  monotonicidade do LGD respeitada: True
                                faixa    n  repr_%  lgd_medio  lgd_std
tipo_garantia: {alienacao_fiduciaria} 5557    50.5     0.2691   0.1396
                tipo_garantia: {aval} 2410    21.9     0.4460   0.1449
     tipo_garantia: {fianca_bancaria} 1967    17.9     0.4968   0.1422
        tipo_garantia: {sem_garantia} 1066     9.7     0.7213   0.1389
```

Aqui as quatro categorias têm LGD bem distinto, então cada uma vira um bin. **Se duas categorias tivessem LGD parecido, a OptBinning as agruparia no mesmo bin** (ex: `{aval, fianca_bancaria}`) para respeitar `min_bin_size` e a monotonicidade — exatamente o que evita um scorecard com categorias raras instáveis. A nota e a descrição saem naturalmente:

```python
seg.grow("tipo_garantia")
print(seg.leaves()[["nota_lgd", "descricao", "repr_%", "lgd_medio"]].to_string(index=False))
```
```
 nota_lgd                                  descricao  repr_%  lgd_medio
        1 tipo de garantia em {alienacao_fiduciaria}    50.5     0.2691
        2                 tipo de garantia em {aval}    21.9     0.4460
        3      tipo de garantia em {fianca_bancaria}    17.9     0.4968
        4         tipo de garantia em {sem_garantia}     9.7     0.7213
```

### Categórico manual — você define os grupos

No modo manual, `splits` é uma **lista de grupos**, cada grupo uma lista de categorias. Categorias com comportamento parecido entram juntas (`seg2` abaixo é um segmentador novo, partindo da raiz):

```python
seg2 = SequentialLGDSegmenter(df, target="lgd", feature_labels=labels)
seg2.grow("tipo_garantia", splits=[
    ["alienacao_fiduciaria"],            # baixo risco isolado
    ["aval", "fianca_bancaria"],         # risco médio agrupado
    ["sem_garantia"],                    # alto risco isolado
])
print(seg2.leaves()[["nota_lgd", "descricao", "repr_%", "lgd_medio"]].to_string(index=False))
```
```
 nota_lgd                                     descricao  repr_%  lgd_medio
        1    tipo de garantia em {alienacao_fiduciaria}    50.5     0.2691
        2 tipo de garantia em {aval ou fianca_bancaria}    39.8     0.4688
        3            tipo de garantia em {sem_garantia}     9.7     0.7213
```

### Misturar categórico e numérico no mesmo caminho

O grande ganho: você pode abrir um ramo **categórico** e depois aprofundá-lo com uma variável **numérica** (ou vice-versa), via `only_segments`. Aqui, dentro do pior segmento de garantia (`sem_garantia`), abrimos por LTV:

```python
seg.grow("ltv", splits=[0.80, 1.00], only_segments=["tipo_garantia: {sem_garantia}"])
print(seg.leaves()[["nota_lgd", "descricao", "repr_%", "lgd_medio"]].to_string(index=False))
```
```
 nota_lgd                                              descricao  repr_%  lgd_medio
        1             tipo de garantia em {alienacao_fiduciaria}    50.5     0.2691
        2                             tipo de garantia em {aval}    21.9     0.4460
        3                  tipo de garantia em {fianca_bancaria}    17.9     0.4968
        4       tipo de garantia em {sem_garantia} e LTV até 0.8     2.7     0.5788
        5 tipo de garantia em {sem_garantia} e LTV entre 0.8 e 1     2.4     0.6846
        6    tipo de garantia em {sem_garantia} e LTV acima de 1     4.6     0.8239
```

A descrição por extenso compõe os dois tipos de condição com " e ", e o `assign` materializa tudo normalmente (`segmento_lgd`, `segmento_lgd_nota`, `segmento_lgd_desc`).

Pontos práticos:

- **Auto-detecção**: colunas `object`, `string`/`category` e `bool` viram `cat`; numéricas viram `num`. Use `dtype=` para forçar (ex: um código numérico que é na verdade categórico, como um `cod_produto`).
- **PSI funciona igual** sobre bins categóricos — útil para detectar drift de mix de produto/garantia entre DES e OOT.
- **Categorias novas em OOT** (que não existiam em DES) ficam fora dos grupos e recebem segmento nulo no `assign` — o que é informativo: categoria nova é, por si só, um sinal de mudança de população.
- A poda (`prune`) trata bins categóricos e numéricos de forma idêntica: colapsa por materialidade ou falta de separação de LGD.

---

## Integração com MLflow (`decision_tree_v1`)

Como toda a estrutura é tabular e auditável, o run fica natural:

```python
import mlflow

mlflow.set_experiment("decision_tree_v1")

with mlflow.start_run(run_name="lgd_segmentacao_veiculo"):
    # parâmetros da construção
    mlflow.log_params({
        "n_folhas": int(seg.leaves().shape[0]),
        "min_repr": 3.0,
        "min_lgd_gap": 0.03,
        "ref_sample": "DES",
    })

    # régua de LGD e PSI como artefatos
    seg.leaves().to_csv("regua_lgd.csv", index=False)
    psi_df = seg.psi()
    psi_df.to_csv("psi_estabilidade.csv", index=False)
    seg.psi_detalhe().to_csv("psi_detalhe.csv", index=False)
    mlflow.log_artifact("regua_lgd.csv")
    mlflow.log_artifact("psi_estabilidade.csv")
    mlflow.log_artifact("psi_detalhe.csv")

    # PSI por amostra como métrica
    for _, r in psi_df.iterrows():
        mlflow.log_metric(f"psi_{r['amostra']}", r["psi"])

    # histórico de construção (auditoria: o que foi ótimo vs. manual)
    import json
    with open("historico_splits.json", "w") as f:
        json.dump(seg.history, f, indent=2, default=str)
    mlflow.log_artifact("historico_splits.json")
```

---

## Cuidados

- **`repr_%` é sempre relativo à base inteira**, não ao ramo pai — é o que importa para materialidade regulatória. Um bin com 2% da carteira é frágil para LGD independentemente de quão grande seja dentro do seu ramo.
- **PSI fica instável com folhas pequenas** (daí o `eps` para evitar divisão por zero). Sempre **pode antes de interpretar o PSI** — segmentos imateriais inflam o índice artificialmente.
- **Mantenha 2–3 níveis** de profundidade no máximo para a régua continuar legível por risco e auditoria.
- **`min_lgd_gap` em vez de p-valor**: com amostras grandes, qualquer diferença vira "estatisticamente significativa". O corte por amplitude mínima de LGD reflete melhor como uma área de risco decide se vale ter um segmento separado — uma diferença significativa mas ínfima de LGD não justifica granularidade extra.

---

## Recursos adicionais

### Faltantes (NaN) em bin própria
Ao dividir por uma variável com valores ausentes, as linhas faltantes **viram um
segmento próprio** (`(faltante)`), em vez de serem descartadas. Isso vale para o
binning, o `predict`, o `to_pyspark` (vira `isNull()`) e o modelo MLflow — a
cobertura da régua sempre fecha 100% da carteira. Em dados de bureau, "não ter
informação" costuma ser preditivo, então o segmento de faltantes é tratado como
qualquer outro.

### Árvore automática e sugestão de split
```python
seg.fit_auto(max_depth=3, min_iv=0.02)   # árvore gulosa por IV, ponto de partida
seg.suggest_split("root")                # melhor variável (maior IV) para uma folha
```
`fit_auto` constrói uma árvore inicial escolhendo, em cada folha, a variável de
maior IV; depois você refina à mão (dividir, fundir, recolher).

### Régua em PySpark (scoring em escala)
```python
print(seg.to_pyspark())     # gera uma função F.when().otherwise() pronta
```
Cola o código gerado no Databricks e aplica a régua (segmento + nota + LGD) sobre
uma tabela Delta inteira, sem trazer para pandas.

### Registro no MLflow / Unity Catalog
```python
seg.log_to_mlflow(
    registered_model_name="catalogo.schema.lgd_segmentacao",
    registry_uri="databricks-uc",   # registra versão no Model Registry da UC
)
# scoring depois:
import mlflow.pyfunc
m = mlflow.pyfunc.load_model("models:/catalogo.schema.lgd_segmentacao/1")
df_scored = m.predict(df_novos[features])
```
Loga parâmetros, métricas (LGD por nota, MAE/RMSE/R², PSI) e artefatos
(`folhas.csv`, `arvore.txt`, `regua.json`, `regua_pyspark.py`), além do modelo
`pyfunc` aplicável. No Unity Catalog é obrigatório o nome em 3 níveis
(`catalogo.schema.modelo`) e permissão `CREATE MODEL` no schema.

### Interface interativa (ipywidgets)
```python
from yggdrasil.credit_risk.lgd import LGDSegmenterUI
ui = LGDSegmenterUI(df, target="lgd", sample_col="amostra", ref_sample="DES",
                    feature_labels={"ltv": "LTV", "garantia": "garantia"})
ui
```
Constrói a árvore clicando: preview, criar segmento, fechar/recolher folha,
**fundir folhas vizinhas**, sugerir split, auto-fit, podar, calcular IC bootstrap
e salvar no MLflow. Para variáveis categóricas, cada categoria ganha um **seletor
de grupo** (mesmas no mesmo grupo formam um nó), ordenadas por LGD — sem precisar
digitar listas. Requer `pip install yggdrasil[ui]` e um cluster interativo no
Databricks (DBR 13.0+ LTS).
