# Animação — Análise de sobrevivência

Roteiro visual em 5 cenas sobre análise de sobrevivência aplicada a crédito,
publicado como canvas de design (uma cena por quadro, cada uma animada em CSS).

| Cena | Arquivo | Assunto |
|---|---|---|
| 1 | `Main.dc.html` | A curva de sobrevivência S(t) e as marcas de censura |
| 2 | `Censura.dc.html` | Censura à direita: do tempo de calendário ao tempo desde a originação |
| 3 | `KaplanMeier.dc.html` | O estimador produto-limite, passo a passo, com a tabela de risco |
| 4 | `Risco.dc.html` | Hazard h(t), curva de maturação e a ponte S(t) = exp(−H(t)) |
| 5 | `Cox.dc.html` | Riscos proporcionais, razão de risco e a ponte com a PD lifetime |

`canvas.json` posiciona os quadros numa faixa horizontal, na ordem da narrativa.

## Como regerar

```bash
python3 build_cenas.py
```

O gerador **calcula** toda a geometria dos gráficos, em vez de fixar coordenadas
à mão. Isso é o que garante que o desenho e o texto nunca divirjam:

- os degraus de Kaplan-Meier vêm do produto ∏(1 − dᵢ/nᵢ) sobre a tabela de risco;
- S(t) da cena 4 é `exp(−H(t))` com H integrada por trapézio **sobre a mesma
  h(t) que está desenhada** no painel de cima;
- na cena 5 a curva do grupo de maior risco é `S₁(t)^HR`, que é o que a hipótese
  de riscos proporcionais implica — não uma segunda curva desenhada a olho;
- há uma trava que aborta a geração se um chip de fator encostar numa marca de
  censura.

Os números que aparecem no texto (média ingênua de 8,0 meses, pico de 4,4% no
mês 11, PD de 24m de 28% e 50%) são formatados a partir desses mesmos cálculos.

## Tema visual

Paleta e chrome dos eixos vêm do tema do projeto (skills `viz` e `video-manim`):
fundo branco, área de plot `#F9F9F9`, eixo `#CCCCCC` a 0.8px, grid `#AAAAAA`
tracejado a 35%, ticks `#444444`, rótulos `#2D2D2D`. Séries em `#2563EB` (azul,
sobrevivência), `#EF4444` (vermelho, evento/maior risco) e `#F59E0B` (âmbar,
hazard).

O estado de repouso de cada elemento **é o quadro final** da animação: com o
tweak `Animar` desligado — ou sob `prefers-reduced-motion` — a cena aparece
inteira, sem depender de nenhuma animação ter rodado.
