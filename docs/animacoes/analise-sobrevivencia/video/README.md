# Renderização do vídeo

As cenas `.dc.html` são animadas em CSS. Para virar MP4 (o formato que o
LinkedIn aceita), os quadros são capturados **congelando cada animação em um
tempo exato** via Web Animations API, em vez de gravar a tela em tempo real:

```js
document.getAnimations().forEach(a => { a.pause(); a.currentTime = t * 1000; });
```

Isso torna a renderização determinística — o quadro N é sempre idêntico, sem
depender de o navegador ter conseguido acompanhar 30 fps.

> Atenção: não use `page.screenshot({ animations: 'disabled' })` do Playwright
> aqui. Ele **adianta** as animações para o estado final antes de capturar, o
> que anula o controle de tempo e gera um vídeo estático.

## Pré-requisitos

- `ffmpeg` com `libx264` (o ffmpeg embutido do Playwright só faz VP8/WebM);
- `playwright-core` e um Chromium;
- as faces do Google Fonts embutidas como data URI no harness (o `file://` não
  carrega webfont remota de forma confiável, e sem elas o vídeo sai na fonte de
  fallback).

## Passos

1. `python3 ../build_cenas.py` — regera as cenas.
2. Monta um HTML por cena: CSS das fontes embutidas + o conteúdo do `.dc.html`
   com `{{modo}}` trocado por `anim-on`, mais o `__seek` acima.
3. `node capturar.mjs` — 1425 quadros a 30 fps (a duração do ciclo é lida do
   próprio `.dc.html`, então não diverge do gerador).
4. Codifica:

```bash
ffmpeg -y -framerate 30 -i quadros/%05d.png \
  -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 \
  -c:v libx264 -profile:v high -pix_fmt yuv420p -crf 18 -preset slow \
  -c:a aac -b:a 128k -shortest -movflags +faststart saida.mp4
```

Saída: 1080×1080, 30 fps, 47,5 s. Para 4:5 (que domina mais tela no feed
mobile): `-vf "pad=1080:1350:0:135:white"`.
