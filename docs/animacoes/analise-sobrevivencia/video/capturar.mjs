import { chromium } from 'playwright-core';
import fs from 'node:fs';
import path from 'node:path';

const AQUI = path.dirname(new URL(import.meta.url).pathname);
const CENAS = [
  { nome: 'Main',        ciclo: 9.0  },
  { nome: 'Censura',     ciclo: 11.0 },
  { nome: 'KaplanMeier', ciclo: 10.0 },
  { nome: 'Risco',       ciclo: 8.0  },
  { nome: 'Cox',         ciclo: 9.5  },
];
const FPS = 30;
const probe = process.argv.includes('--probe');

const saida = path.join(AQUI, probe ? 'probe' : 'quadros');
fs.rmSync(saida, { recursive: true, force: true });
fs.mkdirSync(saida, { recursive: true });

const navegador = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  args: ['--no-sandbox', '--force-color-profile=srgb', '--disable-lcd-text'],
});
const ctx = await navegador.newContext({
  viewport: { width: 1080, height: 1080 },
  deviceScaleFactor: 1,
  reducedMotion: 'no-preference',
});
const pag = await ctx.newPage();

let n = 0;
const t0 = Date.now();
for (const { nome, ciclo } of CENAS) {
  await pag.goto('file://' + path.join(AQUI, 'cenas', nome + '.html'));
  await pag.evaluate(() => document.fonts.ready);
  await pag.waitForTimeout(120);

  const tempos = probe
    ? [0, ciclo * 0.15, ciclo * 0.3, ciclo * 0.5, ciclo * 0.7, ciclo * 0.99]
    : Array.from({ length: Math.round(ciclo * FPS) }, (_, i) => i / FPS);

  for (const t of tempos) {
    await pag.evaluate((tt) => window.__seek(tt), t);
    const arq = probe
      ? path.join(saida, `${nome}-${t.toFixed(2)}.png`)
      : path.join(saida, String(++n).padStart(5, '0') + '.png');
    await pag.screenshot({ path: arq });  // ja congelado por __seek; 'disabled' adiantaria tudo pro fim
  }
  process.stdout.write(`  ${nome.padEnd(13)} ${tempos.length} quadros\n`);
}

await navegador.close();
const seg = ((Date.now() - t0) / 1000).toFixed(0);
console.log(`total ${probe ? 30 : n} quadros em ${seg}s -> ${saida}`);
