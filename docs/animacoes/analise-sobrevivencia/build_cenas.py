#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera as cenas .dc.html da animacao "Analise de sobrevivencia".

Toda a geometria dos graficos (passos de Kaplan-Meier, ticks de censura,
integral do hazard, curvas sob riscos proporcionais) e CALCULADA aqui, para
que os numeros no desenho e os numeros no texto nunca divirjam.

Paleta e chrome dos eixos vem do tema visual do projeto
(skills `viz`/`video-manim`): fundo branco, area de plot #F9F9F9, eixo
#CCCCCC 0.8px, grid #AAAAAA tracejado a 35%, ticks #444444, rotulos #2D2D2D.

Uso:  python3 build_cenas.py
"""

import math
import os

AQUI = os.path.dirname(os.path.abspath(__file__))

# ── Paleta (identica a viz/references/tema.md e video-manim/scripts/theme.py) ──
BLUE   = "#2563EB"   # primary  — S(t), sobrevivencia, grupo de menor risco
RED    = "#EF4444"   # danger   — evento (default), grupo de maior risco
GREEN  = "#10B981"   # secondary
AMBER  = "#F59E0B"   # accent   — h(t), taxa de risco
NEUTRAL= "#6B7280"   # linha de referencia, censura no texto
DARK   = "#1E293B"   # texto principal
BLUE_L = "#93C5FD"

PLOT_BG = "#F9F9F9"  # axes.facecolor
AXIS    = "#CCCCCC"  # axes.edgecolor
GRID    = "#AAAAAA"  # grid.color (usado a 35%)
TICK    = "#444444"  # xtick.color / ytick.color
LABEL   = "#2D2D2D"  # axes.labelcolor
HAIR    = "#E7E9EE"


def n(v, casas=1):
    """Numero curto para coordenada SVG."""
    s = f"{v:.{casas}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


def br(v, casas=2):
    """Numero no formato pt-BR (virgula decimal)."""
    return f"{v:.{casas}f}".replace(".", ",")


def polilinha(pts):
    return " ".join(f"{n(x)},{n(y)}" for x, y in pts)


def comprimento(pts):
    total = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def passos_km(tempos, esses, t_max):
    """Politica de degraus: continua a direita, cai no tempo do evento."""
    pts = [(0.0, 1.0)]
    s_ant = 1.0
    for t, s in zip(tempos, esses):
        pts.append((t, s_ant))
        pts.append((t, s))
        s_ant = s
    pts.append((t_max, s_ant))
    return pts


def s_em(tempos, esses, t):
    """S(t) da funcao escada: valor do ultimo evento com tempo <= t."""
    s = 1.0
    for ti, si in zip(tempos, esses):
        if ti <= t:
            s = si
    return s


# ── CSS compartilhado ────────────────────────────────────────────────────────
CSS_BASE = """
    *, *::before, *::after { box-sizing: border-box; }
    body { margin: 0; background: #FFFFFF; }
    a { color: #2563EB; text-decoration: none; }
    a:hover { color: #1D4ED8; }

    .cena {
      width: 1080px; height: 1080px; overflow: hidden;
      background: #FFFFFF; color: #1E293B;
      font-family: "IBM Plex Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
      font-size: 16px; line-height: 1.5;
      display: flex; flex-direction: column;
      padding: 60px 64px 54px;
    }

    .topo { display: flex; align-items: center; justify-content: space-between; gap: 24px;
            padding-bottom: 16px; border-bottom: 1px solid #E7E9EE; }
    .kicker { font-family: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
              font-size: 15px; font-weight: 500; letter-spacing: 0.16em;
              text-transform: uppercase; color: #6B7280; }
    .passo { font-family: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
             font-size: 15px; font-weight: 500; color: #94A3B8; letter-spacing: 0.08em; }

    h1 { font-family: "Space Grotesk", "Helvetica Neue", Helvetica, Arial, sans-serif;
         font-weight: 700; font-size: 54px; line-height: 1.03; letter-spacing: -0.022em;
         margin: 26px 0 0; color: #1E293B; text-wrap: balance; }
    .deck { font-size: 21px; line-height: 1.5; color: rgba(30,41,59,0.74);
            margin: 15px 0 0; max-width: 880px; text-wrap: pretty; }

    .formula { font-family: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
               font-size: 19px; color: #1E293B; background: #FFFFFF;
               border: 1px solid #CCCCCC; border-radius: 8px;
               padding: 11px 16px; display: inline-flex; align-items: center; gap: 10px;
               white-space: nowrap; }
    .formula em { font-style: italic; color: #2563EB; font-weight: 500; }
    .formula .sub { font-size: 15px; color: #6B7280; }

    .grafico { margin-top: auto; }
    .grafico svg { display: block; width: 100%; height: auto; }

    .nota { border: 1px solid #CCCCCC; border-radius: 10px; background: #FFFFFF;
            padding: 18px 22px; display: flex; gap: 16px; align-items: flex-start; }
    .nota .marca { width: 4px; align-self: stretch; border-radius: 2px; background: #2563EB; flex: none; }
    .nota p { margin: 0; font-size: 19px; line-height: 1.46; color: #1E293B; text-wrap: pretty; }
    .nota strong { font-weight: 600; }

    .legenda { display: flex; flex-wrap: wrap; gap: 12px 28px; align-items: center;
               font-size: 17px; color: #2D2D2D; }
    .legenda .item { display: flex; align-items: center; gap: 9px; }
    .legenda .chave { width: 22px; height: 22px; flex: none; }

    /* chrome dos eixos — valores exatos do tema matplotlib do projeto */
    .ax    { stroke: #CCCCCC; stroke-width: 0.8; fill: none; }
    .grid  { stroke: #AAAAAA; stroke-width: 0.6; stroke-dasharray: 4 4; opacity: 0.35; fill: none; }
    .campo { fill: #F9F9F9; }
    .tick  { font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 15px;
             fill: #444444; font-variant-numeric: tabular-nums; }
    .axlab { font-family: "IBM Plex Sans", Helvetica, Arial, sans-serif; font-size: 16px;
             fill: #2D2D2D; font-weight: 500; }
    .marcador { font-family: "IBM Plex Sans", Helvetica, Arial, sans-serif; font-size: 17px;
                fill: #1E293B; font-weight: 600; }
    .mono { font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 16px;
            fill: #1E293B; font-variant-numeric: tabular-nums; }
    .ref   { stroke: #6B7280; stroke-width: 1.2; stroke-dasharray: 6 5; fill: none; }

    .curva { fill: none; stroke-width: 2.5; stroke-linejoin: round; stroke-linecap: round; }
    .anel  { stroke: #F9F9F9; stroke-width: 2; }

    /* respeita quem pediu menos movimento; o estado de repouso ja e o quadro final */
    @media (prefers-reduced-motion: reduce) {
      .cena *, .cena *::before, .cena *::after { animation: none !important; }
    }
    /* tweak "animar" desligado — mesmo caminho */
    .anim-off *, .anim-off *::before, .anim-off *::after { animation: none !important; }
"""

CABECA = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script src="./support.js"></script>
</head>
<body>
<x-dc>
<helmet>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400;1,500&family=IBM+Plex+Mono:wght@400;500&display=swap">
  <style>
"""

RABO = """</x-dc>
<script data-dc-script data-props='{"$preview":{"width":1080,"height":1080},"animar":{"editor":"boolean","default":true}}'>
class Component extends DCLogic {
  renderVals() {
    return { modo: (this.props.animar ?? true) ? 'anim-on' : 'anim-off' };
  }
}
</script>
</body>
</html>
"""


def cena(css_extra, corpo):
    """Monta o arquivo .dc.html completo de uma cena."""
    return (
        CABECA
        + CSS_BASE
        + css_extra
        + "  </style>\n</helmet>\n"
        + '<div class="cena {{modo}}">\n'
        + corpo
        + "\n</div>\n"
        + RABO
    )


def topo(kicker, passo):
    return (
        '  <div class="topo">\n'
        f'    <span class="kicker">{kicker}</span>\n'
        f'    <span class="passo">{passo}</span>\n'
        "  </div>\n"
    )


def nota(texto):
    return (
        '  <div class="nota">\n'
        '    <span class="marca"></span>\n'
        f"    <p>{texto}</p>\n"
        "  </div>\n"
    )


def eixos(x0, x1, y0, y1, xticks, yticks, xfmt, yfmt, xlab, ylab, fx, fy,
          grid_h=True, grid_v=False):
    """Campo do plot, grid, eixos e ticks — no chrome do tema do projeto."""
    o = []
    o.append(f'<rect class="campo" x="{n(x0)}" y="{n(y0)}" width="{n(x1-x0)}" height="{n(y1-y0)}" rx="3"/>')
    if grid_h:
        for v in yticks:
            y = fy(v)
            o.append(f'<line class="grid" x1="{n(x0)}" y1="{n(y)}" x2="{n(x1)}" y2="{n(y)}"/>')
    if grid_v:
        for v in xticks:
            x = fx(v)
            o.append(f'<line class="grid" x1="{n(x)}" y1="{n(y0)}" x2="{n(x)}" y2="{n(y1)}"/>')
    o.append(f'<line class="ax" x1="{n(x0)}" y1="{n(y1)}" x2="{n(x1)}" y2="{n(y1)}"/>')
    o.append(f'<line class="ax" x1="{n(x0)}" y1="{n(y0)}" x2="{n(x0)}" y2="{n(y1)}"/>')
    for v in xticks:
        x = fx(v)
        o.append(f'<text class="tick" x="{n(x)}" y="{n(y1+26)}" text-anchor="middle">{xfmt(v)}</text>')
    for v in yticks:
        y = fy(v)
        o.append(f'<text class="tick" x="{n(x0-12)}" y="{n(y+5)}" text-anchor="end">{yfmt(v)}</text>')
    o.append(f'<text class="axlab" x="{n(x1)}" y="{n(y1+56)}" text-anchor="end">{xlab}</text>')
    # rotulo do eixo Y girado, como no tema matplotlib do projeto — nao briga com o topo do viewBox
    ym = (y0 + y1) / 2
    o.append(f'<text class="axlab" x="{n(x0-64)}" y="{n(ym)}" text-anchor="middle" '
             f'transform="rotate(-90 {n(x0-64)} {n(ym)})">{ylab}</text>')
    return "\n".join(o)


CSS_COMUM = """
    .deck em { font-style: italic; font-weight: 500; color: #1E293B; }
    .cabeca { flex: none; }
    .rodape { flex: none; margin-top: auto; display: flex; flex-direction: column; gap: 18px; }
    .faixa { font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 14px;
             letter-spacing: 0.1em; text-transform: uppercase; color: #94A3B8;
             display: flex; gap: 24px; flex-wrap: wrap; }
    .faixa span { white-space: nowrap; }
    /* mono mantem o zero inconfundivel; a margem negativa fecha o vao da celula */
    .formula sub, .formula sup { font-size: 0.7em; margin-left: -0.3em; }
    .deck sub, .deck sup { font-size: 0.72em; }
"""


# ═══════════════════════════════════════════════════════════════════════════
# CENA 1 — capa: a curva de sobrevivencia
# ═══════════════════════════════════════════════════════════════════════════
def cena_main():
    W, H = 952, 432
    x0, x1, y0, y1 = 88.0, 904.0, 20.0, 336.0
    t_max = 24.0
    fx = lambda t: x0 + t * (x1 - x0) / t_max
    fy = lambda s: y1 - s * (y1 - y0)

    tempos = [1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 14, 16, 18, 20, 22, 24]
    esses = [0.972, 0.941, 0.905, 0.862, 0.812, 0.757, 0.700, 0.645,
             0.592, 0.548, 0.513, 0.486, 0.466, 0.452]
    censuras = [2.2, 5.2, 8.3, 11.2, 13.0, 15.0, 17.2, 19.0, 21.0, 23.0]

    pts = passos_km(tempos, esses, t_max)
    px = [(fx(t), fy(s)) for t, s in pts]
    L = comprimento(px)

    CICLO = 9.0
    DESENHO = 4.6
    pct_fim = DESENHO / CICLO * 100.0

    g = []
    g.append(eixos(x0, x1, y0, y1,
                   [0, 4, 8, 12, 16, 20, 24], [0, 0.25, 0.5, 0.75, 1.0],
                   lambda v: str(int(v)), lambda v: br(v, 2).rstrip("0").rstrip(",") if v in (0, 1) else br(v, 2),
                   "tempo desde a originação (meses)", "S(t)", fx, fy))

    g.append(f'<polyline class="curva km" points="{polilinha(px)}"/>')

    # marcas de censura: tick vertical sobre o degrau corrente
    for t in censuras:
        s = s_em(tempos, esses, t)
        x, y = fx(t), fy(s)
        f = (x - x0) / (x1 - x0)
        atraso = f * DESENHO
        g.append(
            f'<line class="cens" x1="{n(x)}" y1="{n(y-8)}" x2="{n(x)}" y2="{n(y+8)}" '
            f'style="animation-delay: {n(atraso,2)}s"/>'
        )

    xf, yf = fx(t_max), fy(esses[-1])
    g.append(f'<circle class="ponta anel" cx="{n(xf)}" cy="{n(yf)}" r="5.5" fill="{BLUE}"/>')
    g.append(f'<text class="marcador ponta" x="{n(xf-10)}" y="{n(yf+34)}" text-anchor="end">'
             f'S(24 m) = {br(esses[-1], 2)}</text>')

    css = CSS_COMUM + f"""
    .km {{ stroke: {BLUE}; stroke-dasharray: {n(L,1)}; stroke-dashoffset: 0;
           animation: desenha {CICLO}s linear infinite both; }}
    .cens {{ stroke: {BLUE}; stroke-width: 2.5; stroke-linecap: round; opacity: 1;
             animation: surge {CICLO}s linear infinite both; }}
    .ponta {{ opacity: 1; animation: surgeTarde {CICLO}s linear infinite both; }}

    @keyframes desenha {{ 0% {{ stroke-dashoffset: {n(L,1)}; }}
                          {n(pct_fim,1)}% {{ stroke-dashoffset: 0; }}
                          100% {{ stroke-dashoffset: 0; }} }}
    @keyframes surge {{ 0% {{ opacity: 0; }} 5% {{ opacity: 1; }} 100% {{ opacity: 1; }} }}
    @keyframes surgeTarde {{ 0%, {n(pct_fim,1)}% {{ opacity: 0; }}
                             {n(pct_fim+7,1)}% {{ opacity: 1; }} 100% {{ opacity: 1; }} }}

    .titulo-capa {{ font-size: 68px; }}
    .chip-linha {{ display: flex; gap: 14px; align-items: center; margin-top: 26px; }}
"""

    corpo = (
        '  <div class="cabeca">\n'
        + topo("Estatística aplicada a crédito", "cena 1 / 5")
        + '    <h1 class="titulo-capa">Análise de sobrevivência</h1>\n'
        '    <p class="deck">A pergunta não é <em>se</em> o evento acontece, é <em>quando</em>. '
        "E o dado mais importante é o de quem, até agora, não teve evento nenhum.</p>\n"
        '    <div class="chip-linha">\n'
        '      <span class="formula"><em>S</em>(t) = P(<em>T</em> &gt; t)'
        '<span class="sub">probabilidade de passar de t sem evento</span></span>\n'
        "    </div>\n"
        "  </div>\n"
        '  <div class="grafico">\n'
        f'    <svg viewBox="0 0 {W} {H}" role="img" aria-label="Curva de sobrevivência de Kaplan-Meier caindo de 1,00 a 0,45 ao longo de 24 meses, com marcas de censura">\n'
        + "      " + "\n      ".join(g) + "\n"
        "    </svg>\n"
        "  </div>\n"
        '  <div class="rodape">\n'
        '    <div class="legenda">\n'
        '      <span class="item">\n'
        '        <svg class="chave" viewBox="0 0 22 22" aria-hidden="true">'
        f'<line x1="11" y1="3" x2="11" y2="19" stroke="{BLUE}" stroke-width="2.5" stroke-linecap="round"/>'
        f'<line x1="2" y1="11" x2="20" y2="11" stroke="{BLUE}" stroke-width="2.5" opacity="0.35" stroke-linecap="round"/></svg>\n'
        "        marca de censura — saiu da observação sem ter tido o evento\n"
        "      </span>\n"
        "    </div>\n"
        + nota("A curva cai a cada evento observado e é <strong>atravessada, sem cair</strong>, "
               "por quem saiu antes do fim. Jogar essa gente fora é o erro que a análise de "
               "sobrevivência existe para evitar.")
        + '    <div class="faixa"><span>02 censura</span><span>03 Kaplan-Meier</span>'
        "<span>04 risco</span><span>05 Cox</span></div>\n"
        "  </div>"
    )
    return cena(css, corpo)


# ═══════════════════════════════════════════════════════════════════════════
# CENA 2 — censura a direita: do tempo de calendario ao tempo de analise
# ═══════════════════════════════════════════════════════════════════════════
def cena_censura():
    W, H = 952, 470
    x0, x1 = 140.0, 880.0
    y_ax = 380.0
    ESC = 37.0                      # px por mes
    fx = lambda t: x0 + t * ESC
    CORTE = 18.0                    # janela de observacao fecha aqui

    # (originacao, fim da observacao, teve evento)
    contratos = [
        (0, 5, True), (0, 18, False), (1, 13, True), (2, 18, False),
        (3, 9, True), (4, 18, False), (6, 15, True), (8, 18, False),
    ]
    duracoes_evento = sorted(f - o for o, f, e in contratos if e)
    duracoes_censura = sorted(f - o for o, f, e in contratos if not e)
    media_ingenua = sum(duracoes_evento) / len(duracoes_evento)

    CICLO = 11.0
    p_cresce = 23.6      # barras crescem ate aqui
    p_espera = 38.0      # fim da leitura em calendario
    p_desliza = 52.7     # alinhamento concluido

    g = []
    g.append(f'<rect class="campo" x="{n(x0)}" y="24" width="{n(x1-x0)}" height="{n(y_ax-24)}" rx="3"/>')
    for t in (0, 4, 8, 12, 16, 20):
        g.append(f'<line class="grid" x1="{n(fx(t))}" y1="24" x2="{n(fx(t))}" y2="{n(y_ax)}"/>')
    g.append(f'<line class="ax" x1="{n(x0)}" y1="{n(y_ax)}" x2="{n(x1)}" y2="{n(y_ax)}"/>')
    for t in (0, 4, 8, 12, 16, 20):
        g.append(f'<text class="tick" x="{n(fx(t))}" y="{n(y_ax+26)}" text-anchor="middle">{t}</text>')

    # regiao nao observada + linha de corte (some quando o eixo vira tempo de analise)
    g.append('<g class="corte">')
    g.append(f'  <rect x="{n(fx(CORTE))}" y="24" width="{n(x1-fx(CORTE))}" height="{n(y_ax-24)}" fill="{NEUTRAL}" opacity="0.07"/>')
    g.append(f'  <line class="ref" x1="{n(fx(CORTE))}" y1="24" x2="{n(fx(CORTE))}" y2="{n(y_ax)}"/>')
    g.append(f'  <text class="axlab" x="{n(fx(CORTE)-10)}" y="16" text-anchor="end">a janela fecha aqui</text>')
    g.append("</g>")

    # rotulos das contas ficam fixos: a identidade do contrato nao desliza
    for i in range(len(contratos)):
        y = 50.0 + i * 41.0
        g.append(f'<text class="tick" x="{n(x0-14)}" y="{n(y+5)}" text-anchor="end">C{i+1}</text>')

    for i, (orig, fim, evento) in enumerate(contratos):
        y = 50.0 + i * 41.0
        dx = -orig * ESC
        cor = RED if evento else BLUE
        xa, xb = fx(orig), fx(fim)
        g.append(f'<g class="linha" style="--dx: {n(dx)}px">')
        g.append(f'  <rect class="barra" x="{n(xa)}" y="{n(y-7)}" width="{n(xb-xa)}" height="14" rx="4" fill="{cor}" opacity="0.30"/>')
        g.append(f'  <g class="fim" style="animation-delay: {n(i*0.06, 2)}s">')
        if evento:
            g.append(f'    <circle class="anel" cx="{n(xb)}" cy="{n(y)}" r="6.5" fill="{RED}"/>')
        else:
            g.append(f'    <line x1="{n(xb)}" y1="{n(y)}" x2="{n(xb+15)}" y2="{n(y)}" stroke="{BLUE}" stroke-width="2.5" stroke-linecap="round"/>')
            g.append(f'    <polyline points="{n(xb+9)},{n(y-5.5)} {n(xb+16)},{n(y)} {n(xb+9)},{n(y+5.5)}" fill="none" stroke="{BLUE}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>')
        g.append("  </g>")
        g.append("</g>")

    g.append(f'<text class="axlab eixo-cal" x="{n(x1)}" y="{n(y_ax+56)}" text-anchor="end">tempo de calendário (meses)</text>')
    g.append(f'<text class="axlab eixo-ana" x="{n(x1)}" y="{n(y_ax+56)}" text-anchor="end">tempo desde a originação (meses)</text>')

    css = CSS_COMUM + f"""
    .linha {{ transform: translateX(var(--dx)); animation: desliza {CICLO}s cubic-bezier(.65,0,.35,1) infinite both; }}
    .barra {{ transform-box: fill-box; transform-origin: left center; transform: scaleX(1);
              animation: cresce {CICLO}s cubic-bezier(.33,0,.2,1) infinite both; }}
    .fim {{ opacity: 1; animation: surge {CICLO}s linear infinite both; }}
    .corte {{ opacity: 0; animation: some {CICLO}s linear infinite both; }}
    .eixo-cal {{ opacity: 0; animation: some {CICLO}s linear infinite both; }}
    .eixo-ana {{ opacity: 1; animation: entra {CICLO}s linear infinite both; }}

    @keyframes cresce {{ 0% {{ transform: scaleX(0); }}
                         {n(p_cresce,1)}% {{ transform: scaleX(1); }}
                         100% {{ transform: scaleX(1); }} }}
    @keyframes surge {{ 0%, {n(p_cresce,1)}% {{ opacity: 0; }}
                        {n(p_cresce+3.5,1)}% {{ opacity: 1; }} 100% {{ opacity: 1; }} }}
    @keyframes desliza {{ 0%, {n(p_espera,1)}% {{ transform: translateX(0); }}
                          {n(p_desliza,1)}%, 100% {{ transform: translateX(var(--dx)); }} }}
    @keyframes some {{ 0% {{ opacity: 0; }} 6% {{ opacity: 1; }}
                       {n(p_espera,1)}% {{ opacity: 1; }}
                       {n(p_espera+9,1)}% {{ opacity: 0; }} 100% {{ opacity: 0; }} }}
    @keyframes entra {{ 0%, {n(p_espera+4,1)}% {{ opacity: 0; }}
                        {n(p_desliza,1)}% {{ opacity: 1; }} 100% {{ opacity: 1; }} }}
"""

    lista = ", ".join(str(d) for d in duracoes_censura[:-1]) + " e " + str(duracoes_censura[-1])
    corpo = (
        '  <div class="cabeca">\n'
        + topo("O dado incompleto", "cena 2 / 5")
        + "    <h1>Censura à direita</h1>\n"
        '    <p class="deck">Quando a janela de observação fecha, metade da carteira ainda não '
        "falhou. O tempo dessas contas não é zero e não é infinito: é <em>incompleto</em>. "
        "Realinhar tudo pela originação é o que torna as durações comparáveis.</p>\n"
        "  </div>\n"
        '  <div class="grafico">\n'
        f'    <svg viewBox="0 0 {W} {H}" role="img" aria-label="Oito contratos em linhas do tempo; quatro terminam em default e quatro seguem vivos quando a janela fecha, depois todos são realinhados pela data de originação">\n'
        + "      " + "\n      ".join(g) + "\n"
        "    </svg>\n"
        "  </div>\n"
        '  <div class="rodape">\n'
        '    <div class="legenda">\n'
        '      <span class="item"><svg class="chave" viewBox="0 0 22 22" aria-hidden="true">'
        f'<circle cx="11" cy="11" r="6.5" fill="{RED}"/></svg> evento observado (default)</span>\n'
        '      <span class="item"><svg class="chave" viewBox="0 0 26 22" aria-hidden="true">'
        f'<line x1="2" y1="11" x2="19" y2="11" stroke="{BLUE}" stroke-width="2.5" stroke-linecap="round"/>'
        f'<polyline points="14,5.5 20.5,11 14,16.5" fill="none" stroke="{BLUE}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        " censurado — segue vivo além do que foi visto</span>\n"
        "    </div>\n"
        + nota(
            f"Usar só quem falhou dá <strong>{br(media_ingenua,1)} meses</strong> de média até o "
            f"default. Mas os censurados já passaram {lista} meses sem falhar — e nenhum deles "
            "entrou nessa conta. O viés é sempre para baixo."
        )
        + "  </div>"
    )
    return cena(css, corpo)


# ═══════════════════════════════════════════════════════════════════════════
# CENA 3 — Kaplan-Meier, o estimador produto-limite
# ═══════════════════════════════════════════════════════════════════════════
def cena_kaplan_meier():
    W, H = 952, 322
    x0, x1, y0, y1 = 88.0, 910.0, 14.0, 250.0
    t_max = 15.0
    fx = lambda t: x0 + t * (x1 - x0) / t_max
    fy = lambda s: y1 - s * (y1 - y0)

    # (tempo do evento, n sob risco, eventos)
    riscos = [(3, 10, 1), (5, 8, 2), (9, 5, 1), (12, 3, 1)]
    censuras = [4, 7, 11, 14]

    tempos, esses, fatores = [], [], []
    s = 1.0
    for t, nr, d in riscos:
        f = 1.0 - d / nr
        s *= f
        tempos.append(t); esses.append(s); fatores.append(f)

    pts = passos_km(tempos, esses, t_max)
    px = [(fx(t), fy(sv)) for t, sv in pts]
    L = comprimento(px)

    CICLO, DESENHO = 10.0, 5.2
    p_fim = DESENHO / CICLO * 100.0
    atraso = lambda t: (fx(t) - x0) / (x1 - x0) * DESENHO

    g = []
    g.append(eixos(x0, x1, y0, y1,
                   [0, 3, 6, 9, 12, 15], [0, 0.25, 0.5, 0.75, 1.0],
                   lambda v: str(int(v)),
                   lambda v: br(v, 2).rstrip("0").rstrip(",") if v in (0, 1) else br(v, 2),
                   "tempo (meses)", "Ŝ(t)", fx, fy))
    g.append(f'<polyline class="curva km" points="{polilinha(px)}"/>')

    for t in censuras:
        sv = s_em(tempos, esses, t)
        x, y = fx(t), fy(sv)
        g.append(f'<line class="cens" x1="{n(x)}" y1="{n(y-8)}" x2="{n(x)}" y2="{n(y+8)}" '
                 f'style="animation-delay: {n(atraso(t),2)}s"/>')

    # chips do fator de queda, a esquerda de cada degrau
    caixas_chip = []
    s_ant = 1.0
    for (t, nr, d), f in zip(riscos, fatores):
        topo_y = fy(s_ant) + 18
        xr = fx(t) - 10
        g.append(f'<g class="chip" style="animation-delay: {n(atraso(t)+0.15,2)}s">')
        g.append(f'  <rect x="{n(xr-70)}" y="{n(topo_y)}" width="70" height="24" rx="6" fill="#FFFFFF" stroke="{AXIS}" stroke-width="1"/>')
        g.append(f'  <text class="chip-txt" x="{n(xr-35)}" y="{n(topo_y+16.5)}" text-anchor="middle">×{br(f,3)}</text>')
        g.append("</g>")
        caixas_chip.append((xr - 70, topo_y, xr, topo_y + 24))
        s_ant *= f

    # trava de layout: chip do fator nunca pode encostar numa marca de censura
    for cx0, cy0, cx1, cy1 in caixas_chip:
        for t in censuras:
            tx, ty = fx(t), fy(s_em(tempos, esses, t))
            if cx0 - 6 < tx < cx1 + 6 and cy0 - 6 < ty + 9 and ty - 9 < cy1 + 6:
                raise SystemExit(f"colisao: chip em x={cx0:.0f}..{cx1:.0f} bate na censura t={t}")

    xf, yf = fx(t_max), fy(esses[-1])
    g.append(f'<circle class="ponta anel" cx="{n(xf)}" cy="{n(yf)}" r="5.5" fill="{BLUE}"/>')

    linhas = []
    for (t, nr, d), f, sv in zip(riscos, fatores, esses):
        linhas.append(
            f'        <tr style="animation-delay: {n(atraso(t),2)}s">'
            f"<td>{t}</td><td>{nr}</td><td>{d}</td><td>{br(f,3)}</td>"
            f'<td class="destaque">{br(sv,3)}</td></tr>'
        )

    css = CSS_COMUM + f"""
    .km {{ stroke: {BLUE}; stroke-dasharray: {n(L,1)}; stroke-dashoffset: 0;
           animation: desenha {CICLO}s linear infinite both; }}
    .cens {{ stroke: {BLUE}; stroke-width: 2.5; stroke-linecap: round; opacity: 1;
             animation: surge {CICLO}s linear infinite both; }}
    .chip {{ opacity: 1; animation: surge {CICLO}s linear infinite both; }}
    .chip-txt {{ font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 14px;
                 fill: #1E293B; font-variant-numeric: tabular-nums; }}
    .ponta {{ opacity: 1; animation: surgeTarde {CICLO}s linear infinite both; }}
    @keyframes desenha {{ 0% {{ stroke-dashoffset: {n(L,1)}; }}
                          {n(p_fim,1)}% {{ stroke-dashoffset: 0; }} 100% {{ stroke-dashoffset: 0; }} }}
    @keyframes surge {{ 0% {{ opacity: 0; }} 4% {{ opacity: 1; }} 100% {{ opacity: 1; }} }}
    @keyframes surgeTarde {{ 0%, {n(p_fim,1)}% {{ opacity: 0; }}
                             {n(p_fim+6,1)}% {{ opacity: 1; }} 100% {{ opacity: 1; }} }}

    .tabela {{ width: 100%; border-collapse: collapse; margin-top: 14px;
               font-variant-numeric: tabular-nums; }}
    .tabela th {{ font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace;
                  font-size: 13px; font-weight: 500; letter-spacing: 0.04em;
                  color: #6B7280; text-align: right;
                  padding: 0 0 9px; border-bottom: 1px solid {AXIS}; white-space: nowrap; }}
    .tabela th:first-child, .tabela td:first-child {{ text-align: left; }}
    .tabela td {{ font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace;
                  font-size: 19px; color: #1E293B; text-align: right;
                  padding: 5px 0; border-bottom: 1px solid {HAIR}; }}
    .tabela tbody tr {{ opacity: 1; animation: surge {CICLO}s linear infinite both; }}
    .tabela .destaque {{ font-weight: 500; }}
"""

    corpo = (
        '  <div class="cabeca">\n'
        + topo("O estimador", "cena 3 / 5")
        + "    <h1>Kaplan-Meier, o produto-limite</h1>\n"
        '    <p class="deck">A curva não é ajustada: é <em>construída</em>. A cada tempo de evento '
        "ela é multiplicada pela fração de quem sobreviveu àquele instante.</p>\n"
        '    <div class="chip-linha" style="display:flex; margin-top:24px">\n'
        '      <span class="formula"><em>Ŝ</em>(t) = ∏ (1 − <em>d</em><sub>i</sub> / <em>n</em><sub>i</sub>)'
        '<span class="sub">para cada tempo de evento t<sub>i</sub> ≤ t</span></span>\n'
        "    </div>\n"
        "  </div>\n"
        '  <div class="grafico">\n'
        f'    <svg viewBox="0 0 {W} {H}" role="img" aria-label="Curva de Kaplan-Meier em degraus caindo de 1,000 a 0,360, com o fator de queda anotado em cada degrau">\n'
        + "      " + "\n      ".join(g) + "\n"
        "    </svg>\n"
        '    <table class="tabela">\n'
        "      <thead><tr><th>t<sub>i</sub></th><th>n<sub>i</sub> sob risco</th>"
        "<th>d<sub>i</sub> eventos</th><th>1 − d<sub>i</sub>/n<sub>i</sub></th>"
        "<th>Ŝ(t<sub>i</sub>)</th></tr></thead>\n"
        "      <tbody>\n" + "\n".join(linhas) + "\n      </tbody>\n"
        "    </table>\n"
        "  </div>\n"
        '  <div class="rodape">\n'
        + nota(
            "Em t = 5 dois contratos falham e a curva cai 25%. Em t = 7 um sai censurado: "
            "<strong>a curva não se mexe</strong>, mas o denominador do passo seguinte cai de 6 para 5."
        )
        + "  </div>"
    )
    return cena(css, corpo)


# ═══════════════════════════════════════════════════════════════════════════
# CENA 4 — hazard: a taxa instantanea, e a ponte S(t) = exp(-H(t))
# ═══════════════════════════════════════════════════════════════════════════
def cena_risco():
    W, H_SVG = 952, 442
    x0, x1 = 88.0, 910.0
    yA0, yA1 = 22.0, 176.0          # painel do hazard
    yB0, yB1 = 218.0, 372.0         # painel da sobrevivencia
    t_max, H_MAX = 36.0, 4.8
    fx = lambda t: x0 + t * (x1 - x0) / t_max
    fyA = lambda h: yA1 - (h / H_MAX) * (yA1 - yA0)
    fyB = lambda s: yB1 - s * (yB1 - yB0)

    # curva de maturacao: sobe, atinge o pico e cede (% ao mes)
    nos = [(0, 0.0), (2, 0.85), (4, 2.00), (6, 3.08), (8, 3.90), (10, 4.33),
           (11, 4.40), (12, 4.33), (14, 3.90), (16, 3.33), (18, 2.76),
           (20, 2.28), (24, 1.62), (28, 1.24), (32, 0.99), (36, 0.85)]
    T_PICO = 11.0

    def hz(t):
        if t <= nos[0][0]:
            return nos[0][1]
        for (ta, ha), (tb, hb) in zip(nos, nos[1:]):
            if t <= tb:
                return ha + (hb - ha) * (t - ta) / (tb - ta)
        return nos[-1][1]

    # S(t) = exp(-H(t)), H por trapezio sobre a MESMA h desenhada:
    # os dois paineis nao podem contar historias diferentes.
    PASSO = 0.05
    N = int(round(t_max / PASSO))
    grade = [i * PASSO for i in range(N + 1)]      # sem acumular erro de ponto flutuante
    acum, Hac = [0.0], 0.0
    for i in range(1, N + 1):
        Hac += (hz(grade[i]) + hz(grade[i - 1])) / 2.0 * PASSO / 100.0
        acum.append(Hac)
    surv = [math.exp(-a) for a in acum]
    por_mes = {k: surv[int(round(k / PASSO))] for k in range(int(t_max) + 1)}

    # a integral usa a grade fina; o desenho subamostra (0,25 mes ja e liso)
    RALO = 5
    idx = list(range(0, N + 1, RALO))
    if idx[-1] != N:
        idx.append(N)
    ph = [(fx(grade[i]), fyA(hz(grade[i]))) for i in idx]
    ps = [(fx(grade[i]), fyB(surv[i])) for i in idx]
    Lh, Ls = comprimento(ph), comprimento(ps)

    CICLO, VARRE = 8.0, 4.2
    p_fim = VARRE / CICLO * 100.0

    g = []
    # painel A — hazard
    g.append(f'<text class="painel" x="{n(x0)}" y="14">h(t) — taxa de risco instantânea (% ao mês)</text>')
    g.append(f'<rect class="campo" x="{n(x0)}" y="{n(yA0)}" width="{n(x1-x0)}" height="{n(yA1-yA0)}" rx="3"/>')
    for v in (0, 2, 4):
        g.append(f'<line class="grid" x1="{n(x0)}" y1="{n(fyA(v))}" x2="{n(x1)}" y2="{n(fyA(v))}"/>')
        g.append(f'<text class="tick" x="{n(x0-12)}" y="{n(fyA(v)+5)}" text-anchor="end">{v}</text>')
    g.append(f'<line class="ax" x1="{n(x0)}" y1="{n(yA1)}" x2="{n(x1)}" y2="{n(yA1)}"/>')
    g.append(f'<line class="ax" x1="{n(x0)}" y1="{n(yA0)}" x2="{n(x0)}" y2="{n(yA1)}"/>')
    g.append(f'<polygon class="area-h" points="{polilinha(ph + [(fx(t_max), yA1), (x0, yA1)])}"/>')
    g.append(f'<polyline class="curva ch" points="{polilinha(ph)}"/>')

    # painel B — sobrevivencia
    g.append(f'<text class="painel" x="{n(x0)}" y="{n(yB0-15)}">S(t) — proporção ainda sem evento</text>')
    g.append(f'<rect class="campo" x="{n(x0)}" y="{n(yB0)}" width="{n(x1-x0)}" height="{n(yB1-yB0)}" rx="3"/>')
    for v in (0, 0.5, 1.0):
        g.append(f'<line class="grid" x1="{n(x0)}" y1="{n(fyB(v))}" x2="{n(x1)}" y2="{n(fyB(v))}"/>')
        rot = "0" if v == 0 else ("1" if v == 1.0 else br(v, 1))
        g.append(f'<text class="tick" x="{n(x0-12)}" y="{n(fyB(v)+5)}" text-anchor="end">{rot}</text>')
    g.append(f'<line class="ax" x1="{n(x0)}" y1="{n(yB1)}" x2="{n(x1)}" y2="{n(yB1)}"/>')
    g.append(f'<line class="ax" x1="{n(x0)}" y1="{n(yB0)}" x2="{n(x0)}" y2="{n(yB1)}"/>')
    g.append(f'<polyline class="curva cs" points="{polilinha(ps)}"/>')

    for v in (0, 6, 12, 18, 24, 30, 36):
        g.append(f'<text class="tick" x="{n(fx(v))}" y="{n(yB1+26)}" text-anchor="middle">{v}</text>')
    g.append(f'<text class="axlab" x="{n(x1)}" y="{n(yB1+54)}" text-anchor="end">tempo desde a originação (meses)</text>')

    # anotacoes do pico — entram depois da varredura
    xp = fx(T_PICO)
    sp = surv[int(round(T_PICO / PASSO))]
    g.append('<g class="tardio">')
    g.append(f'  <line class="ref" x1="{n(xp)}" y1="{n(yA0)}" x2="{n(xp)}" y2="{n(yB1)}"/>')
    g.append(f'  <circle class="anel" cx="{n(xp)}" cy="{n(fyA(hz(T_PICO)))}" r="5.5" fill="{AMBER}"/>')
    g.append(f'  <circle class="anel" cx="{n(xp)}" cy="{n(fyB(sp))}" r="5.5" fill="{BLUE}"/>')
    g.append(f'  <text class="nota-painel" x="{n(x1)}" y="14" text-anchor="end">pico: {br(hz(T_PICO),1)}% no mês {int(T_PICO)}</text>')
    g.append(f'  <text class="nota-painel" x="{n(x1)}" y="{n(yB0-15)}" text-anchor="end">é onde S(t) cai mais rápido</text>')
    g.append(f'  <text class="nota-area" x="{n(fx(4.6))}" y="{n(yA1-14)}">área sombreada = H(t), o risco acumulado</text>')
    g.append("</g>")

    # sonda que varre os dois paineis ao mesmo tempo
    g.append('<g class="sonda">')
    g.append(f'  <line class="haste" x1="{n(x0)}" y1="{n(yA0)}" x2="{n(x0)}" y2="{n(yB1)}"/>')
    g.append(f'  <circle class="pt-h anel" cx="0" cy="0" r="6" fill="{AMBER}"/>')
    g.append(f'  <circle class="pt-s anel" cx="0" cy="0" r="6" fill="{BLUE}"/>')
    g.append("</g>")

    def quadros(fy_, val):
        saltos = []
        for k in range(0, 37):
            pct = k / 36.0 * p_fim
            saltos.append(f"      {n(pct,3)}% {{ transform: translate({n(fx(k))}px, {n(fy_(val(k)))}px); }}")
        saltos.append(f"      100% {{ transform: translate({n(fx(36))}px, {n(fy_(val(36)))}px); }}")
        return "\n".join(saltos)

    kf_h = quadros(fyA, lambda k: hz(float(k)))
    kf_s = quadros(fyB, lambda k: por_mes[k])
    pos_h = f"translate({n(fx(36))}px, {n(fyA(hz(36.0)))}px)"
    pos_s = f"translate({n(fx(36))}px, {n(fyB(por_mes[36]))}px)"

    css = CSS_COMUM + f"""
    .painel {{ font-family: "IBM Plex Sans", Helvetica, Arial, sans-serif; font-size: 16px;
               font-weight: 600; fill: #2D2D2D; }}
    .nota-painel {{ font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 15px;
                    fill: #6B7280; }}
    .nota-area {{ font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 14px;
                  fill: #6B7280; }}
    .ch {{ stroke: {AMBER}; stroke-dasharray: {n(Lh,1)}; stroke-dashoffset: 0;
           animation: tracaH {CICLO}s linear infinite both; }}
    .cs {{ stroke: {BLUE}; stroke-dasharray: {n(Ls,1)}; stroke-dashoffset: 0;
           animation: tracaS {CICLO}s linear infinite both; }}
    .area-h {{ fill: {AMBER}; opacity: 0.10; clip-path: inset(0 0 0 0);
               animation: revela {CICLO}s linear infinite both; }}
    .tardio {{ opacity: 1; animation: tardio {CICLO}s linear infinite both; }}
    .sonda {{ opacity: 0; animation: sonda {CICLO}s linear infinite both; }}
    .haste {{ stroke: #94A3B8; stroke-width: 1.4; stroke-dasharray: 3 4;
              transform: translateX({n(x1-x0)}px); animation: varre {CICLO}s linear infinite both; }}
    .pt-h {{ transform: {pos_h}; animation: correH {CICLO}s linear infinite both; }}
    .pt-s {{ transform: {pos_s}; animation: correS {CICLO}s linear infinite both; }}

    @keyframes tracaH {{ 0% {{ stroke-dashoffset: {n(Lh,1)}; }}
                         {n(p_fim,1)}% {{ stroke-dashoffset: 0; }} 100% {{ stroke-dashoffset: 0; }} }}
    @keyframes tracaS {{ 0% {{ stroke-dashoffset: {n(Ls,1)}; }}
                         {n(p_fim,1)}% {{ stroke-dashoffset: 0; }} 100% {{ stroke-dashoffset: 0; }} }}
    @keyframes revela {{ 0% {{ clip-path: inset(0 100% 0 0); }}
                         {n(p_fim,1)}% {{ clip-path: inset(0 0 0 0); }}
                         100% {{ clip-path: inset(0 0 0 0); }} }}
    @keyframes varre {{ 0% {{ transform: translateX(0); }}
                        {n(p_fim,1)}%, 100% {{ transform: translateX({n(x1-x0)}px); }} }}
    @keyframes sonda {{ 0% {{ opacity: 0; }} 3% {{ opacity: 1; }}
                        {n(p_fim,1)}% {{ opacity: 1; }}
                        {n(p_fim+8,1)}% {{ opacity: 0; }} 100% {{ opacity: 0; }} }}
    @keyframes tardio {{ 0%, {n(p_fim,1)}% {{ opacity: 0; }}
                         {n(p_fim+8,1)}% {{ opacity: 1; }} 100% {{ opacity: 1; }} }}
    @keyframes correH {{
{kf_h}
    }}
    @keyframes correS {{
{kf_s}
    }}
    .formula-dupla {{ display: flex; gap: 14px; flex-wrap: wrap; margin-top: 24px; }}
"""
    corpo = (
        '  <div class="cabeca">\n'
        + topo("A taxa por trás da curva", "cena 4 / 5")
        + "    <h1>Risco instantâneo, ou <em>hazard</em></h1>\n"
        '    <p class="deck">h(t) é a taxa de falha <em>entre quem chegou vivo até t</em>. '
        "O denominador são só os sobreviventes — por isso ela pode subir mesmo quando o número "
        "absoluto de eventos por mês já está caindo.</p>\n"
        '    <div class="formula-dupla">\n'
        '      <span class="formula"><em>h</em>(t) = P(evento em [t, t+Δt) | <em>T</em> ≥ t) / Δt'
        '<span class="sub">com Δt → 0</span></span>\n'
        '      <span class="formula"><em>S</em>(t) = exp(−∫₀<sup>t</sup> <em>h</em>(u) du)</span>\n'
        "    </div>\n"
        "  </div>\n"
        '  <div class="grafico">\n'
        f'    <svg viewBox="0 0 {W} {H_SVG}" role="img" aria-label="Dois painéis com o mesmo eixo de tempo: em cima a taxa de risco sobe até um pico no mês 11 e cede; embaixo a curva de sobrevivência cai mais rápido exatamente nesse pico">\n'
        + "      " + "\n      ".join(g) + "\n"
        "    </svg>\n"
        "  </div>\n"
        '  <div class="rodape">\n'
        + nota(
            "Em crédito isso tem nome: <strong>curva de maturação</strong>. O risco sobe, atinge o "
            f"pico por volta do mês {int(T_PICO)} e depois cede — por isso safra nova e safra madura "
            "só se comparam pela mesma idade."
        )
        + "  </div>"
    )
    return cena(css, corpo)


# ═══════════════════════════════════════════════════════════════════════════
# CENA 5 — Cox: riscos proporcionais e a ponte com a PD lifetime
# ═══════════════════════════════════════════════════════════════════════════
def cena_cox():
    W, H = 952, 400
    x0, x1, y0, y1 = 88.0, 812.0, 16.0, 316.0
    t_max, HR = 24.0, 2.1
    fx = lambda t: x0 + t * (x1 - x0) / t_max
    fy = lambda s: y1 - s * (y1 - y0)

    tempos = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    s_bom = [0.988, 0.968, 0.941, 0.910, 0.878, 0.847, 0.818, 0.792, 0.769, 0.749, 0.732, 0.718]
    # sob riscos proporcionais, S2(t) = S1(t)^HR — nao e um desenho solto
    s_mau = [v ** HR for v in s_bom]

    pb = [(fx(t), fy(s)) for t, s in passos_km(tempos, s_bom, t_max)]
    pm = [(fx(t), fy(s)) for t, s in passos_km(tempos, s_mau, t_max)]
    Lb, Lm = comprimento(pb), comprimento(pm)

    CICLO, DESENHO = 9.5, 4.8
    p_fim = DESENHO / CICLO * 100.0

    g = []
    g.append(eixos(x0, x1, y0, y1,
                   [0, 6, 12, 18, 24], [0, 0.25, 0.5, 0.75, 1.0],
                   lambda v: str(int(v)),
                   lambda v: br(v, 2).rstrip("0").rstrip(",") if v in (0, 1) else br(v, 2),
                   "tempo desde a originação (meses)", "S(t)", fx, fy))

    faixa = pb + list(reversed(pm))
    g.append(f'<polygon class="faixa-gap" points="{polilinha(faixa)}"/>')
    g.append(f'<polyline class="curva c-bom" points="{polilinha(pb)}"/>')
    g.append(f'<polyline class="curva c-mau" points="{polilinha(pm)}"/>')

    xf = fx(t_max)
    for s, cor, nome in ((s_bom[-1], BLUE, "Rating A–B"), (s_mau[-1], RED, "Rating D–E")):
        y = fy(s)
        g.append('<g class="tardio">')
        g.append(f'  <circle class="anel" cx="{n(xf)}" cy="{n(y)}" r="5.5" fill="{cor}"/>')
        g.append(f'  <text class="marcador" x="{n(xf+14)}" y="{n(y-2)}">{nome}</text>')
        g.append(f'  <text class="mini" x="{n(xf+14)}" y="{n(y+18)}">PD 24m = {br((1-s)*100,0)}%</text>')
        g.append("</g>")

    g.append('<g class="tardio">')
    g.append(f'  <rect x="120" y="190" width="372" height="100" rx="10" fill="#FFFFFF" stroke="{AXIS}"/>')
    g.append(f'  <text class="hr" x="142" y="228">HR = {br(HR,1)}</text>')
    g.append('  <text class="hr-txt" x="142" y="254">risco instantâneo 2,1× maior, em qualquer t</text>')
    g.append('  <text class="hr-sub" x="142" y="277">hipótese: essa razão é constante ao longo do tempo</text>')
    g.append("</g>")

    css = CSS_COMUM + f"""
    .c-bom {{ stroke: {BLUE}; stroke-dasharray: {n(Lb,1)}; stroke-dashoffset: 0;
              animation: tracaB {CICLO}s linear infinite both; }}
    .c-mau {{ stroke: {RED}; stroke-dasharray: {n(Lm,1)}; stroke-dashoffset: 0;
              animation: tracaM {CICLO}s linear infinite both; }}
    .faixa-gap {{ fill: {NEUTRAL}; opacity: 0.12; clip-path: inset(0 0 0 0);
                  animation: revela {CICLO}s linear infinite both; }}
    .tardio {{ opacity: 1; animation: tardio {CICLO}s linear infinite both; }}
    .mini {{ font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 15px;
             fill: #6B7280; font-variant-numeric: tabular-nums; }}
    .hr {{ font-family: "Space Grotesk", Helvetica, Arial, sans-serif; font-size: 30px;
           font-weight: 700; fill: #1E293B; }}
    .hr-txt {{ font-family: "IBM Plex Sans", Helvetica, Arial, sans-serif; font-size: 16px;
               fill: #2D2D2D; }}
    .hr-sub {{ font-family: "IBM Plex Sans", Helvetica, Arial, sans-serif; font-size: 14px;
               fill: #6B7280; }}
    @keyframes tracaB {{ 0% {{ stroke-dashoffset: {n(Lb,1)}; }}
                         {n(p_fim,1)}% {{ stroke-dashoffset: 0; }} 100% {{ stroke-dashoffset: 0; }} }}
    @keyframes tracaM {{ 0% {{ stroke-dashoffset: {n(Lm,1)}; }}
                         {n(p_fim,1)}% {{ stroke-dashoffset: 0; }} 100% {{ stroke-dashoffset: 0; }} }}
    @keyframes revela {{ 0% {{ clip-path: inset(0 100% 0 0); }}
                         {n(p_fim,1)}% {{ clip-path: inset(0 0 0 0); }}
                         100% {{ clip-path: inset(0 0 0 0); }} }}
    @keyframes tardio {{ 0%, {n(p_fim,1)}% {{ opacity: 0; }}
                         {n(p_fim+8,1)}% {{ opacity: 1; }} 100% {{ opacity: 1; }} }}
"""

    corpo = (
        '  <div class="cabeca">\n'
        + topo("Comparando grupos", "cena 5 / 5")
        + "    <h1>Cox e os riscos proporcionais</h1>\n"
        '    <p class="deck">O modelo de Cox estima o <em>efeito</em> das covariáveis sobre o risco '
        "sem precisar assumir a forma de h<sub>0</sub>(t). O que sai dele é uma razão: quanto o "
        "risco de um grupo é maior que o do outro, a qualquer instante.</p>\n"
        '    <div class="chip-linha" style="display:flex; margin-top:24px">\n'
        '      <span class="formula"><em>h</em>(t | x) = <em>h</em><sub>0</sub>(t) · exp(β′x)'
        '<span class="sub">HR = exp(β)</span></span>\n'
        "    </div>\n"
        "  </div>\n"
        '  <div class="grafico">\n'
        f'    <svg viewBox="0 0 {W} {H}" role="img" aria-label="Duas curvas de sobrevivência que se afastam ao longo de 24 meses: rating A–B termina em 0,72 e rating D–E em 0,50, com razão de risco 2,1">\n'
        + "      " + "\n      ".join(g) + "\n"
        "    </svg>\n"
        "  </div>\n"
        '  <div class="rodape">\n'
        '    <div class="legenda">\n'
        '      <span class="item"><svg class="chave" viewBox="0 0 22 22" aria-hidden="true">'
        f'<line x1="1" y1="11" x2="21" y2="11" stroke="{BLUE}" stroke-width="3" stroke-linecap="round"/></svg>'
        " Rating A–B (menor risco)</span>\n"
        '      <span class="item"><svg class="chave" viewBox="0 0 22 22" aria-hidden="true">'
        f'<line x1="1" y1="11" x2="21" y2="11" stroke="{RED}" stroke-width="3" stroke-linecap="round"/></svg>'
        " Rating D–E (maior risco)</span>\n"
        "    </div>\n"
        + nota(
            "<strong>1 − S(t) é a PD acumulada até t.</strong> A curva não entrega só a PD de 12 "
            "meses: entrega a estrutura a termo inteira — que é exatamente o que a ECL lifetime pede."
        )
        + "  </div>"
    )
    return cena(css, corpo)


# ═══════════════════════════════════════════════════════════════════════════
CANVAS = """{
  "artboards": [
    { "file": "Main.dc.html",        "x": 0,    "y": 0, "w": 1080, "h": 1080, "print": "fixed", "title": "1 · Análise de sobrevivência" },
    { "file": "Censura.dc.html",     "x": 1200, "y": 0, "w": 1080, "h": 1080, "print": "fixed", "title": "2 · Censura à direita" },
    { "file": "KaplanMeier.dc.html", "x": 2400, "y": 0, "w": 1080, "h": 1080, "print": "fixed", "title": "3 · Kaplan-Meier" },
    { "file": "Risco.dc.html",       "x": 3600, "y": 0, "w": 1080, "h": 1080, "print": "fixed", "title": "4 · Risco instantâneo" },
    { "file": "Cox.dc.html",         "x": 4800, "y": 0, "w": 1080, "h": 1080, "print": "fixed", "title": "5 · Cox" }
  ],
  "annotations": [
    { "id": "roteiro", "x": -520, "y": 60, "w": 400, "text": "Roteiro em 5 cenas, da esquerda para a direita.\\n\\nCada quadro é uma cena que roda em loop.\\n\\nCores e chrome dos eixos seguem o tema visual do Yggdrasil-Project (skills viz e video-manim)." }
  ],
  "launch": { "view": "focused", "file": "Main.dc.html" }
}
"""


def main():
    saidas = {
        "Main.dc.html": cena_main(),
        "Censura.dc.html": cena_censura(),
        "KaplanMeier.dc.html": cena_kaplan_meier(),
        "Risco.dc.html": cena_risco(),
        "Cox.dc.html": cena_cox(),
        "canvas.json": CANVAS,
    }
    for nome, conteudo in saidas.items():
        caminho = os.path.join(AQUI, nome)
        with open(caminho, "w", encoding="utf-8") as fh:
            fh.write(conteudo)
        print(f"  {nome:22s} {len(conteudo):>7,} bytes")


if __name__ == "__main__":
    main()
