"""
SatelliteUI
===========
Camada interativa (``ipywidgets``) sobre os **modelos satélite** deste subpacote —
a interface que faltava para o eixo **temporal** do risco de crédito, no mesmo
estilo do :class:`~yggdrasil.credit_risk.model.ModelSegmenterUI` (abas, console,
tema claro/escuro) mas para um fluxo de **série temporal + macro**:

* **Série** — o cartão de entrada: o que é a série (parâmetro, segmento,
  frequência, período, nº de observações), a série no nível e na escala do *link*,
  as macros disponíveis e o **relatório de estacionariedade** (ADF/KPSS/PP +
  ordem de integração) do alvo e de cada macro, com a leitura em texto;
* **Especificação** — a escolha do modelo (com uma linha de ajuda dizendo *quando*
  usar cada um), a **matriz de sinais econômicos esperados**, a grade de
  defasagens/ordens AR/sazonalidade e os parâmetros de seleção; o botão
  **Ajustar agora** ajusta UM modelo com a especificação corrente e mostra os
  coeficientes com p-valor e AIC/BIC — feedback rápido antes de gastar uma busca;
* **Seleção** — a busca *champion-challenger* sobre a grade: o **tamanho da grade**
  antes de rodar (a busca é cara), o progresso por etapa com o tempo decorrido, o
  **ranking** das especificações qualificadas, a lista das **descartadas com o
  motivo** (sinal invertido, VIF alto), a escolha **manual** de uma especificação
  do ranking — champion-challenger de verdade exige poder discordar do critério —
  e a comparação contra os benchmarks ingênuos com o teste de Diebold-Mariano;
* **Diagnóstico** — a bateria de :mod:`.diagnostics` sobre o modelo vigente num
  **placar por família** (resíduo, heterocedasticidade, normalidade, estabilidade,
  colinearidade), a tabela completa com p-valores, os gráficos de ajuste e de
  resíduos e a leitura em texto do **que fazer** quando um teste falha;
* **Cenários & Projeção** — o **fator prospectivo** propriamente dito: os cenários
  por três caminhos (padrão de um clique, choque parametrizado com persistência e
  **colagem** da trajetória que veio da área econômica), a projeção condicional em
  **leque** com o futuro sombreado, a **projeção ponderada** pelos pesos — a curva
  única que segue adiante — e a exportação em formato longo;
* **Backtest** — a outra metade da projeção: reestimação **janela a janela**, erro
  por horizonte e **cobertura empírica** dos intervalos contra a nominal, com
  Kupiec (proporção de violações) e Christoffersen (independência), lidos em texto;
* **Exportar** — o fechamento do estudo: o **relatório HTML** de governança, o
  registro do *run* no **MLflow** (com o que houver no estado: ajuste, busca,
  projeção e backtest), a configuração da sessão em **JSON** — exibir, salvar e
  **carregar**, para que a configuração que gerou a projeção viaje junto com ela —
  e as tabelas (projeção, ranking, diagnóstico, cobertura, coeficientes) em CSV.

Quem já sabe o que quer não precisa percorrer as abas: o botão **Rodar estudo
completo**, na aba *Especificação*, monta a configuração da tela e encadeia
seleção → ajuste → diagnóstico → cenários → projeção numa passada só,
preenchendo todas as abas com o resultado. As abas continuam servindo para
conduzir o estudo passo a passo — e para discordar de qualquer etapa.

A configuração corrente é materializável como
:class:`~yggdrasil.credit_risk.econometric.config.StudyConfig` (:meth:`to_config`)
e restaurável (:meth:`from_config`), de modo que a interface e o pipeline
declarativo falem a **mesma língua**.

Sem dados em mãos? Construa sem argumentos e clique em **Carregar estudo de
referência**: a interface gera um segmento sintético completo (PD, LGD e CCF sobre
a mesma macro, com recessão e evento) para explorar a ferramenta na hora.

    from yggdrasil.credit_risk.econometric.ui import SatelliteUI
    ui = SatelliteUI()                                   # estudo de referência
    ui = SatelliteUI(rs, macro, kind="pd")               # RiskSeries + macro
    ui = SatelliteUI(serie, macro, kind="lgd")           # pandas.Series + macro
    ui

``ipywidgets``, ``statsmodels``, ``arch`` e ``matplotlib`` são **opcionais**: o
import deste módulo não os exige (são carregados sob demanda, com mensagem clara
quando faltarem).
"""
from __future__ import annotations

import math
import re
import unicodedata
from contextlib import contextmanager, suppress
from typing import Mapping, Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Dependências opcionais (import tardio)
# ----------------------------------------------------------------------
W = None            # ipywidgets
_clear_output = None
_display = None


def _require_widgets() -> None:
    """Importa ``ipywidgets``/``IPython`` na **primeira** construção da interface.

    Mantém o import deste módulo livre de dependências opcionais (o pacote é
    importável sem Jupyter) e dá uma mensagem clara quando faltam.
    """
    global W, _clear_output, _display
    if W is not None:
        return
    try:
        import ipywidgets as _W
        from IPython.display import clear_output as _co
        from IPython.display import display as _di
    except Exception as exc:  # pragma: no cover - depende do ambiente
        raise ImportError(
            "SatelliteUI requer ipywidgets e IPython (Jupyter). "
            "Instale com: pip install ipywidgets  (e o extra 'econometric' do "
            "yggdrasil para statsmodels/arch)."
        ) from exc
    W, _clear_output, _display = _W, _co, _di


class _LazyModules:
    """Acesso preguiçoso aos módulos do subpacote.

    ``_E.diagnostics`` importa :mod:`statsmodels` só quando o usuário roda um
    teste — construir a interface (ou importar este módulo) não exige o extra.
    """

    def __getattr__(self, name):
        import importlib

        mod = importlib.import_module(f".{name}", __package__)
        setattr(self, name, mod)
        return mod


_E = _LazyModules()


# ======================================================================
# Estilo (mesmos tokens semânticos das demais UIs do credit_risk)
# ======================================================================
_CSS = """
<style>
/* SEM @import externo de fontes: no Databricks o cluster costuma não ter egress
   e um @import render-blocking adia o 1º paint. Usa-se a font-stack do sistema. */
.satui { --ac:#3b4a63; --ac-deep:#27324a; --ac-soft:#eef1f5; --ac-border:#cdd5e0;
  --ink:#1f2733; --muted:#6b7480; --line:#e7e9ee;
  /* tokens semânticos (status, tabelas, realces): o HTML gerado no Python usa
     var(--...) em vez de hex — o tema escuro só redefine os tokens aqui */
  --ok-ink:#157a52; --ok-bg:#e7f5ee; --ok-tx:#137a3e;
  --warn-ink:#9a6f12; --warn-bg:#fbf3e0; --warn-tx:#9a6b00;
  --bad-ink:#b23a2a; --bad-bg:#fbe7e4; --bad-tx:#b3261e;
  --info-ink:#1f5fa8; --info-bg:#e7eef8; --sus-ink:#6b46c1; --sus-bg:#efe9fb;
  --code-ink:#0b63ce; --code-bg:#e7eef8;
  --strong-ink:#15324a; --body-ink:#3a4658; --sub-ink:#8a93a3;
  --faint-ink:#9aa3ad; --hair:#eef0f3; --tile-bg:#f7f8fa; --rule-bg:#fff;
  --help-bg:#f5f8fc; --help-line:#dbe5f1; --neutral-bg:#f1f3f5;
  --tbl-line:#e1e5ec; --tbl-line-strong:#cdd5e0; --tbl-head-bg:#eef1f5;
  --tbl-head-ink:#27324a; --tbl-head-line:#b9c2d0; --tbl-zebra:#fafbfc;
  --tbl-hover:#eef3f8;
  --notice-bg:#fff8e6; --notice-border:#f0c36d; --notice-ink:#664d03;
  font-family:'IBM Plex Sans', -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  color:var(--ink); }
.satui .mono { font-family:'IBM Plex Mono', ui-monospace, Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums; }
.satui-banner { display:flex; align-items:center; gap:11px; background:#fff;
  border:1px solid var(--line); border-radius:13px; padding:11px 16px; margin-bottom:10px;
  box-shadow:0 1px 3px rgba(16,24,40,.08); }
.satui-banner .logo { width:30px; height:30px; border-radius:9px; background:var(--ac);
  color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700;
  font-size:12px; flex:none; }
.satui-banner .t { font-size:15px; font-weight:600; color:var(--ink); line-height:1.2; }
.satui-banner .s { font-size:11.5px; color:var(--muted); margin-top:1px; }
.satui-card { background:#fff; border:1px solid var(--line); border-radius:12px;
  padding:13px 15px; box-shadow:0 1px 3px rgba(16,24,40,.06); margin-bottom:11px;
  overflow-x:clip; }
.satui-h { font-weight:600; font-size:11px; color:var(--muted); text-transform:uppercase;
  letter-spacing:.07em; margin-bottom:9px; }
.satui-bar { background:#fff; border:1px solid var(--line); border-radius:11px;
  box-shadow:0 1px 3px rgba(16,24,40,.05); padding:8px 12px; overflow-x:auto; }
.pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:11.5px;
  font-weight:600; margin:2px 4px 2px 0; }
.pill-muted  { background:var(--ac-soft); color:var(--ac-deep); }
.pill-green  { background:var(--ok-bg); color:var(--ok-ink); }
.pill-yellow { background:var(--warn-bg); color:var(--warn-ink); }
.pill-red    { background:var(--bad-bg); color:var(--bad-ink); }
.satui-legend { font-size:11px; color:var(--muted); margin:6px 0 2px; line-height:1.55; }
.satui-help { background:var(--help-bg); border:1px solid var(--help-line);
  border-left:3px solid var(--ac); border-radius:9px; padding:11px 14px; font-size:11.5px;
  color:var(--body-ink); line-height:1.62; margin-top:8px; }
.satui-help .ttl { font-size:12px; font-weight:600; color:var(--ink); margin-bottom:5px; }
.satui-help ul { margin:5px 0 0; padding-left:6px; list-style:none; }
.satui-help li { margin:4px 0; }
.satui-help code, .satui-help .pname { font-family:'IBM Plex Mono', ui-monospace, monospace;
  font-weight:600; color:var(--code-ink); background:var(--code-bg); padding:1px 6px;
  border-radius:5px; }
.satui-notice { border:1px solid var(--notice-border); background:var(--notice-bg);
  border-radius:10px; padding:9px 12px; font-size:12px; color:var(--notice-ink);
  margin-bottom:8px; }
.satui-guide { margin-top:10px; font-size:11.5px; }
.satui-guide > summary { cursor:pointer; font-weight:600; color:var(--ac-deep);
  padding:8px 12px; background:var(--ac-soft); border:1px solid var(--ac-border);
  border-radius:8px; list-style:none; user-select:none; }
.satui-guide > summary::-webkit-details-marker { display:none; }
.satui-guide > summary::before { content:'▸ '; }
.satui-guide[open] > summary::before { content:'▾ '; }
.satui-metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(104px,1fr)); gap:6px; }
.satui-metric { background:var(--tile-bg); border:1px solid var(--hair); border-radius:9px;
  padding:7px 10px; }
.satui-metric .k { font-size:10px; text-transform:uppercase; letter-spacing:.04em;
  color:var(--sub-ink); }
.satui-metric .v { font-size:16px; font-weight:600; color:var(--ink); margin-top:2px;
  font-variant-numeric: tabular-nums; }
/* placar de vereditos (aba Diagnóstico): um bloco por família de teste */
.satui-placar { display:grid; grid-template-columns:repeat(auto-fit,minmax(196px,1fr)); gap:8px; }
.satui-bloco { background:var(--tile-bg); border:1px solid var(--hair); border-radius:10px;
  padding:8px 11px; border-left-width:3px; border-left-style:solid;
  border-left-color:var(--faint-ink); }
.satui-bloco.ok   { border-left-color:var(--ok-ink); }
.satui-bloco.warn { border-left-color:var(--warn-ink); }
.satui-bloco.bad  { border-left-color:var(--bad-ink); }
.satui-bloco .k { font-size:10px; text-transform:uppercase; letter-spacing:.04em;
  color:var(--sub-ink); }
.satui-bloco .v { font-size:13px; font-weight:600; margin:3px 0 2px; }
.satui-bloco .d { font-size:11px; color:var(--muted); line-height:1.5;
  font-variant-numeric: tabular-nums; }
.satui-prog { border-collapse:collapse; font-size:12px; width:100%; margin:2px 0 8px; }
.satui-prog th { padding:4px 10px; text-align:left; background:var(--tbl-head-bg);
  color:var(--tbl-head-ink); font-weight:600; }
.satui-prog td { padding:4px 10px; border-top:1px solid var(--tbl-line); }
.satui-signrow { font-size:12px; color:var(--body-ink); }
.satui-signhead { font-size:11px; color:var(--sub-ink); letter-spacing:.03em;
  margin:2px 0 6px; line-height:1.6; }
/* abas — "segmented control" (pílulas), igual às demais UIs do credit_risk */
.satui-tabs { margin-top:10px; border:none !important; box-shadow:none !important; }
.satui-tabs > .widget-tab-contents { padding:30px 2px 2px !important; background:transparent;
  border:none !important; box-shadow:none !important; }
.satui-tabs .lm-TabBar.jupyter-widget-tab-nav,
.satui-tabs .p-TabBar.jupyter-widget-tab-nav { border-bottom:1px solid var(--line) !important;
  padding-bottom:14px !important; margin-bottom:0 !important; box-shadow:none !important; }
.satui-tabs .lm-TabBar-content, .satui-tabs .p-TabBar-content { gap:7px;
  align-items:stretch; border:none; }
.satui-tabs .lm-TabBar-tab, .satui-tabs .p-TabBar-tab { font-size:13px;
  min-width:max-content !important; max-width:none !important; flex:0 0 auto !important;
  margin:0 !important; padding:8px 16px !important;
  border:1px solid var(--line) !important; border-radius:9px !important;
  background:#fff !important; color:var(--muted) !important; font-weight:500;
  line-height:1.15; outline:none !important; box-shadow:none !important;
  transition:background .15s, color .15s, border-color .15s; }
.satui-tabs .lm-TabBar-tab::before, .satui-tabs .lm-TabBar-tab::after,
.satui-tabs .p-TabBar-tab::before, .satui-tabs .p-TabBar-tab::after {
  display:none !important; content:none !important; background:none !important; }
.satui-tabs .lm-TabBar-tab:hover, .satui-tabs .p-TabBar-tab:hover {
  background:var(--ac-soft) !important; color:var(--ac-deep) !important;
  border-color:var(--ac-border) !important; }
.satui-tabs .lm-TabBar-tabLabel, .satui-tabs .p-TabBar-tabLabel {
  white-space:nowrap !important; overflow:visible !important;
  text-overflow:clip !important; max-width:none !important; }
.satui-tabs .lm-TabBar-tab.lm-mod-current,
.satui-tabs .p-TabBar-tab.p-mod-current { color:#fff !important; font-weight:600;
  background:var(--ac) !important; border:1px solid var(--ac) !important;
  outline:none !important; box-shadow:none !important; }
.satui-tabs .lm-TabBar-tab.lm-mod-current:hover,
.satui-tabs .p-TabBar-tab.p-mod-current:hover {
  background:var(--ac-deep) !important; color:#fff !important;
  border-color:var(--ac-deep) !important; }
.satui .jupyter-button { border-radius:8px; font-family:inherit; }
.satui .jupyter-widgets { min-width:0 !important; }
/* ===== TEMA ESCURO (classe .dark no painel raiz) =====
   Paleta alinhada ao dark mode do Databricks (design system DuBois). */
.satui.dark { --ink:#E8ECF0; --muted:#92A4B3; --line:#37444F; --ac-soft:#37444F;
  --ac-border:#5F7281; --ac-deep:#E8ECF0; --ac:#4299E0;
  --ok-ink:#3BA65E; --ok-bg:rgba(39,124,67,.16); --ok-tx:#3BA65E;
  --warn-ink:#DE7921; --warn-bg:rgba(190,80,30,.16); --warn-tx:#DE7921;
  --bad-ink:#E65B77; --bad-bg:rgba(200,45,76,.16); --bad-tx:#E65B77;
  --info-ink:#8ACAFF; --info-bg:rgba(138,202,255,.16);
  --sus-ink:#B592E5; --sus-bg:rgba(138,99,191,.24);
  --code-ink:#8ACAFF; --code-bg:rgba(138,202,255,.16);
  --strong-ink:#E8ECF0; --body-ink:#C0CDD8; --sub-ink:#8396A5;
  --faint-ink:#5F7281; --hair:#37444F; --tile-bg:#11171C; --rule-bg:#11171C;
  --help-bg:#11171C; --help-line:#37444F; --neutral-bg:rgba(144,164,181,.16);
  --tbl-line:#37444F; --tbl-line-strong:#445461; --tbl-head-bg:#11171C;
  --tbl-head-ink:#E8ECF0; --tbl-head-line:#445461; --tbl-zebra:rgba(189,205,219,.04);
  --tbl-hover:rgba(189,205,219,.08);
  --notice-bg:rgba(190,80,30,.16); --notice-border:#DE7921; --notice-ink:#E8ECF0;
  background:#11171C; padding:8px; border-radius:12px; }
.satui.dark .satui-banner, .satui.dark .satui-card, .satui.dark .satui-bar {
  background:#1F272D !important; border-color:#37444F !important; box-shadow:none !important; }
.satui.dark .satui-banner .t { color:#E8ECF0; }
.satui.dark .satui-banner .logo { color:#11171C; }
.satui.dark .satui-guide > summary:hover { background:#445461; }
.satui.dark .satui-tabs .p-TabBar-tab, .satui.dark .satui-tabs .lm-TabBar-tab {
  background:#1F272D !important; color:#92A4B3 !important; border-color:#37444F !important; }
.satui.dark .satui-tabs .p-TabBar-tab:hover,
.satui.dark .satui-tabs .lm-TabBar-tab:hover { background:rgba(138,202,255,.08) !important;
  color:#8ACAFF !important; border-color:#8ACAFF !important; }
.satui.dark .satui-tabs .p-TabBar-tab.p-mod-current,
.satui.dark .satui-tabs .lm-TabBar-tab.lm-mod-current { background:#4299E0 !important;
  color:#11171C !important; border-color:#4299E0 !important; }
.satui.dark .satui-tabs .p-TabBar-tab.p-mod-current:hover,
.satui.dark .satui-tabs .lm-TabBar-tab.lm-mod-current:hover {
  background:#8ACAFF !important; color:#11171C !important; border-color:#8ACAFF !important; }
.satui.dark .widget-text input, .satui.dark .widget-dropdown select, .satui.dark textarea {
  background:#11171C !important; color:#E8ECF0 !important; border-color:#37444F !important; }
.satui.dark .widget-label, .satui.dark .jupyter-widgets label { color:#D1D9E1 !important; }
.satui.dark .jupyter-button:not(.mod-primary):not(.mod-success):not(.mod-info):not(.mod-warning):not(.mod-danger) { background:#37444F !important; color:#E8ECF0 !important; }
.satui.dark .jupyter-button.mod-active { background:#4299E0 !important; color:#11171C !important; }
</style>
"""


# ======================================================================
# Catálogo de modelos da interface
# ======================================================================
#: Modelos oferecidos na aba **Especificação**. ``registry`` é o nome do modelo em
#: :data:`~yggdrasil.credit_risk.econometric.config.MODEL_REGISTRY` (``None`` = o
#: modelo não participa da busca champion-challenger como candidato, só como
#: *benchmark*); ``campos`` diz quais controles específicos aparecem na tela.
MODELOS: dict[str, dict] = {
    "ardl": dict(
        rotulo="ARDL — defasagens distribuídas (padrão)",
        registry="ardl", kinds=("pd", "lgd", "ccf"), campos=("link", "trend", "cov"),
        ajuda="O cavalo de batalha. Regride o parâmetro transformado contra as próprias "
              "defasagens (a inércia) e as macro defasadas. <b>Use como modelo principal</b> "
              "na maioria dos segmentos: interpretável, parcimonioso e projeta bem quando "
              "condicionado a cenário.",
    ),
    "vasicek": dict(
        rotulo="Fator Z (Vasicek) — ponte com o capital",
        registry="vasicek", kinds=("pd",), campos=("trend", "cov", "vasicek"),
        ajuda="Extrai da série o <b>fator sistêmico Z</b> (aproximadamente normal e "
              "estacionário por construção) e o modela contra a macro. <b>Use quando quiser "
              "coerência com o motor de capital</b> (o mesmo ρ) e tratamento natural do "
              "piso/teto da taxa. Exige ρ e o nível de longo prazo bem estimados. "
              "Só para séries de taxa de default.",
    ),
    "beta": dict(
        rotulo="Regressão beta — frações (0,1)",
        registry="beta", kinds=("lgd", "ccf", "pd"), campos=("trend",),
        ajuda="Respeita o suporte (0, 1) <b>nativamente</b> e modela média e dispersão. "
              "<b>Use para severidade/fator de conversão</b>, sobretudo quando a série passa "
              "perto das bordas. Não aceita 0 nem 1 exatos.",
    ),
    "fractional": dict(
        rotulo="Fractional logit — aceita 0 e 1 exatos",
        registry="fractional", kinds=("lgd", "ccf", "pd"), campos=("trend",),
        ajuda="Quase-verossimilhança (Papke &amp; Wooldridge) com erros-padrão robustos. "
              "<b>Use quando a série tem massa exatamente em 0 ou 1</b> (cura total, perda "
              "total) ou quando você quer robustez a má especificação da distribuição.",
    ),
    "arima": dict(
        rotulo="ARIMA / ARIMAX — o benchmark obrigatório",
        registry=None, kinds=("pd", "lgd", "ccf"), campos=("link", "arima"),
        ajuda="ARIMA sobre a série transformada; com variáveis candidatas marcadas vira "
              "ARIMAX. <b>Use como régua</b>: um modelo macro que não supera o ARIMA fora "
              "da amostra ainda não está pronto.",
    ),
    "random_walk": dict(
        rotulo="Passeio aleatório — piso ingênuo",
        registry=None, kinds=("pd", "lgd", "ccf"), campos=(),
        ajuda="A projeção é o <b>último valor observado</b>. Referência ingênua: se o modelo "
              "macro não bate isto fora da amostra, não há sinal macro sendo capturado.",
    ),
    "media_historica": dict(
        rotulo="Média histórica — âncora de reversão",
        registry=None, kinds=("pd", "lgd", "ccf"), campos=(),
        ajuda="A projeção é a <b>média da série</b>. Piso que mede o quanto o ciclo "
              "realmente acrescenta sobre o nível de longo prazo.",
    ),
    "sazonal_ingenuo": dict(
        rotulo="Sazonal ingênuo — repete a estação anterior",
        registry=None, kinds=("pd", "lgd", "ccf"), campos=("periodo",),
        ajuda="Repete o valor da <b>mesma estação do ciclo anterior</b>. Piso útil em séries "
              "mensais com sazonalidade forte (safras de varejo, 13º).",
    ),
}

#: Sinal econômico sugerido por família de variável (heurística do botão
#: *Sugerir sinais*): variáveis de **deterioração** do ciclo empurram o parâmetro
#: de risco para cima (+1), variáveis de **atividade/renda** para baixo (−1).
#: Vale igual para taxa de default, severidade e fator de conversão.
_SINAL_SUGERIDO = {
    "desemprego": 1, "inadimplencia": 1, "juros": 1, "selic": 1, "cdi": 1,
    "inflacao": 1, "ipca": 1, "endividamento": 1, "comprometimento": 1,
    "renda": -1, "pib": -1, "emprego": -1, "massa": -1, "salario": -1,
    "confianca": -1, "atividade": -1, "consumo": -1,
}


def _sem_acento(txt: str) -> str:
    """Minúsculas sem acento (para casar nomes de variável na heurística)."""
    n = unicodedata.normalize("NFKD", str(txt))
    return "".join(c for c in n if not unicodedata.combining(c)).lower()


#: O que fazer quando um bloco da bateria de diagnóstico falha — ``(rótulo,
#: testes da família, conselho)``. Curto e acionável: a aba **Diagnóstico** só
#: mostra as famílias que efetivamente reprovaram.
_CONSELHO_DIAG = (
    ("Autocorrelação residual", ("Ljung-Box", "Breusch-Godfrey", "Durbin-Watson"),
     "sobrou dinâmica que o modelo não capturou. <b>Aumente a ordem AR</b> (1 → 2) ou "
     "<b>inclua outra defasagem da macro</b>; se persistir, estime com covariância "
     "<b>HAC (Newey-West)</b> — os p-valores atuais estão otimistas e os intervalos, "
     "estreitos demais."),
    ("Heterocedasticidade", ("Breusch-Pagan", "White"),
     "a variância do erro muda com o nível (típico: o erro cresce na recessão). O "
     "coeficiente segue não-viesado, o <b>erro-padrão</b> é que fica errado — troque a "
     "covariância para <b>HAC</b> ou modele na escala do <i>link</i>, que estabiliza a "
     "variância perto das bordas."),
    ("Volatilidade condicional (ARCH)", ("ARCH-LM",),
     "a volatilidade vem em <i>clusters</i>. Os pontos projetados continuam válidos, mas "
     "a <b>banda</b> subestima a incerteza justamente no estresse — leia o cenário "
     "adverso como a cauda, não a banda do cenário base."),
    ("Normalidade", ("Jarque-Bera",),
     "resíduo não normal pesa sobre o <b>intervalo</b>, não sobre a média. Prefira bandas "
     "por <b>reamostragem dos próprios resíduos</b> (é o que o motor de projeção faz) e "
     "investigue outliers: um evento isolado pede uma <i>dummy</i>, não um modelo novo."),
    ("Estabilidade", ("Chow", "Quandt-Andrews sup-F", "CUSUM"),
     "o alerta mais sério: os coeficientes <b>mudaram no meio da amostra</b> e a projeção "
     "extrapola uma relação que já não vale. Inclua uma <i>dummy</i> de quebra/evento, "
     "encurte a amostra para o regime atual ou reestime por regime — nunca ignore."),
    ("Colinearidade", ("VIF",),
     "duas macro disputam o mesmo papel: os coeficientes ficam instáveis e podem sair com "
     "o <b>sinal trocado</b> sem que a relação econômica tenha mudado. <b>Remova uma</b> "
     "das redundantes (ou combine-as num índice) e reajuste."),
)


class SatelliteUI:
    """Interface de treinamento de modelos satélite (fatores prospectivos).

    Parameters
    ----------
    series:
        A série do parâmetro de risco. Aceita
        :class:`~yggdrasil.credit_risk.econometric.series.RiskSeries`,
        :class:`~yggdrasil.credit_risk.econometric.series.SyntheticSeries` (usa a
        série e a macro embutidas), uma ``pandas.Series`` de taxa em ``[0, 1]``
        indexada no tempo, ou ``None`` — neste caso a aba **Série** oferece o botão
        *Carregar estudo de referência*.
    macro:
        ``pandas.DataFrame`` das variáveis macro, com o **mesmo índice temporal**
        da série (pode ter períodos a mais; não pode faltar período da série).
    kind:
        ``'pd'``, ``'lgd'`` ou ``'ccf'`` — usado quando ``series`` é uma
        ``pandas.Series`` crua (com :class:`RiskSeries` vale o ``kind`` dela).
    segment, frequency:
        Rótulo do segmento e frequência pandas (``"MS"``, ``"QS"``); a frequência
        é inferida do índice quando omitida.
    candidates:
        Variáveis macro pré-marcadas como candidatas (padrão: todas as numéricas).
    expected_signs:
        Sinais econômicos esperados iniciais ``{variavel: +1|-1}``.
    problem_label:
        Rótulo do parâmetro nos títulos (padrão: o ``kind`` em maiúsculas) — a
        interface serve PD, LGD e CCF sem fixar nenhum na tela.
    name:
        Nome do estudo (vai para :class:`StudyConfig` e para os relatórios).

    Attributes
    ----------
    series, macro, kind:
        Os dados correntes (``RiskSeries``, ``DataFrame`` e o tipo de parâmetro).
    fit_, model_:
        O último :class:`FitResult` e o modelo que o produziu (aba Especificação).
    search_:
        O último :class:`SearchResult` (aba Seleção).
    selected_spec_:
        A :class:`Specification` **adotada** — a campeã da busca ou a que o usuário
        escolheu manualmente no ranking (aba Seleção).
    compare_:
        A tabela de comparação da campeã contra os benchmarks ingênuos/ARIMA, com o
        teste de Diebold-Mariano (aba Seleção).
    diagnostics_, diag_blocks_, vif_:
        A tabela da última bateria de diagnóstico, o placar por família de teste
        (lista de dicionários com veredito e evidência) e a tabela de VIF por
        variável (aba Diagnóstico).
    scenarios_, projection_:
        O último :class:`ScenarioSet` (as trajetórias macro futuras) e a
        :class:`Projection` condicional a elas (aba Cenários).
    weighted_, projection_table_:
        A projeção **ponderada** pelos pesos dos cenários (``pandas.Series``) e a
        projeção em formato longo da última exportação — veja
        :meth:`projection_frame` (aba Cenários).
    backtest_, coverage_:
        O dicionário do último backtest de projeção (métricas, ``bands`` por janela
        e ``coverage``) e a tabela de **cobertura dos intervalos** por horizonte,
        com Kupiec e Christoffersen (aba Backtest).
    study_:
        O último :class:`StudyResult` do botão **Rodar estudo completo** (aba
        Especificação) — a configuração, a busca, o ajuste, o diagnóstico, os
        cenários, a projeção e o HTML do relatório numa estrutura só.
    """

    #: estilo das tabelas (Styler) — cores por token, resolvidas no tema ativo
    _TABLE_STYLES = [
        {"selector": "", "props": [("border-collapse", "collapse"),
                                   ("border", "1px solid var(--tbl-line-strong)"),
                                   ("width", "100%")]},
        {"selector": "th, td", "props": [("border", "1px solid var(--tbl-line)"),
                                         ("padding", "4px 9px"), ("text-align", "right"),
                                         ("white-space", "nowrap")]},
        {"selector": "thead th", "props": [("background-color", "var(--tbl-head-bg)"),
                                           ("color", "var(--tbl-head-ink)"),
                                           ("font-weight", "600"),
                                           ("border-bottom", "2px solid var(--tbl-head-line)"),
                                           ("position", "sticky"), ("top", "0"), ("z-index", "1")]},
        {"selector": "tbody tr:nth-child(even) td",
         "props": [("background-color", "var(--tbl-zebra)")]},
        {"selector": "tbody tr:hover td", "props": [("background-color", "var(--tbl-hover)")]},
    ]

    #: as abas, na ordem do fluxo (dados → especificação → seleção → crítica →
    #: projeção → validação da banda → fechamento)
    ABAS = ("Série", "Especificação", "Seleção", "Diagnóstico",
            "Cenários & Projeção", "Backtest", "Exportar")

    # ==================================================================
    # Construção
    # ==================================================================
    def __init__(self, series=None, macro=None, kind="pd", *, segment="", frequency=None,
                 candidates=None, expected_signs=None, problem_label=None, name="estudo"):
        _require_widgets()
        self.kind = str(kind or "pd").lower()
        self.problem_label = problem_label
        self.study_name = str(name or "estudo")

        rs, mac = self._coerce_data(series, macro, self.kind, segment, frequency)
        self.series = rs
        self.macro = mac
        if rs is not None:
            self.kind = rs.kind

        # --- estado compartilhado com os demais blocos da interface ---------
        self.fit_ = None            # último FitResult (Especificação)
        self.model_ = None          # o modelo que o produziu
        self.search_ = None         # SearchResult (Seleção)
        self.selected_spec_ = None  # Specification adotada (campeã ou escolha manual)
        self.compare_ = None        # comparação vs benchmarks + Diebold-Mariano
        self.diagnostics_ = None    # tabela da última bateria de diagnóstico
        self.diag_blocks_ = None    # placar por família de teste (lista de dicts)
        self.vif_ = None            # VIF por variável do modelo vigente
        self.scenarios_ = None      # ScenarioSet (Cenários & Projeção)
        self.projection_ = None     # Projection (Cenários & Projeção)
        self.weighted_ = None       # projeção ponderada pelos pesos de cenário
        self.projection_table_ = None   # projeção em formato longo (exportação)
        self.backtest_ = None       # dict do backtest de projeção (Backtest)
        self.coverage_ = None       # tabela de cobertura dos intervalos (Backtest)
        self.study_ = None          # StudyResult de um estudo completo (Exportar)
        self.stationarity_ = None   # tabela do último relatório de estacionariedade

        self._log_lines: list = []
        self._search_steps: list = []     # linhas da tabela de progresso da busca
        self._spec_por_desc: dict = {}    # describe() -> Specification da última grade
        self._search_secs = None          # duração da última busca (segundos)
        self._scen_pesos: dict = {}       # peso declarado por cenário manual
        self._bt_steps: list = []         # linhas da tabela de progresso do backtest
        self._bt_secs = None              # duração do último backtest (segundos)
        self._study_steps: list = []      # linhas da tabela de progresso do estudo completo
        self._study_secs = None           # duração do último estudo completo (segundos)
        self.report_path_ = None          # caminho do último relatório HTML gravado
        self.mlflow_run_id_ = None        # run_id do último registro no MLflow
        self._dirty_since_fit = False
        self._init_candidates = list(candidates) if candidates else None
        self._init_signs = dict(expected_signs or {})
        # defaults de cenário (a aba "Cenários & Projeção" pode criar widgets com
        # os nomes dd_stress_var / tx_scen_probs — to_config passa a lê-los)
        self._stress_var_default = None
        self._scenario_probs_default = (0.5, 0.3, 0.2)
        self._keepalive = None
        self._suspend_ka = False

        self._build()
        self._refresh_bar()
        self._refresh_serie()
        self._sync_model_fields()

    # ------------------------------------------------------------------ dados
    def _coerce_data(self, series, macro, kind, segment="", frequency=None):
        """Normaliza a entrada em ``(RiskSeries | None, DataFrame | None)``.

        Aceita ``RiskSeries``, ``SyntheticSeries`` (traz a macro junto),
        ``ReferenceStudy`` (usa o parâmetro ``kind``) ou ``pandas.Series`` crua, e
        **valida o alinhamento** com a macro antes de devolver.
        """
        S = _E.series
        if isinstance(series, S.ReferenceStudy):
            macro = macro if macro is not None else series.macro
            series = getattr(series, kind if kind in S.RISK_KINDS else "pd")
        if isinstance(series, S.SyntheticSeries):
            macro = macro if macro is not None else series.macro
            series = series.series
        if series is None:
            if macro is not None:
                macro = self._check_macro(macro)
            return None, macro

        if isinstance(series, S.RiskSeries):
            rs = series
        else:
            if not isinstance(series, pd.Series):
                series = pd.Series(series)
            if not isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                raise TypeError(
                    "a série do parâmetro precisa de índice temporal "
                    "(DatetimeIndex ou PeriodIndex) — recebido "
                    f"{type(series.index).__name__}."
                )
            freq = frequency or self._infer_freq(series.index)
            rs = S.RiskSeries(values=series.astype(float), kind=kind,
                              segment=segment or "", frequency=freq)
        if macro is not None:
            macro = self._check_macro(macro)
            self._check_alignment(rs, macro)
        return rs, macro

    @staticmethod
    def _infer_freq(index) -> str:
        """Frequência pandas do índice (``"MS"`` quando não dá para inferir)."""
        with suppress(Exception):
            f = pd.infer_freq(index)
            if f:
                return str(f)
        return "MS"

    @staticmethod
    def _check_macro(macro) -> pd.DataFrame:
        if not isinstance(macro, pd.DataFrame):
            raise TypeError("macro deve ser um pandas.DataFrame de variáveis macro.")
        if macro.shape[1] == 0:
            raise ValueError("macro sem colunas — informe ao menos uma variável macro.")
        if not isinstance(macro.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            raise TypeError(
                "o índice da macro deve ser DatetimeIndex ou PeriodIndex "
                f"(recebido {type(macro.index).__name__})."
            )
        return macro

    @staticmethod
    def _check_alignment(rs, macro: pd.DataFrame) -> None:
        """Valida o alinhamento temporal entre série e macro, com erro legível."""
        per_serie = isinstance(rs.index, pd.PeriodIndex)
        per_macro = isinstance(macro.index, pd.PeriodIndex)
        if per_serie != per_macro:
            raise ValueError(
                "série e macro estão desalinhadas: os índices são de tipos diferentes "
                f"({'PeriodIndex' if per_serie else 'DatetimeIndex'} na série × "
                f"{'PeriodIndex' if per_macro else 'DatetimeIndex'} na macro). "
                "Converta um dos dois (por exemplo, macro.index.to_timestamp())."
            )
        faltando = rs.index.difference(macro.index)
        if len(faltando):
            exemplo = ", ".join(str(p)[:10] for p in list(faltando)[:3])
            raise ValueError(
                f"série e macro estão desalinhadas: {len(faltando)} de {len(rs.index)} "
                f"período(s) da série não existem na macro (ex.: {exemplo}). "
                "Reindexe a macro para o mesmo calendário/frequência da série "
                "(macro = macro.reindex(serie.index)) antes de abrir a interface."
            )

    def _macro_cols(self) -> list:
        """Colunas macro **numéricas** disponíveis (as candidatas possíveis)."""
        if self.macro is None:
            return []
        return [c for c in self.macro.columns
                if pd.api.types.is_numeric_dtype(self.macro[c])]

    def set_data(self, series, macro=None, kind=None):
        """Troca os dados da interface (série + macro) e repinta tudo.

        Descarta o ajuste, a busca e a projeção correntes (eles pertenciam aos
        dados antigos) e reconstrói a matriz de sinais com as novas macros.
        """
        rs, mac = self._coerce_data(series, macro, (kind or self.kind), "", None)
        self.series, self.macro = rs, mac
        if rs is not None:
            self.kind = rs.kind
        self.fit_ = self.model_ = self.search_ = None
        self.selected_spec_ = self.compare_ = None
        self.diagnostics_ = self.diag_blocks_ = self.vif_ = None
        self.scenarios_ = self.projection_ = self.backtest_ = self.study_ = None
        self.weighted_ = self.projection_table_ = self.coverage_ = None
        self.stationarity_ = None
        self._spec_por_desc = {}
        self._search_steps = []
        self._search_secs = None
        self._scen_pesos = {}
        self._bt_steps = []
        self._bt_secs = None
        self._study_steps = []
        self._study_secs = None
        self.report_path_ = self.mlflow_run_id_ = None
        self._clear_dirty()
        self._init_candidates = None
        self._init_signs = {}
        # o link padrão acompanha o tipo de parâmetro; Vasicek só vale para PD
        if rs is not None:
            with suppress(Exception):
                self.dd_link.value = rs.default_link
            if self.dd_model.value == "vasicek" and self.kind != "pd":
                self.dd_model.value = "ardl"
                self._log("[dados] o fator Z de Vasicek é específico de PD — modelo "
                          "trocado para ARDL.")
        self._rebuild_signs()
        self._sync_data_widgets()
        self.out_estac.value = ""
        self.out_estac_resumo.value = ""
        self._clear_fit_outputs()
        self._clear_selecao_outputs()
        self._clear_cenarios_outputs()
        self._clear_backtest_outputs()
        self._clear_exportar_outputs()
        self._refresh_bar()
        self._refresh_serie()
        self._sync_model_fields()
        return self

    # ==================================================================
    # Utilitários de renderização (o padrão das UIs do credit_risk)
    # ==================================================================
    def _fig_html(self, fig, border=False, tight=True, stretch=False):
        """Converte uma figura matplotlib em ``<img>`` base64 (e fecha a figura)."""
        import base64
        import io as _io

        import matplotlib.pyplot as plt

        buf = _io.BytesIO()
        save_kw = {"format": "png", "dpi": min(int(fig.get_dpi()), 110)}
        if tight:
            save_kw["bbox_inches"] = "tight"
        fig.savefig(buf, **save_kw)
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        style = "width:100%;height:auto" if stretch else "max-width:100%;height:auto"
        if border:
            style += ";border:1px solid var(--line);border-radius:6px"
        return f"<img src='data:image/png;base64,{b64}' style='{style}'/>"

    # -- colorações semânticas reutilizáveis (passe em ``color_map``) ----
    @staticmethod
    def _css_veredito(v):
        """Verde para 'estacionária', âmbar para o contrário, neutro se indisponível."""
        s = _sem_acento(v)
        if "indisponivel" in s or "inconclusivo" in s or s in ("", "nan", "none", "—"):
            return "color:var(--muted);background-color:var(--neutral-bg)"
        if s.startswith("nao") or "raiz unitaria" in s:
            return "color:var(--warn-tx);background-color:var(--warn-bg);font-weight:600"
        return "color:var(--ok-tx);background-color:var(--ok-bg);font-weight:600"

    @staticmethod
    def _css_ordem(v):
        """I(0) verde · I(1) âmbar · I(2+) vermelho."""
        try:
            d = int(str(v).strip().replace("I(", "").replace(")", ""))
        except (TypeError, ValueError):
            return ""
        if d <= 0:
            return "color:var(--ok-tx);background-color:var(--ok-bg);font-weight:600"
        if d == 1:
            return "color:var(--warn-tx);background-color:var(--warn-bg);font-weight:600"
        return "color:var(--bad-tx);background-color:var(--bad-bg);font-weight:600"

    @staticmethod
    def _css_ok(v):
        """Coluna booleana de 'passou no teste' (bateria de diagnóstico)."""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "color:var(--muted);background-color:var(--neutral-bg)"
        s = str(v).strip().lower()
        if s in ("true", "sim", "ok", "✓"):
            return "color:var(--ok-tx);background-color:var(--ok-bg);font-weight:600"
        if s in ("false", "não", "nao", "✕", "✗"):
            return "color:var(--bad-tx);background-color:var(--bad-bg);font-weight:600"
        return "color:var(--muted);background-color:var(--neutral-bg)"

    @staticmethod
    def _css_pvalor(v):
        """p-valor: significante a 5% em verde, 10% em âmbar, o resto neutro."""
        try:
            p = float(v)
        except (TypeError, ValueError):
            return ""
        if p != p:
            return ""
        if p <= 0.05:
            return "color:var(--ok-tx);font-weight:600"
        if p <= 0.10:
            return "color:var(--warn-tx);font-weight:600"
        return "color:var(--muted)"

    @staticmethod
    def _css_coerencia(v):
        """Coerência do sinal econômico: ✓ verde · ✗ vermelho · — neutro."""
        s = str(v)
        if s.startswith("✓"):
            return "color:var(--ok-tx);background-color:var(--ok-bg);font-weight:600"
        if s.startswith("✗"):
            return "color:var(--bad-tx);background-color:var(--bad-bg);font-weight:600"
        return "color:var(--muted)"

    def _df_html(self, df, max_height=None, center=False, color_map=None,
                 pct_cols=None, precision=4, fmt_cols=None):
        """Tabela HTML no estilo da casa.

        ``color_map`` é ``{coluna: função(valor) -> css}`` — use os helpers
        ``_css_veredito``/``_css_ordem``/``_css_ok``/``_css_pvalor``/``_css_coerencia``
        (todos em tokens de tema, nunca hex). ``fmt_cols`` é
        ``{coluna: "{:.1f}"}`` para colunas que pedem casas decimais próprias
        (AIC ao lado de um RMSE de taxa, por exemplo).
        """
        sty = (df.style.hide(axis="index").set_table_styles(self._TABLE_STYLES)
               .set_properties(**{"font-size": "12px"}))
        if center:
            sty = sty.set_properties(**{"text-align": "center"})
            sty = sty.set_table_styles([{"selector": "th, td",
                                         "props": [("text-align", "center")]}],
                                       overwrite=False)
        else:
            txt = [c for c in df.columns if df[c].dtype == object]
            if txt:
                sty = sty.set_properties(subset=txt, **{"text-align": "left"})
        sty = sty.format(na_rep="—", precision=precision)
        if pct_cols:
            present = [c for c in pct_cols if c in df.columns]
            if present:
                sty = sty.format(lambda v: "" if pd.isna(v) else f"{v * 100:.1f}%",
                                 subset=present)
        for col, fmt in (fmt_cols or {}).items():
            if col in df.columns:
                sty = sty.format(
                    lambda v, _f=fmt: "—" if pd.isna(v) else _f.format(v), subset=[col])
        for col, fn in (color_map or {}).items():
            if col in df.columns:
                sty = sty.map(fn, subset=[col])
        html = sty.to_html()
        if max_height:
            html = f"<div style='max-height:{max_height};overflow:auto'>{html}</div>"
        return html

    @staticmethod
    def _pill(text, cls="muted"):
        return f"<span class='pill pill-{cls}'>{text}</span>"

    def _metric_tiles(self, itens: Mapping) -> str:
        """Mosaico de métricas ``{rótulo: valor}`` (AIC/BIC/R²/…)."""
        blocos = []
        for k, v in itens.items():
            if isinstance(v, (int, np.integer)):
                txt = f"{int(v)}"
            elif isinstance(v, (float, np.floating)):
                txt = "—" if not np.isfinite(v) else (f"{v:.4f}" if abs(v) < 1e4 else f"{v:,.1f}")
            else:
                txt = "—" if v is None else str(v)
            blocos.append(f"<div class='satui-metric'><div class='k'>{k}</div>"
                          f"<div class='v'>{txt}</div></div>")
        return "<div class='satui-metrics'>" + "".join(blocos) + "</div>"

    def _log(self, msg):
        """Escreve no console (mantém só as últimas 40 linhas)."""
        self._log_lines.append(str(msg))
        if len(self._log_lines) > 40:
            self._log_lines = self._log_lines[-40:]
        with self.out_log:
            _clear_output(wait=True)
            print("\n".join(self._log_lines))

    def _on_clear_log(self, _):
        self._log_lines = []
        self.out_log.clear_output()

    @contextmanager
    def _busy(self, *botoes, status=None, msg="processando…"):
        """Desabilita ``botoes`` durante uma ação longa e mostra "ocupado" em
        ``status``; re-habilita SEMPRE ao sair (e só limpa o status se o handler
        não o substituiu por um resultado/erro próprio)."""
        busy_html = f"<div class='satui-legend'><i>⏳ {msg}</i></div>"
        for b in botoes:
            b.disabled = True
        if status is not None:
            status.value = busy_html
        try:
            yield
        finally:
            for b in botoes:
                b.disabled = False
            if status is not None and status.value == busy_html:
                status.value = ""

    def _confirm_twice(self, btn, action, timeout=5.0):
        """Confirmação em DOIS cliques: o 1º arma o botão ("Confirmar?"), o 2º
        executa ``action``; sem o 2º clique, ele desarma sozinho."""
        import threading
        import time

        if not hasattr(btn, "_cc_desc"):
            btn._cc_desc = btn.description
            btn._cc_style = btn.button_style
        now = time.monotonic()
        armado = getattr(btn, "_cc_armed", 0.0)
        if armado and now - armado <= timeout:
            btn._cc_armed = 0.0
            btn.description = btn._cc_desc
            btn.button_style = btn._cc_style
            action()
            return
        btn._cc_armed = now
        btn.description = "Confirmar?"
        btn.button_style = "danger"

        def _revert():
            if getattr(btn, "_cc_armed", 0.0) == now:
                btn._cc_armed = 0.0
                btn.description = btn._cc_desc
                btn.button_style = btn._cc_style

        # daemon: desarmar o botão é cortesia visual — não pode segurar o encerramento
        # do kernel (ou da suíte de testes) por causa de um timer pendente.
        temporizador = threading.Timer(timeout, _revert)
        temporizador.daemon = True
        temporizador.start()

    def _mark_dirty(self, *_args):
        """Marca o ajuste como **desatualizado** (a especificação mudou depois do
        último ``fit``). No-op enquanto não há ajuste."""
        if self.fit_ is None or self._dirty_since_fit:
            return
        self._dirty_since_fit = True
        self.out_fit_warn.value = (
            "<div class='satui-notice'>⚠️ <b>Ajuste desatualizado</b> — a especificação "
            "mudou depois do último ajuste; os coeficientes abaixo são do modelo "
            "ANTERIOR. Clique em <i>Ajustar agora</i> para atualizar.</div>")
        self._invalidate_diag("a especificação mudou depois do último ajuste")
        self._invalidate_cenarios("a especificação mudou depois do último ajuste")
        self._invalidate_backtest("a especificação mudou depois do último ajuste")
        self._invalidate_exportar("a especificação mudou depois do último ajuste")
        self._render_scen_notice()
        self._refresh_bar()

    def _clear_dirty(self):
        self._dirty_since_fit = False
        if getattr(self, "out_fit_warn", None) is not None:
            self.out_fit_warn.value = ""
        # diagnóstico, projeção e backtest pertenciam ao ajuste anterior: somem com ele
        # (os CENÁRIOS ficam: a trajetória montada/colada é do usuário, não do modelo)
        self._invalidate_diag("o modelo vigente mudou")
        self._invalidate_cenarios("o modelo vigente mudou")
        self._invalidate_backtest("o modelo vigente mudou")
        self._invalidate_exportar("o modelo vigente mudou")
        self._render_scen_notice()

    # ==================================================================
    # Construção da interface
    # ==================================================================
    def _build(self):
        self.banner = W.HTML()
        self.bar = W.HTML()
        self.out_log = W.Output(layout=W.Layout(max_height="160px", overflow="auto"))
        self.btn_clear_log = W.Button(description="Limpar log", icon="eraser",
                                      tooltip="Limpa o histórico de mensagens do console",
                                      layout=W.Layout(width="140px"))
        self.btn_clear_log.on_click(self._on_clear_log)

        tab_serie = self._build_tab_serie()
        tab_spec = self._build_tab_spec()
        # cada aba é um VBox próprio: os cartões entram em ``.children`` logo em
        # seguida (a ordem importa — a aba Exportar lê widgets das anteriores).
        self.box_selecao = W.VBox(layout=W.Layout(padding="2px"))
        self.box_diagnostico = W.VBox(layout=W.Layout(padding="2px"))
        self.box_cenarios = W.VBox(layout=W.Layout(padding="2px"))
        self.box_backtest = W.VBox(layout=W.Layout(padding="2px"))
        self.box_exportar = W.VBox(layout=W.Layout(padding="2px"))
        self.box_selecao.children = self._build_tab_selecao()
        self.box_diagnostico.children = self._build_tab_diagnostico()
        self.box_cenarios.children = self._build_tab_cenarios()
        self.box_backtest.children = self._build_tab_backtest()
        self.box_exportar.children = self._build_tab_exportar()

        self.tabs = W.Tab(children=[tab_serie, tab_spec, self.box_selecao,
                                    self.box_diagnostico, self.box_cenarios,
                                    self.box_backtest, self.box_exportar])
        for i, t in enumerate(self.ABAS):
            self.tabs.set_title(i, t)
        self.tabs.add_class("satui-tabs")
        # abrir a aba Seleção recalcula o tamanho da grade (ele depende dos controles
        # da aba Especificação, que o usuário acabou de mexer)
        self.tabs.observe(self._on_tab_change, names="selected_index")

        console = W.VBox([
            W.HBox([W.HTML("<div class='satui-h'>Console</div>"), self.btn_clear_log],
                   layout=W.Layout(justify_content="space-between", align_items="center")),
            self.out_log])
        console.add_class("satui-card")

        self.cb_dark = W.ToggleButton(value=False, description="🌙 Tema escuro",
                                      tooltip="Alterna o tema claro/escuro da interface",
                                      layout=W.Layout(width="150px"))
        self.cb_dark.observe(self._on_dark, names="value")
        # buscas e walk-forward podem levar minutos: no Databricks o cluster
        # desligaria por inatividade no meio da busca (no-op fora do Spark).
        self.cb_keepalive = W.ToggleButton(
            value=False, description="☕ Manter cluster ativo",
            tooltip="Databricks: dispara um job Spark mínimo a cada 2 min para o cluster "
                    "não desligar por inatividade durante uma busca longa",
            layout=W.Layout(width="190px"))
        self.cb_keepalive.observe(self._on_keepalive, names="value")

        topbar = W.HBox([W.HTML(""), W.HBox([self.cb_keepalive, self.cb_dark])],
                        layout=W.Layout(justify_content="space-between"))
        self.panel = W.VBox([W.HTML(_CSS), topbar, self.banner, self.bar, self.tabs, console])
        self.panel.add_class("satui")

    # ------------------------------------------------------------------ aba Série
    def _build_tab_serie(self):
        # --- card: dados / estudo de referência ---------------------------
        self.dd_ref_kind = W.Dropdown(
            options=[("PD — taxa de default", "pd"),
                     ("LGD — severidade", "lgd"),
                     ("CCF — fator de conversão", "ccf")],
            value=self.kind if self.kind in ("pd", "lgd", "ccf") else "pd",
            description="Parâmetro:", style={"description_width": "initial"},
            layout=W.Layout(width="270px"))
        self.sl_ref_n = W.BoundedIntText(value=120, min=36, max=360, step=12,
                                         description="períodos:",
                                         style={"description_width": "initial"},
                                         layout=W.Layout(width="150px"))
        self.btn_ref_study = W.Button(
            description="Carregar estudo de referência", icon="flask", button_style="info",
            layout=W.Layout(width="auto", min_width="252px"),
            tooltip="Gera um segmento sintético completo (série do parâmetro + macro com "
                    "recessão e evento) para explorar a ferramenta na hora.")
        self.btn_ref_study.on_click(self._on_ref_study)
        self.out_ref_status = W.HTML()
        card_dados = W.VBox([
            W.HTML("<div class='satui-h'>Dados do estudo</div>"),
            W.HTML("<div class='satui-legend'>A interface trabalha com uma <b>série "
                   "temporal agregada</b> do parâmetro de risco de um segmento homogêneo "
                   "(taxa em [0,1] por período) e um painel de <b>variáveis macro</b> no "
                   "mesmo calendário. Sem dados em mãos, carregue o estudo de referência "
                   "— ele é sintético, de processo gerador conhecido, e serve para "
                   "aprender o fluxo.</div>"),
            W.HBox([self.dd_ref_kind, self.sl_ref_n, self.btn_ref_study],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_ref_status,
        ])
        card_dados.add_class("satui-card")

        # --- card: cabeçalho da série -------------------------------------
        self.out_serie_head = W.HTML()
        card_head = W.VBox([W.HTML("<div class='satui-h'>A série</div>"), self.out_serie_head])
        card_head.add_class("satui-card")

        # --- card: nível × escala do link ---------------------------------
        self.dd_link_serie = W.Dropdown(
            options=[("logit", "logit"), ("probit", "probit"), ("nível (identity)", "identity")],
            value="logit", description="Transformação:",
            style={"description_width": "initial"}, layout=W.Layout(width="240px"))
        self.dd_link_serie.observe(lambda c: self._render_serie_plots(), names="value")
        self.btn_serie_plot = W.Button(description="Redesenhar", icon="refresh",
                                       layout=W.Layout(width="140px"),
                                       tooltip="Redesenha a série no nível e na escala do "
                                               "link escolhido.")
        self.btn_serie_plot.on_click(self._on_serie_plot)
        self.out_serie_nivel = W.HTML(layout=W.Layout(width="49%"))
        self.out_serie_link = W.HTML(layout=W.Layout(width="49%"))
        card_serie = W.VBox([
            W.HTML("<div class='satui-h'>Série observada e escala de modelagem</div>"),
            W.HTML("<div class='satui-legend'>Os modelos estimam na escala do "
                   "<b>link</b> (logit/probit), que leva a taxa de [0,1] para toda a reta "
                   "— por isso a projeção nunca sai do intervalo. A série transformada é a "
                   "que precisa parecer bem-comportada (sem tendência explosiva).</div>"),
            W.HBox([self.dd_link_serie, self.btn_serie_plot],
                   layout=W.Layout(align_items="center")),
            W.HBox([self.out_serie_nivel, self.out_serie_link],
                   layout=W.Layout(justify_content="space-between")),
        ])
        card_serie.add_class("satui-card")

        # --- card: macros --------------------------------------------------
        self.sel_macro_plot = W.SelectMultiple(
            options=[], rows=6, description="Macros:",
            style={"description_width": "initial"}, layout=W.Layout(width="46%"))
        self.btn_macro_plot = W.Button(description="Desenhar macros", icon="line-chart",
                                       button_style="primary",
                                       layout=W.Layout(width="auto", min_width="170px"))
        self.btn_macro_plot.on_click(self._on_macro_plot)
        self.out_macro_plot = W.HTML()
        card_macro = W.VBox([
            W.HTML("<div class='satui-h'>Variáveis macro disponíveis</div>"),
            W.HTML("<div class='satui-legend'>Selecione uma ou mais (Ctrl/Shift) para ver "
                   "as trajetórias. Vale a pena olhar antes de escolher candidatas: "
                   "variáveis muito parecidas entre si disputam o mesmo papel e inflam o "
                   "VIF.</div>"),
            W.HBox([self.sel_macro_plot, self.btn_macro_plot],
                   layout=W.Layout(align_items="flex-start")),
            self.out_macro_plot,
        ])
        card_macro.add_class("satui-card")

        # --- card: estacionariedade ---------------------------------------
        self.fl_alpha_estac = W.BoundedFloatText(value=0.05, min=0.001, max=0.20, step=0.01,
                                                 description="nível α:",
                                                 style={"description_width": "initial"},
                                                 layout=W.Layout(width="140px"))
        self.cb_estac_todas = W.Checkbox(value=True, indent=False,
                                         description="incluir todas as macros (não só as candidatas)")
        self.btn_estac = W.Button(description="Rodar testes de estacionariedade",
                                  icon="check-square-o", button_style="primary",
                                  layout=W.Layout(width="auto", min_width="266px"),
                                  tooltip="ADF, KPSS e Phillips-Perron do alvo (na escala do "
                                          "link) e de cada macro, com a ordem de integração "
                                          "sugerida.")
        self.btn_estac.on_click(self._on_estac)
        self.out_estac_status = W.HTML()
        self.out_estac = W.HTML()
        self.out_estac_resumo = W.HTML()
        card_estac = W.VBox([
            W.HTML("<div class='satui-h'>Estacionariedade (o passo que todo mundo pula)</div>"),
            W.HTML("<div class='satui-help'><div class='ttl'>Como ler</div>"
                   "<b>ADF</b> tem H0 = <i>raiz unitária</i>: p pequeno ⇒ <b>estacionária</b>. "
                   "<b>KPSS</b> tem a nula <b>oposta</b> (H0 = estacionária): p pequeno ⇒ "
                   "<b>não estacionária</b>. Por isso os dois se leem <b>juntos</b>; "
                   "<b>Phillips-Perron</b> é o desempate robusto a autocorrelação. "
                   "A <b>ordem de integração I(d)</b> é quantas diferenças a série precisa "
                   "para os dois concordarem. Regredir uma série I(1) contra outra I(1) sem "
                   "cuidado produz <b>regressão espúria</b>: R² alto e relação inexistente."
                   "</div>"),
            W.HBox([self.fl_alpha_estac, self.cb_estac_todas, self.btn_estac],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_estac_status, self.out_estac, self.out_estac_resumo,
        ])
        card_estac.add_class("satui-card")

        self._card_dados = card_dados
        return W.VBox([card_dados, card_head, card_serie, card_macro, card_estac],
                      layout=W.Layout(padding="2px"))

    # ------------------------------------------------------------------ aba Especificação
    def _build_tab_spec(self):
        # --- card: modelo --------------------------------------------------
        self.dd_model = W.Dropdown(
            options=[(m["rotulo"], k) for k, m in MODELOS.items()],
            value="ardl", description="Modelo:",
            style={"description_width": "initial"}, layout=W.Layout(width="420px"))
        self.dd_model.observe(self._on_model_change, names="value")
        self.out_model_help = W.HTML()

        self.dd_link = W.Dropdown(
            options=[("logit", "logit"), ("probit", "probit"), ("nível (identity)", "identity")],
            value="logit", description="Link:", style={"description_width": "initial"},
            layout=W.Layout(width="230px"))
        self.dd_trend = W.Dropdown(
            options=[("constante", "c"), ("constante + tendência", "ct"), ("nenhum", "n")],
            value="c", description="Determinístico:",
            style={"description_width": "initial"}, layout=W.Layout(width="270px"))
        self.dd_cov = W.Dropdown(
            options=[("padrão (nonrobust)", "nonrobust"), ("HAC (Newey-West)", "HAC")],
            value="nonrobust", description="Covariância:",
            style={"description_width": "initial"}, layout=W.Layout(width="270px"))
        self.fl_rho = W.BoundedFloatText(value=0.12, min=0.001, max=0.999, step=0.01,
                                         description="ρ (correlação de ativos):",
                                         style={"description_width": "initial"},
                                         layout=W.Layout(width="290px"))
        self.cb_ttc_auto = W.Checkbox(value=True, indent=False,
                                      description="nível de longo prazo pela média da série")
        self.fl_pd_ttc = W.BoundedFloatText(value=0.04, min=0.0001, max=0.9999, step=0.005,
                                            description="nível de longo prazo:",
                                            style={"description_width": "initial"},
                                            layout=W.Layout(width="270px"))
        self.cb_ttc_auto.observe(lambda c: self._sync_model_fields(), names="value")
        self.tx_arima_order = W.Text(value="1,0,0", description="ordem (p,d,q):",
                                     style={"description_width": "initial"},
                                     layout=W.Layout(width="230px"))
        self.tx_arima_seasonal = W.Text(value="0,0,0,0", description="sazonal (P,D,Q,s):",
                                        style={"description_width": "initial"},
                                        layout=W.Layout(width="250px"))
        self.box_model_fields = W.VBox([
            W.HBox([self.dd_link, self.dd_trend, self.dd_cov],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.fl_rho, self.cb_ttc_auto, self.fl_pd_ttc],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.tx_arima_order, self.tx_arima_seasonal],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
        ])
        card_modelo = W.VBox([
            W.HTML("<div class='satui-h'>Modelo</div>"),
            self.dd_model, self.out_model_help, self.box_model_fields,
        ])
        card_modelo.add_class("satui-card")

        # --- card: matriz de sinais esperados -------------------------------
        self.box_signs = W.VBox()
        self.btn_signs_sugerir = W.Button(
            description="Sugerir sinais", icon="magic",
            layout=W.Layout(width="auto", min_width="160px"),
            tooltip="Preenche o sinal esperado pela família da variável (deterioração do "
                    "ciclo ⇒ +1; atividade/renda ⇒ −1). Revise sempre.")
        self.btn_signs_limpar = W.Button(
            description="Limpar sinais", icon="eraser",
            layout=W.Layout(width="auto", min_width="150px"),
            tooltip="Volta todas as variáveis para 'sem restrição'.")
        self.btn_signs_todas = W.Button(
            description="Marcar todas", icon="check-square-o",
            layout=W.Layout(width="auto", min_width="150px"))
        self.btn_signs_nenhuma = W.Button(
            description="Desmarcar todas", icon="square-o",
            layout=W.Layout(width="auto", min_width="160px"))
        self.btn_signs_sugerir.on_click(lambda b: self._on_signs_sugerir())
        self.btn_signs_limpar.on_click(lambda b: self._set_all_signs(0))
        self.btn_signs_todas.on_click(lambda b: self._set_all_candidates(True))
        self.btn_signs_nenhuma.on_click(lambda b: self._set_all_candidates(False))
        card_sinais = W.VBox([
            W.HTML("<div class='satui-h'>Candidatas e sinais econômicos esperados</div>"),
            W.HTML("<div class='satui-legend'>O <b>sinal esperado</b> é o filtro duro da "
                   "coerência econômica: uma especificação cujo efeito líquido da variável "
                   "sai com o sinal trocado é <b>desqualificada</b> na busca, por melhor que "
                   "seja o ajuste. Marque <b>+1</b> quando a alta da variável deve "
                   "<b>piorar</b> o parâmetro, <b>−1</b> quando deve melhorá-lo, e "
                   "<b>livre</b> quando a teoria não decide.</div>"),
            W.HBox([self.btn_signs_sugerir, self.btn_signs_limpar,
                    self.btn_signs_todas, self.btn_signs_nenhuma],
                   layout=W.Layout(flex_flow="row wrap")),
            W.HTML("<div class='satui-signhead'>candidata &nbsp;·&nbsp; sinal esperado "
                   "&nbsp;·&nbsp; defasagem usada no <i>Ajustar agora</i> (a busca varre "
                   "todo o conjunto de defasagens)</div>"),
            self.box_signs,
        ])
        card_sinais.add_class("satui-card")

        # --- card: grade e regras de seleção --------------------------------
        self.tx_lag_set = W.Text(value="0,1,3,6", description="defasagens:",
                                 style={"description_width": "initial"},
                                 layout=W.Layout(width="230px"))
        self.tx_ar_orders = W.Text(value="1", description="ordens AR:",
                                   style={"description_width": "initial"},
                                   layout=W.Layout(width="200px"))
        self.sl_max_vars = W.BoundedIntText(value=3, min=1, max=8, description="máx. variáveis:",
                                            style={"description_width": "initial"},
                                            layout=W.Layout(width="190px"))
        self.cb_seasonal = W.Checkbox(value=False, indent=False, description="dummies sazonais")
        self.sl_seasonal_period = W.BoundedIntText(value=12, min=2, max=52,
                                                   description="período sazonal:",
                                                   style={"description_width": "initial"},
                                                   layout=W.Layout(width="200px"))
        self.fl_vif_max = W.BoundedFloatText(value=5.0, min=1.0, max=100.0, step=0.5,
                                             description="VIF máx.:",
                                             style={"description_width": "initial"},
                                             layout=W.Layout(width="160px"))
        self.dd_criterion = W.Dropdown(
            options=[("erro fora da amostra (oos_rmse)", "oos_rmse"),
                     ("AIC", "aic"), ("BIC", "bic")],
            value="oos_rmse", description="critério:",
            style={"description_width": "initial"}, layout=W.Layout(width="310px"))
        self.sl_horizon = W.BoundedIntText(value=12, min=1, max=120, description="horizonte:",
                                           style={"description_width": "initial"},
                                           layout=W.Layout(width="170px"))
        self.sl_min_train = W.BoundedIntText(value=0, min=0, max=500,
                                             description="mín. de treino (0 = automático):",
                                             style={"description_width": "initial"},
                                             layout=W.Layout(width="330px"))
        self.sl_max_specs = W.BoundedIntText(value=400, min=1, max=20000, step=50,
                                             description="máx. especificações:",
                                             style={"description_width": "initial"},
                                             layout=W.Layout(width="240px"))
        self.tx_nome = W.Text(value=self.study_name, description="nome do estudo:",
                              style={"description_width": "initial"},
                              layout=W.Layout(width="330px"))
        card_grade = W.VBox([
            W.HTML("<div class='satui-h'>Grade de especificações e regras de seleção</div>"),
            W.HTML("<div class='satui-legend'>A busca combina, para cada subconjunto de até "
                   "<b>máx. variáveis</b> candidatas, <b>uma</b> defasagem por variável (do "
                   "conjunto de defasagens) e cada ordem AR. O <b>horizonte</b> e o "
                   "<b>mínimo de treino</b> definem a validação fora da amostra; o "
                   "<b>VIF máx.</b> descarta especificações com macro redundante.</div>"),
            W.HBox([self.tx_lag_set, self.tx_ar_orders, self.sl_max_vars],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.cb_seasonal, self.sl_seasonal_period, self.fl_vif_max],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.dd_criterion, self.sl_horizon, self.sl_min_train],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.sl_max_specs, self.tx_nome],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
        ])
        card_grade.add_class("satui-card")

        # --- card: ajuste único ---------------------------------------------
        self.btn_fit_now = W.Button(description="Ajustar agora", icon="play",
                                    button_style="success",
                                    layout=W.Layout(width="auto", min_width="170px"),
                                    tooltip="Ajusta UM modelo com a especificação corrente "
                                            "(sem busca) e mostra coeficientes, p-valores e "
                                            "AIC/BIC.")
        self.btn_fit_now.on_click(self._on_fit_now)
        self.cb_fit_plot = W.Checkbox(value=True, indent=False,
                                      description="desenhar observado × ajustado")
        self.out_fit_status = W.HTML()
        self.out_fit_warn = W.HTML()
        self.out_fit_spec = W.HTML()
        self.out_fit_metrics = W.HTML()
        self.out_fit_coef = W.HTML()
        self.out_fit_plot = W.HTML()
        card_fit = W.VBox([
            W.HTML("<div class='satui-h'>Ajuste único (feedback rápido)</div>"),
            W.HTML("<div class='satui-legend'>Antes de gastar uma busca inteira: ajuste "
                   "<b>uma</b> especificação e olhe os sinais, os p-valores e o AIC/BIC. "
                   "A coluna <b>coerência</b> compara o sinal estimado de cada variável com "
                   "o sinal que você declarou esperar.</div>"),
            W.HBox([self.btn_fit_now, self.cb_fit_plot],
                   layout=W.Layout(align_items="center")),
            self.out_fit_status, self.out_fit_warn, self.out_fit_spec,
            self.out_fit_metrics, self.out_fit_coef, self.out_fit_plot,
        ])
        card_fit.add_class("satui-card")

        # --- card: estudo completo em um clique -------------------------------
        self.btn_run_study = W.Button(
            description="Rodar estudo completo", icon="rocket", button_style="primary",
            layout=W.Layout(width="auto", min_width="230px"),
            tooltip="Monta a StudyConfig desta tela e roda busca → ajuste → diagnóstico → "
                    "cenários → projeção de uma vez, preenchendo todas as abas.")
        self.btn_run_study.on_click(self._on_run_study)
        self.cb_study_report = W.Checkbox(
            value=True, indent=False,
            description="gerar também o HTML do relatório de governança")
        self.out_study_status = W.HTML()
        self.out_study_progress = W.HTML()
        self.out_study_timer = W.HTML()
        self.out_study_resumo = W.HTML()
        card_study = W.VBox([
            W.HTML("<div class='satui-h'>Estudo completo em um clique</div>"),
            W.HTML("<div class='satui-help'><div class='ttl'>O que este botão faz</div>"
                   "Exatamente as <b>cinco chamadas</b> do fluxo manual, na ordem: "
                   "<b>busca</b> champion-challenger sobre a grade acima → <b>ajuste</b> da "
                   "campeã → <b>bateria de diagnóstico</b> → <b>cenários padrão</b> sobre a "
                   "variável de estresse → <b>projeção condicional</b>. Ao terminar, as abas "
                   "<i>Seleção</i>, <i>Diagnóstico</i> e <i>Cenários &amp; Projeção</i> "
                   "aparecem preenchidas — e você pode discordar de qualquer etapa ali "
                   "mesmo (adotar outra especificação, trocar o cenário, reprojetar). O "
                   "<b>backtest</b> fica de fora de propósito: ele é caro e tem aba própria."
                   "<br>É o atalho de quem já sabe o que quer; para conduzir passo a passo, "
                   "use <i>Ajustar agora</i> acima e siga pelas abas.</div>"),
            W.HBox([self.btn_run_study, self.cb_study_report],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_study_status, self.out_study_progress, self.out_study_timer,
            self.out_study_resumo,
        ])
        card_study.add_class("satui-card")

        # mudanças na especificação deixam o ajuste desatualizado
        for w in (self.dd_model, self.dd_link, self.dd_trend, self.dd_cov, self.fl_rho,
                  self.cb_ttc_auto, self.fl_pd_ttc, self.tx_arima_order,
                  self.tx_arima_seasonal, self.tx_lag_set, self.tx_ar_orders,
                  self.cb_seasonal, self.sl_seasonal_period):
            w.observe(self._mark_dirty, names="value")

        self._rebuild_signs()
        self._render_model_help()
        return W.VBox([card_modelo, card_sinais, card_grade, card_fit, card_study],
                      layout=W.Layout(padding="2px"))

    # ------------------------------------------------------------------ matriz de sinais
    def _rebuild_signs(self):
        """(Re)constrói a matriz candidata × sinal × defasagem a partir da macro."""
        cols = self._macro_cols()
        antigos_cb = getattr(self, "_sign_cbs", {})
        antigos_tg = getattr(self, "_sign_tgs", {})
        antigos_lag = getattr(self, "_sign_lags", {})
        self._sign_cbs, self._sign_tgs, self._sign_lags = {}, {}, {}
        linhas = []
        for var in cols:
            marcada = (var in self._init_candidates) if self._init_candidates is not None else True
            if var in antigos_cb:
                marcada = bool(antigos_cb[var].value)
            cb = W.Checkbox(value=bool(marcada), description=var, indent=False,
                            style={"description_width": "initial"},
                            layout=W.Layout(width="220px", margin="0 6px 0 0"))
            sinal = int(self._init_signs.get(var, 0) or 0)
            if var in antigos_tg:
                sinal = int(antigos_tg[var].value)
            tg = W.ToggleButtons(
                options=[("+1  ↑ piora", 1), ("livre", 0), ("−1  ↓ melhora", -1)],
                value=sinal if sinal in (-1, 0, 1) else 0,
                style={"button_width": "104px"}, layout=W.Layout(width="auto"))
            lag = W.BoundedIntText(value=int(antigos_lag[var].value) if var in antigos_lag else 0,
                                   min=0, max=24, description="lag:",
                                   style={"description_width": "initial"},
                                   layout=W.Layout(width="110px", margin="0 0 0 10px"))
            for w in (cb, tg, lag):
                w.observe(self._mark_dirty, names="value")
            self._sign_cbs[var], self._sign_tgs[var], self._sign_lags[var] = cb, tg, lag
            linha = W.HBox([cb, tg, lag], layout=W.Layout(align_items="center",
                                                          margin="0 0 3px 0"))
            linha.add_class("satui-signrow")
            linhas.append(linha)
        if not linhas:
            linhas = [W.HTML("<div class='satui-legend'>Sem variáveis macro carregadas — "
                             "traga uma macro ou carregue o estudo de referência na aba "
                             "<b>Série</b>.</div>")]
        self.box_signs.children = tuple(linhas)

    def _set_all_candidates(self, valor: bool):
        for cb in getattr(self, "_sign_cbs", {}).values():
            cb.value = bool(valor)

    def _set_all_signs(self, valor: int):
        for tg in getattr(self, "_sign_tgs", {}).values():
            tg.value = int(valor)

    def _on_signs_sugerir(self):
        """Preenche os sinais pela família da variável (heurística revisável)."""
        n = 0
        for var, tg in getattr(self, "_sign_tgs", {}).items():
            nome = _sem_acento(var)
            for chave, sinal in _SINAL_SUGERIDO.items():
                if chave in nome:
                    tg.value = int(sinal)
                    n += 1
                    break
        self._log(f"[sinais] sugestão aplicada a {n} variável(is) — revise: a heurística "
                  "olha só o nome, não a economia do seu segmento.")

    def candidates(self) -> list:
        """Variáveis macro marcadas como candidatas."""
        return [v for v, cb in getattr(self, "_sign_cbs", {}).items() if cb.value]

    def expected_signs(self) -> dict:
        """Sinais esperados declarados (só as candidatas com restrição)."""
        return {v: int(self._sign_tgs[v].value) for v in self.candidates()
                if int(self._sign_tgs[v].value) != 0}

    def lag_por_variavel(self) -> dict:
        """Defasagem escolhida por variável para o **ajuste único**."""
        return {v: int(self._sign_lags[v].value) for v in self.candidates()}

    # ==================================================================
    # Aba Série — render
    # ==================================================================
    def _sync_data_widgets(self):
        """Repõe as opções dos widgets que dependem das colunas da macro."""
        cols = self._macro_cols()
        self.sel_macro_plot.options = cols
        if cols:
            self.sel_macro_plot.value = tuple(cols[: min(3, len(cols))])
        dd = getattr(self, "dd_stress_var", None)     # criado pela aba de cenários
        if dd is not None:
            dd.options = cols

    def _param_label(self) -> str:
        """Rótulo do parâmetro (neutro: PD, LGD ou CCF — ou o rótulo do usuário)."""
        return self.problem_label or str(self.kind).upper()

    def _refresh_serie(self):
        self._sync_data_widgets()
        self._render_serie_head()
        self._render_serie_plots()

    def _on_serie_plot(self, b):
        """Botão *Redesenhar* — sob :meth:`_busy` (desenhar duas figuras não é
        instantâneo em série longa, e um botão mudo parece travado)."""
        with self._busy(self.btn_serie_plot, msg="desenhando…"):
            self._refresh_serie()

    def _render_serie_head(self):
        if self.series is None:
            self.out_serie_head.value = (
                "<div class='satui-legend'>Nenhuma série carregada. Use o botão "
                "<b>Carregar estudo de referência</b> acima ou construa a interface com "
                "<code>SatelliteUI(serie, macro, kind=...)</code>.</div>")
            return
        rs = self.series
        v = rs.values
        idx = v.index
        ini = str(idx[0])[:10]
        fim = str(idx[-1])[:10]
        macros = ", ".join(self._macro_cols()) or "—"
        self.out_serie_head.value = (
            self._metric_tiles({
                "parâmetro": self._param_label(),
                "observações": int(len(v)),
                "frequência": rs.frequency,
                "média": float(v.mean()),
                "mínimo": float(v.min()),
                "máximo": float(v.max()),
            })
            + "<div class='satui-legend' style='margin-top:8px'>"
            f"<b>Segmento:</b> {rs.segment or '—'} &nbsp;·&nbsp; "
            f"<b>Período:</b> {ini} a {fim} &nbsp;·&nbsp; "
            f"<b>Nível de longo prazo (média):</b> {rs.ttc():.4f} &nbsp;·&nbsp; "
            f"<b>Macros:</b> {macros}</div>")

    def _render_serie_plots(self):
        if self.series is None:
            vazio = ("<div class='satui-legend'>— sem série carregada —</div>")
            self.out_serie_nivel.value = vazio
            self.out_serie_link.value = vazio
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - depende do ambiente
            self.out_serie_nivel.value = (
                f"<div class='satui-legend'>matplotlib indisponível: {exc}</div>")
            return
        cores = _E.report._palette()
        v = self.series.values
        idx = _E.report._to_ts(v.index)

        fig, ax = plt.subplots(figsize=(5.6, 3.1))
        ax.plot(idx, v.to_numpy(dtype=float), color=cores["primaria"], lw=1.8)
        ax.set_title(f"{self._param_label()} observado (nível)", fontsize=11)
        ax.set_ylabel(self._param_label())
        ax.grid(alpha=0.25)
        fig.tight_layout()
        self.out_serie_nivel.value = self._fig_html(fig, stretch=True)

        link = self.dd_link_serie.value
        try:
            t = self.series.transformed(link)
        except Exception as exc:  # noqa: BLE001
            self.out_serie_link.value = f"<div class='satui-legend'>{exc}</div>"
            return
        fig2, ax2 = plt.subplots(figsize=(5.6, 3.1))
        ax2.plot(idx, np.asarray(t, dtype=float), color=cores["secundaria"], lw=1.8)
        ax2.set_title(f"Escala de modelagem — {link}", fontsize=11)
        ax2.set_ylabel(f"{link}({self._param_label()})")
        ax2.grid(alpha=0.25)
        fig2.tight_layout()
        self.out_serie_link.value = self._fig_html(fig2, stretch=True)

    def _on_macro_plot(self, b):
        if self.macro is None:
            self.out_macro_plot.value = (
                "<div class='satui-legend'>Nenhuma macro carregada.</div>")
            return
        cols = list(self.sel_macro_plot.value) or self._macro_cols()[:3]
        if not cols:
            self.out_macro_plot.value = (
                "<div class='satui-legend'>Selecione ao menos uma variável.</div>")
            return
        with self._busy(self.btn_macro_plot, msg="desenhando…"):
            import matplotlib.pyplot as plt

            cores = _E.report._palette()
            paleta = [cores["primaria"], cores["secundaria"], "#2ca02c", "#9467bd", "#ff7f0e"]
            n = len(cols)
            fig, axes = plt.subplots(n, 1, figsize=(10.5, 1.9 * n), sharex=True)
            axes = np.atleast_1d(axes)
            idx = _E.report._to_ts(self.macro.index)
            for i, c in enumerate(cols):
                axes[i].plot(idx, self.macro[c].to_numpy(dtype=float),
                             color=paleta[i % len(paleta)], lw=1.5)
                axes[i].set_ylabel(c, fontsize=9)
                axes[i].grid(alpha=0.25)
            axes[0].set_title("Variáveis macro", fontsize=11)
            fig.tight_layout()
            self.out_macro_plot.value = self._fig_html(fig, stretch=True)
            self._log(f"[série] macros desenhadas: {', '.join(cols)}.")

    # ------------------------------------------------------------------ estacionariedade
    def _estac_itens(self):
        """``[(rótulo, série)]`` a testar: o alvo (na escala do link) e as macros."""
        itens = []
        if self.series is not None:
            link = self.dd_link_serie.value
            with suppress(Exception):
                itens.append((f"{self._param_label()} ({link})", self.series.transformed(link)))
            itens.append((f"{self._param_label()} (nível)", self.series.values))
        if self.macro is not None:
            cols = self._macro_cols() if self.cb_estac_todas.value else self.candidates()
            for c in cols:
                itens.append((c, self.macro[c]))
        return itens

    def _on_estac(self, b):
        itens = self._estac_itens()
        if not itens:
            self.out_estac_status.value = (
                "<div class='satui-legend'>Carregue uma série (ou uma macro) antes.</div>")
            return
        alpha = float(self.fl_alpha_estac.value)
        with self._busy(self.btn_estac, status=self.out_estac_status,
                        msg="rodando ADF/KPSS/Phillips-Perron…"):
            try:
                tabela = self._estac_table(itens, alpha)
            except ImportError as exc:
                self.out_estac_status.value = (
                    f"<div class='satui-legend'>Testes indisponíveis: {exc}</div>")
                self._log(f"[estacionariedade] dependência ausente: {exc}")
                return
            except Exception as exc:  # noqa: BLE001
                self.out_estac_status.value = f"<div class='satui-legend'>Erro: {exc}</div>"
                self._log(f"[estacionariedade] erro: {exc}")
                return
        self.stationarity_ = tabela
        self.out_estac.value = self._df_html(
            tabela, max_height="360px",
            color_map={"ADF": self._css_veredito, "KPSS": self._css_veredito,
                       "Phillips-Perron": self._css_veredito, "I(d)": self._css_ordem,
                       "p ADF": self._css_pvalor})
        self.out_estac_resumo.value = self._estac_resumo_html(tabela)
        self._log(f"[estacionariedade] {len(tabela)} série(s) testada(s) a α={alpha:.3f}.")

    def _estac_table(self, itens, alpha: float) -> pd.DataFrame:
        """Monta a tabela ADF/KPSS/PP + ordem de integração por série."""
        diag = _E.diagnostics
        linhas = []
        for nome, s in itens:
            serie = pd.Series(np.asarray(s, dtype=float)).dropna()
            rep = diag.stationarity_report(serie, alpha=alpha)
            linha = {"série": nome, "n": int(serie.shape[0])}
            for teste, rot in (("ADF", "ADF"), ("KPSS", "KPSS"),
                               ("Phillips-Perron", "Phillips-Perron")):
                sub = rep[rep["teste"] == teste]
                if sub.empty:
                    linha[rot] = "indisponível"
                    continue
                r = sub.iloc[0]
                linha[rot] = str(r["conclusao"])
                if teste == "ADF":
                    linha["p ADF"] = float(r["p_valor"]) if pd.notna(r["p_valor"]) else np.nan
                elif teste == "KPSS":
                    linha["p KPSS"] = float(r["p_valor"]) if pd.notna(r["p_valor"]) else np.nan
            d = int(rep.attrs.get("ordem_integracao", 0))
            linha["I(d)"] = f"I({d})"
            linha["_d"] = d
            linhas.append(linha)
        tab = pd.DataFrame(linhas)
        cols = ["série", "n", "ADF", "p ADF", "KPSS", "p KPSS", "Phillips-Perron", "I(d)"]
        tab = tab[[c for c in cols if c in tab.columns] + ["_d"]]
        tab.attrs["ordens"] = {r["série"]: int(r["_d"]) for _, r in tab.iterrows()}
        return tab.drop(columns="_d")

    def _estac_resumo_html(self, tabela: pd.DataFrame) -> str:
        """Leitura em texto do que os testes implicam para a modelagem."""
        ordens = tabela.attrs.get("ordens", {})
        itens = []
        for nome, d in ordens.items():
            if d <= 0:
                itens.append(f"<li><b>{nome}</b> parece <b>I(0)</b> (estacionária): "
                             "pode entrar em nível.</li>")
            elif d == 1:
                itens.append(
                    f"<li><b>{nome}</b> parece <b>I(1)</b>: considere usar a "
                    "<b>primeira diferença</b> (ou variação), apoiar-se nas "
                    "<b>defasagens</b> e no termo AR, ou tratar a relação de longo prazo "
                    "explicitamente (cointegração).</li>")
            else:
                itens.append(f"<li><b>{nome}</b> parece <b>I({d})</b>: diferencie antes de "
                             "usar — em nível, o risco de regressão espúria é alto.</li>")
        # conflito ADF × KPSS: os dois testes com nulas opostas discordando
        conflitos = []
        for _, r in tabela.iterrows():
            adf_est = not _sem_acento(r.get("ADF", "")).startswith("nao")
            kpss_txt = _sem_acento(r.get("KPSS", ""))
            kpss_est = not kpss_txt.startswith("nao")
            if "indisponivel" in kpss_txt:
                continue
            if adf_est != kpss_est:
                conflitos.append(str(r["série"]))
        extra = ""
        if conflitos:
            extra = ("<div style='margin-top:6px'>⚠️ <b>ADF e KPSS discordam</b> em "
                     f"{', '.join(conflitos)} — o caso clássico de série <i>perto</i> da "
                     "raiz unitária em amostra curta. Trate como duvidosa: teste as duas "
                     "leituras (nível e diferença) e escolha pela previsão fora da "
                     "amostra, não pelo R².</div>")
        alvo = self._param_label()
        return ("<div class='satui-help'><div class='ttl'>O que isso implica</div>"
                f"<ul>{''.join(itens)}</ul>{extra}"
                "<div style='margin-top:6px'>Regra de bolso: misturar séries de ordens "
                f"diferentes na mesma regressão (por exemplo, {alvo} I(0) contra uma macro "
                "I(1)) costuma indicar que falta uma transformação — a coerência das ordens "
                "é pré-requisito da coerência dos sinais.</div></div>")

    # ==================================================================
    # Aba Especificação — modelo, especificação e ajuste único
    # ==================================================================
    def _on_model_change(self, change):
        self._render_model_help()
        self._sync_model_fields()
        self._mark_dirty()

    def _render_model_help(self):
        info = MODELOS[self.dd_model.value]
        aviso = ""
        if self.series is not None and self.kind not in info["kinds"]:
            aviso = ("<div style='margin-top:6px'>⚠️ Este modelo não se aplica ao parâmetro "
                     f"<b>{self._param_label()}</b> desta série.</div>")
        if info["registry"] is None:
            aviso += ("<div style='margin-top:6px'>ℹ️ Entra na busca como <b>benchmark</b>, "
                      "não como candidato: a configuração exportada usa o modelo candidato "
                      "equivalente.</div>")
        self.out_model_help.value = (
            f"<div class='satui-help'><div class='ttl'>{info['rotulo']}</div>"
            f"{info['ajuda']}{aviso}"
            "<div style='margin-top:6px'><b>VAR/VECM</b> não aparecem aqui: eles projetam o "
            "<i>sistema</i> inteiro em vez de um parâmetro condicionado a cenário — servem "
            "para gerar cenários internamente consistentes e impulso-resposta (aba "
            "<i>Cenários &amp; Projeção</i>).</div></div>")

    def _sync_model_fields(self):
        """Mostra/esconde os campos específicos do modelo escolhido."""
        campos = MODELOS[self.dd_model.value]["campos"]

        def _vis(w, on):
            w.layout.display = None if on else "none"

        _vis(self.dd_link, "link" in campos)
        _vis(self.dd_trend, "trend" in campos)
        _vis(self.dd_cov, "cov" in campos)
        _vis(self.fl_rho, "vasicek" in campos)
        _vis(self.cb_ttc_auto, "vasicek" in campos)
        _vis(self.fl_pd_ttc, "vasicek" in campos and not self.cb_ttc_auto.value)
        _vis(self.tx_arima_order, "arima" in campos)
        _vis(self.tx_arima_seasonal, "arima" in campos)

    # ------------------------------------------------------------------ especificação
    @staticmethod
    def _parse_ints(txt, campo="valores") -> list:
        """Lê ``"0,1,3, 6"`` como ``[0, 1, 3, 6]`` com erro legível."""
        partes = [p for p in re.split(r"[,;\s]+", str(txt).strip()) if p]
        try:
            vals = [int(p) for p in partes]
        except ValueError:
            raise ValueError(
                f"{campo}: use inteiros separados por vírgula (ex.: 0,1,3,6) — "
                f"recebido {txt!r}.") from None
        if not vals:
            raise ValueError(f"{campo}: informe ao menos um inteiro (ex.: 0,1,3,6).")
        return vals

    def current_spec(self):
        """A :class:`Specification` correspondente aos controles da tela.

        Usa **uma** defasagem por variável (a coluna *lag* da matriz de sinais) e a
        **primeira** ordem AR da lista — é a especificação do *Ajustar agora*; a
        busca varre o conjunto inteiro.
        """
        Specification = _E.base.Specification
        exog = {v: [lag] for v, lag in self.lag_por_variavel().items()}
        ar = self._parse_ints(self.tx_ar_orders.value, "ordens AR")[0]
        return Specification(
            exog=exog, ar=int(ar), trend=self.dd_trend.value, link=self.dd_link.value,
            seasonal=bool(self.cb_seasonal.value),
            seasonal_period=int(self.sl_seasonal_period.value),
            expected_signs=self.expected_signs(),
            name=self.tx_nome.value or "estudo")

    def build_model(self):
        """Instancia (sem ajustar) o modelo da especificação corrente."""
        if self.series is None:
            raise RuntimeError(
                "sem série carregada: traga os dados ou clique em 'Carregar estudo de "
                "referência' na aba Série.")
        key = self.dd_model.value
        spec = self.current_spec()
        if key in ("ardl", "vasicek", "beta", "fractional") and spec.exog and self.macro is None:
            raise RuntimeError("há variáveis candidatas marcadas mas nenhuma macro carregada.")
        if key == "ardl":
            return _E.ardl.ARDL(self.series, self.macro, spec, cov_type=self.dd_cov.value)
        if key == "vasicek":
            pd_ttc = None if self.cb_ttc_auto.value else float(self.fl_pd_ttc.value)
            return _E.vasicek.VasicekZ(self.series, self.macro, spec,
                                       rho=float(self.fl_rho.value), pd_ttc=pd_ttc,
                                       cov_type=self.dd_cov.value)
        if key == "beta":
            return _E.fractional.BetaRegression(self.series, self.macro, spec)
        if key == "fractional":
            return _E.fractional.FractionalLogit(self.series, self.macro, spec)
        if key == "arima":
            order = tuple(self._parse_ints(self.tx_arima_order.value, "ordem (p,d,q)"))
            saz = tuple(self._parse_ints(self.tx_arima_seasonal.value, "sazonal (P,D,Q,s)"))
            if len(order) != 3:
                raise ValueError("ordem (p,d,q): informe exatamente 3 inteiros (ex.: 1,0,0).")
            if len(saz) != 4:
                raise ValueError("sazonal (P,D,Q,s): informe 4 inteiros (ex.: 0,0,0,0).")
            exog = {v: [lag] for v, lag in self.lag_por_variavel().items()} or None
            return _E.arima.ARIMA(self.series, self.macro, exog=exog, order=order,
                                  seasonal_order=saz, link=self.dd_link.value)
        if key == "random_walk":
            return _E.benchmarks.RandomWalk(self.series)
        if key == "media_historica":
            return _E.benchmarks.HistoricalMean(self.series)
        if key == "sazonal_ingenuo":
            return _E.benchmarks.SeasonalNaive(self.series,
                                               period=int(self.sl_seasonal_period.value))
        raise ValueError(f"modelo desconhecido na interface: {key!r}.")

    # ------------------------------------------------------------------ ajuste único
    def _clear_fit_outputs(self):
        for w in ("out_fit_spec", "out_fit_metrics", "out_fit_coef", "out_fit_plot"):
            widget = getattr(self, w, None)
            if widget is not None:
                widget.value = ""

    def _on_fit_now(self, b):
        with self._busy(self.btn_fit_now, status=self.out_fit_status, msg="ajustando…"):
            try:
                modelo = self.build_model()
                fit = modelo.fit()
            except Exception as exc:  # noqa: BLE001
                self.out_fit_status.value = (
                    f"<div class='satui-notice'>Não foi possível ajustar: {exc}</div>")
                self._log(f"[ajuste] erro: {exc}")
                return
            self.model_, self.fit_ = modelo, fit
            self._clear_dirty()
            self._render_fit()
            self._refresh_bar()
            aic = f"{fit.aic:.1f}" if fit.aic is not None else "—"
            self._log(f"[ajuste] {fit.model_name} · {fit.nobs} obs · AIC {aic} · "
                      f"{(fit.spec.describe() if fit.spec else '—')}")

    def _render_fit(self):
        fit = self.fit_
        if fit is None:
            return
        spec = fit.spec
        self.out_fit_spec.value = (
            "<div class='satui-legend'><b>Especificação ajustada:</b> "
            f"<code>{spec.describe() if spec else fit.model_name}</code> &nbsp;·&nbsp; "
            f"modelo <b>{fit.model_name}</b> &nbsp;·&nbsp; link <b>{fit.link}</b></div>")
        m = fit.metrics()
        self.out_fit_metrics.value = self._metric_tiles({
            "obs.": m["nobs"], "params": m["n_params"], "AIC": m["AIC"], "BIC": m["BIC"],
            "R²": m["R2"], "σ resid.": m["sigma"], "RMSE (link)": m["RMSE_link"]})
        self.out_fit_coef.value = self._coef_html(fit)
        if self.cb_fit_plot.value:
            try:
                fig = _E.report.plot_fit(fit, self.series)
                self.out_fit_plot.value = self._fig_html(fig, stretch=True)
            except Exception as exc:  # noqa: BLE001
                self.out_fit_plot.value = (
                    f"<div class='satui-legend'>Gráfico indisponível: {exc}</div>")
        else:
            self.out_fit_plot.value = ""

    @staticmethod
    def _estrelas(p) -> str:
        try:
            p = float(p)
        except (TypeError, ValueError):
            return "—"
        if p != p:
            return "—"
        if p <= 0.01:
            return "*** (1%)"
        if p <= 0.05:
            return "** (5%)"
        if p <= 0.10:
            return "* (10%)"
        return "—"

    def _termo_variavel(self, termo: str) -> Optional[str]:
        """Variável macro por trás de um nome de coeficiente (``desemprego_l3``)."""
        base = re.sub(r"_l\d+$", "", str(termo))
        return base if base in self._macro_cols() else None

    def _coef_html(self, fit) -> str:
        """Tabela de coeficientes com p-valor, significância e coerência de sinal."""
        cf = fit.coef_frame()
        if cf.empty:
            return ("<div class='satui-legend'>Modelo sem coeficientes estimados "
                    "(referência ingênua) — compare-o pelas métricas fora da amostra.</div>")
        tab = cf.reset_index().rename(columns={"index": "termo", "coef": "coeficiente",
                                               "std_err": "erro-padrão", "t": "t",
                                               "p_valor": "p-valor"})
        if "p-valor" in tab.columns:
            tab["significância"] = [self._estrelas(p) for p in tab["p-valor"]]
        esperados = self.expected_signs()
        coer = []
        for termo, coef in zip(tab["termo"], tab["coeficiente"]):
            var = self._termo_variavel(termo)
            esperado = esperados.get(var) if var else None
            if esperado is None:
                coer.append("—")
            elif np.sign(float(coef)) == np.sign(esperado):
                coer.append(f"✓ esperado ({'+' if esperado > 0 else '−'})")
            else:
                coer.append(f"✗ invertido (esperava {'+' if esperado > 0 else '−'})")
        tab["coerência"] = coer
        return self._df_html(
            tab, max_height="420px",
            color_map={"p-valor": self._css_pvalor, "coerência": self._css_coerencia})

    # ==================================================================
    # Progresso e cronômetro (ações longas)
    # ==================================================================
    @staticmethod
    def _fmt_dur(segundos) -> str:
        """Duração legível: ``"12.4s"`` ou ``"3min 07s"``."""
        s = float(segundos)
        if s < 60:
            return f"{s:.1f}s"
        m, r = divmod(int(round(s)), 60)
        return f"{m}min {r:02d}s"

    def _render_progress(self, steps, widget, titulo="Progresso"):
        """Tabela de progresso por etapa (mesmo desenho das demais UIs da casa)."""
        if not steps:
            widget.value = ""
            return
        icone = {"run": "⏳", "ok": "✅", "err": "❌", "skip": "➖"}
        cor = {"run": "var(--warn-ink)", "ok": "var(--ok-ink)", "err": "var(--bad-ink)",
               "skip": "var(--muted)"}
        rot = {"run": "em andamento…", "ok": "concluída", "err": "erro", "skip": "pulada"}
        trs = ""
        for r in steps:
            st = r.get("status", "run")
            trs += (f"<tr><td>{icone.get(st, '')}</td><td>{r['label']}</td>"
                    f"<td style='color:{cor.get(st, 'var(--ink)')};font-weight:600'>"
                    f"{rot.get(st, st)}</td>"
                    f"<td style='color:var(--muted)'>{r.get('detail', '')}</td></tr>")
        widget.value = (
            f"<div class='satui-legend' style='margin-top:6px'>{titulo}</div>"
            "<table class='satui-prog'><thead><tr><th></th><th>Etapa</th><th>Status</th>"
            f"<th>Detalhe</th></tr></thead><tbody>{trs}</tbody></table>")

    def _prog(self, steps, widget, titulo, key, label, status, detail=""):
        """Cria/atualiza a linha ``key`` de uma tabela de progresso e a redesenha.

        Uma única implementação para as três ações longas com etapas (busca,
        backtest e estudo completo) — cada uma passa a **sua** lista e o seu
        widget.
        """
        for row in steps:
            if row["key"] == key:
                row["status"] = status
                if detail:
                    row["detail"] = detail
                break
        else:
            steps.append({"key": key, "label": label, "status": status, "detail": detail})
        self._render_progress(steps, widget, titulo)

    def _prog_erro(self, steps, widget, titulo, exc):
        """Marca como **erro** a etapa que estava em andamento."""
        for row in reversed(steps):
            if row.get("status") == "run":
                row["status"] = "err"
                row["detail"] = type(exc).__name__
                break
        self._render_progress(steps, widget, titulo)

    def _search_prog(self, key, label, status, detail=""):
        """Cria/atualiza a linha ``key`` da tabela de progresso da busca."""
        self._prog(self._search_steps, self.out_search_progress, "Progresso da busca",
                   key, label, status, detail)

    def _search_prog_erro(self, exc):
        """Marca como **erro** a etapa da busca que estava em andamento."""
        self._prog_erro(self._search_steps, self.out_search_progress,
                        "Progresso da busca", exc)

    @contextmanager
    def _cronometro(self, alvo, rotulo="processando"):
        """Mostra o **tempo decorrido** em ``alvo`` enquanto o bloco roda.

        A busca é síncrona (bloqueia a célula) e :func:`~..selection.search` não
        expõe callback de progresso — um cronômetro em *thread* daemon é o sinal de
        vida possível: sem ele a interface parece travada por minutos. Devolve uma
        função que dá os segundos decorridos.
        """
        import threading
        import time

        ini = time.monotonic()
        parar = threading.Event()

        def _pinta(txt):
            with suppress(Exception):
                alvo.value = txt

        def _loop():
            while not parar.wait(1.0):
                _pinta(f"<div class='satui-legend'><i>⏳ {rotulo} — "
                       f"{self._fmt_dur(time.monotonic() - ini)} decorridos…</i></div>")

        _pinta(f"<div class='satui-legend'><i>⏳ {rotulo} — iniciando…</i></div>")
        th = threading.Thread(target=_loop, daemon=True)
        th.start()
        try:
            yield lambda: time.monotonic() - ini
        finally:
            parar.set()
            with suppress(Exception):
                th.join(timeout=2.0)
            _pinta("")

    def _on_tab_change(self, change):
        """Reage à troca de aba: cada aba depende de controles de outra.

        **Seleção** recalcula o tamanho da grade, **Cenários** repõe o aviso do que
        falta para projetar, **Backtest** redimensiona as janelas e **Exportar**
        relê o estado da sessão.
        """
        with suppress(Exception):
            idx = change.get("new")
            if idx == 2:
                self._render_grid_info()
            elif idx == 4:
                self._render_scen_notice()
            elif idx == 5:
                self._render_bt_info()
            elif idx == 6:
                self._render_export_estado()

    # ==================================================================
    # Aba Seleção — busca champion-challenger
    # ==================================================================
    def _build_tab_selecao(self):
        """Cartões da aba **Seleção** (devolve a tupla de filhos do VBox)."""
        # --- card: tamanho da grade -----------------------------------------
        self.btn_grid_size = W.Button(
            description="Conferir a grade", icon="calculator",
            layout=W.Layout(width="auto", min_width="180px"),
            tooltip="Conta quantas especificações a busca vai avaliar com os parâmetros "
                    "da aba Especificação — e quantos ajustes isso custa.")
        self.btn_grid_size.on_click(lambda b: self._render_grid_info())
        self.out_grid_info = W.HTML()
        card_grade = W.VBox([
            W.HTML("<div class='satui-h'>Tamanho da grade (confira antes de rodar)</div>"),
            W.HTML("<div class='satui-legend'>A busca é <b>cara</b>: cada especificação é "
                   "ajustada na amostra cheia e, se passar nos filtros duros, revalidada "
                   "<b>janela a janela</b> (walk-forward). O total cresce com o conjunto de "
                   "defasagens <b>elevado</b> ao número de variáveis — quando o aviso "
                   "aparecer, reduza <b>defasagens</b> ou <b>máx. variáveis</b> na aba "
                   "<b>Especificação</b> em vez de esperar.</div>"),
            self.btn_grid_size, self.out_grid_info,
        ])
        card_grade.add_class("satui-card")

        # --- card: rodar a busca --------------------------------------------
        self.cb_require_signs = W.Checkbox(
            value=True, indent=False,
            description="aplicar o filtro duro de sinal econômico")
        self.cb_include_bench = W.Checkbox(
            value=True, indent=False,
            description="incluir os benchmarks (ARIMA e ingênuos)")
        self.cb_cobertura = W.Checkbox(
            value=False, indent=False,
            description="medir a cobertura dos intervalos (bem mais lento)")
        self.sl_rank_n = W.BoundedIntText(
            value=20, min=1, max=200, description="linhas na tabela:",
            style={"description_width": "initial"}, layout=W.Layout(width="210px"))
        self.btn_search = W.Button(
            description="Rodar busca", icon="search", button_style="success",
            layout=W.Layout(width="auto", min_width="170px"),
            tooltip="Ajusta e valida toda a grade, aplica os filtros duros e ranqueia.")
        self.btn_search.on_click(self._on_search)
        self.out_search_status = W.HTML()
        self.out_search_progress = W.HTML()
        self.out_search_timer = W.HTML()
        self.out_search_resumo = W.HTML()
        card_busca = W.VBox([
            W.HTML("<div class='satui-h'>Busca champion-challenger</div>"),
            W.HTML("<div class='satui-legend'>Para cada especificação: ajuste na amostra "
                   "cheia (AIC/BIC, VIF, sinais), <b>filtros duros</b> (sinal econômico "
                   "coerente e VIF sob o teto) e, para as que passam, <b>validação "
                   "walk-forward</b>. O ranking usa o critério escolhido na aba "
                   "<b>Especificação</b>. Desligar o filtro de sinal serve para "
                   "diagnóstico — não para escolher o modelo.</div>"),
            W.HBox([self.cb_require_signs, self.cb_include_bench],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.cb_cobertura, self.sl_rank_n, self.btn_search],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_search_status, self.out_search_progress, self.out_search_timer,
            self.out_search_resumo,
        ])
        card_busca.add_class("satui-card")

        # --- card: ranking ---------------------------------------------------
        self.out_search_rank = W.HTML()
        card_rank = W.VBox([
            W.HTML("<div class='satui-h'>Ranking das qualificadas (e dos benchmarks)</div>"),
            W.HTML("<div class='satui-legend'>★ marca a <b>campeã</b> pelo critério. As "
                   "linhas de <b>benchmark</b> entram na mesma tabela de propósito: um "
                   "modelo macro que não bate o ARIMA e os ingênuos <b>fora da amostra</b> "
                   "ainda não está pronto, por melhor que seja o R². <b>vs ARIMA</b> abaixo "
                   "de 1 significa erro menor que o do ARIMA.</div>"),
            self.out_search_rank,
        ])
        card_rank.add_class("satui-card")

        # --- card: descartadas ------------------------------------------------
        self.out_search_desq = W.HTML()
        card_desq = W.VBox([
            W.HTML("<div class='satui-h'>Descartadas — e por quê</div>"),
            W.HTML("<div class='satui-help'><div class='ttl'>Os dois filtros duros</div>"
                   "<b>Sinal econômico invertido</b>: o efeito líquido da variável saiu com "
                   "o sinal contrário ao declarado. É <b>desqualificador</b> mesmo com "
                   "ajuste excelente — sob estresse a projeção iria na direção errada.<br>"
                   "<b>VIF acima do teto</b>: duas macro explicam a mesma coisa; os "
                   "coeficientes ficam instáveis e o sinal vira por acaso amostral.<br>"
                   "Ver o que <b>saiu</b> importa tanto quanto ver a campeã: se quase tudo "
                   "caiu por sinal, o problema costuma estar no sinal declarado ou na "
                   "transformação da variável, não nas especificações.</div>"),
            self.out_search_desq,
        ])
        card_desq.add_class("satui-card")

        # --- card: escolha manual ----------------------------------------------
        self.dd_pick_spec = W.Dropdown(
            options=[], description="Especificação:",
            style={"description_width": "initial"}, layout=W.Layout(width="98%"))
        self.btn_pick_fit = W.Button(
            description="Adotar e ajustar", icon="check", button_style="primary",
            layout=W.Layout(width="auto", min_width="185px"),
            tooltip="Ajusta a especificação escolhida na amostra cheia e a torna o modelo "
                    "vigente (Diagnóstico, Cenários e Exportar passam a usá-la).")
        self.btn_pick_fit.on_click(self._on_pick_fit)
        self.out_pick_status = W.HTML()
        self.out_pick_info = W.HTML()
        card_pick = W.VBox([
            W.HTML("<div class='satui-h'>Adotar uma especificação (você pode discordar)</div>"),
            W.HTML("<div class='satui-legend'>A campeã já vem adotada como <b>modelo "
                   "vigente</b>, mas o critério é uma régua, não um veredito: diferenças "
                   "mínimas de RMSE não decidem nada e a especificação mais defensável "
                   "costuma ser a mais <b>parcimoniosa</b> entre as equivalentes. Escolha "
                   "outra aqui — inclusive uma <b>descartada</b>, se você discorda do sinal "
                   "declarado — e ela passa a valer nas abas seguintes.</div>"),
            self.dd_pick_spec,
            W.HBox([self.btn_pick_fit], layout=W.Layout(align_items="center")),
            self.out_pick_status, self.out_pick_info,
        ])
        card_pick.add_class("satui-card")

        # --- card: benchmarks + Diebold-Mariano --------------------------------
        self.btn_dm = W.Button(
            description="Comparar com os benchmarks", icon="balance-scale",
            layout=W.Layout(width="auto", min_width="240px"),
            tooltip="Revalida a especificação adotada e testa, com Diebold-Mariano, se a "
                    "vantagem sobre cada benchmark é estatisticamente distinguível.")
        self.btn_dm.on_click(self._on_dm)
        self.out_dm_status = W.HTML()
        self.out_dm_timer = W.HTML()
        self.out_dm = W.HTML()
        card_bench = W.VBox([
            W.HTML("<div class='satui-h'>Vantagem sobre os benchmarks é real? "
                   "(Diebold-Mariano)</div>"),
            W.HTML("<div class='satui-help'><div class='ttl'>Como ler</div>"
                   "O teste compara o <b>diferencial de perda</b> das duas séries de erro "
                   "fora da amostra. <b>H0: mesma acurácia.</b> p pequeno ⇒ a diferença "
                   "<b>não</b> é ruído amostral. RMSE menor <b>sem</b> p pequeno é o caso "
                   "mais comum em série curta: a campeã parece melhor, mas você não "
                   "consegue distingui-la do passeio aleatório — e aí a parcimônia decide."
                   "</div>"),
            W.HBox([self.btn_dm], layout=W.Layout(align_items="center")),
            self.out_dm_status, self.out_dm_timer, self.out_dm,
        ])
        card_bench.add_class("satui-card")

        return (card_grade, card_busca, card_rank, card_desq, card_pick, card_bench)

    # ------------------------------------------------------------------ grade
    def _grid_params(self) -> dict:
        """Parâmetros da grade lidos da aba **Especificação** (erro legível)."""
        return dict(
            lag_set=tuple(self._parse_ints(self.tx_lag_set.value, "defasagens")),
            ar_orders=tuple(self._parse_ints(self.tx_ar_orders.value, "ordens AR")),
            max_vars=int(self.sl_max_vars.value),
            seasonal=bool(self.cb_seasonal.value),
            seasonal_period=int(self.sl_seasonal_period.value),
            max_specs=int(self.sl_max_specs.value),
        )

    def _grid_size(self, params=None) -> dict:
        """Conta a grade **sem construí-la** e estima o custo da busca."""
        p = params or self._grid_params()
        cands = self.candidates()
        kmax = min(int(p["max_vars"]), len(cands))
        total = sum(math.comb(len(cands), k) * (len(p["lag_set"]) ** k) * len(p["ar_orders"])
                    for k in range(1, kmax + 1))
        efetivo = min(int(total), int(p["max_specs"]))
        n = len(self.series) if self.series is not None else 0
        horizonte = int(self.sl_horizon.value)
        min_train = int(self.sl_min_train.value) or max(24, n // 2)
        janelas = max(0, n - horizonte + 1 - min_train)
        return {"candidatas": len(cands), "total": int(total), "efetivo": efetivo,
                "janelas": int(janelas), "min_train": int(min_train),
                "ajustes": int(efetivo * (1 + janelas))}

    def _render_grid_info(self):
        """Desenha o cartão de dimensionamento da grade (e devolve as contas)."""
        if self.series is None:
            self.out_grid_info.value = (
                "<div class='satui-legend'>Carregue uma série na aba <b>Série</b> antes de "
                "dimensionar a grade.</div>")
            return None
        try:
            params = self._grid_params()
            info = self._grid_size(params)
        except ValueError as exc:
            self.out_grid_info.value = f"<div class='satui-notice'>{exc}</div>"
            return None
        if not info["candidatas"]:
            self.out_grid_info.value = (
                "<div class='satui-notice'>Nenhuma variável marcada como <b>candidata</b> na "
                "aba <b>Especificação</b> — a grade ficaria vazia.</div>")
            return info
        html = self._metric_tiles({
            "candidatas": info["candidatas"],
            "defasagens": len(params["lag_set"]),
            "máx. variáveis": int(params["max_vars"]),
            "grade": info["total"],
            "a avaliar": info["efetivo"],
            "janelas walk-forward": info["janelas"],
            "ajustes (teto)": info["ajustes"],
        })
        avisos = []
        if info["janelas"] <= 0:
            avisos.append(
                "❌ <b>Não sobra janela de validação</b>: com mínimo de treino "
                f"{info['min_train']} e horizonte {int(self.sl_horizon.value)} não há "
                f"origem possível em {len(self.series)} observações. Reduza o horizonte ou "
                "o mínimo de treino na aba <b>Especificação</b>.")
        if info["total"] > info["efetivo"]:
            avisos.append(
                f"⚠️ A grade tem <b>{info['total']}</b> especificações e o teto é "
                f"<b>{info['efetivo']}</b>: as demais <b>não serão avaliadas</b> (o corte é "
                "por ordem de geração, não por qualidade). Aumente <i>máx. especificações</i> "
                "ou reduza <i>defasagens</i>/<i>máx. variáveis</i>.")
        if info["ajustes"] > 20000:
            avisos.append(
                f"⚠️ Cerca de <b>{info['ajustes']:,}</b> ajustes no pior caso — isso são "
                "<b>muitos minutos</b>. Reduza o conjunto de defasagens (cada defasagem a "
                "mais multiplica a grade) ou o número máximo de variáveis.".replace(",", "."))
        elif info["ajustes"] > 5000:
            avisos.append(
                f"ℹ️ Cerca de <b>{info['ajustes']:,}</b> ajustes no pior caso — conte alguns "
                "minutos. Ligue <i>Manter cluster ativo</i> se estiver no Databricks."
                .replace(",", "."))
        if self.cb_cobertura.value:
            avisos.append("ℹ️ A <b>cobertura dos intervalos</b> está ligada: cada janela "
                          "simula trajetórias além do ajuste — conte várias vezes o tempo "
                          "acima.")
        html += "".join(f"<div class='satui-notice' style='margin-top:8px'>{a}</div>"
                        for a in avisos)
        self.out_grid_info.value = html
        return info

    def _clear_selecao_outputs(self):
        """Zera as saídas da aba Seleção (dados novos ⇒ busca antiga sem sentido)."""
        for w in ("out_search_status", "out_search_progress", "out_search_timer",
                  "out_search_resumo", "out_search_rank", "out_search_desq",
                  "out_pick_status", "out_pick_info", "out_dm_status", "out_dm",
                  "out_grid_info"):
            widget = getattr(self, w, None)
            if widget is not None:
                widget.value = ""
        dd = getattr(self, "dd_pick_spec", None)
        if dd is not None:
            dd.options = []

    # ------------------------------------------------------------------ modelo da busca
    def _search_model(self):
        """``(classe, kwargs, aviso)`` do modelo candidato da busca.

        Os modelos que não são candidatos (ARIMA e os ingênuos) entram no estudo
        como **benchmark**: a grade é varrida com ARDL, a mesma convenção de
        :meth:`to_config`.
        """
        key = self.dd_model.value
        registry = MODELOS[key]["registry"]
        aviso = ""
        if registry is None:
            aviso = (f"o modelo '{key}' não é candidato da busca (entra como benchmark) — "
                     "a grade foi varrida com ARDL.")
            registry = "ardl"
        cls = _E.config.MODEL_REGISTRY[registry]
        if registry == "vasicek":
            kwargs = {"rho": float(self.fl_rho.value),
                      "pd_ttc": (None if self.cb_ttc_auto.value else float(self.fl_pd_ttc.value)),
                      "cov_type": self.dd_cov.value}
        elif registry == "ardl":
            kwargs = {"cov_type": self.dd_cov.value}
        else:
            kwargs = {}
        return cls, kwargs, aviso

    # ------------------------------------------------------------------ busca
    def _on_search(self, b):
        if self.series is None:
            self.out_search_status.value = (
                "<div class='satui-notice'>Sem série carregada — traga os dados na aba "
                "<b>Série</b> (ou carregue o estudo de referência).</div>")
            return
        cands = self.candidates()
        if not cands:
            self.out_search_status.value = (
                "<div class='satui-notice'>Marque ao menos uma variável <b>candidata</b> na "
                "aba <b>Especificação</b>.</div>")
            return
        if self.macro is None:
            self.out_search_status.value = (
                "<div class='satui-notice'>Há candidatas marcadas mas nenhuma macro "
                "carregada.</div>")
            return
        try:
            params = self._grid_params()
        except ValueError as exc:
            self.out_search_status.value = f"<div class='satui-notice'>{exc}</div>"
            return
        info = self._grid_size(params)
        self._render_grid_info()
        if info["janelas"] <= 0:
            self.out_search_status.value = (
                "<div class='satui-notice'>Sem janela de validação fora da amostra: reduza o "
                "<b>horizonte</b> ou o <b>mínimo de treino</b> na aba Especificação.</div>")
            return

        cls, kwargs, aviso = self._search_model()
        if aviso:
            self._log(f"[busca] {aviso}")
        band_sims = 200 if self.cb_cobertura.value else 0
        horizonte = int(self.sl_horizon.value)
        min_train = int(self.sl_min_train.value) or None
        self._search_steps = []
        self.out_search_status.value = ""
        for w in (self.out_search_resumo, self.out_search_rank, self.out_search_desq,
                  self.out_dm, self.out_dm_status, self.out_pick_info, self.out_pick_status):
            w.value = ""

        with self._busy(self.btn_search, self.btn_pick_fit, self.btn_dm, self.btn_grid_size,
                        self.btn_fit_now), \
                self._cronometro(self.out_search_timer, "busca em andamento") as decorrido:
            try:
                self._search_prog("grade", "Montar a grade de especificações", "run")
                import warnings

                with warnings.catch_warnings(record=True) as capturados:
                    warnings.simplefilter("always")
                    grid = _E.selection.make_grid(
                        cands, lag_set=params["lag_set"], min_vars=1,
                        max_vars=params["max_vars"], ar_orders=params["ar_orders"],
                        link=self.dd_link.value, expected_signs=self.expected_signs(),
                        seasonal=params["seasonal"],
                        seasonal_period=params["seasonal_period"],
                        max_specs=params["max_specs"])
                for a in capturados:
                    self._log(f"[busca] {a.message}")
                self._spec_por_desc = {s.describe(): s for s in grid}
                self._search_prog("grade", "Montar a grade de especificações", "ok",
                                  f"{len(grid)} especificações")
                rot_aval = (f"Avaliar {len(grid)} especificações (ajuste + walk-forward "
                            f"de {info['janelas']} janelas)")
                self._search_prog("aval", rot_aval, "run")
                if self.cb_include_bench.value:
                    self._search_prog("bench", "Benchmarks: ARIMA e ingênuos", "run")
                res = _E.selection.search(
                    self.series, self.macro, grid, model_cls=cls, model_kwargs=kwargs,
                    expected_signs=self.expected_signs(), link=self.dd_link.value,
                    horizon=horizonte, min_train=min_train,
                    criterion=self.dd_criterion.value, vif_max=float(self.fl_vif_max.value),
                    require_signs=bool(self.cb_require_signs.value),
                    include_benchmarks=bool(self.cb_include_bench.value),
                    band_sims=band_sims)
            except Exception as exc:  # noqa: BLE001
                self._search_prog_erro(exc)
                self.out_search_status.value = (
                    "<div class='satui-notice'>✗ A busca falhou — veja o <b>Console</b> "
                    f"(rodapé): {type(exc).__name__}.</div>")
                self._log(f"[busca] ERRO: {type(exc).__name__}: {exc}")
                return
            secs = decorrido()

        self.search_ = res
        self._search_secs = secs
        rk = res.ranking
        n_qual = int((rk["status"] == "qualificado").sum()) if len(rk) else 0
        self._search_prog("aval", rot_aval, "ok",
                          f"{n_qual} qualificada(s) de {len(grid)} · {self._fmt_dur(secs)}")
        if self.cb_include_bench.value:
            self._search_prog("bench", "Benchmarks: ARIMA e ingênuos", "ok",
                              f"{len(res.benchmarks)} referência(s)")
        campea = res.best_spec.describe() if res.best_spec is not None else "—"
        self._search_prog("rank", "Ranking e escolha da campeã", "ok", campea)

        # a campeã já entra como modelo vigente (o seletor abaixo permite discordar)
        if res.best is not None and res.best.result is not None:
            self.model_, self.fit_ = res.best, res.best.result
            self.selected_spec_ = res.best_spec
            self._clear_dirty()
            self._render_fit()
        self._render_search()
        self._refresh_bar()
        self._log(f"[busca] {len(grid)} especificações avaliadas em {self._fmt_dur(secs)} · "
                  f"{n_qual} qualificada(s) · campeã: {campea}")

    # ------------------------------------------------------------------ ranking
    @staticmethod
    def _motivo_descarte(status) -> tuple:
        """``(situação, motivo)`` legíveis a partir do ``status`` do ranking."""
        s = str(status)
        if s == "qualificado":
            return "qualificada", "passou nos filtros duros"
        if s == "benchmark":
            return "benchmark", "referência (não é candidata)"
        if s.startswith("reprovado: sinal"):
            variaveis = re.findall(r"'([^']+)'", s)
            alvo = ", ".join(variaveis) if variaveis else "—"
            return "descartada", f"sinal econômico invertido em {alvo}"
        if s.startswith("reprovado: VIF"):
            m = re.search(r"VIF=([\d.]+)>([\d.]+)", s)
            if m:
                return "descartada", (f"colinearidade: VIF {m.group(1)} acima do teto "
                                      f"{m.group(2)}")
            return "descartada", "colinearidade (VIF acima do teto)"
        if s.startswith("erro"):
            return "erro", s.split(":", 1)[1].strip() if ":" in s else s
        return "—", s

    @staticmethod
    def _rotulo_sinais(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return "✓ coerentes" if bool(v) else "✗ invertido"

    @staticmethod
    def _css_situacao(v):
        """Situação da especificação no ranking (qualificada/descartada/erro)."""
        s = _sem_acento(v)
        if s.startswith("qualificada"):
            return "color:var(--ok-tx);background-color:var(--ok-bg);font-weight:600"
        if s.startswith("descartada"):
            return "color:var(--bad-tx);background-color:var(--bad-bg);font-weight:600"
        if s.startswith("erro"):
            return "color:var(--warn-tx);background-color:var(--warn-bg);font-weight:600"
        return "color:var(--muted)"

    @staticmethod
    def _css_motivo(v):
        """Motivo do descarte: sinal invertido (vermelho) × VIF (âmbar)."""
        s = _sem_acento(v)
        if "sinal" in s:
            return "color:var(--bad-tx);background-color:var(--bad-bg);font-weight:600"
        if "colinearidade" in s or "vif" in s:
            return "color:var(--warn-tx);background-color:var(--warn-bg);font-weight:600"
        if "passou" in s:
            return "color:var(--ok-tx)"
        return "color:var(--muted)"

    def _css_vif_factory(self, teto):
        """Colore o VIF contra o teto vigente (fecha sobre ``teto``)."""
        meio = max(2.0, float(teto) / 2.0)

        def _css(v):
            try:
                x = float(v)
            except (TypeError, ValueError):
                return ""
            if x != x:
                return "color:var(--muted)"
            if x > float(teto):
                return "color:var(--bad-tx);background-color:var(--bad-bg);font-weight:600"
            if x > meio:
                return "color:var(--warn-tx);font-weight:600"
            return "color:var(--ok-tx)"

        return _css

    def _linhas_ranking(self, df, champ_desc, n):
        """Converte ``n`` linhas do ranking cru em linhas legíveis da tela."""
        linhas = []
        for i, (_, r) in enumerate(df.head(int(n)).iterrows(), start=1):
            desc = str(r["modelo"])
            spec = self._spec_por_desc.get(desc)
            campea = champ_desc is not None and desc == champ_desc
            linhas.append({
                "#": ("★ " if campea else "") + str(i),
                "especificação": desc,
                "variáveis": ", ".join(spec.exog) if spec else "—",
                "defasagens": (", ".join(str(list(l)[0]) for l in spec.exog.values())
                               if spec and spec.exog else "—"),
                "AR": (int(spec.ar) if spec else np.nan),
                "AIC": r.get("AIC", np.nan), "BIC": r.get("BIC", np.nan),
                "RMSE fora": r.get("oos_rmse", np.nan),
                "vs ARIMA": r.get("vs_arima", np.nan),
                "VIF máx.": r.get("max_vif", np.nan),
                "cobertura": r.get("cobertura", np.nan),
                "sinais": self._rotulo_sinais(r.get("sinais_ok")),
                "situação": r["situação"],
                "motivo": r["motivo"],
            })
        return pd.DataFrame(linhas)

    def _render_search(self):
        """Resumo, ranking, descartadas e seletor da última busca."""
        res = self.search_
        if res is None:
            return
        rk = res.ranking.copy()
        if rk.empty:
            self.out_search_status.value = (
                "<div class='satui-notice'>A busca não avaliou nenhuma especificação.</div>")
            return
        situ, motivo = zip(*[self._motivo_descarte(s) for s in rk["status"]])
        rk["situação"], rk["motivo"] = list(situ), list(motivo)
        champ = res.best_spec.describe() if res.best_spec is not None else None
        n_top = int(self.sl_rank_n.value)
        teto = float(self.fl_vif_max.value)
        qual = rk[rk["situação"].isin(("qualificada", "benchmark"))]
        desq = rk[~rk["situação"].isin(("qualificada", "benchmark"))]

        # --- resumo -------------------------------------------------------
        n_sinal = int(desq["motivo"].str.startswith("sinal").sum()) if len(desq) else 0
        n_vif = int(desq["motivo"].str.startswith("colinearidade").sum()) if len(desq) else 0
        n_erro = int((rk["situação"] == "erro").sum())
        tiles = {
            "avaliadas": int((rk["situação"] != "benchmark").sum()),
            "qualificadas": int((rk["situação"] == "qualificada").sum()),
            "sinal invertido": n_sinal,
            "VIF alto": n_vif,
            "erro no ajuste": n_erro,
            "benchmarks": int((rk["situação"] == "benchmark").sum()),
        }
        if self._search_secs is not None:
            tiles["tempo"] = self._fmt_dur(self._search_secs)
        self.out_search_resumo.value = self._metric_tiles(tiles)

        if champ is None:
            self.out_search_status.value = (
                "<div class='satui-notice'>⚠️ <b>Nenhuma especificação qualificada</b> — "
                "todas caíram nos filtros duros. Revise os <b>sinais esperados</b> (é comum "
                "declarar o sinal de uma variável já invertida, como um índice de "
                "confiança), afrouxe o <b>teto de VIF</b> ou reduza o número de variáveis "
                "por especificação. A tabela de descartadas abaixo diz exatamente o que "
                "aconteceu.</div>")
        else:
            self.out_search_status.value = (
                "<div class='satui-legend' style='color:var(--ok-ink)'>✓ Campeã pelo critério "
                f"<b>{self.dd_criterion.value}</b>: <code>{champ}</code> — já adotada como "
                "modelo vigente.</div>")

        # --- ranking ------------------------------------------------------
        tab = self._linhas_ranking(qual, champ, n_top)
        cols = ["#", "especificação", "variáveis", "defasagens", "AR", "AIC", "BIC",
                "RMSE fora", "vs ARIMA", "VIF máx.", "sinais", "situação"]
        if len(tab) and tab["cobertura"].notna().any():
            cols.insert(-2, "cobertura")
        if len(tab):
            self.out_search_rank.value = self._df_html(
                tab[cols], max_height="420px",
                color_map={"situação": self._css_situacao, "sinais": self._css_coerencia,
                           "VIF máx.": self._css_vif_factory(teto)},
                fmt_cols={"AIC": "{:.1f}", "BIC": "{:.1f}", "RMSE fora": "{:.5f}",
                          "vs ARIMA": "{:.2f}", "VIF máx.": "{:.2f}"},
                pct_cols=["cobertura"], precision=4)
            if len(qual) > len(tab):
                self.out_search_rank.value += (
                    f"<div class='satui-legend'>Mostrando {len(tab)} de {len(qual)} linhas "
                    "— aumente <i>linhas na tabela</i> para ver mais.</div>")
        else:
            self.out_search_rank.value = (
                "<div class='satui-legend'>Nenhuma especificação qualificada.</div>")

        # --- descartadas ---------------------------------------------------
        if len(desq):
            tab_d = self._linhas_ranking(desq, None, n_top)
            cols_d = ["#", "especificação", "variáveis", "defasagens", "AIC", "VIF máx.",
                      "sinais", "motivo"]
            self.out_search_desq.value = self._df_html(
                tab_d[cols_d], max_height="360px",
                color_map={"motivo": self._css_motivo, "sinais": self._css_coerencia,
                           "VIF máx.": self._css_vif_factory(teto)},
                fmt_cols={"AIC": "{:.1f}", "VIF máx.": "{:.2f}"}, precision=4)
            if len(desq) > len(tab_d):
                self.out_search_desq.value += (
                    f"<div class='satui-legend'>Mostrando {len(tab_d)} de {len(desq)} "
                    "descartadas.</div>")
        else:
            self.out_search_desq.value = (
                "<div class='satui-legend' style='color:var(--ok-ink)'>Nenhuma especificação "
                "foi descartada pelos filtros duros.</div>")

        # --- seletor -------------------------------------------------------
        opcoes = []
        for _, r in qual.head(n_top).iterrows():
            desc = str(r["modelo"])
            if desc not in self._spec_por_desc:      # linha de benchmark
                continue
            marca = "★ campeã · " if desc == champ else ""
            rmse = r.get("oos_rmse", np.nan)
            rot = f"{marca}{desc}"
            if pd.notna(rmse):
                rot += f" · RMSE {float(rmse):.5f}"
            opcoes.append((rot, desc))
        for _, r in desq.head(max(5, n_top // 2)).iterrows():
            desc = str(r["modelo"])
            if desc not in self._spec_por_desc:
                continue
            opcoes.append((f"⚠ descartada · {desc} · {r['motivo']}", desc))
        self.dd_pick_spec.options = opcoes
        if champ is not None and any(v == champ for _, v in opcoes):
            self.dd_pick_spec.value = champ

    # ------------------------------------------------------------------ escolha manual
    def _on_pick_fit(self, b):
        desc = self.dd_pick_spec.value
        spec = self._spec_por_desc.get(desc) if desc else None
        if spec is None:
            self.out_pick_status.value = (
                "<div class='satui-notice'>Rode a busca e escolha uma especificação do "
                "ranking antes.</div>")
            return
        cls, kwargs, _aviso = self._search_model()
        with self._busy(self.btn_pick_fit, self.btn_search, self.btn_dm,
                        status=self.out_pick_status, msg="ajustando a especificação escolhida…"):
            try:
                modelo = cls(self.series, self.macro, spec, **kwargs)
                fit = modelo.fit()
            except Exception as exc:  # noqa: BLE001
                self.out_pick_status.value = (
                    f"<div class='satui-notice'>Não foi possível ajustar: {exc}</div>")
                self._log(f"[seleção] erro ao ajustar a especificação escolhida: {exc}")
                return
        self.model_, self.fit_ = modelo, fit
        self.selected_spec_ = spec
        self._clear_dirty()
        self._render_fit()
        self._refresh_bar()
        self.out_pick_status.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ Especificação adotada "
            f"como <b>modelo vigente</b>: <code>{spec.describe()}</code>.</div>")
        self.out_pick_info.value = (
            self._metric_tiles({k: v for k, v in fit.metrics().items()
                                if k in ("nobs", "n_params", "AIC", "BIC", "R2", "sigma")})
            + self._coef_html(fit)
            + "<div class='satui-legend'>A aba <b>Especificação</b> continua descrevendo a "
              "<i>sua</i> especificação (candidatas e defasagens da matriz de sinais): "
              "<i>Ajustar agora</i> lá refaz aquela, não esta. As abas <b>Diagnóstico</b>, "
              "<b>Cenários</b> e <b>Exportar</b> usam a adotada aqui.</div>")
        aic = f" · AIC {fit.aic:.1f}" if fit.aic is not None else ""
        self._log(f"[seleção] especificação adotada: {spec.describe()}{aic}")

    # ------------------------------------------------------------------ benchmarks / DM
    def _on_dm(self, b):
        res = self.search_
        if res is None or not res.benchmarks:
            self.out_dm_status.value = (
                "<div class='satui-notice'>Rode a busca com <b>incluir os benchmarks</b> "
                "ligado antes de comparar.</div>")
            return
        spec = self.selected_spec_
        if spec is None:
            self.out_dm_status.value = (
                "<div class='satui-notice'>Adote uma especificação no seletor acima "
                "antes.</div>")
            return
        cls, kwargs, _aviso = self._search_model()
        horizonte = int(self.sl_horizon.value)
        min_train = int(self.sl_min_train.value) or None
        with self._busy(self.btn_dm, self.btn_search, self.btn_pick_fit), \
                self._cronometro(self.out_dm_timer,
                                 "revalidando a especificação adotada janela a janela"):
            try:
                wf = _E.selection.walk_forward(
                    lambda s, m: cls(s, m, spec, **kwargs), self.series, self.macro,
                    min_train=min_train, horizon=horizonte)
            except Exception as exc:  # noqa: BLE001
                self.out_dm_status.value = (
                    f"<div class='satui-notice'>Não foi possível revalidar: {exc}</div>")
                self._log(f"[seleção] erro no walk-forward da adotada: {exc}")
                return
        rmse_ad = float(wf.get("rmse", np.nan))
        linhas = []
        for nome, bwf in res.benchmarks.items():
            dm = _E.selection.diebold_mariano(wf["errors"], bwf["errors"], h=horizonte)
            rmse_b = float(bwf.get("rmse", np.nan))
            razao = rmse_ad / rmse_b if np.isfinite(rmse_b) and rmse_b > 0 else np.nan
            p = dm["pvalue"]
            if not np.isfinite(p if p is not None else np.nan):
                veredito = "— amostra curta para o teste"
            elif np.isfinite(razao) and razao < 1.0:
                veredito = ("✓ adotada melhor e distinguível" if p <= 0.05
                            else "adotada melhor, mas indistinguível")
            elif np.isfinite(razao) and razao > 1.0:
                veredito = ("✗ a referência é melhor" if p <= 0.05
                            else "referência melhor, mas indistinguível")
            else:
                veredito = "empate"
            linhas.append({"referência": nome, "RMSE da referência": rmse_b,
                           "RMSE da adotada": rmse_ad, "razão adotada/ref.": razao,
                           "DM (estat.)": dm["stat"], "p-valor": dm["pvalue"],
                           "janelas": int(bwf.get("n_windows", 0)), "veredito": veredito})
        tab = pd.DataFrame(linhas).sort_values("RMSE da referência").reset_index(drop=True)
        self.compare_ = tab
        self.out_dm.value = self._df_html(
            tab, color_map={"veredito": self._css_coerencia, "p-valor": self._css_pvalor},
            fmt_cols={"RMSE da referência": "{:.5f}", "RMSE da adotada": "{:.5f}",
                      "razão adotada/ref.": "{:.3f}", "DM (estat.)": "{:.3f}"},
            precision=4)
        piores = [r for r in linhas
                  if pd.notna(r["razão adotada/ref."]) and r["razão adotada/ref."] >= 1.0]
        if piores:
            nomes = ", ".join(r["referência"] for r in piores)
            self.out_dm_status.value = (
                f"<div class='satui-notice'>⚠️ A especificação adotada <b>não supera</b> "
                f"{nomes} fora da amostra. Antes de seguir para a projeção, revise a "
                "especificação (defasagens, ordem AR, transformação) — um satélite que "
                "perde para o passeio aleatório não está capturando sinal macro.</div>")
        else:
            self.out_dm_status.value = (
                "<div class='satui-legend' style='color:var(--ok-ink)'>✓ A adotada tem erro "
                "menor que o de todas as referências fora da amostra.</div>")
        self._log(f"[seleção] comparação com {len(tab)} referência(s) · RMSE da adotada "
                  f"{rmse_ad:.5f}")

    # ==================================================================
    # Aba Diagnóstico — a bateria sobre o modelo vigente
    # ==================================================================
    #: blocos do placar: ``(rótulo, testes da bateria, o que a família mede)``
    _BLOCOS_DIAG = (
        ("Resíduo", ("Ljung-Box", "Breusch-Godfrey", "Durbin-Watson"),
         "autocorrelação sobrando"),
        ("Heterocedasticidade", ("Breusch-Pagan", "White", "ARCH-LM"),
         "variância do erro instável"),
        ("Normalidade", ("Jarque-Bera",), "forma da distribuição do erro"),
        ("Estabilidade", ("Chow", "Quandt-Andrews sup-F", "CUSUM"),
         "coeficientes mudando na amostra"),
        ("Colinearidade", ("VIF",), "macro redundante"),
    )

    def _build_tab_diagnostico(self):
        """Cartões da aba **Diagnóstico** (devolve a tupla de filhos do VBox)."""
        self.fl_alpha_diag = W.BoundedFloatText(
            value=0.05, min=0.001, max=0.20, step=0.01, description="nível α:",
            style={"description_width": "initial"}, layout=W.Layout(width="140px"))
        self.dd_chow_break = W.Dropdown(
            options=[], description="quebra do Chow:",
            style={"description_width": "initial"}, layout=W.Layout(width="250px"))
        self.cb_diag_plots = W.Checkbox(value=True, indent=False,
                                        description="desenhar os gráficos")
        self.btn_diag = W.Button(
            description="Rodar diagnóstico", icon="stethoscope", button_style="primary",
            layout=W.Layout(width="auto", min_width="200px"),
            tooltip="Roda a bateria completa sobre os resíduos do modelo vigente.")
        self.btn_diag.on_click(self._on_diag)
        self.out_diag_notice = W.HTML()
        self.out_diag_status = W.HTML()
        card_acao = W.VBox([
            W.HTML("<div class='satui-h'>Bateria de diagnóstico do modelo vigente</div>"),
            W.HTML("<div class='satui-legend'>O modelo vigente é o último ajustado — pela "
                   "aba <b>Especificação</b> ou adotado na aba <b>Seleção</b>. A bateria "
                   "olha o que o ajuste <b>deixou</b> no resíduo: se sobrou dinâmica, se a "
                   "variância é instável, se os coeficientes mudaram no meio da amostra. "
                   "Nenhum teste sozinho reprova um modelo — a leitura é conjunta, e a "
                   "<b>estabilidade</b> é a que mais pesa para projetar.</div>"),
            W.HBox([self.fl_alpha_diag, self.dd_chow_break, self.cb_diag_plots,
                    self.btn_diag],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_diag_notice, self.out_diag_status,
        ])
        card_acao.add_class("satui-card")

        self.out_diag_placar = W.HTML()
        card_placar = W.VBox([
            W.HTML("<div class='satui-h'>Placar por família de teste</div>"),
            W.HTML("<div class='satui-legend'>Cada bloco mostra o <b>pior</b> veredito da "
                   "família e a evidência (estatística e p-valor). ⚠️ significa "
                   "<b>inconclusivo</b> — o teste não rodou ou não tem graus de liberdade —, "
                   "não aprovação.</div>"),
            self.out_diag_placar,
        ])
        card_placar.add_class("satui-card")

        self.out_diag_tabela = W.HTML()
        card_tab = W.VBox([
            W.HTML("<div class='satui-h'>Todos os testes</div>"),
            W.HTML("<div class='satui-legend'>A coluna <b>H0</b> é a hipótese nula: "
                   "<b>p pequeno rejeita a nula</b>. Como a nula muda de teste para teste, "
                   "a coluna <b>ok</b> já traduz tudo para a mesma direção — ✓ é o "
                   "resultado <b>desejável</b> para um bom modelo.</div>"),
            self.out_diag_tabela,
        ])
        card_tab.add_class("satui-card")

        self.out_diag_vif = W.HTML()
        card_vif = W.VBox([
            W.HTML("<div class='satui-h'>Colinearidade das macro (VIF)</div>"),
            self.out_diag_vif,
        ])
        card_vif.add_class("satui-card")

        self.out_diag_leitura = W.HTML()
        card_leitura = W.VBox([
            W.HTML("<div class='satui-h'>O que fazer</div>"),
            self.out_diag_leitura,
        ])
        card_leitura.add_class("satui-card")

        self.out_diag_plot_fit = W.HTML()
        self.out_diag_plot_resid = W.HTML()
        card_plots = W.VBox([
            W.HTML("<div class='satui-h'>Ajuste e resíduos</div>"),
            W.HTML("<div class='satui-legend'>O gráfico de ajuste mostra se o modelo "
                   "acompanha o <b>nível</b> e as <b>viradas</b> do ciclo; o painel de "
                   "resíduos é a leitura visual da bateria (ACF para autocorrelação, "
                   "histograma e QQ para normalidade).</div>"),
            self.out_diag_plot_fit, self.out_diag_plot_resid,
        ])
        card_plots.add_class("satui-card")

        return (card_acao, card_placar, card_tab, card_vif, card_leitura, card_plots)

    def _sync_diag_widgets(self):
        """Repõe as datas candidatas do Chow a partir do design do ajuste."""
        dd = getattr(self, "dd_chow_break", None)
        if dd is None:
            return
        fit = self.fit_
        X = getattr(fit, "exog", None) if fit is not None else None
        if X is None or getattr(X, "shape", (0, 0))[0] < 8:
            dd.options = []
            return
        k = int(X.shape[1])
        n = int(X.shape[0])
        opcoes = [(str(p)[:10], i) for i, p in enumerate(X.index)
                  if k < i < n - k]
        dd.options = opcoes
        if opcoes:
            dd.value = opcoes[len(opcoes) // 2][1]

    def _invalidate_diag(self, motivo="o modelo vigente mudou"):
        """Invalida as saídas do diagnóstico (padrão ``_mark_dirty`` do projeto)."""
        self.diagnostics_ = self.diag_blocks_ = self.vif_ = None
        if getattr(self, "out_diag_plot_resid", None) is None:   # ainda em construção
            return
        for w in ("out_diag_placar", "out_diag_tabela", "out_diag_vif", "out_diag_leitura",
                  "out_diag_plot_fit", "out_diag_plot_resid", "out_diag_status"):
            getattr(self, w).value = ""
        self.out_diag_notice.value = (
            f"<div class='satui-notice'>⚠️ <b>Diagnóstico desatualizado</b> — {motivo}. "
            "Clique em <i>Rodar diagnóstico</i>.</div>") if self.fit_ is not None else ""
        self._sync_diag_widgets()

    @staticmethod
    def _ok3(v):
        """Normaliza a coluna ``ok`` da bateria em ``True``/``False``/``None``."""
        if v is None:
            return None
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, float) and np.isnan(v):
            return None
        s = str(v).strip().lower()
        if s in ("true", "1", "1.0"):
            return True
        if s in ("false", "0", "0.0"):
            return False
        return None

    def _diag_table(self, alpha, break_index=None) -> pd.DataFrame:
        """Bateria de resíduos + estabilidade em data conhecida/desconhecida + VIF."""
        diag = _E.diagnostics
        fit = self.fit_
        rep = fit.diagnostics(alpha=alpha)
        extras = []
        self.vif_ = None
        X = fit.exog
        if X is not None and getattr(X, "shape", (0, 0))[1] > 0:
            y = (fit.fitted_link + fit.resid).to_numpy(dtype=float)
            if break_index is not None:
                try:
                    extras.append(diag.chow_test(y, X, int(break_index), alpha=alpha))
                except Exception as exc:  # noqa: BLE001
                    extras.append(diag.DiagnosticResult(
                        "Chow", np.nan, None, None, "sem quebra estrutural",
                        f"indisponível: {str(exc)[:60]}"))
            try:
                extras.append(diag.quandt_andrews(y, X))
            except Exception as exc:  # noqa: BLE001
                extras.append(diag.DiagnosticResult(
                    "Quandt-Andrews sup-F", np.nan, None, None,
                    "sem quebra (coeficientes estáveis)", f"indisponível: {str(exc)[:60]}"))
            # VIF: só as colunas macro da especificação (o filtro duro da seleção)
            cols = _E.selection._macro_columns(fit)
            if len(cols) >= 2:
                self.vif_ = diag.vif(X[cols])
                mx = float(self.vif_["VIF"].max())
                teto = float(self.fl_vif_max.value)
                extras.append(diag.DiagnosticResult(
                    test="VIF", statistic=mx, pvalue=None, passed=bool(mx <= teto),
                    h0="sem colinearidade excessiva entre as macro",
                    conclusion=("colinearidade sob controle" if mx <= teto
                                else f"VIF máximo {mx:.1f} acima do teto {teto:.1f}")))
            elif len(cols) == 1:
                self.vif_ = pd.DataFrame({"variavel": cols, "VIF": [1.0]})
        if extras:
            rep = pd.concat([rep, pd.DataFrame([r.to_row() for r in extras])],
                            ignore_index=True)
        return rep

    def _bloco_veredito(self, tab, rotulo, testes, resumo) -> dict:
        """Veredito de um bloco do placar: o **pior** teste da família e a evidência."""
        sub = tab[tab["teste"].isin(testes)] if "teste" in tab.columns else tab.iloc[0:0]
        if sub.empty:
            return {"bloco": rotulo, "nivel": "na", "veredito": "não avaliado",
                    "detalhe": f"{resumo} — os testes desta família não se aplicam a este "
                               "modelo", "falhas": 0, "n": 0, "teste": None}
        ordem = {"bad": 0, "warn": 1, "ok": 2}
        pior = None
        falhas = 0
        for _, r in sub.iterrows():
            ok = self._ok3(r.get("ok"))
            nivel = "ok" if ok is True else ("bad" if ok is False else "warn")
            if ok is False:
                falhas += 1
            if pior is None or ordem[nivel] < ordem[pior[0]]:
                pior = (nivel, r)
        nivel, r = pior
        est, p = r.get("estatistica"), r.get("p_valor")
        det = str(r["teste"])
        if pd.notna(est):
            det += f": {float(est):.3f}"
        if pd.notna(p):
            det += f" · p = {float(p):.4f}"
        conclusao = str(r.get("conclusao") or "").strip()
        if conclusao:
            det += f" — {conclusao}"
        veredito = {"ok": "passou", "bad": "reprovado", "warn": "inconclusivo"}[nivel]
        return {"bloco": rotulo, "nivel": nivel, "veredito": veredito, "detalhe": det,
                "falhas": falhas, "n": int(len(sub)), "teste": str(r["teste"])}

    def _placar_html(self, blocos) -> str:
        """Mosaico de blocos coloridos por token semântico."""
        cor = {"ok": "var(--ok-tx)", "warn": "var(--warn-tx)", "bad": "var(--bad-tx)",
               "na": "var(--muted)"}
        icone = {"ok": "✅", "warn": "⚠️", "bad": "❌", "na": "—"}
        cards = []
        for b in blocos:
            n = b["nivel"]
            cls = n if n in ("ok", "warn", "bad") else ""
            extra = (f" ({b['falhas']} de {b['n']} teste{'s' if b['n'] > 1 else ''})"
                     if b["falhas"] else "")
            cards.append(
                f"<div class='satui-bloco {cls}'><div class='k'>{b['bloco']}</div>"
                f"<div class='v' style='color:{cor[n]}'>{icone[n]} {b['veredito']}{extra}"
                f"</div><div class='d'>{b['detalhe']}</div></div>")
        return "<div class='satui-placar'>" + "".join(cards) + "</div>"

    def _diag_leitura_html(self, tab) -> str:
        """Leitura curta e acionável — só das famílias que reprovaram."""
        falhou = {str(r["teste"]) for _, r in tab.iterrows()
                  if self._ok3(r.get("ok")) is False}
        if not falhou:
            return ("<div class='satui-help'><div class='ttl'>Nada a corrigir por aqui</div>"
                    "Nenhum teste da bateria reprovou ao nível escolhido. Isso valida a "
                    "<b>forma</b> do modelo, não a sua <b>capacidade preditiva</b>: quem "
                    "decide se o modelo presta é o erro fora da amostra da aba "
                    "<b>Seleção</b> e a cobertura dos intervalos no <b>Backtest</b>.</div>")
        itens = []
        for rotulo, testes, conselho in _CONSELHO_DIAG:
            caidos = [t for t in testes if t in falhou]
            if not caidos:
                continue
            itens.append(f"<li><b>{rotulo}</b> ({', '.join(caidos)}) — {conselho}</li>")
        inconclusivos = sorted({str(r["teste"]) for _, r in tab.iterrows()
                                if self._ok3(r.get("ok")) is None})
        extra = ""
        if inconclusivos:
            extra = ("<div style='margin-top:6px'>Inconclusivos (sem graus de liberdade ou "
                     f"dependência ausente): {', '.join(inconclusivos)} — trate como "
                     "<b>não testado</b>, não como aprovado.</div>")
        return ("<div class='satui-help'><div class='ttl'>Leitura e próximo passo</div>"
                f"<ul>{''.join(itens)}</ul>{extra}"
                "<div style='margin-top:6px'>Corrija <b>uma</b> coisa por vez e reajuste: "
                "boa parte das falhas de heterocedasticidade e normalidade some quando a "
                "autocorrelação é resolvida.</div></div>")

    def _on_diag(self, b):
        if self.fit_ is None:
            self.out_diag_status.value = (
                "<div class='satui-notice'>Nenhum modelo ajustado — use <i>Ajustar agora</i> "
                "na aba <b>Especificação</b> ou adote uma especificação na aba "
                "<b>Seleção</b>.</div>")
            return
        alpha = float(self.fl_alpha_diag.value)
        quebra = self.dd_chow_break.value if self.dd_chow_break.options else None
        with self._busy(self.btn_diag, status=self.out_diag_status,
                        msg="rodando a bateria de diagnóstico…"):
            try:
                tab = self._diag_table(alpha, quebra)
            except ImportError as exc:
                self.out_diag_status.value = (
                    f"<div class='satui-notice'>Bateria indisponível: {exc}</div>")
                self._log(f"[diagnóstico] dependência ausente: {exc}")
                return
            except Exception as exc:  # noqa: BLE001
                self.out_diag_status.value = (
                    f"<div class='satui-notice'>Não foi possível rodar: {exc}</div>")
                self._log(f"[diagnóstico] erro: {exc}")
                return
        self.diagnostics_ = tab
        blocos = [self._bloco_veredito(tab, rot, testes, resumo)
                  for rot, testes, resumo in self._BLOCOS_DIAG]
        self.diag_blocks_ = blocos
        self.out_diag_notice.value = ""
        self.out_diag_placar.value = self._placar_html(blocos)

        vis = tab.rename(columns={"teste": "teste", "estatistica": "estatística",
                                  "p_valor": "p-valor", "conclusao": "conclusão"})
        vis["ok"] = [{True: "✓", False: "✗"}.get(self._ok3(v), "—") for v in tab["ok"]]
        cols = [c for c in ("teste", "estatística", "p-valor", "ok", "H0", "conclusão")
                if c in vis.columns]
        self.out_diag_tabela.value = self._df_html(
            vis[cols], max_height="420px",
            color_map={"ok": self._css_ok, "p-valor": self._css_pvalor},
            fmt_cols={"estatística": "{:.4f}"}, precision=4)

        if self.vif_ is not None and len(self.vif_):
            teto = float(self.fl_vif_max.value)
            vtab = self.vif_.rename(columns={"variavel": "variável"})
            self.out_diag_vif.value = (
                self._df_html(vtab, color_map={"VIF": self._css_vif_factory(teto)},
                              fmt_cols={"VIF": "{:.2f}"})
                + "<div class='satui-legend'>Regra prática: VIF acima de 5 (alguns usam 10) "
                  "indica que a variável é largamente explicada pelas outras — o teto "
                  f"vigente da seleção é <b>{teto:.1f}</b>.</div>")
        else:
            self.out_diag_vif.value = (
                "<div class='satui-legend'>Menos de duas variáveis macro no modelo vigente: "
                "não há colinearidade a medir.</div>")

        self.out_diag_leitura.value = self._diag_leitura_html(tab)

        if self.cb_diag_plots.value:
            try:
                self.out_diag_plot_fit.value = self._fig_html(
                    _E.report.plot_fit(self.fit_, self.series), stretch=True)
            except Exception as exc:  # noqa: BLE001
                self.out_diag_plot_fit.value = (
                    f"<div class='satui-legend'>Gráfico de ajuste indisponível: {exc}</div>")
            try:
                self.out_diag_plot_resid.value = self._fig_html(
                    _E.report.plot_residual_diagnostics(self.fit_), stretch=True)
            except Exception as exc:  # noqa: BLE001
                self.out_diag_plot_resid.value = (
                    f"<div class='satui-legend'>Painel de resíduos indisponível: {exc}</div>")
        else:
            self.out_diag_plot_fit.value = self.out_diag_plot_resid.value = ""

        reprovados = [b["bloco"] for b in blocos if b["nivel"] == "bad"]
        self.out_diag_status.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ Bateria concluída sem "
            "reprovações.</div>" if not reprovados else
            "<div class='satui-legend' style='color:var(--bad-ink)'>Bloco(s) reprovado(s): "
            f"<b>{', '.join(reprovados)}</b> — veja <i>O que fazer</i> abaixo.</div>")
        self._refresh_bar()
        self._log(f"[diagnóstico] {len(tab)} teste(s) a α={alpha:.3f}"
                  + (f" · reprovados: {', '.join(reprovados)}" if reprovados else " · tudo ok"))

    # ==================================================================
    # Aba Cenários & Projeção — o fator prospectivo
    # ==================================================================
    #: extrapolação da trajetória base (rótulo → método de :func:`extend_macro`)
    _BASE_METODOS = (("reversão à média histórica", "revert"),
                     ("manter o último valor observado", "hold"),
                     ("extrapolar a última variação", "trend"))

    def _build_tab_cenarios(self):
        """Cartões da aba **Cenários & Projeção** (devolve a tupla de filhos)."""
        # --- card: parâmetros comuns da projeção --------------------------
        self.out_scen_notice = W.HTML()
        self.sl_scen_horizon = W.BoundedIntText(
            value=int(self.sl_horizon.value), min=1, max=120, description="horizonte:",
            style={"description_width": "initial"}, layout=W.Layout(width="170px"))
        self.sl_scen_horizon.observe(lambda c: self._render_scen_notice(), names="value")
        self.fl_scen_alpha = W.BoundedFloatText(
            value=0.10, min=0.01, max=0.50, step=0.05, description="α da banda:",
            style={"description_width": "initial"}, layout=W.Layout(width="170px"))
        self.sl_scen_sims = W.BoundedIntText(
            value=2000, min=0, max=20000, step=250, description="simulações da banda:",
            style={"description_width": "initial"}, layout=W.Layout(width="260px"))
        self.dd_stress_var = W.Dropdown(
            options=self._macro_cols(), description="variável de estresse:",
            style={"description_width": "initial"}, layout=W.Layout(width="290px"))
        self.tx_scen_probs = W.Text(
            value=",".join(str(p) for p in self._scenario_probs_default),
            description="pesos (base, otimista, adverso):",
            style={"description_width": "initial"}, layout=W.Layout(width="380px"))
        card_cfg = W.VBox([
            W.HTML("<div class='satui-h'>Projeção condicional — parâmetros</div>"),
            W.HTML("<div class='satui-legend'>O <b>cenário</b> é a trajetória futura das "
                   "variáveis macro; o modelo vigente traduz essa trajetória na trajetória "
                   "do parâmetro de risco. O <b>α da banda</b> define o leque "
                   "(α=0,10 ⇒ intervalo de 90%), obtido por reamostragem dos resíduos "
                   "propagada pela dinâmica do modelo — a incerteza <b>acumula</b> ao longo "
                   "do horizonte. Os <b>pesos</b> valem para a projeção ponderada e seguem "
                   "a ordem base, otimista, adverso (cenários com outro nome recebem o peso "
                   "que sobrar).</div>"),
            W.HBox([self.sl_scen_horizon, self.fl_scen_alpha, self.sl_scen_sims],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.dd_stress_var, self.tx_scen_probs],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_scen_notice,
        ])
        card_cfg.add_class("satui-card")

        # --- card: cenários padrão ----------------------------------------
        self.btn_scen_padrao = W.Button(
            description="Montar base / adverso / otimista", icon="magic",
            button_style="primary", layout=W.Layout(width="auto", min_width="290px"),
            tooltip="Base por reversão à média + adverso (+2 desvios-padrão na variável de "
                    "estresse) e otimista (−1 desvio-padrão).")
        self.btn_scen_padrao.on_click(self._on_scen_padrao)
        self.out_scen_padrao = W.HTML()
        card_padrao = W.VBox([
            W.HTML("<div class='satui-h'>Caminho 1 — cenários padrão (um clique)</div>"),
            W.HTML("<div class='satui-legend'>Ponto de partida para explorar o modelo: a "
                   "trajetória <b>base</b> reverte à média histórica de cada variável e os "
                   "cenários <b>adverso</b>/<b>otimista</b> deslocam a variável de estresse "
                   "em ±desvios-padrão da própria série. Serve para ver a sensibilidade na "
                   "hora — <b>não</b> substitui o cenário oficial da área econômica, que "
                   "entra pelo caminho 3.</div>"),
            W.HBox([self.btn_scen_padrao], layout=W.Layout(align_items="center")),
            self.out_scen_padrao,
        ])
        card_padrao.add_class("satui-card")

        # --- card: choque parametrizado -------------------------------------
        self.dd_scen_base = W.Dropdown(
            options=list(self._BASE_METODOS), value="revert", description="trajetória base:",
            style={"description_width": "initial"}, layout=W.Layout(width="330px"))
        self.fl_shock_mag = W.BoundedFloatText(
            value=2.0, min=-50.0, max=50.0, step=0.25, description="magnitude:",
            style={"description_width": "initial"}, layout=W.Layout(width="180px"))
        self.dd_shock_unit = W.Dropdown(
            options=[("desvios-padrão da variável", "sd"), ("unidades da variável", "abs")],
            value="sd", description="em:", style={"description_width": "initial"},
            layout=W.Layout(width="290px"))
        self.fl_shock_persist = W.BoundedFloatText(
            value=1.0, min=0.0, max=1.0, step=0.05, description="persistência:",
            style={"description_width": "initial"}, layout=W.Layout(width="180px"))
        self.fl_shock_otim = W.BoundedFloatText(
            value=0.5, min=0.0, max=3.0, step=0.1, description="otimista = fração do adverso:",
            style={"description_width": "initial"}, layout=W.Layout(width="320px"))
        self.btn_scen_choque = W.Button(
            description="Montar cenários por choque", icon="bolt", button_style="primary",
            layout=W.Layout(width="auto", min_width="250px"),
            tooltip="Aplica um choque aditivo à trajetória base da variável de estresse.")
        self.btn_scen_choque.on_click(self._on_scen_choque)
        self.out_scen_choque = W.HTML()
        card_choque = W.VBox([
            W.HTML("<div class='satui-h'>Caminho 2 — choque parametrizado</div>"),
            W.HTML("<div class='satui-legend'>O choque é <b>aditivo</b> sobre a trajetória "
                   "base da variável de estresse. A <b>persistência</b> diz o que acontece "
                   "com ele ao longo do horizonte: <b>1,0</b> mantém o choque até o fim "
                   "(deslocamento permanente do nível) e valores menores o fazem decair "
                   "geometricamente (choque temporário, com a variável voltando à base). "
                   "Em <b>desvios-padrão</b> a magnitude fica comparável entre variáveis de "
                   "escalas diferentes.</div>"),
            W.HBox([self.dd_scen_base, self.fl_shock_mag, self.dd_shock_unit],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.fl_shock_persist, self.fl_shock_otim, self.btn_scen_choque],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_scen_choque,
        ])
        card_choque.add_class("satui-card")

        # --- card: cenário manual (colagem) -----------------------------------
        self.tx_scen_nome = W.Text(value="adverso_economia", description="nome do cenário:",
                                   style={"description_width": "initial"},
                                   layout=W.Layout(width="330px"))
        self.fl_scen_peso = W.BoundedFloatText(
            value=0.0, min=0.0, max=1.0, step=0.05, description="peso (0 = automático):",
            style={"description_width": "initial"}, layout=W.Layout(width="260px"))
        self.dd_scen_dec = W.Dropdown(
            options=[("ponto decimal (1.25)", "."), ("vírgula decimal (1,25)", ",")],
            value=".", description="decimal:", style={"description_width": "initial"},
            layout=W.Layout(width="250px"))
        self.ta_scen_paste = W.Textarea(
            placeholder="periodo\tdesemprego\trenda\n2026-01-01\t9.5\t2.1\n…",
            layout=W.Layout(width="99%", height="170px"))
        self.btn_scen_modelo = W.Button(
            description="Gerar modelo para colar", icon="table",
            layout=W.Layout(width="auto", min_width="230px"),
            tooltip="Preenche a caixa com a trajetória base no formato certo — edite as "
                    "colunas (ou cole por cima) e adicione o cenário.")
        self.btn_scen_modelo.on_click(self._on_scen_modelo)
        self.btn_scen_add = W.Button(
            description="Adicionar cenário", icon="plus", button_style="success",
            layout=W.Layout(width="auto", min_width="190px"),
            tooltip="Adiciona (ou substitui, se o nome já existir) o cenário colado.")
        self.btn_scen_add.on_click(self._on_scen_add)
        self.btn_scen_limpar = W.Button(
            description="Limpar cenários", icon="trash",
            layout=W.Layout(width="auto", min_width="180px"),
            tooltip="Descarta todos os cenários montados (pede confirmação).")
        self.btn_scen_limpar.on_click(
            lambda b: self._confirm_twice(b, self._limpar_cenarios))
        self.out_scen_manual = W.HTML()
        card_manual = W.VBox([
            W.HTML("<div class='satui-h'>Caminho 3 — colar o cenário da área econômica</div>"),
            W.HTML("<div class='satui-help'><div class='ttl'>Formato aceito</div>"
                   "Uma linha de <b>cabeçalho</b> com os nomes das variáveis (exatamente os "
                   "nomes das colunas da macro) e <b>uma linha por período</b> do horizonte, "
                   "separados por <b>TAB</b> (colagem direta do Excel), vírgula ou ponto e "
                   "vírgula.<ul>"
                   "<li>Colunas de data/rótulo são <b>ignoradas</b>: o calendário é "
                   "reconstruído a partir do fim da amostra na frequência da série.</li>"
                   "<li>Variáveis do modelo que <b>faltarem</b> na colagem são completadas "
                   "com a trajetória base — a interface diz quais foram.</li>"
                   "<li>O nº de linhas precisa bater com o <b>horizonte</b>: todos os "
                   "cenários compartilham o mesmo calendário.</li>"
                   "<li>Use os nomes <span class='pname'>base</span>, "
                   "<span class='pname'>otimista</span> e <span class='pname'>adverso</span> "
                   "para herdar os pesos declarados acima; qualquer outro nome recebe peso "
                   "próprio (campo ao lado) ou o que sobrar.</li></ul>"
                   "É assim que o cenário oficial entra: sem redigitar, sem planilha "
                   "paralela e com o mesmo motor de projeção dos demais.</div>"),
            W.HBox([self.tx_scen_nome, self.fl_scen_peso, self.dd_scen_dec],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.ta_scen_paste,
            W.HBox([self.btn_scen_modelo, self.btn_scen_add, self.btn_scen_limpar],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_scen_manual,
        ])
        card_manual.add_class("satui-card")

        # --- card: cenários montados -------------------------------------------
        self.out_scen_resumo = W.HTML()
        self.out_scen_tabela = W.HTML()
        self.out_scen_plot = W.HTML()
        card_montados = W.VBox([
            W.HTML("<div class='satui-h'>Cenários montados</div>"),
            W.HTML("<div class='satui-legend'>A área sombreada dos gráficos é o "
                   "<b>futuro</b> — à esquerda da linha tracejada está o observado, à "
                   "direita a trajetória de cada cenário. Só as variáveis que <b>diferem</b> "
                   "entre cenários são desenhadas.</div>"),
            self.out_scen_resumo, self.out_scen_tabela, self.out_scen_plot,
        ])
        card_montados.add_class("satui-card")

        # --- card: projetar ------------------------------------------------------
        self.btn_project = W.Button(
            description="Projetar", icon="line-chart", button_style="success",
            layout=W.Layout(width="auto", min_width="160px"),
            tooltip="Projeta o modelo vigente sobre cada cenário, com o leque de incerteza.")
        self.btn_project.on_click(self._on_project)
        self.cb_proj_plot = W.Checkbox(value=True, indent=False,
                                       description="desenhar o leque")
        self.out_proj_status = W.HTML()
        self.out_proj_timer = W.HTML()
        self.out_proj_tiles = W.HTML()
        self.out_proj_plot = W.HTML()
        self.out_proj_tabela = W.HTML()
        card_proj = W.VBox([
            W.HTML("<div class='satui-h'>Projeção em leque</div>"),
            W.HTML("<div class='satui-legend'>A <b>linha</b> é a trajetória central de cada "
                   "cenário e a <b>faixa</b> é o intervalo do α escolhido. Leia os dois "
                   "juntos: a distância entre os cenários mede o efeito do <b>ciclo</b>; a "
                   "largura da faixa mede a incerteza do <b>modelo</b>. Quando a faixa do "
                   "base engole o adverso, o cenário não está dizendo mais do que o ruído do "
                   "próprio ajuste.</div>"),
            W.HBox([self.btn_project, self.cb_proj_plot],
                   layout=W.Layout(align_items="center")),
            self.out_proj_status, self.out_proj_timer, self.out_proj_tiles,
            self.out_proj_plot, self.out_proj_tabela,
        ])
        card_proj.add_class("satui-card")

        # --- card: projeção ponderada ---------------------------------------------
        self.out_pond_status = W.HTML()
        self.out_pond_tiles = W.HTML()
        self.out_pond_tabela = W.HTML()
        card_pond = W.VBox([
            W.HTML("<div class='satui-h'>Projeção ponderada (a curva única)</div>"),
            W.HTML("<div class='satui-legend'>Média das trajetórias dos cenários pelos seus "
                   "<b>pesos</b> — a curva única que segue para o processo seguinte, "
                   "equivalente a <code>Projection.weighted()</code>. Ela existe para que "
                   "não haja duas respostas para a mesma pergunta: o mesmo conjunto de "
                   "cenários e a mesma projeção alimentam provisionamento, estresse e "
                   "planejamento.</div>"),
            self.out_pond_status, self.out_pond_tiles, self.out_pond_tabela,
        ])
        card_pond.add_class("satui-card")

        # --- card: exportar --------------------------------------------------------
        self.dd_export_fmt = W.Dropdown(
            options=[("TSV — colar no Excel", "tsv"),
                     ("CSV — ponto decimal, vírgula", "csv"),
                     ("CSV — vírgula decimal, ponto e vírgula", "csv_br")],
            value="tsv", description="formato:", style={"description_width": "initial"},
            layout=W.Layout(width="330px"))
        self.btn_scen_export = W.Button(
            description="Exportar projeção", icon="download", button_style="primary",
            layout=W.Layout(width="auto", min_width="200px"),
            tooltip="Monta o DataFrame da projeção (ui.projection_frame()) e o texto para "
                    "copiar.")
        self.btn_scen_export.on_click(self._on_scen_export)
        self.out_export_status = W.HTML()
        self.ta_scen_csv = W.Textarea(
            placeholder="a projeção exportada aparece aqui — selecione tudo (Ctrl+A) e copie "
                        "(Ctrl+C)",
            layout=W.Layout(width="99%", height="170px"))
        card_export = W.VBox([
            W.HTML("<div class='satui-h'>Exportar a projeção</div>"),
            W.HTML("<div class='satui-legend'>Formato longo — uma linha por (cenário, "
                   "período) com média e banda, mais as linhas do cenário "
                   "<b>ponderado</b>. O mesmo objeto está em "
                   "<code>ui.projection_frame()</code> para seguir em código.</div>"),
            W.HBox([self.dd_export_fmt, self.btn_scen_export],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_export_status, self.ta_scen_csv,
        ])
        card_export.add_class("satui-card")

        self._render_scen_notice()
        return (card_cfg, card_padrao, card_choque, card_manual, card_montados,
                card_proj, card_pond, card_export)

    # ------------------------------------------------------------------ pré-requisitos
    def _render_scen_notice(self):
        """Diz, no alto da aba, o que falta para projetar (modelo, macro, horizonte)."""
        if getattr(self, "out_scen_notice", None) is None:
            return
        avisos = []
        if self.series is None:
            avisos.append("Sem série carregada — traga os dados na aba <b>Série</b>.")
        elif self.macro is None or not self._macro_cols():
            avisos.append("Sem variáveis macro: o cenário é uma trajetória macro.")
        if self.fit_ is None:
            avisos.append("Nenhum <b>modelo vigente</b> — ajuste na aba <b>Especificação</b> "
                          "ou adote uma especificação na aba <b>Seleção</b>.")
        elif self._dirty_since_fit:
            avisos.append("O ajuste está <b>desatualizado</b> (a especificação mudou): "
                          "reajuste antes de projetar.")
        ss = self.scenarios_
        if ss is not None and len(ss):
            comp = {s.horizon for s in ss.scenarios}
            if comp != {int(self.sl_scen_horizon.value)}:
                avisos.append(
                    f"Os cenários montados têm {sorted(comp)} período(s) e o horizonte da "
                    f"tela é {int(self.sl_scen_horizon.value)} — remonte os cenários para o "
                    "novo horizonte.")
        self.out_scen_notice.value = "".join(
            f"<div class='satui-notice'>⚠️ {a}</div>" for a in avisos)

    def _scen_pronto(self, alvo) -> bool:
        """Valida série + macro antes de montar cenários (mensagem no ``alvo``)."""
        if self.series is None:
            alvo.value = ("<div class='satui-notice'>Sem série carregada — traga os dados na "
                          "aba <b>Série</b> (ou carregue o estudo de referência).</div>")
            return False
        if self.macro is None or not self._macro_cols():
            alvo.value = ("<div class='satui-notice'>Sem variáveis macro carregadas: um "
                          "cenário é a trajetória futura das macro.</div>")
            return False
        return True

    # ------------------------------------------------------------------ pesos
    def _probs_tupla(self) -> tuple:
        """Os três pesos (base, otimista, adverso) **normalizados** para somar 1."""
        p = list(self._scenario_probs())
        p = [float(x) for x in p[:3]]
        while len(p) < 3:
            p.append(0.0)
        total = float(sum(p))
        if total <= 0:
            return (0.5, 0.3, 0.2)
        return tuple(x / total for x in p)

    def _pesos_por_nome(self) -> dict:
        """``{'base': w, 'otimista': w, 'adverso': w}`` a partir dos pesos da tela."""
        p = self._probs_tupla()
        return {"base": p[0], "otimista": p[1], "adverso": p[2]}

    def _aplica_probs(self, ss) -> dict:
        """Distribui os pesos entre os cenários do conjunto (soma exatamente 1).

        Precedência: peso declarado para aquele cenário (campo do caminho manual) →
        peso do nome padrão (base/otimista/adverso) → o que sobrar, dividido
        igualmente entre os cenários sem peso.
        """
        nomes = ss.names()
        mapa = self._pesos_por_nome()
        pesos, faltando = [], []
        for i, nome in enumerate(nomes):
            p = self._scen_pesos.get(nome, mapa.get(nome))
            if p is None:
                faltando.append(i)
                pesos.append(0.0)
            else:
                pesos.append(max(0.0, float(p)))
        if faltando:
            resto = 1.0 - float(sum(pesos))
            fatia = (resto / len(faltando)) if resto > 1e-9 else (1.0 / len(nomes))
            for i in faltando:
                pesos[i] = fatia
        total = float(sum(pesos))
        if total <= 0:
            pesos = [1.0 / len(nomes)] * len(nomes)
            total = 1.0
        for s, p in zip(ss.scenarios, pesos):
            s.probability = float(p) / total
        return {s.name: s.probability for s in ss.scenarios}

    # ------------------------------------------------------------------ construção
    def _scen_freq(self) -> str:
        """Frequência do calendário futuro (a da série)."""
        return self.series.frequency if self.series is not None else "MS"

    def _base_future(self, horizonte=None) -> pd.DataFrame:
        """Trajetória macro **base** do horizonte corrente (:func:`extend_macro`)."""
        cols = self._macro_cols()
        return _E.scenarios.extend_macro(
            self.macro[cols], int(horizonte or self.sl_scen_horizon.value),
            freq=self._scen_freq(), method=self.dd_scen_base.value)

    def _adota_cenarios(self, ss, origem=""):
        """Adota um :class:`ScenarioSet` como o conjunto vigente e repinta a aba."""
        self.scenarios_ = ss
        pesos = self._aplica_probs(ss)
        # a projeção anterior era de outro conjunto de cenários
        self._invalidate_cenarios("os cenários mudaram")
        self._render_scenarios()
        self._render_scen_notice()
        self._refresh_bar()
        detalhe = ", ".join(f"{k} {v:.0%}" for k, v in pesos.items())
        self._log(f"[cenários] {len(ss)} cenário(s) montado(s){' — ' + origem if origem else ''} "
                  f"· pesos: {detalhe}")

    def _on_scen_padrao(self, b):
        if not self._scen_pronto(self.out_scen_padrao):
            return
        probs = self._probs_tupla()
        with self._busy(self.btn_scen_padrao, self.btn_scen_choque, self.btn_scen_add,
                        status=self.out_scen_padrao, msg="montando os cenários…"):
            try:
                ss = _E.scenarios.standard_scenarios(
                    self.macro[self._macro_cols()], horizon=int(self.sl_scen_horizon.value),
                    freq=self._scen_freq(), stress_var=self._stress_var(),
                    probabilities=probs)
            except Exception as exc:  # noqa: BLE001
                self.out_scen_padrao.value = (
                    f"<div class='satui-notice'>Não foi possível montar: {exc}</div>")
                self._log(f"[cenários] erro nos cenários padrão: {exc}")
                return
        self._scen_pesos = {}
        self._adota_cenarios(ss, origem=f"padrão sobre '{self._stress_var()}'")
        self.out_scen_padrao.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ Base, adverso e otimista "
            f"montados sobre <b>{self._stress_var()}</b> "
            f"({int(self.sl_scen_horizon.value)} períodos).</div>")

    def _on_scen_choque(self, b):
        if not self._scen_pronto(self.out_scen_choque):
            return
        var = self._stress_var()
        if var not in self._macro_cols():
            self.out_scen_choque.value = (
                f"<div class='satui-notice'>A variável de estresse <b>{var}</b> não está na "
                "macro carregada.</div>")
            return
        with self._busy(self.btn_scen_choque, self.btn_scen_padrao, self.btn_scen_add,
                        status=self.out_scen_choque, msg="aplicando o choque…"):
            try:
                base = self._base_future()
                mag = float(self.fl_shock_mag.value)
                if self.dd_shock_unit.value == "sd":
                    sd = float(pd.Series(self.macro[var]).std())
                    if not np.isfinite(sd) or sd <= 0:
                        raise ValueError(
                            f"a variável {var!r} não tem variação histórica: use a magnitude "
                            "em unidades da variável.")
                    delta = mag * sd
                else:
                    delta = mag
                pesos = self._pesos_por_nome()
                ss = _E.scenarios.shock_scenarios(
                    base,
                    shocks={"adverso": {var: delta},
                            "otimista": {var: -delta * float(self.fl_shock_otim.value)}},
                    probabilities={"adverso": pesos["adverso"], "otimista": pesos["otimista"]},
                    include_base=True, base_probability=pesos["base"])
                phi = float(self.fl_shock_persist.value)
                if phi < 0.999:
                    self._decai_choque(ss, base, var, phi)
            except Exception as exc:  # noqa: BLE001
                self.out_scen_choque.value = (
                    f"<div class='satui-notice'>Não foi possível montar: {exc}</div>")
                self._log(f"[cenários] erro no choque: {exc}")
                return
        self._scen_pesos = {}
        self._adota_cenarios(ss, origem=f"choque de {delta:+.3f} em '{var}'")
        persist = ("permanente" if float(self.fl_shock_persist.value) >= 0.999
                   else f"decaindo a {float(self.fl_shock_persist.value):.2f} por período")
        self.out_scen_choque.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ Choque de "
            f"<b>{delta:+.3f}</b> em <b>{var}</b> ({persist}) aplicado sobre a trajetória "
            "base.</div>")

    @staticmethod
    def _decai_choque(ss, base: pd.DataFrame, var: str, phi: float) -> None:
        """Torna o choque **temporário**: o desvio em relação à base decai ``phi^k``.

        :func:`shock_scenarios` aplica um deslocamento constante; com persistência
        menor que 1 o choque some ao longo do horizonte (a variável volta à base).
        """
        b = base[var].to_numpy(dtype=float)
        for s in ss.scenarios:
            if s.name == "base" or var not in s.macro.columns:
                continue
            incr = s.macro[var].to_numpy(dtype=float) - b
            decai = float(phi) ** np.arange(len(incr), dtype=float)
            s.macro[var] = b + incr * decai

    def _on_scen_modelo(self, b):
        """Preenche a caixa de colagem com a trajetória base (o gabarito do formato)."""
        if not self._scen_pronto(self.out_scen_manual):
            return
        try:
            base = self._base_future()
        except Exception as exc:  # noqa: BLE001
            self.out_scen_manual.value = (
                f"<div class='satui-notice'>Não foi possível gerar o modelo: {exc}</div>")
            return
        dec = self.dd_scen_dec.value
        texto = base.round(6).to_csv(sep="\t", index_label="periodo", decimal=dec)
        self.ta_scen_paste.value = texto
        self.out_scen_manual.value = (
            "<div class='satui-legend'>Gabarito preenchido com a trajetória <b>base</b> "
            f"({len(base)} períodos). Edite os valores (ou cole por cima) e clique em "
            "<i>Adicionar cenário</i>.</div>")

    @staticmethod
    def _scen_sep(cabecalho: str, decimal: str = ".") -> str:
        """Separador de coluna do texto colado, decidido pelo **cabeçalho**.

        Não usamos o *sniffer* do pandas: com uma única coluna ele inventa um
        separador a partir das letras do próprio nome da variável.
        """
        if "\t" in cabecalho:
            return "\t"
        if ";" in cabecalho:
            return ";"
        if "," in cabecalho and decimal != ",":
            return ","
        return "\t"          # coluna única (ou vírgula ambígua com decimal vírgula)

    def _parse_scen_paste(self, txt) -> tuple:
        """Lê o texto colado e devolve ``(macro_futura, ignoradas, completadas)``."""
        import io as _io

        bruto = str(txt or "").strip()
        if not bruto:
            raise ValueError(
                "cole a trajetória das variáveis (uma linha de cabeçalho e uma linha por "
                "período) antes de adicionar — ou clique em 'Gerar modelo para colar'.")
        dec = self.dd_scen_dec.value
        sep = self._scen_sep(bruto.splitlines()[0], dec)
        try:
            df = pd.read_csv(_io.StringIO(bruto), sep=sep, engine="python", decimal=dec)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"não consegui ler o texto colado ({type(exc).__name__}). Use uma linha de "
                "cabeçalho com os nomes das variáveis e uma linha por período, separados "
                "por TAB, vírgula ou ponto e vírgula.") from None
        df.columns = [str(c).strip() for c in df.columns]
        macro_cols = self._macro_cols()
        numericas = {}
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                numericas[c] = s
        usadas = {c: v for c, v in numericas.items() if c in macro_cols}
        ignoradas = [c for c in df.columns if c not in usadas]
        if not usadas:
            raise ValueError(
                "nenhuma coluna do texto casa com as variáveis macro carregadas "
                f"({', '.join(macro_cols) or '—'}). O cabeçalho precisa usar exatamente os "
                "mesmos nomes.")
        H = int(self.sl_scen_horizon.value)
        if len(df) != H:
            raise ValueError(
                f"o texto tem {len(df)} linha(s) de dados e o horizonte da tela é {H}. "
                "Ajuste o horizonte (ou o número de linhas): todos os cenários compartilham "
                "o mesmo calendário.")
        idx = _E._engine.future_index(self.macro.index, self._scen_freq(), H)
        base = self._base_future(H)
        dados, completadas = {}, []
        for c in macro_cols:
            vals = base[c].to_numpy(dtype=float).copy()
            if c in usadas:
                colado = usadas[c].to_numpy(dtype=float)[:H]
                falta = ~np.isfinite(colado)
                if falta.any():
                    completadas.append(f"{c} ({int(falta.sum())} período(s) vazio(s))")
                vals = np.where(falta, vals, colado)
            else:
                completadas.append(c)
            dados[c] = vals
        return pd.DataFrame(dados, index=idx), ignoradas, completadas

    def _on_scen_add(self, b):
        if not self._scen_pronto(self.out_scen_manual):
            return
        nome = str(self.tx_scen_nome.value or "").strip()
        if not nome:
            self.out_scen_manual.value = (
                "<div class='satui-notice'>Dê um <b>nome</b> ao cenário (ele aparece no "
                "gráfico, na tabela e na exportação).</div>")
            return
        try:
            futura, ignoradas, completadas = self._parse_scen_paste(self.ta_scen_paste.value)
        except ValueError as exc:
            self.out_scen_manual.value = f"<div class='satui-notice'>{exc}</div>"
            self._log(f"[cenários] colagem recusada: {exc}")
            return
        peso = float(self.fl_scen_peso.value)
        if peso > 0:
            self._scen_pesos[nome] = peso
        else:
            self._scen_pesos.pop(nome, None)
        anteriores = list(self.scenarios_.scenarios) if self.scenarios_ else []
        cenarios = [s for s in anteriores if s.name != nome and s.horizon == len(futura)]
        descartados = [s.name for s in anteriores
                       if s.name != nome and s.horizon != len(futura)]
        cenarios.append(_E.scenarios.Scenario(nome, futura))
        self._adota_cenarios(_E.scenarios.ScenarioSet(cenarios), origem=f"colagem '{nome}'")
        extras = []
        if descartados:
            extras.append("cenários de outro horizonte descartados: <b>"
                          + ", ".join(descartados) + "</b>")
        if completadas:
            extras.append("completadas pela base: <b>" + ", ".join(completadas) + "</b>")
        if ignoradas:
            extras.append("colunas ignoradas: " + ", ".join(str(c) for c in ignoradas))
        self.out_scen_manual.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ Cenário "
            f"<b>{nome}</b> adicionado ({len(futura)} períodos).</div>"
            + (f"<div class='satui-legend'>{' · '.join(extras)}</div>" if extras else ""))

    def _limpar_cenarios(self):
        self.scenarios_ = None
        self._scen_pesos = {}
        self._invalidate_cenarios("os cenários foram descartados")
        self._render_scenarios()
        self._render_scen_notice()
        self._refresh_bar()
        self._log("[cenários] conjunto de cenários descartado.")

    # ------------------------------------------------------------------ render dos cenários
    def _cols_variantes(self) -> list:
        """Variáveis cujas trajetórias **diferem** entre os cenários (as que importam)."""
        ss = self.scenarios_
        if ss is None or not len(ss):
            return []
        cols = [c for c in self._macro_cols() if all(c in s.macro.columns for s in ss.scenarios)]
        ref = ss.scenarios[0].macro

        def _difere(s, c) -> bool:
            a = s.macro[c].to_numpy(dtype=float)
            b = ref[c].to_numpy(dtype=float)
            return len(a) != len(b) or not np.allclose(a, b, equal_nan=True)

        variam = [c for c in cols
                  if any(_difere(s, c) for s in ss.scenarios[1:])]
        if variam:
            return variam
        alvo = self._stress_var()
        return [alvo] if alvo in cols else cols[:3]

    def _render_scenarios(self):
        """Resumo, tabela das trajetórias e gráfico dos cenários vigentes."""
        ss = self.scenarios_
        if ss is None or not len(ss):
            self.out_scen_resumo.value = (
                "<div class='satui-legend'>Nenhum cenário montado — use um dos três caminhos "
                "acima.</div>")
            self.out_scen_tabela.value = ""
            self.out_scen_plot.value = ""
            return
        tiles = {"cenários": len(ss), "horizonte": int(ss.scenarios[0].horizon),
                 "variáveis": int(ss.scenarios[0].macro.shape[1])}
        for s in ss.scenarios:
            tiles[f"peso {s.name}"] = float(s.probability or 0.0)
        self.out_scen_resumo.value = self._metric_tiles(tiles)

        cols = self._cols_variantes()
        linhas = []
        for s in ss.scenarios:
            for per, row in s.macro.iterrows():
                linha = {"cenário": s.name, "período": str(per)[:10]}
                for c in cols:
                    linha[c] = float(row[c])
                linhas.append(linha)
        self.out_scen_tabela.value = self._df_html(pd.DataFrame(linhas), max_height="300px",
                                                   precision=3)
        self.out_scen_plot.value = self._scen_plot_html(cols)

    def _scen_plot_html(self, cols) -> str:
        """Histórico + trajetória de cada cenário, com o futuro sombreado."""
        ss = self.scenarios_
        if ss is None or not cols:
            return ""
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - depende do ambiente
            return f"<div class='satui-legend'>matplotlib indisponível: {exc}</div>"
        cores = _E.report._palette()
        paleta = [cores["primaria"], cores["secundaria"], "#2ca02c", "#9467bd", "#ff7f0e"]
        cols = list(cols)[:4]
        n = len(cols)
        fig, axes = plt.subplots(n, 1, figsize=(10.5, 2.1 * n), sharex=True)
        axes = np.atleast_1d(axes)
        hist_idx = _E.report._to_ts(self.macro.index)
        corte = hist_idx[-1]
        fut_idx = _E.report._to_ts(ss.scenarios[0].macro.index)
        for i, c in enumerate(cols):
            ax = axes[i]
            ax.plot(hist_idx, self.macro[c].to_numpy(dtype=float), color=cores["neutra"],
                    lw=1.4, label="observado")
            for j, s in enumerate(ss.scenarios):
                ax.plot(_E.report._to_ts(s.macro.index), s.macro[c].to_numpy(dtype=float),
                        color=paleta[j % len(paleta)], lw=1.8, label=s.name)
            ax.axvspan(corte, fut_idx[-1], color=cores["neutra"], alpha=0.08, zorder=0)
            ax.axvline(corte, color=cores["neutra"], lw=1.1, ls="--", zorder=1)
            ax.set_ylabel(c, fontsize=9)
            ax.grid(alpha=0.25)
        axes[0].set_title("Trajetórias macro — observado × cenários", fontsize=11)
        axes[0].legend(loc="best", fontsize=8, ncol=min(4, len(ss) + 1))
        fig.tight_layout()
        return self._fig_html(fig, stretch=True)

    # ------------------------------------------------------------------ projeção
    def _on_project(self, b):
        if self.fit_ is None or self.model_ is None:
            self.out_proj_status.value = (
                "<div class='satui-notice'>Nenhum <b>modelo vigente</b>: ajuste na aba "
                "<b>Especificação</b> ou adote uma especificação na aba <b>Seleção</b> "
                "antes de projetar.</div>")
            return
        if self.scenarios_ is None or not len(self.scenarios_):
            self.out_proj_status.value = (
                "<div class='satui-notice'>Monte ao menos um cenário acima — a projeção é "
                "<b>condicional</b> a uma trajetória macro.</div>")
            return
        alpha = float(self.fl_scen_alpha.value)
        n_sims = int(self.sl_scen_sims.value)
        self.out_proj_status.value = ""
        with self._busy(self.btn_project, self.btn_scen_padrao, self.btn_scen_choque,
                        self.btn_scen_add, self.btn_scen_export), \
                self._cronometro(self.out_proj_timer, "projetando os cenários") as decorrido:
            try:
                proj = _E.scenarios.project(
                    self.model_, self.scenarios_, horizon=int(self.sl_scen_horizon.value),
                    alpha=alpha, n_sims=n_sims, seed=0)
            except Exception as exc:  # noqa: BLE001
                self.out_proj_status.value = (
                    f"<div class='satui-notice'>Não foi possível projetar: {exc}</div>")
                self._log(f"[projeção] erro: {type(exc).__name__}: {exc}")
                return
            secs = decorrido()
        self.projection_ = proj
        self.projection_table_ = None
        pesos = {s.name: float(s.probability or 0.0) for s in self.scenarios_.scenarios}
        try:
            self.weighted_ = proj.weighted(pesos)
        except Exception as exc:  # noqa: BLE001
            self.weighted_ = None
            self._log(f"[projeção] projeção ponderada indisponível: {exc}")
        self._render_projection()
        self._render_weighted()
        self._refresh_bar()
        self.out_proj_status.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ "
            f"{len(proj.paths)} cenário(s) projetado(s) em {self._fmt_dur(secs)} — leque de "
            f"{int(round((1 - alpha) * 100))}% "
            + (f"({n_sims} simulações por cenário)." if n_sims else
               "(sem simulação: só a trajetória central).") + "</div>")
        self._log(f"[projeção] {len(proj.paths)} cenário(s) · horizonte {proj.horizon} · "
                  f"α={alpha:.2f} · {self._fmt_dur(secs)}")

    def _fig_projecao(self):
        """Leque + divisor histórico/projetado + a curva ponderada."""
        proj = self.projection_
        nivel = int(round((1 - proj.alpha) * 100))
        fig = _E.report.plot_projection(
            proj, self.series,
            title=f"Projeção condicional — {self._param_label()} (leque {nivel}%)")
        ax = fig.axes[0]
        cores = _E.report._palette()
        fut = _E.report._to_ts(next(iter(proj.paths.values())).index)
        corte = _E.report._to_ts(self.series.index)[-1]
        ax.axvspan(corte, fut[-1], color=cores["neutra"], alpha=0.08, zorder=0)
        ax.axvline(corte, color=cores["neutra"], lw=1.2, ls="--", zorder=4)
        ax.annotate("histórico", xy=(corte, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(-6, -10), textcoords="offset points", ha="right", va="top",
                    fontsize=8, color=cores["neutra"])
        ax.annotate("projetado", xy=(corte, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(6, -10), textcoords="offset points", ha="left", va="top",
                    fontsize=8, color=cores["neutra"])
        if self.weighted_ is not None:
            ax.plot(fut, self.weighted_.to_numpy(dtype=float), color="black", lw=2.2,
                    ls=":", zorder=5, label="ponderada")
        ax.set_ylabel(self._param_label())
        ax.legend(loc="best", fontsize=8, ncol=2)
        return fig

    def _render_projection(self):
        proj = self.projection_
        if proj is None:
            return
        ultimo = float(self.series.values.iloc[-1])
        tiles = {f"{self._param_label()} observado (último)": ultimo}
        for nome, df in proj.paths.items():
            tiles[f"{nome} (fim do horizonte)"] = float(df["mean"].iloc[-1])
        self.out_proj_tiles.value = self._metric_tiles(tiles)
        if self.cb_proj_plot.value:
            try:
                self.out_proj_plot.value = self._fig_html(self._fig_projecao(), stretch=True)
            except Exception as exc:  # noqa: BLE001
                self.out_proj_plot.value = (
                    f"<div class='satui-legend'>Gráfico indisponível: {exc}</div>")
        else:
            self.out_proj_plot.value = ""
        mf = proj.mean_frame()
        dados = {"período": [str(p)[:10] for p in mf.index]}
        for c in mf.columns:
            dados[str(c)] = mf[c].to_numpy(dtype=float)
        if self.weighted_ is not None:
            dados["ponderada"] = self.weighted_.to_numpy(dtype=float)
        self.out_proj_tabela.value = (
            self._df_html(pd.DataFrame(dados), max_height="320px", precision=5)
            + "<div class='satui-legend'>Trajetória <b>central</b> de cada cenário na escala "
              f"do parâmetro ({self._param_label()}). A banda de cada período está na "
              "exportação, em formato longo.</div>")

    def _render_weighted(self):
        if self.projection_ is None:
            return
        if self.weighted_ is None:
            self.out_pond_status.value = (
                "<div class='satui-notice'>Projeção ponderada indisponível — confira os "
                "<b>pesos</b> dos cenários no topo da aba.</div>")
            self.out_pond_tiles.value = self.out_pond_tabela.value = ""
            return
        w = self.weighted_
        ultimo = float(self.series.values.iloc[-1])
        fim = float(w.iloc[-1])
        pico = float(np.nanmax(w.to_numpy(dtype=float)))
        self.out_pond_status.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ Curva única calculada "
            "com os pesos "
            + ", ".join(f"<b>{s.name}</b> {float(s.probability or 0):.0%}"
                        for s in self.scenarios_.scenarios) + ".</div>")
        self.out_pond_tiles.value = self._metric_tiles({
            "último observado": ultimo, "ponderada (fim)": fim,
            "variação (p.p.)": (fim - ultimo) * 100.0, "máximo no horizonte": pico})
        tab = pd.DataFrame({
            "período": [str(p)[:10] for p in w.index],
            "ponderada": w.to_numpy(dtype=float),
            "variação vs. último observado (p.p.)": (w.to_numpy(dtype=float) - ultimo) * 100.0,
        })
        self.out_pond_tabela.value = self._df_html(
            tab, max_height="300px", precision=5,
            fmt_cols={"variação vs. último observado (p.p.)": "{:+.2f}"})

    # ------------------------------------------------------------------ exportação
    def projection_frame(self, incluir_ponderada: bool = True) -> pd.DataFrame:
        """A projeção em **formato longo**, pronta para o processo seguinte.

        Uma linha por (cenário, período) com ``mean``/``lower``/``upper`` na escala
        do parâmetro, identificada por ``parametro`` e ``segmento``. Com
        ``incluir_ponderada``, acrescenta as linhas do cenário ``'ponderado'`` (a
        curva única; sem banda, por ser combinação de cenários).
        """
        if self.projection_ is None:
            raise RuntimeError("nada a exportar: rode a projeção antes.")
        df = self.projection_.to_frame()
        if incluir_ponderada and self.weighted_ is not None:
            w = pd.DataFrame({
                "cenario": "ponderado",
                "periodo": list(self.weighted_.index),
                "mean": self.weighted_.to_numpy(dtype=float),
                "lower": np.nan, "upper": np.nan, "mean_link": np.nan})
            df = pd.concat([df, w[[c for c in df.columns if c in w.columns]]],
                           ignore_index=True)
        df.insert(0, "parametro", self._param_label())
        seg = self.series.segment if self.series is not None else ""
        df.insert(1, "segmento", seg or "—")
        df["alpha"] = float(self.projection_.alpha)
        df["modelo"] = self.fit_.model_name if self.fit_ is not None else "—"
        self.projection_table_ = df
        return df

    def _on_scen_export(self, b):
        try:
            df = self.projection_frame()
        except RuntimeError as exc:
            self.out_export_status.value = f"<div class='satui-notice'>{exc}</div>"
            return
        sep, dec = {"tsv": ("\t", "."), "csv": (",", "."),
                    "csv_br": (";", ",")}[self.dd_export_fmt.value]
        self.ta_scen_csv.value = df.to_csv(sep=sep, index=False, decimal=dec)
        self.out_export_status.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ "
            f"{len(df)} linha(s) prontas — selecione tudo (Ctrl+A) e copie (Ctrl+C). O mesmo "
            "quadro está em <code>ui.projection_frame()</code>.</div>")
        self._log(f"[projeção] exportadas {len(df)} linha(s) da projeção.")

    # ------------------------------------------------------------------ invalidação
    def _clear_cenarios_outputs(self):
        """Zera a aba inteira de cenários (dados novos ⇒ cenários antigos sem sentido)."""
        self._scen_pesos = {}
        for w in ("out_scen_padrao", "out_scen_choque", "out_scen_manual", "out_scen_resumo",
                  "out_scen_tabela", "out_scen_plot", "out_export_status"):
            widget = getattr(self, w, None)
            if widget is not None:
                widget.value = ""
        for w in ("ta_scen_paste", "ta_scen_csv"):
            widget = getattr(self, w, None)
            if widget is not None:
                widget.value = ""
        self._invalidate_cenarios("os dados mudaram")
        self._render_scen_notice()

    def _invalidate_cenarios(self, motivo="o modelo vigente mudou"):
        """Invalida **a projeção** (não os cenários: a trajetória colada é do usuário)."""
        tinha = self.projection_ is not None
        self.projection_ = None
        self.weighted_ = None
        self.projection_table_ = None
        if getattr(self, "out_proj_tabela", None) is None:   # ainda em construção
            return
        for w in ("out_proj_tiles", "out_proj_plot", "out_proj_tabela", "out_pond_status",
                  "out_pond_tiles", "out_pond_tabela"):
            getattr(self, w).value = ""
        self.out_proj_status.value = (
            f"<div class='satui-notice'>⚠️ <b>Projeção desatualizada</b> — {motivo}. "
            "Clique em <i>Projetar</i>.</div>") if tinha else ""

    # ==================================================================
    # Aba Backtest — o argumento de defesa perante a validação
    # ==================================================================
    def _build_tab_backtest(self):
        """Cartões da aba **Backtest** (devolve a tupla de filhos do VBox)."""
        # --- card: configuração e custo -------------------------------------
        self.sl_bt_min_train = W.BoundedIntText(
            value=0, min=0, max=500, description="mín. de treino (0 = automático):",
            style={"description_width": "initial"}, layout=W.Layout(width="330px"))
        self.sl_bt_horizon = W.BoundedIntText(
            value=min(6, int(self.sl_horizon.value)), min=1, max=60, description="horizonte:",
            style={"description_width": "initial"}, layout=W.Layout(width="170px"))
        self.fl_bt_alpha = W.BoundedFloatText(
            value=0.10, min=0.01, max=0.50, step=0.05, description="α da banda:",
            style={"description_width": "initial"}, layout=W.Layout(width="170px"))
        self.sl_bt_step = W.BoundedIntText(
            value=1, min=1, max=12, description="passo entre origens:",
            style={"description_width": "initial"}, layout=W.Layout(width="240px"))
        self.sl_bt_sims = W.BoundedIntText(
            value=300, min=20, max=5000, step=50, description="simulações por janela:",
            style={"description_width": "initial"}, layout=W.Layout(width="260px"))
        self.btn_bt_info = W.Button(
            description="Conferir as janelas", icon="calculator",
            layout=W.Layout(width="auto", min_width="200px"),
            tooltip="Conta quantas janelas o backtest vai reestimar com estes parâmetros.")
        self.btn_bt_info.on_click(lambda b: self._render_bt_info())
        self.out_bt_info = W.HTML()
        for w in (self.sl_bt_min_train, self.sl_bt_horizon, self.sl_bt_step):
            w.observe(lambda c: self._render_bt_info(), names="value")
        card_cfg = W.VBox([
            W.HTML("<div class='satui-h'>Backtest da projeção — configuração</div>"),
            W.HTML("<div class='satui-help'><div class='ttl'>O que este backtest responde</div>"
                   "O erro fora da amostra da aba <b>Seleção</b> mede a <b>trajetória "
                   "central</b>. Aqui testamos a outra metade da projeção: a <b>banda</b>. "
                   "O procedimento reestima o modelo janela a janela, projeta o horizonte "
                   "com a macro <b>efetivamente observada</b> e conta quantas vezes o "
                   "realizado caiu fora do intervalo. Uma banda de 90% honesta erra ~10% das "
                   "vezes: errar muito menos é <b>conservadorismo caro</b>, errar muito mais "
                   "é <b>subestimar a incerteza</b> — e é isso que a validação independente "
                   "vai perguntar.</div>"),
            W.HBox([self.sl_bt_min_train, self.sl_bt_horizon, self.fl_bt_alpha],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.sl_bt_step, self.sl_bt_sims, self.btn_bt_info],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_bt_info,
        ])
        card_cfg.add_class("satui-card")

        # --- card: rodar -------------------------------------------------------
        self.btn_backtest = W.Button(
            description="Rodar backtest", icon="history", button_style="success",
            layout=W.Layout(width="auto", min_width="190px"),
            tooltip="Reestima o modelo vigente janela a janela e mede a cobertura dos "
                    "intervalos (Kupiec e Christoffersen).")
        self.btn_backtest.on_click(self._on_backtest)
        self.out_bt_notice = W.HTML()
        self.out_bt_status = W.HTML()
        self.out_bt_progress = W.HTML()
        self.out_bt_timer = W.HTML()
        self.out_bt_resumo = W.HTML()
        card_run = W.VBox([
            W.HTML("<div class='satui-h'>Rodar</div>"),
            W.HTML("<div class='satui-legend'>Cada janela é um <b>reajuste completo</b> mais "
                   "uma projeção simulada — é a ação mais cara da interface. O modelo "
                   "testado é o <b>vigente</b> (o da aba Especificação ou o adotado na "
                   "Seleção), reconstruído do zero em cada origem, sem ver o futuro.</div>"),
            W.HBox([self.btn_backtest], layout=W.Layout(align_items="center")),
            self.out_bt_notice, self.out_bt_status, self.out_bt_progress, self.out_bt_timer,
            self.out_bt_resumo,
        ])
        card_run.add_class("satui-card")

        # --- card: veredito ------------------------------------------------------
        self.out_bt_placar = W.HTML()
        self.out_bt_leitura = W.HTML()
        card_veredito = W.VBox([
            W.HTML("<div class='satui-h'>Veredito</div>"),
            self.out_bt_placar, self.out_bt_leitura,
        ])
        card_veredito.add_class("satui-card")

        # --- card: erro por horizonte ---------------------------------------------
        self.out_bt_erros = W.HTML()
        card_erros = W.VBox([
            W.HTML("<div class='satui-h'>Erro por horizonte</div>"),
            W.HTML("<div class='satui-legend'>O erro <b>cresce</b> com o passo à frente — é "
                   "assim que deve ser. O que interessa é o formato da curva: um salto "
                   "abrupto num passo específico costuma indicar sazonalidade não modelada "
                   "ou defasagem mal escolhida. O <b>viés</b> (média do erro) revela "
                   "projeção sistematicamente otimista ou pessimista.</div>"),
            self.out_bt_erros,
        ])
        card_erros.add_class("satui-card")

        # --- card: cobertura --------------------------------------------------------
        self.out_bt_cobertura = W.HTML()
        card_cob = W.VBox([
            W.HTML("<div class='satui-h'>Cobertura dos intervalos</div>"),
            W.HTML("<div class='satui-help'><div class='ttl'>Como ler os dois testes</div>"
                   "<b>Kupiec (POF)</b> — H0: a taxa de violação é a nominal (α). "
                   "<b>p pequeno rejeita</b>: a banda tem largura errada. "
                   "<b>Christoffersen</b> — H0: as violações são <b>independentes</b> no "
                   "tempo. p pequeno indica violações em <i>cluster</i>: a banda até acerta "
                   "na média, mas falha justamente quando o ciclo vira — o pior momento "
                   "possível. Passar nos dois é o que sustenta o intervalo perante a "
                   "validação independente.</div>"),
            self.out_bt_cobertura,
        ])
        card_cob.add_class("satui-card")

        # --- card: gráfico ------------------------------------------------------------
        self.dd_bt_passo = W.Dropdown(
            options=[], description="passo à frente:",
            style={"description_width": "initial"}, layout=W.Layout(width="240px"))
        self.dd_bt_passo.observe(lambda c: self._render_bt_plot(), names="value")
        self.out_bt_plot = W.HTML()
        card_plot = W.VBox([
            W.HTML("<div class='satui-h'>Realizado × previsto, janela a janela</div>"),
            W.HTML("<div class='satui-legend'>Para o passo escolhido: o realizado, a "
                   "projeção daquela origem e a banda. Os pontos destacados são as "
                   "<b>violações</b> — vale olhar <i>quando</i> elas acontecem, não só "
                   "quantas.</div>"),
            W.HBox([self.dd_bt_passo], layout=W.Layout(align_items="center")),
            self.out_bt_plot,
        ])
        card_plot.add_class("satui-card")

        return (card_cfg, card_run, card_veredito, card_erros, card_cob, card_plot)

    # ------------------------------------------------------------------ dimensionamento
    def _bt_params(self) -> dict:
        """Parâmetros e nº de janelas do backtest (mesma conta do walk-forward)."""
        n = len(self.series) if self.series is not None else 0
        horizonte = int(self.sl_bt_horizon.value)
        step = max(1, int(self.sl_bt_step.value))
        min_train = int(self.sl_bt_min_train.value) or max(24, n // 2)
        origens = n - horizonte + 1 - min_train
        janelas = int(math.ceil(origens / step)) if origens > 0 else 0
        return {"n": n, "horizonte": horizonte, "step": step, "min_train": min_train,
                "janelas": janelas, "pontos": janelas * horizonte,
                "alpha": float(self.fl_bt_alpha.value), "sims": int(self.sl_bt_sims.value)}

    def _render_bt_info(self):
        """Dimensiona o backtest e avisa quando ele não vai ter poder estatístico."""
        if getattr(self, "out_bt_info", None) is None:
            return None
        if self.series is None:
            self.out_bt_info.value = (
                "<div class='satui-legend'>Carregue uma série na aba <b>Série</b> antes de "
                "dimensionar o backtest.</div>")
            return None
        p = self._bt_params()
        esperadas = p["pontos"] * p["alpha"]
        self.out_bt_info.value = self._metric_tiles({
            "observações": p["n"], "mín. de treino": p["min_train"],
            "janelas": p["janelas"], "pontos testados": p["pontos"],
            "cobertura nominal": f"{(1 - p['alpha']) * 100:.0f}%",
            "violações esperadas": round(esperadas, 1)})
        avisos = []
        if p["janelas"] <= 0:
            avisos.append(
                "❌ <b>Não sobra janela</b>: com mínimo de treino "
                f"{p['min_train']} e horizonte {p['horizonte']} não há origem possível em "
                f"{p['n']} observações. Reduza o horizonte ou o mínimo de treino.")
        elif esperadas < 3:
            avisos.append(
                f"⚠️ Só <b>{esperadas:.1f}</b> violação(ões) esperada(s) em "
                f"{p['pontos']} pontos: com tão poucos eventos o teste de Kupiec quase nunca "
                "rejeita — <b>não rejeitar aqui não é aprovação</b>. Amplie a janela "
                "(mínimo de treino menor) ou trate o resultado como indicativo.")
        if p["janelas"] * p["sims"] > 60000:
            avisos.append(
                f"ℹ️ {p['janelas']} janela(s) × {p['sims']} simulações: conte alguns minutos. "
                "Ligue <i>Manter cluster ativo</i> se estiver no Databricks.")
        self.out_bt_info.value += "".join(
            f"<div class='satui-notice' style='margin-top:8px'>{a}</div>" for a in avisos)
        return p

    def _bt_prog(self, key, label, status, detail=""):
        """Cria/atualiza a linha ``key`` da tabela de progresso do backtest."""
        self._prog(self._bt_steps, self.out_bt_progress, "Progresso do backtest",
                   key, label, status, detail)

    # ------------------------------------------------------------------ execução
    def _on_backtest(self, b):
        if self.fit_ is None or self.model_ is None:
            self.out_bt_status.value = (
                "<div class='satui-notice'>Nenhum <b>modelo vigente</b>: ajuste na aba "
                "<b>Especificação</b> ou adote uma especificação na aba <b>Seleção</b>.</div>")
            return
        p = self._render_bt_info()
        if p is None or p["janelas"] <= 0:
            self.out_bt_status.value = (
                "<div class='satui-notice'>Sem janela de validação: reduza o <b>horizonte</b> "
                "ou o <b>mínimo de treino</b>.</div>")
            return
        self._bt_steps = []
        self.out_bt_status.value = ""
        rot_wf = f"Reestimar e projetar {p['janelas']} janela(s) de {p['horizonte']} passo(s)"
        with self._busy(self.btn_backtest, self.btn_fit_now, self.btn_search, self.btn_project,
                        self.btn_diag), \
                self._cronometro(self.out_bt_timer, "backtest em andamento") as decorrido:
            try:
                self._bt_prog("wf", rot_wf, "run")
                wf = _E.selection.backtest_projection(
                    self.model_, self.series, self.macro,
                    min_train=int(self.sl_bt_min_train.value) or None,
                    horizon=p["horizonte"], step=p["step"], alpha=p["alpha"],
                    n_sims=p["sims"], seed=0)
            except Exception as exc:  # noqa: BLE001
                self._prog_erro(self._bt_steps, self.out_bt_progress,
                                "Progresso do backtest", exc)
                self.out_bt_status.value = (
                    "<div class='satui-notice'>✗ O backtest falhou — veja o <b>Console</b> "
                    f"(rodapé): {type(exc).__name__}.</div>")
                self._log(f"[backtest] ERRO: {type(exc).__name__}: {exc}")
                return
            secs = decorrido()
        self._bt_prog("wf", rot_wf, "ok", f"{int(wf.get('n_windows', 0))} janela(s) · "
                                          f"{self._fmt_dur(secs)}")
        cov = wf.get("coverage")
        self._bt_prog("cov", "Cobertura, Kupiec e Christoffersen",
                      "ok" if cov is not None and len(cov) else "skip",
                      "" if cov is not None and len(cov) else "sem bandas registradas")
        self.backtest_ = wf
        self.coverage_ = cov
        self._bt_secs = secs
        self._render_backtest()
        self._refresh_bar()
        n_win = int(wf.get("n_windows", 0))
        self._log(f"[backtest] {n_win} janela(s) · RMSE {wf.get('rmse', float('nan')):.5f} · "
                  f"{self._fmt_dur(secs)}")

    # ------------------------------------------------------------------ leitura
    def _bt_erros_por_passo(self, bands) -> pd.DataFrame:
        """RMSE/MAE/MAPE e viés por passo à frente (mais a linha agregada)."""
        linhas = []
        for passo, g in bands.groupby("passo", sort=True):
            real = g["real"].to_numpy(dtype=float)
            prev = g["previsto"].to_numpy(dtype=float)
            m = np.isfinite(real) & np.isfinite(prev)
            if not m.any():
                continue
            met = _E.selection._oos_metrics(real[m], prev[m])
            linhas.append({"passo": str(int(passo)), "n": int(m.sum()),
                           "RMSE": met["rmse"], "MAE": met["mae"], "MAPE": met["mape"],
                           "viés": float(np.mean(real[m] - prev[m]))})
        real = bands["real"].to_numpy(dtype=float)
        prev = bands["previsto"].to_numpy(dtype=float)
        m = np.isfinite(real) & np.isfinite(prev)
        if m.any():
            met = _E.selection._oos_metrics(real[m], prev[m])
            linhas.append({"passo": "todos", "n": int(m.sum()), "RMSE": met["rmse"],
                           "MAE": met["mae"], "MAPE": met["mape"],
                           "viés": float(np.mean(real[m] - prev[m]))})
        return pd.DataFrame(linhas)

    @staticmethod
    def _css_pnaorejeita(v):
        """p-valor de teste cuja **nula é o resultado desejado** (Kupiec,
        Christoffersen): p grande (não rejeita) é o bom resultado — o oposto de
        :meth:`_css_pvalor`."""
        try:
            p = float(v)
        except (TypeError, ValueError):
            return ""
        if p != p:
            return "color:var(--muted)"
        if p > 0.10:
            return "color:var(--ok-tx);font-weight:600"
        if p > 0.05:
            return "color:var(--warn-tx);font-weight:600"
        return "color:var(--bad-tx);background-color:var(--bad-bg);font-weight:600"

    @staticmethod
    def _css_cobertura_factory(nominal):
        """Colore a cobertura empírica pela distância da nominal (fecha sobre ela)."""
        alvo = float(nominal)

        def _css(v):
            try:
                x = float(v)
            except (TypeError, ValueError):
                return ""
            if x != x:
                return "color:var(--muted)"
            d = abs(x - alvo)
            if d <= 0.03:
                return "color:var(--ok-tx);background-color:var(--ok-bg);font-weight:600"
            if d <= 0.07:
                return "color:var(--warn-tx);background-color:var(--warn-bg);font-weight:600"
            return "color:var(--bad-tx);background-color:var(--bad-bg);font-weight:600"

        return _css

    def _bt_blocos(self, cov, wf) -> list:
        """Placar do backtest: cobertura, Kupiec e Christoffersen (o pior passo)."""
        tot = cov[cov["passo"] == "todos"]
        blocos = []
        if not len(tot):
            return blocos
        r = tot.iloc[0]
        nominal = float(r["nominal"])
        emp = float(r["cobertura"]) if pd.notna(r["cobertura"]) else np.nan
        n = int(r["n"])
        viol = int(r["violacoes"])
        d = abs(emp - nominal) if np.isfinite(emp) else np.nan
        nivel = "na" if not np.isfinite(d) else ("ok" if d <= 0.03 else
                                                 ("warn" if d <= 0.07 else "bad"))
        blocos.append({
            "bloco": "Cobertura das bandas", "nivel": nivel,
            "veredito": ("—" if not np.isfinite(emp) else f"{emp:.1%} vs {nominal:.0%} nominal"),
            "detalhe": f"{viol} violação(ões) em {n} ponto(s) testado(s)",
            "falhas": 0, "n": 1, "teste": "cobertura"})
        p_kup = float(r["kupiec_pvalue"]) if pd.notna(r["kupiec_pvalue"]) else np.nan
        nivel_k = "na" if not np.isfinite(p_kup) else ("ok" if p_kup > 0.05 else "bad")
        blocos.append({
            "bloco": "Kupiec (POF)", "nivel": nivel_k,
            "veredito": ("inconclusivo" if not np.isfinite(p_kup) else
                         ("não rejeita a cobertura nominal" if p_kup > 0.05
                          else "rejeita a cobertura nominal")),
            "detalhe": (f"LR = {float(r['kupiec_stat']):.3f} · p = {p_kup:.4f}"
                        if np.isfinite(p_kup) else "sem observações válidas"),
            "falhas": 0, "n": 1, "teste": "Kupiec"})
        passos = cov[cov["passo"] != "todos"]
        ps = passos["christoffersen_pvalue"].astype(float)
        if len(ps) and ps.notna().any():
            pior = passos.loc[ps.idxmin()]
            p_chr = float(pior["christoffersen_pvalue"])
            nivel_c = "ok" if p_chr > 0.05 else "bad"
            detalhe = (f"pior passo: {pior['passo']} · LR = "
                       f"{float(pior['christoffersen_stat']):.3f} · p = {p_chr:.4f}")
            veredito = ("violações independentes" if p_chr > 0.05
                        else "violações em cluster")
        else:
            nivel_c, veredito = "na", "inconclusivo"
            detalhe = ("sem violações suficientes para identificar a cadeia — "
                       "trate como não testado")
        blocos.append({"bloco": "Christoffersen (independência)", "nivel": nivel_c,
                       "veredito": veredito, "detalhe": detalhe, "falhas": 0, "n": 1,
                       "teste": "Christoffersen"})
        return blocos

    def _bt_leitura_html(self, cov, wf) -> str:
        """A frase de defesa do modelo — em português, com os números na mão."""
        tot = cov[cov["passo"] == "todos"]
        if not len(tot):
            return ""
        r = tot.iloc[0]
        nominal = float(r["nominal"])
        emp = float(r["cobertura"]) if pd.notna(r["cobertura"]) else np.nan
        n, viol = int(r["n"]), int(r["violacoes"])
        if not np.isfinite(emp) or n == 0:
            return ("<div class='satui-help'><div class='ttl'>Leitura</div>Nenhum ponto com "
                    "banda válida: sem cobertura para avaliar. Aumente as <b>simulações por "
                    "janela</b> ou reduza o horizonte — sem banda não há o que defender."
                    "</div>")
        p_kup = float(r["kupiec_pvalue"]) if pd.notna(r["kupiec_pvalue"]) else np.nan
        frases = [f"As bandas de <b>{nominal:.0%}</b> cobriram <b>{emp:.1%}</b> dos {n} "
                  f"pontos projetados fora da amostra ({viol} violação(ões))."]
        if not np.isfinite(p_kup):
            frases.append("O teste de <b>Kupiec</b> não pôde ser calculado — trate a "
                          "cobertura como <b>não testada</b>.")
            recado = ("Sem teste, o intervalo não está validado: amplie a janela de backtest "
                      "antes de levar a banda para a governança.")
        elif p_kup > 0.05 and abs(emp - nominal) <= 0.07:
            frases.append(f"<b>Kupiec não rejeita</b> a cobertura nominal (p = {p_kup:.3f}): "
                          "a diferença é compatível com variação amostral.")
            recado = ("Este é o resultado que sustenta o intervalo perante a validação "
                      "independente — registre a janela, o horizonte e o α junto do número.")
        elif emp < nominal:
            frases.append(f"<b>Kupiec rejeita</b> (p = {p_kup:.3f}): há <b>violações demais</b> "
                          "— o intervalo é estreito demais para a incerteza real.")
            recado = ("A banda subestima o risco. Caminhos: covariância <b>HAC</b>, incluir a "
                      "dinâmica que sobrou no resíduo (aba <b>Diagnóstico</b>) ou aumentar o "
                      "número de simulações; se o resíduo tem <i>clusters</i> de "
                      "volatilidade, a banda do cenário base nunca cobrirá o estresse — use "
                      "o cenário adverso para essa pergunta.")
        else:
            frases.append(f"<b>Kupiec rejeita</b> (p = {p_kup:.3f}): há <b>violações de "
                          "menos</b> — o intervalo é largo demais.")
            recado = ("Bandas conservadoras demais custam caro: tudo cabe dentro delas e o "
                      "intervalo deixa de informar. Vale rever o horizonte, o α e o pool de "
                      "resíduos usado na simulação.")
        rmse = wf.get("rmse", np.nan)
        if np.isfinite(rmse):
            frases.append(f"A trajetória central errou, em média, <b>{float(rmse):.5f}</b> "
                          f"(RMSE, escala de {self._param_label()}) em "
                          f"{int(wf.get('n_windows', 0))} janela(s).")
        return ("<div class='satui-help'><div class='ttl'>Leitura</div>"
                + " ".join(frases) + f"<div style='margin-top:6px'>{recado}</div></div>")

    def _render_backtest(self):
        """Placar, leitura, erro por horizonte, cobertura e gráfico."""
        wf = self.backtest_
        if wf is None:
            return
        bands = wf.get("bands")
        cov = self.coverage_
        self.out_bt_resumo.value = self._metric_tiles({
            "janelas": int(wf.get("n_windows", 0)),
            "RMSE": float(wf.get("rmse", np.nan)),
            "MAE": float(wf.get("mae", np.nan)),
            "MAPE": f"{float(wf.get('mape', np.nan)) * 100:.1f}%",
            "tempo": self._fmt_dur(self._bt_secs) if self._bt_secs else "—"})
        if bands is None or not len(bands):
            self.out_bt_status.value = (
                "<div class='satui-notice'>O backtest rodou, mas nenhuma banda pôde ser "
                "calculada — sem bandas não há teste de cobertura.</div>")
            return
        self.out_bt_erros.value = self._df_html(
            self._bt_erros_por_passo(bands), max_height="280px",
            fmt_cols={"RMSE": "{:.5f}", "MAE": "{:.5f}", "viés": "{:+.5f}"},
            pct_cols=["MAPE"], precision=5)

        if cov is not None and len(cov):
            nominal = float(cov["nominal"].iloc[0])
            vis = pd.DataFrame({
                "passo": [str(p) for p in cov["passo"]],
                "n": cov["n"].to_numpy(dtype=int),
                "violações": cov["violacoes"].to_numpy(dtype=int),
                "cobertura nominal": cov["nominal"].to_numpy(dtype=float),
                "cobertura empírica": cov["cobertura"].to_numpy(dtype=float),
                "Kupiec (LR)": cov["kupiec_stat"].to_numpy(dtype=float),
                "p Kupiec": cov["kupiec_pvalue"].to_numpy(dtype=float),
                "Christoffersen (LR)": cov["christoffersen_stat"].to_numpy(dtype=float),
                "p Christoffersen": cov["christoffersen_pvalue"].to_numpy(dtype=float),
                "veredito": [{True: "✓ cobertura ok", False: "✗ cobertura fora"}.get(
                    self._ok3(v), "—") for v in cov["ok"]],
            })
            self.out_bt_cobertura.value = self._df_html(
                vis, max_height="320px",
                color_map={"cobertura empírica": self._css_cobertura_factory(nominal),
                           "p Kupiec": self._css_pnaorejeita,
                           "p Christoffersen": self._css_pnaorejeita,
                           "veredito": self._css_coerencia},
                pct_cols=["cobertura nominal", "cobertura empírica"],
                fmt_cols={"Kupiec (LR)": "{:.3f}", "Christoffersen (LR)": "{:.3f}"},
                precision=4)
            blocos = self._bt_blocos(cov, wf)
            self.out_bt_placar.value = self._placar_html(blocos) if blocos else ""
            self.out_bt_leitura.value = self._bt_leitura_html(cov, wf)
            ruins = [b["bloco"] for b in blocos if b["nivel"] == "bad"]
            self.out_bt_status.value = (
                "<div class='satui-legend' style='color:var(--ok-ink)'>✓ Backtest concluído "
                "sem reprovação nos testes de cobertura.</div>" if not ruins else
                "<div class='satui-legend' style='color:var(--bad-ink)'>Atenção: "
                f"<b>{', '.join(ruins)}</b> — veja a leitura abaixo.</div>")
        else:
            self.out_bt_cobertura.value = (
                "<div class='satui-legend'>Cobertura indisponível para estes parâmetros.</div>")

        passos = sorted({int(p) for p in bands["passo"].unique()})
        self.dd_bt_passo.options = [(f"{p} período(s) à frente", p) for p in passos]
        if passos:
            self.dd_bt_passo.value = passos[0]
        self._render_bt_plot()

    def _render_bt_plot(self):
        """Realizado × previsto do passo escolhido, com as violações marcadas."""
        if getattr(self, "out_bt_plot", None) is None:
            return
        wf = self.backtest_
        bands = wf.get("bands") if wf else None
        if bands is None or not len(bands) or self.dd_bt_passo.value is None:
            self.out_bt_plot.value = ""
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - depende do ambiente
            self.out_bt_plot.value = f"<div class='satui-legend'>matplotlib indisponível: {exc}</div>"
            return
        passo = int(self.dd_bt_passo.value)
        sub = bands[bands["passo"] == passo].sort_values("periodo")
        if not len(sub):
            self.out_bt_plot.value = ""
            return
        cores = _E.report._palette()
        x = _E.report._to_ts(pd.Index(list(sub["periodo"])))
        real = sub["real"].to_numpy(dtype=float)
        prev = sub["previsto"].to_numpy(dtype=float)
        lo = sub["lower"].to_numpy(dtype=float)
        hi = sub["upper"].to_numpy(dtype=float)
        viol = sub["violacao"].to_numpy(dtype=float) == 1.0
        nivel = int(round((1 - float(wf.get("alpha", 0.10))) * 100))
        fig, ax = plt.subplots(figsize=(10.5, 4.4))
        if np.isfinite(lo).any():
            ax.fill_between(x, lo, hi, color=cores["primaria"], alpha=0.15,
                            label=f"banda {nivel}%")
        ax.plot(x, real, color=cores["neutra"], lw=1.6, label="realizado")
        ax.plot(x, prev, color=cores["primaria"], lw=1.8, label="previsto")
        if viol.any():
            ax.scatter(np.asarray(x)[viol], real[viol], color=cores["secundaria"], s=42,
                       zorder=5, label=f"violações ({int(viol.sum())})")
        ax.set_title(f"Backtest — {passo} período(s) à frente ({self._param_label()})",
                     fontsize=11)
        ax.set_ylabel(self._param_label())
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        self.out_bt_plot.value = self._fig_html(fig, stretch=True)

    # ------------------------------------------------------------------ invalidação
    def _clear_backtest_outputs(self):
        """Zera a aba de backtest (dados novos ⇒ backtest antigo sem sentido)."""
        for w in ("out_bt_info", "out_bt_resumo", "out_bt_progress", "out_bt_timer"):
            widget = getattr(self, w, None)
            if widget is not None:
                widget.value = ""
        self._bt_steps = []
        self._bt_secs = None
        self._invalidate_backtest("os dados mudaram")

    def _invalidate_backtest(self, motivo="o modelo vigente mudou"):
        """Invalida o backtest (ele pertencia ao ajuste anterior)."""
        tinha = self.backtest_ is not None
        self.backtest_ = None
        self.coverage_ = None
        if getattr(self, "out_bt_cobertura", None) is None:   # ainda em construção
            return
        for w in ("out_bt_placar", "out_bt_leitura", "out_bt_erros", "out_bt_cobertura",
                  "out_bt_plot", "out_bt_status"):
            getattr(self, w).value = ""
        self.dd_bt_passo.options = []
        self.out_bt_notice.value = (
            f"<div class='satui-notice'>⚠️ <b>Backtest desatualizado</b> — {motivo}. "
            "Clique em <i>Rodar backtest</i>.</div>") if tinha else ""

    # ==================================================================
    # Estudo completo em um clique (as cinco chamadas numa passada)
    # ==================================================================
    def _study_prog(self, key, label, status, detail=""):
        """Cria/atualiza a linha ``key`` da tabela de progresso do estudo completo."""
        self._prog(self._study_steps, self.out_study_progress,
                   "Progresso do estudo completo", key, label, status, detail)

    def _on_run_study(self, b):
        """Monta a :class:`StudyConfig` da tela, roda :func:`run_study` e **preenche
        todas as abas** com o resultado (busca, ajuste, diagnóstico, cenários e
        projeção). Equivale às cinco chamadas do fluxo manual."""
        if self.series is None:
            self.out_study_status.value = (
                "<div class='satui-notice'>Sem série carregada — traga os dados na aba "
                "<b>Série</b> (ou carregue o estudo de referência).</div>")
            return
        if self.macro is None or not self.candidates():
            self.out_study_status.value = (
                "<div class='satui-notice'>O estudo completo <b>busca</b> sobre a macro: "
                "carregue as variáveis macro e marque ao menos uma <b>candidata</b> na "
                "matriz de sinais acima.</div>")
            return
        try:
            cfg = self.to_config()
            info = self._grid_size()
        except ValueError as exc:
            self.out_study_status.value = f"<div class='satui-notice'>{exc}</div>"
            return
        if info["janelas"] <= 0:
            self.out_study_status.value = (
                "<div class='satui-notice'>Sem janela de validação fora da amostra: reduza o "
                "<b>horizonte</b> ou o <b>mínimo de treino</b> no cartão da grade.</div>")
            return

        self._study_steps = []
        self.out_study_status.value = self.out_study_resumo.value = ""
        self._study_prog("cfg", "Montar a configuração da tela (StudyConfig)", "ok",
                         f"{len(cfg.candidates)} candidata(s) · até {info['efetivo']} "
                         f"especificações · critério {cfg.criterion}")
        rot = "Rodar o estudo — busca, ajuste, diagnóstico, cenários e projeção"
        self._study_prog("run", rot, "run", f"~{info['ajustes']} ajustes no pior caso")
        with self._busy(self.btn_run_study, self.btn_fit_now, self.btn_search,
                        self.btn_pick_fit, self.btn_dm, self.btn_diag, self.btn_project,
                        self.btn_backtest), \
                self._cronometro(self.out_study_timer, "estudo em andamento") as decorrido:
            try:
                res = _E.config.run_study(cfg, self.series, self.macro,
                                          make_report=bool(self.cb_study_report.value))
            except Exception as exc:  # noqa: BLE001
                self._prog_erro(self._study_steps, self.out_study_progress,
                                "Progresso do estudo completo", exc)
                self.out_study_status.value = (
                    f"<div class='satui-notice'>✗ O estudo não completou: {exc}</div>")
                self._log(f"[estudo] ERRO: {type(exc).__name__}: {exc}")
                return
            secs = decorrido()
        self._study_secs = secs
        self._study_prog("run", rot, "ok", self._fmt_dur(secs))
        self._adota_estudo(res, secs)

    def _adota_estudo(self, res, secs):
        """Espalha o :class:`StudyResult` pelas abas, na ordem em que ele foi produzido."""
        cfg = res.config
        # (1) busca — o ranking e o mapa describe() → Specification do seletor manual.
        # A grade não volta dentro do StudyResult: ela é **remontada** aqui (custo
        # zero, nenhuma estimação) para que a aba Seleção continue permitindo
        # adotar outra especificação.
        self.search_ = res.search
        self._search_secs = None
        try:
            import warnings

            with warnings.catch_warnings(record=True) as capturados:
                warnings.simplefilter("always")
                grid = _E.selection.make_grid(
                    list(cfg.candidates), lag_set=cfg.lag_set, min_vars=1,
                    max_vars=cfg.max_vars, ar_orders=cfg.ar_orders, link=cfg.link,
                    expected_signs=dict(cfg.expected_signs), seasonal=cfg.seasonal,
                    seasonal_period=cfg.seasonal_period, max_specs=cfg.max_specs)
            for a in capturados:
                self._log(f"[estudo] {a.message}")
            self._spec_por_desc = {s.describe(): s for s in grid}
        except Exception as exc:  # noqa: BLE001
            self._spec_por_desc = {}
            self._log(f"[estudo] o seletor manual da aba Seleção ficou vazio "
                      f"(a grade não pôde ser remontada): {exc}")

        # (2) ajuste — a campeã vira o modelo vigente
        self.model_, self.fit_ = res.best, res.fit
        self.selected_spec_ = res.search.best_spec
        self._clear_dirty()          # o que havia pertencia ao modelo anterior
        self._render_fit()
        self._render_search()
        rk = res.search.ranking
        n_qual = int((rk["status"] == "qualificado").sum()) if len(rk) else 0
        campea = (res.search.best_spec.describe()
                  if res.search.best_spec is not None else "—")
        self._study_prog("busca", "Busca champion-challenger", "ok",
                         f"{n_qual} qualificada(s) · campeã: {campea}")
        aic = f" · AIC {res.fit.aic:.1f}" if res.fit.aic is not None else ""
        self._study_prog("ajuste", "Ajuste da campeã na amostra cheia", "ok",
                         f"{res.fit.model_name}{aic}")

        # (3) diagnóstico — a bateria da tela (com Chow, Quandt-Andrews e VIF)
        self._on_diag(None)
        ruins = [x["bloco"] for x in (self.diag_blocks_ or []) if x["nivel"] == "bad"]
        self._study_prog(
            "diag", "Bateria de diagnóstico", "ok" if self.diagnostics_ is not None else "err",
            (f"{len(ruins)} bloco(s) reprovado(s): {', '.join(ruins)}" if ruins
             else "sem reprovações") if self.diagnostics_ is not None else "indisponível")

        # (4) cenários — os padrão que o próprio estudo montou
        with suppress(Exception):
            self.sl_scen_horizon.value = int(cfg.horizon)
        self._scen_pesos = {}
        self._adota_cenarios(res.scenarios, origem="estudo completo")
        self._study_prog("cen", "Cenários padrão", "ok",
                         f"{len(res.scenarios)} cenário(s) sobre '{cfg.stress_var}'")

        # (5) projeção — a do estudo (não reprojetamos: seria outra resposta)
        proj = res.projection
        for w, v in ((self.fl_scen_alpha, float(proj.alpha)), (self.sl_scen_sims, 2000)):
            with suppress(Exception):
                w.value = v
        self.projection_ = proj
        self.projection_table_ = None
        pesos = {s.name: float(s.probability or 0.0) for s in res.scenarios.scenarios}
        try:
            self.weighted_ = proj.weighted(pesos)
        except Exception as exc:  # noqa: BLE001
            self.weighted_ = None
            self._log(f"[estudo] projeção ponderada indisponível: {exc}")
        self._render_projection()
        self._render_weighted()
        nivel = int(round((1 - proj.alpha) * 100))
        self._study_prog("proj", "Projeção condicional aos cenários", "ok",
                         f"{proj.horizon} períodos · leque {nivel}%")
        self._study_prog("rep", "Relatório HTML de governança",
                         "ok" if res.report_html else "skip",
                         "pronto para gravar na aba Exportar" if res.report_html
                         else "não solicitado")

        self.study_ = res
        self._refresh_bar()
        self._render_export_estado()
        tiles = {"tempo": self._fmt_dur(secs), "qualificadas": n_qual,
                 "AIC": res.fit.aic, "cenários": len(res.scenarios),
                 "horizonte": int(proj.horizon)}
        qual = rk[rk["status"] == "qualificado"] if len(rk) else rk
        if len(qual) and "oos_rmse" in qual.columns:
            tiles["RMSE fora"] = float(qual["oos_rmse"].iloc[0])
        self.out_study_resumo.value = (
            self._metric_tiles(tiles)
            + "<div class='satui-legend'>Tudo já está nas abas: o ranking e as descartadas "
              "em <b>Seleção</b>, o placar dos testes em <b>Diagnóstico</b>, as trajetórias "
              "e o leque em <b>Cenários &amp; Projeção</b>, e o relatório, o MLflow e o JSON "
              "da configuração em <b>Exportar</b>. Falta só o <b>Backtest</b> — ele tem aba "
              "própria porque custa caro.</div>")
        self.out_study_status.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ Estudo "
            f"<b>{cfg.name}</b> concluído em {self._fmt_dur(secs)} — campeã "
            f"<code>{campea}</code>.</div>")
        for linha in (
                f"[estudo] '{cfg.name}' concluído em {self._fmt_dur(secs)}",
                f"[estudo] busca: {n_qual} qualificada(s) · campeã {campea}",
                f"[estudo] ajuste: {res.fit.model_name}{aic}",
                "[estudo] diagnóstico: " + (f"reprovados {', '.join(ruins)}" if ruins
                                            else "sem reprovações"),
                f"[estudo] projeção: {len(res.scenarios)} cenário(s) × {proj.horizon} "
                f"períodos (leque {nivel}%)",
                "[estudo] próximos passos: rode o Backtest e feche pela aba Exportar."):
            self._log(linha)

    # ==================================================================
    # Aba Exportar — relatório, MLflow, configuração e tabelas
    # ==================================================================
    #: tabelas oferecidas na exportação (chave → rótulo na tela)
    _TABELAS_EXPORT = (
        ("projecao", "Projeção por cenário (formato longo)"),
        ("ranking", "Ranking da busca champion-challenger"),
        ("diagnostico", "Bateria de diagnóstico do modelo vigente"),
        ("cobertura", "Cobertura dos intervalos (backtest)"),
        ("coeficientes", "Coeficientes do modelo vigente"),
        ("estacionariedade", "Relatório de estacionariedade"),
    )

    def _build_tab_exportar(self):
        """Cartões da aba **Exportar** (devolve a tupla de filhos do VBox)."""
        base = self._nome_arquivo()
        # --- card: o que está pronto ---------------------------------------
        self.out_exp_notice = W.HTML()
        self.out_exp_estado = W.HTML()
        self.btn_exp_estado = W.Button(
            description="Atualizar", icon="refresh",
            layout=W.Layout(width="auto", min_width="130px"),
            tooltip="Relê o estado da sessão (ajuste, busca, diagnóstico, projeção, "
                    "backtest).")
        self.btn_exp_estado.on_click(lambda b: self._render_export_estado())
        card_estado = W.VBox([
            W.HTML("<div class='satui-h'>O que já está pronto nesta sessão</div>"),
            W.HTML("<div class='satui-legend'>Cada saída abaixo exporta o <b>estado "
                   "corrente</b> — o relatório e o registro no MLflow levam o que existir "
                   "(ajuste, busca, projeção, backtest). O que estiver faltando aqui "
                   "simplesmente não vai junto.</div>"),
            self.out_exp_notice, self.out_exp_estado,
            W.HBox([self.btn_exp_estado], layout=W.Layout(align_items="center")),
        ])
        card_estado.add_class("satui-card")

        # --- card: relatório HTML -------------------------------------------
        self.tx_report_path = W.Text(
            value=f"relatorio_{base}.html", description="arquivo:",
            style={"description_width": "initial"}, layout=W.Layout(width="440px"))
        self.btn_report = W.Button(
            description="Gerar relatório HTML", icon="file-text-o", button_style="primary",
            layout=W.Layout(width="auto", min_width="210px"),
            tooltip="Gera o relatório de governança (especificação, coeficientes, métricas, "
                    "diagnóstico e o leque da projeção) e grava no arquivo indicado.")
        self.btn_report.on_click(self._on_report)
        self.out_report_status = W.HTML()
        card_report = W.VBox([
            W.HTML("<div class='satui-h'>Relatório HTML de governança</div>"),
            W.HTML("<div class='satui-legend'>Documento autocontido (um único arquivo, "
                   "figuras embutidas) com a <b>especificação</b>, a tabela de "
                   "<b>coeficientes</b>, as métricas, a <b>bateria de diagnóstico</b> e — se "
                   "houver projeção — o gráfico em leque. É o anexo que acompanha o modelo "
                   "na validação independente. Se o arquivo já existir, o botão pede "
                   "confirmação antes de sobrescrever.</div>"),
            W.HBox([self.tx_report_path, self.btn_report],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_report_status,
        ])
        card_report.add_class("satui-card")

        # --- card: MLflow -----------------------------------------------------
        self.tx_mlflow_exp = W.Text(
            value="", placeholder="vazio = experimento ativo da sessão",
            description="experimento:", style={"description_width": "initial"},
            layout=W.Layout(width="480px"))
        self.tx_mlflow_run = W.Text(
            value=base, description="nome do run:",
            style={"description_width": "initial"}, layout=W.Layout(width="330px"))
        self.btn_mlflow = W.Button(
            description="Registrar no MLflow", icon="database", button_style="primary",
            layout=W.Layout(width="auto", min_width="210px"),
            tooltip="Cria um run com a especificação, as métricas, o diagnóstico, a projeção "
                    "e o backtest disponíveis — mais o relatório em abas como artefato.")
        self.btn_mlflow.on_click(self._on_mlflow)
        self.out_mlflow_status = W.HTML()
        self.out_mlflow_info = W.HTML()
        card_mlflow = W.VBox([
            W.HTML("<div class='satui-h'>Registrar o run no MLflow</div>"),
            W.HTML("<div class='satui-legend'>Versiona o estudo: parâmetros (modelo, "
                   "<i>link</i>, especificação), métricas (AIC/BIC/R², erro fora da amostra, "
                   "cobertura dos intervalos), a bateria de diagnóstico, o ranking da busca, "
                   "as figuras e um relatório em abas — tudo como artefato do <i>run</i>. "
                   "Deixe o <b>experimento</b> vazio para usar o ativo da sessão. Exige "
                   "<code>mlflow</code> instalado e o <i>tracking</i> configurado (no "
                   "Databricks já vem).</div>"),
            W.HBox([self.tx_mlflow_exp, self.tx_mlflow_run],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.btn_mlflow], layout=W.Layout(align_items="center")),
            self.out_mlflow_status, self.out_mlflow_info,
        ])
        card_mlflow.add_class("satui-card")

        # --- card: StudyConfig em JSON ---------------------------------------
        self.btn_cfg_show = W.Button(
            description="Ver JSON da sessão", icon="code",
            layout=W.Layout(width="auto", min_width="195px"),
            tooltip="Serializa a configuração corrente (ui.to_config()) como JSON.")
        self.btn_cfg_show.on_click(self._on_cfg_show)
        self.tx_cfg_path = W.Text(
            value=f"{base}.json", description="arquivo:",
            style={"description_width": "initial"}, layout=W.Layout(width="380px"))
        self.btn_cfg_save = W.Button(
            description="Salvar", icon="save", button_style="success",
            layout=W.Layout(width="auto", min_width="130px"),
            tooltip="Grava o JSON da configuração no arquivo indicado.")
        self.btn_cfg_save.on_click(self._on_cfg_save)
        self.btn_cfg_load = W.Button(
            description="Carregar", icon="upload",
            layout=W.Layout(width="auto", min_width="140px"),
            tooltip="Lê o arquivo e repõe TODOS os controles da interface.")
        self.btn_cfg_load.on_click(self._on_cfg_load)
        self.btn_cfg_apply = W.Button(
            description="Aplicar o JSON abaixo", icon="check",
            layout=W.Layout(width="auto", min_width="205px"),
            tooltip="Aplica na interface o JSON editado/colado na caixa (sem passar por "
                    "arquivo).")
        self.btn_cfg_apply.on_click(self._on_cfg_apply)
        self.out_cfg_status = W.HTML()
        self.ta_config_json = W.Textarea(
            placeholder="o JSON da configuração aparece aqui — edite e clique em Aplicar, ou "
                        "salve em arquivo para versionar junto com a projeção",
            layout=W.Layout(width="99%", height="230px"))
        card_config = W.VBox([
            W.HTML("<div class='satui-h'>Configuração do estudo (JSON) — reprodutibilidade"
                   "</div>"),
            W.HTML("<div class='satui-help'><div class='ttl'>Por que salvar isto</div>"
                   "A projeção só é reproduzível se a <b>configuração que a gerou</b> viajar "
                   "junto: candidatas, sinais esperados, defasagens, ordens AR, critério, "
                   "teto de VIF, horizonte, variável de estresse e pesos dos cenários. Este "
                   "JSON é exatamente a <code>StudyConfig</code> — o mesmo objeto que o "
                   "pipeline declarativo (<code>run_study</code>) consome fora do notebook. "
                   "<b>Carregar</b> repõe todos os controles da interface; reajuste depois "
                   "para reproduzir o resultado.</div>"),
            W.HBox([self.btn_cfg_show, self.btn_cfg_apply],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.tx_cfg_path, self.btn_cfg_save, self.btn_cfg_load],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_cfg_status, self.ta_config_json,
        ])
        card_config.add_class("satui-card")

        # --- card: tabelas em CSV ---------------------------------------------
        self.dd_exp_tabela = W.Dropdown(
            options=[(rot, ch) for ch, rot in self._TABELAS_EXPORT], value="projecao",
            description="tabela:", style={"description_width": "initial"},
            layout=W.Layout(width="430px"))
        self.dd_exp_fmt = W.Dropdown(
            options=[("CSV — ponto decimal, vírgula", "csv"),
                     ("CSV — vírgula decimal, ponto e vírgula", "csv_br"),
                     ("TSV — colar no Excel", "tsv")],
            value="csv", description="formato:", style={"description_width": "initial"},
            layout=W.Layout(width="330px"))
        self.btn_exp_mostrar = W.Button(
            description="Mostrar para copiar", icon="clipboard",
            layout=W.Layout(width="auto", min_width="200px"),
            tooltip="Escreve a tabela na caixa abaixo (Ctrl+A, Ctrl+C).")
        self.btn_exp_mostrar.on_click(self._on_exp_mostrar)
        self.tx_exp_path = W.Text(
            value=f"{base}_projecao.csv", description="arquivo:",
            style={"description_width": "initial"}, layout=W.Layout(width="380px"))
        self.btn_exp_salvar = W.Button(
            description="Salvar CSV", icon="download", button_style="primary",
            layout=W.Layout(width="auto", min_width="165px"),
            tooltip="Grava a tabela escolhida no arquivo indicado.")
        self.btn_exp_salvar.on_click(self._on_exp_salvar)
        self.out_exp_tab_status = W.HTML()
        self.ta_export_tabela = W.Textarea(
            placeholder="a tabela escolhida aparece aqui — selecione tudo (Ctrl+A) e copie "
                        "(Ctrl+C)",
            layout=W.Layout(width="99%", height="200px"))
        self.dd_exp_tabela.observe(self._on_exp_tabela_change, names="value")
        card_tabelas = W.VBox([
            W.HTML("<div class='satui-h'>Tabelas do estudo em CSV</div>"),
            W.HTML("<div class='satui-legend'>As mesmas tabelas que estão nas abas, prontas "
                   "para o processo seguinte: a <b>projeção</b> em formato longo (uma linha "
                   "por cenário e período, com banda e a curva ponderada), o <b>ranking</b> "
                   "da busca com o motivo de cada descarte, a <b>bateria de diagnóstico</b>, "
                   "a <b>cobertura</b> dos intervalos e os <b>coeficientes</b>. Use o "
                   "formato de vírgula decimal para abrir direto no Excel em pt-BR.</div>"),
            W.HBox([self.dd_exp_tabela, self.dd_exp_fmt, self.btn_exp_mostrar],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            W.HBox([self.tx_exp_path, self.btn_exp_salvar],
                   layout=W.Layout(flex_flow="row wrap", align_items="center")),
            self.out_exp_tab_status, self.ta_export_tabela,
        ])
        card_tabelas.add_class("satui-card")

        self._render_export_estado()
        return (card_estado, card_report, card_mlflow, card_config, card_tabelas)

    # ------------------------------------------------------------------ estado
    def _nome_arquivo(self) -> str:
        """Nome do estudo em forma segura para arquivo (sem acento nem espaço)."""
        tx = getattr(self, "tx_nome", None)
        bruto = _sem_acento(tx.value if tx is not None else self.study_name)
        return re.sub(r"[^a-z0-9_.-]+", "_", bruto).strip("_") or "estudo"

    def _estado_blocos(self) -> list:
        """Blocos do placar "o que está pronto" (mesmo desenho do diagnóstico)."""
        def _b(bloco, nivel, veredito, detalhe):
            return {"bloco": bloco, "nivel": nivel, "veredito": veredito,
                    "detalhe": detalhe, "falhas": 0, "n": 0}

        blocos = []
        if self.series is None:
            blocos.append(_b("Série", "bad", "sem dados",
                             "carregue a série na aba Série"))
        else:
            blocos.append(_b("Série", "ok", f"{len(self.series)} períodos",
                             f"{self._param_label()} · {len(self._macro_cols())} variável(is) "
                             "macro"))
        if self.fit_ is None:
            blocos.append(_b("Modelo vigente", "warn", "nenhum ajuste",
                             "ajuste na aba Especificação ou adote uma especificação na "
                             "aba Seleção"))
        elif self._dirty_since_fit:
            blocos.append(_b("Modelo vigente", "warn", "desatualizado",
                             "a especificação mudou depois do ajuste — reajuste antes de "
                             "exportar"))
        else:
            spec = (self.selected_spec_.describe() if self.selected_spec_ is not None
                    else (self.fit_.spec.describe() if self.fit_.spec else "—"))
            blocos.append(_b("Modelo vigente", "ok", self.fit_.model_name, spec))
        if self.search_ is None:
            blocos.append(_b("Busca", "na", "não rodada",
                             "o relatório e o run saem sem o ranking"))
        else:
            rk = self.search_.ranking
            n = int((rk["status"] == "qualificado").sum()) if len(rk) else 0
            blocos.append(_b("Busca", "ok" if n else "warn", f"{n} qualificada(s)",
                             f"{len(rk)} linha(s) no ranking"))
        if self.diagnostics_ is None:
            blocos.append(_b("Diagnóstico", "na", "não rodado",
                             "rode a bateria na aba Diagnóstico"))
        else:
            ruins = [x["bloco"] for x in (self.diag_blocks_ or []) if x["nivel"] == "bad"]
            blocos.append(_b("Diagnóstico", "bad" if ruins else "ok",
                             f"{len(ruins)} reprovado(s)" if ruins else "sem reprovações",
                             ", ".join(ruins) if ruins else
                             f"{len(self.diagnostics_)} teste(s) na bateria"))
        if self.projection_ is None:
            blocos.append(_b("Projeção", "warn", "não projetada",
                             "monte os cenários e projete na aba Cenários & Projeção"))
        else:
            nivel = int(round((1 - self.projection_.alpha) * 100))
            blocos.append(_b("Projeção", "ok",
                             f"{len(self.projection_.paths)} cenário(s)",
                             f"{self.projection_.horizon} períodos · leque {nivel}%"))
        if self.backtest_ is None:
            blocos.append(_b("Backtest", "na", "não rodado",
                             "a cobertura dos intervalos é o que a validação pergunta"))
        else:
            cov = self.coverage_
            det = f"{int(self.backtest_.get('n_windows', 0))} janela(s)"
            if cov is not None and len(cov):
                tot = cov[cov["passo"] == "todos"]
                if len(tot):
                    det += (f" · cobertura {float(tot['cobertura'].iloc[0]):.0%} de "
                            f"{float(tot['nominal'].iloc[0]):.0%}")
            blocos.append(_b("Backtest", "ok", "rodado", det))
        return blocos

    def _render_export_estado(self):
        """Repinta o placar do estado (chamado ao abrir a aba e após cada ação)."""
        if getattr(self, "out_exp_estado", None) is None:
            return
        self.out_exp_estado.value = self._placar_html(self._estado_blocos())

    # ------------------------------------------------------------------ arquivos
    @staticmethod
    def _existe(caminho: str) -> bool:
        import os

        return bool(caminho) and os.path.exists(caminho)

    def _grava_com_confirmacao(self, btn, status, caminho, acao):
        """Executa ``acao`` — pedindo **confirmação em dois cliques** quando o arquivo
        já existe (:meth:`_confirm_twice`, o padrão de sobrescrita da casa)."""
        if not self._existe(caminho):
            acao()
            return
        if not getattr(btn, "_cc_armed", 0.0):
            status.value = (
                f"<div class='satui-notice'>O arquivo <code>{caminho}</code> <b>já "
                "existe</b>. Clique de novo (<i>Confirmar?</i>) para sobrescrever.</div>")
        self._confirm_twice(btn, acao)

    # ------------------------------------------------------------------ relatório
    def _on_report(self, b):
        if self.fit_ is None:
            self.out_report_status.value = (
                "<div class='satui-notice'>Nenhum <b>modelo vigente</b>: ajuste na aba "
                "<b>Especificação</b> ou adote uma especificação na aba <b>Seleção</b> "
                "antes de gerar o relatório.</div>")
            return
        caminho = str(self.tx_report_path.value or "").strip()
        if not caminho:
            self.out_report_status.value = (
                "<div class='satui-notice'>Informe o <b>arquivo</b> de destino "
                "(por exemplo <code>relatorio_estudo.html</code>).</div>")
            return
        self._grava_com_confirmacao(self.btn_report, self.out_report_status, caminho,
                                    lambda: self._salva_report(caminho))

    def _salva_report(self, caminho):
        import os

        with self._busy(self.btn_report, status=self.out_report_status,
                        msg="gerando o relatório…"):
            try:
                html = None
                # o estudo completo já produziu o HTML deste mesmo ajuste: reaproveita
                if (self.study_ is not None and self.study_.report_html
                        and self.study_.fit is self.fit_):
                    html = self.study_.report_html
                if html is None:
                    html = _E.report.model_report(
                        self.fit_, self.series, self.projection_,
                        title=f"Estudo: {self.tx_nome.value or self.study_name}")
                _E.report.save_report(html, caminho)
            except Exception as exc:  # noqa: BLE001
                self.out_report_status.value = (
                    f"<div class='satui-notice'>Não foi possível gerar o relatório: "
                    f"{exc}</div>")
                self._log(f"[relatório] ERRO: {type(exc).__name__}: {exc}")
                return
        self.report_path_ = caminho
        kb = f"{os.path.getsize(caminho) / 1024.0:,.0f}".replace(",", ".")
        com_proj = "com" if self.projection_ is not None else "sem"
        self.out_report_status.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ Relatório gravado em "
            f"<code>{caminho}</code> ({kb} KB) — {com_proj} o gráfico da projeção.</div>")
        self._log(f"[relatório] gravado em {caminho} ({kb} KB).")
        self._render_export_estado()

    # ------------------------------------------------------------------ MLflow
    def _on_mlflow(self, b):
        if self.fit_ is None:
            self.out_mlflow_status.value = (
                "<div class='satui-notice'>Nenhum <b>modelo vigente</b>: ajuste (ou adote "
                "uma especificação) antes de registrar o run.</div>")
            return
        with self._busy(self.btn_mlflow, status=self.out_mlflow_status,
                        msg="registrando no MLflow…"):
            try:
                import mlflow  # noqa: F401
            except ImportError:
                self.out_mlflow_status.value = (
                    "<div class='satui-notice'><b>MLflow não está instalado</b> neste "
                    "ambiente (<code>pip install mlflow</code>). O relatório HTML e o JSON "
                    "da configuração acima não dependem dele.</div>")
                self._log("[mlflow] pacote ausente — registro não realizado.")
                return
            params = {"criterio": self.dd_criterion.value,
                      "horizonte": int(self.sl_horizon.value),
                      "vif_max": float(self.fl_vif_max.value),
                      "candidatas": ",".join(self.candidates()) or "(nenhuma)"}
            tags = {"parametro": self._param_label(),
                    "segmento": (self.series.segment if self.series is not None else "") or "—",
                    "estudo": str(self.tx_nome.value or self.study_name)}
            try:
                rid = _E.tracking.log_satellite_run(
                    self.fit_, series=self.series, projection=self.projection_,
                    search=self.search_, backtest=self.backtest_, params=params, tags=tags,
                    experiment=(str(self.tx_mlflow_exp.value).strip() or None),
                    run_name=(str(self.tx_mlflow_run.value).strip()
                              or str(self.tx_nome.value or self.study_name)))
            except Exception as exc:  # noqa: BLE001
                self.out_mlflow_status.value = (
                    f"<div class='satui-notice'>Não foi possível registrar "
                    f"({type(exc).__name__}): {str(exc)[:300]}<br>Confira se o "
                    "<b>tracking</b> está configurado (variável <code>MLFLOW_TRACKING_URI</code>"
                    ", ou rode dentro do Databricks) e se você tem permissão de escrita no "
                    "experimento informado.</div>")
                self._log(f"[mlflow] ERRO: {type(exc).__name__}: {exc}")
                return
        self.mlflow_run_id_ = rid
        levou = [("ajuste e diagnóstico", True),
                 ("ranking da busca", self.search_ is not None),
                 ("projeção por cenário", self.projection_ is not None),
                 ("cobertura do backtest", self.backtest_ is not None)]
        itens = "".join(
            f"<li>{'✓' if ok else '—'} {rot}{'' if ok else ' (não havia no estado)'}</li>"
            for rot, ok in levou)
        self.out_mlflow_status.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ Run registrado — "
            f"<code>run_id = {rid}</code> (também em <code>ui.mlflow_run_id_</code>).</div>")
        self.out_mlflow_info.value = (
            f"<div class='satui-help'><div class='ttl'>O que foi para o run</div>"
            f"<ul>{itens}</ul>O artefato <code>report.html</code> traz tudo em abas.</div>")
        self._log(f"[mlflow] run_id = {rid}")
        self._render_export_estado()

    # ------------------------------------------------------------------ StudyConfig
    def _config_json(self) -> str:
        """O JSON (indentado) da :class:`StudyConfig` corrente."""
        import json

        return json.dumps(self.to_config().to_dict(), indent=2, ensure_ascii=False)

    def _on_cfg_show(self, b):
        try:
            texto = self._config_json()
        except ValueError as exc:
            self.out_cfg_status.value = f"<div class='satui-notice'>{exc}</div>"
            return
        self.ta_config_json.value = texto
        self.out_cfg_status.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ Configuração corrente "
            "serializada — o mesmo objeto está em <code>ui.to_config()</code>.</div>")

    def _on_cfg_save(self, b):
        caminho = str(self.tx_cfg_path.value or "").strip()
        if not caminho:
            self.out_cfg_status.value = (
                "<div class='satui-notice'>Informe o <b>arquivo</b> de destino "
                "(por exemplo <code>estudo.json</code>).</div>")
            return
        self._grava_com_confirmacao(self.btn_cfg_save, self.out_cfg_status, caminho,
                                    lambda: self._salva_config(caminho))

    def _salva_config(self, caminho):
        with self._busy(self.btn_cfg_save, status=self.out_cfg_status,
                        msg="gravando a configuração…"):
            try:
                texto = self._config_json()
                with open(caminho, "w", encoding="utf-8") as fh:
                    fh.write(texto)
            except Exception as exc:  # noqa: BLE001
                self.out_cfg_status.value = (
                    f"<div class='satui-notice'>Não foi possível gravar: {exc}</div>")
                self._log(f"[config] ERRO ao gravar: {type(exc).__name__}: {exc}")
                return
        self.ta_config_json.value = texto
        self.out_cfg_status.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ Configuração gravada em "
            f"<code>{caminho}</code> — versione este arquivo junto com a projeção.</div>")
        self._log(f"[config] configuração gravada em {caminho}.")

    def _on_cfg_load(self, b):
        import json

        caminho = str(self.tx_cfg_path.value or "").strip()
        if not caminho:
            self.out_cfg_status.value = (
                "<div class='satui-notice'>Informe o <b>arquivo</b> a carregar.</div>")
            return
        with self._busy(self.btn_cfg_load, status=self.out_cfg_status,
                        msg="lendo a configuração…"):
            try:
                with open(caminho, "r", encoding="utf-8") as fh:
                    dados = json.load(fh)
                self.ta_config_json.value = json.dumps(dados, indent=2, ensure_ascii=False)
                self.from_config(dados)
            except FileNotFoundError:
                self.out_cfg_status.value = (
                    f"<div class='satui-notice'>Arquivo não encontrado: "
                    f"<code>{caminho}</code>.</div>")
                return
            except Exception as exc:  # noqa: BLE001
                self.out_cfg_status.value = (
                    f"<div class='satui-notice'>Não foi possível carregar: {exc}</div>")
                self._log(f"[config] ERRO ao carregar: {type(exc).__name__}: {exc}")
                return
        self.out_cfg_status.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ Configuração de "
            f"<code>{caminho}</code> aplicada à interface — <b>reajuste</b> (ou rode o "
            "estudo completo) para reproduzir o resultado.</div>")

    def _on_cfg_apply(self, b):
        import json

        texto = str(self.ta_config_json.value or "").strip()
        if not texto:
            self.out_cfg_status.value = (
                "<div class='satui-notice'>A caixa está vazia — clique em <i>Ver JSON da "
                "sessão</i> ou cole a configuração antes.</div>")
            return
        try:
            dados = json.loads(texto)
            if not isinstance(dados, dict):
                raise TypeError("o JSON precisa ser um objeto com os campos da StudyConfig.")
            self.from_config(dados)
        except Exception as exc:  # noqa: BLE001
            self.out_cfg_status.value = (
                f"<div class='satui-notice'>JSON inválido: {exc}</div>")
            return
        self.out_cfg_status.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ JSON aplicado à "
            "interface — <b>reajuste</b> para que o modelo vigente corresponda a ele.</div>")

    # ------------------------------------------------------------------ tabelas
    def _tabela_export(self, chave):
        """``(DataFrame, sufixo do arquivo)`` da tabela escolhida.

        Levanta :class:`RuntimeError` com a mensagem que a tela deve mostrar quando
        a etapa correspondente ainda não rodou.
        """
        if chave == "projecao":
            return self.projection_frame(), "projecao"
        if chave == "ranking":
            if self.search_ is None:
                raise RuntimeError("nada a exportar: rode a busca na aba Seleção antes.")
            rk = self.search_.ranking.copy()
            if len(rk) and "status" in rk.columns:
                situ, motivo = zip(*[self._motivo_descarte(s) for s in rk["status"]])
                rk["situacao"], rk["motivo"] = list(situ), list(motivo)
            return rk, "ranking"
        if chave == "diagnostico":
            if self.diagnostics_ is None:
                raise RuntimeError("nada a exportar: rode a bateria na aba Diagnóstico antes.")
            return self.diagnostics_.copy(), "diagnostico"
        if chave == "cobertura":
            if self.coverage_ is None or not len(self.coverage_):
                raise RuntimeError("nada a exportar: rode o backtest na aba Backtest antes.")
            return self.coverage_.copy(), "cobertura"
        if chave == "coeficientes":
            if self.fit_ is None:
                raise RuntimeError("nada a exportar: não há modelo vigente ajustado.")
            return (self.fit_.coef_frame().reset_index().rename(columns={"index": "termo"}),
                    "coeficientes")
        if chave == "estacionariedade":
            if self.stationarity_ is None:
                raise RuntimeError("nada a exportar: rode o relatório de estacionariedade na "
                                   "aba Série antes.")
            return self.stationarity_.copy(), "estacionariedade"
        raise RuntimeError(f"tabela desconhecida: {chave!r}.")

    def _on_exp_tabela_change(self, change):
        """Acompanha o nome sugerido de arquivo com a tabela escolhida."""
        with suppress(Exception):
            self.tx_exp_path.value = f"{self._nome_arquivo()}_{change['new']}.csv"

    def _exp_sep(self):
        """``(separador, decimal)`` do formato escolhido na aba Exportar."""
        return {"tsv": ("\t", "."), "csv": (",", "."),
                "csv_br": (";", ",")}[self.dd_exp_fmt.value]

    def _on_exp_mostrar(self, b):
        try:
            df, _sufixo = self._tabela_export(self.dd_exp_tabela.value)
        except RuntimeError as exc:
            self.out_exp_tab_status.value = f"<div class='satui-notice'>{exc}</div>"
            self.ta_export_tabela.value = ""
            return
        sep, dec = self._exp_sep()
        self.ta_export_tabela.value = df.to_csv(sep=sep, index=False, decimal=dec)
        self.out_exp_tab_status.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ "
            f"{len(df)} linha(s) × {df.shape[1]} coluna(s) prontas — selecione tudo (Ctrl+A) "
            "e copie (Ctrl+C).</div>")

    def _on_exp_salvar(self, b):
        caminho = str(self.tx_exp_path.value or "").strip()
        if not caminho:
            self.out_exp_tab_status.value = (
                "<div class='satui-notice'>Informe o <b>arquivo</b> de destino.</div>")
            return
        try:
            df, _sufixo = self._tabela_export(self.dd_exp_tabela.value)
        except RuntimeError as exc:
            self.out_exp_tab_status.value = f"<div class='satui-notice'>{exc}</div>"
            return
        self._grava_com_confirmacao(self.btn_exp_salvar, self.out_exp_tab_status, caminho,
                                    lambda: self._salva_tabela(df, caminho))

    def _salva_tabela(self, df, caminho):
        sep, dec = self._exp_sep()
        with self._busy(self.btn_exp_salvar, status=self.out_exp_tab_status,
                        msg="gravando a tabela…"):
            try:
                df.to_csv(caminho, sep=sep, index=False, decimal=dec, encoding="utf-8-sig")
            except Exception as exc:  # noqa: BLE001
                self.out_exp_tab_status.value = (
                    f"<div class='satui-notice'>Não foi possível gravar: {exc}</div>")
                self._log(f"[exportar] ERRO ao gravar {caminho}: {type(exc).__name__}: {exc}")
                return
        self.out_exp_tab_status.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ "
            f"{len(df)} linha(s) gravadas em <code>{caminho}</code>.</div>")
        self._log(f"[exportar] {len(df)} linha(s) gravadas em {caminho}.")

    # ------------------------------------------------------------------ invalidação
    def _clear_exportar_outputs(self):
        """Zera a aba Exportar (dados novos ⇒ relatório e run antigos sem sentido)."""
        for w in ("out_report_status", "out_mlflow_status", "out_mlflow_info",
                  "out_cfg_status", "out_exp_tab_status"):
            widget = getattr(self, w, None)
            if widget is not None:
                widget.value = ""
        for w in ("ta_config_json", "ta_export_tabela"):
            widget = getattr(self, w, None)
            if widget is not None:
                widget.value = ""
        self._study_steps = []
        self._study_secs = None
        for w in ("out_study_status", "out_study_progress", "out_study_timer",
                  "out_study_resumo"):
            widget = getattr(self, w, None)
            if widget is not None:
                widget.value = ""
        self._invalidate_exportar("os dados mudaram")

    def _invalidate_exportar(self, motivo="o modelo vigente mudou"):
        """Invalida o que a aba Exportar produziu para o modelo **anterior**.

        O relatório e o run já gravados continuam no disco/MLflow — o que cai aqui
        é a garantia de que eles descrevem o modelo vigente.
        """
        tinha = (self.study_ is not None or self.report_path_ is not None
                 or self.mlflow_run_id_ is not None)
        self.study_ = None
        self.report_path_ = self.mlflow_run_id_ = None
        if getattr(self, "out_exp_notice", None) is None:   # ainda em construção
            return
        for w in ("out_report_status", "out_mlflow_status", "out_mlflow_info",
                  "out_exp_tab_status"):
            getattr(self, w).value = ""
        self.ta_export_tabela.value = ""
        self.out_exp_notice.value = (
            f"<div class='satui-notice'>⚠️ <b>Saídas desatualizadas</b> — {motivo}. O "
            "relatório e o registro anteriores descrevem o modelo <b>ANTERIOR</b>: gere-os "
            "de novo depois de reajustar.</div>") if tinha else ""
        self._render_export_estado()

    # ==================================================================
    # Configuração declarativa (StudyConfig)
    # ==================================================================
    def _stress_var(self) -> str:
        """Variável de estresse dos cenários padrão (widget da aba de cenários,
        se existir; senão o último valor restaurado; senão a 1ª candidata)."""
        dd = getattr(self, "dd_stress_var", None)
        if dd is not None and dd.value:
            return str(dd.value)
        cols = self._macro_cols()
        if self._stress_var_default and (not cols or self._stress_var_default in cols):
            return str(self._stress_var_default)
        cands = self.candidates() or cols
        return str(cands[0]) if cands else "desemprego"

    def _scenario_probs(self) -> tuple:
        """Pesos dos cenários (idem: widget da aba de cenários tem precedência)."""
        tx = getattr(self, "tx_scen_probs", None)
        if tx is not None and str(tx.value).strip():
            partes = [p for p in re.split(r"[,;\s]+", str(tx.value).strip()) if p]
            with suppress(ValueError):
                return tuple(float(p) for p in partes)
        return tuple(float(p) for p in self._scenario_probs_default)

    def to_config(self):
        """A configuração corrente como :class:`StudyConfig` (versionável).

        Modelos que não são candidatos da busca (ARIMA e os ingênuos) são
        exportados como ``ardl`` — eles entram no estudo como **benchmarks**, não
        como o modelo a selecionar; a interface avisa no console.
        """
        StudyConfig = _E.config.StudyConfig
        key = self.dd_model.value
        registry = MODELOS[key]["registry"]
        if registry is None:
            registry = "ardl"
            self._log(f"[config] '{key}' entra no estudo como benchmark; a configuração "
                      "exportada usa 'ardl' como modelo candidato.")
        usa_vasicek = registry == "vasicek"
        return StudyConfig(
            kind=self.kind,
            model=registry,
            link=self.dd_link.value,
            candidates=list(self.candidates()),
            expected_signs=dict(self.expected_signs()),
            lag_set=tuple(self._parse_ints(self.tx_lag_set.value, "defasagens")),
            max_vars=int(self.sl_max_vars.value),
            ar_orders=tuple(self._parse_ints(self.tx_ar_orders.value, "ordens AR")),
            seasonal=bool(self.cb_seasonal.value),
            seasonal_period=int(self.sl_seasonal_period.value),
            horizon=int(self.sl_horizon.value),
            min_train=(int(self.sl_min_train.value) or None),
            vif_max=float(self.fl_vif_max.value),
            criterion=self.dd_criterion.value,
            max_specs=int(self.sl_max_specs.value),
            rho=(float(self.fl_rho.value) if usa_vasicek else None),
            pd_ttc=(None if (not usa_vasicek or self.cb_ttc_auto.value)
                    else float(self.fl_pd_ttc.value)),
            stress_var=self._stress_var(),
            scenario_probabilities=self._scenario_probs(),
            name=self.tx_nome.value or "estudo",
        )

    def from_config(self, config):
        """Restaura os controles a partir de um :class:`StudyConfig` (ou ``dict``)."""
        StudyConfig = _E.config.StudyConfig
        cfg = StudyConfig.from_dict(config) if isinstance(config, Mapping) else config
        if self.series is not None and cfg.kind and cfg.kind != self.kind:
            self._log(f"[config] a configuração é de kind='{cfg.kind}' e a série carregada é "
                      f"'{self.kind}' — mantendo o kind da série.")
        elif cfg.kind:
            self.kind = str(cfg.kind)
        # modelo (mantém o corrente quando o do config não está na lista)
        if cfg.model in MODELOS:
            self.dd_model.value = cfg.model
        with suppress(Exception):
            self.dd_link.value = cfg.link
        # candidatas e sinais
        cands = set(cfg.candidates or [])
        signs = dict(cfg.expected_signs or {})
        for var, cb in getattr(self, "_sign_cbs", {}).items():
            cb.value = var in cands if cands else cb.value
            self._sign_tgs[var].value = int(signs.get(var, 0) or 0)
        self.tx_lag_set.value = ",".join(str(int(i)) for i in cfg.lag_set)
        self.tx_ar_orders.value = ",".join(str(int(i)) for i in cfg.ar_orders)
        self.sl_max_vars.value = int(cfg.max_vars)
        self.cb_seasonal.value = bool(cfg.seasonal)
        self.sl_seasonal_period.value = int(cfg.seasonal_period)
        self.sl_horizon.value = int(cfg.horizon)
        self.sl_min_train.value = int(cfg.min_train or 0)
        self.fl_vif_max.value = float(cfg.vif_max)
        with suppress(Exception):
            self.dd_criterion.value = cfg.criterion
        self.sl_max_specs.value = int(cfg.max_specs)
        if cfg.rho is not None:
            self.fl_rho.value = float(cfg.rho)
        self.cb_ttc_auto.value = cfg.pd_ttc is None
        if cfg.pd_ttc is not None:
            self.fl_pd_ttc.value = float(cfg.pd_ttc)
        self._stress_var_default = cfg.stress_var
        dd = getattr(self, "dd_stress_var", None)
        if dd is not None and cfg.stress_var in list(dd.options or []):
            dd.value = cfg.stress_var
        self._scenario_probs_default = tuple(float(p) for p in cfg.scenario_probabilities)
        tx = getattr(self, "tx_scen_probs", None)
        if tx is not None:
            tx.value = ",".join(str(p) for p in self._scenario_probs_default)
        self.tx_nome.value = cfg.name or "estudo"
        self.study_name = self.tx_nome.value
        self._render_model_help()
        self._sync_model_fields()
        self._refresh_bar()
        self._log(f"[config] configuração '{cfg.name}' restaurada na interface.")
        return self

    # ==================================================================
    # Ações da barra superior e refresh
    # ==================================================================
    def _on_ref_study(self, b):
        if self.series is not None:
            self._confirm_twice(self.btn_ref_study, self._load_reference_study)
            return
        self._load_reference_study()

    def _load_reference_study(self):
        """Gera o estudo de referência sintético e o carrega na interface."""
        with self._busy(self.btn_ref_study, status=self.out_ref_status,
                        msg="gerando o estudo de referência…"):
            try:
                kind = self.dd_ref_kind.value
                est = _E.series.make_reference_study(n_periods=int(self.sl_ref_n.value), seed=7)
                syn = getattr(est, kind)
                self.set_data(syn.series, est.macro, kind=kind)
            except Exception as exc:  # noqa: BLE001
                self.out_ref_status.value = (
                    f"<div class='satui-notice'>Não foi possível gerar o estudo: {exc}</div>")
                self._log(f"[dados] erro ao gerar o estudo de referência: {exc}")
                return
        self.out_ref_status.value = (
            "<div class='satui-legend' style='color:var(--ok-ink)'>✓ Estudo de referência "
            f"carregado — {len(self.series)} períodos, "
            f"{len(self._macro_cols())} variáveis macro (com recessão e evento).</div>")
        self._log(f"[dados] estudo de referência carregado (parâmetro {self._param_label()}, "
                  f"{len(self.series)} períodos).")

    def _on_dark(self, change):
        if change["new"]:
            self.panel.add_class("dark")
            self.cb_dark.description = "☀ Tema claro"
        else:
            self.panel.remove_class("dark")
            self.cb_dark.description = "🌙 Tema escuro"

    def _desliga_keepalive(self, msg):
        """Volta o botão para "desligado" **sem** reentrar no observer, e explica."""
        self._suspend_ka = True
        self.cb_keepalive.value = False
        self._suspend_ka = False
        self.cb_keepalive.description = "☕ Manter cluster ativo"
        self._log(f"[keepalive] {msg}")

    def _on_keepalive(self, change):
        if change["new"]:
            try:
                from ...utils.keepalive import ClusterKeepAlive

                if self._keepalive is None:
                    self._keepalive = ClusterKeepAlive(interval_seconds=120)
                if not self._keepalive.has_spark():
                    self._desliga_keepalive("nenhuma SparkSession ativa — recurso só "
                                            "funciona no Databricks (ou com Spark local).")
                    return
                self._keepalive.start()
            except Exception as exc:  # noqa: BLE001 - conveniência, nunca fatal
                self._desliga_keepalive(f"não foi possível ligar ({type(exc).__name__}): "
                                        f"{exc}")
                return
            self.cb_keepalive.description = "☕ Cluster ativo ✓"
            self._log("[keepalive] ligado — job Spark mínimo a cada 2 min mantém o cluster "
                      "ativo durante buscas longas. Desligue ao terminar.")
        else:
            if self._suspend_ka:
                return
            if self._keepalive is not None:
                with suppress(Exception):
                    self._keepalive.stop()
            self.cb_keepalive.description = "☕ Manter cluster ativo"
            self._log("[keepalive] desligado.")

    def _refresh_bar(self):
        rotulo = self._param_label()
        seg = self.series.segment if self.series is not None else ""
        self.banner.value = (
            "<div class='satui-banner'><div class='logo'>ST</div><div>"
            f"<div class='t'>Modelos satélite — {rotulo}</div>"
            "<div class='s'>Fatores prospectivos: série do parâmetro → macro → "
            "especificação → seleção → projeção condicional a cenário"
            f"{' · segmento ' + seg if seg else ''}</div></div></div>")
        pills = [self._pill(f"parâmetro: {rotulo}", "muted")]
        if self.series is None:
            pills.append(self._pill("sem série carregada", "yellow"))
        else:
            v = self.series.values
            pills.append(self._pill(f"observações: {len(v)}", "muted"))
            pills.append(self._pill(f"{str(v.index[0])[:7]} → {str(v.index[-1])[:7]}", "muted"))
        pills.append(self._pill(f"macros: {len(self._macro_cols())}", "muted"))
        pills.append(self._pill(f"candidatas: {len(self.candidates())}", "green"))
        if self.fit_ is None:
            pills.append(self._pill("ajuste: nenhum", "yellow"))
        elif self._dirty_since_fit:
            pills.append(self._pill("ajuste desatualizado — reajustar", "yellow"))
        else:
            pills.append(self._pill(f"ajuste: {self.fit_.model_name}", "green"))
        if self.search_ is not None:
            rk = self.search_.ranking
            n_qual = int((rk["status"] == "qualificado").sum()) if len(rk) else 0
            pills.append(self._pill(f"busca: {n_qual} qualificada(s)",
                                    "green" if n_qual else "yellow"))
        if self.diag_blocks_:
            ruins = [b for b in self.diag_blocks_ if b["nivel"] == "bad"]
            pills.append(self._pill(
                f"diagnóstico: {len(ruins)} bloco(s) reprovado(s)" if ruins
                else "diagnóstico: sem reprovações", "red" if ruins else "green"))
        if self.scenarios_ is not None and len(self.scenarios_):
            pills.append(self._pill(f"cenários: {len(self.scenarios_)}", "muted"))
        if self.projection_ is not None:
            pills.append(self._pill(f"projeção: {self.projection_.horizon} períodos", "green"))
        if self.coverage_ is not None and len(self.coverage_):
            tot = self.coverage_[self.coverage_["passo"] == "todos"]
            if len(tot):
                cob = float(tot["cobertura"].iloc[0])
                nom = float(tot["nominal"].iloc[0])
                ok = self._ok3(tot["ok"].iloc[0])
                pills.append(self._pill(
                    f"cobertura: {cob:.0%} de {nom:.0%}",
                    "green" if ok is True else ("yellow" if ok is None else "red")))
        if self.study_ is not None:
            pills.append(self._pill(f"estudo completo: {self.study_.config.name}", "green"))
        self.bar.value = "<div class='satui-bar'>" + "".join(pills) + "</div>"

    # ------------------------------------------------------------------ display
    def _ipython_display_(self):
        _display(self.panel)

    def display(self):
        _display(self.panel)

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self.series) if self.series is not None else 0
        return (f"SatelliteUI(kind={self.kind!r}, n={n}, "
                f"macros={len(self._macro_cols())}, "
                f"ajuste={'sim' if self.fit_ is not None else 'não'})")


__all__ = ["SatelliteUI", "MODELOS"]
