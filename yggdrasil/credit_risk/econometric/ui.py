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
* **Seleção**, **Diagnóstico**, **Cenários & Projeção**, **Backtest** e
  **Exportar** — as demais etapas do estudo.

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
    scenarios_, projection_:
        O último :class:`ScenarioSet` e a :class:`Projection` (aba Cenários).
    backtest_:
        A última tabela de backtest de cobertura/projeção (aba Backtest).
    study_:
        O último :class:`StudyResult` de um estudo completo (aba Exportar).
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

    #: nomes definitivos das abas (os próximos blocos só preenchem o conteúdo)
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
        self.scenarios_ = None      # ScenarioSet (Cenários & Projeção)
        self.projection_ = None     # Projection (Cenários & Projeção)
        self.backtest_ = None       # backtest de cobertura (Backtest)
        self.study_ = None          # StudyResult de um estudo completo (Exportar)
        self.stationarity_ = None   # tabela do último relatório de estacionariedade

        self._log_lines: list = []
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
        self.scenarios_ = self.projection_ = self.backtest_ = self.study_ = None
        self.stationarity_ = None
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
                 pct_cols=None, precision=4):
        """Tabela HTML no estilo da casa.

        ``color_map`` é ``{coluna: função(valor) -> css}`` — use os helpers
        ``_css_veredito``/``_css_ordem``/``_css_ok``/``_css_pvalor``/``_css_coerencia``
        (todos em tokens de tema, nunca hex).
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

        threading.Timer(timeout, _revert).start()

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
        self._refresh_bar()

    def _clear_dirty(self):
        self._dirty_since_fit = False
        if getattr(self, "out_fit_warn", None) is not None:
            self.out_fit_warn.value = ""

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
        # As abas seguintes nascem VAZIAS (placeholder) com os nomes definitivos —
        # os blocos seguintes só substituem ``.children`` do respectivo VBox.
        self.box_selecao = self._placeholder_box(
            "Seleção",
            "busca champion-challenger sobre a grade de especificações, com os filtros "
            "duros de sinal econômico e VIF, validação walk-forward e ranking contra os "
            "benchmarks.")
        self.box_diagnostico = self._placeholder_box(
            "Diagnóstico",
            "bateria de resíduos (autocorrelação, heterocedasticidade, normalidade, "
            "estabilidade) e os gráficos de ajuste do modelo escolhido.")
        self.box_cenarios = self._placeholder_box(
            "Cenários & Projeção",
            "cenários base/adverso/otimista (ou choques), projeção condicional em leque "
            "e a projeção ponderada pelos pesos de cenário.")
        self.box_backtest = self._placeholder_box(
            "Backtest",
            "backtest da projeção fora da amostra e cobertura dos intervalos "
            "(Kupiec/Christoffersen).")
        self.box_exportar = self._placeholder_box(
            "Exportar",
            "relatório HTML de governança, tabelas, a configuração do estudo e o registro "
            "no MLflow.")

        self.tabs = W.Tab(children=[tab_serie, tab_spec, self.box_selecao,
                                    self.box_diagnostico, self.box_cenarios,
                                    self.box_backtest, self.box_exportar])
        for i, t in enumerate(self.ABAS):
            self.tabs.set_title(i, t)
        self.tabs.add_class("satui-tabs")

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

    def _placeholder_box(self, titulo, texto):
        """VBox de uma aba ainda **em construção** (nome definitivo já no lugar)."""
        box = W.VBox([W.HTML(
            f"<div class='satui-card'><div class='satui-h'>{titulo}</div>"
            f"<div class='satui-legend'>🚧 <b>Em construção.</b> Aqui entram: {texto}</div>"
            "</div>")], layout=W.Layout(padding="2px"))
        return box

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
                                       layout=W.Layout(width="140px"))
        self.btn_serie_plot.on_click(lambda b: self._refresh_serie())
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

        # mudanças na especificação deixam o ajuste desatualizado
        for w in (self.dd_model, self.dd_link, self.dd_trend, self.dd_cov, self.fl_rho,
                  self.cb_ttc_auto, self.fl_pd_ttc, self.tx_arima_order,
                  self.tx_arima_seasonal, self.tx_lag_set, self.tx_ar_orders,
                  self.cb_seasonal, self.sl_seasonal_period):
            w.observe(self._mark_dirty, names="value")

        self._rebuild_signs()
        self._render_model_help()
        return W.VBox([card_modelo, card_sinais, card_grade, card_fit],
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

    def _on_keepalive(self, change):
        from ...utils.keepalive import ClusterKeepAlive

        if change["new"]:
            if self._keepalive is None:
                self._keepalive = ClusterKeepAlive(interval_seconds=120)
            if not self._keepalive.has_spark():
                self._suspend_ka = True
                self.cb_keepalive.value = False
                self._suspend_ka = False
                self.cb_keepalive.description = "☕ Manter cluster ativo"
                self._log("[keepalive] nenhuma SparkSession ativa — recurso só funciona "
                          "no Databricks (ou com Spark local).")
                return
            self._keepalive.start()
            self.cb_keepalive.description = "☕ Cluster ativo ✓"
            self._log("[keepalive] ligado — job Spark mínimo a cada 2 min mantém o cluster "
                      "ativo durante buscas longas. Desligue ao terminar.")
        else:
            if self._suspend_ka:
                return
            if self._keepalive is not None:
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
            pills.append(self._pill("busca: concluída", "green"))
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
