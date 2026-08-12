"""
yggdrasil.credit_risk.model.selection
=====================================
**Esteira de seleção de variáveis** do :class:`~yggdrasil.credit_risk.model.ModelSegmenter`.

Junta num fluxo único — com ordem explícita, parâmetros serializáveis e trilha de
auditoria — o que hoje existe em peças soltas no segmentador: filtro duro de
faltantes/constantes, tratamento de **categóricas** (cardinalidade, categorias
raras, faltantes), IV, PSI, monotonia, redundância (correlação/VIF) e *backward
elimination*. O usuário escolhe **quais** etapas rodar e em **qual ordem**; a
esteira devolve, para cada variável candidata, **onde** ela saiu e **por quê** —
em texto apresentável, pronto para o relatório::

    from yggdrasil.credit_risk.model import run_selection

    res = run_selection(seg, steps=["missing", "constante", "categoricas",
                                    "iv", "psi", "correlacao"], apply=True)
    res.tabela        # uma linha por candidata: decisão + motivo por extenso
    res.funil         # quantas entraram / saíram em cada etapa
    res.politica      # parâmetros efetivos (JSON) — reprodutibilidade
    res.historico     # registro cru por (variável, etapa)

``apply=False`` roda a esteira em **simulação**: nada no segmentador é alterado
(útil para testar réguas antes de aplicar). Com ``apply=True`` a decisão é
gravada no segmentador via ``include``/``exclude`` + ``set_category`` e o campo
``motivo`` do ``var_meta`` (o mesmo que a UI já exibe no ranking).

Nada aqui é específico de um parâmetro de risco: o alvo é sempre nomeado pelo
``problem_label`` do segmentador.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Vocabulário das decisões
# ----------------------------------------------------------------------
#: Decisão final de uma variável na :func:`run_selection`.
SELECIONADA = "selecionada"
EXCLUIDA = "excluida"
REVISAR = "revisar"
#: Decisão de ETAPA para quem passou por ela (não aparece em ``.tabela``).
PASSOU = "passou"

DECISOES = (SELECIONADA, EXCLUIDA, REVISAR)

# Piso de IV abaixo do qual a variável é considerada sem poder discriminante —
# topo da faixa "inútil" de :func:`yggdrasil.credit_risk._common.classifica_iv`
# (escala binária de Siddiqi na classificação; escala contínua na regressão).
_PISO_IV = {"classification": 0.02, "regression": 0.01}

_METODO_PT = {"spearman": "Spearman", "cramers_v": "V de Cramér"}


# ======================================================================
# Helpers de formatação / infraestrutura
# ======================================================================
def _pct(x, nd: int = 1) -> str:
    """Fração (0–1) como percentual legível: ``0.6`` → ``"60%"``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    s = f"{100.0 * v:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return f"{s}%"


def _num(x, nd: int = 4) -> str:
    """Número legível com casas fixas; ``"—"`` quando ausente/não finito."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return f"{v:.{nd}f}" if np.isfinite(v) else "—"


def _finito(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _emit(cb, key: str, label: str, status: str, detail: str = "") -> None:
    """Evento de progresso para a UI, se houver callback.

    Mesmo contrato de ``_emit_progress``/``progress_callback`` do segmentador:
    ``cb(key, label, status, detail)`` com ``status`` ∈ ``"run"`` (etapa
    iniciada), ``"ok"`` (concluída) ou ``"err"``. Nunca derruba a esteira — o
    progresso é cosmético."""
    if cb is None:
        return
    try:
        cb(key, label, status, detail)
    except Exception:  # noqa: BLE001 - progresso é cosmético
        pass


def _json_safe(obj):
    """Converte tipos numpy/pandas para primitivos JSON (NaN/NaT/NA → ``None``)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if obj is None or obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, str):
        return obj
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def _dec(variavel, decisao, motivo, destaque: bool = False, **detalhe) -> dict:
    """Decisão de uma etapa sobre uma variável (registro do histórico)."""
    d = {"variavel": variavel, "decisao": decisao, "motivo": motivo,
         "destaque": bool(destaque)}
    det = {k: v for k, v in detalhe.items() if v is not None}
    if det:
        d["detalhe"] = det
    return d


# ======================================================================
# Resultado
# ======================================================================
#: Colunas de :attr:`SelectionResult.tabela`, nesta ordem.
COLUNAS_TABELA = ("variavel", "rotulo", "tipo", "decisao", "etapa_saida", "motivo",
                  "iv", "forca", "pior_psi", "estabilidade", "tendencia",
                  "n_inversoes", "missing_pct", "n_categorias")
#: Colunas de :attr:`SelectionResult.funil`, nesta ordem.
COLUNAS_FUNIL = ("etapa", "n_entrada", "n_excluidas", "n_revisar", "n_saida")


@dataclass
class SelectionResult:
    """Resultado da esteira de seleção — a **trilha de auditoria** completa.

    Attributes
    ----------
    tabela:
        Uma linha por variável **candidata**, na ordem de entrada. Colunas em
        :data:`COLUNAS_TABELA`: identificação (``variavel``, ``rotulo``,
        ``tipo``), a decisão (``decisao`` ∈ ``selecionada``/``excluida``/
        ``revisar``, ``etapa_saida`` = etapa que excluiu — vazio para quem
        sobreviveu, ``motivo`` por extenso) e as métricas conhecidas (``iv``,
        ``forca``, ``pior_psi``, ``estabilidade``, ``tendencia``,
        ``n_inversoes``, ``missing_pct`` em **0–100**, ``n_categorias``).
        Métricas de quem caiu ANTES da etapa que as calcula ficam vazias — a
        esteira não gasta binning com variável já descartada.
    funil:
        Uma linha por etapa executada (colunas em :data:`COLUNAS_FUNIL`),
        começando pela linha ``"candidatas"``. Fecha aritmeticamente:
        ``n_saida = n_entrada − n_excluidas`` e o ``n_saida`` de uma etapa é o
        ``n_entrada`` da seguinte. ``n_revisar`` é sinalização (não reduz o
        conjunto).
    selecionadas, excluidas, revisar:
        Listas de nomes por decisão (sobreviventes = ``selecionadas`` +
        ``revisar``).
    politica:
        Etapas executadas e **todos** os parâmetros efetivos (inclusive os
        defaults, já resolvidos) — serializável em JSON.
    historico:
        Registro cru por ``(variável, etapa)``: ``variavel``, ``etapa``,
        ``decisao`` (``passou``/``revisar``/``excluida``), ``motivo`` e, quando
        houver, ``detalhe`` (números que embasaram a decisão).
    """

    tabela: pd.DataFrame
    funil: pd.DataFrame
    selecionadas: list
    excluidas: list
    revisar: list
    politica: dict
    historico: list = field(default_factory=list)

    # ---- serialização ------------------------------------------------
    def to_dict(self) -> dict:
        """Dicionário serializável em JSON (NaN/NA viram ``None``)."""
        return _json_safe({
            "versao": 1,
            "politica": self.politica,
            "tabela": self.tabela.to_dict("records"),
            "funil": self.funil.to_dict("records"),
            "selecionadas": list(self.selecionadas),
            "excluidas": list(self.excluidas),
            "revisar": list(self.revisar),
            "historico": list(self.historico),
        })

    @classmethod
    def from_dict(cls, data: dict) -> "SelectionResult":
        """Reconstrói o resultado a partir de :meth:`to_dict` (round-trip JSON)."""
        tabela = _tabela_frame(data.get("tabela") or [])
        funil = pd.DataFrame(data.get("funil") or [], columns=list(COLUNAS_FUNIL))
        return cls(tabela=tabela, funil=funil,
                   selecionadas=list(data.get("selecionadas") or []),
                   excluidas=list(data.get("excluidas") or []),
                   revisar=list(data.get("revisar") or []),
                   politica=dict(data.get("politica") or {}),
                   historico=list(data.get("historico") or []))

    # ---- leitura -----------------------------------------------------
    def resumo(self) -> str:
        """Resumo em texto (uma linha por etapa do funil)."""
        etapas = list(self.politica.get("etapas") or [])
        cab = (f"{len(self.tabela)} candidatas → {len(self.selecionadas)} selecionadas · "
               f"{len(self.excluidas)} excluídas · {len(self.revisar)} a revisar")
        linhas = [cab, "etapas: " + (" → ".join(rotulo_etapa(e) for e in etapas)
                                     if etapas else "(nenhuma)")]
        for _, r in self.funil.iterrows():
            if r["etapa"] == "candidatas":
                continue
            # Sem f-string aninhada aqui: o sufixo sai numa variável porque aspas
            # repetidas dentro de uma f-string só valem do Python 3.12 (PEP 701)
            # e a lib declara requires-python >=3.9.
            n_revisar = int(r["n_revisar"])
            sufixo = f", {n_revisar} a revisar" if n_revisar else ""
            linhas.append(f"  {rotulo_etapa(r['etapa'])}: {int(r['n_entrada'])} → "
                          f"{int(r['n_saida'])} (−{int(r['n_excluidas'])}{sufixo})")
        if not self.politica.get("aplicado", True):
            linhas.append("  (simulação — nada foi aplicado no segmentador)")
        return "\n".join(linhas)

    def __repr__(self) -> str:  # pragma: no cover - representação
        return f"<SelectionResult · {self.resumo()}>"


def _tabela_frame(rows) -> pd.DataFrame:
    """DataFrame da tabela com as colunas e dtypes canônicos."""
    out = pd.DataFrame(list(rows), columns=list(COLUNAS_TABELA))
    for c in ("n_inversoes", "n_categorias"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    for c in ("iv", "pior_psi", "missing_pct"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ======================================================================
# Contexto compartilhado pelas etapas
# ======================================================================
@dataclass
class _Ctx:
    """Estado que as etapas leem/escrevem durante uma execução."""

    seg: object
    params: dict
    ref: pd.DataFrame                      # recorte da amostra de referência
    tipos: dict                            # variável → "num" | "cat"
    extras: dict                           # variável → missing_pct / n_categorias
    metricas: dict = field(default_factory=dict)   # variável → IV/PSI/tendência
    avisos: list = field(default_factory=list)
    progress: object = None
    _rank_ver: int = -1
    _splits_originais: dict = field(default_factory=dict)

    # ---- métricas caras (IV/PSI/tendência), memoizadas ----------------
    def metricas_de(self, feats) -> dict:
        """IV, força, tendência, inversões e PSI das variáveis dadas.

        Delega ao ``variable_iv`` do segmentador (que já memoiza o binning por
        variável) e guarda o resultado por execução — o cache é descartado
        quando o segmentador re-versiona o ranking (ex.: após o agrupamento de
        categorias raras da etapa ``"categoricas"``)."""
        seg = self.seg
        ver = int(getattr(seg, "_rank_version", 0))
        if ver != self._rank_ver:
            self.metricas.clear()
            self._rank_ver = ver
        faltando = [f for f in feats if f not in self.metricas]
        if faltando:
            rk = seg.variable_iv(features=faltando, sample=self.params["sample"],
                                 max_n_bins=self.params["max_n_bins"],
                                 min_bin_size=self.params["min_bin_size"])
            psi_cols = [c for c in rk.columns if c.startswith("psi_")]
            for _, r in rk.iterrows():
                m = {"iv": float(r["iv"]) if _finito(r["iv"]) else np.nan,
                     "forca": str(r.get("forca", "—")),
                     "tendencia": str(r.get("tendencia", "—")),
                     "n_inversoes": int(r.get("n_inversoes", 0) or 0),
                     "n_bins": int(r.get("n_bins", 0) or 0),
                     "pior_psi": np.nan, "estabilidade": "—", "psi_amostra": None}
                if "pior_psi" in rk.columns and _finito(r["pior_psi"]):
                    m["pior_psi"] = float(r["pior_psi"])
                    m["estabilidade"] = str(r.get("estabilidade", "—"))
                    piores = [(float(r[c]), c[4:]) for c in psi_cols if _finito(r[c])]
                    if piores:
                        m["psi_amostra"] = max(piores)[1]
                self.metricas[r["variavel"]] = m
        return {f: self.metricas[f] for f in feats if f in self.metricas}

    # ---- agrupamento de categorias (reusa os bins manuais) ------------
    def aplicar_grupos(self, feature, grupos) -> None:
        """Materializa um agrupamento de categorias na binagem da variável.

        Reutiliza o mecanismo de **bins manuais** do segmentador
        (``set_manual_bins``), que já sobrepõe o binning ótimo em toda a análise
        univariada (tabela, IV, WoE, PSI, inversão) e é o que a UI exibe. Em
        simulação (``apply=False``) o estado anterior é restaurado ao fim da
        esteira por :meth:`restaurar_grupos`."""
        if feature not in self._splits_originais:
            self._splits_originais[feature] = self.seg.manual_bins(feature)
        self.seg.set_manual_bins(feature, grupos)

    def restaurar_grupos(self) -> None:
        """Desfaz os agrupamentos aplicados (usado na simulação)."""
        for feature, antigos in self._splits_originais.items():
            self.seg.set_manual_bins(feature, antigos)
        self._splits_originais.clear()


# ======================================================================
# Etapas
# ======================================================================
def _etapa_missing(ctx: _Ctx, feats) -> list:
    """Filtro duro: percentual de faltantes na amostra de referência."""
    lim = float(ctx.params["max_missing"])
    out = []
    for f in feats:
        pct = ctx.extras[f]["missing_frac"]
        if not _finito(pct):
            continue
        if pct > lim:
            out.append(_dec(f, EXCLUIDA,
                            f"faltantes em {_pct(pct)} da amostra de referência "
                            f"(máximo {_pct(lim)})", missing_pct=100 * pct))
        else:
            out.append(_dec(f, PASSOU, f"faltantes em {_pct(pct)}",
                            missing_pct=100 * pct))
    return out


def _etapa_constante(ctx: _Ctx, feats) -> list:
    """Filtro duro: valor único, variância ~nula ou categoria dominante."""
    dom = float(ctx.params["max_dominancia"])
    min_desvio = float(ctx.params["min_desvio"])
    out = []
    for f in feats:
        obs = ctx.ref[f].dropna()
        if obs.empty:
            out.append(_dec(f, EXCLUIDA,
                            "sem valores observados na amostra de referência"))
            continue
        try:
            vc = obs.value_counts(normalize=True)
        except TypeError:                      # valores não-hasheáveis
            vc = obs.astype(str).value_counts(normalize=True)
        n_unicos = int(len(vc))
        top, top_p = vc.index[0], float(vc.iloc[0])
        if n_unicos <= 1:
            out.append(_dec(f, EXCLUIDA,
                            f"valor único ('{top}') em toda a amostra de referência",
                            n_unicos=n_unicos))
            continue
        if ctx.tipos[f] == "num":
            x = pd.to_numeric(obs, errors="coerce").to_numpy(dtype="float64")
            x = x[~np.isnan(x)]
            desvio = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
            if desvio <= min_desvio:
                out.append(_dec(f, EXCLUIDA,
                                f"variância praticamente nula (desvio-padrão "
                                f"{desvio:.2e})", desvio=desvio, n_unicos=n_unicos))
                continue
        if top_p >= dom:
            rot = "categoria" if ctx.tipos[f] == "cat" else "valor"
            out.append(_dec(f, EXCLUIDA,
                            f"{rot} dominante '{top}' concentra {_pct(top_p)} da "
                            f"amostra (máximo {_pct(dom)})",
                            dominancia=top_p, n_unicos=n_unicos))
            continue
        out.append(_dec(f, PASSOU,
                        f"{n_unicos} valores distintos · mais frequente '{top}' com "
                        f"{_pct(top_p)}", dominancia=top_p, n_unicos=n_unicos))
    return out


def _etapa_categoricas(ctx: _Ctx, feats) -> list:
    """Tratamento nativo das **categóricas**: cardinalidade, categorias raras e
    faltantes.

    * cardinalidade acima de ``max_categorias`` → exclui (é chave, não preditora);
    * categorias com frequência abaixo de ``min_freq_categoria`` são **agrupadas
      de fato** (bins manuais do segmentador) numa faixa "OUTROS" — o IV passa a
      ser medido DEPOIS do agrupamento, sem inflar por categorias minúsculas;
    * faltantes viram **faixa própria** ``(faltante)`` no binning (comportamento
      do ``_resolve_bins`` do segmentador para categóricas com ``NaN``) — o que
      é registrado no motivo.

    Variáveis numéricas não são avaliadas aqui."""
    seg = ctx.seg
    max_cat = int(ctx.params["max_categorias"])
    min_freq = float(ctx.params["min_freq_categoria"])
    agrupar = bool(ctx.params["agrupar_raras"])
    out = []
    for f in feats:
        if ctx.tipos[f] != "cat":
            continue
        obs = ctx.ref[f].dropna().astype(str)
        vc = obs.value_counts(normalize=True)
        n_cat = int(len(vc))
        ctx.extras[f]["n_categorias"] = n_cat
        if n_cat == 0:
            out.append(_dec(f, EXCLUIDA, "sem categorias observadas na amostra de "
                                         "referência", n_categorias=0))
            continue
        if n_cat > max_cat:
            out.append(_dec(f, EXCLUIDA,
                            f"alta cardinalidade: {n_cat} categorias (máximo "
                            f"{max_cat}) — considere agrupar ou usar como chave, "
                            f"não como preditora", n_categorias=n_cat))
            continue
        raras = [c for c, p in vc.items() if p < min_freq]
        frequentes = [c for c, p in vc.items() if p >= min_freq]
        if not frequentes:
            out.append(_dec(f, EXCLUIDA,
                            f"todas as {n_cat} categorias abaixo de "
                            f"{_pct(min_freq)} — nenhuma faixa com massa "
                            f"suficiente", n_categorias=n_cat))
            continue
        notas = []
        if seg.manual_bins(f):
            notas.append("agrupamento manual já definido — preservado")
        elif raras and agrupar:
            grupos = [[c] for c in frequentes] + [list(raras)]
            ctx.aplicar_grupos(f, grupos)
            notas.append(f"{len(raras)} categoria(s) rara(s) (< {_pct(min_freq)}) "
                         f"agrupada(s) em OUTROS")
        elif raras:
            notas.append(f"{len(raras)} categoria(s) rara(s) (< {_pct(min_freq)}) "
                         f"— agrupamento desligado")
        miss = ctx.extras[f]["missing_frac"]
        if _finito(miss) and miss >= min_freq:
            notas.append(f"faltantes ({_pct(miss)}) formam a faixa própria "
                         f"'(faltante)'")
        motivo = f"{n_cat} categorias" + ("; " + "; ".join(notas) if notas else "")
        out.append(_dec(f, PASSOU, motivo, destaque=bool(notas), n_categorias=n_cat,
                        n_raras=len(raras)))
    return out


def _etapa_iv(ctx: _Ctx, feats) -> list:
    """Poder discriminante: IV abaixo do mínimo exclui; IV altíssimo sinaliza."""
    met = ctx.metricas_de(feats)
    min_iv = float(ctx.params["min_iv"])
    out = []
    for f in feats:
        m = met.get(f)
        if m is None:
            continue
        iv, forca = m["iv"], m["forca"]
        if not _finito(iv):
            out.append(_dec(f, EXCLUIDA,
                            "IV não calculado (binning sem faixas válidas) — sem "
                            "poder discriminante mensurável"))
        elif iv < min_iv:
            out.append(_dec(f, EXCLUIDA,
                            f"IV {_num(iv)} abaixo do mínimo {min_iv:g} — sem poder "
                            f"discriminante", iv=iv))
        elif forca == "suspeito":
            out.append(_dec(f, REVISAR,
                            f"IV {_num(iv)} alto demais (força '{forca}') — possível "
                            f"vazamento do alvo; confirme a origem da variável",
                            iv=iv))
        else:
            out.append(_dec(f, PASSOU, f"IV {_num(iv)} (força {forca})",
                            destaque=True, iv=iv))
    return out


def _etapa_psi(ctx: _Ctx, feats) -> list:
    """Estabilidade entre amostras: pior PSI da variável (bins fixados na
    referência)."""
    seg = ctx.seg
    if getattr(seg, "sample_col", None) is None or not seg._nonref_samples():
        ctx.avisos.append("etapa 'psi': não há amostra de comparação além da "
                          "referência — nenhuma variável foi avaliada.")
        return []
    met = ctx.metricas_de(feats)
    max_psi = float(ctx.params["max_psi"])
    psi_warn = float(ctx.params["psi_warn"])
    out = []
    for f in feats:
        m = met.get(f)
        if m is None:
            continue
        psi, amostra = m["pior_psi"], m["psi_amostra"]
        onde = f" na amostra {amostra}" if amostra else ""
        if not _finito(psi):
            out.append(_dec(f, PASSOU, "PSI não calculado (sem faixas comparáveis)"))
        elif psi > max_psi:
            out.append(_dec(f, EXCLUIDA,
                            f"PSI {_num(psi)} acima do máximo {max_psi:g}{onde} — "
                            f"distribuição instável", pior_psi=psi, amostra=amostra))
        elif psi >= psi_warn:
            out.append(_dec(f, REVISAR,
                            f"PSI {_num(psi)} em atenção (entre {psi_warn:g} e "
                            f"{max_psi:g}){onde}", pior_psi=psi, amostra=amostra))
        else:
            out.append(_dec(f, PASSOU, f"PSI {_num(psi)} ({m['estabilidade']}){onde}",
                            pior_psi=psi, amostra=amostra))
    return out


def _etapa_monotonia(ctx: _Ctx, feats) -> list:
    """Monotonia da ordem de risco das faixas — **só para numéricas**.

    Categóricas **nominais** não têm ordem natural entre as categorias: cobrar
    monotonia delas é erro conceitual (a ordem das faixas é arbitrária). Elas
    passam como ``isenta (categórica nominal)``."""
    met = ctx.metricas_de([f for f in feats if ctx.tipos[f] == "num"])
    exclui = bool(ctx.params["monotonia_exclui"])
    out = []
    for f in feats:
        if ctx.tipos[f] != "num":
            out.append(_dec(f, PASSOU, "isenta (categórica nominal) — monotonia não "
                                       "se aplica"))
            continue
        m = met.get(f)
        if m is None:
            continue
        trend, n_inv = m["tendencia"], m["n_inversoes"]
        if trend in ("—", ""):
            out.append(_dec(f, PASSOU, "tendência não avaliada (sem faixas)"))
        elif trend == "não-monotônica" or n_inv > 0:
            motivo = (f"tendência {trend} com {n_inv} inversão(ões) na ordem de risco "
                      f"das faixas — avalie reagrupar")
            out.append(_dec(f, EXCLUIDA if exclui else REVISAR, motivo,
                            tendencia=trend, n_inversoes=n_inv))
        else:
            out.append(_dec(f, PASSOU, f"tendência {trend}, sem inversões",
                            tendencia=trend, n_inversoes=n_inv))
    return out


def _etapa_correlacao(ctx: _Ctx, feats) -> list:
    """Redundância entre pares (Spearman nas numéricas · V de Cramér nas
    categóricas): em cada par acima do limiar, sai a de **menor IV**."""
    seg = ctx.seg
    feats = list(feats)
    if len(feats) < 2:
        ctx.avisos.append("etapa 'correlacao': menos de 2 variáveis sobreviventes — "
                          "nada a comparar.")
        return []
    thr = float(ctx.params["max_corr"])
    rep = seg.correlation_report(threshold=thr, features=feats,
                                 sample=ctx.params["sample"])
    poda = set(rep.attrs.get("poda_sugerida", []))
    # replica a poda GULOSA do relatório (pares já ordenados por associação
    # decrescente) para saber QUAL par causou cada remoção — e documentar o motivo
    causa, removidas, parceiro = {}, set(), {}
    for r in rep.itertuples(index=False):
        f1, f2 = r.variavel_1, r.variavel_2
        parceiro.setdefault(f1, (f2, r.metodo, r.associacao))
        parceiro.setdefault(f2, (f1, r.metodo, r.associacao))
        if f1 in removidas or f2 in removidas:
            continue
        removidas.add(r.remover)
        iv_m = r.iv_1 if r.manter == f1 else r.iv_2
        iv_r = r.iv_2 if r.manter == f1 else r.iv_1
        causa[r.remover] = (r.manter, r.metodo, float(r.associacao),
                            float(iv_m), float(iv_r))
    out = []
    for f in feats:
        if f in poda:
            manter, metodo, assoc, iv_m, iv_r = causa.get(
                f, (None, "spearman", float("nan"), float("nan"), float("nan")))
            lab = seg.label(manter) if manter else "outra variável"
            met_pt = _METODO_PT.get(metodo, metodo)
            comp = (f"; {lab} tem IV maior ({_num(iv_m)} vs {_num(iv_r)})"
                    if _finito(iv_m) and _finito(iv_r) and iv_m > iv_r
                    else f"; mantida {lab} (IV {_num(iv_m)} vs {_num(iv_r)})")
            out.append(_dec(f, EXCLUIDA,
                            f"redundante com {lab} ({met_pt} {assoc:.2f}){comp}",
                            associacao=assoc, mantida=manter))
        elif f in parceiro:
            par, metodo, assoc = parceiro[f]
            out.append(_dec(f, PASSOU,
                            f"associação {_METODO_PT.get(metodo, metodo)} "
                            f"{float(assoc):.2f} com {seg.label(par)} — mantida",
                            associacao=float(assoc)))
        else:
            out.append(_dec(f, PASSOU, f"sem par redundante acima de {thr:g}"))
    return out


def _etapa_vif(ctx: _Ctx, feats) -> list:
    """Multicolinearidade pelo **VIF** da matriz de desenho do modelo vigente.

    O VIF é por **termo** do desenho (uma categórica vira várias *dummies*): os
    termos são reagregados na variável de origem pelo **pior** (maior) VIF antes
    de decidir. Variáveis que não estão no modelo vigente não são avaliadas."""
    seg = ctx.seg
    tab = seg.vif_table(use_labels=False)
    warn = float(ctx.params["vif_warn"])
    lim = float(ctx.params["max_vif"])
    exclui = bool(ctx.params["vif_exclui"])
    agg: dict = {}
    for termo, vif in zip(tab["termo"], tab["vif"]):
        orig = seg._original_feature_of(str(termo))
        try:
            v = float(vif)
        except (TypeError, ValueError):
            continue
        if not np.isnan(v):
            agg[orig] = max(agg.get(orig, float("-inf")), v)
    out = []
    for f in feats:
        v = agg.get(f)
        if v is None:
            out.append(_dec(f, PASSOU, "VIF não disponível (variável fora do modelo "
                                       "vigente)"))
        elif v > lim:
            txt = "∞" if not np.isfinite(v) else f"{v:.2f}"
            out.append(_dec(f, EXCLUIDA if exclui else REVISAR,
                            f"VIF {txt} acima do máximo {lim:g} — multicolinearidade "
                            f"alta (pior termo da variável no desenho)",
                            vif=v if np.isfinite(v) else None))
        elif v >= warn:
            out.append(_dec(f, REVISAR,
                            f"VIF {v:.2f} em atenção (entre {warn:g} e {lim:g})",
                            vif=v))
        else:
            out.append(_dec(f, PASSOU, f"VIF {v:.2f} (ok)", vif=v))
    return out


def _etapa_backward(ctx: _Ctx, feats) -> list:
    """*Backward elimination* por importância, aplicando o passo escolhido.

    Roda ``backward_elimination`` nas sobreviventes e aplica o passo indicado por
    ``backward_criterion`` (``parsimony``/``best`` via ``backward_optimal_step``,
    ou ``manual`` com ``backward_n_variaveis`` via ``backward_subset_at``). Não
    toca no modelo vigente — o segmentador treina modelos temporários."""
    seg = ctx.seg
    feats = list(feats)
    if len(feats) < 2:
        ctx.avisos.append("etapa 'backward': requer ao menos 2 variáveis "
                          "sobreviventes — etapa ignorada.")
        return []
    rot = rotulo_etapa("backward")

    def _prog(done, total, n_vars):
        _emit(ctx.progress, "backward", rot, "run",
              f"passo {done}/{total} · {n_vars} variáveis")

    res = seg.backward_elimination(
        sample=ctx.params["backward_sample"],
        min_features=int(ctx.params["backward_min_features"]),
        features=feats, progress_callback=_prog)
    n_manual = ctx.params["backward_n_variaveis"]
    if n_manual is not None:
        pick = seg.backward_subset_at(res, int(n_manual),
                                      metric=ctx.params["backward_metric"])
    else:
        pick = seg.backward_optimal_step(res, criterion=ctx.params["backward_criterion"],
                                         tol=float(ctx.params["backward_tol"]),
                                         metric=ctx.params["backward_metric"])
    removidas = set(pick.get("removed") or [])
    alvo, metrica = pick.get("target_n"), pick.get("metric") or "métrica"
    melhor = pick.get("best")
    val = f"{metrica} {_num(melhor)}" if _finito(melhor) else str(metrica)
    passo = f"passo com {alvo} variável(is); {val}"
    out = []
    for f in feats:
        if f in removidas:
            out.append(_dec(f, EXCLUIDA, f"removida no backward elimination ({passo})",
                            target_n=alvo, metrica=metrica,
                            valor=melhor if _finito(melhor) else None))
        else:
            out.append(_dec(f, PASSOU, f"mantida no backward elimination ({passo})",
                            destaque=True, target_n=alvo))
    return out


# ======================================================================
# Registro de etapas (plugável, executado na ORDEM pedida pelo usuário)
# ======================================================================
@dataclass(frozen=True)
class SelectionStep:
    """Uma etapa da esteira: nome (chave), rótulo em pt-BR e a função."""

    nome: str
    rotulo: str
    descricao: str
    funcao: object


SELECTION_STEPS: dict = {}


def register_step(nome: str, rotulo: str, descricao: str, funcao) -> SelectionStep:
    """Registra (ou substitui) uma etapa da esteira.

    ``funcao(ctx, feats)`` recebe o contexto e as **sobreviventes** da etapa
    anterior e devolve uma lista de decisões — use :func:`_dec` como molde:
    ``{"variavel", "decisao", "motivo", "destaque"?, "detalhe"?}`` com
    ``decisao`` ∈ ``"passou"``/``"revisar"``/``"excluida"``."""
    step = SelectionStep(nome=nome, rotulo=rotulo, descricao=descricao, funcao=funcao)
    SELECTION_STEPS[nome] = step
    return step


register_step("missing", "faltantes",
              "Exclui variáveis com percentual de faltantes acima de max_missing.",
              _etapa_missing)
register_step("constante", "constantes",
              "Exclui valor único, variância ~nula ou categoria dominante demais.",
              _etapa_constante)
register_step("categoricas", "categóricas",
              "Cardinalidade, agrupamento de categorias raras e faltantes como "
              "categoria.", _etapa_categoricas)
register_step("iv", "IV",
              "Poder discriminante: IV mínimo; IV altíssimo vira revisar "
              "(vazamento).", _etapa_iv)
register_step("psi", "PSI",
              "Estabilidade entre amostras pelo pior PSI da variável.", _etapa_psi)
register_step("monotonia", "monotonia",
              "Tendência da ordem de risco das faixas (só numéricas).",
              _etapa_monotonia)
register_step("correlacao", "correlação",
              "Redundância entre pares: sai a de menor IV.", _etapa_correlacao)
register_step("vif", "VIF",
              "Multicolinearidade pelo VIF do desenho do modelo vigente.",
              _etapa_vif)
register_step("backward", "backward elimination",
              "Backward elimination por importância, aplicando o passo escolhido.",
              _etapa_backward)

#: Sequência default (``steps=None``). Ordem pensada para custo e correção:
#: filtros duros primeiro (baratos, sem binning), depois o tratamento das
#: **categóricas** — que muda a binagem e portanto o IV —, então as métricas
#: univariadas (IV → PSI → monotonia) e, por fim, a redundância entre as que
#: sobraram. ``vif`` e ``backward`` ficam de fora do default: o primeiro exige um
#: modelo já ajustado e o segundo treina dezenas de modelos.
STEPS_DEFAULT = ("missing", "constante", "categoricas", "iv", "psi", "monotonia",
                 "correlacao")


def rotulo_etapa(nome: str) -> str:
    """Rótulo em pt-BR de uma etapa (``"candidatas"`` e nomes livres passam
    direto) — para relatórios e para a coluna ``etapa`` do funil."""
    step = SELECTION_STEPS.get(str(nome))
    return step.rotulo if step is not None else str(nome)


def etapas_disponiveis() -> list:
    """Nomes das etapas registradas, na ordem canônica de execução."""
    return list(SELECTION_STEPS.keys())


# ======================================================================
# Parâmetros
# ======================================================================
#: Parâmetros aceitos por :func:`run_selection` e seus defaults. Todos entram em
#: ``politica['parametros']`` já **resolvidos** (nada fica implícito).
PARAMS_DEFAULT: dict = {
    # escopo / binagem
    "features": None,             # candidatas a avaliar (default: seg.candidates)
    "sample": None,               # amostra de referência (default: seg.ref_sample)
    "max_n_bins": 5,
    "min_bin_size": 0.05,
    # filtros duros
    "max_missing": 0.60,          # fração (0–1) de faltantes tolerada
    "max_dominancia": 0.99,       # fração do valor/categoria mais frequente
    "min_desvio": 1e-12,          # desvio-padrão mínimo (numéricas)
    # categóricas
    "max_categorias": 30,
    "min_freq_categoria": 0.01,   # abaixo disso a categoria é "rara"
    "agrupar_raras": True,
    # métricas univariadas
    "min_iv": None,               # None → 0.02 (classificação) / 0.01 (regressão)
    "max_psi": 0.25,
    "psi_warn": 0.10,
    "monotonia_exclui": False,    # False → não-monotônica vira "revisar"
    # redundância
    "max_corr": 0.85,
    "vif_warn": 5.0,
    "max_vif": 10.0,              # regra de bolso do vif_table: > 10 é alto
    "vif_exclui": False,
    # backward
    "backward_sample": None,
    "backward_criterion": "parsimony",   # "parsimony" | "best" | "manual"
    "backward_tol": 0.01,
    "backward_metric": None,
    "backward_min_features": 1,
    "backward_n_variaveis": None,
    # aplicação
    "incluir_revisar": True,      # "revisar" sobreviveu → segue incluída
}


def _resolve_params(seg, params: dict) -> dict:
    """Valida os parâmetros informados e resolve os defaults (nada fica implícito)."""
    desconhecidos = sorted(set(params) - set(PARAMS_DEFAULT))
    if desconhecidos:
        raise ValueError(
            f"parâmetro(s) desconhecido(s): {', '.join(desconhecidos)}. "
            f"Aceitos: {', '.join(sorted(PARAMS_DEFAULT))}.")
    eff = dict(PARAMS_DEFAULT)
    eff.update(params)
    if eff["min_iv"] is None:
        eff["min_iv"] = _PISO_IV.get(getattr(seg, "task_type", "classification"), 0.02)
    if eff["sample"] is None:
        eff["sample"] = getattr(seg, "ref_sample", None)
    if eff["backward_criterion"] not in ("parsimony", "best", "manual"):
        raise ValueError("backward_criterion deve ser 'parsimony', 'best' ou "
                         f"'manual' (recebido: {eff['backward_criterion']!r}).")
    if eff["backward_criterion"] == "manual" and eff["backward_n_variaveis"] is None:
        raise ValueError("backward_criterion='manual' exige backward_n_variaveis.")
    return eff


def _valida_steps(steps) -> list:
    """Normaliza a lista de etapas — erro claro em etapa desconhecida/repetida."""
    if steps is None:
        return list(STEPS_DEFAULT)
    nomes = [str(s) for s in steps]
    disponiveis = ", ".join(etapas_disponiveis())
    for s in nomes:
        if s not in SELECTION_STEPS:
            raise ValueError(f"etapa desconhecida: {s!r}. Etapas disponíveis: "
                             f"{disponiveis}.")
    repetidas = sorted({s for s in nomes if nomes.count(s) > 1})
    if repetidas:
        raise ValueError(f"etapa(s) repetida(s) em steps: {', '.join(repetidas)}. "
                         f"Cada etapa roda uma única vez.")
    return nomes


def _avisos_de_ordem(steps) -> list:
    """Avisos sobre ordens que produzem números enganosos."""
    avisos = []
    if "categoricas" in steps:
        pos = steps.index("categoricas")
        antes = [e for e in ("iv", "psi", "monotonia")
                 if e in steps and steps.index(e) < pos]
        if antes:
            avisos.append(
                "a etapa 'categoricas' está DEPOIS de "
                f"{', '.join(repr(e) for e in antes)}: o IV (e o PSI/monotonia) "
                "seria medido ANTES do agrupamento das categorias raras e viria "
                "inflado por categorias minúsculas. Ponha 'categoricas' antes de "
                "'iv' na lista de steps.")
    return avisos


# ======================================================================
# Esteira
# ======================================================================
def run_selection(seg, steps=None, apply=True, progress_callback=None,
                  **params) -> SelectionResult:
    """Roda a **esteira de seleção de variáveis** do segmentador e devolve a
    trilha de auditoria.

    Cada etapa recebe as **sobreviventes** da anterior e devolve, por variável,
    uma decisão (``passou``/``revisar``/``excluida``) com o motivo por extenso.
    A ordem é a da lista ``steps``.

    Parameters
    ----------
    seg:
        :class:`~yggdrasil.credit_risk.model.ModelSegmenter` já construído. A
        esteira trabalha pela API pública dele (``variable_iv``,
        ``correlation_report``, ``vif_table``, ``backward_elimination``,
        ``set_manual_bins``, ``include``/``exclude``/``set_category``).
    steps:
        Etapas a executar, na ordem desejada. ``None`` usa
        :data:`STEPS_DEFAULT`. Nomes disponíveis em :data:`SELECTION_STEPS`
        (``missing``, ``constante``, ``categoricas``, ``iv``, ``psi``,
        ``monotonia``, ``correlacao``, ``vif``, ``backward``); nome desconhecido
        levanta ``ValueError`` listando os válidos.
    apply:
        ``True`` (default) grava a decisão no segmentador: ``include``/
        ``exclude``, ``set_category`` (``manter``/``revisar``/``descartar``) e o
        campo ``motivo`` do ``var_meta`` — além de manter o agrupamento de
        categorias raras aplicado. ``False`` é **simulação**: o estado do
        segmentador (seleção, categorias, bins manuais) volta exatamente como
        estava.
    progress_callback:
        ``cb(key, label, status, detail)`` — mesmo contrato de progresso do
        segmentador (``status`` ∈ ``"run"``/``"ok"``/``"err"``).
    **params:
        Réguas da esteira; ver :data:`PARAMS_DEFAULT` (todas com default
        documentado e registradas em ``politica['parametros']``).

    Returns
    -------
    SelectionResult

    Raises
    ------
    ValueError
        Etapa desconhecida/repetida, parâmetro desconhecido, ou ``features`` com
        variável que não é candidata do segmentador.
    RuntimeError
        Etapa ``"vif"`` sem modelo ajustado no segmentador.
    """
    nomes = _valida_steps(steps)
    eff = _resolve_params(seg, params)

    # --- pré-checagens (falham ANTES de gastar binning) -----------------
    if "vif" in nomes and getattr(seg, "model", None) is None:
        raise RuntimeError(
            "a etapa 'vif' precisa de um modelo ajustado: o VIF é medido na "
            "matriz de desenho do modelo vigente. Rode seg.fit(...) (ou "
            "set_model) antes, ou retire 'vif' de steps.")
    cands = list(eff["features"]) if eff["features"] is not None else list(seg.candidates)
    fora = [f for f in cands if f not in seg.candidates]
    if fora:
        raise ValueError(f"variável(is) que não são candidatas do segmentador: "
                         f"{', '.join(map(str, fora))}.")
    cands = [f for f in cands if f in seg.df.columns]

    avisos = _avisos_de_ordem(nomes)
    for msg in avisos:
        warnings.warn(msg, UserWarning, stacklevel=2)

    ref = seg._frame(eff["sample"])
    tipos = {f: seg._detect_kind(f, ref) for f in cands}
    n_ref = max(len(ref), 1)
    extras = {}
    for f in cands:
        col = ref[f]
        n_cat = np.nan
        if tipos[f] == "cat":
            n_cat = int(col.dropna().astype(str).nunique())
        extras[f] = {"missing_frac": float(col.isna().sum()) / n_ref,
                     "n_categorias": n_cat}

    ctx = _Ctx(seg=seg, params=eff, ref=ref, tipos=tipos, extras=extras,
               avisos=avisos, progress=progress_callback)

    sobreviventes = list(cands)
    historico: list = []
    motivo_saida: dict = {}
    etapa_saida: dict = {}
    notas_revisar: dict = {}
    destaques: dict = {}
    funil = [{"etapa": "candidatas", "n_entrada": len(cands), "n_excluidas": 0,
              "n_revisar": 0, "n_saida": len(cands)}]

    try:
        for nome in nomes:
            step = SELECTION_STEPS[nome]
            n_entrada = len(sobreviventes)
            _emit(progress_callback, nome, step.rotulo, "run",
                  f"{n_entrada} variáveis")
            try:
                decisoes = step.funcao(ctx, list(sobreviventes)) or []
            except Exception as exc:  # noqa: BLE001 - a etapa nomeia o erro
                _emit(progress_callback, nome, step.rotulo, "err", str(exc))
                raise
            excluidas_etapa, revisar_etapa = [], []
            for d in decisoes:
                f, dec = d["variavel"], d["decisao"]
                reg = {"variavel": f, "etapa": nome, "decisao": dec,
                       "motivo": d["motivo"]}
                if d.get("detalhe"):
                    reg["detalhe"] = d["detalhe"]
                historico.append(reg)
                if dec == EXCLUIDA:
                    excluidas_etapa.append(f)
                    motivo_saida[f] = d["motivo"]
                    etapa_saida[f] = nome
                elif dec == REVISAR:
                    revisar_etapa.append(f)
                    notas_revisar.setdefault(f, []).append(d["motivo"])
                elif d.get("destaque"):
                    destaques.setdefault(f, []).append(d["motivo"])
            fora_etapa = set(excluidas_etapa)
            sobreviventes = [f for f in sobreviventes if f not in fora_etapa]
            funil.append({"etapa": nome, "n_entrada": n_entrada,
                          "n_excluidas": len(fora_etapa),
                          "n_revisar": len(set(revisar_etapa)),
                          "n_saida": len(sobreviventes)})
            _emit(progress_callback, nome, step.rotulo, "ok",
                  f"{len(fora_etapa)} excluída(s) · {len(set(revisar_etapa))} a "
                  f"revisar · {len(sobreviventes)} seguem")
    finally:
        if not apply:
            ctx.restaurar_grupos()

    # --- decisão final por variável -------------------------------------
    vivos = set(sobreviventes)
    linhas, selecionadas, excluidas, revisar = [], [], [], []
    for f in cands:
        met = ctx.metricas.get(f, {})
        if f not in vivos:
            decisao = EXCLUIDA
            motivo = motivo_saida.get(f, "excluída pela esteira")
            excluidas.append(f)
        elif f in notas_revisar:
            decisao = REVISAR
            motivo = "; ".join(notas_revisar[f])
            revisar.append(f)
        else:
            decisao = SELECIONADA
            motivo = ("; ".join(destaques.get(f, []))
                      or ("aprovada em todas as etapas" if nomes
                          else "nenhuma etapa executada"))
            selecionadas.append(f)
        n_cat = extras[f]["n_categorias"]
        linhas.append({
            "variavel": f, "rotulo": seg.label(f), "tipo": tipos[f],
            "decisao": decisao, "etapa_saida": etapa_saida.get(f, ""),
            "motivo": motivo,
            "iv": met.get("iv", np.nan), "forca": met.get("forca", "—"),
            "pior_psi": met.get("pior_psi", np.nan),
            "estabilidade": met.get("estabilidade", "—"),
            "tendencia": met.get("tendencia", "—"),
            "n_inversoes": met.get("n_inversoes", np.nan),
            "missing_pct": round(100.0 * extras[f]["missing_frac"], 2),
            "n_categorias": n_cat,
        })

    politica = {
        "versao": 1,
        "task_type": getattr(seg, "task_type", None),
        "alvo": getattr(seg, "target", None),
        "problema": getattr(seg, "problem_label", None) or getattr(seg, "target", None),
        "amostra_referencia": eff["sample"],
        "etapas": list(nomes),
        "aplicado": bool(apply),
        "parametros": _json_safe(eff),
        "avisos": list(ctx.avisos),
    }
    res = SelectionResult(tabela=_tabela_frame(linhas),
                          funil=pd.DataFrame(funil, columns=list(COLUNAS_FUNIL)),
                          selecionadas=selecionadas, excluidas=excluidas,
                          revisar=revisar, politica=politica, historico=historico)

    if apply:
        _aplicar(seg, res, incluir_revisar=bool(eff["incluir_revisar"]))
    _emit(progress_callback, "fim", "seleção concluída", "ok",
          f"{len(selecionadas)} selecionadas · {len(excluidas)} excluídas · "
          f"{len(revisar)} a revisar")
    return res


def _aplicar(seg, res: SelectionResult, incluir_revisar: bool = True) -> None:
    """Grava a decisão no segmentador: seleção, categoria e motivo.

    A categoria segue o vocabulário que a UI já exibe: ``manter`` (selecionada),
    ``revisar`` (sobreviveu com ressalva) e ``descartar`` (excluída). O
    ``motivo`` é gravado DEPOIS da categoria — ``set_category`` limpa o motivo
    anterior de propósito."""
    for r in res.tabela.itertuples(index=False):
        f, decisao = r.variavel, r.decisao
        if decisao == SELECIONADA:
            seg.include(f)
            categoria = "manter"
        elif decisao == REVISAR:
            (seg.include if incluir_revisar else seg.exclude)(f)
            categoria = "revisar"
        else:
            seg.exclude(f)
            categoria = "descartar"
        seg.set_category(f, categoria)
        seg.var_meta.setdefault(f, {})["motivo"] = r.motivo


__all__ = ["run_selection", "SelectionResult", "SelectionStep", "SELECTION_STEPS",
           "STEPS_DEFAULT", "PARAMS_DEFAULT", "COLUNAS_TABELA", "COLUNAS_FUNIL",
           "DECISOES", "SELECIONADA", "EXCLUIDA", "REVISAR", "PASSOU",
           "register_step", "rotulo_etapa", "etapas_disponiveis"]
