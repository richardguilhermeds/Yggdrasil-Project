"""
SequentialPDSegmenter
=====================
Construtor sequencial e híbrido de segmentações para modelos de PD
(Probability of Default) — alvo **binário** (0 = adimplente, 1 = default).

- Cresce a segmentação em camadas (`grow`), dividindo cada folha por uma nova
  variável usando OPTIMAL BINNING (OptBinning, alvo binário) ou CORTES MANUAIS.
- Preview do split antes de efetivar (`show_grow`): taxa de default (PD) +
  representatividade.
- Poda de folhas pouco representativas ou sem separação de PD (`prune`).
- PSI de estabilidade entre amostras (DES como referência) usando os próprios
  segmentos como bins (`psi`, `psi_detalhe`).
- Métricas de discriminação da régua como modelo de PD: **KS, ROC/AUC, Gini,
  Acurácia e F1** por amostra (`metrics`).

Contexto: parâmetros de risco de crédito sob Resolução CMN 4.966/2021 e IFRS 9.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from optbinning import OptimalBinning
except ImportError:  # pragma: no cover
    OptimalBinning = None


def _fit_optbinning_splits(b, x, y) -> list:
    """Roda ``b.fit(x, y)`` e devolve ``list(b.splits)``.

    Silencia os ``RuntimeWarning`` de "divide by zero" benignos do optbinning
    (em ``auto_monotonic``, quando algum prebin fica com 0 registros) — o ajuste
    ainda produz cortes válidos. Devolve ``[]`` se o ajuste falhar.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(divide="ignore", invalid="ignore"):
                b.fit(x, y)
        return list(b.splits)
    except Exception:
        return []


# ======================================================================
# Helpers
# ======================================================================
def _fmt(x: float) -> str:
    """Formata limites de faixa de forma legível."""
    if x == -np.inf:
        return "-inf"
    if x == np.inf:
        return "inf"
    return f"{x:.4g}"


def _classifica_psi(psi: float) -> str:
    """Classificação usual de PSI para monitoramento de estabilidade."""
    if psi < 0.10:
        return "estável"
    if psi < 0.25:
        return "atenção"
    return "instável"


def _intervalo_por_extenso(lo: float, hi: float) -> str:
    """Converte um intervalo (lo, hi] em texto legível."""
    if lo == -np.inf:
        return f"até {_fmt(hi)}"
    if hi == np.inf:
        return f"acima de {_fmt(lo)}"
    return f"entre {_fmt(lo)} e {_fmt(hi)}"


def _classifica_iv(iv) -> str:
    """Faixas de força do IV **binário** (WoE clássico, escala de Siddiqi):
    < 0.02 inútil · 0.02–0.10 fraco · 0.10–0.30 médio · 0.30–0.50 forte ·
    ≥ 0.50 suspeito (alto demais — verifique vazamento de informação)."""
    if iv is None or (isinstance(iv, float) and np.isnan(iv)):
        return "—"
    if iv < 0.02:
        return "inútil"
    if iv < 0.10:
        return "fraco"
    if iv < 0.30:
        return "médio"
    if iv < 0.50:
        return "forte"
    return "suspeito"


def _match_conditions_pandas(df, conditions):
    """Máscara das linhas de `df` que satisfazem todas as condições do caminho.

    Condições numéricas/categóricas podem trazer ``include_na=True``: nesse caso
    a faixa também captura os faltantes (bin populado **OU** faltante).
    """
    m = pd.Series(True, index=df.index)
    for c in conditions:
        feat = c["feature"]
        if c["kind"] == "na":
            m &= df[feat].isna()
        elif c["kind"] == "num":
            lo, hi = c.get("lo"), c.get("hi")
            sub = pd.Series(True, index=df.index)
            if lo is not None:
                sub &= df[feat] > lo
            if hi is not None:
                sub &= df[feat] <= hi
            if c.get("include_na"):
                sub |= df[feat].isna()
            m &= sub
        else:
            cats = [str(x) for x in c["cats"]]
            sub = df[feat].astype(str).isin(cats)
            if c.get("include_na"):
                sub |= df[feat].isna()
            m &= sub
    return m


def _aplicar_regua_pandas(regua, df, col_seg="segmento_pd",
                          col_nota="nota_pd", col_pd="pd_regua"):
    """Aplica uma régua (dict de folhas) a um DataFrame pandas."""
    seg = pd.Series(pd.NA, index=df.index, dtype="object")
    nota = pd.Series(pd.NA, index=df.index, dtype="Int64")
    pdcol = pd.Series(np.nan, index=df.index, dtype="float64")
    for leaf in regua["leaves"]:
        m = _match_conditions_pandas(df, leaf["conditions"])
        seg[m] = leaf["id"]
        nota[m] = leaf["nota"]
        pdcol[m] = leaf["pd"]
    return pd.DataFrame({col_seg: seg, col_nota: nota, col_pd: pdcol}, index=df.index)


# ======================================================================
# Classe principal
# ======================================================================
class SequentialPDSegmenter:
    """Segmentação sequencial de PD com binning ótimo/manual, poda e PSI."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str = "target",
        sample_col: str | None = None,
        ref_sample: str = "DES",
        feature_labels: dict[str, str] | None = None,
        min_leaf_rows: int = 50,
        date_col: str | None = None,
        verbose: bool = True,
    ):
        if OptimalBinning is None:
            raise ImportError("optbinning não instalado. Rode: pip install optbinning")

        self.df = df.copy()
        self.target = target
        self.sample_col = sample_col
        self.ref_sample = ref_sample
        # coluna de DATA/safra: NÃO entra na modelagem (fica fora das features);
        # serve só para os gráficos no tempo (PD/variável por safra, PSI por safra).
        self.date_col = date_col
        if date_col is not None and date_col not in self.df.columns:
            raise ValueError(f"Coluna de data '{date_col}' não está no DataFrame "
                             f"(colunas: {list(self.df.columns)}).")
        # rótulos amigáveis por variável para a descrição por extenso
        self.feature_labels = feature_labels or {}
        # mínimo de linhas (na amostra de ajuste) para tentar binning ótimo
        self.min_leaf_rows = min_leaf_rows

        if sample_col is not None:
            if sample_col not in self.df.columns:
                raise ValueError(
                    f"Coluna de amostra '{sample_col}' não está no DataFrame "
                    f"(colunas: {list(self.df.columns)}).")
            amostras = self.df[sample_col].dropna().unique().tolist()
            if ref_sample not in amostras:
                raise ValueError(
                    f"Amostra de referência '{ref_sample}' não encontrada em "
                    f"'{sample_col}'. Disponíveis: {amostras}"
                )
            if verbose:
                print(f"[init] amostras: {amostras} | referência PSI = {ref_sample}")

        # cada segmento: id -> dict(mask, label, depth, is_leaf, path, parent, conditions)
        self.segments: dict[str, dict] = {
            "root": {
                "mask": pd.Series(True, index=df.index),
                "label": "root",
                "depth": 0,
                "is_leaf": True,
                "path": [],
                "parent": None,
                "conditions": [],   # lista de (feature, lo, hi) acumulada no caminho
            }
        }
        self.history: list[dict] = []
        self._psi_detalhe: list[dict] = []
        self._csi_detalhe: list[dict] = []

    # ------------------------------------------------------------------
    # Helpers de bin genéricos (numérico OU categórico)
    # ------------------------------------------------------------------
    def _detect_kind(self, sub, feature, dtype):
        """Decide se a variável é tratada como 'num' ou 'cat'."""
        if dtype in ("num", "cat"):
            return dtype
        col = sub[feature]
        if pd.api.types.is_bool_dtype(col):
            return "cat"
        return "num" if pd.api.types.is_numeric_dtype(col) else "cat"

    def _fit_frame(self, sub, min_bin_size):
        """Amostra usada para AJUSTAR o binning ótimo (só DES, se houver)."""
        if self.sample_col is None:
            return sub
        fit_data = sub[sub[self.sample_col] == self.ref_sample]
        if len(fit_data) < 2 * int(min_bin_size * len(self.df)):
            return sub  # ramo com pouco DES: usa o ramo inteiro
        return fit_data

    def _mask_in(self, frame, feature, b):
        """Máscara booleana das linhas de `frame` que caem no bin `b`."""
        if b["kind"] == "na":
            return frame[feature].isna()
        if b["kind"] == "num":
            return frame[feature].between(b["lo"], b["hi"], inclusive="right")
        return frame[feature].astype(str).isin(b["cats"])

    @staticmethod
    def _bin_label(feature, b):
        if b["kind"] == "na":
            return f"{feature}: (faltante)"
        if b["kind"] == "num":
            return f"{feature}: ({_fmt(b['lo'])}, {_fmt(b['hi'])}]"
        return f"{feature}: {{{', '.join(map(str, b['cats']))}}}"

    @staticmethod
    def _bin_condition(feature, b):
        if b["kind"] == "na":
            return {"feature": feature, "kind": "na"}
        if b["kind"] == "num":
            return {"feature": feature, "kind": "num", "lo": b["lo"], "hi": b["hi"]}
        return {"feature": feature, "kind": "cat", "cats": list(b["cats"])}

    def _target_binary(self, frame) -> np.ndarray:
        """Alvo binário (0/1) de `frame`, sem faltantes — para o binning ótimo."""
        y = frame[self.target].to_numpy(dtype="float64")
        y = y[~np.isnan(y)]
        return y.astype(int)

    def _resolve_bins(self, sub, feature, splits, dtype, max_n_bins, min_bin_size,
                      max_bin_size=None, relax_max=False):
        """Resolve os bins de um ramo. Devolve (bins, modo, kind).
        - num: bins = [{'kind':'num','lo','hi'}, ...]
        - cat: bins = [{'kind':'cat','cats':[...]}, ...]
        No modo ótimo, o binning binário é ajustado só na amostra de referência
        (DES). No modo manual: splits numérico = lista de cortes; splits
        categórico = lista de grupos (cada grupo é uma lista de categorias).
        """
        kind = self._detect_kind(sub, feature, dtype)

        if kind == "num":
            if splits is not None:
                lo, hi = sub[feature].min(), sub[feature].max()
                cortes = [s for s in sorted(splits) if lo < s < hi]
                modo = "manual"
            else:
                fit = self._fit_frame(sub, min_bin_size)
                x = fit[feature].to_numpy(dtype="float64")
                y = fit[self.target].to_numpy(dtype="float64")
                ok = ~np.isnan(y)
                x, y = x[ok], y[ok]                         # alvo NaN não entra no ajuste
                yb = y.astype(int)                          # alvo binário 0/1
                x_obs = x[~np.isnan(x)]
                if (len(yb) < 4 or x_obs.size == 0 or np.unique(x_obs).size < 2
                        or np.unique(yb).size < 2):
                    cortes = []                            # dados degenerados → sem corte
                else:
                    def _opt(mnb, mbs):
                        b = OptimalBinning(
                            name=feature, dtype="numerical", max_n_bins=mnb,
                            min_bin_size=min_bin_size, max_bin_size=mbs,
                            monotonic_trend="auto_asc_desc")
                        return _fit_optbinning_splits(b, x, yb)
                    cortes = _opt(max_n_bins, max_bin_size)
                    if not cortes and max_bin_size is not None and relax_max:
                        # o máximo pode deixar o problema INFEASIBLE p/ este nº de
                        # bins/amostra: tenta mais bins e, por fim, relaxa o máximo
                        for k in range(max_n_bins + 1, 9):
                            cortes = _opt(k, max_bin_size)
                            if cortes:
                                break
                        if not cortes:
                            cortes = _opt(max_n_bins, None)
                modo = "ótimo"
            if not cortes:
                return [], modo, kind
            edges = [-np.inf, *cortes, np.inf]
            bins = [{"kind": "num", "lo": edges[i], "hi": edges[i + 1]}
                    for i in range(len(edges) - 1)]
            if sub[feature].isna().any():
                bins.append({"kind": "na"})
            return bins, modo, kind

        # ---- categórico ----
        na_present = bool(sub[feature].isna().any())
        if splits is not None:
            grupos = [list(g) for g in splits]   # lista de grupos de categorias
            flat = [str(c) for g in grupos for c in g]
            if len(flat) != len(set(flat)):      # grupos devem ser disjuntos
                dup = sorted({c for c in flat if flat.count(c) > 1})
                raise ValueError(
                    f"Grupos categóricos manuais têm categoria(s) repetida(s): {dup}. "
                    "Cada categoria deve estar em um único grupo.")
            modo = "manual"
        else:
            fit = self._fit_frame(sub, min_bin_size)
            fit = fit[fit[feature].notna() & fit[self.target].notna()]  # NaN fora do ajuste
            xs = fit[feature].astype(str).to_numpy()
            ys = fit[self.target].to_numpy(dtype="float64").astype(int)
            if len(ys) < 4 or np.unique(xs).size < 2 or np.unique(ys).size < 2:
                grupos = []                          # dados degenerados → sem grupos
            else:
                b = OptimalBinning(
                    name=feature, dtype="categorical", max_n_bins=max_n_bins,
                    min_bin_size=min_bin_size, max_bin_size=max_bin_size,
                    monotonic_trend="auto_asc_desc")
                grupos = [list(arr) for arr in _fit_optbinning_splits(b, xs, ys)]
            modo = "ótimo"
        _NA_TOK = {"nan", "NaN", "<NA>", "None"}
        bins = []
        for g in grupos:
            cats = [str(c) for c in g if str(c) not in _NA_TOK]
            if cats:
                bins.append({"kind": "cat", "cats": cats})
        if bins and na_present:
            bins.append({"kind": "na"})
        return bins, modo, kind

    # ------------------------------------------------------------------
    # Tabela de bins: taxa de default (PD) + representatividade (num ou cat)
    # ------------------------------------------------------------------
    def _bin_table(self, sub, feature, bins, n_ref):
        linhas = []
        for b in bins:
            m = self._mask_in(sub, feature, b)
            s = sub.loc[m, self.target]
            if len(s) == 0:
                continue
            linhas.append({
                "faixa": self._bin_label(feature, b),
                "n": len(s),
                "repr_%": round(100 * len(s) / n_ref, 1),
                "pd_medio": round(s.mean(), 4),
                "pd_std": round(s.std(), 4),
            })
        tbl = pd.DataFrame(linhas)
        if tbl.empty:                       # todos os bins vazios
            tbl.attrs["mono_ok"] = True
            return tbl
        means = tbl["pd_medio"]
        tbl.attrs["mono_ok"] = bool(
            means.is_monotonic_increasing or means.is_monotonic_decreasing
        )
        return tbl

    # ------------------------------------------------------------------
    # SHOW_GROW: preview do split (não altera estado)
    #   dtype: None=auto-detecta, 'num' ou 'cat' para forçar
    # ------------------------------------------------------------------
    def show_grow(self, feature, splits=None, dtype=None, max_n_bins=4,
                  min_bin_size=0.05, only_segments=None, max_bin_size=None):
        targets = {
            sid: s for sid, s in self.segments.items()
            if s["is_leaf"] and (only_segments is None or sid in only_segments)
        }
        n_total = len(self.df)
        previews = {}
        for sid, seg in targets.items():
            sub = self.df[seg["mask"]]
            bins, modo, kind = self._resolve_bins(
                sub, feature, splits, dtype, max_n_bins, min_bin_size, max_bin_size)
            if not bins:
                print(f"[{sid}] sem corte válido em '{feature}' ({modo})")
                continue
            tbl = self._bin_table(sub, feature, bins, n_total)
            previews[sid] = tbl
            print(f"\n┌─ PREVIEW: dividir '{sid}'")
            print(f"│  feature = {feature} | tipo = {kind} | modo = {modo}")
            print(f"│  monotonicidade da PD respeitada: {tbl.attrs['mono_ok']}")
            print(tbl.to_string(index=False))
        print("\n→ se aprovado, repita como .grow(...) com os mesmos argumentos")
        return previews

    # ------------------------------------------------------------------
    # GROW: efetiva o split (manual ou ótimo, num ou cat) em cada folha-alvo
    # ------------------------------------------------------------------
    def grow(self, feature, splits=None, dtype=None, max_n_bins=4,
             min_bin_size=0.05, only_segments=None, max_bin_size=None, relax_max=False):
        targets = {
            sid: s for sid, s in self.segments.items()
            if s["is_leaf"] and (only_segments is None or sid in only_segments)
        }
        novos, modo_usado = {}, None
        for sid, seg in targets.items():
            sub = self.df[seg["mask"]]
            if splits is None:
                n_fit = len(self._fit_frame(sub, min_bin_size))
                if n_fit < self.min_leaf_rows:
                    print(f"[{sid}] poucas linhas para binning ótimo "
                          f"({n_fit} < {self.min_leaf_rows}) — folha mantida")
                    continue
            bins, modo, kind = self._resolve_bins(
                sub, feature, splits, dtype, max_n_bins, min_bin_size, max_bin_size,
                relax_max=relax_max)
            modo_usado = modo
            if not bins:
                print(f"[{sid}] sem corte válido em '{feature}' ({modo}) — folha mantida")
                continue
            filhos_sid = {}
            for b in bins:
                child_mask = seg["mask"] & self._mask_in(self.df, feature, b)
                if child_mask.sum() == 0:
                    continue
                child_label = self._bin_label(feature, b)
                child_id = f"{sid} | {child_label}" if sid != "root" else child_label
                filhos_sid[child_id] = {
                    "mask": child_mask,
                    "label": child_label,
                    "depth": seg["depth"] + 1,
                    "is_leaf": True,
                    "path": seg["path"] + [child_label],
                    "parent": sid,
                    "conditions": seg["conditions"] + [self._bin_condition(feature, b)],
                }
            if len(filhos_sid) < 2:        # 0/1 filho não-vazio = não separou de fato
                print(f"[{sid}] '{feature}' não separou ({len(filhos_sid)} filho) "
                      "— folha mantida")
                continue
            novos.update(filhos_sid)
            self.segments[sid]["is_leaf"] = False
        self.segments.update(novos)
        self.history.append({"feature": feature, "modo": modo_usado, "splits": splits})
        n_folhas = sum(s["is_leaf"] for s in self.segments.values())
        print(f"[grow] '{feature}' ({modo_usado}) criou {len(novos)} segmentos. "
              f"Folhas atuais: {n_folhas}")
        return self

    # ------------------------------------------------------------------
    # PRUNE: poda por FUSÃO de folhas-IRMÃS (mesmo pai). Em cada rodada funde um
    #   par de irmãs adjacentes que viole um dos critérios:
    #     - diferença de PD média entre as duas irmãs < `min_pd_gap` (ex.: 0.02
    #       = 2 p.p.) → separam pouco, devem ser unidas;
    #     - alguma das duas tem representatividade < `min_repr` % (imaterial) →
    #       é unida à irmã adjacente mais próxima em PD.
    #   Prioriza o par de menor diferença de PD. Só funde irmãs (o nó de
    #   faltantes não entra). Itera até nenhum par violar os critérios.
    # ------------------------------------------------------------------
    def prune(self, min_repr: float = 2.0, min_pd_gap: float = 0.02,
              protect: set | None = None, verbose: bool = True,
              max_rounds: int = 1000):
        protect = set(protect or [])
        n_total = len(self.df)
        n_merges = 0
        for _ in range(max_rounds):
            folhas_por_pai: dict[str, list[str]] = {}
            for sid, s in self.segments.items():
                if s["is_leaf"]:
                    folhas_por_pai.setdefault(s["parent"], []).append(sid)

            melhor = None  # (gap, sid_direita_do_par, motivo)
            for pai, folhas in folhas_por_pai.items():
                if pai is None:
                    continue
                irmaos = [c for c in folhas
                          if self.segments[c]["conditions"]
                          and self.segments[c]["conditions"][-1]["kind"] != "na"]
                if len(irmaos) < 2:
                    continue
                if all(self.segments[c]["conditions"][-1]["kind"] == "num"
                       for c in irmaos):
                    irmaos.sort(key=lambda c: self.segments[c]["conditions"][-1]["lo"])
                else:
                    irmaos.sort(key=lambda c: (self._leaf_target(c).mean()
                                               if len(self._leaf_target(c)) else np.inf))
                reprs = {c: (100 * int(self.segments[c]["mask"].sum()) / n_total
                             if n_total else 0.0) for c in irmaos}
                pds = {c: (self._leaf_target(c).mean()       # PD na referência (DES)
                           if len(self._leaf_target(c)) else np.nan) for c in irmaos}
                for i in range(len(irmaos) - 1):
                    a, b = irmaos[i], irmaos[i + 1]
                    if a in protect or b in protect:          # respeita folhas travadas
                        continue
                    gap = (abs(pds[b] - pds[a])
                           if not (pd.isna(pds[a]) or pd.isna(pds[b])) else np.inf)
                    viola_gap = gap < min_pd_gap
                    viola_repr = (reprs[a] < min_repr) or (reprs[b] < min_repr)
                    if not (viola_gap or viola_repr):
                        continue
                    motivo = "ΔPD" if viola_gap else "repr."
                    if melhor is None or gap < melhor[0]:
                        melhor = (gap, b, motivo)
            if melhor is None:
                break
            _, sid_dir, motivo = melhor
            antes = set(self.segments)
            self.merge_leaf(sid_dir, side="left", verbose=False)
            if set(self.segments) == antes:
                break  # salvaguarda contra laço infinito
            n_merges += 1
            if verbose:
                print(f"[prune] folhas-irmãs unidas ({motivo})")

        n_folhas = sum(s["is_leaf"] for s in self.segments.values())
        if verbose:
            print(f"[prune] {n_merges} fusão(ões) (repr<{min_repr}% ou ΔPD<{min_pd_gap}). "
                  f"Folhas finais: {n_folhas}")
        return self

    # ------------------------------------------------------------------
    # COLLAPSE: recolhe a subárvore enraizada em `sid`, transformando o nó
    #   de volta em folha (remove todos os descendentes). Desfaz um split.
    # ------------------------------------------------------------------
    def collapse(self, sid: str, verbose: bool = True):
        if sid not in self.segments:
            if verbose:
                print(f"[collapse] segmento '{sid}' não existe")
            return self
        if self.segments[sid]["is_leaf"]:
            if verbose:
                print(f"[collapse] '{sid}' já é folha — nada a recolher")
            return self
        # coleta todos os descendentes (BFS pelos ponteiros de pai)
        descendentes, frente = [], [c for c, s in self.segments.items() if s["parent"] == sid]
        while frente:
            atual = frente.pop()
            descendentes.append(atual)
            frente.extend([c for c, s in self.segments.items() if s["parent"] == atual])
        for d in descendentes:
            self.segments.pop(d, None)
        self.segments[sid]["is_leaf"] = True
        if verbose:
            print(f"[collapse] '{sid}' recolhido — {len(descendentes)} segmento(s) removido(s)")
        return self

    # ------------------------------------------------------------------
    # MERGE_LEAF: funde uma folha com a folha-irmã adjacente (mesmo pai).
    #   side="left"  -> vizinha de menor corte (num) / menor PD (cat)
    #   side="right" -> vizinha de maior corte (num) / maior PD (cat)
    #   Numérico: os dois intervalos viram um só (lo=min, hi=max).
    #   Categórico: as categorias dos dois grupos são unidas.
    #   Se o pai ficar com uma única folha, a fusão equivale a recolhê-lo.
    # ------------------------------------------------------------------
    def merge_leaf(self, sid: str, side: str = "left", verbose: bool = True):
        if sid not in self.segments:
            if verbose:
                print(f"[merge] segmento '{sid}' não existe")
            return self
        seg = self.segments[sid]
        if not seg["is_leaf"]:
            if verbose:
                print(f"[merge] '{sid}' não é folha")
            return self
        pai = seg["parent"]
        if pai is None:
            if verbose:
                print("[merge] a raiz não tem folhas vizinhas para fundir")
            return self

        cond = seg["conditions"][-1]
        feat, kind = cond["feature"], cond["kind"]
        if kind == "na":
            if verbose:
                print("[merge] o nó de faltantes não funde por vizinhança — use "
                      "merge_missing(<nó populado>) para juntá-lo a um bin da variável")
            return self

        # irmãos que dá para fundir: mesmo tipo de corte, excluindo o nó de faltantes
        irmaos = [c for c, s in self.segments.items()
                  if s["parent"] == pai and s["conditions"][-1]["kind"] != "na"]
        if kind == "num" and all(self.segments[c]["conditions"][-1]["kind"] == "num"
                                 for c in irmaos):
            irmaos.sort(key=lambda c: self.segments[c]["conditions"][-1]["lo"])
        else:
            # mesma chave (PD na referência/DES) usada por prune/auto_merge/teste —
            # senão a ordem das irmãs diverge e a fusão erra o par ou aborta
            irmaos.sort(key=lambda c: (self._leaf_target(c).mean()
                                       if len(self._leaf_target(c)) else np.inf))

        i = irmaos.index(sid)
        j = i - 1 if side == "left" else i + 1
        lado = "à esquerda" if side == "left" else "à direita"
        if j < 0 or j >= len(irmaos):
            if verbose:
                print(f"[merge] '{sid}' não tem folha {lado} para fundir")
            return self
        viz = irmaos[j]
        if not self.segments[viz]["is_leaf"]:
            if verbose:
                print(f"[merge] a folha {lado} foi subdividida — recolha-a antes de fundir")
            return self

        cond_viz = self.segments[viz]["conditions"][-1]
        inc_na = bool(cond.get("include_na") or cond_viz.get("include_na"))
        if kind == "num":
            new_cond = {"feature": feat, "kind": "num",
                        "lo": min(cond["lo"], cond_viz["lo"]),
                        "hi": max(cond["hi"], cond_viz["hi"])}
        else:
            cats = list(dict.fromkeys(list(cond["cats"]) + list(cond_viz["cats"])))
            new_cond = {"feature": feat, "kind": "cat", "cats": cats}
        if inc_na:
            new_cond["include_na"] = True

        new_mask = seg["mask"] | self.segments[viz]["mask"]
        new_label = self._bin_label(feat, new_cond) + (" + faltante" if inc_na else "")
        new_id = new_label if pai == "root" else f"{pai} | {new_label}"
        pai_seg = self.segments[pai]
        merged = {
            "mask": new_mask, "label": new_label, "depth": seg["depth"],
            "is_leaf": True, "path": pai_seg["path"] + [new_label],
            "parent": pai, "conditions": pai_seg["conditions"] + [new_cond],
        }
        self.segments.pop(sid, None)
        self.segments.pop(viz, None)
        self.segments[new_id] = merged

        filhos_pai = [c for c, s in self.segments.items() if s["parent"] == pai]
        if len(filhos_pai) == 1:
            self.segments.pop(new_id, None)
            self.segments[pai]["is_leaf"] = True
            if verbose:
                print(f"[merge] folhas fundidas {lado}; pai '{pai}' voltou a ser folha")
            return self
        if verbose:
            print(f"[merge] '{sid}' fundida com a folha {lado} → '{new_id}'")
        return self

    # ------------------------------------------------------------------
    # MERGE_MISSING: junta o nó de FALTANTES (na) do mesmo split DENTRO de um
    #   nó POPULADO da variável. A regra do destino passa a ser "<bin> OU
    #   faltante" (condição com include_na=True). Selecione sempre o nó populado
    #   de destino; o nó de faltantes é localizado automaticamente entre os
    #   irmãos. Diferente de merge_leaf (vizinhança), aqui a fusão é sempre
    #   válida porque faltante é disjunto de qualquer bin populado.
    # ------------------------------------------------------------------
    def merge_missing(self, sid: str, verbose: bool = True):
        if sid not in self.segments:
            if verbose:
                print(f"[merge_missing] segmento '{sid}' não existe")
            return self
        seg = self.segments[sid]
        if not seg["is_leaf"]:
            if verbose:
                print(f"[merge_missing] '{sid}' não é folha")
            return self
        if not seg["conditions"]:
            if verbose:
                print("[merge_missing] a raiz não tem nó de faltantes para juntar")
            return self
        if seg["conditions"][-1]["kind"] == "na":
            if verbose:
                print("[merge_missing] selecione o nó POPULADO de destino "
                      "(não o próprio nó de faltantes)")
            return self
        pai = seg["parent"]
        na_irmaos = [c for c, s in self.segments.items()
                     if s["parent"] == pai and s["is_leaf"] and s["conditions"]
                     and s["conditions"][-1]["kind"] == "na"]
        if not na_irmaos:
            if verbose:
                print("[merge_missing] este split não tem nó de faltantes "
                      "(ou ele já foi juntado)")
            return self
        return self._merge_missing_into(na_irmaos[0], sid, verbose=verbose)

    def _merge_missing_into(self, na_sid: str, target_sid: str, verbose: bool = True):
        """Funde o nó de faltantes `na_sid` no nó populado `target_sid` (mesmo pai)."""
        if na_sid not in self.segments or target_sid not in self.segments:
            return self
        na_seg, tgt = self.segments[na_sid], self.segments[target_sid]
        pai = tgt["parent"]
        if na_seg["parent"] != pai:
            if verbose:
                print("[merge_missing] o nó de faltantes é de outro split")
            return self

        cond = dict(tgt["conditions"][-1])      # cópia da última condição do destino
        cond["include_na"] = True
        if cond["kind"] == "num":
            base = self._bin_label(cond["feature"],
                                   {"kind": "num", "lo": cond["lo"], "hi": cond["hi"]})
        else:
            base = self._bin_label(cond["feature"],
                                   {"kind": "cat", "cats": cond["cats"]})
        new_label = base + " + faltante"
        new_id = new_label if pai == "root" else f"{pai} | {new_label}"
        pai_seg = self.segments[pai]
        merged = {
            "mask": tgt["mask"] | na_seg["mask"], "label": new_label,
            "depth": tgt["depth"], "is_leaf": True,
            "path": pai_seg["path"] + [new_label], "parent": pai,
            "conditions": pai_seg["conditions"] + [cond],
        }
        self.segments.pop(na_sid, None)
        self.segments.pop(target_sid, None)
        self.segments[new_id] = merged

        filhos_pai = [c for c, s in self.segments.items() if s["parent"] == pai]
        if len(filhos_pai) == 1:
            self.segments.pop(new_id, None)
            self.segments[pai]["is_leaf"] = True
            if verbose:
                print(f"[merge_missing] faltantes juntados; pai '{pai}' voltou a ser folha")
            return self
        if verbose:
            print(f"[merge_missing] faltantes juntados em → '{new_id}'")
        return self

    # ------------------------------------------------------------------
    # AUTO_MERGE: funde automaticamente pares de folhas-IRMÃS (mesmo pai)
    #   adjacentes que NÃO se distinguem em PD. Em cada rodada, escolhe o par
    #   de irmãs vizinhas mais parecido e o funde se:
    #     - o teste de hipótese não rejeita a igualdade de PD (p > alpha), OU
    #     - a diferença de PD média entre elas é < min_pd_gap.
    #   Só funde irmãs (a única fusão válida na árvore). Por padrão o nó de
    #   faltantes NÃO entra; com `include_missing=True` ele também é juntado ao
    #   bin populado irmão estatisticamente mais próximo. `protect` = ids de
    #   folhas a preservar (ex.: travadas na UI). Itera até nenhum par qualificar.
    # ------------------------------------------------------------------
    def auto_merge(self, alpha: float = 0.05, min_pd_gap: float = 0.0,
                   test: str = "mannwhitney", min_n: int = 8,
                   protect: set | None = None, include_missing: bool = False,
                   max_rounds: int = 200, verbose: bool = True):
        protect = set(protect or [])
        n_merges = 0
        for _ in range(max_rounds):
            # agrupa folhas por pai
            folhas_por_pai: dict[str, list[str]] = {}
            for sid, s in self.segments.items():
                if s["is_leaf"]:
                    folhas_por_pai.setdefault(s["parent"], []).append(sid)

            melhor = None  # (prioridade, ação, *args)
            for pai, folhas in folhas_por_pai.items():
                if pai is None or len(folhas) < 2:
                    continue
                populadas = [c for c in folhas
                             if self.segments[c]["conditions"]
                             and self.segments[c]["conditions"][-1]["kind"] != "na"]
                # --- fusão por vizinhança entre irmãs populadas ---
                if len(populadas) >= 2:
                    if all(self.segments[c]["conditions"][-1]["kind"] == "num"
                           for c in populadas):
                        populadas_ord = sorted(
                            populadas,
                            key=lambda c: self.segments[c]["conditions"][-1]["lo"])
                    else:
                        populadas_ord = sorted(
                            populadas,
                            key=lambda c: (self._leaf_target(c).mean()
                                           if len(self._leaf_target(c)) else np.inf))
                    for i in range(len(populadas_ord) - 1):
                        a, b = populadas_ord[i], populadas_ord[i + 1]
                        if a in protect or b in protect:
                            continue
                        va, vb = self._leaf_target(a), self._leaf_target(b)
                        gap = (abs(va.mean() - vb.mean())
                               if len(va) and len(vb) else np.inf)
                        p = self._pair_pvalue(a, b, test=test, min_n=min_n)
                        if not ((not np.isnan(p) and p > alpha) or gap < min_pd_gap):
                            continue
                        prio = p if not np.isnan(p) else (2.0 - min(gap, 1.0))
                        if melhor is None or prio > melhor[0]:
                            melhor = (prio, "adj", b)
                # --- fusão do nó de faltantes no bin populado mais próximo ---
                if include_missing:
                    na_leaves = [c for c in folhas
                                 if self.segments[c]["conditions"]
                                 and self.segments[c]["conditions"][-1]["kind"] == "na"]
                    for na_sid in na_leaves:
                        if na_sid in protect:
                            continue
                        for tgt in populadas:
                            if tgt in protect:
                                continue
                            va, vb = self._leaf_target(na_sid), self._leaf_target(tgt)
                            gap = (abs(va.mean() - vb.mean())
                                   if len(va) and len(vb) else np.inf)
                            p = self._pair_pvalue(na_sid, tgt, test=test, min_n=min_n)
                            if not ((not np.isnan(p) and p > alpha) or gap < min_pd_gap):
                                continue
                            prio = p if not np.isnan(p) else (2.0 - min(gap, 1.0))
                            if melhor is None or prio > melhor[0]:
                                melhor = (prio, "miss", na_sid, tgt)
            if melhor is None:
                break
            antes = set(self.segments)
            if melhor[1] == "adj":
                self.merge_leaf(melhor[2], side="left", verbose=False)
            else:
                self._merge_missing_into(melhor[2], melhor[3], verbose=False)
            if set(self.segments) == antes:
                break  # salvaguarda: nenhuma mudança → evita laço infinito
            n_merges += 1

        n_folhas = sum(s["is_leaf"] for s in self.segments.values())
        if verbose:
            print(f"[auto_merge] {n_merges} fusão(ões) automática(s) "
                  f"(alpha={alpha}, min_pd_gap={min_pd_gap}, "
                  f"include_missing={include_missing}). Folhas finais: {n_folhas}")
        return self

    # ------------------------------------------------------------------
    # Descrição por extenso de um segmento a partir das suas condições
    # ------------------------------------------------------------------
    def _descrever(self, conditions: list) -> str:
        if not conditions:
            return "população total"
        partes = []
        for c in conditions:
            rotulo = self.feature_labels.get(c["feature"], c["feature"].replace("_", " "))
            na = c.get("include_na")
            if c["kind"] == "na":
                partes.append(f"{rotulo} faltante")
            elif c["kind"] == "num":
                base = f"{rotulo} {_intervalo_por_extenso(c['lo'], c['hi'])}"
                partes.append(f"({base} ou faltante)" if na else base)
            else:
                cats = " ou ".join(map(str, c["cats"]))
                base = f"{rotulo} em {{{cats}}}"
                partes.append(f"({base} ou faltante)" if na else base)
        return " e ".join(partes)

    # ------------------------------------------------------------------
    # Ordenação ESQUERDA→DIREITA das folhas (posição na árvore construída).
    #   A nota_pd passa a ser essa posição: 1, 2, 3, … da esquerda p/ a direita.
    #   Em cada nó os filhos são ordenados pela MENOR PD (DES) de folha do ramo
    #   (asc. = ramo de menor risco à esquerda), espelhando o layout de plot_tree.
    #   Para um split único, posição = ordem de PD (idêntico ao comportamento
    #   anterior); em árvores profundas, as notas leem 1, 2, 3 de fato.
    # ------------------------------------------------------------------
    def _node_pd(self, sid: str) -> float:
        """PD média do nó na amostra de referência (DES), com fallback p/ todas."""
        sub = self.df[self.segments[sid]["mask"]]
        if self.sample_col is not None:
            sr = sub.loc[sub[self.sample_col] == self.ref_sample, self.target]
            if len(sr):
                return float(sr.mean())
        return float(sub[self.target].mean()) if len(sub) else float("nan")

    def _leaf_order(self, ascending: bool = True) -> list:
        """sids das folhas na ordem esquerda→direita da árvore (ver bloco acima)."""
        filhos: dict = {}
        for sid, s in self.segments.items():
            filhos.setdefault(s["parent"], []).append(sid)
        INF = float("inf")
        leaf_pd = {sid: self._node_pd(sid)
                   for sid, s in self.segments.items() if s["is_leaf"]}
        _submin: dict = {}

        def submin(sid):
            if sid not in _submin:
                if self.segments[sid]["is_leaf"]:
                    v = leaf_pd.get(sid, INF)
                    _submin[sid] = INF if (v is None or pd.isna(v)) else v
                else:
                    _submin[sid] = min((submin(c) for c in filhos.get(sid, [])),
                                       default=INF)
            return _submin[sid]

        sgn = 1 if ascending else -1
        order: list = []

        def dfs(sid):
            if self.segments[sid]["is_leaf"]:
                order.append(sid)
                return
            for c in sorted(filhos.get(sid, []), key=lambda c: (sgn * submin(c), str(c))):
                dfs(c)

        dfs("root")
        for sid, s in self.segments.items():
            if s["is_leaf"] and sid not in order:
                order.append(sid)
        return order

    # ------------------------------------------------------------------
    # LEAVES: segmentos-folha finais, com nota de PD e descrição
    #   nota_pd: 1..N pela POSIÇÃO esquerda→direita na árvore
    #   with_psi:  adiciona a contribuição de PSI de cada folha por amostra
    #              (psi_<amostra>), tendo a referência (DES) como base
    #   with_test: adiciona p_vs_prox = p-valor do teste de PD entre a folha e a
    #              IRMÃ adjacente (mesmo pai — só irmãs são comparáveis/fundíveis).
    #              p alto ⇒ irmãs não distinguíveis (candidatas a fusão); folhas sem
    #              irmã à frente ficam com NaN. test: 'mannwhitney' (default)/'welch'
    # ------------------------------------------------------------------
    def leaves(self, ascending: bool = True, with_psi: bool = False,
               with_test: bool = False, test: str = "mannwhitney") -> pd.DataFrame:
        linhas, n_total = [], len(self.df)
        for sid, seg in self.segments.items():
            if not seg["is_leaf"]:
                continue
            sub = self.df[seg["mask"]]
            # pd_medio na referência (DES) = base da régua; assim nota_pd é
            # monotônica na PD que entra em predict/apply_spark (com fallback)
            if self.sample_col is not None:
                ref_t = sub.loc[sub[self.sample_col] == self.ref_sample, self.target]
                pd_m = ref_t.mean() if len(ref_t) else sub[self.target].mean()
            else:
                pd_m = sub[self.target].mean()
            row = {
                "segmento": sid,
                "descricao": self._descrever(seg["conditions"]),
                "profundidade": seg["depth"],
                "n": len(sub),
                "repr_%": round(100 * len(sub) / n_total, 1) if n_total else 0.0,
                "pd_medio": round(pd_m, 4) if not pd.isna(pd_m) else np.nan,
                "pd_std": round(sub[self.target].std(), 4),
            }
            if self.sample_col is not None:
                for amostra in self.df[self.sample_col].dropna().unique():
                    s_am = sub.loc[sub[self.sample_col] == amostra, self.target]
                    row[f"pd_{amostra}"] = round(s_am.mean(), 4) if len(s_am) else np.nan
            linhas.append(row)
        # nota_pd = POSIÇÃO esquerda→direita na árvore (ver _leaf_order); assim os
        # números sempre leem 1, 2, 3 da esquerda p/ a direita no plot_tree.
        ordem = {sid: i for i, sid in enumerate(self._leaf_order(ascending=ascending))}
        out = (
            pd.DataFrame(linhas)
            .sort_values("segmento", key=lambda s: s.map(ordem), kind="stable")
            .reset_index(drop=True)
        )
        out.insert(1, "nota_pd", range(1, len(out) + 1))

        if with_psi and self.sample_col is not None:
            out = self._append_psi_cols(out)
        if with_test:
            out = self._append_adjacency_test(out, test=test)
        return out

    # contribuição de PSI por folha para cada amostra (≠ referência)
    def _append_psi_cols(self, out, eps: float = 1e-6):
        leaf_ids = out["segmento"].tolist()
        masks = {sid: self.segments[sid]["mask"] for sid in leaf_ids}
        ref_mask = self.df[self.sample_col] == self.ref_sample
        n_ref = ref_mask.sum()
        ref_pct = {sid: ((masks[sid] & ref_mask).sum() / n_ref if n_ref else 0.0)
                   for sid in leaf_ids}
        for amostra in self.df[self.sample_col].dropna().unique():
            if amostra == self.ref_sample:
                continue
            s_mask = self.df[self.sample_col] == amostra
            n_s = s_mask.sum()
            col = []
            for sid in leaf_ids:
                p_ref = max(ref_pct[sid], eps)
                p_cur = max((masks[sid] & s_mask).sum() / n_s if n_s else 0.0, eps)
                col.append(round((p_cur - p_ref) * np.log(p_cur / p_ref), 4))
            out[f"psi_{amostra}"] = col
        return out

    # alvo (PD) de uma folha, restrito à amostra de referência quando houver
    # (sem NaN, para os testes/fusões não ficarem cegos por valores ausentes)
    def _leaf_target(self, sid: str) -> np.ndarray:
        m = self.segments[sid]["mask"]
        if self.sample_col is not None:
            m = m & (self.df[self.sample_col] == self.ref_sample)
        vals = self.df.loc[m, self.target].to_numpy(dtype="float64")
        return vals[~np.isnan(vals)]

    # p-valor do teste de igualdade de PD entre duas folhas (na referência)
    def _pair_pvalue(self, sid_a: str, sid_b: str, test: str = "mannwhitney",
                     min_n: int = 8) -> float:
        try:
            from scipy.stats import mannwhitneyu, ttest_ind
        except Exception:
            return np.nan
        a, b = self._leaf_target(sid_a), self._leaf_target(sid_b)
        if len(a) < min_n or len(b) < min_n:
            return np.nan
        try:
            if test == "welch":
                return float(ttest_ind(a, b, equal_var=False).pvalue)
            return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except Exception:
            return np.nan

    # p-valor do teste de PD entre uma folha e a IRMÃ adjacente (mesmo pai).
    #   Só irmãs são diretamente comparáveis (e fundíveis), por isso o teste é
    #   restrito a elas — folhas de pais diferentes não se comparam. A ordem das
    #   irmãs é a mesma da fusão (por corte, no numérico; por PD, no categórico).
    #   A última irmã do grupo e o nó de faltantes ficam com NaN (sem par à frente).
    def _append_adjacency_test(self, out, test="mannwhitney", min_n=8):
        leaf_ids = out["segmento"].tolist()
        por_pai: dict = {}
        for sid in leaf_ids:
            por_pai.setdefault(self.segments[sid]["parent"], []).append(sid)
        prox: dict = {}
        for irmaos in por_pai.values():
            comp = [c for c in irmaos
                    if self.segments[c]["conditions"]
                    and self.segments[c]["conditions"][-1]["kind"] != "na"]
            if len(comp) < 2:
                continue
            if all(self.segments[c]["conditions"][-1]["kind"] == "num" for c in comp):
                comp.sort(key=lambda c: self.segments[c]["conditions"][-1]["lo"])
            else:
                comp.sort(key=lambda c: (self._leaf_target(c).mean()
                                         if len(self._leaf_target(c)) else np.inf))
            for i in range(len(comp) - 1):
                prox[comp[i]] = comp[i + 1]
        pvals = []
        for sid in leaf_ids:
            nxt = prox.get(sid)
            if nxt is None:
                pvals.append(np.nan)
                continue
            p = self._pair_pvalue(sid, nxt, test=test, min_n=min_n)
            pvals.append(round(p, 4) if not np.isnan(p) else np.nan)
        out["p_vs_prox"] = pvals
        return out

    # mapeia segmento -> (nota, descrição) para uso no assign
    def _grade_map(self, ascending: bool = True):
        lv = self.leaves(ascending=ascending)
        nota = dict(zip(lv["segmento"], lv["nota_pd"]))
        desc = dict(zip(lv["segmento"], lv["descricao"]))
        return nota, desc

    # ------------------------------------------------------------------
    # TREE: desenha a árvore hierárquica (nós internos + folhas) em texto.
    #   Cada nó mostra n, representatividade e PD média; folhas trazem [nota N].
    #   Filhos são ordenados por PD (ascendente por padrão).
    # ------------------------------------------------------------------
    def tree(self, ascending: bool = True, prune_preview: bool = False) -> str:
        # mapa pai -> filhos
        filhos: dict = {}
        for sid, seg in self.segments.items():
            filhos.setdefault(seg["parent"], []).append(sid)

        nota_map, _ = self._grade_map(ascending=ascending)
        n_total = len(self.df)
        linhas: list[str] = []

        def stats(sid):
            sub = self.df[self.segments[sid]["mask"]]
            pdv = sub[self.target].mean() if len(sub) else float("nan")
            return len(sub), 100 * len(sub) / n_total, pdv

        def rotulo(sid):
            seg = self.segments[sid]
            if seg["parent"] is None:
                return "TODA A CARTEIRA"
            return self._descrever([seg["conditions"][-1]])

        # ordena os filhos pela MENOR nota do ramo (esquerda→direita), igual ao
        # plot_tree — assim o texto lê as folhas 1, 2, 3 na mesma ordem
        _min_nota: dict = {}

        def min_nota(sid):
            if sid not in _min_nota:
                if self.segments[sid]["is_leaf"]:
                    _min_nota[sid] = nota_map.get(sid, 10 ** 9)
                else:
                    _min_nota[sid] = min((min_nota(c) for c in filhos.get(sid, [])),
                                         default=10 ** 9)
            return _min_nota[sid]

        def rec(sid, prefix, is_last, is_root=False):
            n, rep, pdv = stats(sid)
            seg = self.segments[sid]
            conn = "" if is_root else ("└─ " if is_last else "├─ ")
            tag = f"  [nota {nota_map.get(sid, '?')}]" if seg["is_leaf"] else ""
            linhas.append(f"{prefix}{conn}{rotulo(sid)}  "
                          f"(n={n}, {rep:.1f}%, PD={pdv:.4f}){tag}")
            ch = sorted(filhos.get(sid, []), key=min_nota)
            child_prefix = "" if is_root else prefix + ("   " if is_last else "│  ")
            for i, c in enumerate(ch):
                rec(c, child_prefix, i == len(ch) - 1)

        rec("root", "", True, is_root=True)
        n_folhas = sum(s["is_leaf"] for s in self.segments.values())
        linhas.append(f"\n({n_folhas} folhas | profundidade máxima "
                      f"{max(s['depth'] for s in self.segments.values())})")
        out = "\n".join(linhas)
        print(out)
        return out

    # ------------------------------------------------------------------
    # _pd_color_range: faixa (vmin, vmax) da escala de cor da PD, a partir das
    #   PDs observadas nos nós. Diferente da LGD (fixa 0–1), a PD costuma ser
    #   pequena (ex.: 0–0.3), então a escala é dinâmica para as cores
    #   discriminarem — ancorada em 0 (sem default = verde).
    # ------------------------------------------------------------------
    def _pd_color_range(self):
        vals = []
        for sid, s in self.segments.items():
            sub = self.df[s["mask"]]
            if len(sub):
                v = sub[self.target].mean()
                if not pd.isna(v):
                    vals.append(float(v))
        vmax = max(vals) if vals else 0.01
        return 0.0, (vmax if vmax > 1e-9 else 0.01)

    # ------------------------------------------------------------------
    # PLOT_TREE: desenha a árvore como IMAGEM (matplotlib). Cada nó mostra o
    #   rótulo da condição, n, % e PD média; folhas trazem a nota. A cor do nó
    #   reflete a PD (verde = baixa → vermelho = alta). Devolve a Figure e,
    #   se `save_path` for dado, salva a imagem (PNG/SVG/PDF…).
    # ------------------------------------------------------------------
    def plot_tree(self, ascending: bool = True, figsize=None, cmap: str = "RdYlGn_r",
                  show_samples: bool = False, title: str | None = "Segmentação de PD",
                  save_path: str | None = None, dpi: int = 150, ax=None,
                  highlight: str | None = None):
        try:
            import textwrap

            import matplotlib.pyplot as plt
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            from matplotlib.patches import FancyBboxPatch
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_tree requer matplotlib — use: pip install matplotlib") from e

        filhos: dict = {}
        for sid, s in self.segments.items():
            filhos.setdefault(s["parent"], []).append(sid)

        n_total = len(self.df)
        nota_map, _ = self._grade_map(ascending=ascending)
        ref = self.ref_sample if self.sample_col is not None else None

        def stats(sid):
            m = self.segments[sid]["mask"]
            n = int(m.sum())
            sub = self.df[m]
            if ref is not None:
                sr = sub.loc[sub[self.sample_col] == ref, self.target]
                pdv = sr.mean() if len(sr) else (sub[self.target].mean() if n else float("nan"))
            else:
                pdv = sub[self.target].mean() if n else float("nan")
            return n, (100 * n / n_total if n_total else 0.0), pdv

        def sample_pd(sid, a):
            m = self.segments[sid]["mask"] & (self.df[self.sample_col] == a)
            sub = self.df[m]
            return sub[self.target].mean() if len(sub) else float("nan")

        # menor nota_pd entre as folhas de cada ramo — usada para ordenar os
        # ramos da esquerda (menor nota) para a direita (maior nota)
        _min_nota: dict = {}

        def min_nota(sid):
            if sid not in _min_nota:
                if self.segments[sid]["is_leaf"]:
                    _min_nota[sid] = nota_map.get(sid, 10 ** 9)
                else:
                    _min_nota[sid] = min((min_nota(c) for c in filhos.get(sid, [])),
                                         default=10 ** 9)
            return _min_nota[sid]

        # --- layout: folhas em x sequencial na ORDEM DE nota_pd (esq.→dir.);
        #     nós internos centralizados sobre o intervalo dos filhos ---
        X_GAP, Y_GAP = 2.4, 2.15        # espaçamento entre folhas / entre níveis
        bw, bh = 0.97, 0.74             # meia-largura / meia-altura do box (dados)
        pos: dict = {}
        counter = [0]
        max_depth = [0]

        def place(sid, depth):
            max_depth[0] = max(max_depth[0], depth)
            ch = sorted(filhos.get(sid, []), key=min_nota)   # ramos por menor nota
            if self.segments[sid]["is_leaf"] or not ch:
                x = counter[0] * X_GAP
                counter[0] += 1
            else:
                cxs = [place(c, depth + 1) for c in ch]
                x = 0.5 * (min(cxs) + max(cxs))              # centra sobre os filhos
            pos[sid] = (x, -depth * Y_GAP)
            return x

        place("root", 0)
        n_leaves = max(counter[0], 1)
        md = max_depth[0]
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]

        if figsize is None:
            figsize = (max(7.0, (max(xs) - min(xs) + 2 * bw + 1.0) * 0.95),
                       max(3.5, (md + 1) * Y_GAP * 0.95))
        fig, ax = self._new_ax(figsize, dpi, ax)

        vmin, vmax = self._pd_color_range()      # escala de cor da PD (dinâmica)
        norm = Normalize(vmin, vmax)
        cmap_obj = plt.get_cmap(cmap)

        def rotulo(sid):
            s = self.segments[sid]
            if s["parent"] is None:
                return "TODA A CARTEIRA"
            return self._descrever([s["conditions"][-1]])

        # arestas (atrás dos nós)
        for sid, (x, y) in pos.items():
            for c in filhos.get(sid, []):
                cx, cy = pos[c]
                ax.plot([x, cx], [y - bh, cy + bh], color="#9aa7b2", lw=1.1, zorder=1)

        base_fs = 8.6
        fit_items = []            # (texto, meia-largura, meia-altura) p/ ajuste de fonte
        for sid, (x, y) in pos.items():
            _, rep, pdv = stats(sid)
            color = cmap_obj(norm(pdv)) if not pd.isna(pdv) else (0.88, 0.88, 0.88, 1.0)
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            txt_color = "#15324a" if lum > 0.6 else "#ffffff"
            is_leaf = self.segments[sid]["is_leaf"]
            selecionada = (highlight is not None and sid == highlight)
            # folha selecionada: contorno destacado (âmbar grosso) + leve glow
            if selecionada:
                ax.add_patch(FancyBboxPatch(
                    (x - bw - 0.06, y - bh - 0.06), 2 * bw + 0.12, 2 * bh + 0.12,
                    boxstyle="round,pad=0.02,rounding_size=0.16",
                    linewidth=0, facecolor="#f5a623", alpha=0.30, zorder=1.5))
            ax.add_patch(FancyBboxPatch(
                (x - bw, y - bh), 2 * bw, 2 * bh,
                boxstyle="round,pad=0.02,rounding_size=0.14",
                linewidth=(3.2 if selecionada else 1.3),
                edgecolor=("#e8870b" if selecionada else "#33424f"),
                facecolor=color, zorder=2))
            # 1) QUEBRA (condição do nó) em NEGRITO, no topo da caixa
            cab = "\n".join(textwrap.wrap(rotulo(sid), 18)[:3])
            t_split = ax.text(x, y + 0.34 * bh, cab, ha="center", va="center",
                              fontsize=base_fs, color=txt_color, zorder=3,
                              fontweight="bold", linespacing=1.12, clip_on=True)
            fit_items.append((t_split, bw * 0.94, bh * 0.58))
            # 2) representatividade e PD (em %, 2 casas) na MESMA linha, separados por barra
            pd_txt = f"PD {pdv * 100:.2f}%" if not pd.isna(pdv) else "PD —"
            metr = f"repr. {rep:.1f}%  |  {pd_txt}"
            if show_samples and self.sample_col is not None:
                amostras = list(self.df[self.sample_col].dropna().unique())
                metr += "\n" + " | ".join(f"{a} {sample_pd(sid, a) * 100:.2f}%"
                                          for a in amostras)
            t_metr = ax.text(x, y - 0.42 * bh, metr, ha="center", va="center",
                             fontsize=base_fs - 0.8, color=txt_color, zorder=3,
                             linespacing=1.12, clip_on=True)
            fit_items.append((t_metr, bw * 0.94, bh * 0.34))
            # 3) número da folha no CANTO INFERIOR DIREITO
            if is_leaf:
                ax.text(x + bw * 0.93, y - bh * 0.9, f"folha {nota_map.get(sid, '?')}",
                        ha="right", va="bottom", fontsize=base_fs - 1.8, color=txt_color,
                        zorder=3, fontweight="bold", clip_on=True)

        ax.set_xlim(min(xs) - bw - 0.4, max(xs) + bw + 0.4)
        ax.set_ylim(min(ys) - bh - 0.4, max(ys) + bh + 0.6)
        ax.axis("off")
        if title:
            n_folhas = sum(s["is_leaf"] for s in self.segments.values())
            ax.set_title(f"{title}  ·  {n_folhas} folhas",
                         fontsize=12.5, fontweight="bold", color="#15324a")
        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
        from matplotlib.ticker import PercentFormatter
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
        cbar.set_label(f"PD média{' (DES)' if ref else ''} "
                       f"(escala 0–{vmax * 100:.1f}%)", fontsize=9)
        fig.tight_layout()

        # força cada texto a caber na sua sub-região, encolhendo a fonte se preciso
        self._fit_texts_to_boxes(fig, ax, fit_items)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    @staticmethod
    def _fit_texts_to_boxes(fig, ax, items, pad: float = 0.92, min_fs: float = 4.0):
        """Encolhe a fonte de cada texto até caber na sua sub-região.

        ``items``: lista de ``(texto, meia_largura, meia_altura)`` em coordenadas de
        dados, centradas na posição do texto.
        """
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        except Exception:        # pragma: no cover - backend sem renderer
            return
        for t, hw, hh in items:
            x, y = t.get_position()
            px0, py0 = ax.transData.transform((x - hw, y - hh))
            px1, py1 = ax.transData.transform((x + hw, y + hh))
            box_w, box_h = abs(px1 - px0) * pad, abs(py1 - py0) * pad
            ext = t.get_window_extent(renderer)
            if ext.width <= box_w and ext.height <= box_h:
                continue
            scale = min(box_w / max(ext.width, 1e-6), box_h / max(ext.height, 1e-6))
            t.set_fontsize(max(min_fs, t.get_fontsize() * scale))

    # ------------------------------------------------------------------
    # PSI: estabilidade populacional, segmentos-folha como bins
    # ------------------------------------------------------------------
    def psi(self, eps: float = 1e-6) -> pd.DataFrame:
        if self.sample_col is None:
            raise ValueError("PSI requer sample_col definido no construtor.")
        leaf_ids = [sid for sid, s in self.segments.items() if s["is_leaf"]]
        if not leaf_ids:
            raise ValueError("Nenhuma folha — cresça a segmentação antes do PSI.")

        dist = {}
        for amostra in self.df[self.sample_col].dropna().unique():
            mask_am = self.df[self.sample_col] == amostra
            n_am = mask_am.sum()
            dist[amostra] = {
                sid: ((self.segments[sid]["mask"] & mask_am).sum() / n_am if n_am else 0.0)
                for sid in leaf_ids
            }

        ref = dist[self.ref_sample]
        self._psi_detalhe = []
        linhas = []
        for amostra, pct in dist.items():
            if amostra == self.ref_sample:
                continue
            psi_total = 0.0
            for sid in leaf_ids:
                p_ref = max(ref[sid], eps)
                p_cur = max(pct[sid], eps)
                contrib = (p_cur - p_ref) * np.log(p_cur / p_ref)
                psi_total += contrib
                self._psi_detalhe.append({
                    "amostra": amostra, "segmento": sid,
                    f"%_{self.ref_sample}": round(100 * ref[sid], 2),
                    "%_atual": round(100 * pct[sid], 2),
                    "psi_bin": round(contrib, 4),
                })
            linhas.append({
                "amostra": amostra,
                "psi": round(psi_total, 4),
                "classificacao": _classifica_psi(psi_total),
            })
        return pd.DataFrame(linhas).sort_values("psi", ascending=False).reset_index(drop=True)

    def psi_detalhe(self) -> pd.DataFrame:
        """Contribuição de cada segmento para o PSI (chame após .psi())."""
        return pd.DataFrame(self._psi_detalhe)

    # ------------------------------------------------------------------
    # CSI: Characteristic Stability Index POR VARIÁVEL.
    #   Enquanto o `psi()` mede a estabilidade da SEGMENTAÇÃO (folhas como
    #   bins), o CSI mede a estabilidade da DISTRIBUIÇÃO DE CADA VARIÁVEL DE
    #   ENTRADA entre a referência (DES) e cada outra amostra (OOT, ...). Os
    #   bins são fixados na referência (mesmo binning ótimo/manual do split) e
    #   a fórmula é a do PSI aplicada à variável. Útil para detectar QUAL
    #   característica está migrando antes mesmo de entrar na árvore.
    #   `csi_detalhe()` traz a contribuição de cada faixa (decomposição).
    # ------------------------------------------------------------------
    def csi(self, features: list | None = None, max_n_bins: int = 10,
            min_bin_size: float = 0.05, eps: float = 1e-6) -> pd.DataFrame:
        if self.sample_col is None:
            raise ValueError("CSI requer sample_col definido (precisa de amostras "
                             "para comparar a referência com as demais).")
        if features is None:
            skip = {self.target, self.sample_col, self.date_col}
            features = [c for c in self.df.columns if c not in skip]

        nonref = [a for a in self.df[self.sample_col].dropna().unique()
                  if a != self.ref_sample]
        ref_mask = self.df[self.sample_col] == self.ref_sample
        n_ref = int(ref_mask.sum())
        self._csi_detalhe = []

        rows = []
        for feat in features:
            try:
                bins, _modo, kind = self._resolve_bins(
                    self.df, feat, None, None, max_n_bins, min_bin_size)
            except Exception:
                bins, kind = [], "—"
            row = {"variavel": feat, "tipo": kind, "n_bins": len(bins)}
            if not bins or n_ref == 0:
                for a in nonref:
                    row[f"csi_{a}"] = np.nan
                row["pior_csi"], row["classificacao"] = np.nan, "—"
                rows.append(row)
                continue

            ref_pct = [((self._mask_in(self.df, feat, b) & ref_mask).sum() / n_ref)
                       for b in bins]
            pior = 0.0
            for a in nonref:
                a_mask = self.df[self.sample_col] == a
                n_a = int(a_mask.sum())
                csi_tot = 0.0
                for b, p_ref in zip(bins, ref_pct):
                    p_cur = ((self._mask_in(self.df, feat, b) & a_mask).sum() / n_a
                             if n_a else 0.0)
                    pr, pc = max(p_ref, eps), max(p_cur, eps)
                    contrib = (pc - pr) * np.log(pc / pr)
                    csi_tot += contrib
                    self._csi_detalhe.append({
                        "variavel": feat, "amostra": a,
                        "faixa": self._bin_label(feat, b),
                        f"%_{self.ref_sample}": round(100 * p_ref, 2),
                        "%_atual": round(100 * p_cur, 2),
                        "csi_bin": round(contrib, 4),
                    })
                row[f"csi_{a}"] = round(csi_tot, 4)
                pior = max(pior, csi_tot)
            row["pior_csi"] = round(pior, 4)
            row["classificacao"] = _classifica_psi(pior)
            rows.append(row)

        return (pd.DataFrame(rows)
                .sort_values("pior_csi", ascending=False, na_position="last")
                .reset_index(drop=True))

    def csi_detalhe(self) -> pd.DataFrame:
        """Contribuição de cada faixa para o CSI por variável (chame após .csi())."""
        return pd.DataFrame(self._csi_detalhe)

    # ------------------------------------------------------------------
    # METRICS: avalia a régua como um modelo de PD (classificação).
    #   Score de cada linha = PD média do seu segmento na amostra de referência
    #   (DES). Retorna, por amostra (DES, OOT, ...): taxa de default observada,
    #   **KS**, **AUC** (ROC), **Gini**, **Acurácia** e **F1**. Acurácia/F1 usam
    #   o corte KS-ótimo por amostra (ou `cutoff`, se informado).
    # ------------------------------------------------------------------
    def metrics(self, cutoff: float | None = None) -> pd.DataFrame:
        from yggdrasil.metrics.classification import classification_metrics

        leaf_ids = [sid for sid, s in self.segments.items() if s["is_leaf"]]
        if self.sample_col is not None:
            ref_mask = self.df[self.sample_col] == self.ref_sample
        else:
            ref_mask = pd.Series(True, index=self.df.index)
        overall = self.df.loc[ref_mask, self.target].mean()

        # score por linha = PD média do segmento na referência (régua)
        pred = pd.Series(np.nan, index=self.df.index)
        for sid in leaf_ids:
            m = self.segments[sid]["mask"]
            v = self.df.loc[m & ref_mask, self.target].mean()
            pred[m.values] = overall if pd.isna(v) else v

        # ordem das amostras: referência primeiro
        if self.sample_col is not None:
            todas = list(self.df[self.sample_col].dropna().unique())
            ordem = ([self.ref_sample] + [a for a in todas if a != self.ref_sample])
            grupos = [(a, self.df[self.sample_col] == a) for a in ordem]
        else:
            grupos = [("todos", pd.Series(True, index=self.df.index))]

        linhas = []
        for nome, mask in grupos:
            y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
            yhat = pred[mask.values].to_numpy(dtype="float64")
            valid = ~(np.isnan(y) | np.isnan(yhat))      # ignora alvo/score NaN
            y, yhat = y[valid], yhat[valid]
            base = {"amostra": nome, "n": int(mask.sum())}
            if len(y) == 0:
                base.update({"taxa_default": np.nan, "KS": np.nan, "AUC": np.nan,
                             "Gini": np.nan, "Acuracia": np.nan, "F1": np.nan})
                linhas.append(base)
                continue
            cm = classification_metrics(y, yhat, cutoff=cutoff)
            base.update({
                "taxa_default": round(float(np.mean(y)), 4),
                "KS": cm["ks"], "AUC": cm["auc"], "Gini": cm["gini"],
                "Acuracia": cm["accuracy"], "F1": cm["f1"],
            })
            linhas.append(base)
        return pd.DataFrame(linhas)

    # ------------------------------------------------------------------
    # BOOTSTRAP_CI: intervalo de confiança da PD (taxa de default) por folha,
    #   via reamostragem bootstrap na amostra `sample` (default = referência/DES).
    #   Se houver `check_sample` (default = 1ª não-referência, ex. OOT), traz a
    #   PD dela por folha e verifica a ADERÊNCIA: se a PD de OOT cai dentro do
    #   IC bootstrap do DES (estável) ou fora (acima/abaixo = alerta).
    # ------------------------------------------------------------------
    def bootstrap_ci(self, n_boot: int = 1000, ci: float = 0.95,
                     sample: str | None = None, check_sample: str | None = None,
                     seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        alpha = (1 - ci) / 2
        if self.sample_col is not None:
            if sample is None:
                sample = self.ref_sample
            if check_sample is None:
                nonref = [a for a in self.df[self.sample_col].dropna().unique()
                          if a != self.ref_sample]
                check_sample = nonref[0] if nonref else None
        else:
            sample = check_sample = None

        lv = self.leaves()  # ordem de nota
        rows = []
        for _, r in lv.iterrows():
            sid = r["segmento"]
            m = self.segments[sid]["mask"]
            m_s = (m & (self.df[self.sample_col] == sample)
                   if (self.sample_col is not None and sample is not None) else m)
            vals = self.df.loc[m_s, self.target].to_numpy(dtype="float64")
            vals = vals[~np.isnan(vals)]
            n = len(vals)
            if n >= 2:
                idx = rng.integers(0, n, size=(n_boot, n))
                means = vals[idx].mean(axis=1)
                lo, hi = np.quantile(means, [alpha, 1 - alpha])
                pt = float(vals.mean())
            elif n == 1:
                lo = hi = pt = float(vals[0])
            else:
                lo = hi = pt = np.nan

            row = {
                "nota_pd": r["nota_pd"],
                "descricao": r["descricao"],
                "n": int(n),
                f"pd_{sample or 'todos'}": round(pt, 4) if not np.isnan(pt) else np.nan,
                "ic_low": round(float(lo), 4) if not np.isnan(lo) else np.nan,
                "ic_high": round(float(hi), 4) if not np.isnan(hi) else np.nan,
                "amplitude": round(float(hi - lo), 4) if not np.isnan(hi) else np.nan,
            }
            if check_sample is not None:
                m_c = m & (self.df[self.sample_col] == check_sample)
                cvals = self.df.loc[m_c, self.target].to_numpy(dtype="float64")
                cvals = cvals[~np.isnan(cvals)]
                pd_c = float(cvals.mean()) if len(cvals) else np.nan
                row[f"pd_{check_sample}"] = round(pd_c, 4) if not np.isnan(pd_c) else np.nan
                if np.isnan(pd_c) or np.isnan(lo):
                    row["aderente"] = None
                    row["status_oot"] = "—"
                else:
                    dentro = bool(lo <= pd_c <= hi)
                    row["aderente"] = dentro
                    row["status_oot"] = ("dentro" if dentro
                                         else "acima" if pd_c > hi else "abaixo")
            rows.append(row)

        out = pd.DataFrame(rows)
        out.attrs.update(sample=sample, check_sample=check_sample, ci=ci, n_boot=n_boot)
        return out

    # ==================================================================
    # VALIDAÇÃO: backtesting por safra, monotonicidade e calibração
    # ==================================================================
    def _predicted_pd_series(self) -> pd.Series:
        """PD prevista por linha = PD média do segmento na referência (régua)."""
        pred = pd.Series(np.nan, index=self.df.index, dtype="float64")
        for sid, seg in self.segments.items():
            if not seg["is_leaf"]:
                continue
            sub = self.df[seg["mask"]]
            if self.sample_col is not None:
                ref = sub.loc[sub[self.sample_col] == self.ref_sample, self.target]
                pdv = float(ref.mean()) if len(ref) else float(sub[self.target].mean())
            else:
                pdv = float(sub[self.target].mean()) if len(sub) else np.nan
            pred[seg["mask"]] = pdv
        return pred

    # ------------------------------------------------------------------
    # BACKTEST: PD prevista (régua) vs realizada ao longo do tempo (safra).
    #   `time_col` = coluna de período (ex.: dt_ref). Em cada período traz n,
    #   PD prevista, taxa de default realizada, o gap e um status (ok/alerta
    #   por `tol`).
    # ------------------------------------------------------------------
    def backtest(self, time_col: str, sample: str | None = None,
                 tol: float = 0.03) -> pd.DataFrame:
        if time_col not in self.df.columns:
            raise ValueError(f"Coluna de tempo '{time_col}' não está no DataFrame.")
        pred = self._predicted_pd_series()
        mask = pd.Series(True, index=self.df.index)
        if sample is not None and self.sample_col is not None:
            mask = self.df[self.sample_col] == sample
        base = pd.DataFrame({
            "periodo": self.df.loc[mask, time_col].values,
            "real": self.df.loc[mask, self.target].values,
            "prev": pred[mask].values,
        })
        g = base.groupby("periodo", dropna=False)
        out = g.agg(n=("real", "size"),
                    pd_prevista=("prev", "mean"),
                    pd_realizada=("real", "mean")).reset_index()
        out["gap"] = out["pd_realizada"] - out["pd_prevista"]
        out["status"] = out["gap"].abs().map(lambda d: "ok" if d <= tol else "alerta")
        for c in ("pd_prevista", "pd_realizada", "gap"):
            out[c] = out[c].round(4)
        out = out.sort_values("periodo").reset_index(drop=True)
        out.attrs.update(time_col=time_col, sample=sample, tol=tol)
        return out

    # ------------------------------------------------------------------
    # MONOTONICITY_REPORT: verifica se a PD cresce ao longo das folhas 1..N
    #   (posição esquerda→direita), em cada amostra — tanto na referência (DES)
    #   quanto nas demais (estabilidade). Lista as inversões (pares de notas
    #   consecutivas onde a PD cai).
    # ------------------------------------------------------------------
    def monotonicity_report(self) -> pd.DataFrame:
        lv = self.leaves()
        order = lv["segmento"].tolist()
        if self.sample_col is None:
            amostras = [None]
        else:
            todas = list(self.df[self.sample_col].dropna().unique())
            amostras = [self.ref_sample] + [a for a in todas if a != self.ref_sample]
        rows = []
        for a in amostras:
            vals = []
            for sid in order:
                m = self.segments[sid]["mask"]
                if a is not None:
                    m = m & (self.df[self.sample_col] == a)
                s = self.df.loc[m, self.target]
                vals.append(float(s.mean()) if len(s) else np.nan)
            inv = []
            for i in range(len(vals) - 1):
                v0, v1 = vals[i], vals[i + 1]
                if not (np.isnan(v0) or np.isnan(v1)) and v1 < v0:
                    inv.append((int(lv.iloc[i]["nota_pd"]), int(lv.iloc[i + 1]["nota_pd"])))
            rows.append({"amostra": a if a is not None else "todos",
                         "monotonico": len(inv) == 0,
                         "n_inversoes": len(inv), "inversoes": inv})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # CALIBRATION_TABLE: por folha, PD prevista (régua/DES) vs realizada na
    #   amostra de verificação (default = 1ª não-referência, ex.: OOT).
    # ------------------------------------------------------------------
    def calibration_table(self, check_sample: str | None = None) -> pd.DataFrame:
        if self.sample_col is not None and check_sample is None:
            nonref = [a for a in self.df[self.sample_col].dropna().unique()
                      if a != self.ref_sample]
            check_sample = nonref[0] if nonref else self.ref_sample
        lv = self.leaves()
        rows = []
        for _, r in lv.iterrows():
            sid = r["segmento"]
            prev_vals = self._leaf_target(sid)               # régua = DES (ou todos)
            prev = float(prev_vals.mean()) if len(prev_vals) else np.nan
            m = self.segments[sid]["mask"]
            if check_sample is not None and self.sample_col is not None:
                m = m & (self.df[self.sample_col] == check_sample)
            real_vals = self.df.loc[m, self.target]
            real = float(real_vals.mean()) if len(real_vals) else np.nan
            rows.append({
                "nota_pd": int(r["nota_pd"]), "descricao": r["descricao"],
                "n": int(len(real_vals)),
                "pd_prevista": round(prev, 4) if not np.isnan(prev) else np.nan,
                "pd_realizada": round(real, 4) if not np.isnan(real) else np.nan,
                "gap": (round(real - prev, 4)
                        if not (np.isnan(real) or np.isnan(prev)) else np.nan),
            })
        out = pd.DataFrame(rows)
        out.attrs["check_sample"] = check_sample
        return out

    # ------------------------------------------------------------------
    # PLOT_CALIBRATION: dispersão PD prevista (x) vs realizada (y) por folha,
    #   com a diagonal y=x (calibração perfeita) e faixa de tolerância.
    # ------------------------------------------------------------------
    def plot_calibration(self, check_sample: str | None = None, tol: float = 0.02,
                         figsize=(6.0, 6.0), save_path: str | None = None,
                         dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_calibration requer matplotlib.") from e
        ct = self.calibration_table(check_sample)
        chk = ct.attrs.get("check_sample")
        fig, ax = self._new_ax(figsize, dpi, ax)
        x = ct["pd_prevista"].to_numpy(dtype="float64")
        y = ct["pd_realizada"].to_numpy(dtype="float64")
        lim_hi = float(np.nanmax([np.nanmax(x) if len(x) else 0,
                                  np.nanmax(y) if len(y) else 0, 0.05])) * 1.15
        diag = np.linspace(0, lim_hi, 50)
        ax.fill_between(diag, diag - tol, diag + tol, color="#9bb7c9", alpha=0.25,
                        label=f"tolerância ±{tol}")
        ax.plot(diag, diag, color="#0f3d57", lw=1.3, label="calibração perfeita")
        for _, r in ct.iterrows():
            if np.isnan(r["pd_prevista"]) or np.isnan(r["pd_realizada"]):
                continue
            ok = abs(r["gap"]) <= tol
            ax.scatter(r["pd_prevista"], r["pd_realizada"], s=70,
                       color=("#1aa64b" if ok else "#d6453e"),
                       edgecolor="#33424f", zorder=3)
            ax.annotate(str(r["nota_pd"]),
                        (r["pd_prevista"], r["pd_realizada"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_xlim(0, lim_hi)
        ax.set_ylim(0, lim_hi)
        ax.set_xlabel(f"PD prevista (régua · {self.ref_sample if self.sample_col else 'todos'})")
        ax.set_ylabel(f"PD realizada ({chk if chk else 'todos'})")
        ax.set_title("Calibração da régua de PD por folha", fontsize=12,
                     fontweight="bold", color="#15324a")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # VALIDATION_REPORT: gera um relatório de validação em Markdown reunindo
    #   árvore (imagem), folhas, monotonicidade, PSI, CSI, métricas, calibração
    #   (tabela + imagem) e backtest por safra (se `time_col`). As imagens são
    #   salvas ao lado do .md. Documento único de governança (CMN 4.966/IFRS 9).
    # ------------------------------------------------------------------
    @staticmethod
    def _df_to_md(df: pd.DataFrame) -> str:
        def cell(v):
            try:
                if pd.isna(v):                  # cobre None, NaN, pd.NA, pd.NaT
                    return "—"
            except (TypeError, ValueError):
                pass
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        cols = list(df.columns)
        linhas = ["| " + " | ".join(str(c) for c in cols) + " |",
                  "| " + " | ".join("---" for _ in cols) + " |"]
        for _, r in df.iterrows():
            linhas.append("| " + " | ".join(cell(r[c]) for c in cols) + " |")
        return "\n".join(linhas)

    def validation_report(self, path: str = "relatorio_validacao_pd.md",
                          time_col: str | None = None,
                          title: str = "Relatório de Validação — Segmentação de PD",
                          tol_backtest: float = 0.03, tol_calib: float = 0.02,
                          stamp: str | None = None) -> str:
        import os

        base = os.path.dirname(os.path.abspath(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        img_arvore = f"{stem}_arvore.png"
        img_calib = f"{stem}_calibracao.png"

        import matplotlib.pyplot as plt
        try:
            fig = self.plot_tree(save_path=os.path.join(base, img_arvore))
            plt.close(fig)
            tem_arvore = True
        except Exception:
            plt.close("all")                    # não vaza figura se o plot falhar
            tem_arvore = False
        tem_calib = False
        if self.sample_col is not None:
            try:
                fig = self.plot_calibration(save_path=os.path.join(base, img_calib),
                                            tol=tol_calib)
                plt.close(fig)
                tem_calib = True
            except Exception:
                plt.close("all")
                tem_calib = False

        regua = self._regua_dict()
        feats = self.regua_features()
        n_folhas = len(regua["leaves"])
        prof = max(s["depth"] for s in self.segments.values())

        L = [f"# {title}", ""]
        if stamp:
            L.append(f"_Gerado em {stamp}._\n")
        L += ["## Visão geral", "",
              f"- **Target:** `{self.target}`",
              f"- **Amostra de referência:** `{self.ref_sample}`"
              + ("" if self.sample_col is None else f" (coluna `{self.sample_col}`)"),
              f"- **Folhas:** {n_folhas} · **profundidade máxima:** {prof}",
              f"- **Variáveis usadas:** {', '.join(feats) or '(nenhuma)'}",
              f"- **Linhas:** {len(self.df):,}".replace(",", "."), ""]

        if tem_arvore:
            L += ["## Árvore de segmentação", "", f"![arvore]({img_arvore})", ""]

        L += ["## Folhas", "",
              self._df_to_md(self.leaves(with_psi=(self.sample_col is not None))), ""]

        mr = self.monotonicity_report()
        ok = bool(mr["monotonico"].all())
        L += ["## Monotonicidade da PD por nota", "",
              ("✅ PD monotônica crescente em todas as amostras."
               if ok else "⚠️ Há inversões de monotonicidade — ver tabela."),
              "", self._df_to_md(mr[["amostra", "monotonico", "n_inversoes"]]), ""]

        if self.sample_col is not None:
            try:
                L += ["## PSI (estabilidade da segmentação)", "",
                      self._df_to_md(self.psi()), ""]
            except Exception:
                pass
            try:
                L += ["## CSI por variável (estabilidade das entradas)", "",
                      self._df_to_md(self.csi()), ""]
            except Exception:
                pass

        try:
            L += ["## Discriminação (régua como modelo de PD)", "",
                  self._df_to_md(self.metrics()), ""]
        except Exception:
            pass

        if self.sample_col is not None:
            ct = self.calibration_table()
            L += [f"## Calibração (prevista DES × realizada {ct.attrs.get('check_sample')})",
                  "", self._df_to_md(ct[["nota_pd", "n", "pd_prevista",
                                         "pd_realizada", "gap"]]), ""]
            if tem_calib:
                L += [f"![calibracao]({img_calib})", ""]

        if time_col is not None:
            try:
                bt = self.backtest(time_col, tol=tol_backtest)
                L += ["## Backtest por safra (prevista × realizada no tempo)", "",
                      self._df_to_md(bt), ""]
            except Exception as e:
                L += ["## Backtest por safra", "", f"_não gerado: {e}_", ""]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(L))
        return path

    # ==================================================================
    # QUALIDADE DOS SEGMENTOS / DISCRIMINAÇÃO: gráficos
    # ==================================================================
    @staticmethod
    def _new_ax(figsize, dpi, ax):
        # Figura SEM pyplot (não entra no Gcf): evita o backend inline
        # re-exibir o gráfico (duplicação) além do display explícito do widget.
        if ax is not None:
            return ax.figure, ax
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(figsize=figsize, dpi=dpi)
        FigureCanvasAgg(fig)
        return fig, fig.subplots()

    @staticmethod
    def _wilson_ci(k, n, z=1.96):
        """Intervalo de Wilson para uma proporção (taxa de default por folha)."""
        if n == 0:
            return (np.nan, np.nan, np.nan)
        p = k / n
        denom = 1 + z * z / n
        centro = (p + z * z / (2 * n)) / denom
        meio = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
        return p, max(0.0, centro - meio), min(1.0, centro + meio)

    def _samples_with_both_classes(self):
        """Amostras (referência primeiro) que têm as DUAS classes — usáveis em
        ROC/KS. Sem sample_col, devolve [(None, máscara total)]."""
        if self.sample_col is None:
            y = self.df[self.target].to_numpy(dtype="float64")
            y = y[~np.isnan(y)]
            return [(None, pd.Series(True, index=self.df.index))] if np.unique(y).size >= 2 else []
        todas = list(self.df[self.sample_col].dropna().unique())
        ordem = [self.ref_sample] + [a for a in todas if a != self.ref_sample]
        out = []
        for a in ordem:
            mask = self.df[self.sample_col] == a
            y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
            y = y[~np.isnan(y)]
            if np.unique(y).size >= 2:
                out.append((a, mask))
        return out

    # ------------------------------------------------------------------
    # PLOT_ROC: curva ROC da régua (score = PD prevista) por amostra, com a AUC
    #   na legenda. Mede o poder de ORDENAÇÃO de risco da segmentação.
    # ------------------------------------------------------------------
    def plot_roc(self, samples: list | None = None, figsize=(6.0, 6.0),
                 save_path: str | None = None, dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
            from sklearn.metrics import roc_auc_score, roc_curve
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_roc requer matplotlib e scikit-learn.") from e
        pred = self._predicted_pd_series()
        grupos = self._samples_with_both_classes()
        if samples is not None:
            grupos = [(a, m) for a, m in grupos if a in samples]
        fig, ax = self._new_ax(figsize, dpi, ax)
        ax.plot([0, 1], [0, 1], color="#9aa7b2", lw=1.0, ls="--")
        cores = ["#0f3d57", "#d6453e", "#1aa64b", "#caa000", "#6b3fa0", "#2a9d8f"]
        if not grupos:
            ax.text(0.5, 0.5, "sem as duas classes para a curva ROC", ha="center",
                    va="center", transform=ax.transAxes, color="#889")
        for i, (a, mask) in enumerate(grupos):
            y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
            s = pred[mask.values].to_numpy(dtype="float64")
            ok = ~(np.isnan(y) | np.isnan(s))
            y, s = y[ok], s[ok]
            if np.unique(y).size < 2:
                continue
            fpr, tpr, _ = roc_curve(y, s)
            auc = roc_auc_score(y, s)
            nome = a if a is not None else "todos"
            ax.plot(fpr, tpr, color=cores[i % len(cores)], lw=2.0,
                    label=f"{nome} · AUC {auc:.3f} · Gini {2*auc-1:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Falso positivo (1 − especificidade)")
        ax.set_ylabel("Verdadeiro positivo (sensibilidade)")
        ax.set_title("Curva ROC da régua de PD", fontsize=12, fontweight="bold",
                     color="#15324a")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # PLOT_KS: curva KS — distribuições acumuladas de bons (alvo 0) e maus
    #   (alvo 1) ao longo do score (PD prevista), com a distância KS marcada.
    # ------------------------------------------------------------------
    def plot_ks(self, sample: str | None = None, figsize=(7.0, 4.6),
                save_path: str | None = None, dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_ks requer matplotlib.") from e
        pred = self._predicted_pd_series()
        if sample is None and self.sample_col is not None:
            sample = self.ref_sample
        mask = pd.Series(True, index=self.df.index)
        if sample is not None and self.sample_col is not None:
            mask = self.df[self.sample_col] == sample
        y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
        s = pred[mask.values].to_numpy(dtype="float64")
        ok = ~(np.isnan(y) | np.isnan(s))
        y, s = y[ok], s[ok]
        fig, ax = self._new_ax(figsize, dpi, ax)
        if np.unique(y).size < 2:
            ax.text(0.5, 0.5, "sem as duas classes para o KS", ha="center",
                    va="center", transform=ax.transAxes, color="#889")
            ax.axis("off"); fig.tight_layout()
            return fig
        order = np.argsort(s, kind="mergesort")
        s_ord, y_ord = s[order], y[order]
        tot_bad = y_ord.sum()
        tot_good = (1 - y_ord).sum()
        cum_bad = np.cumsum(y_ord) / tot_bad
        cum_good = np.cumsum(1 - y_ord) / tot_good
        diff = cum_good - cum_bad           # bons acumulam mais rápido nos scores baixos
        j = int(np.argmax(diff))
        ks = float(diff[j])
        x = s_ord
        ax.plot(x, cum_good, color="#1aa64b", lw=2.0, label="acum. bons (alvo 0)")
        ax.plot(x, cum_bad, color="#d6453e", lw=2.0, label="acum. maus (alvo 1)")
        ax.vlines(x[j], cum_bad[j], cum_good[j], color="#0f3d57", lw=1.8, ls="--")
        ax.annotate(f"KS = {ks:.3f}", (x[j], (cum_good[j] + cum_bad[j]) / 2),
                    textcoords="offset points", xytext=(8, 0), fontsize=10,
                    fontweight="bold", color="#0f3d57")
        sfx = f" · {sample}" if sample else ""
        ax.set_xlabel("score (PD prevista)")
        ax.set_ylabel("proporção acumulada")
        ax.set_title(f"Curva KS da régua de PD{sfx}", fontsize=12, fontweight="bold",
                     color="#15324a")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.2)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # PLOT_LEAF_BADRATE: taxa de default (PD) por folha (barras na ordem da
    #   nota), com IC de Wilson — mostra a separação de risco entre folhas e a
    #   incerteza amostral de cada PD. Substitui o boxplot (alvo binário).
    # ------------------------------------------------------------------
    def plot_leaf_badrate(self, sample: str | None = None, ascending: bool = True,
                          cmap: str = "RdYlGn_r", figsize=None,
                          save_path: str | None = None, dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_leaf_badrate requer matplotlib.") from e
        if sample is None and self.sample_col is not None:
            sample = self.ref_sample
        lv = self.leaves(ascending=ascending)
        notas, rates, los, his = [], [], [], []
        for _, r in lv.iterrows():
            sid = r["segmento"]
            m = self.segments[sid]["mask"]
            if sample is not None and self.sample_col is not None:
                m = m & (self.df[self.sample_col] == sample)
            v = self.df.loc[m, self.target].to_numpy(dtype="float64")
            v = v[~np.isnan(v)]
            if len(v) == 0:
                continue
            p, lo, hi = self._wilson_ci(int(v.sum()), len(v))
            notas.append(int(r["nota_pd"])); rates.append(p); los.append(lo); his.append(hi)
        if not rates:
            raise ValueError("Sem dados para a taxa de default por folha.")
        if figsize is None:
            figsize = (max(6.0, len(rates) * 1.1), 4.6)
        fig, ax = self._new_ax(figsize, dpi, ax)
        vmax = max(his) * 1.1 if his else 0.1
        norm = Normalize(0.0, vmax if vmax > 1e-9 else 0.01)
        cmap_obj = plt.get_cmap(cmap)
        xs = list(range(1, len(rates) + 1))
        yerr = [[p - lo for p, lo in zip(rates, los)],
                [hi - p for p, hi in zip(rates, his)]]
        bars = ax.bar(xs, rates, width=0.62, color=[cmap_obj(norm(p)) for p in rates],
                      edgecolor="#33424f", alpha=0.92,
                      yerr=yerr, capsize=4, error_kw=dict(ecolor="#33424f", lw=1.1))
        for x, p in zip(xs, rates):
            ax.text(x, p, f"{p:.3f}", ha="center", va="bottom", fontsize=8.5,
                    color="#15324a", fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"folha {n}" for n in notas], rotation=0, fontsize=9)
        ax.set_ylabel("Taxa de default (PD)")
        sfx = f" · {sample}" if sample else ""
        ax.set_title(f"Taxa de default por folha{sfx} (IC de Wilson 95%)", fontsize=12,
                     fontweight="bold", color="#15324a")
        ax.set_ylim(0, vmax if vmax > 1e-9 else 0.05)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # PLOT_SCORE_DISTRIBUTION: distribuição do score (PD prevista) separada por
    #   classe (bons × maus) — quanto mais separadas as massas, melhor o KS/AUC.
    # ------------------------------------------------------------------
    def plot_score_distribution(self, sample: str | None = None, bins: int = 30,
                                figsize=(7.0, 4.2), save_path: str | None = None,
                                dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_score_distribution requer matplotlib.") from e
        pred = self._predicted_pd_series()
        if sample is None and self.sample_col is not None:
            sample = self.ref_sample
        mask = pd.Series(True, index=self.df.index)
        sfx = ""
        if sample is not None and self.sample_col is not None:
            mask = self.df[self.sample_col] == sample
            sfx = f" — {sample}"
        y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
        s = pred[mask.values].to_numpy(dtype="float64")
        ok = ~(np.isnan(y) | np.isnan(s))
        y, s = y[ok], s[ok]
        fig, ax = self._new_ax(figsize, dpi, ax)
        if s.size == 0:
            ax.text(0.5, 0.5, "sem score para exibir", ha="center", va="center",
                    transform=ax.transAxes, color="#889")
            ax.axis("off"); fig.tight_layout()
            return fig
        rng = (float(np.min(s)), float(np.max(s)) if np.max(s) > np.min(s)
               else float(np.min(s)) + 0.01)
        good, bad = s[y == 0], s[y == 1]
        ax.hist(good, bins=bins, range=rng, color="#1aa64b", alpha=0.55,
                edgecolor="#157a52", label=f"bons (n={good.size})", density=True)
        ax.hist(bad, bins=bins, range=rng, color="#d6453e", alpha=0.55,
                edgecolor="#b23a2a", label=f"maus (n={bad.size})", density=True)
        ax.set_xlabel("score (PD prevista)")
        ax.set_ylabel("densidade")
        ax.set_title(f"Distribuição do score por classe{sfx}", fontsize=12,
                     fontweight="bold", color="#15324a")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # PLOT_FEATURE_PD: preview da variável candidata DENTRO de uma folha —
    #   taxa de default (PD) por faixa da variável (com a representatividade),
    #   nos mesmos bins do split. Mostra a FORMA da relação antes de dividir e
    #   se é monotônica. Use antes do `grow` para escolher a variável/corte.
    # ------------------------------------------------------------------
    def plot_feature_pd(self, feature: str, sid: str | None = None, splits=None,
                        dtype=None, max_n_bins: int = 6, min_bin_size: float = 0.05,
                        max_bin_size=None, figsize=None, save_path: str | None = None,
                        dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_feature_pd requer matplotlib.") from e
        if sid is None or sid not in self.segments:
            sid, sub = "root", self.df
        else:
            sub = self.df[self.segments[sid]["mask"]]
        # `splits` (mesmos do preview/grow) garante que o gráfico bata com a tabela
        bins, modo, kind = self._resolve_bins(sub, feature, splits, dtype,
                                              max_n_bins, min_bin_size, max_bin_size)
        if not bins:
            raise ValueError(f"Sem faixas válidas para '{feature}' nesta folha.")
        tbl = self._bin_table(sub, feature, bins, len(self.df))
        if tbl.empty:
            raise ValueError(f"Sem dados de '{feature}' nesta folha.")
        labels = [s.split(": ", 1)[-1] for s in tbl["faixa"]]
        pds = tbl["pd_medio"].to_numpy()
        reprs = tbl["repr_%"].to_numpy()
        if figsize is None:
            figsize = (max(6.0, len(labels) * 1.25), 4.6)
        fig, ax = self._new_ax(figsize, dpi, ax)
        xs = list(range(len(labels)))

        # BARRAS = representatividade/volumetria (%), eixo esquerdo (steelblue)
        # LINHA  = taxa de default (PD), eixo direito (crimson) — tema padrão da lib
        col_bar, col_line = "steelblue", "crimson"
        bars = ax.bar(xs, reprs, color=col_bar, edgecolor="#2f5d82", alpha=0.85,
                      width=0.62, label="Representatividade (%)")
        ax.set_ylabel("Representatividade (%)", color=col_bar, fontweight="bold")
        ax.tick_params(axis="y", labelcolor=col_bar)
        rmax = float(np.nanmax(reprs)) or 1.0
        ax.set_ylim(0, rmax * 1.25 + 1)
        for x, rp in zip(xs, reprs):
            dentro = rp > rmax * 0.18          # barra alta: rótulo branco dentro
            ax.text(x, rp * 0.5 if dentro else rp + rmax * 0.02, f"{rp:.1f}%",
                    ha="center", va="center" if dentro else "bottom", fontsize=8,
                    color="white" if dentro else col_bar, fontweight="bold")

        # LINHA = PD média por faixa, eixo da direita (vermelho)
        ax2 = ax.twinx()
        line, = ax2.plot(xs, pds, color=col_line, marker="o", lw=2.2,
                         markersize=7, markeredgecolor="#fff", zorder=5,
                         label="Taxa de default (PD)")
        ax2.set_ylabel("Taxa de default (PD)", color=col_line, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor=col_line)
        pmax = float(np.nanmax(pds)) if pds.size else 0.0
        ax2.set_ylim(0, max(pmax * 1.3 + 0.02, 0.02))
        for x, v in zip(xs, pds):
            ax2.text(x, v + max(pmax, 0.01) * 0.04, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8, color=col_line,
                     fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
        rot = self.feature_labels.get(feature, feature)
        mono = "monotônico" if tbl.attrs.get("mono_ok") else "NÃO monotônico"
        ax.set_title(f"'{rot}': representatividade (barra) × PD (linha) — "
                     f"{modo}, {mono}", fontsize=11.5, fontweight="bold", color="#15324a")
        ax.legend(handles=[bars, line], loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # PLOT_FEATURE_HIST: histograma de uma variável NUMÉRICA dentro de uma
    #   folha, com linhas verticais nos cortes propostos (ótimo ou manual).
    #   Mostra a forma da distribuição da variável antes de dividir. Para
    #   variável categórica cai no plot_feature_pd (barras × PD).
    # ------------------------------------------------------------------
    def plot_feature_hist(self, feature: str, sid: str | None = None, splits=None,
                          dtype=None, max_n_bins: int = 6, min_bin_size: float = 0.05,
                          max_bin_size=None, bins_hist: int = 30, figsize=None,
                          save_path: str | None = None, dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_feature_hist requer matplotlib.") from e
        if sid is None or sid not in self.segments:
            sid, sub = "root", self.df
        else:
            sub = self.df[self.segments[sid]["mask"]]
        kind = self._detect_kind(sub, feature, dtype)
        if kind != "num":
            # categórica não tem histograma — barras de representatividade × PD
            return self.plot_feature_pd(feature, sid=sid, splits=splits, dtype=dtype,
                                        max_n_bins=max_n_bins, min_bin_size=min_bin_size,
                                        max_bin_size=max_bin_size, figsize=figsize,
                                        save_path=save_path, dpi=dpi, ax=ax)
        x = sub[feature].to_numpy(dtype="float64")
        x = x[~np.isnan(x)]
        if x.size == 0:
            raise ValueError(f"Sem valores numéricos de '{feature}' nesta folha.")
        # cortes (linhas verticais) a partir dos bins resolvidos
        cortes, modo = [], "—"
        try:
            bins, modo, _ = self._resolve_bins(sub, feature, splits, dtype,
                                               max_n_bins, min_bin_size, max_bin_size)
            cortes = sorted({e for b in bins if b["kind"] == "num"
                             for e in (b["lo"], b["hi"]) if np.isfinite(e)})
        except Exception:
            pass
        if figsize is None:
            figsize = (7.0, 4.4)
        fig, ax = self._new_ax(figsize, dpi, ax)
        # barras = volumetria da variável (steelblue) · cortes = crimson
        ax.hist(x, bins=bins_hist, color="steelblue", alpha=0.85, edgecolor="#2f5d82")
        for c in cortes:
            ax.axvline(c, color="crimson", lw=1.8, ls="--")
        if cortes:
            ax.plot([], [], color="crimson", lw=1.8, ls="--",
                    label=f"cortes propostos ({len(cortes)})")
            ax.legend(fontsize=9, framealpha=0.9)
        rot = self.feature_labels.get(feature, feature)
        ax.set_xlabel(rot)
        ax.set_ylabel("frequência")
        ax.set_title(f"Distribuição de '{rot}' na folha — {modo}", fontsize=11.5,
                     fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # PLOT_LEAF_TARGET_HIST: PD da folha (referência DES por padrão) como uma
    #   barra horizontal com IC de Wilson, comparada à PD da carteira — leitura
    #   rápida do nível de risco da folha. (Alvo binário não tem histograma.)
    # ------------------------------------------------------------------
    def plot_leaf_target_hist(self, sid: str | None = None, sample: str | None = None,
                              bins: int = 24, figsize=(5.2, 2.0),
                              save_path: str | None = None, dpi: int = 150, ax=None):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_leaf_target_hist requer matplotlib.") from e
        mask = (pd.Series(True, index=self.df.index)
                if sid is None or sid not in self.segments
                else self.segments[sid]["mask"].copy())
        if sample is None and self.sample_col is not None:
            sample = self.ref_sample
        sfx = ""
        if sample is not None and self.sample_col is not None:
            mask = mask & (self.df[self.sample_col] == sample)
            sfx = f" · {sample}"
        y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
        y = y[~np.isnan(y)]
        fig, ax = self._new_ax(figsize, dpi, ax)
        if y.size == 0:
            ax.text(0.5, 0.5, "sem alvo nesta folha/amostra", ha="center", va="center",
                    transform=ax.transAxes, color="#889")
            ax.axis("off"); fig.tight_layout()
            return fig
        p, lo, hi = self._wilson_ci(int(y.sum()), len(y))
        # PD da carteira (mesma amostra) como referência
        cart = self.df
        if sample is not None and self.sample_col is not None:
            cart = self.df[self.df[self.sample_col] == sample]
        yc = cart[self.target].to_numpy(dtype="float64")
        yc = yc[~np.isnan(yc)]
        pc = float(np.mean(yc)) if yc.size else np.nan
        xmax = max(hi, pc if not np.isnan(pc) else 0, 0.02) * 1.25
        ax.barh([0], [p], height=0.5, color="#d6453e" if (not np.isnan(pc) and p > pc)
                else "#1aa64b", alpha=0.85, edgecolor="#33424f",
                xerr=[[p - lo], [hi - p]], capsize=5, error_kw=dict(ecolor="#33424f", lw=1.1))
        ax.text(p, 0, f"  PD {p:.3f}", va="center", ha="left", fontsize=10,
                fontweight="bold", color="#15324a")
        if not np.isnan(pc):
            ax.axvline(pc, color="#0f3d57", lw=1.6, ls="--",
                       label=f"PD carteira {pc:.3f}")
            ax.legend(fontsize=8, loc="lower right")
        ax.set_yticks([])
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Taxa de default (PD)")
        ax.set_title(f"PD da folha{sfx} (n={y.size}; maus={int(y.sum())})",
                     fontsize=10.5, fontweight="bold", color="#15324a")
        ax.grid(axis="x", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ==================================================================
    # ANÁLISE DE VARIÁVEIS: perfil estatístico, distribuição e estabilidade
    #   (PSI atual e por safra) de UMA variável de entrada, numa folha.
    # ==================================================================
    def _leaf_mask(self, sid):
        if sid is None or sid not in self.segments:
            return pd.Series(True, index=self.df.index)
        return self.segments[sid]["mask"].copy()

    def variable_summary(self, feature, sid=None, sample=None) -> dict:
        """Resumo de uma variável numa folha (referência DES por padrão):
        %missing, média/mediana/desvio, n, percentis (min, p5, p95, max), o IV
        binário e o PSI atual por amostra (nos bins do IV)."""
        sid = sid if (sid in self.segments) else "root"
        mask = self._leaf_mask(sid)
        if sample is None and self.sample_col is not None:
            sample = self.ref_sample
        if sample is not None and self.sample_col is not None:
            mask = mask & (self.df[self.sample_col] == sample)
        col = self.df.loc[mask, feature]
        kind = self._detect_kind(self.df[mask], feature, None)
        n = int(len(col)); n_miss = int(col.isna().sum())
        res = {"variavel": feature, "tipo": kind, "amostra": sample, "n": n,
               "n_missing": n_miss,
               "pct_missing": round(100 * n_miss / n, 2) if n else float("nan")}
        if kind == "num":
            x = col.to_numpy(dtype="float64"); x = x[~np.isnan(x)]
            if x.size:
                res.update({
                    "media": round(float(np.mean(x)), 4),
                    "mediana": round(float(np.median(x)), 4),
                    "desvio": round(float(np.std(x, ddof=1)) if x.size > 1 else 0.0, 4),
                    "min": round(float(np.min(x)), 4),
                    "p5": round(float(np.percentile(x, 5)), 4),
                    "p95": round(float(np.percentile(x, 95)), 4),
                    "max": round(float(np.max(x)), 4)})
        else:
            vc = col.dropna().astype(str).value_counts(normalize=True)
            res["top_categorias"] = [(c, round(100 * p, 1)) for c, p in vc.head(8).items()]
        # IV binário + PSI atual (nos mesmos bins) desta variável na folha
        res["iv"], res["forca"], res["psi"], res["pior_psi"] = None, "—", {}, None
        try:
            ivt = self.variable_iv(sid, features=[feature], with_psi=True)
            if len(ivt):
                r0 = ivt.iloc[0]
                res["iv"] = None if pd.isna(r0["iv"]) else float(r0["iv"])
                res["forca"] = r0["forca"]
                for c in ivt.columns:
                    if c.startswith("psi_") and c != "psi_classificacao":
                        res["psi"][c[4:]] = None if pd.isna(r0[c]) else float(r0[c])
                if "pior_psi" in ivt.columns and not pd.isna(r0["pior_psi"]):
                    res["pior_psi"] = float(r0["pior_psi"])
        except Exception:
            pass
        return res

    def variable_by_safra(self, feature, time_col=None, sid=None, sample=None) -> pd.DataFrame:
        """Percentis (min, p5, média, p95, max) e %missing da variável NUMÉRICA
        por safra (mês de `time_col`), dentro da folha. `time_col` default = date_col."""
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col no segmenter.")
        mask = self._leaf_mask(sid)
        if sample is not None and self.sample_col is not None:
            mask = mask & (self.df[self.sample_col] == sample)
        sub = self.df[mask]
        if time_col not in sub.columns:
            raise ValueError(f"Coluna de tempo '{time_col}' não existe no DataFrame.")
        safra = pd.to_datetime(sub[time_col], errors="coerce").dt.to_period("M")
        rows = []
        for per, g in sub.groupby(safra):
            col = g[feature]
            x = col.to_numpy(dtype="float64"); x = x[~np.isnan(x)]
            n = int(len(col)); n_miss = int(col.isna().sum())
            row = {"safra": str(per), "n": n,
                   "pct_missing": round(100 * n_miss / n, 1) if n else float("nan")}
            if x.size:
                row.update({"min": round(float(np.min(x)), 3),
                            "p5": round(float(np.percentile(x, 5)), 3),
                            "media": round(float(np.mean(x)), 3),
                            "p95": round(float(np.percentile(x, 95)), 3),
                            "max": round(float(np.max(x)), 3)})
            else:
                row.update({k: float("nan") for k in ("min", "p5", "media", "p95", "max")})
            rows.append(row)
        return pd.DataFrame(rows).sort_values("safra").reset_index(drop=True)

    def variable_psi_by_safra(self, feature, time_col=None, sid=None, max_n_bins=10,
                              min_bin_size=0.05, eps=1e-6) -> pd.DataFrame:
        """PSI da variável por safra vs. a referência (DES), com os bins
        fixados na DES da folha. `time_col` default = date_col."""
        if self.sample_col is None:
            raise ValueError("PSI por safra requer sample_col (referência DES).")
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col no segmenter.")
        mask = self._leaf_mask(sid)
        leaf = self.df[mask]
        if time_col not in leaf.columns:
            raise ValueError(f"Coluna de tempo '{time_col}' não existe no DataFrame.")
        ref = leaf[leaf[self.sample_col] == self.ref_sample]
        if len(ref) == 0:
            raise ValueError("Sem dados da referência (DES) nesta folha.")
        bins, _modo, _kind = self._resolve_bins(leaf, feature, None, None,
                                                max_n_bins, min_bin_size)
        if not bins:
            return pd.DataFrame(columns=["safra", "n", "psi", "classificacao"])
        n_ref = len(ref)
        ref_pct = [max(int(self._mask_in(ref, feature, b).sum()) / n_ref, eps) for b in bins]
        safra = pd.to_datetime(leaf[time_col], errors="coerce").dt.to_period("M")
        rows = []
        for per, g in leaf.groupby(safra):
            n_g = len(g)
            if n_g == 0:
                continue
            psi = 0.0
            for b, p_ref in zip(bins, ref_pct):
                p_cur = max(int(self._mask_in(g, feature, b).sum()) / n_g, eps)
                psi += (p_cur - p_ref) * np.log(p_cur / p_ref)
            rows.append({"safra": str(per), "n": n_g, "psi": round(float(psi), 4),
                         "classificacao": _classifica_psi(psi)})
        return pd.DataFrame(rows).sort_values("safra").reset_index(drop=True)

    def plot_variable_distribution(self, feature, sid=None, sample=None, max_n_bins=8,
                                   min_bin_size=0.03, figsize=(7.2, 3.1),
                                   save_path=None, dpi=150, ax=None):
        """Distribuição da variável na folha (barras por faixa; a barra de
        faltantes aparece em destaque). Numérica usa os bins ótimos."""
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_variable_distribution requer matplotlib.") from e
        mask = self._leaf_mask(sid)
        if sample is None and self.sample_col is not None:
            sample = self.ref_sample
        if sample is not None and self.sample_col is not None:
            mask = mask & (self.df[self.sample_col] == sample)
        sub = self.df[mask]
        rot = self.feature_labels.get(feature, feature)
        fig, ax = self._new_ax(figsize, dpi, ax)
        try:
            bins, _modo, _kind = self._resolve_bins(sub, feature, None, None,
                                                    max_n_bins, min_bin_size)
        except Exception:
            bins = []
        if bins:
            tbl = self._bin_table(sub, feature, bins, max(len(sub), 1))
        else:
            tbl = pd.DataFrame()
        if tbl.empty:                         # fallback: histograma cru
            x = sub[feature].to_numpy(dtype="float64"); x = x[~np.isnan(x)]
            if x.size:
                ax.hist(x, bins=20, color="steelblue", alpha=0.85, edgecolor="#2f5d82")
        else:
            labels = [s.split(": ", 1)[-1] for s in tbl["faixa"]]
            reprs = tbl["repr_%"].to_numpy()
            cols = ["#c98a8a" if "faltante" in f else "steelblue" for f in tbl["faixa"]]
            xs = list(range(len(labels)))
            ax.bar(xs, reprs, color=cols, edgecolor="#2f5d82", alpha=0.9, width=0.72)
            for x0, rp in zip(xs, reprs):
                ax.text(x0, rp, f"{rp:.0f}%", ha="center", va="bottom", fontsize=7.5,
                        color="#15324a")
            ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
            ax.set_xlim(-0.75, len(labels) - 0.25)               # respiro nas bordas (eixo x)
            ax.set_ylim(0, float(np.nanmax(reprs)) * 1.16 + 1)   # espaço p/ os rótulos %
        ax.set_ylabel("% da folha")
        ax.set_title(f"Distribuição de '{rot}'" + (f" · {sample}" if sample else ""),
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def variable_share_by_safra(self, feature, time_col=None, sid=None, sample=None,
                                top=8) -> pd.DataFrame:
        """Representatividade (% por safra) de cada categoria de uma variável
        CATEGÓRICA ao longo do tempo. Linhas = safra; colunas = categorias (as
        `top` mais frequentes; o resto vira 'outras') + '(faltante)'."""
        time_col = time_col or self.date_col
        if time_col is None:
            raise ValueError("Informe time_col ou configure date_col no segmenter.")
        mask = self._leaf_mask(sid)
        if sample is not None and self.sample_col is not None:
            mask = mask & (self.df[self.sample_col] == sample)
        sub = self.df[mask]
        if time_col not in sub.columns:
            raise ValueError(f"Coluna de tempo '{time_col}' não existe no DataFrame.")
        safra = pd.to_datetime(sub[time_col], errors="coerce").dt.to_period("M").astype(str)
        cat = sub[feature]
        keep = list(cat.dropna().astype(str).value_counts().head(top).index)

        def lab(v):
            if pd.isna(v):
                return "(faltante)"
            s = str(v)
            return s if s in keep else "outras"

        tab = pd.crosstab(safra, cat.map(lab))
        tab = tab[tab.index != "NaT"]
        if tab.empty:
            return pd.DataFrame(columns=["safra"])
        pct = tab.div(tab.sum(axis=1), axis=0) * 100
        order = [c for c in keep if c in pct.columns]
        if "outras" in pct.columns:
            order.append("outras")
        if "(faltante)" in pct.columns:
            order.append("(faltante)")
        pct = pct[order].round(1).sort_index()
        pct.index.name = "safra"
        return pct.reset_index()

    def plot_variable_share_timeseries(self, feature, time_col=None, sid=None, sample=None,
                                       figsize=(8.6, 3.6), save_path=None, dpi=150, ax=None):
        """Área empilhada da representatividade (%) de cada categoria por safra —
        mostra como a distribuição da variável categórica migra no tempo."""
        import matplotlib.colors as mcolors
        sh = self.variable_share_by_safra(feature, time_col, sid=sid, sample=sample)
        fig, ax = self._new_ax(figsize, dpi, ax)
        cats = [c for c in sh.columns if c != "safra"]
        if sh.empty or not cats:
            ax.text(0.5, 0.5, "sem dados por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        x = list(range(len(sh)))
        cmap = mcolors.LinearSegmentedColormap.from_list("sc", ["steelblue", "crimson"])
        base = [c for c in cats if c not in ("outras", "(faltante)")]
        colors = []
        for c in cats:
            if c == "(faltante)":
                colors.append("#c98a8a")
            elif c == "outras":
                colors.append("#b9c0cb")
            else:
                i = base.index(c)
                colors.append(cmap(i / max(len(base) - 1, 1)))
        ys = [sh[c].fillna(0).to_numpy() for c in cats]
        ax.stackplot(x, ys, labels=cats, colors=colors, alpha=0.92)
        ax.set_ylim(0, 100); ax.margins(x=0)
        ax.set_xticks(x); ax.set_xticklabels(sh["safra"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% da safra")
        # legenda à DIREITA (vertical) — não colide com as datas do eixo x
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.9, title="categoria")
        rot = self.feature_labels.get(feature, feature)
        ax.set_title(f"'{rot}' ao longo do tempo — representatividade por categoria",
                     fontsize=11, fontweight="bold", color="#15324a")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_variable_timeseries(self, feature, time_col=None, sid=None, sample=None,
                                 figsize=(8.6, 3.4), save_path=None, dpi=150, ax=None):
        """Variável NUMÉRICA: percentis por safra (min–max, p5–p95, média).
        Variável CATEGÓRICA: representatividade de cada categoria por safra."""
        if self._detect_kind(self.df[self._leaf_mask(sid)], feature, None) == "cat":
            return self.plot_variable_share_timeseries(
                feature, time_col, sid=sid, sample=sample, figsize=(figsize[0], 3.6),
                save_path=save_path, dpi=dpi, ax=ax)
        bs = self.variable_by_safra(feature, time_col, sid=sid, sample=sample)
        fig, ax = self._new_ax(figsize, dpi, ax)
        if bs.empty or bs["media"].notna().sum() == 0:
            ax.text(0.5, 0.5, "sem dados por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        x = list(range(len(bs)))
        ax.fill_between(x, bs["min"], bs["max"], color="steelblue", alpha=0.07)
        ax.plot(x, bs["min"], color="#9bb7c9", lw=1.0)
        ax.plot(x, bs["max"], color="#9bb7c9", lw=1.0, label="min / max")
        ax.plot(x, bs["p5"], color="#6f93ad", lw=1.3, ls="--")
        ax.plot(x, bs["p95"], color="#6f93ad", lw=1.3, ls="--", label="p5 / p95")
        ax.plot(x, bs["media"], color="#15324a", lw=2.4, marker="o", markersize=4,
                label="média")
        ax.set_xticks(x); ax.set_xticklabels(bs["safra"], rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8, ncol=3, framealpha=0.9, loc="upper left")
        rot = self.feature_labels.get(feature, feature)
        ax.set_title(f"'{rot}' ao longo do tempo — percentis por safra",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(alpha=0.12)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_variable_psi_by_safra(self, feature, time_col=None, sid=None,
                                   figsize=(9.6, 3.4), save_path=None, dpi=150, ax=None):
        """PSI da variável por safra vs DES (barras coloridas por faixa de PSI)."""
        ps = self.variable_psi_by_safra(feature, time_col, sid=sid)
        fig, ax = self._new_ax(figsize, dpi, ax)
        if ps.empty:
            ax.text(0.5, 0.5, "sem PSI por safra", ha="center", va="center",
                    transform=ax.transAxes, color="#889"); ax.axis("off")
            fig.tight_layout(); return fig
        x = list(range(len(ps)))
        cor = ["#1aa64b" if p < 0.10 else "#caa000" if p < 0.25 else "#d6453e"
               for p in ps["psi"]]
        ax.bar(x, ps["psi"], color=cor, alpha=0.92, width=0.78)
        for x0, p in zip(x, ps["psi"]):
            ax.text(x0, p, f"{p:.2f}", ha="center", va="bottom", fontsize=7, color="#555")
        ax.axhline(0.10, color="#caa000", lw=0.8, ls="--")
        ax.axhline(0.25, color="#d6453e", lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(ps["safra"], rotation=45, ha="right", fontsize=8)
        ax.set_xlim(-0.7, len(ps) - 0.3)                          # respiro nas bordas (eixo x)
        ax.set_ylim(0, float(np.nanmax(ps["psi"])) * 1.16 + 0.05)  # espaço p/ os rótulos
        ax.set_ylabel("PSI")
        rot = self.feature_labels.get(feature, feature)
        ax.set_title(f"PSI de '{rot}' por safra vs DES (bins fixados na DES)",
                     fontsize=11, fontweight="bold", color="#15324a")
        ax.grid(axis="y", alpha=0.12)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # VARIABLE_IV: Information Value (WoE binário) de cada variável candidata em
    #   relação à folha `sid`, para indicar qual variável usar no próximo split.
    #   O alvo é binário, então o IV é o clássico do optimal binning:
    #   IV = Σ (dist_bons_i − dist_maus_i)·WoE_i, WoE_i = ln(dist_bons/dist_maus).
    #   Calculado na amostra de referência (DES), sobre os MESMOS bins ótimos do
    #   split. Quando há amostras, o PSI de cada variável é calculado sobre ESSES
    #   MESMOS bins (DES × cada amostra, dentro da folha): colunas
    #   ``psi_<amostra>``, ``pior_psi`` e ``psi_classificacao``.
    # ------------------------------------------------------------------
    def variable_iv(self, sid: str | None = None, features: list | None = None,
                    max_n_bins: int = 5, min_bin_size: float = 0.05,
                    cutoff="median", with_psi: bool = True) -> pd.DataFrame:
        if features is None:
            skip = {self.target, self.sample_col, self.date_col}
            features = [c for c in self.df.columns if c not in skip]

        if sid is None or sid == "root" or sid not in self.segments:
            leaf_mask = pd.Series(True, index=self.df.index)
        else:
            leaf_mask = self.segments[sid]["mask"]
        if self.sample_col is not None:
            ref_mask = self.df[self.sample_col] == self.ref_sample
            nonref = [a for a in self.df[self.sample_col].dropna().unique()
                      if a != self.ref_sample]
        else:
            ref_mask = pd.Series(True, index=self.df.index)
            nonref = []
        psi_on = with_psi and bool(nonref)

        leaf = self.df[leaf_mask]
        base = self.df[leaf_mask & ref_mask]            # porção DES da folha (referência)
        yb = base[self.target].to_numpy(dtype="float64")
        yb = yb[~np.isnan(yb)]
        ybin = yb.astype(int)
        tot_bad = int(ybin.sum())
        tot_good = int(ybin.size - tot_bad)
        # IV binário precisa das DUAS classes presentes na folha (DES)
        ok_base = ybin.size >= 4 and tot_bad > 0 and tot_good > 0
        pd_global = float(yb.mean()) if yb.size else np.nan
        n_base = int(yb.size)

        # frames e contagens por amostra DENTRO da folha (para o PSI nos bins do IV)
        ref_frame = cur_frames = None
        n_ref_leaf = 0
        if psi_on:
            ref_frame = self.df[leaf_mask & (self.df[self.sample_col] == self.ref_sample)]
            n_ref_leaf = len(ref_frame)
            cur_frames = {a: self.df[leaf_mask & (self.df[self.sample_col] == a)]
                          for a in nonref}

        rows = []
        for feat in features:
            iv, nb, kind = np.nan, 0, "—"
            psi_vals = {a: np.nan for a in nonref}
            if ok_base:
                try:
                    bins, _modo, kind = self._resolve_bins(
                        leaf, feat, None, None, max_n_bins, min_bin_size)
                    nb = len(bins)
                    if bins:
                        # IV binário (WoE): Σ (dist_bons − dist_maus)·ln(dist_bons/dist_maus)
                        iv = 0.0
                        yb_feat = base[self.target].to_numpy(dtype="float64")
                        for b in bins:
                            m = self._mask_in(base, feat, b).to_numpy()
                            yi = yb_feat[m]
                            yi = yi[~np.isnan(yi)].astype(int)
                            if yi.size == 0:
                                continue
                            bad_i = int(yi.sum()); good_i = int(yi.size - bad_i)
                            dg = good_i / tot_good
                            db = bad_i / tot_bad
                            if dg > 0 and db > 0:
                                iv += (dg - db) * np.log(dg / db)
                        # PSI sobre os MESMOS bins (DES × amostra, na folha)
                        if psi_on and n_ref_leaf > 0:
                            for a in nonref:
                                n_a = len(cur_frames[a])
                                if n_a == 0:
                                    continue
                                psi = 0.0
                                for b in bins:
                                    p_ref = max(int(self._mask_in(ref_frame, feat, b).sum())
                                                / n_ref_leaf, 1e-6)
                                    p_cur = max(int(self._mask_in(cur_frames[a], feat, b).sum())
                                                / n_a, 1e-6)
                                    psi += (p_cur - p_ref) * np.log(p_cur / p_ref)
                                psi_vals[a] = round(float(psi), 4)
                except Exception:
                    iv = np.nan
            row = {
                "variavel": feat, "tipo": kind, "n_bins": nb,
                "iv": round(float(iv), 4) if not (iv is None or np.isnan(iv)) else np.nan,
                "forca": _classifica_iv(iv),
            }
            if psi_on:
                for a in nonref:
                    row[f"psi_{a}"] = psi_vals[a]
                validos = [v for v in psi_vals.values() if not pd.isna(v)]
                pior = max(validos) if validos else np.nan
                row["pior_psi"] = round(float(pior), 4) if not pd.isna(pior) else np.nan
                row["psi_classificacao"] = _classifica_psi(pior) if not pd.isna(pior) else "—"
            rows.append(row)
        out = (pd.DataFrame(rows)
               .sort_values("iv", ascending=False, na_position="last")
               .reset_index(drop=True))
        out.attrs["pd_medio"] = round(pd_global, 4) if not np.isnan(pd_global) else np.nan
        out.attrs["cutoff"] = out.attrs["pd_medio"]      # compat (agora é a PD da folha)
        return out

    # ------------------------------------------------------------------
    # ASSIGN: rotula cada linha com seu segmento-folha
    #   Por padrão adiciona também a nota de PD (1..N) e a descrição
    #   por extenso. Colunas: <col>, <col>_nota, <col>_desc
    # ------------------------------------------------------------------
    def assign(
        self,
        col_name: str = "segmento_pd",
        add_grade: bool = True,
        add_desc: bool = True,
        ascending: bool = True,
    ) -> pd.DataFrame:
        out = self.df.copy()
        out[col_name] = pd.Series(pd.NA, index=out.index, dtype="object")

        nota_map, desc_map = self._grade_map(ascending=ascending)
        if add_grade:
            out[f"{col_name}_nota"] = pd.Series(pd.NA, index=out.index, dtype="object")
        if add_desc:
            out[f"{col_name}_desc"] = pd.Series(pd.NA, index=out.index, dtype="object")

        for sid, seg in self.segments.items():
            if not seg["is_leaf"]:
                continue
            out.loc[seg["mask"], col_name] = sid
            if add_grade:
                out.loc[seg["mask"], f"{col_name}_nota"] = nota_map[sid]
            if add_desc:
                out.loc[seg["mask"], f"{col_name}_desc"] = desc_map[sid]

        if add_grade:
            out[f"{col_name}_nota"] = out[f"{col_name}_nota"].astype("Int64")
        return out

    # ==================================================================
    # PERSISTÊNCIA: salvar / carregar a ÁRVORE inteira em JSON
    #   Serializa a estrutura (segmentos + condições + metadados), não as
    #   máscaras (que dependem dos dados). Ao carregar, as máscaras são
    #   reconstruídas a partir das condições sobre o DataFrame fornecido — por
    #   isso `load`/`from_dict` exigem um `df`. Permite versionar a segmentação
    #   e reaplicá-la (mesmos dados ou novos) em qualquer máquina.
    # ==================================================================
    SCHEMA = "yggdrasil.credit_risk.pd.tree/1"

    @staticmethod
    def _conditions_from_json(conditions):
        """Inverso de `_conditions_json`: restaura lo/hi ilimitados como ±inf."""
        out = []
        for c in conditions:
            if c["kind"] == "na":
                out.append({"feature": c["feature"], "kind": "na"})
            elif c["kind"] == "num":
                lo = float("-inf") if c.get("lo") is None else float(c["lo"])
                hi = float("inf") if c.get("hi") is None else float(c["hi"])
                d = {"feature": c["feature"], "kind": "num", "lo": lo, "hi": hi}
                if c.get("include_na"):
                    d["include_na"] = True
                out.append(d)
            else:
                d = {"feature": c["feature"], "kind": "cat",
                     "cats": [str(x) for x in c["cats"]]}
                if c.get("include_na"):
                    d["include_na"] = True
                out.append(d)
        return out

    def to_dict(self) -> dict:
        """Estrutura serializável da árvore (segmentos + metadados, sem máscaras)."""
        segs = {}
        for sid, s in self.segments.items():
            segs[sid] = {
                "label": s["label"],
                "depth": s["depth"],
                "is_leaf": s["is_leaf"],
                "parent": s["parent"],
                "path": list(s["path"]),
                "conditions": self._conditions_json(s["conditions"]),
            }
        return {
            "schema": self.SCHEMA,
            "meta": {
                "target": self.target,
                "sample_col": self.sample_col,
                "ref_sample": self.ref_sample,
                "min_leaf_rows": self.min_leaf_rows,
                "feature_labels": dict(self.feature_labels),
            },
            "segments": segs,
        }

    def _load_segments(self, segs: dict):
        """Reconstrói self.segments a partir da forma serializada (recalcula máscaras)."""
        novo = {}
        for sid, s in segs.items():
            conds = self._conditions_from_json(s["conditions"])
            novo[sid] = {
                "mask": _match_conditions_pandas(self.df, conds),
                "label": s["label"],
                "depth": int(s["depth"]),
                "is_leaf": bool(s["is_leaf"]),
                "path": list(s.get("path", [])),
                "parent": s["parent"],
                "conditions": conds,
            }
        if "root" not in novo:
            raise ValueError("Árvore inválida: segmento 'root' ausente.")
        self.segments = novo
        self.history = []
        return self

    def save(self, path: str) -> str:
        """Salva a árvore em um arquivo JSON. Devolve o caminho."""
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def from_dict(cls, data: dict, df: pd.DataFrame, verbose: bool = False):
        """Reconstrói um segmentador a partir de `to_dict()` + um DataFrame."""
        meta = data.get("meta", {})
        seg = cls(df, target=meta.get("target", "target"),
                  sample_col=meta.get("sample_col"),
                  ref_sample=meta.get("ref_sample", "DES"),
                  feature_labels=meta.get("feature_labels"),
                  min_leaf_rows=meta.get("min_leaf_rows", 50),
                  verbose=verbose)
        seg._load_segments(data["segments"])
        return seg

    @classmethod
    def load(cls, path: str, df: pd.DataFrame, verbose: bool = False):
        """Carrega uma árvore salva em JSON e a aplica ao DataFrame fornecido."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, df, verbose=verbose)

    # ==================================================================
    # RÉGUA / SCORING
    # ==================================================================
    @staticmethod
    def _conditions_json(conditions):
        """Converte condições para forma serializável (lo/hi=None se ilimitado)."""
        out = []
        for c in conditions:
            if c["kind"] == "na":
                out.append({"feature": c["feature"], "kind": "na"})
            elif c["kind"] == "num":
                d = {"feature": c["feature"], "kind": "num",
                     "lo": None if c["lo"] == float("-inf") else float(c["lo"]),
                     "hi": None if c["hi"] == float("inf") else float(c["hi"])}
                if c.get("include_na"):
                    d["include_na"] = True
                out.append(d)
            else:
                d = {"feature": c["feature"], "kind": "cat",
                     "cats": [str(x) for x in c["cats"]]}
                if c.get("include_na"):
                    d["include_na"] = True
                out.append(d)
        return out

    def _regua_dict(self) -> dict:
        """Régua final: folhas com nota, condições e PD (na referência/DES)."""
        nota_map, _ = self._grade_map()
        leaves = []
        for sid, seg in self.segments.items():
            if not seg["is_leaf"]:
                continue
            sub = self.df[seg["mask"]]
            if self.sample_col is not None:
                ref = sub.loc[sub[self.sample_col] == self.ref_sample, self.target]
                pdv = float(ref.mean()) if len(ref) else float(sub[self.target].mean())
            else:
                pdv = float(sub[self.target].mean())
            if np.isnan(pdv):                       # folha vazia → PD global (sem NaN na régua)
                if self.sample_col is not None:
                    g = self.df.loc[self.df[self.sample_col] == self.ref_sample,
                                    self.target].mean()
                else:
                    g = self.df[self.target].mean()
                pdv = float(g) if not pd.isna(g) else 0.0
            leaves.append({"id": sid, "nota": int(nota_map[sid]),
                           "conditions": self._conditions_json(seg["conditions"]),
                           "pd": round(pdv, 6)})
        leaves.sort(key=lambda x: x["nota"])
        return {"target": self.target, "ref_sample": self.ref_sample, "leaves": leaves}

    def predict(self, X: pd.DataFrame, col_seg="segmento_pd",
                col_nota="nota_pd", col_pd="pd_regua") -> pd.DataFrame:
        """Aplica a régua a um DataFrame pandas novo: segmento, nota e PD."""
        return _aplicar_regua_pandas(self._regua_dict(), X, col_seg, col_nota, col_pd)

    # ------------------------------------------------------------------
    # TO_PYSPARK: gera o código da régua como F.when().otherwise() para
    #   aplicar a segmentação (segmento + nota + PD) em escala no Spark.
    # ------------------------------------------------------------------
    def to_pyspark(self, func_name: str = "aplicar_regua_pd") -> str:
        regua = self._regua_dict()

        def cond_expr(conds):
            parts = []
            for c in conds:
                feat = c["feature"]
                if c["kind"] == "na":
                    parts.append(f'F.col("{feat}").isNull()')
                    continue
                if c["kind"] == "num":
                    sub = []
                    if c["lo"] is not None:
                        sub.append(f'(F.col("{feat}") > {c["lo"]})')
                    if c["hi"] is not None:
                        sub.append(f'(F.col("{feat}") <= {c["hi"]})')
                    expr = " & ".join(sub) if sub else "F.lit(True)"
                else:
                    cats = ", ".join(repr(x) for x in c["cats"])
                    # cast p/ string espelha o astype(str) do pandas/pyfunc
                    expr = f'F.col("{feat}").cast("string").isin({cats})'
                if c.get("include_na"):
                    expr = f'(({expr}) | F.col("{feat}").isNull())'
                parts.append(expr)
            return " & ".join(parts) if parts else "F.lit(True)"

        def chain(valfn):
            out = []
            for i, leaf in enumerate(regua["leaves"], 1):
                head = "  F.when" if i == 1 else "   .when"
                out.append(f'        {head}(c{i}, {valfn(leaf)})')
            out.append("           .otherwise(F.lit(None))")
            return "\n".join(out)

        L = ["from pyspark.sql import functions as F", "",
             f"def {func_name}(df, col_seg='segmento_pd', col_nota='nota_pd', col_pd='pd_regua'):",
             '    """Régua de PD gerada por SequentialPDSegmenter (segmento, nota e PD por folha)."""']
        for i, leaf in enumerate(regua["leaves"], 1):
            L.append(f'    c{i} = {cond_expr(leaf["conditions"])}')
        L.append("    seg = (")
        L.append(chain(lambda lf: f'F.lit({lf["id"]!r})'))
        L.append("    )")
        L.append("    nota = (")
        L.append(chain(lambda lf: f'F.lit({lf["nota"]})'))
        L.append("    )")
        L.append("    pd_val = (")
        L.append(chain(lambda lf: f'F.lit({lf["pd"]})'))
        L.append("    )")
        L.append("    return (df.withColumn(col_seg, seg)")
        L.append("              .withColumn(col_nota, nota)")
        L.append("              .withColumn(col_pd, pd_val))")
        return "\n".join(L)

    # ------------------------------------------------------------------
    # REGUA_FEATURES: colunas usadas pela árvore (necessárias na tabela).
    # ------------------------------------------------------------------
    def regua_features(self) -> list:
        feats = []
        for leaf in self._regua_dict()["leaves"]:
            for c in leaf["conditions"]:
                if c["feature"] not in feats:
                    feats.append(c["feature"])
        return feats

    # ------------------------------------------------------------------
    # APPLY_SPARK: aplica a régua DIRETAMENTE num Spark DataFrame, devolvendo-o
    #   com as colunas de segmento, nota e PD ("reconstrói as folhas" na
    #   tabela). Diferente de `to_pyspark` (que só gera o código), aqui a régua
    #   é executada. Exige pyspark e que as colunas usadas na árvore existam no
    #   `sdf` com o MESMO nome (senão levanta erro listando as que faltam).
    # ------------------------------------------------------------------
    def apply_spark(self, sdf, col_seg: str = "segmento_pd",
                    col_nota: str = "nota_pd", col_pd: str = "pd_regua"):
        try:
            from pyspark.sql import functions as F
        except ImportError as e:  # pragma: no cover
            raise ImportError("apply_spark requer pyspark — use: pip install pyspark") from e

        regua = self._regua_dict()
        if not regua["leaves"]:
            raise ValueError("Árvore sem folhas — cresça a segmentação antes de aplicar.")

        faltando = [f for f in self.regua_features() if f not in sdf.columns]
        if faltando:
            raise ValueError(
                f"Colunas ausentes no Spark DataFrame: {faltando}. A tabela precisa "
                f"ter as mesmas colunas usadas na árvore: {self.regua_features()}.")

        def cond_col(conds):
            expr = F.lit(True)
            for c in conds:
                feat = c["feature"]
                if c["kind"] == "na":
                    part = F.col(feat).isNull()
                elif c["kind"] == "num":
                    part = F.lit(True)
                    if c.get("lo") is not None:
                        part = part & (F.col(feat) > c["lo"])
                    if c.get("hi") is not None:
                        part = part & (F.col(feat) <= c["hi"])
                    if c.get("include_na"):
                        part = part | F.col(feat).isNull()
                else:
                    part = F.col(feat).cast("string").isin([str(x) for x in c["cats"]])
                    if c.get("include_na"):
                        part = part | F.col(feat).isNull()
                expr = expr & part
            return expr

        seg_col = nota_col = pd_col = None
        for leaf in regua["leaves"]:
            cond = cond_col(leaf["conditions"])
            if seg_col is None:
                seg_col = F.when(cond, F.lit(leaf["id"]))
                nota_col = F.when(cond, F.lit(leaf["nota"]))
                pd_col = F.when(cond, F.lit(leaf["pd"]))
            else:
                seg_col = seg_col.when(cond, F.lit(leaf["id"]))
                nota_col = nota_col.when(cond, F.lit(leaf["nota"]))
                pd_col = pd_col.when(cond, F.lit(leaf["pd"]))

        return (sdf.withColumn(col_seg, seg_col.otherwise(F.lit(None)))
                   .withColumn(col_nota, nota_col.otherwise(F.lit(None)))
                   .withColumn(col_pd, pd_col.otherwise(F.lit(None))))

    # ------------------------------------------------------------------
    # SUGGEST_SPLIT: recomenda a melhor variável para dividir uma folha,
    #   pelo maior Information Value, e devolve o ranking completo.
    # ------------------------------------------------------------------
    def suggest_split(self, sid: str | None = None, features: list | None = None,
                      max_n_bins: int = 4, min_bin_size: float = 0.05) -> dict:
        if sid is None:
            sid = "root"
        iv = self.variable_iv(sid, features=features, max_n_bins=max_n_bins,
                              min_bin_size=min_bin_size, with_psi=False)
        if len(iv) == 0 or pd.isna(iv.iloc[0]["iv"]):
            return {"sid": sid, "feature": None, "iv": None, "forca": None,
                    "ranking": iv, "msg": "nenhuma variável informativa para esta folha"}
        top = iv.iloc[0]
        return {"sid": sid, "feature": top["variavel"], "tipo": top["tipo"],
                "iv": float(top["iv"]), "forca": top["forca"], "ranking": iv,
                "msg": f"dividir por '{top['variavel']}' (IV={top['iv']:.4f}, {top['forca']})"}

    def _is_descendant_or_self(self, sid, ancestor) -> bool:
        """True se `sid` é o próprio `ancestor` ou um descendente dele."""
        cur, guard = sid, 0
        while cur is not None and cur in self.segments and guard < 10000:
            if cur == ancestor:
                return True
            cur = self.segments[cur].get("parent")
            guard += 1
        return False

    # ------------------------------------------------------------------
    # FIT_AUTO: constrói uma árvore de forma gulosa — em cada folha escolhe a
    #   variável de maior IV e divide com binning ótimo, até a profundidade
    #   máxima ou IV abaixo do mínimo. Ponto de partida para refinar à mão.
    #   Se `subtree` (id de folha) for dado, cresce APENAS a subárvore daquela
    #   folha (até `max_depth` níveis abaixo dela), sem reiniciar nem tocar no
    #   resto da árvore — útil para "auto-ajustar só esta folha".
    # ------------------------------------------------------------------
    def fit_auto(self, features: list | None = None, max_depth: int = 3,
                 min_iv: float = 0.02, max_n_bins: int = 2, min_bin_size: float = 0.05,
                 from_scratch: bool = True, subtree: str | None = None,
                 min_leaf_repr: float | None = None, max_bin_repr: float | None = None,
                 verbose: bool = True):
        """Cresce a árvore de forma gulosa (maior IV por nível).

        ``min_leaf_repr`` e ``max_bin_repr`` são **representatividades GLOBAIS**
        (fração da carteira inteira), não da folha-mãe:

        * ``min_leaf_repr`` — cada folha terminal deve reter ao menos essa fração
          da carteira. Uma folha pequena demais para gerar dois filhos ≥
          ``min_leaf_repr`` não é dividida (vira terminal). Corrige o efeito de o
          ``min_bin_size`` do optbinning ser relativo à folha-mãe e, ao compor por
          nível, deixar folhas profundas com representatividade ínfima.
        * ``max_bin_repr`` — nenhuma quebra pode concentrar mais que essa fração da
          carteira (força granularidade em segmentos dominantes).

        Cada um é traduzido por folha para a fração local exigida pelo optbinning
        (``repr_global · N / n_folha``).
        """
        import io
        import math
        import contextlib
        if subtree is not None and subtree not in self.segments:
            raise ValueError(f"Folha '{subtree}' não existe na árvore atual.")
        if subtree is None:                       # árvore inteira (comportamento padrão)
            if from_scratch:
                root = self.segments["root"]
                root["is_leaf"] = True
                self.segments = {"root": root}
            base_depth = 0
            def in_scope(_sid):
                return True
        else:                                     # só a subárvore da folha escolhida
            base_depth = self.segments[subtree]["depth"]
            def in_scope(_sid):
                return self._is_descendant_or_self(_sid, subtree)
        for depth in range(base_depth, base_depth + max_depth):
            atuais = [sid for sid, s in self.segments.items()
                      if s["is_leaf"] and s["depth"] == depth and in_scope(sid)]
            for sid in atuais:
                if sid in self.segments and not self.segments[sid]["is_leaf"]:
                    continue
                sub = self.df[self.segments[sid]["mask"]]
                n_leaf = len(sub)
                # concentrações GLOBAIS → fração local (da folha) p/ o optbinning
                loc_min = min_bin_size
                if min_leaf_repr is not None and n_leaf:
                    loc_min = min_leaf_repr * len(self.df) / n_leaf
                    if loc_min >= 0.5:        # folha pequena p/ 2 filhos ≥ min_leaf_repr
                        continue              # → não divide (vira terminal)
                # a concentração MÁXIMA por quebra pode exigir mais bins do que o
                # split binário padrão. k bins cada ≤ loc_max precisa de k ≥ 1/loc_max;
                # com monotonicidade no alvo o optbinning fica INFEASIBLE no
                # limite exato, então pedimos 1 bin de folga (e limitamos a 6).
                loc_max, mnb = None, max_n_bins
                if max_bin_repr is not None and n_leaf:
                    loc_max = min(0.999, max_bin_repr * len(self.df) / n_leaf)
                    if loc_max <= loc_min:    # restrição de máx incompatível → ignora
                        loc_max = None
                    else:
                        mnb = max(max_n_bins, min(math.ceil(1.0 / loc_max) + 1, 6))
                        if loc_max * mnb < 1.0:   # inviável mesmo com 6 bins → ignora máx
                            loc_max, mnb = None, max_n_bins
                if len(self._fit_frame(sub, loc_min)) < self.min_leaf_rows:
                    continue
                iv = self.variable_iv(sid, features=features, max_n_bins=mnb,
                                      min_bin_size=loc_min, with_psi=False)
                if len(iv) == 0:
                    continue
                top = iv.iloc[0]
                if pd.isna(top["iv"]) or top["iv"] < min_iv:
                    continue
                with contextlib.redirect_stdout(io.StringIO()):
                    self.grow(top["variavel"], only_segments=[sid], max_n_bins=mnb,
                              min_bin_size=loc_min, max_bin_size=loc_max, relax_max=True)
        n_folhas = sum(s["is_leaf"] for s in self.segments.values())
        prof = max(s["depth"] for s in self.segments.values())
        if verbose:
            escopo = "subárvore" if subtree is not None else "árvore gulosa"
            print(f"[fit_auto] {escopo} construída: profundidade {prof} "
                  f"(máx +{max_depth}), IV mínimo {min_iv} → {n_folhas} folhas")
        return self

    # ------------------------------------------------------------------
    # LOG_TO_MLFLOW: salva a segmentação no MLflow — parâmetros, métricas,
    #   artefatos (folhas, árvore, régua PySpark, régua JSON) e o MODELO
    #   pyfunc (régua aplicável via .predict para scoring).
    # ------------------------------------------------------------------
    def log_to_mlflow(self, experiment: str | None = None,
                      run_name: str | None = None, registered_model_name: str | None = None,
                      artifact_path: str = "modelo_pd", extra_params: dict | None = None,
                      registry_uri: str | None = None, verbose: bool = True) -> str:
        try:
            import mlflow
            import mlflow.pyfunc
            from mlflow.models import infer_signature
        except ImportError as e:  # pragma: no cover
            raise ImportError("mlflow não está instalado — use: pip install mlflow") from e
        import json
        import os
        import tempfile

        regua = self._regua_dict()
        feats = []
        for leaf in regua["leaves"]:
            for c in leaf["conditions"]:
                if c["feature"] not in feats:
                    feats.append(c["feature"])
        n_folhas = len(regua["leaves"])
        prof = max(s["depth"] for s in self.segments.values())
        params = {"target": self.target, "ref_sample": self.ref_sample,
                  "sample_col": self.sample_col, "n_folhas": n_folhas,
                  "profundidade": prof, "min_leaf_rows": self.min_leaf_rows,
                  "variaveis": ", ".join(feats) or "(nenhuma)"}
        if extra_params:
            params.update(extra_params)

        metrics = {"n_folhas": float(n_folhas), "profundidade": float(prof)}
        for leaf in regua["leaves"]:
            metrics[f"pd_nota_{leaf['nota']}"] = leaf["pd"]
        try:
            for _, r in self.metrics().iterrows():
                a = r["amostra"]
                for k in ("KS", "AUC", "Gini", "Acuracia", "F1"):
                    if not pd.isna(r[k]):
                        metrics[f"{k.lower()}_{a}"] = float(r[k])
        except Exception:
            pass
        if self.sample_col is not None:
            try:
                for _, r in self.psi().iterrows():
                    metrics[f"psi_{r['amostra']}"] = float(r["psi"])
            except Exception:
                pass

        class _ReguaPDModel(mlflow.pyfunc.PythonModel):
            def __init__(self, regua):
                self.regua = regua

            def predict(self, context, model_input):
                import numpy as _np
                import pandas as _pd
                df = model_input
                seg = _pd.Series(_pd.NA, index=df.index, dtype="object")
                nota = _pd.Series(_pd.NA, index=df.index, dtype="Int64")
                pdcol = _pd.Series(_np.nan, index=df.index, dtype="float64")
                for leaf in self.regua["leaves"]:
                    m = _pd.Series(True, index=df.index)
                    for c in leaf["conditions"]:
                        feat = c["feature"]
                        if c["kind"] == "na":
                            m &= df[feat].isna()
                            continue
                        if c["kind"] == "num":
                            sub = _pd.Series(True, index=df.index)
                            if c.get("lo") is not None:
                                sub &= df[feat] > c["lo"]
                            if c.get("hi") is not None:
                                sub &= df[feat] <= c["hi"]
                        else:
                            sub = df[feat].astype(str).isin([str(x) for x in c["cats"]])
                        if c.get("include_na"):
                            sub = sub | df[feat].isna()
                        m &= sub
                    seg[m] = leaf["id"]
                    nota[m] = leaf["nota"]
                    pdcol[m] = leaf["pd"]
                return _pd.DataFrame({"segmento_pd": seg, "nota_pd": nota, "pd_regua": pdcol},
                                     index=df.index)

        # Unity Catalog exige set_registry_uri('databricks-uc') e assinatura do modelo
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        if experiment:
            mlflow.set_experiment(experiment)

        model = _ReguaPDModel(regua)
        cols = feats if feats else [c for c in self.df.columns
                                    if c not in (self.target, self.sample_col)]
        input_example = self.df[cols].head(5).reset_index(drop=True).copy()
        try:
            output_example = model.predict(None, input_example)
            signature = infer_signature(input_example, output_example)
        except Exception:
            signature = None

        def _log_model():
            kw = dict(python_model=model, registered_model_name=registered_model_name,
                      signature=signature, input_example=input_example,
                      pip_requirements=["pandas", "numpy"])
            try:
                return mlflow.pyfunc.log_model(name=artifact_path, **kw)
            except TypeError:
                return mlflow.pyfunc.log_model(artifact_path=artifact_path, **kw)

        version = None
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            with tempfile.TemporaryDirectory() as d:
                self.leaves(with_psi=(self.sample_col is not None)).to_csv(
                    os.path.join(d, "folhas.csv"), index=False)
                with open(os.path.join(d, "arvore.txt"), "w") as f:
                    f.write(self.tree())
                with open(os.path.join(d, "regua_pyspark.py"), "w") as f:
                    f.write(self.to_pyspark())
                with open(os.path.join(d, "regua.json"), "w") as f:
                    json.dump(regua, f, ensure_ascii=False, indent=2)
                # árvore completa (estrutura) para recarregar via .load(...)
                with open(os.path.join(d, "arvore.json"), "w", encoding="utf-8") as f:
                    json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
                # CSI por variável (estabilidade das entradas), se houver amostras
                if self.sample_col is not None:
                    try:
                        self.csi().to_csv(os.path.join(d, "csi_variaveis.csv"), index=False)
                    except Exception:
                        pass
                mlflow.log_artifacts(d, artifact_path="regua")
            model_info = _log_model()
            version = getattr(model_info, "registered_model_version", None)
            run_id = run.info.run_id

        if verbose:
            msg = (f"[mlflow] modelo salvo (run {run_id[:8]}…) — {n_folhas} folhas. "
                   f"Artefatos: 'regua/' + modelo pyfunc '{artifact_path}/'.")
            if registered_model_name:
                v = f" v{version}" if version is not None else ""
                msg += f" Registrado em '{registered_model_name}'{v}."
            print(msg)
        return run_id
