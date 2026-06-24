"""
SequentialLGDSegmenter
======================
Construtor sequencial e híbrido de segmentações para modelos de LGD.

- Cresce a segmentação em camadas (`grow`), dividindo cada folha por uma nova
  variável usando OPTIMAL BINNING (OptBinning) ou CORTES MANUAIS.
- Preview do split antes de efetivar (`show_grow`): LGD médio + representatividade.
- Poda de folhas pouco representativas ou sem separação de LGD (`prune`).
- PSI de estabilidade entre amostras (DES como referência) usando os próprios
  segmentos como bins (`psi`, `psi_detalhe`).

Contexto: parâmetros de risco de crédito sob Resolução CMN 4.966/2021 e IFRS 9.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from optbinning import ContinuousOptimalBinning
except ImportError:  # pragma: no cover
    ContinuousOptimalBinning = None


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
    """Faixas usuais de força do Information Value."""
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


def _aplicar_regua_pandas(regua, df, col_seg="segmento_lgd",
                          col_nota="nota_lgd", col_lgd="lgd_regua"):
    """Aplica uma régua (dict de folhas) a um DataFrame pandas."""
    seg = pd.Series(pd.NA, index=df.index, dtype="object")
    nota = pd.Series(pd.NA, index=df.index, dtype="Int64")
    lgd = pd.Series(np.nan, index=df.index, dtype="float64")
    for leaf in regua["leaves"]:
        m = _match_conditions_pandas(df, leaf["conditions"])
        seg[m] = leaf["id"]
        nota[m] = leaf["nota"]
        lgd[m] = leaf["lgd"]
    return pd.DataFrame({col_seg: seg, col_nota: nota, col_lgd: lgd}, index=df.index)


# ======================================================================
# Classe principal
# ======================================================================
class SequentialLGDSegmenter:
    """Segmentação sequencial de LGD com binning ótimo/manual, poda e PSI."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str = "lgd",
        sample_col: str | None = None,
        ref_sample: str = "DES",
        feature_labels: dict[str, str] | None = None,
        min_leaf_rows: int = 50,
        verbose: bool = True,
    ):
        if ContinuousOptimalBinning is None:
            raise ImportError("optbinning não instalado. Rode: pip install optbinning")

        self.df = df.copy()
        self.target = target
        self.sample_col = sample_col
        self.ref_sample = ref_sample
        # rótulos amigáveis por variável para a descrição por extenso
        self.feature_labels = feature_labels or {}
        # mínimo de linhas (na amostra de ajuste) para tentar binning ótimo
        self.min_leaf_rows = min_leaf_rows

        if sample_col is not None:
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

    def _resolve_bins(self, sub, feature, splits, dtype, max_n_bins, min_bin_size):
        """Resolve os bins de um ramo. Devolve (bins, modo, kind).
        - num: bins = [{'kind':'num','lo','hi'}, ...]
        - cat: bins = [{'kind':'cat','cats':[...]}, ...]
        No modo ótimo, o binning é ajustado só na amostra de referência (DES).
        No modo manual: splits numérico = lista de cortes; splits categórico =
        lista de grupos (cada grupo é uma lista de categorias).
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
                x_obs = x[~np.isnan(x)]
                if len(y) < 4 or x_obs.size == 0 or np.unique(x_obs).size < 2:
                    cortes = []                            # dados degenerados → sem corte
                else:
                    b = ContinuousOptimalBinning(
                        name=feature, dtype="numerical", max_n_bins=max_n_bins,
                        min_bin_size=min_bin_size, monotonic_trend="auto_asc_desc")
                    cortes = _fit_optbinning_splits(b, x, y)
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
            modo = "manual"
        else:
            fit = self._fit_frame(sub, min_bin_size)
            fit = fit[fit[feature].notna() & fit[self.target].notna()]  # NaN fora do ajuste
            xs = fit[feature].astype(str).to_numpy()
            ys = fit[self.target].to_numpy(dtype="float64")
            if len(ys) < 4 or np.unique(xs).size < 2:
                grupos = []                          # dados degenerados → sem grupos
            else:
                b = ContinuousOptimalBinning(
                    name=feature, dtype="categorical", max_n_bins=max_n_bins,
                    min_bin_size=min_bin_size, monotonic_trend="auto_asc_desc")
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
    # Tabela de bins: LGD médio + representatividade (num ou cat)
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
                "lgd_medio": round(s.mean(), 4),
                "lgd_std": round(s.std(), 4),
            })
        tbl = pd.DataFrame(linhas)
        means = tbl["lgd_medio"]
        tbl.attrs["mono_ok"] = bool(
            means.is_monotonic_increasing or means.is_monotonic_decreasing
        )
        return tbl

    # ------------------------------------------------------------------
    # SHOW_GROW: preview do split (não altera estado)
    #   dtype: None=auto-detecta, 'num' ou 'cat' para forçar
    # ------------------------------------------------------------------
    def show_grow(self, feature, splits=None, dtype=None, max_n_bins=4,
                  min_bin_size=0.05, only_segments=None):
        targets = {
            sid: s for sid, s in self.segments.items()
            if s["is_leaf"] and (only_segments is None or sid in only_segments)
        }
        n_total = len(self.df)
        previews = {}
        for sid, seg in targets.items():
            sub = self.df[seg["mask"]]
            bins, modo, kind = self._resolve_bins(
                sub, feature, splits, dtype, max_n_bins, min_bin_size)
            if not bins:
                print(f"[{sid}] sem corte válido em '{feature}' ({modo})")
                continue
            tbl = self._bin_table(sub, feature, bins, n_total)
            previews[sid] = tbl
            print(f"\n┌─ PREVIEW: dividir '{sid}'")
            print(f"│  feature = {feature} | tipo = {kind} | modo = {modo}")
            print(f"│  monotonicidade do LGD respeitada: {tbl.attrs['mono_ok']}")
            print(tbl.to_string(index=False))
        print("\n→ se aprovado, repita como .grow(...) com os mesmos argumentos")
        return previews

    # ------------------------------------------------------------------
    # GROW: efetiva o split (manual ou ótimo, num ou cat) em cada folha-alvo
    # ------------------------------------------------------------------
    def grow(self, feature, splits=None, dtype=None, max_n_bins=4,
             min_bin_size=0.05, only_segments=None):
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
                sub, feature, splits, dtype, max_n_bins, min_bin_size)
            modo_usado = modo
            if not bins:
                print(f"[{sid}] sem corte válido em '{feature}' ({modo}) — folha mantida")
                continue
            for b in bins:
                child_mask = seg["mask"] & self._mask_in(self.df, feature, b)
                if child_mask.sum() == 0:
                    continue
                child_label = self._bin_label(feature, b)
                child_id = f"{sid} | {child_label}" if sid != "root" else child_label
                novos[child_id] = {
                    "mask": child_mask,
                    "label": child_label,
                    "depth": seg["depth"] + 1,
                    "is_leaf": True,
                    "path": seg["path"] + [child_label],
                    "parent": sid,
                    "conditions": seg["conditions"] + [self._bin_condition(feature, b)],
                }
            self.segments[sid]["is_leaf"] = False
        self.segments.update(novos)
        self.history.append({"feature": feature, "modo": modo_usado, "splits": splits})
        n_folhas = sum(s["is_leaf"] for s in self.segments.values())
        print(f"[grow] '{feature}' ({modo_usado}) criou {len(novos)} segmentos. "
              f"Folhas atuais: {n_folhas}")
        return self

    # ------------------------------------------------------------------
    # PRUNE: colapsa splits que não compensam (poda bottom-up)
    #   Um split (pai com TODOS os filhos folha) é desfeito se:
    #     - algum filho tem repr_% < min_repr  (materialidade), OU
    #     - a amplitude de LGD entre os filhos < min_lgd_gap (sem separação)
    #   Itera até estabilizar, permitindo poda em cascata.
    # ------------------------------------------------------------------
    def prune(self, min_repr: float = 2.0, min_lgd_gap: float = 0.03,
              verbose: bool = True):
        n_total = len(self.df)
        rodadas = 0
        while True:
            # mapeia pai -> lista de ids de filhos
            filhos_por_pai: dict[str, list[str]] = {}
            for sid, seg in self.segments.items():
                p = seg["parent"]
                if p is not None:
                    filhos_por_pai.setdefault(p, []).append(sid)

            colapsou = False
            for pai, filhos in filhos_por_pai.items():
                # só é colapsável se o pai existe e TODOS os filhos são folhas
                if pai not in self.segments:
                    continue
                if not all(self.segments[f]["is_leaf"] for f in filhos):
                    continue

                # métricas dos filhos
                reprs, means = [], []
                for f in filhos:
                    sub = self.df[self.segments[f]["mask"]]
                    reprs.append(100 * len(sub) / n_total)
                    means.append(sub[self.target].mean())
                gap = (np.nanmax(means) - np.nanmin(means)) if means else 0.0
                min_r = min(reprs) if reprs else 0.0

                motivo = None
                if min_r < min_repr:
                    motivo = f"folha com repr_={min_r:.1f}% < {min_repr}%"
                elif gap < min_lgd_gap:
                    motivo = f"amplitude de LGD {gap:.4f} < {min_lgd_gap}"

                if motivo:
                    for f in filhos:
                        self.segments.pop(f, None)
                    self.segments[pai]["is_leaf"] = True
                    colapsou = True
                    if verbose:
                        alvo = "root" if self.segments[pai]["label"] == "root" else pai
                        print(f"[prune] colapsado '{alvo}' ({len(filhos)} folhas) — {motivo}")
                    break  # recomeça do zero (estado mudou)

            rodadas += 1
            if not colapsou:
                break

        n_folhas = sum(s["is_leaf"] for s in self.segments.values())
        if verbose:
            print(f"[prune] concluído em {rodadas} rodada(s). Folhas finais: {n_folhas}")
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
    #   side="left"  -> vizinha de menor corte (num) / menor LGD (cat)
    #   side="right" -> vizinha de maior corte (num) / maior LGD (cat)
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
            irmaos.sort(key=lambda c: self.df[self.segments[c]["mask"]][self.target].mean())

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
    #   adjacentes que NÃO se distinguem em LGD. Em cada rodada, escolhe o par
    #   de irmãs vizinhas mais parecido e o funde se:
    #     - o teste de hipótese não rejeita a igualdade de LGD (p > alpha), OU
    #     - a diferença de LGD médio entre elas é < min_lgd_gap.
    #   Só funde irmãs (a única fusão válida na árvore). Por padrão o nó de
    #   faltantes NÃO entra; com `include_missing=True` ele também é juntado ao
    #   bin populado irmão estatisticamente mais próximo. `protect` = ids de
    #   folhas a preservar (ex.: travadas na UI). Itera até nenhum par qualificar.
    # ------------------------------------------------------------------
    def auto_merge(self, alpha: float = 0.05, min_lgd_gap: float = 0.0,
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
                        if not ((not np.isnan(p) and p > alpha) or gap < min_lgd_gap):
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
                            if not ((not np.isnan(p) and p > alpha) or gap < min_lgd_gap):
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
                  f"(alpha={alpha}, min_lgd_gap={min_lgd_gap}, "
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
    # LEAVES: segmentos-folha finais, com nota de LGD e descrição
    #   nota_lgd: 1..N ordenada por LGD (1 = menor LGD por padrão)
    #   with_psi:  adiciona a contribuição de PSI de cada folha por amostra
    #              (psi_<amostra>), tendo a referência (DES) como base
    #   with_test: adiciona p_vs_prox = p-valor do teste de LGD entre a folha
    #              e a próxima (na ordem de LGD). p alto ⇒ folhas não distinguíveis
    #              (candidatas a fusão). test: 'mannwhitney' (default) ou 'welch'
    # ------------------------------------------------------------------
    def leaves(self, ascending: bool = True, with_psi: bool = False,
               with_test: bool = False, test: str = "mannwhitney") -> pd.DataFrame:
        linhas, n_total = [], len(self.df)
        for sid, seg in self.segments.items():
            if not seg["is_leaf"]:
                continue
            sub = self.df[seg["mask"]]
            row = {
                "segmento": sid,
                "descricao": self._descrever(seg["conditions"]),
                "profundidade": seg["depth"],
                "n": len(sub),
                "repr_%": round(100 * len(sub) / n_total, 1),
                "lgd_medio": round(sub[self.target].mean(), 4),
                "lgd_std": round(sub[self.target].std(), 4),
            }
            if self.sample_col is not None:
                for amostra in self.df[self.sample_col].dropna().unique():
                    s_am = sub.loc[sub[self.sample_col] == amostra, self.target]
                    row[f"lgd_{amostra}"] = round(s_am.mean(), 4) if len(s_am) else np.nan
            linhas.append(row)
        out = (
            pd.DataFrame(linhas)
            .sort_values("lgd_medio", ascending=ascending)
            .reset_index(drop=True)
        )
        out.insert(1, "nota_lgd", range(1, len(out) + 1))

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

    # alvo (LGD) de uma folha, restrito à amostra de referência quando houver
    def _leaf_target(self, sid: str) -> np.ndarray:
        m = self.segments[sid]["mask"]
        if self.sample_col is not None:
            m = m & (self.df[self.sample_col] == self.ref_sample)
        return self.df.loc[m, self.target].to_numpy()

    # p-valor do teste de igualdade de LGD entre duas folhas (na referência)
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

    # p-valor do teste de LGD entre folhas adjacentes (na ordem de LGD)
    def _append_adjacency_test(self, out, test="mannwhitney", min_n=8):
        amostras = out["segmento"].tolist()
        pvals = []
        for i, sid in enumerate(amostras):
            if i == len(amostras) - 1:
                pvals.append(np.nan); continue
            p = self._pair_pvalue(sid, amostras[i + 1], test=test, min_n=min_n)
            pvals.append(round(p, 4) if not np.isnan(p) else np.nan)
        out["p_vs_prox"] = pvals
        return out

    # mapeia segmento -> (nota, descrição) para uso no assign
    def _grade_map(self, ascending: bool = True):
        lv = self.leaves(ascending=ascending)
        nota = dict(zip(lv["segmento"], lv["nota_lgd"]))
        desc = dict(zip(lv["segmento"], lv["descricao"]))
        return nota, desc

    # ------------------------------------------------------------------
    # TREE: desenha a árvore hierárquica (nós internos + folhas) em texto.
    #   Cada nó mostra n, representatividade e LGD médio; folhas trazem [nota N].
    #   Filhos são ordenados por LGD (ascendente por padrão).
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
            lgd = sub[self.target].mean() if len(sub) else float("nan")
            return len(sub), 100 * len(sub) / n_total, lgd

        def rotulo(sid):
            seg = self.segments[sid]
            if seg["parent"] is None:
                return "TODA A CARTEIRA"
            return self._descrever([seg["conditions"][-1]])

        def lgd_de(sid):
            sub = self.df[self.segments[sid]["mask"]]
            return sub[self.target].mean() if len(sub) else float("inf")

        def rec(sid, prefix, is_last, is_root=False):
            n, rep, lgd = stats(sid)
            seg = self.segments[sid]
            conn = "" if is_root else ("└─ " if is_last else "├─ ")
            tag = f"  [nota {nota_map.get(sid, '?')}]" if seg["is_leaf"] else ""
            linhas.append(f"{prefix}{conn}{rotulo(sid)}  "
                          f"(n={n}, {rep:.1f}%, LGD={lgd:.4f}){tag}")
            ch = sorted(filhos.get(sid, []), key=lgd_de, reverse=not ascending)
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
    # PLOT_TREE: desenha a árvore como IMAGEM (matplotlib). Cada nó mostra o
    #   rótulo da condição, n, % e LGD médio; folhas trazem a nota. A cor do nó
    #   reflete o LGD (verde = baixo → vermelho = alto). Devolve a Figure e,
    #   se `save_path` for dado, salva a imagem (PNG/SVG/PDF…).
    # ------------------------------------------------------------------
    def plot_tree(self, ascending: bool = True, figsize=None, cmap: str = "RdYlGn_r",
                  show_samples: bool = False, title: str | None = "Segmentação de LGD",
                  save_path: str | None = None, dpi: int = 150, ax=None):
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
                lgd = sr.mean() if len(sr) else (sub[self.target].mean() if n else float("nan"))
            else:
                lgd = sub[self.target].mean() if n else float("nan")
            return n, (100 * n / n_total if n_total else 0.0), lgd

        def sample_lgd(sid, a):
            m = self.segments[sid]["mask"] & (self.df[self.sample_col] == a)
            sub = self.df[m]
            return sub[self.target].mean() if len(sub) else float("nan")

        def lgd_sort(sid):
            lg = stats(sid)[2]
            return lg if not pd.isna(lg) else float("inf")

        # --- layout: folhas em x sequencial (DFS), internos centralizados ---
        X_GAP, Y_GAP = 2.4, 2.15        # espaçamento entre folhas / entre níveis
        bw, bh = 0.97, 0.74             # meia-largura / meia-altura do box (dados)
        pos: dict = {}
        counter = [0]
        max_depth = [0]

        def place(sid, depth):
            max_depth[0] = max(max_depth[0], depth)
            ch = sorted(filhos.get(sid, []), key=lgd_sort, reverse=not ascending)
            if self.segments[sid]["is_leaf"] or not ch:
                x = counter[0] * X_GAP
                counter[0] += 1
            else:
                x = float(np.mean([place(c, depth + 1) for c in ch]))
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
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.figure

        leaf_lgds = [stats(s)[2] for s, v in self.segments.items() if v["is_leaf"]]
        leaf_lgds = [v for v in leaf_lgds if not pd.isna(v)]
        lo = min(leaf_lgds) if leaf_lgds else 0.0
        hi = max(leaf_lgds) if leaf_lgds else 1.0
        if hi <= lo:
            hi = lo + 1e-6
        norm = Normalize(lo, hi)
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

        for sid, (x, y) in pos.items():
            n, rep, lgd = stats(sid)
            color = cmap_obj(norm(lgd)) if not pd.isna(lgd) else (0.88, 0.88, 0.88, 1.0)
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            txt_color = "#15324a" if lum > 0.6 else "#ffffff"
            ax.add_patch(FancyBboxPatch(
                (x - bw, y - bh), 2 * bw, 2 * bh,
                boxstyle="round,pad=0.02,rounding_size=0.14",
                linewidth=1.3, edgecolor="#33424f", facecolor=color, zorder=2))
            is_leaf = self.segments[sid]["is_leaf"]
            cab = "\n".join(textwrap.wrap(rotulo(sid), 22)[:2])
            nota = f"   ·   nota {nota_map.get(sid, '?')}" if is_leaf else ""
            linhas = [cab,
                      f"n={n:,} · {rep:.1f}%".replace(",", "."),
                      f"LGD {lgd:.3f}{nota}" if not pd.isna(lgd) else f"LGD —{nota}"]
            if show_samples and self.sample_col is not None:
                amostras = list(self.df[self.sample_col].dropna().unique())
                linhas.append(" | ".join(f"{a} {sample_lgd(sid, a):.3f}" for a in amostras))
            weight = "bold" if is_leaf else "normal"
            ax.text(x, y, "\n".join(linhas), ha="center", va="center",
                    fontsize=8.4, color=txt_color, zorder=3, linespacing=1.3,
                    fontweight=weight)

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
        cbar.set_label(f"LGD médio{' (DES)' if ref else ''}", fontsize=9)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

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
            skip = {self.target, self.sample_col}
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
    # METRICS: avalia a régua como um modelo de LGD.
    #   Predição de cada linha = LGD médio do seu segmento na amostra de
    #   referência (DES). Retorna MAE, RMSE e R² por amostra (DES, OOT, ...).
    # ------------------------------------------------------------------
    def metrics(self) -> pd.DataFrame:
        leaf_ids = [sid for sid, s in self.segments.items() if s["is_leaf"]]
        if self.sample_col is not None:
            ref_mask = self.df[self.sample_col] == self.ref_sample
        else:
            ref_mask = pd.Series(True, index=self.df.index)
        overall = self.df.loc[ref_mask, self.target].mean()

        # predição por linha = média do segmento na referência
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
            y = self.df.loc[mask, self.target].to_numpy()
            yhat = pred[mask.values].to_numpy()
            err = y - yhat
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            linhas.append({
                "amostra": nome,
                "n": int(mask.sum()),
                "MAE": round(float(np.mean(np.abs(err))), 4),
                "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 4),
                "R2": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else np.nan,
            })
        return pd.DataFrame(linhas)

    # ------------------------------------------------------------------
    # BOOTSTRAP_CI: intervalo de confiança da média de LGD por folha, via
    #   reamostragem bootstrap na amostra `sample` (default = referência/DES).
    #   Se houver `check_sample` (default = 1ª não-referência, ex. OOT), traz o
    #   LGD médio dela por folha e verifica a ADERÊNCIA: se o LGD de OOT cai
    #   dentro do IC bootstrap do DES (estável) ou fora (acima/abaixo = alerta).
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
            vals = self.df.loc[m_s, self.target].to_numpy()
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
                "nota_lgd": r["nota_lgd"],
                "descricao": r["descricao"],
                "n": int(n),
                f"lgd_{sample or 'todos'}": round(pt, 4) if not np.isnan(pt) else np.nan,
                "ic_low": round(float(lo), 4) if not np.isnan(lo) else np.nan,
                "ic_high": round(float(hi), 4) if not np.isnan(hi) else np.nan,
                "amplitude": round(float(hi - lo), 4) if not np.isnan(hi) else np.nan,
            }
            if check_sample is not None:
                m_c = m & (self.df[self.sample_col] == check_sample)
                cvals = self.df.loc[m_c, self.target].to_numpy()
                lgd_c = float(cvals.mean()) if len(cvals) else np.nan
                row[f"lgd_{check_sample}"] = round(lgd_c, 4) if not np.isnan(lgd_c) else np.nan
                if np.isnan(lgd_c) or np.isnan(lo):
                    row["aderente"] = None
                    row["status_oot"] = "—"
                else:
                    dentro = bool(lo <= lgd_c <= hi)
                    row["aderente"] = dentro
                    row["status_oot"] = ("dentro" if dentro
                                         else "acima" if lgd_c > hi else "abaixo")
            rows.append(row)

        out = pd.DataFrame(rows)
        out.attrs.update(sample=sample, check_sample=check_sample, ci=ci, n_boot=n_boot)
        return out

    # ------------------------------------------------------------------
    # VARIABLE_IV: Information Value de cada variável candidata em relação à
    #   folha `sid`, para indicar qual variável usar no próximo split.
    #   Como o LGD é contínuo, o alvo é binarizado (evento = LGD ≥ corte,
    #   default = mediana da folha) e o IV é calculado via WoE sobre os MESMOS
    #   bins ótimos que seriam usados no split. Ordena por IV desc.
    # ------------------------------------------------------------------
    def variable_iv(self, sid: str | None = None, features: list | None = None,
                    max_n_bins: int = 5, min_bin_size: float = 0.05,
                    cutoff="median") -> pd.DataFrame:
        if features is None:
            skip = {self.target, self.sample_col}
            features = [c for c in self.df.columns if c not in skip]

        if sid is None or sid == "root" or sid not in self.segments:
            leaf_mask = pd.Series(True, index=self.df.index)
        else:
            leaf_mask = self.segments[sid]["mask"]
        ref_mask = (self.df[self.sample_col] == self.ref_sample
                    if self.sample_col is not None else pd.Series(True, index=self.df.index))

        base = self.df[leaf_mask & ref_mask]            # porção DES da folha
        if len(base) >= 4 and base[self.target].nunique() > 1:
            thr = base[self.target].median() if cutoff == "median" else float(cutoff)
            event = (base[self.target].to_numpy() >= thr).astype(int)
            N1, N0 = int(event.sum()), int(len(event) - event.sum())
        else:
            thr, event, N1, N0 = np.nan, None, 0, 0

        rows = []
        for feat in features:
            iv, nb, kind = np.nan, 0, "—"
            if N1 > 0 and N0 > 0:
                try:
                    bins, _modo, kind = self._resolve_bins(
                        self.df[leaf_mask], feat, None, None, max_n_bins, min_bin_size)
                    nb = len(bins)
                    if bins:
                        iv = 0.0
                        for b in bins:
                            m = self._mask_in(base, feat, b).to_numpy()
                            n1, n0 = int(event[m].sum()), int((1 - event[m]).sum())
                            p1, p0 = max(n1 / N1, 1e-6), max(n0 / N0, 1e-6)
                            iv += (n1 / N1 - n0 / N0) * np.log(p1 / p0)
                except Exception:
                    iv = np.nan
            rows.append({
                "variavel": feat, "tipo": kind, "n_bins": nb,
                "iv": round(float(iv), 4) if not (iv is None or np.isnan(iv)) else np.nan,
                "forca": _classifica_iv(iv),
            })
        out = (pd.DataFrame(rows)
               .sort_values("iv", ascending=False, na_position="last")
               .reset_index(drop=True))
        out.attrs["cutoff"] = (round(float(thr), 4) if not np.isnan(thr) else np.nan)
        return out


    # ------------------------------------------------------------------
    # ASSIGN: rotula cada linha com seu segmento-folha
    #   Por padrão adiciona também a nota de LGD (1..N) e a descrição
    #   por extenso. Colunas: <col>, <col>_nota, <col>_desc
    # ------------------------------------------------------------------
    def assign(
        self,
        col_name: str = "segmento_lgd",
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
    SCHEMA = "yggdrasil.credit_risk.lgd.tree/1"

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
        seg = cls(df, target=meta.get("target", "lgd"),
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
        """Régua final: folhas com nota, condições e LGD (na referência/DES)."""
        nota_map, _ = self._grade_map()
        leaves = []
        for sid, seg in self.segments.items():
            if not seg["is_leaf"]:
                continue
            sub = self.df[seg["mask"]]
            if self.sample_col is not None:
                ref = sub.loc[sub[self.sample_col] == self.ref_sample, self.target]
                lgd = float(ref.mean()) if len(ref) else float(sub[self.target].mean())
            else:
                lgd = float(sub[self.target].mean())
            leaves.append({"id": sid, "nota": int(nota_map[sid]),
                           "conditions": self._conditions_json(seg["conditions"]),
                           "lgd": round(lgd, 6)})
        leaves.sort(key=lambda x: x["nota"])
        return {"target": self.target, "ref_sample": self.ref_sample, "leaves": leaves}

    def predict(self, X: pd.DataFrame, col_seg="segmento_lgd",
                col_nota="nota_lgd", col_lgd="lgd_regua") -> pd.DataFrame:
        """Aplica a régua a um DataFrame pandas novo: segmento, nota e LGD."""
        return _aplicar_regua_pandas(self._regua_dict(), X, col_seg, col_nota, col_lgd)

    # ------------------------------------------------------------------
    # TO_PYSPARK: gera o código da régua como F.when().otherwise() para
    #   aplicar a segmentação (segmento + nota + LGD) em escala no Spark.
    # ------------------------------------------------------------------
    def to_pyspark(self, func_name: str = "aplicar_regua_lgd") -> str:
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
                    expr = f'F.col("{feat}").isin({cats})'
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
             f"def {func_name}(df, col_seg='segmento_lgd', col_nota='nota_lgd', col_lgd='lgd_regua'):",
             '    """Régua de LGD gerada por SequentialLGDSegmenter (segmento, nota e LGD por folha)."""']
        for i, leaf in enumerate(regua["leaves"], 1):
            L.append(f'    c{i} = {cond_expr(leaf["conditions"])}')
        L.append("    seg = (")
        L.append(chain(lambda lf: f'F.lit({lf["id"]!r})'))
        L.append("    )")
        L.append("    nota = (")
        L.append(chain(lambda lf: f'F.lit({lf["nota"]})'))
        L.append("    )")
        L.append("    lgd = (")
        L.append(chain(lambda lf: f'F.lit({lf["lgd"]})'))
        L.append("    )")
        L.append("    return (df.withColumn(col_seg, seg)")
        L.append("              .withColumn(col_nota, nota)")
        L.append("              .withColumn(col_lgd, lgd))")
        return "\n".join(L)

    # ------------------------------------------------------------------
    # SUGGEST_SPLIT: recomenda a melhor variável para dividir uma folha,
    #   pelo maior Information Value, e devolve o ranking completo.
    # ------------------------------------------------------------------
    def suggest_split(self, sid: str | None = None, features: list | None = None,
                      max_n_bins: int = 4, min_bin_size: float = 0.05) -> dict:
        if sid is None:
            sid = "root"
        iv = self.variable_iv(sid, features=features, max_n_bins=max_n_bins,
                              min_bin_size=min_bin_size)
        if len(iv) == 0 or pd.isna(iv.iloc[0]["iv"]):
            return {"sid": sid, "feature": None, "iv": None, "forca": None,
                    "ranking": iv, "msg": "nenhuma variável informativa para esta folha"}
        top = iv.iloc[0]
        return {"sid": sid, "feature": top["variavel"], "tipo": top["tipo"],
                "iv": float(top["iv"]), "forca": top["forca"], "ranking": iv,
                "msg": f"dividir por '{top['variavel']}' (IV={top['iv']:.4f}, {top['forca']})"}

    # ------------------------------------------------------------------
    # FIT_AUTO: constrói uma árvore inicial de forma gulosa — em cada folha
    #   escolhe a variável de maior IV e divide com binning ótimo, até a
    #   profundidade máxima ou IV abaixo do mínimo. Ponto de partida para
    #   refinar à mão depois.
    # ------------------------------------------------------------------
    def fit_auto(self, features: list | None = None, max_depth: int = 3,
                 min_iv: float = 0.02, max_n_bins: int = 2, min_bin_size: float = 0.05,
                 from_scratch: bool = True, verbose: bool = True):
        import io
        import contextlib
        if from_scratch:
            root = self.segments["root"]
            root["is_leaf"] = True
            self.segments = {"root": root}
        for depth in range(max_depth):
            atuais = [sid for sid, s in self.segments.items()
                      if s["is_leaf"] and s["depth"] == depth]
            for sid in atuais:
                if sid in self.segments and not self.segments[sid]["is_leaf"]:
                    continue
                sub = self.df[self.segments[sid]["mask"]]
                if len(self._fit_frame(sub, min_bin_size)) < self.min_leaf_rows:
                    continue
                iv = self.variable_iv(sid, features=features, max_n_bins=max_n_bins,
                                      min_bin_size=min_bin_size)
                if len(iv) == 0:
                    continue
                top = iv.iloc[0]
                if pd.isna(top["iv"]) or top["iv"] < min_iv:
                    continue
                with contextlib.redirect_stdout(io.StringIO()):
                    self.grow(top["variavel"], only_segments=[sid],
                              max_n_bins=max_n_bins, min_bin_size=min_bin_size)
        n_folhas = sum(s["is_leaf"] for s in self.segments.values())
        prof = max(s["depth"] for s in self.segments.values())
        if verbose:
            print(f"[fit_auto] árvore gulosa construída: profundidade {prof} "
                  f"(máx {max_depth}), IV mínimo {min_iv} → {n_folhas} folhas")
        return self

    # ------------------------------------------------------------------
    # LOG_TO_MLFLOW: salva a segmentação no MLflow — parâmetros, métricas,
    #   artefatos (folhas, árvore, régua PySpark, régua JSON) e o MODELO
    #   pyfunc (régua aplicável via .predict para scoring).
    # ------------------------------------------------------------------
    def log_to_mlflow(self, experiment: str | None = None,
                      run_name: str | None = None, registered_model_name: str | None = None,
                      artifact_path: str = "modelo_lgd", extra_params: dict | None = None,
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
            metrics[f"lgd_nota_{leaf['nota']}"] = leaf["lgd"]
        try:
            for _, r in self.metrics().iterrows():
                a = r["amostra"]
                for k in ("MAE", "RMSE", "R2"):
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

        class _ReguaLGDModel(mlflow.pyfunc.PythonModel):
            def __init__(self, regua):
                self.regua = regua

            def predict(self, context, model_input):
                import numpy as _np
                import pandas as _pd
                df = model_input
                seg = _pd.Series(_pd.NA, index=df.index, dtype="object")
                nota = _pd.Series(_pd.NA, index=df.index, dtype="Int64")
                lgd = _pd.Series(_np.nan, index=df.index, dtype="float64")
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
                    lgd[m] = leaf["lgd"]
                return _pd.DataFrame({"segmento_lgd": seg, "nota_lgd": nota, "lgd_regua": lgd},
                                     index=df.index)

        # Unity Catalog exige set_registry_uri('databricks-uc') e assinatura do modelo
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        if experiment:
            mlflow.set_experiment(experiment)

        model = _ReguaLGDModel(regua)
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
