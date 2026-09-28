"""Orquestrador da seleção de features por *book* (pandas ou PySpark).

O backend vem do tipo do DataFrame recebido (ver :mod:`backend`): pandas roda tudo
no driver, Spark distribui. A lógica abaixo é a mesma nos dois casos — só a camada
de estatística muda.

`run_feature_selection` roda, **book a book**, o pipeline:

1. descarta features com missing acima do limiar;
2. descarta features sem variância (P1==P99) / quase-constantes;
3. calcula importância (RF + univariadas) sobre as sobreviventes;
4. remove redundância (|corr| alta), mantendo o representante de cada cluster;
5. roda o Boruta nas representantes;
6. consolida tudo num **consenso** (`selecionada` + `motivo`).

No fim, monta um :class:`FeatureSelectionReport` com a tabela por book, a lista de
selecionadas, um ranking **global** das mais importantes (recalculado sobre todas
as selecionadas) e os painéis gráficos. É um entrypoint INDEPENDENTE — não entra
no pipeline de modelo.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import ColumnConfig
from ..utils import get_logger
from . import plots
from .backend import backend_name, is_pandas
from .books import BooksSpec, Book, resolve_books
from .boruta import boruta_select
from .config import FeatureSelectionConfig
from .encoding import apply_encodings, fit_encodings
from .importance import importance_indicators
from .spark_stats import (
    _require_functions,
    correlation_matrices,
    missing_rate,
    numeric_columns,
    redundancy_clusters,
    variance_flags,
)

_logger = get_logger("yggdrasil.feature_selection")

# Ordem canônica das colunas da tabela de seleção (alinhada entre books).
_COLS = [
    "book", "feature", "pct_missing", "p_low", "p_high", "top1_share",
    "sem_variancia", "near_constante", "rf_importance", "iv", "ks", "auc", "gini",
    "corr_target", "score", "leakage_flag", "cluster", "representante",
    "redundante_com", "boruta_hits", "boruta_decisao", "score_consenso",
    "selecionada", "motivo", "encoding",
]

MOTIVO_NAO_NUMERICA = "não numérica (não avaliada)"


class _Progresso:
    """Prints de andamento da seleção (``verbose=True``).

    ``print`` e não ``logging``: no Databricks/Jupyter o logger costuma não aparecer
    na célula; ``flush`` faz cada linha surgir ao vivo, não só no fim.
    """

    N_ETAPAS = 6

    def __init__(self, ativo: bool):
        self.ativo = ativo
        self.t0 = self.t = time.perf_counter()

    def msg(self, texto: str) -> None:
        if self.ativo:
            print(texto, flush=True)

    def marca(self) -> None:
        self.t = time.perf_counter()

    def inicio(self, i: int, nome: str, detalhe: str) -> None:
        """Anuncia uma etapa lenta antes de rodar (senão a célula parece travada)."""
        self.msg(f"  [{i}/{self.N_ETAPAS}] {nome}: {detalhe}...")

    def etapa(self, i: int, nome: str, detalhe: str) -> None:
        agora = time.perf_counter()
        dt, self.t = agora - self.t, agora
        self.msg(f"  [{i}/{self.N_ETAPAS}] {nome}: {detalhe} ({dt:.1f}s)")

    @staticmethod
    def funil(antes: int, depois: int) -> str:
        cortadas = antes - depois
        return f"{antes} → {depois}" + (f" (−{cortadas})" if cortadas else "")

    def total(self) -> float:
        return time.perf_counter() - self.t0


def _infer_problem_type(sdf, cfg: ColumnConfig) -> str:
    """Heurística: alvo binário {0,1} => classification, senão regression."""
    if is_pandas(sdf):
        from .pandas_stats import infer_problem_type
        return infer_problem_type(sdf, cfg)
    F = _require_functions()
    distintos = [r[0] for r in (sdf.select(cfg.target_col)
                                .where(F.col(cfg.target_col).isNotNull())
                                .distinct().limit(3).collect())]
    try:
        vals = {float(v) for v in distintos}
    except (TypeError, ValueError):
        return "classification"
    return "classification" if len(distintos) <= 2 and vals <= {0.0, 1.0} else "regression"


def _consensus(row: dict, fs_cfg: FeatureSelectionConfig) -> Tuple[float, bool, str]:
    """Score de consenso ∈ [0,1] + decisão de seleção + motivo, para uma feature viva."""
    comps: List[Tuple[float, float]] = []
    imp_norm = row.get("imp_norm", np.nan)
    if np.isfinite(imp_norm):
        comps.append((fs_cfg.peso_importancia, imp_norm))
    if fs_cfg.boruta_enable:
        hr = row.get("boruta_hit_rate", np.nan)
        if np.isfinite(hr):
            comps.append((fs_cfg.peso_boruta, hr))
    corr = row.get("corr_target", np.nan)
    if np.isfinite(corr):
        comps.append((fs_cfg.peso_alvo, min(abs(corr), 1.0)))
    W = sum(w for w, _ in comps)
    score = sum(w * v for w, v in comps) / W if W > 0 else np.nan

    if bool(row.get("leakage_flag")):
        return score, False, "suspeita de leakage (revisar)"
    dec = row.get("boruta_decisao")
    por_consenso = np.isfinite(score) and score >= fs_cfg.consensus_threshold
    if dec == "confirmada":
        return score, True, "selecionada (Boruta confirmada)"
    if por_consenso:
        extra = "; Boruta rejeitou" if dec == "rejeitada" else ""
        return score, True, f"selecionada (consenso{extra})"
    if dec == "rejeitada":
        return score, False, "Boruta rejeitada"
    return score, False, "consenso abaixo do limiar"


def _process_book(
    base, book: Book, target: str, problem_type: str, fs_cfg: FeatureSelectionConfig,
    prog: Optional[_Progresso] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Roda o pipeline de seleção num único book. Retorna (tabela, corr_spearman)."""
    prog = prog or _Progresso(False)
    prog.marca()
    feats = list(book.features)
    rec: Dict[str, dict] = {f: {"book": book.name, "feature": f} for f in feats}

    # 1) missing ---------------------------------------------------------
    miss = missing_rate(base, feats)
    for f in feats:
        rec[f]["pct_missing"] = round(float(miss[f]), 4) if np.isfinite(miss[f]) else np.nan
    vivos = [f for f in feats if not (np.isfinite(miss[f]) and miss[f] > fs_cfg.missing_max)]
    for f in feats:
        if f not in vivos:
            rec[f].update(selecionada=False, motivo="alto missing")
    prog.etapa(1, "Missing", f"{prog.funil(len(feats), len(vivos))} "
                             f"· corte > {fs_cfg.missing_max:.0%} vazio")

    # 2) variância -------------------------------------------------------
    if vivos:
        var = variance_flags(base, vivos, fs_cfg).set_index("feature")
        for f in vivos:
            rec[f].update(p_low=var.loc[f, "p_low"], p_high=var.loc[f, "p_high"],
                          top1_share=var.loc[f, "top1_share"],
                          sem_variancia=bool(var.loc[f, "sem_variancia"]),
                          near_constante=bool(var.loc[f, "near_constante"]))
        vivos2 = [f for f in vivos if not (var.loc[f, "sem_variancia"] or var.loc[f, "near_constante"])]
        for f in vivos:
            if f not in vivos2:
                motivo = "sem variância" if var.loc[f, "sem_variancia"] else "quase-constante"
                rec[f].update(selecionada=False, motivo=motivo)
    else:
        vivos2 = []
    prog.etapa(2, "Variância", prog.funil(len(vivos), len(vivos2)))

    # 3) importância -----------------------------------------------------
    score = pd.Series(dtype=float)
    metricas = "RandomForest + IV/KS/AUC" if problem_type == "classification" else "RandomForest + correlação"
    if vivos2:
        prog.inicio(3, "Importância", f"{metricas} em {len(vivos2)} feature(s)")
        imp = importance_indicators(base, vivos2, target, problem_type, fs_cfg).set_index("feature")
        for f in vivos2:
            for c in ("rf_importance", "iv", "ks", "auc", "gini", "corr_target", "score", "leakage_flag"):
                if c in imp.columns:
                    rec[f][c] = imp.loc[f, c]
        score = imp["score"] if "score" in imp.columns else pd.Series(dtype=float)
        n_leak = int(imp["leakage_flag"].eq(True).sum()) if "leakage_flag" in imp else 0
        prog.etapa(3, "Importância", "calculada" + (f"; {n_leak} suspeita(s) de leakage" if n_leak else ""))
    else:
        prog.etapa(3, "Importância", "nada a avaliar")

    # 4) redundância -----------------------------------------------------
    corr_sp = pd.DataFrame()
    if vivos2:
        cm = correlation_matrices(base, vivos2, fs_cfg)
        corr_sp = cm["spearman"]
        red = redundancy_clusters(corr_sp, fs_cfg.corr_high, importance=score).set_index("feature")
        reps = []
        for f in vivos2:
            if f in red.index:
                rec[f].update(cluster=int(red.loc[f, "cluster"]),
                              representante=bool(red.loc[f, "representante"]),
                              redundante_com=red.loc[f, "redundante_com"])
                is_rep = bool(red.loc[f, "representante"])
            else:  # não-numérica (fora da matriz de correlação) — sem redundância aferível
                rec[f].update(representante=True)
                is_rep = True
            if is_rep:
                reps.append(f)
            else:
                rec[f].update(selecionada=False,
                              motivo=f"redundante c/ {red.loc[f, 'redundante_com']}")
        vivos3 = reps
    else:
        vivos3 = []
    prog.etapa(4, "Redundância", f"{prog.funil(len(vivos2), len(vivos3))} "
                                 f"· corte |corr| > {fs_cfg.corr_high:g}")

    # 5) Boruta ----------------------------------------------------------
    num_boruta = numeric_columns(base, vivos3) if vivos3 else []
    if fs_cfg.boruta_enable and num_boruta:
        prog.inicio(5, "Boruta", f"{len(num_boruta)} feature(s), até {fs_cfg.boruta_max_iter} iterações")
        bor = boruta_select(base, vivos3, target, problem_type, fs_cfg).set_index("feature")
        for f in vivos3:
            if f in bor.index:
                rec[f].update(boruta_hits=int(bor.loc[f, "hits"]),
                              boruta_decisao=bor.loc[f, "decisao"],
                              boruta_hit_rate=float(bor.loc[f, "hit_rate"]))
        dec = bor["decisao"].value_counts()
        prog.etapa(5, "Boruta", ", ".join(f"{int(dec.get(k, 0))} {k}(s)"
                                          for k in ("confirmada", "tentativa", "rejeitada")))
    else:
        prog.etapa(5, "Boruta", "desligado" if not fs_cfg.boruta_enable else "sem features numéricas")

    # 6) consenso (entre as representantes vivas) ------------------------
    if vivos3 and len(score):
        viv_scores = score.reindex(vivos3)
        imp_norm = viv_scores.rank(pct=True)  # normaliza importância dentro do book
        for f in vivos3:
            rec[f]["imp_norm"] = float(imp_norm.get(f, np.nan))
    numericas = set(numeric_columns(base, vivos3)) if vivos3 else set()
    for f in vivos3:
        if f not in numericas:
            # Sem IV/RF/Boruta: não houve avaliação — dizer isso, e não "consenso baixo".
            rec[f].update(score_consenso=np.nan, selecionada=False, motivo=MOTIVO_NAO_NUMERICA)
            continue
        sc, sel, motivo = _consensus(rec[f], fs_cfg)
        rec[f].update(score_consenso=round(sc, 4) if np.isfinite(sc) else np.nan,
                      selecionada=bool(sel), motivo=motivo)
    n_sel = sum(bool(rec[f].get("selecionada")) for f in feats)
    prog.etapa(6, "Consenso + leakage", prog.funil(len(vivos3), n_sel))
    prog.msg(f"  → book '{book.name}': {n_sel} de {len(feats)} feature(s) selecionada(s)")

    df = pd.DataFrame([rec[f] for f in feats])
    df = df.reindex(columns=_COLS)
    if "score" in df.columns:
        df = df.sort_values(["selecionada", "score"], ascending=[False, False], na_position="last")
    return df.reset_index(drop=True), corr_sp


@dataclass
class FeatureSelectionReport:
    """Resultado da seleção de features por book."""

    selection_table: pd.DataFrame
    book_tables: Dict[str, pd.DataFrame]
    selected_features: Dict[str, List[str]]
    selected_overall: List[str]
    overall_importance: pd.DataFrame
    panels: Dict[str, object] = field(default_factory=dict)
    problem_type: Optional[str] = None
    cfg: Optional[ColumnConfig] = None
    # Correlação (Spearman) das features selecionadas — cross-book, para as análises
    # pós-seleção (a redundância do pipeline é por book; esta matriz pega o resto).
    overall_correlation: pd.DataFrame = field(default_factory=pd.DataFrame)
    fs_cfg: Optional[FeatureSelectionConfig] = None
    # Codificação aprendida p/ as não numéricas (``encode_categoricals=True``):
    # {feature: especificação}. Reaplique em outras bases com ``apply_encodings``.
    encodings: Dict[str, dict] = field(default_factory=dict)

    def apply_encodings(self, df):
        """Aplica a mesma codificação da seleção a outra base (OOT, escoragem, treino).

        Aceita pandas ou Spark; mantém os nomes das colunas. Categoria não vista na
        seleção cai em ``OUTROS``.
        """
        return apply_encodings(df, self.encodings)

    def to_csv(self, path: str) -> str:
        self.selection_table.to_csv(path, index=False)
        return path

    def to_html(self, embed_panels: bool = False) -> str:
        return _to_html(self, embed_panels=embed_panels)

    def summary(self) -> pd.DataFrame:
        """Resumo por book: nº de features, selecionadas e descartadas."""
        rows = []
        for name, t in self.book_tables.items():
            rows.append({"book": name, "n_features": len(t),
                         "selecionadas": int(t["selecionada"].eq(True).sum()),
                         "descartadas": int((~t["selecionada"].eq(True)).sum())})
        return pd.DataFrame(rows)


def run_feature_selection(
    sdf,
    cfg: Optional[ColumnConfig] = None,
    fs_cfg: Optional[FeatureSelectionConfig] = None,
    books: BooksSpec = None,
    problem_type: Optional[str] = None,
    with_panels: bool = True,
    mlflow_experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    verbose: bool = True,
) -> FeatureSelectionReport:
    """Roda a seleção de features ponta a ponta sobre um DataFrame pandas ou Spark.

    Parameters
    ----------
    sdf:
        DataFrame **pandas ou Spark** com o contrato de colunas de ``cfg`` (features
        ``feat_*``, alvo, e — opcionalmente — coluna de amostra). O backend é escolhido
        pelo tipo: pandas roda tudo no driver (sem exigir pyspark instalado), Spark
        distribui. O relatório de saída é o mesmo nos dois casos.
    books:
        Definição dos books (ver :func:`yggdrasil.feature_selection.resolve_books`).
        Padrão (None): auto-deriva pelo 1º segmento após o prefixo.
    verbose:
        Imprime o andamento de cada etapa (quantas features entram e saem e o tempo
        gasto). ``False`` silencia.
    """
    prog = _Progresso(verbose)
    local = is_pandas(sdf)
    F = None if local else _require_functions()
    cfg = cfg or ColumnConfig()
    fs_cfg = fs_cfg or FeatureSelectionConfig()
    if cfg.target_col not in sdf.columns:
        raise ValueError(f"Coluna de alvo '{cfg.target_col}' ausente no DataFrame.")
    _logger.debug("Seleção de features no backend '%s'.", backend_name(sdf))

    books_res = resolve_books(sdf, cfg, books)
    if problem_type is None:
        problem_type = _infer_problem_type(sdf, cfg)

    # Usa só a amostra de desenvolvimento p/ a seleção, se a coluna existir.
    base = sdf
    if cfg.sample_col in sdf.columns:
        dev = (sdf[sdf[cfg.sample_col] == cfg.dev_sample] if local
               else sdf.where(F.col(cfg.sample_col) == cfg.dev_sample))
        if (not dev.empty) if local else bool(dev.head(1)):
            base = dev
            _logger.debug("Seleção restrita à amostra de desenvolvimento '%s'.", cfg.dev_sample)

    # Não numéricas: codifica (opt-in) na própria base da seleção, ou avisa que ficam de fora.
    todas = list(dict.fromkeys(f for b in books_res for f in b.features))
    amostra = f"amostra '{cfg.dev_sample}'" if base is not sdf else "base inteira"
    tipo = {"classification": "classificação", "regression": "regressão"}.get(problem_type, problem_type)
    prog.msg(f"Seleção de features · backend {backend_name(sdf)} · {tipo} · "
             f"{len(books_res)} book(s), {len(todas)} feature(s) · {amostra}")
    encodings: Dict[str, dict] = {}
    if fs_cfg.encode_categoricals:
        encodings = fit_encodings(base, todas, cfg.target_col, fs_cfg)
        base = apply_encodings(base, encodings)
        if encodings:
            prog.msg(f"  Codificação: {len(encodings)} feature(s) não numérica(s) → "
                     + ", ".join(f"{f} ({'0/1' if e['tipo'] == 'bool_0_1' else 'target encoding'})"
                                 for f, e in encodings.items()))
    nao_num = [f for f in todas if f not in set(numeric_columns(base, todas))]
    if nao_num:
        dica = ("" if fs_cfg.encode_categoricals
                else " Use FeatureSelectionConfig(encode_categoricals=True) para codificá-las.")
        _logger.warning("%d feature(s) não numérica(s) ficam fora da avaliação: %s.%s",
                        len(nao_num), nao_num, dica)
        prog.msg(f"  Atenção: {len(nao_num)} feature(s) não numérica(s) não serão avaliadas: "
                 f"{nao_num}.{dica}")
    if not local:
        base = base.cache()
    try:
        n_linhas = len(base) if local else base.count()
        prog.msg(f"  {n_linhas:,} linha(s) na base da seleção".replace(",", "."))
        book_tables: Dict[str, pd.DataFrame] = {}
        corr_by_book: Dict[str, pd.DataFrame] = {}
        for i, book in enumerate(books_res, 1):
            _logger.debug("Processando book '%s' (%d features)...", book.name, len(book))
            prog.msg(f"\n▸ Book {i}/{len(books_res)} '{book.name}' ({len(book)} feature(s))")
            tbl, corr_sp = _process_book(base, book, cfg.target_col, problem_type, fs_cfg, prog)
            book_tables[book.name] = tbl
            corr_by_book[book.name] = corr_sp

        tipos_enc = {f: e["tipo"] for f, e in encodings.items()}
        for t in book_tables.values():
            t["encoding"] = t["feature"].map(tipos_enc)
        selection_table = pd.concat(book_tables.values(), ignore_index=True) if book_tables else pd.DataFrame(columns=_COLS)
        selected_features = {
            name: t.loc[t["selecionada"].eq(True), "feature"].tolist()
            for name, t in book_tables.items()
        }
        selected_overall = [f for feats in selected_features.values() for f in feats]

        # Ranking GLOBAL: recalcula importância sobre todas as selecionadas juntas.
        prog.msg(f"\n▸ Ranking global e correlação das {len(selected_overall)} selecionada(s)...")
        prog.marca()
        if selected_overall and numeric_columns(base, selected_overall):
            overall = importance_indicators(base, selected_overall, cfg.target_col, problem_type, fs_cfg)
            book_de = selection_table.set_index("feature")["book"].to_dict()
            overall.insert(1, "book", overall["feature"].map(book_de))
        else:
            overall = pd.DataFrame(columns=["feature", "book", "rf_importance", "score"])

        # Correlação dos SOBREVIVENTES (uma passada Spark; vazia se < 2 numéricas).
        if selected_overall and len(numeric_columns(base, selected_overall)) >= 2:
            overall_corr = correlation_matrices(base, selected_overall, fs_cfg)["spearman"]
        else:
            overall_corr = pd.DataFrame()
        prog.msg(f"  concluído ({time.perf_counter() - prog.t:.1f}s)")
    finally:
        if not local:
            base.unpersist()

    panels: Dict[str, object] = {}
    if with_panels:
        prog.msg("▸ Gerando os gráficos (report.panels)...")
        prog.marca()
        panels["overview"] = plots.plot_book_overview(selection_table)
        panels["overall_importance"] = plots.plot_overall_importance(overall, fs_cfg.top_k_overall)

        # ── análises pós-seleção (facilitam a vida depois da seleção) ────────
        panels["post_selection_dashboard"] = plots.plot_post_selection_dashboard(
            selection_table, problem_type, fs_cfg.boruta_max_iter)
        panels["funnel"] = plots.plot_selection_funnel(selection_table)
        panels["decision_map"] = plots.plot_decision_map(selection_table, problem_type)
        panels["book_power"] = plots.plot_book_power_contribution(selection_table)
        panels["cluster_redundancy"] = plots.plot_cluster_redundancy(selection_table)
        panels["leakage_audit"] = plots.plot_leakage_audit(
            selection_table, problem_type, fs_cfg.leakage_auc, fs_cfg.iv_leakage)
        panels["survivor_scorecard"] = plots.plot_survivor_scorecard(selection_table)
        if (fs_cfg.boruta_enable and "boruta_hits" in selection_table.columns
                and selection_table["boruta_hits"].notna().any()):
            panels["boruta"] = plots.plot_boruta_significance(
                selection_table, fs_cfg.boruta_max_iter, fs_cfg.boruta_alpha)
        if problem_type == "classification":
            panels["power_quadrant"] = plots.plot_power_quadrant_iv_ks(
                selection_table, fs_cfg.iv_min, 0.10, fs_cfg.iv_leakage)
        if not overall_corr.empty and len(overall_corr) >= 2:
            panels["survivor_correlation"] = plots.plot_survivor_corr_heatmap(
                overall_corr, selection_table, fs_cfg.corr_high)

        for name, t in book_tables.items():
            panels[f"book::{name}"] = plots.plot_book_selection(t, name, fs_cfg.top_k_book)
            if not corr_by_book[name].empty and len(corr_by_book[name]) >= 2:
                panels[f"corr::{name}"] = plots.plot_corr_heatmap(corr_by_book[name], f"Correlação · {name}")
        prog.msg(f"  {len(panels)} gráfico(s) ({time.perf_counter() - prog.t:.1f}s)")

    report = FeatureSelectionReport(
        selection_table=selection_table, book_tables=book_tables,
        selected_features=selected_features, selected_overall=selected_overall,
        overall_importance=overall, panels=panels, problem_type=problem_type, cfg=cfg,
        overall_correlation=overall_corr, fs_cfg=fs_cfg, encodings=encodings,
    )
    if mlflow_experiment:
        prog.msg(f"▸ Registrando no MLflow (experimento '{mlflow_experiment}')...")
        _log_mlflow(report, mlflow_experiment, run_name)
    prog.msg(f"\n✓ Seleção concluída em {prog.total():.1f}s · {len(selected_overall)} de "
             f"{len(todas)} feature(s) selecionada(s) → report.selected_overall")
    return report


# ── HTML / MLflow ────────────────────────────────────────────────────────
def _to_html(report: FeatureSelectionReport, embed_panels: bool = False) -> str:
    import base64
    from io import BytesIO

    resumo = report.summary()
    partes = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:Arial,Helvetica,sans-serif;margin:24px;}",
        "table{border-collapse:collapse;margin-bottom:24px;font-size:12px;}",
        "th,td{border:1px solid #ddd;padding:5px 8px;text-align:right;}",
        "th{background:steelblue;color:#fff;}h1,h2,h3{color:#2c3e50;}",
        "</style></head><body>",
        "<h1>Seleção de Features — Yggdrasil</h1>",
        f"<p><b>Tipo de problema:</b> {report.problem_type} &nbsp;|&nbsp; "
        f"<b>Selecionadas (total):</b> {len(report.selected_overall)}</p>",
        "<h2>Resumo por book</h2>", resumo.to_html(index=False, border=0),
        "<h2>Ranking global (selecionadas)</h2>",
        report.overall_importance.head(50).to_html(index=False, border=0),
    ]
    for name, t in report.book_tables.items():
        partes.append(f"<h2>Book: {name}</h2>")
        partes.append(t.to_html(index=False, border=0))
    if embed_panels and report.panels:
        partes.append("<h2>Painéis</h2>")
        for c, fig in report.panels.items():
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
            b64 = base64.b64encode(buf.getvalue()).decode()
            partes.append(f"<h3>{c}</h3><img src='data:image/png;base64,{b64}' style='max-width:100%'/>")
    partes.append("</body></html>")
    return "\n".join(partes)


def _log_mlflow(report: FeatureSelectionReport, experiment: str, run_name: Optional[str]) -> None:
    import tempfile
    import mlflow

    tmp = tempfile.mkdtemp(prefix="yggdrasil_fsel_")
    report.to_csv(os.path.join(tmp, "selection_table.csv"))
    report.overall_importance.to_csv(os.path.join(tmp, "overall_importance.csv"), index=False)
    with open(os.path.join(tmp, "feature_selection.html"), "w", encoding="utf-8") as fh:
        fh.write(report.to_html(embed_panels=False))
    if report.encodings:
        import json
        with open(os.path.join(tmp, "encodings.json"), "w", encoding="utf-8") as fh:
            json.dump(report.encodings, fh, ensure_ascii=False, indent=2)
    pdir = os.path.join(tmp, "panels")
    os.makedirs(pdir, exist_ok=True)
    for c, fig in report.panels.items():
        safe = "".join(ch if ch.isalnum() else "_" for ch in c)
        fig.savefig(os.path.join(pdir, f"{safe}.png"), dpi=90, bbox_inches="tight")

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"framework": "yggdrasil-feature-selection", "stage": "feature_selection"})
        mlflow.log_metric("n_features", int(len(report.selection_table)))
        mlflow.log_metric("n_selecionadas", int(len(report.selected_overall)))
        mlflow.log_metric("n_books", int(len(report.book_tables)))
        mlflow.log_artifacts(tmp, artifact_path="feature_selection")


__all__ = ["run_feature_selection", "FeatureSelectionReport"]
