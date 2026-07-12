"""
yggdrasil.credit_risk._mlflow_report
====================================
Relatório MLflow em **abas** (① Resumo · ② Métricas · ③ Estabilidade),
compartilhado pelos segmentadores de **árvore** e de **modelo**. Loga um artefato
HTML autocontido (CSS + JS inline, sem dependência externa) no run ativo:

* **Resumo** — compara TODOS os runs do experimento (via ``mlflow.search_runs``)
  em **KS**, **RMSE** e **PSI no OOT**; a linha do run atual fica destacada.
* **Métricas** — KS/AUC/Gini/Acurácia/F1 (clf) · MAE/RMSE/R² (reg) por amostra.
* **Estabilidade** — PSI por amostra + ratings/folhas.

Também loga as métricas canônicas ``val_ks``/``val_rmse``/``val_psi`` (amostra de
validação = OOT) para alimentar a aba Resumo, e — opcionalmente — as bases de
treino (DES) e validação (OOT) como artefatos (parquet, com fallback CSV).

Tudo *best-effort*: qualquer falha vira aviso, nunca quebra o ``log_to_mlflow``.
"""
from __future__ import annotations

import html as _html
import os
import tempfile

import numpy as np
import pandas as pd

_CSS = (
    "body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#1f2d3a;"
    "margin:16px;background:#fff}h1{font-size:19px;margin:0 0 2px}"
    ".sub{color:#6b7480;font-size:12.5px;margin:0 0 12px}"
    ".tabs{display:flex;gap:6px;border-bottom:2px solid #e5e9ee;margin-bottom:12px;flex-wrap:wrap}"
    ".tab-btn{border:0;background:none;padding:8px 14px;font-size:13.5px;font-weight:600;"
    "color:#6b7480;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-2px}"
    ".tab-btn.active{color:#0f3d57;border-bottom-color:#0f3d57}"
    ".tab-pane{display:none}.tab-pane.active{display:block}"
    "h3{font-size:13.5px;color:#15324a;margin:14px 0 6px}"
    "table{border-collapse:collapse;font-size:12.5px;margin:0 0 8px}"
    "th,td{border:1px solid #e5e9ee;padding:5px 10px;text-align:center}"
    "th{background:#f4f7f9;font-weight:600;color:#33424f}"
    "tr.hl td{background:#eaf5ee;font-weight:600}"
)
_JS = ("function ygShow(i){var b=document.getElementsByClassName('tab-btn'),"
       "p=document.getElementsByClassName('tab-pane');for(var k=0;k<p.length;k++){"
       "b[k].className='tab-btn'+(k===i?' active':'');"
       "p[k].className='tab-pane'+(k===i?' active':'');}}")


def _fmt(v, nd=4):
    if v is None:
        return "—"
    try:
        if isinstance(v, float) and np.isnan(v):
            return "—"
    except TypeError:
        pass
    if isinstance(v, bool):
        return "sim" if v else "não"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}".replace(",", ".")
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{nd}f}"
    return _html.escape(str(v))


def _table_html(df, hl_col=None, hl_val=None):
    if df is None or len(df) == 0:
        return "<p style='color:#889;font-size:13px'>— sem dados —</p>"
    cols = list(df.columns)
    head = "".join(f"<th>{_html.escape(str(c))}</th>" for c in cols)
    rows = []
    for _, r in df.iterrows():
        hl = (hl_col is not None and str(r.get(hl_col, "")) == str(hl_val))
        tds = "".join(f"<td>{_fmt(r[c])}</td>" for c in cols)
        rows.append(f"<tr class='{'hl' if hl else ''}'>{tds}</tr>")
    return (f"<div style='overflow-x:auto'><table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def build_tabbed_report_html(title, subtitle, tabs, highlight=None):
    """``tabs`` = ``[(nome, [(subtítulo, df), ...]), ...]``. ``highlight`` =
    ``(coluna, valor)`` p/ destacar a linha do run atual na 1ª aba (Resumo)."""
    btns, panes = [], []
    for i, (name, blocks) in enumerate(tabs):
        act = " active" if i == 0 else ""
        btns.append(f"<button class='tab-btn{act}' onclick='ygShow({i})'>"
                    f"{_html.escape(name)}</button>")
        body = []
        for sub, df in blocks:
            hc, hv = highlight if (i == 0 and highlight) else (None, None)
            body.append(f"<h3>{_html.escape(sub)}</h3>" + _table_html(df, hc, hv))
        panes.append(f"<div class='tab-pane{act}' id='yg-tab-{i}'>{''.join(body)}</div>")
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>{_CSS}</style>"
            f"</head><body><h1>{_html.escape(title)}</h1>"
            f"<p class='sub'>{_html.escape(subtitle)}</p>"
            f"<div class='tabs'>{''.join(btns)}</div>{''.join(panes)}"
            f"<script>{_JS}</script></body></html>")


def runs_comparison_df(mlflow, experiment_id, current_run_id):
    """DataFrame comparando os runs do experimento em KS/RMSE/PSI (OOT)."""
    try:
        runs = mlflow.search_runs(experiment_ids=[experiment_id],
                                  order_by=["attributes.start_time DESC"])
    except Exception:
        return pd.DataFrame()
    if runs is None or len(runs) == 0:
        return pd.DataFrame()

    def col(name, default=np.nan):
        return (runs[name] if name in runs.columns
                else pd.Series([default] * len(runs), index=runs.index))

    name = col("tags.mlflow.runName").astype(object)
    if "run_id" in runs.columns:
        name = name.where(name.notna(), runs["run_id"].str[:8])
    quando = pd.to_datetime(col("start_time"), errors="coerce", utc=True)
    atual = (col("run_id") == current_run_id).map({True: "➤", False: ""})
    out = pd.DataFrame({
        "": atual.values,
        "run": name.values,
        "quando": quando.dt.strftime("%d/%m/%y %H:%M").values,
        "algoritmo": col("params.algorithm").fillna(col("params.task_type")).values,
        "KS (OOT)": pd.to_numeric(col("metrics.val_ks"), errors="coerce").round(4).values,
        "RMSE (OOT)": pd.to_numeric(col("metrics.val_rmse"), errors="coerce").round(4).values,
        "PSI (OOT)": pd.to_numeric(col("metrics.val_psi"), errors="coerce").round(4).values,
    })
    return out


def _num(row, *names):
    """Busca case-insensitive de uma métrica numa linha de ``metrics_df``."""
    low = {str(k).lower(): k for k in row.index}
    for n in names:
        k = low.get(n.lower())
        if k is not None:
            try:
                v = float(row[k])
                if np.isfinite(v):
                    return v
            except (TypeError, ValueError):
                pass
    return None


def _log_base(mlflow, d, name, df, verbose):
    if df is None or len(df) == 0:
        return
    try:
        p = os.path.join(d, name + ".parquet")
        df.to_parquet(p, index=False)
    except Exception:                                   # pyarrow ausente → CSV
        p = os.path.join(d, name + ".csv")
        df.to_csv(p, index=False)
    mlflow.log_artifact(p, "base")
    if verbose:
        print(f"[mlflow] base '{os.path.basename(p)}' logada ({len(df):,} linhas).".replace(",", "."))


def log_tabbed_report(mlflow, run, *, title, subtitle, val_sample, metrics_df,
                      psi_df, stability_blocks, save_base=False, dev_df=None,
                      oot_df=None, verbose=True):
    """Loga (DENTRO de um run ativo) as métricas canônicas de comparação, o HTML de
    abas e — se ``save_base`` — as bases DES/OOT. Devolve o dict de val_*."""
    val_ks = val_rmse = val_psi = None
    try:
        if metrics_df is not None and "amostra" in getattr(metrics_df, "columns", []) \
                and val_sample is not None:
            mrow = metrics_df[metrics_df["amostra"] == val_sample]
            if len(mrow):
                val_ks = _num(mrow.iloc[0], "ks")
                val_rmse = _num(mrow.iloc[0], "rmse")
    except Exception:
        pass
    try:
        if psi_df is not None and "amostra" in getattr(psi_df, "columns", []) \
                and val_sample is not None:
            prow = psi_df[psi_df["amostra"] == val_sample]
            if len(prow):
                val_psi = float(prow.iloc[0]["psi"])
    except Exception:
        pass
    for k, v in (("val_ks", val_ks), ("val_rmse", val_rmse), ("val_psi", val_psi)):
        if v is not None and np.isfinite(v):
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                pass

    summary = runs_comparison_df(mlflow, run.info.experiment_id, run.info.run_id)
    tabs = [
        ("① Resumo · modelos", [("Comparação dos runs do experimento — KS · RMSE · PSI (OOT)",
                                 summary)]),
        ("② Métricas", [("Métricas por amostra", metrics_df)]),
        ("③ Estabilidade", list(stability_blocks)),
    ]
    html_doc = build_tabbed_report_html(title, subtitle, tabs, highlight=("", "➤"))
    try:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "relatorio.html")
            with open(p, "w", encoding="utf-8") as f:
                f.write(html_doc)
            mlflow.log_artifact(p, "relatorio")
            if save_base:
                _log_base(mlflow, d, "base_DES", dev_df, verbose)
                _log_base(mlflow, d, "base_OOT", oot_df, verbose)
        if verbose:
            print("[mlflow] relatório em abas logado em 'relatorio/relatorio.html'.")
    except Exception as e:                              # pragma: no cover
        if verbose:
            print(f"[mlflow] relatório em abas não logado: {type(e).__name__}: {e}")
    return {"val_ks": val_ks, "val_rmse": val_rmse, "val_psi": val_psi}
