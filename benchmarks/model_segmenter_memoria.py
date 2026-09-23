"""Benchmark de memória e tempo do ModelSegmenter por etapa.

Gera uma base sintética de ``--n`` linhas × 15 colunas (9 numéricas, 3
categóricas de 5/12/27 níveis, alvo, amostra DES/OOT 75/25 e safra) e mede, para
cada etapa do fluxo da UI, o tempo e o pico de RSS do processo (amostrado a cada
5 ms numa thread). Uso, da raiz do repositório::

    python benchmarks/model_segmenter_memoria.py --n 3000000 --task classification
    python benchmarks/model_segmenter_memoria.py --n 3000000 --task regression --out r.csv

Requer ``psutil``. ``delta_pico_mb`` é o quanto a etapa pediu além do que já
estava residente; ``pico_mb`` é o RSS máximo do processo durante a etapa.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
import warnings

import numpy as np
import pandas as pd
import psutil

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROC = psutil.Process(os.getpid())
MB = 1024 ** 2


def rss() -> float:
    return PROC.memory_info().rss / MB


class Peak:
    """Amostra o RSS a cada 5 ms numa thread e guarda o pico."""

    def __enter__(self):
        gc.collect()
        self.base = rss()
        self.peak = self.base
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        self.t0 = time.perf_counter()
        return self

    def _run(self):
        while not self._stop.is_set():
            r = rss()
            if r > self.peak:
                self.peak = r
            time.sleep(0.005)

    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, rss())
        gc.collect()
        self.after = rss()
        return False


def make_df(n: int, task: str, seed: int = 0, n_cat_levels=(5, 12, 27)) -> pd.DataFrame:
    """15 colunas: alvo, amostra, safra + 9 numéricas + 3 categóricas."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 9)).astype("float64")
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(9)])
    for j, k in enumerate(n_cat_levels):
        codes = rng.integers(0, k, n)
        df[f"cat_{j}"] = pd.Series(np.array([f"c{j}_{i:02d}" for i in range(k)], dtype=object)[codes])
    # faltantes em duas numéricas
    for c in ("num_1", "num_4"):
        m = rng.random(n) < 0.08
        df.loc[m, c] = np.nan
    lin = (0.8 * df["num_0"] - 0.5 * df["num_2"].fillna(0) + 0.3 * df["num_3"]
           + 0.2 * (df["cat_0"].str[-1].astype(int)))
    if task == "classification":
        p = 1 / (1 + np.exp(-(lin - 3.0)))
        df["target"] = (rng.random(n) < p).astype(int)
    else:
        p = 1 / (1 + np.exp(-(lin - 0.5)))
        z = rng.random(n)
        df["target"] = np.where(z < 0.35, 0.0, np.clip(p + 0.15 * rng.standard_normal(n), 0, 1))
    safra = rng.integers(0, 24, n)
    df["safra"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(safra * 30, unit="D")
    df["amostra"] = np.where(safra < 18, "DES", "OOT").astype(object)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--task", default="classification")
    ap.add_argument("--steps", default="all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.environ.get("YGG_PATH", raiz))
    from yggdrasil.credit_risk.model.segmenter import ModelSegmenter

    results = []

    def step(name, fn):
        with Peak() as p:
            try:
                fn()
                status = "ok"
            except Exception as e:  # noqa: BLE001
                status = f"ERRO {type(e).__name__}: {e}"[:160]
        plt.close("all")
        row = {"etapa": name, "tempo_s": round(p.dt, 2), "rss_antes_mb": round(p.base),
               "pico_mb": round(p.peak), "delta_pico_mb": round(p.peak - p.base),
               "rss_depois_mb": round(p.after), "status": status}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    state = {}
    step("gerar_df", lambda: state.__setitem__("df", make_df(args.n, args.task)))
    df = state["df"]
    print(f"# df: {df.shape}, memória deep = {df.memory_usage(deep=True).sum() / MB:.0f} MB",
          flush=True)

    step("ModelSegmenter()", lambda: state.__setitem__("seg", ModelSegmenter(
        df, target="target", task_type=args.task, sample_col="amostra",
        date_col="safra", verbose=False)))
    seg = state["seg"]
    want = set(args.steps.split(",")) if args.steps != "all" else None

    def run(name, fn):
        if want is None or name in want:
            step(name, fn)

    run("variable_iv", lambda: seg.variable_iv())
    algo = "logistica" if args.task == "classification" else "linear"
    run("fit_raw", lambda: seg.fit(algo))
    run("metrics", lambda: seg.metrics())
    run("model_formula", lambda: seg.model_formula())
    run("vif_table", lambda: seg.vif_table())
    if args.task == "classification":
        run("plot_roc", lambda: seg.plot_roc(figsize=(6.6, 4.8)).savefig(os.devnull, format="png"))
        run("plot_ks", lambda: seg.plot_ks(figsize=(6.6, 4.8)).savefig(os.devnull, format="png"))
    else:
        run("plot_calibration", lambda: seg.plot_calibration(figsize=(6.6, 4.8)).savefig(os.devnull, format="png"))
        run("plot_residuals", lambda: seg.plot_residuals(figsize=(6.6, 4.8)).savefig(os.devnull, format="png"))
    run("plot_score_distribution", lambda: seg.plot_score_distribution(figsize=(6.6, 4.8)).savefig(os.devnull, format="png"))
    run("plot_metric_comparison", lambda: seg.plot_metric_comparison(figsize=(6.6, 4.8)).savefig(os.devnull, format="png"))
    run("metrics_ci_n20", lambda: seg.metrics_ci(n_boot=20))
    run("metrics_ci_n200", lambda: seg.metrics_ci(n_boot=200, seed=1))
    run("build_ratings", lambda: seg.build_ratings(method="quantil", n_ratings=10))
    run("rating_table", lambda: seg.rating_table())
    run("fit_woe", lambda: seg.fit(algo, transform="woe"))
    run("fit_hgb", lambda: seg.fit("hist_gradient_boosting"))
    run("fit_lgbm", lambda: seg.fit("lightgbm"))
    run("shap_800", lambda: seg.shap_values(sample_size=800))

    if args.out:
        pd.DataFrame(results).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
