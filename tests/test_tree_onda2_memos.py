"""Onda 2 (itens 2.6 e 2.7) da árvore: memos por MÁSCARA, resultados idênticos.

- 2.6 O IV do painel (``_iv_map``, ``_compute_iv``, ``_cv_iv_table_html``) é
  memoizado pela máscara da folha, não pela versão da árvore: undo/redo e voltar
  a uma folha já vista não refazem o laço de IV. Falha não fica gravada de forma
  permanente, o DataFrame volta como cópia, há backstop e o reset limpa tudo.
- 2.7 PSI de nó (``_leaf_psi``), teste DES×OOT da folha (``_sample_value_test``)
  e p-valor entre folhas (``TreeSegmenter._pair_pvalue``) memoizados por máscara,
  sem ler a matriz arredondada de ``_append_psi_cols``.

As versões antigas estão copiadas LITERALMENTE abaixo (``_velho_*``) e as telas
são comparadas string a string, velho × novo, ao longo de um roteiro de UI.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest

import matplotlib  # noqa: E402
matplotlib.use("Agg")

pytest.importorskip("ipywidgets")

from yggdrasil.credit_risk.tree import TreeSegmenter, TreeSegmenterUI  # noqa: E402
from yggdrasil.credit_risk.tree import segmenter as segmod  # noqa: E402
from yggdrasil.credit_risk.tree import ui as uimod  # noqa: E402

TASKS = ["classification", "regression"]


def _quieto():
    return contextlib.redirect_stdout(io.StringIO())


@contextlib.contextmanager
def _silencio():
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _base(task, n=2400, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.beta(2.5, 3, n) * 1.4 + 0.3
    x[rng.random(n) < 0.06] = np.nan
    gar = rng.choice(list("ABCD"), n, p=[0.5, 0.22, 0.18, 0.1]).astype(object)
    gar[rng.random(n) < 0.05] = None
    lg = {"A": 0.0, "B": 0.10, "C": 0.16, "D": 0.30}
    risco = (0.1 + 0.4 * np.nan_to_num(x - 0.5, nan=0.35)
             + np.array([lg.get(g, 0.2) for g in gar]))
    if task == "classification":
        y = (rng.uniform(0, 1, n) < np.clip(risco, 0.01, 0.95)).astype(float)
    else:
        y = np.clip(risco + rng.normal(0, 0.07, n), 0, 1)
    meses = pd.date_range("2023-01-01", periods=12, freq="MS")
    dt = rng.choice(meses, size=n)
    df = pd.DataFrame({
        "score": x,
        "garantia": gar,
        "num_00": rng.normal(size=n),
        "const": np.ones(n),                 # variável constante → split no-op
        "flag": (rng.random(n) < 0.2).astype(int),
        "renda": rng.gamma(2.0, 1000.0, n),
        "target": y,
    })
    df["dt_ref"] = dt
    amostra = np.where(df["dt_ref"] >= meses[10], "ESTABILIDADE",
                       np.where(df["dt_ref"] >= meses[7], "OOT", "DES"))
    df["amostra"] = amostra
    # alvo NaN espalhado + ESTABILIDADE sem alvo (amostra só de PSI)
    df.loc[rng.random(n) < 0.03, "target"] = np.nan
    df.loc[df["amostra"] == "ESTABILIDADE", "target"] = np.nan
    return df


def _ui(task, df=None):
    df = _base(task) if df is None else df
    with _silencio():
        return TreeSegmenterUI(df, target="target", task_type=task, sample_col="amostra",
                               ref_sample="DES", date_col="dt_ref")


def _folhas(seg):
    return [s for s, v in seg.segments.items() if v["is_leaf"]]


def _split(ui, sid, feat, splits):
    ui._pending = dict(feature=feat, splits=splits, only_segments=[sid])
    with _silencio():
        ui._on_split(None)


# ======================================================================
# Versões ANTIGAS, copiadas literalmente (só o nome do atributo de cache
# muda, para não se misturar com o cache novo)
# ======================================================================
def _velho_leaf_psi(self, sid, sample, eps=1e-6):
    import math
    if self.sample_col is None or sample not in self._sample_masks:
        return float("nan")
    ref_mask = self._sample_masks.get(self.ref_sample)
    if ref_mask is None:
        return float("nan")
    leaf = self.seg.segments[sid]["mask"]
    s_mask = self._sample_masks[sample]
    n_ref, n_s = int(ref_mask.sum()), int(s_mask.sum())
    if n_ref == 0 or n_s == 0:
        return float("nan")
    p_ref = max((leaf & ref_mask).sum() / n_ref, eps)
    p_cur = max((leaf & s_mask).sum() / n_s, eps)
    return float((p_cur - p_ref) * math.log(p_cur / p_ref))


def _velho_sample_value_test(self, sid, a, b, min_n=8):
    name = "Welch t" if self.dd_test.value == "welch" else "Mann-Whitney"
    sm = self._sample_masks
    if a not in sm or b not in sm:
        return name, float("nan"), 0, 0
    leaf = self.seg.segments[sid]["mask"]
    va = self.df.loc[leaf & sm[a], self.target].dropna().to_numpy()
    vb = self.df.loc[leaf & sm[b], self.target].dropna().to_numpy()
    if len(va) < min_n or len(vb) < min_n:
        return name, float("nan"), len(va), len(vb)
    try:
        from scipy.stats import mannwhitneyu, ttest_ind
        if self.dd_test.value == "welch":
            p = float(ttest_ind(va, vb, equal_var=False).pvalue)
        else:
            p = float(mannwhitneyu(va, vb, alternative="two-sided").pvalue)
    except Exception:
        p = float("nan")
    return name, p, len(va), len(vb)


def _velho_iv_map(self, sid):
    sid = sid if sid in self.seg.segments else "root"
    cache_velho = self.__dict__.setdefault("_velho_iv_sort_cache", {})
    key = (sid, self.seg._tree_version)
    if key not in cache_velho:
        try:
            iv = self.seg.variable_iv(sid, features=list(self.features),
                                      with_psi=False)
            cache_velho[key] = dict(zip(iv["variavel"], iv["iv"]))
        except Exception as e:
            self._log(f"(IV p/ ordenação indisponível: {type(e).__name__}: {e})")
            cache_velho[key] = {}
    return cache_velho[key]


def _velho_cv_iv_table_html(self, sid):
    _esc = uimod._esc
    sid = sid if sid in self.seg.segments else "root"
    key = (sid, self.seg._tree_version)
    cache = getattr(self, "_velho_cv_iv_cache", None)
    if cache is not None and cache[0] == key:
        return cache[1]
    iv = self.seg.variable_iv(sid, features=list(self.features), with_psi=True)
    amostras = [c for c in iv.columns
                if c.startswith("psi_") and c != "psi_classificacao"]
    cor = {"ok": "var(--ok-tx)", "warn": "var(--warn-tx)", "bad": "var(--bad-tx)"}

    def uso(r):
        forca = str(r.get("forca", "—"))
        est = str(r.get("psi_classificacao", "estável"))
        if forca == "suspeito":
            return "verificar vazamento", "bad"
        if forca not in ("forte", "médio"):
            return "evitar — fraca", "bad"
        if est == "instável":
            return "evitar — instável", "bad"
        if est == "atenção":
            return "usar com cautela", "warn"
        return "recomendada", "ok"

    th = ("padding:5px 10px;font-size:10px;text-transform:uppercase;"
          "letter-spacing:.05em;color:var(--tbl-head-ink);text-align:left;"
          "border-bottom:2px solid var(--tbl-head-line);white-space:nowrap")
    td = "padding:5px 10px;font-size:11.5px;border-bottom:1px solid var(--tbl-line)"
    cab = ["variável", "faixas", "IV", "força"]
    cab += ["PSI " + ("ESTAB" if c[4:] == "ESTABILIDADE" else c[4:]) for c in amostras]
    cab += (["PSI pior"] if "pior_psi" in iv.columns else []) + ["uso"]
    linhas = []
    for _, r in iv.iterrows():
        rot = self.seg.feature_labels.get(r["variavel"], r["variavel"])
        veredito, classe = uso(r)
        cels = [f"<td style='{td};font-weight:600'>{_esc(rot)}</td>",
                f"<td style='{td}'>{int(r['n_bins'])}</td>",
                "<td style='%s'>%s</td>" % (td, "—" if pd.isna(r["iv"])
                                            else f"{r['iv']:.4f}"),
                f"<td style='{td}'>{_esc(r.get('forca', '—'))}</td>"]
        for c in amostras + (["pior_psi"] if "pior_psi" in iv.columns else []):
            v = r.get(c)
            cels.append("<td style='%s'>%s</td>" % (td, "—" if pd.isna(v)
                                                   else f"{v:.4f}"))
        cels.append(f"<td style='{td};color:{cor[classe]};font-weight:700'>"
                    f"{_esc(veredito)}</td>")
        linhas.append("<tr>" + "".join(cels) + "</tr>")
    html = ("<div style='max-height:460px;overflow:auto'>"
            "<table style='border-collapse:collapse;width:100%'>"
            "<thead><tr>" + "".join(f"<th style='{th}'>{_esc(c)}</th>" for c in cab)
            + "</tr></thead><tbody>" + "".join(linhas) + "</tbody></table></div>")
    self._velho_cv_iv_cache = (key, html)
    return html


def _velho_pair_pvalue(self, sid_a, sid_b, test="mannwhitney", min_n=8):
    mannwhitneyu, ttest_ind = segmod.mannwhitneyu, segmod.ttest_ind
    if mannwhitneyu is None:
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


@contextlib.contextmanager
def _modo_velho(monkeypatch):
    """Troca os métodos novos pelas cópias antigas (literais) dentro do bloco."""
    with monkeypatch.context() as m:
        m.setattr(TreeSegmenterUI, "_leaf_psi", _velho_leaf_psi)
        m.setattr(TreeSegmenterUI, "_sample_value_test", _velho_sample_value_test)
        m.setattr(TreeSegmenterUI, "_iv_map", _velho_iv_map)
        m.setattr(TreeSegmenterUI, "_cv_iv_table_html", _velho_cv_iv_table_html)
        m.setattr(TreeSegmenter, "_pair_pvalue", _velho_pair_pvalue)
        yield


# ======================================================================
# Telas: velho × novo
# ======================================================================
def _mesmo(a, b):
    """Igualdade exata, com NaN == NaN (floats e estruturas aninhadas)."""
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        return
    if isinstance(a, dict):
        assert list(a.keys()) == list(b.keys())
        for k in a:
            _mesmo(a[k], b[k])
        return
    if isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            _mesmo(x, y)
        return
    if isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b)
        return
    assert type(a) is type(b) and a == b, (a, b)


def _telas(ui):
    seg = ui.seg
    sids = list(seg.segments)
    folhas = _folhas(seg)
    amostras = list(ui._samples)
    foco = ui.dd_leaf.value
    with _silencio():
        lv, cols, heads = ui._leaf_table_spec()
        return {
            "header": ui._leaf_header_html(),
            "tree": ui._tree_html(),
            "canvas": repr(ui._canvas_layout()),
            "merge_p": {s: ui._cv_merge_p_html(s) for s in folhas},
            "tabela": lv, "tabela_cols": list(cols), "tabela_heads": list(heads),
            "leaves": seg.leaves(with_psi=True, with_test=True, test=ui.dd_test.value),
            "psi_no": {(s, a): ui._leaf_psi(s, a) for s in sids for a in amostras},
            "teste_amostra": {s: ui._sample_value_test(s, "DES", "OOT") for s in sids},
            "teste_amostra_estab": {s: ui._sample_value_test(s, "DES", "ESTABILIDADE")
                                    for s in folhas},
            "par": {(a, b): seg._pair_pvalue(a, b, test=ui.dd_test.value)
                    for a in folhas for b in folhas if a != b},
            "iv_map": {s: ui._iv_map(s) for s in [foco, "root"]},
            "cv_iv": ui._cv_iv_table_html(foco),
        }


def _confere(ui, monkeypatch):
    novo = _telas(ui)
    novo_de_novo = _telas(ui)           # 2ª passada: tudo vem dos memos
    with _modo_velho(monkeypatch):
        velho = _telas(ui)
    _mesmo(novo, velho)
    _mesmo(novo_de_novo, velho)


def _acao(ui, fn):
    with _silencio():
        fn()


def _seleciona(ui, sid):
    with _silencio():
        ui.dd_leaf.value = sid


def _move_corte(ui):
    """Move o 1º corte numérico móvel para o meio do intervalo válido."""
    for sid in _folhas(ui.seg):
        info = ui.seg.movable_cut(sid)
        if info:
            lo = info["lo"] if np.isfinite(info["lo"]) else info["cut"] - 0.3
            novo = (lo + info["cut"]) / 2
            with _silencio():
                ui._checkpoint()
                ui.seg.move_cut(sid, novo, verbose=False)
                ui._refresh()
            return True
    return False


@pytest.mark.parametrize("task", TASKS)
def test_telas_velho_e_novo_iguais_no_roteiro(task, monkeypatch, tmp_path):
    ui = _ui(task)
    _confere(ui, monkeypatch)                                 # só a raiz
    _split(ui, "root", "score", [0.7, 1.0, 1.3])
    _seleciona(ui, _folhas(ui.seg)[1])
    _confere(ui, monkeypatch)
    _split(ui, _folhas(ui.seg)[0], "garantia", None)           # split categórico
    _confere(ui, monkeypatch)
    _split(ui, _folhas(ui.seg)[-1], "const", [0.5])            # no-op
    _confere(ui, monkeypatch)
    _seleciona(ui, _folhas(ui.seg)[-1])
    _acao(ui, lambda: ui._on_merge("left"))                    # merge
    _confere(ui, monkeypatch)
    _acao(ui, lambda: ui._on_undo(None))
    _confere(ui, monkeypatch)
    _acao(ui, lambda: ui._on_redo(None))
    _confere(ui, monkeypatch)
    _acao(ui, lambda: ui._on_undo(None))
    with _silencio():
        ui.dd_test.value = "welch"                             # troca de teste
    _confere(ui, monkeypatch)
    with _silencio():
        ui.dd_test.value = "mannwhitney"
    _confere(ui, monkeypatch)
    assert _move_corte(ui)                                     # move-cut
    _confere(ui, monkeypatch)
    _seleciona(ui, _folhas(ui.seg)[-1])
    _acao(ui, lambda: ui._on_collapse(None))                   # collapse
    _confere(ui, monkeypatch)
    _acao(ui, lambda: ui._on_automerge(None))                  # auto-merge
    _confere(ui, monkeypatch)
    # save/load JSON pela UI
    path = tmp_path / "arvore.json"
    ui.seg.save(str(path))
    _acao(ui, lambda: ui._on_undo(None))
    ui.tx_json_path.value = str(path)
    _acao(ui, lambda: ui._on_load_json(None))
    _confere(ui, monkeypatch)
    _acao(ui, lambda: ui._on_reset(None))                      # reset
    _confere(ui, monkeypatch)
    _acao(ui, lambda: ui._on_undo(None))                       # undo do reset
    _confere(ui, monkeypatch)


# ======================================================================
# 2.6 — IV memoizado pela máscara
# ======================================================================
def _fresco(ui):
    """Segmentador novo, do zero (sem mask_cache, _bins_cache vazio)."""
    with _silencio():
        return TreeSegmenter.from_dict(ui.seg.to_dict(), ui.df)


def _sid_da_mascara(seg, mask):
    for sid, s in seg.segments.items():
        if s["mask"] is mask:
            return sid
    return None


def _confere_caches_iv(ui):
    """Toda entrada de cache cuja máscara é de um nó ATUAL bate com o cálculo
    do zero (check_exact)."""
    fresco = _fresco(ui)
    n = 0
    with _silencio():
        for (sid_seg, _idm, feats, psi), (mask, iv) in list(ui._iv_df_cache.items()):
            sid = _sid_da_mascara(ui.seg, mask)
            if sid is None or sid_seg != id(ui.seg):
                continue
            ref = fresco.variable_iv(sid, features=None if feats is None else list(feats),
                                     with_psi=psi)
            pd.testing.assert_frame_equal(iv, ref, check_exact=True)
            n += 1
        for key, ent in list(ui._iv_sort_cache.items()):
            sid = _sid_da_mascara(ui.seg, ent["mask"])
            if sid is None or ent["calc"] is None:
                continue
            ref = fresco.variable_iv(sid, features=list(ui.features), with_psi=False)
            _mesmo(ent["calc"], dict(zip(ref["variavel"], ref["iv"])))
            n += 1
    return n


@pytest.mark.parametrize("task", TASKS)
def test_cache_iv_igual_ao_calculo_do_zero_em_sequencia_aleatoria(task, tmp_path):
    ui = _ui(task)
    rng = np.random.default_rng(11)
    path = tmp_path / "a.json"
    feats_split = [("score", None), ("garantia", None), ("num_00", [0.0]),
                   ("const", [0.5]), ("flag", None), ("renda", None)]
    conferidas = 0
    for passo in range(22):
        folhas = _folhas(ui.seg)
        acao = rng.choice(["grow", "grow", "merge", "collapse", "move", "undo", "redo",
                           "foco", "reset", "load", "iv"])
        if acao == "grow":
            f, sp = feats_split[rng.integers(len(feats_split))]
            _split(ui, folhas[rng.integers(len(folhas))], f, sp)
        elif acao == "merge":
            _seleciona(ui, folhas[rng.integers(len(folhas))])
            _acao(ui, lambda: ui._on_merge(rng.choice(["left", "right"])))
        elif acao == "collapse":
            _seleciona(ui, folhas[rng.integers(len(folhas))])
            _acao(ui, lambda: ui._on_collapse(None))
        elif acao == "move":
            _move_corte(ui)
        elif acao == "undo":
            _acao(ui, lambda: ui._on_undo(None))
        elif acao == "redo":
            _acao(ui, lambda: ui._on_redo(None))
        elif acao == "foco":
            _seleciona(ui, folhas[rng.integers(len(folhas))])
        elif acao == "reset":
            _acao(ui, lambda: ui._on_reset(None))
        elif acao == "load":
            ui.seg.save(str(path))
            _acao(ui, lambda: ui._on_undo(None))
            ui.tx_json_path.value = str(path)
            _acao(ui, lambda: ui._on_load_json(None))
        # o painel pede o IV da folha em foco nas três formas
        foco = ui.dd_leaf.value
        with _silencio():
            ui._iv_map(foco)
            ui._cv_iv_table_html(foco)
            ui._compute_iv()
        conferidas += _confere_caches_iv(ui)
    assert conferidas > 30


@pytest.mark.parametrize("task", TASKS)
def test_contador_variable_iv_undo_e_grow_que_recria_o_sid(task, monkeypatch):
    ui = _ui(task)
    _split(ui, "root", "score", [0.8, 1.2])
    x = _folhas(ui.seg)[0]
    y = _folhas(ui.seg)[-1]
    with _silencio():
        ui._iv_map(x)
        ui._cv_iv_table_html(x)
    cont = {"n": 0, "sids": []}
    orig = TreeSegmenter.variable_iv

    def conta(self, sid=None, *a, **k):
        cont["n"] += 1
        cont["sids"].append(sid)
        return orig(self, sid, *a, **k)
    monkeypatch.setattr(TreeSegmenter, "variable_iv", conta)

    # mutação em OUTRA folha + undo: a folha em foco mantém a máscara → 0 cálculos
    mask_x = ui.seg.segments[x]["mask"]
    _split(ui, y, "garantia", None)
    _acao(ui, lambda: ui._on_undo(None))
    cont["n"], cont["sids"] = 0, []
    with _silencio():
        m1 = ui._iv_map(x)
        h1 = ui._cv_iv_table_html(x)
    if ui.seg.segments[x]["mask"] is mask_x:
        assert cont["n"] == 0
    else:   # _prime_mask_cache pode devolver outro objeto de mesmo conteúdo
        assert cont["n"] <= 2
    # o mesmo valor do cálculo do zero
    with _modo_velho(monkeypatch):
        _mesmo(m1, ui._iv_map(x))
        assert h1 == ui._cv_iv_table_html(x)

    # grow que recria o sid (o sid é o rótulo das condições) com OUTRO objeto-
    # máscara → calcula de novo, 1× por forma, e bate com o velho
    _acao(ui, lambda: ui._on_undo(None))                       # volta à raiz
    _split(ui, "root", "score", [0.8, 1.0])
    assert x in ui.seg.segments
    nova = ui.seg.segments[x]["mask"]
    cont["n"], cont["sids"] = 0, []
    with _silencio():
        m2 = ui._iv_map(x)
        h2 = ui._cv_iv_table_html(x)
    if nova is mask_x:
        assert cont["n"] == 0
    else:
        assert cont["n"] == 2
    with _modo_velho(monkeypatch):
        _mesmo(m2, ui._iv_map(x))
        assert h2 == ui._cv_iv_table_html(x)
    cont["n"] = 0
    with _silencio():
        assert ui._iv_map(x) is m2
        assert ui._cv_iv_table_html(x) == h2
    assert cont["n"] == 0


@pytest.mark.parametrize("task", TASKS)
def test_semeado_pelo_compute_iv_vale_so_na_versao(task, monkeypatch):
    ui = _ui(task)
    _split(ui, "root", "score", [0.8, 1.2])
    x, y = _folhas(ui.seg)[0], _folhas(ui.seg)[-1]
    _seleciona(ui, x)
    with _silencio():
        ui._compute_iv()
    _, ent = ui._iv_entry(x)
    assert ent["semeado"][0] == ui.seg._tree_version
    semeado = ent["semeado"][1]
    with _silencio():
        assert ui._iv_map(x) is semeado               # mesma versão → o semeado
    # mutação em outra folha: o semeado expira (como a chave por versão de antes)
    _split(ui, y, "garantia", None)
    with _silencio():
        depois = ui._iv_map(x)
        with _modo_velho(monkeypatch):
            ref = ui._iv_map(x)
    assert depois is not semeado
    _mesmo(depois, ref)


@pytest.mark.parametrize("task", TASKS)
def test_iv_map_erro_nao_fica_gravado(task, monkeypatch):
    ui = _ui(task)
    _split(ui, "root", "score", [0.8, 1.2])
    x, y = _folhas(ui.seg)[0], _folhas(ui.seg)[-1]
    cont = {"n": 0, "falha": True}
    orig = TreeSegmenter.variable_iv

    def talvez_falha(self, *a, **k):
        cont["n"] += 1
        if cont["falha"]:
            raise MemoryError("sem memória")
        return orig(self, *a, **k)
    monkeypatch.setattr(TreeSegmenter, "variable_iv", talvez_falha)
    linhas0 = len(ui._log_lines)
    with _silencio():
        assert ui._iv_map(x) == {}
        assert ui._iv_map(x) == {}            # mesma versão: não tenta de novo
    assert cont["n"] == 1
    assert len(ui._log_lines) == linhas0 + 1
    cont["falha"] = False
    with _silencio():
        assert ui._iv_map(x) == {}            # ainda na mesma versão (como antes)
    assert cont["n"] == 1
    # mutação noutra folha: a falha expira e o cálculo é refeito
    _split(ui, y, "garantia", None)
    cont["n"] = 0
    with _silencio():
        m = ui._iv_map(x)
    assert cont["n"] == 1 and len(m) == len(ui.features)


@pytest.mark.parametrize("task", TASKS)
def test_iv_df_devolve_copia_e_excecao_nao_e_guardada(task, monkeypatch):
    ui = _ui(task)
    with _silencio():
        a = ui._iv_df("root", None, True)
        a.loc[0, "iv"] = 123.0
        a.attrs["valor_medio"] = -1
        b = ui._iv_df("root", None, True)
        ref = ui.seg.variable_iv(None)
    pd.testing.assert_frame_equal(b, ref, check_exact=True)
    assert b.attrs == ref.attrs
    # None, 'root' e sid inexistente usam a máscara da raiz
    with _silencio():
        pd.testing.assert_frame_equal(ui._iv_df(None, None, True), ref, check_exact=True)
        pd.testing.assert_frame_equal(ui._iv_df("nao_existe", None, True), ref,
                                      check_exact=True)
    assert len(ui._iv_df_cache) == 1

    cont = {"n": 0}

    def falha(self, *a, **k):
        cont["n"] += 1
        raise RuntimeError("x")
    monkeypatch.setattr(TreeSegmenter, "variable_iv", falha)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            ui._iv_df("root", ["score"], False)
    assert cont["n"] == 2


@pytest.mark.parametrize("task", TASKS)
def test_compute_iv_igual_com_e_sem_cache(task):
    ui = _ui(task)
    _split(ui, "root", "score", [0.8, 1.2])
    _seleciona(ui, _folhas(ui.seg)[1])
    import re

    def html():
        # o Styler do pandas sorteia um uuid por render (#T_xxxxx): neutraliza
        return re.sub(r"T_[0-9a-f]{5}", "T_X", ui.out_iv.value)
    with _silencio():
        ui._compute_iv()
        a = html()
        ui._compute_iv()                  # do cache
        b = html()
        ui._iv_df_cache.clear()
        ui._compute_iv()                  # do zero
        c = html()
    assert a == b == c
    assert "T_X" in a


def test_backstop_dos_caches_de_iv(monkeypatch):
    ui = _ui("classification")
    leve = pd.DataFrame({"variavel": ["score"], "iv": [0.1]})
    monkeypatch.setattr(TreeSegmenter, "variable_iv", lambda self, *a, **k: leve.copy())
    for i in range(70):
        ui._iv_df("root", [f"f{i}"], True)
    assert 1 <= len(ui._iv_df_cache) <= 65
    for i in range(70):
        monkeypatch.setattr(ui, "features", [f"g{i}"])
        ui._iv_map("root")
    assert 1 <= len(ui._iv_sort_cache) <= 65


@pytest.mark.parametrize("task", TASKS)
def test_reset_limpa_os_memos_da_ui(task):
    ui = _ui(task)
    _split(ui, "root", "score", [0.8, 1.2])
    _seleciona(ui, _folhas(ui.seg)[0])
    _telas(ui)
    assert ui._iv_df_cache and ui._iv_sort_cache
    assert ui.__dict__.get("_leaf_psi_cache") and ui.__dict__.get("_sample_test_cache")
    seg_velho = ui.seg
    _acao(ui, lambda: ui._on_reset(None))
    assert ui.seg is not seg_velho
    _so_do_seg_atual(ui, seg_velho)


def _so_do_seg_atual(ui, seg_velho):
    """Depois da troca de self.seg, nenhum memo da UI guarda entrada da árvore
    anterior (o _refresh do reset já repovoa com a árvore nova)."""
    velhas = {id(s["mask"]) for s in seg_velho.segments.values()}
    novas = {id(s["mask"]) for s in ui.seg.segments.values()}
    velhas -= novas
    for k, ent in ui._iv_sort_cache.items():
        assert k[0] == id(ui.seg) and id(ent["mask"]) not in velhas
    for k, (mask, _iv) in ui._iv_df_cache.items():
        assert k[0] == id(ui.seg) and id(mask) not in velhas
    for nome in ("_leaf_psi_cache", "_sample_test_cache"):
        for v in ui.__dict__.get(nome, {}).values():
            assert id(v[0]) not in velhas
            # as máscaras de amostra guardadas são as do segmentador novo
            for m_am in (v[1], v[2]):
                assert any(m_am is m for m in ui._sample_masks.values())
    for a, (m, _n) in ui.__dict__.get("_sample_n_cache", {}).items():
        assert m is ui._sample_masks[a]


@pytest.mark.parametrize("task", TASKS)
def test_cv_reset_limpa_os_memos_da_ui(task, monkeypatch):
    ui = _ui(task)
    _split(ui, "root", "score", [0.8, 1.2])
    with _silencio():
        ui._cv_iv_table_html("root")
        ui._leaf_psi("root", "OOT")
    # confirma em dois cliques: executa a ação direto
    monkeypatch.setattr(TreeSegmenterUI, "_confirm_twice", lambda self, b, fn: fn())
    seg_velho = ui.seg
    _acao(ui, lambda: ui._on_cv_reset(ui.btn_cv_reset))
    assert ui.seg is not seg_velho
    _so_do_seg_atual(ui, seg_velho)


# ======================================================================
# 2.7 — PSI de nó, teste DES×OOT e p-valor entre folhas
# ======================================================================
def _contador_testes(monkeypatch):
    import scipy.stats as st
    cont = {"mw": 0, "t": 0}
    mw, tt = st.mannwhitneyu, st.ttest_ind

    def mw_c(*a, **k):
        cont["mw"] += 1
        return mw(*a, **k)

    def tt_c(*a, **k):
        cont["t"] += 1
        return tt(*a, **k)
    monkeypatch.setattr(st, "mannwhitneyu", mw_c)
    monkeypatch.setattr(st, "ttest_ind", tt_c)
    monkeypatch.setattr(segmod, "mannwhitneyu", mw_c)
    monkeypatch.setattr(segmod, "ttest_ind", tt_c)
    return cont


@pytest.mark.parametrize("task", TASKS)
def test_segundo_refresh_sem_mudanca_nao_chama_teste(task, monkeypatch):
    ui = _ui(task)
    _split(ui, "root", "score", [0.7, 1.0, 1.3])
    _seleciona(ui, _folhas(ui.seg)[1])
    cont = _contador_testes(monkeypatch)
    ui.seg._pair_p_cache.clear()
    ui.__dict__.pop("_sample_test_cache", None)
    with _silencio():
        ui._refresh()
        ui._sync_cv_actions()
    assert cont["mw"] > 0
    with _silencio():
        _telas(ui)                    # aquece também os pares não adjacentes
    cont["mw"] = cont["t"] = 0
    with _silencio():
        ui._refresh()
        ui._sync_cv_actions()
        _telas(ui)
    assert cont["mw"] == 0 and cont["t"] == 0
    # split numa folha: só os pares/folhas novos chamam o teste de novo
    chaves0 = set(ui.seg._pair_p_cache)
    _split(ui, _folhas(ui.seg)[0], "garantia", None)
    cont["mw"] = 0
    with _silencio():
        ui._refresh()
    novos = set(ui.seg._pair_p_cache) - chaves0
    # cada chamada do segmentador corresponde a um par novo; as da UI, a
    # testes DES×OOT de folhas novas
    assert cont["mw"] <= len(novos) + 2 * len(ui.seg.segments)
    cont["mw"] = 0
    with _silencio():
        ui._refresh()
    assert cont["mw"] == 0
    # trocar o teste: welch é chave nova, voltar a Mann-Whitney é cache-hit
    with _silencio():
        _telas(ui)
        ui.dd_test.value = "welch"
        _telas(ui)
    assert cont["t"] > 0
    cont["mw"] = cont["t"] = 0
    with _silencio():
        ui.dd_test.value = "mannwhitney"
        _telas(ui)
        ui.dd_test.value = "welch"
        _telas(ui)
    assert cont["mw"] == 0 and cont["t"] == 0


@pytest.mark.parametrize("task", TASKS)
def test_pair_pvalue_memo_igual_ao_velho(task, monkeypatch):
    df = _base(task)
    with _silencio():
        seg = TreeSegmenter(df, target="target", task_type=task, sample_col="amostra",
                            ref_sample="DES", date_col="dt_ref", verbose=False)
        seg.grow(feature="score", splits=[0.6, 0.9, 1.2, 1.5])
        seg.grow(feature="garantia", only_segments=[_folhas(seg)[0]])
    folhas = _folhas(seg)
    for test in ("mannwhitney", "welch"):
        for min_n in (8, 50, 10**6):
            for a in folhas:
                for b in folhas:
                    novo = seg._pair_pvalue(a, b, test=test, min_n=min_n)
                    velho = _velho_pair_pvalue(seg, a, b, test=test, min_n=min_n)
                    _mesmo(novo, velho)
                    _mesmo(seg._pair_pvalue(a, b, test=test, min_n=min_n), velho)
    # par NÃO ordenado: (a, b) e (b, a) são chaves distintas, cada uma com o
    # resultado do teste na sua ordem
    cont = _contador_testes(monkeypatch)
    a, b = folhas[0], folhas[1]
    seg._pair_p_cache.clear()
    seg._pair_pvalue(a, b)
    seg._pair_pvalue(a, b)
    assert cont["mw"] == 1
    seg._pair_pvalue(b, a)
    assert cont["mw"] == 2
    seg._pair_pvalue(a, b, min_n=9)
    assert cont["mw"] == 3
    # prune/auto_merge headless dão o mesmo resultado com e sem memo
    with _silencio():
        d0 = seg.to_dict()
        s1 = TreeSegmenter.from_dict(d0, df)
        s1.auto_merge(alpha=0.2)
        s2 = TreeSegmenter.from_dict(d0, df)
        with monkeypatch.context() as m:
            m.setattr(TreeSegmenter, "_pair_pvalue", _velho_pair_pvalue)
            s2.auto_merge(alpha=0.2)
    assert s1.to_dict() == s2.to_dict()


def test_pair_cache_limpo_junto_com_o_leaf_target_cache():
    df = _base("classification")
    with _silencio():
        seg = TreeSegmenter(df, target="target", task_type="classification",
                            sample_col="amostra", ref_sample="DES", date_col="dt_ref",
                            verbose=False)
        seg.grow(feature="score", splits=[0.8, 1.2])
    a, b, c = _folhas(seg)[:3]
    seg._pair_pvalue(a, b)
    assert seg._pair_p_cache
    seg._leaf_target_cache.clear()
    seg._leaf_target_cache.update({-i: (None, None) for i in range(1, 4002)})
    seg._leaf_target(c)                       # miss → backstop esvazia os dois
    assert len(seg._leaf_target_cache) == 1
    assert not seg._pair_p_cache


def test_pair_pvalue_sem_scipy(monkeypatch):
    df = _base("classification")
    with _silencio():
        seg = TreeSegmenter(df, target="target", task_type="classification",
                            sample_col="amostra", ref_sample="DES", verbose=False)
        seg.grow(feature="score", splits=[1.0])
    monkeypatch.setattr(segmod, "mannwhitneyu", None)
    a, b = _folhas(seg)[:2]
    assert np.isnan(seg._pair_pvalue(a, b))
    assert not seg._pair_p_cache


@pytest.mark.parametrize("task", TASKS)
def test_leaf_psi_e_teste_de_amostra_invalidam_por_mascara(task, monkeypatch):
    ui = _ui(task)
    _split(ui, "root", "score", [0.8, 1.2])
    x = _folhas(ui.seg)[0]
    with _silencio():
        p0 = ui._leaf_psi(x, "OOT")
        t0 = ui._sample_value_test(x, "DES", "OOT")
    mask_x = ui.seg.segments[x]["mask"]
    # mesma máscara → hit (nenhuma varredura nova)
    chamadas = {"n": 0}
    orig_sum = pd.Series.sum

    def conta_sum(self, *a, **k):
        chamadas["n"] += 1
        return orig_sum(self, *a, **k)
    monkeypatch.setattr(pd.Series, "sum", conta_sum)
    assert ui._leaf_psi(x, "OOT") == p0
    assert chamadas["n"] == 0
    monkeypatch.setattr(pd.Series, "sum", orig_sum)
    assert ui._sample_value_test(x, "DES", "OOT") == t0
    # amostra/rótulo ausente: mesmas saídas de antes, sem cache
    with _modo_velho(monkeypatch):
        _mesmo(ui._leaf_psi(x, "NAO"), _velho_leaf_psi(ui, x, "NAO"))
    _mesmo(ui._sample_value_test(x, "DES", "NAO"),
           _velho_sample_value_test(ui, x, "DES", "NAO"))
    # sid recriado com outra máscara → recalcula e bate com o velho
    _acao(ui, lambda: ui._on_undo(None))
    _split(ui, "root", "score", [0.8, 1.0])
    assert x in ui.seg.segments
    _mesmo(ui._leaf_psi(x, "OOT"), _velho_leaf_psi(ui, x, "OOT"))
    _mesmo(ui._sample_value_test(x, "DES", "OOT"),
           _velho_sample_value_test(ui, x, "DES", "OOT"))
    for mn in (1, 8, 10**6):
        _mesmo(ui._sample_value_test(x, "DES", "OOT", min_n=mn),
               _velho_sample_value_test(ui, x, "DES", "OOT", min_n=mn))


def test_leaf_psi_sem_amostra():
    df = _base("classification").drop(columns=["amostra"])
    with _silencio():
        ui = TreeSegmenterUI(df, target="target", task_type="classification",
                             date_col="dt_ref")
    _mesmo(ui._leaf_psi("root", "OOT"), _velho_leaf_psi(ui, "root", "OOT"))
    _mesmo(ui._sample_value_test("root", "DES", "OOT"),
           _velho_sample_value_test(ui, "root", "DES", "OOT"))
