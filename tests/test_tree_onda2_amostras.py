"""Onda 2 (itens 2.3 e 2.4) da árvore: resultados idênticos aos de antes.

- 2.3 As máscaras de amostra pré-computadas (``_sample_masks``) passam a ser
  lidas por ``_smask(a)`` em todos os pontos que recomparavam
  ``df[sample_col] == a`` sobre a base inteira; a lista de amostras vira
  ``_samples`` (ordem de ``dropna().unique()``); ``_ordered_samples_with_value``
  é memoizada (tupla); a UI reaproveita as máscaras do segmentador.
- 2.4 Código de folha int32 por versão (``_leaf_codes``); PSI e colunas de PSI
  das folhas por ``np.bincount`` (inválidos filtrados ANTES de combinar os
  códigos); ``_bump_version`` limpa o cache; ``_leaf_id_series`` vira wrapper
  preguiçoso sobre um array object read-only.

As referências "antigas" abaixo são cópias LITERAIS do código anterior.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

import matplotlib  # noqa: E402
matplotlib.use("Agg")

from yggdrasil.credit_risk.tree import TreeSegmenter  # noqa: E402
from yggdrasil.credit_risk.tree.segmenter import (  # noqa: E402
    _classifica_psi, _psi_from_shares)

TASKS = ["classification", "regression"]
DTYPES = ["object", "category", "string", "num"]


def _quieto():
    return contextlib.redirect_stdout(io.StringIO())


def _amostras(dtype):
    """(ref, demais) conforme o dtype da coluna de amostra."""
    if dtype == "num":
        return 1.0, [2.0, 0.0]
    return "DES", ["OOT", "ESTAB"]


def _base(task, dtype="object", n=4000, seed=3, indice="perm"):
    rng = np.random.default_rng(seed)
    am = rng.choice(["ESTAB", "DES", "OOT"], n, p=[0.2, 0.5, 0.3]).astype(object)
    am[0] = "ESTAB"                     # 1ª aparição ≠ ordem alfabética
    am[1] = "OOT"
    am[rng.random(n) < 0.03] = np.nan   # amostra NaN espalhada por TODAS as folhas
    am[rng.random(n) < 0.01] = None
    x = rng.normal(size=n)
    x[rng.random(n) < 0.05] = np.nan
    c = rng.choice(list("abcde"), n).astype(object)
    # só a DES passa de 3,5 → a folha "xd > 3,5" fica VAZIA no OOT/ESTAB
    xd = np.where(am == "DES", rng.normal(size=n) + 5 * (rng.random(n) < 0.15),
                  np.clip(rng.normal(size=n), -3, 3))
    z = 0.8 * np.nan_to_num(x) + (c == "a") * 0.7 + rng.normal(size=n)
    y = (1 / (1 + np.exp(-z))) if task == "regression" else (z > 1.0).astype(float)
    y[rng.random(n) < 0.02] = np.nan
    meses = pd.date_range("2023-01-01", periods=8, freq="MS")
    df = pd.DataFrame({"x": x, "c": c, "xd": xd, "saldo": rng.gamma(2.0, 100.0, n),
                       "dt_ref": rng.choice(meses, size=n), "amostra": am, "target": y})
    if dtype == "category":
        df["amostra"] = pd.Categorical(df["amostra"],
                                       categories=["DES", "OOT", "ESTAB", "NUNCA_USADA"])
    elif dtype == "string":
        df["amostra"] = df["amostra"].astype("string")      # pd.NA nos faltantes
    elif dtype == "num":
        df["amostra"] = df["amostra"].map({"DES": 1.0, "OOT": 2.0, "ESTAB": 0.0})
    if indice == "dup":
        df.index = np.r_[np.arange(n // 2), np.arange(n - n // 2)]
    elif indice == "perm":
        df.index = rng.permutation(n) * 3 + 7
    return df


def _seg(task, dtype="object", indice="perm", peso=False, n=4000):
    df = _base(task, dtype, n=n, indice=indice)
    ref, _ = _amostras(dtype)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = TreeSegmenter(df, target="target", task_type=task, sample_col="amostra",
                            ref_sample=ref, date_col="dt_ref",
                            weight_col="saldo" if peso else None, verbose=False)
    return df, seg


def _cresce(seg, dtype="object"):
    """3 splits (num, num com folha sem OOT, cat). StringDtype com pd.NA quebra o
    grow a partir do 2º split desde antes — fica só no 1º."""
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.grow("x", max_n_bins=3)
        if dtype == "string":
            return seg
        f = [s for s, v in seg.segments.items() if v["is_leaf"]]
        seg.grow("xd", splits=[3.5], only_segments=f[-1:])
        f = [s for s, v in seg.segments.items() if v["is_leaf"]]
        seg.grow("c", max_n_bins=3, only_segments=f[:1])
    return seg


def _orfas(seg):
    """Tira uma folha do conjunto de folhas (as linhas dela viram órfãs)."""
    f = [s for s, v in seg.segments.items() if v["is_leaf"]]
    seg.segments[f[1]]["is_leaf"] = False
    seg._bump_version()
    return f[1]


# ─────────────── referências LITERAIS do código anterior ───────────────
def _leaf_id_antigo(seg, leaf_ids=None):
    ids = leaf_ids if leaf_ids is not None else \
        [sid for sid, s in seg.segments.items() if s["is_leaf"]]
    arr = np.full(len(seg.df), pd.NA, dtype=object)
    for sid in ids:
        arr[seg.segments[sid]["mask"].values] = sid
    return pd.Series(arr, index=seg.df.index, dtype=object).copy()


def _psi_antigo(seg, eps=1e-6):
    leaf_ids = [sid for sid, s in seg.segments.items() if s["is_leaf"]]
    leaf_id_por_linha = _leaf_id_antigo(seg, leaf_ids)
    ct = pd.crosstab(leaf_id_por_linha, seg.df[seg.sample_col])
    dist = {}
    for amostra in seg.df[seg.sample_col].dropna().unique():
        n_am = float(seg._sample_masks[amostra].sum())
        col = ct[amostra] if amostra in ct.columns else None
        dist[amostra] = {
            sid: (float(col.get(sid, 0.0)) / n_am if (col is not None and n_am) else 0.0)
            for sid in leaf_ids
        }
    ref = dist[seg.ref_sample]
    detalhe, linhas = [], []
    for amostra, pct in dist.items():
        if amostra == seg.ref_sample:
            continue
        psi_total, contribs = _psi_from_shares(
            [ref[sid] for sid in leaf_ids], [pct[sid] for sid in leaf_ids],
            eps, return_contrib=True)
        for sid, contrib in zip(leaf_ids, contribs):
            detalhe.append({
                "amostra": amostra, "segmento": sid,
                f"%_{seg.ref_sample}": round(100 * ref[sid], 2),
                "%_atual": round(100 * pct[sid], 2),
                "psi_bin": round(contrib, 4),
            })
        linhas.append({"amostra": amostra, "psi": round(psi_total, 4),
                       "classificacao": _classifica_psi(psi_total)})
    out = pd.DataFrame(linhas).sort_values("psi", ascending=False).reset_index(drop=True)
    return out, pd.DataFrame(detalhe)


def _append_psi_antigo(seg, out, eps=1e-6):
    leaf_ids = out["segmento"].tolist()
    leaf_id = _leaf_id_antigo(seg)
    ct = pd.crosstab(leaf_id, seg.df[seg.sample_col])

    def fracs(amostra):
        n = float(seg._sample_masks[amostra].sum())
        col = ct[amostra] if amostra in ct.columns else None
        return {sid: (float(col.get(sid, 0.0)) / n if (col is not None and n) else 0.0)
                for sid in leaf_ids}

    ref_pct = fracs(seg.ref_sample)
    out[f"repr_{seg.ref_sample}_%"] = [round(100 * ref_pct[sid], 1) for sid in leaf_ids]
    for amostra in seg.df[seg.sample_col].dropna().unique():
        if amostra == seg.ref_sample:
            continue
        cur_pct = fracs(amostra)
        _, contribs = _psi_from_shares([ref_pct[sid] for sid in leaf_ids],
                                       [cur_pct[sid] for sid in leaf_ids],
                                       eps, return_contrib=True)
        out[f"psi_{amostra}"] = [round(c, 4) for c in contribs]
        out[f"repr_{amostra}_%"] = [round(100 * cur_pct[sid], 1) for sid in leaf_ids]
    return out


def _compute_leaves_antigo(seg, ascending=True):
    """Só as colunas de alvo por amostra que mudaram de caminho (sub == a)."""
    linhas = []
    for sid, sg in seg.segments.items():
        if not sg["is_leaf"]:
            continue
        sub = seg._sub(sg["mask"])
        ref_t = sub.loc[sub[seg.sample_col] == seg.ref_sample, seg.target]
        pd_m = ref_t.mean() if len(ref_t) else sub[seg.target].mean()
        row = {"segmento": sid,
               "valor_medio": round(pd_m, 4) if not pd.isna(pd_m) else np.nan}
        for amostra in list(seg._sample_masks):
            s_am = sub.loc[sub[seg.sample_col] == amostra, seg.target]
            row[f"valor_{amostra}"] = round(s_am.mean(), 4) if len(s_am) else np.nan
        linhas.append(row)
    return pd.DataFrame(linhas).set_index("segmento")


def _osv_antigo(seg):
    samples = list(seg.df[seg.sample_col].dropna().unique())
    com_pd = [a for a in samples
              if seg.df.loc[seg.df[seg.sample_col] == a, seg.target].notna().any()]
    ref = [seg.ref_sample] if seg.ref_sample in com_pd else []
    return ref + [a for a in com_pd if a != seg.ref_sample]


def _calib_antigo(seg, check_sample=None):
    if seg.sample_col is not None and check_sample is None:
        nonref = [a for a in seg.df[seg.sample_col].dropna().unique()
                  if a != seg.ref_sample]
        check_sample = nonref[0] if nonref else seg.ref_sample
    lv = seg.leaves()
    rows = []
    for _, r in lv.iterrows():
        sid = r["segmento"]
        prev_vals = seg._leaf_target(sid)
        prev = float(prev_vals.mean()) if len(prev_vals) else np.nan
        m = seg.segments[sid]["mask"]
        if check_sample is not None and seg.sample_col is not None:
            m = m & (seg.df[seg.sample_col] == check_sample)
        real_vals = seg.df.loc[m, seg.target]
        real = float(real_vals.mean()) if len(real_vals) else np.nan
        rows.append({
            "nota": int(r["nota"]), "descricao": r["descricao"],
            "n": int(len(real_vals)),
            "valor_previsto": round(prev, 4) if not np.isnan(prev) else np.nan,
            "valor_realizado": round(real, 4) if not np.isnan(real) else np.nan,
            "gap": (round(real - prev, 4)
                    if not (np.isnan(real) or np.isnan(prev)) else np.nan),
        })
    out = pd.DataFrame(rows)
    out.attrs["check_sample"] = check_sample
    return out


def _mono_antigo(seg):
    lv = seg.leaves()
    order = lv["segmento"].tolist()
    todas = list(seg.df[seg.sample_col].dropna().unique())
    amostras = [seg.ref_sample] + [a for a in todas if a != seg.ref_sample]
    rows = []
    for a in amostras:
        vals = []
        for sid in order:
            m = seg.segments[sid]["mask"] & (seg.df[seg.sample_col] == a)
            s = seg.df.loc[m, seg.target]
            vals.append(float(s.mean()) if len(s) else np.nan)
        inv = []
        for i in range(len(vals) - 1):
            v0, v1 = vals[i], vals[i + 1]
            if not (np.isnan(v0) or np.isnan(v1)) and v1 < v0:
                inv.append((int(lv.iloc[i]["nota"]), int(lv.iloc[i + 1]["nota"])))
        rows.append({"amostra": a, "monotonico": len(inv) == 0,
                     "n_inversoes": len(inv), "inversoes": inv})
    return pd.DataFrame(rows)


def _igual(a, b):
    pd.testing.assert_frame_equal(a, b, check_exact=True)


# ═══════════════════════════════ 2.3 ═══════════════════════════════
@pytest.mark.parametrize("dtype", DTYPES)
def test_mascaras_e_lista_iguais_ao_eq(dtype):
    df, seg = _seg("classification", dtype)
    col = df["amostra"]
    assert seg._samples == list(col.dropna().unique())          # mesma ordem
    assert list(seg._sample_masks) == seg._samples
    assert seg._samples[0] == (0.0 if dtype == "num" else "ESTAB")
    for a in seg._samples:
        velho = col == a
        novo = seg._smask(a)
        assert novo is seg._sample_masks[a]
        pd.testing.assert_series_equal(novo, velho, check_exact=True)
        np.testing.assert_array_equal(velho.fillna(False).to_numpy(bool),
                                      novo.fillna(False).to_numpy(bool))
    # código da amostra = posição em _samples (-1 = sem amostra)
    sc = seg._sample_codes
    assert sc.dtype == np.int32 and not sc.flags.writeable
    for k, a in enumerate(seg._samples):
        np.testing.assert_array_equal(sc == k, (col == a).fillna(False).to_numpy(bool))
    np.testing.assert_array_equal(sc == -1, col.isna().to_numpy())
    # conferência cruzada com o get_indexer do plano
    np.testing.assert_array_equal(
        sc, pd.Index(seg._samples).get_indexer(col).astype(np.int32))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rotulo", ["NAO_EXISTE", np.nan, None, "NUNCA_USADA", 99.0])
def test_smask_rotulo_ausente_igual_ao_eq(dtype, rotulo):
    df, seg = _seg("classification", dtype)
    pd.testing.assert_series_equal(seg._smask(rotulo), df["amostra"] == rotulo,
                                   check_exact=True)


def test_smask_rotulo_nao_hasheavel_mesmo_erro():
    df, seg = _seg("classification")
    with pytest.raises(Exception) as e_velho:
        df["amostra"] == ["DES"]                                   # noqa: B015
    with pytest.raises(type(e_velho.value)):
        seg._smask(["DES"])


def test_init_uma_passada_de_unique(monkeypatch):
    df = _base("classification")
    chamadas = []
    orig = pd.Series.unique

    def espia(self, *a, **k):
        if self.name == "amostra":
            chamadas.append(1)
        return orig(self, *a, **k)
    monkeypatch.setattr(pd.Series, "unique", espia)
    with _quieto():
        TreeSegmenter(df, target="target", sample_col="amostra", ref_sample="DES",
                      verbose=False)
    assert len(chamadas) == 1


def test_init_ref_inexistente_mesmo_erro():
    df = _base("classification")
    with pytest.raises(ValueError, match="não encontrada"):
        TreeSegmenter(df, target="target", sample_col="amostra", ref_sample="XYZ",
                      verbose=False)


def _contador_eq(monkeypatch, seg):
    """Conta comparações `==` sobre a coluna de amostra DA BASE INTEIRA (a
    Series de seg.df, com o mesmo objeto de índice) — recortes não contam."""
    n = {"eq": 0}
    orig = pd.Series.__eq__

    def espia(self, other):
        if (self.name == seg.sample_col and len(self) == len(seg.df)
                and self.index is seg.df.index):
            n["eq"] += 1
        return orig(self, other)
    monkeypatch.setattr(pd.Series, "__eq__", espia)
    return n


@pytest.mark.parametrize("task", TASKS)
def test_zero_comparacoes_da_amostra_depois_do_init(monkeypatch, task):
    df, seg = _seg(task, peso=True)
    _cresce(seg)
    oot, estab = "OOT", "ESTAB"
    folhas = [s for s, v in seg.segments.items() if v["is_leaf"]]
    pai = next(s for s, v in seg.segments.items() if not v["is_leaf"])
    n = _contador_eq(monkeypatch, seg)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg._agg_cache.clear()
        seg.leaves(with_psi=True, with_test=True)
        seg.psi(); seg.metrics(); seg.metrics(cutoff=0.3)
        seg.monotonicity_report()
        seg.calibration_table(); seg.calibration_table(check_sample=estab)
        seg.bootstrap_ci(n_boot=20)
        seg.backtest("dt_ref", sample=oot)
        seg.variable_iv(folhas[-1], features=["x", "c"])
        seg.variable_summary("x", sid=folhas[-1], sample=oot)
        seg.variable_table("x", sid=folhas[-1], sample=oot)
        seg.variable_inversion("x", sid=folhas[-1], sample=oot)
        seg._optbin_numeric_bins("x", sid=folhas[-1])
        seg._sibling_sample_series(pai)
        seg.sibling_value_by_safra(pai, sample=oot)
        seg.psi_by_safra(); seg.n_orfas(); seg._value_color_range()
        seg.csi(features=["x"])
        if task == "classification":
            seg._samples_with_both_classes()
    assert n["eq"] == 0
    # rótulo inexistente cai no `==` literal (1 comparação, resultado de antes)
    seg._smask("NAO_EXISTE")
    assert n["eq"] == 1


@pytest.mark.parametrize("dtype", DTYPES)
def test_mascaras_nao_sao_alteradas_pelo_diagnostico(dtype):
    df, seg = _seg("classification", dtype, peso=True)
    _cresce(seg, dtype)
    antes = {a: m.copy() for a, m in seg._sample_masks.items()}
    ids = {a: id(m) for a, m in seg._sample_masks.items()}
    ref, outros = _amostras(dtype)
    folhas = [s for s, v in seg.segments.items() if v["is_leaf"]]
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.leaves(with_psi=True); seg.psi(); seg.metrics()
        seg.monotonicity_report(); seg.calibration_table()
        for a, m in seg._samples_with_both_classes():
            assert all(m is not mm for mm in seg._sample_masks.values())
            m &= False                                  # chamador mutando a cópia
        seg.variable_table("x", sid=folhas[-1], sample=outros[0])
        seg.variable_inversion("x", sid=folhas[-1], sample=outros[0])
    for a, m in seg._sample_masks.items():
        assert id(m) == ids[a]
        pd.testing.assert_series_equal(m, antes[a], check_exact=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_ordered_samples_memo(dtype):
    df, seg = _seg("classification", dtype)
    esperado = _osv_antigo(seg)
    l1 = seg._ordered_samples_with_value()
    assert l1 == esperado and isinstance(l1, list)
    assert isinstance(seg._osv_memo, tuple)
    l1.append("lixo")                                   # chamador altera a lista
    assert seg._ordered_samples_with_value() == esperado


def test_ordered_samples_amostra_sem_alvo():
    df = _base("classification")
    df.loc[df["amostra"] == "ESTAB", "target"] = np.nan     # público só de PSI
    with _quieto():
        seg = TreeSegmenter(df, target="target", sample_col="amostra",
                            ref_sample="DES", verbose=False)
    assert seg._ordered_samples_with_value() == _osv_antigo(seg) == ["DES", "OOT"]


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indice", ["perm", "dup", "range"])
def test_leaves_mono_calib_iguais(task, dtype, indice):
    df, seg = _seg(task, dtype, indice=indice, peso=True)
    _cresce(seg, dtype)
    for estado in range(2):
        if estado == 1:
            _orfas(seg)
        lv = seg.leaves().set_index("segmento")
        velho = _compute_leaves_antigo(seg).loc[lv.index]
        for c in velho.columns:
            pd.testing.assert_series_equal(lv[c], velho[c], check_exact=True)
        _igual(seg.monotonicity_report(), _mono_antigo(seg))
        _, outros = _amostras(dtype)
        for cs in (None, outros[1], "NAO_EXISTE"):
            novo, velho = seg.calibration_table(cs), _calib_antigo(seg, cs)
            _igual(novo, velho)
            assert novo.attrs == velho.attrs


def test_compute_leaves_sem_codigos_usa_caminho_antigo():
    df, seg = _seg("classification", peso=True)
    _cresce(seg)
    novo = seg.leaves()
    seg._sample_codes = None
    seg._agg_cache.clear()
    _igual(seg.leaves(), novo)


def test_ui_reaproveita_mascaras_do_segmentador(monkeypatch):
    from yggdrasil.credit_risk.tree import TreeSegmenterUI
    df = _base("classification")
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui = TreeSegmenterUI(df, target="target", sample_col="amostra",
                             ref_sample="DES", date_col="dt_ref",
                             features=["x", "c", "xd"])
    assert ui._sample_masks is ui.seg._sample_masks
    assert ui._samples == ui.seg._samples and ui._samples is not ui.seg._samples
    assert ui._samples == list(df["amostra"].dropna().unique())
    for a in ui._samples:
        pd.testing.assert_series_equal(ui._sample_masks[a], df["amostra"] == a,
                                       check_exact=True)
    assert ui._nonref == ["ESTAB", "OOT"]
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui._on_reset(None)
    assert ui._sample_masks is ui.seg._sample_masks             # relido no reset
    for a in ui._samples:
        pd.testing.assert_series_equal(ui._sample_masks[a], df["amostra"] == a,
                                       check_exact=True)


# ═══════════════════════════════ 2.4 ═══════════════════════════════
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cow", [False, True])
def test_psi_bincount_igual_crosstab(task, dtype, cow):
    with pd.option_context("mode.copy_on_write", cow):
        df, seg = _seg(task, dtype, peso=(task == "regression"))
        _cresce(seg, dtype)
        # estado 0: árvore com folha vazia no OOT/ESTAB (xd) e amostra NaN em
        # todas as folhas; estado 1: + linhas órfãs
        for estado in range(2):
            if estado == 1:
                _orfas(seg)
                assert seg.n_orfas() > 0
            psi_v, det_v = _psi_antigo(seg)
            _igual(seg.psi(), psi_v)
            _igual(seg.psi_detalhe(), det_v)
            lv = seg.leaves(with_psi=True)
            base = seg.leaves()
            _igual(lv, _append_psi_antigo(seg, base.copy()))
            assert seg.n_orfas() == int(_leaf_id_antigo(seg).isna().sum())


def test_psi_folha_vazia_numa_amostra_e_nan_fora_da_1a_folha():
    df, seg = _seg("classification")
    _cresce(seg)
    sc = seg._sample_codes
    ids, code = seg._leaf_codes()
    # há linhas com amostra NaN em folhas que NÃO são a primeira (o -1 combinado
    # cairia na célula da folha anterior se o filtro viesse depois)
    assert ((sc == -1) & (code >= 1)).any()
    cnt = seg._leaf_sample_counts()
    vazias = [(sid, a) for sid in ids for a in seg._samples if cnt(sid, a) == 0]
    assert vazias                                 # folha sem OOT/ESTAB existe
    velho = pd.crosstab(_leaf_id_antigo(seg), seg.df["amostra"])
    for sid in ids:
        for a in seg._samples:
            esperado = int(velho[a].get(sid, 0)) if a in velho.columns else 0
            assert cnt(sid, a) == esperado
    assert cnt("nao_existe", "DES") == 0


def test_contagem_subconjunto_e_ordem_arbitraria():
    df, seg = _seg("classification")
    _cresce(seg)
    folhas = [s for s, v in seg.segments.items() if v["is_leaf"]]
    for sub in (folhas[::-2], folhas[1:2], folhas[:1] + folhas):
        cnt = seg._leaf_sample_counts(sub)
        velho = pd.crosstab(_leaf_id_antigo(seg, sub), seg.df["amostra"])
        for sid in folhas:
            for a in seg._samples:
                esperado = int(velho[a].get(sid, 0)) if a in velho.columns else 0
                assert cnt(sid, a) == esperado, (sub, sid, a)
        pd.testing.assert_series_equal(seg._leaf_id_series(sub),
                                       _leaf_id_antigo(seg, sub), check_exact=True)


def test_caminho_antigo_sem_codigos_igual():
    df, seg = _seg("classification")
    _cresce(seg)
    _orfas(seg)
    psi, det = seg.psi(), seg.psi_detalhe()
    lv = seg.leaves(with_psi=True)
    seg._sample_codes = None                      # força o crosstab
    seg._agg_cache.clear()
    _igual(seg.psi(), psi)
    _igual(seg.psi_detalhe(), det)
    _igual(seg.leaves(with_psi=True), lv)


def test_mascaras_sobrepostas_desligam_codigos():
    class Coringa:
        """Igual a 'DES' e a 'OOT' (igualdade não transitiva)."""
        def __eq__(self, o):
            return o in ("DES", "OOT")

        def __hash__(self):
            return 12345

    df = _base("classification", n=600)
    am = df["amostra"].to_numpy().copy()
    am[5] = Coringa()
    df["amostra"] = am
    with _quieto():
        seg = TreeSegmenter(df, target="target", sample_col="amostra",
                            ref_sample="DES", verbose=False)
    assert seg._sample_codes is None


def test_indice_duplicado_mantem_o_erro_de_hoje():
    """O crosstab antigo reindexa e quebra com índice duplicado: psi() e
    leaves(with_psi=True) continuam levantando o MESMO erro (deixar de levantar
    é mudança de comportamento, fora desta onda)."""
    df, seg = _seg("classification", indice="dup")
    _cresce(seg)
    with pytest.raises(ValueError, match="duplicate labels") as e_velho:
        _psi_antigo(seg)
    with pytest.raises(ValueError) as e_novo:
        seg.psi()
    assert str(e_novo.value) == str(e_velho.value)
    with pytest.raises(ValueError, match="duplicate labels"):
        seg.leaves(with_psi=True)
    # o resto (metrics, n_orfas, cores) segue funcionando como antes
    assert seg.n_orfas() == int(_leaf_id_antigo(seg).isna().sum())
    seg.metrics()


@pytest.mark.parametrize("task", TASKS)
def test_wrapper_preguicoso_e_read_only(task):
    df, seg = _seg(task)
    _cresce(seg)
    folhas = [s for s, v in seg.segments.items() if v["is_leaf"]]
    seg.psi()                                    # só o código (sem rótulos)
    v = seg._tree_version
    chaves = {k[0][0] for k in seg._agg_cache}
    assert "leaf_codes" in chaves and "leaf_id_arr" not in chaves
    s1 = seg._leaf_id_series()
    s2 = seg._leaf_id_series(folhas)
    assert s1 is not s2 and s1.values is s2.values   # MESMO array em cache
    assert s1.index is seg.df.index and s1.dtype == object and s1.name is None
    pd.testing.assert_series_equal(s1, _leaf_id_antigo(seg), check_exact=True)
    ids, code = seg._leaf_codes()
    assert code.dtype == np.int32 and ids == tuple(folhas)
    arr = seg._agg_cache[(("leaf_id_arr", tuple(folhas)), v)]
    for a in (arr, code, seg._sample_codes):
        assert not a.flags.writeable
        with pytest.raises(ValueError):
            a[0] = a[1]
    # metadados do wrapper são do chamador (não vazam para o cache)
    s1.name = "x"
    assert seg._leaf_id_series().name is None


@pytest.mark.parametrize("task", TASKS)
def test_cache_so_da_versao_atual(task):
    df, seg = _seg(task)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for passo in range(4):
            seg.leaves(with_psi=True); seg.psi(); seg.metrics(); seg.n_orfas()
            seg._value_color_range()
            assert {k[1] for k in seg._agg_cache} == {seg._tree_version}
            f = [s for s, v in seg.segments.items() if v["is_leaf"]]
            seg.grow("x" if passo % 2 == 0 else "c", max_n_bins=2,
                     only_segments=f[-1:])
            # o bump limpou tudo; só sobra o que o grow já leu na versão nova
            assert {k[1] for k in seg._agg_cache} <= {seg._tree_version}
        assert seg._tree_version >= 2
        snap = seg.to_dict()["segments"]
        seg.psi()
        seg._load_segments(snap)                      # undo/redo da UI
        assert {k[1] for k in seg._agg_cache} <= {seg._tree_version}
        seg._bump_version()
        assert len(seg._agg_cache) == 0


@pytest.mark.parametrize("cow", [False, True])
def test_leitores_do_wrapper_iguais(cow):
    """metrics, psi_by_safra e a faixa de cores leem a série de rótulos — com o
    wrapper read-only (e sob copy-on-write) dão o mesmo resultado de antes."""
    with pd.option_context("mode.copy_on_write", cow):
        df, seg = _seg("classification")
        _cresce(seg)
        _orfas(seg)
        novo = (seg.metrics(), seg.metrics(cutoff=0.2), seg.psi_by_safra(),
                seg._value_color_range())
        seg._leaf_id_series = lambda leaf_ids=None: _leaf_id_antigo(seg, leaf_ids)
        seg._agg_cache.clear()
        velho = (seg.metrics(), seg.metrics(cutoff=0.2), seg.psi_by_safra(),
                 seg._value_color_range())
        for a, b in zip(novo[:3], velho[:3]):
            _igual(a, b)
        assert novo[3] == velho[3]
