"""Onda 2 (item 2.5) da árvore: ``metrics()`` por contagens por folha.

Na classificação, ``_compute_metrics`` passa a montar um bincount
folha×amostra×classe e chama roc_curve/accuracy/f1 com ``sample_weight`` sobre
2L linhas sintéticas, em vez de ordenar as N linhas de cada amostra. A
regressão não muda. A referência ``_metrics_antigo`` abaixo é cópia LITERAL do
código anterior, e tudo é comparado com ``assert_frame_equal(check_exact=True)``.
"""
from __future__ import annotations

import contextlib
import io
import time
import warnings

import numpy as np
import pandas as pd
import pytest

from yggdrasil.credit_risk.tree import TreeSegmenter
from yggdrasil.credit_risk.tree import segmenter as segmod
import yggdrasil.metrics.classification as clfmod

CUTOFFS = [None, 0.1, 0.5]


def _quieto():
    return contextlib.redirect_stdout(io.StringIO())


def _limpa_memo(seg):
    """Força o recálculo do metrics(): memo por versão E a LRU por assinatura."""
    seg._agg_cache.clear()
    seg._agg_lru.clear()


# ─────────────── referência LITERAL do código anterior ───────────────
def _pd_map_antigo(seg, mutante_sem_amostra_fora_do_all=False):
    """pd_map do caminho anterior, LITERAL. ``mutante_sem_amostra_fora_do_all``
    reproduz o DEFEITO que a 2.5 tem de evitar (linhas com sample_col NaN fora
    do fallback all_pd) — só para provar que o cenário do teste o distingue."""
    self = seg
    leaf_ids = [sid for sid, s in self.segments.items() if s["is_leaf"]]
    if self.sample_col is not None:
        ref_mask = self._mask_ref
    else:
        ref_mask = pd.Series(True, index=self.df.index)
    overall = self.df.loc[ref_mask, self.target].mean()

    leaf_id = self._leaf_id_series(leaf_ids)
    ref_pd = self.df.loc[ref_mask, self.target].groupby(leaf_id[ref_mask]).mean()
    all_pd = self.df[self.target].groupby(leaf_id).mean()
    if mutante_sem_amostra_fora_do_all:
        com = self.df[self.sample_col].notna()
        all_pd = self.df.loc[com, self.target].groupby(leaf_id[com]).mean()
    pd_map = {}
    for sid in leaf_ids:
        v = ref_pd.get(sid, np.nan)
        if pd.isna(v):
            v = all_pd.get(sid, np.nan)
        pd_map[sid] = float(v) if pd.notna(v) else float(overall)
    return leaf_ids, leaf_id, pd_map


def _metrics_antigo(seg, cutoff=None, mutante_sem_amostra_fora_do_all=False):
    self = seg
    if self._is_clf:
        from yggdrasil.metrics.classification import classification_metrics

    leaf_ids, leaf_id, pd_map = _pd_map_antigo(
        seg, mutante_sem_amostra_fora_do_all)
    pred = leaf_id.map(pd_map).astype("float64")

    if self.sample_col is not None:
        todas = list(self._samples)
        ordem = ([self.ref_sample] + [a for a in todas if a != self.ref_sample])
        grupos = [(a, self._sample_masks[a]) for a in ordem]
    else:
        grupos = [("todos", pd.Series(True, index=self.df.index))]

    linhas = []
    for nome, mask in grupos:
        y = self.df.loc[mask, self.target].to_numpy(dtype="float64")
        yhat = pred[mask.values].to_numpy(dtype="float64")
        valid = ~(np.isnan(y) | np.isnan(yhat))
        y, yhat = y[valid], yhat[valid]
        base = {"amostra": nome, "n": int(mask.sum())}
        if self._is_clf:
            if len(y) == 0:
                base.update({"taxa_default": np.nan, "KS": np.nan, "AUC": np.nan,
                             "Gini": np.nan, "Acuracia": np.nan, "F1": np.nan})
            else:
                cm = classification_metrics(y, yhat, cutoff=cutoff)
                base.update({
                    "taxa_default": round(float(np.mean(y)), 4),
                    "KS": cm["ks"], "AUC": cm["auc"], "Gini": cm["gini"],
                    "Acuracia": cm["accuracy"], "F1": cm["f1"],
                })
        else:
            if len(y) == 0:
                base.update({"MAE": np.nan, "RMSE": np.nan, "R2": np.nan})
            else:
                err = y - yhat
                ss_res = float(np.sum(err ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                base.update({
                    "MAE": round(float(np.mean(np.abs(err))), 4),
                    "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 4),
                    "R2": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else np.nan,
                })
        linhas.append(base)
    return pd.DataFrame(linhas)


# ─────────────── bases ───────────────
def _base(n=6000, seed=5, task="classification", amostra_dtype="object",
          indice="perm"):
    rng = np.random.default_rng(seed)
    am = rng.choice(["ESTAB", "DES", "OOT"], n, p=[0.2, 0.5, 0.3]).astype(object)
    am[0], am[1] = "ESTAB", "OOT"
    am[rng.random(n) < 0.03] = np.nan
    am[rng.random(n) < 0.01] = None
    u1, u2 = rng.random(n), rng.random(n)
    # "so_fora" > 0.5 só fora da DES (OOT/ESTAB/sem amostra): folha SEM DES
    so_fora = np.where(am == "DES", rng.random(n) * 0.5, rng.random(n))
    z = 2.0 * u1 - 1.5 * u2 + rng.normal(size=n)
    if task == "classification":
        y = (z > 0.4).astype(float)
    else:
        y = 1 / (1 + np.exp(-z))
    y[rng.random(n) < 0.03] = np.nan
    df = pd.DataFrame({"u1": u1, "u2": u2, "so_fora": so_fora,
                       "amostra": am, "target": y})
    if amostra_dtype == "category":
        df["amostra"] = pd.Categorical(df["amostra"],
                                       categories=["DES", "OOT", "ESTAB", "NUNCA"])
    elif amostra_dtype == "string":
        df["amostra"] = df["amostra"].astype("string")
    elif amostra_dtype == "num":
        df["amostra"] = df["amostra"].map({"DES": 1.0, "OOT": 2.0, "ESTAB": 0.0})
    if indice == "dup":
        df.index = np.r_[np.arange(n // 2), np.arange(n - n // 2)]
    elif indice == "perm":
        df.index = rng.permutation(n) * 3 + 7
    return df


def _ref(amostra_dtype):
    return 1.0 if amostra_dtype == "num" else "DES"


def _seg(df, task="classification", sample_col="amostra", ref="DES"):
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TreeSegmenter(df, target="target", task_type=task,
                             sample_col=sample_col,
                             ref_sample=ref if sample_col else None,
                             verbose=False)


def _folhas(seg):
    return [s for s, v in seg.segments.items() if v["is_leaf"]]


def _arvore(seg, n_folhas):
    """Árvores de 1, 5, 13 ou 40 folhas por cortes explícitos. As variáveis não
    têm NaN, então na_to_worst=False não muda nada — só evita o erro antigo do
    _assign_na_to_worst com máscara de amostra StringDtype (pd.NA)."""
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if n_folhas == 1:
            pass
        elif n_folhas == 5:
            seg.grow("u1", splits=[0.2, 0.4, 0.6, 0.8], na_to_worst=False)
        elif n_folhas == 13:
            seg.grow("u1", splits=[0.2, 0.4, 0.6, 0.8], na_to_worst=False)
            seg.grow("u2", splits=[0.33, 0.66], only_segments=_folhas(seg)[:4],
                     na_to_worst=False)
        elif n_folhas == 40:
            seg.grow("u1", splits=[0.125 * k for k in range(1, 8)],
                     na_to_worst=False)
            seg.grow("u2", splits=[0.2, 0.4, 0.6, 0.8], na_to_worst=False)
        else:
            raise ValueError(n_folhas)
    assert len(_folhas(seg)) == n_folhas
    return seg


def _scores_iguais(seg):
    """Score (pd_map) por folha do caminho por contagens × antigo literal, no
    MESMO float (NaN = NaN). Devolve False se a tabela não foi montada."""
    tab = seg._metrics_clf_tabela(_folhas(seg))
    if tab is None:
        return False
    ids, _pos, _neg, score, _col = tab
    leaf_ids, _lid, pd_map = _pd_map_antigo(seg)
    assert list(ids) == list(leaf_ids)
    velho = np.array([pd_map[s] for s in leaf_ids], dtype="float64")
    np.testing.assert_array_equal(score, velho)       # exato; NaN casa com NaN
    return True


def _compara(seg, cutoffs=CUTOFFS, rapido=None):
    """metrics() novo × antigo literal, check_exact. ``rapido``: None = não
    confere; True/False = exige que o caminho por contagens (não) seja usado.
    Com o caminho rápido, confere também o score de CADA folha."""
    if rapido:
        assert _scores_iguais(seg)
    for cut in cutoffs:
        if rapido is not None:
            r = seg._metrics_clf_contagens(cut, _folhas(seg))
            assert (r is not None) == rapido
        _limpa_memo(seg)
        novo = seg.metrics(cutoff=cut)
        velho = _metrics_antigo(seg, cut)
        pd.testing.assert_frame_equal(novo, velho, check_exact=True)


@pytest.fixture
def conta_cm(monkeypatch):
    """Conta as chamadas a classification_metrics (só o caminho antigo usa)."""
    n = {"c": 0}
    orig = clfmod.classification_metrics

    def espiao(*a, **k):
        n["c"] += 1
        return orig(*a, **k)
    monkeypatch.setattr(clfmod, "classification_metrics", espiao)
    return n


# ─────────────── testes ───────────────
@pytest.mark.parametrize("n_folhas", [1, 5, 13, 40])
@pytest.mark.parametrize("amostra_dtype", ["object", "category", "string", "num"])
def test_metrics_clf_igual_ao_antigo(n_folhas, amostra_dtype, conta_cm):
    df = _base(amostra_dtype=amostra_dtype)
    seg = _arvore(_seg(df, ref=_ref(amostra_dtype)), n_folhas)
    assert _scores_iguais(seg)
    _compara(seg)
    # cutoff exatamente igual a um score de folha (empate no >=)
    lid = seg._leaf_id_series()
    ref = seg._mask_ref
    pdm = seg.df.loc[ref, "target"].groupby(lid[ref]).mean().dropna()
    _compara(seg, [float(pdm.iloc[0]), float(pdm.iloc[-1])])
    # todas as chamadas do metrics() novo foram pelo caminho rápido; as do
    # antigo (referência) chamam classification_metrics 3 vezes por cutoff
    conta_cm["c"] = 0
    _limpa_memo(seg)
    for cut in CUTOFFS:
        seg.metrics(cutoff=cut)
    assert conta_cm["c"] == 0


@pytest.mark.parametrize("indice", ["range", "dup"])
def test_metrics_clf_indices(indice):
    seg = _arvore(_seg(_base(indice=indice)), 13)
    _compara(seg, rapido=True)


def test_metrics_clf_sem_sample_col():
    df = _base().drop(columns="amostra")
    seg = _arvore(_seg(df, sample_col=None), 13)
    _compara(seg, rapido=True)
    assert list(seg.metrics()["amostra"]) == ["todos"]


def _base_sem_des(repr_nan, seed=3):
    """Base determinística com uma folha (g=1) SEM linha DES e com linhas de
    sample_col ausente, em que essas linhas MUDAM o score da folha e a ORDEM
    entre folhas:

    - g=0: DES 30/100 maus, OOT 30/100, ESTAB 15/50          → score 0.30
    - g=1: sem DES; OOT 10/100, ESTAB 5/50, SEM AMOSTRA 180/200 (+ 6 linhas de
      alvo NaN sem amostra)                                   → 195/350 ≈ 0.557
      (sem as linhas de amostra ausente seria 15/150 = 0.10)
    - g=2: DES 45/100, OOT 45/100, ESTAB 20/50                → score 0.45

    Certo: g1 > g2 > g0. Mutante: g2 > g0 > g1 — o AUC/KS da OOT/ESTAB muda.
    ``repr_nan``: como a amostra ausente aparece na coluna."""
    blocos = []

    def bloco(g, am, n, maus, alvo_nan=0):
        y = np.r_[np.ones(maus), np.zeros(n - maus), np.full(alvo_nan, np.nan)]
        blocos.append(pd.DataFrame({"g": float(g), "amostra": [am] * len(y),
                                    "target": y}))
    for am, n, maus in [("DES", 100, 30), ("OOT", 100, 30), ("ESTAB", 50, 15)]:
        bloco(0, am, n, maus)
    for am, n, maus in [("OOT", 100, 10), ("ESTAB", 50, 5)]:
        bloco(1, am, n, maus)
    bloco(1, "SEM", 200, 180, alvo_nan=6)
    for am, n, maus in [("DES", 100, 45), ("OOT", 100, 45), ("ESTAB", 50, 20)]:
        bloco(2, am, n, maus)
    df = pd.concat(blocos, ignore_index=True)
    rng = np.random.default_rng(seed)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    df.index = rng.permutation(len(df)) * 5 + 11          # índice embaralhado
    sem = (df["amostra"] == "SEM").to_numpy()
    am = df["amostra"].to_numpy(dtype=object).copy()
    if repr_nan == "nan":
        am[sem] = np.nan
    elif repr_nan == "none":
        am[sem] = None
    elif repr_nan == "misto":
        idx = np.flatnonzero(sem)
        am[idx[::2]] = np.nan
        am[idx[1::2]] = None
    df["amostra"] = am
    if repr_nan == "string":
        df.loc[sem, "amostra"] = None
        df["amostra"] = df["amostra"].astype("string")          # pd.NA
    elif repr_nan == "category":
        df.loc[sem, "amostra"] = None
        df["amostra"] = pd.Categorical(df["amostra"],
                                       categories=["OOT", "DES", "ESTAB", "X"])
    elif repr_nan == "num":
        df["amostra"] = df["amostra"].map({"DES": 1.0, "OOT": 2.0, "ESTAB": 0.0})
    return df


@pytest.mark.parametrize("repr_nan",
                         ["nan", "none", "misto", "string", "category", "num"])
def test_metrics_clf_folha_sem_des_com_amostra_ausente(repr_nan, conta_cm):
    """Ajuste 2 dos revisores: as linhas com sample_col ausente ENTRAM no
    fallback all_pd da folha sem DES, exatamente como no caminho antigo. O
    cenário é montado para que tirá-las MUDE o score da folha e a ordem entre
    folhas — e o teste prova isso contra o próprio caminho antigo mutado."""
    df = _base_sem_des(repr_nan)
    ref = _ref(repr_nan)
    seg = _seg(df, ref=ref)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.grow("g", splits=[0.5, 1.5], na_to_worst=False)
    folhas = _folhas(seg)
    assert len(folhas) == 3
    lid = seg._leaf_id_series()
    sem_amostra = seg.df["amostra"].isna()
    assert int(sem_amostra.sum()) == 206
    f1 = lid[(seg.df["g"] == 1.0).values].iloc[0]
    assert not (lid[seg._mask_ref] == f1).any()            # folha sem DES
    assert (sem_amostra & (lid == f1)).sum() == 206

    # o cenário DISTINGUE o certo do defeito (no próprio caminho antigo)
    _, _, certo = _pd_map_antigo(seg)
    _, _, mut = _pd_map_antigo(seg, mutante_sem_amostra_fora_do_all=True)
    assert certo[f1] == 195 / 350 and mut[f1] == 15 / 150
    for cut in CUTOFFS:
        a = _metrics_antigo(seg, cut)
        b = _metrics_antigo(seg, cut, mutante_sem_amostra_fora_do_all=True)
        assert not a.equals(b), cut
        # o defeito muda o RANKING (AUC da OOT), não só Acurácia/F1
        assert (a.set_index("amostra").loc[["OOT", "ESTAB"] if ref == "DES"
                                           else [2.0, 0.0], "AUC"]
                != b.set_index("amostra").loc[["OOT", "ESTAB"] if ref == "DES"
                                              else [2.0, 0.0], "AUC"]).all()

    # score de CADA folha idêntico ao antigo (inclusive o 195/350 da folha
    # sem DES) e métricas check_exact, sempre pelo caminho por contagens
    tab = seg._metrics_clf_tabela(folhas)
    assert tab is not None
    ids, pos, neg, score, coluna = tab
    assert score[list(ids).index(f1)] == 195 / 350
    # a coluna extra (sem amostra) tem as 200 linhas válidas da folha g=1
    assert pos[list(ids).index(f1), -1] == 180
    assert neg[list(ids).index(f1), -1] == 20
    assert pos[:, -1].sum() + neg[:, -1].sum() == 200
    conta_cm["c"] = 0
    _compara(seg, CUTOFFS + [0.2, 0.35, 0.5, 195 / 350, 15 / 150], rapido=True)
    # 8 cutoffs × 3 amostras no antigo (referência); 0 no novo
    assert conta_cm["c"] == 8 * 3


def test_metrics_clf_folha_sem_des_e_amostra_nan():
    """Base aleatória: folhas só com OOT/ESTAB/amostra NaN — score = média de
    TODAS as linhas da folha (inclusive as de sample_col NaN/None)."""
    df = _base()
    seg = _seg(df)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.grow("so_fora", splits=[0.5], na_to_worst=False)
        seg.grow("u1", splits=[0.5], na_to_worst=False)
    lid = seg._leaf_id_series()
    des = seg._mask_ref
    sem_des = [s for s in _folhas(seg) if not (lid[des] == s).any()]
    assert sem_des, "precisa de pelo menos uma folha sem DES"
    am_nan = seg.df["amostra"].isna()
    assert (am_nan & lid.isin(sem_des)).any()
    _, _, certo = _pd_map_antigo(seg)
    _, _, mut = _pd_map_antigo(seg, mutante_sem_amostra_fora_do_all=True)
    for s in sem_des:
        assert certo[s] != mut[s], "a folha precisa distinguir os scores"
    # scores por folha conferidos dentro do _compara (rapido=True). Cutoffs:
    # EXATAMENTE em cada um dos dois scores (empate no >=) e no ponto médio —
    # um score que ignore as linhas de amostra NaN inverte o y_pred da folha.
    cortes = []
    for s in sem_des:
        cortes += [certo[s], mut[s], (certo[s] + mut[s]) / 2]
    _compara(seg, CUTOFFS + cortes, rapido=True)
    assert any(not _metrics_antigo(seg, c).equals(
        _metrics_antigo(seg, c, mutante_sem_amostra_fora_do_all=True))
        for c in cortes)


def test_metrics_clf_duas_folhas_mesmo_score():
    """Duas folhas com o MESMO pd_map (empate no roc_curve)."""
    g = np.r_[np.zeros(50), np.ones(100), np.full(100, 2.0), np.full(60, 3.0)]
    y = np.r_[np.r_[np.ones(10), np.zeros(40)],       # 0.2
              np.r_[np.ones(20), np.zeros(80)],       # 0.2
              np.r_[np.ones(30), np.zeros(70)],       # 0.3
              np.r_[np.ones(30), np.zeros(30)]]       # 0.5
    rng = np.random.default_rng(0)
    am = np.where(rng.random(len(g)) < 0.7, "DES", "OOT").astype(object)
    am[0], am[1] = "DES", "OOT"
    df = pd.DataFrame({"g": g, "amostra": am, "target": y})
    # força o DES a reproduzir as taxas exatas: todo mundo na DES, OOT duplicado
    df_oot = df.copy()
    df_oot["amostra"] = "OOT"
    df = pd.concat([df.assign(amostra="DES"), df_oot], ignore_index=True)
    seg = _seg(df)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.grow("g", splits=[0.5, 1.5, 2.5], na_to_worst=False)
    lid = seg._leaf_id_series()
    pdm = seg.df.loc[seg._mask_ref, "target"].groupby(lid[seg._mask_ref]).mean()
    assert pdm.duplicated().any()
    _compara(seg, CUTOFFS + [0.2, 0.3], rapido=True)


def test_metrics_clf_amostra_de_uma_classe():
    """OOT só com 0 (e uma folha com score >= 0.5) e ESTAB só com 1: corte 0.5,
    KS/AUC/Gini NaN — como o _roc_pack com np.unique(y) < 2."""
    df = _base()
    df.loc[df["amostra"] == "OOT", "target"] = 0.0
    df.loc[df["amostra"] == "ESTAB", "target"] = 1.0
    seg = _arvore(_seg(df), 13)
    lid = seg._leaf_id_series()
    pdm = seg.df.loc[seg._mask_ref, "target"].groupby(lid[seg._mask_ref]).mean()
    assert (pdm >= 0.5).any()
    _compara(seg, rapido=True)
    m = seg.metrics()
    assert m.set_index("amostra").loc["OOT", ["KS", "AUC", "Gini"]].isna().all()
    assert m.set_index("amostra").loc["ESTAB", ["KS", "AUC", "Gini"]].isna().all()


def test_metrics_clf_amostra_sem_linha_valida():
    df = _base()
    df.loc[df["amostra"] == "ESTAB", "target"] = np.nan
    seg = _arvore(_seg(df), 5)
    _compara(seg, rapido=True)
    assert seg.metrics().set_index("amostra").loc["ESTAB"].drop("n").isna().all()


def test_metrics_clf_referencia_toda_nan():
    """Alvo todo NaN na referência: overall NaN; folhas sem DES válida caem em
    all_pd e, onde também não há, ficam com score NaN (fora das métricas)."""
    df = _base()
    seg = _seg(df)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.grow("so_fora", splits=[0.5], na_to_worst=False)
    seg.df.loc[seg._mask_ref.values, "target"] = np.nan
    lid = seg._leaf_id_series()
    # a folha "<= 0.5" só tem alvo fora da DES nas outras amostras → all_pd; zera
    # o alvo dela inteira para chegar ao overall NaN
    f0 = _folhas(seg)[0]
    seg.df.loc[(lid == f0).values, "target"] = np.nan
    seg._bump_version()
    lid = seg._leaf_id_series()
    assert np.isnan(seg.df.loc[seg._mask_ref, "target"].mean())
    assert seg.df.loc[(lid == f0).values, "target"].isna().all()
    _compara(seg, rapido=True)


def test_metrics_clf_orfas():
    seg = _arvore(_seg(_base()), 13)
    f = _folhas(seg)
    seg.segments[f[2]]["is_leaf"] = False
    seg.segments[f[7]]["is_leaf"] = False
    seg._bump_version()
    _compara(seg, rapido=True)


@pytest.mark.parametrize("alvo", ["int64", "Int64", "Float64", "bool", "uint8"])
def test_metrics_clf_dtypes_do_alvo(alvo, conta_cm):
    df = _base()
    if alvo in ("int64", "bool", "uint8"):
        df["target"] = df["target"].fillna(0).astype(alvo)
    else:
        df["target"] = df["target"].astype(alvo)       # NaN → pd.NA
    seg = _arvore(_seg(df), 13)
    conta_cm["c"] = 0
    _compara(seg)
    # 3 cutoffs × 3 amostras no antigo; 0 no novo
    assert conta_cm["c"] == 9


@pytest.mark.parametrize("alvo", ["float32", "Float32", "object"])
def test_metrics_clf_dtype_fora_usa_caminho_antigo(alvo, conta_cm):
    df = _base()
    seg = _arvore(_seg(df), 5)
    seg.df["target"] = seg.df["target"].astype(alvo)
    seg._bump_version()
    conta_cm["c"] = 0
    seg.metrics()
    assert conta_cm["c"] == 3
    assert seg._metrics_clf_contagens(None, _folhas(seg)) is None
    _compara(seg)


def test_metrics_clf_alvo_fora_de_01_preserva_erro():
    seg = _arvore(_seg(_base()), 5)
    seg.df.loc[seg.df.index[5], "target"] = 2.0
    seg._bump_version()
    assert seg._metrics_clf_contagens(None, _folhas(seg)) is None
    with pytest.raises(ValueError) as e_velho:
        _metrics_antigo(seg)
    with pytest.raises(ValueError) as e_novo:
        seg.metrics()
    assert str(e_novo.value) == str(e_velho.value)


def test_metrics_clf_menos_zero_usa_caminho_antigo():
    seg = _arvore(_seg(_base()), 5)
    y = seg.df["target"].to_numpy(copy=True)
    y[y == 0] = -0.0
    seg.df["target"] = y
    seg._bump_version()
    assert seg._metrics_clf_contagens(None, _folhas(seg)) is None
    _compara(seg)


def test_metrics_clf_overall_na_anulavel_preserva_erro():
    """Int64 com a referência toda NA: overall = pd.NA e float(pd.NA) levanta
    no caminho antigo; o novo devolve None e reproduz o mesmo erro."""
    df = _base()
    df["target"] = df["target"].astype("Int64")
    seg = _seg(df)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.grow("so_fora", splits=[0.5], na_to_worst=False)
    lid = seg._leaf_id_series()
    seg.df.loc[seg._mask_ref.values, "target"] = pd.NA
    f0 = _folhas(seg)[0]
    seg.df.loc[(lid == f0).values, "target"] = pd.NA
    seg._bump_version()
    erros = []
    for fn in (lambda: _metrics_antigo(seg), lambda: seg.metrics()):
        try:
            fn()
            erros.append(None)
        except Exception as e:     # noqa: BLE001
            erros.append((type(e), str(e)))
    assert erros[0] == erros[1]


def _float64_nan_nao_mascarado(y, idx_nan):
    """Float64 anulável com NaN NÃO mascarado em ``idx_nan`` (como o 0/0 de
    uma Series Float64 gera): isna() dá False para ele; o NaN original de
    ``y`` vira NA mascarado."""
    v = np.array(y, dtype="float64", copy=True)
    m = np.isnan(v)
    v[m] = 0.0
    v[idx_nan] = np.nan
    return pd.arrays.FloatingArray(v, m)


def _resultado(fn):
    try:
        return fn()
    except Exception as e:     # noqa: BLE001
        return (type(e), str(e))


@pytest.mark.parametrize("onde", ["folha_sem_des", "des"])
def test_metrics_clf_float64_nan_nao_mascarado_usa_caminho_antigo(onde,
                                                                    conta_cm):
    """NaN não mascarado no Float64: o antigo o propaga (média da folha NaN,
    overall pd.NA → score=overall e KS/AUC 0/0,5 em OOT/ESTAB; ou
    TypeError(float(NAType)) quando cai na DES). O caminho por contagens o
    trataria como ausente, então tem de devolver None e reproduzir o antigo."""
    df = _base()
    y = df["target"].to_numpy(dtype="float64")
    am = df["amostra"].to_numpy()
    if onde == "folha_sem_des":
        sel = (am != "DES") & (df["so_fora"] > 0.5).to_numpy() & ~np.isnan(y)
    else:
        sel = (am == "DES") & ~np.isnan(y)
    df["target"] = _float64_nan_nao_mascarado(y, np.flatnonzero(sel)[:3])
    assert df["target"].isna().sum() < np.isnan(
        df["target"].to_numpy(dtype="float64")).sum()
    seg = _seg(df)
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if onde == "folha_sem_des":
            seg.grow("so_fora", splits=[0.5], na_to_worst=False)
        else:
            seg.grow("u1", splits=[0.5], na_to_worst=False)
    for cut in CUTOFFS:
        assert seg._metrics_clf_contagens(cut, _folhas(seg)) is None
        _limpa_memo(seg)
        conta_cm["c"] = 0
        novo = _resultado(lambda: seg.metrics(cutoff=cut))
        velho = _resultado(lambda: _metrics_antigo(seg, cut))
        if isinstance(velho, tuple):
            # o erro sai antes do classification_metrics (float(overall))
            assert novo == velho
        else:
            assert conta_cm["c"] > 0        # passou pelo caminho antigo
            pd.testing.assert_frame_equal(novo, velho, check_exact=True)
    if onde == "des":
        # o antigo levanta aqui (float(pd.NA)); o novo tem de levantar igual
        assert isinstance(velho, tuple) and velho[0] is TypeError


def test_metrics_clf_float64_so_na_mascarado_segue_rapido():
    """A guarda nova não derruba o caso normal: Float64 só com NA mascarado
    continua no caminho por contagens."""
    df = _base()
    df["target"] = df["target"].astype("Float64")
    seg = _arvore(_seg(df), 5)
    _compara(seg, rapido=True)


def test_metrics_sem_codigos_de_amostra_usa_caminho_antigo(conta_cm):
    seg = _arvore(_seg(_base()), 5)
    seg._sample_codes = None
    conta_cm["c"] = 0
    seg.metrics()
    assert conta_cm["c"] == 3
    _compara(seg)


@pytest.mark.parametrize("cow", [False, True])
def test_metrics_clf_copy_on_write(cow):
    with pd.option_context("mode.copy_on_write", cow):
        seg = _arvore(_seg(_base()), 13)
        _compara(seg, rapido=True)


@pytest.mark.parametrize("n_folhas", [1, 13])
def test_metrics_regressao_inalterada(n_folhas, monkeypatch):
    chamou = {"c": 0}
    orig = TreeSegmenter._metrics_clf_contagens

    def espiao(self, *a, **k):
        chamou["c"] += 1
        return orig(self, *a, **k)
    monkeypatch.setattr(TreeSegmenter, "_metrics_clf_contagens", espiao)
    seg = _arvore(_seg(_base(task="regression"), task="regression"), n_folhas)
    _compara(seg)
    assert chamou["c"] == 0


def test_metrics_memo_por_versao():
    seg = _arvore(_seg(_base()), 5)
    a = seg.metrics()
    b = seg.metrics()
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    a.loc[0, "KS"] = -1.0                          # cópia defensiva
    assert seg.metrics().loc[0, "KS"] != -1.0
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg.grow("u2", splits=[0.5], only_segments=_folhas(seg)[:1],
                 na_to_worst=False)
    _compara(seg)


def test_metrics_clf_mais_rapido():
    """Base de 200k: o caminho por contagens não pode ser mais lento."""
    df = _base(n=200_000, seed=11)
    seg = _arvore(_seg(df), 40)
    t0 = time.perf_counter()
    for _ in range(3):
        _limpa_memo(seg)
        seg.metrics()
    novo = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(3):
        _metrics_antigo(seg)
    velho = time.perf_counter() - t0
    assert novo < velho, (novo, velho)


def test_modulo_exporta_metodos():
    for nome in ("_metrics_clf_contagens", "_metrics_clf_tabela",
                 "_metrics_clf_de_tabela"):
        assert hasattr(segmod.TreeSegmenter, nome)
