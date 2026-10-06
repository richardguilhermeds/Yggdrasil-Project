"""Onda 3.2 de desempenho da TreeSegmenter: código de faixa por linha.

``_bin_ids`` calcula, numa passada por coluna, o índice da faixa de cada linha
(int32; -1 = nenhuma). ``variable_iv`` recorta só a coluna da folha∩amostra (sem o
frame com todas as variáveis) e conta por ``bincount``; os laços de análise
(``_bin_table``, ``csi``, ``variable_table``, ``variable_psi_by_safra``,
``variable_inversion``, ``suggest_splits`` e o gráfico de share por safra)
usam ``ids == k`` no lugar de um ``_mask_in`` por faixa.

Tudo tem de dar o MESMO resultado de antes (igualdade exata):
- ``variable_iv`` é comparada com a cópia LITERAL da versão antiga (abaixo);
- ``(ids == k)`` é comparado com ``_mask_in(...).to_numpy()`` faixa a faixa;
- os laços são comparados com o caminho antigo forçado (``_bin_ids`` → None,
  que volta a um ``_mask_in`` por faixa — o código de antes).
"""
from __future__ import annotations

import tracemalloc
import warnings
from decimal import Decimal

import matplotlib
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from yggdrasil.credit_risk.tree import TreeSegmenter  # noqa: E402
from yggdrasil.credit_risk.tree import segmenter as segmod  # noqa: E402

# Ajustes independentes comparados com igualdade exata: CP-SAT com 1 worker
# (ver ``cpsat_deterministico`` no conftest) para o empate do ótimo não virar
# falha intermitente.
pytestmark = pytest.mark.usefixtures("cpsat_deterministico")


# ----------------------------------------------------------------------
# Versão antiga (cópia literal de variable_iv antes da onda 3.2)
# ----------------------------------------------------------------------
def _variable_iv_velho(self, sid=None, features=None, max_n_bins=5, min_bin_size=0.05,
                       cutoff="median", with_psi=True):
    _psi_from_shares = segmod._psi_from_shares
    _classifica_iv = segmod._classifica_iv
    _classifica_psi = segmod._classifica_psi
    if features is None:
        features = [c for c in self.df.columns if c not in self._nonfeature_cols()]

    if sid is None or sid == "root" or sid not in self.segments:
        leaf_mask = pd.Series(True, index=self.df.index)
    else:
        leaf_mask = self.segments[sid]["mask"]
    if self.sample_col is not None:
        ref_mask = self._smask(self.ref_sample)
        nonref = [a for a in self._samples
                  if a != self.ref_sample]
    else:
        ref_mask = pd.Series(True, index=self.df.index)
        nonref = []
    psi_on = with_psi and bool(nonref)

    base = self._sub(leaf_mask & ref_mask, *features)  # porção DES da folha (referência)
    yb = base[self.target].to_numpy(dtype="float64")
    yb = yb[~np.isnan(yb)]
    if self._is_clf:
        ybin = yb.astype(int)
        tot_bad = int(ybin.sum())
        tot_good = int(ybin.size - tot_bad)
        # IV binário precisa das DUAS classes presentes na folha (DES)
        ok_base = ybin.size >= 4 and tot_bad > 0 and tot_good > 0
    else:
        tot_bad = tot_good = 0
        # IV contínuo exige variação do alvo na folha (DES)
        ok_base = yb.size >= 4 and np.unique(yb).size > 1
    mean_global = float(yb.mean()) if yb.size else np.nan
    n_base = int(yb.size)

    # frames e contagens por amostra DENTRO da folha (para o PSI nos bins do IV)
    ref_frame = cur_frames = None
    n_ref_leaf = 0
    if psi_on:
        ref_frame = base                     # mesma máscara: folha ∩ DES
        n_ref_leaf = len(ref_frame)
        cur_frames = {a: self._sub(leaf_mask & self._smask(a), *features)
                      for a in nonref}

    rows = []
    for feat in features:
        iv, nb, kind = np.nan, 0, "—"
        psi_vals = {a: np.nan for a in nonref}
        if ok_base:
            try:
                bins, _modo, kind = self._resolve_bins_cached(
                    sid, feat, None, None, max_n_bins, min_bin_size)
                nb = len(bins)
                if bins:
                    iv = 0.0
                    yb_feat = base[self.target].to_numpy(dtype="float64")
                    for b in bins:
                        m = self._mask_in(base, feat, b).to_numpy()
                        yi = yb_feat[m]
                        yi = yi[~np.isnan(yi)]
                        if yi.size == 0:
                            continue
                        if self._is_clf:
                            yi = yi.astype(int)
                            bad_i = int(yi.sum()); good_i = int(yi.size - bad_i)
                            dg = good_i / tot_good
                            db = bad_i / tot_bad
                            if dg > 0 and db > 0:
                                iv += (dg - db) * np.log(dg / db)
                        else:
                            iv += (yi.size / n_base) * abs(float(yi.mean()) - mean_global)
                    # PSI sobre os MESMOS bins (DES × amostra, na folha)
                    if psi_on and n_ref_leaf > 0:
                        ref_shares = [int(self._mask_in(ref_frame, feat, b).sum())
                                      / n_ref_leaf for b in bins]
                        for a in nonref:
                            n_a = len(cur_frames[a])
                            if n_a == 0:
                                continue
                            cur_shares = [int(self._mask_in(cur_frames[a], feat,
                                                            b).sum()) / n_a
                                          for b in bins]
                            psi_vals[a] = round(_psi_from_shares(
                                ref_shares, cur_shares, eps=1e-6), 4)
            except Exception:
                iv = np.nan
        row = {
            "variavel": feat, "tipo": kind, "n_bins": nb,
            "iv": round(float(iv), 4) if not (iv is None or np.isnan(iv)) else np.nan,
            "forca": _classifica_iv(iv, self.task_type),
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
    out.attrs["valor_medio"] = round(mean_global, 4) if not np.isnan(mean_global) else np.nan
    out.attrs["cutoff"] = out.attrs["valor_medio"]      # compat (média do alvo na folha)
    return out


# ----------------------------------------------------------------------
# Base com colunas de borda
# ----------------------------------------------------------------------
FAST = ["f64", "f64_inf", "f64_tie", "i64", "u8", "flag", "cat", "cat_lit", "cat_oot",
        "dec", "mix", "bol", "strd", "catd", "hcard", "boolean_na"]
SLOW = ["f32", "i64_big", "Int64_na", "Float64_na", "Int64_ok", "cat_nat"]


def _base(n=3000, seed=7, dup=False, alvo_nan=True):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    amostra = np.where(rng.random(n) < 0.6, "DES",
                       np.where(rng.random(n) < 0.6, "OOT", "ESTAB")).astype(object)
    amostra[rng.random(n) < 0.02] = None                       # linha sem amostra
    p = 1 / (1 + np.exp(-(s - 0.8)))
    y = (rng.random(n) < p).astype(float)
    if alvo_nan:
        y[(amostra == "OOT") & (rng.random(n) < 0.2)] = np.nan  # alvo NaN só no OOT
        y[(amostra == "DES") & (rng.random(n) < 0.03)] = np.nan
    f64 = s + rng.normal(scale=0.7, size=n)
    f64[rng.random(n) < 0.05] = np.nan
    f64[rng.random(n) < 0.02] = -0.0
    f64_inf = f64.copy()
    f64_inf[rng.random(n) < 0.03] = np.inf
    f64_inf[rng.random(n) < 0.03] = -np.inf
    f64_tie = np.round(s * 2) / 2
    i64 = np.round(s * 10).astype(np.int64)
    i64_big = i64 * (2 ** 54) + 1
    u8 = np.clip(np.round(s * 20 + 100), 0, 255).astype(np.uint8)
    cat = rng.choice(["a", "b", "c", "d", None], size=n, p=[.3, .25, .2, .15, .1]).astype(object)
    cat[s > 1.2] = "d"
    cat_lit = rng.choice(["x", "nan", "None", "y"], size=n).astype(object)
    cat_lit[rng.random(n) < 0.05] = None
    cat_lit[rng.random(n) < 0.05] = np.nan
    cat_oot = np.where(s > 0, "hi", "lo").astype(object)
    cat_oot[(amostra == "OOT") & (rng.random(n) < 0.1)] = "so_oot"  # só fora da DES
    dec = np.array([Decimal("1.0") if v < -0.5 else (Decimal("1.00") if v < 0.3
                    else Decimal("2.5")) for v in s], dtype=object)
    mix = np.array([1 if v < -0.4 else ("1" if v < 0.2 else (True if v < 0.9 else 2.0))
                    for v in s], dtype=object)
    cat_nat = np.where(s > 0.3, "NaT", "k").astype(object)
    cat_nat[rng.random(n) < 0.08] = pd.NaT                       # NA que vira 'NaT'
    df = pd.DataFrame({
        "f64": f64, "f64_inf": f64_inf, "f64_tie": f64_tie, "i64": i64, "u8": u8,
        "flag": (s > 0.5).astype(np.int64), "cat": cat, "cat_lit": cat_lit,
        "cat_oot": cat_oot, "dec": dec, "mix": mix,
        "bol": s > 0.1,
        "strd": pd.array(np.where(rng.random(n) < 0.08, None,
                                  np.where(s > 0, "p", "q")), dtype="string"),
        "catd": pd.Categorical(np.where(s > 0.6, "u", np.where(s > -0.6, "v", "w"))),
        "hcard": np.array([f"h{int(v * 40) % 300}" for v in s], dtype=object),
        "f32": f64.astype(np.float32),
        "i64_big": i64_big,
        "Int64_na": pd.array(np.where(rng.random(n) < 0.06, None, i64), dtype="Int64"),
        "Float64_na": pd.array(np.where(rng.random(n) < 0.06, None, f64_tie),
                               dtype="Float64"),
        "Int64_ok": pd.array(i64, dtype="Int64"),
        "boolean_na": pd.array(np.where(rng.random(n) < 0.06, None, s > 0),
                               dtype="boolean"),
        "cat_nat": cat_nat,
        "target": y, "amostra": amostra,
        "dt": pd.to_datetime("2023-01-01") + pd.to_timedelta(
            rng.integers(0, 365, size=n), unit="D"),
    })
    if dup:
        df.index = np.arange(n) % (n // 3)
    return df


FEATS = FAST + SLOW


def _seg(task="classification", dup=False, sample=True, n=3000):
    df = _base(n=n, dup=dup)
    if task == "regression":
        rng = np.random.default_rng(11)
        df["target"] = np.where(df["target"].isna(), np.nan,
                                df["f64_tie"].fillna(0) * 3 + rng.normal(size=len(df)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = TreeSegmenter(df, target="target", task_type=task,
                            sample_col="amostra" if sample else None,
                            date_col="dt", verbose=False)
        seg.grow(feature="f64_tie", splits=[-0.5, 0.5])
        folhas = [s for s, v in seg.segments.items() if v["is_leaf"]]
        seg.grow(feature="cat", only_segments=[folhas[1]])
    return seg


def _folhas(seg):
    return [s for s, v in seg.segments.items() if v["is_leaf"]]


def _internos(seg):
    return [s for s, v in seg.segments.items() if not v["is_leaf"] and s != "root"]


def _iguais(a, b):
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert set(a.attrs) == set(b.attrs)
    for k in a.attrs:
        va, vb = a.attrs[k], b.attrs[k]
        assert type(va) is type(vb) and (va == vb or (pd.isna(va) and pd.isna(vb)))


def _sem_ids(monkeypatch):
    monkeypatch.setattr(TreeSegmenter, "_bin_ids", staticmethod(lambda col, bins: None))


# ----------------------------------------------------------------------
# _bin_ids: (ids == k) idêntico ao _mask_in, e quem usa o caminho rápido
# ----------------------------------------------------------------------
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_bin_ids_igual_mask_in_por_faixa(task):
    seg = _seg(task)
    rapidos, lentos = set(), set()
    for sid in ["root", *_internos(seg), *_folhas(seg)]:
        for feat in FEATS:
            try:
                bins, _m, _k = seg._resolve_bins_cached(sid, feat, None, None, 5, 0.05)
            except Exception:
                continue
            if not bins:
                continue
            for frame in (seg._sub(seg.segments[sid]["mask"], feat),
                          seg._sub(seg._mask_ref, feat), seg.df):
                ids = seg._bin_ids(frame[feat], bins)
                if ids is None:
                    lentos.add(feat)
                    continue
                rapidos.add(feat)
                assert ids.dtype == np.int32 and len(ids) == len(frame)
                for k, b in enumerate(bins):
                    m = seg._mask_in(frame, feat, b).to_numpy()
                    assert m.dtype == bool
                    np.testing.assert_array_equal(ids == k, m)
                # nenhuma linha em duas faixas (senão o código único não vale)
                soma = sum(seg._mask_in(frame, feat, b).to_numpy().astype(int)
                           for b in bins)
                assert soma.max(initial=0) <= 1
    # f64_inf: o optbinning não corta com ±inf (sem faixas); ±inf está coberto
    # em test_bin_ids_num_bordas. boolean anulável é 'cat' (texto) → rápido.
    assert {"f64", "i64", "u8", "cat", "cat_lit", "dec", "mix", "bol",
            "strd", "catd", "hcard", "boolean_na"} <= rapidos
    for f in ("f32", "i64_big", "Int64_na", "Float64_na", "Int64_ok"):
        assert f not in rapidos and f in lentos


def test_bin_ids_na_que_casa_com_categoria_volta_ao_caminho_antigo():
    s = pd.Series(["NaT", pd.NaT, "k", None, "k"], dtype=object)
    bins = [{"kind": "cat", "cats": ["NaT"]}, {"kind": "cat", "cats": ["k"]},
            {"kind": "na"}]
    assert TreeSegmenter._bin_ids(s, bins) is None
    # sem a faixa 'na' a linha NaT cai só na categoria 'NaT' — código único vale
    ids = TreeSegmenter._bin_ids(s, bins[:2])
    np.testing.assert_array_equal(ids, [0, 0, 1, -1, 1])


@pytest.mark.parametrize("bins,esperado", [
    # cortes duplicados: a faixa (c, c] fica vazia, como no between
    ([{"kind": "num", "lo": -np.inf, "hi": 0.0}, {"kind": "num", "lo": 0.0, "hi": 0.0},
      {"kind": "num", "lo": 0.0, "hi": np.inf}, {"kind": "na"}], "ids"),
    # uma faixa só (-inf, inf]
    ([{"kind": "num", "lo": -np.inf, "hi": np.inf}], "ids"),
    # cortes inteiros (corte manual)
    ([{"kind": "num", "lo": -np.inf, "hi": 1}, {"kind": "num", "lo": 1, "hi": np.inf}],
     "ids"),
    # 1º lo não é -inf / faixas não contíguas / corte NaN / longdouble: caminho antigo
    ([{"kind": "num", "lo": -5.0, "hi": 0.0}, {"kind": "num", "lo": 0.0, "hi": np.inf}],
     None),
    ([{"kind": "num", "lo": -np.inf, "hi": 0.0}, {"kind": "num", "lo": 0.5, "hi": np.inf}],
     None),
    ([{"kind": "num", "lo": -np.inf, "hi": np.nan}, {"kind": "num", "lo": np.nan,
                                                     "hi": np.inf}], None),
    # longdouble: só vira caminho antigo onde é maior que float64 (no Windows
    # longdouble == float64 e o código de faixa vale)
    ([{"kind": "num", "lo": -np.inf, "hi": np.longdouble(0.1)},
      {"kind": "num", "lo": np.longdouble(0.1), "hi": np.inf}],
     None if np.dtype(np.longdouble).itemsize > 8 else "ids"),
    ([{"kind": "num", "lo": -np.inf, "hi": 2 ** 60},
      {"kind": "num", "lo": 2 ** 60, "hi": np.inf}], None),
    # mistura num/cat e duas faixas 'na': caminho antigo
    ([{"kind": "num", "lo": -np.inf, "hi": np.inf}, {"kind": "cat", "cats": ["1.0"]}],
     None),
    ([{"kind": "num", "lo": -np.inf, "hi": np.inf}, {"kind": "na"}, {"kind": "na"}],
     None),
])
def test_bin_ids_num_bordas(bins, esperado):
    x = pd.Series([-np.inf, -1.0, -0.0, 0.0, 0.1, 1.0, 1.5, np.inf, np.nan, 2.0])
    ids = TreeSegmenter._bin_ids(x, bins)
    if esperado is None:
        assert ids is None
        return
    for k, b in enumerate(bins):
        np.testing.assert_array_equal(
            ids == k, TreeSegmenter._mask_in_col(x, b).to_numpy())


def test_bin_ids_int_e_cat_bordas():
    bins = [{"kind": "num", "lo": -np.inf, "hi": 0.5}, {"kind": "num", "lo": 0.5,
                                                       "hi": np.inf}]
    xi = pd.Series(np.array([-3, 0, 1, 2 ** 53], dtype=np.int64))
    ids = TreeSegmenter._bin_ids(xi, bins)
    np.testing.assert_array_equal(ids, [0, 0, 1, 1])
    assert TreeSegmenter._bin_ids(pd.Series(np.array([2 ** 53 + 1]), dtype=np.int64),
                                  bins) is None
    assert TreeSegmenter._bin_ids(pd.Series(np.array([], dtype=np.int64)),
                                  bins).size == 0
    # categorias: não-str (np.str_), faixas sobrepostas → caminho antigo
    s = pd.Series(["a", "b", None])
    assert TreeSegmenter._bin_ids(s, [{"kind": "cat", "cats": [np.str_("a")]}]) is None
    assert TreeSegmenter._bin_ids(s, [{"kind": "cat", "cats": ["a"]},
                                      {"kind": "cat", "cats": ["a", "b"]}]) is None
    # Decimal('1.0') e Decimal('1.00') têm texto diferente: nunca se fundem
    d = pd.Series([Decimal("1.0"), Decimal("1.00"), Decimal("1.0")], dtype=object)
    ids = TreeSegmenter._bin_ids(d, [{"kind": "cat", "cats": ["1.0"]},
                                     {"kind": "cat", "cats": ["1.00"]}])
    np.testing.assert_array_equal(ids, [0, 1, 0])
    # 1, '1', True, 1.0 → '1', '1', 'True', '1.0'
    m = pd.Series([1, "1", True, 1.0], dtype=object)
    ids = TreeSegmenter._bin_ids(m, [{"kind": "cat", "cats": ["1"]},
                                     {"kind": "cat", "cats": ["True", "1.0"]}])
    np.testing.assert_array_equal(ids, [0, 0, 1, 1])
    assert TreeSegmenter._bin_ids(pd.DataFrame({"a": [1]}), bins) is None
    assert TreeSegmenter._bin_ids(s, []) is None


# ----------------------------------------------------------------------
# variable_iv: velho × novo
# ----------------------------------------------------------------------
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("dup", [False, True])
def test_variable_iv_igual_ao_antigo(task, dup):
    seg = _seg(task, dup=dup)
    sids = [None, "root", "nao_existe", *_internos(seg), *_folhas(seg)]
    feats_extra = [*FEATS, "nao_existe", "target", "f64"]
    for sid in sids:
        for wp in (True, False):
            novo = seg.variable_iv(sid, with_psi=wp)
            velho = _variable_iv_velho(seg, sid, with_psi=wp)
            _iguais(novo, velho)
            novo = seg.variable_iv(sid, features=feats_extra, with_psi=wp,
                                   max_n_bins=3, min_bin_size=0.1)
            velho = _variable_iv_velho(seg, sid, features=feats_extra, with_psi=wp,
                                       max_n_bins=3, min_bin_size=0.1)
            _iguais(novo, velho)
    # o IV das anuláveis com NA continua NaN (máscara com NA no caminho antigo)
    iv = seg.variable_iv("root", features=["Int64_na", "Float64_na"])
    assert iv["iv"].isna().all()


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_variable_iv_bit_exato_sem_arredondamento(task, monkeypatch):
    """Com o ``round`` desligado nas duas versões, o IV/PSI acumulado tem de ser
    igual bit a bit (pega, p.ex., média por bincount ponderado na regressão —
    soma sequencial ≠ pairwise — que o arredondamento a 4 casas esconderia)."""
    def sem_round(x, n=None):
        return x
    monkeypatch.setattr(segmod, "round", sem_round, raising=False)
    monkeypatch.setitem(globals(), "round", sem_round)
    seg = _seg(task)
    for sid in ["root", *_internos(seg), *_folhas(seg)]:
        for wp in (True, False):
            novo = seg.variable_iv(sid, with_psi=wp)
            velho = _variable_iv_velho(seg, sid, with_psi=wp)
            assert novo["iv"].notna().any()
            _iguais(novo, velho)


def test_variable_iv_sem_sample_col():
    for task in ("classification", "regression"):
        seg = _seg(task, sample=False, n=1500)
        for sid in [None, *_folhas(seg)]:
            for wp in (True, False):
                _iguais(seg.variable_iv(sid, with_psi=wp),
                        _variable_iv_velho(seg, sid, with_psi=wp))


def test_variable_iv_alvo_clf_nao_binario_e_folha_sem_oot():
    seg = _seg("classification", n=2000)
    # alvo truncado (astype(int)) com valores 0.7 e 2.0: maus pelo somatório exato
    alvo = seg.df["target"].to_numpy().copy()
    idx = np.flatnonzero(~np.isnan(alvo))
    alvo[idx[::17]] = 2.0
    alvo[idx[5::23]] = 0.7
    seg.df["target"] = alvo
    for sid in [None, *_folhas(seg)]:
        _iguais(seg.variable_iv(sid), _variable_iv_velho(seg, sid))
    # folha sem nenhuma linha OOT
    seg2 = _seg("classification", n=2000)
    seg2.df.loc[seg2.df["amostra"].eq("OOT").to_numpy() & (seg2.df["f64_tie"] > 0.5)
                .to_numpy(), "amostra"] = "ESTAB"
    seg2 = TreeSegmenter(seg2.df, target="target", sample_col="amostra", verbose=False)
    seg2.grow(feature="f64_tie", splits=[0.5])
    for sid in [None, *_folhas(seg2)]:
        _iguais(seg2.variable_iv(sid), _variable_iv_velho(seg2, sid))


def test_variable_iv_amostra_string_com_na():
    df = _base(n=2000)
    df["amostra"] = pd.array(df["amostra"].to_numpy(), dtype="string")
    seg = TreeSegmenter(df, target="target", sample_col="amostra", verbose=False)
    # na_to_worst=False: o _assign_na_to_worst já falha com amostra anulável
    # (máscara boolean com NA como índice numpy) — fora do escopo desta onda
    seg.grow(feature="f64_tie", splits=[0.0], na_to_worst=False)
    assert not all(m.dtype == bool for m in seg._sample_masks.values())
    for sid in [None, *_folhas(seg)]:
        for wp in (True, False):
            _iguais(seg.variable_iv(sid, with_psi=wp),
                    _variable_iv_velho(seg, sid, with_psi=wp))


def test_csi_amostra_string_com_na_igual_ao_antigo(monkeypatch):
    df = _base(n=2000)
    df["amostra"] = pd.array(df["amostra"].to_numpy(), dtype="string")
    seg = TreeSegmenter(df, target="target", sample_col="amostra", verbose=False)
    feats = ["f64", "cat", "dec", "i64"]
    novo = seg.csi(features=feats), seg.csi_detalhe()
    with monkeypatch.context() as mp:
        _sem_ids(mp)
        velho = seg.csi(features=feats), seg.csi_detalhe()
    for a, b in zip(novo, velho):
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_variable_iv_caminho_rapido_nao_usa_mask_in(monkeypatch):
    seg = _seg("classification", n=2000)
    for f in FAST:            # aquece o cache de bins (o _resolve_bins usa máscaras)
        seg._resolve_bins_cached("root", f, None, None, 5, 0.05)
    chamadas = []
    orig = TreeSegmenter._mask_in_col

    def espia(col, b):
        chamadas.append(col.name)
        return orig(col, b)
    monkeypatch.setattr(TreeSegmenter, "_mask_in_col", staticmethod(espia))
    seg.variable_iv("root", features=[f for f in FAST if f != "cat_nat"])
    assert chamadas == []
    seg.variable_iv("root", features=["f32", "cat_nat"])
    assert set(chamadas) == {"f32", "cat_nat"}


def test_variable_iv_pico_de_memoria_menor():
    rng = np.random.default_rng(5)
    n = 40000
    df = pd.DataFrame({f"v{i:02d}": rng.normal(size=n) for i in range(40)})
    df["target"] = (rng.random(n) < 1 / (1 + np.exp(-df["v00"]))).astype(float)
    df["amostra"] = np.where(rng.random(n) < 0.7, "DES", "OOT")
    seg = TreeSegmenter(df, target="target", sample_col="amostra", verbose=False)
    feats = [f"v{i:02d}" for i in range(40)]
    seg.variable_iv("root", features=feats)          # aquece o cache de bins

    def pico(fn):
        tracemalloc.start()
        fn()
        _, p = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return p
    p_velho = pico(lambda: _variable_iv_velho(seg, "root", features=feats))
    p_novo = pico(lambda: seg.variable_iv("root", features=feats))
    assert p_novo < 0.35 * p_velho, (p_novo, p_velho)


# ----------------------------------------------------------------------
# Laços de análise: código de faixa × caminho antigo forçado
# ----------------------------------------------------------------------
def _saidas_lacos(seg):
    out = {}
    sids = ["root", *_folhas(seg)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in FEATS:
            for sid in sids:
                for nome, fn in [
                    ("table", lambda: seg.variable_table(f, sid=sid)),
                    ("table_oot", lambda: seg.variable_table(f, sid=sid, sample="OOT")),
                    ("inv", lambda: seg.variable_inversion(f, sid=sid, time_col="dt")),
                    ("psi_safra", lambda: seg.variable_psi_by_safra(f, time_col="dt",
                                                                    sid=sid)),
                ]:
                    try:
                        out[(nome, f, sid)] = fn()
                    except Exception as e:      # erro também é saída
                        out[(nome, f, sid)] = ("__erro__", type(e).__name__, str(e))
        # corte manual: as faixas (-inf, c] passam pelo -inf/+inf/NaN da coluna
        for f in ("f64_inf", "f64", "i64", "i64_big", "f32", "Int64_na"):
            try:
                out[("table_man", f)] = seg.variable_table(f, splits=[-0.5, 0.0, 0.5])
            except Exception as e:
                out[("table_man", f)] = ("__erro__", type(e).__name__, str(e))
        try:
            out["csi"] = seg.csi(features=FEATS)
            out["csi_det"] = seg.csi_detalhe()
        except Exception as e:
            out["csi"] = ("__erro__", type(e).__name__, str(e))
        for sid in sids:
            out[("sug", sid)] = seg.suggest_splits(sid, top=6)
        tbl = seg._bin_table(seg.df, "f64", seg._resolve_bins_cached(
            "root", "f64", None, None, 5, 0.05)[0], len(seg.df))
        out["bin_table"] = (tbl, dict(tbl.attrs))
    return out


def _compara(a, b):
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        assert repr(a.attrs) == repr(b.attrs)
    elif isinstance(a, dict):
        assert list(a) == list(b)
        for k in a:
            _compara(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            _compara(x, y)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b)
    else:
        assert type(a) is type(b) and a == b, (a, b)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_lacos_iguais_ao_caminho_antigo(task, monkeypatch):
    seg = _seg(task, n=2000)
    novo = _saidas_lacos(seg)
    with monkeypatch.context() as mp:
        _sem_ids(mp)
        velho = _saidas_lacos(seg)
    assert list(novo) == list(velho)
    for k in novo:
        try:
            _compara(novo[k], velho[k])
        except AssertionError as e:
            raise AssertionError(f"{k}: {e}") from None


def test_lacos_usam_codigo_de_faixa(monkeypatch):
    seg = _seg("classification", n=2000)
    _saidas_lacos(seg)                      # aquece caches de bins
    chamadas = []
    orig = TreeSegmenter._mask_in_col

    def espia(col, b):
        chamadas.append(col.name)
        return orig(col, b)
    monkeypatch.setattr(TreeSegmenter, "_mask_in_col", staticmethod(espia))
    seg.variable_table("cat")
    seg.variable_psi_by_safra("hcard", time_col="dt")
    seg.variable_inversion("f64", time_col="dt")
    seg.csi(features=["f64", "cat", "dec"])
    assert chamadas == []


def test_grafico_share_por_safra_igual(monkeypatch):
    import matplotlib.pyplot as plt
    seg = _seg("classification", n=2000)

    def dados():
        fig = seg.plot_variable_optbin_cumshare_timeseries("f64", time_col="dt")
        d = []
        for ax in fig.axes:
            d += [np.asarray(l.get_ydata(), dtype=float) for l in ax.lines]
            for c in ax.collections:
                d += [np.asarray(p.vertices, dtype=float) for p in c.get_paths()]
        plt.close(fig)
        return d
    novo = dados()
    with monkeypatch.context() as mp:
        _sem_ids(mp)
        velho = dados()
    assert len(novo) == len(velho) and len(novo) > 0
    for a, b in zip(novo, velho):
        np.testing.assert_array_equal(a, b)
