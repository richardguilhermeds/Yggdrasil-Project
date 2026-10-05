"""Onda 2C — ``variable_table`` e ``_variable_psi`` por códigos de faixa +
``np.bincount`` (caminho rápido só com ``_particao_exata``; PSI numa passada).

Cada caso compara contra a versão ANTIGA copiada aqui: tabela com
``assert_frame_equal(check_exact=True)`` + ``attrs`` por ``==`` e PSI por
igualdade exata. Os casos de borda (``-inf``, grupos sobrepostos, pyarrow com
NaN e null, ``sample_col=None``) precisam cair no caminho das máscaras."""
from __future__ import annotations

import contextlib
import decimal
import io
import math

import numpy as np
import pandas as pd
import pytest

pyarrow = pytest.importorskip("pyarrow")

from yggdrasil.credit_risk.model import ModelSegmenter
from yggdrasil.credit_risk.model import segmenter as segmod

D = decimal.Decimal


# ───────────────────────── versão antiga (referência) ─────────────────────────
def _old_variable_table(self, feature, sample=None, max_n_bins=6, min_bin_size=0.05,
                        splits=None):
    _EPS = segmod._EPS
    bins, kind = self._resolve_bins(feature, max_n_bins, min_bin_size, splits, sample)
    sub = self._frame(sample, cols=[feature, self.target])
    n_tot = max(len(sub), 1)
    risco_label = "event_rate" if self.task_type == "classification" else "alvo_medio"
    if not bins:
        out = pd.DataFrame(columns=["faixa", "n", "repr_%", risco_label])
        out.attrs.update(iv=float("nan"), mono_ok=True, kind=kind,
                         risco_label=risco_label)
        return out
    y_all = sub[self.target].to_numpy(dtype="float64")
    mean_global = self._risco(y_all)
    n_evt_tot = float(np.nansum(y_all == 1)) if self.task_type == "classification" else 0.0
    n_non_tot = float(np.nansum(y_all == 0)) if self.task_type == "classification" else 0.0
    n_base = float(np.sum(~np.isnan(y_all)))
    rows, iv_total, is_na = [], 0.0, []
    for b, m in zip(bins, segmod._bin_masks(sub[feature], bins)):
        yi = y_all[m]
        yi_ok = yi[~np.isnan(yi)]
        n_i = int(m.sum())
        if n_i == 0:
            continue
        is_na.append(b.get("kind") == "na")
        risco = self._risco(yi)
        row = {"faixa": self._bin_label(feature, b), "n": n_i,
               "repr_%": round(100 * n_i / n_tot, 1),
               risco_label: round(risco, 4) if np.isfinite(risco) else np.nan}
        if self.task_type == "classification":
            n_evt = float((yi_ok == 1).sum())
            n_non = float((yi_ok == 0).sum())
            d_evt = n_evt / max(n_evt_tot, _EPS)
            d_non = n_non / max(n_non_tot, _EPS)
            woe = float(np.log((d_non + _EPS) / (d_evt + _EPS)))
            logodds = (float(np.log((risco + _EPS) / (1 - risco + _EPS)))
                       if np.isfinite(risco) else np.nan)
            ivp = (d_non - d_evt) * woe
            row.update(woe=round(woe, 4),
                       logodds=round(logodds, 4) if np.isfinite(logodds) else np.nan,
                       iv_parcial=round(ivp, 4))
            iv_total += ivp
        else:
            ivp = ((yi_ok.size / max(n_base, _EPS)) *
                   abs(risco - mean_global)) if np.isfinite(risco) else 0.0
            row["iv_parcial"] = round(ivp, 4)
            iv_total += ivp
        rows.append(row)
    out = pd.DataFrame(rows)
    if risco_label in out and len(out):
        rcol = out.loc[~np.asarray(is_na), risco_label]
    else:
        rcol = pd.Series(dtype=float)
    mono = bool(rcol.is_monotonic_increasing or rcol.is_monotonic_decreasing) \
        if len(rcol) else True
    out.attrs.update(iv=round(float(iv_total), 4), mono_ok=mono, kind=kind,
                     risco_label=risco_label, mean_global=round(mean_global, 4),
                     risco_ordenavel=[float(x) for x in rcol])
    return out


def _old_variable_psi(self, feature, samples, max_n_bins=5, min_bin_size=0.05,
                      eps=1e-6, _codigos=None):
    bins, _kind = self._resolve_bins(feature, max_n_bins, min_bin_size)
    out = {a: np.nan for a in samples}
    if not bins:
        return out
    nb = len(bins)
    bc = segmod._bin_codes(self.df[feature], bins)
    ref_m = self._rows_mask(self.ref_sample, amostrar=False)
    n_ref = max(int(ref_m.sum()), 1)
    ref_cont = np.bincount(bc[ref_m & (bc >= 0)], minlength=nb)
    ref_pct = [max(int(c) / n_ref, eps) for c in ref_cont]
    for a in samples:
        m = self._rows_mask(a, amostrar=False)
        n_cur = int(m.sum())
        if n_cur == 0:
            continue
        cont = np.bincount(bc[m & (bc >= 0)], minlength=nb)
        cur_pct = [int(c) / n_cur for c in cont]
        out[a] = round(segmod._psi_from_shares(ref_pct, cur_pct, eps), 4)
    return out


# ───────────────────────── comparação exata ─────────────────────────
def _igual(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_igual(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _attrs_iguais(a, b):
    assert list(a) == list(b)
    for k in a:
        assert _igual(a[k], b[k]), (k, a[k], b[k])


def _confere_tabela(seg, feat, sample=None, rapido=None, **kw):
    """``rapido``: o caminho por códigos do ranking (``para_psi=True``) rodou?"""
    novo, cod = seg._variable_table(feat, sample=sample, para_psi=True, **kw)
    velho = _old_variable_table(seg, feat, sample=sample, **kw)
    pd.testing.assert_frame_equal(novo, velho, check_exact=True)
    _attrs_iguais(novo.attrs, velho.attrs)
    pd.testing.assert_frame_equal(seg.variable_table(feat, sample=sample, **kw), velho,
                                  check_exact=True)
    if rapido is not None:
        assert (cod is not None) is rapido, (feat, sample)
    return cod


def _confere_psi(seg, feat, samples, codigos=None, **kw):
    novo = seg._variable_psi(feat, samples, _codigos=codigos, **kw)
    velho = _old_variable_psi(seg, feat, samples, **kw)
    assert list(novo) == list(velho)
    for a in novo:
        assert _igual(novo[a], velho[a]), (feat, a, novo[a], velho[a])


# ───────────────────────── base sintética ─────────────────────────
def _df(n=12_000, seed=0, task="classification"):
    rng = np.random.default_rng(seed)
    x = rng.gamma(2, 1500, n)
    logit = -1.2 - 0.0004 * (x - 3000)
    if task == "classification":
        y = rng.binomial(1, 1 / (1 + np.exp(-logit))).astype(float)
    else:
        y = np.clip(rng.beta(2, 5, n) + 0.00003 * x, 0, None) * 0.37
    y[rng.random(n) < .02] = np.nan
    x_nan = x.copy()
    x_nan[rng.random(n) < .1] = np.nan
    x_neg = rng.normal(size=n)
    x_neg[rng.random(n) < .01] = -np.inf
    x_neg[rng.random(n) < .01] = np.inf
    i64 = pd.array(rng.integers(0, 40, n), dtype="Int64")
    i64[rng.random(n) < .08] = pd.NA
    f64 = pd.array(rng.normal(size=n), dtype="Float64")
    f64[rng.random(n) < .08] = pd.NA
    # Float64 com NaN NÃO mascarado (0/0): isna ≠ isnan → fallback
    f64n = pd.Series(pd.array(rng.normal(size=n), dtype="Float64"))
    zero = pd.Series(pd.array(np.where(rng.random(n) < .05, 0.0, 1.0), dtype="Float64"))
    f64n = (f64n * zero) / zero
    # float64[pyarrow] com null E NaN distintos (isna marca só o null) → fallback
    v = rng.normal(size=n)
    v[rng.random(n) < .05] = np.nan
    pa = pd.Series(pd.arrays.ArrowExtensionArray(
        pyarrow.array(v, type=pyarrow.float64(), mask=rng.random(n) < .05)))
    cat = rng.choice([f"c{i:04d}" for i in range(2000)], n).astype(object)
    cat[rng.random(n) < .05] = None
    cat[rng.random(n) < .01] = np.nan
    gar = rng.choice(list("ABCDE"), n).astype(object)
    gar[rng.random(n) < .05] = None
    dec = np.array([D(str(round(v, 1))) for v in rng.choice([0.5, 1.0, 1.5, 2.0], n)],
                   dtype=object)
    dec[rng.random(n) < .03] = D("NaN")
    dec[rng.random(n) < .03] = None
    df = pd.DataFrame({
        "x_nan": x_nan, "x_neg": x_neg, "x_i64": i64, "x_f64": f64,
        "x_f64n": f64n.array, "x_pa": pa.array, "x_cat": cat, "x_gar": gar,
        "x_dec": dec, "x_bool": rng.random(n) < .3, "target": y})
    df["amostra"] = rng.choice(["DES", "OOT", "OOT2"], n, p=[.6, .25, .15])
    return df


def _seg(task="classification", sample_col="amostra", **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return ModelSegmenter(_df(task=task, **kw), target="target", task_type=task,
                              sample_col=sample_col)


FEATS = ["x_nan", "x_neg", "x_i64", "x_f64", "x_f64n", "x_pa", "x_cat", "x_gar",
         "x_dec", "x_bool"]


@pytest.fixture(scope="module", params=["classification", "regression"])
def seg(request):
    return _seg(request.param)


# ───────────────────────── tabela: antigo × novo ─────────────────────────
@pytest.mark.parametrize("feat", FEATS)
@pytest.mark.parametrize("sample", [None, "OOT"])
def test_tabela_igual_a_antiga(seg, feat, sample):
    _confere_tabela(seg, feat, sample=sample)
    _confere_tabela(seg, feat, sample=sample, max_n_bins=5)


def test_caminho_rapido_numerica_e_categorica(seg):
    for feat in ("x_nan", "x_i64", "x_f64", "x_gar", "x_cat"):
        cod = _confere_tabela(seg, feat, rapido=True)
        assert cod[1].dtype == np.int8          # códigos compactos


@pytest.mark.parametrize("feat", ["x_neg", "x_f64n", "x_pa"])
def test_fallback_menos_inf_e_nan_nao_mascarado(seg, feat):
    _confere_tabela(seg, feat, rapido=False)


def test_fallback_grupos_sobrepostos():
    s = _seg()
    s.set_manual_bins("x_gar", "A, B; B, C; D")
    _confere_tabela(s, "x_gar", rapido=False)
    _confere_psi(s, "x_gar", ["OOT", "OOT2"])


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_fallback_sem_sample_col(task):
    s = _seg(task, sample_col=None)
    for feat in ("x_nan", "x_gar"):
        _confere_tabela(s, feat, rapido=False)
        _confere_psi(s, feat, ["DES"])
    assert s._codigos_amostras_psi(["DES"]) is None


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("destino", ["separado", "pior", "melhor", 1])
def test_manuais_com_destino_do_faltante(task, destino):
    s = _seg(task)
    s.set_manual_bins("x_nan", "1500, 3000, 5000", missing=destino)
    s.set_manual_bins("x_gar", "A; B, C; D, E", missing=destino)
    for feat in ("x_nan", "x_gar"):
        for sample in (None, "OOT"):
            _confere_tabela(s, feat, sample=sample, rapido=True)
        _confere_psi(s, feat, ["OOT", "OOT2"])


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_derivada_com_faltante(task):
    s = _seg(task)
    s.set_manual_bins("x_nan", [1500, 3000, 5000])
    nome = s.create_categorical("x_nan", new_name="x_nan_cat")
    _confere_tabela(s, nome)
    _confere_tabela(s, nome, sample="OOT2")
    _confere_psi(s, nome, ["OOT", "OOT2"])
    # grupos manuais sobre as faixas da derivada + "(faltante)" no pior grupo
    rot = [r for r in s.df[nome].dropna().astype(str).unique() if r != "(faltante)"]
    rot.sort()
    s.set_manual_bins(nome, [rot[:2], rot[2:]])
    _confere_tabela(s, nome, rapido=True)
    _confere_psi(s, nome, ["OOT", "OOT2"])
    s.set_missing_bin(nome, "pior")
    _confere_tabela(s, nome)
    _confere_psi(s, nome, ["OOT", "OOT2"])


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_muitas_faixas_tres_amostras_sem_overflow_int8(task):
    """nb = 51 faixas × 3 amostras: (grupo × nb + faixa) passa de 127 — o índice
    combinado do PSI tem de ser montado em int64."""
    s = _seg(task)
    cortes = list(np.quantile(s.df["x_nan"].dropna(), np.linspace(0.02, 0.98, 50)))
    s.set_manual_bins("x_nan", cortes)
    cod = _confere_tabela(s, "x_nan", rapido=True)
    bins, _k = s._resolve_bins("x_nan", 5, 0.05)
    assert len(bins) > 43 and cod[1].dtype == np.int8
    _confere_psi(s, "x_nan", ["OOT", "OOT2"])
    _confere_psi(s, "x_nan", ["OOT", "OOT2"], codigos=cod)


def test_categorica_30k_niveis():
    rng = np.random.default_rng(7)
    n = 90_000
    df = pd.DataFrame({
        "cep": rng.choice([f"{i:05d}" for i in range(30_000)], n).astype(object),
        "target": rng.binomial(1, .2, n).astype(float),
        "amostra": rng.choice(["DES", "OOT", "OOT2"], n, p=[.6, .25, .15])})
    df.loc[rng.random(n) < .03, "cep"] = None
    with contextlib.redirect_stdout(io.StringIO()):
        s = ModelSegmenter(df, target="target", sample_col="amostra")
    _confere_tabela(s, "cep")
    _confere_psi(s, "cep", ["OOT", "OOT2"])


def test_alvo_classificacao_nao_inteiro_usa_media_por_segmento():
    """Classificação com alvo fora de {0,1} não inteiro: soma/n do bincount
    poderia diferir em ULP — vai pela ordenação estável + np.mean."""
    s = _seg()
    for f in ("x_nan", "x_gar"):           # faixas fixadas com o alvo 0/1
        s._resolve_bins(f, 6, 0.05, None, None)
    s.df["target"] = s.df["target"] * 0.1 + np.where(s.df["target"] > 0, 0.07, 0.0)
    _confere_tabela(s, "x_nan", rapido=True)
    _confere_tabela(s, "x_gar", rapido=True)


def test_alvo_int64_com_na_fora_da_referencia():
    """Alvo Int64 com <NA> só fora da referência: converter a coluna inteira e
    recortar dá o mesmo que converter o recorte (se a conversão falhasse, o
    caminho antigo decidiria)."""
    s = _seg()
    y = pd.array(s.df["target"].fillna(0).astype(int), dtype="Int64")
    y[np.flatnonzero(s.df["amostra"].to_numpy() == "OOT")[:5]] = pd.NA
    s.df["target"] = y
    _confere_tabela(s, "x_nan")
    _confere_tabela(s, "x_gar", sample="OOT")


# ───────────────────────── PSI ─────────────────────────
@pytest.mark.parametrize("feat", FEATS)
def test_psi_igual_ao_antigo(seg, feat):
    for samples in (["OOT", "OOT2"], ["OOT2"], ["DES", "OOT"], ["OOT", "OOT", "OOT2"],
                    ["XX"], []):
        _confere_psi(seg, feat, samples)
        _confere_psi(seg, feat, samples, max_n_bins=6)


def test_psi_reusa_codigos_da_tabela_so_com_as_mesmas_faixas(seg, monkeypatch):
    vt, cod = seg._variable_table("x_nan", max_n_bins=5, para_psi=True)
    assert cod is not None
    chamadas = []
    orig = segmod._bin_codes
    monkeypatch.setattr(segmod, "_bin_codes", lambda *a, **k: chamadas.append(1) or orig(*a, **k))
    _confere_psi(seg, "x_nan", ["OOT", "OOT2"], codigos=cod)
    n_ref = len(chamadas)              # 1 do antigo
    # faixas diferentes (max_n_bins=6 ≠ 5): os códigos repassados são ignorados
    _confere_psi(seg, "x_nan", ["OOT", "OOT2"], codigos=cod, max_n_bins=6)
    assert n_ref == 1 and len(chamadas) == 3


def test_amostras_sobrepostas_caem_no_laco(monkeypatch):
    s = _seg()
    orig = ModelSegmenter._frame_mask

    def fm(self, sample=None):          # "OOT2" vira cópia da referência
        return orig(self, "DES" if sample == "OOT2" else sample)
    monkeypatch.setattr(ModelSegmenter, "_frame_mask", fm)
    assert s._codigos_amostras_psi(["OOT", "OOT2"]) is None
    _confere_psi(s, "x_nan", ["OOT", "OOT2"])


# ───────────────────────── ranking e memória ─────────────────────────
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_ranking_igual_ao_antigo_e_sem_encher_bincode_cache(task, monkeypatch):
    novo_seg = _seg(task)
    novo_seg.set_manual_bins("x_gar", "A, B; B, C; D")         # um fallback no meio
    cont = []
    orig = ModelSegmenter._contagens_por_codigo

    def espia(self, *a, **k):
        r = orig(self, *a, **k)
        cont.append(r is not None)
        return r
    monkeypatch.setattr(ModelSegmenter, "_contagens_por_codigo", espia)
    novo = novo_seg.variable_iv(features=FEATS)
    monkeypatch.setattr(ModelSegmenter, "_contagens_por_codigo", orig)
    assert sum(cont) >= 5                      # o ranking passou pelos códigos
    assert novo_seg._bincode_cache == {}
    velho_seg = _seg(task)
    velho_seg.set_manual_bins("x_gar", "A, B; B, C; D")
    monkeypatch.setattr(ModelSegmenter, "_variable_table",
                        lambda self, *a, para_psi=False, **k:
                        (_old_variable_table(self, *a, **k), None))
    monkeypatch.setattr(ModelSegmenter, "_variable_psi", _old_variable_psi)
    velho = velho_seg.variable_iv(features=FEATS)
    assert velho["iv"].notna().sum() >= 5      # a referência rodou de fato
    pd.testing.assert_frame_equal(novo, velho, check_exact=True)


def test_bincode_cache_existente_e_reusado_sem_gravar():
    s = _seg()
    bins, _k = s._resolve_bins("x_nan", 6, 0.05)
    bc = s._feature_bin_codes("x_nan", bins)          # a aba Análise montou
    antes = dict(s._bincode_cache)
    assert s._bin_codes_sem_cache("x_nan", bins) is bc
    _confere_tabela(s, "x_nan", rapido=True)
    s._variable_table("x_gar", para_psi=True)
    assert s._bincode_cache.keys() == antes.keys()


def test_tabela_avulsa_so_pelos_codigos_com_cache():
    """Sem PSI em seguida (tabela avulsa), binar a base inteira só para a tabela
    da referência custaria mais que o recorte: por códigos só com o vetor já no
    ``_bincode_cache`` — e o resultado é o mesmo nos dois caminhos."""
    s = _seg()
    for feat in ("x_nan", "x_gar"):
        vt, cod = s._variable_table(feat)
        assert cod is None
        bins, _k = s._resolve_bins(feat, 6, 0.05, None, None)
        s._feature_bin_codes(feat, bins)
        vt2, cod2 = s._variable_table(feat)
        assert cod2 is not None
        velho = _old_variable_table(s, feat)
        for v in (vt, vt2, s.variable_table(feat)):
            pd.testing.assert_frame_equal(v, velho, check_exact=True)
            _attrs_iguais(v.attrs, velho.attrs)


def test_cache_de_amostras_do_psi_no_registro_e_zerado_no_detached():
    s = _seg()
    s._variable_psi("x_nan", ["OOT", "OOT2"])
    assert ("_psi_amostras_cache", None) in ModelSegmenter._ROW_CACHES
    assert s._psi_amostras_cache is not None
    assert s._detached_scorer()._psi_amostras_cache is None


@pytest.mark.parametrize("valores,spec", [
    ([True, 1, 1.0, "a", "b"], "1; True; a, b"),
    ([D("1.00"), D("1.0"), D("2.0"), D("2.00")], "1.0; 1.00; 2.0, 2.00"),
])
def test_representante_do_factorize_cai_no_fallback(valores, spec, monkeypatch):
    """Objeto misto / Decimal de escala mista: valores iguais no hash viram UM
    nível com o texto do 1º que aparece — na base inteira (1ª linha, OOT) é um,
    no recorte da DES é outro. Os códigos da base inteira contariam diferente
    das máscaras do recorte: tem de cair no caminho antigo."""
    rng = np.random.default_rng(3)
    n = 6000
    amostra = rng.choice(["DES", "OOT"], n, p=[.7, .3])
    amostra[0], amostra[1] = "OOT", "DES"
    obj = np.array([valores[k] for k in rng.integers(0, len(valores), n)], dtype=object)
    obj[0], obj[1] = valores[0], valores[1]     # representantes diferentes
    df = pd.DataFrame({"obj": obj, "target": rng.binomial(1, .3, n).astype(float),
                       "amostra": amostra})
    with contextlib.redirect_stdout(io.StringIO()):
        s = ModelSegmenter(df, target="target", sample_col="amostra")
    s.set_manual_bins("obj", spec)
    _confere_tabela(s, "obj", rapido=False)
    _confere_psi(s, "obj", ["OOT"])
    # a guarda é o que segura: sem ela, a contagem por código diverge
    monkeypatch.setattr(segmod, "_texto_por_valor_estavel", lambda s_: True)
    novo, cod = s._variable_table("obj", para_psi=True)
    assert cod is not None
    assert not novo.equals(_old_variable_table(s, "obj"))


def test_texto_por_valor_estavel():
    f = segmod._texto_por_valor_estavel
    assert f(pd.Series(["a", None, "b"], dtype=object))
    assert f(pd.Series(["a", None], dtype="string"))
    assert f(pd.Series(["a", "b"]).astype("category"))
    assert f(pd.Series([1, 2], dtype="int64")) and f(pd.Series([True, False]))
    assert not f(pd.Series([1, 1.0, True], dtype=object))
    assert not f(pd.Series([D("1.0"), D("1.00")], dtype=object))
    assert not f(pd.Series([0.0, -0.0]))


def test_regressao_desligavel(monkeypatch):
    s = _seg("regression")
    _confere_tabela(s, "x_nan", rapido=True)
    monkeypatch.setattr(ModelSegmenter, "_TABELA_CODIGOS_REGRESSAO", False)
    _confere_tabela(s, "x_nan", rapido=False)


def test_falha_ao_binar_a_base_inteira_cai_no_caminho_antigo(monkeypatch):
    """Se binar a coluna INTEIRA falhar (a tabela antiga só lia o recorte), o
    caminho por códigos desiste e a tabela sai pelas máscaras, igual à antiga."""
    s = _seg()

    def quebra(*a, **k):
        raise RuntimeError("falha só na base inteira")
    monkeypatch.setattr(segmod, "_bin_codes", quebra)
    for feat in ("x_nan", "x_gar"):
        _confere_tabela(s, feat, rapido=False)
