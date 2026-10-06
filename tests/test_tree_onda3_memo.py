"""Onda 3 (item 3.3) da árvore: memo de agregados por ASSINATURA ESTRUTURAL.

O cache por versão (``_agg_cache``) morria a cada undo/redo, mesmo voltando a
um estado já visto com os MESMOS objetos-máscara. Agora, no miss da versão,
leaves/_leaf_order/_grade_map/metrics/psi/feature_importance consultam uma LRU
chaveada pela assinatura (sid, id da máscara, is_leaf, pai, rótulo,
profundidade, caminho, condições), com referência forte às máscaras e
validação por ``is``. Na UI, o mesmo para ``_ordered_leaf_options``.

Oráculo: a MESMA sequência de ações numa instância com a LRU desligada
(``_AGG_LRU_SIGS = 0`` → só o cache por versão, o comportamento antigo), com
igualdade exata — e contagem de recomputes (0 no undo para o estado
imediatamente anterior). ``from_dict`` NÃO serve de oráculo: ele recomputa as
máscaras pelas condições, e com ``include_na`` (NaN fora do ajuste) a máscara
recomputada difere da viva.
"""
from __future__ import annotations

import contextlib
import gc
import io
import json
import warnings

import numpy as np
import pandas as pd
import pytest

import matplotlib  # noqa: E402
matplotlib.use("Agg")

from yggdrasil.credit_risk.tree import TreeSegmenter  # noqa: E402

TASKS = ["classification", "regression"]
COMPUTES = ("_compute_leaves", "_compute_leaf_order", "_compute_grade_map",
            "_compute_metrics", "_compute_psi", "_compute_feature_importance")


def _quieto():
    return contextlib.redirect_stdout(io.StringIO())


@contextlib.contextmanager
def _silencio():
    with _quieto(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _base(task, n=2500, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.beta(2.5, 3, n) * 1.4 + 0.3
    # NaN em x SÓ no OOT: o split numérico sem nó de faltantes manda os NaN
    # para a pior irmã (include_na), e a máscara viva difere da recomputada
    meses = pd.date_range("2023-01-01", periods=10, freq="MS")
    dt = rng.choice(meses, size=n)
    oot = dt >= meses[7]
    x[oot & (rng.random(n) < 0.15)] = np.nan
    gar = rng.choice(list("ABCD"), n, p=[0.5, 0.22, 0.18, 0.1]).astype(object)
    gar[rng.random(n) < 0.05] = None
    z = rng.normal(size=n)
    lg = {"A": 0.0, "B": 0.10, "C": 0.16, "D": 0.30}
    risco = (0.1 + 0.4 * np.nan_to_num(x - 0.5, nan=0.35)
             + np.array([lg.get(g, 0.2) for g in gar]) + 0.05 * (z > 0.5))
    if task == "classification":
        y = (rng.uniform(0, 1, n) < np.clip(risco, 0.01, 0.95)).astype(float)
    else:
        y = np.clip(risco + rng.normal(0, 0.07, n), 0, 1)
    y[rng.random(n) < 0.02] = np.nan
    df = pd.DataFrame({"x": x, "gar": gar, "z": z,
                       "saldo": rng.gamma(2.0, 1000.0, n), "target": y})
    df["dt_ref"] = dt
    df["amostra"] = np.where(oot, "OOT", np.where(rng.random(n) < 0.1, "ESTAB", "DES"))
    return df


def _seg(task, df, lru=True, peso=False):
    with _silencio():
        seg = TreeSegmenter(df, target="target", task_type=task, sample_col="amostra",
                            ref_sample="DES", date_col="dt_ref",
                            weight_col="saldo" if peso else None,
                            feature_labels={"x": "Score X"}, verbose=False)
    if not lru:
        seg._AGG_LRU_SIGS = 0          # memo desligado: só o cache por versão
    return seg


def _folhas(seg):
    return [s for s, v in seg.segments.items() if v["is_leaf"]]


def _estado(seg):
    """Todos os agregados memoizados (e o que deles depende), na ordem fixa."""
    with _silencio():
        out = {}
        for asc in (True, False):
            out[f"leaves_{asc}"] = seg.leaves(ascending=asc)
            out[f"order_{asc}"] = seg._leaf_order(asc)
            out[f"grade_{asc}"] = seg._grade_map(asc)
        out["leaves_psi"] = seg.leaves(with_psi=True)
        out["leaves_tst"] = seg.leaves(with_psi=True, with_test=True)
        out["leaves_welch"] = seg.leaves(with_test=True, test="welch")
        out["metrics"] = seg.metrics()
        out["metrics_02"] = seg.metrics(cutoff=0.2)
        out["psi"] = seg.psi()
        out["psi_det"] = seg.psi_detalhe()
        out["psi_e3"] = seg.psi(eps=1e-3)
        out["psi_det_e3"] = seg.psi_detalhe()
        out["psi_de_novo"] = seg.psi()
        out["psi_det_de_novo"] = seg.psi_detalhe()
        out["fi"] = seg.feature_importance()
        out["fi_bruto"] = seg.feature_importance(normalize=False)
        out["to_dict"] = json.dumps(seg.to_dict(), sort_keys=True, default=str)
        out["regua"] = json.dumps(seg._regua_dict(), sort_keys=True, default=str)
        out["mascaras"] = {s: seg.segments[s]["mask"].to_numpy().copy()
                           for s in seg.segments}
    return out


def _igual(a, b):
    assert a.keys() == b.keys()
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, pd.DataFrame):
            pd.testing.assert_frame_equal(va, vb, check_exact=True, obj=k)
        elif k == "mascaras":
            assert va.keys() == vb.keys()
            for s in va:
                np.testing.assert_array_equal(va[s], vb[s])
        else:
            assert va == vb, k


def _contador(monkeypatch, seg):
    cont = {n: 0 for n in COMPUTES}
    for nome in COMPUTES:
        orig = getattr(seg, nome)

        def wrap(*a, _o=orig, _n=nome, **k):
            cont[_n] += 1
            return _o(*a, **k)
        monkeypatch.setattr(seg, nome, wrap)
    return cont


# ----------------------------------------------------------------------
# Ações (as mesmas da UI: undo/redo = _prime_mask_cache + _load_segments)
# ----------------------------------------------------------------------
class _Hist:
    def __init__(self, seg):
        self.seg, self.undo, self.redo = seg, [], []

    def _snap(self):
        return self.seg.to_dict()["segments"]

    def acao(self, fn):
        self.undo.append(self._snap())
        self.redo.clear()
        with _silencio():
            fn(self.seg)

    def desfaz(self):
        prev = self.undo.pop()
        self.redo.append(self._snap())
        self.seg._prime_mask_cache()
        self.seg._load_segments(prev)

    def refaz(self):
        nxt = self.redo.pop()
        self.undo.append(self._snap())
        self.seg._prime_mask_cache()
        self.seg._load_segments(nxt)


def _roteiro():
    """Passos (nome, função(hist)) — iguais nas duas instâncias."""
    def grow_x(h):
        h.acao(lambda s: s.grow("x", splits=[0.8, 1.2]))

    def grow_gar(h):
        alvo = _folhas(h.seg)[0]
        h.acao(lambda s: s.grow("gar", only_segments=[alvo]))

    def grow_z(h):
        alvo = _folhas(h.seg)[-1]
        h.acao(lambda s: s.grow("z", splits=[0.0], only_segments=[alvo]))

    def collapse_pai(h):
        pai = next(s for s, v in h.seg.segments.items()
                   if not v["is_leaf"] and v["parent"] is not None)
        h.acao(lambda s: s.collapse(pai, verbose=False))

    def merge_volta_folha(h):
        # funde irmãs até o pai de 2 filhos voltar a ser folha
        alvo = next((b for p in {v["parent"] for v in h.seg.segments.values()
                                 if v["is_leaf"] and v["parent"] not in (None, "root")}
                     for _a, b in h.seg._adjacent_sibling_pairs(p)), None)
        assert alvo is not None
        h.acao(lambda s: s.merge_leaf(alvo, side="left", verbose=False))

    def move_cut(h):
        folha = next((s for s, v in h.seg.segments.items()
                      if v["is_leaf"] and v["conditions"]
                      and v["conditions"][-1]["kind"] == "num"
                      and v["conditions"][-1]["feature"] == "x"
                      and v["conditions"][-1]["hi"] == 0.8), None)
        assert folha is not None
        h.acao(lambda s: s.move_cut(folha, 0.9, verbose=False))

    def fit_auto(h):
        h.acao(lambda s: s.fit_auto(features=["x", "gar", "z"], max_depth=2,
                                    verbose=False))

    return [
        ("raiz", None),
        ("grow_x", grow_x), ("grow_gar", grow_gar),
        ("undo", lambda h: h.desfaz()), ("redo", lambda h: h.refaz()),
        ("undo2", lambda h: h.desfaz()),
        ("grow_z", grow_z), ("undo3", lambda h: h.desfaz()),
        ("redo_vazio_grow_gar", grow_gar),
        ("collapse", collapse_pai), ("grow_gar_de_novo", grow_gar),
        ("undo_grow", lambda h: h.desfaz()), ("undo_collapse", lambda h: h.desfaz()),
        ("merge", merge_volta_folha), ("undo_merge", lambda h: h.desfaz()),
        ("undo_para_x", lambda h: h.desfaz()),
        ("move_cut", move_cut), ("undo_move", lambda h: h.desfaz()),
        ("redo_move", lambda h: h.refaz()), ("undo_move2", lambda h: h.desfaz()),
        ("fit_auto", fit_auto), ("undo_fit", lambda h: h.desfaz()),
        ("redo_fit", lambda h: h.refaz()), ("undo_fit2", lambda h: h.desfaz()),
    ]


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("peso", [False, True])
def test_roteiro_igual_ao_memo_desligado(task, peso):
    df = _base(task)
    novo, velho = _seg(task, df, peso=peso), _seg(task, df, lru=False, peso=peso)
    hn, hv = _Hist(novo), _Hist(velho)
    hits = 0
    for nome, passo in _roteiro():
        if passo is not None:
            passo(hn)
            passo(hv)
        gc.collect()                       # pega reciclagem de id de máscara
        en, ev = _estado(novo), _estado(velho)
        _igual(en, ev)
        # 2ª leitura na mesma versão (cache por versão + reatribuição do detalhe)
        _igual(_estado(novo), en)
        hits += len(novo._agg_lru)
    assert len(velho._agg_lru) == 0
    assert hits > 0


@pytest.mark.parametrize("task", TASKS)
def test_undo_para_o_estado_anterior_nao_recalcula(task, monkeypatch):
    df = _base(task)
    seg = _seg(task, df)
    h = _Hist(seg)
    h.acao(lambda s: s.grow("x", splits=[0.8, 1.2]))
    e_a = _estado(seg)
    alvo = _folhas(seg)[0]
    h.acao(lambda s: s.grow("gar", only_segments=[alvo]))
    e_b = _estado(seg)
    cont = _contador(monkeypatch, seg)
    for passo, esperado in ((h.desfaz, e_a), (h.refaz, e_b), (h.desfaz, e_a)):
        v0 = seg._tree_version
        passo()
        gc.collect()
        assert seg._tree_version == v0 + 1          # a versão continua subindo
        _igual(_estado(seg), esperado)
        assert cont == {n: 0 for n in COMPUTES}, cont
    # com a LRU desligada, o mesmo undo recalcula tudo (o comportamento antigo)
    seg._AGG_LRU_SIGS = 0
    h.refaz()
    _estado(seg)
    assert cont["_compute_leaves"] > 0 and cont["_compute_metrics"] > 0
    assert cont["_compute_psi"] > 0


@pytest.mark.parametrize("task", TASKS)
def test_move_cut_ida_e_volta_da_miss_e_resultado_certo(task, monkeypatch):
    """move_cut troca as máscaras dos dois lados (objetos novos): voltar ao corte
    antigo pelo move_cut (e não pelo undo) não reaproveita nada e acerta."""
    df = _base(task)
    novo, velho = _seg(task, df), _seg(task, df, lru=False)
    for s in (novo, velho):
        with _silencio():
            s.grow("x", splits=[0.8, 1.2])
    _igual(_estado(novo), _estado(velho))
    folha = next(s for s, v in novo.segments.items()
                 if v["is_leaf"] and v["conditions"][-1].get("hi") == 0.8)
    for t in (0.9, 0.8):
        for s in (novo, velho):
            with _silencio():
                s.move_cut(folha, t, verbose=False)
        folha = next(s for s, v in novo.segments.items()
                     if v["is_leaf"] and v["conditions"][-1].get("hi") == t)
        cont = _contador(monkeypatch, novo)
        en = _estado(novo)
        assert cont["_compute_leaves"] > 0 and cont["_compute_metrics"] > 0
        _igual(en, _estado(velho))
        monkeypatch.undo()


@pytest.mark.parametrize("task", TASKS)
def test_collapse_e_mesmo_split_resultado_igual(task):
    """collapse seguido do MESMO grow: o grow cria máscaras novas (miss, seguro)
    e o resultado é o do memo desligado."""
    df = _base(task)
    novo, velho = _seg(task, df), _seg(task, df, lru=False)
    for s in (novo, velho):
        with _silencio():
            s.grow("x", splits=[0.8, 1.2])
            s.grow("gar", only_segments=[_folhas(s)[0]])
    _igual(_estado(novo), _estado(velho))
    pai = next(s for s, v in novo.segments.items()
               if not v["is_leaf"] and v["parent"] is not None)
    for s in (novo, velho):
        with _silencio():
            s.collapse(pai, verbose=False)
    gc.collect()
    _igual(_estado(novo), _estado(velho))
    for s in (novo, velho):
        with _silencio():
            s.grow("gar", only_segments=[pai])
    gc.collect()
    _igual(_estado(novo), _estado(velho))


def test_psi_detalhe_acompanha_o_ultimo_psi():
    """psi(eps=a) → psi(eps=b) → psi(eps=a): o detalhe é o de a (antes ficava o
    de b — corrigido; o 1º e o 2º passos são iguais ao antigo)."""
    df = _base("classification")
    seg = _seg("classification", df)
    with _silencio():
        seg.grow("x", splits=[0.8, 1.2])
    seg.psi(eps=1e-6)
    det_a = seg.psi_detalhe()
    seg.psi(eps=0.05)
    det_b = seg.psi_detalhe()
    seg.psi(eps=1e-6)
    pd.testing.assert_frame_equal(seg.psi_detalhe(), det_a, check_exact=True)
    # e o detalhe de b é o de um cálculo do zero com eps=b
    seg._agg_cache.clear(); seg._agg_lru.clear()
    seg._compute_psi(0.05)
    pd.testing.assert_frame_equal(pd.DataFrame(seg._psi_detalhe), det_b,
                                  check_exact=True)
    # o detalhe guardado é cópia: mexer no devolvido não contamina o cache
    seg.psi(eps=1e-6)
    seg._psi_detalhe[0]["psi_bin"] = 999.0
    seg.psi(eps=1e-6)
    pd.testing.assert_frame_equal(seg.psi_detalhe(), det_a, check_exact=True)


def test_psi_detalhe_depois_do_undo_e_o_do_estado_restaurado():
    df = _base("classification")
    novo, velho = _seg("classification", df), _seg("classification", df, lru=False)
    hn, hv = _Hist(novo), _Hist(velho)
    for h in (hn, hv):
        h.acao(lambda s: s.grow("x", splits=[0.8, 1.2]))
        h.seg.psi()
        h.acao(lambda s: s.grow("gar", only_segments=[_folhas(s)[0]]))
        h.seg.psi()
        h.desfaz()
    novo.psi()
    velho.psi()
    pd.testing.assert_frame_equal(novo.psi_detalhe(), velho.psi_detalhe(),
                                  check_exact=True)


def test_valor_da_lru_nao_e_corrompido_pelo_chamador():
    df = _base("classification")
    seg = _seg("classification", df)
    h = _Hist(seg)
    h.acao(lambda s: s.grow("x", splits=[0.8, 1.2]))
    lv, gm, od = seg.leaves(), seg._grade_map(), seg._leaf_order()
    lv.loc[0, "n"] = -1
    gm[0][next(iter(gm[0]))] = -1
    od.append("lixo")
    h.acao(lambda s: s.grow("gar", only_segments=[_folhas(s)[0]]))
    h.desfaz()                                     # volta: LRU hit
    pd.testing.assert_frame_equal(seg.leaves(), _seg_fresco(seg).leaves(),
                                  check_exact=True)
    assert seg._grade_map() == _seg_fresco(seg)._grade_map()
    assert seg._leaf_order() == _seg_fresco(seg)._leaf_order()


def _seg_fresco(seg):
    """Mesma árvore e MESMOS objetos-máscara, sem nenhum cache de agregado."""
    with _silencio():
        f = TreeSegmenter(seg.df, target="target", task_type=seg.task_type,
                          sample_col="amostra", ref_sample="DES", date_col="dt_ref",
                          feature_labels=dict(seg.feature_labels), verbose=False,
                          copy=False)
    f.segments = {sid: dict(s) for sid, s in seg.segments.items()}
    f._AGG_LRU_SIGS = 0
    return f


def test_assinatura_cobre_rotulo_condicoes_e_contexto():
    df = _base("classification")
    seg = _seg("classification", df)
    with _silencio():
        seg.grow("x", splits=[0.8, 1.2])
    s0 = seg._agg_assinatura()
    lv = seg.leaves()
    # mesmas máscaras, rótulo diferente → outra assinatura (não reaproveita)
    sid = _folhas(seg)[0]
    seg.segments[sid]["label"] = "outro"
    seg._bump_version()
    assert seg._agg_assinatura()[0] != s0[0]
    seg.segments[sid]["label"] = s0[0][[p[0] for p in s0[0]].index(sid)][4]
    # condições com include_na diferente → outra assinatura
    c = seg.segments[sid]["conditions"]
    seg.segments[sid]["conditions"] = c[:-1] + [{**c[-1], "include_na": True}]
    assert seg._agg_assinatura()[0] != s0[0]
    seg.segments[sid]["conditions"] = c
    assert seg._agg_assinatura()[0] == s0[0]
    # rótulo de variável trocado → a entrada não vale (descrição mudaria)
    seg._bump_version()
    seg.feature_labels["x"] = "Outro rótulo"
    lv2 = seg.leaves()
    assert lv2["descricao"].tolist() != lv["descricao"].tolist()
    assert all("Outro rótulo" in d for d in lv2["descricao"])
    seg.feature_labels["x"] = "Score X"
    seg._bump_version()
    pd.testing.assert_frame_equal(seg.leaves(), lv, check_exact=True)


def test_limites_inexatos_em_float_nao_casam_com_o_json():
    """int > 2**53 (ou Decimal) no corte: o estado vivo e o restaurado do JSON
    têm chaves diferentes (miss) — a ordem/texto poderiam diferir."""
    seg = _seg("classification", _base("classification"))
    big = 2 ** 53 + 1
    conds = [{"feature": "x", "kind": "num", "lo": float("-inf"), "hi": big}]
    k_vivo = seg._sig_conds(conds)
    restaurado = seg._conditions_from_json(seg._conditions_json(conds))
    assert seg._sig_conds(restaurado) != k_vivo
    ok = [{"feature": "x", "kind": "num", "lo": float("-inf"), "hi": np.float64(0.8)}]
    assert seg._sig_conds(seg._conditions_from_json(seg._conditions_json(ok))) \
        == seg._sig_conds(ok)


def test_lru_limita_assinaturas_e_mascaras(monkeypatch):
    df = _base("classification")
    seg = _seg("classification", df)
    monkeypatch.setattr(seg, "_AGG_LRU_SIGS", 3)
    folhas = []
    with _silencio():
        for t in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1):
            seg.collapse("root", verbose=False)
            seg.grow("x", splits=[t])
            seg.leaves()
            folhas.append(len(seg._agg_lru))
    assert max(folhas) <= 3 and len(seg._agg_lru) == 3
    # teto de máscaras distintas: a entrada nova sempre fica
    monkeypatch.setattr(seg, "_AGG_LRU_MASCARAS", 4)
    with _silencio():
        seg.collapse("root", verbose=False)
        seg.grow("x", splits=[0.65])
        seg.leaves()
    assert len(seg._agg_lru) == 1
    ent = next(iter(seg._agg_lru.values()))
    assert all(a is b for a, b in zip(ent["mascaras"],
                                      [s["mask"] for s in seg.segments.values()]))


def test_mascara_reciclada_nao_casa(monkeypatch):
    """Entrada com o mesmo id mas OUTRO objeto-máscara é inválida (`is`)."""
    df = _base("classification")
    seg = _seg("classification", df)
    with _silencio():
        seg.grow("x", splits=[0.8, 1.2])
    lv = seg.leaves()
    sig = seg._agg_assinatura()
    ent = seg._agg_lru[sig[0]]
    falsa = tuple(m.copy() for m in ent["mascaras"])
    ent["mascaras"] = falsa                      # simula id reciclado
    seg._bump_version()
    cont = _contador(monkeypatch, seg)
    pd.testing.assert_frame_equal(seg.leaves(), lv, check_exact=True)
    assert cont["_compute_leaves"] == 1
    # e a entrada foi substituída pela das máscaras de verdade
    assert all(a is b for a, b in zip(seg._agg_lru[sig[0]]["mascaras"], sig[1]))


def test_arrays_por_linha_ficam_fora_da_lru():
    df = _base("classification")
    seg = _seg("classification", df)
    with _silencio():
        seg.grow("x", splits=[0.8, 1.2])
    seg.metrics(); seg.psi(); seg._leaf_id_series(); seg.leaves(with_psi=True)
    nomes = {k[0] for e in seg._agg_lru.values() for k in e["vals"]}
    assert nomes <= TreeSegmenter._AGG_LRU_NOMES
    assert "leaf_codes" not in nomes and "leaf_id_arr" not in nomes


def test_build_com_excecao_nao_grava():
    df = _base("classification")
    seg = _seg("classification", df)
    with _silencio():
        seg.grow("x", splits=[0.8, 1.2])

    def falha():
        raise RuntimeError("x")
    with pytest.raises(RuntimeError):
        seg._agg_memo(("leaves", "teste"), falha)
    assert all(("leaves", "teste") not in e["vals"] for e in seg._agg_lru.values())


# ----------------------------------------------------------------------
# UI: _ordered_leaf_options e a tela no undo/redo
# ----------------------------------------------------------------------
def _ui(task, df, lru=True):
    pytest.importorskip("ipywidgets")
    from yggdrasil.credit_risk.tree import TreeSegmenterUI
    with _silencio():
        ui = TreeSegmenterUI(df, target="target", task_type=task, sample_col="amostra",
                             ref_sample="DES", date_col="dt_ref")
    if not lru:
        ui.seg._AGG_LRU_SIGS = 0
    return ui


def _split(ui, sid, feat, splits):
    ui._pending = dict(feature=feat, splits=splits, only_segments=[sid])
    with _silencio():
        ui._on_split(None)


def _tela(ui):
    return (ui.bar.value, ui.out_sql.value, ui.out_diag.value, ui.out_tree.value,
            ui.leaf_header.value, ui.leaf_chips.value,
            list(ui.dd_leaf.options), ui.dd_leaf.value,
            list(ui.dd_var_leaf.options), list(ui._ordered_leaf_options()))


@pytest.mark.parametrize("task", TASKS)
def test_ui_undo_redo_igual_ao_memo_desligado(task, monkeypatch):
    df = _base(task)
    novo, velho = _ui(task, df), _ui(task, df, lru=False)
    for ui in (novo, velho):
        _split(ui, "root", "x", [0.8, 1.2])
        _split(ui, _folhas(ui.seg)[0], "gar", None)
        ui.locked = {_folhas(ui.seg)[-1]}
    assert _tela(novo) == _tela(velho)
    for acao in ("_on_undo", "_on_redo", "_on_undo", "_on_undo", "_on_redo"):
        for ui in (novo, velho):
            with _silencio():
                getattr(ui, acao)(None)
        gc.collect()
        assert _tela(novo) == _tela(velho), acao
        _igual(_estado(novo.seg), _estado(velho.seg))


@pytest.mark.parametrize("task", TASKS)
def test_ui_undo_nao_recalcula_agregados_nem_opcoes(task, monkeypatch):
    from yggdrasil.credit_risk.tree import TreeSegmenterUI
    df = _base(task)
    ui = _ui(task, df)
    _split(ui, "root", "x", [0.8, 1.2])
    _split(ui, _folhas(ui.seg)[0], "gar", None)
    with _silencio():
        ui._on_undo(None)                       # estado A (visto antes)
        ui._on_redo(None)                       # estado B (visto antes)
    cont = _contador(monkeypatch, ui.seg)
    n_min = {"n": 0}
    orig = TreeSegmenterUI._min_nota_fn

    def wrap(self, *a, **k):
        n_min["n"] += 1
        return orig(self, *a, **k)
    monkeypatch.setattr(TreeSegmenterUI, "_min_nota_fn", wrap)
    opts_b = list(ui._ordered_leaf_options())
    with _silencio():
        ui._on_undo(None)
    assert cont == {n: 0 for n in COMPUTES}, cont
    with _silencio():
        ui._on_redo(None)
    assert cont == {n: 0 for n in COMPUTES}, cont
    antes = n_min["n"]
    assert ui._ordered_leaf_options() == opts_b
    assert n_min["n"] == antes                  # opções vieram da LRU
    # a lista devolvida é NOVA por versão (como o recálculo)
    assert ui._ordered_leaf_options() is ui._leaf_opts_cache[1]
    assert ui._leaf_opts_cache[1] is not next(reversed(ui._leaf_opts_lru.values()))[3]


@pytest.mark.parametrize("task", TASKS)
def test_ui_opcoes_respeitam_cadeado_e_apelido(task):
    df = _base(task)
    novo, velho = _ui(task, df), _ui(task, df, lru=False)
    for ui in (novo, velho):
        _split(ui, "root", "x", [0.8, 1.2])
    f = _folhas(novo.seg)
    for passo in range(3):
        for ui in (novo, velho):
            ui.locked = {f[passo % len(f)]} if passo != 1 else set()
            ui.seg.leaf_names = {f[0]: "Baixo"} if passo == 2 else {}
            with _silencio():
                _split(ui, f[-1], "gar", None)
                ui._on_undo(None)
            assert ui._ordered_leaf_options()
        assert novo._ordered_leaf_options() == velho._ordered_leaf_options()


@pytest.mark.parametrize("task", TASKS)
def test_ui_opcoes_trocam_so_o_apelido(task):
    """Mesma estrutura e MESMOS cadeados em todos os passos; só o apelido muda.
    Separa o apelido do cadeado na chave da LRU: uma chave sem os apelidos
    devolveria o rótulo velho depois de renomear e fazer undo/redo."""
    df = _base(task)
    novo, velho = _ui(task, df), _ui(task, df, lru=False)
    for ui in (novo, velho):
        _split(ui, "root", "x", [0.8, 1.2])
    f = _folhas(novo.seg)
    assert f == _folhas(velho.seg)
    for ui in (novo, velho):
        ui.locked = {f[1]}                      # cadeado fixo em toda a sequência

    def rotulo_f0(ui):
        # sem o prefixo "[nota]": a nota da folha muda quando o redo reabre o split
        lbl = next(lb for lb, sid in ui._ordered_leaf_options() if sid == f[0])
        return lbl.split("] ", 1)[1]

    for nome in (None, "Baixo", "Alto", None, "Baixo"):
        for ui in (novo, velho):
            ui.seg.set_leaf_name(f[0], nome)
            with _silencio():
                _split(ui, f[-1], "gar", None)  # mesma estrutura já vista ...
                ui._on_undo(None)               # ... e volta a ela (versão nova)
        gc.collect()
        assert novo._ordered_leaf_options() == velho._ordered_leaf_options(), nome
        lbl = rotulo_f0(novo)
        for outro in ("Baixo", "Alto"):
            assert (f"{outro} — " in lbl) == (outro == nome), (nome, lbl)
        # undo/redo entre estados já vistos, com o apelido atual
        for acao in ("_on_redo", "_on_undo"):
            for ui in (novo, velho):
                with _silencio():
                    getattr(ui, acao)(None)
            gc.collect()
            assert _tela(novo) == _tela(velho), (nome, acao)
            assert rotulo_f0(novo) == lbl, (nome, acao)


def test_ui_reset_limpa_lru_de_opcoes():
    df = _base("classification")
    ui = _ui("classification", df)
    _split(ui, "root", "x", [0.8, 1.2])
    ui._ordered_leaf_options()
    assert ui._leaf_opts_lru
    ui._limpa_caches_do_seg()
    assert "_leaf_opts_lru" not in ui.__dict__
