"""Testes dos helpers puros de yggdrasil.credit_risk._common (PSI e classificação)."""

import numpy as np

from yggdrasil.credit_risk._common import classifica_psi, psi_from_shares
from yggdrasil.monitoring.psi import PSI_SIGNIFICANT, PSI_STABLE


# ----------------------------------------------------------------------
# psi_from_shares
# ----------------------------------------------------------------------
def test_psi_from_shares_distribuicoes_iguais_e_zero():
    assert psi_from_shares([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0.0


def test_psi_from_shares_valor_conhecido():
    esperado = (0.7 - 0.5) * np.log(0.7 / 0.5) + (0.3 - 0.5) * np.log(0.3 / 0.5)
    assert psi_from_shares([0.5, 0.5], [0.7, 0.3]) == float(esperado)


def test_psi_from_shares_sequencias_vazias():
    # sem faixas → PSI 0.0 (não NaN, não erro)
    assert psi_from_shares([], []) == 0.0


def test_psi_from_shares_faixa_vazia_usa_eps():
    # share zero é truncado em eps → resultado finito e igual à fórmula com eps
    eps = 1e-6
    esperado = (0.5 - eps) * np.log(0.5 / eps) + (0.5 - 1.0) * np.log(0.5 / 1.0)
    assert psi_from_shares([0.0, 1.0], [0.5, 0.5], eps=eps) == float(esperado)
    assert np.isfinite(psi_from_shares([0.0, 0.5], [0.5, 0.0]))


def test_psi_from_shares_eps_parametrizado():
    # eps maior reduz a contribuição da faixa vazia (log menos extremo)
    alto = psi_from_shares([0.0, 1.0], [0.5, 0.5], eps=1e-6)
    baixo = psi_from_shares([0.0, 1.0], [0.5, 0.5], eps=1e-2)
    assert baixo < alto
    # eps idempotente: shares já truncados na origem não mudam o resultado
    assert (psi_from_shares([1e-6, 1.0], [0.5, 0.5], eps=1e-6)
            == psi_from_shares([0.0, 1.0], [0.5, 0.5], eps=1e-6))


def test_psi_from_shares_return_contrib():
    total, contribs = psi_from_shares([0.5, 0.5], [0.7, 0.3], return_contrib=True)
    assert len(contribs) == 2
    assert total == float(sum(contribs))
    assert all(c >= 0.0 for c in contribs)   # cada parcela do PSI é ≥ 0


# ----------------------------------------------------------------------
# classifica_psi (limiares únicos de yggdrasil.monitoring.psi)
# ----------------------------------------------------------------------
def test_classifica_psi_faixas_e_limiares():
    assert classifica_psi(0.05) == "estável"
    assert classifica_psi(0.15) == "atenção"
    assert classifica_psi(0.40) == "instável"
    # bordas: limiar exato já cai na faixa seguinte (comparação estrita)
    assert classifica_psi(PSI_STABLE) == "atenção"
    assert classifica_psi(PSI_SIGNIFICANT) == "instável"


def test_classifica_psi_indefinido():
    assert classifica_psi(None) == "—"
    assert classifica_psi(float("nan")) == "—"
    assert classifica_psi(np.nan) == "—"
    assert classifica_psi(np.float64("nan")) == "—"
