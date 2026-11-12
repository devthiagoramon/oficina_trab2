import math

import pandas as pd
import pytest

from backend.app.evaluation import metrics
from backend.app.evaluation.split import itens_relevantes, split_leave_one_out, split_temporal


# ---------- métricas (contas feitas à mão) ----------

def test_precision_e_recall():
    recomendados = [1, 2, 3, 4, 5]
    relevantes = {2, 5, 9}
    assert metrics.precision_at_k(recomendados, relevantes, 5) == pytest.approx(2 / 5)
    assert metrics.recall_at_k(recomendados, relevantes, 5) == pytest.approx(2 / 3)
    assert metrics.precision_at_k(recomendados, relevantes, 1) == 0


def test_ndcg_perfeito_vale_1():
    assert metrics.ndcg_at_k([7, 8, 1], {7, 8}, 3) == pytest.approx(1.0)


def test_ndcg_acerto_na_segunda_posicao():
    # só um relevante, na posição 2 -> 1/log2(3)
    assert metrics.ndcg_at_k([1, 2, 3], {2}, 3) == pytest.approx(1 / math.log2(3))


def test_sem_relevantes_da_zero():
    assert metrics.recall_at_k([1, 2], set(), 2) == 0
    assert metrics.ndcg_at_k([1, 2], set(), 2) == 0


def test_cobertura():
    assert metrics.cobertura_catalogo([[1, 2], [2, 3]], 10) == pytest.approx(0.3)


# ---------- split ----------

@pytest.fixture
def interacoes():
    # usuário 1 tem 5 avaliações, usuário 2 só tem 2
    return pd.DataFrame({
        "user_id": [1, 1, 1, 1, 1, 2, 2],
        "app_id": [10, 20, 30, 40, 50, 10, 20],
        "date": pd.to_datetime([
            "2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01",
            "2022-01-01", "2022-02-01",
        ]),
        "is_recommended": [True, True, False, True, False, True, True],
    })


def test_split_temporal_pega_os_mais_recentes(interacoes):
    treino, teste = split_temporal(interacoes, test_frac=0.4, min_interacoes=5)
    assert set(teste["app_id"]) == {40, 50}
    assert set(teste["user_id"]) == {1}  # usuário 2 tem poucas avaliações
    assert len(treino) + len(teste) == len(interacoes)


def test_leave_one_out_sem_vazar_futuro(interacoes):
    treino, teste = split_leave_one_out(interacoes, min_interacoes=5)
    # última positiva do usuário 1 é o jogo 40; o 50 vem depois e sai
    assert list(teste["app_id"]) == [40]
    assert 50 not in set(treino[treino["user_id"] == 1]["app_id"])


def test_relevantes_ignora_nao_recomendados(interacoes):
    _, teste = split_temporal(interacoes, test_frac=0.4, min_interacoes=5)
    assert itens_relevantes(teste) == {1: {40}}

