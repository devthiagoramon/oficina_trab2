import math

import pandas as pd
import pytest

from backend.app.evaluation import metrics
from backend.app.evaluation.baselines import RecomendadorAleatorio, RecomendadorPopulares
from backend.app.evaluation.evaluator import avaliar, comparar
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



# ---------- diversidade ----------

def test_diversidade_itens_identicos_e_diferentes():
    def sim(a, b):
        return 1.0 if a % 2 == b % 2 else 0.0  # pares parecidos entre si

    assert metrics.diversidade_intra_lista([2, 4, 6], sim) == pytest.approx(0.0)
    # pares: (1,2)=1, (1,3)=0, (2,3)=1 -> média 2/3
    assert metrics.diversidade_intra_lista([1, 2, 3], sim) == pytest.approx(2 / 3)
    assert metrics.diversidade_intra_lista([1], sim) == 0


# ---------- baselines ----------

def test_populares_ordena_por_positivos_e_pula_vistos(interacoes):
    rec = RecomendadorPopulares(interacoes)
    # positivos: 10 e 20 têm 2 cada, 40 tem 1 (30 e 50 só negativos)
    assert rec.recomendar(99, set(), 3) == [10, 20, 40]
    assert rec.recomendar(99, {10}, 2) == [20, 40]


def test_aleatorio_reprodutivel_e_sem_vistos():
    catalogo = list(range(100))
    a = RecomendadorAleatorio(catalogo, seed=7).recomendar(1, {0, 1, 2}, 10)
    b = RecomendadorAleatorio(catalogo, seed=7).recomendar(1, {0, 1, 2}, 10)
    assert a == b
    assert len(a) == len(set(a)) == 10
    assert not {0, 1, 2} & set(a)


def test_aleatorio_catalogo_menor_que_n():
    rec = RecomendadorAleatorio([1, 2, 3], seed=0)
    assert sorted(rec.recomendar(1, {2}, 10)) == [1, 3]


# ---------- avaliador ----------

class _Fixo:
    """Recomendador de mentira que sempre devolve a mesma lista."""

    def __init__(self, lista):
        self.lista = lista
        self.vistos_recebidos = {}

    def recomendar(self, user_id, vistos, n):
        self.vistos_recebidos[user_id] = vistos
        return self.lista[:n]


def test_avaliar_media_por_usuario(interacoes):
    treino, teste = split_temporal(interacoes, test_frac=0.4, min_interacoes=5)
    rec = _Fixo([40, 99, 98, 97, 96])
    r = avaliar(rec, treino, teste, ks=(1, 5), tamanho_catalogo=10)

    assert r["usuarios_avaliados"] == 1          # só o usuário 1 tem teste positivo
    assert r["precision@1"] == pytest.approx(1.0)
    assert r["precision@5"] == pytest.approx(1 / 5)
    assert r["recall@5"] == pytest.approx(1.0)
    assert r["hit@5"] == 1.0
    assert r["cobertura@5"] == pytest.approx(0.5)
    # o recomendador recebe só o que o usuário viu no treino
    assert rec.vistos_recebidos[1] == {10, 20, 30}


def test_avaliar_com_diversidade(interacoes):
    treino, teste = split_temporal(interacoes, test_frac=0.4, min_interacoes=5)
    r = avaliar(_Fixo([40, 41]), treino, teste, ks=(2,), similaridade=lambda a, b: 0.25)
    assert r["diversidade@2"] == pytest.approx(0.75)


def test_comparar_monta_tabela(interacoes):
    treino, teste = split_temporal(interacoes, test_frac=0.4, min_interacoes=5)
    tabela = comparar({"a": _Fixo([40]), "b": _Fixo([99])}, treino, teste, ks=(5,))
    assert list(tabela.index) == ["a", "b"]
    assert tabela.loc["a", "hit@5"] > tabela.loc["b", "hit@5"]
