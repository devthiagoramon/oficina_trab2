import pandas as pd
import pytest

from backend.app.evaluation.evaluator import comparar
from backend.app.evaluation.significancia import bootstrap_pareado, comparar_pares


def test_modelos_iguais_nao_sao_significativos():
    valores = [0, 1, 0, 1, 1, 0]
    r = bootstrap_pareado(valores, valores, n_amostras=500)
    assert r["diferenca"] == 0
    assert r["ic_inferior"] == r["ic_superior"] == 0
    assert r["p_valor"] == 1.0


def test_modelo_sempre_melhor_e_significativo():
    a = [1.0] * 50
    b = [0.0] * 50
    r = bootstrap_pareado(a, b, n_amostras=500)
    assert r["diferenca"] == pytest.approx(1.0)
    assert r["ic_inferior"] > 0
    assert r["p_valor"] == 0


def test_poucos_acertos_tem_intervalo_cruzando_zero():
    # 2 acertos contra 1 em 100 usuários: diferença pequena demais
    a = [1, 1] + [0] * 98
    b = [1] + [0] * 99
    r = bootstrap_pareado(a, b, n_amostras=2000)
    assert r["ic_inferior"] <= 0 <= r["ic_superior"]
    assert r["p_valor"] > 0.05


def test_mesma_semente_mesmo_resultado():
    a, b = [0, 1, 1, 0, 1], [0, 0, 1, 0, 0]
    assert bootstrap_pareado(a, b, seed=7) == bootstrap_pareado(a, b, seed=7)


def test_tamanhos_diferentes_dao_erro():
    with pytest.raises(ValueError):
        bootstrap_pareado([1, 0], [1])


class Fixo:
    def __init__(self, lista):
        self.lista = lista

    def recomendar(self, user_id, vistos, n):
        return self.lista[:n]


def test_comparar_preenche_metricas_por_usuario():
    treino = pd.DataFrame({"user_id": [1, 2], "app_id": [10, 10], "is_recommended": [True, True]})
    teste = pd.DataFrame({"user_id": [1, 2], "app_id": [20, 30], "is_recommended": [True, True]})
    por_usuario = {}
    comparar({"A": Fixo([20, 40]), "B": Fixo([40, 50])}, treino, teste, ks=(2,),
             tamanho_catalogo=5, por_usuario=por_usuario)

    assert por_usuario["A"].loc[1, "hit@2"] == 1
    assert por_usuario["A"].loc[2, "hit@2"] == 0
    assert por_usuario["B"]["hit@2"].sum() == 0

    tabela = comparar_pares(por_usuario, "A", ["B"], ["hit@2"], n_amostras=200)
    assert tabela.loc[0, "comparacao"] == "A vs B"
    assert tabela.loc[0, "diferenca"] == pytest.approx(0.5)
