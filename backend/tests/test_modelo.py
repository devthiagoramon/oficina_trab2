"""Verifica ranking, exclusão de vistos e explicação no núcleo de conteúdo."""

import pandas as pd

from backend.app.recommender.modelo import RecomendadorConteudo


def catalogo_pequeno():
    return {
        1: {"app_id": 1, "title": "A", "description": "space strategy fleets", "tags": ["Strategy", "Space"]},
        2: {"app_id": 2, "title": "B", "description": "space strategy galaxies", "tags": ["Strategy", "Space"]},
        3: {"app_id": 3, "title": "C", "description": "farming garden crops", "tags": ["Farming"]},
        4: {"app_id": 4, "title": "D", "description": "space combat ships", "tags": ["Action", "Space"]},
    }


def test_recomendacao_prioriza_estilo_e_exclui_vistos():
    rec = RecomendadorConteudo(catalogo=catalogo_pequeno())
    resultado = rec.recomendar_preferencias([(1, True), (3, False)], 3)
    assert resultado[0]["app_id"] == 2
    assert {1, 3}.isdisjoint(item["app_id"] for item in resultado)
    assert "Space" in resultado[0]["tags_em_comum"]
    assert resultado[0]["parecido_com"] == "A"
    assert 0 < resultado[0]["score"] <= 1


def test_perfil_de_usuario_usa_apenas_treino():
    treino = pd.DataFrame({"user_id": [9, 9], "app_id": [1, 3], "is_recommended": [True, False]})
    rec = RecomendadorConteudo(treino=treino, catalogo=catalogo_pequeno())
    assert rec.recomendar(9, {1, 3}, 2)[0] == 2
    assert rec.recomendar(9, {1, 2, 3, 4}, 2) == []
