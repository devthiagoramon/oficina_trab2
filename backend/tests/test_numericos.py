"""Atributos numéricos: imputação de ausentes, z-score e efeito no ranking."""

import math

from backend.app.recommender.modelo import (LIMITE_Z, PESO_NUMERICO_VALIDADO, RecomendadorConteudo,
                                            _padronizar)


def catalogo():
    base = {"description": "space strategy fleets", "tags": ["Strategy", "Space"]}
    return {
        1: {"app_id": 1, "title": "Barato antigo", "price_final": 2.0, "date_release": "2005-01-01", **base},
        2: {"app_id": 2, "title": "Barato antigo 2", "price_final": 3.0, "date_release": "2006-06-01", **base},
        3: {"app_id": 3, "title": "Caro novo", "price_final": 60.0, "date_release": "2022-03-01", **base},
        4: {"app_id": 4, "title": "Sem dados", "price_final": "", "date_release": "", **base},
    }


def test_ausente_recebe_mediana_e_z_fica_limitado():
    z = _padronizar(catalogo())
    assert set(z) == {1, 2, 3, 4}
    valores = [v for item in z.values() for v in item.values()]
    assert all(abs(v) <= LIMITE_Z and not math.isnan(v) for v in valores)
    # o item sem dados recebe a mediana e não puxa para nenhum extremo
    assert abs(z[4].get("num:ano", 0)) < abs(z[3]["num:ano"])


def test_mesmo_texto_desempata_por_preco_e_epoca():
    rec = RecomendadorConteudo(catalogo=catalogo(), peso_numerico=PESO_NUMERICO_VALIDADO)
    assert rec.similaridade(1, 2) > rec.similaridade(1, 3)
    resultado = rec.recomendar_preferencias([(1, True)], 3)
    assert resultado[0]["app_id"] == 2
    assert resultado[0]["atributos_em_comum"]


def test_modelo_principal_nao_usa_numericos():
    rec = RecomendadorConteudo(catalogo=catalogo())
    assert not any(t.startswith("num:") for v in rec.vetores.values() for t in v)
