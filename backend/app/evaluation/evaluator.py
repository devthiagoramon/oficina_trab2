"""Roda um recomendador em todos os usuários de teste e tira a média das métricas.

Qualquer objeto com o método abaixo pode ser avaliado:
    recomendar(user_id, vistos, n) -> lista de app_id

Assim dá para comparar o modelo de conteúdo e os baselines do mesmo jeito.

Contrato para o modelo de conteúdo (pasta recommender/):
    - `vistos` é o conjunto de jogos que o usuário avaliou no TREINO
      (positivos e negativos). O modelo não pode recomendar nenhum deles.
    - O perfil do usuário deve ser montado só com o treino. Se o modelo
      precisar do histórico com as notas, ele recebe o DataFrame de treino
      no construtor e filtra por `user_id`.
    - A lista devolvida vem ordenada do melhor para o pior.
"""

import time

import pandas as pd

from . import metrics
from .split import itens_relevantes


def avaliar(recomendador, treino, teste, ks=(5, 10), tamanho_catalogo=None,
            similaridade=None):
    """Devolve um dicionário com a média de cada métrica em cada k.

    Só entram na média os usuários que têm pelo menos um jogo relevante
    (recomendado) no teste.

    tamanho_catalogo: nº de jogos que poderiam ser recomendados. Se não for
        passado, usa os jogos que aparecem no treino ou no teste.
    similaridade: função opcional sim(app_a, app_b) -> float. Se vier,
        calcula também a diversidade intra-lista no maior k.
    """
    relevantes_por_usuario = itens_relevantes(teste)
    vistos_por_usuario = treino.groupby("user_id")["app_id"].apply(set).to_dict()

    if tamanho_catalogo is None:
        tamanho_catalogo = len(set(treino["app_id"]) | set(teste["app_id"]))

    n_max = max(ks)
    soma = {}
    todas_listas = []
    soma_diversidade = 0.0

    inicio = time.perf_counter()
    for user_id, relevantes in relevantes_por_usuario.items():
        vistos = vistos_por_usuario.get(user_id, set())
        recomendados = list(recomendador.recomendar(user_id, vistos, n_max))
        todas_listas.append(recomendados)

        for k in ks:
            valores = {
                f"precision@{k}": metrics.precision_at_k(recomendados, relevantes, k),
                f"recall@{k}": metrics.recall_at_k(recomendados, relevantes, k),
                f"ndcg@{k}": metrics.ndcg_at_k(recomendados, relevantes, k),
                f"hit@{k}": metrics.hit_rate_at_k(recomendados, relevantes, k),
            }
            for nome, valor in valores.items():
                soma[nome] = soma.get(nome, 0.0) + valor

        if similaridade is not None:
            soma_diversidade += metrics.diversidade_intra_lista(recomendados, similaridade)

    n_usuarios = len(relevantes_por_usuario)
    resultados = {nome: total / n_usuarios for nome, total in soma.items()} if n_usuarios else {}

    resultados[f"cobertura@{n_max}"] = metrics.cobertura_catalogo(todas_listas, tamanho_catalogo)
    if similaridade is not None and n_usuarios:
        resultados[f"diversidade@{n_max}"] = soma_diversidade / n_usuarios
    resultados["usuarios_avaliados"] = n_usuarios
    resultados["segundos"] = round(time.perf_counter() - inicio, 2)
    return resultados


def comparar(recomendadores, treino, teste, ks=(5, 10), tamanho_catalogo=None,
             similaridade=None):
    """Avalia vários recomendadores e monta a tabela para o relatório.

    `recomendadores` é um dicionário nome -> recomendador.
    Cada linha da tabela é um modelo, cada coluna uma métrica.
    """
    linhas = []
    for nome, rec in recomendadores.items():
        print(f"Avaliando {nome}...")
        linha = avaliar(rec, treino, teste, ks, tamanho_catalogo, similaridade)
        linha["modelo"] = nome
        linhas.append(linha)

    tabela = pd.DataFrame(linhas).set_index("modelo")
    return tabela
