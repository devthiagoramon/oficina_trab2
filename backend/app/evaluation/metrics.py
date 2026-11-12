"""Métricas de ranking para o Top-N.

Todas recebem:
    recomendados: lista de app_id na ordem do ranking (o 1º é o melhor)
    relevantes: conjunto de app_id que o usuário realmente gostou no teste
"""

import math


def precision_at_k(recomendados, relevantes, k):
    """Dos k primeiros recomendados, quantos % o usuário gostou."""
    if k <= 0:
        return 0.0
    top_k = recomendados[:k]
    acertos = sum(1 for item in top_k if item in relevantes)
    return acertos / k


def recall_at_k(recomendados, relevantes, k):
    """Dos jogos que o usuário gostou, quantos % apareceram no top k."""
    if not relevantes:
        return 0.0
    top_k = recomendados[:k]
    acertos = sum(1 for item in top_k if item in relevantes)
    return acertos / len(relevantes)


def ndcg_at_k(recomendados, relevantes, k):
    """nDCG com relevância binária.

    Acertar na 1ª posição vale mais que acertar na 10ª. O desconto é
    1 / log2(posição + 1), com posição começando em 1.
    """
    if not relevantes:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(recomendados[:k]):
        if item in relevantes:
            dcg += 1 / math.log2(i + 2)

    # DCG ideal: todos os relevantes nas primeiras posições
    n_ideal = min(len(relevantes), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(n_ideal))

    return dcg / idcg


def hit_rate_at_k(recomendados, relevantes, k):
    """1 se acertou pelo menos um jogo no top k, senão 0.

    Útil no leave-one-out, onde cada usuário tem só 1 item de teste.
    """
    return 1.0 if any(item in relevantes for item in recomendados[:k]) else 0.0


def cobertura_catalogo(todas_recomendacoes, tamanho_catalogo):
    """Fração do catálogo que apareceu em pelo menos uma recomendação.

    O baseline de populares costuma ter cobertura bem baixa, porque
    recomenda os mesmos jogos para todo mundo.
    """
    if tamanho_catalogo == 0:
        return 0.0
    distintos = set()
    for lista in todas_recomendacoes:
        distintos.update(lista)
    return len(distintos) / tamanho_catalogo


def diversidade_intra_lista(recomendados, similaridade):
    """Média de (1 - similaridade) entre todos os pares da lista.

    `similaridade(a, b)` devolve um valor entre 0 e 1 (ex.: cosseno entre
    os vetores TF-IDF dos dois jogos). Lista com 0 ou 1 item tem
    diversidade 0, porque não há pares para comparar.
    """
    n = len(recomendados)
    if n < 2:
        return 0.0
    total = 0.0
    pares = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1 - similaridade(recomendados[i], recomendados[j])
            pares += 1
    return total / pares
