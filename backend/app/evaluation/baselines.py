"""Baselines para comparar com o modelo baseado em conteúdo.

Os baselines seguem a mesma "assinatura" que o avaliador espera:
    recomendar(user_id, vistos, n) -> lista de app_id
onde `vistos` é o conjunto de jogos que o usuário já avaliou no treino.
"""

import random


class RecomendadorAleatorio:
    """Sorteia n jogos do catálogo que o usuário ainda não avaliou."""

    def __init__(self, catalogo, seed=42):
        # ordena para o sorteio não depender da ordem de um set
        self.catalogo = sorted(set(catalogo))
        self.rng = random.Random(seed)

    def recomendar(self, user_id, vistos, n):
        candidatos = [jogo for jogo in self.catalogo if jogo not in vistos]
        if len(candidatos) <= n:
            self.rng.shuffle(candidatos)
            return candidatos
        return self.rng.sample(candidatos, n)


class RecomendadorPopulares:
    """Recomenda os jogos com mais avaliações positivas no treino."""

    def __init__(self, treino):
        positivos = treino[treino["is_recommended"]]
        contagem = positivos["app_id"].value_counts()
        # desempate pelo app_id para o ranking ser sempre o mesmo
        contagem = contagem.rename_axis("app_id").reset_index(name="n")
        contagem = contagem.sort_values(["n", "app_id"], ascending=[False, True])
        self.ranking = list(contagem["app_id"])

    def recomendar(self, user_id, vistos, n):
        resultado = []
        for jogo in self.ranking:
            if jogo in vistos:
                continue
            resultado.append(jogo)
            if len(resultado) == n:
                break
        return resultado
