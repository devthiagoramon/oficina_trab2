"""Divisão treino/teste por usuário.

A ideia é simular o uso real: o sistema só conhece o passado do usuário
(treino) e tentamos acertar o que ele avaliou depois (teste).

Espera um DataFrame no formato do recommendations.csv do Kaggle,
com pelo menos as colunas: user_id, app_id, date, is_recommended.
"""

import pandas as pd


def split_temporal(df, test_frac=0.2, min_interacoes=5):
    """Para cada usuário, as avaliações mais recentes vão para o teste.

    Usuários com menos de `min_interacoes` ficam inteiros no treino,
    porque não dá para montar um perfil decente com 1 ou 2 jogos.
    """
    df = df.sort_values(["user_id", "date"])

    # posição de cada linha dentro do histórico do usuário (0, 1, 2, ...)
    posicao = df.groupby("user_id").cumcount()
    total = df.groupby("user_id")["app_id"].transform("count")

    n_teste = (total * test_frac).round().clip(lower=1)
    vai_pro_teste = (posicao >= total - n_teste) & (total >= min_interacoes)

    treino = df[~vai_pro_teste].reset_index(drop=True)
    teste = df[vai_pro_teste].reset_index(drop=True)
    return treino, teste


def split_leave_one_out(df, min_interacoes=5):
    """Deixa de fora só a última avaliação positiva de cada usuário.

    Tudo que vem antes dela fica no treino. O que vem depois é descartado,
    senão o perfil "veria o futuro".
    """
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    treino_partes = []
    teste_partes = []

    for user_id, historico in df.groupby("user_id"):
        positivos = historico[historico["is_recommended"]]
        if len(historico) < min_interacoes or positivos.empty:
            treino_partes.append(historico)
            continue

        ultimo = positivos.index[-1]
        treino_partes.append(historico.loc[: ultimo - 1])
        teste_partes.append(historico.loc[[ultimo]])

    treino = pd.concat(treino_partes, ignore_index=True)
    if teste_partes:
        teste = pd.concat(teste_partes, ignore_index=True)
    else:
        teste = df.iloc[0:0]
    return treino, teste


def itens_relevantes(teste):
    """Dicionário user_id -> conjunto de jogos que ele recomendou no teste.

    Só conta como acerto o jogo com is_recommended = True. Jogo que o
    usuário não recomendou não deveria aparecer no Top-N.
    """
    positivos = teste[teste["is_recommended"]]
    return positivos.groupby("user_id")["app_id"].apply(set).to_dict()
