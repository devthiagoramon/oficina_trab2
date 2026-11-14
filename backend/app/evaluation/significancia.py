"""Teste de significância pareado por usuário (bootstrap).

Com poucas centenas de usuários e métricas perto de zero, uma diferença
entre dois modelos pode ser só sorte da amostra. O bootstrap pareado
reamostra os usuários com reposição e recalcula a diferença média entre
os modelos; o intervalo de 95% dessas médias mostra o quanto o resultado
é estável. Se o intervalo não contém zero, a diferença é significativa.

É pareado porque os dois modelos são avaliados nos mesmos usuários:
compara-se a diferença dentro de cada usuário, não as médias soltas.
"""

import numpy as np
import pandas as pd


def bootstrap_pareado(valores_a, valores_b, n_amostras=10_000, seed=42, confianca=0.95):
    """Diferença média (A - B) com intervalo de confiança e p-valor bilateral.

    valores_a, valores_b: métrica de cada usuário, na mesma ordem.
    Devolve dicionário com diferenca, ic_inferior, ic_superior e p_valor.
    """
    a = np.asarray(valores_a, dtype=float)
    b = np.asarray(valores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("os dois modelos precisam ter um valor por usuário")
    n = len(a)
    if n == 0:
        return {"diferenca": 0.0, "ic_inferior": 0.0, "ic_superior": 0.0, "p_valor": 1.0}

    diferencas = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_amostras, n))
    medias = diferencas[indices].mean(axis=1)

    alfa = 1 - confianca
    inferior, superior = np.quantile(medias, [alfa / 2, 1 - alfa / 2])
    # p-valor bilateral: com que frequência a reamostragem cruza o zero
    p_valor = 2 * min((medias <= 0).mean(), (medias >= 0).mean())
    return {
        "diferenca": float(diferencas.mean()),
        "ic_inferior": float(inferior),
        "ic_superior": float(superior),
        "p_valor": float(min(p_valor, 1.0)),
    }


def comparar_pares(por_usuario, modelo, referencias, metricas, **kwargs):
    """Tabela com o bootstrap de `modelo` contra cada referência.

    por_usuario: dicionário nome_do_modelo -> DataFrame (índice user_id,
        uma coluna por métrica), como o preenchido por `comparar`.
    """
    alvo = por_usuario[modelo]
    linhas = []
    for referencia in referencias:
        base = por_usuario[referencia].reindex(alvo.index)
        for metrica in metricas:
            resultado = bootstrap_pareado(alvo[metrica], base[metrica], **kwargs)
            linhas.append({"comparacao": f"{modelo} vs {referencia}",
                           "metrica": metrica, **resultado})
    return pd.DataFrame(linhas)
