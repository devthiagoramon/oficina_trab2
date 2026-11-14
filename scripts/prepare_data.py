"""Cria uma amostra reproduzível de interações reais para avaliação e demo.

Uso: python scripts/prepare_data.py --usuarios 500
Lê recommendations.csv em blocos; não altera os dados originais.
"""

import argparse
from pathlib import Path

import pandas as pd


RAIZ = Path(__file__).resolve().parent.parent
COLUNAS = ["user_id", "app_id", "date", "is_recommended", "hours"]


def fonte():
    for caminho in (RAIZ / "data" / "raw" / "recommendations.csv",
                    RAIZ / "games_metadata" / "recommendations.csv",
                    RAIZ / "games.csv" / "recommendations.csv"):
        if caminho.exists():
            return caminho
    raise FileNotFoundError("Coloque recommendations.csv em data/raw/.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usuarios", type=int, default=500)
    parser.add_argument("--min-interacoes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fracao", type=int, default=2,
                        help="porcentagem aproximada do conjunto de usuários a amostrar antes do filtro")
    args = parser.parse_args()
    if not 1 <= args.fracao <= 100 or args.usuarios <= 0:
        parser.error("--fracao deve estar entre 1 e 100 e --usuarios deve ser positivo")

    partes = []
    caminho = fonte()
    print(f"Lendo {caminho} em blocos...")
    for numero, pedaco in enumerate(pd.read_csv(caminho, usecols=COLUNAS, chunksize=500_000)):
        # Hash aritmético estável: seleciona as mesmas pessoas em cada bloco.
        chave = (pedaco["user_id"].astype("int64") * 1_103_515_245 + args.seed) % 10_007
        partes.append(pedaco[chave < args.fracao * 100])
        if numero % 20 == 0:
            print(f"{(numero + 1) * 500_000:,} linhas examinadas")
    if not partes:
        raise RuntimeError("Arquivo sem interações")
    amostra = pd.concat(partes, ignore_index=True)
    contagem = amostra.groupby("user_id")["app_id"].nunique()
    elegiveis = contagem[contagem >= args.min_interacoes].index
    if len(elegiveis) < 50:
        raise RuntimeError("Menos de 50 usuários elegíveis. Aumente --fracao.")
    escolhidos = pd.Series(elegiveis).sample(n=min(args.usuarios, len(elegiveis)), random_state=args.seed)
    amostra = amostra[amostra["user_id"].isin(escolhidos)]
    amostra = amostra.drop_duplicates(["user_id", "app_id"], keep="last")
    amostra = amostra.sort_values(["user_id", "date", "app_id"])
    saida = RAIZ / "data" / "processed" / "interacoes.csv"
    saida.parent.mkdir(parents=True, exist_ok=True)
    amostra.to_csv(saida, index=False)
    print(f"Salvos {len(amostra)} registros de {amostra['user_id'].nunique()} usuários em {saida}")


if __name__ == "__main__":
    main()
