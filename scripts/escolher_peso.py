"""Escolhe o peso dos atributos numéricos (preço e ano) sem olhar o teste.

O treino do split temporal é dividido de novo, com o mesmo protocolo, em
subtreino e validação. Cada peso é avaliado só na validação; o conjunto de
teste usado em run_evaluation.py nunca entra nesta escolha.

Uso: python scripts/escolher_peso.py [--pesos 0 3 5 8 12 16 24 32]
Saída: results/escolha_peso.csv/.md
"""

import argparse
from pathlib import Path
import sys

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from backend.app.evaluation.evaluator import avaliar  # noqa: E402
from backend.app.evaluation.split import split_temporal  # noqa: E402
from backend.app.recommender.catalogo import carregar_catalogo  # noqa: E402
from backend.app.recommender.modelo import RecomendadorConteudo  # noqa: E402

METRICAS = ["precision@10", "recall@10", "ndcg@10", "hit@10", "cobertura@10"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pesos", type=float, nargs="+", default=[0, 3, 5, 8, 12, 16, 24, 32])
    args = parser.parse_args()

    caminho = RAIZ / "data" / "processed" / "interacoes.csv"
    if not caminho.exists():
        sys.exit("Execute scripts/prepare_data.py primeiro")
    df = pd.read_csv(caminho, parse_dates=["date"])
    catalogo = carregar_catalogo(8000, df["app_id"].unique())
    df = df[df["app_id"].isin(catalogo)].reset_index(drop=True)

    treino, _ = split_temporal(df)            # o teste é descartado aqui
    subtreino, validacao = split_temporal(treino)
    print(f"Subtreino: {len(subtreino)} linhas | Validação: {len(validacao)} linhas")

    linhas = []
    for peso in args.pesos:
        modelo = RecomendadorConteudo(subtreino, catalogo=catalogo, peso_numerico=peso)
        r = avaliar(modelo, subtreino, validacao, tamanho_catalogo=len(catalogo))
        linhas.append({"peso_numerico": peso, **{m: r[m] for m in METRICAS},
                       "usuarios_avaliados": r["usuarios_avaliados"]})
        print(f"peso={peso:g}: " + ", ".join(f"{m}={r[m]:.4f}" for m in METRICAS))

    tabela = pd.DataFrame(linhas)
    pasta = RAIZ / "results"
    pasta.mkdir(exist_ok=True)
    tabela.to_csv(pasta / "escolha_peso.csv", index=False)
    cabecalho = "| " + " | ".join(tabela.columns) + " |\n|" + " --- |" * len(tabela.columns) + "\n"
    corpo = "".join("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in linha) + " |\n"
                    for linha in tabela.itertuples(index=False))
    (pasta / "escolha_peso.md").write_text(cabecalho + corpo, encoding="utf-8")
    melhor = tabela.sort_values(["ndcg@10", "peso_numerico"], ascending=[False, True]).iloc[0]
    print(f"\nMaior nDCG@10 na validação: peso {melhor['peso_numerico']:g}")


if __name__ == "__main__":
    main()
