"""Avaliação offline: reproduz a tabela de resultados do relatório.

Uso (na raiz do projeto):
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --usuarios 2000 --split temporal

Entrada:
    - Se existir data/processed/interacoes.csv (amostra já limpa), usa ela.
    - Senão, amostra direto de data/raw/recommendations.csv (Kaggle).
      O arquivo tem ~41 milhões de linhas, então é lido em pedaços.

Saída (pasta results/):
    metricas_<split>.csv   tabela completa
    metricas_<split>.md    mesma tabela pronta para colar no relatório
    metricas_<split>.png   gráfico de barras (se o matplotlib estiver instalado)
    significancia_<split>.csv/.md   bootstrap pareado do conteúdo contra os baselines
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from backend.app.evaluation.baselines import RecomendadorAleatorio, RecomendadorPopulares  # noqa: E402
from backend.app.evaluation.evaluator import comparar  # noqa: E402
from backend.app.evaluation.significancia import comparar_pares  # noqa: E402
from backend.app.evaluation.split import split_leave_one_out, split_temporal  # noqa: E402

COLUNAS = ["user_id", "app_id", "date", "is_recommended", "hours"]
TAMANHO_PEDACO = 2_000_000


def amostrar_do_bruto(caminho, n_usuarios, min_interacoes, seed):
    """Sorteia `n_usuarios` com pelo menos `min_interacoes` avaliações.

    1ª passada: conta quantas avaliações cada usuário tem.
    2ª passada: guarda só as linhas dos usuários sorteados.
    """
    print(f"Contando avaliações por usuário em {caminho.name}...")
    contagem = None
    for pedaco in pd.read_csv(caminho, usecols=["user_id"], chunksize=TAMANHO_PEDACO):
        parcial = pedaco["user_id"].value_counts()
        contagem = parcial if contagem is None else contagem.add(parcial, fill_value=0)

    elegiveis = sorted(contagem[contagem >= min_interacoes].index)
    print(f"{len(elegiveis)} usuários com >= {min_interacoes} avaliações")
    rng = random.Random(seed)
    escolhidos = set(rng.sample(elegiveis, min(n_usuarios, len(elegiveis))))

    partes = []
    for pedaco in pd.read_csv(caminho, usecols=COLUNAS, chunksize=TAMANHO_PEDACO):
        partes.append(pedaco[pedaco["user_id"].isin(escolhidos)])
    return pd.concat(partes, ignore_index=True)


def carregar_interacoes(args):
    processado = RAIZ / "data" / "processed" / "interacoes.csv"
    brutos = [RAIZ / "data" / "raw" / "recommendations.csv",
              RAIZ / "games_metadata" / "recommendations.csv",
              RAIZ / "games.csv" / "recommendations.csv"]
    bruto = next((c for c in brutos if c.exists()), None)

    if args.entrada:
        df = pd.read_csv(args.entrada)
    elif processado.exists():
        print(f"Usando amostra processada: {processado}")
        df = pd.read_csv(processado)
    elif bruto is not None:
        df = amostrar_do_bruto(bruto, args.usuarios, args.min_interacoes, args.seed)
    else:
        sys.exit(
            "Nenhum dado encontrado. Baixe o recommendations.csv do Kaggle para "
            "data/raw/ ou gere data/processed/interacoes.csv."
        )

    df["date"] = pd.to_datetime(df["date"])
    df["is_recommended"] = df["is_recommended"].astype(bool)
    return df


def montar_recomendadores(treino, seed, catalogo):
    from backend.app.recommender.modelo import RecomendadorConteudo

    modelo = RecomendadorConteudo(treino, catalogo=catalogo)
    ids = sorted(catalogo)
    recomendadores = {
        "Aleatório": RecomendadorAleatorio(ids, seed=seed),
        "Populares": RecomendadorPopulares(treino),
        "Conteúdo (TF-IDF)": modelo,
    }
    return recomendadores, modelo.similaridade, len(ids)


def tabela_markdown(tabela):
    colunas = [tabela.index.name or "modelo"] + list(tabela.columns)
    linhas = ["| " + " | ".join(colunas) + " |", "|" + " --- |" * len(colunas)]
    for modelo, valores in tabela.iterrows():
        celulas = [str(modelo)]
        for v in valores:
            celulas.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        linhas.append("| " + " | ".join(celulas) + " |")
    return "\n".join(linhas) + "\n"


def salvar_grafico(tabela, ks, caminho):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib não instalado; pulando o gráfico.")
        return

    metricas = [f"{m}@{k}" for k in ks for m in ("precision", "recall", "ndcg")]
    dados = tabela[metricas]
    ax = dados.T.plot(kind="bar", figsize=(10, 5), rot=0)
    ax.set_ylabel("valor médio por usuário")
    ax.set_title("Comparação dos recomendadores")
    ax.legend(title="modelo")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrada", type=Path, help="CSV de interações (sobrepõe a busca automática)")
    parser.add_argument("--split", choices=["temporal", "loo"], default="temporal")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--min-interacoes", type=int, default=5)
    parser.add_argument("--usuarios", type=int, default=500, help="tamanho da amostra de usuários")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=10_000,
                        help="reamostragens do teste de significância")
    args = parser.parse_args()

    df = carregar_interacoes(args)
    print(f"{len(df)} avaliações, {df['user_id'].nunique()} usuários, {df['app_id'].nunique()} jogos")
    from backend.app.recommender.catalogo import carregar_catalogo
    catalogo = carregar_catalogo(8000, df["app_id"].unique())
    df = df[df["app_id"].isin(catalogo)].reset_index(drop=True)
    print(f"Após exigir metadados e remover extras: {len(df)} avaliações, "
          f"{df['user_id'].nunique()} usuários, {len(catalogo)} jogos candidatos")

    if args.split == "temporal":
        treino, teste = split_temporal(df, args.test_frac, args.min_interacoes)
    else:
        treino, teste = split_leave_one_out(df, args.min_interacoes)
    print(f"Treino: {len(treino)} linhas | Teste: {len(teste)} linhas")

    recomendadores, similaridade, tamanho_catalogo = montar_recomendadores(treino, args.seed, catalogo)
    por_usuario = {}
    tabela = comparar(recomendadores, treino, teste, tuple(args.ks), tamanho_catalogo, similaridade,
                      por_usuario)

    pasta = RAIZ / "results"
    pasta.mkdir(exist_ok=True)
    base = pasta / f"metricas_{args.split}"
    tabela.to_csv(base.with_suffix(".csv"))
    base.with_suffix(".md").write_text(tabela_markdown(tabela), encoding="utf-8")
    salvar_grafico(tabela, args.ks, base.with_suffix(".png"))

    k = max(args.ks)
    significancia = comparar_pares(
        por_usuario, "Conteúdo (TF-IDF)", ["Populares", "Aleatório"],
        [f"recall@{k}", f"ndcg@{k}", f"hit@{k}"], n_amostras=args.bootstrap, seed=args.seed)
    base_sig = pasta / f"significancia_{args.split}"
    significancia.to_csv(base_sig.with_suffix(".csv"), index=False)
    base_sig.with_suffix(".md").write_text(
        tabela_markdown(significancia.set_index("comparacao")), encoding="utf-8")

    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200,
                           "display.max_columns", None):
        print(tabela)
        print(f"\nBootstrap pareado ({args.bootstrap} reamostragens, IC 95%):")
        print(significancia.to_string(index=False))
    print(f"\nResultados salvos em {pasta}")


if __name__ == "__main__":
    main()
