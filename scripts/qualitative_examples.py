"""Gera três exemplos reais de histórico, Top-10 e acerto/erro do modelo."""

from pathlib import Path
import sys

import pandas as pd


RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from backend.app.evaluation.split import itens_relevantes, split_temporal  # noqa: E402
from backend.app.recommender.catalogo import carregar_catalogo  # noqa: E402
from backend.app.recommender.modelo import RecomendadorConteudo  # noqa: E402


def main():
    caminho = RAIZ / "data" / "processed" / "interacoes.csv"
    if not caminho.exists():
        raise FileNotFoundError("Execute scripts/prepare_data.py primeiro")
    df = pd.read_csv(caminho, parse_dates=["date"])
    catalogo = carregar_catalogo(8000, df["app_id"].unique())
    df = df[df["app_id"].isin(catalogo)]
    treino, teste = split_temporal(df)
    modelo = RecomendadorConteudo(treino, catalogo=catalogo)
    relevantes = itens_relevantes(teste)
    exemplos = {"acerto": [], "erro": []}

    for user_id in sorted(relevantes):
        historico = modelo.historicos[user_id]
        if sum(gostou for _, gostou in historico) < 2:
            continue
        recomendados = modelo.recomendar_preferencias(historico, 10)
        if not recomendados:
            continue
        tipo = "acerto" if any(item["app_id"] in relevantes[user_id] for item in recomendados) else "erro"
        if len(exemplos[tipo]) < (2 if tipo == "acerto" else 1):
            exemplos[tipo].append((user_id, historico, relevantes[user_id], recomendados))
        if len(exemplos["acerto"]) == 2 and len(exemplos["erro"]) == 1:
            break

    linhas = ["# Exemplos qualitativos", "", "Amostra e split temporal com semente 42; apenas treino forma o perfil.", ""]
    for tipo in ("acerto", "erro"):
        for user_id, historico, alvo, recomendados in exemplos[tipo]:
            linhas += [f"## Usuário #{user_id} — {tipo}", ""]
            positivos = [catalogo[app_id]["title"] for app_id, gostou in historico if gostou]
            negativos = [catalogo[app_id]["title"] for app_id, gostou in historico if not gostou]
            linhas += ["**Gostou no treino:** " + "; ".join(positivos), "",
                      "**Não gostou no treino:** " + ("; ".join(negativos) or "nenhum"), "",
                      "**Relevante no teste:** " + "; ".join(catalogo[app_id]["title"] for app_id in sorted(alvo)), "",
                      "| Posição | Jogo | Cosseno | Explicação | No teste? |",
                      "| ---: | --- | ---: | --- | --- |"]
            for posicao, item in enumerate(recomendados, 1):
                motivo = ", ".join(item["tags_em_comum"] or item["termos_em_comum"])
                linhas.append(f"| {posicao} | {item['title'].replace('|', '/')} | {item['score']:.3f} | "
                              f"{motivo.replace('|', '/')} | {'sim' if item['app_id'] in alvo else 'não'} |")
            linhas += [""]
    saida = RAIZ / "results" / "exemplos_qualitativos.md"
    saida.parent.mkdir(exist_ok=True)
    saida.write_text("\n".join(linhas), encoding="utf-8")
    print(f"Exemplos salvos em {saida}")


if __name__ == "__main__":
    main()
