"""API opcional para consumir o mesmo recomendador usado no Streamlit."""

from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .recommender.modelo import RecomendadorConteudo


RAIZ = Path(__file__).resolve().parents[2]
app = FastAPI(title="Steam Style", description="Recomendação de jogos por conteúdo")


class Preferencia(BaseModel):
    app_id: int
    gostou: bool = True


class Pedido(BaseModel):
    preferencias: list[Preferencia] = Field(min_length=1)
    n: int = Field(default=10, ge=1, le=30)


@lru_cache(maxsize=1)
def modelo():
    caminho = RAIZ / "data" / "processed" / "interacoes.csv"
    treino = pd.read_csv(caminho) if caminho.exists() else None
    return RecomendadorConteudo(treino)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/games")
def buscar_jogos(q: str = "", limite: int = Query(20, ge=1, le=100)):
    encontrados = [jogo for jogo in modelo().catalogo.values() if q.casefold() in jogo["title"].casefold()]
    encontrados.sort(key=lambda jogo: -jogo["user_reviews"])
    return [{"app_id": jogo["app_id"], "title": jogo["title"], "tags": jogo["tags"][:4]}
            for jogo in encontrados[:limite]]


@app.post("/recommendations")
def recomendar(pedido: Pedido):
    rec = modelo()
    faltantes = [p.app_id for p in pedido.preferencias if p.app_id not in rec.catalogo]
    if faltantes:
        raise HTTPException(422, f"Jogos fora do catálogo: {faltantes}")
    return {"results": rec.recomendar_preferencias(
        [(p.app_id, p.gostou) for p in pedido.preferencias], pedido.n
    )}


@app.get("/users/{user_id}/recommendations")
def recomendar_usuario(user_id: int, n: int = Query(10, ge=1, le=30)):
    rec = modelo()
    if user_id not in rec.historicos:
        raise HTTPException(404, "Usuário não encontrado na amostra. Execute scripts/prepare_data.py.")
    return {"user_id": user_id, "results": rec.recomendar_preferencias(rec.historicos[user_id], n)}
