"""Leitura do catálogo Steam sem carregar o arquivo de interações em memória."""

import csv
import json
from pathlib import Path
import re


RAIZ = Path(__file__).resolve().parents[3]
EXTRAS = re.compile(r"\b(soundtrack|season pass|artbook|wallpapers?|upgrade|dlc|supporter pack|prologue|texture pack|deluxe edition|dedicated server)\b", re.I)


def caminhos_catalogo():
    for base in (RAIZ / "data" / "raw", RAIZ / "games_metadata", RAIZ / "games.csv"):
        jogos = base / "games.csv"
        metadados = base / "games_metadata.json"
        if jogos.exists() and metadados.exists():
            return jogos, metadados
    raise FileNotFoundError(
        "Catálogo não encontrado. Coloque games.csv e games_metadata.json em data/raw/."
    )


def carregar_catalogo(limite=8000, ids_obrigatorios=()):
    """Seleciona jogos com mais avaliações e inclui os IDs necessários à avaliação."""
    jogos_csv, metadados_json = caminhos_catalogo()
    with jogos_csv.open(encoding="utf-8-sig", newline="") as arquivo:
        linhas = list(csv.DictReader(arquivo))

    linhas.sort(key=lambda jogo: (-int(jogo.get("user_reviews") or 0), int(jogo["app_id"])))
    obrigatorios = set(map(int, ids_obrigatorios))

    metadados = {}
    with metadados_json.open(encoding="utf-8") as arquivo:
        for linha in arquivo:
            try:
                item = json.loads(linha)
                app_id = int(item["app_id"])
            except (ValueError, KeyError, TypeError):
                continue
            metadados[app_id] = item

    catalogo = {}
    for jogo in linhas:
        app_id = int(jogo["app_id"])
        if len(catalogo) >= limite and app_id not in obrigatorios:
            continue
        if EXTRAS.search(jogo["title"]):
            continue
        meta = metadados.get(app_id, {})
        tags = meta.get("tags") or []
        descricao = meta.get("description") or ""
        if not descricao.strip():
            continue
        catalogo[app_id] = {
            "app_id": app_id,
            "title": jogo["title"],
            "description": descricao,
            "tags": tags,
            "user_reviews": int(jogo.get("user_reviews") or 0),
            "positive_ratio": int(jogo.get("positive_ratio") or 0),
            "price_final": float(jogo.get("price_final") or 0),
            "date_release": jogo.get("date_release") or "",
        }
    return catalogo
