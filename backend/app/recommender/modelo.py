"""TF-IDF, perfil e cosseno implementados sem bibliotecas de NLP/recomendação."""

from collections import Counter, defaultdict
import heapq
import math
import re
import unicodedata

from .catalogo import carregar_catalogo


STOPWORDS = set("""a about after again all also an and are as at be been before but by can
could do does for from game games get has have in into is it its just more most of on or
our out over play player players that the their them there these this those through to
up use using very was were what when where which while who will with you your
o a os as um uma uns umas de da do das dos e em no na nos nas para por com sem
que se ao aos à às como mais menos muito jogo jogos jogador jogadores ser ter""".split())


def _normalizar(texto):
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(letra for letra in texto if not unicodedata.combining(letra))


def _stem(palavra):
    # Stemming leve e previsível para descrições em inglês.
    for sufixo in ("ization", "ations", "ation", "ingly", "ments", "ment", "ingly", "ing", "ers", "ies", "ed", "es", "s"):
        if palavra.endswith(sufixo) and len(palavra) - len(sufixo) >= 3:
            return palavra[:-len(sufixo)] + ("y" if sufixo == "ies" else "")
    return palavra


def _tokens(texto):
    return [_stem(p) for p in re.findall(r"[a-z0-9]+", _normalizar(texto))
            if len(p) > 2 and p not in STOPWORDS]


def _termos(jogo):
    termos = Counter(_tokens(jogo.get("description", "")))
    # A tag inteira tem peso maior que uma palavra incidental da descrição.
    for tag in jogo.get("tags", []):
        chave = "tag:" + "_".join(_tokens(tag))
        if chave != "tag:":
            termos[chave] += 4
    return termos


def _produto(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(valor * b.get(termo, 0.0) for termo, valor in a.items())


class RecomendadorConteudo:
    def __init__(self, treino=None, catalogo=None, limite_catalogo=8000, ids_obrigatorios=()):
        obrigatorios = set(treino["app_id"]) if treino is not None else set()
        obrigatorios.update(ids_obrigatorios)
        self.catalogo = catalogo if catalogo is not None else carregar_catalogo(limite_catalogo, obrigatorios)
        self.historicos = defaultdict(list)
        if treino is not None:
            for linha in treino[["user_id", "app_id", "is_recommended"]].itertuples(index=False):
                self.historicos[linha.user_id].append((int(linha.app_id), bool(linha.is_recommended)))

        contagens = {app_id: _termos(jogo) for app_id, jogo in self.catalogo.items()}
        self.tag_labels = {}
        for jogo in self.catalogo.values():
            for tag in jogo.get("tags", []):
                chave = "tag:" + "_".join(_tokens(tag))
                self.tag_labels.setdefault(chave, tag)
        df = Counter()
        for termos in contagens.values():
            df.update(termos.keys())
        total = len(contagens)
        self.vetores = {}
        self.indice = defaultdict(list)
        for app_id, termos in contagens.items():
            if not termos:
                continue
            vetor = {termo: (1 + math.log(freq)) * (math.log((1 + total) / (1 + df[termo])) + 1)
                     for termo, freq in termos.items()}
            norma = math.sqrt(sum(peso * peso for peso in vetor.values()))
            vetor = {termo: peso / norma for termo, peso in vetor.items()}
            self.vetores[app_id] = vetor
            for termo, peso in vetor.items():
                self.indice[termo].append((app_id, peso))

    def perfil(self, preferencias):
        """preferencias: sequência de (app_id, gostou). Itens negativos subtraem."""
        acumulado = defaultdict(float)
        usados = 0
        for app_id, gostou in preferencias:
            vetor = self.vetores.get(int(app_id))
            if vetor is None:
                continue
            usados += 1
            sinal = 1 if gostou else -1
            for termo, peso in vetor.items():
                acumulado[termo] += sinal * peso
        if not usados:
            return {}
        norma = math.sqrt(sum(peso * peso for peso in acumulado.values()))
        return {termo: peso / norma for termo, peso in acumulado.items() if peso} if norma else {}

    def recomendar_preferencias(self, preferencias, n=10):
        preferencias = [(int(app_id), bool(gostou)) for app_id, gostou in preferencias]
        vistos = {app_id for app_id, _ in preferencias}
        perfil = self.perfil(preferencias)
        if not perfil:
            return []
        escores = defaultdict(float)
        for termo, peso_perfil in perfil.items():
            for app_id, peso_jogo in self.indice.get(termo, ()):
                if app_id not in vistos:
                    escores[app_id] += peso_perfil * peso_jogo
        melhores = heapq.nlargest(n, ((max(0.0, score), app_id) for app_id, score in escores.items() if score > 0),
                                  key=lambda par: (par[0], -par[1]))
        resultados = []
        positivos = [app_id for app_id, gostou in preferencias if gostou and app_id in self.vetores]
        for score, app_id in melhores:
            vetor = self.vetores[app_id]
            contrib = sorted(((peso * vetor[termo], termo) for termo, peso in perfil.items()
                              if peso > 0 and termo in vetor), reverse=True)
            tags = [self.tag_labels.get(termo, termo[4:]) for _, termo in contrib if termo.startswith("tag:")][:3]
            parecido = max(positivos, key=lambda item: self.similaridade(item, app_id)) if positivos else None
            resultados.append({"app_id": app_id, "score": round(score, 4),
                               "tags_em_comum": tags,
                               "termos_em_comum": [termo for _, termo in contrib if not termo.startswith("tag:")][:3],
                               "parecido_com": self.catalogo[parecido]["title"] if parecido else None,
                               **self.catalogo[app_id]})
        return resultados

    def recomendar(self, user_id, vistos, n):
        preferencias = self.historicos.get(user_id, [])
        return [item["app_id"] for item in self.recomendar_preferencias(preferencias, n)
                if item["app_id"] not in vistos]

    def similaridade(self, a, b):
        return _produto(self.vetores.get(int(a), {}), self.vetores.get(int(b), {}))
