"""Interface de demonstração do recomendador de jogos."""

from html import escape
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from backend.app.recommender.modelo import RecomendadorConteudo  # noqa: E402


st.set_page_config(page_title="Steam Style · Descubra seu próximo jogo", page_icon="🎮", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { color-scheme: dark; }
.stApp { background: #0a101b; color: #eaf1fc; font-family: 'DM Sans', sans-serif; }
.block-container { max-width: 1300px; padding-top: 2rem; }
header[data-testid="stHeader"] { background: transparent; }
h1,h2,h3 { font-family: 'Space Grotesk',sans-serif; letter-spacing: -.035em; }
[data-testid="stSidebar"] { background: #111c2b; }
.stButton button[kind="primary"] { background: linear-gradient(110deg,#6ce0ce,#75b8ff); color:#07141d; border:0; font-weight:700; border-radius:10px; padding:.65rem 1.2rem; }
.stButton button[kind="primary"]:hover { color:#07141d; border:0; }
.stMultiSelect [data-baseweb="select"], .stSelectbox [data-baseweb="select"] { background:#17263a; border-radius:10px; }
.hero { position:relative; padding:2.2rem 2.4rem; border:1px solid #2a4258; border-radius:22px;
  background:radial-gradient(circle at 84% 12%,#225d6d 0%,transparent 32%),linear-gradient(125deg,#172b42,#101a2b 70%); margin-bottom:1.7rem; overflow:hidden; }
.hero::after { content:'✦'; position:absolute; right:7%; top:7%; color:#6ce0ce; opacity:.2; font-size:10rem; }
.eyebrow { color:#73dfcd; text-transform:uppercase; letter-spacing:.2em; font-size:.73rem; font-weight:700; }
.hero h1 { margin:.55rem 0 .8rem; font-size:clamp(2.3rem,4vw,4rem); line-height:1.05; color:#fff; max-width:650px; }
.hero p { margin:0; max-width:600px; color:#b8cbda; font-size:1.05rem; }
.section-label { color:#7c9ab1; font-size:.75rem; text-transform:uppercase; letter-spacing:.16em; font-weight:700; margin:1rem 0 .4rem; }
.game-card { display:flex; gap:1.1rem; padding:1rem; min-height:204px; border:1px solid #273a4e; border-radius:16px;
  background:linear-gradient(150deg,#18283a,#111d2d); margin-bottom:.9rem; box-shadow:0 12px 32px #02071133; }
.game-art { width:165px; min-width:165px; border-radius:11px; background-size:cover; background-position:center; background-color:#2b5264; display:flex; align-items:end; padding:.7rem; }
.game-art span { color:#d6fff6; font-family:'Space Grotesk',sans-serif; font-size:.65rem; font-weight:700; letter-spacing:.16em; text-shadow:0 2px 7px #07121e; }
.game-info { min-width:0; flex:1; }
.game-top { display:flex; justify-content:space-between; align-items:start; gap:.6rem; }
.game-title { color:#f2f8ff; font-family:'Space Grotesk',sans-serif; font-size:1.27rem; font-weight:700; line-height:1.2; }
.score { color:#70e0ca; white-space:nowrap; font-size:.82rem; font-weight:700; background:#174439; border-radius:30px; padding:.35rem .65rem; }
.tags { display:flex; flex-wrap:wrap; gap:.35rem; margin:.65rem 0; }
.tag { background:#243a4e; color:#acd2e4; border-radius:30px; font-size:.72rem; padding:.2rem .55rem; }
.description { color:#9fb3c5; font-size:.86rem; line-height:1.4; margin:.35rem 0; }
.reason { color:#d2e2ed; font-size:.8rem; border-top:1px solid #2b3c4b; padding-top:.6rem; margin-top:.55rem; }
.reason strong { color:#77decf; }
.tiny { color:#7894a8; font-size:.75rem; }
@media(max-width:700px) { .hero{padding:1.5rem;} .game-card{display:block;} .game-art{width:100%;height:150px;margin-bottom:.8rem;} }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparando o catálogo e os vetores dos jogos...")
def carregar_modelo():
    caminho = RAIZ / "data" / "processed" / "interacoes.csv"
    treino = pd.read_csv(caminho) if caminho.exists() else None
    return RecomendadorConteudo(treino)


try:
    rec = carregar_modelo()
except (FileNotFoundError, ValueError) as erro:
    st.error(str(erro))
    st.stop()

st.markdown("""
<div class="hero">
<div class="eyebrow">Steam Style / descoberta personalizada</div>
<h1>Seu próximo jogo tem a sua cara.</h1>
<p>Escolha alguns títulos que você curtiu. Encontramos jogos com estilos parecidos a partir das descrições e tags da Steam.</p>
</div>
""", unsafe_allow_html=True)

esquerda, direita = st.columns([1.05, 1.6], gap="large")
with esquerda:
    st.markdown('<div class="section-label">01 / Monte seu perfil</div>', unsafe_allow_html=True)
    tem_usuarios = bool(rec.historicos)
    modos = ["Escolher jogos"] + (["Explorar usuário da base"] if tem_usuarios else [])
    modo = st.radio("Como quer começar?", modos, horizontal=True)

    nomes = {f'{jogo["title"]} · #{app_id}': app_id for app_id, jogo in
             sorted(rec.catalogo.items(), key=lambda par: -par[1]["user_reviews"])}
    if modo == "Escolher jogos":
        favoritos = st.multiselect("Jogos que você gostou", list(nomes), placeholder="Busque pelo nome do jogo")
        evitar = st.multiselect("Jogos que não combinam com você (opcional)",
                               [nome for nome in nomes if nome not in favoritos],
                               placeholder="Adicione títulos que você não curtiu")
        preferencias = [(nomes[nome], True) for nome in favoritos] + [(nomes[nome], False) for nome in evitar]
    else:
        usuarios = sorted(rec.historicos, key=lambda uid: -len(rec.historicos[uid]))
        usuario = st.selectbox("Usuário da amostra", usuarios[:1000],
                               format_func=lambda uid: f"Usuário #{uid} · {len(rec.historicos[uid])} avaliações")
        preferencias = rec.historicos[usuario]
        historico = [rec.catalogo[app_id]["title"] for app_id, gostou in preferencias
                     if gostou and app_id in rec.catalogo]
        st.caption("Gostou de: " + ", ".join(historico[:8]) + ("..." if len(historico) > 8 else ""))

    quantidade = st.slider("Quantas recomendações?", 5, 20, 10, step=5)
    st.caption("As recomendações atualizam quando você muda suas escolhas.")
    st.markdown("---")
    st.markdown("**Como funciona**")
    st.caption("Descrições e tags viram vetores TF-IDF. O perfil combina os jogos escolhidos e a similaridade do cosseno ordena os resultados. Jogos já avaliados ficam de fora.")
    st.caption(f"Catálogo: {len(rec.catalogo):,} jogos com metadados · Origem: Kaggle / Steam".replace(",", "."))

with direita:
    st.markdown('<div class="section-label">02 / Descubra novos jogos</div>', unsafe_allow_html=True)
    if not preferencias:
        st.info("Selecione pelo menos um jogo de que você gostou para começar.")
    elif not any(gostou for _, gostou in preferencias):
        st.info("Adicione ao menos um jogo de que você gostou.")
    else:
        resultados = rec.recomendar_preferencias(preferencias, quantidade)
        if not resultados:
            st.warning("Nenhum jogo semelhante encontrado. Tente outros títulos.")
        for posicao, jogo in enumerate(resultados, 1):
            tags = "".join(f'<span class="tag">{escape(tag)}</span>' for tag in jogo["tags"][:4])
            razoes = jogo["tags_em_comum"] or jogo["termos_em_comum"]
            motivo = ", ".join(razoes) if razoes else "descrição semelhante"
            parecido = f' · Lembra <strong>{escape(jogo["parecido_com"])}</strong>' if jogo["parecido_com"] else ""
            descricao = escape(jogo["description"][:145].strip())
            if len(jogo["description"]) > 145:
                descricao += "…"
            st.markdown(f'''
            <div class="game-card">
              <div class="game-art" style="background-image:linear-gradient(0deg,#101b28bb,#101b2810),url('https://cdn.akamai.steamstatic.com/steam/apps/{jogo["app_id"]}/header.jpg')"><span>STEAM STYLE · {jogo["app_id"]}</span></div>
              <div class="game-info">
                <div class="game-top"><div class="game-title">{posicao:02d} · {escape(jogo["title"])}</div><div class="score">{jogo["score"]:.0%} similar</div></div>
                <div class="tags">{tags}</div>
                <div class="description">{descricao}</div>
                <div class="reason"><strong>Por que apareceu?</strong> Em comum: {escape(motivo)}{parecido}</div>
              </div>
            </div>''', unsafe_allow_html=True)
            st.markdown(f'[Ver na Steam ↗](https://store.steampowered.com/app/{jogo["app_id"]}/)', unsafe_allow_html=False)
