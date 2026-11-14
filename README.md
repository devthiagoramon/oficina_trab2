# Steam Style — recomendação de jogos por conteúdo

Projeto de Oficina I. O sistema recomenda jogos parecidos com os que a pessoa gostou usando **descrições e tags**, sem consultar avaliações de outros usuários para calcular a similaridade. A interface permite marcar jogos manualmente ou explorar um usuário real de uma amostra da base.

## Equipe

| Membro | Papel principal |
| --- | --- |
| Danilo Cavalcante | Engenharia de dados |
| Rennan Kaue | Modelagem |
| Thiago Ramon | Avaliação |
| Gabriel Ribeiro | Produto e documentação |

## Dados

Fonte: [Game Recommendations on Steam, Anton Kozyriev (Kaggle)](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam).

### Baixar o dataset

O zip tem cerca de 660 MB, e os arquivos extraídos somam uns 2 GB. Na raiz do projeto:

```powershell
# Windows (PowerShell)
curl.exe -L -o game-recommendations-on-steam.zip https://www.kaggle.com/api/v1/datasets/download/antonkozyriev/game-recommendations-on-steam
Expand-Archive game-recommendations-on-steam.zip -DestinationPath data\raw
Remove-Item data\raw\users.csv, game-recommendations-on-steam.zip
```

```bash
# Linux/macOS
curl -L -o game-recommendations-on-steam.zip https://www.kaggle.com/api/v1/datasets/download/antonkozyriev/game-recommendations-on-steam
unzip game-recommendations-on-steam.zip games.csv games_metadata.json recommendations.csv -d data/raw
rm game-recommendations-on-steam.zip
```

Se o download falhar, baixe pelo botão **Download** da página do Kaggle (exige login) e extraia o zip em `data/raw/`. O `users.csv` que vem no zip não é usado. A pasta deve ficar assim:

```text
data/raw/
├── games.csv
├── games_metadata.json
└── recommendations.csv
```

`games.csv` e `games_metadata.json` bastam para o modo manual. `recommendations.csv` é necessário para selecionar usuários reais e fazer a avaliação offline. Os dados brutos não são versionados. O carregador também reconhece os arquivos nas pastas `games_metadata/` e `games.csv/`, como nesta cópia do projeto.

**Recorte da demonstração:** os 8.000 jogos com mais avaliações em `games.csv` que têm descrição não vazia, excluindo títulos claramente identificados como trilhas sonoras, passes de temporada, DLCs, artbooks, upgrades e servidores dedicados. A escolha é determinística; a ordenação desempata por `app_id`. Na avaliação, os jogos presentes nas interações amostradas são adicionados ao catálogo, inclusive os reservados para teste, se passarem pelos mesmos filtros. Isso usa apenas metadados, sem revelar preferências futuras.

## Instalação

Requer Python 3.10 ou superior.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Em Linux/macOS, ative com `source .venv/bin/activate`.

## Executar a interface

```powershell
streamlit run frontend/app.py
```

Selecione jogos de que gostou; opcionalmente, marque jogos de que não gostou. O Top-N aparece automaticamente, com pontuação de similaridade, termos em comum e o jogo do histórico mais parecido. Os jogos escolhidos são excluídos dos resultados. As imagens e links são carregados da Steam; o cálculo funciona sem eles.

Para habilitar **Explorar usuário da base**, prepare uma amostra real:

```powershell
python scripts/prepare_data.py --usuarios 500
```

O script lê `recommendations.csv` em blocos, usa seleção aritmética estável com semente 42, filtra usuários com pelo menos cinco jogos distintos e salva `data/processed/interacoes.csv`. A fração inicial de usuários pode ser alterada com `--fracao` (padrão: 2%). O original permanece intacto. O preparo percorre o arquivo grande uma vez e pode demorar alguns minutos.

## API opcional

```powershell
uvicorn backend.app.main:app --reload --port 8000
```

Rotas: `GET /health`, `GET /games?q=...`, `POST /recommendations` e `GET /users/{user_id}/recommendations`. Documentação em `http://localhost:8000/docs`.

Exemplo de `POST /recommendations`:

```json
{"preferencias": [{"app_id": 304930, "gostou": true}], "n": 5}
```

## Como o modelo funciona

1. Normaliza descrições para minúsculas, remove acentos, pontuação e stopwords PT/EN e aplica stemming leve. Tags inteiras recebem peso maior que termos incidentais.
2. Calcula TF-IDF em código próprio: TF logarítmica, IDF suavizada e normalização L2.
3. Monta o perfil somando vetores dos jogos recomendados e subtraindo os não recomendados; normaliza o perfil.
4. Calcula o cosseno entre perfil e jogos ainda não avaliados via índice invertido e ordena o Top-N.
5. Explica cada resultado com tags e palavras de maior contribuição positiva ao cosseno e com o jogo positivo mais parecido. A porcentagem exibida é **similaridade de cosseno**, não probabilidade de gostar.

O núcleo está em `backend/app/recommender/`. Não são usadas bibliotecas prontas de NLP/recomendação.

## Avaliação reproduzível

Após preparar a amostra:

```powershell
python scripts/run_evaluation.py
python scripts/run_evaluation.py --split loo
python -m pytest backend/tests
```

Antes do split, a avaliação mantém apenas interações com jogos que têm descrição e passam pelo filtro de extras. O split temporal separa os 20% mais recentes de cada usuário com ao menos cinco interações restantes. O leave-one-out reserva a última avaliação positiva e descarta interações posteriores. Apenas o treino constrói perfis e popularidade. Uma avaliação positiva (`is_recommended = True`) é relevante. O mesmo catálogo candidato e os mesmos usuários são usados nos três métodos: conteúdo, aleatório e mais populares. O baseline aleatório e a amostragem usam semente 42.

O script calcula Precision@5/10, Recall@5/10, nDCG@5/10, Hit Rate@5/10, cobertura e diversidade intra-lista, e grava tabelas CSV/Markdown e gráfico PNG em `results/`. Também faz um **bootstrap pareado por usuário** (10.000 reamostragens, semente 42) do modelo de conteúdo contra cada baseline em Recall@10, nDCG@10 e Hit@10, e grava diferença média, IC 95% e p-valor em `results/significancia_<split>.csv/.md`. Se o intervalo contém zero, a diferença não é significativa na amostra.

Gere os casos qualitativos com `python scripts/qualitative_examples.py`. O protocolo, a tabela obtida e a discussão de acertos/erro estão em [docs/avaliacao.md](docs/avaliacao.md).

## Estrutura

```text
backend/app/recommender/   leitura do catálogo, TF-IDF e recomendação
backend/app/evaluation/    divisão temporal, métricas e baselines
backend/app/main.py        API FastAPI
frontend/app.py            interface Streamlit
scripts/prepare_data.py    seleção reproduzível de interações
scripts/run_evaluation.py  tabela de avaliação
scripts/qualitative_examples.py  exemplos de acerto/erro
data/raw/                  dados originais (não versionados)
data/processed/            amostra de interações (não versionada)
results/                   métricas geradas
```

## Limitações e entrega

O recorte pelos jogos mais avaliados dá menos exposição a títulos pouco conhecidos. A recomendação pode se especializar demais e depende da qualidade das descrições e tags. Um usuário sem jogo positivo precisa escolher ao menos um título. Preço, lançamento e avaliações de terceiros não influenciam a similaridade.

O guia também exige relatório técnico (6–10 páginas), slides em PDF, análise qualitativa de 2–3 usuários e vídeo de demonstração de até três minutos. Esses materiais devem usar a tabela gerada pela avaliação, sem preencher resultados estimados. Prazo: **16/11/2026**.
