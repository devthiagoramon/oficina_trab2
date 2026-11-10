# Sistema de Recomendação por Filtragem Baseada em Conteúdo

Trabalho de Oficina I — recomendação de itens parecidos com os que o usuário já gostou, usando **apenas as características dos itens** (sem depender de outros usuários).

> **Status:** estrutura inicial do projeto. Domínio, dataset e implementação ainda serão definidos.

## Equipe

| Membro | Papel principal |
| --- | --- |
| Danilo Cavalcante | Engenharia de dados |
| Rennan Kaue | Modelagem |
| Thiago Ramon | Avaliação |
| Gabriel Ribeiro | Produto e documentação |

## Domínio e dataset

**Domínio:** jogos da Steam. Recomendamos jogos parecidos com os que o usuário já recomendou, usando só as características dos jogos.

- **Fonte:** [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) (Kaggle, Anton Kozyriev). Confira a licença na página antes de redistribuir.
- **Tabelas:** jogos, usuários, recomendações (~41 milhões, preferência binária recomendou / não recomendou) e um arquivo de metadados com descrição e tags de cada jogo.
- **Como baixar:** manualmente pelo Kaggle, colocando os arquivos em `data/raw/` (ou via script em `scripts/`, a criar). Os dados brutos não são versionados.
- **Amostragem:** a base completa é grande demais. Vamos filtrar uma amostra (ex.: jogos com mínimo de avaliações e usuários com mínimo de interações), documentando os cortes e a semente.
- **Atributos usados:** descrição (TF-IDF), tags (TF-IDF / one-hot) e, opcionalmente, atributos numéricos normalizados (ex.: preço).
- **Idioma:** descrições em inglês, então as stopwords são em inglês.

## Decisão de projeto

O núcleo do sistema **não usa bibliotecas de recomendação ou de NLP prontas** (scikit-learn, surprise, gensim, etc.). Implementamos em código próprio:

1. Pré-processamento de texto (minúsculas, remoção de pontuação, stopwords PT/EN, stemming opcional)
2. **TF-IDF** (TF, IDF e normalização L2)
3. Perfil do usuário (média ponderada pela nota; notas baixas podem ter peso negativo)
4. **Similaridade do cosseno** entre perfil e itens
5. Top-N excluindo itens já consumidos
6. Explicações (termos com maior contribuição e item do histórico mais parecido)
7. Métricas: Precision@N, Recall@N, nDCG@N, cobertura do catálogo, baselines aleatório e populares

`numpy` e `pandas` são usados só como estrutura de dados/álgebra básica.

## Estrutura das pastas

```
ap2/
├── backend/                  # API em Python (FastAPI)
│   ├── app/
│   │   ├── api/              # rotas e schemas (Pydantic)
│   │   ├── recommender/      # pré-processamento, TF-IDF, perfil, cosseno, Top-N, explicação
│   │   └── evaluation/       # split treino/teste, métricas e baselines
│   └── tests/                # testes unitários (pytest)
├── frontend/                 # interface de demonstração (Streamlit)
│   ├── pages/                # páginas da interface
│   └── components/           # componentes reutilizáveis
├── data/
│   ├── raw/                  # dados originais (ou script de download em scripts/)
│   └── processed/            # dados limpos e matrizes geradas
├── scripts/                  # download de dados, execução da avaliação
├── notebooks/                # análise exploratória
├── docs/                     # relatório, slides e figuras
├── results/                  # tabelas e gráficos da avaliação
├── requirements.txt
└── README.md
```

## Como instalar

Requer Python 3.10+.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Como executar

_Comandos a preencher conforme a implementação avançar._

```powershell
# 1. Backend (API)
uvicorn backend.app.main:app --reload --port 8000

# 2. Frontend (em outro terminal)
streamlit run frontend/app.py

# 3. Avaliação offline (reproduz a tabela de resultados)
python scripts/run_evaluation.py
```

## Avaliação

- Divisão por usuário: interações mais recentes (ou leave-one-out) como teste; perfil construído **só com treino**.
- Item relevante: jogo que o usuário **recomendou** (`is_recommended = True`). A preferência é binária, então o perfil usa peso +1 para recomendado e -1 para não recomendado (a confirmar com as horas jogadas como peso opcional).
- Métricas: Precision@N, Recall@N, nDCG@N (N = 5 e 10) + cobertura/diversidade.
- Baselines: aleatório e mais populares.
- Semente aleatória fixa: _a definir_.

## Vídeo de demonstração


## Cronograma

- Entrega final: **16/11/2026**.
