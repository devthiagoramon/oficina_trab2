# Avaliação experimental

## Protocolo

A fonte é o conjunto [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam). `scripts/prepare_data.py --usuarios 500` selecionou 6.214 interações de 500 usuários com semente 42. Após exigir descrição não vazia e remover extras identificados pelo título, restaram 3.688 interações de 483 usuários e 8.328 jogos candidatos. O split temporal reservou os 20% mais recentes de cada usuário com ao menos cinco interações restantes: 3.099 linhas de treino e 589 de teste. Apenas os 185 usuários com pelo menos uma avaliação positiva no teste entraram nas médias.

O perfil e o ranking de popularidade usam apenas treino. Um jogo relevante é aquele com `is_recommended = True` no teste. Conteúdo, aleatório e populares compartilham o mesmo catálogo candidato e excluem os jogos vistos. O baseline aleatório usa semente 42. As métricas foram calculadas por usuário e depois promediadas. Cobertura e diversidade usam o Top-10.

## Resultados

| Modelo | P@5 | R@5 | nDCG@5 | Hit@5 | P@10 | R@10 | nDCG@10 | Hit@10 | Cobertura@10 | Diversidade@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Aleatório | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,2004 | 0,9472 |
| Populares | 0,0054 | 0,0149 | 0,0075 | 0,0270 | 0,0032 | 0,0176 | 0,0085 | 0,0324 | 0,0024 | 0,8540 |
| Conteúdo (TF-IDF) | 0,0065 | 0,0221 | 0,0217 | 0,0324 | 0,0054 | 0,0358 | 0,0265 | 0,0541 | 0,1219 | 0,6992 |

O modelo de conteúdo melhora o Recall@10 e nDCG@10 em relação aos dois baselines e recomenda uma parcela maior do catálogo que o de populares. Os números absolutos são baixos: com mais de oito mil candidatos e poucos jogos positivos futuros por usuário, o Top-10 raramente contém o jogo reservado. A diversidade do modelo é menor que a dos baselines porque itens parecidos tendem a aparecer juntos; isto é esperado numa filtragem baseada em conteúdo e mostra superespecialização.

Os arquivos completos são [tabela CSV](../results/metricas_temporal.csv), [tabela Markdown](../results/metricas_temporal.md), [gráfico](../results/metricas_temporal.png) e [três exemplos qualitativos](../results/exemplos_qualitativos.md). Reproduza com `python scripts/prepare_data.py --usuarios 500`, `python scripts/run_evaluation.py` e `python scripts/qualitative_examples.py`.

## Leitura qualitativa

No usuário **#5170533**, o histórico combina ação e RPG, inclusive jogos de artes marciais. *Fate Seeker II* aparece em quarto lugar, com contribuições de `Martial Arts`, `Hack and Slash` e `RPG`.

No usuário **#5611614**, várias avaliações positivas são de jogos de guerra. *Post Scriptum* aparece em sétimo lugar, e as tags `Military`, `War` e `Historical` explicam a posição.

O usuário **#767** é um erro do Top-10: o teste contém *Dragon Age: Origins - Ultimate Edition*, mas a lista se concentra em jogos de *Half-Life* e fantasia clássica. O perfil mistura *The Witcher*, *Serious Sam* e *Half-Life 2*; termos compartilhados com a franquia *Half-Life* ocupam várias posições. Um passo futuro seria diversificar o ranking e testar pesos que valorizem mais as tags de RPG do que palavras genéricas da descrição.

## Limites do protocolo

O recorte privilegia jogos populares e descrições disponíveis. Avaliações positivas podem depender de preço, promoção, amigos ou contexto, que não entram no perfil. O teste temporal é mais realista que uma divisão aleatória, mas não mede satisfação com recomendações novas; mede recuperação de avaliações positivas registradas depois. Os filtros de extras são heurísticos e podem deixar passar expansões ou excluir algum jogo legítimo.
