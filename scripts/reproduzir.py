"""Reproduz todos os resultados do trabalho com um único comando.

Uso (na raiz do projeto):
    python scripts/reproduzir.py            # usa os dados que já estão em data/raw/
    python scripts/reproduzir.py --baixar   # baixa antes o dataset do Kaggle (~660 MB)

Etapas, todas com semente 42:
    1. (opcional) baixa e extrai games.csv, games_metadata.json e recommendations.csv
    2. amostra de 500 usuários -> data/processed/interacoes.csv
    3. escolha do peso dos atributos numéricos na validação -> results/escolha_peso.*
    4. avaliação com split temporal e leave-one-out -> results/metricas_*, significancia_*
    5. exemplos qualitativos -> results/exemplos_qualitativos.md
    6. análise exploratória: executa notebooks/eda.ipynb -> figuras em results/eda/
    7. testes automatizados (pytest)
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

RAIZ = Path(__file__).resolve().parent.parent
BRUTOS = RAIZ / "data" / "raw"
ARQUIVOS = ("games.csv", "games_metadata.json", "recommendations.csv")
URL = "https://www.kaggle.com/api/v1/datasets/download/antonkozyriev/game-recommendations-on-steam"


def baixar():
    BRUTOS.mkdir(parents=True, exist_ok=True)
    destino = BRUTOS / "steam.zip"
    print(f"Baixando {URL} (~660 MB)...")
    with urllib.request.urlopen(URL) as resposta, destino.open("wb") as arquivo:
        shutil.copyfileobj(resposta, arquivo, length=1 << 20)
    with zipfile.ZipFile(destino) as pacote:
        for nome in ARQUIVOS:
            print(f"Extraindo {nome}...")
            pacote.extract(nome, BRUTOS)
    destino.unlink()


def etapa(titulo, *comando):
    print(f"\n=== {titulo} ===", flush=True)
    inicio = time.perf_counter()
    resultado = subprocess.run([sys.executable, *comando], cwd=RAIZ)
    if resultado.returncode != 0:
        sys.exit(f"Falhou em: {titulo} (código {resultado.returncode})")
    print(f"[ok] {titulo} em {time.perf_counter() - inicio:.0f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baixar", action="store_true", help="baixa o dataset do Kaggle se faltar algum arquivo")
    parser.add_argument("--usuarios", type=int, default=500)
    parser.add_argument("--sem-testes", action="store_true", help="não roda o pytest no final")
    args = parser.parse_args()

    faltando = [nome for nome in ARQUIVOS if not (BRUTOS / nome).exists()]
    if faltando and args.baixar:
        baixar()
    elif faltando:
        sys.exit(f"Faltam em data/raw/: {', '.join(faltando)}. "
                 "Rode com --baixar ou siga a seção 'Baixar o dataset' do README.")

    etapa("Amostra de interações", "scripts/prepare_data.py", "--usuarios", str(args.usuarios))
    etapa("Escolha do peso numérico (validação)", "scripts/escolher_peso.py")
    etapa("Avaliação - split temporal", "scripts/run_evaluation.py", "--split", "temporal")
    etapa("Avaliação - leave-one-out", "scripts/run_evaluation.py", "--split", "loo")
    etapa("Exemplos qualitativos", "scripts/qualitative_examples.py")
    etapa("Análise exploratória (notebook)", "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=1200", "notebooks/eda.ipynb")
    if not args.sem_testes:
        etapa("Testes", "-m", "pytest", "backend/tests", "-q")
    print(f"\nPronto. Resultados em {RAIZ / 'results'}")


if __name__ == "__main__":
    main()
