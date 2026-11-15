"""Gera relatorio.pdf a partir de relatorio.html usando o Edge ou o Chrome em modo headless.

Uso (na raiz do projeto): python docs/relatorio/gerar_pdf.py

O layout ABNT (A4, margens 3/2 cm, Times 12, numeração no canto superior
direito) está no CSS do próprio HTML. Para editar o texto, altere o HTML e
rode este script de novo. Os números de página do sumário são fixos no HTML:
confira-os no PDF depois de mudanças grandes.
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import time

PASTA = Path(__file__).resolve().parent
NAVEGADORES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def main():
    navegador = next((n for n in NAVEGADORES if Path(n).exists()), None)
    if navegador is None:
        sys.exit("Edge ou Chrome não encontrado. Abra relatorio.html no navegador e use Imprimir > Salvar como PDF "
                 "(margens: padrão; sem cabeçalhos e rodapés).")
    html = PASTA / "relatorio.html"
    pdf = PASTA / "relatorio.pdf"
    pdf.unlink(missing_ok=True)
    # Perfil temporário: sem ele, um navegador já aberto "engole" a chamada e não grava o PDF.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as perfil:
        subprocess.run([navegador, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={perfil}", "--allow-file-access-from-files",
                        f"--print-to-pdf={pdf}", html.as_uri()],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        # No Windows o processo pode retornar antes de o PDF terminar de ser gravado.
        for _ in range(120):
            if pdf.exists() and pdf.stat().st_size > 0:
                break
            time.sleep(0.5)
    if not pdf.exists():
        sys.exit("O navegador não gerou o PDF. Abra relatorio.html e use Imprimir > Salvar como PDF.")
    print(f"PDF gerado: {pdf}")


if __name__ == "__main__":
    main()
