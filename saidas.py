"""
saidas.py — ponto único para os caminhos de saída. Tudo que as simulações
gravam passa pelo PathManager (path.py, na raiz do projeto).

Colocar na raiz do projeto, ao lado de path.py.

Layout (PathManager(variable_type="qumodes", sub_folder=<pasta>)):
    <raiz de resultados>/qumodes/<pasta>/              arquivos do experimento
    <raiz de resultados>/qumodes/<pasta>/runs/         um JSON enxuto por execução
    <raiz de resultados>/qumodes/<pasta>/arrays/       históricos pesados (.npz)
    <raiz de resultados>/qumodes/<pasta>/figures/      figuras (PNG)
    <raiz de resultados>/qumodes/<pasta>/logs/         logs

A raiz de resultados é a do PathManager: variável OUTPUT_DIR, ou
/home-ext/clovis/results se existir, ou results/ ao lado do path.py.
Observação do PathManager: variable_type e sub_folder viram minúsculas.
"""
from pathlib import Path
from typing import Optional

from path import PathManager


def gerenciador(pasta: str, variable_type: str = "qumodes") -> PathManager:
    return PathManager(variable_type=variable_type, sub_folder=pasta)


def caminho(nome: str, pasta: str, subdir: Optional[str] = None,
            figura: bool = False) -> str:
    """Caminho completo (str) de um arquivo dentro da pasta do experimento."""
    return str(gerenciador(pasta).get_file_path(nome, is_figure=figura, subdir=subdir))


def pasta_do_experimento(pasta: str) -> Path:
    return Path(caminho("_", pasta)).parent