# path.py
from pathlib import Path
from typing import Union
import os

# # Ancora local do projeto
# PROJECT_ROOT = Path(__file__).resolve().parent

# # Define o caminho base de armazenamento (prioriza /home-ext/clovis se existir)
# EXTERNAL_BASE = Path("/home-ext/clovis/")
# ROOT_DIR = EXTERNAL_BASE if EXTERNAL_BASE.exists() else PROJECT_ROOT / "result"

# Lê da variável de ambiente ou usa /home-ext/clovis se existir, senão usa 'result/' local
env_path = os.getenv("OUTPUT_DIR")
if env_path:
    ROOT_DIR = Path(env_path)
elif Path("/home-ext/clovis").exists():
    ROOT_DIR = Path("/home-ext/clovis/")
else:
    ROOT_DIR = Path(__file__).resolve().parent / "result"


def get_path(
    variable_type: str = "qumodes",
    problem_type: str = "tsp",
    subfolder: Union[str, Path] = None, 
    is_result: bool = False
) -> Path:
    """
    Retorna o diretório base para saída de arquivos em '/home-ext/clovis/result/'.
    Organiza por tipo de problema ('tsp' ou 'vrp') e por categoria ('data' ou 'figures').
    """
    # Salva dentro de /home-ext/clovis/result/vrp/... (ou /home-ext/clovis/vrp/...)
    base_dir = ROOT_DIR / "result" / variable_type.lower() / problem_type.lower()
    
    category_folder = "data" if is_result else "figures"
    target_path = base_dir / category_folder

    if subfolder:
        target_path = target_path / subfolder

    target_path.mkdir(parents=True, exist_ok=True)
    return target_path


def get_images_path(variable_type: str = "qumodes", problem_type: str = "tsp", subfolder: Union[str, Path] = None) -> Path:
    """Atalho para obter a pasta de figuras/imagens."""
    return get_path(variable_type=variable_type, problem_type=problem_type, subfolder=subfolder, is_result=False)


def get_results_path(variable_type: str = "qumodes", problem_type: str = "tsp", subfolder: Union[str, Path] = None) -> Path:
    """Atalho para obter a pasta de dados/resultados JSON."""
    return get_path(variable_type=variable_type, problem_type=problem_type, subfolder=subfolder, is_result=True)


if __name__ == "__main__":
    print("==========================================================")
    print("      TESTANDO NOVO GERENCIADOR DE CAMINHOS (path.py)     ")
    print("==========================================================")
    print(f"Diretório Raiz Configurado: {ROOT_DIR}\n")

    tests = [
        ("Imagens TSP", lambda: get_images_path(problem_type="tsp")),
        ("Imagens VRP", lambda: get_images_path(problem_type="vrp")),
        ("Resultados Data TSP", lambda: get_results_path(problem_type="tsp")),
        ("Resultados Data VRP", lambda: get_results_path(problem_type="vrp")),
        ("Subpasta Personalizada TSP", lambda: get_results_path("run_01", problem_type="tsp")),
    ]

    for name, func in tests:
        try:
            path_created = func()
            print(f"[OK] {name}: {path_created}")
        except Exception as e:
            print(f"[ERRO] Falha ao criar diretório para '{name}': {e}")
            raise e

    print("==========================================================")
    print("Todos os caminhos do path.py foram criados e validados!")