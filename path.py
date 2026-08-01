# path.py
from pathlib import Path
from typing import Union

# Ancora o projeto no diretório raiz onde este arquivo path.py reside
ROOT_DIR = Path(__file__).resolve().parent


def get_path(
    problem_type: str = "tsp", 
    subfolder: Union[str, Path] = None, 
    is_result: bool = False
) -> Path:
    """
    Retorna o diretório base para saída de arquivos na pasta 'result/'.
    Organiza por tipo de problema ('tsp' ou 'vrp') e por categoria ('data' ou 'figures').

    Parameters
    ----------
    problem_type : str, optional
        Tipo de problema ('tsp' ou 'vrp'). Padrão é 'tsp'.
    subfolder : Union[str, Path], optional
        Subpasta adicional caso necessário.
    is_result : bool, optional
        Se True, mapeia para a pasta 'data' (JSONs/CSVs). Caso contrário, para 'figures' (Imagens/Plots).

    Returns
    -------
    Path
        Caminho absoluto para o diretório solicitado.
    """
    base_dir = ROOT_DIR / "result" / problem_type.lower()
    
    category_folder = "data" if is_result else "figures"
    target_path = base_dir / category_folder

    if subfolder:
        target_path = target_path / subfolder

    target_path.mkdir(parents=True, exist_ok=True)
    return target_path


def get_images_path(subfolder: Union[str, Path] = None, problem_type: str = "tsp") -> Path:
    """Atalho para obter a pasta de figuras/imagens (ex: 'result/tsp/figures')."""
    return get_path(problem_type=problem_type, subfolder=subfolder, is_result=False)


def get_results_path(subfolder: Union[str, Path] = None, problem_type: str = "tsp") -> Path:
    """Atalho para obter a pasta de dados/resultados JSON (ex: 'result/tsp/data')."""
    return get_path(problem_type=problem_type, subfolder=subfolder, is_result=True)


if __name__ == "__main__":
    print("==========================================================")
    print("      TESTANDO NOVO GERENCIADOR DE CAMINHOS (path.py)     ")
    print("==========================================================")

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