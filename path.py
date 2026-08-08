# path.py
from pathlib import Path
from typing import Union
import os

# Lê da variável de ambiente ou usa /home-ext/clovis se existir, senão usa 'result/' local
env_path = os.getenv("OUTPUT_DIR")
if env_path:
    ROOT_DIR = Path(env_path)
elif Path("/home-ext/clovis").exists():
    ROOT_DIR = Path("/home-ext/clovis/")
else:
    ROOT_DIR = Path(__file__).resolve().parent / "result"


def get_path(
    variable_type: Union[str, Path] = "qumodes",
    problem_type: str = "tsp",
    sub_folder: Union[str, Path] = None, 
    is_result: bool = False
) -> Path:
    """
    Retorna o diretório base para saída de arquivos e CRIA automaticamente
    todas as pastas e subpastas informadas (ex: 'pasta1/pasta2').
    """
    # Converter para Path garante que barras '/' ou '\' virem estrutura de diretórios
    var_path = Path(str(variable_type).lower())
    
    base_dir = ROOT_DIR / "result" / var_path / str(problem_type).lower()

    
    # Se sub_folder for 'pasta1/pasta2', ela é anexada corretamente ao caminho
    if sub_folder:
        base_dir = base_dir / Path(str(sub_folder).lower())
    
    category_folder = "data" if is_result else "figures"
    target_path = base_dir / category_folder

    # parents=True força a criação de TODAS as pastas pai na árvore (pasta1, pasta2, etc)
    target_path.mkdir(parents=True, exist_ok=True)
    return target_path


def get_images_path(variable_type: Union[str, Path] = "qumodes", problem_type: str = "tsp", sub_folder: Union[str, Path] = None) -> Path:
    """Atalho para obter a pasta de figuras/imagens."""
    return get_path(variable_type=variable_type, problem_type=problem_type, sub_folder=sub_folder, is_result=False)


def get_results_path(variable_type: Union[str, Path] = "qumodes", problem_type: str = "tsp", sub_folder: Union[str, Path] = None) -> Path:
    """Atalho para obter a pasta de dados/resultados JSON/CSV."""
    return get_path(variable_type=variable_type, problem_type=problem_type, sub_folder=sub_folder, is_result=True)


def get_file_path(
    filename: str,
    variable_type: Union[str, Path] = "qumodes",
    problem_type: str = "tsp",
    sub_folder: Union[str, Path] = None,
    is_result: bool = True
) -> Path:
    """
    Função utilitária para quando você quer obter o caminho completo do ARQUIVO.
    Cria todas as pastas pai e retorna o caminho com o nome do arquivo.
    """
    folder = get_path(variable_type=variable_type, problem_type=problem_type, sub_folder=sub_folder, is_result=is_result)
    return folder / filename


if __name__ == "__main__":
    print("==========================================================")
    print("      TESTANDO GERENCIADOR DE CAMINHOS COM SUBPASTAS      ")
    print("==========================================================")

    # Teste enviando "pasta1/pasta2" no sub_folder
    caminho = get_results_path(problem_type="tsp", sub_folder="pasta1/pasta2")
    print(f"[OK] Pasta criada: {caminho}")

    # Teste salvando arquivo direto com subpasta
    arquivo = get_file_path("resultado.csv", problem_type="tsp", sub_folder="experimento_1/execucao_A")
    print(f"[OK] Caminho do arquivo pronto para uso: {arquivo}")