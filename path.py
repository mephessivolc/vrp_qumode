from pathlib import Path
from typing import Optional, Union
import os


class PathManager:

    def __init__(
        self,
        variable_type: Union[str, Path] = "qumodes",
        sub_folder: Optional[Union[str, Path]] = None,
    ):
        self.variable_type = str(variable_type).lower()
        self.sub_folder = str(sub_folder).lower() if sub_folder else None

        # Definição do diretório raiz base
        env_path = os.getenv("OUTPUT_DIR")
        if env_path:
            self.ROOT_DIR = Path(env_path)
        elif Path("/home-ext/clovis").exists():
            self.ROOT_DIR = Path("/home-ext/clovis/result")
        else:
            self.ROOT_DIR = Path(__file__).resolve().parent / "result"

    def _get_path(self, is_result: bool = True) -> Path:
        """
        Retorna o diretório base para saída de arquivos e cria automaticamente
        todas as pastas e subpastas informadas na hierarquia.
        """
        base_dir = self.ROOT_DIR / self.variable_type

        if self.sub_folder:
            base_dir = base_dir / Path(self.sub_folder)

        category_folder = "data" if is_result else "figures"
        target_path = base_dir / category_folder

        # Garante a criação de todo o caminho de diretórios
        target_path.mkdir(parents=True, exist_ok=True)
        return target_path

    def get_images_path(self) -> Path:
        """Retorna o caminho do diretório de figuras."""
        return self._get_path(is_result=False)

    def get_results_path(self) -> Path:
        """Retorna o caminho do diretório de dados (CSV/JSON/NPY)."""
        return self._get_path(is_result=True)

    def get_file_path(self, filename: str, is_result: bool = True) -> Path:
        """Retorna o caminho completo para salvar um arquivo específico."""
        folder = self._get_path(is_result=is_result)
        return folder / filename


if __name__ == "__main__":
    print("==========================================================")
    print("      TESTANDO GERENCIADOR DE CAMINHOS COM SUBPASTAS      ")
    print("==========================================================")

    path_manager = PathManager(
        variable_type="qumodes", sub_folder="experimento_1/execucao_A"
    )

    pasta_resultados = path_manager.get_results_path()
    print(f"[OK] Pasta de resultados criada: {pasta_resultados}")

    caminho_figura = path_manager.get_file_path(
        "convergencia.png", is_result=False
    )
    print(f"[OK] Caminho da imagem pronto: {caminho_figura}")