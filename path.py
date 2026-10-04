from pathlib import Path
from typing import Optional, Union
import os


class PathManager:

    def __init__(
        self,
        variable_type: Union[str, Path] = "qumodes",
        sub_folder: Optional[Union[str, Path]] = None
    ):
        self.variable_type = str(variable_type).lower()
        self.sub_folder = str(sub_folder).lower() if sub_folder else None

        
    def _get_path(self, is_figure: bool = False, subdir: Optional[str] = None) -> Path:
        """
        Retorna o diretório base para saída de arquivos e cria automaticamente
        todas as pastas e subpastas informadas na hierarquia.
        """

        # Definição do diretório raiz base
        env_path = os.getenv("OUTPUT_DIR")
        if env_path:
            ROOT_DIR = Path(env_path)
        elif Path("/home-ext/clovis").exists():
            ROOT_DIR = Path("/home-ext/clovis/results")
        else:
            ROOT_DIR = Path(__file__).resolve().parent / "results"

        target_path = ROOT_DIR / self.variable_type

        if self.sub_folder:
            target_path = target_path / self.sub_folder

        if is_figure:
            target_path = target_path / "figures"
        elif subdir:
            target_path = target_path / str(subdir)

        # Garante a criação de todo o caminho de diretórios
        target_path.mkdir(parents=True, exist_ok=True)
        return target_path

    def get_file_path(self, filename: str, is_figure: bool = False,
                      subdir: Optional[str] = None) -> Path:
        """Caminho completo para um arquivo.

        subdir separa os artefatos por tipo dentro da pasta do experimento:
        "runs" (um JSON enxuto por execução), "arrays" (históricos pesados em
        .npz) e "figures" (via is_figure). O summary.csv e o banco de instâncias
        ficam na raiz, compartilhados por todas as execuções da varredura.
        """
        folder = self._get_path(is_figure=is_figure, subdir=subdir)
        return folder / filename


if __name__ == "__main__":
    print("==========================================================")
    print("      TESTANDO GERENCIADOR DE CAMINHOS COM SUBPASTAS      ")
    print("==========================================================")

    path_manager = PathManager(
        variable_type="qumodes", sub_folder="experimento_1/execucao_A"
    )

    caminho_log = path_manager.get_file_path("execucao.log")
    print(f"[OK] Caminho do log: {caminho_log}")

    caminho_figura = path_manager.get_file_path("convergencia.png", is_figure=True)
    print(f"[OK] Caminho da imagem: {caminho_figura}")