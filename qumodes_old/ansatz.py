from dataclasses import dataclass
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate


@dataclass
class CircuitConfig:
    num_qumodes: int
    num_layers: int = 2


class ContinuousVariableAnsatz:
    """
    Constrói o circuito quântico variacional U(\theta) parametrizado por camadas
    para a simulação em Variáveis Contínuas no Strawberry Fields.
    """

    def __init__(self, config: CircuitConfig):
        self.N = config.num_qumodes
        self.layers = config.num_layers
        self.num_params_per_layer = self._calculate_params_per_layer()
        self.total_params = self.num_params_per_layer * self.layers

    def _calculate_params_per_layer(self) -> int:
        # Parâmetros por qumode: Squeezing (2), Rotação (1), Deslocamento (2), Kerr (1)
        single_mode_params = 6 * self.N
        # Parâmetros de emaranhamento por par adjacente: Beam Splitter (2)
        entangling_params = 2 * (self.N - 1)
        return single_mode_params + entangling_params

    def generate_initial_theta(self, seed: int = 42) -> np.ndarray:
        """Gera um vetor de parâmetros \theta inicial de forma estocástica e controlada."""
        np.random.seed(seed)
        # Inicializa parâmetros próximos a zero para evitar instabilidade inicial de Fock
        return np.random.normal(loc=0.0, scale=0.05, size=self.total_params)

    def build_program(self, theta: np.ndarray) -> sf.Program:
        """
        Mapeia um vetor unidimensional de parâmetros \theta em um circuito sf.Program.
        """
        if len(theta) != self.total_params:
            raise ValueError(
                f"Vetor de parâmetros incompatível. Esperado: {self.total_params}, recebido: {len(theta)}"
            )

        prog = sf.Program(self.N)
        idx = 0

        with prog.context as q:
            for _ in range(self.layers):
                # 1. Camada de Squeezing e Rotação Inicial
                for i in range(self.N):
                    Sgate(theta[idx], theta[idx + 1]) | q[i]
                    Rgate(theta[idx + 2]) | q[i]
                    idx += 3

                # 2. Camada de Emaranhamento (Beam Splitters em cadeia)
                for i in range(self.N - 1):
                    BSgate(theta[idx], theta[idx + 1]) | (q[i], q[i + 1])
                    idx += 2

                # 3. Camada de Deslocamento de Fase e Não-Gaussianidade (Kerr)
                for i in range(self.N):
                    Dgate(theta[idx], theta[idx + 1]) | q[i]
                    Kgate(theta[idx + 2]) | q[i]
                    idx += 3

        return prog