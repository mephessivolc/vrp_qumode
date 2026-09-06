from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class CVRPGraph:
    D: np.ndarray
    demands: np.ndarray
    coords: np.ndarray

    def __iter__(self):
        """Permite o desempacotamento direto: D, demands, coords = graph"""
        return iter((self.D, self.demands, self.coords))


class Graph:

    def __init__(
        self,
        C: int,
        V: int = 1,
        is_warm_start: bool = True,
        warm_start_coords: Optional[np.ndarray] = None,
        demand_range: Tuple[int, int] = (1, 5),
        seed: Optional[int] = 42,
    ):
        """
        Parâmetros:
            C: Número de cidades (excluindo o depósito no índice 0).
            V: Quantidade de veículos. Se V == 1, o problema é tratado como TSP.
            is_warm_start: Se True, gera topologia circular determinística.
            warm_start_coords: Matriz opcional de formato (C+1, 2) com coordenadas customizadas.
            seed: Semente aleatória para reprodutibilidade.
        """
        self.C = C
        self.V = V
        self.is_warm_start_used = is_warm_start
        self.demand_range = demand_range
        self.warm_start_coords = warm_start_coords
        self.seed = seed

        self._graph = self._generate()

    def _generate(self) -> CVRPGraph:
        if self.seed is not None:
            np.random.seed(self.seed)

        # 1. Construção da Topologia
        if self.is_warm_start_used:
            if self.warm_start_coords is not None:
                coords = np.asarray(self.warm_start_coords, dtype=np.float64)
                if coords.shape != (self.C + 1, 2):
                    raise ValueError(
                        f"warm_start_coords deve possuir dimensões ({self.C + 1}, 2), "
                        f"mas recebeu {coords.shape}."
                    )
            else:
                coords = np.zeros((self.C + 1, 2), dtype=np.float64)
                angles = np.linspace(0, 2 * np.pi, self.C, endpoint=False)
                coords[1:, 0] = 5.0 * np.cos(angles)
                coords[1:, 1] = 5.0 * np.sin(angles)
        else:
            coords = np.random.uniform(-10.0, 10.0, size=(self.C + 1, 2))
            coords[0] = [0.0, 0.0]

        # 2. Matriz de Distâncias Euclidianas
        diff = coords[:, None, :] - coords[None, :, :]
        D = np.sqrt(np.sum(diff**2, axis=-1))

        # 3. Lógica de Atribuição de Demandas (CVRP vs TSP)
        if self.V == 1:
            demands = np.zeros(self.C, dtype=np.float64)
        else:
            low, high = self.demand_range
            demands = np.random.randint(low, high + 1, size=self.C).astype(np.float64)

        return CVRPGraph(D=D, demands=demands, coords=coords)

    @property
    def graph(self) -> CVRPGraph:
        return self._graph