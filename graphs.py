from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CVRPGraph:
    D: np.ndarray
    demands: np.ndarray  # Demandas apenas das cidades (tamanho C)
    coords: np.ndarray

    @property
    def full_demands(self) -> np.ndarray:
        """Retorna o vetor de demandas completo incluindo o depósito (índice 0 com demanda 0.0)."""
        return np.insert(self.demands, 0, 0.0)

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

        if self.seed is not None:
            np.random.seed(self.seed)

        self._graph = self._generate()

    def _generate(self) -> CVRPGraph:
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
        self.D = np.sqrt(np.sum(diff**2, axis=-1))

        # 3. Lógica de Atribuição de Demandas (CVRP vs TSP)
        if self.V == 1:
            demands = np.zeros(self.C, dtype=np.float64)
        else:
            low, high = self.demand_range
            demands = np.random.randint(low, high + 1, size=self.C).astype(np.float64)

        return CVRPGraph(D=self.D, demands=demands, coords=coords)

    @property
    def graph(self) -> CVRPGraph:
        return self._graph

    @property
    def dist_matrix(self) -> np.ndarray:
        return self.D

    @property
    def full_demands(self) -> np.ndarray:
        """Atalho para acessar as demandas completas (depósito + cidades)."""
        return self._graph.full_demands

    def plot_original_graph(self, save_path: Union[str, Path]) -> None:
        """
        Gera e salva o gráfico da topologia original contendo o depósito, cidades e demandas.
        
        Parâmetros:
            save_path: Caminho completo para salvar a imagem fornecido pelo PathManager.
        """
        save_path = Path(save_path)
        coords = self._graph.coords
        full_demands = self.full_demands
        n_nodes = len(coords)

        plt.figure(figsize=(8, 6))

        # Arestas do grafo completo em segundo plano
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                plt.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color="gray",
                    linestyle="--",
                    alpha=0.3,
                    zorder=1,
                )

        # Depósito (Índice 0)
        plt.scatter(
            coords[0, 0],
            coords[0, 1],
            c="red",
            s=200,
            marker="s",
            label="Depósito (0)",
            zorder=3,
        )

        # Cidades (Índices 1 a C)
        for i in range(1, n_nodes):
            plt.scatter(
                coords[i, 0],
                coords[i, 1],
                c="blue",
                s=120,
                marker="o",
                zorder=3,
            )
            label_text = f"C{i}\n(d={full_demands[i]:.0f})" if self.V > 1 else f"C{i}"
            plt.annotate(
                label_text,
                (coords[i, 0], coords[i, 1]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                weight="bold",
            )

        plt.title(f"Grafo Original - {self.C} Cidades {self.V} Veículos", fontsize=12)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right")
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_solution_graph(
        self,
        routes: Union[List[int], Dict[int, List[int]]],
        save_path: Union[str, Path],
        title_prefix: str = "Rota Encontrada",
    ) -> None:
        """
        Gera e salva o gráfico do caminho percorrido pelos veículos a partir de uma solução.
        
        Parâmetros:
            routes: Lista de nós (TSP) ou dicionário {veiculo_id: [sub-rota]} (VRP).
            save_path: Caminho completo para salvar a imagem fornecido pelo PathManager.
            title_prefix: Prefixo para o título da figura.
        """
        save_path = Path(save_path)
        coords = self._graph.coords
        full_demands = self.full_demands
        n_nodes = len(coords)

        # Normalização para formato de dicionário de veículos
        routes_dict = {1: routes} if isinstance(routes, list) else routes

        plt.figure(figsize=(8, 6))

        # Depósito
        plt.scatter(
            coords[0, 0],
            coords[0, 1],
            c="red",
            s=200,
            marker="s",
            label="Depósito (0)",
            zorder=4,
        )

        # Cidades
        for i in range(1, n_nodes):
            plt.scatter(
                coords[i, 0],
                coords[i, 1],
                c="blue",
                s=120,
                marker="o",
                zorder=4,
            )
            label_text = f"C{i}\n(d={full_demands[i]:.0f})" if self.V > 1 else f"C{i}"
            plt.annotate(
                label_text,
                (coords[i, 0], coords[i, 1]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                weight="bold",
            )

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(routes_dict), 10)))

        # Plota os trajetos com setas direcionais por veículo
        for idx, (v_id, path_nodes) in enumerate(routes_dict.items()):
            color = colors[idx % len(colors)]

            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                start_x, start_y = coords[u, 0], coords[u, 1]
                end_x, end_y = coords[v, 0], coords[v, 1]

                plt.annotate(
                    "",
                    xy=(end_x, end_y),
                    xytext=(start_x, start_y),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color,
                        lw=2,
                        mutation_scale=15,
                        shrinkA=8,
                        shrinkB=8,
                    ),
                    zorder=3,
                )

            plt.plot([], [], color=color, lw=2, label=f"Veículo {v_id}")

        plt.title(f"{title_prefix} - {self.C} Cidades {self.V} Veículos", fontsize=12)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right")
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        plt.close()