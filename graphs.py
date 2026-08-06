# graphs.py
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Importa o gerenciador de saída para direcionar figuras para 'result/'
from logger import ExperimentLogger


class GraphBuilder:
    """
    Construtor e visualizador de grafos para problemas de otimização de rotas (TSP e VRP).
    
    Permite inicialização por:
      1. Geradores Automáticos via `graph_type` ("euclidean", "circle", "grid", "clustered", "random")
      2. Coordenadas Espaciais 2D/3D (`coords=[(x1,y1), ...]`)
      3. Matriz de distâncias pronta (`matrix=[...]`)
    
    Todos os parâmetros são opcionais com padrões seguros para simulações VQE.
    """
    def __init__(
        self, 
        n: Optional[int] = None, 
        seed: int = 42, 
        graph_type: str = "random",
        coords: Optional[Union[np.ndarray, List[Tuple[float, float]]]] = None,
        matrix: Optional[Union[np.ndarray, List[List[float]]]] = None,
        logger: Optional[ExperimentLogger] = None
    ):
        self.seed = seed
        self.logger = logger if logger is not None else ExperimentLogger()
        self.graph_type = graph_type
        self.coords = None

        if matrix is not None:
            # 1. Modo Matriz Direta
            self.matrix = np.array(matrix, dtype=np.float32)
            if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
                raise ValueError("A matriz de distâncias deve ser quadrada (N x N).")
            matrix_n = self.matrix.shape[0]
            if n is not None and n != matrix_n:
                raise ValueError(f"Conflito de dimensão: 'n' ({n}) != dimensão da matriz ({matrix_n}).")
            self.n = matrix_n

        elif coords is not None:
            # 2. Modo Coordenadas 2D/3D
            self.coords = np.array(coords, dtype=np.float32)
            coords_n = self.coords.shape[0]
            if n is not None and n != coords_n:
                raise ValueError(f"Conflito de dimensão: 'n' ({n}) != número de coordenadas ({coords_n}).")
            self.n = coords_n
            self.matrix = self._build_matrix_from_coords(self.coords)

        else:
            # 3. Modo Gerador de Topologias por Tipo (Padrão)
            self.n = n if n is not None else 3
            self.matrix = self._generate_matrix_by_type()

    def _build_matrix_from_coords(self, coords: np.ndarray) -> np.ndarray:
        """Calcula matriz de distâncias euclidianas a partir de coordenadas 2D/3D."""
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        return np.round(dist_matrix, 2)

    def _generate_matrix_by_type(self) -> np.ndarray:
        """Gera matrizes e posições baseadas no modelo topológico selecionado."""
        np.random.seed(self.seed)

        if self.graph_type == "euclidean":
            # Cidades espalhadas aleatoriamente num plano 2D [0, 100] x [0, 100] (Estilo TSPLIB)
            self.coords = np.random.uniform(10, 90, size=(self.n, 2))
            return self._build_matrix_from_coords(self.coords)

        elif self.graph_type == "circle":
            # Cidades dispostas num círculo regular (Polígono regular)
            angles = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
            self.coords = np.column_stack((50 + 35 * np.cos(angles), 50 + 35 * np.sin(angles)))
            return self._build_matrix_from_coords(self.coords)

        elif self.graph_type == "grid":
            # Cidades dispostas em uma grade (Grid 2D)
            side = int(np.ceil(np.sqrt(self.n)))
            grid_points = []
            for i in range(self.n):
                x = (i % side) * 25.0 + 10.0
                y = (i // side) * 25.0 + 10.0
                grid_points.append((x, y))
            self.coords = np.array(grid_points)
            return self._build_matrix_from_coords(self.coords)

        elif self.graph_type == "clustered":
            # Cidades organizadas em agrupamentos (Clusters geográficos)
            num_clusters = max(2, self.n // 2)
            centers = np.random.uniform(20, 80, size=(num_clusters, 2))
            cluster_pts = []
            for i in range(self.n):
                center = centers[i % num_clusters]
                offset = np.random.normal(0, 4, size=2)
                cluster_pts.append(center + offset)
            self.coords = np.array(cluster_pts)
            return self._build_matrix_from_coords(self.coords)

        else:
            # "random": Matriz de adjacência simétrica direta (Modo legado)
            adj = np.random.uniform(1.0, 10.0, size=(self.n, self.n))
            adj = (adj + adj.T) / 2.0
            np.fill_diagonal(adj, 0.0)
            return np.round(adj, 2)

    def _get_layout(self, G: nx.Graph) -> Dict:
        """Utiliza as coordenadas reais no mapa (se existirem) ou spring_layout."""
        if self.coords is not None:
            return {i: (self.coords[i, 0], self.coords[i, 1]) for i in range(self.n)}
        return nx.spring_layout(G, seed=self.seed)

    def _convert_vector_to_route(self, vector: Union[List, tuple, np.ndarray]) -> List[int]:
        if isinstance(vector, (tuple, np.ndarray)):
            vector = list(vector)

        route = []
        if len(vector) == self.n ** 2:
            matrix_form = np.array(vector).reshape((self.n, self.n))
            for step in range(self.n):
                city = int(np.argmax(matrix_form[:, step])) % self.n
                route.append(city)
        elif len(vector) == self.n:
            if any(isinstance(x, float) for x in vector):
                route = list(np.argsort(vector))
            else:
                route = [int(c) % self.n for c in vector]
        else:
            route = [int(c) % self.n for c in vector]

        valid_route = []
        for node in route:
            if node not in valid_route and 0 <= node < self.n:
                valid_route.append(node)
        
        missing = [i for i in range(self.n) if i not in valid_route]
        valid_route.extend(missing)

        return valid_route[:self.n]

    def plot_original_graph(self, prefix: str = "graph", problem_type: str = "TSP") -> Path:
        out_dir = Path(self.logger.get_figures_dir(problem_type))

        G = nx.Graph()
        for i in range(self.n):
            G.add_node(i)
            for j in range(i + 1, self.n):
                G.add_edge(i, j, weight=self.matrix[i, j])

        pos = self._get_layout(G)

        plt.figure(figsize=(7, 6))
        node_colors = ['gold' if node == 0 else 'lightblue' for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, alpha=0.7)

        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)

        plt.title(f"Grafo Original (N={self.n} | Topologia: {self.graph_type.upper()})", fontsize=12)
        plt.axis('off')

        orig_path = out_dir / f"{prefix}_original.png"
        plt.savefig(orig_path, dpi=300, bbox_inches='tight')
        plt.close()

        return orig_path

    def plot_tsp_route(self, solution_vector: Union[List, tuple, np.ndarray], prefix: str = "tsp") -> Path:
        out_dir = Path(self.logger.get_figures_dir("TSP"))
        route = self._convert_vector_to_route(solution_vector)
        full_cycle = route + [route[0]]

        G_base = nx.Graph()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                G_base.add_edge(i, j, weight=self.matrix[i, j])

        pos = self._get_layout(G_base)

        G_directed = nx.DiGraph()
        for i in range(self.n):
            G_directed.add_node(i)

        route_edges = []
        for i in range(len(full_cycle) - 1):
            u, v = full_cycle[i], full_cycle[i + 1]
            G_directed.add_edge(u, v)
            route_edges.append((u, v))

        plt.figure(figsize=(7, 6))
        nx.draw_networkx_edges(G_base, pos, edge_color='lightgray', width=1.0, style='dashed', alpha=0.5)

        node_colors = ['gold' if node == 0 else 'lightgreen' for node in G_base.nodes()]
        nx.draw_networkx_nodes(G_base, pos, node_color=node_colors, node_size=750)
        nx.draw_networkx_labels(G_base, pos, font_size=12, font_weight='bold')

        nx.draw_networkx_edges(
            G_directed, pos,
            edgelist=route_edges,
            edge_color='crimson',
            width=3.0,
            arrowstyle='->',
            arrowsize=20
        )

        labels = nx.get_edge_attributes(G_base, 'weight')
        nx.draw_networkx_edge_labels(G_base, pos, edge_labels=labels, font_size=10)

        plt.title(f"Trajeto Destacado do TSP: {' -> '.join(map(str, full_cycle))}", fontsize=11)
        plt.axis('off')

        route_path = out_dir / f"{prefix}_route.png"
        plt.savefig(route_path, dpi=300, bbox_inches='tight')
        plt.close()

        return route_path

    def plot_vrp_routes(self, routes: Dict[int, List[int]], prefix: str = "vrp") -> Path:
        out_dir = Path(self.logger.get_figures_dir("VRP"))

        G_base = nx.Graph()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                G_base.add_edge(i, j, weight=self.matrix[i, j])

        pos = self._get_layout(G_base)
        color_palette = ['#E63946', '#1D3557', '#2A9D8F', '#F4A261', '#9C27B0', '#3F51B5']

        plt.figure(figsize=(8, 7))
        nx.draw_networkx_edges(G_base, pos, edge_color='lightgray', width=1.0, style='dashed', alpha=0.4)

        node_colors = ['gold' if node == 0 else 'lightblue' for node in G_base.nodes()]
        nx.draw_networkx_nodes(G_base, pos, node_color=node_colors, node_size=800)
        nx.draw_networkx_labels(G_base, pos, font_size=12, font_weight='bold')

        legend_handles = []
        for idx, (v_id, route) in enumerate(routes.items()):
            color = color_palette[(idx) % len(color_palette)]
            sanitized_route = [int(node) % self.n for node in route]

            G_v = nx.DiGraph()
            v_edges = []
            for i in range(len(sanitized_route) - 1):
                u, v = sanitized_route[i], sanitized_route[i + 1]
                G_v.add_edge(u, v)
                v_edges.append((u, v))

            rad = 0.1 * (idx + 1)
            nx.draw_networkx_edges(
                G_v, pos,
                edgelist=v_edges,
                edge_color=color,
                width=2.5,
                arrowstyle='->',
                arrowsize=18,
                connectionstyle=f"arc3,rad={rad}"
            )
            
            legend_handles.append(plt.Line2D([0], [0], color=color, lw=2.5, label=f"Veículo {v_id}: {sanitized_route}"))

        labels = nx.get_edge_attributes(G_base, 'weight')
        nx.draw_networkx_edge_labels(G_base, pos, edge_labels=labels, font_size=9)

        plt.title(f"Solução VRP Multi-Veículos (N={self.n} Nó Depósito: 0)", fontsize=12)
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        plt.axis('off')

        vrp_path = out_dir / f"{prefix}_routes.png"
        plt.savefig(vrp_path, dpi=300, bbox_inches='tight')
        plt.close()

        return vrp_path

    def plot_graph_and_route(self, solution_vector = None, prefix: str = "graph") -> Tuple[Path, Union[Path, None]]:
        prob_type = "VRP" if isinstance(solution_vector, dict) else "TSP"
        orig_path = self.plot_original_graph(prefix=prefix, problem_type=prob_type)
        route_path = None

        if solution_vector is not None:
            if isinstance(solution_vector, dict):
                route_path = self.plot_vrp_routes(routes=solution_vector, prefix=prefix)
            else:
                route_path = self.plot_tsp_route(solution_vector=solution_vector, prefix=prefix)

        return orig_path, route_path

    def draw(self, filename: str = "graph.png") -> Path:
        prefix = filename.replace(".png", "")
        return self.plot_original_graph(prefix=prefix)