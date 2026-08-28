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
    Construtor e visualizador de grafos para problemas de otimização de rotas (TSP, VRP e CVRP).
    
    Permite inicialização por:
      1. Geradores Automáticos via `graph_type`:
         - Topologias Padrão: "euclidean", "circle", "grid", "clustered", "random"
         - Modo Warm-Start: "warm_start" ou "warm_start_<topologia>" (ex: "warm_start_circle")
      2. Coordenadas Espaciais 2D/3D (`coords=[(x1,y1), ...]`)
      3. Matriz de distâncias pronta (`matrix=[...]`)
      4. Atribuição de demandas de carga para CVRP (`demands=[q0, q1, ...]`)
    """
    def __init__(
        self, 
        n: Optional[int] = None, 
        seed: int = 42, 
        graph_type: str = "random",
        coords: Optional[Union[np.ndarray, List[Tuple[float, float]]]] = None,
        matrix: Optional[Union[np.ndarray, List[List[float]]]] = None,
        demands: Optional[Union[np.ndarray, List[float]]] = None,
        demand_range: Tuple[float, float] = (1.0, 5.0),
        num_vehicles: int = 2,
        vehicle_capacity: Union[float, int, List[float], np.ndarray] = 10.0,
        logger: Optional[ExperimentLogger] = None,
        variable_type_path: str = "QUMODES",
        sub_folder: Union[str, None] = None
    ):
        self.seed = seed
        self.logger = logger if logger is not None else ExperimentLogger()
        self.graph_type = graph_type.lower()
        self.coords = None
        self.variable_type_path = variable_type_path
        self.sub_folder = sub_folder
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity

        # Atributos de armazenamento de soluções Heurísticas / Warm-Start
        self.warm_start_tsp: Optional[List[int]] = None
        self.warm_start_vrp: Optional[Dict[int, List[int]]] = None
        self.warm_start_quadratures: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # 1. Definir Matriz e N
        if matrix is not None:
            self.matrix = np.array(matrix, dtype=np.float32)
            if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
                raise ValueError("A matriz de distâncias deve ser quadrada (N x N).")
            matrix_n = self.matrix.shape[0]
            if n is not None and n != matrix_n:
                raise ValueError(f"Conflito de dimensão: 'n' ({n}) != dimensão da matriz ({matrix_n}).")
            self.n = matrix_n

        elif coords is not None:
            self.coords = np.array(coords, dtype=np.float32)
            coords_n = self.coords.shape[0]
            if n is not None and n != coords_n:
                raise ValueError(f"Conflito de dimensão: 'n' ({n}) != número de coordenadas ({coords_n}).")
            self.n = coords_n
            self.matrix = self._build_matrix_from_coords(self.coords)

        else:
            self.n = n if n is not None else 3
            self.matrix = self._generate_matrix_by_type()

        # 2. Configurar Demandas dos Vértices para CVRP
        if demands is not None:
            self.demands = np.array(demands, dtype=np.float32)
            if len(self.demands) != self.n:
                raise ValueError(f"O vetor de demandas ({len(self.demands)}) deve ter o mesmo tamanho N={self.n}.")
            self.demands[0] = 0.0  # Garante demanda nula para o Depósito
        else:
            np.random.seed(self.seed)
            gen_demands = np.random.uniform(demand_range[0], demand_range[1], size=self.n)
            gen_demands = np.round(gen_demands, 1)
            gen_demands[0] = 0.0  # Depósito (Nó 0)
            self.demands = gen_demands.astype(np.float32)

        # 3. Inicialização Automática por Warm-Start se especificado no `graph_type`
        if "warm_start" in self.graph_type:
            self.build_warm_start()

    def _build_matrix_from_coords(self, coords: np.ndarray) -> np.ndarray:
        """Calcula matriz de distâncias euclidianas a partir de coordenadas 2D/3D."""
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        return np.round(dist_matrix, 2)

    def _generate_matrix_by_type(self) -> np.ndarray:
        """Gera matrizes e posições baseadas no modelo topológico selecionado."""
        np.random.seed(self.seed)

        # Extrai a topologia base caso seja especificado warm_start (ex: "warm_start_circle" -> "circle")
        base_type = self.graph_type.replace("warm_start_", "").replace("warm_start", "").strip("_")
        if not base_type:
            base_type = "euclidean"

        if base_type == "euclidean":
            self.coords = np.random.uniform(10, 90, size=(self.n, 2))
            return self._build_matrix_from_coords(self.coords)

        elif base_type == "circle":
            angles = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
            self.coords = np.column_stack((50 + 35 * np.cos(angles), 50 + 35 * np.sin(angles)))
            return self._build_matrix_from_coords(self.coords)

        elif base_type == "grid":
            side = int(np.ceil(np.sqrt(self.n)))
            grid_points = []
            for i in range(self.n):
                x = (i % side) * 25.0 + 10.0
                y = (i // side) * 25.0 + 10.0
                grid_points.append((x, y))
            self.coords = np.array(grid_points)
            return self._build_matrix_from_coords(self.coords)

        elif base_type == "clustered":
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
            adj = np.random.uniform(1.0, 10.0, size=(self.n, self.n))
            adj = (adj + adj.T) / 2.0
            np.fill_diagonal(adj, 0.0)
            return np.round(adj, 2)

    # -------------------------------------------------------------------------
    # MÉTODOS DE GERACÃO E INICIALIZAÇÃO HEURÍSTICA (WARM-START)
    # -------------------------------------------------------------------------

    def build_warm_start(
        self, 
        num_vehicles: Optional[int] = None, 
        vehicle_capacity: Optional[Union[float, int, List[float], np.ndarray]] = None
    ) -> Dict:
        """
        Executa os algoritmos heurísticos para construir os dados de Warm-Start
        e popula os atributos da classe.
        """
        n_veh = num_vehicles if num_vehicles is not None else self.num_vehicles
        cap_veh = vehicle_capacity if vehicle_capacity is not None else self.vehicle_capacity

        self.warm_start_tsp = self.generate_nearest_neighbor_tsp()
        self.warm_start_vrp = self.generate_greedy_vrp(num_vehicles=n_veh, vehicle_capacity=cap_veh)
        self.warm_start_quadratures = self.get_warm_start_quadratures(num_vehicles=n_veh, vehicle_capacity=cap_veh)

        return {
            "tsp_route": self.warm_start_tsp,
            "vrp_routes": self.warm_start_vrp,
            "quadratures": self.warm_start_quadratures
        }

    def generate_nearest_neighbor_tsp(self, start_node: int = 0) -> List[int]:
        """Gera uma rota TSP heurística via Vizinho Mais Próximo."""
        unvisited = set(range(self.n))
        unvisited.remove(start_node)
        route = [start_node]
        current = start_node

        while unvisited:
            next_node = min(unvisited, key=lambda node: self.matrix[current, node])
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        return route

    def generate_greedy_vrp(
        self, 
        num_vehicles: int, 
        vehicle_capacity: Union[float, int, List[float], np.ndarray]
    ) -> Dict[int, List[int]]:
        """Gera rotas heurísticas para o CVRP respeitando limites de capacidade."""
        if isinstance(vehicle_capacity, (float, int)):
            caps = [float(vehicle_capacity)] * num_vehicles
        else:
            caps = [float(c) for c in vehicle_capacity]

        unvisited = set(range(1, self.n))
        routes = {v + 1: [0] for v in range(num_vehicles)}

        for v_idx in range(num_vehicles):
            v_id = v_idx + 1
            current_cap = caps[v_idx]
            current_node = 0

            while unvisited:
                feasible_nodes = [
                    node for node in unvisited 
                    if self.demands[node] <= current_cap
                ]
                if not feasible_nodes:
                    break

                next_node = min(feasible_nodes, key=lambda node: self.matrix[current_node, node])
                routes[v_id].append(next_node)
                current_cap -= self.demands[next_node]
                unvisited.remove(next_node)
                current_node = next_node

            routes[v_id].append(0)

        if unvisited:
            for idx, node in enumerate(sorted(unvisited)):
                v_id = (idx % num_vehicles) + 1
                routes[v_id].insert(-1, node)

        return routes

    def get_warm_start_quadratures(
        self, 
        num_vehicles: int, 
        vehicle_capacity: Union[float, int, List[float], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mapeia as rotas heurísticas CVRP em alvos de quadraturas contínuas (x, p)."""
        routes = self.generate_greedy_vrp(num_vehicles=num_vehicles, vehicle_capacity=vehicle_capacity)
        
        x_targets = np.zeros(self.n - 1, dtype=np.float32)
        p_targets = np.zeros(self.n - 1, dtype=np.float32)

        for v_id, route in routes.items():
            free_cities = [node for node in route if node != 0]
            for step_idx, city_node in enumerate(free_cities):
                city_idx = city_node - 1
                x_targets[city_idx] = float(step_idx + 1)
                p_targets[city_idx] = float(v_id)

        return x_targets, p_targets

    # -------------------------------------------------------------------------
    # MÉTODOS DE VISUALIZAÇÃO E PLOTAGEM
    # -------------------------------------------------------------------------

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

    def _get_node_labels(self) -> Dict[int, str]:
        """Gera rótulos com ID do nó e sua respectiva demanda."""
        labels = {}
        for i in range(self.n):
            if i == 0:
                labels[i] = "0\n(Depot)"
            else:
                q = self.demands[i]
                q_str = f"{int(q)}" if q.is_integer() else f"{q:.1f}"
                labels[i] = f"{i}\n(q={q_str})"
        return labels

    def plot_original_graph(
        self, 
        prefix: str = "graph", 
        problem_type: str = "VRP",
        sub_folder: Union[str, None] = None
    ) -> Path:
        s_folder = sub_folder if sub_folder is not None else self.sub_folder
        out_dir = Path(self.logger.get_figures_dir(
            variable_type=self.variable_type_path, 
            problem_type=problem_type, 
            sub_folder=s_folder
        ))

        G = nx.Graph()
        for i in range(self.n):
            G.add_node(i)
            for j in range(i + 1, self.n):
                G.add_edge(i, j, weight=self.matrix[i, j])

        pos = self._get_layout(G)

        plt.figure(figsize=(7, 6))
        node_colors = ['gold' if node == 0 else 'lightblue' for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900)
        nx.draw_networkx_labels(G, pos, labels=self._get_node_labels(), font_size=9, font_weight='bold')
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, alpha=0.7)

        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=9)

        plt.title(f"Grafo Original CVRP (N={self.n} | Topologia: {self.graph_type.upper()})", fontsize=11)
        plt.axis('off')

        orig_path = out_dir / f"{prefix}_original.png"
        plt.savefig(orig_path, dpi=300, bbox_inches='tight')
        plt.close()

        return orig_path

    def plot_tsp_route(
        self, 
        solution_vector: Union[List, tuple, np.ndarray], 
        prefix: str = "tsp",
        sub_folder: Union[str, None] = None
    ) -> Path:
        s_folder = sub_folder if sub_folder is not None else self.sub_folder
        out_dir = Path(self.logger.get_figures_dir(
            variable_type=self.variable_type_path, 
            problem_type="TSP", 
            sub_folder=s_folder
        ))

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
        nx.draw_networkx_nodes(G_base, pos, node_color=node_colors, node_size=900)
        nx.draw_networkx_labels(G_base, pos, labels=self._get_node_labels(), font_size=9, font_weight='bold')

        nx.draw_networkx_edges(
            G_directed, pos,
            edgelist=route_edges,
            edge_color='crimson',
            width=3.0,
            arrowstyle='->',
            arrowsize=20
        )

        labels = nx.get_edge_attributes(G_base, 'weight')
        nx.draw_networkx_edge_labels(G_base, pos, edge_labels=labels, font_size=9)

        plt.title(f"Trajeto TSP: {' -> '.join(map(str, full_cycle))}", fontsize=11)
        plt.axis('off')

        route_path = out_dir / f"{prefix}_route.png"
        plt.savefig(route_path, dpi=300, bbox_inches='tight')
        plt.close()

        return route_path

    def plot_vrp_routes(
        self, 
        routes: Dict[int, List[int]], 
        prefix: str = "vrp",
        sub_folder: Union[str, None] = None
    ) -> Path:
        s_folder = sub_folder if sub_folder is not None else self.sub_folder
        out_dir = Path(self.logger.get_figures_dir(
            variable_type=self.variable_type_path, 
            problem_type="VRP", 
            sub_folder=s_folder
        ))

        G_base = nx.Graph()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                G_base.add_edge(i, j, weight=self.matrix[i, j])

        pos = self._get_layout(G_base)
        color_palette = ['#E63946', '#1D3557', '#2A9D8F', '#F4A261', '#9C27B0', '#3F51B5']

        plt.figure(figsize=(8, 7))
        nx.draw_networkx_edges(G_base, pos, edge_color='lightgray', width=1.0, style='dashed', alpha=0.4)

        node_colors = ['gold' if node == 0 else 'lightblue' for node in G_base.nodes()]
        nx.draw_networkx_nodes(G_base, pos, node_color=node_colors, node_size=900)
        nx.draw_networkx_labels(G_base, pos, labels=self._get_node_labels(), font_size=9, font_weight='bold')

        legend_handles = []
        for idx, (v_id, route) in enumerate(routes.items()):
            color = color_palette[(idx) % len(color_palette)]
            sanitized_route = [int(node) % self.n for node in route]

            v_load = sum(self.demands[node] for node in sanitized_route)
            v_load_str = f"{int(v_load)}" if v_load.is_integer() else f"{v_load:.1f}"

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
            
            legend_handles.append(plt.Line2D(
                [0], [0], color=color, lw=2.5, 
                label=f"Veículo {v_id} (Carga={v_load_str}): {sanitized_route}"
            ))

        labels = nx.get_edge_attributes(G_base, 'weight')
        nx.draw_networkx_edge_labels(G_base, pos, edge_labels=labels, font_size=8)

        plt.title(f"Solução CVRP Multi-Veículos (N={self.n} | Depósito: 0)", fontsize=12)
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
        plt.axis('off')

        vrp_path = out_dir / f"{prefix}_routes.png"
        plt.savefig(vrp_path, dpi=300, bbox_inches='tight')
        plt.close()

        return vrp_path

    def plot_graph_and_route(
        self, 
        solution_vector = None, 
        prefix: str = "graph",
        sub_folder: Union[str, None] = None
    ) -> Tuple[Path, Union[Path, None]]:
        
        prob_type = "VRP" if isinstance(solution_vector, dict) else "TSP"
        orig_path = self.plot_original_graph(prefix=prefix, problem_type=prob_type, sub_folder=sub_folder)
        route_path = None

        if solution_vector is not None:
            if isinstance(solution_vector, dict):
                route_path = self.plot_vrp_routes(routes=solution_vector, prefix=prefix, sub_folder=sub_folder)
            else:
                route_path = self.plot_tsp_route(solution_vector=solution_vector, prefix=prefix, sub_folder=sub_folder)

        return orig_path, route_path

    def draw(self, filename: str = "graph.png") -> Path:
        prefix = filename.replace(".png", "")
        return self.plot_original_graph(prefix=prefix)