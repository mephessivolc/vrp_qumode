# graphs.py
from pathlib import Path
from typing import Union, List, Dict, Tuple
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Importa o gerenciador de saída para direcionar figuras para 'result/'
from logger import ExperimentLogger


class GraphBuilder:
    """
    Construtor e visualizador de grafos para problemas de otimização de rotas (TSP e VRP).
    Salva as imagens diretamente nas pastas correspondentes dentro de `result/`.
    """
    def __init__(self, n: int = 3, seed: int = 42, logger: ExperimentLogger = None):
        self.n = n
        self.seed = seed
        self.matrix = self._generate_matrix()
        self.logger = logger if logger is not None else ExperimentLogger()

    def _generate_matrix(self) -> np.ndarray:
        """Gera uma matriz de distâncias simétrica com diagonal nula."""
        np.random.seed(self.seed)
        adj = np.random.uniform(1.0, 10.0, size=(self.n, self.n))
        adj = (adj + adj.T) / 2.0
        np.fill_diagonal(adj, 0.0)
        return np.round(adj, 2)

    def _convert_qaoa_vector_to_route(self, vector: Union[List, tuple, np.ndarray]) -> List[int]:
        """
        Converte diferentes formatos de entrada do TSP (vetores discretos N^2 ou contínuos) 
        em uma lista válida de nós no intervalo [0, n-1].
        """
        if isinstance(vector, (tuple, np.ndarray)):
            vector = list(vector)

        route = []

        # 1. Trata vetor binário/matriz N^2 (ex: QAOA)
        if len(vector) == self.n ** 2:
            matrix_form = np.array(vector).reshape((self.n, self.n))
            for step in range(self.n):
                city = int(np.argmax(matrix_form[:, step])) % self.n
                route.append(city)
        # 2. Trata vetor de tamanho N (ordem direta de visitas ou fases contínuas)
        elif len(vector) == self.n:
            # Se forem valores flutuantes/fases contínuas do CV, mapeia via ordenação (argsort)
            if any(isinstance(x, float) for x in vector):
                route = list(np.argsort(vector))
            else:
                route = [int(c) % self.n for c in vector]
        else:
            route = [int(c) % self.n for c in vector]

        # 3. Sanitização Final: Garante permutação sem duplicatas dentro de [0, n-1]
        valid_route = []
        for node in route:
            if node not in valid_route and 0 <= node < self.n:
                valid_route.append(node)
        
        # Preenche cidades faltantes se a solução do VQE omitiu alguma
        missing = [i for i in range(self.n) if i not in valid_route]
        valid_route.extend(missing)

        return valid_route[:self.n]

    def plot_original_graph(self, prefix: str = "graph", problem_type: str = "TSP") -> Path:
        """Gera e salva a imagem do grafo completo original na pasta do problema dentro de `result/`."""
        out_dir = Path(self.logger.get_figures_dir(problem_type))

        G = nx.Graph()
        for i in range(self.n):
            G.add_node(i)
            for j in range(i + 1, self.n):
                G.add_edge(i, j, weight=self.matrix[i, j])

        pos = nx.spring_layout(G, seed=self.seed)

        plt.figure(figsize=(7, 6))
        
        # Destaca o nó 0 (Depósito)
        node_colors = ['gold' if node == 0 else 'lightblue' for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, alpha=0.7)

        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)

        plt.title(f"Grafo Original (N={self.n} Nós | Nó 0 = Depósito)", fontsize=12)
        plt.axis('off')

        orig_path = out_dir / f"{prefix}_original.png"
        plt.savefig(orig_path, dpi=300, bbox_inches='tight')
        plt.close()

        return orig_path

    def plot_tsp_route(self, solution_vector: Union[List, tuple, np.ndarray], prefix: str = "tsp") -> Path:
        """Gera e salva o trajeto do TSP destacado na pasta `result/tsp/figures`."""
        out_dir = Path(self.logger.get_figures_dir("TSP"))

        # Converte e sanitiza a rota para garantir nós válidos
        route = self._convert_qaoa_vector_to_route(solution_vector)
        full_cycle = route + [route[0]]

        G_base = nx.Graph()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                G_base.add_edge(i, j, weight=self.matrix[i, j])

        pos = nx.spring_layout(G_base, seed=self.seed)

        G_directed = nx.DiGraph()
        for i in range(self.n):
            G_directed.add_node(i)

        route_edges = []
        for i in range(len(full_cycle) - 1):
            u, v = full_cycle[i], full_cycle[i + 1]
            G_directed.add_edge(u, v)
            route_edges.append((u, v))

        plt.figure(figsize=(7, 6))
        
        # Desenha arestas em segundo plano
        nx.draw_networkx_edges(G_base, pos, edge_color='lightgray', width=1.0, style='dashed', alpha=0.5)

        # Desenha nós (Depósito vs Cidades)
        node_colors = ['gold' if node == 0 else 'lightgreen' for node in G_base.nodes()]
        nx.draw_networkx_nodes(G_base, pos, node_color=node_colors, node_size=750)
        nx.draw_networkx_labels(G_base, pos, font_size=12, font_weight='bold')

        # Desenha trajeto único em vermelho
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
        """Gera e salva os trajetos de múltiplos veículos na pasta `result/vrp/figures`."""
        out_dir = Path(self.logger.get_figures_dir("VRP"))

        G_base = nx.Graph()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                G_base.add_edge(i, j, weight=self.matrix[i, j])

        pos = nx.spring_layout(G_base, seed=self.seed)

        color_palette = ['#E63946', '#1D3557', '#2A9D8F', '#F4A261', '#9C27B0', '#3F51B5']

        plt.figure(figsize=(8, 7))
        
        # Desenha fundo pontilhado
        nx.draw_networkx_edges(G_base, pos, edge_color='lightgray', width=1.0, style='dashed', alpha=0.4)

        # Desenha nós
        node_colors = ['gold' if node == 0 else 'lightblue' for node in G_base.nodes()]
        nx.draw_networkx_nodes(G_base, pos, node_color=node_colors, node_size=800)
        nx.draw_networkx_labels(G_base, pos, font_size=12, font_weight='bold')

        # Desenha rota de cada veículo
        legend_handles = []
        for idx, (v_id, route) in enumerate(routes.items()):
            color = color_palette[(idx) % len(color_palette)]
            
            # Sanitiza rotas do VRP para garantir que pertençam ao intervalo [0, n-1]
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
        """Método de compatibilidade para gerar o grafo original e a rota TSP/VRP."""
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
        """Método de conveniência para salvar o grafo original."""
        prefix = filename.replace(".png", "")
        return self.plot_original_graph(prefix=prefix)
