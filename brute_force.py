# core/brute_force.py
import itertools
from typing import Tuple, List, Dict, Union
import numpy as np


class BruteForce:
    """
    Solucionador exato via Força Bruta para Roteamento de Veículos (VRP/TSP).
    
    Por padrão (num_vehicles=1), resolve o problema do Caixeiro Viajante (TSP).
    Caso num_vehicles > 1, reparte exaustivamente o roteamento entre K veículos.
    """
    def __init__(self, dist_matrix: np.ndarray, num_vehicles: int = 1):
        self.dist_matrix = np.array(dist_matrix, dtype=float)
        self.num_nodes = len(dist_matrix)
        self.num_vehicles = num_vehicles

    def calculate_path_cost(self, path: List[int]) -> float:
        """Calcula a soma das distâncias de uma sequência simples de nós [u1, u2, ..., un]."""
        cost = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            cost += float(self.dist_matrix[u, v])
        return cost

    def _generate_partitions(self, sequence: List[int], k: int):
        """Gera todas as formas válidas de dividir uma sequência de cidades em K sub-rotas não-vazias."""
        n = len(sequence)
        if k == 1:
            yield [sequence]
            return

        for cuts in itertools.combinations(range(1, n), k - 1):
            partition = []
            last = 0
            for cut in cuts:
                partition.append(sequence[last:cut])
                last = cut
            partition.append(sequence[last:])
            yield partition

    def solve(self) -> Tuple[float, Union[List[int], Dict[int, List[int]]]]:
        """
        Executa a busca exaustiva. 

        Returns
        -------
        Tuple[float, Union[List[int], Dict[int, List[int]]]]
            - Custo mínimo exato
            - Rota ideal (List para TSP, Dict para VRP com múltiplos veículos)
        """
        depot = 0
        cities = [i for i in range(self.num_nodes) if i != depot]

        best_cost = float('inf')
        best_routes: Union[List[int], Dict[int, List[int]]] = [] if self.num_vehicles == 1 else {}

        # 1. Testa todas as permutações das cidades
        for perm in itertools.permutations(cities):
            # 2. Particiona a permutação entre os veículos
            for partition in self._generate_partitions(list(perm), self.num_vehicles):
                total_cost = 0.0
                current_solution = {}

                for v_idx, sub_route in enumerate(partition, start=1):
                    full_v_route = [depot] + sub_route + [depot]
                    v_cost = self.calculate_path_cost(full_v_route)
                    total_cost += v_cost
                    current_solution[v_idx] = full_v_route

                # Atualiza a melhor solução encontrada
                if total_cost < best_cost:
                    best_cost = total_cost
                    if self.num_vehicles == 1:
                        # Para TSP, extrai diretamente a lista [0, c1, c2, ..., 0]
                        best_routes = current_solution[1]
                    else:
                        # Para VRP, mantém o dicionário {veiculo_1: [...], veiculo_2: [...]}
                        best_routes = current_solution

        return float(best_cost), best_routes