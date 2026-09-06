import itertools
from typing import Tuple, List, Dict, Union, Optional
import numpy as np


class InfeasibleProblemError(ValueError):
    """Exceção lançada quando a instância do problema é inviável devido a restrições de capacidade."""
    pass


class BruteForce:
    """
    Solucionador exato via Força Bruta para Roteamento de Veículos (VRP/TSP).
    
    Por padrão (num_vehicles=1), resolve o problema do Caixeiro Viajante (TSP).
    Caso num_vehicles > 1, reparte exaustivamente o roteamento entre K veículos.
    """
    def __init__(
        self, 
        dist_matrix: np.ndarray, 
        num_vehicles: int = 1,
        capacities: Union[float, List[float]] = 10.0,
        demands: Optional[List[float]] = None
    ):
        self.dist_matrix = np.array(dist_matrix, dtype=float)
        self.num_nodes = len(dist_matrix)
        self.num_vehicles = num_vehicles
        
        # Trata capacidades e demandas
        if isinstance(capacities, (int, float)):
            self.capacities = [float(capacities)] * num_vehicles
        else:
            self.capacities = [float(c) for c in capacities]
            
        self.demands = demands if demands is not None else [0.0] * self.num_nodes

        # Executa validações prévias de viabilidade
        self._validate_feasibility_prechecks()

    def _validate_feasibility_prechecks(self) -> None:
        """Realiza verificações matemáticas rápidas de viabilidade antes de iniciar a busca exaustiva."""
        if self.num_vehicles <= 1:
            return

        total_demand = float(sum(self.demands))
        total_capacity = float(sum(self.capacities))
        max_capacity = max(self.capacities) if self.capacities else 0.0

        # 1. Demanda total do sistema excede a capacidade somada de todos os veículos
        if total_demand > total_capacity:
            raise InfeasibleProblemError(
                f"[Erro de Inviabilidade] Demanda total ({total_demand:.1f}) "
                f"supera a capacidade total combinada dos veículos ({total_capacity:.1f})."
            )

        # 2. Existe alguma cidade cuja demanda sozinha supera o maior veículo disponível
        for node_idx, d in enumerate(self.demands):
            if d > max_capacity:
                raise InfeasibleProblemError(
                    f"[Erro de Inviabilidade] Nó {node_idx} possui demanda ({d:.1f}) "
                    f"superior à capacidade máxima de qualquer veículo ({max_capacity:.1f})."
                )

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
            
        Raises
        ------
        InfeasibleProblemError
            Se nenhuma combinação de rotas for viável quanto às capacidades.
        """
        depot = 0
        cities = [i for i in range(self.num_nodes) if i != depot]

        best_cost = float('inf')
        best_routes = [] if self.num_vehicles == 1 else {}

        for perm in itertools.permutations(cities):
            for partition in self._generate_partitions(list(perm), self.num_vehicles):
                total_cost = 0.0
                current_solution = {}
                is_feasible = True

                for v_idx, sub_route in enumerate(partition, start=1):
                    # Valida capacidade se não for TSP
                    if self.num_vehicles > 1:
                        load = sum(self.demands[node] for node in sub_route)
                        if load > self.capacities[v_idx - 1]:
                            is_feasible = False
                            break  # Sub-rota excede capacidade

                    full_v_route = [depot] + sub_route + [depot]
                    total_cost += self.calculate_path_cost(full_v_route)
                    current_solution[v_idx] = full_v_route

                # Só atualiza se a partição for viável quanto à capacidade
                if is_feasible and total_cost < best_cost:
                    best_cost = total_cost
                    best_routes = current_solution[1] if self.num_vehicles == 1 else current_solution

        # Se após testar todas as partições o custo continuar infinito
        if best_cost == float('inf'):
            raise InfeasibleProblemError(
                f"[Erro de Inviabilidade] Nenhuma partição de rotas atende às restrições "
                f"de capacidade dos veículos {self.capacities} para as demandas {list(self.demands)}."
            )

        return float(best_cost), best_routes