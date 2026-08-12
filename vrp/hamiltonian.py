import tensorflow as tf
import numpy as np
from typing import Tuple, List, Dict, Union

class Hamiltonian:
    def __init__(
        self, 
        dist_matrix: np.ndarray, 
        num_vehicles: int = 2, 
        vehicle_capacity: Union[float, int, List[float], np.ndarray] = 10.0,
        demands: Union[List[float], np.ndarray, None] = None,
        lmbda: Union[float, None] = None,
        lmbda_cap: Union[float, None] = None,
    ):
        self.dist_matrix = np.array(dist_matrix, dtype=np.float32)
        self.num_nodes = len(dist_matrix)
        self.num_vehicles = num_vehicles
        self.num_free_cities = self.num_nodes - 1
        
        # 1. Configuração de Demandas das Cidades
        if demands is None:
            # Caso não sejam informadas, assume demanda unitária (1.0) para cada cidade livre e 0 para o depósito
            self.demands = np.ones(self.num_nodes, dtype=np.float32)
            self.demands[0] = 0.0
        else:
            self.demands = np.array(demands, dtype=np.float32)
            if len(self.demands) != self.num_nodes:
                raise ValueError(f"O tamanho do vetor de demandas ({len(self.demands)}) deve ser igual a N ({self.num_nodes}).")

        # 2. Configuração de Capacidade dos Veículos (Homogênea ou Heterogênea)
        if isinstance(vehicle_capacity, (float, int)):
            self.capacities = np.full(self.num_vehicles, float(vehicle_capacity), dtype=np.float32)
        else:
            self.capacities = np.array(vehicle_capacity, dtype=np.float32)
            if len(self.capacities) != self.num_vehicles:
                raise ValueError(f"O vetor de capacidades ({len(self.capacities)}) deve ter tamanho igual a num_vehicles ({self.num_vehicles}).")

        # 3. Multiplicadores de Penalidade (Lagrange / Penalidade de Restrição)
        if lmbda is not None:
            if not isinstance(lmbda, (float, int)):
                raise TypeError("O parâmetro 'lmbda' deve ser um número float ou int.")
            self.lmbda = float(lmbda)
        else:
            self.lmbda = float(self.num_nodes * np.max(self.dist_matrix))
        
        if lmbda_cap is not None:
            if not isinstance(lmbda_cap, (float, int)):
                raise TypeError("O parâmetro 'lmbda_cap' deve ser um número float ou int.")
            self.lmbda_cap = float(lmbda_cap)
        else:
            self.lmbda_cap = self.lmbda

        self.max_steps = self.num_free_cities

        # Flag para chavear entre TSP (1 veículo) e VRP (> 1 veículos)
        self.is_tsp = (self.num_vehicles == 1)

    def compute_continuous_cost_tf(self, x_tens: tf.Tensor, p_tens: tf.Tensor) -> tf.Tensor:
        # 1. Manter quadraturas dentro do intervalo útil [1, max_steps] e [1, num_vehicles]
        out_x = tf.reduce_sum(
            tf.square(tf.maximum(0.0, 1.0 - x_tens)) + 
            tf.square(tf.maximum(0.0, x_tens - float(self.max_steps)))
        )
        
        out_p = tf.reduce_sum(
            tf.square(tf.maximum(0.0, 1.0 - p_tens)) + 
            tf.square(tf.maximum(0.0, p_tens - float(self.num_vehicles)))
        )
        
        # 2. Repulsão Inversa Ativa (Evita que cidades colidam no espaço de fase)
        col_penalty = 0.0
        for i in range(self.num_free_cities):
            for j in range(i + 1, self.num_free_cities):
                if self.is_tsp:
                    dist_sq = tf.square(x_tens[i] - x_tens[j])
                else:
                    dist_sq = tf.square(p_tens[i] - p_tens[j]) + tf.square(x_tens[i] - x_tens[j])
                
                col_penalty += 1.0 / (dist_sq + 0.1)

        # 3. Penalidade de Capacidade (Substitui a antiga penalidade de veículo vazio)
        capacity_penalty = 0.0
        free_demands = tf.constant(self.demands[1:], dtype=tf.float32)  # Cidades livres 1..N-1
        
        for v_idx in range(self.num_vehicles):
            v_num = float(v_idx + 1)
            # Ponderação suave de pertencimento da cidade livre ao veículo v
            weights = tf.exp(-tf.square(p_tens - v_num)) if not self.is_tsp else tf.ones_like(p_tens)
            
            # Carga estimada no veículo v
            load_v = tf.reduce_sum(free_demands * weights)
            cap_v = float(self.capacities[v_idx])
            
            # Penalidade quadrática para sobrecarga (L_v > C_v)
            overcapacity = tf.maximum(0.0, load_v - cap_v)
            capacity_penalty += tf.square(overcapacity)

        # 4. Aproximação Suave da Distância (Soft Distance)
        soft_dist_cost = 0.0
        for i in range(self.num_free_cities):
            for j in range(i + 1, self.num_free_cities):
                if self.is_tsp:
                    same_vehicle_prob = 1.0
                else:
                    same_vehicle_prob = tf.exp(-tf.square(p_tens[i] - p_tens[j]))

                adj_step_prob = tf.exp(-tf.square(tf.abs(x_tens[i] - x_tens[j]) - 1.0))
                d_ij = self.dist_matrix[i + 1, j + 1]
                soft_dist_cost += d_ij * same_vehicle_prob * adj_step_prob

        cost_capacity_penalty = self.lmbda_cap * capacity_penalty if not self.is_tsp else 0.0
        # Custo Total Diferenciável
        total_loss = (
            10.0 * (out_x + out_p) 
            + self.lmbda * col_penalty 
            + cost_capacity_penalty
            + soft_dist_cost
        )
        return total_loss

    def discretize_quadratures(self, x_vals: List[float], p_vals: List[float]) -> Tuple[List[int], List[int]]:
        x_disc = [int(np.clip(np.round(x), 1, self.max_steps)) for x in x_vals]
        p_disc = [int(np.clip(np.round(p), 1, self.num_vehicles)) for p in p_vals]
        return x_disc, p_disc

    def decode_routes(self, x_vals: List[float], p_vals: List[float]) -> Dict[int, List[int]]:
        x_disc, p_disc = self.discretize_quadratures(x_vals, p_vals)
        routes = {}
        for v in range(1, self.num_vehicles + 1):
            vehicle_cities = [(i + 1, x_disc[i]) for i in range(self.num_free_cities) if p_disc[i] == v]
            if not vehicle_cities:
                routes[v] = [0, 0]
                continue
            vehicle_cities.sort(key=lambda item: item[1])
            routes[v] = [0] + [city_id for city_id, _ in vehicle_cities] + [0]
        return routes

    def compute_cost(self, x_vals: List[float], p_vals: List[float]) -> float:
        x_disc, p_disc = self.discretize_quadratures(x_vals, p_vals)
        cost_dist, penalty_col, penalty_cap = 0.0, 0.0, 0.0

        # 1. Penalidade por Colisão Discreta
        for i in range(self.num_free_cities):
            for j in range(i + 1, self.num_free_cities):
                if self.is_tsp:
                    if x_disc[i] == x_disc[j]:
                        penalty_col += self.lmbda
                else:
                    if p_disc[i] == p_disc[j] and x_disc[i] == x_disc[j]:
                        penalty_col += self.lmbda

        # 2. Penalidade por Excesso de Capacidade Discreta
        for v_idx in range(1, self.num_vehicles + 1):
            assigned_cities = [i + 1 for i in range(self.num_free_cities) if p_disc[i] == v_idx]
            load_v = sum(self.demands[city] for city in assigned_cities)
            cap_v = self.capacities[v_idx - 1]
            
            if load_v > cap_v:
                penalty_cap += self.lmbda_cap * ((load_v - cap_v) ** 2)

        # 3. Custo Real das Distâncias das Rotas
        routes = self.decode_routes(x_vals, p_vals)
        for v, route in routes.items():
            if route != [0, 0]:
                for k in range(len(route) - 1):
                    cost_dist += self.dist_matrix[route[k], route[k + 1]]

        return float(cost_dist + penalty_col + penalty_cap)