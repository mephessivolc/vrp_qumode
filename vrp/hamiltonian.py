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
        # Nomes mantidos para compatibilidade com o main.py
        self.dist_matrix = np.array(dist_matrix, dtype=np.float32)
        self.num_nodes = len(dist_matrix)
        self.num_vehicles = num_vehicles
        self.num_free_cities = self.num_nodes - 1
        
        # 1. Configuração de Demandas
        if demands is None:
            self.demands = np.ones(self.num_nodes, dtype=np.float32)
            self.demands[0] = 0.0
        else:
            self.demands = np.array(demands, dtype=np.float32)
            if len(self.demands) != self.num_nodes:
                raise ValueError(f"Tamanho de demandas ({len(self.demands)}) diferente de N ({self.num_nodes}).")

        # 2. Configuração de Capacidades
        if isinstance(vehicle_capacity, (float, int)):
            self.capacities = np.full(self.num_vehicles, float(vehicle_capacity), dtype=np.float32)
        else:
            self.capacities = np.array(vehicle_capacity, dtype=np.float32)
            if len(self.capacities) != self.num_vehicles:
                raise ValueError(f"Tamanho das capacidades ({len(self.capacities)}) diferente de num_vehicles ({self.num_vehicles}).")

        # 3. Multiplicadores de Penalidade
        self.lmbda = float(lmbda) if lmbda is not None else float(self.num_nodes * np.max(self.dist_matrix))
        self.lmbda_cap = float(lmbda_cap) if lmbda_cap is not None else self.lmbda

        self.max_steps = self.num_free_cities
        self.is_tsp = (self.num_vehicles == 1)

        # 4. Constantes Pré-convertidas para TensorFlow
        self.tf_dist_matrix = tf.constant(self.dist_matrix, dtype=tf.float32)
        self.tf_free_dist = tf.constant(self.dist_matrix[1:, 1:], dtype=tf.float32)
        self.tf_depot_start_dist = tf.constant(self.dist_matrix[0, 1:], dtype=tf.float32)
        self.tf_depot_end_dist = tf.constant(self.dist_matrix[1:, 0], dtype=tf.float32)
        self.tf_free_demands = tf.constant(self.demands[1:], dtype=tf.float32)
        self.tf_capacities = tf.constant(self.capacities, dtype=tf.float32)
        self.tf_vehicles_range = tf.range(1, self.num_vehicles + 1, dtype=tf.float32)

    def compute_continuous_cost_tf(self, x_tens: tf.Tensor, p_tens: tf.Tensor) -> tf.Tensor:

        # Garante que os tensores de entrada sejam float32 reais sem disparar warnings
        if x_tens.dtype.is_complex:
            x_tens = tf.real(x_tens)
        if p_tens.dtype.is_complex:
            p_tens = tf.real(p_tens)
            
        # 1. Confinamento de Domínio Vetorizado (Boundary Loss)
        out_x = tf.reduce_sum(tf.square(tf.nn.relu(1.0 - x_tens)) + tf.square(tf.nn.relu(x_tens - float(self.max_steps))))
        out_p = tf.reduce_sum(tf.square(tf.nn.relu(1.0 - p_tens)) + tf.square(tf.nn.relu(p_tens - float(self.num_vehicles))))
        
        # 2. Matrizes de Diferença no Espaço de Fase
        dx = x_tens[:, None] - x_tens[None, :]  # Shape [N-1, N-1]
        dp = p_tens[:, None] - p_tens[None, :]  # Shape [N-1, N-1]

        # 3. Repulsão Gaussiana Vetorizada entre Cidades (Sem Estouros de Gradiente)
        dist_sq = tf.square(dx) if self.is_tsp else tf.square(dx) + tf.square(dp)
        sigma = 0.5
        gaussian_repulsion = tf.exp(-dist_sq / (2.0 * (sigma ** 2)))
        mask_off_diag = 1.0 - tf.eye(self.num_free_cities, dtype=tf.float32)
        col_penalty = tf.reduce_sum(gaussian_repulsion * mask_off_diag) / 2.0

        # 4. Penalidade de Capacidade Totalmente Vetorizada
        if not self.is_tsp:
            # Pertencimento Gaussiano Suave [N-1, V]
            vehicle_weights = tf.exp(-tf.square(p_tens[:, None] - self.tf_vehicles_range[None, :]))
            load_per_vehicle = tf.reduce_sum(self.tf_free_demands[:, None] * vehicle_weights, axis=0)
            overcapacity = tf.nn.relu(load_per_vehicle - self.tf_capacities)
            capacity_penalty = tf.reduce_sum(tf.square(overcapacity))
            cost_capacity_penalty = self.lmbda_cap * capacity_penalty
        else:
            cost_capacity_penalty = 0.0

        # 5. Custo de Distância Suavizado (Cidades Livres + Conexões com o Depósito)
        same_vehicle_prob = 1.0 if self.is_tsp else tf.exp(-tf.square(dp))
        adj_step_prob = tf.exp(-tf.square(tf.abs(dx) - 1.0))
        
        # Distância Cidades -> Cidades
        free_cities_cost = tf.reduce_sum(self.tf_free_dist * same_vehicle_prob * adj_step_prob * mask_off_diag) / 2.0
        
        # Distância Depósito -> Primeira Cidade e Última Cidade -> Depósito
        prob_first_step = tf.exp(-tf.square(x_tens - 1.0))
        prob_last_step = tf.exp(-tf.square(x_tens - float(self.max_steps)))
        
        depot_start_cost = tf.reduce_sum(prob_first_step * self.tf_depot_start_dist)
        depot_end_cost = tf.reduce_sum(prob_last_step * self.tf_depot_end_dist)
        
        soft_dist_cost = free_cities_cost + depot_start_cost + depot_end_cost

        # Penalidade de discretização para p (mínimos globais nos inteiros 1, 2, ..., V)
        pi = float(tf.constant(np.pi))
        disc_p_penalty = tf.reduce_sum(tf.square(tf.sin(pi * p_tens)))

        # Custo Total Diferenciável
        total_loss = (
            10.0 * (out_x + out_p) 
            + self.lmbda * (col_penalty + disc_p_penalty)
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