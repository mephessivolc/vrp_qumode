# vrp/hamiltonian.py
import tensorflow as tf
import numpy as np
from typing import Tuple, List, Dict

class Hamiltonian:
    def __init__(
        self, 
        dist_matrix: np.ndarray, 
        num_vehicles: int = 2, 
        lmbda: float = 100.0,
        lmbda_empty: float = 150.0
    ):
        self.dist_matrix = np.array(dist_matrix, dtype=np.float32)
        self.num_nodes = len(dist_matrix)
        self.num_vehicles = num_vehicles
        self.num_free_cities = self.num_nodes - 1
        self.lmbda = lmbda
        self.lmbda_empty = lmbda_empty
        self.max_steps = self.num_free_cities

    def compute_continuous_cost_tf(self, x_tens: tf.Tensor, p_tens: tf.Tensor) -> tf.Tensor:
        # 1. Manter quadraturas dentro do intervalo útil [1, max_steps] e [1, num_vehicles]
        out_x = tf.reduce_sum(tf.square(tf.maximum(0.0, 1.0 - x_tens)) + tf.square(tf.maximum(0.0, x_tens - float(self.max_steps))))
        out_p = tf.reduce_sum(tf.square(tf.maximum(0.0, 1.0 - p_tens)) + tf.square(tf.maximum(0.0, p_tens - float(self.num_vehicles))))
        
        # 2. Repulsão Inversa Ativa (Evita atração para a mesma posição no espaço de fase)
        col_penalty = 0.0
        for i in range(self.num_free_cities):
            for j in range(i + 1, self.num_free_cities):
                dist_sq = tf.square(p_tens[i] - p_tens[j]) + tf.square(x_tens[i] - x_tens[j])
                col_penalty += 1.0 / (dist_sq + 0.1)

        # 3. Penalidade por Veículo Vazio
        empty_penalty = 0.0
        for v in range(1, self.num_vehicles + 1):
            cov = tf.reduce_sum(tf.exp(-tf.square(p_tens - float(v))))
            empty_penalty += tf.exp(-1.5 * cov)

        # 4. TERMO NOVO: Aproximação Suave da Distância (Soft Distance)
        # Guia o Adam a aproximar cidades que estão perto no grafo
        soft_dist_cost = 0.0
        for i in range(self.num_free_cities):
            for j in range(i + 1, self.num_free_cities):
                # Probabilidade diferenciável de estarem no mesmo veículo (p próximo)
                same_vehicle_prob = tf.exp(-tf.square(p_tens[i] - p_tens[j]))
                # Probabilidade diferenciável de serem visitadas em sequência (x consecutivo)
                adj_step_prob = tf.exp(-tf.square(tf.abs(x_tens[i] - x_tens[j]) - 1.0))
                
                # Custo da matriz de adjacência (índice + 1 devido ao depósito no nó 0)
                d_ij = self.dist_matrix[i + 1, j + 1]
                soft_dist_cost += d_ij * same_vehicle_prob * adj_step_prob

        # Custo Total Diferenciável
        total_loss = (
            10.0 * (out_x + out_p) 
            + self.lmbda * col_penalty 
            + self.lmbda_empty * empty_penalty 
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
        cost_dist, penalty_col, penalty_empty = 0.0, 0.0, 0.0

        for i in range(self.num_free_cities):
            for j in range(i + 1, self.num_free_cities):
                if p_disc[i] == p_disc[j] and x_disc[i] == x_disc[j]:
                    penalty_col += self.lmbda

        vehicles_used = set(p_disc)
        for v in range(1, self.num_vehicles + 1):
            if v not in vehicles_used:
                penalty_empty += self.lmbda_empty

        routes = self.decode_routes(x_vals, p_vals)
        for v, route in routes.items():
            if route != [0, 0]:
                for k in range(len(route) - 1):
                    cost_dist += self.dist_matrix[route[k], route[k + 1]]

        return float(cost_dist + penalty_col + penalty_empty)