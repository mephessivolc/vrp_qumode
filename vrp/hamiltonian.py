# vrp/hamiltonian.py
import tensorflow as tf
import numpy as np
from typing import Tuple, List, Dict, Union

class Hamiltonian:
    def __init__(
        self, 
        dist_matrix: np.ndarray, 
        num_vehicles: int = 2, 
        lmbda: Union[float, None] = None,
        lmbda_empty: Union[float, None] = None,
    ):
        self.dist_matrix = np.array(dist_matrix, dtype=np.float32)
        self.num_nodes = len(dist_matrix)
        self.num_vehicles = num_vehicles
        self.num_free_cities = self.num_nodes - 1
        if lmbda is not None:
            if not isinstance(lmbda, (float, int)):
                raise TypeError("This 'lmbda' variable must be a float.")
            self.lmbda = float(lmbda)
        else:
            self.lmbda = self.num_nodes * np.max(self.dist_matrix)
        
        if lmbda_empty is not None:
            if not isinstance(lmbda_empty, (float, int)):
                raise TypeError("This 'lmbda_empty' variable must be a float.")
            self.lmbda_empty = float(lmbda_empty)
        else:
            self.lmbda_empty = self.lmbda

        self.max_steps = self.num_free_cities

        # Flag para chavear entre TSP (1 veículo) e VRP (> 1 veículos)
        self.is_tsp = (self.num_vehicles == 1)

    def compute_continuous_cost_tf(self, x_tens: tf.Tensor, p_tens: tf.Tensor) -> tf.Tensor:
        # 1. Manter quadraturas dentro do intervalo útil [1, max_steps] e [1, num_vehicles]
        out_x = tf.reduce_sum(
            tf.square(tf.maximum(0.0, 1.0 - x_tens)) + 
            tf.square(tf.maximum(0.0, x_tens - float(self.max_steps)))
        )
        
        # No TSP, p é trivial/fixado em 1.0, mas mantemos o limite por consistência
        out_p = tf.reduce_sum(
            tf.square(tf.maximum(0.0, 1.0 - p_tens)) + 
            tf.square(tf.maximum(0.0, p_tens - float(self.num_vehicles)))
        )
        
        # 2. Repulsão Inversa Ativa (Evita que cidades colidam no espaço de fase)
        col_penalty = 0.0
        for i in range(self.num_free_cities):
            for j in range(i + 1, self.num_free_cities):
                if self.is_tsp:
                    # TSP: Apenas a posição temporal x importa para a colisão
                    dist_sq = tf.square(x_tens[i] - x_tens[j])
                else:
                    # VRP: Considera posição x e veículo p
                    dist_sq = tf.square(p_tens[i] - p_tens[j]) + tf.square(x_tens[i] - x_tens[j])
                
                col_penalty += 1.0 / (dist_sq + 0.1)

        # 3. Penalidade por Veículo Vazio (Aplicada APENAS para VRP)
        empty_penalty = 0.0
        if not self.is_tsp:
            for v in range(1, self.num_vehicles + 1):
                cov = tf.reduce_sum(tf.exp(-tf.square(p_tens - float(v))))
                empty_penalty += tf.exp(-1.5 * cov)

        # 4. Aproximação Suave da Distância (Soft Distance)
        soft_dist_cost = 0.0
        for i in range(self.num_free_cities):
            for j in range(i + 1, self.num_free_cities):
                if self.is_tsp:
                    # TSP: Todas as cidades estão garantidamente no mesmo veículo
                    same_vehicle_prob = 1.0
                else:
                    # VRP: Probabilidade de estarem no mesmo veículo v
                    same_vehicle_prob = tf.exp(-tf.square(p_tens[i] - p_tens[j]))

                # Probabilidade diferenciável de serem visitadas em sequência temporal (x consecutivo)
                adj_step_prob = tf.exp(-tf.square(tf.abs(x_tens[i] - x_tens[j]) - 1.0))
                
                d_ij = self.dist_matrix[i + 1, j + 1]
                soft_dist_cost += d_ij * same_vehicle_prob * adj_step_prob

        # Custo Total Diferenciável
        total_loss = (
            10.0 * (out_x + out_p) 
            + self.lmbda * col_penalty 
            + (self.lmbda_empty * empty_penalty if not self.is_tsp else 0.0)
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

        # Penalidade por Colisão Discreta
        for i in range(self.num_free_cities):
            for j in range(i + 1, self.num_free_cities):
                if self.is_tsp:
                    # No TSP, duas cidades não podem ocupar o mesmo passo de tempo x
                    if x_disc[i] == x_disc[j]:
                        penalty_col += self.lmbda
                else:
                    # No VRP, duas cidades não podem ocupar o mesmo passo x E o mesmo veículo p
                    if p_disc[i] == p_disc[j] and x_disc[i] == x_disc[j]:
                        penalty_col += self.lmbda

        # Penalidade por Veículo Vazio (Apenas para VRP)
        if not self.is_tsp:
            vehicles_used = set(p_disc)
            for v in range(1, self.num_vehicles + 1):
                if v not in vehicles_used:
                    penalty_empty += self.lmbda_empty

        # Custo Real das Distâncias das Rotas
        routes = self.decode_routes(x_vals, p_vals)
        for v, route in routes.items():
            if route != [0, 0]:
                for k in range(len(route) - 1):
                    cost_dist += self.dist_matrix[route[k], route[k + 1]]

        return float(cost_dist + penalty_col + penalty_empty)