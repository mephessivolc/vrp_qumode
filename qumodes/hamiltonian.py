import tensorflow as tf
import numpy as np
from typing import Tuple, List, Dict, Union, Optional, Any

class Hamiltonian:
    def __init__(
        self, 
        dist_matrix: np.ndarray, 
        num_vehicles: int = 2, 
        vehicle_capacity: Union[float, int, List[float], np.ndarray] = 10.0,
        demands: Union[List[float], np.ndarray, None] = None,
        lmbda_col: Union[float, None] = None,
        lmbda_cap: Union[float, None] = None,
        lmbda_disc: float = 1.0,
        lmbda_bound: float = 10.0,
        lmbda_dist: float = 1.0,
        lmbda: Union[float, None] = None,
    ):
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

        # 3. Multiplicadores de Penalidade Desmembrados (TensorFlow Variables)
        default_lmbda = float(lmbda) if lmbda is not None else float(self.num_nodes * np.max(self.dist_matrix)) 
        
        init_col = float(lmbda_col) if lmbda_col is not None else default_lmbda
        init_cap = float(lmbda_cap) if lmbda_cap is not None else default_lmbda

        self.tf_lmbda_col = tf.Variable(init_col, dtype=tf.float32, trainable=False, name="lmbda_col")
        self.tf_lmbda_cap = tf.Variable(init_cap, dtype=tf.float32, trainable=False, name="lmbda_cap")
        self.tf_lmbda_disc = tf.Variable(lmbda_disc, dtype=tf.float32, trainable=False, name="lmbda_disc")
        self.tf_lmbda_bound = tf.Variable(lmbda_bound, dtype=tf.float32, trainable=False, name="lmbda_bound")
        self.tf_lmbda_dist = tf.Variable(lmbda_dist, dtype=tf.float32, trainable=False, name="lmbda_dist")

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

    # Properties para compatibilidade de interface
    @property
    def lmbda(self) -> float:
        return float(self.tf_lmbda_col.numpy()) 

    @lmbda.setter
    def lmbda(self, value: float):
        self.tf_lmbda_col.assign(float(value)) 

    @property
    def lmbda_cap(self) -> float:
        return float(self.tf_lmbda_cap.numpy()) 

    @lmbda_cap.setter
    def lmbda_cap(self, value: float):
        self.tf_lmbda_cap.assign(float(value)) 

    @property
    def lmbda_col(self) -> float:
        return float(self.tf_lmbda_col.numpy())

    @property
    def lmbda_disc(self) -> float:
        return float(self.tf_lmbda_disc.numpy())

    @property
    def lmbda_bound(self) -> float:
        return float(self.tf_lmbda_bound.numpy())

    @property
    def lmbda_dist(self) -> float:
        return float(self.tf_lmbda_dist.numpy())

    def set_penalties(self, **penalties):
        """Atualiza dinamicamente qualquer multiplicador de penalidade via kwargs ou dict."""
        if "lmbda" in penalties and penalties["lmbda"] is not None:
            self.tf_lmbda_col.assign(float(penalties["lmbda"])) 
        if "lmbda_col" in penalties and penalties["lmbda_col"] is not None:
            self.tf_lmbda_col.assign(float(penalties["lmbda_col"]))
        if "lmbda_cap" in penalties and penalties["lmbda_cap"] is not None:
            self.tf_lmbda_cap.assign(float(penalties["lmbda_cap"]))
        if "lmbda_disc" in penalties and penalties["lmbda_disc"] is not None:
            self.tf_lmbda_disc.assign(float(penalties["lmbda_disc"]))
        if "lmbda_bound" in penalties and penalties["lmbda_bound"] is not None:
            self.tf_lmbda_bound.assign(float(penalties["lmbda_bound"]))
        if "lmbda_dist" in penalties and penalties["lmbda_dist"] is not None:
            self.tf_lmbda_dist.assign(float(penalties["lmbda_dist"]))

    def compute_continuous_cost_tf(self, x_tens: tf.Tensor, p_tens: tf.Tensor) -> tf.Tensor:
        if x_tens.dtype.is_complex:
            x_tens = tf.math.real(x_tens) 
        if p_tens.dtype.is_complex:
            p_tens = tf.math.real(p_tens) 
            
        # 1. Confinamento de Domínio Vetorizado (Boundary Loss)
        out_x = tf.reduce_sum(tf.square(tf.nn.relu(1.0 - x_tens)) + tf.square(tf.nn.relu(x_tens - float(self.max_steps)))) 
        out_p = tf.reduce_sum(tf.square(tf.nn.relu(1.0 - p_tens)) + tf.square(tf.nn.relu(p_tens - float(self.num_vehicles)))) 
        
        # 2. Matrizes de Diferença no Espaço de Fase
        dx = x_tens[:, None] - x_tens[None, :] 
        dp = p_tens[:, None] - p_tens[None, :] 

        # 3. Repulsão Gaussiana Vetorizada entre Cidades
        dist_sq = tf.square(dx) if self.is_tsp else tf.square(dx) + tf.square(dp) 
        sigma = 0.5 
        gaussian_repulsion = tf.exp(-dist_sq / (2.0 * (sigma ** 2))) 
        mask_off_diag = 1.0 - tf.eye(self.num_free_cities, dtype=tf.float32) 
        col_penalty = tf.reduce_sum(gaussian_repulsion * mask_off_diag) / 2.0 

        # 4. Penalidade de Capacidade Suave
        if not self.is_tsp:
            vehicle_weights = tf.exp(-tf.square(p_tens[:, None] - self.tf_vehicles_range[None, :])) 
            load_per_vehicle = tf.reduce_sum(self.tf_free_demands[:, None] * vehicle_weights, axis=0) 
            overcapacity = tf.nn.relu(load_per_vehicle - self.tf_capacities) 
            capacity_penalty = tf.reduce_sum(tf.square(overcapacity)) 
            cost_capacity_penalty = self.tf_lmbda_cap * capacity_penalty
        else:
            cost_capacity_penalty = 0.0 

        # 5. Custo de Distância Suavizado
        same_vehicle_prob = 1.0 if self.is_tsp else tf.exp(-tf.square(dp)) 
        adj_step_prob = tf.exp(-tf.square(tf.abs(dx) - 1.0)) 
        
        free_cities_cost = tf.reduce_sum(self.tf_free_dist * same_vehicle_prob * adj_step_prob * mask_off_diag) / 2.0 
        
        prob_first_step = tf.exp(-tf.square(x_tens - 1.0)) 
        prob_last_step = tf.exp(-tf.square(x_tens - float(self.max_steps))) 
        
        depot_start_cost = tf.reduce_sum(prob_first_step * self.tf_depot_start_dist) 
        depot_end_cost = tf.reduce_sum(prob_last_step * self.tf_depot_end_dist) 
        
        soft_dist_cost = free_cities_cost + depot_start_cost + depot_end_cost 

        # 6. Penalidade de Discretização para P
        pi = float(tf.constant(np.pi)) 
        disc_p_penalty = tf.reduce_sum(tf.square(tf.sin(pi * p_tens))) 

        # Loss Total Diferenciável Desmembrada com Coeficientes Independentes
        total_loss = (
            self.tf_lmbda_bound * (out_x + out_p)
            + self.tf_lmbda_col * col_penalty
            + self.tf_lmbda_disc * disc_p_penalty
            + cost_capacity_penalty
            + self.tf_lmbda_dist * soft_dist_cost
        )
        return total_loss

    def discretize_quadratures(
        self, 
        x_vals: List[float], 
        p_vals: List[float], 
        decoder: Optional[Any] = None
    ) -> Tuple[List[int], List[int]]:
        """Aplica decodificação genérica via objeto externo ou fallback por arredondamento."""
        if decoder is not None and hasattr(decoder, "decode"):
            return decoder.decode(x_vals, p_vals, self.max_steps, self.num_vehicles)
            
        x_disc = [int(np.clip(np.round(x), 1, self.max_steps)) for x in x_vals] 
        p_disc = [int(np.clip(np.round(p), 1, self.num_vehicles)) for p in p_vals] 
        return x_disc, p_disc

    def decode_routes(
        self, 
        x_vals: List[float], 
        p_vals: List[float], 
        decoder: Optional[Any] = None
    ) -> Dict[int, List[int]]:
        x_disc, p_disc = self.discretize_quadratures(x_vals, p_vals, decoder=decoder)
        routes = {}
        for v in range(1, self.num_vehicles + 1): 
            vehicle_cities = [(i + 1, x_disc[i]) for i in range(self.num_free_cities) if p_disc[i] == v] 
            if not vehicle_cities:
                routes[v] = [0, 0] 
                continue
            vehicle_cities.sort(key=lambda item: item[1]) 
            routes[v] = [0] + [city_id for city_id, _ in vehicle_cities] + [0] 
        return routes

    def compute_metrics(
        self, 
        x_vals: List[float], 
        p_vals: List[float], 
        decoder: Optional[Any] = None
    ) -> Dict[str, float]:
        """Calcula a decomposição exata das rotas e violações para avaliação de métricas."""
        x_disc, p_disc = self.discretize_quadratures(x_vals, p_vals, decoder=decoder)
        cost_dist, penalty_col, total_excess_capacity = 0.0, 0.0, 0.0 

        # Colisões discretas
        for i in range(self.num_free_cities): 
            for j in range(i + 1, self.num_free_cities): 
                if self.is_tsp:
                    if x_disc[i] == x_disc[j]:
                        penalty_col += self.lmbda_col
                else:
                    if p_disc[i] == p_disc[j] and x_disc[i] == x_disc[j]:
                        penalty_col += self.lmbda_col

        # Violações de capacidade
        penalty_cap = 0.0 
        for v_idx in range(1, self.num_vehicles + 1): 
            assigned_cities = [i + 1 for i in range(self.num_free_cities) if p_disc[i] == v_idx] 
            load_v = sum(self.demands[city] for city in assigned_cities) 
            cap_v = self.capacities[v_idx - 1] 
            
            if load_v > cap_v: 
                excess = load_v - cap_v 
                total_excess_capacity += excess 
                penalty_cap += self.lmbda_cap * (excess ** 2) 

        # Distância pura das rotas
        routes = self.decode_routes(x_vals, p_vals, decoder=decoder)
        for v, route in routes.items(): 
            if route != [0, 0]: 
                for k in range(len(route) - 1): 
                    cost_dist += self.dist_matrix[route[k], route[k + 1]] 

        total_cost = float(cost_dist + penalty_col + penalty_cap) 
        
        return {
            "total_cost": total_cost, 
            "route_distance": float(cost_dist), 
            "collision_penalty": float(penalty_col), 
            "capacity_penalty": float(penalty_cap), 
            "capacity_violation_magnitude": float(total_excess_capacity), 
            "is_feasible": bool(penalty_col == 0.0 and total_excess_capacity == 0.0) 
        }

    def compute_cost(self, x_vals: List[float], p_vals: List[float], decoder: Optional[Any] = None) -> float:
        return self.compute_metrics(x_vals, p_vals, decoder=decoder)["total_cost"] 

    def compute_composite_score(
        self, 
        exact_cost: float, 
        x_vals: List[float], 
        p_vals: List[float], 
        plateau_iters: int = 0,
        w1: float = 1.0, 
        w2: float = 0.001,
        decoder: Optional[Any] = None
    ) -> float:
        """Calcula a pontuação de otimização combinando aproximação, violação e estagnação."""
        metrics = self.compute_metrics(x_vals, p_vals, decoder=decoder)
        route_dist = metrics["route_distance"] 
        violation = metrics["capacity_violation_magnitude"] 
        
        approx_ratio = exact_cost / route_dist if route_dist > 0 else 0.0 
        score = approx_ratio - (w1 * violation) - (w2 * plateau_iters) 
        return float(score) 