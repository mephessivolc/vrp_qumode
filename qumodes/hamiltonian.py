# qumodes/hamiltonian.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy.sparse.linalg import LinearOperator


@dataclass
class HamiltonianParams:
    sigma_assign: float = 0.30
    sigma_empty: float = 0.40
    lambda_col: float = 20.0
    lambda_gap: float = 20.0
    lambda_vehicle: float = 15.0


class Hamiltonian:
    """
    Hamiltoniano Quântico Hermitiano para problemas de roteamento (TSP e CVRP/VRP)
    baseado em operadores de quadratura contínua e truncamento em espaço de Hilbert.
    """
    def __init__(
        self,
        dist_matrix: np.ndarray,
        num_vehicles: int = 1,
        cutoff: int = 5,
        params: Optional[HamiltonianParams] = None,
        vehicle_capacity: Union[float, int, List[float], np.ndarray] = 10.0,
        demands: Optional[Union[List[float], np.ndarray]] = None,
        **kwargs
    ):
        self.dist_matrix = np.array(dist_matrix, dtype=np.float32)
        self.num_nodes = len(self.dist_matrix)
        self.num_vehicles = num_vehicles
        self.num_free_cities = self.num_nodes - 1
        self.cutoff = cutoff
        self.num_qumodes = self.num_free_cities
        self.d = cutoff

        self.params = params if params is not None else HamiltonianParams()

        # Compatibilidade com parâmetros legados passados via kwargs
        if "lmbda_col" in kwargs and kwargs["lmbda_col"] is not None:
            self.params.lambda_col = float(kwargs["lmbda_col"])
        if "lmbda" in kwargs and kwargs["lmbda"] is not None:
            self.params.lambda_col = float(kwargs["lmbda"])

        # Configuração de Demandas
        if demands is None:
            self.demands = np.ones(self.num_nodes, dtype=np.float32)
            self.demands[0] = 0.0
        else:
            self.demands = np.array(demands, dtype=np.float32)
            self.demands[0] = 0.0

        # Configuração de Capacidades
        if isinstance(vehicle_capacity, (float, int)):
            self.capacities = np.full(self.num_vehicles, float(vehicle_capacity), dtype=np.float32)
        else:
            self.capacities = np.array(vehicle_capacity, dtype=np.float32)

        # Propriedades de retrocompatibilidade
        self.lmbda_col = self.params.lambda_col
        self.lmbda_cap = self.params.lambda_vehicle

        # Construção do Operador Hermitiano e Metadados
        self.operator, self.metadata = self._build_hamiltonian()

    def set_penalties(self, lmbda_col: float, lmbda_cap: float) -> None:
        """Atualiza dinamicamente as penalidades e reconstrói o Hamiltoniano."""
        self.params.lambda_col = float(lmbda_col)
        self.params.lambda_vehicle = float(lmbda_cap)
        self.lmbda_col = float(lmbda_col)
        self.lmbda_cap = float(lmbda_cap)
        self.operator, self.metadata = self._build_hamiltonian()

    # --- OPERADORES HERMITIANOS LOCAIS ---
    def _local_x_operator(self) -> np.ndarray:
        a = np.zeros((self.cutoff, self.cutoff), dtype=np.complex128)
        for n in range(self.cutoff - 1):
            a[n, n + 1] = np.sqrt(n + 1)
        return (a + a.conj().T) / np.sqrt(2.0)

    def _route_coordinate(
        self, 
        x: np.ndarray, 
        x_min: Optional[float] = None, 
        x_max: Optional[float] = None
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x_min is None:
            x_min = float(np.min(x))
        if x_max is None:
            x_max = float(np.max(x))
        if np.isclose(x_max, x_min):
            return np.ones_like(x)

        total_slots = self.num_vehicles * self.num_free_cities
        return 1.0 + (total_slots - 1.0) * ((x - x_min) / (x_max - x_min))

    def _local_z_operator(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = self._local_x_operator()
        x_eigs, Ux = np.linalg.eigh(X)
        z_eigs = self._route_coordinate(x=x_eigs, x_min=float(x_eigs.min()), x_max=float(x_eigs.max()))
        Z = (Ux * z_eigs) @ Ux.conj().T
        return Z, X, x_eigs, z_eigs, Ux

    # --- COMPONENTES DE ENERGIA ---
    def _soft_assignments(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32)
        centers = np.arange(1, self.num_vehicles * self.num_free_cities + 1, dtype=np.float32).reshape(
            self.num_vehicles, self.num_free_cities
        )
        logits = -(z[:, None, None] - centers[None, :, :]) ** 2 / (2.0 * self.params.sigma_assign ** 2)
        flat = logits.reshape(self.num_free_cities, self.num_vehicles * self.num_free_cities)
        flat -= flat.max(axis=1, keepdims=True)
        weights = np.exp(flat)
        weights /= weights.sum(axis=1, keepdims=True)
        return weights.reshape(self.num_free_cities, self.num_vehicles, self.num_free_cities)

    def _distance_term(self, a: np.ndarray) -> float:
        n = np.sum(a, axis=0)
        H_start = np.sum(a[:, :, 0] * self.dist_matrix[0, 1:, None])

        H_internal = 0.0
        D_city = self.dist_matrix[1:, 1:]
        different_city = 1.0 - np.eye(self.num_free_cities)
        for r in range(self.num_free_cities - 1):
            left = a[:, :, r].T
            right = a[:, :, r + 1].T
            edge_mass = left[:, :, None] * right[:, None, :]
            H_internal += np.sum(edge_mass * D_city[None, :, :] * different_city[None, :, :])

        H_return = 0.0
        d_to_depot = self.dist_matrix[1:, 0]
        empty_next = np.exp(-(n[:, 1:] ** 2) / (2.0 * self.params.sigma_empty ** 2))
        for r in range(self.num_free_cities - 1):
            last_mass = a[:, :, r] * empty_next[:, r][None, :]
            H_return += np.sum(last_mass * d_to_depot[:, None])
        H_return += np.sum(a[:, :, -1] * d_to_depot[:, None])

        return float(H_start + H_internal + H_return)

    def _collision_term(self, a: np.ndarray) -> float:
        n = np.sum(a, axis=0)
        pair_mass = 0.5 * (n ** 2 - np.sum(a ** 2, axis=0))
        return float(self.params.lambda_col * np.sum(pair_mass))

    def _gap_term(self, a: np.ndarray) -> float:
        n = np.sum(a, axis=0)
        prefix_before = np.cumsum(n, axis=1) - n
        required_before = np.arange(self.num_free_cities, dtype=np.float32)[None, :]
        return float(self.params.lambda_gap * np.sum(n[:, 1:] * (required_before[:, 1:] - prefix_before[:, 1:]) ** 2))

    def _vehicle_term(self, a: np.ndarray) -> float:
        if self.params.lambda_vehicle == 0.0:
            return 0.0
        n = np.sum(a, axis=0)
        return float(self.params.lambda_vehicle * np.sum((n[:, 0] - 1.0) ** 2))

    def evaluate_energy_from_z(self, z: np.ndarray) -> Dict[str, float]:
        a = self._soft_assignments(z)
        H_dist = self._distance_term(a)
        H_col = self._collision_term(a)
        H_gap = self._gap_term(a)
        H_vehicle = self._vehicle_term(a)
        H_total = H_dist + H_col + H_gap + H_vehicle
        return {
            "total": float(H_total),
            "dist": float(H_dist),
            "col": float(H_col),
            "gap": float(H_gap),
            "vehicle": float(H_vehicle),
        }

    # --- MONTAGEM DO OPERADOR LINEAR NO ESPAÇO DE HILBERT ---
    def _apply_basis_transform(self, vector: np.ndarray, U: np.ndarray, forward: bool) -> np.ndarray:
        psi = vector.reshape([self.cutoff] * self.num_free_cities)
        transform = U.conj().T if forward else U
        for mode in range(self.num_free_cities):
            psi = np.moveaxis(psi, mode, 0)
            shape = psi.shape
            psi = psi.reshape(self.cutoff, -1)
            psi = transform @ psi
            psi = psi.reshape(shape)
            psi = np.moveaxis(psi, 0, mode)
        return psi.reshape(-1)

    def _build_hamiltonian(self) -> Tuple[LinearOperator, Dict]:
        Z_local, X_local, x_eigs, z_eigs, Ux = self._local_z_operator()
        shape = [self.cutoff] * self.num_free_cities
        hilbert_dim = self.cutoff ** self.num_free_cities

        energy_grid = np.empty(shape, dtype=np.float32)
        for flat_idx in range(hilbert_dim):
            indices = np.unravel_index(flat_idx, shape)
            z = np.array([z_eigs[k] for k in indices], dtype=np.float32)
            energy_grid[indices] = self.evaluate_energy_from_z(z)["total"]

        diagonal = energy_grid.reshape(-1)

        def matvec(vector: np.ndarray) -> np.ndarray:
            psi_x = self._apply_basis_transform(vector, Ux, forward=True)
            Hpsi_x = diagonal * psi_x
            return self._apply_basis_transform(Hpsi_x, Ux, forward=False)

        H_operator = LinearOperator(
            shape=(hilbert_dim, hilbert_dim),
            matvec=matvec,
            rmatvec=matvec,
            dtype=np.complex128,
        )

        metadata = {
            "X_local": X_local,
            "Z_local": Z_local,
            "x_eigenvalues": x_eigs,
            "z_eigenvalues": z_eigs,
            "Ux": Ux,
            "energy_grid": energy_grid,
            "diagonal_energies": diagonal,
            "hilbert_dim": hilbert_dim,
        }

        return H_operator, metadata

    def compute_expectation(self, state_vector: np.ndarray) -> float:
        """Calcula o valor esperado de energia <psi|H|psi> para um vetor de estado normalizado."""
        psi = np.asarray(state_vector, dtype=np.complex128).reshape(-1)
        norm = np.linalg.norm(psi)
        if norm > 1e-12:
            psi = psi / norm
        H_psi = self.operator.matvec(psi)
        energy = np.real(np.vdot(psi, H_psi))
        return float(energy)

    def decode_routes_from_state(self, state_vector: np.ndarray) -> Dict[int, List[int]]:
        """Extrai as rotas a partir do estado na base X."""
        psi = np.asarray(state_vector, dtype=np.complex128).reshape(-1)
        Ux = self.metadata["Ux"]
        psi_x = self._apply_basis_transform(psi, Ux, forward=True)
        probs = np.abs(psi_x) ** 2
        
        best_idx = np.argmax(probs)
        shape = [self.cutoff] * self.num_free_cities
        indices = np.unravel_index(best_idx, shape)
        
        z_eigs = self.metadata["z_eigenvalues"]
        z_best = np.array([z_eigs[k] for k in indices], dtype=np.float32)
        a = self._soft_assignments(z_best)
        
        routes = {v: [0] for v in range(1, self.num_vehicles + 1)}
        for r in range(self.num_free_cities):
            for v in range(1, self.num_vehicles + 1):
                city_masses = a[:, v - 1, r]
                best_city = int(np.argmax(city_masses)) + 1
                if city_masses[best_city - 1] > 0.1:
                    routes[v].append(best_city)
        
        for v in routes:
            routes[v].append(0)
            
        return routes

    def compute_metrics(
        self, 
        state_vector: np.ndarray, 
        decoder: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Calcula as métricas do estado quântico simulado."""
        routes = self.decode_routes_from_state(state_vector)
        
        total_dist = 0.0
        cap_violation = 0.0
        is_feasible = True
        
        for v, r in routes.items():
            v_cap = self.capacities[v - 1]
            load = sum(self.demands[c] for c in r)
            if load > v_cap:
                cap_violation += (load - v_cap)
                is_feasible = False
                
            for k in range(len(r) - 1):
                total_dist += float(self.dist_matrix[r[k], r[k + 1]])
                
        energy = self.compute_expectation(state_vector)
        
        return {
            "total_cost": float(total_dist + self.params.lambda_vehicle * cap_violation),
            "route_distance": float(total_dist),
            "capacity_violation_magnitude": float(cap_violation),
            "is_feasible": is_feasible,
            "energy": energy
        }

    # Retrocompatibilidade com as chamadas legadas de main.py e solver.py
    def discretize_quadratures(self, x_vals: List[float], p_vals: List[float], decoder: Optional[Any] = None) -> Tuple[List[int], List[int]]:
        return [int(np.round(x)) for x in x_vals], [int(np.round(p)) for p in p_vals]

    def decode_routes(self, x_vals: List[float], p_vals: List[float], decoder: Optional[Any] = None) -> Dict[int, List[int]]:
        z_dummy = np.array(x_vals, dtype=np.float32)
        a = self._soft_assignments(z_dummy)
        routes = {v: [0] for v in range(1, self.num_vehicles + 1)}
        for r in range(self.num_free_cities):
            for v in range(1, self.num_vehicles + 1):
                best_city = int(np.argmax(a[:, v - 1, r])) + 1
                routes[v].append(best_city)
        for v in routes:
            routes[v].append(0)
        return routes

    def compute_composite_score(self, exact_cost: float, x_vals: List[float], p_vals: List[float], decoder: Optional[Any] = None) -> float:
        return float(exact_cost)