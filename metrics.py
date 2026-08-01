# metrics.py
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Union, Any
import json
import numpy as np


@dataclass
class ExperimentResult:
    """
    Contrato de dados padronizado para armazenar os resultados de simulações TSP e VRP.
    """
    # --- Metadados do Experimento ---
    experiment_id: str
    problem_type: str        # "TSP" ou "VRP"
    timestamp: str
    seed: int

    # --- Parâmetros de Entrada / Hiperparâmetros ---
    n_cities: int
    num_vehicles: int
    p_layers: int
    max_iter: int
    momentum_mass: float     # Massa M do operador momento no espaço contínuo
    lmbda: float             # Constante de penalidade no Hamiltoniano
    lmbda_empty: float

    # --- Solução Exata (Ground Truth via Força Bruta) ---
    exact_cost: float
    exact_route: Union[List[int], Dict[int, List[int]]]
    exact_time_sec: float
    ground_state_energy: float

    # --- Solução Quântica (QAOA ou VQE) ---
    solver_name: str         # "CV-QAOA" ou "CV-VQE"
    quantum_cost: float
    quantum_route: Union[List[int], Dict[int, List[int]]]
    quantum_time_sec: float
    approx_ratio: float      # exact_cost / quantum_cost
    success_probability: float
    evaluations_count: int   # Número de chamadas do circuito (nfev)

    # --- Históricos para Gráficos ---
    optimal_params: List[float] = field(default_factory=list)
    cost_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converte a dataclass para dicionário com tratamento para tipos do NumPy."""
        d = asdict(self)
        return json.loads(
            json.dumps(
                d,
                default=lambda o: float(o) if isinstance(o, (np.float32, np.float64))
                else int(o) if isinstance(o, (np.int32, np.int64))
                else list(o) if isinstance(o, np.ndarray)
                else str(o)
            )
        )