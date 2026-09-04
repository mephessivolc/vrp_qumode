# vrp/main.py
import sys
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import time
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from graphs import GraphBuilder
from brute_force import BruteForce
from metrics import ExperimentResult
from logger import ExperimentLogger
from utils import format_timespan, print_experiment_summary
from path import get_images_path

from qumodes.hamiltonian import Hamiltonian, HamiltonianParams
from qumodes.solver import Solver
from qumodes.decoders import HungarianDecoder, ArgmaxDecoder, SoftmaxAnnealingDecoder
from qumodes.schedulers import (
    AugmentedLagrangianScheduler, 
    ExponentialPenaltyScheduler, 
    AdaptiveConstraintScheduler
)


def plot_phase_space(
    cont_x: list, 
    cont_p: list, 
    disc_x: list, 
    disc_p: list, 
    output_path: Path
):
    plt.figure(figsize=(8, 5))
    num_cities = len(cont_x)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_cities))
    
    for i in range(num_cities):
        city_id = i + 1
        plt.scatter(
            cont_x[i], cont_p[i], 
            color=colors[i], s=120, zorder=3, 
            label=f'Cidade {city_id} (x={cont_x[i]:.2f}, p={cont_p[i]:.2f})'
        )
        plt.scatter(
            disc_x[i], disc_p[i], 
            color=colors[i], marker='x', s=100, linewidths=2, zorder=4,
            label=f'Cidade {city_id} (Discreto)'
        )
        plt.plot([cont_x[i], disc_x[i]], [cont_p[i], disc_p[i]], color=colors[i], linestyle=':', alpha=0.6)

    plt.title("Atribuição de Quadraturas no Espaço de Hilbert Truncado")
    plt.xlabel("Coordenada Mapeada de Rota (z)")
    plt.ylabel("Veículo Virtual")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run(
    n_cities: int = 3,
    num_vehicles: int = 2,
    vehicle_capacity: Union[float, int, List[float], np.ndarray] = 10.0,
    demands: Optional[Union[np.ndarray, List[float]]] = None,
    demand_range: Tuple[float, float] = (1.0, 5.0),
    layers: int = 1,
    cutoff: int = 5,
    maxiter: int = 100,
    optimizer_method: str = "ADAM",
    decoder_name: str = "hungarian",
    scheduler_name: str = "augmented_lagrangian",
    lr: float = 0.01,
    use_warm_start: bool = False,
    penalty_gamma: float = 1.05,
    plateau_patience: int = 15,
    noise_scale: float = 0.02,
    graph_type: str = "random", 
    device: str = "cpu",
    seed: int = 42,
    save_outputs: bool = True,
    variable_type_path: str = "QUMODES",
    sub_folder: Union[str, None] = None
) -> dict:
    str_problem_type = "TSP" if num_vehicles == 1 else "VRP"
    logger = ExperimentLogger(
        variable_type=variable_type_path, 
        problem_type=str_problem_type,
        sub_folder=sub_folder
    )

    total_nodes = n_cities + 1
    gb = GraphBuilder(
        n=total_nodes, 
        seed=seed, 
        graph_type=graph_type, 
        demands=demands,
        demand_range=demand_range,
        logger=logger, 
        variable_type_path=variable_type_path,
        sub_folder=sub_folder
    )

    # 1. GROUND TRUTH (Força Bruta)
    t0 = time.time()
    solver_exato = BruteForce(
        gb.matrix, 
        num_vehicles=num_vehicles, 
        capacities=vehicle_capacity, 
        demands=gb.demands
    )
    exact_cost, exact_route = solver_exato.solve()
    t_exact = time.time() - t0

    # 2. NOVO HAMILTONIANO HERMITIANO
    h_params = HamiltonianParams(
        sigma_assign=0.30,
        sigma_empty=0.40,
        lambda_col=20.0,
        lambda_gap=20.0,
        lambda_vehicle=15.0
    )
    
    hamiltonian = Hamiltonian(
        dist_matrix=gb.matrix, 
        num_vehicles=num_vehicles, 
        cutoff=cutoff,
        params=h_params,
        vehicle_capacity=vehicle_capacity,
        demands=gb.demands
    )

    vqe_solver = Solver(
        hamiltonian=hamiltonian,
        layers=layers,
        device=device
    )

    # 3. EXECUÇÃO DO SOLVER VQE
    t0 = time.time()
    warm_start_routes = exact_route if use_warm_start else None

    vqe_res = vqe_solver.solve(
        warm_start_routes=warm_start_routes,
        maxiter=maxiter,
        optimizer_method=optimizer_method,
        lr=lr,
        exact_cost=exact_cost,
        seed=seed
    )
    t_vqe = time.time() - t0

    vqe_cost = float(vqe_res["best_cost"])
    nfev = len(vqe_res["cost_history"])
    approx_ratio = float(exact_cost / vqe_cost) if vqe_cost != 0 else 0.0

    print("\n" + "="*70)
    print(f"             RESULTADOS FINAIS DO {str_problem_type} (HAMILTONIANO HERMITIANO)             ")
    print("="*70)
    print(f"Custo Exato (Ground Truth)   : {exact_cost:.4f}")
    print(f"Custo Otimizado (VQE)        : {vqe_cost:.4f}")
    print(f"Viabilidade da Solução       : {'SIM' if vqe_res['is_feasible'] else 'NÃO'}")
    print(f"Razão de Aproximação         : {approx_ratio:.4f}")
    print("Decodificação de Rotas:")
    for veh, r in vqe_res["routes"].items():
        print(f"  • Veículo {veh}: {' -> '.join(map(str, r))}")
    print("="*70 + "\n")

    return vqe_res


if __name__ == "__main__":
    run(
        n_cities=5,
        num_vehicles=3,
        vehicle_capacity=8.0,
        cutoff=4,
        maxiter=5,
        use_warm_start=False,
        seed=42,
        save_outputs=False
    )