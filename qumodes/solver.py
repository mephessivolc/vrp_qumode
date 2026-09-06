import matplotlib.pyplot as plt
from typing import Optional, Tuple
import numpy as np

from ansatz import CircuitConfig
from graphs import Graph
from hamiltonian import HamiltonianParams
from solver import ProblemInstance, VQESolver


def run_experiment(
    C: int = 3,  # Número de cidades
    V: int = 2,  # Número de veículos
    max_iter: int = 100,
    num_layer: int = 2,
    method: Optional[str] = "COBYLA",
    Q_val: float = 6.0,  # Capacidade individual dos veículos
    is_warm_start: bool = True,
    demand_range: Tuple[int, int] = (1, 5),
    warm_start_coords: Optional[np.ndarray] = None,
    seed: Optional[int] = 42,
):
    # 1. Instanciação e desempacotamento do Grafo
    g = Graph(
        C=C,
        V=V,
        is_warm_start=is_warm_start,
        warm_start_coords=warm_start_coords,
        demand_range=demand_range,
        seed=seed,
    )

    D, demands, coords = g.graph
    Q = np.array([Q_val] * V, dtype=np.float64)

    instance = ProblemInstance(C=C, V=V, D=D, demands=demands, Q=Q)

    # Desativa penalidade de capacidade para TSP (V = 1)
    lambda_cap = 0.0 if V == 1 else 18.0

    # 2. Configuração dos Hiperparâmetros
    h_params = HamiltonianParams(
        sigma_assign=0.30,
        sigma_empty=0.40,
        lambda_col=25.0,
        lambda_gap=20.0,
        lambda_cap=lambda_cap,
        alpha2=1.0,
        alpha4=1.0,
    )

    circuit_config = CircuitConfig(num_qumodes=C, num_layers=num_layer)

    # 3. Solver VQE
    print(f"--- Iniciando VQE ({C} Cidades, {V} Veículo(s)) ---")
    solver = VQESolver(
        instance=instance,
        circuit_config=circuit_config,
        hamiltonian_params=h_params,
        cutoff=8,
    )

    if method is None:
        method = "COBYLA"
        
    metrics = solver.solve(method=method, maxiter=max_iter)

    # 4. Métricas e Visualização
    print("\n=== Resultados da Otimização ===")
    print(f"Tempo de Execução: {metrics.execution_time_seconds:.2f} s")
    print(f"Avaliações da Função de Custo: {metrics.total_evaluations}")
    print(f"Energia Final Obtida: {metrics.final_energy:.4f}")
    print("\nDetalhamento dos Termos de Energia:")
    for key, val in metrics.energy_components.items():
        print(f"  - H_{key}: {val:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(metrics.cost_history, label="Energia Total $\\langle H \\rangle$")
    plt.xlabel("Iterações")
    plt.ylabel("Energia")
    plt.title("Curva de Convergência VQE - CVRP")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()