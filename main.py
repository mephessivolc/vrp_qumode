import matplotlib.pyplot as plt
from typing import Optional, Tuple
import numpy as np

from qumodes.ansatz import CircuitConfig
from qumodes.hamiltonian import HamiltonianParams
from qumodes.solver import ProblemInstance, VQESolver

from graphs import Graph


def run_experiment(
    C: int = 3,  # Número de Cidades
    V: int = 2,  # Número de Veículos
    max_iter: int = 100,
    num_layer: int = 2,
    method: Optional[str] = "COBYLA",
    Q_val: float = 6.0,  # Capacidade individual dos veículos
    h_params: Optional[HamiltonianParams] = None, 
    is_warm_start: bool = True,
    demand_range: Tuple[int, int] = (1, 5),
    warm_start_coords: Optional[np.ndarray] = None,
    seed: Optional[int] = 42,
):
    # 1. Configuração dos Parâmetros de Entrada da Simulação
    g = Graph(
        C=C,
        V=V,
        is_warm_start=is_warm_start,
        warm_start_coords=warm_start_coords,
        demand_range=demand_range,
        seed=seed,
    )

    # 2. Configuração dos Hiperparâmetros do Hamiltoniano (com fallback padrão)
    if h_params is None:
        h_params = HamiltonianParams(
            sigma_assign=0.30,
            sigma_empty=0.40,
            lambda_col=25.0,
            lambda_gap=20.0,
            lambda_cap= 0.0 if V == 1 else 18.0,
            alpha2=1.0,
            alpha4=1.0,
        )

    D, demands, coords = g.graph
    Q = np.array([Q_val] * V, dtype=np.float64)

    instance = ProblemInstance(C=C, V=V, D=D, demands=demands, Q=Q)
    circuit_config = CircuitConfig(num_qumodes=C, num_layers=num_layer)

    # 3. Inicialização e Execução do Solver
    print(f"--- Iniciando VQE para CVRP ({C} Cidades, {V} Veículos) ---")
    solver = VQESolver(
        instance=instance,
        circuit_config=circuit_config,
        hamiltonian_params=h_params,
        cutoff=8,
    )

    if method is None:
        method = "COBYLA"

    metrics = solver.solve(method=method, maxiter=max_iter)

    # 4. Exibição e Análise de Métricas
    print("\n=== Resultados da Otimização ===")
    print(f"Método Utilizado: {method}")
    print(f"Tempo de Execução: {metrics.execution_time_seconds:.2f} s")
    print(f"Avaliações da Função de Custo: {metrics.total_evaluations}")
    print(f"Energia Final Obtida: {metrics.final_energy:.4f}")
    print("\nDetalhamento dos Termos de Energia:")
    for key, val in metrics.energy_components.items():
        print(f"  - H_{key}: {val:.4f}")

    # Plotagem da curva de convergência
    plt.figure(figsize=(8, 4))
    plt.plot(metrics.cost_history, label="Energia Total $\\langle H \\rangle$")
    plt.xlabel("Iterações")
    plt.ylabel("Energia")
    plt.title(f"Curva de Convergência VQE ({method}) - CVRP")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()