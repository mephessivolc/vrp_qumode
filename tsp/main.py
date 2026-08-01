# tsp/main.py
import sys
from pathlib import Path

# --- RESOLUÇÃO DE IMPORTS (Adiciona o diretório 'src/' ao sys.path) ---
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Importações da raiz (src/)
from graphs import GraphBuilder
from brute_force import BruteForce
from metrics import ExperimentResult
from logger import ExperimentLogger
from utils import format_timespan, print_experiment_summary
from path import get_images_path, get_results_path

# Módulos específicos do TSP
from tsp.hamiltonian import Hamiltonian
from tsp.solver import Solver as VqeSolver


def run(
    n_cities: int = 3,
    lmbda: float = 50.0,
    lmbda_empty: float = 20.0,
    layers: int = 1,
    maxiter: int = 50,
    optimizer_method: str = "COBYLA",
    seed: int = 42,
    save_outputs: bool = True
) -> dict:
    """
    Orquestrador e ponto de entrada para execução dos experimentos do TSP.
    Pode ser importado por outros scripts (ex: main principal da raiz).
    """
    logger = ExperimentLogger(problem_type="TSP")
    logger.info(f"Iniciando Experimento TSP (N={n_cities}, Layers={layers}, MaxIter={maxiter})")

    constructor_info_name = f"N{n_cities}_Layers{layers}_MaxIter={maxiter}"

    # 1. GERAÇÃO DO GRAFO
    logger.info("1. Gerando matriz de adjacência do grafo...")
    gb = GraphBuilder(n=n_cities, seed=seed, logger=logger)

    # 2. GROUND TRUTH (Força Bruta Clássica)
    logger.info("2. Executando Busca Exaustiva Clássica (Ground Truth)...")
    t0 = time.time()
    solver_exato = BruteForce(gb.matrix, num_vehicles=1)
    exact_cost, exact_route = solver_exato.solve()
    t_exact = time.time() - t0
    logger.info(f"   ► Custo Exato: {exact_cost:.4f} | Tempo: {format_timespan(t_exact)}")

    # 3. CONSTRUÇÃO DO HAMILTONIANO
    logger.info("3. Construindo operadores do Hamiltoniano CV...")
    hamiltonian = Hamiltonian(gb.matrix, lmbda=lmbda)
    _, _, H_tot = hamiltonian.get_hamiltonian_matrices()
    
    e0_exact = float(np.min(np.real(np.linalg.eigvalsh(H_tot))))
    logger.info(f"   ► Autovalor Fundamental H_total (E0): {e0_exact:.4f}")

    # 4. EXECUÇÃO DO SOLVER VQE
    logger.info(f"4. Executando CV-VQE Solver ({optimizer_method})...")
    t0 = time.time()
    vqe_solver = VqeSolver(hamiltonian_builder=hamiltonian, layers=layers)
    vqe_res = vqe_solver.solve(maxiter=maxiter, optimizer_method=optimizer_method, seed=seed)
    t_vqe = time.time() - t0

    vqe_cost = float(vqe_res["best_energy"])
    nfev = len(vqe_res["history"])
    prob = float(vqe_res.get("probability", 1.0))
    approx_ratio = float(exact_cost / vqe_cost) if vqe_cost != 0 else 0.0

    logger.info(f"   ► VQE Custo Otimizado: {vqe_cost:.4f} | Tempo: {format_timespan(t_vqe)}")

    # 5. ESTRUTURAÇÃO DO RESULTADO (ExperimentResult)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"TSP_{constructor_info_name}_{now_str}"

    opt_params = vqe_res["opt_params"]
    if isinstance(opt_params, np.ndarray):
        opt_params = opt_params.tolist()

    experiment_res = ExperimentResult(
        # Metadados
        experiment_id=exp_id,
        problem_type="TSP",
        timestamp=now_str,
        seed=seed,
        
        # Parâmetros de Entrada
        n_cities=n_cities,
        num_vehicles=1,
        p_layers=layers,
        max_iter=maxiter,
        momentum_mass=1.0,  # Valor padrão do operador momento contínuo
        lmbda=lmbda,
        lmbda_empty=lmbda_empty,

        # Solução Exata
        exact_cost=exact_cost,
        exact_route=exact_route,
        exact_time_sec=t_exact,
        ground_state_energy=e0_exact,

        # Solução Quântica
        solver_name="CV-VQE",
        quantum_cost=vqe_cost,
        quantum_route=vqe_res["solution_vector"],
        quantum_time_sec=t_vqe,
        approx_ratio=approx_ratio,
        success_probability=prob,
        evaluations_count=nfev,

        # Históricos
        optimal_params=opt_params,
        cost_history=vqe_res["history"]
    )

    # Exibe resumo formatado no terminal
    print_experiment_summary(
        problem_type="tsp",
        n_cities=n_cities,
        exact_cost=exact_cost,
        exact_time=t_exact,
        quantum_cost=vqe_cost,
        quantum_time=t_vqe,
        evals=nfev
    )

    # 6. SALVAMENTO DE ARTEFATOS E GRÁFICOS (result/tsp/)
    if save_outputs:
        figures_dir = Path(logger.get_figures_dir("TSP"))

        # a) Salva via ExperimentLogger (JSON e registro no CSV acumulativo)
        logger.save_experiment(experiment_res)

        # b) Plot do Grafo e Trajeto TSP
        gb.plot_graph_and_route(
            solution_vector=vqe_res["solution_vector"],
            prefix=f"vqe_{constructor_info_name}"
        )

        # c) Plot e Salvamento da Curva de Convergência
        plt.figure(figsize=(8, 4.5))
        plt.axhline(y=exact_cost, color='r', linestyle='--', label=f'Ground Truth Exato ({exact_cost:.2f})')
        plt.plot(vqe_res["history"], label='Convergência CV-VQE', color='green', marker='o', alpha=0.7)
        plt.title(f"Convergência VQE - TSP (N={n_cities}, Camadas={layers})")
        plt.xlabel("Avaliações de Custo (Iterações)")
        plt.ylabel("Energia Esperada <H_total>")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"convergence_{constructor_info_name}.png", dpi=300)
        plt.close()

        logger.info("Artefatos salvos com sucesso na pasta 'result/tsp/'")

    return experiment_res.to_dict()


if __name__ == "__main__":
    run(
        n_cities=3,
        lmbda=50.0,
        layers=1,
        maxiter=40,
        optimizer_method="COBYLA",
        seed=42,
        save_outputs=True
    )