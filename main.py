# vrp/main.py
import sys
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any

# --- RESOLUÇÃO DE IMPORTS ---
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Importações da raiz do projeto
from graphs import GraphBuilder
from brute_force import BruteForce
from metrics import ExperimentResult
from logger import ExperimentLogger
from utils import format_timespan, print_experiment_summary
from path import get_images_path

# Módulos do VRP
from vrp.hamiltonian import Hamiltonian 
from vrp.solver import Solver

def plot_phase_space(
    cont_x: list, 
    cont_p: list, 
    disc_x: list, 
    disc_p: list, 
    output_path: Path
):
    """Gera o gráfico do Espaço de Fase (x, p) mostrando a transição Contínuo -> Discreto."""
    plt.figure(figsize=(8, 5))
    num_cities = len(cont_x)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, num_cities))
    
    for i in range(num_cities):
        city_id = i + 1
        # Ponto contínuo (medido)
        plt.scatter(
            cont_x[i], cont_p[i], 
            color=colors[i], s=120, zorder=3, 
            label=f'Cidade {city_id} (Contínuo: x={cont_x[i]:.2f}, p={cont_p[i]:.2f})'
        )
        # Ponto discreto (arredondado)
        plt.scatter(
            disc_x[i], disc_p[i], 
            color=colors[i], marker='x', s=100, linewidths=2, zorder=4,
            label=f'Cidade {city_id} (Discreto: x={disc_x[i]}, p={disc_p[i]})'
        )
        # Linha conectando contínuo ao discreto
        plt.plot([cont_x[i], disc_x[i]], [cont_p[i], disc_p[i]], color=colors[i], linestyle=':', alpha=0.6)

    plt.title("Atribuição no Espaço de Fase: Posição (x) vs Momento (p)")
    plt.xlabel("Posição / Ordem Temporal (x)")
    plt.ylabel("Momento / Identidade do Veículo (p)")
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
    reps: int = 1,
    maxiter: int = 500,
    lmbda: Optional[float] = None,
    lmbda_cap: Optional[float] = None,
    optimizer_method: str = "ADAM",
    lr: float = 0.01,
    graph_type: str = "random", 
    device: str = "cpu",
    seed: int = 42,
    save_outputs: bool = True,
    variable_type_path: str = "QUMODES",
    sub_folder: Union[str, None] = None
) -> dict:
    """Orquestrador principal para simulações e experimentos do CVRP em CV-VQE."""
    str_problem_type = "TSP" if num_vehicles == 1 else "VRP"
    logger = ExperimentLogger(
        variable_type=variable_type_path, 
        problem_type=str_problem_type,
        sub_folder=sub_folder
    )
    logger.info(
        f"Iniciando Experimento {variable_type_path} {str_problem_type} (N={n_cities}, Vehicles={num_vehicles}, "
        f"Cap={vehicle_capacity}, Layers={layers}, Reps={reps}, Opt={optimizer_method}, LR={lr}, Device={device.upper()}, MaxIter={maxiter})"
    )

    cap_info = vehicle_capacity if isinstance(vehicle_capacity, (int, float)) else "het"
    constructor_info_name = (
        f"N{n_cities}_V{num_vehicles}_C{cap_info}_L{layers}_R{reps}_O{optimizer_method.lower()}_LR={lr}_M{maxiter}_G{graph_type.lower()}"
    )

    # 1. GERAÇÃO DO GRAFO E DEMANDAS (Inclui Depósito no índice 0)
    logger.info("1. Gerando grafo e demandas das cidades (Depósito + Cidades)...")
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

    # 2. GROUND TRUTH (Força Bruta Exata)
    logger.info("2. Executando Busca Exaustiva Clássica (Ground Truth)...")
    t0 = time.time()
    solver_exato = BruteForce(
            gb.matrix, 
            num_vehicles=num_vehicles, 
            capacities=vehicle_capacity, 
            demands=gb.demands
        )
    
    exact_cost, exact_route = solver_exato.solve()
    t_exact = time.time() - t0
    logger.info(f"   ► Custo Exato: {exact_cost:.4f} | Tempo: {format_timespan(t_exact)}")

    # 3. HAMILTONIANO DO CVRP (Insere Capacidade e Demandas)
    logger.info("3. Construindo operadores do Hamiltoniano CV para CVRP...")
    hamiltonian = Hamiltonian(
        dist_matrix=gb.matrix, 
        num_vehicles=num_vehicles, 
        vehicle_capacity=vehicle_capacity,
        demands=gb.demands,
        lmbda=lmbda, 
        lmbda_cap=lmbda_cap
    )
    vqe_solver = Solver(
        hamiltonian=hamiltonian,
        layers=layers,
        reps=reps,
        device=device
    )

    # 4. SOLVER VQE
    logger.info(f"4. Otimizando circuito VQE ({optimizer_method} | lr={lr} | device={device.upper()})...")
    t0 = time.time()
    
    vqe_res = vqe_solver.solve(
        maxiter=maxiter,
        optimizer_method=optimizer_method,
        lr=lr,
        seed=seed
    )
    t_vqe = time.time() - t0

    vqe_cost = float(vqe_res["best_cost"])
    nfev = len(vqe_res["cost_history"])
    approx_ratio = float(exact_cost / vqe_cost) if vqe_cost != 0 else 0.0

    # 5. APRESENTAÇÃO LEGÍVEL PARA HUMANOS (INTERPRETAÇÃO FINAL)
    cont_x = vqe_res["continuous_x"]
    cont_p = vqe_res["continuous_p"]
    disc_x = vqe_res["disc_x"]
    disc_p = vqe_res["disc_p"]
    routes = vqe_res["routes"]

    print("\n" + "="*70)
    print(f"                     RESULTADOS FINAIS DO {str_problem_type} (CVRP)                     ")
    print("="*70)
    print(f"Custo Exato (Ground Truth): {exact_cost:.4f}")
    print(f"Custo Otimizado (CV-VQE)  : {vqe_cost:.4f}")
    print(f"Razão de Aproximação       : {approx_ratio:.4f}")
    print(f"Demandas das Cidades      : {gb.demands.tolist()}")
    print(f"Capacidades dos Veículos  : {hamiltonian.capacities.tolist()}")
    print("-" * 70)
    print("Medições das Quadraturas no Espaço de Fase:")
    for i in range(n_cities):
        city_id = i + 1
        print(f"  • Cidade {city_id} (Demanda={gb.demands[city_id]}) -> "
              f"x_cont: {cont_x[i]:6.3f} | p_cont: {cont_p[i]:6.3f}  ===>  "
              f"Passo Temporal (x): {disc_x[i]} | Veículo (p): {disc_p[i]}")
    print("-" * 70)
    print("Decodificação Discreta das Rotas dos Veículos:")
    for veh, r in routes.items():
        route_str = " -> ".join(map(str, r))
        v_load = sum(gb.demands[node] for node in r)
        print(f"  • Veículo {veh} (Carga={v_load:.1f}/{hamiltonian.capacities[veh-1]}): {route_str}")
    print("="*70 + "\n")

    # 6. ESTRUTURAÇÃO DOS RESULTADOS E ARTEFATOS
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{str_problem_type}_{constructor_info_name}_{now_str}"

    opt_params = vqe_res.get("opt_params", None)
    if isinstance(opt_params, np.ndarray):
        opt_params = opt_params.tolist()

    experiment_res = ExperimentResult(
        experiment_id=exp_id,
        problem_type=str_problem_type,
        variable_type=variable_type_path,
        timestamp=now_str,
        seed=seed,
        n_cities=n_cities,
        num_vehicles=num_vehicles,
        vehicle_capacity=hamiltonian.capacities.tolist(),  # Lista com as capacidades dos veículos
        demands=gb.demands.tolist(),                        # Lista com as demandas das cidades
        p_layers=layers,
        max_iter=maxiter,
        momentum_mass=1.0,
        lmbda=hamiltonian.lmbda,
        lmbda_cap=hamiltonian.lmbda_cap,
        exact_cost=exact_cost,
        exact_route=exact_route,
        exact_time_sec=t_exact,
        ground_state_energy=exact_cost,
        solver_name="CV-VQE",
        quantum_cost=vqe_cost,
        quantum_route=routes,
        quantum_time_sec=t_vqe,
        approx_ratio=approx_ratio,
        success_probability=1.0,
        evaluations_count=nfev,
        optimal_params=opt_params,
        cost_history=vqe_res["cost_history"]
    )

    # Anexa o histórico da perda contínua no dicionário exportado
    exp_dict = experiment_res.to_dict()
    exp_dict["continuous_loss_history"] = vqe_res["continuous_loss_history"]

    print_experiment_summary(
        problem_type=str_problem_type.lower(),
        n_cities=n_cities,
        exact_cost=exact_cost,
        exact_time=t_exact,
        quantum_cost=vqe_cost,
        quantum_time=t_vqe,
        evals=nfev
    )

    if save_outputs:
        figures_dir = get_images_path(
            variable_type=variable_type_path,
            problem_type=str_problem_type.lower(),
            sub_folder=sub_folder
        )
        
        # a) Salva JSON/CSV via Logger
        logger.save_experiment(experiment_res)

        # b) Plot do Grafo com Rotas Decodificadas
        gb.plot_graph_and_route(
            solution_vector=routes,
            prefix=f"vqe_{constructor_info_name}"
        )

        # c) Plot da Curva de Convergência (Dual Axis: Perda Contínua Suave vs. Custo Discreto)
        conv_path = figures_dir / f"convergence_{constructor_info_name}.png"
        fig, ax1 = plt.subplots(figsize=(9, 5))

        color = 'tab:blue'
        ax1.set_xlabel('Iterações / Passo do Gradiente', fontsize=11)
        ax1.set_ylabel('Perda Contínua (Loss Suave)', color=color, fontsize=11)
        ax1.plot(vqe_res["continuous_loss_history"], color=color, linewidth=1.8, label='Perda Contínua (TF Loss)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax2 = ax1.twinx()  
        color = 'tab:green'
        ax2.set_ylabel(f'Custo Discreto do {str_problem_type} (H_total)', color=color, fontsize=11)
        ax2.plot(vqe_res["cost_history"], color=color, linestyle='--', linewidth=1.8, alpha=0.85, label='Custo Discreto (Rotas)')
        ax2.axhline(y=exact_cost, color='red', linestyle=':', label=f'Ground Truth ({exact_cost:.2f})')
        ax2.tick_params(axis='y', labelcolor=color)

        # Combina legendas dos dois eixos
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.title(f"Convergência CV-VQE - {str_problem_type} (N={n_cities}, Veículos={num_vehicles})", fontsize=12)
        fig.tight_layout()
        plt.savefig(conv_path, dpi=300)
        plt.close()

        # d) Plot do Espaço de Fase (x, p)
        phase_path = figures_dir / f"phase_space_{constructor_info_name}.png"
        plot_phase_space(cont_x, cont_p, disc_x, disc_p, phase_path)

        logger.info(f"Todos os artefatos e gráficos foram salvos na pasta {figures_dir}")

    return exp_dict


if __name__ == "__main__":
    import os
    # Silencia logs do C++ do TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    from itertools import product

    maxiter = 500
    vehicles = [1,2,3,4]
    l_params = [None,10,50,75,100]
    
    # Bateria de testes de Penalização
    for comb in list(product(vehicles, l_params)):
        vehicle, l_param = comb
        run(
            n_cities=5,
            num_vehicles=vehicle,
            vehicle_capacity=10.0,
            layers=2,
            maxiter=maxiter,
            lmbda=l_param,
            lmbda_cap=None,
            optimizer_method="ADAM",
            lr=0.01,
            graph_type="random",
            device="cuda",
            seed=42,
            save_outputs=True,
            sub_folder="PENALIZATION-LAST"
        )
        

    # Bateria de testes Principais (Tamanho da Cidade, Veículos, Camadas)
    city = [3, 4, 5]
    vehicles = [1, 2, 3]
    layers = [2, 1, 3]
    for comb in product(city, vehicles, layers):
        c, v, l = comb
        run(
            n_cities=c,
            num_vehicles=v,
            vehicle_capacity=8.0,
            layers=l,
            maxiter=maxiter,
            optimizer_method="ADAM",
            lr=0.01,
            graph_type="random",
            device="cuda",
            seed=42,
            save_outputs=True,
            sub_folder="MAIN-LAST"
        )

    # Bateria de testes por Topologia de Grafo
    vehicles = [1, 3]
    graph_type = ["euclidean", "circle", "grid", "clustered"]
    for comb in list(product(vehicles, graph_type)):
        v, g_type = comb
        run(
            n_cities=5,
            num_vehicles=v,
            vehicle_capacity=10.0,
            layers=2,
            maxiter=maxiter,
            lmbda=10.0,
            optimizer_method="ADAM",
            lr=0.01,
            graph_type=g_type,
            device="cuda",
            seed=42,
            save_outputs=True,
            sub_folder="TOPOLOGY-LAST"
        )