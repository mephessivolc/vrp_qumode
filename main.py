# vrp/main.py
import sys
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any

# --- RESOLUÇÃO DE IMPORTS ---
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import time
import os
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

# Módulos do VRP (Continuous-Variable Qumodes)
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
    """Gera o gráfico do Espaço de Fase (x, p) mostrando a transição Contínuo -> Discreto."""
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
        if i < len(disc_x) and i < len(disc_p):
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


def _get_decoder_instance(decoder_name: str) -> Any:
    """Instancia o decodificador com base na estratégia selecionada."""
    name = decoder_name.lower()
    if name == "hungarian":
        return HungarianDecoder()
    elif name == "softmax":
        return SoftmaxAnnealingDecoder()
    elif name == "argmax":
        return ArgmaxDecoder()
    else:
        return HungarianDecoder()


def _get_scheduler_instance(scheduler_name: str, penalty_gamma: float) -> Optional[Any]:
    """Instancia o agendador de penalidades com base na estratégia selecionada."""
    name = scheduler_name.lower()
    if name == "augmented_lagrangian":
        return AugmentedLagrangianScheduler()
    elif name == "exponential":
        return ExponentialPenaltyScheduler(gamma=penalty_gamma)
    elif name == "adaptive":
        return AdaptiveConstraintScheduler()
    elif name in ["none", "fixed"]:
        return None
    else:
        return AugmentedLagrangianScheduler()


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
    """Orquestrador principal para simulações do CVRP em CV-VQE com o Hamiltoniano Hermitiano."""
    str_problem_type = "TSP" if num_vehicles == 1 else "VRP"
    logger = ExperimentLogger(
        variable_type=variable_type_path, 
        problem_type=str_problem_type,
        sub_folder=sub_folder
    )

    logger.info(
        f"Iniciando Experimento {variable_type_path} {str_problem_type} (N={n_cities}, Veículos={num_vehicles}, "
        f"Cap={vehicle_capacity}, Layers={layers}, Cutoff={cutoff}, Opt={optimizer_method}, Decoder={decoder_name.upper()}, "
        f"Scheduler={scheduler_name.upper()}, LR={lr}, WarmStart={use_warm_start}, Device={device.upper()}, MaxIter={maxiter})"
    )

    constructor_info_name = (
        f"N{n_cities}_V{num_vehicles}_L{layers}_C{cutoff}_O{optimizer_method.lower()}_"
        f"D{decoder_name.lower()}_S{scheduler_name.lower()}_WS{use_warm_start}_M{maxiter}_G{graph_type.lower()}"
    )

    # 1. GERAÇÃO DO GRAFO E DEMANDAS
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
    logger.info("Executando Busca Exaustiva Clássica (Ground Truth)...")
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

    # 3. HAMILTONIANO HERMITIANO E SOLVER VQE
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

    decoder = _get_decoder_instance(decoder_name)
    penalty_scheduler = _get_scheduler_instance(scheduler_name, penalty_gamma)

    # 4. EXECUÇÃO DA OTIMIZAÇÃO VQE
    logger.info(f"Otimizando circuito VQE ({optimizer_method} | Decoder={decoder_name.upper()} | Scheduler={scheduler_name.upper()})...")
    t0 = time.time()
    warm_start_routes = exact_route if use_warm_start else None

    # Tenta passar decodificador e agendador se aceito pela assinatura do solver
    try:
        vqe_res = vqe_solver.solve(
            warm_start_routes=warm_start_routes,
            maxiter=maxiter,
            optimizer_method=optimizer_method,
            lr=lr,
            penalty_gamma=penalty_gamma,
            penalty_scheduler=penalty_scheduler,
            decoder=decoder,
            plateau_patience=plateau_patience,
            noise_scale=noise_scale,
            exact_cost=exact_cost,
            seed=seed
        )
    except TypeError:
        # Fallback para versão minimalista do Solver.solve()
        vqe_res = vqe_solver.solve(
            warm_start_routes=warm_start_routes,
            maxiter=maxiter,
            optimizer_method=optimizer_method,
            lr=lr,
            exact_cost=exact_cost,
            seed=seed
        )
    t_vqe = time.time() - t0

    # Extração robusta de resultados
    vqe_cost = float(vqe_res.get("best_cost", float("inf")))
    cost_history = vqe_res.get("cost_history", [])
    nfev = len(cost_history)
    approx_ratio = float(exact_cost / vqe_cost) if vqe_cost > 0 and vqe_cost != float("inf") else 0.0
    is_feasible = bool(vqe_res.get("is_feasible", False))
    routes = vqe_res.get("routes", {})

    # Injeta custo exato no dicionário de resposta
    vqe_res["exact_cost"] = exact_cost
    vqe_res["approx_ratio"] = approx_ratio
    vqe_res["quantum_cost"] = vqe_cost

    # 5. IMPRESSÃO DOS RESULTADOS
    print("\n" + "="*70)
    print(f"             RESULTADOS FINAIS DO {str_problem_type} (HAMILTONIANO HERMITIANO)             ")
    print("="*70)
    print(f"Custo Exato (Ground Truth)   : {exact_cost:.4f}")
    print(f"Custo Otimizado (CV-VQE)     : {vqe_cost:.4f}")
    print(f"Viabilidade da Solução       : {'SIM' if is_feasible else 'NÃO'}")
    print(f"Razão de Aproximação         : {approx_ratio:.4f}")
    print(f"Tempo de Execução (VQE)      : {format_timespan(t_vqe)}")
    print("Decodificação de Rotas:")
    for veh, r in routes.items():
        print(f"  • Veículo {veh}: {' -> '.join(map(str, r))}")
    print("="*70 + "\n")

    # 6. REGISTRO DE ARTEFATOS E PERSISTÊNCIA EM ARQUIVO ÚNICO
    if save_outputs:
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{str_problem_type}_{constructor_info_name}_{now_str}"

        cap_list = (
            vehicle_capacity if isinstance(vehicle_capacity, list)
            else [float(vehicle_capacity)] * num_vehicles
        )
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
            vehicle_capacity=cap_list,
            demands=gb.demands.tolist(),
            p_layers=layers,
            max_iter=maxiter,
            momentum_mass=1.0,
            lmbda=20.0,
            lmbda_cap=15.0,
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
            cost_history=cost_history
        )

        # Salva e anexa o registro no JSON único e atualiza o CSV
        logger.save_experiment(experiment_res, sub_folder=sub_folder)

        figures_dir = get_images_path(
            variable_type=variable_type_path,
            problem_type=str_problem_type.lower(),
            sub_folder=sub_folder
        )

        # Plot do Grafo com Rotas
        try:
            gb.plot_graph_and_route(
                solution_vector=routes,
                prefix=f"vqe_{constructor_info_name}"
            )
        except Exception as e:
            logger.warning(f"Não foi possível plotar o grafo de rotas: {e}")

        # Plot da Curva de Convergência
        if cost_history:
            conv_path = figures_dir / f"convergence_{constructor_info_name}.png"
            plt.figure(figsize=(8, 4.5))
            
            cont_loss = vqe_res.get("continuous_loss_history")
            if cont_loss:
                plt.plot(cont_loss, label='Perda Contínua', color='tab:blue', alpha=0.7)
            
            plt.plot(cost_history, label='Custo Discreto', color='tab:green', linewidth=1.8)
            plt.axhline(y=exact_cost, color='red', linestyle=':', label=f'Ground Truth ({exact_cost:.2f})')
            plt.title(f"Convergência VQE - {str_problem_type} (N={n_cities}, V={num_vehicles})")
            plt.xlabel("Iteração")
            plt.ylabel("Custo / Perda")
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(conv_path, dpi=300)
            plt.close()

        # Plot do Espaço de Fase
        cont_x = vqe_res.get("continuous_x")
        cont_p = vqe_res.get("continuous_p")
        if cont_x is not None and cont_p is not None:
            phase_path = figures_dir / f"phase_space_{constructor_info_name}.png"
            plot_phase_space(
                cont_x=cont_x,
                cont_p=cont_p,
                disc_x=vqe_res.get("disc_x", []),
                disc_p=vqe_res.get("disc_p", []),
                output_path=phase_path
            )

        logger.info(f"Artefatos salvos com sucesso em: {figures_dir}")

    return vqe_res


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf

    # Configuração de alocação de memória dinâmica para GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    from itertools import product

    maxiter = 5
    # Exemplo de Bateria de Testes com o novo main.py
    cities = [3]
    vehicles = [2]
    optimizers = ["ADAM"]
    decoders = ["hungarian", "argmax"]
    schedulers = ["augmented_lagrangian", "exponential"]

    for c, v, opt, dec, sched in product(cities, vehicles, optimizers, decoders, schedulers):
        run(
            n_cities=c,
            num_vehicles=v,
            vehicle_capacity=8.0,
            demand_range=(1.0, 4.0),
            layers=1,
            cutoff=4,
            maxiter=maxiter,
            optimizer_method=opt,
            decoder_name=dec,
            scheduler_name=sched,
            use_warm_start=False,
            penalty_gamma=1.02,
            lr=0.01,
            graph_type="random",
            device="cpu",
            seed=42,
            save_outputs=True,
            sub_folder="NEW_HAMILTONIAN"
        )