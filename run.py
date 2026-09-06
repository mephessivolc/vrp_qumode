import time
from typing import Optional, Tuple
import numpy as np

from brute_force import BruteForce, InfeasibleProblemError
from graphs import Graph
from logger import setup_logger
from path import PathManager

from utils import format_timespan, plot_convergence, save_experiment_json

from qumodes.ansatz import CircuitConfig
from qumodes.hamiltonian import HamiltonianParams
from qumodes.solver import ProblemInstance, VQESolver


def run_experiment(
    C: int = 3,  # Número de Cidades
    V: int = 2,  # Número de Veículos
    max_iter: int = 5,
    num_layer: int = 2,
    method: Optional[str] = "COBYLA",
    Q_val: float = 8.0,  # Capacidade individual dos veículos
    h_params: Optional[HamiltonianParams] = None,
    is_warm_start: bool = True,
    demand_range: Tuple[int, int] = (1, 5),
    warm_start_coords: Optional[np.ndarray] = None,
    seed: Optional[int] = 42,
    sub_folder: Optional[str] = "test",
    exp_name: str = "experiment_01",
):
    # 1. Configuração do Gerenciador de Caminhos e Logger
    path = PathManager(variable_type="qumodes", sub_folder=sub_folder)
    log_file_path = path.get_file_path(f"logs/{exp_name}.log")
    logger = setup_logger(name=exp_name, log_file=log_file_path)

    logger.info(f"=== Iniciando Experimento: {exp_name} ===")
    logger.info(f"Parâmetros: Cidades={C}, Veículos={V}, Capacidade={Q_val}, Seed={seed}")

    results_payload = {
        "experiment_name": exp_name,
        "config": {
            "num_cities": C,
            "num_vehicles": V,
            "capacity": Q_val,
            "max_iter": max_iter,
            "num_layer": num_layer,
            "method": method,
            "seed": seed,
            "is_warm_start": is_warm_start,
        },
        "status": "PENDING",
    }

    # 2. Configuração do Grafo, Instância e Plotagem Inicial
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

    results_payload["graph_data"] = {
        "dist_matrix": D,
        "demands": demands,
        "coords": coords,
    }

    # Salva figura do grafo original
    fig_orig_path = path.get_file_path(f"{exp_name}_original_graph.png", is_figure=True)
    g.plot_original_graph(fig_orig_path)
    logger.info(f"Figura do grafo original salva em: {fig_orig_path}")

    # 3. Execução e Validação do BruteForce
    try:
        logger.info("Executando solução clássica (BruteForce)...")
        brute_force = BruteForce(
            dist_matrix=g.dist_matrix,
            num_vehicles=V,
            capacities=Q,
            demands=g.full_demands,
        )
        start_bf = time.time()
        bf_best_cost, bf_best_routes = brute_force.solve()
        bf_time = time.time() - start_bf

        logger.info(f"BruteForce Concluído em {format_timespan(bf_time)}")
        logger.info(f"Melhor Custo Exato: {bf_best_cost:.4f}")
        logger.info(f"Melhores Rotas Exatas: {bf_best_routes}")

        results_payload["exact_solution"] = {
            "cost": bf_best_cost,
            "routes": bf_best_routes,
            "execution_time_seconds": bf_time,
        }

        # Salva a figura da solução exata encontrada
        fig_bf_route_path = path.get_file_path(f"{exp_name}_bf_route.png", is_figure=True)
        g.plot_solution_graph(bf_best_routes, fig_bf_route_path, title_prefix="Solução BruteForce")
        logger.info(f"Figura das rotas exatas salva em: {fig_bf_route_path}")

    except InfeasibleProblemError as err:
        logger.error(f"Inviabilidade detectada: {err}")
        results_payload["status"] = "INFEASIBLE"
        results_payload["error_message"] = str(err)

        json_path = path.get_file_path(f"{exp_name}_results.json")
        save_experiment_json(results_payload, json_path)
        logger.info(f"Resultado de inviabilidade salvo em: {json_path}")
        return

    # 4. Configuração dos Hiperparâmetros e Solver Quântico VQE
    if h_params is None:
        h_params = HamiltonianParams(
            sigma_assign=0.30,
            sigma_empty=0.40,
            lambda_col=25.0,
            lambda_gap=20.0,
            lambda_cap=0.0 if V == 1 else 18.0,
            alpha2=1.0,
            alpha4=1.0,
        )

    instance = ProblemInstance(C=C, V=V, D=D, demands=demands, Q=Q)
    circuit_config = CircuitConfig(num_qumodes=C, num_layers=num_layer)

    logger.info("Iniciando otimização VQE...")
    solver = VQESolver(
        instance=instance,
        circuit_config=circuit_config,
        hamiltonian_params=h_params,
        cutoff=8,
    )

    opt_method = method if method is not None else "COBYLA"
    metrics = solver.solve(method=opt_method, maxiter=max_iter)

    # 5. Cálculo das Métricas Finais e Persistência dos Dados
    gap = ((metrics.final_energy - bf_best_cost) / bf_best_cost) * 100

    logger.info("=== Otimização VQE Concluída ===")
    logger.info(f"Tempo VQE: {format_timespan(metrics.execution_time_seconds)}")
    logger.info(f"Avaliações da Função (nfev): {metrics.total_evaluations}")
    logger.info(f"Energia Final: {metrics.final_energy:.4f}")
    logger.info(f"GAP em relação ao exato: {gap:+.2f}%")
    logger.info(f"Melhores Rotas VQE: {metrics.best_routes}")

    results_payload["status"] = "SUCCESS"
    results_payload["quantum_solution"] = {
        "final_energy": metrics.final_energy,
        "gap_percent": gap,
        "total_evaluations": metrics.total_evaluations,
        "execution_time_seconds": metrics.execution_time_seconds,
        "best_routes": metrics.best_routes,
        "energy_components": metrics.energy_components,
        "cost_history": metrics.cost_history,
    }

    # Salva o arquivo JSON com todos os dados do experimento
    json_path = path.get_file_path(f"{exp_name}_results.json")
    save_experiment_json(results_payload, json_path)
    logger.info(f"Resultados completos salvos em JSON: {json_path}")

    # Salva a figura do grafo das rotas encontradas pelo VQE
    fig_vqe_route_path = path.get_file_path(f"{exp_name}_vqe_route.png", is_figure=True)
    g.plot_solution_graph(metrics.best_routes, fig_vqe_route_path, title_prefix="Solução VQE")
    logger.info(f"Figura das rotas VQE salva em: {fig_vqe_route_path}")

    # Salva o gráfico de convergência do VQE
    fig_conv_path = path.get_file_path(f"{exp_name}_convergence.png", is_figure=True)
    plot_convergence(metrics=metrics, method=opt_method, fig_path_name=fig_conv_path)
    logger.info(f"Gráfico de convergência salvo em: {fig_conv_path}")


if __name__ == "__main__":
    run_experiment()