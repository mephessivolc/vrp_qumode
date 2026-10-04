"""
run.py — Executa um experimento VQE em qumodes e salva tudo o que é necessário
para análises posteriores.

Saídas (em PathManager(variable_type="qumodes", sub_folder=...)):
    {exp}_results.json      configuração completa, soluções e métricas (schema 2.0)
    {exp}_state.npz         θ inicial/final, kets, distribuições e traço do ADAM
    experiments_summary.csv uma linha por experimento, para comparar execuções
    figuras                 grafo, rotas (BruteForce e VQE), convergência, energia por iteração
    logs/{exp}.log

Compatibilidade: a assinatura anterior continua válida (novos argumentos têm
padrão) e as chaves antigas de "quantum_solution" foram mantidas.
"""
import dataclasses
import inspect
import time
import traceback
from typing import Dict, Optional, Tuple

import numpy as np
import strawberryfields as sf

from brute_force import BruteForce, InfeasibleProblemError
from graphs import Graph
from logger import setup_logger
from path import PathManager
from initial_states import build_initial_kets, describe_kets
import verify_encoding as VE

from utils import format_timespan, plot_convergence, save_experiment_json

# Todas as bibliotecas quânticas do MESMO pacote: o solver importa o
# hamiltonian.py relativo ao próprio pacote (from .hamiltonian import ...).
from qumodes import analysis as A
from qumodes import hamiltonian as H
from qumodes.ansatz import CircuitConfig
from qumodes.hamiltonian import HamiltonianParams, check_penalties
from qumodes.solver import ProblemInstance, VQESolver


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------
def _defaults_of(func) -> Dict[str, object]:
    """Valores padrão de uma função (registra hiperparâmetros internos do solver)."""
    try:
        return {k: p.default for k, p in inspect.signature(func).parameters.items()
                if p.default is not inspect.Parameter.empty}
    except (TypeError, ValueError):
        return {}


def _run_circuit(solver: VQESolver, theta: np.ndarray, cutoff: int):
    prog = solver.ansatz.build_program(theta)
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    return eng.run(prog).state


def _ket(state) -> np.ndarray:
    ket = state.ket()
    if ket is None:
        raise RuntimeError("Estado misto: a análise exige estado puro (state.ket()).")
    return np.asarray(ket)


def _pct(value: Optional[float], ref: float) -> Optional[float]:
    if value is None or ref == 0:
        return None
    return (value - ref) / ref * 100.0


def _plot_energy_trace(energy, c_star, floor, vacuum, fig_path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(energy)), energy, marker="o", ms=2, label="⟨H⟩ por iteração (VQE)")
    ax.axhline(c_star, color="green", ls="--", label=f"Ótimo clássico C* = {c_star:.3f}")
    if vacuum is not None:
        ax.axhline(vacuum, color="gray", ls=":", label=f"Vácuo = {vacuum:.2f}")
    if floor is not None:
        ax.axhline(floor, color="red", ls="-.", label=f"Piso do cutoff = {floor:.2f}")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Energia")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return True


def _save_json(payload: Dict, path: PathManager, exp_name: str, logger) -> str:
    json_path = path.get_file_path(f"{exp_name}.json", subdir="runs")
    save_experiment_json(A.to_jsonable(payload), json_path)
    logger.info(f"JSON salvo em: {json_path}")
    return json_path


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------
def run_experiment(
    C: int = 3,  # Número de Cidades
    V: int = 2,  # Número de Veículos
    max_iter: int = 5,
    num_layer: int = 2,
    method: Optional[str] = "ADAM",
    lr: float = 0.01,  # Taxa de aprendizado inicial do ADAM
    lr_final: Optional[float] = None,  # se != None, decaimento cosseno lr -> lr_final
    fd_eps: float = 1e-2,  # passo da diferença finita central
    Q_val: float = 8.0,  # Capacidade individual dos veículos
    h_params: Optional[HamiltonianParams] = None,
    is_warm_start: bool = True,
    demand_range: Tuple[int, int] = (1, 5),
    warm_start_coords: Optional[np.ndarray] = None,
    seed: Optional[int] = 42,
    sub_folder: Optional[str] = "test",
    exp_name: str = "experiment_01",
    # --- novos (todos com padrão) ---
    cutoff: int = 8,
    extraction_shots: int = 1000,
    shots_grid: Tuple[int, ...] = (10, 100, 1000),
    penalty_continuous_starts: int = 50,
    require_all_vehicles: Optional[bool] = None,
    compute_cutoff_floor: bool = False,
    save_state: bool = True,
    summary_csv: Optional[str] = "summary.csv",
    notes: str = "",
    # --- codificação e estado inicial ---
    R_positions: Optional[int] = None,     # posições por veículo; None = auto (R = C)
    initial_state: str = "vacuum",         # vacuum | squeezed | gkp
    use_kerr: bool = True,                 # False: circuito puramente gaussiano (ablação)
    init_scale: float = 0.05,              # ruído da inicialização de theta
    squeeze_r: float = 0.0,                # r inicial dos Sgate (0 = como antes)
    gkp_delta: float = 0.35,
    gkp_kappa: float = 4.0,
    # --- rotulagem da varredura (vão para o summary.csv) ---
    sweep: str = "",
    instance_id: str = "",
    # --- verificação da codificação ---
    verify_encoding: bool = True,          # roda os cinco testes antes do VQE
    abort_on_invalid_encoding: bool = True,  # não gasta VQE com codificação inválida
    verify_max_sequences: int = VE.DEFAULT_MAX_SEQUENCES,
) -> Dict:
    t_start = time.time()
    timing: Dict[str, float] = {}
    theta_seed = 42 if seed is None else int(seed)
    extraction_seed = theta_seed

    # 1. Caminhos e logger ---------------------------------------------------
    path = PathManager(variable_type="qumodes", sub_folder=sub_folder)
    log_file_path = path.get_file_path(f"logs/{exp_name}.log")
    logger = setup_logger(name=exp_name, log_file=log_file_path)
    logger.info(f"=== Iniciando Experimento: {exp_name} ===")
    logger.info(f"Parâmetros: Cidades={C}, Veículos={V}, Método={method}, LR={lr}, "
                f"Capacidade={Q_val}, Seed={seed}, Cutoff={cutoff}, Camadas={num_layer}")

    opt_method = method if method is not None else "ADAM"

    payload: Dict = {
        "schema_version": A.SCHEMA_VERSION,
        "experiment_name": exp_name,
        "status": "PENDING",
        "notes": notes,
        "timestamps": {"start": A.now_iso()},
        "environment": A.environment_info(),
        # "config" mantém as chaves antigas no primeiro nível
        "config": {
            "num_cities": C, "num_vehicles": V, "capacity": Q_val,
            "max_iter": max_iter, "learning_rate": lr, "num_layer": num_layer,
            "method": opt_method, "seed": seed, "is_warm_start": is_warm_start,
            "demand_range": list(demand_range),
            "warm_start_coords": warm_start_coords,
            "simulation": {"backend": "fock", "cutoff": cutoff},
            "initial_theta_seed": theta_seed,
            "extraction": {"shots": extraction_shots, "seed": extraction_seed,
                           "shots_grid": list(shots_grid)},
            "analysis": {"penalty_continuous_starts": penalty_continuous_starts,
                         "compute_cutoff_floor": compute_cutoff_floor},
        },
        "timing_seconds": timing,
        "artifacts": {"log": log_file_path},
    }

    # 2. Grafo ---------------------------------------------------------------
    if V > C:
        # O BruteForce exige V sub-rotas não vazias: com V > C não há partição
        # (ele acusaria "capacidade", o que é enganoso).
        logger.error(f"Inviável: {V} veículos obrigatórios para apenas {C} cidades.")
        payload["status"] = "INFEASIBLE"
        payload["error_message"] = f"V={V} > C={C}: o BruteForce exige todos os veículos com >= 1 cidade."
        payload["timestamps"]["end"] = A.now_iso()
        timing["total"] = time.time() - t_start
        _save_json(payload, path, exp_name, logger)
        return payload
    t0 = time.time()
    g = Graph(C=C, V=V, is_warm_start=is_warm_start, warm_start_coords=warm_start_coords,
              demand_range=demand_range, seed=seed)
    D, demands, coords = g.graph
    Q = np.array([Q_val] * V, dtype=np.float64)
    payload["graph_data"] = {"dist_matrix": D, "demands": demands, "coords": coords,
                             "full_demands": g.full_demands}

    # Hamiltoniano: o BruteForce reparte as cidades em V sub-rotas NÃO vazias,
    # logo, com V > 1, todos os veículos são obrigatórios (lambda_vehicle > 0).
    if h_params is None:
        h_params = HamiltonianParams(
            sigma_assign=0.30,
            sigma_empty=0.40,
            lambda_col=25.0,
            lambda_gap=20.0,
            lambda_cap=0.0 if V == 1 else 18.0,
            alpha2=1.0,
            alpha4=1.0,
            lambda_disc=5.0,
            lambda_vehicle=0.0 if V == 1 else float(np.ceil(H.vehicle_penalty_bound(D)) + 1.0),
        )
    req_all = H.requires_all_vehicles(h_params)
    if require_all_vehicles is not None and bool(require_all_vehicles) != req_all:
        raise ValueError(
            "require_all_vehicles deve coincidir com h_params.lambda_vehicle > 0: o solver usa "
            "essa regra para definir as posições por veículo e a extração de rotas.")
    if V > 1 and not req_all:
        logger.warning("V > 1 com lambda_vehicle = 0: o BruteForce exige todos os veículos, "
                       "então a energia mínima pode ficar ABAIXO do custo exato.")
    payload["config"]["hamiltonian"] = dataclasses.asdict(h_params)
    payload["config"]["require_all_vehicles"] = req_all
    fig_orig_path = path.get_file_path(f"{exp_name}_original_graph.png", is_figure=True)
    g.plot_original_graph(fig_orig_path)
    payload["artifacts"]["fig_original_graph"] = fig_orig_path
    timing["graph"] = time.time() - t0

    # 3. BruteForce ----------------------------------------------------------
    try:
        logger.info("Executando solução clássica (BruteForce)...")
        brute_force = BruteForce(dist_matrix=g.dist_matrix, num_vehicles=V,
                                 capacities=Q, demands=g.full_demands)
        t0 = time.time()
        bf_best_cost, bf_best_routes = brute_force.solve()
        timing["brute_force"] = time.time() - t0
        logger.info(f"BruteForce: custo {bf_best_cost:.4f}, rotas {bf_best_routes} "
                    f"({format_timespan(timing['brute_force'])})")
        payload["exact_solution"] = {"cost": bf_best_cost, "routes": bf_best_routes,
                                     "execution_time_seconds": timing["brute_force"]}
        fig_bf = path.get_file_path(f"{exp_name}_bf_route.png", is_figure=True)
        g.plot_solution_graph(bf_best_routes, fig_bf, title_prefix="Solução BruteForce")
        payload["artifacts"]["fig_bf_route"] = fig_bf
    except InfeasibleProblemError as err:
        logger.error(f"Inviabilidade detectada: {err}")
        payload["status"] = "INFEASIBLE"
        payload["error_message"] = str(err)
        payload["timestamps"]["end"] = A.now_iso()
        timing["total"] = time.time() - t_start
        _save_json(payload, path, exp_name, logger)
        return payload

    # 4. Codificação e penalidades ------------------------------------------
    t0 = time.time()
    inst = H.instance_from_legacy(C, V, D, demands, Q,
                                  R=(R_positions if R_positions else "auto"),
                                  require_all_vehicles=req_all)
    grid_points = H.zak_grid_points(inst, cutoff)
    payload["encoding"] = {"positions_per_vehicle": inst.R, "zak_grid_points": grid_points,
                           "zak_grid_limit": H.get_max_grid_points()}
    if grid_points > H.get_max_grid_points():
        msg = (f"Grade de Zak com {grid_points:,} pontos excede o limite de "
               f"{H.get_max_grid_points():,} (~{80 * grid_points / 1e9:.1f} GB no pico). "
               "Reduza C, V ou o cutoff, ou aumente QUMODES_MAX_GRID_POINTS.")
        logger.error(msg)
        payload["status"] = "UNSUPPORTED"
        payload["error_message"] = msg
        payload["timestamps"]["end"] = A.now_iso()
        timing["total"] = time.time() - t_start
        _save_json(payload, path, exp_name, logger)
        return payload
    penalty_check = check_penalties(C, V, D, demands, Q, h_params,
                                    continuous_starts=penalty_continuous_starts, cutoff=cutoff)
    space = A.configuration_space(inst, h_params, require_all_vehicles=req_all)
    enc_summary = A.configuration_summary(space)
    payload["encoding"] = {**payload["encoding"], "slots_per_city": inst.M * inst.R,
                           "lattice_a": inst.a, "lattice_b": inst.b, "hbar": inst.hbar,
                           **enc_summary, "penalty_check": penalty_check,
                           "matches_brute_force": bool(np.isclose(enc_summary["best_feasible_cost"],
                                                                  bf_best_cost)),
                           "vehicle_penalty_bound": H.vehicle_penalty_bound(D)}
    payload["encoding"]["energy_minimum_equals_exact"] = bool(
        penalty_check["ok"] and payload["encoding"]["matches_brute_force"])
    timing["penalty_check"] = time.time() - t0
    logger.info(f"Codificação: R={inst.R}, {enc_summary['num_configurations']} configurações, "
                f"{enc_summary['num_feasible']} viáveis, {enc_summary['num_optimal']} ótimas")
    logger.info(f"Checagem de penalidades: {penalty_check}")
    if not penalty_check["ok"]:
        logger.warning("Penalidades insuficientes: a energia pode ficar abaixo do ótimo "
                       "(aumente lambda_col, lambda_gap, lambda_cap ou lambda_disc).")
    if not payload["encoding"]["matches_brute_force"]:
        logger.warning("Ótimo da codificação difere do BruteForce: verifique lambda_vehicle "
                       "(veículos obrigatórios) ou a convenção de D/demands.")

    # 4b. Verificação da codificação (conjuntos de rotas, não só energias) ----
    verifier = None
    enc_report = None
    if verify_encoding:
        t_ver = time.time()
        verifier = VE.get_verifier(inst, require_all_vehicles=req_all,
                                   use_capacity=h_params.lambda_cap > 0,
                                   max_sequences=verify_max_sequences)
        enc_report = verifier.verify(h_params, continuous_starts=penalty_continuous_starts,
                                     logger=logger)
        payload["encoding"]["verification"] = enc_report
        timing["encoding_verification"] = time.time() - t_ver
        if enc_report["passed"] is False and abort_on_invalid_encoding:
            msg = ("Codificação inválida: o mínimo do Hamiltoniano não decodifica "
                   "para uma rota ótima. Ajuste as penalidades antes de gastar VQE.")
            logger.error(msg)
            payload["status"] = "ENCODING_INVALID"
            payload["error_message"] = msg
            payload["timestamps"]["end"] = A.now_iso()
            timing["total"] = time.time() - t_start
            payload["timing_seconds"] = timing
            _save_json(payload, path, exp_name, logger)
            return payload

    # 5. VQE -----------------------------------------------------------------
    instance = ProblemInstance(C=C, V=V, D=D, demands=demands, Q=Q)
    circuit_config = CircuitConfig(num_qumodes=C, num_layers=num_layer)
    init_kets = build_initial_kets(initial_state, N=C, cutoff=cutoff, R=inst.R,
                                   hbar=inst.hbar, Delta=gkp_delta, kappa=gkp_kappa,
                                   squeeze_r=squeeze_r or 1.2)
    ket_info = describe_kets(init_kets)
    if init_kets is not None and ket_info["max_truncation_loss"] > 1e-3:
        logger.warning(f"Estado inicial '{initial_state}' perde "
                       f"{ket_info['max_truncation_loss']:.2%} da norma no topo de Fock: "
                       f"aumente o cutoff (atual {cutoff}).")
    solver = VQESolver(instance=instance, circuit_config=circuit_config,
                       hamiltonian_params=h_params, cutoff=cutoff,
                       initial_ket=init_kets, use_kerr=use_kerr)
    n_params = solver.ansatz.total_params
    labels = A.parameter_labels(C, num_layer)
    if len(labels) != n_params:
        logger.warning("Rótulos de parâmetros não conferem com o ansatz; usando índices.")
        labels = [f"p{k}" for k in range(n_params)]
    initial_theta = solver.ansatz.generate_initial_theta(
        seed=theta_seed, scale=init_scale, squeeze_r=squeeze_r)
    payload["config"]["initial_state"] = {"kind": initial_state, "use_kerr": use_kerr,
                                          "init_scale": init_scale, "squeeze_r": squeeze_r,
                                          "gkp_delta": gkp_delta, "gkp_kappa": gkp_kappa,
                                          **ket_info}
    payload["config"]["circuit"] = {"num_qumodes": C, "num_layers": num_layer,
                                    "params_per_layer": solver.ansatz.num_params_per_layer,
                                    "total_params": n_params, "parameter_labels": labels}
    payload["config"]["optimizer"] = {"method": opt_method, "max_iter": max_iter, "lr": lr,
                                      "lr_final": lr_final,
                                      "finite_difference_eps": fd_eps,
                                      "adam_defaults": {k: v for k, v in _defaults_of(solver._adam_optimize).items()
                                                        if k not in ("maxiter", "lr")}}

    logger.info(f"Iniciando VQE ({opt_method}) com {n_params} parâmetros...")
    t0 = time.time()
    try:
        metrics = solver.solve(method=opt_method, maxiter=max_iter, lr=lr,
                               lr_final=lr_final, fd_eps=fd_eps,
                               initial_theta=initial_theta)
    except (Exception, KeyboardInterrupt) as err:
        interrupted = isinstance(err, KeyboardInterrupt)
        timing["vqe"] = time.time() - t0
        payload["status"] = "INTERRUPTED" if interrupted else "FAILED"
        payload["error_message"] = repr(err)
        payload["traceback"] = traceback.format_exc()
        history = list(solver.cost_history)
        payload["partial_quantum_solution"] = {
            "cost_history": history,
            "circuit_evaluations": solver.circuit_evaluations,
            "total_evaluations": solver.circuit_evaluations,
            "last_energy_components": solver.last_energy_components}
        if solver.iterate_history:
            payload["partial_quantum_solution"]["optimization_trace"] = {
                "iterations_completed": max(0, len(solver.iterate_history) - 1),
                "energy_per_iteration": list(solver.iterate_history),
                "grad_norm_per_iteration": list(solver.grad_norm_history),
                "lr_per_iteration": list(solver.lr_history)}
        payload["timestamps"]["end"] = A.now_iso()
        timing["total"] = time.time() - t_start
        logger.error(f"VQE {payload['status']}: {err!r} — histórico parcial salvo.")
        _save_json(payload, path, exp_name, logger)
        raise
    timing["vqe"] = time.time() - t0

    # 6. Análise -------------------------------------------------------------
    t0 = time.time()
    c_star = float(bf_best_cost)
    final_state = _run_circuit(solver, metrics.optimal_theta, cutoff)
    initial_state = _run_circuit(solver, initial_theta, cutoff)
    ket_final, ket_initial = _ket(final_state), _ket(initial_state)

    final_m = A.state_metrics(ket_final, inst, h_params, space, shots_grid)
    initial_m = A.state_metrics(ket_initial, inst, h_params, space, shots_grid)
    baselines = A.baseline_metrics(inst, h_params, space, cutoff, shots_grid, seed=theta_seed)

    routes, route_info = H.extract_routes(
        state=final_state, N=C, M=V, D=D, demands=demands, Q=Q, cutoff=cutoff,
        params=h_params, shots=extraction_shots, seed=extraction_seed,
        require_all_vehicles=req_all, return_details=True)
    vacuum_routes, vacuum_info = H.extract_routes(
        state=_VacuumState(cutoff, C), N=C, M=V, D=D, demands=demands, Q=Q, cutoff=cutoff,
        params=h_params, shots=extraction_shots, seed=extraction_seed,
        require_all_vehicles=req_all, return_details=True)

    trace = None
    convergence = None
    if metrics.iterate_history:
        # Registro direto do solver: uma energia por iteração. Não se reconstrói
        # mais o traço a partir de cost_history (que inclui as sondagens).
        trace = {"iterations_completed": metrics.iterations_completed,
                 "evaluations_per_iteration": metrics.evaluations_per_iteration,
                 "energy_per_iteration": list(metrics.iterate_history),
                 "grad_norm_per_iteration": list(metrics.grad_norm_history),
                 "lr_per_iteration": list(metrics.lr_history),
                 "circuit_evaluations": metrics.circuit_evaluations}
        convergence = A.convergence_summary(trace["energy_per_iteration"],
                                            trace["grad_norm_per_iteration"])

    floor = None
    ground_ket = None
    if compute_cutoff_floor:
        try:
            logger.info("Calculando o piso de energia do cutoff (Lanczos)...")
            floor = A.cutoff_energy_floor(inst, h_params, cutoff)
            ground_ket = floor.pop("ground_ket")
            floor["ground_state"] = A.state_metrics(ground_ket, inst, h_params, space,
                                                    shots_grid, top_k=3)
        except MemoryError as err:
            logger.warning(f"Piso do cutoff não calculado: {err}")
            floor = {"error": str(err)}

    e_final = float(metrics.final_energy)
    e_initial = float(metrics.cost_history[0]) if metrics.cost_history else None
    floor_value = floor.get("min_energy") if isinstance(floor, dict) else None
    energy_block = {
        "initial": e_initial,
        "final": e_final,
        "best_iterate": metrics.best_iterate_energy,
        "best_iterate_index": metrics.best_iterate_index,
        "best_evaluated_any": float(np.min(metrics.cost_history)) if metrics.cost_history else None,
        "optimum_c_star": c_star,
        "energy_gap_percent": _pct(e_final, c_star),
        "vacuum": baselines["vacuum"]["energy_components"]["total"],
        "uniform": baselines["uniform"]["energy_components"]["total"],
        "cutoff_floor": floor_value,
        "floor_gap_percent": _pct(floor_value, c_star),
        "gap_to_floor": None if floor_value is None else e_final - floor_value,
        "fraction_of_available_descent": (
            None if floor_value is None or e_initial is None or e_initial == floor_value
            else (e_initial - e_final) / (e_initial - floor_value)),
    }
    route_block = {
        "routes": routes,
        "solver_best_routes": metrics.best_routes,
        "cost": route_info["cost"],
        "feasible": route_info["feasible"],
        "route_gap_percent": _pct(route_info["cost"], c_star),
        "found_optimum": bool(route_info["feasible"] and np.isclose(route_info["cost"], c_star)),
        "shots": extraction_shots,
        "vacuum_same_protocol": {"routes": vacuum_routes, "cost": vacuum_info["cost"],
                                 "feasible": vacuum_info["feasible"]},
    }
    if verifier is not None:
        # Mais confiável que found_optimum: compara CONJUNTOS de rotas, de modo
        # que empate numérico com uma rota diferente não passa por ótimo.
        route_block.update(verifier.audit_routes(routes))
        vac_audit = verifier.audit_routes(vacuum_routes)
        route_block["vacuum_same_protocol"]["route_in_optimal_set"] = \
            vac_audit["route_in_optimal_set"]
    payload["analysis"] = {
        "energy": energy_block,
        "routes": route_block,
        "optimization_trace": trace,
        "convergence": convergence,
        "final_state": final_m,
        "initial_state": initial_m,
        "baselines": baselines,
        "cutoff_floor": floor,
    }
    timing["analysis"] = time.time() - t0

    # 7. Solução quântica (chaves antigas preservadas) -----------------------
    payload["status"] = "SUCCESS"
    payload["quantum_solution"] = {
        "final_energy": e_final,
        "gap_percent": _pct(e_final, c_star),
        "total_evaluations": metrics.total_evaluations,
        "execution_time_seconds": metrics.execution_time_seconds,
        "best_routes": metrics.best_routes,
        "cost_history_in_npz": True,
        "energy_components": metrics.energy_components,
        "optimal_theta": metrics.optimal_theta,
        "initial_theta": initial_theta,
        "seconds_per_evaluation": (metrics.execution_time_seconds / metrics.total_evaluations
                                   if metrics.total_evaluations else None),
    }

    logger.info("=== VQE concluído ===")
    logger.info(f"Tempo VQE: {format_timespan(metrics.execution_time_seconds)} "
                f"({metrics.total_evaluations} avaliações)")
    logger.info(f"Energia: inicial {e_initial:.4f} -> final {e_final:.4f} | C* {c_star:.4f} | "
                f"GAP de energia {energy_block['energy_gap_percent']:+.2f}%")
    if floor_value is not None:
        logger.info(f"Piso do cutoff {cutoff}: {floor_value:.4f} "
                    f"(GAP mínimo possível {energy_block['floor_gap_percent']:+.2f}%)")
    logger.info(f"Rotas: {routes} | custo {route_info['cost']:.4f} | "
                f"GAP de rota {route_block['route_gap_percent']:+.2f}% | viável {route_info['feasible']}")
    logger.info(f"P(ótimo): final {final_m['p_optimal']:.4f} | inicial {initial_m['p_optimal']:.4f} "
                f"| vácuo {baselines['vacuum']['p_optimal']:.4f} "
                f"| uniforme {baselines['uniform']['p_optimal']:.4f}")
    if convergence and convergence.get("still_decreasing"):
        logger.warning("A energia ainda estava caindo ao final: aumente max_iter ou lr.")
    if min(final_m["state_norm"], 1.0) < 0.99 or max(final_m["last_fock_level_population"]) > 0.01:
        logger.warning("Sinais de truncagem de Fock (norma < 0,99 ou população no último nível "
                       "> 1%): considere aumentar o cutoff.")

    # 8. Artefatos -----------------------------------------------------------
    if save_state:
        npz_path = path.get_file_path(f"{exp_name}_state.npz", subdir="arrays")
        arrays = {"initial_theta": initial_theta, "optimal_theta": metrics.optimal_theta,
                  "ket_final": ket_final, "ket_initial": ket_initial,
                  "cost_history": np.asarray(metrics.cost_history),
                  "iterate_history": np.asarray(metrics.iterate_history),
                  "grad_norm_history": np.asarray(metrics.grad_norm_history),
                  "config_energy": space["energy"], "config_feasible": space["feasible"],
                  "config_cost": space["cost"], "config_optimal": space["optimal"]}
        if trace is not None:
            arrays["energy_per_iteration"] = np.asarray(trace["energy_per_iteration"])
            arrays["grad_norm_per_iteration"] = np.asarray(trace["grad_norm_per_iteration"])
        if ground_ket is not None:
            arrays["ket_cutoff_ground"] = ground_ket
        np.savez_compressed(npz_path, **arrays)
        payload["artifacts"]["state_npz"] = npz_path

    fig_vqe = path.get_file_path(f"{exp_name}_vqe_route.png", is_figure=True)
    g.plot_solution_graph(metrics.best_routes, fig_vqe, title_prefix="Solução VQE")
    payload["artifacts"]["fig_vqe_route"] = fig_vqe
    fig_conv = path.get_file_path(f"{exp_name}_convergence.png", is_figure=True)
    plot_convergence(metrics=metrics, method=opt_method, fig_path_name=fig_conv,
                     c_star=c_star, uniform_energy=energy_block["uniform"],
                     cutoff_floor=floor_value)
    fig_conv_norm = path.get_file_path(f"{exp_name}_convergence_normalized.png", is_figure=True)
    plot_convergence(metrics=metrics, method=opt_method, fig_path_name=fig_conv_norm,
                     c_star=c_star, uniform_energy=energy_block["uniform"],
                     cutoff_floor=floor_value, normalize=True)
    payload["artifacts"]["fig_convergence_normalized"] = fig_conv_norm
    payload["artifacts"]["fig_convergence"] = fig_conv
    if trace is not None:
        fig_trace = path.get_file_path(f"{exp_name}_energy_per_iteration.png", is_figure=True)
        if _plot_energy_trace(trace["energy_per_iteration"], c_star, floor_value,
                              energy_block["vacuum"], fig_trace):
            payload["artifacts"]["fig_energy_per_iteration"] = fig_trace

    payload["timestamps"]["end"] = A.now_iso()
    timing["total"] = time.time() - t_start
    json_path = path.get_file_path(f"{exp_name}.json", subdir="runs")
    csv_path = path.get_file_path(summary_csv) if summary_csv else None
    payload["artifacts"]["results_json"] = json_path
    if csv_path:
        payload["artifacts"]["summary_csv"] = csv_path
    _save_json(payload, path, exp_name, logger)

    # 9. Resumo tabular --------------------------------------------------------
    if csv_path:
        A.append_summary_csv(csv_path, {
            "experiment_name": exp_name, "timestamp": payload["timestamps"]["end"],
            "sweep": sweep, "instance_id": instance_id,
            "status": payload["status"], "C": C, "V": V, "Q": Q_val, "seed": seed,
            "theta_seed": theta_seed, "initial_state": initial_state,
            "use_kerr": use_kerr, "squeeze_r": squeeze_r, "init_scale": init_scale,
            "method": opt_method, "lr": lr, "max_iter": max_iter, "num_layer": num_layer,
            "n_params": n_params, "cutoff": cutoff, "positions_R": inst.R,
            "lambda_col": h_params.lambda_col, "lambda_gap": h_params.lambda_gap,
            "lambda_cap": h_params.lambda_cap, "lambda_disc": h_params.lambda_disc,
            "lambda_vehicle": h_params.lambda_vehicle, "penalty_ok": penalty_check["ok"],
            "energy_min_equals_exact": payload["encoding"]["energy_minimum_equals_exact"],
            "c_star": c_star, "energy_initial": e_initial, "energy_final": e_final,
            "energy_gap_percent": energy_block["energy_gap_percent"],
            "cutoff_floor": floor_value, "floor_gap_percent": energy_block["floor_gap_percent"],
            "vacuum_energy": energy_block["vacuum"], "uniform_energy": energy_block["uniform"],
            "route_cost": route_info["cost"], "route_gap_percent": route_block["route_gap_percent"],
            "route_feasible": route_info["feasible"], "found_optimum": route_block["found_optimum"],
            "p_opt_final": final_m["p_optimal"], "p_opt_initial": initial_m["p_optimal"],
            "p_opt_vacuum": baselines["vacuum"]["p_optimal"],
            "p_opt_uniform": baselines["uniform"]["p_optimal"],
            "p_feas_final": final_m["p_feasible"],
            "still_decreasing": None if convergence is None else convergence["still_decreasing"],
            "state_norm_final": final_m["state_norm"],
            "iterations_completed": metrics.iterations_completed,
            "circuit_evaluations": metrics.circuit_evaluations,
            "evaluations_per_iteration": metrics.evaluations_per_iteration,
            "best_iterate": metrics.best_iterate_energy,
            "total_evaluations": metrics.total_evaluations,
            "rho": (final_m["p_optimal"] / baselines["uniform"]["p_optimal"]
                    if baselines["uniform"]["p_optimal"] else None),
            "rho_vacuum": (baselines["vacuum"]["p_optimal"] / baselines["uniform"]["p_optimal"]
                           if baselines["uniform"]["p_optimal"] else None),
            "p_feas_uniform": baselines["uniform"]["p_feasible"],
            "expected_cost_given_feasible": final_m.get("expected_cost_given_feasible"),
            "energy_normalized_final": ((e_final - c_star) / (energy_block["uniform"] - c_star)
                                        if energy_block["uniform"] != c_star else None),
            "h_disc_per_mode": (metrics.energy_components.get("disc", 0.0)
                                / h_params.lambda_disc / C
                                if h_params.lambda_disc else None),
            "delta_dist": initial_m["energy_components"]["dist"] - final_m["energy_components"]["dist"],
            "delta_disc": initial_m["energy_components"]["disc"] - final_m["energy_components"]["disc"],
            "mean_photons_max": max(final_m["mean_photons_per_mode"]),
            "grad_norm_final": (metrics.grad_norm_history[-1] if metrics.grad_norm_history else None),
            **VE.summary_columns(enc_report),
            "route_in_optimal_set": route_block.get("route_in_optimal_set"),
            "num_optimal": enc_summary["num_optimal"],
            "num_feasible": enc_summary["num_feasible"],
            "cost_levels": len(enc_summary["feasible_cost_levels"]),
            "zak_grid_points": grid_points,
            "vqe_seconds": timing["vqe"], "total_seconds": timing["total"],
            "results_json": json_path,
        })

    return payload


class _VacuumState:
    """Estado de vácuo com a interface mínima de um estado do Strawberry Fields."""
    is_pure = True

    def __init__(self, cutoff: int, modes: int):
        self._ket = np.zeros((cutoff,) * modes)
        self._ket[(0,) * modes] = 1.0

    def ket(self):
        return self._ket


if __name__ == "__main__":
    run_experiment()
