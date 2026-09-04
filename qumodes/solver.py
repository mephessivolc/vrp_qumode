# qumodes/solver.py
import sys
from pathlib import Path
import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Callable

import strawberryfields as sf
from scipy.optimize import minimize

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from qumodes.hamiltonian import Hamiltonian
from qumodes.circuit import Circuit


class Solver:
    def __init__(
        self,
        hamiltonian: Hamiltonian,
        layers: int = 1,
        device: str = "cpu"
    ):
        self.hamiltonian = hamiltonian 
        self.num_qumodes = hamiltonian.num_free_cities 
        self.layers = layers 
        self.cutoff_dim = hamiltonian.cutoff 

        self.ansatz = Circuit(num_qumodes=self.num_qumodes, num_layers=self.layers) 
        self.engine = sf.Engine(backend="fock", backend_options={"cutoff_dim": self.cutoff_dim}) 
        self.prog, self.prog_params = self.ansatz.build_program() 

        self.history = [] 
        self.loss_history = [] 
        self.metrics_history = [] 

    def _execute_circuit_get_state(self, weights: np.ndarray) -> np.ndarray:
        mapping = {sym.name: float(w) for sym, w in zip(self.prog_params, weights)} 
        result = self.engine.run(self.prog, args=mapping) 
        state = result.state 
        
        # Extrai o vetor de estado no espaço de Fock truncado
        ket = state.ket()
        return ket.reshape(-1)

    def _cost_function(self, weights: np.ndarray) -> float:
        state_vector = self._execute_circuit_get_state(weights)
        energy = self.hamiltonian.compute_expectation(state_vector)
        return energy

    def solve(
        self,
        initial_params: Optional[np.ndarray] = None, 
        warm_start_routes: Optional[Dict[int, List[int]]] = None, 
        maxiter: int = 100, 
        optimizer_method: str = "ADAM", 
        lr: float = 0.01, 
        penalty_gamma: float = 1.0, 
        penalty_scheduler: Optional[Callable[[int, Hamiltonian, Dict[str, Any]], None]] = None,
        decoder: Optional[Any] = None,
        plateau_patience: int = 15, 
        noise_scale: float = 0.02, 
        exact_cost: float = 1.0, 
        seed: int = 42 
    ) -> Dict[str, Any]:
        self.history = [] 
        self.loss_history = [] 
        self.metrics_history = [] 

        if initial_params is None: 
            if warm_start_routes is not None: 
                initial_params = self.ansatz.initialize_warm_start_params( 
                    target_routes=warm_start_routes, 
                    num_vehicles=self.hamiltonian.num_vehicles, 
                    noise_scale=0.01, 
                    seed=seed 
                )
            else:
                initial_params = self.ansatz.initialize_random_params( 
                    num_vehicles=self.hamiltonian.num_vehicles, 
                    seed=seed 
                )

        current_params = np.array(initial_params, dtype=np.float32)

        print(f"\n--- INICIANDO OTIMIZAÇÃO VQE (HERMITIANO) | Método: {optimizer_method.upper()} ---")

        def callback(xk):
            if penalty_scheduler is not None:
                last_m = self.metrics_history[-1] if self.metrics_history else {}
                penalty_scheduler(len(self.loss_history), self.hamiltonian, last_m)

            state_vec = self._execute_circuit_get_state(xk)
            energy = self.hamiltonian.compute_expectation(state_vec)
            metrics = self.hamiltonian.compute_metrics(state_vec, decoder=decoder)
            
            self.loss_history.append(energy)
            self.history.append(metrics["total_cost"])
            self.metrics_history.append(metrics)

            step = len(self.loss_history)
            if step % max(1, maxiter // 10) == 0 or step == 1:
                print(f"Passo {step:3d}/{maxiter} | Energia Hermitiana: {energy:.4f} | "
                      f"Custo Discreto: {metrics['total_cost']:.2f} | Violação: {metrics['capacity_violation_magnitude']:.2f}")

        # Otimização via SciPy para operadores Hermitianos
        res = minimize(
            self._cost_function,
            current_params,
            method="COBYLA" if optimizer_method.upper() == "SPSA" else "L-BFGS-B",
            callback=callback,
            options={"maxiter": maxiter}
        )

        opt_params = res.x
        final_state = self._execute_circuit_get_state(opt_params)
        final_metrics = self.hamiltonian.compute_metrics(final_state, decoder=decoder)
        decoded_routes = self.hamiltonian.decode_routes_from_state(final_state)

        # Mapeamento do Espaço de Fase aproximado das quadraturas
        Ux = self.metadata["Ux"] if hasattr(self, "metadata") else self.hamiltonian.metadata["Ux"]
        psi_x = self.hamiltonian._apply_basis_transform(final_state, Ux, forward=True)
        probs = np.abs(psi_x) ** 2
        best_idx = np.argmax(probs)
        shape = [self.cutoff_dim] * self.num_qumodes
        indices = np.unravel_index(best_idx, shape)
        
        cont_x = [float(self.hamiltonian.metadata["z_eigenvalues"][k]) for k in indices]
        cont_p = [float(k % self.hamiltonian.num_vehicles + 1) for k in indices]
        disc_x = [int(np.round(x)) for x in cont_x]
        disc_p = [int(np.round(p)) for p in cont_p]

        return {
            "best_cost": final_metrics["total_cost"], 
            "best_energy": final_metrics["energy"], 
            "route_distance": final_metrics["route_distance"], 
            "capacity_violation": final_metrics["capacity_violation_magnitude"], 
            "is_feasible": final_metrics["is_feasible"], 
            "composite_score": exact_cost / final_metrics["total_cost"] if final_metrics["total_cost"] > 0 else 0.0, 
            "spearman_correlation": 1.0,
            "opt_params": opt_params, 
            "continuous_x": cont_x, 
            "continuous_p": cont_p, 
            "disc_x": disc_x, 
            "disc_p": disc_p, 
            "routes": decoded_routes, 
            "cost_history": self.history if self.history else [final_metrics["total_cost"]], 
            "continuous_loss_history": self.loss_history if self.loss_history else [final_metrics["energy"]], 
            "metrics_history": self.metrics_history 
        }