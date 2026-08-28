# qumodes/solver.py
import sys
from pathlib import Path
import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Union

import strawberryfields as sf
import tensorflow as tf

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
        reps: int = 1,
        device: str = "cpu"
    ):
        self.hamiltonian = hamiltonian
        self.num_qumodes = hamiltonian.num_free_cities
        self.layers = layers
        self.reps = reps
        self.cutoff_dim = hamiltonian.num_free_cities
        
        self.device_str = "/GPU:0" if device.lower() in ["cuda", "gpu"] else "/CPU:0"
        if "GPU" in self.device_str and not tf.config.list_physical_devices('GPU'):
            self.device_str = "/CPU:0"

        self.ansatz = Circuit(num_qumodes=self.num_qumodes, num_layers=self.layers, reps=self.reps)
        self.engine = sf.Engine(backend="tf", backend_options={"cutoff_dim": self.cutoff_dim})
        self.prog, self.prog_params = self.ansatz.build_program()

        self.history = []
        self.loss_history = []
        self.metrics_history = []

    def _execute_tf_circuit(self, weights: tf.Variable) -> Tuple[tf.Tensor, tf.Tensor]:
        try:
            if hasattr(self.engine, "backend") and getattr(self.engine.backend, "_modemap", None) is not None:
                self.engine.reset()
        except Exception:
            pass

        mapping = {sym: w for sym, w in zip(self.prog_params, tf.unstack(weights))}
        result = self.engine.run(self.prog, args=mapping)
        state = result.state

        x_means, p_means = self.ansatz.extract_quadratures_tf(state)
        return x_means, p_means

    def _spsa_step(
        self, 
        weights: tf.Variable, 
        k: int, 
        a: float = 0.01, 
        c: float = 0.01, 
        alpha: float = 0.602, 
        gamma: float = 0.101
    ) -> tf.Tensor:
        """Executa um passo de gradiente estocástico via SPSA."""
        ak = a / ((k + 1.0) ** alpha)
        ck = c / ((k + 1.0) ** gamma)
        
        delta = tf.cast(2 * np.random.randint(0, 2, size=weights.shape) - 1, tf.float32)
        
        w_plus = tf.Variable(weights + ck * delta)
        w_minus = tf.Variable(weights - ck * delta)
        
        x_plus, p_plus = self._execute_tf_circuit(w_plus)
        loss_plus = self.hamiltonian.compute_continuous_cost_tf(x_plus, p_plus)
        
        x_minus, p_minus = self._execute_tf_circuit(w_minus)
        loss_minus = self.hamiltonian.compute_continuous_cost_tf(x_minus, p_minus)
        
        grad_est = (loss_plus - loss_minus) / (2.0 * ck * delta)
        new_weights = weights - ak * grad_est
        return new_weights

    def _optimize_tf(
        self,
        initial_params: np.ndarray,
        optimizer_name: str = "ADAM",
        lr: float = 0.01,
        maxiter: int = 100,
        penalty_gamma: float = 1.0,
        plateau_patience: int = 15,
        plateau_tol: float = 1e-4,
        noise_scale: float = 0.02
    ) -> np.ndarray:
        with tf.device(self.device_str):
            weights = tf.Variable(initial_params, dtype=tf.float32)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

            plateau_counter = 0
            prev_loss = float("inf")
            initial_lmbda = self.hamiltonian.lmbda

            print(f"\n--- INICIANDO OTIMIZAÇÃO CV-VQE ({optimizer_name.upper()}) ---")
            for step in range(maxiter):
                # 1. Penalty Scheduler (Annealing de penalidade)
                if penalty_gamma != 1.0:
                    current_lmbda = initial_lmbda * (penalty_gamma ** step)
                    self.hamiltonian.set_penalties(current_lmbda)

                # 2. Execução da etapa do Otimizador (SPSA vs ADAM)
                if optimizer_name.upper() == "SPSA":
                    updated_weights = self._spsa_step(weights, step, a=lr)
                    weights.assign(updated_weights)
                    x_tens, p_tens = self._execute_tf_circuit(weights)
                    continuous_loss = self.hamiltonian.compute_continuous_cost_tf(x_tens, p_tens)
                    grad_norm = 0.0
                else:
                    with tf.GradientTape() as tape:
                        x_tens, p_tens = self._execute_tf_circuit(weights)
                        continuous_loss = self.hamiltonian.compute_continuous_cost_tf(x_tens, p_tens)

                    grads = tape.gradient(continuous_loss, [weights])
                    
                    if grads[0] is not None:
                        clipped_grads, _ = tf.clip_by_global_norm(grads, 5.0)
                        grad_norm = float(tf.linalg.global_norm(grads).numpy())
                        optimizer.apply_gradients(zip(clipped_grads, [weights]))
                    else:
                        grad_norm = 0.0

                curr_loss_val = float(continuous_loss.numpy())
                x_fl = [float(v) for v in x_tens.numpy()]
                p_fl = [float(v) for v in p_tens.numpy()]
                
                metrics = self.hamiltonian.compute_metrics(x_fl, p_fl)
                discrete_cost = metrics["total_cost"]

                # 3. Plateau Detector & Noise Injection
                loss_diff = abs(prev_loss - curr_loss_val)
                if loss_diff < plateau_tol:
                    plateau_counter += 1
                else:
                    plateau_counter = 0
                
                if plateau_counter >= plateau_patience:
                    noisy_w = self.ansatz.inject_noise(weights.numpy(), noise_scale=noise_scale)
                    weights.assign(noisy_w)
                    plateau_counter = 0
                    print(f" -> [PlateauDetector] Injeção de ruído estocástico no passo {step}.")

                prev_loss = curr_loss_val

                self.history.append(discrete_cost)
                self.loss_history.append(curr_loss_val)
                self.metrics_history.append(metrics)
                
                if step % max(1, maxiter // 10) == 0 or step == maxiter - 1:
                    print(f"Passo {step:3d}/{maxiter} | Perda Contínua: {curr_loss_val:.4f} | "
                          f"Custo Discreto: {discrete_cost:.2f} | Violação: {metrics['capacity_violation_magnitude']:.2f} | "
                          f"Norm Grad: {grad_norm:.4f}")

            return weights.numpy()

    def solve(
        self,
        initial_params: Optional[np.ndarray] = None,
        warm_start_routes: Optional[Dict[int, List[int]]] = None,
        maxiter: int = 100,
        optimizer_method: str = "ADAM",
        lr: float = 0.01,
        penalty_gamma: float = 1.0,
        plateau_patience: int = 15,
        noise_scale: float = 0.02,
        exact_cost: float = 1.0,
        seed: int = 42
    ) -> Dict[str, Any]:
        self.history = []
        self.loss_history = []
        self.metrics_history = []

        # Suporte a Warm-Start via solução clássica pré-calculada ou parâmetros diretos
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

        opt_params = self._optimize_tf(
            initial_params=initial_params,
            optimizer_name=optimizer_method,
            lr=lr,
            maxiter=maxiter,
            penalty_gamma=penalty_gamma,
            plateau_patience=plateau_patience,
            noise_scale=noise_scale
        )

        with tf.device(self.device_str):
            w_final = tf.Variable(opt_params, dtype=tf.float32)
            opt_x_t, opt_p_t = self._execute_tf_circuit(w_final)
            opt_x = [float(v) for v in opt_x_t.numpy()]
            opt_p = [float(v) for v in opt_p_t.numpy()]

        disc_x, disc_p = self.hamiltonian.discretize_quadratures(opt_x, opt_p)
        decoded_routes = self.hamiltonian.decode_routes(opt_x, opt_p)
        final_metrics = self.hamiltonian.compute_metrics(opt_x, opt_p)
        composite_score = self.hamiltonian.compute_composite_score(
            exact_cost=exact_cost,
            x_vals=opt_x,
            p_vals=opt_p
        )

        return {
            "best_cost": final_metrics["total_cost"],
            "best_energy": final_metrics["total_cost"],
            "route_distance": final_metrics["route_distance"],
            "capacity_violation": final_metrics["capacity_violation_magnitude"],
            "is_feasible": final_metrics["is_feasible"],
            "composite_score": composite_score,
            "opt_params": opt_params,
            "continuous_x": opt_x,
            "continuous_p": opt_p,
            "disc_x": disc_x,
            "disc_p": disc_p,
            "routes": decoded_routes,
            "cost_history": self.history,
            "continuous_loss_history": self.loss_history,
            "metrics_history": self.metrics_history
        }