# vrp/solver.py
import sys
from pathlib import Path
import numpy as np
from typing import Tuple, Dict, Any

import strawberryfields as sf
import tensorflow as tf

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vrp.hamiltonian import Hamiltonian
from vrp.circuit import Circuit


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
        self.loss_history = []  # Registra a Perda Contínua (Diferenciável)

    def _execute_tf_circuit(self, weights: tf.Variable) -> Tuple[tf.Tensor, tf.Tensor]:
        try:
            if hasattr(self.engine, "backend") and getattr(self.engine.backend, "_modemap", None) is not None:
                self.engine.reset()
        except Exception:
            pass

        mapping = {sym: w for sym, w in zip(self.prog_params, tf.unstack(weights))}
        result = self.engine.run(self.prog, args=mapping)
        state = result.state

        x_means, p_means = [], []

        for mode in range(self.num_qumodes):
            x_mean, _ = state.quad_expectation(mode, phi=0.0)
            p_mean, _ = state.quad_expectation(mode, phi=np.pi / 2)
            
            # CORREÇÃO PONTO 1: tf.math.real elimina o aviso de cast complex64 -> float32
            x_means.append(tf.math.real(x_mean))
            p_means.append(tf.math.real(p_mean))

        return tf.stack(x_means), tf.stack(p_means)

    def _optimize_tf(
        self,
        initial_params: np.ndarray,
        optimizer_name: str = "ADAM",
        lr: float = 0.01,
        maxiter: int = 100
    ) -> np.ndarray:
        with tf.device(self.device_str):
            weights = tf.Variable(initial_params, dtype=tf.float32)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

            print("\n--- INICIANDO OTIMIZAÇÃO CV-VQE ---")
            for step in range(maxiter):
                with tf.GradientTape() as tape:
                    x_tens, p_tens = self._execute_tf_circuit(weights)
                    continuous_loss = self.hamiltonian.compute_continuous_cost_tf(x_tens, p_tens)

                grads = tape.gradient(continuous_loss, [weights])
                
                if grads[0] is not None:
                    # CORREÇÃO PONTO 2: Clip de gradiente evita a explosão da norma
                    clipped_grads, _ = tf.clip_by_global_norm(grads, 5.0)
                    grad_norm = tf.linalg.global_norm(grads).numpy()
                    optimizer.apply_gradients(zip(clipped_grads, [weights]))
                else:
                    grad_norm = 0.0

                x_fl = [float(v) for v in x_tens.numpy()]
                p_fl = [float(v) for v in p_tens.numpy()]
                discrete_cost = self.hamiltonian.compute_cost(x_fl, p_fl)

                self.history.append(discrete_cost)
                self.loss_history.append(float(continuous_loss.numpy()))
                
                if step % max(1, maxiter // 10) == 0 or step == maxiter - 1:
                    print(f"Passo {step:3d}/{maxiter} | Perda Contínua: {float(continuous_loss):.4f} | "
                          f"Custo Discreto: {discrete_cost:.2f} | Norm Gradiente: {grad_norm:.4f}")

            return weights.numpy()

    def solve(
        self,
        initial_params: np.ndarray = None,
        maxiter: int = 100,
        optimizer_method: str = "ADAM",
        lr: float = 0.01,
        seed: int = 42
    ) -> Dict[str, Any]:
        self.history = []
        self.loss_history = []

        if initial_params is None:
            initial_params = self.ansatz.initialize_random_params(seed=seed)

        opt_params = self._optimize_tf(
            initial_params=initial_params,
            optimizer_name=optimizer_method,
            lr=lr,
            maxiter=maxiter
        )

        with tf.device(self.device_str):
            w_final = tf.Variable(opt_params, dtype=tf.float32)
            opt_x_t, opt_p_t = self._execute_tf_circuit(w_final)
            opt_x = [float(v) for v in opt_x_t.numpy()]
            opt_p = [float(v) for v in opt_p_t.numpy()]

        disc_x, disc_p = self.hamiltonian.discretize_quadratures(opt_x, opt_p)
        decoded_routes = self.hamiltonian.decode_routes(opt_x, opt_p)
        final_discrete_cost = self.hamiltonian.compute_cost(opt_x, opt_p)

        return {
                "best_cost": final_discrete_cost,
                "best_energy": final_discrete_cost,
                "opt_params": opt_params,
                "continuous_x": opt_x,
                "continuous_p": opt_p,
                "disc_x": disc_x,
                "disc_p": disc_p,
                "routes": decoded_routes,
                "cost_history": self.history,          # Custo discreto
                "continuous_loss_history": self.loss_history  # Perda contínua (muda a cada passo)
            }