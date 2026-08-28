# qumodes/circuit.py
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
from typing import Tuple, List, Optional, Dict, Union


class Circuit:
    def __init__(self, num_qumodes: int, num_layers: int = 1, reps: int = 1):
        self.num_qumodes = num_qumodes
        self.num_layers = num_layers
        self.reps = reps
        self.num_params = self._calculate_num_params()

    def _calculate_num_params(self) -> int:
        params_per_rep = (
            2 * self.num_qumodes + 
            2 * ((self.num_qumodes * (self.num_qumodes - 1)) // 2) + 
            2 * self.num_qumodes + 
            self.num_qumodes
        )
        return params_per_rep * self.num_layers * self.reps

    def build_program(self) -> Tuple[sf.Program, List]:
        prog = sf.Program(self.num_qumodes)
        params = [prog.params(f"p_{i}") for i in range(self.num_params)]
        
        param_idx = 0
        with prog.context as q:
            for _ in range(self.num_layers * self.reps):
                # 1. Squeezing Gate
                for i in range(self.num_qumodes):
                    ops.Sgate(params[param_idx], params[param_idx + 1]) | q[i]
                    param_idx += 2

                # 2. Interferômetro (Beam Splitters)
                for i in range(self.num_qumodes):
                    for j in range(i + 1, self.num_qumodes):
                        ops.BSgate(params[param_idx], params[param_idx + 1]) | (q[i], q[j])
                        param_idx += 2

                # 3. Displacement Gate (Posicionamento no Espaço de Fase x, p)
                for i in range(self.num_qumodes):
                    ops.Dgate(params[param_idx], params[param_idx + 1]) | q[i]
                    param_idx += 2

                # 4. Kerr Gate (Não-linearidade)
                for i in range(self.num_qumodes):
                    ops.Kgate(params[param_idx]) | q[i]
                    param_idx += 1

        return prog, params

    def extract_quadratures_tf(self, state) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Calcula os valores esperados das quadraturas <x> e <p> para cada qumode.
        Utiliza tf.math.real() para sanitização de tipo e prevenção de erros de diferenciação.
        """
        x_list = []
        p_list = []
        
        for i in range(self.num_qumodes):
            mean_x, _ = state.quad_expectation(i, phi=0.0)
            mean_p, _ = state.quad_expectation(i, phi=np.pi / 2.0)
            
            x_real = tf.cast(tf.math.real(mean_x), dtype=tf.float32)
            p_real = tf.cast(tf.math.real(mean_p), dtype=tf.float32)
            
            x_list.append(x_real)
            p_list.append(p_real)
            
        return tf.stack(x_list), tf.stack(p_list)

    def initialize_warm_start_params(
        self,
        target_routes: Dict[int, List[int]],
        num_vehicles: int,
        noise_scale: float = 0.01,
        seed: int = 42
    ) -> np.ndarray:
        """
        Inicializa o vetor de parâmetros variacionais a partir de uma solução clássica (Warm-Start),
        mapeando posições temporais (x) e alocações de veículos (p) diretamente para as portas Dgate
        da primeira camada, garantindo a quebra de simetria entre os qumodes.
        """
        rng = np.random.default_rng(seed)
        params = np.zeros(self.num_params, dtype=np.float32)
        
        # Mapeamento de cada cidade livre (1 até N-1) para seu (x_target, p_target)
        x_targets = np.zeros(self.num_qumodes, dtype=np.float32)
        p_targets = np.zeros(self.num_qumodes, dtype=np.float32)

        for vehicle_id, route in target_routes.items():
            # Desconsidera depósitos no início/fim ([0, ..., 0])
            cities_in_route = [c for c in route if c != 0]
            for step_idx, city_id in enumerate(cities_in_route):
                qumode_idx = city_id - 1  # Cidades indexadas de 1 a N-1
                if 0 <= qumode_idx < self.num_qumodes:
                    x_targets[qumode_idx] = float(step_idx + 1)
                    p_targets[qumode_idx] = float(vehicle_id)

        param_idx = 0
        total_steps = self.num_layers * self.reps

        for step in range(total_steps):
            # 1. Squeezing inicial reduzido
            for _ in range(self.num_qumodes):
                params[param_idx] = rng.normal(0.0, noise_scale)
                params[param_idx + 1] = 0.0
                param_idx += 2

            # 2. Beam Splitters leves
            bs_pairs = (self.num_qumodes * (self.num_qumodes - 1)) // 2
            for _ in range(bs_pairs):
                params[param_idx] = rng.uniform(0, np.pi / 16)
                params[param_idx + 1] = 0.0
                param_idx += 2

            # 3. Displacement Gate (Injeção de Warm-Start na camada 0)
            for mode_i in range(self.num_qumodes):
                if step == 0:
                    x_t = x_targets[mode_i] if x_targets[mode_i] > 0 else float(mode_i + 1)
                    p_t = p_targets[mode_i] if p_targets[mode_i] > 0 else float((mode_i % num_vehicles) + 1)

                    # No Strawberry Fields (hbar=2): <x> = 2*r*cos(phi), <p> = 2*r*sin(phi)
                    alpha_x = x_t / 2.0
                    alpha_p = p_t / 2.0

                    r_target = np.sqrt(alpha_x**2 + alpha_p**2)
                    phi_target = np.arctan2(alpha_p, alpha_x)

                    params[param_idx] = r_target + rng.normal(0.0, noise_scale)
                    params[param_idx + 1] = phi_target + rng.normal(0.0, noise_scale)
                else:
                    params[param_idx] = rng.normal(0.0, noise_scale)
                    params[param_idx + 1] = rng.normal(0.0, noise_scale)

                param_idx += 2

            # 4. Kerr inicial nulo com ruído mínimo
            for _ in range(self.num_qumodes):
                params[param_idx] = rng.normal(0.0, noise_scale)
                param_idx += 1

        return params

    def initialize_random_params(
        self, 
        num_vehicles: int = 1, 
        seed: int = 42
    ) -> np.ndarray:
        """
        Inicialização padrão com quebra estocástica de simetria de veículos.
        """
        rng = np.random.default_rng(seed)
        params = np.zeros(self.num_params, dtype=np.float32)

        param_idx = 0
        total_steps = self.num_layers * self.reps

        for step in range(total_steps):
            # 1. Squeezing
            for _ in range(self.num_qumodes):
                params[param_idx] = rng.normal(0.0, 0.01)
                params[param_idx + 1] = 0.0
                param_idx += 2

            # 2. Beam Splitters
            bs_pairs = (self.num_qumodes * (self.num_qumodes - 1)) // 2
            for _ in range(bs_pairs):
                params[param_idx] = rng.uniform(0, np.pi / 8)
                params[param_idx + 1] = 0.0
                param_idx += 2

            # 3. Displacement
            for mode_i in range(self.num_qumodes):
                if step == 0:
                    x_target = float(mode_i + 1)
                    p_target = float((mode_i % num_vehicles) + 1)

                    alpha_x = x_target / 2.0
                    alpha_p = p_target / 2.0

                    r_target = np.sqrt(alpha_x**2 + alpha_p**2)
                    phi_target = np.arctan2(alpha_p, alpha_x)

                    params[param_idx] = r_target
                    params[param_idx + 1] = phi_target
                else:
                    params[param_idx] = 0.0
                    params[param_idx + 1] = 0.0

                param_idx += 2

            # 4. Kerr
            for _ in range(self.num_qumodes):
                params[param_idx] = 0.0
                param_idx += 1

        return params

    def inject_noise(self, params: np.ndarray, noise_scale: float = 0.01, seed: Optional[int] = None) -> np.ndarray:
        """
        Injeta ruído gaussiano nos parâmetros variacionais para escape estocástico de platôs de gradiente.
        """
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, noise_scale, size=params.shape).astype(np.float32)
        return params + noise