# vrp/circuit.py
import numpy as np
import strawberryfields as sf
from strawberryfields import ops
from typing import Tuple, List


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
        
        # CORREÇÃO AQUI: Usar 'prog.params' (instância) e não 'sf.Program.params' (classe)
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

                # 3. Displacement Gate (Posicionamento no Espaço de Fase)
                for i in range(self.num_qumodes):
                    ops.Dgate(params[param_idx], params[param_idx + 1]) | q[i]
                    param_idx += 2

                # 4. Kerr Gate (Não-linearidade)
                for i in range(self.num_qumodes):
                    ops.Kgate(params[param_idx]) | q[i]
                    param_idx += 1

        return prog, params

    def initialize_random_params(self, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        params = np.zeros(self.num_params, dtype=np.float32)

        param_idx = 0
        for _ in range(self.num_layers * self.reps):
            # Squeezing pequeno
            for _ in range(self.num_qumodes):
                params[param_idx] = rng.normal(0.0, 0.01)
                params[param_idx + 1] = 0.0
                param_idx += 2

            # Beam Splitters
            bs_pairs = (self.num_qumodes * (self.num_qumodes - 1)) // 2
            for _ in range(bs_pairs):
                params[param_idx] = rng.uniform(0, np.pi / 8)
                params[param_idx + 1] = 0.0
                param_idx += 2

            # Quebra de simetria inicial no espaço de fase (x, p)
            for mode_i in range(self.num_qumodes):
                r_target = 1.0 + (mode_i * 0.3)
                phi_target = (mode_i * np.pi) / (2 * max(1, self.num_qumodes - 1))
                params[param_idx] = r_target
                params[param_idx + 1] = phi_target
                param_idx += 2

            # Kerr inicial zerado
            for _ in range(self.num_qumodes):
                params[param_idx] = 0.0
                param_idx += 1

        return params