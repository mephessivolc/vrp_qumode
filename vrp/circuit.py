# vrp/circuit.py
import numpy as np
import strawberryfields as sf
from strawberryfields import ops
from typing import Tuple, List, Optional


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

    def initialize_random_params(
        self, 
        num_vehicles: int = 1, 
        seed: int = 42
    ) -> np.ndarray:
        """
        Inicializa os parâmetros do circuito no espaço de fase.
        Distribui as cidades de forma equilibrada entre os veículos disponíveis.
        """
        rng = np.random.default_rng(seed)
        params = np.zeros(self.num_params, dtype=np.float32)

        param_idx = 0
        total_steps = self.num_layers * self.reps

        for step in range(total_steps):
            # 1. Squeezing pequeno (próximo ao estado de vácuo)
            for _ in range(self.num_qumodes):
                params[param_idx] = rng.normal(0.0, 0.01)
                params[param_idx + 1] = 0.0
                param_idx += 2

            # 2. Beam Splitters (acoplamento leve entre modos)
            bs_pairs = (self.num_qumodes * (self.num_qumodes - 1)) // 2
            for _ in range(bs_pairs):
                params[param_idx] = rng.uniform(0, np.pi / 8)
                params[param_idx + 1] = 0.0
                param_idx += 2

            # 3. Displacement (Posicionamento inicial x=Ordem Temporal, p=Veículo)
            for mode_i in range(self.num_qumodes):
                if step == 0:
                    # Define posições de destino iniciais (x_target em [1, N], p_target em [1, V])
                    x_target = float(mode_i + 1)
                    # Intercala atribuição aos veículos: 1, 2, ..., V, 1, 2, ...
                    p_target = float((mode_i % num_vehicles) + 1)

                    # Conversão das coordenadas (x, p) para raio e fase do Dgate
                    # No Strawberry Fields (hbar=2): <x> = 2*r*cos(phi), <p> = 2*r*sin(phi)
                    alpha_x = x_target / 2.0
                    alpha_p = p_target / 2.0

                    r_target = np.sqrt(alpha_x**2 + alpha_p**2)
                    phi_target = np.arctan2(alpha_p, alpha_x)

                    params[param_idx] = r_target
                    params[param_idx + 1] = phi_target
                else:
                    # Camadas subsequentes iniciam com deslocamento nulo para estabilidade
                    params[param_idx] = 0.0
                    params[param_idx + 1] = 0.0

                param_idx += 2

            # 4. Kerr inicial zerado
            for _ in range(self.num_qumodes):
                params[param_idx] = 0.0
                param_idx += 1

        return params