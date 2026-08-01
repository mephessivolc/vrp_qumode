# tsp/circuit.py
from typing import Dict, Any
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import Dgate, Rgate, BSgate, Sgate


class Circuit:
    """
    Constrói e simula o circuito quântico variacional em Variáveis Contínuas (CV)
    para VQE no Strawberry Fields, utilizando a base de Fock.
    """
    def __init__(self, hamiltonian_builder, layers: int = 1):
        """
        Parameters
        ----------
        hamiltonian_builder : Hamiltonian
            Instância da classe Hamiltonian contendo as matrizes e o cutoff.
        layers : int
            Número de camadas (p) do ansatz variacional do VQE.
        """
        self.hb = hamiltonian_builder
        self.N = hamiltonian_builder.N
        self.cutoff_dim = hamiltonian_builder.cutoff_dim
        self.layers = layers
        _, _, self.H_total_matrix = self.hb.get_hamiltonian_matrices()

        # Cada camada do ansatz possui:
        # - N deslocamentos D(r, phi) -> 2 * N parâmetros
        # - N rotações R(theta)       -> N parâmetros
        # - (N - 1) divisores BS(theta)-> N - 1 parâmetros
        self.params_per_layer = (2 * self.N) + self.N + (self.N - 1)
        self.num_params = self.layers * self.params_per_layer

    def build_ansatz(self, params: np.ndarray) -> sf.Program:
        """
        Monta o circuito variacional (Ansatz) do VQE parametrizado por `params`.
        
        Parameters
        ----------
        params : np.ndarray
            Vetor unidimensional de parâmetros variacionais contínuos.
        """
        prog = sf.Program(self.N)
        param_idx = 0

        with prog.context as q:
            for _ in range(self.layers):
                # 1. Operadores de Deslocamento D(r, phi) por qumode
                for k in range(self.N):
                    r = params[param_idx]
                    phi = params[param_idx + 1]
                    Dgate(r, phi) | q[k]
                    param_idx += 2

                # 2. Rotações de Fase R(theta)
                for k in range(self.N):
                    theta = params[param_idx]
                    Rgate(theta) | q[k]
                    param_idx += 1

                # 3. Emaranhamento via Beam Splitters BS(theta, 0) nos qumodes vizinhos
                for k in range(self.N - 1):
                    theta_bs = params[param_idx]
                    BSgate(theta_bs, 0.0) | (q[k], q[k + 1])
                    param_idx += 1

        return prog

    def evaluate_energy(self, params: np.ndarray) -> float:
        """
        Executa a simulação e calcula o valor esperado da energia <H_total>.
        Esta é a função de custo otimizada pelo VQE.
        """
        prog = self.build_ansatz(params)
        engine = sf.Engine(backend="fock", backend_options={"cutoff_dim": self.cutoff_dim})
        results = engine.run(prog)
        state = results.state

        if state.is_pure:
            ket = state.ket().flatten()
            expectation_value = np.real(np.vdot(ket, self.H_total_matrix @ ket))
        else:
            rho = state.dm()
            dim_total = self.H_total_matrix.shape[0]
            rho_2d = rho.reshape((dim_total, dim_total))
            expectation_value = np.real(np.trace(rho_2d @ self.H_total_matrix))

        return float(expectation_value)

    def get_solution_vector(self, params: np.ndarray) -> Dict[str, Any]:
        """Extrai o estado de Fock (ocupação de fótons) com maior probabilidade."""
        prog = self.build_ansatz(params)
        engine = sf.Engine(backend="fock", backend_options={"cutoff_dim": self.cutoff_dim})
        state = engine.run(prog).state

        probs = state.all_fock_probs()
        max_idx = np.unravel_index(np.argmax(probs), probs.shape)
        max_prob = probs[max_idx]

        return {
            "state_vector": list(max_idx),
            "probability": float(max_prob)
        }


if __name__ == "__main__":
    from graphs import GraphBuilder
    from tsp.hamiltonian import Hamiltonian

    print("==========================================================")
    print("      TESTANDO CIRCUITO VARIACIONAL (VQE PURAMENTE)       ")
    print("==========================================================")

    N = 3
    gb = GraphBuilder(n=N, seed=42)
    hb = Hamiltonian(gb.matrix, lmbda=50.0)
    
    # Instancia o circuito VQE com 1 camada
    circuit = Circuit(hb, layers=1)

    # Gera parâmetros iniciais aleatórios para o VQE
    np.random.seed(42)
    initial_params = np.random.uniform(-0.1, 0.1, size=circuit.num_params)

    energy = circuit.evaluate_energy(initial_params)
    solution = circuit.get_solution_vector(initial_params)

    print(f"Número de Qumodes: {circuit.N}")
    print(f"Total de Parâmetros do VQE: {circuit.num_params}")
    print(f"Energia Esperada Inicial <H_total>: {energy:.4f}")
    print(f"Estado de Fock Mais Provável: {solution['state_vector']}")
    print(f"Probabilidade do Estado: {solution['probability']:.4%}")
    print("==========================================================")