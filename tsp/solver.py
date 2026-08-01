# tsp/solver.py
from typing import Dict, List, Optional
import numpy as np
from scipy.optimize import minimize, dual_annealing

from tsp.circuit import Circuit


class Solver:
    """
    Solver VQE para o problema do Caixeiro Viajante (TSP) em Variáveis Contínuas (CV).
    Conecta o otimizador clássico à simulação quântica do circuito no Strawberry Fields.
    """
    def __init__(self, hamiltonian_builder, layers: int = 1):
        """
        Parameters
        ----------
        hamiltonian_builder : Hamiltonian
            Instância do Hamiltoniano do TSP.
        layers : int
            Número de camadas (p) do ansatz variacional do VQE.
        """
        self.hamiltonian = hamiltonian_builder
        self.num_cities = hamiltonian_builder.N
        self.circuit = Circuit(hamiltonian_builder, layers=layers)
        self.history: List[float] = []

    def solve(
        self, 
        maxiter: int = 100, 
        optimizer_method: str = "COBYLA", 
        seed: int = 42
    ) -> Dict:
        """
        Executa o loop de otimização variacional do VQE.
        
        Parameters
        ----------
        maxiter : int
            Número máximo de iterações do otimizador clássico.
        optimizer_method : str
            Algoritmo de otimização (ex: 'COBYLA', 'Nelder-Mead', 'Powell', 'DUAL-ANNEALING').
        seed : int
            Semente aleatória para reprodutibilidade dos parâmetros iniciais.
            
        Returns
        -------
        Dict contendo os resultados da otimização, parâmetros ótimos,
        estado final e histórico de convergência.
        """
        self.history = []
        np.random.seed(seed)
        
        # O número de parâmetros é definido pela estrutura do Circuit
        num_params = self.circuit.num_params
        initial_params = np.random.uniform(-0.1, 0.1, size=num_params)
        bounds = [(-np.pi, np.pi)] * num_params

        # Função objetivo avaliada pelo simulador quântico
        def objective_function(params: np.ndarray) -> float:
            energy = self.circuit.evaluate_energy(params)
            self.history.append(energy)
            return energy

        method_upper = optimizer_method.upper()

        if method_upper in ["COBYLA", "NELDER-MEAD", "POWELL"]:
            res = minimize(
                fun=objective_function,
                x0=initial_params,
                method=optimizer_method,
                options={"maxiter": maxiter}
            )
        elif method_upper == "DUAL-ANNEALING":
            res = dual_annealing(
                func=objective_function,
                bounds=bounds,
                maxiter=maxiter,
                seed=seed
            )
        else:
            res = minimize(
                fun=objective_function,
                x0=initial_params,
                method=optimizer_method,
                bounds=bounds,
                options={"maxiter": maxiter}
            )

        # Extrai a solução amostrada do estado quântico final
        solution_data = self.circuit.get_solution_vector(res.x)

        return {
            "opt_result": res,
            "best_energy": float(res.fun),
            "opt_params": res.x,
            "solution_vector": solution_data["state_vector"],
            "probability": solution_data["probability"],
            "history": self.history
        }


if __name__ == "__main__":
    from graphs import GraphBuilder
    from tsp.hamiltonian import Hamiltonian

    print("==========================================================")
    print("         TESTANDO O SOLVER VQE PARA O TSP                 ")
    print("==========================================================")

    N = 3
    gb = GraphBuilder(n=N, seed=42)
    hb = Hamiltonian(gb.matrix, lmbda=50.0)
    
    # Instancia e executa o solver
    solver = Solver(hb, layers=1)
    results = solver.solve(maxiter=30, optimizer_method="COBYLA", seed=42)

    print(f"Número de Cidades: {N}")
    print(f"Número de Parâmetros do VQE: {solver.circuit.num_params}")
    print(f"Energia Mínima Encontrada <H_total>: {results['best_energy']:.4f}")
    print(f"Estado de Fock (Solução Amostrada): {results['solution_vector']}")
    print(f"Probabilidade do Estado: {results['probability']:.4%}")
    print(f"Total de Iterações Realizadas: {len(results['history'])}")
    print("==========================================================")