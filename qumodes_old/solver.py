from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Union
import numpy as np
from scipy.optimize import minimize
import strawberryfields as sf

# Importações relativas internas do pacote qumodes
from .ansatz import CircuitConfig, ContinuousVariableAnsatz
from .hamiltonian import HamiltonianParams, evaluate_sf_state


@dataclass
class ProblemInstance:
    V: int
    C: int
    D: np.ndarray
    demands: np.ndarray
    Q: np.ndarray


@dataclass
class SolverMetrics:
    optimal_theta: np.ndarray
    final_energy: float
    energy_components: Dict[str, float]
    cost_history: List[float]
    execution_time_seconds: float
    total_evaluations: int
    best_routes: Union[List[int], Dict[int, List[int]]]


class VQESolver:

    def __init__(
        self,
        instance: ProblemInstance,
        circuit_config: CircuitConfig,
        hamiltonian_params: HamiltonianParams,
        cutoff: int = 10,
    ):
        self.instance = instance
        self.circuit_config = circuit_config
        self.h_params = hamiltonian_params
        self.cutoff = cutoff

        self.ansatz = ContinuousVariableAnsatz(config=circuit_config)
        self.cost_history: List[float] = []
        self.last_energy_components: Dict[str, float] = {}

    def _cost_function(self, theta: np.ndarray) -> float:
        prog = self.ansatz.build_program(theta)

        eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff})
        results = eng.run(prog)

        energies = evaluate_sf_state(
            state=results.state,
            M=self.instance.V,
            N=self.instance.C,
            D=self.instance.D,
            demands=self.instance.demands,
            Q=self.instance.Q,
            cutoff=self.cutoff,
            params=self.h_params,
        )

        total_cost = energies["total"]
        self.cost_history.append(total_cost)
        self.last_energy_components = energies

        return total_cost

    def _compute_numerical_gradient(
        self, theta: np.ndarray, eps: float = 1e-4
    ) -> np.ndarray:
        """Calcula o gradiente numérico via diferença finita central."""
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_minus = theta.copy()

            theta_plus[i] += eps
            theta_minus[i] -= eps

            c_plus = self._cost_function(theta_plus)
            c_minus = self._cost_function(theta_minus)

            grad[i] = (c_plus - c_minus) / (2.0 * eps)
        return grad

    def _adam_optimize(
        self,
        initial_theta: np.ndarray,
        maxiter: int = 200,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps_adam: float = 1e-8,
    ) -> np.ndarray:
        """Executa a otimização ADAM utilizando gradientes por diferenças finitas."""
        theta = initial_theta.copy()
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)

        self._cost_function(theta)

        for t in range(1, maxiter + 1):
            grad = self._compute_numerical_gradient(theta)

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad**2)

            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)

            theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
            self._cost_function(theta)

        return theta

    def _extract_routes(self, state) -> Union[List[int], Dict[int, List[int]]]:
        """
        Decodifica o estado quântico otimizado para extrair o vetor de melhores rotas.
        Utiliza os valores esperados da quadratura de posição <x> de cada qumode para determinar a sequência de visitação.
        """
        x_means = []
        for i in range(self.instance.C):
            try:
                x_val, _ = state.quad_expectation(i)
            except Exception:
                x_val = float(i)
            x_means.append(x_val)

        # Ordena as cidades (índices 1 a C) de acordo com o valor esperado x
        ordered_cities = [
            city_idx for _, city_idx in sorted(zip(x_means, range(1, self.instance.C + 1)))
        ]

        depot = 0
        if self.instance.V == 1:
            return [depot] + ordered_cities + [depot]

        # Particiona a sequência entre V veículos com base na capacidade Q
        routes = {}
        city_ptr = 0
        num_cities = len(ordered_cities)

        for v_idx in range(1, self.instance.V + 1):
            sub_route = []
            curr_load = 0.0
            cap = (
                self.instance.Q[v_idx - 1]
                if len(self.instance.Q) >= v_idx
                else self.instance.Q[0]
            )

            while city_ptr < num_cities:
                city = ordered_cities[city_ptr]
                demand = (
                    float(self.instance.demands[city - 1])
                    if len(self.instance.demands) >= city
                    else 0.0
                )

                if curr_load + demand <= cap or v_idx == self.instance.V:
                    sub_route.append(city)
                    curr_load += demand
                    city_ptr += 1
                else:
                    break

            routes[v_idx] = [depot] + sub_route + [depot]

        return routes

    def solve(
        self,
        method: str = "ADAM",
        maxiter: int = 200,
        lr: float = 0.01,
        initial_theta: Optional[np.ndarray] = None,
    ) -> SolverMetrics:
        if initial_theta is None:
            initial_theta = self.ansatz.generate_initial_theta()

        self.cost_history = []
        start_time = time.time()

        if method.upper() == "ADAM":
            optimal_theta = self._adam_optimize(
                initial_theta=initial_theta,
                maxiter=maxiter,
                lr=lr,
            )
            final_energy = self.cost_history[-1] if self.cost_history else float("inf")
        else:
            res = minimize(
                fun=self._cost_function,
                x0=initial_theta,
                method=method,
                options={"maxiter": maxiter, "disp": False},
            )
            optimal_theta = res.x
            final_energy = float(res.fun)

        elapsed_time = time.time() - start_time

        # Executa o circuito final com os parâmetros otimizados para extrair o estado final e decodificar a rota
        opt_prog = self.ansatz.build_program(optimal_theta)
        eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff})
        final_results = eng.run(opt_prog)
        best_routes = self._extract_routes(final_results.state)

        return SolverMetrics(
            optimal_theta=optimal_theta,
            final_energy=final_energy,
            energy_components=self.last_energy_components,
            cost_history=self.cost_history,
            execution_time_seconds=elapsed_time,
            total_evaluations=len(self.cost_history),
            best_routes=best_routes,
        )