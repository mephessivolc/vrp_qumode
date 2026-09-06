from dataclasses import dataclass
import time
from typing import Dict, List
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

    def solve(
        self,
        method: str = "COBYLA",
        maxiter: int = 200,
        initial_theta: np.ndarray = None,
    ) -> SolverMetrics:
        if initial_theta is None:
            initial_theta = self.ansatz.generate_initial_theta()

        self.cost_history = []
        start_time = time.time()

        res = minimize(
            fun=self._cost_function,
            x0=initial_theta,
            method=method,
            options={"maxiter": maxiter, "disp": False},
        )

        elapsed_time = time.time() - start_time

        return SolverMetrics(
            optimal_theta=res.x,
            final_energy=float(res.fun),
            energy_components=self.last_energy_components,
            cost_history=self.cost_history,
            execution_time_seconds=elapsed_time,
            total_evaluations=len(self.cost_history),
        )