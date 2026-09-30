from dataclasses import dataclass, field
import math
import time
from typing import Dict, List, Optional, Union
import numpy as np
from scipy.optimize import minimize
import strawberryfields as sf

# Importações relativas internas do pacote qumodes
from .ansatz import CircuitConfig, ContinuousVariableAnsatz
from .hamiltonian import HamiltonianParams, evaluate_sf_state, extract_routes


@dataclass
class ProblemInstance:
    V: int
    C: int
    D: np.ndarray
    demands: np.ndarray
    Q: np.ndarray


@dataclass
class SolverMetrics:
    """Métricas de uma execução do VQE.

    Distinção central entre os dois registros de energia:

    cost_history      toda avaliação do circuito, inclusive as 2P sondagens
                      theta +/- eps da diferença finita. Serve para contabilizar
                      custo computacional, NÃO para plotar convergência: as
                      sondagens de uma mesma iteração têm energia quase idêntica
                      e produzem o artefato de "escada".

    iterate_history   energia nos iterados do ADAM, E(theta_t), com exatamente
                      maxiter + 1 pontos. É a curva de convergência.
    """
    optimal_theta: np.ndarray
    final_energy: float
    energy_components: Dict[str, float]
    cost_history: List[float]
    execution_time_seconds: float
    total_evaluations: int
    best_routes: Union[List[int], Dict[int, List[int]]]
    # --- novos ---
    iterate_history: List[float] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)
    iterations_completed: int = 0
    circuit_evaluations: int = 0
    evaluations_per_iteration: Optional[int] = None
    best_iterate_energy: Optional[float] = None
    best_iterate_index: Optional[int] = None

    @property
    def energy_per_iteration(self) -> List[float]:
        """Alias explícito para uso em figuras e no JSON de resultados."""
        return self.iterate_history


class VQESolver:
    """VQE em variáveis contínuas para o TSP/CVRP na codificação modular.

    Parâmetros de instrumentação
    ----------------------------
    record_probes : mantém em `cost_history` as avaliações de gradiente
        (theta +/- eps). True preserva a compatibilidade com
        `analysis.reconstruct_adam_trace`, que reconstrói o traço a partir do
        histórico bruto. False deixa `cost_history` só com os iterados, o que
        economiza memória em execuções longas mas invalida aquela reconstrução.
        Em qualquer dos casos `iterate_history` é preenchido diretamente.
    """

    def __init__(
        self,
        instance: ProblemInstance,
        circuit_config: CircuitConfig,
        hamiltonian_params: HamiltonianParams,
        cutoff: int = 10,
        record_probes: bool = True,
    ):
        self.instance = instance
        self.circuit_config = circuit_config
        self.h_params = hamiltonian_params
        self.cutoff = cutoff
        self.record_probes = record_probes

        self.ansatz = ContinuousVariableAnsatz(config=circuit_config)
        self.cost_history: List[float] = []
        self.iterate_history: List[float] = []
        self.grad_norm_history: List[float] = []
        self.lr_history: List[float] = []
        self.circuit_evaluations: int = 0
        self.last_energy_components: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Avaliação
    # ------------------------------------------------------------------
    def _reset_history(self) -> None:
        self.cost_history = []
        self.iterate_history = []
        self.grad_norm_history = []
        self.lr_history = []
        self.circuit_evaluations = 0

    def _cost_function(self, theta: np.ndarray, record: bool = True) -> float:
        """Energia do estado preparado por theta.

        `record` controla apenas o registro em `cost_history`; a contagem
        `circuit_evaluations` é sempre incrementada, porque o custo
        computacional é o mesmo.
        """
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
        self.circuit_evaluations += 1
        if record:
            self.cost_history.append(total_cost)
        self.last_energy_components = energies

        return total_cost

    def _compute_numerical_gradient(
        self, theta: np.ndarray, eps: float = 1e-2
    ) -> np.ndarray:
        """Gradiente numérico por diferença finita central.

        O padrão de `eps` é 1e-2, e não 1e-4: a diferença central amplifica o
        ruído da simulação de Fock por um fator 1/(2 eps), de modo que passos
        pequenos demais degradam a direção do gradiente em vez de refiná-la.
        """
        grad = np.zeros_like(theta)
        record = self.record_probes
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_minus = theta.copy()

            theta_plus[i] += eps
            theta_minus[i] -= eps

            c_plus = self._cost_function(theta_plus, record=record)
            c_minus = self._cost_function(theta_minus, record=record)

            grad[i] = (c_plus - c_minus) / (2.0 * eps)
        return grad

    # ------------------------------------------------------------------
    # Otimização
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_lr(lr0: float, lr_final: Optional[float], t: int, maxiter: int) -> float:
        """Decaimento cosseno de lr0 a lr_final ao longo de maxiter iterações."""
        if lr_final is None or maxiter <= 1:
            return lr0
        frac = (t - 1) / (maxiter - 1)
        return lr_final + 0.5 * (lr0 - lr_final) * (1.0 + math.cos(math.pi * frac))

    def _adam_optimize(
        self,
        initial_theta: np.ndarray,
        maxiter: int = 200,
        lr: float = 0.01,
        lr_final: Optional[float] = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps_adam: float = 1e-8,
        fd_eps: float = 1e-2,
    ) -> np.ndarray:
        """ADAM com gradiente por diferenças finitas.

        Registra uma energia por iteração em `iterate_history`, com exatamente
        maxiter + 1 pontos (o ponto inicial e um por atualização de theta).
        """
        theta = initial_theta.copy()
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)

        self.iterate_history.append(self._cost_function(theta))

        for t in range(1, maxiter + 1):
            grad = self._compute_numerical_gradient(theta, eps=fd_eps)
            self.grad_norm_history.append(float(np.linalg.norm(grad)))

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad**2)

            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)

            lr_t = self._cosine_lr(lr, lr_final, t, maxiter)
            self.lr_history.append(lr_t)

            theta = theta - lr_t * m_hat / (np.sqrt(v_hat) + eps_adam)
            self.iterate_history.append(self._cost_function(theta))

        return theta

    def _extract_routes(self, state) -> Union[List[int], Dict[int, List[int]]]:
        """
        Decodifica o estado otimizado pela codificação modular (x mod a -> posição,
        p mod b -> veículo): sorteia configurações da distribuição de Zak e devolve
        a rota viável de menor custo. Formato: lista (V == 1) ou dict {veículo: rota}.
        """
        return extract_routes(
            state=state,
            N=self.instance.C,
            M=self.instance.V,
            D=self.instance.D,
            demands=self.instance.demands,
            Q=self.instance.Q,
            cutoff=self.cutoff,
            params=self.h_params,
        )

    def solve(
        self,
        method: str = "ADAM",
        maxiter: int = 200,
        lr: float = 0.01,
        lr_final: Optional[float] = None,
        fd_eps: float = 1e-2,
        initial_theta: Optional[np.ndarray] = None,
    ) -> SolverMetrics:
        if initial_theta is None:
            initial_theta = self.ansatz.generate_initial_theta()

        self._reset_history()
        start_time = time.time()

        n_params = len(initial_theta)
        evals_per_iter = None

        if method.upper() == "ADAM":
            optimal_theta = self._adam_optimize(
                initial_theta=initial_theta,
                maxiter=maxiter,
                lr=lr,
                lr_final=lr_final,
                fd_eps=fd_eps,
            )
            final_energy = self.iterate_history[-1]
            evals_per_iter = 2 * n_params + 1
        else:
            res = minimize(
                fun=lambda th: self._cost_function(th),
                x0=initial_theta,
                method=method,
                options={"maxiter": maxiter, "disp": False},
            )
            optimal_theta = res.x
            final_energy = float(res.fun)
            # sem iterados explícitos: o histórico bruto é a melhor aproximação
            self.iterate_history = list(self.cost_history)

        elapsed_time = time.time() - start_time

        # Circuito final com os parâmetros otimizados, para extrair o estado e
        # decodificar a rota. Não entra em nenhum histórico de convergência.
        opt_prog = self.ansatz.build_program(optimal_theta)
        eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff})
        final_results = eng.run(opt_prog)
        best_routes = self._extract_routes(final_results.state)

        best_idx = int(np.argmin(self.iterate_history)) if self.iterate_history else None
        best_val = float(self.iterate_history[best_idx]) if best_idx is not None else None

        return SolverMetrics(
            optimal_theta=optimal_theta,
            final_energy=float(final_energy),
            energy_components=self.last_energy_components,
            cost_history=self.cost_history,
            execution_time_seconds=elapsed_time,
            total_evaluations=self.circuit_evaluations,
            best_routes=best_routes,
            iterate_history=self.iterate_history,
            grad_norm_history=self.grad_norm_history,
            lr_history=self.lr_history,
            iterations_completed=max(0, len(self.iterate_history) - 1),
            circuit_evaluations=self.circuit_evaluations,
            evaluations_per_iteration=evals_per_iter,
            best_iterate_energy=best_val,
            best_iterate_index=best_idx,
        )
