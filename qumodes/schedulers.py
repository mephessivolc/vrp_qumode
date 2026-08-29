# qumodes/schedulers.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BasePenaltyScheduler(ABC):
    """Classe base abstrata para agendadores dinâmicos de penalidades no Hamiltoniano."""

    @abstractmethod
    def __call__(self, step: int, hamiltonian: Any, last_metrics: Dict[str, Any]) -> None:
        """
        Atualiza os multiplicadores de penalidade do Hamiltoniano para o passo atual.

        Args:
            step: Número da iteração atual do otimizador.
            hamiltonian: Instância da classe Hamiltonian.
            last_metrics: Dicionário com as métricas e violações calculadas no passo anterior.
        """
        pass


class ExponentialPenaltyScheduler(BasePenaltyScheduler):
    """
    Agendador de penalidade com crescimento exponencial (Annealing).
    Aumenta gradualmente os multiplicadores ao longo das iterações por um fator constante gamma.
    """

    def __init__(
        self, 
        gamma: float = 1.05, 
        max_col_penalty: float = 1e4, 
        max_cap_penalty: float = 1e4
    ):
        self.gamma = gamma
        self.max_col_penalty = max_col_penalty
        self.max_cap_penalty = max_cap_penalty

    def __call__(self, step: int, hamiltonian: Any, last_metrics: Dict[str, Any]) -> None:
        if step == 0:
            return

        new_col = min(hamiltonian.lmbda_col * self.gamma, self.max_col_penalty)
        new_cap = min(hamiltonian.lmbda_cap * self.gamma, self.max_cap_penalty)

        hamiltonian.set_penalties(
            lmbda_col=new_col,
            lmbda_cap=new_cap
        )


class AugmentedLagrangianScheduler(BasePenaltyScheduler):
    """
    Agendador baseado no método do Lagrangeano Aumentado.
    Aumenta a penalidade apenas quando há violações ativas das restrições de colisão ou capacidade.
    """

    def __init__(
        self, 
        eta_col: float = 0.5, 
        eta_cap: float = 0.5, 
        max_col_penalty: float = 1e5, 
        max_cap_penalty: float = 1e5
    ):
        self.eta_col = eta_col
        self.eta_cap = eta_cap
        self.max_col_penalty = max_col_penalty
        self.max_cap_penalty = max_cap_penalty

    def __call__(self, step: int, hamiltonian: Any, last_metrics: Dict[str, Any]) -> None:
        if not last_metrics:
            return

        col_penalty_val = last_metrics.get("collision_penalty", 0.0)
        cap_violation = last_metrics.get("capacity_violation_magnitude", 0.0)

        # Atualização proporcional à magnitude das violações
        new_col = hamiltonian.lmbda_col
        if col_penalty_val > 0.0:
            new_col += self.eta_col * col_penalty_val

        new_cap = hamiltonian.lmbda_cap
        if cap_violation > 0.0:
            new_cap += self.eta_cap * (cap_violation ** 2)

        new_col = min(new_col, self.max_col_penalty)
        new_cap = min(new_cap, self.max_cap_penalty)

        hamiltonian.set_penalties(
            lmbda_col=new_col,
            lmbda_cap=new_cap
        )


class AdaptiveConstraintScheduler(BasePenaltyScheduler):
    """
    Agendador adaptativo que ajusta penalidades dinamicamente aumentando quando há violação
    e reduzindo ligeiramente quando a solução se mantém viável.
    """

    def __init__(
        self, 
        increase_factor: float = 1.1, 
        decrease_factor: float = 0.98,
        min_penalty: float = 1.0,
        max_penalty: float = 1e4
    ):
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty

    def __call__(self, step: int, hamiltonian: Any, last_metrics: Dict[str, Any]) -> None:
        if not last_metrics:
            return

        col_penalty_val = last_metrics.get("collision_penalty", 0.0)
        cap_violation = last_metrics.get("capacity_violation_magnitude", 0.0)

        # Reajuste para colisões
        if col_penalty_val > 0.0:
            new_col = hamiltonian.lmbda_col * self.increase_factor
        else:
            new_col = hamiltonian.lmbda_col * self.decrease_factor

        # Reajuste para capacidade
        if cap_violation > 0.0:
            new_cap = hamiltonian.lmbda_cap * self.increase_factor
        else:
            new_cap = hamiltonian.lmbda_cap * self.decrease_factor

        new_col = np.clip(new_col, self.min_penalty, self.max_penalty)
        new_cap = np.clip(new_cap, self.min_penalty, self.max_penalty)

        hamiltonian.set_penalties(
            lmbda_col=float(new_col),
            lmbda_cap=float(new_cap)
        )