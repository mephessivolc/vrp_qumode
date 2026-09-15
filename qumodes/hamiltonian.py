from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
import strawberryfields as sf


@dataclass
class HamiltonianParams:
    """Parâmetros e penalidades do Hamiltoniano CCV-Hermitiano."""
    sigma_assign: float = 0.30
    sigma_empty: float = 0.40
    lambda_col: float = 25.0
    lambda_gap: float = 20.0
    lambda_cap: float = 18.0
    alpha2: float = 1.0
    alpha4: float = 1.0
    lambda_dist: float = 1.0


def distance_term(x_exp: np.ndarray, p_exp: np.ndarray, D: np.ndarray, lambda_dist: float = 1.0) -> float:
    """H_dist: Custo de deslocamento acumulado sobre a matriz de distâncias D."""
    num_qumodes = len(x_exp)
    h_dist = 0.0
    for i in range(num_qumodes):
        j = (i + 1) % num_qumodes
        h_dist += D[i, j] * 0.5 * (x_exp[i] * p_exp[j] + p_exp[i] * x_exp[j])
    return float(lambda_dist * h_dist)


def collision_term(x_exp: np.ndarray, p_exp: np.ndarray, lambda_col: float) -> float:
    """H_col: Penalidade de viabilidade e colisão no circuito/rota."""
    sum_x = np.sum(x_exp)
    sum_p = np.sum(p_exp)
    h_col = (sum_x - 1.0)**2 + (sum_p - 1.0)**2 + np.sum(((x_exp + p_exp) - 1.0)**2)
    return float(lambda_col * h_col)


def gap_term(x_exp: np.ndarray, p_exp: np.ndarray, lambda_gap: float) -> float:
    """H_gap: Potencial quártico para forçar a binarização das quadraturas {0, 1}."""
    h_bin_x = np.sum((x_exp**2) * ((x_exp - 1.0)**2))
    h_bin_p = np.sum((p_exp**2) * ((p_exp - 1.0)**2))
    return float(lambda_gap * (h_bin_x + h_bin_p))


def capacity_term(
    x_exp: np.ndarray,
    demands: np.ndarray,
    Q: Union[float, np.ndarray],
    lambda_cap: float,
    alpha2: float = 1.0,
    alpha4: float = 1.0,
) -> float:
    """H_cap: Penalidade de violação da capacidade máxima dos veículos Q."""
    if lambda_cap <= 0.0:
        return 0.0

    demands = np.asarray(demands, dtype=np.float64)
    Q_val = np.sum(Q) if isinstance(Q, (np.ndarray, list)) else float(Q)

    assigned_load = np.sum(demands[:len(x_exp)] * x_exp)
    excess = assigned_load - Q_val

    if excess <= 0:
        return 0.0

    hinge = alpha2 * (excess**2) + alpha4 * (excess**4)
    return float(lambda_cap * hinge)


def hamiltonian_energy(
    x_exp: np.ndarray,
    p_exp: np.ndarray,
    D: np.ndarray,
    demands: np.ndarray,
    Q: Union[float, np.ndarray],
    params: Optional[HamiltonianParams] = None,
) -> Dict[str, float]:
    """Calcula a energia total do Hamiltoniano e decompõe nas componentes individuais."""
    if params is None:
        params = HamiltonianParams()

    D = np.asarray(D, dtype=np.float64)

    h_dist = distance_term(x_exp, p_exp, D, lambda_dist=params.lambda_dist)
    h_col = collision_term(x_exp, p_exp, lambda_col=params.lambda_col)
    h_gap = gap_term(x_exp, p_exp, lambda_gap=params.lambda_gap)
    h_cap = capacity_term(
        x_exp,
        demands,
        Q,
        lambda_cap=params.lambda_cap,
        alpha2=params.alpha2,
        alpha4=params.alpha4,
    )

    h_total = h_dist + h_col + h_gap + h_cap

    return {
        "total": float(h_total),
        "dist": float(h_dist),
        "col": float(h_col),
        "gap": float(h_gap),
        "capacity": float(h_cap),
    }


def evaluate_sf_state(
    state: sf.backends.states.BaseState,
    N: int,
    M: int,
    D: np.ndarray,
    demands: np.ndarray,
    Q: Union[float, np.ndarray],
    cutoff: int = 8,
    params: Optional[HamiltonianParams] = None,
) -> Dict[str, float]:
    """Extrai os valores esperados do estado quântico e calcula as energias (compatível com solver.py)."""
    num_qumodes = N
    x_exp = np.array([state.quad_expectation(i, phi=0)[0] for i in range(num_qumodes)])
    p_exp = np.array([state.quad_expectation(i, phi=np.pi / 2)[0] for i in range(num_qumodes)])

    return hamiltonian_energy(
        x_exp=x_exp,
        p_exp=p_exp,
        D=D,
        demands=demands,
        Q=Q,
        params=params,
    )