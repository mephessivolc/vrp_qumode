"""
initial_states.py — estados iniciais em escala de Zak para o ansatz.

Motivação
---------
O vácuo tem sigma_x = sqrt(hbar/2) = 1 (hbar = 2), enquanto a rede modular vale
a = sqrt(2*pi*hbar) ~ 3.545. O vácuo cabe, portanto, dentro de UMA célula: não
tem estrutura para resolver os nós de posição theta_r = 2*pi*(r-1)/R, e em
coordenadas de Zak corresponde a "todas as cidades no nó 0", que é colisão
máxima. Daí a probabilidade de viabilidade do vácuo ficar abaixo da uniforme.

Duas classes de estado inicial são fornecidas:

squeezed_ket  gaussiano espremido. O mínimo do termo estabilizador
              h_disc = 2 - <cos(R.theta)> - <cos(M.phi)> sobre gaussianos é
              ~1.0 por modo, atingido apenas com squeezing alto, e esse piso
              NÃO depende de R nem de M: ganho em theta é pago em phi.

gkp_ket       GKP de energia finita (pente de gaussianas de largura Delta,
              espaçadas de a, sob envelope de largura kappa). Desacopla as duas
              exigências e desce abaixo de 1.0, ao custo de <n>. O <n>
              necessário cresce com R:
                  R=3 -> h_disc ~ 0.53 com <n> ~ 10
                  R=4 -> h_disc ~ 0.74 com <n> ~ 10
                  R=5 -> h_disc ~ 0.89 com <n> ~ 10
              Como <n> ~ 10 pede cutoff ~ 30, e cutoff 30 com C = 4 estoura o
              zak_grid_limit, a demonstração GKP deve ser feita com C = 3.
"""
from typing import List, Optional
import math
import numpy as np

from qumodes.hamiltonian import hermite_functions


def _project_to_fock(psi_x: np.ndarray, x: np.ndarray, cutoff: int,
                     hbar: float = 2.0) -> np.ndarray:
    """Projeta uma função de onda amostrada em x na base de Fock truncada."""
    phi_n = hermite_functions(cutoff, x, hbar)          # (cutoff, len(x))
    ket = phi_n @ psi_x * (x[1] - x[0])
    nrm = np.linalg.norm(ket)
    if nrm < 1e-12:
        raise ValueError("projeção nula: verifique a grade em x e os parâmetros.")
    return ket / nrm


def _x_grid(a: float, n_cells: int, pts_per_cell: int) -> np.ndarray:
    half = n_cells * a / 2.0
    return np.linspace(-half, half, n_cells * pts_per_cell)


def fock_truncation_loss(ket: np.ndarray) -> float:
    """Fração da norma no último nível de Fock — sinal de cutoff insuficiente."""
    return float(abs(ket[-1]) ** 2)


def mean_photons(ket: np.ndarray) -> float:
    n = np.arange(len(ket))
    return float(np.sum(n * np.abs(ket) ** 2))


def squeezed_ket(cutoff: int, r: float, x0: float = 0.0, hbar: float = 2.0,
                 a: Optional[float] = None, n_cells: int = 16,
                 pts_per_cell: int = 256) -> np.ndarray:
    """Gaussiano espremido centrado em x0. sigma_x = sqrt(hbar/2)*exp(-r)."""
    a = a or math.sqrt(2.0 * math.pi * hbar)
    sigma = math.sqrt(hbar / 2.0) * math.exp(-r)
    x = _x_grid(a, n_cells, pts_per_cell)
    psi = np.exp(-((x - x0) ** 2) / (4.0 * sigma ** 2))
    return _project_to_fock(psi, x, cutoff, hbar)


def gkp_ket(cutoff: int, Delta: float, kappa: float, R: int = 1, r_node: int = 0,
            hbar: float = 2.0, a: Optional[float] = None, n_cells: int = 16,
            pts_per_cell: int = 256) -> np.ndarray:
    """GKP de energia finita, deslocado para o nó de posição r_node.

    Delta   largura de cada pico (resolve theta): precisa de Delta << a/(2*pi*R).
    kappa   largura do envelope (resolve phi): precisa de kappa >> hbar/(2*pi/M).
    r_node  nó de posição alvo; o pente é centrado em r_node*a/R.
    """
    a = a or math.sqrt(2.0 * math.pi * hbar)
    x = _x_grid(a, n_cells, pts_per_cell)
    x0 = (r_node % max(R, 1)) * a / max(R, 1)
    s = np.arange(-n_cells, n_cells + 1)[:, None]
    comb = np.exp(-((x[None, :] - x0 - s * a) ** 2) / (4.0 * Delta ** 2)).sum(0)
    psi = comb * np.exp(-((x - x0) ** 2) / (4.0 * kappa ** 2))
    return _project_to_fock(psi, x, cutoff, hbar)


def build_initial_kets(kind: str, N: int, cutoff: int, R: int = 1,
                       hbar: float = 2.0, Delta: float = 0.35,
                       kappa: float = 4.0, squeeze_r: float = 1.2,
                       nodes: Optional[List[int]] = None) -> Optional[List[np.ndarray]]:
    """Lista de N kets, um por cidade, para ContinuousVariableAnsatz(initial_ket=...).

    kind: "vacuum" (devolve None, o ansatz parte do vácuo), "squeezed" ou "gkp".
    nodes: nó de posição inicial de cada cidade. O padrão distribui as cidades em
           nós distintos (0, 1, 2, ...), que é o análogo contínuo do warm start:
           uma permutação viável em vez de todas as cidades no mesmo nó.
    """
    kind = (kind or "vacuum").lower()
    if kind == "vacuum":
        return None
    if nodes is None:
        nodes = [i % max(R, 1) for i in range(N)]
    if len(nodes) != N:
        raise ValueError(f"nodes deve ter {N} entradas.")

    a = math.sqrt(2.0 * math.pi * hbar)
    if kind == "squeezed":
        return [squeezed_ket(cutoff, squeeze_r, x0=nodes[i] * a / max(R, 1), hbar=hbar)
                for i in range(N)]
    if kind == "gkp":
        return [gkp_ket(cutoff, Delta, kappa, R=R, r_node=nodes[i], hbar=hbar)
                for i in range(N)]
    raise ValueError(f"kind desconhecido: {kind!r} (use vacuum, squeezed ou gkp).")


def describe_kets(kets: Optional[List[np.ndarray]]) -> dict:
    """Diagnóstico para registrar no JSON do experimento."""
    if kets is None:
        return {"kind": "vacuum", "mean_photons": [0.0], "max_truncation_loss": 0.0}
    return {"mean_photons": [mean_photons(k) for k in kets],
            "max_truncation_loss": max(fock_truncation_loss(k) for k in kets)}
