"""
sf_bridge.py — ponte entre o backend Fock do Strawberry Fields e o Hamiltoniano
em operadores (qumodes/hamiltonian.py).

Colocar em  qumodes/sf_bridge.py

Por que existe
--------------
solver.py e initial_states.py ainda importavam da API antiga do hamiltonian.py
(evaluate_sf_state, extract_routes, hermite_functions), que não existe no módulo
em operadores. Esta ponte reimplementa as três sobre a API nova, sem alterar o
Hamiltoniano:

  evaluate_sf_state   energia (todos os termos) do estado final do circuito, via
                      H.energy_components sobre o ket do backend Fock.
  extract_routes      decodifica o ket: distribuição do POVM sobre as (M*R)^N
                      configurações, sorteio de `shots` configurações e rota
                      VÁLIDA de menor custo entre as sorteadas.
  hermite_functions   funções de onda de Fock phi_n(x), com x = sqrt(hbar/2)(a+a†),
                      usadas por initial_states.py.

Convenções
----------
* O ket de SF tem forma (cutoff,)*N, com o eixo i = modo i = cidade i, a mesma
  ordem de energy_components (conferido por verificar_ponte em exp_vqe.py).
* A energia é calculada sobre o estado RENORMALIZADO. A norma do ket do backend
  (perda por truncamento) vai em out["norm"].
* Formato de extract_routes: lista (M == 1) ou dict {veículo: rota}.
"""
from __future__ import annotations

import math
from typing import Dict, List, Union

import numpy as np

import qumodes.hamiltonian as H

__all__ = ["hermite_functions", "prepare", "evaluate_sf_state", "extract_routes",
           "routes_from_slots", "is_valid_assignment", "clear_cache"]

_CACHE: Dict[tuple, tuple] = {}


def clear_cache() -> None:
    _CACHE.clear()


# ---------------------------------------------------------------------------
# funções de Hermite
# ---------------------------------------------------------------------------
def hermite_functions(cutoff: int, x, hbar: float = 2.0) -> np.ndarray:
    """phi_n(x), n = 0..cutoff-1, forma (cutoff, len(x)).

    phi_0 = (pi hbar)^(-1/4) exp(-x²/(2 hbar)): sigma_x = sqrt(hbar/2), o que dá 1
    para hbar = 2. A recorrência reproduz x phi_n = sqrt(hbar/2) (sqrt(n+1) phi_{n+1}
    + sqrt(n) phi_{n-1}), isto é, x = sqrt(hbar/2)(a + a†) na base de Fock usual.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    xi = x / math.sqrt(hbar)
    out = np.empty((int(cutoff), x.size), dtype=np.float64)
    out[0] = (math.pi * hbar) ** -0.25 * np.exp(-0.5 * xi ** 2)
    if cutoff > 1:
        out[1] = math.sqrt(2.0) * xi * out[0]
    for n in range(1, int(cutoff) - 1):
        out[n + 1] = (math.sqrt(2.0 / (n + 1)) * xi * out[n]
                      - math.sqrt(n / (n + 1.0)) * out[n - 1])
    return out


# ---------------------------------------------------------------------------
# instância e operadores (com cache)
# ---------------------------------------------------------------------------
def prepare(M: int, N: int, D, demands, Q, cutoff: int, params, hbar: float = 2.0):
    """(inst, ops) para o problema; construídos uma vez por combinação."""
    D = np.asarray(D, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    req = bool(H.requires_all_vehicles(params, None))
    key = (int(M), int(N), int(cutoff), req, float(hbar),
           D.tobytes(), demands.tobytes(), Q.tobytes())
    hit = _CACHE.get(key)
    if hit is None:
        inst = H.instance_from_legacy(N, M, D, demands, Q, hbar=hbar,
                                      require_all_vehicles=req)
        ops = H.build_modular_operators(cutoff, inst.M, inst.R, hbar)
        hit = (inst, ops)
        _CACHE[key] = hit
    return hit


def _ket_of(state, cutoff: int, N: int) -> np.ndarray:
    ket = state.ket()
    if ket is None:
        raise ValueError(
            f"state.ket() devolveu None (state.is_pure = {getattr(state, 'is_pure', '?')}). "
            "O backend Fock precisa devolver estado PURO. Causa provável: estado "
            "inicial preparado com um Ket por modo; use um único Ket multimodo "
            "(ansatz.py corrigido) ou parta do vácuo.")
    return np.asarray(ket, dtype=np.complex128).reshape((int(cutoff),) * int(N))


# ---------------------------------------------------------------------------
# energia
# ---------------------------------------------------------------------------
def evaluate_sf_state(state, M: int, N: int, D, demands, Q, cutoff: int,
                      params) -> Dict[str, float]:
    """Energia do estado de SF: dist, col, gap, vehicle, capacity, disc,
    encoding, mean_photons, exc, total, mais 'norm' (norma do ket do backend)."""
    inst, ops = prepare(M, N, D, demands, Q, cutoff, params)
    ket = _ket_of(state, cutoff, N)
    ec = H.energy_components(ket, inst, ops, params)
    out = {k: float(v) for k, v in ec.items() if np.ndim(v) == 0}
    out["norm"] = float(np.real(np.vdot(ket, ket)))
    return out


# ---------------------------------------------------------------------------
# decodificação
# ---------------------------------------------------------------------------
def routes_from_slots(slots, inst) -> List[List[int]]:
    """Cada cidade i está no nó (v_i, r_i), com slot = v*R + r. Dentro de cada
    veículo, ordena por (r, i)."""
    R = inst.R
    v = [int(s) // R for s in slots]
    r = [int(s) % R for s in slots]
    routes = []
    for veh in range(inst.M):
        cidades = [i for i in range(inst.N) if v[i] == veh]
        cidades.sort(key=lambda i: (r[i], i))
        routes.append([i + 1 for i in cidades])
    return routes


def is_valid_assignment(slots, inst, require_all: bool) -> bool:
    """Em cada veículo as posições são distintas e formam {0, ..., k-1}; com
    require_all, todo veículo atende ao menos uma cidade."""
    R = inst.R
    pos = [[] for _ in range(inst.M)]
    for s in slots:
        pos[int(s) // R].append(int(s) % R)
    for p in pos:
        if sorted(p) != list(range(len(p))):
            return False
    if require_all and any(len(p) == 0 for p in pos):
        return False
    return True


def extract_routes(state, N: int, M: int, D, demands, Q, cutoff: int, params,
                   shots: int = 2048, seed: int = 0
                   ) -> Union[List[int], Dict[int, List[int]]]:
    """Rota VÁLIDA de menor custo entre `shots` configurações sorteadas da
    distribuição do POVM. Se nenhuma for válida, devolve a mais provável."""
    inst, ops = prepare(M, N, D, demands, Q, cutoff, params)
    ket = _ket_of(state, cutoff, N)
    dist = np.asarray(H.assignment_distribution(ket, inst, ops), dtype=np.float64)
    flat = np.clip(dist.reshape(-1), 0.0, None)
    flat = flat / flat.sum()
    rng = np.random.default_rng(seed)
    sorteio = np.unique(rng.choice(flat.size, size=int(shots), p=flat))
    req = bool(H.requires_all_vehicles(params, None))

    melhor = None
    for f in sorteio:
        slots = np.unravel_index(int(f), dist.shape)
        if not is_valid_assignment(slots, inst, req):
            continue
        rotas = routes_from_slots(slots, inst)
        custo = H.routes_cost(rotas, inst.D)
        if melhor is None or custo < melhor[0]:
            melhor = (custo, rotas)
    if melhor is None:
        slots = np.unravel_index(int(np.argmax(flat)), dist.shape)
        rotas = routes_from_slots(slots, inst)
    else:
        rotas = melhor[1]
    return rotas[0] if M == 1 else {v: r for v, r in enumerate(rotas)}