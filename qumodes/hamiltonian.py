"""
hamiltonian.py — Hamiltoniano modular (Zak/Fejér) para o VRP/CVRP em qumodes.

Codificação
-----------
Um qumode por cidade i. Posição na rota em x̂_i, veículo em p̂_i, via as
quadraturas modulares

    θ_i = 2π x̂_i / a  (mod 2π),     φ_i = 2π p̂_i / b  (mod 2π),     a·b = 2πħ.

Com a·b = 2πħ, as funções periódicas de x̂_i e p̂_i comutam (representação de
Zak), de modo que o Hamiltoniano é uma função F(θ, φ) de operadores que
comutam e inf spec(Ĥ) = min F.

Nós da grade:  θ_r = 2π(r-1)/R  (r = 1..R),   φ_v = 2π(v-1)/M  (v = 1..M).

Atribuição (núcleo de Fejér):  a_{i,v,r} = F_R(θ_i - θ_r) · F_M(φ_i - φ_v),
exata nos nós (delta de Kronecker), com derivada nula nos nós e soma 1.

Termos
------
    H = λ_dist·H_dist + λ_col·H_col + λ_gap·H_gap + λ_vehicle·H_vehicle
        + λ_cap·H_cap + λ_disc·H_disc

Convenção de ħ: a mesma do Strawberry Fields (x̂ = sqrt(ħ/2)(â + â†)), ħ = 2
por padrão.

Integração: `evaluate_sf_state` e `extract_routes` mantêm a assinatura usada
por solver.py (N=C, M=V, D, demands, Q, cutoff, params) e o formato de rotas
esperado por run.py.

Índices internos são 0-based: cidade i = 0..N-1 corresponde ao nó i+1 da
matriz de distâncias; veículo v = 0..M-1; posição r = 0..R-1.
"""
from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    # integração com solver.py / run.py
    "HamiltonianParams", "evaluate_sf_state", "extract_routes", "check_penalties",
    "instance_from_legacy", "auto_positions", "requires_all_vehicles", "vehicle_penalty_bound",
    "clear_cache", "ZakEvaluator", "zak_grid_points", "get_max_grid_points", "set_max_grid_points",
    # núcleo do modelo
    "VRPInstance", "fejer", "quadratures_to_angles", "assignments", "hamiltonian_energy",
    "suggest_penalties", "decode", "routes_cost", "is_feasible",
    "brute_force_grid", "continuous_minimum", "ideal_energy_table",
    "estimate_from_samples", "hermite_functions", "zak_distribution_from_ket",
    "expectation_from_distribution", "sample_from_distribution",
]


# ---------------------------------------------------------------------------
# Parâmetros e instância
# ---------------------------------------------------------------------------
@dataclass
class HamiltonianParams:
    """Pesos do Hamiltoniano.

    Campos compatíveis com a versão anterior (usados por run.py):
    sigma_assign, sigma_empty, alpha2 e alpha4 são aceitos, mas IGNORADOS no
    modelo modular (a atribuição é exata pelo núcleo de Fejér).

    Use `check_penalties` antes do VQE para confirmar que o mínimo do
    Hamiltoniano coincide com o ótimo clássico.
    """
    # --- legado (aceitos e ignorados) ---
    sigma_assign: float = 0.30
    sigma_empty: float = 0.40
    # --- pesos ---
    lambda_col: float = 25.0
    lambda_gap: float = 20.0
    lambda_cap: float = 18.0
    # --- legado (aceitos e ignorados) ---
    alpha2: float = 1.0
    alpha4: float = 1.0
    # --- pesos ---
    lambda_dist: float = 1.0
    lambda_vehicle: float = 0.0   # > 0: todos os M veículos são obrigatórios (como no BruteForce)
    lambda_disc: float = 5.0      # evita mínimos fracionários; confira com check_penalties

    def weights(self) -> Tuple[float, ...]:
        return (self.lambda_dist, self.lambda_col, self.lambda_gap,
                self.lambda_vehicle, self.lambda_cap, self.lambda_disc)


@dataclass
class VRPInstance:
    """Instância do VRP.

    D        : matriz (N+1)x(N+1); o nó 0 é o depósito.
    M        : número de veículos.
    R        : posições por veículo (padrão N). Se a capacidade é em número de
               cidades (demandas unitárias), use R = Q e dispense λ_cap.
    demands  : demandas das N cidades (necessário se λ_cap > 0).
    Q        : capacidade (escalar, igual para todos) ou vetor com M capacidades.
    hbar     : convenção de ħ (Strawberry Fields usa 2).
    """
    D: np.ndarray
    M: int
    R: Optional[int] = None
    demands: Optional[np.ndarray] = None
    Q: Optional[object] = None
    hbar: float = 2.0

    def __post_init__(self) -> None:
        self.D = np.asarray(self.D, dtype=np.float64)
        if self.D.ndim != 2 or self.D.shape[0] != self.D.shape[1] or self.D.shape[0] < 2:
            raise ValueError("D deve ser quadrada (N+1)x(N+1), com o depósito no índice 0.")
        if self.R is None:
            self.R = self.N
        if self.M < 1 or self.R < 1:
            raise ValueError("M e R devem ser >= 1.")
        if self.M * self.R < self.N:
            raise ValueError(f"M·R = {self.M * self.R} < N = {self.N}: não há slots suficientes.")
        if self.demands is not None:
            self.demands = np.asarray(self.demands, dtype=np.float64).reshape(-1)
            if self.demands.shape == (self.N + 1,):       # vetor com o depósito
                self.demands = self.demands[1:]
            if self.demands.shape != (self.N,):
                raise ValueError("demands deve ter comprimento N (ou N+1 incluindo o depósito).")
        if self.Q is not None:
            q = np.asarray(self.Q, dtype=np.float64).reshape(-1)
            if q.size == 1:
                q = np.full(self.M, q[0])
            if q.size != self.M:
                raise ValueError("Q deve ser escalar ou ter comprimento M.")
            self.Q = q

    # --- tamanhos -------------------------------------------------------
    @property
    def N(self) -> int:
        return self.D.shape[0] - 1

    @property
    def d0(self) -> np.ndarray:           # depósito -> cidade
        return self.D[0, 1:]

    @property
    def dback(self) -> np.ndarray:        # cidade -> depósito
        return self.D[1:, 0]

    @property
    def Dcc(self) -> np.ndarray:          # cidade -> cidade, diagonal zerada
        Dc = self.D[1:, 1:].copy()
        np.fill_diagonal(Dc, 0.0)
        return Dc

    # --- rede modular ---------------------------------------------------
    @property
    def a(self) -> float:
        return math.sqrt(2.0 * math.pi * self.hbar)

    @property
    def b(self) -> float:
        return 2.0 * math.pi * self.hbar / self.a   # garante a·b = 2πħ

    @property
    def theta_nodes(self) -> np.ndarray:
        return 2.0 * np.pi * np.arange(self.R) / self.R

    @property
    def phi_nodes(self) -> np.ndarray:
        return 2.0 * np.pi * np.arange(self.M) / self.M


# ---------------------------------------------------------------------------
# Codificação
# ---------------------------------------------------------------------------
def fejer(t: np.ndarray, K: int) -> np.ndarray:
    """Núcleo de Fejér F_K(t) = |Σ_{k<K} e^{ikt}|²/K² (série de Fourier finita)."""
    t = np.asarray(t, dtype=np.float64)
    out = np.full(t.shape, 1.0 / K)
    for m in range(1, K):
        out += 2.0 * (K - m) / K**2 * np.cos(m * t)
    return out


def quadratures_to_angles(x: np.ndarray, p: np.ndarray, inst: VRPInstance) -> Tuple[np.ndarray, np.ndarray]:
    """Converte quadraturas (x, p) em ângulos modulares (θ, φ) ∈ [0, 2π)."""
    theta = np.mod(2.0 * np.pi * np.asarray(x, dtype=np.float64) / inst.a, 2.0 * np.pi)
    phi = np.mod(2.0 * np.pi * np.asarray(p, dtype=np.float64) / inst.b, 2.0 * np.pi)
    return theta, phi


def assignments(theta: np.ndarray, phi: np.ndarray, inst: VRPInstance) -> np.ndarray:
    """a[..., i, v, r] = F_R(θ_i - θ_r) F_M(φ_i - φ_v).  theta, phi: (..., N)."""
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    fr = fejer(theta[..., :, None] - inst.theta_nodes, inst.R)   # (..., N, R)
    fv = fejer(phi[..., :, None] - inst.phi_nodes, inst.M)       # (..., N, M)
    return fv[..., :, :, None] * fr[..., :, None, :]             # (..., N, M, R)


# ---------------------------------------------------------------------------
# Energia (símbolo clássico F(θ, φ); exato como operador por comutarem)
# ---------------------------------------------------------------------------
def _terms(a: np.ndarray, inst: VRPInstance) -> Dict[str, np.ndarray]:
    n = a.sum(axis=-3)                                            # (..., M, R)

    # distância
    h_start = np.einsum("...iv,i->...", a[..., 0], inst.d0)
    h_last = np.einsum("...iv,i->...", a[..., -1], inst.dback)
    if inst.R > 1:
        h_internal = np.einsum("...ivr,ij,...jvr->...", a[..., :-1], inst.Dcc, a[..., 1:])
        h_ret = np.einsum("...ivr,i,...vr->...", a[..., :-1], inst.dback, 1.0 - n[..., 1:])
    else:
        h_internal = np.zeros(a.shape[:-3])
        h_ret = np.zeros(a.shape[:-3])
    h_dist = h_start + h_internal + h_ret + h_last

    # colisão: Σ_{v,r} Σ_{i<j} a_i a_j = ½ Σ (n² − Σ_i a²)
    h_col = 0.5 * np.sum(n**2 - np.sum(a**2, axis=-3), axis=(-2, -1))

    # lacunas: Σ_v Σ_{r≥2} n_{v,r} [(r−1) − Σ_{q<r} n_{v,q}]²
    if inst.R > 1:
        prefix = np.cumsum(n, axis=-1)[..., :-1]                  # Σ_{q<r} n, r = 2..R
        target = np.arange(1, inst.R, dtype=np.float64)
        h_gap = np.sum(n[..., 1:] * (target - prefix) ** 2, axis=(-2, -1))
    else:
        h_gap = np.zeros(a.shape[:-3])

    # uso de veículos
    h_vehicle = np.sum((n[..., 0] - 1.0) ** 2, axis=-1)

    return {"dist": h_dist, "col": h_col, "gap": h_gap, "vehicle": h_vehicle, "n": n}


def hamiltonian_energy(
    theta: np.ndarray,
    phi: np.ndarray,
    inst: VRPInstance,
    params: Optional[HamiltonianParams] = None,
) -> Dict[str, np.ndarray]:
    """Energia total e componentes (já ponderadas) em pontos (θ, φ).

    theta, phi: arrays (..., N). Aceita lotes (ex.: amostras ou grade).
    """
    params = params or HamiltonianParams()
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    a = assignments(theta, phi, inst)
    t = _terms(a, inst)

    # capacidade: Σ_v max(0, L_v − Q)², L_v = Σ_i q_i Σ_r a_{i,v,r}
    if params.lambda_cap > 0.0:
        if inst.demands is None or inst.Q is None:
            raise ValueError("λ_cap > 0 exige demands e Q na instância.")
        load = np.einsum("...iv,i->...v", a.sum(axis=-1), inst.demands)
        h_cap = np.sum(np.maximum(0.0, load - inst.Q) ** 2, axis=-1)   # Q por veículo
    else:
        h_cap = np.zeros(theta.shape[:-1])

    # discretização (tipo estabilizador GKP)
    h_disc = np.sum(2.0 - np.cos(inst.R * theta) - np.cos(inst.M * phi), axis=-1)

    out = {
        "dist": params.lambda_dist * t["dist"],
        "col": params.lambda_col * t["col"],
        "gap": params.lambda_gap * t["gap"],
        "vehicle": params.lambda_vehicle * t["vehicle"],
        "capacity": params.lambda_cap * h_cap,
        "disc": params.lambda_disc * h_disc,
    }
    out["total"] = sum(out.values())
    return out


# ---------------------------------------------------------------------------
# Limiares suficientes na grade discreta
# ---------------------------------------------------------------------------
def _min_positive_excess(demands: np.ndarray, Q: float, max_sums: int = 2_000_000) -> float:
    """Menor excesso L − Q > 0 entre as cargas atingíveis (somas de subconjuntos)."""
    sums = {0.0}
    for q in demands:
        sums |= {round(s + q, 9) for s in sums}
        if len(sums) > max_sums:
            raise RuntimeError("Somas de subconjuntos demais; informe delta_cap manualmente.")
    excess = [s - Q for s in sums if s - Q > 1e-9]
    return min(excess) if excess else math.inf


def suggest_penalties(
    inst: VRPInstance,
    upper_bound: Optional[float] = None,
    require_all_vehicles: bool = False,
    use_capacity: bool = False,
    lambda_disc: float = 1.0,
    margin: float = 1.0,
) -> HamiltonianParams:
    """Limiares suficientes para que o mínimo NA GRADE seja C*.

    U (upper_bound) é qualquer custo >= C*. Por padrão, U = (N + M)·d_max,
    válido para qualquer solução viável (no máximo N + M arcos).
        λ_col     > N·d_max + U
        λ_gap     > U
        λ_vehicle > U                (se todos os veículos são exigidos)
        λ_cap     > U / δ²           (δ = menor excesso de carga atingível)
    São conservadores; valores menores costumam bastar (teste com
    `brute_force_grid`). λ_disc deve ser verificado com `continuous_minimum`.
    """
    dmax = float(inst.D.max())
    U = float(upper_bound) if upper_bound is not None else (inst.N + inst.M) * dmax
    lam_cap = 0.0
    if use_capacity:
        if inst.demands is None or inst.Q is None:
            raise ValueError("use_capacity exige demands e Q.")
        delta = min(_min_positive_excess(inst.demands, float(q)) for q in np.unique(inst.Q))
        lam_cap = 0.0 if math.isinf(delta) else float(U / delta**2 + margin)
    return HamiltonianParams(
        lambda_dist=1.0,
        lambda_col=inst.N * dmax + U + margin,
        lambda_gap=U + margin,
        lambda_vehicle=(U + margin) if require_all_vehicles else 0.0,
        lambda_cap=lam_cap,
        lambda_disc=lambda_disc,
    )


# ---------------------------------------------------------------------------
# Decodificação e custo clássico
# ---------------------------------------------------------------------------
def decode(theta: Sequence[float], phi: Sequence[float], inst: VRPInstance) -> Dict[str, object]:
    """Arredonda (θ, φ) para o nó mais próximo e monta as rotas.

    Retorna slots (v, r) 0-based por cidade e as rotas com nós 1..N,
    ordenadas pela posição. Colisões são mantidas na ordem da cidade.
    """
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    r = np.mod(np.rint(theta * inst.R / (2 * np.pi)).astype(int), inst.R)
    v = np.mod(np.rint(phi * inst.M / (2 * np.pi)).astype(int), inst.M)
    routes: List[List[int]] = []
    for veh in range(inst.M):
        cities = [i for i in range(inst.N) if v[i] == veh]
        cities.sort(key=lambda i: (r[i], i))
        routes.append([i + 1 for i in cities])
    return {"position": r, "vehicle": v, "routes": routes}


def routes_cost(routes: Sequence[Sequence[int]], D: np.ndarray) -> float:
    """Custo clássico: depósito -> rota -> depósito, rotas vazias custam 0."""
    total = 0.0
    for route in routes:
        if not route:
            continue
        nodes = [0, *route, 0]
        total += sum(D[nodes[k], nodes[k + 1]] for k in range(len(nodes) - 1))
    return float(total)


def is_feasible(decoded: Dict[str, object], inst: VRPInstance,
                require_all_vehicles: bool = False, use_capacity: bool = False) -> bool:
    """Viabilidade da solução decodificada (sem colisão, sem lacuna, capacidade)."""
    r, v = decoded["position"], decoded["vehicle"]
    slots = list(zip(v, r))
    if len(set(slots)) != len(slots):
        return False
    for veh in range(inst.M):
        pos = sorted(int(r[i]) for i in range(inst.N) if v[i] == veh)
        if pos != list(range(len(pos))):
            return False
        if require_all_vehicles and not pos:
            return False
        if use_capacity:
            load = sum(inst.demands[i] for i in range(inst.N) if v[i] == veh)
            if load > inst.Q[veh] + 1e-9:
                return False
    return True


# ---------------------------------------------------------------------------
# Verificações clássicas
# ---------------------------------------------------------------------------
def _grid_angles(inst: VRPInstance) -> Tuple[np.ndarray, np.ndarray]:
    """Todas as (M·R)^N configurações da grade; índice por cidade s = v·R + r."""
    S = inst.M * inst.R
    idx = np.array(list(itertools.product(range(S), repeat=inst.N)), dtype=int)
    r, v = idx % inst.R, idx // inst.R
    return 2 * np.pi * r / inst.R, 2 * np.pi * v / inst.M


def ideal_energy_table(inst: VRPInstance, params: Optional[HamiltonianParams] = None,
                       max_size: int = 5_000_000) -> np.ndarray:
    """Energias de todas as configurações da grade (limite ideal de Zak).

    Vetor de tamanho (M·R)^N, ordenado como um registrador de qudits de
    dimensão M·R (cidade 1 no dígito mais significativo, s = v·R + r).
    É a diagonal do Hamiltoniano de custo para simular o QAOA ideal.
    """
    size = (inst.M * inst.R) ** inst.N
    if size > max_size:
        raise MemoryError(f"(M·R)^N = {size} configurações excede max_size.")
    th, ph = _grid_angles(inst)
    return hamiltonian_energy(th, ph, inst, params)["total"]


def brute_force_grid(inst: VRPInstance, params: Optional[HamiltonianParams] = None,
                     top: int = 5) -> Dict[str, object]:
    """Mínimo do Hamiltoniano na grade e as `top` configurações de menor energia."""
    th, ph = _grid_angles(inst)
    E = hamiltonian_energy(th, ph, inst, params)["total"]
    order = np.argsort(E, kind="stable")
    emin = float(E[order[0]])
    best = []
    for k in order[:top]:
        dec = decode(th[k], ph[k], inst)
        best.append({"energy": float(E[k]), "routes": dec["routes"],
                     "cost": routes_cost(dec["routes"], inst.D)})
    return {"min_energy": emin,
            "degeneracy": int(np.sum(np.isclose(E, emin))),
            "best": best}


def continuous_minimum(inst: VRPInstance, params: Optional[HamiltonianParams] = None,
                       n_starts: int = 200, seed: int = 0) -> Dict[str, object]:
    """Mínimo de F no toro T^{2N} por múltiplos pontos iniciais (L-BFGS-B).

    Deve coincidir com o mínimo da grade; se ficar abaixo, aumente λ_disc.
    Heurístico: não garante o mínimo global.
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    N = inst.N

    def f(u: np.ndarray) -> float:
        return float(hamiltonian_energy(u[:N], u[N:], inst, params)["total"])

    best = None
    for _ in range(n_starts):
        res = minimize(f, rng.uniform(0, 2 * np.pi, 2 * N), method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    u = np.mod(best.x, 2 * np.pi)
    return {"min_energy": float(best.fun), "theta": u[:N], "phi": u[N:],
            "decoded": decode(u[:N], u[N:], inst)}


# ---------------------------------------------------------------------------
# Estimativa a partir de amostras
# ---------------------------------------------------------------------------
def estimate_from_samples(
    x_samples: np.ndarray,
    p_samples: np.ndarray,
    inst: VRPInstance,
    params: Optional[HamiltonianParams] = None,
    require_all_vehicles: bool = False,
    use_capacity: bool = False,
) -> Dict[str, object]:
    """Estimador amostral de ⟨H⟩ (análogo à eq. 10 do Madani) e melhor rota.

    x_samples, p_samples: (shots, N).

    ATENÇÃO: x e p de uma mesma linha precisam vir do MESMO disparo, por
    medição conjunta de (x mod a, p mod b). Parear resultados de disparos
    homodinos separados destrói a correlação posição–veículo e enviesa a
    estimativa quando o estado não está concentrado numa única configuração.
    """
    x_samples = np.atleast_2d(x_samples)
    p_samples = np.atleast_2d(p_samples)
    th, ph = quadratures_to_angles(x_samples, p_samples, inst)
    E = hamiltonian_energy(th, ph, inst, params)["total"]

    best_cost, best_routes, n_feasible = math.inf, None, 0
    for k in range(th.shape[0]):
        dec = decode(th[k], ph[k], inst)
        if is_feasible(dec, inst, require_all_vehicles, use_capacity):
            n_feasible += 1
            c = routes_cost(dec["routes"], inst.D)
            if c < best_cost:
                best_cost, best_routes = c, dec["routes"]
    shots = th.shape[0]
    return {"mean_energy": float(E.mean()),
            "std_error": float(E.std(ddof=1) / math.sqrt(shots)) if shots > 1 else math.nan,
            "feasible_fraction": n_feasible / shots,
            "best_cost": best_cost, "best_routes": best_routes}


# ---------------------------------------------------------------------------
# Valor esperado exato a partir de um ket de Fock (distribuição de Zak)
# ---------------------------------------------------------------------------
def hermite_functions(cutoff: int, x: np.ndarray, hbar: float = 2.0) -> np.ndarray:
    """⟨x|n⟩ para n < cutoff, convenção x̂ = sqrt(ħ/2)(â + â†). Retorna (cutoff, len(x))."""
    x = np.asarray(x, dtype=np.float64)
    u = x / math.sqrt(hbar)
    out = np.zeros((cutoff, x.size))
    out[0] = (math.pi * hbar) ** -0.25 * np.exp(-u**2 / 2.0)
    if cutoff > 1:
        out[1] = math.sqrt(2.0) * u * out[0]
    for n in range(1, cutoff - 1):
        out[n + 1] = math.sqrt(2.0 / (n + 1)) * u * out[n] - math.sqrt(n / (n + 1)) * out[n - 1]
    return out


def _zak_grid(inst: VRPInstance, cutoff: int, G: int, n_cells: Optional[int]) -> Tuple[int, int, np.ndarray]:
    G = inst.R * max(1, math.ceil(G / inst.R))           # θ-grid contém os nós de posição
    xmax = 1.2 * math.sqrt(2.0 * inst.hbar * (cutoff + 1))
    need = 2 * math.ceil(xmax / inst.a) + 2
    Nc = max(need, n_cells or 0)
    Nc = inst.M * math.ceil(Nc / inst.M)                 # p-grid contém os nós de veículo
    x = (np.arange(Nc)[:, None] - Nc // 2 + np.arange(G)[None, :] / G) * inst.a
    return Nc, G, x.reshape(-1)


def zak_distribution_from_ket(ket: np.ndarray, inst: VRPInstance, G: int = 8,
                              n_cells: Optional[int] = None) -> Dict[str, object]:
    """Distribuição conjunta discreta de (θ_i, φ_i) para um ket de Fock com N modos.

    Z(x0, p) = Σ_n e^{-i p n a/ħ} ψ(x0 + n a),  p ∈ [0, b)  (FFT nas células).
    Retorna P com eixos (k_1, g_1, ..., k_N, g_N): k indexa φ = 2πk/Nc e
    g indexa θ = 2πg/G.  Custo de memória ~ (Nc·G)^N: use só para N pequeno.
    """
    ket = np.asarray(ket)
    if ket.ndim != inst.N:
        raise ValueError(f"ket deve ter {inst.N} eixos (um por cidade).")
    cutoff = ket.shape[0]
    Nc, G, x = _zak_grid(inst, cutoff, G, n_cells)
    phi_n = hermite_functions(cutoff, x, inst.hbar)       # (cutoff, Nc·G)

    psi = ket.astype(np.complex128)
    for _ in range(inst.N):                               # contrai eixo 0 e anexa x no fim
        psi = np.tensordot(psi, phi_n, axes=([0], [0]))
    psi = psi.reshape(sum(((Nc, G) for _ in range(inst.N)), ()))
    Z = np.fft.fftn(psi, axes=tuple(range(0, 2 * inst.N, 2)))
    P = np.abs(Z) ** 2
    P /= P.sum()
    return {"P": P, "Nc": Nc, "G": G}


def _points_from_index(idx: Tuple[np.ndarray, ...], Nc: int, G: int, N: int) -> Tuple[np.ndarray, np.ndarray]:
    k = np.stack(idx[0::2], axis=-1)
    g = np.stack(idx[1::2], axis=-1)
    return 2 * np.pi * g / G, 2 * np.pi * k / Nc


def expectation_from_distribution(dist: Dict[str, object], inst: VRPInstance,
                                  params: Optional[HamiltonianParams] = None,
                                  chunk: int = 200_000, tol: float = 1e-12) -> Dict[str, float]:
    """⟨H⟩ = Σ P(θ, φ) F(θ, φ), por componente (soma em blocos)."""
    P, Nc, G = dist["P"], dist["Nc"], dist["G"]
    flat = np.flatnonzero(P > tol * P.max())
    acc: Dict[str, float] = {}
    for s in range(0, flat.size, chunk):
        ids = flat[s:s + chunk]
        th, ph = _points_from_index(np.unravel_index(ids, P.shape), Nc, G, inst.N)
        w = P.reshape(-1)[ids]
        for key, val in hamiltonian_energy(th, ph, inst, params).items():
            acc[key] = acc.get(key, 0.0) + float(np.dot(w, val))
    norm = float(P.reshape(-1)[flat].sum())
    return {key: val / norm for key, val in acc.items()}


def sample_from_distribution(dist: Dict[str, object], inst: VRPInstance, shots: int,
                             seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Amostras ideais de (x mod a, p mod b) com a distribuição de Zak."""
    rng = np.random.default_rng(seed)
    P, Nc, G = dist["P"], dist["Nc"], dist["G"]
    ids = rng.choice(P.size, size=shots, p=P.reshape(-1))
    th, ph = _points_from_index(np.unravel_index(ids, P.shape), Nc, G, inst.N)
    return th * inst.a / (2 * np.pi), ph * inst.b / (2 * np.pi)


# ---------------------------------------------------------------------------
# Integração com solver.py / run.py (Strawberry Fields, backend Fock)
# ---------------------------------------------------------------------------
_MAX_GRID_POINTS = int(os.environ.get("QUMODES_MAX_GRID_POINTS", 30_000_000))
_EVALUATOR_CACHE: Dict[tuple, "ZakEvaluator"] = {}


def auto_positions(N: int, M: int, demands: Optional[np.ndarray], Q: Optional[object],
                   require_all_vehicles: bool = False) -> int:
    """Posições por veículo (limites que valem para QUALQUER solução viável).

    - capacidade: nenhum veículo leva mais que floor(max Q / menor demanda) cidades;
    - todos os veículos obrigatórios: nenhum leva mais que N − M + 1 cidades.
    Menos posições significam slots mais espaçados em x (melhor resolução por cutoff).
    """
    R = N
    if demands is not None and Q is not None:
        d = np.asarray(demands, dtype=np.float64).reshape(-1)
        if d.size == N + 1:
            d = d[1:]
        pos = d[d > 0]
        if pos.size:
            R = min(R, int(math.floor(float(np.max(Q)) / float(pos.min()) + 1e-9)))
    if require_all_vehicles and M > 1:
        R = min(R, N - M + 1)
    R = max(1, R)
    if M * R < N:           # instância inviável pela contagem; mantém N para não mascarar
        return N
    return R


def requires_all_vehicles(params: Optional["HamiltonianParams"],
                          require_all_vehicles: Optional[bool] = None) -> bool:
    """Regra única do pacote: explícito se informado; senão, lambda_vehicle > 0."""
    if require_all_vehicles is not None:
        return bool(require_all_vehicles)
    return bool(params is not None and params.lambda_vehicle > 0)


def vehicle_penalty_bound(D: np.ndarray) -> float:
    """λ_vehicle suficiente para que veículos vazios nunca compensem.

    Toda solução com um veículo vazio pode ganhar esse veículo destacando uma
    cidade x de uma rota com >= 2 cidades; o custo sobe no máximo
    d(prev, next) + d(0, x) + d(x, 0) <= max d_ij + max d_0i + max d_i0.
    """
    D = np.asarray(D, dtype=np.float64)
    return float(D[1:, 1:].max() + D[0, 1:].max() + D[1:, 0].max())


def instance_from_legacy(N: int, M: int, D: np.ndarray, demands: Optional[np.ndarray] = None,
                         Q: Optional[object] = None, R: Optional[object] = "auto",
                         hbar: float = 2.0, require_all_vehicles: bool = False) -> VRPInstance:
    """Constrói VRPInstance a partir dos argumentos usados por solver.py (N=C, M=V)."""
    D = np.asarray(D, dtype=np.float64)
    if D.shape != (N + 1, N + 1):
        raise ValueError(f"D deve ser ({N + 1}x{N + 1}) com o depósito no índice 0; recebido {D.shape}.")
    if R == "auto":
        R = auto_positions(N, M, demands, Q, require_all_vehicles)
    return VRPInstance(D=D, M=M, R=R, demands=demands, Q=Q, hbar=hbar)


def _instance_key(inst: VRPInstance) -> tuple:
    return (inst.D.tobytes(), inst.M, inst.R, inst.hbar,
            None if inst.demands is None else inst.demands.tobytes(),
            None if inst.Q is None else np.asarray(inst.Q).tobytes())


class ZakEvaluator:
    """Avaliador com cache para um par (instância, cutoff).

    A grade (x mod a, p mod b) não depende do estado: as funções de Hermite e
    as energias de cada componente em todos os pontos da grade são calculadas
    uma única vez. Cada chamada do VQE custa apenas a transformada de Zak do
    ket e produtos escalares Σ P·E.
    """

    COMPONENTS = ("dist", "col", "gap", "vehicle", "capacity", "disc")

    def __init__(self, inst: VRPInstance, cutoff: int, G: int = 8, n_cells: Optional[int] = None):
        self.inst = inst
        self.cutoff = int(cutoff)
        self.Nc, self.G, x = _zak_grid(inst, self.cutoff, G, n_cells)
        self.points_per_mode = self.Nc * self.G
        total = self.points_per_mode ** inst.N
        if total > get_max_grid_points():
            raise MemoryError(
                f"Grade de Zak com {total:,} pontos (N={inst.N}, {self.points_per_mode} por modo). "
                "Reduza G/cutoff ou o número de cidades; o método exato é viável para N <= 4.")
        self.hermite = hermite_functions(self.cutoff, x, inst.hbar)
        self.shape = tuple(s for _ in range(inst.N) for s in (self.Nc, self.G))
        self._unit: Optional[Dict[str, np.ndarray]] = None   # componentes sem peso
        self._node_map = self._build_node_map()

    # --- energias na grade (sem os pesos λ) ---------------------------
    def _unit_components(self) -> Dict[str, np.ndarray]:
        if self._unit is None:
            unit = HamiltonianParams(lambda_dist=1, lambda_col=1, lambda_gap=1, lambda_vehicle=1,
                                     lambda_cap=1 if self._has_capacity() else 0, lambda_disc=1)
            acc = {c: np.empty(int(np.prod(self.shape)), dtype=np.float32) for c in self.COMPONENTS}
            n = acc["dist"].size
            chunk = 200_000
            for s in range(0, n, chunk):
                ids = np.arange(s, min(s + chunk, n))
                th, ph = _points_from_index(np.unravel_index(ids, self.shape), self.Nc, self.G, self.inst.N)
                vals = hamiltonian_energy(th, ph, self.inst, unit)
                for c in self.COMPONENTS:
                    acc[c][ids] = vals[c]
            self._unit = acc
        return self._unit

    def _has_capacity(self) -> bool:
        return self.inst.demands is not None and self.inst.Q is not None

    # --- distribuição de Zak ------------------------------------------
    def distribution(self, ket: np.ndarray) -> np.ndarray:
        ket = np.asarray(ket)
        if ket.shape != (self.cutoff,) * self.inst.N:
            raise ValueError(f"ket com forma {ket.shape}; esperado {(self.cutoff,) * self.inst.N}.")
        psi = ket.astype(np.complex128)
        for _ in range(self.inst.N):
            psi = np.tensordot(psi, self.hermite, axes=([0], [0]))
        psi = psi.reshape(self.shape)
        P = np.abs(np.fft.fftn(psi, axes=tuple(range(0, 2 * self.inst.N, 2)))) ** 2
        return P / P.sum()

    def energy(self, P: np.ndarray, params: HamiltonianParams) -> Dict[str, float]:
        unit = self._unit_components()
        w = {"dist": params.lambda_dist, "col": params.lambda_col, "gap": params.lambda_gap,
             "vehicle": params.lambda_vehicle, "capacity": params.lambda_cap, "disc": params.lambda_disc}
        if params.lambda_cap > 0 and not self._has_capacity():
            raise ValueError("λ_cap > 0 exige demands e Q.")
        Pf = P.reshape(-1)
        out = {c: float(w[c] * np.dot(Pf, unit[c].astype(np.float64, copy=False))) if w[c] != 0 else 0.0 for c in self.COMPONENTS}
        out["total"] = float(sum(out.values()))
        return out

    # --- agregação por slot (v, r) e amostragem ------------------------
    def _build_node_map(self) -> np.ndarray:
        """Matriz (Nc·G) x (M·R): ponto da grade -> slot mais próximo (s = v·R + r)."""
        k = np.repeat(np.arange(self.Nc), self.G)
        g = np.tile(np.arange(self.G), self.Nc)
        r = np.mod(np.rint(g * self.inst.R / self.G).astype(int), self.inst.R)
        v = np.mod(np.rint(k * self.inst.M / self.Nc).astype(int), self.inst.M)
        S = self.inst.M * self.inst.R
        onehot = np.zeros((self.points_per_mode, S))
        onehot[np.arange(self.points_per_mode), v * self.inst.R + r] = 1.0
        return onehot

    def slot_distribution(self, P: np.ndarray) -> np.ndarray:
        """Probabilidade de cada configuração discreta, forma (M·R,)*N."""
        T = P.reshape((self.points_per_mode,) * self.inst.N)
        for _ in range(self.inst.N):
            T = np.tensordot(T, self._node_map, axes=([0], [0]))
        return T


def get_max_grid_points() -> int:
    return _MAX_GRID_POINTS


def set_max_grid_points(n: int) -> None:
    """Limite de pontos da grade de Zak (memória ~ 80 bytes por ponto no pico)."""
    global _MAX_GRID_POINTS
    _MAX_GRID_POINTS = int(n)


def zak_grid_points(inst: VRPInstance, cutoff: int, G: int = 8) -> int:
    """Número de pontos da grade de Zak usada por evaluate_sf_state."""
    Nc, Gx, _ = _zak_grid(inst, int(cutoff), G, None)
    return (Nc * Gx) ** inst.N


def _get_evaluator(inst: VRPInstance, cutoff: int, G: int, n_cells: Optional[int]) -> ZakEvaluator:
    key = (_instance_key(inst), int(cutoff), int(G), n_cells)
    ev = _EVALUATOR_CACHE.get(key)
    if ev is None:
        ev = ZakEvaluator(inst, cutoff, G, n_cells)
        _EVALUATOR_CACHE[key] = ev
    return ev


def clear_cache() -> None:
    """Libera as grades de energia em cache."""
    _EVALUATOR_CACHE.clear()


def _resolve_instance(N, M, D, demands, Q, inst, positions, require_all: bool = False) -> VRPInstance:
    if isinstance(N, VRPInstance):          # chamada nova: evaluate_sf_state(state, inst, ...)
        return N
    if inst is not None:
        return inst
    if N is None or M is None or D is None:
        raise ValueError("Informe N, M e D (ou uma VRPInstance).")
    return instance_from_legacy(int(N), int(M), D, demands, Q, R=positions,
                                require_all_vehicles=require_all)


def _ket_of(state) -> np.ndarray:
    if hasattr(state, "is_pure") and not state.is_pure:
        raise NotImplementedError("Somente estados puros: o backend Fock sem ruído gera estados puros.")
    ket = state.ket()
    if ket is None:
        raise NotImplementedError("state.ket() retornou None (estado misto).")
    return np.asarray(ket)


def evaluate_sf_state(
    state,
    N: Optional[object] = None,
    M: Optional[int] = None,
    D: Optional[np.ndarray] = None,
    demands: Optional[np.ndarray] = None,
    Q: Optional[object] = None,
    cutoff: int = 8,
    params: Optional[HamiltonianParams] = None,
    *,
    inst: Optional[VRPInstance] = None,
    positions: object = "auto",
    G: int = 8,
    n_cells: Optional[int] = None,
) -> Dict[str, float]:
    """⟨Ĥ⟩ exato (na grade de Zak) de um estado do Strawberry Fields.

    Assinatura compatível com solver.py:
        evaluate_sf_state(state=..., M=V, N=C, D=D, demands=..., Q=..., cutoff=..., params=...)

    Retorna {"total", "dist", "col", "gap", "capacity", "vehicle", "disc"}.
    Como P >= 0 e soma 1, o valor retornado nunca fica abaixo do menor valor
    de F nos pontos da grade de Zak. Se `check_penalties(..., cutoff=cutoff)`
    retornar ok=True, esse limite é o custo ótimo (GAP >= 0 no run.py).
    """
    params = params or HamiltonianParams()
    instance = _resolve_instance(N, M, D, demands, Q, inst, positions, requires_all_vehicles(params))
    ket = _ket_of(state)
    ev = _get_evaluator(instance, ket.shape[0], G, n_cells)
    return ev.energy(ev.distribution(ket), params)


def extract_routes(
    state,
    N: Optional[object] = None,
    M: Optional[int] = None,
    D: Optional[np.ndarray] = None,
    demands: Optional[np.ndarray] = None,
    Q: Optional[object] = None,
    cutoff: int = 8,
    params: Optional[HamiltonianParams] = None,
    *,
    inst: Optional[VRPInstance] = None,
    positions: object = "auto",
    mode: str = "best_sample",
    shots: int = 1000,
    seed: Optional[int] = 42,
    require_all_vehicles: Optional[bool] = None,
    G: int = 8,
    n_cells: Optional[int] = None,
    return_details: bool = False,
):
    """Decodifica rotas no formato esperado por solver.py e run.py.

    Formato: V == 1 -> [0, c1, ..., 0];  V > 1 -> {1: [0, ..., 0], 2: [0, ..., 0], ...}

    mode = "best_sample"   : sorteia `shots` configurações da distribuição de slots
                              e devolve a viável de menor custo (como no Madani).
    mode = "most_probable" : configuração discreta de maior probabilidade.
    Se nenhuma amostra for viável, usa a mais provável.
    require_all_vehicles=None segue a regra do pacote (lambda_vehicle > 0).
    """
    params = params or HamiltonianParams()
    require_all_vehicles = requires_all_vehicles(params, require_all_vehicles)
    instance = _resolve_instance(N, M, D, demands, Q, inst, positions,
                                 requires_all_vehicles(params))
    ket = _ket_of(state)
    ev = _get_evaluator(instance, ket.shape[0], G, n_cells)
    S = instance.M * instance.R
    probs = ev.slot_distribution(ev.distribution(ket)).reshape(-1)
    probs = np.clip(probs, 0, None)
    probs /= probs.sum()
    use_cap = params.lambda_cap > 0 and instance.demands is not None and instance.Q is not None

    def dec(flat_idx: int) -> Dict[str, object]:
        s = np.array(np.unravel_index(flat_idx, (S,) * instance.N))
        r, v = s % instance.R, s // instance.R
        return decode(2 * np.pi * r / instance.R, 2 * np.pi * v / instance.M, instance)

    chosen, cost, feasible = None, math.inf, False
    if mode == "best_sample":
        rng = np.random.default_rng(seed)
        for idx in np.unique(rng.choice(probs.size, size=shots, p=probs)):
            d = dec(int(idx))
            if is_feasible(d, instance, require_all_vehicles, use_cap):
                c = routes_cost(d["routes"], instance.D)
                if c < cost:
                    chosen, cost, feasible = d, c, True
    elif mode != "most_probable":
        raise ValueError("mode deve ser 'best_sample' ou 'most_probable'.")
    if chosen is None:
        chosen = dec(int(np.argmax(probs)))
        feasible = is_feasible(chosen, instance, require_all_vehicles, use_cap)
        cost = routes_cost(chosen["routes"], instance.D)

    routes = chosen["routes"]
    formatted = ([0, *routes[0], 0] if instance.M == 1
                 else {v + 1: [0, *routes[v], 0] for v in range(instance.M)})
    if return_details:
        return formatted, {"cost": cost, "feasible": feasible,
                           "prob_most_probable": float(probs.max()), "positions": instance.R}
    return formatted


def check_penalties(
    N: object,
    M: Optional[int] = None,
    D: Optional[np.ndarray] = None,
    demands: Optional[np.ndarray] = None,
    Q: Optional[object] = None,
    params: Optional[HamiltonianParams] = None,
    *,
    positions: object = "auto",
    require_all_vehicles: Optional[bool] = None,
    continuous_starts: int = 0,
    cutoff: Optional[int] = None,
    G: int = 8,
    max_grid: int = 2_000_000,
) -> Dict[str, object]:
    """Confere se min(Ĥ) na grade é igual ao ótimo viável da própria codificação.

    Compare "best_feasible_cost" com o custo do BruteForce: se coincidirem e
    "ok" for True, a energia mínima do Hamiltoniano é o custo ótimo.
    Com continuous_starts > 0 também procura mínimos fracionários no toro.
    Com cutoff informado, confere o menor valor de F na grade de Zak usada por
    evaluate_sf_state: é o limite inferior exato da energia reportada ao VQE.
    """
    params = params or HamiltonianParams()
    require_all_vehicles = requires_all_vehicles(params, require_all_vehicles)
    instance = _resolve_instance(N, M, D, demands, Q, None, positions,
                                 requires_all_vehicles(params))
    size = (instance.M * instance.R) ** instance.N
    if size > max_grid:
        raise MemoryError(f"(M·R)^N = {size:,} configurações; aumente max_grid ou reduza a instância.")
    th, ph = _grid_angles(instance)
    E = hamiltonian_energy(th, ph, instance, params)["total"]
    use_cap = params.lambda_cap > 0 and instance.demands is not None and instance.Q is not None
    best_feasible = math.inf
    for k in np.argsort(E, kind="stable"):
        if E[k] > best_feasible + 1e-9:
            break
        d = decode(th[k], ph[k], instance)
        if is_feasible(d, instance, require_all_vehicles, use_cap):
            best_feasible = min(best_feasible, routes_cost(d["routes"], instance.D))
    out = {"positions": instance.R, "require_all_vehicles": require_all_vehicles,
           "grid_min_energy": float(E.min()),
           "best_feasible_cost": best_feasible,
           "ok": bool(np.isclose(E.min(), best_feasible))}
    if continuous_starts > 0:
        cm = continuous_minimum(instance, params, n_starts=continuous_starts)
        out["continuous_min_energy"] = cm["min_energy"]
        out["ok"] = out["ok"] and cm["min_energy"] >= best_feasible - 1e-6
    if cutoff is not None:
        ev = _get_evaluator(instance, int(cutoff), G, None)
        unit = ev._unit_components()
        w = {"dist": params.lambda_dist, "col": params.lambda_col, "gap": params.lambda_gap,
             "vehicle": params.lambda_vehicle, "capacity": params.lambda_cap if use_cap else 0.0,
             "disc": params.lambda_disc}
        zmin = float(min(sum(w[c] * unit[c] for c in w if w[c] != 0)))
        out["zak_grid_min_energy"] = zmin
        out["ok"] = out["ok"] and zmin >= best_feasible - 1e-6
    return out
