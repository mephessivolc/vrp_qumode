"""
hamiltonian.py — Hamiltoniano modular do TSP/VRP em qumodes, formulação de
OPERADORES na base de Fock truncada.

Mudança em relação à versão anterior
------------------------------------
A versão anterior avaliava H como uma FUNÇÃO CLÁSSICA no toro, sobre a
distribuição de Zak: <H> = sum_(theta,phi) P(theta,phi) F(theta,phi). Isso
trata todos os observáveis como comutantes e escolhe implicitamente o
ordenamento diagonal — incorreto nos termos com produtos de operadores no
MESMO modo (gap, vehicle), onde a^2 como número não é <A^2> como operador.

Aqui H é um operador hermitiano no espaço de Fock truncado, e <psi|H|psi> é
uma expectativa quântica de verdade. Só assim "o mínimo do Hamiltoniano é a
solução do VRP" é uma afirmação sobre autovalores.

Operadores de deslocamento exatos
---------------------------------
U = exp(i (2pi/a) x) e V = exp(i (2pi/b) p) são operadores de DESLOCAMENTO:
U = D(i k /sqrt(2 hbar) ... ) — ver `displacement_operator`. Pela relação de
Weyl, D(al)D(be) = exp(i Im(al be*)) D(al+be), e com a*b = 2 pi hbar o fator
de fase vale exp(-i pi) nos dois ordenamentos, logo [U, V] = 0 EXATAMENTE.

NÃO use matrix_exp da quadratura truncada: exp de uma matriz já cortada não é
o corte de exp(i k x). Medido (hbar=1, M=R=2): ||UV-VU||_2 fica em ~2.0 em
qualquer cutoff com matrix_exp, contra ~0.99 no espaço todo e 1.5e-15 no
quarto inferior do espectro com D(alpha) exato em cutoff 50. A violação vive
nos níveis altos de Fock; o encoding é fiel num subespaço de baixa energia
cujo tamanho cresce com o cutoff.

Ordenamentos
------------
Critério adotado, uniforme em todo o arquivo:
  PENALIDADES usam sanduíche   B n B      (hermitiano E positivo semidefinido)
  CUSTOS      usam anticomutador {A,B}/2  (hermitiano; não precisa ser PSD)

Para o termo de lacuna, a alternativa (1/2){n, B^2} tem autovalor mínimo
medido -1.22e-2: existiria estado em que ABRIR uma lacuna reduz a energia.

Precisão
--------
A construção dos operadores é feita UMA vez em numpy float64 e congelada. Em
TensorFlow ela entra como constante: o gradiente flui só pelo estado, nunca
através de eigh. Medido no backend tf do Strawberry Fields (complex64), com
circuito de 44 parâmetros: ||ket_fock - ket_tf||_inf = 2.1e-7, suficiente
para energias ~1e2 com descidas ~1e-2 por iteração.

Verificação exata (sem VQE)
---------------------------
  node_energy_table    <H> em estados-produto centrados em cada configuração
                       de nós — análogo direto do antigo brute_force_grid
  dense_hamiltonian    matriz densa para diagonalização exata; o fundamental
                       deve decodificar para a rota ótima clássica
Ambas em numpy float64. Não use TF aqui: verificação é padrão-ouro.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "VRPInstance", "HamiltonianParams", "ModularOperators",
    "auto_positions", "requires_all_vehicles", "instance_from_legacy",
    "decode", "routes_cost", "is_feasible",
    "suggest_penalties", "vehicle_penalty_bound", "suggest_lambda_exc",
    "displacement_operator", "fejer_operator", "build_modular_operators",
    "energy_components", "assignment_distribution",
    "node_energy_table", "dense_hamiltonian", "ground_state",
]


# ---------------------------------------------------------------------------
# 1. Definição do problema
# ---------------------------------------------------------------------------
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
    lambda_exc: float = 0.0       # confinamento: λ_exc Σ_i ⟨n̂_i⟩ (ver nota abaixo)

    # ------------------------------------------------------------------
    # Sobre λ_exc (termo de confinamento)
    # ------------------------------------------------------------------
    # Todos os demais termos são funções apenas das variáveis MODULARES
    # θ = 2πx/a mod 2π e φ = 2πp/b mod 2π: o Hamiltoniano vive no toro e é
    # invariante sob x -> x + a, p -> p + b. Ele é portanto CEGO ao envelope,
    # e nada cobra pela excitação do modo.
    #
    # A consequência é que ⟨Ĥ_enc⟩ não tem minimizador sobre os estados
    # físicos: o termo λ_disc decresce monotonamente rumo a 1 por modo ao
    # longo do squeezing, com ⟨n⟩ -> ∞, e só o corte de Fock limita essa
    # direção — limitando-a por destruir a norma do estado.
    #
    # É o mesmo fenômeno dos códigos GKP, cujo Hamiltoniano estabilizador
    # -E_q cos(·x̂) - E_p cos(·p̂) tem fundamentais não normalizáveis. A
    # correção padrão na literatura é somar um potencial de confinamento
    # fraco (ω0/2)(x̂² + p̂²) = ħω0(n̂ + ½), obtendo o Hamiltoniano GKP de
    # energia finita. Ver Gottesman, Kitaev & Preskill, Phys. Rev. A 64,
    # 012310 (2001), arXiv:quant-ph/0008040; arXiv:2308.02913 (revisão, com
    # H_GKP e H_GKP,Δ); arXiv:2411.04993 (não normalizabilidade);
    # arXiv:2009.07941 e arXiv:2305.05262 (operador envelope e^{-Δ²n̂}).
    #
    # λ_exc = 0 (padrão) reproduz exatamente o comportamento anterior.
    # Faixa útil medida: 0.001 a 0.01. Acima de ~0.05 o custo em fótons
    # anula a vantagem de estados tipo GKP sobre gaussianos. Use
    # `suggest_lambda_exc` para o teto imposto pelo espaçamento de custos.

    def weights(self) -> Tuple[float, ...]:
        return (self.lambda_dist, self.lambda_col, self.lambda_gap,
                self.lambda_vehicle, self.lambda_cap, self.lambda_disc,
                self.lambda_exc)

# ---------------------------------------------------------------------------
# Codificação
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 2. Decodificação clássica (independente da representação)
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
# 3. Penalidades
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

def vehicle_penalty_bound(D: np.ndarray) -> float:
    """λ_vehicle suficiente para que veículos vazios nunca compensem.

    Toda solução com um veículo vazio pode ganhar esse veículo destacando uma
    cidade x de uma rota com >= 2 cidades; o custo sobe no máximo
    d(prev, next) + d(0, x) + d(x, 0) <= max d_ij + max d_0i + max d_i0.
    """
    D = np.asarray(D, dtype=np.float64)
    return float(D[1:, 1:].max() + D[0, 1:].max() + D[1:, 0].max())

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


def suggest_lambda_exc(inst: VRPInstance, ops: "ModularOperators",
                       params: Optional[HamiltonianParams] = None,
                       photon_budget: float = 10.0) -> Dict[str, float]:
    """Teto para λ_exc imposto pelo espaçamento entre níveis de custo.

    O confinamento não pode tornar lucrativo trocar qualidade de rota por
    fótons: se δ é a distância entre o nível ótimo e o seguinte na tabela de
    estados de nó, e Δ<n> é a diferença de excitação entre o estado que
    resolve a codificação e um mais barato, é preciso λ_exc·Δ<n> < δ.

    photon_budget é o Δ<n> assumido; 10 é a ordem necessária para um estado
    tipo GKP resolver R ~ 4 nós de posição.
    """
    tab = node_energy_table(inst, ops, params)
    levels = np.unique(np.round(tab["energies"].reshape(-1), 9))
    gap = float(levels[1] - levels[0]) if levels.size > 1 else float("inf")
    budget = max(float(photon_budget), 1e-9)
    ceiling = gap / budget
    return {"cost_gap": gap, "photon_budget": budget,
            "lambda_exc_max": ceiling,
            "lambda_exc_suggested": float(min(0.01, max(0.001, 0.1 * ceiling))),
            "num_energy_levels": int(levels.size)}

# ---------------------------------------------------------------------------
# 4. Camada de operadores (numpy float64, construída UMA vez)
# ---------------------------------------------------------------------------
def _hermitian(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.conj().T)


def _psd_sqrt(A: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Raiz PSD por decomposição espectral, com autovalores negativos zerados.

    Feita em float64 e CONGELADA: nunca diferencie através desta função.
    """
    w, V = np.linalg.eigh(_hermitian(A))
    w = np.clip(w.real, tol, None)
    return (V * np.sqrt(w)) @ V.conj().T


def displacement_operator(cutoff: int, alpha: complex) -> np.ndarray:
    """D(alpha) = exp(alpha a^dag - alpha* a), truncado, elementos EXATOS.

    <m|D(alpha)|n> = sqrt(n!/m!) alpha^(m-n) e^{-|alpha|^2/2} L_n^(m-n)(|alpha|^2)
    para m >= n, e o conjugado-transposto com -alpha* no caso n > m, onde
    L_n^(k) é o polinômio de Laguerre associado.

    Esta é a truncagem do operador exato, e não a exponencial de uma matriz já
    truncada. A diferença não é cosmética: é ela que preserva [U, V] = 0 no
    subespaço de baixa energia.
    """
    from scipy.special import eval_genlaguerre

    n = np.arange(cutoff)
    r2 = float(abs(alpha) ** 2)
    # log-fatoriais evitam overflow em cutoff alto
    logfact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, cutoff)))])
    D = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for nn in range(cutoff):
        for mm in range(cutoff):
            if mm >= nn:
                k = mm - nn
                pref = math.exp(0.5 * (logfact[nn] - logfact[mm]))
                val = pref * (alpha ** k) * eval_genlaguerre(nn, k, r2)
            else:
                k = nn - mm
                pref = math.exp(0.5 * (logfact[mm] - logfact[nn]))
                val = pref * ((-np.conj(alpha)) ** k) * eval_genlaguerre(mm, k, r2)
            D[mm, nn] = val
    return D * math.exp(-r2 / 2.0)


def fejer_operator(W: np.ndarray, K: int, node_index: int) -> np.ndarray:
    """Núcleo de Fejér de ordem K centrado no nó 2*pi*node_index/K.

    F_j = (1/K^2) [ K I + sum_{m=1}^{K-1} (K-m) ( e^{-i m nodo} W^m + h.c. ) ]

    Fejér e não Dirichlet porque é não negativo: os F_j são PSD e somam a
    identidade, o que os torna um POVM legítimo sobre os nós.
    """
    d = W.shape[0]
    I = np.eye(d, dtype=np.complex128)
    if K == 1:
        return I
    node = 2.0 * math.pi * node_index / K
    out = float(K) * I
    Wm = I.copy()
    for m in range(1, K):
        Wm = Wm @ W
        term = np.exp(-1j * m * node) * Wm
        out = out + (K - m) * (term + term.conj().T)
    return _hermitian(out / float(K * K))


@dataclass
class ModularOperators:
    """Operadores de um modo, fixos dados (cutoff, M, R, hbar)."""
    cutoff: int
    M: int
    R: int
    hbar: float
    a: float
    b: float
    U: np.ndarray                  # exp(i 2pi x / a)
    V: np.ndarray                  # exp(i 2pi p / b)
    F: np.ndarray                  # (R, d, d) nós de posição
    G: np.ndarray                  # (M, d, d) nós de veículo
    A: np.ndarray                  # (M, R, d, d) efeitos do POVM
    P_vehicle: np.ndarray          # (M, d, d) marginal por veículo
    H_disc: np.ndarray             # 2I - cos(R theta) - cos(M phi)
    n_op: np.ndarray               # operador número (diagonal)

    def weyl_commutator(self, block: Optional[int] = None) -> float:
        """||UV - VU||_2, opcionalmente só no bloco inferior de dimensão `block`.

        No espaço todo o valor fica alto em qualquer cutoff, porque a truncagem
        quebra a relação canônica nos níveis mais altos. O que converge — e o
        que importa — é o bloco de baixa energia.
        """
        C = self.U @ self.V - self.V @ self.U
        if block:
            C = C[:block, :block]
        return float(np.linalg.norm(C, 2))

    def completeness_error(self) -> float:
        """||sum_{v,r} A_{v,r} - I||_2. Deve ser ~1e-15."""
        return float(np.linalg.norm(self.A.sum(axis=(0, 1)) - np.eye(self.cutoff), 2))

    def min_eigenvalue(self) -> float:
        """Menor autovalor entre todos os A_{v,r}. Deve ser >= ~ -1e-15 (PSD)."""
        return float(min(np.linalg.eigvalsh(self.A[v, r]).min()
                         for v in range(self.M) for r in range(self.R)))

    def diagnostics(self) -> Dict[str, float]:
        d = self.cutoff
        return {"weyl_full": self.weyl_commutator(),
                "weyl_half": self.weyl_commutator(max(2, d // 2)),
                "weyl_quarter": self.weyl_commutator(max(2, d // 4)),
                "completeness": self.completeness_error(),
                "povm_min_eigenvalue": self.min_eigenvalue()}


_OPS_CACHE: Dict[tuple, ModularOperators] = {}


def build_modular_operators(cutoff: int, M: int, R: int, hbar: float = 2.0,
                            use_cache: bool = True) -> ModularOperators:
    """Monta os operadores modulares. Resultado em float64, cacheado."""
    key = (int(cutoff), int(M), int(R), float(hbar))
    if use_cache and key in _OPS_CACHE:
        return _OPS_CACHE[key]

    a = math.sqrt(2.0 * math.pi * hbar)
    b = 2.0 * math.pi * hbar / a

    # x = sqrt(hbar/2)(adag + a)  =>  exp(i k x) = D(i k sqrt(hbar/2))
    # p = i sqrt(hbar/2)(adag - a) =>  exp(i k' p) = D(-k' sqrt(hbar/2))
    s = math.sqrt(hbar / 2.0)
    U = displacement_operator(cutoff, 1j * (2.0 * math.pi / a) * s)
    V = displacement_operator(cutoff, -(2.0 * math.pi / b) * s)

    F = np.stack([fejer_operator(U, R, r) for r in range(R)])
    G = np.stack([fejer_operator(V, M, v) for v in range(M)])
    sqrtF = [_psd_sqrt(F[r]) for r in range(R)]

    # A_{v,r} = sqrt(F_r) G_v sqrt(F_r): hermitiano e PSD por construção.
    # O produto simples F_r G_v não seria nem um nem outro, porque no cutoff
    # finito [F_r, G_v] != 0, e a leitura probabilística se perderia.
    A = np.stack([np.stack([_hermitian(sqrtF[r] @ G[v] @ sqrtF[r])
                            for r in range(R)]) for v in range(M)])

    UR = np.linalg.matrix_power(U, R)
    VM = np.linalg.matrix_power(V, M)
    I = np.eye(cutoff, dtype=np.complex128)
    H_disc = _hermitian(2.0 * I - 0.5 * (UR + UR.conj().T) - 0.5 * (VM + VM.conj().T))

    ops = ModularOperators(cutoff=cutoff, M=M, R=R, hbar=hbar, a=a, b=b,
                           U=U, V=V, F=F, G=G, A=A, P_vehicle=A.sum(axis=1),
                           H_disc=H_disc,
                           n_op=np.diag(np.arange(cutoff)).astype(np.complex128))
    if use_cache:
        _OPS_CACHE[key] = ops
    return ops


def operators_for(inst: VRPInstance, cutoff: int) -> ModularOperators:
    return build_modular_operators(cutoff, inst.M, inst.R, inst.hbar)


# ---------------------------------------------------------------------------
# 5. Avaliação da energia — um algoritmo, dois backends
# ---------------------------------------------------------------------------
class _NumpyBackend:
    """Backend de referência: float64/complex128. Padrão-ouro."""
    name = "numpy"

    @staticmethod
    def const(A, like=None):
        return np.asarray(A, dtype=np.complex128)

    @staticmethod
    def apply_local(psi, op, mode):
        N = psi.ndim
        out = np.tensordot(psi, op, axes=([mode], [1]))
        perm = list(range(mode)) + [N - 1] + list(range(mode, N - 1))
        return np.transpose(out, perm)

    @staticmethod
    def inner_real(psi, phi):
        return float(np.real(np.sum(np.conj(psi) * phi)))

    @staticmethod
    def abs2(psi):
        return np.abs(psi) ** 2

    @staticmethod
    def sum_all(x):
        return float(np.sum(x))

    @staticmethod
    def zeros_like_scalar(psi):
        return 0.0


class _TFBackend:
    """Backend TensorFlow: o gradiente flui pelo estado, nunca pelos operadores.

    Os operadores entram como constantes complex64. Nunca use tf.cast de um
    complexo para float: isso descarta a parte imaginária em silêncio. Use
    tf.math.real no ÚLTIMO passo, e tf.abs para probabilidades.
    """
    name = "tf"

    def __init__(self):
        import tensorflow as tf
        self.tf = tf

    def const(self, A, like=None):
        """Converte um operador em constante TF.

        SEM cache por id(): ops.A[v, r] devolve uma VIEW temporária do array
        4-D, e depois que esta função retorna o id pode ser reaproveitado pela
        próxima view — o cache então entregava o operador errado. Os
        operadores são matrizes d x d; converter a cada chamada é desprezível
        perto das contrações sobre o tensor d^N.
        """
        tf = self.tf
        dtype = like.dtype if like is not None else tf.complex64
        return tf.constant(np.asarray(A, dtype=np.complex128), dtype=dtype)

    def apply_local(self, psi, op, mode):
        tf = self.tf
        N = len(psi.shape)
        out = tf.tensordot(psi, op, axes=[[mode], [1]])
        perm = list(range(mode)) + [N - 1] + list(range(mode, N - 1))
        return tf.transpose(out, perm)

    def inner_real(self, psi, phi):
        tf = self.tf
        return tf.math.real(tf.reduce_sum(tf.math.conj(psi) * phi))

    def abs2(self, psi):
        return self.tf.square(self.tf.abs(psi))

    def sum_all(self, x):
        return self.tf.reduce_sum(x)

    def zeros_like_scalar(self, psi):
        return self.tf.cast(0.0, self.tf.float32 if psi.dtype == self.tf.complex64
                            else self.tf.float64)


def _get_backend(psi):
    if isinstance(psi, np.ndarray):
        return _NumpyBackend()
    return _TFBackend()


def _occupancy(bk, psi, A_const, v, r, N):
    """n_{v,r} psi = sum_i A^{(i)}_{v,r} psi."""
    out = None
    for i in range(N):
        t = bk.apply_local(psi, A_const[v][r], i)
        out = t if out is None else out + t
    return out


def energy_components(psi, inst: VRPInstance, ops: ModularOperators,
                      params: Optional[HamiltonianParams] = None,
                      require_all_vehicles: Optional[bool] = None,
                      normalize: bool = True) -> Dict[str, object]:
    """<psi|H|psi> decomposto por termo.

    psi: tensor de posto N (um eixo por cidade), numpy OU tensorflow. O backend
    é detectado pelo tipo; o algoritmo é o mesmo.

    Chaves devolvidas:
      dist, col, gap, vehicle, capacity, disc   componentes ponderadas
      encoding   soma das acima — comparável com C* e com as linhas de base
      exc        lambda_exc * sum_i <n_i>
      mean_photons  sum_i <n_i>, sem peso
      total      encoding + exc — é o que o VQE minimiza
      state_norm2  norma bruta de psi ANTES da normalização (diagnóstico de
                   truncagem: deve ficar >= 0.99)
    """
    params = params or HamiltonianParams()
    bk = _get_backend(psi)
    N, M, R = inst.N, inst.M, inst.R
    req_all = requires_all_vehicles(params, require_all_vehicles)

    raw = bk.inner_real(psi, psi)
    if normalize:
        if isinstance(psi, np.ndarray):
            psi = psi / math.sqrt(max(float(raw), 1e-300))
        else:
            tf = bk.tf
            psi = psi / tf.cast(tf.sqrt(tf.maximum(raw, 1e-30)), psi.dtype)

    A = [[bk.const(ops.A[v, r], like=psi) for r in range(R)] for v in range(M)]
    D = np.asarray(inst.D, dtype=np.float64)

    # --- distância ---------------------------------------------------------
    h_dist = bk.zeros_like_scalar(psi)
    for v in range(M):                                   # depósito -> primeira
        for i in range(N):
            h_dist = h_dist + float(D[0, i + 1]) * bk.inner_real(psi, bk.apply_local(psi, A[v][0], i))
    for v in range(M):                                   # arcos internos
        for r in range(R - 1):
            for i in range(N):
                Ai = bk.apply_local(psi, A[v][r], i)
                for j in range(N):
                    if i == j:
                        continue
                    h_dist = h_dist + float(D[i + 1, j + 1]) * bk.inner_real(
                        psi, bk.apply_local(Ai, A[v][r + 1], j))
    for v in range(M):                                   # última -> depósito
        for r in range(R - 1):
            # custo, não penalidade: anticomutador {A, I - n}/2, hermitiano.
            # <{X,Y}/2> = Re<X psi|Y psi> para X, Y hermitianos.
            Bpsi = psi - _occupancy(bk, psi, A, v, r + 1, N)
            for i in range(N):
                h_dist = h_dist + float(D[i + 1, 0]) * bk.inner_real(
                    bk.apply_local(psi, A[v][r], i), Bpsi)
        for i in range(N):
            h_dist = h_dist + float(D[i + 1, 0]) * bk.inner_real(
                psi, bk.apply_local(psi, A[v][R - 1], i))

    # --- colisão: modos distintos, comutam, sem ambiguidade ----------------
    h_col = bk.zeros_like_scalar(psi)
    for v in range(M):
        for r in range(R):
            for i in range(N):
                Ai = bk.apply_local(psi, A[v][r], i)
                for j in range(i + 1, N):
                    h_col = h_col + bk.inner_real(psi, bk.apply_local(Ai, A[v][r], j))

    # --- lacuna: PENALIDADE -> sanduíche B n B ------------------------------
    h_gap = bk.zeros_like_scalar(psi)
    for v in range(M):
        for r in range(1, R):
            Bpsi = float(r) * psi
            for q in range(r):
                Bpsi = Bpsi - _occupancy(bk, psi, A, v, q, N)
            # <B n B> = <B psi| n |B psi> >= 0 porque n é PSD
            for i in range(N):
                h_gap = h_gap + bk.inner_real(Bpsi, bk.apply_local(Bpsi, A[v][r], i))

    # --- veículos obrigatórios: ||(n - I) psi||^2 ---------------------------
    h_veh = bk.zeros_like_scalar(psi)
    if req_all:
        for v in range(M):
            y = _occupancy(bk, psi, A, v, 0, N) - psi
            h_veh = h_veh + bk.inner_real(y, y)

    # --- capacidade: cálculo funcional espectral exato ----------------------
    h_cap = bk.zeros_like_scalar(psi)
    if params.lambda_cap > 0:
        if inst.demands is None or inst.Q is None:
            raise ValueError("λ_cap > 0 exige demands e Q na instância.")
        for v in range(M):
            evals, evecs = np.linalg.eigh(_hermitian(ops.P_vehicle[v]))
            Vh = bk.const(evecs.conj().T, like=psi)
            coeff = psi
            for i in range(N):
                coeff = bk.apply_local(coeff, Vh, i)
            # L_v = sum_i q_i P_v^(i): termos em modos distintos, comutam
            load = np.zeros((evals.size,) * N, dtype=np.float64)
            for i in range(N):
                shape = [1] * N
                shape[i] = evals.size
                load = load + float(inst.demands[i]) * evals.real.reshape(shape)
            excess2 = np.maximum(load - float(inst.Q[v]), 0.0) ** 2
            w = (excess2 if isinstance(psi, np.ndarray)
                 else _tf_real_const(bk, excess2, psi))
            h_cap = h_cap + bk.sum_all(bk.abs2(coeff) * w)

    # --- estabilizador e excitação -----------------------------------------
    Hd = bk.const(ops.H_disc, like=psi)
    h_disc = bk.zeros_like_scalar(psi)
    for i in range(N):
        h_disc = h_disc + bk.inner_real(psi, bk.apply_local(psi, Hd, i))

    nop = bk.const(ops.n_op, like=psi)
    photons = bk.zeros_like_scalar(psi)
    for i in range(N):
        photons = photons + bk.inner_real(psi, bk.apply_local(psi, nop, i))

    out = {"dist": params.lambda_dist * h_dist,
           "col": params.lambda_col * h_col,
           "gap": params.lambda_gap * h_gap,
           "vehicle": params.lambda_vehicle * h_veh,
           "capacity": params.lambda_cap * h_cap,
           "disc": params.lambda_disc * h_disc}
    encoding = out["dist"] + out["col"] + out["gap"] + out["vehicle"] + out["capacity"] + out["disc"]
    out["encoding"] = encoding
    out["mean_photons"] = photons
    out["exc"] = params.lambda_exc * photons
    out["total"] = encoding + out["exc"]
    out["state_norm2"] = raw
    return out


def _tf_real_const(bk, arr, like):
    tf = bk.tf
    dt = tf.float32 if like.dtype == tf.complex64 else tf.float64
    return tf.constant(np.asarray(arr, dtype=np.float64), dtype=dt)


def assignment_distribution(psi, inst: VRPInstance, ops: ModularOperators) -> np.ndarray:
    """p(configuração) = <psi| prod_i A_{v_i, r_i}^{(i)} |psi>, forma (M*R,)*N.

    Como sum_{v,r} A_{v,r} = I e cada A é PSD, estas probabilidades somam 1 e
    são não negativas: é a leitura do POVM. Substitui a amostragem da
    distribuição de Zak na decodificação.
    """
    psi = np.asarray(psi)
    psi = psi / np.linalg.norm(psi)
    N, M, R = inst.N, inst.M, inst.R
    bk = _NumpyBackend()
    out = np.zeros((M * R,) * N, dtype=np.float64)
    for slots in itertools.product(range(M * R), repeat=N):
        phi = psi
        for i, s in enumerate(slots):
            v, r = divmod(s, R)
            phi = bk.apply_local(phi, ops.A[v, r], i)
        out[slots] = max(0.0, bk.inner_real(psi, phi))
    tot = out.sum()
    return out / tot if tot > 0 else out


def decode_distribution(dist: np.ndarray, inst: VRPInstance, top: int = 1):
    """Converte a distribuição do POVM em rotas, da mais provável para a menos."""
    R = inst.R
    flat = np.argsort(dist.reshape(-1))[::-1][:top]
    out = []
    for f in flat:
        slots = np.unravel_index(f, dist.shape)
        v = np.array([s // R for s in slots])
        r = np.array([s % R for s in slots])
        routes: List[List[int]] = []
        for veh in range(inst.M):
            cities = [i for i in range(inst.N) if v[i] == veh]
            cities.sort(key=lambda i: (r[i], i))
            routes.append([i + 1 for i in cities])
        out.append({"probability": float(dist.reshape(-1)[f]),
                    "position": r, "vehicle": v, "routes": routes,
                    "cost": routes_cost(routes, inst.D)})
    return out


# ---------------------------------------------------------------------------
# 6. Verificação exata — sempre em numpy float64
# ---------------------------------------------------------------------------
def node_state(ops: ModularOperators, v: int, r: int) -> np.ndarray:
    """Melhor estado de um modo para o nó (v, r): autovetor dominante de A_{v,r}.

    No limite de cutoff infinito este seria um estado de Zak perfeitamente
    localizado no nó. No cutoff finito é o estado representável que mais se
    aproxima disso, e o autovalor associado mede quão perto.
    """
    w, V = np.linalg.eigh(_hermitian(ops.A[v, r]))
    return V[:, -1]


def node_energy_table(inst: VRPInstance, ops: ModularOperators,
                      params: Optional[HamiltonianParams] = None,
                      require_all_vehicles: Optional[bool] = None,
                      max_configs: int = 2_000_000) -> Dict[str, object]:
    """<H> em estados-produto centrados em cada configuração de nós.

    Análogo direto do antigo brute_force_grid: percorre as (M*R)^N
    configurações e devolve a energia de cada uma. O argmin deve decodificar
    para a solução ótima clássica.
    """
    N, M, R = inst.N, inst.M, inst.R
    size = (M * R) ** N
    if size > max_configs:
        raise MemoryError(f"(M·R)^N = {size:,} excede max_configs={max_configs:,}.")
    params = params or HamiltonianParams()
    kets = {(v, r): node_state(ops, v, r) for v in range(M) for r in range(R)}

    energies = np.empty((M * R,) * N, dtype=np.float64)
    for slots in itertools.product(range(M * R), repeat=N):
        psi = None
        for s in slots:
            v, r = divmod(s, R)
            psi = kets[(v, r)] if psi is None else np.multiply.outer(psi, kets[(v, r)])
        e = energy_components(psi, inst, ops, params, require_all_vehicles)
        energies[slots] = float(e["encoding"])

    flat = int(np.argmin(energies))
    slots = np.unravel_index(flat, energies.shape)
    v = np.array([s // R for s in slots])
    r = np.array([s % R for s in slots])
    routes: List[List[int]] = []
    for veh in range(M):
        cities = [i for i in range(N) if v[i] == veh]
        cities.sort(key=lambda i: (r[i], i))
        routes.append([i + 1 for i in cities])
    return {"energies": energies, "min_energy": float(energies.min()),
            "argmin_slots": slots, "routes": routes,
            "cost": routes_cost(routes, inst.D), "size": size}


def dense_hamiltonian(inst: VRPInstance, ops: ModularOperators,
                      params: Optional[HamiltonianParams] = None,
                      require_all_vehicles: Optional[bool] = None,
                      max_dim: int = 30_000) -> np.ndarray:
    """Matriz densa de H no espaço de Fock truncado, dimensão cutoff^N.

    Só para verificação: o fundamental decodificado deve ser a rota ótima. É a
    afirmação mais forte possível, porque o fundamental é o mínimo sobre TODOS
    os estados — não existe mínimo espúrio, dentro ou fora dos nós.
    """
    params = params or HamiltonianParams()
    d, N, M, R = ops.cutoff, inst.N, inst.M, inst.R
    dim = d ** N
    if dim > max_dim:
        raise MemoryError(f"cutoff^N = {dim:,} excede max_dim={max_dim:,}; "
                          "use scipy.sparse.linalg.eigsh sobre um operador-função.")
    req_all = requires_all_vehicles(params, require_all_vehicles)
    I = np.eye(dim, dtype=np.complex128)

    def emb(op, mode):
        mats = [np.eye(d, dtype=np.complex128)] * N
        mats[mode] = op
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    Ae = {(i, v, r): emb(ops.A[v, r], i)
          for i in range(N) for v in range(M) for r in range(R)}
    occ = {(v, r): sum(Ae[i, v, r] for i in range(N))
           for v in range(M) for r in range(R)}
    D = inst.D

    Hd = np.zeros((dim, dim), dtype=np.complex128)
    for v in range(M):
        for i in range(N):
            Hd += D[0, i + 1] * Ae[i, v, 0]
        for r in range(R - 1):
            for i in range(N):
                for j in range(N):
                    if i != j:
                        Hd += D[i + 1, j + 1] * (Ae[i, v, r] @ Ae[j, v, r + 1])
            B = I - occ[v, r + 1]
            for i in range(N):
                Hd += D[i + 1, 0] * 0.5 * (Ae[i, v, r] @ B + B @ Ae[i, v, r])
        for i in range(N):
            Hd += D[i + 1, 0] * Ae[i, v, R - 1]

    Hc = sum(Ae[i, v, r] @ Ae[j, v, r] for v in range(M) for r in range(R)
             for i in range(N) for j in range(i + 1, N))
    Hc = Hc if isinstance(Hc, np.ndarray) else np.zeros((dim, dim), dtype=np.complex128)

    Hg = np.zeros((dim, dim), dtype=np.complex128)
    for v in range(M):
        for r in range(1, R):
            B = float(r) * I - sum(occ[v, q] for q in range(r))
            for i in range(N):
                Hg += B @ Ae[i, v, r] @ B

    Hv = np.zeros((dim, dim), dtype=np.complex128)
    if req_all:
        for v in range(M):
            y = occ[v, 0] - I
            Hv += y @ y

    Hcap = np.zeros((dim, dim), dtype=np.complex128)
    if params.lambda_cap > 0:
        for v in range(M):
            Lv = sum(float(inst.demands[i]) * emb(ops.P_vehicle[v], i) for i in range(N))
            w, Vv = np.linalg.eigh(_hermitian(Lv))
            Hcap += (Vv * np.maximum(w.real - float(inst.Q[v]), 0.0) ** 2) @ Vv.conj().T

    Hdisc = sum(emb(ops.H_disc, i) for i in range(N))
    Hexc = sum(emb(ops.n_op, i) for i in range(N))

    H = (params.lambda_dist * Hd + params.lambda_col * Hc + params.lambda_gap * Hg
         + params.lambda_vehicle * Hv + params.lambda_cap * Hcap
         + params.lambda_disc * Hdisc + params.lambda_exc * Hexc)
    return _hermitian(H)


def ground_state(inst: VRPInstance, ops: ModularOperators,
                 params: Optional[HamiltonianParams] = None,
                 require_all_vehicles: Optional[bool] = None,
                 k: int = 1, max_dim: int = 30_000) -> Dict[str, object]:
    """Diagonaliza H e decodifica o fundamental pelo POVM.

    max_dim é repassado a `dense_hamiltonian`: acima dele a montagem da matriz
    densa é recusada com MemoryError, porque cutoff^N ao quadrado é o que
    realmente pesa (em cutoff^N = 32768 a matriz já passa de 17 GB). Para
    instâncias maiores é preciso um caminho esparso, com eigsh sobre um
    LinearOperator em vez da matriz montada.
    """
    H = dense_hamiltonian(inst, ops, params, require_all_vehicles,
                          max_dim=max_dim)
    w, V = np.linalg.eigh(H)
    psi = V[:, 0].reshape((ops.cutoff,) * inst.N)
    dist = assignment_distribution(psi, inst, ops)
    best = decode_distribution(dist, inst, top=1)[0]
    return {"eigenvalues": w[:max(k, 1)].real, "energy": float(w[0].real),
            "psi": psi, "distribution": dist, "map": best,
            "gap": float(w[1].real - w[0].real) if w.size > 1 else float("nan")}