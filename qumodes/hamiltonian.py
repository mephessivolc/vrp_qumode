from dataclasses import dataclass
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate


@dataclass
class HamiltonianParams:
    sigma_assign: float = 0.30
    sigma_empty: float = 0.40
    lambda_col: float = 20.0
    lambda_gap: float = 20.0
    lambda_cap: float = 15.0
    alpha2: float = 1.0
    alpha4: float = 1.0


def local_x_operator(cutoff: int) -> np.ndarray:
    a = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for n in range(cutoff - 1):
        a[n, n + 1] = np.sqrt(n + 1)
    return (a + a.conj().T) / np.sqrt(2.0)


def route_coordinate(x, M: int, N: int, x_min=None, x_max=None):
    x = np.asarray(x, dtype=np.float64)
    if x_min is None:
        x_min = float(np.min(x))
    if x_max is None:
        x_max = float(np.max(x))
    if np.isclose(x_max, x_min):
        raise ValueError("Affine mapping requires x_max != x_min.")
    return 1.0 + (M * N - 1.0) * ((x - x_min) / (x_max - x_min))


def local_z_operator(M: int, N: int, cutoff: int):
    X = local_x_operator(cutoff)
    x_eigs, Ux = np.linalg.eigh(X)
    z_eigs = route_coordinate(
        x=x_eigs,
        M=M,
        N=N,
        x_min=float(x_eigs.min()),
        x_max=float(x_eigs.max()),
    )
    Z = (Ux * z_eigs) @ Ux.conj().T
    return Z, X, x_eigs, z_eigs, Ux


def soft_assignments(z, M: int, N: int, sigma_assign: float):
    z = np.asarray(z, dtype=np.float64)
    centers = np.arange(1, M * N + 1, dtype=np.float64).reshape(M, N)
    logits = -(z[:, None, None] - centers[None, :, :]) ** 2 / (2.0 * sigma_assign**2)
    flat = logits.reshape(N, M * N)
    flat -= flat.max(axis=1, keepdims=True)
    weights = np.exp(flat)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights.reshape(N, M, N)


def occupancies(a):
    return np.sum(a, axis=0)


def distance_term(a, D, sigma_empty: float) -> float:
    N, M, _ = a.shape
    n = occupancies(a)

    H_start = np.sum(a[:, :, 0] * D[0, 1:, None])

    H_internal = 0.0
    D_city = D[1:, 1:]
    different_city = 1.0 - np.eye(N)
    for r in range(N - 1):
        left = a[:, :, r].T
        right = a[:, :, r + 1].T
        edge_mass = left[:, :, None] * right[:, None, :]
        H_internal += np.sum(edge_mass * D_city[None, :, :] * different_city[None, :, :])

    H_return = 0.0
    d_to_depot = D[1:, 0]
    empty_next = np.exp(-(n[:, 1:] ** 2) / (2.0 * sigma_empty**2))
    for r in range(N - 1):
        last_mass = a[:, :, r] * empty_next[:, r][None, :]
        H_return += np.sum(last_mass * d_to_depot[:, None])
    H_return += np.sum(a[:, :, -1] * d_to_depot[:, None])

    return float(H_start + H_internal + H_return)


def collision_term(a, lambda_col: float) -> float:
    n = occupancies(a)
    pair_mass = 0.5 * (n**2 - np.sum(a**2, axis=0))
    return float(lambda_col * np.sum(pair_mass))


def gap_term(a, lambda_gap: float) -> float:
    n = occupancies(a)
    prefix_before = np.cumsum(n, axis=1) - n
    N = a.shape[2]
    required_before = np.arange(N, dtype=np.float64)[None, :]
    return float(lambda_gap * np.sum(n[:, 1:] * (required_before[:, 1:] - prefix_before[:, 1:]) ** 2))


def capacity_term(a, demands, Q, lambda_cap: float, alpha2: float = 1.0, alpha4: float = 1.0) -> float:
    """
    Penalidade de capacidade via Hinge Polinomial.
    L_v = sum_{i, r} d_i * a[i, v, r]
    """
    demands = np.asarray(demands, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    L_v = np.sum(a * demands[:, None, None], axis=(0, 2))
    excess = L_v - Q
    hinge = alpha2 * (excess**2) + alpha4 * (excess**4)
    return float(lambda_cap * np.sum(hinge))


def hamiltonian_energy(z, M: int, N: int, D, demands, Q, params: HamiltonianParams = None):
    if params is None:
        params = HamiltonianParams()

    D = np.asarray(D, dtype=np.float64)
    a = soft_assignments(z=z, M=M, N=N, sigma_assign=params.sigma_assign)

    H_dist = distance_term(a=a, D=D, sigma_empty=params.sigma_empty)
    H_col = collision_term(a=a, lambda_col=params.lambda_col)
    H_gap = gap_term(a=a, lambda_gap=params.lambda_gap)
    H_capacity = capacity_term(
        a=a,
        demands=demands,
        Q=Q,
        lambda_cap=params.lambda_cap,
        alpha2=params.alpha2,
        alpha4=params.alpha4,
    )

    H_total = H_dist + H_col + H_gap + H_capacity

    return {
        "total": float(H_total),
        "dist": float(H_dist),
        "col": float(H_col),
        "gap": float(H_gap),
        "capacity": float(H_capacity),
    }


def create_vrp_program(N: int, gate_params: dict) -> sf.Program:
    """
    Cria um objeto sf.Program parametrizado para N qumodes.
    gate_params deve ser um dicionario com os parametros variacionais do circuito quântico.
    """
    prog = sf.Program(N)
    with prog.context as q:
        for i in range(N):
            Sgate(gate_params.get(f"sq_r_{i}", 0.1), gate_params.get(f"sq_phi_{i}", 0.0)) | q[i]
            Dgate(gate_params.get(f"d_r_{i}", 0.1), gate_params.get(f"d_phi_{i}", 0.0)) | q[i]
            Kgate(gate_params.get(f"kerr_{i}", 0.05)) | q[i]

        for i in range(N - 1):
            BSgate(gate_params.get(f"bs_theta_{i}", 0.785), gate_params.get(f"bs_phi_{i}", 0.0)) | (q[i], q[i + 1])

    return prog


def evaluate_sf_state(
    state: sf.backends.states.BaseState,
    M: int,
    N: int,
    D,
    demands,
    Q,
    cutoff: int = 10,
    params: HamiltonianParams = None,
) -> dict:
    """
    Recebe um estado quântico exportado pelo backend Fock do Strawberry Fields (state)
    e calcula o valor esperado dos componentes do Hamiltoniano na base de autovetores de X.
    """
    if params is None:
        params = HamiltonianParams()

    _, _, _, z_eigs, Ux = local_z_operator(M=M, N=N, cutoff=cutoff)

    # Extrai o vetor de estado / matriz densidade
    if state.is_pure:
        psi_fock = state.ket()
        shape = [cutoff] * N
        psi_tensor = psi_fock.reshape(shape)

        # Transforma da base de Fock para a base de autovetores de X
        psi_x = psi_tensor
        for mode in range(N):
            psi_x = np.moveaxis(psi_x, mode, 0)
            sh = psi_x.shape
            psi_x = Ux.conj().T @ psi_x.reshape(cutoff, -1)
            psi_x = psi_x.reshape(sh)
            psi_x = np.moveaxis(psi_x, 0, mode)

        probs = np.abs(psi_x) ** 2
    else:
        dm = state.dm()
        # Para estados mistos, calcula-se a diagonal apos mudança de base
        probs = np.real(np.diag(dm)).reshape([cutoff] * N)

    total_dim = cutoff**N
    exp_energies = {"total": 0.0, "dist": 0.0, "col": 0.0, "gap": 0.0, "capacity": 0.0}

    # Integração sobre a malha de autovalores
    for flat_idx in range(total_dim):
        indices = np.unravel_index(flat_idx, [cutoff] * N)
        prob = probs[indices]
        if prob < 1e-12:
            continue

        z = np.array([z_eigs[k] for k in indices], dtype=np.float64)
        energies = hamiltonian_energy(z=z, M=M, N=N, D=D, demands=demands, Q=Q, params=params)

        for key in exp_energies:
            exp_energies[key] += prob * energies[key]

    return exp_energies


def run_sf_hamiltonian_pipeline(
    gate_params: dict,
    M: int,
    N: int,
    D,
    demands,
    Q,
    cutoff: int = 10,
    params: HamiltonianParams = None,
):
    """
    Constrói o circuito no Strawberry Fields, executa-o no backend de Fock e retorna
    o objeto Result do SF e as energias esperadas do Hamiltoniano refatorado.
    """
    prog = create_vrp_program(N=N, gate_params=gate_params)
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    results = eng.run(prog)

    energies = evaluate_sf_state(
        state=results.state,
        M=M,
        N=N,
        D=D,
        demands=demands,
        Q=Q,
        cutoff=cutoff,
        params=params,
    )

    return results, energies