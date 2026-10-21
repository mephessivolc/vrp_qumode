"""
hamiltonian_sparse.py — fundamental do Hamiltoniano modular SEM montar a matriz.

Colocar em  qumodes/hamiltonian_sparse.py

Por que existe
--------------
dense_hamiltonian (hamiltonian.py) monta N·M·R matrizes dim x dim (dim = d^N),
mais M·R de ocupação e ~10 auxiliares. O consumo real é da ordem de
(N·M·R + M·R + 10)·dim²·16 bytes, e não dim²·16. Em C=4, V=1, d=10 só as
matrizes locais já passam de 25 GB.

H é uma soma de produtos de operadores de UM modo (A_{v,r}, H_disc, n) e de
uma função do operador de carga. Logo H|psi> sai de contrações locais sobre o
tensor psi de forma (d,)*N, sem nunca formar H: custo O(dim·d) por contração,
memória O(dim) (mais N·M·R·dim para os vetores intermediários T). Um
eigsh/eigs sobre o LinearOperator dá o fundamental.

Fidelidade
----------
apply() reproduz dense_hamiltonian termo a termo, com os MESMOS ordenamentos:
  distância   A_i^{(r)} A_j^{(r+1)} (modos distintos) e {A, I-n}/2
  colisão     sum_{i<j} A_i A_j = (n^2 - sum_i A_i^2) / 2
              (identidade exata: modos distintos comutam)
  lacuna      B n B,  B = c I - sum_Q n_q   (sanduíche; c e Q vêm de
              H.gap_operator_spec conforme params.gap_form)
  veículo     (n_0 - I)^2
  capacidade  cálculo funcional na base própria de P_v (produto de modos)
  disc, exc   somas de um modo
NÃO confie nos resultados antes de rodar self_check(): ele compara H|psi> com
a matriz densa em casos pequenos (inclui capacidade e confinamento).

Observação sobre o solver
-------------------------
H é complexo hermitiano (U, V e G_v têm parte imaginária quando M >= 3).
Para dtype complexo o scipy.sparse.linalg.eigsh delega a `eigs` (Arnoldi) com
which='SR'. Funciona, mas o custo por iteração é maior que o de Lanczos real.

Uso
---
    import qumodes.hamiltonian_sparse as HS
    gs = HS.ground_state_sparse(inst, ops, params)    # mesmo formato de H.ground_state
    python -m qumodes.hamiltonian_sparse              # self_check
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

import qumodes.hamiltonian as H

__all__ = ["ModularHamiltonianOperator", "ground_state_sparse", "self_check"]


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def _loc(psi: np.ndarray, op: np.ndarray, mode: int) -> np.ndarray:
    """Aplica `op` (d x d) ao eixo `mode` de psi.

    Mesma contração do _NumpyBackend.apply_local de hamiltonian.py:
    out[..., m, ...] = sum_n op[m, n] psi[..., n, ...].
    """
    n = psi.ndim
    out = np.tensordot(psi, op, axes=([mode], [1]))
    perm = list(range(mode)) + [n - 1] + list(range(mode, n - 1))
    return np.transpose(out, perm)


def _sum_arrays(arrs: List[np.ndarray]) -> np.ndarray:
    acc = np.array(arrs[0], dtype=np.complex128, order="C", copy=True)
    for t in arrs[1:]:
        acc += t
    return acc


# ---------------------------------------------------------------------------
# Operador
# ---------------------------------------------------------------------------
class ModularHamiltonianOperator(LinearOperator):
    """H como LinearOperator complex128 sobre vetores de comprimento d^N."""

    def __init__(self, inst: H.VRPInstance, ops: H.ModularOperators,
                 params: Optional[H.HamiltonianParams] = None,
                 require_all_vehicles: Optional[bool] = None):
        params = params or H.HamiltonianParams()
        nc, nv, nr, d = inst.N, inst.M, inst.R, ops.cutoff
        if (ops.M, ops.R) != (nv, nr):
            raise ValueError(f"ops (M={ops.M}, R={ops.R}) não corresponde à "
                             f"instância (M={nv}, R={nr}).")
        dim = d ** nc
        super().__init__(dtype=np.dtype(np.complex128), shape=(dim, dim))

        self._p = params
        self._nc, self._nv, self._nr, self._d = nc, nv, nr, d
        self._req_all = H.requires_all_vehicles(params, require_all_vehicles)

        self._A = [[np.ascontiguousarray(ops.A[v, r]) for r in range(nr)]
                   for v in range(nv)]
        self._A2 = [[a @ a for a in row] for row in self._A]
        self._Hdisc = np.ascontiguousarray(ops.H_disc)
        self._d0 = np.asarray(inst.d0, dtype=np.float64)
        self._dback = np.asarray(inst.dback, dtype=np.float64)
        self._Dcc = np.asarray(inst.Dcc, dtype=np.float64)   # diagonal zerada

        # confinamento: sum_i n_i é diagonal, vira produto elemento a elemento
        self._nsum = None
        if params.lambda_exc != 0.0:
            nsum = np.zeros((d,) * nc, dtype=np.float64)
            for i in range(nc):
                shape = [1] * nc
                shape[i] = d
                nsum = nsum + np.arange(d, dtype=np.float64).reshape(shape)
            self._nsum = nsum

        # capacidade: L_v = sum_i q_i P_v^(i) tem base própria = produto das
        # bases próprias de P_v; f(L) = max(L - Q, 0)^2 é diagonal nela.
        self._cap = None
        if params.lambda_cap > 0:
            if inst.demands is None or inst.Q is None:
                raise ValueError("lambda_cap > 0 exige demands e Q na instância.")
            self._cap = []
            for v in range(nv):
                evals, evecs = np.linalg.eigh(H._hermitian(ops.P_vehicle[v]))
                load = np.zeros((d,) * nc, dtype=np.float64)
                for i in range(nc):
                    shape = [1] * nc
                    shape[i] = d
                    load = load + float(inst.demands[i]) * evals.real.reshape(shape)
                w = np.maximum(load - float(inst.Q[v]), 0.0) ** 2
                self._cap.append((np.ascontiguousarray(evecs),
                                  np.ascontiguousarray(evecs.conj().T), w))

        self.n_matvec = 0

    # --- interface LinearOperator -------------------------------------
    def _adjoint(self):
        return self                       # H é hermitiano

    def _matvec(self, x):
        self.n_matvec += 1
        psi = np.asarray(x, dtype=np.complex128).reshape((self._d,) * self._nc)
        return self.apply(psi).reshape(-1)

    # --- H|psi> --------------------------------------------------------
    def apply(self, psi: np.ndarray) -> np.ndarray:
        N, M, R = self._nc, self._nv, self._nr
        A, A2, p = self._A, self._A2, self._p
        loc = _loc

        # T[v][r][i] = A_i^{(v,r)} psi ;  nps[v][r] = n_{v,r} psi
        T = [[[loc(psi, A[v][r], i) for i in range(N)] for r in range(R)]
             for v in range(M)]
        nps = [[_sum_arrays(T[v][r]) for r in range(R)] for v in range(M)]

        def n_apply(x, v, r):
            return _sum_arrays([loc(x, A[v][r], i) for i in range(N)])

        out = np.zeros_like(psi)

        # --- distância ---------------------------------------------------
        if p.lambda_dist != 0.0:
            h = np.zeros_like(psi)
            for v in range(M):
                # depósito -> primeira posição
                for i in range(N):
                    h += self._d0[i] * T[v][0][i]
                # arcos internos: sum_{i != j} D_ij A_i^{(r)} A_j^{(r+1)}
                # e última -> depósito: (1/2) sum_i D_i0 {A_i^{(r)}, I - n_{r+1}}
                for r in range(R - 1):
                    for i in range(N):
                        W = np.zeros_like(psi)
                        for j in range(N):
                            c = self._Dcc[i, j]
                            if c != 0.0:
                                W += c * T[v][r + 1][j]
                        h += loc(W, A[v][r], i)
                    Bpsi = psi - nps[v][r + 1]
                    part1 = np.zeros_like(psi)      # sum_i D_i0 A_i^{(r)} (B psi)
                    S = np.zeros_like(psi)          # sum_i D_i0 A_i^{(r)} psi
                    for i in range(N):
                        part1 += self._dback[i] * loc(Bpsi, A[v][r], i)
                        S += self._dback[i] * T[v][r][i]
                    part2 = S - n_apply(S, v, r + 1)          # B S
                    h += 0.5 * (part1 + part2)
                # cidade na última posição -> depósito
                for i in range(N):
                    h += self._dback[i] * T[v][R - 1][i]
            out += p.lambda_dist * h

        # --- colisão -------------------------------------------------------
        if p.lambda_col != 0.0:
            hc = np.zeros_like(psi)
            for v in range(M):
                for r in range(R):
                    nn = n_apply(nps[v][r], v, r)             # n^2 psi
                    sq = np.zeros_like(psi)                   # sum_i A_i^2 psi
                    for i in range(N):
                        sq += loc(psi, A2[v][r], i)
                    hc += 0.5 * (nn - sq)
            out += p.lambda_col * hc

        # --- lacuna: B n B ---------------------------------------------------
        if p.lambda_gap != 0.0 and R > 1:
            hg = np.zeros_like(psi)
            for v in range(M):
                for r in range(1, R):
                    c_gap, qs = H.gap_operator_spec(p.gap_form, r)
                    y = c_gap * psi
                    for q in qs:
                        y = y - nps[v][q]                     # y = B psi
                    z = n_apply(y, v, r)                      # n_r B psi
                    w = c_gap * z
                    for q in qs:
                        w = w - n_apply(z, v, q)              # B n_r B psi
                    hg += w
            out += p.lambda_gap * hg

        # --- veículos obrigatórios: (n_0 - I)^2 -------------------------------
        if self._req_all and p.lambda_vehicle != 0.0:
            hv = np.zeros_like(psi)
            for v in range(M):
                y = nps[v][0] - psi
                hv += n_apply(y, v, 0) - y
            out += p.lambda_vehicle * hv

        # --- capacidade ---------------------------------------------------------
        if self._cap is not None:
            hcap = np.zeros_like(psi)
            for evecs, Vh, w in self._cap:
                c = psi
                for i in range(N):
                    c = loc(c, Vh, i)
                c = c * w
                for i in range(N):
                    c = loc(c, evecs, i)
                hcap += c
            out += p.lambda_cap * hcap

        # --- estabilizador e confinamento ---------------------------------------
        if p.lambda_disc != 0.0:
            hd = np.zeros_like(psi)
            for i in range(N):
                hd += loc(psi, self._Hdisc, i)
            out += p.lambda_disc * hd
        if self._nsum is not None:
            out += p.lambda_exc * (self._nsum * psi)

        return out


# ---------------------------------------------------------------------------
# Fundamental
# ---------------------------------------------------------------------------
def ground_state_sparse(inst: H.VRPInstance, ops: H.ModularOperators,
                        params: Optional[H.HamiltonianParams] = None,
                        require_all_vehicles: Optional[bool] = None,
                        k: int = 2, ncv: int = 48, tol: float = 1e-9,
                        maxiter: int = 5000, v0: Optional[np.ndarray] = None,
                        seed: int = 0, force_sparse: bool = False,
                        decode: bool = True
                        ) -> Dict[str, object]:
    """Fundamental de H sem matriz densa. Mesmo formato de H.ground_state.

    decode=False calcula SÓ a energia (e residual/convergência): psi, distribution
    e map vêm como None. A decodificação faz (M*R)^N contrações sobre o tensor d^N
    e domina o custo em cutoff alto; para varrer E0 contra o cutoff ela é
    desnecessária.

    Chaves extras: residual (||H psi - E psi||), converged, n_matvec.
    v0 padrão: vetor aleatório com semente fixa (não enviesa para a resposta).
    Para dim < 64 delega à diagonalização densa, salvo force_sparse=True.
    """
    params = params or H.HamiltonianParams()
    d, N = ops.cutoff, inst.N
    dim = d ** N
    if dim < 64 and not force_sparse:
        return H.ground_state(inst, ops, params, require_all_vehicles,
                              k=k, max_dim=max(dim, 64))

    Hop = ModularHamiltonianOperator(inst, ops, params, require_all_vehicles)
    k = max(1, min(int(k), dim - 2))
    ncv = max(k + 2, min(int(ncv), dim))
    if v0 is None:
        rng = np.random.default_rng(seed)
        v0 = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)

    converged = True
    try:
        w, Vs = eigsh(Hop, k=k, which="SA", ncv=ncv, tol=tol,
                      maxiter=maxiter, v0=v0)
    except ArpackNoConvergence as exc:
        converged = False
        w, Vs = exc.eigenvalues, exc.eigenvectors
        if w is None or len(w) == 0:
            raise
    order = np.argsort(np.real(w))
    w = np.real(w)[order]
    Vs = Vs[:, order]

    vec = Vs[:, 0]
    vec = vec / np.linalg.norm(vec)
    residual = float(np.linalg.norm(Hop.matvec(vec) - w[0] * vec))

    if not decode:
        return {"eigenvalues": w[:max(k, 1)], "energy": float(w[0]),
                "psi": None, "distribution": None, "map": None,
                "gap": float(w[1] - w[0]) if w.size > 1 else float("nan"),
                "residual": residual, "converged": converged,
                "n_matvec": int(Hop.n_matvec)}

    psi = vec.reshape((d,) * N)
    dist = H.assignment_distribution(psi, inst, ops)
    best = H.decode_distribution(dist, inst, top=1)[0]
    return {"eigenvalues": w[:max(k, 1)], "energy": float(w[0]),
            "psi": psi, "distribution": dist, "map": best,
            "gap": float(w[1] - w[0]) if w.size > 1 else float("nan"),
            "residual": residual, "converged": converged,
            "n_matvec": int(Hop.n_matvec)}


# ---------------------------------------------------------------------------
# Verificação contra a matriz densa (rodar ANTES de usar o caminho esparso)
# ---------------------------------------------------------------------------
def _make_case(C, V, d, seed, hbar, heavy, gap_form="soma"):
    rng = np.random.default_rng(seed)
    D = np.zeros((C + 1, C + 1))
    for i in range(C + 1):
        for j in range(i + 1, C + 1):
            D[i, j] = D[j, i] = float(rng.integers(1, 21))
    if heavy:
        # exercita capacidade (Q=1.5 < demanda acumulada) e confinamento
        inst = H.VRPInstance(D=D, M=V, R=C, demands=np.ones(C), Q=1.5, hbar=hbar)
        par = H.HamiltonianParams(
            lambda_dist=1.0, lambda_col=30.0, lambda_gap=30.0,
            lambda_vehicle=30.0 if V > 1 else 0.0,
            lambda_cap=7.0, lambda_disc=5.0, lambda_exc=0.3,
            gap_form=gap_form)
    else:
        inst = H.instance_from_legacy(C, V, D, np.ones(C), np.full(V, float(C)),
                                      hbar=hbar, require_all_vehicles=(V > 1))
        par = H.HamiltonianParams(
            lambda_dist=1.0, lambda_col=30.0, lambda_gap=30.0,
            lambda_vehicle=30.0 if V > 1 else 0.0,
            lambda_cap=0.0, lambda_disc=5.0, lambda_exc=0.0,
            gap_form=gap_form)
    ops = H.build_modular_operators(d, inst.M, inst.R, hbar)
    return inst, ops, par


def self_check(cases=((3, 1, 4), (3, 2, 4), (3, 3, 5), (4, 2, 4)),
               hbar: float = 2.0, tol_matvec: float = 1e-10) -> bool:
    """1) H|psi> esparso == matriz densa @ psi, e hermiticidade.
       2) energias do fundamental (k=2) esparso == denso.
    """
    ok_all = True
    rng = np.random.default_rng(123)

    print("--- (1) matvec contra a matriz densa ---")
    for heavy in (False, True):
        for (C, V, d) in cases:
            inst, ops, par = _make_case(C, V, d, 42, hbar, heavy)
            Hop = ModularHamiltonianOperator(inst, ops, par)
            dim = d ** C
            Hd = H.dense_hamiltonian(inst, ops, par, max_dim=dim + 1)
            x = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            y = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            x /= np.linalg.norm(x)
            y /= np.linalg.norm(y)
            ref = Hd @ x
            got = Hop.matvec(x)
            err = float(np.linalg.norm(ref - got) / np.linalg.norm(ref))
            herm = abs(np.vdot(y, Hop.matvec(x)) - np.vdot(Hop.matvec(y), x))
            herm = float(herm / max(abs(np.vdot(y, Hop.matvec(x))), 1e-300))
            ok = (err < tol_matvec) and (herm < tol_matvec)
            ok_all &= ok
            print(f"heavy={int(heavy)} C={C} V={V} d={d}  erro_rel={err:.2e}  "
                  f"herm={herm:.2e}  {'OK' if ok else 'FALHOU'}")

    print("\n--- (2) fundamental: esparso vs denso (k=2) ---")
    for (C, V, d) in ((3, 2, 6), (4, 1, 4)):
        inst, ops, par = _make_case(C, V, d, 7, hbar, False)
        gd = H.ground_state(inst, ops, par, k=2, max_dim=d ** C + 1)
        gs = ground_state_sparse(inst, ops, par, k=2, tol=1e-12,
                                 force_sparse=True)
        dE = float(np.max(np.abs(np.asarray(gd["eigenvalues"])
                                 - np.asarray(gs["eigenvalues"]))))
        same = gd["map"]["routes"] == gs["map"]["routes"]
        ok = (dE < 1e-6) and (gs["residual"] < 1e-5)
        ok_all &= ok
        print(f"C={C} V={V} d={d}  max|dE|={dE:.2e}  residual={gs['residual']:.2e}  "
              f"matvecs={gs['n_matvec']}  rota_igual={same}  "
              f"{'OK' if ok else 'FALHOU'}")
        print(f"      denso : E={gd['energy']:.6f}  rotas={gd['map']['routes']}")
        print(f"      esparso: E={gs['energy']:.6f}  rotas={gs['map']['routes']}")

    print("\n" + ("TODOS OK" if ok_all else "HA FALHAS — nao use o caminho esparso"))
    return ok_all


def self_check_formas(hbar: float = 2.0, tol: float = 1e-10) -> bool:
    """gap_form='predecessor':
       (3) H|psi> esparso == matriz densa (inclui capacidade e confinamento);
       (4) invariante: as duas formas dão a MESMA matriz quando R <= 2 e
           matrizes DIFERENTES quando R >= 3.
    """
    ok_all = True
    rng = np.random.default_rng(321)

    print("\n--- (3) gap_form='predecessor': matvec contra a matriz densa ---")
    for heavy in (False, True):
        for (C, V, d) in ((3, 1, 4), (3, 2, 4), (4, 2, 4)):
            inst, ops, par = _make_case(C, V, d, 42, hbar, heavy,
                                        gap_form="predecessor")
            Hop = ModularHamiltonianOperator(inst, ops, par)
            dim = d ** C
            Hd = H.dense_hamiltonian(inst, ops, par, max_dim=dim + 1)
            x = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            x /= np.linalg.norm(x)
            ref = Hd @ x
            err = float(np.linalg.norm(ref - Hop.matvec(x)) / np.linalg.norm(ref))
            ok = err < tol
            ok_all &= ok
            print(f"heavy={int(heavy)} C={C} V={V} d={d} R={inst.R}  "
                  f"erro_rel={err:.2e}  {'OK' if ok else 'FALHOU'}")

    print("\n--- (4) soma contra predecessor: iguais se R<=2, diferentes se R>=3 ---")
    for (C, V, d) in ((3, 2, 4), (4, 3, 3), (3, 1, 4), (4, 2, 4)):
        inst, ops, p_soma = _make_case(C, V, d, 42, hbar, False, "soma")
        _, _, p_pred = _make_case(C, V, d, 42, hbar, False, "predecessor")
        dim = d ** C
        H1 = H.dense_hamiltonian(inst, ops, p_soma, max_dim=dim + 1)
        H2 = H.dense_hamiltonian(inst, ops, p_pred, max_dim=dim + 1)
        dif = float(np.linalg.norm(H1 - H2) / np.linalg.norm(H1))
        igual_esperado = inst.R <= 2
        ok = (dif < 1e-12) if igual_esperado else (dif > 1e-6)
        ok_all &= ok
        print(f"C={C} V={V} d={d} R={inst.R}  dif_rel={dif:.2e}  "
              f"esperado={'igual' if igual_esperado else 'diferente'}  "
              f"{'OK' if ok else 'FALHOU'}")

    print("\n" + ("FORMAS OK" if ok_all else "HA FALHAS nas formas do termo de lacuna"))
    return ok_all


if __name__ == "__main__":
    ok1 = self_check()
    ok2 = self_check_formas()
    raise SystemExit(0 if (ok1 and ok2) else 1)