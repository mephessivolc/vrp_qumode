"""
product_energy.py — energia de ESTADOS-PRODUTO sem formar o tensor d^N.

Colocar em  qumodes/product_energy.py

Ideia
-----
Para |Psi> = ⊗_i |phi_i>, o valor esperado de um produto ordenado de operadores
de UM modo fatora cidade a cidade (modos distintos comutam):

    <Psi| X_1 ... X_k |Psi> = Π_i < phi_i | (produto, em ordem, dos fatores da cidade i) | phi_i >

Cada termo de H vira, então, polinômio nos MOMENTOS de uma cidade:

    a[i, x]        = <phi_i| A_x |phi_i>                       1o momento
    b[i, x, y]     = <phi_i| A_x A_y |phi_i>                   2o momento
    t[i, x, y, z]  = <phi_i| A_x A_y A_z |phi_i>               3o momento
    hd[i], nph[i]  = <phi_i| H_disc |phi_i>, <phi_i| n |phi_i>

com x = v*R + r o índice do nó (v, r). O custo cai de O(d^N) para O(N^2) por
configuração, depois de um pré-cálculo de O(X^2 d^2) por estado (X = M*R).

Fórmulas (todas conferidas contra energy_components por `verificar`):
  dist     depósito->primeira, arcos internos, anticomutador {A, I-n}/2
  col      sum_{i<j} a_i a_j
  gap      <B A_i B> = c^2 a_i - 2c (Re u_i + a_i s_-i)
                       + a_i (s_-i^2 + sum_{j!=i} Var_j(S)) + 2 s_-i Re u_i + w_i
           com S_j = sum_{q in Q} A_j^{(q)}, s_j = <S_j>, u_i = <S_i A_i>,
           w_i = <S_i A_i S_i> e Var_j = <S_j^2> - <S_j>^2.
           (c, Q) vêm de hamiltonian.gap_operator_spec(gap_form, r).
  vehicle  <(n_0 - I)^2> = <n_0^2> - 2<n_0> + 1
  disc, exc  somas de um modo

NÃO suporta lambda_cap > 0 (o termo de capacidade é um funcional espectral da
carga total, não um polinômio nos momentos).

A saída traz as 6 "peças" do termo de lacuna (sem peso lambda_gap), úteis para
localizar o mecanismo do viés entre perfis de rota:
  P1 c^2 <A>          P2 termo cruzado        P3a campo médio de outras cidades
  P3b flutuação (Var) das outras cidades      P4 2 s_-i Re u      P5 <S A S> (mesma cidade)
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

import qumodes.hamiltonian as H

__all__ = ["Moments", "NodeTable", "operator_stack", "state_moments",
           "moments_from_states", "energy_from_moments", "GAP_PIECES"]

GAP_PIECES = ("P1", "P2", "P3a", "P3b", "P4", "P5")


class Moments:
    """Momentos por cidade: primeira dimensão = cidades (N)."""
    __slots__ = ("a", "b", "t", "hd", "nph")

    def __init__(self, a, b, t, hd, nph):
        self.a, self.b, self.t, self.hd, self.nph = a, b, t, hd, nph


def operator_stack(ops, M: int, R: int):
    """(A, H_disc, n_op): A tem forma (X, d, d), X = M*R, índice x = v*R + r."""
    A = np.stack([np.asarray(ops.A[v, r]) for v in range(M) for r in range(R)])
    A = A.astype(np.complex128)
    A = 0.5 * (A + np.conj(np.transpose(A, (0, 2, 1))))
    return (A, np.asarray(ops.H_disc, dtype=np.complex128),
            np.asarray(ops.n_op, dtype=np.complex128))


def state_moments(phi, A, Hd, nop):
    """Momentos de UM estado de um modo."""
    phi = np.asarray(phi, dtype=np.complex128).reshape(-1)
    phi = phi / np.linalg.norm(phi)
    Av = A @ phi                                    # (X, d): A_x |phi>
    a = np.real(Av @ phi.conj())                    # <phi|A_x|phi>
    b = Av.conj() @ Av.T                            # <A_x phi | A_y phi>
    AAv = np.einsum("yij,zj->yzi", A, Av)           # A_y A_z |phi>
    t = np.einsum("xi,yzi->xyz", Av.conj(), AAv)    # <phi|A_x A_y A_z|phi>
    hd = float(np.real(phi.conj() @ (Hd @ phi)))
    nph = float(np.real(phi.conj() @ (nop @ phi)))
    return a, b, t, hd, nph


def moments_from_states(phis: Sequence[np.ndarray], A, Hd, nop) -> Moments:
    """Momentos de um estado-produto qualquer (um vetor por cidade)."""
    cols = [state_moments(p, A, Hd, nop) for p in phis]
    return Moments(np.stack([c[0] for c in cols]), np.stack([c[1] for c in cols]),
                   np.stack([c[2] for c in cols]),
                   np.array([c[3] for c in cols]), np.array([c[4] for c in cols]))


class NodeTable:
    """Momentos dos estados de nó (v, r) para um dado ops. Reutilizável: uma
    configuração de nós é só uma seleção de linhas."""

    def __init__(self, ops, M: int, R: int):
        A, Hd, nop = operator_stack(ops, M, R)
        cols = []
        for x in range(M * R):
            v, r = divmod(x, R)
            cols.append(state_moments(H.node_state(ops, v, r), A, Hd, nop))
        self.a = np.stack([c[0] for c in cols])      # (X estados, X operadores)
        self.b = np.stack([c[1] for c in cols])
        self.t = np.stack([c[2] for c in cols])
        self.hd = np.array([c[3] for c in cols])
        self.nph = np.array([c[4] for c in cols])

    def take(self, slots: Sequence[int]) -> Moments:
        s = list(slots)
        return Moments(self.a[s], self.b[s], self.t[s], self.hd[s], self.nph[s])


def energy_from_moments(mom: Moments, inst, par,
                        gap_variant: str = "padrao") -> Dict[str, object]:
    """Mesmas chaves de energy_components (dist, col, gap, vehicle, capacity,
    disc, encoding, mean_photons, exc, total) mais 'gap_pieces' (6 peças).

    gap_variant
      "padrao"   sanduíche B n_r B, com B contando TODAS as cidades (atual)
      "distinta" para a verificação da cidade i, B_i conta só as OUTRAS cidades:
                 sum_i A_i^{(r)} (c - sum_{q in Q} sum_{j != i} A_j^{(q)})^2.
                 Os fatores agem em modos diferentes e COMUTAM: não há
                 ordenamento nem sanduíche. Mesmo conjunto de zeros e também PSD.
                 Só existe aqui, para teste: NÃO está em energy_components.
                 Em produto: <A_i> [ (c - s_-i)^2 + sum_{j != i} Var_j(S) ].
    """
    if gap_variant not in ("padrao", "distinta"):
        raise ValueError(f"gap_variant inválido: {gap_variant!r}")
    if par.lambda_cap > 0:
        raise NotImplementedError("lambda_cap > 0 não é suportado por esta fatoração.")
    N, M, R = inst.N, inst.M, inst.R
    a, b, t = mom.a, mom.b, mom.t
    D = np.asarray(inst.D, dtype=np.float64)
    d0, dback, Dcc = D[0, 1:], D[1:, 0], np.asarray(inst.Dcc, dtype=np.float64)

    def idx(v, r):
        return v * R + r

    # --- distância ----------------------------------------------------------
    h_dist = 0.0
    for v in range(M):
        h_dist += float(d0 @ a[:, idx(v, 0)])                    # depósito -> primeira
        for r in range(R - 1):
            xr, xn = idx(v, r), idx(v, r + 1)
            h_dist += float(a[:, xr] @ Dcc @ a[:, xn])           # arcos internos
            tot_next = float(a[:, xn].sum())
            for i in range(N):                                   # {A, I - n}/2
                outros = tot_next - a[i, xn]
                h_dist += dback[i] * (a[i, xr] * (1.0 - outros) - b[i, xr, xn].real)
        h_dist += float(dback @ a[:, idx(v, R - 1)])             # última -> depósito

    # --- colisão --------------------------------------------------------------
    h_col = 0.0
    for v in range(M):
        for r in range(R):
            col = a[:, idx(v, r)]
            h_col += 0.5 * (float(col.sum()) ** 2 - float((col ** 2).sum()))

    # --- lacuna: <B n_r B>, B = c I - sum_Q n_q --------------------------------
    pieces = np.zeros(6)
    por_vr = {}                      # contribuição do termo de lacuna por (v, r)
    for v in range(M):
        for r in range(1, R):
            c, qs = H.gap_operator_spec(par.gap_form, r)
            xr = idx(v, r)
            xq = [idx(v, q) for q in qs]
            ai = a[:, xr]                                         # <A_r>, por cidade
            sj = a[:, xq].sum(axis=1)                             # s_j = <S_j>
            u = b[:, xq, xr].sum(axis=1)                          # <S_i A_i>
            m2 = b[:, xq][:, :, xq].sum(axis=(1, 2)).real         # <S_j^2>
            w = t[:, xq, xr, :][:, :, xq].sum(axis=(1, 2)).real   # <S_i A_i S_i>
            s_menos = sj.sum() - sj                               # sum_{j != i} s_j
            var_outros = (m2.sum() - m2) - ((sj ** 2).sum() - sj ** 2)
            ru = u.real
            if gap_variant == "distinta":
                pv = np.array([
                    float((c * c * ai).sum()),                    # P1
                    float((-2.0 * c * ai * s_menos).sum()),       # P2
                    float((ai * s_menos ** 2).sum()),             # P3a
                    float((ai * var_outros).sum()),               # P3b
                    0.0, 0.0,                                     # P4, P5 não existem
                ])
                pieces += pv
                por_vr[(v, r)] = float(pv.sum())
                continue
            pv = np.array([
                float((c * c * ai).sum()),                        # P1
                float((-2.0 * c * (ru + ai * s_menos)).sum()),    # P2
                float((ai * s_menos ** 2).sum()),                 # P3a
                float((ai * var_outros).sum()),                   # P3b
                float((2.0 * s_menos * ru).sum()),                # P4
                float(w.sum()),                                   # P5
            ])
            pieces += pv
            por_vr[(v, r)] = float(pv.sum())
    h_gap = float(pieces.sum())

    # --- veículos obrigatórios ----------------------------------------------------
    h_veh = 0.0
    if H.requires_all_vehicles(par, None):
        for v in range(M):
            x0 = idx(v, 0)
            asum = float(a[:, x0].sum())
            sq = float((a[:, x0] ** 2).sum())
            m2 = float(b[:, x0, x0].real.sum())
            h_veh += (m2 + asum * asum - sq) - 2.0 * asum + 1.0

    h_disc = float(mom.hd.sum())
    photons = float(mom.nph.sum())

    out = {"dist": par.lambda_dist * h_dist,
           "col": par.lambda_col * h_col,
           "gap": par.lambda_gap * h_gap,
           "vehicle": par.lambda_vehicle * h_veh,
           "capacity": 0.0,
           "disc": par.lambda_disc * h_disc}
    encoding = (out["dist"] + out["col"] + out["gap"] + out["vehicle"]
                + out["capacity"] + out["disc"])
    out["encoding"] = encoding
    out["mean_photons"] = photons
    out["exc"] = par.lambda_exc * photons
    out["total"] = encoding + out["exc"]
    out["gap_pieces"] = pieces
    out["gap_por_vr"] = por_vr          # {(v, r): contribuição sem peso lambda_gap}
    return out