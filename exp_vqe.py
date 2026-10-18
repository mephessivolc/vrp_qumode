"""
exp_vqe.py — VQE contra o fundamental exato.

Tudo é lançado pelo exp_run.py. Este arquivo NÃO usa run.py, analysis.py nem
verify_encoding.py (ainda na API antiga do hamiltonian.py): chama o VQESolver
direto, com a ponte qumodes/sf_bridge.py.

verificar_ponte   integração SF -> energia, SEM VQE. Com theta = 0 o circuito é a
                  identidade e o estado final é o produto dos kets iniciais. A
                  energia obtida pelo caminho do solver (sf_bridge) deve coincidir
                  com a do avaliador fatorado (product_energy), que é independente
                  e já foi validado contra energy_components. Como os kets diferem
                  por modo, isso também confere a ORDEM dos modos entre SF e H.

vqe_vs_exato      roda o VQE (ADAM, gradiente por diferença finita) e compara com
                  o fundamental exato no MESMO cutoff:
                    E_vqe e E0, fidelidade |<GS|VQE>|^2,
                    P(válida), P(ótima|válida) e P(ótima) por disparo bruto,
                    disparos esperados até ver a rota ótima (1 / P(ótima)).
                  As colunas do fundamental são o alvo: um VQE não passa dele.

Custo (estimativa, não medida): cada iteração do ADAM faz 2P + 1 avaliações do
circuito, com P = (6N + 2(N-1)) * camadas parâmetros. Para N=3 e 2 camadas, P=44
e são 89 avaliações por iteração. Cada avaliação é uma simulação Fock mais um
energy_components, ambos em d^N dimensões: use d pequeno (6 a 8) para validar o
pipeline antes de subir.
"""
import csv
import os
import time

import numpy as np
import strawberryfields as sf

import qumodes.hamiltonian as H
import qumodes.hamiltonian_sparse as HS
import qumodes.product_energy as PE
import qumodes.sf_bridge as SB
from qumodes.ansatz import CircuitConfig, ContinuousVariableAnsatz
from qumodes.solver import ProblemInstance, VQESolver
from exp_cutoff_minimo import (HBAR, build_instance, classical_optima,
                               make_params, optimal_mass, route_masses,
                               valid_stats)


def _kets_de_nos(ops, inst, C):
    """Um ket de nó por cidade, em nós distintos (0, 1, 2, ...)."""
    X = inst.M * inst.R
    slots = tuple(i % X for i in range(C))
    return slots, [H.node_state(ops, s // inst.R, s % inst.R) for s in slots]


# ---------------------------------------------------------------------------
# 1. integração SF -> energia
# ---------------------------------------------------------------------------
def verificar_ponte(casos=((3, 1, 6), (3, 2, 6)), seed=42, tol=1e-9):
    ok_all = True
    chaves = ("dist", "col", "gap", "vehicle", "disc", "exc", "total")
    print("=" * 96)
    print("VERIFICAÇÃO DA PONTE  SF (backend Fock) -> energia, com theta = 0")
    print("=" * 96)
    for (C, V, d) in casos:
        inst = build_instance(C, V, seed)
        par = make_params(inst, V, "sugerida", 0.05, "soma")     # inclui confinamento
        ops = H.build_modular_operators(d, inst.M, inst.R, HBAR)
        slots, kets = _kets_de_nos(ops, inst, C)
        pi = ProblemInstance(V=V, C=C, D=inst.D, demands=inst.demands, Q=inst.Q)

        ans = ContinuousVariableAnsatz(CircuitConfig(C, 1), initial_ket=kets,
                                       use_kerr=True)
        prog = ans.build_program(np.zeros(ans.total_params))
        eng = sf.Engine("fock", backend_options={"cutoff_dim": d})
        estado = eng.run(prog).state

        got = SB.evaluate_sf_state(estado, V, C, pi.D, pi.demands, pi.Q, d, par)
        A, Hd, nop = PE.operator_stack(ops, inst.M, inst.R)
        ref = PE.energy_from_moments(PE.moments_from_states(kets, A, Hd, nop), inst, par)
        pior = max(abs(got[k] - float(ref[k])) / (1.0 + abs(float(ref[k])))
                   for k in chaves)
        ok = (pior < tol) and abs(got["norm"] - 1.0) < 1e-6
        ok_all &= ok
        print(f"C={C} V={V} d={d} R={inst.R} nós={slots}  "
              f"erro_rel_max={pior:.2e}  norma={got['norm']:.9f}  "
              f"{'OK' if ok else 'FALHOU'}")
        print(f"      E_ponte={got['total']:.6f}   E_fatorado={float(ref['total']):.6f}")
    print("\n" + ("PONTE OK" if ok_all else "FALHOU — não rode o VQE"))
    return ok_all


# ---------------------------------------------------------------------------
# 2. VQE contra o fundamental exato
# ---------------------------------------------------------------------------
def _estatisticas(psi, inst, ops, par, optset, req):
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    psi_t = psi.reshape((ops.cutoff,) * inst.N)
    dist = H.assignment_distribution(psi_t, inst, ops)
    p_valid, p_opt_valid, top_ok = valid_stats(dist, inst, optset, req)
    ec = H.energy_components(psi_t, inst, ops, par)
    return dict(E=float(ec["total"]),
                nbar=float(ec["mean_photons"]),
                p_valid=p_valid,
                p_opt_cond=(p_opt_valid / p_valid if p_valid > 0 else float("nan")),
                p_opt_raw=p_opt_valid,
                disparos=(1.0 / p_opt_valid if p_opt_valid > 0 else float("inf")),
                massa_otima=optimal_mass(dist, inst, optset),
                top_ok=int(top_ok))


def vqe_vs_exato(C=3, V=2, seed=42, cutoff=8, layers=2, maxiter=100, lr=0.05,
                 lr_final=0.005, fd_eps=1e-2, init="vacuo", theta_scale=0.05,
                 csv_path="vqe_resultados.csv", tag=None):
    """init: 'vacuo' (o ansatz parte do vácuo) ou 'nos' (um ket de nó por cidade,
    em nós distintos: warm start que NÃO conhece qual configuração é a ótima)."""
    inst = build_instance(C, V, seed)
    cstar, optset = classical_optima(inst, V > 1)
    par = make_params(inst, V, "sugerida", 0.0, "soma")
    ops = H.build_modular_operators(cutoff, inst.M, inst.R, HBAR)
    req = V > 1
    tag = tag or f"C{C}V{V}s{seed}d{cutoff}_{init}"

    print("=" * 100)
    print(f"VQE  C={C} V={V} semente={seed} d={cutoff} R={inst.R} camadas={layers} "
          f"init={init}  C*={cstar:g}  rotas ótimas={len(optset)}")
    print("=" * 100)

    # --- alvo: fundamental exato no mesmo cutoff -----------------------------
    t0 = time.time()
    gs = HS.ground_state_sparse(inst, ops, par)
    est_gs = _estatisticas(gs["psi"], inst, ops, par, optset, req)
    print(f"fundamental exato: E0={gs['energy']:.4f}  ({time.time() - t0:.1f} s)")

    # --- VQE ---------------------------------------------------------------------
    kets = None
    if init == "nos":
        _, kets = _kets_de_nos(ops, inst, C)
    pi = ProblemInstance(V=V, C=C, D=inst.D, demands=inst.demands, Q=inst.Q)
    solver = VQESolver(pi, CircuitConfig(C, layers), par, cutoff=cutoff,
                       initial_ket=kets)
    theta0 = solver.ansatz.generate_initial_theta(seed=seed, scale=theta_scale)
    print(f"parâmetros P={len(theta0)}  avaliações por iteração={2 * len(theta0) + 1}  "
          f"iterações={maxiter}")
    t0 = time.time()
    m = solver.solve(method="ADAM", maxiter=maxiter, lr=lr, lr_final=lr_final,
                     fd_eps=fd_eps, initial_theta=theta0)
    secs = time.time() - t0

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    ket = np.asarray(eng.run(solver.ansatz.build_program(m.optimal_theta)).state.ket())
    est_vq = _estatisticas(ket, inst, ops, par, optset, req)
    ket_n = ket.reshape(-1) / np.linalg.norm(ket)
    gs_n = np.asarray(gs["psi"]).reshape(-1)
    gs_n = gs_n / np.linalg.norm(gs_n)
    fid = float(abs(np.vdot(gs_n, ket_n)) ** 2)

    # --- relatório ---------------------------------------------------------------
    print(f"\nVQE terminou em {secs:.1f} s ({m.circuit_evaluations} avaliações)")
    print(f"{'':>22}{'E':>12}{'nbar':>8}{'P(válida)':>11}{'P(ót|vál)':>11}"
          f"{'P(ótima)':>11}{'disparos':>11}")
    for nome, e in (("fundamental exato", est_gs), ("VQE", est_vq)):
        print(f"{nome:>22}{e['E']:>12.4f}{e['nbar']:>8.2f}{e['p_valid']:>11.4f}"
              f"{e['p_opt_cond']:>11.4f}{e['p_opt_raw']:>11.5f}{e['disparos']:>11.1f}")
    print(f"\nE_vqe - E0 = {est_vq['E'] - est_gs['E']:.4f}   "
          f"fidelidade |<GS|VQE>|^2 = {fid:.4f}   rota decodificada: {m.best_routes}")

    with open(f"vqe_iterados_{tag}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["iteracao", "energia"])
        for i, e in enumerate(m.iterate_history):
            w.writerow([i, e])

    linha = dict(tag=tag, C=C, V=V, seed=seed, cutoff=cutoff, layers=layers,
                 init=init, maxiter=maxiter, lr=lr, c_star=cstar,
                 E0=gs["energy"], E_vqe=est_vq["E"],
                 dE=est_vq["E"] - est_gs["E"], fidelidade=fid,
                 p_valid_gs=est_gs["p_valid"], p_valid_vqe=est_vq["p_valid"],
                 p_opt_cond_gs=est_gs["p_opt_cond"], p_opt_cond_vqe=est_vq["p_opt_cond"],
                 disparos_gs=est_gs["disparos"], disparos_vqe=est_vq["disparos"],
                 segundos=round(secs, 1))
    novo = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linha))
        if novo:
            w.writeheader()
        w.writerow(linha)
    return linha