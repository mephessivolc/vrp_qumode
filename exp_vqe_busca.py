"""
exp_vqe_busca.py — o VQE encontra a solução viável e ÓTIMA (igual à força bruta)?

Critério de sucesso (o que importa)
-----------------------------------
Para cada instância pequena, a referência é a FORÇA BRUTA (classical_optima):
o custo ótimo C* e o conjunto de rotas ótimas. O VQE é bem-sucedido quando a
solução decodificada do seu estado final é

    VIÁVEL (cada cidade uma vez, cada veículo com <= R cidades, todos os
    veículos usados quando exigido)   E   com custo numericamente igual a C*.

O fundamental exato de H é APENAS apresentado como referência (coluna E0); a
energia do VQE não precisa chegar a ele. Observação importante: a energia de H
inclui o overhead de codificação (centenas de unidades em cutoff finito), então
ela NÃO é numericamente igual a C*. A igualdade numérica cobrada é entre o CUSTO
da rota decodificada e C*.

Como a solução é decodificada
-----------------------------
Estado final -> distribuição do POVM sobre as (M*R)^N configurações -> `shots`
disparos -> rota VÁLIDA de menor custo entre os sorteados (sf_bridge.extract_routes).
Para não depender de uma realização do sorteio, também se calcula, EXATAMENTE a
partir da distribuição, a probabilidade de acerto com s disparos:

    P_acerto(s) = 1 - (1 - p_otima)^s,    p_otima = P(configuração válida e ótima)

Convergência da solução (não só da energia)
-------------------------------------------
A cada `checkpoint_cada` iterações o estado é decodificado e se registra
P_acerto(1024). `it_acerto` é a primeira iteração com P_acerto(1024) >= 0,99.

Variação principal: cidades C, veículos V e camadas L (num_layers).

Tudo é lançado pelo exp_run.py. Custos de tempo: cada iteração do ADAM faz
2P + 1 avaliações do circuito, com P = (6C + 2(C-1)) * L.
"""
import csv
import os
import time

import numpy as np
import strawberryfields as sf

import qumodes.hamiltonian as H
import qumodes.hamiltonian_sparse as HS
import qumodes.sf_bridge as SB
from qumodes.ansatz import CircuitConfig
from qumodes.solver import ProblemInstance, VQESolver
from exp_cutoff_minimo import (HBAR, _is_valid, build_instance, canon,
                               classical_optima, make_params,
                               routes_from_slots)

SHOTS = (16, 64, 256, 1024)
LIMIAR_ACERTO = 0.99

CAMPOS = ["C", "V", "R", "seed", "theta_seed", "layers", "params", "cutoff",
          "maxiter", "lr", "c_star", "n_otimas", "E_vqe", "E0_ref", "custo_dec",
          "viavel", "ok_custo", "ok_rota", "p_valid", "p_otima", "p_ot_cond",
          "custo_medio_valido", "razao_custo_medio", "pacerto_16", "pacerto_64",
          "pacerto_256", "pacerto_1024", "it_acerto", "avaliacoes", "segundos"]


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------
def _p_acerto(p_otima, s):
    return 1.0 - (1.0 - p_otima) ** s


def _viavel(rotas, inst, req):
    """Cada cidade 1..N exatamente uma vez; cada rota com no máximo R cidades;
    com req, nenhum veículo vazio."""
    todas = sorted(c for r in rotas for c in r)
    if todas != list(range(1, inst.N + 1)):
        return False
    if any(len(r) > inst.R for r in rotas):
        return False
    if req and any(len(r) == 0 for r in rotas):
        return False
    return True


def _estado(solver, theta, cutoff):
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    return eng.run(solver.ansatz.build_program(theta)).state


def _estatisticas(ket, inst, ops, optset, req):
    """Estatísticas exatas da distribuição do POVM: validade, massa ótima válida
    e custo médio entre as configurações válidas."""
    psi = np.asarray(ket, dtype=np.complex128).reshape((ops.cutoff,) * inst.N)
    psi = psi / np.linalg.norm(psi)
    dist = np.asarray(H.assignment_distribution(psi, inst, ops), dtype=np.float64)
    flat = np.clip(dist.reshape(-1), 0.0, None)
    flat = flat / flat.sum()
    p_valid = p_otima = soma_custo = 0.0
    for f in np.flatnonzero(flat > 1e-12):
        slots = np.unravel_index(int(f), dist.shape)
        if not _is_valid(slots, inst, req):
            continue
        w = float(flat[f])
        rotas = routes_from_slots(slots, inst)
        p_valid += w
        soma_custo += w * H.routes_cost(rotas, inst.D)
        if canon(rotas) in optset:
            p_otima += w
    return dict(p_valid=p_valid, p_otima=p_otima,
                p_ot_cond=(p_otima / p_valid if p_valid > 0 else float("nan")),
                custo_medio_valido=(soma_custo / p_valid if p_valid > 0
                                    else float("nan")))


# ---------------------------------------------------------------------------
# uma execução
# ---------------------------------------------------------------------------
def vqe_busca_otima(C=3, V=2, seed=42, cutoff=8, layers=2, theta_seed=0,
                    maxiter=100, lr=0.05, lr_final=0.005, fd_eps=1e-2,
                    theta_scale=0.05, checkpoint_cada=10, referencia_gs=True,
                    shots_decodificacao=1024):
    inst = build_instance(C, V, seed)
    req = V > 1
    t0 = time.time()
    cstar, optset = classical_optima(inst, req)          # FORÇA BRUTA
    t_brute = time.time() - t0
    par = make_params(inst, V, "sugerida", 0.0, "soma")
    ops = H.build_modular_operators(cutoff, inst.M, inst.R, HBAR)

    e0 = float("nan")
    if referencia_gs:
        e0 = float(HS.ground_state_sparse(inst, ops, par)["energy"])

    pi = ProblemInstance(V=V, C=C, D=inst.D, demands=inst.demands, Q=inst.Q)
    solver = VQESolver(pi, CircuitConfig(C, layers), par, cutoff=cutoff)
    theta0 = solver.ansatz.generate_initial_theta(seed=theta_seed, scale=theta_scale)

    print("\n" + "=" * 100)
    print(f"VQE  C={C} V={V} R={inst.R} semente={seed} d={cutoff} camadas={layers} "
          f"P={len(theta0)} semente_theta={theta_seed}")
    print(f"força bruta: C*={cstar:g}  rotas ótimas={len(optset)}  "
          f"({t_brute:.3f} s)    referência E0 do fundamental = {e0:.3f}")
    print("=" * 100)
    print(f"{'iter':>5}{'energia':>12}{'P(válida)':>11}{'P(ótima)':>11}"
          f"{'P_ac@1024':>11}{'custo médio':>13}")

    historico = []
    it_acerto = {"v": None}

    def cb(t, theta, energia):
        if t % checkpoint_cada != 0:
            return
        est = _estatisticas(_estado(solver, theta, cutoff).ket(), inst, ops, optset, req)
        pa = _p_acerto(est["p_otima"], 1024)
        historico.append((t, energia, est["p_valid"], est["p_otima"], pa))
        if it_acerto["v"] is None and pa >= LIMIAR_ACERTO:
            it_acerto["v"] = t
        print(f"{t:>5}{energia:>12.3f}{est['p_valid']:>11.4f}{est['p_otima']:>11.5f}"
              f"{pa:>11.4f}{est['custo_medio_valido']:>13.2f}")

    t0 = time.time()
    m = solver.solve(method="ADAM", maxiter=maxiter, lr=lr, lr_final=lr_final,
                     fd_eps=fd_eps, initial_theta=theta0, callback=cb)
    segundos = time.time() - t0

    estado = _estado(solver, m.optimal_theta, cutoff)
    est = _estatisticas(estado.ket(), inst, ops, optset, req)
    dec = SB.extract_routes(estado, C, V, pi.D, pi.demands, pi.Q, cutoff, par,
                            shots=shots_decodificacao, seed=theta_seed)
    rotas = [dec] if V == 1 else [dec[v] for v in sorted(dec)]
    custo = H.routes_cost(rotas, inst.D)
    viavel = _viavel(rotas, inst, req)
    ok_custo = bool(viavel and abs(custo - cstar) < 1e-9)
    ok_rota = bool(viavel and canon(rotas) in optset)
    pac = {s: _p_acerto(est["p_otima"], s) for s in SHOTS}

    print(f"\nE_vqe={m.final_energy:.3f}   (E0 de referência {e0:.3f})")
    print(f"decodificado ({shots_decodificacao} disparos): rotas={rotas}  custo={custo:g}  "
          f"viável={'sim' if viavel else 'NÃO'}  custo == C*: "
          f"{'SIM' if ok_custo else 'NÃO'}   rota ótima: {'SIM' if ok_rota else 'NÃO'}")
    print("P(acerto) com s disparos: " + "  ".join(f"{s}: {pac[s]:.3f}" for s in SHOTS)
          + f"   | custo médio das válidas {est['custo_medio_valido']:.2f} "
          f"({est['custo_medio_valido'] / cstar:.2f} x C*)   | 1o acerto em it="
          f"{it_acerto['v']}   | {segundos:.0f} s")

    return dict(C=C, V=V, R=inst.R, seed=seed, theta_seed=theta_seed, layers=layers,
                params=len(theta0), cutoff=cutoff, maxiter=maxiter, lr=lr,
                c_star=cstar, n_otimas=len(optset), E_vqe=m.final_energy, E0_ref=e0,
                custo_dec=custo, viavel=int(viavel), ok_custo=int(ok_custo),
                ok_rota=int(ok_rota), p_valid=est["p_valid"], p_otima=est["p_otima"],
                p_ot_cond=est["p_ot_cond"],
                custo_medio_valido=est["custo_medio_valido"],
                razao_custo_medio=est["custo_medio_valido"] / cstar,
                pacerto_16=pac[16], pacerto_64=pac[64], pacerto_256=pac[256],
                pacerto_1024=pac[1024],
                it_acerto=-1 if it_acerto["v"] is None else it_acerto["v"],
                avaliacoes=m.circuit_evaluations, segundos=round(segundos, 1))


# ---------------------------------------------------------------------------
# varredura em cidades, veículos e camadas
# ---------------------------------------------------------------------------
def varredura_vqe(cidades=(3,), veiculos=(1, 2), camadas=(1, 2, 3),
                  cutoffs=None, seeds_instancia=(42,), seeds_theta=(0, 1),
                  maxiter=100, lr=0.05, lr_final=0.005,
                  csv_path="vqe_varredura.csv", **kw):
    cutoffs = cutoffs or {3: 8, 4: 10}
    linhas = []
    for C in cidades:
        for V in veiculos:
            if V >= C:                 # V=C é degenerado (uma única rota válida)
                continue
            for s in seeds_instancia:
                for L in camadas:
                    for ts in seeds_theta:
                        r = vqe_busca_otima(C=C, V=V, seed=s, cutoff=cutoffs[C],
                                            layers=L, theta_seed=ts, maxiter=maxiter,
                                            lr=lr, lr_final=lr_final, **kw)
                        linhas.append(r)
                        novo = not os.path.exists(csv_path)
                        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
                            w = csv.DictWriter(fh, fieldnames=CAMPOS)
                            if novo:
                                w.writeheader()
                            w.writerow(r)
    return linhas


def resumo_varredura(linhas):
    """Tabela final: uma por (C, V, semente da instância)."""
    print("\n" + "=" * 110)
    print("RESUMO: o VQE decodifica a solução viável e igual à força bruta?")
    print("  ok = custo decodificado == C* (viável)   P@s = probabilidade de ver a "
          "ótima em s disparos")
    print("=" * 110)
    chaves = []
    for r in linhas:
        k = (r["C"], r["V"], r["seed"])
        if k not in chaves:
            chaves.append(k)
    for (C, V, s) in chaves:
        sub = [r for r in linhas if (r["C"], r["V"], r["seed"]) == (C, V, s)]
        print(f"\nC={C} V={V} R={sub[0]['R']} semente={s}  C*={sub[0]['c_star']:g}  "
              f"d={sub[0]['cutoff']}")
        print(f"{'camadas':>8}{'θ-sem':>6}{'P':>5}{'E_vqe':>10}{'E0':>10}{'custo':>8}"
              f"{'ok':>5}{'P@16':>8}{'P@64':>8}{'P@256':>8}{'P@1024':>8}"
              f"{'it_ac':>7}{'t(s)':>8}")
        for r in sorted(sub, key=lambda x: (x["layers"], x["theta_seed"])):
            print(f"{r['layers']:>8}{r['theta_seed']:>6}{r['params']:>5}"
                  f"{r['E_vqe']:>10.2f}{r['E0_ref']:>10.2f}{r['custo_dec']:>8.0f}"
                  f"{'sim' if r['ok_custo'] else 'nao':>5}"
                  f"{r['pacerto_16']:>8.3f}{r['pacerto_64']:>8.3f}"
                  f"{r['pacerto_256']:>8.3f}{r['pacerto_1024']:>8.3f}"
                  f"{r['it_acerto']:>7}{r['segundos']:>8.0f}")