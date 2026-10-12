"""
exp_cutoff_minimo.py — qual o menor cutoff em que a formulação modular ainda
resolve o problema?

A pergunta
----------
A modular mantém N qumodes mas precisa de cutoff maior que a Etapa 1, porque
posição e veículo moram nas duas quadraturas e precisam ser RESOLVIDOS, não
apenas rotulados. Quanto maior? Se d_min crescer proporcionalmente a M·R, a
modular não tem vantagem de recursos sobre uma extensão direta da Etapa 1
(que usaria cutoff = M·R exatamente). Se crescer mais devagar, ela ganha — e
o ponto de cruzamento fica determinado.

Dois critérios, do mais fraco ao mais forte:

  d_tab   menor cutoff em que o ARGMIN DA TABELA DE NÓS decodifica para uma
          rota ótima clássica. Análogo do antigo brute_force_grid.

  d_gs    menor cutoff em que o FUNDAMENTAL da diagonalização exata decodifica
          para uma rota ótima. É o critério forte: o fundamental é o mínimo
          sobre TODOS os estados, então não há mínimo espúrio em lugar nenhum.

d_gs >= d_tab é o esperado. O que vai para a tabela comparativa é d_gs.

Nada aqui usa VQE. Tudo em numpy float64.

Uso
---
    python exp_cutoff_minimo.py                 # grade padrão
    python exp_cutoff_minimo.py --cities 3 4 5 --vehicles 1 2
    python exp_cutoff_minimo.py --max-dim 50000 --max-configs 500000

Saída: tabela no terminal e cutoff_minimo.csv. Mande os dois para análise.
"""
import argparse
import csv
import itertools
import time

import numpy as np

import qumodes.hamiltonian as H

HBAR = 2.0


# ---------------------------------------------------------------------------
# Instância e ótimo clássico (enumeração independente do Hamiltoniano)
# ---------------------------------------------------------------------------
def build_instance(C, V, seed=42, lo=1, hi=20):
    rng = np.random.default_rng(seed)
    D = np.zeros((C + 1, C + 1))
    for i in range(C + 1):
        for j in range(i + 1, C + 1):
            D[i, j] = D[j, i] = float(rng.integers(lo, hi + 1))
    demands = np.ones(C)
    caps = np.full(V, float(C))
    return H.instance_from_legacy(C, V, D, demands, caps, hbar=HBAR,
                                  require_all_vehicles=(V > 1))


def canon(routes):
    return tuple(sorted(tuple(r) for r in routes if r))


def classical_optima(inst, require_all):
    """Permutações das cidades + cortes entre veículos. Não usa decode()."""
    N, M, R, D = inst.N, inst.M, inst.R, inst.D
    best, opt = np.inf, set()
    for perm in itertools.permutations(range(1, N + 1)):
        for cuts in itertools.combinations(range(1, N), M - 1):
            idx = [0, *cuts, N]
            rts = [list(perm[idx[k]:idx[k + 1]]) for k in range(M)]
            if require_all and any(not r for r in rts):
                continue
            if any(len(r) > R for r in rts):
                continue
            c = H.routes_cost(rts, D)
            if c < best - 1e-9:
                best, opt = c, {canon(rts)}
            elif abs(c - best) <= 1e-9:
                opt.add(canon(rts))
    return float(best), opt


# ---------------------------------------------------------------------------
# Varredura
# ---------------------------------------------------------------------------
def params_for(V):
    return H.HamiltonianParams(
        lambda_dist=1.0, lambda_col=30.0, lambda_gap=30.0,
        lambda_vehicle=30.0 if V > 1 else 0.0,
        lambda_cap=0.0, lambda_disc=5.0, lambda_exc=0.0)


def sweep_case(C, V, cutoffs, max_dim, max_configs, verbose=True):
    inst = build_instance(C, V)
    req = (V > 1)
    cstar, optset = classical_optima(inst, req)
    par = params_for(V)
    MR = inst.M * inst.R
    n_cfg = MR ** C

    row = {"C": C, "V": V, "R": inst.R, "MR": MR, "n_configs": n_cfg,
           "c_star": cstar, "n_optimal": len(optset),
           "d_tab": None, "d_gs": None,
           "rho_at_dgs": None, "gap_at_dgs": None, "weyl_qtr_at_dgs": None,
           "E0_at_dgs": None, "map_p_at_dgs": None, "seconds": 0.0,
           "skipped": ""}

    if n_cfg > max_configs:
        row["skipped"] = f"configs {n_cfg:,} > max_configs"
        return row

    t0 = time.time()
    for d in cutoffs:
        dim = d ** C
        ops = H.build_modular_operators(d, inst.M, inst.R, HBAR)

        if row["d_tab"] is None:
            tab = H.node_energy_table(inst, ops, par, max_configs=max_configs)
            if canon(tab["routes"]) in optset:
                row["d_tab"] = d
                if verbose:
                    print(f"      d={d:<3} tabela OK  (E_min={tab['min_energy']:.4f})")

        if row["d_gs"] is None and dim <= max_dim:
            gs = H.ground_state(inst, ops, par, max_dim=max_dim)
            if canon(gs["map"]["routes"]) in optset:
                dist = gs["distribution"].reshape(-1)
                massa = 0.0
                for f in np.flatnonzero(dist > 1e-6):
                    onehot = np.zeros_like(dist)
                    onehot[f] = 1.0
                    r = H.decode_distribution(
                        onehot.reshape(gs["distribution"].shape), inst, top=1)[0]
                    if canon(r["routes"]) in optset:
                        massa += float(dist[f])
                unif = len(optset) / dist.size
                row.update(d_gs=d, rho_at_dgs=massa / unif if unif else None,
                           gap_at_dgs=gs["gap"], E0_at_dgs=gs["energy"],
                           map_p_at_dgs=gs["map"]["probability"],
                           weyl_qtr_at_dgs=ops.weyl_commutator(max(2, d // 4)))
                if verbose:
                    print(f"      d={d:<3} fundamental OK  rho={row['rho_at_dgs']:.2f} "
                          f"gap={gs['gap']:.4f}")

        if row["d_tab"] is not None and row["d_gs"] is not None:
            break
        if dim > max_dim and row["d_tab"] is not None:
            row["skipped"] = f"dim {dim:,} > max_dim antes de achar d_gs"
            break

    row["seconds"] = time.time() - t0
    return row


def main(
        cities=[3,4,5], 
        vehicles=[1,2,3], 
        cutoffs=[3,4,5,6,7,8,10,12,14,16,20],
        max_dim=40000,
        max_configs=300000,
        out="cutoff_minimo.csv"):

    rows = []
    for C in cities:
        for V in vehicles:
            if V > C:
                continue
            print(f"\n=== C={C} V={V} ===")
            r = sweep_case(C, V, cutoffs, max_dim, max_configs)
            if r["skipped"]:
                print(f"      pulado: {r['skipped']}")
            rows.append(r)

    print("\n" + "=" * 92)
    print("CUTOFF MÍNIMO — a modular resolve a partir de que d?")
    print("=" * 92)
    print(f"{'C':>3}{'V':>3}{'R':>3}{'M*R':>5}{'#ótim':>7}"
          f"{'d_tab':>7}{'d_gs':>6}{'d_gs/MR':>9}{'rho':>8}{'gap':>8}"
          f"{'weyl¼':>10}{'seg':>8}")
    for r in rows:
        f = lambda k, fmt: ("—" if r[k] is None else format(r[k], fmt))
        ratio = ("—" if r["d_gs"] is None else f"{r['d_gs']/r['MR']:.2f}")
        print(f"{r['C']:>3}{r['V']:>3}{r['R']:>3}{r['MR']:>5}{r['n_optimal']:>7}"
              f"{f('d_tab','>7') if r['d_tab'] else '      —'}"
              f"{f('d_gs','>6') if r['d_gs'] else '     —'}"
              f"{ratio:>9}{f('rho_at_dgs','>8.2f'):>8}{f('gap_at_dgs','>8.4f'):>8}"
              f"{f('weyl_qtr_at_dgs','>10.2e'):>10}{r['seconds']:>8.1f}")

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsalvo em {out}")
    print("\nA coluna d_gs/MR é a resposta: se ficar ~constante, a modular escala")
    print("como a extensão da Etapa 1 e o ganho vem de outro lugar; se CAIR com")
    print("M*R, a modular tem vantagem de recursos a partir de certo tamanho.")


if __name__ == "__main__":
    main(
        cities=[3,4],
        vehicles=[1,2,3]
    )