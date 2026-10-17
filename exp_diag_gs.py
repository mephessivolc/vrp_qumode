"""
exp_diag_gs.py — por que o fundamental decodifica para uma rota subótima?

Contexto
--------
Em cutoff_minimo_trace.csv, C=4 V=1 decodifica para custo 37 (C*=32) de d=5 a
d=10, com massa ótima estacionada em ~0,12; C=4 V=2 decodifica para 52 (C*=41)
de d=5 a d=10, com massa ótima CAINDO com d. Isso não parece "falta de um pouco
de cutoff". Este script separa as hipóteses.

O que ele faz
-------------
Para cada variante de parâmetros e cada cutoff, calcula o fundamental
(denso até dense_limit, esparso acima) e imprime:

  E0         energia do fundamental
  dE_chk     E_total(energy_components) - E0. Deve ser ~1e-6 ou menos: é uma
             conferência INDEPENDENTE entre o operador esparso/denso e o
             avaliador de energia por termos.
  ok, custo  se o MAP decodifica rota ótima e qual o custo decodificado
  massa      massa de probabilidade do POVM sobre rotas ótimas
  dist..disc contribuição de cada termo (já ponderada) para a energia
  nbar       fótons médios por modo
  top        maior população, entre os modos, no quarto superior de Fock

Variantes
---------
  pen = base      parâmetros de exp_cutoff_minimo (lambda_col=lambda_gap=30, ...)
  pen = sugerida  H.suggest_penalties (limiares suficientes para o mínimo NA GRADE)
  lam_exc         confinamento: 0 (padrão do experimento) e 0.005 (faixa documentada
                  em HamiltonianParams, 0.001 a 0.01)

Como ler
--------
  H1  penalidade fraca:   col+gap+veh NÃO desprezíveis em 'base' e que caem em
                          'sugerida'; se a rota decodificada passa a ser ótima,
                          o d_gs medido com lambda=30 media a penalidade, não o cutoff.
  H2  corte de Fock:      top alto (> ~0,05) com lam_exc=0 e queda de top com
                          lam_exc=0.005; E0 continua caindo com d sem estabilizar.
  H3  viés estrutural:    nada muda com as variantes, col+gap+veh ~ 0 e disc
                          domina: o erro de codificação depende da rota/nó e supera
                          a diferença de custo entre rotas (5 a 11 unidades).
Reprodutibilidade: (pen=base, lam_exc=0) deve reproduzir E0 do trace
(C=4 V=1: d=5 105.32484, d=6 102.27530, d=8 100.96005, d=10 98.76585).

Threads
-------
Este script fixa OMP/OPENBLAS/MKL_NUM_THREADS=1 se não estiverem definidos. No
trace há um salto em degrau de 50 a 170 vezes no tempo ao cruzar dim ~ 4e3 a
8e3, em todas as séries, que a contagem de contrações não explica. Compare o
tempo de (base, 0, d=10) com os 487 s do trace. Para repetir com threads
padrão: defina OMP_NUM_THREADS antes de rodar.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import csv
import dataclasses
import time

import numpy as np

import qumodes.hamiltonian as H
import qumodes.hamiltonian_sparse as HS
from exp_cutoff_minimo import (HBAR, build_instance, canon, classical_optima,
                               optimal_mass, params_for)

FIELDS = ["pen", "lam_exc", "d", "dim", "method", "E0", "dE_chk", "ok",
          "decoded_cost", "c_star", "massa_otima", "t_dist", "t_col", "t_gap",
          "t_veh", "t_disc", "t_exc", "nbar_medio", "top_peso_max", "gap",
          "residual", "seconds"]


def make_params(inst, V, pen, lam_exc):
    base = params_for(V)
    if pen == "base":
        p = base
    elif pen == "sugerida":
        p = H.suggest_penalties(inst, require_all_vehicles=(V > 1),
                                lambda_disc=base.lambda_disc)
    else:
        raise ValueError(f"pen desconhecido: {pen}")
    return dataclasses.replace(p, lambda_exc=lam_exc)


def run(C=4, V=1, cutoffs=(5, 6, 8, 10), penalties=("base", "sugerida"),
        lam_excs=(0.0, 0.005), dense_limit=700, max_dim=60_000,
        out="diag_gs.csv", seed=42):
    inst = build_instance(C, V, seed)
    cstar, optset = classical_optima(inst, V > 1)
    print(f"C={C} V={V}  R={inst.R}  C*={cstar}  rotas ótimas={len(optset)}")
    for pen in penalties:
        p = make_params(inst, V, pen, 0.0)
        print(f"  pen={pen}: lambda_col={p.lambda_col:.1f} lambda_gap={p.lambda_gap:.1f} "
              f"lambda_vehicle={p.lambda_vehicle:.1f} lambda_disc={p.lambda_disc:.1f}")
    print()
    print(f"{'pen':>9}{'lexc':>7}{'d':>4}{'met':>7}{'E0':>10}{'dE_chk':>10}"
          f"{'ok':>3}{'custo':>7}{'massa':>8}{'dist':>9}{'col':>8}{'gap':>8}"
          f"{'veh':>8}{'disc':>9}{'exc':>7}{'nbar':>7}{'top':>7}{'seg':>8}")

    rows = []
    for pen in penalties:
        for lam_exc in lam_excs:
            par = make_params(inst, V, pen, lam_exc)
            for d in cutoffs:
                dim = d ** C
                if dim > max_dim:
                    continue
                ops = H.build_modular_operators(d, inst.M, inst.R, HBAR)
                method = "dense" if dim <= dense_limit else "sparse"
                t0 = time.time()
                if method == "dense":
                    gs = H.ground_state(inst, ops, par, max_dim=dense_limit)
                else:
                    gs = HS.ground_state_sparse(inst, ops, par)
                secs = time.time() - t0

                psi = gs["psi"]
                ec = H.energy_components(psi, inst, ops, par)

                pop = np.abs(psi) ** 2
                pop = pop / pop.sum()
                q = max(2, d // 4)
                topw, nbar = [], []
                for i in range(C):
                    axes = tuple(j for j in range(C) if j != i)
                    marg = pop.sum(axis=axes)
                    topw.append(float(marg[-q:].sum()))
                    nbar.append(float((np.arange(d) * marg).sum()))

                row = dict(
                    pen=pen, lam_exc=lam_exc, d=d, dim=dim, method=method,
                    E0=gs["energy"],
                    dE_chk=float(ec["total"] - gs["energy"]),
                    ok=int(canon(gs["map"]["routes"]) in optset),
                    decoded_cost=gs["map"]["cost"], c_star=cstar,
                    massa_otima=optimal_mass(gs["distribution"], inst, optset),
                    t_dist=float(ec["dist"]), t_col=float(ec["col"]),
                    t_gap=float(ec["gap"]), t_veh=float(ec["vehicle"]),
                    t_disc=float(ec["disc"]), t_exc=float(ec["exc"]),
                    nbar_medio=float(np.mean(nbar)), top_peso_max=max(topw),
                    gap=gs["gap"], residual=gs.get("residual", ""),
                    seconds=round(secs, 1))
                rows.append(row)
                print(f"{pen:>9}{lam_exc:>7.3f}{d:>4}{method:>7}{row['E0']:>10.4f}"
                      f"{row['dE_chk']:>10.1e}{row['ok']:>3}"
                      f"{row['decoded_cost']:>7.1f}{row['massa_otima']:>8.3f}"
                      f"{row['t_dist']:>9.2f}{row['t_col']:>8.2f}{row['t_gap']:>8.2f}"
                      f"{row['t_veh']:>8.2f}{row['t_disc']:>9.2f}{row['t_exc']:>7.3f}"
                      f"{row['nbar_medio']:>7.2f}{row['top_peso_max']:>7.3f}"
                      f"{row['seconds']:>8.1f}")
                with open(out, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=FIELDS)
                    w.writeheader()
                    w.writerows(rows)
    print(f"\nsalvo em {out}")
    return rows


if __name__ == "__main__":
    run(C=4, V=1, cutoffs=(5, 6, 8, 10))