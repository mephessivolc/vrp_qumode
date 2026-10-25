"""
exp_scan_gap.py — o viés de PERFIL encolhe se o peso do termo de lacuna diminuir?

Contexto
--------
exp_perfil_rotas.py mostrou que, em C=4 V=2, o fundamental sempre decodifica o
melhor custo do perfil (2,2), mesmo quando o ótimo está em (1,3) (5 de 6
instâncias). A diferença de overhead entre perfis vem do termo de lacuna
(gap): 79% a 92% da diferença. Se esse overhead escala com lambda_gap, reduzir
o peso deve fazer o ótimo voltar a vencer quando a diferença de custo entre os
perfis (2 a 14) superar a diferença de overhead.

O que faz
---------
Para cada semente, cada lambda_gap e cada cutoff, calcula o fundamental e
imprime:

  custo_top   custo da rota de maior massa do POVM (todas as configurações)
  P(val)      probabilidade de a configuração ser VÁLIDA
  P(ot|val)   P(rota ótima | válida)
  razão       P(ot|val) / acaso   (acaso = n_ótimas / n_rotas_válidas)
  top_ok      a melhor rota válida é ótima?
  t_dist, t_gap   contribuição desses termos à energia

Como ler
--------
  Se P(ot|val) sobe e custo_top passa a C* nas sementes que falham (42, 1)
  quando lambda_gap diminui, o viés é do peso do termo de lacuna.
  Atenção ao outro lado: lambda_gap pequeno demais permite configurações com
  lacuna. Se P(val) cair muito ou custo_top ficar ABAIXO de C* (rotas sem
  arcos), o remédio custa validade.
  A semente 3 funciona hoje: serve de controle (deve continuar funcionando).

Uso
---
    python exp_scan_gap.py          # grava resultado_gap.txt e scan_gap.csv
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
from exp_cutoff_minimo import (HBAR, build_instance, chance_valid,
                               classical_optima, make_params, route_masses,
                               valid_stats)
from exp_perfil_rotas import saida

FIELDS = ["seed", "lambda_gap", "d", "dim", "c_star", "custo_top", "E0",
          "p_valid", "p_opt_cond", "chance", "razao", "top_ok", "t_dist",
          "t_gap", "seconds"]


def run(C=4, V=2, seeds=(42, 1, 3), cutoffs=(12, 16),
        gaps=(20.0, 40.0, 70.0, None), dense_limit=700, max_dim=70_000,
        out=None, pasta="diagnosticos"):
    from saidas import resolver           # todo resultado passa pelo PathManager
    out = resolver(out, pasta, "scan_gap.csv")
    print(f"C={C} V={V}  seeds={seeds}  cutoffs={cutoffs}  "
          f"lambda_gap={['sugerido' if g is None else g for g in gaps]}")
    print(f"{'sem':>4}{'lgap':>7}{'d':>4}{'C*':>5}{'custo_top':>10}{'E0':>10}"
          f"{'P(val)':>8}{'P(ot|val)':>10}{'acaso':>8}{'razao':>8}{'top_ok':>7}"
          f"{'t_dist':>9}{'t_gap':>8}{'seg':>8}")
    rows = []
    for seed in seeds:
        inst = build_instance(C, V, seed)
        cstar, optset = classical_optima(inst, V > 1)
        chance, _ = chance_valid(inst, optset, V > 1)
        base = make_params(inst, V, "sugerida", 0.0)
        for lg in gaps:
            lam = base.lambda_gap if lg is None else float(lg)
            par = dataclasses.replace(base, lambda_gap=lam)
            for d in cutoffs:
                dim = d ** C
                if dim > max_dim:
                    continue
                ops = H.build_modular_operators(d, inst.M, inst.R, HBAR)
                t0 = time.time()
                if dim <= dense_limit:
                    gs = H.ground_state(inst, ops, par, max_dim=dense_limit)
                else:
                    gs = HS.ground_state_sparse(inst, ops, par)
                secs = time.time() - t0

                dist = gs["distribution"]
                p_valid, p_opt_valid, top_ok = valid_stats(dist, inst, optset,
                                                           V > 1)
                cond = p_opt_valid / p_valid if p_valid > 0 else float("nan")
                masses = route_masses(dist, inst)
                top = max(masses, key=masses.get)
                custo_top = H.routes_cost(top, inst.D)
                ec = H.energy_components(gs["psi"], inst, ops, par)

                row = dict(seed=seed, lambda_gap=lam, d=d, dim=dim,
                           c_star=cstar, custo_top=custo_top, E0=gs["energy"],
                           p_valid=p_valid, p_opt_cond=cond, chance=chance,
                           razao=cond / chance if chance > 0 else float("nan"),
                           top_ok=int(top_ok), t_dist=float(ec["dist"]),
                           t_gap=float(ec["gap"]), seconds=round(secs, 1))
                rows.append(row)
                print(f"{seed:>4}{lam:>7.0f}{d:>4}{cstar:>5.0f}{custo_top:>10.1f}"
                      f"{row['E0']:>10.2f}{p_valid:>8.3f}{cond:>10.4f}"
                      f"{chance:>8.4f}{row['razao']:>8.2f}{row['top_ok']:>7}"
                      f"{row['t_dist']:>9.2f}{row['t_gap']:>8.2f}"
                      f"{row['seconds']:>8.1f}")
                with open(out, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=FIELDS)
                    w.writeheader()
                    w.writerows(rows)
    return rows


if __name__ == "__main__":
    with saida("resultado_gap.txt"):
        run()
