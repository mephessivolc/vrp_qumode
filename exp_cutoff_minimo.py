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

d_gs >= d_tab é o esperado (não é garantido). O que vai para a tabela
comparativa é d_gs.

Mudanças desta versão
---------------------
1. Caminho esparso (matrix-free) para o fundamental quando cutoff^C passa de
   `dense_limit` (qumodes/hamiltonian_sparse.py). A diagonalização densa fica
   só para dimensões pequenas, porque dense_hamiltonian monta N·M·R matrizes
   dim x dim. `max_dim` agora é o limite do caminho ESPARSO.
2. Censura explícita: se d_gs não foi achado, o CSV registra d_gs_lb (maior
   cutoff em que o fundamental foi calculado e FALHOU) e status_gs:
       MEDIDO       d_gs encontrado
       CENSURADO    d_gs > d_gs_lb (parou por max_dim ou fim da grade)
       NAO_TESTADO  nenhum fundamental foi calculado
3. Rastro completo em cutoff_minimo_trace.csv: uma linha por (caso, cutoff,
   critério), com custo decodificado, energia, gap e massa ótima. É o que
   permite investigar a não-monotonicidade de d_tab em V.
4. O CSV é regravado a cada caso concluído (uma falha no C=5 não perde o C=3,4).
5. table=False pula a tabela de nós (a parte mais cara em C>=4) quando só
   d_gs interessa.

Nada aqui usa VQE. Tudo em numpy float64.

Uso
---
    python exp_cutoff_minimo.py
    # ou, a partir de exp_run.py: cutoff_main(cities=[3,4,5], ...)

Saída: tabela no terminal, cutoff_minimo.csv e cutoff_minimo_trace.csv.
"""
import csv
import itertools
import time

import numpy as np

import qumodes.hamiltonian as H
import qumodes.hamiltonian_sparse as HS

HBAR = 2.0

TRACE_FIELDS = ["C", "V", "d", "dim", "kind", "ok", "decoded_cost", "c_star",
                "energy", "gap", "massa_otima", "rho", "method", "seconds"]


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


def routes_from_slots(slots, inst):
    """Mesma decodificação de H.decode_distribution, para um índice só."""
    R = inst.R
    v = [int(s) // R for s in slots]
    r = [int(s) % R for s in slots]
    routes = []
    for veh in range(inst.M):
        cities = [i for i in range(inst.N) if v[i] == veh]
        cities.sort(key=lambda i: (r[i], i))
        routes.append([i + 1 for i in cities])
    return routes


def optimal_mass(dist, inst, optset):
    """Massa de probabilidade do POVM sobre configurações que decodificam
    para uma rota ótima (sem ordenar a distribuição inteira a cada índice)."""
    flat = dist.reshape(-1)
    total = 0.0
    for f in np.flatnonzero(flat > 1e-12):
        slots = np.unravel_index(int(f), dist.shape)
        if canon(routes_from_slots(slots, inst)) in optset:
            total += float(flat[f])
    return total


# ---------------------------------------------------------------------------
# Varredura
# ---------------------------------------------------------------------------
def params_for(V):
    return H.HamiltonianParams(
        lambda_dist=1.0, lambda_col=30.0, lambda_gap=30.0,
        lambda_vehicle=30.0 if V > 1 else 0.0,
        lambda_cap=0.0, lambda_disc=5.0, lambda_exc=0.0)


def _tr(C, V, d, kind, ok, cost, cstar, energy, gap, massa, rho, method, secs):
    return dict(C=C, V=V, d=d, dim=d ** C, kind=kind, ok=int(ok),
                decoded_cost=cost, c_star=cstar, energy=energy, gap=gap,
                massa_otima=massa, rho=rho, method=method,
                seconds=round(secs, 3))


def sweep_case(C, V, cutoffs, max_dim, max_configs, dense_limit=2000,
               table=True, verbose=True, trace=None):
    trace = trace if trace is not None else []
    inst = build_instance(C, V)
    cstar, optset = classical_optima(inst, V > 1)
    par = params_for(V)
    MR = inst.M * inst.R
    n_cfg = MR ** C

    row = {"C": C, "V": V, "R": inst.R, "MR": MR, "n_configs": n_cfg,
           "c_star": cstar, "n_optimal": len(optset),
           "d_tab": None, "d_gs": None, "d_gs_lb": None,
           "status_gs": "NAO_TESTADO",
           "rho_at_dgs": None, "massa_at_dgs": None, "gap_at_dgs": None,
           "weyl_qtr_at_dgs": None, "E0_at_dgs": None, "map_p_at_dgs": None,
           "method_at_dgs": "", "seconds": 0.0, "skipped": ""}

    if n_cfg > max_configs:
        row["skipped"] = f"configs {n_cfg:,} > max_configs"
        return row

    t0 = time.time()
    for d in cutoffs:
        dim = d ** C
        ops = H.build_modular_operators(d, inst.M, inst.R, HBAR)

        try:
            # --- critério fraco: tabela de nós -------------------------------
            if table and row["d_tab"] is None:
                ts = time.time()
                tab = H.node_energy_table(inst, ops, par, max_configs=max_configs)
                ok = canon(tab["routes"]) in optset
                trace.append(_tr(C, V, d, "tabela", ok, tab["cost"], cstar,
                                 tab["min_energy"], "", "", "", "nos",
                                 time.time() - ts))
                if ok:
                    row["d_tab"] = d
                    if verbose:
                        print(f"      d={d:<3} tabela OK  "
                              f"(E_min={tab['min_energy']:.4f})")
                elif verbose:
                    print(f"      d={d:<3} tabela falhou  "
                          f"(custo decodificado {tab['cost']:.1f} vs C*={cstar:.1f})")

            # --- critério forte: fundamental ----------------------------------
            if row["d_gs"] is None and dim <= max_dim:
                ts = time.time()
                method = "dense" if dim <= dense_limit else "sparse"
                if method == "dense":
                    gs = H.ground_state(inst, ops, par, max_dim=dense_limit)
                else:
                    gs = HS.ground_state_sparse(inst, ops, par)
                dist = gs["distribution"]
                massa = optimal_mass(dist, inst, optset)
                unif = len(optset) / dist.size
                rho = massa / unif if unif else float("nan")
                ok = canon(gs["map"]["routes"]) in optset
                trace.append(_tr(C, V, d, "fundamental", ok, gs["map"]["cost"],
                                 cstar, gs["energy"], gs["gap"], massa, rho,
                                 method, time.time() - ts))
                if ok:
                    row.update(d_gs=d, rho_at_dgs=rho, massa_at_dgs=massa,
                               gap_at_dgs=gs["gap"], E0_at_dgs=gs["energy"],
                               map_p_at_dgs=gs["map"]["probability"],
                               weyl_qtr_at_dgs=ops.weyl_commutator(max(2, d // 4)),
                               method_at_dgs=method)
                    if verbose:
                        print(f"      d={d:<3} fundamental OK [{method}]  "
                              f"massa={massa:.3f} rho={rho:.2f} gap={gs['gap']:.4f}")
                else:
                    row["d_gs_lb"] = d
                    if verbose:
                        print(f"      d={d:<3} fundamental falhou [{method}]  "
                              f"(custo decodificado {gs['map']['cost']:.1f} vs "
                              f"C*={cstar:.1f}; massa_otima={massa:.3f})")
        except MemoryError as exc:
            row["skipped"] = f"MemoryError em d={d}: {exc}"
            break

        tab_done = (not table) or (row["d_tab"] is not None)
        gs_done = row["d_gs"] is not None
        if tab_done and gs_done:
            break
        if dim > max_dim and not gs_done and tab_done:
            # as dimensões só crescem: nenhum cutoff maior pode ser testado
            row["skipped"] = f"dim {dim:,} > max_dim em d={d}; d_gs > {row['d_gs_lb']}"
            break

    if row["d_gs"] is not None:
        row["status_gs"] = "MEDIDO"
    elif row["d_gs_lb"] is not None:
        row["status_gs"] = "CENSURADO"
    row["seconds"] = time.time() - t0
    return row


# ---------------------------------------------------------------------------
# Saída
# ---------------------------------------------------------------------------
def write_csv(path, rows, fields=None):
    fields = fields or (list(rows[0]) if rows else None)
    if not fields:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_table(rows):
    print("\n" + "=" * 104)
    print("CUTOFF MÍNIMO — a modular resolve a partir de que d?")
    print("=" * 104)
    print(f"{'C':>3}{'V':>3}{'R':>3}{'M*R':>5}{'#ótim':>7}{'d_tab':>7}{'d_gs':>7}"
          f"{'d_gs/MR':>9}{'massa':>8}{'rho':>9}{'gap':>9}{'metodo':>8}{'seg':>9}"
          f"  status")
    for r in rows:
        d_tab = "—" if r["d_tab"] is None else str(r["d_tab"])
        if r["d_gs"] is not None:
            d_gs, ratio = str(r["d_gs"]), f"{r['d_gs'] / r['MR']:.2f}"
        elif r["d_gs_lb"] is not None:
            d_gs = f">{r['d_gs_lb']}"
            ratio = f">{r['d_gs_lb'] / r['MR']:.2f}"
        else:
            d_gs, ratio = "—", "—"

        def fm(key, fmt):
            return "—" if r[key] is None else format(r[key], fmt)

        print(f"{r['C']:>3}{r['V']:>3}{r['R']:>3}{r['MR']:>5}{r['n_optimal']:>7}"
              f"{d_tab:>7}{d_gs:>7}{ratio:>9}{fm('massa_at_dgs', '.3f'):>8}"
              f"{fm('rho_at_dgs', '.2f'):>9}{fm('gap_at_dgs', '.4f'):>9}"
              f"{(r['method_at_dgs'] or '—'):>8}{r['seconds']:>9.1f}"
              f"  {r['status_gs']}")


def main(cities=(3, 4, 5),
         vehicles=(1, 2, 3),
         cutoffs=(3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20),
         max_dim=200_000,        # limite do caminho ESPARSO (cutoff^C)
         dense_limit=2000,       # até aqui usa diagonalização densa
         max_configs=300_000,
         table=True,
         out="cutoff_minimo.csv",
         trace_out="cutoff_minimo_trace.csv"):

    rows, trace = [], []
    for C in cities:
        for V in vehicles:
            if V > C:
                continue
            print(f"\n=== C={C} V={V} ===")
            r = sweep_case(C, V, list(cutoffs), max_dim, max_configs,
                           dense_limit, table, True, trace)
            if r["skipped"]:
                print(f"      pulado: {r['skipped']}")
            rows.append(r)
            write_csv(out, rows)
            write_csv(trace_out, trace, TRACE_FIELDS)

    print_table(rows)
    print(f"\nsalvo em {out} e {trace_out}")
    print("\nLeitura:")
    print("  d_gs/MR  é a resposta. Se ficar ~constante, a modular escala como a")
    print("           extensão da Etapa 1; se CAIR com M*R, há vantagem de recursos.")
    print("  '>k'     censurado: o fundamental falhou até d=k e não deu para testar")
    print("           mais (limite de dimensão). É cota inferior, NÃO medida.")
    print("  massa    probabilidade do POVM sobre rotas ótimas — comparável entre")
    print("           casos; rho cresce com a dimensão e não é.")


if __name__ == "__main__":
    main(cities=[3, 4], vehicles=[1, 2, 3])