"""
exp_cutoff_minimo.py — qual o menor cutoff em que a formulação modular ainda
resolve o problema?   (v2)

A pergunta
----------
A modular mantém N qumodes mas precisa de cutoff maior que a Etapa 1, porque
posição e veículo moram nas duas quadraturas e precisam ser RESOLVIDOS, não
apenas rotulados. Se d_min crescer proporcionalmente a M·R, a modular não tem
vantagem de recursos sobre uma extensão direta da Etapa 1 (cutoff = M·R). Se
crescer mais devagar, ela ganha.

O que a v1 mostrou e a v2 corrige
---------------------------------
Com lambda_col = lambda_gap = lambda_vehicle = 30, C=4 V=1 decodificava a
rota de custo 37 (C*=32) de d=5 a d=10. Com as penalidades de
H.suggest_penalties o mesmo caso decodifica a rota ótima em d=8 e d=10
(exp_diag_gs.py). Logo o d_gs da v1 media a penalidade, não o cutoff. Por isso:

1. penalties="sugerida" é o padrão (use "base" para reproduzir a v1).
2. Critério de sucesso por ROTA agregada (criterion="route"): soma a
   probabilidade do POVM sobre todas as configurações que decodificam para a
   mesma rota e exige que a rota de maior massa seja ótima. O critério antigo
   (criterion="config") olha só a configuração mais provável; como a massa fica
   espalhada em muitas configurações, ele é frágil.
3. Confirmação: o fundamental é calculado em TODOS os cutoffs da grade com
   dim <= max_dim (agora leva segundos). d_gs é o primeiro acerto; d_gs_stable
   é o menor d tal que ele e todos os cutoffs testados depois também acertam,
   com pelo menos `confirm` cutoffs depois. É d_gs_stable que o exp_recursos usa.
4. O rastro registra col_medio = <sum_{i<j} A_i A_j> sem peso (colisões
   esperadas segundo o POVM) e dE_chk (conferência independente da energia).
   col_medio ~ 1 mesmo com penalidade 5x maior indica resolução pobre que a
   penalidade não elimina.
5. Threads fixadas em 1 (OMP/OPENBLAS/MKL) se não definidas. A v1 sofria um salto
   de 50 a 170 vezes no tempo por volta de dim ~ 5e3 a 8e3.
6. table=False por padrão: a tabela de nós respondeu por 99,5% das 67 h do
   C=5 V=1 e d_tab não é monotônico em d.

v3 (a partir do trace da v2)
----------------------------
O critério por rota também acerta por acaso. Em C=4 V=3, d=2, a massa ótima é
0,0743; configurações UNIFORMES decodificam para a rota ótima com massa
96/1296 = 0,0741. Mesmo assim a v2 declarou d_est=5, com massa 0,128 (1,7x o
acaso). Além disso, o desempate de colisões por índice de cidade dá a cada rota
um "volume" de decodificação muito desigual (C=4 V=1: 35 configurações para a
permutação crescente, 1 para a decrescente), o que enviesa p_best_sub.
Por isso a v3 mede, além do critério por rota:
  massa_unif    massa ótima se o POVM fosse uniforme (piso do acaso)
  lift          massa_otima / massa_unif
  p_valid       probabilidade de a configuração ser VÁLIDA: posições distintas
                e contíguas de 0 a k-1 em cada veículo (e todos os veículos
                usados se V>1)
  p_opt_valid   massa ótima entre as válidas (pós-seleção: sem desempate)
  chance_valid  n_ótimas / n_rotas_válidas (piso do acaso após a pós-seleção)
criterion="valid" (padrão): sucesso se a melhor rota válida é ótima e
P(ótima | válida) >= lift_min * chance_valid. lift_min precisa ser reportado
junto com d_min; exp_dmin_limiar.py mostra d_min em função do limiar.

status_gs:  MEDIDO (d_gs_stable achado) | CENSURADO (só há d_gs_lb: d > lb) |
            NAO_CONFIRMADO (acertos, mas sem cutoffs suficientes depois) |
            NAO_TESTADO.

Nada aqui usa VQE. Tudo em numpy float64.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import csv
import dataclasses
import itertools
import time

import numpy as np

import qumodes.hamiltonian as H
import qumodes.hamiltonian_sparse as HS

HBAR = 2.0

TRACE_FIELDS = ["C", "V", "d", "dim", "kind", "ok", "ok_cfg", "ok_route",
                "decoded_cost", "c_star", "energy", "dE_chk", "gap",
                "massa_otima", "massa_unif", "lift", "p_best_opt", "p_best_sub",
                "rho", "p_valid", "p_opt_valid", "chance_valid", "top_valid_ok",
                "ok_valid", "col_medio", "method", "n_matvec", "seconds"]


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


def route_masses(dist, inst):
    """Massa do POVM agregada por ROTA (canon -> probabilidade)."""
    flat = dist.reshape(-1)
    masses = {}
    for f in np.flatnonzero(flat > 1e-12):
        slots = np.unravel_index(int(f), dist.shape)
        key = canon(routes_from_slots(slots, inst))
        masses[key] = masses.get(key, 0.0) + float(flat[f])
    return masses


def summarize_routes(masses, optset):
    """(massa ótima total, maior massa de rota ótima, maior massa de rota
    subótima, a rota de maior massa é ótima?)"""
    opt = [m for k, m in masses.items() if k in optset]
    sub = [m for k, m in masses.items() if k not in optset]
    p_opt = max(opt) if opt else 0.0
    p_sub = max(sub) if sub else 0.0
    return float(sum(opt)), float(p_opt), float(p_sub), bool(p_opt > p_sub)


def optimal_mass(dist, inst, optset):
    return summarize_routes(route_masses(dist, inst), optset)[0]


def _is_valid(slots, inst, require_all):
    """Configuração válida: em cada veículo as posições são distintas e
    formam {0, ..., k-1}; com require_all, todo veículo atende alguma cidade."""
    R = inst.R
    pos = [[] for _ in range(inst.M)]
    for s in slots:
        pos[int(s) // R].append(int(s) % R)
    for p in pos:
        if sorted(p) != list(range(len(p))):
            return False
    if require_all and any(len(p) == 0 for p in pos):
        return False
    return True


def valid_stats(dist, inst, optset, require_all):
    """(P(válida), massa das rotas ótimas entre as válidas, a melhor rota
    válida é ótima?). Sem colisões não há desempate por índice de cidade."""
    flat = dist.reshape(-1)
    p_valid, p_opt_valid = 0.0, 0.0
    masses = {}
    for f in np.flatnonzero(flat > 1e-12):
        slots = np.unravel_index(int(f), dist.shape)
        if not _is_valid(slots, inst, require_all):
            continue
        w = float(flat[f])
        key = canon(routes_from_slots(slots, inst))
        masses[key] = masses.get(key, 0.0) + w
        p_valid += w
        if key in optset:
            p_opt_valid += w
    top_ok = bool(masses) and (max(masses, key=masses.get) in optset)
    return p_valid, p_opt_valid, top_ok


def uniform_baseline(inst, optset, shape):
    """Massa ótima se o POVM fosse uniforme sobre as (M*R)^N configurações."""
    u = np.full(shape, 1.0 / float(np.prod(shape)))
    return optimal_mass(u, inst, optset)


def chance_valid(inst, optset, require_all):
    """(n_ótimas / n_rotas_válidas, n_rotas_válidas): piso do acaso para
    P(ótima | válida). Enumera as (M*R)^N configurações uma vez por caso."""
    keys = set()
    for slots in itertools.product(range(inst.M * inst.R), repeat=inst.N):
        if _is_valid(slots, inst, require_all):
            keys.add(canon(routes_from_slots(slots, inst)))
    if not keys:
        return float("nan"), 0
    return len(optset) / len(keys), len(keys)


# ---------------------------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------------------------
def params_for(V):
    """Parâmetros da v1 (penalidades 30)."""
    return H.HamiltonianParams(
        lambda_dist=1.0, lambda_col=30.0, lambda_gap=30.0,
        lambda_vehicle=30.0 if V > 1 else 0.0,
        lambda_cap=0.0, lambda_disc=5.0, lambda_exc=0.0)


def make_params(inst, V, penalties="sugerida", lambda_exc=0.0):
    if penalties == "base":
        p = params_for(V)
    elif penalties == "sugerida":
        p = H.suggest_penalties(inst, require_all_vehicles=(V > 1),
                                lambda_disc=5.0)
    else:
        raise ValueError(f"penalties desconhecido: {penalties}")
    return dataclasses.replace(p, lambda_exc=lambda_exc)


# ---------------------------------------------------------------------------
# Varredura
# ---------------------------------------------------------------------------
def _tr(C, V, d, kind, ok, ok_cfg, ok_route, cost, cstar, energy, dE, gap,
        massa, p_opt, p_sub, rho, col, method, n_matvec, secs, extra=None):
    out = dict(C=C, V=V, d=d, dim=d ** C, kind=kind, ok=int(ok),
               ok_cfg=ok_cfg if ok_cfg == "" else int(ok_cfg),
               ok_route=ok_route if ok_route == "" else int(ok_route),
               decoded_cost=cost, c_star=cstar, energy=energy, dE_chk=dE,
               gap=gap, massa_otima=massa, p_best_opt=p_opt, p_best_sub=p_sub,
               rho=rho, col_medio=col, method=method, n_matvec=n_matvec,
               seconds=round(secs, 3))
    out.update(extra or {})
    return out


def sweep_case(C, V, cutoffs, max_dim, max_configs, dense_limit=700,
               table=False, penalties="sugerida", lambda_exc=0.0,
               criterion="valid", confirm=2, lift_min=2.0, verbose=True,
               trace=None, seed=42):
    trace = trace if trace is not None else []
    inst = build_instance(C, V, seed)
    cstar, optset = classical_optima(inst, V > 1)
    par = make_params(inst, V, penalties, lambda_exc)
    MR = inst.M * inst.R
    n_cfg = MR ** C

    row = {"C": C, "V": V, "seed": seed, "R": inst.R, "MR": MR,
           "n_configs": n_cfg,
           "c_star": cstar, "n_optimal": len(optset),
           "penalties": penalties, "lambda_exc": lambda_exc,
           "criterion": criterion, "lift_min": lift_min,
           "massa_unif": None, "chance_valid": None, "n_valid_routes": None,
           "lambda_col": par.lambda_col,
           "lambda_gap": par.lambda_gap, "lambda_vehicle": par.lambda_vehicle,
           "d_tab": None, "d_gs": None, "d_gs_stable": None, "d_gs_lb": None,
           "status_gs": "NAO_TESTADO", "n_tested": 0, "n_ok": 0,
           "rho_at_dgs": None, "massa_at_dgs": None, "gap_at_dgs": None,
           "weyl_qtr_at_dgs": None, "E0_at_dgs": None, "map_p_at_dgs": None,
           "method_at_dgs": "", "seconds": 0.0, "skipped": ""}

    if n_cfg > max_configs:
        row["skipped"] = f"configs {n_cfg:,} > max_configs"
        return row

    base_unif = uniform_baseline(inst, optset, (MR,) * C)
    chance, n_valid = chance_valid(inst, optset, V > 1)
    row.update(massa_unif=base_unif, chance_valid=chance, n_valid_routes=n_valid)
    if verbose:
        print(f"      acaso: massa_unif={base_unif:.4f}  chance_valid={chance:.4f} "
              f"({len(optset)} ótimas / {n_valid} rotas válidas)")

    gs_log = []                      # [(d, ok)] na ordem da grade
    t0 = time.time()
    for d in cutoffs:
        dim = d ** C
        ops = H.build_modular_operators(d, inst.M, inst.R, HBAR)
        try:
            # --- critério fraco (opcional): tabela de nós --------------------
            if table and row["d_tab"] is None:
                ts = time.time()
                tab = H.node_energy_table(inst, ops, par, max_configs=max_configs)
                ok = canon(tab["routes"]) in optset
                trace.append(_tr(C, V, d, "tabela", ok, "", "", tab["cost"],
                                 cstar, tab["min_energy"], "", "", "", "", "",
                                 "", "", "nos", "", time.time() - ts))
                if ok:
                    row["d_tab"] = d
                if verbose:
                    print(f"      d={d:<3} tabela {'OK' if ok else 'falhou'}  "
                          f"(custo {tab['cost']:.1f} vs C*={cstar:.1f})")

            # --- critério forte: fundamental ---------------------------------
            if dim <= max_dim:
                ts = time.time()
                method = "dense" if dim <= dense_limit else "sparse"
                if method == "dense":
                    gs = H.ground_state(inst, ops, par, max_dim=dense_limit)
                else:
                    gs = HS.ground_state_sparse(inst, ops, par)
                secs = time.time() - ts

                dist = gs["distribution"]
                masses = route_masses(dist, inst)
                massa, p_opt, p_sub, ok_route = summarize_routes(masses, optset)
                ok_cfg = canon(gs["map"]["routes"]) in optset
                lift = massa / base_unif if base_unif > 0 else float("nan")
                p_valid, p_opt_valid, top_valid_ok = valid_stats(
                    dist, inst, optset, V > 1)
                p_cond = p_opt_valid / p_valid if p_valid > 0 else float("nan")
                ok_valid = bool(top_valid_ok and p_valid > 0
                                and p_cond >= lift_min * chance)
                ok_lift = bool(lift >= lift_min)
                ok = {"route": ok_route, "config": ok_cfg,
                      "lift": ok_lift, "valid": ok_valid}[criterion]
                cost_cfg = gs["map"]["cost"]
                cost_route = (H.routes_cost(max(masses, key=masses.get), inst.D)
                              if masses else float("nan"))
                cost = cost_cfg if criterion == "config" else cost_route
                unif = len(optset) / dist.size
                rho = massa / unif if unif else float("nan")

                ec = H.energy_components(gs["psi"], inst, ops, par)
                dE = float(ec["total"] - gs["energy"])
                col = (float(ec["col"]) / par.lambda_col
                       if par.lambda_col != 0 else float("nan"))

                gs_log.append((d, ok))
                trace.append(_tr(C, V, d, "fundamental", ok, ok_cfg, ok_route,
                                 cost, cstar, gs["energy"], dE, gs["gap"],
                                 massa, p_opt, p_sub, rho, col, method,
                                 gs.get("n_matvec", ""), secs,
                                 extra=dict(massa_unif=base_unif, lift=lift,
                                            p_valid=p_valid,
                                            p_opt_valid=p_opt_valid,
                                            chance_valid=chance,
                                            top_valid_ok=int(top_valid_ok),
                                            ok_valid=int(ok_valid))))
                if ok and row["d_gs"] is None:
                    row.update(d_gs=d, rho_at_dgs=rho, massa_at_dgs=massa,
                               gap_at_dgs=gs["gap"], E0_at_dgs=gs["energy"],
                               map_p_at_dgs=gs["map"]["probability"],
                               weyl_qtr_at_dgs=ops.weyl_commutator(max(2, d // 4)),
                               method_at_dgs=method)
                if verbose:
                    print(f"      d={d:<3} fundamental {'OK   ' if ok else 'falhou'}"
                          f" [{method}] custo={cost:.1f} (C*={cstar:.1f}) "
                          f"massa={massa:.3f} lift={lift:.2f} "
                          f"p_ot={p_opt:.3f} p_sub={p_sub:.3f} "
                          f"P(val)={p_valid:.3f} P(ot|val)={p_cond:.3f} "
                          f"(acaso {chance:.3f}) col={col:.2f} {secs:.1f}s")
        except MemoryError as exc:
            row["skipped"] = f"MemoryError em d={d}: {exc}"
            break

        tab_done = (not table) or (row["d_tab"] is not None)
        if dim > max_dim and tab_done:
            row["skipped"] = f"dim {dim:,} > max_dim em d={d}"
            break

    # --- estabilidade e censura ---------------------------------------------
    row["n_tested"] = len(gs_log)
    row["n_ok"] = sum(1 for _, ok in gs_log if ok)
    fails = [d for d, ok in gs_log if not ok]
    if fails:
        row["d_gs_lb"] = max(fails)
    for k, (d, ok) in enumerate(gs_log):
        later = gs_log[k + 1:]
        if ok and len(later) >= confirm and all(o for _, o in later):
            row["d_gs_stable"] = d
            break

    if row["d_gs_stable"] is not None:
        row["status_gs"] = "MEDIDO"
    elif row["d_gs_lb"] is not None:
        row["status_gs"] = "CENSURADO"
    elif gs_log:
        row["status_gs"] = "NAO_CONFIRMADO"
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
    print("\n" + "=" * 112)
    print("CUTOFF MÍNIMO — a modular resolve a partir de que d?")
    print("=" * 112)
    print(f"{'C':>3}{'V':>3}{'R':>3}{'M*R':>5}{'#ótim':>7}{'lcol':>7}{'d_gs':>6}"
          f"{'d_est':>7}{'d_est/MR':>10}{'massa':>8}{'gap':>9}{'ok/test':>9}"
          f"{'seg':>9}  status")
    for r in rows:
        d_gs = "—" if r["d_gs"] is None else str(r["d_gs"])
        if r["d_gs_stable"] is not None:
            d_est = str(r["d_gs_stable"])
            ratio = f"{r['d_gs_stable'] / r['MR']:.2f}"
        elif r["d_gs_lb"] is not None:
            d_est = f">{r['d_gs_lb']}"
            ratio = f">{r['d_gs_lb'] / r['MR']:.2f}", file=f)
        else:
            d_est, ratio = "—", "—"

        def fm(key, fmt):
            return "—" if r[key] is None else format(r[key], fmt)

        print(f"{r['C']:>3}{r['V']:>3}{r['R']:>3}{r['MR']:>5}{r['n_optimal']:>7}"
              f"{r['lambda_col']:>7.0f}{d_gs:>6}{d_est:>7}{ratio:>10}"
              f"{fm('massa_at_dgs', '.3f'):>8}{fm('gap_at_dgs', '.4f'):>9}"
              f"{str(r['n_ok']) + '/' + str(r['n_tested']):>9}"
              f"{r['seconds']:>9.1f}  {r['status_gs']}")


def main(cities=(3, 4, 5),
         vehicles=(1, 2, 3),
         cutoffs=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20),
         max_dim=100_000,        # limite do caminho ESPARSO (cutoff^C)
         dense_limit=700,        # até aqui usa diagonalização densa
         max_configs=300_000,
         table=False,
         penalties="sugerida",   # "sugerida" | "base" (reproduz a v1)
         lambda_exc=0.0,
         criterion="valid",      # "valid" | "lift" | "route" | "config"
         confirm=2,
         lift_min=2.0,           # limiar: P(ótima|válida) >= lift_min * acaso
         seed=42,                # semente da instância (distâncias aleatórias)
         out="cutoff_minimo.csv",
         trace_out="cutoff_minimo_trace.csv"):

    rows, trace = [], []
    for C in cities:
        for V in vehicles:
            if V >= C:      # V=C é degenerado: uma única rota válida (chance=1)
                continue
            print(f"\n=== C={C} V={V} ===")
            r = sweep_case(C, V, list(cutoffs), max_dim, max_configs,
                           dense_limit, table, penalties, lambda_exc,
                           criterion, confirm, lift_min, True, trace, seed)
            if r["skipped"]:
                print(f"      parou: {r['skipped']}")
            rows.append(r)
            write_csv(out, rows)
            write_csv(trace_out, trace, TRACE_FIELDS)

    print_table(rows)
    print(f"\nsalvo em {out} e {trace_out}")
    print("\nLeitura:")
    print("  d_gs     primeiro cutoff com acerto.")
    print("  d_est    d_gs_stable: menor d em que ele e todos os testados depois")
    print("           acertam (>= `confirm` depois). É o que vai para exp_recursos.")
    print("  '>k'     censurado: houve falha até d=k. Cota inferior, NÃO medida.")
    print("  massa    probabilidade do POVM sobre rotas ótimas no primeiro acerto.")
    print("  Veja col_medio no trace: colisões esperadas pelo POVM. Se ficar ~1 em")
    print("  todo d testado, o acerto por rota ocorre em regime de resolução pobre.")


if __name__ == "__main__":
    main(cities=[3, 4], vehicles=[1, 2, 3])