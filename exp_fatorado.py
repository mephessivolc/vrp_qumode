"""
exp_fatorado.py — caminho B: estados-produto sem tensor d^N.

Usa qumodes/product_energy.py para calcular a energia de qualquer configuração
de nós em milissegundos, até d=64 e além. Tudo é lançado pelo exp_run.py.

Funções
-------
verificar              compara o avaliador fatorado com H.energy_components em
                       estados-produto ALEATÓRIOS e em estados de nó, para as
                       duas gap_form. Se falhar, nada abaixo vale.
overhead_por_perfil    overhead por perfil, termo a termo, em d até 64 (a parte
                       C só ia até 16 e por amostra). Reproduz a parte B em d=8.
decisao_por_semente    rota que minimiza a energia de estado-produto, por
                       semente e d: prevê o fundamental? em que d vira o ótimo?
pecas_da_lacuna        decompõe o termo de lacuna em 6 peças e mostra qual peça
                       cria a diferença entre perfis (candidata a mecanismo).
campo_medio_por_perfil o termo de lacuna em estados-produto é quase todo de campo
                       médio: por verificação, a_i * Bbar^2 com
                       Bbar = c - sum <n_q>. Separa <n_q> em parte PRÓPRIA (nó
                       do predecessor) e VAZAMENTO (outras cidades), e mostra a
                       matriz de sobreposição entre nós.

Limites
-------
Estado-produto NÃO é o fundamental: a decisão por semente só vale como
substituto se reproduzir o custo decodificado pelo fundamental (conferência em
d=16 com GS_COST_D16). Capacidade (lambda_cap > 0) não é suportada.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import itertools
from collections import defaultdict

import numpy as np

import qumodes.hamiltonian as H
import qumodes.product_energy as PE
from exp_cutoff_minimo import (HBAR, _is_valid, build_instance, canon,
                               classical_optima, make_params,
                               routes_from_slots)
from exp_perfil_rotas import GS_COST_D16, perfil

# Parte B (d=8, soma, semente 42, TODAS as configurações válidas): min/mediana/máx
# do overhead total. O avaliador fatorado deve reproduzir estes números.
REF_PARTE_B_D8 = {(1, 3): (240.34, 253.26, 264.12), (2, 2): (210.26, 218.22, 224.02)}


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------
def configs_validas(inst, V):
    X = inst.M * inst.R
    return [s for s in itertools.product(range(X), repeat=inst.N)
            if _is_valid(s, inst, V > 1)]


def _produto(phis):
    psi = phis[0]
    for p in phis[1:]:
        psi = np.multiply.outer(psi, p)
    return psi


def _estado_aleatorio(rng, d):
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# 1. verificação contra energy_components
# ---------------------------------------------------------------------------
def verificar(casos=((3, 1, 4), (3, 2, 4), (4, 2, 4), (4, 1, 4)),
              formas=("soma", "predecessor"), tol=1e-9, seed=7):
    """True se o avaliador fatorado reproduz energy_components em todos os
    termos (dist, col, gap, vehicle, disc, exc, total, mean_photons)."""
    rng = np.random.default_rng(seed)
    chaves = ("dist", "col", "gap", "vehicle", "disc", "exc", "total",
              "mean_photons")
    ok_all = True
    print("=" * 96)
    print("VERIFICAÇÃO: energia fatorada contra H.energy_components "
          f"(tolerância {tol:g})")
    print("=" * 96)
    for (C, V, d) in casos:
        for forma in formas:
            inst = build_instance(C, V, 42)
            par = make_params(inst, V, "sugerida", 0.05, forma)   # inclui confinamento
            ops = H.build_modular_operators(d, inst.M, inst.R, HBAR)
            A, Hd, nop = PE.operator_stack(ops, inst.M, inst.R)
            tab = PE.NodeTable(ops, inst.M, inst.R)
            X = inst.M * inst.R
            for tipo in ("aleatorio", "nos"):
                if tipo == "aleatorio":
                    phis = [_estado_aleatorio(rng, d) for _ in range(C)]
                    mom = PE.moments_from_states(phis, A, Hd, nop)
                else:
                    slots = tuple(int(x) for x in rng.integers(0, X, size=C))
                    phis = [H.node_state(ops, s // inst.R, s % inst.R) for s in slots]
                    mom = tab.take(slots)
                ref = H.energy_components(_produto(phis), inst, ops, par)
                got = PE.energy_from_moments(mom, inst, par)
                pior = max(abs(float(got[k]) - float(ref[k]))
                           / (1.0 + abs(float(ref[k]))) for k in chaves)
                ok = pior < tol
                ok_all &= ok
                print(f"C={C} V={V} d={d} R={inst.R} forma={forma:<11} "
                      f"estado={tipo:<9} erro_rel_max={pior:.2e}  "
                      f"{'OK' if ok else 'FALHOU'}")
    print("\n" + ("AVALIADOR FATORADO OK" if ok_all
                  else "FALHOU — não use os resultados fatorados"))
    return ok_all


# ---------------------------------------------------------------------------
# 2. overhead por perfil em d grande
# ---------------------------------------------------------------------------
def overhead_por_perfil(C=4, V=2, ds=(8, 12, 16, 24, 32, 48, 64), seed=42,
                        formas=("soma", "predecessor")):
    inst = build_instance(C, V, seed)
    configs = configs_validas(inst, V)
    info = []
    for s in configs:
        rotas = routes_from_slots(s, inst)
        info.append((s, perfil(rotas), H.routes_cost(rotas, inst.D)))
    termos = ("dist-custo", "col", "gap", "vehicle", "disc", "total")
    for forma in formas:
        par = make_params(inst, V, "sugerida", 0.0, forma)
        print("\n" + "=" * 100)
        print(f"OVERHEAD POR PERFIL  C={C} V={V} semente={seed} gap_form={forma} "
              f"({len(configs)} configurações válidas; média por perfil)")
        print("=" * 100)
        print(f"{'d':>4}{'perfil':>10}{'n':>5}" + "".join(f"{t:>12}" for t in termos))
        for d in ds:
            ops = H.build_modular_operators(d, inst.M, inst.R, HBAR)
            tab = PE.NodeTable(ops, inst.M, inst.R)
            acc = defaultdict(lambda: defaultdict(list))
            for (s, p, custo) in info:
                e = PE.energy_from_moments(tab.take(s), inst, par)
                acc[p]["dist-custo"].append(e["dist"] - custo)
                for k in ("col", "gap", "vehicle", "disc"):
                    acc[p][k].append(e[k])
                acc[p]["total"].append(e["encoding"] - custo)
            medias = {}
            for p in sorted(acc):
                medias[p] = {k: float(np.mean(v)) for k, v in acc[p].items()}
                print(f"{d:>4}{str(p):>10}{len(acc[p]['total']):>5}"
                      + "".join(f"{medias[p][t]:>12.2f}" for t in termos))
            ps = sorted(medias)
            if len(ps) > 1:
                print(f"{'':>4}{'diferença':>15}" + "".join(
                    f"{medias[ps[0]][t] - medias[ps[-1]][t]:>12.2f}" for t in termos)
                    + f"   ({ps[0]} menos {ps[-1]})")
            if (C, V, seed, forma, d) == (4, 2, 42, "soma", 8):
                print("    conferência com a parte B (d=8, soma): mín / mediana / máx do total")
                for p in sorted(acc):
                    v = np.array(acc[p]["total"])
                    ref = REF_PARTE_B_D8.get(p)
                    print(f"      {str(p):>8}: {v.min():.2f} / {np.median(v):.2f} / "
                          f"{v.max():.2f}   referência {ref}")


# ---------------------------------------------------------------------------
# 3. decisão por semente: o argmin de estado-produto prevê o fundamental?
# ---------------------------------------------------------------------------
def decisao_por_semente(C=4, V=2, seeds=(42, 1, 2, 3, 4, 5),
                        ds=(8, 12, 16, 24, 32, 48, 64),
                        formas=("soma", "predecessor")):
    preparado = {}
    for s in seeds:
        inst = build_instance(C, V, s)
        cstar, optset = classical_optima(inst, V > 1)
        configs = configs_validas(inst, V)
        rotas = [routes_from_slots(c, inst) for c in configs]
        custo = np.array([H.routes_cost(r, inst.D) for r in rotas])
        otim = np.array([canon(r) in optset for r in rotas])
        preparado[s] = (inst, cstar, configs, custo, otim)

    inst0 = preparado[seeds[0]][0]
    tabs = {}
    for d in ds:
        ops = H.build_modular_operators(d, inst0.M, inst0.R, HBAR)
        tabs[d] = PE.NodeTable(ops, inst0.M, inst0.R)

    for forma in formas:
        res = {}
        for s in seeds:
            inst, cstar, configs, custo, otim = preparado[s]
            par = make_params(inst, V, "sugerida", 0.0, forma)
            for d in ds:
                E = np.array([PE.energy_from_moments(tabs[d].take(c), inst, par)["encoding"]
                              for c in configs])
                k = int(np.argmin(E))
                dE = float(E[otim].min() - E.min())
                res[(s, d)] = (float(custo[k]), bool(otim[k]), dE)

        print("\n" + "=" * 100)
        print(f"DECISÃO POR SEMENTE  C={C} V={V} gap_form={forma}: custo da rota que "
              f"minimiza a energia de estado-produto  ('*' = ótima)")
        print("=" * 100)
        print(f"{'semente':>8}{'C*':>6}" + "".join(f"{d:>9}" for d in ds)
              + f"{'d_vira':>9}")
        for s in seeds:
            cstar = preparado[s][1]
            linha = f"{s:>8}{cstar:>6.0f}"
            for d in ds:
                c, ok, _ = res[(s, d)]
                linha += f"{(f'{c:.0f}*' if ok else f'{c:.0f}'):>9}"
            vira = None
            for i, d in enumerate(ds):
                if all(res[(s, dd)][1] for dd in ds[i:]):
                    vira = d
                    break
            print(linha + f"{(str(vira) if vira else '>' + str(ds[-1])):>9}")

        print(f"\nenergia extra do melhor ótimo sobre o mínimo (0 = o ótimo vence)")
        print(f"{'semente':>8}" + "".join(f"{d:>9}" for d in ds))
        for s in seeds:
            print(f"{s:>8}" + "".join(f"{res[(s, d)][2]:>9.2f}" for d in ds))

        conhecido = GS_COST_D16.get((C, V), {})
        if conhecido and 16 in ds:
            print("\nconferência com o fundamental em d=16 (custo decodificado)")
            for s in seeds:
                if s in conhecido:
                    prev = res[(s, 16)][0]
                    print(f"  semente {s}: fundamental {conhecido[s]:.0f}   "
                          f"estado-produto {prev:.0f}   "
                          f"{'casa' if abs(prev - conhecido[s]) < 1e-9 else 'NÃO casa'}")


# ---------------------------------------------------------------------------
# 4. peças do termo de lacuna por perfil
# ---------------------------------------------------------------------------
def pecas_da_lacuna(C=4, V=2, ds=(8, 16, 32, 64), seed=42,
                    formas=("soma", "predecessor")):
    inst = build_instance(C, V, seed)
    configs = configs_validas(inst, V)
    perfis = [perfil(routes_from_slots(c, inst)) for c in configs]
    print("\n" + "=" * 100)
    print("PEÇAS DO TERMO DE LACUNA (sem peso lambda_gap), média por perfil")
    print("  P1 c²<A>   P2 cruzado   P3a campo médio   P3b FLUTUAÇÃO das outras "
          "cidades   P4   P5 <SAS>")
    print("=" * 100)
    for forma in formas:
        par = make_params(inst, V, "sugerida", 0.0, forma)
        print(f"\ngap_form={forma}")
        print(f"{'d':>4}{'perfil':>10}" + "".join(f"{p:>11}" for p in PE.GAP_PIECES)
              + f"{'soma':>11}")
        for d in ds:
            ops = H.build_modular_operators(d, inst.M, inst.R, HBAR)
            tab = PE.NodeTable(ops, inst.M, inst.R)
            acc = defaultdict(list)
            for s, p in zip(configs, perfis):
                acc[p].append(PE.energy_from_moments(tab.take(s), inst, par)["gap_pieces"])
            medias = {p: np.mean(np.array(v), axis=0) for p, v in acc.items()}
            for p in sorted(medias):
                print(f"{d:>4}{str(p):>10}" + "".join(f"{x:>11.3f}" for x in medias[p])
                      + f"{medias[p].sum():>11.3f}")
            ps = sorted(medias)
            if len(ps) > 1:
                dif = medias[ps[0]] - medias[ps[-1]]
                print(f"{'':>4}{'diferença':>10}" + "".join(f"{x:>11.3f}" for x in dif)
                      + f"{dif.sum():>11.3f}   ({ps[0]} menos {ps[-1]})")


# ---------------------------------------------------------------------------
# 5. campo médio por verificação
# ---------------------------------------------------------------------------
def campo_medio_por_perfil(C=4, V=2, ds=(8, 16, 32, 64), seed=42,
                           formas=("soma", "predecessor")):
    """Em estados-produto o termo de lacuna é quase todo de CAMPO MÉDIO: por
    verificação (cidade i em r >= 1), a_i * Bbar^2 com

        Bbar = c - sum_{q in Q} <n_q>,    <n_q> = sum_j a_j[nó (v, q)].

    (a peça de flutuação P3b é ~0, ver pecas_da_lacuna.) <n_q> se separa em
        próprio    a da cidade que ocupa o nó (v, q): infidelidade do nó, 1 - F
        vazamento  soma das OUTRAS cidades no nó (v, q): sobreposição entre nós
    Imprime, por perfil e por posição r da cidade verificada, as médias de
    c - próprio, vazamento, Bbar, a_i e a_i*Bbar^2, e a matriz de sobreposição
    O[x, y] = <phi_x|A_y|phi_x> dos nós do veículo 0.
    """
    inst = build_instance(C, V, seed)
    R = inst.R
    configs = configs_validas(inst, V)
    perfis = [perfil(routes_from_slots(c, inst)) for c in configs]
    for d in ds:
        ops = H.build_modular_operators(d, inst.M, inst.R, HBAR)
        tab = PE.NodeTable(ops, inst.M, inst.R)
        print("\n" + "=" * 100)
        print(f"CAMPO MÉDIO POR VERIFICAÇÃO  C={C} V={V} semente={seed} d={d}")
        print("=" * 100)
        print("sobreposição O[x,y] = <phi_x|A_y|phi_x>  (nós do veículo 0; linha = estado)")
        print(f"{'':>10}" + "".join(f"{'A(0,' + str(y) + ')':>10}" for y in range(R)))
        for x in range(R):
            print(f"{'phi(0,' + str(x) + ')':>10}"
                  + "".join(f"{tab.a[x, y]:>10.4f}" for y in range(R)))
        for forma in formas:
            print(f"\ngap_form={forma}")
            print(f"{'perfil':>10}{'r':>3}{'n':>5}{'c-próprio':>11}{'vazamento':>11}"
                  f"{'Bbar':>9}{'a_i':>8}{'a_i*Bbar^2':>12}")
            acc = defaultdict(list)
            for s, p in zip(configs, perfis):
                a = tab.a[list(s)]                        # (N, X)
                cidade_em = {divmod(x, R): i for i, x in enumerate(s)}
                for i, x in enumerate(s):
                    v, r = divmod(x, R)
                    if r < 1:
                        continue
                    c, qs = H.gap_operator_spec(forma, r)
                    proprio, vaz = 0.0, 0.0
                    for q in qs:
                        xq = v * R + q
                        j0 = cidade_em[(v, q)]
                        proprio += a[j0, xq]
                        vaz += a[:, xq].sum() - a[j0, xq]
                    bbar = c - proprio - vaz
                    acc[(p, r)].append((c - proprio, vaz, bbar, a[i, x],
                                        a[i, x] * bbar ** 2))
            for (p, r) in sorted(acc):
                m = np.mean(np.array(acc[(p, r)]), axis=0)
                print(f"{str(p):>10}{r:>3}{len(acc[(p, r)]):>5}{m[0]:>11.4f}"
                      f"{m[1]:>11.4f}{m[2]:>9.4f}{m[3]:>8.4f}{m[4]:>12.5f}")