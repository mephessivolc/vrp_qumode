"""
exp_perfil_rotas.py — o viés de V>=2 vem do PERFIL de comprimentos das rotas?

Hipótese (a testar, NÃO estabelecida)
-------------------------------------
Na decomposição por termo (diag_v2_seed42 contra diag_v2_seed3, C=4 V=2, d=16)
tudo exceto a distância é igual nas duas instâncias:

    col + gap + veh + disc = 144.94 (semente 42)  e  144.93 (semente 3)

enquanto a distância difere (46.86 contra 32.08). Se os dois estados pagassem
overheads diferentes por rotas diferentes, essa soma não coincidiria até a 2a
casa. A explicação mais simples: o overhead de codificação depende dos NÓS
(veículo, posição) ocupados, logo do PERFIL de comprimentos das rotas, por
exemplo (2,2) contra (1,3) em C=4 V=2. Os dois fundamentais moram no mesmo
perfil; o melhor perfil para a codificação pode não conter a rota ótima.

Isso também explica o padrão de casos:
  V=1           perfil único (C,)                     -> funciona (5 de 6)
  R=2           perfis equivalentes por troca de
                rótulo de veículo: (3,2), (4,3)       -> funcionam
  M>=2 e R>=3   perfis com conjuntos de nós diferentes:
                (4,2), (5,2), (5,3)                   -> falham

Parte A (segundos, sem simulação quântica)
------------------------------------------
Para cada semente enumera todas as rotas válidas e mostra o custo ótimo, o
perfil das rotas ótimas e o MELHOR custo dentro de cada perfil. Compara com o
custo decodificado pelo fundamental em d=16 (do trace). Previsão da hipótese:
nas sementes que falham (42, 1, 2, 4, 5) o custo decodificado (52, 65, 44, 85,
61) é o melhor custo de UM perfil que não contém o ótimo; na semente 3 (que
funciona) o ótimo está nesse perfil. Se o custo decodificado não for o melhor de
nenhum perfil, a hipótese não explica aquele caso.

Parte B (alguns minutos, usa node_energy_table)
-----------------------------------------------
Mede o overhead = <H>(estado-produto no nó) - custo clássico, agrupado por
perfil, sobre todas as configurações VÁLIDAS. O overhead não depende da
instância (col, gap, veh e disc não usam D), então uma instância basta.
Previsão: o perfil que contém o ótimo nas sementes que falham tem overhead
maior que o perfil favorecido, por mais que a diferença de custo entre rotas
(2 a 14 unidades nas sementes testadas).

Uso
---
    python exp_perfil_rotas.py            # só a parte A
    python exp_perfil_rotas.py 8          # A e B com d=8
    python exp_perfil_rotas.py 8 12       # A e B com d=8 e d=12
    python exp_perfil_rotas.py c 8 12 16  # só a parte C (amostra, termo a termo)

Parte C: por amostra de configurações válidas de cada perfil, calcula as
contribuições col, gap, veh e disc e a diferença entre perfis para cada termo,
em vários d. Mostra QUAL termo cria o viés e se ele encolhe com d.

Arquivos gravados (na pasta de execução; sobrescrevem os anteriores):
    parte A      -> resultado_a.txt
    parte B      -> resultado_b.txt
    parte C      -> resultado_c.txt
A saída continua aparecendo na tela, então não precisa redirecionar.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import contextlib
import itertools
import sys
from collections import defaultdict

import numpy as np

import qumodes.hamiltonian as H
from exp_cutoff_minimo import (HBAR, _is_valid, build_instance, canon,
                               make_params, routes_from_slots)

# custo da rota de maior massa do fundamental em d=16 (coluna decoded_cost dos
# traces seed*_trace.csv e cutoff_minimo_trace_v3.csv)
GS_COST_D16 = {
    (4, 2): {42: 52.0, 1: 65.0, 2: 44.0, 3: 34.0, 4: 85.0, 5: 61.0},
    (4, 1): {42: 32.0, 1: 36.0, 2: 30.0, 3: 30.0, 4: 58.0, 5: 43.0},
}


def perfil(rotas):
    """Comprimentos das rotas, ordenados: (2, 2) ou (1, 3)."""
    return tuple(sorted(len(r) for r in rotas))


def todas_as_rotas(inst, require_all):
    """canon -> custo, para todas as rotas viáveis (cada veículo <= R cidades)."""
    N, M, R = inst.N, inst.M, inst.R
    out = {}
    for perm in itertools.permutations(range(1, N + 1)):
        for cuts in itertools.combinations(range(1, N), M - 1):
            idx = [0, *cuts, N]
            rts = [list(perm[idx[k]:idx[k + 1]]) for k in range(M)]
            if require_all and any(not r for r in rts):
                continue
            if any(len(r) > R for r in rts):
                continue
            key = canon(rts)
            if key not in out:
                out[key] = H.routes_cost(rts, inst.D)
    return out


def parte_a(C=4, V=2, seeds=(42, 1, 2, 3, 4, 5)):
    print("=" * 96)
    print(f"PARTE A  C={C} V={V}: perfil das rotas ótimas e melhor custo por perfil")
    print("=" * 96)
    for s in seeds:
        inst = build_instance(C, V, s)
        rotas = todas_as_rotas(inst, V > 1)
        cstar = min(rotas.values())
        otimas = [k for k, c in rotas.items() if abs(c - cstar) < 1e-9]
        melhor = {}
        for k, c in rotas.items():
            p = perfil(k)
            if p not in melhor or c < melhor[p][0]:
                melhor[p] = (c, k)
        perfis_otimos = sorted({perfil(k) for k in otimas})
        print(f"\nsemente {s}:  C*={cstar:g}  perfis das rotas ótimas: {perfis_otimos}")
        print("   melhor custo por perfil: " + "   ".join(
            f"{p}: {c:g}" for p, (c, _) in sorted(melhor.items())))
        gs = GS_COST_D16.get((C, V), {}).get(s)
        if gs is None:
            continue
        casa = [p for p, (c, _) in melhor.items() if abs(c - gs) < 1e-9]
        ok = abs(gs - cstar) < 1e-9
        print(f"   fundamental (d=16) decodificou custo {gs:g}"
              f"  {'(ótimo)' if ok else '(subótimo)'}"
              f"  -> melhor custo do(s) perfil(is): {casa if casa else 'NENHUM'}")


def parte_b(C=4, V=2, d=8, seed=42, max_configs=300_000):
    inst = build_instance(C, V, seed)
    par = make_params(inst, V, "sugerida", 0.0)
    ops = H.build_modular_operators(d, inst.M, inst.R, HBAR)
    print("\n" + "=" * 96)
    print(f"PARTE B  C={C} V={V} d={d}: overhead = <H>(nó) - custo, por perfil "
          f"(configurações válidas)")
    print("=" * 96)
    tab = H.node_energy_table(inst, ops, par, max_configs=max_configs)
    en = tab["energies"]
    grupos = defaultdict(list)
    for slots in itertools.product(range(inst.M * inst.R), repeat=C):
        if not _is_valid(slots, inst, V > 1):
            continue
        rotas = routes_from_slots(slots, inst)
        grupos[perfil(rotas)].append(float(en[slots]) - H.routes_cost(rotas, inst.D))
    print(f"{'perfil':>14}{'n_config':>10}{'mínimo':>10}{'mediana':>10}{'máximo':>10}")
    medianas = {}
    for p, v in sorted(grupos.items()):
        medianas[p] = float(np.median(v))
        print(f"{str(p):>14}{len(v):>10}{min(v):>10.2f}{medianas[p]:>10.2f}"
              f"{max(v):>10.2f}")
    if len(medianas) > 1:
        print(f"\ndiferença entre as medianas dos perfis: "
              f"{max(medianas.values()) - min(medianas.values()):.2f}"
              "  (compare com a diferença de custo entre rotas, 2 a 14)")


def parte_c(C=4, V=2, ds=(8, 12, 16), n_amostra=8, seed=42, rng_seed=0):
    """Overhead por perfil, TERMO A TERMO, numa amostra de configurações
    válidas (não percorre as (M*R)^N configurações: leva segundos a poucos
    minutos mesmo em d=16). Responde: (1) qual termo cria a diferença entre
    perfis; (2) como a diferença muda com d."""
    inst = build_instance(C, V, seed)
    par = make_params(inst, V, "sugerida", 0.0)
    rng = np.random.default_rng(rng_seed)
    por_perfil = defaultdict(list)
    for slots in itertools.product(range(inst.M * inst.R), repeat=C):
        if _is_valid(slots, inst, V > 1):
            por_perfil[perfil(routes_from_slots(slots, inst))].append(slots)
    termos = ("dist-custo", "col", "gap", "vehicle", "disc", "total")
    for d in ds:
        ops = H.build_modular_operators(d, inst.M, inst.R, HBAR)
        kets = {(v, r): H.node_state(ops, v, r)
                for v in range(inst.M) for r in range(inst.R)}
        print("\n" + "=" * 96)
        print(f"PARTE C  C={C} V={V} d={d}: overhead por termo e perfil "
              f"(média de até {n_amostra} configurações válidas)")
        print("=" * 96)
        print(f"{'perfil':>12}{'n':>4}" + "".join(f"{t:>12}" for t in termos))
        medias = {}
        for p, lista in sorted(por_perfil.items()):
            idx = rng.choice(len(lista), size=min(n_amostra, len(lista)),
                             replace=False)
            acc = defaultdict(list)
            for i in idx:
                slots = lista[int(i)]
                psi = None
                for s in slots:
                    v, r = divmod(s, inst.R)
                    psi = (kets[(v, r)] if psi is None
                           else np.multiply.outer(psi, kets[(v, r)]))
                e = H.energy_components(psi, inst, ops, par)
                custo = H.routes_cost(routes_from_slots(slots, inst), inst.D)
                acc["dist-custo"].append(float(e["dist"]) - custo)
                for k in ("col", "gap", "vehicle", "disc"):
                    acc[k].append(float(e[k]))
                acc["total"].append(float(e["encoding"]) - custo)
            medias[p] = {k: float(np.mean(v)) for k, v in acc.items()}
            print(f"{str(p):>12}{len(idx):>4}"
                  + "".join(f"{medias[p][t]:>12.2f}" for t in termos))
        if len(medias) > 1:
            ps = sorted(medias)
            print(f"{'diferença':>16}" + "".join(
                f"{medias[ps[0]][t] - medias[ps[-1]][t]:>12.2f}" for t in termos)
                + f"   ({ps[0]} menos {ps[-1]})")


class _Tee:
    """Escreve na tela e no arquivo ao mesmo tempo (o arquivo é esvaziado a
    cada escrita, então enche durante a execução mesmo se ela for interrompida)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, texto):
        for st in self.streams:
            st.write(texto)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def saida(nome):
    """Tudo o que for impresso dentro do bloco vai para a tela E para `nome`
    (UTF-8, na pasta de execução; sobrescreve o arquivo anterior)."""
    with open(nome, "w", encoding="utf-8") as fh:
        antigo = sys.stdout
        sys.stdout = _Tee(antigo, fh)
        try:
            yield
        finally:
            sys.stdout = antigo
    print(f"saída gravada em {nome}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0].lower() == "c":
        ds = tuple(int(x) for x in args[1:]) or (8, 12, 16)
        with saida("resultado_c.txt"):
            parte_c(4, 2, ds=ds)
    else:
        with saida("resultado_a.txt"):
            parte_a(4, 2)
            parte_a(4, 1)
        if args:
            with saida("resultado_b.txt"):
                for arg in args:
                    parte_b(4, 2, d=int(arg))