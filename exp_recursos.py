"""
exp_recursos.py — tabela comparativa de recursos entre as formulações.

As quatro codificações
----------------------
QUBO        qumodes usados como qubits, um por par (cidade, posição):
            N^2 modos, 2 níveis cada  ->  dim = 2^(N^2)

Etapa 1     um qumode por cidade, nível de Fock = posição na rota:
            N modos, cutoff = N       ->  dim = N^N

E1-VRP      extensão direta da Etapa 1 ao VRP, com o nível de Fock rotulando
            o par (veículo, posição):
            N modos, cutoff = M*R     ->  dim = (M*R)^N
            É a alternativa honesta contra a qual a modular deve ser medida.

Modular     posição em theta = 2pi x/a, veículo em phi = 2pi p/b:
            N modos, cutoff = d_min   ->  dim = d_min^N
            d_min é MEDIDO por exp_cutoff_minimo.py, não arbitrado.

Por que E1-VRP é a comparação certa
-----------------------------------
QUBO e Etapa 1 estabelecem a progressão histórica da tese (N^2 -> N modos).
Mas a pergunta viva é outra: já que a modular mantém N modos e paga em cutoff,
ela compensa? A extensão direta da Etapa 1 usaria cutoff = M*R EXATAMENTE, sem
margem de resolução. A modular precisa de margem. A questão é se (M*R)^N cresce
mais rápido que d_min^N — ou seja, se d_min/(M*R) cai com o tamanho.

Uso
---
    python exp_recursos.py                          # d_min estimado
    python exp_recursos.py --from-csv cutoff_minimo.csv   # d_min MEDIDO

Sem o CSV o script usa `--default-cutoff` como d_min e MARCA a tabela como
estimada. Rode exp_cutoff_minimo.py antes para ter a versão medida.
"""
import argparse
import csv
import math

import numpy as np

import qumodes.hamiltonian as H


def human(n):
    """Dimensão legível."""
    if n < 1e6:
        return f"{n:,}"
    return f"{n:.3e}"


def memory(n, bytes_per=16):
    b = n * bytes_per
    for u in ["B", "kB", "MB", "GB", "TB", "PB", "EB"]:
        if b < 1024:
            return f"{b:.1f}{u}"
        b /= 1024
    return f"{b:.1e}EB"


def positions(C, V):
    req = (V > 1)
    return H.auto_positions(C, V, np.ones(C), np.full(V, float(C)),
                            require_all_vehicles=req)


def load_dmin(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("d_gs") not in (None, "", "None"):
                out[(int(row["C"]), int(row["V"]))] = int(float(row["d_gs"]))
    return out


def main(
        cities=[3, 4, 5, 6, 7],
        vehicles=[1, 2, 3],
        from_csv="cutoff_minimo.csv",
        default_cutoff=8,
        out="recursos,csv"

):
    dmin = load_dmin(from_csv) if from_csv else {}
    origem = "MEDIDO" if dmin else "ESTIMADO"

    print("=" * 100)
    print(f"RECURSOS POR FORMULAÇÃO   (d_min: {origem})")
    print("=" * 100)
    print(f"{'C':>3}{'V':>3}{'R':>3}{'M*R':>5} | {'QUBO 2^(N²)':>14}"
          f"{'E1 N^N':>12}{'E1-VRP (MR)^N':>15}{'modular d^N':>14}{'d':>4}"
          f" | {'mod/E1-VRP':>11}{'memória mod':>13}")

    rows = []
    for C in cities:
        for V in vehicles:
            if V > C:
                continue
            R = positions(C, V)
            MR = V * R
            d = dmin.get((C, V), default_cutoff)
            dim_qubo = 2 ** (C * C)
            dim_e1 = C ** C
            dim_e1vrp = MR ** C
            dim_mod = d ** C
            fator = dim_mod / dim_e1vrp
            print(f"{C:>3}{V:>3}{R:>3}{MR:>5} | {human(dim_qubo):>14}"
                  f"{human(dim_e1):>12}{human(dim_e1vrp):>15}"
                  f"{human(dim_mod):>14}{d:>4}"
                  f" | {fator:>10.2f}x{memory(dim_mod):>13}")
            rows.append(dict(C=C, V=V, R=R, MR=MR, d_min=d,
                             dim_qubo=dim_qubo, dim_e1=dim_e1,
                             dim_e1vrp=dim_e1vrp, dim_modular=dim_mod,
                             ratio_mod_over_e1vrp=fator,
                             d_min_source=origem))

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print()
    print("Leitura da última coluna de razão:")
    print("  > 1  a modular custa mais que a extensão direta da Etapa 1")
    print("  < 1  a modular custa menos — e aí o passo modular tem ganho de")
    print("       recursos além de estender ao VRP")
    cross = [r for r in rows if r["ratio_mod_over_e1vrp"] < 1.0]
    if cross:
        print(f"\n  Cruzamento a partir de: " +
              ", ".join(f"C={r['C']},V={r['V']}" for r in cross[:6]))
    else:
        print("\n  Sem cruzamento nesta grade: a modular custa mais em todos os casos.")
    if origem == "ESTIMADO":
        print("\n  ATENÇÃO: d_min foi ARBITRADO em "
              f"{default_cutoff}. Rode exp_cutoff_minimo.py e repita com")
        print("  --from-csv cutoff_minimo.csv antes de levar esta tabela para a tese.")
    print(f"\nsalvo em {out}")


if __name__ == "__main__":
    main(
        from_csv="cutoff_minimo.csv"
    )