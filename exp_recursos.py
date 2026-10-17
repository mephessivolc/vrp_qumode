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

Origem de d_min (coluna d, e d_min_source no CSV)
-------------------------------------------------
  MEDIDO      d_gs encontrado:                      d      -> razão exata
  CENSURADO   só há d_gs > lb (limite de dimensão): >=lb+1 -> razão é COTA INFERIOR
  ESTIMADO    caso ausente do CSV: --default-cutoff   (~d)

Como ler a razão mod/E1-VRP com cotas inferiores
------------------------------------------------
  MEDIDO     < 1   a modular custa menos que a extensão direta da Etapa 1
  CENSURADO  >= 1  a modular certamente custa MAIS (a razão real só pode ser maior)
  CENSURADO  < 1   INDETERMINADO: o d_gs real pode estar acima da cota
Só pontos MEDIDOS sustentam uma afirmação de cruzamento.

Uso
---
    python exp_recursos.py
Rode exp_cutoff_minimo.py antes para ter d_min medido.
"""
import csv

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


def _num(x):
    return x not in (None, "", "None")


def load_dmin(path):
    """(C, V) -> ("med", d_gs)  ou  ("lb", d_gs_lb)  [significa d_gs > d_gs_lb]."""
    out = {}
    try:
        fh = open(path, newline="", encoding="utf-8")
    except FileNotFoundError:
        return out
    with fh:
        for row in csv.DictReader(fh):
            key = (int(row["C"]), int(row["V"]))
            if "d_gs_stable" in row:          # CSV v2: só vale o d confirmado
                if _num(row["d_gs_stable"]):
                    out[key] = ("med", int(float(row["d_gs_stable"])))
                elif _num(row.get("d_gs_lb")):
                    out[key] = ("lb", int(float(row["d_gs_lb"])))
            elif _num(row.get("d_gs")):       # CSV v1
                out[key] = ("med", int(float(row["d_gs"])))
            elif _num(row.get("d_gs_lb")):
                out[key] = ("lb", int(float(row["d_gs_lb"])))
    return out


def main(
        cities=[3, 4, 5, 6, 7],
        vehicles=[1, 2, 3],
        from_csv="cutoff_minimo.csv",
        default_cutoff=8,
        out="recursos.csv"):

    dmin = load_dmin(from_csv) if from_csv else {}

    print("=" * 104)
    print("RECURSOS POR FORMULAÇÃO   (d: n = medido, >=n = cota inferior, ~n = estimado)")
    print("=" * 104)
    print(f"{'C':>3}{'V':>3}{'R':>3}{'M*R':>5} | {'QUBO 2^(N²)':>14}"
          f"{'E1 N^N':>12}{'E1-VRP (MR)^N':>15}{'modular d^N':>14}{'d':>5}"
          f" | {'mod/E1-VRP':>12}{'memória mod':>14}")

    rows = []
    for C in cities:
        for V in vehicles:
            if V > C:
                continue
            R = positions(C, V)
            MR = V * R
            info = dmin.get((C, V))
            if info is None:
                d, src, pref = default_cutoff, "ESTIMADO", "~"
            elif info[0] == "med":
                d, src, pref = info[1], "MEDIDO", ""
            else:
                d, src, pref = info[1] + 1, "CENSURADO", ">="

            dim_qubo = 2 ** (C * C)
            dim_e1 = C ** C
            dim_e1vrp = MR ** C
            dim_mod = d ** C
            fator = dim_mod / dim_e1vrp

            print(f"{C:>3}{V:>3}{R:>3}{MR:>5} | {human(dim_qubo):>14}"
                  f"{human(dim_e1):>12}{human(dim_e1vrp):>15}"
                  f"{human(dim_mod):>14}{pref + str(d):>5}"
                  f" | {pref + format(fator, '.2f') + 'x':>12}"
                  f"{pref + memory(dim_mod):>14}")
            rows.append(dict(C=C, V=V, R=R, MR=MR, d_min=d,
                             d_min_is_lower_bound=int(src == "CENSURADO"),
                             dim_qubo=dim_qubo, dim_e1=dim_e1,
                             dim_e1vrp=dim_e1vrp, dim_modular=dim_mod,
                             ratio_mod_over_e1vrp=fator,
                             d_min_source=src))

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def lab(rs):
        return ", ".join(f"C={r['C']},V={r['V']}" for r in rs)

    meds = [r for r in rows if r["d_min_source"] == "MEDIDO"]
    cens = [r for r in rows if r["d_min_source"] == "CENSURADO"]
    est = [r for r in rows if r["d_min_source"] == "ESTIMADO"]
    cross = [r for r in meds if r["ratio_mod_over_e1vrp"] < 1.0]
    pior = [r for r in cens if r["ratio_mod_over_e1vrp"] >= 1.0]
    indet = [r for r in cens if r["ratio_mod_over_e1vrp"] < 1.0]

    print()
    print("Leitura da coluna de razão:")
    print("  MEDIDO     < 1   a modular custa menos que a extensão direta da Etapa 1")
    print("  CENSURADO  >= 1  a modular certamente custa mais (razão real só pode subir)")
    print("  CENSURADO  < 1   indeterminado: d_gs real pode estar acima da cota")
    print()
    print(f"  Cruzamento medido (razão < 1): {lab(cross) if cross else 'nenhum'}")
    if pior:
        print(f"  Modular custa mais (cota inferior já >= 1): {lab(pior)}")
    if indet:
        print(f"  Indeterminado (censurado, cota ainda < 1): {lab(indet)}")
    if est:
        print(f"\n  ATENÇÃO: d_min foi ARBITRADO em {default_cutoff} para: {lab(est)}.")
        print("  Esses casos não estão no CSV; rode exp_cutoff_minimo.py para eles.")
    print(f"\nsalvo em {out}")


if __name__ == "__main__":
    main(
        from_csv="cutoff_minimo.csv"
    )