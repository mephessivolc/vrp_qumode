"""
exp_dmin_limiar.py — d_min em função do critério, lido do trace da v3.

Não roda simulação: só lê cutoff_minimo_trace.csv (gerado por
exp_cutoff_minimo.py v3, com as colunas p_valid, p_opt_valid, chance_valid).

Para cada (C, V) e cada limiar k, procura o menor d tal que
    P(ótima | válida) / chance_valid >= k     (e a melhor rota válida é ótima,
                                               se need_top=True)
vale nesse d e em todos os cutoffs testados depois, com pelo menos `confirm`
cutoffs depois. Se não houver, imprime ">lb": o maior d que falhou (cota
inferior, não medida).

Por que existe: d_min NÃO é um limiar nítido. A massa ótima cresce de forma
contínua com d, então o valor de d_min depende de k. Reporte d_min sempre com o
k usado e mostre a sensibilidade (esta tabela).

Colunas
-------
  razão por d   P(ótima|válida) / chance_valid (1 = acaso)
  P(val)        probabilidade de a configuração ser válida no d_min (custo de
                descartar disparos inválidos)
  d/MR          MR = V * R, com R = C se V=1 e C-V+1 se V>1 (instâncias do
                experimento: demandas 1, capacidade C).

Uso
---
    python exp_dmin_limiar.py                          # cutoff_minimo_trace.csv
    python exp_dmin_limiar.py trace.csv 1.5 2 3 5
"""
import csv
import sys
from collections import defaultdict

NAN = float("nan")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def load(path):
    series = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("kind") != "fundamental" or r.get("p_valid", "") == "":
                continue
            pv = _f(r["p_valid"])
            po = _f(r["p_opt_valid"])
            ch = _f(r["chance_valid"])
            cond = po / pv if pv > 0 else NAN
            ratio = cond / ch if ch > 0 else NAN
            series[(int(r["C"]), int(r["V"]))].append(dict(
                d=int(r["d"]), pv=pv, ratio=ratio, chance=ch,
                top=int(_f(r.get("top_valid_ok", 0)) or 0),
                col=_f(r.get("col_medio"))))
    for k in series:
        series[k].sort(key=lambda s: s["d"])
    return series


def stable_d(series, k, need_top=True, confirm=2):
    flags = [(s["d"], bool(s["ratio"] >= k and (s["top"] == 1 or not need_top)),
              s) for s in series]
    fails = [d for d, ok, _ in flags if not ok]
    lb = max(fails) if fails else None
    for i, (d, ok, s) in enumerate(flags):
        later = flags[i + 1:]
        if ok and len(later) >= confirm and all(o for _, o, _ in later):
            return d, lb, s
    return None, lb, None


def main(path="cutoff_minimo_trace.csv", ks=(1.5, 2.0, 3.0, 5.0),
         need_top=True, confirm=2):
    series = load(path)
    if not series:
        print("Nenhuma linha com p_valid no trace: rode a v3 de exp_cutoff_minimo.py.")
        return
    print("=" * 100)
    print("d_min por limiar k:  P(ótima|válida) >= k * acaso  (confirmado nos cutoffs seguintes)")
    print("=" * 100)
    for (C, V), ser in sorted(series.items()):
        R = C if V == 1 else C - V + 1
        MR = V * R
        chance = ser[0]["chance"]
        print(f"\nC={C} V={V}  MR={MR}  acaso(P(ótima|válida))={chance:.4f}")
        print("  razão por d: " + "  ".join(
            f"{s['d']}:{s['ratio']:.2f}" for s in ser))
        for k in ks:
            d, lb, s = stable_d(ser, k, need_top, confirm)
            if d is not None:
                print(f"  k={k:<4} d_min={d:<3} d/MR={d / MR:5.2f}   "
                      f"P(val)={s['pv']:.3f}   col={s['col']:.2f}")
            elif lb is not None:
                print(f"  k={k:<4} d_min >{lb:<3} d/MR >{lb / MR:4.2f}   (censurado)")
            else:
                print(f"  k={k:<4} sem cutoffs suficientes para confirmar")


if __name__ == "__main__":
    args = sys.argv[1:]
    path = args[0] if args else "cutoff_minimo_trace.csv"
    ks = tuple(float(x) for x in args[1:]) or (1.5, 2.0, 3.0, 5.0)
    main(path, ks)