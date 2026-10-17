"""
exp_dmin_limiar.py — d_min em função do critério, lido do trace da v3.

Não roda simulação: só lê cutoff_minimo_trace.csv (gerado por
exp_cutoff_minimo.py v3, com as colunas p_valid, p_opt_valid, chance_valid).

Dois tipos de limiar sobre P(ótima | válida) = p_opt_valid / p_valid
-------------------------------------------------------------------
  relativo  k     P(ótima|válida) / chance_valid >= k       (ex.: 2  3)
  absoluto  a<t>  P(ótima|válida) >= t                      (ex.: a0.3  a0.5)

O relativo fica mais fácil quanto maior o espaço de busca, porque o acaso
(chance_valid) encolhe: 4,5x o acaso em C=5 V=1 ainda é só ~7% de acerto. Para
comparar C diferentes use o absoluto. Reporte d_min sempre com o limiar.

  ganho     s<f>  (P - acaso) / (1 - acaso) >= f                (ex.: s0.5  s0.8)
                  Vai de 0 (acaso) a 1 (certeza). Não tem o teto do relativo:
                  com acaso = 1/3 o limiar "3x acaso" exige P = 1 e é
                  inatingível, o que aparece como "> 20" sem significado.
                  Para comparar C diferentes, é o mais justo dos três.

Para cada (C, V) e cada limiar procura o menor d tal que o critério vale nesse
d e em todos os cutoffs testados depois, com pelo menos `confirm` depois (e a
melhor rota válida é ótima, se need_top=True). Sem isso imprime "> lb": o maior
d que falhou (cota inferior, não medida).

Colunas
-------
  d/MR          MR = V * R, R = C se V=1 e C-V+1 se V>1 (instâncias do
                experimento: demandas 1, capacidade C)
  P(val)        probabilidade de a configuração ser válida no d_min (o resto
                dos disparos é descartado)
  P(ótima)      P(val) * P(ótima|válida): chance de um disparo bruto dar a rota
                ótima

Uso
---
    python exp_dmin_limiar.py                               # limiares padrão
    python exp_dmin_limiar.py cutoff_minimo_trace.csv 2 3 a0.3 a0.5
"""
import csv
import os
import sys
from collections import defaultdict

NAN = float("nan")

DEFAULT_SPECS = (("ratio", 2.0), ("ratio", 3.0),
                 ("cond", 0.3), ("cond", 0.5), ("cond", 0.8),
                 ("skill", 0.5), ("skill", 0.8))


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
            skill = (cond - ch) / (1.0 - ch) if 0 <= ch < 1 else NAN
            series[(int(r["C"]), int(r["V"]))].append(dict(
                d=int(r["d"]), pv=pv, cond=cond, ratio=ratio, skill=skill,
                chance=ch,
                top=int(_f(r.get("top_valid_ok", 0)) or 0),
                col=_f(r.get("col_medio"))))
    for k in series:
        series[k].sort(key=lambda s: s["d"])
    return series


def stable_d(series, metric, thr, need_top=True, confirm=2):
    flags = [(s["d"], bool(s[metric] >= thr and (s["top"] == 1 or not need_top)),
              s) for s in series]
    fails = [d for d, ok, _ in flags if not ok]
    lb = max(fails) if fails else None
    for i, (d, ok, s) in enumerate(flags):
        later = flags[i + 1:]
        if ok and len(later) >= confirm and all(o for _, o, _ in later):
            return d, lb, s
    return None, lb, None


def label(spec):
    kind, v = spec
    if kind == "ratio":
        return f"{v:g}x acaso"
    if kind == "skill":
        return f"ganho normalizado >= {v:g}"
    return f"P(ótima|válida) >= {v:g}"


def parse_specs(tokens):
    specs = []
    for t in tokens:
        if t.startswith("a"):
            specs.append(("cond", float(t[1:])))
        elif t.startswith("s"):
            specs.append(("skill", float(t[1:])))
        else:
            specs.append(("ratio", float(t)))
    return tuple(specs)


def main(path="cutoff_minimo_trace.csv", specs=DEFAULT_SPECS, need_top=True,
         confirm=2):
    series = load(path)
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = f"result_{base}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        if not series:
            msg = "Nenhuma linha com p_valid no trace: rode a v3 de exp_cutoff_minimo.py."
            print(msg, file=f)
            print(msg)
            return
        print("=" * 104, file=f)
        print("d_min por limiar (confirmado nos cutoffs seguintes)", file=f)
        print("=" * 104, file=f)
        for (C, V), ser in sorted(series.items()):
            if V >= C:
                continue
            R = C if V == 1 else C - V + 1
            MR = V * R
            print(f"\nC={C} V={V}  MR={MR}  acaso P(ótima|válida)={ser[0]['chance']:.4f}", file=f)
            print("  P(ótima|válida) por d: " + "  ".join(
                f"{s['d']}:{s['cond']:.3f}" for s in ser), file=f)
            print("  razão ao acaso por d : " + "  ".join(
                f"{s['d']}:{s['ratio']:.2f}" for s in ser), file=f)
            print("  ganho normalizado/d  : " + "  ".join(
                f"{s['d']}:{s['skill']:.2f}" for s in ser), file=f)
            for spec in specs:
                d, lb, s = stable_d(ser, spec[0], spec[1], need_top, confirm)
                name = f"{label(spec):<26}"
                if d is not None:
                    print(f"  {name} d_min={d:<3} d/MR={d / MR:5.2f}   "
                        f"P(val)={s['pv']:.3f}  P(ótima)={s['pv'] * s['cond']:.4f}  "
                        f"col={s['col']:.2f}", file=f)
                elif lb is not None:
                    print(f"  {name} d_min >{lb:<3} d/MR >{lb / MR:4.2f}   (censurado)", file=f)
                else:
                    print(f"  {name} sem cutoffs suficientes para confirmar", file=f)
    print(f"saída gravada em {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    path = "cutoff_minimo_trace.csv"
    if args and args[0].lower().endswith(".csv"):
        path, args = args[0], args[1:]
    main(path, parse_specs(args) or DEFAULT_SPECS)