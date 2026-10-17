from exp_cutoff_minimo import main as cutoff_main

# 1) fechar C=4, V=1 e V=3 (d=18 e 20 precisam de max_dim maior)
cutoff_main(cities=[4], vehicles=[1, 3], max_dim=170_000,
            out="c4_grande.csv", trace_out="c4_grande_trace.csv")

# 2) cinco instâncias para V=1 (funciona) e V=2 (falha)
for s in range(1, 6):
    cutoff_main(cities=[4], vehicles=[1, 2], cutoffs=(5, 8, 12, 16), seed=s,
                out=f"seed{s}.csv", trace_out=f"seed{s}_trace.csv")