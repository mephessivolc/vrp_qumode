from exp_perfil_rotas import parte_a, parte_b, parte_c, saida

with saida("resultado_a.txt"):
    parte_a(4, 2)
    parte_a(4, 1)

with saida("resultado_b.txt"):
    parte_b(4, 2, d=8)

ds = (8, 12, 16)
with saida("resultado_c.txt"):
    parte_c(4, 2, ds=ds)