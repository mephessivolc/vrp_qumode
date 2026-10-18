"""
exp_run.py — ÚNICO ponto de execução dos testes.

Todo teste sai daqui: ligue ou desligue os blocos em EXECUTAR e rode este
arquivo. Nada é lançado por linha de comando com argumentos. Cada bloco grava
seus resultados em arquivo (nomes abaixo) e também mostra na tela.

bloco                  arquivo(s) gerado(s)                          duração
vqe_ponte              resultado_vqe_ponte.txt                       segundos
vqe_pequeno            resultado_vqe.txt, vqe_resultados.csv,        ~10-30 min
                       vqe_iterados_<tag>.csv
fatorado_verificacao   resultado_f_verificacao.txt                   segundos
fatorado_fantasma      resultado_f_fantasma.txt                      segundos
fatorado_validacao     resultado_f_validacao.txt                     minutos
fatorado_decomposicao_exata  resultado_f_decomposicao.txt            segundos
fatorado_distinta      resultado_f_distinta_overhead.txt,            segundos
                       resultado_f_distinta_decisao.txt
fatorado_campo_medio   resultado_f_campo_medio.txt                   segundos
fatorado_overhead      resultado_f_overhead.txt                      segundos
fatorado_decisao       resultado_f_decisao.txt                       segundos
fatorado_pecas         resultado_f_pecas_lacuna.txt                  segundos
d_fidelidade_dos_nos   resultado_d.txt                               segundos
a_perfis               resultado_a.txt                               segundos
c_soma                 resultado_c.txt                               minutos
c_predecessor          resultado_c_predecessor.txt                   minutos
sementes_v4            seed{s}.csv, seed{s}_trace.csv                ~20 min
c4_grande              c4_grande.csv, c4_grande_trace.csv            ~15 min
scan_gap               resultado_gap.txt, scan_gap.csv               ~25 min

Os blocos "fatorado_*" (caminho B) dependem de qumodes/product_energy.py e de
exp_fatorado.py. A verificação roda SEMPRE primeiro; se falhar, os outros
blocos fatorados são pulados.

Cada execução sobrescreve os arquivos com o mesmo nome: renomeie antes os que
quiser guardar. Os blocos já executados e analisados ficam desligados.
"""

EXECUTAR = {
    # caminho B: estados-produto sem tensor d^N
    # VQE: ponte SF -> Hamiltoniano em operadores (NOVO)
    "vqe_ponte": True,                    # roda sempre antes do VQE; se falhar, o VQE é pulado
    "vqe_pequeno": True,                  # VQE contra o fundamental exato, C=3, d=8
    # caminho B (estados-produto) — já analisados
    "fatorado_verificacao": False,
    "fatorado_fantasma": False,
    "fatorado_validacao": False,
    "fatorado_decomposicao_exata": False, # já analisado
    "fatorado_distinta": False,           # já analisado
    "fatorado_campo_medio": False,        # já analisado
    "fatorado_overhead": False,         # já analisado
    "fatorado_decisao": False,          # já analisado
    "fatorado_pecas": False,            # já analisado
    # já executados e analisados
    "d_fidelidade_dos_nos": False,
    "a_perfis": False,
    "c_soma": False,
    "c_predecessor": False,
    "sementes_v4": False,
    "c4_grande": False,
    "scan_gap": False,
}


def main():
    from exp_perfil_rotas import parte_a, parte_c, parte_d, saida

    # --- VQE ---------------------------------------------------------------------
    if EXECUTAR["vqe_ponte"] or EXECUTAR["vqe_pequeno"]:
        import exp_vqe as EV

        with saida("resultado_vqe_ponte.txt"):
            ponte_ok = EV.verificar_ponte()
        if not ponte_ok:
            print("\nA ponte FALHOU: o VQE foi pulado.")
        elif EXECUTAR["vqe_pequeno"]:
            with saida("resultado_vqe.txt"):
                # R=2: caso em que o fundamental já pesa na rota ótima
                EV.vqe_vs_exato(C=3, V=2, seed=42, cutoff=8, layers=2, maxiter=100)
                # R=3, perfil único
                EV.vqe_vs_exato(C=3, V=1, seed=42, cutoff=8, layers=2, maxiter=100)

    # --- caminho B ----------------------------------------------------------
    blocos_fatorados = ("fatorado_verificacao", "fatorado_fantasma",
                        "fatorado_validacao", "fatorado_decomposicao_exata",
                        "fatorado_distinta", "fatorado_campo_medio",
                        "fatorado_overhead", "fatorado_decisao", "fatorado_pecas")
    if any(EXECUTAR[k] for k in blocos_fatorados):
        import exp_fatorado as F

        with saida("resultado_f_verificacao.txt"):
            ok = F.verificar()
        if not ok:
            print("\nA verificação FALHOU: os demais blocos fatorados foram pulados.")
        else:
            if EXECUTAR["fatorado_fantasma"]:
                with saida("resultado_f_fantasma.txt"):
                    F.decomposicao_fantasma()
            if EXECUTAR["fatorado_validacao"]:
                formas_v = ("soma", "predecessor+distinta")
                with saida("resultado_f_validacao.txt"):
                    # V=2, sementes NOVAS (6 a 15): o diagnóstico usou 42 e 1 a 5
                    F.decisao_por_semente(C=4, V=2, seeds=tuple(range(6, 16)),
                                          formas=formas_v)
                    # V=1: a variante não pode piorar o caso que funciona
                    F.decisao_por_semente(C=4, V=1, seeds=(42, 1, 2, 3, 4, 5),
                                          formas=formas_v)
                    # C=5, V=2 e V=3: o fundamental só foi testado até d=10
                    F.decisao_por_semente(C=5, V=2, seeds=(42, 1, 2, 3, 4),
                                          ds=(8, 16, 32, 64), formas=formas_v)
                    F.decisao_por_semente(C=5, V=3, seeds=(42, 1, 2, 3, 4),
                                          ds=(8, 16, 32, 64), formas=formas_v)
            if EXECUTAR["fatorado_decomposicao_exata"]:
                with saida("resultado_f_decomposicao.txt"):
                    F.decomposicao_exata_da_lacuna()
            if EXECUTAR["fatorado_distinta"]:
                formas_d = ("soma", "soma+distinta", "predecessor+distinta")
                with saida("resultado_f_distinta_overhead.txt"):
                    F.overhead_por_perfil(ds=(8, 16, 32, 64), formas=formas_d)
                with saida("resultado_f_distinta_decisao.txt"):
                    F.decisao_por_semente(formas=formas_d)
            if EXECUTAR["fatorado_campo_medio"]:
                with saida("resultado_f_campo_medio.txt"):
                    F.campo_medio_por_perfil()
            if EXECUTAR["fatorado_overhead"]:
                with saida("resultado_f_overhead.txt"):
                    F.overhead_por_perfil()
            if EXECUTAR["fatorado_decisao"]:
                with saida("resultado_f_decisao.txt"):
                    F.decisao_por_semente()
            if EXECUTAR["fatorado_pecas"]:
                with saida("resultado_f_pecas_lacuna.txt"):
                    F.pecas_da_lacuna()

    # --- blocos anteriores ----------------------------------------------------
    if EXECUTAR["d_fidelidade_dos_nos"]:
        with saida("resultado_d.txt"):
            parte_d(C=4, V=2, ds=(8, 12, 16, 24, 32, 48, 64))

    if EXECUTAR["a_perfis"]:
        with saida("resultado_a.txt"):
            parte_a(4, 2)
            parte_a(4, 1)

    if EXECUTAR["c_soma"]:
        with saida("resultado_c.txt"):
            parte_c(4, 2, ds=(8, 12, 16), gap_form="soma")

    if EXECUTAR["c_predecessor"]:
        with saida("resultado_c_predecessor.txt"):
            parte_c(4, 2, ds=(8, 12, 16), gap_form="predecessor")

    if EXECUTAR["sementes_v4"] or EXECUTAR["c4_grande"]:
        from exp_cutoff_minimo import main as cutoff_main

        if EXECUTAR["c4_grande"]:
            cutoff_main(cities=[4], vehicles=[1, 3], max_dim=170_000,
                        out="c4_grande.csv", trace_out="c4_grande_trace.csv")

        if EXECUTAR["sementes_v4"]:
            for s in range(1, 6):
                cutoff_main(cities=[4], vehicles=[1, 2], cutoffs=(5, 8, 12, 16),
                            seed=s, out=f"seed{s}.csv",
                            trace_out=f"seed{s}_trace.csv")

    if EXECUTAR["scan_gap"]:
        from exp_scan_gap import run as scan_gap_run

        with saida("resultado_gap.txt"):
            scan_gap_run()


if __name__ == "__main__":
    main()