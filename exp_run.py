"""
exp_run.py — ÚNICO ponto de execução dos testes.

Todo teste sai daqui: ligue ou desligue os blocos em EXECUTAR e rode este
arquivo. Nada é lançado por linha de comando com argumentos. Cada bloco grava
seus resultados em arquivo (nomes na tabela abaixo) e também mostra na tela.

bloco                     arquivo(s) gerado(s)                       duração
d_fidelidade_dos_nos      resultado_d.txt                            segundos
a_perfis                  resultado_a.txt                            segundos
c_soma                    resultado_c.txt                            minutos
c_predecessor             resultado_c_predecessor.txt                minutos
sementes_v4               seed{s}.csv, seed{s}_trace.csv             ~20 min
c4_grande                 c4_grande.csv, c4_grande_trace.csv         ~15 min
scan_gap                  resultado_gap.txt, scan_gap.csv            ~25 min

Cada execução sobrescreve os arquivos com o mesmo nome: renomeie antes os que
quiser guardar. Os blocos já executados e analisados ficam desligados.
"""

EXECUTAR = {
    "d_fidelidade_dos_nos": True,
    "a_perfis": False,
    "c_soma": False,
    "c_predecessor": False,
    "sementes_v4": False,
    "c4_grande": False,
    "scan_gap": False,
}


def main():
    from exp_perfil_rotas import parte_a, parte_c, parte_d, saida

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
        from exp_perfil_rotas import saida as _saida
        from exp_scan_gap import run as scan_gap_run

        with _saida("resultado_gap.txt"):
            scan_gap_run()


if __name__ == "__main__":
    main()