"""
exp_run.py — ÚNICO ponto de execução dos testes.

Todo teste sai daqui: ligue ou desligue os blocos em EXECUTAR e rode este
arquivo. Nada é lançado por linha de comando com argumentos.

TODO arquivo gerado passa pelo PathManager (path.py, na raiz), em
PathManager("qumodes", <pasta>). A raiz de resultados é a do PathManager
(variável OUTPUT_DIR, ou /home-ext/clovis/results se existir, ou results/ ao lado
do path.py). Cada bloco usa a sua pasta (nomes em minúsculas):

bloco                       pasta (dentro de qumodes/)       execuções
vqe_ponte                   vqe_ponte                         gate do VQE
tese_instancias             <pasta de cada bloco>             gate das tuplas (C, V, L)
energia_cutoff              vqe_cutoff                        C=5 fixo; V=1 (TSP) e V=2 (VRP); d = 5, 7, ..., 29 (13 cutoffs cada)
valido_otimo                vqe_valido_otimo                  17 tuplas (C=3-6 x V=1-5, V<=C; L=2; cutoff=C+1)
grade_cv                    vqe_grade_cv                      12 tuplas (C=3-6 x V=1-3; L=2; cutoff=C*V); as de dimensão > DIM_MAX_CV são puladas
v2_c3_grade                 vqe_tese_v2                       90 (C=3; V=1,2; L=1-3; 5 cutoffs x 3 lr)
v2_c4                       vqe_tese_v2                       18 (C=4; V=1,2,3; L=1-3; lr 0,05 e 0,02)
v2_c5                       vqe_tese_v2                       12 (C=5; V=1,2,3; L=1-2; lr 0,05 e 0,02)

Os blocos v2 gravam TODOS os resultados na mesma pasta (vqe_tese_v2): summary.csv
único (uma linha por experimento), instancias.csv, runs/, arrays/, logs/, figures/.
Novidades v2 na tabela: fundamental exato decodificado (gs_*), rho inicial e final,
ganho do treino, P(acerto, 16 disparos), solução mais provável e critério "forte".
(Blocos abaixo = resultados anteriores, já analisados.)
tese_base                   vqe_tese_base                     6  (C=3; V=1,2; L=1,2,3)
tese_hiperparametros        vqe_tese_hp                       8  (V=1,2; L=2; base, it500, lr02, lr10)
tese_cutoff                 vqe_tese_cutoff                   10 (V=1,2; cutoff = C, C+2, C+5, C+7, C+9)
tese_c4                     vqe_tese_c4                       6  (C=4; V=1,2,3; L=1,2)

Capacidade: construída da demanda total (Q_VAL = "auto", FOLGA); λ_cap = 0 para
V = 1 e λ_cap > 0 para V > 1. O tempo de execução não é critério (simulação em
computador clássico).

Resultados da tese (por tupla (C, V, L) e conjunto de sobreparâmetros):
  summary.csv, instancias.csv, runs/*.json, arrays/*_state.npz, logs/*.log e
  figures/*_{original_graph,bf_route,vqe_route,convergence,decodificacao}.png

Blocos anteriores (diagnósticos já analisados) ficam desligados; as saídas deles
também vão para o PathManager (pastas cutoff_minimo, caminho_b, vqe_antigo).

Cada execução sobrescreve os arquivos com o mesmo nome (menos os summary.csv, que
acumulam linhas): renomeie antes o que quiser guardar. As estimativas de duração
são extrapolações, não medidas.
"""

# Capacidade de cada veículo (CVRP). "auto": construída da demanda total, a menor
# capacidade inteira >= max(maior demanda, ceil(FOLGA * demanda total / V)) para a
# qual existe solução viável (toda instância tem solução e a restrição de
# capacidade continua ativa). Um número fixa a capacidade (pode tornar a instância
# inviável). FOLGA próxima de 1 = capacidade apertada; maior = mais soluções viáveis.
# Capacidade dos veículos:
#   "desigual": DIFERENTE por veículo, com pelo menos uma solução garantida por
#               construção (partição de referência sorteada; Q_v = ceil(FOLGA * carga))
#   "auto":     igual para todos, construída da demanda total (pode inviabilizar tuplas)
#   número:     igual e fixa para todos
Q_VAL = "desigual"
FOLGA = 1.2

# Sobreparâmetros da tese: iterações, taxa de aprendizado (lr -> lr_final por
# cosseno) e acréscimo ao cutoff base (C=3: 8, C=4: 10).
HP_BASE = {"base": dict(max_iter=200, lr=0.05, lr_final=0.005, cutoff_extra=0)}
HP_VARIANTES = {
    "base": dict(max_iter=200, lr=0.05, lr_final=0.005, cutoff_extra=0),
    "it500": dict(max_iter=500, lr=0.05, lr_final=0.005, cutoff_extra=0),
    "lr02": dict(max_iter=500, lr=0.02, lr_final=0.002, cutoff_extra=0),
    "lr10": dict(max_iter=500, lr=0.10, lr_final=0.010, cutoff_extra=0),
}

# ---- energia exata contra o cutoff (hipótese dos pesquisadores) --------------------
# Menor energia de H (diagonalização exata sem matriz densa) para C=5, V=2 em
# cutoffs de 5 a 29, de 2 em 2. LIMITE É A MEMÓRIA: o script estima o uso antes de
# cada cutoff e PULA os que passarem de MEMORIA_MAX_GB. Ajuste ao RAM disponível
# (com R=4: d=21 ~ 7 GB, d=25 ~ 17 GB, d=29 ~ 36 GB).
PASTA_ENERGIA = "vqe_cutoff"
CUTOFFS_ENERGIA = tuple(range(5, 30, 2))
CIDADES_ENERGIA = 5                    # FIXO: o número de cidades não varia
VEICULOS_ENERGIA = (1, 2)              # 1 = TSP;  2 = VRP (capacidade ativa)
INSTANCIAS_ENERGIA = [(CIDADES_ENERGIA, V) for V in VEICULOS_ENERGIA]
# "auto": 70% da memória EFETIVA, lida sem sudo (sistema.py: /proc/meminfo, cgroup e
# SLURM; vale a menor). Um número fixa o limite em GB.
MEMORIA_MAX_GB = "auto"

# ---- teste: o VQE entrega um vetor VÁLIDO e ÓTIMO mesmo com cutoff pequeno? ---------
# Cidades 3-6 x veículos 1-5 (V <= C), camadas fixas, cutoff = C + MARGEM_VO.
# Com margem 1 o cutoff fica ABAIXO do nº de nós por modo (M*R) em vários casos
# (ex.: 6 cidades, 3 veículos: 12 nós, cutoff 7): é exatamente o regime a testar.
# Dimensão = (C + margem)^C: com margem 1, no máximo 7^6 = 117.649.
PASTA_VO = "vqe_valido_otimo"
CIDADES_VO = (3, 4, 5, 6)
VEICULOS_VO = (1, 2, 3, 4, 5)
CAMADAS_VO = 2
MARGEM_VO = 1
HP_VO = {f"m{MARGEM_VO}": dict(max_iter=300, lr=0.05, lr_final=0.005,
                                cutoff_rel_C=MARGEM_VO)}
DIM_MAX_VO = 200_000

# ---- grade (C, V): cidades 3-6 x veículos 1-3, cutoff = C*V, 2 camadas ---------------
PASTA_CV = "vqe_grade_cv"
CIDADES_CV = (3, 4, 5, 6)
VEICULOS_CV = (1, 2, 3)
CAMADAS_CV = 2
HP_CV = {"cv": dict(max_iter=300, lr=0.05, lr_final=0.005, cutoff_CxV=True)}
# Dimensão do espaço de Fock = cutoff**C. Com cutoff = C*V:
#   (3,1)=27  (3,2)=216  (3,3)=729  (4,1)=256  (4,2)=4.096  (4,3)=20.736
#   (5,1)=3.125  (5,2)=100.000  (5,3)=759.375  (6,1)=46.656
#   (6,2)=2.985.984  (6,3)=34.012.224   <- estes dois estão fora do alcance do VQE atual
# Tuplas acima de DIM_MAX_CV são PULADAS e registradas (status PULADO_DIMENSAO).
DIM_MAX_CV = 1_000_000

# ---- vqe_tese_v2: todos os resultados novos vão para UMA pasta -------------------
PASTA_V2 = "vqe_tese_v2"

# C = 4 e 5: duas taxas de aprendizado (a taxa foi o sobreparâmetro decisivo em C=3).
HP_V2 = {
    "lr05": dict(max_iter=300, lr=0.05, lr_final=0.005, cutoff_extra=0),
    "lr02": dict(max_iter=300, lr=0.02, lr_final=0.002, cutoff_extra=0),
}

# C = 3: grade cutoff (C + k, k = 0, 2, 5, 7, 9 -> 3, 5, 8, 10, 12) x lr (0,02; 0,05;
# 0,10). Combinada com camadas 1-3 e V = 1, 2 na tupla (C, V, L).
HP_GRADE = {
    f"dC{k}_lr{nome}": dict(max_iter=300, lr=lr, lr_final=lr / 10.0, cutoff_rel_C=k)
    for k in (0, 2, 5, 7, 9)
    for nome, lr in (("02", 0.02), ("05", 0.05), ("10", 0.10))
}

# Cutoff como sobreparâmetro: cutoff = C + k (k = 0: cutoff igual ao número de
# cidades). Hipótese em teste: cutoff maior, solução melhor. Para C=3 isto dá
# cutoffs 3, 5, 8, 10 e 12.
HP_CUTOFF = {
    f"dC{k}": dict(max_iter=200, lr=0.05, lr_final=0.005, cutoff_rel_C=k)
    for k in (0, 2, 5, 7, 9)
}

EXECUTAR = {
    # --- resultados da tese (CVRP; instâncias do Graph; capacidade ativa) ----------
    # ESTADO ATUAL: só o estudo de energia exata contra o cutoff está ligado.
    # Cidades FIXAS em 5; varia apenas o cutoff; duas versões: V=1 (TSP) e V=2 (VRP).
    "informacao_sistema": True,            # memória e CPUs do servidor (sem sudo); vai para vqe_cutoff/sistema.txt
    "energia_cutoff": True,                # C=5; V=1 (TSP) e V=2 (VRP); cutoffs 5, 7, ..., 29
    # --- VQE (desligado; o estudo de energia NÃO usa VQE nem a ponte) ----------------------
    "valido_otimo": True,                  # C=3-6 x V=1-5 (V<=C), L=2, cutoff=C+1 -> pasta vqe_valido_otimo
    "grade_cv": False,                     # C=3-6 x V=1-3, L=2, cutoff=C*V -> pasta vqe_grade_cv (usa o VQE)
    "vqe_ponte": True,                     # roda antes de qualquer VQE; se falhar, o VQE é pulado
    "tese_instancias": True,               # confere viabilidade de cada tupla antes de rodar o VQE
    # --- pasta vqe_tese_v2: critérios novos (ρ inicial/final, fundamental decodificado) ---
    "v2_c3_grade": False,                  # C=3, V=1,2, L=1-3, cutoff (C+k) x lr  (90 execuções)
    "v2_c4": False,                        # C=4, V=1,2,3, L=1-3, lr 0,05 e 0,02   (18 execuções)
    "v2_c5": False,                        # C=5, V=1,2,3, L=1-2, lr 0,05 e 0,02   (12 execuções)
    # --- resultados anteriores (já analisados; pastas vqe_tese_base/hp/cutoff/c4) ---------
    "tese_base": False,
    "tese_hiperparametros": False,
    "tese_cutoff": False,
    "tese_c4": False,
    # --- diagnósticos anteriores (já analisados) ------------------------------------
    "vqe_pequeno": False,
    "vqe_varredura_c3": False,
    "vqe_varredura_c4": False,
    "vqe_c4": False,
    "fatorado_verificacao": False,
    "fatorado_fantasma": False,
    "fatorado_validacao": False,
    "fatorado_decomposicao_exata": False,
    "fatorado_distinta": False,
    "fatorado_campo_medio": False,
    "fatorado_overhead": False,
    "fatorado_decisao": False,
    "fatorado_pecas": False,
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
    from saidas import caminho as CAM

    # ======================================================================
    # VQE: ponte, instâncias e resultados da tese
    # ======================================================================
    blocos_vqe = ("vqe_ponte", "tese_instancias", "valido_otimo", "grade_cv", "v2_c3_grade", "v2_c4", "v2_c5",
                  "tese_base", "tese_hiperparametros", "tese_cutoff", "tese_c4",
                  "vqe_pequeno", "vqe_varredura_c3", "vqe_varredura_c4", "vqe_c4")
    ponte_ok = False
    if any(EXECUTAR[k] for k in blocos_vqe):
        import exp_vqe as EV

        with saida("resultado_vqe_ponte.txt", "vqe_ponte"):
            ponte_ok = EV.verificar_ponte()
        if not ponte_ok:
            print("\nA ponte FALHOU: o VQE foi pulado.")

    if ponte_ok and (EXECUTAR["valido_otimo"] or EXECUTAR["grade_cv"] or EXECUTAR["v2_c3_grade"]
                     or EXECUTAR["v2_c4"] or EXECUTAR["v2_c5"]
                     or EXECUTAR["tese_base"] or EXECUTAR["tese_hiperparametros"]
                     or EXECUTAR["tese_cutoff"] or EXECUTAR["tese_c4"]):
        import exp_vqe_tese as TV

        def bloco_tese(nome, tuplas, hiperparametros, pasta, **extra):
            if EXECUTAR["tese_instancias"]:
                with saida(f"instancias_{nome}.txt", pasta):
                    viaveis = TV.verificar_instancias(tuplas, pasta=pasta, Q_val=Q_VAL,
                                                      folga=FOLGA)
            else:
                viaveis = tuplas
            if not viaveis:
                print(f"\n[{nome}] nenhuma tupla viável: bloco pulado.")
                return
            with saida(f"resultado_{nome}.txt", pasta):
                linhas = TV.varredura_tese(viaveis, hiperparametros=hiperparametros,
                                           pasta=pasta, Q_val=Q_VAL, folga=FOLGA, **extra)
                TV.resumo_tese(linhas)

        # --- vqe_tese_v2: critérios novos; todos os resultados na MESMA pasta ---------
        if EXECUTAR["v2_c3_grade"]:
            bloco_tese("v2_c3_grade",
                       [(3, V, L) for V in (1, 2) for L in (1, 2, 3)],
                       HP_GRADE, PASTA_V2)
        if EXECUTAR["v2_c4"]:
            bloco_tese("v2_c4",
                       [(4, V, L) for V in (1, 2, 3) for L in (1, 2, 3)],
                       HP_V2, PASTA_V2)
        if EXECUTAR["v2_c5"]:
            # N = 5: (M*R)^N configurações por distribuição; menos checkpoints
            bloco_tese("v2_c5",
                       [(5, V, L) for V in (1, 2, 3) for L in (1, 2)],
                       HP_V2, PASTA_V2, n_checkpoints=10)

        # --- vetor válido e ótimo com cutoff pequeno (C + margem), camadas fixas ---------
        if EXECUTAR["valido_otimo"]:
            tuplas_vo = sorted(((C, V, CAMADAS_VO) for C in CIDADES_VO
                                for V in VEICULOS_VO if V <= C),
                               key=lambda t: (t[0] + MARGEM_VO) ** t[0] * t[1])
            bloco_tese("valido_otimo", tuplas_vo, HP_VO, PASTA_VO,
                       dim_max=DIM_MAX_VO, n_checkpoints=10)

        # --- grade (C, V): cutoff = C*V, 2 camadas; da menor para a maior dimensão ---------
        if EXECUTAR["grade_cv"]:
            tuplas_cv = sorted(((C, V, CAMADAS_CV) for C in CIDADES_CV for V in VEICULOS_CV),
                               key=lambda t: (t[0] * t[1]) ** t[0])
            bloco_tese("grade_cv", tuplas_cv, HP_CV, PASTA_CV,
                       dim_max=DIM_MAX_CV, n_checkpoints=10)

        if EXECUTAR["tese_base"]:
            bloco_tese("tese_base",
                       [(3, V, L) for V in (1, 2) for L in (1, 2, 3)],
                       HP_BASE, "vqe_tese_base")
        if EXECUTAR["tese_hiperparametros"]:
            bloco_tese("tese_hiperparametros",
                       [(3, V, 2) for V in (1, 2)],
                       HP_VARIANTES, "vqe_tese_hp")
        if EXECUTAR["tese_cutoff"]:
            bloco_tese("tese_cutoff",
                       [(3, V, 2) for V in (1, 2)],
                       HP_CUTOFF, "vqe_tese_cutoff")
        if EXECUTAR["tese_c4"]:
            bloco_tese("tese_c4",
                       [(4, V, L) for V in (1, 2, 3) for L in (1, 2)],
                       HP_BASE, "vqe_tese_c4")

    # ======================================================================
    # diagnósticos anteriores (desligados)
    # ======================================================================
    # ======================================================================
    # energia exata contra o cutoff (independe do VQE e da ponte)
    # ======================================================================
    if EXECUTAR["informacao_sistema"]:
        from sistema import relatorio_sistema

        with saida("sistema.txt", PASTA_ENERGIA):
            print(relatorio_sistema())

    if EXECUTAR["energia_cutoff"]:
        import exp_energia_cutoff as EC

        with saida("resultado_energia_cutoff.txt", PASTA_ENERGIA):
            por_veiculos = {}
            for (C_e, V_e) in INSTANCIAS_ENERGIA:
                linhas_e, metricas_e = EC.energia_vs_cutoff(
                    C=C_e, V=V_e, cutoffs=CUTOFFS_ENERGIA, pasta=PASTA_ENERGIA,
                    Q_val=Q_VAL, folga=FOLGA, memoria_max_gb=MEMORIA_MAX_GB)
                print(f"\n##### instância C={C_e} V={V_e} "
                      f"({'TSP' if V_e == 1 else 'VRP'}) #####")
                EC.resumo_energia_cutoff(linhas_e, metricas_e)
                por_veiculos[V_e] = linhas_e
            if len(por_veiculos) > 1:
                EC.figura_comparativa(PASTA_ENERGIA, CIDADES_ENERGIA, por_veiculos)

    if ponte_ok and EXECUTAR["vqe_pequeno"]:
        with saida("resultado_vqe.txt", "vqe_antigo"):
            EV.vqe_vs_exato(C=3, V=2, seed=42, cutoff=8, layers=2, maxiter=100,
                            csv_path=CAM("vqe_resultados_v2.csv", "vqe_antigo"))
            EV.vqe_vs_exato(C=3, V=1, seed=42, cutoff=8, layers=2, maxiter=100,
                            csv_path=CAM("vqe_resultados_v2.csv", "vqe_antigo"))
    if ponte_ok and EXECUTAR["vqe_varredura_c3"]:
        import exp_vqe_busca as EB

        with saida("resultado_vqe_varredura_c3.txt", "vqe_antigo"):
            linhas = EB.varredura_vqe(cidades=(3,), veiculos=(1, 2), camadas=(1, 2, 3),
                                      cutoffs={3: 8}, seeds_instancia=(42,),
                                      seeds_theta=(0, 1), maxiter=100,
                                      csv_path=CAM("vqe_varredura_c3.csv", "vqe_antigo"))
            EB.resumo_varredura(linhas)
    if ponte_ok and EXECUTAR["vqe_varredura_c4"]:
        import exp_vqe_busca as EB

        with saida("resultado_vqe_varredura_c4.txt", "vqe_antigo"):
            linhas = EB.varredura_vqe(cidades=(4,), veiculos=(1, 2, 3), camadas=(1, 2),
                                      cutoffs={4: 10}, seeds_instancia=(42,),
                                      seeds_theta=(0,), maxiter=100,
                                      csv_path=CAM("vqe_varredura_c4.csv", "vqe_antigo"))
            EB.resumo_varredura(linhas)
    if ponte_ok and EXECUTAR["vqe_c4"]:
        with saida("resultado_vqe_c4.txt", "vqe_antigo"):
            for (v, s) in ((1, 42), (2, 42), (2, 3)):
                EV.vqe_vs_exato(C=4, V=v, seed=s, cutoff=10, layers=2, maxiter=100,
                                theta_seed=0,
                                csv_path=CAM("vqe_resultados_v2.csv", "vqe_antigo"))

    # --- caminho B (estados-produto) -------------------------------------------------
    blocos_fatorados = ("fatorado_verificacao", "fatorado_fantasma", "fatorado_validacao",
                        "fatorado_decomposicao_exata", "fatorado_distinta",
                        "fatorado_campo_medio", "fatorado_overhead", "fatorado_decisao",
                        "fatorado_pecas")
    if any(EXECUTAR[k] for k in blocos_fatorados):
        import exp_fatorado as F

        with saida("resultado_f_verificacao.txt", "caminho_b"):
            ok = F.verificar()
        if not ok:
            print("\nA verificação FALHOU: os demais blocos fatorados foram pulados.")
        else:
            if EXECUTAR["fatorado_fantasma"]:
                with saida("resultado_f_fantasma.txt", "caminho_b"):
                    F.decomposicao_fantasma()
            if EXECUTAR["fatorado_validacao"]:
                formas_v = ("soma", "predecessor+distinta")
                with saida("resultado_f_validacao.txt", "caminho_b"):
                    F.decisao_por_semente(C=4, V=2, seeds=tuple(range(6, 16)), formas=formas_v)
                    F.decisao_por_semente(C=4, V=1, seeds=(42, 1, 2, 3, 4, 5), formas=formas_v)
                    F.decisao_por_semente(C=5, V=2, seeds=(42, 1, 2, 3, 4),
                                          ds=(8, 16, 32, 64), formas=formas_v)
                    F.decisao_por_semente(C=5, V=3, seeds=(42, 1, 2, 3, 4),
                                          ds=(8, 16, 32, 64), formas=formas_v)
            if EXECUTAR["fatorado_decomposicao_exata"]:
                with saida("resultado_f_decomposicao.txt", "caminho_b"):
                    F.decomposicao_exata_da_lacuna()
            if EXECUTAR["fatorado_distinta"]:
                formas_d = ("soma", "soma+distinta", "predecessor+distinta")
                with saida("resultado_f_distinta_overhead.txt", "caminho_b"):
                    F.overhead_por_perfil(ds=(8, 16, 32, 64), formas=formas_d)
                with saida("resultado_f_distinta_decisao.txt", "caminho_b"):
                    F.decisao_por_semente(formas=formas_d)
            if EXECUTAR["fatorado_campo_medio"]:
                with saida("resultado_f_campo_medio.txt", "caminho_b"):
                    F.campo_medio_por_perfil()
            if EXECUTAR["fatorado_overhead"]:
                with saida("resultado_f_overhead.txt", "caminho_b"):
                    F.overhead_por_perfil()
            if EXECUTAR["fatorado_decisao"]:
                with saida("resultado_f_decisao.txt", "caminho_b"):
                    F.decisao_por_semente()
            if EXECUTAR["fatorado_pecas"]:
                with saida("resultado_f_pecas_lacuna.txt", "caminho_b"):
                    F.pecas_da_lacuna()

    # --- blocos anteriores -------------------------------------------------------------
    if EXECUTAR["d_fidelidade_dos_nos"]:
        with saida("resultado_d.txt", "caminho_b"):
            parte_d(C=4, V=2, ds=(8, 12, 16, 24, 32, 48, 64))
    if EXECUTAR["a_perfis"]:
        with saida("resultado_a.txt", "caminho_b"):
            parte_a(4, 2)
            parte_a(4, 1)
    if EXECUTAR["c_soma"]:
        with saida("resultado_c.txt", "caminho_b"):
            parte_c(4, 2, ds=(8, 12, 16), gap_form="soma")
    if EXECUTAR["c_predecessor"]:
        with saida("resultado_c_predecessor.txt", "caminho_b"):
            parte_c(4, 2, ds=(8, 12, 16), gap_form="predecessor")

    if EXECUTAR["sementes_v4"] or EXECUTAR["c4_grande"]:
        from exp_cutoff_minimo import main as cutoff_main

        if EXECUTAR["c4_grande"]:
            cutoff_main(cities=[4], vehicles=[1, 3], max_dim=170_000,
                        out=CAM("c4_grande.csv", "cutoff_minimo"),
                        trace_out=CAM("c4_grande_trace.csv", "cutoff_minimo"))
        if EXECUTAR["sementes_v4"]:
            for s in range(1, 6):
                cutoff_main(cities=[4], vehicles=[1, 2], cutoffs=(5, 8, 12, 16), seed=s,
                            out=CAM(f"seed{s}.csv", "cutoff_minimo"),
                            trace_out=CAM(f"seed{s}_trace.csv", "cutoff_minimo"))

    if EXECUTAR["scan_gap"]:
        from exp_scan_gap import run as scan_gap_run

        with saida("resultado_gap.txt", "caminho_b"):
            scan_gap_run(out=CAM("scan_gap.csv", "caminho_b"))


if __name__ == "__main__":
    main()