"""
exp_energia_cutoff.py — a menor energia de H se aproxima de C* quando o cutoff cresce?

Hipótese (discutida com os outros pesquisadores): a diferença entre a menor energia
de H, E0, e o custo ótimo da força bruta, C*, vem do cutoff finito (o estado não
resolve os nós de posição e de veículo). Se for isso, E0(d) deve se aproximar de C*
quando d cresce.

Método: para uma instância (C, V) fixa, a menor energia de H por diagonalização
exata, sem matriz densa (qumodes/hamiltonian_sparse.py), para cada cutoff d. A
DECODIFICAÇÃO do fundamental é feita só até `decodificar_ate`: ela custa (M*R)^N
contrações em d^N e domina o tempo em cutoff alto; para E0 não é necessária.

Limite prático: MEMÓRIA. O operador guarda N*M*R vetores intermediários de d^N
amplitudes complexas, mais os vetores do Arnoldi. A estimativa (com fator 1,3 para
temporários) é calculada ANTES de cada cutoff; se passar de `memoria_max_gb`, o
cutoff é pulado e registrado (um estouro de memória do sistema mata o Python sem
aviso e perde tudo).

Saídas (PathManager("qumodes", pasta); a pasta padrão é vqe_cutoff):
    energia_cutoff.csv                    uma linha por cutoff (gravada a cada passo)
    metricas_cutoff.csv                   métricas da relação cutoff x energia (por instância):
                                          monotonia, Spearman, queda total/relativa,
                                          inclinação, expoente do overhead
    runs/energia_cutoff_c{C}_v{V}.json    configuração, instância, tabela e ajustes
    figures/..._E0_vs_cutoff.png          E0(d) com a reta de C*
    figures/..._overhead_loglog.png       E0 - C* contra d em log-log, com ajuste
    logs/

Cuidado de leitura: o ajuste de lei de potência supõe que E0 -> C* quando d -> inf.
Se E0 estabilizar acima de C*, o expoente ajustado subestima o piso; a extrapolação
é só uma ordem de grandeza.
"""
import math
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import qumodes.hamiltonian as H
import qumodes.hamiltonian_sparse as HS
from brute_force import BruteForce
from exp_vqe_tese import (HBAR, _decodificar, _estatisticas, _hamiltoniano,
                          anexar_csv, enumerar_solucoes, montar_instancia)
from logger import setup_logger
from saidas import caminho, gerenciador
from utils import format_timespan, save_experiment_json

plt.switch_backend("Agg")


def estimar_gb(d, N, M, R, ncv):
    """Memória estimada (GB): N*M*R vetores T + M*R vetores de ocupação + ncv vetores
    do Arnoldi + ~12 temporários, de d^N amplitudes complexas (16 bytes), com fator
    1,3 de folga."""
    n_vet = N * M * R + M * R + ncv + 12
    return 1.3 * n_vet * 16.0 * float(d) ** N / 1e9


def ajuste_potencia(ds, overhead):
    """Ajusta overhead ~ a * d^(-s) por mínimos quadrados em log-log. Devolve
    (s, a) ou (nan, nan) se não houver pontos positivos suficientes."""
    ds = np.asarray(ds, dtype=float)
    ov = np.asarray(overhead, dtype=float)
    ok = ov > 0
    if ok.sum() < 3:
        return float("nan"), float("nan")
    p = np.polyfit(np.log(ds[ok]), np.log(ov[ok]), 1)
    return float(-p[0]), float(math.exp(p[1]))


def metricas_relacao(ds, e0):
    """Métricas da relação cutoff x menor energia (hipótese: E0 decresce com d).

    monotono_decrescente   E0 nunca sobe de um cutoff para o seguinte (tolerância 1e-9)
    n_violacoes            quantos passos consecutivos subiram
    spearman_rho           correlação de postos entre d e E0 (-1 = decrescimento perfeito)
    queda_total            E0(menor d) - E0(maior d)
    queda_relativa_pct     queda_total em % de E0(menor d)
    inclinacao_E0_d        inclinação do ajuste linear de E0 contra d
    """
    ds = np.asarray(ds, dtype=float)
    e0 = np.asarray(e0, dtype=float)
    passos = np.diff(e0)
    rho, p = spearmanr(ds, e0)
    queda = float(e0[0] - e0[-1])
    return dict(n_pontos=int(ds.size), cutoff_min=int(ds[0]), cutoff_max=int(ds[-1]),
                E0_cutoff_min=float(e0[0]), E0_cutoff_max=float(e0[-1]),
                monotono_decrescente=bool(np.all(passos <= 1e-9)),
                n_violacoes=int(np.sum(passos > 1e-9)),
                spearman_rho=float(rho), spearman_p=float(p),
                queda_total=queda, queda_relativa_pct=float(100.0 * queda / e0[0]),
                inclinacao_E0_d=float(np.polyfit(ds, e0, 1)[0]))


def energia_vs_cutoff(C=5, V=2, cutoffs=tuple(range(5, 30, 2)), pasta="vqe_cutoff",
                      seed=42, Q_val="auto", folga=1.2, demand_range=(1, 5),
                      memoria_max_gb=16.0, decodificar_ate=9, ncv_grande=24,
                      tol=1e-9, shots=1024):
    exp = f"c{C}_v{V}"
    gm = gerenciador(pasta)
    log = setup_logger(name=f"{exp}_energia_cutoff",
                       log_file=gm.get_file_path(f"logs/{exp}_energia_cutoff.log"))
    if memoria_max_gb == "auto":
        from sistema import memoria_gb, memoria_max_automatica
        memoria_max_gb = memoria_max_automatica()
        log.info(f"memória detectada: {memoria_gb()}  ->  limite automático "
                 f"(70% da efetiva) = {memoria_max_gb:.1f} GB")

    # --- instância, força bruta, Hamiltoniano ------------------------------------
    g, D, demands, Q = montar_instancia(C, V, seed, Q_val, demand_range, folga)
    bf = BruteForce(dist_matrix=g.dist_matrix, num_vehicles=V, capacities=Q,
                    demands=g.full_demands)
    c_star, bf_rotas = bf.solve()
    _, optset, n_viaveis = enumerar_solucoes(D, V, demands, Q)
    inst, par = _hamiltoniano(C, V, D, demands, Q)
    req, cap = V > 1, V > 1
    X = inst.M * inst.R
    est_u = _estatisticas(np.full((X,) * C, 1.0 / float(X) ** C), inst, D, optset, req, cap)
    log.info(f"instância C={C} V={V} R={inst.R} Q={Q[0]:g} demandas={list(demands)}")
    log.info(f"força bruta: C*={c_star:.4f} rotas={bf_rotas} #ótimas={len(optset)} "
             f"#viáveis={n_viaveis}")
    log.info(f"memória máxima permitida: {memoria_max_gb} GB; decodificação até d={decodificar_ate}")

    destino_csv = caminho("energia_cutoff.csv", pasta)
    linhas = []
    for d in cutoffs:
        dim = d ** C
        ncv = 48 if dim < 200_000 else ncv_grande
        gb = estimar_gb(d, C, inst.M, inst.R, ncv)
        linha = dict(C=C, V=V, R=inst.R, Q=float(Q[0]), cutoff=d, dim=dim,
                     c_star=c_star, mem_estimada_gb=round(gb, 2))
        if gb > memoria_max_gb:
            linha["status"] = "PULADO_MEMORIA"
            log.warning(f"d={d}: memória estimada {gb:.1f} GB > {memoria_max_gb} GB; pulado")
            anexar_csv(destino_csv, linha)
            linhas.append(linha)
            continue
        ops = H.build_modular_operators(d, inst.M, inst.R, HBAR)
        decodificar = d <= decodificar_ate
        t0 = time.time()
        try:
            gs = HS.ground_state_sparse(inst, ops, par, k=1, ncv=ncv, tol=tol,
                                        decode=decodificar)
        except MemoryError as err:
            linha.update(status="ERRO_MEMORIA", erro=str(err))
            log.error(f"d={d}: MemoryError")
            anexar_csv(destino_csv, linha)
            linhas.append(linha)
            continue
        t = time.time() - t0
        e0 = float(gs["energy"])
        linha.update(status="OK", E0=e0, overhead=e0 - c_star, razao_E0_cstar=e0 / c_star,
                     residual=gs["residual"], converged=int(gs["converged"]),
                     n_matvec=gs["n_matvec"], tempo_s=round(t, 1))
        if decodificar and gs.get("distribution") is not None:
            dist = np.asarray(gs["distribution"], dtype=np.float64)
            est = _estatisticas(dist, inst, D, optset, req, cap)
            dec = _decodificar(dist, inst, D, req, cap, shots, seed)
            custo_dec = dec[0] if dec is not None else float("nan")
            linha.update(gs_rho=(est["p_otima"] / est_u["p_otima"]
                                 if est_u["p_otima"] > 0 else float("nan")),
                         gs_p_otima=est["p_otima"], gs_p_valid=est["p_valid"],
                         gs_top_otima=int(est["top_otima"]), gs_custo_dec=custo_dec,
                         gs_ok_custo=int(dec is not None and abs(custo_dec - c_star) < 1e-9))
        log.info(f"d={d:>2} dim={dim:>10,}  E0={e0:10.4f}  E0-C*={e0 - c_star:9.4f}  "
                 f"resíduo={gs['residual']:.1e}  conv={gs['converged']}  "
                 f"{format_timespan(t)}  (~{gb:.1f} GB)")
        anexar_csv(destino_csv, linha)
        linhas.append(linha)

    # --- ajuste e figuras ----------------------------------------------------------
    ok = [r for r in linhas if r["status"] == "OK"]
    ajustes = {}
    metricas = {}
    if len(ok) >= 3:
        ds = [r["cutoff"] for r in ok]
        ov = [r["overhead"] for r in ok]
        s_all, a_all = ajuste_potencia(ds, ov)
        s_ult, a_ult = ajuste_potencia(ds[-4:], ov[-4:]) if len(ok) >= 4 else (float("nan"),) * 2
        ajustes = dict(expoente_todos=s_all, coef_todos=a_all, expoente_ultimos4=s_ult,
                       coef_ultimos4=a_ult)
        s_ref = s_ult if not math.isnan(s_ult) else s_all
        d_ult, ov_ult = ds[-1], ov[-1]
        if s_ref > 0.05 and ov_ult > 0:
            for rot, alvo in (("1pct_de_Cstar", 0.01 * c_star), ("1_unidade", 1.0)):
                ajustes[f"d_necessario_{rot}"] = float(d_ult * (ov_ult / alvo) ** (1.0 / s_ref))
        log.info(f"ajuste overhead ~ d^(-s): s(todos)={s_all:.3f}  s(últimos 4)={s_ult:.3f}")
        metricas = metricas_relacao(ds, [r["E0"] for r in ok])
        metricas.update(ajustes)
        log.info(f"relação cutoff x energia: monótono decrescente="
                 f"{metricas['monotono_decrescente']} (violações={metricas['n_violacoes']}), "
                 f"Spearman={metricas['spearman_rho']:.3f}, queda total="
                 f"{metricas['queda_total']:.3f} ({metricas['queda_relativa_pct']:.1f}%)")
        anexar_csv(caminho("metricas_cutoff.csv", pasta),
                   dict(C=C, V=V, c_star=c_star, **metricas))
        _figuras(gm, exp, ok, c_star, ajustes, metricas, C, V)
    else:
        log.info("menos de 3 cutoffs concluídos: sem métricas, ajuste nem figuras")

    payload = dict(config=dict(C=C, V=V, cutoffs=list(cutoffs), seed=seed, Q=Q.tolist(),
                               folga=folga, memoria_max_gb=memoria_max_gb,
                               decodificar_ate=decodificar_ate, R=inst.R),
                   instancia=dict(coords=g.graph.coords, D=D, demands=demands),
                   forca_bruta=dict(custo=c_star, rotas=bf_rotas, n_otimas=len(optset),
                                    n_viaveis=n_viaveis),
                   tabela=linhas, ajustes=ajustes, metricas_relacao=metricas)
    save_experiment_json(payload, str(gm.get_file_path(f"energia_cutoff_{exp}.json", subdir="runs")))
    return linhas, metricas


def _figuras(gm, exp, ok, c_star, ajustes, metricas, C, V):
    ds = np.array([r["cutoff"] for r in ok], dtype=float)
    e0 = np.array([r["E0"] for r in ok])
    ov = e0 - c_star

    # gráfico da RELAÇÃO cutoff x energia (a hipótese: a energia diminui com o cutoff)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ds, e0, "o-", color="tab:red", lw=1.6, label="Menor energia de $H$: $E_0(d)$")
    ax.axhline(c_star, ls="--", lw=1.2, color="tab:green",
               label=f"Força bruta: $C^*$ = {c_star:.2f}")
    ax.set_xlabel("Cutoff $d$")
    ax.set_ylabel("Energia")
    ax.set_title(f"Relação entre o cutoff e a menor energia de $H$ — C={C}, V={V}")
    ax.grid(True, alpha=0.3)
    if metricas:
        ax.text(0.97, 0.95,
                f"Spearman $\\rho$ = {metricas['spearman_rho']:.2f}\n"
                f"queda: {metricas['queda_total']:.1f} ({metricas['queda_relativa_pct']:.1f}%)\n"
                f"monótona: {'sim' if metricas['monotono_decrescente'] else 'não'}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(gm.get_file_path(f"{exp}_E0_vs_cutoff.png", is_figure=True), dpi=300)
    plt.close(fig)

    pos = ov > 0
    if pos.sum() >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(ds[pos], ov[pos], "o", color="tab:red", label="$E_0(d) - C^*$")
        s = ajustes.get("expoente_todos", float("nan"))
        a = ajustes.get("coef_todos", float("nan"))
        if not math.isnan(s):
            xx = np.linspace(ds.min(), ds.max(), 100)
            ax.loglog(xx, a * xx ** (-s), "-", color="tab:blue", lw=1.2,
                      label=f"ajuste $\\propto d^{{-{s:.2f}}}$")
        ax.set_xlabel("Cutoff $d$")
        ax.set_ylabel("Sobra de energia $E_0 - C^*$")
        ax.set_title(f"Overhead de codificação contra o cutoff — C={C}, V={V}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(gm.get_file_path(f"{exp}_overhead_loglog.png", is_figure=True), dpi=300)
        plt.close(fig)


def figura_comparativa(pasta, C, por_veiculos):
    """TSP (V=1) x VRP (V>=2) com as MESMAS cidades (mesma semente do Graph).

    Painel esquerdo: E0(d) de cada versão, com a reta do seu C*.
    Painel direito: sobra RELATIVA (E0 - C*)/C* em log-log. Normalizar por C* deixa
    comparar versões com custos ótimos diferentes: quanto mais íngreme a queda, mais
    rápido a energia do fundamental se aproxima do ótimo quando o cutoff cresce.
    Grava c{C}_tsp_vs_vrp_cutoff.png e comparacao_tsp_vrp.csv na pasta.
    """
    gm = gerenciador(pasta)
    rot = lambda V: "TSP (V=1)" if V == 1 else f"VRP (V={V})"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    print("\n" + "=" * 112)
    print(f"COMPARAÇÃO TSP x VRP — {C} cidades fixas; só o cutoff varia")
    print("=" * 112)
    for i, (V, linhas) in enumerate(sorted(por_veiculos.items())):
        ok = [r for r in linhas if r["status"] == "OK"]
        if not ok:
            continue
        ds = np.array([r["cutoff"] for r in ok], dtype=float)
        e0 = np.array([r["E0"] for r in ok])
        c_star = float(ok[0]["c_star"])
        cor = f"C{i}"
        ax1.plot(ds, e0, "o-", color=cor, lw=1.6, label=f"{rot(V)}: $E_0(d)$")
        ax1.axhline(c_star, ls="--", lw=1.0, color=cor, alpha=0.7,
                    label=f"{rot(V)}: $C^*$ = {c_star:.2f}")
        rel = (e0 - c_star) / c_star
        pos = rel > 0
        s = float("nan")
        if pos.sum() >= 3:
            s, _ = ajuste_potencia(ds[pos], rel[pos])
        if pos.sum() >= 2:
            lab = rot(V) + (f"  ($\\propto d^{{-{s:.2f}}}$)" if not math.isnan(s) else "")
            ax2.loglog(ds[pos], rel[pos], "o-", color=cor, lw=1.4, label=lab)
        print(f"  {rot(V):<10} C*={c_star:9.3f}   E0: {e0[0]:9.3f} (d={int(ds[0])}) -> "
              f"{e0[-1]:9.3f} (d={int(ds[-1])})   sobra relativa: {rel[0]:.2f} -> {rel[-1]:.2f}"
              f"   expoente s = {s:.3f}")
        anexar_csv(caminho("comparacao_tsp_vrp.csv", pasta),
                   dict(C=C, V=V, tipo=("TSP" if V == 1 else "VRP"), c_star=c_star,
                        cutoff_min=int(ds[0]), cutoff_max=int(ds[-1]),
                        E0_cutoff_min=float(e0[0]), E0_cutoff_max=float(e0[-1]),
                        sobra_rel_cutoff_min=float(rel[0]), sobra_rel_cutoff_max=float(rel[-1]),
                        expoente_sobra_relativa=s))
    ax1.set_xlabel("Cutoff $d$")
    ax1.set_ylabel("Menor energia de $H$")
    ax1.set_title(f"Energia contra o cutoff — {C} cidades")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="best")
    ax2.set_xlabel("Cutoff $d$")
    ax2.set_ylabel("Sobra relativa $(E_0 - C^*)/C^*$")
    ax2.set_title("Aproximação do ótimo: TSP x VRP")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(gm.get_file_path(f"c{C}_tsp_vs_vrp_cutoff.png", is_figure=True), dpi=300)
    plt.close(fig)


def resumo_energia_cutoff(linhas, metricas=None):
    if metricas:
        print("\n" + "=" * 112)
        print("MÉTRICAS DA RELAÇÃO CUTOFF x MENOR ENERGIA (hipótese: a energia diminui com o cutoff)")
        print("=" * 112)
        print(f"  cutoffs: {metricas['cutoff_min']} a {metricas['cutoff_max']} "
              f"({metricas['n_pontos']} pontos)   E0: {metricas['E0_cutoff_min']:.3f} -> "
              f"{metricas['E0_cutoff_max']:.3f}")
        print(f"  monótona decrescente: {'SIM' if metricas['monotono_decrescente'] else 'NÃO'} "
              f"(passos que subiram: {metricas['n_violacoes']})")
        print(f"  Spearman(d, E0) = {metricas['spearman_rho']:.3f} (p = {metricas['spearman_p']:.2g});  "
              f"queda total = {metricas['queda_total']:.3f} ({metricas['queda_relativa_pct']:.1f}% "
              f"de E0 em d={metricas['cutoff_min']})")
        print(f"  inclinação linear dE0/dd = {metricas['inclinacao_E0_d']:.4f}   "
              f"expoente do overhead (E0 - C*) ~ d^-s: s(todos) = {metricas.get('expoente_todos', float('nan')):.3f}, "
              f"s(últimos 4) = {metricas.get('expoente_ultimos4', float('nan')):.3f}")
    print("\n" + "=" * 112)
    print("MENOR ENERGIA DE H CONTRA O CUTOFF   (C* = custo ótimo da força bruta)")
    print("=" * 112)
    print(f"{'d':>4}{'dim':>13}{'mem GB':>8}{'E0':>11}{'E0-C*':>10}{'E0/C*':>8}"
          f"{'rho_gs':>8}{'gs_top':>7}{'conv':>5}{'resíduo':>10}  status")
    for r in linhas:
        if r["status"] != "OK":
            print(f"{r['cutoff']:>4}{r['dim']:>13,}{r['mem_estimada_gb']:>8.1f}  {r['status']}")
            continue
        rho = f"{r['gs_rho']:>8.2f}" if "gs_rho" in r else f"{'—':>8}"
        top = (("sim" if r["gs_top_otima"] else "nao") if "gs_top_otima" in r else "—")
        print(f"{r['cutoff']:>4}{r['dim']:>13,}{r['mem_estimada_gb']:>8.1f}{r['E0']:>11.3f}"
              f"{r['overhead']:>10.3f}{r['razao_E0_cstar']:>8.2f}{rho}{top:>7}"
              f"{r['converged']:>5}{r['residual']:>10.1e}  OK")