"""
exp_vqe_tese.py — resultados do VQE para a tese (CVRP, instâncias pequenas).

Colocar na raiz do projeto (ao lado de graphs.py, brute_force.py, path.py, ...).
Tudo é lançado pelo exp_run.py.

Para cada tupla (C, V, L) = (cidades, veículos, camadas) e cada conjunto de
sobreparâmetros (iterações, lr, cutoff) gera, em PathManager("qumodes", pasta):

    summary.csv                         uma linha por experimento (raiz da pasta)
    runs/<exp>.json                     configuração, instância, soluções, métricas
    arrays/<exp>_state.npz              θ, históricos e distribuição final
    logs/<exp>.log
    figures/<exp>_original_graph.png    grafo SEM solução
    figures/<exp>_bf_route.png          grafo com a solução da FORÇA BRUTA
    figures/<exp>_vqe_route.png         grafo com a solução decodificada do VQE
    figures/<exp>_convergence.png       energia do VQE por iteração, com duas retas:
                                        o ótimo da força bruta (C*) e a menor
                                        energia de H no cutoff (fundamental exato)
    figures/<exp>_decodificacao.png     custo da rota decodificada por checkpoint,
                                        contra C*

Instância: Graph(C, V, is_warm_start=False): coordenadas aleatórias (semente 42),
D euclidiana, demandas inteiras em [1, 5] se V > 1 (zeros se V = 1, isto é, TSP).
A CAPACIDADE é construída da demanda total (capacidade_por_demanda): a menor
capacidade inteira >= max(maior demanda, ceil(folga * demanda total / V)) para a
qual existe partição viável em V rotas não vazias. Assim toda instância tem
solução e a restrição de capacidade continua ativa para V > 1. O termo de
capacidade do Hamiltoniano segue a regra: V = 1 -> lambda_cap = 0; V > 1 ->
lambda_cap > 0. O parâmetro is_warm_start não é exposto (gera instâncias
degeneradas).

O cutoff é um sobreparâmetro testado (cutoff_rel_C: C + k; k = 0 é cutoff igual ao
número de cidades). A hipótese em teste: cutoff maior, solução melhor.

Referência: BruteForce (brute_force.py). Para contar soluções ótimas e viáveis
(ρ e P(acerto)) a enumeração é refeita aqui com a MESMA regra de viabilidade, e o
custo ótimo é conferido contra o do BruteForce.

Hamiltoniano: H.suggest_penalties, com capacidade ativa para V > 1.
"""
import csv
import dataclasses
import inspect
import itertools
import json
import math
import os
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import strawberryfields as sf

import qumodes.hamiltonian as H
import qumodes.hamiltonian_sparse as HS
import qumodes.sf_bridge as SB
from qumodes.ansatz import CircuitConfig
from qumodes.solver import ProblemInstance, VQESolver

from brute_force import BruteForce, InfeasibleProblemError
from graphs import Graph
from logger import setup_logger
from saidas import caminho, gerenciador
from utils import format_timespan, save_experiment_json

plt.switch_backend("Agg")

HBAR = 2.0
SHOTS = (16, 64, 256, 1024)
LIMIAR_ACERTO = 0.99
CUTOFF_BASE = {3: 8, 4: 10, 5: 10}
HIPERPARAMETROS_BASE = {
    "base": dict(max_iter=200, lr=0.05, lr_final=0.005, cutoff_extra=0),
}


# ---------------------------------------------------------------------------
# instância, força bruta e conjunto de soluções ótimas
# ---------------------------------------------------------------------------
def _particao_viavel(demands, V, Q):
    """Existe atribuição das cidades a V veículos, todos NÃO vazios, com carga
    <= Q em cada um? (busca exaustiva; C pequeno)"""
    n = len(demands)
    for atrib in itertools.product(range(V), repeat=n):
        if len(set(atrib)) != V:            # algum veículo ficaria vazio
            continue
        carga = [0.0] * V
        for i, v in enumerate(atrib):
            carga[v] += demands[i]
        if all(c <= Q + 1e-9 for c in carga):
            return True
    return False


def capacidade_por_demanda(demands, V, folga=1.2):
    """Capacidade (igual para todos os veículos) construída da demanda total:
    a menor capacidade inteira >= max(maior demanda, ceil(folga * demanda total / V))
    para a qual existe partição viável das cidades em V rotas não vazias. Garante
    que a instância tem solução e mantém a restrição de capacidade ativa
    (folga próxima de 1 = capacidade apertada). Para V = 1 (TSP) a capacidade não
    é usada e vale 1."""
    d = [float(x) for x in demands]
    if V == 1 or sum(d) <= 0:
        return 1.0
    q = float(max(max(d), math.ceil(folga * sum(d) / V)))
    while not _particao_viavel(d, V, q):
        q += 1.0
    return q


def montar_instancia(C, V, seed=42, Q_val="auto", demand_range=(1, 5), folga=1.2):
    """Q_val = "auto" (padrão): capacidade construída da demanda total (ver
    capacidade_por_demanda); número: capacidade fixa por veículo."""
    g = Graph(C=C, V=V, is_warm_start=False, demand_range=demand_range, seed=seed)
    D, demands, _ = g.graph
    if Q_val is None or Q_val == "auto":
        q = capacidade_por_demanda(demands, V, folga)
    else:
        q = float(Q_val)
    Q = np.array([q] * V, dtype=np.float64)
    return g, D, demands, Q


def _rota_custo(D, r):
    if not r:
        return 0.0
    c = float(D[0, r[0]]) + float(D[r[-1], 0])
    for a, b in zip(r[:-1], r[1:]):
        c += float(D[a, b])
    return c


def _custo_rotas(D, rotas):
    return sum(_rota_custo(D, r) for r in rotas if r)


def _canon(rotas):
    return tuple(sorted(tuple(r) for r in rotas if r))


def enumerar_solucoes(D, V, demands, Q, tol=1e-9):
    """(custo ótimo, conjunto de soluções ótimas canônicas, nº de soluções viáveis
    canônicas). A solução é o conjunto de rotas (rótulos de veículo ignorados,
    pois as capacidades são iguais); o sentido de cada rota conta como solução
    distinta, como no POVM."""
    C = D.shape[0] - 1
    custos = {}
    for perm in itertools.permutations(range(1, C + 1)):
        for cuts in itertools.combinations(range(1, C), V - 1):
            idx = [0, *cuts, C]
            rotas = [list(perm[idx[k]:idx[k + 1]]) for k in range(V)]
            if V > 1:
                if any(len(r) == 0 for r in rotas):
                    continue
                if any(sum(demands[c - 1] for c in r) > Q[k] + 1e-9
                       for k, r in enumerate(rotas)):
                    continue
            chave = _canon(rotas)
            if chave not in custos:
                custos[chave] = _custo_rotas(D, rotas)
    if not custos:
        raise InfeasibleProblemError("nenhuma solução viável na enumeração")
    melhor = min(custos.values())
    otimas = {k for k, c in custos.items() if abs(c - melhor) <= tol}
    return float(melhor), otimas, len(custos)


def _hamiltoniano(C, V, D, demands, Q):
    req = V > 1
    inst = H.instance_from_legacy(C, V, D, demands, Q, hbar=HBAR,
                                  require_all_vehicles=req)
    kw = dict(require_all_vehicles=req, lambda_disc=5.0)
    if "use_capacity" in inspect.signature(H.suggest_penalties).parameters:
        kw["use_capacity"] = req
    par = H.suggest_penalties(inst, **kw)
    if req and par.lambda_cap <= 0:
        par = dataclasses.replace(par, lambda_cap=float(par.lambda_gap))
    return inst, par


# ---------------------------------------------------------------------------
# estatísticas do estado e decodificação
# ---------------------------------------------------------------------------
def _distribuicao(ket, inst, ops):
    psi = np.asarray(ket, dtype=np.complex128).reshape((ops.cutoff,) * inst.N)
    psi = psi / np.linalg.norm(psi)
    return np.asarray(H.assignment_distribution(psi, inst, ops), dtype=np.float64)


def _estatisticas(dist, inst, D, optset, req, cap):
    flat = np.clip(dist.reshape(-1), 0.0, None)
    flat = flat / flat.sum()
    p_valid = p_otima = soma = 0.0
    for f in np.flatnonzero(flat > 1e-12):
        slots = np.unravel_index(int(f), dist.shape)
        if not SB.is_valid_assignment(slots, inst, req, cap):
            continue
        w = float(flat[f])
        rotas = SB.routes_from_slots(slots, inst)
        p_valid += w
        soma += w * _custo_rotas(D, rotas)
        if _canon(rotas) in optset:
            p_otima += w
    return dict(p_valid=p_valid, p_otima=p_otima,
                p_ot_cond=(p_otima / p_valid if p_valid > 0 else float("nan")),
                custo_medio_valido=(soma / p_valid if p_valid > 0 else float("nan")))


def _decodificar(dist, inst, D, req, cap, shots, seed):
    """Rota viável de menor custo entre `shots` configurações sorteadas; None se
    nenhuma sorteada for viável."""
    flat = np.clip(dist.reshape(-1), 0.0, None)
    flat = flat / flat.sum()
    rng = np.random.default_rng(seed)
    melhor = None
    for f in np.unique(rng.choice(flat.size, size=int(shots), p=flat)):
        slots = np.unravel_index(int(f), dist.shape)
        if not SB.is_valid_assignment(slots, inst, req, cap):
            continue
        rotas = SB.routes_from_slots(slots, inst)
        custo = _custo_rotas(D, rotas)
        if melhor is None or custo < melhor[0] - 1e-12:
            melhor = (custo, rotas)
    return melhor


def _estado(solver, theta, cutoff):
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    return eng.run(solver.ansatz.build_program(theta)).state


def _p_acerto(p, s):
    return 1.0 - (1.0 - p) ** s


def _rotas_com_deposito(rotas, V):
    if V == 1:
        return [0, *rotas[0], 0]
    return {v + 1: [0, *r, 0] for v, r in enumerate(rotas) if r}


# ---------------------------------------------------------------------------
# figuras
# ---------------------------------------------------------------------------
def plot_convergencia(iterate_history, c_star, e0, cutoff, titulo, destino):
    y = np.asarray(iterate_history, dtype=float)
    x = np.arange(y.size)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color="tab:blue", lw=1.8, label="VQE")
    k = int(np.argmin(y))
    ax.plot([k], [y[k]], "o", ms=5, color="tab:blue",
            label=f"melhor iterado (it. {k})")
    ax.axhline(c_star, ls="--", lw=1.2, color="tab:green",
               label=f"Força bruta: $C^*$ = {c_star:.2f}")
    ax.axhline(e0, ls="--", lw=1.2, color="tab:red",
               label=f"Menor energia de $H$ (cutoff {cutoff}): $E_0$ = {e0:.2f}")
    ax.set_xlabel("Iteração")
    ax.set_ylabel(r"Energia $\langle H \rangle$")
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(destino, dpi=300)
    plt.close(fig)


def plot_decodificacao(checkpoints, c_star, titulo, destino):
    its = [c["it"] for c in checkpoints]
    custo = [c["custo_dec"] for c in checkpoints]
    pac = [c["pacerto_1024"] for c in checkpoints]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(its, custo, "o-", color="tab:blue", lw=1.5, ms=4,
            label="custo da rota decodificada (1024 disparos)")
    ax.axhline(c_star, ls="--", lw=1.2, color="tab:green",
               label=f"Força bruta: $C^*$ = {c_star:.2f}")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Custo da rota")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(its, pac, "-", color="tab:orange", lw=1.2, alpha=0.8,
             label="P(ver a ótima em 1024 disparos)")
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_ylabel("Probabilidade")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(destino, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# tabela
# ---------------------------------------------------------------------------
def _simples(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return json.dumps(v.tolist())
    if isinstance(v, (list, dict, tuple)):
        return json.dumps(v, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return v


def anexar_csv(destino, linha):
    """Acrescenta uma linha; se surgirem colunas novas, reescreve o cabeçalho."""
    linha = {k: _simples(v) for k, v in linha.items()}
    novo = not os.path.exists(destino)
    if not novo:
        with open(destino, newline="", encoding="utf-8") as fh:
            cab = next(csv.reader(fh), [])
        campos = cab + [k for k in linha if k not in cab]
        if campos != cab:
            with open(destino, newline="", encoding="utf-8") as fh:
                antigas = list(csv.DictReader(fh))
            with open(destino, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=campos)
                w.writeheader()
                w.writerows(antigas)
    else:
        campos = list(linha)
    with open(destino, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        if novo:
            w.writeheader()
        w.writerow(linha)


# ---------------------------------------------------------------------------
# verificação prévia das instâncias (segundos; roda antes do VQE)
# ---------------------------------------------------------------------------
def verificar_instancias(tuplas, pasta="vqe_tese", seed=42, Q_val="auto",
                         demand_range=(1, 5), folga=1.2):
    """Para cada (C, V): a instância é viável? C*, nº de soluções ótimas e viáveis,
    demandas e a confirmação do custo ótimo contra o BruteForce. Devolve a lista
    de tuplas viáveis."""
    print("=" * 100)
    print(f"INSTÂNCIAS  (semente {seed}, demandas em {demand_range}, capacidade "
          f"{'construída da demanda (folga ' + str(folga) + ')' if Q_val in (None, 'auto') else Q_val})")
    print("=" * 100)
    print(f"{'C':>3}{'V':>3}{'R':>3}{'Q':>5}{'C*':>10}{'BF':>10}{'#ótimas':>9}{'#viáveis':>10}"
          f"  demandas / situação")
    ok = []
    vistos = {}
    for (C, V, L) in tuplas:
        if (C, V) not in vistos:
            g, D, demands, Q = montar_instancia(C, V, seed, Q_val, demand_range, folga)
            try:
                bf = BruteForce(dist_matrix=g.dist_matrix, num_vehicles=V,
                                capacities=Q, demands=g.full_demands)
                bf_custo, _ = bf.solve()
                custo, otimas, n_viaveis = enumerar_solucoes(D, V, demands, Q)
                inst, _ = _hamiltoniano(C, V, D, demands, Q)
                situacao = ("OK" if abs(custo - bf_custo) < 1e-6
                            else "DIVERGE do BruteForce")
                vistos[(C, V)] = (situacao == "OK")
                linha = dict(C=C, V=V, R=inst.R, c_star=custo, c_star_bruteforce=bf_custo,
                             n_otimas=len(otimas), n_viaveis=n_viaveis,
                             demandas=[float(x) for x in demands], Q=float(Q[0]),
                             situacao=situacao)
                print(f"{C:>3}{V:>3}{inst.R:>3}{float(Q[0]):>5.0f}{custo:>10.3f}{bf_custo:>10.3f}"
                      f"{len(otimas):>9}{n_viaveis:>10}  {list(demands)}  {situacao}")
            except InfeasibleProblemError as err:
                vistos[(C, V)] = False
                linha = dict(C=C, V=V, situacao=f"INVIÁVEL: {err}")
                print(f"{C:>3}{V:>3}{'':>3}  INVIÁVEL: {err}")
            anexar_csv(caminho("instancias.csv", pasta), linha)
        if vistos[(C, V)]:
            ok.append((C, V, L))
    print(f"\n{len(ok)} de {len(tuplas)} tuplas (C, V, L) são viáveis.")
    return ok


# ---------------------------------------------------------------------------
# uma execução
# ---------------------------------------------------------------------------
def executar_experimento(C, V, L, cutoff, max_iter, lr, lr_final, nome_hp="base",
                         pasta="vqe_tese", seed=42, Q_val="auto", demand_range=(1, 5),
                         folga=1.2, fd_eps=1e-2, theta_scale=0.05, checkpoint_cada=None,
                         shots_decodificacao=1024, figura_decodificacao=True):
    exp = f"c{C}_v{V}_l{L}_{nome_hp}"
    checkpoint_cada = checkpoint_cada or max(1, max_iter // 20)
    gm = gerenciador(pasta)
    log = setup_logger(name=exp, log_file=gm.get_file_path(f"logs/{exp}.log"))
    log.info(f"=== {exp}: C={C} V={V} L={L} cutoff={cutoff} it={max_iter} lr={lr}->{lr_final} ===")
    t_total = time.time()
    req, cap = V > 1, V > 1
    resumo_csv = caminho("summary.csv", pasta)
    linha = dict(exp=exp, C=C, V=V, L=L, hp=nome_hp, cutoff=cutoff, max_iter=max_iter,
                 lr=lr, lr_final=lr_final, seed=seed, status="PENDING")
    art = {}

    # 1. instância e força bruta ---------------------------------------------
    g, D, demands, Q = montar_instancia(C, V, seed, Q_val, demand_range, folga)
    linha["Q"] = float(Q[0])
    log.info(f"capacidade por veículo Q={Q[0]:g}  demandas={list(demands)}")
    art["fig_original_graph"] = str(gm.get_file_path(f"{exp}_original_graph.png", is_figure=True))
    g.plot_original_graph(art["fig_original_graph"])
    t0 = time.time()
    try:
        bf = BruteForce(dist_matrix=g.dist_matrix, num_vehicles=V,
                        capacities=Q, demands=g.full_demands)
        bf_custo, bf_rotas = bf.solve()
        custo_otimo, optset, n_viaveis = enumerar_solucoes(D, V, demands, Q)
    except InfeasibleProblemError as err:
        log.error(f"inviável: {err}")
        linha.update(status="INFEASIBLE", erro=str(err))
        anexar_csv(resumo_csv, linha)
        return linha
    t_bf = time.time() - t0
    if abs(custo_otimo - bf_custo) > 1e-6:
        log.warning(f"custo ótimo da enumeração ({custo_otimo}) difere do BruteForce ({bf_custo})")
    art["fig_bf_route"] = str(gm.get_file_path(f"{exp}_bf_route.png", is_figure=True))
    g.plot_solution_graph(bf_rotas, art["fig_bf_route"], title_prefix="Solução Força Bruta")
    log.info(f"força bruta: C*={bf_custo:.4f} rotas={bf_rotas} #ótimas={len(optset)} "
             f"#viáveis={n_viaveis} ({format_timespan(t_bf)})")

    # 2. Hamiltoniano, referência exata e linha de base uniforme ---------------
    inst, par = _hamiltoniano(C, V, D, demands, Q)
    ops = H.build_modular_operators(cutoff, inst.M, inst.R, HBAR)
    t0 = time.time()
    e0 = float(HS.ground_state_sparse(inst, ops, par)["energy"])
    t_gs = time.time() - t0
    X = inst.M * inst.R
    dist_u = np.full((X,) * C, 1.0 / float(X) ** C)
    est_u = _estatisticas(dist_u, inst, D, optset, req, cap)
    log.info(f"menor energia de H (cutoff {cutoff}): E0={e0:.4f} ({format_timespan(t_gs)}); "
             f"P(ótima) uniforme={est_u['p_otima']:.5f}")

    # 3. VQE -------------------------------------------------------------------
    pi = ProblemInstance(V=V, C=C, D=D, demands=demands, Q=Q)
    solver = VQESolver(pi, CircuitConfig(C, L), par, cutoff=cutoff)
    theta0 = solver.ansatz.generate_initial_theta(seed=seed, scale=theta_scale)
    log.info(f"VQE: P={len(theta0)} parâmetros, {2 * len(theta0) + 1} avaliações/iteração")
    checkpoints = []

    def cb(t, theta, energia):
        if t % checkpoint_cada != 0 and t != max_iter:
            return
        dist = _distribuicao(_estado(solver, theta, cutoff).ket(), inst, ops)
        est = _estatisticas(dist, inst, D, optset, req, cap)
        dec = _decodificar(dist, inst, D, req, cap, shots_decodificacao, seed)
        cp = dict(it=t, energia=energia, p_valid=est["p_valid"], p_otima=est["p_otima"],
                  pacerto_1024=_p_acerto(est["p_otima"], 1024),
                  custo_dec=(dec[0] if dec is not None else float("nan")))
        checkpoints.append(cp)
        log.info(f"it {t:>4}  E={energia:9.3f}  P(válida)={est['p_valid']:.4f}  "
                 f"P(ótima)={est['p_otima']:.5f}  custo decodificado={cp['custo_dec']:.3f}")

    try:
        t0 = time.time()
        m = solver.solve(method="ADAM", maxiter=max_iter, lr=lr, lr_final=lr_final,
                         fd_eps=fd_eps, initial_theta=theta0, callback=cb)
        t_vqe = time.time() - t0
    except Exception as err:                       # registra e segue a varredura
        log.error("falha no VQE:\n" + traceback.format_exc())
        linha.update(status="ERROR", erro=repr(err), c_star=bf_custo, E0=e0)
        anexar_csv(resumo_csv, linha)
        return linha

    # 4. estado final, decodificação e métricas --------------------------------
    dist = _distribuicao(_estado(solver, m.optimal_theta, cutoff).ket(), inst, ops)
    est = _estatisticas(dist, inst, D, optset, req, cap)
    dec = _decodificar(dist, inst, D, req, cap, shots_decodificacao, seed)
    if dec is None:
        custo_dec, rotas_dec, viavel = float("nan"), None, False
    else:
        custo_dec, rotas_dec, viavel = dec[0], dec[1], True
    ok_custo = bool(viavel and abs(custo_dec - bf_custo) < 1e-9)
    ok_rota = bool(viavel and _canon(rotas_dec) in optset)
    pac = {s: _p_acerto(est["p_otima"], s) for s in SHOTS}
    it_acerto = next((c["it"] for c in checkpoints if c["pacerto_1024"] >= LIMIAR_ACERTO), -1)
    rho = est["p_otima"] / est_u["p_otima"] if est_u["p_otima"] > 0 else float("nan")
    hist = np.asarray(m.iterate_history, dtype=float)
    log.info(f"VQE: E_final={hist[-1]:.3f} E_min={hist.min():.3f} E0={e0:.3f} | decodificado: "
             f"custo={custo_dec} (C*={bf_custo:.4f}) igual={ok_custo} | rho={rho:.2f} "
             f"| {format_timespan(t_vqe)}")

    # 5. figuras ---------------------------------------------------------------
    if rotas_dec is not None:
        art["fig_vqe_route"] = str(gm.get_file_path(f"{exp}_vqe_route.png", is_figure=True))
        g.plot_solution_graph(_rotas_com_deposito(rotas_dec, V), art["fig_vqe_route"],
                              title_prefix="Solução VQE")
    titulo = f"Convergência do VQE — C={C}, V={V}, L={L} (cutoff {cutoff})"
    art["fig_convergence"] = str(gm.get_file_path(f"{exp}_convergence.png", is_figure=True))
    plot_convergencia(hist, bf_custo, e0, cutoff, titulo, art["fig_convergence"])
    if figura_decodificacao and checkpoints:
        art["fig_decodificacao"] = str(gm.get_file_path(f"{exp}_decodificacao.png", is_figure=True))
        plot_decodificacao(checkpoints, bf_custo,
                           f"Solução decodificada — C={C}, V={V}, L={L} (cutoff {cutoff})",
                           art["fig_decodificacao"])

    # 6. arquivos de dados --------------------------------------------------------
    art["npz"] = str(gm.get_file_path(f"{exp}_state.npz", subdir="arrays"))
    np.savez_compressed(
        art["npz"], theta0=theta0, theta_final=np.asarray(m.optimal_theta),
        iterate_history=hist, grad_norm_history=np.asarray(m.grad_norm_history),
        lr_history=np.asarray(m.lr_history),
        cp_iter=np.array([c["it"] for c in checkpoints]),
        cp_energia=np.array([c["energia"] for c in checkpoints]),
        cp_p_valid=np.array([c["p_valid"] for c in checkpoints]),
        cp_p_otima=np.array([c["p_otima"] for c in checkpoints]),
        cp_custo_dec=np.array([c["custo_dec"] for c in checkpoints]),
        dist_final=dist, coords=g.graph.coords, D=D, demands=demands, Q=Q)

    resultados = dict(
        c_star=bf_custo, n_otimas=len(optset), n_viaveis=n_viaveis, E0=e0,
        E_vqe_final=float(hist[-1]), E_vqe_min=float(hist.min()),
        dE_final_E0=float(hist[-1] - e0), custo_dec=custo_dec, viavel=int(viavel),
        ok_custo=int(ok_custo), ok_rota=int(ok_rota), p_valid=est["p_valid"],
        p_otima=est["p_otima"], p_ot_cond=est["p_ot_cond"], rho=rho,
        p_otima_uniforme=est_u["p_otima"], custo_medio_valido=est["custo_medio_valido"],
        razao_custo_medio=est["custo_medio_valido"] / bf_custo,
        pacerto_16=pac[16], pacerto_64=pac[64], pacerto_256=pac[256],
        pacerto_1024=pac[1024], it_acerto=it_acerto, params=len(theta0),
        avaliacoes=m.circuit_evaluations, tempo_bruteforce=t_bf, tempo_gs=t_gs,
        tempo_vqe=t_vqe)
    payload = dict(
        experimento=exp, status="OK",
        config=dict(C=C, V=V, L=L, cutoff=cutoff, max_iter=max_iter, lr=lr,
                    lr_final=lr_final, fd_eps=fd_eps, seed=seed, Q_val=str(Q_val),
                    Q=Q.tolist(), folga=folga,
                    demand_range=list(demand_range), theta_scale=theta_scale,
                    checkpoint_cada=checkpoint_cada, shots=shots_decodificacao,
                    hamiltoniano=dataclasses.asdict(par), R=inst.R),
        instancia=dict(coords=g.graph.coords, D=D, demands=demands, Q=Q),
        forca_bruta=dict(custo=bf_custo, rotas=bf_rotas, n_otimas=len(optset),
                         n_viaveis=n_viaveis, tempo=t_bf),
        solucao_vqe=dict(rotas=(_rotas_com_deposito(rotas_dec, V) if rotas_dec else None),
                         custo=custo_dec, viavel=viavel, igual_ao_otimo=ok_custo),
        resultados=resultados, checkpoints=checkpoints, artefatos=art)
    art["json"] = str(gm.get_file_path(f"{exp}.json", subdir="runs"))
    save_experiment_json(payload, art["json"])

    linha.update(status="OK", **resultados,
                 tempo_total=time.time() - t_total,
                 rotas_bruteforce=bf_rotas,
                 rotas_vqe=(_rotas_com_deposito(rotas_dec, V) if rotas_dec else None),
                 **{f"arquivo_{k}": v for k, v in art.items()})
    anexar_csv(resumo_csv, linha)
    return linha


# ---------------------------------------------------------------------------
# varredura
# ---------------------------------------------------------------------------
def varredura_tese(tuplas, hiperparametros=None, pasta="vqe_tese", cutoff_base=None,
                   **kw):
    """tuplas: lista de (C, V, L). hiperparametros: {nome: dict(max_iter, lr,
    lr_final, e UMA das chaves de cutoff)}:
        cutoff_rel_C   cutoff = C + k   (k = 0: cutoff igual ao nº de cidades)
        cutoff_extra   cutoff = cutoff base de C + k   (padrão: 0)
    O nome do conjunto entra nos nomes dos arquivos: use letras, dígitos e '_'."""
    hiperparametros = hiperparametros or HIPERPARAMETROS_BASE
    cutoff_base = cutoff_base or CUTOFF_BASE
    linhas = []
    for (C, V, L) in tuplas:
        for nome, hp in hiperparametros.items():
            if "cutoff_rel_C" in hp:
                cutoff = C + int(hp["cutoff_rel_C"])
            else:
                cutoff = cutoff_base[C] + hp.get("cutoff_extra", 0)
            linhas.append(executar_experimento(
                C, V, L, cutoff, hp["max_iter"], hp["lr"], hp.get("lr_final"),
                nome_hp=nome, pasta=pasta, **kw))
    return linhas


def resumo_tese(linhas):
    print("\n" + "=" * 118)
    print("RESUMO: custo decodificado == C* (viável)?   ok = igual   rho = P(ótima)/P(ótima uniforme)")
    print("=" * 118)
    print(f"{'exp':<22}{'C*':>9}{'custo_dec':>11}{'ok':>5}{'E0':>10}{'E_final':>10}"
          f"{'rho':>8}{'P@16':>8}{'P@1024':>8}{'it_ac':>7}{'t_vqe':>9}")
    for r in linhas:
        if r.get("status") != "OK":
            print(f"{r['exp']:<22}  {r['status']}")
            continue
        print(f"{r['exp']:<22}{r['c_star']:>9.3f}{r['custo_dec']:>11.3f}"
              f"{'sim' if r['ok_custo'] else 'nao':>5}{r['E0']:>10.2f}"
              f"{r['E_vqe_final']:>10.2f}{r['rho']:>8.2f}{r['pacerto_16']:>8.3f}"
              f"{r['pacerto_1024']:>8.3f}{r['it_acerto']:>7}"
              f"{format_timespan(r['tempo_vqe']):>9}")