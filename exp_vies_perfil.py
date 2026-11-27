"""
exp_vies_perfil.py — o caso de capacidades desiguais (12/8) é uma instância do VIÉS
DE PERFIL? (pendência do artigo: "[A DEFINIR: confirmar se o caso de capacidades
desiguais é uma instância desse viés antes de afirmar a ligação]")

O viés de perfil (caracterizado em C=4, V=2)
---------------------------------------------
O perfil de uma solução é o vetor de números de cidades por rota (ex.: (2, 3) ou
(1, 4)). Com cutoff finito, cada perfil carrega uma SOBRA DE ENERGIA de codificação
diferente, e o fundamental passa a preferir o perfil de menor sobra, escolhendo a
melhor rota DESSE perfil em vez da ótima. Assinaturas, todas vistas em C=4:
  - o perfil da solução mais provável difere do perfil da ótima;
  - a solução mais provável é a de MENOR CUSTO do seu perfil;
  - a diferença de sobra entre perfis supera a diferença de custo Δc entre a
    ótima e a melhor do perfil favorecido;
  - essa diferença vem do termo de LACUNA (verificação "fantasma-lacuna").

A explicação concorrente no caso 12/8 é um VIÉS DE CAPACIDADE: a truncagem
penalizar rotas com carga perto do limite do veículo menor (8). Ela tem a assinatura
oposta: a diferença de sobra vem do termo de CAPACIDADE, não do de lacuna.

Critérios (fixados antes de rodar)
-----------------------------------
  C1  o perfil da solução mais provável do fundamental difere do perfil da ótima
  C2  a solução mais provável é a de menor custo do seu perfil
  C3  no estado-produto dos nós, a sobra média do perfil favorecido é menor, e a
      diferença entre perfis supera Δc
  C4  o termo que mais contribui para essa diferença é a lacuna (gap)
  Veredito: C1 e C2 e C3 e C4             -> instância do viés de perfil
            C1 e C2 e C3, termo = capacidade -> viés de capacidade (outro mecanismo)
            C1 falso                          -> não é viés de perfil (o erro é
                                                 dentro do mesmo perfil)
A versão de capacidade IGUAL (10/10) é o controle: lá o fundamental acerta a ótima.

Duas partes
-----------
  A  fundamental (C1, C2): lida do runs/energia_cutoff_c5_v2.json do bloco
     energia_configuracao (chave "perfis"), que precisa ter rodado antes na mesma
     subpasta. Sem o JSON, a parte A é pulada e o veredito fica parcial.
  B  estado-produto dos nós (C3, C4): para cada configuração válida, o estado em que
     cada cidade está no seu nó; <H> por termo via hamiltonian.energy_components
     (o mesmo avaliador da diagonalização; product_energy.py não serve aqui porque
     não suporta lambda_cap > 0). Sobra = <H> - custo da rota. Não diagonaliza:
     só d^C amplitudes por configuração (d = 13: 371 mil).
Saídas (PathManager, mesma pasta do bloco energia_configuracao):
  vies_perfil.csv   uma linha por (cutoff, perfil): sobra média e por termo
  runs/vies_perfil_c{C}_v{V}.json
"""
import json
import time

import numpy as np

import exp_vqe_tese as TV
import qumodes.hamiltonian as H
import qumodes.sf_bridge as SB
from exp_energia_cutoff import _perfil
from exp_vqe_tese import HBAR, _canon, _custo_rotas, _hamiltoniano, _q_texto, montar_instancia
from saidas import caminho, gerenciador
from utils import save_experiment_json

TERMOS = ("dist", "col", "gap", "vehicle", "capacity", "disc")


def _produto_nos(kets, slots, R):
    psi = None
    for s in slots:
        k = kets[divmod(int(s), R)]
        psi = k if psi is None else np.multiply.outer(psi, k)
    return psi


def sobra_por_perfil(inst, ops, par, D, cfgs, req):
    """{perfil: médias de sobra total e por termo} e a configuração de menor <H>
    de cada perfil, sobre todas as configurações válidas."""
    kets = {(v, r): H.node_state(ops, v, r) for v in range(inst.M) for r in range(inst.R)}
    acc = {}
    for slots in cfgs:
        rotas = SB.routes_from_slots(slots, inst)
        custo = _custo_rotas(D, rotas)
        ec = H.energy_components(_produto_nos(kets, slots, inst.R), inst, ops, par, req)
        perf = _perfil(_canon(rotas))
        a = acc.setdefault(perf, dict(n=0, sobra=0.0, **{t: 0.0 for t in TERMOS},
                                      melhor_energia=float("inf"), melhor_custo=None))
        a["n"] += 1
        a["sobra"] += float(ec["encoding"]) - custo
        for t in TERMOS:
            a[t] += float(ec[t]) - (custo if t == "dist" else 0.0)
        if float(ec["encoding"]) < a["melhor_energia"]:
            a["melhor_energia"], a["melhor_custo"] = float(ec["encoding"]), custo
    for a in acc.values():
        n = a.pop("n")
        a["n_config"] = n
        for k in ("sobra",) + TERMOS:
            a[k] /= n
    return acc


def _ler_perfis_gs(pasta, C, V):
    """(perfis, topk) do JSON do bloco energia_configuracao; dicionários vazios se
    não houver. JSONs gerados antes da chave "perfis" ainda têm "topk", que basta
    para C1 e C2 (a solução mais provável é a primeira de topk[d]["solucoes"])."""
    try:
        arq = gerenciador(pasta).get_file_path(f"energia_cutoff_c{C}_v{V}.json", subdir="runs")
        with open(arq, encoding="utf-8") as fh:
            dados = json.load(fh)
        return dados.get("perfis", {}), dados.get("topk", {})
    except (OSError, ValueError):
        return {}, {}


def teste_vies(C=5, V=2, cutoffs=(5, 7, 9, 13), pasta="vqe_cutoff_config/desigual",
               Q_val="desigual", folga=1.2, seed=42, demand_range=(1, 5)):
    g, D, demands, Q = montar_instancia(C, V, seed, Q_val, demand_range, folga)
    inst, par = _hamiltoniano(C, V, D, demands, Q)
    req = cap = V > 1
    tabela = TV.enumerar_custos(D, V, demands, Q)
    c_star = min(tabela.values())
    otimas = [k for k, c in tabela.items() if abs(c - c_star) < 1e-9]
    melhor_por_perfil = {}
    for k, c in tabela.items():
        p = _perfil(k)
        melhor_por_perfil[p] = min(melhor_por_perfil.get(p, float("inf")), c)
    cfgs = TV.configuracoes_validas(inst, req, cap)
    perf_ot = sorted({_perfil(k) for k in otimas})
    gs, topk_gs = _ler_perfis_gs(pasta, C, V)
    rotulo = lambda p: "-".join(map(str, p))
    for d_txt, tk in topk_gs.items():          # parte A a partir do topk, se faltar "perfis"
        if d_txt in gs or not tk.get("solucoes"):
            continue
        s0 = tk["solucoes"][0]
        p0 = _perfil(_canon(s0["rotas"]))
        gs[d_txt] = dict(perfil_top=rotulo(p0), perfil_otima=[rotulo(p) for p in perf_ot],
                         top_custo=s0["custo"],
                         top_e_melhor_do_perfil=bool(abs(s0["custo"] - melhor_por_perfil[p0])
                                                     < 1e-9))

    print("=" * 108)
    print(f"VIÉS DE PERFIL?  C={C} V={V} Q={_q_texto(Q)} demandas={list(demands)}  "
          f"C*={c_star:.3f}  perfil(s) da ótima: {perf_ot}")
    print("  melhor custo por perfil: " + "   ".join(
        f"{p}: {c:.3f}" for p, c in sorted(melhor_por_perfil.items())))
    print(f"  {len(cfgs)} configurações válidas; parte A (fundamental) "
          f"{'lida do JSON' if gs else 'AUSENTE: rode energia_configuracao antes'}")
    print("=" * 108)

    destino = caminho("vies_perfil.csv", pasta)
    resultado = dict(C=C, V=V, Q=Q.tolist(), c_star=c_star,
                     perfil_otima=[list(p) for p in perf_ot],
                     melhor_por_perfil={"-".join(map(str, p)): c
                                        for p, c in melhor_por_perfil.items()},
                     por_cutoff={})
    for d in cutoffs:
        t0 = time.time()
        ops = H.build_modular_operators(d, inst.M, inst.R, HBAR)
        acc = sobra_por_perfil(inst, ops, par, D, cfgs, req)
        favorecido = min(acc, key=lambda p: acc[p]["melhor_energia"])
        print(f"\n  d={d}  ({time.time() - t0:.0f} s)   sobra média por perfil "
              f"(<H> do estado-produto - custo), por termo:")
        print(f"  {'perfil':>8}{'n':>5}{'sobra':>10}" + "".join(f"{t:>10}" for t in TERMOS)
              + f"{'min <H>':>10}{'custo':>9}")
        for p, a in sorted(acc.items()):
            print(f"  {str(p):>8}{a['n_config']:>5}{a['sobra']:>10.2f}"
                  + "".join(f"{a[t]:>10.2f}" for t in TERMOS)
                  + f"{a['melhor_energia']:>10.2f}{a['melhor_custo']:>9.2f}")
            TV.anexar_csv(destino, dict(C=C, V=V, Q=_q_texto(Q), cutoff=d,
                                        perfil="-".join(map(str, p)), **a))
        # C3 e C4 (estado-produto): favorecido x perfil da ótima
        p_ot = perf_ot[0]
        c3 = c4 = None
        termo_dom = None
        if favorecido != p_ot:
            dif = {t: acc[p_ot][t] - acc[favorecido][t] for t in TERMOS}
            dif_total = acc[p_ot]["sobra"] - acc[favorecido]["sobra"]
            delta_c = melhor_por_perfil[favorecido] - c_star
            termo_dom = max(dif, key=dif.get)
            c3 = bool(dif_total > delta_c)
            c4 = termo_dom == "gap"
            print(f"  estado-produto favorece {favorecido} (perfil da ótima: {p_ot}); "
                  f"diferença de sobra = {dif_total:.2f} contra Δc = {delta_c:.2f}; "
                  f"termo dominante: {termo_dom} ({dif[termo_dom]:.2f})")
        else:
            print(f"  estado-produto favorece o perfil da ótima ({p_ot})")
        # C1 e C2 (fundamental), se o JSON tiver este cutoff
        pf = gs.get(str(d))
        c1 = c2 = None
        if pf:
            c1 = pf["perfil_top"] not in pf["perfil_otima"]
            c2 = bool(pf["top_e_melhor_do_perfil"])
            print(f"  fundamental: solução mais provável no perfil {pf['perfil_top']} "
                  f"(custo {pf['top_custo']:.3f}); ótima em {pf['perfil_otima']}; "
                  f"é a melhor do seu perfil: {'sim' if c2 else 'não'}")
        if c1 is None:
            veredito = "parcial: falta a parte A (fundamental) neste cutoff"
        elif not c1:
            veredito = ("não é viés de perfil: o fundamental fica no perfil da ótima"
                        + (" e acerta" if pf["top_custo"] - c_star < 1e-9 else
                           ", mas erra dentro do perfil"))
        elif c2 and c3 and c4:
            veredito = "INSTÂNCIA DO VIÉS DE PERFIL (C1, C2, C3 e C4)"
        elif c2 and c3 and termo_dom == "capacity":
            veredito = "VIÉS DE CAPACIDADE: muda o perfil, mas pelo termo de capacidade"
        else:
            veredito = (f"inconclusivo: C1={c1} C2={c2} C3={c3} C4={c4} "
                        f"(termo dominante {termo_dom})")
        print(f"  VEREDITO d={d}: {veredito}")
        resultado["por_cutoff"][str(d)] = dict(
            perfis={"-".join(map(str, p)): a for p, a in acc.items()},
            favorecido_produto="-".join(map(str, favorecido)), termo_dominante=termo_dom,
            C1=c1, C2=c2, C3=c3, C4=c4, veredito=veredito)
    save_experiment_json(resultado, str(gerenciador(pasta).get_file_path(
        f"vies_perfil_c{C}_v{V}.json", subdir="runs")))
    return resultado