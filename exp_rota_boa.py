"""
exp_rota_boa.py — o VQE entrega uma rota VÁLIDA e BOA (entre as melhores da força
bruta), mesmo sem a menor energia coincidir com o custo ótimo C*?

As métricas posicionam a rota decodificada na lista COMPLETA de soluções viáveis
da força bruta (exp_vqe_tese.enumerar_custos) e comparam cada fonte com duas
referências, calculadas exatamente (sem simulação extra):

    uni   distribuição uniforme sobre as configurações (o acaso)
    ini   estado inicial do VQE (it = 0, antes do treino)
    fin   estado final do VQE
    gs    estado fundamental exato de H (o melhor que a codificação permite)

Por fonte (prefixo q_<fonte>_):
    p_valid           probabilidade de uma configuração válida
    p_top1/3/5pct     massa nas soluções de posição <= 1, <= 3 e <= k5 (5% melhores)
    custo{n}, gap{n}  melhor rota válida em n disparos (n = 16, 64, 1024) e o gap
                      relativo (custo - C*)/C* em %
    rank{n}           posição dessa rota na lista ordenada (1 = ótima; empates
                      dividem a mesma posição)
    top5_{n}          1 se rank{n} <= k5
    p_top5_em_{n}     P(a melhor de n disparos cair nas 5% melhores) = 1 - (1 - p_top5pct)^n

Colunas gerais:
    q_n_solucoes, q_k5          nº de soluções viáveis e tamanho do grupo "5% melhores"
    q_rho_top5                  p_top5pct(fin) / p_top5pct(uni)
    q_rho_top5_gs               p_top5pct(gs)  / p_top5pct(uni)
    q_treino_top5               1 se p_top5pct(fin) > p_top5pct(ini)
    q_rota_boa                  CRITÉRIO (fixado antes de rodar): rota válida com 64
                                disparos E entre as 5% melhores E q_rho_top5 >= 2

Este módulo não roda nada sozinho: é chamado por exp_vqe_tese.executar_experimento
quando metricas_qualidade=True (bloco "rota_boa" do exp_run.py).
"""
import math

import numpy as np

import exp_vqe_tese as TV
import qumodes.sf_bridge as SB

SHOTS_QUALIDADE = (16, 64, 1024)
FRACAO_TOPO = 0.05
LIMIAR_RHO_TOPO = 2.0
SHOTS_CRITERIO = 64


def _ranks(tabela):
    """{solução: posição}, com empates na mesma posição (1 = ótima)."""
    custos = sorted(tabela.values())
    return {k: 1 + sum(1 for c in custos if c < v - 1e-9) for k, v in tabela.items()}


def _rank_custo(custo, custos_ordenados):
    return 1 + sum(1 for c in custos_ordenados if c < custo - 1e-9)


def _massas(dist, inst, req, cap):
    """{solução canônica: massa} sobre as configurações VÁLIDAS e a P(válida)."""
    massas = {}
    if isinstance(dist, TV.DistValida):
        pares = zip(dist.cfgs, dist.probs)
    else:
        flat = np.clip(dist.reshape(-1), 0.0, None)
        flat = flat / flat.sum()
        pares = []
        for f in np.flatnonzero(flat > 1e-12):
            slots = np.unravel_index(int(f), dist.shape)
            if SB.is_valid_assignment(slots, inst, req, cap):
                pares.append((slots, float(flat[f])))
    for slots, w in pares:
        if w <= 1e-15:
            continue
        k = TV._canon(SB.routes_from_slots(slots, inst))
        massas[k] = massas.get(k, 0.0) + float(w)
    return massas, sum(massas.values())


def qualidade(dist, inst, D, req, cap, tabela, seed=42):
    """Métricas de qualidade de UMA distribuição (ver docstring do módulo)."""
    ranks = _ranks(tabela)
    custos_ord = sorted(tabela.values())
    c_star = custos_ord[0]
    k5 = max(1, math.ceil(FRACAO_TOPO * len(tabela)))
    massas, p_valid = _massas(dist, inst, req, cap)

    def massa_ate(k):
        return sum(m for s, m in massas.items() if ranks.get(s, 10 ** 9) <= k)

    q = dict(p_valid=p_valid, p_top1=massa_ate(1), p_top3=massa_ate(3),
             p_top5pct=massa_ate(k5))
    for n in SHOTS_QUALIDADE:
        dec = TV._decodificar(dist, inst, D, req, cap, n, seed)
        if dec is None:
            q.update({f"custo{n}": float("nan"), f"gap{n}": float("nan"),
                      f"rank{n}": -1, f"top5_{n}": 0})
        else:
            r = _rank_custo(dec[0], custos_ord)
            q.update({f"custo{n}": float(dec[0]),
                      f"gap{n}": 100.0 * (dec[0] - c_star) / c_star,
                      f"rank{n}": int(r), f"top5_{n}": int(r <= k5)})
        q[f"p_top5_em_{n}"] = 1.0 - (1.0 - q["p_top5pct"]) ** n
    return q, k5


def colunas_qualidade(fontes, inst, D, req, cap, tabela, seed=42, log=None):
    """Colunas q_* para o summary.csv. fontes: {"uni"|"ini"|"fin"|"gs": dist ou None}."""
    out = dict(q_n_solucoes=len(tabela))
    por_fonte = {}
    for nome, dist in fontes.items():
        if dist is None:
            continue
        q, k5 = qualidade(dist, inst, D, req, cap, tabela, seed)
        out["q_k5"] = k5
        por_fonte[nome] = q
        out.update({f"q_{nome}_{k}": v for k, v in q.items()})

    def razao(a, b):
        if a not in por_fonte or b not in por_fonte or por_fonte[b]["p_top5pct"] <= 0:
            return float("nan")
        return por_fonte[a]["p_top5pct"] / por_fonte[b]["p_top5pct"]

    out["q_rho_top5"] = razao("fin", "uni")
    out["q_rho_top5_gs"] = razao("gs", "uni")
    out["q_treino_top5"] = int("fin" in por_fonte and "ini" in por_fonte
                               and por_fonte["fin"]["p_top5pct"] > por_fonte["ini"]["p_top5pct"])
    fin = por_fonte.get("fin", {})
    rho = out["q_rho_top5"]
    out["q_rota_boa"] = int(fin.get(f"rank{SHOTS_CRITERIO}", -1) > 0
                            and fin.get(f"top5_{SHOTS_CRITERIO}", 0) == 1
                            and not math.isnan(rho) and rho >= LIMIAR_RHO_TOPO)
    if log is not None:
        log.info(f"qualidade da rota: {len(tabela)} soluções viáveis, 5% melhores = posição <= "
                 f"{out.get('q_k5')}; fin: rank@64={fin.get('rank64')} gap@64="
                 f"{fin.get('gap64', float('nan')):.2f}%  rho_top5={rho:.2f}  "
                 f"rota boa={bool(out['q_rota_boa'])}")
    return out


def resumo_rota_boa(linhas):
    print("\n" + "=" * 132)
    print("A ROTA DO VQE ESTÁ ENTRE AS MELHORES DA FORÇA BRUTA?   rank = posição entre as "
          "soluções viáveis (1 = ótima); gap = (custo - C*)/C*")
    print(f"critério 'boa': rota válida com {SHOTS_CRITERIO} disparos, entre as 5% melhores, "
          f"e massa nas 5% melhores >= {LIMIAR_RHO_TOPO:g}x a do acaso")
    print("=" * 132)
    print(f"{'exp':<24}{'#sol':>6}{'k5':>4}{'Q':>9}"
          f"{'rk@16':>7}{'rk@64':>7}{'rk@1024':>8}{'gap@64%':>9}"
          f"{'top5 uni':>10}{'top5 ini':>10}{'top5 fin':>10}{'top5 gs':>9}"
          f"{'rho':>7}{'treino':>8}{'boa':>5}")
    for r in linhas:
        if r.get("status") != "OK":
            print(f"{r.get('exp', '?'):<24}  {r.get('status')}")
            continue
        if "q_rota_boa" not in r:
            continue
        f = lambda k: r.get(k, float("nan"))
        print(f"{r['exp']:<24}{r['q_n_solucoes']:>6}{r['q_k5']:>4}{str(r.get('Q', '')):>9}"
              f"{f('q_fin_rank16'):>7}{f('q_fin_rank64'):>7}{f('q_fin_rank1024'):>8}"
              f"{f('q_fin_gap64'):>9.2f}"
              f"{f('q_uni_p_top5pct'):>10.4f}{f('q_ini_p_top5pct'):>10.4f}"
              f"{f('q_fin_p_top5pct'):>10.4f}{f('q_gs_p_top5pct'):>9.4f}"
              f"{f('q_rho_top5'):>7.2f}{'sim' if r['q_treino_top5'] else 'nao':>8}"
              f"{'sim' if r['q_rota_boa'] else 'nao':>5}")