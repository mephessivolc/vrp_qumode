"""
verify_encoding.py — o argmin do Hamiltoniano decodifica para uma rota ótima?

Por que não basta comparar energias
-----------------------------------
`check_penalties` compara grid_min_energy com best_feasible_cost, e
`matches_brute_force` compara custos. Ambos são necessários, nenhum é
suficiente: uma configuração pode ter energia igual a C* e ainda assim
decodificar para OUTRA rota, se o mapa (theta, phi) -> rota estiver trocado.
Aqui a comparação é entre CONJUNTOS DE ROTAS.

Independência
-------------
O lado clássico enumera rotas diretamente (permutações das cidades + cortes
entre veículos), sem passar por `decode` nem pela grade de nós do Hamiltoniano.
Se os dois lados usassem o mesmo mapa, o teste seria circular.

Os cinco testes
---------------
T1 INTERSECÇÃO    argmin(H) contém pelo menos uma rota ótima
T2 CORRETUDE      argmin(H) SÓ contém rotas ótimas (sem mínimo espúrio)
T3 ENERGIA        min H == C* numericamente
T4 CONTÍNUO       nenhum mínimo fora dos nós abaixo de min H (lambda_disc ok)
T5 COMPLETUDE     toda rota ótima aparece em argmin(H)

`passed` exige T1..T4. T5 falhando não é erro: indica degenerescência quebrada
por alguma penalidade, e vale registrar qual.

Uso em run.py
-------------
    import verify_encoding as VE

    verifier = VE.get_verifier(inst, require_all_vehicles=req_all,
                               use_capacity=h_params.lambda_cap > 0)
    enc_report = verifier.verify(h_params, continuous_starts=50, logger=logger)
    payload["encoding"]["verification"] = enc_report
    ...
    payload["analysis"]["routes"].update(verifier.audit_routes(metrics.best_routes))

O lado clássico é enumerado UMA vez por instância e fica em cache, de modo que
uma varredura de lambda paga a enumeração uma só vez.

Custo
-----
A enumeração clássica é O(N! * C(N-1, M-1)). O verificador mede esse número
antes de começar e, acima de `max_sequences`, roda apenas T3 e T4 (que não
dependem dela), marcando o laudo como parcial em vez de falhar.
"""
from itertools import combinations, permutations
from math import comb, factorial
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

import qumodes.hamiltonian as H

Canon = Tuple[Tuple[int, ...], ...]

DEFAULT_MAX_SEQUENCES = 500_000
DEFAULT_MAX_CONFIGS = 2_000_000   # tamanho da grade de nós (R*M)^N
TEST_NAMES = ("T1_intersection", "T2_no_spurious", "T3_energy_matches",
              "T4_no_offnode_minimum", "T5_all_optima_reachable")


# ---------------------------------------------------------------------------
# Forma canônica
# ---------------------------------------------------------------------------
def is_symmetric(D: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(np.allclose(D, D.T, atol=tol))


def canonical(routes: Sequence[Sequence[int]], symmetric: bool) -> Canon:
    """Rotas em forma canônica, para comparar conjuntos.

    Remove rotas vazias, inverte cada rota para o menor representante quando D é
    simétrica (ida e volta custam o mesmo) e ordena as rotas entre si, já que
    trocar os rótulos dos veículos não muda a solução física.
    """
    out: List[Tuple[int, ...]] = []
    for r in routes:
        t = tuple(int(c) for c in r)
        if not t:
            continue
        if symmetric:
            t = min(t, t[::-1])
        out.append(t)
    return tuple(sorted(out))


def canon_to_str(c: Optional[Canon]) -> Optional[str]:
    """Forma canônica como texto, para caber em JSON e no CSV."""
    if c is None:
        return None
    return " | ".join("-".join(str(i) for i in route) for route in c)


def normalize_routes(routes) -> List[List[int]]:
    """Aceita lista de rotas, dict {veículo: rota} ou rota única."""
    if routes is None:
        return []
    if isinstance(routes, dict):
        return [list(routes[k]) for k in sorted(routes)]
    if len(routes) and not isinstance(routes[0], (list, tuple, np.ndarray)):
        return [list(routes)]
    return [list(r) for r in routes]


def enumeration_size(N: int, M: int) -> int:
    """Número de sequências que a enumeração clássica percorre."""
    return factorial(N) * comb(N - 1, M - 1) if N >= M >= 1 else 0


# ---------------------------------------------------------------------------
# Verificador
# ---------------------------------------------------------------------------
class EncodingVerifier:
    """Verifica a codificação de uma instância; reutilizável entre lambdas.

    O conjunto ótimo clássico depende só da instância (e das regras de
    viabilidade), nunca dos pesos do Hamiltoniano. Por isso ele é calculado na
    primeira chamada e reaproveitado em todas as seguintes.
    """

    def __init__(self, inst: H.VRPInstance, require_all_vehicles: bool = False,
                 use_capacity: bool = False, tol: float = 1e-6,
                 max_sequences: int = DEFAULT_MAX_SEQUENCES,
                 max_configs: int = DEFAULT_MAX_CONFIGS):
        self.inst = inst
        self.require_all_vehicles = bool(require_all_vehicles)
        self.use_capacity = bool(use_capacity)
        self.tol = float(tol)
        self.max_sequences = int(max_sequences)
        self.max_configs = int(max_configs)
        self.symmetric = is_symmetric(inst.D)
        # Dois custos independentes, cada um com seu teto:
        #   n_sequences  enumeração clássica, O(N! * C(N-1, M-1))
        #   n_configs    grade de nós do Hamiltoniano, (R*M)^N
        # Acima do teto, o teste correspondente é pulado em vez de estourar a
        # memória; T4 (contínuo) é barato e roda sempre.
        self.n_sequences = enumeration_size(inst.N, inst.M)
        self.n_configs = int((inst.R * inst.M) ** inst.N)
        self.grid_tractable = self.n_configs <= self.max_configs
        self.tractable = (self.n_sequences <= self.max_sequences) and self.grid_tractable
        self._classical: Optional[Dict[str, object]] = None

    # -- lado clássico, independente do Hamiltoniano ------------------------
    @property
    def classical(self) -> Dict[str, object]:
        if self._classical is None:
            self._classical = self._enumerate_classical()
        return self._classical

    @property
    def optimal_set(self) -> Set[Canon]:
        return self.classical["optimal_set"]

    def _enumerate_classical(self) -> Dict[str, object]:
        inst, tol = self.inst, self.tol
        N, M, D = inst.N, inst.M, inst.D
        demands = np.asarray(inst.demands, float) if inst.demands is not None else None
        Q = np.asarray(inst.Q, float) if inst.Q is not None else None

        best_cost, best, n_feasible = np.inf, set(), 0
        for perm in permutations(range(1, N + 1)):
            for cuts in combinations(range(1, N), M - 1):
                idx = [0, *cuts, N]
                routes = [list(perm[idx[k]:idx[k + 1]]) for k in range(M)]
                if self.require_all_vehicles and any(not r for r in routes):
                    continue
                if self.use_capacity and demands is not None and Q is not None:
                    loads = [sum(demands[c - 1] for c in r) for r in routes]
                    caps = [float(Q[v]) if Q.ndim else float(Q) for v in range(M)]
                    if any(l > c + tol for l, c in zip(loads, caps)):
                        continue
                n_feasible += 1
                cost = H.routes_cost(routes, D)
                if cost < best_cost - tol:
                    best_cost, best = cost, {canonical(routes, self.symmetric)}
                elif abs(cost - best_cost) <= tol:
                    best.add(canonical(routes, self.symmetric))

        return {"best_cost": float(best_cost), "optimal_set": best,
                "num_optimal_canonical": len(best),
                "num_feasible_sequences": n_feasible}

    # -- lado do Hamiltoniano ----------------------------------------------
    def hamiltonian_argmin(self, params: H.HamiltonianParams) -> Dict[str, object]:
        """Configurações de nó que atingem o mínimo de H, decodificadas."""
        th, ph = H._grid_angles(self.inst)
        E = H.hamiltonian_energy(th, ph, self.inst, params)["total"]
        emin = float(E.min())
        hits = np.flatnonzero(E <= emin + self.tol)

        argmin_set: Set[Canon] = set()
        for k in hits:
            dec = H.decode(th[k], ph[k], self.inst)
            argmin_set.add(canonical(dec["routes"], self.symmetric))

        return {"min_energy": emin, "argmin_set": argmin_set,
                "degeneracy_configs": int(hits.size),
                "degeneracy_canonical": len(argmin_set),
                "grid_size": int(E.size)}

    # -- verificação completa ----------------------------------------------
    def verify(self, params: Optional[H.HamiltonianParams] = None,
               continuous_starts: int = 50, logger=None,
               max_examples: int = 3) -> Dict[str, object]:
        """Laudo serializável em JSON. Nunca levanta exceção por falha de teste."""
        params = params or H.HamiltonianParams()
        cont = H.continuous_minimum(self.inst, params, n_starts=continuous_starts)
        qtm = (self.hamiltonian_argmin(params) if self.grid_tractable else
               {"min_energy": float(cont["min_energy"]), "argmin_set": set(),
                "degeneracy_configs": None, "degeneracy_canonical": None,
                "grid_size": self.n_configs})

        t4 = (bool(cont["min_energy"] >= qtm["min_energy"] - self.tol)
              if self.grid_tractable else None)

        rep: Dict[str, object] = {
            "tractable": self.tractable,
            "grid_tractable": self.grid_tractable,
            "n_sequences": self.n_sequences,
            "n_configs": self.n_configs,
            "symmetric_D": self.symmetric,
            "require_all_vehicles": self.require_all_vehicles,
            "use_capacity": self.use_capacity,
            "h_min_energy": qtm["min_energy"],
            "continuous_min_energy": float(cont["min_energy"]),
            "num_argmin_canonical": qtm["degeneracy_canonical"],
            "num_argmin_configs": qtm["degeneracy_configs"],
            "grid_size": qtm["grid_size"],
            "T4_no_offnode_minimum": t4,
        }

        if self.tractable:
            cls = self.classical
            opt, arg = cls["optimal_set"], qtm["argmin_set"]
            inter, spurious, missing = arg & opt, arg - opt, opt - arg
            rep.update({
                "c_star": cls["best_cost"],
                "num_optimal_classical": cls["num_optimal_canonical"],
                "num_feasible_sequences": cls["num_feasible_sequences"],
                "T1_intersection": bool(inter),
                "T2_no_spurious": not spurious,
                "T3_energy_matches": bool(abs(qtm["min_energy"] - cls["best_cost"]) <= self.tol),
                "T5_all_optima_reachable": not missing,
                "witness": canon_to_str(sorted(inter)[0]) if inter else None,
                "spurious_examples": [canon_to_str(c) for c in sorted(spurious)[:max_examples]],
                "missing_examples": [canon_to_str(c) for c in sorted(missing)[:max_examples]],
            })
            rep["passed"] = bool(rep["T1_intersection"] and rep["T2_no_spurious"]
                                 and rep["T3_energy_matches"] and t4)
        else:
            for name in ("T1_intersection", "T2_no_spurious",
                         "T3_energy_matches", "T5_all_optima_reachable"):
                rep[name] = None
            rep.update({"c_star": None, "num_optimal_classical": None,
                        "witness": None, "spurious_examples": [], "missing_examples": []})
            rep["passed"] = None          # nada conclusivo foi testado
            why = []
            if self.n_sequences > self.max_sequences:
                why.append(f"enumeração clássica com {self.n_sequences:,} sequências "
                           f"excede max_sequences={self.max_sequences:,}")
            if not self.grid_tractable:
                why.append(f"grade de nós com {self.n_configs:,} configurações "
                           f"excede max_configs={self.max_configs:,}")
            rep["skipped_reason"] = ("; ".join(why) +
                                     ("; só T4 foi aplicado" if self.grid_tractable
                                      else "; nenhum teste pôde ser aplicado"))

        if logger is not None:
            self.log(rep, params, logger)
        return rep

    def _energy_reference(self, qtm) -> float:
        return self.classical["best_cost"]

    # -- auditoria de uma rota decodificada ---------------------------------
    def audit_routes(self, routes) -> Dict[str, object]:
        """A rota extraída pelo VQE está no conjunto ótimo clássico?

        Mais confiável que comparar custos: distingue solução ótima de empate
        numérico com uma rota diferente.
        """
        rr = normalize_routes(routes)
        if not rr:
            return {"route_canonical": None, "route_in_optimal_set": None,
                    "route_cost": None, "route_gap_to_optimum": None}
        can = canonical(rr, self.symmetric)
        cost = H.routes_cost(rr, self.inst.D)
        in_set = (can in self.optimal_set) if self.tractable else None
        gap = (cost - self.classical["best_cost"]) if self.tractable else None
        return {"route_canonical": canon_to_str(can), "route_in_optimal_set": in_set,
                "route_cost": float(cost),
                "route_gap_to_optimum": (float(gap) if gap is not None else None)}

    # -- apresentação -------------------------------------------------------
    def log(self, rep: Dict[str, object], params: H.HamiltonianParams, logger) -> None:
        inst = self.inst
        head = (f"Verificação da codificação: N={inst.N} M={inst.M} R={inst.R} "
                f"grade={rep['grid_size']} lambda_disc={params.lambda_disc}")
        flags = " ".join(f"{n.split('_')[0]}={_mark(rep.get(n))}" for n in TEST_NAMES)
        body = (f"{flags} | min H={rep['h_min_energy']:.9f} "
                f"C*={rep['c_star'] if rep['c_star'] is None else format(rep['c_star'], '.9f')} "
                f"contínuo={rep['continuous_min_energy']:.9f}")
        emit = logger.info if rep["passed"] is not False else logger.error
        emit(head)
        emit(body)
        for key, label in (("spurious_examples", "mínimos espúrios"),
                           ("missing_examples", "ótimas não alcançadas")):
            if rep.get(key):
                logger.warning(f"  {label}: {rep[key]}")
        if rep.get("skipped_reason"):
            logger.warning(f"  {rep['skipped_reason']}")

    def report_text(self, rep: Dict[str, object], params: H.HamiltonianParams) -> str:
        inst = self.inst
        lines = [f"N={inst.N} M={inst.M} R={inst.R} | grade={rep['grid_size']} "
                 f"| lambda_disc={params.lambda_disc} lambda_col={params.lambda_col}",
                 f"  C* (clássico)        = {rep['c_star']:.9f}" if rep["c_star"] is not None
                 else "  C* (clássico)        = (não enumerado)",
                 f"  min H (grade de nós) = {rep['h_min_energy']:.9f}",
                 f"  min H (contínuo)     = {rep['continuous_min_energy']:.9f}",
                 f"  ótimas clássicas: {rep['num_optimal_classical']}   "
                 f"argmin H: {rep['num_argmin_canonical']} "
                 f"({rep['num_argmin_configs']} configurações)"]
        for name in TEST_NAMES:
            lines.append(f"  [{_mark(rep.get(name))}] {name}")
        if rep.get("witness"):
            lines.append(f"  testemunha: {rep['witness']}")
        for key, label in (("spurious_examples", "espúrios"),
                           ("missing_examples", "faltando")):
            if rep.get(key):
                lines.append(f"  {label}: {rep[key]}")
        return "\n".join(lines)


def _mark(v: Optional[bool]) -> str:
    return "----" if v is None else ("PASSA" if v else "FALHA")


# ---------------------------------------------------------------------------
# Cache por instância
# ---------------------------------------------------------------------------
_CACHE: Dict[tuple, EncodingVerifier] = {}


def get_verifier(inst: H.VRPInstance, require_all_vehicles: bool = False,
                 use_capacity: bool = False, tol: float = 1e-6,
                 max_sequences: int = DEFAULT_MAX_SEQUENCES,
                 max_configs: int = DEFAULT_MAX_CONFIGS) -> EncodingVerifier:
    """Verificador para a instância, reaproveitado entre chamadas do processo.

    A chave inclui as regras de viabilidade, porque elas mudam o conjunto ótimo;
    não inclui os lambdas, que não mudam.
    """
    key = (H._instance_key(inst), bool(require_all_vehicles), bool(use_capacity),
           float(tol), int(max_sequences), int(max_configs))
    v = _CACHE.get(key)
    if v is None:
        v = EncodingVerifier(inst, require_all_vehicles, use_capacity, tol,
                             max_sequences, max_configs)
        _CACHE[key] = v
    return v


def clear_cache() -> None:
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Colunas para o summary.csv
# ---------------------------------------------------------------------------
def summary_columns(rep: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Subconjunto achatado do laudo, para uma linha do CSV."""
    if not rep:
        return {"enc_passed": None, "enc_T1": None, "enc_T2": None,
                "enc_T3": None, "enc_T4": None, "enc_T5": None,
                "enc_argmin_canonical": None, "enc_witness": None}
    return {"enc_passed": rep.get("passed"),
            "enc_T1": rep.get("T1_intersection"),
            "enc_T2": rep.get("T2_no_spurious"),
            "enc_T3": rep.get("T3_energy_matches"),
            "enc_T4": rep.get("T4_no_offnode_minimum"),
            "enc_T5": rep.get("T5_all_optima_reachable"),
            "enc_argmin_canonical": rep.get("num_argmin_canonical"),
            "enc_witness": rep.get("witness")}


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------
def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Verifica se o argmin do Hamiltoniano decodifica para uma rota ótima.")
    ap.add_argument("-C", "--cities", type=int, default=4)
    ap.add_argument("-V", "--vehicles", type=int, default=1)
    ap.add_argument("-R", "--positions", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--circular", action="store_true")
    ap.add_argument("--lambda-disc", type=float, default=5.0)
    ap.add_argument("--lambda-col", type=float, default=25.0)
    ap.add_argument("--lambda-gap", type=float, default=20.0)
    ap.add_argument("--lambda-cap", type=float, default=0.0)
    ap.add_argument("--lambda-vehicle", type=float, default=0.0)
    ap.add_argument("--require-all-vehicles", action="store_true")
    ap.add_argument("--starts", type=int, default=200)
    args = ap.parse_args()

    C = args.cities
    if args.circular:
        ang = np.linspace(0, 2 * np.pi, C, endpoint=False)
        co = np.vstack([[0.0, 0.0], np.c_[5 * np.cos(ang), 5 * np.sin(ang)]])
    else:
        rng = np.random.default_rng(args.seed)
        co = np.vstack([[0.0, 0.0], rng.uniform(-10, 10, (C, 2))])
    d = co[:, None, :] - co[None, :, :]
    D = np.sqrt((d ** 2).sum(-1))

    inst = H.instance_from_legacy(C, args.vehicles, D,
                                  R=(args.positions if args.positions else "auto"),
                                  require_all_vehicles=args.require_all_vehicles)
    params = H.HamiltonianParams(lambda_disc=args.lambda_disc, lambda_col=args.lambda_col,
                                 lambda_gap=args.lambda_gap, lambda_cap=args.lambda_cap,
                                 lambda_vehicle=args.lambda_vehicle)
    v = get_verifier(inst, args.require_all_vehicles, args.lambda_cap > 0)
    rep = v.verify(params, continuous_starts=args.starts)
    print(v.report_text(rep, params))
    return 0 if rep["passed"] is not False else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
