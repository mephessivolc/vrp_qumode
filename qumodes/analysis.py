"""
analysis.py — Métricas e diagnósticos dos experimentos VQE em qumodes.

Usado por run.py. Não altera solver.py, ansatz.py nem hamiltonian.py: apenas
lê o que eles já expõem (histórico de custos, θ ótimo, estado final).

Principais blocos:
    parameter_labels          rótulos dos parâmetros na ordem do ansatz.py
    reconstruct_adam_trace    energia e gradiente por ITERAÇÃO a partir do cost_history
    configuration_space       todas as configurações discretas: viáveis, custos, ótimas
    state_metrics             P(ótimo), P(viável), top-k, fótons, norma do estado
    baseline_metrics          vácuo e sorteio uniforme (linhas de base)
    cutoff_energy_floor       menor ⟨H⟩ possível no espaço de Fock truncado
    to_jsonable, environment_info, append_summary_csv
"""
from __future__ import annotations

import csv
import dataclasses
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import hamiltonian as H

SCHEMA_VERSION = "2.0"


# ---------------------------------------------------------------------------
# Serialização e ambiente
# ---------------------------------------------------------------------------
def to_jsonable(obj):
    """Converte numpy, dataclasses, tuplas e não finitos (-> None) para JSON."""
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {"real": to_jsonable(obj.real.tolist()), "imag": to_jsonable(obj.imag.tolist())}
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    return obj


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def environment_info() -> Dict[str, object]:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "numpy": np.__version__}
    for mod in ("scipy", "strawberryfields"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info


# ---------------------------------------------------------------------------
# Parâmetros do ansatz e traço do ADAM
# ---------------------------------------------------------------------------
def parameter_labels(num_qumodes: int, num_layers: int) -> List[str]:
    """Rótulos na mesma ordem de ContinuousVariableAnsatz.build_program."""
    labels: List[str] = []
    for layer in range(num_layers):
        for i in range(num_qumodes):
            labels += [f"L{layer}.S_r.q{i}", f"L{layer}.S_phi.q{i}", f"L{layer}.R.q{i}"]
        for i in range(num_qumodes - 1):
            labels += [f"L{layer}.BS_theta.q{i}q{i + 1}", f"L{layer}.BS_phi.q{i}q{i + 1}"]
        for i in range(num_qumodes):
            labels += [f"L{layer}.D_r.q{i}", f"L{layer}.D_phi.q{i}", f"L{layer}.K.q{i}"]
    return labels


def _group(label: str) -> str:
    parts = label.split(".")
    return parts[1] if len(parts) >= 3 else "param"


def reconstruct_adam_trace(cost_history: Sequence[float], n_params: int,
                           fd_eps: float = 1e-4,
                           labels: Optional[Sequence[str]] = None) -> Optional[Dict[str, object]]:
    """Separa o cost_history do ADAM (solver.py) em iterações.

    Estrutura do histórico: [E(θ0)] + por iteração: 2·P avaliações (θ ± eps)
    e 1 avaliação no novo θ. Aceita histórico parcial (execução interrompida).
    """
    h = np.asarray(cost_history, dtype=np.float64)
    step = 2 * n_params + 1
    if h.size < 1 or n_params < 1:
        return None
    n_iter = (h.size - 1) // step
    energy = [float(h[0])] + [float(h[step * t]) for t in range(1, n_iter + 1)]
    grad_norm, by_group = [], {}
    groups = [_group(l) for l in labels] if labels is not None and len(labels) == n_params else None
    for t in range(1, n_iter + 1):
        base = 1 + step * (t - 1)
        g = (h[base:base + 2 * n_params:2] - h[base + 1:base + 2 * n_params:2]) / (2.0 * fd_eps)
        grad_norm.append(float(np.linalg.norm(g)))
        if groups is not None:
            acc: Dict[str, float] = {}
            for name, val in zip(groups, g):
                acc[name] = acc.get(name, 0.0) + float(val) ** 2
            for name, val in acc.items():
                by_group.setdefault(name, []).append(math.sqrt(val))
    return {"iterations_completed": n_iter,
            "evaluations_per_iteration": step,
            "incomplete_tail_evaluations": int(h.size - 1 - n_iter * step),
            "energy_per_iteration": energy,
            "grad_norm_per_iteration": grad_norm,
            "grad_norm_by_group": by_group}


def convergence_summary(energy_per_iteration: Sequence[float],
                        grad_norm: Optional[Sequence[float]] = None,
                        window: int = 10, tol: float = 1e-3) -> Dict[str, object]:
    """Diagnóstico de convergência sobre a energia por iteração."""
    e = np.asarray(energy_per_iteration, dtype=np.float64)
    if e.size < 2:
        return {"iterations": int(e.size) - 1}
    d = np.diff(e)
    w = min(window, d.size)
    last = float(d[-w:].mean())
    out = {
        "iterations": int(d.size),
        "total_decrease": float(e[0] - e[-1]),
        "relative_decrease": float((e[0] - e[-1]) / abs(e[0])) if e[0] != 0 else None,
        "mean_delta_first_window": float(d[:w].mean()),
        "mean_delta_last_window": last,
        "increasing_steps": int((d > 0).sum()),
        "still_decreasing": bool(last < -tol),
        "window": w, "tol": tol,
    }
    if grad_norm:
        out["grad_norm_first"] = float(grad_norm[0])
        out["grad_norm_last"] = float(grad_norm[-1])
        out["grad_norm_min"] = float(np.min(grad_norm))
    return out


# ---------------------------------------------------------------------------
# Espaço de configurações discretas
# ---------------------------------------------------------------------------
def configuration_space(inst: "H.VRPInstance", params: "H.HamiltonianParams",
                        require_all_vehicles: bool = False,
                        max_configs: int = 300_000) -> Dict[str, np.ndarray]:
    """Energia, viabilidade e custo de todas as (M·R)^N configurações.

    A ordem coincide com ZakEvaluator.slot_distribution (s = v·R + r,
    cidade 1 no dígito mais significativo).
    """
    size = (inst.M * inst.R) ** inst.N
    if size > max_configs:
        raise MemoryError(f"(M·R)^N = {size:,} configurações excede max_configs.")
    th, ph = H._grid_angles(inst)
    energy = H.hamiltonian_energy(th, ph, inst, params)["total"]
    use_cap = params.lambda_cap > 0 and inst.demands is not None and inst.Q is not None
    feasible = np.zeros(size, dtype=bool)
    cost = np.full(size, np.nan)
    for k in range(size):
        dec = H.decode(th[k], ph[k], inst)
        if H.is_feasible(dec, inst, require_all_vehicles, use_cap):
            feasible[k] = True
            cost[k] = H.routes_cost(dec["routes"], inst.D)
    best = float(np.nanmin(cost)) if feasible.any() else math.inf
    optimal = feasible & np.isclose(cost, best)
    return {"theta": th, "phi": ph, "energy": energy, "feasible": feasible,
            "cost": cost, "optimal": optimal, "best_cost": best}


def configuration_summary(space: Dict[str, np.ndarray]) -> Dict[str, object]:
    feas = space["feasible"]
    levels = np.unique(np.round(space["cost"][feas], 6)) if feas.any() else np.array([])
    return {"num_configurations": int(feas.size),
            "num_feasible": int(feas.sum()),
            "num_optimal": int(space["optimal"].sum()),
            "best_feasible_cost": space["best_cost"],
            "feasible_cost_levels": levels.tolist(),
            "uniform_p_optimal": float(space["optimal"].mean()),
            "uniform_p_feasible": float(feas.mean())}


# ---------------------------------------------------------------------------
# Métricas de estados
# ---------------------------------------------------------------------------
def _p_find(p: float, shots: Iterable[int]) -> Dict[str, float]:
    return {str(s): float(1.0 - (1.0 - p) ** s) for s in shots}


def _routes_of(space, k, inst) -> List[List[int]]:
    return H.decode(space["theta"][k], space["phi"][k], inst)["routes"]


def state_metrics(ket: np.ndarray, inst: "H.VRPInstance", params: "H.HamiltonianParams",
                  space: Dict[str, np.ndarray], shots: Sequence[int] = (10, 100, 1000),
                  top_k: int = 5, G: int = 8) -> Dict[str, object]:
    """Energia, distribuição sobre configurações e diagnósticos do ket."""
    ket = np.asarray(ket)
    cutoff = ket.shape[0]
    norm = float(np.sum(np.abs(ket) ** 2))
    ev = H._get_evaluator(inst, cutoff, G, None)
    P = ev.distribution(ket)
    energy = ev.energy(P, params, ket=ket)
    probs = np.clip(ev.slot_distribution(P).reshape(-1), 0, None)
    probs = probs / probs.sum()

    feas, opt = space["feasible"], space["optimal"]
    p_opt, p_feas = float(probs[opt].sum()), float(probs[feas].sum())
    order = np.argsort(probs)[::-1][:top_k]
    top = [{"probability": float(probs[k]), "energy": float(space["energy"][k]),
            "feasible": bool(feas[k]),
            "cost": None if not feas[k] else float(space["cost"][k]),
            "routes": _routes_of(space, k, inst)} for k in order]
    exp_cost = float(np.nansum(probs[feas] * space["cost"][feas]) / p_feas) if p_feas > 0 else None

    photons = []
    for mode in range(ket.ndim):
        marg = np.sum(np.abs(np.moveaxis(ket, mode, 0)) ** 2, axis=tuple(range(1, ket.ndim)))
        photons.append(float(np.sum(np.arange(cutoff) * marg) / max(norm, 1e-300)))
        # população no último nível de Fock: indicador de truncagem
    last_level = [float(np.sum(np.abs(np.take(ket, cutoff - 1, axis=m)) ** 2) / max(norm, 1e-300))
                  for m in range(ket.ndim)]

    return {"energy_components": energy,
            "p_optimal": p_opt, "p_feasible": p_feas,
            "p_optimal_over_uniform": p_opt / float(opt.mean()) if opt.any() else None,
            "expected_cost_given_feasible": exp_cost,
            "p_find_optimum_in_shots": _p_find(p_opt, shots),
            "top_configurations": top,
            "state_norm": norm,
            "mean_photons_per_mode": photons,
            "last_fock_level_population": last_level}


def baseline_metrics(inst: "H.VRPInstance", params: "H.HamiltonianParams",
                     space: Dict[str, np.ndarray], cutoff: int,
                     shots: Sequence[int] = (10, 100, 1000),
                     uniform_samples: int = 200_000, seed: int = 0) -> Dict[str, object]:
    """Linhas de base: vácuo (sem circuito) e sorteio uniforme."""
    vac = np.zeros((cutoff,) * inst.N)
    vac[(0,) * inst.N] = 1.0
    vacuum = state_metrics(vac, inst, params, space, shots, top_k=3)

    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, (uniform_samples, inst.N))
    ph = rng.uniform(0, 2 * np.pi, (uniform_samples, inst.N))
    parts = H.hamiltonian_energy(th, ph, inst, params)
    p_opt = float(space["optimal"].mean())
    uniform = {"energy_components": {k: float(np.mean(v)) for k, v in parts.items()},
               "energy_std_error": float(np.std(parts["total"]) / math.sqrt(uniform_samples)),
               "p_optimal": p_opt, "p_feasible": float(space["feasible"].mean()),
               "p_find_optimum_in_shots": _p_find(p_opt, shots)}
    return {"vacuum": vacuum, "uniform": uniform}


# ---------------------------------------------------------------------------
# Piso de energia do cutoff (menor autovalor projetado)
# ---------------------------------------------------------------------------
def cutoff_energy_floor(inst: "H.VRPInstance", params: "H.HamiltonianParams", cutoff: int,
                        G: int = 8, tol: float = 1e-4, maxiter: int = 300,
                        max_dim: int = 100_000) -> Dict[str, object]:
    """Menor ⟨H⟩ entre TODOS os estados com cutoff dado (Lanczos).

    Nenhum circuito simulado com esse cutoff pode reportar energia menor.
    Custo: segundos para C=3; cerca de 1-2 min e ~1 GB para C=4.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh
    import scipy.fft as sfft

    dim = cutoff ** inst.N
    if dim > max_dim:
        raise MemoryError(f"cutoff^N = {dim:,} excede max_dim.")
    t0 = time.time()
    ev = H._get_evaluator(inst, cutoff, G, None)
    unit = ev._unit_components()
    use_cap = params.lambda_cap > 0 and inst.demands is not None and inst.Q is not None
    w = {"dist": params.lambda_dist, "col": params.lambda_col, "gap": params.lambda_gap,
         "vehicle": params.lambda_vehicle, "capacity": params.lambda_cap if use_cap else 0.0,
         "disc": params.lambda_disc}
    F = np.zeros(int(np.prod(ev.shape)), dtype=np.float32)
    for c, lam in w.items():
        if lam:
            F += np.float32(lam) * unit[c]
    F = F.reshape(ev.shape)
    Hm = ev.hermite.astype(np.float32)
    N, ax, count = inst.N, tuple(range(0, 2 * inst.N, 2)), [0]

    def Z(v):
        t = v.reshape((cutoff,) * N).astype(np.complex64)
        for _ in range(N):
            t = np.tensordot(t, Hm, axes=([0], [0]))
        return sfft.fftn(t.reshape(ev.shape), axes=ax, overwrite_x=True, workers=-1)

    def Zadj(t):
        t = sfft.ifftn(t, axes=ax, overwrite_x=True, workers=-1).reshape((ev.points_per_mode,) * N)
        for _ in range(N):
            t = np.tensordot(t, Hm.T, axes=([0], [0]))
        return (t.reshape(-1) * ev.Nc ** N).astype(np.complex128)

    rng = np.random.default_rng(0)
    probe = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    probe /= np.linalg.norm(probe)
    metric = float(np.sum(np.abs(Z(probe)) ** 2))   # Z†Z = c·I na grade

    def matvec(v):
        count[0] += 1
        t = Z(v)
        t *= F
        return Zadj(t) / metric

    op = LinearOperator((dim, dim), matvec=matvec, dtype=np.complex128)
    vals, vecs = eigsh(op, k=1, which="SA", tol=tol, maxiter=maxiter, ncv=min(12, dim - 1))
    return {"min_energy": float(vals[0]), "matvecs": count[0],
            "seconds": time.time() - t0, "cutoff": cutoff, "G": ev.G,
            "ground_ket": vecs[:, 0].reshape((cutoff,) * N)}


# ---------------------------------------------------------------------------
# Resumo tabular
# ---------------------------------------------------------------------------
def append_summary_csv(csv_path: str, row: Dict[str, object]) -> None:
    """Acrescenta uma linha por experimento (cria o cabeçalho se necessário)."""
    row = {k: to_jsonable(v) for k, v in row.items()}
    new = not os.path.exists(csv_path)
    if not new:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), [])
        fields = header + [k for k in row if k not in header]
        if fields != header:            # novas colunas: reescreve com o cabeçalho ampliado
            with open(csv_path, newline="", encoding="utf-8") as fh:
                old_rows = list(csv.DictReader(fh))
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                wr = csv.DictWriter(fh, fieldnames=fields)
                wr.writeheader()
                wr.writerows(old_rows)
    else:
        fields = list(row)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        if new:
            wr.writeheader()
        wr.writerow(row)