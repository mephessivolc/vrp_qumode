# utils.py
import numpy as np
from typing import Union, List, Dict


def format_timespan(seconds: float) -> str:
    """Converte um intervalo de tempo em segundos para representação humana."""
    if seconds < 0:
        return "0.00 s"
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"

    MINUTE = 60
    HOUR = 3600
    DAY = 86400

    secs = float(seconds)

    days, secs = divmod(secs, DAY)
    hours, secs = divmod(secs, HOUR)
    minutes, secs = divmod(secs, MINUTE)

    parts = []

    if days > 0:
        parts.append(f"{int(days)}d")
    if hours > 0:
        parts.append(f"{int(hours)}h")
    if minutes > 0:
        parts.append(f"{int(minutes)}m")
    if secs > 0 or not parts:
        parts.append(f"{secs:.1f}s")

    if len(parts) > 2:
        return f"{parts[0]} e {parts[1]}"
    return " ".join(parts)


def calculate_cost_from_matrix(
    route: Union[List[int], Dict[int, List[int]]], 
    dist_matrix: np.ndarray
) -> float:
    """
    Calcula o custo total de uma rota (TSP ou VRP) a partir da matriz de distâncias.
    Útil para validações cruzadas nas métricas do experimento.
    """
    total_cost = 0.0
    
    # Caso VRP (Dicionário de rotas por veículo)
    if isinstance(route, dict):
        for _, sub_route in route.items():
            for i in range(len(sub_route) - 1):
                u, v = sub_route[i], sub_route[i + 1]
                total_cost += float(dist_matrix[u, v])
        return round(total_cost, 2)

    # Caso TSP (Lista simples de cidades)
    full_route = list(route)
    if full_route[0] != full_route[-1]:
        full_route.append(full_route[0])  # Fecha o ciclo se necessário

    for i in range(len(full_route) - 1):
        u, v = full_route[i], full_route[i + 1]
        total_cost += float(dist_matrix[u, v])

    return round(total_cost, 2)


def print_experiment_summary(
    problem_type: str,
    n_cities: int,
    exact_cost: float,
    exact_time: float,
    quantum_cost: float,
    quantum_time: float,
    evals: int
):
    """Exibe um resumo elegante do experimento no terminal."""
    gap = ((quantum_cost - exact_cost) / exact_cost) * 100
    print("\n" + "="*55)
    print(f"      RESUMO DO EXPERIMENTO [{problem_type.upper()} - N={n_cities}]")
    print("="*55)
    print(f" • Custo Exato (Força Bruta): {exact_cost:.2f} ({format_timespan(exact_time)})")
    print(f" • Custo Quântico (CV):  {quantum_cost:.2f} ({format_timespan(quantum_time)})")
    print(f" • Desvio Relativo (GAP):     {gap:+.2f}%")
    print(f" • Avaliações de Função (nfev): {evals}")
    print("="*55 + "\n")