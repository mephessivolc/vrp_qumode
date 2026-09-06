from pathlib import Path
from typing import Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np


def format_timespan(seconds: float) -> str:
    """Converte um intervalo de tempo em segundos para representação legível."""
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


def plot_convergence(
    metrics,
    method: str,
    fig_path_name: Union[str, Path] = "test_convergence",
):
    """Gera e salva a curva de convergência do VQE no caminho especificado pelo PathManager."""
    fig_path = Path(fig_path_name)
    if not fig_path.suffix:
        fig_path = fig_path.with_suffix(".png")

    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(metrics.cost_history, label="Energia Total $\\langle H \\rangle$")
    plt.xlabel("Iterações")
    plt.ylabel("Energia")
    plt.title(f"Curva de Convergência VQE ({method}) - CVRP")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path)
    plt.close()
    print(f"[Info] Gráfico de convergência salvo em: {fig_path}")