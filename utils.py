import json
from pathlib import Path
from typing import Any, Dict, Union
import matplotlib.pyplot as plt
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Encoder customizado para converter objetos do NumPy em tipos nativos do Python."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_experiment_json(
    data: Dict[str, Any], json_path: Union[str, Path]
) -> None:
    """Salva os resultados da simulação em um arquivo JSON formatado."""
    target_path = Path(json_path)
    if not target_path.suffix:
        target_path = target_path.with_suffix(".json")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)


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
    """Gera e salva a curva de convergência do VQE no caminho especificado."""
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