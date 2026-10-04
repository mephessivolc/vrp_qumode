import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
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
    c_star: Optional[float] = None,
    uniform_energy: Optional[float] = None,
    cutoff_floor: Optional[float] = None,
    show_grad_norm: bool = True,
    normalize: bool = False,
):
    """Curva de convergência do VQE: uma energia por ITERAÇÃO.

    Usa `metrics.iterate_history`, que tem exatamente maxiter + 1 pontos. O
    campo `metrics.cost_history` NÃO deve ser usado aqui: ele inclui as 2P
    sondagens theta +/- eps da diferença finita, cujas energias são quase
    idênticas dentro de uma mesma iteração e produzem o artefato de "escada",
    além de multiplicar o eixo x por 2P + 1.

    Parâmetros opcionais
    --------------------
    c_star, uniform_energy, cutoff_floor
        linhas de referência: ótimo clássico, energia da distribuição uniforme
        e piso imposto pelo corte de Fock.
    normalize
        plota a energia normalizada (E - C*) / (E_uniforme - C*), comparável
        entre instâncias: 1 é o acaso e 0 é o ótimo. Exige c_star e
        uniform_energy.
    show_grad_norm
        adiciona um painel inferior com a norma do gradiente por iteração,
        quando `metrics.grad_norm_history` estiver disponível.
    """
    fig_path = Path(fig_path_name)
    if not fig_path.suffix:
        fig_path = fig_path.with_suffix(".png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    energy = list(getattr(metrics, "iterate_history", None) or [])
    x_label = "Iteração"
    if not energy:
        # compatibilidade com execuções antigas: sem iterados, resta o histórico
        # bruto, e nesse caso o eixo x é de AVALIAÇÕES, não de iterações.
        energy = list(getattr(metrics, "cost_history", []) or [])
        x_label = "Avaliação do circuito"
    if not energy:
        return False

    grad = list(getattr(metrics, "grad_norm_history", None) or [])
    use_grad = bool(show_grad_norm and grad and x_label == "Iteração")

    y = np.asarray(energy, dtype=float)
    y_label = "Energia $\\langle H \\rangle$"
    refs = {}
    if normalize:
        if c_star is None or uniform_energy is None:
            raise ValueError("normalize=True exige c_star e uniform_energy.")
        scale = float(uniform_energy) - float(c_star)
        if abs(scale) < 1e-12:
            raise ValueError("uniform_energy coincide com c_star: instância sem contraste.")
        y = (y - float(c_star)) / scale
        y_label = "Energia normalizada $(E - C^*)/(E_{unif} - C^*)$"
        refs["Ótimo $C^*$"] = (0.0, "tab:green")
        refs["Uniforme"] = (1.0, "tab:gray")
        if cutoff_floor is not None:
            refs["Piso do cutoff"] = ((float(cutoff_floor) - float(c_star)) / scale, "tab:red")
    else:
        if c_star is not None:
            refs["Ótimo $C^*$"] = (float(c_star), "tab:green")
        if uniform_energy is not None:
            refs["Uniforme"] = (float(uniform_energy), "tab:gray")
        if cutoff_floor is not None:
            refs["Piso do cutoff"] = (float(cutoff_floor), "tab:red")

    x = np.arange(y.size)

    if use_grad:
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(8, 5.5), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax2 = None

    ax.plot(x, y, color="tab:blue", lw=1.8, label="VQE")
    k = int(np.argmin(y))
    ax.plot([k], [y[k]], "o", ms=5, color="tab:blue",
            label=f"melhor iterado (it. {k})")
    for name, (value, color) in refs.items():
        ax.axhline(value, ls="--", lw=1.0, color=color, label=name)

    ax.set_ylabel(y_label)
    ax.set_title(f"Convergência do VQE ({method})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    if not normalize and np.all(y > 0) and y.max() / max(y.min(), 1e-12) > 50:
        ax.set_yscale("log")

    if ax2 is not None:
        ax2.plot(np.arange(1, len(grad) + 1), grad, color="tab:orange", lw=1.2)
        ax2.set_ylabel(r"$\|\nabla E\|$")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel(x_label)
    else:
        ax.set_xlabel(x_label)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return True
