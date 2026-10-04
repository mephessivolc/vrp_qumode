"""
main.py — grade de experimentos.

Penalidades
-----------
`h_params` é o que controla lambda_col, lambda_gap, lambda_cap, lambda_disc,
lambda_vehicle e lambda_dist. Se não for passado, run.py monta um padrão
(col=25, gap=20, cap=0 ou 18, disc=5).

Mudar um lambda NÃO move o mínimo do Hamiltoniano, desde que as penalidades
fiquem acima do limiar de separação — muda a PAISAGEM (gradientes, bacias,
faixa dinâmica relativa), não o argmin. Quem garante isso é a verificação que
roda automaticamente antes do VQE: se a codificação deixar de ser válida, a
execução aborta com status ENCODING_INVALID sem gastar simulação.

Duas armadilhas:
  - lambda_vehicle > 0 exige require_all_vehicles coerente (run.py valida);
  - lambda_cap > 0 exige demands e Q na instância, inclusive com V = 1.
"""
from dataclasses import replace
from itertools import product

from qumodes.hamiltonian import HamiltonianParams
from run import run_experiment

# ---------------------------------------------------------------------------
# Penalidades base. Alterar aqui muda TODOS os experimentos.
# ---------------------------------------------------------------------------
BASE = HamiltonianParams(
    lambda_dist=1.0,      # peso do custo da rota (mantenha em 1: é a referência)
    lambda_col=25.0,      # colisão: duas cidades no mesmo nó de posição
    lambda_gap=20.0,      # lacuna: nó de posição vazio antes de um ocupado
    lambda_cap=0.0,       # capacidade: > 0 exige demands e Q
    lambda_vehicle=0.0,   # > 0: todos os M veículos obrigatórios
    lambda_disc=5.0,      # estabilizador de Zak: evita mínimo fora dos nós
)


def params_for(vehicle: int, lambda_disc: float) -> HamiltonianParams:
    """Penalidades de um ponto da grade, derivadas de BASE."""
    return replace(
        BASE,
        lambda_disc=lambda_disc,
        lambda_cap=18.0 if vehicle > 1 else 0.0,
        lambda_vehicle=50.0 if vehicle > 1 else 0.0,
    )


if __name__ == "__main__":
    cities = [3, 4]
    vehicles = [1, 2]
    num_layers = [1, 2]
    is_warm_start = [True]
    lambdas_disc = [5.0]          # varredura: [0.5, 1.0, 2.0, 5.0]

    max_iter = 100
    lr = 0.01
    lr_final = None               # com decaimento: lr=0.05, lr_final=0.005
    cutoff = 8
    sub_folder = "experiment_Madani_Hamiltonian"
    exp_name = "exp"

    sim = list(product(cities, vehicles, num_layers, is_warm_start, lambdas_disc))
    print(f"{len(sim)} experimentos\n")

    for num_exp, (city, vehicle, layer, ws, lam_disc) in enumerate(sim, start=1):
        sim_name = f"{exp_name}_{str(num_exp).zfill(4)}"
        h = params_for(vehicle, lam_disc)

        print(f"[{num_exp}/{len(sim)}] {sim_name}: C={city} V={vehicle} L={layer} "
              f"| col={h.lambda_col} gap={h.lambda_gap} cap={h.lambda_cap} "
              f"disc={h.lambda_disc} veh={h.lambda_vehicle}")

        payload = run_experiment(
            C=city,
            V=vehicle,
            max_iter=max_iter,
            num_layer=layer,
            lr=lr,
            lr_final=lr_final,
            cutoff=cutoff,
            is_warm_start=ws,
            h_params=h,                       # <-- as penalidades entram por aqui
            sub_folder=sub_folder,
            exp_name=sim_name,
            sweep="B",                        # rótulo da varredura no summary.csv
        )

        if payload.get("status") == "ENCODING_INVALID":
            print("  -> codificação inválida com esses lambdas; experimento abortado\n")
            continue
        print("\n")