# main.py (na raiz do projeto)
from tsp.main import run as run_tsp
from vrp.main import run as run_vrp

def main():
    print("=== INICIANDO BATCH DE EXPERIMENTOS CV-VQE ===")

    for city in [4, 5, 6]:
        for iter in range(25, 201, 25):
            for layer in [1,2,3,4,5]:

                # 1. Executa TSP com 4 cidades
                res_tsp = run_tsp(
                    n_cities=city,
                    layers=layer,
                    maxiter=iter,
                    optimizer_method="COBYLA",
                    seed=42
                )

                list_resp_vrp = []
                for vehicle in [2,3,4,5,6]:
                    res_vrp = run_vrp(
                        n_cities=city,
                        num_vehicles=vehicle,
                        layers=layer,
                        maxiter=iter,
                        optimizer_method="COBYLA",
                        seed=42
                        )

                    list_resp_vrp.append(f"{vehicle}: {res_vrp['quantum_cost']:.4f}")

                # Exemplo de acesso direto aos resultados sem reescrever nada
                print(f"\n[Resumo da Simulação]")
                print(f"Custo TSP (VQE): {res_tsp['quantum_cost']:.4f}")

                print(f"Custo VRP (VQE):")
                for r in list_resp_vrp:
                    print(r)

if __name__ == "__main__":
    main()
