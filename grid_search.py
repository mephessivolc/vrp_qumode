
# grid_search.py
import time
import json
from typing import Dict, List, Any, Optional, Callable
import itertools
import numpy as np

# Importações dos módulos do ecossistema do projeto
from graphs import GraphBuilder, plot_training_diagnostics
from logger import ExperimentLogger
from path import get_file_path, get_results_path, get_images_path 


class GridSearchRunner:
    """
    Orquestrador de experimentos em lote para benchmark do solver CV-VQE.
    Permite varredura sistemática em matrizes de parâmetros (Decodificadores, Schedulers, Otimizadores, Topologias).
    Utiliza as funções auxiliares do `path.py` para gerenciamento automatizado de diretórios.
    """
    def __init__(
        self,
        solver_class: Any,
        exact_solver_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        logger: Optional[ExperimentLogger] = None,
        variable_type: str = "qumodes",
        problem_type: str = "vrp",
        sub_folder: Optional[str] = "grid_search"
    ):
        """
        Args:
            solver_class: Classe do Solver VQE que será instanciada a cada execução.
            exact_solver_func: Função opcional que calcula o custo exato global (Ground Truth).
            logger: Instância do ExperimentLogger.
            variable_type: Tipo de variável ('qumodes', 'qubits', etc.) para organização de pastas .
            problem_type: Tipo do problema ('vrp', 'tsp', etc.) .
            sub_folder: Subpasta de destino do experimento dentro de data/figuras .
        """
        self.solver_class = solver_class
        self.exact_solver_func = exact_solver_func
        self.logger = logger if logger is not None else ExperimentLogger()
        self.variable_type = variable_type
        self.problem_type = problem_type
        self.sub_folder = sub_folder

        # Inicializa e garante a criação das pastas de figuras e dados via path.py 
        self.results_dir = get_results_path(
            variable_type=self.variable_type,
            problem_type=self.problem_type,
            sub_folder=self.sub_folder
        ) 
        self.images_dir = get_images_path(
            variable_type=self.variable_type,
            problem_type=self.problem_type,
            sub_folder=self.sub_folder
        ) 

    def run_grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        n_runs_per_config: int = 1,
        save_plots: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executa a varredura cartesiana de parâmetros fornecidos na malha.
        
        Args:
            param_grid: Dicionário contendo as listas de parâmetros a testar.
            n_runs_per_config: Número de repetições estocásticas por configuração.
            save_plots: Se True, gera e salva os painéis de diagnósticos individuais.
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        total_experiments = len(combinations) * n_runs_per_config
        print(f"=== Iniciando Grid Search: {len(combinations)} combinações únicas ({total_experiments} execuções) ===")

        results = []
        exp_id = 0

        for comb in combinations:
            config = dict(zip(keys, comb))

            for run_idx in range(n_runs_per_config):
                exp_id += 1
                base_seed = config.get("seed", 42)
                run_seed = base_seed + run_idx
                
                print(f"\n[{exp_id}/{total_experiments}] Config: {config} | Run Seed: {run_seed}")

                # 1. Construção do Problema (Grafo/Topologia)
                builder = GraphBuilder(
                    n=config.get("n_nodes", 4),
                    seed=run_seed,
                    graph_type=config.get("graph_type", "euclidean"),
                    num_vehicles=config.get("num_vehicles", 2),
                    vehicle_capacity=config.get("vehicle_capacity", 10.0),
                    logger=self.logger,
                    variable_type_path=self.variable_type,
                    sub_folder=self.sub_folder
                )

                # 2. Resolução Exata para Benchmark (se fornecida)
                exact_cost = None
                if self.exact_solver_func is not None:
                    try:
                        exact_cost = float(self.exact_solver_func(builder.matrix, builder.demands))
                    except Exception as e:
                        print(f"  [Aviso] Falha ao calcular solução exata: {e}")

                # 3. Execução do Solver Quantum CV-VQE
                start_time = time.time()
                
                solver_kwargs = {
                    "graph_builder": builder,
                    "decoder_type": config.get("decoder_type", "softmax"),
                    "scheduler_type": config.get("scheduler_type", "dynamic"),
                    "optimizer_name": config.get("optimizer", "adam"),
                    "learning_rate": config.get("lr", 0.01),
                    "max_iterations": config.get("max_iterations", 100)
                }
                
                # Adiciona parâmetros extras presentes no config
                for k, v in config.items():
                    if k not in solver_kwargs and k not in ["n_nodes", "graph_type", "num_vehicles", "vehicle_capacity", "seed"]:
                        solver_kwargs[k] = v

                solver_instance = self.solver_class(**solver_kwargs)
                vqe_res = solver_instance.solve()
                
                elapsed_time = time.time() - start_time

                # 4. Processamento de Métricas de Desempenho
                best_cost = vqe_res.get("best_cost", float("inf"))
                is_feasible = vqe_res.get("is_feasible", False)
                
                approx_ratio = None
                if exact_cost is not None and exact_cost > 0 and best_cost != float("inf"):
                    approx_ratio = round(best_cost / exact_cost, 4)

                res_entry = {
                    "exp_id": exp_id,
                    "run_idx": run_idx,
                    "seed": run_seed,
                    **config,
                    "exact_cost": exact_cost,
                    "best_cost": float(best_cost) if best_cost != float("inf") else None,
                    "approx_ratio": approx_ratio,
                    "is_feasible": is_feasible,
                    "execution_time_sec": round(elapsed_time, 3),
                    "best_iteration": vqe_res.get("best_iteration", None)
                }

                results.append(res_entry)

                # 5. Exportação do Gráfico de Diagnóstico utilizando get_file_path 
                if save_plots:
                    diag_path = get_file_path(
                        filename=f"diag_exp_{exp_id}.png",
                        variable_type=self.variable_type,
                        problem_type=self.problem_type,
                        sub_folder=self.sub_folder,
                        is_result=False
                    ) 
                    plot_training_diagnostics(
                        vqe_res=vqe_res,
                        exact_cost=exact_cost if exact_cost else 0.0,
                        output_path=diag_path,
                        title_suffix=f"(Exp #{exp_id})"
                    )

        # 6. Salvar Consolidado Final
        self._save_summary(results)
        return results

    def _save_summary(self, results: List[Dict[str, Any]]) -> None:
        """Exporta os relatórios consolidados nos formatos JSON e CSV através do path.py."""
        json_path = get_file_path(
            filename="grid_search_results.json",
            variable_type=self.variable_type,
            problem_type=self.problem_type,
            sub_folder=self.sub_folder,
            is_result=True
        ) 
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        csv_path = get_file_path(
            filename="grid_search_results.csv",
            variable_type=self.variable_type,
            problem_type=self.problem_type,
            sub_folder=self.sub_folder,
            is_result=True
        ) 
        if results:
            headers = list(results[0].keys())
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
                for row in results:
                    line = ",".join(str(row[h]) if row[h] is not None else "" for h in headers)
                    f.write(line + "\n")

        print(f"\n[OK] Varredura Concluída! Relatórios gerados em: {self.results_dir}")


if __name__ == "__main__":
    # --- Exemplo de Teste do Orquestrador ---
    
    PARAM_GRID = {
        "n_nodes": [4],
        "graph_type": ["euclidean", "clustered"],
        "decoder_type": ["softmax", "argmax"],
        "scheduler_type": ["dynamic", "constant"],
        "optimizer": ["adam"],
        "lr": [0.01],
        "max_iterations": [30]
    }

    # Solver Mock apenas para validação da estrutura
    class DummyCVVQESolver:
        def __init__(self, graph_builder, **kwargs):
            self.gb = graph_builder
            self.max_iterations = kwargs.get("max_iterations", 30)

        def solve(self):
            steps = self.max_iterations
            return {
                "continuous_loss_history": list(np.linspace(15.0, 3.2, steps)),
                "cost_history": list(np.linspace(20.0, 8.0, steps)),
                "best_cost": 8.0,
                "is_feasible": True,
                "best_iteration": int(steps * 0.8),
                "metrics_history": [
                    {
                        "lmbda_col": 1.0 + 0.1 * i,
                        "lmbda_cap": 1.0 + 0.05 * i,
                        "collision_penalty": max(0.0, 5.0 - 0.2 * i),
                        "capacity_violation_magnitude": max(0.0, 3.0 - 0.1 * i)
                    }
                    for i in range(steps)
                ]
            }

    runner = GridSearchRunner(
        solver_class=DummyCVVQESolver,
        variable_type="qumodes",
        problem_type="vrp",
        sub_folder="grid_search"
    )
    runner.run_grid_search(PARAM_GRID, n_runs_per_config=1)