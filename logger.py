# logger.py
import os
import json
import logging
import pandas as pd
from metrics import ExperimentResult


class ExperimentLogger:
    """
    Gerenciador de logs no terminal e de persistência de experimentos.
    Separa a estrutura de saída na pasta 'result', dividindo entre TSP e VRP.
    """
    def __init__(self, problem_type: str = "TSP", base_dir: str = "result"):
        self.base_dir = base_dir
        self.problem_type = problem_type
        
        # Configuração do logger do Python para terminal
        self.logger = logging.getLogger(f"ExperimentLogger_{problem_type}")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # --- MÉTODOS DE LOGGING (Terminal) ---
    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    # --- MÉTODOS DE GERENCIAMENTO DE DIRETÓRIOS E PERSISTÊNCIA ---
    def _get_problem_paths(self, problem_type: str = None):
        """Retorna os caminhos organizados para o tipo de problema especificado."""
        target_problem = problem_type if problem_type is not None else self.problem_type
        folder_name = target_problem.lower()
        prob_dir = os.path.join(self.base_dir, folder_name)
        
        data_dir = os.path.join(prob_dir, "data")
        figures_dir = os.path.join(prob_dir, "figures")
        csv_path = os.path.join(prob_dir, f"{folder_name}_summary.csv")

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        return data_dir, figures_dir, csv_path

    def get_figures_dir(self, problem_type: str = None) -> str:
        """Helper para obter o diretório correto onde salvar plots e gráficos."""
        _, figures_dir, _ = self._get_problem_paths(problem_type)
        return figures_dir

    def save_experiment(self, result: ExperimentResult) -> str:
        """Salva a execução em JSON individual e atualiza o CSV acumulativo da modalidade."""
        data_dir, _, csv_path = self._get_problem_paths(result.problem_type)
        res_dict = result.to_dict()

        # 1. Salva o JSON completo (com o histórico das iterações)
        json_filename = f"{result.solver_name.lower()}_{result.experiment_id}.json"
        json_path = os.path.join(data_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(res_dict, f, indent=4, ensure_ascii=False)

        # 2. Registra na Tabela CSV Consolidada
        summary_dict = {
            k: v for k, v in res_dict.items() 
            if k not in ['cost_history', 'optimal_params']
        }
        summary_dict['exact_route'] = str(summary_dict['exact_route'])
        summary_dict['quantum_route'] = str(summary_dict['quantum_route'])

        df_row = pd.DataFrame([summary_dict])

        if not os.path.exists(csv_path):
            df_row.to_csv(csv_path, index=False)
        else:
            df_row.to_csv(csv_path, mode='a', header=False, index=False)

        self.info(f"Registrado com sucesso em: result/{result.problem_type.lower()}/")
        return json_path