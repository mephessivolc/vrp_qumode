# logger.py
import os
import json
import logging
import pandas as pd
from metrics import ExperimentResult

# --- INTEGRAÇÃO COM PATH.PY ---
# Importa as funções responsáveis pela gestão de caminhos
from path import get_results_path, get_images_path, get_path


class ExperimentLogger:
    """
    Gerenciador de logs no terminal e de persistência de experimentos.
    Delega a gestão dos caminhos de saída para o módulo path.py.
    """
    def __init__(self, problem_type: str = "TSP"):
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
        """Retorna os caminhos organizados integrados via path.py."""
        target_problem = problem_type if problem_type is not None else self.problem_type
        folder_name = target_problem.lower()
        
        # Obtém os diretórios absolutos dinamicamente do path.py
        data_dir = get_results_path(problem_type=folder_name)
        figures_dir = get_images_path(problem_type=folder_name)
        
        # O CSV consolidado fica na raiz da pasta do problema (ex: .../vrp/vrp_summary.csv)
        prob_dir = get_path(problem_type=folder_name).parent
        csv_path = prob_dir / f"{folder_name}_summary.csv"

        return data_dir, figures_dir, csv_path

    def get_figures_dir(self, problem_type: str = None) -> str:
        """Helper para obter o diretório correto onde salvar plots e gráficos."""
        _, figures_dir, _ = self._get_problem_paths(problem_type, variable_type)
        return str(figures_dir)

    def save_experiment(self, result: ExperimentResult) -> str:
        """Salva a execução em JSON individual e atualiza o CSV acumulativo da modalidade."""
        data_dir, _, csv_path = self._get_problem_paths(result.problem_type, result.variable_type)
        res_dict = result.to_dict()

        # 1. Salva o JSON completo (com o histórico das iterações)
        json_filename = f"{result.solver_name.lower()}_{result.experiment_id}.json"
        json_path = data_dir / json_filename
        
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

        self.info(f"Registrado com sucesso em: {json_path}")
        return str(json_path)