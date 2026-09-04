# logger.py
import os
import json
import logging
import numpy as np
import pandas as pd
from metrics import ExperimentResult

from path import get_results_path, get_images_path, get_path


class NpEncoder(json.JSONEncoder):
    """Encoder para garantir que tipos do NumPy/TensorFlow sejam serializáveis em JSON."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(NpEncoder, self).default(obj)


class ExperimentLogger:
    """
    Gerenciador de logs no terminal e de persistência unificada de experimentos.
    Delega a gestão dos caminhos de saída para o módulo path.py.
    """
    def __init__(self, variable_type: str = "QUMODES", problem_type: str = "TSP", sub_folder: str = None):
        self.variable_type = variable_type
        self.problem_type = problem_type
        self.sub_folder = sub_folder
        
        self.logger = logging.getLogger(f"ExperimentLogger_{variable_type}_{problem_type}")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def _manage_names(self, variable_type: str = "QUMODES", problem_type: str = None, sub_folder: str = None) -> tuple:
        variable_type_name = variable_type.lower()
        target_problem = problem_type if problem_type is not None else self.problem_type
        folder_problem_name = target_problem.lower() if target_problem is not None else "vrp"
        
        target_sub_folder = sub_folder if sub_folder is not None else self.sub_folder
        sub_folder_name = target_sub_folder.lower() if target_sub_folder is not None else None

        return variable_type_name, folder_problem_name, sub_folder_name

    def _get_problem_paths(self, variable_type: str = "QUMODES", problem_type: str = None, sub_folder: str = None):
        variable_type_name, folder_name, sub_folder_name = self._manage_names(variable_type, problem_type, sub_folder)
        
        data_dir = get_results_path(variable_type=variable_type_name, problem_type=folder_name, sub_folder=sub_folder_name)
        figures_dir = get_images_path(variable_type=variable_type_name, problem_type=folder_name, sub_folder=sub_folder_name)
        
        prob_dir = get_path(variable_type=variable_type_name, problem_type=folder_name, sub_folder=sub_folder_name).parent
        csv_path = prob_dir / f"{variable_type_name}_{folder_name}_summary.csv"

        return data_dir, figures_dir, csv_path

    def get_figures_dir(self, variable_type: str = "QUMODES", problem_type: str = None, sub_folder: str = None) -> str:
        _, figures_dir, _ = self._get_problem_paths(variable_type=variable_type, problem_type=problem_type, sub_folder=sub_folder)
        return str(figures_dir)

    def save_experiment(self, result: ExperimentResult, sub_folder: str = None) -> str:
        """Salva/anexa a execução em um único JSON consolidado e atualiza o CSV da modalidade."""
        target_sub_folder = sub_folder if sub_folder is not None else self.sub_folder
        data_dir, _, csv_path = self._get_problem_paths(result.variable_type, result.problem_type, sub_folder=target_sub_folder)
        res_dict = result.to_dict()

        # 1. Caminho para o JSON único acumulativo
        v_type_str, p_type_str, _ = self._manage_names(result.variable_type, result.problem_type, target_sub_folder)
        json_path = data_dir / f"{v_type_str}_{p_type_str}_results.json"

        # 2. Carrega histórico existente ou inicializa lista vazia
        experiments_list = []
        if json_path.exists() and json_path.stat().st_size > 0:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    experiments_list = json.load(f)
            except Exception as e:
                self.warning(f"Falha ao ler JSON existente ({e}). Criando novo registro.")
                experiments_list = []

        # 3. Anexa o novo experimento e reescreve o arquivo
        experiments_list.append(res_dict)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(experiments_list, f, indent=4, ensure_ascii=False, cls=NpEncoder)

        # 4. Registra na Tabela CSV Consolidada
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

        self.info(f"Registro anexado com sucesso em: {json_path}")
        return str(json_path)