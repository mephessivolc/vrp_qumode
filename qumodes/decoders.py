# qumodes/decoders.py
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment


class BaseDecoder(ABC):
    """Classe base abstrata para decodificadores de quadraturas em rotas discretas."""

    @abstractmethod
    def decode(
        self, 
        x_vals: List[float], 
        p_vals: List[float], 
        max_steps: int, 
        num_vehicles: int
    ) -> Tuple[List[int], List[int]]:
        """
        Mapeia quadraturas contínuas (x_vals, p_vals) para atribuições discretas.

        Args:
            x_vals: Posições contínuas da sequência na rota.
            p_vals: Atribuições contínuas de veículos.
            max_steps: Número máximo de passos de tempo discretos.
            num_vehicles: Número total de veículos disponíveis.

        Returns:
            Tuple[List[int], List[int]]: (x_discreto, p_discreto)
        """
        pass


class ArgmaxDecoder(BaseDecoder):
    """
    Decodificador padrão por arredondamento e truncamento (Clip/Round).
    Atribui cada quadratura ao inteiro válido mais próximo.
    """

    def decode(
        self, 
        x_vals: List[float], 
        p_vals: List[float], 
        max_steps: int, 
        num_vehicles: int
    ) -> Tuple[List[int], List[int]]:
        x_disc = [int(np.clip(np.round(x), 1, max_steps)) for x in x_vals]
        p_disc = [int(np.clip(np.round(p), 1, num_vehicles)) for p in p_vals]
        return x_disc, p_disc


class HungarianDecoder(BaseDecoder):
    """
    Decodificador baseado no Algoritmo Húngaro (Linear Sum Assignment).
    Garante a eliminação de colisões em x ao associar cada cidade a um passo único.
    """

    def decode(
        self, 
        x_vals: List[float], 
        p_vals: List[float], 
        max_steps: int, 
        num_vehicles: int
    ) -> Tuple[List[int], List[int]]:
        num_cities = len(x_vals)
        
        # Matriz de custo quadrático entre posição contínua e passos discretos
        cost_x = np.zeros((num_cities, max_steps), dtype=np.float32)
        for i in range(num_cities):
            for k in range(max_steps):
                target_step = k + 1.0
                cost_x[i, k] = (x_vals[i] - target_step) ** 2

        # Bipartite matching para obter mapeamento 1-para-1 único
        row_ind, col_ind = linear_sum_assignment(cost_x)
        
        x_disc = [0] * num_cities
        for r, c in zip(row_ind, col_ind):
            x_disc[r] = int(c + 1)

        # Atribuição de veículo por aproximação direta
        p_disc = [int(np.clip(np.round(p), 1, num_vehicles)) for p in p_vals]

        return x_disc, p_disc


class SoftmaxAnnealingDecoder(BaseDecoder):
    """
    Decodificador probabilístico suavizado por temperatura (Softmax).
    Útil para suavizar a transição discreta em fases iniciais da otimização.
    """

    def __init__(self, temperature: float = 1.0, min_temperature: float = 0.01):
        self.temperature = max(temperature, min_temperature)
        self.min_temperature = min_temperature

    def set_temperature(self, temperature: float):
        self.temperature = max(float(temperature), self.min_temperature)

    def decode(
        self, 
        x_vals: List[float], 
        p_vals: List[float], 
        max_steps: int, 
        num_vehicles: int
    ) -> Tuple[List[int], List[int]]:
        step_targets = np.arange(1, max_steps + 1, dtype=np.float32)
        vehicle_targets = np.arange(1, num_vehicles + 1, dtype=np.float32)

        x_disc = []
        for x in x_vals:
            distances_sq = -(np.array(x, dtype=np.float32) - step_targets) ** 2
            logits = distances_sq / self.temperature
            probs = np.exp(logits - np.max(logits))
            probs /= np.sum(probs)
            selected_step = int(step_targets[np.argmax(probs)])
            x_disc.append(selected_step)

        p_disc = []
        for p in p_vals:
            distances_sq = -(np.array(p, dtype=np.float32) - vehicle_targets) ** 2
            logits = distances_sq / self.temperature
            probs = np.exp(logits - np.max(logits))
            probs /= np.sum(probs)
            selected_vehicle = int(vehicle_targets[np.argmax(probs)])
            p_disc.append(selected_vehicle)

        return x_disc, p_disc