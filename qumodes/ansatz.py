from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Ket, Rgate, Sgate


@dataclass
class CircuitConfig:
    num_qumodes: int
    num_layers: int = 2


class ContinuousVariableAnsatz:
    """Circuito variacional U(theta) em variáveis contínuas.

    Parâmetros de experimento
    -------------------------
    initial_ket : lista de N kets de Fock (um por modo, normalizados) injetados
        antes das camadas. Serve para preparar o estado em escala de Zak — um
        GKP de energia finita, por exemplo — em vez de partir do vácuo. O vácuo
        tem sigma_x = sqrt(hbar/2) e cabe dentro de UMA célula da rede
        (a = sqrt(2*pi*hbar) ~ 3.54), de modo que não resolve os nós de posição.
    use_kerr : desliga a porta Kerr, tornando o circuito puramente gaussiano.
        Serve para a ablação de não-gaussianidade: o mínimo gaussiano do termo
        estabilizador é ~1.0 por modo, e só um estado não-gaussiano desce abaixo
        disso. O avanço do índice de parâmetros NÃO muda quando a porta é
        desligada, para que as duas versões tenham o mesmo vetor theta e sejam
        diretamente comparáveis.
    """

    def __init__(self, config: CircuitConfig,
                 initial_ket: Optional[Sequence[np.ndarray]] = None,
                 use_kerr: bool = True):
        self.N = config.num_qumodes
        self.layers = config.num_layers
        self.num_params_per_layer = self._calculate_params_per_layer()
        self.total_params = self.num_params_per_layer * self.layers
        self.use_kerr = bool(use_kerr)
        self.initial_ket = None
        if initial_ket is not None:
            kets = [np.asarray(k, dtype=np.complex128) for k in initial_ket]
            if len(kets) != self.N:
                raise ValueError(f"initial_ket deve ter {self.N} kets (um por modo).")
            dims = {k.shape for k in kets}
            if len(dims) != 1 or kets[0].ndim != 1:
                raise ValueError("todos os kets devem ser vetores 1-D de mesma dimensão.")
            self.initial_ket = [k / np.linalg.norm(k) for k in kets]

    def _calculate_params_per_layer(self) -> int:
        # Parâmetros por qumode: Squeezing (2), Rotação (1), Deslocamento (2), Kerr (1)
        single_mode_params = 6 * self.N
        # Parâmetros de emaranhamento por par adjacente: Beam Splitter (2)
        entangling_params = 2 * (self.N - 1)
        return single_mode_params + entangling_params

    def generate_initial_theta(self, seed: int = 42, scale: float = 0.05,
                               squeeze_r: float = 0.0) -> np.ndarray:
        """Vetor theta inicial.

        seed       : usa um Generator local; NÃO altera o estado global de numpy,
                     para que repetições independentes sejam realmente independentes.
        scale      : desvio do ruído aplicado a todos os parâmetros.
        squeeze_r  : se != 0, inicializa os parâmetros r de cada Sgate nesse valor
                     (mais ruído pequeno). Com hbar = 2, r ~ 1.2 leva sigma_x à
                     ordem de uma célula da rede; exige cutoff compatível (~16),
                     pois <n> = sinh^2(r) ~ 2.7.
        """
        rng = np.random.default_rng(seed)
        theta = rng.normal(loc=0.0, scale=scale, size=self.total_params)
        if squeeze_r:
            for layer in range(self.layers):
                base = layer * self.num_params_per_layer
                for i in range(self.N):
                    theta[base + 3 * i] = squeeze_r + rng.normal(0.0, 0.01)
        return theta

    def build_program(self, theta: np.ndarray) -> sf.Program:
        """Mapeia theta em um sf.Program."""
        if len(theta) != self.total_params:
            raise ValueError(
                f"Vetor de parâmetros incompatível. Esperado: {self.total_params}, recebido: {len(theta)}"
            )

        prog = sf.Program(self.N)
        idx = 0

        with prog.context as q:
            # 0. Estado inicial (opcional): substitui o vácuo
            if self.initial_ket is not None:
                for i in range(self.N):
                    Ket(self.initial_ket[i]) | q[i]

            for _ in range(self.layers):
                # 1. Camada de Squeezing e Rotação Inicial
                for i in range(self.N):
                    Sgate(theta[idx], theta[idx + 1]) | q[i]
                    Rgate(theta[idx + 2]) | q[i]
                    idx += 3

                # 2. Camada de Emaranhamento (Beam Splitters em cadeia)
                for i in range(self.N - 1):
                    BSgate(theta[idx], theta[idx + 1]) | (q[i], q[i + 1])
                    idx += 2

                # 3. Camada de Deslocamento e Não-Gaussianidade (Kerr)
                for i in range(self.N):
                    Dgate(theta[idx], theta[idx + 1]) | q[i]
                    if self.use_kerr:
                        Kgate(theta[idx + 2]) | q[i]
                    idx += 3          # avanço fixo: theta é o mesmo com ou sem Kerr

        return prog
