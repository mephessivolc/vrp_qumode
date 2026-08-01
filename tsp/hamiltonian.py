# tsp/hamiltonian.py
import numpy as np
from scipy import sparse
from typing import Tuple

# Usaremos o BruteForce da raiz para os testes do módulo
from brute_force import BruteForce


class Hamiltonian:
    """
    Constrói os operadores do Hamiltoniano do TSP para Variáveis Contínuas (CV)
    utilizando N qumodes na base de Fock.
    """
    def __init__(self, dist_matrix: np.ndarray, lmbda: float = 50.0):
        """
        Parameters
        ----------
        dist_matrix : np.ndarray
            Matriz de distâncias/pesos do grafo (N x N).
        lmbda : float, optional
            Fator de penalização para rotas inválidas (cidades repetidas).
        """
        self.dist_matrix = np.array(dist_matrix, dtype=float)
        self.N = len(dist_matrix)
        self.lmbda = lmbda
        # O cutoff dimension precisa acomodar os estados |1> ate |N> (índices 0..N)
        self.cutoff_dim = self.N + 1

    def build_fock_projector(self, city_a: int) -> sparse.csr_matrix:
        """
        Cria a matriz do operador projetor |a><a| no espaço de Hilbert de 1 qumode.
        """
        proj = sparse.dok_matrix((self.cutoff_dim, self.cutoff_dim), dtype=complex)
        if city_a < self.cutoff_dim:
            proj[city_a, city_a] = 1.0
        return proj.tocsr()

    def get_hamiltonian_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gera as matrizes densas de H_dist, H_penalty e H_total no espaço de Fock global.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (H_dist, H_penalty, H_total) como matrizes NumPy de dimensão (N+1)^N x (N+1)^N.
        """
        dim_total = self.cutoff_dim ** self.N
        
        H_dist_sparse = sparse.csr_matrix((dim_total, dim_total), dtype=complex)
        H_penalty_sparse = sparse.csr_matrix((dim_total, dim_total), dtype=complex)
        I_single = sparse.eye(self.cutoff_dim, dtype=complex)

        # 1. CONSTRUÇÃO DO H_dist = \sum d_{ab} |a><a|_k \otimes |b><b|_{k+1}
        for k in range(self.N):
            k_next = (k + 1) % self.N  # Ciclo fechado da rota

            # As cidades variam no intervalo [0, N-1] referente aos índices da matriz de adjacência
            for a in range(self.N):
                for b in range(self.N):
                    d_ab = float(self.dist_matrix[a, b])
                    
                    if d_ab > 0:
                        # No Fock state usaremos os rótulos 1..N (deslocados por +1 para evitar o estado de vácuo |0>)
                        proj_a = self.build_fock_projector(a + 1)
                        proj_b = self.build_fock_projector(b + 1)

                        op_list = [I_single] * self.N
                        op_list[k] = proj_a
                        op_list[k_next] = proj_b

                        # Produto Kronecker esparso sequencial
                        term = op_list[0]
                        for op in op_list[1:]:
                            term = sparse.kron(term, op, format="csr")

                        H_dist_sparse += d_ab * term

        # 2. CONSTRUÇÃO DO H_penalty = \sum_a (\hat{N}_a - I)^2
        # Onde \hat{N}_a = \sum_k |a><a|_k
        I_total = sparse.eye(dim_total, dtype=complex)
        
        for a in range(1, self.N + 1):
            N_a_operator = sparse.csr_matrix((dim_total, dim_total), dtype=complex)
            proj_a = self.build_fock_projector(a)

            for k in range(self.N):
                op_list = [I_single] * self.N
                op_list[k] = proj_a

                term = op_list[0]
                for op in op_list[1:]:
                    term = sparse.kron(term, op, format="csr")

                N_a_operator += term

            diff = N_a_operator - I_total
            H_penalty_sparse += diff @ diff

        H_total_sparse = H_dist_sparse + self.lmbda * H_penalty_sparse

        # Converte para denso apenas no retorno final
        return H_dist_sparse.toarray(), H_penalty_sparse.toarray(), H_total_sparse.toarray()


if __name__ == "__main__":
    from graphs import GraphBuilder

    N = 3
    gb = GraphBuilder(n=N, seed=42)

    print("==========================================================")
    print("      TESTANDO CONSTRUÇÃO DO HAMILTONIANO (TSP)           ")
    print("==========================================================")
    print("Matriz de Adjacência:")
    print(gb.matrix)
    print("----------------------------------------------------------\n")

    # Calculando Ground Truth pelo BruteForce unificado
    bf = BruteForce(gb.matrix, num_vehicles=1)
    exact_cost, exact_route = bf.solve()

    # Construindo o Hamiltoniano
    hamiltonian_builder = Hamiltonian(gb.matrix, lmbda=100.0)
    H_dist, H_pen, H_tot = hamiltonian_builder.get_hamiltonian_matrices()

    # Diagonalização exata para checar o autovalor fundamental
    eigenvalues = np.real(np.linalg.eigvalsh(H_tot))
    e0 = np.min(eigenvalues)

    print(f"Dimensão da Matriz Global: {H_tot.shape[0]} x {H_tot.shape[1]}")
    print(f"Custo Mínimo Exato (Força Bruta): {exact_cost:.4f} (Rota: {exact_route})")
    print(f"Menor Autovalor de H_total (E0 Quântico): {e0:.4f}")
    
    # Validação se E0 do Hamiltoniano bate com a melhor rota da Força Bruta
    if np.isclose(e0, exact_cost):
        print("\n[SUCESSO] A energia fundamental do Hamiltoniano coincide perfeitamente com a Força Bruta!")
    else:
        print("\n[ATENÇÃO] Houve divergência entre E0 e a Força Bruta. Verifique o valor de lambda ou o cutoff.")
    print("==========================================================")