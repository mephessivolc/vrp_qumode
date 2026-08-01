# Explicação da construção do Hamiltoniano usado para o VRP

## Forma Matricial

A modelagem matemática do seu código do VRP baseia-se na **Codificação por Quadraturas Contínuas no Espaço de Fase** $(\hat{x}, \hat{p})$.

Em vez de usar estados discretos da base de Fock ($\ket{1}, \ket{2}, \dots$) em matrizes globais de Hilbert, este algoritmo mapeia o problema diretamente nas **expectativas do espaço de fase quântico**:

$$\mathbf{x} = \begin{pmatrix} \langle \hat{x}_1 \rangle \\ \langle \hat{x}_2 \rangle \\ \vdots \\ \langle \hat{x}_{N-1} \rangle \end{pmatrix}, \quad \mathbf{p} = \begin{pmatrix} \langle \hat{p}_1 \rangle \\ \langle \hat{p}_2 \rangle \\ \vdots \\ \langle \hat{p}_{N-1} \rangle \end{pmatrix}$$

Onde cada qumode $i \in \{1, 2, \dots, N-1\}$ corresponde unicamente a uma **cidade livre** (excluindo o depósito $0$).

---

### 1. As Variáveis Contínuas e a Discretização

Os autovalores contínuos lidos nas quadraturas $(\hat{x}_i, \hat{p}_i)$ são mapeados para variáveis discretas de tempo $\tilde{x}_i \in \{1, 2, \dots, T\}$ e veículo $\tilde{p}_i \in \{1, 2, \dots, V\}$ através de um operador de arredondamento limitado (*clipping*):

$$\tilde{x}_i = \text{clip}\left(\lfloor \langle \hat{x}_i \rangle \rceil, 1, T\right)$$

$$\tilde{p}_i = \text{clip}\left(\lfloor \langle \hat{p}_i \rangle \rceil, 1, V\right)$$

Onde:

* $\lfloor \cdot \rceil$ representa o arredondamento para o inteiro mais próximo (`np.round`).
* $T = N - 1$ é o número máximo de passos de tempo que uma rota pode ter.
* $V$ é o número total de veículos disponíveis.

---

### 2. A Função Indicadora da Rota (Decodificação)

Para converter as atribuições $\tilde{x}_i$ e $\tilde{p}_i$ em uma rota real no grafo, define-se uma matriz binária indicadora implícita $y_{i, v, k}$:

$$y_{i, v, k} = \begin{cases} 1, & \text{se } \tilde{p}_i = v \text{ e } \tilde{x}_i = k \\ 0, & \text{caso contrário} \end{cases}$$

Dado um veículo $v$, seja $R(v) = (u_0, u_1, u_2, \dots, u_{m_v}, u_{m_v + 1})$ a sequência ordenada de paradas onde:

1. $u_0 = 0$ (Partida do Depósito).
2. As cidades $u_j$ para $j \in \{1, \dots, m_v\}$ satisfazem $\tilde{p}_{u_j} = v$ e são ordenadas de forma estritamente crescente pelos seus tempos: $\tilde{x}_{u_1} \le \tilde{x}_{u_2} \le \dots \le \tilde{x}_{u_{m_v}}$.
3. $u_{m_v + 1} = 0$ (Retorno ao Depósito).

---

### 3. O Hamiltoniano do VRP

O Hamiltoniano total avaliado no estado quântico é a soma do custo das distâncias do grafo com o termo de penalidade de colisão:

$$H_{\text{VRP}}(\mathbf{x}, \mathbf{p}) = H_{\text{dist}}(\mathbf{x}, \mathbf{p}) + H_{\text{penalty}}(\mathbf{x}, \mathbf{p})$$

---

### A. Hamiltoniano de Custo / Distância ($H_{\text{dist}}$)

Representa a soma do peso de todas as arestas $(u, w)$ percorridas pela frota de veículos $V$:

$$H_{\text{dist}}(\mathbf{x}, \mathbf{p}) = \sum_{v=1}^V \sum_{k=0}^{\vert{}R(v)\vert{} - 1} d_{R(v)_k, R(v)_{k+1}}$$

Onde $d_{u, w}$ é o elemento $(u, w)$ da matriz de adjacência do grafo.

---

### B. Hamiltoniano de Penalidade por Colisão ($H_{\text{penalty}}$)

Impede que duas cidades distintas $i$ e $j$ sejam visitadas pelo mesmo veículo $v$ no mesmo instante de tempo $k$ (conflito espacial e temporal):

$$H_{\text{penalty}}(\mathbf{x}, \mathbf{p}) = \lambda \sum_{i=1}^{N-1} \sum_{j=i+1}^{N-1} \delta(\tilde{p}_i, \tilde{p}_j) \cdot \delta(\tilde{x}_i, \tilde{x}_j)$$

Onde a função delta de Kronecker $\delta(a, b)$ ativa a penalidade quando há sobreposição:

$$\delta(a, b) = \begin{cases} 1, & \text{se } a = b \\ 0, & \text{se } a \neq b \end{cases}$$

E $\lambda > 0$ é o hiperparâmetro multiplicador de Lagrange.

---

## 4. Formulação Unificada do Hamiltoniano

Agrupando todos os termos, a expressão matemática completa que seu código executa é:

$$H_{\text{total}}(\mathbf{x}, \mathbf{p}) = \sum_{v=1}^V \sum_{k=0}^{\vert{}R(v)\vert{} - 1} d_{R(v)_k, R(v)_{k+1}} + \lambda \sum_{i < j} \delta\Big(\text{clip}\lfloor \langle \hat{p}_i \rangle \rceil, \text{clip}\lfloor \langle \hat{p}_j \rangle \rceil\Big) \cdot \delta\Big(\text{clip}\lfloor \langle \hat{x}_i \rangle \rceil, \text{clip}\lfloor \langle \hat{x}_j \rangle \rceil\Big)$$

---

## Resumo Comparativo dos Modelos

| Métrica | Modelo Fock (Seu TSP) | Modelo Espaço de Fase $(\hat{x}, \hat{p})$ (Seu VRP) |
| --- | --- | --- |
| **Qumodes necessários** | $N$ qumodes | $N-1$ qumodes |
| **Operadores do Hamiltoniano** | Projetores matriciais $\ket{a}\bra{a}_k$ | Funções de expectativa sobre quadraturas $\langle \hat{x}_i \rangle$ e $\langle \hat{p}_i \rangle$ |
| **Cálculo da Energia** | Multiplicação de matrizes $(N+1)^N \times (N+1)^N$ | Amostragem direta / Expectativa das medições no circuito |
| **Escalabilidade Computacional** | Limitada pelo tamanho da matriz de Hilbert | Altamente escalável para simulação variacional no Strawberry Fields |

---
---
# Para o Solver

Explicação do porque no simulador é possível ter apenas uma única execução e duas medidas sem perder coerencia

Exatamente! **Você está absolutamente correto na sua leitura do código.**

No simulador do Strawberry Fields utilizado no seu `solver.py`, a chamada `self.engine.run(prog)` não executa "disparos discretos" (shots), mas sim calcula o **vetor de estado contínuo / matriz densa** de todo o sistema na memória do computador.

Isso é possível por conta da mecânica de simulação analítica do simulador.

---

## O que acontece no seu `solver.py` passo a passo:

```python
# 1. O motor calcula o estado quântico exato |ψ(θ)⟩ a partir dos parâmetros
result = self.engine.run(prog)
state = result.state  # Contém a função de onda analítica completa

# 2. O estado retornado permite consultar <X> e <P> analiticamente
for mode in range(self.num_qumodes):
    x_mean, _ = state.quad_expectation(mode, phi=0.0)         # <X> exato
    p_mean, _ = state.quad_expectation(mode, phi=np.pi/2)     # <P> exato

```

### Como isso resolve o problema do Princípio da Incerteza?

* **No simulador (`fock` ou `gaussian`):** O objeto `state` armazena a matriz densa $\rho$ ou o vetor de estado $\vert{}\psi\rangle$ do qumode em memória sem colapsá-lo. As funções `.quad_expectation(mode, phi)` apenas realizam a integração matemática / produto interno formal da matriz densa com o operador de quadratura:

$$\langle \hat{x} \rangle = \text{Tr}\big(\rho \, \hat{x}\big) \quad \text{e} \quad \langle \hat{p} \rangle = \text{Tr}\big(\rho \, \hat{p}\big)$$

Como é um cálculo puramente matemático sobre a matriz densa guardada na memória, chamar o valor esperado de $\hat{x}$ **não altera e nem destrói** o objeto `state`, permitindo consultar $\hat{p}$ na linha seguinte no mesmo objeto!

---

## A diferença fundamental: Simulador vs. Hardware Real

É vital manter essa distinção em mente para o seu trabalho:

| Aspecto | No seu código atual (`solver.py`) | Em Hardware Quântico Real (Fotônica) |
| --- | --- | --- |
| **Simulação / Execução** | Analítica via Matriz Densa na memória (`state`). | Projeção / Colapso por medições ópticas. |
| **Uso de Shots** | `shots=1` (padrão implícito). Devolve o estado exato. | Exige `shots = N` (ex: 1000 repetições por fase). |
| **Medição das Quadraturas** | `quad_expectation()` lê ambos os operadores do **mesmo** objeto `state`. | Exige **rodar o circuito duas vezes**: uma com detectores em $\phi=0$ e outra com detectores em $\phi=\pi/2$. |
| **Ruído de Amostragem** | Zero (gradiente limpo para o otimizador COBYLA). | Ruído estatístico de amostragem proporcional a $1/\sqrt{\text{shots}}$. |

---

## Conclusão

Sua implementação no `solver.py` é a **forma padrão e ideal de se fazer pesquisa simulada de VQE**:

1. Ela evita o ruído estatístico de amostragem no loop do otimizador clássico (COBYLA), permitindo que ele converja muito mais rápido.
2. Ela aproveita a facilidade do simulador de acessar $\langle \hat{x} \rangle$ e $\langle \hat{p} \rangle$ diretamente do estado analítico sem precisar reconstruir a função de onda ou duplicar as chamadas do circuito no simulador.