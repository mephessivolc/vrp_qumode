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

---

# Debugging do Groun State Energy

Analisando o arquivo `vrp/solver.py`, o parâmetro **`ground_state_energy` (ou `exact_cost`) não é calculado dentro deste arquivo**.

O `solver.py` apenas calcula duas métricas durante o processo:

1. **`continuous_loss`**: A perda contínua e diferenciável calculada a cada passo de otimização via `self.hamiltonian.compute_continuous_cost_tf(x_tens, p_tens)`.


2. **`discrete_cost`**: O custo discreto real da rota decodificada retornado como `best_cost` e `best_energy` ao final do método `solve()`.



O `ground_state_energy` (custo exato/ótimo do grafo) costuma ser gerado externamente por busca exata (força bruta ou programação linear) no script principal (como `main.py`).

---

## Diagnóstico para o Debugging dos Gráficos

A divergência nos gráficos entre o valor otimizado, o custo discreto e o `ground_state_energy` ocorre pelos dois motivos descritos abaixo:

### 1. Por que o valor otimizado aparece ABAIXO do `exact_cost`?

Isso acontece quando o gráfico compara o **`continuous_loss_history`** com o **`exact_cost`** discreto:

* **Causa:** O `continuous_loss` é uma relaxação contínua que utiliza valores esperados das quadraturas $\langle X \rangle$ e $\langle P \rangle$. Nesse espaço contínuo, funções suaves de distância (como os termos exponenciais $e^{-\Delta x^2}$) permitem que o otimizador Adam obtenha custos fracionários irreais (estados não-físicos onde posições se sobrepõem parcialmente).


* **Consequência:** A perda contínua pode cair abaixo da menor distância discreta possível do grafo. Esse valor contínuo abaixo do ótimo é um "artefato da relaxação" e desaparece no momento da discretização (`discretize_quadratures`).



### 2. Por que a otimização NÃO ALCANÇA o `ground_state_energy` em alguns casos?

Quando o custo discreto estagna em um valor superior ao ótimo exato, o problema está na transição entre o espaço contínuo e a discretização:

* **Armadilhas de Mínimos Locais:** A paisagem de perda contínua criada pelas penalidades de repulsão contêm múltiplos poços potenciais locais. O otimizador Adam pode convergir para uma região onde os gradientes zeram antes de encontrar a bacia de atração do ótimo global.
* **Arredondamento na Discretização:** Pequenas variações nos valores contínuos de $x$ e $p$ geram arredondamentos abruptos no `discretize_quadratures`. Se duas cidades contínuas próximas forem arredondadas para o mesmo inteiro, a função discreta ativa penalidades de colisão severas ($\lambda = 100$), elevando o custo final desproporcionalmente.



---

## Como corrigir o código para os testes?

1. **Garantir a mesma métrica na plotagem:** No seu script de plotagem/benchmark, certifique-se de comparar o `exact_cost` contra o **`cost_history`** (custo discreto) em vez do **`continuous_loss_history`**.


2. **Normalização do Custo Exato:** Verifique se o cálculo do `exact_cost` exato considera apenas as distâncias da matriz $d_{ij}$ ou se inclui as penalidades $\lambda$. Para comparar corretamente com a rota válida, o `exact_cost` deve refletir estritamente a soma das distâncias das arestas percorrida no grafo.

---
---
# Modelagem para o hamiltoniano do VRP e TSP

O modelo matematico do Hamiltoniano para o **CVRP** (e **TSP**) em VQE de Variáveis Contínuas (CV-VQE) mapeia as variáveis discretas de roteamento em observáveis quânticos contínuos (quadraturas de Posição $\hat{x}$ e Momento $\hat{p}$) medidos em cada qumode (cidade $i \in \{1, \dots, N\}$).


## 1. Mapeamento do Espaço de Fase Quântico

Cada cidade $i$ é associada a um qumode cuja medição fornece o par de quadraturas $(x_i, p_i)$:

* **Posição ($x_i$):** Representa continuous/soft a **ordem de visitação/passo temporal** da cidade no circuito.
* **Momento ($p_i$):** Representa continuous/soft o **veículo atribuído** à cidade $i$, onde $p_i \in [1, K]$ para $K$ veículos.



## 2. Formulação Geral do Hamiltoniano Total

A função de perda diferenciável (Hamiltoniano efetivo) otimizada pelo VQE é composta pela soma do custo contínuo de distância e dos termos de penalização:

$$H_{\text{total}} = H_{\text{dist}} + \lambda \cdot H_{\text{col}} + H_{\text{cap}} + H_{\text{bound}}$$



## 3. Relações Matemáticas das Penalizações

### A. Penalidade de Colisão Temporal / Posição ($H_{\text{col}}$)

Impede que duas cidades distintas $i \neq j$ ocupem o mesmo passo temporal dentro da rota do mesmo veículo. A sobreposição é aproximada continuamente via curvas Gaussianas no espaço de fase:

$$H_{\text{col}} = \sum_{i=1}^{N} \sum_{j \neq i}^{N} \exp\left(-\frac{(x_i - x_j)^2}{2\sigma_x^2}\right) \cdot \exp\left(-\frac{(p_i - p_j)^2}{2\sigma_p^2}\right)$$

* **$\exp\left(-\frac{(x_i - x_j)^2}{2\sigma_x^2}\right)$:** Penaliza cidades com ordens temporais $x_i \approx x_j$ idênticas.
* **$\exp\left(-\frac{(p_i - p_j)^2}{2\sigma_p^2}\right)$:** Garante que a penalidade de ordem só seja aplicada fortemente se as cidades pertencerem ao mesmo veículo ($p_i \approx p_j$).

### B. Penalidade de Capacidade de Veículo ($H_{\text{cap}}$)

Garante que a soma das demandas $q_i$ atribuídas ao veículo $v$ não ultrapasse sua capacidade máxima $C_v$.

1. **Estimativa de Carga Contínua do Veículo $v$ ($L_v$):**
Mede a carga total associada suavemente ao veículo $v \in \{1, \dots, K\}$ através do grau de pertinência Gaussiano de cada cidade ao veículo:

$$L_v = \sum_{i=1}^{N} q_i \cdot \exp\left(-\frac{(p_i - v)^2}{2\sigma_p^2}\right)$$


2. **Excesso de Capacidade ($\text{Overcapacity}_v$):**
Computa a violação utilizando a função ReLU/Maximum suave:

$$\text{Overcapacity}_v = \max(0, L_v - C_v)$$


3. **Termo de Penalização:**

$$H_{\text{cap}} = \begin{cases} \lambda_{\text{cap}} \displaystyle\sum_{v=1}^{K} \left[\max(0, L_v - C_v)\right]^2, & \text{se } \text{CVRP } (K > 1) \\ 0.0, & \text{se } \text{TSP } (K = 1) \end{cases}$$


### C. Penalidade de Confinamento / Limites de Domínio ($H_{\text{bound}}$)

Força as expectativas do circuito quântico a permanecerem dentro dos limites físicos válidos do problema ($x_i \in [1, N]$ e $p_i \in [1, K]$):

$$H_{\text{bound}} = \alpha \sum_{i=1}^{N} \left[ \max(0, 1 - x_i)^2 + \max(0, x_i - N)^2 + \max(0, 1 - p_i)^2 + \max(0, p_i - K)^2 \right]$$

Onde $\alpha$ é uma constante de confinamento (e.g., $\alpha = 10.0$).

### D. Custo Suave de Distância ($H_{\text{dist}}$)

Estima a distância total percorrida conectando sequencialmente os vértices ordenados $x_i \to x_j$ para cidades associadas ao mesmo veículo $p_i \approx p_j \approx v$:

$$H_{\text{dist}} = \sum_{v=1}^{K} \sum_{i=1}^{N} \sum_{j \neq i}^{N} d_{ij} \cdot \exp\left(-\frac{(x_j - x_i - 1)^2}{2\sigma_x^2}\right) \cdot \exp\left(-\frac{(p_i - v)^2 + (p_j - v)^2}{2\sigma_p^2}\right)$$

## Resumo dos Multiplicadores e Comportamento

| Termo | Função no Otimizador | Escala Sugerida |
| --- | --- | --- |
| $\lambda$ ($H_{\text{col}}$) | Evita sobreposição de passos temporais entre cidades no mesmo veículo. | $100.0 - 500.0$ |
| $\lambda_{\text{cap}}$ ($H_{\text{cap}}$) | Evita sobrecarga de veículos ($L_v > C_v$). Zera automaticamente em TSPs. | $500.0 - 1000.0$ |
| $\alpha$ ($H_{\text{bound}}$) | Impede divergência do circuito para regiões fora de $[1, N] \times [1, K]$. | $10.0$ |

---
---
# Gap de distância
Com certeza! Essa alteração no **gap de busca** (ou espaçamento/resolução no espaço de fase) é um dos pontos mais críticos quando migramos a formulação quântica do TSP para o **CVRP**.

No TSP, trabalhamos praticamente em uma única dimensão temporal ($x$), pois há apenas 1 veículo ($p \approx 1.0$). Quando passamos para o CVRP, o espaço de fase precisa mapear $K$ veículos distintos, e a definição desse "gap" afeta diretamente a capacidade do otimizador de encontrar soluções viáveis.

---

## O que é o "Gap" no contexto de CV-VQE?

Quando falamos de alteração no gap do modelo quântico de Variáveis Contínuas (CV), geralmente estamos lidando com três conceitos interligados:

### 1. Espaçamento entre Atribuições de Veículos ($\Delta p$)

* **O Problema:** No espaço de fase, o valor de $p_i$ determina a qual veículo a cidade $i$ pertence.
* **A Melhoria:** Definir um gap discreto claro (ex: $p \in \{1.0, 2.0, \dots, K\}$ com $\Delta p = 1.0$) garante que o otimizador consiga diferenciar com precisão quando uma cidade muda do Veículo 1 para o Veículo 2, sem que estados intermediários gerem ambiguidades.

### 2. Largura de Suavização Gaussiana ($\sigma_p$ e $\sigma_x$)

* **O Problema:** As penalidades de colisão e capacidade usam funções do tipo $\exp\left(-\frac{(p_i - v)^2}{2\sigma_p^2}\right)$.
* **A Melhoria:** O parâmetro $\sigma_p$ atua como a "largura do gap" de atração contínua.
* Se $\sigma_p$ for **muito pequeno**, o gradiente desaparece quando a cidade está um pouco distante do inteiro ($p \approx 1.4$).
* Se $\sigma_p$ for **adequado**, cria-se um "vale" suave que guia o gradiente do otimizador em direção ao veículo correto sem travar em platôs nulos.



### 3. Janela de Discretização e Arredondamento (*Binning Gap*)

* **O Problema:** Na medição do estado quântico, obtemos valores contínuos como $p_1 = 1.82$ e $x_1 = 2.15$.
* **A Melhoria:** O gap de discretização estabelece a tolerância para mapear $p_1 \to 2$ (Veículo 2) e $x_1 \to 2$ (Passo temporal 2). Ajustar esse limite evita que cidades sejam atribuídas erroneamente ao veículo errado devido a pequenas flutuações quânticas.


---
# Condições no algoritom para evitar Degenerecencia Espúria 

A implementação das condições de contorno e a auditoria de semelhança do modelo CV-VQE estão mapeadas nos arquivos do projeto.

**Mapeamento das Condições de Contorno no Código**

* **Confinamento de Domínio ($H_{\text{bound}}$):**
Implementado em `vrp/hamiltonian.py` no método `compute_continuous_cost_tf`. Penaliza quadraticamente extravasamentos de $x \notin [1, \text{max\_steps}]$ e $p \notin [1, \text{num\_vehicles}]$ com fator multiplicativo $10.0$.


* **Repulsão e Prevenção de Colisão:**
Executado via potencial inverso $1.0 / (\text{dist\_sq} + 0.1)$ no espaço de fase $(x, p)$. Impede que duas cidades colapsem no mesmo ponto temporal e veículo durante a otimização.


* **Penalidade Suave de Capacidade:**
Modelado pela ponderação gaussiana do pertencimento da cidade ao veículo $v$ ($\exp(-(p_i - v)^2)$). Sobrecargas aplicam penalidade quadrática multiplicada por $\lambda_{\text{cap}}$.


* **Quebra de Simetria e Posições Iniciais:**
Definido em `vrp/circuit.py` dentro de `initialize_random_params`. Intercala $p_{\text{target}} = (i \pmod K) + 1$ no primeiro passo do `Dgate`, evitando que o VQE inicie em um ponto selha simétrico.


* **Estabilidade de Otimização:**
Garantido em `vrp/solver.py` com `tf.clip_by_global_norm(grads, 5.0)` e extração da parte real via `tf.math.real`, impedindo explosões de gradiente no espaço de fase.



---

**Métricas de Semelhança e Validação Integro-Clássica**

* **Validação do Ground Truth (`core/brute_force.py`):**
O `BruteForce` descarta partições onde `load > self.capacities[v_idx - 1]`. Garante que a base do `approx_ratio` venha de uma solução exata viável do CVRP.


* **Contrato de Dados do Experimento (`metrics.py`):**
A dataclass `ExperimentResult` registra os hiperparâmetros de contorno (`lmbda`, `lmbda_cap`, `vehicle_capacity`, `demands`), o histórico de convergência e a razão de eficiência $\text{approx\_ratio} = \frac{\text{exact\_cost}}{\text{quantum\_cost}}$.


* **Análise Visual da Discretização (`vrp/main.py`):**
A função `plot_phase_space` plota o desvio vetorial entre as quadraturas contínuas $(x_{\text{cont}}, p_{\text{cont}})$ do circuito e suas projeções discretas $(x_{\text{disc}}, p_{\text{disc}})$.



---

**Ponto de Ajuste Recomendado no `vrp/hamiltonian.py**`

No `compute_continuous_cost_tf`, o termo de colisão utiliza repulsão por inverso da distância ($\frac{1}{\text{dist\_sq} + 0.1}$). Em instâncias com muitas cidades, essa formulação pode gerar gradientes muito altos se duas cidades se aproximarem. Substituir por uma gaussiana repulsiva do tipo $\exp(-\frac{\text{dist\_sq}}{2\sigma_x^2})$ confina a repulsão localmente e suaviza a superfície de perda.


___
---
---
# Depois da Refatoração
A refatoração do sistema consolida a transição de um otimizador contínuo genérico para um framework de CV-VQE focado em problemas de roteamento (CVRP/TSP), permitindo extrair métricas de viabilidade física, qualidade de rota e dinâmica de convergência.

**Métricas Esperadas**

* **`is_feasible` (Booleano):** Indica se as rotas decodificadas respeitam simultaneamente a visitação única de cada cidade, a ordenação temporal e os limites de capacidade dos veículos.
* **`route_distance` (Float):** Distância física total percorrida pela frota, isolada de qualquer termo de penalidade matemática do Hamiltoniano.
* **`capacity_violation` (Float):** Soma escalar do excesso de carga alocada aos veículos além de suas capacidades máximas.
* **`composite_score` (Float):** Métrica sintética que pondera a razão de aproximação relativa ao valor exato e a penalidade por violação de restrições.
* **`approx_ratio` (Float):** Qualidade da solução em relação ao valor ótimo global ($\frac{C_{\text{exato}}}{C_{\text{VQE}}}$).
* **`continuous_loss_history` e `cost_history` (Arrays):** Evolução temporal da perda contínua diferenciável (suave) calculada pelo TensorFlow e do custo discreto decorrente do arredondamento das quadraturas no espaço de fase ($x, p$).

**Resultados e Comportamentos Esperados**

* **Aceleração de Convergência via Warm-Start:**
* *Resultado Esperado:* Aumento expressivo da taxa de sucesso (`is_feasible = True`) e redução no número de iterações para atingir a convergência quando comparado à inicialização aleatória.
* *Justificativa:* O mapeamento de rotas válidas para os parâmetros do `Dgate` via raio $r = \sqrt{\alpha_x^2 + \alpha_p^2}$ e fase $\phi = \arctan2(\alpha_p, \alpha_x)$ inicializa os qumodes diretamente nas regiões de maior probabilidade no espaço de fase, quebrando a simetria inicial dos modos e evitando mínimos locais superficiais.


* **Superação de Platôs via Injeção de Ruído:**
* *Resultado Esperado:* Presença de picos pontuais na perda contínua (`continuous_loss_history`) seguidos por quedas para níveis de energia inferiores ao estado estagnado anterior.
* *Justificativa:* Quando a variação da perda atinge $\vert{}\Delta L\vert{} < 10^{-4}$ durante o período de paciência (`plateau_patience`), a adição de ruído gaussiano aos parâmetros variacionais desloca o estado quântico de platôs de gradiente nulo (*barren plateaus*).


* **Evolução da Viabilidade via Annealing de Penalidades (`penalty_gamma`):**
* *Resultado Esperado:* Estágios iniciais focados na minimização da distância geométrica, seguidos por uma rápida correção em direção a rotas válidas nas iterações finais.
* *Justificativa:* O escalonamento dinâmico $\lambda(k) = \lambda_0 \cdot \gamma^k$ evita que os termos de penalidade se sobreponham excessivamente aos termos de distância no início do treinamento, garantindo maior explorabilidade do espaço de busca antes de impor o colapso estrito das restrições.


* **Divergência de Trajetória entre ADAM e SPSA:**
* *Resultado Esperado:* O ADAM apresentará curvas de convergência mais suaves e estáveis, enquanto o SPSA apresentará oscilações estocásticas em cada passo, mas com menor custo de processamento por iteração.
* *Justificativa:* O ADAM utiliza diferenciação automática exata via `tf.GradientTape`, calculando o gradiente analítico das quadraturas. O SPSA aproxima o gradiente amostrando apenas duas avaliações perturbadas ($w \pm c_k \delta$), o que introduz ruído na estimativa da direção de descida.

___
___
___

**Perda Contínua**
É o valor da função objetivo (expectativa do Hamiltoniano) otimizada diretamente pelo otimizador contínuo (como o otimizador Adam) a cada iteração. Ela combina o custo estimado do trajeto com as penalidades matemáticas adicionadas para forçar o circuito quântico a respeitar as restrições do problema (por exemplo, visitar cada cidade exatamente uma vez). Por ser avaliada em um espaço contínuo de parâmetros quânticos, essa curva costuma apresentar oscilações e valores elevados devido ao peso das penalidades.

**Custo Discreto**
É a distância física real da rota resultante quando o estado quântico contínuo daquela iteração é decodificado/amostrado em uma solução combinatorial válida (uma sequência discreta de cidades, como `[0, 1, 2, 3, 0]`). Esse parâmetro mede a qualidade prática da rota que o algoritmo quântico gerou naquele momento, desconsiderando os termos adicionais de penalidade da função de perda contínua.

**Ground Truth**
É a solução ótima global calculada previamente por um método clássico exato (como o Brute Force). Ela representa o menor custo físico possível para percorrer todas as cidades da instância (no seu caso, o valor 24,05). Nos gráficos de convergência, o Ground Truth é plotado como uma linha horizontal fixa para servir de baseline, permitindo visualizar em qual iteração o *Custo Discreto* atinge ou se aproxima do resultado ideal.