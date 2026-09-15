A adaptação do Traveling Salesperson Problem (TSP) e do Vehicle Routing Problem (VRP) para o arcabouço **CCV-QAOA** de Madani et al. permite reduzir o número de qumodes pela metade em relação à formulação CV tradicional, mapeando pares de variáveis binárias nos operadores Hermitianos de posição ($\hat{x}$) e momento ($\hat{p}$).

**Mapeamento no Espaço de Fase e Redução de Qumodes**
Para um TSP com $N$ cidades e $N$ passos temporais, agrupa-se a matriz binária tradicional $X_{i,t} \in \{0,1\}$ em qumodes complexos $z_{i,m} = \hat{x}_{i,m} + i\hat{p}_{i,m}$, onde $m \in \{1, \dots, \lceil N/2 \rceil\}$. A quadratura $\hat{x}_{i,m}$ representa a presença da cidade $i$ no tempo $t=2m-1$, enquanto $\hat{p}_{i,m}$ representa o tempo $t=2m$.

**Hamiltoniano de Custo CCV-Hermitiano Proposto ($H_C$)**
O Hamiltoniano total de custo é puramente Hermitiano e formulado como:


$$H_C = H_{\text{dist}} + \lambda_1 H_{\text{tour}} + \lambda_2 H_{\text{bin}} + \lambda_3 H_{\text{cap}}$$

* **Hamiltoniano de Distância ($H_{\text{dist}}$):** Mede o custo de deslocamento entre cidades consecutivas na matriz $D_{ij}$. Para preservar a Hermiticitade nos termos cruzados entre quadraturas, aplica-se o produto simetrizado de operadores:



$$H_{\text{dist}} = \sum_{i,j} D_{ij} \sum_{m} \left[ \frac{1}{2}(\hat{x}_{i,m}\hat{p}_{j,m} + \hat{p}_{j,m}\hat{x}_{i,m}) + \frac{1}{2}(\hat{p}_{i,m}\hat{x}_{j,m+1} + \hat{x}_{j,m+1}\hat{p}_{i,m}) \right]$$


* **Potencial Poço de Binarização ($H_{\text{bin}}$):** Como as quadraturas variam continuamente em $\mathbb{R}$, introduz-se um potencial quártico não-Gaussiano para forçar os autovalores de $\hat{x}$ e $\hat{p}$ a colapsarem em $\{0,1\}$:



$$H_{\text{bin}} = \sum_{i,m} \left( \hat{x}_{i,m}^2(\hat{x}_{i,m}-1)^2 + \hat{p}_{i,m}^2(\hat{p}_{i,m}-1)^2 \right)$$


* **Penalidade de Estrutura do Tour ($H_{\text{tour}}$):** Utiliza-se a reformulação de penalidade de igualdade $V_E(z) = \lambda \vert{}\vert{}g(z)\vert{}\vert{}^2$ proposta no artigo para impor a restrição de que cada cidade seja visitada uma vez e cada passo contenha apenas uma cidade:

$$H_{\text{tour}} = \sum_{m} \left( \sum_i \hat{x}_{i,m} - 1 \right)^2 + \sum_{m} \left( \sum_i \hat{p}_{i,m} - 1 \right)^2 + \sum_i \left( \sum_m (\hat{x}_{i,m} + \hat{p}_{i,m}) - 1 \right)^2$$


* **Extensão VRP: Capacidade dos Veículos ($H_{\text{cap}}$):** Para o VRP com $K$ veículos de capacidade $C_k$ e demandas $d_i$, codifica-se a restrição de desigualdade via função retificadora suave Swish $V_I(z) = \sum R(\lambda h_j(z))$ descrita por Madani et al., onde $R(u) = u \cdot \sigma(u)$ e $\sigma(u) = (1 + e^{-u})^{-1}$:

$$H_{\text{cap}} = \sum_{k=1}^K R\left(\lambda_{\text{cap}} \left( \sum_i d_i \hat{x}_{i,k}^{\text{veh}} - C_k \right)\right)$$



Esta proposta reduz a dimensão do espaço de Hilbert simulado de $\mathcal{O}(D^{N^2})$ para $\mathcal{O}(D^{N^2/2})$ através do empacotamento complexo, garantindo a evolução unitária Hermitiana necessária para o circuito variacional $e^{-i\gamma H_C}$.

# Demonstração de Hamiltoniano Hermetiano

A adaptação do Traveling Salesperson Problem (TSP) e do Vehicle Routing Problem (VRP) para o arcabouço **CCV-QAOA** de Madani et al. permite reduzir o número de qumodes pela metade em relação à formulação CV tradicional, mapeando pares de variáveis binárias nos operadores Hermitianos de posição ($\hat{x}$) e momento ($\hat{p}$).

**Mapeamento no Espaço de Fase e Redução de Qumodes**
Para um TSP com $N$ cidades e $N$ passos temporais, agrupa-se a matriz binária tradicional $X_{i,t} \in \{0,1\}$ em qumodes complexos $z_{i,m} = \hat{x}_{i,m} + i\hat{p}_{i,m}$, onde $m \in \{1, \dots, \lceil N/2 \rceil\}$. A quadratura $\hat{x}_{i,m}$ representa a presença da cidade $i$ no tempo $t=2m-1$, enquanto $\hat{p}_{i,m}$ representa o tempo $t=2m$.

**Hamiltoniano de Custo CCV-Hermitiano Proposto ($H_C$)**
O Hamiltoniano total de custo é puramente Hermitiano e formulado como:


$$H_C = H_{\text{dist}} + \lambda_1 H_{\text{tour}} + \lambda_2 H_{\text{bin}} + \lambda_3 H_{\text{cap}}$$

* **Hamiltoniano de Distância ($H_{\text{dist}}$):** Mede o custo de deslocamento entre cidades consecutivas na matriz $D_{ij}$. Para preservar a Hermiticitade nos termos cruzados entre quadraturas, aplica-se o produto simetrizado de operadores:



$$H_{\text{dist}} = \sum_{i,j} D_{ij} \sum_{m} \left[ \frac{1}{2}(\hat{x}_{i,m}\hat{p}_{j,m} + \hat{p}_{j,m}\hat{x}_{i,m}) + \frac{1}{2}(\hat{p}_{i,m}\hat{x}_{j,m+1} + \hat{x}_{j,m+1}\hat{p}_{i,m}) \right]$$


* **Potencial Poço de Binarização ($H_{\text{bin}}$):** Como as quadraturas variam continuamente em $\mathbb{R}$, introduz-se um potencial quártico não-Gaussiano para forçar os autovalores de $\hat{x}$ e $\hat{p}$ a colapsarem em $\{0,1\}$:



$$H_{\text{bin}} = \sum_{i,m} \left( \hat{x}_{i,m}^2(\hat{x}_{i,m}-1)^2 + \hat{p}_{i,m}^2(\hat{p}_{i,m}-1)^2 \right)$$


* **Penalidade de Estrutura do Tour ($H_{\text{tour}}$):** Utiliza-se a reformulação de penalidade de igualdade $V_E(z) = \lambda \vert{}\vert{}g(z)\vert{}\vert{}^2$ proposta no artigo para impor a restrição de que cada cidade seja visitada uma vez e cada passo contenha apenas uma cidade:

$$H_{\text{tour}} = \sum_{m} \left( \sum_i \hat{x}_{i,m} - 1 \right)^2 + \sum_{m} \left( \sum_i \hat{p}_{i,m} - 1 \right)^2 + \sum_i \left( \sum_m (\hat{x}_{i,m} + \hat{p}_{i,m}) - 1 \right)^2$$


* **Extensão VRP: Capacidade dos Veículos ($H_{\text{cap}}$):** Para o VRP com $K$ veículos de capacidade $C_k$ e demandas $d_i$, codifica-se a restrição de desigualdade via função retificadora suave Swish $V_I(z) = \sum R(\lambda h_j(z))$ descrita por Madani et al., onde $R(u) = u \cdot \sigma(u)$ e $\sigma(u) = (1 + e^{-u})^{-1}$:

$$H_{\text{cap}} = \sum_{k=1}^K R\left(\lambda_{\text{cap}} \left( \sum_i d_i \hat{x}_{i,k}^{\text{veh}} - C_k \right)\right)$$



Esta proposta reduz a dimensão do espaço de Hilbert simulado de $\mathcal{O}(D^{N^2})$ para $\mathcal{O}(D^{N^2/2})$ através do empacotamento complexo, garantindo a evolução unitária Hermitiana necessária para o circuito variacional $e^{-i\gamma H_C}$.