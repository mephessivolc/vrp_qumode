
# Hamiltoniano utilizado

A modelagem descrita propõe uma formulação em variáveis contínuas (CV - *Continuous Variable*) para o Problema de Roteamento de Veículos (VRP) utilizando $N$ *qumodes* quânticos para representar $N$ cidades.

---

**1. Codificação do Espaço de Rotas (*Route-Slot Encoding*)**

* **Compactação de Variáveis:** Em abordagens binárias tradicionais com *qubits*, a representação de $N$ cidades, $M$ veículos e $N$ posições requer $M N^2$ *qubits*. A abordagem CV associa apenas $1$ *qumode* por cidade ($N$ *qumodes* no total).


* **Achateamento do Espaço de Posições:** Cada par slot-veículo $(v, r)$ — onde $v \in \{1, \dots, M\}$ é o veículo e $r \in \{1, \dots, N\}$ é a posição na rota — é mapeado para um único índice unidimensional $s(v, r) = (v - 1)N + r \in \{1, \dots, MN\}$.


* **Mapeamento Espectral Afim:** O operador de quadratura de posição $\hat{x}_i$ de cada *qumode* tem seu espectro discreto truncado (com corte de Fock $d$) projetado afimmente no intervalo contínuo de coordenadas de rota $[1, MN]$ gerando o operador $\hat{z}_i$.


* **Atribuição Suave (*Soft Assignment*):** Como a coordenada $z_i$ é contínua, utiliza-se uma função suave normalizada (estilo Softmax) $a_{i,v,r}$ para calcular a probabilidade/massa de uma cidade $i$ pertencer ao slot de rota $(v, r)$.


* **Ocupação do Slot ($n_{v,r}$):** A soma das atribuições $n_{v,r} = \sum_{i} a_{i,v,r}$ quantifica a quantidade de cidades alocadas na posição $r$ do veículo $v$.



---

**2. Componentes do Hamiltoniano de Custo ($H_{\text{VRP}}$)**

O Hamiltoniano de custo total é a soma de quatro termos restritivos e objetivos:

$$H_{\text{VRP}} = H_{\text{dist}} + H_{\text{col}} + H_{\text{gap}} + H_{\text{vehicle}}$$

* **Custo de Distância ($H_{\text{dist}}$):** Soma a distância da partida do depósito à primeira cidade ($H_{\text{start}}$), as transições consecutivas entre cidades na mesma rota ($H_{\text{internal}}$) e o retorno ao depósito ($H_{\text{return}}$), utilizando uma função de atenuação de ocupação $e_{v,r+1}$ para identificar dinamicamente a última cidade visitada antes do depósito.


* **Penalidade de Colisão ($H_{\text{col}}$):** Penaliza configurações onde duas cidades diferentes ocupam a mesma posição no mesmo veículo ($a_{i,v,r} a_{j,v,r}$).


* **Penalidade de Lacunas ($H_{\text{gap}}$):** Garante o empacotamento à esquerda (*left-packing*), exigindo que a posição $r$ só seja ocupada caso todas as posições anteriores $1, \dots, r-1$ do veículo também estejam ocupadas.


* **Uso de Veículos ($H_{\text{vehicle}}$):** Se $\lambda_{\text{vehicle}} > 0$, força a alocação da primeira posição de todos os $M$ veículos; se $\lambda_{\text{vehicle}} = 0$, permite o uso flexível de até $M$ veículos.



---

**3. Decodificação e Validação da Solução**

Após a otimização, os autovalores contínuos otimizados $z_i^*$ são convertidos para rotas discretas:

1. Arredonda-se a coordenada contínua para o slot inteiro mais próximo $s_i^* = \text{round}(z_i^*)$.


2. O veículo é extraído por $v_i = \lfloor (s_i^* - 1) / N \rfloor + 1$ e a posição na rota por $r_i = ((s_i^* - 1) \bmod N) + 1$.


3. As cidades de cada veículo são ordenadas de acordo com suas posições $r_i$.


4. A solução é válida se cada cidade aparecer exatamente uma vez, sem colisões de slot e sem lacunas na sequência.



---

**4. Dinâmica Quântica e Misturadores (Mixers)**

* **Diagonalidade do Custo:** Como todos os termos de $H_{\text{VRP}}$ dependem apenas de funções das posições $\hat{x}_i$, e como os operadores de posição de modos distintos comutam ($[\hat{x}_i, \hat{x}_j] = 0$), $H_{\text{VRP}}$ é diagonal na base de autovalores de posição e sozinho não produz vantagem quântica.


* **Dinamismo Não Comutativo:** Para induzir tunelamento e interferência quântica, introduz-se um Hamiltoniano misturador não comutativo $H_{\text{mix}}$ que utiliza o momento conjugado $\hat{p}_i$ (como $H_{\text{mix}} = \sum_i \hat{p}_i^2$), gerando a dinâmica do QAOA quântico em variáveis contínuas por alternância de camadas $e^{-i \beta_l H_{\text{mix}}} e^{-i \gamma_l H_{\text{VRP}}}$.


* **Mixers Conscientes da Rota (*Route-Aware*):** São propostos operadores misturadores mais estruturados que respeitam a topologia do problema:


* *Intra-rota:* Promove transições suaves entre posições adjacentes do mesmo veículo.


* *Inter-veículo:* Facilita a transferência de uma cidade para outro veículo mantendo a posição.


* *Correlated Swap:* Troca as posições de duas cidades simultaneamente, mantendo o setor viável sem gerar colisões.


* *Potencial de Tunelamento ($V_{\text{route}}$):* Modula as barreiras de potencial contínuas para facilitar trocas de rota com relevância física.

# Formulação Hinge Polinomial

A formulação por **Hinge Polinomial** (ou aproximação polinomial quártica) é a mais indicada para simulações no **Strawberry Fields**, pois mapeia a restrição de capacidade diretamente para o conjunto universal de portas do simulador sem adicionar *qumodes* e sem exigir compiladores de funções transcendentais em tempo de execução.

* **Mapeamento Direto para Portas Nativas:** O Strawberry Fields possui suporte nativo a geradores de base polinomial, tais como as portas de Squeezing/Rotação para termos quadráticos ($\hat{x}^2$), a porta de **Fase Cúbica** ($V(\gamma) = e^{i \gamma \hat{x}^3}$) para termos cúbicos e a porta **Kerr** ($K(\kappa) = e^{i \kappa \hat{n}^2} \propto e^{i \kappa \hat{x}^4}$) para termos quárticos. Um Hinge polinomial do tipo $\alpha_2 (L_v - Q_v)^2 + \alpha_4 (L_v - Q_v)^4$ converte-se diretamente nessas sequências de portas.


* **Controle Estável do Corte de Fock ($D$):** Se a função Swish fosse expandida via série de Taylor diretamente no simulador, a presença de operadores de ordens elevadas ($\hat{x}^6, \hat{x}^8, \dots$) exigiria um $D$ excessivamente alto para evitar erros de truncamento em estados com alto número de fótons. O Hinge polinomial fixa o operador em grau máximo 4, permitindo obter convergência física com cortes moderados ($D \approx 10 - 15$).


* **Alinhamento com o Framework do Artigo:** O artigo demonstra a otimização de funções não-convexas no Strawberry Fields (como a função Styblinski-Tang e o benchmark quártico complexo) utilizando exatamente essa estratégia: a aproximação da paisagem por termos quárticos resolvidos via *backend* Fock e portas Kerr.


* **Preservação de Recursos Computacionais:** A abordagem elimina os $M$ *qumodes* extras que seriam exigidos pela reformulação com variáveis de folga (*slack*), mantendo a dimensão do espaço de Hilbert limitada a $\mathcal{O}(D^N)$ em vez de sofrer uma explosão para $\mathcal{O}(D^{N+M})$.

---

# Implementação 

O Hamiltoniano para o Problema de Roteamento de Veículos Capulados em Variáveis Contínuas (CV-CVRP) é refatorado pela substituição do termo $H_{\text{vehicle}}$ pelo operador de capacidade Hinge Polinomial $H_{\text{capacity}}$:

$$H_{\text{CV-CVRP}} = H_{\text{dist}} + H_{\text{col}} + H_{\text{gap}} + H_{\text{capacity}}$$

---

**1. Remoção do Termo $H_{\text{vehicle}}$**

A eliminação de $H_{\text{vehicle}}$ remove a restrição que forçava a alocação obrigatória da primeira posição de todos os $M$ veículos. Sem esse termo, o algoritmo ajusta livremente a quantidade de veículos utilizados, determinando o número de rotas ativas de forma dinâmica a partir da demanda total $D_{\text{total}} = \sum_{i=1}^N d_i$ e da capacidade $Q_v$.

---

**2. Operador de Carga Acumulada ($L_v(\hat{x})$)**

A carga total alocada ao veículo $v$ é calculada somando a demanda $d_i$ de cada cidade $i$ multiplicada pela sua probabilidade de atribuição $a_{i,v,r}(\hat{x})$ ao longo de todas as posições $r$ do veículo:

$$L_v(\hat{x}) = \sum_{i=1}^N \sum_{r=1}^N d_i \, a_{i,v,r}(\hat{x})$$

---

**3. Formulação Hinge Polinomial ($H_{\text{capacity}}$)**

O excesso de carga em relação à capacidade do veículo $v$ é medido por $h_v(\hat{x}) = L_v(\hat{x}) - Q_v$. Para evitar o uso de *qumodes* extras e manter a compatibilidade com operadores nativos de ordem superior em simulações contínuas, aplica-se uma barreira quártica par:

$$H_{\text{capacity}} = \lambda_{\text{cap}} \sum_{v=1}^M \left[ \alpha_2 \left( L_v(\hat{x}) - Q_v \right)^2 + \alpha_4 \left( L_v(\hat{x}) - Q_v \right)^4 \right]$$

* **Comportamento da Barreira:** Para $L_v(\hat{x}) \le Q_v$, a contribuição de penalidade permanece baixa. Quando $L_v(\hat{x}) > Q_v$, os coeficientes de escala $\alpha_2 > 0$ e $\alpha_4 > 0$ elevam a energia do estado de forma rápida, impedindo que o otimizador variacional convirja para soluções inviáveis.



---

**4. Mapeamento de Portas no Strawberry Fields**

A expansão algébrica de $H_{\text{capacity}}$ gera termos em potências dos operadores de posição $\hat{x}_i$ que se traduzem no conjunto universal de portas CV:

* **Termos de Grau $\le 2$:** Os componentes da forma $\hat{x}_i$ e $\hat{x}_i \hat{x}_j$ advindos da parte quadrática $\left( L_v(\hat{x}) - Q_v \right)^2$ são executados diretamente via portas Gaussianas: Deslocamento $D(\alpha)$, Squeezing $S(r)$ e Fase Controlada $Z(\kappa)$.


* **Termos de Grau $4$:** Os componentes quárticos $\hat{x}_i^4$ provenientes de $\left( L_v(\hat{x}) - Q_v \right)^4$ são codificados via portas não-Gaussianas **Kerr** ($K(\kappa) = e^{i \kappa \hat{n}^2} \propto e^{i \kappa \hat{x}^4}$) ou aproximações por comutadores encadeados.



---

**5. Manutenção dos Termos de Consistência de Rota**

* **$H_{\text{dist}}$:** Mantém o cálculo das distâncias percorridas pelas rotas ativas (saída do depósito, transições internas entre cidades e retorno ao depósito).


* **$H_{\text{col}}$:** Impede que duas cidades distintas compartilhem o mesmo slot $(v, r)$.


* **$H_{\text{gap}}$:** Assegura o empacotamento contínuo das rotas (*left-packing*), garantindo que um veículo só ocupe a posição $r$ se a posição $r-1$ estiver preenchida.

