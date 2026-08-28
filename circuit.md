# Principais Alterações no Arquivo `circuit.py`
## Método initialize_warm_start_params:

Recebe a estrutura clássica de rotas target_routes (ex.: {1: [0, 1, 0], 2: [0, 2, 3, 0]}) e converte diretamente a ordem x das cidades e o identificador do veículo p para as magnitudes (r) e fases (ϕ) da porta de deslocamento (Dgate) na primeira camada do circuito.  

Garante a quebra explícita de simetria nos qumodes para que os veículos não convirjam para o mesmo estado nulo ou idêntico.  
JSON

## Método inject_noise:

Fornece suporte direto ao mecanismo de escape de platô (Plateau Escape) estipulado no plano DOE, adicionando perturbações gaussianas reguladas aos parâmetros variacionais quando a estagnação do gradiente for identificada.

Próximo Passo
Envie o código fonte de qumodes/solver.py e/ou main.py para realizarmos a refatoração do loop de otimização, suporte aos otimizadores SPSA/CMA-ES/Adam, dinâmicas de callback (PlateauDetector / PenaltyScheduler) e orquestração do experimento.


---
---

# Principais Alterações no Módulo Solver
- Integração de **Warm-Start**: O método solve aceita a estrutura de rotas clássica via **warm_start_routes**. Caso seja passada, chama `ansatz.initialize_warm_start_params` para instanciar o circuito próximo do ponto viável.  

- Detector de Platô e Injeção de Ruído: Monitora variações $∣\Delta C∣<10^{−4}$ durante plateau_patience iterações consecutivas. Ao atingir o limite, aciona a perturbação gaussiana inject_noise nos pesos variacionais.  

- Algoritmo `SPSA` Integrado: Suporte para alternar entre otimização baseada em gradiente analítico com clipping (`ADAM`) e aproximação estocástica sem gradientes explícitos (`SPSA`).  

- Annealing Dinâmico de Penalidades: Atualização contínua de $\lambda(k)$ via parâmetro `penalty_gamma` em cada passo de otimização.  

- Retorno Enriquecido de Métricas: Preserva retrocompatibilidade completa com as chaves `cost_history`, `routes`, `opt_params`, e adiciona os campos `is_feasible`, `route_distance` e `composite_score`.  


---
---
---

# Síntese da Refatoração Completa do Projeto
Todos os quatro módulos da pilha CV-VQE foram integrados:  

- `qumodes/hamiltonian.py`: Gerencia multiplicadores dinâmicos λ via tf.Variable, executa o cálculo da viabilidade física is_feasible e provê a métrica de composite_score.  

- `qumodes/circuit.py`: Suporta injeção de estado inicial clássico via initialize_warm_start_params e escape estocástico via inject_noise.  

- `qumodes/solver.py`: Suporta os otimizadores ADAM e SPSA, scheduler de penalidades (Annealing), detecção de platô e extração completa de métricas sanitizadas.  

- `main.py`: Controla os benchmarks configuráveis para avaliação das estratégias de inicialização (Warm-Start vs Cold-Start) e exportação dos resultados com os novos indicadores.  