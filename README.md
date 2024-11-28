# CUDA vs OpenMPI

CUDA e OpenMPI são tecnologias para computação paralela, mas com enfoques distintos. CUDA utiliza GPUs para paralelismo massivo em tarefas locais, ideal para cálculos intensivos. OpenMPI distribui tarefas entre múltiplos nós via rede, adequado para problemas escaláveis com alta comunicação.

O problema atual é a contagem eficiente de cliques de tamanho arbitrário 
𝑘 em um grafo não direcionado. Isso envolve identificar subgrafos completos de 
𝑘 vértices. O desafio está em balancear desempenho e memória ao paralelizar o cálculo em CUDA e lidar com sincronização eficiente.
