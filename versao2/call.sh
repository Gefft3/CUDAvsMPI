#!/bin/bash

# Caminho para o dataset
dataset="../datasets/citeseer.edgelist"

# Tamanho do clique
k_cliques=3

# Número mínimo de processos MPI (2 no mínimo)
num_procs=2

# Executa o programa utilizando 2 processos MPI
mpirun -np $num_procs ./programa $dataset $k_cliques