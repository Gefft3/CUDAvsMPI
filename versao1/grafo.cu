#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <chrono>

using namespace std;
using namespace chrono;

class Graph {
public:
    vector<pair<int, int>> arestas;
    vector<set<int>> vizinhos;
    set<int> vertices;

    Graph(vector<pair<int, int>> edgeList) {
        for (auto edge : edgeList) {
            int maxVertex = max(edge.first, edge.second);
            if (vizinhos.size() <= maxVertex) {
                vizinhos.resize(maxVertex + 1);
            }
            arestas.push_back(edge);
            vertices.insert(edge.first);
            vertices.insert(edge.second);
            vizinhos[edge.first].insert(edge.second);
            vizinhos[edge.second].insert(edge.first);
        }
    }

    vector<int> getNeighbours(int vertex) {
        vector<int> neighbours(vizinhos[vertex].begin(), vizinhos[vertex].end());
        return neighbours;
    }

    void release() {
        arestas.clear();
        vertices.clear();
        vizinhos.clear();
    }
};

// Função para carregar dataset e renomear vértices
vector<pair<int, int>> rename(const string& dataset) {
    ifstream inputFile(dataset);
    map<int, int> nodeMap;
    vector<pair<int, int>> edges;
    int nodeCounter = 0;

    if (!inputFile.is_open()) {
        cerr << "Erro ao abrir arquivo: " << dataset << endl;
        exit(1);
    }

    int u, v;
    while (inputFile >> u >> v) {
        if (nodeMap.find(u) == nodeMap.end()) {
            nodeMap[u] = nodeCounter++;
        }
        if (nodeMap.find(v) == nodeMap.end()) {
            nodeMap[v] = nodeCounter++;
        }
        edges.emplace_back(nodeMap[u], nodeMap[v]);
    }
    inputFile.close();
    return edges;
}

// Kernel CUDA para contagem de cliques
__global__ void contagem_cliques_kernel(int* d_vizinhos, int* d_offsets, int numVertices, int k, int* d_cliqueCount, int* d_cliqueBuffer, int* d_workQueue, int* d_workQueueIndex) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int queueIndex = atomicAdd(d_workQueueIndex, 1);

    // Se não houver mais trabalho na fila, a thread deve retornar
    if (queueIndex >= numVertices) return;

    if (tid < numVertices){

    // Consumir um vértice da fila global de trabalho
    int vertex = d_workQueue[queueIndex];

    // Inicializar o buffer de clique
    int cliqueStart = 0;
    d_cliqueBuffer[cliqueStart] = vertex;
    int cliqueSize = 1;

    // Contagem de cliques de tamanho k
    for (int cliqueLevel = 1; cliqueLevel < k; cliqueLevel++) {
        int lastVertex = d_cliqueBuffer[cliqueStart + cliqueSize - 1];
        int start = d_offsets[lastVertex];
        int end = d_offsets[lastVertex + 1];
        bool expanded = false;

        for (int i = start; i < end; i++) {
            int vizinho = d_vizinhos[i];
            bool isClique = true;

            // Verifique conectividade com todos os vértices no clique atual
            for (int j = 0; j < cliqueSize; j++) {
                int cliqueVertex = d_cliqueBuffer[cliqueStart + j];
                int neighborStart = d_offsets[cliqueVertex];
                int neighborEnd = d_offsets[cliqueVertex + 1];
                bool found = false;

                for (int n = neighborStart; n < neighborEnd; n++) {
                    if (d_vizinhos[n] == vizinho) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    isClique = false;
                    break;
                }
            }

            if (isClique) {
                d_cliqueBuffer[cliqueStart + cliqueSize] = vizinho;
                cliqueSize++;
                expanded = true;

                if (cliqueSize == k) {
                    atomicAdd(d_cliqueCount, 1);
                    break;
                }
            }
        }
        if (!expanded) break;
    }
}
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Uso: " << argv[0] << " <dataset> <k-clique>" << endl;
        return 1;
    }

    string dataset = argv[1];
    int k_clique = atoi(argv[2]);

    // Carregar o grafo
    vector<pair<int, int>> edges = rename(dataset);
    Graph g(edges);
    int numVertices = g.vertices.size();

    // Preparar dados para GPU
    vector<int> vizinhosFlat;
    vector<int> offsets(numVertices + 1, 0);
    for (int v = 0; v < numVertices; v++) {
        vector<int> vizinhos = g.getNeighbours(v);
        offsets[v + 1] = offsets[v] + vizinhos.size();
        vizinhosFlat.insert(vizinhosFlat.end(), vizinhos.begin(), vizinhos.end());
    }

    // Fila global de trabalho com todos os vértices
    vector<int> workQueue(numVertices);
    for (int i = 0; i < numVertices; i++) {
        workQueue[i] = i;
    }

    // Alocar memória na GPU
    int* d_vizinhos, *d_offsets, *d_cliqueCount, *d_cliqueBuffer, *d_workQueue, *d_workQueueIndex;
    int cliqueBufferSize = numVertices * k_clique;
    cudaMalloc(&d_vizinhos, vizinhosFlat.size() * sizeof(int));
    cudaMalloc(&d_offsets, offsets.size() * sizeof(int));
    cudaMalloc(&d_cliqueCount, sizeof(int));
    cudaMalloc(&d_cliqueBuffer, cliqueBufferSize * sizeof(int));
    cudaMalloc(&d_workQueue, workQueue.size() * sizeof(int));
    cudaMalloc(&d_workQueueIndex, sizeof(int));

    // Copiar dados para a GPU
    cudaMemcpy(d_vizinhos, vizinhosFlat.data(), vizinhosFlat.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_workQueue, workQueue.data(), workQueue.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_cliqueCount, 0, sizeof(int));
    cudaMemset(d_workQueueIndex, 0, sizeof(int));

    // Configurar kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numVertices + threadsPerBlock - 1) / threadsPerBlock;

    auto start = high_resolution_clock::now();
    contagem_cliques_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_vizinhos, d_offsets, numVertices, k_clique, d_cliqueCount, d_cliqueBuffer, d_workQueue, d_workQueueIndex);
    cudaDeviceSynchronize();

    // Verificar erros
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Erro no kernel: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    auto end = high_resolution_clock::now();

    // Recuperar resultado
    int cliqueCount;
    cudaMemcpy(&cliqueCount, d_cliqueCount, sizeof(int), cudaMemcpyDeviceToHost);

    // Exibir resultados
    duration<double> duration = end - start;
    cout << "Número de cliques: " << cliqueCount << endl;
    cout << "Tempo de execução: " << duration.count() << " segundos" << endl;

    // Liberar memória
    cudaFree(d_vizinhos);
    cudaFree(d_offsets);
    cudaFree(d_cliqueCount);
    cudaFree(d_cliqueBuffer);
    cudaFree(d_workQueue);
    cudaFree(d_workQueueIndex);

    g.release();
    return 0;
}
