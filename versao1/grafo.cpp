#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
#include <thread>
#include <map>
#include <vector>
#include "sort.h"
#include <algorithm>
#include <set>

using namespace std;
using namespace std::chrono;

class Graph {
    public:

        vector<pair<int, int>> arestas;
        vector<set<int>> vizinhos; 
        set<int> vertices; 

        Graph(vector<pair<int, int>> edgeList) {
            for (auto edge : edgeList) {
                const int tamanho_vizinhos = vizinhos.size();
                if (tamanho_vizinhos <= max(edge.first, edge.second)) {
                    vizinhos.resize(max(edge.first, edge.second) + 1);
                }
                arestas.push_back(edge);
                vertices.insert(edge.first); 
                vertices.insert(edge.second); 
                vizinhos[edge.first].insert(edge.second); 
                vizinhos[edge.second].insert(edge.first); 
            }
        }

        int contagem_cliques_serial(int k);
        bool esta_na_clique(int vertex, vector<int> clique);
        bool se_conecta_a_todos_os_vertices_da_clique(int vertex, vector<int> clique);
        bool formar_clique(int vertex, vector<int> clique);

        vector<int> getNeighbours(int vertex) {
            vector<int> neighbours;
            for (int neighbour : vizinhos[vertex]) {
                neighbours.push_back(neighbour);
            }
            return neighbours;
        }

        bool isNeighbour(int vertex, int neighbour) {
            return vizinhos[vertex].find(neighbour) != vizinhos[vertex].end();
        }

        void printar_grafo() {
            for (auto v : vertices) {
                cout << v << ": ";
                for (auto n : vizinhos[v]) {
                    cout << n << " ";
                }
                cout << endl;
            }
        }

        void printar_clique(vector<int> clique) {
            for (auto v : clique) {
                cout << v << " ";
            }
        }

        void release() {
            arestas.clear();
            vertices.clear();
            vizinhos.clear();
        }
};


vector<pair<int, int>> rename(const string& dataset) {
    
    ifstream inputFile(dataset);
    map<int, int> nodeMap;
    vector<pair<int, int>> edges;
    int nodeCounter = 0;
    
    if (!inputFile.is_open()) {
        cerr << "nao abriu, arquivo: " << dataset << endl;
    }

    int u, v;

    while (inputFile >> u >> v) {
        if (nodeMap.find(u) == nodeMap.end()) {
            nodeMap[u] = nodeCounter++;
        }
        if (nodeMap.find(v) == nodeMap.end()) {
            nodeMap[v] = nodeCounter++;
        }
        edges.push_back({nodeMap[u], nodeMap[v]});
    }

    inputFile.close();

    return edges;
}

bool Graph::esta_na_clique(int vertex, vector<int> clique) {
    for (size_t i = 0; i < clique.size(); i++) {
        if (vertex == clique[i]) {
            return true;
        }
    }
    return false;
}

bool Graph::se_conecta_a_todos_os_vertices_da_clique(int vertex, vector<int> clique) {
    for (int v : clique) {
        if (!isNeighbour(vertex, v)) {
            return false;
        }
    }
    return true;
}

bool Graph::formar_clique(int vertex, vector<int> clique) {
    bool cond1 = se_conecta_a_todos_os_vertices_da_clique(vertex, clique);
    bool cond2 = esta_na_clique(vertex, clique);

    //cout << "vertice vizinho: " << vertex << endl;
    //cout << "se conecta a todos (tem que): " << cond1 << endl;
    //cout << "já está na clique (não pode): " << cond2 << endl;

    bool condf = (cond1 && !cond2); 
    //cout << "decisão: " << condf << endl;
    return condf;
}



int Graph::contagem_cliques_serial(int k) {
    set<vector<int>> cliques;

    

    for(auto v: vertices) {
        cliques.insert({v});
    }

    int count = 0;
    // int iteracoes = 0;
    while(!cliques.empty()){
        //cout << "-----------------------------------" << endl;
        //cout << "interação - " << ++iteracoes << endl;
        
        vector<int> clique = *cliques.cbegin();
        //cout << "Size of cliques before pop: " << cliques.size() << endl;
        
        cliques.erase(find(cliques.begin(), cliques.end(), clique));
        //cout << "Size of cliques after pop: " << cliques.size() << endl;

        
        //cout << "Clique atual: ";
        // printar_clique(clique);
        //cout << endl;
        int tamanho_clique = clique.size();
        if(tamanho_clique == k){
            //cout << "Clique encontrada: ";
            count++;
            continue;
        }
        
        int ultimo_vertice = clique.back();
        
        //cout << "Ultimo vertice: " << ultimo_vertice << endl;

        for(int vertice : clique){
            vector<int> vizinhos_atual = getNeighbours(vertice); 
            //cout << "Vizinhos do vertice " << vertice << ": ";
            // for(auto v: vizinhos_atual){
                //cout << v << " ";
            // }
            //cout << endl;
            
            
            for(int vizinho: vizinhos_atual){
                if(vizinho > ultimo_vertice && formar_clique(vizinho, clique)){
                    //cout << "ENTROU!!! " << endl;
                    vector<int> nova_clique = clique;
                    nova_clique.push_back(vizinho);
                    cliques.insert(nova_clique);
                }
            }
        }

        //cout << "voltou para cima" << endl;

    }
    
    return count;
}

int main(int argc, char* argv[]) {

    string dataset = argv[1];
    int k_clique = atoi(argv[2]);

    vector<pair<int, int>> edges = rename(dataset);
    Graph* g = new Graph(edges);
    

    auto start = high_resolution_clock::now();
    int clique_count = g->contagem_cliques_serial(k_clique);
    auto end = chrono::high_resolution_clock::now();
    duration<double> duration = end - start;
    
    cout << "Resultado: " << clique_count << endl;
    cout << "Tempo de execução: " << duration.count() << " segundos" << endl;
    
    g->release();
    delete g;
}