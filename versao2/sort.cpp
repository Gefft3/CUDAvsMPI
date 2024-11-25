#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include "sort.h"

using namespace std;

vector<pair<int, int>> sort(const string& inputFileName) {
    
    ifstream inputFile(inputFileName);
    map<int, int> nodeMap;
    vector<pair<int, int>> edges;
    int nodeCounter = 0;
    
    if (!inputFile.is_open()) {
        cerr << "Erro ao abrir o arquivo de entrada: " << inputFileName << endl;
    }

    int u, v;
    
    // Ler o grafo do arquivo de entrada
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


    cout << "O grafo foi renomeado com sucesso!" << endl;

    return edges;
}
