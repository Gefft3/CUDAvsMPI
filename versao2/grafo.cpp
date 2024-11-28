#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <mpi.h>
#include <algorithm>
#include <queue>

using namespace std;

class Graph {
public:
    vector<set<int>> adjList;
    set<int> vertices;

    // Function to read the graph from a file
    void readGraph(const string& filename) {
        ifstream inputFile(filename);
        if (!inputFile.is_open()) {
            cerr << "Unable to open file: " << filename << endl;
            exit(EXIT_FAILURE);
        }

        map<int, int> vertexMapping;
        int vertexCounter = 0;
        int u, v;
        vector<pair<int, int>> edges;

        while (inputFile >> u >> v) {
            if (vertexMapping.find(u) == vertexMapping.end()) {
                vertexMapping[u] = vertexCounter++;
                vertices.insert(vertexMapping[u]);
            }
            if (vertexMapping.find(v) == vertexMapping.end()) {
                vertexMapping[v] = vertexCounter++;
                vertices.insert(vertexMapping[v]);
            }
            u = vertexMapping[u];
            v = vertexMapping[v];
            edges.emplace_back(u, v);
        }

        adjList.resize(vertexCounter);

        for (auto& edge : edges) {
            adjList[edge.first].insert(edge.second);
            adjList[edge.second].insert(edge.first);
        }

        inputFile.close();
    }

    // Function to get the neighbors of a vertex
    vector<int> getNeighbours(int vertex) {
        return vector<int>(adjList[vertex].begin(), adjList[vertex].end());
    }

    // Function to check if a vertex is in the clique
    bool esta_na_clique(int vertex, vector<int>& clique) {
        return find(clique.begin(), clique.end(), vertex) != clique.end();
    }

    // Function to check if a vertex connects to all vertices in the clique
    bool se_conecta_a_todos_os_vertices_da_clique(int vertex, vector<int>& clique) {
        for (int v : clique) {
            if (adjList[vertex].find(v) == adjList[vertex].end()) {
                return false;
            }
        }
        return true;
    }

    // Function to determine if a vertex can form a clique with the current clique
    bool formar_clique(int vertex, vector<int>& clique) {
        return se_conecta_a_todos_os_vertices_da_clique(vertex, clique) && !esta_na_clique(vertex, clique);
    }

    // Function to perform the parallel clique counting
    int contagem_cliques_parallel(int k, int rank, int size);
};

int Graph::contagem_cliques_parallel(int k, int rank, int size) {
    int clique_count = 0;

    if (rank == 0) {
        // Manager process
        queue<int> vertexQueue;
        for (int v : vertices) {
            vertexQueue.push(v);
        }

        int numWorkers = size - 1;
        int activeWorkers = numWorkers;

        // Send initial vertices to workers
        for (int i = 1; i <= numWorkers; ++i) {
            if (!vertexQueue.empty()) {
                int vertex = vertexQueue.front();
                vertexQueue.pop();
                MPI_Send(&vertex, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            } else {
                // No more vertices to process
                int stopSignal = -1;
                MPI_Send(&stopSignal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                --activeWorkers;
            }
        }

        // Receive results and send new vertices
        while (activeWorkers > 0) {
            int workerRank;
            MPI_Status status;
            int localCliqueCount;

            // Receive local clique count from any worker
            MPI_Recv(&localCliqueCount, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
            workerRank = status.MPI_SOURCE;
            clique_count += localCliqueCount;

            if (!vertexQueue.empty()) {
                int vertex = vertexQueue.front();
                vertexQueue.pop();
                MPI_Send(&vertex, 1, MPI_INT, workerRank, 0, MPI_COMM_WORLD);
            } else {
                int stopSignal = -1;
                MPI_Send(&stopSignal, 1, MPI_INT, workerRank, 0, MPI_COMM_WORLD);
                --activeWorkers;
            }
        }

        // Now, the manager has collected results from all workers
        cout << "Resultado final: " << clique_count << endl;

    } else {
        // Worker process

        // Receive the adjacency list from the manager
        int numVertices;
        if (adjList.empty()) {
            // Send request to manager to get the adjacency list
            MPI_Bcast(&numVertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
            adjList.resize(numVertices);
            for (int i = 0; i < numVertices; ++i) {
                int numNeighbors;
                MPI_Bcast(&numNeighbors, 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (numNeighbors > 0) {
                    vector<int> neighbors(numNeighbors);
                    MPI_Bcast(&neighbors[0], numNeighbors, MPI_INT, 0, MPI_COMM_WORLD);
                    adjList[i].insert(neighbors.begin(), neighbors.end());
                }
            }
        }

        while (true) {
            // Receive a vertex to process
            int vertex;
            MPI_Recv(&vertex, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (vertex == -1) {
                // No more vertices to process
                break;
            }

            // Perform clique counting starting from the received vertex
            set<vector<int>> local_cliques;
            local_cliques.insert({vertex});
            int localCliqueCount = 0;

            while (!local_cliques.empty()) {
                vector<int> clique = *local_cliques.begin();
                local_cliques.erase(local_cliques.begin());

                if ((int)clique.size() == k) {
                    localCliqueCount++;
                    continue;
                }

                // Get the intersection of neighbors of all vertices in the clique
                set<int> candidateVertices(adjList[clique[0]]);
                for (size_t i = 1; i < clique.size(); ++i) {
                    set<int> tempSet;
                    set_intersection(candidateVertices.begin(), candidateVertices.end(),
                                     adjList[clique[i]].begin(), adjList[clique[i]].end(),
                                     inserter(tempSet, tempSet.begin()));
                    candidateVertices = tempSet;
                }

                // Consider only vertices with IDs greater than the last vertex in the clique to avoid duplicates
                int lastVertex = clique.back();
                for (int neighbor : candidateVertices) {
                    if (neighbor > lastVertex) {
                        vector<int> newClique = clique;
                        newClique.push_back(neighbor);
                        local_cliques.insert(newClique);
                    }
                }
            }

            // Send the local clique count back to the manager
            MPI_Send(&localCliqueCount, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    }

    return clique_count;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <dataset_file> <k_clique>" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    string datasetFile = argv[1];
    int k_clique = atoi(argv[2]);

    Graph g;

    if (rank == 0) {
        // Manager reads the graph from the dataset
        g.readGraph(datasetFile);

        // Broadcast the adjacency list to workers
        int numVertices = g.adjList.size();
        MPI_Bcast(&numVertices, 1, MPI_INT, 0, MPI_COMM_WORLD);

        for (int i = 0; i < numVertices; ++i) {
            int numNeighbors = g.adjList[i].size();
            MPI_Bcast(&numNeighbors, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (numNeighbors > 0) {
                vector<int> neighbors(g.adjList[i].begin(), g.adjList[i].end());
                MPI_Bcast(&neighbors[0], numNeighbors, MPI_INT, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        // Workers receive the adjacency list
        int numVertices;
        MPI_Bcast(&numVertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
        g.adjList.resize(numVertices);

        for (int i = 0; i < numVertices; ++i) {
            int numNeighbors;
            MPI_Bcast(&numNeighbors, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (numNeighbors > 0) {
                vector<int> neighbors(numNeighbors);
                MPI_Bcast(&neighbors[0], numNeighbors, MPI_INT, 0, MPI_COMM_WORLD);
                g.adjList[i].insert(neighbors.begin(), neighbors.end());
            }
        }
    }

    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    g.contagem_cliques_parallel(k_clique, rank, size);

    // End timing
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        cout << "Execution time: " << end_time - start_time << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}