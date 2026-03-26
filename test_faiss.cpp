/*
 * Simple test to verify FAISS is working correctly
 * Compile: g++ -std=c++17 -I/usr/local/include test_faiss.cpp -L/usr/local/lib -lfaiss -o test_faiss
 * Run: ./test_faiss
 */

#include <faiss/IndexFlat.h>
#include <iostream>
#include <vector>

int main() {
    // Dimension of vectors
    int d = 64;
    
    // Create a flat index (L2 distance)
    faiss::IndexFlatL2 index(d);
    
    std::cout << "FAISS Test" << std::endl;
    std::cout << "==========" << std::endl;
    std::cout << "Index dimension: " << d << std::endl;
    std::cout << "Index is trained: " << (index.is_trained ? "Yes" : "No") << std::endl;
    std::cout << "Index total vectors: " << index.ntotal << std::endl;
    
    // Add some random vectors
    int nb = 100;
    std::vector<float> database(nb * d);
    for (int i = 0; i < nb * d; i++) {
        database[i] = (float)rand() / RAND_MAX;
    }
    
    index.add(nb, database.data());
    
    std::cout << "Added " << nb << " vectors" << std::endl;
    std::cout << "Index total vectors: " << index.ntotal << std::endl;
    
    // Search
    int nq = 5;  // number of queries
    std::vector<float> queries(nq * d);
    for (int i = 0; i < nq * d; i++) {
        queries[i] = (float)rand() / RAND_MAX;
    }
    
    int k = 4;  // number of nearest neighbors
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> indices(nq * k);
    
    index.search(nq, queries.data(), k, distances.data(), indices.data());
    
    std::cout << "\nSearch Results:" << std::endl;
    for (int i = 0; i < nq; i++) {
        std::cout << "Query " << i << " nearest neighbors: ";
        for (int j = 0; j < k; j++) {
            std::cout << indices[i * k + j] << " (dist=" << distances[i * k + j] << ") ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n✓ FAISS is working correctly!" << std::endl;
    return 0;
}

