#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Simple linear search for k-nearest neighbors
std::vector<std::pair<float, int>> linearKNN(const std::vector<std::vector<float>>& data, 
                                            const std::vector<float>& query, 
                                            int k) {
    std::vector<std::pair<float, int>> distances;
    
    for (size_t i = 0; i < data.size(); ++i) {
        float dist = 0.0;
        for (size_t j = 0; j < query.size(); ++j) {
            float diff = query[j] - data[i][j];
            dist += diff * diff;
        }
        distances.push_back({std::sqrt(dist), static_cast<int>(i)});
    }
    
    std::sort(distances.begin(), distances.end());
    distances.resize(std::min(k, static_cast<int>(distances.size())));
    
    return distances;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <query.json> <data.json> <K>\n";
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Load query
    std::ifstream query_file(argv[1]);
    json query_json;
    query_file >> query_json;
    
    // Load data
    std::ifstream data_file(argv[2]);
    json data_json;
    data_file >> data_json;
    
    int K = std::stoi(argv[3]);
    
    // Extract query embedding
    std::vector<float> query = query_json[0]["embedding"].get<std::vector<float>>();
    
    // Extract data embeddings
    std::vector<std::vector<float>> data;
    for (const auto& item : data_json) {
        data.push_back(item["embedding"].get<std::vector<float>>());
    }
    
    // Find k-nearest neighbors
    auto neighbors = linearKNN(data, query, K);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    
    // Output results
    std::cout << "Query: " << query_json[0]["text"] << std::endl;
    std::cout << "Nearest neighbors:" << std::endl;
    
    for (size_t i = 0; i < neighbors.size(); ++i) {
        int idx = neighbors[i].second;
        float dist = neighbors[i].first;
        std::cout << "  " << (i+1) << ". ID: " << data_json[idx]["id"] 
                  << ", Distance: " << dist 
                  << ", Text: " << data_json[idx]["text"] << std::endl;
    }
    
    std::cout << "Execution time: " << duration << " ms" << std::endl;
    
    return 0;
}
