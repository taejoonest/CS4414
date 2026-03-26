#include "knn.hpp"
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <chrono>

using json = nlohmann::json;

template <typename T>
int runMain(char **argv)
{
    auto program_start = std::chrono::high_resolution_clock::now();
    auto parse_start = std::chrono::high_resolution_clock::now();

    // Load and parse query JSON
    std::ifstream query_ifs(argv[0]);
    if (!query_ifs) {
        std::cerr << "Error opening query file: " << argv[0] << "\n";
        return 1;
    }
    json query_json;
    query_ifs >> query_json;
    if (!query_json.is_array() || query_json.size() < 1) {
        std::cerr << "Query JSON must be an array with at least 1 element\n";
        return 1;
    }

    // Load and parse passages JSON
    std::ifstream passages_ifs(argv[1]);
    if (!passages_ifs) {
        std::cerr << "Error opening passages file: " << argv[1] << "\n";
        return 1;
    }
    json passages_json;
    passages_ifs >> passages_json;
    if (!passages_json.is_array() || passages_json.size() < 1) {
        std::cerr << "Passages JSON must be an array with at least 1 element\n";
        return 1;
    }

    // Convert JSON array to dict id->object (used later for lookup)
    std::unordered_map<int, json> dict;
    for (auto &elem : passages_json) {
        int id = elem["id"].get<int>();
        dict[id] = elem;
    }

    // Parse K
    int K = std::stoi(argv[2]);

   // auto parse_end = std::chrono::high_resolution_clock::now();
   // auto parse_time = std::chrono::duration_cast<std::chrono::microseconds>(parse_end - parse_start).count() / 1000.0;

    // Extract the query embedding from query_json[0]
    auto query_obj = query_json[0];
    T qemb;
    if constexpr (std::is_same_v<T, float>) {
        qemb = query_obj["embedding"].get<float>();
    } else {
        const size_t dim = Embedding_T<T>::Dim();
        if (dim == 0) {
            std::cerr << "Error: Dimension not set properly\n";
            return 1;
        }
        qemb.resize(dim);
        for (size_t i = 0; i < dim; ++i) {
            qemb[i] = query_obj["embedding"][i].get<float>();
        }
    }
    // Query embedding is now passed as parameter to knnSearch

    // Collect all passages into allPoints
    std::vector<std::pair<T, int>> allPoints;
    allPoints.reserve(passages_json.size());
    for (const auto& elem : passages_json) {
        T emb;
        if constexpr (std::is_same_v<T, float>) {
            emb = elem["embedding"].get<float>();
        } else {
            const size_t dim = Embedding_T<T>::Dim();
            if (dim == 0) {
                std::cerr << "Error: Dimension not set properly\n";
                return 1;
            }
            emb.resize(dim);
            for (size_t i = 0; i < dim; ++i) {
                emb[i] = elem["embedding"][i].get<float>();
            }
        }
        int idx = elem["id"].get<int>();
        allPoints.emplace_back(emb, idx);
    }
    auto parse_end = std::chrono::high_resolution_clock::now();
    auto parse_time = std::chrono::duration_cast<std::chrono::microseconds>(parse_end - parse_start).count() / 1000.0;
    // Build balanced KD‐tree
    auto buildtree_start = std::chrono::high_resolution_clock::now();
    Node<T>* root = buildKD(allPoints, 0);
    auto buildtree_end = std::chrono::high_resolution_clock::now();
    auto buildtree_time = std::chrono::duration_cast<std::chrono::microseconds>(buildtree_end - buildtree_start).count() / 1000.0;

    // Perform K‐NN search and collect results
    auto query_start = std::chrono::high_resolution_clock::now();
    Node<T>::queryEmbedding = qemb;  // Set the query embedding for comparison
    MaxHeap heap;
    knnSearch(root, 0, K, heap);
    auto query_end = std::chrono::high_resolution_clock::now();
    auto query_time = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start).count() / 1000.0;

    // Collect and sort ascending by distance
    std::vector<PQItem> out;
    while (!heap.empty()) {
        out.push_back(heap.top());
        heap.pop();
    }
    std::sort(out.begin(), out.end(),
              [](auto &a, auto &b) { return a.first < b.first; });

    auto program_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(program_end - program_start).count() / 1000.0;

    // Print query and its top‐K neighbors
    std::cout << "query:\n";
    // std::cout << "  embedding: " << query_obj["embedding"] << "\n";
    std::cout << "  text:    " << query_obj["text"] << "\n\n";

    // nlohmann::json output_json = nlohmann::json::array();

    for (int i = 0; i < (int)out.size(); ++i) {
        auto &p      = out[i];
        float dist   = p.first;
        int   idx    = p.second;
        auto &elem   = dict[idx];

        std::cout << "Neighbor " << (i + 1) << ":\n";
        std::cout << "  id:      " << idx
                  << ", dist = " << dist << "\n";
        // std::cout << "  embedding: " << elem["embedding"] << "\n";
        std::cout << "  text:    " << elem["text"] << "\n\n";

        nlohmann::json entry;
        entry["id"]      = idx;
        entry["dist"]    = dist;
        entry["embedding"] = elem["embedding"];
        entry["text"]    = elem["text"];

        // output_json.push_back(entry);
    }


    // std::string output_json_file = (std::is_same_v<T, float>) ? "neighbors_scalar.json" : "neighbors_vector.json";
    // std::ofstream file(output_json_file);
    // file << output_json.dump(2);
    // file.close();

    
    // Print timing information
    std::cout << "Part 2 Timing Results:" << std::endl;
    std::cout << "  Total time: " << total_time << " ms" << std::endl;
    std::cout << "  Parse time: " << parse_time << " ms" << std::endl;
    std::cout << "  Build time: " << buildtree_time << " ms" << std::endl;
    std::cout << "  Search time: " << query_time << " ms" << std::endl;
    
    // Output IDs array
    std::cout << "IDs: [";
    for (size_t i = 0; i < out.size(); ++i) {
        std::cout << out[i].second;  // out[i].second is the ID
        if (i < out.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    freeTree(root);
    return 0;
}



int main(int argc, char **argv)
{
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <dim> <query.json> <data.json> <K>\n";
        return 1;
    }

    // mode 0: scalar float, mode 1: fixed-size array<float,20>
    size_t dim = std::stoi(argv[1]);
    assert (dim >= 1);
    runtime_dim() = dim;

    char* new_argv[3];
    new_argv[0] = argv[2];   // pass query JSON‐filename as argv[0]
    new_argv[1] = argv[3];   // pass data JSON‐filename as argv[1]
    new_argv[2] = argv[4];   // pass K as argv[2]

    if (dim == 1) {
        return runMain<float>(new_argv);
    } else {
         return runMain<std::vector<float>>(new_argv);
    }
}