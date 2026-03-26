#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <faiss/IndexFlat.h>
#include "llama.h"

#include "vector_db.h"
#include "encode.h"

struct IndexComparison {
    std::string index_type;
    int num_queries;
    double avg_search_time_ms;
    double min_search_time_ms;
    double max_search_time_ms;
    double std_dev_ms;
    double queries_per_second;
    double recall;  // Recall@K compared to exact search
    std::vector<std::vector<faiss::idx_t>> all_results;  // Store results for recall calculation
};

static void custom_log_callback(enum ggml_log_level level, const char* text, void* user_data) {
    (void)user_data;
    std::string msg(text);
    if (msg.find("create_tensor") != std::string::npos ||
        msg.find("print_info") != std::string::npos ||
        msg.find("llama_model_loader") != std::string::npos ||
        msg.find("Dumping metadata") != std::string::npos ||
        msg.find("- kv") != std::string::npos) {
        return;
    }
    if (level >= GGML_LOG_LEVEL_WARN) {
        fprintf(stderr, "%s", text);
    }
}

std::vector<std::string> load_queries(const std::string& query_file, int max_queries = -1) {
    std::ifstream ifs(query_file);
    if (!ifs) {
        throw std::runtime_error("Failed to open query file: " + query_file);
    }
    
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();
    
    std::vector<std::string> queries;
    size_t pos = 0;
    int count = 0;
    
    while ((pos = content.find("\"text\":", pos)) != std::string::npos) {
        pos += 7;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
        
        if (pos >= content.size() || content[pos] != '"') continue;
        
        pos++;
        size_t end_pos = pos;
        while (end_pos < content.size() && content[end_pos] != '"') {
            if (content[end_pos] == '\\') end_pos++;
            end_pos++;
        }
        
        if (end_pos >= content.size()) break;
        
        std::string query = content.substr(pos, end_pos - pos);
        queries.push_back(query);
        count++;
        
        if (max_queries > 0 && count >= max_queries) break;
        
        pos = end_pos + 1;
    }
    
    return queries;
}

IndexComparison benchmark_index_type(const std::string& docs_path,
                                     IndexType index_type,
                                     size_t nlist,
                                     const std::vector<std::string>& queries,
                                     llama_context* embedding_ctx,
                                     const llama_model* embedding_model,
                                     size_t top_k) {
    using clock = std::chrono::high_resolution_clock;
    
    std::string type_name = (index_type == IndexType::FLAT) ? "Flat" : "IVFFlat";
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Testing " << type_name << " Index";
    if (index_type == IndexType::IVF_FLAT) {
        std::cout << " (nlist=" << nlist << ")";
    }
    std::cout << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Load database and build index
    std::cout << "Building index..." << std::flush;
    auto start_build = clock::now();
    VectorDB vector_db(docs_path, index_type, nlist);
    vector_db.load_embeddings();
    vector_db.build_index();
    auto end_build = clock::now();
    double build_time_ms = std::chrono::duration<double, std::milli>(end_build - start_build).count();
    std::cout << " Done (" << build_time_ms / 1000.0 << " seconds)" << std::endl;
    
    // Run queries and measure search time
    std::vector<double> search_times;
    std::vector<std::vector<faiss::idx_t>> all_results;
    
    std::cout << "Running " << queries.size() << " queries..." << std::endl;
    
    for (size_t i = 0; i < queries.size(); ++i) {
        if (i % 10 == 0) {
            std::cout << "  Progress: " << i << "/" << queries.size() << "\r" << std::flush;
        }
        
        // Encode query
        std::vector<float> embedding = encode_query(embedding_ctx, embedding_model, queries[i]);
        normalize_embedding(embedding);
        
        // Measure search time only
        auto start_search = clock::now();
        SearchResult search_result = vector_db.search(embedding, top_k);
        auto end_search = clock::now();
        
        double search_time_ms = std::chrono::duration<double, std::milli>(end_search - start_search).count();
        search_times.push_back(search_time_ms);
        
        // Store results for recall calculation
        all_results.push_back(search_result.indices);
    }
    
    std::cout << "  Progress: " << queries.size() << "/" << queries.size() << std::endl;
    
    // Calculate statistics
    double sum = 0.0;
    for (double t : search_times) sum += t;
    double avg = sum / search_times.size();
    
    double min_time = *std::min_element(search_times.begin(), search_times.end());
    double max_time = *std::max_element(search_times.begin(), search_times.end());
    
    double variance = 0.0;
    for (double t : search_times) {
        variance += (t - avg) * (t - avg);
    }
    double std_dev = std::sqrt(variance / search_times.size());
    
    double qps = (queries.size() / sum) * 1000.0;
    
    IndexComparison result;
    result.index_type = type_name;
    result.num_queries = queries.size();
    result.avg_search_time_ms = avg;
    result.min_search_time_ms = min_time;
    result.max_search_time_ms = max_time;
    result.std_dev_ms = std_dev;
    result.queries_per_second = qps;
    result.recall = 1.0;  // Will be calculated later for IVFFlat
    result.all_results = all_results;
    
    return result;
}

// Calculate Recall@K: how many of the exact top-K are found in approximate top-K
double calculate_recall(const std::vector<std::vector<faiss::idx_t>>& ground_truth,
                        const std::vector<std::vector<faiss::idx_t>>& approximate) {
    if (ground_truth.size() != approximate.size()) {
        return 0.0;
    }
    
    double total_recall = 0.0;
    
    for (size_t q = 0; q < ground_truth.size(); ++q) {
        const auto& gt = ground_truth[q];
        const auto& approx = approximate[q];
        
        int hits = 0;
        for (faiss::idx_t gt_idx : gt) {
            for (faiss::idx_t approx_idx : approx) {
                if (gt_idx == approx_idx) {
                    hits++;
                    break;
                }
            }
        }
        
        double query_recall = static_cast<double>(hits) / gt.size();
        total_recall += query_recall;
    }
    
    return total_recall / ground_truth.size();
}

int main(int argc, char** argv) {
    try {
        std::string docs_path = "preprocessed_documents.json";
        std::string embedding_model_path = "bge-base-en-v1.5-f32.gguf";
        std::string query_file = "queries.json";
        std::string output_file = "index_comparison.csv";
        int num_queries = 100;
        size_t top_k = 3;
        size_t nlist = 100;  // Number of clusters for IVF
        
        // Parse arguments
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--queries" && i + 1 < argc) {
                query_file = argv[++i];
            } else if (arg == "--num-queries" && i + 1 < argc) {
                num_queries = std::stoi(argv[++i]);
            } else if (arg == "--output" && i + 1 < argc) {
                output_file = argv[++i];
            } else if (arg == "--documents" && i + 1 < argc) {
                docs_path = argv[++i];
            } else if (arg == "--embedding-model" && i + 1 < argc) {
                embedding_model_path = argv[++i];
            } else if (arg == "--top-k" && i + 1 < argc) {
                top_k = std::stoull(argv[++i]);
            } else if (arg == "--nlist" && i + 1 < argc) {
                nlist = std::stoull(argv[++i]);
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  --queries <path>         Path to queries.json (default: queries.json)\n"
                          << "  --num-queries <n>        Number of queries to run (default: 100)\n"
                          << "  --output <path>          Output CSV file (default: index_comparison.csv)\n"
                          << "  --documents <path>       Path to preprocessed_documents.json\n"
                          << "  --embedding-model <path> Path to BGE model\n"
                          << "  --top-k <n>              Number of documents to retrieve (default: 3)\n"
                          << "  --nlist <n>              Number of IVF clusters (default: 100)\n"
                          << "  --help, -h               Show this help message\n";
                return EXIT_SUCCESS;
            }
        }

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "INDEX TYPE COMPARISON BENCHMARK" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Queries:      " << query_file << std::endl;
        std::cout << "  Num queries:  " << num_queries << std::endl;
        std::cout << "  Top-K:        " << top_k << std::endl;
        std::cout << "  IVF nlist:    " << nlist << std::endl;
        std::cout << "  Output:       " << output_file << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        // Initialize llama.cpp
        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
        llama_log_set(custom_log_callback, nullptr);
        
        // Load embedding model
        std::cout << "\nLoading embedding model..." << std::flush;
        llama_model_params embedding_model_params = llama_model_default_params();
        embedding_model_params.n_gpu_layers = 99;
        llama_model* embedding_model = llama_model_load_from_file(embedding_model_path.c_str(), 
                                                                   embedding_model_params);
        if (!embedding_model) {
            throw std::runtime_error("Failed to load embedding model: " + embedding_model_path);
        }
        
        llama_context_params embedding_ctx_params = llama_context_default_params();
        embedding_ctx_params.embeddings = true;
        embedding_ctx_params.n_ctx = 512;
        embedding_ctx_params.n_threads = 8;
        embedding_ctx_params.n_batch = 512;
        
        llama_context* embedding_ctx = llama_init_from_model(embedding_model, embedding_ctx_params);
        if (!embedding_ctx) {
            throw std::runtime_error("Failed to create embedding context");
        }
        std::cout << " Done" << std::endl;
        
        // Load queries
        std::cout << "Loading queries..." << std::flush;
        std::vector<std::string> queries = load_queries(query_file, num_queries);
        std::cout << " Done (" << queries.size() << " queries)" << std::endl;
        
        // Benchmark both index types
        std::vector<IndexComparison> results;
        
        results.push_back(benchmark_index_type(docs_path, IndexType::FLAT, 0, 
                                               queries, embedding_ctx, embedding_model, top_k));
        
        results.push_back(benchmark_index_type(docs_path, IndexType::IVF_FLAT, nlist, 
                                               queries, embedding_ctx, embedding_model, top_k));
        
        // Calculate recall for IVFFlat (Flat is ground truth with 100% recall)
        results[0].recall = 1.0;  // Flat search is exact
        results[1].recall = calculate_recall(results[0].all_results, results[1].all_results);
        
        // Print comparison
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "COMPARISON RESULTS" << std::endl;
        std::cout << std::string(80, '=') << std::endl << std::endl;
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Index Type    Avg Search    Min         Max         StdDev      QPS         Recall@K" << std::endl;
        std::cout << std::string(90, '-') << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::setw(12) << std::left << result.index_type
                      << std::setw(13) << std::right << result.avg_search_time_ms << "ms"
                      << std::setw(12) << result.min_search_time_ms << "ms"
                      << std::setw(12) << result.max_search_time_ms << "ms"
                      << std::setw(12) << result.std_dev_ms << "ms"
                      << std::setw(10) << result.queries_per_second
                      << std::setw(12) << (result.recall * 100.0) << "%" << std::endl;
        }
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "ANALYSIS:" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        double speedup = results[0].avg_search_time_ms / results[1].avg_search_time_ms;
        std::cout << "\nIVFFlat speedup: " << speedup << "x ";
        if (speedup > 1.0) {
            std::cout << "(faster)";
        } else {
            std::cout << "(slower)";
        }
        std::cout << std::endl;
        
        double qps_improvement = ((results[1].queries_per_second / results[0].queries_per_second) - 1.0) * 100.0;
        std::cout << "Throughput improvement: " << std::showpos << qps_improvement << std::noshowpos << "%" << std::endl;
        
        std::cout << "\nNote: IVFFlat provides approximate nearest neighbor search." << std::endl;
        std::cout << "It trades accuracy for speed, especially beneficial with larger datasets." << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        // Save to CSV
        std::ofstream ofs(output_file);
        if (ofs) {
            ofs << "index_type,avg_search_ms,min_search_ms,max_search_ms,std_dev_ms,queries_per_second,recall\n";
            for (const auto& result : results) {
                ofs << result.index_type << ","
                    << result.avg_search_time_ms << ","
                    << result.min_search_time_ms << ","
                    << result.max_search_time_ms << ","
                    << result.std_dev_ms << ","
                    << result.queries_per_second << ","
                    << result.recall << "\n";
            }
            ofs.close();
            std::cout << "\nResults saved to: " << output_file << std::endl;
        }
        
        // Cleanup
        llama_free(embedding_ctx);
        llama_model_free(embedding_model);
        llama_backend_free();
        
        std::cout << "\n✅ Index comparison complete!\n" << std::endl;
        
        return EXIT_SUCCESS;
        
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}



