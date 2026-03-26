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

struct BatchTiming {
    int batch_size;
    double total_time_ms;
    double time_per_query_ms;
    double queries_per_second;
    double encoding_time_ms;
    double search_time_ms;
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

BatchTiming run_batch_benchmark(const std::vector<std::string>& queries,
                                 int batch_size,
                                 VectorDB* vector_db,
                                 llama_context* embedding_ctx,
                                 const llama_model* embedding_model,
                                 size_t top_k) {
    using clock = std::chrono::high_resolution_clock;
    
    int num_batches = queries.size() / batch_size;
    if (num_batches == 0) {
        throw std::runtime_error("Not enough queries for this batch size");
    }
    
    double total_encoding_time = 0.0;
    double total_search_time = 0.0;
    
    auto start_total = clock::now();
    
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        int start_idx = batch_idx * batch_size;
        
        // Encode batch of queries
        auto start_encode = clock::now();
        std::vector<std::vector<float>> batch_embeddings;
        for (int i = 0; i < batch_size; ++i) {
            std::vector<float> embedding = encode_query(embedding_ctx, embedding_model, 
                                                        queries[start_idx + i]);
            normalize_embedding(embedding);
            batch_embeddings.push_back(embedding);
        }
        auto end_encode = clock::now();
        total_encoding_time += std::chrono::duration<double, std::milli>(end_encode - start_encode).count();
        
        // Search ALL queries in batch at once (true FAISS batch search)
        auto start_search = clock::now();
        SearchResult result = vector_db->batch_search(batch_embeddings, top_k);
        auto end_search = clock::now();
        total_search_time += std::chrono::duration<double, std::milli>(end_search - start_search).count();
    }
    
    auto end_total = clock::now();
    
    int total_queries = num_batches * batch_size;
    double total_time_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    double time_per_query_ms = total_time_ms / total_queries;
    double queries_per_second = (total_queries / total_time_ms) * 1000.0;
    
    BatchTiming timing;
    timing.batch_size = batch_size;
    timing.total_time_ms = total_time_ms;
    timing.time_per_query_ms = time_per_query_ms;
    timing.queries_per_second = queries_per_second;
    timing.encoding_time_ms = total_encoding_time;
    timing.search_time_ms = total_search_time;
    
    return timing;
}

void save_batch_results(const std::vector<BatchTiming>& results, const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("Failed to open output file: " + filename);
    }
    
    ofs << "batch_size,total_time_ms,time_per_query_ms,queries_per_second,encoding_time_ms,search_time_ms\n";
    
    for (const auto& timing : results) {
        ofs << timing.batch_size << ","
            << timing.total_time_ms << ","
            << timing.time_per_query_ms << ","
            << timing.queries_per_second << ","
            << timing.encoding_time_ms << ","
            << timing.search_time_ms << "\n";
    }
    
    ofs.close();
    std::cout << "\nResults saved to: " << filename << std::endl;
}

int main(int argc, char** argv) {
    try {
        std::string docs_path = "preprocessed_documents.json";
        std::string embedding_model_path = "bge-base-en-v1.5-f32.gguf";
        std::string query_file = "queries.json";
        std::string output_file = "batch_results.csv";
        int num_queries = 256;  // Use 256 queries to accommodate batch size 128
        size_t top_k = 3;
        
        std::vector<int> batch_sizes = {1, 4, 8, 16, 32, 64, 128};
        
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
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  --queries <path>         Path to queries.json (default: queries.json)\n"
                          << "  --num-queries <n>        Number of queries to use (default: 256)\n"
                          << "  --output <path>          Output CSV file (default: batch_results.csv)\n"
                          << "  --documents <path>       Path to preprocessed_documents.json\n"
                          << "  --embedding-model <path> Path to BGE model\n"
                          << "  --top-k <n>              Number of documents to retrieve (default: 3)\n"
                          << "  --help, -h               Show this help message\n";
                return EXIT_SUCCESS;
            }
        }

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "BATCH PROCESSING BENCHMARK" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Queries:      " << query_file << std::endl;
        std::cout << "  Num queries:  " << num_queries << std::endl;
        std::cout << "  Top-K:        " << top_k << std::endl;
        std::cout << "  Batch sizes:  ";
        for (size_t i = 0; i < batch_sizes.size(); ++i) {
            std::cout << batch_sizes[i];
            if (i < batch_sizes.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        std::cout << "  Output:       " << output_file << std::endl;
        std::cout << std::string(80, '=') << std::endl << std::endl;
        
        // Initialize llama.cpp
        std::cout << "Initializing system..." << std::endl;
        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
        llama_log_set(custom_log_callback, nullptr);
        
        // Load vector database
        std::cout << "[1/3] Loading document database..." << std::flush;
        VectorDB vector_db(docs_path);
        vector_db.load_embeddings();
        vector_db.build_index();
        std::cout << " Done" << std::endl;
        
        // Load embedding model
        std::cout << "[2/3] Loading embedding model..." << std::flush;
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
        std::cout << "[3/3] Loading queries..." << std::flush;
        std::vector<std::string> base_queries = load_queries(query_file, -1);  // Load all queries
        
        // Repeat queries to reach num_queries if needed
        std::vector<std::string> queries;
        queries.reserve(num_queries);
        for (int i = 0; i < num_queries; ++i) {
            queries.push_back(base_queries[i % base_queries.size()]);
        }
        std::cout << " Done (" << queries.size() << " queries, " << base_queries.size() << " unique)" << std::endl << std::endl;
        
        // Run batch benchmarks
        std::vector<BatchTiming> results;
        
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "RUNNING BATCH EXPERIMENTS" << std::endl;
        std::cout << std::string(80, '=') << std::endl << std::endl;
        
        for (int batch_size : batch_sizes) {
            std::cout << std::string(80, '-') << std::endl;
            std::cout << "Batch Size: " << batch_size << std::endl;
            std::cout << std::string(80, '-') << std::endl;
            
            try {
                BatchTiming timing = run_batch_benchmark(queries, batch_size, &vector_db,
                                                         embedding_ctx, embedding_model, top_k);
                results.push_back(timing);
                
                std::cout << std::fixed << std::setprecision(2);
                std::cout << "  Total Time:        " << timing.total_time_ms << " ms" << std::endl;
                std::cout << "  Time per Query:    " << timing.time_per_query_ms << " ms" << std::endl;
                std::cout << "  Throughput:        " << timing.queries_per_second << " queries/sec" << std::endl;
                std::cout << "  Encoding Time:     " << timing.encoding_time_ms << " ms" << std::endl;
                std::cout << "  Search Time:       " << timing.search_time_ms << " ms" << std::endl;
                std::cout << std::endl;
                
            } catch (const std::exception& ex) {
                std::cerr << "  Error: " << ex.what() << std::endl << std::endl;
            }
        }
        
        // Print summary
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "SUMMARY" << std::endl;
        std::cout << std::string(80, '=') << std::endl << std::endl;
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Batch Size    Latency/Query    Throughput       Speedup" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        double baseline_qps = results.empty() ? 0.0 : results[0].queries_per_second;
        
        for (const auto& timing : results) {
            double speedup = timing.queries_per_second / baseline_qps;
            std::cout << std::setw(10) << timing.batch_size
                      << std::setw(17) << timing.time_per_query_ms << " ms"
                      << std::setw(17) << timing.queries_per_second << " q/s"
                      << std::setw(10) << speedup << "x" << std::endl;
        }
        
        std::cout << std::string(80, '=') << std::endl;
        
        // Cleanup
        llama_free(embedding_ctx);
        llama_model_free(embedding_model);
        llama_backend_free();
        
        // Save results
        save_batch_results(results, output_file);
        
        std::cout << "\n✅ Batch benchmark complete!\n" << std::endl;
        
        return EXIT_SUCCESS;
        
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
