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
#include "llm_generation.h"

struct ComponentTiming {
    double encoding_ms;
    double search_ms;
    double augmentation_ms;
    double generation_ms;
    double total_ms;
};

struct BenchmarkResult {
    std::string query;
    ComponentTiming timing;
};

struct Statistics {
    double mean;
    double median;
    double min;
    double max;
    double std_dev;
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

std::string build_augmented_prompt(const std::string& query, 
                                   const VectorDB& db, 
                                   const SearchResult& result) {
    std::ostringstream oss;
    oss << "Answer the user's question clearly and concisely. ";
    oss << "Use the provided reference documents as supporting information when relevant.\n\n";
    oss << "Reference Documents:\n";
    for (size_t i = 0; i < result.k; ++i) {
        faiss::idx_t doc_idx = result.indices[i];
        const Document& doc = db.get_document_by_index(doc_idx);
        oss << "[" << (i + 1) << "] " << doc.text << "\n\n";
    }
    oss << "User Question: " << query << "\n\n";
    oss << "Answer (directly address the question):";
    return oss.str();
}

struct RAGSystem {
    VectorDB* vector_db;
    llama_model* embedding_model;
    llama_context* embedding_ctx;
    llama_model* llm_model;
    llama_context* llm_ctx;
    size_t top_k;
    
    RAGSystem() : vector_db(nullptr), embedding_model(nullptr), embedding_ctx(nullptr),
                  llm_model(nullptr), llm_ctx(nullptr), top_k(3) {}
    
    ~RAGSystem() {
        if (embedding_ctx) llama_free(embedding_ctx);
        if (embedding_model) llama_model_free(embedding_model);
        if (llm_ctx) llama_free(llm_ctx);
        if (llm_model) llama_model_free(llm_model);
        if (vector_db) delete vector_db;
        llama_backend_free();
    }
    
    void initialize(const std::string& docs_path,
                   const std::string& embedding_model_path,
                   const std::string& llm_model_path) {
        std::cout << "Initializing RAG System for benchmarking..." << std::endl;
        
        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
        llama_log_set(custom_log_callback, nullptr);

        std::cout << "[1/4] Loading document database..." << std::flush;
        vector_db = new VectorDB(docs_path);
        vector_db->load_embeddings();
        vector_db->build_index();
        std::cout << " Done" << std::endl;
        
        std::cout << "[2/4] Loading embedding model..." << std::flush;
        llama_model_params embedding_model_params = llama_model_default_params();
        embedding_model_params.n_gpu_layers = 99;
        embedding_model = llama_model_load_from_file(embedding_model_path.c_str(), embedding_model_params);
        if (!embedding_model) {
            throw std::runtime_error("Failed to load embedding model: " + embedding_model_path);
        }
        
        llama_context_params embedding_ctx_params = llama_context_default_params();
        embedding_ctx_params.embeddings = true;
        embedding_ctx_params.n_ctx = 512;
        embedding_ctx_params.n_threads = 8;
        embedding_ctx_params.n_batch = 512;
        
        embedding_ctx = llama_init_from_model(embedding_model, embedding_ctx_params);
        if (!embedding_ctx) {
            throw std::runtime_error("Failed to create embedding context");
        }
        std::cout << " Done" << std::endl;
        
        std::cout << "[3/4] Loading LLM model..." << std::flush;
        llama_model_params llm_model_params = llama_model_default_params();
        llm_model_params.n_gpu_layers = 99;
        llm_model = llama_model_load_from_file(llm_model_path.c_str(), llm_model_params);
        if (!llm_model) {
            throw std::runtime_error("Failed to load LLM model: " + llm_model_path);
        }
        
        llama_context_params llm_ctx_params = llama_context_default_params();
        llm_ctx_params.embeddings = false;
        llm_ctx_params.n_ctx = 4096;  // Increased for larger Top-K prompts
        llm_ctx_params.n_threads = 8;
        llm_ctx_params.n_batch = 4096;  // Increased to handle large prompts (K=5,10)
        
        llm_ctx = llama_init_from_model(llm_model, llm_ctx_params);
        if (!llm_ctx) {
            throw std::runtime_error("Failed to create LLM context");
        }
        std::cout << " Done" << std::endl;
        std::cout << "[4/4] System ready" << std::endl << std::endl;
    }
    
    std::string process_query(const std::string& query, ComponentTiming* timing) {
        using clock = std::chrono::high_resolution_clock;
        auto start_total = clock::now();
        
        // Component 1: Encoding
        auto start_encode = clock::now();
        std::vector<float> query_embedding = encode_query(embedding_ctx, embedding_model, query);
        normalize_embedding(query_embedding);
        auto end_encode = clock::now();
        double encoding_ms = std::chrono::duration<double, std::milli>(end_encode - start_encode).count();
        
        // Component 2: Vector Search
        auto start_search = clock::now();
        SearchResult result = vector_db->search(query_embedding, top_k);
        auto end_search = clock::now();
        double search_ms = std::chrono::duration<double, std::milli>(end_search - start_search).count();
        
        // Component 3: Prompt Augmentation
        auto start_aug = clock::now();
        std::string augmented_prompt = build_augmented_prompt(query, *vector_db, result);
        auto end_aug = clock::now();
        double augmentation_ms = std::chrono::duration<double, std::milli>(end_aug - start_aug).count();

        // Component 4: LLM Generation
        auto start_gen = clock::now();
        std::string response = generate_response(llm_ctx, llm_model, augmented_prompt, 512);
        auto end_gen = clock::now();
        double generation_ms = std::chrono::duration<double, std::milli>(end_gen - start_gen).count();
        
        auto end_total = clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        
        if (timing) {
            timing->encoding_ms = encoding_ms;
            timing->search_ms = search_ms;
            timing->augmentation_ms = augmentation_ms;
            timing->generation_ms = generation_ms;
            timing->total_ms = total_ms;
        }
        
        return response;
    }
};

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

Statistics calculate_statistics(const std::vector<double>& values) {
    Statistics stats;
    
    if (values.empty()) {
        stats.mean = stats.median = stats.min = stats.max = stats.std_dev = 0.0;
        return stats;
    }
    
    // Mean
    double sum = 0.0;
    for (double v : values) sum += v;
    stats.mean = sum / values.size();
    
    // Min/Max
    stats.min = *std::min_element(values.begin(), values.end());
    stats.max = *std::max_element(values.begin(), values.end());
    
    // Median
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    size_t mid = sorted.size() / 2;
    if (sorted.size() % 2 == 0) {
        stats.median = (sorted[mid-1] + sorted[mid]) / 2.0;
    } else {
        stats.median = sorted[mid];
    }
    
    // Standard Deviation
    double variance = 0.0;
    for (double v : values) {
        variance += (v - stats.mean) * (v - stats.mean);
    }
    stats.std_dev = std::sqrt(variance / values.size());
    
    return stats;
}

void save_results_csv(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("Failed to open output file: " + filename);
    }
    
    ofs << "query,encoding_ms,search_ms,augmentation_ms,generation_ms,total_ms\n";
    
    for (const auto& result : results) {
        ofs << "\"" << result.query << "\","
            << result.timing.encoding_ms << ","
            << result.timing.search_ms << ","
            << result.timing.augmentation_ms << ","
            << result.timing.generation_ms << ","
            << result.timing.total_ms << "\n";
    }
    
    ofs.close();
    std::cout << "Results saved to: " << filename << std::endl;
}

void print_statistics(const std::vector<BenchmarkResult>& results) {
    std::vector<double> encoding_times, search_times, aug_times, gen_times, total_times;
    
    for (const auto& result : results) {
        encoding_times.push_back(result.timing.encoding_ms);
        search_times.push_back(result.timing.search_ms);
        aug_times.push_back(result.timing.augmentation_ms);
        gen_times.push_back(result.timing.generation_ms);
        total_times.push_back(result.timing.total_ms);
    }
    
    Statistics enc_stats = calculate_statistics(encoding_times);
    Statistics search_stats = calculate_statistics(search_times);
    Statistics aug_stats = calculate_statistics(aug_times);
    Statistics gen_stats = calculate_statistics(gen_times);
    Statistics total_stats = calculate_statistics(total_times);
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "BENCHMARK STATISTICS (" << results.size() << " queries)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nComponent            Mean      Median    Min       Max       StdDev" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    std::cout << "Encoding         " << std::setw(8) << enc_stats.mean 
              << std::setw(10) << enc_stats.median
              << std::setw(10) << enc_stats.min
              << std::setw(10) << enc_stats.max
              << std::setw(10) << enc_stats.std_dev << " ms" << std::endl;
              
    std::cout << "Vector Search    " << std::setw(8) << search_stats.mean
              << std::setw(10) << search_stats.median
              << std::setw(10) << search_stats.min
              << std::setw(10) << search_stats.max
              << std::setw(10) << search_stats.std_dev << " ms" << std::endl;
              
    std::cout << "Augmentation     " << std::setw(8) << aug_stats.mean
              << std::setw(10) << aug_stats.median
              << std::setw(10) << aug_stats.min
              << std::setw(10) << aug_stats.max
              << std::setw(10) << aug_stats.std_dev << " ms" << std::endl;
              
    std::cout << "LLM Generation   " << std::setw(8) << gen_stats.mean
              << std::setw(10) << gen_stats.median
              << std::setw(10) << gen_stats.min
              << std::setw(10) << gen_stats.max
              << std::setw(10) << gen_stats.std_dev << " ms" << std::endl;
              
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "TOTAL            " << std::setw(8) << total_stats.mean
              << std::setw(10) << total_stats.median
              << std::setw(10) << total_stats.min
              << std::setw(10) << total_stats.max
              << std::setw(10) << total_stats.std_dev << " ms" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Component percentages
    std::cout << "\nComponent Breakdown (% of total time):" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    double total = total_stats.mean;
    std::cout << "  Encoding:       " << std::setw(6) << (enc_stats.mean / total * 100) << "%" << std::endl;
    std::cout << "  Vector Search:  " << std::setw(6) << (search_stats.mean / total * 100) << "%" << std::endl;
    std::cout << "  Augmentation:   " << std::setw(6) << (aug_stats.mean / total * 100) << "%" << std::endl;
    std::cout << "  LLM Generation: " << std::setw(6) << (gen_stats.mean / total * 100) << "%" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

int main(int argc, char** argv) {
    try {
        std::string docs_path = "preprocessed_documents.json";
        std::string embedding_model_path = "bge-base-en-v1.5-f32.gguf";
        std::string llm_model_path = "qwen2-1_5b-instruct-q4_0.gguf";
        std::string query_file = "queries.json";
        std::string output_file = "benchmark_results.csv";
        int num_queries = 50;  // Default: run 50 queries
        size_t top_k = 3;
        
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
            } else if (arg == "--llm-model" && i + 1 < argc) {
                llm_model_path = argv[++i];
            } else if (arg == "--top-k" && i + 1 < argc) {
                top_k = std::stoull(argv[++i]);
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  --queries <path>         Path to queries.json (default: queries.json)\n"
                          << "  --num-queries <n>        Number of queries to run (default: 50)\n"
                          << "  --output <path>          Output CSV file (default: benchmark_results.csv)\n"
                          << "  --documents <path>       Path to preprocessed_documents.json\n"
                          << "  --embedding-model <path> Path to BGE model\n"
                          << "  --llm-model <path>       Path to LLM model\n"
                          << "  --top-k <n>              Number of documents to retrieve (default: 3)\n"
                          << "  --help, -h               Show this help message\n";
                return EXIT_SUCCESS;
            }
        }

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "RAG SYSTEM BENCHMARK" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Queries:     " << query_file << std::endl;
        std::cout << "  Num queries: " << num_queries << std::endl;
        std::cout << "  Top-K:       " << top_k << std::endl;
        std::cout << "  Output:      " << output_file << std::endl;
        std::cout << std::string(80, '=') << std::endl << std::endl;
        
        // Initialize RAG system
        RAGSystem rag_system;
        rag_system.top_k = top_k;
        rag_system.initialize(docs_path, embedding_model_path, llm_model_path);
        
        // Load queries
        std::cout << "Loading queries from " << query_file << "..." << std::endl;
        std::vector<std::string> queries = load_queries(query_file, num_queries);
        std::cout << "Loaded " << queries.size() << " queries" << std::endl << std::endl;
        
        // Warm-up phase: run 3 queries without timing to warm up caches
        std::cout << "Warm-up phase (3 queries)..." << std::endl;
        for (int w = 0; w < 3 && w < (int)queries.size(); ++w) {
            ComponentTiming dummy;
            rag_system.process_query(queries[w], &dummy);
            std::cout << "  Warm-up " << (w+1) << "/3 complete" << std::endl;
        }
        std::cout << "Warm-up complete. Starting timed benchmark.\n" << std::endl;
        
        // Run benchmark
        std::vector<BenchmarkResult> results;
        std::cout << "Running benchmark..." << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (size_t i = 0; i < queries.size(); ++i) {
            std::cout << "[" << (i+1) << "/" << queries.size() << "] " 
                      << queries[i].substr(0, 50);
            if (queries[i].size() > 50) std::cout << "...";
            std::cout << std::flush;
            
            BenchmarkResult result;
            result.query = queries[i];
            rag_system.process_query(queries[i], &result.timing);
            results.push_back(result);
            
            std::cout << " [" << std::fixed << std::setprecision(0) 
                      << result.timing.total_ms << " ms]" << std::endl;
        }
        
        std::cout << std::string(80, '-') << std::endl;
        std::cout << "Benchmark complete!" << std::endl << std::endl;
        
        // Save results
        save_results_csv(results, output_file);
        
        // Print statistics
        print_statistics(results);
        
        return EXIT_SUCCESS;
        
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
