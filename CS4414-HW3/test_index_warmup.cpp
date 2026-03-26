#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include "llama.h"
#include "vector_db.h"
#include "encode.h"

static void custom_log_callback(enum ggml_log_level level, const char* text, void* user_data) {
    (void)user_data; (void)level; (void)text;
}

int main() {
    llama_backend_init();
    llama_log_set(custom_log_callback, nullptr);
    
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 99;
    llama_model* model = llama_model_load_from_file("bge-base-en-v1.5-f32.gguf", params);
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    ctx_params.n_ctx = 512;
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    
    // Pre-encode 50 queries
    std::vector<std::vector<float>> embeddings;
    std::vector<std::string> queries = {"what is ml", "neural network", "backprop", "gradient", "deep learning"};
    for (int i = 0; i < 50; ++i) {
        auto emb = encode_query(ctx, model, queries[i % 5]);
        normalize_embedding(emb);
        embeddings.push_back(emb);
    }
    
    using clock = std::chrono::high_resolution_clock;
    std::ofstream csv("index_comparison_clean.csv");
    csv << "index_type,avg_search_ms,min_search_ms,max_search_ms,std_dev_ms,accuracy\n";
    
    // Store Flat results for accuracy comparison
    std::vector<std::vector<faiss::idx_t>> flat_results;
    
    for (int idx_type = 0; idx_type < 2; ++idx_type) {
        VectorDB db("preprocessed_documents.json", idx_type == 0 ? IndexType::FLAT : IndexType::IVF_FLAT, 100);
        db.load_embeddings();
        db.build_index();
        
        // Warm-up: 3 queries
        for (int w = 0; w < 3; ++w) db.search(embeddings[w], 3);
        
        std::vector<double> times;
        std::vector<std::vector<faiss::idx_t>> results;
        
        for (int i = 0; i < 50; ++i) {
            auto start = clock::now();
            auto res = db.search(embeddings[i], 3);
            auto end = clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            results.push_back(res.indices);
        }
        
        if (idx_type == 0) flat_results = results;
        
        // Calculate accuracy (recall)
        double accuracy = 1.0;
        if (idx_type == 1) {
            int hits = 0, total = 0;
            for (int i = 0; i < 50; ++i) {
                for (auto gt : flat_results[i]) {
                    for (auto r : results[i]) {
                        if (gt == r) { hits++; break; }
                    }
                    total++;
                }
            }
            accuracy = (double)hits / total;
        }
        
        double sum = 0; for (double t : times) sum += t;
        double mean = sum / times.size();
        double min_t = *std::min_element(times.begin(), times.end());
        double max_t = *std::max_element(times.begin(), times.end());
        double var = 0; for (double t : times) var += (t-mean)*(t-mean);
        double std = std::sqrt(var / times.size());
        
        std::cout << (idx_type == 0 ? "Flat" : "IVFFlat") 
                  << ": avg=" << mean << "ms, min=" << min_t << "ms, max=" << max_t 
                  << "ms, accuracy=" << accuracy*100 << "%\n";
        
        csv << (idx_type == 0 ? "Flat" : "IVFFlat") << "," << mean << "," << min_t << "," << max_t << "," << std << "," << accuracy << "\n";
    }
    
    csv.close();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
