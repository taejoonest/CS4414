#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>

#include "llama.h"
#include "vector_db.h"
#include "encode.h"

static void custom_log_callback(enum ggml_log_level level, const char* text, void* user_data) {
    (void)user_data; (void)level; (void)text;
}

int main() {
    llama_backend_init();
    llama_log_set(custom_log_callback, nullptr);
    
    VectorDB db("preprocessed_documents.json");
    db.load_embeddings();
    db.build_index();
    
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 99;
    llama_model* model = llama_model_load_from_file("bge-base-en-v1.5-f32.gguf", params);
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    ctx_params.n_ctx = 512;
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    
    // Pre-encode 128 queries (repeat if needed)
    std::vector<std::string> base_queries = {
        "what is machine learning", "how does neural network work", "what is backpropagation",
        "explain gradient descent", "what is deep learning", "how do transformers work",
        "what is attention mechanism", "explain BERT model"
    };
    
    const int NUM_QUERIES = 128;
    const int EMB_DIM = 768;
    
    std::cout << "Pre-encoding 128 queries into flat array..." << std::endl;
    // Flat array: 128 queries × 768 dimensions
    std::vector<float> all_embeddings(NUM_QUERIES * EMB_DIM);
    for (int i = 0; i < NUM_QUERIES; ++i) {
        std::vector<float> emb = encode_query(ctx, model, base_queries[i % base_queries.size()]);
        normalize_embedding(emb);
        // Copy into flat array
        memcpy(&all_embeddings[i * EMB_DIM], emb.data(), EMB_DIM * sizeof(float));
    }
    std::cout << "Encoding complete. Now testing batch search only.\n" << std::endl;
    
    std::ofstream csv("batch_search_only.csv");
    csv << "batch_size,search_time_ms,time_per_query_ms,queries_per_second\n";
    
    using clock = std::chrono::high_resolution_clock;
    std::vector<int> batch_sizes = {1, 4, 8, 16, 32, 64, 128};
    
    std::cout << "Batch Size | Search Time (ms) | Per Query (ms) | Throughput (q/s)\n";
    std::cout << std::string(65, '-') << "\n";
    
    for (int batch_size : batch_sizes) {
        int num_batches = NUM_QUERIES / batch_size;
        
        auto start = clock::now();
        for (int b = 0; b < num_batches; ++b) {
            // Point directly to the flat array section - no copying
            const float* batch_ptr = &all_embeddings[b * batch_size * EMB_DIM];
            SearchResult res = db.batch_search_flat(batch_ptr, batch_size, EMB_DIM, 3);
        }
        auto end = clock::now();
        
        double search_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double per_query = search_ms / 128.0;
        double qps = 128000.0 / search_ms;
        
        std::cout << std::setw(10) << batch_size << " | "
                  << std::setw(16) << std::fixed << std::setprecision(2) << search_ms << " | "
                  << std::setw(14) << per_query << " | "
                  << std::setw(16) << qps << "\n";
        
        csv << batch_size << "," << search_ms << "," << per_query << "," << qps << "\n";
    }
    
    csv.close();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    std::cout << "\nSaved to batch_search_only.csv\n";
    return 0;
}
