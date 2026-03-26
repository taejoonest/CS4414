#include <chrono>
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
    
    // Load vector DB
    VectorDB db("preprocessed_documents.json");
    db.load_embeddings();
    db.build_index();
    
    // Load embedding model
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 99;
    llama_model* model = llama_model_load_from_file("bge-base-en-v1.5-f32.gguf", params);
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    ctx_params.n_ctx = 512;
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    
    std::vector<std::string> queries = {
        "what is machine learning",
        "how does neural network work",
        "what is backpropagation",
        "explain gradient descent",
        "what is deep learning",
        "how do transformers work",
        "what is attention mechanism",
        "explain BERT model",
        "what is GPT",
        "how does LSTM work",
        "what is CNN",
        "explain random forest",
        "what is SVM",
        "how does k-means work",
        "what is PCA",
        "explain decision trees",
        "what is reinforcement learning",
        "how does Q-learning work",
        "what is GAN",
        "explain autoencoder"
    };
    
    std::ofstream csv("encode_search_only.csv");
    csv << "query_idx,encoding_ms,search_ms\n";
    
    using clock = std::chrono::high_resolution_clock;
    
    std::cout << "Running 20 queries (encoding + search ONLY, no LLM):\n";
    std::cout << "Index | Encoding (ms) | Search (ms)\n";
    std::cout << std::string(45, '-') << "\n";
    
    for (int i = 0; i < 20; ++i) {
        auto t1 = clock::now();
        std::vector<float> emb = encode_query(ctx, model, queries[i]);
        normalize_embedding(emb);
        auto t2 = clock::now();
        SearchResult res = db.search(emb, 3);
        auto t3 = clock::now();
        
        double enc_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double search_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        
        std::cout << std::setw(5) << i << " | " 
                  << std::setw(13) << std::fixed << std::setprecision(2) << enc_ms << " | "
                  << std::setw(10) << search_ms << "\n";
        
        csv << i << "," << enc_ms << "," << search_ms << "\n";
    }
    
    csv.close();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    std::cout << "\nSaved to encode_search_only.csv\n";
    return 0;
}
