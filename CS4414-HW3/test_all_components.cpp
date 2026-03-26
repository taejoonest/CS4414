#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "llama.h"
#include "vector_db.h"
#include "encode.h"
#include "llm_generation.h"

static void custom_log_callback(enum ggml_log_level level, const char* text, void* user_data) {
    (void)user_data; (void)level; (void)text;
}

std::string build_prompt(const std::string& query, VectorDB& db, const SearchResult& result) {
    std::ostringstream oss;
    oss << "Based on the following documents, answer the question.\n\n";
    for (size_t i = 0; i < result.indices.size() && i < 3; ++i) {
        const Document& doc = db.get_document_by_index(result.indices[i]);
        oss << "Document " << (i+1) << ": " << doc.text.substr(0, 300) << "...\n\n";
    }
    oss << "Question: " << query << "\n\nAnswer:";
    return oss.str();
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
    llama_model* emb_model = llama_model_load_from_file("bge-base-en-v1.5-f32.gguf", params);
    llama_context_params emb_ctx_params = llama_context_default_params();
    emb_ctx_params.embeddings = true;
    emb_ctx_params.n_ctx = 512;
    llama_context* emb_ctx = llama_init_from_model(emb_model, emb_ctx_params);
    
    // Load LLM
    llama_model* llm_model = llama_model_load_from_file("qwen2-1_5b-instruct-q4_0.gguf", params);
    llama_context_params llm_ctx_params = llama_context_default_params();
    llm_ctx_params.n_ctx = 2048;
    llm_ctx_params.n_batch = 512;
    llama_context* llm_ctx = llama_init_from_model(llm_model, llm_ctx_params);
    
    std::vector<std::string> queries = {
        "what is machine learning", "how does neural network work", "what is backpropagation",
        "explain gradient descent", "what is deep learning", "how do transformers work",
        "what is attention mechanism", "explain BERT model", "what is GPT", "how does LSTM work",
        "what is CNN", "explain random forest", "what is SVM", "how does k-means work",
        "what is PCA", "explain decision trees", "what is reinforcement learning",
        "how does Q-learning work", "what is GAN", "explain autoencoder"
    };
    
    std::ofstream csv("all_components.csv");
    csv << "query_idx,encoding_ms,search_ms,generation_ms\n";
    
    using clock = std::chrono::high_resolution_clock;
    
    std::cout << "Running 20 queries (all 3 components):\n";
    std::cout << "Index | Encoding | Search  | Generation\n";
    std::cout << std::string(50, '-') << "\n";
    
    for (int i = 0; i < 20; ++i) {
        auto t1 = clock::now();
        std::vector<float> emb = encode_query(emb_ctx, emb_model, queries[i]);
        normalize_embedding(emb);
        auto t2 = clock::now();
        SearchResult res = db.search(emb, 3);
        auto t3 = clock::now();
        std::string prompt = build_prompt(queries[i], db, res);
        std::string response = generate_response(llm_ctx, llm_model, prompt, 100);
        auto t4 = clock::now();
        
        double enc_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double search_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double gen_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
        
        std::cout << std::setw(5) << i << " | " 
                  << std::setw(8) << std::fixed << std::setprecision(1) << enc_ms << " | "
                  << std::setw(7) << search_ms << " | "
                  << std::setw(10) << gen_ms << "\n";
        
        csv << i << "," << enc_ms << "," << search_ms << "," << gen_ms << "\n";
    }
    
    csv.close();
    llama_free(emb_ctx);
    llama_free(llm_ctx);
    llama_model_free(emb_model);
    llama_model_free(llm_model);
    llama_backend_free();
    
    std::cout << "\nSaved to all_components.csv\n";
    return 0;
}
