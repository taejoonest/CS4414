#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "llama.h"
#include "vector_db.h"
#include "encode.h"
#include "llm_generation.h"

static void custom_log_callback(enum ggml_log_level level, const char* text, void* user_data) {
    (void)user_data;
    (void)level;
    (void)text;
    // Suppress all logs
}

std::string build_prompt(const std::string& query, VectorDB& db, const SearchResult& result) {
    std::ostringstream oss;
    oss << "Based on the following documents, answer the question.\n\n";
    for (size_t i = 0; i < result.indices.size() && i < 3; ++i) {
        const Document& doc = db.get_document_by_index(result.indices[i]);
        oss << "Document " << (i+1) << ": " << doc.text.substr(0, 500) << "...\n\n";
    }
    oss << "Question: " << query << "\n\nAnswer:";
    return oss.str();
}

int main(int argc, char** argv) {
    std::string llm_model = "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf";
    std::string model_name = "TinyLlama-1.1B";
    
    if (argc > 1) {
        llm_model = argv[1];
        model_name = (argc > 2) ? argv[2] : llm_model;
    }
    
    llama_backend_init();
    llama_log_set(custom_log_callback, nullptr);
    
    // Load vector DB
    VectorDB db("preprocessed_documents.json");
    db.load_embeddings();
    db.build_index();
    
    // Load embedding model
    llama_model_params emb_params = llama_model_default_params();
    emb_params.n_gpu_layers = 99;
    llama_model* emb_model = llama_model_load_from_file("bge-base-en-v1.5-f32.gguf", emb_params);
    llama_context_params emb_ctx_params = llama_context_default_params();
    emb_ctx_params.embeddings = true;
    emb_ctx_params.n_ctx = 512;
    llama_context* emb_ctx = llama_init_from_model(emb_model, emb_ctx_params);
    
    // Load LLM
    llama_model_params llm_params = llama_model_default_params();
    llm_params.n_gpu_layers = 99;
    llama_model* llm_model_ptr = llama_model_load_from_file(llm_model.c_str(), llm_params);
    llama_context_params llm_ctx_params = llama_context_default_params();
    llm_ctx_params.n_ctx = 4096;
    llm_ctx_params.n_batch = 4096;
    llama_context* llm_ctx = llama_init_from_model(llm_model_ptr, llm_ctx_params);
    
    // Test queries
    std::vector<std::string> queries = {
        "What is machine learning?",
        "How does neural network training work?",
        "What is the purpose of backpropagation?"
    };
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "LLM QUALITY COMPARISON: " << model_name << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    for (size_t i = 0; i < queries.size(); ++i) {
        std::cout << "\n" << std::string(80, '-') << std::endl;
        std::cout << "Query " << (i+1) << ": " << queries[i] << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        // Encode and search
        std::vector<float> emb = encode_query(emb_ctx, emb_model, queries[i]);
        normalize_embedding(emb);
        SearchResult result = db.search(emb, 3);
        
        // Build prompt and generate
        std::string prompt = build_prompt(queries[i], db, result);
        std::string response = generate_response(llm_ctx, llm_model_ptr, prompt, 150);
        
        std::cout << "\nResponse:\n" << response << std::endl;
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    
    llama_free(emb_ctx);
    llama_free(llm_ctx);
    llama_model_free(emb_model);
    llama_model_free(llm_model_ptr);
    llama_backend_free();
    
    return 0;
}
