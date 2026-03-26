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
    (void)user_data; (void)level; (void)text;
}

std::vector<std::string> load_queries(const std::string& file, int n) {
    std::ifstream ifs(file);
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    std::vector<std::string> queries;
    size_t pos = 0;
    while ((pos = content.find("\"text\":", pos)) != std::string::npos && (int)queries.size() < n) {
        pos += 7; while (pos < content.size() && content[pos] != '"') pos++; pos++;
        size_t end = pos;
        while (end < content.size() && content[end] != '"') { if (content[end] == '\\') end++; end++; }
        queries.push_back(content.substr(pos, end - pos));
        pos = end + 1;
    }
    return queries;
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
    
    VectorDB db("preprocessed_documents.json");
    db.load_embeddings();
    db.build_index();
    
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 99;
    
    // Load embedding model to get search results
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
    
    std::vector<std::string> queries = load_queries("queries.json", 21);
    
    // Pre-compute all prompts first (no timing)
    std::cout << "Pre-computing prompts..." << std::endl;
    std::vector<std::string> prompts;
    for (int i = 0; i <= 20; ++i) {
        auto emb = encode_query(emb_ctx, emb_model, queries[i]);
        normalize_embedding(emb);
        SearchResult res = db.search(emb, 3);
        prompts.push_back(build_prompt(queries[i], db, res));
    }
    
    // Warm-up LLM with query 0
    std::cout << "Warming up LLM..." << std::endl;
    generate_response(llm_ctx, llm_model, prompts[0], 50);
    std::cout << "Warm-up complete.\n" << std::endl;
    
    std::ofstream csv("llm_only.csv");
    csv << "query_idx,generation_ms\n";
    
    using clock = std::chrono::high_resolution_clock;
    
    std::cout << "Running LLM generation for queries 1-20:\n";
    std::cout << "Index | Generation\n";
    std::cout << std::string(25, '-') << "\n";
    
    for (int i = 1; i <= 20; ++i) {
        auto t1 = clock::now();
        generate_response(llm_ctx, llm_model, prompts[i], 100);
        auto t2 = clock::now();
        
        double gen_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        std::cout << std::setw(5) << i << " | " 
                  << std::setw(10) << std::fixed << std::setprecision(1) << gen_ms << "\n";
        
        csv << i << "," << gen_ms << "\n";
    }
    
    csv.close();
    llama_free(emb_ctx);
    llama_free(llm_ctx);
    llama_model_free(emb_model);
    llama_model_free(llm_model);
    llama_backend_free();
    
    std::cout << "\nSaved to llm_only.csv\n";
    return 0;
}
