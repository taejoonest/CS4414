#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "llama.h"
#include "vector_db.h"
#include "encode.h"

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

int main() {
    llama_backend_init();
    llama_log_set(custom_log_callback, nullptr);
    
    VectorDB db("preprocessed_documents.json");
    db.load_embeddings();
    db.build_index();
    
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 99;
    llama_model* emb_model = llama_model_load_from_file("bge-base-en-v1.5-f32.gguf", params);
    llama_context_params emb_ctx_params = llama_context_default_params();
    emb_ctx_params.embeddings = true;
    emb_ctx_params.n_ctx = 512;
    llama_context* emb_ctx = llama_init_from_model(emb_model, emb_ctx_params);
    
    std::vector<std::string> queries = load_queries("queries.json", 25);
    
    // Warm-up with queries 0 AND 1
    std::cout << "Warming up with 2 queries..." << std::endl;
    for (int w = 0; w < 2; ++w) {
        auto emb = encode_query(emb_ctx, emb_model, queries[w]);
        normalize_embedding(emb);
        db.search(emb, 3);
    }
    std::cout << "Warm-up complete.\n" << std::endl;
    
    std::ofstream csv("encode_search_only.csv");
    csv << "query_idx,encoding_ms,search_ms\n";
    
    using clock = std::chrono::high_resolution_clock;
    
    std::cout << "Running queries 2-21 (encode + search only):\n";
    std::cout << "Index | Encoding | Search\n";
    std::cout << std::string(30, '-') << "\n";
    
    for (int i = 2; i <= 21; ++i) {
        auto t1 = clock::now();
        std::vector<float> e = encode_query(emb_ctx, emb_model, queries[i]);
        normalize_embedding(e);
        auto t2 = clock::now();
        db.search(e, 3);
        auto t3 = clock::now();
        
        double enc_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double search_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        
        std::cout << std::setw(5) << (i-1) << " | "  // Display as 1-20
                  << std::setw(8) << std::fixed << std::setprecision(1) << enc_ms << " | "
                  << std::setw(6) << search_ms << "\n";
        
        csv << (i-1) << "," << enc_ms << "," << search_ms << "\n";
    }
    
    csv.close();
    llama_free(emb_ctx);
    llama_model_free(emb_model);
    llama_backend_free();
    
    std::cout << "\nSaved to encode_search_only.csv\n";
    return 0;
}
