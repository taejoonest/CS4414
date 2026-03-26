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
    
    auto queries = load_queries("queries.json", 50);
    
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 99;
    llama_model* model = llama_model_load_from_file("bge-base-en-v1.5-f32.gguf", params);
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true; ctx_params.n_ctx = 512;
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    
    std::vector<std::vector<float>> embeddings;
    for (const auto& q : queries) {
        auto emb = encode_query(ctx, model, q);
        normalize_embedding(emb);
        embeddings.push_back(emb);
    }
    
    using clock = std::chrono::high_resolution_clock;
    std::vector<std::vector<faiss::idx_t>> flat_results;
    
    std::cout << "Index Type | Avg (ms) | Min (ms) | Max (ms) | Accuracy | Speedup\n";
    std::cout << std::string(65, '-') << "\n";
    
    double flat_avg = 0;
    
    for (int idx_type = 0; idx_type < 2; ++idx_type) {
        VectorDB db("preprocessed_documents.json", idx_type == 0 ? IndexType::FLAT : IndexType::IVF_FLAT, 100);
        db.load_embeddings();
        db.build_index();
        
        // Warm-up
        for (int w = 0; w < 5; ++w) db.search(embeddings[w], 3);
        
        std::vector<double> times;
        std::vector<std::vector<faiss::idx_t>> results;
        
        for (size_t i = 0; i < embeddings.size(); ++i) {
            auto start = clock::now();
            auto res = db.search(embeddings[i], 3);
            auto end = clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            results.push_back(res.indices);
        }
        
        if (idx_type == 0) { flat_results = results; }
        
        double accuracy = 1.0;
        if (idx_type == 1) {
            int hits = 0, total = 0;
            for (size_t i = 0; i < results.size(); ++i) {
                for (auto gt : flat_results[i]) {
                    for (auto r : results[i]) { if (gt == r) { hits++; break; } }
                    total++;
                }
            }
            accuracy = (double)hits / total;
        }
        
        double sum = 0; for (double t : times) sum += t;
        double mean = sum / times.size();
        double min_t = *std::min_element(times.begin(), times.end());
        double max_t = *std::max_element(times.begin(), times.end());
        
        if (idx_type == 0) flat_avg = mean;
        double speedup = flat_avg / mean;
        
        std::cout << (idx_type == 0 ? "Flat    " : "IVFFlat ") << " | "
                  << mean << " | " << min_t << " | " << max_t << " | "
                  << accuracy*100 << "% | " << speedup << "x\n";
    }
    
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
