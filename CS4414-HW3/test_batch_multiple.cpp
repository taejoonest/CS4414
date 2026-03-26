#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include "llama.h"
#include "vector_db.h"
#include "encode.h"

static void custom_log_callback(enum ggml_log_level level, const char* text, void* user_data) {
    (void)user_data; (void)level; (void)text;
}

std::vector<std::string> load_queries(const std::string& file, int max_q) {
    std::ifstream ifs(file);
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    std::vector<std::string> queries;
    size_t pos = 0;
    while ((pos = content.find("\"text\":", pos)) != std::string::npos && (int)queries.size() < max_q) {
        pos += 7;
        while (pos < content.size() && content[pos] != '"') pos++;
        pos++;
        size_t end = pos;
        while (end < content.size() && content[end] != '"') {
            if (content[end] == '\\') end++;
            end++;
        }
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
    llama_model* model = llama_model_load_from_file("bge-base-en-v1.5-f32.gguf", params);
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    ctx_params.n_ctx = 512;
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    
    std::vector<std::string> all_queries = load_queries("queries.json", 100);
    
    // Pre-encode all 100 + 28 repeated = 128 queries
    std::vector<std::vector<float>> all_embeddings;
    for (int i = 0; i < 128; ++i) {
        std::vector<float> emb = encode_query(ctx, model, all_queries[i % 100]);
        normalize_embedding(emb);
        all_embeddings.push_back(emb);
    }
    
    // Warm up
    for (int i = 0; i < 3; ++i) {
        db.batch_search({all_embeddings[0]}, 3);
    }
    
    using clock = std::chrono::high_resolution_clock;
    std::vector<int> batch_sizes = {1, 4, 8, 16, 32, 64, 128};
    int num_runs = 5;
    
    std::cout << "Running " << num_runs << " trials per batch size:\n\n";
    std::cout << "Batch | Run1 | Run2 | Run3 | Run4 | Run5 | Mean | StdDev\n";
    std::cout << std::string(70, '-') << "\n";
    
    for (int batch_size : batch_sizes) {
        std::vector<std::vector<float>> batch_emb;
        int n = (batch_size <= 64) ? batch_size : 128;
        for (int i = 0; i < n; ++i) batch_emb.push_back(all_embeddings[i]);
        
        std::vector<double> times;
        for (int r = 0; r < num_runs; ++r) {
            auto start = clock::now();
            db.batch_search(batch_emb, 3);
            auto end = clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(ms / n);  // per query
        }
        
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double sq_sum = 0;
        for (double t : times) sq_sum += (t - mean) * (t - mean);
        double stddev = std::sqrt(sq_sum / times.size());
        
        std::cout << std::setw(5) << batch_size << " |";
        for (double t : times) std::cout << std::setw(6) << std::fixed << std::setprecision(2) << t;
        std::cout << " |" << std::setw(6) << mean << " |" << std::setw(7) << stddev << "\n";
    }
    
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
