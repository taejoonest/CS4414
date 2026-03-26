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
    
    // Load all 100 queries
    std::vector<std::string> all_queries = load_queries("queries.json", 100);
    std::cout << "Loaded " << all_queries.size() << " queries\n";
    
    // Pre-encode all 100 queries
    std::cout << "Pre-encoding all queries..." << std::endl;
    std::vector<std::vector<float>> all_embeddings;
    for (const auto& q : all_queries) {
        std::vector<float> emb = encode_query(ctx, model, q);
        normalize_embedding(emb);
        all_embeddings.push_back(emb);
    }
    std::cout << "Encoding complete.\n" << std::endl;
    
    std::ofstream csv("batch_search_correct.csv");
    csv << "batch_size,num_queries,search_time_ms,time_per_query_ms,queries_per_second\n";
    
    using clock = std::chrono::high_resolution_clock;
    std::vector<int> batch_sizes = {1, 4, 8, 16, 32, 64, 128};
    
    std::cout << "Batch Size | Queries | Search (ms) | Per Query (ms) | Throughput (q/s)\n";
    std::cout << std::string(70, '-') << "\n";
    
    for (int batch_size : batch_sizes) {
        // Select embeddings: use first N for sizes <= 64, use all 100 + 28 repeated for 128
        std::vector<std::vector<float>> selected_emb;
        int num_queries;
        
        if (batch_size <= 64) {
            // Use first batch_size queries (overlapping: 64 includes 32 includes 16 etc.)
            num_queries = batch_size;
            for (int i = 0; i < batch_size; ++i) {
                selected_emb.push_back(all_embeddings[i]);
            }
        } else {
            // batch_size = 128: use all 100 + repeat first 28
            num_queries = 128;
            for (int i = 0; i < 100; ++i) {
                selected_emb.push_back(all_embeddings[i]);
            }
            for (int i = 0; i < 28; ++i) {
                selected_emb.push_back(all_embeddings[i]);  // repeat first 28
            }
        }
        
        // Run batch search
        auto start = clock::now();
        SearchResult res = db.batch_search(selected_emb, 3);
        auto end = clock::now();
        
        double search_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double per_query = search_ms / num_queries;
        double qps = num_queries * 1000.0 / search_ms;
        
        std::cout << std::setw(10) << batch_size << " | "
                  << std::setw(7) << num_queries << " | "
                  << std::setw(11) << std::fixed << std::setprecision(2) << search_ms << " | "
                  << std::setw(14) << per_query << " | "
                  << std::setw(16) << qps << "\n";
        
        csv << batch_size << "," << num_queries << "," << search_ms << "," << per_query << "," << qps << "\n";
    }
    
    csv.close();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    std::cout << "\nSaved to batch_search_correct.csv\n";
    return 0;
}
