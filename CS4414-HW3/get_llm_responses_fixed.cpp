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

std::string build_prompt(const std::string& query, VectorDB& db, const SearchResult& result, int k) {
    std::ostringstream oss;
    oss << "Answer the question based on these documents:\n\n";
    for (int i = 0; i < k && i < (int)result.indices.size(); ++i) {
        const Document& doc = db.get_document_by_index(result.indices[i]);
        oss << "Doc" << (i+1) << ": " << doc.text.substr(0, 200) << "...\n";
    }
    oss << "\nQuestion: " << query << "\nAnswer:";
    return oss.str();
}

int main(int argc, char** argv) {
    std::string llm_path = argv[1];
    std::string model_name = argv[2];
    int top_k = std::stoi(argv[3]);
    
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
    
    llama_model* llm = llama_model_load_from_file(llm_path.c_str(), params);
    llama_context_params llm_ctx_params = llama_context_default_params();
    llm_ctx_params.n_ctx = 2048;
    llama_context* llm_ctx = llama_init_from_model(llm, llm_ctx_params);
    
    // Load queries from queries.json - use queries 1, 5, 8 (diverse topics)
    auto all_queries = load_queries("queries.json", 10);
    std::vector<std::string> queries = {all_queries[0], all_queries[4], all_queries[7]};
    
    using clock = std::chrono::high_resolution_clock;
    
    std::cout << "MODEL:" << model_name << ",K=" << top_k << std::endl;
    
    for (const auto& query : queries) {
        auto t1 = clock::now();
        std::vector<float> emb = encode_query(emb_ctx, emb_model, query);
        normalize_embedding(emb);
        SearchResult res = db.search(emb, top_k);
        std::string prompt = build_prompt(query, db, res, top_k);
        std::string response = generate_response(llm_ctx, llm, prompt, 80);
        auto t2 = clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        // Clean response
        size_t pos = response.find('\n');
        if (pos != std::string::npos && pos < 150) response = response.substr(0, pos);
        if (response.length() > 150) response = response.substr(0, 147) + "...";
        
        std::cout << "Q:" << query << std::endl;
        std::cout << "R:" << response << std::endl;
        std::cout << "T:" << ms << std::endl;
    }
    
    llama_free(emb_ctx);
    llama_free(llm_ctx);
    llama_model_free(emb_model);
    llama_model_free(llm);
    llama_backend_free();
    return 0;
}
