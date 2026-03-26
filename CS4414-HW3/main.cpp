#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <faiss/IndexFlat.h>
#include "llama.h"

#include "vector_db.h"
#include "encode.h"
#include "llm_generation.h"

struct ComponentTiming {
    double encoding_ms;
    double search_ms;
    double augmentation_ms;
    double generation_ms;
    double total_ms;
};

static void custom_log_callback(enum ggml_log_level level, const char* text, void* user_data) {
    (void)user_data;

    std::string msg(text);
    if (msg.find("create_tensor") != std::string::npos ||
        msg.find("print_info") != std::string::npos ||
        msg.find("llama_model_loader") != std::string::npos ||
        msg.find("Dumping metadata") != std::string::npos ||
        msg.find("- kv") != std::string::npos) {
        return;
    }

    if (level >= GGML_LOG_LEVEL_WARN) {
        fprintf(stderr, "%s", text);
    }
}

namespace {

std::string build_augmented_prompt(const std::string& query, 
                                   const VectorDB& db, 
                                   const SearchResult& result) {
    std::ostringstream oss;
    
    oss << "Answer the user's question clearly and concisely. ";
    oss << "Use the provided reference documents as supporting information when relevant.\n\n";
    
    oss << "Reference Documents:\n";
    for (size_t i = 0; i < result.k; ++i) {
        faiss::idx_t doc_idx = result.indices[i];
        const Document& doc = db.get_document_by_index(doc_idx);
        oss << "[" << (i + 1) << "] " << doc.text << "\n\n";
    }

    oss << "User Question: " << query << "\n\n";
    oss << "Answer (directly address the question):";
    
    return oss.str();
}

struct RAGSystem {
    VectorDB* vector_db;
    llama_model* embedding_model;
    llama_context* embedding_ctx;
    llama_model* llm_model;
    llama_context* llm_ctx;
    size_t top_k;
    
    RAGSystem() : vector_db(nullptr), embedding_model(nullptr), embedding_ctx(nullptr),
                  llm_model(nullptr), llm_ctx(nullptr), top_k(3) {}
    
    ~RAGSystem() {
        if (embedding_ctx) llama_free(embedding_ctx);
        if (embedding_model) llama_model_free(embedding_model);
        if (llm_ctx) llama_free(llm_ctx);
        if (llm_model) llama_model_free(llm_model);
        if (vector_db) delete vector_db;
        llama_backend_free();
    }
    
    void initialize(const std::string& docs_path,
                   const std::string& embedding_model_path,
                   const std::string& llm_model_path) {

        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
        
        llama_log_set(custom_log_callback, nullptr);

        std::cout << "[1/4] Loading document database..." << std::endl;
        vector_db = new VectorDB(docs_path);
        vector_db->load_embeddings();
        vector_db->build_index();
        std::cout << std::endl;
        
        std::cout << "[2/4] Loading embedding model..." << std::endl;
        llama_model_params embedding_model_params = llama_model_default_params();
        embedding_model_params.n_gpu_layers = 99;
        embedding_model = llama_model_load_from_file(embedding_model_path.c_str(), embedding_model_params);
        if (!embedding_model) {
            throw std::runtime_error("Failed to load embedding model: " + embedding_model_path);
        }
        
        llama_context_params embedding_ctx_params = llama_context_default_params();
        embedding_ctx_params.embeddings = true;
        embedding_ctx_params.n_ctx = 512;
        embedding_ctx_params.n_threads = 8;
        embedding_ctx_params.n_batch = 512;
        
        embedding_ctx = llama_init_from_model(embedding_model, embedding_ctx_params);
        if (!embedding_ctx) {
            throw std::runtime_error("Failed to create embedding context");
        }
        std::cout << "Embedding model loaded (BGE)" << std::endl << std::endl;
        
        std::cout << "[3/4] Loading LLM model..." << std::endl;
        llama_model_params llm_model_params = llama_model_default_params();
        llm_model_params.n_gpu_layers = 99;
        llm_model = llama_model_load_from_file(llm_model_path.c_str(), llm_model_params);
        if (!llm_model) {
            throw std::runtime_error("Failed to load LLM model: " + llm_model_path);
        }
        
        llama_context_params llm_ctx_params = llama_context_default_params();
        llm_ctx_params.embeddings = false;
        llm_ctx_params.n_ctx = 2048;
        llm_ctx_params.n_threads = 8;
        llm_ctx_params.n_batch = 512;
        
        llm_ctx = llama_init_from_model(llm_model, llm_ctx_params);
        if (!llm_ctx) {
            throw std::runtime_error("Failed to create LLM context");
        }
        std::cout << "LLM model loaded" << std::endl << std::endl;
        
        std::cout << "[4/4] Done." << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Documents: " << vector_db->size() << std::endl;
        std::cout << "  Top-K: " << top_k << std::endl;
        std::cout << std::string(70, '=') << std::endl << std::endl;
    }
    
    std::string process_query(const std::string& query) {
        return process_query(query, nullptr);
    }
    
    std::string process_query(const std::string& query, ComponentTiming* timing) {
        using clock = std::chrono::high_resolution_clock;
        auto start_total = clock::now();
        
        std::cout << "\n" << std::string(70, '-') << std::endl;
        std::cout << "Query: " << query << std::endl;
        std::cout << std::string(70, '-') << std::endl << std::endl;
        
        // Component 1: Encoding
        std::cout << "[1/4] Encoding query..." << std::flush;
        auto start_encode = clock::now();
        std::vector<float> query_embedding = encode_query(embedding_ctx, embedding_model, query);
        normalize_embedding(query_embedding);
        auto end_encode = clock::now();
        double encoding_ms = std::chrono::duration<double, std::milli>(end_encode - start_encode).count();
        
        // Component 2: Vector Search
        std::cout << "[2/4] Retrieving relevant documents..." << std::flush;
        auto start_search = clock::now();
        SearchResult result = vector_db->search(query_embedding, top_k);
        auto end_search = clock::now();
        double search_ms = std::chrono::duration<double, std::milli>(end_search - start_search).count();
        
        std::cout << "\nTop " << top_k << " Retrieved Documents:" << std::endl;
        for (size_t i = 0; i < result.k; ++i) {
            faiss::idx_t doc_idx = result.indices[i];
            float distance = result.distances[i];
            const Document& doc = vector_db->get_document_by_index(doc_idx);
            
            std::cout << "  [" << (i + 1) << "] ID: " << doc.id 
                      << " (distance: " << std::fixed << std::setprecision(3) << distance << ")" << std::endl;
            std::string preview = doc.text.substr(0, std::min(size_t(80), doc.text.size()));
            std::cout << "      " << preview;
            if (doc.text.size() > 80) std::cout << "...";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        
        // Component 3: Prompt Augmentation
        std::cout << "[3/4] Building augmented prompt..." << std::flush;
        auto start_aug = clock::now();
        std::string augmented_prompt = build_augmented_prompt(query, *vector_db, result);
        auto end_aug = clock::now();
        double augmentation_ms = std::chrono::duration<double, std::milli>(end_aug - start_aug).count();
        
        std::cout << "\n" << std::string(70, '-') << std::endl;
        std::cout << "Augmented Prompt:" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        std::cout << augmented_prompt << std::endl;
        std::cout << std::string(70, '-') << std::endl << std::endl;

        // Component 4: LLM Generation
        std::cout << "[4/4] Generating response..." << std::flush;
        auto start_gen = clock::now();
        std::string response = generate_response(llm_ctx, llm_model, augmented_prompt, 512);
        auto end_gen = clock::now();
        double generation_ms = std::chrono::duration<double, std::milli>(end_gen - start_gen).count();
        
        auto end_total = clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        
        if (timing) {
            timing->encoding_ms = encoding_ms;
            timing->search_ms = search_ms;
            timing->augmentation_ms = augmentation_ms;
            timing->generation_ms = generation_ms;
            timing->total_ms = total_ms;
        }
        
        return response;
    }
    
    void run_interactive() {
        std::cout << "\n" << std::string(70, '=') << std::endl;

        std::cout << std::string(70, '=') << std::endl;
        std::cout << "\nType your questions and get AI-powered answers!" << std::endl;
        std::cout << "Commands:" << std::endl;
        std::cout << "  - Type your question and press Enter" << std::endl;
        std::cout << "  - Type 'exit', 'quit', or Ctrl+D to exit" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        while (true) {
            std::cout << "\n> ";
            std::string query;
            
            if (!std::getline(std::cin, query)) {

                break;
            }
            
            query.erase(0, query.find_first_not_of(" \t\n\r"));
            query.erase(query.find_last_not_of(" \t\n\r") + 1);
            
            if (query.empty()) {
                continue;
            }
            
            if (query == "exit" || query == "quit") {
                break;
            }
            
            try {
                ComponentTiming timing;
                std::string response = process_query(query, &timing);
                
                std::cout << "\n" << std::string(70, '=') << std::endl;
                std::cout << "ANSWER:" << std::endl;
                std::cout << std::string(70, '=') << std::endl;
                std::cout << response << std::endl;
                std::cout << std::string(70, '=') << std::endl;
                
                std::cout << "\n" << std::string(70, '-') << std::endl;
                std::cout << "PERFORMANCE BREAKDOWN:" << std::endl;
                std::cout << std::string(70, '-') << std::endl;
                std::cout << std::fixed << std::setprecision(2);
                std::cout << "  Encoding:       " << std::setw(8) << timing.encoding_ms << " ms" << std::endl;
                std::cout << "  Vector Search:  " << std::setw(8) << timing.search_ms << " ms" << std::endl;
                std::cout << "  Augmentation:   " << std::setw(8) << timing.augmentation_ms << " ms" << std::endl;
                std::cout << "  LLM Generation: " << std::setw(8) << timing.generation_ms << " ms" << std::endl;
                std::cout << "  " << std::string(66, '-') << std::endl;
                std::cout << "  TOTAL:          " << std::setw(8) << timing.total_ms << " ms" << std::endl;
                std::cout << std::string(70, '-') << std::endl;
            } catch (const std::exception& ex) {
                std::cerr << "\nError processing query: " << ex.what() << std::endl;
            }
        }
    }
};

}

int main(int argc, char** argv) {
    try {
        std::string docs_path = "preprocessed_documents.json";
        std::string embedding_model_path = "bge-base-en-v1.5-f32.gguf";
        std::string llm_model_path = "qwen2-1_5b-instruct-q4_0.gguf";
        size_t top_k = 3;
        
        for (int i = 1; i < argc; ++i) {
            std::string_view arg(argv[i]);
            auto need_value = [&](const char* flag) -> std::string {
                if (i + 1 >= argc) {
                    std::cerr << "Error: Missing value for " << flag << "\n";
                    std::exit(EXIT_FAILURE);
                }
                return argv[++i];
            };
            if (arg == "--documents") {
                docs_path = need_value("--documents");
            } else if (arg == "--embedding-model") {
                embedding_model_path = need_value("--embedding-model");
            } else if (arg == "--llm-model") {
                llm_model_path = need_value("--llm-model");
            } else if (arg == "--top-k") {
                top_k = std::stoull(need_value("--top-k"));
            } else {
                std::cerr << "Error: Unknown argument: " << arg << "\n";
                std::exit(EXIT_FAILURE);
            }
        }

        RAGSystem rag_system;
        rag_system.top_k = top_k;
        rag_system.initialize(docs_path, embedding_model_path, llm_model_path);
        rag_system.run_interactive();
        
        return EXIT_SUCCESS;
        
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
