#include "encode.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

void normalize_embedding(std::vector<float>& vec) {
    double sum_sq = 0.0;
    for (float v : vec) {
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
    }
    if (sum_sq <= 0.0) {
        return;
    }
    double inv_norm = 1.0 / std::sqrt(sum_sq);
    for (float& v : vec) {
        v = static_cast<float>(v * inv_norm);
    }
}

std::vector<float> encode_query(llama_context* ctx,
                                 const llama_model* model,
                                 const std::string& text) {
    const int dim = llama_model_n_embd(model);

    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        throw std::runtime_error("Failed to get vocab from model");
    }

    const int n_max_tokens = 512;
    std::vector<llama_token> tokens(n_max_tokens);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
                                   tokens.data(), n_max_tokens, true, false);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
                                   tokens.data(), -n_tokens, true, false);
    }
    if (n_tokens < 0) {
        throw std::runtime_error("Failed to tokenize input");
    }
    tokens.resize(n_tokens);

    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);

    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.seq_id[i][0] = 0;
        batch.n_seq_id[i] = 1;
        batch.logits[i] = -1;
    }
    batch.n_tokens = n_tokens;

    if (llama_encode(ctx, batch) < 0) {
        llama_batch_free(batch);
        throw std::runtime_error("llama_encode failed");
    }

    std::vector<float> embedding(dim);

    float* embd = llama_get_embeddings_seq(ctx, 0);
    if (!embd) {
        embd = llama_get_embeddings(ctx);
    }

    if (!embd) {
        llama_batch_free(batch);
        throw std::runtime_error("Failed to get embeddings");
    }

    bool all_zero = true;
    for (int i = 0; i < dim && all_zero; ++i) {
        if (std::abs(embd[i]) > 1e-9) {
            all_zero = false;
        }
    }

    if (all_zero) {
        llama_batch_free(batch);
        throw std::runtime_error("Embeddings are all zeros - model may not be working correctly");
    }

    std::copy(embd, embd + dim, embedding.begin());

    llama_batch_free(batch);
    return embedding;
}

#ifdef ENCODE_STANDALONE

#include <fstream>
#include <iomanip>
#include <iostream>

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

struct Options {
    std::string query;
    std::string model_path;
    std::string output_path;
    bool normalize = true;
};

[[noreturn]] void usage_and_exit(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --query <text> --model <model.gguf> [options]\n"
        << "Options:\n"
        << "  --query <text>         Query text to encode\n"
        << "  --model <model.gguf>   Path to BGE model\n"
        << "  --output <path>        Save embedding to file (JSON array)\n"
        << "  --no-normalize         Disable L2 normalization of embeddings\n";
    std::exit(EXIT_FAILURE);
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        auto need_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                usage_and_exit(argv[0]);
            }
            return argv[++i];
        };
        if (arg == "--query") {
            opt.query = need_value("--query");
        } else if (arg == "--model") {
            opt.model_path = need_value("--model");
        } else if (arg == "--output") {
            opt.output_path = need_value("--output");
        } else if (arg == "--no-normalize") {
            opt.normalize = false;
        } else if (arg == "--help" || arg == "-h") {
            usage_and_exit(argv[0]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            usage_and_exit(argv[0]);
        }
    }

    if (opt.query.empty() || opt.model_path.empty()) {
        std::cerr << "Both --query and --model are required.\n";
        usage_and_exit(argv[0]);
    }
    return opt;
}

}

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        std::cout << "Query Encoder - Component 1" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        std::cout << "Query: \"" << opt.query << "\"" << std::endl;
        std::cout << "Model: " << opt.model_path << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

        llama_log_set(custom_log_callback, nullptr);

        std::cout << "Loading model..." << std::endl;
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 99;
        llama_model* model = llama_model_load_from_file(opt.model_path.c_str(), model_params);
        if (!model) {
            throw std::runtime_error("Failed to load model: " + opt.model_path);
        }
        std::cout << "Model loaded successfully" << std::endl;

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.embeddings = true;
        ctx_params.n_ctx = 512;
        ctx_params.n_threads = 8;
        ctx_params.n_batch = 512;

        llama_context* ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            llama_model_free(model);
            throw std::runtime_error("Failed to create llama context");
        }

        const int embedding_dim = llama_model_n_embd(model);
        if (embedding_dim != 768) {
            std::cerr << "Warning: Model produces " << embedding_dim
                      << "-dimensional embeddings, expected 768" << std::endl;
        } else {
            std::cout << "Embedding dimension: 768" << std::endl;
        }

        std::cout << "Encoding query..." << std::endl;
        std::vector<float> embedding = encode_query(ctx, model, opt.query);

        if (embedding.size() != 768) {
            throw std::runtime_error(
                "Embedding dimension mismatch: got " + std::to_string(embedding.size())
                + ", expected 768");
        }

        if (opt.normalize) {
            normalize_embedding(embedding);
            std::cout << "Embedding normalized" << std::endl;
        }

        std::cout << "\nQuery Embedding (768-dimensional vector):" << std::endl;
        std::cout << "[";
        for (size_t i = 0; i < embedding.size(); ++i) {
            if (i > 0) std::cout << ", ";
            if (i % 8 == 0 && i > 0) std::cout << "\n ";
            std::cout << std::fixed << std::setprecision(6) << embedding[i];
        }
        std::cout << "]" << std::endl;

        double sum = 0.0, sum_sq = 0.0;
        float min_val = embedding[0], max_val = embedding[0];
        for (float v : embedding) {
            sum += v;
            sum_sq += v * v;
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        double mean = sum / embedding.size();
        double norm = std::sqrt(sum_sq);

        std::cout << "\nEmbedding Statistics:" << std::endl;
        std::cout << "  Dimension: " << embedding.size() << std::endl;
        std::cout << "  Mean: " << std::fixed << std::setprecision(6) << mean << std::endl;
        std::cout << "  L2 Norm: " << std::fixed << std::setprecision(6) << norm << std::endl;
        std::cout << "  Min: " << std::fixed << std::setprecision(6) << min_val << std::endl;
        std::cout << "  Max: " << std::fixed << std::setprecision(6) << max_val << std::endl;

        if (!opt.output_path.empty()) {
            std::cout << "\nSaving embedding to " << opt.output_path << "..." << std::endl;
            std::ofstream ofs(opt.output_path);
            if (!ofs) {
                throw std::runtime_error("Failed to open output file: " + opt.output_path);
            }

            ofs << "[\n";
            for (size_t i = 0; i < embedding.size(); ++i) {
                ofs << "  " << std::fixed << std::setprecision(8) << embedding[i];
                if (i < embedding.size() - 1) {
                    ofs << ",";
                }
                ofs << "\n";
            }
            ofs << "]\n";
            ofs.close();
            std::cout << "Embedding saved" << std::endl;
        }

        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();

        std::cout << "\nEncoding completed" << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}

#endif
