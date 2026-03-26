#include "llm_generation.h"
#include <stdexcept>
#include <vector>

std::string generate_response(llama_context* ctx,
                              const llama_model* model,
                              const std::string& prompt,
                              int max_tokens) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        throw std::runtime_error("Failed to get vocab from model");
    }

    const int n_max_tokens = 4096;  // Increased for larger Top-K prompts
    std::vector<llama_token> tokens(n_max_tokens);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                   tokens.data(), n_max_tokens, true, false);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                   tokens.data(), -n_tokens, true, false);
    }
    if (n_tokens < 0) {
        throw std::runtime_error("Failed to tokenize prompt");
    }
    tokens.resize(n_tokens);

    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);

    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.seq_id[i][0] = 0;
        batch.n_seq_id[i] = 1;
        batch.logits[i] = (i == n_tokens - 1) ? 1 : -1;
    }
    batch.n_tokens = n_tokens;

    if (llama_decode(ctx, batch) < 0) {
        llama_batch_free(batch);
        throw std::runtime_error("llama_decode failed");
    }

    std::string response;
    int n_cur = n_tokens;
    std::vector<llama_token> generated_tokens;

    for (int i = 0; i < max_tokens; ++i) {

        auto* logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            break;
        }

        int n_vocab = llama_vocab_n_tokens(vocab);
        llama_token new_token = 0;
        float max_logit = logits[0];
        for (int j = 1; j < n_vocab; ++j) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                new_token = j;
            }
        }

        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        generated_tokens.push_back(new_token);

        if (generated_tokens.size() >= 20) {
            bool is_repeating = true;
            for (size_t k = 0; k < 10 && is_repeating; ++k) {
                if (generated_tokens[generated_tokens.size() - 10 + k] !=
                    generated_tokens[generated_tokens.size() - 20 + k]) {
                    is_repeating = false;
                }
            }
            if (is_repeating) {

                break;
            }
        }

        char buf[256];
        int n_chars = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
        if (n_chars > 0) {
            response.append(buf, n_chars);
        }

        if (response.length() > 100 && i >= 50) {
            size_t len = response.length();
            if (len > 0 && (response[len-1] == '.' || response[len-1] == '!' || response[len-1] == '?')) {

                if (len > 2 && response[len-2] == ' ') {

                    break;
                }
            }
        }

        batch.n_tokens = 1;
        batch.token[0] = new_token;
        batch.pos[0] = n_cur;
        batch.seq_id[0][0] = 0;
        batch.n_seq_id[0] = 1;
        batch.logits[0] = 1;

        if (llama_decode(ctx, batch) < 0) {
            break;
        }

        n_cur++;
    }

    llama_batch_free(batch);
    return response;
}
