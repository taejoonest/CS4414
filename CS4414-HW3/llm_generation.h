#pragma once

#include "llama.h"
#include <string>

std::string generate_response(llama_context* ctx,
                              const llama_model* model,
                              const std::string& prompt,
                              int max_tokens = 256);
