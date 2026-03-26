#pragma once

#include "llama.h"
#include <string>
#include <vector>

void normalize_embedding(std::vector<float>& vec);

std::vector<float> encode_query(llama_context* ctx,
                                 const llama_model* model,
                                 const std::string& text);
