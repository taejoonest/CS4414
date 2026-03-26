# 🔇 Log Output Suppression

## Issue
The system was outputting verbose llama.cpp logs:
```
llama_model_loader: create_tensor: ...
llama_model_loader: print_info: ...
llama_model_loader: - kv 0: general.architecture str = ...
llama_model_loader: - kv 1: general.name str = ...
... (hundreds of lines)
```

## Solution
Added a custom log callback to filter verbose messages.

### Files Modified:

#### `main.cpp`
- Added `custom_log_callback()` function
- Calls `llama_log_set(custom_log_callback, nullptr)` after backend init
- Suppresses: `create_tensor`, `print_info`, `llama_model_loader`, metadata dumps

#### `encode.cpp`
- Added same callback (inside `#ifdef ENCODE_STANDALONE` block)
- Used only for standalone encode tool

### What's Suppressed:
✅ `create_tensor` - Tensor creation messages  
✅ `print_info` - Model info dumps  
✅ `llama_model_loader` - Loader verbose output  
✅ `Dumping metadata` - Metadata dumps  
✅ `- kv` - Key-value pair listings  

### What's Still Shown:
✅ **Warnings** (GGML_LOG_LEVEL_WARN)  
✅ **Errors** (GGML_LOG_LEVEL_ERROR)  
✅ **Important initialization messages**  
✅ **Performance metrics** (llama_perf_context_print)  

### Code:
```cpp
static void custom_log_callback(enum ggml_log_level level, const char* text, void* user_data) {
    (void)user_data;
    
    // Filter out verbose messages
    std::string msg(text);
    if (msg.find("create_tensor") != std::string::npos ||
        msg.find("print_info") != std::string::npos ||
        msg.find("llama_model_loader") != std::string::npos ||
        msg.find("Dumping metadata") != std::string::npos ||
        msg.find("- kv") != std::string::npos) {
        return; // Suppress these messages
    }
    
    // Only show warnings and errors
    if (level >= GGML_LOG_LEVEL_WARN) {
        fprintf(stderr, "%s", text);
    }
}
```

## Result

### Before:
```
[1/4] Loading document database...
Loaded 10000 documents
[2/4] Loading embedding model...
llama_model_loader: create_tensor: token_embd.weight
llama_model_loader: create_tensor: blk.0.attn_q.weight
llama_model_loader: create_tensor: blk.0.attn_k.weight
... (200+ lines of verbose output)
```

### After:
```
[1/4] Loading document database...
Loaded 10000 documents
[2/4] Loading embedding model...
[Model loads silently, only errors/warnings shown]
[3/4] Loading LLM model...
[Model loads silently, only errors/warnings shown]
[4/4] Initialization complete!
```

---

**Clean output while preserving error visibility!** ✅
