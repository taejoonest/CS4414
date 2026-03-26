# 🚀 Quick Start Guide

## ✅ FIXED: Dynamic Library Loading

The Makefile now includes the correct `rpath` flag, so the executable can find `libllama.dylib` at runtime.

## How to Run

### Simple Command:
```bash
./main
```

Wait for the initialization (30-60 seconds):
```
======================================================================
        RAG SYSTEM INITIALIZATION
======================================================================

[1/4] Loading document database...
Loaded 10000 documents
Embedding dimension: 768 ✓

[2/4] Loading embedding model...
[BGE model loading messages from llama.cpp]

[3/4] Loading LLM model...
[Qwen2 model loading messages from llama.cpp]

[4/4] Initialization complete!

RAG System Ready!
Enter your query (or 'quit' to exit): _
```

### Then Ask Questions:
```
Enter your query: What is machine learning?
[System generates response]

Enter your query: How does a neural network work?
[System generates response]

Enter your query: quit
Goodbye!
```

## Optional: Custom Settings

```bash
# Retrieve more documents per query
./main --top-k 5

# Use different models (if you have them)
./main --embedding-model bge-base-en-v1.5-f32.gguf \
       --llm-model qwen2-1_5b-instruct-q4_0.gguf \
       --documents preprocessed_documents.json
```

## What You'll See

### Normal Output (keep this for submission):
- `llama_model_load`: Model loading info
- `Metal backend`: GPU acceleration enabled ✓
- `llama_perf_context_print`: Performance metrics per query

These messages are from `llama.cpp` and demonstrate that your system is using GPU acceleration!

## Troubleshooting

### If you see: `dyld: Library not loaded`
- **Fixed!** Rebuild with: `make clean && make all`

### If loading takes too long:
- First load can take 30-60 seconds (models are large)
- Subsequent queries will be faster

### If out of memory:
- Close other applications
- Use a smaller LLM (though qwen2-1.5B is already small)

---

## Files Needed for Submission

✅ Source files:
- `main.cpp` - Components 4 & 6
- `encode.cpp` + `encode.h` - Component 1
- `vector_db.cpp` + `vector_db.h` - Components 2 & 3
- `llm_generation.cpp` + `llm_generation.h` - Component 5
- `Makefile` - Build system

✅ Data files:
- `preprocessed_documents.json` (148MB)
- `bge-base-en-v1.5-f32.gguf` (416MB)
- `qwen2-1_5b-instruct-q4_0.gguf` (894MB)

✅ Documentation:
- `README.md` - Overview
- `FINAL_STRUCTURE.md` - Component details

---

**Ready! Just run `./main` and start asking questions!** 🎉
