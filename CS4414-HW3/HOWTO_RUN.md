# 🚀 How to Run the RAG System

## Quick Start (Interactive Mode)

### Option 1: Run with default settings
```bash
./main
```

### Option 2: Run with custom top-k
```bash
./main --top-k 5
```

### Option 3: Use the convenience script
```bash
./run_rag_interactive.sh
```

---

## Full Command-Line Options

```bash
./main \
  --bge-model bge-base-en-v1.5-f32.gguf \
  --llm-model qwen2-1_5b-instruct-q4_0.gguf \
  --db-path preprocessed_documents.json \
  --top-k 3
```

### Parameters:
- `--bge-model` - BGE embedding model (default: `bge-base-en-v1.5-f32.gguf`)
- `--llm-model` - LLM for generation (default: `qwen2-1_5b-instruct-q4_0.gguf`)
- `--db-path` - Preprocessed documents (default: `preprocessed_documents.json`)
- `--top-k` - Number of documents to retrieve (default: 3)

---

## Interactive Usage

Once running, you'll see:
```
RAG System Ready!
Enter your query (or 'quit' to exit): _
```

### Example Session:
```
Enter your query: What is machine learning?

[System retrieves documents, augments prompt, generates response]

Generated Response:
Machine learning is a subset of artificial intelligence...

Enter your query: quit
Goodbye!
```

---

## Testing Individual Components

### Component 1: Query Encoding
```bash
./encode --query "What is AI?" \
         --model bge-base-en-v1.5-f32.gguf \
         --output query_embedding.json
```

### Components 2+3: Vector Search + Document Retrieval
```bash
./vector_db --input preprocessed_documents.json \
            --query-embedding query_embedding.json \
            --top-k 3
```

---

## Expected Output

When running `./main`, you'll see:

1. **Loading messages** (from llama.cpp - this is normal):
   ```
   llama_model_load: loaded model from ...
   llama_new_context_with_model: ... Metal backend enabled
   ```

2. **Ready prompt**:
   ```
   RAG System Ready!
   Enter your query (or 'quit' to exit):
   ```

3. **Per-query output**:
   ```
   llama_perf_context_print: load time = X ms
   llama_perf_context_print: sample time = Y ms
   ```

4. **Generated response**:
   ```
   [Your AI-generated answer based on retrieved documents]
   ```

---

## Troubleshooting

### Error: "Failed to load model"
- Check model paths exist
- Verify `qwen2-1_5b-instruct-q4_0.gguf` is not 0 bytes (should be ~865MB)

### Error: "Failed to load database"
- Verify `preprocessed_documents.json` exists (should be 148MB)

### No GPU acceleration
- Metal logs should show "Metal backend" - if not, check llama.cpp was built with Metal support

---

## Quick Commands Reference

```bash
# Build everything
make clean && make all

# Run interactive system
./main

# Run with more documents retrieved
./main --top-k 5

# Test single query encoding
./encode --query "test query" --model bge-base-en-v1.5-f32.gguf

# Test document search
./vector_db --input preprocessed_documents.json \
            --query-embedding query_embedding.json
```

---

**Ready to go!** Just run `./main` to start! 🎉
