# CS4414 Homework 3 - RAG System

## Complete Interactive RAG System ✨

This project implements a complete **Retrieval-Augmented Generation (RAG)** pipeline with an interactive command-line interface.

## Quick Start

### 1. Build Everything
```bash
make all
```

### 2. Preprocess Documents (One-time)
```bash
export DYLD_LIBRARY_PATH=/Users/mmm/Downloads/CS4414/llama.cpp/build/bin:$DYLD_LIBRARY_PATH
./data_preprocess --input documents.json --model bge-base-en-v1.5-f32.gguf --output preprocessed_documents.json
```

### 3. Run the Interactive RAG System
```bash
./run_rag_interactive.sh
```

Or manually:
```bash
export DYLD_LIBRARY_PATH=/Users/mmm/Downloads/CS4414/llama.cpp/build/bin:$DYLD_LIBRARY_PATH
./main
```

Then start asking questions!

## System Components

### Complete RAG Pipeline (All 5 Components)

| Component | Implementation | File | Status |
|-----------|---------------|------|--------|
| **1. Query Encoder** | BGE embedding model | `encode.cpp` | ✅ Complete |
| **2. Vector Search** | FAISS IndexFlatL2 | `vector_db.cpp` | ✅ Complete |
| **3. Document Retrieval** | std::map lookup | `vector_db.cpp` | ✅ Complete |
| **4. Prompt Augmentation** | Context builder | `vector_db.cpp` | ✅ Complete |
| **5. LLM Generation** | llama.cpp inference | `main.cpp` | ✅ Complete |
| **6. Interactive Interface** | Command-line REPL | `main.cpp` | ✅ Complete |

## Programs

### `main` - Interactive RAG System (⭐ Main Program)
The complete RAG system with interactive interface.

**Usage:**
```bash
./main [--documents <path>] [--embedding-model <path>] [--llm-model <path>] [--top-k <n>]
```

**Features:**
- Ask questions and get AI-powered answers
- Real-time document retrieval visualization
- Continuous interactive loop
- All 5 RAG components integrated

See `MAIN_USAGE.md` for detailed documentation.

### `data_preprocess` - Document Preprocessing
Processes documents and generates embeddings.

**Usage:**
```bash
./data_preprocess --input documents.json --model bge-base-en-v1.5-f32.gguf --output preprocessed_documents.json
```

### `encode` - Query Encoder
Encodes queries into 768-dimensional vectors.

**Usage:**
```bash
./encode --query "What is AI?" --model bge-base-en-v1.5-f32.gguf --output query_embedding.json
```

### `vector_db` - Vector Search & Retrieval
Searches documents and builds augmented prompts.

**Usage:**
```bash
./vector_db --input preprocessed_documents.json --query-embedding query_embedding.json --top-k 3 --augmented-prompt
```

## Build Instructions

### Build All Programs
```bash
make all
```

### Build Individual Programs
```bash
make data_preprocess  # Document preprocessing
make encode           # Query encoder
make vector_db        # Vector search
make main             # Interactive RAG system
```

### Clean Build Files
```bash
make clean
```

## Documentation

- **`MAIN_USAGE.md`** - Complete guide for the interactive RAG system
- **`COMPONENT6_SUMMARY.md`** - Implementation details of Component 6
- **`PART2_GUIDE.md`** - Guide for Components 1-2
- **`COMPONENT4_GUIDE.md`** - Guide for Component 4
- **`RAG_PIPELINE_STATUS.md`** - Overall pipeline status

## Example Session

```
$ ./main

======================================================================
        RAG SYSTEM INITIALIZATION
======================================================================

[1/4] Loading document database...
✓ Loaded 10000 documents
✓ Index built with 10000 vectors

[2/4] Loading embedding model...
✓ Embedding model loaded (BGE)

[3/4] Loading LLM model...
✓ LLM model loaded

[4/4] System ready!

======================================================================
        INTERACTIVE RAG SYSTEM
======================================================================

> What is machine learning?

----------------------------------------------------------------------
Query: What is machine learning?
----------------------------------------------------------------------

[1/4] Encoding query... ✓
[2/4] Retrieving relevant documents... ✓

Top 3 Retrieved Documents:
  [1] ID: 1234 (distance: 0.456)
      Machine learning is a subset of artificial intelligence...

[3/4] Building augmented prompt... ✓
[4/4] Generating response... ✓

======================================================================
ANSWER:
======================================================================
Machine learning is a branch of artificial intelligence that focuses
on creating systems that can learn and improve from experience...
======================================================================

> exit

Thank you for using the RAG system! Goodbye.
```

## Architecture

```
User Query
    ↓
[Component 1: Query Encoder]
    ↓ (768-dim embedding)
[Component 2: Vector Search]
    ↓ (Top-K document indices)
[Component 3: Document Retrieval]
    ↓ (Full document texts)
[Component 4: Prompt Augmentation]
    ↓ (Query + context)
[Component 5: LLM Generation]
    ↓
AI-Generated Answer
```

## Prerequisites

- **Documents**: `documents.json`
- **Models**:
  - BGE embedding model: `bge-base-en-v1.5-f32.gguf`
  - LLM model: `qwen2-1_5b-instruct-q4_0.gguf` or `tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf`
- **Libraries**:
  - FAISS (in `../faiss`)
  - llama.cpp (in `../llama.cpp`)
  - OpenMP
  - Accelerate framework (macOS)

## Performance

- **Initialization**: 5-10 seconds (one-time)
- **Query Processing**: 3-6 seconds per query
  - Encoding: ~0.5s
  - Search: ~50ms
  - Retrieval: <1ms
  - Augmentation: <1ms
  - Generation: 2-5s

## Memory Requirements

- Vector Database: ~150 MB
- BGE Model: ~500 MB
- LLM Model: ~1-2 GB
- **Total**: ~2-3 GB RAM

## Troubleshooting

### Library Not Found Error
```bash
export DYLD_LIBRARY_PATH=/Users/mmm/Downloads/CS4414/llama.cpp/build/bin:$DYLD_LIBRARY_PATH
```

### Models Not Found
Ensure all model files are in the current directory or specify full paths.

### Out of Memory
Use a smaller LLM model or reduce top-k value.

## Files Overview

| File | Purpose | Lines | Component |
|------|---------|-------|-----------|
| `main.cpp` | Interactive RAG system | 800+ | All 1-6 |
| `data_preprocess.cpp` | Document preprocessing | 468 | Part 1 |
| `encode.cpp` | Query encoder | 283 | Component 1 |
| `vector_db.cpp` | Vector search & retrieval | 651 | Components 2-4 |
| `Makefile` | Build configuration | 64 | Build |
| `run_rag_interactive.sh` | Launcher script | 60 | Convenience |

## Implementation Highlights

- ✅ **All 5 RAG components** integrated
- ✅ **Interactive interface** for continuous querying  
- ✅ **Efficient model reuse** (load once, query many times)
- ✅ **Real-time feedback** with progress indicators
- ✅ **Comprehensive error handling**
- ✅ **Production-ready code** with proper cleanup
- ✅ **Extensive documentation**

## Credits

CS4414 - Systems Programming  
Homework 3 - RAG System Implementation
