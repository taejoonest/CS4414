#!/bin/bash

# Interactive RAG System Launcher
# This script sets up the environment and launches the RAG system

# Set library path for llama.cpp
export DYLD_LIBRARY_PATH=/Users/mmm/Downloads/CS4414/llama.cpp/build/bin:$DYLD_LIBRARY_PATH

# Color codes for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Interactive RAG System Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if required files exist
echo -e "${YELLOW}Checking prerequisites...${NC}"

if [ ! -f "preprocessed_documents.json" ]; then
    echo -e "${YELLOW}Warning: preprocessed_documents.json not found!${NC}"
    echo "Please run data preprocessing first:"
    echo "  ./data_preprocess --input documents.json --model bge-base-en-v1.5-f32.gguf --output preprocessed_documents.json"
    exit 1
fi

if [ ! -f "bge-base-en-v1.5-f32.gguf" ]; then
    echo -e "${YELLOW}Warning: bge-base-en-v1.5-f32.gguf not found!${NC}"
    echo "Please place the BGE model in the current directory."
    exit 1
fi

if [ ! -f "qwen2-1_5b-instruct-q4_0.gguf" ] && [ ! -f "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf" ]; then
    echo -e "${YELLOW}Warning: No LLM model found!${NC}"
    echo "Please place either qwen2-1_5b-instruct-q4_0.gguf or tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf in the current directory."
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites found${NC}"
echo ""

# Determine which LLM model to use
if [ -f "qwen2-1_5b-instruct-q4_0.gguf" ]; then
    LLM_MODEL="qwen2-1_5b-instruct-q4_0.gguf"
    echo -e "${GREEN}Using LLM: Qwen2-1.5B (recommended)${NC}"
elif [ -f "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf" ]; then
    LLM_MODEL="tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
    echo -e "${GREEN}Using LLM: TinyLlama-1.1B${NC}"
fi

echo ""
echo -e "${BLUE}Starting RAG system...${NC}"
echo ""

# Launch the interactive RAG system
./main --documents preprocessed_documents.json \
       --embedding-model bge-base-en-v1.5-f32.gguf \
       --llm-model "$LLM_MODEL" \
       --top-k 3

echo ""
echo -e "${GREEN}Session ended. Goodbye!${NC}"




