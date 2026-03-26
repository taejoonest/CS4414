#!/bin/bash

# Test script to verify Component 3: Document Retrieval
# Tests multiple queries from queries.json to check retrieval quality

export DYLD_LIBRARY_PATH=/Users/mmm/Downloads/CS4414/llama.cpp/build/bin:$DYLD_LIBRARY_PATH

echo "=============================================================="
echo "Testing Component 3: Document Retrieval with std::map"
echo "=============================================================="
echo ""

# Test 1: Computer-related query
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: 'what is a hardware in a computer'"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./encode --query "what is a hardware in a computer" \
         --model bge-base-en-v1.5-f32.gguf \
         --output query_embedding.json 2>&1 | grep -A 5 "Embedding Statistics"
echo ""
./vector_db --input preprocessed_documents.json \
            --query-embedding query_embedding.json \
            --top-k 3 | grep -A 100 "Top 3"
echo ""
echo ""

# Test 2: Math-related query  
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: 'what is a apothem' (geometry term)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./encode --query "what is a apothem" \
         --model bge-base-en-v1.5-f32.gguf \
         --output query_embedding.json 2>&1 | grep -A 5 "Embedding Statistics"
echo ""
./vector_db --input preprocessed_documents.json \
            --query-embedding query_embedding.json \
            --top-k 3 | grep -A 100 "Top 3"
echo ""
echo ""

# Test 3: Biology-related query
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: 'what does proteus mirabilis look like' (biology)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./encode --query "what does proteus mirabilis look like" \
         --model bge-base-en-v1.5-f32.gguf \
         --output query_embedding.json 2>&1 | grep -A 5 "Embedding Statistics"
echo ""
./vector_db --input preprocessed_documents.json \
            --query-embedding query_embedding.json \
            --top-k 3 | grep -A 100 "Top 3"
echo ""
echo ""

echo "=============================================================="
echo "Retrieval Tests Complete!"
echo "=============================================================="
echo ""
echo "✓ Verify that retrieved documents are relevant to each query"
echo "✓ Check that std::map lookup is working correctly"
echo "✓ Confirm document IDs and texts are properly retrieved"





