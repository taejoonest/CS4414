#!/bin/bash

# Batch Processing Experiment Script for Part 3
# Tests different batch sizes and analyzes throughput/latency tradeoffs

set -e  # Exit on error

echo "========================================================================"
echo "              BATCH PROCESSING EXPERIMENT"
echo "========================================================================"
echo ""

# Configuration
NUM_QUERIES=256  # Need enough queries for batch size 128
OUTPUT_FILE="batch_results.csv"

# Check if batch_benchmark exists
if [ ! -f "./batch_benchmark" ]; then
    echo "Error: batch_benchmark executable not found!"
    echo "Please run 'make batch_benchmark' first."
    exit 1
fi

# Check if required files exist
if [ ! -f "preprocessed_documents.json" ]; then
    echo "Error: preprocessed_documents.json not found!"
    exit 1
fi

if [ ! -f "queries.json" ]; then
    echo "Error: queries.json not found!"
    exit 1
fi

echo "Running batch processing experiment..."
echo "Number of queries: $NUM_QUERIES"
echo "Batch sizes: 1, 4, 8, 16, 32, 64, 128"
echo "Output: $OUTPUT_FILE"
echo ""

# Run batch benchmark
./batch_benchmark \
    --num-queries $NUM_QUERIES \
    --output "$OUTPUT_FILE" \
    2>&1 | tee batch_experiment.log

echo ""
echo "========================================================================"
echo "              GENERATING PLOTS"
echo "========================================================================"
echo ""

# Generate plots
if [ -f "$OUTPUT_FILE" ]; then
    python3 plot_batch_results.py "$OUTPUT_FILE"
else
    echo "Error: Results file not found: $OUTPUT_FILE"
    exit 1
fi

echo ""
echo "========================================================================"
echo "              EXPERIMENT COMPLETE!"
echo "========================================================================"
echo ""
echo "Results:"
echo "  - $OUTPUT_FILE (raw data)"
echo "  - batch_experiment.log (detailed log)"
echo "  - batch_*.png (plots)"
echo ""






