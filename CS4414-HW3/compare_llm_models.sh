#!/bin/bash

# LLM Model Comparison Script for Part 3
# Tests different LLM models and compares speed vs quality tradeoffs

set -e  # Exit on error

echo "========================================================================"
echo "              LLM MODEL COMPARISON EXPERIMENT"
echo "========================================================================"
echo ""

# Configuration
NUM_QUERIES=30  # Use fewer queries since LLM generation is slow
RESULTS_DIR="llm_results"

# Models to test (add/remove as needed)
declare -A MODELS
MODELS["tinyllama"]="tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
MODELS["qwen2"]="qwen2-1_5b-instruct-q4_0.gguf"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check if benchmark exists
if [ ! -f "./benchmark" ]; then
    echo "Error: benchmark executable not found!"
    echo "Please run 'make benchmark' first."
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

echo "Testing LLM models:"
for name in "${!MODELS[@]}"; do
    model_file="${MODELS[$name]}"
    if [ -f "$model_file" ]; then
        echo "  ✓ $name: $model_file"
    else
        echo "  ✗ $name: $model_file (NOT FOUND - will skip)"
    fi
done
echo ""
echo "Number of queries per model: $NUM_QUERIES"
echo ""

# Run benchmark for each model
for name in "${!MODELS[@]}"; do
    model_file="${MODELS[$name]}"
    
    # Skip if model file doesn't exist
    if [ ! -f "$model_file" ]; then
        echo "Skipping $name (model file not found)"
        continue
    fi
    
    echo "------------------------------------------------------------------------"
    echo "Testing Model: $name ($model_file)"
    echo "------------------------------------------------------------------------"
    
    OUTPUT_FILE="$RESULTS_DIR/benchmark_${name}.csv"
    LOG_FILE="$RESULTS_DIR/benchmark_${name}.log"
    
    ./benchmark \
        --num-queries $NUM_QUERIES \
        --llm-model "$model_file" \
        --output "$OUTPUT_FILE" \
        2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "✓ Completed $name"
    echo "  Results saved to: $OUTPUT_FILE"
    echo ""
done

echo "========================================================================"
echo "              GENERATING COMPARISON ANALYSIS"
echo "========================================================================"
echo ""

# Create Python comparison script
cat > "$RESULTS_DIR/compare_models.py" << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import glob

# Find all result files
result_files = glob.glob("llm_results/benchmark_*.csv")

if not result_files:
    print("Error: No result files found in llm_results/")
    sys.exit(1)

print("\n" + "="*80)
print("LLM MODEL COMPARISON ANALYSIS")
print("="*80 + "\n")

results = []
for csv_file in result_files:
    # Extract model name from filename
    model_name = os.path.basename(csv_file).replace("benchmark_", "").replace(".csv", "")
    
    df = pd.read_csv(csv_file)
    results.append({
        'model': model_name,
        'total_mean': df['total_ms'].mean(),
        'encoding_mean': df['encoding_ms'].mean(),
        'search_mean': df['search_ms'].mean(),
        'augmentation_mean': df['augmentation_ms'].mean(),
        'generation_mean': df['generation_ms'].mean(),
        'generation_median': df['generation_ms'].median(),
        'num_queries': len(df)
    })

if not results:
    print("Error: Could not parse any result files!")
    sys.exit(1)

comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('total_mean')

# Print comparison table
print("Performance Comparison:")
print("-"*80)
print(f"{'Model':<15} {'Total':<12} {'Generation':<12} {'Speedup':<10} {'Queries':<8}")
print("-"*80)

baseline_total = comparison_df.iloc[0]['total_mean']
for _, row in comparison_df.iterrows():
    speedup = baseline_total / row['total_mean']
    print(f"{row['model']:<15} {row['total_mean']:>10.1f}ms "
          f"{row['generation_mean']:>10.1f}ms {speedup:>8.2f}x {int(row['num_queries']):>6}")

print("\n" + "="*80)
print("OBSERVATIONS:")
print("="*80)

fastest = comparison_df.iloc[0]
slowest = comparison_df.iloc[-1]

print(f"\n1. Fastest Model: {fastest['model']}")
print(f"   - Average total time: {fastest['total_mean']:.1f}ms")
print(f"   - Average generation time: {fastest['generation_mean']:.1f}ms")

print(f"\n2. Slowest Model: {slowest['model']}")
print(f"   - Average total time: {slowest['total_mean']:.1f}ms")
print(f"   - Average generation time: {slowest['generation_mean']:.1f}ms")

speedup_ratio = slowest['total_mean'] / fastest['total_mean']
print(f"\n3. Speed Difference: {speedup_ratio:.2f}x")

# Generation time percentage
for _, row in comparison_df.iterrows():
    gen_pct = (row['generation_mean'] / row['total_mean']) * 100
    print(f"\n4. {row['model']}: LLM generation is {gen_pct:.1f}% of total time")

print("\n" + "="*80)
print("TRADEOFF ANALYSIS:")
print("="*80)
print("\nSmaller models (e.g., TinyLlama):")
print("  ✓ Faster generation")
print("  ✓ Lower memory usage")
print("  ✗ May produce less coherent answers")
print("  ✗ May miss nuances in complex queries")

print("\nLarger models (e.g., Qwen2-1.5B):")
print("  ✓ Better answer quality")
print("  ✓ More coherent and detailed responses")
print("  ✗ Slower generation")
print("  ✗ Higher memory usage")

print("\n" + "="*80)

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Total latency comparison
ax = axes[0, 0]
models = comparison_df['model']
totals = comparison_df['total_mean']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(models)]

bars = ax.bar(models, totals, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('Total Latency (ms)', fontsize=11, fontweight='bold')
ax.set_title('Total End-to-End Latency by Model', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

for bar, total in zip(bars, totals):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 50,
            f'{total:.0f}ms', ha='center', va='bottom', fontweight='bold')

# Plot 2: Component breakdown
ax = axes[0, 1]
components = ['encoding_mean', 'search_mean', 'augmentation_mean', 'generation_mean']
labels = ['Encoding', 'Search', 'Augmentation', 'Generation']
colors_comp = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

x = range(len(comparison_df))
width = 0.6
bottom = [0] * len(comparison_df)

for comp, label, color in zip(components, labels, colors_comp):
    values = comparison_df[comp]
    ax.bar(x, values, width, bottom=bottom, label=label, color=color, alpha=0.8, edgecolor='black')
    bottom = [b + v for b, v in zip(bottom, values)]

ax.set_ylabel('Cumulative Latency (ms)', fontsize=11, fontweight='bold')
ax.set_title('Component Breakdown by Model', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['model'], rotation=15)
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Plot 3: Generation time comparison
ax = axes[1, 0]
gen_times = comparison_df['generation_mean']
bars = ax.bar(models, gen_times, color='#e74c3c', alpha=0.8, edgecolor='black')
ax.set_ylabel('LLM Generation Time (ms)', fontsize=11, fontweight='bold')
ax.set_title('LLM Generation Latency by Model', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

for bar, gen_time in zip(bars, gen_times):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 50,
            f'{gen_time:.0f}ms', ha='center', va='bottom', fontweight='bold')

# Plot 4: Generation time as percentage
ax = axes[1, 1]
gen_percentages = (comparison_df['generation_mean'] / comparison_df['total_mean'] * 100)
bars = ax.bar(models, gen_percentages, color='#9b59b6', alpha=0.8, edgecolor='black')
ax.set_ylabel('Percentage of Total Time (%)', fontsize=11, fontweight='bold')
ax.set_title('LLM Generation as % of Total Time', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0, 100])

for bar, pct in zip(bars, gen_percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('llm_results/model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Comparison plot saved: llm_results/model_comparison.png\n")
plt.close()
EOF

chmod +x "$RESULTS_DIR/compare_models.py"
python3 "$RESULTS_DIR/compare_models.py"

echo "========================================================================"
echo "              EXPERIMENT COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved in: $RESULTS_DIR/"
echo ""
echo "Generated files:"
echo "  - benchmark_*.csv (raw data for each model)"
echo "  - benchmark_*.log (detailed logs)"
echo "  - model_comparison.png (comparison plot)"
echo ""
echo "NOTE: To evaluate answer QUALITY, manually review a few responses"
echo "from each model and compare coherence, accuracy, and completeness."
echo ""






