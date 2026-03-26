#!/bin/bash

# Top-K Experiment Script for Part 3
# Tests different top-K values and analyzes performance/accuracy tradeoffs

set -e  # Exit on error

echo "========================================================================"
echo "              TOP-K RETRIEVAL EXPERIMENT"
echo "========================================================================"
echo ""

# Configuration
NUM_QUERIES=50
TOP_K_VALUES=(1 2 3 5 10)
RESULTS_DIR="topk_results"

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

echo "Running experiments with different top-K values..."
echo "Number of queries: $NUM_QUERIES"
echo "Top-K values: ${TOP_K_VALUES[@]}"
echo ""

# Run benchmark for each top-K value
for k in "${TOP_K_VALUES[@]}"; do
    echo "------------------------------------------------------------------------"
    echo "Running benchmark with top-K = $k"
    echo "------------------------------------------------------------------------"
    
    OUTPUT_FILE="$RESULTS_DIR/benchmark_topk${k}.csv"
    
    ./benchmark \
        --num-queries $NUM_QUERIES \
        --top-k $k \
        --output "$OUTPUT_FILE" \
        2>&1 | tee "$RESULTS_DIR/benchmark_topk${k}.log"
    
    echo ""
    echo "✓ Completed top-K = $k"
    echo "  Results saved to: $OUTPUT_FILE"
    echo ""
done

echo "========================================================================"
echo "              GENERATING COMPARISON PLOTS"
echo "========================================================================"
echo ""

# Generate plots for each configuration
for k in "${TOP_K_VALUES[@]}"; do
    echo "Generating plots for top-K = $k..."
    python3 plot_benchmark.py "$RESULTS_DIR/benchmark_topk${k}.csv" "$RESULTS_DIR/topk${k}_"
done

echo ""
echo "========================================================================"
echo "              GENERATING COMPARISON ANALYSIS"
echo "========================================================================"
echo ""

# Create comparison script if Python is available
cat > "$RESULTS_DIR/compare_topk.py" << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

top_k_values = [1, 2, 3, 5, 10]
results = []

print("\n" + "="*80)
print("TOP-K COMPARISON ANALYSIS")
print("="*80 + "\n")

for k in top_k_values:
    csv_file = f"topk_results/benchmark_topk{k}.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        results.append({
            'top_k': k,
            'encoding_mean': df['encoding_ms'].mean(),
            'search_mean': df['search_ms'].mean(),
            'augmentation_mean': df['augmentation_ms'].mean(),
            'generation_mean': df['generation_ms'].mean(),
            'total_mean': df['total_ms'].mean(),
            'total_median': df['total_ms'].median(),
            'total_std': df['total_ms'].std()
        })

if not results:
    print("Error: No result files found!")
    sys.exit(1)

comparison_df = pd.DataFrame(results)

# Print comparison table
print(f"{'Top-K':<8} {'Encoding':<12} {'Search':<12} {'Augment':<12} {'Generate':<12} {'Total':<12}")
print("-"*80)
for _, row in comparison_df.iterrows():
    print(f"{int(row['top_k']):<8} {row['encoding_mean']:>10.2f}ms "
          f"{row['search_mean']:>10.2f}ms {row['augmentation_mean']:>10.2f}ms "
          f"{row['generation_mean']:>10.2f}ms {row['total_mean']:>10.2f}ms")

print("\n" + "="*80)
print("OBSERVATIONS:")
print("="*80)
print(f"\n1. Total Latency Range: {comparison_df['total_mean'].min():.1f}ms - {comparison_df['total_mean'].max():.1f}ms")

search_increase = ((comparison_df[comparison_df['top_k']==10]['search_mean'].values[0] / 
                    comparison_df[comparison_df['top_k']==1]['search_mean'].values[0]) - 1) * 100
print(f"2. Search time increase (K=1 -> K=10): {search_increase:+.1f}%")

aug_increase = ((comparison_df[comparison_df['top_k']==10]['augmentation_mean'].values[0] / 
                 comparison_df[comparison_df['top_k']==1]['augmentation_mean'].values[0]) - 1) * 100
print(f"3. Augmentation time increase (K=1 -> K=10): {aug_increase:+.1f}%")

gen_increase = ((comparison_df[comparison_df['top_k']==10]['generation_mean'].values[0] / 
                 comparison_df[comparison_df['top_k']==1]['generation_mean'].values[0]) - 1) * 100
print(f"4. Generation time increase (K=1 -> K=10): {gen_increase:+.1f}%")

total_increase = ((comparison_df[comparison_df['top_k']==10]['total_mean'].values[0] / 
                   comparison_df[comparison_df['top_k']==1]['total_mean'].values[0]) - 1) * 100
print(f"5. Total time increase (K=1 -> K=10): {total_increase:+.1f}%")

print("\n" + "="*80)

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Component comparison
ax = axes[0, 0]
components = ['encoding_mean', 'search_mean', 'augmentation_mean', 'generation_mean']
labels = ['Encoding', 'Search', 'Augmentation', 'Generation']
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

x = comparison_df['top_k']
for comp, label, color in zip(components, labels, colors):
    ax.plot(x, comparison_df[comp], marker='o', linewidth=2, markersize=8, 
            label=label, color=color)

ax.set_xlabel('Top-K', fontsize=11, fontweight='bold')
ax.set_ylabel('Latency (ms)', fontsize=11, fontweight='bold')
ax.set_title('Component Latency vs Top-K', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, linestyle='--')

# Plot 2: Total latency
ax = axes[0, 1]
ax.plot(comparison_df['top_k'], comparison_df['total_mean'], 
        marker='o', linewidth=2, markersize=10, color='purple')
ax.fill_between(comparison_df['top_k'], 
                comparison_df['total_mean'] - comparison_df['total_std'],
                comparison_df['total_mean'] + comparison_df['total_std'],
                alpha=0.3, color='purple')
ax.set_xlabel('Top-K', fontsize=11, fontweight='bold')
ax.set_ylabel('Total Latency (ms)', fontsize=11, fontweight='bold')
ax.set_title('Total End-to-End Latency vs Top-K', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, linestyle='--')

# Plot 3: Stacked bar chart
ax = axes[1, 0]
bottom = [0] * len(comparison_df)
for comp, label, color in zip(components, labels, colors):
    ax.bar(comparison_df['top_k'], comparison_df[comp], bottom=bottom, 
           label=label, color=color, alpha=0.8, edgecolor='black')
    bottom = [b + v for b, v in zip(bottom, comparison_df[comp])]

ax.set_xlabel('Top-K', fontsize=11, fontweight='bold')
ax.set_ylabel('Cumulative Latency (ms)', fontsize=11, fontweight='bold')
ax.set_title('Stacked Component Latency', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Plot 4: Percentage impact
ax = axes[1, 1]
totals = comparison_df['total_mean']
for comp, label, color in zip(components, labels, colors):
    percentages = (comparison_df[comp] / totals * 100)
    ax.plot(comparison_df['top_k'], percentages, marker='o', linewidth=2, 
            markersize=8, label=label, color=color)

ax.set_xlabel('Top-K', fontsize=11, fontweight='bold')
ax.set_ylabel('Percentage of Total Time (%)', fontsize=11, fontweight='bold')
ax.set_title('Component Time Distribution vs Top-K', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('topk_results/topk_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Comparison plot saved: topk_results/topk_comparison.png\n")
plt.close()
EOF

chmod +x "$RESULTS_DIR/compare_topk.py"
python3 "$RESULTS_DIR/compare_topk.py"

echo "========================================================================"
echo "              EXPERIMENT COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved in: $RESULTS_DIR/"
echo ""
echo "Generated files:"
echo "  - benchmark_topk*.csv (raw data)"
echo "  - benchmark_topk*.log (benchmark logs)"
echo "  - topk*_*.png (individual plots)"
echo "  - topk_comparison.png (comparison plot)"
echo ""
echo "To view results:"
echo "  ls -lh $RESULTS_DIR/"
echo ""






